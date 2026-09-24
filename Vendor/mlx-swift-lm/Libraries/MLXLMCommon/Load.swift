// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

#if canImport(Darwin)
import Darwin

/// Tell the kernel to start prefetching every shard into the unified buffer
/// cache. Non-blocking; the actual reads happen in the SSD controller's own
/// queue. Safe to call even when files are already cached (it's just a hint).
private func prefetchShards(_ urls: [URL]) {
    DispatchQueue.concurrentPerform(iterations: urls.count) { idx in
        let path = urls[idx].path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { return }
        defer { close(fd) }

        var sb = stat()
        guard fstat(fd, &sb) == 0 else { return }

        var ra = radvisory(ra_offset: 0, ra_count: Int32(min(Int(sb.st_size), Int(Int32.max))))
        _ = fcntl(fd, F_RDADVISE, &ra)
    }
}
#else
private func prefetchShards(_ urls: [URL]) { }
#endif

/// Lock-protected scratch space used by the parallel shard reader. Wrapped in
/// a final class so Swift 6 strict concurrency can see we mean to share it
/// across the concurrent closures.
private final class ParallelShardState: @unchecked Sendable {
    typealias ShardResult = (weights: [String: MLXArray], metadata: [String: String])
    let lock = NSLock()
    var results: [ShardResult?]
    var firstError: Error?

    init(shardCount: Int) {
        self.results = Array(repeating: nil, count: shardCount)
    }

    func store(index: Int, result: ShardResult) {
        lock.lock()
        defer { lock.unlock() }
        results[index] = result
    }

    /// Called only after concurrentPerform joined and aggregation completed.
    /// Fused-output materialization must not retain all original shard arrays.
    func releaseResults() {
        lock.lock()
        defer { lock.unlock() }
        results.removeAll(keepingCapacity: false)
    }

    func recordError(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }
}

/// Implemented by models whose ``BaseLanguageModel/sanitize(weights:)``
/// renames or fuses modules relative to the checkpoint layout.
///
/// Per-layer quantization tables in `config.json` are keyed on the
/// checkpoint's module paths. When a sanitizer renames a module (e.g. fusing
/// split `gate_proj`/`up_proj` experts into one `gate_up_proj`), the new path
/// no longer matches its explicit overrides, and
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``
/// would fall back to the default quantization — or to none at all for
/// mixed-precision configs without a default — and the subsequent strict
/// update would reject the renamed `.scales`/`.biases` tensors. Aliases
/// restore the lookup: the first alias with an explicit per-layer entry wins.
public protocol QuantizationPathAliasing {
    /// Checkpoint-layout config paths to consult, in order, when `path`
    /// itself has no explicit per-layer quantization entry.
    func quantizationPathAliases(for path: String) -> [String]
}

/// Resolve the quantization for one module path during weight loading,
/// consulting `aliasing` when the per-layer table has no explicit entry for
/// a post-sanitize (renamed/fused) module path.
public func resolveQuantization(
    path: String,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization,
    aliasing: QuantizationPathAliasing?
) -> BaseConfiguration.Quantization? {
    if perLayerQuantization.perLayerQuantization[path] == nil, let aliasing {
        for alias in aliasing.quantizationPathAliases(for: path)
        where perLayerQuantization.perLayerQuantization[alias] != nil {
            return perLayerQuantization.quantization(layer: alias)
        }
    }
    return perLayerQuantization.quantization(layer: path)
}

/// Implemented by models whose ``BaseLanguageModel/sanitize(weights:)``
/// makes module-topology decisions that depend on the checkpoint's
/// quantization policy.
///
/// Fusing two checkpoint modules into one (e.g. split `gate_proj`/`up_proj`
/// experts into a fused `gate_up_proj`) is only representable when both
/// halves share one quantization policy — a single quantized projection has
/// one bits/group_size/mode. Sanitizers consult the staged policy to keep
/// such pairs split (and reshape the module tree accordingly) whenever the
/// halves' effective policies differ.
///
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``
/// stages the checkpoint's resolved policy here before calling `sanitize`;
/// a uniform `quantization` is staged as a table with only a default.
public protocol QuantizationPolicyReceiving: AnyObject {
    /// Quantization policy of the checkpoint currently being loaded, or
    /// `nil` for unquantized checkpoints.
    var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization? { get set }
}

/// Implemented by models that must keep some checkpoint tensors out of the
/// load.
///
/// The contract:
///
/// * ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``
///   asks `shouldLoadWeight(named:)` about every tensor name in every shard.
/// * The name is the original name on the disk, before `sanitize` renames it.
/// * The loader asks before it materializes the array. mlx reads the
///   safetensors header and makes one unevaluated array for each tensor, so
///   an excluded tensor never reads its bytes.
/// * An excluded tensor is not in the dictionary that goes to
///   ``BaseLanguageModel/sanitize(weights:metadata:)``.
/// * A model that does not conform to this protocol keeps all tensors.
public protocol WeightNameFiltering {
    /// Answer `false` to keep the tensor `name` out of the load.
    func shouldLoadWeight(named name: String) -> Bool
}

/// Opt-in load-time materialization for a model whose sanitizer creates
/// large lazy packed-weight copies. Other models retain the existing load
/// sequence. The hook runs after strict update and removal of both loader
/// staging owners, before dtype conversion and the final model eval.
public protocol IncrementalCheckpointMaterializing: AnyObject {
    var needsIncrementalCheckpointMaterialization: Bool { get }
    func materializeCheckpointWeightsIncrementally() throws
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    let bench = ProcessInfo.processInfo.environment["BENCH_VERBOSE"] != nil
    var t = CFAbsoluteTimeGetCurrent()
    func mark(_ label: String) {
        if bench {
            let now = CFAbsoluteTimeGetCurrent()
            FileHandle.standardError.write(
                Data("    [stage] \(label): \(String(format: "%.1f", (now - t) * 1000)) ms\n"
                    .utf8))
            t = now
        }
    }

    // Gather the shard URLs first so we can fan them out concurrently.
    var shardURLs: [URL] = []
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator where url.pathExtension == "safetensors" {
        shardURLs.append(url)
    }
    shardURLs.sort { $0.lastPathComponent < $1.lastPathComponent }

    // Models can exclude tensors by name before they are materialized. The
    // shard tasks below run the predicate concurrently; it only reads a name
    // and answers, so `nonisolated(unsafe)` is the correct annotation for the
    // non-Sendable model reference (same reason `ParallelShardState` above is
    // `@unchecked Sendable`).
    nonisolated(unsafe) let weightFilter = model as? WeightNameFiltering

    // Hand the kernel a head start on every shard. F_RDADVISE is Darwin's
    // async-prefetch primitive — it issues a non-blocking advisory read into
    // the unified buffer cache, letting the SSD start streaming pages before
    // we ask for them. Net cost is one open/fcntl/close per shard. By the
    // time the DispatchQueue.concurrentPerform tasks below try to read, the
    // pages may already be resident.
    prefetchShards(shardURLs)
    mark("rdadvise")

    // Load shards in parallel. Each task forces eval() on its arrays so MLX
    // actually performs the disk read inside the task rather than deferring all
    // of it to a single later eval(model) call. This lets concurrent shards
    // overlap their I/O.
    typealias ShardResult = (weights: [String: MLXArray], metadata: [String: String])
    let shared = ParallelShardState(shardCount: shardURLs.count)
    let urls = shardURLs  // immutable Sendable snapshot for the closure

    DispatchQueue.concurrentPerform(iterations: urls.count) { idx in
        do {
            var (w, m) = try loadArraysAndMetadata(url: urls[idx])
            // Drop the excluded names BEFORE the eval() below. mlx's
            // safetensors reader parses only the header and returns one
            // unevaluated array per tensor, so an array that is dropped here
            // never reads its bytes.
            if let weightFilter {
                w = w.filter { weightFilter.shouldLoadWeight(named: $0.key) }
            }
            if !w.isEmpty {
                eval(Array(w.values))
            }
            shared.store(index: idx, result: (w, m))
        } catch {
            shared.recordError(error)
        }
    }
    if let err = shared.firstError { throw err }

    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for (i, slot) in shared.results.enumerated() {
        guard let (w, m) = slot else { continue }
        for (key, value) in w { weights[key] = value }
        // Match the original "first iterated shard's metadata" semantics by
        // taking shard 0's, falling back to next non-empty for safety.
        if i == 0 || metadata.isEmpty { metadata = m }
    }
    mark("read shards (parallel)")

    let prism = try (model as? any PrismHadamardLoading).map {
        try PrismHadamardCheckpoint(directory: modelDirectory,
            configuration: $0.prismCheckpoint, weights: weights)
    }
    if prism != nil { weights = weights.filter { !$0.key.hasSuffix(".signs") } }

    // Stage the checkpoint's quantization policy for sanitizers whose
    // module-topology decisions depend on it (e.g. the Qwen3.5 routed-expert
    // gate/up fusion, which must keep heterogeneous pairs split).
    if let policyReceiving = model as? QuantizationPolicyReceiving {
        policyReceiving.checkpointPerLayerQuantization =
            perLayerQuantization
            ?? quantization.map {
                BaseConfiguration.PerLayerQuantization(
                    quantization: $0, perLayerQuantization: [:])
            }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)
    mark("sanitize")

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        let aliasing = model as? QuantizationPathAliasing
        quantize(model: model) { path, module in
            guard weights["\(path).scales"] != nil else { return nil }
            if let perLayerQuantization {
                return resolveQuantization(
                    path: path, perLayerQuantization: perLayerQuantization,
                    aliasing: aliasing)?.asTuple
            } else {
                return quantization?.asTuple
            }
        }
    }
    mark("quantize wire")
    try prism?.install(model: model, weights: weights)

    // apply the loaded weights
    var parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])
    mark("update params")

    // Drop the staging dictionary before dtype conversion so we don't keep
    // two copies of safetensor arrays alive during the bf16 pass.
    weights.removeAll(keepingCapacity: false)
    if let materializing = model as? IncrementalCheckpointMaterializing,
        materializing.needsIncrementalCheckpointMaterialization
    {
        parameters = ModuleParameters()
        shared.releaseResults()
        try materializing.materializeCheckpointWeightsIncrementally()
        mark("incremental checkpoint materialization")
    }
    MLX.Memory.clearCache()

    // Convert fp16 parameters to bf16 to eliminate AsType cascades.
    // Metal's kernel dispatcher promotes mixed fp16/fp32 operations to full fp32,
    // causing extra kernel dispatches across all layers. bf16 shares fp32's exponent
    // range, so the promotion is eliminated. Quantization scales/biases are also
    // converted — QuantizedMatmul uses scales dtype to determine output dtype.
    //
    // Controlled by DARKBLOOM_BF16_WEIGHTS (default: ON, set to "0" to disable).
    let bf16Env = ProcessInfo.processInfo.environment["DARKBLOOM_BF16_WEIGHTS"] ?? "1"
    if bf16Env == "1" && prism == nil {
        convertToBFloat16(model: model)
        mark("bf16 convert")
    }

    eval(model)
    mark("eval")
}

// MARK: - BFloat16 Weight Conversion

/// Convert float16 model parameters to bfloat16 to prevent AsType cascades.
///
/// Metal's kernel dispatcher promotes mixed float16/float32 operations to full float32,
/// causing speed regression for models where routing/normalization runs at float32.
/// bfloat16 avoids this because it shares float32's exponent range.
///
/// Quantization scales/biases are also converted — QuantizedMatmul uses scales dtype to
/// determine output dtype, so float16 scales → float16 output → AsType when multiplied
/// with bfloat16 norms. Converting scales to bfloat16 eliminates this cascade.
///
/// The conversion is chunked to bound transient memory. Large quantized models carry
/// tens of GB of scale/bias metadata; converting all arrays at once keeps both fp16 and
/// bf16 copies alive until the final eval.
private func convertToBFloat16(model: Module) {
    let t0 = CFAbsoluteTimeGetCurrent()

    let convertibleParams: [(key: String, convertedBytes: Int)] = {
        let flat = model.parameters().flattened()
        return flat.compactMap { key, array in
            guard array.dtype == .float16 else { return nil }
            return (key: key, convertedBytes: estimatedByteCount(array, as: .bfloat16))
        }
    }()

    guard !convertibleParams.isEmpty else { return }

    let totalBytes = convertibleParams.reduce(0) { $0 + $1.convertedBytes }
    let chunkLimit = bfloat16ConversionChunkLimit()
    var index = 0
    var convertedCount = 0

    while index < convertibleParams.count {
        var converted = [String: MLXArray]()
        var chunkBytes = 0

        do {
            let current = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            while index < convertibleParams.count {
                let entry = convertibleParams[index]
                if !converted.isEmpty,
                    chunkBytes + entry.convertedBytes > chunkLimit
                {
                    break
                }

                guard let array = current[entry.key],
                    array.dtype == DType.float16
                else {
                    index += 1
                    continue
                }

                converted[entry.key] = array.asType(DType.bfloat16)
                chunkBytes += entry.convertedBytes
                index += 1
            }
        }

        guard !converted.isEmpty else { continue }

        let values = Array(converted.values)
        MLX.eval(values)
        convertedCount += converted.count

        let params = ModuleParameters.unflattened(converted)
        do {
            try model.update(parameters: params, verify: [])
        } catch {
            FileHandle.standardError.write(
                Data("[bf16] model.update failed: \(error)\n".utf8))
        }
        MLX.Memory.clearCache()
    }

    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
    let mb = Double(totalBytes) / (1024 * 1024)
    FileHandle.standardError.write(
        Data(
            "[bf16] converted \(convertedCount) params (\(String(format: "%.1f", mb)) MB) fp16→bf16 in \(String(format: "%.0f", elapsed)) ms\n"
                .utf8))
}

/// Maximum bytes of bf16 output to accumulate per conversion chunk.
/// Override with DARKBLOOM_BF16_CHUNK_MB environment variable.
private func bfloat16ConversionChunkLimit() -> Int {
    if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_BF16_CHUNK_MB"],
        let mb = Int(raw), mb > 0
    {
        return mb * 1024 * 1024
    }
    return 256 * 1024 * 1024
}

private func estimatedByteCount(_ array: MLXArray, as dtype: DType) -> Int {
    let elements = array.shape.reduce(1) { partial, dim in
        partial * max(dim, 1)
    }
    return elements * dtypeByteWidth(dtype)
}

private func dtypeByteWidth(_ dtype: DType) -> Int {
    if dtype == .bool || dtype == .int8 || dtype == .uint8 {
        return 1
    }
    if dtype == .float16 || dtype == .bfloat16
        || dtype == .int16 || dtype == .uint16
    {
        return 2
    }
    if dtype == .int64 || dtype == .uint64 {
        return 8
    }
    return 4  // float32, int32, uint32, complex64
}
