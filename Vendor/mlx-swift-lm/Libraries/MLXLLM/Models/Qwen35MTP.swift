// Copyright © 2026 Eigen Labs.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/qwen35_model.py  (MTPDecoderLayer, MTPModule)
//   patches/mlx_lm_mtp/__init__.py        (is_mtp_active / set_mtp_active)

import CoreFoundation
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Module-level MTP flag

/// Controls whether Qwen3.5/3.6 model inits attach the MTP head.
/// Set to `true` before calling `MLXLLM.load(...)` when MTP should be active.
/// Mirrors omlx `is_mtp_active()` / `set_mtp_active()` from
/// patches/mlx_lm_mtp/__init__.py.
public nonisolated(unsafe) var _qwen35MTPEnabled: Bool = false

/// Fail-loud errors for the production-safe inline Qwen MTP loader.
///
/// Unlike the legacy process-global attachment flag above, this loader builds
/// an assistant explicitly for one already-loaded target and never changes
/// model construction behavior process-wide.
public enum Qwen35InlineMTPError: Error, LocalizedError, Sendable, Equatable {
    case invalidConfiguration(String)
    case incompatibleTarget(field: String, artifact: Int, target: Int)
    case invalidWeightIndex(String)
    case missingWeights
    case duplicateWeight(String)
    case missingQuantization(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Invalid inline Qwen MTP configuration: \(detail)."
        case .incompatibleTarget(let field, let artifact, let target):
            return "Inline Qwen MTP target mismatch at \(field): artifact=\(artifact), target=\(target)."
        case .invalidWeightIndex(let detail):
            return "Invalid inline Qwen MTP weight index: \(detail)."
        case .missingWeights:
            return "The checkpoint declares inline Qwen MTP but contains no matching tensors."
        case .duplicateWeight(let key):
            return "The inline Qwen MTP tensor \(key) appears more than once."
        case .missingQuantization(let path):
            return "The quantized inline Qwen MTP module \(path) has no matching quantization entry."
        }
    }
}

/// Parsed, bounded metadata for an inline Qwen MTP assistant.
/// Internal so focused tests can validate checkpoint interpretation without
/// constructing hundreds of MiB of weights.
struct Qwen35InlineMTPMetadata: Sendable {
    let textConfiguration: Qwen35TextConfiguration
    let prefix: String
    let blockSize: Int
    let quantization: BaseConfiguration.PerLayerQuantization?

    func resolvedQuantization(
        for path: String
    ) -> BaseConfiguration.Quantization? {
        quantization?.quantization(layer: path)
    }
}

// MARK: - MTP history KV

extension Qwen35Attention {
    /// Append committed proposal-head history without computing unused query,
    /// gate, attention, or output-projection rows.
    func appendMTPHistoryKV(_ x: MLXArray, cache: any KVCache) {
        let batch = x.dim(0)
        let length = x.dim(1)
        var keys = kNorm(kProj(x).reshaped(batch, length, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)
        _ = cache.update(keys: keys, values: values)
    }
}

// MARK: - MTPDecoderLayer

/// Full-attention transformer layer used inside the Qwen3.5/3.6 MTP head.
/// Unlike `Qwen35DecoderLayer`, this always uses full attention (never SSM/linear).
/// MoE config is honoured when `num_experts > 0`.
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPDecoderLayer
final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        if args.numExperts > 0 {
            // Split gate/up: the assistant's quantization table
            // (`mtplx_mtp_quantization`) and the checkpoint's mtp.* tensors
            // are keyed on the split module paths, and the MTP head loads
            // outside the target sanitizers that perform gate/up fusion.
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args, fuseGateUp: false)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)?
    ) -> MLXArray {
        // omlx: MTPDecoderLayer.__call__
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    /// Populate this layer's K/V history without computing a dead decoder
    /// output. Only valid when no later MTP layer consumes that output.
    func appendHistoryKV(_ x: MLXArray, cache: any KVCache) {
        selfAttn.appendMTPHistoryKV(inputLayerNorm(x), cache: cache)
    }
}

// MARK: - MTPModule

/// Multi-Token Prediction head for Qwen3.5/3.6.
///
/// Fuses the backbone's final-normalized hidden state at position t with the
/// embedding of the sampled main token (t+1) to predict the draft token at (t+2).
///
/// Architecture (port of PR #990):
/// ```
/// pre_fc_norm_hidden:    RMSNorm(hidden_size)
/// pre_fc_norm_embedding: RMSNorm(hidden_size)
/// fc:                    Linear(hidden_size * 2 → hidden_size, bias: false)
/// layers:                [MTPDecoderLayer]  × mtp_num_hidden_layers
/// norm:                  RMSNorm(hidden_size)
/// ```
/// omlx: patches/mlx_lm_mtp/qwen35_model.py MTPModule
final class Qwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    // `layers` uses the default ModuleInfo key derived from the property name.
    let layers: [Qwen35MTPDecoderLayer]
    let norm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _preFcNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFcNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self.layers = (0 ..< args.mtpNumHiddenLayers).map { _ in
            Qwen35MTPDecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray {
        // omlx: MTPModule.__call__
        // 1. Embed next-token ids and fuse with normed hidden state.
        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        var fused = fc(concatenated([e, h], axis: -1))

        // 2. Compute attention mask from the first cache entry (or nil if empty).
        let firstCache: (any KVCache)? = cache.first
        let mask = createAttentionMask(h: fused, cache: firstCache)

        // 3. Run each MTPDecoderLayer.
        for (i, layer) in layers.enumerated() {
            let c: (any KVCache)? = i < cache.count ? cache[i] : nil
            fused = layer(fused, mask: mask, cache: c)
        }

        // 4. Return pre-lm_head hidden (norm applied; lm_head is in TextModel).
        return norm(fused)
    }

    /// Append every leading committed row through a K/V-only path and compute
    /// a full decoder output only for the final proposal row. Multi-layer heads
    /// fail closed before any cache mutation because later layers require the
    /// omitted leading outputs.
    func lastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [any KVCache]
    ) -> MLXArray? {
        guard layers.count == 1, cache.count == 1,
            hidden.dim(1) > 1,
            nextTokenIds.dim(1) == hidden.dim(1)
        else { return nil }

        let embeds = embedTokens(nextTokenIds)
        let e = preFcNormEmbedding(embeds)
        let h = preFcNormHidden(hidden)
        let fused = fc(concatenated([e, h], axis: -1))
        let historyCount = fused.dim(1) - 1
        layers[0].appendHistoryKV(
            fused[0..., 0 ..< historyCount, 0...], cache: cache[0])

        let current = fused[0..., historyCount..., 0...]
        let mask = createAttentionMask(h: current, cache: cache[0])
        return norm(layers[0](current, mask: mask, cache: cache[0]))
    }
}

// MARK: - Artifact-scoped inline assistant

/// An explicitly loaded Qwen3.5/3.6 MTP assistant bound to one target.
///
/// The assistant owns only the tensors below `mtp.*`. Target embeddings and
/// the LM head are called through the bound target, so loading this object does
/// not duplicate the target checkpoint and does not require
/// `_qwen35MTPEnabled`.
public final class Qwen35InlineMTPAssistant: Module, @unchecked Sendable {
    static let cacheAllocationStep = 256

    private let mtp: Qwen35MTPModule
    private let target: Qwen35TextModel
    private let installedVerificationMode: CBv2MTPVerificationMode

    public let blockSize: Int
    public var targetIdentity: ObjectIdentifier { ObjectIdentifier(target) }

    var prefixCheckpointGeometry: (width: Int, dtype: DType, vocabulary: Int, verification: String) {
        (target.configuration.hiddenSize, target.model.norm.weight.dtype,
         target.vocabularySize, installedVerificationMode.rawValue)
    }

    private init(
        configuration: Qwen35TextConfiguration,
        blockSize: Int,
        target: Qwen35TextModel,
        verificationMode: CBv2MTPVerificationMode?
    ) {
        self.mtp = Qwen35MTPModule(configuration)
        self.blockSize = blockSize
        self.target = target
        self.installedVerificationMode = Self.resolvedVerificationMode(
            requested: verificationMode,
            forceSerialEnvironment: Self.forceSerialVerification)
        super.init()
    }

    static func resolvedVerificationMode(
        requested: CBv2MTPVerificationMode?, forceSerialEnvironment: Bool
    ) -> CBv2MTPVerificationMode {
        if forceSerialEnvironment { return .serialTarget }
        return requested ?? .rectangular
    }

    /// Allocate the assistant's own autoregressive full-attention KV.
    /// Target recurrent and attention state is never reused as assistant KV.
    public func makeCache() -> [any KVCache] {
        mtp.layers.map { _ in
            let cache = KVCacheSimple()
            cache.step = Self.cacheAllocationStep
            return cache as any KVCache
        }
    }

    /// Advance the assistant by one or more already-chosen target/draft tokens.
    /// Returns logits from the target's shared output projection and the
    /// assistant hidden used to continue a draft chain.
    public func forward(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let output = mtp(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
        return (headLogits(output), output)
    }

    /// The target's shared output projection over assistant hidden states.
    func headLogits(_ output: MLXArray) -> MLXArray {
        if target.configuration.tieWordEmbeddings {
            return target.model.embedTokens.asLinear(output)
        }
        return target.lmHead!(output)
    }

    /// Apply the exact final norm of the bound target once at the boundary
    /// where target-derived hidden enters assistant history. Target hidden
    /// capture is deliberately pre-norm; recursive assistant hidden is already
    /// post-`mtp.norm` and must never pass through this helper.
    func targetFinalNorm(_ hidden: MLXArray) -> MLXArray {
        target.model.norm(hidden)
    }

    /// Assistant hidden states WITHOUT the output projection — the draft
    /// path applies the head to the last position only (or to a shortlist).
    func moduleForward(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> MLXArray {
        mtp(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
    }

    /// History-flush specialization. Returns one final hidden row while every
    /// leading trusted row contributes only K/V state. nil means the head
    /// geometry cannot safely omit intermediate layer outputs.
    func moduleLastHiddenWithKVOnlyHistory(
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [any KVCache]
    ) -> MLXArray? {
        mtp.lastHiddenWithKVOnlyHistory(
            hidden: hidden,
            nextTokenIds: tokens,
            embedTokens: target.model.embedTokens,
            cache: cache)
    }

    /// Draft logits over an engine-provided shortlist of token ids. The
    /// draft needs only an argmax, so score the gathered head rows instead
    /// of streaming the full `[V, H]` output projection (~K/V of the bytes;
    /// the 4-bit Qwen3.6 lm_head alone is ~286 MB per read). Quantized
    /// heads gather packed rows plus their scales/biases and stay on the
    /// quantized-matmul path; float heads (small test fixtures) gather
    /// plain rows.
    ///
    /// A PACKED HADAMARD head folds the transform into its rows, so the
    /// rotation the module applies in `callAsFunction` has to be applied to
    /// the hidden state here as well. `HadamardQuantizedLinear` IS a
    /// `QuantizedLinear`, so without the rotation the cast below succeeds and
    /// scores folded rows against an unrotated hidden state.
    /// `HadamardQuantizedEmbedding` is NOT a `QuantizedEmbedding`, so without
    /// its own branch the tied path reads packed integers as floats.
    func shortlistLogits(hidden: MLXArray, ids: MLXArray) -> MLXArray {
        if target.configuration.tieWordEmbeddings {
            let embed = target.model.embedTokens
            if let packed = embed as? HadamardQuantizedEmbedding {
                return quantizedMM(
                    packed.transform(hidden), packed.weight[ids],
                    scales: packed.scales[ids],
                    biases: packed.biases.map { $0[ids] },
                    transpose: true,
                    groupSize: packed.groupSize, bits: packed.bits,
                    mode: packed.mode)
            }
            if let quantized = embed as? QuantizedEmbedding {
                return quantizedMM(
                    hidden, quantized.weight[ids],
                    scales: quantized.scales[ids],
                    biases: quantized.biases.map { $0[ids] },
                    transpose: true,
                    groupSize: quantized.groupSize, bits: quantized.bits,
                    mode: quantized.mode)
            }
            return matmul(hidden, embed.weight[ids].transposed(1, 0))
        }
        let head = target.lmHead!
        let projected = (head as? HadamardQuantizedLinear).map { $0.transform(hidden) } ?? hidden
        if let quantized = head as? QuantizedLinear {
            var logits = quantizedMM(
                projected, quantized.weight[ids],
                scales: quantized.scales[ids],
                biases: quantized.biases.map { $0[ids] },
                transpose: true,
                groupSize: quantized.groupSize, bits: quantized.bits,
                mode: quantized.mode)
            if let bias = quantized.bias { logits = logits + bias[ids] }
            return logits
        }
        var logits = matmul(projected, head.weight[ids].transposed(1, 0))
        if let bias = head.bias { logits = logits + bias[ids] }
        return logits
    }

    /// Strictly load either an inline assistant declared by a combined
    /// checkpoint or a standalone `qwen3_5_mtp` artifact. Inline artifacts
    /// read only indexed keys below their declared prefix; standalone
    /// artifacts read every tensor from their bounded safetensors directory.
    public static func load(
        from modelDirectory: URL,
        target: any LanguageModel,
        verificationMode: CBv2MTPVerificationMode? = nil
    ) throws -> Qwen35InlineMTPAssistant {
        let target = try qwen35TextTarget(target)
        let resolvedVerificationMode = Self.resolvedVerificationMode(
            requested: verificationMode,
            forceSerialEnvironment: Self.forceSerialVerification)
        if resolvedVerificationMode == .rectangularExact && !target.model.exactTargetVerify {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "rectangular_exact verification requires exact target arithmetic")
        }
        let metadata = try loadMetadata(from: modelDirectory)
        try validate(metadata.textConfiguration, against: target.configuration)

        let indexed: [String: MLXArray]
        if metadata.prefix.isEmpty {
            indexed = try loadStandaloneWeights(from: modelDirectory)
        } else {
            indexed = try loadIndexedWeights(
                from: modelDirectory, prefix: metadata.prefix)
        }
        let assistant = Qwen35InlineMTPAssistant(
            configuration: metadata.textConfiguration,
            blockSize: metadata.blockSize,
            target: target,
            verificationMode: verificationMode)

        let scaledPaths = Set(indexed.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        })
        for path in scaledPaths
        where metadata.resolvedQuantization(for: path) == nil {
            throw Qwen35InlineMTPError.missingQuantization(path)
        }
        if !scaledPaths.isEmpty {
            quantize(model: assistant.mtp) { path, _ in
                guard scaledPaths.contains(path) else { return nil }
                return metadata.resolvedQuantization(for: path)?.asTuple
            }
        }

        try assistant.mtp.update(
            parameters: ModuleParameters.unflattened(indexed), verify: [.all])
        eval(assistant.mtp)
        return assistant
    }

    private static func qwen35TextTarget(
        _ target: any LanguageModel
    ) throws -> Qwen35TextModel {
        if let target = target as? Qwen35TextModel { return target }
        if let target = target as? Qwen35Model { return target.languageModel }
        throw Qwen35InlineMTPError.invalidConfiguration(
            "target type \(String(describing: type(of: target))) is not Qwen3.5/3.6")
    }

    private static let globalQuantizationKeys: Set<String> = [
        "group_size", "bits", "mode"
    ]

    /// Scalar keys understood or deliberately ignored by
    /// `BaseConfiguration.QuantizationContainer`; all other keys name modules.
    private static let quantizationMetadataKeys =
        globalQuantizationKeys.union([
            "quant_method", "linear_class", "quantization_mode"
        ])

    private static func selectedJSONObject(
        in root: [String: Any],
        primaryKey: String,
        fallbackKey: String
    ) throws -> [String: Any]? {
        for key in [primaryKey, fallbackKey] {
            guard let raw = root[key], !(raw is NSNull) else { continue }
            guard let object = raw as? [String: Any] else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "standalone MTP quantization must be an object")
            }
            return object
        }
        return nil
    }

    private static func decodeQuantization(
        _ rawQuantization: [String: Any]
    ) throws -> BaseConfiguration.PerLayerQuantization {
        var perLayer = [String: BaseConfiguration.QuantizationOption]()
        for (path, raw) in rawQuantization
        where !quantizationMetadataKeys.contains(path) {
            if CFGetTypeID(raw as CFTypeRef) == CFBooleanGetTypeID() {
                guard (raw as? Bool) == false else {
                    throw Qwen35InlineMTPError.invalidConfiguration(
                        "quantization entry \(path) must be false or an object")
                }
                perLayer[path] = .skip
            } else {
                guard let object = raw as? [String: Any] else {
                    throw Qwen35InlineMTPError.invalidConfiguration(
                        "quantization entry \(path) must be false or an object")
                }
                let data = try JSONSerialization.data(withJSONObject: object)
                perLayer[path] = .quantize(
                    try JSONDecoder().decode(
                        BaseConfiguration.Quantization.self, from: data))
            }
        }

        let hasGlobalQuantization = rawQuantization.keys.contains {
            globalQuantizationKeys.contains($0)
        }
        let global: BaseConfiguration.Quantization?
        if hasGlobalQuantization {
            let data = try JSONSerialization.data(withJSONObject: rawQuantization)
            global = try JSONDecoder().decode(
                BaseConfiguration.Quantization.self, from: data)
        } else {
            global = nil
        }
        return BaseConfiguration.PerLayerQuantization(
            quantization: global, perLayerQuantization: perLayer)
    }

    static func loadMetadata(from directory: URL) throws -> Qwen35InlineMTPMetadata {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = root["text_config"] as? [String: Any]
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "text_config is required")
        }

        let prefix: String
        let blockSize: Int
        let rawQuantization: [String: Any]?
        if let inline = root["mtplx_mtp"] as? [String: Any],
            inline["included"] as? Bool == true
        {
            prefix = (inline["prefix"] as? String) ?? "mtp."
            guard prefix == "mtp." else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "only the mtp. prefix is supported")
            }
            blockSize = (inline["block_size"] as? NSNumber)?.intValue ?? 3
            guard let quantization = root["mtplx_mtp_quantization"] as? [String: Any]
            else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "mtplx_mtp_quantization is required")
            }
            rawQuantization = quantization
        } else {
            guard (root["model_type"] as? String)?.lowercased() == "qwen3_5_mtp"
            else {
                throw Qwen35InlineMTPError.invalidConfiguration(
                    "mtplx_mtp.included=true or model_type=qwen3_5_mtp is required")
            }
            prefix = ""
            blockSize = (root["block_size"] as? NSNumber)?.intValue ?? 3
            rawQuantization = try selectedJSONObject(
                in: root,
                primaryKey: "quantization",
                fallbackKey: "quantization_config")
        }
        guard (2...8).contains(blockSize) else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "block_size \(blockSize) is outside 2...8")
        }
        let textData = try JSONSerialization.data(withJSONObject: text)
        let configuration = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: textData)
        guard configuration.mtpNumHiddenLayers > 0,
            configuration.mtpNumHiddenLayers <= 4
        else {
            throw Qwen35InlineMTPError.invalidConfiguration(
                "mtp_num_hidden_layers must be within 1...4")
        }

        let quantization: BaseConfiguration.PerLayerQuantization?
        if let rawQuantization, !rawQuantization.isEmpty {
            quantization = try decodeQuantization(rawQuantization)
        } else {
            quantization = nil
        }
        return Qwen35InlineMTPMetadata(
            textConfiguration: configuration,
            prefix: prefix,
            blockSize: blockSize,
            quantization: quantization)
    }

    private struct WeightIndex: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    private static func loadIndexedWeights(
        from directory: URL,
        prefix: String
    ) throws -> [String: MLXArray] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let index: WeightIndex
        do {
            index = try JSONDecoder().decode(WeightIndex.self, from: Data(contentsOf: indexURL))
        } catch {
            throw Qwen35InlineMTPError.invalidWeightIndex(String(describing: error))
        }

        var byFile: [String: [(source: String, destination: String)]] = [:]
        for (key, file) in index.weightMap where key.hasPrefix(prefix) {
            guard file == URL(fileURLWithPath: file).lastPathComponent,
                file.hasSuffix(".safetensors")
            else {
                throw Qwen35InlineMTPError.invalidWeightIndex(
                    "unsafe shard path for \(key)")
            }
            let destination = String(key.dropFirst(prefix.count))
            guard !destination.isEmpty else {
                throw Qwen35InlineMTPError.invalidWeightIndex("empty stripped key")
            }
            byFile[file, default: []].append((key, destination))
        }
        guard !byFile.isEmpty else { throw Qwen35InlineMTPError.missingWeights }

        var weights: [String: MLXArray] = [:]
        for file in byFile.keys.sorted() {
            let url = directory.appendingPathComponent(file)
            let (shard, _) = try loadArraysAndMetadata(url: url)
            for entry in byFile[file]! {
                guard let value = shard[entry.source] else {
                    throw Qwen35InlineMTPError.invalidWeightIndex(
                        "indexed tensor \(entry.source) is absent from \(file)")
                }
                guard weights.updateValue(value, forKey: entry.destination) == nil else {
                    throw Qwen35InlineMTPError.duplicateWeight(entry.destination)
                }
            }
        }
        return weights
    }

    static func loadStandaloneWeights(
        from directory: URL
    ) throws -> [String: MLXArray] {
        let urls: [URL]
        do {
            urls = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
                .filter { $0.pathExtension == "safetensors" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
        } catch {
            throw Qwen35InlineMTPError.invalidWeightIndex(String(describing: error))
        }
        guard !urls.isEmpty, urls.count <= 64 else {
            throw Qwen35InlineMTPError.missingWeights
        }

        var weights: [String: MLXArray] = [:]
        for url in urls {
            let (shard, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in shard {
                guard !key.isEmpty, key.utf8.count <= 1024 else {
                    throw Qwen35InlineMTPError.invalidWeightIndex(
                        "standalone tensor key is empty or oversized")
                }
                guard weights.updateValue(value, forKey: key) == nil else {
                    throw Qwen35InlineMTPError.duplicateWeight(key)
                }
            }
        }
        guard !weights.isEmpty else { throw Qwen35InlineMTPError.missingWeights }
        return weights
    }

    private static func validate(
        _ artifact: Qwen35TextConfiguration,
        against target: Qwen35TextConfiguration
    ) throws {
        let fields: [(String, Int, Int)] = [
            ("hidden_size", artifact.hiddenSize, target.hiddenSize),
            ("vocab_size", artifact.vocabularySize, target.vocabularySize),
            ("num_attention_heads", artifact.attentionHeads, target.attentionHeads),
            ("num_key_value_heads", artifact.kvHeads, target.kvHeads),
            ("head_dim", artifact.headDim ?? 0, target.headDim ?? 0),
            ("num_experts", artifact.numExperts, target.numExperts),
            ("num_experts_per_tok", artifact.numExpertsPerTok, target.numExpertsPerTok),
        ]
        for (field, artifactValue, targetValue) in fields
        where artifactValue != targetValue {
            throw Qwen35InlineMTPError.incompatibleTarget(
                field: field, artifact: artifactValue, target: targetValue)
        }
    }
}

extension Qwen35InlineMTPAssistant: CBv2MTPRequestStatefulDrafter {
    final class RequestState: CBv2MTPRequestState {
        var caches: [any KVCache]

        /// Trusted target transitions not yet appended to the persistent head KV.
        /// Each hidden row at position t is paired with the token at t+1.
        var backlogHidden: [MLXArray] = []
        var backlogTokens: [MLXArray] = []
        /// Last trusted target hidden in an observed chunk. It becomes the
        /// preceding row when the next observed target chunk crosses a boundary.
        var targetHiddenFrontier: MLXArray?

        /// Cache geometry captured before and after the round's trusted flush.
        /// Every later head-chain input is speculative and is trimmed to
        /// `roundValidHistoryOffset` at finalize.
        var roundBaseOffset = 0
        var roundValidHistoryOffset = 0
        var roundDraftSteps = 0
        var roundInFlight = false
        var isReleased = false

        /// Trusted inputs moved out of the backlog for this round. Retaining
        /// their original roots both fences lazy concatenation and lets discard
        /// restore them without a host read.
        var roundTrustedHidden: [MLXArray] = []
        var roundTrustedTokens: [MLXArray] = []
        /// Final proposal hidden rows and draft ids that retain each lazy
        /// head-step graph until the engine's existing finalize synchronization.
        var roundRoots: [MLXArray] = []

        var cacheOffset: Int {
            guard let first = caches.first else { return 0 }
            precondition(
                caches.dropFirst().allSatisfy { $0.offset == first.offset },
                "inline Qwen MTP cache offsets diverged")
            return first.offset
        }

        private var backlogInputCount: Int {
            backlogTokens.reduce(0) { total, tokens in
                let (next, overflow) = total.addingReportingOverflow(tokens.dim(1))
                return overflow ? Int.max : next
            }
        }

        var committedInputCount: Int {
            let committedCache =
                roundInFlight ? roundValidHistoryOffset : cacheOffset
            let (total, overflow) = committedCache.addingReportingOverflow(
                backlogInputCount)
            return overflow ? Int.max : total
        }

        var stagedInputCount: Int {
            guard roundInFlight else { return 0 }
            return max(0, cacheOffset - roundValidHistoryOffset)
        }

        var materializedBytes: Int {
            let arrays =
                caches.flatMap { $0.innerState() }
                + backlogHidden + backlogTokens
                + [targetHiddenFrontier].compactMap { $0 }
                + roundTrustedHidden + roundTrustedTokens + roundRoots
            return arrays.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }

        init(caches: [any KVCache]) { self.caches = caches }

        func clearRound() {
            roundBaseOffset = cacheOffset
            roundValidHistoryOffset = cacheOffset
            roundDraftSteps = 0
            roundInFlight = false
            roundTrustedHidden.removeAll(keepingCapacity: true)
            roundTrustedTokens.removeAll(keepingCapacity: true)
            roundRoots.removeAll(keepingCapacity: true)
        }

        func clearAll() {
            caches.removeAll(keepingCapacity: false)
            backlogHidden.removeAll(keepingCapacity: false)
            backlogTokens.removeAll(keepingCapacity: false)
            targetHiddenFrontier = nil
            roundTrustedHidden.removeAll(keepingCapacity: false)
            roundTrustedTokens.removeAll(keepingCapacity: false)
            roundRoots.removeAll(keepingCapacity: false)
            roundBaseOffset = 0
            roundValidHistoryOffset = 0
            roundDraftSteps = 0
            roundInFlight = false
            isReleased = true
        }
    }

    private final class UnusedPreparedCapture: CBv2MTPPreparedCapture {}

    public var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }
    /// Verification policy. Rectangular scores all 1+k columns in one
    /// `[B, 1+k]` target forward. Widths one and two stage ordinary captured
    /// recurrent states. At S>=3 each GatedDeltaNet layer runs one full-window
    /// recurrence and retains compact transformed inputs: full acceptance
    /// installs the final state directly, while a strict accepted prefix lazily
    /// replays only that prefix. This replaces the serial per-column loop whose
    /// k+1 full-model re-reads made the round ≤1.0x by construction.
    ///
    /// NUMERICS POLICY: bitwise greedy parity with serial decode is NOT the
    /// bar here — batched verify changes accumulation geometry exactly like
    /// every other CBv2 batch-shape change (chunked prefill, B>1 decode).
    /// Distribution-exactness is the invariant: committed tokens are always
    /// target-authoritative (argmax when greedy, genuine target samples
    /// under target-prefix acceptance). The serial oracle remains available
    /// via `DARKBLOOM_QWEN_MTP_SERIAL=1` for A/B and certification runs.
    public var requiredVerificationMode: CBv2MTPVerificationMode? {
        installedVerificationMode
    }
    /// Production Qwen drafting self-applies the recurrent MTP head up to
    /// four times. The legacy double-forward oracle stays a k=1 A/B control.
    public var maximumDraftTokens: Int? { Self.forceDoubleForward ? 1 : 4 }
    public var maximumSpeculativeBatch: Int? { 1 }
    /// Target-prefix acceptance is exact at any temperature (every committed
    /// token IS a target sample), so this drafter lifts the greedy gate when
    /// the engine sampler supports verify pre-sampling.
    public var supportsTargetPrefixAcceptance: Bool { true }

    /// `DARKBLOOM_QWEN_MTP_SERIAL=1/true/yes/on` pins the serial per-column
    /// verification oracle (the pre-capture-verify behavior).
    static let forceSerialVerification: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_MTP_SERIAL"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()
    /// `DARKBLOOM_QWEN_MTP_DOUBLE_FORWARD=1/true/yes/on` pins the pre-trim
    /// draft step (proposal forward + staging re-forward, both through the
    /// full lm_head) as the paired A/B control arm.
    static let forceDoubleForward: Bool = {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "DARKBLOOM_QWEN_MTP_DOUBLE_FORWARD"]
        else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }()

    /// `DARKBLOOM_QWEN_MTP_SHORTLIST=<K>` opts the draft head into an
    /// engine-provided top-K shortlist (score K gathered lm_head rows
    /// instead of streaming all 248,320). DEFAULT OFF: measured on M4 Max
    /// (release, B=1 greedy, 256 tok, paired session 2026-08-13) the
    /// shortlist is a net LOSS at every practical K because the ids come
    /// from the target's distribution one position BEHIND the draft:
    ///   K=256   acceptance 0.79→0.58,  95.5 tok/s (full head: 116.5)
    ///   K=2048  acceptance      0.68, 110.3 tok/s
    ///   K=16384 acceptance      0.77, 115.2 tok/s — coverage recovered,
    ///           but per-round verify-side top-K + row gather costs more
    ///           than the ~1.3 ms full 286 MB 4-bit head read it replaces.
    /// Kept env-gated for deeper draft chains (k≥2 amortizes the top-K
    /// cost across several head evaluations per round).
    static let shortlistSize: Int? = {
        guard
            let raw = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN_MTP_SHORTLIST"],
            let value = Int(raw), value > 0
        else { return nil }
        return value
    }()

    /// Shortlisting and the double-forward oracle are intentionally separate
    /// controls: the oracle scores the complete shared output head.
    public var draftShortlistSize: Int? {
        Self.forceDoubleForward ? nil : Self.shortlistSize
    }
    public var requestStateBytesPerToken: Int {
        Self.stateBytesPerToken(
            configuration: target.configuration,
            layerCount: mtp.layers.count,
            cacheElementBytes: mtp.norm.weight.dtype.size,
            hiddenElementBytes: target.model.norm.weight.dtype.size)
    }
    public var requestStateTokenGranularity: Int { Self.cacheAllocationStep }
    public var requestStateTokenAllocationPadding: Int { 4 }

    public var requestStateAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
        let config = target.configuration
        let (elements, geometryOverflow) = config.kvHeads.multipliedReportingOverflow(by: config.headDim ?? 0)
        let (cacheRow, cacheOverflow) = elements.multipliedReportingOverflow(by: mtp.norm.weight.dtype.size)
        let (cacheCount, countOverflow) = mtp.layers.count.multipliedReportingOverflow(by: 2)
        let (hiddenRow, hiddenOverflow) = config.hiddenSize.multipliedReportingOverflow(
            by: target.model.norm.weight.dtype.size)
        guard !geometryOverflow, !cacheOverflow, !countOverflow, !hiddenOverflow else { return nil }
        return [
            .init(bytesPerToken: cacheRow, allocationCount: cacheCount,
                  tokenGranularity: Self.cacheAllocationStep, tokenPadding: 4),
            // Backlog and frontier are views of normalized committed chunks;
            // the bound covers every possible partition of the retained rows.
            .init(bytesPerToken: hiddenRow, tokenPadding: 4, partitioned: true),
            .init(bytesPerToken: MemoryLayout<Int32>.stride, tokenPadding: 4, partitioned: true),
        ]
    }

    static func cacheBytesPerToken(
        configuration: Qwen35TextConfiguration,
        layerCount: Int,
        elementBytes: Int
    ) -> Int {
        let (headsByDimension, geometryOverflow) = configuration.kvHeads
            .multipliedReportingOverflow(by: configuration.headDim ?? 0)
        let (kvElements, kvOverflow) = headsByDimension.multipliedReportingOverflow(by: 2)
        let (layerElements, layerOverflow) = kvElements.multipliedReportingOverflow(
            by: layerCount)
        let (bytes, byteOverflow) = layerElements.multipliedReportingOverflow(
            by: elementBytes)
        return geometryOverflow || kvOverflow || layerOverflow || byteOverflow ? Int.max : bytes
    }

    static func stateBytesPerToken(
        configuration: Qwen35TextConfiguration,
        layerCount: Int,
        cacheElementBytes: Int,
        hiddenElementBytes: Int
    ) -> Int {
        let cacheBytes = cacheBytesPerToken(
            configuration: configuration, layerCount: layerCount,
            elementBytes: cacheElementBytes)
        let (hiddenBytes, hiddenOverflow) = configuration.hiddenSize
            .multipliedReportingOverflow(by: hiddenElementBytes)
        guard cacheBytes != Int.max, !hiddenOverflow else { return Int.max }
        let (withHidden, hiddenAdditionOverflow) = cacheBytes.addingReportingOverflow(
            hiddenBytes)
        let (withToken, tokenAdditionOverflow) = withHidden.addingReportingOverflow(
            MemoryLayout<Int32>.stride)
        return hiddenAdditionOverflow || tokenAdditionOverflow ? Int.max : withToken
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        RequestState(caches: makeCache())
    }

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP observed released request state")
        precondition(!state.roundInFlight, "inline Qwen MTP observed target during a round")
        precondition(
            observation.tokens.ndim == 2 && observation.hidden.ndim == 3
                && observation.tokens.dim(0) == 1 && observation.hidden.dim(0) == 1
                && observation.tokens.dim(1) == observation.hidden.dim(1),
            "inline Qwen MTP target observation shape mismatch")

        let count = observation.tokens.dim(1)
        guard count > 0 else { return }
        let normalizedHidden = targetFinalNorm(observation.hidden)

        // Cross-chunk transition: the preceding chunk's final normalized
        // target hidden pairs with this chunk's first target input.
        if let frontier = state.targetHiddenFrontier {
            state.backlogHidden.append(frontier)
            state.backlogTokens.append(observation.tokens[0..., 0 ..< 1])
        }
        // Intra-chunk transitions: finalNorm(hidden[t]) conditions token[t+1].
        if count > 1 {
            state.backlogHidden.append(normalizedHidden[0..., 0 ..< count - 1, 0...])
            state.backlogTokens.append(observation.tokens[0..., 1 ..< count])
        }
        state.targetHiddenFrontier =
            normalizedHidden[0..., (count - 1) ..< count, 0...]
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        UnusedPreparedCapture()
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("inline Qwen MTP requires request-owned assistant state")
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP drafted with released request state")
        precondition(
            tokens.ndim == 2 && hidden.ndim == 3
                && tokens.dim(0) == 1 && tokens.dim(1) == 1
                && hidden.dim(0) == 1 && hidden.dim(1) == 1,
            "inline Qwen MTP draft input shape mismatch")

        if Self.forceDoubleForward {
            return legacyDoubleForwardDraftStep(tokens: tokens, hidden: hidden, state: state)
        }

        let isFirstStep = !state.roundInFlight
        let feed: (tokens: MLXArray, hidden: MLXArray)
        if isFirstStep {
            feed = beginRound(tokens: tokens, hidden: hidden, state: state)
        } else {
            precondition(
                state.roundDraftSteps < 4,
                "inline Qwen MTP exceeded its four-step draft chain")
            feed = (tokens, hidden)
        }

        let output =
            isFirstStep
            ? (moduleLastHiddenWithKVOnlyHistory(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
                ?? moduleForward(
                    hidden: feed.hidden, tokens: feed.tokens, cache: state.caches))
            : moduleForward(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
        let lastHidden = output[0..., (output.dim(1) - 1)..., 0...]
        let draft = draftToken(hidden: lastHidden, shortlist: shortlist)
        state.roundRoots.append(contentsOf: [lastHidden, draft])
        state.roundDraftSteps += 1

        if isFirstStep {
            // The engine immediately submits evaluationTargets for this first
            // cache generation before constructing a deeper draft step.
            state.roundValidHistoryOffset = state.cacheOffset
        }
        return (draft, lastHidden)
    }

    private func beginRound(
        tokens: MLXArray, hidden: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        precondition(!state.roundInFlight, "inline Qwen MTP round already in flight")
        precondition(
            state.backlogHidden.count == state.backlogTokens.count,
            "inline Qwen MTP trusted backlog diverged")

        state.roundBaseOffset = state.cacheOffset
        state.roundValidHistoryOffset = state.cacheOffset
        state.roundDraftSteps = 0
        state.roundInFlight = true
        state.roundTrustedHidden = state.backlogHidden
        state.roundTrustedTokens = state.backlogTokens
        state.backlogHidden.removeAll(keepingCapacity: true)
        state.backlogTokens.removeAll(keepingCapacity: true)

        // The current target carry is trusted and completes the frontier
        // transition. Normalize it exactly once before it enters head history.
        state.roundTrustedHidden.append(targetFinalNorm(hidden))
        state.roundTrustedTokens.append(tokens)
        state.targetHiddenFrontier = nil

        if state.roundTrustedTokens.count == 1 {
            return (state.roundTrustedTokens[0], state.roundTrustedHidden[0])
        }
        let feedTokens = concatenated(state.roundTrustedTokens, axis: 1)
        let feedHidden = concatenated(state.roundTrustedHidden, axis: 1)
        return (feedTokens, feedHidden)
    }

    private func draftToken(hidden: MLXArray, shortlist: MLXArray?) -> MLXArray {
        if let shortlist {
            let logits = shortlistLogits(hidden: hidden, ids: shortlist)
            return shortlist[argMax(logits[0..., -1, 0...], axis: -1)]
                .asType(.int32)
        }
        return argMax(headLogits(hidden)[0..., -1, 0...], axis: -1)
            .asType(.int32)
    }

    /// Explicit pre-cutover A/B oracle. It shares production history upkeep,
    /// but stages the proposed draft through a second complete MTP/lm-head
    /// forward and performs the historical blocking mid-round evaluation.
    private func legacyDoubleForwardDraftStep(
        tokens: MLXArray, hidden: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        precondition(!state.roundInFlight, "inline Qwen MTP round already in flight")
        let feed = beginRound(tokens: tokens, hidden: hidden, state: state)
        let output =
            moduleLastHiddenWithKVOnlyHistory(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
            ?? moduleForward(
                hidden: feed.hidden, tokens: feed.tokens, cache: state.caches)
        let lastHidden = output[0..., (output.dim(1) - 1)..., 0...]
        let logits = headLogits(lastHidden)
        let draft = argMax(logits[0..., -1, 0...], axis: -1).asType(.int32)
        state.roundRoots.append(contentsOf: [lastHidden, draft])
        state.roundDraftSteps = 1
        state.roundValidHistoryOffset = state.cacheOffset

        eval([draft, lastHidden] + state.caches.flatMap { $0.innerState() })
        _ = forward(
            hidden: lastHidden, tokens: draft.reshaped([1, 1]), cache: state.caches)
        return (draft, lastHidden)
    }

    public func evaluationTargets(
        for requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        guard let state = requestState as? RequestState, !state.isReleased else {
            return []
        }
        return state.caches.flatMap { $0.innerState() }
            + state.backlogHidden + state.backlogTokens
            + [state.targetHiddenFrontier].compactMap { $0 }
            + state.roundTrustedHidden + state.roundTrustedTokens + state.roundRoots
    }

    public func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        guard let state = requestState as? RequestState else {
            preconditionFailure("inline Qwen MTP received foreign request state")
        }
        precondition(!state.isReleased, "inline Qwen MTP finalized released request state")
        precondition(state.roundInFlight, "inline Qwen MTP finalized without a round")
        precondition(
            (0 ... state.roundDraftSteps + 1).contains(confirmedInputTokens),
            "inline Qwen MTP confirmed prefix exceeds the draft round")
        precondition(
            committedDraftTokens.ndim == 2 && committedTargetHidden.ndim == 3
                && committedDraftTokens.dim(0) == 1
                && committedTargetHidden.dim(0) == 1
                && committedDraftTokens.dim(1) == committedTargetHidden.dim(1),
            "inline Qwen MTP committed target rows mismatch")
        let committedDraftCount = committedDraftTokens.dim(1)
        precondition(
            committedDraftCount <= state.roundDraftSteps
                && committedDraftCount <= max(0, confirmedInputTokens - 1),
            "inline Qwen MTP committed drafts exceed confirmed target inputs")

        trim(state: state, to: state.roundValidHistoryOffset)
        if committedDraftCount > 0 {
            // These are pre-final-norm target verify hiddens, never speculative
            // assistant hiddens. Normalize exactly once, retain lazily, and
            // flush them with the next carry.
            state.backlogTokens.append(committedDraftTokens)
            state.backlogHidden.append(targetFinalNorm(committedTargetHidden))
        }
        state.clearRound()
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState,
            !state.isReleased, state.roundInFlight
        else { return }

        trim(state: state, to: state.roundBaseOffset)
        // Restore all trusted transitions consumed by the abandoned graph in
        // original order. No speculative assistant hidden enters the backlog.
        state.backlogHidden =
            state.roundTrustedHidden + state.backlogHidden
        state.backlogTokens =
            state.roundTrustedTokens + state.backlogTokens
        state.clearRound()
    }

    private func trim(state: RequestState, to offset: Int) {
        let rollback = state.cacheOffset - offset
        precondition(rollback >= 0, "inline Qwen MTP cache checkpoint moved forward")
        guard rollback > 0 else { return }
        for cache in state.caches {
            precondition(cache.trim(rollback) == rollback)
        }
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState, !state.isReleased else {
            return
        }
        state.clearAll()
    }
}

// MARK: - Attention prework that writes the layer's KV append in place

/// `Qwen35AttentionPrework` (q/k RMSNorm, head transpose, partial rotary
/// embedding) whose key rows go straight into the layer's full-sequence KV
/// storage, with the value rows copied there by the same launch
/// (`CBv2InPlaceKVAppend`): the append's two slice-update copies per
/// attention layer (32 launches per verify round of the 27B) are gone and the
/// keys are not written twice. The q/k arithmetic is the prework's, line for
/// line; only the destination of the key rows differs. At model construction
/// the kernel is compiled inside the MLX error handler, checked bit for bit
/// against the prework plus `updateAndAttend` (attention outputs, KV contents
/// and offsets over appends, a rollback and a storage growth), and timed
/// against it over 16-layer steps; it is used only when the self-test passes
/// and the trial finds it at least 1% faster. The prompt-width row-block
/// route (the o_proj rotation reading the attention's query blocks in place)
/// keeps the slice updates. `MLXFAST_INPLACE_KV_APPEND=0` keeps them everywhere.
enum Qwen35AttentionPreworkKV {
    // grid (TPG * (HQ + 2 * HK), L, 1), threadgroup (TPG, 1, 1), TPG = D / 4.
    // q [1, L, HQ, D], k and v [1, L, HK, D] (any strides), wq/wk [D] FP32,
    // offs [1] int32, epsq/epsk/scale/lbase FP32, axis (= D) uint32, wr int32
    // (first written row), prev [1] int32 (the previous write's fence, not
    // read: it orders this launch after that one); kc [1, HK, cap, D] FP32 and
    // vc [1, HK, cap, D] (v's dtype) are written in place. Outputs qo
    // [1, HQ, L, D] FP32 and fence [1] int32 (orders the readers).
    private static let source = """
        constexpr int NR = 4;
        constexpr int HALF = RD / 2;
        const uint lid = thread_position_in_threadgroup.x;
        const uint hh = threadgroup_position_in_grid.x;
        const uint t = threadgroup_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int Ln = int(q_shape[1]);
        const int64_t row = int64_t(wr) + int64_t(t);

        threadgroup float local_sums[32];
        threadgroup float local_inv[1];
        threadgroup float rot[RD];

        if (hh == 0 && t == 0 && lid == 0) {
          fence[0] = 1;
        }
        if (hh >= uint(HQ + HK)) {
          // Value head hh - HQ - HK, row t: a copy into row wr + t.
          const int64_t h = int64_t(hh - uint(HQ + HK));
          const int64_t vb = int64_t(t) * v_strides[1] + h * v_strides[2];
          const int64_t cb = h * vc_strides[1] + row * vc_strides[2];
          for (int i = 0; i < NR; i++) {
            const int64_t c = int64_t(lid * NR + uint(i));
            vc[cb + c * vc_strides[3]] = v[vb + c * v_strides[3]];
          }
          return;
        }
        const bool isq = hh < uint(HQ);
        const uint h = isq ? hh : hh - uint(HQ);

        // rms_single_row: lane lid holds channels NR*lid .. NR*lid+NR-1.
        const int64_t base = isq
            ? int64_t(t) * q_strides[1] + int64_t(h) * q_strides[2]
            : int64_t(t) * k_strides[1] + int64_t(h) * k_strides[2];
        const int64_t cs = isq ? q_strides[3] : k_strides[3];
        auto src = isq ? q : k;
        float acc = 0;
        float thread_x[NR];
        for (int i = 0; i < NR; i++) {
          thread_x[i] = static_cast<float>(src[base + int64_t(lid * NR + i) * cs]);
          acc += thread_x[i] * thread_x[i];
        }
        acc = simd_sum(acc);
        if (sg == 0) {
          local_sums[lane] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
          local_sums[sg] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          acc = simd_sum(local_sums[lane]);
          if (lane == 0) {
            const float eps = isq ? epsq : epsk;
            local_inv[0] = metal::precise::rsqrt(acc / axis + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto w = isq ? wq : wk;
        // q heads to qo [1, HQ, L, D]; k heads to the cache row wr + t.
        device float* dst = isq
            ? qo + (size_t(h) * size_t(Ln) + size_t(t)) * size_t(D)
            : kc + (int64_t(h) * kc_strides[1] + row * kc_strides[2]);
        const int64_t ds = isq ? 1 : kc_strides[3];
        const float inv = local_inv[0];
        for (int i = 0; i < NR; i++) {
          const uint c = lid * NR + uint(i);
          const float n = w[c] * static_cast<float>(thread_x[i] * inv);
          if (c < uint(RD)) {
            rot[c] = n;
          } else {
            dst[int64_t(c) * ds] = n;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // rope on channels [0, RD): pair (j, j + HALF), as MLX's rope kernel.
        if (lid < uint(HALF)) {
          const int off = offs[0];
          float d = static_cast<float>(lid) / static_cast<float>(HALF);
          float inv_freq = metal::exp2(-d * lbase);
          float L = scale * static_cast<float>(t + off);
          float theta = L * inv_freq;
          float costheta = metal::fast::cos(theta);
          float sintheta = metal::fast::sin(theta);
          float x1 = rot[lid];
          float x2 = rot[lid + HALF];
          float rx1 = x1 * costheta - x2 * sintheta;
          float rx2 = x1 * sintheta + x2 * costheta;
          dst[int64_t(lid) * ds] = rx1;
          dst[int64_t(lid + HALF) * ds] = rx2;
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_attn_prework_kv",
        inputNames: [
            "q", "k", "v", "wq", "wk", "offs", "epsq", "epsk", "axis", "lbase", "scale", "wr",
            "prev", "kc", "vc",
        ],
        outputNames: ["qo", "fence"],
        source: Qwen35IO32.narrow(source, count: 4, "bonsai_attn_prework_kv"),
        ensureRowContiguous: false,
        mutableInputs: ["kc", "vc"])

    /// `BONSAI_ATTN_PREWORK_IO32=0` keeps the 64-bit text.
    static let io32Enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_ATTN_PREWORK_IO32"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // `source` with its 22 int64_t, 4 size_t and 15 stride reads as uint; nil
    // if a count moved. It runs where each stride (unsigned) is under a cap
    // that keeps every offset below 2^31 for L <= 16 and rows below 2^17 (the
    // launch checks), else `source` runs (lazy views: strides known at eval).
    private static let fastSource: String? = {
        guard let reads = try? NSRegularExpression(pattern: #"(\w+_strides\[\d\])"#) else { return nil }
        let all = NSRange(source.startIndex..., in: source)
        guard source.components(separatedBy: "int64_t").count == 23,
            source.components(separatedBy: "size_t").count == 5,
            reads.numberOfMatches(in: source, range: all) == 15
        else { return nil }
        let narrow = reads.stringByReplacingMatches(
            in: source, range: all, withTemplate: "uint($1)"
        ).replacingOccurrences(of: "int64_t", with: "uint").replacingOccurrences(of: "size_t", with: "uint")
        let caps = [
            ("q", "1ul << 26", "(1ul << 29) / HQ"), ("k", "1ul << 26", "(1ul << 29) / HK"),
            ("v", "1ul << 26", "(1ul << 29) / HK"), ("kc", "(1ul << 29) / HK", "1ul << 13"),
            ("vc", "(1ul << 29) / HK", "1ul << 13"),
        ].map {
            "(ulong(\($0)_strides[1]) <= (\($1))) & (ulong(\($0)_strides[2]) <= (\($2)))"
                + " & (ulong(\($0)_strides[3]) <= (1ul << 29) / D)"
        }
        return "if (\(caps.joined(separator: " & "))) {\n\(narrow)\n} else {\n\(source)\n}\n"
    }()

    private static let fastKernel: MLXFast.MLXFastKernel? = fastSource.map {
        MLXFast.metalKernel(
            name: "bonsai_attn_prework_kv32",
            inputNames: [
                "q", "k", "v", "wq", "wk", "offs", "epsq", "epsk", "axis", "lbase", "scale", "wr",
                "prev", "kc", "vc",
            ],
            outputNames: ["qo", "fence"], source: $0, ensureRowContiguous: false,
            mutableInputs: ["kc", "vc"])
    }

    private struct Geometry: Hashable {
        let hq: Int, hk: Int, d: Int, rd: Int, dtype: String, vdtype: String
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [Geometry: Bool] = [:]
    /// The FP32 geometry (HQ, HK, D, RD) whose twin passed its self-test.
    nonisolated(unsafe) private static var fastPassed: (Int, Int, Int, Int)?

    /// The attention output `[1, HQ, L, D]` of `cache.updateAndAttend` over
    /// the prework's queries and keys and `v` transposed, with the append
    /// written by the prework launch; nil where that does not apply (the
    /// caller then runs the prework and `updateAndAttend`).
    static func attend(
        q: MLXArray, k: MLXArray, v: MLXArray, qNorm: RMSNorm, kNorm: RMSNorm,
        ropeDims: Int, ropeBase: Float, cache: any CBv2AttendingLayerCache, scale: Float
    ) -> MLXArray? {
        guard CBv2InPlaceKVAppend.enabled, Qwen35AttentionPrework.enabled,
            ObjectIdentifier(type(of: qNorm)) == ObjectIdentifier(RMSNorm.self),
            ObjectIdentifier(type(of: kNorm)) == ObjectIdentifier(RMSNorm.self),
            let layer = cache as? CBv2LayerCache,
            q.ndim == 4, k.ndim == 4, v.ndim == 4,
            Qwen35AttentionPrework.verified(
                hq: q.dim(2), hk: k.dim(2), d: q.dim(3), rd: ropeDims, dtype: q.dtype),
            lock.withLock({
                verdicts[
                    Geometry(
                        hq: q.dim(2), hk: k.dim(2), d: q.dim(3), rd: ropeDims,
                        dtype: "\(q.dtype)", vdtype: "\(v.dtype)")] ?? false
            })
        else { return nil }
        return attendUnchecked(
            q: q, k: k, v: v, wq: qNorm.weight, wk: kNorm.weight, epsQ: qNorm.eps,
            epsK: kNorm.eps, ropeDims: ropeDims, ropeBase: ropeBase, layer: layer, scale: scale)
    }

    private static func attendUnchecked(
        q: MLXArray, k: MLXArray, v: MLXArray, wq: MLXArray, wk: MLXArray,
        epsQ: Float, epsK: Float, ropeDims: Int, ropeBase: Float,
        layer: CBv2LayerCache, scale: Float
    ) -> MLXArray? {
        let L = q.dim(1)
        let HQ = q.dim(2)
        let HK = k.dim(2)
        let D = q.dim(3)
        let offsets = layer.positionOffsets
        guard q.dim(0) == 1, k.dim(0) == 1, k.dim(1) == L, k.dim(3) == D, v.shape == k.shape,
            q.dtype == k.dtype, [DType.float32, .float16, .bfloat16].contains(q.dtype),
            wq.dtype == .float32, wk.dtype == .float32, wq.shape == [D], wk.shape == [D],
            offsets.dtype == .int32, offsets.ndim <= 1, offsets.size == 1,
            L > 0, L < 65536,
            let destination = layer.inPlaceAppendDestination(
                count: L, keyDType: .float32, valueDType: v.dtype),
            destination.keys.dim(3) == D, destination.values.dim(3) == D
        else { return nil }
        let offs = offsets.ndim == 1 ? offsets : offsets.reshaped([1])
        // No strings here: the in-place trial times this host path too.
        let fast =
            L <= 16 && destination.row >= 0 && destination.row <= (1 << 17) - L && HQ * D <= 1 << 26
            && q.dtype == .float32 && v.dtype == .float32
            && lock.withLock({ fastPassed.map { $0 == (HQ, HK, D, ropeDims) } ?? false })
        let outputs = encode(
            (fast ? fastKernel : nil) ?? kernel, q: q, k: k, v: v, wq: wq, wk: wk, offs: offs,
            epsQ: epsQ, epsK: epsK, ropeDims: ropeDims, ropeBase: ropeBase, row: destination.row,
            prev: destination.previous ?? MLXArray([Int32(0)]), kc: destination.keys,
            vc: destination.values)
        return layer.attendAfterInPlaceAppend(queries: outputs[0], fence: outputs[1], scale: scale)
    }

    private static func encode(
        _ kern: MLXFast.MLXFastKernel, q: MLXArray, k: MLXArray, v: MLXArray, wq: MLXArray,
        wk: MLXArray, offs: MLXArray, epsQ: Float, epsK: Float, ropeDims: Int, ropeBase: Float,
        row: Int, prev: MLXArray, kc: MLXArray, vc: MLXArray
    ) -> [MLXArray] {
        let (L, HQ, HK, D) = (q.dim(1), q.dim(2), k.dim(2), q.dim(3))
        return kern(
            [
                q, k, v, wq, wk, offs, MLXArray(epsQ), MLXArray(epsK), MLXArray(UInt32(D)),
                MLXArray(log2(ropeBase)), MLXArray(Float(1)), MLXArray(Int32(row)), prev, kc, vc,
            ],
            template: [("D", D), ("RD", ropeDims), ("HQ", HQ), ("HK", HK)],
            grid: ((D / 4) * (HQ + 2 * HK), L, 1), threadGroup: (D / 4, 1, 1),
            outputShapes: [[1, HQ, L, D], [1]],
            outputDTypes: [.float32, .int32])
    }

    /// The twin against the 64-bit text at 16, 9, 3 and 1 rows, every bit of
    /// qo and both caches: column views of one stack (row 1500), with edge
    /// values (the last rows), reversed (the 64-bit branch). A mismatch or an
    /// MLX error keeps the 64-bit text.
    private static func fastSelfTest(_ geo: Geometry, ropeBase: Float, epsQ: Float, epsK: Float) {
        guard io32Enabled, let fastKernel else { return }
        var (values, bad, detail) = (0, 0, "")
        let keys = MLXRandom.split(key: MLXRandom.key(0x6b76_3332), into: 5)
        let w = keys[..<2].map { 1 + 0.25 * MLXRandom.normal([geo.d], key: $0) }
        do {
            try withError { error in
                for rows in [16, 9, 3, 1] {
                    for form in 0 ..< 3 {
                        var o = operands(geo, rows: rows, key: keys[2 + form], edge: form == 1)
                        if form == 2 {
                            o = (o.q[.ellipsis, .stride(by: -1)], o.k[.ellipsis, .stride(by: -1)],
                                o.v[.ellipsis, .stride(by: -1)])
                        }
                        let row = form == 1 ? 2048 - rows : 1500
                        let outs = [kernel, fastKernel].map { kern -> [MLXArray] in
                            let kv = (0 ..< 2).map { _ in MLXArray.zeros([1, geo.hk, 2048, geo.d]) }
                            eval(kv)
                            let r = encode(
                                kern, q: o.q, k: o.k, v: o.v, wq: w[0], wk: w[1], offs: MLXArray([Int32(row)]),
                                epsQ: epsQ, epsK: epsK, ropeDims: geo.rd, ropeBase: ropeBase, row: row,
                                prev: MLXArray([Int32(0)]), kc: kv[0], vc: kv[1])
                            eval(r)
                            return [r[0]] + kv
                        }
                        let c = stacked(
                            zip(outs[0], outs[1]).map {
                                ($0.view(dtype: .uint32) .!= $1.view(dtype: .uint32)).asType(.int32).sum()
                            }
                        ).sum()
                        eval(c)
                        try error.check()
                        values += outs[0].reduce(0) { $0 + $1.size }
                        bad += Int(c.item(Int32.self))
                    }
                }
            }
        } catch {
            (bad, detail) = (max(bad, 1), " (\(error))")
        }
        lock.withLock { fastPassed = bad == 0 ? (geo.hq, geo.hk, geo.d, geo.rd) : nil }
        Memory.clearCache()
        FileHandle.standardError.write(
            Data(
                ("qwen35 in-place KV append, 32-bit offsets: self-test \(bad == 0 ? "passed" : "FAILED") "
                    + "(12 cases, \(values) values, \(bad) mismatches\(detail))\n").utf8))
    }

    /// Compile, self-test and time the in-place form for one attention
    /// geometry (FP32 activations, as the verify's prework runs), once per
    /// process, at model construction. One stderr line.
    static func prepare(
        hq: Int, hk: Int, d: Int, ropeDims rd: Int, ropeBase: Float, epsQ: Float, epsK: Float
    ) {
        guard CBv2InPlaceKVAppend.enabled, Qwen35AttentionPrework.enabled, d % 128 == 0,
            d <= 4096, rd > 0, rd % 4 == 0, rd <= d, rd / 2 <= d / 4, hq > 0, hk > 0
        else { return }
        let geometry = Geometry(
            hq: hq, hk: hk, d: d, rd: rd, dtype: "\(DType.float32)", vdtype: "\(DType.float32)")
        if lock.withLock({ verdicts[geometry] != nil }) { return }
        // The reference is the prework itself; where it is not used, neither is this.
        guard Qwen35AttentionPrework.verified(hq: hq, hk: hk, d: d, rd: rd, dtype: .float32)
        else { return }
        fastSelfTest(geometry, ropeBase: ropeBase, epsQ: epsQ, epsK: epsK)
        let checked = selfTest(geometry, ropeBase: ropeBase, epsQ: epsQ, epsK: epsK)
        var use = checked.passed
        var timing = ""
        if use, let trial = trial(geometry, ropeBase: ropeBase, epsQ: epsQ, epsK: epsK) {
            // Adopted only when at least 1% faster (a tie keeps the slice updates).
            use = trial.inPlace <= trial.slices * 0.99
            timing = String(
                format: "; trial per 16-layer step: slice updates %.1f us, in place %.1f us",
                trial.slices, trial.inPlace)
        } else if use {
            use = false
            timing = "; trial failed"
        }
        lock.withLock { verdicts[geometry] = use }
        FileHandle.standardError.write(
            ("qwen35 in-place KV append: self-test \(checked.passed ? "passed" : "FAILED") "
                + "(\(checked.detail))\(timing); "
                + (use ? "in place\n" : "slice updates kept\n")).data(using: .utf8)!)
    }

    /// One attention layer's operands as the verify produces them: q, k and v
    /// column views of one stacked projection, wide magnitude spread.
    /// `edge`: zero, 1e30, +-inf and -0.0 blocks in every 4 D columns.
    private static func operands(_ geo: Geometry, rows: Int, key: MLXArray, edge: Bool = false)
        -> (q: MLXArray, k: MLXArray, v: MLXArray)
    {
        let (width, d) = (geo.hq * 2 * geo.d + 2 * geo.hk * geo.d, geo.d)
        let pair = MLXRandom.split(key: key)
        var wide = MLXRandom.normal([1, rows, width], key: pair.0)
            * exp(MLXRandom.normal([1, rows, width], key: pair.1))
        let i = MLXArray(0 ..< width) % (4 * d)
        for (lo, hi, v) in [
            (0, d, Float(0)), (d, d + 64, Float(1e30)), (2 * d + 5, 2 * d + 6, Float.infinity),
            (3 * d + 9, 3 * d + 10, -Float.infinity), (3 * d + 128, 4 * d, Float(-0.0)),
        ] where edge {
            wide = which((i .>= lo) .&& (i .< hi), MLXArray(v), wide)
        }
        let parts = MLX.split(
            wide, indices: [geo.hq * 2 * geo.d, geo.hq * 2 * geo.d + geo.hk * geo.d], axis: -1)
        return (
            parts[0].reshaped(1, rows, geo.hq, -1).split(parts: 2, axis: -1)[0],
            parts[1].reshaped(1, rows, geo.hk, -1),
            parts[2].reshaped(1, rows, geo.hk, -1)
        )
    }

    /// Two caches holding the same prefix rows (separate storage).
    private static func twinCaches(_ geo: Geometry, prefix: Int, key: MLXArray)
        -> (CBv2LayerCache, CBv2LayerCache)
    {
        let pair = MLXRandom.split(key: key)
        let pk = MLXRandom.normal([1, geo.hk, prefix, geo.d], key: pair.0)
        let pv = MLXRandom.normal([1, geo.hk, prefix, geo.d], key: pair.1)
        let kind = CBv2LayerKind(
            attention: .full, headDim: geo.d, kvHeads: geo.hk, queryHeads: geo.hq)
        func make() -> CBv2LayerCache {
            let row = CBv2FullSequenceKV(
                promptLength: prefix, maxLength: 1 << 16, kvHeads: geo.hk, headDim: geo.d)
            _ = row.update(keys: pk, values: pv)
            return CBv2LayerCache(layerIndex: 0, kind: kind, rows: [row])
        }
        return (make(), make())
    }

    private static func slicePath(
        _ o: (q: MLXArray, k: MLXArray, v: MLXArray), wq: MLXArray, wk: MLXArray,
        geo: Geometry, ropeBase: Float, epsQ: Float, epsK: Float, layer: CBv2LayerCache,
        scale: Float
    ) -> MLXArray? {
        guard
            let (qo, ko) = Qwen35AttentionPrework.runUnchecked(
                q: o.q, k: o.k, wq: wq, wk: wk, epsQ: epsQ, epsK: epsK,
                offsets: layer.positionOffsets, ropeDims: geo.rd, ropeBase: ropeBase)
        else { return nil }
        return layer.updateAndAttend(
            queries: qo, keys: ko, values: o.v.transposed(0, 2, 1, 3), scale: scale, sinks: nil)
    }

    private static func rollback(_ layer: CBv2LayerCache, _ n: Int) {
        let rows = layer.rows
        for row in rows { row.rollback(n) }
        layer.setRows(rows)
    }

    private static func selfTest(
        _ geo: Geometry, ropeBase: Float, epsQ: Float, epsK: Float
    ) -> (passed: Bool, detail: String) {
        let keys = MLXRandom.split(key: MLXRandom.key(0x6b76_6170), into: 16)
        let wq = 1 + 0.25 * MLXRandom.normal([geo.d], key: keys[0])
        let wk = 1 + 0.25 * MLXRandom.normal([geo.d], key: keys[1])
        let scale = pow(Float(geo.d), -0.5)
        // (prefix rows, steps): a step appends `rows`, or rolls back `-rows`.
        // The second scenario grows the storage (capacity prefix + 256).
        // 1000 + n appends n rows through the slice updates on both caches
        // (an update after in-place writes); 2000 + n appends n rows and
        // leaves them unevaluated, so the next append joins the same graph.
        let scenarios: [(prefix: Int, steps: [Int])] = [
            (611, [16, -7, 16, 1, 1016, 16]), (40, [16, 300]), (100, [2016, 16]),
        ]
        var values = 0
        var steps = 0
        var mismatches = 0
        do {
            try withError { error in
                var differ: [MLXArray] = []
                func compare(_ a: MLXArray, _ b: MLXArray) throws {
                    guard a.shape == b.shape, a.dtype == b.dtype, a.dtype == .float32 else {
                        throw Qwen35InPlaceKVError.mismatch("shape or dtype")
                    }
                    differ.append(
                        (a.view(dtype: .uint32) .!= b.view(dtype: .uint32)).asType(.int32).sum())
                    values += a.size
                }
                var keyIndex = 2
                for scenario in scenarios {
                    let (slices, inPlace) = twinCaches(
                        geo, prefix: scenario.prefix, key: keys[keyIndex])
                    keyIndex += 1
                    for rows in scenario.steps {
                        if rows < 0 {
                            rollback(slices, -rows)
                            rollback(inPlace, -rows)
                            continue
                        }
                        let bothSlices = rows > 1000 && rows < 2000
                        let lazy = rows > 2000
                        let o = operands(
                            geo, rows: rows % 1000,
                            key: keys[keyIndex % keys.count])
                        keyIndex += 1
                        guard
                            let reference = slicePath(
                                o, wq: wq, wk: wk, geo: geo, ropeBase: ropeBase, epsQ: epsQ,
                                epsK: epsK, layer: slices, scale: scale),
                            let written = bothSlices
                                ? slicePath(
                                    o, wq: wq, wk: wk, geo: geo, ropeBase: ropeBase, epsQ: epsQ,
                                    epsK: epsK, layer: inPlace, scale: scale)
                                : attendUnchecked(
                                    q: o.q, k: o.k, v: o.v, wq: wq, wk: wk, epsQ: epsQ,
                                    epsK: epsK, ropeDims: geo.rd, ropeBase: ropeBase,
                                    layer: inPlace, scale: scale)
                        else { throw Qwen35InPlaceKVError.mismatch("no launch at \(rows) rows") }
                        try compare(reference, written)
                        steps += 1
                        if lazy { continue }
                        let a = slices.rows[0].snapshot()
                        let b = inPlace.rows[0].snapshot()
                        guard a.offset == b.offset else {
                            throw Qwen35InPlaceKVError.mismatch("offsets \(a.offset) \(b.offset)")
                        }
                        try compare(a.keys, b.keys)
                        try compare(a.values, b.values)
                        differ.append(
                            (slices.positionOffsets .!= inPlace.positionOffsets).asType(.int32)
                                .sum())
                        let count = stacked(differ).sum()
                        eval(count)
                        try error.check()
                        mismatches += Int(count.item(Int32.self))
                        differ.removeAll()
                    }
                }
            }
        } catch {
            return (false, "\(error)")
        }
        return (
            mismatches == 0 && steps == 9,
            "\(steps) appends, \(values) values bitwise, \(mismatches) mismatches")
    }

    /// Median time of one 16-layer verify step (16 rows over ~600 cached
    /// rows: prework, append and attention per layer), slice updates vs in
    /// place, interleaved; nil on any MLX error.
    private static func trial(
        _ geo: Geometry, ropeBase: Float, epsQ: Float, epsK: Float
    ) -> (slices: Double, inPlace: Double)? {
        let keys = MLXRandom.split(key: MLXRandom.key(0x7472_6961), into: 20)
        let wq = 1 + 0.25 * MLXRandom.normal([geo.d], key: keys[0])
        let wk = 1 + 0.25 * MLXRandom.normal([geo.d], key: keys[1])
        let scale = pow(Float(geo.d), -0.5)
        let layers = 16
        let rows = 16
        do {
            return try withError { error in
                var caches: [(CBv2LayerCache, CBv2LayerCache)] = []
                var inputs: [(q: MLXArray, k: MLXArray, v: MLXArray)] = []
                for i in 0 ..< layers {
                    caches.append(twinCaches(geo, prefix: 600, key: keys[2 + i % 18]))
                    let o = operands(geo, rows: rows, key: keys[(3 + i) % 20])
                    eval(o.q, o.k, o.v)
                    inputs.append(o)
                }
                eval(caches.flatMap { $0.0.innerState() + $0.1.innerState() })
                try error.check()
                var slices: [Double] = []
                var inPlace: [Double] = []
                for rep in 0 ..< 13 {
                    for variant in (rep % 2 == 0 ? [0, 1] : [1, 0]) {
                        let start = DispatchTime.now().uptimeNanoseconds
                        var outputs: [MLXArray] = []
                        for (i, pair) in caches.enumerated() {
                            let layer = variant == 0 ? pair.0 : pair.1
                            let o = inputs[i]
                            let y =
                                variant == 0
                                ? slicePath(
                                    o, wq: wq, wk: wk, geo: geo, ropeBase: ropeBase, epsQ: epsQ,
                                    epsK: epsK, layer: layer, scale: scale)
                                : attendUnchecked(
                                    q: o.q, k: o.k, v: o.v, wq: wq, wk: wk, epsQ: epsQ,
                                    epsK: epsK, ropeDims: geo.rd, ropeBase: ropeBase,
                                    layer: layer, scale: scale)
                            guard let y else { throw Qwen35InPlaceKVError.mismatch("no launch") }
                            outputs.append(y)
                            outputs.append(contentsOf: layer.innerState())
                        }
                        eval(outputs)
                        let micros = Double(DispatchTime.now().uptimeNanoseconds - start) / 1000
                        try error.check()
                        if rep > 0, variant == 0 { slices.append(micros) }
                        if rep > 0, variant == 1 { inPlace.append(micros) }
                        for pair in caches { rollback(variant == 0 ? pair.0 : pair.1, rows) }
                    }
                }
                slices.sort()
                inPlace.sort()
                return (slices[slices.count / 2], inPlace[inPlace.count / 2])
            }
        } catch {
            return nil
        }
    }
}

enum Qwen35InPlaceKVError: Error, CustomStringConvertible {
    case mismatch(String)
    var description: String {
        switch self {
        case .mismatch(let what): return "mismatch: \(what)"
        }
    }
}
