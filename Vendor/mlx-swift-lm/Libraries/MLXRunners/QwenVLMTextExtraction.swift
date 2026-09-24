// Copyright © 2026 Eigen Labs.
//
// MLXRunners — weight-sharing Qwen 3.5 text-target extraction from a loaded
// `MLXVLM.Qwen35` wrapper.
//
// Ported from d-inference's `EngineV2VLMTextExtraction`, which this replaces:
// the contract puts serving-model derivation — VLM text-tower extraction
// included — inside `Runner.adopt`, so the provider no longer carries it.
//
// Gemma 4 is deliberately absent: its MLXVLM wrapper directly OWNS the
// `Gemma4TextModel` CBv2 serves, so that family needs a property read, not
// an extraction. Both dense and MoE `MLXVLM.Qwen35` wrappers need a separate
// MLXLLM target over the same immutable weight arrays:
//
//   1. decode the checkpoint with the matching MLXLLM config type and
//      construct a lazy skeleton (nothing is materialized — MLXArray init is
//      lazy until eval);
//   2. re-apply the checkpoint's quantization STRUCTURE to the skeleton the
//      exact way `loadWeights` did for the wrapper (scales-presence gate plus
//      the same per-layer table from `BaseConfiguration`, whose keys live in
//      the checkpoint's `language_model.`-prefixed key space);
//   3. retain only the wrapper's live `language_model.*` target tree, run the
//      target sanitizer, and `update(parameters:verify:[.all])` — missing,
//      extra and mis-shaped keys all THROW;
//   4. run a tiny forward through BOTH the wrapper's text path and the
//      extracted model and require cross-containment of each side's greedy
//      argmax in the other's top-5, plus a bounded max |Δlogit|.
//
// The result is a SEPARATE module instance (its own norms, rope and layer
// objects) sharing only the immutable parameter arrays — zero extra weight
// memory, and no module-level mutable state shared with the wrapper.
//
// NO TENSOR IS READ FROM DISK. `adopt`'s contract holds: the arrays come
// from the module the caller already has resident, and the only file this
// touches is `config.json`.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM

/// Failure modes of Qwen VLM text-model extraction.
public enum QwenVLMTextExtractionError: Error, CustomStringConvertible, Equatable {
    /// The loaded module is not a VLM wrapper this extraction understands.
    case unsupportedWrapper(String)
    /// `config.json` was unreadable or had no decodable text configuration.
    case invalidConfig(String)
    /// The checkpoint has quantized weights but no `quantization` block in
    /// `config.json` to derive the skeleton's quantization structure from.
    case missingQuantizationConfig
    /// The forward parity gate failed: the extracted text model disagrees
    /// with the wrapper's own text path on the same weights.
    case parityMismatch(String)

    public var description: String {
        switch self {
        case .unsupportedWrapper(let type):
            return "qwen vlm extraction: unsupported VLM wrapper \(type)"
        case .invalidConfig(let detail):
            return "qwen vlm extraction: config.json unusable (\(detail))"
        case .missingQuantizationConfig:
            return "qwen vlm extraction: quantized weights but no quantization block "
                + "in config.json"
        case .parityMismatch(let detail):
            return "qwen vlm extraction: wrapper/extracted forward parity failed (\(detail))"
        }
    }
}

/// Weight-sharing extraction of the CBv2-adapted MLXLLM Qwen target.
public enum QwenVLMTextExtraction {

    /// Env kill switch for the load-time forward parity gate ("0"/"false"/
    /// "no"/"off" disables). Default ON — the check is one tiny prefill and
    /// is the backstop against silent architecture drift between MLXVLM's
    /// inline text model and MLXLLM's.
    public static let parityCheckFlag = "DARKBLOOM_ENGINE_V2_VLM_PARITY_CHECK"

    /// Checkpoint key-space prefix of the wrapper's language model.
    private static let languageModelPrefix = "language_model."

    /// Top-k window for the bidirectional argmax-containment check.
    static let parityTopK = 5

    public enum Family: String, Sendable {
        case qwen35Dense
        case qwen35MoE
    }

    /// One extraction: the CBv2-adapted text model (sharing the wrapper's
    /// weight arrays) plus the parity probe's max |Δlogit| (nil when the
    /// gate was disabled).
    public struct Extraction {
        public let servingModel: Qwen35Model
        public let family: Family
        public let parityMaxAbsLogitDiff: Float?
    }

    // MARK: - Identity cache

    /// One extracted target per wrapper INSTANCE.
    ///
    /// Not an optimization — a correctness requirement. A drafter binds to
    /// its target by `ObjectIdentifier` (`CBv2MTPDrafter.mtpTargetIdentity`),
    /// and the engine gates speculation on that identity matching the model
    /// it steps. Extracting twice for one wrapper would hand the drafter and
    /// the engine two different targets over the same arrays, and MTP would
    /// go quietly inactive with nothing to see. The entry is weak, so it
    /// lives exactly as long as the wrapper does.
    private final class WeakTarget {
        weak var value: Qwen35Model?
        init(_ value: Qwen35Model) { self.value = value }
    }
    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var cache: [ObjectIdentifier: WeakTarget] = [:]

    /// The MLXLLM target for `model`, extracting it once per wrapper.
    ///
    /// A module that is ALREADY an MLXLLM target passes through: the LLM
    /// factory hands those back directly and there is nothing to extract.
    public static func target(
        for model: any LanguageModel,
        directory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Qwen35Model {
        if let target = model as? Qwen35Model { return target }
        guard let wrapper = model as? MLXVLM.Qwen35 else {
            throw QwenVLMTextExtractionError.unsupportedWrapper(
                String(describing: type(of: model)))
        }

        let key = ObjectIdentifier(wrapper)
        cacheLock.lock()
        if let cached = cache[key]?.value {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        let extraction = try extract(
            wrapper: wrapper, directory: directory, environment: environment)

        cacheLock.lock()
        // Another thread may have won the race; keep whichever target is
        // already published so identity stays single-valued.
        if let cached = cache[key]?.value {
            cacheLock.unlock()
            return cached
        }
        cache[key] = WeakTarget(extraction.servingModel)
        cache = cache.filter { $0.value.value != nil }
        cacheLock.unlock()
        return extraction.servingModel
    }

    // MARK: - Extraction

    public static func extract(
        wrapper: MLXVLM.Qwen35,
        directory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Extraction {
        let configURL = directory.appendingPathComponent("config.json")
        let configData: Data
        do {
            configData = try Data(contentsOf: configURL)
        } catch {
            throw QwenVLMTextExtractionError.invalidConfig(
                "read \(configURL.path): \(error)")
        }
        // Quantization table: `BaseConfiguration` holds the checkpoint-wide
        // default plus the per-layer overrides, keyed in the checkpoint's
        // `language_model.`-prefixed key space.
        let baseConfig = try? JSONDecoder.json5().decode(
            BaseConfiguration.self, from: configData)

        // The combined artifact may declare and carry inline `mtp.*` tensors.
        // This module is the serving TARGET only: force the MLXLLM skeleton's
        // optional inline head off independently of the process-global MTP
        // loader flag. The assistant is loaded separately.
        let targetConfig = try decodeConfiguration(configData: configData)
        let family: Family
        let skeleton: Qwen35Model
        if wrapper is MLXVLM.Qwen35MoE {
            family = .qwen35MoE
            skeleton = Qwen35MoEModel(targetConfig)
        } else {
            family = .qwen35Dense
            skeleton = Qwen35Model(targetConfig)
        }

        let targetWeights = reKeyedTargetWeights(
            flattenedWeights: wrapper.parameters().flattened(), sanitizer: skeleton)
        try applyQuantizationStructure(
            skeleton: skeleton,
            weights: targetWeights,
            perLayerQuantization: baseConfig?.perLayerQuantization)
        try skeleton.update(
            parameters: ModuleParameters.unflattened(targetWeights), verify: [.all])

        var parityDiff: Float?
        if parityCheckEnabled(environment: environment) {
            defer {
                MLX.Stream().synchronize()
                MLX.Memory.clearCache()
            }
            parityDiff = try assertForwardParity(
                wrapper: wrapper,
                extracted: skeleton,
                vocabSize: skeleton.vocabularySize)
        }
        return Extraction(
            servingModel: skeleton, family: family, parityMaxAbsLogitDiff: parityDiff)
    }

    /// Decode the top-level Qwen target configuration using MLXLLM's type,
    /// after disabling the optional inline MTP module in a COPIED JSON
    /// object. The checkpoint and the loaded wrapper are never mutated.
    static func decodeConfiguration(configData: Data) throws -> MLXLLM.Qwen35Configuration {
        var root = try configurationObject(configData: configData)
        if var textConfig = root["text_config"] as? [String: Any] {
            textConfig["mtp_num_hidden_layers"] = 0
            root["text_config"] = textConfig
        } else {
            root["mtp_num_hidden_layers"] = 0
        }
        do {
            let targetData = try JSONSerialization.data(withJSONObject: root)
            return try JSONDecoder.json5().decode(
                MLXLLM.Qwen35Configuration.self, from: targetData)
        } catch {
            throw QwenVLMTextExtractionError.invalidConfig(
                "decode Qwen target config: \(error)")
        }
    }

    private static func configurationObject(configData: Data) throws -> [String: Any] {
        do {
            guard
                let parsed = try JSONSerialization.jsonObject(with: configData)
                    as? [String: Any]
            else {
                throw QwenVLMTextExtractionError.invalidConfig("config.json is not an object")
            }
            return parsed
        } catch let error as QwenVLMTextExtractionError {
            throw error
        } catch {
            throw QwenVLMTextExtractionError.invalidConfig("parse config.json: \(error)")
        }
    }

    /// Select only the live target arrays from the wrapper's parameter tree.
    /// `mtp.*` is top-level and therefore excluded; vision parameters live
    /// under `vision_tower.*` and are excluded too. The MLXLLM target keeps
    /// the same `language_model.*` namespace, so no second prefix transform
    /// is needed. Dictionary assignment retains the same LAZY MLXArray
    /// handles — this is where "shares arrays, copies nothing" happens.
    static func reKeyedTargetWeights(
        flattenedWeights: [(String, MLXArray)], sanitizer: Qwen35Model
    ) -> [String: MLXArray] {
        var targetWeights: [String: MLXArray] = [:]
        targetWeights.reserveCapacity(flattenedWeights.count)
        for (key, value) in flattenedWeights where key.hasPrefix(languageModelPrefix) {
            targetWeights[key] = value
        }
        return sanitizer.sanitize(weights: targetWeights)
    }

    /// Mirror `loadWeights`' quantization pass onto the skeleton: a module is
    /// quantized iff the (re-keyed, live) weights carry `<path>.scales`, with
    /// (groupSize, bits, mode) resolved from the checkpoint's per-layer table
    /// under the module's `language_model.`-prefixed checkpoint key — the
    /// exact lookup the wrapper's own load performed, so the two module trees
    /// can never disagree on quantization structure.
    static func applyQuantizationStructure(
        skeleton: some Module,
        weights: [String: MLXArray],
        perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    ) throws {
        let hasQuantizedWeights = weights.keys.contains { $0.hasSuffix(".scales") }
        guard hasQuantizedWeights else { return }
        guard let perLayerQuantization else {
            throw QwenVLMTextExtractionError.missingQuantizationConfig
        }
        quantize(model: skeleton) { path, _ in
            guard weights["\(path).scales"] != nil else { return nil }
            return perLayerQuantization.quantization(layer: path)?.asTuple
        }
    }

    // MARK: - Parity gate

    static func parityCheckEnabled(environment: [String: String]) -> Bool {
        guard
            let raw = environment[parityCheckFlag]?
                .trimmingCharacters(in: .whitespaces).lowercased(), !raw.isEmpty
        else { return true }
        return !["0", "false", "no", "off"].contains(raw)
    }

    /// Qwen text-only mRoPE uses three identical position axes, so the
    /// wrapper and the MLXLLM target should be substantially closer than the
    /// known Gemma implementation split. Keep the top-k catastrophe guard,
    /// and scale the magnitude bound to the wrapper's fixed-probe logits
    /// because Qwen has no final-logit softcap.
    private static func assertForwardParity(
        wrapper: MLXVLM.Qwen35,
        extracted: Qwen35Model,
        vocabSize: Int
    ) throws -> Float {
        let probeTokens = [1, 17, 29, 43, 61].map { min($0, max(0, vocabSize - 1)) }
        let inputs = MLXArray(probeTokens.map(Int32.init)).expandedDimensions(axis: 0)
        let wrapperLogits = wrapper(inputs, cache: nil).asType(.float32)
        let extractedLogits = extracted(inputs, cache: nil).asType(.float32)
        let wrapperMaxMagnitude = MLX.abs(wrapperLogits).max()
        eval(wrapperMaxMagnitude)
        let maxAbsDiffLimit = max(2, wrapperMaxMagnitude.item(Float.self) * 0.25)

        let k = parityTopK
        // Top-k token ids per position, [1, L, k] (unordered within k).
        let wrapperTopK = argPartition(-wrapperLogits, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        let extractedTopK = argPartition(-extractedLogits, kth: k - 1, axis: -1)[
            .ellipsis, ..<k]
        let wrapperArgmax = argMax(wrapperLogits, axis: -1)
        let extractedArgmax = argMax(extractedLogits, axis: -1)
        let maxAbsDiff = MLX.abs(wrapperLogits - extractedLogits).max()
        eval(wrapperTopK, extractedTopK, wrapperArgmax, extractedArgmax, maxAbsDiff)

        let wrapperIds = wrapperArgmax.asArray(Int32.self)
        let extractedIds = extractedArgmax.asArray(Int32.self)
        let wrapperTop = wrapperTopK.asArray(Int32.self)  // row-major [L × k]
        let extractedTop = extractedTopK.asArray(Int32.self)
        let diff = maxAbsDiff.item(Float.self)

        for position in 0 ..< probeTokens.count {
            let window = position * k ..< (position + 1) * k
            let wrapperWindow = Set(wrapperTop[window])
            let extractedWindow = Set(extractedTop[window])
            guard extractedWindow.contains(wrapperIds[position]),
                wrapperWindow.contains(extractedIds[position])
            else {
                throw QwenVLMTextExtractionError.parityMismatch(
                    "top-\(k) containment failed at position \(position): "
                        + "wrapper argmax \(wrapperIds[position]) vs extracted top-\(k) "
                        + "\(extractedWindow.sorted()); extracted argmax "
                        + "\(extractedIds[position]) vs wrapper top-\(k) "
                        + "\(wrapperWindow.sorted()); maxAbsLogitDiff=\(diff)")
            }
        }
        guard diff <= maxAbsDiffLimit else {
            throw QwenVLMTextExtractionError.parityMismatch(
                "max |Δlogit| \(diff) exceeds \(maxAbsDiffLimit) "
                    + "(argmax containment held — distributions drifted)")
        }
        return diff
    }
}
