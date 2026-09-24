import Foundation
import MLX
import MLXLMCommon

/// Prompt-output policy. Decode uses the ordinary GPTOSS forward. The
/// final-position default avoids vocabulary projection of discarded rows;
/// full/intermediate modes retain the original frontier projection geometry.
enum GPTOSSPrefillOutputPolicy: String, CaseIterable {
    case full
    case intermediate
    case last
    case lastLayer = "last-layer"

    static func resolve(_ raw: String?) -> Self {
        raw.flatMap(Self.init(rawValue:)) ?? .last
    }

    static let serving = resolve(
        ProcessInfo.processInfo.environment["DARKBLOOM_GPTOSS_PREFILL_OUTPUT"])
}

extension GPTOSSModel: CBv2LanguageModelPrefillForwardable {
    public func cbv2Prefill(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray?,
        cache: [KVCache]?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        prefillOutput(inputs, inputEmbedding: inputEmbedding, cache: cache,
                      requirement: requirement, policy: .serving)
    }

    /// Head-only policies preserve the full transformer trunk. The explicit
    /// last-layer arm additionally narrows the final query when supported.
    /// Cache inner state stays in the scheduler's normal evaluation set.
    func prefillOutput(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray?,
        cache: [KVCache]?,
        requirement: CBv2PrefillRequirement,
        policy: GPTOSSPrefillOutputPolicy
    ) -> MLXArray {
        let pruneLastLayer = policy == .lastLayer && requirement == .lastPositionLogits
        let hidden = model(inputs, cache: cache, inputEmbeddings: inputEmbedding,
                           lastLayerLastQuery: pruneLastLayer)
        switch requirement {
        case .evaluationOnly:
            if policy == .full { return lmHead(hidden)[0..., -1, 0..<1] }
            // Intermediate chunks need a nonempty dependency handle, not
            // vocabulary logits. Its values are deliberately unspecified.
            return hidden[0..., -1, 0..<1]
        case .lastPositionLogits:
            if policy == .last || policy == .lastLayer {
                return lmHead(hidden[0..., -1, 0...])
            }
            // Preserve the original QMM geometry and reduction order at the
            // frontier in the full and intermediate-only arms.
            return lmHead(hidden)[0..., -1, 0...]
        }
    }
    // Packed prefill remains the protocol's false default.
}
