// SteppableAdapterV2.swift
//
// Bridges WS-F's model v2 branches into the engine's `CBv2SteppableModel`
// seam. Gemma 4 and GPT-OSS detect CBv2 layer caches through their existing
// `callAsFunction(_:cache:)` entry points (the caches conform to the legacy
// `KVCache` protocol with trapping legacy methods), so the adapter is a
// thin cast-and-forward: no model file changes, no mask construction, no
// padding — the v2 branch inside the model owns the dispatch.
//
// Cache construction stays with `model.newCacheV2(makeLayerCache:)` (the
// single entry point — GPT-OSS primes its sinks-activation probe there);
// wrap the result in a `CBv2LayerCacheBank` for the engine.

import Foundation
import MLX

/// `CBv2SteppableModel` over any `LanguageModel` whose forward path
/// understands `CBv2AttendingLayerCache` (Gemma 4, GPT-OSS, test fixtures).
public final class CBv2SteppableLanguageModelAdapter: CBv2SteppableModel {

    private let model: any LanguageModel

    public init(_ model: any LanguageModel) {
        self.model = model
    }

    public func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        precondition(
            !(model is any CBv2RecurrentLanguageModelForwardable),
            "CBv2 recurrent model requires explicit request-owned recurrent state")
        return model(tokens, cache: asKVCaches(caches))
    }

    private func asKVCaches(_ caches: [CBv2AttendingLayerCache]) -> [KVCache] {
        caches.map { cache -> KVCache in
            guard let kv = cache as? KVCache else {
                fatalError(
                    "CBv2 layer cache \(type(of: cache)) must conform to KVCache to drive "
                        + "\(type(of: model)) through callAsFunction(_:cache:)")
            }
            return kv
        }
    }
}

extension CBv2SteppableLanguageModelAdapter: CBv2KeepMaskRequiringModel {
    public var cbv2RequiresKeepMask: Bool {
        (model as? any CBv2KeepMaskRequiringModel)?.cbv2RequiresKeepMask ?? false
    }
}

extension CBv2SteppableLanguageModelAdapter: CBv2PositionAxisProviding {
    public var cbv2PositionAxisCount: Int? {
        (model as? any CBv2PositionAxisProviding)?.cbv2PositionAxisCount
    }
}

extension CBv2SteppableLanguageModelAdapter: CBv2CompleteCheckpointKVTypeProviding {
    public var cbv2CompleteCheckpointKVDTypes: [DType]? {
        (model as? any CBv2CompleteCheckpointKVTypeProviding)?.cbv2CompleteCheckpointKVDTypes
    }
}

extension CBv2SteppableLanguageModelAdapter: CBv2PositionedSteppableModel {
    public var supportsPositionedForwarding: Bool {
        if model is any CBv2RecurrentLanguageModelForwardable {
            return model is any CBv2PositionedRecurrentLanguageModelForwardable
                && model is any CBv2PositionedRecurrentEmbeddingForwardable
        }
        return model is CBv2PositionedLanguageModelForwardable
            && model is CBv2PositionedEmbeddingForwardable
    }

    public func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        positionIds: MLXArray?
    ) -> MLXArray {
        guard let positioned = model as? CBv2PositionedLanguageModelForwardable else {
            preconditionFailure(
                "CBv2 positioned attention capability invariant violated after request validation")
        }
        return positioned.cbv2Forward(
            tokens, cache: asKVCaches(caches), positionIds: positionIds)
    }
}

// MARK: - Request-owned recurrent state

extension CBv2SteppableLanguageModelAdapter: CBv2RecurrentSteppableModel {
    public var cbv2Capabilities: CBv2ModelCapabilities {
        (model as? any CBv2ModelCapabilityProviding)?.cbv2Capabilities ?? .attentionOnly
    }

    public var recurrentStateSpec: CBv2RecurrentStateSpec? {
        (model as? any CBv2RecurrentLanguageModelForwardable)?.cbv2RecurrentStateSpec
    }

    public func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        guard let recurrent = model as? any CBv2RecurrentLanguageModelForwardable else {
            preconditionFailure(
                "CBv2 recurrent forward reached a model without recurrent-state support")
        }
        return recurrent.cbv2Forward(
            tokens, caches: asKVCaches(caches), recurrentState: recurrentState)
    }
}

extension CBv2SteppableLanguageModelAdapter: CBv2PositionedRecurrentSteppableModel {
    public func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        guard let recurrent = model as? any CBv2PositionedRecurrentLanguageModelForwardable else {
            preconditionFailure(
                "CBv2 positioned recurrent capability invariant violated after request validation")
        }
        return recurrent.cbv2Forward(
            tokens, caches: asKVCaches(caches), recurrentState: recurrentState,
            positionIds: positionIds)
    }
}

extension CBv2SteppableLanguageModelAdapter: CBv2RecurrentPrefillSteppableModel {
    /// Recurrent prompt-output narrowing. Fail SAFE, not fatal: a model
    /// without the narrowing conformance reproduces the full positioned
    /// forward and the engine's own slicing, so this conformance can never
    /// change what a non-conforming model computes.
    public func recurrentPrefill(
        tokens: MLXArray,
        inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        if let prefillable = model as? any CBv2RecurrentLanguageModelPrefillForwardable {
            return prefillable.cbv2RecurrentPrefill(
                tokens, inputEmbedding: inputEmbeddings, cache: asKVCaches(caches),
                recurrentState: recurrentState, positionIds: positionIds,
                requirement: requirement)
        }
        let logits: MLXArray
        if let inputEmbeddings {
            guard let positioned = model as? any CBv2PositionedRecurrentEmbeddingForwardable
            else {
                preconditionFailure(
                    "CBv2 recurrent embedding prefill reached an unsupported model")
            }
            logits = positioned.embeddingForward(
                tokens, inputEmbedding: inputEmbeddings,  // non-nil in this branch
                cache: asKVCaches(caches),
                recurrentState: recurrentState, positionIds: positionIds)
        } else if let positioned = model
            as? any CBv2PositionedRecurrentLanguageModelForwardable
        {
            logits = positioned.cbv2Forward(
                tokens, caches: asKVCaches(caches), recurrentState: recurrentState,
                positionIds: positionIds)
        } else {
            // A model conforming only to the unpositioned recurrent protocol
            // keeps its pre-seam prefill path; explicit positions without
            // the positioned refinement were unreachable before the seam
            // and stay a programmer error.
            guard positionIds == nil,
                let recurrent = model as? any CBv2RecurrentLanguageModelForwardable
            else {
                preconditionFailure(
                    "CBv2 recurrent prefill reached an unsupported model")
            }
            logits = recurrent.cbv2Forward(
                tokens, caches: asKVCaches(caches), recurrentState: recurrentState)
        }
        switch requirement {
        case .evaluationOnly: return logits[0..., -1, 0 ..< 1]
        case .lastPositionLogits: return logits[0..., -1, 0...]
        }
    }
}

// MARK: - Prompt-output narrowing (prefill only)

/// Answered at RUNTIME like the multimodal/MTP capabilities: only models
/// conforming to `CBv2LanguageModelPrefillForwardable` (Gemma4TextModel) can
/// narrow their prompt output. Everything else keeps the full-logits
/// `forward` contract and is sliced by the engine, so this conformance can
/// never change what a non-conforming model computes.
extension CBv2SteppableLanguageModelAdapter: CBv2PackedPrefillSteppableModel {

    public var supportsPackedPrefill: Bool {
        guard cbv2Capabilities.supportsPackedPrefill else { return false }
        if let claim = (model as? CBv2LanguageModelPrefillForwardable)?
            .cbv2SupportsPackedPrefill
        {
            return claim
        }
        return (model as? CBv2RecurrentLanguageModelPrefillForwardable)?
            .cbv2SupportsPackedPrefill ?? false
    }

    public var supportsPackedMultimodalPrefill: Bool {
        (model as? CBv2LanguageModelPrefillForwardable)?
            .cbv2SupportsPackedMultimodalPrefill ?? false
    }

    public func prefill(
        tokens: MLXArray,
        inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        guard let prefillable = model as? CBv2LanguageModelPrefillForwardable else {
            // Fail SAFE, not fatal: reproduce `forward` + the engine's own
            // slicing. `EngineLoopV2.prefillOutput` only routes here after a
            // successful cast, so this is belt-and-braces for direct callers.
            let logits: MLXArray
            if let inputEmbeddings {
                logits = forward(
                    tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches)
            } else {
                logits = forward(tokens: tokens, caches: caches)
            }
            switch requirement {
            case .evaluationOnly: return logits[0..., -1, 0 ..< 1]
            case .lastPositionLogits: return logits[0..., -1, 0...]
            }
        }
        return prefillable.cbv2Prefill(
            tokens,
            inputEmbedding: inputEmbeddings,
            cache: asKVCaches(caches),
            requirement: requirement)
    }
}

// MARK: - Multimodal (vision prefill)

/// The adapter answers the multimodal capability at RUNTIME: it can wrap any
/// `LanguageModel`, and only models conforming to `CBv2EmbeddingForwardable`
/// (Gemma4TextModel) can prefill from spliced embeddings. Conformance alone
/// is structural, not capability — a Gemma4TextModel loaded from a TEXT-ONLY
/// config (`use_bidirectional_attention` nil/non-`vision`) can execute the
/// embedding forward but was never trained for the bidirectional span masks
/// CBv2 applies, so the capability check also consults the model-level
/// `supportsVisionSpanPrefill` flag (PR#63 review). Requests against
/// non-conforming models or unsupported configs are rejected at submit
/// (`CBv2MultimodalError.unsupportedModel`), so the trapping guards below
/// are unreachable in a correctly gated engine.
extension CBv2SteppableLanguageModelAdapter: CBv2MultimodalSteppableModel {

    public var supportsMultimodalPrefill: Bool {
        guard let model = model as? CBv2EmbeddingForwardable else { return false }
        return model.supportsVisionSpanPrefill || model.supportsCausalVisionPrefill
    }

    public func supportsMultimodalPrefill(attention: CBv2MultimodalAttention) -> Bool {
        guard let model = model as? CBv2EmbeddingForwardable else { return false }
        switch attention {
        case .bidirectionalSpans: return model.supportsVisionSpanPrefill
        case .causal: return model.supportsCausalVisionPrefill
        }
    }

    public func embedPromptTokens(_ tokens: MLXArray) -> MLXArray {
        guard let embeddable = model as? CBv2EmbeddingForwardable else {
            preconditionFailure(
                "CBv2 multimodal: \(type(of: model)) is not CBv2EmbeddingForwardable — submit gating failed"
            )
        }
        return embeddable.scaledInputEmbeddings(tokens)
    }

    public func forward(
        tokens: MLXArray, inputEmbeddings: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        guard let embeddable = model as? CBv2EmbeddingForwardable else {
            preconditionFailure(
                "CBv2 multimodal: \(type(of: model)) is not CBv2EmbeddingForwardable — submit gating failed"
            )
        }
        return embeddable.embeddingForward(
            tokens, inputEmbedding: inputEmbeddings, cache: asKVCaches(caches))
    }
}
extension CBv2SteppableLanguageModelAdapter: CBv2DeepstackMultimodalSteppableModel {
    public var deepstackLayerCount: Int {
        (model as? CBv2DeepstackEmbeddingForwardable)?.deepstackLayerCount ?? 0
    }

    public func forward(
        tokens: MLXArray,
        inputEmbeddings: MLXArray,
        deepstackEmbeddings: [MLXArray],
        caches: [CBv2AttendingLayerCache],
        positionIds: MLXArray?
    ) -> MLXArray {
        guard let deepstack = model as? CBv2DeepstackEmbeddingForwardable else {
            preconditionFailure(
                "CBv2 DeepStack forward reached a model without DeepStack support")
        }
        return deepstack.embeddingForward(
            tokens,
            inputEmbedding: inputEmbeddings,
            deepstackEmbeddings: deepstackEmbeddings,
            cache: asKVCaches(caches),
            positionIds: positionIds)
    }
}

extension CBv2SteppableLanguageModelAdapter: CBv2PositionedEmbeddingSteppableModel {
    public func forward(
        tokens: MLXArray,
        inputEmbeddings: MLXArray,
        caches: [CBv2AttendingLayerCache],
        positionIds: MLXArray?
    ) -> MLXArray {
        guard let positioned = model as? CBv2PositionedEmbeddingForwardable else {
            preconditionFailure(
                "CBv2 positioned embedding capability invariant violated after request validation")
        }
        return positioned.embeddingForward(
            tokens,
            inputEmbedding: inputEmbeddings,
            cache: asKVCaches(caches),
            positionIds: positionIds)
    }
}

extension CBv2SteppableLanguageModelAdapter: CBv2PositionedMultimodalSteppableModel {
    public func forward(
        tokens: MLXArray,
        inputEmbeddings: MLXArray,
        caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> MLXArray {
        guard let positioned = model as? any CBv2PositionedRecurrentEmbeddingForwardable else {
            preconditionFailure(
                "CBv2 positioned multimodal capability invariant violated after request validation")
        }
        return positioned.embeddingForward(
            tokens,
            inputEmbedding: inputEmbeddings,
            cache: asKVCaches(caches),
            recurrentState: recurrentState,
            positionIds: positionIds)
    }
}

// MARK: - MTP (speculative decoding)

/// Answered at RUNTIME like the multimodal capability: the adapter wraps any
/// `LanguageModel`, and only `CBv2MTPForwardable` conformers (Gemma4TextModel)
/// can drive MTP rounds. The engine gates speculation on
/// `mtpCaptureLayers != nil` before ever calling `forwardWithHidden`, so the
/// trapping guard below is unreachable in a correctly gated engine.
extension CBv2SteppableLanguageModelAdapter: CBv2MTPSteppableModel {

    public var mtpCaptureLayers: CBv2MTPCaptureLayers? {
        if let forwardable = model as? CBv2MTPForwardable {
            return forwardable.cbv2MTPCaptureLayers
        }
        return (model as? any CBv2RecurrentMTPForwardable) != nil
            ? CBv2MTPCaptureLayers(full: 0, sliding: 0) : nil
    }

    public var mtpTargetIdentity: ObjectIdentifier? {
        if let target = model as? any CBv2MTPForwardable {
            return target.cbv2MTPTargetIdentity
        }
        return (model as? any CBv2RecurrentMTPForwardable)?.cbv2MTPTargetIdentity
    }

    public var supportsRequestStatefulMTP: Bool {
        model is any CBv2RecurrentMTPForwardable
    }

    public func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        guard let forwardable = model as? CBv2MTPForwardable else {
            preconditionFailure(
                "CBv2 MTP: \(type(of: model)) is not CBv2MTPForwardable — engine gating failed")
        }
        return forwardable.cbv2ForwardWithHidden(tokens, caches: asKVCaches(caches))
    }
}

extension CBv2SteppableLanguageModelAdapter: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(
        _ logits: MLXArray
    ) -> (ids: MLXArray, values: MLXArray) {
        guard let provider = model as? any CBv2MTPPolicyTopTwoProviding else {
            preconditionFailure(
                "CBv2 MTP top-two reached a target without the additive provider")
        }
        return provider.cbv2MTPTopTwo(logits)
    }
}

extension CBv2SteppableLanguageModelAdapter:
    CBv2MTPPolicyTopTwoCapabilityProviding
{
    public var cbv2MTPPolicyTopTwoAvailable: Bool {
        model is any CBv2MTPPolicyTopTwoProviding
    }
}

extension CBv2SteppableLanguageModelAdapter: CBv2RecurrentMTPSteppableModel {
    public func forwardWithHiddenForPrefill(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        if let forwardable = model as? any CBv2RecurrentPrefillHiddenForwardable {
            return forwardable.cbv2ForwardWithHiddenForPrefill(
                tokens, caches: asKVCaches(caches), recurrentState: recurrentState,
                positionIds: positionIds, requirement: requirement)
        }
        return forwardWithHidden(tokens: tokens, caches: caches, recurrentState: recurrentState,
                                 positionIds: positionIds)
    }

    public func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        guard let forwardable = model as? any CBv2RecurrentMTPForwardable else {
            preconditionFailure(
                "CBv2 recurrent MTP: \(type(of: model)) lacks hidden capture")
        }
        return forwardable.cbv2ForwardWithHidden(
            tokens, caches: asKVCaches(caches), recurrentState: recurrentState,
            positionIds: positionIds)
    }

    public var supportsCapturedVerifyWindow: Bool {
        model is any CBv2RecurrentCaptureMTPForwardable
    }

    public func forwardWithHiddenCaptured(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        guard let forwardable = model as? any CBv2RecurrentCaptureMTPForwardable else {
            preconditionFailure(
                "CBv2 capture-verify: \(type(of: model)) lacks captured-window support")
        }
        return forwardable.cbv2ForwardWithHiddenCaptured(
            tokens, caches: asKVCaches(caches), recurrentState: recurrentState,
            positionIds: positionIds)
    }
}



extension CBv2SteppableLanguageModelAdapter: CBv2HistoricalAttentionCheckpointProviding {
    public var cbv2SupportsHistoricalAttentionCheckpoint: Bool {
        (model as? any CBv2HistoricalAttentionCheckpointProviding)?.cbv2SupportsHistoricalAttentionCheckpoint == true
    }
}
