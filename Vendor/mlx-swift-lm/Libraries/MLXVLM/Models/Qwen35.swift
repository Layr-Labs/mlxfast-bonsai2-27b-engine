//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/25.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_5
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

private enum Qwen35VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

// MARK: - Gated Delta Helpers
//
// The gated-delta (GDN) recurrence is shared with the LLM Qwen3.5 / Qwen3-Next
// models via `MLXLLM.gatedDeltaUpdate`. This file previously carried a private
// duplicate that ran the recurrent state (and g/beta) in bf16, which drifted
// from the LLM twin's fp32 fix (upstream c566c95 + 93cf322) and degraded
// Qwen3.5-VL SSM fidelity. The duplicate was deleted; the model below calls the
// shared fp32 implementation directly.

// MARK: - Configuration

public struct Qwen35Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public var modelType: String = ""
        public var hiddenSize: Int = 4096
        public var hiddenLayers: Int = 32
        public var intermediateSize: Int = 14_336
        public var attentionHeads: Int = 32
        public var kvHeads: Int = 8
        public var linearNumValueHeads: Int = 64
        public var linearNumKeyHeads: Int = 16
        public var linearKeyHeadDim: Int = 192
        public var linearValueHeadDim: Int = 128
        public var linearConvKernelDim: Int = 4
        public var rmsNormEps: Float = 1e-6
        public var vocabularySize: Int = 248_320
        public var ropeTheta: Float = 100_000.0
        public var partialRotaryFactor: Float = 0.25
        public var maxPositionEmbeddings: Int = 131_072
        public var tieWordEmbeddings: Bool = false
        public var attentionBias: Bool = false
        public var headDim: Int?
        public var ropeParameters: [String: StringOrNumber]?
        public var fullAttentionInterval: Int = 4

        // MoE fields
        public var numExperts: Int = 0
        public var numExpertsPerTok: Int = 0
        public var decoderSparseStep: Int = 1
        public var sharedExpertIntermediateSize: Int = 0
        public var moeIntermediateSize: Int = 0
        public var normTopkProb: Bool = true

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case headDim = "head_dim"
            case ropeParameters = "rope_parameters"
            case fullAttentionInterval = "full_attention_interval"
            case numExperts = "num_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case decoderSparseStep = "decoder_sparse_step"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case normTopkProb = "norm_topk_prob"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14_336
            self.attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            self.linearNumValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
            self.linearNumKeyHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
            self.linearKeyHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
            self.linearValueHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
            self.linearConvKernelDim =
                try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
            self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            self.vocabularySize =
                try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248_320
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
            self.tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            self.attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            self.fullAttentionInterval =
                try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

            self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
            self.numExpertsPerTok =
                try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
            self.decoderSparseStep =
                try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
            self.sharedExpertIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
            self.moeIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
            self.normTopkProb =
                try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

            let defaultRopeParameters: [String: StringOrNumber] = [
                "type": .string("default"),
                "mrope_section": .ints([11, 11, 10]),
                "rope_theta": .float(100_000.0),
                "partial_rotary_factor": .float(0.25),
            ]

            var decodedRope = try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

            if decodedRope == nil {
                let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                let partial = try container.decodeIfPresent(
                    Float.self, forKey: .partialRotaryFactor)
                if ropeTheta != nil || partial != nil {
                    decodedRope = defaultRopeParameters
                    if let ropeTheta {
                        decodedRope?["rope_theta"] = .float(ropeTheta)
                    }
                    if let partial {
                        decodedRope?["partial_rotary_factor"] = .float(partial)
                    }
                }
            }

            if var decodedRope {
                if decodedRope["type"] == nil, let ropeType = decodedRope["rope_type"] {
                    decodedRope["type"] = ropeType
                }
                self.ropeParameters = decodedRope
                self.ropeTheta = decodedRope["rope_theta"]?.asFloat() ?? 100_000.0
                self.partialRotaryFactor = decodedRope["partial_rotary_factor"]?.asFloat() ?? 0.25
            } else {
                self.ropeParameters = defaultRopeParameters
                self.ropeTheta = 100_000.0
                self.partialRotaryFactor = 0.25
            }

            if self.headDim == nil {
                self.headDim = self.hiddenSize / self.attentionHeads
            }
        }
    }

    public typealias VisionConfiguration = Qwen3VLConfiguration.VisionConfiguration

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 248_056 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 248_057 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? 248_045 }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? 248_046 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabularySize }
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Public Serving Seams

/// Attention semantics for Qwen3.5 visual placeholder tokens in the language
/// model. Unlike Gemma, Qwen keeps these tokens on the ordinary causal path.
public enum Qwen35VisionTokenAttention: Sendable, Equatable {
    case causal
}

/// Public configuration facts needed to locate visual spans and reproduce
/// Qwen3.5 M-RoPE positions outside the legacy `prepare` path.
public struct Qwen35VisionSeamConfiguration: Sendable, Equatable {
    /// Token ids whose text embeddings are replaced by vision features.
    public let imagePlaceholderTokenId: Int
    public let videoPlaceholderTokenId: Int

    /// Token ids consumed by Qwen's M-RoPE position calculation. These are
    /// normally identical to the placeholder ids, but both config fields are
    /// exposed because checkpoints may encode them separately.
    public let imagePositionTokenId: Int
    public let videoPositionTokenId: Int
    public let visionStartTokenId: Int
    public let visionEndTokenId: Int

    public let spatialMergeSize: Int
    public let temporalPatchSize: Int
    public let attention: Qwen35VisionTokenAttention
}

/// One text-space slice of Qwen3.5 vision tower output.
public struct Qwen35VisionFeature: @unchecked Sendable {
    public enum Kind: Sendable, Equatable {
        /// One processor image, in image packing order.
        case image(index: Int)
        /// One temporal vision frame within one processor video grid.
        ///
        /// A temporal vision frame represents `temporalPatchSize` sampled
        /// source frames (the processor pads the last group when necessary).
        case videoFrame(videoIndex: Int, frameIndex: Int)
    }

    public let kind: Kind

    /// `[1, visualTokens, textHidden]`, in the language token-embedding dtype.
    public let features: MLXArray

    fileprivate init(kind: Kind, features: MLXArray) {
        self.kind = kind
        self.features = features
    }
}

/// Exact final text-space vision features used by `Qwen35.prepare`.
///
/// `ordered` contains all images first, followed by videos in video order and
/// each video's temporal frames in time order. This is the same order in which
/// `prepare` packs image pixels followed by video pixels. Concatenating the
/// entries along their token axis yields `flattenedFeatures` exactly.
///
/// `@unchecked Sendable` is limited to immutable, request-owned `MLXArray`
/// handles. Callers crossing isolation must `eval` the arrays first and must
/// not mutate them after transfer.
public struct Qwen35VisionFeatures: @unchecked Sendable {
    public let ordered: [Qwen35VisionFeature]

    /// `[totalVisualTokens, textHidden]`, exactly what `prepare` scatters over
    /// image and video placeholder tokens.
    public let flattenedFeatures: MLXArray

    fileprivate init(ordered: [Qwen35VisionFeature], flattenedFeatures: MLXArray) {
        self.ordered = ordered
        self.flattenedFeatures = flattenedFeatures
    }
}

public enum Qwen35VisionSeamError: Error, Equatable {
    case missingImagePixels
    case missingImageGrids
    case missingVideoPixels
    case missingVideoGrids
    case invalidGrid(kind: String, index: Int)
    case invalidPixelRank(kind: String, actual: Int)
    case pixelCountMismatch(kind: String, expected: Int, actual: Int)
    case featureCountMismatch(expected: Int, actual: Int)
}

/// Request-owned decode position state. It contains no model-global mutable
/// state and can therefore remain attached to one serving request while other
/// requests prefill or decode on the same model.
///
public struct Qwen35PositionState: Sendable, Equatable {
    /// Per-request M-RoPE deltas, one host value per batch row. Host values
    /// make the state safely transferable without sharing an MLX graph or a
    /// mutable model property across requests.
    public let deltas: [Int32]
    public var batchSize: Int { deltas.count }

    public init(deltas: [Int32]) {
        self.deltas = deltas
    }

    /// Build `[3, batch, length]` decode position ids from this request's
    /// delta and its current cache offset.
    public func decodePositionIds(cacheOffset: Int, length: Int = 1) -> MLXArray {
        precondition(cacheOffset >= 0, "cacheOffset must be non-negative")
        precondition(length > 0, "decode position length must be positive")

        var base = MLXArray(0 ..< length).asType(.int32)
        base = broadcast(base[.newAxis, 0...], to: [batchSize, length])
        let offset = MLXArray(cacheOffset).asType(.int32) + MLXArray(deltas)
        base = base + offset[0..., .newAxis]
        return broadcast(base[.newAxis, 0..., 0...], to: [3, batchSize, length])
    }
}

/// Prompt M-RoPE positions and the request-owned state used for subsequent
/// decode positions.
///
/// `@unchecked Sendable` is limited to the immutable prompt `MLXArray` handle.
/// Evaluate it under model-container isolation before transferring the result;
/// `decodeState` itself is ordinary host data with checked `Sendable` safety.
public struct Qwen35PositionResult: @unchecked Sendable {
    /// Full prompt position ids, shape `[3, batch, promptLength]`.
    public let promptPositionIds: MLXArray
    public let decodeState: Qwen35PositionState
    public let promptLength: Int

    public init(
        promptPositionIds: MLXArray,
        decodeState: Qwen35PositionState,
        promptLength: Int
    ) {
        self.promptPositionIds = promptPositionIds
        self.decodeState = decodeState
        self.promptLength = promptLength
    }

    /// Slice prompt positions for chunked prefill without recomputing or
    /// mutating request state.
    public func promptPositionIds(in range: Range<Int>) -> MLXArray {
        precondition(
            range.lowerBound >= 0 && range.upperBound <= promptLength,
            "prompt position range must be within 0..<promptLength")
        return promptPositionIds[0..., 0..., range]
    }
}

public enum Qwen35PositionSeamError: Error, Equatable {
    case invalidInputRank(Int)
    case invalidAttentionMaskRank(Int)
    case attentionMaskShapeMismatch
    case multimodalBatchUnsupported(Int)
    case invalidGrid(kind: String, index: Int)
    case visualTokenRunMismatch(kind: String, gridIndex: Int, expected: Int, actual: Int)
}

// MARK: - Language

enum Qwen35Language {

    final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let mropeIndices: MLXArray

        init(dim: Int, base: Float, mropeSection: [Int]) {
            let safeDim = max(1, dim)
            var freq = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
            freq = freq / Float(safeDim)
            self.invFreq = 1.0 / pow(MLXArray(base), freq)

            // Precompute which of the three position planes (t/h/w) owns each
            // frequency so the interleave is a single takeAlong instead of a
            // per-frequency slice loop building ~dims lazy ops per forward.
            let sections = mropeSection.count >= 3 ? mropeSection : [11, 11, 10]
            var indices = [Int32](repeating: 0, count: freq.dim(0))
            for (dimension, offset) in [(1, 1), (2, 2)] {
                let end = min(sections[dimension] * 3, indices.count)
                for index in stride(from: offset, to: end, by: 3) {
                    indices[index] = Int32(dimension)
                }
            }
            self.mropeIndices = MLXArray(indices).reshaped(1, 1, 1, -1)
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            takeAlong(freqs, mropeIndices, axis: 0).squeezed(axis: 0)
        }

        func callAsFunction(x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = broadcast(
                    positionIds[.newAxis, 0..., 0...],
                    to: [3, positionIds.dim(0), positionIds.dim(1)])
            }

            let pos = positionIds.asType(.float32)
            var inv = invFreq.asType(.float32)
            inv = inv[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * inv
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            return (cos(emb).asType(x.dtype), sin(emb).asType(x.dtype))
        }
    }

    static func applyMultimodalRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cos = expandedDimensions(cos, axis: 1)
        let sin = expandedDimensions(sin, axis: 1)

        let rotaryDim = cos.dim(-1)
        let qDim = q.dim(-1)
        let kDim = k.dim(-1)

        let qRot = q[.ellipsis, ..<rotaryDim]
        let kRot = k[.ellipsis, ..<rotaryDim]

        let qEmbedded = (qRot * cos) + (QwenVL.rotateHalf(qRot) * sin)
        let kEmbedded = (kRot * cos) + (QwenVL.rotateHalf(kRot) * sin)

        let qOut: MLXArray
        if rotaryDim < qDim {
            qOut = concatenated([qEmbedded, q[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            qOut = qEmbedded
        }

        let kOut: MLXArray
        if rotaryDim < kDim {
            kOut = concatenated([kEmbedded, k[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            kOut = kEmbedded
        }

        return (qOut, kOut)
    }

    final class RMSNormGated: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float

        init(dimensions: Int, eps: Float = 1e-6) {
            self.eps = eps
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
            var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
            if let gate {
                x = x * silu(gate)
            }
            return x
        }
    }

    final class Attention: Module {
        let numKeyValueHeads: Int
        let numAttentionHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.numKeyValueHeads = args.kvHeads
            self.numAttentionHeads = args.attentionHeads
            self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
            self.scale = pow(Float(headDim), -0.5)

            _qProj.wrappedValue = Linear(
                args.hiddenSize, numAttentionHeads * headDim * 2, bias: args.attentionBias)
            _kProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _vProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _oProj.wrappedValue = Linear(
                numAttentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

            let mrope = args.ropeParameters?["mrope_section"]?.asInts() ?? [11, 11, 10]
            let rotaryDim = Int(Float(headDim) * args.partialRotaryFactor)
            self.rotaryEmbedding = RotaryEmbedding(
                dim: rotaryDim, base: args.ropeTheta, mropeSection: mrope)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            let qProjOutput = qProj(x)
            let qSplit = qProjOutput.reshaped(B, L, numAttentionHeads, -1).split(parts: 2, axis: -1)
            var queries = qSplit[0]
            let gate = qSplit[1].reshaped(B, L, -1)

            var keys = kProj(x)
            var values = vProj(x)

            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys.reshaped(B, L, numKeyValueHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

            var kvSeqLen = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSeqLen += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + L, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [B, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else if let cache {
                kvSeqLen += cache.offset + 1
            }

            let (cosValues, sinValues) = rotaryEmbedding(x: values, positionIds: positionIds!)
            (queries, keys) = applyMultimodalRotaryPosEmb(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                attentionMask = .array(mask[.ellipsis, 0 ..< kvSeqLen])
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return oProj(output * sigmoid(gate))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class GatedDeltaNet: Module {
        let hiddenSize: Int
        let numVHeads: Int
        let numKHeads: Int
        let headKDim: Int
        let headVDim: Int
        let keyDim: Int
        let valueDim: Int
        let convKernelSize: Int
        let convDim: Int

        @ModuleInfo(key: "conv1d") var conv1d: Conv1d
        @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
        @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
        @ModuleInfo(key: "in_proj_b") var inProjB: Linear
        @ModuleInfo(key: "in_proj_a") var inProjA: Linear

        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
        @ParameterInfo(key: "A_log") var aLog: MLXArray

        @ModuleInfo(key: "norm") var norm: RMSNormGated
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.hiddenSize = args.hiddenSize
            self.numVHeads = args.linearNumValueHeads
            self.numKHeads = args.linearNumKeyHeads
            self.headKDim = args.linearKeyHeadDim
            self.headVDim = args.linearValueHeadDim
            self.keyDim = headKDim * numKHeads
            self.valueDim = headVDim * numVHeads
            self.convKernelSize = args.linearConvKernelDim
            self.convDim = keyDim * 2 + valueDim

            precondition(
                numVHeads % numKHeads == 0,
                "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
            )

            _conv1d.wrappedValue = Conv1d(
                inputChannels: convDim,
                outputChannels: convDim,
                kernelSize: convKernelSize,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: convDim,
                bias: false
            )

            _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
            _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
            _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
            _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

            _dtBias.wrappedValue = MLXArray.ones([numVHeads])
            let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
            _aLog.wrappedValue = log(a)

            _norm.wrappedValue = RMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
            _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            mask: MLXArray? = nil,
            cache: MambaCache? = nil
        ) -> MLXArray {
            let B = inputs.dim(0)
            let S = inputs.dim(1)

            var mixedQKV = inProjQKV(inputs)
            let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
            let b = inProjB(inputs)
            let a = inProjA(inputs)

            let convState: MLXArray
            if let cacheState = cache?[0] {
                convState = cacheState
            } else {
                convState = MLXArray.zeros(
                    [B, max(0, convKernelSize - 1), convDim], dtype: inputs.dtype)
            }

            if let mask {
                mixedQKV = MLX.where(mask[.ellipsis, .newAxis], mixedQKV, 0)
            }

            let convInput = concatenated([convState, mixedQKV], axis: 1)
            if let cache, convKernelSize > 1 {
                cache[0] = convInput[0..., (-(convKernelSize - 1))...]
            }

            let convOut = silu(conv1d(convInput))
            let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = split[0].reshaped(B, S, numKHeads, headKDim)
            let k = split[1].reshaped(B, S, numKHeads, headKDim)
            let v = split[2].reshaped(B, S, numVHeads, headVDim)

            var state = cache?[1]
            let dtype = q.dtype
            let invScale = pow(Float(headKDim), -0.5)
            let qNormed =
                MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
            let kNormed =
                MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

            var out: MLXArray
            (out, state) = gatedDeltaUpdate(
                q: qNormed,
                k: kNormed,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: state,
                mask: mask
            )

            if let cache {
                cache[1] = state
            }

            out = norm(out, gate: z)
            return outProj(out.reshaped(B, S, -1))
        }
    }

    final class SparseMoeBlock: Module, UnaryLayer {
        let normTopkProb: Bool
        let numExperts: Int
        let topK: Int

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

        @ModuleInfo(key: "shared_expert") var sharedExpert: MLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.normTopkProb = args.normTopkProb
            self.numExperts = args.numExperts
            self.topK = args.numExpertsPerTok

            _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
            // Fused gate_up: one gather_qmm serves gate+up. The wrapper's
            // sanitize concatenates split checkpoints into this layout
            // (`qwen35FuseSwitchMLPGateUp`).
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.numExperts,
                fuseGateUp: true
            )

            _sharedExpert.wrappedValue = MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize
            )
            _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var gates = gate(x)
            gates = MLX.softmax(gates, axis: -1, precise: true)

            let kth = gates.dim(-1) - topK
            let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
            var scores = MLX.takeAlong(gates, inds, axis: -1)
            if normTopkProb {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }

            let y = switchMLP(x, inds)
            let combined = weightedExpertSum(y, scores.asType(y.dtype))

            var sharedY = sharedExpert(x)
            sharedY = sigmoid(sharedExpertGate(x)) * sharedY

            return combined + sharedY
        }
    }

    final class DecoderLayer: Module {
        let isLinear: Bool

        @ModuleInfo(key: "self_attn") var selfAttn: Attention?
        @ModuleInfo(key: "linear_attn") var linearAttn: GatedDeltaNet?

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        @ModuleInfo(key: "mlp") var mlp: Module

        init(_ args: Qwen35Configuration.TextConfiguration, layerIdx: Int) {
            self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

            if isLinear {
                _linearAttn.wrappedValue = GatedDeltaNet(args)
            } else {
                _selfAttn.wrappedValue = Attention(args)
            }

            if args.numExperts > 0 {
                _mlp.wrappedValue = SparseMoeBlock(args)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            }

            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)

            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXArray?,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let r: MLXArray
            if isLinear {
                r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
            } else {
                r = selfAttn!(
                    inputLayerNorm(x), mask: attentionMask, cache: cache, positionIds: positionIds)
            }

            let h = x + r
            return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") fileprivate var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        let ssmIdx: Int
        let faIdx: Int

        init(_ args: Qwen35Configuration.TextConfiguration) {
            precondition(args.vocabularySize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            _layers.wrappedValue = (0 ..< args.hiddenLayers).map {
                DecoderLayer(args, layerIdx: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

            self.ssmIdx = 0
            self.faIdx = args.fullAttentionInterval - 1
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil
        ) -> MLXArray {
            var hiddenStates: MLXArray
            if let inputsEmbeds {
                hiddenStates = inputsEmbeds
            } else {
                hiddenStates = embedTokens(inputs)
            }

            var cacheArray = cache
            if cacheArray == nil {
                cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
            }

            let faMaskMode = createAttentionMask(
                h: hiddenStates, cache: cacheArray?[faIdx], returnArray: true)
            let faMask: MLXArray?
            if case .array(let arrayMask) = faMaskMode {
                faMask = arrayMask
            } else {
                faMask = nil
            }
            let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

            for (index, layer) in layers.enumerated() {
                let layerSSMMask = layer.isLinear ? ssmMask : nil
                hiddenStates = layer(
                    hiddenStates,
                    attentionMask: faMask,
                    ssmMask: layerSSMMask,
                    cache: cacheArray?[index],
                    positionIds: positionIds
                )
            }

            return norm(hiddenStates)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen35Configuration
        let textConfig: Qwen35Configuration.TextConfiguration
        let modelType: String
        let kvHeads: [Int]

        /// Rope deltas from the most recent prefill, used by the decode-step
        /// offset path. NOTE: module-level state — request-scoped in spirit.
        /// Safe for text-only traffic (deltas are always 0) and for the
        /// serialized media path (prepare() resets + recomputes per request),
        /// but concurrent media + batched-text traffic on one module could
        /// race it (pre-existing design constraint inherited from the
        /// HF/mlx-vlm ports, which keep rope_deltas on the model too).
        fileprivate var ropeDeltas: MLXArray? = nil

        init(_ config: Qwen35Configuration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.modelType = config.textConfiguration.modelType
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.kvHeads,
                count: config.textConfiguration.hiddenLayers
            )

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabularySize,
                    bias: false)
            }
            super.init()
        }

        func resetPositionState() {
            ropeDeltas = nil
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            mask: MLXArray? = nil,
            positionIds providedPositionIds: MLXArray? = nil,
            pixelValues: MLXArray? = nil,
            imageGridTHW: [THW]? = nil,
            videoGridTHW: [THW]? = nil
        ) -> LMOutput {
            // Ensure inputs is 2D [batch, seq]. Text-only callers (e.g.
            // WiredMemoryUtils, TokenIterator) may pass 1D token arrays.
            let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs

            if pixelValues != nil {
                ropeDeltas = nil
            }

            var cacheOffset = 0
            if let cache, let faCache = cache[model.faIdx] {
                cacheOffset = faCache.offset
            }

            var ropeMask = mask
            if let mask, mask.dim(-1) != inputs.dim(-1) {
                ropeMask = nil
            }

            var positionIds = providedPositionIds
            if positionIds == nil && (ropeMask == nil || ropeMask?.ndim == 2) {
                if (cache != nil && cache?[model.faIdx] != nil && cacheOffset == 0)
                    || ropeDeltas == nil
                    || cache == nil
                {
                    // ALWAYS recompute at the start of a sequence (offset 0) or
                    // when no deltas exist — mirrors Qwen3VL.swift and HF
                    // transformers (`cache_position[0] == 0` ⇒ recompute).
                    //
                    // This branch previously reused a `precomputedPositionIds`
                    // module property when set. That was per-REQUEST state cached
                    // on the long-lived model module: the NEXT text-only request
                    // (fresh cache, offset 0) would slice the previous request's
                    // position ids, MLX clamps the out-of-range slice to the OLD
                    // prompt length, and the first full-attention layer died with
                    //   [broadcast_shapes] (1,H,L_new,64) vs (1,1,L_old,64)
                    // whenever the new prompt was longer (provider crash: any
                    // request after the startup self-test decode).
                    let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                        inputIds: inputs,
                        imageGridTHW: imageGridTHW,
                        videoGridTHW: videoGridTHW,
                        spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                        imageTokenId: config.imageTokenId,
                        videoTokenId: config.videoTokenId,
                        visionStartTokenId: config.visionStartTokenId,
                        attentionMask: ropeMask)
                    positionIds = computed
                    ropeDeltas = deltas
                } else {
                    let batchSize = inputs.dim(0)
                    let seqLength = inputs.dim(1)

                    // `ropeDeltas` is module-level state written by the most
                    // recent prefill (shape [prefillBatch]). Under continuous
                    // batching a cold prefill of newly-admitted requests runs
                    // BETWEEN the live batch's decode steps and overwrites it,
                    // so at decode time its batch dim can differ from this
                    // batch's. Those deltas belong to OTHER rows — never apply
                    // them here. (The old `repeated(delta, count: batchSize)`
                    // "adjustment" repeated each element batchSize times,
                    // yielding prefillBatch×batchSize rows and killing the
                    // provider with
                    //   [broadcast_shapes] (B,1) vs (prefillBatch·B,1)
                    // — d-inference issue #513. The old `>` branch silently
                    // applied a different request's deltas instead.)
                    //
                    // Only a matching batch dim can be the same-rows
                    // continuation case (media decode is B==1 via the
                    // serialized prepare path; batched text rows always have
                    // delta 0, so dropping a stale array is a numeric no-op
                    // for them).
                    //
                    // The guard is shape-total: anything other than a 1-D
                    // [batchSize] array is some other batch's state and is
                    // skipped. A mismatch is the ROUTINE continuous-batching
                    // condition (every mid-decode admission overwrites the
                    // module state), not an anomaly — so skipping, not
                    // logging or trapping, is the correct handling.
                    var delta = MLXArray(cacheOffset).asType(.int32)
                    if let ropeDeltas, ropeDeltas.ndim == 1, ropeDeltas.dim(0) == batchSize {
                        delta = delta + ropeDeltas.asType(.int32)
                    }

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = broadcast(base[.newAxis, 0...], to: [batchSize, seqLength])

                    if delta.ndim == 0 {
                        delta = broadcast(delta, to: [batchSize])
                    }

                    // Invariant, total by construction: delta is the scalar
                    // cacheOffset broadcast to [batchSize], optionally plus a
                    // ropeDeltas that the guard above proved is [batchSize].
                    // (Debug-only; compiles out of release builds.)
                    assert(
                        delta.ndim == 1 && delta.dim(0) == batchSize,
                        "position-id delta must be [batchSize] before broadcast")

                    base = base + delta[0..., .newAxis]
                    positionIds = broadcast(
                        base[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
                }
            }

            var out = model(
                inputs,
                inputsEmbeds: inputsEmbeds,
                cache: cache,
                positionIds: positionIds
            )

            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }

            return LMOutput(logits: out)
        }

        func makeCache(maxKVSize: Int?) -> [KVCache] {
            model.layers.map { layer in
                if layer.isLinear {
                    return MambaCache()
                }
                if let maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            }
        }
    }
}

// MARK: - Model

public class Qwen35: Module, VLMModel {
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") fileprivate var languageModel: Qwen35Language.LanguageModel

    public let config: Qwen35Configuration

    /// Checkpoint quantization policy staged by `loadWeights` (via
    /// `QuantizationPolicyReceiving`) before `sanitize` runs; drives the
    /// per-layer decision whether routed-expert gate/up halves may fuse
    /// (`Qwen35MoE.sanitize`). `nil` for unquantized checkpoints.
    public var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization?

    public init(_ config: Qwen35Configuration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen35Language.LanguageModel(config)
        super.init()
    }

    public var vocabularySize: Int { config.vocabSize }

    public var visionSeamConfiguration: Qwen35VisionSeamConfiguration {
        Qwen35VisionSeamConfiguration(
            imagePlaceholderTokenId: config.imageTokenIndex,
            videoPlaceholderTokenId: config.videoTokenIndex,
            imagePositionTokenId: config.imageTokenId,
            videoPositionTokenId: config.videoTokenId,
            visionStartTokenId: config.visionStartTokenId,
            visionEndTokenId: config.visionEndTokenId,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            temporalPatchSize: config.visionConfiguration.temporalPatchSize,
            attention: .causal)
    }

    public var imagePlaceholderTokenId: Int { config.imageTokenIndex }
    public var videoPlaceholderTokenId: Int { config.videoTokenIndex }

    /// Compute all prompt and decode positions without reading or writing the
    /// legacy language module's `ropeDeltas` property.
    public func positionResult(
        tokens: MLXArray,
        imageGrids: [THW]? = nil,
        videoGrids: [THW]? = nil,
        attentionMask: MLXArray? = nil
    ) throws -> Qwen35PositionResult {
        guard tokens.ndim == 1 || tokens.ndim == 2 else {
            throw Qwen35PositionSeamError.invalidInputRank(tokens.ndim)
        }
        let tokens = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens

        var mask = attentionMask
        if let attentionMask {
            guard attentionMask.ndim == 1 || attentionMask.ndim == 2 else {
                throw Qwen35PositionSeamError.invalidAttentionMaskRank(attentionMask.ndim)
            }
            mask = attentionMask.ndim == 1
                ? attentionMask.expandedDimensions(axis: 0) : attentionMask
            guard mask?.shape == tokens.shape else {
                throw Qwen35PositionSeamError.attentionMaskShapeMismatch
            }
        }

        let imageGrids = imageGrids?.nilIfEmpty
        let videoGrids = videoGrids?.nilIfEmpty
        if (imageGrids != nil || videoGrids != nil), tokens.dim(0) != 1 {
            throw Qwen35PositionSeamError.multimodalBatchUnsupported(tokens.dim(0))
        }

        let merge = config.visionConfiguration.spatialMergeSize
        for (kind, grids) in [("image", imageGrids ?? []), ("video", videoGrids ?? [])] {
            for (index, grid) in grids.enumerated()
            where grid.t <= 0 || grid.h <= 0 || grid.w <= 0
                || grid.h % merge != 0 || grid.w % merge != 0
            {
                throw Qwen35PositionSeamError.invalidGrid(kind: kind, index: index)
            }
        }
        try validateVisualTokenRuns(
            tokens: tokens, attentionMask: mask,
            imageGrids: imageGrids ?? [], videoGrids: videoGrids ?? [], merge: merge)

        let (positionIds, delta) = Qwen3VLLanguage.getRopeIndex(
            inputIds: tokens,
            imageGridTHW: imageGrids,
            videoGridTHW: videoGrids,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId,
            visionStartTokenId: config.visionStartTokenId,
            attentionMask: mask)

        return Qwen35PositionResult(
            promptPositionIds: positionIds,
            decodeState: Qwen35PositionState(deltas: delta.asArray(Int32.self)),
            promptLength: tokens.dim(1))
    }

    private func validateVisualTokenRuns(
        tokens: MLXArray,
        attentionMask: MLXArray?,
        imageGrids: [THW],
        videoGrids: [THW],
        merge: Int
    ) throws {
        guard tokens.dim(0) == 1 else { return }
        let values = tokens[0, 0...].asArray(Int32.self).map(Int.init)
        let maskValues = attentionMask?.asType(.int32)[0, 0...].asArray(Int32.self)
        var imageIndex = 0
        var videoIndex = 0
        var cursor = 0

        while cursor < values.count {
            let visible = maskValues.map { $0[cursor] == 1 } ?? true
            let token = values[cursor]
            guard visible, token == config.imageTokenId || token == config.videoTokenId else {
                cursor += 1
                continue
            }

            let kind = token == config.imageTokenId ? "image" : "video"
            let gridIndex = kind == "image" ? imageIndex : videoIndex
            let grids = kind == "image" ? imageGrids : videoGrids
            var end = cursor + 1
            while end < values.count,
                (maskValues.map { $0[end] == 1 } ?? true), values[end] == token
            {
                end += 1
            }
            let actual = end - cursor
            guard gridIndex < grids.count else {
                throw Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: kind, gridIndex: gridIndex, expected: 0, actual: actual)
            }
            let expected = try mergedTokenCount(
                grids[gridIndex], merge: merge, kind: kind, index: gridIndex)
            guard actual == expected else {
                throw Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: kind, gridIndex: gridIndex, expected: expected, actual: actual)
            }
            if kind == "image" { imageIndex += 1 } else { videoIndex += 1 }
            cursor = end
        }

        if imageIndex < imageGrids.count {
            let expected = try mergedTokenCount(
                imageGrids[imageIndex], merge: merge, kind: "image", index: imageIndex)
            throw Qwen35PositionSeamError.visualTokenRunMismatch(
                kind: "image", gridIndex: imageIndex, expected: expected, actual: 0)
        }
        if videoIndex < videoGrids.count {
            let expected = try mergedTokenCount(
                videoGrids[videoIndex], merge: merge, kind: "video", index: videoIndex)
            throw Qwen35PositionSeamError.visualTokenRunMismatch(
                kind: "video", gridIndex: videoIndex, expected: expected, actual: 0)
        }
    }

    private func mergedTokenCount(
        _ grid: THW, merge: Int, kind: String, index: Int
    ) throws -> Int {
        let (spatial, spatialOverflow) = (grid.h / merge).multipliedReportingOverflow(
            by: grid.w / merge)
        let (count, temporalOverflow) = grid.t.multipliedReportingOverflow(by: spatial)
        guard !spatialOverflow, !temporalOverflow else {
            throw Qwen35PositionSeamError.invalidGrid(kind: kind, index: index)
        }
        return count
    }

    /// Run the exact vision tower path used by `prepare`, cast its final
    /// output to the text embedding dtype, and return stable ordered slices.
    ///
    /// Images are emitted first in `imageGrids` order. Videos follow in
    /// `videoGrids` order, split into temporal vision frames. Qwen visual
    /// tokens remain ordinary causal language tokens; this API supplies no
    /// Gemma-style bidirectional span mask.
    public func visionFeatures(
        imagePixels: MLXArray? = nil,
        imageGrids: [THW]? = nil,
        videoPixels: MLXArray? = nil,
        videoGrids: [THW]? = nil
    ) throws -> Qwen35VisionFeatures {
        if imagePixels != nil, imageGrids?.isEmpty != false {
            throw Qwen35VisionSeamError.missingImageGrids
        }
        if imagePixels == nil, imageGrids?.isEmpty == false {
            throw Qwen35VisionSeamError.missingImagePixels
        }
        if videoPixels != nil, videoGrids?.isEmpty != false {
            throw Qwen35VisionSeamError.missingVideoGrids
        }
        if videoPixels == nil, videoGrids?.isEmpty == false {
            throw Qwen35VisionSeamError.missingVideoPixels
        }

        let imageGrids = imageGrids ?? []
        let videoGrids = videoGrids ?? []
        let merge = config.visionConfiguration.spatialMergeSize

        func validate(grids: [THW], kind: String) throws {
            for (index, grid) in grids.enumerated()
            where grid.t <= 0 || grid.h <= 0 || grid.w <= 0
                || grid.h % merge != 0 || grid.w % merge != 0
            {
                throw Qwen35VisionSeamError.invalidGrid(kind: kind, index: index)
            }
        }
        try validate(grids: imageGrids, kind: "image")
        try validate(grids: videoGrids, kind: "video")

        var pixelParts: [MLXArray] = []
        if let imagePixels {
            guard imagePixels.ndim == 2 else {
                throw Qwen35VisionSeamError.invalidPixelRank(
                    kind: "image", actual: imagePixels.ndim)
            }
            pixelParts.append(imagePixels)
        }
        if let videoPixels {
            guard videoPixels.ndim == 2 else {
                throw Qwen35VisionSeamError.invalidPixelRank(
                    kind: "video", actual: videoPixels.ndim)
            }
            pixelParts.append(videoPixels)
        }

        func validatePixels(
            _ pixels: MLXArray?, grids: [THW], kind: String
        ) throws {
            var expected = 0
            for (index, grid) in grids.enumerated() {
                let (spatial, spatialOverflow) = grid.h.multipliedReportingOverflow(by: grid.w)
                let (count, temporalOverflow) = grid.t.multipliedReportingOverflow(by: spatial)
                let (total, totalOverflow) = expected.addingReportingOverflow(count)
                guard !spatialOverflow, !temporalOverflow, !totalOverflow else {
                    throw Qwen35VisionSeamError.invalidGrid(kind: kind, index: index)
                }
                expected = total
            }
            let actual = pixels?.dim(0) ?? 0
            guard expected == actual else {
                throw Qwen35VisionSeamError.pixelCountMismatch(
                    kind: kind, expected: expected, actual: actual)
            }
        }
        try validatePixels(imagePixels, grids: imageGrids, kind: "image")
        try validatePixels(videoPixels, grids: videoGrids, kind: "video")

        let grids = imageGrids + videoGrids
        let textDType = languageModel.model.embedTokens(
            MLXArray([Int32(0)]).reshaped(1, 1)
        ).dtype
        guard !pixelParts.isEmpty else {
            return Qwen35VisionFeatures(
                ordered: [],
                flattenedFeatures: MLXArray.zeros(
                    [0, config.textConfiguration.hiddenSize], dtype: textDType))
        }

        let pixels = (pixelParts.count == 1 ? pixelParts[0] : concatenated(pixelParts))
            .asType(visionModel.patchEmbed.proj.weight.dtype)
        let (visionHidden, _) = visionModel(pixels, gridTHW: grids)
        let flattened = visionHidden.asType(textDType)
        let expectedFeatures = grids.reduce(0) { $0 + $1.product / (merge * merge) }
        guard flattened.dim(0) == expectedFeatures else {
            throw Qwen35VisionSeamError.featureCountMismatch(
                expected: expectedFeatures, actual: flattened.dim(0))
        }

        var ordered: [Qwen35VisionFeature] = []
        ordered.reserveCapacity(imageGrids.count + videoGrids.reduce(0) { $0 + $1.t })
        var cursor = 0

        for (index, grid) in imageGrids.enumerated() {
            let count = grid.product / (merge * merge)
            let slice = flattened[cursor ..< cursor + count, 0...]
                .expandedDimensions(axis: 0)
            ordered.append(.init(kind: .image(index: index), features: slice))
            cursor += count
        }
        for (videoIndex, grid) in videoGrids.enumerated() {
            let count = (grid.h / merge) * (grid.w / merge)
            for frameIndex in 0 ..< grid.t {
                let slice = flattened[cursor ..< cursor + count, 0...]
                    .expandedDimensions(axis: 0)
                ordered.append(
                    .init(
                        kind: .videoFrame(
                            videoIndex: videoIndex, frameIndex: frameIndex),
                        features: slice))
                cursor += count
            }
        }

        return Qwen35VisionFeatures(ordered: ordered, flattenedFeatures: flattened)
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.makeCache(maxKVSize: parameters?.maxKVSize)
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = imageMask .|| videoMask

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen35VLError.featureTokenMismatch(expected: nImageTokens, actual: nImageFeatures)
        }

        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let indices = nonZero(flattenedMask.asType(.bool))

        var result = flattenedEmbeds
        if !indices.isEmpty && indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }

        result = result.reshaped(originalShape)
        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(imageFrames: [THW]?, videoFrames: [THW]?) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens

        var pixelValues: MLXArray?
        var imageFrames: [THW]?
        var videoFrames: [THW]?

        let visionDType = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(visionDType))
            imageFrames = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels.asType(visionDType))
            videoFrames = video.frames
        }
        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray?

        if pixelValues != nil,
            combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames).nilIfEmpty != nil
        {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let features = try visionFeatures(
                imagePixels: input.image?.pixels,
                imageGrids: imageFrames,
                videoPixels: input.video?.pixels,
                videoGrids: videoFrames)

            let (mergedEmbeds, _) = try mergeInputIdsWithImageFeatures(
                imageFeatures: features.flattenedFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex
            )
            inputEmbeddings = mergedEmbeds
        } else {
            languageModel.resetPositionState()
        }

        let typedCache = castCache(cache)
        let output = languageModel(
            inputIds,
            inputsEmbeds: inputEmbeddings,
            cache: typedCache,
            mask: input.text.mask,
            positionIds: nil,
            pixelValues: pixelValues,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames
        )

        return .logits(output)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let typedCache = castCacheOptional(cache)
        let result = languageModel(
            inputs,
            inputsEmbeds: nil,
            cache: typedCache,
            mask: nil,
            positionIds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        )
        return result.logits
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        if metadata["format"]?.lowercased() == "mlx" {
            // A combined Qwen3.5/3.6 checkpoint may keep the native MTP head
            // under `mtp.*` in the same shards. The VLM wrapper has no MTP
            // module, so partition those tensors for the dedicated assistant
            // loader while preserving every already-converted target tensor
            // byte-for-byte. Calling the source-layout sanitizer here would
            // reapply its RMSNorm +1 conversion to serialized MLX weights.
            return weights.filter { !$0.key.hasPrefix("mtp.") }
        }
        return sanitize(weights: weights)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights.filter { !$0.key.contains("mtp.") }

        if config.textConfiguration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (key, originalValue) in weights {
            var key = key
            var value = originalValue

            if key.contains("model") {
                if key.contains("model.language_model") {
                    key = key.replacingOccurrences(
                        of: "model.language_model", with: "language_model.model")
                } else if key.contains("model.visual") {
                    key = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
                } else if key.hasPrefix("model.") {
                    // Unified Qwen 3.5 checkpoints (e.g. Qwen3.5-0.8B-MLX-4bit) ship
                    // language model tensors at bare `model.*` paths instead of
                    // `model.language_model.*`. Mirror the LLM-side fallback.
                    key = "language_model." + key
                }
            } else if key.contains("lm_head") {
                key = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            if normKeys.contains(where: { key.hasSuffix($0) }) && value.ndim == 1 {
                value = value + MLXArray(1, dtype: value.dtype)
            }

            sanitized[key] = value
        }

        return visionModel.sanitize(weights: sanitized)
    }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

extension Qwen35 {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}
