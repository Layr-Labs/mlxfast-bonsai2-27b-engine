// Copyright © 2026 Apple Inc.

// DFlash 2 block drafter.
//
// This is a port of the reference MLX Python implementation, `dflash/model_mlx.py`
// of `z-lab/dflash` at `07ebd93`: `DFlashAttention`, `GroupedDynamicCausalConv`,
// `DFlash2DecoderLayer`, `CandidateSelector`, `DFlash2DraftModel`.
//
// The drafter is a BLOCK drafter, not a chain drafter. One `propose` call
// consumes a block of `depth + 1` input tokens — the last committed token
// followed by `depth` mask tokens — and returns `depth` draft tokens. The block
// attends to itself without a causal mask and to a rotating window of context
// keys and values projected from the target's own hidden states.
//
// The drafter owns NO embedding table and NO output projection. It binds the
// target's. On this track both are packed Hadamard modules, so this file calls
// the modules and never reads a raw `.weight`. See
// `docs/bonsai2-27b-port-notes.md` section 6.1 for the defect that rule exists
// to prevent.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public enum DFlash2LayerType: String, Codable, Sendable, Equatable {
    case fullAttention = "full_attention"
    case slidingAttention = "sliding_attention"
}

public enum DFlash2Error: LocalizedError, Sendable, Equatable {
    case missingSlidingWindow
    case invalidCacheCount(expected: Int, actual: Int)
    case invalidBlockSize(Int)
    case targetHiddenSizeMismatch(expected: Int, actual: Int)
    case notBound
    case vocabularyMismatch(drafter: Int, target: Int)
    case hiddenSizeMismatch(drafter: Int, target: Int)
    case targetLayerOutOfRange(layerId: Int, layerCount: Int)
    case emptyTapLayerIds
    case duplicateTapLayerIds([Int])
    case missingConfig(String)
    case unreadableDirectory(String)
    case emptyBlockContext

    public var errorDescription: String? {
        switch self {
        case .missingSlidingWindow:
            return "A DFlash 2 sliding_attention layer needs sliding_window."
        case .invalidCacheCount(let expected, let actual):
            return "The DFlash 2 drafter needs \(expected) caches; it got \(actual)."
        case .invalidBlockSize(let blockSize):
            return "The DFlash 2 block size \(blockSize) is less than 2."
        case .targetHiddenSizeMismatch(let expected, let actual):
            return
                "The fused target hidden state is \(actual) wide; the drafter needs \(expected)."
        case .notBound:
            return "The DFlash 2 drafter is not bound to a target model."
        case .vocabularyMismatch(let drafter, let target):
            return
                "The DFlash 2 drafter has vocabulary \(drafter); the target has \(target)."
        case .hiddenSizeMismatch(let drafter, let target):
            return
                "The DFlash 2 drafter has hidden size \(drafter); the target has \(target)."
        case .targetLayerOutOfRange(let layerId, let layerCount):
            return
                "The DFlash 2 target layer id \(layerId) is outside 0..<\(layerCount)."
        case .emptyTapLayerIds:
            return "The DFlash 2 tap needs at least one target layer id."
        case .duplicateTapLayerIds(let ids):
            return "The DFlash 2 target layer ids must be unique; they are \(ids)."
        case .missingConfig(let path):
            return "The DFlash 2 drafter directory has no config.json: \(path)."
        case .unreadableDirectory(let path):
            return "The DFlash 2 drafter directory cannot be read: \(path)."
        case .emptyBlockContext:
            return "The DFlash 2 drafter has no target context rows to propose from."
        }
    }
}

/// The DFlash 2 drafter configuration, decoded from the drafter's own
/// `config.json`.
///
/// The reference reads the block fields out of the `dflash_config` object and
/// the geometry out of the top level. `rope_theta` is NOT at the top level of a
/// DFlash 2 config: it is inside `rope_parameters`, and the reference falls back
/// to that object. This decoder does the same.
public struct DFlash2Configuration: Decodable, Sendable, Equatable {
    /// The `dflash_config` object.
    public struct Block: Codable, Sendable, Equatable {
        public var blockSize: Int
        public var maskTokenId: Int
        public var targetLayerIds: [Int]
        public var convKernelSize: Int
        public var convGroupSize: Int
        public var selectorRank: Int
        public var selectorTopK: Int
        public var inputEmbeddingScale: Float
        public var outputMultiplier: Float
        public var finalLogitSoftcapping: Float?

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case maskTokenId = "mask_token_id"
            case targetLayerIds = "target_layer_ids"
            case convKernelSize = "conv_kernel_size"
            case convGroupSize = "conv_group_size"
            case selectorRank = "selector_rank"
            case selectorTopK = "selector_top_k"
            case inputEmbeddingScale = "input_embedding_scale"
            case outputMultiplier = "output_multiplier"
            case finalLogitSoftcapping = "final_logit_softcapping"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 16
            self.maskTokenId = try container.decode(Int.self, forKey: .maskTokenId)
            self.targetLayerIds = try container.decode([Int].self, forKey: .targetLayerIds)
            self.convKernelSize =
                try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 0
            self.convGroupSize =
                try container.decodeIfPresent(Int.self, forKey: .convGroupSize) ?? 0
            self.selectorRank = try container.decodeIfPresent(Int.self, forKey: .selectorRank) ?? 0
            self.selectorTopK = try container.decodeIfPresent(Int.self, forKey: .selectorTopK) ?? 0
            self.inputEmbeddingScale =
                try container.decodeIfPresent(Float.self, forKey: .inputEmbeddingScale) ?? 1
            self.outputMultiplier =
                try container.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 1
            self.finalLogitSoftcapping =
                try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        }
    }

    /// The `rope_parameters` object. A DFlash 2 config carries the rope base
    /// here rather than at the top level.
    public struct RopeParameters: Codable, Sendable, Equatable {
        public var ropeTheta: Float?
        public var ropeType: String?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
        }
    }

    public var architectures: [String]
    public var modelType: String
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var vocabularySize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int
    public var numTargetLayers: Int
    public var layerTypes: [DFlash2LayerType]
    public var slidingWindow: Int?
    public var isCausal: Bool
    public var dflash: Block

    public var blockSize: Int { dflash.blockSize }
    public var maskTokenId: Int { dflash.maskTokenId }
    public var targetLayerIds: [Int] { dflash.targetLayerIds }
    /// The width of the fused target hidden state the drafter's `fc` consumes.
    public var targetHiddenSize: Int { dflash.targetLayerIds.count * hiddenSize }
    /// The number of draft tokens one block of `blockSize` inputs produces.
    public var draftDepth: Int { blockSize - 1 }

    enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numTargetLayers = "num_target_layers"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case isCausal = "is_causal"
        case dflash = "dflash_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.dflash = try container.decode(Block.self, forKey: .dflash)
        self.numTargetLayers =
            try container.decodeIfPresent(Int.self, forKey: .numTargetLayers) ?? hiddenLayers
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.isCausal = try container.decodeIfPresent(Bool.self, forKey: .isCausal) ?? false

        let rope =
            try container.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
            ?? container.decodeIfPresent(RopeParameters.self, forKey: .ropeScaling)
        self.ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? rope?.ropeTheta ?? 10000

        let layerTypes =
            try container.decodeIfPresent([DFlash2LayerType].self, forKey: .layerTypes)
            ?? Array(repeating: .fullAttention, count: hiddenLayers)
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription:
                    "layer_types has \(layerTypes.count) entries; num_hidden_layers is \(hiddenLayers)."
            )
        }
        if layerTypes.contains(.slidingAttention), slidingWindow == nil {
            throw DecodingError.dataCorruptedError(
                forKey: .slidingWindow, in: container,
                debugDescription:
                    "A config with sliding_attention layers must define sliding_window.")
        }
        self.layerTypes = layerTypes

        guard dflash.blockSize >= 2 else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.block_size must be at least 2.")
        }
        guard dflash.maskTokenId >= 0, dflash.maskTokenId < vocabularySize else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription:
                    "dflash_config.mask_token_id \(dflash.maskTokenId) is outside 0..<\(vocabularySize)."
            )
        }
        guard !dflash.targetLayerIds.isEmpty else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.target_layer_ids must not be empty.")
        }
        guard Set(dflash.targetLayerIds).count == dflash.targetLayerIds.count else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.target_layer_ids must be unique.")
        }
        for layerId in dflash.targetLayerIds
        where layerId < 0 || layerId >= numTargetLayers {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription:
                    "dflash_config target layer id \(layerId) is outside 0..<\(numTargetLayers).")
        }
        guard dflash.selectorTopK >= 1, dflash.selectorTopK <= vocabularySize else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription:
                    "dflash_config.selector_top_k \(dflash.selectorTopK) is outside 1...\(vocabularySize)."
            )
        }
        guard dflash.convKernelSize >= 1 else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription: "dflash_config.conv_kernel_size must be at least 1.")
        }
        guard dflash.convGroupSize >= 1, hiddenSize % dflash.convGroupSize == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .dflash, in: container,
                debugDescription:
                    "dflash_config.conv_group_size \(dflash.convGroupSize) must divide hidden_size \(hiddenSize)."
            )
        }
    }
}

// MARK: - The target binding

/// The target-model surface the DFlash 2 drafter binds to.
///
/// The drafter owns no embedding table and no output projection. Both of these
/// MUST call the target's MODULE. On a packed Hadamard pack a path that reads
/// the raw `.weight` skips the transform and returns wrong numbers while it
/// still type-checks.
public protocol DFlash2Target: AnyObject {
    var dFlash2VocabularySize: Int { get }
    var dFlash2HiddenSize: Int { get }

    /// The target's embedding table, applied through the target's module.
    func embedTokensForDFlash2(_ tokens: MLXArray) -> MLXArray
    /// The target's output projection, applied through the target's module. The
    /// drafter applies any config-level logit transform itself.
    func logitsForDFlash2Hidden(_ hidden: MLXArray) -> MLXArray
}

/// Where a tower keeps its tap state, deliberately OFF the module tree.
///
/// `Module` reflection classifies a stored property BY ITS DYNAMIC TYPE
/// (`Vendor/mlx-swift/Source/MLXNN/Module.swift`, `ModuleItem.build`):
///
/// ```swift
/// case let v as MLXArray:
///     return .value(.parameters(v))
/// ```
///
/// So a tapped hidden state stored directly on a `Module` becomes a PARAMETER
/// the moment it is non-nil. It would then join `parameters()`, and with it
/// strict checkpoint key verification, the quantization walk, and `eval(model)`
/// -- and on this track the loaded target is re-verified before every measured
/// window, so an unexpected parameter key is a refusal at measurement time. A
/// plain class is `.other`, which reflection ignores, so the tap lives here.
public final class DFlash2TapSlot {
    /// The layers whose OUTPUT hidden state the next forward keeps, in the
    /// order the drafter's `fc` expects them. Nil turns the tap off.
    public var layerIds: [Int]?

    /// The tapped layers of the last forward, fused along the feature axis.
    public var tappedHidden: MLXArray?

    public init() {}
}

/// A target that can hand out the OUTPUT hidden state of named layers.
///
/// The drafter never calls this. The ENGINE does: it asks the target for the
/// tapped layers of the positions the target has just consumed, and feeds them
/// to the drafter as the next block's context.
///
/// COST WHEN OFF. `dFlash2TapLayerIds` is nil unless a DFlash 2 drafter is
/// attached, and a nil id list means the tower does nothing beyond one
/// comparison per layer and retains nothing.
public protocol DFlash2TapTarget: DFlash2Target {
    /// The layers whose OUTPUT hidden state the next forward must keep, in the
    /// order the drafter's `fc` expects them. Nil turns the tap off.
    var dFlash2TapLayerIds: [Int]? { get set }

    /// The tapped layers of the LAST forward, fused along the feature axis:
    /// `[B, L, taps * hidden]`. Nil when the tap was off for that forward.
    var dFlash2TappedHidden: MLXArray? { get }

    /// How many layers the tower has, so the ids can be checked before a run.
    var dFlash2LayerCount: Int { get }
}

extension DFlash2TapTarget {
    /// Check the ids against this target and turn the tap on.
    ///
    /// A bad id is a refusal, not a clamp: a drafter reading the wrong layers
    /// would still produce tokens, and the only visible symptom would be a poor
    /// accept rate that looks like a property of the pairing.
    public func armDFlash2Tap(layerIds: [Int]) throws {
        guard !layerIds.isEmpty else {
            throw DFlash2Error.emptyTapLayerIds
        }
        guard Set(layerIds).count == layerIds.count else {
            throw DFlash2Error.duplicateTapLayerIds(layerIds)
        }
        let count = dFlash2LayerCount
        for layerId in layerIds where layerId < 0 || layerId >= count {
            throw DFlash2Error.targetLayerOutOfRange(layerId: layerId, layerCount: count)
        }
        dFlash2TapLayerIds = layerIds
    }
}

// MARK: - The sliding-window mask

/// The block attention mask the reference builds when a layer slides.
///
/// A DFlash 2 block is NOT causal inside itself: every block position sees
/// every other block position. Only the context half is windowed.
///
/// Reference (`DFlashAttention.__call__`):
/// ```python
/// query = ctx_len + mx.arange(L)[:, None]
/// key = mx.arange(ctx_len + L)[None]
/// context = (key < ctx_len) & (query - key < self.sliding_window)
/// block = key >= ctx_len
/// if self.is_causal:
///     block = block & (key <= query)
/// mask = context | block
/// ```
public enum DFlash2SlidingMask {
    /// `true` where attention is allowed. Shape `[blockLength, contextLength + blockLength]`.
    public static func make(
        contextLength: Int,
        blockLength: Int,
        slidingWindow: Int,
        isCausal: Bool
    ) -> MLXArray {
        let query = MLXArray(Int32(contextLength) ..< Int32(contextLength + blockLength))
            .reshaped(blockLength, 1)
        let key = MLXArray(Int32(0) ..< Int32(contextLength + blockLength))
            .reshaped(1, contextLength + blockLength)
        let context = (key .< Int32(contextLength)) .&& ((query - key) .< Int32(slidingWindow))
        var block = key .>= Int32(contextLength)
        if isCausal {
            block = block .&& (key .<= query)
        }
        return context .|| block
    }

    /// How many leading context rows the layer drops before it projects them,
    /// and the offset the cache advances by to stay aligned with the rows it
    /// keeps. `sliding_window - 1` rows survive.
    public static func contextSkip(contextLength: Int, slidingWindow: Int) -> Int {
        Swift.max(0, contextLength - (slidingWindow - 1))
    }
}

/// The sliding mask of ONE block forward, built once and handed to every layer
/// that asks for the same geometry.
///
/// Every layer of a forward sees the same context rows and the same block, so
/// its mask inputs are equal and the mask is the same array. Building it per
/// layer re-ran the same comparison graph once per layer. The memo is keyed on
/// every input of ``DFlash2SlidingMask/make(contextLength:blockLength:slidingWindow:isCausal:)``,
/// so a layer whose inputs differ still gets its own mask. It lives for one
/// forward only and is a plain class, off the module tree (see
/// ``DFlash2TapSlot``).
final class DFlash2SlidingMaskMemo {
    private var key: [Int]?
    private var cached: MLXArray?

    func mask(
        contextLength: Int,
        blockLength: Int,
        slidingWindow: Int,
        isCausal: Bool
    ) -> MLXArray {
        let requested = [contextLength, blockLength, slidingWindow, isCausal ? 1 : 0]
        if let cached, key == requested {
            return cached
        }
        let made = DFlash2SlidingMask.make(
            contextLength: contextLength,
            blockLength: blockLength,
            slidingWindow: slidingWindow,
            isCausal: isCausal)
        key = requested
        cached = made
        return made
    }
}

// MARK: - Attention

private final class DFlash2Attention: Module {
    let layerType: DFlash2LayerType
    let slidingWindow: Int?
    let isCausal: Bool
    let heads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    private let qkv = DFlash2QKVStack()

    init(_ config: DFlash2Configuration, layerIndex: Int) {
        self.layerType = config.layerTypes[layerIndex]
        self.slidingWindow = layerType == .slidingAttention ? config.slidingWindow : nil
        self.isCausal = config.isCausal
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: false)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        _oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        super.init()
    }

    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        qkv.clear()
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    /// - Parameters:
    ///   - x: the block, `[B, blockLength, hidden]`.
    ///   - context: the projected target hidden state, `[B, contextLength, hidden]`.
    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPELayer, cache: KVCache,
        masks: DFlash2SlidingMaskMemo
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        var context = context
        var contextLength = context.dim(1)

        if let slidingWindow {
            let skip = DFlash2SlidingMask.contextSkip(
                contextLength: contextLength, slidingWindow: slidingWindow)
            if skip > 0 {
                context = context[0..., skip..., 0...]
                contextLength = context.dim(1)
                // The dropped rows still happened, so the cache's notion of
                // where the block sits has to move with them.
                if let base = cache as? BaseKVCache {
                    base.offset += skip
                }
            }
        }

        // The block sits immediately after the context, so both the queries and
        // the block's own keys rotate at the context's far end.
        let blockOffset = cache.offset + contextLength

        let queries: MLXArray
        // The keys and values the block attends over: every cached context
        // row followed by the block's own rows.
        let keys: MLXArray
        let values: MLXArray
        let cachedLength: Int
        if dflash2KVConcatEnabled {
            // One K and one V projection over [context; block]. The block's
            // positions continue the context's, so one rope at the context's
            // offset rotates every row where the two separate ropes did.
            let rows = concatenated([context, x], axis: 1)
            let n = contextLength + L
            let projectedQ: MLXArray
            let projectedK: MLXArray
            let projectedV: MLXArray
            if let stacked = qkv.apply(rows, blockRows: L, q: qProj, k: kProj, v: vProj) {
                (projectedQ, projectedK, projectedV) = stacked
            } else {
                (projectedQ, projectedK, projectedV) = (qProj(x), kProj(rows), vProj(rows))
            }
            queries = rope(
                qNorm(projectedQ.reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3),
                offset: blockOffset)
            let allKeys = rope(
                kNorm(projectedK.reshaped(B, n, kvHeads, -1)).transposed(0, 2, 1, 3),
                offset: cache.offset)
            let allValues = projectedV.reshaped(B, n, kvHeads, -1).transposed(0, 2, 1, 3)
            if let block = cache as? DFlash2BlockKVCache,
                let held = block.updateBlock(
                    keys: allKeys, values: allValues, contextRows: contextLength)
            {
                // The context rows entered the cache in place and the block
                // rows sit right after them in the same buffer; nothing is
                // concatenated. `cachedLength` is the context the cache
                // holds, as the plain path's `cachedKeys.dim(2)`.
                (keys, values) = held
                cachedLength = keys.dim(2) - L
            } else {
                let contextKeys = allKeys[0..., 0..., ..<contextLength, 0...]
                let contextValues = allValues[0..., 0..., ..<contextLength, 0...]
                let blockKeys = allKeys[0..., 0..., contextLength..., 0...]
                let blockValues = allValues[0..., 0..., contextLength..., 0...]
                // Only the CONTEXT keys and values enter the cache. The block's
                // own keys and values are concatenated for this forward and
                // then dropped.
                let (cachedKeys, cachedValues) = cache.update(
                    keys: contextKeys, values: contextValues)
                cachedLength = cachedKeys.dim(2)
                keys = concatenated([cachedKeys, blockKeys], axis: 2)
                values = concatenated([cachedValues, blockValues], axis: 2)
            }
        } else {
            queries = rope(
                qNorm(qProj(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3),
                offset: blockOffset)
            let contextKeys = rope(
                kNorm(kProj(context).reshaped(B, contextLength, kvHeads, -1))
                    .transposed(0, 2, 1, 3),
                offset: cache.offset)
            let contextValues = vProj(context).reshaped(B, contextLength, kvHeads, -1)
                .transposed(0, 2, 1, 3)
            let blockKeys = rope(
                kNorm(kProj(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3),
                offset: blockOffset)
            let blockValues = vProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
            let (cachedKeys, cachedValues) = cache.update(
                keys: contextKeys, values: contextValues)
            cachedLength = cachedKeys.dim(2)
            keys = concatenated([cachedKeys, blockKeys], axis: 2)
            values = concatenated([cachedValues, blockValues], axis: 2)
        }

        var mask: MLXArray?
        if let slidingWindow {
            // Every query sits at cachedLength + i (i < L), so the context
            // term `query - key < window` holds for every context key once
            // cachedLength + L <= window, and a non-causal block term is all
            // true: the mask would allow everything.
            if !(dflash2NoMaskEnabled && !isCausal && cachedLength + L <= slidingWindow) {
                mask = masks.mask(
                    contextLength: cachedLength,
                    blockLength: L,
                    slidingWindow: slidingWindow,
                    isCausal: isCausal)
            }
        } else if isCausal {
            mask = createCausalMask(n: L, offset: cachedLength)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return DFlash2TensorMatmul.linear(oProj, output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// The q, k and v projections' weights stacked along the output axis, as
/// `DFlash2GateUpStack` stacks gate and up: the same BF16 bytes concatenated
/// once on first use, held off the module tree. One matmul over the
/// `[context; block]` rows replaces three (q over the block rows only, k and
/// v over every row); the two 8-threadgroup k and v launches join the q
/// launch in one 48-threadgroup pass over the stack. The q rows the
/// context would produce are computed and dropped, which costs nothing at
/// these widths. `DARKBLOOM_DFLASH2_STACK_QKV=0` keeps the three matmuls.
private final class DFlash2QKVStack {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH2_STACK_QKV"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()
    private var weight: MLXArray?
    private var qEnd = 0
    private var kEnd = 0

    func clear() {
        weight = nil
        qEnd = 0
        kEnd = 0
    }

    /// `(q(rows[-blockRows...]), k(rows), v(rows))` from one matmul, or nil
    /// when the stack does not apply.
    func apply(
        _ rows: MLXArray, blockRows: Int, q: Linear, k: Linear, v: Linear
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard Self.enabled, q.bias == nil, k.bias == nil, v.bias == nil,
            q.weight.ndim == 2, k.weight.ndim == 2, v.weight.ndim == 2,
            q.weight.dtype == k.weight.dtype, k.weight.dtype == v.weight.dtype,
            q.weight.dim(1) == k.weight.dim(1), k.weight.dim(1) == v.weight.dim(1),
            rows.ndim == 3, blockRows <= rows.dim(1)
        else { return nil }
        if weight == nil {
            weight = concatenated([q.weight, k.weight, v.weight], axis: 0)
            qEnd = q.weight.dim(0)
            kEnd = qEnd + k.weight.dim(0)
        }
        let y = matmul(rows, weight!.T)
        let n = rows.dim(1)
        return (
            y[0..., (n - blockRows)..., ..<qEnd],
            y[.ellipsis, qEnd ..< kEnd],
            y[.ellipsis, kEnd...]
        )
    }
}

/// Kill switch for the one-projection context+block K/V (default on).
private let dflash2KVConcatEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_KV_CONCAT"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// Kill switch for dropping a sliding mask that allows everything (default on).
private let dflash2NoMaskEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_NOMASK"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

// MARK: - The grouped dynamic causal convolution

/// A two-tap causal convolution over the block whose per-position, per-group
/// taps are predicted from the block itself.
///
/// Reference (`_grouped_dynamic_convolve`):
/// ```python
/// output[b, l, g, c] = sum_o (base[o][g * group + c] + dynamic[b, l, o, g])
///                            * hidden[b, l - o, g, c]
/// ```
/// with `hidden[b, l - o] = 0` when `l - o < 0`. The convolution carries no
/// state between blocks: a block is a fresh sequence, and context reaches the
/// block through attention only.
final class DFlash2GroupedDynamicCausalConv: Module {
    let kernelSize: Int
    let groupSize: Int
    let groups: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        self.groups = hiddenSize / groupSize
        // Tap 0 wraps the sub-layer input, tap 1 wraps its output.
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(
            hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    static func convolve(
        hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let hiddenSize = hidden.dim(2)
        let groups = hiddenSize / groupSize
        let kernelSize = base.dim(0)
        let blocks = hidden.reshaped(batch, length, groups, groupSize)
        let dynamic = dynamic.reshaped(batch, length, kernelSize, groups, 1)

        var output = MLXArray.zeros(like: blocks)
        for offset in 0 ..< kernelSize {
            let values =
                offset == 0
                ? blocks
                : concatenated(
                    [
                        MLXArray.zeros(
                            [batch, offset, groups, groupSize], dtype: hidden.dtype),
                        blocks[0..., ..<(length - offset), 0..., 0...],
                    ], axis: 1)
            let kernel = base[offset].reshaped(1, 1, groups, groupSize).asType(hidden.dtype)
            output = output + kernel * values
            output = output + dynamic[0..., 0..., offset, 0..., 0...] * values
        }
        return output.reshaped(hidden.shape)
    }

    /// ``convolve(hidden:dynamic:base:groupSize:)`` for tap `tap` as ONE
    /// kernel (`dflash2GroupedConvKernel`), plus `residual` when given, or
    /// nil when the inputs do not qualify (the caller keeps the op chain).
    ///
    /// The op chain issues ten element-wise launches per call (four products
    /// and sums per offset, and a two-input concatenate for the shifted
    /// block); the kernel computes the same element in one. Each element
    /// takes the chain's operations in the chain's order on the same
    /// `T`-typed values — `out = 0 + k*v`, `+ d*v`, then offset 1 — with
    /// every product and sum rounded to `T` as the separate launches round
    /// it, and contraction off. The residual sum is the layer's own
    /// `x + conv`. Bit-identical to the chain it replaces.
    private func fusedConvolve(
        _ hidden: MLXArray, projection: MLXArray, tap: Int, residual: MLXArray?
    ) -> MLXArray? {
        guard dflash2FusedConvEnabled, hidden.ndim == 3 else { return nil }
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let hiddenSize = hidden.dim(2)
        let dtype = hidden.dtype
        guard [DType.bfloat16, .float16, .float32].contains(dtype),
            batch > 0, length > 0, hiddenSize == groups * groupSize,
            hiddenSize % 256 == 0,
            projection.shape == [batch, length, 2 * kernelSize * groups],
            projection.dtype == dtype,
            baseKernel.shape == [2, kernelSize, hiddenSize], baseKernel.dtype == dtype,
            residual.map({ $0.shape == hidden.shape && $0.dtype == dtype }) ?? true
        else { return nil }
        let template: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("KS", kernelSize), ("GS", groupSize), ("TAP", tap),
        ]
        let grid = (hiddenSize, length, batch)
        if let residual {
            return dflash2GroupedConvResidualKernel(
                [hidden, projection, baseKernel, residual],
                template: template, grid: grid, threadGroup: (256, 1, 1),
                outputShapes: [hidden.shape], outputDTypes: [dtype])[0]
        }
        return dflash2GroupedConvKernel(
            [hidden, projection, baseKernel],
            template: template, grid: grid, threadGroup: (256, 1, 1),
            outputShapes: [hidden.shape], outputDTypes: [dtype])[0]
    }

    /// The first tap. Returns the convolved input and the dynamic-tap
    /// projection the matching ``finish(_:projection:residual:)`` needs.
    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let projection = kernelProjection(hidden)
        if let fused = fusedConvolve(hidden, projection: projection, tap: 0, residual: nil) {
            return (fused, projection)
        }
        let dynamic = projection
            .reshaped(hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return (
            Self.convolve(
                hidden: hidden,
                dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0],
                groupSize: groupSize),
            projection
        )
    }

    /// The second tap, over the sub-layer's output, added to the layer's
    /// residual stream: `residual + conv(hidden)`.
    func finish(_ hidden: MLXArray, projection: MLXArray, residual: MLXArray) -> MLXArray {
        if let fused = fusedConvolve(
            hidden, projection: projection, tap: 1, residual: residual)
        {
            return fused
        }
        let dynamic = projection
            .reshaped(hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return residual
            + Self.convolve(
                hidden: hidden, dynamic: dynamic[0..., 0..., 1, 0..., 0...],
                base: baseKernel[1], groupSize: groupSize)
    }
}

/// Kill switch for the one-launch grouped convolution (default on).
private let dflash2FusedConvEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH2_FUSED_CONV"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

/// One element of `DFlash2GroupedDynamicCausalConv.convolve` for tap `TAP`.
///
/// `h` is the block `[B, L, H]`, `dyn` the kernel projection `[B, L, 2*KS*G]`
/// (viewed `[B, L, 2, KS, G]`), `base` the base kernel `[2, KS, H]`. The
/// operations and their order are the op chain's: `out` starts at zero; for
/// each offset `o` the shifted value `v` (zero before the block start) is
/// multiplied by the base tap and added, then multiplied by the dynamic tap
/// and added. Every intermediate is a `T`, as each separate launch stores it.
private let dflash2GroupedConvHeader = """
    template <typename T, int KS, int GS, int TAP>
    inline T dflash2_grouped_conv(
        const device T* h, const device T* dyn, const device T* base,
        uint b, uint l, uint c, uint L, uint H) {
    #pragma clang fp contract(off)
      const uint G = H / GS;
      const uint g = c / GS;
      const size_t row = size_t(b) * L + l;
      T out = static_cast<T>(0.0f);
      for (int o = 0; o < KS; ++o) {
        const T v = (l >= uint(o)) ? h[(row - o) * H + c] : static_cast<T>(0.0f);
        const T kb = base[(size_t(TAP) * KS + o) * H + c];
        const T kv = kb * v;
        out = out + kv;
        const T d = dyn[row * (2 * KS * G) + (size_t(TAP) * KS + o) * G + g];
        const T dv = d * v;
        out = out + dv;
      }
      return out;
    }
    """

private let dflash2GroupedConvSource = """
    const uint c = thread_position_in_grid.x;
    const uint l = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint H = threads_per_grid.x;
    const uint L = threads_per_grid.y;
    out[(size_t(b) * L + l) * H + c] =
        dflash2_grouped_conv<T, KS, GS, TAP>(h, dyn, base, b, l, c, L, H);
    """

private let dflash2GroupedConvResidualSource = """
    #pragma clang fp contract(off)
    const uint c = thread_position_in_grid.x;
    const uint l = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint H = threads_per_grid.x;
    const uint L = threads_per_grid.y;
    const size_t i = (size_t(b) * L + l) * H + c;
    const T conv = dflash2_grouped_conv<T, KS, GS, TAP>(h, dyn, base, b, l, c, L, H);
    out[i] = res[i] + conv;
    """

private let dflash2GroupedConvKernel = MLXFast.metalKernel(
    name: "dflash2_grouped_conv",
    inputNames: ["h", "dyn", "base"],
    outputNames: ["out"],
    source: dflash2GroupedConvSource,
    header: dflash2GroupedConvHeader,
    ensureRowContiguous: true)

private let dflash2GroupedConvResidualKernel = MLXFast.metalKernel(
    name: "dflash2_grouped_conv_residual",
    inputNames: ["h", "dyn", "base", "res"],
    outputNames: ["out"],
    source: dflash2GroupedConvResidualSource,
    header: dflash2GroupedConvHeader,
    ensureRowContiguous: true)

// MARK: - The decoder layer

/// The gate and up projections' weights stacked along the output axis: the
/// same BF16 bytes as the two loaded weights, concatenated once on first use
/// and held in a plain class (never a stored `MLXArray` on the module, so
/// reflection cannot add it to the parameter tree). One matmul over the
/// stack replaces two over the same input, and the block's 16 rows fill one
/// wider tensor tile instead of two. `DARKBLOOM_DFLASH2_STACK_GATEUP=0` keeps
/// the two matmuls.
/// The drafter's BF16 projections at a block width (<= 16 rows) on the tensor
/// unit with `tensor` operands (`bfloat x bfloat -> float`, MetalPerformance-
/// Primitives `matmul2d`): each threadgroup owns 32 output columns, its four
/// simdgroups take four contiguous K quarters (one 16 x 32 x 256 op per
/// chunk) and their partials are summed through threadgroup memory. The
/// weights are the layer's BF16 arrays read as stored; the result is the same
/// FP32-accumulated product in a different summation order, rounded to BF16.
/// The drafter only proposes. `DARKBLOOM_DFLASH2_TENSOR_MATMUL=0` keeps the
/// core's GEMM.
enum DFlash2TensorMatmul {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH2_TENSOR_MATMUL"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let rowsPerTile = 16

    // The trailing newline matters: the JIT appends the kernel signature
    // directly after the header text.
    private static let header = """
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """

    // grid: (N / 32 * 128, 1, 1), threadgroup (128, 1, 1). Inputs: x bfloat
    // [16, K], w bfloat [N, K], ksz int32 [K, 16, N]. K % 1024 == 0.
    private static let source = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int kq = K / 4;
        const int k0 = int(sg) * kq;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 256, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> A((device bfloat*)x, dextents<int, 2>(K, M));
        tensor<device bfloat, dextents<int, 2>, tensor_inline> B((device bfloat*)w, dextents<int, 2>(K, N));
        auto tA0 = A.template slice<256, 16>(0, 0);
        auto tB0 = B.template slice<256, 32>(0, n0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB0)>, float>();
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) { cT[i] = 0.0f; }
        for (int k = k0; k < k0 + kq; k += 256) {
          auto tA = A.template slice<256, 16>(k, 0);
          auto tB = B.template slice<256, 32>(k, n0);
          op.run(tA, tB, cT);
        }
        // Destination layout: element i -> n = n0 + fn + (i & 3) + 16 * ((i >> 3) & 1),
        // m = fm + 8 * ((i >> 2) & 1).
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        threadgroup float red[3][16 * 32];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < 16; i++) { red[sg - 1][i * 32 + lane] = cT[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < 16; i++) {
            const float v = cT[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
            out[(size_t)(fm + 8 * mh) * N + n0 + fn + c + 16 * nh] = OutT(v);
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "dflash2_bf16_matmul_m16",
        inputNames: ["x", "w", "ksz"],
        outputNames: ["out"],
        source: source,
        header: header,
        ensureRowContiguous: true)

    private static let dimsLock = NSLock()
    nonisolated(unsafe) private static var dims: [[Int]: MLXArray] = [:]
    private static func dimsArray(k: Int, n: Int) -> MLXArray {
        dimsLock.withLock {
            if let cached = dims[[k, n]] { return cached }
            let array = MLXArray([Int32(k), Int32(rowsPerTile), Int32(n)])
            dims[[k, n]] = array
            return array
        }
    }

    /// `x @ weight.T` for a BF16 `x` of at most 16 rows and a BF16 `weight`
    /// `[N, K]`; nil when it does not apply.
    static func apply(_ x: MLXArray, weight: MLXArray) -> MLXArray? {
        guard enabled, Qwen35TensorPackedMatmul.tensorOperandsAvailable,
            x.dtype == .bfloat16, weight.dtype == .bfloat16, weight.ndim == 2, x.ndim >= 2
        else { return nil }
        let k = x.dim(-1)
        let rows = x.size / k
        let n = weight.dim(0)
        guard rows >= 1, rows <= rowsPerTile, weight.dim(1) == k, k % 1024 == 0, n % 32 == 0
        else { return nil }
        var a = x.reshaped(rows, k)
        if rows < rowsPerTile {
            a = concatenated(
                [a, MLXArray.zeros([rowsPerTile - rows, k], dtype: .bfloat16)], axis: 0)
        }
        let y = kernel(
            [a, weight, dimsArray(k: k, n: n)], template: [("OutT", DType.bfloat16)],
            grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[rowsPerTile, n]], outputDTypes: [.bfloat16])[0]
        let rowsOut = rows < rowsPerTile ? y[0 ..< rows] : y
        return rowsOut.reshaped(Array(x.shape.dropLast()) + [n])
    }

    /// `layer(x)` through the tensor kernel when it applies (no bias).
    static func linear(_ layer: Linear, _ x: MLXArray) -> MLXArray {
        if layer.bias == nil, let y = apply(x, weight: layer.weight) {
            return y
        }
        return layer(x)
    }
}

private final class DFlash2GateUpStack {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH2_STACK_GATEUP"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()
    private var weight: MLXArray?
    private var boundary = 0

    func clear() {
        weight = nil
        boundary = 0
    }

    /// `(gate(x), up(x))` from one matmul, or nil when the stack does not apply.
    func apply(_ x: MLXArray, gate: Linear, up: Linear) -> (MLXArray, MLXArray)? {
        guard Self.enabled, gate.bias == nil, up.bias == nil,
            gate.weight.dtype == up.weight.dtype, gate.weight.dim(1) == up.weight.dim(1),
            gate.weight.ndim == 2, up.weight.ndim == 2
        else { return nil }
        if weight == nil {
            weight = concatenated([gate.weight, up.weight], axis: 0)
            boundary = gate.weight.dim(0)
        }
        let y = DFlash2TensorMatmul.apply(x, weight: weight!) ?? matmul(x, weight!.T)
        return (y[.ellipsis, ..<boundary], y[.ellipsis, boundary...])
    }
}

private final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    private let gateUp = DFlash2GateUpStack()

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        gateUp.clear()
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let (g, u) = gateUp.apply(x, gate: gate, up: up) {
            return DFlash2TensorMatmul.linear(down, silu(g) * u)
        }
        return DFlash2TensorMatmul.linear(down, silu(gate(x)) * up(x))
    }
}

private final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlash2Attention
    @ModuleInfo var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2GroupedDynamicCausalConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2GroupedDynamicCausalConv

    init(_ config: DFlash2Configuration, layerIndex: Int) {
        _selfAttn.wrappedValue = DFlash2Attention(config, layerIndex: layerIndex)
        _mlp.wrappedValue = DFlash2MLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionConv.wrappedValue = DFlash2GroupedDynamicCausalConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        _mlpConv.wrappedValue = DFlash2GroupedDynamicCausalConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPELayer, cache: KVCache,
        masks: DFlash2SlidingMaskMemo
    ) -> MLXArray {
        let (attentionInput, attentionTaps) = attentionConv.prepare(inputLayerNorm(x))
        let attended = attentionConv.finish(
            selfAttn(
                attentionInput, context: context, rope: rope, cache: cache,
                masks: masks),
            projection: attentionTaps, residual: x)
        let (mlpInput, mlpTaps) = mlpConv.prepare(postAttentionLayerNorm(attended))
        return mlpConv.finish(mlp(mlpInput), projection: mlpTaps, residual: attended)
    }
}

// MARK: - The candidate selector

/// The low-rank edge-scored greedy path over the per-position candidate lists.
///
/// The track is greedy, so only the temperature-0 path of the reference
/// `CandidateSelector.select` is ported. The sampling path is not needed and is
/// not here.
final class DFlash2CandidateSelector: Module {
    let topK: Int

    @ParameterInfo(key: "predecessor_codebook") var predecessorCodebook: MLXArray
    @ParameterInfo(key: "successor_codebook") var successorCodebook: MLXArray
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    init(_ config: DFlash2Configuration) {
        self.topK = config.dflash.selectorTopK
        _predecessorCodebook.wrappedValue = MLXArray.zeros([
            config.vocabularySize, config.dflash.selectorRank,
        ])
        _successorCodebook.wrappedValue = MLXArray.zeros([
            config.vocabularySize, config.dflash.selectorRank,
        ])
        _hiddenProjection.wrappedValue = Linear(
            config.hiddenSize, config.dflash.selectorRank, bias: false)
        super.init()
    }

    /// The greedy path.
    ///
    /// - Parameters:
    ///   - hidden: the drafter's final hidden state, `[B, L, hidden]`.
    ///   - logits: the drafter's logits over the same positions, `[B, L, vocab]`.
    ///   - anchor: the token each path starts from, `[B]`.
    /// - Returns: the selected token at each position, `[B, L]`.
    func selectGreedy(hidden: MLXArray, logits: MLXArray, anchor: MLXArray) -> MLXArray {
        let vocabularySize = logits.dim(-1)
        let candidates: MLXArray
        let unary: MLXArray
        if let top = DFlash2TopK.select(logits, k: topK) {
            (candidates, unary) = top
        } else {
            candidates = argPartition(logits, kth: vocabularySize - topK, axis: -1)[
                0..., 0..., (vocabularySize - topK)...]
            unary = takeAlong(logits, candidates, axis: -1)
        }
        let projected = hiddenProjection(hidden)

        if let path = DFlash2GreedyWalk.select(
            candidates: candidates, unary: unary, projected: projected, anchor: anchor,
            predecessorCodebook: predecessorCodebook, successorCodebook: successorCodebook)
        {
            return path
        }

        var predecessor = anchor
        var path = [MLXArray]()
        path.reserveCapacity(hidden.dim(1))
        for position in 0 ..< hidden.dim(1) {
            let positionCandidates = candidates[0..., position, 0...]
            let edges =
                (take(predecessorCodebook, predecessor, axis: 0)
                    .expandedDimensions(axis: 1)
                    * projected[0..., position, 0...].expandedDimensions(axis: 1)
                    * take(successorCodebook, positionCandidates, axis: 0))
                .sum(axis: -1)
            let selected = (unary[0..., position, 0...] + edges).argMax(axis: -1)
            predecessor = takeAlong(
                positionCandidates, selected.expandedDimensions(axis: -1), axis: -1)[0..., 0]
            path.append(predecessor)
        }
        // `argPartition` indexes in UInt32, so the path inherits that dtype.
        // Draft tokens are token ids, and the engine reads them as Int32.
        return stacked(path, axis: 1).asType(.int32)
    }
}

/// The top-`k` logits of each row, as `argPartition` + `takeAlong` return them.
///
/// `argPartition` runs MLX's stable ascending merge sort, so its last `k`
/// slots hold the `k` largest values in ascending order, equal values in
/// index order (the higher index ranks higher), NaN above everything. Two
/// launches reproduce that exactly: each threadgroup reduces one chunk of a
/// row to its top `k` by (value, index), then one simdgroup per row merges
/// the chunk lists and writes them in the sort's slot order, gathering the
/// values from the logits. Candidates and unary scores are therefore the
/// same arrays, element for element.
enum DFlash2TopK {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_TOPK_KERNEL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let chunks = 8
    private static let threads = 128

    static func select(_ logits: MLXArray, k: Int) -> (MLXArray, MLXArray)? {
        guard enabled, logits.ndim == 3, logits.dim(0) == 1,
            logits.dtype == .float32 || logits.dtype == .float16
        else { return nil }
        let rows = logits.dim(1)
        let vocabularySize = logits.dim(2)
        guard rows >= 1, k >= 1, k <= 32, vocabularySize >= k,
            vocabularySize < Int(Int32.max)
        else { return nil }
        let flat = logits.reshaped([rows, vocabularySize])
        let template: [(String, any KernelTemplateArg)] = [
            ("NV", vocabularySize), ("S", chunks), ("TPG", threads), ("KTOP", k),
        ]
        let parts = chunkKernel(
            [flat], template: template,
            grid: (threads * chunks, rows, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, chunks, k], [rows, chunks, k]],
            outputDTypes: [.uint32, .uint32])
        let merged = mergeKernel(
            [flat, parts[0], parts[1]], template: template,
            grid: (32, rows, 1), threadGroup: (32, 1, 1),
            outputShapes: [[rows, k], [rows, k]],
            outputDTypes: [.uint32, .float32])
        return (merged[0].reshaped([1, rows, k]), merged[1].reshaped([1, rows, k]))
    }

    private static let header = """
            // Order-preserving key: larger float -> larger key; every NaN -> 0xffffffff
            // (the sort's LessThan puts NaN above +inf); -0 and +0 share a key (they
            // compare equal). Key 0 is never produced by data and marks an empty slot.
            inline uint mlxfast_topk_key(float x) {
                if (isnan(x)) return 0xffffffffu;
                uint u = as_type<uint>(x == 0.0f ? 0.0f : x);
                return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            }
            // Entries rank by (key, idx): the stable ascending sort keeps equal values in
            // index order, so among ties the higher index ranks higher.
            // KK rounds of simdgroup max over per-lane descending lists; each winner pops
            // its head. Lane r receives the r-th ranked entry (r < KK).
            template <int KK>
            inline void mlxfast_topk_simd_merge(thread uint (&k)[KK], thread uint (&id)[KK],
                                                uint lane, thread uint& outk, thread uint& outi) {
                for (int r = 0; r < KK; r++) {
                    uint km = simd_max(k[0]);
                    uint im = simd_max(k[0] == km ? id[0] : 0u);
                    if (lane == uint(r)) { outk = km; outi = im; }
                    if (k[0] == km && id[0] == im) {
                        for (int j = 0; j < KK - 1; j++) { k[j] = k[j + 1]; id[j] = id[j + 1]; }
                        k[KK - 1] = 0u; id[KK - 1] = 0u;
                    }
                }
            }
            """

    private static let chunkKernel = MLXFast.metalKernel(
        name: "mlxfast_dflash_topk_chunk",
        inputNames: ["logits"],
        outputNames: ["part_key", "part_idx"],
        source: """
            // grid (TPG * S, rows): threadgroup (chunk, row) reduces one chunk of a row to its top-KK
            constexpr int KK = KTOP;
            static_assert(TPG % 32 == 0 && TPG / 32 <= 32 && KK <= 32, "one merge simdgroup");
            const uint t = thread_position_in_threadgroup.x;
            const uint chunk = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            constexpr uint CH = (NV + S - 1) / S;
            const uint lo = chunk * CH;
            const uint hi = min(lo + CH, uint(NV));
            // `logits` is float or half (the drafter's FP16 head read); the
            // key and the gathered value are the float the half widens to.
            auto x = logits + size_t(row) * NV;
            uint k[KK], id[KK];
            for (int j = 0; j < KK; j++) { k[j] = 0u; id[j] = 0u; }
            for (uint v = lo + t; v < hi; v += TPG) {
                uint kx = mlxfast_topk_key(x[v]);
                if (kx >= k[KK - 1]) {   // this thread visits indices in increasing order
                    uint ix = v;
                    for (int j = 0; j < KK; j++) {
                        bool sw = kx >= k[j];
                        uint tk = k[j], ti = id[j];
                        k[j] = sw ? kx : tk; id[j] = sw ? ix : ti;
                        kx = sw ? tk : kx; ix = sw ? ti : ix;
                    }
                }
            }
            uint ok = 0u, oi = 0u;
            mlxfast_topk_simd_merge<KK>(k, id, lane, ok, oi);
            constexpr uint NSG = TPG / 32;
            threadgroup uint sk[NSG * KK], si[NSG * KK];
            if (lane < uint(KK)) { sk[sg * KK + lane] = ok; si[sg * KK + lane] = oi; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                for (int j = 0; j < KK; j++) { k[j] = 0u; id[j] = 0u; }
                if (lane < NSG) {
                    for (int j = 0; j < KK; j++) { k[j] = sk[lane * KK + j]; id[j] = si[lane * KK + j]; }
                }
                mlxfast_topk_simd_merge<KK>(k, id, lane, ok, oi);
                if (lane < uint(KK)) {
                    size_t o = (size_t(row) * S + chunk) * KK + lane;
                    part_key[o] = ok; part_idx[o] = oi;
                }
            }
            """,
        header: header)

    private static let mergeKernel = MLXFast.metalKernel(
        name: "mlxfast_dflash_topk_merge",
        inputNames: ["logits", "part_key", "part_idx"],
        outputNames: ["cand", "val"],
        source: """
            // one simdgroup per row merges the S chunk lists (S <= 32), then writes the
            // top-KK in the sort's ascending order, with the values gathered from logits.
            constexpr int KK = KTOP;
            static_assert(S <= 32 && KK <= 32, "one merge simdgroup");
            const uint lane = thread_index_in_simdgroup;
            const uint row = threadgroup_position_in_grid.y;
            uint k[KK], id[KK];
            for (int j = 0; j < KK; j++) { k[j] = 0u; id[j] = 0u; }
            if (lane < uint(S)) {
                for (int j = 0; j < KK; j++) {
                    size_t o = (size_t(row) * S + lane) * KK + j;
                    k[j] = part_key[o]; id[j] = part_idx[o];
                }
            }
            uint ok = 0u, oi = 0u;
            mlxfast_topk_simd_merge<KK>(k, id, lane, ok, oi);
            if (lane < uint(KK)) {
                uint slot = KK - 1 - lane;
                uint ii = min(oi, uint(NV - 1));
                cand[row * KK + slot] = ii;
                val[row * KK + slot] = logits[size_t(row) * NV + ii];
            }
            """,
        header: header)
}

/// The greedy candidate walk with every edge score computed up front.
///
/// Position 0 scores its candidates against the anchor; every later position
/// scores its candidates against each candidate of the position before it, as
/// one `[L-1, K, K]` table built from the same element-wise products and the
/// same final-axis sum as the per-position loop. One small kernel then walks
/// the table in order, so the selected path is the loop's path.
enum DFlash2GreedyWalk {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_FUSED_WALK"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static func select(
        candidates: MLXArray, unary: MLXArray, projected: MLXArray, anchor: MLXArray,
        predecessorCodebook: MLXArray, successorCodebook: MLXArray
    ) -> MLXArray? {
        guard enabled, candidates.ndim == 3, candidates.dim(0) == 1, anchor.size == 1,
            unary.dtype == .float32
        else { return nil }
        let length = candidates.dim(1)
        let k = candidates.dim(2)
        let rank = projected.dim(-1)
        guard length >= 2, k >= 1, k <= 32, rank > 0 else { return nil }
        let c = candidates[0]
        // Gather only the codebook rows the candidate lists can visit. The
        // fused kernel scores each edge and advances the greedy walk in one
        // pass, instead of materializing an [L-1, K, K, rank] broadcast.
        let anchorPredecessor = take(predecessorCodebook, anchor, axis: 0)
            .asType(.float32).reshaped([-1])
        let previous = take(predecessorCodebook, c[0 ..< (length - 1)], axis: 0)
            .asType(.float32).reshaped([-1])
        let next = take(successorCodebook, c, axis: 0).asType(.float32).reshaped([-1])
        let projectedRows = projected[0].asType(.float32).reshaped([-1])
        let scores = unary[0].asType(.float32).reshaped([-1])
        let candidateIds = c.asType(.uint32).reshaped([-1])
        let path = kernel(
            [anchorPredecessor, previous, next, projectedRows, scores, candidateIds],
            template: [("L", length), ("K", k), ("R", rank)],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[length]],
            outputDTypes: [.int32])[0]
        return path.reshaped([1, length])
    }

    private static let kernel = MLXFast.metalKernel(
        name: "mlxfast_dflash_fused_greedy_walk",
        inputNames: [
            "anchor_predecessor", "previous", "next", "projected", "unary", "cand",
        ],
        outputNames: ["path"],
        source: """
            uint c = thread_index_in_simdgroup;
            uint previous_slot = 0;
            for (uint i = 0; i < L; i++) {
                float score = -INFINITY;
                if (c < K) {
                    const uint pred_base = i == 0 ? 0 : ((i - 1) * K + previous_slot) * R;
                    const uint succ_base = (i * K + c) * R;
                    const device float* pred_ptr = (i == 0) ? anchor_predecessor : (previous + pred_base);
                    const device float* proj_ptr = projected + i * R;
                    const device float* succ_ptr = next + succ_base;
                    float edge = 0.0f;
                    #pragma clang loop unroll(full)
                    for (uint d = 0; d < R; d++) {
                        edge += (pred_ptr[d] * proj_ptr[d]) * succ_ptr[d];
                    }
                    score = unary[i * K + c] + edge;
                }
                float m = simd_max(score);
                uint sel = simd_min((c < K && score == m) ? c : 0xffffffffu);
                previous_slot = sel;
                if (c == 0) path[i] = int(cand[i * K + sel]);
            }
            """)
}

// MARK: - The drafter

public final class DFlash2DraftModel: Module, @unchecked Sendable {
    public let config: DFlash2Configuration

    @ModuleInfo(key: "fc") public var fc: Linear
    @ModuleInfo(key: "hidden_norm") public var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") private var layers: [DFlash2DecoderLayer]
    @ModuleInfo public var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlash2CandidateSelector

    private let rope: RoPELayer
    private var target: (any DFlash2Target)?

    /// The drafter's own parameter dtype. The Bonsai trunk runs its norms in
    /// FP32 and hands out FP32 activations, so the two tensors that cross from
    /// the target into the drafter — the embedded block and the fused target
    /// hidden state — are cast to this before they reach a drafter weight.
    public var dtype: DType { norm.weight.dtype }

    public init(config: DFlash2Configuration) {
        self.config = config
        _fc.wrappedValue = Linear(config.targetHiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            DFlash2DecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _candidateSelector.wrappedValue = DFlash2CandidateSelector(config)
        self.rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    // MARK: Binding

    public func bind(target: any DFlash2Target) throws {
        guard target.dFlash2VocabularySize == config.vocabularySize else {
            throw DFlash2Error.vocabularyMismatch(
                drafter: config.vocabularySize, target: target.dFlash2VocabularySize)
        }
        guard target.dFlash2HiddenSize == config.hiddenSize else {
            throw DFlash2Error.hiddenSizeMismatch(
                drafter: config.hiddenSize, target: target.dFlash2HiddenSize)
        }
        self.target = target
    }

    // MARK: The cache

    public func makeCache() throws -> [KVCache] {
        try config.layerTypes.map { layerType in
            switch layerType {
            case .fullAttention:
                return StandardKVCache()
            case .slidingAttention:
                guard let slidingWindow = config.slidingWindow else {
                    throw DFlash2Error.missingSlidingWindow
                }
                if DFlash2BlockKVCache.enabled {
                    return DFlash2BlockKVCache(maxSize: slidingWindow - 1, keep: 0)
                }
                return RotatingKVCache(maxSize: slidingWindow - 1, keep: 0)
            }
        }
    }

    /// The number of context rows the drafter can use. A drafter whose layers
    /// all slide keeps `sliding_window - 1` rows, so the engine never needs to
    /// hand it more than that at prefill.
    public var contextRowLimit: Int? {
        guard config.layerTypes.allSatisfy({ $0 == .slidingAttention }),
            let slidingWindow = config.slidingWindow
        else {
            return nil
        }
        return slidingWindow - 1
    }

    /// Align the drafter's cache with the number of tokens the target has
    /// committed.
    ///
    /// Under the block protocol the engine feeds exactly the accepted positions'
    /// hidden states each round, so the cache offset already tracks the
    /// committed length and this trims nothing. It is here because the
    /// reference keeps the same guard.
    public func trimCache(_ cache: [KVCache], toCommittedLength committed: Int) {
        guard let first = cache.first else { return }
        let excess = first.offset - committed
        guard excess > 0 else { return }
        for c in cache {
            c.trim(excess)
        }

    }

    // MARK: The forward pass

    /// The drafter trunk over one block.
    ///
    /// - Parameters:
    ///   - inputs: the block's token ids, `[B, blockLength]`.
    ///   - targetHidden: the fused target hidden state, `[B, contextLength, targetHiddenSize]`.
    ///   - logitsStart: how many leading block positions to drop before the head.
    func hiddenStates(
        _ inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [KVCache],
        logitsStart: Int
    ) throws -> MLXArray {
        guard let target else { throw DFlash2Error.notBound }
        guard cache.count == layers.count else {
            throw DFlash2Error.invalidCacheCount(expected: layers.count, actual: cache.count)
        }
        guard targetHidden.dim(-1) == config.targetHiddenSize else {
            throw DFlash2Error.targetHiddenSizeMismatch(
                expected: config.targetHiddenSize, actual: targetHidden.dim(-1))
        }

        // Both crossings from the target cast here. The target's embedding is a
        // MODULE call, never a raw weight read.
        var h = target.embedTokensForDFlash2(inputs).asType(dtype)
        if config.dflash.inputEmbeddingScale != 1 {
            h = h * config.dflash.inputEmbeddingScale
        }
        let context = hiddenNorm(DFlash2TensorMatmul.linear(fc, targetHidden.asType(dtype)))

        let masks = DFlash2SlidingMaskMemo()
        for (index, layer) in layers.enumerated() {
            h = layer(h, context: context, rope: rope, cache: cache[index], masks: masks)
        }
        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }
        return norm(h)
    }

    func logits(_ hidden: MLXArray) throws -> MLXArray {
        guard let target else { throw DFlash2Error.notBound }
        var logits = target.logitsForDFlash2Hidden(hidden)
        if config.dflash.outputMultiplier != 1 {
            logits = logits * config.dflash.outputMultiplier
        }
        if let cap = config.dflash.finalLogitSoftcapping, cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    // MARK: Proposing

    /// Draft `blockSize - 1` tokens in one forward.
    ///
    /// The block is the last committed token followed by `blockSize - 1` mask
    /// tokens. The head reads the mask positions only, and the greedy candidate
    /// path turns their logits into the draft.
    ///
    /// - Parameters:
    ///   - anchor: the last committed token, one per row.
    ///   - targetHidden: the fused target hidden state of the positions the
    ///     target has already consumed, `[B, contextLength, targetHiddenSize]`.
    /// - Returns: the draft tokens, `[B, blockSize - 1]`.
    public func propose(
        anchor: [Int],
        targetHidden: MLXArray,
        cache: [KVCache],
        blockSize: Int
    ) throws -> MLXArray {
        guard blockSize >= 2 else { throw DFlash2Error.invalidBlockSize(blockSize) }
        let masks = Array(repeating: Int32(config.maskTokenId), count: blockSize - 1)
        let rows = anchor.flatMap { [Int32($0)] + masks }
        let block = MLXArray(rows, [anchor.count, blockSize])

        let hidden = try hiddenStates(
            block, targetHidden: targetHidden, cache: cache, logitsStart: 1)
        return candidateSelector.selectGreedy(
            hidden: hidden,
            logits: try logits(hidden),
            anchor: MLXArray(anchor.map { Int32($0) }))
    }

    // MARK: Loading

    public static func isDFlash2Directory(_ directory: URL) -> Bool {
        let configURL = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: configURL),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return false
        }
        let architectures = object["architectures"] as? [String] ?? []
        return architectures.contains("DFlash2DraftModel") && object["dflash_config"] != nil
    }

    public static func loadConfiguration(from directory: URL) throws -> DFlash2Configuration {
        let configURL = directory.appending(component: "config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw DFlash2Error.missingConfig(directory.path)
        }
        return try JSONDecoder().decode(
            DFlash2Configuration.self, from: Data(contentsOf: configURL))
    }

    /// Load the drafter from a staged directory.
    ///
    /// The weights load with `verify: [.all]`, so a checkpoint whose key set
    /// does not match this model exactly is a refusal rather than a silent
    /// partial load. The drafter carries no `embed_tokens` and no `lm_head`; it
    /// binds the target's.
    public static func load(from directory: URL) throws -> DFlash2DraftModel {
        let config = try loadConfiguration(from: directory)
        let drafter = DFlash2DraftModel(config: config)

        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)
        else {
            throw DFlash2Error.unreadableDirectory(directory.path)
        }
        var weights = [String: MLXArray]()
        for url in entries where url.pathExtension == "safetensors" {
            weights.merge(try loadArrays(url: url)) { _, new in new }
        }

        try drafter.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(drafter)
        return drafter
    }
}
