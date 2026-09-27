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
        // The comparison graph is the same for every layer of every later
        // round with this geometry. Realize it once, here, so those rounds
        // read the stored mask instead of replaying the graph.
        eval(made)
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

    /// The weights a block forward reads here in place of stored ones (the
    /// q|k|v stack once built, o_proj as `DFlash2TensorMatmul` reads it), and
    /// the stored weights they stand in for (`DFlash2ResidencyPrefetch`).
    func residencyWeights() -> (read: [MLXArray], replaced: [MLXArray]) {
        var read: [MLXArray] = []
        var replaced: [MLXArray] = []
        if let stacked = qkv.residentWeight {
            read.append(stacked)
            replaced += [qProj.weight, kProj.weight, vProj.weight]
        }
        let o = DFlash2TensorMatmul.readWeight(oProj.weight)
        read.append(o)
        if o !== oProj.weight { replaced.append(oProj.weight) }
        return (read, replaced)
    }

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
    ///   - context: the projected target hidden state, `[B, contextLength, hidden]`,
    ///     or nil when this round brings no new context rows (the cache already
    ///     holds every committed row; see `absorbContext`).
    func callAsFunction(
        _ x: MLXArray, context: MLXArray?, joined: MLXArray? = nil, rope: RoPELayer,
        cache: KVCache, masks: DFlash2SlidingMaskMemo
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        var context = context
        var contextLength = context?.dim(1) ?? 0

        if let slidingWindow, let rows = context {
            let skip = DFlash2SlidingMask.contextSkip(
                contextLength: contextLength, slidingWindow: slidingWindow)
            if skip > 0 {
                context = rows[0..., skip..., 0...]
                contextLength = context!.dim(1)
                // The dropped rows still happened, so the cache's notion of
                // where the block sits has to move with them.
                if let base = cache as? BaseKVCache {
                    base.offset += skip
                }
            }
        }
        precondition(
            context != nil || (dflash2KVConcatEnabled && cache is DFlash2BlockKVCache),
            "DFlash 2: a block without context rows needs the in-place block cache")

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
            let rows = joined ?? context.map { DFlash2Concat.concatenate([$0, x], axis: 1) } ?? x
            let n = contextLength + L
            let projectedQ: MLXArray
            let projectedK: MLXArray
            let projectedV: MLXArray
            var product: MLXArray?
            if let stacked = qkv.apply(rows, blockRows: L, q: qProj, k: kProj, v: vProj) {
                (projectedQ, projectedK, projectedV, product) = stacked
            } else {
                (projectedQ, projectedK, projectedV) = (qProj(x), kProj(rows), vProj(rows))
            }
            let allKeys: MLXArray
            if let product, let (q, k) = DFlash2QKPrework.apply(
                product, blockRows: L, heads: heads, qNorm: qNorm, kNorm: kNorm,
                offsets: [blockOffset, cache.offset])
            {
                (queries, allKeys) = (q, k)
            } else {
                queries = rope(
                    DFlash2StridedRMSNorm.apply(qNorm, projectedQ.reshaped(B, L, heads, -1))
                        .transposed(0, 2, 1, 3),
                    offset: blockOffset)
                allKeys = rope(
                    DFlash2StridedRMSNorm.apply(kNorm, projectedK.reshaped(B, n, kvHeads, -1))
                        .transposed(0, 2, 1, 3),
                    offset: cache.offset)
            }
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
            let context = context!
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

extension DFlash2Attention {
    /// Whether a block forward prepends `context` unchanged (one projection
    /// over `[context; block]`, no row skipped), so the caller may hand it
    /// those rows already joined.
    func joins(_ context: MLXArray?) -> Bool {
        guard dflash2KVConcatEnabled, let context else { return false }
        return slidingWindow.map {
            DFlash2SlidingMask.contextSkip(contextLength: context.dim(1), slidingWindow: $0) == 0
        } ?? true
    }

    /// Write the keys and values of `context` (`[B, contextLength, hidden]`,
    /// the projected target hidden state of committed positions) into this
    /// layer's cache, exactly as a block forward over the same context would,
    /// without a block. A layer's context keys and values depend on the
    /// context alone (the block reads them through attention), so they can
    /// enter the cache before the block's anchor is known. Returns false,
    /// having written nothing, when the in-place cache cannot take them.
    ///
    /// BIT-EXACT: the projections come from the stacked q|k|v weight the
    /// block forward multiplies `[context; block]` by (`qkv.applyContext`),
    /// not from `kProj`/`vProj`. MLX routes a separate `[512, 5120] x
    /// [5120, 1024]` projection to its split-K GEMM (M4 Max: 32 x 64 output
    /// tiles <= 2048 with K >= max(M, N); M5: the NAX split-K, K >= 3 max(M,
    /// N)), whose partial sums round differently from the regular GEMM the
    /// 6144-wide stack takes (N > K rules both split-K routes out), so the
    /// absorbed rows were off by an ulp in ~130 of 524288 elements per layer
    /// and the drafter's proposals drifted from the unprefetched path's. The
    /// regular GEMM's rows do not depend on the row count, so a regular GEMM
    /// over the context alone gives the rows the stack over `[context;
    /// block]` gave. (With the stack switched off,
    /// `DARKBLOOM_DFLASH2_STACK_QKV=0`, the separate projections remain and
    /// carry no such guarantee.)
    func absorbContext(_ context: MLXArray, rope: RoPELayer, cache: KVCache) -> Bool {
        let B = context.dim(0)
        let contextLength = context.dim(1)
        guard let block = cache as? DFlash2BlockKVCache,
            block.canAbsorb(contextRows: contextLength)
        else { return false }
        if let slidingWindow {
            guard
                DFlash2SlidingMask.contextSkip(
                    contextLength: contextLength, slidingWindow: slidingWindow) == 0
            else { return false }
        }
        let (projectedK, projectedV) =
            qkv.applyContext(context, q: qProj, k: kProj, v: vProj)
            ?? (kProj(context), vProj(context))
        let keys = rope(
            DFlash2StridedRMSNorm.apply(
                kNorm, projectedK.reshaped(B, contextLength, kvHeads, -1))
                .transposed(0, 2, 1, 3),
            offset: cache.offset)
        let values = projectedV.reshaped(B, contextLength, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        return block.updateBlock(keys: keys, values: values, contextRows: contextLength) != nil
    }

    /// The stacked q|k|v weight the 32-row tensor kernel reads, if any.
    func stackedQKVWeight() -> MLXArray? { qkv.stackWeight(q: qProj, k: kProj, v: vProj) }

    var speculativeCapable: Bool {
        dflash2KVConcatEnabled && !isCausal && slidingWindow != nil
            && qkv.applies(q: qProj, k: kProj, v: vProj)
    }

    /// The block forward for a device-valued confirmed count `c`: `base`
    /// (projected verify context, then zero rows) with the block written at
    /// row `c` puts every row of today's `[context; block]` matmul at its row;
    /// all `2L` key/value rows go to the cursor, the queries are rows `c..<c+L`
    /// and the key tail past `held + c + L` is masked (exact zeros).
    func speculative(
        _ x: MLXArray, base: MLXArray, confirmed: MLXArray, queryOffset: MLXArray,
        rope: RoPELayer, cache: DFlash2BlockKVCache, keyMask: MLXArray
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray)? {
        let (B, L, n) = (x.dim(0), x.dim(1), base.dim(1))
        guard B == 1, n == 2 * L, let held = cache.inPlaceRows else { return nil }
        let start = confirmed.reshaped([1])
        let rows = dynamicSliceUpdate(base, update: x, start: start, axes: [1])
        guard case let (y, qEnd, kEnd)? = qkv.applyStacked(rows, q: qProj, k: kProj, v: vProj)
        else { return nil }
        let blockRows = dynamicSlice(
            y, start: start, axes: [1], sliceSize: [Int32(B), Int32(L), Int32(y.dim(2))])
        let queries = rope(
            qNorm(blockRows[.ellipsis, ..<qEnd].reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3),
            offset: queryOffset)
        let allKeys = rope(
            kNorm(y[.ellipsis, qEnd ..< kEnd].reshaped(B, n, kvHeads, -1)).transposed(0, 2, 1, 3),
            offset: cache.offset)
        let allValues = y[.ellipsis, kEnd...].reshaped(B, n, kvHeads, -1).transposed(0, 2, 1, 3)
        guard case let (keys, values)? = cache.speculativeRows(keys: allKeys, values: allValues)
        else { return nil }
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys[.ellipsis, ..<(held + n), 0...],
            values: values[.ellipsis, ..<(held + n), 0...], scale: scale, mask: keyMask)
        return (
            DFlash2TensorMatmul.linear(oProj, output.transposed(0, 2, 1, 3).reshaped(B, L, -1)),
            keys, values)
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

    /// The stacked weight once built (nil before its first use).
    var residentWeight: MLXArray? { weight }

    /// The stacked weight, concatenated on first use, or nil when the stack
    /// does not apply to these projections.
    private func stacked(q: Linear, k: Linear, v: Linear) -> MLXArray? {
        guard Self.enabled, q.bias == nil, k.bias == nil, v.bias == nil,
            q.weight.ndim == 2, k.weight.ndim == 2, v.weight.ndim == 2,
            q.weight.dtype == k.weight.dtype, k.weight.dtype == v.weight.dtype,
            q.weight.dim(1) == k.weight.dim(1), k.weight.dim(1) == v.weight.dim(1)
        else { return nil }
        if weight == nil {
            weight = concatenated([q.weight, k.weight, v.weight], axis: 0)
            qEnd = q.weight.dim(0)
            kEnd = qEnd + k.weight.dim(0)
        }
        return weight
    }

    /// `(k(rows), v(rows))` with the bits `apply`'s stacked matmul gives the
    /// same rows, for context rows that enter the cache ahead of their block
    /// (`DFlash2Attention.absorbContext`), or nil when the stack does not
    /// apply.
    ///
    /// Above `kvOnlyMinimumRows` rows the matmul runs over the stack's k|v
    /// rows alone (a view, N = 2048), a third of the full stack's work: the
    /// regular GEMM computes every output element from its own row and
    /// column, so dropping the q columns changes no k or v bit. It stays
    /// regular there on both routes MLX could split: the NAX split-K needs K
    /// >= 3 max(M, N) or max(M, N) <= 1024 (K = 5120, N = 2048: never), and
    /// the non-NAX split-K needs ceil(M/16) * ceil(N/16) <= 2048 (M <= 256
    /// at N = 2048). At or below it the full 6144-wide stack runs (N > K:
    /// neither split-K route applies at any M) and the q columns are dropped.
    func applyContext(
        _ rows: MLXArray, q: Linear, k: Linear, v: Linear
    ) -> (MLXArray, MLXArray)? {
        guard rows.ndim == 3, let weight = stacked(q: q, k: k, v: v) else { return nil }
        if rows.dim(1) > Self.kvOnlyMinimumRows {
            let y = matmul(rows, weight[qEnd...].T)
            let kWidth = kEnd - qEnd
            return (y[.ellipsis, ..<kWidth], y[.ellipsis, kWidth...])
        }
        // Up to 16 context rows: their `[context; block]` forward (at most 32
        // rows) takes the tensor kernel, whose rows do not depend on the row
        // count, so these rows take it too.
        let y = DFlash2TensorMatmul.applyContextRows(rows, weight: weight) ?? matmul(rows, weight.T)
        return (y[.ellipsis, qEnd ..< kEnd], y[.ellipsis, kEnd...])
    }

    /// The row count at or below which `applyContext` keeps the full stack.
    static let kvOnlyMinimumRows = 256

    func applies(q: Linear, k: Linear, v: Linear) -> Bool { stacked(q: q, k: k, v: v) != nil }

    /// The stacked weight (built on first use), or nil when the stack does not apply.
    func stackWeight(q: Linear, k: Linear, v: Linear) -> MLXArray? { stacked(q: q, k: k, v: v) }

    /// `apply`'s matmul, unsliced, with its q and k column ends.
    func applyStacked(
        _ rows: MLXArray, q: Linear, k: Linear, v: Linear
    ) -> (y: MLXArray, qEnd: Int, kEnd: Int)? {
        guard rows.ndim == 3, let weight = stacked(q: q, k: k, v: v) else { return nil }
        // The same route as `apply`, so every row matches today's block forward.
        return (DFlash2TensorMatmul.apply(rows, weight: weight) ?? matmul(rows, weight.T), qEnd, kEnd)
    }

    /// `(q(rows[-blockRows...]), k(rows), v(rows), product)` from one matmul, or nil
    /// when the stack does not apply.
    func apply(
        _ rows: MLXArray, blockRows: Int, q: Linear, k: Linear, v: Linear
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray)? {
        guard rows.ndim == 3, blockRows <= rows.dim(1),
            let weight = stacked(q: q, k: k, v: v)
        else { return nil }
        let y = DFlash2TensorMatmul.apply(rows, weight: weight) ?? matmul(rows, weight.T)
        let n = rows.dim(1)
        return (
            y[0..., (n - blockRows)..., ..<qEnd],
            y[.ellipsis, qEnd ..< kEnd],
            y[.ellipsis, kEnd...],
            y
        )
    }
}

/// Kill switch for the one-projection context+block K/V (default on).
/// The grouped convolutions' `kernel_projection` (5120 -> 1280 on the
/// 16-row block) through the drafter's tensor kernel instead of MLX's GEMM.
/// `DARKBLOOM_DFLASH2_TENSOR_KPROJ=0` keeps the GEMM.
private let dflash2KernelProjectionTensor: Bool = {
    guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH2_TENSOR_KPROJ"]
    else { return true }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

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
        _ hidden: MLXArray, projection: MLXArray, tap: Int, residual: MLXArray?,
        context: MLXArray? = nil
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
            residual.map({ $0.shape == hidden.shape && $0.dtype == dtype }) ?? true,
            context.map({
                $0.ndim == 3 && $0.dim(0) == batch && $0.dim(2) == hiddenSize && $0.dtype == dtype
                    && Self.joinVerified(dtype, kernelSize, groupSize, hiddenSize)
            }) ?? true
        else { return nil }
        let template: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("KS", kernelSize), ("GS", groupSize), ("TAP", tap),
        ]
        if let context {
            let rows = context.dim(1) + length
            return dflash2GroupedConvJoinKernel(
                [hidden, projection, baseKernel, context],
                template: template, grid: (hiddenSize, rows, batch), threadGroup: (256, 1, 1),
                outputShapes: [[batch, rows, hiddenSize]], outputDTypes: [dtype])[0]
        }
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
        let projection = dflash2KernelProjectionTensor
            ? DFlash2TensorMatmul.linear(kernelProjection, hidden) : kernelProjection(hidden)
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

    /// ``prepare(_:)`` with `context`'s rows ahead of the convolved block in
    /// the SAME launch: `(block, taps, [context; block])`, the rows the
    /// attention's one-launch concatenation joins from the two, or nil. The
    /// context rows are copied; the block rows are the tap-0 kernel's.
    func prepare(_ hidden: MLXArray, joining context: MLXArray) -> (MLXArray, MLXArray, MLXArray)? {
        // The same projection as `prepare(_:)` (the tensor kernel when
        // `DARKBLOOM_DFLASH2_TENSOR_KPROJ` is on), so the taps keep its bits.
        let projection = dflash2KernelProjectionTensor
            ? DFlash2TensorMatmul.linear(kernelProjection, hidden) : kernelProjection(hidden)
        guard
            let joined = fusedConvolve(
                hidden, projection: projection, tap: 0, residual: nil, context: context)
        else { return nil }
        return (joined[0..., context.dim(1)..., 0...], projection, joined)
    }

    private static let joinLock = NSLock()
    nonisolated(unsafe) private static var joinVerdicts: [String: Bool] = [:]

    /// The joined launch's self-test (at bind), bit for bit against the
    /// tap-0 kernel's rows after the context rows by `DFlash2Concat`.
    /// `MLXFAST_DFLASH_CONV_JOIN=0` keeps the two launches.
    static func joinVerified(_ dtype: DType, _ ks: Int, _ gs: Int, _ hs: Int) -> Bool {
        guard dflash2ConvJoinEnabled, dflash2FusedConvEnabled, hs % gs == 0,
            [DType.bfloat16, .float16].contains(dtype)
        else { return false }
        return joinLock.withLock {
            let key = "\(dtype) \(ks) \(gs) \(hs)"
            if let verdict = joinVerdicts[key] { return verdict }
            var same = true
            var compared = 0
            do {
                try withError { error in
                    for (seed, c) in [(61, 1), (62, 7)] {
                        let l = 16
                        func draw(_ shape: [Int], _ salt: Int) -> MLXArray {
                            (MLXRandom.normal(shape, key: MLXRandom.key(UInt64(seed * 8 + salt)))
                                * exp(MLXRandom.uniform(
                                    Float(-7) ..< Float(8), [shape[0], shape[1], 1],
                                    key: MLXRandom.key(UInt64(seed * 8 + salt + 4))))
                            ).asType(dtype)
                        }
                        let h = draw([1, l, hs], 0)
                        let dyn = draw([1, l, 2 * ks * (hs / gs)], 1)
                        let base = draw([2, ks, hs], 2)
                        let context = draw([1, c, hs], 3)
                        let template: [(String, any KernelTemplateArg)] = [
                            ("T", dtype), ("KS", ks), ("GS", gs), ("TAP", 0),
                        ]
                        let fused = dflash2GroupedConvJoinKernel(
                            [h, dyn, base, context], template: template, grid: (hs, c + l, 1),
                            threadGroup: (256, 1, 1), outputShapes: [[1, c + l, hs]],
                            outputDTypes: [dtype])[0]
                        let conv = dflash2GroupedConvKernel(
                            [h, dyn, base], template: template, grid: (hs, l, 1),
                            threadGroup: (256, 1, 1), outputShapes: [h.shape],
                            outputDTypes: [dtype])[0]
                        let reference = DFlash2Concat.concatenate([context, conv], axis: 1)
                        same = same && fused.shape == reference.shape
                            && all(fused.view(dtype: .uint16) .== reference.view(dtype: .uint16))
                                .item(Bool.self)
                        compared += reference.size
                    }
                    try error.check()
                }
            } catch {
                same = false
            }
            joinVerdicts[key] = same
            FileHandle.standardError.write(
                ("dflash2 conv+concat join (\(key)): "
                    + (same
                        ? "self-test passed: \(compared) values compared bitwise, 0 mismatches; one launch\n"
                        : "self-test failed; two launches kept\n")).data(using: .utf8)!)
            return same
        }
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

/// Kill switch for the tap-0 convolution joined to its context rows (default on).
private let dflash2ConvJoinEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_CONV_JOIN"]
    else { return false }
    return !["0", "false", "no", "off"].contains(raw.lowercased())
}()

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

/// `[ctx; conv(h)]`: rows below `ctx`'s count are its rows, the rest the
/// tap's convolution of the block.
private let dflash2GroupedConvJoinSource = """
    const uint c = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint H = threads_per_grid.x;
    const uint R = threads_per_grid.y;
    const uint C = uint(ctx_shape[1]);
    out[(size_t(b) * R + r) * H + c] = r < C
        ? ctx[(size_t(b) * C + r) * H + c]
        : dflash2_grouped_conv<T, KS, GS, TAP>(h, dyn, base, b, r - C, c, R - C, H);
    """

private let dflash2GroupedConvJoinKernel = MLXFast.metalKernel(
    name: "dflash2_grouped_conv_join",
    inputNames: ["h", "dyn", "base", "ctx"],
    outputNames: ["out"],
    source: dflash2GroupedConvJoinSource,
    header: dflash2GroupedConvHeader,
    ensureRowContiguous: true)

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
/// Primitives `matmul2d`): each threadgroup owns 32 output columns. Wide
/// projections use two simdgroups over contiguous K halves; smaller ones use
/// four over K quarters. Each chunk is a 16 x 32 x 256 op, and the partials
/// are summed through threadgroup memory. The
/// weights are the layer's BF16 arrays read as stored; the result is the same
/// FP32-accumulated product in a different summation order, rounded to BF16.
/// The drafter only proposes. `DARKBLOOM_DFLASH2_TENSOR_MATMUL=0` keeps the
/// core's GEMM.
///
/// `TILED` (tiling idea from Subflatus3 bb781255's verify int8 copy): the
/// kernel reads a tiled copy of the weight (`tile`), in which each 32-column
/// block's 256-wide K step is one contiguous 16 KB tile, so a simdgroup's
/// K slab of its column block is one contiguous run instead of 32 row pieces
/// a row stride apart. The tensor op, its operands' values, the K order and
/// the reduction are unchanged; only the slice's base and row stride move,
/// so the output is bitwise the stored layout's (self-tested at load on
/// every copy; the in-situ trial then decides, see `DFlash2KernelTrial`).
///
/// Kernel variants on the tiled copies (`Kernel`, `variantSource`): a K split
/// of 2, 4 or 8 simdgroups and 32 or 64 columns per threadgroup per class
/// (the wide stacked gate|up, the narrow o_proj and down_proj), and a
/// one-step prefetch of the next K step's weight tile. With the record's K
/// split they are bit for bit the record's; a changed split reorders the FP32
/// sum. Each is self-tested at load, and the in-situ trial picks one.
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

    // grid: (N / 32 * (32 * SPLITS), 1, 1), threadgroup (32 * SPLITS, 1, 1).
    // Inputs: x bfloat
    // [16, K], w bfloat [N, K], ksz int32 [K, 16, N]. K % 1024 == 0.
    private static let source = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int kq = K / SPLITS;
        const int k0 = int(sg) * kq;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 256, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> A((device bfloat*)x, dextents<int, 2>(K, M));
        // TILED: `w` is `[N/32, K/256, 32, 256]`, viewed as rows of 256: column
        // block n0 / 32's K step k / 256 is the 32 rows from tb + (k / 256) * 32.
        const int tb = (n0 / 32) * (K / 256) * 32;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> B((device bfloat*)w,
            TILED ? dextents<int, 2>(256, (N / 32) * (K / 256) * 32) : dextents<int, 2>(K, N));
        auto tA0 = A.template slice<256, 16>(0, 0);
        auto tB0 = B.template slice<256, 32>(0, TILED ? tb : n0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB0)>, float>();
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) { cT[i] = 0.0f; }
        for (int k = k0; k < k0 + kq; k += 256) {
          auto tA = A.template slice<256, 16>(k, 0);
          auto tB = B.template slice<256, 32>(TILED ? 0 : k, TILED ? tb + (k / 256) * 32 : n0);
          op.run(tA, tB, cT);
        }
        // Destination layout: element i -> n = n0 + fn + (i & 3) + 16 * ((i >> 3) & 1),
        // m = fm + 8 * ((i >> 2) & 1).
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        threadgroup float red[SPLITS - 1][16 * 32];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < 16; i++) { red[sg - 1][i * 32 + lane] = cT[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < 16; i++) {
            float v;
            if constexpr (SPLITS == 2) {
              v = cT[i] + red[0][i * 32 + lane];
            } else {
              v = cT[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
            }
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
            out[(size_t)(fm + 8 * mh) * N + n0 + fn + c + 16 * nh] = OutT(v);
          }
        }
        """

    // Two 16-row tiles over the same weight tile: x bfloat [32, K]; every
    // K step runs the 16 x 32 x 256 op on rows 0-15 and on rows 16-31
    // against one B slice, so a context-plus-block forward of up to 32 rows
    // reads the weights once. Same K partitions and reduction as the m16
    // kernel, per row tile.
    private static let source32 = """
        const int K = ksz[0]; const int M = 32; const int N = ksz[2];
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int kq = K / SPLITS;
        const int k0 = int(sg) * kq;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 256, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> A((device bfloat*)x, dextents<int, 2>(K, M));
        tensor<device bfloat, dextents<int, 2>, tensor_inline> B((device bfloat*)w, dextents<int, 2>(K, N));
        auto tA0 = A.template slice<256, 16>(0, 0);
        auto tB0 = B.template slice<256, 32>(0, n0);
        auto cT0 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB0)>, float>();
        auto cT1 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB0)>, float>();
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) { cT0[i] = 0.0f; cT1[i] = 0.0f; }
        for (int k = k0; k < k0 + kq; k += 256) {
          auto tB = B.template slice<256, 32>(k, n0);
          auto tAlo = A.template slice<256, 16>(k, 0);
          auto tAhi = A.template slice<256, 16>(k, 16);
          op.run(tAlo, tB, cT0);
          op.run(tAhi, tB, cT1);
        }
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        threadgroup float red[SPLITS - 1][2 * 16 * 32];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < 16; i++) {
            red[sg - 1][i * 32 + lane] = cT0[i];
            red[sg - 1][(16 + i) * 32 + lane] = cT1[i];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < 16; i++) {
            float v0, v1;
            if constexpr (SPLITS == 2) {
              v0 = cT0[i] + red[0][i * 32 + lane];
              v1 = cT1[i] + red[0][(16 + i) * 32 + lane];
            } else {
              v0 = cT0[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
              v1 = cT1[i] + red[0][(16 + i) * 32 + lane] + red[1][(16 + i) * 32 + lane]
                  + red[2][(16 + i) * 32 + lane];
            }
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
            out[(size_t)(fm + 8 * mh) * N + n0 + fn + c + 16 * nh] = OutT(v0);
            out[(size_t)(16 + fm + 8 * mh) * N + n0 + fn + c + 16 * nh] = OutT(v1);
          }
        }
        """

    private static let kernel32 = MLXFast.metalKernel(
        name: "dflash2_bf16_matmul_m32",
        inputNames: ["x", "w", "ksz"],
        outputNames: ["out"],
        source: source32,
        header: header,
        ensureRowContiguous: true)

    /// `DARKBLOOM_DFLASH2_TENSOR_M32=0` keeps MLX's GEMM for 17-32 rows.
    private static let rows32Enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH2_TENSOR_M32"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "dflash2_bf16_matmul_m16",
        inputNames: ["x", "w", "ksz"],
        outputNames: ["out"],
        source: source,
        header: header,
        ensureRowContiguous: true)

    /// `sourceSwapped` (on after its load-time self-test; `MLXFAST_DRAFT_SWAP=0`
    /// keeps `source`): the same product with the operands' roles swapped.
    /// Each simdgroup loads its 32-column weight slice for one KT-wide K
    /// step (KT = 128) into a cooperative left operand (`[32, KT]`, K
    /// contiguous as stored) and multiplies it by the transposed 16-row input
    /// slice, so the op is `32 x 16 x KT` into a `[32 columns, 16 rows]` FP32
    /// accumulator. The weight read is a plain cooperative load issued ahead
    /// of the op instead of the op's own operand fetch. Each output element is
    /// still the FP32 sum of the same BF16 products over each simdgroup's K
    /// partition, accumulated in K order, and the partitions (`SPLITS`, the
    /// same `K / SPLITS` slabs) are added in the same order as `source`, so
    /// the output is bitwise `source`'s: `prepareSwapped` compares every
    /// weight's output bit for bit at load and keeps `source` on any
    /// mismatch. Stored layout only (no tiled copy is read).
    private static let sourceSwapped = """
        const int K = ksz[0]; const int N = ksz[2];
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int kq = K / SPLITS;
        const int k0 = int(sg) * kq;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            32, 16, KT, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> W((device bfloat*)w, dextents<int, 2>(K, N));
        tensor<device bfloat, dextents<int, 2>, tensor_inline> X((device bfloat*)x, dextents<int, 2>(K, 16));
        auto tW0 = W.template slice<KT, 32>(0, n0);
        auto tX0 = X.template slice<KT, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tW0)>, metal::remove_addrspace_t<decltype(tX0)>, float>();
        auto lw = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        const uint16_t cap = cT.get_capacity();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); i++) { cT[i] = 0.0f; }
        for (int k = k0; k < k0 + kq; k += KT) {
          auto tW = W.template slice<KT, 32>(k, n0);
          auto tX = X.template slice<KT, 16>(k, 0);
          lw.load(tW);
          op.run(lw, tX, cT);
        }
        threadgroup float red[SPLITS - 1][16 * 32];
        if (sg > 0) {
          for (uint16_t i = 0; i < cap; i++) { red[sg - 1][i * 32 + lane] = cT[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          for (uint16_t i = 0; i < cap; i++) {
            if (!cT.is_valid_element(i)) continue;
            float v;
            if constexpr (SPLITS == 2) {
              v = cT[i] + red[0][i * 32 + lane];
            } else {
              v = cT[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
            }
            // Destination coordinates: [0] the input row, [1] the column in the block.
            auto idx = cT.get_multidimensional_index(i);
            out[(size_t)idx[0] * N + n0 + idx[1]] = OutT(v);
          }
        }
        """

    private static let kernelSwapped = MLXFast.metalKernel(
        name: "dflash2_bf16_matmul_m16s",
        inputNames: ["x", "w", "ksz"],
        outputNames: ["out"],
        source: sourceSwapped,
        header: header,
        ensureRowContiguous: true)

    /// `source32` in `sourceSwapped`'s form: one cooperative `[32, KT]`
    /// weight slice per K step, multiplied by the input's rows 0-15 and
    /// 16-31 (two `32 x 16 x KT` ops into two accumulators), the same K
    /// partitions and the same partial order as `source32`, so its bits
    /// (`prepareSwapped` compares them at load).
    private static let sourceSwapped32 = """
        const int K = ksz[0]; const int N = ksz[2];
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int kq = K / SPLITS;
        const int k0 = int(sg) * kq;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            32, 16, KT, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> W((device bfloat*)w, dextents<int, 2>(K, N));
        tensor<device bfloat, dextents<int, 2>, tensor_inline> X((device bfloat*)x, dextents<int, 2>(K, 32));
        auto tW0 = W.template slice<KT, 32>(0, n0);
        auto tX0 = X.template slice<KT, 16>(0, 0);
        auto cT0 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tW0)>, metal::remove_addrspace_t<decltype(tX0)>, float>();
        auto cT1 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tW0)>, metal::remove_addrspace_t<decltype(tX0)>, float>();
        auto lw = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        const uint16_t cap = cT0.get_capacity();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT0.get_capacity(); i++) { cT0[i] = 0.0f; cT1[i] = 0.0f; }
        for (int k = k0; k < k0 + kq; k += KT) {
          auto tW = W.template slice<KT, 32>(k, n0);
          lw.load(tW);
          auto tXlo = X.template slice<KT, 16>(k, 0);
          op.run(lw, tXlo, cT0);
          auto tXhi = X.template slice<KT, 16>(k, 16);
          op.run(lw, tXhi, cT1);
        }
        threadgroup float red[SPLITS - 1][2 * 16 * 32];
        if (sg > 0) {
          for (uint16_t i = 0; i < cap; i++) {
            red[sg - 1][i * 32 + lane] = cT0[i];
            red[sg - 1][(16 + i) * 32 + lane] = cT1[i];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          for (uint16_t i = 0; i < cap; i++) {
            if (!cT0.is_valid_element(i)) continue;
            float v0, v1;
            if constexpr (SPLITS == 2) {
              v0 = cT0[i] + red[0][i * 32 + lane];
              v1 = cT1[i] + red[0][(16 + i) * 32 + lane];
            } else {
              v0 = cT0[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
              v1 = cT1[i] + red[0][(16 + i) * 32 + lane] + red[1][(16 + i) * 32 + lane]
                  + red[2][(16 + i) * 32 + lane];
            }
            auto idx = cT0.get_multidimensional_index(i);
            out[(size_t)idx[0] * N + n0 + idx[1]] = OutT(v0);
            out[(size_t)(16 + idx[0]) * N + n0 + idx[1]] = OutT(v1);
          }
        }
        """

    private static let kernelSwapped32 = MLXFast.metalKernel(
        name: "dflash2_bf16_matmul_m32s",
        inputNames: ["x", "w", "ksz"],
        outputNames: ["out"],
        source: sourceSwapped32,
        header: header,
        ensureRowContiguous: true)

    /// Whether 17-32-row launches take `sourceSwapped32` (set by `prepareSwapped`).
    nonisolated(unsafe) static var swapped32Active = false

    /// The K step of `sourceSwapped` (`MLXFAST_DRAFT_SWAP_KT`: 64 or 128, default 128).
    private static let swappedKT: Int = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DRAFT_SWAP_KT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return raw == "64" ? 64 : 128
    }()

    /// `MLXFAST_DRAFT_SWAP=0` keeps `source` for the stored layout.
    private static let swappedSetting: Bool = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DRAFT_SWAP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(raw ?? "")
    }()

    /// Whether stored-layout launches take `sourceSwapped`: set once by
    /// `prepareSwapped` when every weight's output matched bit for bit.
    nonisolated(unsafe) static var swappedActive = false

    // The variant kernel, on the tiled copy only. grid: (N / TN * (32 *
    // SPLITS), 1, 1), threadgroup (32 * SPLITS, 1, 1). Simdgroup s takes K
    // steps [s * steps / SPLITS, (s + 1) * steps / SPLITS) (the record's
    // contiguous parts when SPLITS divides the step count) and runs the
    // record's 16 x 32 x 256 op on each of its TN / 32 column blocks per
    // step; the parts are added in simdgroup order, as the record adds them.
    // PF = 1 loads four 128-byte lines per lane of the next step's tiles
    // before the op and folds them into a never-stored word after it (a
    // prefetch; the arithmetic is unchanged).
    private static let variantSource = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        constexpr int NH = TN / 32;
        const int n0 = int(threadgroup_position_in_grid.x) * TN;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int steps = K / 256;
        const int KS = steps * 32;
        const int s0 = int(sg) * steps / SPLITS;
        const int s1 = (int(sg) + 1) * steps / SPLITS;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 256, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> A((device bfloat*)x, dextents<int, 2>(K, M));
        tensor<device bfloat, dextents<int, 2>, tensor_inline> B((device bfloat*)w, dextents<int, 2>(256, (N / 32) * KS));
        const int tb = (n0 / 32) * KS;
        auto tA0 = A.template slice<256, 16>(0, 0);
        auto tB0 = B.template slice<256, 32>(0, tb);
        auto cT0 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB0)>, float>();
        auto cT1 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB0)>, float>();
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) {
          cT0[i] = 0.0f;
          if constexpr (NH == 2) { cT1[i] = 0.0f; }
        }
        const device uint* wl = (const device uint*)w + (size_t)lane * 32;
        uint touched = 0;
        for (int s = s0; s < s1; s++) {
          uint p[NH][4];
          const bool ahead = PF != 0 && s + 1 < s1;
          if (ahead) {
            #pragma clang loop unroll(full)
            for (int h = 0; h < NH; h++) {
              const device uint* nx = wl + (size_t)(tb + h * KS + (s + 1) * 32) * 128;
              #pragma clang loop unroll(full)
              for (int j = 0; j < 4; j++) { p[h][j] = nx[j * 1024]; }
            }
          }
          auto tA = A.template slice<256, 16>(s * 256, 0);
          auto tB = B.template slice<256, 32>(0, tb + s * 32);
          op.run(tA, tB, cT0);
          if constexpr (NH == 2) {
            auto tB1 = B.template slice<256, 32>(0, tb + KS + s * 32);
            op.run(tA, tB1, cT1);
          }
          if (ahead) {
            #pragma clang loop unroll(full)
            for (int h = 0; h < NH; h++) {
              #pragma clang loop unroll(full)
              for (int j = 0; j < 4; j++) { touched ^= p[h][j]; }
            }
          }
        }
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        threadgroup float red[SPLITS - 1][NH][16 * 32];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < 16; i++) {
            red[sg - 1][0][i * 32 + lane] = cT0[i];
            if constexpr (NH == 2) { red[sg - 1][NH - 1][i * 32 + lane] = cT1[i]; }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < 16; i++) {
              float v = h == 0 ? cT0[i] : cT1[i];
              if constexpr (SPLITS == 2) {
                v = v + red[0][h][i * 32 + lane];
              } else if constexpr (SPLITS == 4) {
                v = v + red[0][h][i * 32 + lane] + red[1][h][i * 32 + lane] + red[2][h][i * 32 + lane];
              } else {
                #pragma clang loop unroll(full)
                for (int j = 0; j < SPLITS - 1; j++) { v = v + red[j][h][i * 32 + lane]; }
              }
              const int c = i & 3; const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
              out[(size_t)(fm + 8 * mh) * N + n0 + 32 * h + fn + c + 16 * nh] = OutT(v);
            }
          }
        }
        // Never taken (ksz[1] is 16): keeps the prefetch loads.
        if (PF != 0 && touched == 0x7fbfffffu && ksz[1] == 0) { out[0] = OutT(0.0f); }
        """

    private static let variantKernel = MLXFast.metalKernel(
        name: "dflash2_bf16_matmul_m16_kvar",
        inputNames: ["x", "w", "ksz"],
        outputNames: ["out"],
        source: variantSource,
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

    /// `apply` for context rows that enter the cache ahead of their block:
    /// at most 16 rows, and only while 17-32 rows (their `[context; block]`
    /// forward) take the tensor kernel too; nil otherwise.
    static func applyContextRows(_ x: MLXArray, weight: MLXArray) -> MLXArray? {
        guard rows32Enabled, x.ndim >= 2, x.size / max(x.dim(-1), 1) <= rowsPerTile else { return nil }
        return apply(x, weight: weight)
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
        guard rows >= 1, rows <= 2 * rowsPerTile, weight.dim(1) == k, k % 1024 == 0, n % 32 == 0
        else { return nil }
        if rows > rowsPerTile {
            guard rows32Enabled else { return nil }
            var a = x.reshaped(rows, k)
            if rows < 2 * rowsPerTile {
                a = concatenated(
                    [a, MLXArray.zeros([2 * rowsPerTile - rows, k], dtype: .bfloat16)], axis: 0)
            }
            let y = launch32(a, weight, k: k, n: n, swapped: swapped32Active, outputDType: .bfloat16)
            let rowsOut = rows < 2 * rowsPerTile ? y[0 ..< rows] : y
            return rowsOut.reshaped(Array(x.shape.dropLast()) + [n])
        }
        var a = x.reshaped(rows, k)
        if rows < rowsPerTile {
            a = DFlash2Concat.padRows(a, to: rowsPerTile)
        }
        let chosen = current
        let y: MLXArray
        if swappedActive {
            // The swapped kernel reads the stored weight (no copy exists while it is on).
            y = launch(a, weight, k: k, n: n, tiled: false, outputDType: .bfloat16)
        } else if chosen.tiled, let tiled = tiledCopy(weight) {
            y = chosen.variant(n: n)
                ? launchVariant(a, tiled, k: k, n: n, kernel: chosen, outputDType: .bfloat16)
                : launch(a, tiled, k: k, n: n, tiled: true, outputDType: .bfloat16)
        } else {
            y = launch(a, weight, k: k, n: n, tiled: false, outputDType: .bfloat16)
        }
        let rowsOut = rows < rowsPerTile ? y[0 ..< rows] : y
        return rowsOut.reshaped(Array(x.shape.dropLast()) + [n])
    }

    /// The variant kernel over a 16-row `a` and the tiled copy `t` of an
    /// `[N, K]` weight, with `kernel`'s split, width and prefetch for N.
    private static func launchVariant(
        _ a: MLXArray, _ t: MLXArray, k: Int, n: Int, kernel: Kernel, outputDType: DType
    ) -> MLXArray {
        var (splits, tn) = kernel.split(n: n)
        if n % tn != 0 { tn = 32 }
        if k / 256 < splits { splits = Kernel.stockSplits(n: n) }
        let threads = splits * 32
        return variantKernel(
            [a, t, dimsArray(k: k, n: n)],
            template: [
                ("OutT", outputDType), ("SPLITS", splits), ("TN", tn), ("PF", kernel.prefetch),
            ],
            grid: (n / tn * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rowsPerTile, n]], outputDTypes: [outputDType])[0]
    }

    /// The kernel over a 16-row `a` and `w` (the stored `[N, K]` weight, or
    /// its tiled copy when `tiled`).
    private static func launch(
        _ a: MLXArray, _ w: MLXArray, k: Int, n: Int, tiled: Bool, outputDType: DType
    ) -> MLXArray {
        // Wide projections expose enough output tiles to use fewer K partitions.
        // Keep the accepted four-way route for the smaller projections.
        let splits = Kernel.stockSplits(n: n)
        let threads = splits * 32
        if !tiled && swappedActive {
            return launchSwapped(a, w, k: k, n: n, splits: splits, outputDType: outputDType)
        }
        return kernel(
            [a, w, dimsArray(k: k, n: n)],
            template: [("OutT", outputDType), ("SPLITS", splits), ("TILED", tiled ? 1 : 0)],
            grid: (n / 32 * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rowsPerTile, n]], outputDTypes: [outputDType])[0]
    }

    /// `source32` (or `sourceSwapped32` when `swapped`) over a 32-row `a`.
    private static func launch32(
        _ a: MLXArray, _ w: MLXArray, k: Int, n: Int, swapped: Bool, outputDType: DType
    ) -> MLXArray {
        let splits = n >= 16384 ? 2 : 4
        let threads = splits * 32
        if swapped {
            return kernelSwapped32(
                [a, w, dimsArray(k: k, n: n)],
                template: [("OutT", outputDType), ("SPLITS", splits), ("KT", swappedKT)],
                grid: (n / 32 * threads, 1, 1), threadGroup: (threads, 1, 1),
                outputShapes: [[2 * rowsPerTile, n]], outputDTypes: [outputDType])[0]
        }
        return kernel32(
            [a, w, dimsArray(k: k, n: n)],
            template: [("OutT", outputDType), ("SPLITS", splits)],
            grid: (n / 32 * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[2 * rowsPerTile, n]], outputDTypes: [outputDType])[0]
    }

    /// `sourceSwapped` over a 16-row `a` and the stored `[N, K]` weight, with
    /// the stock kernel's K split.
    private static func launchSwapped(
        _ a: MLXArray, _ w: MLXArray, k: Int, n: Int, splits: Int, outputDType: DType
    ) -> MLXArray {
        let threads = splits * 32
        return kernelSwapped(
            [a, w, dimsArray(k: k, n: n)],
            template: [("OutT", outputDType), ("SPLITS", splits), ("KT", swappedKT)],
            grid: (n / 32 * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rowsPerTile, n]], outputDTypes: [outputDType])[0]
    }

    /// Runs `sourceSwapped` and the stock kernel on every one of `weights`
    /// (stored layout) against the same random 16-row BF16 input, FP32 and
    /// BF16 outputs, and compares every output bit; `swappedActive` only when
    /// all match. One stderr line either way. Nothing runs when the kernel is
    /// off, the toolchain has no tensor operands, or `MLXFAST_DRAFT_SWAP=0`.
    static func prepareSwapped(_ weights: [MLXArray], rows32 weights32: [MLXArray] = []) -> Bool {
        swappedActive = false
        swapped32Active = false
        guard enabled, Qwen35TensorPackedMatmul.tensorOperandsAvailable, swappedSetting else {
            return false
        }
        let eligible = weights.filter {
            $0.dtype == .bfloat16 && $0.ndim == 2 && $0.dim(1) % 1024 == 0 && $0.dim(0) % 32 == 0
        }
        guard !eligible.isEmpty else { return false }
        let start = DispatchTime.now().uptimeNanoseconds
        var inputs: [Int: MLXArray] = [:]
        var differing: [MLXArray] = []
        var values = 0
        for (index, w) in eligible.enumerated() {
            let k = w.dim(1)
            let n = w.dim(0)
            let splits = n >= 16384 ? 2 : 4
            let a = inputs[k] ?? MLXRandom.normal(
                [rowsPerTile, k], key: MLXRandom.key(UInt64(9203 + index))
            ).asType(.bfloat16)
            inputs[k] = a
            for (outputDType, bits) in [(DType.float32, DType.uint32), (.bfloat16, .uint16)] {
                let stock = kernel(
                    [a, w, dimsArray(k: k, n: n)],
                    template: [("OutT", outputDType), ("SPLITS", splits), ("TILED", 0)],
                    grid: (n / 32 * splits * 32, 1, 1), threadGroup: (splits * 32, 1, 1),
                    outputShapes: [[rowsPerTile, n]], outputDTypes: [outputDType])[0]
                let swapped = launchSwapped(a, w, k: k, n: n, splits: splits, outputDType: outputDType)
                differing.append(
                    (stock.view(dtype: bits) .!= swapped.view(dtype: bits)).asType(.int32).sum())
                values += rowsPerTile * n
            }
            if differing.count >= 16 {
                let partial = stacked(differing).sum()
                eval(partial)
                differing = [partial]
            }
        }
        let mismatches = stacked(differing).sum().item(Int.self)
        swappedActive = mismatches == 0
        // The 32-row form, only while the 16-row one is on.
        var mismatches32 = 0
        var values32 = 0
        let eligible32 = weights32.filter {
            $0.dtype == .bfloat16 && $0.ndim == 2 && $0.dim(1) % 1024 == 0 && $0.dim(0) % 32 == 0
        }
        if swappedActive, rows32Enabled, !eligible32.isEmpty {
            var differing32: [MLXArray] = []
            for (index, w) in eligible32.enumerated() {
                let k = w.dim(1)
                let n = w.dim(0)
                let a = MLXRandom.normal(
                    [2 * rowsPerTile, k], key: MLXRandom.key(UInt64(9403 + index))
                ).asType(.bfloat16)
                for (outputDType, bits) in [(DType.float32, DType.uint32), (.bfloat16, .uint16)] {
                    let stock = launch32(a, w, k: k, n: n, swapped: false, outputDType: outputDType)
                    let swapped = launch32(a, w, k: k, n: n, swapped: true, outputDType: outputDType)
                    differing32.append(
                        (stock.view(dtype: bits) .!= swapped.view(dtype: bits)).asType(.int32).sum())
                    values32 += 2 * rowsPerTile * n
                }
            }
            mismatches32 = stacked(differing32).sum().item(Int.self)
            swapped32Active = mismatches32 == 0
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        FileHandle.standardError.write(
            Data(
                ("dflash2 swapped m16 kernel: self-test \(swappedActive ? "passed" : "FAILED") "
                    + "(\(eligible.count) weights, \(values) values bitwise, \(mismatches) mismatches); "
                    + (swappedActive ? "on, K step \(swappedKT)" : "stock kept")
                    + "; 32-row form \(eligible32.count) weights, \(values32) values, "
                    + "\(mismatches32) mismatches, " + (swapped32Active ? "on" : "off")
                    + String(format: "; %.0f ms\n", elapsed)).utf8))
        return swappedActive
    }

    // MARK: Tiled weights

    /// `MLXFAST_DRAFT_TILED=1` / `0` forces the tiled / stored layout (a
    /// forced `1` still needs the self-test); unset, the in-situ trial picks.
    static let tiledSetting: Bool? = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DRAFT_TILED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ["1", "true", "yes", "on"].contains(raw ?? "") { return true }
        if ["0", "false", "no", "off"].contains(raw ?? "") { return false }
        return nil
    }()

    /// The kernel `apply` runs. Set at load (forced), by the in-situ trial's
    /// round boundaries, and once by its verdict; read when a drafter graph
    /// is built.
    nonisolated(unsafe) static var current = Kernel.stock

    /// Which kernel reads the weights that have a tiled copy (`fc` has none:
    /// it always runs the record's kernel on its stored weight). `stock`: the
    /// record's kernel on the stored weights; `tiled`: the record's kernel on
    /// the tiled copies; otherwise the variant kernel on the tiled copies with
    /// a K split and a width per class (wide: N >= 16384, the stacked
    /// gate|up; narrow: o_proj and down_proj) and the prefetch. A class that
    /// keeps the record's split, 32 columns and no prefetch runs the record's
    /// kernel on its tiled copy.
    struct Kernel: Equatable {
        var tiled: Bool
        var wideSplits = 2, wideTN = 32, narrowSplits = 4, narrowTN = 32, prefetch = 0

        static let stock = Kernel(tiled: false)
        static let tiledStock = Kernel(tiled: true)

        /// The record's K split (`launch`).
        static func stockSplits(n: Int) -> Int { n >= 16384 ? 2 : 4 }

        func split(n: Int) -> (splits: Int, tn: Int) {
            n >= 16384 ? (wideSplits, wideTN) : (narrowSplits, narrowTN)
        }

        /// Whether a weight of `n` rows runs the variant kernel.
        func variant(n: Int) -> Bool {
            let (splits, tn) = split(n: n)
            return tiled && (splits != Self.stockSplits(n: n) || tn != 32 || prefetch != 0)
        }

        /// The record's split on both classes: the same partial sums, added in
        /// the same order, so the record's bits (self-tested). A changed split
        /// reorders the FP32 sum.
        var bitwise: Bool { wideSplits == 2 && narrowSplits == 4 }

        var name: String {
            !tiled
                ? "stock"
                : self == .tiledStock
                    ? "tiled" : "w\(wideSplits)x\(wideTN)n\(narrowSplits)x\(narrowTN)p\(prefetch)"
        }

        init(
            tiled: Bool, wideSplits: Int = 2, wideTN: Int = 32, narrowSplits: Int = 4,
            narrowTN: Int = 32, prefetch: Int = 0
        ) {
            (self.tiled, self.wideSplits, self.wideTN) = (tiled, wideSplits, wideTN)
            (self.narrowSplits, self.narrowTN, self.prefetch) = (narrowSplits, narrowTN, prefetch)
        }

        /// `stock`, `tiled`, or `w<S>x<TN>n<S>x<TN>p<PF>` (wide then narrow
        /// class; S 2/4/8, TN 32/64, PF 0/1), e.g. `w2x64n4x32p1`.
        init?(name: String) {
            if name == "stock" { self = .stock; return }
            if name == "tiled" { self = .tiledStock; return }
            let v = name.split(whereSeparator: { "wxnp".contains($0) }).compactMap { Int($0) }
            guard v.count == 5, [2, 4, 8].contains(v[0]), [32, 64].contains(v[1]),
                [2, 4, 8].contains(v[2]), [32, 64].contains(v[3]), [0, 1].contains(v[4])
            else { return nil }
            self.init(
                tiled: true, wideSplits: v[0], wideTN: v[1], narrowSplits: v[2], narrowTN: v[3],
                prefetch: v[4])
            guard self.name == name || self == .tiledStock else { return nil }
        }
    }

    /// `MLXFAST_DRAFT_KVAR=0`: no kernel variants (the trial is the record's
    /// stored against tiled one).
    static let variantsEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DRAFT_KVAR"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(raw ?? "")
    }()

    /// `MLXFAST_DRAFT_KVAR_FORCE=<name>` (`Kernel(name:)`): that kernel, no
    /// trial, once it passes its self-test (the stored layout otherwise). A
    /// variant is not forced with the variants off.
    static let forcedKernel: Kernel? = {
        guard let raw = ProcessInfo.processInfo.environment["MLXFAST_DRAFT_KVAR_FORCE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !raw.isEmpty,
            let kernel = Kernel(name: raw), variantsEnabled || kernel == .stock || kernel == .tiledStock
        else { return nil }
        return kernel
    }()

    /// The in-situ trial chooses (nothing forces a kernel).
    static var trialWanted: Bool { tiledSetting == nil && forcedKernel == nil }

    /// The bit-for-bit variants the trial's first request times beside
    /// `stock` and `tiled`: 64 columns on the wide class, the prefetch, both.
    static let bitwiseVariants: [Kernel] = [
        Kernel(tiled: true, wideTN: 64), Kernel(tiled: true, prefetch: 1),
        Kernel(tiled: true, wideTN: 64, prefetch: 1),
    ]

    /// The changed-split variants of `base` (its widths and prefetch, on the
    /// tiled copies): the wide class split 4, the narrow class split 8, both.
    static func splitVariants(of base: Kernel) -> [Kernel] {
        var b = base
        b.tiled = true
        var w = b, n = b, wn = b
        w.wideSplits = 4
        n.narrowSplits = 8
        (wn.wideSplits, wn.narrowSplits) = (4, 8)
        return [w, n, wn]
    }

    /// The stage-1 variants that passed their self-test at load.
    nonisolated(unsafe) static var passedBitwiseVariants: [Kernel] = []

    /// A changed split's FP32 outputs must lie within this fraction of each
    /// output row's largest magnitude of the record's (2^-15; locally the
    /// reordering moves them by at most 2.5e-6).
    static let splitTolerance: Float = 1.0 / 32768

    /// One stored weight and its tiled copy per production shape, and one
    /// random 16-row input per K, for the variants' self-tests.
    nonisolated(unsafe) private static var testWeights: [(source: MLXArray, tiled: MLXArray)] = []
    nonisolated(unsafe) private static var testInputs: [Int: MLXArray] = [:]

    /// `kernel` against the record's kernel on the stored weights, on every
    /// production shape it runs the variant kernel on, one fixed random
    /// 16-row input per K: where a class keeps the record's split, its BF16
    /// outputs (the form the drafter reads) bit for bit; where it changes the
    /// split, its FP32 outputs within `splitTolerance`, the same argmax in
    /// every row, and its BF16 form built and run. Each kernel runs in its own
    /// error scope: a form that fails to compile or run fails the test, never
    /// the process. Returns the verdict and the largest relative difference.
    static func selfTest(_ kernel: Kernel) -> (passed: Bool, maxRelative: Float) {
        let result = try? withError { scoped -> (Bool, Float) in
            var passed = true
            var worst: Float = 0
            for (w, t) in testWeights where kernel.variant(n: w.dim(0)) {
                let (n, k) = (w.dim(0), w.dim(1))
                guard let a = testInputs[k] else { return (false, worst) }
                if kernel.split(n: n).splits == Kernel.stockSplits(n: n) {
                    let reference = launch(a, w, k: k, n: n, tiled: false, outputDType: .bfloat16)
                    let y = launchVariant(a, t, k: k, n: n, kernel: kernel, outputDType: .bfloat16)
                    let differing = (reference.view(dtype: .uint16) .!= y.view(dtype: .uint16))
                        .asType(.int32).sum()
                    eval(differing)
                    try scoped.check()
                    if differing.item(Int32.self) != 0 { passed = false }
                } else {
                    let reference = launch(a, w, k: k, n: n, tiled: false, outputDType: .float32)
                    let y = launchVariant(a, t, k: k, n: n, kernel: kernel, outputDType: .float32)
                    let scale = maximum(
                        abs(reference).max(axis: -1, keepDims: true), MLXArray(Float(1e-30)))
                    let relative = (abs(y - reference) / scale).max()
                    let sameArgmax = all(
                        argMax(reference, axis: -1) .== argMax(y, axis: -1))
                    let production = launchVariant(
                        a, t, k: k, n: n, kernel: kernel, outputDType: .bfloat16)
                    eval(relative, sameArgmax, production)
                    try scoped.check()
                    let r = relative.item(Float.self)
                    worst = max(worst, r)
                    if !(r <= splitTolerance) || !sameArgmax.item(Bool.self) { passed = false }
                }
            }
            return (passed, worst)
        }
        return result ?? (false, .nan)
    }

    private static let tiledLock = NSLock()
    /// Each stored weight (held, so its identity is never reused) with its copy.
    nonisolated(unsafe) private static var tiledCopies:
        [ObjectIdentifier: (source: MLXArray, tiled: MLXArray)] = [:]

    private static func tiledCopy(_ weight: MLXArray) -> MLXArray? {
        tiledLock.withLock { tiledCopies[ObjectIdentifier(weight)]?.tiled }
    }

    /// The array `apply` reads for `weight` at block width: its tiled copy
    /// while the chosen kernel reads the tiled layout (`current.tiled`), the
    /// weight itself otherwise.
    static func readWeight(_ weight: MLXArray) -> MLXArray {
        (current.tiled && !swappedActive ? tiledCopy(weight) : nil) ?? weight
    }

    /// `[N, K]` reordered to `[N/32, K/256, 32, 256]` (returned as `[N, K]`):
    /// for each 32-column block and 256-wide K step, the 32 columns' 256
    /// values in column order. Only positions change.
    static func tile(_ w: MLXArray) -> MLXArray {
        let n = w.dim(0)
        let k = w.dim(1)
        return w.reshaped([n / 32, 32, k / 256, 256]).transposed(0, 2, 1, 3).contiguous()
            .reshaped([n, k])
    }

    /// Frees the copies (the trial kept the stored layout) and the variants'
    /// self-test operands.
    static func dropTiledCopies() {
        tiledLock.withLock { tiledCopies.removeAll() }
        dropTestOperands()
    }

    /// Frees the variants' self-test operands (the copies stay registered).
    static func dropTestOperands() {
        testWeights = []
        testInputs = [:]
    }

    /// Builds the tiled copy of each of `weights` the kernel serves, then
    /// runs the kernel on every copy (TILED) and on its stored weight against
    /// the same random 16-row BF16 input, FP32 and BF16 outputs, and compares
    /// every output bit. Registers the copies and returns true only when all
    /// match; then self-tests the forced kernel or the bit-for-bit variants
    /// (`bitwiseVariants`). One stderr line either way. Nothing runs when the
    /// kernel is off, the toolchain has no tensor operands, or
    /// `MLXFAST_DRAFT_TILED=0`.
    static func prepareTiled(_ weights: [MLXArray]) -> Bool {
        dropTiledCopies()
        current = .stock
        passedBitwiseVariants = []
        // The swapped kernel reads the stored layout: no copy, no kernel trial.
        guard enabled, Qwen35TensorPackedMatmul.tensorOperandsAvailable, tiledSetting != false,
            !swappedActive
        else { return false }
        let eligible = weights.filter {
            $0.dtype == .bfloat16 && $0.ndim == 2 && $0.dim(1) % 1024 == 0 && $0.dim(0) % 32 == 0
        }
        guard !eligible.isEmpty else { return false }
        let start = DispatchTime.now().uptimeNanoseconds
        var copies: [(source: MLXArray, tiled: MLXArray)] = []
        var bytes = 0
        for w in eligible {
            let t = tile(w)
            eval(t)
            copies.append((w, t))
            bytes += w.nbytes
        }
        var inputs: [Int: MLXArray] = [:]
        var differing: [MLXArray] = []
        var values = 0
        for (index, (w, t)) in copies.enumerated() {
            let k = w.dim(1)
            let n = w.dim(0)
            let a = inputs[k] ?? MLXRandom.normal(
                [rowsPerTile, k], key: MLXRandom.key(UInt64(8101 + index))
            ).asType(.bfloat16)
            inputs[k] = a
            for (outputDType, bits) in [(DType.float32, DType.uint32), (.bfloat16, .uint16)] {
                let stored = launch(a, w, k: k, n: n, tiled: false, outputDType: outputDType)
                let copy = launch(a, t, k: k, n: n, tiled: true, outputDType: outputDType)
                differing.append(
                    (stored.view(dtype: bits) .!= copy.view(dtype: bits)).asType(.int32).sum())
                values += rowsPerTile * n
            }
        }
        let mismatches = stacked(differing).sum().item(Int.self)
        let passed = mismatches == 0
        var verdict = "stored layout kept"
        if passed {
            tiledLock.withLock {
                for (w, t) in copies { tiledCopies[ObjectIdentifier(w)] = (w, t) }
            }
            var shapes = Set<[Int]>()
            testWeights = copies.filter { shapes.insert($0.source.shape).inserted }
            testInputs = inputs
            let variantStart = DispatchTime.now().uptimeNanoseconds
            if let forced = forcedKernel {
                var test = (passed: true, maxRelative: Float(0))
                if forced.tiled, forced != .tiledStock { test = selfTest(forced) }
                current = test.passed ? forced : .stock
                verdict = "\(forced.name) forced by MLXFAST_DRAFT_KVAR_FORCE"
                    + (forced.tiled && forced != .tiledStock
                        ? ", self-test \(test.passed ? "passed" : "FAILED, stored layout kept")"
                            + (forced.bitwise ? " (bitwise)" : String(format: " (max rel %.1e)", test.maxRelative))
                        : "")
                dropTestOperands()
                if !current.tiled { dropTiledCopies() }
            } else if tiledSetting == true {
                current = .tiledStock
                verdict = "tiled forced on by MLXFAST_DRAFT_TILED=1"
                dropTestOperands()
            } else {
                verdict = "the in-situ trial decides"
                if variantsEnabled {
                    var failed: [Kernel] = []
                    for kernel in bitwiseVariants {
                        if selfTest(kernel).passed {
                            passedBitwiseVariants.append(kernel)
                        } else {
                            failed.append(kernel)
                        }
                    }
                    verdict += "; variants bitwise [\(passedBitwiseVariants.map(\.name).joined(separator: " "))]"
                        + (failed.isEmpty ? "" : " FAILED [\(failed.map(\.name).joined(separator: " "))]")
                        + String(format: " in %.0f ms", Double(DispatchTime.now().uptimeNanoseconds - variantStart) / 1e6)
                } else {
                    verdict += "; variants off (MLXFAST_DRAFT_KVAR=0)"
                }
            }
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        FileHandle.standardError.write(
            Data(
                ("dflash2 tiled weights: self-test \(passed ? "passed" : "FAILED") "
                    + "(\(copies.count) weights, "
                    + String(format: "%.2f GB, ", Double(bytes) / 1e9)
                    + "\(values) values bitwise, \(mismatches) mismatches); \(verdict); "
                    + String(format: "%.0f ms\n", elapsed)).utf8))
        return passed
    }

    /// `layer(x)` through the tensor kernel when it applies (no bias).
    static func linear(_ layer: Linear, _ x: MLXArray) -> MLXArray {
        if layer.bias == nil, let y = apply(x, weight: layer.weight) {
            return y
        }
        return layer(x)
    }
}

/// The drafter kernel's in-situ trial, in the load-time warm
/// (`Qwen35DFlash2Assistant.runKernelTrial`), modelled on the verify int8
/// kernels' `NarrowInSituTrial`: engine requests whose rounds rotate through
/// the candidate kernels, `perCandidate` timed rounds each; per candidate the
/// median round (outliers above 1.5x the median dropped); a candidate is
/// adopted only when it beats the reference by more than `adoptMargin`.
///
/// Stage 1 rotates `stock`, `tiled` and the bit-for-bit variants that passed
/// their self-test; the fastest is kept only if it beats `stock`. Every
/// stage-1 kernel gives the record's bits, so each round's draft ids are the
/// record body's; they are kept, with the committed position each round
/// proposed from, as the tape. Stage 2 (variants on, and a changed-split
/// variant of the stage-1 pick passing its self-test) runs the same request
/// again (same prompt, same budget: the same rounds while the drafts match),
/// rotating the pick and its changed-split variants. A variant is adopted only
/// if every one of its rounds proposed exactly the tape's ids from the tape's
/// position, the pick's own rounds matched the tape as well, and it beats the
/// pick; otherwise the pick stays. The target decides every emitted token
/// either way; the check keeps the drafter's proposals, and so the
/// acceptance, the record's on the trial's rounds.
///
/// A round is the host time from its proposal to the next. A proposal is
/// built with its round's kernel: set at `roundBoundary()` (the top of every
/// proposal) and, for a block built before the previous round's readback
/// (`Qwen35DFlash2Assistant.speculateBlock`), at `aheadOfRound()`, so a
/// round's drafter work runs the kernel the round is timed for. The first
/// round is discarded.
enum DFlash2KernelTrial {
    typealias Kernel = DFlash2TensorMatmul.Kernel

    nonisolated(unsafe) static var armed = false
    nonisolated(unsafe) static var active = false
    nonisolated(unsafe) private static var candidates: [Kernel] = [.stock]
    nonisolated(unsafe) private static var perCandidate = 12
    nonisolated(unsafe) private static var stage = 0
    nonisolated(unsafe) private static var roundIndex = 0
    nonisolated(unsafe) private static var lastBoundary: UInt64 = 0
    nonisolated(unsafe) private static var durations: [Int: UInt64] = [:]
    nonisolated(unsafe) private static var proposals: [Int: Proposal] = [:]
    nonisolated(unsafe) private static var tape: [Int: (position: Int, ids: [Int32])] = [:]
    nonisolated(unsafe) private static var onEnough: (() -> Void)?
    nonisolated(unsafe) private static var pick = Kernel.stock
    nonisolated(unsafe) private static var proposalCount = 0
    nonisolated(unsafe) private static var log = ""
    /// Host times: stage 1 begun, its request done, stage 2's self-tests done.
    nonisolated(unsafe) private static var marks: [UInt64] = []
    /// The rounds both requests budget for (the same budget, the same rounds).
    nonisolated(unsafe) private(set) static var requestRounds = 0

    private struct Proposal {
        let tag: Int  // the candidate it was built with; -1 unknown
        let position: Int
        let tokens: MLXArray
    }

    static let adoptMargin = 0.005
    static let outlierFactor = 1.5
    /// Timed rounds per candidate: stage 1 with variants (12 for the two-way
    /// trial), stage 2.
    static let roundsPerVariant = 8
    static let roundsPerSplit = 6

    /// The discarded round, the seed-side boundary, then every candidate's rounds.
    static var roundsNeeded: Int { 2 + candidates.count * perCandidate }

    @inline(__always) static func roundBoundary() {
        guard active else { return }
        boundary()
    }

    private static func boundary() {
        let now = DispatchTime.now().uptimeNanoseconds
        if roundIndex >= 2 { durations[roundIndex - 1] = now - lastBoundary }
        lastBoundary = now
        DFlash2TensorMatmul.current = candidates[roundIndex % candidates.count]
        roundIndex += 1
        if roundIndex >= roundsNeeded {
            active = false
            let enough = onEnough
            onEnough = nil
            enough?()
        }
    }

    /// Before a block is built ahead of its round: sets that round's kernel
    /// (the next boundary's) and returns its candidate; nil with no trial.
    static func aheadOfRound() -> Int? {
        guard active else { return nil }
        let tag = roundIndex % candidates.count
        DFlash2TensorMatmul.current = candidates[tag]
        return tag
    }

    /// The round's proposal, built at its boundary (`proposeBlock`) from
    /// `position` committed rows.
    @inline(__always) static func recordProposed(_ tokens: MLXArray, position: Int) {
        guard active, roundIndex >= 1 else { return }
        record(tokens, tag: (roundIndex - 1) % candidates.count, position: position)
    }

    /// The round's proposal, built ahead with `tag` (`aheadOfRound()`) and
    /// adopted at its boundary from `position` committed rows.
    @inline(__always) static func recordAdopted(_ tokens: MLXArray, tag: Int?, position: Int) {
        guard active, roundIndex >= 1 else { return }
        record(tokens, tag: tag ?? -1, position: position)
    }

    private static func record(_ tokens: MLXArray, tag: Int, position: Int) {
        proposals[roundIndex - 1] = Proposal(tag: tag, position: position, tokens: tokens)
    }

    /// Stage 1's candidates, and the budget both requests use.
    static func beginFirstStage() {
        candidates = [.stock, .tiledStock] + DFlash2TensorMatmul.passedBitwiseVariants
        perCandidate = candidates.count > 2 ? roundsPerVariant : 12
        stage = 1
        pick = .stock
        tape = [:]
        log = ""
        proposalCount = 0
        marks = [DispatchTime.now().uptimeNanoseconds]
        requestRounds = max(
            roundsNeeded, DFlash2TensorMatmul.variantsEnabled ? 2 + 4 * roundsPerSplit : 0)
    }

    static func begin(onEnough: @escaping () -> Void) {
        guard armed else { return }
        roundIndex = 0
        lastBoundary = 0
        durations = [:]
        proposals = [:]
        self.onEnough = onEnough
        DFlash2TensorMatmul.current = candidates[0]
        active = true
    }

    private static func median(_ values: [UInt64]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 1
            ? Double(sorted[mid]) : (Double(sorted[mid - 1]) + Double(sorted[mid])) / 2
    }

    private typealias Timing = (median: Double?, kept: Int, timed: Int)

    /// Per candidate. A round counts for the candidate it was scheduled for
    /// only when its proposal was built with that candidate.
    private static func timings() -> [Timing] {
        var times = Array(repeating: [UInt64](), count: candidates.count)
        for (index, duration) in durations {
            let tag = index % candidates.count
            guard proposals[index]?.tag == tag else { continue }
            times[tag].append(duration)
        }
        return times.map { values in
            guard let first = median(values) else { return (nil, 0, 0) }
            let kept = values.filter { Double($0) <= outlierFactor * first }
            return (median(kept), kept.count, values.count)
        }
    }

    /// The fastest eligible candidate if it beats candidate 0 by more than
    /// the margin, else candidate 0.
    private static func choose(_ t: [Timing], eligible: [Bool]) -> Int {
        guard let reference = t[0].median else { return 0 }
        var best = 0
        for c in 1 ..< t.count where eligible[c] {
            if let m = t[c].median, m < (t[best].median ?? .infinity) { best = c }
        }
        return best > 0 && t[best].median! < reference * (1 - adoptMargin) ? best : 0
    }

    private static func describe(_ t: [Timing], _ note: (Int) -> String = { _ in "" }) -> String {
        candidates.indices.map { c in
            "\(candidates[c].name) "
                + (t[c].median.map { String(format: "%.2f", $0 / 1e6) } ?? "-")
                + " ms (\(t[c].kept)/\(t[c].timed)\(note(c)))"
        }.joined(separator: ", ")
    }

    private static func change(_ t: [Timing], _ c: Int) -> String {
        guard c > 0, let m = t[c].median, let r = t[0].median else { return "" }
        return String(format: " (%+.2f%%)", (m / r - 1) * 100)
    }

    /// The stage-1 pick and (variants on) the tape, once.
    private static func concludeFirstStage() {
        guard stage == 1, log.isEmpty else { return }
        let t = timings()
        let best = choose(t, eligible: Array(repeating: true, count: candidates.count))
        pick = candidates[best]
        log = describe(t) + " -> \(pick.name)\(change(t, best))"
        proposalCount += roundIndex
        if DFlash2TensorMatmul.variantsEnabled {
            for (index, p) in proposals {
                tape[index] = (p.position, p.tokens.asType(.int32).asArray(Int32.self))
            }
        }
        proposals = [:]
    }

    /// After the first request: `concludeFirstStage`, then the changed-split
    /// variants of the pick that pass their self-test. True when the second
    /// request is to run.
    static func beginSecondStage() -> Bool {
        active = false
        onEnough = nil
        guard armed, stage == 1 else { return false }
        marks.append(DispatchTime.now().uptimeNanoseconds)
        concludeFirstStage()
        guard DFlash2TensorMatmul.variantsEnabled, !tape.isEmpty else { return false }
        var passed: [Kernel] = []
        var tested: [String] = []
        for kernel in DFlash2TensorMatmul.splitVariants(of: pick) {
            let test = DFlash2TensorMatmul.selfTest(kernel)
            if test.passed { passed.append(kernel) }
            tested.append(
                kernel.name
                    + (test.passed ? String(format: " max rel %.1e", test.maxRelative) : " FAILED"))
        }
        log += "; split variants self-test [\(tested.joined(separator: ", "))]"
        marks.append(DispatchTime.now().uptimeNanoseconds)
        guard !passed.isEmpty else { return false }
        candidates = [pick] + passed
        perCandidate = roundsPerSplit
        stage = 2
        return true
    }

    /// Installs the verdict (the stage-1 pick unless a stage-2 variant
    /// qualified; the copies are freed when the stored layout stays), logs
    /// one line and disarms. Safe when nothing ran.
    static func finish(elapsedNanoseconds: UInt64) {
        active = false
        onEnough = nil
        guard armed else { return }
        armed = false
        concludeFirstStage()
        var adopted = pick
        if stage == 2 {
            proposalCount += roundIndex
            let t = timings()
            let n = candidates.count
            var recorded = Array(repeating: 0, count: n)
            var compared = recorded
            var equal = recorded
            var firstOff = Int.max
            var unknownDiffers = false
            for index in proposals.keys.sorted() {
                let p = proposals[index]!
                if p.tag >= 0 { recorded[p.tag] += 1 }
                guard index < firstOff, let entry = tape[index], entry.position == p.position else {
                    firstOff = min(firstOff, index)
                    continue
                }
                let same = p.tokens.asType(.int32).asArray(Int32.self) == entry.ids
                if p.tag < 0 {
                    if !same { unknownDiffers = true }
                    continue
                }
                compared[p.tag] += 1
                if same { equal[p.tag] += 1 }
            }
            func clean(_ c: Int) -> Bool {
                recorded[c] > 0 && compared[c] == recorded[c] && equal[c] == recorded[c]
            }
            let eligible = (0 ..< n).map { c in c > 0 && clean(c) && clean(0) && !unknownDiffers }
            let best = choose(t, eligible: eligible)
            adopted = candidates[best]
            log += "; " + describe(t) { c in ", ids \(equal[c])/\(recorded[c])" }
                + " -> \(adopted.name)\(change(t, best))"
        }
        DFlash2TensorMatmul.current = adopted
        if adopted.tiled {
            DFlash2TensorMatmul.dropTestOperands()
        } else {
            DFlash2TensorMatmul.dropTiledCopies()
        }
        marks.append(DispatchTime.now().uptimeNanoseconds)
        let phases = zip(marks.dropFirst(), marks).map { String(format: "%.0f", Double($0 - $1) / 1e6) }
        FileHandle.standardError.write(
            Data(
                ("dflash2 kernel trial: \(log); adopted \(adopted.name) "
                    + "(\(adopted.tiled ? adopted.bitwise ? "bitwise" : "changed split" : "stored layout")); "
                    + "\(proposalCount) proposals; "
                    + String(format: "%.0f ms", Double(elapsedNanoseconds) / 1e6)
                    + " (\(phases.joined(separator: " + ")))\n").utf8))
        proposals = [:]
        tape = [:]
        durations = [:]
        stage = 0
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

    /// The stacked weight once built (nil before its first use).
    var residentWeight: MLXArray? { weight }

    /// The stacked weight, concatenated on first use, or nil when the stack
    /// does not apply.
    func stacked(gate: Linear, up: Linear) -> MLXArray? {
        guard Self.enabled, gate.bias == nil, up.bias == nil,
            gate.weight.dtype == up.weight.dtype, gate.weight.dim(1) == up.weight.dim(1),
            gate.weight.ndim == 2, up.weight.ndim == 2
        else { return nil }
        if weight == nil {
            weight = concatenated([gate.weight, up.weight], axis: 0)
            boundary = gate.weight.dim(0)
        }
        return weight
    }

    /// `(gate(x), up(x))` from one matmul, or nil when the stack does not apply.
    func apply(_ x: MLXArray, gate: Linear, up: Linear) -> (MLXArray, MLXArray)? {
        guard let weight = stacked(gate: gate, up: up) else { return nil }
        let y = DFlash2TensorMatmul.apply(x, weight: weight) ?? matmul(x, weight.T)
        return (y[.ellipsis, ..<boundary], y[.ellipsis, boundary...])
    }
}

private final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    private let gateUp = DFlash2GateUpStack()

    /// As `DFlash2Attention.residencyWeights`: the gate|up stack and
    /// down_proj as `DFlash2TensorMatmul` reads them.
    func residencyWeights() -> (read: [MLXArray], replaced: [MLXArray]) {
        var read: [MLXArray] = []
        var replaced: [MLXArray] = []
        if let stacked = gateUp.residentWeight {
            read.append(DFlash2TensorMatmul.readWeight(stacked))
            replaced += [gate.weight, up.weight]
        }
        let d = DFlash2TensorMatmul.readWeight(down.weight)
        read.append(d)
        if d !== down.weight { replaced.append(down.weight) }
        return (read, replaced)
    }

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

    /// The weights a block-width forward reads through `DFlash2TensorMatmul`.
    func tensorWeights() -> [MLXArray] {
        (gateUp.stacked(gate: gate, up: up).map { [$0] } ?? [])
            + (down.bias == nil ? [down.weight] : [])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let (g, u) = gateUp.apply(x, gate: gate, up: up) {
            return DFlash2TensorMatmul.linear(down, DFlash2SwiGLU.apply(g, u))
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

    func absorbContext(_ context: MLXArray, rope: RoPELayer, cache: KVCache) -> Bool {
        selfAttn.absorbContext(context, rope: rope, cache: cache)
    }

    func tensorWeights() -> [MLXArray] {
        (selfAttn.oProj.bias == nil ? [selfAttn.oProj.weight] : []) + mlp.tensorWeights()
    }

    /// Every array of this layer a block forward reads, in forward order
    /// (the attention's, then the MLP's, then every other parameter), and
    /// the stored weights a stack or a tiled copy stands in for.
    func residencyWeights() -> (read: [MLXArray], replaced: [MLXArray]) {
        let attention = selfAttn.residencyWeights()
        let feedForward = mlp.residencyWeights()
        let replaced = attention.replaced + feedForward.replaced
        let skipped = Set(replaced.map { ObjectIdentifier($0) })
        let rest = parameters().flattened().map(\.1).filter {
            !skipped.contains(ObjectIdentifier($0))
        }
        return (attention.read + feedForward.read + rest, replaced)
    }

    /// The two convolutions' tap projections (16-row tensor kernel too).
    func projectionWeights() -> [MLXArray] {
        [attentionConv.kernelProjection, mlpConv.kernelProjection]
            .filter { $0.bias == nil }.map { $0.weight }
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray?, rope: RoPELayer, cache: KVCache,
        masks: DFlash2SlidingMaskMemo
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let joined = selfAttn.joins(context)
            ? context.flatMap { attentionConv.prepare(normed, joining: $0) } : nil
        let (attentionInput, attentionTaps) =
            joined.map { ($0.0, $0.1) } ?? attentionConv.prepare(normed)
        let attended = attentionConv.finish(
            selfAttn(
                attentionInput, context: context, joined: joined?.2, rope: rope, cache: cache,
                masks: masks),
            projection: attentionTaps, residual: x)
        let (mlpInput, mlpTaps) = mlpConv.prepare(postAttentionLayerNorm(attended))
        return mlpConv.finish(mlp(mlpInput), projection: mlpTaps, residual: attended)
    }

    /// `callAsFunction` through `DFlash2Attention.speculative`.
    func speculative(
        _ x: MLXArray, base: MLXArray, confirmed: MLXArray, queryOffset: MLXArray,
        rope: RoPELayer, cache: DFlash2BlockKVCache, keyMask: MLXArray
    ) -> (hidden: MLXArray, keys: MLXArray, values: MLXArray)? {
        let (attentionInput, attentionTaps) = attentionConv.prepare(inputLayerNorm(x))
        guard
            let a = selfAttn.speculative(
                attentionInput, base: base, confirmed: confirmed, queryOffset: queryOffset,
                rope: rope, cache: cache, keyMask: keyMask)
        else { return nil }
        let attended = attentionConv.finish(a.output, projection: attentionTaps, residual: x)
        let (mlpInput, mlpTaps) = mlpConv.prepare(postAttentionLayerNorm(attended))
        return (mlpConv.finish(mlp(mlpInput), projection: mlpTaps, residual: attended), a.keys, a.values)
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

    /// Four-wide loads when a chunk length is a multiple of four.
    /// `MLXFAST_DFLASH_TOPK_VEC=0` reads one logit at a time.
    static let vectorScan: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_TOPK_VEC"]?
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
        let vector = vectorScan && vocabularySize % (chunks * 4) == 0
        let template: [(String, any KernelTemplateArg)] = [
            ("NV", vocabularySize), ("S", chunks), ("TPG", threads), ("KTOP", k),
            ("VEC", vector ? 1 : 0),
            ("HALF", logits.dtype == .float16 ? 1 : 0),
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
            // Insert one (key, index). The caller offers indices in increasing
            // order, so an equal key keeps the higher index.
            template <int KK>
            inline void mlxfast_topk_consider(thread uint (&k)[KK], thread uint (&id)[KK],
                                               uint kx, uint ix) {
                if (kx >= k[KK - 1]) {
                    for (int j = 0; j < KK; j++) {
                        bool sw = kx >= k[j];
                        uint tk = k[j], ti = id[j];
                        k[j] = sw ? kx : tk; id[j] = sw ? ix : ti;
                        kx = sw ? tk : kx; ix = sw ? ti : ix;
                    }
                }
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
            if (VEC && (CH % 4u) == 0u) {
                // Increasing indices, four-wide. CH % 4 covers the chunk.
                for (uint v = lo + t * 4u; v + 3u < hi; v += TPG * 4u) {
                    if (HALF) {
                        const half4 q = *(const device half4*)(x + v);
                        mlxfast_topk_consider<KK>(k, id, mlxfast_topk_key(float(q[0])), v);
                        mlxfast_topk_consider<KK>(k, id, mlxfast_topk_key(float(q[1])), v + 1u);
                        mlxfast_topk_consider<KK>(k, id, mlxfast_topk_key(float(q[2])), v + 2u);
                        mlxfast_topk_consider<KK>(k, id, mlxfast_topk_key(float(q[3])), v + 3u);
                    } else {
                        const float4 q = *(const device float4*)(x + v);
                        mlxfast_topk_consider<KK>(k, id, mlxfast_topk_key(q[0]), v);
                        mlxfast_topk_consider<KK>(k, id, mlxfast_topk_key(q[1]), v + 1u);
                        mlxfast_topk_consider<KK>(k, id, mlxfast_topk_key(q[2]), v + 2u);
                        mlxfast_topk_consider<KK>(k, id, mlxfast_topk_key(q[3]), v + 3u);
                    }
                }
            } else {
                for (uint v = lo + t; v < hi; v += TPG) {
                    mlxfast_topk_consider<KK>(k, id, mlxfast_topk_key(float(x[v])), v);
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

    /// Four-wide loads of the rank-256 dot. Each product is still added in
    /// increasing index order, so the edge equals the scalar chain.
    /// `MLXFAST_DFLASH_WALK_VEC=0` reads one rank element at a time.
    static let vectorRank: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_WALK_VEC"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static func select(
        candidates: MLXArray, unary: MLXArray, projected: MLXArray, anchor: MLXArray,
        predecessorCodebook: MLXArray, successorCodebook: MLXArray
    ) -> MLXArray? {
        guard enabled, candidates.ndim == 3, candidates.dim(0) == 1, anchor.size == 1,
            unary.dtype == .float32 || unary.dtype == .float16 || unary.dtype == .bfloat16
        else { return nil }
        let length = candidates.dim(1)
        let k = candidates.dim(2)
        let rank = projected.dim(-1)
        guard length >= 2, k >= 1, k <= 32, rank > 0 else { return nil }
        if narrowOperands, unary.ndim == 3, unary.dim(0) == 1, projected.ndim == 3,
            projected.dim(0) == 1, predecessorCodebook.dtype == successorCodebook.dtype,
            [DType.bfloat16, .float16, .float32].contains(predecessorCodebook.dtype),
            [DType.bfloat16, .float16, .float32].contains(projected.dtype),
            narrowVerified(
                codebook: predecessorCodebook.dtype, projected: projected.dtype,
                unary: unary.dtype)
        {
            return selectNarrow(
                candidates: candidates, unary: unary, projected: projected, anchor: anchor,
                predecessorCodebook: predecessorCodebook, successorCodebook: successorCodebook)
        }
        return selectWide(
            candidates: candidates, unary: unary, projected: projected, anchor: anchor,
            predecessorCodebook: predecessorCodebook, successorCodebook: successorCodebook)
    }

    /// The walk over operands widened to FP32 first (as recorded).
    private static func selectWide(
        candidates: MLXArray, unary: MLXArray, projected: MLXArray, anchor: MLXArray,
        predecessorCodebook: MLXArray, successorCodebook: MLXArray
    ) -> MLXArray? {
        let length = candidates.dim(1)
        let k = candidates.dim(2)
        let rank = projected.dim(-1)
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
            template: [
                ("L", length), ("K", k), ("R", rank),
                ("WALKVEC", vectorRank && rank % 4 == 0 ? 1 : 0),
            ],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[length]],
            outputDTypes: [.int32])[0]
        return path.reshaped([1, length])
    }

    /// The walk reading its operands as they are: the batch axis squeezed
    /// (a view) where `[0]` gathered a copy of candidates, scores and
    /// projections, and the codebook rows and projections in their own dtype,
    /// widened to FP32 in the kernel (exact) where four launches widened them
    /// first. Same FP32 values, same arithmetic, same order: seven launches
    /// fewer per proposal. Self-tested once per operand dtypes (at bind for
    /// the ones in use) against the widened walk on random operands, every
    /// path id compared; `MLXFAST_DFLASH_WALK_NARROW=0` widens first.
    static let narrowOperands: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_WALK_NARROW"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    private static func selectNarrow(
        candidates: MLXArray, unary: MLXArray, projected: MLXArray, anchor: MLXArray,
        predecessorCodebook: MLXArray, successorCodebook: MLXArray
    ) -> MLXArray {
        let length = candidates.dim(1)
        let c = candidates.squeezed(axis: 0)
        let path = narrowKernel(
            [
                take(predecessorCodebook, anchor, axis: 0).reshaped([-1]),
                take(predecessorCodebook, c[0 ..< (length - 1)], axis: 0).reshaped([-1]),
                take(successorCodebook, c, axis: 0).reshaped([-1]),
                projected.squeezed(axis: 0).reshaped([-1]),
                unary.squeezed(axis: 0).reshaped([-1]),
                c.asType(.uint32).reshaped([-1]),
            ],
            template: [("L", length), ("K", candidates.dim(2)), ("R", projected.dim(-1))],
            grid: (32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[length]], outputDTypes: [.int32])[0]
        return path.reshaped([1, length])
    }

    private static let narrowLock = NSLock()
    nonisolated(unsafe) private static var narrowVerdicts: [String: Bool] = [:]

    /// Runs the narrow walk's self-test for these dtypes now (load time).
    static func prepare(codebook: DType, projected: DType, unary: DType) {
        guard enabled, narrowOperands else { return }
        _ = narrowVerified(codebook: codebook, projected: projected, unary: unary)
    }

    private static func narrowVerified(codebook: DType, projected: DType, unary: DType) -> Bool {
        narrowLock.withLock {
            let key = "\(codebook) \(projected) \(unary)"
            if let verdict = narrowVerdicts[key] { return verdict }
            var same = true
            var walks = 0
            do {
                try withError { error in
                    let (vocab, length, k, rank) = (4096, 15, 16, 256)
                    for seed in 0 ..< 16 {
                        func key(_ salt: Int) -> MLXArray { MLXRandom.key(UInt64(seed * 8 + salt)) }
                        let pred = MLXRandom.normal([vocab, rank], key: key(0)).asType(codebook)
                        let succ = MLXRandom.normal([vocab, rank], key: key(1)).asType(codebook)
                        let proj = MLXRandom.normal([1, length, rank], key: key(2)).asType(projected)
                        let una = (MLXRandom.normal([1, length, k], key: key(3)) * 4).asType(unary)
                        let cand = MLXRandom.randInt(Int32(0) ..< Int32(vocab), [1, length, k], key: key(4))
                            .asType(.uint32)
                        let anchor = MLXArray([Int32(seed * 97 % vocab)])
                        guard
                            let wide = selectWide(
                                candidates: cand, unary: una, projected: proj, anchor: anchor,
                                predecessorCodebook: pred, successorCodebook: succ)
                        else { same = false; return }
                        let narrow = selectNarrow(
                            candidates: cand, unary: una, projected: proj, anchor: anchor,
                            predecessorCodebook: pred, successorCodebook: succ)
                        same = same && all(wide .== narrow).item(Bool.self)
                        walks += 1
                    }
                    try error.check()
                }
            } catch {
                same = false
            }
            narrowVerdicts[key] = same
            FileHandle.standardError.write(
                ("dflash2 greedy walk narrow operands (\(key)): "
                    + (same
                        ? "self-test passed: \(walks) walks of 15 ids identical; narrow\n"
                        : "self-test failed; widened operands kept\n")).data(using: .utf8)!)
            return same
        }
    }

    private static let narrowKernel = MLXFast.metalKernel(
        name: "mlxfast_dflash_fused_greedy_walk_narrow",
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
                    auto pred_ptr = (i == 0) ? anchor_predecessor : (previous + pred_base);
                    auto proj_ptr = projected + i * R;
                    auto succ_ptr = next + succ_base;
                    float edge = 0.0f;
                    #pragma clang loop unroll(full)
                    for (uint d = 0; d < R; d++) {
                        edge += (float(pred_ptr[d]) * float(proj_ptr[d])) * float(succ_ptr[d]);
                    }
                    score = float(unary[i * K + c]) + edge;
                }
                float m = simd_max(score);
                uint sel = simd_min((c < K && score == m) ? c : 0xffffffffu);
                previous_slot = sel;
                if (c == 0) path[i] = int(cand[i * K + sel]);
            }
            """)

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
                    if (WALKVEC && (R % 4u) == 0u) {
                        for (uint d = 0; d < R; d += 4u) {
                            const float4 pd = *(const device float4*)(pred_ptr + d);
                            const float4 qd = *(const device float4*)(proj_ptr + d);
                            const float4 sd = *(const device float4*)(succ_ptr + d);
                            edge += (pd[0] * qd[0]) * sd[0];
                            edge += (pd[1] * qd[1]) * sd[1];
                            edge += (pd[2] * qd[2]) * sd[2];
                            edge += (pd[3] * qd[3]) * sd[3];
                        }
                    } else {
                        #pragma clang loop unroll(full)
                        for (uint d = 0; d < R; d++) {
                            edge += (pred_ptr[d] * proj_ptr[d]) * succ_ptr[d];
                        }
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

// MARK: - Residency prefetch

/// One command buffer that binds a group of arrays: a single thread reads the
/// first element of each input. It exists for the buffers it binds, not for
/// its output (see `DFlash2ResidencyPrefetch`).
private enum DFlash2ResidencyTouch {
    /// Inputs per kernel; with the output well inside Metal's 31 buffer slots.
    static let maximumInputs = 24

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [Int: MLXFast.MLXFastKernel] = [:]

    private static func kernel(inputs count: Int) -> MLXFast.MLXFastKernel {
        lock.withLock {
            if let kernel = kernels[count] { return kernel }
            let names = (0 ..< count).map { "w\($0)" }
            let reads = names.map { "acc += static_cast<float>(\($0)[0]);" }
                .joined(separator: "\n")
            let kernel = MLXFast.metalKernel(
                name: "dflash2_residency_touch_\(count)", inputNames: names,
                outputNames: ["out"],
                source: "float acc = 0.0f;\n" + reads + "\nout[0] = acc;\n",
                ensureRowContiguous: false)
            kernels[count] = kernel
            return kernel
        }
    }

    /// One touch per `maximumInputs` arrays of `arrays`.
    static func touch(_ arrays: [MLXArray]) -> [MLXArray] {
        stride(from: 0, to: arrays.count, by: maximumInputs).map { start in
            let chunk = Array(arrays[start ..< min(start + maximumInputs, arrays.count)])
            return kernel(inputs: chunk.count)(
                chunk, grid: (1, 1, 1), threadGroup: (1, 1, 1),
                outputShapes: [[1]], outputDTypes: [.float32])[0]
        }
    }
}

/// Makes the drafter's weights GPU-resident again inside the prompt forward
/// of a request that will draft with it, behind that forward's own work.
///
/// Nothing reads the drafter between one request's last round and the next
/// request's first: the target's prompt forwards and the benchmark's idle
/// gates leave its weights unwired, and the first block forward of the next
/// decode pays to make them resident again. Command-buffer timestamps show
/// it: the round's buffers are committed within half a millisecond, but the
/// GPU idles before each one in proportion to the drafter bytes it binds,
/// about 0.1 ms per 10 MB, ~37 ms over the ~3.8 GB the forward reads. Later
/// rounds show no such gaps.
///
/// Making a buffer resident is done as its command buffer is submitted, while
/// the GPU keeps running the buffers queued ahead of it: a standalone probe
/// (eight 512 MB arrays unwired by a 30 s idle, each bound by a one-thread
/// kernel queued behind a 9.4 ms matmul chain) ran in 89.2 ms against 89.4 ms
/// for the chains alone, while the same binds with nothing queued ahead cost
/// ~5.5 ms each. A decode's prompt forward keeps up to MLX's ten buffers
/// (~20-45 ms of prompt layers) queued ahead of the host. So an engine build
/// that drafts arms this prefetch, and its prompt forward binds the drafter's
/// arrays (`DFlash2DraftModel.residencyArrays`, in forward order) in four
/// groups of about equal bytes, each right after the forward's early
/// submission at layer 8, 16, 32 and 48 (`Qwen35TrunkSubmission.promptFused`).
/// Any group still due at the end of the forward is bound there.
///
/// Each group costs one single-thread kernel per 24 arrays on the GPU. No
/// value of either model is read into any result, and nothing about the
/// arithmetic changes. `DARKBLOOM_DFLASH2_RESIDENCY_PREFETCH=0` turns it off.
public enum DFlash2ResidencyPrefetch {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH2_RESIDENCY_PREFETCH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The prompt layers after which one group each is due.
    static let dueAfterLayers = [8, 16, 32, 48]

    private static let lock = NSLock()
    nonisolated(unsafe) private static var groups: [[MLXArray]] = []
    nonisolated(unsafe) private static var nextGroup = 0

    /// Arms the prefetch for the next prompt forward: called when an engine
    /// that drafts with `drafter` is built.
    static func arm(_ drafter: DFlash2DraftModel) {
        guard enabled else { return }
        let arrays = drafter.residencyArrays()
        let total = arrays.reduce(0) { $0 + $1.nbytes }
        let share = max(1, total / dueAfterLayers.count)
        var built: [[MLXArray]] = []
        var group: [MLXArray] = []
        var bytes = 0
        for array in arrays {
            group.append(array)
            bytes += array.nbytes
            if bytes >= share * (built.count + 1), built.count < dueAfterLayers.count - 1 {
                built.append(group)
                group = []
            }
        }
        if !group.isEmpty { built.append(group) }
        lock.withLock {
            groups = built
            nextGroup = 0
        }
    }

    /// Submits every group due once the prompt forward has submitted its
    /// first `completedLayers` layers. A no-op unless armed.
    static func submitDue(completedLayers: Int) {
        let due: [[MLXArray]] = lock.withLock {
            var due: [[MLXArray]] = []
            while nextGroup < groups.count,
                nextGroup >= dueAfterLayers.count
                    || dueAfterLayers[nextGroup] <= completedLayers
            {
                due.append(groups[nextGroup])
                nextGroup += 1
            }
            if nextGroup >= groups.count {
                groups = []
                nextGroup = 0
            }
            return due
        }
        guard !due.isEmpty else { return }
        asyncEval(due.flatMap { DFlash2ResidencyTouch.touch($0) })
    }

    /// Submits whatever is still due: the end of a prompt forward.
    static func submitRemaining() {
        submitDue(completedLayers: .max)
    }
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
    // Sliding masks depend only on block geometry. Keep the memo with the
    // drafter so repeated speculative forwards can reuse the same graph
    // (ercumentyildirim / terrapinelf `ff96d1e`).
    private let masks = DFlash2SlidingMaskMemo()
    private var target: (any DFlash2Target)?
    private var maskTokenEmbedding: MLXArray?
    /// Retained broadcast of `maskTokenEmbedding` for the common single-stream
    /// block shape `[1, blockSize-1, hidden]`. Rebuilding that broadcast every
    /// propose round repeats an identical geometry graph.
    private var cachedMaskEmbeddingBlock: (cols: Int, array: MLXArray)?

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
        let maskEmbedding = target.embedTokensForDFlash2(
            MLXArray([Int32(config.maskTokenId)], [1, 1])).asType(dtype)
        eval(maskEmbedding)
        self.maskTokenEmbedding = maskEmbedding
        // The one-launch concatenations' self-tests, before any timed forward:
        // each layer's [context; block] rows and the target's tapped states.
        DFlash2Concat.prepare(inputs: 2, dtype: dtype)
        DFlash2Concat.prepare(inputs: 2, dtype: .float16)
        DFlash2SwiGLU.prepare(dtype: dtype, width: config.intermediateSize)
        DFlash2StridedRMSNorm.prepare(dtype: dtype, headDim: config.headDim, eps: config.rmsNormEps)
        _ = DFlash2GroupedDynamicCausalConv.joinVerified(
            dtype, config.dflash.convKernelSize, config.dflash.convGroupSize, config.hiddenSize)
        DFlash2QKPrework.prepare(
            rope: rope, base: config.ropeTheta, dtype: dtype, heads: config.attentionHeads,
            kvHeads: config.kvHeads, headDim: config.headDim, eps: config.rmsNormEps)
        DFlash2GreedyWalk.prepare(
            codebook: candidateSelector.predecessorCodebook.dtype,
            projected: candidateSelector.hiddenProjection.weight.dtype, unary: .float32)
        DFlash2Concat.prepare(inputs: config.targetLayerIds.count, dtype: .float16)
    }

    /// Builds and self-tests the tiled copies of the weights the block
    /// forward reads through `DFlash2TensorMatmul` (every layer's o_proj,
    /// stacked gate|up and down_proj). `fc` keeps its stored layout: the
    /// prompt's context rows read it through the core's GEMM in the seed, so
    /// a copy only the rounds read would be one more array to make GPU-
    /// resident again at each decode window's first round.
    func prepareTiledWeights() -> Bool {
        let layerWeights = layers.flatMap { $0.tensorWeights() }
        _ = DFlash2TensorMatmul.prepareSwapped(
            layerWeights + layers.flatMap { $0.projectionWeights() } + (fc.bias == nil ? [fc.weight] : []),
            rows32: layers.compactMap { $0.selfAttn.stackedQKVWeight() })
        return DFlash2TensorMatmul.prepareTiled(layerWeights)
    }

    /// Every array a block forward reads that this drafter owns, in forward
    /// order: each layer's (`DFlash2DecoderLayer.residencyWeights`), then
    /// every other parameter (`fc`, the norms, the candidate selector's
    /// projection and codebooks). A stored weight that a stack or a tiled
    /// copy stands in for is left out; the block forward does not read it.
    /// The target's embedding and head are the target's, read by its own
    /// forwards.
    func residencyArrays() -> [MLXArray] {
        var arrays: [MLXArray] = []
        var seen = Set<ObjectIdentifier>()
        var replaced = Set<ObjectIdentifier>()
        for layer in layers {
            let weights = layer.residencyWeights()
            for array in weights.replaced { replaced.insert(ObjectIdentifier(array)) }
            for array in weights.read where seen.insert(ObjectIdentifier(array)).inserted {
                arrays.append(array)
            }
        }
        for (_, array) in parameters().flattened()
        where !replaced.contains(ObjectIdentifier(array))
            && seen.insert(ObjectIdentifier(array)).inserted
        {
            arrays.append(array)
        }
        return arrays
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
    ///   - leadingLayers: when positive, the trunk `asyncEval`s its hidden state
    ///     as soon as this many layers (at most all of them) are built, so the
    ///     GPU starts them, and the context projection they read, while the
    ///     host builds the rest.
    func hiddenStates(
        _ inputs: MLXArray,
        targetHidden: MLXArray?,
        cache: [KVCache],
        logitsStart: Int,
        submittingLeadingLayers leadingLayers: Int = 0
    ) throws -> MLXArray {
        guard let target else { throw DFlash2Error.notBound }
        guard cache.count == layers.count else {
            throw DFlash2Error.invalidCacheCount(expected: layers.count, actual: cache.count)
        }
        if let targetHidden, targetHidden.dim(-1) != config.targetHiddenSize {
            throw DFlash2Error.targetHiddenSizeMismatch(
                expected: config.targetHiddenSize, actual: targetHidden.dim(-1))
        }

        // Both crossings from the target cast here. The target's embedding is a
        // MODULE call, never a raw weight read.
        // Every proposed block has one committed anchor followed by copies of
        // the same mask token. The target embedding is a packed 2-bit lookup
        // followed by an inverse Hadamard transform, so embedding all mask
        // positions independently repeats identical dequantization and
        // transform work. The mask row was computed and evaluated at bind
        // time; broadcast that exact value across the block. Keep the general
        // one-row case unchanged.
        let embeddedInputs: MLXArray
        if inputs.dim(1) > 1 {
            embeddedInputs = try blockEmbedding(
                anchorIDs: inputs[0..., ..<1], maskColumns: inputs.dim(1) - 1)
        } else {
            embeddedInputs = target.embedTokensForDFlash2(inputs).asType(dtype)
        }
        var h = embeddedInputs
        if config.dflash.inputEmbeddingScale != 1 {
            h = h * config.dflash.inputEmbeddingScale
        }
        let context = targetHidden.map { contextProjection($0) }

        let submitAfter = DFlash2DraftSubmission.layers
        let leadAt = leadingLayers > 0 ? min(leadingLayers, layers.count) : 0
        for (index, layer) in layers.enumerated() {
            h = layer(h, context: context, rope: rope, cache: cache[index], masks: masks)
            // EARLY SUBMISSION: hand the GPU the drafter layers built so far
            // while the host builds the rest and the head. Same kernels, same
            // order; only command-buffer boundaries move. One submission per
            // layer at most, whichever of the two asks for it.
            if index + 1 == leadAt
                || (!submitAfter.isEmpty && submitAfter.contains(index + 1))
            {
                asyncEval([h])
            }
        }
        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }
        return norm(h)
    }

    /// `[anchor; masks]` embedded (the anchor `[B, 1]`, host or device ids).
    func blockEmbedding(anchorIDs: MLXArray, maskColumns cols: Int) throws -> MLXArray {
        guard let target, let maskEmbedding = maskTokenEmbedding else {
            throw DFlash2Error.notBound
        }
        let anchorEmbedding = target.embedTokensForDFlash2(anchorIDs).asType(dtype)
        let batch = anchorIDs.dim(0)
        var repeatedMasks: MLXArray
        if batch == 1,
            let cached = cachedMaskEmbeddingBlock,
            cached.cols == cols
        {
            repeatedMasks = cached.array
        } else {
            repeatedMasks = broadcast(
                maskEmbedding, to: [batch, cols, config.hiddenSize])
            if batch == 1 {
                // Retained row-contiguous (the same values) where the
                // one-launch concatenation reads it: a broadcast view
                // would be copied into place by every launch.
                if DFlash2Concat.enabled {
                    repeatedMasks = contiguous(repeatedMasks)
                }
                eval(repeatedMasks)
                cachedMaskEmbeddingBlock = (cols: cols, array: repeatedMasks)
            }
        }
        return DFlash2Concat.concatenate([anchorEmbedding, repeatedMasks], axis: 1)
    }

    /// `hiddenNorm(fc(rows))`. `BONSAI_DRAFT_CONTEXT_PAD16=1` (validation aid)
    /// pads to 16 rows, as the tensor kernel does, so MLX's matmul (kernel by
    /// row count) gets the tensor route's row independence.
    func contextProjection(_ targetHidden: MLXArray) -> MLXArray {
        let rows = targetHidden.asType(dtype)
        if DFlash2ContextPadding.enabled, rows.ndim == 3, rows.dim(1) < 16 {
            let padded = concatenated(
                [rows, MLXArray.zeros([rows.dim(0), 16 - rows.dim(1), rows.dim(2)], dtype: rows.dtype)],
                axis: 1)
            return hiddenNorm(DFlash2TensorMatmul.linear(fc, padded)[0..., ..<rows.dim(1), 0...])
        }
        return hiddenNorm(DFlash2TensorMatmul.linear(fc, rows))
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
    ///   - leadingLayers: see `hiddenStates`; 0 submits nothing.
    /// - Returns: the draft tokens, `[B, blockSize - 1]`.
    public func propose(
        anchor: [Int],
        targetHidden: MLXArray?,
        cache: [KVCache],
        blockSize: Int,
        submittingLeadingLayers leadingLayers: Int = 0
    ) throws -> MLXArray {
        guard blockSize >= 2 else { throw DFlash2Error.invalidBlockSize(blockSize) }
        let masks = Array(repeating: Int32(config.maskTokenId), count: blockSize - 1)
        let rows = anchor.flatMap { [Int32($0)] + masks }
        let block = MLXArray(rows, [anchor.count, blockSize])

        let hidden = try hiddenStates(
            block, targetHidden: targetHidden, cache: cache, logitsStart: 1,
            submittingLeadingLayers: leadingLayers)
        return candidateSelector.selectGreedy(
            hidden: hidden,
            logits: try logits(hidden),
            anchor: MLXArray(anchor.map { Int32($0) }))
    }

    /// Enter `targetHidden` (`[B, contextLength, targetHiddenSize]`, committed
    /// positions' fused target hidden state) into every layer's cache without a
    /// block: the context keys and values a block forward over the same rows
    /// would write, computed from the context alone. The next `propose` then
    /// passes no context rows. Returns false, having written nothing, when a
    /// layer's cache cannot take the rows in place; the caller keeps the rows
    /// and hands them to the next block as before.
    public func absorbContext(targetHidden: MLXArray, cache: [KVCache]) throws -> Bool {
        guard target != nil else { throw DFlash2Error.notBound }
        guard cache.count == layers.count else {
            throw DFlash2Error.invalidCacheCount(expected: layers.count, actual: cache.count)
        }
        guard targetHidden.dim(-1) == config.targetHiddenSize else {
            throw DFlash2Error.targetHiddenSizeMismatch(
                expected: config.targetHiddenSize, actual: targetHidden.dim(-1))
        }
        let rows = targetHidden.dim(1)
        guard rows >= 1,
            cache.allSatisfy({ ($0 as? DFlash2BlockKVCache)?.canAbsorb(contextRows: rows) ?? false }),
            config.layerTypes.allSatisfy({ $0 == .slidingAttention }),
            let slidingWindow = config.slidingWindow,
            DFlash2SlidingMask.contextSkip(contextLength: rows, slidingWindow: slidingWindow) == 0
        else { return false }
        let context = hiddenNorm(DFlash2TensorMatmul.linear(fc, targetHidden.asType(dtype)))
        for (index, layer) in layers.enumerated() {
            let absorbed = layer.absorbContext(context, rope: rope, cache: cache[index])
            precondition(absorbed, "DFlash 2: a checked cache refused its context rows")
        }
        return true
    }

    // MARK: The next block before the readback

    /// `(held rows, offset)` of every layer when a speculative block fits
    /// (today's block in place and unmasked for every count, room for 2x rows).
    public func speculativeGeometry(cache: [KVCache], blockSize: Int) -> (rows: Int, offset: Int)? {
        guard target != nil, cache.count == layers.count, dflash2NoMaskEnabled,
            !config.isCausal, let window = config.slidingWindow,
            config.layerTypes.allSatisfy({ $0 == .slidingAttention }),
            layers.allSatisfy({ $0.selfAttn.speculativeCapable })
        else { return nil }
        var geometry: (rows: Int, offset: Int)?
        for c in cache {
            guard let block = c as? DFlash2BlockKVCache, let rows = block.inPlaceRows,
                rows + blockSize <= block.contextRowLimit, rows + 2 * blockSize <= window,
                rows + 2 * blockSize <= block.inPlaceCapacity,
                geometry == nil || (geometry!.rows == rows && geometry!.offset == block.offset)
            else { return nil }
            geometry = (rows, block.offset)
        }
        return geometry
    }

    /// The next block from a device confirmed count and anchor over the whole
    /// verify context (its leading `contextRows` projected). No cursor moves;
    /// with `submitLead` the leading layers' writes are installed (donating
    /// their buffers) and those layers submitted; otherwise the whole block is
    /// submitted with the round's other drafter work after adoption.
    public func proposeSpeculative(
        anchor: MLXArray, confirmed: MLXArray, verifyContext: MLXArray, contextRows: Int,
        cache: [KVCache], blockSize: Int, leadingLayers: Int, submitLead: Bool
    ) throws -> DFlash2SpeculativeBlock? {
        guard let geometry = speculativeGeometry(cache: cache, blockSize: blockSize),
            anchor.size == 1, confirmed.size == 1,
            verifyContext.shape == [1, blockSize, config.targetHiddenSize],
            (1 ... blockSize).contains(contextRows)
        else { return nil }
        let caches = cache.map { $0 as! DFlash2BlockKVCache }
        let n = 2 * blockSize
        var h = try blockEmbedding(anchorIDs: anchor.reshaped([1, 1]), maskColumns: blockSize - 1)
            .asType(dtype)
        if config.dflash.inputEmbeddingScale != 1 {
            h = h * config.dflash.inputEmbeddingScale
        }
        let context = contextProjection(verifyContext[0..., ..<contextRows, 0...])
        let base = concatenated(
            [context, MLXArray.zeros([1, n - contextRows, config.hiddenSize], dtype: context.dtype)],
            axis: 1)
        let c = confirmed.reshaped([]).asType(.int32)
        let queryOffset = MLXArray(Int32(geometry.offset)) + c
        let keys = geometry.rows + n
        let keyMask = (MLXArray(Int32(0) ..< Int32(keys)) .< (MLXArray(Int32(geometry.rows + blockSize)) + c))
            .reshaped([1, keys])
        let leadAt = submitLead ? min(max(leadingLayers, 0), layers.count) : 0
        var writes: [(keys: MLXArray, values: MLXArray)] = []
        var lead: MLXArray?
        for (index, layer) in layers.enumerated() {
            guard
                let out = layer.speculative(
                    h, base: base, confirmed: c, queryOffset: queryOffset, rope: rope,
                    cache: caches[index], keyMask: keyMask)
            else { preconditionFailure("DFlash 2: a checked layer refused its speculative block") }
            h = out.hidden
            writes.append((out.keys, out.values))
            if index + 1 == leadAt { lead = h }
        }
        let hidden = norm(h[0..., 1..., 0...])
        let tokens = candidateSelector.selectGreedy(
            hidden: hidden, logits: try logits(hidden), anchor: anchor.reshaped([1]))
        if let lead {
            for i in 0 ..< leadAt { caches[i].installSpeculative(keys: writes[i].keys, values: writes[i].values) }
            asyncEval([lead])
        }
        return DFlash2SpeculativeBlock(
            tokens: tokens, writes: writes, installedLayers: leadAt, contextRows: contextRows)
    }

    /// Install the remaining writes; cursors advance as today's block's do.
    public func adoptSpeculative(_ block: DFlash2SpeculativeBlock, confirmed: Int, cache: [KVCache]) {
        for (index, c) in cache.enumerated() {
            let blockCache = c as! DFlash2BlockKVCache
            if index >= block.installedLayers {
                blockCache.installSpeculative(keys: block.writes[index].keys, values: block.writes[index].values)
            }
            blockCache.commitSpeculativeContext(confirmed)
        }
    }

    /// Per confirmed count `c`, the largest `m >= c` whose projection of `m`
    /// rows gives the projection of `c` rows bit for bit (MLX's matmul picks
    /// gemv, wide gemv or split-K by row count; the tensor kernel does not).
    public func contextRowClasses(rows: Int) -> [Int] {
        let probe = MLXRandom.normal([1, rows, config.targetHiddenSize], key: MLXRandom.key(0x5eed))
            .asType(dtype)
        let outputs = (1 ... rows).map { contextProjection(probe[0..., ..<$0, 0...]) }
        eval(outputs)
        let bits = outputs.map { $0.asType(.float32).asArray(Float.self).map(\.bitPattern) }
        let width = config.hiddenSize
        return [0] + (1 ... rows).map { c in
            stride(from: rows, to: c, by: -1).first { Array(bits[$0 - 1][..<(c * width)]) == bits[c - 1] } ?? c
        }
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

/// A block proposed before its round's readback.
public final class DFlash2SpeculativeBlock {
    public let tokens: MLXArray
    let writes: [(keys: MLXArray, values: MLXArray)]
    let installedLayers: Int
    public let contextRows: Int  // its row class

    init(tokens: MLXArray, writes: [(keys: MLXArray, values: MLXArray)], installedLayers: Int, contextRows: Int) {
        (self.tokens, self.writes, self.installedLayers, self.contextRows) =
            (tokens, writes, installedLayers, contextRows)
    }
}

enum DFlash2ContextPadding {
    static let enabled = ProcessInfo.processInfo.environment["BONSAI_DRAFT_CONTEXT_PAD16"] == "1"
}

/// Layer counts after which the drafter trunk `asyncEval`s its hidden state.
/// Default: after the first layer, so the GPU starts the block (it has been
/// idle since the verify readback) while the host builds the other layers and
/// the head; measured locally ~0.2-0.4% decode. `MLXFAST_DRAFT_SLICE_LAYERS`
/// overrides it with a `,`/`;` list of counts (a count equal to the layer
/// count submits the trunk before the head); `0`/`off` turns it off.
enum DFlash2DraftSubmission {
    static let layers: [Int] = {
        guard let raw = ProcessInfo.processInfo.environment["MLXFAST_DRAFT_SLICE_LAYERS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
            !raw.isEmpty
        else { return [] }  // off by default here: submission slices lengthened the window on this lineage
        if ["0", "off", "false", "no"].contains(raw) { return [] }
        return raw.split(whereSeparator: { $0 == "," || $0 == ";" }).compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }.filter { $0 > 0 }
    }()
}

// MARK: - One-launch concatenation

/// A concatenation of arrays of one dtype as ONE launch that copies every
/// element to its place, where MLX's `Concatenate` runs one copy launch per
/// input: the target's tapped hidden states (five launches per verify window
/// of the 27B) and each drafter layer's `[context; block]` rows (two per
/// layer). A copy: the output holds the inputs' bits in concatenation order.
/// Self-tested once per input count and dtype (at bind for the two in use),
/// bit for bit against `concatenated` on odd shapes along both a middle and
/// the last axis; a mismatch or an MLX error keeps `concatenated`.
/// `MLXFAST_ONE_LAUNCH_CONCAT=0` keeps `concatenated`.
enum DFlash2Concat {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_ONE_LAUNCH_CONCAT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    private static let kernelLock = NSLock()
    nonisolated(unsafe) private static var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]

    /// Input `i` is `[outer, inner_i]` row-major, the output `[outer, total]`;
    /// `dims` = `[outer, total, inner_0, ...]` at run time, so one pipeline per
    /// input count and dtype serves every shape (a timed round never builds one).
    private static func kernel(_ n: Int) -> MLXFast.MLXFastKernel {
        kernelLock.withLock {
            if let kernel = kernels[n] { return kernel }
            var source = """
                const uint idx = thread_position_in_grid.x;
                const uint total = uint(dims[1]);
                if (idx >= uint(dims[0]) * total) { return; }
                const uint o = idx / total;
                uint j = idx - o * total;

                """
            for i in 0 ..< n {
                source += "{ const uint w = uint(dims[\(i + 2)]);\n"
                source += "if (j < w) { out[idx] = x\(i)[o * w + j]; return; }\n"
                source += "j -= w; }\n"
            }
            let kernel = MLXFast.metalKernel(
                name: "dflash2_concat\(n)", inputNames: (0 ..< n).map { "x\($0)" } + ["dims"],
                outputNames: ["out"], source: source, ensureRowContiguous: true)
            kernels[n] = kernel
            return kernel
        }
    }

    private static func launch(_ parts: [MLXArray], axis: Int) -> MLXArray {
        var shape = parts[0].shape
        let outer = shape[..<axis].reduce(1, *)
        let inners = parts.map { $0.shape[axis...].reduce(1, *) }
        let total = inners.reduce(0, +)
        shape[axis] = parts.reduce(0) { $0 + $1.dim(axis) }
        let dims = MLXArray(([outer, total] + inners).map { Int32($0) })
        let threads = outer * total
        return kernel(parts.count)(
            parts + [dims], grid: (threads, 1, 1),
            threadGroup: (min(256, threads), 1, 1), outputShapes: [shape],
            outputDTypes: [parts[0].dtype])[0]
    }

    /// `concatenated(parts, axis: axis)`, in one launch where it applies.
    static func concatenate(_ parts: [MLXArray], axis: Int) -> MLXArray {
        guard enabled, parts.count >= 2, parts.count <= 8 else {
            return concatenated(parts, axis: axis)
        }
        let first = parts[0]
        let ax = axis < 0 ? axis + first.ndim : axis
        guard ax >= 0, ax < first.ndim,
            [DType.float16, .bfloat16, .float32].contains(first.dtype),
            parts.allSatisfy({ part in
                part.dtype == first.dtype && part.ndim == first.ndim
                    && (0 ..< first.ndim).allSatisfy { $0 == ax || part.dim($0) == first.dim($0) }
            }),
            parts.reduce(0, { $0 + $1.size }) < Int(Int32.max),
            verified(parts.count, first.dtype)
        else { return concatenated(parts, axis: axis) }
        return launch(parts, axis: ax)
    }

    private static let zerosLock = NSLock()
    nonisolated(unsafe) private static var zeros: [[Int]: MLXArray] = [:]

    /// `concatenated([x, zeros([rows - x.dim(0), k])], axis: 0)` for a 2-D
    /// `x`: the zero rows are a view of one retained, never-written zeros
    /// array per width and dtype (materialized by its first use), so the pad
    /// is one launch instead of a fill and two copies.
    static func padRows(_ x: MLXArray, to rows: Int) -> MLXArray {
        let (have, k) = (x.dim(0), x.dim(1))
        guard enabled, x.ndim == 2, have < rows else {
            return concatenated(
                [x, MLXArray.zeros([rows - have, k], dtype: x.dtype)], axis: 0)
        }
        let block: MLXArray = zerosLock.withLock {
            let key = [rows - 1, k, x.dtype == .bfloat16 ? 1 : x.dtype == .float16 ? 2 : 3]
            if let hit = zeros[key] { return hit }
            let made = MLXArray.zeros([rows - 1, k], dtype: x.dtype)
            zeros[key] = made
            return made
        }
        return concatenate([x, block[0 ..< (rows - have)]], axis: 0)
    }

    /// Runs the self-test for `inputs` inputs of `dtype` now (load time).
    static func prepare(inputs: Int, dtype: DType) {
        guard enabled else { return }
        _ = verified(inputs, dtype)
    }

    private static func verified(_ n: Int, _ dtype: DType) -> Bool {
        lock.withLock {
            let key = "\(n) \(dtype)"
            if let verdict = verdicts[key] { return verdict }
            var same = true
            var compared = 0
            do {
                try withError { error in
                    let bits: DType = dtype == .float32 ? .uint32 : .uint16
                    for (seed, axis) in [(71, 1), (72, 2)] {
                        let parts = (0 ..< n).map { i -> MLXArray in
                            var shape = [2, 3, 40]
                            shape[axis] = [1, 5, 24, 16, 3, 7, 40, 2][i]
                            return (MLXRandom.normal(shape, key: MLXRandom.key(UInt64(seed * 16 + i)))
                                * 1000).asType(dtype)
                        }
                        let fused = launch(parts, axis: axis)
                        let reference = concatenated(parts, axis: axis)
                        same = same && fused.shape == reference.shape
                            && all(fused.view(dtype: bits) .== reference.view(dtype: bits))
                                .item(Bool.self)
                        compared += reference.size
                    }
                    try error.check()
                }
            } catch {
                same = false
            }
            verdicts[key] = same
            FileHandle.standardError.write(
                ("dflash2 one-launch concat (\(key)): "
                    + (same
                        ? "self-test passed: \(compared) values compared bitwise, 0 mismatches; one launch\n"
                        : "self-test failed; concatenated kept\n")).data(using: .utf8)!)
            return same
        }
    }
}

// MARK: - Head RMSNorm over a strided view

/// The drafter's q and k head norms read the stacked q|k|v product through
/// its strides. `RMSNorm` first copies a view that is not row-contiguous (a
/// column slice of the stacked product), one launch per norm; this is MLX's
/// `rms_single_row` body (one simdgroup per row of `D` <= 128, four reads per
/// lane, the same FP32 sum, `simd_sum`s, `precise::rsqrt` and the two
/// roundings of the output) reading the view in place and writing the same
/// contiguous output. Self-tested once per dtype and head size (at bind) on
/// column slices of a stacked product with rows of very different
/// magnitudes (and a zero row), bit for bit against `RMSNorm`; a mismatch or
/// an MLX error keeps `RMSNorm`. `MLXFAST_DFLASH_STRIDED_NORM=0` keeps it.
enum DFlash2StridedRMSNorm {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_STRIDED_NORM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "dflash2_strided_rms_single_row",
        inputNames: ["x", "w", "eps"],
        outputNames: ["out"],
        source: """
            constexpr int N_READS = 4;
            constexpr int SIMD_SIZE = 32;
            const uint gid = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const uint axis_size = uint(D);
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];

            const uint H = uint(x_shape[2]);
            const uint T = uint(x_shape[1]);
            const uint h = gid % H;
            const uint t = (gid / H) % T;
            const uint b = gid / (H * T);
            const device auto* xr = x + int64_t(b) * x_strides[0] + int64_t(t) * x_strides[1]
                + int64_t(h) * x_strides[2] + lid * N_READS;
            const device auto* wr = w + lid * N_READS;

            float acc = 0;
            float thread_x[N_READS];
            if (lid * N_READS + N_READS <= axis_size) {
              for (int i = 0; i < N_READS; i++) {
                thread_x[i] = xr[i];
                acc += thread_x[i] * thread_x[i];
              }
            } else {
              for (int i = 0; i < N_READS; i++) {
                thread_x[i] = (lid * N_READS + i < axis_size) ? (float)xr[i] : 0;
                acc += thread_x[i] * thread_x[i];
              }
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) {
              local_sums[simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) {
              local_sums[simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
              acc = simd_sum(local_sums[simd_lane_id]);
              if (simd_lane_id == 0) {
                local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            device auto* o = out + size_t(gid) * axis_size + lid * N_READS;
            if (lid * N_READS + N_READS <= axis_size) {
              for (int i = 0; i < N_READS; i++) {
                o[i] = wr[i] * static_cast<OutT>(thread_x[i] * local_inv_mean[0]);
              }
            } else {
              for (int i = 0; i < N_READS; i++) {
                if ((lid * N_READS + i) < axis_size) {
                  o[i] = wr[i] * static_cast<OutT>(thread_x[i] * local_inv_mean[0]);
                }
              }
            }
            """,
        ensureRowContiguous: false)

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]
    nonisolated(unsafe) private static var epsArrays: [Float: MLXArray] = [:]

    private static func epsArray(_ eps: Float) -> MLXArray {
        if let hit = epsArrays[eps] { return hit }
        let made = MLXArray([eps])
        epsArrays[eps] = made
        return made
    }

    private static func launch(_ x: MLXArray, weight: MLXArray, eps: MLXArray) -> MLXArray {
        let (b, t, h, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return kernel(
            [x, weight, eps],
            template: [("OutT", x.dtype), ("D", d)],
            grid: (32 * b * t * h, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
    }

    /// `norm(x)` for a `[B, T, H, D]` view whose rows of `D` are contiguous.
    static func apply(_ norm: RMSNorm, _ x: MLXArray) -> MLXArray {
        let weight = norm.weight
        guard enabled, x.ndim == 4, x.dim(3) <= 128, x.dim(3) % 4 == 0,
            weight.ndim == 1, weight.dim(0) == x.dim(3), weight.dtype == x.dtype,
            [DType.bfloat16, .float16].contains(x.dtype),
            x.size < Int(Int32.max),
            verified(x.dtype, headDim: x.dim(3), eps: norm.eps)
        else { return norm(x) }
        let eps: MLXArray = lock.withLock { epsArray(norm.eps) }
        return launch(x, weight: weight, eps: eps)
    }

    /// Runs the self-test for this dtype and head size now (load time).
    static func prepare(dtype: DType, headDim: Int, eps: Float) {
        guard enabled, [DType.bfloat16, .float16].contains(dtype), headDim <= 128,
            headDim % 4 == 0
        else { return }
        _ = verified(dtype, headDim: headDim, eps: eps)
    }

    private static func verified(_ dtype: DType, headDim d: Int, eps: Float) -> Bool {
        lock.withLock {
            let key = "\(dtype) \(d) \(eps)"
            if let verdict = verdicts[key] { return verdict }
            var same = true
            var compared = 0
            do {
                try withError { error in
                    let epsA = epsArray(eps)
                    for seed in [81, 82] {
                        let rows = 17
                        // Row scales from 1e-3 to 3e3 and one zero row.
                        let scale = exp(MLXRandom.uniform(
                            Float(-7) ..< Float(8), [rows, 1], key: MLXRandom.key(UInt64(seed))))
                            * (MLXArray(0 ..< rows) .!= MLXArray(Int32(5))).asType(.float32)
                                .reshaped(rows, 1)
                        let stacked = (MLXRandom.normal(
                            [rows, 48 * d], key: MLXRandom.key(UInt64(seed + 100))) * scale)
                            .asType(dtype)
                        let weight = (1 + 0.3 * MLXRandom.normal(
                            [d], key: MLXRandom.key(UInt64(seed + 200)))).asType(dtype)
                        let views = [
                            stacked[1..., ..<(32 * d)].reshaped(1, rows - 1, 32, d),
                            stacked[0..., (32 * d) ..< (40 * d)].reshaped(1, rows, 8, d),
                        ]
                        for view in views {
                            let fused = launch(view, weight: weight, eps: epsA)
                            let reference = MLXFast.rmsNorm(view, weight: weight, eps: eps)
                            same = same && fused.shape == reference.shape
                                && all(fused.view(dtype: .uint16) .== reference.view(dtype: .uint16))
                                    .item(Bool.self)
                            compared += reference.size
                        }
                    }
                    try error.check()
                }
            } catch {
                same = false
            }
            verdicts[key] = same
            FileHandle.standardError.write(
                ("dflash2 strided head norm (\(key)): "
                    + (same
                        ? "self-test passed: \(compared) values compared bitwise, 0 mismatches; strided\n"
                        : "self-test failed; RMSNorm kept\n")).data(using: .utf8)!)
            return same
        }
    }
}

// MARK: - Head norms and ropes in one launch

/// The q and k head norms and their ropes (four launches per layer) as ONE
/// launch over the stacked q|k|v product. Each simdgroup takes one head row
/// through `DFlash2StridedRMSNorm`'s body verbatim (reads, FP32 sum,
/// `simd_sum`s, `precise::rsqrt`, both roundings), then MLX's `rope` body on
/// the rounded row (`exp2(-d * log2(base))`, `fast::cos`/`fast::sin`, the
/// same FP32 products, one rounding), the other half of each pair taken from
/// lane `^ 16`; the outputs are `rope`'s contiguous `[B, H, T, D]`.
/// Self-tested at bind against `rmsNorm` + the drafter's own rope, bit for
/// bit, on rows from 1e-3 to 3e3 with zero rows at three offsets; a mismatch
/// or an MLX error keeps the four launches. `MLXFAST_DFLASH_QK_PREWORK=0`
/// keeps them.
enum DFlash2QKPrework {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_QK_PREWORK"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "dflash2_qk_prework",
        inputNames: ["y", "qw", "kw", "p", "pos"],
        outputNames: ["q", "k"],
        source: """
            constexpr int N_READS = 4;
            constexpr int SIMD_SIZE = 32;
            const uint gid = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const uint axis_size = uint(D);
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];

            const uint n = uint(y_shape[1]);
            const uint BL = uint(pos[2]);
            const uint qRows = uint(y_shape[0]) * BL * HQ;
            const bool isQ = gid < qRows;
            const uint r = isQ ? gid : gid - qRows;
            const uint H = isQ ? HQ : HK;
            const uint NT = isQ ? BL : n;
            const uint h = r % H;
            const uint t = (r / H) % NT;
            const uint b = r / (H * NT);
            const device auto* xr = y + (size_t(b) * n + (isQ ? n - BL + t : t)) * uint(y_shape[2])
                + (isQ ? 0 : HQ * D) + h * D + lid * N_READS;
            const device auto* wr = (isQ ? qw : kw) + lid * N_READS;

            float acc = 0;
            float thread_x[N_READS];
            for (int i = 0; i < N_READS; i++) {
              thread_x[i] = xr[i];
              acc += thread_x[i] * thread_x[i];
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) {
              local_sums[simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) {
              local_sums[simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
              acc = simd_sum(local_sums[simd_lane_id]);
              if (simd_lane_id == 0) {
                local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + p[0]);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const bool lo = lid < 16;
            float Lp = p[2] * static_cast<float>(t + uint(pos[isQ ? 0 : 1]));
            device T* o = (isQ ? q : k) + ((size_t(b) * H + h) * NT + t) * D + lid * N_READS;
            for (int i = 0; i < N_READS; i++) {
              const T v = wr[i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
              const float xv = static_cast<float>(v);
              const float other = simd_shuffle_xor(xv, ushort(16));
              float d = static_cast<float>((lid & 15) * N_READS + i) / static_cast<float>(D / 2);
              float inv_freq = metal::exp2(-d * p[1]);
              float theta = Lp * inv_freq;
              float costheta = metal::fast::cos(theta);
              float sintheta = metal::fast::sin(theta);
              float x1 = lo ? xv : other;
              float x2 = lo ? other : xv;
              float rx1 = x1 * costheta - x2 * sintheta;
              float rx2 = x1 * sintheta + x2 * costheta;
              o[i] = static_cast<T>(lo ? rx1 : rx2);
            }
            """,
        ensureRowContiguous: true)

    private static let lock = NSLock()
    /// `[eps, log2(base), scale]` per self-tested geometry.
    nonisolated(unsafe) private static var ready: [String: MLXArray] = [:]

    private static func launch(
        _ y: MLXArray, _ l: Int, _ hq: Int, _ hk: Int, _ qw: MLXArray, _ kw: MLXArray,
        _ p: MLXArray, _ offsets: [Int]
    ) -> (MLXArray, MLXArray) {
        let (b, n, d) = (y.dim(0), y.dim(1), qw.dim(0))
        let out = kernel(
            [y, qw, kw, p, MLXArray(offsets.map { Int32($0) } + [Int32(l)])],
            template: [("T", y.dtype), ("D", d), ("HQ", hq), ("HK", hk)],
            grid: (32 * b * (l * hq + n * hk), 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[b, hq, l, d], [b, hk, n, d]], outputDTypes: [y.dtype, y.dtype])
        return (out[0], out[1])
    }

    /// `(rope(qNorm(q)), rope(kNorm(k)))` from the stacked product `y`
    /// (`[B, n, (heads + 2 kvHeads) * D]`, the q rows its last `l`), rotated
    /// at `offsets` = [q, k], or nil when the launch does not apply.
    static func apply(
        _ y: MLXArray, blockRows l: Int, heads hq: Int, qNorm: RMSNorm, kNorm: RMSNorm,
        offsets: [Int]
    ) -> (MLXArray, MLXArray)? {
        let d = qNorm.weight.dim(0)
        let hk = (y.dim(-1) / d - hq) / 2
        guard enabled, y.ndim == 3, d == 128, y.dim(2) == (hq + 2 * hk) * d, hk > 0,
            l >= 1, l <= y.dim(1), qNorm.eps == kNorm.eps, kNorm.weight.shape == [d],
            qNorm.weight.dtype == y.dtype, kNorm.weight.dtype == y.dtype,
            y.size < Int(Int32.max), offsets.allSatisfy({ $0 >= 0 }),
            let p = lock.withLock({ ready["\(y.dtype) \(hq) \(hk) \(qNorm.eps)"] })
        else { return nil }
        return launch(y, l, hq, hk, qNorm.weight, kNorm.weight, p, offsets)
    }

    /// The self-test for the drafter's geometry and rope, at bind.
    static func prepare(
        rope: RoPELayer, base: Float, dtype: DType, heads hq: Int, kvHeads hk: Int,
        headDim d: Int, eps: Float
    ) {
        guard enabled, d == 128, hk > 0, [DType.bfloat16, .float16].contains(dtype) else { return }
        lock.withLock {
            let key = "\(dtype) \(hq) \(hk) \(eps)"
            guard ready[key] == nil else { return }
            let p = MLXArray([eps, log2(base), 1])
            var same = true
            var compared = 0
            do {
                try withError { error in
                    for (seed, n, off) in [(71, 24, 37), (72, 16, 4077), (73, 40, 65514)] {
                        let (l, w) = (16, hq + 2 * hk)
                        let index = MLXArray(0 ..< n * w).reshaped(n, w, 1)
                        // Head rows from 1e-3 to 3e3, a zero q row and a zero k row.
                        let scale = exp(MLXRandom.uniform(
                            Float(-7) ..< Float(8), [n, w, 1], key: MLXRandom.key(UInt64(seed))))
                            * ((index .!= MLXArray(Int32((n - 1) * w + 3)))
                                .&& (index .!= MLXArray(Int32(hq + 1)))).asType(.float32)
                        let y = (MLXRandom.normal(
                            [n, w, d], key: MLXRandom.key(UInt64(seed + 100))) * scale)
                            .asType(dtype).reshaped(1, n, w * d)
                        let qw = (1 + 0.3 * MLXRandom.normal(
                            [d], key: MLXRandom.key(UInt64(seed + 200)))).asType(dtype)
                        let kw = (1 + 0.3 * MLXRandom.normal(
                            [d], key: MLXRandom.key(UInt64(seed + 300)))).asType(dtype)
                        let (fq, fk) = launch(y, l, hq, hk, qw, kw, p, [off + n - l, off])
                        let rq = rope(
                            MLXFast.rmsNorm(
                                y[0..., (n - l)..., ..<(hq * d)].reshaped(1, l, hq, d),
                                weight: qw, eps: eps
                            ).transposed(0, 2, 1, 3), offset: off + n - l)
                        let rk = rope(
                            MLXFast.rmsNorm(
                                y[0..., 0..., (hq * d) ..< ((hq + hk) * d)].reshaped(1, n, hk, d),
                                weight: kw, eps: eps
                            ).transposed(0, 2, 1, 3), offset: off)
                        for (f, r) in [(fq, rq), (fk, rk)] {
                            same = same && f.shape == r.shape
                                && all(f.view(dtype: .uint16) .== r.view(dtype: .uint16))
                                    .item(Bool.self)
                            compared += r.size
                        }
                    }
                    try error.check()
                }
            } catch {
                same = false
            }
            if same { ready[key] = p }
            FileHandle.standardError.write(
                ("dflash2 q/k norm+rope prework (\(key)): "
                    + (same
                        ? "self-test passed: \(compared) values compared bitwise, 0 mismatches; one launch\n"
                        : "self-test failed; norms and ropes kept\n")).data(using: .utf8)!)
        }
    }
}

// MARK: - SwiGLU in one launch

/// `silu(g) * u` for the drafter's MLP as ONE launch over the stacked
/// gate|up product's two column halves, where the compiled SiLU and the
/// product ran as two: the compiled kernel's own ops in its own order and
/// dtype (MLX's `Sigmoid` body on the BF16 gate, `x * sigmoid(x)` rounded to
/// BF16, then the product with `u` rounded to BF16). Self-tested once per
/// dtype and width (at bind), bit for bit against `silu(g) * u` on halves of
/// a stacked product with rows from 1e-3 to 3e3 (both saturations) and a
/// zero row; a mismatch or an MLX error keeps the two launches.
/// `MLXFAST_DFLASH_SWIGLU=0` keeps them.
enum DFlash2SwiGLU {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SWIGLU"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "dflash2_swiglu_strided",
        inputNames: ["g", "u"],
        outputNames: ["out"],
        source: """
            const uint idx = thread_position_in_grid.x;
            const uint N = uint(g_shape[2]);
            if (idx >= uint(g_shape[1]) * N) { return; }
            const uint r = idx / N;
            const uint c = idx - r * N;
            const OutT gv = g[int64_t(r) * g_strides[1] + int64_t(c) * g_strides[2]];
            const OutT uv = u[int64_t(r) * u_strides[1] + int64_t(c) * u_strides[2]];
            const OutT sg = SigmoidMLX()(gv);
            const OutT act = MultiplyMLX()(gv, sg);
            out[idx] = MultiplyMLX()(act, uv);
            """,
        header: """
            struct SigmoidMLX {
              template <typename T>
              T operator()(T x) thread {
                auto y = 1 / (1 + metal::exp(metal::abs(x)));
                return (x < 0) ? y : 1 - y;
              }
            };
            struct MultiplyMLX {
              template <typename T>
              T operator()(T x, T y) { return x * y; }
            };
            """,
        ensureRowContiguous: false)

    private static func launch(_ g: MLXArray, _ u: MLXArray) -> MLXArray {
        let (r, n) = (g.dim(1), g.dim(2))
        return kernel(
            [g, u], template: [("OutT", g.dtype)],
            grid: (r * n, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [g.shape], outputDTypes: [g.dtype])[0]
    }

    static func apply(_ g: MLXArray, _ u: MLXArray) -> MLXArray {
        guard enabled, g.ndim == 3, g.dim(0) == 1, g.shape == u.shape, g.dtype == u.dtype,
            [DType.bfloat16, .float16].contains(g.dtype), g.size < Int(Int32.max),
            verified(g.dtype, width: g.dim(2))
        else { return silu(g) * u }
        return launch(g, u)
    }

    static func prepare(dtype: DType, width: Int) {
        guard enabled, [DType.bfloat16, .float16].contains(dtype) else { return }
        _ = verified(dtype, width: width)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]

    private static func verified(_ dtype: DType, width: Int) -> Bool {
        lock.withLock {
            let key = "\(dtype) \(width)"
            if let verdict = verdicts[key] { return verdict }
            var same = true
            var compared = 0
            do {
                try withError { error in
                    for seed in [91, 92] {
                        let rows = 16
                        let scale = exp(MLXRandom.uniform(
                            Float(-7) ..< Float(8), [rows, 1], key: MLXRandom.key(UInt64(seed))))
                            * (MLXArray(0 ..< rows) .!= MLXArray(Int32(3))).asType(.float32)
                                .reshaped(rows, 1)
                        let stacked = (MLXRandom.normal(
                            [rows, 2 * width], key: MLXRandom.key(UInt64(seed + 100))) * scale)
                            .asType(dtype).reshaped(1, rows, 2 * width)
                        let g = stacked[.ellipsis, ..<width]
                        let u = stacked[.ellipsis, width...]
                        let fused = launch(g, u)
                        let reference = silu(g) * u
                        same = same && fused.shape == reference.shape
                            && all(fused.view(dtype: .uint16) .== reference.view(dtype: .uint16))
                                .item(Bool.self)
                        compared += reference.size
                    }
                    try error.check()
                }
            } catch {
                same = false
            }
            verdicts[key] = same
            FileHandle.standardError.write(
                ("dflash2 one-launch SwiGLU (\(key)): "
                    + (same
                        ? "self-test passed: \(compared) values compared bitwise, 0 mismatches; one launch\n"
                        : "self-test failed; silu(g) * u kept\n")).data(using: .utf8)!)
            return same
        }
    }
}
