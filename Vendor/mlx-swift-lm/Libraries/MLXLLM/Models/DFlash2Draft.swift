// Copyright © 2026 Apple Inc.

// DFlash 2 block drafter.
// The wide FP32 selector reuses singleton-batch views from the narrow path;
// selector arithmetic, output ids, and the pinned parameter format are unchanged.
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

/// The speculative block's attention (16 queries over the held keys) through
/// MLX's NAX attention, whose one-query-block case (at most 16 query rows, no
/// causal mask, no sinks) now pipelines the key blocks over the 64-row
/// tile's four simdgroups (`steel_attention_nax`): each round, simdgroup s
/// scores key block 4j + s (the loop's loads, mma chain, scale and masks),
/// every simdgroup runs the running-max chain over the round's blocks,
/// simdgroup s forms its block's exp2(S - m) and rescale factor, and every
/// simdgroup then folds the four blocks in key order into its own pair of
/// O's head-dim fragments with the loop's own statements (l *= factor, the
/// probabilities' row_reduce, O *= factor, O += P V). Before, simdgroup 0
/// alone ran the whole chain while the other three repeated it over empty
/// rows; the scores of four blocks are now computed at once (about 2.5x
/// faster at 1,000 keys). Every output element is the same operations on
/// the same operands. `verify` checks it at bind against the untouched path
/// (the same queries plus one row, 17 rows, which the pipeline does not
/// take; rows are independent) on the drafter's geometry, every output bit;
/// on a mismatch the block's queries go in with one zero row appended (the
/// untouched path) and the extra output row is dropped.
enum DFlash2AttentionPipeline {
    nonisolated(unsafe) static var padQueries = false
    nonisolated(unsafe) private static var verified = false

    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?
    ) -> MLXArray {
        let rows = queries.dim(2)
        if padQueries, rows <= 16, queries.ndim == 4 {
            let padded = concatenated(
                [queries, MLXArray.zeros([queries.dim(0), queries.dim(1), 17 - rows, queries.dim(3)],
                    dtype: queries.dtype)], axis: 2)
            return MLXFast.scaledDotProductAttention(
                queries: padded, keys: keys, values: values, scale: scale, mask: mask)[
                    0..., 0..., ..<rows, 0...]
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }

    static func verify(dtype: DType, heads: Int, kvHeads: Int, headDim: Int) {
        guard !verified else { return }
        verified = true
        var same = true
        var compared = 0
        do {
            try withError { error in
                for (seed, keysLength) in [(0, 48), (1, 545), (2, 1000), (3, 2080)] {
                    func key(_ salt: Int) -> MLXArray { MLXRandom.key(UInt64(0x5d9a + seed * 8 + salt)) }
                    let q = (MLXRandom.normal([1, heads, 17, headDim], key: key(0)) * 2).asType(dtype)
                    let k = (MLXRandom.normal([1, kvHeads, keysLength, headDim], key: key(1)) * 2).asType(dtype)
                    let v = MLXRandom.normal([1, kvHeads, keysLength, headDim], key: key(2)).asType(dtype)
                    let mask = (MLXArray(Int32(0) ..< Int32(keysLength)) .< MLXArray(Int32(keysLength - 7 * seed - 3)))
                        .reshaped([1, keysLength])
                    let scale = 1 / Float(headDim).squareRoot()
                    let reference = MLXFast.scaledDotProductAttention(
                        queries: q, keys: k, values: v, scale: scale, mask: mask)[0..., 0..., ..<16, 0...]
                    let split = MLXFast.scaledDotProductAttention(
                        queries: q[0..., 0..., ..<16, 0...], keys: k, values: v, scale: scale, mask: mask)
                    same = same && split.shape == reference.shape
                        && all(split.view(dtype: .uint16) .== reference.view(dtype: .uint16)).item(Bool.self)
                    compared += reference.size
                }
                try error.check()
            }
        } catch {
            same = false
        }
        padQueries = !same
        FileHandle.standardError.write(
            ("dflash2 block attention key pipeline: "
                + (same ? "self-test passed: \(compared) values compared bitwise, 0 mismatches; pipelined\n"
                    : "self-test failed; queries padded to 17 rows (untouched path)\n")).data(using: .utf8)!)
    }
}

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
            // The stack stays: the prompt's context rows read it too.
            read.append(stacked)
            read += DFlash2PackedWeights.copy(of: stacked)?.arrays ?? []
            replaced += [qProj.weight, kProj.weight, vProj.weight]
        }
        if let packed = DFlash2PackedWeights.copy(of: oProj.weight) {
            read += packed.arrays
            replaced.append(oProj.weight)
            return (read, replaced)
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
        let kRows = projectedK.reshaped(B, contextLength, kvHeads, -1)
        let vRows = projectedV.reshaped(B, contextLength, kvHeads, -1)
        // A fresh cache: the norm, rope and first append in one launch.
        if DFlash2AbsorbKV.enabled,
            let capacity = block.firstAppendCapacity(contextRows: contextLength),
            let (keys, values) = DFlash2AbsorbKV.apply(
                kRows, vRows, kNorm: kNorm, offset: cache.offset, capacity: capacity),
            block.installFirst(keys: keys, values: values, contextRows: contextLength)
        {
            return true
        }
        let keys = rope(
            DFlash2StridedRMSNorm.apply(kNorm, kRows).transposed(0, 2, 1, 3),
            offset: cache.offset)
        let values = vRows.transposed(0, 2, 1, 3)
        return block.updateBlock(keys: keys, values: values, contextRows: contextLength) != nil
    }

    func prepareQueryWindow() -> Bool {
        qkv.prepareQueryWindow(q: qProj, k: kProj, v: vProj)
    }

    func disableQueryWindow() { qkv.disableQueryWindow() }

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
    ///
    /// `joined`, when given, is that `rows` matrix already (`x` then only
    /// gives the block's shape; `DFlash2SpeculativeFront.rows`), and `pos`
    /// (`[offset, offset, L]`) lets the q and k heads take one launch
    /// (`DFlash2SpeculativeFront.heads`).
    func speculative(
        _ x: MLXArray, joined: MLXArray? = nil, base: MLXArray, confirmed: MLXArray,
        queryOffset: MLXArray, pos: MLXArray? = nil, rope: RoPELayer,
        cache: DFlash2BlockKVCache, keyMask: MLXArray
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray)? {
        let (B, L, n) = (x.dim(0), x.dim(1), base.dim(1))
        guard B == 1, n == 2 * L, let held = cache.inPlaceRows,
            joined.map({ $0.shape == [B, n, x.dim(2)] }) ?? true
        else { return nil }
        let start = confirmed.reshaped([1])
        let rows = joined ?? dynamicSliceUpdate(base, update: x, start: start, axes: [1])
        guard case let (y, qEnd, kEnd)? = qkv.applyStacked(
            rows, q: qProj, k: kProj, v: vProj, confirmed: confirmed)
        else { return nil }
        let queries: MLXArray
        let allKeys: MLXArray
        if let pos, qEnd == heads * qNorm.weight.dim(0),
            let (q, k) = DFlash2SpeculativeFront.heads(
                y, blockRows: L, qNorm: qNorm, kNorm: kNorm, confirmed: confirmed, pos: pos)
        {
            (queries, allKeys) = (q, k)
        } else {
            let blockRows = dynamicSlice(
                y, start: start, axes: [1], sliceSize: [Int32(B), Int32(L), Int32(y.dim(2))])
            queries = rope(
                qNorm(blockRows[.ellipsis, ..<qEnd].reshaped(B, L, heads, -1))
                    .transposed(0, 2, 1, 3),
                offset: queryOffset)
            allKeys = rope(
                kNorm(y[.ellipsis, qEnd ..< kEnd].reshaped(B, n, kvHeads, -1))
                    .transposed(0, 2, 1, 3),
                offset: cache.offset)
        }
        let allValues = y[.ellipsis, kEnd...].reshaped(B, n, kvHeads, -1).transposed(0, 2, 1, 3)
        guard case let (keys, values)? = cache.speculativeRows(keys: allKeys, values: allValues)
        else { return nil }
        let output = DFlash2AttentionPipeline.attend(
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
    private var queryWindowChoice: DFlash2PackedWeights.QueryWindowChoice?

    func clear() {
        weight = nil
        qEnd = 0
        kEnd = 0
        queryWindowChoice = nil
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

    func prepareQueryWindow(q: Linear, k: Linear, v: Linear) -> Bool {
        queryWindowChoice = nil
        guard let w = stacked(q: q, k: k, v: v) else { return false }
        queryWindowChoice = DFlash2PackedWeights.prepareQueryWindow(weight: w, qColumns: qEnd)
        return queryWindowChoice != nil
    }

    func disableQueryWindow() { queryWindowChoice = nil }

    /// `apply`'s matmul, unsliced, with its q and k column ends.
    func applyStacked(
        _ rows: MLXArray, q: Linear, k: Linear, v: Linear, confirmed: MLXArray? = nil
    ) -> (y: MLXArray, qEnd: Int, kEnd: Int)? {
        guard rows.ndim == 3, let weight = stacked(q: q, k: k, v: v) else { return nil }
        if let confirmed, let choice = queryWindowChoice,
            let y = DFlash2PackedWeights.applyQueryWindow(
                rows, weight: weight, confirmed: confirmed, choice: choice)
        {
            return (y, qEnd, kEnd)
        }
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
            ("QUAD", dflash2ConvQuad(dtype, groupSize)),
        ]
        if let context {
            let rows = context.dim(1) + length
            return dflash2GroupedConvJoinKernel(
                [hidden, projection, baseKernel, context],
                template: template, grid: (dflash2ConvColumns(hiddenSize, dtype, groupSize), rows, batch), threadGroup: (256, 1, 1),
                outputShapes: [[batch, rows, hiddenSize]], outputDTypes: [dtype])[0]
        }
        let grid = (dflash2ConvColumns(hiddenSize, dtype, groupSize), length, batch)
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

    /// ``prepare(_:)`` written straight into the speculative block's rows:
    /// `dynamicSliceUpdate([context; zeros], update: conv, start: confirmed)`
    /// over `rows` rows in one launch (`DFlash2SpeculativeFront.rows`), with
    /// the same projection; nil where the tap-0 kernel or that launch does
    /// not apply.
    func prepareSpeculative(
        _ hidden: MLXArray, context: MLXArray, confirmed: MLXArray, rows: Int
    ) -> (rows: MLXArray, projection: MLXArray)? {
        guard dflash2FusedConvEnabled, DFlash2SpeculativeFront.rowsChosen, hidden.ndim == 3,
            hidden.dim(2) == groups * groupSize, hidden.dim(2) % 256 == 0,
            baseKernel.shape == [2, kernelSize, hidden.dim(2)], baseKernel.dtype == hidden.dtype
        else { return nil }
        let projection = dflash2KernelProjectionTensor
            ? DFlash2TensorMatmul.linear(kernelProjection, hidden) : kernelProjection(hidden)
        guard projection.shape == [hidden.dim(0), hidden.dim(1), 2 * kernelSize * groups],
            projection.dtype == hidden.dtype,
            let joined = DFlash2SpeculativeFront.rows(
                hidden, projection: projection, base: baseKernel, context: context,
                confirmed: confirmed, rows: rows, kernelSize: kernelSize, groupSize: groupSize)
        else { return nil }
        return (joined, projection)
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
                            ("T", dtype), ("KS", ks), ("GS", gs), ("TAP", 0), ("QUAD", dflash2ConvQuad(dtype, gs)),
                        ]
                        let fused = dflash2GroupedConvJoinKernel(
                            [h, dyn, base, context], template: template, grid: (dflash2ConvColumns(hs, dtype, gs), c + l, 1),
                            threadGroup: (256, 1, 1), outputShapes: [[1, c + l, hs]],
                            outputDTypes: [dtype])[0]
                        let conv = dflash2GroupedConvKernel(
                            [h, dyn, base], template: template, grid: (dflash2ConvColumns(hs, dtype, gs), l, 1),
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
/// Four adjacent columns share one dynamic tap. Packed 16-bit loads/stores
/// preserve each T rounding; FP32 and groups narrower than four keep scalar.
private func dflash2ConvQuad(_ dtype: DType, _ groupSize: Int) -> Bool {
    [DType.bfloat16, .float16].contains(dtype) && groupSize % 4 == 0
}
private func dflash2ConvColumns(_ width: Int, _ dtype: DType, _ groupSize: Int) -> Int {
    dflash2ConvQuad(dtype, groupSize) ? width / 4 : width
}

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
    template <typename T, int KS, int GS, int TAP>
    inline uint2 dflash2_grouped_conv4(
        const device T* h, const device T* dyn, const device T* base,
        uint b, uint l, uint c, uint L, uint H) {
    #pragma clang fp contract(off)
      const uint G = H / GS;
      const uint g = c / GS;
      const size_t row = size_t(b) * L + l;
      T acc[4] = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
      #pragma clang loop unroll(full)
      for (int o = 0; o < KS; ++o) {
        const uint2 vb = (l >= uint(o)) ? *((const device uint2*)(h + (row - o) * H + c)) : uint2(0u);
        const uint2 kbb = *((const device uint2*)(base + (size_t(TAP) * KS + o) * H + c));
        const T d = dyn[row * (2 * KS * G) + (size_t(TAP) * KS + o) * G + g];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 4u; ++r) {
          const T v = as_type<T>(ushort((vb[r >> 1] >> ((r & 1u) * 16u)) & 65535u));
          const T kb = as_type<T>(ushort((kbb[r >> 1] >> ((r & 1u) * 16u)) & 65535u));
          const T kv = kb * v;
          acc[r] = acc[r] + kv;
          const T dv = d * v;
          acc[r] = acc[r] + dv;
        }
      }
      return uint2(uint(as_type<ushort>(acc[0])) | (uint(as_type<ushort>(acc[1])) << 16u),
                                 uint(as_type<ushort>(acc[2])) | (uint(as_type<ushort>(acc[3])) << 16u));
    }
    template <typename T>
    inline uint2 dflash2_conv_residual4(uint2 conv, uint2 residual) {
    #pragma clang fp contract(off)
      T result[4];
      #pragma clang loop unroll(full)
      for (uint r = 0; r < 4u; ++r) {
        const T cv = as_type<T>(ushort((conv[r >> 1] >> ((r & 1u) * 16u)) & 65535u));
        const T rv = as_type<T>(ushort((residual[r >> 1] >> ((r & 1u) * 16u)) & 65535u));
        result[r] = rv + cv;
      }
      return uint2(uint(as_type<ushort>(result[0])) | (uint(as_type<ushort>(result[1])) << 16u),
                   uint(as_type<ushort>(result[2])) | (uint(as_type<ushort>(result[3])) << 16u));
    }
    """

private let dflash2GroupedConvSource = """
    const uint c = thread_position_in_grid.x * (QUAD ? 4u : 1u);
    const uint l = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint H = threads_per_grid.x * (QUAD ? 4u : 1u);
    const uint L = threads_per_grid.y;
    if constexpr (QUAD) {
      *((device uint2*)(out + (size_t(b) * L + l) * H + c)) =
          dflash2_grouped_conv4<T, KS, GS, TAP>(h, dyn, base, b, l, c, L, H);
    } else {
      out[(size_t(b) * L + l) * H + c] =
          dflash2_grouped_conv<T, KS, GS, TAP>(h, dyn, base, b, l, c, L, H);
    }
    """

private let dflash2GroupedConvResidualSource = """
    #pragma clang fp contract(off)
    const uint c = thread_position_in_grid.x * (QUAD ? 4u : 1u);
    const uint l = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint H = threads_per_grid.x * (QUAD ? 4u : 1u);
    const uint L = threads_per_grid.y;
    const size_t i = (size_t(b) * L + l) * H + c;
    if constexpr (QUAD) {
      const uint2 conv = dflash2_grouped_conv4<T, KS, GS, TAP>(h, dyn, base, b, l, c, L, H);
      *((device uint2*)(out + i)) = dflash2_conv_residual4<T>(conv, *((const device uint2*)(res + i)));
    } else {
      const T conv = dflash2_grouped_conv<T, KS, GS, TAP>(h, dyn, base, b, l, c, L, H);
      out[i] = res[i] + conv;
    }
    """

/// `[ctx; conv(h)]`: rows below `ctx`'s count are its rows, the rest the
/// tap's convolution of the block.
private let dflash2GroupedConvJoinSource = """
    const uint c = thread_position_in_grid.x * (QUAD ? 4u : 1u);
    const uint r = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint H = threads_per_grid.x * (QUAD ? 4u : 1u);
    const uint R = threads_per_grid.y;
    const uint C = uint(ctx_shape[1]);
    if constexpr (QUAD) {
      const uint2 v = r < C ? *((const device uint2*)(ctx + (size_t(b) * C + r) * H + c))
          : dflash2_grouped_conv4<T, KS, GS, TAP>(h, dyn, base, b, r - C, c, R - C, H);
      *((device uint2*)(out + (size_t(b) * R + r) * H + c)) = v;
    } else {
      out[(size_t(b) * R + r) * H + c] = r < C
          ? ctx[(size_t(b) * C + r) * H + c]
          : dflash2_grouped_conv<T, KS, GS, TAP>(h, dyn, base, b, r - C, c, R - C, H);
    }
    """

/// 32-bit element offsets (`Qwen35IO32`, `MLXFAST_IO32_GDN=0` keeps the stock
/// texts) for the drafter's per-round kernels: the grouped conv, its residual,
/// join and `_join_at` forms, the BF16 tensor matmuls (m16, m32, their swapped
/// forms and the K-split variant), `dflash2_qk_prework_at`, the top-k chunk,
/// threshold and merge passes, and the four-wide concat. Every narrowed offset
/// is a non-negative element / word index below 2^31: at most (262144 context
/// + 64) rows x 6144 columns, 32 rows x the 248320-entry vocabulary, a
/// 34816 x 5120 weight, or a concat of at most 2^20 elements. The grouped
/// conv header's eight (block row, h / dyn / base offsets) stay below each
/// operand's element count: (262144 + 64) rows x 5120 at most.
private let dflash2GroupedConvHeaderIO32 = Qwen35IO32.narrow(
    dflash2GroupedConvHeader, count: 8, "dflash2_grouped_conv_header")

private let dflash2GroupedConvJoinKernel = MLXFast.metalKernel(
    name: "dflash2_grouped_conv_join",
    inputNames: ["h", "dyn", "base", "ctx"],
    outputNames: ["out"],
    source: Qwen35IO32.narrow(dflash2GroupedConvJoinSource, count: 4, "dflash2_grouped_conv_join"),
    header: dflash2GroupedConvHeaderIO32,
    ensureRowContiguous: true)

private let dflash2GroupedConvKernel = MLXFast.metalKernel(
    name: "dflash2_grouped_conv",
    inputNames: ["h", "dyn", "base"],
    outputNames: ["out"],
    source: Qwen35IO32.narrow(dflash2GroupedConvSource, count: 2, "dflash2_grouped_conv"),
    header: dflash2GroupedConvHeaderIO32,
    ensureRowContiguous: true)

private let dflash2GroupedConvResidualKernel = MLXFast.metalKernel(
    name: "dflash2_grouped_conv_residual",
    inputNames: ["h", "dyn", "base", "res"],
    outputNames: ["out"],
    source: Qwen35IO32.narrow(dflash2GroupedConvResidualSource, count: 2, "dflash2_grouped_conv_residual"),
    header: dflash2GroupedConvHeaderIO32,
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
          for (int i = 0; i < 16; i += 4) {
            float v[4];
            #pragma clang loop unroll(full)
            for (int c = 0; c < 4; c++) {
              if constexpr (SPLITS == 2) {
                v[c] = cT[(i + c)] + red[0][(i + c) * 32 + lane];
              } else {
                v[c] = cT[(i + c)] + red[0][(i + c) * 32 + lane] + red[1][(i + c) * 32 + lane] + red[2][(i + c) * 32 + lane];
              }
            }
            const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
            const size_t base = (size_t)(fm + 8 * mh) * N + n0 + fn + 16 * nh;
            *(device vec<OutT, 4>*)(out + base) = vec<OutT, 4>(
                OutT(v[0]), OutT(v[1]), OutT(v[2]), OutT(v[3]));
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
          for (int i = 0; i < 16; i += 4) {
            float v0[4], v1[4];
            #pragma clang loop unroll(full)
            for (int c = 0; c < 4; c++) {
              if constexpr (SPLITS == 2) {
                v0[c] = cT0[(i + c)] + red[0][(i + c) * 32 + lane];
                v1[c] = cT1[(i + c)] + red[0][(16 + (i + c)) * 32 + lane];
              } else {
                v0[c] = cT0[(i + c)] + red[0][(i + c) * 32 + lane] + red[1][(i + c) * 32 + lane] + red[2][(i + c) * 32 + lane];
                v1[c] = cT1[(i + c)] + red[0][(16 + (i + c)) * 32 + lane] + red[1][(16 + (i + c)) * 32 + lane]
                    + red[2][(16 + (i + c)) * 32 + lane];
              }
            }
            const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
            const size_t base = (size_t)(fm + 8 * mh) * N + n0 + fn + 16 * nh;
            *(device vec<OutT, 4>*)(out + base) = vec<OutT, 4>(
                OutT(v0[0]), OutT(v0[1]), OutT(v0[2]), OutT(v0[3]));
            *(device vec<OutT, 4>*)(out + base + 16 * N) = vec<OutT, 4>(
                OutT(v1[0]), OutT(v1[1]), OutT(v1[2]), OutT(v1[3]));
          }
        }
        """

    private static let kernel32 = MLXFast.metalKernel(
        name: "dflash2_bf16_matmul_m32",
        inputNames: ["x", "w", "ksz"],
        outputNames: ["out"],
        source: Qwen35IO32.narrow(source32, count: 2, "dflash2_bf16_matmul_m32"),
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
        source: Qwen35IO32.narrow(source, count: 2, "dflash2_bf16_matmul_m16"),
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
        using OutIndexT = metal::conditional_t<IO32 != 0, uint, size_t>;
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
            } else if constexpr (SPLITS == 4) {
              v = cT[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
            } else {
              v = cT[i];
              for (int j = 0; j < SPLITS - 1; j++) { v += red[j][i * 32 + lane]; }
            }
            // Destination coordinates: [0] the input row, [1] the column in the block.
            auto idx = cT.get_multidimensional_index(i);
            out[(OutIndexT)idx[0] * N + n0 + idx[1]] = OutT(v);
          }
        }
        """

    private static let kernelSwapped = MLXFast.metalKernel(
        name: "dflash2_bf16_matmul_m16s",
        inputNames: ["x", "w", "ksz"],
        outputNames: ["out"],
        source: Qwen35IO32.narrow(sourceSwapped, count: 1, "dflash2_bf16_matmul_m16s"),
        header: header,
        ensureRowContiguous: true)

    /// `source32` in `sourceSwapped`'s form: one cooperative `[32, KT]`
    /// weight slice per K step, multiplied by the input's rows 0-15 and
    /// 16-31 (two `32 x 16 x KT` ops into two accumulators), the same K
    /// partitions and the same partial order as `source32`, so its bits
    /// (`prepareSwapped` compares them at load).
    private static let sourceSwapped32 = """
        using OutIndexT = metal::conditional_t<IO32 != 0, uint, size_t>;
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
            } else if constexpr (SPLITS == 4) {
              v0 = cT0[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
              v1 = cT1[i] + red[0][(16 + i) * 32 + lane] + red[1][(16 + i) * 32 + lane]
                  + red[2][(16 + i) * 32 + lane];
            } else {
              v0 = cT0[i]; v1 = cT1[i];
              for (int j = 0; j < SPLITS - 1; j++) {
                v0 += red[j][i * 32 + lane]; v1 += red[j][(16 + i) * 32 + lane];
              }
            }
            auto idx = cT0.get_multidimensional_index(i);
            out[(OutIndexT)idx[0] * N + n0 + idx[1]] = OutT(v0);
            out[(OutIndexT)(16 + idx[0]) * N + n0 + idx[1]] = OutT(v1);
          }
        }
        """

    private static let kernelSwapped32 = MLXFast.metalKernel(
        name: "dflash2_bf16_matmul_m32s",
        inputNames: ["x", "w", "ksz"],
        outputNames: ["out"],
        source: Qwen35IO32.narrow(sourceSwapped32, count: 1, "dflash2_bf16_matmul_m32s"),
        header: header,
        ensureRowContiguous: true)

    /// Whether 17-32-row launches take `sourceSwapped32` (set by `prepareSwapped`).
    nonisolated(unsafe) static var swapped32Active = false

    /// The standard K step of `sourceSwapped` (`MLXFAST_DRAFT_SWAP_KT`: 64 or
    /// 128, default 128; `SwapTrial` may set another per shape).
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
            for (int i = 0; i < 16; i += 4) {
              float v[4];
              #pragma clang loop unroll(full)
              for (int c = 0; c < 4; c++) {
                v[c] = h == 0 ? cT0[(i + c)] : cT1[(i + c)];
                if constexpr (SPLITS == 2) {
                  v[c] = v[c] + red[0][h][(i + c) * 32 + lane];
                } else if constexpr (SPLITS == 4) {
                  v[c] = v[c] + red[0][h][(i + c) * 32 + lane] + red[1][h][(i + c) * 32 + lane] + red[2][h][(i + c) * 32 + lane];
                } else {
                  #pragma clang loop unroll(full)
                  for (int j = 0; j < SPLITS - 1; j++) { v[c] = v[c] + red[j][h][(i + c) * 32 + lane]; }
                }
              }
              const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
              const size_t base = (size_t)(fm + 8 * mh) * N + n0 + 32 * h + fn + 16 * nh;
              *(device vec<OutT, 4>*)(out + base) = vec<OutT, 4>(
                  OutT(v[0]), OutT(v[1]), OutT(v[2]), OutT(v[3]));
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
        source: Qwen35IO32.narrow(variantSource, count: 4, "dflash2_bf16_matmul_m16_kvar"),
        header: header,
        ensureRowContiguous: true)

    private static let dimsLock = NSLock()
    nonisolated(unsafe) private static var dims: [[Int]: MLXArray] = [:]
    fileprivate static func dimsArray(k: Int, n: Int) -> MLXArray {
        dimsLock.withLock {
            if let cached = dims[[k, n]] { return cached }
            let array = MLXArray([Int32(k), Int32(rowsPerTile), Int32(n)])
            dims[[k, n]] = array
            return array
        }
    }

    static var packed32Available: Bool {
        enabled && rows32Enabled && Qwen35TensorPackedMatmul.tensorOperandsAvailable
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
            // The packed copy's kernel gives the same bits (`DFlash2PackedWeights`).
            let y = DFlash2PackedWeights.apply32(a, weight, outputDType: .bfloat16)
                ?? launch32(a, weight, k: k, n: n, swapped: swapped32Active, outputDType: .bfloat16)
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
            // The swapped kernel reads the stored weight (no tiled copy exists
            // while it is on), or its packed copy (`DFlash2PackedWeights`).
            y = DFlash2PackedWeights.apply(a, weight, outputDType: .bfloat16)
                ?? launch(a, weight, k: k, n: n, tiled: false, outputDType: .bfloat16)
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
            return launchSwapped(a, w, k: k, n: n, outputDType: outputDType)
        }
        return kernel(
            [a, w, dimsArray(k: k, n: n)],
            template: [("OutT", outputDType), ("SPLITS", splits), ("TILED", tiled ? 1 : 0)],
            grid: (n / 32 * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rowsPerTile, n]], outputDTypes: [outputDType])[0]
    }

    /// `source32` (or `sourceSwapped32` when `swapped`, with `tiling` or the
    /// shape's `swapTiling`) over a 32-row `a`.
    private static func launch32(
        _ a: MLXArray, _ w: MLXArray, k: Int, n: Int, swapped: Bool, tiling: SwapTiling? = nil,
        outputDType: DType
    ) -> MLXArray {
        if swapped {
            let t = tiling ?? swapTiling(k: k, n: n, rows32: true)
            return kernelSwapped32(
                [a, w, dimsArray(k: k, n: n)],
                template: [
                    ("OutT", outputDType), ("SPLITS", t.splits), ("KT", t.kt),
                    ("IO32", n > 0 && n <= Int(Int32.max) / 32 ? 1 : 0),
                ],
                grid: (n / 32 * t.splits * 32, 1, 1), threadGroup: (t.splits * 32, 1, 1),
                outputShapes: [[2 * rowsPerTile, n]], outputDTypes: [outputDType])[0]
        }
        let splits = n >= 16384 ? 2 : 4
        let threads = splits * 32
        return kernel32(
            [a, w, dimsArray(k: k, n: n)],
            template: [("OutT", outputDType), ("SPLITS", splits)],
            grid: (n / 32 * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[2 * rowsPerTile, n]], outputDTypes: [outputDType])[0]
    }

    /// `sourceSwapped` over a 16-row `a` and the stored `[N, K]` weight, with
    /// `tiling` or the shape's `swapTiling` (the stock kernel's K split unless
    /// a forced choice or `SwapTrial` set another).
    private static func launchSwapped(
        _ a: MLXArray, _ w: MLXArray, k: Int, n: Int, tiling: SwapTiling? = nil, outputDType: DType
    ) -> MLXArray {
        let t = tiling ?? swapTiling(k: k, n: n, rows32: false)
        let threads = t.splits * 32
        return kernelSwapped(
            [a, w, dimsArray(k: k, n: n)],
            template: [
                ("OutT", outputDType), ("SPLITS", t.splits), ("KT", t.kt),
                ("IO32", n > 0 && n <= Int(Int32.max) / 32 ? 1 : 0),
            ],
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
        swapTilings = [:]
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
                let swapped = launchSwapped(a, w, k: k, n: n, outputDType: outputDType)
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

    // MARK: The block-width kernels' tiling

    /// A block-width launch's tiling. The swapped BF16 kernel (`sourceSwapped`,
    /// `sourceSwapped32`; `ahead` nil): the K step `KT` and the K split `SPLITS`
    /// (simdgroups per 32-column threadgroup, each over one contiguous
    /// `K / SPLITS` slab). The packed 12-bit kernel (`DFlash2PackedWeights`,
    /// `[cols, 64]` tiles, 16 columns per threadgroup by default): its K step
    /// is the copy's tile (`DFlash2PackedWeights.ks`: another K step or tile
    /// width would need another copy, so neither is a knob), so its knobs are the
    /// split and `ahead`, how many tiles before its op a simdgroup reads a
    /// tile's share (the record: one). A K step keeps each slab's products in
    /// K order and a look-ahead moves only the reads (bit for bit where the op
    /// accumulates in K order: checked); a split reorders the FP32 sum.
    struct SwapTiling: Hashable, CustomStringConvertible {
        var kt: Int, splits: Int, ahead: Int?
        var description: String { "t\(kt)s\(splits)" + (ahead.map { "a\($0)" } ?? "") }
        func fits(k: Int) -> Bool { k % splits == 0 && (k / splits) % kt == 0 }

        init(kt: Int, splits: Int, ahead: Int? = nil) { (self.kt, self.splits, self.ahead) = (kt, splits, ahead) }

        /// `t<KT>s<S>` for the swapped kernel (KT 64, 128 or 256) or
        /// `t<tile>s<S>a<A>` for the packed one (A 0, 1 or 2), `standard`'s
        /// kernel only; S 2, 4, 8, or 0 for `standard`'s split.
        init?(name: String, standard: SwapTiling) {
            let v = name.split(whereSeparator: { "tsa".contains($0) }).compactMap { Int($0) }
            let packed = standard.ahead != nil
            guard v.count == (packed ? 3 : 2), name == "t\(v[0])s\(v[1])" + (packed ? "a\(v[2])" : ""),
                (packed ? [DFlash2PackedWeights.ks] : [64, 128, 256]).contains(v[0]),
                [0, 2, 4, 8].contains(v[1]), !packed || (0 ... 2).contains(v[2])
            else { return nil }
            self.init(kt: v[0], splits: v[1] == 0 ? standard.splits : v[1], ahead: packed ? v[2] : nil)
        }
    }

    /// A drafter projection as a block-width launch sees it: `[N, K]`, 16 or 32 rows.
    struct SwapShape: Hashable, CustomStringConvertible {
        let k: Int, n: Int, rows32: Bool
        var description: String { "\(n)x\(k)" + (rows32 ? "r32" : "") }
        /// The record's tiling of the kernel that runs it: the packed kernel's
        /// (its tile, the stock split, one tile ahead) or the swapped one's
        /// (`swappedKT` and the stock split).
        func standard(packed: Bool) -> SwapTiling {
            let splits = Kernel.stockSplits(n: n)
            return packed
                ? SwapTiling(kt: DFlash2PackedWeights.ks, splits: splits, ahead: 1)
                : SwapTiling(kt: swappedKT, splits: splits)
        }
    }

    /// Each shape's tiling where it is not the standard, read only by its own
    /// kernel (packed or swapped): set at load (forced) or by `SwapTrial` at
    /// the deferred warm, before anything is served.
    nonisolated(unsafe) static var swapTilings: [SwapShape: SwapTiling] = [:]

    static func swapTiling(k: Int, n: Int, rows32: Bool, packed: Bool = false) -> SwapTiling {
        let shape = SwapShape(k: k, n: n, rows32: rows32)
        guard !swapTilings.isEmpty, let t = swapTilings[shape], (t.ahead != nil) == packed else {
            return shape.standard(packed: packed)
        }
        return t
    }

    /// The block-width kernels' tiling per drafter projection shape, chosen
    /// at the deferred load warm, on the kernel the record runs there: the
    /// packed 12-bit kernel when `DFlash2PackedWeights` is on (splits 2, 4,
    /// 8 by look-ahead 0, 1, 2), else the swapped BF16 kernel (K steps 64,
    /// 128, 256 by splits 2, 4, 8). Each candidate runs on every weight of its
    /// shape against the standard (BF16 outputs): bitwise (`=`) or close (`~`:
    /// every value within one BF16 ulp of its row's largest magnitude), else
    /// dropped. A sample is one evaluation of a chain of launches over the
    /// shape's weights in turn, each input plus the previous output's first
    /// value, so they run one after another streaming distinct weights, as in
    /// a block forward. Per shape, rotating order, a discarded round and
    /// `rounds` more; a score is the median ratio to the standard. The pick is
    /// the fastest bitwise tiling, or a close one faster still by
    /// `pickMargin`, if it beats the standard by `pickMargin`; the picks are
    /// adopted only if a chain over every projection weight takes at most
    /// `1 - adoptMargin` of the standard's. A close tiling changes only the
    /// drafter's proposals. `MLXFAST_DRAFT_TILING_TRIAL=0`: no trial;
    /// `..._FORCE=<N>x<K>[r32]=<tiling>,...` installs those at load after the
    /// check; `..._LIST=` replaces the candidates.
    enum SwapTrial {
        private static func env(_ suffix: String) -> String? {
            let raw = ProcessInfo.processInfo.environment["MLXFAST_DRAFT_TILING_TRIAL" + suffix]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return raw?.isEmpty == false ? raw : nil
        }

        static let enabled = !["0", "false", "no", "off"].contains(env("")?.lowercased() ?? "")
        static let forced = env("_FORCE")
        static let listed = env("_LIST")?.split(separator: ",").map(String.init)
        static let rounds = 8
        static let adoptMargin = 0.02
        static let pickMargin = 0.01
        static let dropMargin = 0.05

        /// Whether the shapes run the packed kernel (set by `note`).
        nonisolated(unsafe) private static var packed = false
        nonisolated(unsafe) private static var weights: [SwapShape: [MLXArray]] = [:]
        nonisolated(unsafe) private static var sequence: [(SwapShape, MLXArray)] = []
        nonisolated(unsafe) private static var inputs: [SwapShape: MLXArray] = [:]
        private static var order: [SwapShape] {
            var seen = Set<SwapShape>()
            return sequence.map { $0.0 }.filter { seen.insert($0).inserted }
        }

        private static var list: [String] {
            listed ?? [2, 4, 8].flatMap { s in
                packed
                    ? (0 ... 2).map { "t\(DFlash2PackedWeights.ks)s\(s)a\($0)" } : [64, 128, 256].map { "t\($0)s\(s)" }
            }
        }

        private static func log(_ line: String) {
            FileHandle.standardError.write(Data(("dflash2 drafter GEMM tiling: " + line + "\n").utf8))
        }

        private static func median(_ values: [Double]) -> Double {
            let v = values.sorted()
            return v.isEmpty ? .nan : (v[(v.count - 1) / 2] + v[v.count / 2]) / 2
        }

        /// The block-width launches' weights by shape and the kernel that runs
        /// them (after `DFlash2PackedWeights.prepare`), then any forced tilings.
        static func note(_ ws: [MLXArray], rows32 ws32: [MLXArray]) {
            (weights, sequence, inputs) = ([:], [], [:])
            guard swappedActive else { return }
            packed = DFlash2PackedWeights.active
            let runs = { (w: MLXArray) -> Bool in
                w.dtype == .bfloat16 && w.ndim == 2 && w.dim(1) % 1024 == 0 && w.dim(0) % 32 == 0
                    && (!packed || DFlash2PackedWeights.copy(of: w) != nil)
            }
            sequence = ws.filter(runs).map { (SwapShape(k: $0.dim(1), n: $0.dim(0), rows32: false), $0) }
                + (rows32Enabled && (packed || swapped32Active) ? ws32.filter(runs) : []).map {
                    (SwapShape(k: $0.dim(1), n: $0.dim(0), rows32: true), $0)
                }
            weights = Dictionary(grouping: sequence, by: { $0.0 }).mapValues { $0.map { $0.1 } }
            guard let forced else { return }
            let shapes = order
            let lines = forced.split(separator: ",").map { entry -> String in
                let parts = entry.split(separator: "=").map(String.init)
                guard parts.count == 2, let shape = shapes.first(where: { "\($0)" == parts[0] }),
                    let t = SwapTiling(name: parts[1], standard: shape.standard(packed: packed)), t.fits(k: shape.k)
                else { return "\(entry) ignored (not a shape and fitting tiling)" }
                guard let c = check(t, shape), c.within else { return "\(shape)=\(t) FAILED its check" }
                swapTilings[shape] = t
                return "\(shape)=\(t) " + (c.exact ? "bitwise" : String(format: "close (%.2f ulp)", c.worst))
            }
            log("forced (\(packed ? "packed" : "swapped") kernel): " + lines.joined(separator: ", ")
                + "; elsewhere the record's; no trial")
        }

        private static func input(_ shape: SwapShape) -> MLXArray {
            if let x = inputs[shape] { return x }
            let x = MLXRandom.normal(
                [shape.rows32 ? 2 * rowsPerTile : rowsPerTile, shape.k], key: MLXRandom.key(UInt64(9603 + shape.k))
            ).asType(.bfloat16)
            eval(x)
            inputs[shape] = x
            return x
        }

        private static func launch(_ x: MLXArray, _ w: MLXArray, _ shape: SwapShape, _ t: SwapTiling) -> MLXArray {
            if t.ahead != nil, let c = DFlash2PackedWeights.copy(of: w) {
                return DFlash2PackedWeights.launch(
                    x, c, rows: shape.rows32 ? 2 * rowsPerTile : rowsPerTile, outputDType: .bfloat16, tiling: t)
            }
            return shape.rows32
                ? launch32(x, w, k: shape.k, n: shape.n, swapped: true, tiling: t, outputDType: .bfloat16)
                : launchSwapped(x, w, k: shape.k, n: shape.n, tiling: t, outputDType: .bfloat16)
        }

        /// `t` against the standard on every weight of `shape`: bitwise, the worst
        /// distance in ulps of the row's largest magnitude, all within one; nil
        /// on an MLX error.
        private static func check(_ t: SwapTiling, _ shape: SwapShape) -> (exact: Bool, worst: Float, within: Bool)? {
            try? withError { error -> (exact: Bool, worst: Float, within: Bool) in
                var (worst, within, differ) = ([MLXArray](), [MLXArray](), [MLXArray]())
                for w in weights[shape] ?? [] {
                    let ref16 = launch(input(shape), w, shape, shape.standard(packed: packed))
                    let y16 = launch(input(shape), w, shape, t)
                    let ref = ref16.asType(.float32)
                    let ulp = MLX.pow(MLXArray(Float(2)), MLX.floor(MLX.log2(MLX.abs(ref).max(axis: -1, keepDims: true))) - 7)
                    let ratio = MLX.abs(y16.asType(.float32) - ref) / ulp
                    worst.append(ratio.max())
                    within.append((ratio .<= Float(1)).all())
                    differ.append((y16.view(dtype: .uint16) .!= ref16.view(dtype: .uint16)).asType(.int32).sum())
                }
                let (w, ok, d) = (stacked(worst).max(), stacked(within).all(), stacked(differ).sum())
                eval(w, ok, d)
                try error.check()
                return (d.item(Int.self) == 0, w.item(Float.self), ok.item(Bool.self))
            }
        }

        /// Host nanoseconds of one evaluation of `steps` in turn, each input
        /// plus the previous output's first value (so they run one after another).
        private static func time(_ steps: [(SwapShape, MLXArray, SwapTiling)]) -> Double {
            var outputs: [MLXArray] = []
            for (shape, w, t) in steps {
                let x = outputs.last.map { input(shape) + $0[0 ..< 1, 0 ..< 1] } ?? input(shape)
                outputs.append(launch(x, w, shape, t))
            }
            let begin = DispatchTime.now().uptimeNanoseconds
            eval(outputs)
            return Double(DispatchTime.now().uptimeNanoseconds - begin)
        }

        /// The trial (see the type's notes), once; nothing unless `note` ran.
        static func run() {
            guard !sequence.isEmpty, swappedActive else { return }
            defer { (weights, sequence, inputs) = ([:], [], [:]) }
            guard enabled else { return log("MLXFAST_DRAFT_TILING_TRIAL=0; the record's tiling, no trial") }
            guard forced == nil else { return }
            let start = DispatchTime.now().uptimeNanoseconds
            var picks: [SwapShape: SwapTiling] = [:]
            var parts: [String] = []
            var (bitwise, close, far, errors) = (0, 0, 0, 0)
            // Every shape's checks (and so every first compile) before any timing.
            var survivors: [SwapShape: (opts: [SwapTiling], tags: [String])] = [:]
            for shape in order {
                let std = shape.standard(packed: packed)
                var (opts, tags) = ([std], ["="])
                for name in list {
                    guard let t = SwapTiling(name: name, standard: std), t.fits(k: shape.k), !opts.contains(t)
                    else { continue }
                    guard let c = check(t, shape) else { errors += 1; continue }
                    guard c.within else { far += 1; continue }
                    opts.append(t)
                    tags.append(c.exact ? "=" : "~")
                    if c.exact { bitwise += 1 } else { close += 1 }
                }
                survivors[shape] = (opts, tags)
            }
            let checking = Double(DispatchTime.now().uptimeNanoseconds - start)
            // The checks leave the GPU mostly idle (seconds of first compiles in a fresh
            // process), so the record's chain keeps it busy for 300 ms (at least 5 chains)
            // before anything is timed.
            let record = sequence.map { ($0.0, $0.1, $0.0.standard(packed: packed)) }
            let settle = DispatchTime.now().uptimeNanoseconds
            var settled = 0
            while settled < 5 || DispatchTime.now().uptimeNanoseconds - settle < 300_000_000 {
                _ = time(record)
                settled += 1
            }
            for shape in order {
                let std = shape.standard(packed: packed)
                let (opts, tags) = survivors[shape] ?? ([std], ["="])
                let ws = weights[shape] ?? []
                let count = max(ws.count, 10)
                var times = [[Double]](repeating: [], count: opts.count)
                var active = Array(opts.indices)
                func score(_ f: Int) -> Double { median(zip(times[f], times[0]).map { $0 / $1 }) - 1 }
                for round in 0 ... rounds {
                    for j in active.indices {
                        let f = active[(j + round) % active.count]
                        let t = time((0 ..< count).map { (shape, ws[$0 % ws.count], opts[f]) })
                        if round > 0 { times[f].append(t) }
                    }
                    if round == 2 { active = active.filter { $0 == 0 || score($0) <= dropMargin } }
                }
                let ranked = opts.indices.sorted { score($0) < score($1) }
                var best = ranked.first { tags[$0] == "=" } ?? 0
                if let c = ranked.first(where: { tags[$0] == "~" }), score(c) < score(best) - pickMargin { best = c }
                if best > 0, score(best) < -pickMargin { picks[shape] = opts[best] }
                parts.append("\(shape) x\(count) \(std) " + String(format: "%.3f ms", median(times[0]) / 1e6)
                    + opts.indices.dropFirst().map { " \(opts[$0])\(tags[$0]) " + String(format: "%+.1f%%", score($0) * 100) }.joined())
            }
            var verdict = "the record's kept (no pick)"
            if !picks.isEmpty {
                func chain(_ mine: Bool) -> [(SwapShape, MLXArray, SwapTiling)] {
                    mine ? record.map { ($0.0, $0.1, picks[$0.0] ?? $0.2) } : record
                }
                var (base, mine) = ([Double](), [Double]())
                for round in 0 ... rounds {
                    let first = round % 2 == 1
                    let (a, b) = (time(chain(first)), time(chain(!first)))
                    if round > 0 { base.append(first ? b : a); mine.append(first ? a : b) }
                }
                let ratio = median(zip(mine, base).map { $0 / $1 })
                let adopt = ratio <= 1 - adoptMargin
                if adopt { swapTilings = picks }
                verdict = "picks " + order.compactMap { s in picks[s].map { "\(s)=\($0)" } }.joined(separator: ",")
                    + String(format: ": all %d projections %.3f -> %.3f ms (%+.1f%%) -> ", sequence.count,
                        median(base) / 1e6, median(mine) / 1e6, (ratio - 1) * 100)
                    + (adopt ? "adopted" : String(format: "the record's kept (under %.0f%%)", adoptMargin * 100))
            }
            log("trial on the \(packed ? "packed 12-bit" : "swapped BF16") kernel (\(rounds) rounds; \(bitwise) bitwise, \(close) close, \(far) past 1 ulp, \(errors) MLX errors): "
                + parts.joined(separator: " | ") + " -> " + verdict
                + String(format: "; %.0f ms (checks and compiles %.0f)",
                    Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6, checking / 1e6))
        }
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
    /// while `current` reads the tiled copies (`tiled` and every variant),
    /// the weight itself otherwise (`stock`, or a weight without a copy).
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

    /// The live kernel over a 16- or 32-row `a` and the stored `[N, K]` weight
    /// `w`, FP32 or BF16 out: `DFlash2PackedWeights`' reference.
    static func storedLayoutProduct(_ a: MLXArray, _ w: MLXArray, outputDType: DType) -> MLXArray {
        let (k, n) = (w.dim(1), w.dim(0))
        return a.dim(0) > rowsPerTile
            ? launch32(a, w, k: k, n: n, swapped: swapped32Active, outputDType: outputDType)
            : launch(a, w, k: k, n: n, tiled: false, outputDType: outputDType)
    }

    /// True when `linear(layer, x)` runs every BF16 input of 1...16 rows as the
    /// one 16-row launch, the missing rows zero (`apply`'s padding).
    static func padsShortInputs(_ layer: Linear) -> Bool {
        let w = layer.weight
        return enabled && Qwen35TensorPackedMatmul.tensorOperandsAvailable && layer.bias == nil
            && w.dtype == .bfloat16 && w.ndim == 2 && w.dim(1) % 1024 == 0 && w.dim(0) % 32 == 0
    }

    /// `layer(x)` through the tensor kernel when it applies (no bias).
    static func linear(_ layer: Linear, _ x: MLXArray) -> MLXArray {
        if layer.bias == nil, let y = apply(x, weight: layer.weight) {
            return y
        }
        return layer(x)
    }
}

/// Lossless 12-bit resident copies of the weights the swapped kernel reads
/// (`DFlash2TensorMatmul.sourceSwapped`, `sourceSwapped32`), and that kernel
/// over them.
///
/// At 16 or 32 rows the block forward's projections are weight-stream
/// bound: the tensor op waits on the BF16 weight read, and the bytes are the
/// cost (on the M5 Pro every class streams at the device's read rate, and
/// the projections are ~95% of a block forward's GPU time). The weights use
/// few exponents: in every matrix the kernel reads, ~99.98% of the BF16
/// exponent fields are one of fifteen consecutive values (112-126 in most
/// tiles). Each weight is kept as its sign-and-mantissa byte plus a
/// four-bit code, its exponent's offset from its tile's base, 12 bits; code
/// 0 is an escape, and the weight's 16 bits then sit in its tile's escape
/// list. Each tile has its own base: a block of columns of small weights
/// would otherwise escape together and serialize its simdgroups on the
/// lookups. The kernel rebuilds every BF16 weight in registers, straight into
/// the cooperative left operand the swapped kernel `load`s, and runs the
/// same `32 x 16 x KT` op on it: a quarter fewer bytes for the same
/// products.
///
/// Layout: a tile is one simdgroup's `[32 columns, KT]` weight slice (column
/// block n0 / 32, K step k / KT), in the cooperative operand's own element
/// order: element i of lane l is byte i % 16 of the lane's word i / 16 (the
/// 32 lanes' words interleaved, so each read is contiguous across the
/// simdgroup), code i % 16 of its code word. The encoder fills the operand
/// from the stored weight with the operand's own `load` and writes each
/// lane's elements in that order, so the decoder refills the operand element
/// for element without any assumed index mapping. A tile's escapes are
/// `position << 16 | BF16 bits` words (position lane * KT + i), in position
/// order, from `offsets[tile]` to `offsets[tile + 1]`. KT = 64.
///
/// EXACT: every rebuilt weight is the stored BF16 weight, bit for bit, and
/// the op, the K partitions (`SPLITS`) and their reduction order are the
/// swapped kernel's, so the outputs are that kernel's bits. `prepare` checks
/// both at load on every packed weight: the rebuilt matrix against the
/// stored one, every element, and the products of random 16-row (and, for
/// the 32-row weights, 32-row) inputs against the live kernel's, FP32 and
/// BF16 outputs. On any mismatch or MLX error nothing is packed and every
/// weight keeps the stored layout. `MLXFAST_DRAFT_PACK12=0` keeps the stored
/// layout everywhere.
///
/// Tiling (`DFlash2TensorMatmul.SwapTrial`): the K step and the columns are
/// the tile's, fixed by the copy; the split and `AHEAD` (how many tiles ahead of its op a
/// simdgroup reads a tile's share; the record reads one) are template values.
/// A look-ahead moves only the reads, a split reorders the FP32 sum; a shape
/// takes another tiling only if the load-time trial checked and adopted it.
enum DFlash2PackedWeights {
    private static let setting: Bool = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DRAFT_PACK12"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(raw ?? "")
    }()

    /// A tile's columns: 16 by default, `[16 columns, 64 K]` tiles of 32
    /// weights per lane, twice the simdgroups of `[32, 64]` and half the
    /// registers per lane, so the decode's latency is hidden on every
    /// weight. `DARKBLOOM_DRAFT_PACK12_COLS=32` keeps the `[32, 64]` tiles.
    static let cols: Int = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_DRAFT_PACK12_COLS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return raw == "32" ? 32 : 16
    }()

    /// A tile's K step.
    static let ks = 64

    /// A lane's weights per tile, the cooperative operand's capacity
    /// (`cols * ks / 32`).
    static var kt: Int { cols * ks / 32 }

    private static var geometry: [(String, any KernelTemplateArg)] {
        [("KT", kt), ("KS", ks), ("COLS", cols)]
    }

    /// `DARKBLOOM_DRAFT_PACK12_ALL=1` packs every eligible weight, as the
    /// packed kernel's first form did.
    private static let packAll: Bool = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_DRAFT_PACK12_ALL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(raw ?? "")
    }()

    /// Whether the packed kernel is the faster read of an `[n, k]` weight.
    /// It reads a quarter fewer bytes but adds decode work to every tile,
    /// which only enough tiles in flight hide: a wide weight (many column
    /// blocks, as the gate|up stack) or a deep one (many K tiles per
    /// simdgroup, as down_proj and fc). A narrow, shallow weight waits on the
    /// decode instead. On an M5 Max, chained launches over the real weights
    /// (each launch a different layer's copy, so every read streams), packed
    /// against the stored layout: gate|up -20%, down_proj -3%, fc +1%, the
    /// q|k|v stack +2% (32 rows +1%), o_proj +7% (+8%), and the tap
    /// projections +63% (+64%). A weight left out keeps the swapped kernel
    /// over its stored layout, whose products the packed kernel matches bit
    /// for bit, so no value changes either way.
    ///
    /// The `[16, 64]` tiles hide it on every weight: on the same probe,
    /// packed against the stored layout at 16 rows, gate|up -20%, down_proj
    /// -20%, fc -19%, the q|k|v stack -17% (32 rows -17%), o_proj -17% and
    /// the tap projections -5% (32 rows -5%), so every eligible weight is
    /// packed.
    static func worthPacking(n: Int, k: Int) -> Bool {
        packAll || cols == 16 || n >= 16384 || k >= 16384
    }

    struct Copy {
        let source: MLXArray
        let mantissas: MLXArray
        let codes: MLXArray
        let first: MLXArray
        let bases: MLXArray
        let offsets: MLXArray
        let escapes: MLXArray
        let escapeCount: Int
        let index32Safe: Bool
        var arrays: [MLXArray] { [mantissas, codes, first, bases, offsets, escapes] }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var copies: [ObjectIdentifier: Copy] = [:]
    /// Set once by `prepare`, when every packed weight passed.
    nonisolated(unsafe) static var active = false

    /// The packed copy the block-width kernel reads for `weight`, if any.
    static func copy(of weight: MLXArray) -> Copy? {
        guard active else { return nil }
        return lock.withLock { copies[ObjectIdentifier(weight)] }.flatMap {
            $0.source === weight ? $0 : nil
        }
    }

    private static let header = """
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        // One lane's share of a packed tile, as read: its KT / 16 byte words
        // and code words, and the tile's first four escape words.
        template <int KT>
        struct dflash2_pack12_share {
          uint4 m[KT / 16];
          uint2 e[KT / 16];
          uint4 first;
          uint base;
        };

        template <int KT, typename IndexT = size_t>
        inline void dflash2_pack12_read(
            thread dflash2_pack12_share<KT>& t, const device uint4* mant, const device uint2* code,
            const device uint4* first4, const device uint* bases, int tile, uint lane) {
          constexpr int WORDS = KT / 16;
          const device uint4* mp = mant + (IndexT)tile * WORDS * 32 + lane;
          const device uint2* cp = code + (IndexT)tile * WORDS * 32 + lane;
          #pragma clang loop unroll(full)
          for (int j = 0; j < WORDS; j++) {
            t.m[j] = mp[j * 32];
            t.e[j] = cp[j * 32];
          }
          t.first = first4[tile];
          t.base = bases[tile];
        }

        // Lane `lane`'s KT weights of tile `tile` into the cooperative left
        // operand `lw`, from its share `t`. Two weights per 32-bit word:
        // element 2q from the low byte and code of its pair, 2q + 1 from the
        // high ones, each `sign | (tile base + code) << 7 | mantissa`; a zero
        // code (escape) takes the 16 bits of the tile's escape word at
        // position lane * KT + i: one of the tile's first four (read with the
        // share, so the common escape waits on no further read), else one
        // past them.
        template <int KT, typename LW>
        inline void dflash2_pack12_decode(
            thread LW& lw, thread const dflash2_pack12_share<KT>& t,
            const device uint* offsets, const device uint* escapes, int tile, uint lane) {
          constexpr int WORDS = KT / 16;
          const uint E2 = t.base | (t.base << 16);
          #pragma clang loop unroll(full)
          for (int j = 0; j < WORDS; j++) {
            const uint4 m = t.m[j];
            const uint2 e = t.e[j];
            uint w[8];
            #pragma clang loop unroll(full)
            for (int q = 0; q < 8; q++) {
              uint a = (m[q >> 1] >> (16 * (q & 1))) & 0xffffu;
              a = (a | (a << 8)) & 0x00ff00ffu;
              uint u = (e[q >> 2] >> (8 * (q & 3))) & 0xffu;
              u = (u | (u << 12)) & 0x000f000fu;
              w[q] = ((a & 0x00800080u) << 8) | (a & 0x007f007fu) | ((u + E2) << 7);
            }
            // Bit 3 of a nibble is set here iff that code is zero.
            const uint zx = ~(((e.x & 0x77777777u) + 0x77777777u) | e.x | 0x77777777u);
            const uint zy = ~(((e.y & 0x77777777u) + 0x77777777u) | e.y | 0x77777777u);
            if ((zx | zy) != 0u) {
              #pragma clang loop unroll(full)
              for (int b = 0; b < 16; b++) {
                if (((e[b >> 3] >> (4 * (b & 7))) & 0xfu) == 0u) {
                  const uint pos = lane * uint(KT) + uint(j * 16 + b);
                  uint bits = 0u;
                  if ((t.first.x >> 16) == pos) {
                    bits = t.first.x & 0xffffu;
                  } else if ((t.first.y >> 16) == pos) {
                    bits = t.first.y & 0xffffu;
                  } else if ((t.first.z >> 16) == pos) {
                    bits = t.first.z & 0xffffu;
                  } else if ((t.first.w >> 16) == pos) {
                    bits = t.first.w & 0xffffu;
                  } else {
                    const uint s1 = offsets[tile + 1];
                    for (uint s = offsets[tile] + 4u; s < s1; s++) {
                      const uint v = escapes[s];
                      if ((v >> 16) == pos) {
                        bits = v & 0xffffu;
                        break;
                      }
                    }
                  }
                  w[b >> 1] = (b & 1) ? ((w[b >> 1] & 0x0000ffffu) | (bits << 16))
                                      : ((w[b >> 1] & 0xffff0000u) | bits);
                }
              }
            }
            #pragma clang loop unroll(full)
            for (int q = 0; q < 8; q++) {
              const bfloat2 p = as_type<bfloat2>(w[q]);
              lw[j * 16 + 2 * q] = p.x;
              lw[j * 16 + 2 * q + 1] = p.y;
            }
          }
        }

        // `bits`' exponent code for base `base`: 1-15, or 0 (an escape).
        inline uint dflash2_pack12_code(uint bits, uint base) {
          const uint x = (bits >> 7) & 0xffu;
          return (x > base && x <= base + 15u) ? x - base : 0u;
        }

        """

    // The encoder's first pass. grid: (tiles * 32, 1, 1), threadgroup (32,
    // 1, 1): one simdgroup per tile. Inputs: w bfloat [N, K] (stored), ksz
    // int32 [K, 16, N]. Picks the tile's base (of the four whose window ends
    // zero to three below its largest exponent, the one with the fewest
    // escapes), writes the tile's bytes, codes, base and escape count (all
    // ones when the operand's capacity is not KT: no order to keep).
    private static let encodeSource = """
        const int K = ksz[0]; const int N = ksz[2];
        const uint lane = thread_index_in_simdgroup;
        const int steps = K / KS;
        const int tile = int(threadgroup_position_in_grid.x);
        const int n0 = (tile / steps) * COLS;
        const int k = (tile % steps) * KS;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            COLS, 16, KS, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> W((device bfloat*)w, dextents<int, 2>(K, N));
        auto lw = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        if (lw.get_capacity() != KT) {
          if (lane == 0) { count[tile] = 0xffffffffu; }
          return;
        }
        lw.load(W.template slice<KS, COLS>(k, n0));
        uint top = 0u;
        #pragma clang loop unroll(full)
        for (int i = 0; i < KT; i++) {
          top = max(top, (uint(as_type<ushort>(bfloat(lw[i]))) >> 7) & 0xffu);
        }
        top = simd_max(top);
        uint base = 0u;
        uint fewest = 0xffffffffu;
        for (uint below = 15u; below < 19u; below++) {
          const uint b = top >= below ? top - below : 0u;
          uint n = 0u;
          #pragma clang loop unroll(full)
          for (int i = 0; i < KT; i++) {
            n += dflash2_pack12_code(uint(as_type<ushort>(bfloat(lw[i]))), b) == 0u ? 1u : 0u;
          }
          n = simd_sum(n);
          if (n < fewest) {
            fewest = n;
            base = b;
          }
        }
        constexpr int WORDS = KT / 16;
        device uint4* mo = (device uint4*)mant + (size_t)tile * WORDS * 32 + lane;
        device uint2* co = (device uint2*)code + (size_t)tile * WORDS * 32 + lane;
        uint own = 0u;
        #pragma clang loop unroll(full)
        for (int j = 0; j < WORDS; j++) {
          uint4 m = uint4(0u);
          uint2 e = uint2(0u);
          #pragma clang loop unroll(full)
          for (int b = 0; b < 16; b++) {
            const uint bits = uint(as_type<ushort>(bfloat(lw[j * 16 + b])));
            const uint c = dflash2_pack12_code(bits, base);
            m[b >> 2] |= (((bits >> 8) & 0x80u) | (bits & 0x7fu)) << (8 * (b & 3));
            e[b >> 3] |= c << (4 * (b & 7));
            own += c == 0u ? 1u : 0u;
          }
          mo[j * 32] = m;
          co[j * 32] = e;
        }
        const uint total = simd_sum(own);
        if (lane == 0) {
          count[tile] = total;
          bases[tile] = base;
        }
        """

    // The encoder's second pass, the same grid: the tile's escapes at its
    // offset, in position order (lane order, then element order), and its
    // first four again in `first4`.
    private static let escapeSource = """
        const int K = ksz[0]; const int N = ksz[2];
        const uint lane = thread_index_in_simdgroup;
        const int steps = K / KS;
        const int tile = int(threadgroup_position_in_grid.x);
        const int n0 = (tile / steps) * COLS;
        const int k = (tile % steps) * KS;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            COLS, 16, KS, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> W((device bfloat*)w, dextents<int, 2>(K, N));
        auto lw = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        lw.load(W.template slice<KS, COLS>(k, n0));
        const uint base = bases[tile];
        uint own = 0u;
        #pragma clang loop unroll(full)
        for (int i = 0; i < KT; i++) {
          own += dflash2_pack12_code(uint(as_type<ushort>(bfloat(lw[i]))), base) == 0u ? 1u : 0u;
        }
        const uint start = offsets[tile];
        uint slot = simd_prefix_exclusive_sum(own);
        #pragma clang loop unroll(full)
        for (int i = 0; i < KT; i++) {
          const uint bits = uint(as_type<ushort>(bfloat(lw[i])));
          if (dflash2_pack12_code(bits, base) == 0u) {
            const uint v = ((lane * uint(KT) + uint(i)) << 16) | bits;
            escapes[start + slot] = v;
            if (slot < 4u) { first4[(size_t)tile * 4 + slot] = v; }
            slot += 1u;
          }
        }
        // Unused first words: all ones (no position matches).
        if (lane >= offsets[tile + 1] - start && lane < 4u) {
          first4[(size_t)tile * 4 + lane] = 0xffffffffu;
        }
        """

    // The self-test's rebuild: every tile decoded and stored back into
    // `[N, K]` through the operand's own `store`.
    private static let unpackSource = """
        const int K = ksz[0]; const int N = ksz[2];
        const uint lane = thread_index_in_simdgroup;
        const int steps = K / KS;
        const int tile = int(threadgroup_position_in_grid.x);
        const int n0 = (tile / steps) * COLS;
        const int k = (tile % steps) * KS;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            COLS, 16, KS, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> O((device bfloat*)out, dextents<int, 2>(K, N));
        auto lw = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        dflash2_pack12_share<KT> t;
        dflash2_pack12_read<KT>(
            t, (const device uint4*)mant, (const device uint2*)code, (const device uint4*)first4, bases,
            tile, lane);
        dflash2_pack12_decode<KT>(lw, t, offsets, escapes, tile, lane);
        lw.store(O.template slice<KS, COLS>(k, n0));
        """

    // `sourceSwapped` with the operand rebuilt from the packed tile instead
    // of loaded: the same op, K partitions and reduction. grid: (N / 32 *
    // (32 * SPLITS), 1, 1), threadgroup (32 * SPLITS, 1, 1). x bfloat [16, K].
    private static let source = """
        using IndexT = metal::conditional_t<IO32 != 0, uint, size_t>;
        const int K = ksz[0]; const int N = ksz[2];
        const int nb = int(threadgroup_position_in_grid.x);
        const int n0 = nb * COLS;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int kq = K / SPLITS;
        const int k0 = int(sg) * kq;
        const int steps = K / KS;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            COLS, 16, KS, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        // `Wt` only names the operand type the destination is built for.
        tensor<device bfloat, dextents<int, 2>, tensor_inline> Wt((device bfloat*)x, dextents<int, 2>(K, N));
        tensor<device bfloat, dextents<int, 2>, tensor_inline> X((device bfloat*)x, dextents<int, 2>(K, 16));
        auto tW0 = Wt.template slice<KS, COLS>(0, n0);
        auto tX0 = X.template slice<KS, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tW0)>, metal::remove_addrspace_t<decltype(tX0)>, float>();
        auto lw = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        const uint16_t cap = cT.get_capacity();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); i++) { cT[i] = 0.0f; }
        // A tile's share is read AHEAD tiles before its op (the record: one;
        // 0: just before its decode). The ops are the same for every AHEAD.
        const device uint4* mp = (const device uint4*)mant;
        const device uint2* cp = (const device uint2*)code;
        const device uint4* fp = (const device uint4*)first4;
        dflash2_pack12_share<KT> cur, nxt, far;
        if constexpr (AHEAD > 0) {
          dflash2_pack12_read<KT, IndexT>(cur, mp, cp, fp, bases, nb * steps + k0 / KS, lane);
        }
        if constexpr (AHEAD > 1) {
          if (KS < kq) { dflash2_pack12_read<KT, IndexT>(nxt, mp, cp, fp, bases, nb * steps + k0 / KS + 1, lane); }
        }
        for (int k = k0; k < k0 + kq; k += KS) {
          const int tile = nb * steps + k / KS;
          if constexpr (AHEAD == 0) {
            dflash2_pack12_read<KT, IndexT>(cur, mp, cp, fp, bases, tile, lane);
          } else if constexpr (AHEAD == 1) {
            if (k + KS < k0 + kq) { dflash2_pack12_read<KT, IndexT>(nxt, mp, cp, fp, bases, tile + 1, lane); }
          } else {
            if (k + 2 * KS < k0 + kq) { dflash2_pack12_read<KT, IndexT>(far, mp, cp, fp, bases, tile + 2, lane); }
          }
          dflash2_pack12_decode<KT>(lw, cur, offsets, escapes, tile, lane);
          auto tX = X.template slice<KS, 16>(k, 0);
          op.run(lw, tX, cT);
          if constexpr (AHEAD > 0) { cur = nxt; }
          if constexpr (AHEAD > 1) { nxt = far; }
        }
        threadgroup float red[SPLITS - 1][16 * COLS];
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
            } else if constexpr (SPLITS == 4) {
              v = cT[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
            } else {
              v = cT[i];
              for (int j = 0; j < SPLITS - 1; j++) { v += red[j][i * 32 + lane]; }
            }
            auto idx = cT.get_multidimensional_index(i);
            out[(IndexT)idx[0] * N + n0 + idx[1]] = OutT(v);
          }
        }
        """

    // `sourceSwapped32` the same way: one rebuilt operand per K step,
    // multiplied by rows 0-15 and 16-31. x bfloat [32, K].
    private static let source32 = """
        using IndexT = metal::conditional_t<IO32 != 0, uint, size_t>;
        const int K = ksz[0]; const int N = ksz[2];
        const int nb = int(threadgroup_position_in_grid.x);
        const int n0 = nb * COLS;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int kq = K / SPLITS;
        const int k0 = int(sg) * kq;
        const int steps = K / KS;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            COLS, 16, KS, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> Wt((device bfloat*)x, dextents<int, 2>(K, N));
        tensor<device bfloat, dextents<int, 2>, tensor_inline> X((device bfloat*)x, dextents<int, 2>(K, 32));
        auto tW0 = Wt.template slice<KS, COLS>(0, n0);
        auto tX0 = X.template slice<KS, 16>(0, 0);
        auto cT0 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tW0)>, metal::remove_addrspace_t<decltype(tX0)>, float>();
        auto cT1 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tW0)>, metal::remove_addrspace_t<decltype(tX0)>, float>();
        auto lw = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        const uint16_t cap = cT0.get_capacity();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT0.get_capacity(); i++) { cT0[i] = 0.0f; cT1[i] = 0.0f; }
        const device uint4* mp = (const device uint4*)mant;
        const device uint2* cp = (const device uint2*)code;
        const device uint4* fp = (const device uint4*)first4;
        dflash2_pack12_share<KT> cur, nxt, far;
        if constexpr (AHEAD > 0) {
          dflash2_pack12_read<KT, IndexT>(cur, mp, cp, fp, bases, nb * steps + k0 / KS, lane);
        }
        if constexpr (AHEAD > 1) {
          if (KS < kq) { dflash2_pack12_read<KT, IndexT>(nxt, mp, cp, fp, bases, nb * steps + k0 / KS + 1, lane); }
        }
        for (int k = k0; k < k0 + kq; k += KS) {
          const int tile = nb * steps + k / KS;
          if constexpr (AHEAD == 0) {
            dflash2_pack12_read<KT, IndexT>(cur, mp, cp, fp, bases, tile, lane);
          } else if constexpr (AHEAD == 1) {
            if (k + KS < k0 + kq) { dflash2_pack12_read<KT, IndexT>(nxt, mp, cp, fp, bases, tile + 1, lane); }
          } else {
            if (k + 2 * KS < k0 + kq) { dflash2_pack12_read<KT, IndexT>(far, mp, cp, fp, bases, tile + 2, lane); }
          }
          dflash2_pack12_decode<KT>(lw, cur, offsets, escapes, tile, lane);
          if constexpr (AHEAD > 0) { cur = nxt; }
          if constexpr (AHEAD > 1) { nxt = far; }
          auto tXlo = X.template slice<KS, 16>(k, 0);
          op.run(lw, tXlo, cT0);
          auto tXhi = X.template slice<KS, 16>(k, 16);
          op.run(lw, tXhi, cT1);
        }
        threadgroup float red[SPLITS - 1][2 * 16 * COLS];
        if (sg > 0) {
          for (uint16_t i = 0; i < cap; i++) {
            red[sg - 1][i * 32 + lane] = cT0[i];
            red[sg - 1][(COLS / 2 + i) * 32 + lane] = cT1[i];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          for (uint16_t i = 0; i < cap; i++) {
            if (!cT0.is_valid_element(i)) continue;
            float v0, v1;
            if constexpr (SPLITS == 2) {
              v0 = cT0[i] + red[0][i * 32 + lane];
              v1 = cT1[i] + red[0][(COLS / 2 + i) * 32 + lane];
            } else if constexpr (SPLITS == 4) {
              v0 = cT0[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
              v1 = cT1[i] + red[0][(COLS / 2 + i) * 32 + lane] + red[1][(COLS / 2 + i) * 32 + lane]
                  + red[2][(COLS / 2 + i) * 32 + lane];
            } else {
              v0 = cT0[i]; v1 = cT1[i];
              for (int j = 0; j < SPLITS - 1; j++) {
                v0 += red[j][i * 32 + lane]; v1 += red[j][(COLS / 2 + i) * 32 + lane];
              }
            }
            auto idx = cT0.get_multidimensional_index(i);
            out[(IndexT)idx[0] * N + n0 + idx[1]] = OutT(v0);
            out[(IndexT)(16 + idx[0]) * N + n0 + idx[1]] = OutT(v1);
          }
        }
        """

    /// Speculative Q only: preserve all 32 K/V rows, calculate the 16 Q rows
    /// the device confirmed count selects. The normal m32 source stays intact.
    private static let queryWindowSource = """
        using IndexT = metal::conditional_t<IO32 != 0, uint, size_t>;
        const int K = ksz[0]; const int N = ksz[2];
        const int nb = int(threadgroup_position_in_grid.x);
        const int n0 = nb * COLS;
        // Uniform per threadgroup: Q needs only the block's 16 query rows.
        static_assert(QCOLS % COLS == 0, "Q/KV boundary must be a whole tile");
        const bool query = n0 < QCOLS;
        const int firstRow = query ? clamp(int(confirmed[0]), 0, 16) : 0;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int kq = K / SPLITS;
        const int k0 = int(sg) * kq;
        const int steps = K / KS;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            COLS, 16, KS, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device bfloat, dextents<int, 2>, tensor_inline> Wt((device bfloat*)x, dextents<int, 2>(K, N));
        tensor<device bfloat, dextents<int, 2>, tensor_inline> X((device bfloat*)x, dextents<int, 2>(K, 32));
        auto tW0 = Wt.template slice<KS, COLS>(0, n0);
        auto tX0 = X.template slice<KS, 16>(0, 0);
        auto cT0 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tW0)>, metal::remove_addrspace_t<decltype(tX0)>, float>();
        auto cT1 = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tW0)>, metal::remove_addrspace_t<decltype(tX0)>, float>();
        auto lw = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
        const uint16_t cap = cT0.get_capacity();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT0.get_capacity(); i++) { cT0[i] = 0.0f; cT1[i] = 0.0f; }
        const device uint4* mp = (const device uint4*)mant;
        const device uint2* cp = (const device uint2*)code;
        const device uint4* fp = (const device uint4*)first4;
        dflash2_pack12_share<KT> cur, nxt, far;
        if constexpr (AHEAD > 0) {
          dflash2_pack12_read<KT, IndexT>(cur, mp, cp, fp, bases, nb * steps + k0 / KS, lane);
        }
        if constexpr (AHEAD > 1) {
          if (KS < kq) { dflash2_pack12_read<KT, IndexT>(nxt, mp, cp, fp, bases, nb * steps + k0 / KS + 1, lane); }
        }
        for (int k = k0; k < k0 + kq; k += KS) {
          const int tile = nb * steps + k / KS;
          if constexpr (AHEAD == 0) {
            dflash2_pack12_read<KT, IndexT>(cur, mp, cp, fp, bases, tile, lane);
          } else if constexpr (AHEAD == 1) {
            if (k + KS < k0 + kq) { dflash2_pack12_read<KT, IndexT>(nxt, mp, cp, fp, bases, tile + 1, lane); }
          } else {
            if (k + 2 * KS < k0 + kq) { dflash2_pack12_read<KT, IndexT>(far, mp, cp, fp, bases, tile + 2, lane); }
          }
          dflash2_pack12_decode<KT>(lw, cur, offsets, escapes, tile, lane);
          if constexpr (AHEAD > 0) { cur = nxt; }
          if constexpr (AHEAD > 1) { nxt = far; }
          auto tXlo = X.template slice<KS, 16>(k, firstRow);
          op.run(lw, tXlo, cT0);
          if (!query) {
            auto tXhi = X.template slice<KS, 16>(k, 16);
            op.run(lw, tXhi, cT1);
          }
        }
        threadgroup float red[SPLITS - 1][2 * 16 * COLS];
        if (sg > 0) {
          for (uint16_t i = 0; i < cap; i++) {
            red[sg - 1][i * 32 + lane] = cT0[i];
            if (!query) red[sg - 1][(COLS / 2 + i) * 32 + lane] = cT1[i];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          for (uint16_t i = 0; i < cap; i++) {
            if (!cT0.is_valid_element(i)) continue;
            float v0, v1 = 0.0f;
            if constexpr (SPLITS == 2) {
              v0 = cT0[i] + red[0][i * 32 + lane];
              if (!query) v1 = cT1[i] + red[0][(COLS / 2 + i) * 32 + lane];
            } else if constexpr (SPLITS == 4) {
              v0 = cT0[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
              if (!query) v1 = cT1[i] + red[0][(COLS / 2 + i) * 32 + lane] + red[1][(COLS / 2 + i) * 32 + lane]
                  + red[2][(COLS / 2 + i) * 32 + lane];
            } else {
              v0 = cT0[i]; if (!query) v1 = cT1[i];
              for (int j = 0; j < SPLITS - 1; j++) {
                v0 += red[j][i * 32 + lane]; if (!query) v1 += red[j][(COLS / 2 + i) * 32 + lane];
              }
            }
            auto idx = cT0.get_multidimensional_index(i);
            if (query) {
              // Exactly one store per output: real Q rows and an unused complement.
              const int r = int(idx[0]);
              const int unused = r < firstRow ? r : r + 16;
              out[(IndexT)(firstRow + r) * N + n0 + idx[1]] = OutT(v0);
              out[(IndexT)unused * N + n0 + idx[1]] = OutT(0.0f);
            } else {
              out[(IndexT)idx[0] * N + n0 + idx[1]] = OutT(v0);
              out[(IndexT)(16 + idx[0]) * N + n0 + idx[1]] = OutT(v1);
            }
          }
        }
        """

    private static let queryWindowKernel = MLXFast.metalKernel(
        name: "dflash2_pack12_query_window_m32",
        inputNames: ["x", "mant", "code", "first4", "bases", "offsets", "escapes", "ksz", "confirmed"],
        outputNames: ["out"], source: queryWindowSource, header: header, ensureRowContiguous: true)

    static let queryWindowEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["BONSAI_DFLASH_QUERY_WINDOW"]?.lowercased()
        return !["0", "false", "no", "off"].contains(raw ?? "")
    }()

    struct QueryWindowChoice {
        let source: MLXArray
        let qColumns: Int
        let tiling: DFlash2TensorMatmul.SwapTiling
    }

    private static func queryWindowLaunch(
        _ a: MLXArray, _ c: Copy, confirmed: MLXArray, qColumns: Int,
        tiling t: DFlash2TensorMatmul.SwapTiling
    ) -> MLXArray {
        let n = c.source.dim(0)
        return queryWindowKernel(
            [a] + c.arrays + [dims(c.source), confirmed.reshaped([1])],
            template: [("OutT", DType.bfloat16), ("SPLITS", t.splits),
                ("IO32", c.index32Safe ? 1 : 0), ("QCOLS", qColumns)]
                + geometry + [("AHEAD", t.ahead ?? 1)],
            grid: (n / cols * t.splits * 32, 1, 1), threadGroup: (t.splits * 32, 1, 1),
            outputShapes: [[32, n]], outputDTypes: [.bfloat16])[0]
    }

    static func applyQueryWindow(
        _ x: MLXArray, weight: MLXArray, confirmed: MLXArray, choice: QueryWindowChoice
    ) -> MLXArray? {
        guard queryWindowEnabled, DFlash2TensorMatmul.packed32Available,
            choice.source === weight, x.dtype == .bfloat16, weight.dtype == .bfloat16,
            x.ndim == 3, x.dim(0) == 1, x.dim(1) == 32, x.dim(2) == weight.dim(1),
            confirmed.size == 1, confirmed.dtype == .int32, let c = copy(of: weight),
            DFlash2TensorMatmul.swapTiling(k: weight.dim(1), n: weight.dim(0), rows32: true, packed: true)
                == choice.tiling
        else { return nil }
        return queryWindowLaunch(x.reshaped(32, -1), c, confirmed: confirmed,
            qColumns: choice.qColumns, tiling: choice.tiling).reshaped(1, 32, weight.dim(0))
    }

    /// Remote startup safeguard on each actual stacked weight, after its
    /// tiling is settled. Nothing here uses a prompt, expected token or tape.
    /// Compare the consumed Q and all K/V bits for every confirmed count.
    /// Retain the original launch on errors, any mismatch or a timing tie.
    static func prepareQueryWindow(weight: MLXArray, qColumns q: Int) -> QueryWindowChoice? {
        guard queryWindowEnabled, DFlash2TensorMatmul.packed32Available,
            weight.ndim == 2, weight.dtype == .bfloat16, q > 0, q < weight.dim(0),
            q % cols == 0, weight.dim(0) % 32 == 0, weight.dim(1) % 1024 == 0,
            let c = copy(of: weight)
        else { return nil }
        let (k, n) = (weight.dim(1), weight.dim(0))
        let t = DFlash2TensorMatmul.swapTiling(k: k, n: n, rows32: true, packed: true)
        guard [2, 4, 8].contains(t.splits), t.fits(k: k), (0 ... 2).contains(t.ahead ?? 1)
        else { return nil }
        var same = true
        let warm = MLXRandom.normal([32, k], key: MLXRandom.key(0x715eed)).asType(.bfloat16)
        do {
            try withError { error in
                for pattern in 0 ..< 3 {
                    let a: MLXArray
                    if pattern == 0 {
                        a = warm
                    } else if pattern == 1 {
                        a = (MLXRandom.normal([32, k], key: MLXRandom.key(0x716eed)) * 8)
                            .asType(.bfloat16)
                    } else {
                        let zeroRows = MLXArray(0 ..< Int32(32)).reshaped(32, 1) % 2
                        a = broadcast(
                            MLX.where(zeroRows .== 0, MLXArray(Float(-0.0)), MLXArray(Float(0.0))),
                            to: [32, k]).asType(.bfloat16)
                    }
                    let stock = launch(a, c, rows: 32, outputDType: .bfloat16, tiling: t)
                    eval(a, stock)
                    for count in 1 ... 16 {
                        let fast = queryWindowLaunch(a, c, confirmed: MLXArray([Int32(count)]),
                            qColumns: q, tiling: t)
                        let sq = stock[count ..< (count + 16), ..<q].view(dtype: .uint16)
                        let fq = fast[count ..< (count + 16), ..<q].view(dtype: .uint16)
                        same = same && all(sq .== fq).item(Bool.self)
                            && all(stock[0..., q...].view(dtype: .uint16)
                                .== fast[0..., q...].view(dtype: .uint16)).item(Bool.self)
                        if !same { return }
                    }
                }
                try error.check()
            }
        } catch { same = false }
        guard same else {
            FileHandle.standardError.write("dflash2 query window: bit check failed; stock kept\n".data(using: .utf8)!)
            return nil
        }
        eval(warm)
        let counts = [1, 8, 16].map { MLXArray([Int32($0)]) }
        let builders: [() -> [MLXArray]] = [
            { counts.map { _ in launch(warm, c, rows: 32, outputDType: .bfloat16, tiling: t) } },
            { counts.map { queryWindowLaunch(warm, c, confirmed: $0, qColumns: q, tiling: t) } },
        ]
        let times = DFlash2LaunchTrial.race(builders, copies: 8, samples: 11)
        guard times.count == 2, times.allSatisfy({ $0.isFinite && $0 > 0 }), times[1] < times[0] * 0.98
        else { return nil }
        // Reverse order in an independent confirmation, so fixed-order bias
        // or one transient startup fluctuation cannot select this kernel.
        let confirm = DFlash2LaunchTrial.race(Array(builders.reversed()), copies: 8, samples: 11)
        guard confirm.count == 2, confirm.allSatisfy({ $0.isFinite && $0 > 0 }), confirm[0] < confirm[1] * 0.98
        else { return nil }
        FileHandle.standardError.write(
            ("dflash2 query window: consumed bits passed for all 16 counts; 32->16 Q rows adopted\n")
                .data(using: .utf8)!)
        return QueryWindowChoice(source: weight, qColumns: q, tiling: t)
    }

    private static let encodeKernel = MLXFast.metalKernel(
        name: "dflash2_pack12_encode", inputNames: ["w", "ksz"],
        outputNames: ["mant", "code", "bases", "count"], source: encodeSource, header: header,
        ensureRowContiguous: true)

    private static let escapeKernel = MLXFast.metalKernel(
        name: "dflash2_pack12_escapes", inputNames: ["w", "bases", "offsets", "ksz"],
        outputNames: ["escapes", "first4"], source: escapeSource, header: header,
        ensureRowContiguous: true)

    private static let unpackKernel = MLXFast.metalKernel(
        name: "dflash2_pack12_unpack",
        inputNames: ["mant", "code", "first4", "bases", "offsets", "escapes", "ksz"],
        outputNames: ["out"], source: unpackSource, header: header, ensureRowContiguous: true)

    private static let kernel = MLXFast.metalKernel(
        name: "dflash2_pack12_matmul_m16",
        inputNames: ["x", "mant", "code", "first4", "bases", "offsets", "escapes", "ksz"],
        outputNames: ["out"], source: source, header: header, ensureRowContiguous: true)

    private static let kernel32 = MLXFast.metalKernel(
        name: "dflash2_pack12_matmul_m32",
        inputNames: ["x", "mant", "code", "first4", "bases", "offsets", "escapes", "ksz"],
        outputNames: ["out"], source: source32, header: header, ensureRowContiguous: true)

    private static func dims(_ w: MLXArray) -> MLXArray {
        DFlash2TensorMatmul.dimsArray(k: w.dim(1), n: w.dim(0))
    }

    /// The packed product over a 16-row `a` (`[16, K]`), or nil when
    /// `weight` has no packed copy.
    static func apply(_ a: MLXArray, _ weight: MLXArray, outputDType: DType) -> MLXArray? {
        guard let c = copy(of: weight) else { return nil }
        return launch(a, c, rows: 16, outputDType: outputDType)
    }

    /// The packed product over a 32-row `a` (`[32, K]`), or nil.
    static func apply32(_ a: MLXArray, _ weight: MLXArray, outputDType: DType) -> MLXArray? {
        guard let c = copy(of: weight) else { return nil }
        return launch(a, c, rows: 32, outputDType: outputDType)
    }

    /// The swapped kernel's grid (`launch`, `launch32`) over the copy, with
    /// `tiling` or the shape's (`DFlash2TensorMatmul.swapTiling`: the stock
    /// split and one tile ahead unless a forced choice or its trial set another).
    fileprivate static func launch(
        _ a: MLXArray, _ c: Copy, rows: Int, outputDType: DType, tiling: DFlash2TensorMatmul.SwapTiling? = nil
    ) -> MLXArray {
        let n = c.source.dim(0)
        let t = tiling ?? DFlash2TensorMatmul.swapTiling(k: c.source.dim(1), n: n, rows32: rows > 16, packed: true)
        return (rows == 16 ? kernel : kernel32)(
            [a] + c.arrays + [dims(c.source)],
            template: [("OutT", outputDType), ("SPLITS", t.splits), ("IO32", c.index32Safe ? 1 : 0)] + geometry + [("AHEAD", t.ahead ?? 1)],
            grid: (n / cols * t.splits * 32, 1, 1), threadGroup: (t.splits * 32, 1, 1),
            outputShapes: [[rows, n]], outputDTypes: [outputDType])[0]
    }

    /// `w` packed, or nil when the operand's capacity is not KT (the
    /// first pass flags every tile).
    private static func pack(_ w: MLXArray) -> Copy? {
        let tiles = w.dim(0) / cols * (w.dim(1) / ks)
        let first = encodeKernel(
            [w, dims(w)], template: geometry,
            grid: (tiles * 32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[tiles * kt * 8], [tiles * kt * 4], [tiles], [tiles]],
            outputDTypes: [.uint32, .uint32, .uint32, .uint32])
        let counts = first[3]
        let largest = counts.max()
        let total = counts.asType(.int64).sum()
        let offsets = concatenated(
            [MLXArray.zeros([1], dtype: .uint32), cumsum(counts, axis: 0).asType(.uint32)], axis: 0)
        eval(first + [largest, total, offsets])
        guard largest.item(UInt32.self) <= UInt32(32 * kt) else { return nil }
        let escapeCount = total.item(Int.self)
        let written = escapeKernel(
            [w, first[2], offsets, dims(w)], template: geometry,
            grid: (tiles * 32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[max(escapeCount, 1)], [tiles * 4]], outputDTypes: [.uint32, .uint32])
        eval(written)
        let limit = Int(Int32.max)
        let arrays = [first[0], first[1], written[1], first[2], offsets, written[0]]
        let index32Safe = w.size > 0 && w.size <= limit
            && w.dim(0) <= limit / 32 && w.dim(1) <= limit / 32
            && arrays.allSatisfy { $0.size > 0 && $0.size <= limit }
        return Copy(
            source: w, mantissas: first[0], codes: first[1], first: written[1], bases: first[2],
            offsets: offsets, escapes: written[0], escapeCount: escapeCount, index32Safe: index32Safe)
    }

    /// Packs every eligible one of `weights` (16-row products) and
    /// `weights32` (16- and 32-row products), self-tests every packed weight
    /// against its stored bits and the live kernel's products, and turns the
    /// packed kernel on only when everything matched. One stderr line.
    /// Nothing runs unless the swapped kernel is on.
    static func prepare(_ weights: [MLXArray], rows32 weights32: [MLXArray]) {
        active = false
        lock.withLock { copies.removeAll() }
        guard setting, DFlash2TensorMatmul.swappedActive else { return }
        let start = DispatchTime.now().uptimeNanoseconds
        // The rebuilt matrices are one-off sizes: keep them out of the buffer
        // cache the served phases allocate from.
        let cacheLimit = Memory.cacheLimit
        Memory.cacheLimit = 0
        defer { Memory.cacheLimit = cacheLimit }
        let rows32 = Set(weights32.map { ObjectIdentifier($0) })
        var seen = Set<ObjectIdentifier>()
        let eligible = (weights + weights32).filter {
            guard $0.dtype == .bfloat16, $0.ndim == 2, $0.dim(0) % 32 == 0 else { return false }
            let splits = DFlash2TensorMatmul.Kernel.stockSplits(n: $0.dim(0))
            return $0.dim(1) % (splits * ks) == 0 && worthPacking(n: $0.dim(0), k: $0.dim(1))
                && seen.insert(ObjectIdentifier($0)).inserted
        }
        var packed: [Copy] = []
        var escapes = 0
        var bytes = (stored: 0, packed: 0)
        var rebuilt = 0
        var products = 0
        var mismatches = 0
        var passed = false
        do {
            try withError { error in
                var inputs: [[Int]: MLXArray] = [:]
                for (index, w) in eligible.enumerated() {
                    guard let c = pack(w) else {
                        mismatches += 1
                        break
                    }
                    let (n, k) = (w.dim(0), w.dim(1))
                    // Every weight rebuilt, against its stored bits.
                    let tiles = n / cols * (k / ks)
                    let unpacked = unpackKernel(
                        c.arrays + [dims(w)], template: geometry,
                        grid: (tiles * 32, 1, 1), threadGroup: (32, 1, 1),
                        outputShapes: [[n, k]], outputDTypes: [.bfloat16])[0]
                    var differing = [
                        (unpacked.view(dtype: .uint16) .!= w.view(dtype: .uint16)).asType(.int32).sum()
                    ]
                    rebuilt += n * k
                    // The products, against the live kernel's.
                    for rows in rows32.contains(ObjectIdentifier(w)) ? [16, 32] : [16] {
                        let a = inputs[[rows, k]] ?? MLXRandom.normal(
                            [rows, k], key: MLXRandom.key(UInt64(7717 + index))
                        ).asType(.bfloat16)
                        inputs[[rows, k]] = a
                        for (outputDType, bits) in [(DType.float32, DType.uint32), (.bfloat16, .uint16)] {
                            let live = DFlash2TensorMatmul.storedLayoutProduct(a, w, outputDType: outputDType)
                            let mine = launch(a, c, rows: rows, outputDType: outputDType)
                            differing.append(
                                (live.view(dtype: bits) .!= mine.view(dtype: bits)).asType(.int32).sum())
                            products += rows * n
                        }
                    }
                    let wrong = stacked(differing).sum()
                    eval(wrong)
                    mismatches += wrong.item(Int.self)
                    packed.append(c)
                    escapes += c.escapeCount
                    bytes.stored += w.nbytes
                    bytes.packed += c.arrays.reduce(0) { $0 + $1.nbytes }
                }
                try error.check()
            }
            passed = mismatches == 0 && !packed.isEmpty
        } catch {
            passed = false
        }
        if passed {
            lock.withLock {
                for c in packed { copies[ObjectIdentifier(c.source)] = c }
            }
            active = true
        }
        packed = []
        Memory.clearCache()
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        FileHandle.standardError.write(
            Data(
                ("dflash2 packed 12-bit weights: self-test \(passed ? "passed" : "FAILED") "
                    + "(\(eligible.count) weights, [\(cols), \(ks)] tiles; \(rebuilt) values rebuilt bit for bit, "
                    + "\(escapes) escapes; "
                    + "\(products) products bitwise, \(mismatches) mismatches); "
                    + (passed
                        ? String(format: "on, %.2f GB read as %.2f GB", Double(bytes.stored) / 1e9,
                            Double(bytes.packed) / 1e9)
                        : "stored layout kept")
                    + String(format: "; %.0f ms\n", elapsed)).utf8))
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
            read += DFlash2PackedWeights.copy(of: stacked)?.arrays
                ?? [DFlash2TensorMatmul.readWeight(stacked)]
            replaced += [gate.weight, up.weight]
        }
        if let packed = DFlash2PackedWeights.copy(of: down.weight) {
            read += packed.arrays
            replaced.append(down.weight)
            return (read, replaced)
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
        // The tap projections' packed copies stand in for them.
        let packed = projectionWeights().compactMap { w in
            DFlash2PackedWeights.copy(of: w).map { (w, $0.arrays) }
        }
        let replaced = attention.replaced + feedForward.replaced + packed.map(\.0)
        let skipped = Set(replaced.map { ObjectIdentifier($0) })
        let rest = parameters().flattened().map(\.1).filter {
            !skipped.contains(ObjectIdentifier($0))
        }
        return (attention.read + feedForward.read + packed.flatMap(\.1) + rest, replaced)
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
    /// With `context` (the rows `base` starts with) and `pos`, the front
    /// takes `DFlash2SpeculativeFront`'s launches where they apply.
    func speculative(
        _ x: MLXArray, base: MLXArray, context: MLXArray? = nil, confirmed: MLXArray,
        queryOffset: MLXArray, pos: MLXArray? = nil, rope: RoPELayer,
        cache: DFlash2BlockKVCache, keyMask: MLXArray
    ) -> (hidden: MLXArray, keys: MLXArray, values: MLXArray)? {
        let normed = inputLayerNorm(x)
        var attentionInput = normed
        let attentionTaps: MLXArray
        var joined: MLXArray?
        if let context,
            let front = attentionConv.prepareSpeculative(
                normed, context: context, confirmed: confirmed, rows: base.dim(1))
        {
            (attentionTaps, joined) = (front.projection, front.rows)
        } else {
            (attentionInput, attentionTaps) = attentionConv.prepare(normed)
        }
        guard
            let a = selfAttn.speculative(
                attentionInput, joined: joined, base: base, confirmed: confirmed,
                queryOffset: queryOffset, pos: pos, rope: rope, cache: cache, keyMask: keyMask)
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
            let w = DFlash2GreedyWalk.edgeWeight
            let selected = (unary[0..., position, 0...] + (w == 1 ? edges : MLXArray(w) * edges))
                .argMax(axis: -1)
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
        let parts = (vector && k % (threads / 32) == 0 && thresholdActive ? thresholdChunkKernel : chunkKernel)(
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
        source: Qwen35IO32.narrow("""
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
            """, count: 3, "mlxfast_dflash_topk_chunk"),
        header: header)

    /// `MLXFAST_DFLASH_TOPK_THRESHOLD=0` keeps the one-pass chunk scan.
    static let thresholdSetting: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_TOPK_THRESHOLD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Whether the chunk pass takes `thresholdChunkKernel`: set once by
    /// `prepareThreshold` (at bind) when its lists matched the stock pass's.
    nonisolated(unsafe) static var thresholdActive = false
    nonisolated(unsafe) private static var thresholdChecked = false

    /// Runs `select` with the one-pass and the two-pass chunk kernels on the
    /// same logits (FP16 and FP32; random rows, rows with a few large values,
    /// rows of heavy ties, NaN and signed zeros) and compares every candidate
    /// id and every gathered value bit for bit; the two-pass scan is used only
    /// when all match. One stderr line.
    static func prepareThreshold(vocabularySize: Int, k: Int) {
        guard !thresholdChecked else { return }
        thresholdChecked = true
        guard enabled, thresholdSetting, vocabularySize % (chunks * 4) == 0, k % (threads / 32) == 0
        else { return }
        var same = true
        var values = 0
        do {
            try withError { error in
                for (seed, dtype) in [(0, DType.float16), (1, .float16), (2, .float32), (3, .float16)] {
                    let rows = 16
                    var logits = MLXRandom.normal([1, rows, vocabularySize], key: MLXRandom.key(UInt64(0x70c + seed))) * 3
                    let ids = MLXArray(0 ..< Int32(vocabularySize)).reshaped([1, 1, vocabularySize])
                    let row = MLXArray(0 ..< Int32(rows)).reshaped([1, rows, 1])
                    // heavy ties on every fourth row, a few large values elsewhere
                    logits = MLX.where((row % 4) .== 1, (ids % 5).asType(.float32), logits)
                    logits = MLX.where(((ids * 7 + row * 131) % 2503) .== 0, logits + 12, logits)
                    if seed == 3 {
                        logits = MLX.where(ids .== 777, MLXArray(Float.nan), logits)
                        logits = MLX.where(ids .== 99999 % vocabularySize, MLXArray(Float(-0.0)), logits)
                    }
                    let input = logits.asType(dtype)
                    thresholdActive = false
                    guard let stock = select(input, k: k) else { same = false; return }
                    thresholdActive = true
                    guard let fast = select(input, k: k) else { same = false; return }
                    thresholdActive = false
                    same = same && all(stock.0 .== fast.0).item(Bool.self)
                        && all(stock.1.view(dtype: .uint32) .== fast.1.view(dtype: .uint32)).item(Bool.self)
                    values += rows * k
                }
                try error.check()
            }
        } catch {
            same = false
        }
        thresholdActive = same
        FileHandle.standardError.write(
            ("dflash2 top-k threshold scan: "
                + (same ? "self-test passed: \(values) candidates and values identical; on\n"
                    : "self-test failed; one-pass scan kept\n")).data(using: .utf8)!)
    }

    /// `chunkKernel` in two passes. Pass 1 bounds the chunk's KK-th largest
    /// key from below: each of the NSG simdgroups takes the (KK / NSG)-th
    /// largest of its lanes' maxima, and T is the smallest of those, so at
    /// least KK chunk elements have a key >= T and no element below T can rank
    /// in the chunk's top KK. Pass 2 is the stock scan (same threads, same
    /// increasing indices, same insertion) offering only keys >= T, then the
    /// stock merges. The stock scan offers every logit to a 16-slot insertion
    /// that nearly every simdgroup step takes (some lane's list is still
    /// filling or beaten); here it is offered a handful per chunk.
    private static let thresholdChunkKernel = MLXFast.metalKernel(
        name: "mlxfast_dflash_topk_chunk_threshold",
        inputNames: ["logits"],
        outputNames: ["part_key", "part_idx"],
        source: Qwen35IO32.narrow("""
            constexpr int KK = KTOP;
            constexpr uint NSG = TPG / 32;
            static_assert(TPG % 32 == 0 && NSG <= 32 && KK <= 32 && (KK % NSG) == 0 && VEC, "shape");
            constexpr int PER = KK / NSG;
            const uint t = thread_position_in_threadgroup.x;
            const uint chunk = threadgroup_position_in_grid.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            constexpr uint CH = (NV + S - 1) / S;
            static_assert((CH % 4u) == 0u, "four-wide");
            const uint lo = chunk * CH;
            const uint hi = min(lo + CH, uint(NV));
            auto x = logits + size_t(row) * NV;
            uint mx = 0u;
            for (uint v = lo + t * 4u; v + 3u < hi; v += TPG * 4u) {
                uint k0, k1, k2, k3;
                if (HALF) {
                    const half4 q = *(const device half4*)(x + v);
                    k0 = mlxfast_topk_key(float(q[0])); k1 = mlxfast_topk_key(float(q[1]));
                    k2 = mlxfast_topk_key(float(q[2])); k3 = mlxfast_topk_key(float(q[3]));
                } else {
                    const float4 q = *(const device float4*)(x + v);
                    k0 = mlxfast_topk_key(q[0]); k1 = mlxfast_topk_key(q[1]);
                    k2 = mlxfast_topk_key(q[2]); k3 = mlxfast_topk_key(q[3]);
                }
                mx = max(mx, max(max(k0, k1), max(k2, k3)));
            }
            uint m = mx;
            uint thr = 0u;
            for (int r = 0; r < PER; r++) {
                const uint top = simd_max(m);
                thr = top;
                const uint who = simd_min(m == top ? lane : 0xffffffffu);
                if (lane == who) m = 0u;
            }
            threadgroup uint sthr[NSG];
            if (lane == 0) sthr[sg] = thr;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint T = 0xffffffffu;
            for (uint j = 0; j < NSG; j++) T = min(T, sthr[j]);
            uint k[KK], id[KK];
            for (int j = 0; j < KK; j++) { k[j] = 0u; id[j] = 0u; }
            for (uint v = lo + t * 4u; v + 3u < hi; v += TPG * 4u) {
                uint k0, k1, k2, k3;
                if (HALF) {
                    const half4 q = *(const device half4*)(x + v);
                    k0 = mlxfast_topk_key(float(q[0])); k1 = mlxfast_topk_key(float(q[1]));
                    k2 = mlxfast_topk_key(float(q[2])); k3 = mlxfast_topk_key(float(q[3]));
                } else {
                    const float4 q = *(const device float4*)(x + v);
                    k0 = mlxfast_topk_key(q[0]); k1 = mlxfast_topk_key(q[1]);
                    k2 = mlxfast_topk_key(q[2]); k3 = mlxfast_topk_key(q[3]);
                }
                if (max(max(k0, k1), max(k2, k3)) >= T) {
                    if (k0 >= T) mlxfast_topk_consider<KK>(k, id, k0, v);
                    if (k1 >= T) mlxfast_topk_consider<KK>(k, id, k1, v + 1u);
                    if (k2 >= T) mlxfast_topk_consider<KK>(k, id, k2, v + 2u);
                    if (k3 >= T) mlxfast_topk_consider<KK>(k, id, k3, v + 3u);
                }
            }
            uint ok = 0u, oi = 0u;
            mlxfast_topk_simd_merge<KK>(k, id, lane, ok, oi);
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
            """, count: 3, "mlxfast_dflash_topk_chunk_threshold"),
        header: header)

    private static let mergeKernel = MLXFast.metalKernel(
        name: "mlxfast_dflash_topk_merge",
        inputNames: ["logits", "part_key", "part_idx"],
        outputNames: ["cand", "val"],
        source: Qwen35IO32.narrow("""
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
            """, count: 3, "mlxfast_dflash_topk_merge"),
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

    /// `MLXFAST_DFLASH_EDGE_WEIGHT` (default 0.75): the walk scores
    /// `unary + w * edge`. The drafter's edge overweights the predecessor
    /// against its own unary logits; 0.75 ranks the target's continuation
    /// first more often (offline over the chain runs' candidate dumps). The
    /// target verifies every proposal, so tokens do not change. 1 is the
    /// recorded walk exactly (the unweighted kernels); a weighted walk uses
    /// its own kernels, the serial, parallel and narrow ones alike, so their
    /// self-tests compare the weighted walks.
    static let edgeWeight: Float = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_EDGE_WEIGHT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let w = raw.flatMap(Float.init), w.isFinite, w >= 0 else { return 0.75 }
        return w
    }()
    private static let weightSuffix =
        edgeWeight == 1 ? "" : "_w" + String(edgeWeight.bitPattern, radix: 16)
    private static func scored(_ source: String) -> String {
        edgeWeight == 1
            ? source
            : source.replacingOccurrences(
                of: "+ edge;", with: "+ as_type<float>(\(edgeWeight.bitPattern)u) * edge;")
    }

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
        // Only the measured singleton 15 x 16 x 256 BF16 selector uses the
        // narrow parallel edges. Any other shape/dtype keeps its old path.
        if narrowParallelSetting, parallelActive, length == 15, k == 16, rank == 256,
            unary.ndim == 3, unary.dim(0) == 1, unary.dim(1) == length,
            unary.dim(2) == k, unary.dtype == .float32,
            projected.ndim == 3, projected.dim(0) == 1, projected.dim(1) == length,
            projected.dtype == .bfloat16,
            predecessorCodebook.ndim == 2, predecessorCodebook.dim(1) == rank,
            predecessorCodebook.dtype == .bfloat16,
            successorCodebook.ndim == 2, successorCodebook.dim(1) == rank,
            successorCodebook.dtype == .bfloat16,
            narrowParallelActive
        {
            return selectNarrowParallel(
                candidates: candidates, unary: unary, projected: projected, anchor: anchor,
                predecessorCodebook: predecessorCodebook, successorCodebook: successorCodebook)
        }
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
        // Reuse the narrow path's singleton-batch views while keeping this
        // path's FP32 widening and stock kernel. Preserve general indexing.
        let c = candidates.dim(0) == 1 ? candidates.squeezed(axis: 0) : candidates[0]
        // Gather only the codebook rows the candidate lists can visit. The
        // fused kernel scores each edge and advances the greedy walk in one
        // pass, instead of materializing an [L-1, K, K, rank] broadcast.
        let anchorPredecessor = take(predecessorCodebook, anchor, axis: 0)
            .asType(.float32).reshaped([-1])
        let previous = take(predecessorCodebook, c[0 ..< (length - 1)], axis: 0)
            .asType(.float32).reshaped([-1])
        let next = take(successorCodebook, c, axis: 0).asType(.float32).reshaped([-1])
        let projectedBatch = projected.dim(0) == 1 ? projected.squeezed(axis: 0) : projected[0]
        let projectedRows = projectedBatch.asType(.float32).reshaped([-1])
        let unaryBatch = unary.dim(0) == 1 ? unary.squeezed(axis: 0) : unary[0]
        let scores = unaryBatch.asType(.float32).reshaped([-1])
        let candidateIds = c.asType(.uint32).reshaped([-1])
        let operands = [anchorPredecessor, previous, next, projectedRows, scores, candidateIds]
        let path =
            parallelActive
            ? walkParallel(operands, length: length, k: k, rank: rank)
            : walkSerial(operands, length: length, k: k, rank: rank)
        return path.reshaped([1, length])
    }

    /// The recorded walk: one simdgroup scores each position's candidates
    /// against the previous pick and advances, position by position.
    private static func walkSerial(_ operands: [MLXArray], length: Int, k: Int, rank: Int) -> MLXArray {
        kernel(
            operands,
            template: [
                ("L", length), ("K", k), ("R", rank),
                ("WALKVEC", vectorRank && rank % 4 == 0 ? 1 : 0),
            ],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[length]],
            outputDTypes: [.int32])[0]
    }

    /// The same walk with every edge scored up front: one thread per edge
    /// (position 0's K anchor edges, then each later position's K x K
    /// predecessor-slot x candidate edges) runs the walk kernel's own chain
    /// over the rank in the same order, so each edge is the value the walk
    /// computes for that pair; then one simdgroup walks the table, adding the
    /// same unary score and taking the same first maximum. The serial walk
    /// computes 15 x 16 of these edges one position after the other, each
    /// waiting on the previous pick; here all 3,600 run at once (about 80 us
    /// -> 13 us per proposal). Self-tested at bind against `walkSerial`
    /// (`parallelVerified`), every path id compared; `MLXFAST_DFLASH_WALK_PARALLEL=0`
    /// keeps the serial walk.
    private static func walkParallel(_ operands: [MLXArray], length: Int, k: Int, rank: Int) -> MLXArray {
        let edges = length >= 1 ? k + (length - 1) * k * k : 0
        let table = edgeKernel(
            operands,
            template: [
                ("L", length), ("K", k), ("R", rank),
                ("WALKVEC", vectorRank && rank % 4 == 0 ? 1 : 0),
            ],
            grid: (edges, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[edges]],
            outputDTypes: [.float32])[0]
        return tableKernel(
            [table, operands[4], operands[5]],
            template: [("L", length), ("K", k)],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[length]],
            outputDTypes: [.int32])[0]
    }

    /// Keep the wide path's exact edge layout and FP32 table walk. Only the
    /// first four operands differ: gathered BF16 rows instead of FP32 copies.
    private static func walkNarrowParallel(
        _ operands: [MLXArray], length: Int, k: Int, rank: Int
    ) -> MLXArray {
        let edges = k + (length - 1) * k * k
        let table = narrowEdgeKernel(
            operands,
            template: [("L", length), ("K", k), ("R", rank)],
            grid: (edges, 1, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[edges]],
            outputDTypes: [.float32])[0]
        return tableKernel(
            [table, operands[4], operands[5]],
            template: [("L", length), ("K", k)],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[length]],
            outputDTypes: [.int32])[0]
    }

    /// `MLXFAST_DFLASH_WALK_NARROW_PARALLEL=0` keeps the stock widened
    /// parallel edges, independently of the older serial-narrow opt-in.
    static let narrowParallelSetting: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_WALK_NARROW_PARALLEL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static let parallelSetting: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_WALK_PARALLEL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Whether the widened walk takes `walkParallel`: set once by
    /// `parallelVerified` (at bind) when every compared path matched.
    nonisolated(unsafe) static var parallelActive = false
    nonisolated(unsafe) private static var parallelChecked = false

    /// Runs `walkSerial` and `walkParallel` on the same random FP32 operands
    /// (the widened walk's; 64 walks of 15 positions over 16 candidates at
    /// rank 256, a third of them with tied unary scores and a third with small
    /// projections so edges tie too) and compares every path id; the parallel
    /// walk is used only when all match. One stderr line.
    static func parallelVerified() -> Bool {
        narrowLock.withLock {
            if parallelChecked { return parallelActive }
            parallelChecked = true
            guard enabled, parallelSetting else { return false }
            var same = true
            var walks = 0
            do {
                try withError { error in
                    let (vocab, length, k, rank) = (4096, 15, 16, 256)
                    for seed in 0 ..< 64 {
                        func key(_ salt: Int) -> MLXArray { MLXRandom.key(UInt64(0x3a1f + seed * 8 + salt)) }
                        let pred = MLXRandom.normal([vocab, rank], key: key(0)).asType(.bfloat16)
                        let succ = MLXRandom.normal([vocab, rank], key: key(1)).asType(.bfloat16)
                        var proj = MLXRandom.normal([length, rank], key: key(2))
                        if seed % 3 == 1 { proj = proj * 0.001 }
                        var una = (MLXRandom.normal([length, k], key: key(3)) * 4).asType(.float16)
                        if seed % 3 == 2 { una = MLX.floor(una) }
                        let cand = MLXRandom.randInt(Int32(0) ..< Int32(vocab), [length, k], key: key(4))
                            .asType(.uint32)
                        let anchor = MLXArray([Int32(seed * 97 % vocab)])
                        let operands = [
                            take(pred, anchor, axis: 0).asType(.float32).reshaped([-1]),
                            take(pred, cand[0 ..< (length - 1)], axis: 0).asType(.float32).reshaped([-1]),
                            take(succ, cand, axis: 0).asType(.float32).reshaped([-1]),
                            proj.asType(.bfloat16).asType(.float32).reshaped([-1]),
                            una.asType(.float32).reshaped([-1]),
                            cand.reshaped([-1]),
                        ]
                        let serial = walkSerial(operands, length: length, k: k, rank: rank)
                        let parallel = walkParallel(operands, length: length, k: k, rank: rank)
                        same = same && all(serial .== parallel).item(Bool.self)
                        walks += 1
                    }
                    try error.check()
                }
            } catch {
                same = false
            }
            parallelActive = same
            FileHandle.standardError.write(
                ("dflash2 greedy walk parallel edges: "
                    + (same
                        ? "self-test passed: \(walks) walks of 15 ids identical; on; edge weight \(edgeWeight)\n"
                        : "self-test failed; serial walk kept\n")).data(using: .utf8)!)
            return same
        }
    }

    private static let edgeKernel = MLXFast.metalKernel(
        name: "mlxfast_dflash_walk_edges",
        inputNames: [
            "anchor_predecessor", "previous", "next", "projected", "unary", "cand",
        ],
        outputNames: ["edges"],
        source: """
            // One thread per edge: gid < K scores position 0's candidate gid
            // against the anchor; the rest are position i >= 1's
            // (predecessor slot p, candidate c) pairs. The chain is
            // `kernel`'s, statement for statement (no unrolling hint, so the
            // compiler sees the same loop).
            const uint gid = thread_position_in_grid.x;
            if (gid >= uint(K + (L - 1) * K * K)) return;
            uint i, p, c;
            if (gid < uint(K)) { i = 0; p = 0; c = gid; }
            else { const uint e = gid - K; i = 1 + e / (K * K); p = (e / K) % K; c = e % K; }
            const uint pred_base = i == 0 ? 0 : ((i - 1) * K + p) * R;
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
            edges[gid] = edge;
            """)

    private static let narrowEdgeKernel = MLXFast.metalKernel(
        name: "mlxfast_dflash_walk_edges_bf16",
        inputNames: [
            "anchor_predecessor", "previous", "next", "projected", "unary", "cand",
        ],
        outputNames: ["edges"],
        source: """
            // Match edgeKernel's gid layout and left-to-right FP32 sum. Only
            // the four gathered BF16 operands change; unary stays FP32 in
            // the unchanged tableKernel, which owns weighting and tie breaks.
            const uint gid = thread_position_in_grid.x;
            if (gid >= uint(K + (L - 1) * K * K)) return;
            uint i, p, c;
            if (gid < uint(K)) { i = 0; p = 0; c = gid; }
            else { const uint e = gid - K; i = 1 + e / (K * K); p = (e / K) % K; c = e % K; }
            const uint pred_base = i == 0 ? 0 : ((i - 1) * K + p) * R;
            const uint succ_base = (i * K + c) * R;
            auto pred_ptr = (i == 0) ? anchor_predecessor : (previous + pred_base);
            auto proj_ptr = projected + i * R;
            auto succ_ptr = next + succ_base;
            float edge = 0.0f;
            #pragma clang loop unroll(full)
            for (uint d = 0; d < R; d++) {
                edge += (float(pred_ptr[d]) * float(proj_ptr[d])) * float(succ_ptr[d]);
            }
            edges[gid] = edge;
            """)

    private static let tableKernel = MLXFast.metalKernel(
        name: "mlxfast_dflash_walk_table" + weightSuffix,
        inputNames: ["edges", "unary", "cand"],
        outputNames: ["path"],
        source: scored("""
            uint c = thread_index_in_simdgroup;
            uint previous_slot = 0;
            for (uint i = 0; i < L; i++) {
                float score = -INFINITY;
                if (c < K) {
                    const float edge = i == 0
                        ? edges[c] : edges[K + ((i - 1) * K + previous_slot) * K + c];
                    score = unary[i * K + c] + edge;
                }
                float m = simd_max(score);
                uint sel = simd_min((c < K && score == m) ? c : 0xffffffffu);
                previous_slot = sel;
                if (c == 0) path[i] = int(cand[i * K + sel]);
            }
            """))

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

    /// The parallel narrow path removes all four BF16-to-FP32 copies made by
    /// selectWide. It keeps selectWide's FP32 unary and unchanged tableKernel.
    private static func selectNarrowParallel(
        candidates: MLXArray, unary: MLXArray, projected: MLXArray, anchor: MLXArray,
        predecessorCodebook: MLXArray, successorCodebook: MLXArray
    ) -> MLXArray {
        let length = candidates.dim(1)
        let k = candidates.dim(2)
        let rank = projected.dim(2)
        let c = candidates.squeezed(axis: 0)
        let operands = [
            take(predecessorCodebook, anchor, axis: 0).reshaped([-1]),
            take(predecessorCodebook, c[0 ..< (length - 1)], axis: 0).reshaped([-1]),
            take(successorCodebook, c, axis: 0).reshaped([-1]),
            projected.squeezed(axis: 0).reshaped([-1]),
            unary.squeezed(axis: 0).asType(.float32).reshaped([-1]),
            c.asType(.uint32).reshaped([-1]),
        ]
        return walkNarrowParallel(operands, length: length, k: k, rank: rank)
            .reshaped([1, length])
    }

    nonisolated(unsafe) private static var narrowParallelChecked = false
    nonisolated(unsafe) private static var narrowParallelActive = false

    /// Check complete Int32 paths bitwise against the stock widened parallel
    /// path. Failure or a GPU error keeps the prior dispatch; never accept a
    /// merely close FP score or a subset of matched positions.
    private static func narrowParallelVerified() -> Bool {
        narrowLock.withLock {
            if narrowParallelChecked { return narrowParallelActive }
            narrowParallelChecked = true
            guard enabled, parallelActive, narrowParallelSetting else { return false }
            var same = true
            var walks = 0
            do {
                try withError { error in
                    let (vocab, length, k, rank) = (4096, 15, 16, 256)
                    for seed in 0 ..< 32 {
                        func key(_ salt: Int) -> MLXArray {
                            MLXRandom.key(UInt64(0x74a1 + seed * 8 + salt))
                        }
                        let pred = MLXRandom.normal([vocab, rank], key: key(0)).asType(.bfloat16)
                        let succ = MLXRandom.normal([vocab, rank], key: key(1)).asType(.bfloat16)
                        var proj = MLXRandom.normal([1, length, rank], key: key(2))
                        if seed % 3 == 1 { proj = proj * 0.001 }
                        let projected = proj.asType(.bfloat16)
                        var una = MLXRandom.normal([1, length, k], key: key(3)) * 4
                        if seed % 3 == 2 { una = MLX.floor(una) }
                        let cand = MLXRandom.randInt(
                            Int32(0) ..< Int32(vocab), [1, length, k], key: key(4))
                            .asType(.uint32)
                        let anchor = MLXArray([Int32(seed * 97 % vocab)])
                        guard let wide = selectWide(
                            candidates: cand, unary: una, projected: projected, anchor: anchor,
                            predecessorCodebook: pred, successorCodebook: succ)
                        else { same = false; return }
                        let narrow = selectNarrowParallel(
                            candidates: cand, unary: una, projected: projected, anchor: anchor,
                            predecessorCodebook: pred, successorCodebook: succ)
                        same = same && all(
                            wide.view(dtype: .uint32) .== narrow.view(dtype: .uint32)).item(Bool.self)
                        walks += 1
                        if !same { break }
                    }
                    try error.check()
                }
            } catch {
                same = false
            }
            narrowParallelActive = same
            FileHandle.standardError.write(
                ("dflash2 greedy walk BF16 parallel edges: "
                    + (same
                        ? "self-test passed: \(walks) walks of 15 ids bitwise identical; on\n"
                        : "self-test failed; prior walk kept\n")).data(using: .utf8)!)
            return same
        }
    }

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
        _ = parallelVerified()
        if codebook == .bfloat16, projected == .bfloat16, unary == .float32,
            parallelActive, narrowParallelSetting
        {
            _ = narrowParallelVerified()
        }
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
        name: "mlxfast_dflash_fused_greedy_walk_narrow" + weightSuffix,
        inputNames: [
            "anchor_predecessor", "previous", "next", "projected", "unary", "cand",
        ],
        outputNames: ["path"],
        source: scored("""
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
            """))

    private static let kernel = MLXFast.metalKernel(
        name: "mlxfast_dflash_fused_greedy_walk" + weightSuffix,
        inputNames: [
            "anchor_predecessor", "previous", "next", "projected", "unary", "cand",
        ],
        outputNames: ["path"],
        source: scored("""
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
            """))
}

// MARK: - Residency prefetch

/// One command buffer that binds a group of arrays: a single thread reads the
/// first element of each input. It exists for the buffers it binds, not for
/// its output (see `DFlash2ResidencyPrefetch`).
///
/// Every launch has `inputs` inputs of one dtype (a group's arrays by dtype,
/// the last launch of a dtype repeating its last array), so the kernels a
/// seed can meet are one per dtype, all compiled by `prewarm` at the first
/// arming (the load-time engine warm): which arrays are due depends on the
/// kernels the load-time trials adopt, and a new mix must not JIT in a seed.
private enum DFlash2ResidencyTouch {
    /// Inputs per kernel; with the output well inside Metal's 31 buffer slots.
    static let inputs = 24

    private static let kernel: MLXFast.MLXFastKernel = {
        let names = (0 ..< inputs).map { "w\($0)" }
        let reads = names.map { "acc += static_cast<float>(\($0)[0]);" }.joined(separator: "\n")
        return MLXFast.metalKernel(
            name: "dflash2_residency_touch", inputNames: names, outputNames: ["out"],
            source: "float acc = 0.0f;\n" + reads + "\nout[0] = acc;\n",
            ensureRowContiguous: false)
    }()

    private static func launch(_ chunk: [MLXArray]) -> MLXArray {
        kernel(
            chunk, grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [[1]], outputDTypes: [.float32])[0]
    }

    /// The touches of `arrays`: per dtype (first-seen order), `inputs` at a time.
    static func touch(_ arrays: [MLXArray]) -> [MLXArray] {
        var order: [DType] = []
        var byType: [DType: [MLXArray]] = [:]
        for array in arrays {
            if byType[array.dtype] == nil { order.append(array.dtype) }
            byType[array.dtype, default: []].append(array)
        }
        return order.flatMap { dtype -> [MLXArray] in
            let same = byType[dtype]!
            return stride(from: 0, to: same.count, by: inputs).map { start in
                var chunk = Array(same[start ..< min(start + inputs, same.count)])
                chunk += Array(repeating: chunk[chunk.count - 1], count: inputs - chunk.count)
                return launch(chunk)
            }
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var warmed = false

    /// Compiles the touch for every dtype a weight can have, once.
    static func prewarm() {
        guard lock.withLock({ () -> Bool in
            defer { warmed = true }
            return !warmed
        }) else { return }
        let types: [DType] = [.bfloat16, .float16, .float32, .uint32, .uint16, .uint8, .int8, .int32]
        eval(types.map { launch(Array(repeating: MLXArray.zeros([1], dtype: $0), count: inputs)) })
    }
}

/// The drafter's registered parameter arrays, flattened once per drafter.
/// `residencyArrays()` runs on every engine build that drafts (at the prompt
/// forward's first submission under the deferred arm), and its walk of the
/// parameters is a reflection over the whole module tree: host time inside the
/// timed seed, for a list that cannot change. The drafter's parameters are
/// bound once at load and frozen (its stacked and tiled forms live outside
/// them and are still read per call in `residencyArrays`), so the walk's
/// result, the same arrays in the same order, is kept after the first one.
/// `MLXFAST_RESIDENCY_LIST_CACHE=0` walks every time.
enum DFlash2ResidencyParameterList {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_RESIDENCY_LIST_CACHE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var cached: (drafter: DFlash2DraftModel, arrays: [MLXArray])?

    static func arrays(of drafter: DFlash2DraftModel) -> [MLXArray] {
        guard enabled else { return drafter.parameters().flattened().map { $0.1 } }
        if let hit = lock.withLock({ () -> [MLXArray]? in
            guard let cached, cached.drafter === drafter else { return nil }
            return cached.arrays
        }) {
            return hit
        }
        let arrays = drafter.parameters().flattened().map { $0.1 }
        lock.withLock { cached = (drafter, arrays) }
        return arrays
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
/// The same holds for any array only the window reads, so the list goes on
/// with the target's (`arm`'s `window`): the head rows the drafter scores,
/// stored words and constants the seed's own head read (the verify route's
/// tiled copy) never binds, and the verify int8 operands that the kernels the
/// load-time trials adopted read and the prompt route does not (the
/// FP32-widened scales of a form that reads them, every operand of a
/// projection the prompt route never runs). The drafter's own list follows the
/// adopted drafter kernel: its tiled copies while `DFlash2TensorMatmul.current`
/// reads them (the tiled kernel or any variant), the stored weights otherwise.
/// Kernel outputs (the cross-threadgroup bodies' partial planes included) are
/// fresh allocations in every round, not arrays to bind ahead.
///
/// Each group costs one single-thread kernel per 24 arrays on the GPU. No
/// value of either model is read into any result, and nothing about the
/// arithmetic changes. `DARKBLOOM_DFLASH2_RESIDENCY_PREFETCH=0` turns it off.
///
/// WHERE THE LIST IS BUILT. The engine build runs at the head of the seed
/// window, before any GPU work, so whatever it does there is host time the
/// seed pays in full: walking the drafter's modules for its arrays and
/// splitting them into the byte groups took 0.6-2.3 ms of the 0.9-3.5 ms
/// engine build (local M4 phase markers). Nothing reads the groups before the
/// prompt forward's first early submission, so `arm` keeps only the drafter
/// and the window's arrays (read at arm, as before) and the first
/// `submitDue` builds the same list and the same groups there, behind the
/// GPU work that submission just queued. The touches, their order, their
/// groups and the layers they follow are unchanged.
/// `MLXFAST_RESIDENCY_DEFERRED_ARM=0` builds the groups at `arm` again.
public enum DFlash2ResidencyPrefetch {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH2_RESIDENCY_PREFETCH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Builds the groups at the prompt forward's first submission rather than
    /// at `arm` (default on; see the type's comment).
    static let deferredArm: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_RESIDENCY_DEFERRED_ARM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The prompt layers after which one group each is due.
    static let dueAfterLayers = [8, 16, 32, 48]

    private static let lock = NSLock()
    nonisolated(unsafe) private static var groups: [[MLXArray]] = []
    nonisolated(unsafe) private static var nextGroup = 0
    /// An arm whose groups are not built yet: the drafter and the window's
    /// arrays, as `arm` read them or, deferred, to be read with the groups.
    nonisolated(unsafe) private static var unbuilt: (drafter: DFlash2DraftModel, window: () -> [MLXArray])?

    /// Under the deferred arm, the window's arrays are read where the groups
    /// are built, at the prompt forward's first early submission, instead of
    /// at `arm`. `arm` runs inside the engine build at the head of the seed
    /// window, before any GPU work, and the window read (every verify site's
    /// adopted form and existing operands, then the head rows) was most of
    /// what it still did there: 0.5-0.6 ms of host time before the seed's
    /// first command buffer (local M4 markers after the decode gate). The
    /// read depends on state the load, the load-time trials and the warm-up
    /// settle before any timed request, and nothing between an engine build
    /// and its prompt forward's first submission changes it, so it returns the
    /// same arrays there. The touches read nothing into any result either way.
    /// `MLXFAST_RESIDENCY_WINDOW_DEFERRED=0` reads the window at `arm` again.
    static let deferredWindow: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_RESIDENCY_WINDOW_DEFERRED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Arms the prefetch for the next prompt forward: called when an engine
    /// that drafts with `drafter` is built. `window`: the target's arrays the
    /// window reads and the seed does not (the drafter's head rows, the verify
    /// kernels' window-only operands), after the drafter's own.
    static func arm(
        _ drafter: DFlash2DraftModel, window: @escaping @autoclosure () -> [MLXArray] = []
    ) {
        guard enabled else { return }
        DFlash2ResidencyTouch.prewarm()
        if deferredArm && deferredWindow {
            lock.withLock {
                unbuilt = (drafter, window)
                groups = []
                nextGroup = 0
            }
            return
        }
        let windowArrays = window()
        if deferredArm {
            lock.withLock {
                unbuilt = (drafter, { windowArrays })
                groups = []
                nextGroup = 0
            }
            return
        }
        let built = buildGroups(of: drafter, window: windowArrays)
        lock.withLock {
            unbuilt = nil
            groups = built
            nextGroup = 0
        }
    }

    /// The drafter's arrays, then the window's, each once, in four groups of
    /// about equal bytes.
    private static func buildGroups(of drafter: DFlash2DraftModel, window: [MLXArray]) -> [[MLXArray]] {
        var seen = Set<ObjectIdentifier>()
        let arrays = (drafter.residencyArrays() + window).filter {
            seen.insert(ObjectIdentifier($0)).inserted
        }
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
        return built
    }

    /// Submits every group due once the prompt forward has submitted its
    /// first `completedLayers` layers. A no-op unless armed.
    static func submitDue(completedLayers: Int) {
        // A deferred arm's groups, built at the forward's first submission.
        let pending = lock.withLock { () -> (drafter: DFlash2DraftModel, window: () -> [MLXArray])? in
            defer { unbuilt = nil }
            return unbuilt
        }
        if let pending {
            let built = buildGroups(of: pending.drafter, window: pending.window())
            lock.withLock {
                groups = built
                nextGroup = 0
            }
        }
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
        DFlash2ContextRows.prepare(width: config.targetHiddenSize, dtype: dtype)
        if let slidingWindow = config.slidingWindow {
            DFlash2AbsorbKV.prepare(
                rope: rope, base: config.ropeTheta, dtype: dtype, heads: config.attentionHeads,
                kvHeads: config.kvHeads, headDim: config.headDim,
                kNorms: layers.map { $0.selfAttn.kNorm }, cacheSize: slidingWindow - 1)
        }
        DFlash2AttentionPipeline.verify(
            dtype: dtype, heads: config.attentionHeads, kvHeads: config.kvHeads, headDim: config.headDim)
        DFlash2TopK.prepareThreshold(vocabularySize: Qwen35TextModel.drafterVocabularyRows > 0
            ? min(Qwen35TextModel.drafterVocabularyRows, config.vocabularySize) : config.vocabularySize,
            k: candidateSelector.topK)
        DFlash2GreedyWalk.prepare(
            codebook: candidateSelector.predecessorCodebook.dtype,
            projected: candidateSelector.hiddenProjection.weight.dtype, unary: .float32)
        DFlash2Concat.prepare(inputs: config.targetLayerIds.count, dtype: .float16)
        DFlash2Concat.raceSmall(
            taps: config.targetLayerIds.count, rows: 16, width: config.hiddenSize,
            hidden: config.hiddenSize, dtype: dtype)
        DFlash2RoundFuse2.prepare(hidden: config.hiddenSize, maskRow: maskEmbedding)
    }

    /// `DFlash2SpeculativeFront`'s self-tests and trial for blocks of
    /// `blockSize` rows (at load, before the speculation self-test).
    public func prepareSpeculativeFront(blockSize: Int) {
        DFlash2SpeculativeFront.prepare(
            rope: rope, base: config.ropeTheta, dtype: dtype, heads: config.attentionHeads,
            kvHeads: config.kvHeads, headDim: config.headDim, eps: config.rmsNormEps,
            hidden: config.hiddenSize, kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize, blockSize: blockSize)
        DFlash2PacketFront.prepare(depth: blockSize - 1, width: config.targetHiddenSize, dtype: dtype)
    }

    /// Builds and self-tests the tiled copies of the weights the block
    /// forward reads through `DFlash2TensorMatmul` (every layer's o_proj,
    /// stacked gate|up and down_proj). `fc` keeps its stored layout: the
    /// prompt's context rows read it through the core's GEMM in the seed, so
    /// a copy only the rounds read would be one more array to make GPU-
    /// resident again at each decode window's first round.
    func prepareTiledWeights() -> Bool {
        let layerWeights = layers.flatMap { $0.tensorWeights() }
        let blockWeights =
            layerWeights + layers.flatMap { $0.projectionWeights() } + (fc.bias == nil ? [fc.weight] : [])
        let qkvWeights = layers.compactMap { $0.selfAttn.stackedQKVWeight() }
        _ = DFlash2TensorMatmul.prepareSwapped(blockWeights, rows32: qkvWeights)
        DFlash2PackedWeights.prepare(blockWeights, rows32: qkvWeights)
        // The tiling trial's weights, on the kernel the two self-tests left on.
        DFlash2TensorMatmul.SwapTrial.note(blockWeights, rows32: qkvWeights)
        return DFlash2TensorMatmul.prepareTiled(layerWeights)
    }

    /// After the deferred tiling trial, before any served request. Each
    /// attention chooses independently; ordinary proposing remains unchanged.
    func prepareQueryWindows() -> Bool {
        var chosen = false
        for layer in layers { chosen = layer.selfAttn.prepareQueryWindow() || chosen }
        return chosen
    }

    func disableQueryWindows() {
        for layer in layers { layer.selfAttn.disableQueryWindow() }
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
        for array in DFlash2ResidencyParameterList.arrays(of: self)
        where !replaced.contains(ObjectIdentifier(array))
            && seen.insert(ObjectIdentifier(array)).inserted
        {
            arrays.append(array)
        }
        // `fc` stays (the prompt's context rows read it); its rounds read its packed copy.
        for array in DFlash2PackedWeights.copy(of: fc.weight)?.arrays ?? []
        where seen.insert(ObjectIdentifier(array)).inserted {
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
        // A leading submission of several layers also submits its first
        // layer on its own (`DFlash2DraftSubmission.firstLayerAhead`).
        let firstAt =
            DFlash2DraftSubmission.firstLayerAhead && leadAt > 1 ? 1 : 0
        for (index, layer) in layers.enumerated() {
            h = layer(h, context: context, rope: rope, cache: cache[index], masks: masks)
            // EARLY SUBMISSION: hand the GPU the drafter layers built so far
            // while the host builds the rest and the head. Same kernels, same
            // order; only command-buffer boundaries move. One submission per
            // layer at most, whichever of the two asks for it.
            if index + 1 == leadAt || index + 1 == firstAt
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
        let anchorRow = target.embedTokensForDFlash2(anchorIDs)
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
                if DFlash2Concat.mayLaunch {
                    repeatedMasks = contiguous(repeatedMasks)
                }
                eval(repeatedMasks)
                cachedMaskEmbeddingBlock = (cols: cols, array: repeatedMasks)
            }
        }
        if batch == 1, dtype == .bfloat16, DFlash2RoundFuse2.anchorReady,
            let joined = DFlash2Concat.anchorJoin(anchorRow, repeatedMasks)
        {
            return joined
        }
        return DFlash2Concat.concatenate([anchorRow.asType(dtype), repeatedMasks], axis: 1)
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
    ///
    /// `acceptancePacket` (the verify's `[drafts | targets]` that `anchor`
    /// and `confirmed` were read from, depth `blockSize - 1`) lets the block's
    /// device inputs take one launch (`DFlash2PacketFront`); `anchor` and
    /// `confirmed` are then not read.
    public func proposeSpeculative(
        anchor: MLXArray, confirmed: MLXArray, verifyContext: MLXArray, contextRows: Int,
        cache: [KVCache], blockSize: Int, leadingLayers: Int, submitLead: Bool,
        maskUnconfirmed: Bool = false, acceptancePacket: MLXArray? = nil
    ) throws -> DFlash2SpeculativeBlock? {
        guard let geometry = speculativeGeometry(cache: cache, blockSize: blockSize),
            anchor.size == 1, confirmed.size == 1,
            verifyContext.shape == [1, blockSize, config.targetHiddenSize],
            (1 ... blockSize).contains(contextRows)
        else { return nil }
        let caches = cache.map { $0 as! DFlash2BlockKVCache }
        let n = 2 * blockSize
        let keys = geometry.rows + n
        let packetFront = acceptancePacket.flatMap {
            DFlash2PacketFront.apply(
                $0, depth: blockSize - 1, context: verifyContext, rows: contextRows, dtype: dtype,
                masked: maskUnconfirmed, keys: keys, keyBound: geometry.rows + blockSize,
                offset: geometry.offset)
        }
        let anchor = packetFront?.anchor ?? anchor
        var h = try blockEmbedding(anchorIDs: anchor.reshaped([1, 1]), maskColumns: blockSize - 1)
            .asType(dtype)
        if config.dflash.inputEmbeddingScale != 1 {
            h = h * config.dflash.inputEmbeddingScale
        }
        let c = (packetFront?.confirmed ?? confirmed).reshaped([]).asType(.int32)
        let verifyRows = verifyContext[0..., ..<contextRows, 0...]
        // `DFlash2ExactRowClasses`: the rows at and past the confirmed count
        // become the zero rows today's padded projection reads.
        let context = contextProjection(
            packetFront?.rows
                ?? (maskUnconfirmed
                    ? DFlash2ContextRows.masked(verifyRows, c, dtype: dtype)
                        ?? DFlash2ContextRows.reference(verifyRows, c)
                    : verifyRows))
        let base = concatenated(
            [context, MLXArray.zeros([1, n - contextRows, config.hiddenSize], dtype: context.dtype)],
            axis: 1)
        let queryOffset = packetFront?.queryOffset.reshaped([]) ?? (MLXArray(Int32(geometry.offset)) + c)
        let keyMask = packetFront?.keyMask
            ?? DFlash2RoundFuse2.keyMask(
                keys: keys, bound: geometry.rows + blockSize, c, fused: DFlash2RoundFuse2.maskReady
            ).reshaped([1, keys])
        let leadAt = submitLead ? min(max(leadingLayers, 0), layers.count) : 0
        // The front's one-launch forms read `context` and `c` themselves;
        // `base` is then never evaluated.
        let front = DFlash2SpeculativeFront.active
        let pos: MLXArray? = DFlash2SpeculativeFront.headsChosen
            ? MLXArray([Int32(geometry.offset), Int32(geometry.offset), Int32(blockSize)]) : nil
        var writes: [(keys: MLXArray, values: MLXArray)] = []
        var lead: MLXArray?
        // Not submitted before the readback: the same leading layers are
        // submitted at adoption instead (`DFlash2AdoptionStage`).
        let stageAt = leadAt == 0 && DFlash2AdoptionStage.enabled
            ? min(max(leadingLayers, 0), layers.count) : 0
        var stage: MLXArray?
        for (index, layer) in layers.enumerated() {
            guard
                let out = layer.speculative(
                    h, base: base, context: front ? context : nil, confirmed: c,
                    queryOffset: queryOffset, pos: pos, rope: rope,
                    cache: caches[index], keyMask: keyMask)
            else { preconditionFailure("DFlash 2: a checked layer refused its speculative block") }
            h = out.hidden
            writes.append((out.keys, out.values))
            if index + 1 == leadAt { lead = h }
            if index + 1 == stageAt { stage = h }
        }
        let hidden = norm(h[0..., 1..., 0...])
        let tokens = candidateSelector.selectGreedy(
            hidden: hidden, logits: try logits(hidden), anchor: anchor.reshaped([1]))
        if let lead {
            for i in 0 ..< leadAt { caches[i].installSpeculative(keys: writes[i].keys, values: writes[i].values) }
            asyncEval([lead])
        }
        return DFlash2SpeculativeBlock(
            tokens: tokens, writes: writes, installedLayers: leadAt, contextRows: contextRows,
            stage: stage)
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

    /// True when `contextProjection` computes any 1...16 rows as the one
    /// 16-row product, the missing rows zero: the tensor kernel's padding, or
    /// `BONSAI_DRAFT_CONTEXT_PAD16`.
    public var contextProjectionPadsRows: Bool {
        DFlash2ContextPadding.enabled
            || (dtype == .bfloat16 && DFlash2TensorMatmul.padsShortInputs(fc))
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
    /// The hidden state after the leading layers when none were submitted
    /// before the readback (`DFlash2AdoptionStage`), else nil.
    let stage: MLXArray?

    init(
        tokens: MLXArray, writes: [(keys: MLXArray, values: MLXArray)], installedLayers: Int,
        contextRows: Int, stage: MLXArray? = nil
    ) {
        (self.tokens, self.writes, self.installedLayers, self.contextRows, self.stage) =
            (tokens, writes, installedLayers, contextRows, stage)
    }
}

enum DFlash2ContextPadding {
    static let enabled = ProcessInfo.processInfo.environment["BONSAI_DRAFT_CONTEXT_PAD16"] == "1"
}

/// The leading layers of an adopted next block, submitted at adoption.
///
/// A block built before the readback (`DFlash2DraftModel.proposeSpeculative`)
/// submits its leading layers before the readback only when one row class
/// fits every confirmed count; otherwise the whole block waited for the
/// round's deferred submission, behind the committed recurrent state's host
/// work, while the GPU sat idle from the verify's end. With this on, the
/// block keeps the hidden state after those same leading layers
/// (`CBv2MTPDraftBeforeReadback.leadingLayers`) and adoption submits it as
/// soon as the block's writes are installed, which is where today's block
/// path submits its own leading layers. Same kernels, same order, same
/// values: only a command-buffer boundary moves, and the load-time
/// speculation self-test (every confirmed count, bit for bit against
/// `finalizeRound` + `proposeBlock`) adopts through it.
/// `MLXFAST_DFLASH_ADOPT_STAGE=0` keeps the single deferred submission.
enum DFlash2AdoptionStage {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_ADOPT_STAGE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
}

/// Row classes, for the next block built before the readback, that hold by
/// construction instead of by probe.
///
/// `DFlash2DraftModel.contextRowClasses` probes once, with random rows, which
/// row counts project their leading rows to the same bits, and the block then
/// projects its class's count of verify rows. The probe compares BF16
/// outputs: two products that accumulate in different orders give BF16
/// results that agree on almost every element, so one probe can pass kernels
/// that do differ (on the M4's route-off matmuls, 15-row against 1-, 4- or
/// 6-row products), and the adopted block then differs from today's block in
/// some rounds (7 of 43 blocks on nonquote_poem route off; tokens and
/// acceptance unchanged there, the later draft positions not). With this on,
/// a class widens a count only when the projection is the same product by
/// construction:
/// - when `contextProjection` runs every count of 1...16 rows as the one
///   16-row product with the missing rows zero (the drafter tensor kernel's
///   padding, or `BONSAI_DRAFT_CONTEXT_PAD16`), the block zeroes the verify
///   rows at and past the device confirmed count and projects all 16: today's
///   padded product exactly, for every count. One class, so its leading
///   layers are submitted before the readback;
/// - otherwise every count is its own class: the block projects the previous
///   round's count and is adopted only when this round confirms the same.
/// The load-time speculation self-test proves every count bit for bit either
/// way. `MLXFAST_DFLASH_EXACT_ROW_CLASSES=0` keeps the probed classes.
enum DFlash2ExactRowClasses {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_EXACT_ROW_CLASSES"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `[0] + class per count 1...rows`, and whether the block zeroes its
    /// unconfirmed verify rows; nil when off (the probe decides).
    static func plan(drafter: DFlash2DraftModel, rows: Int) -> (classes: [Int], masked: Bool)? {
        guard enabled else { return nil }
        if drafter.contextProjectionPadsRows {
            return ([0] + Array(repeating: rows, count: rows), true)
        }
        return (Array(0 ... rows), false)
    }
}

/// Layer counts after which the drafter trunk `asyncEval`s its hidden state.
/// Default: after the first layer, so the GPU starts the block (it has been
/// idle since the verify readback) while the host builds the other layers and
/// the head; measured locally ~0.2-0.4% decode. `MLXFAST_DRAFT_SLICE_LAYERS`
/// overrides it with a `,`/`;` list of counts (a count equal to the layer
/// count submits the trunk before the head); `0`/`off` turns it off.
enum DFlash2DraftSubmission {
    /// The block a round proposes after its readback (the early block whose
    /// speculative twin was not adopted) submits its leading layers once they
    /// are built (`EngineLoopV2.earlyDraftLeadingLayers`, 3). The GPU sits idle
    /// from the verify's end until that first submission, and building the
    /// three layers delays it. With this on, the first of those layers (and
    /// the context projection it reads) is also submitted as soon as it is
    /// built. The leading submission still follows after the third layer, so
    /// the GPU never has less work queued than before. The same kernels run
    /// in the same order on the same inputs; only a command-buffer boundary
    /// moves earlier, so every value is bit-identical. On an M4 Max it
    /// commits the block's first command buffer 69 us sooner after the
    /// readback (median of 39 rounds). `MLXFAST_DRAFT_FIRST_LAYER_AHEAD=0`
    /// submits the leading layers together, as before.
    static let firstLayerAhead: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DRAFT_FIRST_LAYER_AHEAD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

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
/// `MLXFAST_ONE_LAUNCH_CONCAT=1` turns it on at every size.
///
/// Small joins (P6, default on): outputs of at most `smallLimit` elements —
/// the verify window's five tapped target states (`fc`'s input, five copy
/// launches per round) and the block's `[anchor; masks]` embedding (two) —
/// take the one launch once its self-tests pass and a serialized trial at
/// those two shapes (interleaved medians) finds it at least 1% faster than
/// `concatenated`. The prompt's 512-row taps stay on `concatenated`.
/// `MLXFAST_DFLASH_SMALL_CONCAT=0` keeps `concatenated` for them.
/// The speculative block's context rows in one launch. The record forms them
/// as four: the row compare against the confirmed count, a zero fill, the
/// select over the window's FP16 rows (819 KB written at 16 rows of 25,600),
/// then the cast to the drafter's BF16 that `contextProjection` reads (819 KB
/// read, 819 KB written). Here the same ops run as one compiled MLX chain
/// (compare, select, cast fused into one kernel; the zero is a host scalar).
/// Exact: the select moves values and the cast is the same conversion.
/// Self-tested at bind for every row count against the record's chain, bit
/// for bit: every FP16 bit pattern, every confirmed count from 0 to rows + 1.
/// A mismatch or an MLX error keeps the chain.
/// `MLXFAST_DFLASH_CONTEXT_ROWS_FUSED=0` keeps it.
enum DFlash2ContextRows {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_CONTEXT_ROWS_FUSED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
    nonisolated(unsafe) private(set) static var active = false

    private static let fused: @Sendable ([MLXArray]) -> [MLXArray] = compile { inputs in
        let index = broadcast(inputs[2], to: inputs[0].shape)
        return [which(index .< inputs[1], inputs[0], inputs[3]).asType(.bfloat16)]
    }

    /// The record's chain (FP16 out; `contextProjection` casts it).
    static func reference(_ rows: MLXArray, _ c: MLXArray) -> MLXArray {
        let n = rows.dim(1)
        return which(
            (MLXArray(Int32(0) ..< Int32(n)) .< c).reshaped([1, n, 1]),
            rows, MLXArray.zeros([1, 1, 1], dtype: rows.dtype))
    }

    /// The rows in BF16 from one launch, or nil (off, or another form).
    static func masked(_ rows: MLXArray, _ c: MLXArray, dtype: DType, force: Bool = false) -> MLXArray? {
        guard active || force, dtype == .bfloat16, rows.dtype == .float16, rows.ndim == 3,
            rows.dim(0) == 1, c.dtype == .int32, c.ndim == 0
        else { return nil }
        let n = rows.dim(1)
        return fused([rows, c, MLXArray(Array(Int32(0) ..< Int32(n)), [1, n, 1]), MLXArray(Float16(0))])[0]
    }

    static func prepare(width: Int, dtype: DType) {
        guard enabled, dtype == .bfloat16, width > 0 else { return }
        var same = true
        var values = 0
        do {
            try withError { error in
                for n in [16, 15, 8, 1] {
                    let count = n * width
                    let bits = (0 ..< count).map { UInt16(truncatingIfNeeded: $0 &* 40503) }
                    let rows = MLXArray(bits, [1, n, width]).view(dtype: .float16)
                    for k in 0 ... (n + 1) {
                        let c = MLXArray(Int32(k))
                        guard let y = masked(rows, c, dtype: dtype, force: true) else { same = false; continue }
                        let r = reference(rows, c).asType(dtype)
                        same = same && y.dtype == r.dtype && y.shape == r.shape
                            && all(y.view(dtype: .uint16) .== r.view(dtype: .uint16)).item(Bool.self)
                        values += count
                    }
                }
                try error.check()
            }
        } catch {
            same = false
        }
        active = same
        FileHandle.standardError.write(
            ("dflash2 context rows in one launch: "
                + (same
                    ? "self-test passed: \(values) values identical (every FP16 pattern, rows 16/15/8/1); on\n"
                    : "self-test failed; four-launch chain kept\n")).data(using: .utf8)!)
    }
}

enum DFlash2Concat {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_ONE_LAUNCH_CONCAT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    static let smallEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SMALL_CONCAT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static let smallLimit = 1 << 20

    /// Set once at bind by `raceSmall`: 0 `concatenated`, 1 `kernel`, 2
    /// `quadKernel`.
    nonisolated(unsafe) private(set) static var smallForm = 0

    /// Whether an input of the block embedding must be row-contiguous for
    /// the one launch (it is, whenever the launch can run).
    static var mayLaunch: Bool { enabled || smallForm > 0 }

    nonisolated(unsafe) private static var quadKernels: [Int: MLXFast.MLXFastKernel] = [:]

    /// `kernel(n)` four consecutive elements per thread on an
    /// `(total / 4, outer)` grid, for inputs whose inner sizes are all
    /// multiples of four (no quad straddles two inputs): the same copies.
    /// With `cast0` (ROUNDFUSE2) input 0 is FP16 and each of its elements is
    /// written as `v_copy` casts it, `static_cast<bfloat16_t>`, to the BF16 out.
    private static func quadKernel(_ n: Int, cast0: Bool = false) -> MLXFast.MLXFastKernel {
        kernelLock.withLock {
            if let kernel = quadKernels[cast0 ? -n : n] { return kernel }
            var source = """
                const uint o = thread_position_in_grid.y;
                const uint total = uint(dims[1]);
                uint j = thread_position_in_grid.x * 4;
                if (j >= total) { return; }
                device auto* d = out + size_t(o) * total + j;

                """
            for i in 0 ..< n {
                let x = (0 ..< 4).map { cast0 && i == 0 ? "static_cast<bfloat16_t>(x[\($0)])" : "x[\($0)]" }
                source += "{ const uint w = uint(dims[\(i + 2)]);\n"
                source += "if (j < w) { const device auto* x = x\(i) + size_t(o) * w + j;\n"
                source += "d[0] = \(x[0]); d[1] = \(x[1]); d[2] = \(x[2]); d[3] = \(x[3]); return; }\n"
                source += "j -= w; }\n"
            }
            let name = "dflash2_concat_quad\(n)" + (cast0 ? "_anchor" : "")
            let kernel = MLXFast.metalKernel(
                name: name,
                inputNames: (0 ..< n).map { "x\($0)" } + ["dims"],
                outputNames: ["out"], source: Qwen35IO32.narrow(source, count: n + 1, name),
                ensureRowContiguous: true)
            quadKernels[cast0 ? -n : n] = kernel
            return kernel
        }
    }

    private static func quadLaunch(_ parts: [MLXArray], axis: Int, cast0: Bool = false) -> MLXArray? {
        var shape = parts[0].shape
        let outer = shape[..<axis].reduce(1, *)
        let inners = parts.map { $0.shape[axis...].reduce(1, *) }
        guard inners.allSatisfy({ $0 % 4 == 0 }) else { return nil }
        let total = inners.reduce(0, +)
        shape[axis] = parts.reduce(0) { $0 + $1.dim(axis) }
        let dims = MLXArray(([outer, total] + inners).map { Int32($0) })
        return quadKernel(parts.count, cast0: cast0)(
            parts + [dims], grid: (total / 4, outer, 1),
            threadGroup: (min(256, total / 4), 1, 1), outputShapes: [shape],
            outputDTypes: [cast0 ? .bfloat16 : parts[0].dtype])[0]
    }

    private static let kernelLock = NSLock()
    nonisolated(unsafe) private static var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]

    /// Input `i` is `[outer, inner_i]` row-major, the output `[outer, total]`;
    /// `dims` = `[outer, total, inner_0, ...]` at run time, so one pipeline per
    /// input count and dtype serves every shape (a timed round never builds one).
    /// `cast0` as `quadKernel`'s.
    private static func kernel(_ n: Int, cast0: Bool = false) -> MLXFast.MLXFastKernel {
        kernelLock.withLock {
            if let kernel = kernels[cast0 ? -n : n] { return kernel }
            var source = """
                const uint idx = thread_position_in_grid.x;
                const uint total = uint(dims[1]);
                if (idx >= uint(dims[0]) * total) { return; }
                const uint o = idx / total;
                uint j = idx - o * total;

                """
            for i in 0 ..< n {
                source += "{ const uint w = uint(dims[\(i + 2)]);\n"
                let x = cast0 && i == 0 ? "static_cast<bfloat16_t>(x0[o * w + j])" : "x\(i)[o * w + j]"
                source += "if (j < w) { out[idx] = \(x); return; }\n"
                source += "j -= w; }\n"
            }
            let kernel = MLXFast.metalKernel(
                name: "dflash2_concat\(n)" + (cast0 ? "_anchor" : ""),
                inputNames: (0 ..< n).map { "x\($0)" } + ["dims"],
                outputNames: ["out"], source: source, ensureRowContiguous: true)
            kernels[cast0 ? -n : n] = kernel
            return kernel
        }
    }

    private static func launch(_ parts: [MLXArray], axis: Int, cast0: Bool = false) -> MLXArray {
        var shape = parts[0].shape
        let outer = shape[..<axis].reduce(1, *)
        let inners = parts.map { $0.shape[axis...].reduce(1, *) }
        let total = inners.reduce(0, +)
        shape[axis] = parts.reduce(0) { $0 + $1.dim(axis) }
        let dims = MLXArray(([outer, total] + inners).map { Int32($0) })
        let threads = outer * total
        return kernel(parts.count, cast0: cast0)(
            parts + [dims], grid: (threads, 1, 1),
            threadGroup: (min(256, threads), 1, 1), outputShapes: [shape],
            outputDTypes: [cast0 ? .bfloat16 : parts[0].dtype])[0]
    }

    /// `concatenated(parts, axis: axis)`, in one launch where it applies.
    static func concatenate(_ parts: [MLXArray], axis: Int) -> MLXArray {
        // The small joins run only on forms whose self-tests already ran (at
        // bind): never a self-test inside a timed forward.
        let small = !enabled && smallForm > 0
            && parts.reduce(0, { $0 + $1.size }) <= smallLimit
        guard enabled || small, parts.count >= 2, parts.count <= 8 else {
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
            parts.reduce(0, { $0 + $1.size }) < Int(Int32.max)
        else { return concatenated(parts, axis: axis) }
        if small {
            let key = "\(parts.count) \(first.dtype)"
            let (plain, quad) = lock.withLock { (verdicts[key] == true, quadVerdicts[key] == true) }
            if smallForm == 2, quad, let joined = quadLaunch(parts, axis: ax) { return joined }
            return smallForm == 1 && plain ? launch(parts, axis: ax) : concatenated(parts, axis: axis)
        }
        guard verified(parts.count, first.dtype) else { return concatenated(parts, axis: axis) }
        return launch(parts, axis: ax)
    }

    /// `concatenate([anchor.asType(.bfloat16), masks], axis: 1)` for an FP16
    /// `[B, 1, H]` anchor row and BF16 `[B, cols, H]` mask rows, in the launch
    /// `concatenate` would make for the cast row, its `cast0` twin reading the
    /// FP16 row (ROUNDFUSE2); nil where `concatenate` would not launch one.
    static func anchorJoin(_ anchor: MLXArray, _ masks: MLXArray) -> MLXArray? {
        guard anchor.dtype == .float16, masks.dtype == .bfloat16, anchor.ndim == 3, masks.ndim == 3,
            anchor.dim(1) == 1, anchor.dim(0) == masks.dim(0), anchor.dim(2) == masks.dim(2),
            anchor.size + masks.size < Int(Int32.max)
        else { return nil }
        let parts = [anchor, masks]
        let small = !enabled && smallForm > 0 && anchor.size + masks.size <= smallLimit
        if small {
            let key = "2 \(DType.bfloat16)"
            let (plain, quad) = lock.withLock { (verdicts[key] == true, quadVerdicts[key] == true) }
            if smallForm == 2 { return quad ? quadLaunch(parts, axis: 1, cast0: true) : nil }
            return smallForm == 1 && plain ? launch(parts, axis: 1, cast0: true) : nil
        }
        return enabled && verified(2, .bfloat16) ? launch(parts, axis: 1, cast0: true) : nil
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
        guard enabled || smallEnabled else { return }
        _ = verified(inputs, dtype)
    }

    /// The small joins' trial (load time, after their self-tests): `taps`
    /// `[1, rows, width]` FP16 states along the last axis and a
    /// `[1, 1, hidden]` + `[1, rows - 1, hidden]` embedding of `dtype`.
    static func raceSmall(taps: Int, rows: Int, width: Int, hidden: Int, dtype: DType) {
        guard smallEnabled, !enabled, taps >= 2, taps <= 8, rows >= 2,
            verified(taps, .float16), verified(2, dtype)
        else { return }
        let states = (0 ..< taps).map {
            MLXRandom.normal([1, rows, width], key: MLXRandom.key(UInt64(95 + $0))).asType(.float16)
        }
        let anchor = MLXRandom.normal([1, 1, hidden], key: MLXRandom.key(94)).asType(dtype)
        let masks = MLXRandom.normal([1, rows - 1, hidden], key: MLXRandom.key(93)).asType(dtype)
        eval(states + [anchor, masks])
        let quads = quadVerified(taps, .float16) && quadVerified(2, dtype)
        var builders: [() -> [MLXArray]] = [
            { [concatenated(states, axis: -1), concatenated([anchor, masks], axis: 1)] },
            { [launch(states, axis: 2), launch([anchor, masks], axis: 1)] },
        ]
        if quads {
            builders.append {
                [quadLaunch(states, axis: 2), quadLaunch([anchor, masks], axis: 1)].compactMap { $0 }
            }
        }
        let t = DFlash2LaunchTrial.race(builders)
        smallForm = 0
        if t.count == builders.count, let best = (1 ..< t.count).min(by: { t[$0] < t[$1] }),
            t[best] <= t[0] * DFlash2LaunchTrial.tolerance
        {
            smallForm = best
        }
        FileHandle.standardError.write(
            (t.count == builders.count
                ? String(
                    format: "dflash2 small one-launch concat trial (%ld taps of [%ld, %ld] and the "
                        + "block embedding, 5 rounds): %.1f us concatenated, %.1f us one launch, "
                        + "%@ us four per thread; %@\n",
                    taps, rows, width, t[0], t[1], quads ? String(format: "%.1f", t[2]) : "-",
                    ["concatenated kept", "one launch", "one launch, four per thread"][smallForm])
                : "dflash2 small one-launch concat trial failed; concatenated kept\n")
                .data(using: .utf8)!)
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

    nonisolated(unsafe) private static var quadVerdicts: [String: Bool] = [:]

    /// `quadKernel`'s self-test, as `verified`'s on shapes whose inner sizes
    /// are multiples of four.
    private static func quadVerified(_ n: Int, _ dtype: DType) -> Bool {
        lock.withLock {
            let key = "\(n) \(dtype)"
            if let verdict = quadVerdicts[key] { return verdict }
            var same = true
            var compared = 0
            do {
                try withError { error in
                    let bits: DType = dtype == .float32 ? .uint32 : .uint16
                    for (seed, axis) in [(73, 1), (74, 2)] {
                        let parts = (0 ..< n).map { i -> MLXArray in
                            var shape = [2, 3, 40]
                            shape[axis] = [4, 8, 24, 16, 12, 28, 40, 4][i]
                            return (MLXRandom.normal(shape, key: MLXRandom.key(UInt64(seed * 16 + i)))
                                * 1000).asType(dtype)
                        }
                        guard let fused = quadLaunch(parts, axis: axis) else {
                            same = false
                            continue
                        }
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
            quadVerdicts[key] = same
            FileHandle.standardError.write(
                ("dflash2 one-launch concat, four per thread (\(key)): "
                    + (same
                        ? "self-test passed: \(compared) values compared bitwise, 0 mismatches\n"
                        : "self-test failed\n")).data(using: .utf8)!)
            return same
        }
    }
}

// MARK: - Round fusions 2

/// ROUNDFUSE2: two launches fewer in every decode round's drafter work, each
/// exact by construction and self-tested once per process at bind, bit for
/// bit against the chain it replaces; a mismatch or an MLX error keeps that
/// chain. `MLXFAST_ROUND_FUSE2=0` keeps both chains.
/// - The speculative block's key mask `arange(keys) .< (bound + c)` (an
///   `ss_Add`, then a `vs_Less`) is `arange(-bound, keys - bound) .< c`: the
///   bound is folded into the host range, so every key meets the same int32
///   comparison (small integers, far from overflow) in one launch.
/// - The block embedding's FP16 anchor row was cast to BF16 (a `v_copy`) and
///   then joined with the mask rows; the join's launch now reads the FP16 row
///   and casts each element as `v_copy` does (`DFlash2Concat.anchorJoin`).
enum DFlash2RoundFuse2 {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_ROUND_FUSE2"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Set once at bind by `prepare`.
    nonisolated(unsafe) private(set) static var maskReady = false
    nonisolated(unsafe) private(set) static var anchorReady = false
    nonisolated(unsafe) private static var prepared = false
    private static let lock = NSLock()

    /// The block's key mask: `i < bound + c` for `i` in `0 ..< keys` and a
    /// 0-d int32 `c`, folded (`fused`) or as the chain forms it.
    static func keyMask(keys: Int, bound: Int, _ c: MLXArray, fused: Bool) -> MLXArray {
        fused
            ? MLXArray(Int32(-bound) ..< Int32(keys - bound)) .< c
            : MLXArray(Int32(0) ..< Int32(keys)) .< (MLXArray(Int32(bound)) + c)
    }

    /// Both self-tests, at bind after `DFlash2Concat.raceSmall` (the join
    /// takes the form it chose). The mask: 0 to 2,000 held rows, blocks of 1
    /// to 16 rows, every count from -1 to the block size plus one. The join:
    /// all 65,536 FP16 bit patterns as anchor rows of `hidden`, beside the
    /// production mask block and beside random BF16 rows.
    static func prepare(hidden: Int, maskRow: MLXArray) {
        lock.withLock {
            guard enabled, !prepared else { return }
            prepared = true
            var same = true
            var compared = 0
            do {
                try withError { error in
                    var equal: [MLXArray] = []
                    for (rows, block) in [(0, 16), (1, 16), (539, 16), (2000, 16), (100, 8), (7, 1)] {
                        let keys = rows + 2 * block
                        for count in -1 ... block + 1 {
                            let c = MLXArray(Int32(count))
                            let fused = keyMask(keys: keys, bound: rows + block, c, fused: true)
                            let chain = keyMask(keys: keys, bound: rows + block, c, fused: false)
                            same = same && fused.shape == chain.shape && fused.dtype == chain.dtype
                            equal.append(fused.view(dtype: .uint8) .== chain.view(dtype: .uint8))
                            compared += keys
                        }
                    }
                    same = same && all(concatenated(equal)).item(Bool.self)
                    try error.check()
                }
            } catch {
                same = false
            }
            maskReady = same
            FileHandle.standardError.write(
                ("dflash2 block key mask (ROUNDFUSE2): "
                    + (same
                        ? "self-test passed: \(compared) mask bits compared bitwise, 0 mismatches; one launch\n"
                        : "self-test failed; the add and the compare kept\n")).data(using: .utf8)!)

            var joined = maskRow.dtype == .bfloat16 && hidden >= 1
            var launched = joined
            compared = 0
            do {
                try withError { error in
                    guard joined else { return }
                    let rows = (65536 + hidden - 1) / hidden
                    let anchors = MLXArray((0 ..< rows * hidden).map { UInt16(truncatingIfNeeded: $0) })
                        .view(dtype: .float16).reshaped([rows, 1, hidden])
                    let block = contiguous(broadcast(maskRow, to: [1, 15, hidden]))
                    let random = MLXRandom.normal([1, 15, hidden], key: MLXRandom.key(91)).asType(.bfloat16)
                    eval([anchors, block, random])
                    for r in 0 ..< rows {
                        let anchor = anchors[r ..< r + 1]
                        let masks = r % 2 == 0 ? block : random
                        guard let fused = DFlash2Concat.anchorJoin(anchor, masks) else {
                            (joined, launched) = (false, false)
                            return
                        }
                        let chain = DFlash2Concat.concatenate([anchor.asType(.bfloat16), masks], axis: 1)
                        joined = joined && fused.shape == chain.shape && fused.dtype == chain.dtype
                            && all(fused.view(dtype: .uint16) .== chain.view(dtype: .uint16)).item(Bool.self)
                        compared += chain.size
                    }
                    try error.check()
                }
            } catch {
                joined = false
            }
            anchorReady = joined
            FileHandle.standardError.write(
                ("dflash2 anchor row join (ROUNDFUSE2): "
                    + (joined
                        ? "self-test passed: \(compared) values compared bitwise (every FP16 bit pattern "
                            + "as an anchor element), 0 mismatches; the join reads the FP16 row\n"
                        : launched || compared > 0
                            ? "self-test failed; the cast kept\n"
                            : "the join is not one launch here; the cast kept\n")).data(using: .utf8)!)
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
        source: source,
        ensureRowContiguous: true)

    /// The kernel body, shared with `DFlash2SpeculativeFront`'s variant.
    static let source = """
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
            """

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

// MARK: - Context keys and values in one launch

/// The absorbed context's k head norm, its rope and the first append of the
/// keys and values into a fresh cache (four launches per layer: MLX's
/// `RMSNorm` first copies the strided k view) as ONE launch that writes the
/// cache's two `[1, kvHeads, capacity, D]` buffers. Each simdgroup takes one
/// k head row through `DFlash2StridedRMSNorm`'s body (MLX's `rms_single_row`:
/// reads, FP32 sum, `simd_sum`s, `precise::rsqrt`, both roundings), then
/// MLX's `rope` body on the rounded row as `DFlash2QKPrework` runs it (the
/// other half of each pair from lane `^ 16`, one rounding), and copies the
/// same v head row; rows past the context stay unwritten, as the first
/// append leaves them. Self-tested at bind, bit for bit, against the chain it
/// replaces (`kNorm` of every layer, the drafter's rope, a fresh cache's
/// `updateBlock`) at 512, 513 and 7 context rows (the k|v-only product and
/// the full q|k|v stack), rope offsets 0, 37 and 4077, rows from 1e-3 to 3e3
/// and a zero row; a mismatch or an MLX error keeps the chain.
/// `MLXFAST_DFLASH_ABSORB_FUSED=0` keeps it.
enum DFlash2AbsorbKV {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_ABSORB_FUSED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "dflash2_absorb_kv",
        inputNames: ["k", "v", "w", "p", "pos"],
        outputNames: ["ko", "vo"],
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

            const uint H = uint(k_shape[2]);
            const uint t = gid / H;
            const uint h = gid % H;
            const int64_t kr = int64_t(t) * k_strides[1] + int64_t(h) * k_strides[2];
            const int64_t vr = int64_t(t) * v_strides[1] + int64_t(h) * v_strides[2];
            const device auto* wr = w + lid * N_READS;

            float acc = 0;
            float thread_x[N_READS];
            for (int i = 0; i < N_READS; i++) {
              thread_x[i] = k[kr + int64_t(lid * N_READS + i) * k_strides[3]];
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
            float Lp = p[2] * static_cast<float>(t + uint(pos[0]));
            const size_t o = (size_t(h) * uint(pos[1]) + t) * D + lid * N_READS;
            for (int i = 0; i < N_READS; i++) {
              const T nv = wr[i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
              const float xv = static_cast<float>(nv);
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
              ko[o + i] = static_cast<T>(lo ? rx1 : rx2);
              vo[o + i] = v[vr + int64_t(lid * N_READS + i) * v_strides[3]];
            }
            """,
        ensureRowContiguous: false)

    private static let lock = NSLock()
    /// `[eps, log2(base), scale]` per self-tested geometry.
    nonisolated(unsafe) private static var ready: [String: MLXArray] = [:]

    private static func launch(
        _ k: MLXArray, _ v: MLXArray, _ weight: MLXArray, _ p: MLXArray, offset: Int, capacity: Int
    ) -> (MLXArray, MLXArray) {
        let (n, h, d) = (k.dim(1), k.dim(2), k.dim(3))
        let out = kernel(
            [k, v, weight, p, MLXArray([Int32(offset), Int32(capacity)])],
            template: [("T", k.dtype), ("D", d)],
            grid: (32 * n * h, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[1, h, capacity, d], [1, h, capacity, d]],
            outputDTypes: [k.dtype, k.dtype])
        return (out[0], out[1])
    }

    /// The two `[1, H, capacity, D]` buffers a fresh cache's first append
    /// leaves for the k and v head rows `k`, `v` (`[1, n, H, D]` views) after
    /// the k norm and the rope at `offset`, or nil when the launch does not
    /// apply.
    static func apply(
        _ k: MLXArray, _ v: MLXArray, kNorm: RMSNorm, offset: Int, capacity: Int
    ) -> (MLXArray, MLXArray)? {
        guard enabled, k.ndim == 4, k.shape == v.shape, k.dim(0) == 1, k.dim(1) >= 1,
            k.dim(3) == 128, v.dtype == k.dtype, kNorm.weight.shape == [k.dim(3)],
            kNorm.weight.dtype == k.dtype, offset >= 0, capacity >= k.dim(1),
            offset + k.dim(1) < Int(Int32.max), k.dim(2) * capacity * k.dim(3) < Int(Int32.max),
            let p = lock.withLock({ ready["\(k.dtype) \(k.dim(2)) \(kNorm.eps)"] })
        else { return nil }
        return launch(k, v, kNorm.weight, p, offset: offset, capacity: capacity)
    }

    /// The self-test for the drafter's geometry, rope and k norms, at bind.
    static func prepare(
        rope: RoPELayer, base: Float, dtype: DType, heads hq: Int, kvHeads hk: Int,
        headDim d: Int, kNorms: [RMSNorm], cacheSize: Int
    ) {
        guard enabled, d == 128, hk > 0, cacheSize >= 513, let eps = kNorms.first?.eps,
            kNorms.allSatisfy({ $0.eps == eps && $0.weight.shape == [d] && $0.weight.dtype == dtype }),
            [DType.bfloat16, .float16].contains(dtype)
        else { return }
        lock.withLock {
            let key = "\(dtype) \(hk) \(eps)"
            guard ready[key] == nil else { return }
            let p = MLXArray([eps, log2(base), 1])
            let w = hk * d
            var same = true
            var compared = 0
            do {
                try withError { error in
                    // (rows, rope offset, k column in the product): above 256
                    // rows the k|v-only product, at or below the full stack.
                    let cases = [(512, 0, 0), (513, 0, 0), (7, 0, hq * d), (512, 37, 0), (7, 4077, hq * d)]
                    for (index, (n, off, k0)) in cases.enumerated() {
                        let heads = (k0 + 2 * w) / d
                        let row = MLXArray(0 ..< n * heads).reshaped(n, heads, 1)
                        // Head rows from 1e-3 to 3e3 and a zero k row.
                        let scale = exp(MLXRandom.uniform(
                            Float(-7) ..< Float(8), [n, heads, 1], key: MLXRandom.key(UInt64(91 + index))))
                            * (row .!= MLXArray(Int32((n - 1) * heads + k0 / d + 1))).asType(.float32)
                        let y = (MLXRandom.normal(
                            [n, heads, d], key: MLXRandom.key(UInt64(191 + index))) * scale)
                            .asType(dtype).reshaped(1, n, heads * d)
                        let kRows = y[0..., 0..., k0 ..< (k0 + w)].reshaped(1, n, hk, d)
                        let vRows = y[0..., 0..., (k0 + w)...].reshaped(1, n, hk, d)
                        let kNorm = kNorms[index % kNorms.count]
                        let live = DFlash2BlockKVCache(maxSize: cacheSize, keep: 0)
                        let fused = DFlash2BlockKVCache(maxSize: cacheSize, keep: 0)
                        guard
                            let (lk, lv) = live.updateBlock(
                                keys: rope(
                                    DFlash2StridedRMSNorm.apply(kNorm, kRows).transposed(0, 2, 1, 3),
                                    offset: off),
                                values: vRows.transposed(0, 2, 1, 3), contextRows: n),
                            let capacity = fused.firstAppendCapacity(contextRows: n)
                        else {
                            same = false
                            break
                        }
                        let (fk, fv) = launch(kRows, vRows, kNorm.weight, p, offset: off, capacity: capacity)
                        same = same && fused.installFirst(keys: fk, values: fv, contextRows: n)
                            && fused.offset == live.offset && fused.inPlaceRows == live.inPlaceRows
                            && fused.inPlaceCapacity == live.inPlaceCapacity
                        for (f, l) in [(fk, lk), (fv, lv)] {
                            same = same && l.shape == [1, hk, n, d]
                                && all(f[.ellipsis, ..<n, 0...].view(dtype: .uint16) .== l.view(dtype: .uint16))
                                    .item(Bool.self)
                            compared += l.size
                        }
                    }
                    try error.check()
                }
            } catch {
                same = false
            }
            if same { ready[key] = p }
            FileHandle.standardError.write(
                ("dflash2 absorbed context K/V install (\(key)): "
                    + (same
                        ? "self-test passed: \(compared) values compared bitwise, 0 mismatches; one launch\n"
                        : "self-test failed; norm, rope and first append kept\n")).data(using: .utf8)!)
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
/// Default on (P6): at bind, after the self-test, a serialized trial at the
/// block's shape (five layers' products, interleaved medians) keeps the two
/// launches unless a one-launch form measures at least 1% faster.
/// `MLXFAST_DFLASH_SWIGLU=0` keeps them.
enum DFlash2SwiGLU {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SWIGLU"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    /// The trial's verdict per dtype and width (set at bind by `prepare`):
    /// 0 the two launches, 1 `kernel`, 2 and 3 `quadKernel` at 64 and 256
    /// threads per threadgroup.
    nonisolated(unsafe) private static var raced: [String: Int] = [:]

    /// `kernel`'s element, four consecutive columns per thread on a
    /// `(columns / 4, rows)` grid (no per-element division); the same
    /// operations on the same values, so the same bits.
    private static let quadKernel = MLXFast.metalKernel(
        name: "dflash2_swiglu_quad",
        inputNames: ["g", "u"],
        outputNames: ["out"],
        source: """
            const uint q = thread_position_in_grid.x;
            const uint r = thread_position_in_grid.y;
            const uint N = uint(g_shape[2]);
            const device OutT* gr = g + int64_t(r) * g_strides[1];
            const device OutT* ur = u + int64_t(r) * u_strides[1];
            device OutT* o = out + size_t(r) * N;
            for (uint i = 0; i < 4; ++i) {
              const uint c = q * 4 + i;
              if (c < N) {
                const OutT gv = gr[int64_t(c) * g_strides[2]];
                const OutT uv = ur[int64_t(c) * u_strides[2]];
                const OutT sg = SigmoidMLX()(gv);
                const OutT act = MultiplyMLX()(gv, sg);
                o[c] = MultiplyMLX()(act, uv);
              }
            }
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

    private static func quadLaunch(_ g: MLXArray, _ u: MLXArray, threads: Int) -> MLXArray {
        let (r, n) = (g.dim(1), g.dim(2))
        return quadKernel(
            [g, u], template: [("OutT", g.dtype)],
            grid: ((n + 3) / 4, r, 1), threadGroup: (threads, 1, 1),
            outputShapes: [g.shape], outputDTypes: [g.dtype])[0]
    }

    private static func launch(_ g: MLXArray, _ u: MLXArray, form: Int) -> MLXArray {
        form == 1 ? launch(g, u) : quadLaunch(g, u, threads: form == 2 ? 64 : 256)
    }

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
            let form = lock.withLock({ raced["\(g.dtype) \(g.dim(2))"] }), form > 0,
            verified(g.dtype, width: g.dim(2))
        else { return silu(g) * u }
        return launch(g, u, form: form)
    }

    /// The self-test, then the trial at `rows` rows (load time).
    static func prepare(dtype: DType, width: Int, rows: Int = 16) {
        guard enabled, [DType.bfloat16, .float16].contains(dtype),
            verified(dtype, width: width)
        else { return }
        let stacked = MLXRandom.normal([1, rows, 2 * width], key: MLXRandom.key(93)).asType(dtype)
        eval(stacked)
        let g = stacked[.ellipsis, ..<width]
        let u = stacked[.ellipsis, width...]
        let t = DFlash2LaunchTrial.race(
            [{ [silu(g) * u] }] + (1 ... 3).map { form in { [launch(g, u, form: form)] } })
        var form = 0
        if t.count == 4, let best = (1 ... 3).min(by: { t[$0] < t[$1] }),
            t[best] <= t[0] * DFlash2LaunchTrial.tolerance
        {
            form = best
        }
        lock.withLock { raced["\(dtype) \(width)"] = form }
        let names = ["two launches kept", "one launch", "one launch, 4 columns x 64", "one launch, 4 columns x 256"]
        FileHandle.standardError.write(
            (t.count == 4
                ? String(
                    format: "dflash2 one-launch SwiGLU trial (%@ %ld, 5 layers): %.1f us two launches, "
                        + "%.1f us one launch, %.1f/%.1f us 4 columns x 64/256; %@\n", "\(dtype)",
                    width, t[0], t[1], t[2], t[3], names[form])
                : "dflash2 one-launch SwiGLU trial failed; two launches kept\n").data(using: .utf8)!)
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
                        let reference = silu(g) * u
                        for fused in [launch(g, u), quadLaunch(g, u, threads: 64)] {
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
                ("dflash2 one-launch SwiGLU (\(key)): "
                    + (same
                        ? "self-test passed: \(compared) values compared bitwise, 0 mismatches; one launch\n"
                        : "self-test failed; silu(g) * u kept\n")).data(using: .utf8)!)
            return same
        }
    }
}

// MARK: - The speculative block's front from the device count

/// The layer front of the block built before the readback
/// (`DFlash2DraftModel.proposeSpeculative`), whose rows sit at a DEVICE row
/// `c`. Per layer MLX copied `base` (`[context; zeros]`) and wrote the tap-0
/// convolution into it at `c` (a copy, an offset launch and a dynamic copy),
/// then sliced the block's q rows back out of the stacked q|k|v product (an
/// offset launch and a dynamic copy), copied the q and k column views
/// contiguous, and ran two norms and two ropes: eleven launches around the
/// q|k|v matmul. Here two launches do that work:
/// - `joinKernel`: `dflash2_grouped_conv`'s element verbatim for rows
///   `c ..< c + L`, the context's rows below `C` and zeros elsewhere, i.e.
///   `dynamicSliceUpdate([context; zeros], conv, c)` in one pass;
/// - `headsKernel`: `DFlash2QKPrework`'s norm+rope body with the q rows read
///   at `c + t` and rotated at `offset + c + t` (the block's device query
///   offset), the k rows at `offset + t`.
/// Both are copies plus the same per-element arithmetic, so their outputs
/// are the composed ops' bit for bit. At bind: a self-test against the
/// composed ops exactly as `DFlash2Attention.speculative` issues them (every
/// count 1...16 for the rows, three counts and offsets for the heads), then
/// a serialized trial at the drafter's shapes (interleaved medians of five
/// layers' fronts); a mismatch, an MLX error or a slower front keeps the
/// composed ops. `MLXFAST_DFLASH_SPEC_FRONT=0` keeps them.
enum DFlash2SpeculativeFront {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPEC_FRONT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let joinKernel = MLXFast.metalKernel(
        name: "dflash2_grouped_conv_join_at",
        inputNames: ["h", "dyn", "base", "ctx", "cdev"],
        outputNames: ["out"],
        source: Qwen35IO32.narrow("""
            const uint c = thread_position_in_grid.x * (QUAD ? 4u : 1u);
            const uint r = thread_position_in_grid.y;
            const uint b = thread_position_in_grid.z;
            const uint H = threads_per_grid.x * (QUAD ? 4u : 1u);
            const uint R = threads_per_grid.y;
            const uint L = uint(h_shape[1]);
            const uint C = uint(ctx_shape[1]);
            const uint s = metal::min(uint(cdev[0]), R - L);
            if constexpr (QUAD) {
              uint2 v;
              if (r >= s && r - s < L) {
                v = dflash2_grouped_conv4<T, KS, GS, TAP>(h, dyn, base, b, r - s, c, L, H);
              } else if (r < C) {
                v = *((const device uint2*)(ctx + (size_t(b) * C + r) * H + c));
              } else { v = uint2(0u); }
              *((device uint2*)(out + (size_t(b) * R + r) * H + c)) = v;
            } else {
              T v;
              if (r >= s && r - s < L) {
                v = dflash2_grouped_conv<T, KS, GS, TAP>(h, dyn, base, b, r - s, c, L, H);
              } else if (r < C) {
                v = ctx[(size_t(b) * C + r) * H + c];
              } else { v = static_cast<T>(0.0f); }
              out[(size_t(b) * R + r) * H + c] = v;
            }
            """, count: 4, "dflash2_grouped_conv_join_at"),
        header: dflash2GroupedConvHeaderIO32,
        ensureRowContiguous: true)

    /// `DFlash2QKPrework.source` with the q rows at `cdev + t`; nil when its
    /// text no longer has the lines the variant rewrites.
    private static let headsKernel: MLXFast.MLXFastKernel? = {
        var source = DFlash2QKPrework.source
        let edits: [(String, String)] = [
            (
                "const uint BL = uint(pos[2]);",
                "const uint BL = uint(pos[2]);\n            const uint qs = uint(cdev[0]);"
            ),
            ("(isQ ? n - BL + t : t)", "(isQ ? qs + t : t)"),
            (
                "static_cast<float>(t + uint(pos[isQ ? 0 : 1]))",
                "static_cast<float>(t + uint(pos[isQ ? 0 : 1]) + (isQ ? qs : 0u))"
            ),
        ]
        for (from, to) in edits {
            guard source.components(separatedBy: from).count == 2 else { return nil }
            source = source.replacingOccurrences(of: from, with: to)
        }
        return MLXFast.metalKernel(
            name: "dflash2_qk_prework_at", inputNames: ["y", "qw", "kw", "p", "pos", "cdev"],
            outputNames: ["q", "k"], source: Qwen35IO32.narrow(source, count: 2, "dflash2_qk_prework_at"),
            ensureRowContiguous: true)
    }()

    private static let lock = NSLock()
    /// Set once at bind by `prepare`: the rows launch and the heads launch
    /// passed their self-tests and the trial.
    nonisolated(unsafe) private(set) static var rowsChosen = false
    nonisolated(unsafe) private(set) static var headsChosen = false

    /// Either one-launch form is in use.
    static var active: Bool { rowsChosen || headsChosen }

    /// Back to the composed front (the speculation self-test failed with it).
    static func deactivate() {
        lock.withLock { (rowsChosen, headsChosen) = (false, false) }
    }
    /// `[eps, log2(base), scale]` of the drafter's head geometry.
    nonisolated(unsafe) private static var parameters: MLXArray?
    nonisolated(unsafe) private static var geometry: (hq: Int, hk: Int, d: Int, eps: Float)?

    private static func joinLaunch(
        _ h: MLXArray, _ dyn: MLXArray, _ base: MLXArray, _ ctx: MLXArray, _ c: MLXArray,
        rows: Int, ks: Int, gs: Int
    ) -> MLXArray {
        let (b, hs) = (h.dim(0), h.dim(2))
        return joinKernel(
            [h, dyn, base, ctx, c.reshaped([1])],
            template: [("T", h.dtype), ("KS", ks), ("GS", gs), ("TAP", 0), ("QUAD", dflash2ConvQuad(h.dtype, gs))],
            grid: (dflash2ConvColumns(hs, h.dtype, gs), rows, b), threadGroup: (256, 1, 1),
            outputShapes: [[b, rows, hs]], outputDTypes: [h.dtype])[0]
    }

    private static func headsLaunch(
        _ kernel: MLXFast.MLXFastKernel, _ y: MLXArray, _ l: Int, _ hq: Int, _ hk: Int,
        _ qw: MLXArray, _ kw: MLXArray, _ p: MLXArray, _ pos: MLXArray, _ c: MLXArray
    ) -> (MLXArray, MLXArray) {
        let (b, n, d) = (y.dim(0), y.dim(1), qw.dim(0))
        let out = kernel(
            [y, qw, kw, p, pos, c.reshaped([1])],
            template: [("T", y.dtype), ("D", d), ("HQ", hq), ("HK", hk)],
            grid: (32 * b * (l * hq + n * hk), 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[b, hq, l, d], [b, hk, n, d]], outputDTypes: [y.dtype, y.dtype])
        return (out[0], out[1])
    }

    /// `dynamicSliceUpdate([context; zeros], update: conv0(hidden), start: c)`
    /// over `rows` rows, or nil (the caller keeps the composed ops).
    static func rows(
        _ hidden: MLXArray, projection: MLXArray, base: MLXArray, context: MLXArray,
        confirmed c: MLXArray, rows: Int, kernelSize ks: Int, groupSize gs: Int
    ) -> MLXArray? {
        guard rowsChosen, hidden.ndim == 3, context.ndim == 3, hidden.dim(0) == 1,
            context.dim(0) == 1, context.dim(2) == hidden.dim(2), context.dtype == hidden.dtype,
            hidden.dim(1) <= rows, context.dim(1) <= rows, c.size == 1, c.dtype == .int32
        else { return nil }
        return joinLaunch(hidden, projection, base, context, c, rows: rows, ks: ks, gs: gs)
    }

    /// `(rope(qNorm(q rows c ..< c + l)), rope(kNorm(k)))` from the stacked
    /// product `y` at `pos` = `[offset, offset, l]`, or nil.
    static func heads(
        _ y: MLXArray, blockRows l: Int, qNorm: RMSNorm, kNorm: RMSNorm, confirmed c: MLXArray,
        pos: MLXArray
    ) -> (MLXArray, MLXArray)? {
        guard headsChosen, let kernel = headsKernel, let p = parameters, let g = geometry,
            y.ndim == 3, y.dim(0) == 1, y.dim(2) == (g.hq + 2 * g.hk) * g.d, l <= y.dim(1),
            qNorm.weight.shape == [g.d], kNorm.weight.shape == [g.d], qNorm.eps == g.eps,
            kNorm.eps == g.eps, qNorm.weight.dtype == y.dtype, kNorm.weight.dtype == y.dtype,
            c.size == 1, c.dtype == .int32
        else { return nil }
        return headsLaunch(kernel, y, l, g.hq, g.hk, qNorm.weight, kNorm.weight, p, pos, c)
    }

    /// The self-tests and the trial, once, at bind.
    static func prepare(
        rope: RoPELayer, base theta: Float, dtype: DType, heads hq: Int, kvHeads hk: Int,
        headDim d: Int, eps: Float, hidden hs: Int, kernelSize ks: Int, groupSize gs: Int,
        blockSize l: Int
    ) {
        guard enabled, d == 128, hk > 0, hs % gs == 0, hs % 256 == 0, l >= 2,
            [DType.bfloat16, .float16].contains(dtype)
        else { return }
        lock.withLock {
            guard parameters == nil else { return }
            let start = DispatchTime.now().uptimeNanoseconds
            let n = 2 * l
            let w = hq + 2 * hk
            let p = MLXArray([eps, log2(theta), 1])
            func draw(_ shape: [Int], _ key: UInt64) -> MLXArray {
                // Rows from 1e-3 to 3e3, as the other drafter self-tests draw.
                (MLXRandom.normal(shape, key: MLXRandom.key(key))
                    * exp(MLXRandom.uniform(
                        Float(-7) ..< Float(8), [shape[0], shape[1], 1],
                        key: MLXRandom.key(key &+ 7)))).asType(dtype)
            }
            var rowsSame = true
            var headsSame = headsKernel != nil
            var compared = 0
            // Inputs of the trial (the last case's).
            var trialRows: (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)?
            var trialHeads: (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)?
            do {
                try withError { error in
                    let template: [(String, any KernelTemplateArg)] = [
                        ("T", dtype), ("KS", ks), ("GS", gs), ("TAP", 0), ("QUAD", dflash2ConvQuad(dtype, gs)),
                    ]
                    for count in [5, l] {
                        let h = draw([1, l, hs], UInt64(600 + count))
                        let dyn = draw([1, l, 2 * ks * (hs / gs)], UInt64(610 + count))
                        let kernelBase = draw([2, ks, hs], UInt64(620 + count))
                        let context = draw([1, count, hs], UInt64(630 + count))
                        let conv = dflash2GroupedConvKernel(
                            [h, dyn, kernelBase], template: template, grid: (dflash2ConvColumns(hs, dtype, gs), l, 1),
                            threadGroup: (256, 1, 1), outputShapes: [h.shape],
                            outputDTypes: [dtype])[0]
                        let base = concatenated(
                            [context, MLXArray.zeros([1, n - count, hs], dtype: dtype)], axis: 1)
                        eval(h, dyn, kernelBase, context, conv, base)
                        for c in 1 ... l {
                            let cdev = MLXArray(Int32(c))
                            let fused = joinLaunch(
                                h, dyn, kernelBase, context, cdev, rows: n, ks: ks, gs: gs)
                            let reference = dynamicSliceUpdate(
                                base, update: conv, start: cdev.reshaped([1]), axes: [1])
                            rowsSame = rowsSame && fused.shape == reference.shape
                                && all(fused.view(dtype: .uint16) .== reference.view(dtype: .uint16))
                                    .item(Bool.self)
                            compared += reference.size
                        }
                        trialRows = (h, dyn, kernelBase, context, base)
                    }
                    if let kernel = headsKernel {
                        for (seed, c, off) in [(1, 1, 37), (2, 9, 4077), (3, l, 65514)] {
                            let y = draw([1, n, w * d], UInt64(700 + seed))
                            let qw = (1 + 0.3 * MLXRandom.normal(
                                [d], key: MLXRandom.key(UInt64(710 + seed)))).asType(dtype)
                            let kw = (1 + 0.3 * MLXRandom.normal(
                                [d], key: MLXRandom.key(UInt64(720 + seed)))).asType(dtype)
                            let cdev = MLXArray(Int32(c))
                            let pos = MLXArray([Int32(off), Int32(off), Int32(l)])
                            let (fq, fk) = headsLaunch(kernel, y, l, hq, hk, qw, kw, p, pos, cdev)
                            // As `DFlash2Attention.speculative` issues them.
                            let blockRows = dynamicSlice(
                                y, start: cdev.reshaped([1]), axes: [1],
                                sliceSize: [Int32(1), Int32(l), Int32(y.dim(2))])
                            let rq = rope(
                                MLXFast.rmsNorm(
                                    blockRows[.ellipsis, ..<(hq * d)].reshaped(1, l, hq, d),
                                    weight: qw, eps: eps
                                ).transposed(0, 2, 1, 3),
                                offset: MLXArray(Int32(off)) + cdev)
                            let rk = rope(
                                MLXFast.rmsNorm(
                                    y[.ellipsis, (hq * d) ..< ((hq + hk) * d)].reshaped(1, n, hk, d),
                                    weight: kw, eps: eps
                                ).transposed(0, 2, 1, 3),
                                offset: off)
                            for (f, r) in [(fq, rq), (fk, rk)] {
                                headsSame = headsSame && f.shape == r.shape
                                    && all(f.view(dtype: .uint16) .== r.view(dtype: .uint16))
                                        .item(Bool.self)
                                compared += r.size
                            }
                            trialHeads = (y, qw, kw, pos, cdev)
                        }
                    }
                    try error.check()
                }
            } catch {
                rowsSame = false
                headsSame = false
            }
            // The trial: five layers' fronts, composed against fused, at the
            // drafter's shapes (the q|k|v matmul between them is the same
            // launch either way and is left out).
            var note = ""
            var rowsWins = rowsSame
            var headsWins = headsSame
            if rowsSame || headsSame, let (h, dyn, kernelBase, context, base) = trialRows,
                let (y, qw, kw, pos, cdev) = trialHeads, let kernel = headsKernel
            {
                let c = cdev
                let composedRows = {
                    [dynamicSliceUpdate(
                        base,
                        update: dflash2GroupedConvKernel(
                            [h, dyn, kernelBase],
                            template: [("T", dtype), ("KS", ks), ("GS", gs), ("TAP", 0), ("QUAD", dflash2ConvQuad(dtype, gs))],
                            grid: (dflash2ConvColumns(hs, dtype, gs), l, 1), threadGroup: (256, 1, 1), outputShapes: [h.shape],
                            outputDTypes: [dtype])[0],
                        start: c.reshaped([1]), axes: [1])]
                }
                let fusedRows = {
                    [joinLaunch(h, dyn, kernelBase, context, c, rows: n, ks: ks, gs: gs)]
                }
                let composedHeads = { () -> [MLXArray] in
                    let blockRows = dynamicSlice(
                        y, start: c.reshaped([1]), axes: [1],
                        sliceSize: [Int32(1), Int32(l), Int32(y.dim(2))])
                    return [
                        rope(
                            MLXFast.rmsNorm(
                                blockRows[.ellipsis, ..<(hq * d)].reshaped(1, l, hq, d),
                                weight: qw, eps: eps
                            ).transposed(0, 2, 1, 3), offset: MLXArray(Int32(37)) + c),
                        rope(
                            MLXFast.rmsNorm(
                                y[.ellipsis, (hq * d) ..< ((hq + hk) * d)].reshaped(1, n, hk, d),
                                weight: kw, eps: eps
                            ).transposed(0, 2, 1, 3), offset: 37),
                    ]
                }
                let fusedHeads = { () -> [MLXArray] in
                    let (q, k) = headsLaunch(kernel, y, l, hq, hk, qw, kw, p, pos, c)
                    return [q, k]
                }
                let t = DFlash2LaunchTrial.race([composedRows, fusedRows, composedHeads, fusedHeads])
                if t.count == 4 {
                    rowsWins = rowsSame && t[1] <= t[0] * DFlash2LaunchTrial.tolerance
                    headsWins = headsSame && t[3] <= t[2] * DFlash2LaunchTrial.tolerance
                    note = String(
                        format: "; trial per 5 layers: rows %.1f us composed, %.1f us one launch; "
                            + "heads %.1f us composed, %.1f us one launch", t[0], t[1], t[2], t[3])
                } else {
                    (rowsWins, headsWins) = (false, false)
                    note = "; trial failed"
                }
            }
            rowsChosen = rowsWins
            headsChosen = headsWins
            parameters = p
            geometry = (hq, hk, d, eps)
            let ms = (DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            FileHandle.standardError.write(
                ("dflash2 speculative front: self-test "
                    + (rowsSame && headsSame ? "passed" : "FAILED (rows \(rowsSame), heads \(headsSame))")
                    + " (\(compared) values compared bitwise)\(note); rows "
                    + (rowsWins ? "one launch" : "composed") + ", heads "
                    + (headsWins ? "one launch" : "composed") + "; \(ms) ms\n").data(using: .utf8)!)
        }
    }
}

/// The next block's device inputs from the verify's acceptance packet, in ONE
/// launch (`DFlash2DraftModel.proposeSpeculative`).
///
/// A block built before the readback reads the round's outcome on the
/// device: `accepted` (the cumulative product of `drafts == targets`,
/// summed), the anchor (the target at `accepted`), the confirmed count
/// `c = accepted + 1`, the verify context's rows in the drafter's dtype with
/// the rows at and past `c` zeroed (`DFlash2ExactRowClasses`), the block
/// attention's key mask (`key < held + L + c`) and the queries' rope offset
/// (`offset + c`). Composed, that is a dependent chain of small launches
/// (compare, widen, scan, sum, take, add, compare, the select and its zero
/// fill, cast, add, add, compare) in front of the block's first projection,
/// all on the GPU's path from the verify's end to the drafter, each waiting
/// for the one before it: a fixed few microseconds a launch whatever the
/// device's bandwidth. Here every thread counts the leading matches itself
/// (at most `k` integer compares of the packet), and one launch writes every
/// output.
///
/// EXACT: the same integers, the same select and the same cast (a
/// `static_cast` of the context element to the drafter's dtype, as the
/// composed `asType`). `prepare` runs it against the composed ops at load,
/// for every accept count 0...k, the context dtypes the block meets, masked
/// and unmasked, on rows carrying NaN, infinities, signed zeros and
/// subnormals, every output compared bit for bit; the speculation self-test
/// then proves whole blocks through it. A mismatch or an MLX error keeps the
/// composed ops. `MLXFAST_DFLASH_PACKET_FRONT=0` keeps them.
enum DFlash2PacketFront {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_PACKET_FRONT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let lock = NSLock()
    /// Set once at bind by `prepare`: the self-test passed and the trial kept it.
    nonisolated(unsafe) private(set) static var active = false
    nonisolated(unsafe) private static var prepared = false

    /// Back to the composed ops (the speculation self-test failed with it).
    static func deactivate() {
        lock.withLock { active = false }
    }

    private static let kernel = MLXFast.metalKernel(
        name: "dflash2_packet_front",
        inputNames: ["packet", "ctx", "dims"],
        outputNames: ["rows", "anchor", "conf", "keymask", "qoff"],
        source: """
            // dims: [k, context rows, width, keys, key bound at c = 0, rope offset at c = 0]
            const uint gid = thread_position_in_grid.x;
            const int k = dims[0];
            int a = 0;
            while (a < k && packet[a] == packet[k + a]) {
              a++;
            }
            const int c = a + 1;
            if (gid == 0) {
              anchor[0] = packet[k + a];
              conf[0] = c;
              qoff[0] = dims[5] + c;
            }
            if (gid < uint(dims[3])) {
              keymask[gid] = int(gid) < dims[4] + c;
            }
            const uint w = uint(dims[2]);
            const uint e = gid * 4;
            if (e < uint(dims[1]) * w) {
              // `w % 4 == 0`: the four elements share a row.
              const bool kept = !MASK || int(e / w) < c;
              for (uint i = 0; i < 4; i++) {
                rows[e + i] = kept ? static_cast<O>(ctx[e + i]) : static_cast<O>(static_cast<T>(0));
              }
            }
            """,
        ensureRowContiguous: true)

    typealias Inputs = (
        anchor: MLXArray, confirmed: MLXArray, rows: MLXArray, keyMask: MLXArray,
        queryOffset: MLXArray
    )

    private static func launch(
        _ packet: MLXArray, depth k: Int, context: MLXArray, rows: Int, dtype: DType,
        masked: Bool, keys: Int, keyBound: Int, offset: Int
    ) -> Inputs {
        let width = context.dim(2)
        let dims = MLXArray(
            [Int32(k), Int32(rows), Int32(width), Int32(keys), Int32(keyBound), Int32(offset)])
        let out = kernel(
            [packet, context[0..., ..<rows, 0...], dims],
            template: [("T", context.dtype), ("O", dtype), ("MASK", masked)],
            grid: (max(rows * width / 4, keys), 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, width], [1, 1], [1], [1, keys], [1]],
            outputDTypes: [dtype, .int32, .int32, .bool, .int32])
        return (out[1], out[2], out[0], out[3], out[4])
    }

    /// The composed ops the launch stands in for, as `speculateBlock` and
    /// `proposeSpeculative` build them.
    static func composed(
        _ packet: MLXArray, depth k: Int, context: MLXArray, rows: Int, dtype: DType,
        masked: Bool, keys: Int, keyBound: Int, offset: Int
    ) -> Inputs {
        let targets = packet[k ..< (2 * k + 1)]
        let accepted = cumprod((packet[0 ..< k] .== targets[0 ..< k]).asType(.int32), axis: 0)
            .sum().asType(.int32)
        let anchor = targets.take(accepted.reshaped([1]), axis: 0)
        let c = (accepted + MLXArray(Int32(1))).reshaped([]).asType(.int32)
        let verifyRows = context[0..., ..<rows, 0...]
        let projected =
            masked
            ? which(
                (MLXArray(Int32(0) ..< Int32(rows)) .< c).reshaped([1, rows, 1]),
                verifyRows, MLXArray.zeros([1, 1, 1], dtype: verifyRows.dtype))
            : verifyRows
        let keyMask = (MLXArray(Int32(0) ..< Int32(keys)) .< (MLXArray(Int32(keyBound)) + c))
            .reshaped([1, keys])
        return (
            anchor.reshaped([1, 1]), c.reshaped([1]), projected.asType(dtype), keyMask,
            (MLXArray(Int32(offset)) + c).reshaped([1])
        )
    }

    /// The launch's outputs for `packet` (`[drafts | targets]`, int32, depth
    /// `k`) and the verify `context` (its leading `rows` rows), or nil when it
    /// does not apply (the caller keeps the composed ops).
    static func apply(
        _ packet: MLXArray, depth k: Int, context: MLXArray, rows: Int, dtype: DType,
        masked: Bool, keys: Int, keyBound: Int, offset: Int
    ) -> Inputs? {
        guard active, packet.ndim == 1, packet.dtype == .int32, k >= 1,
            packet.dim(0) >= 2 * k + 1, context.ndim == 3, context.dim(0) == 1,
            (1 ... context.dim(1)).contains(rows), context.dim(2) % 4 == 0,
            [DType.float16, .bfloat16, .float32].contains(context.dtype),
            [DType.float16, .bfloat16].contains(dtype), keys >= 1
        else { return nil }
        return launch(
            packet, depth: k, context: context, rows: rows, dtype: dtype, masked: masked,
            keys: keys, keyBound: keyBound, offset: offset)
    }

    /// The self-test and the trial, once, at bind (the drafter's block of
    /// `k + 1` rows over `width`-wide verify rows, in `dtype`).
    static func prepare(depth k: Int, width: Int, dtype: DType) {
        guard enabled, k >= 1, width % 4 == 0, [DType.float16, .bfloat16].contains(dtype) else {
            return
        }
        lock.withLock {
            guard !prepared else { return }
            prepared = true
            let start = DispatchTime.now().uptimeNanoseconds
            var same = true
            var compared = 0
            var trialInputs: (MLXArray, MLXArray)?
            do {
                try withError { error in
                    let rows = k + 1
                    for (seed, (contextType, masked)) in [
                        (DType.float16, true), (.float32, true), (.bfloat16, true), (.float16, false),
                    ].enumerated() {
                        // Rows from 1e-8 to 1e8, then NaN, infinities, signed
                        // zeros and values below the narrow types' normals.
                        var values = MLXRandom.normal(
                            [1, rows, width], key: MLXRandom.key(UInt64(0x7a3 + seed)))
                            * exp(MLXRandom.uniform(
                                Float(-18) ..< Float(18), [1, rows, 1],
                                key: MLXRandom.key(UInt64(0x7b3 + seed))))
                        let column = MLXArray(Int32(0) ..< Int32(width)).reshaped([1, 1, width])
                        for (index, special) in [
                            Float.nan, .infinity, -.infinity, -0.0, 1e-7, -3e-39, 70000,
                        ].enumerated() {
                            values = which(column .== Int32(97 * index + 5), MLXArray(special), values)
                        }
                        let context = values.asType(contextType)
                        eval(context)
                        for a in 0 ... k {
                            // Drafts match the targets on their first `a`.
                            let targets = (0 ... k).map { Int32(1000 + ($0 &* 7919 &+ a &* 104_729 &+ seed) % 90_000) }
                            var drafts = Array(targets[..<k])
                            if a < k { drafts[a] &+= 1 + Int32(seed) }
                            for i in stride(from: a + 1, to: k, by: 2) { drafts[i] = targets[i] }
                            let packet = MLXArray(drafts + targets + [Int32(-5)])
                            let keys = 37 + 91 * a + seed
                            let keyBound = keys - 2 * rows + rows
                            let offset = 4077 * a + seed
                            let fused = launch(
                                packet, depth: k, context: context, rows: rows, dtype: dtype,
                                masked: masked, keys: keys, keyBound: keyBound, offset: offset)
                            let reference = composed(
                                packet, depth: k, context: context, rows: rows, dtype: dtype,
                                masked: masked, keys: keys, keyBound: keyBound, offset: offset)
                            let pairs = [
                                (fused.anchor, reference.anchor), (fused.confirmed, reference.confirmed),
                                (fused.queryOffset, reference.queryOffset),
                                (fused.keyMask.asType(.int32), reference.keyMask.asType(.int32)),
                                (fused.rows.view(dtype: .uint16).asType(.int32),
                                    reference.rows.view(dtype: .uint16).asType(.int32)),
                            ]
                            for (f, r) in pairs {
                                same = same && f.shape == r.shape && all(f .== r).item(Bool.self)
                                compared += r.size
                            }
                            trialInputs = (packet, context)
                        }
                    }
                    try error.check()
                }
            } catch {
                same = false
            }
            // The trial: the composed chain against the launch, at the
            // drafter's shapes, followed by one reader of every output.
            var note = ""
            var wins = same
            if same, let (packet, context) = trialInputs {
                let rows = k + 1
                let keys = 600
                func read(_ x: Inputs) -> [MLXArray] {
                    [x.rows[0..., 0 ..< 1, 0 ..< 8].asType(.float32).sum()
                        + x.anchor.asType(.float32).sum() + x.confirmed.asType(.float32).sum()
                        + x.keyMask.asType(.float32).sum() + x.queryOffset.asType(.float32).sum()]
                }
                let t = DFlash2LaunchTrial.race([
                    {
                        read(composed(
                            packet, depth: k, context: context, rows: rows, dtype: dtype,
                            masked: true, keys: keys, keyBound: keys - rows, offset: 512))
                    },
                    {
                        read(launch(
                            packet, depth: k, context: context, rows: rows, dtype: dtype,
                            masked: true, keys: keys, keyBound: keys - rows, offset: 512))
                    },
                ])
                if t.count == 2 {
                    wins = t[1] <= t[0] * DFlash2LaunchTrial.tolerance
                    note = String(format: "; trial per 5 fronts: composed %.1f us, one launch %.1f us", t[0], t[1])
                } else {
                    wins = false
                    note = "; trial failed"
                }
            }
            active = wins
            let ms = (DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            FileHandle.standardError.write(
                ("dflash2 packet front: self-test " + (same ? "passed" : "FAILED")
                    + " (accept counts 0-\(k), \(compared) values compared bitwise)\(note); "
                    + (wins ? "one launch" : "composed") + "; \(ms) ms\n").data(using: .utf8)!)
        }
    }
}

/// Serialized load-time timing of alternative launch sequences: interleaved
/// samples, each `copies` graphs of one builder evaluated at once (wall time
/// of the `eval`, inputs already on the device), the median per builder in
/// microseconds. Empty when MLX raised.
enum DFlash2LaunchTrial {
    /// A one-launch form is adopted only when its median is at most this
    /// fraction of the composed form's (at least 1% faster; a tie keeps the
    /// composed launches).
    static let tolerance = 0.99

    static func race(
        _ builders: [() -> [MLXArray]], copies: Int = 20, samples: Int = 11
    ) -> [Double] {
        var times = [[Double]](repeating: [], count: builders.count)
        do {
            try withError { error in
                for b in builders { eval((0 ..< copies).flatMap { _ in b() }) }
                for _ in 0 ..< samples {
                    for (i, b) in builders.enumerated() {
                        let outputs = (0 ..< copies).flatMap { _ in b() }
                        let t0 = DispatchTime.now().uptimeNanoseconds
                        eval(outputs)
                        times[i].append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000)
                    }
                }
                try error.check()
            }
        } catch {
            return []
        }
        return times.map { t in
            let s = t.sorted()
            // Per five graphs (a round's worth: five layers, or five rounds).
            return s.isEmpty ? .infinity : s[s.count / 2] * 5 / Double(copies)
        }
    }
}
