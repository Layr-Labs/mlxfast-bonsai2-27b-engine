//
//  Qwen4ExpText.swift
//  mlx-swift-lm
//
//  Text tower of Qwen 3.8 Flash-Next (HF `model_type` "qwen4_exp", text tower
//  "qwen4_exp_text").
//
//  REFERENCE AND LICENSE. This is a Swift port of the MIT-licensed mlx-lm
//  reference implementation: ml-explore/mlx-lm PR #1788
//  `mlx_lm/models/qwen4_exp.py` at head c961f839 (repository LICENSE: MIT,
//  Copyright (c) 2023 Apple Inc.). No AGPL-licensed source was read or ported.
//  The gated deltanet recurrence itself is the fork's own `GatedDelta.swift`,
//  shared with `Qwen3Next.swift`.
//
//  WHAT IS NEW versus `qwen3_next`, which is the closest model already in this
//  fork:
//   1. QSA sparse attention. Every full-attention layer carries an indexer that
//      scores pooled key blocks and keeps a per-query budget of them.
//   2. Hyper-connections. The residual stream is `hc_count` parallel streams;
//      a gated low-rank mixer reads them before each block and injects the
//      block output back with per-stream weights. There is no final `norm`
//      tensor: the last mixer carries it.
//   3. A sharded n-gram / PLE embedding. One layer adds an embedding looked up
//      by a hash of the recent 2- and 3-token history. The table is far too
//      large to hold in memory, so the rows come from an injected row source.
//      See `Qwen4ExpNGramRowSource` and docs/ngram-cache-design.md.
//   4. Split gated-deltanet projections: `in_proj_qkv`, `in_proj_z`,
//      `in_proj_b` and `in_proj_a` instead of the fused `in_proj_qkvz` /
//      `in_proj_ba` pair.
//
//  TEXT TOWER ONLY. The checkpoint carries a vision tower. It is dropped at
//  load, like every other text-only port in this fork.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration of the `"qwen4_exp_text"` tower.
public struct Qwen4ExpTextConfiguration: Codable, Sendable {
    public var modelType: String = "qwen4_exp_text"
    public var hiddenSize: Int = 2560
    public var hiddenLayers: Int = 48
    public var attentionHeads: Int = 24
    public var kvHeads: Int = 2
    public var headDim: Int = 256
    public var vocabularySize: Int = 248_320
    public var rmsNormEps: Float = 1e-6
    /// Additive offset every NON-GATED `Qwen4ExpRMSNorm` applies to its stored
    /// weight: 1 when the checkpoint holds zero-centered weights, 0 when it
    /// already bakes the offset in. Absent from every published
    /// `config.json`, so it defaults to the zero-centered convention the
    /// reference implements, and the engine overrides it from the pinned
    /// checkpoint fact. `MLXFastConstants.RMSNormConvention` carries the
    /// reasoning and the load-time verification.
    /// Added to a stored norm weight before it scales the normalized
    /// activation. See ``Qwen4ExpRMSNorm/weightOffset``.
    ///
    /// ZERO IS THIS FAMILY'S DEFAULT, because this checkpoint BAKES the
    /// offset: its non-gated norm tensors store `1 + w`, so the model must
    /// compute `y * w`. Nothing in `config.json` says so — the fact was
    /// measured on the box over all 157 non-gated norm tensors and pinned by
    /// the previous engine as
    /// `MLXFastConstants.rmsNormConvention == .offsetBaked`, whose
    /// `weightOffset` is 0 (mlxfast-qwen38-125b-a6b-engine-dev
    /// `Sources/MLXFastCore/Constants.swift:139-144,176`).
    ///
    /// A `rms_norm_weight_offset` in `config.json` still wins, so a
    /// zero-centered conversion can say `1` and be served correctly. The
    /// value is decided HERE, at configuration decode, because the module's
    /// `weightOffset` is a `let` fixed at init — by the time a container or
    /// a runner holds the module it is far too late to change it.
    ///
    /// Getting this wrong scales every non-gated norm in the tower by about
    /// one unit and produces incoherent output while every shape and digest
    /// still checks out, which is why
    /// `Qwen4ExpNormConvention.validate(model:expectedOffset:)` refuses a
    /// module whose loaded tensors contradict the configured value.
    public var rmsNormWeightOffset: Float = 0

    /// True when `rms_norm_weight_offset` was actually present where this
    /// configuration was decoded from.
    ///
    /// The key appears in two places depending on the tree: INSIDE
    /// `text_config` on the raw HF checkpoint, and at the TOP LEVEL of the
    /// transformed tree, whose transform flattens `text_config` away
    /// entirely. `Qwen4ExpConfiguration` reads the root only when the nested
    /// block did not carry it, and this is how it tells the difference
    /// between "the nested block said 0" and "the nested block said
    /// nothing".
    public internal(set) var rmsNormWeightOffsetIsExplicit: Bool = false

    /// WHERE the resolved offset came from, for the `--verbose` load
    /// summary. The value alone does not say whether the checkpoint spoke,
    /// and this defect is exactly the case where that mattered.
    public enum RMSNormWeightOffsetSource: String, Sendable {
        case textConfig = "text_config"
        case topLevel = "top_level"
        case familyDefault = "family_default"
    }
    public internal(set) var rmsNormWeightOffsetSource: RMSNormWeightOffsetSource =
        .familyDefault
    public var layerTypes: [String] = []
    public var fullAttentionInterval: Int = 4

    // Mixture of experts
    public var numExperts: Int = 512
    public var numExpertsPerTok: Int = 10
    public var moeIntermediateSize: Int = 640
    public var sharedExpertIntermediateSize: Int = 640

    // Gated deltanet
    public var linearNumKeyHeads: Int = 16
    public var linearNumValueHeads: Int = 48
    public var linearKeyHeadDim: Int = 128
    public var linearValueHeadDim: Int = 128
    public var linearConvKernelDim: Int = 4
    public var outputGateType: String = "sigmoid"

    // Hyper-connections
    public var hcCount: Int = 4
    public var hcLowrank: Int = 320

    // QSA indexer
    public var indexerHeads: Int = 4
    public var indexerKVHeads: Int = 1
    public var indexerHeadDim: Int = 128
    public var indexerBudget: Int = 2048
    public var indexerCompressRatio: Int = 4

    // n-gram / PLE
    public var ngramSize: Int = 3
    public var headsPerNGram: Int = 8
    public var ngramVocabSizeBase: Int = 20_000_000
    public var makeNGramVocabSizeDivisibleBy: Int = 128
    public var splitNGramParts: Int = 128
    public var pleEmbedDim: Int = 2560
    public var pleLayerIds: [Int] = [2]
    public var pleConvKernelSize: Int = 4
    public var seed: Int = 0

    public var eosTokenId: Int = 248_044
    public var partialRotaryFactor: Float = 0.25
    public var ropeTheta: Float = 10_000_000
    public var maxPositionEmbeddings: Int = 262_144
    public var tieWordEmbeddings: Bool = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case rmsNormWeightOffset = "rms_norm_weight_offset"
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case outputGateType = "output_gate_type"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case indexerHeads = "indexer_n_heads"
        case indexerKVHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case ngramSize = "ngram_size"
        case headsPerNGram = "heads_per_ngram"
        case ngramVocabSizeBase = "ngram_vocab_size_base"
        case makeNGramVocabSizeDivisibleBy = "make_ngram_vocab_size_divisible_by"
        case splitNGramParts = "split_ngram_parts"
        case pleEmbedDim = "ple_embed_dim"
        case pleLayerIds = "ple_layer_ids"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case seed
        case eosTokenId = "eos_token_id"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    /// Key of the `rope_parameters` sub-object. It is read but never written,
    /// so it stays out of `CodingKeys` and the synthesized encoder.
    enum RopeCodingKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    /// `rope_parameters` sub-object. `rope_theta` and `partial_rotary_factor`
    /// appear BOTH at the top level of `text_config` and inside this object;
    /// the reference gives the nested pair priority, so this port does too.
    ///
    /// `mrope_interleaved` with `mrope_section` [11, 11, 10] is a MULTIMODAL
    /// fact. For a text-only tower every section shares one position per token,
    /// which makes interleaved mrope identical to plain rope over the same
    /// positions. This tower therefore applies plain partial rope, exactly like
    /// the reference.
    struct RopeParameters: Codable, Sendable {
        var ropeTheta: Float?
        var partialRotaryFactor: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func int(_ k: CodingKeys, _ fallback: Int) throws -> Int {
            try c.decodeIfPresent(Int.self, forKey: k) ?? fallback
        }
        func float(_ k: CodingKeys, _ fallback: Float) throws -> Float {
            try c.decodeIfPresent(Float.self, forKey: k) ?? fallback
        }

        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp_text"
        self.hiddenSize = try int(.hiddenSize, 2560)
        self.hiddenLayers = try int(.hiddenLayers, 48)
        self.attentionHeads = try int(.attentionHeads, 24)
        self.kvHeads = try int(.kvHeads, 2)
        self.headDim = try int(.headDim, 256)
        self.vocabularySize = try int(.vocabularySize, 248_320)
        self.rmsNormEps = try float(.rmsNormEps, 1e-6)
        let declaredOffset = try c.decodeIfPresent(Float.self, forKey: .rmsNormWeightOffset)
        self.rmsNormWeightOffset = declaredOffset ?? 0
        self.rmsNormWeightOffsetIsExplicit = declaredOffset != nil
        self.rmsNormWeightOffsetSource = declaredOffset != nil ? .textConfig : .familyDefault
        self.fullAttentionInterval = try int(.fullAttentionInterval, 4)
        self.numExperts = try int(.numExperts, 512)
        self.numExpertsPerTok = try int(.numExpertsPerTok, 10)
        self.moeIntermediateSize = try int(.moeIntermediateSize, 640)
        self.sharedExpertIntermediateSize = try int(.sharedExpertIntermediateSize, 640)
        self.linearNumKeyHeads = try int(.linearNumKeyHeads, 16)
        self.linearNumValueHeads = try int(.linearNumValueHeads, 48)
        self.linearKeyHeadDim = try int(.linearKeyHeadDim, 128)
        self.linearValueHeadDim = try int(.linearValueHeadDim, 128)
        self.linearConvKernelDim = try int(.linearConvKernelDim, 4)
        self.outputGateType =
            try c.decodeIfPresent(String.self, forKey: .outputGateType) ?? "sigmoid"
        self.hcCount = try int(.hcCount, 4)
        self.hcLowrank = try int(.hcLowrank, 320)
        self.indexerHeads = try int(.indexerHeads, 4)
        self.indexerKVHeads = try int(.indexerKVHeads, 1)
        self.indexerHeadDim = try int(.indexerHeadDim, 128)
        self.indexerBudget = try int(.indexerBudget, 2048)
        self.indexerCompressRatio = try int(.indexerCompressRatio, 4)
        self.ngramSize = try int(.ngramSize, 3)
        self.headsPerNGram = try int(.headsPerNGram, 8)
        self.ngramVocabSizeBase = try int(.ngramVocabSizeBase, 20_000_000)
        self.makeNGramVocabSizeDivisibleBy = try int(.makeNGramVocabSizeDivisibleBy, 128)
        self.splitNGramParts = try int(.splitNGramParts, 128)
        self.pleEmbedDim = try int(.pleEmbedDim, 2560)
        self.pleLayerIds = try c.decodeIfPresent([Int].self, forKey: .pleLayerIds) ?? [2]
        self.pleConvKernelSize = try int(.pleConvKernelSize, 4)
        self.seed = try int(.seed, 0)
        self.maxPositionEmbeddings = try int(.maxPositionEmbeddings, 262_144)
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        // `eos_token_id` is a scalar in `text_config` and a list at the root of
        // the checkpoint config. The n-gram hash needs ONE id -- the first.
        if let single = try? c.decode(Int.self, forKey: .eosTokenId) {
            self.eosTokenId = single
        } else if let many = try? c.decode([Int].self, forKey: .eosTokenId), let first = many.first
        {
            self.eosTokenId = first
        }

        var theta = try float(.ropeTheta, 10_000_000)
        var partial = try float(.partialRotaryFactor, 0.25)
        let ropeContainer = try decoder.container(keyedBy: RopeCodingKeys.self)
        if let rope = try ropeContainer.decodeIfPresent(
            RopeParameters.self, forKey: .ropeParameters)
        {
            theta = rope.ropeTheta ?? theta
            partial = rope.partialRotaryFactor ?? partial
        }
        self.ropeTheta = theta
        self.partialRotaryFactor = partial

        let declared = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        if declared.isEmpty {
            self.layerTypes = (0 ..< hiddenLayers).map { i in
                (i + 1) % fullAttentionInterval == 0 ? "full_attention" : "linear_attention"
            }
        } else {
            self.layerTypes = declared
        }
    }

    /// Rotary dimensions: only the first quarter of each head is rotated.
    public var rotaryDimensions: Int { max(1, Int(Float(headDim) * partialRotaryFactor)) }

    /// One PLE layer index per entry of `ple_layer_ids`, which is 1-based.
    public var pleLayerIndices: [Int] { pleLayerIds.map { $0 - 1 } }
}

// MARK: - Norms

/// Zero-centered RMSNorm: `y = rmsNorm(x) * (1 + weight)`.
///
/// The checkpoint stores zero-centered weights, so the scale is `1 + weight`
/// and the initial weight is zero. This is done HERE and not in `sanitize`, so
/// a saved-and-reloaded model stays consistent with the checkpoint.
///
/// With `groupSize` set the input is split into `dim / groupSize` groups and
/// each group is normalized on its own statistic; the scale still applies to
/// the flat vector. Hyper-connections use this to normalize each of the
/// `hc_count` residual streams separately under one weight.
public final class Qwen4ExpRMSNorm: Module {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    let eps: Float
    let groupSize: Int?

    /// Added to the stored weight before it scales the normalized activation.
    ///
    /// ONE OR ZERO, AND THE CHECKPOINT DECIDES WHICH. A zero-centered
    /// checkpoint stores `w` and wants `y * (1 + w)`; a checkpoint that bakes
    /// the offset stores `1 + w` and wants `y * w`. Applying the wrong one
    /// multiplies every activation in the tower by roughly 0 or roughly 2 and
    /// produces incoherent output at every quantization level, with nothing in
    /// `config.json` to say which is right --
    /// `MLXFastConstants.RMSNormConvention` is where the fact lives and
    /// `validateLoadedNormConvention` is what verifies it against the loaded
    /// tensors.
    public let weightOffset: Float

    public init(dimensions: Int, groupSize: Int? = nil, eps: Float, weightOffset: Float = 1) {
        if let groupSize {
            precondition(
                dimensions % groupSize == 0,
                "Qwen4ExpRMSNorm: \(dimensions) is not divisible by group size \(groupSize)")
        }
        self.eps = eps
        self.groupSize = groupSize
        self.weightOffset = weightOffset
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // EXACT WHEN THE OFFSET IS ZERO: `MLXArray(0) + weight` is the weight
        // itself, bit for bit, so the baked convention multiplies by the
        // checkpoint's own bytes and introduces no rounding of its own.
        let scale =
            weightOffset == 0 ? weight : MLXArray(weightOffset, dtype: weight.dtype) + weight
        guard let groupSize else {
            return MLXFast.rmsNorm(x, weight: scale, eps: eps)
        }
        let shape = x.shape
        let grouped = x.reshaped(shape.dropLast() + [-1, groupSize])
        let normed = MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps)
            .reshaped(shape)
        return normed * scale
    }
}

/// Conventional RMSNorm with an optional output gate, used by the gated
/// deltanet. Unlike ``Qwen4ExpRMSNorm`` the weight is a plain scale and starts
/// at one.
public final class Qwen4ExpRMSNormGated: Module {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    let eps: Float
    let useSigmoid: Bool

    public init(dimensions: Int, eps: Float, activation: String) {
        self.eps = eps
        self.useSigmoid = activation == "sigmoid"
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, gate: MLXArray? = nil) -> MLXArray {
        let out = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        guard let gate else { return out.asType(x.dtype) }
        let g = useSigmoid ? sigmoid(gate.asType(.float32)) : silu(gate.asType(.float32))
        return (g * out.asType(.float32)).asType(x.dtype)
    }
}

// MARK: - Rotary embedding

/// Partial rotary embedding evaluated at explicit positions.
///
/// This is not a `Module`: it holds no parameters, and the inverse frequencies
/// are rebuilt on each call from two integers so that nothing here can be
/// mistaken for a checkpoint tensor.
public struct Qwen4ExpRotary {
    public let dimensions: Int
    public let base: Float

    public init(dimensions: Int, base: Float) {
        self.dimensions = dimensions
        self.base = base
    }

    /// cos/sin for `positions`, shaped `positions.shape + [dimensions]`.
    public func cosSin(_ positions: MLXArray) -> (MLXArray, MLXArray) {
        let even = MLXArray(stride(from: Int32(0), to: Int32(dimensions), by: 2).map { $0 })
            .asType(.float32)
        let invFreq = MLX.exp(even * (-Foundation.log(base) / Float(dimensions)))
        let freqs = positions.asType(.float32)[.ellipsis, .newAxis] * invFreq
        let emb = concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb), MLX.sin(emb))
    }
}

/// Rotate only the leading `cos.dim(-1)` entries of the last axis.
public func qwen4ExpRopePartial(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let d = cos.dim(-1)
    // cos/sin are float32. Without this cast they promote `x` and the whole
    // attention silently runs in float32.
    let c = cos.asType(x.dtype)
    let s = sin.asType(x.dtype)
    let rotated = x[.ellipsis, ..<d]
    let half = d / 2
    let x1 = rotated[.ellipsis, ..<half]
    let x2 = rotated[.ellipsis, half...]
    let swapped = concatenated([-x2, x1], axis: -1)
    let out = rotated * c + swapped * s
    if x.dim(-1) == d { return out }
    return concatenated([out, x[.ellipsis, d...]], axis: -1)
}

/// Positions of `count` tokens starting at `offset`, shaped `[1, count]`.
public func qwen4ExpPositions(offset: Int, count: Int) -> MLXArray {
    MLXArray(Int32(offset) ..< Int32(offset + count))[.newAxis]
}

/// QSA block scores: `out[b, s, n, h] = sum_d q[b, s, h, d] * pooled[b, n, d]`.
///
/// This is `einsum("bshd,bnd->bsnh")` written as one matmul. It is written out
/// because `MLX.einsum` reaches a `dot_product` kernel that is not present in
/// every built metallib, and a missing kernel aborts the process. A matmul is
/// the same contraction over `d` and is always available.
///
/// BOTH indexer paths call this -- the legacy `KVCache` one and the
/// ContinuousBatchingV2 one -- so the two can never disagree about the scores.
public func qwen4ExpIndexerBlockScores(q: MLXArray, pooled: MLXArray) -> MLXArray {
    let b = q.dim(0)
    let s = q.dim(1)
    let h = q.dim(2)
    let d = q.dim(3)
    let n = pooled.dim(1)
    let flat = q.reshaped(b, s * h, d)
    let contracted = MLX.matmul(flat, pooled.transposed(0, 2, 1))
    return contracted.reshaped(b, s, h, n).transposed(0, 1, 3, 2)
}

// MARK: - QSA indexer

/// Selects, per query, a budget of compressed key blocks.
///
/// The indexer keeps its own untouched key tape (see
/// `Qwen4ExpAttentionCache`, which lives in `MLXLMCommon` and so cannot be a
/// symbol link from here), pools it into blocks of `compressRatio`
/// tokens, scores every block against the query, and keeps the best
/// `budget / compressRatio` of them. The result is a boolean keep mask that is
/// combined with the causal mask.
///
/// Below the budget there is nothing to select: every visible token would be
/// kept, so the indexer returns `nil` and attention stays plain causal.
///
/// LEFT PADDING IS NOT SUPPORTED. The reference has a second code path for a
/// batched cache whose rows are left padded. This port serves the caches this
/// fork actually builds, which are not padded, and refuses the padded case by
/// name rather than computing the wrong blocks.
public final class Qwen4ExpQSAIndexer: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let tokenBudget: Int
    let compressRatio: Int
    let blockTopK: Int

    @ModuleInfo(key: "index_qk_proj") var indexQKProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayerNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayerNorm: Qwen4ExpRMSNorm

    public init(_ args: Qwen4ExpTextConfiguration) {
        self.heads = args.indexerHeads
        self.kvHeads = args.indexerKVHeads
        self.headDim = args.indexerHeadDim
        self.tokenBudget = args.indexerBudget
        self.compressRatio = args.indexerCompressRatio
        self.blockTopK = args.indexerBudget / args.indexerCompressRatio

        _indexQKProj.wrappedValue = Linear(
            args.hiddenSize, (heads + kvHeads) * headDim, bias: false)
        _qLayerNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: headDim, eps: args.rmsNormEps,
            weightOffset: args.rmsNormWeightOffset)
        _kLayerNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: headDim, eps: args.rmsNormEps,
            weightOffset: args.rmsNormWeightOffset)
        super.init()
    }

    /// - Returns: a boolean keep mask `[B, 1, S, kvLength]`, or `nil` when the
    ///   visible context still fits the budget.
    public func callAsFunction(
        _ x: MLXArray,
        rope: Qwen4ExpRotary,
        cache: Qwen4ExpAttentionCache?,
        offset: Int
    ) -> MLXArray? {
        let B = x.dim(0)
        let S = x.dim(1)

        let qk = indexQKProj(x)
        let split = heads * headDim
        var q = qk[.ellipsis, ..<split].reshaped(B, S, heads, headDim)
        var rawK = qk[.ellipsis, split...].reshaped(B, S, headDim)

        if let cache {
            rawK = cache.updateIndexer(keys: rawK)
        }
        let kvLength = rawK.dim(1)
        if kvLength <= tokenBudget { return nil }

        let blocks = kvLength / compressRatio
        var pooled = rawK[0..., ..<(blocks * compressRatio), 0...]
            .reshaped(B, blocks, compressRatio, headDim)
        pooled = kLayerNorm(pooled.asType(.float32).mean(axis: 2).asType(rawK.dtype))

        // Block n holds the logical positions n * compressRatio and up.
        let blockStarts = MLXArray(Int32(0) ..< Int32(blocks)) * Int32(compressRatio)
        let (cosK, sinK) = rope.cosSin(blockStarts[.newAxis])
        pooled = qwen4ExpRopePartial(pooled, cos: cosK, sin: sinK)

        let qPos = qwen4ExpPositions(offset: offset, count: S)
        let (cosQ, sinQ) = rope.cosSin(qPos)
        q = qLayerNorm(q)
        q = qwen4ExpRopePartial(
            q, cos: cosQ[0..., 0..., .newAxis, 0...], sin: sinQ[0..., 0..., .newAxis, 0...])

        // Sum over heads of relu(q . k), per block.
        var scores = qwen4ExpIndexerBlockScores(
            q: q.asType(.float32), pooled: pooled.asType(.float32))
        scores = maximum(scores, MLXArray(Float(0))).sum(axis: -1) / Foundation.sqrt(Float(headDim))

        // A block is a candidate only when it lies entirely in the query's past.
        //
        // FLOOR DIVISION, AND IT IS LOAD-BEARING. `/` on an MLX array is TRUE
        // division: it promotes int32 to float and keeps the remainder. The
        // reference computes `//`, an integer block COUNT, and both uses below
        // need that count exactly:
        //
        //   * `visible` compares block ids against it. With 0.25 instead of 0,
        //     a query admits the INCOMPLETE block it sits in, so the keep mask
        //     lets row r attend to keys r+1 ... block end -- future tokens. It
        //     also shifts which blocks the top-k chooses among.
        //   * `ownStart` scales it back up. With 1.0 instead of 0 the "own
        //     partial block" tail starts PAST the query, so the query loses the
        //     keys it must always keep.
        //
        // Measured against the MIT mlx-lm reference (@c961f839) on shared
        // random weights: 52 disagreeing mask cells out of 576 at S=24, and
        // rows 0-2, 4-6, 8-10, 12-14 keeping strictly future keys.
        let complete = maximum(qPos + Int32(1), MLXArray(Int32(0)))
            .floorDivide(Int32(compressRatio))
        let blockIds = MLXArray(Int32(0) ..< Int32(blocks))
        var visible = blockIds[.newAxis, .newAxis, 0...] .< complete[.ellipsis, .newAxis]
        visible = MLX.broadcast(visible, to: [B, S, blocks])
        scores = MLX.where(visible, scores, MLXArray(-Float.infinity))

        let k = Swift.min(blockTopK, blocks)
        let top = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        let picked = takeAlong(visible, top, axis: -1)

        // Remap block -> tokens. The trailing partial block of the tape belongs
        // to no candidate block, so it is never selectable on its own.
        let sentinel = MLXArray(Int32(blocks))
        let keepBlocks = putAlong(
            MLXArray.zeros([B, S, blocks + 1], dtype: .bool),
            MLX.where(picked, top, sentinel),
            values: MLXArray(true),
            axis: -1
        )[.ellipsis, ..<blocks]
        var keep = repeated(keepBlocks, count: compressRatio, axis: -1)
        let rest = kvLength - blocks * compressRatio
        if rest > 0 {
            keep = concatenated([keep, MLXArray.zeros([B, S, rest], dtype: .bool)], axis: -1)
        }

        // The reference also keeps the tail of each query's own visible list --
        // the partial block the query sits in. Without it a query whose past
        // holds fewer than compressRatio tokens gets an all-masked row, which
        // softmax turns into a uniform average over every key, future ones
        // included.
        let ownStart = complete * Int32(compressRatio)
        let tokens = MLXArray(Int32(0) ..< Int32(kvLength))
        let kvPos = MLXArray(Int32(kvLength - S) ..< Int32(kvLength))
        let own =
            (tokens[.newAxis, .newAxis, 0...] .>= ownStart[.ellipsis, .newAxis])
            & (tokens[.newAxis, .newAxis, 0...] .<= kvPos[.newAxis, 0..., .newAxis])

        return expandedDimensions(keep | own, axis: 1)
    }
}

// MARK: - Full attention

public final class Qwen4ExpAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "indexer") public var indexer: Qwen4ExpQSAIndexer

    public init(_ args: Qwen4ExpTextConfiguration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = Foundation.pow(Float(args.headDim), -0.5)

        let d = args.hiddenSize
        // `q_proj` also carries the output gate, hence the doubled width.
        _qProj.wrappedValue = Linear(d, heads * headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(d, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(d, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, d, bias: false)
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: headDim, eps: args.rmsNormEps,
            weightOffset: args.rmsNormWeightOffset)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: headDim, eps: args.rmsNormEps,
            weightOffset: args.rmsNormWeightOffset)
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(args)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        rope: Qwen4ExpRotary,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        let offset = cache?.offset ?? 0

        // SEAM (reported, needs a ruling): the QSA keep mask is a CUSTOM array
        // mask. `attentionWithCacheUpdate` DISCARDS custom masks on the
        // ContinuousBatchingV2 path, so a v2 cache would silently attend
        // densely instead of sparsely. Refuse by name rather than serve a
        // different model.
        let qsaCache: Qwen4ExpAttentionCache?
        if let cache {
            guard let typed = cache as? Qwen4ExpAttentionCache else {
                preconditionFailure(
                    """
                    Qwen4ExpAttention requires a Qwen4ExpAttentionCache: the QSA \
                    indexer keeps its own key tape and emits a custom array mask. \
                    Got \(type(of: cache)).
                    """)
            }
            qsaCache = typed
        } else {
            qsaCache = nil
        }

        let sparse = indexer(x, rope: rope, cache: qsaCache, offset: offset)

        let projected = qProj(x).reshaped(B, S, heads, -1).split(parts: 2, axis: -1)
        var queries = projected[0]
        let gate = projected[1].reshaped(B, S, -1)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, S, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, S, kvHeads, -1).transposed(0, 2, 1, 3)

        let (cos, sin) = rope.cosSin(qwen4ExpPositions(offset: offset, count: S))
        queries = qwen4ExpRopePartial(queries, cos: cos[0..., .newAxis], sin: sin[0..., .newAxis])
        keys = qwen4ExpRopePartial(keys, cos: cos[0..., .newAxis], sin: sin[0..., .newAxis])

        var effectiveMask = mask
        if let sparse {
            // Keep the combination BOOLEAN: adding it would drop the causality
            // that the symbolic ".causal" mode carries.
            let kvLength = offset + S
            switch mask {
            case .none:
                effectiveMask = .array(sparse)
            case .causal:
                let rinds = MLXArray(Int32(0) ..< Int32(kvLength))
                let linds = MLXArray(Int32(kvLength - S) ..< Int32(kvLength))[0..., .newAxis]
                effectiveMask = .array((linds .>= rinds) & sparse)
            case .array(let m) where m.dtype == .bool:
                effectiveMask = .array(m & sparse)
            case .array(let m):
                let neg = MLXArray(-Float.greatestFiniteMagnitude).asType(m.dtype)
                effectiveMask = .array(m + MLX.where(sparse, MLXArray(0).asType(m.dtype), neg))
            default:
                preconditionFailure(
                    "Qwen4ExpAttention: cannot combine the QSA keep mask with \(mask)")
            }
        }

        let out = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: effectiveMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, S, -1)

        return oProj(out * sigmoid(gate))
    }
}

// MARK: - Gated deltanet

public final class Qwen4ExpGatedDeltaNet: Module {
    let valueHeads: Int
    let keyHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int
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
    @ModuleInfo(key: "norm") var norm: Qwen4ExpRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    public init(_ args: Qwen4ExpTextConfiguration) {
        self.valueHeads = args.linearNumValueHeads
        self.keyHeads = args.linearNumKeyHeads
        self.keyHeadDim = args.linearKeyHeadDim
        self.valueHeadDim = args.linearValueHeadDim
        self.keyDim = keyHeadDim * keyHeads
        self.valueDim = valueHeadDim * valueHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

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
        // Unlike qwen3_next these projections are split, not fused.
        _inProjQKV.wrappedValue = Linear(args.hiddenSize, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(args.hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(args.hiddenSize, valueHeads, bias: false)
        _inProjA.wrappedValue = Linear(args.hiddenSize, valueHeads, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([valueHeads])
        _aLog.wrappedValue = MLXArray.zeros([valueHeads])
        _norm.wrappedValue = Qwen4ExpRMSNormGated(
            dimensions: valueHeadDim, eps: args.rmsNormEps, activation: args.outputGateType)
        _outProj.wrappedValue = Linear(valueDim, args.hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: Qwen4ExpLayerCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)

        var mixedQKV = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, valueHeads, valueHeadDim)
        let b = inProjB(x)
        let a = inProjA(x)

        let convState =
            cache?[Qwen4ExpLayerCache.deltaConvSlot]
            ?? MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: x.dtype)

        if let mask {
            mixedQKV = MLX.where(
                expandedDimensions(mask, axis: -1), mixedQKV, MLXArray.zeros(like: mixedQKV))
        }

        let convInput = concatenated([convState, mixedQKV], axis: 1)
        if let cache {
            cache[Qwen4ExpLayerCache.deltaConvSlot] =
                contiguous(convInput[0..., (1 - convKernelSize)..., 0...])
        }
        let convOut = silu(conv1d(convInput))
        let parts = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)

        var q = parts[0].reshaped(B, S, keyHeads, keyHeadDim)
        var k = parts[1].reshaped(B, S, keyHeads, keyHeadDim)
        let v = parts[2].reshaped(B, S, valueHeads, valueHeadDim)

        let invScale = Foundation.pow(Float(keyHeadDim), -0.5)
        q =
            MLXArray(invScale * invScale).asType(x.dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        k =
            MLXArray(invScale).asType(x.dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, state) = gatedDeltaUpdate(
            q: q,
            k: k,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: cache?[Qwen4ExpLayerCache.deltaStateSlot],
            mask: mask
        )
        if let cache {
            cache[Qwen4ExpLayerCache.deltaStateSlot] = state
            cache.advance(S)
        }
        return outProj(norm(out, gate: z).reshaped(B, S, -1))
    }
}

// MARK: - Mixture of experts

public final class Qwen4ExpMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

public final class Qwen4ExpSparseMoeBlock: Module {
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen4ExpMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    public init(_ args: Qwen4ExpTextConfiguration) {
        self.topK = args.numExpertsPerTok
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )
        _sharedExpert.wrappedValue = Qwen4ExpMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // The router runs in float32 and the softmax is taken AFTER selection,
        // over the selected logits only -- this is the reference order.
        let logits = gate(x.asType(.float32))
        let indices = argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let weights = MLX.softmax(takeAlong(logits, indices, axis: -1), axis: -1, precise: true)
        let routed = (switchMLP(x, indices) * weights[.ellipsis, .newAxis])
            .sum(axis: -2)
            .asType(x.dtype)
        return routed + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }
}

// MARK: - Hyper-connections

/// Gated residual mixer for the hyper-connection stream.
///
/// The residual carries `hc_count` streams side by side. Before a block the
/// mixer normalizes each stream, builds a low-rank gate over all of them, and
/// averages them into one width-`hidden` input. After the block the output is
/// injected back into each stream with its own weight.
///
/// The last mixer of the tower runs without an inject head; it is what stands
/// in for the final `norm` tensor, which this checkpoint does not have.
public final class Qwen4ExpGatedResidual: Module {
    let hcCount: Int
    let dimensions: Int

    @ModuleInfo(key: "hc_norm") var hcNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var mixDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var mixUp: Linear
    @ModuleInfo(key: "block_inject_weight") var blockInject: Linear?

    public init(_ args: Qwen4ExpTextConfiguration, useInject: Bool = true) {
        self.hcCount = args.hcCount
        self.dimensions = args.hiddenSize
        let wide = args.hcCount * args.hiddenSize

        _hcNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: wide, groupSize: args.hiddenSize, eps: args.rmsNormEps,
            weightOffset: args.rmsNormWeightOffset)
        _mixDown.wrappedValue = Linear(wide, args.hcLowrank, bias: false)
        _mixUp.wrappedValue = Linear(args.hcLowrank, wide, bias: false)
        if useInject {
            _blockInject.wrappedValue = Linear(wide, args.hcCount, bias: false)
        }
        super.init()
    }

    private func mixed(_ normed: MLXArray) -> MLXArray {
        var w = silu(mixDown(normed) / Float(hcCount))
        w = sigmoid(mixUp(w))
        let lead = w.shape.dropLast()
        return
            (w.reshaped(lead + [hcCount, dimensions])
            * normed.reshaped(lead + [hcCount, dimensions])).mean(axis: -2)
    }

    /// Mixer without an inject head; the tower's final projection.
    public func callAsFunction(_ hyper: MLXArray) -> MLXArray {
        mixed(hcNorm(hyper))
    }

    /// - Returns: `(blockInput, residual, injectWeights)`.
    public func mixWithInject(_ hyper: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        guard let blockInject else {
            preconditionFailure("Qwen4ExpGatedResidual: this mixer has no inject head")
        }
        let normed = hcNorm(hyper)
        let inject = 2 * sigmoid(blockInject(normed) / Float(hcCount))
        return (mixed(normed), hyper, inject)
    }
}

/// Inject a block output back into the hyper-connection stream.
public func qwen4ExpInject(residual: MLXArray, output: MLXArray, inject: MLXArray) -> MLXArray {
    let lead = output.shape.dropLast()
    let spread = output[.ellipsis, .newAxis, 0...] * inject[.ellipsis, .newAxis]
    return residual + spread.reshaped(lead + [-1])
}
