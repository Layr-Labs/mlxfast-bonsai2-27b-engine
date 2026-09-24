//
//  Qwen4Exp.swift
//  mlx-swift-lm
//
//  Qwen 3.8 Flash-Next: decoder layer, tower and model.
//  HF `model_type` "qwen4_exp"; the text tower declares "qwen4_exp_text".
//
//  REFERENCE AND LICENSE. Swift port of the MIT-licensed mlx-lm reference
//  implementation: ml-explore/mlx-lm PR #1788 `mlx_lm/models/qwen4_exp.py` at
//  head c961f839. No AGPL-licensed source was read or ported.
//
//  The components live in `Qwen4ExpText.swift` and `Qwen4ExpNGram.swift`; the
//  native multi-token-prediction head lives in `Qwen4ExpMTP.swift`.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration of the `"qwen4_exp"` model.
///
/// The checkpoint nests the tower under `text_config` and carries a
/// `vision_config` beside it. This port is text only, so the vision half is
/// read past and its tensors are dropped at load. A flat configuration -- one
/// that IS the text tower -- is accepted as well, which is what
/// `"qwen4_exp_text"` gives.
public struct Qwen4ExpConfiguration: Codable, Sendable {
    public var modelType: String = "qwen4_exp"
    public var textConfig: Qwen4ExpTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    /// Root keys that are READ but never written, so they stay out of
    /// `CodingKeys` and the synthesized encoder. Same shape as
    /// `Qwen4ExpTextConfiguration.RopeCodingKeys`.
    enum RootCodingKeys: String, CodingKey {
        case rmsNormWeightOffset = "rms_norm_weight_offset"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp"

        if let nested = try container.decodeIfPresent(
            Qwen4ExpTextConfiguration.self, forKey: .textConfig)
        {
            var text = nested
            // THE NORM CONVENTION IS THE ONE ROOT KEY THIS READS, and the
            // trees are why. The raw HF checkpoint carries
            // `rms_norm_weight_offset` inside `text_config`; the TRANSFORMED
            // tree has no `text_config` at all — the transform flattens it —
            // and carries the key at the root. The flattened case already
            // lands in the `else` branch below, and this covers the third
            // shape: a `text_config` that does not name the offset while the
            // root does. Nested still wins when it names it, so nothing that
            // works today changes meaning.
            if !text.rmsNormWeightOffsetIsExplicit,
                let root = try? decoder.container(keyedBy: RootCodingKeys.self),
                let declared = try root.decodeIfPresent(
                    Float.self, forKey: .rmsNormWeightOffset)
            {
                text.rmsNormWeightOffset = declared
                text.rmsNormWeightOffsetIsExplicit = true
                text.rmsNormWeightOffsetSource = .topLevel
            }
            self.textConfig = text
        } else {
            var text = try Qwen4ExpTextConfiguration(from: decoder)
            // Decoded FROM THE ROOT, so an explicit key was a top-level one.
            if text.rmsNormWeightOffsetIsExplicit {
                text.rmsNormWeightOffsetSource = .topLevel
            }
            self.textConfig = text
        }

        // THE ROOT IS NOT READ, AND THAT IS DELIBERATE. `text_config` is the
        // whole tower contract, which is what the MIT mlx-lm reference builds
        // its `TextArgs` from.
        //
        // The root also carries an `eos_token_id`, and on this checkpoint it
        // is a LIST whose first entry (248046, the turn-end token) is NOT the
        // text tower's scalar (248044, end-of-text). Taking the root's first
        // entry would look harmless and would not be: that scalar is the
        // n-gram hash's SEGMENT BOUNDARY and its initial history fill, so it
        // changes which rows the per-layer embedding reads for every token
        // near a boundary. The engine-side gate pins 248044 for the same
        // reason; the two paths have to agree.
    }
}

// MARK: - Decoder layer

public final class Qwen4ExpDecoderLayer: Module {
    public let isLinear: Bool

    @ModuleInfo(key: "self_attn") public var selfAttn: Qwen4ExpAttention?
    @ModuleInfo(key: "linear_attn") public var linearAttn: Qwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "mlp") var mlp: Qwen4ExpSparseMoeBlock
    @ModuleInfo(key: "ple") public var ple: Qwen4ExpPLELayer?
    @ModuleInfo(key: "attn_hyper_connection") var attnHyperConnection: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHyperConnection: Qwen4ExpGatedResidual

    /// - Parameters:
    ///   - args: the text tower configuration this layer is built from.
    ///   - isLinear: build the gated-deltanet recurrence instead of full
    ///     attention.
    ///   - pleLayerIndex: index INTO `ple_layer_ids`, not the layer number;
    ///     `nil` on a layer that carries no PLE block.
    public init(_ args: Qwen4ExpTextConfiguration, isLinear: Bool, pleLayerIndex: Int?) {
        self.isLinear = isLinear
        if isLinear {
            _linearAttn.wrappedValue = Qwen4ExpGatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen4ExpAttention(args)
        }
        _mlp.wrappedValue = Qwen4ExpSparseMoeBlock(args)
        if let pleLayerIndex {
            _ple.wrappedValue = Qwen4ExpPLELayer(args, pleLayerIndex: pleLayerIndex)
        }
        _attnHyperConnection.wrappedValue = Qwen4ExpGatedResidual(args)
        _mlpHyperConnection.wrappedValue = Qwen4ExpGatedResidual(args)
        super.init()
    }

    public convenience init(_ args: Qwen4ExpTextConfiguration, layerIndex: Int) {
        self.init(
            args,
            isLinear: args.layerTypes[layerIndex] != "full_attention",
            pleLayerIndex: args.pleLayerIndices.firstIndex(of: layerIndex)
        )
    }

    public func callAsFunction(
        _ hyper: MLXArray,
        rope: Qwen4ExpRotary,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        convMask: MLXArray?,
        cache: KVCache?,
        ids: MLXArray,
        previousContext: MLXArray?
    ) -> MLXArray {
        var stream = hyper

        if let ple, let previousContext {
            stream =
                stream
                + ple(
                    stream, ids: ids, previousContext: previousContext,
                    cache: cache as? Qwen4ExpLayerCache)
        }

        var (input, residual, inject) = attnHyperConnection.mixWithInject(stream)
        let attended: MLXArray
        if isLinear {
            attended = linearAttn!(input, mask: convMask, cache: cache as? Qwen4ExpLayerCache)
        } else {
            attended = selfAttn!(input, rope: rope, mask: mask, cache: cache)
        }
        stream = qwen4ExpInject(residual: residual, output: attended, inject: inject)

        (input, residual, inject) = mlpHyperConnection.mixWithInject(stream)
        return qwen4ExpInject(residual: residual, output: mlp(input), inject: inject)
    }
}

// MARK: - Tower

public final class Qwen4ExpTower: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    public let layers: [Qwen4ExpDecoderLayer]

    /// There is no final `norm` tensor in this checkpoint; this mixer is what
    /// closes the hyper-connection stream and stands in for it.
    @ModuleInfo(key: "hyper_connection_mixer") var hyperConnectionMixer: Qwen4ExpGatedResidual

    let args: Qwen4ExpTextConfiguration
    let rope: Qwen4ExpRotary
    let contextLength: Int
    let firstFullAttentionIndex: Int?
    let firstLinearIndex: Int?
    public let pleLayerIndices: [Int]

    public init(_ args: Qwen4ExpTextConfiguration) {
        precondition(args.vocabularySize > 0)
        self.args = args
        self.rope = Qwen4ExpRotary(dimensions: args.rotaryDimensions, base: args.ropeTheta)
        self.contextLength = args.ngramSize - 1

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { Qwen4ExpDecoderLayer(args, layerIndex: $0) }
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(args, useInject: false)

        self.firstFullAttentionIndex = args.layerTypes.firstIndex(of: "full_attention")
        self.firstLinearIndex = args.layerTypes.firstIndex { $0 != "full_attention" }
        self.pleLayerIndices = args.pleLayerIndices
        super.init()
    }

    /// Install the n-gram row source on every PLE layer.
    public func install(ngramRowSource source: Qwen4ExpNGramRowSource) {
        for index in pleLayerIndices where index < layers.count {
            layers[index].ple?.pleEmbedding.install(rowSource: source)
        }
    }

    /// The PLE layers, for a runtime that needs their table geometry.
    public var pleEmbeddings: [Qwen4ExpNGramEmbedding] {
        pleLayerIndices.compactMap { $0 < layers.count ? layers[$0].ple?.pleEmbedding : nil }
    }

    public func callAsFunction(
        _ ids: MLXArray,
        cache: [KVCache?]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        streams(ids, cache: cache, inputEmbeddings: inputEmbeddings).mixed
    }

    /// Both tower outputs.
    ///
    /// - `mixed` is the collapsed hidden state the head reads.
    /// - `multi` is the hyper-connection stream BEFORE the final mixer, which
    ///   is what the native MTP head consumes. Keeping it costs nothing: the
    ///   mixer reads it anyway.
    public func streams(
        _ ids: MLXArray,
        cache: [KVCache?]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> (mixed: MLXArray, multi: MLXArray) {
        var hidden = inputEmbeddings ?? embedTokens(ids)
        let caches = cache ?? Array(repeating: nil as KVCache?, count: layers.count)

        let mask = makeAttentionMask(
            n: hidden.dim(1), cache: firstFullAttentionIndex.flatMap { caches[$0] })
        // The deltanet reads the tokens in order, so a padded batch must have
        // the padding zeroed. Its own cache carries the per-row lengths.
        let convMask =
            firstLinearIndex
            .flatMap { caches[$0] as? Qwen4ExpLayerCache }?
            .makeMask(N: hidden.dim(1))

        // The n-gram hash reads `ids` in order, so the history has to be
        // carried across calls. It lives in the PLE layer's own cache.
        var previousContext: MLXArray? = nil
        if let pleIndex = pleLayerIndices.first, pleIndex < layers.count {
            let pleCache = caches[pleIndex] as? Qwen4ExpLayerCache
            let carried = pleCache?[Qwen4ExpLayerCache.ngramHistorySlot]
            let context =
                carried
                ?? MLXArray.full(
                    [ids.dim(0), contextLength],
                    values: MLXArray(int64: args.eosTokenId), dtype: ids.dtype)
            previousContext = context
            if let pleCache {
                let history = concatenated([context, ids], axis: 1)
                pleCache[Qwen4ExpLayerCache.ngramHistorySlot] =
                    history[0..., (-contextLength)...]
            }
        }

        hidden = tiled(hidden, repetitions: [1, 1, args.hcCount])
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                rope: rope,
                mask: mask,
                convMask: convMask,
                cache: caches[index],
                ids: ids,
                previousContext: previousContext
            )
        }
        return (hyperConnectionMixer(hidden), hidden)
    }
}

// MARK: - Model

public final class Qwen4ExpModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "model") public var model: Qwen4ExpTower
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// The native multi-token-prediction head, which lives in the SAME
    /// checkpoint as the target under `language_model.mtp.*`. It is a sibling
    /// of `model`, not a child of it, which is why it hangs here.
    @ModuleInfo(key: "mtp") public var mtp: Qwen4ExpMTPModule?

    public let configuration: Qwen4ExpTextConfiguration

    public convenience init(_ args: Qwen4ExpConfiguration) {
        self.init(text: args.textConfig)
    }

    /// - Parameters:
    ///   - args: the text tower configuration.
    ///   - withMTP: build the native MTP head. On by default: this checkpoint
    ///     always carries it, and the track's speculative arm is that head.
    ///     Pass `false` for a serial-only load, which then drops the `mtp.*`
    ///     tensors at sanitize.
    ///   - mtpLayerCount: number of layers in the MTP head.
    public init(
        text args: Qwen4ExpTextConfiguration,
        withMTP: Bool = true,
        mtpLayerCount: Int = 1
    ) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self._model.wrappedValue = Qwen4ExpTower(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        if withMTP {
            _mtp.wrappedValue = Qwen4ExpMTPModule(args, layerCount: mtpLayerCount)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        head(model(inputs, cache: cache))
    }

    /// Both tower streams, for a speculative round: `mixed` feeds the head,
    /// `multi` feeds the MTP module.
    public func streams(_ inputs: MLXArray, cache: [KVCache]?)
        -> (mixed: MLXArray, multi: MLXArray)
    {
        model.streams(inputs, cache: cache)
    }

    /// Logits for a hidden state, from the target's own head.
    public func head(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    /// Caches for the MTP head's own layers.
    public func makeMTPCache() -> [KVCache] {
        mtp?.makeCache() ?? []
    }

    /// One MTP draft step.
    ///
    /// - Parameters:
    ///   - nextTokenIds: the ids the target just produced, `[B, S]`.
    ///   - multiStream: `multi` from `streams(_:cache:)` on the first step, and
    ///     the `multi` this call returns on every step after that.
    ///   - cache: the MTP head's own caches from ``makeMTPCache()``.
    ///   - stepIndex: which head layer runs, for a multi-layer head.
    /// - Returns: draft logits and the multi stream for the next step.
    public func mtpStep(
        nextTokenIds: MLXArray,
        multiStream: MLXArray,
        cache: [KVCache],
        stepIndex: Int = 0
    ) -> (logits: MLXArray, multi: MLXArray) {
        guard let mtp else {
            preconditionFailure(
                """
                Qwen4ExpModel: no MTP head is loaded. Build the model with                 withMTP: true to drive the native speculative arm.
                """)
        }
        let step = mtp(
            nextTokenIds: nextTokenIds,
            multiStream: multiStream,
            embedTokens: model.embedTokens,
            cache: cache,
            stepIndex: stepIndex
        )
        return (head(step.sample), step.multi)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { type in
            type == "full_attention" ? Qwen4ExpAttentionCache() : Qwen4ExpLayerCache()
        }
    }

    public func makeCache() -> [KVCache] {
        newCache(parameters: nil)
    }

    /// Install the n-gram row source. The forward pass refuses without one.
    public func install(ngramRowSource source: Qwen4ExpNGramRowSource) {
        model.install(ngramRowSource: source)
    }

    public var pleEmbeddings: [Qwen4ExpNGramEmbedding] { model.pleEmbeddings }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (rawKey, value) in weights {
            // Vision tower: this is a text-only port and the module tree has no
            // place for it.
            if rawKey.hasPrefix("vision_tower.") || rawKey.hasPrefix("visual.")
                || rawKey.hasPrefix("model.visual.")
            {
                continue
            }

            // The checkpoint nests everything under `language_model.`; an
            // upstream torch checkpoint writes `model.language_model.`.
            var key = rawKey
            if key.hasPrefix("model.language_model.") {
                key = "model." + key.dropFirst("model.language_model.".count)
            } else if key.hasPrefix("language_model.") {
                key = String(key.dropFirst("language_model.".count))
            }

            // The native MTP head is a sibling of `model`; an upstream torch
            // checkpoint nests it one level deeper. It is dropped only on a
            // serial-only load, where the module is absent.
            if key.hasPrefix("model.mtp.") {
                key = "mtp." + key.dropFirst("model.mtp.".count)
            }
            if key.hasPrefix("mtp.") && mtp == nil {
                continue
            }

            // The n-gram shards are never model parameters. They stay on disk
            // and reach the model through a `Qwen4ExpNGramRowSource`. The
            // loader already excludes them by name (`WeightNameFiltering`
            // below); this stays for the callers that sanitize weights they
            // read themselves.
            if Self.isNGramShard(key) {
                continue
            }

            if configuration.tieWordEmbeddings && key.hasPrefix("lm_head.") {
                continue
            }

            // Torch stores a depthwise kernel as (C, 1, K); MLX wants (C, K, 1).
            // Idempotent: an already converted weight has dim(1) == kernelSize.
            if key.hasSuffix("conv1d.weight") && value.ndim == 3 && value.dim(1) == 1 {
                out[key] = value.transposed(0, 2, 1)
                continue
            }

            out[key] = value
        }
        return out
    }
}

// MARK: - Weight name filter

extension Qwen4ExpModel: WeightNameFiltering {
    /// True for an n-gram shard tensor.
    ///
    /// The shards hold the n-gram table. They are not model parameters: the
    /// table reaches the model through a `Qwen4ExpNGramRowSource`, which reads
    /// the rows it needs from the files. Reading them here would cost the full
    /// table in memory for nothing.
    static func isNGramShard(_ name: String) -> Bool {
        name.contains(".ngram_embedding.shard_")
    }

    /// Keeps the n-gram shards out of the load, before the loader materializes
    /// them. The name is the name on the disk, so both checkpoint layouts
    /// (`language_model.` prefixed or not) match.
    public func shouldLoadWeight(named name: String) -> Bool {
        !Self.isNGramShard(name)
    }
}

// MARK: - LoRA

extension Qwen4ExpModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
