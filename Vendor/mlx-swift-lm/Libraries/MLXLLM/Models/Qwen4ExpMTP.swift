//
//  Qwen4ExpMTP.swift
//  mlx-swift-lm
//
//  Native multi-token-prediction head of Qwen 3.8 Flash-Next.
//
//  The head is EMBEDDED in the target checkpoint under `language_model.mtp.*`
//  (76 tensors). There is no separate head artifact to stage and no dedicated
//  embedding table: the head rides the target's `embed_tokens` and the target's
//  `lm_head`.
//
//  REFERENCE AND LICENSE. The tensor names and shapes are read off the pinned
//  checkpoint. The wiring follows the Apache-2.0 vLLM reference
//  `vllm/models/qwen4_exp/nvidia/mtp.py` (vllm-project/vllm PR #53896, head
//  2a4cd640), which is the only permissively licensed description of this head.
//  The surrounding module shape follows this fork's own MTP lineage
//  (`Qwen35MTP.swift`, `DeepseekV4MTP.swift`). No AGPL-licensed source was read
//  or ported.
//
//  SHAPE, stated because it is the part that is easy to get wrong. The head
//  takes the target's PRE-final-mixer hyper-connection stream, which is
//  `hc_count * hidden` wide, NOT the collapsed tower output:
//
//     e  = fc_embedding(pre_fc_norm_embedding(embed_tokens(next_ids)))  [B,S,H]
//     h  = fc_hidden(pre_fc_norm_hidden(multi).reshape(B,S,hc,H))       [B,S,hc,H]
//     x  = (e broadcast over hc) + h                     -> flatten     [B,S,hc*H]
//     x  = layers[0](x)                                                 [B,S,hc*H]
//     sample = hyper_connection_mixer(x)                                [B,S,H]
//
//  `sample` goes to the target's head; `x` is the multi stream the NEXT draft
//  step consumes. `pre_fc_norm_hidden` normalizes the flat `hc*H` vector on ONE
//  statistic -- it is not a per-stream norm, unlike the hyper-connection norms.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The `mtp` submodule of the checkpoint.
public final class Qwen4ExpMTPModule: Module {
    let hcCount: Int
    let hiddenSize: Int

    @ModuleInfo(key: "pre_fc_norm_embedding") public var preFCNormEmbedding: Qwen4ExpRMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") public var preFCNormHidden: Qwen4ExpRMSNorm
    @ModuleInfo(key: "fc_embedding") public var fcEmbedding: Linear
    @ModuleInfo(key: "fc_hidden") public var fcHidden: Linear
    /// One full-attention layer, keyed `layers.0`, with no PLE.
    let layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var hyperConnectionMixer: Qwen4ExpGatedResidual

    let rope: Qwen4ExpRotary
    public let layerCount: Int

    public init(_ args: Qwen4ExpTextConfiguration, layerCount: Int = 1) {
        precondition(layerCount >= 1, "Qwen4ExpMTPModule needs at least one layer")
        self.hcCount = args.hcCount
        self.hiddenSize = args.hiddenSize
        self.layerCount = layerCount
        self.rope = Qwen4ExpRotary(dimensions: args.rotaryDimensions, base: args.ropeTheta)

        _preFCNormEmbedding.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps,
            weightOffset: args.rmsNormWeightOffset)
        // One statistic over the whole hc * hidden vector -- see the file note.
        _preFCNormHidden.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: args.hiddenSize * args.hcCount, eps: args.rmsNormEps,
            weightOffset: args.rmsNormWeightOffset)
        _fcEmbedding.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: false)
        _fcHidden.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: false)
        self.layers = (0 ..< layerCount).map { _ in
            Qwen4ExpDecoderLayer(args, isLinear: false, pleLayerIndex: nil)
        }
        _hyperConnectionMixer.wrappedValue = Qwen4ExpGatedResidual(args, useInject: false)
        super.init()
    }

    /// Caches for the head's own layers.
    public func makeCache() -> [KVCache] {
        (0 ..< layerCount).map { _ in Qwen4ExpAttentionCache() }
    }

    /// One draft step.
    ///
    /// - Parameters:
    ///   - nextTokenIds: the ids the target just produced, `[B, S]`.
    ///   - multiStream: the target's pre-final-mixer stream, `[B, S, hc * H]`.
    ///   - embedTokens: the TARGET's embedding table; the head has none.
    ///   - cache: the head's own caches, one per head layer.
    ///   - stepIndex: which head layer runs, for a multi-layer head. A
    ///     single-layer head ignores it.
    /// - Returns: `sample` `[B, S, H]` for the target head, and `multi`
    ///   `[B, S, hc * H]` for the next draft step.
    public func callAsFunction(
        nextTokenIds: MLXArray,
        multiStream: MLXArray,
        embedTokens: Embedding,
        cache: [KVCache],
        stepIndex: Int = 0
    ) -> (sample: MLXArray, multi: MLXArray) {
        let B = nextTokenIds.dim(0)
        let S = nextTokenIds.dim(1)

        let embedded = fcEmbedding(preFCNormEmbedding(embedTokens(nextTokenIds)))
        var stream = preFCNormHidden(multiStream).reshaped(B, S, hcCount, hiddenSize)
        stream = fcHidden(stream)
        stream = embedded[.ellipsis, .newAxis, 0...] + stream
        var hyper = stream.reshaped(B, S, hcCount * hiddenSize)

        let index = layerCount == 1 ? 0 : stepIndex % layerCount
        let layerCache: KVCache? = index < cache.count ? cache[index] : nil
        let mask = makeAttentionMask(n: S, cache: layerCache)
        hyper = layers[index](
            hyper,
            rope: rope,
            mask: mask,
            convMask: nil,
            cache: layerCache,
            ids: nextTokenIds,
            previousContext: nil
        )
        return (hyperConnectionMixer(hyper), hyper)
    }
}
