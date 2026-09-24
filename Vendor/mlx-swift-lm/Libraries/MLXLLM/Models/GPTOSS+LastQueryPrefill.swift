import MLX
import MLXLMCommon

/// A last-query capability is required; paged/legacy/custom caches without
/// it retain the existing full prompt path. No one-token or sliding call
/// can reach the capability's full-attention, kvL>1 preconditions.
func gptossLastQueryPrefillEligible(
    sequenceLength: Int, isFinalFullLayer: Bool, cache: KVCache?
) -> Bool {
    guard sequenceLength > 1, isFinalFullLayer,
          let cache = cache as? any CBv2LastQueryPrefillLayerCache else { return false }
    return cache.kind.attention == .full && cache.kind.sharesKVWithLayer == nil
}

extension AttentionBlock {
    /// Preserve ALL K/V projections, rotary positions and writes; evaluate
    /// Q and attention only for the final position. This intentionally changes
    /// Q/O projection geometry and is an experimental numeric arm.
    func prefillLastQuery(
        _ x: MLXArray, cache: any CBv2LastQueryPrefillLayerCache
    ) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1)
        let offsets = cache.positionOffsets + 0
        let tail = x[0..., (length - 1)..<length, 0...]
        var queries = qProj(tail).reshaped(batch, 1, -1, headDim).swappedAxes(1, 2)
        var keys = kProj(x).reshaped(batch, length, -1, headDim).swappedAxes(1, 2)
        let values = vProj(x).reshaped(batch, length, -1, headDim).swappedAxes(1, 2)
        queries = rope(queries, offset: offsets + Int32(length - 1))
        keys = rope(keys, offset: offsets)
        let attended = cache.updateAndAttendLastQuery(
            queries: queries, keys: keys, values: values,
            scale: smScale, sinks: sinksActiveResolved() ? sinks : nil)
        return oProj(attended.swappedAxes(1, 2).reshaped(batch, 1, -1))
    }
}

extension GPTOSSTransformerBlock {
    func prefillLastQuery(
        _ x: MLXArray, cache: any CBv2LastQueryPrefillLayerCache
    ) -> MLXArray {
        // Input normalization stays full-shape so every last-layer K/V
        // byte is produced by the unchanged projection input graph.
        let attention = selfAttn.prefillLastQuery(inputLayerNorm(x), cache: cache)
        let residual = x[0..., (x.dim(1) - 1)..<x.dim(1), 0...] + attention
        return residual + mlp(postAttentionLayerNorm(residual))
    }
}
