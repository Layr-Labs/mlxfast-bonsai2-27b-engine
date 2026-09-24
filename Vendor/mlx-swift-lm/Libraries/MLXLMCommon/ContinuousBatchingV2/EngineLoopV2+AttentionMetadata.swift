import MLX

extension EngineLoopV2 {
    /// Uses the same identity rule as the actual-logit hook: chained decode
    /// consumes one still-unconfirmed sample, selecting generatedCount + 1.
    func bindAttentionMetadata(caches: [CBv2AttendingLayerCache],
                               ids: [CBv2RequestID], chained: Bool)
        -> CBv2AttentionMetadataForward?
    {
        guard let state = attentionMetadata else { return nil }
        for id in ids where id.raw == state.configuration.requestID {
            guard let rec = scheduler.record(for: id),
                let forward = state.select(
                    requestID: id.raw,
                    outputIndex: rec.generatedTokenCount + (chained ? 1 : 0),
                    batchSize: ids.count, phase: chained ? "chained_decode" : "decode")
            else { return nil }
            guard rec.request.multimodal == nil,
                caches.count == state.expectedOwners.count,
                caches.count == layerKinds.count,
                caches.allSatisfy({ $0 is any CBv2AttentionMetadataBinding }) else {
                state.refuse("unsupported_cache_or_multimodal_request")
                forward.finish(succeeded: false)
                return nil
            }
            for (storage, cache) in caches.enumerated() {
                guard forward.bindOwner(cache: cache, storageLayerIndex: storage, kind: layerKinds[storage]) else {
                    forward.finish(succeeded: false)
                    return nil
                }
            }
            for cache in caches {
                (cache as? any CBv2AttentionMetadataBinding)?.attentionMetadata = forward
            }
            return forward
        }
        return nil
    }

    func finishAttentionMetadata(_ forward: CBv2AttentionMetadataForward?,
                                 caches: [CBv2AttendingLayerCache], succeeded: Bool) {
        guard let forward else { return }
        for cache in caches {
            (cache as? any CBv2AttentionMetadataBinding)?.attentionMetadata = nil
        }
        forward.finish(succeeded: succeeded)
    }
}
