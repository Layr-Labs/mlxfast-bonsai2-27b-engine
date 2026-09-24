import MLX

extension EngineLoopV2 {
    /// Separate from metadata/logit bindings: these options may coexist without
    /// replacing each other's selected request, output, owner or capture state.
    func bindAttentionPacket(caches: [CBv2AttendingLayerCache], ids: [CBv2RequestID], chained: Bool)
        -> CBv2AttentionPacketForward? {
        guard let state = attentionPacket else { return nil }
        for id in ids where id.raw == state.configuration.requestID {
            guard let rec = scheduler.record(for: id),
                let forward = state.select(requestID: id.raw,
                    outputIndex: rec.generatedTokenCount + (chained ? 1 : 0),
                    batchSize: ids.count, phase: chained ? "chained_decode" : "decode")
            else { return nil }
            let storage = state.configuration.storageLayerIndex
            guard rec.request.multimodal == nil, caches.count == layerKinds.count,
                caches.indices.contains(storage),
                let binding = caches[storage] as? any CBv2AttentionPacketBinding else {
                state.refuse("unsupported_packet_cache_or_multimodal_request")
                forward.finish(succeeded: false)
                return nil
            }
            guard forward.metadata.bindOwner(cache: caches[storage], storageLayerIndex: storage,
                                             kind: layerKinds[storage]) else {
                forward.finish(succeeded: false)
                return nil
            }
            binding.attentionPacket = forward
            return forward
        }
        return nil
    }

    func finishAttentionPacket(_ forward: CBv2AttentionPacketForward?,
                               caches: [CBv2AttendingLayerCache], succeeded: Bool) {
        guard let forward else { return }
        for cache in caches {
            (cache as? any CBv2AttentionPacketBinding)?.attentionPacket = nil
        }
        forward.finish(succeeded: succeeded)
    }

    func materializeAttentionPacket(_ step: CBv2InFlightStep) {
        guard let packet = step.attentionPacket else { return }
        defer { step.attentionPacket = nil }
        packet.materialize(discarded: step.discard.contains(
            CBv2RequestID(packet.state.configuration.requestID)))
    }
}
