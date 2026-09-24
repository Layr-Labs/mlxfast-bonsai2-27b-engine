import Foundation

extension EngineV2 {
    public func setResidentPrefixPublicationHandler(
        _ handler: (@Sendable (CBv2RequestID, [Int]) -> Void)?
    ) {
        hybridPrefixCache?.setPublicationHandler(handler)
    }

    func hybridPrefixLookup(
        for request: CBv2Request, cache: CBv2HybridPrefixCache
    ) -> CBv2PrefixLookup {
        guard request.prefixCacheEnabled, request.multimodal == nil, request.positionState == nil else {
            return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0)
        }
        let (maximumSequenceLength, overflow) = request.promptTokens.count.addingReportingOverflow(
            max(1, request.maxTokens))
        guard !overflow else {
            return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0)
        }
        let maximumChunk = max(
            schedulerConfig.prefillChunkSize, schedulerConfig.soloPrefillStripeTokens ?? 0)
        guard let hit = cache.lookup(
            tokens: request.promptTokens, cacheSalt: request.cacheSalt,
            maximumChunkSize: maximumChunk)
        else { return .init(adoption: nil, outcome: .miss, matchedTokens: 0) }
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: layerKinds, backend: .contiguousUnquantized)
        guard var plan = capability.plan(
            matchedBoundary: hit.checkpoint.position,
            exactStagedFullKVBytes: hit.kvPrefix.reduce(0) { total, row in
                total + (row.map { $0.keys.nbytes + $0.values.nbytes } ?? 0)
            },
            maximumSequenceLength: maximumSequenceLength,
            // Native adoption builds a lazy copy into fresh request storage.
            // Keep its source backing covered even if the donor is evicted
            // before that graph finishes; do not infer dtype from slab slack.
            additionalStagedBackingBytes: hit.kvBackingBytes,
            reserveFullSequenceTokens: true),
            plan.strategy == .direct, plan.replayStart == hit.checkpoint.position
        else {
            cache.endAdoption(pin: hit.pin)
            return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: hit.checkpoint.position)
        }
        plan.recurrentChunkSize = hit.checkpoint.chunkSize
        plan.recurrentPromptLength = request.promptTokens.count
        return .init(
            adoption: .init(
                requestID: request.id, tokens: request.promptTokens,
                matched: hit.checkpoint.position, plan: plan, prefix: hit.kvPrefix,
                cacheSalt: request.cacheSalt,
                recurrentCheckpoint: hit.checkpoint, hybridPin: hit.pin),
            outcome: .adoptionFailed, matchedTokens: hit.checkpoint.position)
    }
}
