import Foundation

extension EngineV2 {
    /// Charge the first manifest decrypt before the provider has a validated
    /// import plan. This shares the slot's admission ceiling, including when
    /// the caller has no provider-wide budget. No file or model work occurs.
    public func reserveCompleteCheckpointReadScratch() throws -> CBv2CompleteCheckpointIOLease {
        guard let completeCheckpointCodec else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        if completeCheckpointCodec.admission.hasProcessMemoryOwner {
            // The provider must charge manifest/decrypt buffers before reading.
            // Expose this binding so a missing host authority fails before IO.
            return CBv2CompleteCheckpointIOLease(
                reservation: .init(onRelease: {}), usesProcessMemoryOwner: true)
        }
        return CBv2CompleteCheckpointIOLease(reservation: try completeCheckpointCodec.admission.reserveTransient(
            bytes: CBv2CompleteCheckpointManifest.maximumProviderScratchBytes))
    }

    /// Called from the asynchronous provider stage path; no model execution,
    /// file access or tensor allocation occurs while planning.
    public func planCompleteCheckpointImport(
        manifest: CBv2CompleteCheckpointManifest, request: CBv2Request
    ) throws -> CBv2CompleteCheckpointImportPlan {
        guard let completeCheckpointCodec else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        return try completeCheckpointCodec.plan(
            manifest: manifest, request: request,
            minimumChunkSize: schedulerConfig.prefillChunkSize,
            maximumChunkSize: max(schedulerConfig.prefillChunkSize, schedulerConfig.soloPrefillStripeTokens ?? 0))
    }

    public func setCompletePrefixPublicationHandler(
        _ handler: (@Sendable (CBv2RequestID, [Int]) -> Void)?
    ) {
        completeCheckpointCapture?.setPublicationHandler(handler)
    }

    func completeCheckpointLookup(for request: CBv2Request) -> CBv2PrefixLookup {
        guard request.prefixCacheEnabled, request.multimodal == nil, request.positionState == nil,
            let receiptID = request.prefixCacheReceiptID, let completePrefixCache
        else { return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0) }
        let (maximumLength, overflow) = request.promptTokens.count.addingReportingOverflow(max(1, request.maxTokens))
        guard !overflow else { return .init(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0) }
        guard let staged = completePrefixCache.takeStaged(
            requestID: receiptID, tokens: request.promptTokens, cacheSalt: request.cacheSalt,
            maximumSequenceLength: maximumLength)
        else { return .init(adoption: nil, outcome: .miss, matchedTokens: 0) }
        do {
            _ = try planCompleteCheckpointImport(manifest: staged.manifest, request: request)
            guard staged.maximumSequenceLength == maximumLength,
                  staged.codec === completeCheckpointCodec else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let matched = staged.manifest.position
            var plan: CBv2PrefixReusePlan
            if staged.codec.historicalLayout != nil {
                plan = try staged.codec.historicalReusePlan(position: matched, maximumSequenceLength: maximumLength)
            } else {
                let exactKVBytes = staged.manifest.tensors.reduce(0) { total, tensor in
                    total + ((tensor.role == .keys || tensor.role == .values) ? tensor.byteCount : 0)
                }
                let reuseBackend: CBv2PrefixReuseBackend = staged.usesPagedBacking ? .pagedFP16 : .contiguousUnquantized
                let capability = CBv2PrefixReuseCapability.derive(layerKinds: layerKinds, backend: reuseBackend)
                guard let recurrentPlan = capability.plan(
                    matchedBoundary: matched, exactStagedFullKVBytes: exactKVBytes,
                    maximumSequenceLength: maximumLength,
                    nominalFullKVBytesPerToken: staged.codec.admission.fullKVBytesPerToken,
                    reserveFullSequenceTokens: true),
                    recurrentPlan.strategy == .direct, recurrentPlan.replayStart == matched
                else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                plan = recurrentPlan
            }
            plan.recurrentChunkSize = staged.manifest.chunkSize
            plan.recurrentPromptLength = request.promptTokens.count
            return .init(
                adoption: .init(
                    requestID: receiptID, tokens: request.promptTokens, matched: matched,
                    plan: plan, prefix: [], cacheSalt: request.cacheSalt, completeCheckpoint: staged),
                outcome: .adoptionFailed, matchedTokens: matched)
        } catch {
            staged.close()
            return .init(adoption: nil, outcome: .adoptionFailed, matchedTokens: staged.manifest.position)
        }
    }
}
