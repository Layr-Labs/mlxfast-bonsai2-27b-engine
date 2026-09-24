import Foundation
import MLX

extension EngineLoopV2 {
    func captureRecurrentCheckpoints(_ step: CBv2InFlightStep) {
        guard hybridPrefixCache != nil || completeCheckpointCapture != nil else { return }
        var roots: [MLXArray] = []
        for (id, range) in step.computedRanges {
            guard !step.discard.contains(id), step.recurrentEvaluations[id] != nil,
                let rec = scheduler.record(for: id),
                rec.request.prefixCacheEnabled, rec.request.multimodal == nil,
                rec.request.positionState == nil, rec.preemptionCount == 0,
                let cap = step.recurrentCheckpointChunkSizes[id],
                cap >= scheduler.config.prefillChunkSize,
                CBv2AttentionV1.queryBlockSize <= 0 || cap % CBv2AttentionV1.queryBlockSize == 0
            else { continue }
            var geometry = recurrentCheckpointGeometry[id] ?? .init()
            let capture = geometry.record(
                range: range, cap: cap, promptLength: rec.request.promptTokens.count,
                packed: step.packedPrefixRows.contains(id))
            recurrentCheckpointGeometry[id] = geometry
            guard capture,
                let layers = recurrentStates[id]?.confirmedStateSnapshot()
            else { continue }
            if let completeCheckpointCapture {
                let assistantState = step.mtpRound?.committedObservationRows.first(where: { $0.id == id })?.assistantState
                roots.append(contentsOf: completeCheckpointCapture.capture(
                    requestID: id, position: range.upperBound, chunkSize: cap,
                    layers: layers, assistantState: assistantState))
                continue
            }
            // The durable codec already owns the loaded recurrent geometry.
            // Resolve the resident codec's spec only for an actual capture:
            // native model getters may build dtype-probe graphs, so reading
            // them on every decode finalization adds work without a checkpoint.
            guard let cache = hybridPrefixCache,
                let spec = (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec
            else { continue }
            var assistant: (any CBv2MTPPrefixCheckpoint)?
            if let mtp, mtp.tracksPersistentHistory {
                guard let drafter = mtp.drafter as? any CBv2MTPPrefixCheckpointDrafter,
                    let observation = step.mtpRound?.committedObservationRows.first(where: { $0.id == id }),
                    let checkpoint = drafter.capturePrefixCheckpoint(
                        requestState: observation.assistantState, targetInputCount: range.upperBound)
                else { continue }
                assistant = checkpoint
            }
            roots.append(contentsOf: cache.capture(
                requestID: id, position: range.upperBound, chunkSize: cap,
                spec: spec, layers: layers, assistant: assistant))
        }
        // Detach on the sole evaluator of the live graph. Publication/drop
        // may then wait for these events on another queue without a graph race.
        if !roots.isEmpty {
            if completeCheckpointCapture != nil {
                do { try withError { asyncEval(roots) } }
                catch {
                    for id in step.computedRanges.keys { discardHybridCheckpoints(id) }
                }
            } else {
                asyncEval(roots)
            }
        }
    }

    func adoptRecurrentCheckpoint(_ checkpoint: CBv2RecurrentCheckpoint, requestID: CBv2RequestID)
        throws
    {
        guard let spec = (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec,
            recurrentStates[requestID] == nil
        else { throw CBv2KVError.backendIneligible(reason: "invalid recurrent checkpoint target") }
        let recurrent = try CBv2RecurrentRequestState(spec: spec, adoptedCommitted: checkpoint.layers)
        let existing = recurrentStates.values.reduce(0) { Self.saturatingAdd($0, $1.byteCount) }
        let total = Self.saturatingAdd(backend.bytesReserved, Self.saturatingAdd(existing, recurrent.byteCount))
        guard total <= backend.bytesCapacity else {
            try recurrent.release()
            throw CBv2KVError.capacityExhausted(
                needed: recurrent.byteCount, available: max(0, backend.bytesCapacity - backend.bytesReserved - existing))
        }
        if let mtp, mtp.tracksPersistentHistory {
            guard let checkpoint = checkpoint.assistant,
                let drafter = mtp.drafter as? any CBv2MTPPrefixCheckpointDrafter,
                let assistant = drafter.restorePrefixCheckpoint(checkpoint)
            else {
                try recurrent.release()
                throw CBv2KVError.backendIneligible(reason: "assistant checkpoint cannot be restored")
            }
            mtp.restoreAssistantState(assistant, for: requestID)
        }
        recurrentStates[requestID] = recurrent
        recurrentCheckpointGeometry[requestID] = .init(
            position: checkpoint.position, chunkSize: checkpoint.chunkSize)
    }

    func hybridPublicationIntent(for rec: CBv2ScheduledRequest, reason: CBv2FinishReason)
        -> CBv2DonationIntent?
    {
        if completeCheckpointCapture?.hasCheckpoints(requestID: rec.id) == true {
            var intent = CBv2DonationIntent(
                requestID: rec.id, tokens: rec.request.promptTokens,
                cacheSalt: rec.request.cacheSalt, receiptID: rec.request.prefixCacheReceiptID)
            switch reason {
            case .stop, .length: intent.allowsCompletePublication = rec.generatedTokenCount > 0
            case .cancelled, .error, .terminal: intent.allowsCompletePublication = false
            }
            return intent
        }
        guard hybridPrefixCache?.hasStagedCheckpoints(requestID: rec.id) == true,
            rec.request.prefixCacheEnabled,
            rec.request.multimodal == nil, rec.request.positionState == nil,
            rec.generatedTokenCount > 0
        else { return nil }
        switch reason {
        case .stop, .length:
            return .init(
                requestID: rec.id, tokens: rec.request.promptTokens,
                cacheSalt: rec.request.cacheSalt, receiptID: rec.request.prefixCacheReceiptID)
        case .cancelled, .error, .terminal: return nil
        }
    }
}
