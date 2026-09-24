// EngineLoopV2+PagedPrefix.swift
//
// Engine-queue adoption and finalized-block publication for the resident
// physical-page prefix cache. Exact step ranges keep chained N+1 work out of
// the index while N is being finalized.

import Foundation

enum CBv2ResidentAdoptionResult {
    case adopted
    case refused
}

extension EngineLoopV2 {
    /// Metadata only: deadline admission can price replay before retaining pages.
    func residentReusePlan(
        for match: CBv2PagedPrefixMatch, request: CBv2Request
    ) -> CBv2PrefixReusePlan? {
        guard let plan = prefixReuseCapability.plan(
            matchedBoundary: match.matchedTokens,
            maximumSequenceLength: request.promptTokens.count + max(request.maxTokens, 1),
            nominalFullKVBytesPerToken: nominalFullKVBytesPerToken,
            reserveFullSequenceTokens: scheduler.reserveFullSequenceTokens),
            plan.prefillTokensSaved > 0
        else { return nil }
        return plan
    }

    func applyResidentAdoption(
        _ match: CBv2PagedPrefixMatch, requestID: CBv2RequestID
    ) -> CBv2ResidentAdoptionResult {
        guard let residentPrefixBackend,
            let rec = scheduler.record(for: requestID),
            kvStates[requestID] == nil
        else {
            markPrefixAdoptionFailed(requestID, outcome: .adoptionFailed)
            return .refused
        }
        prefixUsageByID[requestID]?.matchedTokens = match.matchedTokens
        let maxLength = rec.request.promptTokens.count + max(rec.request.maxTokens, 1)
        guard let plan = residentReusePlan(for: match, request: rec.request) else {
            markPrefixAdoptionFailed(requestID, outcome: .skippedPolicy)
            return .refused
        }

        if let capacity {
            do {
                try capacity.reserve(
                    id: requestID,
                    additionalTokens: plan.capacityReservationTokens,
                    additionalBytes: plan.initialAdditionalCapacityBytes)
            } catch {
                markPrefixAdoptionFailed(requestID, outcome: .skippedCapacity)
                return .refused
            }
        }
        do {
            let state = try residentPrefixBackend.makeSequenceState(
                sharing: match, plan: plan, layerKinds: layerKinds,
                maxLength: maxLength)
            kvStates[requestID] = state
            rec.numComputedTokens = plan.replayStart
            rec.prefixReusePlan = plan
            prefixHitTokens[requestID] = plan.prefillTokensSaved
            prefixUsageByID[requestID]?.outcome = .hit
            prefixUsageByID[requestID]?.tier = .resident
            prefixUsageByID[requestID]?.prefillTokensSaved = plan.prefillTokensSaved
            prefixUsageByID[requestID]?.strategy = plan.strategy
            prefixUsageByID[requestID]?.replayTokens = plan.replayTokens
            if var cursor = residentPrefixCursorByID[requestID] {
                // Tail replay installs only [0,C); [C,M) is recomputed into
                // new pages and must be allowed to replace the matched donor's
                // aliases. Direct/frozen plans restore through M, so this is
                // the full matched count for those strategies.
                cursor.publishedBlockCount =
                    plan.restoredFullTokens / cursor.hasher.blockSize
                residentPrefixCursorByID[requestID] = cursor
            }
            return .adopted
        } catch {
            capacity?.unreserve(
                id: requestID,
                tokens: plan.capacityReservationTokens,
                bytes: plan.initialAdditionalCapacityBytes)
            let outcome: CBv2PrefixCacheOutcome
            if let kvError = error as? CBv2KVError, case .capacityExhausted = kvError {
                outcome = .skippedCapacity
            } else {
                outcome = .adoptionFailed
            }
            markPrefixAdoptionFailed(requestID, outcome: outcome)
            return .refused
        }
    }

    /// Publish every newly completed whole block through `safeComputedEnd`.
    /// Hashes are extended incrementally; page metadata is captured without
    /// constructing or evaluating an MLXArray snapshot.
    func publishFinalizedResidentBlocks(
        requestID: CBv2RequestID, safeComputedEnd: Int,
        state explicitState: [CBv2SequenceKV?]? = nil
    ) {
        guard let residentPrefixBackend,
            var cursor = residentPrefixCursorByID[requestID],
            let rec = scheduler.record(for: requestID),
            let state = explicitState ?? kvStates[requestID],
            rec.request.prefixCacheEnabled,
            rec.request.multimodal == nil
        else { return }

        let safeEnd = min(max(0, safeComputedEnd), rec.tokens.count)
        let completedBlocks = cursor.hasher.fullBlockCount(tokenCount: safeEnd)
        guard completedBlocks > cursor.publishedBlockCount else { return }

        while cursor.chainHashes.count < completedBlocks {
            let blockIndex = cursor.chainHashes.count
            let start = blockIndex * cursor.hasher.blockSize
            let end = start + cursor.hasher.blockSize
            guard end <= rec.tokens.count else { return }
            let hash = cursor.hasher.blockHash(
                parent: cursor.chainHashes.last,
                blockTokens: rec.tokens[start ..< end],
                blockIndex: blockIndex)
            cursor.chainHashes.append(hash)
        }

        let range = cursor.publishedBlockCount ..< completedBlocks
        let published = residentPrefixBackend.publishResidentPrefixBlocks(
            state: state, chainHashes: cursor.chainHashes, blockIndices: range)
        cursor.publishedBlockCount += published
        residentPrefixCursorByID[requestID] = cursor
    }

    /// A preempted row recomputes from zero. Previously published entries stay
    /// valid until their pages are reused, but the new row must be allowed to
    /// publish replacement physical realizations.
    func resetResidentPrefixPublication(_ requestID: CBv2RequestID) {
        guard var cursor = residentPrefixCursorByID[requestID] else { return }
        cursor.publishedBlockCount = 0
        residentPrefixCursorByID[requestID] = cursor
    }
}
