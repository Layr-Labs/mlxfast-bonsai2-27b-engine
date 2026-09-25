// EngineLoopV2+MTPFinalize.swift
//
// Finalize-time target-authoritative acceptance, streaming, and KV rollback.

import Foundation
import MLX

extension EngineLoopV2 {
    /// Minimum target top-K probability mass (parts-per-million) at the
    /// carry position before the next draft may score only the shortlist
    /// rows. Below this the shortlist would too often miss the token the
    /// target actually produces next, costing acceptance for no byte win —
    /// the round simply reads the full head instead. 0.90 keeps greedy
    /// production traffic (mass typically ≥0.99 at K=256) shortlisted while
    /// flat/uncertain positions fall back.
    static let mtpShortlistMassThresholdPPM: Int32 = 900_000

    /// Runs at the step's existing host-sync boundary after ordinary sampled
    /// rows finalize and before deferred KV releases.
    func finalizeMTPRound(_ step: CBv2InFlightStep) {
        guard let mtp, let round = step.mtpRound else { return }

        // Plain prompt/decode observations may own request-state arrays that
        // the just-fenced target graph referenced. Restore or release them
        // only now, never at cancellation time while evaluation is live.
        for observation in round.committedObservationRows {
            if step.discard.contains(observation.id)
                || scheduler.record(for: observation.id) == nil
            {
                round.deferredAssistantReleases.append(
                    observation.assistantState)
            } else {
                mtp.restoreAssistantState(
                    observation.assistantState, for: observation.id)
                if !["0", "false", "no", "off"].contains(
                    ProcessInfo.processInfo.environment["MLXFAST_DFLASH_PREFILL_CARRY"]?
                        .lowercased() ?? "")
                {
                    if mtp.blockDrafter != nil,
                        let rec = scheduler.record(for: observation.id),
                        rec.generatedTokenCount == 1,
                        rec.pendingSamples == 0,
                        rec.numComputedTokens == rec.tokens.count - 1,
                        !mtp.hasValidCarry(for: rec),
                        let lastToken = rec.tokens.last
                    {
                        let dummyHidden = MLXArray.zeros([1, 1, 1])
                        mtp.storeCarry(
                            id: observation.id,
                            token: lastToken,
                            hidden: dummyHidden,
                            tokensCount: rec.tokens.count,
                            kvOffset: rec.numComputedTokens)
                    }
                }
            }
        }

        // The ordinary finalize loop has confirmed each seed row's bonus.
        if let seedHidden = round.seedHidden {
            let seedPolicyTopTwo =
                round.seedPolicyTopTwoValues?.asArray(Float.self)
            if seedPolicyTopTwo != nil { CBv2CoreInstrumentation.recordHostSync() }
            for (id, decodeIndex) in round.seedRows {
                guard !step.discard.contains(id),
                    let rec = scheduler.record(for: id)
                else { continue }
                let margin = seedPolicyTopTwo.map { values in
                    let base = decodeIndex * 2
                    return Double(values[base] - values[base + 1])
                }
                mtp.storeCarry(
                    id: id, token: rec.tokens.last!,
                    hidden: seedHidden[decodeIndex ..< (decodeIndex + 1), 0..., 0...],
                    previousTopTwoMargin: margin,
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
                round.finalizedSeedIDs.insert(id)
            }
        }

        guard let verify = round.verify else { return }
        let k = verify.k
        // Host readbacks of the MTP round, each counted: an MTP-round
        // finalize adds up to three syncs to the step's one (seed policy
        // margin above, acceptance packet, verify policy margin). Serial
        // target verification adds one blocking eval per column at launch
        // (`EngineLoopV2+MTPTargetVerification`); a logprob segment adds
        // three readbacks (`CBv2Logprobs.assemble`); a round whose capture
        // could not be fenced adds one blocking eval (`CBv2MTPCaptureFence`
        // fallback in `EngineLoopV2+MTPExecution`).
        let host = verify.acceptancePacket.asArray(Int32.self)
        CBv2CoreInstrumentation.recordHostSync()
        let policyTopTwoHost = verify.policyTopTwoValues?.asArray(Float.self)
        if policyTopTwoHost != nil { CBv2CoreInstrumentation.recordHostSync() }
        let draftCount = verify.rows.count * k
        let targetWidth = 1 + k
        var anyRejected = false

        struct RowOutcome {
            let batchIndex: Int
            let metadata: CBv2MTPRoundInFlight.VerifyRow
            let rec: CBv2ScheduledRequest
            let targets: [Int]
            let accepted: Int
        }

        // Resolve each row's natural target-authoritative prefix, then choose
        // one committed width for the rectangular step.
        var outcomes: [RowOutcome] = []
        outcomes.reserveCapacity(verify.rows.count)
        var commonEmitted = targetWidth

        for (batchIndex, metadata) in verify.rows.enumerated() {
            let id = metadata.id
            if step.discard.contains(id) || scheduler.record(for: id) == nil {
                if let evaluations = verify.recurrentEvaluations[id] {
                    do {
                        for evaluation in evaluations.reversed() { try evaluation.rollback() }
                    } catch {
                        preconditionFailure(
                            "CBv2 recurrent MTP discard rollback failed for \(id): \(error)")
                    }
                }
                if let state = metadata.assistantState {
                    round.deferredAssistantReleases.append(state)
                }
                anyRejected = true
                continue
            }
            let rec = scheduler.record(for: id)!
            let drafts = (0 ..< k).map { Int(host[batchIndex * k + $0]) }
            let targets = (0 ..< targetWidth).map {
                Int(host[draftCount + batchIndex * targetWidth + $0])
            }

            var accepted = 0
            while accepted < k, targets[accepted] == drafts[accepted] { accepted += 1 }
            var naturalEmitted = accepted + 1
            naturalEmitted = min(
                naturalEmitted,
                rec.request.maxTokens - rec.generatedTokenCount)
            if let stopIndex = targets[..<naturalEmitted].firstIndex(where: {
                rec.request.stopTokens.contains($0)
            }) {
                naturalEmitted = stopIndex + 1
            }
            commonEmitted = min(commonEmitted, naturalEmitted)
            outcomes.append(
                RowOutcome(
                    batchIndex: batchIndex,
                    metadata: metadata,
                    rec: rec,
                    targets: targets,
                    accepted: accepted))
        }

        round.finalizedVerifyIDs = Set(outcomes.map { $0.metadata.id })
        round.claimedSeedCostNanos = mtp.claimPendingSeedCost(
            decodeRowBucket: mtp.planDecodeRowBucket,
            finalizedVerifyIDs: round.finalizedVerifyIDs,
            measurement: step.mtpMeasurement)

        if !outcomes.isEmpty {
            let stepAccepted = outcomes.map { min($0.accepted, commonEmitted) }.min() ?? 0
            let observedDrafts =
                commonEmitted <= stepAccepted
                ? commonEmitted : min(k, stepAccepted + 1)
            mtp.recordStepAcceptance(
                drafted: k,
                accepted: stepAccepted,
                observedDrafts: observedDrafts,
                decodeRowBucket: mtp.planDecodeRowBucket,
                measurement: step.mtpMeasurement)
        }

        for outcome in outcomes {
            let batchIndex = outcome.batchIndex
            let metadata = outcome.metadata
            let id = metadata.id
            let rec = outcome.rec
            let accepted = outcome.accepted
            let emitted = Array(outcome.targets.prefix(commonEmitted))

            // Confirm in order with the same stop and length semantics as the
            // ordinary finalize loop.
            let detokenizer = detokenizers[id]
            let hasStopStrings = !rec.request.stopStrings.isEmpty
            var kept: [Int] = []
            var textPieces: [String] = []
            var finishReason: CBv2FinishReason?
            for token in emitted {
                scheduler.recordSampled(id: id, token: token)
                kept.append(token)
                if rec.request.stopTokens.contains(token) {
                    finishReason = .stop
                    break
                }
                if hasStopStrings {
                    textPieces.append(detokenizer?.push([token]) ?? "")
                    if detokenizer?.matchedStopString == true {
                        finishReason = .stop
                        break
                    }
                }
                if rec.generatedTokenCount >= rec.request.maxTokens {
                    finishReason = .length
                    break
                }
            }

            // Correct KV and scheduler state before any terminal release.
            let confirmed = kept.count
            round.committedVerifyTokenCount += kept.filter {
                !rec.request.stopTokens.contains($0)
            }.count
            for packet in verify.diagnostics where packet.requestID == id {
                let drafts = (0..<k).map { Int(host[batchIndex * k + $0]) }
                packet.reconcile(
                    accepted: accepted, confirmed: confirmed, drafts: drafts, targets: outcome.targets)
            }
            let rejected = (1 + k) - confirmed
            if rejected > 0 {
                for sequence in metadata.storageRows { sequence.rollback(rejected) }
                anyRejected = true
            }
            for sequence in metadata.storageRows { sequence.commitSpeculativeWrite() }
            let committedDraftCount = min(accepted, max(0, confirmed - 1))
            if let stateful = mtp.drafter as? any CBv2MTPRequestStatefulDrafter,
                let state = metadata.assistantState
            {
                // Target KV truth is committed first (the recurrent commit
                // follows the early block proposal below; the drafter never
                // reads target recurrent state). The assistant
                // receives only accepted draft inputs and their trusted target
                // hidden rows; every speculative head suffix is discarded.
                //
                // A BLOCK drafter's next block reads the tapped context of
                // exactly the columns this round CONFIRMED — the reference's
                // `hidden = hidden[:, :accepted + 1, :]`. That is one column
                // wider than the chain drafter's accepted-draft slice,
                // because the correction token's own position is context for
                // the next block.
                let committedHidden: MLXArray
                if let blockContext = verify.blockContext {
                    committedHidden = blockContext[
                        batchIndex ..< batchIndex + 1,
                        CBv2MTPBlockContextIndex.confirmedColumns(
                            confirmed: confirmed, verifyWidth: targetWidth),
                        0...]
                } else {
                    committedHidden = verify.lastHidden[
                        batchIndex ..< batchIndex + 1,
                        0 ..< committedDraftCount,
                        0...]
                }
                stateful.finalizeRound(
                    requestState: state,
                    confirmedInputTokens: 1 + committedDraftCount,
                    committedDraftTokens: verify.draftIDs[
                        batchIndex ..< batchIndex + 1,
                        0 ..< committedDraftCount],
                    committedTargetHidden: committedHidden)
            }
            if rejected > 0 {
                scheduler.discardPendingSamples(id: id, count: rejected)
                scheduler.rollbackComputed(id: id, tokens: rejected)
            }
            // EARLY BLOCK PROPOSAL. Everything the next round's block drafter
            // reads is final here: the anchor is the carry token stored below
            // (`kept[confirmed - 1]`), the committed context was just queued
            // by `finalizeRound`, and the drafter cache sits at this row's
            // new committed length (`rec.numComputedTokens`, after the
            // rollback above). Proposing NOW, before the recurrent commit
            // below, the rest of finalize and the next step's planning, puts
            // the drafter forward on the GPU while the host does that work
            // instead of after it. The proposal is the same call on the same
            // inputs the next verify build makes (`mtpBuildVerifyGraph`), so
            // it is the same graph and the same draft; the target still
            // decides every token. Only a fixed-depth block leg with a full
            // next round ahead takes this path, so the next round's depth is
            // the depth proposed here.
            var earlyBlock: CBv2MTPEarlyBlockProposal?
            if finishReason == nil, confirmed > 0, let block = mtp.blockDrafter,
                let state = metadata.assistantState,
                let fixedDepth = mtp.config.fixedDraftTokens, fixedDepth == k,
                rec.request.maxTokens - rec.generatedTokenCount > k
            {
                let anchor = kept[confirmed - 1]
                let kvOffset = rec.numComputedTokens
                if let tokens = try? block.proposeBlock(
                    anchor: anchor, depth: k, requestState: state)
                {
                    block.trimBlockState(state, toCommittedLength: kvOffset)
                    asyncEval([tokens] + block.evaluationTargets(for: state))
                    earlyBlock = CBv2MTPEarlyBlockProposal(
                        tokens: tokens, depth: k, anchor: anchor, kvOffset: kvOffset)
                }
            }
            if let evaluations = verify.recurrentEvaluations[id] {
                if evaluations.count == 1, evaluations[0].isCaptured {
                    // Capture-verify: one transaction spans the window. The
                    // accepted prefix commits by device-side selection of the
                    // captured state at position `confirmed`; zero confirmed
                    // tokens restore the pre-verify snapshot by rollback.
                    do {
                        if confirmed > 0 {
                            try evaluations[0].commit(keepPositions: confirmed)
                            if !["0", "false", "no", "off"].contains(
                                ProcessInfo.processInfo.environment["BONSAI_EARLY_REPLAY"]?
                                    .lowercased() ?? "")
                            {
                                if let committedStates = recurrentStates[id]?.confirmedStateSnapshot() {
                                    let arrays = committedStates.values.flatMap {
                                        [$0.conv, $0.ssm].compactMap { $0 }
                                    }
                                    if !arrays.isEmpty {
                                        asyncEval(arrays)
                                    }
                                }
                            }
                        } else {
                            try evaluations[0].rollback()
                        }
                    } catch {
                        preconditionFailure(
                            "CBv2 captured MTP finalization failed for \(id): \(error)")
                    }
                } else {
                    precondition(
                        evaluations.count == 1 + k,
                        "CBv2 recurrent MTP verification generation count mismatch")
                    do {
                        for evaluation in evaluations.suffix(rejected).reversed() {
                            try evaluation.rollback()
                        }
                        for evaluation in evaluations.prefix(confirmed) {
                            try evaluation.commit()
                        }
                    } catch {
                        preconditionFailure(
                            "CBv2 recurrent MTP finalization failed for \(id): \(error)")
                    }
                }
            }
            // The speculative suffix is now reconciled in BOTH the page tables
            // and scheduler record. Publish only the accepted frontier; doing
            // this before rollback would make a rejected block cache-visible.
            let launchedEnd = step.computedRanges[id]?.upperBound ?? rec.numComputedTokens
            publishFinalizedResidentBlocks(
                requestID: id,
                safeComputedEnd: min(launchedEnd, rec.numComputedTokens))

            if hasStopStrings {
                stream(for: id)?.emit(
                    .delta(text: textPieces.joined(), tokens: kept, logprobs: nil))
            } else {
                let stream = stream(for: id)
                stream?.reserveEmission()
                let endsWithStopToken = finishReason == .stop
                let pushTokens = endsWithStopToken ? Array(kept.dropLast()) : kept
                let allTokens = kept
                detokQueue.async {
                    let text = pushTokens.isEmpty ? "" : (detokenizer?.push(pushTokens) ?? "")
                    stream?.emit(
                        .delta(text: text, tokens: allTokens, logprobs: nil),
                        consumingReservation: true)
                }
            }

            let observedAccepted = min(accepted, confirmed)
            // Per-request timing: this verify row confirmed at the step's
            // readback-done instant (already read by `finalize`).
            rec.recordStepParticipation(step: step, batchRows: step.tokenProducingRows)
            rec.recordMTPRound(drafted: k, accepted: observedAccepted)
            if confirmed > 0 {
                rec.timing.decodeSteps &+= 1
                decodeRowsTotal = Self.saturatingAdd(decodeRowsTotal, 1)
            }
            // Acceptance/rollback audit record (observability): every value is
            // already on the host at this boundary. The scheduler fields are
            // read AFTER recordSampled/rollbackComputed above, so the record
            // states the row's post-round accounting — the boundary invariant
            // a consumer checks is
            // `numComputedAfter == tokensCountAfter - 1`.
            mtp.recordRound(
                drafted: k, accepted: observedAccepted, emitted: confirmed,
                audit: CBv2MTPRoundAuditRecord(
                    requestID: id.raw,
                    k: k,
                    draftTokens: Array(
                        host[batchIndex * k ..< (batchIndex + 1) * k].map(Int.init)),
                    targetTokens: outcome.targets,
                    accepted: accepted,
                    confirmed: confirmed,
                    rejected: rejected,
                    tokensCountAfter: rec.tokens.count,
                    numComputedAfter: rec.numComputedTokens,
                    generatedAfter: rec.generatedTokenCount,
                    finishReason: finishReason.map { String(describing: $0) }))
            let rejectionObserved = accepted < k && confirmed > accepted
            let acceptanceTruncated =
                !rejectionObserved && confirmed <= accepted && confirmed < k
            mtp.observeRequestAcceptance(
                id: id,
                draftedDepth: k,
                acceptedDepth: observedAccepted,
                rejectionObserved: rejectionObserved,
                endedByTruncation: acceptanceTruncated)

            if let finishReason {
                if let state = metadata.assistantState {
                    mtp.releaseDetachedAssistantState(state)
                }
                finishRequest(id, reason: finishReason)
            } else {
                if let state = metadata.assistantState {
                    mtp.restoreAssistantState(state, for: id)
                }
                // No inline deadline check: an MTP row that just confirmed
                // tokens is making progress and its decode lease is refreshed
                // in `refreshProgressLeases` (run at the end of the enclosing
                // `finalize`, after this round). Lease expiry is evaluated
                // centrally in `processLeaseExpiry` — identical typed-terminal
                // semantics to the ordinary decode path.
                let hiddenColumn = CBv2MTPHiddenIndex.carryColumn(
                    targetOutputIndex: confirmed - 1, draftDepth: k)
                // Shortlist coverage gate: hand the accepted position's
                // target top-K ids to the next draft only when their
                // probability mass (packet tail, parts-per-million) clears
                // the threshold; otherwise the draft falls back to the full
                // head for that round.
                var carryShortlist: MLXArray?
                if let shortlistIDs = verify.shortlistIDs {
                    let massBase = draftCount + verify.rows.count * targetWidth
                    let mass = host[massBase + batchIndex * targetWidth + hiddenColumn]
                    if mass >= Self.mtpShortlistMassThresholdPPM {
                        carryShortlist = shortlistIDs[batchIndex, hiddenColumn]
                    }
                }
                let previousTopTwoMargin: Double? = policyTopTwoHost.map { values in
                    let base =
                        (batchIndex * targetWidth + hiddenColumn) * 2
                    return Double(values[base] - values[base + 1])
                }
                mtp.storeCarry(
                    id: id, token: kept[confirmed - 1],
                    hidden: verify.lastHidden[
                        batchIndex ..< (batchIndex + 1),
                        hiddenColumn ..< (hiddenColumn + 1), 0...],
                    shortlist: carryShortlist,
                    previousTopTwoMargin: previousTopTwoMargin,
                    needsHistoryTransition: mtp.carryNeedsHistoryTransition,
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens,
                    earlyBlock: earlyBlock)
            }
        }

        // Verify rows emitted tokens without passing through the sampler;
        // drop their configured row state so the next ordinary sample
        // reconfigures from confirmed history (exact penalty counts and
        // per-request RNG step indices).
        if !outcomes.isEmpty {
            sampler.mtpRoundDidCommit(requestIDs: outcomes.map { $0.metadata.id })
        }

        // Rejected suffixes advanced eager device offsets past host truth.
        if anyRejected {
            eagerCompositionStale = true
        }
    }
}
