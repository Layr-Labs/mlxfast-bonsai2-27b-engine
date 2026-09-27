// EngineLoopV2+MTPExecution.swift
//
// MTP row classification and lazy MLX graph construction.

import Foundation
import MLX

/// On unless explicitly disabled: a round whose driver does not use the
/// marginal depth policy drops the dead verify top-two readback.
/// `DARKBLOOM_MTP_SKIP_DEAD_MARGIN=0` keeps the readback.
enum CBv2MTPDeadMarginSkip {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_MTP_SKIP_DEAD_MARGIN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
}

struct CBv2MTPRowWork {
    let rec: CBv2ScheduledRequest
    let start: Int
    let count: Int
    let samples: Bool
    let isDecode: Bool
    let isSeed: Bool
    /// Non-nil for verify rows: the consumed carry.
    let carry: CBv2MTPCarry?
    /// Verify-produced carry transition that must prime assistant history
    /// before this target-only row processes the carry token.
    let historyCarry: CBv2MTPCarry?
}

struct CBv2MTPGraphBuild {
    let sampledRows: [CBv2RequestID]
    let sampledTokens: MLXArray?
    /// Non-sampling prefill handles retained by the in-flight step.
    let prefillEvalTargets: [MLXArray]
    let asyncEvalTargets: [MLXArray]
    let diagnostics: [CBv2LogitDiagnosticPacket]
    let logprobSegments: [CBv2StepLogprobs]
    let verify: CBv2MTPRoundInFlight.Verify?
    let seedRows: [(id: CBv2RequestID, decodeIndex: Int)]
    let seedHidden: MLXArray?
    let seedPolicyTopTwoValues: MLXArray?
    let recurrentEvaluations: [CBv2RequestID: CBv2RecurrentStateEvaluation]
    let committedObservationRows: [CBv2MTPRoundInFlight.CommittedObservationRow]
    /// Prompt rows that sampled their first token in this step and can carry
    /// straight into a block-drafter round (no seed forward).
    let prefillCarries: [(id: CBv2RequestID, hidden: MLXArray)]
    /// Work submitted in its own command buffer AFTER `asyncEvalTargets`, so
    /// the step's sampled tokens never wait for it: a block drafter's
    /// absorption of a prompt's committed context.
    let lateEvalTargets: [MLXArray]
}

extension EngineLoopV2 {

    /// Record only the assignments the scheduler demoted. Known capacity
    /// reasons retain their provenance; an unclassified width mismatch means
    /// the reservation changed after MTP planning and is one step-level race.
    func mtpRecordSchedulerDemotions(_ plan: CBv2StepPlan) {
        guard let mtp else { return }
        var sawReservationRace = false
        for assignment in plan.assignments {
            guard let k = mtp.roundMark(for: assignment.id),
                assignment.numTokens != 1 + k
            else { continue }
            switch plan.speculationFallbacks[assignment.id] {
            case .tokenBudget: mtp.recordSkip("token_budget")
            case .kvHeadroom: mtp.recordSkip("kv_headroom")
            case nil: sawReservationRace = true
            }
        }
        if sawReservationRace {
            mtp.recordControllerFallback("step_reservation_race")
        }
    }

    func mtpPrepareRoundWork(
        _ plan: CBv2StepPlan,
        driver mtp: CBv2MTPRoundDriver,
        demoteAllRounds: Bool,
        launchNanos: UInt64
    ) -> [CBv2MTPRowWork] {
        var work: [CBv2MTPRowWork] = []
        work.reserveCapacity(plan.assignments.count)

        for (id, assignedTokens) in plan.assignments {
            guard let rec = scheduler.record(for: id) else { continue }
            // Admission stamp BEFORE `ensureKVState`, mirroring
            // `executeMixed`: a capacity-requeued row is then already
            // stamped, so its next waiting→running crossing counts as a
            // re-admission on both launch paths (same `readmissions`).
            rec.stampAdmission(launchNanos: launchNanos)
            guard ensureKVState(rec) != nil else { continue }
            var count = assignedTokens
            var preserveHistorySeed = false

            if let k = mtp.roundMark(for: id) {
                if demoteAllRounds {
                    if count == 1 + k { scheduler.rollbackComputed(id: id, tokens: k) }
                    count = 1
                    if mtp.tracksPersistentHistory {
                        preserveHistorySeed = true
                    } else {
                        mtp.invalidateCarry(id)
                    }
                } else if count == 1 + k, let carry = mtp.consumeCarry(for: id) {
                    work.append(
                        CBv2MTPRowWork(
                            rec: rec, start: rec.numComputedTokens - count, count: count,
                            samples: true, isDecode: false, isSeed: false,
                            carry: carry, historyCarry: nil))
                    continue
                } else if count == 1 + k {
                    // A marked assignment without a consumable carry demotes
                    // exactly like the scheduler's headroom retry.
                    scheduler.rollbackComputed(id: id, tokens: k)
                    count = 1
                    preserveHistorySeed = mtp.tracksPersistentHistory
                }
            }

            // Mirror executeMixed's row classification. MTP decode-shaped
            // work stays eager; final-token image spans remain prefill work.
            let start = rec.numComputedTokens - count
            let samples = rec.numComputedTokens == rec.effectiveTokenCount
            let finalTokenIsImageSpan =
                multimodalByID[id]?.containsSpan(at: rec.tokens.count - 1) ?? false
            let isDecode =
                count == 1 && samples && start == rec.tokens.count - 1 && !finalTokenIsImageSpan
            work.append(
                CBv2MTPRowWork(
                    rec: rec, start: start, count: count, samples: samples,
                    isDecode: isDecode,
                    isSeed:
                        isDecode && (mtp.isSeedMarked(id) || preserveHistorySeed),
                    carry: nil,
                    historyCarry: mtp.pendingHistoryCarry(for: id)))
        }
        return work
    }

    func mtpBuildRoundGraph(
        _ work: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver, launchNanos: UInt64
    ) throws -> CBv2MTPGraphBuild {
        var cacheInnerState: [MLXArray] = []
        var logprobSegments: [CBv2StepLogprobs] = []
        var diagnostics: [CBv2LogitDiagnosticPacket] = []

        // Plain and seed rows share one eager [B, 1] target batch. Seed rows
        // retain the pre-norm hidden; logits remain identical to plain eager.
        let decodeRows = work.filter(\.isDecode)
        var decodeSampled: MLXArray?
        var seedRows: [(id: CBv2RequestID, decodeIndex: Int)] = []
        var seedHidden: MLXArray?
        var seedPolicyTopTwoValues: MLXArray?
        var recurrentEvaluations: [CBv2RequestID: CBv2RecurrentStateEvaluation] = [:]
        var committedObservationRows: [CBv2MTPRoundInFlight.CommittedObservationRow] = []
        var committedObservationEvalTargets: [MLXArray] = []
        var observationsTransferred = false
        defer {
            if !observationsTransferred {
                // Keep detached owners alive until the engine fences already
                // submitted work, then retires the failed cohort explicitly.
                for observation in committedObservationRows {
                    mtp.restoreAssistantState(observation.assistantState, for: observation.id)
                }
            }
        }

        func observeCommittedTarget(
            row: CBv2MTPRowWork, tokens: MLXArray, hidden: MLXArray
        ) throws {
            guard mtp.tracksPersistentHistory, mtpBasicEligible(row.rec),
                let state = try mtp.takeOrMakeAssistantState(
                    for: row.rec.id,
                    maximumSequenceLength: row.rec.request.promptTokens.count
                        + max(row.rec.request.maxTokens, 1))
            else { return }
            if let carry = row.historyCarry {
                mtp.observeCommittedTarget(
                    id: row.rec.id,
                    observation: CBv2MTPCommittedTargetObservation(
                        tokens: MLXArray([Int32(carry.token)]).reshaped([1, 1]),
                        hidden: carry.hidden),
                    detachedState: state)
                committedObservationEvalTargets.append(carry.hidden)
            }
            mtp.observeCommittedTarget(
                id: row.rec.id,
                observation: CBv2MTPCommittedTargetObservation(
                    tokens: tokens, hidden: hidden),
                detachedState: state)
            committedObservationRows.append(
                .init(id: row.rec.id, assistantState: state))
            committedObservationEvalTargets.append(hidden)
        }
        if !decodeRows.isEmpty {
            let inputs = MLXArray(decodeRows.map { Int32($0.rec.tokens[$0.start]) })
                .reshaped([decodeRows.count, 1])
            let caches = eagerCaches(rowStates: decodeRows.map { kvStates[$0.rec.id]! })
            let diagnosticOffsets = logitDiagnostic == nil ? nil : decodeRows.map {
                Self.positionOffset(kvStates[$0.rec.id]!)
            }
            var diagnosticTopTwo: (ids: MLXArray, values: MLXArray)?
            let logits: MLXArray
            let hidden: MLXArray
            if let recurrentModel = mtp.model as? any CBv2RecurrentMTPSteppableModel,
                recurrentModel.recurrentStateSpec != nil
            {
                let evaluations = decodeRows.map { row -> CBv2RecurrentStateEvaluation in
                    guard let state = recurrentStates[row.rec.id] else {
                        preconditionFailure("CBv2 recurrent MTP seed state missing")
                    }
                    do { return try state.bind() } catch {
                        preconditionFailure("CBv2 recurrent MTP seed bind failed: \(error)")
                    }
                }
                let positionIds = CBv2PositionState.decodePositionIds(
                    states: decodeRows.map(\.rec.request.positionState),
                    cacheOffsets: decodeRows.map { Self.positionOffset(kvStates[$0.rec.id]!) })
                let output = try checkedModelForward(phase: decodeRows.allSatisfy { $0.start < $0.rec.request.promptTokens.count }
                    ? .prefill : (decodeRows.allSatisfy { $0.start >= $0.rec.request.promptTokens.count }
                        ? .decode : .mixedFrontier)) { recurrentModel.forwardWithHidden(
                    tokens: inputs, caches: caches, recurrentState: evaluations,
                    positionIds: positionIds) }
                logits = output.logits
                hidden = output.lastHidden
                for (row, evaluation) in zip(decodeRows, evaluations) {
                    do {
                        cacheInnerState.append(contentsOf: try evaluation.evaluate())
                    } catch {
                        preconditionFailure("CBv2 recurrent MTP seed evaluation failed: \(error)")
                    }
                    recurrentEvaluations[row.rec.id] = evaluation
                }
            } else {
                let output = try checkedModelForward(phase: decodeRows.allSatisfy { $0.start < $0.rec.request.promptTokens.count }
                    ? .prefill : (decodeRows.allSatisfy { $0.start >= $0.rec.request.promptTokens.count }
                        ? .decode : .mixedFrontier)) { mtp.model.forwardWithHidden(tokens: inputs, caches: caches) }
                logits = output.logits
                hidden = output.lastHidden
            }
            cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))
            decodeSampled = sampler.sample(
                logits: logits[0..., -1, 0...],
                params: decodeRows.map(\.rec.request.sampling),
                requestIDs: decodeRows.map(\.rec.id),
                stepIndex: stepCount,
                pendingSampledTokens: nil,
                rowContext: { decodeRows.map { Self.samplerRow($0.rec) } })
            if let stepLogprobs = sampler.takeStepLogprobs() {
                logprobSegments.append(stepLogprobs)
            }
            for (index, row) in decodeRows.enumerated() where row.isSeed {
                seedRows.append((id: row.rec.id, decodeIndex: index))
            }
            if !seedRows.isEmpty { seedHidden = hidden }
            if mtp.usesMarginalPolicy, !seedRows.isEmpty {
                guard let provider = mtp.model as? any CBv2MTPPolicyTopTwoProviding else {
                    preconditionFailure("CBv2 adaptive seed target lacks top-two provider")
                }
                let vocabulary = logits.dim(-1)
                let topTwo = provider.cbv2MTPTopTwo(
                    logits.reshaped([1, decodeRows.count, vocabulary]))
                if logitDiagnostic != nil {
                    diagnosticTopTwo = (
                        topTwo.ids.reshaped([decodeRows.count, 2]),
                        topTwo.values.reshaped([decodeRows.count, 2]))
                }
                seedPolicyTopTwoValues = topTwo.values
                    .reshaped([decodeRows.count, 1, 2])
                    .asType(.float32)
            }
            if let diagnosticOffsets, let diagnostic = logitDiagnostic {
                for (index, row) in decodeRows.enumerated()
                where row.rec.id.raw == diagnostic.configuration.requestID
                    && row.rec.generatedTokenCount == diagnostic.configuration.outputIndex {
                    let retainedTopTwo = diagnosticTopTwo.map {
                        (ids: $0.ids[index], values: $0.values[index])
                    }
                    if let packet = makeLogitDiagnostic(
                        logits: logits[index, -1], requestID: row.rec.id,
                        outputIndex: row.rec.generatedTokenCount,
                        phase: row.start < row.rec.request.promptTokens.count ? "prefill"
                            : row.isSeed ? "seed" : "plain",
                        batchIndex: index, batchSize: decodeRows.count,
                        seedToken: row.rec.tokens[row.start], cacheOffset: diagnosticOffsets[index],
                        policyTopTwo: retainedTopTwo)
                    { diagnostics.append(packet) }
                }
            }
            // What committed history CARRIES differs by drafter: a chain
            // drafter takes the pre-norm last hidden, a BLOCK drafter takes
            // the target's fused tapped context for the same positions. Read
            // it here, while this forward is still the target's last one.
            let observed =
                mtp.tracksPersistentHistory
                ? mtp.committedObservationHidden(hidden) : hidden
            for (index, row) in decodeRows.enumerated() {
                try observeCommittedTarget(
                    row: row,
                    tokens: inputs[index ..< index + 1, 0...],
                    hidden: observed[index ..< index + 1, 0..., 0...])
            }
        }

        // Chunked prefills remain per-request [1, chunk], matching executeMixed.
        var prefillSampled: [CBv2RequestID: MLXArray] = [:]
        var prefillEvalTargets: [MLXArray] = []
        var prefillCarries: [(id: CBv2RequestID, hidden: MLXArray)] = []
        var lateEvalTargets: [MLXArray] = []
        for row in work where !row.isDecode && row.carry == nil {
            let rec = row.rec
            let slice = rec.tokens[row.start ..< row.start + row.count]
            let inputs = MLXArray(slice.map(Int32.init)).reshaped([1, row.count])
            let caches = eagerCaches(rowStates: [kvStates[rec.id]!])
            let diagnosticOffset = logitDiagnostic == nil ? 0 : Self.positionOffset(kvStates[rec.id]!)
            let requirement: CBv2PrefillRequirement =
                row.samples ? .lastPositionLogits : .evaluationOnly
            let output: MLXArray
            var observedHidden: MLXArray?
            // ONE `multimodalByID` lookup per prefill row; the span test
            // iterates spans without allocating and runs only for rows that
            // carry multimodal input. The row's prefill-chunk timing stamp
            // rides the same binding (mirrors `executeMixed`).
            let multimodal = multimodalByID[rec.id]
            let visionChunk = multimodal?.hasSpans(start: row.start, count: row.count) ?? false
            rec.stampPrefillChunkLaunch(
                tokens: row.count, packed: false, vision: visionChunk,
                stripe: !visionChunk && row.count > scheduler.config.prefillChunkSize,
                launchNanos: launchNanos)
            if visionChunk, let multimodal {
                let forward = try multimodalChunkForward(
                    tokens: inputs, start: row.start, count: row.count,
                    id: rec.id, multimodal: multimodal, caches: caches,
                    requirement: requirement)
                output = forward.output
                cacheInnerState.append(contentsOf: forward.innerState)
                recurrentEvaluations.merge(forward.recurrent) { _, _ in
                    preconditionFailure("duplicate recurrent evaluation")
                }
            } else if mtp.tracksPersistentHistory,
                let recurrentModel = mtp.model as? any CBv2RecurrentMTPSteppableModel,
                recurrentModel.recurrentStateSpec != nil
            {
                guard let recurrentState = recurrentStates[rec.id] else {
                    preconditionFailure("CBv2 recurrent MTP prefill state missing")
                }
                let evaluation: CBv2RecurrentStateEvaluation
                do { evaluation = try recurrentState.bind() } catch {
                    preconditionFailure("CBv2 recurrent MTP prefill bind failed: \(error)")
                }
                let positions = rec.request.positionState?.promptSlice(
                    row.start ..< row.start + row.count)
                let forward = try checkedModelForward(phase: .prefill) { recurrentModel.forwardWithHiddenForPrefill(
                    tokens: inputs, caches: caches, recurrentState: [evaluation],
                    positionIds: positions, requirement: requirement) }
                output = narrowPrefillOutput(forward.logits, requirement: requirement)
                observedHidden = mtp.committedObservationHidden(forward.lastHidden)
                if row.samples, Self.mtpPrefillCarryEnabled, mtp.blockDrafter != nil {
                    let width = forward.lastHidden.dim(1)
                    prefillCarries.append(
                        (id: rec.id, hidden: forward.lastHidden[0..., (width - 1)..., 0...]))
                }
                do {
                    cacheInnerState.append(contentsOf: try evaluation.evaluate())
                } catch {
                    preconditionFailure(
                        "CBv2 recurrent MTP prefill evaluation failed: \(error)")
                }
                recurrentEvaluations[rec.id] = evaluation
            } else if let recurrentModel = model as? any CBv2RecurrentSteppableModel,
                recurrentModel.recurrentStateSpec != nil
            {
                let positions = rec.request.positionState?.promptSlice(
                    row.start ..< row.start + row.count)
                let forward = try targetForward(
                    tokens: inputs, caches: caches, ids: [rec.id],
                    positionIds: positions, phase: .prefill)
                output = narrowPrefillOutput(forward.logits, requirement: requirement)
                cacheInnerState.append(contentsOf: forward.innerState)
                recurrentEvaluations.merge(forward.recurrent) { _, _ in
                    preconditionFailure("duplicate recurrent evaluation")
                }
            } else if mtp.tracksPersistentHistory {
                let forward = try checkedModelForward(phase: .prefill) { mtp.model.forwardWithHidden(tokens: inputs, caches: caches) }
                output = narrowPrefillOutput(forward.logits, requirement: requirement)
                observedHidden = mtp.committedObservationHidden(forward.lastHidden)
            } else {
                output = try prefillOutput(
                    tokens: inputs, inputEmbeddings: nil, caches: caches,
                    requirement: requirement)
            }
            if let observedHidden {
                try observeCommittedTarget(row: row, tokens: inputs, hidden: observedHidden)
                // A prompt row that carries into a block round: its drafter
                // context is a function of the rows just observed, not of the
                // token this step samples, so it is absorbed now and
                // submitted behind the step (`lateEvalTargets`).
                if row.samples, Self.mtpPrefillCarryEnabled, let block = mtp.blockDrafter,
                    let observed = committedObservationRows.last, observed.id == rec.id
                {
                    lateEvalTargets.append(
                        contentsOf: block.prefetchCommittedContext(
                            requestState: observed.assistantState))
                }
            }
            cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))
            if row.samples {
                if logitDiagnostic != nil,
                    let packet = makeLogitDiagnostic(
                        logits: output[0], requestID: rec.id, outputIndex: rec.generatedTokenCount,
                        phase: "prefill", batchIndex: 0, batchSize: 1,
                        seedToken: slice.last, cacheOffset: diagnosticOffset)
                { diagnostics.append(packet) }
                prefillSampled[rec.id] = sampler.sample(
                    logits: output,
                    params: [rec.request.sampling],
                    requestIDs: [rec.id],
                    stepIndex: stepCount,
                    pendingSampledTokens: nil,
                    rowContext: { [Self.samplerRow(rec)] })
                if let stepLogprobs = sampler.takeStepLogprobs() {
                    logprobSegments.append(stepLogprobs)
                }
            } else {
                prefillEvalTargets.append(output)
            }
        }

        let verifyRows = work.filter { $0.carry != nil }
        let verify = try mtpBuildVerifyGraph(
            verifyRows, driver: mtp, cacheInnerState: &cacheInnerState)

        // Plain sampled tokens stay in plan order. Verify rows are finalized
        // from the target-authoritative acceptance packet instead.
        var pieces: [MLXArray] = []
        var sampledRows: [CBv2RequestID] = []
        var decodeIndex = 0
        for row in work {
            if row.isDecode {
                pieces.append(decodeSampled![decodeIndex ..< decodeIndex + 1])
                decodeIndex += 1
                sampledRows.append(row.rec.id)
            } else if let sampled = prefillSampled[row.rec.id] {
                pieces.append(sampled)
                sampledRows.append(row.rec.id)
            }
        }
        let sampledTokens: MLXArray? =
            pieces.isEmpty ? nil : (pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0))

        if let verify { diagnostics.append(contentsOf: verify.diagnostics) }
        var asyncEvalTargets = prefillEvalTargets
        for packet in diagnostics { asyncEvalTargets.append(contentsOf: packet.evaluationTargets) }
        if let sampledTokens { asyncEvalTargets.append(sampledTokens) }
        for segment in logprobSegments {
            asyncEvalTargets.append(contentsOf: segment.evalTargets)
        }
        if let verify {
            asyncEvalTargets.append(verify.acceptancePacket)
            asyncEvalTargets.append(verify.lastHidden)
            if let shortlistIDs = verify.shortlistIDs {
                asyncEvalTargets.append(shortlistIDs)
            }
            if let policyTopTwoValues = verify.policyTopTwoValues {
                asyncEvalTargets.append(policyTopTwoValues)
            }
            if let blockContext = verify.blockContext {
                asyncEvalTargets.append(blockContext)
            }
        }
        if let seedHidden { asyncEvalTargets.append(seedHidden) }
        if let seedPolicyTopTwoValues {
            asyncEvalTargets.append(seedPolicyTopTwoValues)
        }
        asyncEvalTargets.append(contentsOf: committedObservationEvalTargets)
        if !cacheInnerState.isEmpty {
            asyncEvalTargets.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }

        observationsTransferred = true
        return CBv2MTPGraphBuild(
            sampledRows: sampledRows,
            sampledTokens: sampledTokens,
            prefillEvalTargets: prefillEvalTargets,
            asyncEvalTargets: asyncEvalTargets,
            diagnostics: diagnostics,
            logprobSegments: logprobSegments,
            verify: verify,
            seedRows: seedRows,
            seedHidden: seedHidden,
            seedPolicyTopTwoValues: seedPolicyTopTwoValues,
            recurrentEvaluations: recurrentEvaluations,
            committedObservationRows: committedObservationRows,
            prefillCarries: prefillCarries,
            lateEvalTargets: lateEvalTargets)
    }

    /// A BLOCK drafter's first block needs only the prompt's tapped context
    /// and the prompt's sampled token as its anchor, so a prompt row can carry
    /// straight into a round: the one-token seed forward that re-established a
    /// carry after the prompt is skipped (its position is computed by that
    /// first round's verify instead). `DARKBLOOM_BONSAI_PREFILL_CARRY=0`
    /// restores the seed step.
    static let mtpPrefillCarryEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_PREFILL_CARRY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private func mtpBuildVerifyGraph(
        _ verifyRows: [CBv2MTPRowWork],
        driver mtp: CBv2MTPRoundDriver,
        cacheInnerState: inout [MLXArray]
    ) throws -> CBv2MTPRoundInFlight.Verify? {
        guard !verifyRows.isEmpty else { return nil }
        let draftStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let depths = Set(verifyRows.compactMap { mtp.roundMark(for: $0.rec.id) })
        precondition(depths.count == 1, "CBv2 MTP: one plan must use one uniform depth")
        let k = depths.first!
        let batch = verifyRows.count
        var captures: [CBv2MTPRowCapture] = []
        var rowMetadata: [CBv2MTPRoundInFlight.VerifyRow] = []
        var assistantOwnersTransferred = false
        defer {
            if !assistantOwnersTransferred {
                // Draft column zero may already be evaluating. Restore its
                // owner before unwinding; request retirement synchronizes and
                // calls releaseRequestState before returning capacity.
                for row in rowMetadata {
                    if let state = row.assistantState {
                        mtp.restoreAssistantState(state, for: row.id)
                    }
                }
            }
        }
        var seedTokens: [Int32] = []
        var carryHiddens: [MLXArray] = []
        // Each capture paired with the row it was gathered from, so
        // `mtpFreezeCaptures` can fence it against that row's own storage.
        var captured: [(row: CBv2SequenceKV, keys: MLXArray, values: MLXArray)] = []
        captures.reserveCapacity(batch)
        captured.reserveCapacity(2 * batch)

        for row in verifyRows {
            let state = kvStates[row.rec.id]!
            let carry = row.carry!
            if let captureLayers = mtp.captureLayers, !mtp.usesRequestStatefulDrafter {
                // Capture before target verification writes the speculative
                // columns; paged storage is fenced below before those writes.
                let fullRow = state[captureLayers.full]!
                let slidingRow = state[captureLayers.sliding]!
                precondition(
                    fullRow.absoluteOffset == carry.kvOffset,
                    "CBv2 MTP: verify row anchor \(fullRow.absoluteOffset) != carry \(carry.kvOffset)"
                )
                let fullSnapshot = fullRow.snapshot()
                let slidingSnapshot = slidingRow.snapshot()
                captures.append(
                    CBv2MTPRowCapture(
                        fullKeys: fullSnapshot.keys,
                        fullValues: fullSnapshot.values,
                        slidingKeys: slidingSnapshot.keys,
                        slidingValues: slidingSnapshot.values,
                        slidingStart: slidingRow.absoluteOffset - slidingRow.retainedCount,
                        anchor: fullRow.absoluteOffset))
                captured.append((fullRow, fullSnapshot.keys, fullSnapshot.values))
                captured.append((slidingRow, slidingSnapshot.keys, slidingSnapshot.values))
            } else {
                precondition(
                    state.compactMap { $0 }.allSatisfy { $0.absoluteOffset == carry.kvOffset },
                    "CBv2 request-stateful MTP target KV is not aligned with its carry")
            }
            rowMetadata.append(
                CBv2MTPRoundInFlight.VerifyRow(
                    id: row.rec.id, storageRows: state.compactMap { $0 },
                    assistantState:
                        mtp.usesRequestStatefulDrafter
                        ? mtp.takeAssistantState(for: row.rec.id) : nil))
            if let assistantState = rowMetadata.last?.assistantState,
                let stateful = mtp.drafter as? any CBv2MTPRequestStatefulDrafter
            {
                try stateful.configureRequestState(
                    assistantState,
                    maximumSequenceLength: row.rec.request.promptTokens.count
                        + max(row.rec.request.maxTokens, 1))
            }
            seedTokens.append(Int32(carry.token))
            carryHiddens.append(carry.hidden)
        }

        let includesAssistantPrefill = rowMetadata.contains {
            $0.assistantState?.hasPendingPrefillForCostAccounting == true
        }
        mtpFreezeCaptures(captured)
        let seedColumn = MLXArray(seedTokens).reshaped([batch, 1])
        var draftSteps: [MLXArray] = []
        draftSteps.reserveCapacity(k)
        var assistantEvalTargets: [MLXArray] = []
        var blockDraftIDs: MLXArray?
        if let block = mtp.blockDrafter {
            // ONE propose per round. The block is the row's last committed
            // token followed by k mask tokens, and the drafter's single
            // forward reads the mask positions; there is no chain to step, so
            // depth costs one drafter forward rather than k.
            var proposals: [MLXArray] = []
            proposals.reserveCapacity(batch)
            for (index, row) in verifyRows.enumerated() {
                guard let requestState = rowMetadata[index].assistantState else {
                    preconditionFailure(
                        "CBv2 block MTP assistant state missing for \(row.rec.id)")
                }
                let carry = row.carry!
                let proposal: MLXArray
                if let early = carry.earlyBlock {
                    // Proposed and submitted at the previous round's finalize
                    // with this carry's anchor and offset (`storeCarry`
                    // checks both), and the drafter cache already trimmed.
                    // A fixed-depth leg plans that same depth; a smaller
                    // plan (never taken while the early gate holds) reads a
                    // prefix of the block, which is still only a proposal.
                    precondition(
                        k <= early.depth,
                        "CBv2 block MTP: round depth \(k) exceeds early proposal \(early.depth)")
                    proposal = k == early.depth ? early.tokens : early.tokens[0..., ..<k]
                } else {
                    proposal = try block.proposeBlock(
                        anchor: carry.token, depth: k, requestState: requestState)
                    // Align the drafter's context cache with the TARGET's
                    // committed length, exactly where the reference does it:
                    // after the proposal absorbed this round's context rows.
                    // `kvOffset` IS that length (the row's `numComputedTokens`
                    // when the carry was captured).
                    block.trimBlockState(
                        requestState, toCommittedLength: carry.kvOffset)
                }
                proposals.append(proposal)
                assistantEvalTargets.append(proposal)
                assistantEvalTargets.append(
                    contentsOf: block.evaluationTargets(for: requestState))
            }
            let batched =
                proposals.count == 1 ? proposals[0] : concatenated(proposals, axis: 0)
            // The block proposal already has the [B, k] draft-ID layout; keep
            // it instead of re-stacking its columns (terrapinelf `7502085`).
            blockDraftIDs = batched
            // The whole block is known before target construction starts, so
            // publish it now; finalization still joins it through the
            // acceptance packet.
            asyncEval(assistantEvalTargets)
            mtp.recordEarlyDraftSubmission()
            for position in 0 ..< k {
                draftSteps.append(batched[0..., position])
            }
        } else if let stateful = mtp.drafter as? any CBv2MTPRequestStatefulDrafter {
            var currentTokens = (0 ..< batch).map {
                seedColumn[$0 ..< $0 + 1, 0...]
            }
            var currentHidden = carryHiddens
            for draftIndex in 0 ..< k {
                var nextRows: [MLXArray] = []
                var nextHiddens: [MLXArray] = []
                var stepEvalTargets: [MLXArray] = []
                nextRows.reserveCapacity(batch)
                nextHiddens.reserveCapacity(batch)
                for (index, row) in verifyRows.enumerated() {
                    guard let requestState = rowMetadata[index].assistantState else {
                        preconditionFailure(
                            "CBv2 request-stateful MTP assistant state missing for \(row.rec.id)")
                    }
                    let result = stateful.draftStep(
                        tokens: currentTokens[index],
                        hidden: currentHidden[index],
                        shortlist: draftIndex == 0 ? row.carry!.shortlist : nil,
                        requestState: requestState)
                    let next = result.tokens.reshaped([1])
                    nextRows.append(next)
                    nextHiddens.append(result.hidden)
                    stepEvalTargets.append(next)
                    stepEvalTargets.append(result.hidden)
                    stepEvalTargets.append(
                        contentsOf: stateful.evaluationTargets(for: requestState))
                }
                // Publish the first mutable head-cache generation before
                // constructing a deeper generation. This is nonblocking and
                // joins the round's sole finalize fence.
                if draftIndex == 0 {
                    asyncEval(stepEvalTargets)
                    mtp.recordEarlyDraftSubmission()
                }
                assistantEvalTargets.append(contentsOf: stepEvalTargets)
                let stepTokens = concatenated(nextRows, axis: 0)
                draftSteps.append(stepTokens)
                currentTokens = nextRows.map { $0.reshaped([1, 1]) }
                currentHidden = nextHiddens
            }
        } else {
            let prepared = mtp.drafter.prepare(rows: captures)
            var draftInput = seedColumn
            var draftHidden = concatenated(carryHiddens, axis: 0)
            for draftIndex in 0 ..< k {
                let (next, nextHidden) = mtp.drafter.draftStep(
                    tokens: draftInput, hidden: draftHidden, prepared: prepared)
                // Captures are already fenced. Overlap the read-only draft
                // with target graph construction; finalization still joins
                // this token through the acceptance packet. At depth one,
                // nextHidden is unused and need not be materialized.
                if draftIndex == 0 && mtp.drafter.supportsEarlyDraftSubmission {
                    asyncEval(next)
                    mtp.recordEarlyDraftSubmission()
                }
                draftSteps.append(next)
                draftInput = next.reshaped([batch, 1])
                draftHidden = nextHidden
            }
        }
        let draftIDs = blockDraftIDs ?? stacked(draftSteps, axis: 1)
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.draft.build", seconds: CFAbsoluteTimeGetCurrent() - draftStart)
        }

        // Windowed rows stage provisional writes; other supported storage
        // backends implement the transaction hooks as exact no-ops/rollback.
        let verifyStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        for metadata in rowMetadata {
            for sequence in metadata.storageRows { sequence.beginSpeculativeWrite() }
        }
        let targetColumns = [seedColumn] + draftSteps.map { $0.reshaped([batch, 1]) }

        let target = try mtpBuildTargetVerification(
            columns: targetColumns, rows: verifyRows, driver: mtp)
        cacheInnerState.append(contentsOf: target.cacheInnerState)
        cacheInnerState.append(contentsOf: assistantEvalTargets)
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.verify.build", seconds: CFAbsoluteTimeGetCurrent() - verifyStart)
        }
        var packetParts = [draftIDs.reshaped([-1]), target.scores.reshaped([-1])]
        if let shortlist = target.shortlist {
            packetParts.append(shortlist.massScaled.reshaped([-1]))
        }
        let acceptancePacket = concatenated(packetParts, axis: 0)
        assistantOwnersTransferred = true
        var result = CBv2MTPRoundInFlight.Verify(
            k: k,
            rows: rowMetadata,
            acceptancePacket: acceptancePacket,
            draftIDs: draftIDs,
            lastHidden: target.hidden,
            shortlistIDs: target.shortlist?.ids,
            recurrentEvaluations: target.recurrent,
            // The verify top-two values feed only the marginal depth policy
            // (`previousTopTwoMargin`); with a fixed draft depth nothing
            // reads them, so the round neither retains nor reads them back.
            policyTopTwoValues: (mtp.usesMarginalPolicy || !CBv2MTPDeadMarginSkip.enabled)
                ? target.policyTopTwo?.values : nil,
            blockContext: target.blockContext)
        result.diagnostics = target.diagnostics
        result.includesAssistantPrefill = includesAssistantPrefill
        return result
    }

    /// Freeze the round's pre-write KV captures against the in-place writes
    /// the very same graph is about to perform. See `CBv2MTPCaptureFence`
    /// for the hazard and the mechanism.
    private func mtpFreezeCaptures(
        _ captured: [(row: CBv2SequenceKV, keys: MLXArray, values: MLXArray)]
    ) {
        // Contiguous rows are ARC-owned by their views and need nothing.
        // `requiresMaterializedSnapshots` is the bit that already documents
        // exactly this recyclable-storage hazard: true for `PagedKVBackend`,
        // false everywhere else, so contiguous stays byte-identical.
        guard backend.requiresMaterializedSnapshots else { return }
        let unfenceable = CBv2MTPCaptureFence.publish(captured)
        if !unfenceable.isEmpty {
            // The fence contract's blunt fallback: one host sync, counted.
            eval(unfenceable)
            CBv2CoreInstrumentation.recordHostSync()
        }
    }

}
