// Copyright © 2026 Eigen Labs.
//
// First-token deadline admission for the serialized-prefill scheduler.
//
// The scheduler owns only a deterministic work projection. Deadline policy
// (remaining duration and the caller's conservative service rate) stays in
// EngineV2, where enqueue, prefix adoption, the verdict, and activation are
// serialized against engine steps.

/// Caller-owned policy inputs for first-token deadline admission.
///
/// `deadline` is an absolute `ContinuousClock` instant. The engine reads its
/// own monotonic clock only after the serialized engine-queue closure starts;
/// time spent waiting for that closure therefore consumes the original budget
/// instead of receiving a fresh relative allowance.
///
/// Phase rates are optional conservative lower bounds. A projection that
/// contains work for a phase whose rate is nil fails closed. In particular, a
/// mixed prefill/decode posture needs both rates: the engine never prices
/// decode assignments at prefill throughput.
public struct CBv2FirstTokenDeadlineAdmission: Sendable, Equatable {
    public let deadline: ContinuousClock.Instant
    public let conservativePrefillTokensPerSecond: Double?
    public let conservativeDecodeTokensPerSecond: Double?

    public init(
        deadline: ContinuousClock.Instant,
        conservativePrefillTokensPerSecond: Double?,
        conservativeDecodeTokensPerSecond: Double?
    ) {
        self.deadline = deadline
        self.conservativePrefillTokensPerSecond = conservativePrefillTokensPerSecond
        self.conservativeDecodeTokensPerSecond = conservativeDecodeTokensPerSecond
    }
}

/// Scheduler work through the step that can produce the target's first token.
///
/// `mixedSteps` counts steps containing both phases. The service bound charges
/// phase envelopes serially:
///
///     T = prefillTokens / prefillRate + decodeTokens / decodeRate
///
/// This is deliberately conservative for one `asyncEval`: if phase execution
/// overlaps perfectly, the excess is at most the smaller phase envelope for
/// that step. Summed over the projection, false rejection from serialization
/// is therefore bounded by the sum of those smaller envelopes. If either
/// lower-bound phase rate is unavailable, mixed work is unbounded and
/// enforcement fails closed.
public struct CBv2FirstTokenScheduledWork: Sendable, Equatable {
    public let prefillTokens: Int
    public let decodeTokens: Int
    public let scheduledSteps: Int
    public let mixedSteps: Int

    public init(
        prefillTokens: Int,
        decodeTokens: Int,
        scheduledSteps: Int,
        mixedSteps: Int
    ) {
        self.prefillTokens = prefillTokens
        self.decodeTokens = decodeTokens
        self.scheduledSteps = scheduledSteps
        self.mixedSteps = mixedSteps
    }

    public var scheduledTokens: Int {
        let (total, overflow) = prefillTokens.addingReportingOverflow(decodeTokens)
        return overflow ? .max : total
    }
}

/// Conservative work through the step that can produce the new request's
/// first token.
public enum CBv2FirstTokenProjectedWork: Sendable, Equatable {
    /// `work` includes the whole target step, not just assignments preceding
    /// the target in plan order: all rows in one asyncEval complete at the
    /// same readback boundary.
    case bounded(
        work: CBv2FirstTokenScheduledWork,
        serviceDuration: Duration
    )
    /// The current scheduler state cannot produce a finite safe projection
    /// (for example, serialized prefill is not configured, all running slots
    /// are paused, the request is multimodal, or the projection complexity
    /// guard is reached). Deadline admission fails closed.
    case unbounded
}

/// Generation-bound acknowledgement that an admitted request no longer owns
/// scheduler, KV-ledger, or backend state. This is intentionally separate
/// from stream termination: a watchdog can deliver a terminal event while a
/// wedged engine step still references the row.
public struct CBv2RequestRetirement: Sendable {
    private let waitUntilRetired: @Sendable () async -> Void

    public init(waitUntilRetired: @escaping @Sendable () async -> Void) {
        self.waitUntilRetired = waitUntilRetired
    }

    init(stream: CBv2OutputStream) {
        self.init {
            await stream.waitUntilEngineOwnershipReleased()
        }
    }

    /// For implementations whose returned stream has no separately retained
    /// engine ownership (immediate results and protocol test doubles).
    public static var acknowledged: Self {
        Self(waitUntilRetired: {})
    }

    public func wait() async {
        await waitUntilRetired()
    }
}

/// Cancellation won after queue-side admission committed. The caller may
/// return promptly only by transferring both values to a cleanup owner that
/// cancels and retains `stream`, and retains external resources until
/// `retirement`.
public struct CBv2FirstTokenAdmissionCancellation: Error, Sendable {
    public let stream: AsyncStream<CBv2Event>
    public let retirement: CBv2RequestRetirement

    public init(
        stream: AsyncStream<CBv2Event>,
        retirement: CBv2RequestRetirement
    ) {
        self.stream = stream
        self.retirement = retirement
    }
}

/// Atomic submit result. A deadline rejection never enters a GPU prompt step.
public enum CBv2FirstTokenDeadlineResult: Sendable {
    case admitted(
        stream: AsyncStream<CBv2Event>,
        projectedWork: CBv2FirstTokenProjectedWork,
        admittedAt: ContinuousClock.Instant,
        retirement: CBv2RequestRetirement
    )
    case deadlineUnreachable(projectedWork: CBv2FirstTokenProjectedWork)
}

/// Scheduler-only projection before conversion through a caller-owned rate.
enum CBv2FirstTokenWorkProjection: Equatable {
    case bounded(
        work: CBv2FirstTokenScheduledWork,
        capacityOperations: [CBv2ProjectedCapacityOperation]
    )
    case unbounded
}

struct CBv2ProjectedCapacityReservation: Sendable, Equatable {
    let id: CBv2RequestID
    let additionalTokens: Int
    let additionalBytes: Int
}

enum CBv2ProjectedCapacityOperation: Sendable, Equatable {
    case reserve(CBv2ProjectedCapacityReservation)
    /// Runtime's chained-decode fast path executes only when its conservative
    /// headroom probe succeeds. A failed optional reservation skips that
    /// speculative chain; it does not make the target unreachable.
    case reserveIfAvailable(CBv2ProjectedCapacityReservation)
    case unreserve(CBv2ProjectedCapacityReservation)
    case release(CBv2RequestID)
}

private struct CBv2ProjectionRow {
    let id: CBv2RequestID
    let promptTokens: Int
    let maxTokens: Int
    let isPaused: Bool
    let cancelRequested: Bool
    let prefixReusePlan: CBv2PrefixReusePlan?
    let fullSequenceCapacityTokens: Int?
    var knownTokens: Int
    var computedTokens: Int
    var generatedTokens: Int

    var remainingKnownTokens: Int {
        knownTokens - computedTokens
    }
}

private struct CBv2ProjectionAssignment {
    let id: CBv2RequestID
    let count: Int
    let startComputedTokens: Int
    let knownTokensBeforeStep: Int
    let prefillCount: Int

    var decodeCount: Int {
        count - prefillCount
    }
}

extension SchedulerV2 {
    /// Project scheduler work through `id`'s first sampled token.
    ///
    /// This method is pure: it does not call `plan()`, reserve capacity, read a
    /// clock, or mutate records. `inFlightAssignments` are work already
    /// launched but not finalized. Their optimistic advances are subtracted
    /// to recover confirmed cursors, then the full assignments are charged
    /// once from the admission instant; no elapsed fraction is credited.
    ///
    /// The finite model assumes successful future execution and the current
    /// queue/pause/cancel state. Future arrivals, pause transitions, and
    /// capacity failures are external state changes, not facts available to
    /// an atomic snapshot.
    func firstTokenWorkProjection(
        for id: CBv2RequestID,
        inFlightAssignments: [(id: CBv2RequestID, numTokens: Int)] = [],
        inFlightAllowsChainedSuccessor: Bool = true,
        unmaterializedPrefixAdoption: Bool = false
    ) -> CBv2FirstTokenWorkProjection {
        guard config.maxConcurrentPartialPrefills == 1,
            config.maxConcurrentRequests > 0,
            config.maxBatchedTokensPerStep > 0,
            config.prefillChunkSize > 0,
            record(for: id) != nil
        else {
            return .unbounded
        }

        var inFlightByID: [CBv2RequestID: Int] = [:]
        for assignment in inFlightAssignments {
            guard assignment.numTokens > 0,
                let nextForID = Self.projectionAdd(
                    inFlightByID[assignment.id, default: 0], assignment.numTokens)
            else {
                return .unbounded
            }
            inFlightByID[assignment.id] = nextForID
        }
        var projectedPrefillTokens = 0
        var projectedDecodeTokens = 0
        var projectedSteps = 0
        var projectedMixedSteps = 0
        var projectedCapacityOperations: [CBv2ProjectedCapacityOperation] = []

        func chargeStep(_ assignments: [CBv2ProjectionAssignment]) -> Bool {
            var prefillTokens = 0
            var decodeTokens = 0
            for assignment in assignments {
                guard assignment.count > 0,
                    assignment.prefillCount >= 0,
                    assignment.prefillCount <= assignment.count,
                    let nextPrefill = Self.projectionAdd(
                        prefillTokens, assignment.prefillCount),
                    let nextDecode = Self.projectionAdd(
                        decodeTokens, assignment.decodeCount)
                else {
                    return false
                }
                prefillTokens = nextPrefill
                decodeTokens = nextDecode
            }
            guard !assignments.isEmpty,
                let nextPrefill = Self.projectionAdd(
                    projectedPrefillTokens, prefillTokens),
                let nextDecode = Self.projectionAdd(
                    projectedDecodeTokens, decodeTokens),
                let nextSteps = Self.projectionAdd(projectedSteps, 1)
            else {
                return false
            }
            projectedPrefillTokens = nextPrefill
            projectedDecodeTokens = nextDecode
            projectedSteps = nextSteps
            if prefillTokens > 0, decodeTokens > 0 {
                guard let nextMixed = Self.projectionAdd(projectedMixedSteps, 1) else {
                    return false
                }
                projectedMixedSteps = nextMixed
            }
            return true
        }

        func chargeDecodeStretch(tokens: Int, steps: Int) -> Bool {
            guard tokens > 0, steps > 0,
                let nextDecode = Self.projectionAdd(projectedDecodeTokens, tokens),
                let nextSteps = Self.projectionAdd(projectedSteps, steps)
            else {
                return false
            }
            projectedDecodeTokens = nextDecode
            projectedSteps = nextSteps
            return true
        }

        func boundedProjection() -> CBv2FirstTokenWorkProjection {
            .bounded(
                work: CBv2FirstTokenScheduledWork(
                    prefillTokens: projectedPrefillTokens,
                    decodeTokens: projectedDecodeTokens,
                    scheduledSteps: projectedSteps,
                    mixedSteps: projectedMixedSteps),
                capacityOperations: projectedCapacityOperations)
        }

        var rows: [CBv2RequestID: CBv2ProjectionRow] = [:]
        for rec in running + waiting {
            let assigned = inFlightByID[rec.id, default: 0]
            guard rec.numComputedTokens >= assigned else {
                return .unbounded
            }
            let confirmedComputed = rec.numComputedTokens - assigned
            guard confirmedComputed >= 0, confirmedComputed <= rec.tokens.count else {
                return .unbounded
            }
            // A pending sample without the launched assignment that owns it
            // cannot be placed on this projection's timeline.
            guard rec.pendingSamples == 0 || assigned > 0 else {
                return .unbounded
            }
            // Multimodal token counts are not a safe service-work proxy: image
            // spans can carry model-specific superlinear work.
            guard rec.cancelRequested || rec.multimodalBlocks.isEmpty else {
                return .unbounded
            }
            rows[rec.id] = CBv2ProjectionRow(
                id: rec.id,
                promptTokens: rec.request.promptTokens.count,
                maxTokens: max(0, rec.request.maxTokens),
                isPaused: rec.isPaused,
                cancelRequested: rec.cancelRequested,
                prefixReusePlan: rec.prefixReusePlan,
                fullSequenceCapacityTokens: rec.fullSequenceCapacityTokens,
                knownTokens: rec.tokens.count,
                computedTokens: confirmedComputed,
                generatedTokens: rec.generatedTokenCount)
        }

        if unmaterializedPrefixAdoption {
            guard let plan = rows[id]?.prefixReusePlan,
                plan.capacityReservationTokens >= 0,
                plan.initialAdditionalCapacityBytes >= 0
            else {
                return .unbounded
            }
            projectedCapacityOperations.append(
                .reserve(CBv2ProjectedCapacityReservation(
                    id: id,
                    additionalTokens: plan.capacityReservationTokens,
                    additionalBytes: plan.initialAdditionalCapacityBytes)))
        }

        var runningIDs = running.map(\.id)
        var waitingIDs = waiting.map(\.id)

        func capacityReservation(
            id reservationID: CBv2RequestID,
            start: Int,
            count: Int
        ) -> CBv2ProjectedCapacityReservation? {
            guard let row = rows[reservationID], start >= 0, count > 0 else {
                return nil
            }
            let tokens = CBv2ScheduledRequest.capacityTokensForChunk(
                start: start, count: count, plan: row.prefixReusePlan,
                fullSequenceTokens: row.fullSequenceCapacityTokens)
            let bytes =
                row.prefixReusePlan?.capacityBytesForChunk(start: start, count: count)
                ?? 0
            guard tokens >= 0, bytes >= 0 else { return nil }
            return CBv2ProjectedCapacityReservation(
                id: reservationID,
                additionalTokens: tokens,
                additionalBytes: bytes)
        }

        func appendCapacityReservation(
            id reservationID: CBv2RequestID,
            start: Int,
            count: Int
        ) -> Bool {
            guard let reservation = capacityReservation(
                id: reservationID, start: start, count: count)
            else {
                return false
            }
            if reservation.additionalTokens > 0 || reservation.additionalBytes > 0 {
                projectedCapacityOperations.append(.reserve(reservation))
            }
            return true
        }

        func appendOptionalCapacityReservation(
            id reservationID: CBv2RequestID,
            start: Int,
            count: Int
        ) -> Bool {
            guard let reservation = capacityReservation(
                id: reservationID, start: start, count: count)
            else {
                return false
            }
            if reservation.additionalTokens > 0 || reservation.additionalBytes > 0 {
                projectedCapacityOperations.append(.reserveIfAvailable(reservation))
            }
            return true
        }

        func appendCapacityUnreservation(
            id reservationID: CBv2RequestID,
            start: Int,
            count: Int
        ) -> Bool {
            guard let reservation = capacityReservation(
                id: reservationID, start: start, count: count)
            else {
                return false
            }
            if reservation.additionalTokens > 0 || reservation.additionalBytes > 0 {
                projectedCapacityOperations.append(.unreserve(reservation))
            }
            return true
        }

        func projectionAssignment(
            id assignmentID: CBv2RequestID,
            count: Int,
            row: CBv2ProjectionRow
        ) -> CBv2ProjectionAssignment? {
            guard count > 0,
                row.computedTokens >= 0,
                row.promptTokens >= 0
            else {
                return nil
            }
            // A replay assignment can cross the prompt/decode boundary.
            // Split its token charge at that exact cursor even though it is
            // one rectangular engine assignment and one scheduler step.
            let remainingPrompt = max(
                0, row.promptTokens - row.computedTokens)
            return CBv2ProjectionAssignment(
                id: assignmentID,
                count: count,
                startComputedTokens: row.computedTokens,
                knownTokensBeforeStep: row.knownTokens,
                prefillCount: min(count, remainingPrompt))
        }

        func removeTerminatedRows() {
            let terminated = Set(rows.values.compactMap { row -> CBv2RequestID? in
                row.cancelRequested || row.generatedTokens >= row.maxTokens ? row.id : nil
            })
            guard !terminated.isEmpty else { return }
            runningIDs.removeAll { terminated.contains($0) }
            waitingIDs.removeAll { terminated.contains($0) }
            for terminatedID in terminated {
                rows.removeValue(forKey: terminatedID)
                projectedCapacityOperations.append(.release(terminatedID))
            }
        }

        /// The engine may launch one pure-decode successor before finalizing
        /// the current samples. When a full running set keeps the target in
        /// waiting, that successor cannot include the target and still runs
        /// even if current finalization retires one of its rows. Charge the
        /// whole one-step-late batch; leaving surviving rows unadvanced is
        /// conservative and avoids depending on speculative acceptance.
        func chargeTerminalChainedStepIfNeeded(
            sampledIDs: [CBv2RequestID],
            allowsChainedSuccessor: Bool
        ) -> Bool {
            guard waitingIDs.contains(id),
                runningIDs.count >= config.maxConcurrentRequests,
                allowsChainedSuccessor
            else {
                return true
            }
            let activeDecodeIDs = runningIDs.compactMap { runningID -> CBv2RequestID? in
                guard let row = rows[runningID],
                    !row.isPaused,
                    !row.cancelRequested,
                    row.remainingKnownTokens == 1
                else {
                    return nil
                }
                return runningID
            }
            guard !activeDecodeIDs.isEmpty,
                runningIDs.allSatisfy({ runningID in
                    guard let row = rows[runningID] else { return false }
                    return row.isPaused
                        || (!row.cancelRequested && row.remainingKnownTokens == 1)
                }),
                activeDecodeIDs == sampledIDs,
                activeDecodeIDs.contains(where: { runningID in
                    guard let row = rows[runningID] else { return false }
                    return row.generatedTokens >= row.maxTokens
                })
            else {
                return true
            }
            for runningID in activeDecodeIDs {
                guard let row = rows[runningID],
                    appendOptionalCapacityReservation(
                        id: runningID,
                        start: row.computedTokens,
                        count: 1)
                else {
                    return false
                }
            }
            return chargeDecodeStretch(tokens: activeDecodeIDs.count, steps: 1)
        }

        /// Apply one executed assignment. Speculative decode may assign more
        /// slots than known inputs; only one output is credited, the minimum
        /// target-authoritative progress, while all assigned work is charged.
        func applyExecutedAssignment(
            id assignedID: CBv2RequestID,
            count: Int,
            startComputedTokens: Int,
            knownTokensBeforeStep: Int
        ) -> Bool? {
            guard var row = rows[assignedID] else { return false }
            guard startComputedTokens >= 0,
                knownTokensBeforeStep >= startComputedTokens,
                count > 0
            else {
                return nil
            }
            let remaining = knownTokensBeforeStep - startComputedTokens
            if count < remaining {
                guard let computed = Self.projectionAdd(startComputedTokens, count) else {
                    return nil
                }
                row.computedTokens = computed
                rows[assignedID] = row
                return false
            }
            guard remaining > 0 else { return nil }
            // A 1+k speculative assignment reserves every target-write slot,
            // but this conservative service path credits only the guaranteed
            // first target token. Mirror finalize-time rollback for the
            // rejected suffix so repeated rounds do not accumulate impossible
            // live KV. The full width remains charged at the step peak because
            // its reserve operation precedes this temporal unreserve.
            let rejectedSpeculativeSuffix = count - remaining
            if rejectedSpeculativeSuffix > 0 {
                guard appendCapacityUnreservation(
                    id: assignedID,
                    start: startComputedTokens + remaining,
                    count: rejectedSpeculativeSuffix)
                else {
                    return nil
                }
            }
            row.computedTokens = knownTokensBeforeStep
            if !row.cancelRequested {
                guard let generated = Self.projectionAdd(row.generatedTokens, 1),
                    let known = Self.projectionAdd(row.knownTokens, 1)
                else {
                    return nil
                }
                row.generatedTokens = generated
                row.knownTokens = known
            }
            rows[assignedID] = row
            return assignedID == id && !row.cancelRequested
        }

        // Place the currently launched step first. Scheduler records already
        // contain its optimistic advances; reconstructing from the confirmed
        // cursor prevents those assignments from being treated as free.
        if !inFlightAssignments.isEmpty {
            var targetSampled = false
            var sampledIDs: [CBv2RequestID] = []
            var projectedAssignments: [CBv2ProjectionAssignment] = []
            projectedAssignments.reserveCapacity(inFlightAssignments.count)
            for assignment in inFlightAssignments {
                guard let row = rows[assignment.id], row.remainingKnownTokens > 0 else {
                    return .unbounded
                }
                guard let projectedAssignment = projectionAssignment(
                    id: assignment.id,
                    count: assignment.numTokens,
                    row: row)
                else {
                    return .unbounded
                }
                projectedAssignments.append(projectedAssignment)
                if assignment.numTokens >= row.remainingKnownTokens {
                    sampledIDs.append(assignment.id)
                }
                guard
                    let sampled = applyExecutedAssignment(
                        id: assignment.id,
                        count: assignment.numTokens,
                        startComputedTokens: row.computedTokens,
                        knownTokensBeforeStep: row.knownTokens)
                else {
                    return .unbounded
                }
                targetSampled = targetSampled || sampled
            }
            guard chargeStep(projectedAssignments) else {
                return .unbounded
            }
            if targetSampled {
                return boundedProjection()
            }
            guard chargeTerminalChainedStepIfNeeded(
                sampledIDs: sampledIDs,
                allowsChainedSuccessor: inFlightAllowsChainedSuccessor)
            else {
                return .unbounded
            }
            removeTerminatedRows()
        } else {
            removeTerminatedRows()
        }

        // A request can carry very large output budgets. Decode-only full-slot
        // stretches are jumped exactly below; this guard bounds adversarial
        // prompt/chunk shapes so projection itself cannot monopolize the
        // engine queue. Failing closed is safer than returning a partial sum.
        let maxProjectionIterations = 32_768
        var iterations = 0

        while rows[id] != nil {
            iterations += 1
            guard iterations <= maxProjectionIterations else {
                return .unbounded
            }

            // Exact acceleration for the common "all slots are decode/paused,
            // target waits" stretch. Every active plain-decode row gets one
            // token per step when the step budget can seat all of them.
            if waitingIDs.contains(id),
                runningIDs.count >= config.maxConcurrentRequests,
                speculationPlanner == nil
            {
                let activeDecodeIDs = runningIDs.filter { runningID in
                    guard let row = rows[runningID], !row.isPaused else { return false }
                    return row.remainingKnownTokens == 1
                }
                let allRunningDecodeOrPaused = runningIDs.allSatisfy { runningID in
                    guard let row = rows[runningID] else { return false }
                    return row.isPaused || row.remainingKnownTokens == 1
                }
                if allRunningDecodeOrPaused,
                    !activeDecodeIDs.isEmpty,
                    activeDecodeIDs.count <= config.maxBatchedTokensPerStep
                {
                    let jump = activeDecodeIDs.compactMap { runningID -> Int? in
                        guard let row = rows[runningID] else { return nil }
                        return row.maxTokens - row.generatedTokens
                    }.min() ?? 0
                    guard jump > 0,
                        let jumpWork = Self.projectionMultiply(activeDecodeIDs.count, jump)
                    else {
                        return .unbounded
                    }
                    for runningID in activeDecodeIDs {
                        guard var row = rows[runningID],
                            appendCapacityReservation(
                                id: runningID,
                                start: row.computedTokens,
                                count: jump),
                            let generated = Self.projectionAdd(row.generatedTokens, jump),
                            let known = Self.projectionAdd(row.knownTokens, jump),
                            let computed = Self.projectionAdd(row.computedTokens, jump)
                        else {
                            return .unbounded
                        }
                        row.generatedTokens = generated
                        row.knownTokens = known
                        row.computedTokens = computed
                        rows[runningID] = row
                    }
                    guard chargeDecodeStretch(tokens: jumpWork, steps: jump) else {
                        return .unbounded
                    }
                    guard
                        chargeTerminalChainedStepIfNeeded(
                            sampledIDs: activeDecodeIDs,
                            allowsChainedSuccessor: true)
                    else {
                        return .unbounded
                    }
                    removeTerminatedRows()
                    continue
                }
            }

            let baseBudget = config.maxBatchedTokensPerStep
            let liveRunningIDs = runningIDs.filter { runningID in
                guard let row = rows[runningID] else { return false }
                return !row.cancelRequested && row.remainingKnownTokens > 0
            }
            let soloStripe: (id: CBv2RequestID, tokens: Int)? = {
                guard let stripe = config.soloPrefillStripeTokens,
                    stripe > config.prefillChunkSize
                else {
                    return nil
                }
                let candidateID: CBv2RequestID?
                if liveRunningIDs.count + waitingIDs.count == 1 {
                    candidateID = liveRunningIDs.first ?? waitingIDs.first
                } else if liveRunningIDs.count <= 1,
                    !liveRunningIDs.contains(where: { runningID in
                        guard let row = rows[runningID] else { return false }
                        return !row.isPaused && row.remainingKnownTokens == 1
                    })
                {
                    candidateID = liveRunningIDs.first ?? waitingIDs.first
                } else {
                    candidateID = nil
                }
                guard let candidateID, let candidate = rows[candidateID],
                    !candidate.isPaused,
                    !candidate.cancelRequested,
                    candidate.remainingKnownTokens > 1
                else {
                    return nil
                }
                return (candidateID, stripe)
            }()

            var budget = max(baseBudget, soloStripe?.tokens ?? 0)
            var totalAssignedTokens = 0
            var prefillTokensAssigned = 0
            var midPrefillAssigned = 0
            var assignments: [CBv2ProjectionAssignment] = []

            let hasDecodeWork = runningIDs.contains { runningID in
                guard let row = rows[runningID] else { return false }
                return !row.isPaused && !row.cancelRequested
                    && row.remainingKnownTokens == 1
            }
            let prefillCap = hasDecodeWork ? mixedStepPrefillTokenCap.map { max(0, $0) } : nil

            func prefillHeadroom() -> Int {
                guard let prefillCap else { return .max }
                return max(0, prefillCap - prefillTokensAssigned)
            }

            func chunkCap(for rowID: CBv2RequestID) -> Int {
                rows[rowID]?.prefixReusePlan?.recurrentChunkSize
                    ?? (soloStripe?.id == rowID ? soloStripe!.tokens : config.prefillChunkSize)
            }

            func prefixClamp(
                row: CBv2ProjectionRow,
                proposed: Int
            ) -> Int {
                row.prefixReusePlan?.clampedChunk(
                    start: row.computedTokens,
                    proposed: proposed) ?? proposed
            }

            // RUNNING first, preserving authoritative scheduler order.
            for runningID in runningIDs where budget > 0 {
                guard var row = rows[runningID],
                    !row.isPaused,
                    !row.cancelRequested
                else {
                    continue
                }
                let remaining = row.remainingKnownTokens
                guard remaining > 0 else { continue }
                let isPrefill = remaining > 1
                if isPrefill, midPrefillAssigned >= 1 {
                    continue
                }

                var count: Int
                if isPrefill {
                    count = min(remaining, chunkCap(for: runningID), budget)
                    if prefillCap != nil {
                        let headroom = prefillHeadroom()
                        guard headroom > 0 else { continue }
                        count = min(count, headroom)
                    }
                    count = prefixClamp(row: row, proposed: count)
                    if count == 0, row.prefixReusePlan?.recurrentChunkSize != nil {
                        // The live scheduler may cold-restart after bounded
                        // geometry waits. A prefix-only bound would underprice
                        // that fallback, so this projection must fail closed.
                        return .unbounded
                    }
                } else {
                    // The live MTP planner mutates controller/round state and
                    // cannot be called from this pure projection. Charge its
                    // configured 1+k ceiling (at most the tested width), with
                    // the scheduler's exact plain-decode fallback when that
                    // width does not fit the remaining step budget.
                    if speculationPlanner != nil {
                        guard let upperBound = speculationDraftTokenUpperBound else {
                            return .unbounded
                        }
                        guard let speculativeWidth = Self.projectionAdd(
                            1, max(0, upperBound))
                        else {
                            return .unbounded
                        }
                        count = speculativeWidth <= budget ? speculativeWidth : 1
                    } else {
                        count = 1
                    }
                }
                guard count > 0 else { continue }

                guard let assignment = projectionAssignment(
                    id: runningID,
                    count: count,
                    row: row)
                else {
                    return .unbounded
                }
                assignments.append(assignment)
                guard let computed = Self.projectionAdd(row.computedTokens, count),
                    let assigned = Self.projectionAdd(totalAssignedTokens, count)
                else {
                    return .unbounded
                }
                row.computedTokens = computed
                rows[runningID] = row
                budget -= count
                totalAssignedTokens = assigned
                if isPrefill {
                    guard let prefillAssigned = Self.projectionAdd(
                        prefillTokensAssigned, count)
                    else {
                        return .unbounded
                    }
                    prefillTokensAssigned = prefillAssigned
                    if row.knownTokens - row.computedTokens > 1 {
                        midPrefillAssigned += 1
                    }
                }
            }

            // WAITING admission. Cancel-pending rows are absent from this
            // projection because the engine processes cancellation before its
            // next plan. Paused waiters are skipped but retain queue position.
            var waitingIndex = 0
            while budget > 0,
                runningIDs.count < config.maxConcurrentRequests,
                waitingIndex < waitingIDs.count
            {
                let waitingID = waitingIDs[waitingIndex]
                guard var row = rows[waitingID] else {
                    waitingIDs.remove(at: waitingIndex)
                    continue
                }
                if row.isPaused {
                    waitingIndex += 1
                    continue
                }
                guard midPrefillAssigned < 1 else { break }
                let admissionHeadroom = prefillHeadroom()
                guard admissionHeadroom > 0 else { break }

                let normalHeadroom =
                    soloStripe?.id == waitingID
                    ? budget
                    : max(0, baseBudget - totalAssignedTokens)
                var count = min(
                    row.remainingKnownTokens,
                    chunkCap(for: waitingID),
                    budget,
                    normalHeadroom,
                    admissionHeadroom)
                count = prefixClamp(row: row, proposed: count)
                if count == 0, row.prefixReusePlan?.recurrentChunkSize != nil {
                    return .unbounded
                }
                guard count > 0 else { break }

                guard let assignment = projectionAssignment(
                    id: waitingID,
                    count: count,
                    row: row)
                else {
                    return .unbounded
                }
                assignments.append(assignment)
                guard let computed = Self.projectionAdd(row.computedTokens, count),
                    let assigned = Self.projectionAdd(totalAssignedTokens, count),
                    let prefillAssigned = Self.projectionAdd(prefillTokensAssigned, count)
                else {
                    return .unbounded
                }
                row.computedTokens = computed
                rows[waitingID] = row
                budget -= count
                totalAssignedTokens = assigned
                prefillTokensAssigned = prefillAssigned
                waitingIDs.remove(at: waitingIndex)
                runningIDs.append(waitingID)
                if row.knownTokens - row.computedTokens > 1 {
                    midPrefillAssigned += 1
                }
            }

            guard !assignments.isEmpty else {
                return .unbounded
            }
            for assignment in assignments {
                guard appendCapacityReservation(
                    id: assignment.id,
                    start: assignment.startComputedTokens,
                    count: assignment.count)
                else {
                    return .unbounded
                }
            }
            guard chargeStep(assignments) else {
                return .unbounded
            }

            // Finalize the whole projected step before testing the target.
            // Later queue rows admitted beside the target's final chunk are
            // already included in `totalAssignedTokens` above.
            var targetSampled = false
            var sampledIDs: [CBv2RequestID] = []
            for assignment in assignments {
                if assignment.count
                    >= assignment.knownTokensBeforeStep - assignment.startComputedTokens
                {
                    sampledIDs.append(assignment.id)
                }
                guard
                    let sampled = applyExecutedAssignment(
                        id: assignment.id,
                        count: assignment.count,
                        startComputedTokens: assignment.startComputedTokens,
                        knownTokensBeforeStep: assignment.knownTokensBeforeStep)
                else {
                    return .unbounded
                }
                targetSampled = targetSampled || sampled
            }
            if targetSampled {
                return boundedProjection()
            }
            // Future MTP decisions are intentionally not invoked from this
            // pure projection. A projected terminal step may therefore be a
            // plain fallback that chains; charging one successor is safe.
            // When live MTP suppresses it, false rejection is bounded by one
            // decode batch at the slot-release boundary.
            guard chargeTerminalChainedStepIfNeeded(
                sampledIDs: sampledIDs,
                allowsChainedSuccessor: true)
            else {
                return .unbounded
            }
            removeTerminatedRows()
        }

        return .unbounded
    }

    private static func projectionAdd(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : value
    }

    private static func projectionMultiply(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : value
    }
}
