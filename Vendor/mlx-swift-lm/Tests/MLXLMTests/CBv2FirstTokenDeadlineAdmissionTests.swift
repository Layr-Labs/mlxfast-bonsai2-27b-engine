// Copyright © 2026 Eigen Labs.
//
// Deterministic first-token deadline admission tests. Pure scheduler cases
// pin ordering/accounting; engine cases pin the serial-queue atomicity seam.

import Foundation
import XCTest

@testable import MLXLMCommon

private final class CBv2DeadlineAdmissionHookGate: @unchecked Sendable {
    private let entered = DispatchSemaphore(value: 0)
    private let release = DispatchSemaphore(value: 0)

    func hook() {
        entered.signal()
        release.wait()
    }

    func waitUntilEntered(timeout: TimeInterval = 2) -> Bool {
        entered.wait(timeout: .now() + timeout) == .success
    }

    func unblock() {
        release.signal()
    }
}

final class CBv2FirstTokenWorkProjectionTests: XCTestCase {
    private func makeScheduler(
        maxConcurrent: Int = 4,
        budget: Int = 8,
        chunk: Int = 4
    ) -> SchedulerV2 {
        SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent,
                maxBatchedTokensPerStep: budget,
                prefillChunkSize: chunk,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 64))
    }

    private func assertBounded(
        _ projection: CBv2FirstTokenWorkProjection,
        tokens: Int,
        steps: Int,
        prefillTokens: Int? = nil,
        decodeTokens: Int = 0,
        mixedSteps: Int = 0,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard case .bounded(let work, _) = projection else {
            XCTFail("expected bounded projection, got \(projection)", file: file, line: line)
            return
        }
        XCTAssertEqual(work.scheduledTokens, tokens, file: file, line: line)
        XCTAssertEqual(work.prefillTokens, prefillTokens ?? tokens - decodeTokens, file: file, line: line)
        XCTAssertEqual(work.decodeTokens, decodeTokens, file: file, line: line)
        XCTAssertEqual(work.scheduledSteps, steps, file: file, line: line)
        XCTAssertEqual(work.mixedSteps, mixedSteps, file: file, line: line)
    }

    func testT0AndTPlusTwoChargeFullInFlightAssignmentWithoutElapsedCredit() throws {
        let scheduler = makeScheduler()
        let t0 = Date(timeIntervalSince1970: 1_000)
        let tPlusTwo = t0.addingTimeInterval(2)
        let first = CBv2Request(
            id: CBv2RequestID(10_001),
            promptTokens: Array(0 ..< 8),
            maxTokens: 1)
        let second = CBv2Request(
            id: CBv2RequestID(10_002),
            promptTokens: Array(20 ..< 24),
            maxTokens: 1)

        let firstRecord = try scheduler.enqueue(first, now: t0)
        XCTAssertEqual(firstRecord.submittedAt, t0)
        assertBounded(
            scheduler.firstTokenWorkProjection(for: first.id),
            tokens: 8,
            steps: 2)

        let inFlight = scheduler.plan()
        XCTAssertEqual(inFlight.assignments.map(\.numTokens), [4])
        let secondRecord = try scheduler.enqueue(second, now: tPlusTwo)
        XCTAssertEqual(secondRecord.submittedAt, tPlusTwo)

        // The first 4-token assignment is optimistic and still in flight. It
        // is charged in full, then A's final 4 and B's 4 share the next step.
        // No Date/submission-age credit is applied at T+2.
        assertBounded(
            scheduler.firstTokenWorkProjection(
                for: second.id,
                inFlightAssignments: inFlight.assignments),
            tokens: 12,
            steps: 2)
    }

    func testPrefixReplayStartIsTheConfirmedCursor() throws {
        let scheduler = makeScheduler(budget: 4, chunk: 4)
        let request = CBv2Request(
            id: CBv2RequestID(10_010),
            promptTokens: Array(0 ..< 12),
            maxTokens: 1)
        let record = try scheduler.enqueue(request)
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: CBv2SchedFixtures.tinyLayerKinds(),
            backend: .contiguousUnquantized)
        let plan = try XCTUnwrap(capability.plan(matchedBoundary: 8))
        XCTAssertEqual(plan.replayStart, 8)
        record.prefixReusePlan = plan
        record.numComputedTokens = plan.replayStart

        assertBounded(
            scheduler.firstTokenWorkProjection(for: request.id),
            tokens: 4,
            steps: 1)
    }

    func testMixedDecodeAndTrailingFinalChunkChargeWholeTargetStep() throws {
        let scheduler = makeScheduler(budget: 6, chunk: 4)
        let decoder = CBv2Request(
            id: CBv2RequestID(10_020),
            promptTokens: [1],
            maxTokens: 4)
        try scheduler.enqueue(decoder)
        _ = CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())

        let target = CBv2Request(
            id: CBv2RequestID(10_021),
            promptTokens: [2, 3, 4, 5],
            maxTokens: 1,
            priority: 1)
        let trailing = CBv2Request(
            id: CBv2RequestID(10_022),
            promptTokens: [6],
            maxTokens: 1)
        try scheduler.enqueue(target)
        try scheduler.enqueue(trailing)

        // One mixed decode + target's final 4-token prompt + the lower-
        // priority one-token waiter all ride one asyncEval. Target TTFT waits
        // for the full six-token step.
        assertBounded(
            scheduler.firstTokenWorkProjection(for: target.id),
            tokens: 6,
            steps: 1,
            prefillTokens: 5,
            decodeTokens: 1,
            mixedSteps: 1)
    }

    func testPreemptedReplayCancelledRowAndFinalChunkSlotRelease() throws {
        let scheduler = makeScheduler(maxConcurrent: 3, budget: 3, chunk: 2)
        let preempted = CBv2Request(
            id: CBv2RequestID(10_030),
            promptTokens: [1, 2],
            maxTokens: 3)
        try scheduler.enqueue(preempted)
        _ = CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())
        XCTAssertTrue(scheduler.requeueOnCapacity(preempted.id))
        XCTAssertEqual(scheduler.record(for: preempted.id)?.status, .preempted)
        XCTAssertEqual(scheduler.record(for: preempted.id)?.generatedTokenCount, 1)

        let cancelled = CBv2Request(
            id: CBv2RequestID(10_031),
            promptTokens: Array(0 ..< 5),
            maxTokens: 1)
        let target = CBv2Request(
            id: CBv2RequestID(10_032),
            promptTokens: [9],
            maxTokens: 1,
            priority: -1)
        try scheduler.enqueue(cancelled)
        scheduler.requestCancel(cancelled.id)
        try scheduler.enqueue(target)

        // The preempted row must replay prompt + confirmed output from zero.
        // Its first 2-token chunk leaves one known token, which is decode-
        // ready and therefore releases the partial-prefill slot in the SAME
        // plan; the cancelled row is removed and target consumes token 3.
        assertBounded(
            scheduler.firstTokenWorkProjection(for: target.id),
            tokens: 3,
            steps: 1)
    }

    func testFullSlotTerminalSampleChargesOneStepLateChain() throws {
        let scheduler = makeScheduler(maxConcurrent: 1, budget: 1, chunk: 1)
        let blocker = CBv2Request(
            id: CBv2RequestID(10_035),
            promptTokens: [1],
            maxTokens: 1)
        try scheduler.enqueue(blocker)
        let inFlight = scheduler.plan()
        XCTAssertEqual(inFlight.assignments.map(\.numTokens), [1])

        let target = CBv2Request(
            id: CBv2RequestID(10_036),
            promptTokens: [2],
            maxTokens: 1)
        try scheduler.enqueue(target)

        // The full running slot lets the engine launch one chained decode
        // before blocker finalization observes its length limit. That
        // one-step-late work runs without target, then target gets its turn.
        assertBounded(
            scheduler.firstTokenWorkProjection(
                for: target.id,
                inFlightAssignments: inFlight.assignments),
            tokens: 3,
            steps: 3,
            prefillTokens: 2,
            decodeTokens: 1)
    }

    func testFullSlotDecodeStretchChargesTerminalChain() throws {
        let scheduler = makeScheduler(maxConcurrent: 1, budget: 1, chunk: 1)
        let blocker = CBv2Request(
            id: CBv2RequestID(10_037),
            promptTokens: [1],
            maxTokens: 2)
        try scheduler.enqueue(blocker)
        _ = CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())

        let target = CBv2Request(
            id: CBv2RequestID(10_038),
            promptTokens: [2],
            maxTokens: 1)
        try scheduler.enqueue(target)

        // The accelerated decode stretch ends at blocker's length limit, but
        // the engine has already launched one terminal chained decode before
        // that limit becomes host-visible.
        assertBounded(
            scheduler.firstTokenWorkProjection(for: target.id),
            tokens: 3,
            steps: 3,
            prefillTokens: 1,
            decodeTokens: 2)
    }

    func testPausedRunningSlotFailsClosedWhenTargetCannotEnter() throws {
        let scheduler = makeScheduler(maxConcurrent: 1, budget: 2, chunk: 2)
        let blocker = CBv2Request(
            id: CBv2RequestID(10_040),
            promptTokens: [1, 2, 3, 4],
            maxTokens: 1)
        try scheduler.enqueue(blocker)
        _ = scheduler.plan()
        scheduler.pause(blocker.id)

        let target = CBv2Request(
            id: CBv2RequestID(10_041),
            promptTokens: [5],
            maxTokens: 1)
        try scheduler.enqueue(target)
        XCTAssertEqual(
            scheduler.firstTokenWorkProjection(for: target.id),
            .unbounded)
    }

    func testSpeculativeProjectionChargesConfiguredMTPWidthNotStepBudget() throws {
        let scheduler = makeScheduler(maxConcurrent: 2, budget: 64, chunk: 4)
        scheduler.speculationPlanner = { _ in 7 }
        scheduler.speculationDraftTokenUpperBound = 7
        let decoder = CBv2Request(
            id: CBv2RequestID(10_050),
            promptTokens: [1],
            maxTokens: 20)
        try scheduler.enqueue(decoder)
        _ = CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())

        let target = CBv2Request(
            id: CBv2RequestID(10_051),
            promptTokens: [2, 3, 4, 5],
            maxTokens: 1)
        try scheduler.enqueue(target)

        assertBounded(
            scheduler.firstTokenWorkProjection(for: target.id),
            tokens: 12,
            steps: 1,
            prefillTokens: 4,
            decodeTokens: 8,
            mixedSteps: 1)
    }

    func testMTPInFlightTerminalDoesNotInventChainedDecode() throws {
        let scheduler = makeScheduler(maxConcurrent: 1, budget: 8, chunk: 1)
        scheduler.speculationPlanner = { _ in 7 }
        scheduler.speculationDraftTokenUpperBound = 7
        let blocker = CBv2Request(
            id: CBv2RequestID(10_052),
            promptTokens: [1],
            maxTokens: 1)
        try scheduler.enqueue(blocker)
        let inFlight = scheduler.plan()

        let target = CBv2Request(
            id: CBv2RequestID(10_053),
            promptTokens: [2],
            maxTokens: 1)
        try scheduler.enqueue(target)

        assertBounded(
            scheduler.firstTokenWorkProjection(
                for: target.id,
                inFlightAssignments: inFlight.assignments,
                inFlightAllowsChainedSuccessor: false),
            tokens: 2,
            steps: 2)
    }

    func testFixedZeroMTPPlainFallbackChargesTerminalChain() throws {
        let scheduler = makeScheduler(maxConcurrent: 1, budget: 1, chunk: 1)
        scheduler.speculationPlanner = { _ in 0 }
        scheduler.speculationDraftTokenUpperBound = 0
        let blocker = CBv2Request(
            id: CBv2RequestID(10_054),
            promptTokens: [1],
            maxTokens: 1)
        try scheduler.enqueue(blocker)
        let inFlight = scheduler.plan()
        XCTAssertEqual(inFlight.assignments.map(\.numTokens), [1])

        let target = CBv2Request(
            id: CBv2RequestID(10_055),
            promptTokens: [2],
            maxTokens: 1)
        try scheduler.enqueue(target)

        // Fixed k=0 uses the ordinary plain-decode path. Its prompt sample is
        // therefore a valid chain base even though an MTP planner is installed.
        assertBounded(
            scheduler.firstTokenWorkProjection(
                for: target.id,
                inFlightAssignments: inFlight.assignments,
                inFlightAllowsChainedSuccessor: true),
            tokens: 3,
            steps: 3,
            prefillTokens: 2,
            decodeTokens: 1)
    }

    func testSpeculativeCapacityRollbackPreventsImpossibleAccumulation() throws {
        let kind = CBv2LayerKind(
            attention: .full,
            headDim: 1,
            kvHeads: 1,
            queryHeads: 1)
        // One token occupies four bytes (K + V, fp16). Four live token
        // positions fit; five do not.
        let admission = AdmissionV2(
            layerKinds: [kind],
            bytesCapacity: 16,
            config: .init(watermarkFraction: 0))
        let scheduler = SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 2,
                prefillChunkSize: 1,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 4),
            capacity: admission)
        scheduler.speculationPlanner = { _ in 1 }
        scheduler.speculationDraftTokenUpperBound = 1

        let blocker = CBv2Request(
            id: CBv2RequestID(10_056),
            promptTokens: [1],
            maxTokens: 3)
        try scheduler.enqueue(blocker)
        _ = CBv2SchedSim.confirm(scheduler, plan: scheduler.plan())

        let target = CBv2Request(
            id: CBv2RequestID(10_057),
            promptTokens: [2],
            maxTokens: 1)
        try scheduler.enqueue(target)

        guard case .bounded(_, let operations) =
            scheduler.firstTokenWorkProjection(for: target.id)
        else {
            return XCTFail("speculative projection should remain bounded")
        }
        XCTAssertTrue(
            operations.contains {
                if case .unreserve = $0 { return true }
                return false
            })
        XCTAssertTrue(admission.canGuarantee(projectedOperations: operations))

        // Omitting finalize-time speculative rollback falsely carries both
        // rejected suffixes into round two and asks this four-token ledger to
        // hold five positions.
        let withoutRollback = operations.filter {
            if case .unreserve = $0 { return false }
            return true
        }
        XCTAssertFalse(
            admission.canGuarantee(projectedOperations: withoutRollback))
    }

    func testFullLedgerSkipsOptionalTerminalChainBeforeSlotRelease() throws {
        let kind = CBv2LayerKind(
            attention: .full,
            headDim: 1,
            kvHeads: 1,
            queryHeads: 1)
        let admission = AdmissionV2(
            layerKinds: [kind],
            bytesCapacity: 4,
            config: .init(watermarkFraction: 0))
        let scheduler = SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 1,
                prefillChunkSize: 1,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 2),
            capacity: admission)
        let blocker = CBv2Request(
            id: CBv2RequestID(10_058),
            promptTokens: [1],
            maxTokens: 1)
        try scheduler.enqueue(blocker)
        let inFlight = scheduler.plan()

        let target = CBv2Request(
            id: CBv2RequestID(10_059),
            promptTokens: [2],
            maxTokens: 1)
        try scheduler.enqueue(target)

        guard case .bounded(let work, let operations) =
            scheduler.firstTokenWorkProjection(
                for: target.id,
                inFlightAssignments: inFlight.assignments)
        else {
            return XCTFail("full ledger should skip, not require, the chain")
        }
        XCTAssertEqual(work.scheduledTokens, 3)
        XCTAssertTrue(
            operations.contains {
                if case .reserveIfAvailable = $0 { return true }
                return false
            })
        XCTAssertTrue(
            admission.canGuarantee(projectedOperations: operations),
            "runtime skips the chain, releases the blocker, then seats target")
    }
}

final class CBv2FirstTokenDeadlineEngineTests: XCTestCase {
    private func assertBounded(
        _ work: CBv2FirstTokenProjectedWork,
        tokens: Int,
        steps: Int,
        duration: Duration,
        prefillTokens: Int? = nil,
        decodeTokens: Int = 0,
        mixedSteps: Int = 0,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard
            case .bounded(
                let scheduledWork,
                let actualDuration) = work
        else {
            XCTFail("expected bounded work, got \(work)", file: file, line: line)
            return
        }
        XCTAssertEqual(scheduledWork.scheduledTokens, tokens, file: file, line: line)
        XCTAssertEqual(
            scheduledWork.prefillTokens,
            prefillTokens ?? tokens - decodeTokens,
            file: file,
            line: line)
        XCTAssertEqual(scheduledWork.decodeTokens, decodeTokens, file: file, line: line)
        XCTAssertEqual(scheduledWork.scheduledSteps, steps, file: file, line: line)
        XCTAssertEqual(scheduledWork.mixedSteps, mixedSteps, file: file, line: line)
        XCTAssertEqual(actualDuration, duration, file: file, line: line)
    }

    private static func deadlineAdmission(
        after duration: Duration,
        prefillRate: Double = 1,
        decodeRate: Double? = 1
    ) -> CBv2FirstTokenDeadlineAdmission {
        CBv2FirstTokenDeadlineAdmission(
            deadline: ContinuousClock.now.advanced(by: duration),
            conservativePrefillTokensPerSecond: prefillRate,
            conservativeDecodeTokensPerSecond: decodeRate)
    }

    func testForcedTerminalDoesNotAcknowledgeEngineOwnership() async {
        let stream = CBv2OutputStream(id: CBv2RequestID(11_000))
        stream.finish(
            reason: .terminal(cause: .watchdog, message: "test"),
            usage: CBv2Usage(promptTokens: 1, completionTokens: 0))

        XCTAssertTrue(stream.isFinished)
        XCTAssertFalse(stream.isEngineOwnershipReleased)

        stream.releaseEngineOwnership()
        await stream.waitUntilEngineOwnershipReleased()
        XCTAssertTrue(stream.isEngineOwnershipReleased)
    }

    func testUnreachableRequestIsRemovedBeforeAnyForward() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4,
                maxBatchedTokensPerStep: 4,
                prefillChunkSize: 4,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 8))
        let request = CBv2Request(
            id: CBv2RequestID(11_001),
            promptTokens: Array(0 ..< 8),
            maxTokens: 1)

        let result = try await harness.engine.submit(
            request,
            firstTokenDeadline: Self.deadlineAdmission(after: .seconds(7)))
        guard case .deadlineUnreachable(let work) = result else {
            XCTFail("request should be rejected")
            return
        }
        assertBounded(work, tokens: 8, steps: 2, duration: .seconds(8))
        harness.engine.loopForTesting.onEngineQueueSync {
            XCTAssertNil(harness.engine.loopForTesting.scheduler.record(for: request.id))
        }
        XCTAssertTrue(harness.model.forwardShapes.isEmpty)
        XCTAssertEqual(harness.backend.liveStates, 0)
        XCTAssertEqual(harness.engine.capacity().activeRequests, 0)
        XCTAssertEqual(harness.engine.capacity().waitingRequests, 0)
        await harness.engine.shutdown()
    }

    func testUnrepresentableServiceDurationsFailClosedWithoutTrapping() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 1,
                prefillChunkSize: 1,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 1))
        let request = CBv2Request(
            id: CBv2RequestID(11_005),
            promptTokens: [1],
            maxTokens: 1)

        let result = try await harness.engine.submit(
            request,
            firstTokenDeadline: Self.deadlineAdmission(
                after: .seconds(1),
                prefillRate: 1e-19))
        guard case .deadlineUnreachable(let work) = result else {
            XCTFail("unrepresentable service duration must fail closed")
            return
        }
        XCTAssertEqual(work, .unbounded)

        let subAttosecond = CBv2Request(
            id: CBv2RequestID(11_006),
            promptTokens: [1],
            maxTokens: 1)
        let subAttosecondResult = try await harness.engine.submit(
            subAttosecond,
            firstTokenDeadline: Self.deadlineAdmission(
                after: .zero,
                prefillRate: .greatestFiniteMagnitude))
        guard case .deadlineUnreachable(let subAttosecondWork) = subAttosecondResult else {
            XCTFail("positive work rounded to zero must fail closed")
            return
        }
        XCTAssertEqual(subAttosecondWork, .unbounded)
        XCTAssertTrue(harness.model.forwardShapes.isEmpty)
        await harness.engine.shutdown()
    }

    func testTaskCancellationResumesQueuedAdmissionAndReleasesGauge() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2,
                maxBatchedTokensPerStep: 4,
                prefillChunkSize: 4,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 2))
        harness.model.forwardDelay = 0.25
        let blocker = CBv2Request(
            id: CBv2RequestID(11_007),
            promptTokens: Array(0 ..< 8),
            maxTokens: 1)
        let blockerStream = try harness.engine.submit(blocker)
        let blockerLaunched = await cbv2SchedWait {
            harness.model.forwardShapes.count == 1
        }
        XCTAssertTrue(blockerLaunched)
        let waitingBeforeSubmission = harness.engine.capacity().waitingRequests

        let queued = CBv2Request(
            id: CBv2RequestID(11_008),
            promptTokens: [20, 21, 22, 23],
            maxTokens: 1)
        let queuedAdmission = Self.deadlineAdmission(after: .seconds(60))
        let submission = Task {
            try await harness.engine.submit(
                queued,
                firstTokenDeadline: queuedAdmission)
        }
        let submissionQueued = await cbv2SchedWait {
            harness.engine.capacity().waitingRequests == waitingBeforeSubmission + 1
        }
        XCTAssertTrue(submissionQueued)

        submission.cancel()
        do {
            _ = try await submission.value
            XCTFail("cancelled admission must throw CancellationError")
        } catch is CancellationError {
            // Expected.
        }
        XCTAssertLessThanOrEqual(
            harness.engine.capacity().waitingRequests,
            waitingBeforeSubmission)
        let immediateRetry = try harness.engine.submit(queued)

        _ = await cbv2SchedCollect(blockerStream)
        let retry = await cbv2SchedCollect(immediateRetry)
        XCTAssertEqual(retry.finishReason, .length)
        harness.engine.loopForTesting.onEngineQueueSync {
            XCTAssertNil(harness.engine.loopForTesting.scheduler.record(for: queued.id))
        }
        await harness.engine.shutdown()
    }

    func testDeadlineBudgetKeepsRunningWhileAdmissionWaitsForEngineQueue() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2,
                maxBatchedTokensPerStep: 4,
                prefillChunkSize: 4,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 2))
        harness.model.forwardDelay = 0.35
        let blocker = CBv2Request(
            id: CBv2RequestID(11_030),
            promptTokens: [1, 2, 3, 4],
            maxTokens: 1)
        let blockerStream = try harness.engine.submit(blocker)
        let blockerLaunched = await cbv2SchedWait {
            harness.model.forwardShapes.count == 1
        }
        XCTAssertTrue(blockerLaunched)

        let target = CBv2Request(
            id: CBv2RequestID(11_031),
            promptTokens: [5],
            maxTokens: 1)
        let result = try await harness.engine.submit(
            target,
            firstTokenDeadline: Self.deadlineAdmission(
                after: .milliseconds(100),
                prefillRate: 1_000,
                decodeRate: 1_000))
        guard case .deadlineUnreachable = result else {
            XCTFail("engine-queue wait must consume the absolute deadline")
            return
        }

        _ = await cbv2SchedCollect(blockerStream)
        harness.engine.loopForTesting.onEngineQueueSync {
            XCTAssertNil(harness.engine.loopForTesting.scheduler.record(for: target.id))
        }
        await harness.engine.shutdown()
    }

    func testCancellationAfterInitialGuardRetainsOwnershipUntilQueueAcknowledges()
        async throws
    {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 1,
                prefillChunkSize: 1,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 2))
        let request = CBv2Request(
            id: CBv2RequestID(11_032),
            promptTokens: [1],
            maxTokens: 1)
        let gate = CBv2DeadlineAdmissionHookGate()
        harness.engine.loopForTesting
            .setDeadlineAdmissionInitialGuardHookForTesting { _ in
                gate.hook()
            }
        let cancellationAdmission = Self.deadlineAdmission(after: .seconds(60))
        let submission = Task {
            try await harness.engine.submit(
                request,
                firstTokenDeadline: cancellationAdmission)
        }
        let entered = await Task.detached {
            gate.waitUntilEntered()
        }.value
        XCTAssertTrue(entered)

        submission.cancel()
        XCTAssertThrowsError(try harness.engine.submit(request)) { error in
            guard let schedulerError = error as? CBv2SchedulerError,
                case .duplicateRequestID(let duplicateID) = schedulerError
            else {
                return XCTFail("unexpected pre-acknowledgement error: \(error)")
            }
            XCTAssertEqual(duplicateID, request.id)
        }
        gate.unblock()
        do {
            _ = try await submission.value
            XCTFail("cancelled deadline admission must throw CancellationError")
        } catch is CancellationError {
            // Expected after the engine queue acknowledges cancellation.
        }

        let retry = try harness.engine.submit(request)
        let output = await cbv2SchedCollect(retry)
        XCTAssertEqual(output.finishReason, .length)
        harness.engine.loopForTesting.onEngineQueueSync {
            XCTAssertNil(harness.engine.loopForTesting.scheduler.record(for: request.id))
        }
        XCTAssertEqual(harness.engine.capacity().activeRequests, 0)
        XCTAssertEqual(harness.engine.capacity().waitingRequests, 0)
        await harness.engine.shutdown()
    }

    func testCancellationAfterCommitTransfersEngineRowRetirement() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 1,
                prefillChunkSize: 1,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 2))
        let request = CBv2Request(
            id: CBv2RequestID(11_036),
            promptTokens: [1],
            maxTokens: 1)
        let gate = CBv2DeadlineAdmissionHookGate()
        harness.engine.loopForTesting
            .setDeadlineAdmissionCommittedHookForTesting { _ in
                gate.hook()
            }
        let admission = Self.deadlineAdmission(after: .seconds(60))
        let submission = Task {
            try await harness.engine.submit(
                request,
                firstTokenDeadline: admission)
        }
        let entered = await Task.detached {
            gate.waitUntilEntered()
        }.value
        XCTAssertTrue(entered)

        submission.cancel()
        gate.unblock()
        do {
            _ = try await submission.value
            XCTFail("post-commit cancellation must transfer retirement ownership")
        } catch let cancellation as CBv2FirstTokenAdmissionCancellation {
            await cancellation.retirement.wait()
        } catch {
            XCTFail("unexpected post-commit cancellation error: \(error)")
        }
        XCTAssertNil(harness.engine.loopForTesting.stream(for: request.id))
        harness.engine.loopForTesting.onEngineQueueSync {
            XCTAssertNil(harness.engine.loopForTesting.scheduler.record(for: request.id))
        }
        XCTAssertEqual(harness.engine.capacity().activeRequests, 0)
        XCTAssertEqual(harness.engine.capacity().waitingRequests, 0)

        let retry = try harness.engine.submit(request)
        let retryOutput = await cbv2SchedCollect(retry)
        XCTAssertEqual(retryOutput.finishReason, .length)
        await harness.engine.shutdown()
    }

    func testMixedProjectionUsesSeparateRatesAndMissingDecodeRateFailsClosed()
        async throws
    {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2,
                maxBatchedTokensPerStep: 5,
                prefillChunkSize: 4,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 2))
        let decoder = CBv2Request(
            id: CBv2RequestID(11_033),
            promptTokens: [1],
            maxTokens: 4)
        try harness.engine.loopForTesting.onEngineQueueSync {
            // Freeze immediately after the loop launches exactly one decode
            // step. Admissions still execute on the engine queue, but neither
            // can race finalization or a successor step under suite load.
            harness.engine.loopForTesting.suspendStepExecutionAtCountForTesting = 1
            try harness.engine.loopForTesting.scheduler.enqueue(decoder)
            _ = CBv2SchedSim.confirm(
                harness.engine.loopForTesting.scheduler,
                plan: harness.engine.loopForTesting.scheduler.plan())
        }
        let decodeLaunched = await cbv2SchedWait {
            harness.engine.loopForTesting.onEngineQueueSync {
                harness.engine.loopForTesting.stepCount == 1
            }
        }
        XCTAssertTrue(decodeLaunched)
        harness.engine.loopForTesting.onEngineQueueSync {
            let record = harness.engine.loopForTesting.scheduler.record(for: decoder.id)
            XCTAssertEqual(record?.generatedTokenCount, 1)
            XCTAssertEqual(record?.numComputedTokens, 2)
            XCTAssertEqual(record?.pendingSamples, 1)
        }

        let noDecodeRate = CBv2Request(
            id: CBv2RequestID(11_034),
            promptTokens: [2, 3, 4, 5],
            maxTokens: 1)
        let unbounded = try await harness.engine.submit(
            noDecodeRate,
            firstTokenDeadline: Self.deadlineAdmission(
                after: .seconds(60),
                prefillRate: 4,
                decodeRate: nil))
        if case .deadlineUnreachable(let unboundedWork) = unbounded {
            XCTAssertEqual(unboundedWork, .unbounded)
        } else {
            XCTFail("mixed work without a decode lower bound must fail closed")
        }
        harness.engine.loopForTesting.onEngineQueueSync {
            XCTAssertEqual(harness.engine.loopForTesting.stepCount, 1)
            let record = harness.engine.loopForTesting.scheduler.record(for: decoder.id)
            XCTAssertEqual(record?.generatedTokenCount, 1)
            XCTAssertEqual(record?.numComputedTokens, 2)
            XCTAssertEqual(record?.pendingSamples, 1)
        }

        let priced = CBv2Request(
            id: CBv2RequestID(11_035),
            promptTokens: [6, 7, 8, 9],
            maxTokens: 1)
        let result = try await harness.engine.submit(
            priced,
            firstTokenDeadline: Self.deadlineAdmission(
                after: .milliseconds(2_500),
                prefillRate: 4,
                decodeRate: 1))
        if case .deadlineUnreachable(let work) = result {
            assertBounded(
                work,
                tokens: 6,
                steps: 2,
                duration: .seconds(3),
                prefillTokens: 4,
                decodeTokens: 2,
                mixedSteps: 1)
        } else {
            XCTFail("phase-priced mixed work must exceed the 2.5-second budget")
        }

        harness.engine.loopForTesting.onEngineQueueSync {
            harness.engine.loopForTesting.finishRequest(decoder.id, reason: .cancelled)
            harness.engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
        let decoderRetired = await cbv2SchedWait {
            harness.backend.liveStates == 0
        }
        XCTAssertTrue(decoderRetired)
        await harness.engine.shutdown()
    }

    func testWatchdogResumesAdmissionBehindWedgedEngineQueue() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2,
                maxBatchedTokensPerStep: 4,
                prefillChunkSize: 4,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 2),
            loopConfig: CBv2EngineLoopConfig(
                stepTimeout: 0.2,
                watchdogInterval: 0.01))
        harness.model.forwardDelay = 1
        let blocker = CBv2Request(
            id: CBv2RequestID(11_009),
            promptTokens: [1],
            maxTokens: 1)
        let blockerStream = try harness.engine.submit(blocker)
        let blockerLaunched = await cbv2SchedWait {
            harness.model.forwardShapes.count == 1
        }
        XCTAssertTrue(blockerLaunched)
        let waitingBeforeSubmission = harness.engine.capacity().waitingRequests

        let queued = CBv2Request(
            id: CBv2RequestID(11_011),
            promptTokens: [2],
            maxTokens: 1)
        let queuedAdmission = Self.deadlineAdmission(after: .seconds(60))
        let submission = Task {
            try await harness.engine.submit(
                queued,
                firstTokenDeadline: queuedAdmission)
        }
        let submissionQueued = await cbv2SchedWait {
            harness.engine.capacity().waitingRequests == waitingBeforeSubmission + 1
        }
        XCTAssertTrue(submissionQueued)

        do {
            _ = try await submission.value
            XCTFail("watchdog must fail an admission blocked behind the wedged queue")
        } catch let error as CBv2KVError {
            guard case .capacityExhausted = error else {
                return XCTFail("unexpected watchdog admission error: \(error)")
            }
        }
        XCTAssertEqual(
            harness.engine.capacity().waitingRequests,
            waitingBeforeSubmission)

        let postWedge = CBv2Request(
            id: CBv2RequestID(11_012),
            promptTokens: [3],
            maxTokens: 1)
        let postWedgeStarted = Date()
        do {
            _ = try await harness.engine.submit(
                postWedge,
                firstTokenDeadline: Self.deadlineAdmission(after: .seconds(60)))
            XCTFail("admission registered after a watchdog report must fail")
        } catch let error as CBv2KVError {
            guard case .capacityExhausted = error else {
                return XCTFail("unexpected post-watchdog error: \(error)")
            }
        }
        XCTAssertLessThan(
            Date().timeIntervalSince(postWedgeStarted),
            0.5,
            "post-watchdog admission must not wait for the wedged queue")
        XCTAssertNil(
            harness.engine.loopForTesting.stream(for: postWedge.id),
            "unhealthy fast rejection must not retain a stream")

        _ = await cbv2SchedCollect(blockerStream)
        let cleaned = await cbv2SchedWait {
            harness.engine.loopForTesting.onEngineQueueSync {
                harness.engine.loopForTesting.scheduler.record(for: queued.id) == nil
                    && harness.engine.loopForTesting.scheduler.record(for: postWedge.id) == nil
            }
        }
        XCTAssertTrue(cleaned)
        await harness.engine.shutdown()
    }

    func testPrefixMetadataPrecedesVerdictAndReducesProjectedWork() async throws {
        let cache = CBv2SchedScriptedPrefixCache(matched: 8)
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4,
                maxBatchedTokensPerStep: 4,
                prefillChunkSize: 4,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 8,
                enablePrefixCache: true),
            prefixCache: cache)
        let request = CBv2Request(
            id: CBv2RequestID(11_010),
            promptTokens: Array(0 ..< 12),
            maxTokens: 1)

        let result = try await harness.engine.submit(
            request,
            firstTokenDeadline: Self.deadlineAdmission(after: .seconds(5)))
        guard case .admitted(let stream, let work, _, _) = result else {
            XCTFail("adopted four-token tail should be reachable")
            return
        }
        assertBounded(work, tokens: 4, steps: 1, duration: .seconds(4))
        let output = await cbv2SchedCollect(stream)
        XCTAssertEqual(output.finishReason, .length)
        XCTAssertTrue(harness.model.forwardShapes.contains([1, 4]))
        XCTAssertEqual(cache.lookups, 1)
        XCTAssertEqual(cache.endAdoptions, 1)
        await harness.engine.shutdown()
    }

    func testRejectedPagedPrefixHitDoesNotWireOrReserveSlabs() async throws {
        let kind = CBv2LayerKind(
            attention: .full,
            headDim: 64,
            kvHeads: 1,
            queryHeads: 1)
        let backend = try PagedKVBackend(
            layerKinds: [kind],
            config: PagedKVPoolConfig(
                capacityBytes: 4 << 20,
                maxPrefillChunk: 256,
                nominalMaxSequenceLength: 512,
                prefixSharingBlockSize: CBv2BlockHasher.defaultBlockSize))
        let cache = CBv2SchedScriptedPrefixCache(matched: 256)
        let engine = EngineV2(
            model: CBv2SchedScriptedModel(),
            layerKinds: [kind],
            backend: backend,
            cacheProvider: CBv2SchedMockCacheProvider(layerKinds: [kind]),
            sampler: CBv2GreedySampler(),
            detokenizerFactory: CBv2SchedScriptedDetokFactory(),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 256,
                prefillChunkSize: 256,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 1,
                enablePrefixCache: true),
            admissionConfig: .init(watermarkFraction: 0),
            prefixCache: cache)
        let request = CBv2Request(
            id: CBv2RequestID(11_013),
            promptTokens: Array(0 ..< 300),
            maxTokens: 1)

        let result = try await engine.submit(
            request,
            firstTokenDeadline: Self.deadlineAdmission(after: .seconds(43)))
        guard case .deadlineUnreachable(let work) = result else {
            XCTFail("44-token replay tail must miss a 43-second budget")
            return
        }
        assertBounded(work, tokens: 44, steps: 1, duration: .seconds(44))
        XCTAssertFalse(backend.slabsAreWired)
        XCTAssertEqual(backend.bytesWired, 0)
        XCTAssertEqual(backend.bytesReserved, 0)
        XCTAssertEqual(backend.bytesInUse, 0)
        XCTAssertEqual(cache.endAdoptions, 1)
        await engine.shutdown()
    }

    func testRejectedResidentPrefixPricesReplayWithoutRetainingOrWiringPages() async throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)
        let backend = try PagedKVBackend(
            layerKinds: [kind],
            config: PagedKVPoolConfig(
                pageSize: 16, capacityBytes: 4 << 20,
                maxPrefillChunk: 32, nominalMaxSequenceLength: 128,
                maxBufferLength: Int.max),
            residentPrefixCache: .init(
                blockSize: 32, promptContractID: "deadline-test", scopeID: "model-test"))
        let tokens = Array(0 ..< 36)
        let probe = try XCTUnwrap(backend.preparePrefixProbe(tokens: tokens, cacheSalt: "tenant"))
        let needs = backend.pageNeeds(layerKinds: [kind], maxLength: 32)
        try backend.pool.reserve(needs)
        let donor = PagedSequenceKV(
            pool: backend.pool, kind: kind, groupKey: backend.pool.groupKey(forLayer: 0), maxLength: 32,
            reservedPages: PagedKVPool.pageDemand(
                kind: kind, maxLength: 32, config: backend.pool.config))
        for _ in 0 ..< 32 { _ = donor.prepareDecodeWrite() }
        XCTAssertEqual(
            backend.publishResidentPrefixBlocks(
                state: [donor], chainHashes: probe.chainHashes, blockIndices: 0 ..< 1), 1)
        backend.release([donor])

        let engine = EngineV2(
            model: CBv2SchedScriptedModel(), layerKinds: [kind], backend: backend,
            cacheProvider: CBv2SchedMockCacheProvider(layerKinds: [kind]),
            sampler: CBv2GreedySampler(), detokenizerFactory: CBv2SchedScriptedDetokFactory(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 32,
                prefillChunkSize: 32, maxConcurrentPartialPrefills: 1,
                maxWaiting: 1, enablePrefixCache: true),
            admissionConfig: .init(watermarkFraction: 0))
        let request = CBv2Request(
            id: CBv2RequestID(11_014), promptTokens: tokens, maxTokens: 1,
            cacheSalt: "tenant")
        let result = try await engine.submit(
            request, firstTokenDeadline: Self.deadlineAdmission(after: .seconds(3)))
        guard case .deadlineUnreachable(let work) = result else {
            await engine.shutdown()
            return XCTFail("four-token resident tail must miss a three-second budget")
        }
        assertBounded(work, tokens: 4, steps: 1, duration: .seconds(4))
        XCTAssertFalse(backend.slabsAreWired)
        XCTAssertEqual(backend.bytesReserved, 0)
        XCTAssertEqual(backend.bytesInUse, 0)
        XCTAssertEqual(backend.residentPrefixCacheStats?.blockCount, 1)
        engine.loopForTesting.onEngineQueueSync {
            XCTAssertNil(engine.loopForTesting.scheduler.record(for: request.id))
            XCTAssertNil(engine.loopForTesting.residentPrefixCursorByID[request.id])
        }
        await engine.shutdown()
    }

    func testPausedLedgerOwnerMakesOtherwiseOpenSlotFailClosed() async throws {
        let harness = CBv2SchedHarness(
            backendCapacity: 32,
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2,
                maxBatchedTokensPerStep: 1,
                prefillChunkSize: 1,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 2))
        let paused = CBv2Request(
            id: CBv2RequestID(11_014),
            promptTokens: [1],
            maxTokens: 1)
        try harness.engine.loopForTesting.onEngineQueueSync {
            try harness.engine.loopForTesting.scheduler.enqueue(paused)
            harness.engine.loopForTesting.scheduler.pause(paused.id)
            // Two tokens consume the full 32-byte ledger. The paused row does
            // not consume the second scheduler slot, but its KV remains live.
            try harness.engine.admissionForTesting.reserve(
                id: paused.id,
                additionalTokens: 2)
        }

        let target = CBv2Request(
            id: CBv2RequestID(11_015),
            promptTokens: [2],
            maxTokens: 1)
        let result = try await harness.engine.submit(
            target,
            firstTokenDeadline: Self.deadlineAdmission(after: .seconds(60)))
        guard case .deadlineUnreachable(let work) = result else {
            XCTFail("live KV ownership must fail closed despite an open slot")
            return
        }
        XCTAssertEqual(work, .unbounded)
        XCTAssertTrue(harness.model.forwardShapes.isEmpty)

        harness.engine.loopForTesting.onEngineQueueSync {
            _ = harness.engine.loopForTesting.scheduler.finish(
                id: paused.id,
                reason: .cancelled)
            harness.engine.admissionForTesting.releaseAll(id: paused.id)
        }
        await harness.engine.shutdown()
    }

    func testCapacityProjectionReleasesRetiredBlockerBeforeTargetReservation()
        async throws
    {
        let harness = CBv2SchedHarness(
            backendCapacity: 64,
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 1,
                prefillChunkSize: 1,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 2))
        let blocker = CBv2Request(
            id: CBv2RequestID(11_017),
            promptTokens: [1],
            maxTokens: 1)
        let target = CBv2Request(
            id: CBv2RequestID(11_018),
            promptTokens: [2],
            maxTokens: 1)
        let operations = try harness.engine.loopForTesting.onEngineQueueSync {
            try harness.engine.loopForTesting.scheduler.enqueue(blocker)
            _ = CBv2SchedSim.confirm(
                harness.engine.loopForTesting.scheduler,
                plan: harness.engine.loopForTesting.scheduler.plan())
            try harness.engine.admissionForTesting.reserve(
                id: blocker.id,
                additionalTokens: 3)
            try harness.engine.loopForTesting.scheduler.enqueue(target)
            guard case .bounded(_, let operations) =
                harness.engine.loopForTesting.scheduler
                    .firstTokenWorkProjection(for: target.id)
            else {
                throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
            }
            return operations
        }

        XCTAssertTrue(
            harness.engine.admissionForTesting.canGuarantee(
                projectedOperations: operations))
        XCTAssertEqual(
            operations.first,
            .release(blocker.id))

        harness.engine.loopForTesting.onEngineQueueSync {
            _ = harness.engine.loopForTesting.scheduler.finish(
                id: blocker.id,
                reason: .cancelled)
            _ = harness.engine.loopForTesting.scheduler.finish(
                id: target.id,
                reason: .cancelled)
            harness.engine.admissionForTesting.releaseAll(id: blocker.id)
        }
        await harness.engine.shutdown()
    }

    func testLateCancelFromRetiredGenerationDoesNotPoisonImmediateRetry() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 1,
                prefillChunkSize: 1,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 1))
        let request = CBv2Request(
            id: CBv2RequestID(11_016),
            promptTokens: [1],
            maxTokens: 1)
        let first = await cbv2SchedCollect(try harness.engine.submit(request))
        XCTAssertEqual(first.finishReason, .length)

        // Deterministically emulate an abandoned-stream callback arriving
        // after its generation has retired but before the id is reused.
        harness.engine.cancel(request.id)
        let retry = try await harness.engine.submit(
            request,
            firstTokenDeadline: Self.deadlineAdmission(after: .seconds(2)))
        guard case .admitted(let stream, _, _, _) = retry else {
            XCTFail("late cancel from the retired stream poisoned the retry")
            return
        }
        let retried = await cbv2SchedCollect(stream)
        XCTAssertEqual(retried.finishReason, .length)
        await harness.engine.shutdown()
    }

    func testEngineQueueVerdictChargesLaunchedInFlightPlan() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4,
                maxBatchedTokensPerStep: 8,
                prefillChunkSize: 4,
                maxConcurrentPartialPrefills: 1,
                maxWaiting: 8))
        // Appending the forward shape happens before this delay. Once the
        // test observes it, the engine queue is deterministically still
        // inside A's first graph build; B's admission block queues ahead of
        // the self-scheduled next step and sees A as in-flight.
        harness.model.forwardDelay = 0.15
        let first = CBv2Request(
            id: CBv2RequestID(11_020),
            promptTokens: Array(0 ..< 8),
            maxTokens: 1)
        let firstStream = try harness.engine.submit(first)
        let launched = await cbv2SchedWait {
            harness.model.forwardShapes.count == 1
        }
        XCTAssertTrue(launched)

        let second = CBv2Request(
            id: CBv2RequestID(11_021),
            promptTokens: [20, 21, 22, 23],
            maxTokens: 1)
        let result = try await harness.engine.submit(
            second,
            firstTokenDeadline: Self.deadlineAdmission(after: .seconds(11)))
        guard case .deadlineUnreachable(let work) = result else {
            XCTFail("full in-flight assignment must make the 11s budget unreachable")
            return
        }
        assertBounded(work, tokens: 12, steps: 2, duration: .seconds(12))

        let firstOutput = await cbv2SchedCollect(firstStream)
        XCTAssertEqual(firstOutput.finishReason, .length)
        // The final prompt sample may launch the engine's documented
        // one-step-late chained decode before length finalization discards it.
        // There is still no B row: an admitted B would make the final prompt
        // step rectangular [2, 4].
        XCTAssertEqual(harness.model.forwardShapes, [[1, 4], [1, 4], [1, 1]])
        harness.engine.loopForTesting.onEngineQueueSync {
            XCTAssertNil(harness.engine.loopForTesting.scheduler.record(for: second.id))
        }
        await harness.engine.shutdown()
    }
}
