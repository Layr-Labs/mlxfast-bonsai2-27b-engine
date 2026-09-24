// Copyright © 2026 Eigen Labs.
//
// Solo-prefill stripe (`CBv2SchedulerConfig.soloPrefillStripeTokens`): the
// stripe may only widen a prefill chunk when ONE live text request holds the
// scheduler's entire schedulable population, and must degrade to the plain
// chunk size — never to preemption or a dropped step — when KV capacity
// cannot hold it. Pure scheduler tests (no MLX arrays) plus one scripted
// engine pass proving end-to-end shapes and token invariance.

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBv2SoloStripeTests: XCTestCase {

    private func makeScheduler(
        budget: Int = 2048, chunk: Int = 512, stripe: Int? = 2048,
        capacity: CBv2StepCapacity? = nil
    ) -> SchedulerV2 {
        SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: budget,
                prefillChunkSize: chunk, soloPrefillStripeTokens: stripe,
                maxWaiting: 64),
            capacity: capacity)
    }

    // MARK: Arming

    func testSoloTextPrefillStripesAdmissionAndRunningChunks() throws {
        let scheduler = makeScheduler()
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 8192), maxTokens: 2))

        // Admission path (waiting -> running) stripes the first chunk.
        let first = scheduler.plan()
        XCTAssertEqual(first.assignments.map(\.numTokens), [2048])
        _ = CBv2SchedSim.confirm(scheduler, plan: first)
        // Running path stripes subsequent chunks.
        let second = scheduler.plan()
        XCTAssertEqual(second.assignments.map(\.numTokens), [2048])
        _ = CBv2SchedSim.confirm(scheduler, plan: second)
    }

    func testStripeLargerThanStepBudgetRaisesTheSoloBudget() throws {
        let scheduler = makeScheduler(budget: 2048, stripe: 4096)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 8192), maxTokens: 2))
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.numTokens), [4096])
    }

    func testStripeOffByDefaultKeepsPlainChunks() throws {
        let scheduler = makeScheduler(stripe: nil)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 4096), maxTokens: 2))
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [512])
    }

    func testStripeAtOrBelowPlainChunkIsIgnored() throws {
        let scheduler = makeScheduler(stripe: 512)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 4096), maxTokens: 2))
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [512])
    }

    // MARK: Disarming — anyone else's work

    func testSecondWaiterDisarmsTheStripe() throws {
        let scheduler = makeScheduler()
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 4096), maxTokens: 2))
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 64), maxTokens: 2))
        let plan = scheduler.plan()
        // Both admitted, both plain-chunked: 512 + 64.
        XCTAssertEqual(
            plan.assignments.map(\.numTokens).sorted(), [64, 512],
            "company must force plain chunks")
    }

    func testDecodeReadyCompanionDisarmsTheStripe() throws {
        let scheduler = makeScheduler()
        // Companion first: fully prefilled after two plans (64 then decode-ready).
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 64), maxTokens: 4))
        let companion = scheduler.plan()  // companion prefill (64) — solo at this instant
        _ = CBv2SchedSim.confirm(scheduler, plan: companion)  // samples → decode-ready
        // Now the big request arrives; companion decodes alongside.
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 4096), maxTokens: 2))
        let plan = scheduler.plan()
        let chunks = plan.assignments.map(\.numTokens).sorted()
        XCTAssertEqual(chunks, [1, 512], "decode row + plain 512 chunk expected: \(chunks)")
    }

    func testSoloDecodeFrontierNeverStripes() throws {
        let scheduler = makeScheduler()
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 256), maxTokens: 4))
        let prefill = scheduler.plan()
        XCTAssertEqual(prefill.assignments.map(\.numTokens), [256])
        _ = CBv2SchedSim.confirm(scheduler, plan: prefill)
        // Frontier reached: decode rows never enter stripe math.
        XCTAssertEqual(scheduler.plan().assignments.map(\.numTokens), [1])
    }

    // MARK: Capacity degradation

    func testStripeFallsBackToPlainChunkWhenCapacityRejectsIt() throws {
        // Capacity holds one plain chunk but never a stripe: the striped solo
        // row must degrade to 512 — not preempt itself, not drop the step.
        let capacity = CBv2SchedMockCapacity(tokenLimit: 600)
        let scheduler = makeScheduler(capacity: capacity)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 4096), maxTokens: 2))
        let plan = scheduler.plan()
        XCTAssertEqual(plan.assignments.map(\.numTokens), [512])
        XCTAssertTrue(plan.preemptions.isEmpty)
    }

    func testRunningPathStripeCapacityFallback() throws {
        // First chunk fits striped, later stripe exceeds the limiter: the
        // RUNNING path (not admission) must take the same one-shot shrink.
        let capacity = CBv2SchedMockCapacity(tokenLimit: 2600)
        let scheduler = makeScheduler(capacity: capacity)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 8192), maxTokens: 2))
        let first = scheduler.plan()
        XCTAssertEqual(first.assignments.map(\.numTokens), [2048])
        _ = CBv2SchedSim.confirm(scheduler, plan: first)
        let second = scheduler.plan()
        XCTAssertEqual(second.assignments.map(\.numTokens), [512])
        XCTAssertTrue(second.preemptions.isEmpty)
    }
}

// MARK: - End-to-end scripted engine

final class CBv2SoloStripeEngineTests: XCTestCase {

    /// The stripe changes the FORWARD SHAPES of a solo prefill and nothing
    /// about its tokens: the scripted model (next = input + 1) is chunking-
    /// invariant, so equal outputs across configs prove the plumbing.
    func testSoloStripeWidensForwardShapesAndPreservesTokens() async throws {
        let plain = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 16, maxWaiting: 8))
        let striped = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 16, soloPrefillStripeTokens: 48, maxWaiting: 8))

        let prompt = Array(0 ..< 40)
        let plainOut = await cbv2SchedCollect(
            try plain.engine.submit(CBv2SchedFixtures.request(prompt: prompt, maxTokens: 3)))
        let stripedOut = await cbv2SchedCollect(
            try striped.engine.submit(CBv2SchedFixtures.request(prompt: prompt, maxTokens: 3)))

        XCTAssertEqual(plainOut.tokens, stripedOut.tokens)
        XCTAssertEqual(stripedOut.finishReason, .length)
        XCTAssertTrue(
            plain.model.forwardShapes.contains([1, 16]),
            "plain arm must chunk at 16: \(plain.model.forwardShapes)")
        XCTAssertTrue(
            striped.model.forwardShapes.contains([1, 40]),
            "striped arm must run the whole prompt in one chunk: \(striped.model.forwardShapes)")
        XCTAssertFalse(
            striped.model.forwardShapes.contains([1, 16]),
            "striped solo prefill must not fall back to plain chunks: \(striped.model.forwardShapes)")
    }
}


// MARK: - Mean-TTFT prefill serialization (maxConcurrentPartialPrefills)

final class CBv2PartialPrefillCapTests: XCTestCase {

    private func makeScheduler(cap: Int?, stripe: Int? = nil) -> SchedulerV2 {
        SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 2048,
                prefillChunkSize: 512, soloPrefillStripeTokens: stripe,
                maxConcurrentPartialPrefills: cap, maxWaiting: 64),
            capacity: nil)
    }

    func testCapOneSerializesPrefillAdmission() throws {
        let scheduler = makeScheduler(cap: 1)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 1024), maxTokens: 2))
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 1024), maxTokens: 2))

        let p1 = scheduler.plan()
        XCTAssertEqual(p1.assignments.map(\.numTokens), [512], "only A admits under cap 1")
        _ = CBv2SchedSim.confirm(scheduler, plan: p1)
        // A's FINAL prefill chunk samples this very step, so admitting B
        // alongside it delays nobody and starts B one step earlier.
        let p2 = scheduler.plan()
        XCTAssertEqual(
            p2.assignments.map(\.numTokens).sorted(), [512, 512],
            "B admits alongside A's final (sampling) chunk")
        _ = CBv2SchedSim.confirm(scheduler, plan: p2)
        let p3 = scheduler.plan()
        XCTAssertEqual(
            p3.assignments.map(\.numTokens).sorted(), [1, 512],
            "A decodes alongside B's remaining prefill")
    }


    func testNonpositiveCapIsTreatedAsUnlimited() throws {
        for cap in [0, -1] {
            let scheduler = makeScheduler(cap: cap)
            try scheduler.enqueue(
                CBv2SchedFixtures.request(prompt: Array(0 ..< 1024), maxTokens: 2))
            try scheduler.enqueue(
                CBv2SchedFixtures.request(prompt: Array(0 ..< 1024), maxTokens: 2))
            XCTAssertEqual(
                scheduler.plan().assignments.map(\.numTokens), [512, 512],
                "cap \(cap) must fail open to the unlimited interleave, never starve")
        }
    }


    func testSuccessorAdmittedInFinalStripedStepTakesPlainChunk() throws {
        // A = 2,500 tokens: one 2,048 stripe, then a 452 remainder that
        // SAMPLES. The successor admitted alongside that final step must
        // take the plain 512 chunk — never the leftover striped budget —
        // or its oversized first forward delays A's own first token.
        let scheduler = makeScheduler(cap: 1, stripe: 2048)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 2500), maxTokens: 2))
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 4096), maxTokens: 2))

        let p1 = scheduler.plan()
        XCTAssertEqual(p1.assignments.map(\.numTokens), [2048])
        _ = CBv2SchedSim.confirm(scheduler, plan: p1)
        let p2 = scheduler.plan()
        XCTAssertEqual(
            p2.assignments.map(\.numTokens).sorted(), [452, 512],
            "successor must admit at the PLAIN chunk beside A's final remainder")
    }


    func testPausedPartialHoldsNoSlotButResumeReserializesWork() throws {
        // A pauses mid-prefill: B must still admit (a stalled consumer must
        // not head-of-line block admission). On resume, only ONE of them
        // receives prompt work per step — FCFS gives the elder A the slot.
        let scheduler = makeScheduler(cap: 1)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 2048), maxTokens: 2))
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 2048), maxTokens: 2))
        let p1 = scheduler.plan()
        XCTAssertEqual(p1.assignments.map(\.numTokens), [512], "cap admits only A")
        _ = CBv2SchedSim.confirm(scheduler, plan: p1)

        let a = scheduler.running[0]
        a.isPaused = true
        let p2 = scheduler.plan()
        XCTAssertEqual(
            p2.assignments.map(\.numTokens), [512],
            "paused A holds no slot: B admits and progresses")
        _ = CBv2SchedSim.confirm(scheduler, plan: p2)

        a.isPaused = false
        // Both A and B are now mid-prefill (transient over-membership), but
        // each subsequent step assigns prompt work to exactly ONE of them.
        for _ in 0 ..< 2 {
            let plan = scheduler.plan()
            XCTAssertEqual(
                plan.assignments.map(\.numTokens), [512],
                "post-resume steps serialize prompt work to one row")
            _ = CBv2SchedSim.confirm(scheduler, plan: plan)
        }
    }


    func testStripeSurplusNeverLeaksToSuccessorBeyondStepBudget() throws {
        // Non-default posture: stripe (2048) > step budget (512). The raise
        // exists only for the armed row; the successor beside A's final
        // 152-token remainder is bounded by the ORDINARY limit, so the step
        // never exceeds maxBatchedTokensPerStep.
        let scheduler = SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 512,
                prefillChunkSize: 512, soloPrefillStripeTokens: 2048,
                maxConcurrentPartialPrefills: 1, maxWaiting: 64),
            capacity: nil)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 2200), maxTokens: 2))
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 4096), maxTokens: 2))

        let p1 = scheduler.plan()
        XCTAssertEqual(p1.assignments.map(\.numTokens), [2048], "armed row may exceed the step budget")
        _ = CBv2SchedSim.confirm(scheduler, plan: p1)
        let p2 = scheduler.plan()
        XCTAssertEqual(
            p2.assignments.map(\.numTokens).sorted(), [152, 360],
            "successor bounded by the ordinary limit: 152 + 360 == 512")
        XCTAssertLessThanOrEqual(
            p2.assignments.reduce(0) { $0 + $1.numTokens },
            scheduler.config.maxBatchedTokensPerStep,
            "non-armed consumers never exceed maxBatchedTokensPerStep")
    }

    func testCapUnsetKeepsInterleavedAdmission() throws {
        let scheduler = makeScheduler(cap: nil)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 1024), maxTokens: 2))
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 1024), maxTokens: 2))
        XCTAssertEqual(
            scheduler.plan().assignments.map(\.numTokens), [512, 512],
            "unlimited interleave is unchanged")
    }

    func testCapOnePlusStripeStripesTheActiveRowDespiteWaiters() throws {
        let scheduler = makeScheduler(cap: 1, stripe: 2048)
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 4096), maxTokens: 2))
        try scheduler.enqueue(CBv2SchedFixtures.request(prompt: Array(0 ..< 4096), maxTokens: 2))

        let p1 = scheduler.plan()
        XCTAssertEqual(
            p1.assignments.map(\.numTokens), [2048],
            "serialized-prefill policy stripes the active row even with a waiter")
        _ = CBv2SchedSim.confirm(scheduler, plan: p1)
        let p2 = scheduler.plan()
        XCTAssertEqual(p2.assignments.map(\.numTokens), [2048])
        _ = CBv2SchedSim.confirm(scheduler, plan: p2)
        // A sampled -> decode-ready company: stripe disarms; B admits at 512.
        let p3 = scheduler.plan()
        XCTAssertEqual(
            p3.assignments.map(\.numTokens).sorted(), [1, 512],
            "decode company disarms the stripe for B")
    }
}
