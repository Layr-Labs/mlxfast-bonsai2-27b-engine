// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — per-request timing (`CBv2RequestTiming`) and the
// cumulative step counters on `CBv2CapacitySnapshot`.
//
// Runs on the scripted no-weights harness (`CBv2SchedHarness`): the model
// emits next-token = (input + 1) % vocab, so every completion count is exact
// and the timing invariants below are deterministic in STRUCTURE (ordering,
// counts) while the nanosecond values are merely required to be observed.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2RequestTimingTests: XCTestCase {

    // Host-sync counting is gated off on the step path; this suite (serial
    // XCTest) switches it on per test and asserts one sync per step.
    override func setUp() {
        super.setUp()
        CBv2CoreInstrumentation.countingEnabled = true
    }

    override func tearDown() {
        CBv2CoreInstrumentation.countingEnabled = false
        super.tearDown()
    }

    private func timing(_ collected: CBv2SchedCollected, _ label: String) throws
        -> CBv2RequestTiming
    {
        let usage = try XCTUnwrap(collected.usage, "\(label): terminal usage missing")
        return usage.timing
    }

    /// The phase stamps every finished request must satisfy, in order —
    /// the chain the coordinator validates on the exported profile:
    /// admitted ≤ kvAllocated ≤ prefillFirstLaunch ≤ promptComputed ≤
    /// firstToken ≤ finished.
    private func assertPhaseOrdering(_ t: CBv2RequestTiming, _ label: String) {
        XCTAssertGreaterThan(t.admittedNanos, 0, "\(label): admitted")
        // KV is allocated at first launch (`ensureKVState`) or, for a prefix
        // hit, at enqueue-time adoption — which `stampAdmission` raises to
        // the admission instant, so the stamp never precedes admission and
        // always precedes the prompt being confirmed computed.
        XCTAssertGreaterThan(t.kvAllocatedNanos, 0, "\(label): kv allocated")
        XCTAssertLessThanOrEqual(
            t.admittedNanos, t.kvAllocatedNanos, "\(label): admitted ≤ kv")
        XCTAssertLessThanOrEqual(
            t.kvAllocatedNanos, t.prefillFirstLaunchNanos, "\(label): kv ≤ prefill launch")
        XCTAssertLessThanOrEqual(
            t.kvAllocatedNanos, t.promptComputedNanos, "\(label): kv ≤ promptComputed")
        XCTAssertGreaterThanOrEqual(
            t.prefillFirstLaunchNanos, t.admittedNanos, "\(label): prefill ≥ admitted")
        XCTAssertGreaterThanOrEqual(
            t.promptComputedNanos, t.prefillFirstLaunchNanos,
            "\(label): promptComputed ≥ prefill launch")
        XCTAssertGreaterThanOrEqual(
            t.firstTokenNanos, t.promptComputedNanos, "\(label): firstToken ≥ promptComputed")
        XCTAssertGreaterThanOrEqual(
            t.finishedNanos, t.firstTokenNanos, "\(label): finished ≥ firstToken")
    }

    // MARK: B = 1

    func testSingleDecodeRunPopulatesTiming() async throws {
        let harness = CBv2SchedHarness()
        let request = CBv2SchedFixtures.request(prompt: [3], maxTokens: 8)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))
        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertEqual(collected.usage?.completionTokens, 8)
        let t = try timing(collected, "B=1")

        assertPhaseOrdering(t, "B=1")
        // The final (only) prefill chunk samples the first token, so the
        // prompt is computed exactly when the first token is confirmed.
        XCTAssertEqual(t.promptComputedNanos, t.firstTokenNanos)

        XCTAssertEqual(t.prefillChunks, 1)
        XCTAssertEqual(t.packedPrefillChunks, 0)
        XCTAssertEqual(t.visionChunks, 0)
        XCTAssertEqual(t.soloStripeChunks, 0)
        XCTAssertEqual(t.prefillChunkTokensMax, 1)

        // Every token beyond the first came from a decode step; the wasted
        // chained step after `.length` is discarded, never counted.
        XCTAssertEqual(t.decodeSteps, 7, "decodeSteps == completion − 1")
        XCTAssertGreaterThanOrEqual(t.chainedDecodeSteps, 1, "steady decode chains")
        XCTAssertLessThanOrEqual(t.chainedDecodeSteps, t.decodeSteps)

        // Alone in the engine: batch size 1 in all 8 participated steps.
        XCTAssertEqual(t.batchRowsMin, 1)
        XCTAssertEqual(t.batchRowsMax, 1)
        XCTAssertEqual(t.batchRowsSum, 8)
        XCTAssertGreaterThan(t.stepLatencyNanosSum, 0)
        XCTAssertGreaterThan(t.stepLatencyNanosMax, 0)
        XCTAssertLessThanOrEqual(t.stepLatencyNanosMax, t.stepLatencyNanosSum)

        XCTAssertEqual(t.readmissions, 0)
        XCTAssertEqual(t.preemptions, 0)
        XCTAssertEqual(t.capacityRequeues, 0)
        XCTAssertEqual(t.mtpRounds, 0)
        XCTAssertEqual(t.mtpProposed, 0)
        XCTAssertEqual(t.mtpAccepted, 0)
        XCTAssertEqual(t.pauseCount, 0)
        XCTAssertEqual(t.pausedNanos, 0)
        // Passthrough request: the first token's detok hop was measured.
        XCTAssertGreaterThan(t.detokDelayFirstNanos, 0)
        // No prefix cache configured: nothing to time.
        XCTAssertEqual(t.prefixLookupNanos, 0)
        XCTAssertEqual(t.prefixAdoptionNanos, 0)

        // Cumulative counters after a quiescent single run.
        let released = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(released)
        let snapshot = harness.engine.capacity()
        XCTAssertGreaterThan(snapshot.stepsExecuted, 0)
        XCTAssertGreaterThan(snapshot.stepWallNanosTotal, 0)
        XCTAssertEqual(snapshot.decodeRowsTotal, UInt64(t.decodeSteps))
        XCTAssertGreaterThanOrEqual(snapshot.stepWallNanosTotal, t.stepLatencyNanosSum)
    }

    // MARK: B = 4

    func testBatchedDecodeRunPopulatesTimingWithoutHostSyncs() async throws {
        let harness = CBv2SchedHarness()
        let requests = [
            CBv2SchedFixtures.request(prompt: [3], maxTokens: 10),
            CBv2SchedFixtures.request(prompt: [20, 21], maxTokens: 12),
            CBv2SchedFixtures.request(prompt: [40, 41, 42], maxTokens: 9),
            CBv2SchedFixtures.request(prompt: [50], maxTokens: 11),
        ]
        let syncsBefore = CBv2CoreInstrumentation.hostSyncs
        let streams = try requests.map { try harness.engine.submit($0) }
        var collected: [CBv2SchedCollected] = []
        await withTaskGroup(of: (Int, CBv2SchedCollected).self) { group in
            for (index, stream) in streams.enumerated() {
                group.addTask { (index, await cbv2SchedCollect(stream)) }
            }
            var byIndex: [Int: CBv2SchedCollected] = [:]
            for await (index, out) in group { byIndex[index] = out }
            collected = (0 ..< streams.count).map { byIndex[$0]! }
        }
        // Host-sync accounting is asserted below, once every step is fenced.

        var totalDecodeSteps: UInt64 = 0
        var widestBatch: UInt32 = 0
        var maxLatencySum: UInt64 = 0
        for (request, out) in zip(requests, collected) {
            let label = "request \(request.id)"
            XCTAssertEqual(out.finishReason, .length, label)
            XCTAssertEqual(out.usage?.completionTokens, request.maxTokens, label)
            let t = try timing(out, label)
            assertPhaseOrdering(t, label)
            XCTAssertEqual(t.prefillChunks, 1, label)
            XCTAssertEqual(t.prefillChunkTokensMax, UInt32(request.promptTokens.count), label)
            XCTAssertEqual(t.decodeSteps, UInt32(request.maxTokens - 1), label)
            XCTAssertGreaterThanOrEqual(t.batchRowsMin, 1, label)
            XCTAssertGreaterThanOrEqual(t.batchRowsMax, t.batchRowsMin, label)
            XCTAssertLessThanOrEqual(t.batchRowsMax, 4, label)
            // Σ batch rows over decodeSteps + 1 participated steps is bounded
            // by min × steps ≤ Σ ≤ max × steps.
            let steps = UInt64(t.decodeSteps + 1)
            XCTAssertGreaterThanOrEqual(t.batchRowsSum, UInt64(t.batchRowsMin) * steps, label)
            XCTAssertLessThanOrEqual(t.batchRowsSum, UInt64(t.batchRowsMax) * steps, label)
            XCTAssertGreaterThan(t.stepLatencyNanosSum, 0, label)
            XCTAssertGreaterThan(t.detokDelayFirstNanos, 0, label)
            XCTAssertEqual(t.readmissions, 0, label)
            XCTAssertEqual(t.preemptions, 0, label)
            totalDecodeSteps += UInt64(t.decodeSteps)
            widestBatch = max(widestBatch, t.batchRowsMax)
            maxLatencySum = max(maxLatencySum, t.stepLatencyNanosSum)
        }
        XCTAssertGreaterThanOrEqual(widestBatch, 2, "four concurrent rows must batch")

        let released = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(released)
        let snapshot = harness.engine.capacity()
        // Timing is always on and adds no host sync: the step loop's ONLY
        // CBv2 sync is the finalize readback — exactly one per executed step
        // (every launched step is finalized once; all fenced by now). No
        // row asks for logprobs (three readbacks per segment) and MTP is
        // off, so the per-step formula is exactly 1. Any sync introduced
        // later breaks this equality.
        XCTAssertEqual(
            CBv2CoreInstrumentation.hostSyncs - syncsBefore, snapshot.stepsExecuted,
            "one host sync per executed step")
        XCTAssertEqual(
            snapshot.decodeRowsTotal, totalDecodeSteps,
            "decodeRowsTotal == Σ per-request decodeSteps")
        XCTAssertGreaterThanOrEqual(snapshot.stepWallNanosTotal, maxLatencySum)
        XCTAssertGreaterThan(snapshot.stepsExecuted, 0)
    }

    // MARK: Prefill shapes

    func testMultiChunkPrefillCountsEveryChunk() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64, prefillChunkSize: 16,
                maxWaiting: 8))
        let request = CBv2SchedFixtures.request(prompt: Array(0 ..< 40), maxTokens: 3)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))
        XCTAssertEqual(collected.tokens, [40, 41, 42])
        let t = try timing(collected, "chunked")

        assertPhaseOrdering(t, "chunked")
        XCTAssertEqual(t.prefillChunks, 3, "16 + 16 + 8")
        XCTAssertEqual(t.prefillChunkTokensMax, 16)
        XCTAssertEqual(t.packedPrefillChunks, 0)
        XCTAssertEqual(t.soloStripeChunks, 0, "no stripe armed: nothing wider than a chunk")
        XCTAssertEqual(t.decodeSteps, 2)
        XCTAssertEqual(t.promptComputedNanos, t.firstTokenNanos)
        // Two non-sampling chunks precede the sampling one: the first
        // launch strictly precedes the prompt-computed confirmation.
        XCTAssertLessThan(t.prefillFirstLaunchNanos, t.promptComputedNanos)
    }

    func testSoloPrefillStripeIsCounted() async throws {
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64, prefillChunkSize: 16,
                soloPrefillStripeTokens: 32, maxWaiting: 8))
        let request = CBv2SchedFixtures.request(prompt: Array(0 ..< 40), maxTokens: 2)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))
        XCTAssertEqual(collected.tokens, [40, 41])
        let t = try timing(collected, "stripe")
        XCTAssertGreaterThanOrEqual(t.soloStripeChunks, 1, "a solo row takes the stripe")
        XCTAssertEqual(t.prefillChunkTokensMax, 32)
        XCTAssertEqual(t.prefillChunks, 2, "32 + 8")
    }

    // MARK: Detokenization paths

    func testStopStringRequestLeavesDetokDelayUnobserved() async throws {
        let harness = CBv2SchedHarness(stopTrigger: 20)
        let request = CBv2SchedFixtures.request(
            prompt: [17], maxTokens: 100, stopStrings: ["<stop>"])
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))
        XCTAssertEqual(collected.tokens, [18, 19, 20])
        XCTAssertEqual(collected.finishReason, .stop)
        let t = try timing(collected, "stop-string")
        assertPhaseOrdering(t, "stop-string")
        XCTAssertEqual(t.decodeSteps, 2)
        // Synchronous detokenization on the engine thread: no deferred hop
        // to measure, so the passthrough-only probe stays unobserved.
        XCTAssertEqual(t.detokDelayFirstNanos, 0)
    }

    // MARK: Terminal before first token

    func testCancelDuringPrefillLeavesFirstTokenUnobserved() async throws {
        // Ten [1, 16] prefill chunks, one per step, each held ~50 ms inside
        // the scripted forward: the cancel lands after the first chunk
        // launched and long before the sampling chunk could finalize.
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64, prefillChunkSize: 16,
                maxWaiting: 8))
        harness.model.forwardDelay = 0.05
        let request = CBv2SchedFixtures.request(prompt: Array(0 ..< 160), maxTokens: 4)
        let stream = try harness.engine.submit(request)
        let collector = Task { await cbv2SchedCollect(stream) }
        let launched = await cbv2SchedWait { harness.engine.stepCount >= 1 }
        XCTAssertTrue(launched, "first prefill chunk must launch")
        harness.engine.cancel(request.id)
        let collected = await collector.value

        XCTAssertEqual(collected.finishReason, .cancelled)
        XCTAssertTrue(collected.tokens.isEmpty, "cancelled mid-prefill: no tokens")
        let t = try timing(collected, "cancel-in-prefill")

        // Never observed: the prompt was not computed through and no token
        // was confirmed or detokenized.
        XCTAssertEqual(t.promptComputedNanos, 0)
        XCTAssertEqual(t.firstTokenNanos, 0)
        XCTAssertEqual(t.decodeSteps, 0)
        XCTAssertEqual(t.detokDelayFirstNanos, 0)

        // Observed and ordered: admitted ≤ kvAllocated ≤ prefillFirstLaunch
        // ≤ finished (the coordinator validates every PRESENT pair).
        XCTAssertGreaterThan(t.admittedNanos, 0)
        XCTAssertGreaterThan(t.kvAllocatedNanos, 0)
        XCTAssertGreaterThan(t.prefillFirstLaunchNanos, 0)
        XCTAssertGreaterThan(t.finishedNanos, 0)
        XCTAssertLessThanOrEqual(t.admittedNanos, t.kvAllocatedNanos)
        XCTAssertLessThanOrEqual(t.kvAllocatedNanos, t.prefillFirstLaunchNanos)
        XCTAssertLessThanOrEqual(t.prefillFirstLaunchNanos, t.finishedNanos)

        XCTAssertGreaterThanOrEqual(t.prefillChunks, 1, "at least the launched chunk")
        XCTAssertLessThan(t.prefillChunks, 10, "cancelled before the prompt was through")
        XCTAssertEqual(t.prefillChunkTokensMax, 16)

        let released = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(released, "cancelled request's KV must be freed")
    }

    // MARK: Re-admission, preemption, backpressure

    func testPreemptedRequestCountsPreemptionAndReadmission() async throws {
        // Same fixture as the loop suite's preemption test: a 12-token soft
        // ledger cannot hold two 10-token rows, so the younger is preempted
        // mid-decode, re-prefills (kept tokens), and finishes seamlessly.
        let harness = CBv2SchedHarness(backendCapacity: 200)
        let older = CBv2SchedFixtures.request(prompt: [1, 2, 3, 4], maxTokens: 6)
        let younger = CBv2SchedFixtures.request(prompt: [40, 41, 42, 43], maxTokens: 6)
        let olderStream = try harness.engine.submit(older)
        let youngerStream = try harness.engine.submit(younger)
        let olderTask = Task { await cbv2SchedCollect(olderStream) }
        let youngerTask = Task { await cbv2SchedCollect(youngerStream) }
        let olderCollected = await olderTask.value
        let youngerCollected = await youngerTask.value
        XCTAssertEqual(olderCollected.tokens, [5, 6, 7, 8, 9, 10])
        XCTAssertEqual(youngerCollected.tokens, [44, 45, 46, 47, 48, 49])
        XCTAssertGreaterThan(harness.engine.preemptionCount, 0)

        let olderTiming = try timing(olderCollected, "older")
        let youngerTiming = try timing(youngerCollected, "younger")
        assertPhaseOrdering(olderTiming, "older")
        assertPhaseOrdering(youngerTiming, "younger")
        XCTAssertEqual(olderTiming.decodeSteps, 5, "every non-first token is a decode step")
        XCTAssertEqual(youngerTiming.decodeSteps, 5, "every non-first token is a decode step")

        var victims = 0
        for t in [olderTiming, youngerTiming] where t.preemptions > 0 {
            victims += 1
            XCTAssertGreaterThanOrEqual(t.readmissions, t.preemptions, "each preemption re-admits")
            XCTAssertGreaterThanOrEqual(t.prefillChunks, 2, "re-prefill launches a new chunk")
        }
        XCTAssertGreaterThan(victims, 0, "capacity forces at least one preemption")
    }

    func testBackpressurePauseIsAccounted() async throws {
        let harness = CBv2SchedHarness(
            loopConfig: CBv2EngineLoopConfig(eventBufferCapacity: 4))
        let slow = CBv2SchedFixtures.request(prompt: [1], maxTokens: 30)
        let fast = CBv2SchedFixtures.request(prompt: [20], maxTokens: 30)
        let slowStream = try harness.engine.submit(slow)
        let fastStream = try harness.engine.submit(fast)

        let fastOut = await cbv2SchedCollect(fastStream)
        XCTAssertEqual(fastOut.finishReason, .length)
        let paused = await cbv2SchedWait {
            harness.engine.loopForTesting.pausedIDsSnapshot() == [slow.id]
        }
        XCTAssertTrue(paused, "slow stream must be paused while unread")
        // Hold the pause long enough to be measurable, then drain.
        try await Task.sleep(nanoseconds: 20_000_000)
        let slowOut = await cbv2SchedCollect(slowStream)
        XCTAssertEqual(slowOut.finishReason, .length)

        let t = try timing(slowOut, "slow")
        XCTAssertGreaterThanOrEqual(t.pauseCount, 1)
        XCTAssertGreaterThanOrEqual(t.pausedNanos, 10_000_000, "≥ 10 ms of the 20 ms hold")
        XCTAssertEqual(try timing(fastOut, "fast").pauseCount, 0)
        // Let the last chained step finalize before the next test measures
        // the process-global host-sync counter.
        let released = await cbv2SchedWait { harness.backend.liveStates == 0 }
        XCTAssertTrue(released)
    }

    // MARK: Prefix cache

    func testPrefixHitRecordsLookupAndAdoptionDurations() async throws {
        let prefixCache = CBv2SchedScriptedPrefixCache(matched: 2)
        let harness = CBv2SchedHarness(
            schedulerConfig: CBv2SchedulerConfig(enablePrefixCache: true),
            prefixCache: prefixCache)
        let request = CBv2SchedFixtures.request(prompt: [1, 2, 3, 4], maxTokens: 3)
        let collected = await cbv2SchedCollect(try harness.engine.submit(request))
        XCTAssertEqual(collected.tokens, [5, 6, 7])
        XCTAssertEqual(collected.usage?.prefixCacheOutcome, .hit)
        let t = try timing(collected, "prefix hit")
        assertPhaseOrdering(t, "prefix hit")
        XCTAssertGreaterThan(t.prefixLookupNanos, 0, "submit-thread lookup timed")
        XCTAssertGreaterThan(t.prefixAdoptionNanos, 0, "applyAdoption timed")
        // Adoption allocates KV at enqueue, before the first plan; the
        // exported stamp is raised to the admission instant so the
        // coordinator's `admitted ≤ kvAllocated` order holds. The adoption
        // itself is evidenced by `prefixAdoptionNanos > 0` above.
        XCTAssertEqual(
            t.kvAllocatedNanos, t.admittedNanos, "adopted KV stamp raised to admission")
        XCTAssertEqual(t.prefillChunks, 1, "only the un-adopted tail is prefilled")
    }

    // MARK: Cumulative counters

    func testCumulativeStepCountersAreMonotonic() async throws {
        let harness = CBv2SchedHarness()
        let request = CBv2SchedFixtures.request(prompt: [7], maxTokens: 40)
        let stream = try harness.engine.submit(request)
        let collector = Task { await cbv2SchedCollect(stream) }

        var samples: [CBv2CapacitySnapshot] = []
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            samples.append(harness.engine.capacity())
            if let last = samples.last, last.activeRequests == 0, samples.count > 10 { break }
            try await Task.sleep(nanoseconds: 500_000)
        }
        let collected = await collector.value
        XCTAssertEqual(collected.finishReason, .length)
        samples.append(harness.engine.capacity())

        XCTAssertGreaterThan(samples.count, 2, "sampled the run while it was live")
        for (previous, next) in zip(samples, samples.dropFirst()) {
            XCTAssertGreaterThanOrEqual(next.stepsExecuted, previous.stepsExecuted)
            XCTAssertGreaterThanOrEqual(next.stepWallNanosTotal, previous.stepWallNanosTotal)
            XCTAssertGreaterThanOrEqual(next.decodeRowsTotal, previous.decodeRowsTotal)
        }
        let t = try timing(collected, "monotonic")
        let final = try XCTUnwrap(samples.last)
        XCTAssertEqual(final.decodeRowsTotal, UInt64(t.decodeSteps))
        XCTAssertGreaterThan(final.stepWallNanosTotal, 0)
    }

    // MARK: Contract shape

    func testTimingIsEquatableAndNumericOnly() {
        var a = CBv2RequestTiming()
        XCTAssertEqual(a, CBv2RequestTiming(), "defaults are all zero")
        a.admittedNanos = 5
        a.decodeSteps = 3
        a.batchRowsSum = 12
        let b = a
        XCTAssertEqual(a, b, "value round trip")
        a.mtpAccepted += 1
        XCTAssertNotEqual(a, b)

        // Confidentiality guard: the struct carries unsigned integers only —
        // no token ids, text, hashes, or pointers can ever ride it.
        let children = Mirror(reflecting: CBv2RequestTiming()).children
        XCTAssertEqual(children.count, 29, "field count pinned to the contract")
        for child in children {
            XCTAssertTrue(
                child.value is UInt64 || child.value is UInt32,
                "\(child.label ?? "?") must be UInt64/UInt32, got \(type(of: child.value))")
        }

        // Defaulted additions keep the existing usage/snapshot shapes intact.
        let usage = CBv2Usage(promptTokens: 1, completionTokens: 2)
        XCTAssertEqual(usage.timing, CBv2RequestTiming())
        let snapshot = CBv2CapacitySnapshot(
            activeRequests: 0, waitingRequests: 0, kvBytesInUse: 0, kvBytesCapacity: 0,
            activeTokens: 0)
        XCTAssertEqual(snapshot.stepWallNanosTotal, 0)
        XCTAssertEqual(snapshot.decodeRowsTotal, 0)
    }
}
