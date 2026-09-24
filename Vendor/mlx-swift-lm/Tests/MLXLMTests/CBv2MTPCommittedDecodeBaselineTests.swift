import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2MTPCommittedDecodeBaseline")
struct CBv2MTPCommittedDecodeBaselineTests {
    @Test func commitIntervalsPartitionSteadyDecodeTime() {
        var clock = CBv2MTPCommittedDecodeClock()
        let rows = [CBv2RequestID(1)]
        #expect(clock.observe(completedAtNanos: 100, rowIDs: rows, eligible: true) == nil)
        let first = clock.observe(completedAtNanos: 108, rowIDs: rows, eligible: true)
        let second = clock.observe(completedAtNanos: 116, rowIDs: rows, eligible: true)
        #expect(first == 8)
        #expect(second == 8)
        #expect((first ?? 0) + (second ?? 0) == 116 - 100)
    }

    @Test func idleAndCohortChangesCannotPolluteTheBaseline() {
        var clock = CBv2MTPCommittedDecodeClock()
        let first = [CBv2RequestID(1), CBv2RequestID(2)]
        let second = [CBv2RequestID(1), CBv2RequestID(3)]
        #expect(clock.observe(completedAtNanos: 100, rowIDs: first, eligible: true) == nil)
        #expect(clock.observe(completedAtNanos: 108, rowIDs: first, eligible: true) == 8)
        // Same bucket, different membership: the next commit is an anchor.
        #expect(clock.observe(completedAtNanos: 116, rowIDs: second, eligible: true) == nil)
        #expect(clock.observe(completedAtNanos: 124, rowIDs: second, eligible: true) == 8)
        #expect(clock.observe(completedAtNanos: 132, rowIDs: second, eligible: false) == nil)
        #expect(clock.observe(completedAtNanos: 9_000, rowIDs: second, eligible: true) == nil)
        #expect(clock.observe(completedAtNanos: 9_008, rowIDs: second, eligible: true) == 8)
        #expect(clock.observe(completedAtNanos: 9_016, rowIDs: [], eligible: true) == nil)
        #expect(clock.observe(completedAtNanos: 20_000, rowIDs: first, eligible: true) == nil)
    }

    @Test func invalidClockSamplesNeverUnderflow() {
        var clock = CBv2MTPCommittedDecodeClock()
        let rows = [CBv2RequestID(1)]
        #expect(clock.observe(completedAtNanos: 0, rowIDs: rows, eligible: true) == nil)
        #expect(clock.observe(completedAtNanos: 100, rowIDs: rows, eligible: true) == nil)
        #expect(clock.observe(completedAtNanos: 100, rowIDs: rows, eligible: true) == nil)
        #expect(clock.observe(completedAtNanos: 90, rowIDs: rows, eligible: true) == nil)
        #expect(clock.observe(completedAtNanos: 98, rowIDs: rows, eligible: true) == 8)
    }

    @Test func shortRequestsStayOrdinaryUntilThreeSteadyIntervalsExist() {
        let controller = CBv2MTPDepthController(
            maxDepth: 1, fixedDepth: nil, useCommittedDecodeBaseline: true)
        let isolated = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(isolated.reason == "warmup_baseline")
        #expect(controller.requiresNonChainedDepthZeroProbe(isolated))
        complete(controller, isolated, cost: 12_000_000)
        for _ in 0 ..< 3 {
            let calibration = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            #expect(calibration.depth == 0)
            #expect(calibration.reason == "warmup_chained_baseline")
            #expect(!calibration.isExploration)
            #expect(!controller.requiresNonChainedDepthZeroProbe(calibration))
            controller.observeCommittedDecodeInterval(
                decodeRowBucket: 1, wallTimeNanos: 8_000_000)
        }
        let probe = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(probe.depth == 1)
        #expect(probe.reason == "explore_cost")
        #expect(controller.select(plannedDecodeRows: 2, canSpeculate: true).depth == 0)
    }

    @Test func fasterChainedAlternativeRejectsApparentlyProfitableMTP() throws {
        let legacy = seededController(committedBaseline: false)
        let controller = seededController(committedBaseline: true)
        // Isolated C0=12ms suggests 83.3 tokens/s. At acceptance .81,
        // C1=20ms produces 90.5 tokens/s, but chained C0 produces 125.
        #expect(legacy.select(plannedDecodeRows: 1, canSpeculate: true).depth == 1)
        let ordinary = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(ordinary.depth == 0)
        #expect(ordinary.reason == "unprofitable")
        #expect(abs(controller.snapshot().conditionalAcceptance[0] - 0.81) < 0.000_001)
        // A chained step's overlapping latency must never replace cadence.
        complete(controller, ordinary, cost: 1_000_000_000, chained: true)
        let input = try #require(controller.snapshot().costInputs.first { $0.depth == 0 })
        #expect(input.samples == 3)
        #expect(input.ewmaWallTimeNanos == 8_000_000)
        #expect(input.totalWallTimeNanos == 24_000_000)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
    }

    @Test func cooldownStillProbesAndRecoversWhenMTPBecomesFaster() {
        let controller = seededController(committedBaseline: true)
        for _ in 0 ..< 7 {
            let ordinary = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            #expect(ordinary.depth == 0)
            #expect(!ordinary.isExploration)
            complete(controller, ordinary, cost: 999_000_000, chained: true)
        }
        let failedProbe = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(failedProbe.depth == 1)
        #expect(failedProbe.reason == "explore_deeper")
        complete(controller, failedProbe, cost: 20_000_000)
        #expect(controller.probeIntervalForTesting(decodeRowBucket: 1) == 16)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)

        var recovered = false
        var successfulProbes = 0
        for _ in 0 ..< 2_048 {
            let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            if decision.depth == 0 {
                controller.observeCommittedDecodeInterval(
                    decodeRowBucket: 1, wallTimeNanos: 8_000_000)
                complete(controller, decision, cost: 999_000_000, chained: true)
            } else {
                if decision.isExploration { successfulProbes += 1 }
                complete(controller, decision, cost: 6_000_000)
                if !decision.isExploration {
                    recovered = true
                    break
                }
            }
        }
        #expect(successfulProbes > 1)
        #expect(recovered)
        #expect(controller.activeDepthForTesting(decodeRowBucket: 1) == 1)
        #expect(controller.probeIntervalForTesting(decodeRowBucket: 1) == 8)
    }

    @Test func cooldownRefreshTracksAChangedOrdinaryDecodeCost() {
        let controller = seededController(committedBaseline: true)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
        for _ in 0 ..< 40 {
            controller.observeCommittedDecodeInterval(
                decodeRowBucket: 1, wallTimeNanos: 12_000_000)
        }
        let recovered = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(recovered.depth == 1)
        #expect(recovered.reason == "goodput")
        #expect(!recovered.isExploration)
    }

    @Test func fixedDiagnosticsAndLegacyPoliciesKeepTheirExistingBaseline() throws {
        let legacy = seededController(committedBaseline: false)
        #expect(!legacy.usesCommittedDecodeBaseline)
        let input = try #require(legacy.snapshot().costInputs.first { $0.depth == 0 })
        #expect(input.ewmaWallTimeNanos == 12_000_000)
        #expect(input.samples == 1)
        let fixed = CBv2MTPDepthController(
            maxDepth: 1, fixedDepth: 1, useCommittedDecodeBaseline: true)
        #expect(!fixed.usesCommittedDecodeBaseline)
        #expect(fixed.select(plannedDecodeRows: 1, canSpeculate: true).depth == 1)
        #expect(fixed.select(plannedDecodeRows: 1, canSpeculate: true).reason == "fixed")
    }

    @Test func verificationRequiresAnIsolatedFullRoundCost() {
        let controller = seededController(committedBaseline: true)
        let decision = CBv2MTPDepthDecision(
            depth: 1, decodeRowBucket: 1, reason: "explore_deeper", isExploration: true)
        #expect(!controller.recordFinalizedStep(
            decision: decision, actualDepth: 1, wallTimeNanos: 1,
            costEligible: true, chained: true,
            finalizedPlainWork: false, finalizedVerification: true))
        #expect(controller.snapshot().costInputs.first { $0.depth == 1 }?.samples == 1)
        #expect(controller.probeIntervalForTesting(decodeRowBucket: 1) == 8)
    }

    private func seededController(committedBaseline: Bool) -> CBv2MTPDepthController {
        let controller = CBv2MTPDepthController(
            maxDepth: 1, fixedDepth: nil, useCommittedDecodeBaseline: committedBaseline)
        controller.observeCost(decodeRowBucket: 1, depth: 0, wallTimeNanos: 12_000_000)
        controller.observeCost(decodeRowBucket: 1, depth: 1, wallTimeNanos: 20_000_000)
        for _ in 0 ..< 3 {
            controller.observeCommittedDecodeInterval(
                decodeRowBucket: 1, wallTimeNanos: 8_000_000)
        }
        for _ in 0 ..< 10 {
            controller.observeAcceptance(decodeRowBucket: 1, drafted: 1, accepted: 1)
        }
        for _ in 0 ..< 2 {
            controller.observeAcceptance(decodeRowBucket: 1, drafted: 1, accepted: 0)
        }
        return controller
    }

    private func complete(
        _ controller: CBv2MTPDepthController, _ decision: CBv2MTPDepthDecision,
        cost: UInt64, chained: Bool = false
    ) {
        #expect(controller.recordFinalizedStep(
            decision: decision, actualDepth: decision.depth,
            wallTimeNanos: cost, costEligible: true, chained: chained,
            finalizedPlainWork: decision.depth == 0,
            finalizedVerification: decision.depth > 0))
    }
}
