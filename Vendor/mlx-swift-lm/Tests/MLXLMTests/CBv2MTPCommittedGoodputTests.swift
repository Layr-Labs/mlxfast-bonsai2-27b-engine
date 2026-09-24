import Testing

@testable import MLXLMCommon

@Suite("CBv2MTPCommittedGoodput")
struct CBv2MTPCommittedGoodputTests {
    @Test func seedAndVerifyOwnDisjointTimeAndAllCommittedOutputs() {
        var clock = CBv2MTPCommittedGoodputClock()
        let rows = [CBv2RequestID(1)]
        #expect(clock.observe(
            measurement: measurement(depth: 0), completedAtNanos: 100,
            isolatedWallTimeNanos: 12, rowIDs: rows, committedTokens: 1) == nil)
        #expect(clock.observe(
            measurement: measurement(depth: 1, seed: true), completedAtNanos: 108,
            isolatedWallTimeNanos: 7, rowIDs: rows, committedTokens: 1) == nil)
        let seeded = clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 124,
            isolatedWallTimeNanos: 14, rowIDs: rows, committedTokens: 2)
        #expect(seeded == .init(wallTimeNanos: 24, committedTokens: 3))
        let continued = clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 140,
            isolatedWallTimeNanos: 14, rowIDs: rows, committedTokens: 2)
        #expect(continued == .init(wallTimeNanos: 16, committedTokens: 2))
        #expect((seeded?.wallTimeNanos ?? 0) + (continued?.wallTimeNanos ?? 0) == 140 - 100)
    }

    @Test func cancellationCohortAndIdleResetCannotChargeAnotherRequest() {
        var clock = CBv2MTPCommittedGoodputClock()
        let first = [CBv2RequestID(1)]
        let second = [CBv2RequestID(2)]
        _ = clock.observe(
            measurement: measurement(depth: 1, seed: true), completedAtNanos: 100,
            isolatedWallTimeNanos: 8, rowIDs: first, committedTokens: 1)
        #expect(clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 500,
            isolatedWallTimeNanos: 16, rowIDs: second, committedTokens: 1)
                == .init(wallTimeNanos: 16, committedTokens: 1))
        clock.reset()
        #expect(clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 50_000,
            isolatedWallTimeNanos: 17, rowIDs: second, committedTokens: 2)
                == .init(wallTimeNanos: 17, committedTokens: 2))
        #expect(clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 50_010,
            isolatedWallTimeNanos: 10, rowIDs: [], committedTokens: 0) == nil)
        #expect(clock.observe(
            measurement: measurement(depth: 1), completedAtNanos: 90_000,
            isolatedWallTimeNanos: 18, rowIDs: second, committedTokens: 1)
                == .init(wallTimeNanos: 18, committedTokens: 1))
    }

    @Test(arguments: [1, 2, 4])
    func wholeWindowDiscoversWarmedShapeWithRejectedDraft(rows: Int) throws {
        let controller = warmedFreshController(rows: rows)
        let probe = controller.select(plannedDecodeRows: rows, canSpeculate: true)
        // Seed 12ms/one output + eight 15ms verification rounds. One of
        // eight drafts rejects (87.5% acceptance): 132ms / 16 outputs.
        for index in 0 ..< 8 {
            let decision = controller.select(plannedDecodeRows: rows, canSpeculate: true)
            #expect(decision.depth == 1)
            let recorded = controller.recordCommittedVerification(
                decision: decision, wallTimeNanos: index == 0 ? 27_000_000 : 15_000_000,
                committedTokens: (index == 0 ? 3 : (index == 4 ? 1 : 2)) * rows,
                rowCount: rows)
            #expect(recorded == (index == 7))
            if index < 7 {
                #expect(controller.snapshot().costInputs.allSatisfy { $0.depth == 0 })
                #expect(controller.select(plannedDecodeRows: rows, canSpeculate: true).reason == "explore_window")
            }
        }
        let input = try #require(controller.snapshot().costInputs.first { $0.depth == 1 })
        #expect(input.samples == 1)
        #expect(input.ewmaWallTimeNanos == 132_000_000)
        #expect(input.ewmaNanosPerCommittedToken == 8_250_000)
        #expect(controller.select(plannedDecodeRows: rows, canSpeculate: true).reason == "goodput")
        #expect(probe.reason == "explore_cost")
    }

    @Test func windowHardCapCannotBeExtendedByAdditionalObservations() {
        var window = CBv2MTPCommittedWindow(
            decision: .init(depth: 1, decodeRowBucket: 1, reason: "explore_cost", isExploration: true),
            rowCount: 1)
        for index in 0 ..< 20 {
            window.observe(wallTimeNanos: 15_000_000, committedTokens: 2, warmup: index == 0)
        }
        #expect(window.verifiedRounds == 8)
        #expect(window.wallTimeNanos == 105_000_000)
        #expect(window.committedTokens == 14)
    }

    @Test func shapeWarmupConsumesOneOfEightSlotsWithoutAnchoringEWMA() throws {
        let controller = calibrated(rows: 1)
        for index in 0 ..< 8 {
            let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            #expect(decision.depth == 1)
            let recorded = controller.recordCommittedVerification(
                decision: decision, wallTimeNanos: index == 0 ? 1_000_000_000 : 15_000_000,
                committedTokens: index == 0 ? 3 : 2, rowCount: 1)
            #expect(recorded == (index == 7))
        }
        let input = try #require(controller.snapshot().costInputs.first { $0.depth == 1 })
        #expect(input.samples == 1)
        #expect(input.totalWallTimeNanos == 105_000_000)
        #expect(input.ewmaNanosPerCommittedToken == 7_500_000)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason == "goodput")
    }

    @Test(arguments: [3, 4])
    func warmupUsesExactVerificationRowsWithinSharedBucket(firstRows: Int) throws {
        let controller = CBv2MTPDepthController(
            maxDepth: 1, fixedDepth: nil, useCommittedDecodeBaseline: true)
        let secondRows = firstRows == 3 ? 4 : 3
        // Both shapes map to bucket four. Each new physical shape must exclude
        // its cold compile, but returning to an exact warmed shape must keep
        // the entire first interval, including seed work.
        for (generation, rows) in [firstRows, secondRows, firstRows].enumerated() {
            controller.beginWorkload(rowIDs: (0 ..< rows).map {
                CBv2RequestID(UInt64(generation * 100 + $0 + 1))
            })
            controller.observeCost(decodeRowBucket: 4, depth: 0, wallTimeNanos: 12_000_000)
            for _ in 0 ..< 3 {
                controller.observeCommittedDecodeInterval(
                    decodeRowBucket: 4, wallTimeNanos: 8_700_000)
            }
            for index in 0 ..< 8 {
                let decision = controller.select(plannedDecodeRows: rows, canSpeculate: true)
                #expect(decision.depth == 1)
                let firstCost: UInt64 = generation < 2 ? 1_000_000_000 : 27_000_000
                #expect(controller.recordCommittedVerification(
                    decision: decision, wallTimeNanos: index == 0 ? firstCost : 15_000_000,
                    committedTokens: (index == 0 ? 3 : 2) * rows, rowCount: rows)
                    == (index == 7))
            }
            let input = try #require(controller.snapshot().costInputs.first { $0.depth == 1 })
            #expect(input.samples == 1)
            #expect(input.totalWallTimeNanos == (generation < 2 ? 105_000_000 : 132_000_000))
            #expect(controller.select(plannedDecodeRows: rows, canSpeculate: true).reason == "goodput")
        }
    }

    @Test func matchedWindowWeightsDoNotManufactureSeedProfit() throws {
        let controller = warmedFreshController(rows: 1)
        recordWindow(controller, costs: Array(repeating: 16_000_000, count: 8), tokens: Array(repeating: 2, count: 8))
        recordWindow(controller, costs: Array(repeating: 28_000_000, count: 8), tokens: Array(repeating: 3, count: 8))
        let input = try #require(controller.snapshot().costInputs.first { $0.depth == 1 })
        #expect(input.ewmaWallTimeNanos == 156_800_000)
        #expect(input.ewmaNanosPerCommittedToken == 8_521_739)
    }

    @Test func aSingleRejectedDraftCannotInterruptAnActiveWindow() {
        let controller = warmedFreshController(rows: 1)
        recordWindow(controller, costs: Array(repeating: 15_000_000, count: 8), tokens: Array(repeating: 2, count: 8))
        recordWindow(controller, costs: Array(repeating: 15_000_000, count: 8), tokens: Array(repeating: 2, count: 8))
        #expect(controller.activeDepthForTesting(decodeRowBucket: 1) == 1)
        let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(!controller.recordCommittedVerification(
            decision: decision, wallTimeNanos: 15_000_000, committedTokens: 1, rowCount: 1))
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason == "goodput_window")
        for index in 1 ..< 8 {
            let continuation = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            #expect(continuation.depth == 1)
            #expect(controller.recordCommittedVerification(
                decision: continuation, wallTimeNanos: 15_000_000, committedTokens: 2, rowCount: 1)
                == (index == 7))
        }
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 1)
    }

    @Test func slowWindowsBackOffOnceAndRecoveryStillProbes() {
        let controller = warmedFreshController(rows: 1)
        recordWindow(controller, costs: Array(repeating: 30_000_000, count: 8), tokens: Array(repeating: 2, count: 8))
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
        for _ in 0 ..< 7 { completeOrdinary(controller) }
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason == "explore_deeper")
        for index in 0 ..< 8 {
            let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            #expect(decision.depth == 1)
            _ = controller.recordCommittedVerification(
                decision: decision, wallTimeNanos: 30_000_000, committedTokens: 2, rowCount: 1)
            #expect(controller.probeIntervalForTesting(decodeRowBucket: 1) == (index == 7 ? 16 : 8))
        }
        var recovered = false
        for _ in 0 ..< 2_048 {
            let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            if decision.depth == 0 {
                completeOrdinary(controller)
            } else {
                _ = controller.recordCommittedVerification(
                    decision: decision, wallTimeNanos: 12_000_000, committedTokens: 2, rowCount: 1)
                if !decision.isExploration { recovered = true; break }
            }
        }
        #expect(recovered)
    }

    @Test func invalidAndCohortChangingWorkAbortsIncompleteWindows() {
        let controller = warmedFreshController(rows: 1)
        let probe = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 27_000_000, committedTokens: 3, rowCount: 1)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason == "explore_window")
        controller.cancelCommittedWindow(decodeRowBucket: 1)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason == "explore_cost")
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 27_000_000, committedTokens: 3, rowCount: 1)
        #expect(!controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 15_000_000, committedTokens: 0, rowCount: 1))
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason == "explore_cost")
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 27_000_000, committedTokens: 3, rowCount: 1)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: false).depth == 0)
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason == "explore_cost")
        _ = controller.recordCommittedVerification(
            decision: probe, wallTimeNanos: 27_000_000, committedTokens: 3, rowCount: 1)
        controller.beginWorkload(rowIDs: [CBv2RequestID(99)])
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).reason == "warmup_chained_baseline")
        #expect(controller.snapshot().costInputs.allSatisfy { $0.depth == 0 })
    }

    @Test func actualTruncatedOutputsCannotBorrowHistoricalAcceptance() {
        let controller = warmedFreshController(rows: 1)
        for _ in 0 ..< 20 {
            controller.observeAcceptance(decodeRowBucket: 1, drafted: 1, accepted: 1)
        }
        recordWindow(controller, costs: Array(repeating: 12_000_000, count: 8), tokens: Array(repeating: 1, count: 8))
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
    }

    @Test func fasterOrdinaryDecodeStillExitsWithoutFivePercentMargin() {
        let controller = warmedFreshController(rows: 1)
        recordWindow(controller, costs: Array(repeating: 16_000_000, count: 8), tokens: Array(repeating: 2, count: 8))
        recordWindow(controller, costs: Array(repeating: 16_000_000, count: 8), tokens: Array(repeating: 2, count: 8))
        #expect(controller.activeDepthForTesting(decodeRowBucket: 1) == 1)
        for _ in 0 ..< 30 {
            controller.observeCommittedDecodeInterval(decodeRowBucket: 1, wallTimeNanos: 7_800_000)
        }
        #expect(controller.select(plannedDecodeRows: 1, canSpeculate: true).depth == 0)
    }

    private func recordWindow(_ controller: CBv2MTPDepthController, costs: [UInt64], tokens: [Int]) {
        #expect(costs.count == 8 && tokens.count == 8)
        for index in costs.indices {
            let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
            #expect(decision.depth == 1)
            #expect(controller.recordCommittedVerification(
                decision: decision, wallTimeNanos: costs[index], committedTokens: tokens[index], rowCount: 1)
                == (index == 7))
        }
    }

    private func completeOrdinary(_ controller: CBv2MTPDepthController) {
        let decision = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(decision.depth == 0)
        #expect(controller.recordFinalizedStep(
            decision: decision, actualDepth: 0, wallTimeNanos: 999_000_000,
            costEligible: true, chained: true, finalizedPlainWork: true, finalizedVerification: false))
    }

    private func warmedFreshController(rows: Int) -> CBv2MTPDepthController {
        let controller = calibrated(rows: rows)
        for _ in 0 ..< 8 {
            let decision = controller.select(plannedDecodeRows: rows, canSpeculate: true)
            _ = controller.recordCommittedVerification(
                decision: decision, wallTimeNanos: 15_000_000, committedTokens: 2 * rows, rowCount: rows)
        }
        controller.beginWorkload(rowIDs: (0 ..< rows).map { CBv2RequestID(UInt64($0 + 100)) })
        for _ in 0 ..< 3 {
            controller.observeCommittedDecodeInterval(decodeRowBucket: rows, wallTimeNanos: 8_700_000)
        }
        return controller
    }

    private func calibrated(rows: Int) -> CBv2MTPDepthController {
        let controller = CBv2MTPDepthController(
            maxDepth: 1, fixedDepth: nil, useCommittedDecodeBaseline: true)
        controller.beginWorkload(rowIDs: (0 ..< rows).map { CBv2RequestID(UInt64($0 + 1)) })
        controller.observeCost(decodeRowBucket: rows, depth: 0, wallTimeNanos: 12_000_000)
        for _ in 0 ..< 3 {
            controller.observeCommittedDecodeInterval(decodeRowBucket: rows, wallTimeNanos: 8_700_000)
        }
        return controller
    }

    private func measurement(depth: Int, seed: Bool = false) -> CBv2MTPStepMeasurement {
        CBv2MTPStepMeasurement(
            decision: .init(depth: depth, decodeRowBucket: 1, reason: "test", isExploration: false),
            actualDepth: seed ? 0 : depth, costEligible: true, chained: false, seedOnly: seed)
    }
}
