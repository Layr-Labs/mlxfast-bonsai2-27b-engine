import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2MTPWorkloadLifecycle")
struct CBv2MTPWorkloadLifecycleTests {
    @Test(arguments: [false, true])
    func assistantPrefillExclusionPreservesLaunchedWorkloadGeneration(includesPrefill: Bool) {
        let original = CBv2MTPStepMeasurement(
            decision: .init(depth: 1, decodeRowBucket: 2, reason: "explore_cost", isExploration: true),
            actualDepth: 1, costEligible: true, chained: true, seedOnly: false,
            workloadGeneration: 73)
        let filtered = original.excludingAssistantPrefill(includesPrefill)
        #expect(filtered.workloadGeneration == 73)
        #expect(filtered.costEligible == !includesPrefill)
        #expect(filtered.chained)
        #expect(filtered.actualDepth == original.actualDepth)
        #expect(filtered.decision.depth == original.decision.depth)
        #expect(filtered.decision.decodeRowBucket == original.decision.decodeRowBucket)
        #expect(filtered.seedOnly == original.seedOnly)
        #expect(original.excludingAssistantPrefill(true)
            .excludingAssistantPrefill(false).costEligible == false)
    }

    @Test(arguments: [1, 2, 4])
    func reusedNumericCohortRequiresFreshChainedBaseline(width: Int) throws {
        let driver = try makeDriver()
        let rows = (0 ..< width).map { CBv2RequestID(UInt64($0 + 1)) }
        calibrate(driver, rows: rows)
        for index in 0 ..< 8 {
            let decision = begin(driver, rows: rows)
            #expect(decision.depth == 1)
            record(driver, decision: decision, rows: rows, depth: 1,
                   completedAt: 2_000_000_000 + UInt64(index) * 15_000_000)
        }
        #expect(begin(driver, rows: rows).reason == "goodput")
        #expect(driver.metricsSnapshot().costInputs.contains { $0.depth == 1 })

        // The same numeric membership represents a new request lifetime.
        // Finishing even one batch member invalidates the whole shared cost.
        driver.requestDidFinish(rows[0])
        #expect(begin(driver, rows: rows).reason == "warmup_chained_baseline")
        #expect(driver.metricsSnapshot().costInputs.allSatisfy { $0.depth == 0 })
        #expect(driver.activeDepthForTesting(decodeRowBucket: width) == 0)
        #expect(driver.probeIntervalForTesting(decodeRowBucket: width) == 8)
        let decision = driver.controllerDecision
        for index in 0 ..< 4 {
            baseline(driver, decision: decision, rows: rows,
                     completedAt: 60_000_000_000 + UInt64(index) * 8_000_000)
            let next = begin(driver, rows: rows)
            #expect(index == 3 ? next.depth == 1 : next.reason == "warmup_chained_baseline")
        }
        let cost = try #require(driver.metricsSnapshot().costInputs.first { $0.depth == 0 })
        #expect(cost.samples == 3)
        #expect(cost.totalWallTimeNanos == 24_000_000)
    }

    @Test func finishDuringInflightSuppressesLateSeedAndVerificationLearning() throws {
        let driver = try makeDriver()
        let rows = [CBv2RequestID(1)]
        calibrate(driver, rows: rows)
        let decision = begin(driver, rows: rows)
        record(driver, decision: decision, rows: rows, depth: 0,
               seed: true, completedAt: 2_000_000_000)
        #expect(driver.pendingSeedCostCountForTesting == 1)
        driver.requestDidFinish(rows[0])
        #expect(driver.pendingSeedCostCountForTesting == 0)

        // Engine finalization can retire a row before delivering the old
        // step's host-only measurements. Those cannot revive its workload.
        record(driver, decision: decision, rows: rows, depth: 0,
               seed: true, completedAt: 2_015_000_000)
        #expect(driver.pendingSeedCostCountForTesting == 0)
        for index in 0 ..< 8 {
            driver.recordStepAcceptance(drafted: 1, accepted: 0,
                                       observedDrafts: 1, decodeRowBucket: 1)
            record(driver, decision: decision, rows: rows, depth: 1,
                   completedAt: 2_030_000_000 + UInt64(index) * 15_000_000)
        }
        #expect(driver.metricsSnapshot().costInputs.allSatisfy { $0.depth == 0 })
        #expect(driver.metricsSnapshot().conditionalAcceptance.isEmpty)
        // Real executed work remains visible even though it cannot train.
        #expect(driver.metricsSnapshot().totalRoundWallTimeNanos == 120_000_000)
        #expect(begin(driver, rows: rows).reason == "warmup_chained_baseline")
    }

    @Test func staleChainedSuccessorCannotTrainReusedIDOrClaimItsSeed() throws {
        let driver = try makeDriver()
        let rows = [CBv2RequestID(1)]
        calibrate(driver, rows: rows)
        let oldDecision = begin(driver, rows: rows)
        let oldGeneration = driver.workloadGeneration
        let oldMeasurement = CBv2MTPStepMeasurement(
            decision: oldDecision, actualDepth: 1, costEligible: true,
            chained: false, seedOnly: false, workloadGeneration: oldGeneration)
        driver.requestDidFinish(rows[0])
        let fresh = begin(driver, rows: rows)
        #expect(driver.workloadGeneration != oldGeneration)
        #expect(fresh.reason == "warmup_chained_baseline")

        // A successor was already launched before the finish. Its callbacks
        // arrive after the replacement lifetime's beginPlan enabled learning.
        for index in 0 ..< 10 {
            baseline(driver, decision: fresh, rows: rows,
                     completedAt: 3_000_000_000 + UInt64(index) * 8_000_000,
                     generation: oldGeneration)
            driver.recordStepAcceptance(drafted: 1, accepted: 0, observedDrafts: 1,
                                        decodeRowBucket: 1, measurement: oldMeasurement)
            record(driver, decision: oldDecision, rows: rows, depth: 1,
                   completedAt: 4_000_000_000 + UInt64(index) * 15_000_000,
                   generation: oldGeneration)
        }
        record(driver, decision: oldDecision, rows: rows, depth: 0, seed: true,
               completedAt: 5_000_000_000, generation: oldGeneration)
        #expect(driver.pendingSeedCostCountForTesting == 0)
        #expect(driver.metricsSnapshot().conditionalAcceptance.isEmpty)
        #expect(driver.metricsSnapshot().costInputs.allSatisfy { $0.depth == 0 })
        #expect(begin(driver, rows: rows).reason == "warmup_chained_baseline")
        for index in 0 ..< 4 {
            baseline(driver, decision: fresh, rows: rows,
                     completedAt: 60_000_000_000 + UInt64(index) * 8_000_000)
            let next = begin(driver, rows: rows)
            #expect(index == 3 ? next.depth == 1 : next.reason == "warmup_chained_baseline")
        }
        let probe = driver.controllerDecision
        record(driver, decision: probe, rows: rows, depth: 0, seed: true,
               completedAt: 61_000_000_000)
        #expect(driver.pendingSeedCostCountForTesting == 1)
        #expect(driver.claimPendingSeedCost(decodeRowBucket: 1, finalizedVerifyIDs: Set(rows),
                                           measurement: oldMeasurement) == 0)
        #expect(driver.pendingSeedCostCountForTesting == 1)
        #expect(driver.claimPendingSeedCost(decodeRowBucket: 1, finalizedVerifyIDs: Set(rows))
                == 15_000_000)
    }

    @Test func drainClearsWorkloadClocksForSameIDReuse() throws {
        let driver = try makeDriver()
        let rows = [CBv2RequestID(1)]
        calibrate(driver, rows: rows)
        let oldGeneration = driver.workloadGeneration
        #expect(begin(driver, rows: rows).depth == 1)
        driver.removeAllRequestState()
        #expect(driver.workloadGeneration != oldGeneration)
        #expect(begin(driver, rows: rows).reason == "warmup_chained_baseline")
        #expect(driver.requestStateCountForTesting == 0)
    }

    @Test func unrelatedFinishDoesNotAbortCurrentObservationWindow() throws {
        let driver = try makeDriver()
        let rows = [CBv2RequestID(1)]
        calibrate(driver, rows: rows)
        let decision = begin(driver, rows: rows)
        record(driver, decision: decision, rows: rows, depth: 1, completedAt: 2_000_000_000)
        #expect(begin(driver, rows: rows).reason == "explore_window")
        driver.requestDidFinish(CBv2RequestID(99))
        #expect(begin(driver, rows: rows).reason == "explore_window")
    }

    @Test func fixedDepthRetainsCostAccountingAfterFinish() throws {
        let driver = try makeDriver(fixedDepth: 1)
        let rows = [CBv2RequestID(1)]
        let decision = begin(driver, rows: rows)
        driver.requestDidFinish(rows[0])
        record(driver, decision: decision, rows: rows, depth: 1, completedAt: 2_000_000_000)
        #expect(driver.metricsSnapshot().costInputs.contains { $0.depth == 1 })
        #expect(begin(driver, rows: rows).depth == 1)
    }

    private func makeDriver(fixedDepth: Int? = nil) throws -> CBv2MTPRoundDriver {
        let model = LifecycleModel()
        return try #require(CBv2MTPRoundDriver.build(
            model: model, drafter: LifecycleDrafter(target: model),
            config: CBv2MTPConfig(enabled: true, maxDraftTokens: 1,
                                  maxSpeculativeBatch: 4, fixedDraftTokens: fixedDepth,
                                  maxAutomaticRectangularTokens: 8)))
    }

    private func begin(_ driver: CBv2MTPRoundDriver, rows: [CBv2RequestID]) -> CBv2MTPDepthDecision {
        driver.beginPlan(plannedDecodeRows: rows.count, canSpeculate: true, rowIDs: rows)
        return driver.controllerDecision
    }

    private func calibrate(_ driver: CBv2MTPRoundDriver, rows: [CBv2RequestID]) {
        let initial = begin(driver, rows: rows)
        record(driver, decision: initial, rows: rows, depth: 0, completedAt: 100_000_000)
        let decision = begin(driver, rows: rows)
        #expect(decision.reason == "warmup_chained_baseline")
        for index in 0 ..< 4 {
            baseline(driver, decision: decision, rows: rows,
                     completedAt: 1_000_000_000 + UInt64(index) * 8_000_000)
        }
    }

    private func baseline(_ driver: CBv2MTPRoundDriver, decision: CBv2MTPDepthDecision,
                          rows: [CBv2RequestID], completedAt: UInt64,
                          generation: UInt64? = nil) {
        driver.recordCommittedDecodeBaseline(
            measurement: .init(decision: decision, actualDepth: 0, costEligible: true,
                               chained: true, seedOnly: false,
                               workloadGeneration: generation ?? driver.workloadGeneration),
            completedAtNanos: completedAt, sampledRows: rows,
            finalizedPlainRowCount: rows.count, hasChainedSuccessor: true)
    }

    private func record(_ driver: CBv2MTPRoundDriver, decision: CBv2MTPDepthDecision,
                        rows: [CBv2RequestID], depth: Int, seed: Bool = false,
                        completedAt: UInt64, generation: UInt64? = nil) {
        driver.recordStepCost(
            .init(decision: decision, actualDepth: depth, costEligible: true,
                  chained: false, seedOnly: seed,
                  workloadGeneration: generation ?? driver.workloadGeneration),
            wallTimeNanos: 15_000_000, finalizedPlainWork: depth == 0,
            finalizedSeedIDs: seed ? Set(rows) : [], finalizedVerification: depth > 0,
            claimedSeedCostNanos: 0, completedAtNanos: completedAt,
            committedRows: rows, committedTokenCount: rows.count * (depth + 1))
    }
}

private final class LifecycleCapture: CBv2MTPPreparedCapture {}
private final class LifecycleDrafter: CBv2MTPDrafter {
    let mtpTargetIdentity: ObjectIdentifier?
    let supportsTargetPrefixAcceptance = true
    init(target: LifecycleModel) { mtpTargetIdentity = ObjectIdentifier(target) }
    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { LifecycleCapture() }
    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray) { (tokens, hidden) }
}
private final class LifecycleModel: CBv2MTPSteppableModel {
    let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 0)
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        fatalError("lifecycle tests do not execute model graphs")
    }
    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray) {
        fatalError("lifecycle tests do not execute model graphs")
    }
}
