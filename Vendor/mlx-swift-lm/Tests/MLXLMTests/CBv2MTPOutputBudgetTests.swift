import Testing

@testable import MLXLMCommon

@Suite("CBv2 MTP output budget")
struct CBv2MTPOutputBudgetTests {
    @Test func explorationCannotDraftIntoTheLastOutputSlot() {
        let controller = CBv2MTPDepthController(maxDepth: 4, fixedDepth: nil)
        controller.observeCost(
            decodeRowBucket: 1, depth: 0, wallTimeNanos: 100)
        let exploration = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(exploration.isExploration)
        #expect(exploration.depth == 1)

        // Exploration bypasses the request's marginal-cost policy. The common
        // output bound must still prevent its useless final-token round.
        #expect(EngineLoopV2.mtpDepthWithinOutputBudget(
            offeredDepth: exploration.depth, remainingTokens: [1]) == 0)
        #expect(EngineLoopV2.mtpDepthWithinOutputBudget(
            offeredDepth: exploration.depth, remainingTokens: [2]) == 1)
        // A request-length clamp neither trains nor resets shared controller state.
        #expect(controller.preview(plannedDecodeRows: 1, canSpeculate: true) == exploration)
    }

    @Test func fixedDepthReservesOneOutputForTheTargetToken() {
        let controller = CBv2MTPDepthController(maxDepth: 7, fixedDepth: 7)
        let fixed = controller.select(plannedDecodeRows: 1, canSpeculate: true)
        #expect(fixed.reason == "fixed")
        #expect(!fixed.isExploration)

        for (remaining, expectedDepth) in [(1, 0), (2, 1), (4, 3), (8, 7), (100, 7)] {
            #expect(EngineLoopV2.mtpDepthWithinOutputBudget(
                offeredDepth: fixed.depth, remainingTokens: [remaining]) == expectedDepth)
        }
    }

    @Test func shortestRowBoundsTheWholeRectangularCohort() {
        for budgets in [[1], [100, 1], [100, 100, 1, 100]] {
            #expect(EngineLoopV2.mtpDepthWithinOutputBudget(
                offeredDepth: 4, remainingTokens: budgets) == 0)
        }
        #expect(EngineLoopV2.mtpDepthWithinOutputBudget(
            offeredDepth: 4, remainingTokens: [100, 3, 100, 100]) == 2)
        #expect(EngineLoopV2.mtpDepthWithinOutputBudget(
            offeredDepth: 4, remainingTokens: [100, 100, 3, 100]) == 2)
    }

    @Test func budgetCannotIncreaseAnAlreadyLimitedOffer() {
        for offeredDepth in [0, 1, 2, 4, 7] {
            #expect(EngineLoopV2.mtpDepthWithinOutputBudget(
                offeredDepth: offeredDepth, remainingTokens: [100, 100]) == offeredDepth)
        }
    }

    @Test func emptyAndExhaustedBudgetsNeverSpeculate() {
        for budgets in [[], [0], [100, 0], [Int.min]] {
            #expect(EngineLoopV2.mtpDepthWithinOutputBudget(
                offeredDepth: 7, remainingTokens: budgets) == 0)
        }
        #expect(EngineLoopV2.mtpDepthWithinOutputBudget(
            offeredDepth: 7, remainingTokens: [Int.max]) == 7)
    }
}
