// Weight-free tests for the CBv2 BLOCK drafter seam.
//
// Nothing here loads a model or a drafter. Three things are pinned:
//
//   1. THE DEPTH CEILING. A block decoder declares depths up to 16 and the
//      chain decoder's 7 does not move. The ceiling is carried by the
//      configuration and honoured by the depth controller, so a depth of 8
//      to 16 can never be clamped to 7 without anyone noticing.
//   2. THE CONFIRMED-COLUMN SLICE. A round hands the block drafter the tapped
//      context of exactly the columns it confirmed, which is one column wider
//      than the chain drafter's accepted-draft slice.
//   3. THE BLOCK SEAM ITSELF. A minimal drafter conforming to
//      `CBv2MTPBlockDrafter` exercises the queue arithmetic a round depends
//      on — one propose per round, the context the propose consumes, and the
//      committed length the cache is trimmed to — driven by the SAME
//      expressions the engine uses.
//
// What is NOT covered: the engine loop itself. Driving `EngineLoopV2` needs a
// recurrent target fixture and a real forward, so the round sequence below
// replays the engine's arithmetic rather than executing it.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2MTPBlockDrafter")
struct CBv2MTPBlockDrafterTests {

    // MARK: - The depth ceiling

    @Test func chainDecoderKeepsTheTestedSevenTokenCeiling() {
        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 16, maxSpeculativeBatch: 1,
            fixedDraftTokens: 16)
        #expect(config.draftTokenCeiling == CBv2MTPConfig.testedMaxDraftTokens)
        #expect(config.maxDraftTokens == 7)
        #expect(config.fixedDraftTokens == 7)
    }

    @Test func blockDecoderReachesSixteen() {
        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 16, maxSpeculativeBatch: 1,
            fixedDraftTokens: 16,
            draftTokenCeiling: CBv2MTPConfig.testedMaxBlockDraftTokens)
        #expect(config.maxDraftTokens == 16)
        #expect(config.fixedDraftTokens == 16)
    }

    @Test func blockCeilingStillClampsAboveItself() {
        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 64, maxSpeculativeBatch: 1,
            fixedDraftTokens: 64,
            draftTokenCeiling: CBv2MTPConfig.testedMaxBlockDraftTokens)
        #expect(config.maxDraftTokens == 16)
        #expect(config.fixedDraftTokens == 16)
    }

    @Test func depthControllerHonoursTheBlockCeiling() {
        let block = CBv2MTPDepthController(
            maxDepth: 16, fixedDepth: 16,
            ceiling: CBv2MTPConfig.testedMaxBlockDraftTokens)
        #expect(block.maxDepth == 16)
        #expect(
            block.select(plannedDecodeRows: 1, canSpeculate: true).depth == 16)

        let chain = CBv2MTPDepthController(maxDepth: 16, fixedDepth: 16)
        #expect(chain.maxDepth == CBv2MTPConfig.testedMaxDraftTokens)
        #expect(
            chain.select(plannedDecodeRows: 1, canSpeculate: true).depth == 7)
    }

    // MARK: - The confirmed-column slice

    @Test func confirmedColumnsCoverTheCorrectionToken() {
        // A depth-7 round verifies 8 columns. Accepting 3 drafts confirms 4
        // tokens, so the next block's context is 4 rows — the 3 accepted
        // drafts AND the correction the target emitted at the divergence.
        #expect(
            CBv2MTPBlockContextIndex.confirmedColumns(confirmed: 4, verifyWidth: 8)
                == 0 ..< 4)
        // Full acceptance keeps the whole window.
        #expect(
            CBv2MTPBlockContextIndex.confirmedColumns(confirmed: 8, verifyWidth: 8)
                == 0 ..< 8)
        // A round that accepted nothing still confirms the target's own token.
        #expect(
            CBv2MTPBlockContextIndex.confirmedColumns(confirmed: 1, verifyWidth: 8)
                == 0 ..< 1)
    }

    // MARK: - The block seam

    /// One prompt, one seed step and two rounds, replaying the engine's own
    /// expressions: the prefill and seed rows arrive through
    /// `observeCommittedTarget`, a round proposes once and trims to the
    /// carry's committed length, and the round's confirmed columns arrive
    /// through `finalizeRound`.
    @Test func oneProposePerRoundConsumesTheConfirmedColumns() throws {
        let taps = 5
        let hidden = 8
        let promptLength = 6
        let drafter = BlockSeamTestDrafter(contextRowLimit: 4)
        let state = drafter.makeRequestState()

        func context(_ rows: Int) -> MLXArray {
            MLXArray.zeros([1, rows, taps * hidden])
        }

        // Prefill: one chunk of the whole prompt. Only the last four rows
        // survive the drafter's context window.
        drafter.observeCommittedTarget(
            CBv2MTPCommittedTargetObservation(
                tokens: MLXArray.zeros([1, promptLength], dtype: .int32),
                hidden: context(promptLength)),
            requestState: state)
        // The seed step: one plain decode position, at the prompt's end.
        drafter.observeCommittedTarget(
            CBv2MTPCommittedTargetObservation(
                tokens: MLXArray.zeros([1, 1], dtype: .int32),
                hidden: context(1)),
            requestState: state)
        #expect(state.committedInputCount == promptLength + 1)
        #expect(state.stagedInputCount == 0)

        // Round one at depth 7, from a carry whose committed length is the
        // seed's `numComputedTokens` == 7.
        _ = try drafter.proposeBlock(anchor: 11, depth: 7, requestState: state)
        drafter.trimBlockState(state, toCommittedLength: promptLength + 1)
        #expect(drafter.proposals == [BlockSeamProposal(anchor: 11, depth: 7, contextRows: 4)])
        #expect(drafter.trims == [promptLength + 1])

        // Three drafts accepted plus the correction: four confirmed columns.
        drafter.finalizeRound(
            requestState: state, confirmedInputTokens: 4,
            committedDraftTokens: MLXArray.zeros([1, 3], dtype: .int32),
            committedTargetHidden: context(4))
        #expect(state.committedInputCount == promptLength + 1 + 4)

        // Round two: the context is exactly those four confirmed rows, and
        // the committed length has advanced by them.
        _ = try drafter.proposeBlock(anchor: 12, depth: 7, requestState: state)
        drafter.trimBlockState(state, toCommittedLength: promptLength + 1 + 4)
        #expect(drafter.proposals.count == 2)
        #expect(drafter.proposals[1] == BlockSeamProposal(anchor: 12, depth: 7, contextRows: 4))
        #expect(drafter.trims == [7, 11])
    }

    @Test func aRoundWithNoContextRefuses() throws {
        let drafter = BlockSeamTestDrafter(contextRowLimit: nil)
        let state = drafter.makeRequestState()
        #expect(throws: BlockSeamTestDrafter.Failure.noContext) {
            _ = try drafter.proposeBlock(anchor: 1, depth: 4, requestState: state)
        }
    }
}

// MARK: - Fixtures

private struct BlockSeamProposal: Equatable {
    let anchor: Int
    let depth: Int
    let contextRows: Int
}

/// The smallest thing that is a block drafter. It keeps the same context
/// queue the real adapter keeps and records what each round asked of it.
private final class BlockSeamTestDrafter: CBv2MTPBlockDrafter {
    enum Failure: Error, Equatable { case noContext }

    final class State: CBv2MTPRequestState {
        var pending: [MLXArray] = []
        var pendingRows = 0
        var observedRows = 0
        var committedInputCount: Int { observedRows }
        var stagedInputCount: Int { 0 }
    }

    private let contextRowLimit: Int?
    private(set) var proposals: [BlockSeamProposal] = []
    private(set) var trims: [Int] = []
    private(set) var armed = false

    init(contextRowLimit: Int?) { self.contextRowLimit = contextRowLimit }

    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }
    var maximumDraftTokens: Int? { 16 }
    var maximumSpeculativeBatch: Int? { 1 }

    func setBlockContextArmed(_ armed: Bool) throws { self.armed = armed }
    func blockContextHidden() -> MLXArray? { nil }

    func makeRequestState() -> any CBv2MTPRequestState { State() }
    func releaseRequestState(_ requestState: any CBv2MTPRequestState) {}
    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] { [] }
    func discardRound(requestState: any CBv2MTPRequestState) {}

    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        append(observation.hidden, to: requestState)
    }

    func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        append(committedTargetHidden, to: requestState)
    }

    func proposeBlock(
        anchor: Int, depth: Int, requestState: any CBv2MTPRequestState
    ) throws -> MLXArray {
        let state = requestState as! State
        guard state.pendingRows > 0 else { throw Failure.noContext }
        proposals.append(
            BlockSeamProposal(
                anchor: anchor, depth: depth, contextRows: state.pendingRows))
        state.pending.removeAll(keepingCapacity: true)
        state.pendingRows = 0
        return MLXArray.zeros([1, depth], dtype: .int32)
    }

    func trimBlockState(
        _ requestState: any CBv2MTPRequestState, toCommittedLength committed: Int
    ) {
        trims.append(committed)
    }

    private func append(_ hidden: MLXArray, to requestState: any CBv2MTPRequestState) {
        let state = requestState as! State
        let rows = hidden.dim(1)
        guard rows > 0 else { return }
        state.pending.append(hidden)
        state.pendingRows += rows
        state.observedRows += rows
        guard let contextRowLimit, state.pendingRows > contextRowLimit else { return }
        while let first = state.pending.first,
            state.pendingRows - first.dim(1) >= contextRowLimit
        {
            state.pending.removeFirst()
            state.pendingRows -= first.dim(1)
        }
        if state.pendingRows > contextRowLimit, let first = state.pending.first {
            let drop = state.pendingRows - contextRowLimit
            state.pending[0] = first[0..., drop..., 0...]
            state.pendingRows -= drop
        }
    }
}
