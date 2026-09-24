import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpForwardParityTests: XCTestCase {

    /// Opt-in switch for the three tests in this file.
    static let optInVariable = "MLXLM_FULL_AOT_METALLIB"

    /// Skip unless the process runs against a COMPLETE metallib.
    ///
    /// WHY THESE THREE NEED ONE. Each drives a single-token decode, and a
    /// decode through the mixture-of-experts shared expert gate is
    /// `Linear(hidden, 1)`. MLX lowers that to its `dot_product`
    /// specialization (`matmul.cpp`, guard `M == 1 && N == 1 &&
    /// batch_size_out == 1`). mlx-swift's `Cmlx` target does not compile the
    /// Metal kernels through SwiftPM -- `Package.swift` excludes the kernels
    /// directory -- so the library the ordinary `xcodebuild` build ships
    /// carries neither `dot_product` nor `gemm` (459 symbols against 17322 in
    /// a complete build). A missing kernel does not throw: MLX aborts the
    /// PROCESS, which would take this whole bundle down rather than fail one
    /// test.
    ///
    /// So the gate is explicit and off by default. It is NOT a statement
    /// about the code under test, which passes; it is a statement about the
    /// library the process was linked against.
    private func requireCompleteMetallib() throws {
        let raw = ProcessInfo.processInfo.environment[Self.optInVariable]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard ["1", "true", "yes", "on"].contains(raw ?? "") else {
            throw XCTSkip(
                "Set \(Self.optInVariable)=1 to run this test. It needs a full "
                    + "ahead-of-time mlx.metallib carrying the dot_product kernels. "
                    + "The metallib the standard xcodebuild build produces does not "
                    + "have them, and MLX aborts the process instead of failing the "
                    + "test. Build one with cmake -DMLX_METAL_JIT=OFF --target "
                    + "mlx-metallib, the way d-inference scripts/fetch-metallib.sh "
                    + "does. Then copy it over every default.metallib under "
                    + "Build/Products/Debug, including the bundle copies inside each "
                    + ".xctest. CI has no such step yet, which is why this is opt-in.")
        }
    }

    // MARK: - (b) CBv2 forward is token-exact against the legacy path

    /// Prompt short enough that the visible context stays inside the indexer
    /// budget: the keep mask never fires and attention is plain causal.
    func testCBv2MatchesLegacyBelowTheIndexerBudget() throws {
        try requireCompleteMetallib()
        try assertCBv2MatchesLegacy(prompt: [3, 9, 14, 2], decodeSteps: 3)
    }

    /// Prompt long enough that the tape passes the budget during prefill, so
    /// every full-attention layer runs with a keep mask.
    func testCBv2MatchesLegacyAboveTheIndexerBudget() throws {
        try requireCompleteMetallib()
        try assertCBv2MatchesLegacy(
            prompt: [3, 9, 14, 2, 21, 6, 11, 30, 4, 17, 25, 8], decodeSteps: 4)
    }

    private func assertCBv2MatchesLegacy(prompt: [Int], decodeSteps: Int) throws {
        let model = try Qwen4ExpFixture.model(withMTP: false)
        let expected = qwen4ExpLegacyGreedy(
            model: model, prompt: prompt, count: decodeSteps + 1)

        let session = try Qwen4ExpCBv2Session(model: model)
        var produced: [Int] = []
        var next = session.greedy(try session.forward(prompt).logits)
        produced.append(next)
        for _ in 0 ..< decodeSteps {
            next = session.greedy(try session.forward([next]).logits)
            produced.append(next)
        }

        XCTAssertEqual(produced, expected)
        // The mask must really have fired on the long prompt, or this test
        // would be asserting the dense path twice.
        let tapeLength = session.caches[0].indexerTapeLength
        XCTAssertEqual(tapeLength, prompt.count + decodeSteps)
        if prompt.count > Qwen4ExpFixture.indexerBudget {
            XCTAssertGreaterThan(tapeLength, Qwen4ExpFixture.indexerBudget)
        }
    }

    // MARK: - (c) MTP draft and verify are lossless at depths 1...3

    func testMTPRoundsAreLosslessAtEveryPermittedDepth() throws {
        try requireCompleteMetallib()
        let prompt = [3, 9, 14, 2, 21, 6, 11, 30, 4]
        let steps = 6

        let baselineModel = try Qwen4ExpFixture.model()
        let baselineSession = try Qwen4ExpCBv2Session(model: baselineModel)
        var baseline: [Int] = []
        var token = baselineSession.greedy(try baselineSession.forward(prompt).logits)
        baseline.append(token)
        for _ in 1 ..< steps {
            token = baselineSession.greedy(try baselineSession.forward([token]).logits)
            baseline.append(token)
        }

        for depth in 1 ... Qwen4ExpInlineMTPAssistant.maximumDepth {
            let produced = try speculativeGreedy(prompt: prompt, steps: steps, depth: depth)
            XCTAssertEqual(
                produced, baseline,
                "MTP output diverged from serial output at depth \(depth)")
        }
    }

    /// Greedy generation driven by the embedded head, with the accept walk the
    /// engine's serial-target verification performs: chain `depth` drafts,
    /// then feed the window one column at a time and stop at the first
    /// divergence. A rejected draft is never fed, so the committed history is
    /// exactly the history serial decode would have written.
    private func speculativeGreedy(prompt: [Int], steps: Int, depth: Int) throws -> [Int] {
        let model = try Qwen4ExpFixture.model()
        let session = try Qwen4ExpCBv2Session(model: model)
        let drafter = try XCTUnwrap(Qwen4ExpInlineMTPAssistant(target: model))
        XCTAssertEqual(drafter.maximumDraftTokens, Qwen4ExpInlineMTPAssistant.maximumDepth)
        XCTAssertEqual(drafter.maximumSpeculativeBatch, 1)
        XCTAssertNil(drafter.requiredVerificationMode, "the target verifies its window in one forward")
        XCTAssertEqual(drafter.mtpTargetIdentity, ObjectIdentifier(model))

        let requestState = drafter.makeRequestState()
        defer { drafter.releaseRequestState(requestState) }

        // Prefill, then hand the head the prompt's trusted transitions.
        let prefill = try session.forward(prompt)
        drafter.observeCommittedTarget(
            CBv2MTPCommittedTargetObservation(
                tokens: MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count]),
                hidden: prefill.multi),
            requestState: requestState)

        var produced: [Int] = []
        var carryToken = session.greedy(prefill.logits)
        var carryMulti = prefill.multi[0..., -1 ..< prefill.multi.dim(1), 0...]
        produced.append(carryToken)

        while produced.count < steps {
            // Chain the draft.
            var proposals: [Int] = []
            var chainTokens = MLXArray([Int32(carryToken)]).reshaped([1, 1])
            var chainMulti = carryMulti
            for _ in 0 ..< depth {
                let draft = drafter.draftStep(
                    tokens: chainTokens, hidden: chainMulti, shortlist: nil,
                    requestState: requestState)
                eval(draft.tokens, draft.hidden)
                proposals.append(draft.tokens.item(Int.self))
                chainTokens = draft.tokens.reshaped([1, 1])
                chainMulti = draft.hidden
            }

            // Verify column by column against the target itself.
            var accepted = 0
            var conditioningMulti: [MLXArray] = []
            var feed = carryToken
            var emitted = 0
            while true {
                let out = try session.forward([feed])
                conditioningMulti.append(out.multi[0..., -1 ..< out.multi.dim(1), 0...])
                let argmax = session.greedy(out.logits)
                carryMulti = conditioningMulti[conditioningMulti.count - 1]
                produced.append(argmax)
                emitted += 1
                carryToken = argmax
                guard accepted < proposals.count, argmax == proposals[accepted],
                    produced.count < steps
                else { break }
                accepted += 1
                feed = argmax
            }
            XCTAssertGreaterThan(emitted, 0)

            // Close the round on the head: the accepted proposals became
            // canonical target history, each paired with the multi stream
            // that conditioned it.
            let committedTokens =
                accepted > 0
                ? MLXArray(proposals[0 ..< accepted].map { Int32($0) }).reshaped([1, accepted])
                : MLXArray([Int32]()).reshaped([1, 0])
            let committedHidden =
                accepted > 0
                ? concatenated(Array(conditioningMulti[0 ..< accepted]), axis: 1)
                : MLXArray.zeros([1, 0, carryMulti.dim(2)], dtype: carryMulti.dtype)
            drafter.finalizeRound(
                requestState: requestState,
                confirmedInputTokens: accepted + 1,
                committedDraftTokens: committedTokens,
                committedTargetHidden: committedHidden)
        }
        return Array(produced.prefix(steps))
    }
}
