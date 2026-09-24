import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// The MTP capture-verify window must be the serial path, position for
/// position: same logits at every window position, and the committed
/// recurrent state after `keepPositions` equal to the state the same tokens
/// fed one at a time would leave.
final class Qwen4ExpCapturedVerifyTests: XCTestCase {

    private func requireCompleteMetallib() throws {
        let raw = ProcessInfo.processInfo.environment[Qwen4ExpForwardParityTests.optInVariable]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard ["1", "true", "yes", "on"].contains(raw ?? "") else {
            throw XCTSkip("Set \(Qwen4ExpForwardParityTests.optInVariable)=1 (needs a complete metallib)")
        }
    }

    /// Committed recurrent state per state layer, read back through a fresh
    /// binding's input view.
    private func committedStates(_ session: Qwen4ExpCBv2Session) throws -> [Int: CBv2RecurrentLayerState] {
        let evaluation = try session.recurrent.bind()
        defer { try? evaluation.evaluate() }
        var out: [Int: CBv2RecurrentLayerState] = [:]
        for layer in session.model.cbv2RecurrentStateSpec.layers {
            if let state = evaluation.inputState(modelLayerIndex: layer.modelLayerIndex) {
                out[layer.modelLayerIndex] = state
            }
        }
        return out
    }

    private func assertSame(_ a: MLXArray?, _ b: MLXArray?, _ label: String) {
        guard let a, let b else {
            XCTAssertNil(a, label); XCTAssertNil(b, label); return
        }
        XCTAssertEqual(a.shape, b.shape, label)
        XCTAssertTrue(
            allClose(a.asType(.float32), b.asType(.float32), rtol: 0, atol: 0).item(Bool.self),
            "\(label): max |delta| \(MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self))")
    }

    func testCapturedWindowEqualsSequentialAtEveryPosition() throws {
        try requireCompleteMetallib()
        let prompt = [3, 9, 14, 2, 21, 6, 11, 30, 4, 17, 25, 8]
        let window = [7, 12, 5]

        // Sequential: the same tokens one at a time, states after each.
        let sequentialModel = try Qwen4ExpFixture.model()
        let sequential = try Qwen4ExpCBv2Session(model: sequentialModel)
        _ = try sequential.forward(prompt)
        var sequentialLogits: [MLXArray] = []
        var sequentialStates: [[Int: CBv2RecurrentLayerState]] = []
        for token in window {
            sequentialLogits.append(try sequential.forward([token]).logits)
            sequentialStates.append(try committedStates(sequential))
        }

        for keep in 1 ... window.count {
            let model = try Qwen4ExpFixture.model()
            let session = try Qwen4ExpCBv2Session(model: model)
            _ = try session.forward(prompt)

            let ids = MLXArray(window.map { Int32($0) }).reshaped([1, window.count])
            let evaluation = try session.recurrent.bind()
            for cache in session.caches { cache.mtpSerializesRectangularAttention = true }
            let out = model.cbv2ForwardWithHiddenCaptured(
                ids, caches: session.kvCaches, recurrentState: [evaluation], positionIds: nil)
            for cache in session.caches { cache.mtpSerializesRectangularAttention = false }
            eval(out.logits, out.lastHidden)
            XCTAssertTrue(evaluation.isCaptured)
            try evaluation.evaluate()
            try evaluation.commit(keepPositions: keep)

            for position in 0 ..< window.count {
                assertSame(
                    out.logits[0..., position, 0...], sequentialLogits[position],
                    "logits at window position \(position) (keep \(keep))")
            }
            let committed = try committedStates(session)
            let expected = sequentialStates[keep - 1]
            XCTAssertEqual(Set(committed.keys), Set(expected.keys))
            for (layer, state) in expected {
                assertSame(committed[layer]?.conv, state.conv, "conv state layer \(layer) after keep \(keep)")
                assertSame(committed[layer]?.ssm, state.ssm, "ssm state layer \(layer) after keep \(keep)")
            }
        }
    }
}
