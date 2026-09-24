import MLX
import XCTest
@testable import MLXLMCommon

final class CBv2CapturedStateAccountingTests: XCTestCase {
    private var spec: CBv2RecurrentStateSpec {
        .init(layers: [.init(modelLayerIndex: 0,
            convShape: [1, 2, 3], convDType: .float32,
            ssmShape: [1, 1, 2, 2], ssmDType: .float32)])
    }

    private func captured(_ state: CBv2RecurrentRequestState, positions: Int) throws
        -> CBv2RecurrentStateEvaluation {
        let evaluation = try state.bind()
        try evaluation.stageCaptured(modelLayerIndex: 0,
            conv: MLXArray(0..<(positions * 6)).asType(.float32).reshaped(positions, 2, 3),
            ssm: MLXArray(0..<(positions * 4)).asType(.float32).reshaped(positions, 1, 2, 2),
            positions: positions)
        eval(try evaluation.evaluate())
        return evaluation
    }

    private func plain(_ state: CBv2RecurrentRequestState) throws -> CBv2RecurrentStateEvaluation {
        let evaluation = try state.bind()
        try evaluation.stage(modelLayerIndex: 0,
            conv: MLXArray(0..<6).asType(.float32).reshaped(1, 2, 3),
            ssm: MLXArray(0..<4).asType(.float32).reshaped(1, 1, 2, 2))
        eval(try evaluation.evaluate())
        return evaluation
    }

    func testCommittedPrefixCountsWholeBackingThroughRollbackAndReplacement() throws {
        for width in 2...4 {
            for keep in 1...width {
                let state = try CBv2RecurrentRequestState(spec: spec)
                let bytes = state.byteCount
                let first = try captured(state, positions: width)
                XCTAssertEqual(state.materializedByteCount, width * bytes)
                try first.commit(keepPositions: keep)
                XCTAssertEqual(state.materializedByteCount, width * bytes,
                    "A prefix slice retains all \(width) captured positions, including keep=\(keep)")
                let saved = try XCTUnwrap(state.state(modelLayerIndex: 0)?.ssm)
                XCTAssertEqual(saved[0, 0, 0, 0].item(Float.self), Float((keep - 1) * 4))

                let rejected = try captured(state, positions: 2)
                XCTAssertEqual(state.materializedByteCount, (width + 2) * bytes)
                try rejected.rollback()
                XCTAssertEqual(state.materializedByteCount, width * bytes,
                    "Rollback must retain the prior committed stack obligation")
                XCTAssertEqual(try XCTUnwrap(state.state(modelLayerIndex: 0)?.ssm).asData(access: .copy).data,
                               saved.asData(access: .copy).data)

                let replacement = try plain(state)
                XCTAssertEqual(state.materializedByteCount, (width + 1) * bytes)
                try replacement.commit()
                XCTAssertEqual(state.materializedByteCount, bytes,
                    "A later plain committed generation releases the captured obligation")
                try state.release()
                XCTAssertEqual(state.materializedByteCount, 0)
            }
        }
    }

    func testNewCapturedCommitReplacesRatherThanAccumulatesBackingObligation() throws {
        let state = try CBv2RecurrentRequestState(spec: spec)
        let first = try captured(state, positions: 4)
        try first.commit(keepPositions: 1)
        let next = try captured(state, positions: 2)
        XCTAssertEqual(state.materializedByteCount, 6 * state.byteCount)
        try next.commit(keepPositions: 2)
        XCTAssertEqual(state.materializedByteCount, 2 * state.byteCount)
        try state.release()
        XCTAssertEqual(state.materializedByteCount, 0)
    }
}
