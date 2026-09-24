import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class Qwen35PositionCBv2Tests: XCTestCase {
    private func values(_ array: MLXArray) -> [Int32] {
        array.asType(.int32).asArray(Int32.self)
    }

    func testPromptChunkSlicesPreserveAllThreeAxes() {
        let prompt = MLXArray([
            Int32(0), 1, 2, 3, 4, 5,
            10, 11, 12, 13, 14, 15,
            20, 21, 22, 23, 24, 25,
        ]).reshaped([3, 1, 6])
        let state = CBv2PositionState(promptPositionIds: prompt, decodeDeltas: [-2])

        let before = state.promptSlice(0 ..< 2)
        let media = state.promptSlice(2 ..< 5)
        let after = state.promptSlice(5 ..< 6)
        XCTAssertEqual(before.shape, [3, 1, 2])
        XCTAssertEqual(values(before), [0, 1, 10, 11, 20, 21])
        XCTAssertEqual(values(media), [2, 3, 4, 12, 13, 14, 22, 23, 24])
        XCTAssertEqual(values(after), [5, 15, 25])
    }

    func testRectangularDecodeUsesEachRowsDeltaAndHistory() {
        let prompt = MLXArray.zeros([3, 1, 1], dtype: .int32)
        let media = CBv2PositionState(promptPositionIds: prompt, decodeDeltas: [-2])
        let text = CBv2PositionState(promptPositionIds: prompt, decodeDeltas: [0])

        let decoded = CBv2PositionState.decodePositionIds(
            states: [media, text, nil], cacheOffsets: [8, 3, 11])!
        XCTAssertEqual(decoded.shape, [3, 3, 1])
        XCTAssertEqual(values(decoded), [6, 3, 11, 6, 3, 11, 6, 3, 11])
    }

    func testIndependentPositionStatesRemainImmutable() {
        let prompt = MLXArray.zeros([3, 1, 1], dtype: .int32)
        let first = CBv2PositionState(promptPositionIds: prompt, decodeDeltas: [-4])
        let second = CBv2PositionState(promptPositionIds: prompt, decodeDeltas: [7])

        XCTAssertEqual(
            values(CBv2PositionState.decodePositionIds(states: [first], cacheOffsets: [10])!),
            [6, 6, 6])
        XCTAssertEqual(
            values(CBv2PositionState.decodePositionIds(states: [second], cacheOffsets: [2])!),
            [9, 9, 9])
        XCTAssertEqual(
            values(CBv2PositionState.decodePositionIds(states: [first], cacheOffsets: [12])!),
            [8, 8, 8])
    }
}
