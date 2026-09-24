// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen35PositionStateTests: XCTestCase {
    private func makeConfig() throws -> Qwen35Configuration {
        let json = """
            {
                "model_type": "qwen3_5_moe_vl",
                "image_token_id": 10,
                "video_token_id": 11,
                "image_token_index": 12,
                "video_token_index": 13,
                "vision_start_token_id": 9,
                "vision_end_token_id": 14,
                "text_config": {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "linear_num_value_heads": 1,
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8,
                    "linear_conv_kernel_dim": 2,
                    "vocab_size": 64,
                    "full_attention_interval": 1,
                    "num_experts": 0,
                    "num_experts_per_tok": 0
                },
                "vision_config": {
                    "model_type": "qwen3_5_moe_vl",
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "out_hidden_size": 8,
                    "num_heads": 1,
                    "patch_size": 1,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 1,
                    "num_position_embeddings": 16,
                    "in_channels": 1
                }
            }
            """
        return try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
    }

    private func values(_ array: MLXArray) -> [Int] {
        array.asType(.int32).asArray(Int32.self).map(Int.init)
    }

    func testTextPositionsAndChunkSlices() throws {
        let model = Qwen35MoE(try makeConfig())
        let result = try model.positionResult(tokens: MLXArray([Int32(1), 2, 3, 4]))

        XCTAssertEqual(result.promptPositionIds.shape, [3, 1, 4])
        XCTAssertEqual(
            values(result.promptPositionIds),
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        XCTAssertEqual(result.decodeState.batchSize, 1)
        XCTAssertEqual(result.decodeState.deltas, [0])

        let slice = result.promptPositionIds(in: 1 ..< 3)
        XCTAssertEqual(slice.shape, [3, 1, 2])
        XCTAssertEqual(values(slice), [1, 2, 1, 2, 1, 2])

        let decode = result.decodeState.decodePositionIds(cacheOffset: 4, length: 2)
        XCTAssertEqual(decode.shape, [3, 1, 2])
        XCTAssertEqual(values(decode), [4, 5, 4, 5, 4, 5])
    }

    func testIndependentRequestsKeepIndependentDecodeDeltas() throws {
        let model = Qwen35MoE(try makeConfig())

        // One 4x4 image with merge=2 occupies four placeholder tokens. Its
        // three-axis grid compresses an 8-token prompt to max position 5,
        // yielding decode delta -2.
        let mediaTokens = MLXArray([Int32(1), 9, 10, 10, 10, 10, 2, 3])
        let media = try model.positionResult(
            tokens: mediaTokens,
            imageGrids: [THW(1, 4, 4)])
        XCTAssertEqual(media.decodeState.deltas, [-2])

        // Computing another request must not overwrite the first request's
        // state. Text has delta zero.
        let text = try model.positionResult(tokens: MLXArray([Int32(4), 5, 6]))
        XCTAssertEqual(text.decodeState.deltas, [0])
        XCTAssertEqual(
            values(text.decodeState.decodePositionIds(cacheOffset: 3)),
            [3, 3, 3])

        XCTAssertEqual(
            values(media.decodeState.decodePositionIds(cacheOffset: 8, length: 2)),
            [6, 7, 6, 7, 6, 7],
            "request A's decode positions must still use request A's delta")
    }

    func testVisualTokenRunsMustExactlyMatchOrderedGrids() throws {
        let model = Qwen35MoE(try makeConfig())

        XCTAssertThrowsError(
            try model.positionResult(
                tokens: MLXArray([Int32(9), 10, 10, 10, 14]),
                imageGrids: [THW(1, 4, 4)])) { error in
            XCTAssertEqual(
                error as? Qwen35PositionSeamError,
                .visualTokenRunMismatch(
                    kind: "image", gridIndex: 0, expected: 4, actual: 3))
        }
        XCTAssertThrowsError(
            try model.positionResult(
                tokens: MLXArray([Int32(1), 2, 3]),
                imageGrids: [THW(1, 4, 4)])) { error in
            XCTAssertEqual(
                error as? Qwen35PositionSeamError,
                .visualTokenRunMismatch(
                    kind: "image", gridIndex: 0, expected: 4, actual: 0))
        }
        XCTAssertThrowsError(
            try model.positionResult(
                tokens: MLXArray([Int32(9), 10, 10, 10, 10, 14]))) { error in
            XCTAssertEqual(
                error as? Qwen35PositionSeamError,
                .visualTokenRunMismatch(
                    kind: "image", gridIndex: 0, expected: 0, actual: 4))
        }
        XCTAssertThrowsError(
            try model.positionResult(
                tokens: MLXArray([
                    Int32(9), 10, 10, 10, 10, 14,
                    9, 11, 14,
                ]),
                imageGrids: [THW(1, 4, 4)],
                videoGrids: [THW(2, 2, 2)])) { error in
            XCTAssertEqual(
                error as? Qwen35PositionSeamError,
                .visualTokenRunMismatch(
                    kind: "video", gridIndex: 0, expected: 2, actual: 1))
        }
    }

    func testPositionSeamDoesNotChangeLegacySerialCalls() throws {
        MLXRandom.seed(0)
        let model = Qwen35MoE(try makeConfig())
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)

        let before = model(tokens, cache: nil)
        eval(before)

        let requestState = try model.positionResult(tokens: tokens)
        eval(requestState.promptPositionIds)

        let after = model(tokens, cache: nil)
        eval(after)
        XCTAssertTrue(
            allClose(before, after, rtol: 0, atol: 0).item(Bool.self),
            "request-owned position computation must not mutate legacy ropeDeltas")
    }

    func testPositionTypesAreExplicitlySendable() {
        func requireSendable<T: Sendable>(_: T.Type) {}
        requireSendable(Qwen35PositionResult.self)
        requireSendable(Qwen35PositionState.self)
    }
}
