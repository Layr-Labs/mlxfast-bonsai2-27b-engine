// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Qwen35VisionSeamTests: XCTestCase {
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
                    "spatial_merge_size": 1,
                    "temporal_patch_size": 1,
                    "num_position_embeddings": 16,
                    "in_channels": 1
                }
            }
            """
        return try JSONDecoder().decode(
            Qwen35Configuration.self, from: Data(json.utf8))
    }

    func testPlaceholderAndCausalAttentionFacts() throws {
        let model = Qwen35MoE(try makeConfig())
        let facts = model.visionSeamConfiguration

        XCTAssertEqual(model.imagePlaceholderTokenId, 12)
        XCTAssertEqual(model.videoPlaceholderTokenId, 13)
        XCTAssertEqual(facts.imagePlaceholderTokenId, 12)
        XCTAssertEqual(facts.videoPlaceholderTokenId, 13)
        XCTAssertEqual(facts.imagePositionTokenId, 10)
        XCTAssertEqual(facts.videoPositionTokenId, 11)
        XCTAssertEqual(facts.visionStartTokenId, 9)
        XCTAssertEqual(facts.visionEndTokenId, 14)
        XCTAssertEqual(facts.spatialMergeSize, 1)
        XCTAssertEqual(facts.temporalPatchSize, 1)
        XCTAssertEqual(facts.attention, .causal)
    }

    func testFeatureOrderingAndPerFrameSlicesMatchFlattenedPrepareOrder() throws {
        MLXRandom.seed(1)
        let model = Qwen35MoE(try makeConfig())

        // Two images (1 and 2 visual tokens), followed by one two-frame video
        // (2 visual tokens per temporal frame). This mirrors prepare's pixel
        // packing: all image grids first, then all video grids.
        let imagePixels = MLXArray([Float(1), 2, 3]).reshaped(3, 1)
        let imageGrids = [THW(1, 1, 1), THW(1, 1, 2)]
        let videoPixels = MLXArray([Float(4), 5, 6, 7]).reshaped(4, 1)
        let videoGrids = [THW(2, 1, 2)]

        let result = try model.visionFeatures(
            imagePixels: imagePixels,
            imageGrids: imageGrids,
            videoPixels: videoPixels,
            videoGrids: videoGrids)
        eval(result.flattenedFeatures, result.ordered.map(\.features))

        XCTAssertEqual(result.flattenedFeatures.shape, [7, 8])
        XCTAssertEqual(result.ordered.count, 4)
        XCTAssertEqual(result.ordered.map(\.kind), [
            .image(index: 0),
            .image(index: 1),
            .videoFrame(videoIndex: 0, frameIndex: 0),
            .videoFrame(videoIndex: 0, frameIndex: 1),
        ])
        XCTAssertEqual(result.ordered.map { $0.features.shape }, [
            [1, 1, 8], [1, 2, 8], [1, 2, 8], [1, 2, 8],
        ])

        let reconstructed = concatenated(result.ordered.map(\.features), axis: 1)
            .squeezed(axis: 0)
        XCTAssertTrue(
            allClose(reconstructed, result.flattenedFeatures, rtol: 0, atol: 0)
                .item(Bool.self),
            "ordered slices must reconstruct the exact flattened prepare features")
    }

    func testMixedMediaPixelCountsAreValidatedPerKind() throws {
        let model = Qwen35MoE(try makeConfig())
        let imagePixels = MLXArray([Float(1), 2, 3]).reshaped(3, 1)
        let videoPixels = MLXArray([Float(4), 5, 6]).reshaped(3, 1)

        XCTAssertThrowsError(
            try model.visionFeatures(
                imagePixels: imagePixels,
                imageGrids: [THW(1, 1, 2)],
                videoPixels: videoPixels,
                videoGrids: [THW(1, 1, 4)])) { error in
            XCTAssertEqual(
                error as? Qwen35VisionSeamError,
                .pixelCountMismatch(kind: "image", expected: 2, actual: 3))
        }
        XCTAssertThrowsError(
            try model.visionFeatures(
                imagePixels: MLXArray([Float(1), 2]).reshaped(2, 1),
                imageGrids: [THW(1, 1, 2)],
                videoPixels: videoPixels,
                videoGrids: [THW(1, 1, 4)])) { error in
            XCTAssertEqual(
                error as? Qwen35VisionSeamError,
                .pixelCountMismatch(kind: "video", expected: 4, actual: 3))
        }
    }

    func testLegacyPrepareStillAcceptsMixedMedia() throws {
        MLXRandom.seed(2)
        let model = Qwen35MoE(try makeConfig())
        let image = LMInput.ProcessedImage(
            pixels: MLXArray([Float(1), 2, 3]).reshaped(3, 1),
            frames: [THW(1, 1, 1), THW(1, 1, 2)])
        let video = LMInput.ProcessedVideo(
            pixels: MLXArray([Float(4), 5, 6, 7]).reshaped(4, 1),
            frames: [THW(2, 1, 2)])

        // Position ids use image_token_id/video_token_id. Embedding scatter
        // uses image_token_index/video_token_index, preserving legacy support
        // for checkpoints where those config facts differ.
        let tokens = MLXArray([
            Int32(9), 10, 12, 14,
            9, 10, 10, 12, 12, 14,
            9, 11, 11, 11, 11, 13, 13, 13, 13, 14,
        ]).reshaped(1, 20)
        let input = LMInput(text: .init(tokens: tokens), image: image, video: video)

        let prepared = try model.prepare(input, cache: model.newCache(parameters: nil), windowSize: nil)
        guard case .logits(let output) = prepared else {
            return XCTFail("Qwen35 legacy prepare must continue returning logits")
        }
        eval(output.logits)
        XCTAssertEqual(output.logits.shape, [1, 20, 64])
    }

    func testVisionTypesAreExplicitlySendable() {
        func requireSendable<T: Sendable>(_: T.Type) {}
        requireSendable(Qwen35VisionFeature.self)
        requireSendable(Qwen35VisionFeatures.self)
    }
}
