import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class Qwen35VisionCBv2SemanticsTests: XCTestCase {
    func testCausalVisionDoesNotCoalesceOrSnapChunks() throws {
        let model = TinyTestModel.make(seed: 314)
        let span = CBv2ImageSpan(tokenOffset: 2, length: 5)
        let embedding = MLXArray.ones([1, 5, model.config.hiddenSize])
        let input = CBv2MultimodalInput(
            spans: [span], attention: .causal
        ) { [embedding] }
        let resolved = try CBv2MultimodalPlan.resolve(
            input, promptTokenCount: 10, model: model,
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
            maxBatchedTokensPerStep: 4)

        XCTAssertEqual(resolved.blocks, [])
        XCTAssertNil(resolved.chunkContext(start: 0, count: 4))
        XCTAssertEqual(resolved.spansInChunk(start: 0, count: 4).first?.span,
            CBv2ImageSpan(tokenOffset: 2, length: 2))

        var request = CBv2Request(
            id: CBv2RequestID(1), promptTokens: Array(repeating: 7, count: 10),
            maxTokens: 1, multimodal: input)
        request.positionState = nil
        let record = CBv2ScheduledRequest(request: request, arrivalSeq: 0, submittedAt: Date())
        XCTAssertEqual(record.snappedChunkTokens(start: 0, proposed: 4, budget: 4), 4)
        XCTAssertEqual(record.snappedChunkTokens(start: 4, proposed: 4, budget: 4), 4)
    }

    func testGemmaBidirectionalSemanticsRemainUnchanged() throws {
        let model = TinyTestModel.make(seed: 315)
        let span = CBv2ImageSpan(tokenOffset: 2, length: 5)
        let embedding = MLXArray.ones([1, 5, model.config.hiddenSize])
        let input = CBv2MultimodalInput(spans: [span]) { [embedding] }
        let resolved = try CBv2MultimodalPlan.resolve(
            input, promptTokenCount: 10, model: model,
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
            maxBatchedTokensPerStep: 8)

        XCTAssertEqual(resolved.blocks, [span])
        XCTAssertNotNil(resolved.chunkContext(start: 2, count: 5))
        let request = CBv2Request(
            id: CBv2RequestID(2), promptTokens: Array(repeating: 7, count: 10),
            maxTokens: 1, multimodal: input)
        let record = CBv2ScheduledRequest(request: request, arrivalSeq: 0, submittedAt: Date())
        XCTAssertEqual(record.snappedChunkTokens(start: 0, proposed: 4, budget: 8), 2)
        XCTAssertEqual(record.snappedChunkTokens(start: 2, proposed: 4, budget: 8), 5)
    }
}
