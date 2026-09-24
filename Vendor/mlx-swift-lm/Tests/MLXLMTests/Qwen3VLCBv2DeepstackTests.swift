import MLX
import XCTest

@testable import MLXLMCommon

final class Qwen3VLCBv2DeepstackTests: XCTestCase {
    func testMaterializesEveryLayerAndSpanExactlyOnce() throws {
        let model = DeepstackFixture()
        let spans = [
            CBv2ImageSpan(tokenOffset: 2, length: 3),
            CBv2ImageSpan(tokenOffset: 7, length: 2),
        ]
        var embeddingCalls = 0
        var deepstackCalls = 0
        let input = CBv2MultimodalInput(
            spans: spans,
            attention: .causal,
            deepstackEmbeddings: {
                deepstackCalls += 1
                return [
                    [rows([10, 11, 12]), rows([20, 21])],
                    [rows([30, 31, 32]), rows([40, 41])],
                ]
            },
            embeddings: {
                embeddingCalls += 1
                return [rows([1, 2, 3]), rows([4, 5])]
            })

        let resolved = try CBv2MultimodalPlan.materialize(
            input, blocks: [], model: model)

        XCTAssertEqual(embeddingCalls, 1)
        XCTAssertEqual(deepstackCalls, 1)
        XCTAssertEqual(resolved.embeddings.map(\.shape), [[1, 3, 4], [1, 2, 4]])
        XCTAssertEqual(
            resolved.deepstackEmbeddings.map { $0.map(\.shape) },
            [
                [[1, 3, 4], [1, 2, 4]],
                [[1, 3, 4], [1, 2, 4]],
            ])
    }

    func testCausalChunkSlicesEveryDeepstackSpan() throws {
        let model = DeepstackFixture()
        let input = CBv2MultimodalInput(
            spans: [
                CBv2ImageSpan(tokenOffset: 2, length: 3),
                CBv2ImageSpan(tokenOffset: 7, length: 2),
            ],
            attention: .causal,
            deepstackEmbeddings: {
                [
                    [rows([10, 11, 12]), rows([20, 21])],
                    [rows([30, 31, 32]), rows([40, 41])],
                ]
            },
            embeddings: { [rows([1, 2, 3]), rows([4, 5])] })
        let resolved = try CBv2MultimodalPlan.materialize(
            input, blocks: [], model: model)

        let chunk = resolved.deepstackInChunk(
            start: 3, count: 5, hidden: 4, dtype: .float32)
        eval(chunk)

        XCTAssertEqual(chunk.map(\.shape), [[1, 5, 4], [1, 5, 4]])
        XCTAssertEqual(chunk[0][0, 0..., 0].asArray(Float.self), [11, 12, 0, 0, 20])
        XCTAssertEqual(chunk[1][0, 0..., 0].asArray(Float.self), [31, 32, 0, 0, 40])
    }

    func testRejectsMissingDeepstackLayer() {
        let model = DeepstackFixture()
        let input = CBv2MultimodalInput(
            spans: [CBv2ImageSpan(tokenOffset: 0, length: 1)],
            attention: .causal,
            deepstackEmbeddings: { [[rows([2])]] },
            embeddings: { [rows([1])] })

        XCTAssertThrowsError(
            try CBv2MultimodalPlan.materialize(input, blocks: [], model: model)
        ) { error in
            guard case CBv2MultimodalError.embeddingMismatch(let message) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(message.contains("returned 1 layers"))
            XCTAssertTrue(message.contains("expects 2"))
        }
    }
}

private final class DeepstackFixture: CBv2DeepstackMultimodalSteppableModel {
    let deepstackLayerCount = 2

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        MLXArray.zeros([tokens.dim(0), tokens.dim(1), 8])
    }

    func embedPromptTokens(_ tokens: MLXArray) -> MLXArray {
        MLXArray.zeros([tokens.dim(0), tokens.dim(1), 4])
    }

    func forward(
        tokens: MLXArray,
        inputEmbeddings: MLXArray,
        caches: [CBv2AttendingLayerCache]
    ) -> MLXArray {
        MLXArray.zeros([tokens.dim(0), tokens.dim(1), 8])
    }

    func forward(
        tokens: MLXArray,
        inputEmbeddings: MLXArray,
        deepstackEmbeddings: [MLXArray],
        caches: [CBv2AttendingLayerCache],
        positionIds: MLXArray?
    ) -> MLXArray {
        MLXArray.zeros([tokens.dim(0), tokens.dim(1), 8])
    }
}

private func rows(_ values: [Float]) -> MLXArray {
    MLXArray(values.flatMap { Array(repeating: $0, count: 4) }, [values.count, 4])
}
