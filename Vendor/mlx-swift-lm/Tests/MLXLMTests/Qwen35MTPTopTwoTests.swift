import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite("Qwen35 MTP hierarchical top-2", .serialized)
struct Qwen35MTPTopTwoTests {
    private func reduce(
        _ rows: [[Float]]
    ) -> (ids: MLXArray, values: MLXArray, flatIDs: [Int32], flatValues: [Float]) {
        let vocabularySize = rows[0].count
        let logits = MLXArray(rows.flatMap { $0 }).reshaped([1, rows.count, vocabularySize])
        let result = qwen35MTPTopTwoRows(logits)
        eval(result.ids, result.values)
        return (
            result.ids,
            result.values,
            result.ids.asArray(Int32.self),
            result.values.asArray(Float.self)
        )
    }

    private func policyProvider() throws -> Qwen35TextModel {
        let data = Data(
            """
            {
              "model_type": "qwen3_5_moe",
              "hidden_size": 8,
              "num_hidden_layers": 4,
              "intermediate_size": 16,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "linear_num_value_heads": 1,
              "linear_num_key_heads": 1,
              "linear_key_head_dim": 8,
              "linear_value_head_dim": 8,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 8,
              "head_dim": 8,
              "full_attention_interval": 4,
              "num_experts": 0,
              "num_experts_per_tok": 0,
              "mtp_num_hidden_layers": 1
            }
            """.utf8)
        return Qwen35TextModel(
            try JSONDecoder().decode(Qwen35TextConfiguration.self, from: data))
    }

    @Test("returns ordered ids and values for every row without row mixing")
    func multipleRows() {
        let result = reduce([
            [0, 1, 9, -3, 7, 2],
            [100, -1, 2, 3, 4, 99],
            [-5, 8, 7, 6, 9, -4],
        ])

        #expect(result.ids.shape == [3, 2])
        #expect(result.values.shape == [3, 2])
        #expect(result.ids.dtype == .int32)
        #expect(result.values.dtype == .float32)
        #expect(result.flatIDs == [2, 4, 0, 5, 4, 1])
        #expect(result.flatValues == [9, 7, 100, 99, 9, 8])
    }

    @Test("model provider preserves batch and sequence row order lazily")
    func providerRectangularShape() throws {
        let logits = MLXArray([
            Float(0), 9, 8, 1,
            7, 0, 6, 1,
            0, 5, 4, 3,
            2, 1, 0, 8,
            4, 3, 9, 0,
            6, 7, 1, 0,
        ]).reshaped([2, 3, 4])
        let provider: any CBv2MTPPolicyTopTwoProviding = try policyProvider()
        let result = provider.cbv2MTPTopTwo(logits)
        #expect(result.ids.shape == [2, 3, 2])
        #expect(result.values.shape == [2, 3, 2])
        eval(result.ids, result.values)
        #expect(result.ids.asArray(Int32.self) == [
            1, 2,
            0, 2,
            1, 2,
            3, 0,
            2, 0,
            1, 0,
        ])
    }

    @Test("exact ties choose lower token ids")
    func exactTieOrdering() {
        let result = reduce([[3, 7, 7, 7, 2]])

        #expect(result.flatIDs == [1, 2])
        #expect(result.flatValues == [7, 7])
    }

    @Test("NaNs sort after numbers and tie by token id")
    func nanOrdering() {
        let finiteThenNaN = reduce([[.nan, .nan, -4, .nan]])
        #expect(finiteThenNaN.flatIDs == [2, 0])
        #expect(finiteThenNaN.flatValues[0] == -4)
        #expect(finiteThenNaN.flatValues[1].isNaN)

        let allNaN = reduce([[.nan, .nan, .nan, .nan]])
        #expect(allNaN.flatIDs == [0, 1])
        #expect(allNaN.flatValues.allSatisfy { $0.isNaN })
    }

    @Test("reduces a production-sized vocabulary across distant stripes")
    func largeVocabularyRow() {
        let vocabularySize = 248_320
        var row = [Float](repeating: -1_000, count: vocabularySize)
        row[8_193] = 100
        row[vocabularySize - 1] = 99

        let result = reduce([row])

        #expect(result.flatIDs == [8_193, Int32(vocabularySize - 1)])
        #expect(result.flatValues == [100, 99])
    }
}
