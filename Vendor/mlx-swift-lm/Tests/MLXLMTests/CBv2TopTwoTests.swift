import MLX
import Testing
@_spi(Diagnostics) @testable import MLXLMCommon
@testable import MLXLLM

@Suite("CBv2 shared exact top-two", .serialized)
struct CBv2TopTwoTests {
    private func check(_ logits: MLXArray) {
        let generic = cbv2TopTwoRows(logits)
        let qwen = qwen35MTPTopTwoRows(logits)
        #expect(generic.ids.shape == [logits.dim(1), 2])
        #expect(generic.ids.dtype == .int32 && generic.values.dtype == .float32)
        eval(generic.ids, generic.values, qwen.ids, qwen.values)
        #expect(generic.ids.asArray(Int32.self) == qwen.ids.asArray(Int32.self))
        #expect(generic.values.asArray(Float.self).map(\.bitPattern)
            == qwen.values.asArray(Float.self).map(\.bitPattern))
        let rows = logits.asType(.float32).asArray(Float.self)
        let ids = generic.ids.asArray(Int32.self)
        let values = generic.values.asArray(Float.self)
        for row in 0..<logits.dim(1) {
            let start = row * logits.dim(2)
            let input = Array(rows[start..<(start + logits.dim(2))])
            let expected = input.indices.sorted { left, right in
                if input[left].isNaN != input[right].isNaN { return !input[left].isNaN }
                if input[left] > input[right] { return true }
                if input[left] < input[right] { return false }
                return left < right
            }.prefix(2)
            #expect(Array(ids[(row * 2)..<(row * 2 + 2)]) == expected.map(Int32.init))
            for (position, index) in expected.enumerated() {
                let actual = values[row * 2 + position]
                if input[index].isNaN { #expect(actual.isNaN) }
                else { #expect(actual.bitPattern == input[index].bitPattern) }
            }
        }
    }

    @Test(arguments: [DType.float16, .bfloat16, .float32])
    func exactOrderingIncludesNonfiniteAndSignedZeros(_ dtype: DType) {
        let rows: [[Float]] = [
            [7, 7, 7, -1, 0, 2], [.nan, -.infinity, .infinity, .nan, .infinity, 0],
            [.nan, .nan, .nan, .nan, .nan, .nan], [-0.0, 0, -1, -2, -3, -4],
            [1.0001, 1.0002, 1.0003, 1, 0, -1],
        ]
        check(MLXArray(rows.flatMap { $0 }).reshaped([1, rows.count, 6]).asType(dtype))
    }

    @Test(arguments: [DType.float16, .bfloat16, .float32])
    func stridedRowsKeepVocabularyAndRowAxes(_ dtype: DType) {
        let source = MLXArray((0..<3 * 19).map { Float(($0 * 13) % 23) - 11 })
            .reshaped([1, 19, 3]).asType(dtype)
        let logits = source.transposed(0, 2, 1)
        eval(logits)
        #expect(logits.strides[2] != 1)
        check(logits)
    }

    @Test func productionVocabularyAndMinimumVocabulary() {
        check(MLXArray([Float.nan, -.infinity, 4, 4]).reshaped([1, 2, 2]))
        var values = [Float](repeating: -1_000, count: 248_320)
        values[8_193] = 100
        values[248_319] = 100
        check(MLXArray(values).reshaped([1, 1, values.count]))
    }

    @Test func retainedPolicyReductionIsUsedWithoutRecomputation() async throws {
        let model = TinyTestModel.make(seed: 88, fullAttentionOnly: true)
        let engine = EngineV2(
            model: model, layerKinds: model.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 24)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds))
        try engine.configureLogitDiagnostic(.init(
            requestID: 7, outputIndex: 3, candidateIDs: [1, 2], maximumRecords: 1))
        let packet = engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.makeLogitDiagnostic(
                logits: MLXArray([Float(10), 20, 30]), requestID: .init(7), outputIndex: 3,
                phase: "seed", batchIndex: 0, batchSize: 1, seedToken: 1, cacheOffset: 8,
                policyTopTwo: (MLXArray([Int32(2), 1]), MLXArray([Float(999), 998])))
        }
        let observed = try #require(packet)
        eval(observed.evaluationTargets)
        let record = observed.materialize()
        #expect(record.argMaxID == 2, "independent argMax still reflects actual logits")
        #expect(record.topTwoIDs == [2, 1])
        #expect(Array(record.valueBits[1...2]) == [Float(999).bitPattern, Float(998).bitPattern],
            "sentinel retained policy values prove the fallback did not replace existing evidence")
        await engine.shutdown()
    }
}
