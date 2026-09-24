import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite("Nemotron 3.5 real checkpoint serial oracle", .serialized)
struct NemotronH35RealCheckpointTests {
    @Test(.enabled(if:
        ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_REAL_MODEL"] != nil))
    func greedyTokensMatchMLXLM() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_REAL_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let configData = try Data(
            contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(
            NemotronH35Configuration.self, from: configData)
        let base = try JSONDecoder().decode(
            BaseConfiguration.self, from: configData)
        let model = NemotronH35Model(configuration)
        try loadWeights(
            modelDirectory: directory,
            model: model,
            perLayerQuantization: base.perLayerQuantization)
        eval(model.parameters().flattened().map(\.1))

        let recurrentSpec = model.cbv2RecurrentStateSpec
        #expect(recurrentSpec.layers.count == 23)
        #expect(recurrentSpec.layers.allSatisfy { $0.convDType == .bfloat16 })
        #expect(recurrentSpec.layers.allSatisfy { $0.ssmDType == .float32 })
        #expect(try recurrentSpec.fixedBytesPerRequest() == 49_082_368)

        let prompt = [
            10, 25708, 1010, 11, 1010, 10, 3263, 1010, 53156, 1454,
            10693, 1058, 1464, 15187, 6918, 1082, 2918, 1335, 55882,
            23313, 11, 1010, 10, 1503, 19464, 1010, 12, 13,
        ]
        let expected = [
            1078, 15187, 6918, 1082, 2918, 1335, 55882, 23313, 11,
        ]
        let caches = model.newCache(parameters: nil)
        var logits = model(
            MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count),
            cache: caches)
        var actual: [Int] = []
        for _ in expected.indices {
            eval(logits)
            let token = logits[0, -1].argMax().item(Int.self)
            actual.append(token)
            logits = model(
                MLXArray([Int32(token)]).reshaped(1, 1), cache: caches)
        }

        #expect(actual == expected)

        // Cross both 256-token SSM scan boundaries used by mlx-lm. A
        // monolithic quadratic scan can agree on short prompts yet diverge
        // here because the recurrent rounding boundary is part of the oracle.
        let longPrompt =
            [10, 25708, 1010, 11, 1010, 10, 3263, 1010]
            + (0 ..< 600).map { 1000 + ($0 % 200) }
            + [11, 1010, 10, 1503, 19464, 1010, 12, 1010]
        let longExpected = [
            11745, 1681, 1261, 11483, 2832, 2100, 1049, 1046,
            1032, 1603, 25280, 73448, 7777, 16196, 24185, 1256,
        ]
        let longCaches = model.newCache(parameters: nil)
        logits = model(
            MLXArray(longPrompt.map(Int32.init)).reshaped(1, longPrompt.count),
            cache: longCaches)
        actual = []
        for _ in longExpected.indices {
            eval(logits)
            let token = logits[0, -1].argMax().item(Int.self)
            actual.append(token)
            logits = model(
                MLXArray([Int32(token)]).reshaped(1, 1),
                cache: longCaches)
        }

        #expect(actual == longExpected)
    }
}
