import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class NemotronH35MTPRealTests: XCTestCase {
    func testLoadedTargetGreedyAndEmbeddedHeadAgainstPythonReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["DARKBLOOM_NEMOTRON35_MTP_MODEL"],
            let oracle = env["DARKBLOOM_NEMOTRON35_MTP_ORACLE"] else {
            throw XCTSkip("Requires converted embedded-MTP artifact and independent Python fixture")
        }
        let dir = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        let args = try JSONDecoder().decode(NemotronH35Configuration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let target = NemotronH35Model(args)
        try loadWeights(modelDirectory: dir, model: target, perLayerQuantization: base.perLayerQuantization)
        eval(target)
        print("Nemotron target attention dtypes: \(target.cbv2CompleteCheckpointKVDTypes ?? [])")
        print("Nemotron target convolution dtypes: \(target.cbv2RecurrentStateSpec.layers.map(\.convDType))")
        let assistant = try NemotronH35MTPAssistant.load(from: dir, target: target)
        let fixture = try loadArrays(url: URL(fileURLWithPath: oracle))
        let prompt = try XCTUnwrap(fixture["prompt"])
        let expected = try XCTUnwrap(fixture["expected"]).asArray(Int32.self)
        let caches = target.newCache(parameters: nil)
        var logits = target(prompt.reshaped(1, prompt.size), cache: caches)
        for (index, expectedID) in expected.enumerated() {
            let id = argMax(logits[0..., -1, 0...], axis: -1).asType(.int32)
            eval(id)
            XCTAssertEqual(id.item(Int32.self), expectedID, "target greedy differs at position \(index)")
            logits = target(id.reshaped(1, 1), cache: caches)
        }
        let headCache = KVCacheSimple()
        let fullHistory = KVCacheSimple(), kvHistory = KVCacheSimple()
        for i in 0..<2 {
            let h = try XCTUnwrap(fixture["hidden_\(i)"])
            let embedding = target.mtpEmbedding(try XCTUnwrap(fixture["token_\(i)"]))
            let output = assistant.module(hidden: h, embedding: embedding, cache: fullHistory)
            let roots = assistant.module.appendTrustedKV(hidden: h, embedding: embedding, cache: kvHistory)
            eval([output] + roots + fullHistory.innerState() + kvHistory.innerState())
            for (a, b) in zip(fullHistory.state, kvHistory.state) {
                XCTAssertEqual(a.asData(access: .copy).data, b.asData(access: .copy).data)
            }
        }
        let h = try XCTUnwrap(fixture["hidden_2"])
        let embedding = target.mtpEmbedding(try XCTUnwrap(fixture["token_2"]))
        let fullNext = assistant.module(hidden: h, embedding: embedding, cache: fullHistory)
        let kvNext = assistant.module(hidden: h, embedding: embedding, cache: kvHistory)
        eval(fullNext, kvNext)
        XCTAssertEqual(fullNext.asData(access: .copy).data, kvNext.asData(access: .copy).data)
        if fixture["stage_fused"] != nil {
            let l0 = assistant.module.layers[0] as! NemotronH35MTPAttentionBlock
            let l1 = assistant.module.layers[1] as! NemotronH35MTPExpertBlock
            let e = target.mtpEmbedding(fixture["token_0"]!)
            let en = l0.embeddingNorm(e), hn = l0.hiddenNorm(fixture["hidden_0"]!)
            let fused = l0.projection(concatenated([en, hn], axis: -1))
            let attn = fused + l0.mixer(l0.norm(fused), attentionMask: .none, ssmMask: nil, cache: KVCacheSimple())
            let expert = attn + l1.mixer(l1.norm(attn))
            for (name, value) in [("embedding", e), ("en", en), ("hn", hn), ("fused", fused), ("attn", attn), ("expert", expert)] {
                let expectedValue = fixture["stage_" + name]!
                let error = abs(value - expectedValue).max().item(Float.self)
                print("Nemotron MTP stage=\(name) dtype=\(value.dtype)/\(expectedValue.dtype) max_abs=\(error)")
            }
        }
        for i in 0..<3 {
            let hidden = try XCTUnwrap(fixture["hidden_\(i)"])
            let token = try XCTUnwrap(fixture["token_\(i)"])
            let expectedHidden = try XCTUnwrap(fixture["draft_hidden_\(i)"])
            let expectedLogits = try XCTUnwrap(fixture["draft_logits_\(i)"])
            let actualHidden = assistant.module(hidden: hidden, embedding: target.mtpEmbedding(token), cache: headCache)
            let actual = target.logits(actualHidden)
            eval(actual, actualHidden)
            print("Nemotron MTP step=\(i) hidden_max_abs=\(abs(actualHidden - expectedHidden).max().item(Float.self)) logits_max_abs=\(abs(actual - expectedLogits).max().item(Float.self))")
            XCTAssertEqual(argMax(actual, axis: -1).item(Int32.self), argMax(expectedLogits, axis: -1).item(Int32.self))
            // Cross-runtime numerical audit is separate from target exactness:
            // Python and Swift link different MLX QMM implementations. Keep
            // this explicitly selectable audit (including the original failed
            // evidence); do not call matching draft IDs byte-identical logits.
            if env["DARKBLOOM_NEMOTRON35_REQUIRE_CROSS_RUNTIME_LOGITS"] == "1" {
                XCTAssertTrue(allClose(actualHidden, expectedHidden, rtol: 1e-4, atol: 1e-4).item(Bool.self), "head hidden differs at step \(i)")
                XCTAssertTrue(allClose(actual, expectedLogits, rtol: 1e-4, atol: 1e-4).item(Bool.self), "head logits differ at step \(i)")
            }
        }
    }
}
