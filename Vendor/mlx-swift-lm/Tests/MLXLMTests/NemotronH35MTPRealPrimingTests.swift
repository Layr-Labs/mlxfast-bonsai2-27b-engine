import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class NemotronH35MTPRealPrimingTests: XCTestCase {
    func testLoadedHeadPromptKVAndFirstDraftMatchFullModule() throws {
        guard let path = ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_MTP_MODEL"] else {
            throw XCTSkip("Requires the retained embedded-MTP artifact")
        }
        let directory = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let target = NemotronH35Model(try JSONDecoder().decode(NemotronH35Configuration.self, from: data))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        try loadWeights(modelDirectory: directory, model: target, perLayerQuantization: base.perLayerQuantization)
        eval(target)
        let assistant = try NemotronH35MTPAssistant.load(from: directory, target: target)
        MLXRandom.seed(35003513)
        // Synthetic post-norm-shaped inputs isolate the loaded head's state
        // contract. Whole-engine target/output parity is a separate gate.
        let hidden = MLXRandom.normal([1, 128, target.configuration.hiddenSize]).asType(.bfloat16)
        let tokens = MLXArray((0..<128).map { Int32(1000 + $0) }).reshaped(1, 128)
        eval(hidden, tokens)
        for chunks in [[128], [63, 65]] {
            let state = try XCTUnwrap(assistant.makeRequestState() as? NemotronH35MTPAssistant.State)
            try assistant.configureRequestState(state, maximumSequenceLength: 256)
            defer { assistant.releaseRequestState(state) }
            let reference = KVCacheSimple()
            var start = 0
            for count in chunks {
                let end = start + count
                assistant.observeCommittedTarget(.init(tokens: tokens[0..., start..<end],
                    hidden: hidden[0..., start..<end, 0...]), requestState: state)
                start = end
            }
            let primedReference = assistant.module(
                hidden: hidden[0..., 0..<127, 0...],
                embedding: target.mtpEmbedding(tokens[0..., 1..<128]), cache: reference)
            eval([primedReference] + reference.innerState())
            XCTAssertEqual(state.pendingTokens.reduce(0) { $0 + $1.dim(1) }, 127)
            let checkpoint = try XCTUnwrap(assistant.capturePrefixCheckpoint(
                requestState: state, targetInputCount: 128))
            let encoded = try XCTUnwrap(assistant.encodePrefixCheckpoint(checkpoint))
            eval(encoded)
            let decoded = try XCTUnwrap(assistant.decodePrefixCheckpoint(
                tensors: encoded, prefixTokens: (0..<128).map { 1000 + $0 }))
            let warm = try XCTUnwrap(assistant.restorePrefixCheckpoint(decoded))
            try assistant.configureRequestState(warm, maximumSequenceLength: 256)
            defer { assistant.releaseRequestState(warm) }
            let seed = MLXArray([Int32(1128)]).reshaped(1, 1)
            let frontier = hidden[0..., 127..<128, 0...]
            let expected = assistant.module(hidden: frontier, embedding: target.mtpEmbedding(seed), cache: reference)
            let expectedIDs = argMax(target.logits(expected)[0..., -1, 0...], axis: -1).asType(.int32)
            eval([expected, expectedIDs] + reference.innerState())
            let actual = assistant.draftStep(tokens: seed, hidden: frontier, shortlist: nil, requestState: state)
            let warmResult = assistant.draftStep(
                tokens: seed, hidden: frontier, shortlist: nil, requestState: warm)
            eval([actual.hidden, actual.tokens, warmResult.hidden, warmResult.tokens]
                + assistant.evaluationTargets(for: state)
                + assistant.evaluationTargets(for: warm))
            XCTAssertEqual(state.cacheOffset, 128)
            XCTAssertEqual(actual.hidden.dtype, .bfloat16)
            XCTAssertTrue(allClose(actual.hidden, expected, rtol: 1e-4, atol: 1e-4)
                .item(Bool.self))
            XCTAssertEqual(actual.tokens.asArray(Int32.self), expectedIDs.asArray(Int32.self))
            XCTAssertTrue(allClose(warmResult.hidden, actual.hidden, rtol: 1e-4, atol: 1e-4)
                .item(Bool.self))
            XCTAssertEqual(warmResult.tokens.asArray(Int32.self), actual.tokens.asArray(Int32.self))
            let warmState = try XCTUnwrap(warm as? NemotronH35MTPAssistant.State)
            for (coldKV, warmKV) in zip(state.cacheSnapshot, warmState.cacheSnapshot) {
                XCTAssertEqual(coldKV.asData(access: .copy).data,
                    warmKV.asData(access: .copy).data)
            }
            XCTAssertEqual(state.cacheSnapshot.count, reference.state.count)
            for (a, b) in zip(state.cacheSnapshot, reference.state) {
                XCTAssertEqual(a.dtype, b.dtype)
                XCTAssertEqual(a.asData(access: .copy).data, b.asData(access: .copy).data)
            }
            assistant.discardRound(requestState: state)
            assistant.discardRound(requestState: warm)
            let retry = assistant.draftStep(tokens: seed, hidden: frontier, shortlist: nil, requestState: state)
            eval([retry.hidden] + assistant.evaluationTargets(for: state))
            XCTAssertEqual(retry.hidden.asData(access: .copy).data, expected.asData(access: .copy).data)
            assistant.releaseRequestState(state)
            XCTAssertEqual(state.materializedBytes, 0)
        }
    }
}
