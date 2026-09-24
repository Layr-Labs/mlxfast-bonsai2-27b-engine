import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class NemotronH35MTPTests: XCTestCase {
    func testUnweightedMambaNormPreservesNativeActivationDType() {
        let norm = NemotronHRMSNormGated(dimensions: 64, eps: 1e-5, groupSize: 16)
        norm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.ones([64], dtype: .bfloat16)]))
        let x = MLXArray.ones([1, 1, 64], dtype: .bfloat16)
        XCTAssertEqual(norm(x, gate: x).dtype, .bfloat16,
            "Python unweighted rms_norm preserves BF16; an FP32 identity weight must not widen it")
    }
    private func model() throws -> NemotronH35Model {
        let data = Data("""
        {"model_type":"nemotron_h","vocab_size":128,"hidden_size":64,
         "num_hidden_layers":2,"num_attention_heads":1,"num_key_value_heads":1,
         "head_dim":64,"mamba_num_heads":4,"mamba_head_dim":16,"ssm_state_size":16,
         "conv_kernel":4,"n_groups":2,"intermediate_size":128,"moe_intermediate_size":64,
         "moe_shared_expert_intermediate_size":64,"n_routed_experts":4,
         "num_experts_per_tok":2,"layers_block_type":["mamba","attention"],
         "mamba_ssm_cache_dtype":"float32","num_nextn_predict_layers":1,
         "mtp_layers_block_type":["attention","moe"]}
        """.utf8)
        return NemotronH35Model(try JSONDecoder().decode(NemotronH35Configuration.self, from: data))
    }

    func testOfficialNamespaceAndQuantizedHeadsRetainAllModules() throws {
        let head = NemotronH35MTPModule(try model().configuration)
        let original = Set(head.parameters().flattened().map(\.0))
        XCTAssertTrue(original.contains("layers.0.eh_proj.weight"))
        XCTAssertTrue(original.contains("layers.1.final_layernorm.weight"))
        XCTAssertTrue(original.contains("layers.1.mixer.switch_mlp.fc1.weight"))
        quantize(model: head, groupSize: 32, bits: 4)
        let converted = Set(head.parameters().flattened().map(\.0))
        XCTAssertTrue(original.isSubset(of: converted))
        XCTAssertTrue(converted.contains("layers.0.eh_proj.scales"))
        XCTAssertTrue(converted.contains("layers.1.mixer.switch_mlp.fc2.scales"))
    }

    func testDiscardRestoresHeadAndRequestIsolation() throws {
        let assistant = NemotronH35MTPAssistant(target: try model())
        eval(assistant)
        let a = assistant.makeRequestState(), b = assistant.makeRequestState()
        try assistant.configureRequestState(a, maximumSequenceLength: 256)
        let tokens = MLXArray([Int32(7)]).reshaped(1, 1)
        let hidden = MLXArray.ones([1, 1, 64])
        let first = assistant.draftStep(tokens: tokens, hidden: hidden, shortlist: nil, requestState: a)
        eval(assistant.evaluationTargets(for: a) + [first.tokens, first.hidden])
        XCTAssertEqual(a.stagedInputCount, 1)
        XCTAssertEqual(b.materializedBytes, 0)
        assistant.discardRound(requestState: a)
        XCTAssertEqual(a.stagedInputCount, 0)
        try assistant.configureRequestState(b, maximumSequenceLength: 256)
        let repeatA = assistant.draftStep(tokens: tokens, hidden: hidden, shortlist: nil, requestState: a)
        let freshB = assistant.draftStep(tokens: tokens, hidden: hidden, shortlist: nil, requestState: b)
        eval(repeatA.tokens, repeatA.hidden, freshB.tokens, freshB.hidden)
        XCTAssertEqual(first.hidden.asData(access: .copy).data, repeatA.hidden.asData(access: .copy).data)
        XCTAssertEqual(repeatA.hidden.asData(access: .copy).data, freshB.hidden.asData(access: .copy).data)
        assistant.releaseRequestState(a)
        assistant.releaseRequestState(a)
        XCTAssertEqual(a.materializedBytes, 0)
        XCTAssertGreaterThan(b.materializedBytes, 0)
    }

    func testAcceptedDraftIsReplayedWithTargetHiddenAndReleaseDropsOwnership() throws {
        let assistant = NemotronH35MTPAssistant(target: try model())
        let state = assistant.makeRequestState()
        try assistant.configureRequestState(state, maximumSequenceLength: 256)
        XCTAssertTrue(state.hasPendingPrefillForCostAccounting)
        let seed = MLXArray([Int32(4)]).reshaped(1, 1)
        let hidden = MLXArray.ones([1, 1, 64])
        let draft = assistant.draftStep(tokens: seed, hidden: hidden, shortlist: nil, requestState: state)
        eval(draft.tokens, draft.hidden, assistant.evaluationTargets(for: state))
        assistant.finalizeRound(requestState: state, confirmedInputTokens: 2,
            committedDraftTokens: draft.tokens.reshaped(1, 1), committedTargetHidden: hidden * 2)
        XCTAssertEqual(state.committedInputCount, 2)
        XCTAssertFalse(state.hasPendingPrefillForCostAccounting)
        let typed = try XCTUnwrap(state as? NemotronH35MTPAssistant.State)
        XCTAssertEqual(typed.pendingHidden.count, 1)
        XCTAssertEqual(typed.cacheOffset, 1)
        let next = assistant.draftStep(tokens: seed, hidden: hidden, shortlist: nil, requestState: state)
        eval(next.tokens, assistant.evaluationTargets(for: state))
        XCTAssertEqual(typed.cacheOffset, 3)
        assistant.discardRound(requestState: state)
        XCTAssertEqual(typed.cacheOffset, 1)
        XCTAssertEqual(typed.pendingHidden.count, 1)
        assistant.releaseRequestState(state)
        XCTAssertEqual(state.materializedBytes, 0)
    }

    func testOptimizedVerificationAndDepthAreDefault() throws {
        let assistant = NemotronH35MTPAssistant(target: try model())
        XCTAssertEqual(assistant.requiredVerificationMode, .rectangularExact)
        XCTAssertEqual(assistant.maximumDraftTokens,
            Int(ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_MTP_MAX_DRAFT_TOKENS"] ?? "7"))
        XCTAssertTrue(assistant.supportsTargetPrefixAcceptance)
        XCTAssertEqual(assistant.mtpTargetIdentity, ObjectIdentifier(assistant.target))
        XCTAssertTrue(assistant.target.cbv2Capabilities.supportsMTP)
        let adapter = CBv2SteppableLanguageModelAdapter(assistant.target)
        XCTAssertNotNil(CBv2MTPRoundDriver.build(model: adapter, drafter: assistant,
            config: CBv2MTPConfig(enabled: true)))
        let top = assistant.target.cbv2MTPTopTwo(MLXArray([Float(1), 3, 2]).reshaped(1, 1, 3))
        eval(top.ids, top.values)
        XCTAssertEqual(top.ids.asArray(Int32.self), [1, 2])
        XCTAssertEqual(top.values.asArray(Float.self), [3, 2])
    }

    func testKVOnlyHistoryPreservesCacheAndNextHeadOutput() throws {
        let target = try model()
        let head = NemotronH35MTPModule(target.configuration)
        quantize(model: head, groupSize: 32, bits: 4)
        let full = KVCacheSimple(), kvOnly = KVCacheSimple()
        let hidden = MLXArray.ones([1, 1, 64])
        let embedding = target.mtpEmbedding(MLXArray([Int32(7)]).reshaped(1, 1))
        for step in 1...7 {
            let h = hidden * Float(step)
            let discarded = head(hidden: h, embedding: embedding, cache: full)
            let appended = head.appendTrustedKV(hidden: h, embedding: embedding, cache: kvOnly)
            eval([discarded] + appended + full.innerState() + kvOnly.innerState())
            XCTAssertEqual(full.offset, kvOnly.offset)
            for (a, b) in zip(full.state, kvOnly.state) {
                XCTAssertEqual(a.asData(access: .copy).data, b.asData(access: .copy).data)
            }
        }
        let a = head(hidden: hidden, embedding: embedding, cache: full)
        let b = head(hidden: hidden, embedding: embedding, cache: kvOnly)
        eval(a, b)
        XCTAssertEqual(a.asData(access: .copy).data, b.asData(access: .copy).data)
    }

    func testSevenDraftChainRestoresAuthoritativeHistoryForEveryAcceptanceLength() throws {
        for accepted in 0...7 {
            let assistant = NemotronH35MTPAssistant(target: try model(), maximumDraftTokens: 7)
            let requestState = assistant.makeRequestState()
            try assistant.configureRequestState(requestState, maximumSequenceLength: 256)
            let state = try XCTUnwrap(requestState as? NemotronH35MTPAssistant.State)
            var token = MLXArray([Int32(7)]).reshaped(1, 1)
            var hidden = MLXArray.ones([1, 1, 64])
            for _ in 0..<7 {
                let result = assistant.draftStep(tokens: token, hidden: hidden, shortlist: nil, requestState: state)
                token = result.tokens.reshaped(1, 1)
                hidden = result.hidden
            }
            eval(assistant.evaluationTargets(for: state))
            XCTAssertEqual(state.stagedInputCount, 7)
            XCTAssertEqual(state.cacheOffset, 7)
            assistant.finalizeRound(requestState: state, confirmedInputTokens: 1 + accepted,
                committedDraftTokens: MLXArray(Array([Int32(8), 9, 10, 11, 12, 13, 14].prefix(accepted))).reshaped(1, accepted),
                committedTargetHidden: MLXArray.ones([1, accepted, 64]))
            XCTAssertEqual(state.cacheOffset, 1)
            XCTAssertEqual(state.committedInputCount, 1 + accepted)
            let next = assistant.draftStep(tokens: token, hidden: hidden, shortlist: nil, requestState: state)
            eval([next.tokens] + assistant.evaluationTargets(for: state))
            XCTAssertEqual(state.cacheOffset, 2 + accepted)
            assistant.discardRound(requestState: state)
            XCTAssertEqual(state.cacheOffset, 1)
            XCTAssertEqual(state.pendingTokens.reduce(0) { $0 + $1.dim(1) }, accepted)
            assistant.releaseRequestState(state)
            XCTAssertEqual(state.materializedBytes, 0)
        }
    }
}
