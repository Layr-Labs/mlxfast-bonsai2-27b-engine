import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class NemotronH35MTPPrimingTests: XCTestCase {
    private func assistant(_ dtype: DType) throws -> NemotronH35MTPAssistant {
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
        MLXRandom.seed(35003512)
        let target = NemotronH35Model(try JSONDecoder().decode(NemotronH35Configuration.self, from: data))
        let cast = target.castPredicate
        target.update(parameters: ModuleParameters.unflattened(target.parameters().flattened().map {
            ($0.0, cast?($0.0) == false ? $0.1 : $0.1.asType(dtype))
        }))
        quantize(model: target, groupSize: 32, bits: 4)
        let assistant = NemotronH35MTPAssistant(target: target, maximumDraftTokens: 3)
        assistant.module.update(parameters: ModuleParameters.unflattened(assistant.module.parameters().flattened().map {
            ($0.0, $0.1.asType(dtype))
        }))
        quantize(model: assistant.module, groupSize: 32, bits: 4)
        eval(assistant)
        return assistant
    }

    func testFullAndChunkedPromptPrimeMatchFullModuleKVAndFirstDraft() throws {
        for dtype: DType in [.bfloat16, .float32] {
            for chunks in [[7], [3, 4]] {
                let a = try assistant(dtype)
                let state = try XCTUnwrap(a.makeRequestState() as? NemotronH35MTPAssistant.State)
                let untouched = try XCTUnwrap(a.makeRequestState() as? NemotronH35MTPAssistant.State)
                try a.configureRequestState(state, maximumSequenceLength: 256)
                let tokens = MLXArray([Int32(5), 8, 13, 21, 34, 55, 89]).reshaped(1, 7)
                let hidden = (MLXArray(0..<448).asType(.float32).reshaped(1, 7, 64) * 0.01).asType(dtype)
                let reference = KVCacheSimple()
                var start = 0
                for count in chunks {
                    let end = start + count
                    a.observeCommittedTarget(.init(tokens: tokens[0..., start..<end],
                        hidden: hidden[0..., start..<end, 0...]), requestState: state)
                    start = end
                }
                let primedReference = a.module(
                    hidden: hidden[0..., 0..<6, 0...],
                    embedding: a.target.mtpEmbedding(tokens[0..., 1..<7]), cache: reference)
                eval([primedReference] + reference.innerState())
                XCTAssertEqual(state.pendingTokens.reduce(0) { $0 + $1.dim(1) }, 6)
                XCTAssertEqual(state.committedInputCount, 7)
                XCTAssertEqual(untouched.cacheOffset, 0)
                XCTAssertEqual(untouched.materializedBytes, 0)
                let seed = MLXArray([Int32(100)]).reshaped(1, 1)
                let frontier = hidden[0..., 6..<7, 0...]
                let expected = a.module(hidden: frontier, embedding: a.target.mtpEmbedding(seed), cache: reference)
                eval([expected] + reference.innerState())
                let actual = a.draftStep(tokens: seed, hidden: frontier, shortlist: nil, requestState: state)
                eval([actual.tokens, actual.hidden] + a.evaluationTargets(for: state))
                XCTAssertEqual(state.cacheOffset, 7)
                XCTAssertEqual(actual.hidden.dtype, dtype)
                XCTAssertTrue(allClose(actual.hidden, expected, rtol: 1e-4, atol: 1e-4)
                    .item(Bool.self))
                let expectedToken = argMax(a.target.logits(expected)[0..., -1, 0...], axis: -1)
                    .asType(.int32)
                eval(expectedToken)
                XCTAssertEqual(actual.tokens.asData(access: .copy).data,
                    expectedToken.asData(access: .copy).data)
                for (left, right) in zip(state.cacheSnapshot, reference.state) {
                    XCTAssertEqual(left.dtype, right.dtype)
                    XCTAssertEqual(left.asData(access: .copy).data, right.asData(access: .copy).data)
                }
                let firstBytes = actual.hidden.asData(access: .copy).data
                a.discardRound(requestState: state)
                XCTAssertEqual(state.cacheOffset, 0)
                XCTAssertEqual(state.pendingTokens.reduce(0) { $0 + $1.dim(1) }, 6)
                let retry = a.draftStep(tokens: seed, hidden: frontier, shortlist: nil, requestState: state)
                eval([retry.hidden] + a.evaluationTargets(for: state))
                XCTAssertEqual(retry.hidden.asData(access: .copy).data, firstBytes)
                a.finalizeRound(requestState: state, confirmedInputTokens: 1,
                    committedDraftTokens: MLXArray.zeros([1, 0], dtype: .int32),
                    committedTargetHidden: MLXArray.zeros([1, 0, 64], dtype: dtype))
                XCTAssertEqual(state.cacheOffset, 7)
                a.releaseRequestState(state)
                a.releaseRequestState(untouched)
                XCTAssertEqual(state.materializedBytes, 0)
                XCTAssertEqual(untouched.materializedBytes, 0)
            }
        }
    }

    func testMaterializedBytesChargesWholeFrontierBackingWithoutEvaluation() throws {
        let a = try assistant(.bfloat16)
        let state = try XCTUnwrap(a.makeRequestState() as? NemotronH35MTPAssistant.State)
        let parent = MLXArray.ones([1, 1024, 64], dtype: .bfloat16)
        let view = parent[0..., 1023..<1024, 0...]
        eval(view)
        let info = try XCTUnwrap(view.evaluatedBufferInfo())
        XCTAssertGreaterThan(info.allocatedBytes, view.nbytes)
        state.frontier = view
        XCTAssertEqual(state.materializedBytes, info.allocatedBytes)
        a.releaseRequestState(state)
        XCTAssertEqual(state.materializedBytes, 0)
    }

    func testReleaseBeforeFirstDraftDropsCompletePromptBacklog() throws {
        let a = try assistant(.bfloat16)
        let state = a.makeRequestState()
        a.observeCommittedTarget(.init(tokens: MLXArray([Int32(4), 7, 9]).reshaped(1, 3),
            hidden: MLXArray.ones([1, 3, 64], dtype: .bfloat16)), requestState: state)
        XCTAssertGreaterThan(state.materializedBytes, 0)
        a.releaseRequestState(state)
        XCTAssertEqual(state.materializedBytes, 0)
    }

    func testPagedPrefixCheckpointRestoresColdFirstDraftExactly() throws {
        for dtype: DType in [.bfloat16, .float32] {
            let a = try assistant(dtype)
            let donor = a.makeRequestState()
            let tokens = MLXArray([Int32(5), 8, 13, 21]).reshaped(1, 4)
            let hidden = (MLXArray(0..<256).asType(.float32).reshaped(1, 4, 64) * 0.01)
                .asType(dtype)
            a.observeCommittedTarget(.init(tokens: tokens, hidden: hidden), requestState: donor)
            let checkpoint = try XCTUnwrap(a.capturePrefixCheckpoint(
                requestState: donor, targetInputCount: 4))
            let encoded = try XCTUnwrap(a.encodePrefixCheckpoint(checkpoint))
            eval(encoded)
            let restoredCheckpoint = try XCTUnwrap(a.decodePrefixCheckpoint(
                tensors: encoded, prefixTokens: [5, 8, 13, 21]))
            let warm = try XCTUnwrap(a.restorePrefixCheckpoint(restoredCheckpoint))
            try a.configureRequestState(warm, maximumSequenceLength: 256)
            try a.configureRequestState(donor, maximumSequenceLength: 256)
            let seed = MLXArray([Int32(34)]).reshaped(1, 1)
            let frontier = hidden[0..., 3..<4, 0...]
            let coldResult = a.draftStep(
                tokens: seed, hidden: frontier, shortlist: nil, requestState: donor)
            let warmResult = a.draftStep(
                tokens: seed, hidden: frontier, shortlist: nil, requestState: warm)
            eval([coldResult.tokens, coldResult.hidden, warmResult.tokens, warmResult.hidden]
                + a.evaluationTargets(for: donor) + a.evaluationTargets(for: warm))
            XCTAssertEqual(coldResult.tokens.asData(access: .copy).data,
                warmResult.tokens.asData(access: .copy).data)
            XCTAssertEqual(coldResult.hidden.asData(access: .copy).data,
                warmResult.hidden.asData(access: .copy).data)
            XCTAssertNil(a.capturePrefixCheckpoint(requestState: donor, targetInputCount: 4))
            XCTAssertNil(a.decodePrefixCheckpoint(
                tensors: Array(encoded.dropLast()), prefixTokens: [5, 8, 13, 21]))
            XCTAssertNil(a.decodePrefixCheckpoint(
                tensors: encoded, prefixTokens: [5, 8, 13, 22]))
            a.discardRound(requestState: donor)
            a.discardRound(requestState: warm)
            a.releaseRequestState(donor)
            a.releaseRequestState(warm)
            XCTAssertEqual(donor.materializedBytes, 0)
            XCTAssertEqual(warm.materializedBytes, 0)
        }
    }
}
