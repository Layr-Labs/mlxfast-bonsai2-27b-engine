import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen35MoECompleteCheckpointTests: XCTestCase {
    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }

    private func configuration() throws -> Qwen35Configuration {
        try JSONDecoder().decode(Qwen35Configuration.self, from: Data("""
            {
              "model_type": "qwen3_5_moe",
              "text_config": {
                "model_type": "qwen3_5_moe_text", "hidden_size": 64,
                "num_hidden_layers": 4, "intermediate_size": 32,
                "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16,
                "linear_num_value_heads": 1, "linear_num_key_heads": 1,
                "linear_key_head_dim": 64, "linear_value_head_dim": 64,
                "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
                "vocab_size": 64, "num_experts": 4, "num_experts_per_tok": 2,
                "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                "norm_topk_prob": true
              }
            }
            """.utf8))
    }

    private func model(dtype: DType) throws -> Qwen35MoEModel {
        MLXRandom.seed(4817)
        let model = Qwen35MoEModel(try configuration())
        model.update(parameters: ModuleParameters.unflattened(
            model.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
        // Exercise the packed-U32 embedding with a native floating activation.
        quantize(model: model, groupSize: 32, bits: 4) { _, module in module is Embedding }
        eval(model)
        return model
    }

    private func engine(
        model: Qwen35MoEModel, dtype: DType, store: CompleteCheckpointFixtureStore?
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        let kinds = model.cbv2LayerKinds
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 64 << 20, kvDType: dtype))
        let caches = model.newCacheV2 { index, kind in CBv2LayerCache(layerIndex: index, kind: kind) }
        return (EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store), backend)
    }

    func testMoETopologyRetainsExactBackendAndActivationGates() throws {
        for dtype: DType in [.float32, .bfloat16, .float16] {
            let model = try model(dtype: dtype)
            XCTAssertTrue(model.cbv2Capabilities.supportsRecurrentCheckpointReuse)
            XCTAssertFalse(model.cbv2Capabilities.supportsPrefixReuse, "KV alone cannot restore recurrent layers")
            XCTAssertTrue(model.cbv2Capabilities.supportsPagedKV)
            XCTAssertTrue(model.cbv2Capabilities.requiresNativePagedKV)
            XCTAssertEqual(model.cbv2LayerKinds.map(\.modelLayerIndex), [1, 3])
            XCTAssertEqual(model.cbv2CompleteCheckpointKVDTypes, [dtype, dtype])
            XCTAssertEqual(model.cbv2RecurrentStateSpec.layers.map(\.convDType), [dtype, dtype])
            XCTAssertEqual(model.cbv2RecurrentStateSpec.layers.map(\.ssmDType), [.float32, .float32])
            let embedding = try XCTUnwrap(model.languageModel.model.embedTokens as? QuantizedEmbedding)
            let activation = embedding(MLXArray([Int32(1), 2]).reshaped([1, 2]))
            eval(activation)
            XCTAssertEqual(embedding.weight.dtype, .uint32)
            XCTAssertEqual(activation.dtype, dtype)
            let biases = try XCTUnwrap(embedding.biases)
            let other: DType = dtype == .float16 ? .float32 : .float16
            embedding.update(parameters: ModuleParameters.unflattened(["biases": biases.asType(other)]))
            XCTAssertNil(model.cbv2CompleteCheckpointKVDTypes, "mixed affine metadata stays ineligible")
        }
    }

    func testMoEEncodedStateRestoresIntoFreshEngineAcrossSuffixAndBranchShapes() async throws {
        for dtype: DType in [.float32, .bfloat16, .float16] {
            let donorModel = try model(dtype: dtype)
            let store = CompleteCheckpointFixtureStore()
            let (donor, donorBackend) = engine(model: donorModel, dtype: dtype, store: store)
            let prompt = (0 ..< 2 * chunk + 7).map { 1 + ($0 * 7) % 61 }
            let request = CBv2Request(id: .init(1), promptTokens: prompt, maxTokens: 6,
                                      cacheSalt: "tenant", prefixCacheReceiptID: .init(1001))
            let donated = await cbv2SchedCollect(try donor.submit(request))
            XCTAssertEqual(donated.finishReason, .length)
            XCTAssertEqual(store.saved.map(\.manifest.position), [chunk, 2 * chunk])
            XCTAssertNil(donor.hybridPrefixCache)
            XCTAssertEqual(donor.admissionForTesting.bytesReserved, 0)
            XCTAssertEqual(donorBackend.bytesReserved, 0)
            await donor.shutdown()

            // Reopen only encoded manifests/bytes with a new model and engine;
            // no recurrent, attention, or assistant array is carried by the store.
            let restoredStore = CompleteCheckpointFixtureStore(archives: store.saved)
            let (warm, warmBackend) = engine(model: try model(dtype: dtype), dtype: dtype, store: restoredStore)
            let (cold, coldBackend) = engine(model: try model(dtype: dtype), dtype: dtype, store: nil)
            var branch = prompt
            branch[chunk + 3] = (branch[chunk + 3] % 61) + 1
            let cases = [(prompt, 2 * chunk), (Array(prompt.prefix(2 * chunk + 1)), 2 * chunk), (branch, chunk)]
            for (index, pair) in cases.enumerated() {
                let id = CBv2RequestID(UInt64(10 + index))
                let request = CBv2Request(id: id, promptTokens: pair.0, maxTokens: 6,
                    cacheSalt: "tenant", prefixCacheReceiptID: .init(UInt64(2000 + index)))
                let expected = await cbv2SchedCollect(try cold.submit(request))
                XCTAssertTrue(try restoredStore.stage(engine: warm, request: request))
                let actual = await cbv2SchedCollect(try warm.submit(request))
                XCTAssertEqual(actual.tokens, expected.tokens, "MoE cold/restored IDs differ for \(dtype), case \(index)")
                XCTAssertEqual(actual.finishReason, .length)
                XCTAssertEqual(actual.usage?.prefixCachePrefillTokensSaved, pair.1)
                XCTAssertEqual(actual.usage?.prefixCacheReplayTokens, 0)
                XCTAssertEqual(actual.usage?.prefixCacheTier, .snapshot)
                XCTAssertEqual(warm.admissionForTesting.bytesReserved, 0)
                XCTAssertEqual(warmBackend.bytesReserved, 0)
                XCTAssertEqual(cold.admissionForTesting.bytesReserved, 0)
                XCTAssertEqual(coldBackend.bytesReserved, 0)
                if index == 0 { XCTAssertEqual(actual.tokens, donated.tokens) }
            }
            let foreign = CBv2Request(id: .init(99), promptTokens: prompt, maxTokens: 6,
                cacheSalt: "another-tenant", prefixCacheReceiptID: .init(2099))
            XCTAssertFalse(try restoredStore.stage(engine: warm, request: foreign))
            var disabled = request
            disabled.prefixCacheEnabled = false
            let manifest = try XCTUnwrap(restoredStore.saved.first).manifest
            XCTAssertThrowsError(try warm.planCompleteCheckpointImport(
                manifest: manifest, request: disabled))
            var positioned = request
            positioned.positionState = .init(
                promptPositionIds: MLXArray.zeros([3, 1, prompt.count], dtype: .int32), decodeDeltas: [0])
            XCTAssertThrowsError(try warm.planCompleteCheckpointImport(manifest: manifest, request: positioned))
            var visual = request
            visual.multimodal = .init(spans: [], attention: .causal) {
                XCTFail("checkpoint planning must not evaluate visual input")
                return []
            }
            XCTAssertThrowsError(try warm.planCompleteCheckpointImport(manifest: manifest, request: visual))
            XCTAssertEqual(warm.admissionForTesting.bytesReserved, 0)
            await warm.shutdown()
            await cold.shutdown()
        }
    }
}
