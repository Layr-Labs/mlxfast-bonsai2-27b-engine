// Copyright © 2025 Apple Inc.

import Foundation
import MLX
@testable import MLXLLM
@testable import MLXLMCommon
import MLXNN
import XCTest

public class NemotronHTests: XCTestCase {

    func testLightningCheckpointTypesMatchActualCachesWithoutChangingOutput() throws {
        let config = try makeTestLightningConfiguration()
        let model = NemotronH35Model(config)
        eval(model)
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let caches = model.newCache(parameters: nil)
        let before = model(tokens, cache: caches)
        eval(before, caches)
        let types = try XCTUnwrap(model.cbv2CompleteCheckpointKVDTypes)
        var expected: [DType] = []
        var cacheIndex = 0
        for block in config.target.hybridOverridePattern {
            if block == "M" || block == "*" {
                if block == "*" {
                    XCTAssertEqual(caches[cacheIndex].state[0].dtype, caches[cacheIndex].state[1].dtype)
                    expected.append(caches[cacheIndex].state[0].dtype)
                }
                cacheIndex += 1
            }
        }
        XCTAssertEqual(types, expected)
        XCTAssertEqual(types.count, model.cbv2LayerKinds.count)
        let after = model(tokens, cache: model.newCache(parameters: nil))
        eval(after)
        XCTAssertTrue(all(before .== after).item(Bool.self))
        XCTAssertTrue(model.cbv2Capabilities.supportsRecurrentCheckpointReuse)
    }

    /// Create a minimal test configuration for NemotronH
    /// Uses small dimensions to keep tests fast
    private func makeTestConfig(pattern: String = "M*M-E") -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: pattern.count,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: pattern,
            layerNormEpsilon: 1e-5,
            nGroup: 2,
            topkGroup: 1
        )
    }

    private func makeTestLightningConfiguration() throws
        -> NemotronH35Configuration
    {
        let json = """
            {
              "model_type": "nemotron_h",
              "vocab_size": 100,
              "hidden_size": 64,
              "num_hidden_layers": 4,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 64,
              "mamba_num_heads": 4,
              "mamba_head_dim": 16,
              "ssm_state_size": 16,
              "conv_kernel": 4,
              "n_groups": 2,
              "intermediate_size": 128,
              "moe_intermediate_size": 64,
              "moe_shared_expert_intermediate_size": 64,
              "n_routed_experts": 4,
              "num_experts_per_tok": 2,
              "layers_block_type": ["mamba", "moe", "mamba", "attention"],
              "norm_eps": 0.00001,
              "mamba_ssm_cache_dtype": "float32"
            }
            """
        return try JSONDecoder().decode(
            NemotronH35Configuration.self,
            from: Data(json.utf8))
    }

    // MARK: - Configuration Decoding Tests

    func testLightningTimestepMetadataAndExplicitLimitsRoundTrip() throws {
        let base = try makeTestLightningConfiguration()
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as? [String: Any])
        json["time_step_min"] = 0.001
        json["time_step_max"] = 0.1
        let metadata = try JSONDecoder().decode(NemotronH35Configuration.self,
            from: JSONSerialization.data(withJSONObject: json))
        XCTAssertFalse(metadata.hasExplicitExecutionLimits)
        XCTAssertEqual(NemotronH35Model(metadata).configuration.timeStepLimitMin, 0)
        XCTAssertEqual(NemotronH35Model(metadata).configuration.timeStepLimitMax, .infinity)
        let reopened = try JSONDecoder().decode(NemotronH35Configuration.self,
            from: JSONEncoder().encode(metadata))
        XCTAssertFalse(reopened.hasExplicitExecutionLimits)
        XCTAssertEqual(reopened.target.timeStepLimitMin, Float(0.001))
        XCTAssertEqual(NemotronH35Model(reopened).configuration.timeStepLimitMax, .infinity)

        json["time_step_limit"] = [0.2, 0.8]
        let explicit = try JSONDecoder().decode(NemotronH35Configuration.self,
            from: JSONSerialization.data(withJSONObject: json))
        let roundTrip = try JSONDecoder().decode(NemotronH35Configuration.self,
            from: JSONEncoder().encode(explicit))
        XCTAssertTrue(roundTrip.hasExplicitExecutionLimits)
        XCTAssertEqual(NemotronH35Model(roundTrip).configuration.timeStepLimitMin, Float(0.2))
        XCTAssertEqual(NemotronH35Model(roundTrip).configuration.timeStepLimitMax, Float(0.8))
    }

    func testMalformedTimestepLimitsThrowInsteadOfTrappingOrFallingBack() throws {
        let base = try makeTestLightningConfiguration()
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as? [String: Any])
        for value: Any in [[], [0, 1, 2], [1, 0], [-1, 1], "invalid"] {
            var json = object
            json["time_step_limit"] = value
            let data = try JSONSerialization.data(withJSONObject: json)
            XCTAssertThrowsError(try JSONDecoder().decode(NemotronH35Configuration.self, from: data))
        }
        var legacy = object
        legacy["time_step_limit_min"] = []
        XCTAssertThrowsError(try JSONDecoder().decode(NemotronHConfiguration.self,
            from: JSONSerialization.data(withJSONObject: legacy)))
    }

    func testConfigurationDecodingFromJSON() throws {
        let json = """
            {
                "model_type": "nemotron_h",
                "vocab_size": 131072,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "mamba_num_heads": 64,
                "mamba_head_dim": 64,
                "ssm_state_size": 128,
                "conv_kernel": 4,
                "n_groups": 8,
                "intermediate_size": 16384,
                "moe_intermediate_size": 1024,
                "moe_shared_expert_intermediate_size": 8192,
                "n_routed_experts": 64,
                "num_experts_per_tok": 4,
                "hybrid_override_pattern": "M*M-E*",
                "layer_norm_epsilon": 1e-5,
                "n_group": 4,
                "topk_group": 2
            }
            """

        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.vocabSize, 131072)
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.numHiddenLayers, 32)
        XCTAssertEqual(config.numAttentionHeads, 32)
        XCTAssertEqual(config.numKeyValueHeads, 8)
        XCTAssertEqual(config.mambaNumHeads, 64)
        XCTAssertEqual(config.mambaHeadDim, 64)
        XCTAssertEqual(config.ssmStateSize, 128)
        XCTAssertEqual(config.convKernel, 4)
        XCTAssertEqual(config.nGroups, 8)
        XCTAssertEqual(config.intermediateSize, 16384)
        XCTAssertEqual(config.moeIntermediateSize, 1024)
        XCTAssertEqual(config.nRoutedExperts, 64)
        XCTAssertEqual(config.numExpertsPerTok, 4)
        XCTAssertEqual(config.hybridOverridePattern, "M*M-E*")
        XCTAssertEqual(config.nGroup, 4)
        XCTAssertEqual(config.topkGroup, 2)
    }

    func testConfigurationDecodingWithArrayPattern() throws {
        // Some configs have hybrid_override_pattern as array of strings
        let json = """
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "hybrid_override_pattern": ["M", "*", "M", "-"]
            }
            """

        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.hybridOverridePattern, "M*M-")
    }

    func testNemotron35LightningConfigurationDecoding() async throws {
        let json = """
            {
                "model_type": "nemotron_h",
                "vocab_size": 131072,
                "hidden_size": 2688,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "mamba_num_heads": 64,
                "mamba_head_dim": 64,
                "ssm_state_size": 128,
                "conv_kernel": 4,
                "n_groups": 8,
                "intermediate_size": 1856,
                "moe_intermediate_size": 1856,
                "moe_shared_expert_intermediate_size": 3712,
                "n_routed_experts": 128,
                "n_shared_experts": 1,
                "num_experts_per_tok": 6,
                "layers_block_type": ["mamba", "moe", "attention", "mlp"],
                "norm_eps": 0.00001,
                "time_step_min": 0.001,
                "time_step_max": 0.1,
                "chunk_size": 128,
                "mamba_ssm_cache_dtype": "float32",
                "num_nextn_predict_layers": 1,
                "mtp_layers_block_type": ["attention", "moe"]
            }
            """

        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.hybridOverridePattern, "ME*-")
        XCTAssertEqual(config.layerNormEpsilon, 1e-5)
        XCTAssertEqual(config.timeStepLimitMin, 0.001)
        XCTAssertEqual(config.timeStepLimitMax, 0.1)
        XCTAssertEqual(config.chunkSize, 128)
        XCTAssertEqual(config.mambaSSMCacheDType, "float32")
        XCTAssertEqual(config.numNextnPredictLayers, 1)
        XCTAssertEqual(config.mtpLayersBlockType, ["attention", "moe"])

        let lightning = try JSONDecoder().decode(
            NemotronH35Configuration.self, from: json.data(using: .utf8)!)
        XCTAssertEqual(lightning.target.hybridOverridePattern, "ME*-")

        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: json.data(using: .utf8)!,
            modelType: "nemotron_h")
        XCTAssertTrue(model is NemotronH35Model)
        let lightningModel = try XCTUnwrap(model as? NemotronH35Model)
        XCTAssertEqual(
            lightningModel.lightningConfiguration.target.timeStepLimitMin,
            0.001)
        XCTAssertFalse(lightningModel.cbv2Capabilities.supportsPrefixReuse)
        XCTAssertTrue(lightningModel.cbv2Capabilities.supportsRecurrentCheckpointReuse)
        let capabilityProvider =
            lightningModel as any CBv2ModelCapabilityProviding
        XCTAssertTrue(
            capabilityProvider.cbv2Capabilities.supportsRecurrentCheckpointReuse)

        let roundTripModel = try await LLMTypeRegistry.shared.createModel(
            configuration: JSONEncoder().encode(lightning),
            modelType: "nemotron_h")
        XCTAssertTrue(roundTripModel is NemotronH35Model)
    }

    func testLegacyNemotronRegistryStillSelectsNanoModel() async throws {
        let config = NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: "M*",
            timeStepLimitMax: 100)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: JSONEncoder().encode(config),
            modelType: "nemotron_h")
        XCTAssertTrue(model is NemotronHModel)
        XCTAssertFalse(model is NemotronH35Model)
    }

    func testNemotron35RejectsUnknownBlockType() throws {
        let json = """
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "layers_block_type": ["unsupported"]
            }
            """

        XCTAssertThrowsError(
            try JSONDecoder().decode(
                NemotronHConfiguration.self, from: json.data(using: .utf8)!))
    }

    func testNemotron35CBv2StateAndAttentionLayout() throws {
        let config = makeTestConfig(pattern: "M*ME")
        let layerKinds = config.cbv2LayerKinds
        XCTAssertEqual(layerKinds.count, 1)
        XCTAssertEqual(layerKinds[0].modelLayerIndex, 1)
        XCTAssertEqual(layerKinds[0].headDim, 16)
        XCTAssertEqual(layerKinds[0].kvHeads, 2)
        XCTAssertEqual(layerKinds[0].queryHeads, 4)

        let state = config.cbv2RecurrentStateSpec(activationDType: .bfloat16)
        XCTAssertEqual(state.modelLayerIndices, [0, 2])
        XCTAssertEqual(state.layers[0].convShape, [1, 3, 128])
        XCTAssertEqual(state.layers[0].convDType, .bfloat16)
        XCTAssertEqual(state.layers[0].ssmShape, [1, 4, 16, 16])
        XCTAssertEqual(state.layers[0].ssmDType, .float32)
        XCTAssertEqual(try state.fixedBytesPerRequest(), 9_728)

        XCTAssertFalse(config.cbv2Capabilities.supportsPrefixReuse)
        XCTAssertFalse(config.cbv2Capabilities.supportsRecurrentCheckpointReuse)
        XCTAssertFalse(config.cbv2Capabilities.supportsPagedKV)
        XCTAssertFalse(config.cbv2Capabilities.supportsPackedPrefill)
        XCTAssertFalse(config.cbv2Capabilities.supportsMTP)
    }

    func testSelectedLightningArtifactGeometry() throws {
        let pattern = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        let config = NemotronHConfiguration(
            vocabSize: 131_072,
            hiddenSize: 2_688,
            numHiddenLayers: 52,
            numAttentionHeads: 32,
            numKeyValueHeads: 2,
            mambaNumHeads: 64,
            mambaHeadDim: 64,
            ssmStateSize: 128,
            convKernel: 4,
            nGroups: 8,
            intermediateSize: 1_856,
            moeIntermediateSize: 1_856,
            moeSharedExpertIntermediateSize: 3_712,
            nRoutedExperts: 128,
            numExpertsPerTok: 6,
            hybridOverridePattern: pattern,
            headDim: 128,
            nSharedExperts: 1,
            routedScalingFactor: 2.5,
            timeStepLimitMin: 0.001,
            timeStepLimitMax: 0.1,
            numNextnPredictLayers: 1,
            mtpLayersBlockType: ["attention", "moe"])

        XCTAssertEqual(pattern.count, 52)
        XCTAssertEqual(pattern.filter { $0 == "M" }.count, 23)
        XCTAssertEqual(pattern.filter { $0 == "E" }.count, 23)
        XCTAssertEqual(
            config.cbv2LayerKinds.compactMap(\.modelLayerIndex),
            [5, 12, 19, 26, 33, 42])

        let recurrent = config.cbv2RecurrentStateSpec()
        XCTAssertEqual(recurrent.modelLayerIndices.count, 23)
        XCTAssertEqual(recurrent.layers[0].convShape, [1, 3, 6_144])
        XCTAssertEqual(recurrent.layers[0].ssmShape, [1, 64, 64, 128])
        XCTAssertEqual(try recurrent.fixedBytesPerRequest(), 49_082_368)
    }

    func testNemotron35CBv2MambaStagesRequestOwnedState() throws {
        let model = NemotronHModel(makeTestConfig(pattern: "M"))
        let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
        let serial = model(tokens, cache: model.newCache(parameters: nil))
        let state = try CBv2RecurrentRequestState(
            spec: model.cbv2RecurrentStateSpec)
        let binding = try state.bind()
        let logits = model.cbv2Forward(
            tokens,
            caches: [],
            recurrentState: [binding])
        let roots = try binding.evaluate()
        eval([serial, logits] + roots)
        try binding.commit()

        XCTAssertEqual(logits.shape, [1, 3, 100])
        XCTAssertTrue(arrayEqual(serial, logits).item(Bool.self))
        let committed = try XCTUnwrap(state.state(modelLayerIndex: 0))
        XCTAssertEqual(committed.conv?.shape, [1, 3, 128])
        XCTAssertEqual(committed.ssm?.shape, [1, 4, 16, 16])
        XCTAssertEqual(committed.ssm?.dtype, .float32)

        let next = try state.bind()
        let nextLogits = model.cbv2Forward(
            MLXArray([4]).reshaped(1, 1),
            caches: [],
            recurrentState: [next])
        eval([nextLogits] + (try next.evaluate()))
        try next.commit()
        XCTAssertEqual(nextLogits.shape, [1, 1, 100])
    }

    func testNemotron35CBv2ChunkedPrefillMatchesOneShot() throws {
        let model = NemotronHModel(makeTestConfig(pattern: "M"))

        let oneShotState = try CBv2RecurrentRequestState(
            spec: model.cbv2RecurrentStateSpec)
        let oneShotBinding = try oneShotState.bind()
        let oneShot = model.cbv2Forward(
            MLXArray([1, 2, 3]).reshaped(1, 3),
            caches: [],
            recurrentState: [oneShotBinding])
        eval([oneShot] + (try oneShotBinding.evaluate()))
        try oneShotBinding.commit()

        let chunkedState = try CBv2RecurrentRequestState(
            spec: model.cbv2RecurrentStateSpec)
        let firstBinding = try chunkedState.bind()
        _ = model.cbv2Forward(
            MLXArray([1, 2]).reshaped(1, 2),
            caches: [],
            recurrentState: [firstBinding])
        eval(try firstBinding.evaluate())
        try firstBinding.commit()
        let secondBinding = try chunkedState.bind()
        let chunked = model.cbv2Forward(
            MLXArray([3]).reshaped(1, 1),
            caches: [],
            recurrentState: [secondBinding])
        eval([chunked] + (try secondBinding.evaluate()))
        try secondBinding.commit()

        XCTAssertEqual(
            oneShot[0, -1].argMax().item(Int.self),
            chunked[0, -1].argMax().item(Int.self))
        let oneShotFinal = try XCTUnwrap(
            oneShotState.state(modelLayerIndex: 0))
        let chunkedFinal = try XCTUnwrap(
            chunkedState.state(modelLayerIndex: 0))
        XCTAssertEqual(oneShotFinal.conv?.dtype, chunkedFinal.conv?.dtype)
        XCTAssertEqual(oneShotFinal.ssm?.dtype, .float32)
        XCTAssertEqual(chunkedFinal.ssm?.dtype, .float32)
    }

    func testNemotron35HybridEngineDonatesExactRecurrentState() async throws {
        let model = NemotronH35Model(
            try makeTestLightningConfiguration())
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        XCTAssertEqual(model.cbv2RecurrentStateSpec.layers.last?.convDType, .float32)
        let layerKinds = model.cbv2LayerKinds
        let chunk = max(32, CBv2AttentionV1.queryBlockSize)
        func engine(_ store: CompleteCheckpointFixtureStore?) -> (EngineV2, CBv2ContiguousKVBackend) {
            let backend = CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 64 << 20, kvDType: .float32))
            let caches = model.newCacheV2 { index, kind in CBv2LayerCache(layerIndex: index, kind: kind) }
            return (EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model), layerKinds: layerKinds,
                backend: backend, cacheProvider: CBv2LayerCacheBank(caches: caches),
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                    prefillChunkSize: chunk, maxWaiting: 2, enablePrefixCache: store != nil),
                admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store), backend)
        }
        let store = CompleteCheckpointFixtureStore()
        let (donor, donorBackend) = engine(store)
        let prompt = (0..<2 * chunk + 7).map { 1 + ($0 * 7) % 97 }
        let request = CBv2Request(id: .init(3501), promptTokens: prompt, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(4501))
        let donated = await cbv2SchedCollect(try donor.submit(request))
        XCTAssertEqual(donated.finishReason, .length)
        XCTAssertEqual(store.saved.map(\.manifest.position), [chunk, 2 * chunk])
        XCTAssertEqual(donorBackend.bytesReserved, 0)
        await donor.shutdown()

        // Carry encoded bytes only into a new engine, never donor KV/SSM arrays.
        let restoredStore = CompleteCheckpointFixtureStore(archives: store.saved)
        let (warm, warmBackend) = engine(restoredStore)
        let (cold, coldBackend) = engine(nil)
        var branch = prompt
        branch[chunk + 3] = (branch[chunk + 3] % 97) + 1
        for (index, entry) in [(prompt, 2 * chunk), (branch, chunk)].enumerated() {
            let req = CBv2Request(id: .init(UInt64(3502 + index)), promptTokens: entry.0,
                maxTokens: 4, cacheSalt: "tenant", prefixCacheReceiptID: .init(UInt64(4502 + index)))
            let expected = await cbv2SchedCollect(try cold.submit(req))
            XCTAssertTrue(try restoredStore.stage(engine: warm, request: req))
            let actual = await cbv2SchedCollect(try warm.submit(req))
            XCTAssertEqual(actual.tokens, expected.tokens)
            XCTAssertEqual(actual.finishReason, .length)
            XCTAssertEqual(actual.usage?.prefixCachePrefillTokensSaved, entry.1)
            XCTAssertEqual(actual.usage?.prefixCacheTier, .snapshot)
            XCTAssertEqual(warmBackend.bytesReserved, 0)
            XCTAssertEqual(coldBackend.bytesReserved, 0)
        }
        let foreign = CBv2Request(id: .init(3599), promptTokens: prompt, maxTokens: 4,
            cacheSalt: "other-tenant", prefixCacheReceiptID: .init(4599))
        XCTAssertFalse(try restoredStore.stage(engine: warm, request: foreign))
        await warm.shutdown()
        await cold.shutdown()
    }

    func testLightningNativePagedCheckpointRoundTrip() async throws {
        let model = NemotronH35Model(try makeTestLightningConfiguration())
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        let chunk = max(32, CBv2AttentionV1.queryBlockSize)
        let kinds = model.cbv2LayerKinds
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let observed = try CBv2NativeKVTypeProbe.run(model: adapter, layerKinds: kinds,
            caches: model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        XCTAssertEqual(observed.layerDTypes, model.cbv2CompleteCheckpointKVDTypes)

        func engine(_ store: CompleteCheckpointFixtureStore?) throws -> (EngineV2, PagedKVBackend) {
            let backend = try PagedKVBackend(layerKinds: kinds, config: .init(
                capacityBytes: 96 << 20, maxPrefillChunk: chunk, nominalMaxSequenceLength: 512,
                segmentSizeBytes: 64 << 10, layerDTypes: observed.layerDTypes))
            let storage = backend.makeLayerCaches()
            let indices = Dictionary(uniqueKeysWithValues: kinds.enumerated().map {
                ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
            })
            let caches = model.newCacheV2 { index, _ in storage[indices[index]!] }
            return (EngineV2(model: adapter, layerKinds: kinds, backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
                schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                    prefillChunkSize: chunk, maxWaiting: 2, enablePrefixCache: store != nil),
                admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store), backend)
        }
        let store = CompleteCheckpointFixtureStore()
        let (donor, donorBackend) = try engine(store)
        let prompt = (0..<2 * chunk + 7).map { 1 + ($0 * 7) % 97 }
        let req = CBv2Request(id: .init(3601), promptTokens: prompt, maxTokens: 6,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(4601))
        let expected = await cbv2SchedCollect(try donor.submit(req))
        XCTAssertEqual(expected.finishReason, .length)
        XCTAssertEqual(store.saved.map(\.manifest.position), [chunk, 2 * chunk])
        XCTAssertTrue(store.saved.allSatisfy {
            $0.manifest.backendLayout == CBv2CompleteCheckpointManifest.pagedLayout
        })
        XCTAssertEqual(donorBackend.bytesReserved, 0)
        await donor.shutdown()

        let reopened = CompleteCheckpointFixtureStore(archives: store.saved)
        let (warm, warmBackend) = try engine(reopened)
        XCTAssertTrue(try reopened.stage(engine: warm, request: req))
        let actual = await cbv2SchedCollect(try warm.submit(req))
        XCTAssertEqual(actual.tokens, expected.tokens)
        XCTAssertEqual(actual.finishReason, .length)
        XCTAssertEqual(actual.usage?.prefixCachePrefillTokensSaved, 2 * chunk)
        XCTAssertEqual(warmBackend.bytesReserved, 0)
        await warm.shutdown()
    }

    func testConfigurationDecodingWithTimeStepLimitArray() throws {
        // time_step_limit can be an array [min, max]
        let json = """
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "hybrid_override_pattern": "M*",
                "time_step_limit_min": [0.0, 1000.0]
            }
            """

        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(config.timeStepLimitMin, 0.0)
        XCTAssertEqual(config.timeStepLimitMax, 1000.0)
    }

    func testConfigurationDecodingWithDefaults() throws {
        // Minimal config - should use defaults for optional fields
        let json = """
            {
                "vocab_size": 100,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "hybrid_override_pattern": "M*"
            }
            """

        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: json.data(using: .utf8)!)

        // Check defaults
        XCTAssertEqual(config.attentionBias, false)
        XCTAssertEqual(config.mambaProjBias, false)
        XCTAssertEqual(config.mlpBias, false)
        XCTAssertEqual(config.useConvBias, true)
        XCTAssertEqual(config.tieWordEmbeddings, false)
        XCTAssertEqual(config.layerNormEpsilon, 1e-5)
        XCTAssertEqual(config.ropeTheta, 10000.0)
        XCTAssertEqual(config.nGroup, 1)
        XCTAssertEqual(config.topkGroup, 1)
        XCTAssertEqual(config.normTopkProb, true)
        XCTAssertEqual(config.routedScalingFactor, 1.0)
    }

    // MARK: - Weight Sanitization Tests

    func testSanitizeConv1dWeights() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // The sanitization swaps axes 1 and 2 when dim(-1) != 1
        // Python format comes in as [convDim, inputChannels, kernelSize]
        // Swift expects: [convDim, kernelSize, inputChannels]
        let convDim =
            config.mambaNumHeads * config.mambaHeadDim + 2 * config.nGroups * config.ssmStateSize
        // Create weight with shape [convDim, 1, kernelSize] - this has dim(-1) = kernelSize != 1
        let mockConvWeight = MLXArray.ones([convDim, 1, config.convKernel])

        var weights = [String: MLXArray]()
        weights["backbone.layers.0.mixer.conv1d.weight"] = mockConvWeight

        let sanitized = model.sanitize(weights: weights)

        // After swapping axes 1 and 2: [convDim, kernelSize, 1]
        let sanitizedConv = sanitized["backbone.layers.0.mixer.conv1d.weight"]!
        XCTAssertEqual(sanitizedConv.shape, [convDim, config.convKernel, 1])
    }

    func testSanitizeConv1dWeightsNoOpWhenAlreadyCorrect() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // When dim(-1) == 1, no transpose needed
        let convDim =
            config.mambaNumHeads * config.mambaHeadDim + 2 * config.nGroups * config.ssmStateSize
        let mockConvWeight = MLXArray.ones([convDim, config.convKernel, 1])

        var weights = [String: MLXArray]()
        weights["backbone.layers.0.mixer.conv1d.weight"] = mockConvWeight

        let sanitized = model.sanitize(weights: weights)

        // Should remain unchanged [convDim, kernelSize, 1]
        let sanitizedConv = sanitized["backbone.layers.0.mixer.conv1d.weight"]!
        XCTAssertEqual(sanitizedConv.shape, [convDim, config.convKernel, 1])
    }

    func testSanitizeStripsUnboundMTPWeights() throws {
        let model = NemotronHModel(makeTestConfig(pattern: "M*"))
        let weights = [
            "backbone.embeddings.weight": MLXArray.ones([100, 64]),
            "mtp.layers.0.enorm.weight": MLXArray.ones([64]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["backbone.embeddings.weight"])
        XCTAssertNil(sanitized["mtp.layers.0.enorm.weight"])
    }

    func testSanitizeExpertWeights() throws {
        let config = makeTestConfig(pattern: "E")
        let model = NemotronHModel(config)

        // Create mock expert weights that need stacking
        var weights = [String: MLXArray]()
        for e in 0 ..< config.nRoutedExperts {
            weights["backbone.layers.0.mixer.experts.\(e).up_proj.weight"] =
                MLXArray.ones([config.moeIntermediateSize, config.hiddenSize])
            weights["backbone.layers.0.mixer.experts.\(e).down_proj.weight"] =
                MLXArray.ones([config.hiddenSize, config.moeIntermediateSize])
        }

        let sanitized = model.sanitize(weights: weights)

        // Experts should be stacked into switch_mlp format
        let stackedFc1 = sanitized["backbone.layers.0.mixer.switch_mlp.fc1.weight"]
        let stackedFc2 = sanitized["backbone.layers.0.mixer.switch_mlp.fc2.weight"]

        XCTAssertNotNil(stackedFc1)
        XCTAssertNotNil(stackedFc2)
        XCTAssertEqual(
            stackedFc1!.shape,
            [config.nRoutedExperts, config.moeIntermediateSize, config.hiddenSize])
        XCTAssertEqual(
            stackedFc2!.shape,
            [config.nRoutedExperts, config.hiddenSize, config.moeIntermediateSize])

        // Original expert keys should be removed
        XCTAssertNil(sanitized["backbone.layers.0.mixer.experts.0.up_proj.weight"])
    }

    func testSanitizePreservesOtherWeights() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        var weights = [String: MLXArray]()
        weights["backbone.embeddings.weight"] = MLXArray.ones([config.vocabSize, config.hiddenSize])
        weights["backbone.norm_f.weight"] = MLXArray.ones([config.hiddenSize])

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["backbone.embeddings.weight"])
        XCTAssertNotNil(sanitized["backbone.norm_f.weight"])
        XCTAssertEqual(
            sanitized["backbone.embeddings.weight"]!.shape, [config.vocabSize, config.hiddenSize])
    }

    // MARK: - Basic Forward Pass Tests

    func testNemotronHForwardPass() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

    func testNemotronHWithMambaOnly() throws {
        let config = makeTestConfig(pattern: "MMM")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHWithAttentionOnly() throws {
        let config = makeTestConfig(pattern: "***")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHWithMLP() throws {
        let config = makeTestConfig(pattern: "M-*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHWithMoE() throws {
        let config = makeTestConfig(pattern: "ME*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHFullPattern() throws {
        // Test a pattern with all block types
        let config = makeTestConfig(pattern: "M-E*M-E*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3, 4])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 4, 100])
    }

    // MARK: - Cache Tests

    func testNemotronHCacheCreation() throws {
        // Pattern: M*M- has 2 Mamba + 1 Attention = 3 caches
        let config = makeTestConfig(pattern: "M*M-")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // Only Mamba (M) and Attention (*) layers have caches
        // Pattern M*M- has M, *, M = 3 cacheable layers
        XCTAssertEqual(cache.count, 3)
    }

    func testNemotronHCacheCountMambaOnly() throws {
        let config = makeTestConfig(pattern: "MMM")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // 3 Mamba layers = 3 caches
        XCTAssertEqual(cache.count, 3)
    }

    func testNemotronHCacheCountAttentionOnly() throws {
        let config = makeTestConfig(pattern: "***")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // 3 Attention layers = 3 caches
        XCTAssertEqual(cache.count, 3)
    }

    func testNemotronHCacheCountMixed() throws {
        // Pattern with MLP (-) and MoE (E) which don't have caches
        let config = makeTestConfig(pattern: "M-E*-E")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // Only M and * have caches: M, * = 2 caches
        XCTAssertEqual(cache.count, 2)
    }

    // MARK: - Incremental Generation Tests

    func testNemotronHIncrementalGeneration() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // First pass - process prompt
        let prompt = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let cache = model.newCache(parameters: nil)
        let promptOutput = model.callAsFunction(prompt, cache: cache)

        XCTAssertEqual(promptOutput.shape, [1, 5, 100])

        // Second pass - generate next token
        let nextToken = MLXArray([6])[.newAxis, .ellipsis]
        let nextOutput = model.callAsFunction(nextToken, cache: cache)

        XCTAssertEqual(nextOutput.shape, [1, 1, 100])
    }

    // MARK: - KV Heads Tests

    func testNemotronHKVHeads() throws {
        let config = makeTestConfig(pattern: "M*M*")
        let model = NemotronHModel(config)

        // kvHeads should have entries for Mamba (0) and Attention (numKeyValueHeads)
        // Pattern M*M* = [0, 2, 0, 2] where 2 is numKeyValueHeads
        XCTAssertEqual(model.kvHeads.count, 4)
        XCTAssertEqual(model.kvHeads[0], 0)  // Mamba
        XCTAssertEqual(model.kvHeads[1], 2)  // Attention
        XCTAssertEqual(model.kvHeads[2], 0)  // Mamba
        XCTAssertEqual(model.kvHeads[3], 2)  // Attention
    }

    // MARK: - Vocabulary Size Tests

    func testNemotronHVocabularySize() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        XCTAssertEqual(model.vocabularySize, 100)
    }

    // MARK: - Batch Processing Tests

    func testNemotronHBatchProcessing() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // Batch of 2 sequences - use reshaped to create 2D input
        let flat = MLXArray([1, 2, 3, 4, 5, 6])
        let input = flat.reshaped(2, 3)
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [2, 3, 100])
    }

    // MARK: - Tied Embeddings Test

    func testNemotronHTiedEmbeddings() throws {
        let config = NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: "M*",
            tieWordEmbeddings: true
        )
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHUntiedEmbeddings() throws {
        let config = NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: "M*",
            tieWordEmbeddings: false
        )
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    // MARK: - Shared Experts Test

    func testNemotronHWithSharedExperts() throws {
        let config = NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: "E",
            nSharedExperts: 1
        )
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    // MARK: - Cast Predicate Test

    func testCastPredicateExcludesSpecialParameters() throws {
        let config = makeTestConfig(pattern: "ME")
        let model = NemotronHModel(config)

        let castPredicate = model.castPredicate!

        // These should NOT be cast (return false)
        XCTAssertFalse(castPredicate("backbone.layers.0.mixer.e_score_correction_bias"))
        XCTAssertFalse(castPredicate("backbone.layers.0.mixer.A_log"))

        // Regular parameters should be cast (return true)
        XCTAssertTrue(castPredicate("backbone.layers.0.mixer.in_proj.weight"))
        XCTAssertTrue(castPredicate("backbone.embeddings.weight"))
        XCTAssertTrue(castPredicate("backbone.norm_f.weight"))
    }

    // MARK: - LoRA Layers Test

    func testNemotronHLoRALayers() throws {
        let config = makeTestConfig(pattern: "M*M-E")
        let model = NemotronHModel(config)

        // loraLayers should return backbone.layers
        XCTAssertEqual(model.loraLayers.count, 5)
    }

    // MARK: - Edge Cases

    func testNemotronHSingleMambaLayer() throws {
        let config = makeTestConfig(pattern: "M")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
        XCTAssertEqual(model.kvHeads, [0])  // Mamba has 0 kv heads
    }

    func testNemotronHSingleAttentionLayer() throws {
        let config = makeTestConfig(pattern: "*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
        XCTAssertEqual(model.kvHeads, [2])  // numKeyValueHeads = 2
    }

    func testNemotronHLongSequence() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // Test with a longer sequence
        let input = MLXArray(0 ..< 128)[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 128, 100])
    }

    func testNemotronHMultipleGenerationSteps() throws {
        let config = makeTestConfig(pattern: "M*M*")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // Initial prompt
        let prompt = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let _ = model.callAsFunction(prompt, cache: cache)

        // Multiple generation steps
        for tokenId in 6 ..< 10 {
            let nextToken = MLXArray([tokenId])[.newAxis, .ellipsis]
            let output = model.callAsFunction(nextToken, cache: cache)
            XCTAssertEqual(output.shape, [1, 1, 100])
        }
    }

    // MARK: - Complex Pattern Tests

    func testNemotronHAlternatingPattern() throws {
        // Alternating Mamba and Attention layers
        let config = makeTestConfig(pattern: "M*M*M*M*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])

        // kvHeads should alternate: [0, 2, 0, 2, 0, 2, 0, 2]
        XCTAssertEqual(model.kvHeads, [0, 2, 0, 2, 0, 2, 0, 2])
    }

    func testNemotronHMoEHeavyPattern() throws {
        // Pattern with multiple MoE layers
        let config = makeTestConfig(pattern: "MEE*EE")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])

        // Only M and * contribute to kvHeads
        XCTAssertEqual(model.kvHeads, [0, 2])
    }

    func testNemotronHSSMPreservesActivationAndStateDTypes() throws {
        // Nemotron 3.5 keeps model activations at bf16 while its recurrent
        // accumulator remains fp32. Regressing either side changes greedy
        // logits on the selected checkpoint.
        let hidden = MLXArray.ones([1, 2, 2, 4]).asType(.bfloat16)
        // castPredicate intentionally retains A_log at fp32; this is what
        // promotes the persistent decay accumulator without widening x.
        let aLog = MLXArray.zeros([2]).asType(.float32)
        let b = MLXArray.ones([1, 2, 1, 3]).asType(.bfloat16)
        let c = MLXArray.ones([1, 2, 1, 3]).asType(.bfloat16)
        let d = MLXArray.ones([2]).asType(.bfloat16)
        let dt = MLXArray.zeros([1, 2, 2]).asType(.bfloat16)
        let dtBias = MLXArray.zeros([2]).asType(.bfloat16)

        let (prefill, prefillState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: b,
            C: c,
            D: d,
            dt: dt,
            dtBias: dtBias)
        eval(prefill, prefillState)

        XCTAssertEqual(prefill.dtype, .bfloat16)
        XCTAssertEqual(prefillState.dtype, .float32)

        let (decode, decodeState) = ssmUpdate(
            hiddenStates: hidden[0..., (-1)..., 0..., 0...],
            ALog: aLog,
            B: b[0..., (-1)..., 0..., 0...],
            C: c[0..., (-1)..., 0..., 0...],
            D: d,
            dt: dt[0..., (-1)..., 0...],
            dtBias: dtBias,
            state: prefillState)
        eval(decode, decodeState)

        XCTAssertEqual(decode.dtype, .bfloat16)
        XCTAssertEqual(decodeState.dtype, .float32)
    }
}
