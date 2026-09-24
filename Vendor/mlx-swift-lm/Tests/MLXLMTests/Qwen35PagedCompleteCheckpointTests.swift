import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Real target and production-loaded inline assistant, with small supported
/// attention/GatedDeltaNet geometry. These are mechanics tests, not fleet
/// numerical or latency evidence.
final class Qwen35PagedCompleteCheckpointTests: XCTestCase {
    private struct Fixture {
        let model: Qwen35TextModel
        let assistant: Qwen35InlineMTPAssistant
    }

    private func fixture(experts: Int) throws -> Fixture {
        let json = Data("""
            {
              "model_type": "qwen3_5_moe", "mtplx_mtp": {
                "included": true, "prefix": "mtp.", "block_size": 3
              }, "mtplx_mtp_quantization": {},
              "text_config": {
                "model_type": "qwen3_5_moe_text", "hidden_size": 64,
                "num_hidden_layers": 4, "intermediate_size": 64,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
                "linear_num_value_heads": 1, "linear_num_key_heads": 1,
                "linear_key_head_dim": 64, "linear_value_head_dim": 64,
                "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
                "vocab_size": 64, "num_experts": \(experts),
                "num_experts_per_tok": \(experts > 0 ? 2 : 0),
                "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                "norm_topk_prob": true, "mtp_num_hidden_layers": 1
              }
            }
            """.utf8)
        let object = try JSONSerialization.jsonObject(with: json) as! [String: Any]
        let configuration = try JSONDecoder().decode(Qwen35TextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: object["text_config"]!))
        MLXRandom.seed(7029)
        let model = Qwen35TextModel(configuration)
        model.update(parameters: ModuleParameters.unflattened(
            model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        quantize(model: model, groupSize: 32, bits: 4) { _, module in module is Embedding }
        let draft = Qwen35MTPModule(configuration)
        let weights = Dictionary(uniqueKeysWithValues: draft.parameters().flattened().map {
            ("mtp." + $0.0, $0.1.asType(.bfloat16))
        })
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-paged-complete-mtp-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try json.write(to: directory.appendingPathComponent("config.json"))
        let shard = "model-00001-of-00001.safetensors"
        try save(arrays: weights, url: directory.appendingPathComponent(shard))
        let index = ["weight_map": Dictionary(uniqueKeysWithValues: weights.keys.map { ($0, shard) })]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        eval(model)
        let assistant = try Qwen35InlineMTPAssistant.load(
            from: directory, target: model, verificationMode: .serialTarget)
        return Fixture(model: model, assistant: assistant)
    }

    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }

    private func engine(_ fixture: Fixture, store: CompleteCheckpointFixtureStore?) throws -> (EngineV2, PagedKVBackend) {
        let model = fixture.model
        let kinds = model.cbv2LayerKinds
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let observed = try CBv2NativeKVTypeProbe.run(model: adapter, layerKinds: kinds,
            caches: model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let backend = try PagedKVBackend(layerKinds: kinds, config: .init(
            capacityBytes: 96 << 20, maxPrefillChunk: chunk, nominalMaxSequenceLength: 512,
            segmentSizeBytes: 64 << 10, layerDTypes: observed.layerDTypes))
        let storage = backend.makeLayerCaches()
        let indices = Dictionary(uniqueKeysWithValues: kinds.enumerated().map {
            ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
        })
        let caches = model.newCacheV2 { index, _ in storage[indices[index]!] }
        let engine = EngineV2(model: adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: store != nil),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store,
            mtpDrafter: fixture.assistant, mtpConfig: .init(enabled: true, maxDraftTokens: 2,
                fixedDraftTokens: 2, verificationMode: .serialTarget))
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        let fixedGeneration = try model.cbv2RecurrentStateSpec.allocationBytesPerGeneration(policy: policy)
        XCTAssertEqual(engine.admissionForTesting.fixedBytesPerRequest, 4 * fixedGeneration)
        let projection = try XCTUnwrap(CBv2AuxiliaryAllocationProjection(
            policy: policy, buffers: try XCTUnwrap(fixture.assistant.requestStateAllocationSpecs)))
        let tokens = 257
        let rounded = ((tokens + 4 + 255) / 256) * 256
        let expectedAux = max(rounded * fixture.assistant.requestStateBytesPerToken,
                              try XCTUnwrap(projection.bytes(forTokens: tokens)))
        let logicalTarget = zip(kinds, observed.layerDTypes).reduce(0) { $0 + 2 * $1.0.kvHeads * $1.0.headDim * $1.1.size }
        let target = ((tokens + backend.pool.config.pageSize - 1) / backend.pool.config.pageSize)
            * backend.pool.config.pageSize * logicalTarget
        XCTAssertEqual(engine.admissionForTesting.allocatedBytes(forTokens: tokens),
                       target + 4 * fixedGeneration + expectedAux)
        XCTAssertNil(engine.hybridPrefixCache)
        if store != nil {
            XCTAssertNotNil(engine.completeCheckpointCodec)
            XCTAssertEqual(engine.completeCheckpointCodec?.backendLayout, CBv2CompleteCheckpointManifest.pagedLayout)
        }
        return (engine, backend)
    }

    private func assertReleased(_ engine: EngineV2, _ backend: PagedKVBackend) {
        let state = engine.loopForTesting.onEngineQueueSync {
            (engine.admissionForTesting.bytesReserved, backend.bytesReserved, backend.bytesWired,
             engine.loopForTesting.recurrentStates.isEmpty,
             engine.loopForTesting.mtp?.requestStateCountForTesting)
        }
        XCTAssertEqual(state.0, 0)
        XCTAssertEqual(state.1, 0)
        XCTAssertEqual(state.2, 0)
        XCTAssertTrue(state.3)
        XCTAssertEqual(state.4, 0)
    }

    func testDenseAndMoEReopenEncodedPagesAndNormalMTPAtExactHistoricalBoundaries() async throws {
        for experts in [0, 4] {
            let store = CompleteCheckpointFixtureStore(segmentBytes: 4096)
            let (donor, donorBackend) = try engine(fixture(experts: experts), store: store)
            let prompt = (0 ..< 2 * chunk + 7).map { 1 + ($0 * 7) % 61 }
            let request = CBv2Request(id: .init(1), promptTokens: prompt, sampling: .init(temperature: 0),
                maxTokens: 12, cacheSalt: "tenant", prefixCacheReceiptID: .init(1001))
            let donated = await cbv2SchedCollect(try donor.submit(request))
            XCTAssertEqual(donated.finishReason, .length)
            XCTAssertEqual(store.saved.map(\.manifest.position), [chunk, 2 * chunk])
            XCTAssertTrue(store.saved.allSatisfy {
                $0.manifest.backendLayout == CBv2CompleteCheckpointManifest.pagedLayout
                    && $0.manifest.assistantCodecID != nil
                    && $0.manifest.tensors.contains { $0.role == .assistantHidden }
            })
            assertReleased(donor, donorBackend)
            await donor.shutdown()

            // The reopened store owns only Data and manifests. Target weights,
            // native pool, recurrent rows and the production assistant are new.
            let reopened = CompleteCheckpointFixtureStore(archives: store.saved, segmentBytes: 4096)
            let (warm, warmBackend) = try engine(fixture(experts: experts), store: reopened)
            let (cold, coldBackend) = try engine(fixture(experts: experts), store: nil)
            var branch = prompt
            branch[chunk + 3] = (branch[chunk + 3] % 61) + 1
            for (index, pair) in [(prompt, 2 * chunk), (Array(prompt.prefix(2 * chunk + 1)), 2 * chunk),
                                   (branch, chunk)].enumerated() {
                let request = CBv2Request(id: .init(UInt64(index + 10)), promptTokens: pair.0,
                    sampling: .init(temperature: 0), maxTokens: 12, cacheSalt: "tenant",
                    prefixCacheReceiptID: .init(UInt64(index + 2000)))
                let expected = await cbv2SchedCollect(try cold.submit(request))
                XCTAssertTrue(try reopened.stage(engine: warm, request: request))
                let actual = await cbv2SchedCollect(try warm.submit(request))
                XCTAssertEqual(actual.finishReason, .length)
                XCTAssertEqual(actual.tokens, expected.tokens, "experts=\(experts), history=\(pair.1), case=\(index)")
                XCTAssertEqual(actual.usage?.prefixCachePrefillTokensSaved, pair.1)
                XCTAssertEqual(actual.usage?.prefixCacheReplayTokens, 0)
                XCTAssertEqual(actual.usage?.prefixCacheTier, .snapshot)
                if index == 0 { XCTAssertEqual(actual.tokens, donated.tokens) }
                assertReleased(warm, warmBackend)
                assertReleased(cold, coldBackend)
            }
            XCTAssertGreaterThan(try XCTUnwrap(warm.mtpMetricsSnapshot()).serialVerificationRounds, 0)
            XCTAssertGreaterThan(try XCTUnwrap(warm.mtpMetricsSnapshot()).draftedTokens, 0)
            XCTAssertEqual(reopened.releaseCount, 3)
            await warm.shutdown()
            await cold.shutdown()
        }
    }

    func testStagedCancelAndAdoptionFailuresColdFallbackReleaseOwners() async throws {
        let store = CompleteCheckpointFixtureStore(segmentBytes: 4096)
        let (donor, donorBackend) = try engine(fixture(experts: 0), store: store)
        let request = CBv2Request(id: .init(7), promptTokens: (0 ..< 2 * chunk + 7).map { 1 + $0 % 61 },
            sampling: .init(temperature: 0), maxTokens: 8, cacheSalt: "tenant", prefixCacheReceiptID: .init(1007))
        let expected = await cbv2SchedCollect(try donor.submit(request))
        XCTAssertEqual(expected.finishReason, .length)
        assertReleased(donor, donorBackend)
        await donor.shutdown()

        let reopened = CompleteCheckpointFixtureStore(archives: store.saved, segmentBytes: 4096)
        let (candidate, backend) = try engine(fixture(experts: 0), store: reopened)
        XCTAssertTrue(try reopened.stage(engine: candidate, request: request))
        candidate.loopForTesting.onEngineQueueSync {
            candidate.loopForTesting.suspendStepExecutionAtCountForTesting = 0
        }
        let cancelled = try candidate.submit(request)
        candidate.cancel(request.id)
        candidate.loopForTesting.onEngineQueueSync {
            candidate.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
        let cancelledResult = await cbv2SchedCollect(cancelled)
        XCTAssertEqual(cancelledResult.finishReason, .cancelled)
        assertReleased(candidate, backend)
        XCTAssertEqual(reopened.releaseCount, 1)

        // Fail only the post-staging suffix allocation. Its rollback must leave
        // cold admission and the same engine usable; the second allocation runs.
        XCTAssertTrue(try reopened.stage(engine: candidate, request: request))
        var allocationCalls = 0
        candidate.loopForTesting.onEngineQueueSync {
            backend.pool.slabEval = { array in
                allocationCalls += 1
                if allocationCalls == 1 { throw CBv2CompleteCheckpointError.allocationFailed }
                try withError { eval(array) }
            }
        }
        let fallback = await cbv2SchedCollect(try candidate.submit(request))
        XCTAssertEqual(fallback.finishReason, .length)
        XCTAssertEqual(fallback.tokens, expected.tokens)
        XCTAssertEqual(fallback.usage?.prefixCachePrefillTokensSaved, 0)
        XCTAssertEqual(fallback.usage?.prefixCacheOutcome, .adoptionFailed)
        XCTAssertGreaterThan(allocationCalls, 1)
        assertReleased(candidate, backend)
        XCTAssertEqual(reopened.releaseCount, 2)
        await candidate.shutdown()

        // An authenticated frame can still have an incompatible assistant
        // history. Fail after page attachment, then restore cold request state.
        let corrupted = store.saved.map { archive in
            CompleteCheckpointFixtureStore.Archive(manifest: archive.manifest,
                chunks: archive.chunks.map { part in
                    guard archive.manifest.tensors[part.tensor].role == .assistantTokens else { return part }
                    return .init(tensor: part.tensor, offset: part.offset, bytes: Data(count: part.bytes.count))
                })
        }
        let rejectedStore = CompleteCheckpointFixtureStore(archives: corrupted, segmentBytes: 4096)
        let (rejected, rejectedBackend) = try engine(fixture(experts: 0), store: rejectedStore)
        XCTAssertTrue(try rejectedStore.stage(engine: rejected, request: request))
        let restoredCold = await cbv2SchedCollect(try rejected.submit(request))
        XCTAssertEqual(restoredCold.finishReason, .length)
        XCTAssertEqual(restoredCold.tokens, expected.tokens)
        XCTAssertEqual(restoredCold.usage?.prefixCachePrefillTokensSaved, 0)
        XCTAssertEqual(restoredCold.usage?.prefixCacheOutcome, .adoptionFailed)
        XCTAssertEqual(rejectedStore.releaseCount, 1)
        assertReleased(rejected, rejectedBackend)
        await rejected.shutdown()
    }
}
