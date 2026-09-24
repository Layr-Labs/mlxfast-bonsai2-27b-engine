import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Real target and production-loaded inline assistant, with small supported
/// attention/GatedDeltaNet geometry. These are mechanics tests, not fleet
/// numerical or latency evidence.
final class Qwen35PagedExecutionTests: XCTestCase {
    private struct Fixture {
        let model: Qwen35TextModel
        let assistant: Qwen35InlineMTPAssistant
    }

    private func fixture(experts: Int, mode: CBv2MTPVerificationMode = .serialTarget) throws -> Fixture {
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
            .appendingPathComponent("qwen-paged-mtp-\(UUID().uuidString)", isDirectory: true)
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
            from: directory, target: model, verificationMode: mode)
        return Fixture(model: model, assistant: assistant)
    }

    private func engine(
        _ fixture: Fixture, paged: Bool, chunk: Int,
        mode: CBv2MTPVerificationMode = .serialTarget,
        wrap: ((CBv2SteppableLanguageModelAdapter) -> any CBv2SteppableModel)? = nil
    ) throws -> (EngineV2, CBv2KVBackend) {
        let model = fixture.model
        let kinds = model.cbv2LayerKinds
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let probeCaches = model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }
        let observed = try CBv2NativeKVTypeProbe.run(model: adapter, layerKinds: kinds, caches: probeCaches)
        XCTAssertEqual(observed.layerDTypes, [.bfloat16, .bfloat16])
        XCTAssertEqual(observed.observations.map(\.modelLayerIndex), [1, 3, 1, 3])
        XCTAssertTrue(probeCaches.allSatisfy { $0.rows.isEmpty })
        let backend: CBv2KVBackend
        let caches: [any CBv2AttendingLayerCache]
        if paged {
            let pages = try PagedKVBackend(layerKinds: kinds, config: .init(
                capacityBytes: 64 << 20, maxPrefillChunk: chunk,
                nominalMaxSequenceLength: 512, segmentSizeBytes: 64 << 10,
                layerDTypes: observed.layerDTypes))
            backend = pages
            let storage = pages.makeLayerCaches()
            let indices = Dictionary(uniqueKeysWithValues: kinds.enumerated().map {
                ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
            })
            caches = model.newCacheV2 { index, _ in storage[indices[index]!] }
        } else {
            backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 64 << 20, kvDType: .bfloat16))
            caches = model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }
        }
        XCTAssertNil(EngineV2.backendCapabilityViolation(capabilities: model.cbv2Capabilities, backend: backend))
        return (EngineV2(
            model: wrap?(adapter) ?? adapter, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0), mtpDrafter: fixture.assistant,
            mtpConfig: .init(enabled: true, maxDraftTokens: 2, fixedDraftTokens: 2,
                             verificationMode: mode)), backend)
    }

    /// Runs the production target before introducing a value-only dtype
    /// change at its paged write boundary. Detection is keyed to the live
    /// speculative transaction, so prompts and seed forwards remain real.
    /// Serial column two fails after column one has actually evaluated;
    /// captured rectangular verification fails before its first evaluation.
    private final class DTypeChangingTarget: CBv2RecurrentMTPSteppableModel,
        CBv2MTPPolicyTopTwoProviding, CBv2MTPPolicyTopTwoCapabilityProviding
    {
        let base: CBv2SteppableLanguageModelAdapter
        private let lock = NSLock()
        private var failOnVerifyCall: Int?
        private var verifyCalls = 0
        private var capturedVerifyCalls = 0
        var verificationCallCounts: (total: Int, captured: Int) {
            lock.withLock { (verifyCalls, capturedVerifyCalls) }
        }
        init(_ base: CBv2SteppableLanguageModelAdapter, failOnVerifyCall: Int) {
            self.base = base
            self.failOnVerifyCall = failOnVerifyCall
        }
        func allowValidCalls() { lock.withLock { failOnVerifyCall = nil } }
        var failureWasInjected: Bool {
            lock.withLock { failOnVerifyCall.map { verifyCalls >= $0 } ?? false }
        }
        var cbv2Capabilities: CBv2ModelCapabilities { base.cbv2Capabilities }
        var recurrentStateSpec: CBv2RecurrentStateSpec? { base.recurrentStateSpec }
        var mtpCaptureLayers: CBv2MTPCaptureLayers? { base.mtpCaptureLayers }
        var mtpTargetIdentity: ObjectIdentifier? { base.mtpTargetIdentity }
        var supportsRequestStatefulMTP: Bool { base.supportsRequestStatefulMTP }
        var supportsCapturedVerifyWindow: Bool { base.supportsCapturedVerifyWindow }
        var cbv2MTPPolicyTopTwoAvailable: Bool { base.cbv2MTPPolicyTopTwoAvailable }
        func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
            base.cbv2MTPTopTwo(logits)
        }
        func forward(tokens: MLXArray, caches: [any CBv2AttendingLayerCache]) -> MLXArray {
            base.forward(tokens: tokens, caches: caches)
        }
        func forward(tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
                     recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray {
            base.forward(tokens: tokens, caches: caches, recurrentState: recurrentState)
        }
        func forwardWithHidden(tokens: MLXArray, caches: [any CBv2AttendingLayerCache])
            -> (logits: MLXArray, lastHidden: MLXArray) {
            base.forwardWithHidden(tokens: tokens, caches: caches)
        }
        func forwardWithHidden(tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
                               recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?)
            -> (logits: MLXArray, lastHidden: MLXArray) {
            let output = base.forwardWithHidden(tokens: tokens, caches: caches,
                recurrentState: recurrentState, positionIds: positionIds)
            injectDuringVerification(tokens: tokens, caches: caches, captured: false)
            return output
        }
        func forwardWithHiddenCaptured(tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
                                       recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?)
            -> (logits: MLXArray, lastHidden: MLXArray) {
            let output = base.forwardWithHiddenCaptured(tokens: tokens, caches: caches,
                recurrentState: recurrentState, positionIds: positionIds)
            injectDuringVerification(tokens: tokens, caches: caches, captured: true)
            return output
        }
        private func injectDuringVerification(
            tokens: MLXArray, caches: [any CBv2AttendingLayerCache], captured: Bool
        ) {
            guard let cache = caches.compactMap({ $0 as? PagedLayerCache }).first,
                  let row = cache.rows.first as? PagedSequenceKV, row.speculativeBase != nil
            else { return }
            let fail = lock.withLock {
                verifyCalls += 1
                if captured { capturedVerifyCalls += 1 }
                return verifyCalls == failOnVerifyCall
            }
            guard fail else { return }
            let kind = cache.kind, batch = tokens.dim(0), length = tokens.dim(1)
            let queries = MLXArray.zeros([batch, kind.queryHeads, length, kind.headDim], dtype: .bfloat16)
            let keys = MLXArray.zeros([batch, kind.kvHeads, length, kind.headDim], dtype: .bfloat16)
            _ = cache.updateAndAttend(queries: queries, keys: keys,
                                      values: keys.asType(.float32), scale: 0.125, sinks: nil)
        }
    }

    func testMTPDTypeFaultRetiresEvaluatedAndCapturedTransactionsThenRecovers() async throws {
        for (mode, failingCall) in [(CBv2MTPVerificationMode.serialTarget, 2), (.rectangular, 1)] {
            let request = CBv2Request(id: .init(921), promptTokens: (1 ... 17).map { $0 },
                sampling: .init(temperature: 0), maxTokens: 12)
            let (reference, _) = try engine(fixture(experts: 0, mode: mode),
                                             paged: false, chunk: 32, mode: mode)
            let expected = await cbv2SchedCollect(try reference.submit(request))
            XCTAssertEqual(expected.finishReason, .length)
            await reference.shutdown()
            var target: DTypeChangingTarget!
            let (candidate, backend) = try engine(fixture(experts: 0, mode: mode),
                paged: true, chunk: 32, mode: mode, wrap: {
                    let changing = DTypeChangingTarget($0, failOnVerifyCall: failingCall)
                    target = changing
                    return changing
                })
            let failure = await cbv2SchedCollect(try candidate.submit(request))
            guard case .error(let message) = failure.finishReason else {
                XCTFail("verification dtype change did not terminate the affected request")
                await candidate.shutdown()
                continue
            }
            XCTAssertTrue(message.contains("paged KV dtype mismatch") && message.contains("values float32"))
            XCTAssertTrue(target.failureWasInjected)
            let counts = target.verificationCallCounts
            XCTAssertEqual(counts.total, failingCall)
            XCTAssertEqual(counts.captured, mode == .rectangular ? 1 : 0,
                           "the requested serial/captured seam must actually execute")
            assertReleased(candidate, backend: backend)
            candidate.loopForTesting.onEngineQueueSync {
                XCTAssertFalse((backend as! PagedKVBackend).pool.writeValidation.isFaulted)
            }
            target.allowValidCalls()
            let recovered = await cbv2SchedCollect(try candidate.submit(request))
            XCTAssertEqual(recovered.finishReason, .length)
            XCTAssertEqual(recovered.tokens, expected.tokens, "failed target and assistant state cannot survive same-ID reuse")
            assertReleased(candidate, backend: backend)
            await candidate.shutdown()
        }
    }

    private func assertReleased(_ engine: EngineV2, backend: CBv2KVBackend) {
        // Terminal delivery uses a separate queue. Establish a real engine
        // barrier before reading queue-owned metadata or its published gauge.
        let state = engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.publishGauges()
            return (engine.admissionForTesting.bytesReserved, backend.bytesReserved,
                    engine.capacity().pagedStorage?.committedBytes,
                    engine.loopForTesting.recurrentStates.isEmpty,
                    engine.loopForTesting.mtp?.requestStateCountForTesting)
        }
        XCTAssertEqual(state.0, 0)
        XCTAssertEqual(state.1, 0)
        XCTAssertEqual(state.2, 0)
        XCTAssertTrue(state.3)
        XCTAssertEqual(state.4, 0)
    }

    func testRecurrentGateRequiresObservedNativeSegmentedStorage() throws {
        let target = try fixture(experts: 0).model
        let kinds = target.cbv2LayerKinds
        let capabilities = target.cbv2Capabilities
        XCTAssertTrue(capabilities.supportsPagedKV)
        XCTAssertTrue(capabilities.requiresNativePagedKV)
        for config in [
            PagedKVPoolConfig(capacityBytes: 1 << 20),
            PagedKVPoolConfig(capacityBytes: 1 << 20, layerDTypes: [.bfloat16, .bfloat16]),
            PagedKVPoolConfig(capacityBytes: 1 << 20, segmentSizeBytes: 64 << 10)
        ] {
            let backend = try PagedKVBackend(layerKinds: kinds, config: config)
            XCTAssertNotNil(EngineV2.backendCapabilityViolation(capabilities: capabilities, backend: backend))
        }
    }

    func testDenseAndMoEMTPRollbackAndCancellationMatchContiguousTokens() async throws {
        for experts in [0, 4] {
            let (reference, _) = try engine(fixture(experts: experts), paged: false, chunk: 32)
            let request = CBv2Request(id: .init(91), promptTokens: (0 ..< 71).map { 1 + ($0 * 7) % 61 },
                sampling: .init(temperature: 0), maxTokens: 18)
            let expected = await cbv2SchedCollect(try reference.submit(request))
            XCTAssertEqual(expected.finishReason, .length)
            await reference.shutdown()

            for chunk in [32, 48] {
                let (candidate, backend) = try engine(fixture(experts: experts), paged: true, chunk: chunk)
                let actual = await cbv2SchedCollect(try candidate.submit(request))
                XCTAssertEqual(actual.tokens, expected.tokens, "experts=\(experts), chunk=\(chunk)")
                XCTAssertEqual(actual.finishReason, .length)
                let metrics = try XCTUnwrap(candidate.mtpMetricsSnapshot())
                XCTAssertGreaterThan(metrics.serialVerificationRounds, 0)
                XCTAssertGreaterThan(metrics.draftedTokens, metrics.acceptedTokens, "real rejected drafts must exercise rollback")
                assertReleased(candidate, backend: backend)

                var cancelled = request
                cancelled.maxTokens = 256
                var finish: CBv2FinishReason?
                var sawDelta = false
                for await event in try candidate.submit(cancelled) {
                    switch event {
                    case .delta:
                        sawDelta = true
                        candidate.cancel(cancelled.id)
                    case .finished(let reason, _): finish = reason
                    }
                }
                XCTAssertTrue(sawDelta)
                XCTAssertEqual(finish, .cancelled)
                let retry = await cbv2SchedCollect(try candidate.submit(request))
                XCTAssertEqual(retry.tokens, expected.tokens, "same-ID retry must not retain old recurrent or assistant state")
                assertReleased(candidate, backend: backend)
                await candidate.shutdown()
            }
        }
    }
}
