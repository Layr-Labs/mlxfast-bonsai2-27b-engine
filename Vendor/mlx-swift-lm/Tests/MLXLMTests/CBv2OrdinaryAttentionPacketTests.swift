import Foundation
import MLX
import XCTest
@_spi(Diagnostics) @testable import MLXLMCommon

final class CBv2OrdinaryAttentionPacketTests: XCTestCase {
    private func run(capture: Bool, direct: Bool, backendName: String = "contiguous",
                     action: String? = nil, independentPositions: Bool = false,
                     dtype: DType = .float32) async throws
        -> AttentionPacketEngineRun {
        let model = AttentionPacketEngineModel(dtype: dtype)
        let backend: any CBv2KVBackend
        let caches: [any CBv2AttendingLayerCache]
        if backendName == "contiguous" {
            backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 26, kvDType: dtype))
            caches = model.kinds.map { CBv2LayerCache(layerIndex: $0.modelLayerIndex!, kind: $0) }
        } else {
            var config = PagedKVPoolConfig(capacityBytes: 1 << 26, dtype: dtype, maxPrefillChunk: 8)
            if backendName == "segmented" { config.segmentSizeBytes = 1 << 20 }
            let paged = try PagedKVBackend(layerKinds: model.kinds, config: config)
            backend = paged
            caches = paged.makeLayerCaches()
        }
        let engine = EngineV2(model: model, layerKinds: model.kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 8, maxWaiting: 4, enablePrefixCache: false))
        if action == "cancel" { model.selectedAction = { [weak engine] in engine?.cancel(.init(2)) } }
        if action == "fail" {
            let paged = try XCTUnwrap(backend as? PagedKVBackend)
            model.selectedAction = {
                // Inject the existing typed unwind AFTER capture graph construction,
                // before checkedModelForward can submit any of that failed graph.
                paged.pool.writeValidation.record(.init(layerIndex: 1, expected: .float32,
                                                        keys: .float16, values: .float32))
            }
        }
        do {
            if capture {
                try engine.configureAttentionPacket(.init(requestID: 2, outputIndex: 3, storageLayerIndex: 1))
                try engine.configureAttentionMetadata(.init(requestID: 2, outputIndex: independentPositions ? 2 : 3))
                try engine.configureLogitDiagnostic(.init(requestID: 2, outputIndex: independentPositions ? 4 : 3,
                                                          candidateIDs: [1, 2], maximumRecords: 1))
            }
            let result = await cbv2SchedCollect(try engine.submit(.init(
                id: .init(2), promptTokens: makePromptTokens(length: 17, seed: 71),
                sampling: .init(temperature: 0), maxTokens: 8, prefixCacheEnabled: false,
                tokenConstraint: direct ? AttentionPacketAllTokens() : nil)))
            var packet: CBv2AttentionPacketSnapshot?
            let idle = await cbv2SchedWait {
                do { packet = try engine.takeAttentionPacketSnapshot(); return true }
                catch { return false }
            }
            XCTAssertTrue(idle)
            let metadata = try engine.takeAttentionMetadataSnapshot()
            let logits = try engine.takeLogitDiagnosticSnapshot()
            XCTAssertTrue(model.lastCaches.allSatisfy {
                ($0 as? any CBv2AttentionPacketBinding)?.attentionPacket == nil
                    && ($0 as? any CBv2AttentionMetadataBinding)?.attentionMetadata == nil
            })
            XCTAssertNil(model.observedPacket, "retirement/failure must release the selected raw forward")
            if backendName == "contiguous" { XCTAssertEqual(caches.map(\.layerIndex), [3, 7]) }
            try engine.configureAttentionPacket(nil)
            XCTAssertNil(try engine.takeAttentionPacketSnapshot())
            await engine.shutdown()
            XCTAssertNil(engine.loopForTesting.attentionPacket)
            XCTAssertNil(engine.loopForTesting.attentionMetadata)
            return .init(tokens: result.tokens, shapes: model.shapes, staged: model.staged,
                         packetForwards: model.packetForwards, metadataForwards: model.metadataForwards,
                         topTwoCalls: model.topTwoCalls, forwardsAfterPacket: model.forwardsAfterPacket,
                         successorObservedPendingPacket: model.successorObservedPendingPacket,
                         packet: packet, metadata: metadata, logits: logits)
        } catch {
            await engine.shutdown()
            throw error
        }
    }

    func testDirectAndChainedPacketsPreserveOrdinaryForwardsAndCoexistingDiagnostics() async throws {
        for backend in ["contiguous", "fixed", "segmented"] {
            for direct in [false, true] {
                let control = try await run(capture: false, direct: direct, backendName: backend)
                let captured = try await run(capture: true, direct: direct, backendName: backend)
                XCTAssertEqual(control.tokens, captured.tokens)
                XCTAssertEqual(control.shapes, captured.shapes)
                XCTAssertEqual(control.staged, captured.staged)
                XCTAssertEqual(captured.staged, captured.shapes.count)
                XCTAssertEqual(control.packetForwards + control.metadataForwards + control.topTwoCalls, 0)
                XCTAssertEqual(captured.packetForwards, 1)
                XCTAssertEqual(captured.metadataForwards, 1)
                XCTAssertEqual(captured.topTwoCalls, 0, "logit observation must not invoke model policy reductions")
                XCTAssertGreaterThan(captured.forwardsAfterPacket, 0)
                XCTAssertFalse(captured.successorObservedPendingPacket,
                               "actual next model forward must wait for packet retirement, including a chained selected step")
                let packet = try XCTUnwrap(captured.packet)
                XCTAssertEqual(packet.evaluationStatus, "completed")
                XCTAssertEqual(packet.tensors.count, 6)
                XCTAssertEqual(packet.metadata.sampleOutcome, "confirmed")
                XCTAssertEqual(packet.metadata.expectedOwnerCount, 1)
                XCTAssertEqual(packet.metadata.records.count, 1)
                XCTAssertTrue(packet.metadata.refusals.isEmpty)
                XCTAssertEqual(packet.metadata.seedToken, captured.tokens[2])
                XCTAssertEqual(packet.metadata.targetToken, captured.tokens[3])
                let record = try XCTUnwrap(packet.metadata.records.first)
                XCTAssertEqual(record.storageLayerIndex, 1)
                XCTAssertEqual(record.modelLayerIndex, 7)
                XCTAssertEqual(record.phase, direct ? "decode" : "chained_decode")
                XCTAssertEqual(record.offsetBefore, 19)
                XCTAssertEqual(record.offsetAfter, 20)
                XCTAssertEqual(captured.metadata?.records.count, 2)
                XCTAssertEqual(captured.logits?.records.first?.targetToken, captured.tokens[3])
            }
        }
    }

    func testIndependentSelectionsDoNotOverwriteOtherCaptureBindings() async throws {
        let captured = try await run(capture: true, direct: false, independentPositions: true)
        XCTAssertEqual(captured.metadata?.configuration.outputIndex, 2)
        XCTAssertEqual(captured.metadata?.targetToken, captured.tokens[2])
        XCTAssertEqual(captured.packet?.metadata.configuration.outputIndex, 3)
        XCTAssertEqual(captured.packet?.metadata.targetToken, captured.tokens[3])
        XCTAssertEqual(captured.logits?.records.first?.outputIndex, 4)
        XCTAssertEqual(captured.logits?.records.first?.targetToken, captured.tokens[4])
    }

    func testBFloat16OrdinaryPacketsRetireBeforeTheNextForward() async throws {
        for backend in ["contiguous", "fixed", "segmented"] {
            let control = try await run(capture: false, direct: false, backendName: backend, dtype: .bfloat16)
            let captured = try await run(capture: true, direct: false, backendName: backend, dtype: .bfloat16)
            XCTAssertEqual(control.tokens, captured.tokens)
            XCTAssertEqual(control.shapes, captured.shapes)
            XCTAssertFalse(captured.successorObservedPendingPacket)
            let packet = try XCTUnwrap(captured.packet)
            XCTAssertEqual(packet.evaluationStatus, "completed")
            XCTAssertEqual(packet.tensors.count, 6)
            XCTAssertTrue(packet.tensors.values.allSatisfy { $0.dtype == "bfloat16" })
            XCTAssertTrue(packet.metadata.refusals.isEmpty,
                          "the nonblocking availability guard must prove all bytes reached the existing fence")
        }
    }

    func testCancellationAndTypedForwardFailureDrainRawHandlesWithoutConfirmation() async throws {
        for action in ["cancel", "fail"] {
            let captured = try await run(capture: true, direct: false, backendName: "segmented", action: action)
            let packet = try XCTUnwrap(captured.packet)
            XCTAssertEqual(captured.packetForwards, 1)
            XCTAssertTrue(packet.tensors.isEmpty)
            XCTAssertNotEqual(packet.evaluationStatus, "completed")
            XCTAssertNotEqual(packet.metadata.sampleOutcome, "confirmed")
            XCTAssertNil(packet.metadata.targetToken)
            XCTAssertLessThanOrEqual(captured.tokens.count, 3)
            if action == "cancel" { XCTAssertEqual(packet.metadata.sampleOutcome, "discarded") }
            else { XCTAssertFalse(packet.metadata.forwardSucceeded) }
        }
    }
}
