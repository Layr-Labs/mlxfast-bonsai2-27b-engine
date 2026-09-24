import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

private final class CompleteCheckpointFixtureModel:
    CBv2RecurrentSteppableModel, CBv2CompleteCheckpointKVTypeProviding
{
    var cbv2Capabilities: CBv2ModelCapabilities {
        var result = CBv2ModelCapabilities.initialRecurrentTarget
        result.supportsRecurrentCheckpointReuse = true
        return result
    }
    let cbv2CompleteCheckpointKVDTypes: [DType]? = [.float32]
    private let spec: CBv2RecurrentStateSpec = .init(layers: [.init(
        modelLayerIndex: 0, convShape: [1, 1, 1], convDType: .float32,
        ssmShape: [1, 1, 1, 1], ssmDType: .float32)])
    private(set) var recurrentSpecReads = 0
    var recurrentStateSpec: CBv2RecurrentStateSpec? {
        recurrentSpecReads += 1
        return spec
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        preconditionFailure("explicit recurrent state is required")
    }
    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let length = tokens.dim(1)
        let qkv = MLXArray.zeros([1, 1, length, 1])
        for cache in caches {
            _ = cache.updateAndAttend(queries: qkv, keys: qkv, values: qkv, scale: 1, sinks: nil)
        }
        let previous = recurrentState[0].inputState(modelLayerIndex: 0)?.ssm
            ?? MLXArray.zeros([1, 1, 1, 1])
        let value = previous.reshaped([]) + sum(tokens.asType(.float32))
        try! recurrentState[0].stage(
            modelLayerIndex: 0, conv: value.reshaped([1, 1, 1]), ssm: value.reshaped([1, 1, 1, 1]))
        let target = value.asType(.int32) % 16
        let logits = MLX.where(MLXArray(Int32(0) ..< Int32(16)) .== target, 10, -10)
        return broadcast(logits.reshaped([1, 1, 16]), to: [1, length, 16])
    }
}

final class CheckpointPublicationGate: @unchecked Sendable {
    let entered = DispatchSemaphore(value: 0)
    let resume = DispatchSemaphore(value: 0)
    func waitUntilEntered() -> Bool { entered.wait(timeout: .now() + 10) == .success }
    func block() { entered.signal(); _ = resume.wait(timeout: .now() + 10) }
}

/// Encoded-byte stand-in for a durable store. It deliberately never keeps MLX
/// arrays; reopening copies only manifests/bytes to simulate provider restart.
final class CompleteCheckpointFixtureStore: CBv2CompletePrefixCache, @unchecked Sendable {
    struct Chunk: Sendable { let tensor: Int; let offset: Int; let bytes: Data }
    struct Archive: Sendable { let manifest: CBv2CompleteCheckpointManifest; let chunks: [Chunk] }
    let identity = CBv2CompleteCheckpointIdentity(
        modelAggregateHash: "tiny-qwen", promptContractID: "same-template",
        buildID: "same-build", numericsFingerprint: "native-fp32")
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "test.complete-checkpoint-store")
    private var archives: [Archive]
    private var tickets: [CBv2RequestID: CBv2StagedCompleteCheckpoint] = [:]
    private var closed = false
    private var releases = 0
    let maximumPosition: Int
    let segmentBytes: Int
    let gate: CheckpointPublicationGate?

    init(archives: [Archive] = [], maximumPosition: Int = .max, gate: CheckpointPublicationGate? = nil,
         segmentBytes: Int = 64) {
        self.archives = archives
        self.maximumPosition = maximumPosition
        self.segmentBytes = segmentBytes
        self.gate = gate
    }
    var saved: [Archive] { lock.lock(); defer { lock.unlock() }; return archives }
    var releaseCount: Int { lock.lock(); defer { lock.unlock() }; return releases }
    func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool { position <= maximumPosition }

    func takeStaged(
        requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?, maximumSequenceLength: Int
    ) -> CBv2StagedCompleteCheckpoint? {
        lock.lock()
        defer { lock.unlock() }
        return tickets.removeValue(forKey: requestID)
    }

    func stage(engine: EngineV2, request: CBv2Request) throws -> Bool {
        guard let receipt = request.prefixCacheReceiptID else { return false }
        let matching = saved.filter {
            $0.manifest.cacheSalt == request.cacheSalt
                && $0.manifest.position < request.promptTokens.count
                && $0.manifest.prefixTokens.elementsEqual(request.promptTokens.prefix($0.manifest.position))
        }.max { $0.manifest.position < $1.manifest.position }
        guard let matching else { return false }
        let plan = try engine.planCompleteCheckpointImport(manifest: matching.manifest, request: request)
        let sink = try plan.allocate { [self] in
            lock.lock(); releases += 1; lock.unlock()
        }
        defer { sink.close() }
        for chunk in matching.chunks {
            try sink.appendSegment(tensorIndex: chunk.tensor, byteOffset: chunk.offset, data: chunk.bytes)
        }
        let ready = try sink.finish()
        lock.lock()
        tickets[receipt] = ready
        lock.unlock()
        return true
    }

    func donate(
        _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?, tokens: [Int],
        cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void
    ) {
        queue.async { [self, source] in
            gate?.block()
            lock.lock()
            let cancelled = closed
            lock.unlock()
            guard !cancelled else { source.close(); completion([]); return }
            do {
                var chunks: [Chunk] = []
                for (index, tensor) in source.manifest.tensors.enumerated() {
                    var offset = 0
                    while offset < tensor.byteCount {
                        let data = try source.readSegment(tensorIndex: index, byteOffset: offset, maximumBytes: segmentBytes)
                        chunks.append(.init(tensor: index, offset: offset, bytes: data))
                        offset += data.count
                    }
                }
                // The archive is a caller-owned encoded store, not a retained
                // native manifest value. Round-trip the wire form so reopening
                // cannot carry an old engine's host-metadata permit.
                let manifest = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self,
                    from: JSONEncoder().encode(source.manifest))
                source.close()
                lock.lock()
                let commit = !closed
                if commit { archives.append(.init(manifest: manifest, chunks: chunks)) }
                lock.unlock()
                completion(commit ? [source.manifest.position] : [])
            } catch { source.close(); completion([]) }
        }
    }

    func close() {
        lock.lock()
        closed = true
        let old = Array(tickets.values)
        tickets.removeAll()
        lock.unlock()
        old.forEach { $0.close() }
        gate?.resume.signal()
    }
}

private final class CompleteCheckpointReceiptLog: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [(CBv2RequestID, [Int])] = []
    var count: Int { lock.lock(); defer { lock.unlock() }; return entries.count }
    func append(_ id: CBv2RequestID, _ positions: [Int]) {
        lock.lock(); entries.append((id, positions)); lock.unlock()
    }
}

final class CBv2CompleteCheckpointEngineTests: XCTestCase {
    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }
    private func engine(
        _ store: CompleteCheckpointFixtureStore,
        model: CompleteCheckpointFixtureModel = CompleteCheckpointFixtureModel()
    ) -> (EngineV2, CBv2ContiguousKVBackend) {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1, modelLayerIndex: 1)]
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 64 << 20, kvDType: .float32))
        return (EngineV2(
            model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: true),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store), backend)
    }

    func testCompleteCacheSkipsRecurrentSpecReadsUntilEligibleCapture() async throws {
        let store = CompleteCheckpointFixtureStore()
        let model = CompleteCheckpointFixtureModel()
        let (engine, _) = engine(store, model: model)

        let decodeOnly = await cbv2SchedCollect(try engine.submit(CBv2Request(
            id: .init(5), promptTokens: [1, 2, 3], maxTokens: 6,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1005))))
        XCTAssertEqual(decodeOnly.finishReason, .length)
        XCTAssertTrue(store.saved.isEmpty)
        let readsAfterDecode = model.recurrentSpecReads
        let emptyFinalize = CBv2InFlightStep(
            assignments: [], participants: [], sampledRows: [], sampledTokens: nil,
            evalTargets: [], wallStartedNanos: 0)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.captureRecurrentCheckpoints(emptyFinalize)
        }
        XCTAssertEqual(model.recurrentSpecReads, readsAfterDecode,
            "finalization without eligible checkpoint geometry must not probe recurrent spec")

        let capture = await cbv2SchedCollect(try engine.submit(CBv2Request(
            id: .init(6), promptTokens: Array(repeating: 1, count: chunk + 1), maxTokens: 2,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1006))))
        XCTAssertEqual(capture.finishReason, .length)
        XCTAssertEqual(store.saved.map(\.manifest.position), [chunk])
        await engine.shutdown()
    }

    func testInitialReadScratchUsesSlotCeilingAndReleasesExactlyOnce() async throws {
        let (engine, _) = engine(CompleteCheckpointFixtureStore())
        let scratch = CBv2CompleteCheckpointManifest.maximumProviderScratchBytes
        engine.updateKVBytesCapacity(scratch - 1)
        XCTAssertThrowsError(try engine.reserveCompleteCheckpointReadScratch())
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)

        engine.updateKVBytesCapacity(scratch)
        var lease: CBv2CompleteCheckpointIOLease? = try engine.reserveCompleteCheckpointReadScratch()
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, scratch)
        XCTAssertEqual(engine.admissionForTesting.transientBytesReserved, scratch)
        XCTAssertThrowsError(try engine.reserveCompleteCheckpointReadScratch())
        lease?.close()
        lease?.close()
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        lease = try engine.reserveCompleteCheckpointReadScratch()
        withExtendedLifetime(lease) {
            XCTAssertEqual(engine.admissionForTesting.bytesReserved, scratch)
        }
        lease = nil
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        await engine.shutdown()
    }

    func testEncodedCheckpointSurvivesEngineRestartAndLeavesZeroIdleCacheArrays() async throws {
        let store = CompleteCheckpointFixtureStore()
        let (first, firstBackend) = engine(store)
        let request = CBv2Request(
            id: .init(7), promptTokens: Array(repeating: 1, count: 2 * chunk + 1), maxTokens: 3,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1007))
        let cold = await cbv2SchedCollect(try first.submit(request))
        XCTAssertEqual(cold.finishReason, .length)
        XCTAssertEqual(store.saved.map(\.manifest.position), [chunk, 2 * chunk])
        XCTAssertNil(first.hybridPrefixCache)
        XCTAssertEqual(first.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(firstBackend.bytesReserved, 0)
        await first.shutdown()

        let reopened = CompleteCheckpointFixtureStore(archives: store.saved)
        let (second, secondBackend) = engine(reopened)
        let warmRequest = CBv2Request(
            id: .init(7), promptTokens: request.promptTokens, maxTokens: 3,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(2007))
        XCTAssertTrue(try reopened.stage(engine: second, request: warmRequest))
        let warm = await cbv2SchedCollect(try second.submit(warmRequest))
        XCTAssertEqual(warm.tokens, cold.tokens, "restored recurrent state affects the sampled IDs")
        XCTAssertEqual(warm.usage?.prefixCacheTier, .snapshot)
        XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved, 2 * chunk)
        XCTAssertEqual(reopened.saved.count, 2, "inherited-only repeats perform no new writes")
        XCTAssertEqual(reopened.releaseCount, 1)
        XCTAssertEqual(second.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(secondBackend.bytesReserved, 0)
        let foreign = CBv2Request(
            id: .init(8), promptTokens: request.promptTokens, maxTokens: 3,
            cacheSalt: "another-tenant", prefixCacheReceiptID: .init(2008))
        XCTAssertFalse(try reopened.stage(engine: second, request: foreign))
        await second.shutdown()
    }

    func testUnreadableLatestEndpointPreservesDeepestEligibleCheckpoint() async throws {
        let store = CompleteCheckpointFixtureStore(maximumPosition: 2 * chunk)
        let (engine, backend) = engine(store)
        let result = await cbv2SchedCollect(try engine.submit(.init(
            id: .init(9), promptTokens: Array(repeating: 1, count: 4 * chunk + 1), maxTokens: 1,
            prefixCacheReceiptID: .init(1009))))
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(store.saved.map(\.manifest.position), [chunk, 2 * chunk])
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(backend.bytesReserved, 0)
        await engine.shutdown()
    }

    func testCancellationAfterCaptureDropsAllPayloadBeforeTerminal() async throws {
        let store = CompleteCheckpointFixtureStore()
        let (engine, backend) = engine(store)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = 2
        }
        let request = CBv2Request(
            id: .init(10), promptTokens: Array(repeating: 1, count: 3 * chunk + 1), maxTokens: 5,
            prefixCacheReceiptID: .init(1010))
        let stream = try engine.submit(request)
        let captured = await cbv2SchedWait {
            engine.loopForTesting.onEngineQueueSync {
                engine.completeCheckpointCapture?.hasCheckpoints(requestID: request.id) == true
            }
        }
        XCTAssertTrue(captured)
        engine.cancel(request.id)
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
        let result = await cbv2SchedCollect(stream)
        XCTAssertEqual(result.finishReason, .cancelled)
        XCTAssertTrue(store.saved.isEmpty)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(backend.bytesReserved, 0)
        await engine.shutdown()
    }

    func testShutdownDrainsBlockedExportAndLateCancellationPreservesGenerationFence() async throws {
        let gate = CheckpointPublicationGate()
        let store = CompleteCheckpointFixtureStore(gate: gate)
        let (engine, backend) = engine(store)
        let receipts = CompleteCheckpointReceiptLog()
        engine.setCompletePrefixPublicationHandler { receipts.append($0, $1) }
        let request = CBv2Request(
            id: .init(11), promptTokens: Array(repeating: 1, count: chunk + 1), maxTokens: 1,
            prefixCacheReceiptID: .init(1011))
        let stream = try engine.submit(request)
        let collection = Task { await cbv2SchedCollect(stream) }
        let entered = await Task.detached { gate.waitUntilEntered() }.value
        XCTAssertTrue(entered)
        XCTAssertGreaterThan(backend.bytesReserved, 0)
        XCTAssertGreaterThan(engine.admissionForTesting.bytesReserved, 0)
        engine.cancel(request.id)
        XCTAssertThrowsError(try engine.submit(.init(
            id: request.id, promptTokens: request.promptTokens, maxTokens: 1,
            prefixCacheReceiptID: .init(2011))))
        await engine.shutdown()
        let result = await collection.value
        XCTAssertEqual(result.finishReason, .length)
        XCTAssertEqual(receipts.count, 0)
        XCTAssertTrue(store.saved.isEmpty)
        XCTAssertEqual(backend.bytesReserved, 0)
        XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
        XCTAssertTrue(engine.loopForTesting.recurrentStates.isEmpty)
        store.close()
    }
}
