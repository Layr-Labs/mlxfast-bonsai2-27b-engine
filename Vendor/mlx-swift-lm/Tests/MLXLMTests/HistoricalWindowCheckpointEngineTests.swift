import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Both the earlier window and its dependent full layer affect every later
/// logit. Borrowed layers read their owner's cache, never a synthetic own row.
private final class HistoricalAttentionModel: CBv2SteppableModel, CBv2HistoricalAttentionCheckpointProviding {
    let cbv2SupportsHistoricalAttentionCheckpoint = true

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        let batch = tokens.dim(0), length = tokens.dim(1)
        var hidden = tokens.asType(.float32).reshaped([batch, 1, length, 1]) / Float(32)
        for cache in caches {
            let q = MLXArray.zeros([batch, 2, length, 64], dtype: .float32)
            let result: MLXArray
            if let owner = cache.kind.sharesKVWithLayer {
                result = cache.attendBorrowing(source: caches[owner], queries: q, scale: 0.125, sinks: nil)
            } else {
                let kv = broadcast(hidden, to: [batch, 1, length, 64])
                result = cache.updateAndAttend(queries: q, keys: kv, values: kv, scale: 0.125, sinks: nil)
            }
            hidden = mean(result, axes: [1, 3], keepDims: true)
        }
        let target = MLX.round(hidden.reshaped([batch, length, 1]) * Float(128)).asType(.int32) % 31
        return MLX.where(MLXArray(Int32(0) ..< Int32(32)) .== target, Float(10), Float(-10))
    }
}

final class HistoricalWindowCheckpointEngineTests: XCTestCase {
    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }

    private func engine(_ store: CompleteCheckpointFixtureStore) throws -> (EngineV2, PagedKVBackend) {
        let kinds = [
            CBv2LayerKind(attention: .slidingWindow(17), headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .slidingWindow(17), sharesKVWithLayer: 0,
                          headDim: 64, kvHeads: 1, queryHeads: 2),
            CBv2LayerKind(attention: .full, sharesKVWithLayer: 1,
                          headDim: 64, kvHeads: 1, queryHeads: 2),
        ]
        let backend = try PagedKVBackend(layerKinds: kinds, config: .init(capacityBytes: 256 << 20,
            maxPrefillChunk: chunk, segmentSizeBytes: 64 << 10, layerDTypes: Array(repeating: .float32, count: 4)))
        let engine = EngineV2(model: HistoricalAttentionModel(), layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: chunk,
                prefillChunkSize: chunk, maxWaiting: 4, enablePrefixCache: true),
            admissionConfig: .init(watermarkFraction: 0), completePrefixCache: store)
        XCTAssertEqual(engine.completeCheckpointCodec?.backendLayout,
                       CBv2CompleteCheckpointManifest.historicalAttentionLayout)
        return (engine, backend)
    }

    func testRestartUsesHistoricalWindowAfterDonorHasWrappedSeveralTimes() async throws {
        let store = CompleteCheckpointFixtureStore(segmentBytes: 258)
        let (first, firstBackend) = try engine(store)
        let tokens = (0 ..< 3 * chunk + 1).map { ($0 * 7) % 29 }
        let request = CBv2Request(id: .init(1), promptTokens: tokens, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(1001))
        let cold = await cbv2SchedCollect(try first.submit(request))
        XCTAssertEqual(cold.finishReason, .length)
        XCTAssertEqual(store.saved.map(\.manifest.position), [chunk, 3 * chunk])
        XCTAssertEqual(first.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(firstBackend.bytesWired, 0)
        await first.shutdown()

        // Reopen only the EARLIER boundary, whose donor ring was overwritten
        // twice before publication. Reusing the terminal window cannot pass.
        let reopened = CompleteCheckpointFixtureStore(archives: store.saved.filter { $0.manifest.position == chunk })
        let (second, secondBackend) = try engine(reopened)
        let warmRequest = CBv2Request(id: .init(2), promptTokens: tokens, maxTokens: 4,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(2002))
        XCTAssertTrue(try reopened.stage(engine: second, request: warmRequest))
        let warm = await cbv2SchedCollect(try second.submit(warmRequest))
        XCTAssertEqual(warm.tokens, cold.tokens)
        XCTAssertEqual(warm.usage?.prefixCachePrefillTokensSaved, chunk)
        XCTAssertEqual(warm.usage?.prefixCacheReplayTokens, 0)
        XCTAssertEqual(second.admissionForTesting.bytesReserved, 0)
        XCTAssertEqual(secondBackend.bytesWired, 0)
        await second.shutdown()
    }

    func testExpiredStageClosesBeforeSubmissionAndFallsBackToCold() async throws {
        let store = CompleteCheckpointFixtureStore()
        let (first, _) = try engine(store)
        let tokens = (0 ..< 2 * chunk + 1).map { ($0 * 11) % 29 }
        let request = CBv2Request(id: .init(1), promptTokens: tokens, maxTokens: 2,
            cacheSalt: "tenant", prefixCacheReceiptID: .init(3001))
        let cold = await cbv2SchedCollect(try first.submit(request))
        await first.shutdown()
        let reopened = CompleteCheckpointFixtureStore(archives: store.saved)
        let (second, backend) = try engine(reopened)
        XCTAssertTrue(try reopened.stage(engine: second, request: request))
        let stage = reopened.takeStaged(requestID: .init(3001), tokens: tokens, cacheSalt: "tenant",
                                       maximumSequenceLength: tokens.count + 2)
        XCTAssertNotNil(stage)
        stage?.close()
        let result = await cbv2SchedCollect(try second.submit(request))
        XCTAssertEqual(result.tokens, cold.tokens)
        XCTAssertEqual(result.usage?.prefixCachePrefillTokensSaved, 0)
        // The closed stage still owns its small manifest until this handle is
        // dropped; it never owns a page or native destination after expiry.
        XCTAssertEqual(backend.bytesWired, 0)
        await second.shutdown()
    }

    func testRawNativeCopyErrorTerminatesCohortWithoutSampleOrDonation() async throws {
        for duringConstruction in [true, false] {
            let store = CompleteCheckpointFixtureStore()
            let (engine, backend) = try engine(store)
            engine.loopForTesting.onEngineQueueSync {
                engine.completeCheckpointCapture?.makeHistoricalWindow = { row, position, admission in
                    if duringConstruction {
                        return try CBv2HistoricalWindow(row: row, position: position, admission: admission,
                            afterConstruction: { _ in throw MLXError.caught("injected native construction failure") })
                    }
                    return try CBv2HistoricalWindow(row: row, position: position, admission: admission,
                        evaluate: { array in
                            try withError { eval(array) }
                            throw MLXError.caught("injected native copy failure")
                        })
                }
            }
            let result = await cbv2SchedCollect(try engine.submit(.init(id: .init(1),
                promptTokens: Array(repeating: 7, count: 2 * chunk + 1), maxTokens: 3,
                prefixCacheReceiptID: .init(4001))))
            if case .error? = result.finishReason {} else { XCTFail("native error was swallowed as an optional cache miss") }
            XCTAssertTrue(result.tokens.isEmpty)
            XCTAssertTrue(store.saved.isEmpty)
            XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
            XCTAssertEqual(backend.bytesWired, 0)
            await engine.shutdown()
        }
    }

    func testCancelAndShutdownCannotLaunchSuccessorBeforePrivateCopyDrains() async throws {
        for shuttingDown in [false, true] {
            let store = CompleteCheckpointFixtureStore()
            let (engine, backend) = try engine(store)
            let entered = DispatchSemaphore(value: 0)
            let release = DispatchSemaphore(value: 0)
            let firstCopy = DispatchSemaphore(value: 0)
            firstCopy.signal()
            engine.loopForTesting.onEngineQueueSync {
                engine.completeCheckpointCapture?.makeHistoricalWindow = { row, position, admission in
                    try CBv2HistoricalWindow(row: row, position: position, admission: admission,
                        evaluate: { array in
                            if firstCopy.wait(timeout: .now()) == .success {
                                entered.signal()
                                _ = release.wait(timeout: .now() + 10)
                            }
                            try withError { eval(array) }
                        })
                }
            }
            let stream = try engine.submit(.init(id: .init(1),
                promptTokens: Array(repeating: 7, count: 2 * chunk + 1), maxTokens: 2,
                prefixCacheReceiptID: .init(5001)))
            let collected = Task { await cbv2SchedCollect(stream) }
            let blocked = await withCheckedContinuation { continuation in
                DispatchQueue.global().async {
                    continuation.resume(returning: entered.wait(timeout: .now() + 5) == .success)
                }
            }
            XCTAssertTrue(blocked)
            // The queue is inside first-step private readback. No later graph
            // can have run or advanced the ring while this owner is pending.
            XCTAssertEqual(engine.stepCount, 1)
            XCTAssertGreaterThan(engine.admissionForTesting.bytesReserved, 0)
            let shutdown = shuttingDown ? Task { await engine.shutdown() } : nil
            if !shuttingDown { engine.cancel(.init(1)) }
            release.signal()
            let result = await collected.value
            if shuttingDown {
                await shutdown?.value
            } else {
                XCTAssertEqual(result.finishReason, .cancelled)
                await engine.shutdown()
            }
            XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
            XCTAssertEqual(backend.bytesWired, 0)
        }
    }

}
