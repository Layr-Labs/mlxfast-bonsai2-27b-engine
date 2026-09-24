import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

final class CBv2HybridPrefixCacheTests: XCTestCase {
    private let spec = CBv2RecurrentStateSpec(layers: [
        .init(modelLayerIndex: 0, convShape: [1, 2, 2], convDType: .float32,
              ssmShape: [1, 1, 2, 2], ssmDType: .float32)
    ])

    private func cache(bytes: Int = 4096, checkpoints: Int = 2, entries: Int = 4) -> CBv2HybridPrefixCache {
        .init(config: .init(
            maximumBytes: bytes, maximumEntries: entries, maximumCheckpointsPerRequest: checkpoints,
            modelID: "model", promptContractID: "template", buildID: "test-build"))
    }

    private func stage(_ cache: CBv2HybridPrefixCache, id: UInt64, position: Int) {
        let roots = cache.capture(
            requestID: .init(id), position: position, chunkSize: 4, spec: spec,
            layers: [0: .init(conv: MLXArray.zeros([1, 2, 2]), ssm: MLXArray.zeros([1, 1, 2, 2]))])
        if !roots.isEmpty { asyncEval(roots) }
    }

    private func publish(
        _ cache: CBv2HybridPrefixCache, id: UInt64, tokens: [Int], salt: String?,
        backingBytes: Int = 256, receipt: UInt64? = nil
    ) async {
        let row = MLXArray.zeros([1, 1, tokens.count, 2])
        // Production gets evaluated KV from a retired row. Detach test KV
        // here too so the queue only waits on an immutable scheduled array.
        asyncEval(row)
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            cache.publish(
                requestID: .init(id), tokens: tokens, cacheSalt: salt,
                receiptID: receipt.map(CBv2RequestID.init),
                kv: [(keys: row, values: row, offset: tokens.count)], backingBytes: backingBytes
            ) { continuation.resume() }
        }
    }

    func testExactCheckpointScopeAndActualAdoptionAccounting() async throws {
        let cache = cache()
        stage(cache, id: 1, position: 4)
        stage(cache, id: 1, position: 8)
        await publish(cache, id: 1, tokens: Array(0 ..< 12), salt: "tenant-a")
        XCTAssertNil(cache.lookup(tokens: Array(0 ..< 12), cacheSalt: "tenant-b", maximumChunkSize: 4))
        XCTAssertNil(cache.lookup(tokens: [0, 1, 2, 99, 4], cacheSalt: "tenant-a", maximumChunkSize: 4))
        let hit = try XCTUnwrap(cache.lookup(
            tokens: [0, 1, 2, 3, 20, 21], cacheSalt: "tenant-a", maximumChunkSize: 4))
        XCTAssertEqual(hit.checkpoint.position, 4)
        XCTAssertEqual(cache.stats.adoptions, 0)
        cache.endAdoption(pin: hit.pin, tokensSaved: 4)
        XCTAssertEqual(cache.stats.adoptions, 1)
        XCTAssertEqual(cache.stats.tokensSaved, 4)
        // A complete repeated eight-token prompt still needs a live logit
        // token, so its latest eligible checkpoint is four, not eight.
        let repeatHit = try XCTUnwrap(cache.lookup(
            tokens: Array(0 ..< 8), cacheSalt: "tenant-a", maximumChunkSize: 4))
        XCTAssertEqual(repeatHit.checkpoint.position, 4)
        cache.endAdoption(pin: repeatHit.pin)
        cache.close()
        XCTAssertEqual(cache.stats.retainedBytes, 0)
    }

    func testPinnedStorageAndPublishingShareHardByteBudget() async throws {
        let cache = cache(bytes: 600)
        stage(cache, id: 1, position: 4)
        await publish(cache, id: 1, tokens: Array(0 ..< 8), salt: nil, backingBytes: 512)
        let hit = try XCTUnwrap(cache.lookup(tokens: Array(0 ..< 8), cacheSalt: nil, maximumChunkSize: 4))
        stage(cache, id: 2, position: 4)
        XCTAssertLessThanOrEqual(cache.stats.retainedBytes, 600)
        await publish(cache, id: 2, tokens: Array(10 ..< 18), salt: nil, backingBytes: 512)
        XCTAssertEqual(cache.stats.entries, 1, "an in-use donor cannot fund another publication")
        XCTAssertGreaterThan(cache.stats.capacityRefusals, 0)
        XCTAssertLessThanOrEqual(cache.stats.retainedBytes, 600)
        cache.close()
        XCTAssertEqual(cache.stats.entries, 1, "close preserves the adoption pin")
        cache.endAdoption(pin: hit.pin)
        XCTAssertEqual(cache.stats.retainedBytes, 0)
    }

    func testOversizedDonorCompactsBothExactEndpointsWithoutChangingKVBits() async throws {
        let cache = cache(bytes: 192)
        stage(cache, id: 1, position: 4)
        stage(cache, id: 1, position: 8)
        let values: [Float] = [0, -0.0, 1, -1, .infinity, -.infinity, 2.5, -2.5]
        let data = Array(repeating: values, count: 3).flatMap { $0 }
        let row = MLXArray(data).reshaped([1, 1, 12, 2])
        asyncEval(row)
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            cache.publish(
                requestID: .init(1), tokens: Array(0 ..< 12), cacheSalt: "tenant",
                kv: [(keys: row, values: row, offset: 12)], backingBytes: row.nbytes * 2
            ) { done.resume() }
        }
        XCTAssertEqual(cache.stats.retainedBytes, 192, "128 compact KV bytes plus two 32-byte checkpoints")
        XCTAssertEqual(cache.stats.checkpoints, 2)
        XCTAssertEqual(cache.stats.kvCompactions, 1)
        XCTAssertEqual(cache.stats.kvCompactionBytes, 128)
        XCTAssertEqual(cache.stats.capacityRefusals, 0)
        let latest = try XCTUnwrap(cache.lookup(
            tokens: Array(0 ..< 12), cacheSalt: "tenant", maximumChunkSize: 4))
        XCTAssertEqual(latest.checkpoint.position, 8)
        XCTAssertEqual(latest.kvBackingBytes, 128)
        let copied = try XCTUnwrap(latest.kvPrefix[0])
        XCTAssertEqual(copied.keys.dtype, .float32)
        XCTAssertEqual(copied.keys.asArray(Float.self).map(\.bitPattern), data.prefix(16).map(\.bitPattern))
        XCTAssertEqual(copied.values.asArray(Float.self).map(\.bitPattern), data.prefix(16).map(\.bitPattern))
        cache.endAdoption(pin: latest.pin, tokensSaved: 8)
        let branch = try XCTUnwrap(cache.lookup(
            tokens: [0, 1, 2, 3, 99], cacheSalt: "tenant", maximumChunkSize: 4))
        XCTAssertEqual(branch.checkpoint.position, 4)
        cache.endAdoption(pin: branch.pin, tokensSaved: 4)
        cache.close()
        XCTAssertEqual(cache.stats.retainedBytes, 0)
    }

    func testCompactionKeepsEarlierCheckpointWhenLatestCannotFit() async throws {
        let cache = cache(bytes: 96)
        stage(cache, id: 1, position: 4)
        stage(cache, id: 1, position: 8)
        await publish(cache, id: 1, tokens: Array(0 ..< 12), salt: nil)
        XCTAssertEqual(cache.stats.retainedBytes, 96, "64 compact KV bytes plus the first 32-byte checkpoint")
        XCTAssertEqual(cache.stats.checkpoints, 1)
        XCTAssertEqual(cache.stats.kvCompactions, 1)
        XCTAssertEqual(cache.stats.capacityRefusals, 0)
        let hit = try XCTUnwrap(cache.lookup(tokens: Array(0 ..< 12), cacheSalt: nil, maximumChunkSize: 4))
        XCTAssertEqual(hit.checkpoint.position, 4)
        cache.endAdoption(pin: hit.pin)
        cache.close()
    }

    func testNoCompactionOrReceiptWhenFirstCheckpointCannotFitOrCacheClosed() async throws {
        let tooSmall = cache(bytes: 64)
        let unexpected = DispatchSemaphore(value: 0)
        tooSmall.setPublicationHandler { _, _ in unexpected.signal() }
        stage(tooSmall, id: 1, position: 4)
        await publish(tooSmall, id: 1, tokens: Array(0 ..< 12), salt: nil, receipt: 101)
        XCTAssertEqual(tooSmall.stats.retainedBytes, 0)
        XCTAssertEqual(tooSmall.stats.kvCompactions, 0)
        XCTAssertEqual(tooSmall.stats.kvCompactionBytes, 0)
        XCTAssertEqual(tooSmall.stats.capacityRefusals, 1)
        tooSmall.close()

        let closed = cache(bytes: 192)
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        stage(closed, id: 1, position: 4)
        XCTAssertTrue(closed.dropStaged(requestID: .init(1)) {
            entered.signal()
            _ = release.wait(timeout: .now() + 10)
        })
        let blocked = await cbv2SchedWait { entered.wait(timeout: .now()) == .success }
        XCTAssertTrue(blocked)
        closed.setPublicationHandler { _, _ in unexpected.signal() }
        stage(closed, id: 2, position: 4)
        let row = MLXArray.zeros([1, 1, 12, 2])
        asyncEval(row)
        let done = expectation(description: "closed fallback drains without copying")
        closed.publish(
            requestID: .init(2), tokens: Array(0 ..< 12), cacheSalt: nil, receiptID: .init(102),
            kv: [(keys: row, values: row, offset: 12)], backingBytes: row.nbytes * 2
        ) { done.fulfill() }
        closed.close()
        release.signal()
        await fulfillment(of: [done], timeout: 5)
        XCTAssertEqual(closed.stats.retainedBytes, 0)
        XCTAssertEqual(closed.stats.kvCompactions, 0)
        XCTAssertEqual(closed.stats.kvCompactionBytes, 0)
        XCTAssertEqual(unexpected.wait(timeout: .now()), .timedOut)
    }

    func testRollingCheckpointsAdvanceAndCancellationNeverPublishes() async throws {
        let cache = cache()
        stage(cache, id: 1, position: 4)
        stage(cache, id: 1, position: 8)
        stage(cache, id: 1, position: 12)
        await publish(cache, id: 1, tokens: Array(0 ..< 16), salt: nil)
        let hit = try XCTUnwrap(cache.lookup(tokens: Array(0 ..< 16), cacheSalt: nil, maximumChunkSize: 4))
        XCTAssertEqual(hit.checkpoint.position, 12)
        XCTAssertEqual(cache.stats.checkpoints, 2)
        cache.endAdoption(pin: hit.pin)
        stage(cache, id: 2, position: 4)
        cache.dropStaged(requestID: .init(2))
        XCTAssertNil(cache.lookup(tokens: Array(20 ..< 28), cacheSalt: nil, maximumChunkSize: 4))
        cache.close()
        let drained = await cbv2SchedWait { cache.stats.retainedBytes == 0 }
        XCTAssertTrue(drained)
    }
    func testWarmContinuationRetainsOriginalBranchAfterDonorEviction() async throws {
        let cache = cache(entries: 1)
        stage(cache, id: 1, position: 4)
        await publish(cache, id: 1, tokens: Array(0 ..< 8), salt: "tenant")
        let before = cache.stats.retainedBytes
        let hit = try XCTUnwrap(cache.lookup(
            tokens: Array(0 ..< 16), cacheSalt: "tenant", maximumChunkSize: 4))
        cache.inheritCheckpoint(pin: hit.pin, checkpoint: hit.checkpoint, requestID: .init(2))
        XCTAssertEqual(cache.stats.retainedBytes, before, "shared immutable roots are charged once")
        cache.endAdoption(pin: hit.pin, tokensSaved: 4)
        stage(cache, id: 2, position: 8)
        stage(cache, id: 2, position: 12)
        await publish(cache, id: 2, tokens: Array(0 ..< 16), salt: "tenant")
        XCTAssertEqual(cache.stats.entries, 1)
        XCTAssertGreaterThan(cache.stats.evictions, 0)
        XCTAssertEqual(cache.stats.checkpoints, 2)
        let original = try XCTUnwrap(cache.lookup(
            tokens: Array(0 ..< 8), cacheSalt: "tenant", maximumChunkSize: 4))
        XCTAssertEqual(original.checkpoint.position, 4)
        let latest = try XCTUnwrap(cache.lookup(
            tokens: Array(0 ..< 16), cacheSalt: "tenant", maximumChunkSize: 4))
        XCTAssertEqual(latest.checkpoint.position, 12)
        cache.endAdoption(pin: original.pin)
        cache.endAdoption(pin: latest.pin)
        cache.close()
        XCTAssertEqual(cache.stats.retainedBytes, 0)
    }

    func testRefusedRollingCapturePreservesExistingLatestCheckpoint() async throws {
        let cache = cache()
        stage(cache, id: 1, position: 4)
        stage(cache, id: 1, position: 8)
        let retained = cache.stats.retainedBytes
        XCTAssertEqual(cache.resizeReservation(to: retained), retained)
        stage(cache, id: 1, position: 12)
        XCTAssertEqual(cache.stats.retainedBytes, retained)
        XCTAssertGreaterThan(cache.stats.capacityRefusals, 0)
        _ = cache.resizeReservation(to: 4096)
        await publish(cache, id: 1, tokens: Array(0 ..< 16), salt: nil)
        let hit = try XCTUnwrap(cache.lookup(tokens: Array(0 ..< 16), cacheSalt: nil, maximumChunkSize: 4))
        XCTAssertEqual(hit.checkpoint.position, 8)
        cache.endAdoption(pin: hit.pin)
        cache.close()
        XCTAssertEqual(cache.stats.retainedBytes, 0)
    }

    func testRepeatedLatestEndpointDoesNotDuplicateBackingOrRegressAfterEviction() async throws {
        let cache = cache(entries: 1)
        let prompt = Array(0 ..< 12)
        stage(cache, id: 1, position: 4)
        stage(cache, id: 1, position: 8)
        await publish(cache, id: 1, tokens: prompt, salt: nil)
        let initialBytes = cache.stats.retainedBytes
        for id in UInt64(2) ... 6 {
            let hit = try XCTUnwrap(cache.lookup(tokens: prompt, cacheSalt: nil, maximumChunkSize: 4))
            XCTAssertEqual(hit.checkpoint.position, 8)
            cache.inheritCheckpoint(pin: hit.pin, checkpoint: hit.checkpoint, requestID: .init(id))
            cache.endAdoption(pin: hit.pin, tokensSaved: 8)
            await publish(cache, id: id, tokens: prompt, salt: nil)
            XCTAssertEqual(cache.stats.entries, 1)
            XCTAssertEqual(cache.stats.checkpoints, 2)
            XCTAssertEqual(cache.stats.retainedBytes, initialBytes)
            XCTAssertEqual(cache.stats.evictions, 0, "identical repeats keep one useful KV backing")
        }
        let hit = try XCTUnwrap(cache.lookup(tokens: prompt, cacheSalt: nil, maximumChunkSize: 4))
        cache.inheritCheckpoint(pin: hit.pin, checkpoint: hit.checkpoint, requestID: .init(7))
        cache.endAdoption(pin: hit.pin)
        // Another tenant evicts the donor before the warm request completes.
        stage(cache, id: 8, position: 4)
        await publish(cache, id: 8, tokens: Array(20 ..< 28), salt: "other")
        XCTAssertNil(cache.candidate(tokens: prompt, cacheSalt: nil, maximumChunkSize: 4))
        await publish(cache, id: 7, tokens: prompt, salt: nil)
        let latest = try XCTUnwrap(cache.lookup(tokens: prompt, cacheSalt: nil, maximumChunkSize: 4))
        XCTAssertEqual(latest.checkpoint.position, 8)
        cache.endAdoption(pin: latest.pin)
        let earlier = try XCTUnwrap(cache.lookup(tokens: Array(0 ..< 8), cacheSalt: nil, maximumChunkSize: 4))
        XCTAssertEqual(earlier.checkpoint.position, 4)
        cache.endAdoption(pin: earlier.pin)
        cache.close()
        XCTAssertEqual(cache.stats.retainedBytes, 0)
    }

    func testLastPinReleaseHonorsShrinkWithoutClosingCache() async throws {
        let cache = cache()
        stage(cache, id: 1, position: 4)
        await publish(cache, id: 1, tokens: Array(0 ..< 8), salt: nil)
        let hit = try XCTUnwrap(cache.lookup(tokens: Array(0 ..< 8), cacheSalt: nil, maximumChunkSize: 4))
        let pinnedBytes = cache.stats.retainedBytes
        XCTAssertEqual(cache.resizeReservation(to: 0), pinnedBytes)
        cache.endAdoption(pin: hit.pin)
        XCTAssertEqual(cache.stats.retainedBytes, 0)
        XCTAssertEqual(cache.stats.entries, 0)
        XCTAssertNil(cache.candidate(tokens: Array(0 ..< 8), cacheSalt: nil, maximumChunkSize: 4))
        _ = cache.resizeReservation(to: 4096)
        stage(cache, id: 2, position: 4)
        await publish(cache, id: 2, tokens: Array(0 ..< 8), salt: nil)
        XCTAssertNotNil(cache.candidate(tokens: Array(0 ..< 8), cacheSalt: nil, maximumChunkSize: 4))
        cache.close()
    }

    func testResizeAndCloseKeepPinnedAndQueuedPublicationChargedUntilRelease() async throws {
        let cache = cache()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let unexpectedPublication = DispatchSemaphore(value: 0)
        cache.setPublicationHandler { receipt, _ in
            if receipt == .init(101) {
                entered.signal()
                _ = release.wait(timeout: .now() + 10)
            } else {
                unexpectedPublication.signal()
            }
        }
        stage(cache, id: 1, position: 4)
        let firstRow = MLXArray.zeros([1, 1, 8, 2])
        asyncEval(firstRow)
        let first = expectation(description: "pinned publication completes")
        cache.publish(
            requestID: .init(1), tokens: Array(0 ..< 8), cacheSalt: nil, receiptID: .init(101),
            kv: [(keys: firstRow, values: firstRow, offset: 8)], backingBytes: 256
        ) { first.fulfill() }
        let blocked = await cbv2SchedWait { entered.wait(timeout: .now()) == .success }
        XCTAssertTrue(blocked)
        let pin = try XCTUnwrap(cache.lookup(tokens: Array(0 ..< 8), cacheSalt: nil, maximumChunkSize: 4))
        let pinnedBytes = cache.stats.residentBytes
        stage(cache, id: 2, position: 4)
        let row = MLXArray.zeros([1, 1, 8, 2])
        asyncEval(row)
        let second = expectation(description: "closed queued publication completes")
        cache.publish(
            requestID: .init(2), tokens: Array(10 ..< 18), cacheSalt: nil, receiptID: .init(102),
            kv: [(keys: row, values: row, offset: 8)], backingBytes: 256
        ) { second.fulfill() }
        let retained = cache.stats.retainedBytes
        XCTAssertGreaterThan(retained, pinnedBytes)
        XCTAssertEqual(cache.resizeReservation(to: 0), retained)
        cache.close()
        XCTAssertEqual(cache.stats.retainedBytes, retained, "live queue and adoption retain their reservation")
        XCTAssertNil(cache.lookup(tokens: Array(0 ..< 8), cacheSalt: nil, maximumChunkSize: 4))
        release.signal()
        await fulfillment(of: [first, second], timeout: 5)
        XCTAssertEqual(unexpectedPublication.wait(timeout: .now()), .timedOut)
        XCTAssertEqual(cache.stats.retainedBytes, pinnedBytes)
        XCTAssertEqual(cache.resizeReservation(to: 0), pinnedBytes)
        cache.endAdoption(pin: pin.pin)
        XCTAssertEqual(cache.stats.retainedBytes, 0)
        XCTAssertEqual(cache.resizeReservation(to: 0), 0)
    }

}
