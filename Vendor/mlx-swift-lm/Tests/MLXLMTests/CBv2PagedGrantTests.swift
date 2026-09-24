import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 dynamic paged grant", .serialized)
struct CBv2PagedGrantTests {
    private let pageBytes = 2 * 16 * 64 * 2
    private var kind: CBv2LayerKind {
        .init(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
    }
    private func backend(capacityBytes: Int) throws -> PagedKVBackend {
        try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: capacityBytes, dtype: .bfloat16, maxPrefillChunk: 64,
            nominalMaxSequenceLength: 2048, maxBufferLength: 1 << 20,
            segmentSizeBytes: 9 * pageBytes))
    }
    private func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }

    private func footprint(physicalPages: Int) throws -> Int {
        try Memory.allocationFootprintUpperBound(byteCount: physicalPages * pageBytes)
    }

    private func storageBytes(_ backend: PagedKVBackend) throws -> (logical: Int, allocated: Int) {
        var logical = 0, allocated = 0
        for group in backend.pool.groups.values {
            for segment in group.segments.values {
                let observed = try segment.storage.evaluatedBufferInfo()
                let info = try #require(observed)
                logical += segment.storage.nbytes
                allocated += info.allocatedBytes
            }
        }
        return (logical, allocated)
    }

    @Test func shrinkPreservesOwnersAndRegrowthReusesGenerationCheckedRanges() throws {
        let growthBound = try footprint(physicalPages: 5)
        let backend = try backend(capacityBytes: growthBound)
        let first = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        let row = try #require(first[0] as? PagedSequenceKV)
        let k = MLXArray.ones([1, 64, 64], dtype: .bfloat16)
        row.write(keys: k, values: k)
        let snapshot = row.snapshot()
        eval(snapshot.keys, snapshot.values)
        let group = backend.pool.group(row.groupKey)
        let handle = group.currentHandle(row.table[0])
        let original = group.segments.mapValues { ObjectIdentifier($0.storage) }
        let firstBytes = try storageBytes(backend)
        #expect(firstBytes.logical == 5 * pageBytes && backend.bytesWired == firstBytes.allocated)

        backend.updateBytesCapacity(firstBytes.allocated + growthBound)
        let second = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        let bothBytes = try storageBytes(backend)
        #expect(bothBytes.logical == 10 * pageBytes && backend.bytesWired == bothBytes.allocated)
        for (index, identity) in original { #expect(ObjectIdentifier(group.segments[index]!.storage) == identity) }
        #expect(bytes(row.snapshot().keys) == bytes(k.expandedDimensions(axis: 0)))

        backend.updateBytesCapacity(0)
        #expect(backend.bytesCapacity == 0)
        #expect(backend.pool.segmentStorageSnapshot?.overGrantBytes == bothBytes.allocated)
        var allocationAttempts = 0
        backend.pool.slabEval = { _ in allocationAttempts += 1 }
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 1)
        }
        #expect(allocationAttempts == 0)
        backend.pool.slabEval = { eval($0) }
        backend.release(first)
        #expect(backend.bytesWired == bothBytes.allocated - firstBytes.allocated)
        backend.release(second)
        #expect(backend.bytesWired == 0 && backend.bytesReserved == 0)
        #expect(backend.pool.segmentStorageSnapshot?.overGrantBytes == 0)
        #expect(!group.isValid(handle))

        backend.updateBytesCapacity(growthBound)
        let third = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        let reused = try #require(third[0] as? PagedSequenceKV)
        reused.write(keys: k, values: k)
        eval(reused.snapshot().keys)
        #expect(reused.table[0] == handle.page)
        #expect(group.currentHandle(reused.table[0]).generation != handle.generation)
        backend.release(third)
        #expect(backend.bytesWired == 0)
    }

    @Test func reducedGrantAllowsReuseWithoutIncreasingPhysicalDebt() throws {
        let initialBound = try footprint(physicalPages: 3) + 2 * footprint(physicalPages: 2)
        let backend = try backend(capacityBytes: initialBound)
        let first = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 48)
        let second = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 16)
        let firstRow = try #require(first[0] as? PagedSequenceKV)
        let secondRow = try #require(second[0] as? PagedSequenceKV)
        // Interleave physical ownership so releasing the larger reservation
        // leaves one live and one free page in the same native segment.
        let single = MLXArray.ones([1, 1, 64], dtype: .bfloat16)
        let pair = MLXArray.ones([1, 32, 64], dtype: .bfloat16)
        secondRow.write(keys: single, values: single)
        firstRow.write(keys: pair, values: pair)
        eval(firstRow.snapshot().keys, secondRow.snapshot().keys)
        backend.release(first)
        let retainedBytes = try storageBytes(backend)
        #expect(retainedBytes.logical == 3 * pageBytes && backend.bytesWired == retainedBytes.allocated)
        backend.updateBytesCapacity(2 * pageBytes)
        let before = backend.pool.segmentStorageSnapshot
        let third = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 16)
        #expect(backend.bytesWired == before?.committedBytes)
        #expect(backend.pool.segmentStorageSnapshot?.overGrantBytes == retainedBytes.allocated - 2 * pageBytes)
        #expect(backend.bytesReserved == 2 * pageBytes)
        backend.release(second)
        backend.release(third)
        #expect(backend.bytesWired == 0 && backend.bytesReserved == 0)
    }

    @Test func grantEpochChangeDiscardsPreparedGrowthWithoutPublishingPages() throws {
        let backend = try backend(capacityBytes: 100 * pageBytes)
        let first = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        defer { backend.release(first) }
        let group = backend.pool.group(backend.pool.groupKey(forLayer: 0))
        let identities = group.segments.mapValues { ObjectIdentifier($0.storage) }
        let free = group.freeList, generations = group.generations
        let reserved = backend.bytesReserved, committed = backend.bytesWired
        var changed = false
        backend.pool.slabEval = { array in
            eval(array)
            if !changed {
                changed = true
                // Both ceilings have room. Epoch identity, not just bytes,
                // must reject publication prepared against the old grant.
                backend.updateBytesCapacity(99 * pageBytes)
            }
        }
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 128)
        }
        #expect(changed)
        #expect(group.segments.mapValues { ObjectIdentifier($0.storage) } == identities)
        #expect(group.freeList == free && group.generations == generations)
        #expect(backend.bytesReserved == reserved && backend.bytesWired == committed)
        #expect(backend.bytesCapacity == 99 * pageBytes)
        backend.pool.slabEval = { eval($0) }
        let retry = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 128)
        backend.release(retry)
        #expect(backend.bytesWired == committed)
    }

    @Test func sizeClassReuseBoundsHistoricalAddressMetadata() throws {
        var layout = try PagedKVSegmentLayout(
            pageBytes: 4096, targetBytes: 9 * 4096, maximumBufferBytes: 64 << 20)
        let demands = [1, 2, 3, 4, 7, 8, 9, 15, 31, 63]
        for demand in demands { layout = try layout.adding(usablePages: demand, excluding: []).layout }
        let highWater = layout.pageCount
        for _ in 0 ..< 32 {
            for demand in demands.reversed() {
                let planned = try layout.adding(usablePages: demand, excluding: [])
                #expect(planned.segmentIDs.reduce(0) { $0 + planned.layout.range($1).count - 1 } == demand)
                layout = planned.layout
                #expect(layout.pageCount == highWater)
            }
        }
        let bounded = try PagedKVSegmentLayout(
            pageBytes: 4096, targetBytes: 9 * 4096, maximumBufferBytes: 64 << 20,
            maximumAddressPages: 17)
        #expect(throws: CBv2KVError.self) {
            try bounded.adding(usablePages: 16, excluding: [])
        }
        #expect(bounded.pageCount == 0)
    }
}
