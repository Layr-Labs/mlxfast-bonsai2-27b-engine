import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("CBv2 segmented paged storage", .serialized)
struct CBv2PagedSegmentTests {
    private func kind(_ dim: Int = 64) -> CBv2LayerKind {
        CBv2LayerKind(attention: .full, headDim: dim, kvHeads: 1, queryHeads: 2)
    }

    private func backend(dtype: DType = .bfloat16, segmentPages: Int = 3,
                         capacityBytes: Int = 4 << 20, kinds: [CBv2LayerKind]? = nil) throws -> PagedKVBackend {
        let pageBytes = 2 * 16 * 64 * dtype.size
        return try PagedKVBackend(
            layerKinds: kinds ?? [kind()],
            config: PagedKVPoolConfig(
                capacityBytes: capacityBytes, dtype: dtype, maxPrefillChunk: 512,
                nominalMaxSequenceLength: 2048, maxBufferLength: 16 << 20,
                segmentSizeBytes: pageBytes * segmentPages))
    }

    private func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }

    @Test func layoutDoesNotCapSegmentCountOrExceedBufferLimit() throws {
        let layout = try PagedKVSegmentLayout(
            pageCount: 1_000_001, pageBytes: 32_768,
            targetBytes: 64 << 20, maximumBufferBytes: 16 << 20)
        #expect(layout.segmentCount > 17)
        #expect(layout.physicalBytes <= 1_000_001 * 32_768)
        #expect(layout.usablePageCount == layout.pageCount - layout.segmentCount)
        for index in 0 ..< layout.segmentCount {
            let range = layout.range(index)
            #expect(range.count >= 2)
            #expect(range.count * layout.pageBytes <= 16 << 20)
            #expect(!layout.isUsable(Int32(range.lowerBound)))
            #expect(layout.segmentIndex(page: Int32(range.upperBound - 1)) == index)
        }
        #expect(throws: CBv2KVError.self) {
            try PagedKVSegmentLayout(pageCount: 2, pageBytes: 4096,
                                     targetBytes: 4096, maximumBufferBytes: 8192)
        }
    }

    @Test func wholePartitionMappingFitsEveryPageOffsetAndMetalBindingLimit() throws {
        for pageSize in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let ptok = PagedSegmentDispatchPlan.boundedPartitionTokens(256, pageSize: pageSize)
            for offset in 0 ..< pageSize {
                let tokens = 3 * ptok + 1
                let pageCount = (offset + tokens - 1) / pageSize + 1
                // One usable page per segment deliberately maximizes bindings.
                let layout = try PagedKVSegmentLayout(
                    pageCount: 2 * pageCount, pageBytes: 1,
                    targetBytes: 2, maximumBufferBytes: 2)
                let pages = (0 ..< pageCount).map { Int32(2 * $0 + 1) }
                let plan = PagedSegmentDispatchPlan(
                    rows: [.init(pages: pages, info: .init(
                        attendStart: offset, attendLength: tokens, tableLength: pages.count,
                        writePage: pages.last!, writeSlot: (offset + tokens - 1) % pageSize))],
                    layout: layout, pageSize: pageSize, partitionTokens: ptok, hasWrite: true)
                #expect(plan.buckets.count > (pageCount > 17 ? 1 : 0))
                var partitions: [Int] = []
                for bucket in plan.buckets {
                    #expect(bucket.segmentIDs.count <= 17)
                    #expect(PagedSegmentAttention.inputNames(bindings: bucket.bindingClass).count + 1 <= 31)
                    #expect(!PagedSegmentAttention.body(bindings: bucket.bindingClass).contains("_shape"))
                    #expect(!PagedSegmentAttention.body(bindings: bucket.bindingClass).contains("_strides"))
                    for i in 0 ..< bucket.workCount {
                        let start = i * PagedSegmentDispatchPlan.recordStride
                        let record = Array(bucket.records[start ..< start + PagedSegmentDispatchPlan.recordStride])
                        let partition = Int(record[1])
                        partitions.append(partition)
                        let first = (offset + partition * ptok) / pageSize
                        #expect(Int(record[2]) == first)
                        for page in 0 ..< Int(record[3]) {
                            let segment = bucket.segmentIDs[Int(record[8 + 2 * page])]
                            let global = layout.range(segment).lowerBound + Int(record[9 + 2 * page])
                            #expect(global == Int(pages[first + page]))
                        }
                    }
                }
                #expect(partitions == Array(0 ..< plan.maxPartitions))
            }
        }
        #expect(PagedSegmentAttention.inputNames(bindings: 17).count + 1 == 28)
    }

    @Test(arguments: [false, true])
    func rejectedGrowthLeavesExistingBackingAndReservationUntouched(inSecondGroup: Bool) throws {
        struct InjectedFailure: Error {}
        let kinds = [kind(), kind(128)]
        let backend = try backend(segmentPages: 6, kinds: kinds)
        let initial = try backend.makeSequenceState(layerKinds: kinds, promptLength: 0, maxLength: 16)
        defer { backend.release(initial) }
        let beforeBytes = backend.bytesWired
        let beforeReserved = backend.bytesReserved
        let before = Dictionary(uniqueKeysWithValues: backend.pool.groupKeys.map { key in
            (key, backend.pool.group(key).segments.mapValues { ObjectIdentifier($0.storage) })
        })
        let beforeFree = backend.pool.groups.mapValues(\.freeList)
        let beforeGenerations = backend.pool.groups.mapValues(\.generations)
        var calls = 0
        var reachedSecondGroup = false
        backend.pool.slabEval = { array in
            calls += 1
            reachedSecondGroup = array.dim(4) == 128
            if inSecondGroup ? reachedSecondGroup : calls == 2 { throw InjectedFailure() }
            eval(array)
        }
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: kinds, promptLength: 0, maxLength: 512)
        }
        #expect(calls >= 2)
        #expect(reachedSecondGroup == inSecondGroup)
        #expect(backend.bytesReserved == beforeReserved)
        #expect(backend.bytesWired == beforeBytes)
        for key in backend.pool.groupKeys {
            #expect(backend.pool.group(key).segments.mapValues { ObjectIdentifier($0.storage) } == before[key])
            #expect(backend.pool.group(key).freeList == beforeFree[key])
            #expect(backend.pool.group(key).generations == beforeGenerations[key])
        }
        backend.pool.slabEval = { eval($0) }
        let retry = try backend.makeSequenceState(layerKinds: kinds, promptLength: 0, maxLength: 512)
        #expect(backend.bytesWired > beforeBytes)
        backend.release(retry)
        #expect(backend.bytesWired == beforeBytes)
    }

    @Test(arguments: [DType.bfloat16, .float16, .float32])
    func nativeTransfersGrowWithoutCopyAndRetireAfterRelease(dtype: DType) throws {
        let backend = try backend(dtype: dtype, segmentPages: 3)
        #expect(backend.bytesWired == 0)
        let first = try backend.makeSequenceState(layerKinds: [kind()], promptLength: 0, maxLength: 64)
        let row = try #require(first[0] as? PagedSequenceKV)
        let k = MLXRandom.normal([1, 64, 64], key: MLXRandom.key(11)).asType(dtype)
        let v = MLXRandom.normal([1, 64, 64], key: MLXRandom.key(12)).asType(dtype)
        row.write(keys: k, values: v)
        let snap = row.snapshot()
        eval(snap.keys, snap.values)
        #expect(bytes(snap.keys) == bytes(k.expandedDimensions(axis: 0)))
        #expect(bytes(snap.values) == bytes(v.expandedDimensions(axis: 0)))
        let group = backend.pool.group(row.groupKey)
        let original = group.segments.mapValues { ObjectIdentifier($0.storage) }
        let handle = group.currentHandle(row.table[0])
        let firstOnly = bytes(row.snapshot().keys)
        let sourceBefore = bytes(group.segment(for: row.table[0]).storage)
        let second = try backend.makeSequenceState(layerKinds: [kind()], promptLength: 0, maxLength: 1024)
        #expect(group.segments.count > 17)
        for (id, pointer) in original { #expect(ObjectIdentifier(group.segments[id]!.storage) == pointer) }
        let expected = bytes(k.expandedDimensions(axis: 0))
        #expect(firstOnly == expected)
        #expect(bytes(group.segment(for: row.table[0]).storage) == sourceBefore)
        // A keys-only read must itself depend on completion; separately
        // evaluating the fence cannot repair an already copied stale slice.
        let single = row.snapshot().keys
        #expect(bytes(single) == expected)
        backend.release(second)
        #expect(group.segments.count == original.count)
        backend.release(first)
        #expect(backend.bytesWired == 0)
        #expect(backend.bytesReserved == 0 && backend.bytesInUse == 0)
        #expect(!group.isValid(handle))
        let third = try backend.makeSequenceState(layerKinds: [kind()], promptLength: 0, maxLength: 64)
        let reused = try #require(third[0] as? PagedSequenceKV)
        reused.write(keys: k, values: v)
        eval(reused.snapshot().keys)
        #expect(reused.table[0] == handle.page)
        #expect(group.currentHandle(reused.table[0]).generation != handle.generation)
        backend.release(third)
        #expect(backend.bytesWired == 0)
    }

    @Test(arguments: [DType.bfloat16, .float16, .float32])
    func segmentedDecodeMatchesDirectSlabBitsAcrossBuckets(dtype: DType) throws {
        let requestCount = 4, maximumTokens = 1024
        let pageBytes = 2 * 16 * 64 * dtype.size
        // Each segment has one poison page and one usable page.
        let segmentCount = requestCount * (maximumTokens / 16)
        let capacity = try segmentCount * Memory.allocationFootprintUpperBound(byteCount: 2 * pageBytes)
        let backend = try backend(dtype: dtype, segmentPages: 2, capacityBytes: capacity)
        var states: [[CBv2SequenceKV?]] = []
        var descriptors: [PagedSegmentDispatchPlan.Row] = []
        var snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)] = []
        defer { for state in states { backend.release(state) } }
        for index in 0 ..< requestCount {
            let state = try backend.makeSequenceState(layerKinds: [kind()], promptLength: 0, maxLength: maximumTokens)
            states.append(state)
            let row = try #require(state[0] as? PagedSequenceKV)
            let count = 513 + index * 16
            row.write(keys: MLXRandom.normal([1, count, 64], key: MLXRandom.key(UInt64(index + 30))).asType(dtype),
                      values: MLXRandom.normal([1, count, 64], key: MLXRandom.key(UInt64(index + 40))).asType(dtype))
            descriptors.append(.init(pages: row.table, info: row.seqInfoRow(attending: row.decodeAttendRange)))
            snapshots.append(row.snapshot())
        }
        let group = backend.pool.group(backend.pool.groupKey(forLayer: 0))
        #expect(group.segments.count > 17)
        let q = MLXRandom.normal([4, 2, 64], key: MLXRandom.key(71)).asType(dtype)
        let params = MLXArray([Float(0), 0.125, 0, 0, 0, 0, 0, 0])
        // Build an independent direct-slab oracle from the request snapshots;
        // production segmented decode never performs this concatenation.
        let maxPages = descriptors.map { $0.pages.count }.max()!
        var keyPages: [MLXArray] = [], valuePages: [MLXArray] = []
        var table: [Int32] = []
        var physical = 0
        for (index, snap) in snapshots.enumerated() {
            let tokens = descriptors[index].info.attendLength
            let pages = descriptors[index].pages.count
            let padding = pages * 16 - tokens
            func pageShape(_ a: MLXArray) -> MLXArray {
                let padded = padding == 0 ? a : concatenated([a, MLXArray.zeros([1, 1, padding, 64], dtype: dtype)], axis: 2)
                return padded.reshaped([pages, 1, 16, 64])
            }
            keyPages.append(pageShape(snap.keys)); valuePages.append(pageShape(snap.values))
            table += (0 ..< pages).map { Int32(physical + $0) }
                + Array(repeating: 0, count: maxPages - pages)
            physical += pages
        }
        let kSlab = concatenated(keyPages, axis: 0), vSlab = concatenated(valuePages, axis: 0)
        let (info, maxLength) = PagedAttentionKernel.seqinfo(descriptors.map(\.info))
        let direct = try PagedAttentionKernel.decode(
            queries: q, kSlab: kSlab, vSlab: vSlab,
            tables: MLXArray(table, [4, maxPages]), seqinfo: info,
            maxAttendLength: maxLength, sinks: nil, params: params, softcap: false,
            pageSize: 16, writeFence: group.writeFence, kernelSource: backend.pool.kernelSource).out
        let segmented = PagedSegmentAttention.decode(
            queries: q, newKeys: nil, newValues: nil, group: group, rows: descriptors,
            sinks: nil, params: params, softcap: false, source: backend.pool.kernelSource)
        eval(direct, segmented)
        #expect(bytes(segmented) == bytes(direct))
    }

    @Test(arguments: [DType.bfloat16, .float16, .float32], [1, 2, 4])
    func fusedWindowDecodePreservesBitsAcrossUnalignedSeventeenPageSpans(
        dtype: DType, queryHeads: Int
    ) throws {
        // The one- and two-head cases exercise metadata smaller than MLX's
        // eight-element device-pointer threshold before padding. Both direct
        // and segmented merges must compile and retain identical output bits.
        let kind = CBv2LayerKind(attention: .slidingWindow(256), headDim: 64, kvHeads: 1, queryHeads: queryHeads)
        var config = PagedKVPoolConfig(
            capacityBytes: 2 << 20, dtype: dtype, maxPrefillChunk: 128,
            nominalMaxSequenceLength: 2048, maxBufferLength: 16 << 20)
        let fixed = try PagedKVBackend(layerKinds: [kind], config: config)
        config.segmentSizeBytes = 2 * 2 * 16 * 64 * dtype.size
        let segmented = try PagedKVBackend(layerKinds: [kind], config: config)
        let oldState = try fixed.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 2048)
        let newState = try segmented.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 2048)
        defer { fixed.release(oldState); segmented.release(newState) }
        let oldRow = try #require(oldState[0] as? PagedSequenceKV)
        let newRow = try #require(newState[0] as? PagedSequenceKV)
        for chunk in 0 ..< 4 {
            let k = MLXRandom.normal([1, 128, 64], key: MLXRandom.key(UInt64(90 + chunk))).asType(dtype)
            let v = MLXRandom.normal([1, 128, 64], key: MLXRandom.key(UInt64(100 + chunk))).asType(dtype)
            oldRow.write(keys: k, values: v); newRow.write(keys: k, values: v)
        }
        let oldGroup = fixed.pool.group(oldRow.groupKey)
        let newGroup = segmented.pool.group(newRow.groupKey)
        let params = MLXArray([Float(2), 0.125, 0, 0, 0, 0, 0, 0])
        let sinks = MLXArray([Float(0.3), -0.2, 0, 0, 0, 0, 0, 0])
        for step in 0 ..< 18 {
            let q = MLXRandom.normal([1, queryHeads, 64], key: MLXRandom.key(UInt64(200 + step))).asType(dtype)
            let k = MLXRandom.normal([1, 1, 64], key: MLXRandom.key(UInt64(300 + step))).asType(dtype)
            let v = MLXRandom.normal([1, 1, 64], key: MLXRandom.key(UInt64(400 + step))).asType(dtype)
            let oldTarget = oldRow.prepareDecodeWrite(), newTarget = newRow.prepareDecodeWrite()
            let oldInfo = oldRow.seqInfoRow(attending: oldRow.decodeAttendRange, writeTarget: oldTarget)
            let newInfo = newRow.seqInfoRow(attending: newRow.decodeAttendRange, writeTarget: newTarget)
            let (info, maxLength) = PagedAttentionKernel.seqinfo([oldInfo])
            let direct = try PagedAttentionKernel.decode(
                queries: q, newKeys: k, newValues: v, kSlab: oldGroup.kSlab, vSlab: oldGroup.vSlab,
                tables: MLXArray(oldRow.table, [1, oldRow.table.count]), seqinfo: info,
                maxAttendLength: maxLength, sinks: sinks, params: params, softcap: true,
                pageSize: 16, writeFence: oldGroup.writeFence, kernelSource: fixed.pool.kernelSource)
            oldGroup.writeFence = direct.nextWriteFence!
            let got = PagedSegmentAttention.decode(
                queries: q, newKeys: k, newValues: v, group: newGroup,
                rows: [.init(pages: newRow.table, info: newInfo)], sinks: sinks,
                params: params, softcap: true, source: segmented.pool.kernelSource)
            eval(direct.out, got)
            #expect(bytes(got) == bytes(direct.out))
            #expect(bytes(newRow.snapshot().keys) == bytes(oldRow.snapshot().keys))
        }
    }

}
