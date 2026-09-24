import Foundation
import MLX
import MLXRandom
import Testing
@testable import MLXLMCommon

@Suite("CBv2 segmented dispatch metadata cache", .serialized)
struct CBv2PagedSegmentDispatchCacheTests {
    private let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)

    private func backend(dtype: DType = .bfloat16, segmentPages: Int = 3,
                         kind: CBv2LayerKind? = nil) throws -> PagedKVBackend {
        let selected = kind ?? self.kind
        return try PagedKVBackend(layerKinds: [selected], config: .init(
            capacityBytes: 16 << 20, dtype: dtype, maxPrefillChunk: 512,
            nominalMaxSequenceLength: 8192, maxBufferLength: 16 << 20,
            segmentSizeBytes: 2 * 16 * selected.kvHeads * selected.headDim * dtype.size * segmentPages))
    }

    private func state(_ backend: PagedKVBackend, maxLength: Int = 320) throws
        -> (state: [CBv2SequenceKV?], row: PagedSequenceKV) {
        let state = try backend.makeSequenceState(layerKinds: backend.layerKinds,
                                                  promptLength: 0, maxLength: maxLength)
        return (state, try #require(state[0] as? PagedSequenceKV))
    }

    private func descriptor(_ row: PagedSequenceKV, write: (page: Int32, slot: Int)? = nil)
        -> PagedSegmentDispatchPlan.Row {
        .init(pages: row.table, info: row.seqInfoRow(attending: row.decodeAttendRange, writeTarget: write),
              identity: row.windowSize == nil ? .init(serial: row.serial, tableVersion: row.tableVersion) : nil)
    }

    private func assertFresh(_ actual: PagedSegmentPreparedDispatch,
                             rows: [PagedSegmentDispatchPlan.Row], group: PagedKVGroup,
                             ptok: Int = 64, hasWrite: Bool = true) {
        let fresh = PagedSegmentPreparedDispatch(rows: rows, group: group,
                                                 partitionTokens: ptok, hasWrite: hasWrite)
        #expect(actual.plan.partitionTokens == fresh.plan.partitionTokens)
        #expect(actual.plan.maxPartitions == fresh.plan.maxPartitions)
        #expect(actual.plan.buckets.count == fresh.plan.buckets.count)
        for (left, right) in zip(actual.plan.buckets, fresh.plan.buckets) {
            #expect(left.segmentIDs == right.segmentIDs)
            #expect(left.records == right.records)
        }
        for (left, right) in zip(actual.metadata, fresh.metadata) {
            #expect(left.records.asArray(Int32.self) == right.records.asArray(Int32.self))
            #expect(left.valueOffsets.asArray(Int64.self) == right.valueOffsets.asArray(Int64.self))
        }
    }

    @Test func growingLengthReusesRecordsAtEveryPageAndPartitionBoundary() throws {
        let backend = try backend(segmentPages: 2)
        let (state, row) = try state(backend)
        defer { backend.release(state) }
        let group = backend.pool.group(row.groupKey)
        let cache = PagedSegmentDispatchCache()
        #expect(cache.statistics == nil)
        cache.statistics = .init()
        var previous: PagedSegmentPreparedDispatch?
        for length in 1...273 {
            let rows = [descriptor(row, write: row.prepareDecodeWrite())]
            let plan = cache.prepare(rows: rows, group: group, partitionTokens: 64, hasWrite: true)
            if length > 1 { #expect((plan === previous) == (length % 16 != 1)) }
            assertFresh(plan, rows: rows, group: group)
            previous = plan
        }
        #expect(cache.statistics?.rebuilds == 18)
        #expect(cache.statistics?.hits == 255)
        #expect(cache.statistics?.bypasses == 0)
        #expect(cache.statistics!.keyNanoseconds > 0 && cache.statistics!.preparationNanoseconds > 0)
        #expect(previous!.plan.buckets.count > 1, "exercise more than 17 segment bindings")
    }

    @Test func qwenGeometryReusesExactMetadataWithoutRereadingEveryPage() throws {
        let qwen = CBv2LayerKind(attention: .full, headDim: 256, kvHeads: 2, queryHeads: 16)
        let backend = try backend(segmentPages: 257, kind: qwen)
        let (state, row) = try state(backend, maxLength: 5600)
        defer { backend.release(state) }
        for _ in 0..<5585 { _ = row.prepareDecodeWrite() }
        let group = backend.pool.group(row.groupKey)
        let cache = PagedSegmentDispatchCache()
        let firstRows = [descriptor(row)]
        let first = cache.prepare(rows: firstRows, group: group, partitionTokens: 256, hasWrite: false)
        #expect(first.plan.maxPartitions == 22)
        for _ in 0..<15 {
            _ = row.prepareDecodeWrite()
            let rows = [descriptor(row)]
            let next = cache.prepare(rows: rows, group: group, partitionTokens: 256, hasWrite: false)
            #expect(next === first)
            assertFresh(next, rows: rows, group: group, ptok: 256, hasWrite: false)
        }
    }

    @Test func batchOrderRowIdentityPartitionPolicyAndWriterAreKeys() throws {
        let backend = try backend()
        let (a, rowA) = try state(backend), (b, rowB) = try state(backend)
        defer { backend.release(a); backend.release(b) }
        for _ in 0..<81 { _ = rowA.prepareDecodeWrite() }
        for _ in 0..<33 { _ = rowB.prepareDecodeWrite() }
        let rows = [descriptor(rowA), descriptor(rowB)]
        let group = backend.pool.group(rowA.groupKey)
        let cache = PagedSegmentDispatchCache()
        let first = cache.prepare(rows: rows, group: group, partitionTokens: 64, hasWrite: false)
        assertFresh(first, rows: rows, group: group, hasWrite: false)
        let reordered = cache.prepare(rows: rows.reversed(), group: group, partitionTokens: 64, hasWrite: false)
        #expect(reordered !== first)
        assertFresh(reordered, rows: rows.reversed(), group: group, hasWrite: false)
        let changedPartition = cache.prepare(rows: rows, group: group, partitionTokens: 128, hasWrite: false)
        assertFresh(changedPartition, rows: rows, group: group, ptok: 128, hasWrite: false)
        #expect(changedPartition !== reordered)
        let written = [descriptor(rowA, write: rowA.prepareDecodeWrite()),
                       descriptor(rowB, write: rowB.prepareDecodeWrite())]
        let changedWriter = cache.prepare(rows: written, group: group, partitionTokens: 128, hasWrite: true)
        #expect(changedWriter !== changedPartition)
        assertFresh(changedWriter, rows: written, group: group, ptok: 128)
    }

    @Test func rollbackAndSwiftArrayCopyOnWritePreserveInflightMetadata() throws {
        let backend = try backend()
        let (state, row) = try state(backend)
        defer { backend.release(state) }
        for _ in 0..<31 { _ = row.prepareDecodeWrite() }
        let group = backend.pool.group(row.groupKey)
        let cache = PagedSegmentDispatchCache()
        let oldRows = [descriptor(row, write: row.prepareDecodeWrite())]
        let old = cache.prepare(rows: oldRows, group: group, partitionTokens: 64, hasWrite: true)
        let oldRecords = old.plan.buckets.map(\.records)
        let rows = [descriptor(row, write: row.prepareDecodeWrite())]
        let appended = cache.prepare(rows: rows, group: group, partitionTokens: 64, hasWrite: true)
        #expect(appended !== old && oldRows[0].pages.count == 2 && row.table.count == 3)
        row.rollback(2)
        let rollbackRows = [descriptor(row, write: row.prepareDecodeWrite())]
        let rolledBack = cache.prepare(rows: rollbackRows, group: group, partitionTokens: 64, hasWrite: true)
        #expect(rolledBack !== appended)
        assertFresh(rolledBack, rows: rollbackRows, group: group)
        #expect(old.plan.buckets.map(\.records) == oldRecords)
        assertFresh(old, rows: oldRows, group: group)
    }

    @Test func directAndWindowedCallersBypassAndDropTheLastPlan() throws {
        let backend = try backend()
        let (state, row) = try state(backend)
        defer { backend.release(state) }
        for _ in 0..<33 { _ = row.prepareDecodeWrite() }
        let group = backend.pool.group(row.groupKey)
        let cache = PagedSegmentDispatchCache()
        cache.statistics = .init()
        let full = descriptor(row)
        _ = cache.prepare(rows: [full], group: group, partitionTokens: 64, hasWrite: false)
        let direct = PagedSegmentDispatchPlan.Row(pages: full.pages, info: full.info)
        let window = PagedSegmentDispatchPlan.Row(pages: full.pages, info: .init(
            attendStart: 1, attendLength: 32, tableLength: full.pages.count), identity: full.identity)
        for unsupported in [direct, window] {
            let result = cache.prepare(rows: [unsupported], group: group, partitionTokens: 64, hasWrite: false)
            #expect(cache.prepared == nil)
            assertFresh(result, rows: [unsupported], group: group, hasWrite: false)
        }
        #expect(cache.statistics?.bypasses == 2)
    }

    @Test func growthInvalidatesButMetadataDoesNotKeepRetiredStorageAlive() throws {
        let backend = try backend()
        var first = try state(backend, maxLength: 32)
        for _ in 0..<17 { _ = first.row.prepareDecodeWrite() }
        let group = backend.pool.group(first.row.groupKey)
        weak var storage = group.segments.values.first!.storage
        let cache = PagedSegmentDispatchCache()
        let rows = [descriptor(first.row)]
        let before = cache.prepare(rows: rows, group: group, partitionTokens: 64, hasWrite: false)
        let (larger, _) = try state(backend, maxLength: 320)
        let grown = cache.prepare(rows: rows, group: group, partitionTokens: 64, hasWrite: false)
        #expect(grown !== before)
        assertFresh(grown, rows: rows, group: group, hasWrite: false)
        backend.release(larger)
        let oldSerial = first.row.serial
        let oldPages = first.row.table
        backend.release(first.state)
        #expect(storage == nil, "metadata must not retain an unused segment allocation")
        #expect(backend.bytesWired == 0)
        first = try state(backend, maxLength: 32)
        defer { backend.release(first.state) }
        for _ in 0..<17 { _ = first.row.prepareDecodeWrite() }
        #expect(first.row.serial != oldSerial && first.row.table == oldPages)
        let nextRows = [descriptor(first.row)]
        let reused = cache.prepare(rows: nextRows, group: group, partitionTokens: 64, hasWrite: false)
        #expect(reused !== grown)
        assertFresh(reused, rows: nextRows, group: group, hasWrite: false)
    }

    @Test func identicalSetRowsKeepsPlanAndMembershipChangesReleaseIt() throws {
        let backend = try backend()
        let (a, rowA) = try state(backend), (b, rowB) = try state(backend)
        defer { backend.release(a); backend.release(b) }
        _ = rowA.prepareDecodeWrite(); _ = rowB.prepareDecodeWrite()
        let layer = try #require(backend.makeLayerCaches()[0] as? PagedLayerCache)
        layer.setRows([rowA, rowB])
        let cache = layer.segmentDispatchCache
        let first = cache.prepare(rows: [descriptor(rowA), descriptor(rowB)],
                                  group: backend.pool.group(rowA.groupKey), partitionTokens: 64, hasWrite: false)
        layer.setRows([rowA, rowB])
        #expect(cache.prepared === first)
        layer.setRows([rowB, rowA])
        #expect(cache.prepared == nil)
        _ = cache.prepare(rows: [descriptor(rowB)], group: backend.pool.group(rowA.groupKey),
                          partitionTokens: 64, hasWrite: false)
        layer.setRows([])
        #expect(cache.prepared == nil)
    }
}
