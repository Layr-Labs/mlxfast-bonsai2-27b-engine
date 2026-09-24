import Foundation
import MLX

/// Immutable host/device metadata only. In-flight graphs may retain these
/// arrays across a cache replacement; none refer to KV storage or a fence.
final class PagedSegmentPreparedDispatch {
    struct Metadata {
        let records: MLXArray
        let valueOffsets: MLXArray
    }
    let plan: PagedSegmentDispatchPlan
    let metadata: [Metadata]

    init(rows: [PagedSegmentDispatchPlan.Row], group: PagedKVGroup,
         partitionTokens: Int, hasWrite: Bool) {
        plan = PagedSegmentDispatchPlan(
            rows: rows, layout: group.segmentLayout!, pageSize: group.pageSize,
            partitionTokens: partitionTokens, hasWrite: hasWrite)
        metadata = plan.buckets.map { bucket in
            var offsets = bucket.segmentIDs.map { Int64(group.segments[$0]!.valueOffset) }
            offsets.append(contentsOf: repeatElement(
                offsets[0], count: max(8, bucket.bindingClass) - offsets.count))
            return Metadata(records: MLXArray(bucket.records), valueOffsets: MLXArray(offsets))
        }
    }
}

/// A layer owns one last plan, scoped to its pool. Growing token lengths and
/// write slots stay in freshly bound seqinfo, not in this cache. A full row's
/// records depend only on its addressed pages, partitions and final writer.
final class PagedSegmentDispatchCache {
    struct Statistics {
        var hits = 0
        var rebuilds = 0
        var bypasses = 0
        var keyNanoseconds: UInt64 = 0
        var preparationNanoseconds: UInt64 = 0
    }

    /// Opt-in host measurements for tests/profilers; normal decode reads no clock.
    var statistics: Statistics?
    private var key: Key?
    private(set) var prepared: PagedSegmentPreparedDispatch?

    private struct Key: Equatable {
        struct Row: Equatable {
            let identity: PagedSegmentDispatchPlan.Row.Identity
            let tableLength: Int
            let pageCount: Int
            let partitions: Int
            let writePage: Int32?
        }
        let rows: [Row]
        let pageSize: Int
        let partitionTokens: Int
        let hasWrite: Bool
        // Exact value identity of address mapping and segment value offsets.
        // Ranges append but never rebase. Retiring/recreating backing of the
        // same range needs no rebuild: current storage is fetched at dispatch.
        let layoutRanges: [Range<Int>]
        let groupKey: PagedKVGroupKey
    }

    func clear() {
        key = nil
        prepared = nil
    }

    func prepare(rows: [PagedSegmentDispatchPlan.Row], group: PagedKVGroup,
                 partitionTokens: Int, hasWrite: Bool) -> PagedSegmentPreparedDispatch {
        let start = statistics == nil ? nil : DispatchTime.now().uptimeNanoseconds
        let candidate = makeKey(rows: rows, group: group,
                                partitionTokens: partitionTokens, hasWrite: hasWrite)
        let hit = candidate != nil && candidate == key
        if let start { statistics?.keyNanoseconds += DispatchTime.now().uptimeNanoseconds - start }
        if hit, let prepared {
            statistics?.hits += 1
            return prepared
        }
        let prepareStart = statistics == nil ? nil : DispatchTime.now().uptimeNanoseconds
        let result = PagedSegmentPreparedDispatch(
            rows: rows, group: group, partitionTokens: partitionTokens, hasWrite: hasWrite)
        if let prepareStart {
            statistics?.preparationNanoseconds += DispatchTime.now().uptimeNanoseconds - prepareStart
        }
        if let candidate {
            key = candidate
            prepared = result
            statistics?.rebuilds += 1
        } else {
            clear()
            statistics?.bypasses += 1
        }
        return result
    }

    private func makeKey(rows: [PagedSegmentDispatchPlan.Row], group: PagedKVGroup,
                         partitionTokens: Int, hasWrite: Bool) -> Key? {
        guard !rows.isEmpty, let layout = group.segmentLayout else { return nil }
        var identities: [Key.Row] = []
        identities.reserveCapacity(rows.count)
        for row in rows {
            let info = row.info
            guard let identity = row.identity, info.attendStart == 0, info.attendLength > 0,
                  info.tableLength == row.pages.count else { return nil }
            let pageCount = (info.attendLength - 1) / group.pageSize + 1
            guard pageCount <= info.tableLength else { return nil }
            identities.append(.init(
                identity: identity, tableLength: info.tableLength, pageCount: pageCount,
                partitions: (info.attendLength - 1) / partitionTokens + 1,
                writePage: hasWrite ? info.writePage : nil))
        }
        return Key(rows: identities, pageSize: group.pageSize,
                   partitionTokens: partitionTokens, hasWrite: hasWrite,
                   layoutRanges: layout.ranges, groupKey: group.key)
    }
}
