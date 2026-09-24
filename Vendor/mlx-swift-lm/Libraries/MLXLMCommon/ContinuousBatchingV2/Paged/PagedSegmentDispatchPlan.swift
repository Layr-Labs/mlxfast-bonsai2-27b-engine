import Foundation

/// Host-only partition mapping. Every work item keeps one complete token
/// partition, so physical placement cannot change the softmax reduction order.
struct PagedSegmentDispatchPlan {
    static let maximumBindings = 17
    static let recordHeader = 8
    static let recordStride = recordHeader + 2 * maximumBindings
    static let bindingClasses = [1, 4, 8, 17]

    struct Row {
        /// Only layer-bound full-attention rows supply an identity. Direct
        /// kernel probes and windowed rows keep using fresh dispatch plans.
        struct Identity: Equatable {
            let serial: UInt64
            let tableVersion: Int
        }
        let pages: [Int32]
        let info: PagedAttentionKernel.SeqInfoRow
        let identity: Identity?

        init(pages: [Int32], info: PagedAttentionKernel.SeqInfoRow,
             identity: Identity? = nil) {
            self.pages = pages
            self.info = info
            self.identity = identity
        }
    }

    struct Bucket {
        let segmentIDs: [Int]
        let records: [Int32]
        var workCount: Int { records.count / PagedSegmentDispatchPlan.recordStride }
        var bindingClass: Int {
            PagedSegmentDispatchPlan.bindingClasses.first { $0 >= segmentIDs.count }!
        }
    }

    let buckets: [Bucket]
    let partitionTokens: Int
    let maxPartitions: Int

    /// PTOK + an unaligned first page can span at most 17 physical pages.
    /// Smaller page configurations get an explicit smaller PTOK specialization.
    static func boundedPartitionTokens(_ proposed: Int, pageSize: Int) -> Int {
        precondition(pageSize > 0 && proposed > 0 && proposed % pageSize == 0)
        return min(proposed / pageSize, maximumBindings - 1) * pageSize
    }

    init(rows: [Row], layout: PagedKVSegmentLayout, pageSize: Int,
         partitionTokens: Int, hasWrite: Bool) {
        precondition(!rows.isEmpty)
        precondition(partitionTokens > 0 && partitionTokens % pageSize == 0)
        precondition(partitionTokens / pageSize + 1 <= Self.maximumBindings)
        self.partitionTokens = partitionTokens
        let maxPartitions = rows.map {
            $0.info.attendLength / partitionTokens + ($0.info.attendLength % partitionTokens == 0 ? 0 : 1)
        }.max()!
        self.maxPartitions = maxPartitions

        struct Item {
            let row: Int
            let partition: Int
            let firstPage: Int
            let pages: [Int32]
            let write: Int32?
        }
        var result: [Bucket] = []
        var pending: [Item] = []
        var segments = Set<Int>()
        func flush() {
            guard !pending.isEmpty else { return }
            let ids = segments.sorted()
            let binding = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
            var records: [Int32] = []
            records.reserveCapacity(pending.count * Self.recordStride)
            for item in pending {
                let writer = item.write.map { (binding[layout.segmentIndex(page: $0)]!, layout.localPage($0)) }
                var record: [Int32] = [
                    Int32(item.row), Int32(item.partition), Int32(item.firstPage),
                    Int32(item.pages.count), Int32(writer?.0 ?? 0), Int32(writer?.1 ?? 0),
                    Int32(maxPartitions), 0]
                for page in item.pages {
                    record.append(Int32(binding[layout.segmentIndex(page: page)]!))
                    record.append(Int32(layout.localPage(page)))
                }
                record.append(contentsOf: repeatElement(0, count: Self.recordStride - record.count))
                records.append(contentsOf: record)
            }
            result.append(Bucket(segmentIDs: ids, records: records))
            pending.removeAll(keepingCapacity: true)
            segments.removeAll(keepingCapacity: true)
        }
        for (rowIndex, row) in rows.enumerated() {
            let info = row.info
            precondition(info.attendStart >= 0 && info.attendLength > 0 && info.tableLength > 0)
            let partitions = info.attendLength / partitionTokens + (info.attendLength % partitionTokens == 0 ? 0 : 1)
            for partition in 0 ..< partitions {
                let start = info.attendStart + partition * partitionTokens
                let end = info.attendStart + min(info.attendLength, (partition + 1) * partitionTokens)
                let first = start / pageSize
                let last = (end - 1) / pageSize
                let pages = (first ... last).map { logical -> Int32 in
                    let index = logical % info.tableLength
                    precondition(index < row.pages.count)
                    let page = row.pages[index]
                    precondition(layout.isUsable(page), "work item names a poison or invalid page")
                    return page
                }
                let writer: Int32? = hasWrite && partition == partitions - 1 ? info.writePage : nil
                if let writer {
                    precondition(writer == pages.last, "fused write must target the newest attended page")
                }
                let needed = Set(pages.map { layout.segmentIndex(page: $0) })
                precondition(needed.count <= Self.maximumBindings)
                if segments.union(needed).count > Self.maximumBindings { flush() }
                segments.formUnion(needed)
                pending.append(Item(row: rowIndex, partition: partition, firstPage: first,
                                    pages: pages, write: writer))
            }
        }
        flush()
        self.buckets = result
    }
}
