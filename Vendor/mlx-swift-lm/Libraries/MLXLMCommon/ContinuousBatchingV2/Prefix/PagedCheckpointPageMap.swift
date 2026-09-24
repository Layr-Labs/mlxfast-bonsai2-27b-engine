#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import MLX

/// One immutable host table shared by a row's K and V export sources. Each
/// record retains its segment directly; there is no persistent dictionary or
/// duplicate K/V map. Anonymous VM gives an explicit page-rounded byte bound,
/// independent of Swift collection capacity or malloc size classes.
final class CBv2PagedCheckpointPageMap {
    struct Record {
        let segment: PagedKVSegment
        let localPage: Int
    }

    let key: PagedKVGroupKey
    let pageSize: Int
    let position: Int
    let previous: MLXArray
    let byteCapacity: Int
    private let records: UnsafeMutablePointer<Record>
    private let allocation: UnsafeMutableRawPointer
    private let count: Int
    private let reservation: CBv2CheckpointReservation
    private let readinessLock = NSLock()
    private var readable = false

    static func allocationBytes(pageCount: Int) throws -> Int {
        let (bytes, overflow) = pageCount.multipliedReportingOverflow(by: MemoryLayout<Record>.stride)
        let page = Int(getpagesize())
        let (padded, paddingOverflow) = bytes.addingReportingOverflow(page - 1)
        guard pageCount > 0, page > 0, !overflow, !paddingOverflow else {
            throw CBv2CompleteCheckpointError.invalidManifest
        }
        return padded / page * page
    }

    convenience init(row: PagedSequenceKV, position: Int, admission: AdmissionV2) throws {
        guard row.windowSize == nil, row.baseOffset == 0, row.absoluteOffset >= position,
              row.pool.config.segmentSizeBytes != nil, row.pool.config.layerDTypes != nil,
              let layout = row.pool.group(row.groupKey).segmentLayout
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let group = row.pool.group(row.groupKey)
        try self.init(key: row.groupKey, pageSize: group.pageSize, position: position,
                      table: row.table, layout: layout, segments: group.segments,
                      previous: group.writeFence, admission: admission)
    }

    init<Table: Collection>(
        key: PagedKVGroupKey, pageSize: Int, position: Int, table: Table,
        layout: PagedKVSegmentLayout, segments: [Int: PagedKVSegment], previous: MLXArray,
        admission: AdmissionV2, beforeAllocation: () throws -> Void = {}
    ) throws where Table.Element == Int32 {
        guard position > 1, pageSize > 0, key.headDim >= 64, key.kvHeads > 0,
              [.float16, .bfloat16, .float32].contains(key.dtype)
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let count = (position - 1) / pageSize + 1
        guard table.count >= count else { throw CBv2CompleteCheckpointError.incompleteTransfer }
        let bytes = try Self.allocationBytes(pageCount: count)
        let reservation = try admission.reserveTransient(bytes: bytes)
        defer { withExtendedLifetime(reservation) {} }
        // Deterministic refusal/fault seam: no mapped storage or retained page
        // records can be built until the real admission transaction succeeds.
        try beforeAllocation()
        guard let allocation = mmap(nil, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0),
              allocation != MAP_FAILED else { throw CBv2CompleteCheckpointError.allocationFailed }
        let records = allocation.bindMemory(to: Record.self, capacity: count)
        var initialized = 0
        do {
            for page in table.prefix(count) {
                guard layout.isUsable(page), let segment = segments[layout.segmentIndex(page: page)],
                      segment.storage.size <= Int(UInt32.max), segment.storage.dtype == key.dtype
                else { throw CBv2CompleteCheckpointError.incompleteTransfer }
                records.advanced(by: initialized).initialize(to: .init(segment: segment, localPage: layout.localPage(page)))
                initialized += 1
            }
        } catch {
            records.deinitialize(count: initialized)
            precondition(munmap(allocation, bytes) == 0)
            throw error
        }
        self.key = key
        self.pageSize = pageSize
        self.position = position
        self.previous = previous
        self.byteCapacity = bytes
        self.records = records
        self.allocation = allocation
        self.count = count
        self.reservation = reservation
    }

    subscript(index: Int) -> Record {
        precondition(index >= 0 && index < count)
        return records[index]
    }

    /// K/V share one completion wait. The captured fence includes every write
    /// to these immutable prefix spans; it is never replaced by later writes.
    /// No Admission, process, or grant lock is held while waiting for the GPU.
    func prepareForReading(
        evaluate: (MLXArray) throws -> Void = { array in try withError { array.eval() } }
    ) throws {
        try readinessLock.withLock {
            guard !readable else { return }
            try evaluate(previous)
            readable = true
        }
    }

    deinit {
        records.deinitialize(count: count)
        precondition(munmap(allocation, byteCapacity) == 0)
        // Explicitly release only after the final K/V source dropped the map
        // and every record/host mapping has retired.
        reservation.release()
    }
}
