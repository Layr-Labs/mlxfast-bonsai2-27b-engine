import Cmlx
import Foundation
import MLX

/// Stable address ranges, independent of a pool's current byte grant. Retired
/// ranges are reused by size class; live pages never move when the grant grows.
struct PagedKVSegmentLayout: Sendable {
    static let defaultTargetBytes = 64 << 20

    let pageBytes: Int
    let maximumUsablePages: Int
    let maximumAddressPages: Int
    private(set) var ranges: [Range<Int>] = []
    private var segmentForPage: [Int32] = []

    init(pageBytes: Int, targetBytes: Int, maximumBufferBytes: Int,
         maximumAddressPages: Int = Int(Int32.max)) throws {
        guard pageBytes > 0, targetBytes > 0, maximumBufferBytes > 0,
              maximumAddressPages >= 2, maximumAddressPages <= Int(Int32.max) else {
            throw CBv2KVError.backendIneligible(reason: "invalid paged segment geometry")
        }
        let physical = min(targetBytes, maximumBufferBytes) / pageBytes
        guard physical >= 2 else {
            throw CBv2KVError.backendIneligible(
                reason: "paged segment must fit a poison page and a usable page")
        }
        self.pageBytes = pageBytes
        self.maximumAddressPages = maximumAddressPages
        self.maximumUsablePages = Self.sizeClass(atMost: physical - 1)
    }

    /// Eager geometry used by diagnostics. Runtime pools start empty instead.
    init(pageCount: Int, pageBytes: Int, targetBytes: Int, maximumBufferBytes: Int) throws {
        try self.init(pageBytes: pageBytes, targetBytes: targetBytes,
                      maximumBufferBytes: maximumBufferBytes)
        guard pageCount >= 2 else {
            throw CBv2KVError.backendIneligible(reason: "invalid paged segment page count")
        }
        var remaining = pageCount
        while remaining >= 2 {
            let usable = Self.sizeClass(atMost: min(maximumUsablePages, remaining - 1))
            try appendRange(usablePages: usable)
            remaining -= usable + 1
        }
    }

    var pageCount: Int { segmentForPage.count }
    var segmentCount: Int { ranges.count }
    var usablePageCount: Int { pageCount - segmentCount }
    var physicalBytes: Int { pageCount * pageBytes }

    /// Exact eager capacity arithmetic without building a grant-sized map.
    func usablePages(fittingPhysicalPages pages: Int) -> Int {
        guard pages >= 2 else { return 0 }
        let whole = pages / (maximumUsablePages + 1)
        var usable = whole * maximumUsablePages
        var remainder = pages % (maximumUsablePages + 1)
        while remainder >= 2 {
            let count = Self.sizeClass(atMost: remainder - 1)
            usable += count
            remainder -= count + 1
        }
        return usable
    }

    /// Exact bytes for a new batch of size-class segments, without building a
    /// page-address map. Each full class and each binary remainder class owns
    /// one poison page. Used by submit/deadline probes before any allocation.
    func physicalBytes(addingUsablePages usable: Int) -> Int? {
        guard usable >= 0 else { return nil }
        let segments = usable / maximumUsablePages
            + (usable % maximumUsablePages).nonzeroBitCount
        let (pages, pageOverflow) = usable.addingReportingOverflow(segments)
        let (bytes, byteOverflow) = pages.multipliedReportingOverflow(by: pageBytes)
        guard !pageOverflow, !byteOverflow, pages <= maximumAddressPages else { return nil }
        return bytes
    }

    /// One bound per allocator buffer, including poison and size-class padding.
    /// This prices complete classes arithmetically without a page-address map.
    func allocationBytes(addingUsablePages usable: Int) -> Int? {
        guard usable >= 0, physicalBytes(addingUsablePages: usable) != nil else { return nil }
        do {
            let whole = usable / maximumUsablePages
            var total = 0
            if whole > 0 {
                let bytes = try allocationBytes(usablePages: maximumUsablePages)
                let (product, overflow) = whole.multipliedReportingOverflow(by: bytes)
                guard !overflow else { return nil }
                total = product
            }
            var remaining = usable % maximumUsablePages
            while remaining > 0 {
                let count = Self.sizeClass(atMost: remaining)
                let (next, overflow) = total.addingReportingOverflow(try allocationBytes(usablePages: count))
                guard !overflow else { return nil }
                total = next
                remaining -= count
            }
            return total
        } catch { return nil }
    }

    func allocationBytes(forSegment index: Int) throws -> Int {
        try allocationBytes(usablePages: ranges[index].count - 1)
    }

    private func allocationBytes(usablePages: Int) throws -> Int {
        let (bytes, overflow) = (usablePages + 1).multipliedReportingOverflow(by: pageBytes)
        guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        return try Memory.allocationFootprintUpperBound(byteCount: bytes)
    }

    func range(_ index: Int) -> Range<Int> { ranges[index] }
    func segmentIndex(page: Int32) -> Int {
        precondition(page >= 0 && Int(page) < pageCount)
        return Int(segmentForPage[Int(page)])
    }
    func localPage(_ page: Int32) -> Int {
        Int(page) - ranges[segmentIndex(page: page)].lowerBound
    }
    func isUsable(_ page: Int32) -> Bool {
        page >= 0 && Int(page) < pageCount && localPage(page) != 0
    }

    /// Prepare exact additional usable demand without mutating the live map.
    /// Reusing the same size class bounds historical address growth by peak
    /// concurrent demand per class, rather than the number of requests served.
    func adding(usablePages: Int, excluding committed: Set<Int>) throws
        -> (layout: PagedKVSegmentLayout, segmentIDs: [Int])
    {
        precondition(usablePages >= 0)
        var result = self
        var reusable: [Int: [Int]] = [:]
        for index in ranges.indices.reversed() where !committed.contains(index) {
            reusable[ranges[index].count - 1, default: []].append(index)
        }
        var needed = usablePages
        var selected: [Int] = []
        while needed > 0 {
            let usable = Self.sizeClass(atMost: min(maximumUsablePages, needed))
            let index: Int
            if let available = reusable[usable]?.popLast() {
                index = available
            } else {
                index = result.segmentCount
                try result.appendRange(usablePages: usable)
            }
            selected.append(index)
            needed -= usable
        }
        return (result, selected)
    }

    private mutating func appendRange(usablePages: Int) throws {
        let (count, overflow) = pageCount.addingReportingOverflow(usablePages + 1)
        let (_, byteOverflow) = count.multipliedReportingOverflow(by: pageBytes)
        guard !overflow, !byteOverflow, count <= maximumAddressPages else {
            throw CBv2KVError.backendIneligible(reason: "paged range exceeds checked page-ID limit")
        }
        let start = pageCount
        let index = Int32(ranges.count)
        segmentForPage.append(contentsOf: repeatElement(index, count: usablePages + 1))
        ranges.append(start ..< count)
    }

    private static func sizeClass(atMost value: Int) -> Int {
        precondition(value > 0)
        return 1 << (Int.bitWidth - 1 - value.leadingZeroBitCount)
    }
}

/// One stable, evaluated native buffer. The first local page in both regions
/// is poison. Kernels receive this array as an input and order in-place writes
/// with the owning group's fence; they never create a second writable alias.
/// Exclusive staging or the engine queue owns cover/retirement mutations.
/// Rebased segment wrappers share this evaluated allocation and one credit.
/// Invalidating coverage never drops the array or refunds its native charge.
final class PagedKVSegmentBacking {
    let array: MLXArray
    let allocatedBytes: Int
    private var coverage: CBv2MemoryCoverage?
    private var admissionIdentity: ObjectIdentifier?

    init(_ array: MLXArray, allocationBound: Int) throws {
        guard let info = try array.evaluatedBufferInfo(), info.isUnique,
            info.dataOffset == 0, info.dataElements == array.size, info.isRowContiguous,
            info.allocatedBytes >= array.nbytes, info.allocatedBytes <= allocationBound
        else { throw CBv2CompleteCheckpointError.allocationFailed }
        self.array = array
        self.allocatedBytes = info.allocatedBytes
    }

    func belongs(to admission: AdmissionV2) -> Bool {
        admissionIdentity == nil || admissionIdentity == ObjectIdentifier(admission)
    }

    func cover(using admission: AdmissionV2, bytes: Int) throws {
        guard admission.hasProcessMemoryOwner else { return }
        guard belongs(to: admission) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        if coverage != nil { return }
        guard array.nbytes == bytes else { throw CBv2CompleteCheckpointError.allocationFailed }
        coverage = try admission.coverEvaluatedAllocation(bytes: allocatedBytes)
        admissionIdentity = ObjectIdentifier(admission)
    }

    func invalidateCoverage() {
        coverage?.invalidate()
        coverage = nil
    }

    deinit { coverage?.invalidate() }
}

final class PagedKVSegment {
    let index: Int
    let pages: Range<Int>
    let backing: PagedKVSegmentBacking
    var storage: MLXArray { backing.array }
    let valueOffset: Int
    let byteCount: Int
    var allocatedBytes: Int { backing.allocatedBytes }

    /// Ownership metadata changes; native storage and byte offsets do not.
    /// Both ranges use the same size class, so this creates no KV copy.
    init(rebasing source: PagedKVSegment, index: Int, layout: PagedKVSegmentLayout) {
        precondition(source.pages.count == layout.range(index).count)
        self.index = index
        self.pages = layout.range(index)
        self.backing = source.backing
        self.valueOffset = source.valueOffset
        self.byteCount = source.byteCount
    }

    init(index: Int, layout: PagedKVSegmentLayout, key: PagedKVGroupKey,
         pageSize: Int, dtype: DType, evaluate: (MLXArray) throws -> Void,
         admission: AdmissionV2? = nil) throws {
        self.index = index
        let pages = layout.range(index)
        self.pages = pages
        self.valueOffset = pages.count * key.kvHeads * pageSize * key.headDim
        self.byteCount = pages.count * layout.pageBytes
        let allocationStream = StreamOrDevice.default
        let storage = try withError { fault in
            let array = MLXArray.zeros(
                [2, pages.count, key.kvHeads, pageSize, key.headDim], dtype: dtype,
                stream: allocationStream)
            do {
                try fault.check()
                try evaluate(array)
            } catch {
                allocationStream.stream.synchronize()
                if error is MLXError { throw error }
                try fault.check()
                throw error
            }
            // Evaluation signals before Metal completion handlers drop temporary references.
            allocationStream.stream.synchronize()
            try fault.check()
            return array
        }
        self.backing = try PagedKVSegmentBacking(
            storage, allocationBound: layout.allocationBytes(forSegment: index))
        if let admission { try backing.cover(using: admission, bytes: byteCount) }
    }
}
