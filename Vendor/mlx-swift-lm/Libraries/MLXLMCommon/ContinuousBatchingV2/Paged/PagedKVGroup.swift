import Foundation
import MLX

/// Page ownership for one attention geometry; storage may be fixed or segmented.
final class PagedKVGroup {
    let key: PagedKVGroupKey
    let pageSize: Int
    let dtype: DType
    let writeValidation: CBv2PagedKVWriteValidation
    private let fixedPageCount: Int
    var pageCount: Int { segmentLayout?.pageCount ?? fixedPageCount }
    /// `[pageCount, kvHeads, pageSize, headDim]`. STABLE arrays: written
    /// in place by the write kernels, never replaced (see file header).
    private let fixedK: MLXArray?
    private let fixedV: MLXArray?
    var kSlab: MLXArray { precondition(segmentLayout == nil); return fixedK! }
    var vSlab: MLXArray { precondition(segmentLayout == nil); return fixedV! }
    private(set) var segmentLayout: PagedKVSegmentLayout?
    private(set) var segments: [Int: PagedKVSegment] = [:]
    var committedUsablePages: Int {
        guard segmentLayout != nil else { return usablePageCount }
        return segments.values.reduce(0) { $0 + $1.pages.count - 1 }
    }
    var committedSegmentBytes: Int { segments.values.reduce(0) { $0 + $1.allocatedBytes } }
    var committedLogicalBytes: Int { segments.values.reduce(0) { $0 + $1.byteCount } }
    /// Latest fence of the group's bulk-write chain (`[1]` int32). Gathers
    /// consume it so page reads order after every prior bulk write.
    var writeFence: MLXArray
    /// Intrusive free queue. The computed array preserves the historical
    /// test/diagnostic surface without making arbitrary removal O(n).
    private var freeQueue: PagedBlockFreeQueue
    var freeList: [Int32] { freeQueue.elements }
    /// Per-page active-owner count. A cached page may have count zero (in the
    /// free queue) or positive (shared by one or more request tables).
    var refCounts: [Int]
    /// Incremented immediately before a zero-ref page is reused for a write.
    /// Prefix metadata carries this generation to close page-id ABA races.
    var generations: [UInt64]
    /// Pages currently held by sequences (refCount > 0).
    private(set) var pagesInUse: Int = 0
    /// Pages promised to admitted sequences (lazily materialized).
    var pagesReserved: Int = 0
    /// Pages queued for release by an in-flight speculative transaction
    /// (WS-3.2c). They keep `refCount > 0` until `drainDeferredFrees()`, so
    /// they cannot be recycled to another row while a round's captures
    /// still name them.
    var deferredFrees: [Int32] = []
    /// Which of this group's slabs `materializeSlabs` has ACTUALLY made
    /// resident. Tracked explicitly — set only after the slab's blocking
    /// eval returned — because MLX exposes no public "is this array
    /// evaluated" API (mlx-c's `_mlx_array_is_available` is documented
    /// internal and mlx-swift does not surface it). A retry after a
    /// partial commit re-attempts only the slabs still unset here, and an
    /// already-materialized pool commits for free (nothing left to eval).
    var kSlabMaterialized = false
    var vSlabMaterialized = false

    /// Bytes of ONE slab (K or V alone), poison page included — the unit
    /// `materializeSlabs` allocates and tracks.
    var slabBytes: Int {
        pageCount * key.kvHeads * pageSize * key.headDim * dtype.size
    }

    /// Local page zero is inert and pinned in every backing buffer. Fixed
    /// slabs use global page zero; segmented kernels translate invalid/padded
    /// references to the first local page of an already-bound segment.
    static let poisonPage: Int32 = 0
    var poisonPage: Int32 { Self.poisonPage }

    /// Tenant capacity excludes each backing buffer's poison page.
    var usablePageCount: Int { segmentLayout == nil ? pageCount - 1 : committedUsablePages }

    /// Bytes of ONE page counting both K and V slabs.
    var pageBytes: Int {
        2 * key.kvHeads * pageSize * key.headDim * dtype.size
    }

    init(key: PagedKVGroupKey, pageCount: Int, pageSize: Int, dtype: DType,
         segmentLayout: PagedKVSegmentLayout? = nil,
         writeValidation: CBv2PagedKVWriteValidation = CBv2PagedKVWriteValidation()) {
        self.writeValidation = writeValidation
        precondition(
            segmentLayout != nil || pageCount >= 2,
            "[PagedKVPool] group \(key) needs at least one usable page beyond the poison page")
        self.key = key
        self.pageSize = pageSize
        self.dtype = dtype
        self.fixedPageCount = pageCount
        let shape = [pageCount, key.kvHeads, pageSize, key.headDim]
        self.segmentLayout = segmentLayout
        self.fixedK = segmentLayout == nil ? MLXArray.zeros(shape, dtype: dtype) : nil
        self.fixedV = segmentLayout == nil ? MLXArray.zeros(shape, dtype: dtype) : nil
        self.writeFence = MLXArray.zeros([1], dtype: .int32)
        // Initial FIFO order is ascending ids, so fresh allocations remain
        // physically consecutive (enables run-coalesced writes). The poison
        // page is excluded permanently.
        self.freeQueue = PagedBlockFreeQueue(
            pageCount: pageCount, excluding: Self.poisonPage,
            initiallyEmpty: segmentLayout != nil)
        // The slabs are zero-initialised and no write can ever address the
        // poison page (every slot comes from an allocated page id), so
        // pinning its refcount at 1 is the whole of "permanently zeroed":
        // it can never be allocated, retained, freed or written.
        var counts = [Int](repeating: 0, count: pageCount)
        if let segmentLayout {
            for index in 0 ..< segmentLayout.segmentCount {
                counts[segmentLayout.range(index).lowerBound] = 1
            }
        } else {
            counts[Int(Self.poisonPage)] = 1
        }
        self.refCounts = counts
        self.generations = [UInt64](repeating: 0, count: pageCount)
    }

    /// True when `page` is a page a sequence row can own. The poison page
    /// and out-of-range ids are not.
    func isAllocatable(_ page: Int32) -> Bool {
        segmentLayout?.isUsable(page) ?? (page != Self.poisonPage && page >= 0 && Int(page) < pageCount)
    }

    func currentHandle(_ page: Int32) -> PagedKVPageHandle {
        precondition(isAllocatable(page))
        return PagedKVPageHandle(
            group: key, page: page, generation: generations[Int(page)])
    }

    func isValid(_ handle: PagedKVPageHandle) -> Bool {
        handle.group == key && isAllocatable(handle.page)
            && (segmentLayout.map { segments[$0.segmentIndex(page: handle.page)] != nil } ?? true)
            && generations[Int(handle.page)] == handle.generation
    }

    func allocatePage(willReuse: (PagedKVPageHandle) -> Void) -> Int32 {
        precondition(
            freeQueue.count > 0,
            "[PagedKVPool] free list underflow for group \(key) — reservation accounting bug")
        let page = freeQueue.first
        precondition(
            isAllocatable(page),
            "[PagedKVPool] poison page escaped the free list for group \(key)")
        precondition(refCounts[Int(page)] == 0)
        // The prefix index is non-owning. Invalidate every alias while the
        // page still names its old generation, before it can be returned for
        // a write. The observer is internal/nonthrowing; it may only reorder
        // generation-valid zero-ref pages whose last aliases disappear.
        // This selected page stays queued until the callback returns, then is
        // removed explicitly below, so no half-allocation can escape.
        willReuse(currentHandle(page))
        freeQueue.remove(page)
        let nextGeneration = generations[Int(page)] &+ 1
        precondition(nextGeneration != 0, "page generation overflow in group \(key)")
        generations[Int(page)] = nextGeneration
        refCounts[Int(page)] = 1
        pagesInUse += 1
        return page
    }

    /// Retain every handle after an all-or-nothing validation pass in the
    /// owning pool. A zero-ref cached page is resurrected from the middle of
    /// the free queue; an already-active page simply gains another owner.
    func retain(_ handle: PagedKVPageHandle) {
        precondition(isValid(handle))
        let index = Int(handle.page)
        if refCounts[index] == 0 {
            precondition(freeQueue.contains(handle.page))
            freeQueue.remove(handle.page)
            pagesInUse += 1
        } else {
            precondition(!freeQueue.contains(handle.page))
        }
        refCounts[index] += 1
    }

    func reclassifyAsUncached(_ handle: PagedKVPageHandle) {
        guard isValid(handle) else { return }
        let index = Int(handle.page)
        guard refCounts[index] == 0, freeQueue.contains(handle.page) else { return }
        freeQueue.moveToFront(handle.page)
    }

    /// Release in caller-supplied eviction order. vLLM frees a request's
    /// page tables tail-first; cached pages append in that same order so a
    /// branch suffix is reused before its more valuable ancestors. Uncached
    /// pages prepend for immediate reuse.
    func release<S: Sequence>(
        _ pages: S, isCached: (PagedKVPageHandle) -> Bool
    ) where S.Element == Int32 {
        var uncached: [Int32] = []
        var cached: [Int32] = []
        for page in pages {
            precondition(
                isAllocatable(page),
                "[PagedKVPool] free of the reserved poison page in group \(key) — a row "
                    + "should never have held it")
            let index = Int(page)
            precondition(refCounts[index] > 0, "double free of page \(page) in group \(key)")
            refCounts[index] -= 1
            guard refCounts[index] == 0 else { continue }
            precondition(!freeQueue.contains(page))
            pagesInUse -= 1
            if isCached(currentHandle(page)) {
                cached.append(page)
            } else {
                uncached.append(page)
            }
        }
        freeQueue.prepend(contentsOf: uncached)
        freeQueue.append(contentsOf: cached)
    }

    /// Queue `page` for release at the end of a speculative transaction.
    /// The page keeps its refcount until the drain.
    func deferFree(_ page: Int32) {
        precondition(
            isAllocatable(page),
            "[PagedKVPool] deferred free of the reserved poison page in group \(key)")
        deferredFrees.append(page)
    }

    func drainDeferredFrees(isCached: (PagedKVPageHandle) -> Bool) {
        release(deferredFrees, isCached: isCached)
        deferredFrees.removeAll(keepingCapacity: true)
    }

    func segment(for page: Int32) -> PagedKVSegment {
        precondition(isAllocatable(page))
        guard let layout = segmentLayout, let segment = segments[layout.segmentIndex(page: page)] else {
            preconditionFailure("page has no committed segment")
        }
        return segment
    }

    struct GrowthPlan {
        let layout: PagedKVSegmentLayout
        let segmentIDs: [Int]
        let physicalBytes: Int
    }

    struct PreparedGrowth {
        let layout: PagedKVSegmentLayout
        let segments: [Int: PagedKVSegment]
        let refCounts: [Int]
        let generations: [UInt64]
        let freeQueue: PagedBlockFreeQueue
    }

    /// Metadata-only plan. Callers check aggregate bytes before materializing
    /// any candidate, then publish every group's replacement as one transaction.
    func planGrowth(usablePages: Int) throws -> GrowthPlan {
        precondition(segmentLayout != nil)
        let planned = try segmentLayout!.adding(
            usablePages: max(0, usablePages - committedUsablePages),
            excluding: Set(segments.keys))
        var physical = committedSegmentBytes
        for index in planned.segmentIDs {
            let (next, overflow) = physical.addingReportingOverflow(
                try planned.layout.allocationBytes(forSegment: index))
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            physical = next
        }
        return GrowthPlan(layout: planned.layout, segmentIDs: planned.segmentIDs, physicalBytes: physical)
    }

    func prepareGrowth(_ plan: GrowthPlan, evaluate: (MLXArray) throws -> Void,
                       admission: AdmissionV2? = nil) throws
        -> PreparedGrowth
    {
        var replacement = segments
        var counts = refCounts
        var versions = generations
        var queue = freeQueue
        let appended = plan.layout.pageCount - pageCount
        if appended > 0 {
            counts.append(contentsOf: repeatElement(0, count: appended))
            versions.append(contentsOf: repeatElement(0, count: appended))
            queue.extend(to: plan.layout.pageCount)
        }
        for index in plan.segmentIDs {
            let segment = try PagedKVSegment(
                index: index, layout: plan.layout, key: key, pageSize: pageSize,
                dtype: dtype, evaluate: evaluate, admission: admission)
            replacement[index] = segment
            counts[segment.pages.lowerBound] = 1
            for page in segment.pages.dropFirst() { queue.append(Int32(page)) }
        }
        return PreparedGrowth(layout: plan.layout, segments: replacement,
                              refCounts: counts, generations: versions, freeQueue: queue)
    }

    /// Only complete ownership swaps occur here. Preparation already performed
    /// all host/GPU allocation, so the pool can recheck its grant under lock.
    func installGrowth(_ prepared: PreparedGrowth) {
        segmentLayout = prepared.layout
        segments = prepared.segments
        refCounts = prepared.refCounts
        generations = prepared.generations
        freeQueue = prepared.freeQueue
    }

    struct ImportPlan {
        let layout: PagedKVSegmentLayout
        let importedIDs: [Int]
        let sourceIDs: [Int]
        let growthIDs: [Int]
        let physicalBytes: Int
        let additionalReservedPages: Int
        let pages: [Int32]
    }

    struct PreparedImport {
        let growth: PreparedGrowth
        let pagesInUse: Int
        let pagesReserved: Int
    }

    /// Rebase only the staged M pages. Existing free backing supplies the
    /// remaining full-N promise before this planner creates any new segments.
    func planImport(_ source: CBv2PagedCheckpointStorage.Group,
                    additionalReservedPages: Int) throws -> ImportPlan {
        guard let current = segmentLayout,
            current.pageBytes == source.layout.pageBytes,
            current.maximumUsablePages == source.layout.maximumUsablePages,
            additionalReservedPages >= source.pages.count
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let sourceIDs = source.segments.keys.sorted()
        let imported = try current.adding(
            usablePages: source.pages.count, excluding: Set(segments.keys))
        guard imported.segmentIDs.count == sourceIDs.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var pageMap: [Int32: Int32] = [:]
        var importedBytes = 0
        for (old, new) in zip(sourceIDs, imported.segmentIDs) {
            let oldRange = source.layout.range(old)
            let newRange = imported.layout.range(new)
            guard oldRange.count == newRange.count else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            importedBytes += source.segments[old]!.allocatedBytes
            for (oldPage, newPage) in zip(oldRange.dropFirst(), newRange.dropFirst()) {
                pageMap[Int32(oldPage)] = Int32(newPage)
            }
        }
        let (promised, overflow) = pagesReserved.addingReportingOverflow(additionalReservedPages)
        guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        let needed = max(0, promised - committedUsablePages - source.pages.count)
        let grown = try imported.layout.adding(
            usablePages: needed, excluding: Set(segments.keys).union(imported.segmentIDs))
        var physical = committedSegmentBytes + importedBytes
        for id in grown.segmentIDs {
            let (next, overflow) = physical.addingReportingOverflow(try grown.layout.allocationBytes(forSegment: id))
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            physical = next
        }
        return ImportPlan(
            layout: grown.layout, importedIDs: imported.segmentIDs, sourceIDs: sourceIDs,
            growthIDs: grown.segmentIDs, physicalBytes: physical,
            additionalReservedPages: additionalReservedPages, pages: try source.pages.map {
                guard let page = pageMap[$0] else { throw CBv2CompleteCheckpointError.invalidManifest }
                return page
            })
    }

    /// All replacement dictionaries, tables and free queues are prepared
    /// before grant publication. Only missing suffix buffers allocate/evaluate.
    func prepareImport(_ plan: ImportPlan, source: CBv2PagedCheckpointStorage.Group,
                       evaluate: (MLXArray) throws -> Void,
                       admission: AdmissionV2? = nil) throws -> PreparedImport {
        var replacement = segments
        var counts = refCounts
        var versions = generations
        var queue = freeQueue
        let appended = plan.layout.pageCount - pageCount
        if appended > 0 {
            counts.append(contentsOf: repeatElement(0, count: appended))
            versions.append(contentsOf: repeatElement(0, count: appended))
            queue.extend(to: plan.layout.pageCount)
        }
        for (old, new) in zip(plan.sourceIDs, plan.importedIDs) {
            replacement[new] = PagedKVSegment(
                rebasing: source.segments[old]!, index: new, layout: plan.layout)
            let range = plan.layout.range(new)
            counts[range.lowerBound] = 1
            for page in range.dropFirst() {
                precondition(counts[page] == 0)
                let next = versions[page] &+ 1
                guard next != 0 else { throw CBv2CompleteCheckpointError.invalidManifest }
                versions[page] = next
                counts[page] = 1
            }
        }
        for id in plan.growthIDs {
            let segment = try PagedKVSegment(
                index: id, layout: plan.layout, key: key, pageSize: pageSize,
                dtype: dtype, evaluate: evaluate, admission: admission)
            replacement[id] = segment
            counts[segment.pages.lowerBound] = 1
            for page in segment.pages.dropFirst() { queue.append(Int32(page)) }
        }
        return PreparedImport(
            growth: PreparedGrowth(layout: plan.layout, segments: replacement,
                                   refCounts: counts, generations: versions, freeQueue: queue),
            pagesInUse: pagesInUse + plan.pages.count,
            pagesReserved: pagesReserved + plan.additionalReservedPages)
    }

    func installImport(_ prepared: PreparedImport) {
        installGrowth(prepared.growth)
        pagesInUse = prepared.pagesInUse
        pagesReserved = prepared.pagesReserved
    }

    /// Called only at the row-release host-sync boundary. Drop cached aliases
    /// before backing; generations survive retirement so reused page IDs cannot
    /// resurrect stale handles. Outstanding reservations retain enough free
    /// physical pages for every nonthrowing future write.
    func trimSegments(willRetire: (PagedKVPageHandle) -> Void) {
        guard segmentLayout != nil else { return }
        var remaining = committedUsablePages
        for index in segments.keys.sorted(by: >) {
            guard let segment = segments[index] else { continue }
            let usable = segment.pages.count - 1
            guard remaining - usable >= pagesReserved else { continue }
            let pages = segment.pages.dropFirst()
            guard pages.allSatisfy({ refCounts[$0] == 0 }) else { continue }
            for page in pages {
                let id = Int32(page)
                willRetire(currentHandle(id))
                freeQueue.remove(id)
                let next = generations[page] &+ 1
                precondition(next != 0, "page generation overflow")
                generations[page] = next
            }
            segment.backing.invalidateCoverage()
            segments.removeValue(forKey: index)
            remaining -= usable
        }
    }

}
