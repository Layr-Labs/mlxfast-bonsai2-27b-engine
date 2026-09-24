// One native paged KV pool per loaded model. Layers with equal
// native storage layout share page ownership and a generation-checked free queue.
//
// The reference layout uses one fixed K/V slab pair per group. The explicit
// segmented configuration instead commits bounded combined K/V buffers as
// admitted demand grows; the total segment count has no dispatch-level cap.
// Both preserve the selected FP16, BF16 or FP32 dtype without KV quantization.
//
// Admission reserves each request's worst-case page demand and materializes
// enough backing before creating its nonthrowing row. Actual page ownership
// grows as tokens are written. Poison pages are excluded from usable capacity.
//
// Buffers remain stable: in-place Metal writes avoid whole-pool slice copies.
// Explicit fence inputs establish GPU buffer barriers for writes and gathers;
// segmented decode preserves complete token partitions and merge order across
// its bounded segment-binding buckets. Retired handles are invalidated before
// page reuse, and release follows the engine's finalized-step discipline.

import Foundation
import MLX

public enum CBv2PagedDefaults {
    /// Page size in tokens. Constant for now; revisit with benchmark data.
    public static let pageSize = 16
}

/// Configuration for a `PagedKVPool`.
public struct PagedKVPoolConfig: Sendable {
    /// Tokens per page. Must divide the decode kernel's expectations; keep
    /// at `CBv2PagedDefaults.pageSize` unless benchmarks say otherwise.
    public var pageSize: Int
    /// Total physical byte budget for K/V backing, including poison pages.
    public var capacityBytes: Int
    /// Native page dtype: `.float16`, `.bfloat16` or `.float32`.
    public var dtype: DType
    /// Actual post-projection/RoPE KV types in attention-storage order. A
    /// build-time forward probe must validate this table before serving. nil
    /// preserves the uniform fixed-reference policy.
    public var layerDTypes: [DType]?
    /// Explicit foundation opt-in. nil retains the fixed-slab reference.
    /// Segments grow under the same byte grant; this is not a total-pool cap.
    public var segmentSizeBytes: Int?
    /// Upper bound on tokens written to a WINDOWED layer in one
    /// `update(keys:values:)` call (i.e. the scheduler's max prefill chunk).
    /// Bounds one windowed update; larger updates trap. Since WS-1.2 the
    /// ring is sized from the WINDOW, not from this, so a chunk no longer
    /// has to fit inside the ring alongside the window — but it must still
    /// fit inside the ring on its own, which `checkedRingPageCount` guards.
    public var maxPrefillChunk: Int
    /// Nominal per-sequence length used only to split `capacityBytes`
    /// across layer groups proportionally to their demand.
    public var nominalMaxSequenceLength: Int
    /// Metal's per-buffer limit, checked for each fixed slab or combined
    /// segment before allocation. Segment growth never copies existing KV.
    public var maxBufferLength: Int
    /// Prefix-cache block size this pool must be able to DONATE and ADOPT
    /// at, or `nil` for a pool that will never participate in block sharing
    /// (unit fixtures, microbenchmarks).
    ///
    /// Set this to the block size of the page-sharing cache. `PagedKVBackend`
    /// does that automatically when `residentPrefixCache` is configured. It
    /// arms the WS-0.6 chunk-coverage guard below, which
    /// is not merely advisory: WS-4's windowed-sharing residency proof
    /// assumes one prefill chunk plus the frontier's partial page covers a
    /// whole block, so a pool that cannot do that must be refused rather
    /// than silently donate blocks the proof does not cover.
    ///
    /// It is opt-IN because `maxPrefillChunk` is an operator/ITL knob
    /// (`SchedulerV2` prefill chunking, and the mixed-prefill cap that
    /// bounds prompt tokens on decode-carrying steps). Making a small chunk
    /// fail engine build unconditionally would turn a latency knob into an
    /// outage for pools that never share a block.
    public var prefixSharingBlockSize: Int?
    public init(
        pageSize: Int = CBv2PagedDefaults.pageSize,
        capacityBytes: Int,
        dtype: DType = .float16,
        maxPrefillChunk: Int = 512,
        nominalMaxSequenceLength: Int = 8192,
        maxBufferLength: Int = MLX.GPU.deviceInfo().maxBufferSize,
        prefixSharingBlockSize: Int? = nil,
        segmentSizeBytes: Int? = nil,
        layerDTypes: [DType]? = nil
    ) {
        self.pageSize = pageSize
        self.capacityBytes = capacityBytes
        self.dtype = dtype
        self.layerDTypes = layerDTypes
        self.maxPrefillChunk = maxPrefillChunk
        self.nominalMaxSequenceLength = nominalMaxSequenceLength
        self.maxBufferLength = maxBufferLength
        self.prefixSharingBlockSize = prefixSharingBlockSize
        self.segmentSizeBytes = segmentSizeBytes
    }
}

/// Global slab pool per model. Thread-affinity: all mutation must happen on
/// the engine loop thread (matching the CBv2 engine discipline); the pool
/// performs no internal locking.
public final class PagedKVPool {
    public let config: PagedKVPoolConfig
    let layerKinds: [CBv2LayerKind]
    /// Authoritative native types and roots in dense attention-storage order.
    public let layerDTypes: [DType]
    private let layerGroupKeys: [PagedKVGroupKey]

    public func groupKey(forLayer index: Int) -> PagedKVGroupKey {
        layerGroupKeys[index]
    }
    /// Validated Metal source retained for the pool's lifetime. Kernel
    /// dispatch never touches Bundle.module and therefore has no
    /// request-time resource-failure path.
    let kernelSource: String
    let segmentGrant: PagedKVGrant?
    /// One engine-ledger owner covers committed and privately preparing native
    /// segments. Only this pool's nominal request KV can offset its floor.
    var physicalLease: CBv2BackendPhysicalLease?
    var memoryAdmission: AdmissionV2?
    var storageTelemetry = PagedKVStorageTelemetry()
    let writeValidation = CBv2PagedKVWriteValidation()
    /// Deterministic failure-order observer; production leaves this unset.
    var checkpointImportBeforeRollback: (() -> Void)?
    let groupDemandBytes: [PagedKVGroupKey: Int]
    let totalDemandBytes: Int
    private(set) var groups: [PagedKVGroupKey: PagedKVGroup] = [:]
    /// At most one page-native prefix index may observe a pool. Weak avoids a
    /// cache↔backend ownership cycle; the cache retains the backend/pool.
    weak var pageReuseObserver: (any PagedKVPageReuseObserver)?

    /// Monotonic identity for every `PagedSequenceKV` minted against this
    /// pool. Unlike `ObjectIdentifier` (a heap address, reusable after
    /// dealloc), serials are NEVER reused, so device block-table caches
    /// fingerprinted by serial can never confuse a finished request's rows
    /// with a new request's (see `PagedLayerCache.deviceTables`).
    private var lastRowSerial: UInt64 = 0

    func nextRowSerial() -> UInt64 {
        lastRowSerial += 1
        return lastRowSerial
    }

    func installPageReuseObserver(_ observer: any PagedKVPageReuseObserver) {
        if let existing = pageReuseObserver {
            precondition(
                existing === observer,
                "[PagedKVPool] only one page-native prefix cache may observe a pool")
        }
        pageReuseObserver = observer
    }

    /// Groups in deterministic order (for tests/telemetry).
    public var groupKeys: [PagedKVGroupKey] {
        groups.keys.sorted { $0.sortKey < $1.sortKey }
    }

    /// Build a pool sized for `layerKinds` (one entry per model layer;
    /// KV-shared layers own no storage and contribute no demand).
    ///
    /// `capacityBytes` is split across groups proportionally to each
    /// group's worst-case per-sequence demand at
    /// `nominalMaxSequenceLength` (windowed layers capped at their ring).
    public init(layerKinds: [CBv2LayerKind], config: PagedKVPoolConfig) throws {
        self.layerKinds = layerKinds
        guard [.float16, .bfloat16, .float32].contains(config.dtype) else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: unsupported page dtype \(config.dtype)")
        }
        guard config.pageSize > 0, config.capacityBytes > 0,
            config.maxPrefillChunk > 0, config.nominalMaxSequenceLength > 0
        else {
            throw CBv2KVError.backendIneligible(reason: "PagedKVPool: invalid config")
        }
        guard config.maxBufferLength > 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: Metal maxBufferLength is unavailable")
        }
        // The decode kernel partitions attention into fixed
        // `PagedAttentionKernel.partitionTokens`-token slices and requires
        // page boundaries to align with them (`PTOK % pageSize == 0` is a
        // kernel-launch precondition). Reject misaligned page sizes HERE,
        // at construction, so a bad config fails engine build with a clear
        // `backendIneligible` instead of trapping on the first decode
        // (PR#62 review).
        guard PagedAttentionKernel.partitionTokens % config.pageSize == 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: pageSize \(config.pageSize) must evenly divide "
                    + "PagedAttentionKernel.partitionTokens "
                    + "(\(PagedAttentionKernel.partitionTokens)) — use a power-of-two "
                    + "divisor such as \(CBv2PagedDefaults.pageSize)")
        }
        // WS-0.6, invariant 1: a prefix-cache BLOCK must be a whole number
        // of pages.
        //
        // `CBv2BlockHasher.defaultBlockSize` (256) and
        // `CBv2PagedDefaults.pageSize` (16) are declared in two files with
        // no cross-reference, and their divisibility is what makes windowed
        // sharing a pointer swap: every matched block boundary is then also
        // a page boundary, so an adopter's post-adoption writes start at
        // slot 0 of a fresh page and `restoreWindow(_:at:)`
        // never has to copy a partial page (PagedSeamContract.swift, WS-4.1
        // — which explicitly defers the assertion to this guard). Violating
        // it does not fail loudly anywhere; it silently makes adoption
        // wrong. Checked unconditionally: it is a property of two
        // constants, so no legitimate configuration can need it relaxed.
        guard CBv2BlockHasher.defaultBlockSize % config.pageSize == 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: pageSize \(config.pageSize) must evenly divide the "
                    + "prefix-cache block size \(CBv2BlockHasher.defaultBlockSize) "
                    + "(CBv2BlockHasher.defaultBlockSize) — otherwise a matched block "
                    + "boundary is not a page boundary and windowed sharing cannot adopt "
                    + "pages without a partial-page copy")
        }
        // WS-0.6, invariant 2: one prefill chunk plus the frontier's
        // partial page must cover a whole block, so a block can always be
        // completed without a second chunk straddling it.
        //
        // Armed only for pools that declare they will share blocks — see
        // `PagedKVPoolConfig.prefixSharingBlockSize` for why this is opt-in
        // rather than a blanket refusal of small prefill chunks.
        if let blockSize = config.prefixSharingBlockSize {
            guard blockSize > 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: prefixSharingBlockSize \(blockSize) must be positive")
            }
            guard blockSize % config.pageSize == 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: pageSize \(config.pageSize) must evenly divide the "
                        + "declared prefix sharing block size \(blockSize)")
            }
            // `maxPrefillChunk` is operator-influenced and may be Int.max in
            // hostile-size tests, so the sum is overflow-checked rather than
            // written inline.
            let chunkSpan = try Self.checkedAdd(
                config.maxPrefillChunk, config.pageSize, context: "prefill chunk block span")
            guard chunkSpan >= blockSize else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: maxPrefillChunk \(config.maxPrefillChunk) + pageSize "
                        + "\(config.pageSize) = \(chunkSpan) cannot cover the declared prefix "
                        + "sharing block size \(blockSize) — a chunk that cannot complete a "
                        + "block leaves WS-4's windowed-sharing residency proof unsatisfied; "
                        + "raise maxPrefillChunk, lower prefixSharingBlockSize, or set it to "
                        + "nil for a pool that never shares blocks")
            }
        }
        let resolvedTypes = try PagedKVStorageLayout.resolve(layerKinds: layerKinds, config: config)
        let groupKeys = layerKinds.enumerated().map { index, kind in
            PagedKVGroupKey(
                kind, dtype: resolvedTypes[index],
                separateWindow: config.segmentSizeBytes != nil || config.layerDTypes != nil)
        }
        // Demand-proportional capacity split.
        let owning = layerKinds.filter { $0.sharesKVWithLayer == nil }
        guard !owning.isEmpty else {
            throw CBv2KVError.backendIneligible(reason: "PagedKVPool: no storage-owning layers")
        }
        for kind in owning {
            guard kind.kvHeads > 0, kind.headDim > 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: non-positive KV shape "
                        + "\(kind.kvHeads)x\(kind.headDim)")
            }
            if case .slidingWindow(let window) = kind.attention {
                guard window > 0 else {
                    throw CBv2KVError.backendIneligible(
                        reason: "PagedKVPool: invalid sliding window \(window)")
                }
                _ = try Self.checkedRingPageCount(window: window, config: config)
            }
        }

        let source: String
        do {
            source = try PagedAttentionResources.loadSourceForCurrentProcess()
        } catch {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: paged-attention runtime resource unavailable: \(error)")
        }
        self.config = config
        self.layerDTypes = resolvedTypes
        self.layerGroupKeys = groupKeys
        self.kernelSource = source
        self.segmentGrant = config.segmentSizeBytes == nil ? nil : PagedKVGrant(bytes: config.capacityBytes)

        var demandTokens: [PagedKVGroupKey: Int] = [:]
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            let key = groupKeys[index]
            let tokens = Self.perSequenceTokenDemand(kind: kind, config: config)
            demandTokens[key] = try Self.checkedAdd(
                demandTokens[key, default: 0],
                tokens,
                context: "group token demand")
        }
        var demandBytes: [PagedKVGroupKey: Int] = [:]
        var totalDemand = 0
        for (key, tokens) in demandTokens {
            let bytesPerToken = try Self.checkedMultiply(
                [2, key.kvHeads, key.headDim, key.dtype.size],
                context: "bytes per token")
            let bytes = try Self.checkedMultiply(
                [tokens, bytesPerToken],
                context: "group byte demand")
            demandBytes[key] = bytes
            totalDemand = try Self.checkedAdd(
                totalDemand, bytes, context: "total byte demand")
        }
        guard totalDemand > 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: zero byte demand")
        }
        self.groupDemandBytes = demandBytes
        self.totalDemandBytes = totalDemand
        for (key, bytes) in demandBytes {
            let pageBytes = try Self.checkedMultiply(
                [2, key.kvHeads, config.pageSize, key.headDim, key.dtype.size],
                context: "page bytes")
            if let target = config.segmentSizeBytes {
                let layout = try PagedKVSegmentLayout(
                    pageBytes: pageBytes, targetBytes: target,
                    maximumBufferBytes: config.maxBufferLength,
                    maximumAddressPages: Int(Int32.max) / config.pageSize)
                groups[key] = PagedKVGroup(
                    key: key, pageCount: 0, pageSize: config.pageSize,
                    dtype: key.dtype, segmentLayout: layout, writeValidation: writeValidation)
                continue
            }
            // Keep the fixed reference's construction quota unchanged.
            let share = Double(bytes) / Double(totalDemand)
            let groupBytes = Int(share * Double(config.capacityBytes))
            let pageCount = groupBytes / pageBytes
            guard pageCount >= 2 else {
                throw CBv2KVError.capacityExhausted(
                    needed: try Self.checkedMultiply([2, pageBytes], context: "minimum group bytes"),
                    available: groupBytes)
            }
            guard pageCount <= Int(Int32.max) else {
                throw CBv2KVError.backendIneligible(reason: "paged group exceeds Int32 page-table limit")
            }
            let slabBytes = try Self.checkedMultiply(
                [pageCount, key.kvHeads, config.pageSize, key.headDim, key.dtype.size],
                context: "slab bytes")
            guard slabBytes <= config.maxBufferLength else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: group \(key) slab requires \(slabBytes) B, "
                        + "over Metal maxBufferLength \(config.maxBufferLength) B")
            }
            groups[key] = PagedKVGroup(
                key: key, pageCount: pageCount, pageSize: config.pageSize, dtype: key.dtype,
                writeValidation: writeValidation)
        }
    }

    private static func checkedAdd(
        _ lhs: Int,
        _ rhs: Int,
        context: String
    ) throws -> Int {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: integer overflow computing \(context)")
        }
        return sum
    }

    private static func checkedMultiply(
        _ values: [Int],
        context: String
    ) throws -> Int {
        var product = 1
        for value in values {
            let (next, overflow) = product.multipliedReportingOverflow(by: value)
            guard !overflow else {
                throw CBv2KVError.backendIneligible(
                    reason: "PagedKVPool: integer overflow computing \(context)")
            }
            product = next
        }
        return product
    }

    /// `ringPageCount` with BOTH sizing invariants enforced, each on its own,
    /// and with every intermediate overflow-checked so an operator-supplied
    /// window, chunk or page size fails engine build instead of trapping the
    /// process.
    ///
    /// The two bounds are checked SEPARATELY and neither is derived from the
    /// other, because which one binds is pure geometry:
    ///
    ///     window 1024, chunk  512  ->  cache 1032, row  512   cache-bound
    ///     window  128, chunk 2048  ->  cache  136, row 2048   row-bound
    ///
    /// A ring sized from the cache bound alone gives the second config 9
    /// pages for a 128-page write. `ringSizingIsBoundedOnBothSides` in
    /// CBv2PagedPoolGuardTests is the desk-speed version of that; this is the
    /// engine-build one.
    ///
    /// The page count comes from `ringPageCount`, NOT from a second copy of
    /// the arithmetic — the guards below have to be able to catch an edit to
    /// the shipping formula, which they cannot do if they re-derive it.
    private static func checkedRingPageCount(
        window: Int,
        config: PagedKVPoolConfig
    ) throws -> Int {
        // Overflow-check what `ringPageCount` computes unchecked, BEFORE
        // calling it, so a hostile config is `backendIneligible` and not a
        // trap.
        let cacheTokens = try ringCacheBoundTokens(window: window)
        let rowTokens = ringRowBoundTokens(config: config)
        _ = try checkedAdd(
            max(cacheTokens, rowTokens), config.pageSize - 1, context: "window ring rounding")
        let pages = ringPageCount(window: window, config: config)
        let ringTokens = try checkedMultiply(
            [pages, config.pageSize], context: "window ring tokens")

        // CACHE bound. Under-size it and `gatherRange` trips "gather of
        // evicted window range" on ordinary prefill (a process abort), or a
        // rolled-back speculative write aliases a live in-window entry
        // (corrupted KV, no crash at all).
        guard ringTokens >= cacheTokens else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: windowed ring of \(pages) pages (\(ringTokens) tokens) "
                    + "cannot hold the exposed window \(PagedSequenceKV.maxWindowExposure(window: window)) "
                    + "(window \(window)) plus the speculative span "
                    + "\(CBv2PagedSpeculation.maxSpeculativeSpan)")
        }
        // ROW bound. Under-size it and one `PagedSequenceKV.write` puts two
        // of its own tokens in the same physical slot inside a single kernel
        // dispatch, with no ordering between them.
        guard ringTokens >= rowTokens else {
            throw CBv2KVError.backendIneligible(
                reason: "PagedKVPool: windowed ring of \(pages) pages (\(ringTokens) tokens) "
                    + "cannot hold one maxPrefillChunk write of \(rowTokens) tokens — a chunk "
                    + "longer than the ring laps itself inside one bulk-write dispatch. This "
                    + "bound is INDEPENDENT of the window; do not re-derive the ring from the "
                    + "window alone")
        }
        return pages
    }

    /// CACHE bound, in tokens: the widest range a row exposes plus one
    /// speculative round.
    ///
    /// `PagedSequenceKV.maxWindowExposure` is the authority for the first
    /// term — it is the same function `retainedCount` clamps to, so a change
    /// that re-widens what a row can be asked to gather grows the ring here
    /// instead of silently out-running it.
    private static func ringCacheBoundTokens(window: Int) throws -> Int {
        try checkedAdd(
            PagedSequenceKV.maxWindowExposure(window: window),
            CBv2PagedSpeculation.maxSpeculativeSpan,
            context: "windowed attendable span")
    }

    /// ROW bound, in tokens: one `PagedSequenceKV.write` must fit the ring.
    private static func ringRowBoundTokens(config: PagedKVPoolConfig) -> Int {
        config.maxPrefillChunk
    }

    /// Worst-case tokens a single sequence can pin in one layer of `kind`.
    static func perSequenceTokenDemand(kind: CBv2LayerKind, config: PagedKVPoolConfig) -> Int {
        let maxLen = config.nominalMaxSequenceLength
        switch kind.attention {
        case .full:
            return maxLen
        case .slidingWindow(let window):
            return min(maxLen, ringPageCount(window: window, config: config) * config.pageSize)
        }
    }

    /// Ring length (in pages) for a windowed layer: the SMALLEST page count
    /// covering both sizing bounds at once.
    ///
    ///     ring * pageSize  >=  max(maxWindowExposure(window) + maxSpeculativeSpan,
    ///                              maxPrefillChunk)
    ///
    /// CACHE bound (`maxWindowExposure + maxSpeculativeSpan`). The widest
    /// range a row can be asked to GATHER is `PagedSequenceKV
    /// .maxWindowExposure(window:)` — that function is the authority, and
    /// `retainedCount` clamps to it — plus one speculative round, because
    /// writing position `p` destroys whatever held `p - ringTokens`.
    ///
    /// ROW bound (`maxPrefillChunk`). One `PagedSequenceKV.write` scatters a
    /// whole chunk in ONE dispatch; longer than the ring and two of its own
    /// tokens land in one physical slot with no ordering between them. This
    /// bound used to be implied by the chunk term inside the cache bound. It
    /// is not implied any more, so it is explicit, and it DOMINATES whenever
    /// the chunk outruns the window: window 128 / chunk 2,048 needs 128
    /// pages where the cache bound alone would hand out 9.
    ///
    /// gemma-4 (window 1,024, chunk 512, pageSize 16, span 8):
    /// `ceil(max(1032, 512) / 16) == 65` pages == 1,040 tokens, against 97
    /// pages == 1,552 tokens before. That 528-token overshoot was the whole
    /// of the measured 1.10x per-sequence KV regression versus the contiguous
    /// backend at 10k context, on 25 of gemma-4's 30 layers.
    ///
    /// ### 65 IS CONDITIONAL. READ THIS BEFORE TOUCHING ANY OF IT.
    ///
    /// A 65-page ring was tried once before and REVERTED: it aborted the
    /// daemon in ordinary windowed prefill, reproduced from the row side. The
    /// number is not what changed. Three things did, and 65 is wrong again
    /// the moment any of them stops holding:
    ///
    ///  1. `PagedLayerCache.prefillKV` gathers a chunk's window history
    ///     BEFORE `row.write` and attends `gather ++ chunk`. Gathering after
    ///     the write asks for `window - 1 + chunk` — 1,535 tokens out of
    ///     1,040 — and trips `gatherRange`'s eviction precondition. That is
    ///     the abort.
    ///  2. `PagedSequenceKV.update` does the same on the protocol path
    ///     (tests and `PagedDecodeProfiler`; the serving path is (1)), which
    ///     is what collapses `retainedCount` to `min(written, window)` and
    ///     removes the chunk term from the cache bound. `maxWindowExposure`
    ///     is the coupling: this formula reads it, so widening the row's
    ///     exposure grows the ring rather than out-running it.
    ///  3. `PagedKVPool.gather` publishes a fence BACK-edge, so a chunk write
    ///     cannot overtake a pre-write gather that has not materialised. At
    ///     1,552 tokens the history and the chunk never shared a ring slot
    ///     and the missing edge was benign; at 1,040 they do share slots.
    ///
    /// `checkedRingPageCount` re-checks (1)/(2) as arithmetic at pool build
    /// and `CBv2PagedSeamContractRingFormulaTests` pins the condition rather
    /// than the number. The speculative term binds to
    /// `CBv2PagedSpeculation.maxSpeculativeSpan` rather than a literal so
    /// this sizing and `PagedSequenceKV.speculativeHeadroom` cannot disagree.
    ///
    /// So the live hazard is no longer "someone shrinks the ring" — that is
    /// arithmetic, and `checkedRingPageCount` catches it. It is "someone
    /// re-widens what a row exposes, or removes the ordering (3) supplies".
    /// Three ways to do that which all read as tidying:
    ///
    ///  1. Hoisting `PagedLayerCache`'s gather OUT of the per-row body of
    ///     `CBv2AttentionV1.packedPerRow` so it runs once across the batch.
    ///     It reads as deduplication; it reintroduces a post-write gather
    ///     for every row after the first. The gather must stay hoisted PER
    ///     ROW — before that row's own write.
    ///  2. Reading `retainedPrefillKV[0]` instead of
    ///     `retainedPrefillKV[index]` in `PagedLayerCache.attendBorrowing`.
    ///     It compiles, and it silently serves row 0's history to every
    ///     KV-shared sibling of a packed prefill.
    ///  3. Deleting the fence BACK-edge at the end of `PagedKVPool.gather`
    ///     as a redundant no-op. It multiplies the fence by zero, so it
    ///     looks like dead arithmetic and reads like a performance win; it
    ///     is the only thing ordering a later bulk write after this read.
    ///
    /// (1) and (2) widen exposure, so `checkedRingPageCount` can catch them.
    /// (3) does not: no precondition trips and the KV corrupts silently.
    static func ringPageCount(window: Int, config: PagedKVPoolConfig) -> Int {
        let tokens = max(
            PagedSequenceKV.maxWindowExposure(window: window)
                + CBv2PagedSpeculation.maxSpeculativeSpan,
            config.maxPrefillChunk)
        return (tokens + config.pageSize - 1) / config.pageSize
    }

    /// Pages one layer of `kind` must be able to hold for a sequence
    /// bounded by `maxLength` tokens. This is the ADMISSION charge, and it
    /// is also the row's hard allocation cap (`PagedSequenceKV
    /// .reservedPages`), so it may never be less than the row's true peak.
    ///
    /// EXACTNESS (WS-1.3). For a row that writes from position 0 this is
    /// not a bound, it is an identity: it equals the peak `table.count`
    /// exactly, with zero slack, for every (pageSize, window,
    /// maxPrefillChunk, maxLength) combination. That falls out of
    /// `PagedSequenceKV.ensurePage`, which grows the table to
    /// `maxSlotTouched + 1` where the slot of absolute position `p` is
    /// `(p / pageSize) % ringPages`: a row sweeping [0, maxLength) touches
    /// slots 0…min(ceil(maxLength / pageSize), ringPages) - 1 and no
    /// others. Change either side and this stops holding — the identity is
    /// pinned by `chargeEqualsPeakResidency` in CBv2PagedPoolGuardTests.
    ///
    /// CONSEQUENCE: there is no safe reduction available HERE. WS-1.3's
    /// "charge min(ctx, window)" reads as a change to this function, but the
    /// charge is already the tight bound — what makes it large is the ring,
    /// and charging below the ring is a free-list underflow:
    /// `PagedKVGroup.allocatePage` traps, which is a daemon abort under
    /// load, not a rejected request.
    ///
    /// The win therefore lived in `ringPageCount`, and it has LANDED: the
    /// ring is now `ceil(max(maxWindowExposure(window) + maxSpeculativeSpan,
    /// maxPrefillChunk) / pageSize)`, and this `min` inherited the reduction
    /// for free with no change to the line below. On gemma-4 (window 1,024,
    /// pageSize 16, chunk 512) that is 65 pages / 1,040 tokens, down from
    /// the 97 / 1,552 the earlier `window - 1 + maxPrefillChunk` ring
    /// charged. Derive the figure from `ringPageCount` rather than trusting
    /// either number here — this comment is the third place they have
    /// rotted.
    ///
    /// The shorter ring is only legitimate because THREE things now hold
    /// together; an earlier attempt at 65 without them aborted the daemon in
    /// ordinary prefill. (1) The pre-write gather is on both the layer and
    /// the row path, so a chunk attends `gather(ring) ++ chunk` rather than
    /// re-reading slots it is about to overwrite. (2) `retainedCount` is
    /// clamped to `PagedSequenceKV.maxWindowExposure`, which is also the
    /// first term of the ring, so widening a row's exposure grows the ring
    /// instead of silently out-running it. (3) `PagedKVPool.gather`
    /// publishes a fence BACK-edge. That last one was a latent bug, not a
    /// new requirement: the gather and `writeTokens` were graph SIBLINGS,
    /// benign at 1,552 tokens because a chunk's history and the chunk never
    /// shared a ring slot, and silently corrupting at 1,040 because they do.
    /// Removing any of the three re-arms the abort; see `ringPageCount`.
    ///
    /// Rows adopted mid-stream (`fastForward`) are the one CONSERVATIVE
    /// case: their first write lands at ring slot `(base / pageSize) %
    /// ringPages`, so they allocate that slot's prefix eagerly but never
    /// exceed the ring. The charge over-reserves them by up to `ringPages -
    /// ceil((maxLength - base) / pageSize)` pages. Tightening that needs
    /// the adoption offset, which only `PagedKVBackend` knows at admission.
    static func pageDemand(kind: CBv2LayerKind, maxLength: Int, config: PagedKVPoolConfig) -> Int {
        let maxPages = (maxLength + config.pageSize - 1) / config.pageSize
        switch kind.attention {
        case .full:
            return maxPages
        case .slidingWindow(let window):
            return min(maxPages, ringPageCount(window: window, config: config))
        }
    }

    func group(_ key: PagedKVGroupKey) -> PagedKVGroup {
        guard let g = groups[key] else {
            fatalError("[PagedKVPool] unknown group \(key) — sequence built for another pool?")
        }
        return g
    }

    // MARK: - Poison page (WS-0.5)

    /// The group's reserved, permanently-zeroed, never-allocatable page.
    ///
    /// Call sites that must pad an array of page ids up to a kernel's
    /// minimum length pad with THIS, never with a real page id. See
    /// `PagedKVGroup.poisonPage` for why it is page 0.
    public func poisonPage(group key: PagedKVGroupKey) -> Int32 {
        group(key).poisonPage
    }

    /// Pages of `key` a sequence row can own — the reservation ceiling.
    /// One less than the group's physical page count.
    public func usablePageCount(group key: PagedKVGroupKey) -> Int {
        group(key).usablePageCount
    }

    /// True when `page` is a page a sequence row can own. False for the
    /// poison page and for out-of-range ids.
    public func isAllocatablePage(_ page: Int32, group key: PagedKVGroupKey) -> Bool {
        group(key).isAllocatable(page)
    }

    // MARK: - Reservation (admission)

    /// Atomically reserve worst-case page counts per group; throws
    /// `capacityExhausted` (in bytes) without partial effects.
    public func reserve(_ needs: [PagedKVGroupKey: Int]) throws {
        if segmentGrant != nil {
            try reserveSegments(needs)
            return
        }
        // Validate everything first — no partial reservations.
        for (key, pages) in needs {
            guard pages >= 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "negative paged KV reservation for group \(key): \(pages)")
            }
            let g = group(key)
            let available = g.usablePageCount - g.pagesReserved
            if pages > available {
                let (neededBytes, overflow) = pages.multipliedReportingOverflow(by: g.pageBytes)
                throw CBv2KVError.capacityExhausted(
                    needed: overflow ? Int.max : neededBytes,
                    available: max(0, available) * g.pageBytes)
            }
        }
        for (key, pages) in needs {
            group(key).pagesReserved += pages
        }
    }

    public func unreserve(_ needs: [PagedKVGroupKey: Int]) {
        // Validate the complete release before touching any group so a bad
        // caller cannot partially corrupt a multi-group reservation ledger.
        for (key, pages) in needs {
            let reserved = group(key).pagesReserved
            precondition(pages >= 0, "negative paged KV unreservation for group \(key)")
            precondition(
                pages <= reserved,
                "unreserve underflow for group \(key): \(pages) > \(reserved)")
        }
        for (key, pages) in needs {
            group(key).pagesReserved -= pages
        }
        trimFreeSegments()
    }

    /// Retire free native segments after reservation/row release. Fixed pools
    /// retain their reference behavior until segmented execution is promoted.
    func trimFreeSegments() {
        guard config.segmentSizeBytes != nil else { return }
        for key in groupKeys {
            group(key).trimSegments { [unowned self] handle in
                pageReuseObserver?.pagedKVPool(self, willReuse: handle)
            }
        }
        physicalLease?.release(to: bytesMaterialized)
    }

    deinit {
        // Native segment aliases must leave the pool before its charge does.
        for group in groups.values {
            for segment in group.segments.values { segment.backing.invalidateCoverage() }
        }
        groups.removeAll()
        physicalLease?.close()
    }

    // MARK: - Page lifecycle

    func allocatePage(group key: PagedKVGroupKey) -> Int32 {
        group(key).allocatePage { [unowned self] handle in
            pageReuseObserver?.pagedKVPool(self, willReuse: handle)
        }
    }

    func freePages(group key: PagedKVGroupKey, pages: some Sequence<Int32>) {
        let g = group(key)
        g.release(pages) { [unowned self] handle in
            pageReuseObserver?.pagedKVPool(self, isCached: handle) ?? false
        }
    }

    func currentHandle(group key: PagedKVGroupKey, page: Int32) -> PagedKVPageHandle {
        group(key).currentHandle(page)
    }

    func isValid(_ handle: PagedKVPageHandle) -> Bool {
        groups[handle.group]?.isValid(handle) == true
    }

    /// All-or-nothing generation validation, followed by one retain per page
    /// table reference. Must run on the engine queue.
    func retainPages(_ handles: some Collection<PagedKVPageHandle>) -> Bool {
        guard handles.allSatisfy({ isValid($0) }) else { return false }
        for handle in handles {
            group(handle.group).retain(handle)
        }
        return true
    }

    func refCount(for handle: PagedKVPageHandle) -> Int? {
        guard isValid(handle) else { return nil }
        return group(handle.group).refCounts[Int(handle.page)]
    }

    /// Move a generation-valid, zero-ref page to the immediate-reuse end after
    /// its final cache alias disappears. Active pages are classified when
    /// their last owner releases them instead.
    func reclassifyAsUncached(_ handle: PagedKVPageHandle) {
        groups[handle.group]?.reclassifyAsUncached(handle)
    }

    /// Queue `pages` for release at the END of a speculative transaction
    /// (WS-3.2c). Queued pages keep `refCount > 0`, so they stay out of the
    /// free list — a page cannot be recycled to another row while a
    /// speculative round's captures still name it.
    ///
    /// The caller MUST have already removed `pages` from its own page
    /// table: the queue and any live table must stay disjoint, or the
    /// drain and the row's own release both free the same page and trip
    /// `freePage`'s double-free precondition.
    func deferFreePages(group key: PagedKVGroupKey, pages: some Sequence<Int32>) {
        let g = group(key)
        for page in pages {
            g.deferFree(page)
        }
    }

    /// Release everything queued by `deferFreePages`, across ALL groups.
    /// No-op when nothing is queued, so callers may invoke it
    /// unconditionally at commit and at release.
    ///
    /// POOL-WIDE BY CONTRACT (`PagedSeamContract.swift` freezes the no-arg
    /// signature). If two rows are mid-transaction and one commits, the
    /// other's queued pages drain too. That is safe only because every row
    /// of an MTP round commits inside the SAME finalize loop
    /// (`MTP/EngineLoopV2+MTPFinalize.swift:146`), on the engine thread,
    /// between steps — so no queued page is still named by an in-flight
    /// capture at any drain point. Committing rows across step boundaries
    /// would break this and must not be introduced without making the
    /// queue per-row.
    func drainDeferredFrees() {
        for g in groups.values {
            g.drainDeferredFrees { [unowned self] handle in
                pageReuseObserver?.pagedKVPool(self, isCached: handle) ?? false
            }
        }
    }

    // MARK: - Writes
    //
    // All bulk writes go through the in-place write kernel and advance the
    // group's fence chain (see file header — slice updates on the slabs
    // are forbidden). Runtime K/V must match their observed native dtype.

    /// Scatter `slots.count` tokens into the group's slabs. `keys`/`values`
    /// are `[kvHeads, n, headDim]`; `slots[i]` is token `i`'s physical
    /// position as `page * pageSize + slot`.
    func writeTokens(
        group key: PagedKVGroupKey, slots: [Int32], keys: MLXArray, values: MLXArray
    ) {
        guard !slots.isEmpty else { return }
        let g = group(key)
        guard writeValidation.validate(keys: keys, values: values, expected: g.dtype) else { return }
        if g.segmentLayout != nil {
            PagedSegmentTransfers.write(group: g, slots: slots, keys: keys, values: values)
            return
        }
        precondition(keys.dim(1) == slots.count && values.dim(1) == slots.count)
        let k = keys
        let v = values
        // Pad to >= 8 entries so the generated kernel signature keeps the
        // device address space. The pad target is the group's reserved
        // POISON page, never a real slot.
        //
        // Pad entries are provably never dereferenced: `bulkWrite`
        // dispatches grid `(headDim, kvHeads, n)` with `n` the TRUE token
        // count, so the kernel only ever indexes `slots[0 ..< n]`. This
        // pad used to repeat `slots[n - 1]`, a live physical slot of the
        // writing row, which made the safety argument fail-OPEN: it rested
        // entirely on that grid bound, and a violation would silently
        // rewrite a real token. Poison makes it fail-SAFE — a stray write
        // lands on a page no sequence can own.
        var padded = slots
        if padded.count < 8 {
            let poisonSlot = g.poisonPage * Int32(g.pageSize)
            padded.append(contentsOf: repeatElement(poisonSlot, count: 8 - padded.count))
        }
        do {
            g.writeFence = try PagedAttentionKernel.bulkWrite(
                kSlab: g.kSlab, vSlab: g.vSlab,
                keys: k, values: v,
                slots: MLXArray(padded),
                prevFence: g.writeFence,
                pageSize: g.pageSize,
                kernelSource: kernelSource)
        } catch {
            writeValidation.record(error)
        }
    }

    // MARK: - Reads

    /// Gather `count` tokens in temporal order from `pages` (ordered oldest
    /// to newest), where the first token lives at `firstSlot` of `pages[0]`.
    /// Returns `([1, kvHeads, count, headDim]) x 2` in the pool dtype.
    ///
    /// This materializes a copy (MLX gathers always do); callers on the hot
    /// decode path must prefer the kernel, which reads the slabs in place.
    /// The returned arrays are LAZY reads of the shared slabs — evaluate
    /// or drop them within the current engine step: the slabs are mutated
    /// in place, so a stale unevaluated gather could observe pages after
    /// they were recycled and rewritten by another row.
    ///
    /// ORDERING, BOTH WAYS. The fence edge into `idx` orders this read after
    /// every prior write of the group. The BACK-edge published at the end
    /// orders every LATER write of the group after this read, and that half
    /// is not optional: the pre-write gather in `PagedLayerCache.prefillKV`
    /// and `PagedSequenceKV.update` is lazy, and without the back-edge it and
    /// the chunk write that follows it are graph SIBLINGS — both merely
    /// consume `writeFence` — so which one MLX runs first is an
    /// implementation detail. Under the old 1,552-token ring that was benign
    /// because a chunk's history (`window - 1`) and the chunk itself never
    /// shared a ring slot. Under the 1,040-token ring they do, so a write
    /// winning the race silently hands the chunk its own tail as history.
    /// `CBv2MTPCaptureFence` publishes the same edge for MTP captures and
    /// documents the hazard at length; this is the unconditional version, and
    /// it must stay unconditional — gating it on ring geometry re-arms the
    /// race at the next resize.
    func gather(
        group key: PagedKVGroupKey, pages: [Int32], firstSlot: Int, count: Int
    ) -> (keys: MLXArray, values: MLXArray) {
        let g = group(key)
        if g.segmentLayout != nil {
            return PagedSegmentTransfers.gather(group: g, pages: pages, firstSlot: firstSlot, count: count)
        }
        let h = g.key.kvHeads
        let d = g.key.headDim
        let s = g.pageSize
        guard count > 0 else {
            let empty = MLXArray.zeros([1, h, 0, d], dtype: g.dtype)
            return (empty, empty)
        }
        precondition(firstSlot < s)
        precondition(pages.count * s >= firstSlot + count, "gather range exceeds page list")
        // Consume the group's write fence: in-place writes are invisible
        // to MLX's hazard tracking, so the dependency edge (and the memory
        // barrier it induces) is what orders this gather after every prior
        // bulk write of the group.
        let idx = MLXArray(pages) + g.writeFence * 0
        func assemble(_ slab: MLXArray) -> MLXArray {
            // [np, H, S, D] -> [H, np, S, D] -> [H, np*S, D] -> slice -> [1, H, count, D]
            take(slab, idx, axis: 0)
                .transposed(1, 0, 2, 3)
                .reshaped([h, pages.count * s, d])[0..., firstSlot ..< (firstSlot + count), 0...]
                .expandedDimensions(axis: 0)
        }
        let keys = assemble(g.kSlab)
        let values = assemble(g.vSlab)
        // The back-edge. ONE element of each is enough: MLX schedules whole
        // primitives, so a dependency on any slice of the gather forces the
        // gather itself. (`CBv2MTPCaptureFence` publishes the identical edge
        // the same way, for the same reason.) `* 0` in int32 is exactly zero
        // for EVERY input, including whatever an out-of-range or NaN
        // float->int conversion produces, so the fence keeps its VALUE and
        // gains only the edge.
        let probe = keys[0, 0, 0, 0] + values[0, 0, 0, 0]
        g.writeFence = g.writeFence + probe.asType(g.writeFence.dtype) * 0
        return (keys, values)
    }

    // MARK: - Accounting

    /// Truthful bytes behind pages sequences have actually touched.
    public var bytesInUse: Int {
        groups.values.reduce(0) { $0 + $1.pagesInUse * $1.pageBytes }
    }

    /// Bytes promised to admitted sequences (the admission-relevant figure).
    public var bytesReserved: Int {
        groups.values.reduce(0) { $0 + $1.pagesReserved * $1.pageBytes }
    }

    /// Segmented pools expose the current shared physical grant; their exact
    /// growth planner charges poison/slack inside it. Fixed pools retain their
    /// construction-time tenant capacity with poison pages excluded.
    public var bytesCapacity: Int {
        if let segmentGrant { return segmentGrant.snapshot().bytes }
        return groups.values.reduce(0) { $0 + $1.usablePageCount * $1.pageBytes }
    }

    /// Current physical grant (fixed pools retain their configured ceiling).
    /// A segmented pool may retain pre-shrink owners above this value; use
    /// segmentStorageSnapshot.overGrantBytes to observe that debt.
    public var bytesPhysical: Int {
        if let segmentGrant { return segmentGrant.snapshot().bytes }
        return groups.values.reduce(0) { $0 + $1.pageCount * $1.pageBytes }
    }

    /// Uncommitted portion of the configured backing. Segmented retirement
    /// increases this value; a later admission may commit that capacity again.
    public var bytesUnmaterialized: Int {
        if let segmentGrant {
            return max(0, segmentGrant.snapshot().bytes - bytesMaterialized)
        }
        return groups.values.reduce(0) {
            $0 + ($1.kSlabMaterialized ? 0 : $1.slabBytes)
                + ($1.vSlabMaterialized ? 0 : $1.slabBytes)
        }
    }

    /// Native ownership is not clamped to a shrunken grant. Existing rows
    /// retain their backing until release, and diagnostics expose that debt.
    public var bytesMaterialized: Int {
        if segmentGrant != nil {
            return groups.values.reduce(0) { $0 + $1.committedSegmentBytes }
        }
        return bytesPhysical - bytesUnmaterialized
    }

    /// How one slab is evaluated into residency. Internal seam so tests can
    /// inject a deterministic allocation failure mid-materialization;
    /// production is the scoped-handler eval below and nothing else.
    var slabEval: (MLXArray) throws -> Void = { slab in
        // `withError` binds MLX's task-local SCOPED error handler for
        // exactly this eval: a Metal allocation failure inside the C++
        // layer is caught at the mlx-c boundary (after a clean C++ unwind —
        // no error ever throws across C++ frames) and surfaces as a thrown
        // `MLXError` rather than reaching the process-fatal default
        // handler. Never a process-global handler swap.
        try withError { eval(slab) }
    }

    /// Explicitly materialize the configured pool for eager profiling.
    /// Fixed slabs record progress individually so a partial failure retries
    /// only missing slabs. Segmented buffers publish as one transaction.
    public func materializeSlabs() throws {
        if config.segmentSizeBytes != nil {
            try materializeSegments(all: true)
            return
        }
        for key in groups.keys.sorted(by: { $0.description < $1.description }) {
            let g = groups[key]!
            if !g.kSlabMaterialized {
                try slabEval(g.kSlab)
                g.kSlabMaterialized = true
            }
            if !g.vSlabMaterialized {
                try slabEval(g.vSlab)
                g.vSlabMaterialized = true
            }
        }
    }

    /// Allocate only enough native segments to honor currently admitted rows.
    /// Evaluate every candidate privately, then install all groups together.
    /// A failed allocation cannot leave newly committed backing or queue edits.
    func materializeReservedSegments() throws {
        precondition(config.segmentSizeBytes != nil)
        try materializeSegments(all: false)
    }

}
