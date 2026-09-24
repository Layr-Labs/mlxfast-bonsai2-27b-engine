// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: truthful admission with soft reservations.
//
// Submit-time check: could the request EVER fit (worst case promptLen +
// maxTokens, window-capped per layer) in total KV capacity? Step-time check:
// vLLM-style optimism — admit while `bytesReserved + nextStepNeed <
// capacity - watermark`; preemption is the backstop when optimism loses.

import Foundation
import MLX

// MARK: - Capacity oracle (scheduler ↔ admission)

/// Soft KV-capacity oracle consulted by `SchedulerV2.plan()` for every
/// assignment. Implemented by `AdmissionV2`; tests use scripted fakes.
/// (WS-B-internal — not part of the frozen contract; see
/// docs/engine-v2/CONTRACT-ISSUES-B-scheduler.md §6.)
public protocol CBv2StepCapacity: AnyObject {
    /// Reserve headroom for `additionalTokens` more KV entries of request
    /// `id`. Throws `CBv2KVError.capacityExhausted` when the soft ledger
    /// would cross the watermark — the scheduler preempts in response.
    func reserve(id: CBv2RequestID, additionalTokens: Int) throws
    /// Reserve token-derived bytes plus an exact native-dtype adjustment in
    /// one admission operation.
    func reserve(
        id: CBv2RequestID, additionalTokens: Int, additionalBytes: Int
    ) throws
    /// Partially undo a reservation (optimistic-advance rollback).
    func unreserve(id: CBv2RequestID, tokens: Int)
    func unreserve(id: CBv2RequestID, tokens: Int, bytes: Int)
    /// Drop every reservation held by `id` (finish/cancel/preempt).
    func releaseAll(id: CBv2RequestID)
    /// Non-throwing conservative probe used by the chained-decode fast path.
    func hasHeadroom(additionalTokens: Int) -> Bool
    /// The ledger's current byte ceiling (runtime-resizable). Feeds the
    /// engine's capacity snapshot so re-slices read back consistently.
    /// Defaulted to 0 (= unknown) so simple test capacities need not
    /// implement it.
    var bytesCapacity: Int { get }
}

extension CBv2StepCapacity {
    public var bytesCapacity: Int { 0 }
    public func reserve(
        id: CBv2RequestID, additionalTokens: Int, additionalBytes: Int
    ) throws {
        guard additionalBytes == 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "capacity oracle cannot reserve exact native KV bytes")
        }
        try reserve(id: id, additionalTokens: additionalTokens)
    }
    public func unreserve(id: CBv2RequestID, tokens: Int, bytes: Int) {
        precondition(bytes == 0, "capacity oracle cannot unreserve exact native KV bytes")
        unreserve(id: id, tokens: tokens)
    }
}

// MARK: - Backend storage policy

/// How many token ROWS of ONE layer's KV a backend actually occupies once a
/// sequence is bounded by `tokens` tokens. `AdmissionV2` multiplies these
/// rows by that layer's per-token byte cost, so the ledger charges what the
/// backend will really allocate instead of a figure inferred from the layer
/// kind alone.
///
/// The shipped policies differ exactly where the fixed-ring charge is right
/// and where it is wrong:
///
/// * CONTIGUOUS — `CBv2WindowedSequenceKV.allocateIfNeeded` allocates
///   `MLXArray.zeros([1, kvHeads, window, headDim])` on the FIRST write, so
///   a windowed layer occupies its whole ring from token one. Charging only
///   `min(tokens, window)` hid ~0.9 GB of overshoot at B=8 on Gemma-style
///   hybrids (PR#87).
/// * PAGED — `PagedKVPool.pageDemand` reserves
///   `min(ceil(tokens / pageSize), ringPageCount(window))` pages, so a short
///   row holds a handful of pages and NEVER the whole ring. Charging the
///   whole ring there rejects requests the pool can serve: a one-token
///   request needs ONE page, not 1,024 window rows (PR#87 review).
///
/// The backend answers for itself — admission never switches on a backend
/// enum, because the knowledge belongs where the allocation happens.
public protocol CBv2KVResidencyPolicy: Sendable {
    /// nil when the figure cannot be represented in `Int`; admission then
    /// fails cold instead of admitting on wrapped arithmetic.
    func residentRows(layer kind: CBv2LayerKind, tokens: Int) -> Int?
    /// Rows are handed out in multiples of this (1 for per-token contiguous
    /// arrays, `pageSize` for paged). The conservative headroom probe rounds
    /// its token count up to this so a decode step that crosses a page
    /// boundary is never counted as costing a single row.
    var rowGranularity: Int { get }
}

extension CBv2KVResidencyPolicy {
    public var rowGranularity: Int { 1 }
}

/// Per-sequence contiguous arrays (`CBv2ContiguousKVBackend`): full layers
/// grow with the sequence; windowed layers allocate their whole `window`-row
/// ring on the first write and never grow again.
///
/// This is also the DEFAULT for any backend that does not declare a policy,
/// deliberately: it is the CONSERVATIVE one. A backend that forgets to
/// implement `kvResidency` over-charges its windowed rows and under-admits,
/// which costs throughput — never the reverse, which costs the process.
public struct CBv2ContiguousKVResidency: CBv2KVResidencyPolicy {
    public init() {}

    public func residentRows(layer kind: CBv2LayerKind, tokens: Int) -> Int? {
        guard tokens >= 0 else { return nil }
        switch kind.attention {
        case .full: return tokens
        case .slidingWindow(let window): return max(0, window)
        }
    }
}

// MARK: - AdmissionV2

/// Byte-ledger admission controller. Thread-safe: `canEverFit` runs on the
/// submitter's thread while reserve/release run on the engine thread.
///
/// Bytes are estimated from `CBv2LayerKind`: layers that share KV storage
/// (`sharesKVWithLayer != nil`) own no bytes; sliding-window layers plateau
/// at `window` tokens — so a long-context request on Gemma-style hybrids is
/// charged truthfully, not as if every layer were full-attention.
///
/// The LEDGER charges `allocatedBytes(forTokens:)`, NOT
/// `estimatedBytes(forTokens:)`: occupancy, not retained content. What a
/// row OCCUPIES is a property of the BACKEND, so the conversion runs
/// through the `CBv2KVResidencyPolicy` the backend hands `EngineV2` — the
/// ledger never guesses from the layer kind alone.
///
/// * Contiguous rows allocate a fixed `window`-row ring on the FIRST write
///   (`CBv2WindowedSequenceKV.allocateIfNeeded`) instead of growing with
///   the sequence, so a 500-token request against a 1024 window occupies
///   the whole ring the moment it is touched. Charging only the retained
///   tokens left the gate believing it had margin it did not have (~0.9 GB
///   of hidden overshoot at B=8 on Gemma-style hybrids), and on unified
///   memory that surfaces as swap pressure, not a clean rejection (PR#87).
/// * Paged rows do NOT commit a ring: `PagedKVPool.pageDemand` caps the
///   reservation at `min(ceil(tokens / pageSize), ringPageCount)` pages.
///   Charging them the whole ring made `canEverFit` and the step ledger
///   reject short requests the pool can serve (PR#87 review).
public final class AdmissionV2: CBv2StepCapacity, @unchecked Sendable {
    public struct Config: Sendable {
        /// Fraction of capacity kept free as the optimism watermark.
        public var watermarkFraction: Double
        /// Default bytes per KV element (2 = fp16/bf16) — the fallback for
        /// layers not listed in `layerElementBytes`.
        public var elementBytes: Int
        /// OPTIONAL per-layer bytes-per-element (aligned to `layerKinds`),
        /// for models whose layers cache K/V at different precisions —
        /// e.g. GPT-OSS full-attention layers cache fp32 K/V (YarnRoPE
        /// computes in fp32) while sliding layers cache bf16. Assuming a
        /// flat 2 bytes/element there under-charges the fp32 rows ~2x and
        /// over-admits. Derive it from the model's probed cache dtypes
        /// (the compiled decode path probes per-layer dtypes at warmup —
        /// see `CBv2CompiledDecode`) via `Config.elementBytes(forDTypes:)`.
        /// nil ⇒ uniform `elementBytes`. Entries for KV-shared layers are
        /// ignored (those layers own no storage).
        public var layerElementBytes: [Int]?
        /// Fixed non-KV residency charged once for every active request.
        /// Hybrid recurrent models use this for conv + SSM state; attention-
        /// only models keep the source-compatible zero default.
        public var fixedBytesPerRequest: Int
        /// Variable request-owned residency outside target KV. Stateful MTP
        /// assistants use this for their independent autoregressive cache.
        public var auxiliaryBytesPerToken: Int
        /// Physical token-block allocation granularity of auxiliary state.
        /// `1` preserves linear per-token accounting.
        public var auxiliaryTokenGranularity: Int
        /// Retained speculative high-water state beyond committed tokens.
        public var auxiliaryTokenAllocationPadding: Int
        /// Immutable per-buffer allocator bounds resolved at engine creation.
        public var auxiliaryAllocationProjection: CBv2AuxiliaryAllocationProjection? = nil
        public init(
            watermarkFraction: Double = 0.05, elementBytes: Int = 2,
            layerElementBytes: [Int]? = nil,
            fixedBytesPerRequest: Int = 0
        ) {
            self.init(
                watermarkFraction: watermarkFraction,
                elementBytes: elementBytes,
                layerElementBytes: layerElementBytes,
                fixedBytesPerRequest: fixedBytesPerRequest,
                auxiliaryBytesPerToken: 0,
                auxiliaryTokenGranularity: 1,
                auxiliaryTokenAllocationPadding: 0)
        }

        public init(
            watermarkFraction: Double, elementBytes: Int,
            layerElementBytes: [Int]?, fixedBytesPerRequest: Int,
            auxiliaryBytesPerToken: Int,
            auxiliaryTokenGranularity: Int = 1,
            auxiliaryTokenAllocationPadding: Int = 0
        ) {
            self.watermarkFraction = watermarkFraction
            self.elementBytes = elementBytes
            self.layerElementBytes = layerElementBytes
            self.fixedBytesPerRequest = fixedBytesPerRequest
            self.auxiliaryBytesPerToken = auxiliaryBytesPerToken
            self.auxiliaryTokenGranularity = auxiliaryTokenGranularity
            self.auxiliaryTokenAllocationPadding = auxiliaryTokenAllocationPadding
        }

        /// Build a per-layer element-bytes table from probed cache dtypes
        /// (nil entries — KV-shared layers, unprobed — fall back to
        /// `defaultElementBytes`). Conservative: a wider dtype is charged
        /// its full element size.
        public static func elementBytes(
            forDTypes dtypes: [DType?], defaultElementBytes: Int = 2
        ) -> [Int] {
            dtypes.map { $0.map(\.size) ?? defaultElementBytes }
        }
    }

    private let lock = NSLock()
    private let layerKinds: [CBv2LayerKind]
    /// Resolved bytes-per-element per layer (aligned to `layerKinds`).
    private let perLayerElementBytes: [Int]
    /// Watermark fraction retained so `updateBytesCapacity` can recompute
    /// `watermark` against the new capacity.
    private let watermarkFraction: Double
    /// Lock-protected: recomputed whenever the capacity changes.
    private var watermark: Int
    /// Per-token bytes if every storage-owning layer retained the token
    /// (upper bound; used for the conservative headroom probe).
    private let maxPerTokenBytes: Int
    public let fixedBytesPerRequest: Int
    public let auxiliaryBytesPerToken: Int
    public let auxiliaryTokenGranularity: Int
    public let auxiliaryTokenAllocationPadding: Int
    private let auxiliaryAllocationProjection: CBv2AuxiliaryAllocationProjection?
    /// Nominal bytes per token for storage-owning full-attention rows under
    /// this ledger's actual per-layer dtype assumptions.
    public let fullKVBytesPerToken: Int

    /// Total KV byte budget. Runtime-resizable via `updateBytesCapacity`
    /// (multi-model co-residency re-slicing); reads take the ledger lock.
    public var bytesCapacity: Int {
        lock.lock()
        defer { lock.unlock() }
        return _bytesCapacity
    }
    private var _bytesCapacity: Int

    /// Bytes carved out of the budget for an EXTERNAL worst-case obligation
    /// the ledger cannot see per-request — today the compiled decode path's
    /// padding reserve. REFUNDABLE: if the obligation disappears (compiled
    /// decode disables itself at warmup after a trace failure), the engine
    /// calls `refundExternalReserve()` so admission is not permanently
    /// tighter than the hardware truth (PR#62 review). Lock-protected.
    private var externalReserveBytes: Int

    /// Backend storage policy: how many rows of each layer a row of this
    /// engine's backend really occupies. Immutable — a live engine never
    /// swaps backends.
    private let residency: any CBv2KVResidencyPolicy

    /// Cumulative reserved tokens per request. Under the CONTIGUOUS policy
    /// the token→byte conversion charges every windowed layer its whole
    /// fixed ring once, so decode reservations past a layer's window add
    /// zero bytes for that layer — and so do reservations below it, the
    /// ring being already paid. Under the PAGED policy the same conversion
    /// charges page-capped rows, so short rows stay cheap and the charge
    /// plateaus at the ring instead of starting there.
    private var reservedTokens: [CBv2RequestID: Int] = [:]
    private var reservedExactBytes: [CBv2RequestID: Int] = [:]
    private var ledgerBytes = 0
    private var transientBytes = 0
    /// Private, unscheduled rows already allocated by the backend. Their KV
    /// offsets the same physical floor as scheduled rows; their auxiliary
    /// state remains additive until the call retires all buffer owners.
    private var transientTargetBytes = 0
    private var checkpointStages: [UUID: CBv2CheckpointStageEntry] = [:]
    private var checkpointRequestOwners: [CBv2RequestID: UUID] = [:]
    /// Subset of reservedExactBytes belonging to imported recurrent/MTP state,
    /// not target KV. It follows the same request generation and retirement.
    private var checkpointAuxiliaryBytes: [CBv2RequestID: Int] = [:]
    private var detachedNonBackendBytes = 0
    private var physicalFloor = CBv2BackendPhysicalFloor()
    private let processMemoryOwner: (any CBv2ProcessMemoryOwner)?
    private var processChargedBytes = 0
    private var processMaterializedBytes = 0
    var hasProcessMemoryOwner: Bool { processMemoryOwner != nil }

    public init(
        layerKinds: [CBv2LayerKind], bytesCapacity: Int, config: Config = .init(),
        residency: any CBv2KVResidencyPolicy = CBv2ContiguousKVResidency(),
        externalReserveBytes: Int = 0,
        processMemoryOwner: (any CBv2ProcessMemoryOwner)? = nil
    ) {
        precondition(processMemoryOwner == nil || externalReserveBytes == 0,
                     "process-owned engines cannot start with an unreserved external carve")
        self.processMemoryOwner = processMemoryOwner
        self.residency = residency
        self.layerKinds = layerKinds
        self._bytesCapacity = bytesCapacity
        self.externalReserveBytes = max(0, externalReserveBytes)
        self.fixedBytesPerRequest = max(0, config.fixedBytesPerRequest)
        self.auxiliaryBytesPerToken = max(0, config.auxiliaryBytesPerToken)
        self.auxiliaryTokenGranularity = max(1, config.auxiliaryTokenGranularity)
        self.auxiliaryTokenAllocationPadding = max(
            0, config.auxiliaryTokenAllocationPadding)
        self.auxiliaryAllocationProjection = config.auxiliaryAllocationProjection
        if let table = config.layerElementBytes {
            precondition(
                table.count == layerKinds.count,
                "AdmissionV2: layerElementBytes count \(table.count) != layer count \(layerKinds.count)"
            )
            self.perLayerElementBytes = table
        } else {
            self.perLayerElementBytes = Array(
                repeating: config.elementBytes, count: layerKinds.count)
        }
        self.watermarkFraction = config.watermarkFraction
        self.watermark = Int(Double(bytesCapacity) * config.watermarkFraction)
        var perToken = 0
        var fullPerToken = 0
        var accountingOverflow = false
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            guard
                let bytes = Self.storageBytesPerToken(
                    kind: kind,
                    elementBytes: self.perLayerElementBytes[index]),
                let newPerToken = Self.add(perToken, bytes)
            else {
                accountingOverflow = true
                break
            }
            perToken = newPerToken
            if case .full = kind.attention {
                guard let newFullPerToken = Self.add(fullPerToken, bytes) else {
                    accountingOverflow = true
                    break
                }
                fullPerToken = newFullPerToken
            }
        }
        let (maximumAuxiliaryGrowth, auxiliaryGrowthOverflow) =
            self.auxiliaryBytesPerToken.multipliedReportingOverflow(
                by: self.auxiliaryTokenGranularity)
        let (totalPerToken, auxiliaryOverflow) = perToken.addingReportingOverflow(
            max(maximumAuxiliaryGrowth, auxiliaryAllocationProjection?.maximumGrowthBytes ?? 0))
        self.maxPerTokenBytes = accountingOverflow || auxiliaryOverflow
            || auxiliaryGrowthOverflow
            ? Int.max : totalPerToken
        self.fullKVBytesPerToken = accountingOverflow ? Int.max : fullPerToken
    }

    /// Engine construction binds its one immutable segmented backend before
    /// accepting requests. Other slots own independent AdmissionV2 instances.
    func bindBackendPhysicalFloor(initialBytes: Int) -> CBv2BackendPhysicalLease {
        precondition(initialBytes >= 0)
        precondition(!hasProcessMemoryOwner || initialBytes == 0,
                     "process-owned segmented pools must bind before allocation")
        lock.lock()
        precondition(!physicalFloor.isBound && ledgerBytes == 0 && reservedTokens.isEmpty,
                     "a physical floor must bind once before requests exist")
        physicalFloor.isBound = true
        physicalFloor.physicalBytes = initialBytes
        lock.unlock()
        return CBv2BackendPhysicalLease(bytes: initialBytes, resize: { [self] bytes in
            try resizeBackendPhysicalFloor(to: bytes)
        }, onClose: { [self] in
            lock.lock()
            physicalFloor.physicalBytes = 0
            publishProcessReductionLocked()
            lock.unlock()
        }, snapshot: { [self] in
            lock.withLock {
                CBv2BackendPhysicalAccounting(
                    nominalKVBytes: physicalFloor.nominalBytes,
                    physicalFloorOverheadBytes: physicalFloor.overheadBytes)
            }
        }, admissionIdentity: ObjectIdentifier(self))
    }

    private func resizeBackendPhysicalFloor(to bytes: Int) throws {
        guard bytes >= 0 else { throw CBv2KVError.backendIneligible(reason: "negative physical floor") }
        lock.lock()
        defer { lock.unlock() }
        guard let after = physicalFloor.chargedBytes(base: ledgerBytes, physical: bytes) else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        // Existing debt may be reused or reduced, but never increased.
        guard bytes <= physicalFloor.physicalBytes || after <= mutationCeiling else {
            throw CBv2KVError.capacityExhausted(
                needed: max(0, after - chargedLedgerBytes),
                available: max(0, reserveCeiling - chargedLedgerBytes))
        }
        try acceptProcessChargeLocked(after)
        physicalFloor.physicalBytes = bytes
    }

    /// Callers hold lock. Only nominal target KV offsets the physical floor;
    /// recurrent/MTP state, exact extra backing and scratch remain additive.
    private var chargedLedgerBytes: Int {
        physicalFloor.chargedBytes(base: ledgerBytes) ?? Int.max
    }
    private var mutationCeiling: Int {
        physicalFloor.isBound ? max(reserveCeiling, chargedLedgerBytes) : reserveCeiling
    }
    private func nominalTargetBytes(forTokens tokens: Int, allocated: Int? = nil) -> Int? {
        guard let total = allocated ?? allocatedBytesChecked(forTokens: tokens),
              let auxiliary = nonBackendBytesChecked(forTokens: tokens), total >= auxiliary else { return nil }
        return total - auxiliary
    }
    private func projectedNominal(
        from nominal: Int, oldTokens: Int, newTokens: Int,
        oldAllocated: Int? = nil, newAllocated: Int? = nil
    ) -> Int? {
        guard physicalFloor.isBound else { return 0 }
        guard let old = nominalTargetBytes(forTokens: oldTokens, allocated: oldAllocated),
              let new = nominalTargetBytes(forTokens: newTokens, allocated: newAllocated),
              let value = Self.add(nominal, new - old), value >= 0 else { return nil }
        return value
    }

    /// Called while holding Admission's lock, before local metadata commits.
    private func acceptProcessChargeLocked(_ charged: Int) throws {
        guard let processMemoryOwner else { return }
        guard let complete = Self.add(charged, externalReserveBytes), complete >= 0 else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        // A closing owner disappears at zero. Repeated empty pool retirement
        // must not submit a new transaction to that retired generation.
        if complete == processChargedBytes { return }
        do {
            try processMemoryOwner.replaceCharge(UInt64(complete))
            processChargedBytes = complete
        }
        catch {
            throw CBv2KVError.capacityExhausted(
                needed: max(0, charged - chargedLedgerBytes), available: 0)
        }
    }

    /// Callers have withdrawn coverage and dropped any retiring buffer aliases.
    private func publishProcessReductionLocked() {
        do { try acceptProcessChargeLocked(chargedLedgerBytes) }
        catch { preconditionFailure("process charge retirement refused: \(error)") }
    }

    /// Only evaluated, exclusively owned native allocations may call this.
    /// A handle is shared when the same backing changes ownership metadata.
    func coverEvaluatedAllocation(bytes: Int) throws -> CBv2MemoryCoverage? {
        guard let processMemoryOwner else { return nil }
        return try lock.withLock {
            guard bytes >= 0, let next = Self.add(processMaterializedBytes, bytes),
                next <= chargedLedgerBytes
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            try processMemoryOwner.recordMaterialization(UInt64(next))
            processMaterializedBytes = next
            return CBv2MemoryCoverage { [self] in
                lock.withLock {
                    precondition(bytes <= processMaterializedBytes)
                    do { try processMemoryOwner.withdrawCoverage(UInt64(bytes)) }
                    catch { preconditionFailure("process coverage retirement refused: \(error)") }
                    processMaterializedBytes -= bytes
                }
            }
        }
    }

    func closeProcessMemoryOwner() { processMemoryOwner?.retire() }

    deinit { processMemoryOwner?.retire() }

    // MARK: Estimation

    /// KV bytes whose CONTENT is retained after processing `tokens` tokens
    /// of one sequence (windowed layers cap at their window). This is NOT
    /// what a sequence occupies — see `allocatedBytes(forTokens:)`, the
    /// figure the ledger charges.
    public func estimatedBytes(forTokens tokens: Int) -> Int {
        estimatedBytesChecked(forTokens: tokens) ?? Int.max
    }

    private func estimatedBytesChecked(forTokens tokens: Int) -> Int? {
        guard tokens > 0 else { return 0 }
        guard var total = nonBackendBytesChecked(forTokens: tokens) else { return nil }
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            let retained: Int
            switch kind.attention {
            case .full: retained = tokens
            case .slidingWindow(let window): retained = min(tokens, window)
            }
            guard retained >= 0,
                let perTokenBytes = Self.storageBytesPerToken(
                    kind: kind,
                    elementBytes: perLayerElementBytes[index]),
                let retainedBytes = Self.multiply(retained, perTokenBytes),
                let newTotal = Self.add(total, retainedBytes)
            else { return nil }
            total = newTotal
        }
        return total
    }

    private func nonBackendBytesChecked(forTokens tokens: Int) -> Int? {
        guard tokens > 0 else { return 0 }
        guard let paddedTokens = Self.add(tokens, auxiliaryTokenAllocationPadding),
            let auxiliaryTokens = Self.roundUp(paddedTokens, to: auxiliaryTokenGranularity),
            let auxiliary = Self.multiply(auxiliaryTokens, auxiliaryBytesPerToken)
        else { return nil }
        let physicalAuxiliary: Int
        if let auxiliaryAllocationProjection {
            guard let bytes = auxiliaryAllocationProjection.bytes(forTokens: tokens) else { return nil }
            physicalAuxiliary = max(auxiliary, bytes)
        } else { physicalAuxiliary = auxiliary }
        return Self.add(fixedBytesPerRequest, physicalAuxiliary)
    }

    /// Bytes the BACKEND occupies at `tokens` beyond the content
    /// `estimatedBytes(forTokens:)` retains. `allocatedBytesChecked` folds
    /// this into every ledger charge, which is why no caller adds it a
    /// second time. nil on accounting overflow.
    ///
    /// Under the contiguous policy this is exactly the still-unfilled
    /// remainder of every fixed sliding ring — the windowed rows allocate
    /// the whole ring on first write, so the gap is real occupancy from the
    /// first token onward, and the name is literal. Under the paged policy
    /// there is no fixed ring to fill: the gap is the page-granularity
    /// remainder of `PagedKVPool.pageDemand`, which for a short row is a
    /// page or two rather than a whole window.
    public func fixedWindowBytesShortfall(afterReservingTokens tokens: Int) -> Int? {
        var total = 0
        let held = max(0, tokens)
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            let retained: Int
            switch kind.attention {
            case .full: retained = held
            case .slidingWindow(let window): retained = max(0, min(held, window))
            }
            guard let occupied = residency.residentRows(layer: kind, tokens: held),
                occupied >= 0
            else { return nil }
            let missing = max(0, occupied - retained)
            guard
                let perTokenBytes = Self.storageBytesPerToken(
                    kind: kind,
                    elementBytes: perLayerElementBytes[index]),
                let missingBytes = Self.multiply(missing, perTokenBytes),
                let newTotal = Self.add(total, missingBytes)
            else { return nil }
            total = newTotal
        }
        return total
    }

    /// KV bytes a sequence OCCUPIES once it has processed `tokens` tokens:
    /// retained content plus whatever the backend holds on top of it. This
    /// is what the ledger charges and what the backend actually allocates —
    /// a contiguous windowed layer costs its whole `window`-row ring from
    /// the first token, while the same layer on the paged backend costs
    /// `min(ceil(tokens / pageSize), ringPageCount)` pages.
    public func allocatedBytes(forTokens tokens: Int) -> Int {
        allocatedBytesChecked(forTokens: tokens) ?? Int.max
    }

    private func allocatedBytesChecked(forTokens tokens: Int) -> Int? {
        guard tokens > 0 else { return 0 }
        guard let retained = estimatedBytesChecked(forTokens: tokens),
            let ringShortfall = fixedWindowBytesShortfall(afterReservingTokens: tokens),
            let total = Self.add(retained, ringShortfall)
        else { return nil }
        return total
    }

    /// Bytes a single request may ever reserve: `reserve` enforces
    /// `capacity - externalReserve - watermark`, so feasibility must be
    /// judged against the same ceiling. (Judging against full capacity
    /// admitted requests in `(ceiling, capacity]` that could NEVER reserve
    /// their last tokens — they hit the wall, self-preempted, restarted,
    /// and livelocked until their deadline.)
    public var admissibleBytesCapacity: Int {
        lock.lock()
        defer { lock.unlock() }
        return _bytesCapacity - externalReserveBytes - watermark
    }

    /// The ceiling every reservation is checked against. Callers hold `lock`.
    private var reserveCeiling: Int { _bytesCapacity - externalReserveBytes - watermark }

    /// Runtime capacity update (multi-model co-residency re-slicing: the
    /// provider shrinks resident engines to fair shares before granting a
    /// newcomer, and grows survivors back on unload). The watermark is
    /// recomputed from the configured fraction against the new capacity.
    ///
    /// Shrink semantics are safe by construction: reservations already in
    /// the ledger are untouched (nothing is evicted here — preemption
    /// remains the scheduler's job). A segmented pool may reuse already
    /// charged storage without increasing its debt; reservations requiring
    /// additional bytes fail until the charge fits the new ceiling.
    /// Grow takes effect immediately.
    public func updateBytesCapacity(_ bytes: Int) {
        lock.lock()
        _bytesCapacity = max(0, bytes)
        watermark = Int(Double(_bytesCapacity) * watermarkFraction)
        lock.unlock()
    }

    /// Release the external carve-out (idempotent). Called by the engine
    /// when the obligation it covered no longer exists — compiled decode
    /// disabled itself at warmup, so its padding can never materialize and
    /// the bytes belong to regular admission again (PR#62 review).
    /// Live external (compiled padding) reserve — the construction value
    /// until `refundExternalReserve()`, then 0. Read by the engine loop's
    /// gauge publish so the snapshot's `kvBytesReserved` carries the FULL
    /// not-available-for-new-admissions figure (backend promises + this
    /// carve), keeping "capacity − reserved" truthful for planners.
    public var bytesExternallyReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return externalReserveBytes
    }

    public func refundExternalReserve() {
        lock.lock()
        externalReserveBytes = 0
        publishProcessReductionLocked()
        lock.unlock()
    }

    /// Truthful submit-time check: worst case (promptLen + maxTokens) vs the
    /// watermark-adjusted capacity (`admissibleBytesCapacity` — the most
    /// `reserve` will ever grant). Requests that could never fit are
    /// rejected up front; requests that fit only sometimes are admitted
    /// optimistically and preempted if optimism loses. Judged on OCCUPANCY
    /// (fixed sliding rings included), so the gate agrees with both what
    /// `reserve` charges and what the backend allocates.
    public func canEverFit(
        promptTokens: Int, maxTokens: Int, additionalBackendBytes: Int = 0
    ) -> Bool {
        let (tokens, overflow) = promptTokens.addingReportingOverflow(max(maxTokens, 0))
        guard !overflow, additionalBackendBytes >= 0,
            let allocated = allocatedBytesChecked(forTokens: tokens),
            let bytes = Self.add(allocated, additionalBackendBytes) else {
            return false
        }
        return bytes <= admissibleBytesCapacity
    }

    /// Non-mutating replay of the exact reserve/unreserve/release timeline a
    /// deadline projection expects before the target's first sample. The
    /// snapshot is taken under the same lock as `reserve`, so paused rows and
    /// unrelated live ledger owners remain charged. Speculative suffixes are
    /// unreserved only after their full step-width peak has been checked, and
    /// a row is released only after the projected step that retires it.
    func canGuarantee(
        projectedOperations: [CBv2ProjectedCapacityOperation],
        projectedPhysicalBytes: (([CBv2RequestID: Int]) -> Int?)? = nil
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        var simulatedTokens = reservedTokens
        var simulatedExactBytes = reservedExactBytes
        var simulatedLedgerBytes = ledgerBytes
        var simulatedNominalBytes = physicalFloor.nominalBytes
        var simulatedPhysicalBytes = physicalFloor.physicalBytes

        /// nil means malformed/unrepresentable; false means a valid
        /// reservation that cannot fit and was left unapplied.
        func applyReservation(
            _ reservation: CBv2ProjectedCapacityReservation
        ) -> Bool? {
            guard reservation.additionalTokens >= 0,
                reservation.additionalBytes >= 0
            else {
                return nil
            }
            let oldTokens = simulatedTokens[reservation.id] ?? 0
            let oldExactBytes = simulatedExactBytes[reservation.id] ?? 0
            let (newTokens, tokenOverflow) = oldTokens.addingReportingOverflow(
                reservation.additionalTokens)
            let (newExactBytes, exactOverflow) = oldExactBytes.addingReportingOverflow(
                reservation.additionalBytes)
            guard !tokenOverflow, !exactOverflow,
                let oldTokenBytes = allocatedBytesChecked(forTokens: oldTokens),
                let newTokenBytes = allocatedBytesChecked(forTokens: newTokens)
            else {
                return nil
            }
            let (tokenDelta, tokenDeltaOverflow) =
                newTokenBytes.subtractingReportingOverflow(oldTokenBytes)
            let (delta, deltaOverflow) = tokenDelta.addingReportingOverflow(
                reservation.additionalBytes)
            let (after, ledgerOverflow) =
                simulatedLedgerBytes.addingReportingOverflow(delta)
            guard !tokenDeltaOverflow, !deltaOverflow, !ledgerOverflow else {
                return nil
            }
            var nextTokens = simulatedTokens
            nextTokens[reservation.id] = newTokens
            let physical: Int
            if let projectedPhysicalBytes {
                guard let estimated = projectedPhysicalBytes(nextTokens), estimated >= 0 else { return false }
                // A projected release does not promise that free native
                // segments or asynchronous export owners have retired yet.
                physical = max(simulatedPhysicalBytes, estimated)
            } else { physical = simulatedPhysicalBytes }
            guard let nominal = projectedNominal(
                from: simulatedNominalBytes, oldTokens: oldTokens, newTokens: newTokens,
                oldAllocated: oldTokenBytes, newAllocated: newTokenBytes),
                let charged = physicalFloor.chargedBytes(base: after, nominal: nominal, physical: physical),
                let before = physicalFloor.chargedBytes(
                    base: simulatedLedgerBytes, nominal: simulatedNominalBytes,
                    physical: simulatedPhysicalBytes)
            else { return nil }
            let ceiling = physicalFloor.isBound ? max(reserveCeiling, before) : reserveCeiling
            guard charged <= ceiling else { return false }
            simulatedNominalBytes = nominal
            simulatedPhysicalBytes = physical
            simulatedTokens = nextTokens
            simulatedExactBytes[reservation.id] = newExactBytes
            simulatedLedgerBytes = after
            return true
        }

        for operation in projectedOperations {
            switch operation {
            case .reserve(let reservation):
                guard applyReservation(reservation) == true else { return false }

            case .reserveIfAvailable(let reservation):
                // Chained decode probes headroom before planning. Capacity
                // refusal skips that optional chain, so a valid-but-full
                // simulation continues without mutating the ledger.
                guard applyReservation(reservation) != nil else { return false }

            case .unreserve(let reservation):
                guard reservation.additionalTokens >= 0,
                    reservation.additionalBytes >= 0
                else {
                    return false
                }
                let oldTokens = simulatedTokens[reservation.id] ?? 0
                let oldExactBytes = simulatedExactBytes[reservation.id] ?? 0
                let newTokens = max(0, oldTokens - reservation.additionalTokens)
                let newExactBytes = max(
                    0, oldExactBytes - reservation.additionalBytes)
                guard let oldTokenBytes = allocatedBytesChecked(forTokens: oldTokens),
                    let newTokenBytes = allocatedBytesChecked(forTokens: newTokens)
                else {
                    return false
                }
                let (releasedTokenBytes, tokenOverflow) =
                    oldTokenBytes.subtractingReportingOverflow(newTokenBytes)
                let (releasedExactBytes, exactOverflow) =
                    oldExactBytes.subtractingReportingOverflow(newExactBytes)
                let (releasedBytes, releaseOverflow) =
                    releasedTokenBytes.addingReportingOverflow(releasedExactBytes)
                let (after, ledgerOverflow) =
                    simulatedLedgerBytes.subtractingReportingOverflow(releasedBytes)
                guard !tokenOverflow, !exactOverflow, !releaseOverflow,
                    !ledgerOverflow, after >= 0,
                    let nominal = projectedNominal(
                        from: simulatedNominalBytes, oldTokens: oldTokens, newTokens: newTokens,
                        oldAllocated: oldTokenBytes, newAllocated: newTokenBytes)
                else {
                    return false
                }
                simulatedNominalBytes = nominal
                if newTokens == 0 {
                    simulatedTokens.removeValue(forKey: reservation.id)
                } else {
                    simulatedTokens[reservation.id] = newTokens
                }
                if newExactBytes == 0 {
                    simulatedExactBytes.removeValue(forKey: reservation.id)
                } else {
                    simulatedExactBytes[reservation.id] = newExactBytes
                }
                simulatedLedgerBytes = after

            case .release(let id):
                let oldTokens = simulatedTokens.removeValue(forKey: id) ?? 0
                let oldExactBytes =
                    simulatedExactBytes.removeValue(forKey: id) ?? 0
                guard let tokenBytes = allocatedBytesChecked(forTokens: oldTokens)
                else {
                    return false
                }
                let (releasedBytes, releaseOverflow) =
                    tokenBytes.addingReportingOverflow(oldExactBytes)
                let (after, ledgerOverflow) =
                    simulatedLedgerBytes.subtractingReportingOverflow(releasedBytes)
                guard !releaseOverflow, !ledgerOverflow, after >= 0,
                    let nominal = projectedNominal(
                        from: simulatedNominalBytes, oldTokens: oldTokens, newTokens: 0,
                        oldAllocated: tokenBytes, newAllocated: 0)
                else { return false }
                simulatedNominalBytes = nominal
                simulatedLedgerBytes = after
            }
        }
        return true
    }

    // MARK: CBv2StepCapacity

    public func reserve(id: CBv2RequestID, additionalTokens: Int) throws {
        try reserve(id: id, additionalTokens: additionalTokens, additionalBytes: 0)
    }

    public func reserve(
        id: CBv2RequestID, additionalTokens: Int, additionalBytes: Int
    ) throws {
        guard additionalTokens >= 0, additionalBytes >= 0 else {
            throw CBv2KVError.backendIneligible(reason: "negative KV reservation")
        }
        lock.lock()
        defer { lock.unlock() }
        let old = reservedTokens[id] ?? 0
        let (new, tokenCountOverflow) = old.addingReportingOverflow(additionalTokens)
        let oldExact = reservedExactBytes[id] ?? 0
        let (newExact, exactOverflow) = oldExact.addingReportingOverflow(additionalBytes)
        guard !tokenCountOverflow, !exactOverflow,
            let oldTokenBytes = allocatedBytesChecked(forTokens: old),
            let newTokenBytes = allocatedBytesChecked(forTokens: new)
        else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        let (tokenDelta, tokenOverflow) = newTokenBytes.subtractingReportingOverflow(oldTokenBytes)
        let (delta, deltaOverflow) = tokenDelta.addingReportingOverflow(additionalBytes)
        guard !tokenOverflow, !deltaOverflow else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        let (after, afterOverflow) = ledgerBytes.addingReportingOverflow(delta)
        guard !afterOverflow else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        guard let nominal = projectedNominal(
            from: physicalFloor.nominalBytes, oldTokens: old, newTokens: new,
            oldAllocated: oldTokenBytes, newAllocated: newTokenBytes),
              let charged = physicalFloor.chargedBytes(base: after, nominal: nominal) else {
            throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0)
        }
        guard charged <= mutationCeiling else {
            throw CBv2KVError.capacityExhausted(
                needed: max(0, charged - chargedLedgerBytes),
                available: max(0, reserveCeiling - chargedLedgerBytes))
        }
        try acceptProcessChargeLocked(charged)
        physicalFloor.nominalBytes = nominal
        reservedTokens[id] = new
        reservedExactBytes[id] = newExact
        ledgerBytes = after
    }

    public func unreserve(id: CBv2RequestID, tokens: Int) {
        unreserve(id: id, tokens: tokens, bytes: 0)
    }

    public func unreserve(id: CBv2RequestID, tokens: Int, bytes: Int) {
        precondition(tokens >= 0 && bytes >= 0)
        lock.lock()
        defer { lock.unlock() }
        let old = reservedTokens[id] ?? 0
        let new = max(0, old - tokens)
        let oldExact = reservedExactBytes[id] ?? 0
        let newExact = max(0, oldExact - bytes)
        if let auxiliary = checkpointAuxiliaryBytes[id] {
            let remaining = min(auxiliary, newExact)
            checkpointAuxiliaryBytes[id] = remaining > 0 ? remaining : nil
        }
        let oldAllocated = allocatedBytes(forTokens: old)
        let newAllocated = allocatedBytes(forTokens: new)
        guard let nominal = projectedNominal(
            from: physicalFloor.nominalBytes, oldTokens: old, newTokens: new,
            oldAllocated: oldAllocated, newAllocated: newAllocated) else {
            preconditionFailure("invalid nominal KV release")
        }
        physicalFloor.nominalBytes = nominal
        ledgerBytes += newAllocated - oldAllocated + newExact - oldExact
        publishProcessReductionLocked()
        if new == 0 {
            reservedTokens.removeValue(forKey: id)
            checkpointRequestOwners.removeValue(forKey: id)
        } else {
            reservedTokens[id] = new
        }
        if newExact == 0 {
            reservedExactBytes.removeValue(forKey: id)
        } else {
            reservedExactBytes[id] = newExact
        }
    }

    public func releaseAll(id: CBv2RequestID) {
        lock.withLock { releaseAllLocked(id: id) }
    }

    /// Only the current imported request generation may release this active
    /// charge. detachReservation removes its marker before an ID can be reused.
    func releaseCheckpointRequest(id: CBv2RequestID, ownerIdentity: UUID) {
        lock.withLock {
            guard checkpointRequestOwners[id] == ownerIdentity else { return }
            releaseAllLocked(id: id)
        }
    }

    private func releaseAllLocked(id: CBv2RequestID) {
        checkpointRequestOwners.removeValue(forKey: id)
        checkpointAuxiliaryBytes.removeValue(forKey: id)
        let old = reservedTokens.removeValue(forKey: id) ?? 0
        let exact = reservedExactBytes.removeValue(forKey: id) ?? 0
        guard let nominal = projectedNominal(
            from: physicalFloor.nominalBytes, oldTokens: old, newTokens: 0) else {
            preconditionFailure("invalid nominal KV completion")
        }
        physicalFloor.nominalBytes = nominal
        ledgerBytes -= allocatedBytes(forTokens: old) + exact
        publishProcessReductionLocked()
    }

    /// Short-lived cache transfer buffers share the ordinary admission ceiling
    /// but never masquerade as scheduled requests or reusable sampling IDs.
    func reserveTransient(bytes: Int) throws -> CBv2CheckpointReservation {
        guard bytes >= 0 else { throw CBv2CompleteCheckpointError.invalidManifest }
        lock.lock()
        let (after, overflow) = ledgerBytes.addingReportingOverflow(bytes)
        guard !overflow, let charged = physicalFloor.chargedBytes(base: after),
              charged <= mutationCeiling else {
            let available = max(0, reserveCeiling - chargedLedgerBytes)
            lock.unlock()
            throw CBv2KVError.capacityExhausted(needed: bytes, available: available)
        }
        do { try acceptProcessChargeLocked(charged) }
        catch { lock.unlock(); throw error }
        ledgerBytes = after
        transientBytes += bytes
        lock.unlock()
        return CBv2CheckpointReservation { [self] in
            lock.lock()
            ledgerBytes -= bytes
            transientBytes -= bytes
            publishProcessReductionLocked()
            lock.unlock()
        }
    }

    /// Price a private teacher row with the ordinary request projection,
    /// including the resolved recurrent peak, allocator padding and any
    /// configured auxiliary headroom. The backend row must already exist;
    /// no forward may run until this reservation succeeds. The caller drops
    /// every row/state alias before releasing the returned lease.
    func reserveUnscheduledRequest(maximumTokens: Int, minimumTargetBytes: Int)
        throws -> CBv2CheckpointReservation
    {
        guard maximumTokens > 0, minimumTargetBytes >= 0,
            let allocated = allocatedBytesChecked(forTokens: maximumTokens),
            let auxiliary = nonBackendBytesChecked(forTokens: maximumTokens),
            allocated >= auxiliary,
            let bytes = Self.add(max(allocated - auxiliary, minimumTargetBytes), auxiliary)
        else { throw CBv2KVError.capacityExhausted(needed: Int.max, available: 0) }
        let target = bytes - auxiliary
        let nominalDelta = try lock.withLock {
            let delta = physicalFloor.isBound ? target : 0
            guard let after = Self.add(ledgerBytes, bytes),
                let nominal = Self.add(physicalFloor.nominalBytes, delta),
                let charged = physicalFloor.chargedBytes(base: after, nominal: nominal),
                charged <= mutationCeiling
            else {
                throw CBv2KVError.capacityExhausted(
                    needed: bytes, available: max(0, reserveCeiling - chargedLedgerBytes))
            }
            try acceptProcessChargeLocked(charged)
            physicalFloor.nominalBytes = nominal
            ledgerBytes = after
            transientBytes += bytes
            transientTargetBytes += target
            return delta
        }
        return CBv2CheckpointReservation { [self] in
            lock.withLock {
                ledgerBytes -= bytes
                transientBytes -= bytes
                transientTargetBytes -= target
                physicalFloor.nominalBytes -= nominalDelta
                publishProcessReductionLocked()
            }
        }
    }

    /// Engine-created staging identities are the only source of a same-buffer
    /// destination transfer. Generic transient release callbacks cannot credit it.
    func reserveCheckpointStage(targetBytes: Int, auxiliaryBytes: Int, scratchBytes: Int)
        throws -> CBv2CheckpointStageLease
    {
        guard targetBytes >= 0, auxiliaryBytes >= 0, scratchBytes >= 0,
            let destination = Self.add(targetBytes, auxiliaryBytes),
            let bytes = Self.add(destination, scratchBytes)
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        let identity = UUID()
        try lock.withLock {
            guard let after = Self.add(ledgerBytes, bytes),
                let charged = physicalFloor.chargedBytes(base: after),
                charged <= mutationCeiling
            else {
                throw CBv2KVError.capacityExhausted(
                    needed: bytes, available: max(0, reserveCeiling - chargedLedgerBytes))
            }
            try acceptProcessChargeLocked(charged)
            ledgerBytes = after
            transientBytes += bytes
            checkpointStages[identity] = .init(bytes: bytes, transferred: false)
        }
        return CBv2CheckpointStageLease(
            admission: self, identity: identity, targetBytes: targetBytes,
            auxiliaryBytes: auxiliaryBytes, scratchBytes: scratchBytes)
    }

    /// Typed private destination settlement, after all evaluated allocations
    /// succeeded. This removes only unused allowance, never backing or scratch.
    func settleCheckpointStage(identity: UUID, expectedBytes: Int, reduction: Int) throws {
        try lock.withLock {
            guard var entry = checkpointStages[identity], !entry.transferred, !entry.settled,
                entry.bytes == expectedBytes, reduction >= 0, reduction <= entry.bytes
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            entry.bytes -= reduction
            entry.settled = true
            checkpointStages[identity] = entry
            ledgerBytes -= reduction
            transientBytes -= reduction
            publishProcessReductionLocked()
        }
    }

    func closeCheckpointStage(identity: UUID) {
        lock.withLock {
            let remaining = checkpointStages.removeValue(forKey: identity)?.bytes ?? 0
            ledgerBytes -= remaining
            transientBytes -= remaining
            publishProcessReductionLocked()
        }
    }

    /// Called under the pool's physical-lease lock, before allocating missing
    /// suffix backing. The stage destination becomes ordinary request/floor
    /// ownership in one Admission mutation; only its scratch stays transient.
    /// The caller owns the payload exclusively and must discard it on failure.
    func transferCheckpointStage(
        _ stage: CBv2CheckpointStageLease, requestID: CBv2RequestID,
        maximumTokens: Int, previousPhysicalBytes: Int, physicalBytes: Int
    ) throws -> CBv2CheckpointAdoptionReservation {
        // The stage lock precedes Admission; never acquire it while holding
        // this ledger lock. Settlement happens before the frame is exposed.
        let destination = stage.destination
        let targetBytes = destination.targetBytes
        let destinationBytes = destination.bytes
        let totalBytes = destinationBytes + stage.scratchBytes
        let rollback = try lock.withLock {
            guard stage.admission === self, physicalFloor.isBound,
                physicalFloor.physicalBytes == previousPhysicalBytes,
                physicalBytes >= previousPhysicalBytes,
                physicalBytes - previousPhysicalBytes >= targetBytes,
                maximumTokens > 0, reservedTokens[requestID] == nil,
                reservedExactBytes[requestID] == nil,
                checkpointStages[stage.identity]?.transferred == false,
                checkpointStages[stage.identity]?.bytes == totalBytes,
                let allocated = allocatedBytesChecked(forTokens: maximumTokens),
                let auxiliary = nonBackendBytesChecked(forTokens: maximumTokens),
                let allocatedWithShortfall = Self.add(
                    allocated, max(0, destination.auxiliaryBytes - auxiliary)),
                let nominal = projectedNominal(
                    from: physicalFloor.nominalBytes, oldTokens: 0,
                    newTokens: maximumTokens, newAllocated: allocated),
                let after = Self.add(ledgerBytes - destinationBytes, allocatedWithShortfall),
                let charged = physicalFloor.chargedBytes(
                    base: after, nominal: nominal, physical: physicalBytes)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            guard charged <= mutationCeiling else {
                throw CBv2KVError.capacityExhausted(
                    needed: max(0, charged - chargedLedgerBytes),
                    available: max(0, reserveCeiling - chargedLedgerBytes))
            }
            try acceptProcessChargeLocked(charged)
            let previousNominal = physicalFloor.nominalBytes
            ledgerBytes = after
            transientBytes -= destinationBytes
            checkpointStages[stage.identity] = .init(bytes: stage.scratchBytes, transferred: true)
            reservedTokens[requestID] = maximumTokens
            let auxiliaryShortfall = allocatedWithShortfall - allocated
            if auxiliaryShortfall > 0 {
                reservedExactBytes[requestID] = auxiliaryShortfall
                checkpointAuxiliaryBytes[requestID] = auxiliaryShortfall
            }
            checkpointRequestOwners[requestID] = stage.identity
            physicalFloor.nominalBytes = nominal
            physicalFloor.physicalBytes = physicalBytes
            return (allocated: allocatedWithShortfall, auxiliaryShortfall: auxiliaryShortfall,
                    nominalDelta: nominal - previousNominal)
        }
        return CBv2CheckpointAdoptionReservation { [self] in
            lock.withLock {
                // Preparation is engine-queue exclusive and has not exposed
                // this request yet. The ticket cannot outlive publication.
                precondition(reservedTokens[requestID] == maximumTokens)
                precondition(checkpointRequestOwners[requestID] == stage.identity)
                precondition((reservedExactBytes[requestID] ?? 0) == rollback.auxiliaryShortfall)
                precondition(physicalFloor.physicalBytes == physicalBytes)
                reservedTokens.removeValue(forKey: requestID)
                reservedExactBytes.removeValue(forKey: requestID)
                checkpointAuxiliaryBytes.removeValue(forKey: requestID)
                checkpointRequestOwners.removeValue(forKey: requestID)
                ledgerBytes -= rollback.allocated
                physicalFloor.nominalBytes -= rollback.nominalDelta
                physicalFloor.physicalBytes = previousPhysicalBytes
                publishProcessReductionLocked()
            }
        }
    }

    /// Finish scheduling without freeing buffers still owned by asynchronous
    /// retirement. The detached lease can never release a reused request ID.
    func detachReservation(id: CBv2RequestID) -> CBv2CheckpointReservation {
        lock.lock()
        checkpointRequestOwners.removeValue(forKey: id)
        let tokens = reservedTokens.removeValue(forKey: id) ?? 0
        let exact = reservedExactBytes.removeValue(forKey: id) ?? 0
        let bytes = allocatedBytes(forTokens: tokens) + exact
        let auxiliary = checkpointAuxiliaryBytes.removeValue(forKey: id) ?? 0
        let nonBackendBytes = (nonBackendBytesChecked(forTokens: tokens) ?? 0) + auxiliary
        let nominalBytes = physicalFloor.isBound
            ? (nominalTargetBytes(forTokens: tokens) ?? 0) : 0
        detachedNonBackendBytes += nonBackendBytes
        lock.unlock()
        return CBv2CheckpointReservation { [self] in
            lock.lock()
            ledgerBytes -= bytes
            physicalFloor.nominalBytes -= nominalBytes
            detachedNonBackendBytes -= nonBackendBytes
            publishProcessReductionLocked()
            lock.unlock()
        }
    }

    /// Conservative (window caps ignored): may under-report headroom, which
    /// only breaks a decode chain — never over-admits. The token count is
    /// first rounded UP to the backend's row granularity, so on a paged
    /// backend a single decode token that crosses a page boundary is priced
    /// as the whole page it can cost, not as one row.
    public func hasHeadroom(additionalTokens: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard additionalTokens >= 0,
            let chargedTokens = Self.roundUp(additionalTokens, to: residency.rowGranularity),
            let additionalBytes = Self.multiply(chargedTokens, maxPerTokenBytes),
            let after = Self.add(chargedLedgerBytes, additionalBytes)
        else { return false }
        return after <= mutationCeiling
    }

    /// Smallest multiple of `granularity` at or above `value`; nil on
    /// overflow. A non-positive granularity means "no rounding".
    private static func roundUp(_ value: Int, to granularity: Int) -> Int? {
        guard granularity > 1 else { return value }
        guard let bumped = add(value, granularity - 1) else { return nil }
        return (bumped / granularity) * granularity
    }

    private static func storageBytesPerToken(
        kind: CBv2LayerKind,
        elementBytes: Int
    ) -> Int? {
        guard kind.kvHeads >= 0, kind.headDim >= 0, elementBytes >= 0,
            let elements = multiply(kind.kvHeads, kind.headDim),
            let kvElements = multiply(elements, 2)
        else { return nil }
        return multiply(kvElements, elementBytes)
    }

    private static func multiply(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : value
    }

    private static func add(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : value
    }

    // MARK: Telemetry

    /// Ledger bytes currently reserved (soft truth; the backend's
    /// `bytesInUse` is the hard truth once arrays materialize).
    public var bytesReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return chargedLedgerBytes
    }

    public var transientBytesReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return transientBytes
    }

    /// Live obligations the KV backend does not own: fixed recurrent state,
    /// block-rounded assistant state, and the external compiled-decode carve.
    /// Gauge publication adds this once after reconciling backend target-KV
    /// promises with the materialized and waiting target ledger partitions.
    public var nonBackendBytesReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        guard let transfers = Self.add(transientBytes - transientTargetBytes, detachedNonBackendBytes),
            let withExternal = Self.add(externalReserveBytes, transfers),
            var total = Self.add(withExternal, physicalFloor.overheadBytes)
        else { return Int.max }
        for tokens in reservedTokens.values where tokens > 0 {
            guard let bytes = nonBackendBytesChecked(forTokens: tokens),
                let next = Self.add(total, bytes)
            else { return Int.max }
            total = next
        }
        for auxiliary in checkpointAuxiliaryBytes.values {
            guard let next = Self.add(total, auxiliary) else { return Int.max }
            total = next
        }
        return total
    }

    /// Target-KV ledger split by whether the engine has already materialized
    /// that request's backend rows. Materialized target charges overlap the
    /// backend reservation; unmaterialized charges do not and remain additive.
    public func targetBytesReserved(
        partitionedBy materializedIDs: Set<CBv2RequestID>
    ) -> (materialized: Int, unmaterialized: Int) {
        lock.lock()
        defer { lock.unlock() }
        var materialized = transientTargetBytes
        var unmaterialized = 0
        let ids = Set(reservedTokens.keys).union(reservedExactBytes.keys)
        for id in ids {
            let tokens = reservedTokens[id] ?? 0
            guard let allocated = allocatedBytesChecked(forTokens: tokens),
                let nonBackend = nonBackendBytesChecked(forTokens: tokens)
            else { return (Int.max, Int.max) }
            let (target, subtractionOverflow) = allocated.subtractingReportingOverflow(nonBackend)
            let (withExact, exactOverflow) = target.addingReportingOverflow(
                (reservedExactBytes[id] ?? 0) - (checkpointAuxiliaryBytes[id] ?? 0))
            guard !subtractionOverflow, !exactOverflow else { return (Int.max, Int.max) }
            if materializedIDs.contains(id) {
                guard let next = Self.add(materialized, withExact) else {
                    return (Int.max, Int.max)
                }
                materialized = next
            } else {
                guard let next = Self.add(unmaterialized, withExact) else {
                    return (Int.max, Int.max)
                }
                unmaterialized = next
            }
        }
        return (materialized, unmaterialized)
    }

    public func snapshot(
        activeRequests: Int, waitingRequests: Int, activeTokens: Int, backendBytesInUse: Int? = nil
    ) -> CBv2CapacitySnapshot {
        lock.lock()
        let ledger = chargedLedgerBytes
        // Honor the field's contract (see `CBv2CapacitySnapshot
        // .kvBytesReserved`): reserved carries the live external (compiled
        // padding) carve too, so "capacity − reserved" matches what
        // `canEverFit`/`reserve` will actually admit — the same figure the
        // engine loop's gauge publish reports. The carve is NOT storage,
        // so the in-use fallback stays ledger-only.
        let reserved = ledger + externalReserveBytes
        let capacity = _bytesCapacity
        lock.unlock()
        return CBv2CapacitySnapshot(
            activeRequests: activeRequests,
            waitingRequests: waitingRequests,
            kvBytesInUse: backendBytesInUse ?? ledger,
            kvBytesCapacity: capacity,
            kvBytesReserved: reserved,
            activeTokens: activeTokens)
    }
}
