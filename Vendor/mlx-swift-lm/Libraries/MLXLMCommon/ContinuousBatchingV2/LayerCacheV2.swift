// LayerCacheV2.swift
//
// The per-layer batch-facing cache object v2 models interact with.
//
// `CBv2LayerCache` conforms to BOTH:
//  - `CBv2AttendingLayerCache` — the v2 surface: `updateAndAttend` owns the
//    KV update AND the attention computation (backends are swappable), and
//  - the legacy `KVCache` protocol — so it can travel through existing
//    `[KVCache]` plumbing. `update(keys:values:)` TRAPS: v2-adapted models
//    must call `updateAndAttend` (via the hook in `attentionWithCacheUpdate`).
//
// Batch membership is object membership: join = `appendRow`, leave =
// `removeRow`. There is no shared frontier, no left padding, no batch-wide
// trim — a row's KV state and positions cannot be affected by its batchmates.

import Foundation
import MLX

// MARK: - B2 shared position offsets (Option A)

/// Owner-held single `[B]` absolute-RoPE-offsets array, shared by every
/// storage-owning layer cache bound to the same batch composition.
///
/// Every layer's rows carry the same per-sequence `absoluteOffset`, so the
/// N per-layer `[B]` arrays (and their N per-step `+ L` device ops) are one
/// value computed N times. The box holds that value once: the bank binds
/// every owning layer to the same composition, and `broadcast` adopts (or
/// creates) the box whose host mirror already equals the rows, so all
/// layers read the same device array and exactly one of them advances it
/// per step.
///
/// Advancing is functional (`current = current + d`, never in place), so a
/// layer that captured the array before an advance keeps a valid snapshot.
/// Exactly-once falls out of the host mirror: rows are already advanced by
/// the attention call when `advanceForStep` runs, so the first layer whose
/// rows moved past the mirror advances (device `+ d`, mirror `+= d`) and
/// every later layer observes mirror == rows and skips — no new graph
/// node, no counter move. A stale box (rollback without rebind, foreign
/// sharer) self-heals the same way: the next layer to step sees a uniform
/// non-zero delta and advances to row truth.
///
/// Counters keep option (a): the per-cache `positionOffsetsHostRebuilds`
/// and the global instrumentation bump ONLY in `rebuildPositionOffsets`
/// (the broadcast), never on the step path. KV-shared layers keep a
/// private rowless box that is never advanced: they own no rows and reuse
/// the source's pre-update capture.
final class CBv2SharedPositionOffsets {
    private static let registryLock = NSLock()
    nonisolated(unsafe) private static var registry: [[Int32]: WeakSharedOffsets] = [:]

    private final class WeakSharedOffsets {
        weak var box: CBv2SharedPositionOffsets?
        init(_ box: CBv2SharedPositionOffsets) { self.box = box }
    }

    private let lock = NSLock()
    private var _current: MLXArray
    private var _mirror: [Int32]

    /// Private box (no registry): construction-time rows, before any bind.
    /// Same upload the old per-layer init performed; no counter move.
    init(values: [Int32]) {
        _current = MLXArray(values)
        _mirror = values
    }

    /// Membership-change broadcast: adopt the live box whose mirror already
    /// equals `values, or create and register one, then re-upload from host
    /// truth so every sharer observes the fresh base. The caller bumps its
    /// counters (option (a)) — this never runs on the step path.
    static func broadcast(values: [Int32]) -> CBv2SharedPositionOffsets {
        registryLock.lock()
        defer { registryLock.unlock() }
        registry = registry.filter { $0.value.box != nil }
        if let live = registry[values]?.box, live.mirrorEquals(values) {
            live.reupload(values: values)
            return live
        }
        let fresh = CBv2SharedPositionOffsets(values: values)
        registry[values] = WeakSharedOffsets(fresh)
        return fresh
    }

    /// The shared device array. Old-or-new across a concurrent advance —
    /// both are valid snapshots. Takes the (uncontended, engine-confined)
    /// box lock; the cost is noise next to a kernel dispatch.
    var current: MLXArray {
        lock.lock()
        defer { lock.unlock() }
        return _current
    }

    private func mirrorEquals(_ values: [Int32]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return _mirror == values
    }

    private func reupload(values: [Int32]) {
        lock.lock()
        defer { lock.unlock() }
        _current = MLXArray(values)
        _mirror = values
    }

    /// The step's single functional advance for this box. `lastQuery`
    /// selects the path branch: ordinary steps advance by `queries.dim(2)`
    /// (rectangular — preconditioned equal to `keys.dim(2)`), the final-
    /// layer last-query specialization by `keys.dim(2)` (preconditioned
    /// qL == 1, kvL > 1). The tensor shapes are validated up front; the
    /// ADVANCE itself follows the rows' observed uniform delta, which is
    /// ground truth for RoPE (it also absorbs a rollback that skipped a
    /// rebind, where the delta legitimately differs from the tensor L).
    /// A non-uniform delta means rows advanced unevenly — rectangular calls
    /// never do that — so it traps instead of corrupting N-1 rows.
    func advanceForStep(
        queries: MLXArray, keys: MLXArray,
        rowOffsets: some Sequence<Int>, rowCount: Int,
        lastQuery: Bool, layerIndex: Int
    ) {
        let batch = queries.dim(0)
        let queryLength = queries.dim(2)
        let keyLength = keys.dim(2)
        precondition(
            batch == rowCount,
            "CBv2SharedPositionOffsets: step batch \(batch) != bound rows \(rowCount) (layer \(layerIndex))")
        precondition(
            keys.dim(0) == batch,
            "CBv2SharedPositionOffsets: keys batch \(keys.dim(0)) != queries batch \(batch) (layer \(layerIndex))")
        let stepLength: Int
        if lastQuery {
            precondition(
                queryLength == 1 && keyLength > 1,
                "CBv2SharedPositionOffsets: last-query step needs qL == 1 and kvL > 1"
                    + ", got qL=\(queryLength) kvL=\(keyLength) (layer \(layerIndex))")
            stepLength = keyLength
        } else {
            precondition(
                queryLength == keyLength,
                "CBv2SharedPositionOffsets: rectangular step needs qL == kvL"
                    + ", got qL=\(queryLength) kvL=\(keyLength) (layer \(layerIndex))")
            stepLength = queryLength
        }

        lock.lock()
        defer { lock.unlock() }
        var offsets = rowOffsets.makeIterator()
        guard let first = offsets.next() else {
            // Rowless degenerate step: no row truth to compare against, so
            // advance by the tensor length, exactly as the pre-B2 code did.
            _current = _current + Int32(stepLength)
            return
        }
        precondition(
            _mirror.count == rowCount,
            "CBv2SharedPositionOffsets: mirror holds \(_mirror.count) entries for"
                + " \(rowCount) rows (layer \(layerIndex))")
        let delta = first - Int(_mirror[0])
        var index = 1
        while index < rowCount {
            guard let next = offsets.next() else { break }
            precondition(
                next - Int(_mirror[index]) == delta,
                "CBv2SharedPositionOffsets: rows advanced non-uniformly (layer \(layerIndex))"
                    + " — rectangular steps move every row by the same length")
            index += 1
        }
        precondition(
            index == rowCount,
            "CBv2SharedPositionOffsets: row count drifted mid-step (layer \(layerIndex))")
        guard delta != 0 else { return }  // a sharer already advanced this step
        _current = _current + Int32(delta)
        _mirror = _mirror.map { $0 + Int32(delta) }
    }
}

/// Per-layer, batch-facing cache + attention dispatcher for the v2 engine.
public final class CBv2LayerCache: CBv2AttendingLayerCache {

    var attentionMetadata: CBv2AttentionMetadataForward?
    var attentionPacket: CBv2AttentionPacketForward?

    public let layerIndex: Int
    public let kind: CBv2LayerKind

    /// Ordered per-row sequence states (row order == batch row order).
    /// Empty for KV-shared layers (`kind.sharesKVWithLayer != nil`), which
    /// own no storage and borrow via `attendBorrowing`.
    public private(set) var rows: [CBv2SequenceKV]

    /// Per-row absolute RoPE offsets `[B]` (int32, device array), held by the
    /// owner box shared with every layer bound to the same composition (B2).
    ///
    /// REBUILT from host integers only on membership changes (the broadcast
    /// in `rebuildPositionOffsets`); ADVANCED on-device once per step by
    /// whichever sharing layer steps first (`advanceForStep`). The step loop
    /// therefore never uploads fresh host arrays and never syncs (`.item()`)
    /// — the engine loop's per-step `asyncEval` (over `innerState()`)
    /// collapses the lazy advance chain so it cannot grow O(steps)
    /// (DAR-325).
    ///
    /// NOTE: models must read this BEFORE calling `updateAndAttend` for the
    /// step (it holds the offsets of the tokens about to be processed), and
    /// KV-shared layers must reuse the SOURCE layer's pre-update capture —
    /// the same discipline as `gemma4CapturePositionOffset`. KV-shared
    /// layers keep their private rowless box (their rows are empty and
    /// `advanceForStep` is never called on their behalf).
    public var positionOffsets: MLXArray { sharedOffsets.current }

    private var sharedOffsets: CBv2SharedPositionOffsets

    /// MTP-only verification policy. When true, an L>1 update still projects
    /// and stores the whole rectangle once, but attention evaluates each
    /// query with the canonical L=1 SDPA path and its exact visible KV prefix.
    public var mtpSerializesRectangularAttention = false
    public var mtpBatchesRectangularAttention = false

    /// Times `positionOffsets` was rebuilt from host integers. Tests assert
    /// this only moves on membership changes — never inside the step loop.
    public private(set) var positionOffsetsHostRebuilds = 0

    /// Optional attention-logit soft cap (`cap * tanh(qk / cap)` before
    /// softmax, Gemma-2 style). Construction-time configuration from model
    /// config — identical plumbing on both backends (`PagedLayerCache` takes
    /// the same parameter); never part of the per-call contract surface.
    public let attentionSoftcap: Float?

    /// Optional vision span context for each CURRENT prefill row. The engine
    /// binds this array immediately before graph construction and clears it
    /// immediately after. nil outside that window; nil entries are ordinary
    /// text rows sharing a rectangular call.
    private(set) var boundSpanContexts: [CBv2SpanChunkContext?]?

    public init(
        layerIndex: Int, kind: CBv2LayerKind, rows: [CBv2SequenceKV] = [],
        attentionSoftcap: Float? = nil
    ) {
        precondition(
            kind.sharesKVWithLayer == nil || rows.isEmpty,
            "CBv2LayerCache: KV-shared layers own no rows")
        self.layerIndex = layerIndex
        self.kind = kind
        self.rows = rows
        self.attentionSoftcap = attentionSoftcap
        self.sharedOffsets = CBv2SharedPositionOffsets(values: rows.map { Int32($0.absoluteOffset) })
    }

    // MARK: - Membership (the ONLY places positionOffsets is host-rebuilt)

    public func appendRow(_ row: CBv2SequenceKV) {
        precondition(
            kind.sharesKVWithLayer == nil, "CBv2LayerCache: cannot add rows to a KV-shared layer")
        rows.append(row)
        rebuildPositionOffsets()
    }

    public func removeRow(at index: Int) {
        rows.remove(at: index)
        rebuildPositionOffsets()
    }

    /// Replace the whole row set (batch recomposition). Also the correct way
    /// to re-sync `positionOffsets` after out-of-band row mutation
    /// (e.g. rollback during speculative verification).
    public func setRows(_ newRows: [CBv2SequenceKV]) {
        precondition(
            kind.sharesKVWithLayer == nil || newRows.isEmpty,
            "CBv2LayerCache: KV-shared layers own no rows")
        rows = newRows
        rebuildPositionOffsets()
    }

    // MARK: - CBv2AttendingLayerCache

    public func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: sinks,
            keepMask: nil)
    }

    public func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, keepMask: MLXArray?
    ) -> MLXArray {
        precondition(
            kind.sharesKVWithLayer == nil,
            "CBv2LayerCache: KV-shared layer \(layerIndex) must use attendBorrowing")
        // An observed forward receipt states a dense causal/window mask: the
        // replay reference attends the whole retained KV. A keep mask removes
        // keys that reference would attend, so its output can never replay.
        // Refuse the capture BY NAME instead of recording a bad receipt.
        var metadata: CBv2AttentionMetadataObservation?
        var packet: CBv2AttentionPacketObservation?
        if keepMask == nil {
            let spans = boundSpanContexts?.contains(where: { $0 != nil }) ?? false
            metadata = attentionMetadata?.begin(
                cache: self, queries: queries, keys: keys, values: values, scale: scale,
                sinks: sinks, softcap: attentionSoftcap, spans: spans)
            packet = attentionPacket?.begin(
                cache: self, queries: queries, keys: keys, values: values, scale: scale,
                sinks: sinks, softcap: attentionSoftcap, spans: spans)
        } else {
            attentionMetadata?.state.refuse("keep_masked_attention_not_replayable")
            attentionPacket?.state.refuse("keep_masked_attention_not_replayable")
        }
        let output = CBv2AttentionV1.updateAndAttend(
            rows: rows, kind: kind,
            queries: queries, keys: keys, values: values,
            scale: scale, sinks: sinks, softcap: attentionSoftcap,
            spanContexts: boundSpanContexts,
            serializeQueries: mtpSerializesRectangularAttention,
            keepMask: keepMask, metadata: metadata, packet: packet)
        // Advance the shared offsets ON-DEVICE through the single step
        // helper. Decode and packed prefill are rectangular, so the rows
        // just advanced by exactly queries.dim(2); the last-query path
        // below advances by keys.dim(2) instead (see the branch there).
        // Whichever sharing layer steps first performs the advance — the
        // rest observe it and skip.
        sharedOffsets.advanceForStep(
            queries: queries, keys: keys,
            rowOffsets: rows.lazy.map { $0.absoluteOffset }, rowCount: rows.count,
            lastQuery: false, layerIndex: layerIndex)
        return output
    }

    /// Final-layer prompt specialization (see LastQueryPrefillV2.swift):
    /// commit the whole chunk's K/V, attend only its newest query row.
    /// Offsets advance by the K/V length, NOT the query length — the chunk
    /// consumed `keys.dim(2)` positions even though one query was evaluated.
    public func updateAndAttendLastQuery(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(
            kind.sharesKVWithLayer == nil,
            "CBv2LayerCache: KV-shared layer \(layerIndex) owns no storage to commit")
        precondition(
            !mtpSerializesRectangularAttention,
            "CBv2LayerCache: last-query prefill is never part of an MTP verify round")
        let output = CBv2AttentionV1.updateAndAttendLastQuery(
            rows: rows, kind: kind,
            queries: queries, keys: keys, values: values,
            scale: scale, sinks: sinks, softcap: attentionSoftcap)
        // Same single helper, last-query branch: the chunk consumed
        // keys.dim(2) positions even though one query was evaluated.
        sharedOffsets.advanceForStep(
            queries: queries, keys: keys,
            rowOffsets: rows.lazy.map { $0.absoluteOffset }, rowCount: rows.count,
            lastQuery: true, layerIndex: layerIndex)
        return output
    }

    public func attendBorrowing(
        source: CBv2AttendingLayerCache,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(
            kind.sharesKVWithLayer != nil,
            "CBv2LayerCache: attendBorrowing called on a storage-owning layer")
        precondition(
            kind.sharesKVWithLayer == source.layerIndex,
            "CBv2LayerCache: layer \(layerIndex) shares KV with \(kind.sharesKVWithLayer!), not \(source.layerIndex)"
        )
        return CBv2AttentionV1.attendBorrowing(
            sourceRows: source.rows, sourceKind: source.kind, kind: kind,
            queries: queries, scale: scale, sinks: sinks, softcap: attentionSoftcap,
            spanContexts: boundSpanContexts,
            serializeQueries: mtpSerializesRectangularAttention)
    }

    // MARK: - Private

    private func rebuildPositionOffsets() {
        positionOffsetsHostRebuilds += 1
        CBv2CoreInstrumentation.recordPositionOffsetsHostRebuild()
        // Broadcast: adopt the live box for this composition (or create
        // it) and re-upload from host truth, so every layer sharing the
        // composition observes the fresh base. Counters bump here (option
        // (a)) — never on the step path.
        sharedOffsets = CBv2SharedPositionOffsets.broadcast(
            values: rows.map { Int32($0.absoluteOffset) })
    }
}

// MARK: - Final-layer last-query prefill

extension CBv2LayerCache: CBv2LastQueryPrefillLayerCache {}

// MARK: - Keep-mask support

/// The contiguous backend composes the keep mask with the causal or window
/// mask in absolute coordinates, so it is exact on every layer this cache
/// vends.
extension CBv2LayerCache: CBv2KeepMaskCapableCache {
    public var honorsKeepMask: Bool { true }
}

// MARK: - Vision span-mask binding

extension CBv2LayerCache: CBv2PackedSpanMaskBinding {
    public func bindSpanContext(_ context: CBv2SpanChunkContext?) {
        boundSpanContexts = context.map { [$0] }
    }

    public func bindSpanContexts(_ contexts: [CBv2SpanChunkContext?]?) {
        boundSpanContexts = contexts
    }
}

// MARK: - Legacy KVCache conformance

extension CBv2LayerCache: KVCache {
    /// Legacy scalar offset: max row offset. Host integers only — no sync.
    public var offset: Int {
        rows.reduce(0) { max($0, $1.absoluteOffset) }
    }

    public var maxSize: Int? {
        switch kind.attention {
        case .full: return nil
        case .slidingWindow(let window): return window
        }
    }

    /// The engine loop evaluates cache inner state each step (asyncEval) to
    /// collapse lazy chains: per-row storage plus the shared positionOffsets
    /// chain (vending the box's current array keeps the single shared chain
    /// collapsed no matter which layer's inner state is evaluated).
    public func innerState() -> [MLXArray] {
        var arrays = [sharedOffsets.current]
        for row in rows {
            if let provider = row as? CBv2InnerStateProviding {
                arrays.append(contentsOf: provider.cbv2InnerState())
            }
        }
        return arrays
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "CBv2LayerCache.update(keys:values:) is unsupported — v2-adapted models must call updateAndAttend (layer \(layerIndex))"
        )
    }

    public var state: [MLXArray] {
        get { [] }
        set {
            fatalError("CBv2LayerCache has no serializable state (layer \(layerIndex))")
        }
    }

    public var metaState: [String] {
        get { [] }
        set {
            fatalError("CBv2LayerCache has no metaState (layer \(layerIndex))")
        }
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        fatalError(
            "CBv2LayerCache.makeMask is unsupported — v2 attention owns its masks (layer \(layerIndex))"
        )
    }

    public func copy() -> any KVCache {
        fatalError(
            "CBv2LayerCache.copy is unsupported — v2 rows are engine-owned (layer \(layerIndex))")
    }
}
