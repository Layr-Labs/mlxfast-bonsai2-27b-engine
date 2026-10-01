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

    /// Per-row absolute RoPE offsets `[B]` (int32, device array).
    ///
    /// REBUILT from host integers only on membership changes; ADVANCED
    /// on-device (`+ L`) inside `updateAndAttend`. The step loop therefore
    /// never uploads fresh host arrays and never syncs (`.item()`) — the
    /// engine loop's per-step `asyncEval` (over `innerState()`) collapses
    /// the lazy `+ L` chain so it cannot grow O(steps) (DAR-325).
    ///
    /// NOTE: models must read this BEFORE calling `updateAndAttend` for the
    /// step (it holds the offsets of the tokens about to be processed), and
    /// KV-shared layers must reuse the SOURCE layer's pre-update capture —
    /// the same discipline as `gemma4CapturePositionOffset`.
    public var positionOffsets: MLXArray { cachedPositionOffsets }

    private var cachedPositionOffsets: MLXArray
    /// Host copy of `cachedPositionOffsets`: set where it is rebuilt,
    /// advanced by the same `+ L` (see `CBv2HostPositionOffsets`).
    private var hostPositionOffsets: [Int32] = []

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
        self.cachedPositionOffsets = Self.buildPositionOffsets(rows)
        self.hostPositionOffsets = rows.map { Int32($0.absoluteOffset) }
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
        // Advance offsets ON-DEVICE. Decode and packed prefill are
        // rectangular, so L is uniform across every bound row.
        advancePositionOffsets(by: queries.dim(2))
        return output
    }

    /// `updateAndAttend(queries:keys:values:scale:sinks:)` (no keep mask) for
    /// a one-row chunk on the query-block path, returning the blocks'
    /// outputs in query order (see the protocol). Declines, before any state
    /// changes, when an observation is armed, a span overlay is bound, the
    /// layer borrows its K/V, or the chunk would not attend in query blocks.
    public func updateAndAttendQueryBlocks(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> [MLXArray]? {
        guard kind.sharesKVWithLayer == nil, attentionMetadata == nil, attentionPacket == nil,
            rows.count == 1, boundSpanContexts?.contains(where: { $0 != nil }) != true,
            let blocks = CBv2AttentionV1.updateAndAttendQueryBlocks(
                row: rows[0], kind: kind, queries: queries, keys: keys, values: values,
                scale: scale, sinks: sinks, softcap: attentionSoftcap)
        else { return nil }
        // The offset advance `updateAndAttend` makes, host copy included
        // (with `CBv2HostPositionOffsets` on, a device-only add here would
        // leave the host copy behind by this chunk).
        advancePositionOffsets(by: queries.dim(2))
        return blocks
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
        advancePositionOffsets(by: keys.dim(2))
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

    // MARK: - Append written in place by a kernel (`CBv2InPlaceKVAppend`)

    /// Where an `n`-row append a kernel writes itself goes: the one bound
    /// row's key and value storage and the first row, or nil wherever
    /// `updateAndAttend` would do anything but a plain update of one
    /// contiguous full-attention row followed by `attendRowAfterUpdate`
    /// (receipts, span or keep masks, sinks, several rows, other storage).
    public func inPlaceAppendDestination(count n: Int, keyDType: DType, valueDType: DType)
        -> (keys: MLXArray, values: MLXArray, row: Int, previous: MLXArray?)?
    {
        guard CBv2InPlaceKVAppend.enabled, kind.sharesKVWithLayer == nil,
            case .full = kind.attention, !kind.isBidirectional, !kind.hasSinks,
            rows.count == 1, let row = rows[0] as? CBv2FullSequenceKV,
            attentionMetadata == nil, attentionPacket == nil, boundSpanContexts == nil
        else { return nil }
        return row.inPlaceAppendDestination(count: n, keyDType: keyDType, valueDType: valueDType)
    }

    /// `updateAndAttend(queries:keys:values:scale:sinks: nil)` for rows a
    /// kernel already wrote at `inPlaceAppendDestination` (`fence` is an
    /// output of that kernel): the same views, attention and offset advance.
    public func attendAfterInPlaceAppend(queries: MLXArray, fence: MLXArray, scale: Float)
        -> MLXArray
    {
        let row = rows[0] as! CBv2FullSequenceKV
        let (cachedKeys, cachedValues) = row.commitInPlaceAppend(
            count: queries.dim(2), fence: fence)
        let output = CBv2AttentionV1.attendRowAfterUpdate(
            kind: kind, queries: queries, cachedKeys: cachedKeys, cachedValues: cachedValues,
            scale: scale, sinks: nil, softcap: attentionSoftcap, spanContext: nil)
        advancePositionOffsets(by: queries.dim(2))
        return output
    }

    // MARK: - Private

    private func rebuildPositionOffsets() {
        positionOffsetsHostRebuilds += 1
        CBv2CoreInstrumentation.recordPositionOffsetsHostRebuild()
        cachedPositionOffsets = Self.buildPositionOffsets(rows)
        hostPositionOffsets = rows.map { Int32($0.absoluteOffset) }
    }

    /// `positionOffsets + n`: on the host (the same int32 values, no
    /// launch; one array shared by every layer that holds them), or the
    /// on-device add when `CBv2HostPositionOffsets` is off, made once for
    /// the layers that hand in the same array (`CBv2SharedPositionAdvance`).
    private func advancePositionOffsets(by n: Int) {
        let advanced = hostPositionOffsets.map { $0 &+ Int32(n) }
        if CBv2HostPositionOffsets.enabled,
            CBv2HostPositionOffsets.agrees(
                device: { self.cachedPositionOffsets + Int32(n) }, host: advanced)
        {
            cachedPositionOffsets = CBv2HostPositionOffsets.array(advanced)
        } else {
            cachedPositionOffsets = CBv2SharedPositionAdvance.advance(cachedPositionOffsets, by: n)
        }
        hostPositionOffsets = advanced
    }

    private static func buildPositionOffsets(_ rows: [CBv2SequenceKV]) -> MLXArray {
        CBv2SharedPositionAdvance.built(rows.map { Int32($0.absoluteOffset) })
    }
}

/// The on-device `positionOffsets + L` of a step, made once for the layers
/// that hand in the same array instead of once per layer (16 one-element
/// launches in every verify window of the 27B, on the command buffer the
/// acceptance readback waits on). Every layer of a step holds the same
/// values, so a host rebuild hands them one array (`built`: the array built
/// last, while the host values are the same) and an advance takes the add
/// made last of that same array by the same `L` (`advance`). The memo holds
/// its input, so no other array can take that identity, and arrays are
/// never written in place, so what it returns is that input's add, values
/// and dtype, whichever layer asks; a rebuild or rollback hands in another
/// array and misses. The first reuse also runs the layer's own add and
/// compares it bit for bit; a mismatch keeps one array and one add per
/// layer for the process. `MLXFAST_SHARED_POSITION_ADVANCE=0` keeps them.
enum CBv2SharedPositionAdvance {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_SHARED_POSITION_ADVANCE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var lastBuilt: (values: [Int32], array: MLXArray)?
    nonisolated(unsafe) private static var lastAdvance:
        (input: MLXArray, n: Int, output: MLXArray)?
    nonisolated(unsafe) private static var verdict: Bool?

    /// `MLXArray(values)`: the array built last while the values are the same.
    static func built(_ values: [Int32]) -> MLXArray {
        guard enabled else { return MLXArray(values) }
        return lock.withLock {
            guard verdict != false else { return MLXArray(values) }
            if let lastBuilt, lastBuilt.values == values { return lastBuilt.array }
            let array = MLXArray(values)
            lastBuilt = (values, array)
            return array
        }
    }

    /// `input + n`: the add made last when it was made of this array by `n`.
    static func advance(_ input: MLXArray, by n: Int) -> MLXArray {
        guard enabled, input.size > 0 else { return input + Int32(n) }
        return lock.withLock {
            guard verdict != false else { return input + Int32(n) }
            if let last = lastAdvance, last.input === input, last.n == n {
                if verdict == nil {
                    let own = input + Int32(n)
                    let same =
                        own.shape == last.output.shape && own.dtype == last.output.dtype
                        && (own .== last.output).all().item(Bool.self)
                    let line =
                        same
                        ? "equals the layer's own add (\(own.size) \(own.dtype) bitwise); one add per step"
                        : "differs from the layer's own add; one add per layer kept"
                    FileHandle.standardError.write(
                        Data("mlxfast shared position advance: first reuse \(line)\n".utf8))
                    verdict = same
                    if !same {
                        lastBuilt = nil
                        lastAdvance = nil
                        return own
                    }
                }
                return last.output
            }
            let output = input + Int32(n)
            lastAdvance = (input, n, output)
            return output
        }
    }
}

/// The advanced `positionOffsets` of an attention layer's step, built from a
/// host copy advanced by the same `+ L` instead of by an on-device add per
/// layer (16 one-element launches per verify window of the 27B). The host
/// copy is set wherever the array is rebuilt and advanced exactly as the
/// device chain is, so the values are the chain's, stale or not. Every layer
/// of a step holds the same values, so they share one array.
/// `MLXFAST_HOST_POSITION_OFFSETS=0` keeps the on-device add.
enum CBv2HostPositionOffsets {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_HOST_POSITION_OFFSETS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var last: (values: [Int32], array: MLXArray)?
    nonisolated(unsafe) private static var verdict: Bool?

    /// The first advance also runs the device add and compares it with the
    /// host values; a mismatch keeps the device add for the process.
    static func agrees(device: () -> MLXArray, host: [Int32]) -> Bool {
        lock.withLock {
            if let verdict { return verdict }
            let same = device().asArray(Int32.self) == host
            FileHandle.standardError.write(
                (same
                    ? "mlxfast host position offsets: first advance equals the device add; host copy\n"
                    : "mlxfast host position offsets: mismatch; on-device add kept\n")
                    .data(using: .utf8)!)
            verdict = same
            return same
        }
    }

    static func array(_ values: [Int32]) -> MLXArray {
        lock.withLock {
            if let last, last.values == values { return last.array }
            let array = MLXArray(values)
            last = (values, array)
            return array
        }
    }
}

/// An attention layer's KV append written by the kernel that produces the
/// keys (the fused q/k prework stores the new key rows and copies the value
/// rows straight into the row's storage) instead of by the two slice
/// updates of `CBv2FullSequenceKV.update` (two copy launches per attention
/// layer). The kernel's own self-test and trial decide whether it is used;
/// `MLXFAST_INPLACE_KV_APPEND=0` keeps the slice updates everywhere.
public enum CBv2InPlaceKVAppend {
    public static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_INPLACE_KV_APPEND"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
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
    /// collapse lazy chains: per-row storage plus the positionOffsets chain.
    public func innerState() -> [MLXArray] {
        var arrays = [cachedPositionOffsets]
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
