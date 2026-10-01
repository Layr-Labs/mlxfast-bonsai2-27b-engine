// ContiguousKVBackend.swift
//
// The v1 `CBv2KVBackend`: per-sequence contiguous MLX buffers
// (`CBv2FullSequenceKV` / `CBv2WindowedSequenceKV`).
// The paged backend (workstream C) implements the same protocol behind a
// Metal kernel; the scheduler and models never see the difference.

import Foundation
import Cmlx
import MLX

/// Configuration for `CBv2ContiguousKVBackend`.
public struct CBv2ContiguousBackendConfig: Sendable {
    /// Byte budget for all live sequence KV (admission ceiling). This is
    /// the INITIAL budget; the backend's live ceiling can be re-sliced at
    /// runtime via `CBv2ContiguousKVBackend.updateBytesCapacity(_:)`.
    public var bytesCapacity: Int
    /// dtype assumed for admission estimates (actual allocation adopts the
    /// dtype of the first appended K/V).
    public var kvDType: DType

    public init(
        bytesCapacity: Int,
        kvDType: DType = .float16
    ) {
        self.bytesCapacity = bytesCapacity
        self.kvDType = kvDType
    }
}

/// A request's engine is assembled first (`free_decode_begin` builds one
/// before its seed forward), and a GPU left idle starts its first command
/// buffer late: on an M4 Max after a 20 s idle, 13-15 ms after the commit
/// (0.5 ms warm). One scalar add committed as the assembly begins starts that
/// wake-up before the ~4 ms of assembly and graph building ahead of the first
/// real submission. Nothing reads it. `MLXFAST_GPU_WAKE=0` turns it off.
enum CBv2GPUWake {
    static let enabled = !["0", "false", "no", "off"].contains(
        ProcessInfo.processInfo.environment["MLXFAST_GPU_WAKE"]?.lowercased() ?? "")

    static func now() {
        guard enabled else { return }
        asyncEval(MLXArray(Int32(1)) + MLXArray(Int32(1)))
    }
}

/// Factory + accounting for per-sequence contiguous KV state.
///
/// Thread-safe: the live-row registry is lock-protected (`makeSequenceState`
/// runs on the admission path while `release` runs on the engine loop).
/// `bytesInUse` is truthful — it sums the ACTUAL allocated bytes of live
/// rows (which grow by doubling), not a worst-case estimate.
///
/// Admission RESERVES: rows allocate lazily (`byteCount == 0` until their
/// first update), so judging capacity against `bytesInUse` alone would let
/// several same-step admissions collectively exceed `bytesCapacity`. Each
/// admitted row therefore holds a reservation equal to its estimated
/// initial bytes until its actual allocation exceeds it
/// (`max(byteCount, reservation)` per row — see `bytesReserved`), and the
/// capacity check + registration are a single atomic section.
public final class CBv2ContiguousKVBackend: CBv2KVBackend {

    public let config: CBv2ContiguousBackendConfig
    public var prefixReuseBackend: CBv2PrefixReuseBackend { .contiguousUnquantized }

    private let lock = NSLock()
    private var live: [ObjectIdentifier: CBv2SequenceKV] = [:]
    /// Admission reservation per live row (estimated initial bytes),
    /// released with the row. NOTE: estimates assume `config.kvDType`; a
    /// model that caches wider elements (e.g. fp32) under-reserves until
    /// the first update trues the row up to its actual `byteCount` —
    /// `AdmissionV2` (which can carry per-layer element sizes) remains the
    /// primary admission gate.
    private var reservations: [ObjectIdentifier: Int] = [:]
    /// Live byte budget, seeded from `config.bytesCapacity` and resizable
    /// at runtime (`updateBytesCapacity`). Lock-protected: the atomic
    /// admit-and-register check reads it inside its critical section.
    private var liveBytesCapacity: Int

    public init(config: CBv2ContiguousBackendConfig) {
        CBv2GPUWake.now()
        self.config = config
        self.liveBytesCapacity = config.bytesCapacity
    }

    public var bytesCapacity: Int {
        lock.lock()
        defer { lock.unlock() }
        return liveBytesCapacity
    }

    /// Runtime capacity update (multi-model co-residency re-slicing).
    /// Shrink never evicts live rows: registrations above a new lower
    /// ceiling stay resident and new admissions fail until usage drains
    /// below the new ceiling; grow admits immediately
    /// (`CBv2KVBackend.updateBytesCapacity`).
    public func updateBytesCapacity(_ bytes: Int) {
        lock.lock()
        liveBytesCapacity = max(0, bytes)
        lock.unlock()
    }

    public var bytesInUse: Int {
        lock.lock()
        defer { lock.unlock() }
        return live.values.reduce(0) { $0 + $1.byteCount }
    }

    /// Actual bytes plus outstanding admission reservations — what the
    /// capacity check judges against (`CBv2KVBackend.bytesReserved`).
    public var bytesReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return accountedBytesLocked()
    }

    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try validate(layerKinds: layerKinds)
        guard promptLength <= maxLength else {
            throw CBv2KVError.backendIneligible(
                reason: "promptLength \(promptLength) exceeds maxLength \(maxLength)")
        }

        let state = layerKinds.map { kind -> CBv2SequenceKV? in
            makeRow(kind: kind, promptLength: promptLength, maxLength: maxLength)
        }
        try registerReserving(
            state,
            estimates: rowEstimates(
                layerKinds: layerKinds, promptLength: promptLength, maxLength: maxLength))
        return state
    }

    public func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        plan: CBv2PrefixReusePlan,
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try validate(layerKinds: layerKinds)
        guard prefix.count == layerKinds.count else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix count \(prefix.count) != layer count \(layerKinds.count)")
        }

        guard plan.backend == prefixReuseBackend else {
            throw CBv2KVError.backendIneligible(
                reason:
                    "prefix plan backend \(plan.backend.rawValue) != \(prefixReuseBackend.rawValue)")
        }
        guard plan.matchedBoundary <= maxLength,
            plan.replayStart >= 0,
            plan.replayStart <= plan.matchedBoundary,
            plan.replayTokens == plan.matchedBoundary - plan.replayStart
        else {
            throw CBv2KVError.backendIneligible(reason: "invalid prefix replay plan")
        }

        // Full snapshots use either C (ordinary safe-layout replay) or M
        // (frozen-full replay). Every owning full row must agree.
        let expectedSnapshotOffset = plan.restoredFullTokens
        var sawOwningFull = false
        for (index, entry) in prefix.enumerated() {
            let kind = layerKinds[index]
            if kind.sharesKVWithLayer != nil {
                guard entry == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) is KV-shared but received a prefix snapshot")
                }
                continue
            }
            if case .slidingWindow = kind.attention {
                guard entry == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason:
                            "layer \(index) is windowed but received a prefix snapshot")
                }
                continue
            }
            sawOwningFull = true
            guard let entry else {
                throw CBv2KVError.backendIneligible(
                    reason: "owning full layer \(index) is missing its prefix snapshot")
            }
            guard kind.sharesKVWithLayer == nil else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index) is KV-shared but received a prefix snapshot")
            }
            guard case .full = kind.attention else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "layer \(index) is windowed but received a prefix snapshot (windowed layers are recomputed)"
                )
            }
            guard entry.offset == expectedSnapshotOffset else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "prefix offset \(entry.offset) != planned \(expectedSnapshotOffset) at layer \(index)"
                )
            }
            guard entry.keys.dim(2) == entry.offset, entry.values.dim(2) == entry.offset else {
                throw CBv2KVError.backendIneligible(
                    reason: "full prefix snapshot at layer \(index) does not exactly cover its offset")
            }
        }
        guard sawOwningFull else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix replay requires at least one storage-owning full layer")
        }

        let state = layerKinds.enumerated().map { index, kind -> CBv2SequenceKV? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                // Sliding rows always start empty at C and rebuild through R.
                return CBv2WindowedSequenceKV(
                    window: window, kvHeads: kind.kvHeads, headDim: kind.headDim,
                    initialOffset: plan.replayStart)
            case .full:
                let entry = prefix[index]!
                if plan.strategy == .frozenFullReplay {
                    return CBv2FrozenReplayFullSequenceKV(
                        snapshot: entry,
                        replayStart: plan.replayStart,
                        maxLength: maxLength,
                        kvHeads: kind.kvHeads,
                        headDim: kind.headDim)
                }
                let row = makeRow(
                    kind: kind,
                    promptLength: expectedSnapshotOffset,
                    maxLength: maxLength)!
                _ = row.update(keys: entry.keys, values: entry.values)
                return row
            }
        }
        try registerReserving(
            state,
            estimates: adoptionRowEstimates(
                prefix: prefix,
                layerKinds: layerKinds,
                maxLength: maxLength))
        return state
    }

    public func release(_ state: [CBv2SequenceKV?]) {
        lock.lock()
        defer { lock.unlock() }
        for row in state {
            guard let row else { continue }
            let key = ObjectIdentifier(row)
            live.removeValue(forKey: key)
            reservations.removeValue(forKey: key)
        }
    }

    /// Prepared rows were filled under an external stage reservation. Register
    /// their exact final allocation atomically before that reservation ends.
    func adoptPreparedCheckpoint(_ state: [CBv2SequenceKV?]) throws {
        guard !state.isEmpty, state.allSatisfy({ $0 is CBv2FullSequenceKV }) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        try registerReserving(state, estimates: state.map { $0?.byteCount })
    }

    // MARK: - Private

    /// Bytes currently charged against capacity: each live row counts for
    /// the LARGER of its actual allocation and its outstanding admission
    /// reservation, so a not-yet-allocated row still occupies its estimate
    /// and a grown row is charged its true size (`bytesInUse`-truthful).
    /// Caller holds `lock`.
    private func accountedBytesLocked() -> Int {
        live.reduce(0) { total, entry in
            total + max(entry.value.byteCount, reservations[entry.key] ?? 0)
        }
    }

    /// Atomically admit + register: the capacity check and the reservation
    /// write share one critical section so N same-step admissions cannot
    /// collectively overshoot `bytesCapacity` (the pre-fix bug — rows report
    /// `byteCount == 0` until first update).
    private func registerReserving(_ state: [CBv2SequenceKV?], estimates: [Int?]) throws {
        precondition(state.count == estimates.count, "estimate/state count mismatch")
        lock.lock()
        defer { lock.unlock() }
        let needed = zip(state, estimates).reduce(0) { total, pair in
            guard let row = pair.0 else { return total }
            return total + max(row.byteCount, pair.1 ?? 0)
        }
        let available = liveBytesCapacity - accountedBytesLocked()
        guard needed <= available else {
            throw CBv2KVError.capacityExhausted(needed: needed, available: max(0, available))
        }
        for (row, estimate) in zip(state, estimates) {
            guard let row else { continue }
            let key = ObjectIdentifier(row)
            live[key] = row
            reservations[key] = estimate ?? 0
        }
    }

    private func makeRow(kind: CBv2LayerKind, promptLength: Int, maxLength: Int)
        -> CBv2SequenceKV?
    {
        guard kind.sharesKVWithLayer == nil else { return nil }
        switch kind.attention {
        case .slidingWindow(let window):
            return CBv2WindowedSequenceKV(
                window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
        case .full:
            return CBv2FullSequenceKV(
                promptLength: promptLength, maxLength: maxLength,
                kvHeads: kind.kvHeads, headDim: kind.headDim)
        }
    }

    private func validate(layerKinds: [CBv2LayerKind]) throws {
        for (index, kind) in layerKinds.enumerated() {
            if case .slidingWindow(let window) = kind.attention, window <= 0 {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): non-positive window \(window)")
            }
            if let source = kind.sharesKVWithLayer {
                guard source >= 0, source < layerKinds.count, source != index else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index): invalid KV-share source \(source)")
                }
                guard layerKinds[source].sharesKVWithLayer == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason:
                            "layer \(index): KV-share source \(source) is itself a shared layer")
                }
            }
            // The v1 contiguous backend attends through MLXFast SDPA, which
            // supports attention sinks natively — sink models are eligible.
        }
    }

    /// Per-layer estimated initial allocation bytes, aligned to `layerKinds`
    /// (nil for KV-shared layers, which own no storage). Full layers allocate
    /// `promptLength + 256` slots capped at maxLength; windowed layers
    /// allocate their full ring up front. The sum is the reservation charged
    /// against capacity at admission.
    private func rowEstimates(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) -> [Int?] {
        let itemSize = config.kvDType.size
        return layerKinds.map { kind -> Int? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                return window * kind.kvHeads * kind.headDim * itemSize * 2
            case .full:
                let slots = min(maxLength, max(1, promptLength + CBv2FullSequenceKV.initialSlack))
                return slots * kind.kvHeads * kind.headDim * itemSize * 2
            }
        }
    }

    /// Adoption transfers native-dtype full rows from staging. Reserve their
    /// full request span before publication so later capacity growth cannot
    /// outrun the backend hard ceiling. Sliding rows retain their fixed ring
    /// estimate.
    private func adoptionRowEstimates(
        prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind],
        maxLength: Int
    ) -> [Int?] {
        layerKinds.enumerated().map { index, kind in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                return window * kind.kvHeads * kind.headDim * config.kvDType.size * 2
            case .full:
                guard let entry = prefix[index] else { return 0 }
                return maxLength * kind.kvHeads * kind.headDim
                    * (entry.keys.dtype.size + entry.values.dtype.size)
            }
        }
    }
}

/// A second GPU command queue for submissions no result ever reads
/// (`MLXFAST_SEEDGAP`, default on; `MLXFAST_SEEDGAP=0` restores the original).
///
/// The seed's residency touches (`DFlash2ResidencyPrefetch`) bind the drafter's
/// weights and the window's arrays behind the prompt forward's early
/// submissions. On the prompt's own queue every touch command buffer sits
/// between two prompt command buffers, so the prompt (and the seed token's
/// readback behind it) can wait for a residency restore that nothing in the
/// seed reads. Here the same touches, in the same groups, at the same layer
/// points, inside the same `free_decode_begin`, are committed on their own
/// queue instead. They read element 0 of arrays that are already evaluated and
/// write a scalar no one reads, so no value of any result can change.
///
/// The queue is created once per process by `prepare()`, at load (the touch
/// prewarm), with the same process-global registration the default GPU stream
/// uses (`mlx_thread_unsafe_gpu_stream_new`), so the engine thread can submit
/// to it whichever thread created it.
public enum CBv2SideQueue {
    public static let enabled: Bool = !["0", "false", "no", "off"].contains(
        ProcessInfo.processInfo.environment["MLXFAST_SEEDGAP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "")

    private static let lock = NSLock()
    nonisolated(unsafe) private static var stream: mlx_stream?
    nonisolated(unsafe) private static var kernels: [String: mlx_fast_metal_kernel] = [:]

    /// Creates the side queue (once). Idempotent; call at load.
    public static func prepare() {
        guard enabled else { return }
        _ = sideStream()
    }

    private static func sideStream() -> mlx_stream {
        lock.withLock {
            if let stream { return stream }
            let created = mlx_thread_unsafe_gpu_stream_new()
            stream = created
            return created
        }
    }

    private static func kernel(name: String, inputs: Int, source: String) -> mlx_fast_metal_kernel {
        lock.withLock {
            let key = "\(name)/\(inputs)"
            if let cached = kernels[key] { return cached }
            let inputNames = mlx_vector_string_new()
            defer { mlx_vector_string_free(inputNames) }
            for i in 0 ..< inputs { mlx_vector_string_append_value(inputNames, "w\(i)") }
            let outputNames = mlx_vector_string_new()
            defer { mlx_vector_string_free(outputNames) }
            mlx_vector_string_append_value(outputNames, "out")
            let made = mlx_fast_metal_kernel_new(name, inputNames, outputNames, source, "", false, false)
            kernels[key] = made
            return made
        }
    }

    /// One single-thread launch per chunk (each chunk exactly `inputs` arrays),
    /// all committed together on the side queue. The scalar outputs are dropped,
    /// exactly as the default-queue touches drop theirs.
    public static func submitTouches(
        _ chunks: [[MLXArray]], name: String, inputs: Int, source: String
    ) {
        guard !chunks.isEmpty else { return }
        let stream = sideStream()
        let kernel = kernel(name: name, inputs: inputs, source: source)
        let outputs = mlx_vector_array_new()
        defer { mlx_vector_array_free(outputs) }
        for chunk in chunks {
            precondition(chunk.count == inputs, "CBv2SideQueue: a touch chunk must hold \(inputs) arrays")
            let config = mlx_fast_metal_kernel_config_new()
            defer { mlx_fast_metal_kernel_config_free(config) }
            mlx_fast_metal_kernel_config_set_grid(config, 1, 1, 1)
            mlx_fast_metal_kernel_config_set_thread_group(config, 1, 1, 1)
            let shape: [Int32] = [1]
            mlx_fast_metal_kernel_config_add_output_arg(config, shape, 1, MLX_FLOAT32)
            let ins = mlx_vector_array_new()
            defer { mlx_vector_array_free(ins) }
            for array in chunk { mlx_vector_array_append_value(ins, array.ctx) }
            var result = mlx_vector_array_new()
            defer { mlx_vector_array_free(result) }
            mlx_fast_metal_kernel_apply(&result, kernel, ins, config, stream)
            for i in 0 ..< mlx_vector_array_size(result) {
                var out = mlx_array_new()
                mlx_vector_array_get(&out, result, i)
                mlx_vector_array_append_value(outputs, out)
                mlx_array_free(out)
            }
        }
        mlx_async_eval(outputs)
    }
}
