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
///
/// THE SUBMITTING THREAD (`MLXFAST_SEEDGAP_THREAD`, default on). Committing a
/// command buffer that binds unwired memory blocks the committing thread while
/// the box wires it (~10-20 ms/GB after the gates' idle, ~3.8 GB here). Done on
/// the engine thread, that stall is time the engine cannot encode the prompt's
/// remaining layers in, and with MLX's ten-buffer cap the GPU drains behind it.
/// So the touches are handed to ONE serial `DispatchQueue` made at load with
/// the side stream: the engine thread snapshots each due group's arrays
/// (retained C handles; checks each is already evaluated) and returns; the
/// queue builds the chunked operands, applies the kernels on the side stream
/// and commits. After load the side stream is used from that queue only. The
/// queue never takes mlx-swift's `evalLock` (taking it would put the engine's
/// next `asyncEval` behind the blocked commit again); MLX's own scheduler,
/// allocator, residency, library and kernel caches are locked, and the side
/// stream has its own command encoder (see `seedgap/CPRIME.md`).
/// Nothing is left pending past the seed: the engine's finalize calls
/// `settle()` right after a step's sampled tokens are read back, before any
/// token is emitted, so every touch is committed before `free_decode_begin`
/// answers (the touches overlapped the prompt's GPU work, so this finds the
/// queue idle and costs one lock). At process exit an `atexit` hook, registered
/// at load after MLX's global encoder map exists (so it runs before that map's
/// destructor), closes the queue and waits for it, bounded.
/// `MLXFAST_SEEDGAP_THREAD=0` commits from the engine thread as before.
public enum CBv2SideQueue {
    public static let enabled: Bool = !["0", "false", "no", "off"].contains(
        ProcessInfo.processInfo.environment["MLXFAST_SEEDGAP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "")

    /// Submits from the dedicated queue rather than the engine thread.
    public static let threaded: Bool = enabled && !["0", "false", "no", "off"].contains(
        ProcessInfo.processInfo.environment["MLXFAST_SEEDGAP_THREAD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "")

    private static let lock = NSLock()
    nonisolated(unsafe) private static var stream: mlx_stream?
    nonisolated(unsafe) private static var kernels: [String: mlx_fast_metal_kernel] = [:]
    nonisolated(unsafe) private static var queue: DispatchQueue?
    /// Set at exit: later submissions are dropped (touches only).
    nonisolated(unsafe) private static var closed = false
    nonisolated(unsafe) private static var waitNoted = false
    nonisolated(unsafe) private static var settleNoted = false
    /// Submissions handed to the queue and not yet finished, and whether exit
    /// has begun (no handoff after it); both under `idle`.
    private static let idle = NSCondition()
    nonisolated(unsafe) private static var pending = 0
    nonisolated(unsafe) private static var exiting = false
    /// The touch outputs of committed submissions, kept until their command
    /// buffer has completed (submission queue only). Released to the cache
    /// while the side kernel could still write them, a buffer could be handed
    /// to an engine array of the same size and then written by that kernel:
    /// the two queues are not ordered and MLX's buffers are hazard-untracked.
    nonisolated(unsafe) private static var retained: [mlx_vector_array] = []

    /// Creates the side queue (once), and under `threaded` the dispatch queue
    /// that alone submits to it. Idempotent; call at load.
    public static func prepare() {
        guard enabled else { return }
        _ = sideStream()
        if threaded { _ = submissionQueue() }
    }

    private static func sideStream() -> mlx_stream {
        lock.withLock {
            if let stream { return stream }
            let created = mlx_thread_unsafe_gpu_stream_new()
            stream = created
            return created
        }
    }

    private static func submissionQueue() -> DispatchQueue {
        lock.withLock {
            if let queue { return queue }
            let made = DispatchQueue(
                label: "mlxfast.seedgap.side-queue", qos: .userInitiated,
                autoreleaseFrequency: .workItem)
            queue = made
            // Registered after MLX's global encoder map was built: mlx-swift's
            // default `Stream.gpu` registers there at load, and `prepare()`
            // makes the side stream (also there) before this queue. So `exit`,
            // which runs atexit hooks and static destructors in reverse
            // registration order, runs this before that map's destructor
            // commits the side stream's encoder.
            atexit { CBv2SideQueue.closeAtExit() }
            return made
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

    /// The `threaded` form of `submitTouches`: `groups` (each group's arrays,
    /// in order) are handed to the submission queue, which forms each group's
    /// chunks with `chunk` and commits them all as one `submitTouches` would.
    ///
    /// On the calling (engine) thread only: one retained C handle per array
    /// (so the queue never reads a Swift `MLXArray`, and every array stays
    /// alive until it has been submitted), and an availability check per array
    /// (`array::is_available`). The check settles an evaluated array to
    /// `available` with its event detached HERE, on the thread that owns the
    /// arrays, so the queue's eval and any later engine eval of the same
    /// arrays only read their status and event. Should an array still be
    /// unscheduled or in flight (not expected for these weights), it is
    /// scheduled here by the engine's usual `asyncEval`, and the group is still
    /// committed by the queue but the engine thread waits for it (the original
    /// stall, correctness first; noted once on stderr).
    public static func submitTouchesOnQueue(
        _ groups: [[MLXArray]], name: String, inputs: Int, source: String,
        chunk: @escaping ([mlx_array]) -> [[mlx_array]]
    ) {
        guard !groups.isEmpty else { return }
        let queue = submissionQueue()
        var handles: [[mlx_array]] = []
        handles.reserveCapacity(groups.count)
        var unsettled: [MLXArray] = []
        for group in groups {
            var row: [mlx_array] = []
            row.reserveCapacity(group.count)
            for array in group {
                var available = false
                _mlx_array_is_available(&available, array.ctx)
                if !available { unsettled.append(array) }
                var handle = mlx_array_new()
                mlx_array_set(&handle, array.ctx)
                row.append(handle)
            }
            handles.append(row)
        }
        idle.lock()
        if exiting {
            idle.unlock()
            for row in handles { for handle in row { mlx_array_free(handle) } }
            return
        }
        pending += 1
        idle.unlock()
        let submission = Submission(
            handles: handles, chunk: chunk, name: name, inputs: inputs, source: source)
        let work: @Sendable () -> Void = { submission.run() }
        if unsettled.isEmpty {
            queue.async(execute: work)
        } else {
            // Whatever is not yet scheduled is scheduled here, on its own
            // stream under the engine's usual `asyncEval` (as the touch's eval
            // would have), and the engine thread waits for the queue, so no
            // thread writes these arrays while the queue reads them.
            asyncEval(unsettled)
            let note = lock.withLock { () -> Bool in
                defer { waitNoted = true }
                return !waitNoted
            }
            if note {
                FileHandle.standardError.write(
                    Data("seedgap side queue: a touched array was still in flight; that group waited\n".utf8))
            }
            queue.sync(execute: work)
        }
    }

    /// One group set handed to the submission queue: its retained handles
    /// (released once submitted, or dropped after exit closed the queue) and
    /// the chunking.
    private final class Submission: @unchecked Sendable {
        let handles: [[mlx_array]]
        let chunk: ([mlx_array]) -> [[mlx_array]]
        let name: String
        let inputs: Int
        let source: String

        init(
            handles: [[mlx_array]], chunk: @escaping ([mlx_array]) -> [[mlx_array]],
            name: String, inputs: Int, source: String
        ) {
            self.handles = handles
            self.chunk = chunk
            self.name = name
            self.inputs = inputs
            self.source = source
        }

        /// On the submission queue only.
        func run() {
            defer {
                for row in handles { for handle in row { mlx_array_free(handle) } }
                idle.lock()
                pending -= 1
                if pending == 0 { idle.broadcast() }
                idle.unlock()
            }
            releaseCompleted()
            guard !lock.withLock({ closed }) else { return }
            commit(handles.flatMap { chunk($0) }, name: name, inputs: inputs, source: source)
        }
    }

    /// Releases the retained outputs whose command buffer has completed (each
    /// output's event signalled). Submission queue only: nothing else
    /// references these outputs. What is still retained at exit is left to it.
    private static func releaseCompleted() {
        retained.removeAll { outputs in
            var done = true
            for i in 0 ..< mlx_vector_array_size(outputs) where done {
                var out = mlx_array_new()
                mlx_vector_array_get(&out, outputs, i)
                var available = false
                _mlx_array_is_available(&available, out)
                mlx_array_free(out)
                done = available
            }
            if done { mlx_vector_array_free(outputs) }
            return done
        }
    }

    /// `submitTouches` over C handles, on the submission queue. The outputs are
    /// retained until their command buffer completes (`retained`).
    private static func commit(_ chunks: [[mlx_array]], name: String, inputs: Int, source: String) {
        guard !chunks.isEmpty else { return }
        let stream = sideStream()
        let kernel = kernel(name: name, inputs: inputs, source: source)
        let outputs = mlx_vector_array_new()
        defer { retained.append(outputs) }
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
            for handle in chunk { mlx_vector_array_append_value(ins, handle) }
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

    /// The raw dtype of a handle (the queue's chunking key).
    public static func dtypeKey(_ handle: mlx_array) -> UInt32 {
        mlx_array_dtype(handle).rawValue
    }

    /// Waits until every submission handed to the queue so far has finished
    /// (its touches committed), bounded by `timeout` seconds. Called by the
    /// engine's finalize after a step's sampled tokens are read back and before
    /// they are emitted: by then the prompt's GPU work is done and the queue,
    /// which committed alongside it, is normally idle, so this is one lock. A
    /// no-op unless `threaded`.
    public static func settle(timeout: Double = 5) {
        guard threaded else { return }
        idle.lock()
        defer { idle.unlock() }
        guard pending > 0 else { return }
        let deadline = Date().addingTimeInterval(timeout)
        while pending > 0 {
            if !idle.wait(until: deadline) {
                if !settleNoted {
                    settleNoted = true
                    FileHandle.standardError.write(
                        Data("seedgap side queue: settle timed out after \(timeout) s\n".utf8))
                }
                return
            }
        }
    }

    /// The `atexit` hook: refuse any later handoff (checked under the same
    /// `idle` lock that counts `pending`, so an engine thread still mid-prompt
    /// on a fault exit cannot slip one in after the wait), let queued blocks
    /// skip their touches, then wait (bounded) for the queue, so `exit`'s
    /// static destructors (the global command-encoder map with the side
    /// stream's encoder) never meet a block mid-encode.
    static func closeAtExit() {
        idle.lock()
        exiting = true
        idle.unlock()
        lock.withLock { closed = true }
        settle(timeout: 5)
    }
}
