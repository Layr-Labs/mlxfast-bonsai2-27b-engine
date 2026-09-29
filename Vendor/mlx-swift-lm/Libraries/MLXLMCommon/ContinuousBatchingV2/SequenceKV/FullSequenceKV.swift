// FullSequenceKV.swift
//
// ContinuousBatchingV2 per-sequence KV storage for FULL attention layers.
//
// One instance owns the K/V for ONE sequence at ONE layer. There is no shared
// batch frontier, no left padding, and no batch-wide trim: joining or leaving
// a batch never touches this object (batch membership is just list membership
// in `CBv2LayerCache.rows`).

import Foundation
import MLX
import MLXFast

/// Counters for the v2 core runtime's own host-interaction points.
///
/// The engine step loop must never force a host sync (`.item()`, `asArray`,
/// blocking `eval`) and must never rebuild per-row metadata arrays from host
/// integers outside membership changes. Any CBv2 core code that *does* touch
/// the host goes through these counters so tests can assert the step loop is
/// clean (see `CBv2CoreTests`). This deliberately does not instrument MLX
/// itself — only our own sync points.
public enum CBv2CoreInstrumentation {
    private static let lock = NSLock()

    nonisolated(unsafe) private static var _hostSyncs = 0
    nonisolated(unsafe) private static var _positionOffsetsHostRebuilds = 0

    /// Number of host syncs performed by CBv2 core code.
    public static var hostSyncs: Int {
        lock.lock()
        defer { lock.unlock() }
        return _hostSyncs
    }

    /// Number of times a layer cache rebuilt `positionOffsets` from host
    /// integers. Must only ever increase on batch membership changes.
    public static var positionOffsetsHostRebuilds: Int {
        lock.lock()
        defer { lock.unlock() }
        return _positionOffsetsHostRebuilds
    }

    /// Gate for `recordHostSync`: the engine's finalize readbacks count
    /// only while a test has switched counting on. Off (the default) the
    /// step path pays one static bool read per readback and never touches
    /// the lock — the same pattern as the engine's signposter gate. Tests
    /// flip it on before stepping and reset it afterwards.
    nonisolated(unsafe) static var countingEnabled = false

    static func recordHostSync() {
        guard countingEnabled else { return }
        lock.lock()
        defer { lock.unlock() }
        _hostSyncs += 1
    }

    static func recordPositionOffsetsHostRebuild() {
        lock.lock()
        defer { lock.unlock() }
        _positionOffsetsHostRebuilds += 1
    }
}

/// Internal hook so `CBv2LayerCache` can hand a row's lazily-mutated storage
/// arrays to the engine loop's `asyncEval` (graph/metadata hygiene: no
/// unconsumed lazy chain may grow O(steps) — DAR-325).
protocol CBv2InnerStateProviding {
    func cbv2InnerState() -> [MLXArray]
}

/// `CBv2SequenceKV` for full (non-windowed) attention.
///
/// Storage is one contiguous `[1, kvHeads, capacity, headDim]` buffer per
/// K and V, grown by doubling (initial capacity = promptLength + 256, capped
/// at `maxLength`). Appends are slice assignments — `mlx_slice_update`
/// donates the input buffer when refcount permits, so an append is O(n), not
/// O(cache). `update` returns temporal-order zero-copy strided views
/// `[..., 0..<retained, :]`; MLX SDPA accepts strided K/V.
public final class CBv2FullSequenceKV: CBv2SequenceKV, CBv2InnerStateProviding {

    /// Extra slots allocated beyond the prompt so the first decode steps
    /// don't immediately grow the buffer.
    static let initialSlack = 256

    public private(set) var absoluteOffset: Int = 0
    public var retainedCount: Int { absoluteOffset }

    /// Hard cap on this sequence's length; growth beyond it is an engine
    /// admission bug and traps.
    public let maxLength: Int

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray?
    private var values: MLXArray?
    /// Fence of the last append a kernel wrote into `keys`/`values` in place
    /// (`commitInPlaceAppend`); nil once the storage arrays are ordered after it.
    private var pendingWrite: MLXArray?
    private var capacity: Int

    /// - Parameters:
    ///   - promptLength: expected prompt length, used to size the initial
    ///     allocation (`promptLength + 256`, capped at `maxLength`).
    ///   - maxLength: maximum total tokens this sequence may ever hold.
    ///   - kvHeads/headDim: from the layer's `CBv2LayerKind`; validated
    ///     against the arrays passed to `update`.
    public init(promptLength: Int, maxLength: Int, kvHeads: Int, headDim: Int) {
        precondition(maxLength > 0, "CBv2FullSequenceKV: maxLength must be > 0")
        precondition(
            promptLength <= maxLength,
            "CBv2FullSequenceKV: promptLength \(promptLength) exceeds maxLength \(maxLength)")
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.capacity = min(maxLength, max(1, promptLength + Self.initialSlack))
    }

    public var byteCount: Int {
        (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
    }

    /// Transfer an exclusively owned, fully authenticated native destination.
    /// No prefix copy or lazy assignment may retain the staging buffers.
    init(
        restoredKeys: MLXArray, restoredValues: MLXArray, offset: Int,
        maxLength: Int, kvHeads: Int, headDim: Int
    ) throws {
        let shape = [1, kvHeads, maxLength, headDim]
        guard maxLength > 0, offset > 0, offset <= maxLength,
            restoredKeys.shape == shape, restoredValues.shape == shape,
            restoredKeys.dtype == restoredValues.dtype
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.capacity = maxLength
        self.absoluteOffset = offset
        self.keys = restoredKeys
        self.values = restoredValues
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1,
            "CBv2FullSequenceKV holds ONE sequence; got batch \(newKeys.dim(0))")
        precondition(newKeys.dim(1) == kvHeads,
            "CBv2FullSequenceKV: kvHeads mismatch (\(newKeys.dim(1)) != \(kvHeads))")
        precondition(newValues.dim(2) == n,
            "CBv2FullSequenceKV: keys/values token count mismatch")
        precondition(
            absoluteOffset + n <= maxLength,
            "CBv2FullSequenceKV: append past maxLength (\(absoluteOffset) + \(n) > \(maxLength)) — admission bug"
        )

        if keys == nil, absoluteOffset == 0,
            let (firstKeys, firstValues) = CBv2KVFirstAppend.apply(
                newKeys, newValues, capacity: min(maxLength, max(capacity, n)))
        {
            capacity = firstKeys.dim(2)
            keys = firstKeys
            values = firstValues
            absoluteOffset = n
            return (firstKeys[.ellipsis, ..<n, 0...], firstValues[.ellipsis, ..<n, 0...])
        }

        orderStorageAfterWrites()
        ensureCapacity(absoluteOffset + n, keyTemplate: newKeys, valueTemplate: newValues)

        let (writtenKeys, writtenValues) = CBv2SqueezedKVUpdate.updates(newKeys, newValues)
        keys![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] = writtenKeys
        values![.ellipsis, absoluteOffset ..< (absoluteOffset + n), 0...] = writtenValues
        absoluteOffset += n

        return (
            keys![.ellipsis, ..<absoluteOffset, 0...],
            values![.ellipsis, ..<absoluteOffset, 0...]
        )
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        guard let keys, let values else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        if let fence = pendingWrite {
            // Views ordered after the last in-place append.
            let views = depends(
                inputs: [
                    keys[.ellipsis, ..<absoluteOffset, 0...],
                    values[.ellipsis, ..<absoluteOffset, 0...],
                ],
                dependencies: [fence])
            return (views[0], views[1], absoluteOffset)
        }
        return (
            keys[.ellipsis, ..<absoluteOffset, 0...],
            values[.ellipsis, ..<absoluteOffset, 0...],
            absoluteOffset
        )
    }

    /// Plain rollback is already value-exact (see `rollback`: the offset
    /// decrement makes the un-confirmed tail structurally unreachable and
    /// the confirmed prefix is untouched), so speculative begin/commit are
    /// the contract's default no-ops.
    public var supportsSpeculativeWrites: Bool { true }

    /// Rollback the last `n` tokens (speculative rejection). The un-confirmed
    /// tail is structurally unreachable afterwards: every view this class
    /// hands out is sliced to `..<absoluteOffset`, and the tail slots are
    /// overwritten by the next `update` before they can ever be exposed —
    /// so no zeroing pass is needed.
    public func rollback(_ n: Int) {
        precondition(n >= 0, "CBv2FullSequenceKV.rollback: negative n")
        precondition(
            n <= absoluteOffset,
            "CBv2FullSequenceKV.rollback(\(n)) exceeds retained \(absoluteOffset)")
        absoluteOffset -= n
    }

    func cbv2InnerState() -> [MLXArray] {
        [keys, values, pendingWrite].compactMap { $0 }
    }

    // MARK: - Append written in place (`CBv2InPlaceKVAppend`)

    /// The storage an `n`-row append written by a kernel goes into: both
    /// buffers (grown exactly as `update` grows them), the first row, and
    /// the previous in-place write's fence (the kernel takes it as an input,
    /// so writes to this row stay in order), or nil when there is no storage
    /// yet or it holds other dtypes. The rows `row ..< row + n` are past
    /// every view this row has handed out at its current offset: the rows
    /// `update` would assign.
    func inPlaceAppendDestination(count n: Int, keyDType: DType, valueDType: DType)
        -> (keys: MLXArray, values: MLXArray, row: Int, previous: MLXArray?)?
    {
        guard n > 0, absoluteOffset + n <= maxLength, let keys, let values,
            keys.dtype == keyDType, values.dtype == valueDType,
            keys.ndim == 4, values.ndim == 4, keys.dim(0) == 1, values.dim(0) == 1,
            keys.dim(1) == kvHeads, values.dim(1) == kvHeads
        else { return nil }
        if absoluteOffset + n > capacity { orderStorageAfterWrites() }
        ensureCapacity(absoluteOffset + n, keyTemplate: self.keys!, valueTemplate: self.values!)
        return (self.keys!, self.values!, absoluteOffset, pendingWrite)
    }

    /// Adopt the `n` rows a kernel wrote at `inPlaceAppendDestination`'s row
    /// (`fence` is an output of that kernel): the views `update` returns,
    /// ordered after the write. The storage arrays stay as they are; the
    /// fence orders every later reader or writer of them
    /// (`orderStorageAfterWrites`, the next kernel's input, `snapshot`).
    func commitInPlaceAppend(count n: Int, fence: MLXArray) -> (MLXArray, MLXArray) {
        pendingWrite = fence
        absoluteOffset += n
        let views = depends(
            inputs: [
                keys![.ellipsis, ..<absoluteOffset, 0...],
                values![.ellipsis, ..<absoluteOffset, 0...],
            ],
            dependencies: [fence])
        return (views[0], views[1])
    }

    /// The storage arrays themselves ordered after the last in-place write,
    /// before an operation that reads or copies them whole.
    private func orderStorageAfterWrites() {
        guard let fence = pendingWrite, let keys, let values else { return }
        let ordered = depends(inputs: [keys, values], dependencies: [fence])
        self.keys = ordered[0]
        self.values = ordered[1]
        pendingWrite = nil
    }

    // MARK: - Private

    private func ensureCapacity(_ needed: Int, keyTemplate: MLXArray, valueTemplate: MLXArray) {
        if keys == nil {
            capacity = min(maxLength, max(capacity, needed))
            keys = MLXArray.zeros(
                [1, kvHeads, capacity, keyTemplate.dim(3)], dtype: keyTemplate.dtype)
            values = MLXArray.zeros(
                [1, kvHeads, capacity, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
            return
        }
        guard needed > capacity else { return }

        // Grow by doubling, capped at maxLength. The concat copies the old
        // buffer once per doubling — amortized O(1) per appended token.
        let newCapacity = min(maxLength, max(capacity * 2, needed))
        let growth = newCapacity - capacity
        keys = concatenated(
            [keys!, MLXArray.zeros([1, kvHeads, growth, keys!.dim(3)], dtype: keys!.dtype)],
            axis: 2)
        values = concatenated(
            [values!, MLXArray.zeros([1, kvHeads, growth, values!.dim(3)], dtype: values!.dtype)],
            axis: 2)
        capacity = newCapacity
    }
}

/// A sequence's first K/V append (its prompt chunk) as ONE launch that
/// allocates both buffers and writes the appended rows, instead of two
/// zero-filled `[1, kvHeads, capacity, headDim]` allocations (MLX `Full`) and
/// two slice updates: three launches and the fill of both buffers fewer per
/// attention layer and prompt. The slots past the appended rows are left
/// unwritten; they are structurally unreachable (every view this class hands
/// out is sliced to `..<absoluteOffset`, and each later append writes its
/// slots before the offset exposes them; see `rollback`), so every value the
/// cache exposes is the same. The kernel copies each element in its own
/// dtype through the operands' strides. Checked once, on first use, bit for
/// bit against the zero-filled slice updates (FP32, FP16, BF16; a
/// head-transposed key view and a contiguous value, then a second append on
/// both). `MLXFAST_KV_FIRST_APPEND=0` keeps the zero-filled allocation.
enum CBv2KVFirstAppend {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_KV_FIRST_APPEND"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "mlxfast_kv_first_append",
        inputNames: ["k", "v", "dims"],
        outputNames: ["ko", "vo"],
        source: """
            const uint d = thread_position_in_grid.x;
            const uint r = thread_position_in_grid.y;
            const uint h = thread_position_in_grid.z;
            const size_t o = (size_t(h) * size_t(dims[0]) + size_t(r)) * size_t(dims[1]) + size_t(d);
            ko[o] = k[int64_t(h) * k_strides[1] + int64_t(r) * k_strides[2] + int64_t(d) * k_strides[3]];
            vo[o] = v[int64_t(h) * v_strides[1] + int64_t(r) * v_strides[2] + int64_t(d) * v_strides[3]];
            """,
        ensureRowContiguous: false)

    private static func launch(_ k: MLXArray, _ v: MLXArray, capacity: Int) -> (MLXArray, MLXArray) {
        let (heads, n, d) = (k.dim(1), k.dim(2), k.dim(3))
        // The capacity and head size are runtime operands, not template
        // constants: one compiled kernel serves every prompt length.
        let outputs = kernel(
            [k, v, MLXArray([Int32(capacity), Int32(d)])],
            grid: (d, n, heads), threadGroup: (min(d, 256), 1, 1),
            outputShapes: [[1, heads, capacity, d], [1, heads, capacity, d]],
            outputDTypes: [k.dtype, v.dtype])
        return (outputs[0], outputs[1])
    }

    static func apply(_ k: MLXArray, _ v: MLXArray, capacity: Int) -> (MLXArray, MLXArray)? {
        guard enabled, k.ndim == 4, v.ndim == 4, k.shape == v.shape, k.dtype == v.dtype,
            k.dim(0) == 1, k.dim(3) <= 1024, capacity >= k.dim(2), k.dim(2) > 0,
            [DType.float32, .float16, .bfloat16].contains(k.dtype), verified
        else { return nil }
        return launch(k, v, capacity: capacity)
    }

    private static let verified: Bool = {
        var same = true
        for dtype in [DType.float32, .float16, .bfloat16] {
            let wide = MLXRandom.normal([1, 20, 7, 64], key: MLXRandom.key(43)).asType(dtype)
            let k = wide[0..., 0..., 1 ..< 5, 0...].transposed(0, 2, 1, 3)
            let v = MLXRandom.normal([1, 4, 20, 64], key: MLXRandom.key(44)).asType(dtype)
            let more = MLXRandom.normal([1, 4, 3, 64], key: MLXRandom.key(45)).asType(dtype)
            var plainK = MLXArray.zeros([1, 4, 32, 64], dtype: dtype)
            var plainV = MLXArray.zeros([1, 4, 32, 64], dtype: dtype)
            plainK[.ellipsis, 0 ..< 20, 0...] = k
            plainV[.ellipsis, 0 ..< 20, 0...] = v
            var (firstK, firstV) = launch(k, v, capacity: 32)
            let bits = dtype == .float32 ? DType.uint32 : .uint16
            func equal(_ a: MLXArray, _ b: MLXArray, _ n: Int) -> Bool {
                all(a[.ellipsis, ..<n, 0...].view(dtype: bits) .== b[.ellipsis, ..<n, 0...].view(dtype: bits))
                    .item(Bool.self)
            }
            same = same && equal(plainK, firstK, 20) && equal(plainV, firstV, 20)
            plainK[.ellipsis, 20 ..< 23, 0...] = more
            plainV[.ellipsis, 20 ..< 23, 0...] = more
            firstK[.ellipsis, 20 ..< 23, 0...] = more
            firstV[.ellipsis, 20 ..< 23, 0...] = more
            same = same && equal(plainK, firstK, 23) && equal(plainV, firstV, 23)
        }
        FileHandle.standardError.write(
            (same
                ? "mlxfast KV first append: self-test passed (3 dtypes bitwise); one launch\n"
                : "mlxfast KV first append: mismatch; zero-filled allocation kept\n")
                .data(using: .utf8)!)
        return same
    }()
}

/// The KV append's updates with the batch axis squeezed (a view) before the
/// slice assignment. The assignment drops leading singleton axes itself by a
/// reshape, and MLX's reshape copies a strided input whose first axis is 1
/// (`prepare_reshape` keeps that axis when collapsing): the head-transposed
/// values of a verify window (a column slice of the q|k|v product) took one
/// copy launch per attention layer before the slice update. The squeeze is a
/// view and the slice update reads the same elements through their strides.
/// Checked once, on first use, bit for bit against the unsqueezed assignment
/// on a head-transposed column slice; `MLXFAST_KV_SQUEEZED_UPDATE=0` keeps the
/// unsqueezed updates.
enum CBv2SqueezedKVUpdate {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_KV_SQUEEZED_UPDATE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let verified: Bool = {
        var same = true
        for dtype in [DType.float32, .float16, .bfloat16] {
            let wide = MLXRandom.normal([1, 16, 14336], key: MLXRandom.key(41)).asType(dtype)
            let update = wide[0..., 0..., 13312...].reshaped(1, 16, 4, 256).transposed(0, 2, 1, 3)
            var plain = MLXArray.zeros([1, 4, 64, 256], dtype: dtype)
            var squeezed = MLXArray.zeros([1, 4, 64, 256], dtype: dtype)
            plain[.ellipsis, 5 ..< 21, 0...] = update
            squeezed[.ellipsis, 5 ..< 21, 0...] = update.squeezed(axis: 0)
            let bits = dtype == .float32 ? DType.uint32 : .uint16
            same = same
                && all(plain.view(dtype: bits) .== squeezed.view(dtype: bits)).item(Bool.self)
        }
        FileHandle.standardError.write(
            (same
                ? "mlxfast squeezed KV update: self-test passed (3 dtypes bitwise); squeezed\n"
                : "mlxfast squeezed KV update: mismatch; unsqueezed updates kept\n")
                .data(using: .utf8)!)
        return same
    }()

    static func updates(_ keys: MLXArray, _ values: MLXArray) -> (MLXArray, MLXArray) {
        guard enabled, keys.ndim == 4, values.ndim == 4, keys.dim(0) == 1, values.dim(0) == 1,
            verified
        else { return (keys, values) }
        return (keys.squeezed(axis: 0), values.squeezed(axis: 0))
    }
}
