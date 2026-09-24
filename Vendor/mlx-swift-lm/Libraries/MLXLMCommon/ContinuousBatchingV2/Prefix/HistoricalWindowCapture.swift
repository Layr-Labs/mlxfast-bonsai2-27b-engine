import Cmlx
import Foundation
import MLX

/// One immutable [2,1,H,W,D] output captured before any successor graph writes
/// its donor ring. No materialization credit is inferred from aliases/nbytes.
/// Its conservative native allocation charge survives all export K/V sources.
final class CBv2HistoricalWindow: @unchecked Sendable {
    let start: Int
    let position: Int
    let heads: Int
    let headDim: Int
    let dtype: DType
    private var combined: MLXArray?
    private var reservation: CBv2CheckpointReservation?
    private enum Evaluation { case pending, ready, failed }
    private var evaluation = Evaluation.pending
    private var submitted = false
    private var drained = false
    // Snapshot the task-local default once. Retirement may run outside its
    // construction scope; neither a later default nor global GPU/CPU is ours.
    let copyStream: StreamOrDevice
    private let synchronize: (StreamOrDevice) throws -> Void
    private let evaluate: (MLXArray) throws -> Void
    var evaluationRoot: MLXArray? { combined }

    static func reservationBytes(row: PagedSequenceKV, position: Int) throws -> Int {
        guard let window = row.windowSize, row.pool.segmentGrant != nil,
              position == row.absoluteOffset, position > 1 else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let count = min(window, position)
        guard position - count >= row.oldestValidPosition else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let bytes = try CBv2CheckpointTensorDescriptor.checkedByteCount(
            shape: [2, 1, row.groupKey.kvHeads, count, row.groupKey.headDim], dtype: row.groupKey.dtype)
        let pageSpan = (count + row.pool.config.pageSize - 2) / row.pool.config.pageSize + 1
        let segments = min(pageSpan, row.pool.group(row.groupKey).segments.count)
        func bound(_ bytes: Int) throws -> Int { try Memory.allocationFootprintUpperBound(byteCount: bytes) }
        // Each bucket has <=count records. Bound separately because cached
        // buffers may be larger than logical sizes; a total-byte bound is wrong.
        let output = try bound(bytes)
        let records = try bound(max(24, count * 3) * MemoryLayout<Int32>.stride)
        let fence = try bound(MemoryLayout<Int32>.stride)
        // The zeros scalar and prospective completion witness are distinct.
        let scalar = try bound(row.groupKey.dtype.size)
        let host = (64 << 10) + 4 * (count * 4 * MemoryLayout<Int32>.stride + pageSpan * MemoryLayout<Int32>.stride)
        let (transfer, multiplyOverflow) = segments.multipliedReportingOverflow(by: records + fence)
        let (total, overflow) = transfer.addingReportingOverflow(output + scalar + fence + host)
        guard !multiplyOverflow, !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        return total
    }

    init(row: PagedSequenceKV, position: Int, admission: AdmissionV2,
         stream: StreamOrDevice = .default,
         beforeAllocation: () throws -> Void = {},
         afterConstruction: (MLXArray) throws -> Void = { _ in },
         evaluate: @escaping (MLXArray) throws -> Void = { array in try withError { eval(array) } },
         synchronize: @escaping (StreamOrDevice) throws -> Void = { stream in
             try withError { stream.stream.synchronize() }
         }) throws {
        let bytes = try Self.reservationBytes(row: row, position: position)
        let permit = try admission.reserveTransient(bytes: bytes)
        self.position = position
        start = max(0, position - row.windowSize!)
        heads = row.groupKey.kvHeads
        headDim = row.groupKey.headDim
        dtype = row.groupKey.dtype
        reservation = permit
        self.evaluate = evaluate
        copyStream = stream
        self.synchronize = synchronize
        do {
            // No per-capture page list, host records or lazy GPU graph exists
            // before the permit. Production never calls the fault seam.
            try beforeAllocation()
            try withError {
                let pageSize = row.pool.config.pageSize
                let pages = (start / pageSize ... (position - 1) / pageSize).map {
                    row.table[$0 % row.ringPages!]
                }
                combined = PagedSegmentTransfers.gatherCombined(
                    group: row.pool.group(row.groupKey), pages: pages,
                    firstSlot: start % pageSize, count: position - start, publishReadFence: false,
                    stream: copyStream)
                if let combined { try afterConstruction(combined) }
            }
        } catch {
            // Construction submits no copy work and never publishes a group
            // fence. Private graph locals have unwound; no serving alias can
            // retain this destination after the output is dropped.
            evaluation = .failed
            combined = nil
            reservation = nil
            throw error
        }
    }

    /// Called by the owning step before any successor may build cache writes.
    /// A failed private copy cannot poison the target's serving write fence.
    func markSubmitted() { submitted = true }

    func finishEvaluation() throws {
        switch evaluation {
        case .ready: return
        case .failed: throw CBv2CompleteCheckpointError.allocationFailed
        case .pending: break
        }
        guard let combined else { throw CBv2CompleteCheckpointError.closed }
        do {
            submitted = true
            try evaluate(combined)
            evaluation = .ready
        } catch {
            // asyncEval may already have submitted part of the private copy.
            // Drain issued work; never retry the failed graph during retirement.
            var failure = error
            do {
                try synchronize(copyStream)
                drained = true
            } catch { failure = error }
            evaluation = .failed
            throw failure
        }
    }

    func read(values: Bool, byteOffset: Int, maximumBytes: Int) throws -> Data {
        guard let combined else { throw CBv2CompleteCheckpointError.closed }
        let bytes = heads * (position - start) * headDim * dtype.size
        guard byteOffset >= 0, byteOffset < bytes, byteOffset % dtype.size == 0,
              maximumBytes >= dtype.size, maximumBytes <= CBv2CompleteCheckpointManifest.maximumSegmentBytes
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        // Launch submitted the completion edge on the engine's sole evaluator.
        // Readback here only waits for that already detached immutable output.
        try finishEvaluation()
        guard let info = try combined.evaluatedBufferInfo(), info.isRowContiguous,
              info.dataElements == combined.size,
              let pointer = mlx_array_data_uint8(combined.ctx)
        else { throw CBv2CompleteCheckpointError.allocationFailed }
        let count = min(maximumBytes - maximumBytes % dtype.size, bytes - byteOffset)
        return Data(bytes: UnsafeRawPointer(pointer).advanced(by: (values ? bytes : 0) + byteOffset), count: count)
    }

    deinit {
        // eval() can return before Metal completion handlers drop their data
        // references. Keep the output and permit through that final retirement
        // even for a successful .ready capture; reads need no additional wait.
        if submitted && !drained {
            // Drain only submitted work, never evaluate/retry the graph. An
            // evaluation failure already drained successfully is not waited twice.
            try? synchronize(copyStream)
        }
        combined = nil
        reservation?.release()
        reservation = nil
    }
}

final class CBv2HistoricalWindowTensorSource {
    private var window: CBv2HistoricalWindow?
    private let values: Bool
    init(window: CBv2HistoricalWindow, values: Bool) { self.window = window; self.values = values }

    func matches(_ descriptor: CBv2CheckpointTensorDescriptor) -> Bool {
        guard let window else { return false }
        return descriptor.role == (values ? .values : .keys)
            && descriptor.dtype.mlxDType == window.dtype
            && descriptor.shape == [1, window.heads, window.position - window.start, window.headDim]
    }

    func readSegment(byteOffset: Int, maximumBytes: Int) throws -> Data {
        guard let window else { throw CBv2CompleteCheckpointError.closed }
        return try window.read(values: values, byteOffset: byteOffset, maximumBytes: maximumBytes)
    }

    func close() { window = nil }
}
