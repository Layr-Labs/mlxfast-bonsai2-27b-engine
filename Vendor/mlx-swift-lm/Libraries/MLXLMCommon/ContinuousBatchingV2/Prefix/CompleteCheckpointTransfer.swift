import Cmlx
import Foundation
import MLX

/// No-copy source views stay private and immutable until close. Packing reads
/// only one bounded segment, including when KV has strided donor-capacity gaps.
public final class CBv2CompleteCheckpointExport: @unchecked Sendable {
    public let manifest: CBv2CompleteCheckpointManifest
    /// Shared process mode: the provider owns only its host I/O buffers;
    /// Admission already owns native packing scratch and donor backing.
    public let usesProcessMemoryOwner: Bool
    private let lock = NSLock()
    private var sources: [CBv2CompleteCheckpointTensorSource]?

    convenience init(manifest: CBv2CompleteCheckpointManifest, arrays: [MLXArray],
                     usesProcessMemoryOwner: Bool = false) {
        self.init(manifest: manifest, sources: arrays.map { .array($0) },
                  usesProcessMemoryOwner: usesProcessMemoryOwner)
    }

    init(manifest: CBv2CompleteCheckpointManifest, sources: [CBv2CompleteCheckpointTensorSource],
         usesProcessMemoryOwner: Bool = false) {
        self.manifest = manifest
        self.sources = sources
        self.usesProcessMemoryOwner = usesProcessMemoryOwner
    }

    public func readSegment(tensorIndex: Int, byteOffset: Int, maximumBytes: Int) throws -> Data {
        try lock.withLock {
            guard let sources else { throw CBv2CompleteCheckpointError.closed }
            guard sources.indices.contains(tensorIndex) else { throw CBv2CompleteCheckpointError.invalidSegment }
            return try sources[tensorIndex].readSegment(
                descriptor: manifest.tensors[tensorIndex], byteOffset: byteOffset, maximumBytes: maximumBytes)
        }
    }

    public func close() {
        lock.withLock {
            sources?.forEach { $0.close() }
            sources = nil
        }
    }
}

/// Filled on one staging queue, before any array is exposed to GPU consumers.
/// Each zeroed MLX destination is exclusively owned here. Raw writes cannot
/// race lazy graphs: allocation is evaluated once, then only authenticated CPU
/// segments touch the buffers until finish transfers their ownership.
public final class CBv2CompleteCheckpointImport: @unchecked Sendable {
    private let lock = NSLock()
    private let plan: CBv2CompleteCheckpointImportPlan
    private var arrays: [MLXArray]?
    private var pagedStorage: CBv2PagedCheckpointStorage?
    private var stageLease: CBv2CheckpointStageLease?
    private var reservation: CBv2CheckpointReservation?
    private var tensorIndex = 0
    private var byteOffset = 0
    private var nativeDestinationBytes = 0

    init(plan: CBv2CompleteCheckpointImportPlan, reservation: CBv2CheckpointReservation,
         stageLease: CBv2CheckpointStageLease? = nil) throws {
        self.plan = plan
        self.reservation = reservation
        self.stageLease = stageLease
        if let storagePlan = plan.pagedStoragePlan {
            pagedStorage = try CBv2PagedCheckpointStorage(
                plan: storagePlan, evaluate: { try plan.evaluateDestinations([$0]) }, admission: plan.codec.admission)
        }
        let destinationDescriptors = plan.pagedStoragePlan == nil ? plan.manifest.tensors
            : Array(plan.manifest.tensors.dropFirst(plan.codec.targetTensorCount))
        let allocationStream = StreamOrDevice.default
        let destinations = try withError { fault in
            let values = zip(plan.destinationShapes, destinationDescriptors).map { shape, tensor in
                MLXArray.zeros(shape, dtype: tensor.dtype.mlxDType, stream: allocationStream)
            }
            do {
                try fault.check()
                try plan.evaluateDestinations(values)
            } catch {
                // Drain partial work before its stage charge can be returned.
                allocationStream.stream.synchronize()
                if error is MLXError { throw error }
                try fault.check()
                throw error
            }
            // Evaluation can signal before Metal completion drops temporary
            // Data references. Fence once before checking exclusive ownership.
            allocationStream.stream.synchronize()
            try fault.check()
            return values
        }
        guard destinations.allSatisfy({ mlx_array_data_uint8($0.ctx) != nil }) else {
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        let footprint = try CBv2CheckpointAllocationFootprint.freshBytes(destinations)
        nativeDestinationBytes = try CBv2CheckpointAllocationFootprint.add(
            footprint.actual, pagedStorage?.allocatedBytes ?? 0)
        arrays = destinations
    }

    public func appendSegment(tensorIndex: Int, byteOffset: Int, data: Data) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let arrays else { throw CBv2CompleteCheckpointError.closed }
        guard tensorIndex == self.tensorIndex, byteOffset == self.byteOffset,
            plan.manifest.tensors.indices.contains(tensorIndex), !data.isEmpty,
            data.count <= CBv2CompleteCheckpointManifest.maximumSegmentBytes
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        let descriptor = plan.manifest.tensors[tensorIndex]
        let itemSize = descriptor.dtype.mlxDType.size
        guard data.count % itemSize == 0, data.count <= descriptor.byteCount - byteOffset else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let kvTensorCount = plan.codec.targetTensorCount
        if let pagedStorage, tensorIndex < kvTensorCount {
            try pagedStorage.append(layerIndex: tensorIndex / 2, values: tensorIndex % 2 == 1,
                                    byteOffset: byteOffset, data: data)
        } else {
            let index = pagedStorage == nil ? tensorIndex : tensorIndex - kvTensorCount
            guard let pointer = mlx_array_data_uint8(arrays[index].ctx) else {
                throw CBv2CompleteCheckpointError.allocationFailed
            }
            // Fresh, evaluated, unpublished destinations. This writable
            // pointer never escapes or survives the move performed by finish.
            let destination = UnsafeMutableRawPointer(mutating: pointer)
            let strides = CBv2CheckpointByteLayout.contiguousStrides(plan.destinationShapes[index])
            data.withUnsafeBytes { source in
                CBv2CheckpointByteLayout.copy(
                    shape: descriptor.shape, strides: strides, itemSize: itemSize,
                    byteOffset: byteOffset, count: data.count
                ) { physicalOffset, packedOffset, length in
                    destination.advanced(by: physicalOffset).copyMemory(
                        from: source.baseAddress!.advanced(by: packedOffset), byteCount: length)
                }
            }
        }
        self.byteOffset += data.count
        if self.byteOffset == descriptor.byteCount {
            self.tensorIndex += 1
            self.byteOffset = 0
        }
    }

    public func finish() throws -> CBv2StagedCompleteCheckpoint {
        lock.lock()
        defer { lock.unlock() }
        guard let arrays, let reservation else { throw CBv2CompleteCheckpointError.closed }
        guard tensorIndex == plan.manifest.tensors.count, byteOffset == 0 else {
            throw CBv2CompleteCheckpointError.incompleteTransfer
        }
        let prepared: CBv2PreparedCompleteCheckpoint
        if let pagedStorage, let stageLease {
            prepared = .init(pagedFrame: try .init(storage: pagedStorage, auxiliary: arrays, lease: stageLease))
        } else {
            prepared = try plan.codec.preparedState(
                manifest: plan.manifest, arrays: arrays, maximumSequenceLength: plan.maximumSequenceLength)
        }
        let result = CBv2StagedCompleteCheckpoint(plan: plan, prepared: prepared,
            nativeDestinationBytes: nativeDestinationBytes, reservation: reservation)
        self.arrays = nil
        self.pagedStorage = nil
        self.stageLease = nil
        self.reservation = nil
        return result
    }

    public func close() {
        lock.lock()
        arrays = nil
        pagedStorage?.close()
        pagedStorage = nil
        let stageLease = self.stageLease
        self.stageLease = nil
        let reservation = self.reservation
        self.reservation = nil
        lock.unlock()
        stageLease?.closeAfterDroppingOwners()
        reservation?.release()
    }

    deinit { close() }
}

/// Single-use request state, not a cache entry. close drops buffers before its
/// provider reservation. Successful engine adoption releases that reservation
/// only after both active admission and the backend have charged ownership.
public final class CBv2StagedCompleteCheckpoint: @unchecked Sendable {
    public let manifest: CBv2CompleteCheckpointManifest
    public let maximumSequenceLength: Int
    public let nativeDestinationBytes: Int
    let codec: CBv2CompleteCheckpointCodec
    var usesPagedBacking: Bool { codec.pagedConfig != nil }
    private let lock = NSLock()
    private var prepared: CBv2PreparedCompleteCheckpoint?
    private var reservation: CBv2CheckpointReservation?

    init(
        plan: CBv2CompleteCheckpointImportPlan, prepared: CBv2PreparedCompleteCheckpoint,
        nativeDestinationBytes: Int,
        reservation: CBv2CheckpointReservation
    ) {
        self.manifest = plan.manifest
        self.maximumSequenceLength = plan.maximumSequenceLength
        self.nativeDestinationBytes = nativeDestinationBytes
        self.codec = plan.codec
        self.prepared = prepared
        self.reservation = reservation
    }

    func consumePreparedState<Result>(
        _ adopt: (CBv2PreparedCompleteCheckpoint) throws -> Result
    ) throws -> Result {
        lock.lock()
        guard let prepared else {
            lock.unlock()
            throw CBv2CompleteCheckpointError.closed
        }
        // Move the payload out before work. Concurrent close sees an empty
        // handle and cannot refund a destination while adoption consumes it.
        self.prepared = nil
        let reservation = self.reservation
        self.reservation = nil
        lock.unlock()
        defer {
            prepared.clear()
            reservation?.release()
        }
        return try adopt(prepared)
    }

    public func close() {
        lock.lock()
        prepared?.clear()
        prepared = nil
        let reservation = self.reservation
        self.reservation = nil
        lock.unlock()
        reservation?.release()
    }

    deinit { close() }
}

/// Slot-owned scratch for the initial encrypted manifest read. Import-plan
/// allocation later reserves its own native destinations and transfer scratch;
/// callers release this lease only after that reservation succeeds, or after
/// abandoning the read. There are no tensor allocations in this owner.
public final class CBv2CompleteCheckpointIOLease: @unchecked Sendable {
    /// A bound native owner excludes provider IO; the caller must supply its
    /// own process host-buffer reservation before reading the manifest.
    public let usesProcessMemoryOwner: Bool
    private let reservation: CBv2CheckpointReservation

    init(reservation: CBv2CheckpointReservation, usesProcessMemoryOwner: Bool = false) {
        self.reservation = reservation
        self.usesProcessMemoryOwner = usesProcessMemoryOwner
    }

    public func close() { reservation.release() }
    deinit { close() }
}

final class CBv2CheckpointReservation: @unchecked Sendable {
    private let lock = NSLock()
    private var onRelease: (@Sendable () -> Void)?
    init(onRelease: @escaping @Sendable () -> Void) { self.onRelease = onRelease }
    func release() {
        lock.lock()
        let callback = onRelease
        onRelease = nil
        lock.unlock()
        callback?()
    }
    deinit { release() }
}

/// Map a logical packed span onto contiguous runs in strided native storage.
/// All products were checked while validating the descriptor/allocation plan.
enum CBv2CheckpointByteLayout {
    static func contiguousStrides(_ shape: [Int]) -> [Int] {
        var result = Array(repeating: 1, count: shape.count)
        for index in stride(from: shape.count - 2, through: 0, by: -1) {
            result[index] = result[index + 1] * shape[index + 1]
        }
        return result
    }

    static func copy(
        shape: [Int], strides: [Int], itemSize: Int, byteOffset: Int, count: Int,
        run: (Int, Int, Int) -> Void
    ) {
        var contiguousElements = 1
        for index in shape.indices.reversed() {
            if shape[index] == 1 { continue }
            guard strides[index] == contiguousElements else { break }
            contiguousElements *= shape[index]
        }
        var logicalElement = byteOffset / itemSize
        var copied = 0
        while copied < count {
            var remainder = logicalElement
            var physicalElement = 0
            for index in shape.indices.reversed() {
                physicalElement += (remainder % shape[index]) * strides[index]
                remainder /= shape[index]
            }
            let available = (contiguousElements - logicalElement % contiguousElements) * itemSize
            let bytes = min(available, count - copied)
            run(physicalElement * itemSize, copied, bytes)
            copied += bytes
            logicalElement += bytes / itemSize
        }
    }
}
