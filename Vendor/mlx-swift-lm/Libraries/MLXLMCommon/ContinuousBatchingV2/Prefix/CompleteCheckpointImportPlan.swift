import Foundation
import MLX

/// Plan without native tensor allocation. Its retained manifest has a host
/// metadata owner. With a shared process owner, native destination and
/// scratch belong to Admission; the provider reserves its host I/O only. Without
/// one, the provider also maintains its historical global envelope. The callback
/// remains attached until cancellation or transfer to charged request storage.
public final class CBv2CompleteCheckpointImportPlan: @unchecked Sendable {
    public let manifest: CBv2CompleteCheckpointManifest
    public let maximumSequenceLength: Int
    public let nativeDestinationBytes: Int
    public let usesProcessMemoryOwner: Bool
    let pagedStoragePlan: CBv2PagedCheckpointStoragePlan?
    let auxiliaryBytes: Int
    public let scratchBytes: Int
    let codec: CBv2CompleteCheckpointCodec
    private(set) var destinationShapes: [[Int]]
    /// Internal deterministic allocation-fault seam. Production uses scoped
    /// MLX error collection; no failed allocation reaches a raw pointer read.
    var evaluateDestinations: ([MLXArray]) throws -> Void = { arrays in
        try withError { eval(arrays) }
    }

    init(
        codec: CBv2CompleteCheckpointCodec, manifest: CBv2CompleteCheckpointManifest,
        maximumSequenceLength: Int
    ) throws {
        self.codec = codec
        self.manifest = try manifest.owningMetadata(admission: codec.admission)
        self.maximumSequenceLength = maximumSequenceLength
        self.usesProcessMemoryOwner = codec.admission.hasProcessMemoryOwner
        let paged = try codec.pagedConfig.map {
            try CBv2PagedCheckpointStoragePlan(layerKinds: codec.layerKinds, config: $0, position: manifest.position,
                historicalLayout: codec.historicalLayout, maximumSequenceLength: maximumSequenceLength)
        }
        self.pagedStoragePlan = paged
        let destinations = paged == nil ? manifest.tensors
            : Array(manifest.tensors.dropFirst(codec.targetTensorCount))
        var shapes: [[Int]] = []
        var bytes = 0, initializationScratch = 0
        for tensor in destinations {
            var shape = tensor.shape
            if tensor.role == .keys || tensor.role == .values { shape[2] = maximumSequenceLength }
            let count = try CBv2CheckpointTensorDescriptor.checkedByteCount(shape: shape, dtype: tensor.dtype.mlxDType)
            bytes = try CBv2CheckpointAllocationFootprint.add(bytes, CBv2CheckpointAllocationFootprint.bound(count))
            // MLX zeros constructs one same-dtype scalar before its Full
            // destination. These batch-created scalars coexist until eval.
            initializationScratch = try CBv2CheckpointAllocationFootprint.add(
                initializationScratch, CBv2CheckpointAllocationFootprint.bound(tensor.dtype.mlxDType.size))
            shapes.append(shape)
        }
        self.destinationShapes = shapes
        // Native page buffers are initialized and evaluated sequentially,
        // before constructing the auxiliary batch. Only one page-fill scalar
        // overlaps at a time; provider Data has its own I/O owner.
        var pageInitializationScratch = 0
        for group in paged?.groups ?? [] {
            pageInitializationScratch = max(pageInitializationScratch,
                try CBv2CheckpointAllocationFootprint.bound(group.key.dtype.size))
        }
        self.scratchBytes = max(initializationScratch, pageInitializationScratch)
        self.auxiliaryBytes = paged == nil ? 0 : bytes
        let (total, overflow) = bytes.addingReportingOverflow(paged?.nativeBytes ?? 0)
        guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        self.nativeDestinationBytes = total
    }

    public func allocate(onRelease: @escaping @Sendable () -> Void) throws -> CBv2CompleteCheckpointImport {
        let providerReservation = CBv2CheckpointReservation(onRelease: onRelease)
        do {
            let scratch = try CBv2CheckpointAllocationFootprint.add(scratchBytes,
                usesProcessMemoryOwner ? 0 : CBv2CompleteCheckpointManifest.maximumProviderScratchBytes)
            if let pagedStoragePlan {
                let stage = try codec.admission.reserveCheckpointStage(
                    targetBytes: pagedStoragePlan.nativeBytes, auxiliaryBytes: auxiliaryBytes, scratchBytes: scratch)
                do {
                    return try .init(plan: self, reservation: providerReservation, stageLease: stage)
                } catch {
                    // Failed initializer locals and segment owners have
                    // unwound before this native charge is returned.
                    stage.closeAfterDroppingOwners()
                    throw error
                }
            }
            let (bytes, overflow) = nativeDestinationBytes.addingReportingOverflow(scratch)
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            let engineReservation = try codec.admission.reserveTransient(bytes: bytes)
            let reservation = CBv2CheckpointReservation {
                engineReservation.release()
                providerReservation.release()
            }
            return try .init(plan: self, reservation: reservation)
        } catch {
            providerReservation.release()
            throw error
        }
    }

    deinit {
        // Auxiliary shapes may alias descriptor arrays. Retire those aliases
        // before automatic destruction can release the manifest's permit.
        destinationShapes.removeAll(keepingCapacity: false)
    }
}
