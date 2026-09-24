import Foundation

/// Host manifest storage is separate from GPU packing and provider I/O. Value
/// copies of a manifest share this owner, including after an export is closed.
final class CBv2CheckpointManifestMemory: @unchecked Sendable {
    private(set) var tokens: [Int]
    private(set) var tensors: [CBv2CheckpointTensorDescriptor]
    private(set) var attentionLayers: [CBv2CheckpointAttentionLayer]?
    let permit: Permit?

    final class Permit: @unchecked Sendable {
        let admission: ObjectIdentifier
        let bytes: Int
        private let reservation: CBv2CheckpointReservation

        init(admission: AdmissionV2, position: Int) throws {
            self.bytes = try CBv2CheckpointManifestMemory.reservationBytes(position: position)
            self.admission = ObjectIdentifier(admission)
            self.reservation = try admission.reserveTransient(bytes: bytes)
        }
    }

    /// A conservative host allocation envelope, NOT exact stdlib heap usage.
    /// Token capacity gets 2x its element bytes. All 4096 allowed descriptors
    /// get their value, eight shape dimensions, 128 bytes of per-entry allocation
    /// allowance, and 2x capacity. 64 KiB covers scalar/string/control metadata.
    /// These limits bound even the pre-validation construction working set.
    static func reservationBytes(position: Int) throws -> Int {
        guard position > 1, position <= CBv2CompleteCheckpointManifest.maximumEncodedBytes / 2 else {
            throw CBv2CompleteCheckpointError.invalidManifest
        }
        let descriptor = MemoryLayout<CBv2CheckpointTensorDescriptor>.stride
            + 8 * MemoryLayout<Int>.stride + 128
        return 2 * position * MemoryLayout<Int>.stride + 2 * 4096 * descriptor + 2 * 2048 * MemoryLayout<CBv2CheckpointAttentionLayer>.stride + (64 << 10)
    }

    init(tokens: [Int], tensors: [CBv2CheckpointTensorDescriptor],
         attentionLayers: [CBv2CheckpointAttentionLayer]? = nil, permit: Permit? = nil) {
        self.tokens = tokens
        self.tensors = tensors
        self.attentionLayers = attentionLayers
        self.permit = permit
    }

    deinit {
        // A permit may also be held by construction locals; the final permit
        // cannot retire until the actual stored arrays have been dropped here.
        tokens.removeAll(keepingCapacity: false)
        tensors.removeAll(keepingCapacity: false)
        attentionLayers = nil
    }
}

extension CBv2CompleteCheckpointManifest {
    /// The provider still holds its I/O envelope during this ownership handoff.
    /// Replanning on the same loaded engine reuses the existing native owner.
    func owningMetadata(admission: AdmissionV2) throws -> Self {
        if metadata.permit?.admission == ObjectIdentifier(admission) { return self }
        let permit = try CBv2CheckpointManifestMemory.Permit(admission: admission, position: position)
        return replacingMetadata(.init(tokens: prefixTokens, tensors: tensors, attentionLayers: attentionLayers, permit: permit))
    }
}
