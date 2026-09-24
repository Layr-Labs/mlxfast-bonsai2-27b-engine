import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Typed checkpoint admission transfer", .serialized)
struct CBv2CheckpointAdmissionTransferTests {
    private func ledger(bytes: Int, auxiliary: Bool = true) -> AdmissionV2 {
        let kind = CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        return AdmissionV2(layerKinds: [kind], bytesCapacity: bytes, config: .init(
            watermarkFraction: 0, elementBytes: 2, layerElementBytes: nil,
            fixedBytesPerRequest: auxiliary ? 8 : 0,
            auxiliaryBytesPerToken: auxiliary ? 2 : 0,
            auxiliaryTokenGranularity: 16, auxiliaryTokenAllocationPadding: 3),
            residency: CBv2PagedKVResidency(config: .init(
                pageSize: 16, capacityBytes: bytes, maxBufferLength: 1 << 20)))
    }

    @Test func exactFitTransfersOnlyDestinationAndKeepsOtherOwnersAndScratch() throws {
        let admission = ledger(bytes: 536)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 256)
        defer { physical.close() }
        try admission.reserve(id: .init(1), additionalTokens: 16)
        let stage = try admission.reserveCheckpointStage(targetBytes: 128, auxiliaryBytes: 72, scratchBytes: 8)
        #expect(admission.bytesReserved == 536)
        let ticket = try physical.transferCheckpoint(to: 384, admission: admission) { previous in
            try admission.transferCheckpointStage(stage, requestID: .init(2), maximumTokens: 16,
                previousPhysicalBytes: previous, physicalBytes: 384)
        }
        ticket.commit()
        #expect(admission.bytesReserved == 536 && physical.bytes == 384)
        #expect(throws: CBv2KVError.self) { try admission.reserveTransient(bytes: 1) }
        stage.closeAfterDroppingOwners()
        stage.closeAfterDroppingOwners()
        #expect(admission.bytesReserved == 528) // Only 8 scratch bytes left the stage.
        admission.releaseAll(id: .init(2))
        #expect(admission.bytesReserved == 456) // Existing request's 72 aux remain.
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 384)
        physical.release(to: 0)
        #expect(admission.bytesReserved == 0)
    }

    @Test func oneExtraPhysicalByteRefusesWithoutConsumingStage() throws {
        let admission = ledger(bytes: 208)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 0)
        defer { physical.close() }
        let stage = try admission.reserveCheckpointStage(targetBytes: 128, auxiliaryBytes: 72, scratchBytes: 8)
        #expect(throws: CBv2KVError.self) {
            try physical.transferCheckpoint(to: 129, admission: admission) { previous in
                try admission.transferCheckpointStage(stage, requestID: .init(1), maximumTokens: 16,
                    previousPhysicalBytes: previous, physicalBytes: 129)
            }
        }
        #expect(admission.bytesReserved == 208 && physical.bytes == 0)
        let ticket = try physical.transferCheckpoint(to: 128, admission: admission) { previous in
            try admission.transferCheckpointStage(stage, requestID: .init(1), maximumTokens: 16,
                previousPhysicalBytes: previous, physicalBytes: 128)
        }
        // No arrays exist in this accounting-only test. Rollback discards the
        // transferred destination; it does not resurrect an externally mutable stage.
        ticket.rollbackAfterDroppingOwners()
        ticket.rollbackAfterDroppingOwners()
        #expect(admission.bytesReserved == 8 && physical.bytes == 0)
        stage.closeAfterDroppingOwners()
        #expect(admission.bytesReserved == 0)
    }

    @Test func explicitConsumptionStateRejectsZeroDestinationReuseAndForeignPool() throws {
        let admission = ledger(bytes: 1024, auxiliary: false)
        let other = ledger(bytes: 1024, auxiliary: false)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 0)
        let foreign = other.bindBackendPhysicalFloor(initialBytes: 0)
        defer { physical.close(); foreign.close() }
        let stage = try admission.reserveCheckpointStage(targetBytes: 0, auxiliaryBytes: 0, scratchBytes: 8)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try foreign.transferCheckpoint(to: 0, admission: admission) { previous in
                try admission.transferCheckpointStage(stage, requestID: .init(1), maximumTokens: 16,
                    previousPhysicalBytes: previous, physicalBytes: 0)
            }
        }
        let ticket = try physical.transferCheckpoint(to: 0, admission: admission) { previous in
            try admission.transferCheckpointStage(stage, requestID: .init(1), maximumTokens: 16,
                previousPhysicalBytes: previous, physicalBytes: 0)
        }
        ticket.commit()
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try physical.transferCheckpoint(to: 0, admission: admission) { previous in
                try admission.transferCheckpointStage(stage, requestID: .init(2), maximumTokens: 16,
                    previousPhysicalBytes: previous, physicalBytes: 0)
            }
        }
        stage.closeAfterDroppingOwners()
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 0 && other.bytesReserved == 0)
    }

    @Test func paddedAuxiliaryTransfersAtomicallyAndStaysOutOfTargetTelemetry() throws {
        let admission = ledger(bytes: 209)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 0)
        defer { physical.close() }
        let stage = try admission.reserveCheckpointStage(targetBytes: 128, auxiliaryBytes: 73, scratchBytes: 8)
        let id = CBv2RequestID(1)
        let ticket = try physical.transferCheckpoint(to: 128, admission: admission) { previous in
            try admission.transferCheckpointStage(stage, requestID: id, maximumTokens: 16,
                previousPhysicalBytes: previous, physicalBytes: 128)
        }
        ticket.commit()
        #expect(admission.bytesReserved == 209 && physical.bytes == 128)
        #expect(admission.nonBackendBytesReserved == (128 - 64) + 73 + 8) // Physical overhead + auxiliary + scratch.
        let target = admission.targetBytesReserved(partitionedBy: [id])
        #expect(target.materialized == 64 && target.unmaterialized == 0)
        #expect(throws: CBv2KVError.self) { try admission.reserveTransient(bytes: 1) }
        stage.closeAfterDroppingOwners()
        let detached = admission.detachReservation(id: id)
        #expect(admission.nonBackendBytesReserved == (128 - 64) + 73)
        physical.release(to: 0)
        admission.updateBytesCapacity(1024)
        try admission.reserve(id: id, additionalTokens: 1)
        let beforeStaleClose = admission.bytesReserved
        stage.closeAfterDroppingOwners()
        admission.releaseCheckpointRequest(id: id, ownerIdentity: stage.identity)
        #expect(admission.bytesReserved == beforeStaleClose, "old generation cannot release a reused ID")
        detached.release()
        detached.release()
        #expect(admission.bytesReserved == admission.allocatedBytes(forTokens: 1))
        #expect(admission.nonBackendBytesReserved == 40)
        admission.releaseAll(id: id)
        #expect(admission.bytesReserved == 0 && admission.nonBackendBytesReserved == 0)
    }

    @Test func preemptionReleasesAuxiliaryPartitionBeforeIDReuse() throws {
        let admission = ledger(bytes: 1024)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 0)
        defer { physical.close() }
        let stage = try admission.reserveCheckpointStage(targetBytes: 128, auxiliaryBytes: 99, scratchBytes: 8)
        let id = CBv2RequestID(1)
        let ticket = try physical.transferCheckpoint(to: 128, admission: admission) { previous in
            try admission.transferCheckpointStage(stage, requestID: id, maximumTokens: 16,
                previousPhysicalBytes: previous, physicalBytes: 128)
        }
        ticket.commit()
        stage.closeAfterDroppingOwners()
        // Accounting fixture has no native aliases: this is the scheduler's
        // preemption release, after the engine drops that request's state.
        admission.releaseAll(id: id)
        #expect(admission.nonBackendBytesReserved == 128 && admission.bytesReserved == 128) // No nominal request remains; the floor is all overhead.
        try admission.reserve(id: id, additionalTokens: 16)
        admission.releaseCheckpointRequest(id: id, ownerIdentity: stage.identity)
        #expect(admission.nonBackendBytesReserved == (128 - 64) + 72 && admission.bytesReserved == 200)
        admission.releaseAll(id: id)
        physical.release(to: 0)
        #expect(admission.bytesReserved == 0)
    }

    @Test func paddedAuxiliaryRollbackRemovesItsTypedPartition() throws {
        let admission = ledger(bytes: 1024)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 0)
        defer { physical.close() }
        let stage = try admission.reserveCheckpointStage(targetBytes: 128, auxiliaryBytes: 99, scratchBytes: 8)
        let ticket = try physical.transferCheckpoint(to: 128, admission: admission) { previous in
            try admission.transferCheckpointStage(stage, requestID: .init(1), maximumTokens: 16,
                previousPhysicalBytes: previous, physicalBytes: 128)
        }
        #expect(admission.nonBackendBytesReserved == (128 - 64) + 99 + 8)
        // Accounting-only fixture: no aliases remain when rollback starts.
        ticket.rollbackAfterDroppingOwners()
        #expect(admission.bytesReserved == 8 && admission.nonBackendBytesReserved == 8)
        #expect(admission.targetBytesReserved(partitionedBy: []).unmaterialized == 0)
        stage.closeAfterDroppingOwners()
        #expect(admission.bytesReserved == 0 && admission.nonBackendBytesReserved == 0)
    }
}
