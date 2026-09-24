import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Queue-captured paged storage accounting", .serialized)
struct CBv2PagedStorageTelemetryTests {
    private let pageBytes = 2 * 16 * 64 * 2
    private var kind: CBv2LayerKind {
        .init(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
    }

    private func fixture() throws -> (PagedKVBackend, AdmissionV2) {
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 100 * pageBytes, dtype: .bfloat16, maxPrefillChunk: 64,
            maxBufferLength: 1 << 20, segmentSizeBytes: 9 * pageBytes, layerDTypes: [.bfloat16]))
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 4 * (try Memory.allocationFootprintUpperBound(byteCount: 5 * pageBytes)),
            config: .init(watermarkFraction: 0, elementBytes: 2), residency: backend.kvResidency)
        backend.pool.bindAdmission(admission)
        return (backend, admission)
    }

    @Test("heartbeat reads and grant-only updates cannot freshen an allocator capture")
    func frozenSnapshotProvenance() throws {
        let (backend, _) = try fixture()
        let first = try #require(backend.pool.segmentStorageSnapshot)
        let gauges = CBv2EngineGauges(kvBytesCapacity: 100 * pageBytes, pagedStorage: first)
        #expect(gauges.read().pagedStorage == first)
        #expect(gauges.read().pagedStorage == first)
        backend.updateBytesCapacity(50 * pageBytes)
        gauges.updateKVBytesCapacity(50 * pageBytes, backendCapacity: 50 * pageBytes)
        let updated = gauges.read()
        #expect(updated.kvBytesCapacity == 50 * pageBytes)
        #expect(updated.kvBytesBackendCapacity == 50 * pageBytes)
        #expect(updated.pagedStorage == first)
        let next = try #require(backend.pool.segmentStorageSnapshot)
        #expect(next.grantBytes == 50 * pageBytes)
        #expect(next.generation == first.generation && next.captureSequence == first.captureSequence + 1)
        #expect(next.capturedUptimeNanoseconds >= first.capturedUptimeNanoseconds)
        var published = updated
        published.pagedStorage = next
        gauges.update(published)
        #expect(gauges.read().pagedStorage == next)
        let (other, _) = try fixture()
        #expect(other.pool.segmentStorageSnapshot?.generation != first.generation)
    }

    @Test("allocation, ledger, grant and stale-epoch failures are counted once at their actual boundaries")
    func failureClassification() throws {
        let (backend, admission) = try fixture()
        try admission.reserve(id: .init(1), additionalTokens: 64)
        let first = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        defer { backend.release(first); admission.releaseAll(id: .init(1)) }
        backend.pool.slabEval = { _ in throw CBv2CompleteCheckpointError.allocationFailed }
        // The public backend normalizes allocation failures to a retryable
        // capacity error; the counter below retains the native cause.
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        }
        var snapshot = try #require(backend.pool.segmentStorageSnapshot)
        #expect(snapshot.allocationFailures == 1 && snapshot.admissionRefusals == 0)
        #expect(snapshot.grantRefusals == 0 && snapshot.grantEpochRetries == 0)

        admission.updateBytesCapacity(5 * pageBytes)
        var allocations = 0
        backend.pool.slabEval = { array in allocations += 1; eval(array) }
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        }
        #expect(allocations == 0)
        snapshot = try #require(backend.pool.segmentStorageSnapshot)
        #expect(snapshot.admissionRefusals == 1 && snapshot.allocationFailures == 1)

        admission.updateBytesCapacity(4 * (try Memory.allocationFootprintUpperBound(byteCount: 5 * pageBytes)))
        backend.updateBytesCapacity(0)
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        }
        snapshot = try #require(backend.pool.segmentStorageSnapshot)
        #expect(snapshot.grantRefusals == 1 && snapshot.grantEpochRetries == 0)
        #expect(allocations == 0)

        backend.updateBytesCapacity(100 * pageBytes)
        backend.pool.slabEval = { array in eval(array); backend.updateBytesCapacity(99 * pageBytes) }
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        }
        snapshot = try #require(backend.pool.segmentStorageSnapshot)
        #expect(snapshot.grantEpochRetries == 1 && snapshot.grantRefusals == 1)
        #expect(snapshot.allocationFailures == 1 && snapshot.admissionRefusals == 1)
        #expect(snapshot.committedBytes - snapshot.allocatorPaddingBytes == 5 * pageBytes
            && snapshot.reservedPageBytes == 4 * pageBytes)
        backend.pool.slabEval = { eval($0) }
        let retry = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        backend.release(retry)
        let success = try #require(backend.pool.segmentStorageSnapshot)
        #expect(success.allocationFailures == snapshot.allocationFailures)
        #expect(success.admissionRefusals == snapshot.admissionRefusals)
        #expect(success.grantRefusals == snapshot.grantRefusals)
        #expect(success.grantEpochRetries == snapshot.grantEpochRetries)
    }

    @Test("nominal accounting includes detached owners until their release and never sums other slots")
    func detachedNominalOwnership() throws {
        let (backend, admission) = try fixture()
        let id = CBv2RequestID(7)
        try admission.reserve(id: id, additionalTokens: 64)
        let first = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        let retired = admission.detachReservation(id: id)
        var snapshot = try #require(backend.pool.segmentStorageSnapshot)
        #expect(snapshot.nominalKVBytes == 4 * pageBytes && snapshot.physicalFloorOverheadBytes == snapshot.committedBytes - 4 * pageBytes)
        try admission.reserve(id: id, additionalTokens: 64)
        let second = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        snapshot = try #require(backend.pool.segmentStorageSnapshot)
        #expect(snapshot.nominalKVBytes == 8 * pageBytes && snapshot.physicalFloorOverheadBytes == snapshot.committedBytes - 8 * pageBytes)
        backend.release(first)
        retired.release()
        retired.release()
        snapshot = try #require(backend.pool.segmentStorageSnapshot)
        #expect(snapshot.nominalKVBytes == 4 * pageBytes && snapshot.physicalFloorOverheadBytes == snapshot.committedBytes - 4 * pageBytes)
        let (other, _) = try fixture()
        #expect(other.pool.segmentStorageSnapshot?.nominalKVBytes == 0)
        backend.release(second)
        admission.releaseAll(id: id)
        snapshot = try #require(backend.pool.segmentStorageSnapshot)
        #expect(snapshot.committedBytes == 0 && snapshot.nominalKVBytes == 0 && snapshot.physicalFloorOverheadBytes == 0)
    }

    @Test("publication classifies epoch and capacity under the grant lock without partial install")
    func typedGrantOutcome() {
        let grant = PagedKVGrant(bytes: 64)
        let old = grant.snapshot()
        var installs = 0
        grant.update(bytes: 128)
        #expect(grant.publish(expected: old, physicalBytes: 32, install: { installs += 1 }) == .staleEpoch)
        #expect(grant.publish(expected: grant.snapshot(), physicalBytes: 129, install: { installs += 1 }) == .exceedsGrant)
        #expect(installs == 0)
        #expect(grant.publish(expected: grant.snapshot(), physicalBytes: 128, install: { installs += 1 }) == .installed)
        #expect(installs == 1)
    }
}
