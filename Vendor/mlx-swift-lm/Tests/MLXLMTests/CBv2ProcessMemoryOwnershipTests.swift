import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Deterministic cross-engine capacity seam. Actual process U/M arithmetic is
/// tested by ProviderCore; these tests exercise native transaction/array order.
private final class NativeProcessCapacity: @unchecked Sendable {
    private let lock = NSLock()
    private let cap: UInt64
    private var charges: [UUID: UInt64] = [:]

    init(cap: UInt64 = .max) { self.cap = cap }

    func replace(_ id: UUID, bytes: UInt64) throws {
        try lock.withLock {
            let other = charges.filter { $0.key != id }.values.reduce(0, +)
            guard bytes <= cap - other else { throw NativeProcessOwner.Failure.capacity }
            charges[id] = bytes
        }
    }
}

private final class NativeProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
    enum Failure: Error { case capacity, invalidCoverage }
    private let lock = NSLock()
    private let capacity: NativeProcessCapacity
    private let id = UUID()
    private var charge: UInt64 = 0
    private var materialized: UInt64 = 0
    private var closing = false
    private var retired = false
    private var events: [String] = []

    init(_ capacity: NativeProcessCapacity = .init()) { self.capacity = capacity }

    func replaceCharge(_ bytes: UInt64) throws {
        try lock.withLock {
            guard !retired, bytes >= materialized else { throw Failure.invalidCoverage }
            guard !closing || bytes <= charge else { throw Failure.capacity }
            try capacity.replace(id, bytes: bytes)
            charge = bytes
            if closing && bytes == 0 { retired = true }
            events.append("C:\(bytes)")
        }
    }

    func recordMaterialization(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes >= materialized, bytes <= charge else { throw Failure.invalidCoverage }
            materialized = bytes
            events.append("M:\(bytes)")
        }
    }

    func withdrawCoverage(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= materialized else { throw Failure.invalidCoverage }
            materialized -= bytes
            events.append("M:\(materialized)")
        }
    }

    func retire() { lock.withLock { closing = true; if charge == 0 { retired = true } } }
    func state() -> (charge: UInt64, materialized: UInt64, events: [String]) {
        lock.withLock { (charge, materialized, events) }
    }
}

private final class WeakProcessArray { weak var array: MLXArray?; init(_ array: MLXArray) { self.array = array } }

@Suite("Native process ownership", .serialized)
struct CBv2ProcessMemoryOwnershipTests {
    private func fixture(_ owner: NativeProcessOwner, types: [DType] = [.bfloat16])
        throws -> (PagedKVBackend, AdmissionV2, [CBv2LayerKind])
    {
        let kinds = types.map { _ in CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4) }
        let config = PagedKVPoolConfig(
            capacityBytes: 32 << 20, maxPrefillChunk: 64, maxBufferLength: 32 << 20,
            segmentSizeBytes: 32 << 10, layerDTypes: types)
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 32 << 20,
            config: .init(watermarkFraction: 0, layerElementBytes: types.map(\.size)),
            residency: CBv2PagedKVResidency(config: config), processMemoryOwner: owner)
        backend.pool.bindAdmission(admission)
        return (backend, admission, kinds)
    }

    @Test func completeChargePrecedesAllocationAndReleaseWithdrawsCoverage() throws {
        let owner = NativeProcessOwner()
        let (backend, admission, kinds) = try fixture(owner)
        try admission.reserve(id: .init(1), additionalTokens: 65)
        var allocated: [WeakProcessArray] = []
        backend.pool.slabEval = { array in
            #expect(owner.state().charge >= UInt64(array.nbytes))
            try withError { eval(array) }
            allocated.append(WeakProcessArray(array))
        }
        let rows = try backend.makeSequenceState(layerKinds: kinds, promptLength: 17, maxLength: 65)
        let physical = UInt64(backend.pool.bytesMaterialized)
        #expect(physical > 0)
        #expect(owner.state().charge == UInt64(admission.bytesReserved))
        #expect(owner.state().materialized == physical)
        #expect(allocated.allSatisfy { $0.array != nil })
        backend.release(rows)
        #expect(owner.state().materialized == 0)
        #expect(allocated.allSatisfy { $0.array == nil })
        admission.releaseAll(id: .init(1))
        #expect(owner.state().charge == 0)
        #expect(owner.state().events.last == "C:0")
    }

    @Test func sharedRefusalLeavesNativeRequestAndPhysicalStateUntouched() throws {
        let capacity = NativeProcessCapacity(cap: 128)
        let kind = CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        func ledger(_ owner: NativeProcessOwner) -> AdmissionV2 {
            AdmissionV2(layerKinds: [kind], bytesCapacity: 1024,
                        config: .init(watermarkFraction: 0, elementBytes: 2),
                        processMemoryOwner: owner)
        }
        let a = ledger(NativeProcessOwner(capacity)), b = ledger(NativeProcessOwner(capacity))
        let floor = a.bindBackendPhysicalFloor(initialBytes: 0)
        defer { floor.close() }
        try a.reserve(id: .init(1), additionalTokens: 16)
        try b.reserve(id: .init(1), additionalTokens: 16)
        #expect(a.bytesReserved == 64 && b.bytesReserved == 64)
        #expect(throws: CBv2KVError.self) { try floor.resize(to: 65) }
        #expect(floor.bytes == 0 && a.bytesReserved == 64)
        #expect(throws: CBv2KVError.self) { try b.reserveTransient(bytes: 1) }
        #expect(b.transientBytesReserved == 0 && b.bytesReserved == 64)
        a.releaseAll(id: .init(1))
        try b.reserve(id: .init(2), additionalTokens: 16)
        #expect(b.bytesReserved == 128)
        b.releaseAll(id: .init(1)); b.releaseAll(id: .init(2))
    }

    @Test func partialPreparationDropsEvaluatedCoverageBeforeRollback() throws {
        let owner = NativeProcessOwner()
        let (backend, admission, kinds) = try fixture(owner, types: [.bfloat16, .float32])
        try admission.reserve(id: .init(1), additionalTokens: 17)
        var allocations = 0
        var allocated: [WeakProcessArray] = []
        backend.pool.slabEval = { array in
            allocations += 1
            if allocations == 2 { throw CBv2CompleteCheckpointError.allocationFailed }
            try withError { eval(array) }
            allocated.append(WeakProcessArray(array))
        }
        #expect(throws: (any Error).self) {
            try backend.makeSequenceState(layerKinds: kinds, promptLength: 17, maxLength: 17)
        }
        #expect(allocations == 2)
        #expect(allocated.count == 1 && allocated.allSatisfy { $0.array == nil })
        #expect(owner.state().materialized == 0)
        #expect(backend.pool.bytesMaterialized == 0)
        #expect(owner.state().charge == UInt64(admission.bytesReserved))
        #expect(owner.state().events.contains { $0.hasPrefix("M:") && $0 != "M:0" })
        admission.releaseAll(id: .init(1))
        #expect(owner.state().charge == 0)
    }

    @Test func stagedBackingKeepsOneCreditAcrossRebasedWrappers() throws {
        let owner = NativeProcessOwner()
        let (backend, admission, kinds) = try fixture(owner)
        let plan = try CBv2PagedCheckpointStoragePlan(
            layerKinds: kinds, config: backend.pool.config, position: 17)
        let lease = try admission.reserveCheckpointStage(
            targetBytes: plan.nativeBytes, auxiliaryBytes: 0, scratchBytes: 16)
        let storage = try CBv2PagedCheckpointStorage(
            plan: plan, evaluate: { array in try withError { eval(array) } }, admission: admission)
        #expect(owner.state().materialized == UInt64(storage.allocatedBytes))
        let frame = try CBv2PagedCheckpointFrame(storage: storage, auxiliary: [], lease: lease)
        let adopted = try backend.pool.importCheckpoint(
            frame, admission: admission, requestID: .init(1), layerKinds: kinds, maximumTokens: 65)
        #expect(owner.state().materialized == UInt64(backend.pool.bytesMaterialized))
        #expect(owner.state().charge == UInt64(admission.bytesReserved))
        let after = owner.state()
        frame.close(); storage.close()
        #expect(owner.state().charge == after.charge && owner.state().materialized == after.materialized)
        adopted.release()
        #expect(owner.state().charge == 0 && owner.state().materialized == 0)
    }

    @Test func retainedReadAliasDoesNotKeepCreditAfterPoolRetirement() throws {
        let owner = NativeProcessOwner()
        let (backend, admission, kinds) = try fixture(owner)
        try admission.reserve(id: .init(1), additionalTokens: 17)
        let rows = try backend.makeSequenceState(layerKinds: kinds, promptLength: 17, maxLength: 17)
        let group = backend.pool.group(backend.pool.groupKey(forLayer: 0))
        let retained = try #require(group.segments.values.first?.storage)
        let physical = owner.state().materialized
        #expect(physical > 0)
        backend.release(rows)
        admission.releaseAll(id: .init(1))
        #expect(owner.state().charge == 0 && owner.state().materialized == 0)
        // The real backing remains readable and counted in allocator U without
        // claiming it still belongs to this engine's native charge C.
        #expect(retained.nbytes > 0)
        #expect(Memory.snapshot().activeMemory >= retained.nbytes)
        withExtendedLifetime(retained) {}
    }

    @Test func emptyPoolTeardownDoesNotTransactWithRetiredGeneration() throws {
        let owner = NativeProcessOwner()
        do {
            let (backend, admission, _) = try fixture(owner)
            admission.closeProcessMemoryOwner()
            backend.pool.physicalLease?.close()
            backend.pool.physicalLease?.close()
        }
        #expect(owner.state().charge == 0 && owner.state().events.isEmpty)
    }

    @Test func detachedPromiseAndClosingOwnerRetainTheirActualCharge() throws {
        let owner = NativeProcessOwner()
        let (backend, admission, kinds) = try fixture(owner)
        try admission.reserve(id: .init(1), additionalTokens: 17)
        let rows = try backend.makeSequenceState(layerKinds: kinds, promptLength: 17, maxLength: 17)
        let detached = admission.detachReservation(id: .init(1))
        let held = owner.state()
        admission.closeProcessMemoryOwner()
        #expect(owner.state().charge == held.charge && owner.state().materialized == held.materialized)
        backend.release(rows)
        #expect(owner.state().materialized == 0 && owner.state().charge > 0)
        detached.release()
        #expect(owner.state().charge == 0)
    }
}
