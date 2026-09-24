import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private final class FootprintProcessOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
    enum Fault: Error { case invalid }
    private let lock = NSLock()
    private var c: UInt64 = 0
    private var m: UInt64 = 0
    private var log: [(UInt64, UInt64)] = []
    var state: (charge: UInt64, materialized: UInt64) { lock.withLock { (c, m) } }
    var events: [(UInt64, UInt64)] { lock.withLock { log } }
    func replaceCharge(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes >= m else { throw Fault.invalid }
            c = bytes
            log.append((c, m))
        }
    }
    func recordMaterialization(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes >= m, bytes <= c else { throw Fault.invalid }
            m = bytes
            log.append((c, m))
        }
    }
    func withdrawCoverage(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= m else { throw Fault.invalid }
            m -= bytes
            log.append((c, m))
        }
    }
    func retire() {}
}

private final class FootprintWeakArray {
    weak var value: MLXArray?
    init(_ value: MLXArray) { self.value = value }
}

@Suite("Allocator footprints in native page ownership", .serialized)
struct CBv2AllocatorFootprintTests {
    private let pageBytes = 2 * 2 * 16 * 64 * 2
    private var kind: CBv2LayerKind {
        .init(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
    }

    private func fixture(capacity: Int = 8 << 20) throws -> (PagedKVBackend, AdmissionV2, FootprintProcessOwner) {
        let owner = FootprintProcessOwner()
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(capacityBytes: capacity,
            maxPrefillChunk: 64, segmentSizeBytes: 32 << 10, layerDTypes: [.bfloat16]))
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: capacity,
            config: .init(watermarkFraction: 0, elementBytes: 2), residency: backend.kvResidency,
            processMemoryOwner: owner)
        backend.pool.bindAdmission(admission)
        return (backend, admission, owner)
    }

    private func fullBackingView(_ array: MLXArray) throws -> MLXArray {
        let stream = StreamOrDevice.default
        let view = array.reshaped([array.size], stream: stream)
        try withError { eval(view); stream.stream.synchronize() }
        let observedOriginal = try array.evaluatedBufferInfo()
        let observedView = try view.evaluatedBufferInfo()
        let original = try #require(observedOriginal)
        let shared = try #require(observedView)
        #expect(!original.isUnique && !shared.isUnique)
        #expect(shared.allocatedBytes == original.allocatedBytes)
        #expect(view.nbytes == array.nbytes && shared.dataOffset == 0)
        #expect(shared.dataElements == array.size && shared.isRowContiguous)
        return view
    }

    @Test func metadataPricesIndependentSizeClassBuffersBeforeAllocation() throws {
        let layout = try PagedKVSegmentLayout(pageBytes: pageBytes, targetBytes: 9 * pageBytes,
                                              maximumBufferBytes: 8 << 20)
        let before = Memory.snapshot()
        let expected = try [9, 5, 2].reduce(0) { total, pages in
            total + (try Memory.allocationFootprintUpperBound(byteCount: pages * pageBytes))
        }
        #expect(layout.allocationBytes(addingUsablePages: 13) == expected)
        #expect(expected > (try Memory.allocationFootprintUpperBound(byteCount: 16 * pageBytes)),
                "one combined rounding cannot price three independent cached buffers")
        let after = Memory.snapshot()
        #expect(before.activeMemory == after.activeMemory && before.cacheMemory == after.cacheMemory)
    }

    @Test func evaluatedBackingSettlesBothPhysicalChargeAndCoverageToAllocatorBytes() throws {
        let bound = try Memory.allocationFootprintUpperBound(byteCount: 3 * pageBytes)
        let (backend, admission, owner) = try fixture(capacity: bound)
        try admission.reserve(id: .init(1), additionalTokens: 32)
        backend.pool.slabEval = { array in
            #expect(owner.state.charge == UInt64(bound), "preallocation charge covers cache reuse and alignment")
            try withError { eval(array) }
        }
        let rows = try backend.makeSequenceState(layerKinds: [kind], promptLength: 17, maxLength: 32)
        let actual = backend.pool.groups.values.reduce(0) { total, group in
            total + group.segments.values.reduce(0) { $0 + $1.allocatedBytes }
        }
        #expect(actual >= 3 * pageBytes && actual <= bound)
        #expect(owner.state.charge == UInt64(actual) && owner.state.materialized == UInt64(actual))
        #expect(backend.pool.physicalLease?.bytes == actual && backend.bytesWired == actual)
        let snapshot = try #require(backend.pool.segmentStorageSnapshot)
        #expect(snapshot.allocatorPaddingBytes == actual - 3 * pageBytes)
        #expect(snapshot.lastAllocationAllowanceBytes == bound - actual)
        #expect(snapshot.committedBytes == snapshot.reservedPageBytes + snapshot.poisonBytes
            + snapshot.slackBytes + snapshot.allocatorPaddingBytes)
        backend.release(rows)
        admission.releaseAll(id: .init(1))
        #expect(owner.state.charge == 0 && owner.state.materialized == 0)
    }

    @Test func retainedExternalViewCannotAcquireCoverageAfterCompletionFence() throws {
        let (backend, admission, owner) = try fixture()
        try admission.reserve(id: .init(1), additionalTokens: 32)
        let nominal = admission.bytesReserved
        var retainedView: MLXArray?
        backend.pool.slabEval = { array in
            try withError { eval(array) }
            let view = try fullBackingView(array)
            retainedView = view
        }
        #expect(throws: CBv2KVError.self) {
            let unexpected = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 17, maxLength: 32)
            backend.release(unexpected)
        }
        let view = try #require(retainedView)
        let observed = try view.evaluatedBufferInfo()
        let info = try #require(observed)
        #expect(info.allocatedBytes >= view.nbytes && info.dataElements == view.size)
        #expect(owner.state.materialized == 0 && owner.state.charge == UInt64(nominal))
        #expect(backend.bytesWired == 0 && backend.pool.physicalLease?.bytes == 0)
        admission.releaseAll(id: .init(1))
        #expect(owner.state.charge == 0 && owner.state.materialized == 0)
        withExtendedLifetime(view) {}
    }

    @Test func evaluatedThenThrownAllocationKeepsRetainedViewInAllocatorUsage() throws {
        let (backend, admission, owner) = try fixture()
        try admission.reserve(id: .init(1), additionalTokens: 32)
        let nominal = admission.bytesReserved
        var retainedView: MLXArray?
        var allocatedBytes = 0
        backend.pool.slabEval = { array in
            try withError { eval(array) }
            let view = try fullBackingView(array)
            retainedView = view
            let observed = try view.evaluatedBufferInfo()
            allocatedBytes = try #require(observed).allocatedBytes
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        #expect(throws: CBv2KVError.self) {
            let unexpected = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 17, maxLength: 32)
            backend.release(unexpected)
        }
        weak var weakView = retainedView
        #expect(weakView != nil && allocatedBytes > 0)
        #expect(owner.state.materialized == 0 && owner.state.charge == UInt64(nominal))
        #expect(backend.bytesWired == 0 && backend.pool.physicalLease?.bytes == 0)
        admission.releaseAll(id: .init(1))
        #expect(owner.state.charge == 0 && owner.state.materialized == 0)
        Memory.clearCache()
        let retained = Memory.snapshot()
        #expect(retained.activeMemory >= allocatedBytes)
        withExtendedLifetime(retainedView) {}
        retainedView = nil
        #expect(weakView == nil)
        Memory.clearCache()
        let drained = Memory.snapshot()
        #expect(drained.activeMemory + allocatedBytes <= retained.activeMemory)
    }

    @Test func recordedMLXFaultWinsOverInjectedEvaluationError() throws {
        let (backend, _, _) = try fixture()
        let key = backend.pool.groupKey(forLayer: 0)
        let layout = try PagedKVSegmentLayout(
            pageCount: 3, pageBytes: pageBytes, targetBytes: 32 << 10,
            maximumBufferBytes: 8 << 20)
        #expect(throws: MLXError.self) {
            try PagedKVSegment(index: 0, layout: layout, key: key,
                pageSize: 16, dtype: .bfloat16, evaluate: { array in
                    eval(array)
                    _ = array + MLXArray.zeros([3], dtype: .bfloat16)
                    throw CBv2CompleteCheckpointError.allocationFailed
                })
        }
    }

    @Test func oneByteBelowPreallocationBoundRefusesBeforeCreatingArray() throws {
        let bound = try Memory.allocationFootprintUpperBound(byteCount: 3 * pageBytes)
        let (backend, admission, owner) = try fixture(capacity: bound - 1)
        try admission.reserve(id: .init(1), additionalTokens: 32)
        var allocations = 0
        backend.pool.slabEval = { _ in allocations += 1 }
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 17, maxLength: 32)
        }
        #expect(allocations == 0 && backend.bytesWired == 0 && owner.state.materialized == 0)
        admission.releaseAll(id: .init(1))
        #expect(owner.state.charge == 0)
    }

    @Test func privateStageSettlesActualPagesAndKeepsScratchUntilClose() throws {
        let (backend, admission, owner) = try fixture()
        let plan = try CBv2PagedCheckpointStoragePlan(layerKinds: [kind], config: backend.pool.config, position: 32)
        let lease = try admission.reserveCheckpointStage(targetBytes: plan.nativeBytes, auxiliaryBytes: 0, scratchBytes: 37)
        var weakOwners: [FootprintWeakArray] = []
        let storage = try CBv2PagedCheckpointStorage(plan: plan, evaluate: { array in
            try withError { eval(array) }
            weakOwners.append(FootprintWeakArray(array))
        }, admission: admission)
        let actual = storage.allocatedBytes
        #expect(owner.state.charge == UInt64(plan.nativeBytes + 37))
        #expect(owner.state.materialized == UInt64(actual))
        let frame = try CBv2PagedCheckpointFrame(storage: storage, auxiliary: [], lease: lease)
        #expect(lease.targetBytes == actual && lease.scratchBytes == 37)
        #expect(owner.state.charge == UInt64(actual + 37) && owner.state.materialized == UInt64(actual))
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try lease.settleDestinationAfterEvaluation(targetBytes: actual, auxiliaryBytes: 0)
        }
        frame.close()
        #expect(weakOwners.allSatisfy { $0.value == nil })
        #expect(owner.state.charge == 0 && owner.state.materialized == 0)
    }

    @Test func settledStageTransfersOnceAndSuffixFootprintSettlesAfterCommit() throws {
        let (backend, admission, owner) = try fixture()
        let plan = try CBv2PagedCheckpointStoragePlan(layerKinds: [kind], config: backend.pool.config, position: 32)
        let lease = try admission.reserveCheckpointStage(targetBytes: plan.nativeBytes, auxiliaryBytes: 0, scratchBytes: 37)
        let storage = try CBv2PagedCheckpointStorage(plan: plan, evaluate: { array in try withError { eval(array) } }, admission: admission)
        let frame = try CBv2PagedCheckpointFrame(storage: storage, auxiliary: [], lease: lease)
        let adoption = try backend.pool.importCheckpoint(frame, admission: admission,
            requestID: .init(7), layerKinds: [kind], maximumTokens: 65)
        let actual = backend.bytesWired
        #expect(actual > lease.targetBytes)
        #expect(owner.state.charge == UInt64(actual) && owner.state.materialized == UInt64(actual))
        #expect(backend.pool.physicalLease?.bytes == actual)
        #expect(owner.events.contains { $0.0 > UInt64(actual) && $0.1 == UInt64(actual) },
                "actual coverage appears before unused preallocation allowance is settled")
        frame.close()
        adoption.release()
        #expect(owner.state.charge == 0 && owner.state.materialized == 0)
    }

    @Test(arguments: [false, true])
    func failedAllocationOrEpochDropsActualCoverageBeforeReturningBound(epoch: Bool) throws {
        let (backend, admission, owner) = try fixture()
        try admission.reserve(id: .init(7), additionalTokens: 65)
        let nominal = admission.bytesReserved
        var arrays: [FootprintWeakArray] = []
        var calls = 0
        backend.pool.slabEval = { array in
            try withError { eval(array) }
            arrays.append(FootprintWeakArray(array))
            calls += 1
            if epoch { backend.updateBytesCapacity((8 << 20) - 1) }
            else if calls == 2 { throw CBv2CompleteCheckpointError.allocationFailed }
        }
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 17, maxLength: 65)
        }
        #expect(calls >= 2 && arrays.allSatisfy { $0.value == nil })
        #expect(owner.state.materialized == 0 && owner.state.charge == UInt64(nominal))
        #expect(backend.bytesWired == 0 && backend.pool.physicalLease?.bytes == 0)
        admission.releaseAll(id: .init(7))
        #expect(owner.state.charge == 0)
    }

    @Test func privateAuxiliarySettlesActualCWithoutMintingMaterializationCoverage() throws {
        let (_, admission, owner) = try fixture()
        let bound = (try Memory.allocationFootprintUpperBound(byteCount: 514))
            + (try Memory.allocationFootprintUpperBound(byteCount: 4100))
        let lease = try admission.reserveCheckpointStage(targetBytes: 0, auxiliaryBytes: bound, scratchBytes: 17)
        let allocationStream = StreamOrDevice.default
        var arrays: [MLXArray]? = [MLXArray.zeros([257], dtype: .bfloat16, stream: allocationStream),
                                   MLXArray.zeros([1025], dtype: .float32, stream: allocationStream)]
        let weakOwners = arrays!.map(FootprintWeakArray.init)
        try withError { eval(arrays!); allocationStream.stream.synchronize() }
        let bytes = try CBv2CheckpointAllocationFootprint.freshBytes(arrays!)
        #expect(bytes.bound == bound && bytes.actual >= 4614 && bytes.actual <= bound)
        try lease.settleDestinationAfterEvaluation(targetBytes: 0, auxiliaryBytes: bytes.actual)
        #expect(owner.state.charge == UInt64(bytes.actual + 17) && owner.state.materialized == 0)
        #expect(lease.scratchBytes == 17)
        arrays = nil
        #expect(weakOwners.allSatisfy { $0.value == nil })
        lease.closeAfterDroppingOwners()
        #expect(owner.state.charge == 0 && owner.state.materialized == 0)
    }

    @Test func stageSettlementRejectsGrowthClosedGenerationAndDuplicateUse() throws {
        let (_, admission, _) = try fixture()
        let first = try admission.reserveCheckpointStage(targetBytes: 100, auxiliaryBytes: 9, scratchBytes: 7)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try first.settleDestinationAfterEvaluation(targetBytes: 101, auxiliaryBytes: 9)
        }
        #expect(admission.bytesReserved == 116)
        first.closeAfterDroppingOwners()
        let next = try admission.reserveCheckpointStage(targetBytes: 100, auxiliaryBytes: 9, scratchBytes: 7)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try first.settleDestinationAfterEvaluation(targetBytes: 72, auxiliaryBytes: 9)
        }
        #expect(admission.bytesReserved == 116)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try next.settleDestinationAfterEvaluation(targetBytes: 72, auxiliaryBytes: 10)
        }
        try next.settleDestinationAfterEvaluation(targetBytes: 72, auxiliaryBytes: 5)
        #expect(admission.bytesReserved == 84 && next.targetBytes == 72 && next.auxiliaryBytes == 5)
        #expect(next.scratchBytes == 7)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try next.settleDestinationAfterEvaluation(targetBytes: 71, auxiliaryBytes: 9)
        }
        next.closeAfterDroppingOwners()
        #expect(admission.bytesReserved == 0)
    }
}
