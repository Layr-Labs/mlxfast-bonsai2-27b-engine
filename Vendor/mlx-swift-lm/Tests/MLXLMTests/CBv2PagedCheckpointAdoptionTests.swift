import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Page-native checkpoint adoption", .serialized)
struct CBv2PagedCheckpointAdoptionTests {
    private func fixture(types: [DType] = [.bfloat16], segmentPages: Int = 2)
        throws -> (PagedKVBackend, AdmissionV2, [CBv2LayerKind])
    {
        let kinds = types.map { _ in CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4) }
        let largestPage = 2 * 2 * 16 * 64 * types.map(\.size).max()!
        let config = PagedKVPoolConfig(
            capacityBytes: 32 << 20, maxPrefillChunk: 64, maxBufferLength: 32 << 20,
            segmentSizeBytes: largestPage * (segmentPages + 1), layerDTypes: types)
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: 32 << 20, config: .init(
            watermarkFraction: 0, elementBytes: 2, layerElementBytes: types.map(\.size),
            fixedBytesPerRequest: 8), residency: CBv2PagedKVResidency(config: config))
        backend.pool.bindAdmission(admission)
        return (backend, admission, kinds)
    }

    private func rawBytes(count: Int, salt: Int) -> Data {
        Data((0 ..< count).map { UInt8(truncatingIfNeeded: ($0 * 73) ^ ($0 >> 3) ^ salt) })
    }

    private func frame(backend: PagedKVBackend, admission: AdmissionV2, kinds: [CBv2LayerKind],
                       position: Int = 17, weakOwners: inout [WeakImportedArray])
        throws -> CBv2PagedCheckpointFrame
    {
        let plan = try CBv2PagedCheckpointStoragePlan(
            layerKinds: kinds, config: backend.pool.config, position: position)
        let lease = try admission.reserveCheckpointStage(
            targetBytes: plan.nativeBytes, auxiliaryBytes: try Memory.allocationFootprintUpperBound(byteCount: 8), scratchBytes: 16)
        let storage = try CBv2PagedCheckpointStorage(plan: plan) { array in
            try withError { eval(array) }
            weakOwners.append(WeakImportedArray(array))
        }
        do {
            for (index, layer) in plan.layers.enumerated() {
                for values in [false, true] {
                    let data = rawBytes(count: 2 * position * 64 * layer.key.dtype.size,
                                        salt: index * 19 + (values ? 7 : 3))
                    try storage.append(layerIndex: index, values: values, byteOffset: 0, data: data)
                }
            }
            let allocationStream = StreamOrDevice.default
            let auxiliary = MLXArray.zeros([4], dtype: .float16, stream: allocationStream)
            try withError { eval(auxiliary); allocationStream.stream.synchronize() }
            weakOwners.append(WeakImportedArray(auxiliary))
            return try CBv2PagedCheckpointFrame(storage: storage, auxiliary: [auxiliary], lease: lease)
        } catch {
            storage.close()
            lease.closeAfterDroppingOwners()
            throw error
        }
    }

    private func read(_ row: PagedSequenceKV, position: Int, values: Bool, admission: AdmissionV2) throws -> Data {
        let source = try CBv2PagedCheckpointTensorSource(row: row, position: position, values: values, admission: admission)
        defer { source.close() }
        return try source.readSegment(byteOffset: 0, maximumBytes: 4 << 20)
    }

    @Test("M-page import preserves native bits and appends into its exclusive partial frontier",
          arguments: [DType.bfloat16, .float16, .float32])
    func partialFrontier(dtype: DType) throws {
        let (backend, admission, kinds) = try fixture(types: [dtype])
        var weakOwners: [WeakImportedArray] = []
        let staged = try frame(backend: backend, admission: admission, kinds: kinds, weakOwners: &weakOwners)
        let stageIDs = weakOwners.dropLast().map { ObjectIdentifier($0.array!) }
        let adopted = try backend.pool.importCheckpoint(
            staged, admission: admission, requestID: .init(1), layerKinds: kinds, maximumTokens: 65)
        defer { adopted.release() }
        staged.close() // Empty public handle cannot release installed pages.
        #expect(throws: CBv2CompleteCheckpointError.closed) {
            try backend.pool.importCheckpoint(staged, admission: admission,
                requestID: .init(2), layerKinds: kinds, maximumTokens: 65)
        }
        let row = try #require(adopted.rows.first)
        #expect(row.absoluteOffset == 17 && row.table.count == 2 && row.reservedPages == 5)
        let liveIDs = backend.pool.group(row.groupKey).segments.values.map { ObjectIdentifier($0.storage) }
        #expect(stageIDs.allSatisfy(liveIDs.contains), "native buffers are moved, never copied")
        for values in [false, true] {
            #expect(try read(row, position: 17, values: values, admission: admission)
                == rawBytes(count: 2 * 17 * 64 * dtype.size, salt: values ? 7 : 3))
        }
        #expect(backend.pool.group(row.groupKey).refCounts[Int(row.table.last!)] == 1)
        let one = MLXArray.ones([2, 1, 64], dtype: dtype)
        row.write(keys: one, values: one)
        #expect(try read(row, position: 17, values: false, admission: admission)
            == rawBytes(count: 2 * 17 * 64 * dtype.size, salt: 3))
        #expect(row.absoluteOffset == 18 && row.table.count == 2)
        adopted.release()
        #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)
        #expect(admission.bytesReserved == 0 && weakOwners.allSatisfy { $0.array == nil })
    }

    @Test func retainedFreeBackingSuppliesSuffixWithoutFullNPrivateAllocation() throws {
        let (backend, admission, kinds) = try fixture(segmentPages: 8)
        let key = backend.pool.groupKey(forLayer: 0)
        // One 8-page segment remains pinned by a 4-page promise. The remaining
        // four usable pages must serve the new request's suffix promise.
        try admission.reserve(id: .init(1), additionalTokens: 64)
        try backend.pool.reserve([key: 8])
        try backend.commitSlabs()
        backend.pool.unreserve([key: 4])
        #expect(backend.pool.group(key).committedUsablePages == 8)
        let oldPhysical = backend.bytesWired
        var weakOwners: [WeakImportedArray] = []
        let staged = try frame(backend: backend, admission: admission, kinds: kinds, weakOwners: &weakOwners)
        var allocations = 0
        backend.pool.slabEval = { _ in allocations += 1 }
        let adopted = try backend.pool.importCheckpoint(staged, admission: admission,
            requestID: .init(2), layerKinds: kinds, maximumTokens: 64)
        #expect(allocations == 0)
        let importedPhysical = try weakOwners.dropLast().reduce(0) { total, owner in
            let array = try #require(owner.array)
            let observed = try array.evaluatedBufferInfo()
            return total + (try #require(observed)).allocatedBytes
        }
        #expect(backend.bytesWired == oldPhysical + importedPhysical)
        #expect(backend.pool.group(key).pagesReserved == 8)
        #expect(backend.pool.group(key).committedUsablePages == 10)
        adopted.release()
        backend.pool.unreserve([key: 4])
        admission.releaseAll(id: .init(1))
        #expect(backend.bytesWired == 0 && admission.bytesReserved == 0)
    }

    @Test("allocation and grant-epoch failures discard every private alias before floor refund",
          arguments: [false, true])
    func prepareFailure(epochChange: Bool) throws {
        let (backend, admission, kinds) = try fixture(types: [.bfloat16, .float32])
        var weakOwners: [WeakImportedArray] = []
        let staged = try frame(backend: backend, admission: admission, kinds: kinds, weakOwners: &weakOwners)
        var weakGrowth: [WeakImportedArray] = []
        var allocations = 0
        backend.pool.slabEval = { array in
            try withError { eval(array) }
            weakGrowth.append(WeakImportedArray(array))
            allocations += 1
            if epochChange {
                backend.updateBytesCapacity((32 << 20) - 1)
            } else if allocations == 2 {
                throw CBv2CompleteCheckpointError.allocationFailed
            }
        }
        var observed = false
        backend.pool.checkpointImportBeforeRollback = {
            observed = true
            #expect(weakOwners.allSatisfy { $0.array == nil })
            #expect(weakGrowth.allSatisfy { $0.array == nil })
            #expect(admission.bytesReserved > 0, "full request/floor remains charged until aliases drain")
            #expect(backend.pool.physicalLease!.bytes > 0)
            #expect(backend.bytesWired == 0 && backend.bytesReserved == 0)
        }
        #expect(throws: (any Error).self) {
            try backend.pool.importCheckpoint(staged, admission: admission,
                requestID: .init(1), layerKinds: kinds, maximumTokens: 129)
        }
        #expect(observed && allocations >= 2)
        #expect(admission.bytesReserved == 0 && backend.pool.physicalLease!.bytes == 0)
        #expect(backend.pool.groups.values.allSatisfy { $0.segments.isEmpty && $0.pagesInUse == 0 })
        staged.close()
        #expect(admission.bytesReserved == 0)
    }

    @Test func incompatibleLayerCountAndGeometryRefuseBeforePoolMutation() throws {
        let (destination, destinationAdmission, oneKind) = try fixture()
        // Two staged rows share one geometry group, as in Qwen. Group-count
        // equality must not let the second layer index past a one-layer pool.
        let (source, _, twoKinds) = try fixture(types: [.bfloat16, .bfloat16])
        var weakOwners: [WeakImportedArray] = []
        let twoRows = try frame(backend: source, admission: destinationAdmission,
                               kinds: twoKinds, weakOwners: &weakOwners)
        var allocations = 0
        destination.pool.slabEval = { _ in allocations += 1 }
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try destination.pool.importCheckpoint(twoRows, admission: destinationAdmission,
                requestID: .init(1), layerKinds: twoKinds, maximumTokens: 65)
        }
        #expect(allocations == 0 && destination.bytesReserved == 0 && destination.bytesWired == 0)
        #expect(destinationAdmission.bytesReserved == 0 && weakOwners.allSatisfy { $0.array == nil })
        for alteration in 0 ..< 3 {
            let staged = try frame(backend: destination, admission: destinationAdmission,
                                   kinds: oneKind, weakOwners: &weakOwners)
            var invalid = oneKind
            switch alteration {
            case 0: invalid[0].headDim = 128
            case 1: invalid[0].kvHeads = 1
            default: invalid[0].modelLayerIndex = 99
            }
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                try destination.pool.importCheckpoint(staged, admission: destinationAdmission,
                    requestID: .init(2), layerKinds: invalid, maximumTokens: 65)
            }
            #expect(allocations == 0 && destination.bytesReserved == 0 && destination.bytesWired == 0)
            #expect(destinationAdmission.bytesReserved == 0 && weakOwners.allSatisfy { $0.array == nil })
        }
    }

    @Test func detachedOldAdoptionCannotReleaseReusedSamplingID() throws {
        let (backend, admission, kinds) = try fixture()
        var weakOwners: [WeakImportedArray] = []
        let staged = try frame(backend: backend, admission: admission, kinds: kinds, weakOwners: &weakOwners)
        let id = CBv2RequestID(7)
        let adopted = try backend.pool.importCheckpoint(staged, admission: admission,
            requestID: id, layerKinds: kinds, maximumTokens: 65)
        let auxiliaryBytes: Int = try {
            let array = try #require(weakOwners.last?.array)
            let observed = try array.evaluatedBufferInfo()
            return try #require(observed).allocatedBytes
        }()
        let extra = max(0, auxiliaryBytes - 8)
        let detached = admission.detachReservation(id: id)
        try admission.reserve(id: id, additionalTokens: 32)
        adopted.release()
        adopted.release()
        #expect(weakOwners.allSatisfy { $0.array == nil })
        #expect(admission.bytesReserved == admission.allocatedBytes(forTokens: 65) + extra
            + admission.allocatedBytes(forTokens: 32))
        detached.release()
        #expect(admission.bytesReserved == admission.allocatedBytes(forTokens: 32))
        admission.releaseAll(id: id)
        #expect(admission.bytesReserved == 0 && backend.bytesWired == 0)
    }

    @Test func concurrentCloseAndConsumeHaveExactlyOneOwner() throws {
        let (backend, admission, kinds) = try fixture()
        for _ in 0 ..< 8 {
            var weakOwners: [WeakImportedArray] = []
            let staged = try frame(backend: backend, admission: admission, kinds: kinds, weakOwners: &weakOwners)
            let result = ConsumedCheckpointOwner()
            DispatchQueue.concurrentPerform(iterations: 2) { branch in
                if branch == 0 { result.set(try? staged.consume()) }
                else { staged.close() }
            }
            staged.close()
            if let owner = result.take() {
                #expect(admission.bytesReserved > 0)
                #expect(weakOwners.allSatisfy { $0.array != nil })
                owner.close()
            }
            #expect(admission.bytesReserved == 0)
            #expect(weakOwners.allSatisfy { $0.array == nil })
        }
    }

    @Test func providerCloseWinsBeforeConsumptionAndLeavesNoRetainedPrefixPayload() throws {
        let (backend, admission, kinds) = try fixture()
        var weakOwners: [WeakImportedArray] = []
        let staged = try frame(backend: backend, admission: admission, kinds: kinds, weakOwners: &weakOwners)
        staged.close()
        staged.close()
        #expect(weakOwners.allSatisfy { $0.array == nil } && admission.bytesReserved == 0)
        #expect(throws: CBv2CompleteCheckpointError.closed) {
            try backend.pool.importCheckpoint(staged, admission: admission,
                requestID: .init(1), layerKinds: kinds, maximumTokens: 65)
        }
    }
}

private final class WeakImportedArray {
    weak var array: MLXArray?
    init(_ array: MLXArray) { self.array = array }
}

/// Only the frame is shared between concurrent branches; the consumed native
/// owner is transferred through this test box and inspected after both join.
private final class ConsumedCheckpointOwner: @unchecked Sendable {
    private let lock = NSLock()
    private var owner: CBv2PagedCheckpointOwner?
    func set(_ owner: CBv2PagedCheckpointOwner?) { lock.withLock { self.owner = owner } }
    func take() -> CBv2PagedCheckpointOwner? {
        lock.withLock {
            let value = owner
            owner = nil
            return value
        }
    }
}
