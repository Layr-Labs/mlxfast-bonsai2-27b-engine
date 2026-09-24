import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Private page-native checkpoint transfers", .serialized)
struct CBv2PagedCheckpointStorageTests {
    private func kind() -> CBv2LayerKind {
        .init(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
    }

    private func config(types: [DType], usablePagesPerSegment: Int = 2) -> PagedKVPoolConfig {
        let largestPage = 2 * 2 * 16 * 64 * types.map(\.size).max()!
        return .init(capacityBytes: 16 << 20, maxPrefillChunk: 64,
                     maxBufferLength: 16 << 20,
                     segmentSizeBytes: largestPage * (usablePagesPerSegment + 1), layerDTypes: types)
    }

    /// Includes arbitrary floating payloads (signed zero, NaNs and infinities
    /// among them). A checkpoint copy must preserve bits without arithmetic.
    private func bytes(count: Int, salt: Int) -> Data {
        Data((0 ..< count).map { UInt8(truncatingIfNeeded: ($0 * 73) ^ ($0 >> 3) ^ salt) })
    }

    private func fill(_ storage: CBv2PagedCheckpointStorage, layer: Int, values: Bool, bytes: Data) throws {
        let width = storage.plan.layers[layer].key.dtype.size
        let sizes = [width, 127 * width, 509 * width, 31 * width]
        var offset = 0
        var chunk = 0
        while offset < bytes.count {
            let end = min(bytes.count, offset + sizes[chunk % sizes.count])
            try storage.append(layerIndex: layer, values: values, byteOffset: offset,
                               data: bytes.subdata(in: offset ..< end))
            offset = end
            chunk += 1
        }
    }

    @Test("private fill/export preserves raw bits across head, page and segment seams",
          arguments: [DType.bfloat16, .float16, .float32])
    func packedRoundTrip(dtype: DType) throws {
        let admission = AdmissionV2(layerKinds: [kind(), kind()], bytesCapacity: 16 << 20)
        let plan = try CBv2PagedCheckpointStoragePlan(
            layerKinds: [kind(), kind()], config: config(types: [dtype, dtype]), position: 65)
        #expect(plan.groups.count == 1)
        #expect(plan.groups[0].usablePages == 10)
        let storage = try CBv2PagedCheckpointStorage(plan: plan) { array in try withError { eval(array) } }
        defer { storage.close() }
        #expect(storage.allocatedBytes <= plan.nativeBytes)
        #expect(storage.allocatedBytes == storage.groups.values.flatMap { $0.segments.values }
            .reduce(0) { $0 + $1.allocatedBytes })
        for layer in plan.layers.indices {
            for values in [false, true] {
                let expected = bytes(count: 2 * 65 * 64 * dtype.size, salt: layer * 19 + (values ? 7 : 3))
                try fill(storage, layer: layer, values: values, bytes: expected)
                let source = try CBv2PagedCheckpointTensorSource(storage: storage, layerIndex: layer, values: values, admission: admission)
                defer { source.close() }
                // First reads exercise the returned source directly: no
                // explicit fence eval or other view may repair ordering first.
                #expect(try source.readSegment(byteOffset: 0, maximumBytes: dtype.size)
                        == expected.prefix(dtype.size))
                #expect(try source.readSegment(byteOffset: 0, maximumBytes: expected.count) == expected)
                for offset in [dtype.size, 63 * dtype.size, 64 * 16 * dtype.size - dtype.size] {
                    let count = min(521 * dtype.size, expected.count - offset)
                    #expect(try source.readSegment(byteOffset: offset, maximumBytes: count)
                            == expected.subdata(in: offset ..< offset + count))
                }
                source.close()
                #expect(throws: CBv2CompleteCheckpointError.closed) {
                    try source.readSegment(byteOffset: 0, maximumBytes: dtype.size)
                }
            }
        }
    }

    @Test("live donor export uses captured page owners across more than seventeen segments",
          arguments: [DType.bfloat16, .float16, .float32])
    func liveDonorBoundedExport(dtype: DType) throws {
        let kinds = [kind()]
        let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: 16 << 20)
        let backend = try PagedKVBackend(layerKinds: kinds, config: config(types: [dtype], usablePagesPerSegment: 1))
        let rows = try backend.makeSequenceState(layerKinds: kinds, promptLength: 600, maxLength: 640)
        defer { backend.release(rows) }
        let row = try #require(rows[0] as? PagedSequenceKV)
        let keys = (MLXArray(0 ..< 2 * 600 * 64).asType(.float32) / 113)
            .reshaped([2, 600, 64]).asType(dtype)
        let values = (keys + 3).asType(dtype)
        row.write(keys: keys, values: values)
        #expect(backend.pool.group(row.groupKey).segments.count > 17)
        for (isValues, original) in [(false, keys), (true, values)] {
            let source = try CBv2PagedCheckpointTensorSource(row: row, position: 577, values: isValues, admission: admission)
            defer { source.close() }
            // Oracle is the original input, never row.snapshot/full gather.
            let expected = original[0..., ..<577, 0...].asData(access: .copy).data
            #expect(try source.readSegment(byteOffset: 0, maximumBytes: expected.count) == expected)
        }
    }

    @Test("mixed native groups are priced before allocation and second-group failure drops private owners")
    func mixedPlanAndAllocationFailure() throws {
        let types: [DType] = [.bfloat16, .float32]
        let plan = try CBv2PagedCheckpointStoragePlan(
            layerKinds: [kind(), kind()], config: config(types: types), position: 65)
        #expect(plan.groups.count == 2)
        #expect(plan.groups.allSatisfy { $0.layout.pageCount == 0 }, "planning allocates no page-address table")
        let first = try plan.groups[0].layout.adding(usablePages: plan.groups[0].usablePages, excluding: [])
        let failAt = first.segmentIDs.count + 1
        var calls = 0
        var weakOwners: [WeakCheckpointArray] = []
        #expect(throws: CBv2CompleteCheckpointError.allocationFailed) {
            _ = try CBv2PagedCheckpointStorage(plan: plan) { array in
                try withError { eval(array) }
                weakOwners.append(WeakCheckpointArray(array))
                calls += 1
                if calls == failAt { throw CBv2CompleteCheckpointError.allocationFailed }
            }
        }
        #expect(calls == failAt)
        #expect(weakOwners.allSatisfy { $0.array == nil })
    }

    @Test("window/absent-table plans and invalid writes refuse without exposing active page IDs")
    func refusalAndClose() throws {
        var missing = config(types: [.bfloat16])
        missing.layerDTypes = nil
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try CBv2PagedCheckpointStoragePlan(layerKinds: [kind()], config: missing, position: 65)
        }
        let window = CBv2LayerKind(attention: .slidingWindow(16), headDim: 64, kvHeads: 2, queryHeads: 4)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try CBv2PagedCheckpointStoragePlan(layerKinds: [window], config: config(types: [.bfloat16]), position: 65)
        }
        let plan = try CBv2PagedCheckpointStoragePlan(
            layerKinds: [kind()], config: config(types: [.bfloat16]), position: 65)
        let storage = try CBv2PagedCheckpointStorage(plan: plan) { array in try withError { eval(array) } }
        #expect(throws: CBv2CompleteCheckpointError.invalidSegment) {
            try storage.append(layerIndex: 0, values: false, byteOffset: 1, data: Data([0, 1]))
        }
        #expect(throws: CBv2CompleteCheckpointError.invalidSegment) {
            try storage.append(layerIndex: 1, values: false, byteOffset: 0, data: Data([0, 1]))
        }
        storage.close()
        storage.close()
        #expect(storage.groups.isEmpty)
        #expect(throws: CBv2CompleteCheckpointError.closed) {
            try storage.append(layerIndex: 0, values: false, byteOffset: 0, data: Data([0, 1]))
        }
    }
}

private final class WeakCheckpointArray {
    weak var array: MLXArray?
    init(_ array: MLXArray) { self.array = array }
}
