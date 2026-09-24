import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Direct paged checkpoint export", .serialized)
struct PagedCheckpointDirectExportTests {
    private var kind: CBv2LayerKind {
        .init(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
    }

    private func config(_ dtype: DType) -> PagedKVPoolConfig {
        .init(capacityBytes: 64 << 20, maxPrefillChunk: 64,
              segmentSizeBytes: 1 << 20, layerDTypes: [dtype])
    }

    private func rawBytes(_ count: Int, salt: Int) -> Data {
        Data((0 ..< count).map { UInt8(truncatingIfNeeded: ($0 * 73) ^ ($0 >> 3) ^ salt) })
    }

    @Test("bounded Data preserves arbitrary bits and survives source/backing retirement",
          arguments: [DType.float16, .bfloat16, .float32])
    func independentBoundedData(dtype: DType) throws {
        let position = 16_401
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 64 << 20)
        let plan = try CBv2PagedCheckpointStoragePlan(
            layerKinds: [kind], config: config(dtype), position: position)
        let storage = try CBv2PagedCheckpointStorage(plan: plan) { array in try withError { eval(array) } }
        defer { storage.close() }
        let expected = rawBytes(2 * position * 64 * dtype.size, salt: 29)
        func append(_ data: Data) throws {
            let limit = CBv2CompleteCheckpointManifest.maximumSegmentBytes
            for offset in stride(from: 0, to: data.count, by: limit) {
                try storage.append(layerIndex: 0, values: true, byteOffset: offset,
                    data: data.subdata(in: offset ..< min(offset + limit, data.count)))
            }
        }
        try append(expected)
        let source = try CBv2PagedCheckpointTensorSource(
            storage: storage, layerIndex: 0, values: true, admission: admission)
        let maximum = CBv2CompleteCheckpointManifest.maximumSegmentBytes
        var result = Data()
        while result.count < expected.count {
            let segment = try source.readSegment(byteOffset: result.count, maximumBytes: maximum)
            #expect(segment.count <= maximum)
            result.append(segment)
        }
        // A non-aligned request size rounds down only to the native dtype;
        // partial feature vectors and head/page boundaries remain byte-exact.
        let offset = (16 * 64 - 1) * dtype.size
        let fragment = try source.readSegment(byteOffset: offset, maximumBytes: 513 * dtype.size + 1)
        #expect(fragment == expected.subdata(in: offset ..< offset + 513 * dtype.size))
        #expect(throws: CBv2CompleteCheckpointError.invalidSegment) {
            try source.readSegment(byteOffset: 0, maximumBytes: maximum + 1)
        }
        source.close()
        try append(Data(repeating: 0, count: expected.count))
        storage.close()
        #expect(result == expected)
        #expect(throws: CBv2CompleteCheckpointError.closed) {
            try source.readSegment(byteOffset: 0, maximumBytes: dtype.size)
        }
        #expect(admission.bytesReserved == 0, "last source drops its mapped page permit")
    }

    @Test("pending native writes complete once for shared K/V and later suffix writes preserve the prefix",
          arguments: [DType.float16, .bfloat16, .float32])
    func capturedFenceAndPartialFrontier(dtype: DType) throws {
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 64 << 20)
        let backend = try PagedKVBackend(layerKinds: [kind], config: config(dtype))
        let rows = try backend.makeSequenceState(layerKinds: [kind], promptLength: 65, maxLength: 128)
        defer { backend.release(rows) }
        let row = try #require(rows[0] as? PagedSequenceKV)
        let keys = (MLXArray(0 ..< 2 * 65 * 64).asType(.float32) / 113)
            .reshaped([2, 65, 64]).asType(dtype)
        let values = (keys + 3).asType(dtype)
        // Build the oracle before constructing or evaluating native writes.
        let expectedKeys = keys.asData(access: .copy).data
        let expectedValues = values.asData(access: .copy).data
        row.write(keys: keys, values: values)
        let map = try CBv2PagedCheckpointPageMap(row: row, position: 65, admission: admission)
        let keySource = try CBv2PagedCheckpointTensorSource(pageMap: map, values: false)
        let valueSource = try CBv2PagedCheckpointTensorSource(pageMap: map, values: true)
        defer { keySource.close(); valueSource.close() }
        try withError { asyncEval(map.previous) }
        #expect(throws: CBv2CompleteCheckpointError.allocationFailed) {
            try map.prepareForReading { _ in throw CBv2CompleteCheckpointError.allocationFailed }
        }
        // The first read must itself wait; the failed wait above cannot mark
        // this source ready or permit a read of unfinished GPU output.
        #expect(try keySource.readSegment(byteOffset: 0, maximumBytes: expectedKeys.count) == expectedKeys)
        try map.prepareForReading { _ in
            Issue.record("shared K/V readiness evaluated an already completed fence")
        }
        #expect(try valueSource.readSegment(byteOffset: 0, maximumBytes: expectedValues.count) == expectedValues)

        row.write(keys: MLXArray.zeros([2, 5, 64], dtype: dtype),
                  values: MLXArray.zeros([2, 5, 64], dtype: dtype))
        try withError { eval(backend.pool.group(row.groupKey).writeFence) }
        #expect(try keySource.readSegment(byteOffset: 0, maximumBytes: expectedKeys.count) == expectedKeys)
        #expect(try valueSource.readSegment(byteOffset: 0, maximumBytes: expectedValues.count) == expectedValues)
    }
}
