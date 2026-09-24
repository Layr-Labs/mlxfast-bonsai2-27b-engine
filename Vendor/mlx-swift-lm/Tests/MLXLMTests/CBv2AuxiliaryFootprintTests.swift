import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Auxiliary allocator projection and capture", .serialized)
struct CBv2AuxiliaryFootprintTests {
    @Test func partitionedHistoryAndBatchedRowsStayWithinCapturedPolicy() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        for unit in [4, 32, 4096, 8192, 16384, 65536] {
            let projection = try #require(CBv2AuxiliaryAllocationProjection(policy: policy, buffers: [
                .init(bytesPerToken: unit, partitioned: true)
            ]))
            for partitions in [[1], [5], [1, 2, 5], [1, 255, 256, 257], [4097]] {
                let expected = try partitions.reduce(0) { sum, count in
                    sum + (try #require(policy.upperBound(byteCount: count * unit)))
                }
                let projected = try #require(projection.bytes(forTokens: partitions.reduce(0, +)))
                #expect(projected >= expected, "unit=\(unit), partitions=\(partitions)")
            }
        }
    }

    @Test func blockPaddingAndFirstTokenGrowthAreConservativeWithoutAllocation() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        let projection = try #require(CBv2AuxiliaryAllocationProjection(policy: policy, buffers: [
            .init(bytesPerToken: 512, allocationCount: 2, tokenGranularity: 256, tokenPadding: 4),
            .init(bytesPerToken: 32, tokenPadding: 4, partitioned: true),
            .init(bytesPerToken: 4, tokenPadding: 4, partitioned: true),
        ]))
        let before = Memory.snapshot()
        var previous = 0
        for tokens in 0 ... 1025 {
            let bytes = try #require(projection.bytes(forTokens: tokens))
            #expect(bytes >= previous && bytes - previous <= projection.maximumGrowthBytes)
            previous = bytes
            if tokens > 0 {
                let rows = ((tokens + 4 + 255) / 256) * 256
                let caches = 2 * (try #require(policy.upperBound(byteCount: rows * 512)))
                let hidden = try #require(policy.upperBound(byteCount: (tokens + 4) * 32))
                let history = try #require(policy.upperBound(byteCount: (tokens + 4) * 4))
                #expect(bytes >= caches + hidden + history)
            }
        }
        let after = Memory.snapshot()
        #expect(before.activeMemory == after.activeMemory && before.cacheMemory == after.cacheMemory)
        #expect(projection.bytes(forTokens: -1) == nil)
        #expect(projection.bytes(forTokens: Int.max) == nil)
        #expect(CBv2AuxiliaryAllocationProjection(policy: policy, buffers: [
            .init(bytesPerToken: Int.max, allocationCount: 2)
        ]) == nil)
    }

    @Test func coldAdmissionChargesPhysicalAuxiliaryAndRecurrentBatchPadding() throws {
        let policy = try #require(Memory.allocationFootprintPolicy())
        let spec = CBv2RecurrentStateSpec(layers: [.init(modelLayerIndex: 0,
            convShape: [1, 1024], convDType: .float32,
            ssmShape: [1, 2048], ssmDType: .float32)])
        let generation = try spec.allocationBytesPerGeneration(policy: policy)
        for batch in [1, 2, 5, 8, 17] {
            let buffers = (try #require(policy.upperBound(byteCount: batch * 4096)))
                + (try #require(policy.upperBound(byteCount: batch * 8192)))
            #expect(batch * generation >= buffers)
        }
        let projection = try #require(CBv2AuxiliaryAllocationProjection(policy: policy, buffers: [
            .init(bytesPerToken: 4096, allocationCount: 2, tokenGranularity: 16, tokenPadding: 4)
        ]))
        let kind = CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        var config = AdmissionV2.Config(watermarkFraction: 0, elementBytes: 2,
            layerElementBytes: nil, fixedBytesPerRequest: 3 * generation, auxiliaryBytesPerToken: 8192,
            auxiliaryTokenGranularity: 16, auxiliaryTokenAllocationPadding: 4)
        config.auxiliaryAllocationProjection = projection
        let expected = 4 * 17 + 3 * generation + (try #require(projection.bytes(forTokens: 17)))
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: expected, config: config)
        #expect(admission.allocatedBytes(forTokens: 17) == expected)
        try admission.reserve(id: .init(1), additionalTokens: 17)
        #expect(admission.bytesReserved == expected)
        #expect(admission.nonBackendBytesReserved == expected - 4 * 17)
        #expect(throws: CBv2KVError.self) { try admission.reserve(id: .init(2), additionalTokens: 1) }
        admission.releaseAll(id: .init(1))
        #expect(admission.nonBackendBytesReserved == 0 && admission.bytesReserved == 0)
    }

    @Test func captureChargesEntireRetainedBackingAndAllNewCopyBuffers() throws {
        let backing = MLXArray.zeros([1, 8, 1024], dtype: .float32)
        let row = backing[0..., 0 ..< 1, 0...]
        try withError { eval(backing, row) }
        let descriptors = try [
            CBv2CheckpointTensorDescriptor(role: .convolution, layer: 0, shape: [1, 8], dtype: .float32),
            .init(role: .recurrent, layer: 0, shape: row.shape, dtype: .float32),
            .init(role: .assistantHidden, shape: [1, 32, 8], dtype: .bfloat16),
            .init(role: .assistantTokens, shape: [1, 32], dtype: .int32),
            .init(role: .assistantFrontier, shape: [1, 1, 8], dtype: .bfloat16),
        ]
        let backingInfo = try backing.evaluatedBufferInfo()
        let retained = try #require(backingInfo).allocatedBytes
        #expect(retained > row.nbytes)
        let boolean = try Memory.allocationFootprintUpperBound(byteCount: MemoryLayout<Bool>.size)
        var expected = retained + 4 * boolean
        for index in [0, 4] { expected += try Memory.allocationFootprintUpperBound(byteCount: descriptors[index].byteCount) }
        for index in [2, 3] { expected += 2 * (try Memory.allocationFootprintUpperBound(byteCount: descriptors[index].byteCount)) }
        #expect(try CBv2CheckpointAllocationFootprint.captureBytes(descriptors,
            layers: [0: .init(conv: nil, ssm: row)]) == expected)
        #expect(throws: CBv2CompleteCheckpointError.allocationFailed) {
            try CBv2CheckpointAllocationFootprint.freshBytes([row])
        }
    }

    @Test func drainedFreshBackingStillRejectsAnExternalFullSizeView() throws {
        let stream = StreamOrDevice.default
        let array = MLXArray.zeros([1024], dtype: .float32, stream: stream)
        try withError { eval(array); stream.stream.synchronize() }
        let freshInfo = try array.evaluatedBufferInfo()
        let fresh = try #require(freshInfo)
        #expect(fresh.isUnique)
        let view = array.reshaped([2, 512], stream: stream)
        try withError { eval(view); stream.stream.synchronize() }
        let aliasedInfo = try array.evaluatedBufferInfo()
        let info = try #require(aliasedInfo)
        #expect(info.dataOffset == 0 && info.dataElements == array.size && info.isRowContiguous)
        #expect(!info.isUnique)
        #expect(throws: CBv2CompleteCheckpointError.allocationFailed) {
            try CBv2CheckpointAllocationFootprint.freshBytes([array])
        }
        withExtendedLifetime(view) {}
    }

}
