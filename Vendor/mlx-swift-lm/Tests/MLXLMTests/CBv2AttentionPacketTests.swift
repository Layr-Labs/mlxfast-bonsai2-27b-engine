import Foundation
import MLX
import Testing
@_spi(Diagnostics) @testable import MLXLMCommon

@Suite("Native attention packet roundtrip", .serialized)
struct CBv2AttentionPacketTests {
    @Test(arguments: ["contiguous", "fixed", "segmented"],
          [DType.float16, .bfloat16, .float32])
    func nativeRoundtrip(backend: String, dtype: DType) throws {
        try roundtrip(backend: backend, query: dtype, storage: dtype, history: 32)
    }

    @Test(arguments: ["contiguous", "fixed", "segmented"], [DType.float16, .bfloat16])
    func originalWiderQueryIsPreserved(backend: String, dtype: DType) throws {
        try roundtrip(backend: backend, query: .float32, storage: dtype, history: 32)
    }

    @Test(arguments: ["fixed", "segmented"], [DType.bfloat16, .float32])
    func finalPartialPartitionBeyond4096HasAllNativeBytes(backend: String, dtype: DType) throws {
        try roundtrip(backend: backend, query: dtype, storage: dtype, history: 4_096)
    }

    private func roundtrip(backend: String, query: DType, storage: DType, history: Int) throws {
        let fixture = try AttentionPacketFixture(backend, dtype: storage, history: history)
        let (state, forward, q, k, v, output) = try fixture.capture(qType: query)
        #expect(forward.arrays.count == 6)
        #expect(state.tensors.isEmpty, "host bytes do not exist during graph construction")
        AttentionPacketFixture.finalize(forward, output: output)
        #expect(forward.arrays.isEmpty, "retirement must release every MLX handle")
        let packet = state.takeSnapshot()
        #expect(packet.evaluationStatus == "completed")
        #expect(packet.metadata.sampleOutcome == "confirmed")
        #expect(packet.metadata.expectedOwnerCount == 1 && packet.metadata.records.count == 1)
        #expect(packet.metadata.refusals.isEmpty)
        let record = try #require(packet.metadata.records.first)
        #expect(record.storageLayerIndex == 0 && record.modelLayerIndex == 3)
        #expect(record.offsetBefore == history && record.offsetAfter == history + 1)
        #expect(record.phase == "chained_decode" && record.scaleBits == Float(0.0625).bitPattern)
        #expect(record.dispatch == (backend == "contiguous" ? "contiguous_sdpa" : "paged_\(backend)_decode"))
        #expect(record.kernelOutputDType == String(describing: backend == "contiguous" ? query : storage))
        let expected = ["queries": q, "incomingKeys": k, "incomingValues": v, "output": output,
                        "storedKeys": concatenated([fixture.historyKeys, k], axis: 2),
                        "storedValues": concatenated([fixture.historyValues, v], axis: 2)]
        for (name, original) in expected {
            let tensor = try #require(packet.tensors[name])
            let raw = original.asData(access: .copy)
            #expect(tensor.dtype == String(describing: original.dtype))
            #expect(tensor.shape == original.shape && tensor.packedStrides == raw.strides)
            #expect(tensor.data == raw.data, "\(name) must preserve original native bytes")
            let rebuilt = MLXArray(tensor.data, tensor.shape, dtype: original.dtype)
            #expect(rebuilt.asData(access: .copy).data == raw.data)
        }
        // Verify reconstruction preserves the FP32 reference input exactly;
        // operator accuracy and release bars remain in the existing oracle suites.
        func decoded(_ name: String, _ dtype: DType) throws -> MLXArray {
            let value = try #require(packet.tensors[name])
            return MLXArray(value.data, value.shape, dtype: dtype).asType(.float32)
        }
        let reconstructed = PagedAttentionReference.composedAttention(
            queries: try decoded("queries", query), keys: try decoded("storedKeys", storage),
            values: try decoded("storedValues", storage), scale: 0.0625, boolMask: nil, sinks: nil)
        let original = PagedAttentionReference.composedAttention(
            queries: q.asType(.float32), keys: expected["storedKeys"]!.asType(.float32),
            values: expected["storedValues"]!.asType(.float32), scale: 0.0625, boolMask: nil, sinks: nil)
        #expect(reconstructed.asData(access: .copy).data == original.asData(access: .copy).data)
        #expect(packet.tensors.values.reduce(0) { $0 + $1.data.count } <= packet.reservedBytes)
        #expect(state.takeSnapshot().tensors.isEmpty)
    }

    @Test func invalidConfigurationAndConservativeBudgetRefuseBeforePayloadWork() throws {
        for (id, position, layer, bytes): (UInt64, Int, Int, Int) in [
            (0, 62, 0, 32), (2, 0, 0, 32), (2, 1_000_001, 0, 32), (2, 62, -1, 32),
            (2, 62, 1_024, 32), (2, 62, 0, 0), (2, 62, 0, CBv2AttentionPacketConfig.byteLimit + 1),
        ] {
            #expect(throws: CBv2AttentionPacketError.self) {
                try CBv2AttentionPacketConfig(requestID: id, outputIndex: position,
                                             storageLayerIndex: layer, maximumBytes: bytes)
            }
        }
        let state = try CBv2AttentionPacketState(.init(requestID: 2, outputIndex: 62, storageLayerIndex: 0))
        #expect(!state.reserve(queryHeads: 16, kvHeads: 2, headDim: 256, visibleTokens: Int.max))
        #expect(state.reservedBytes == 0 && state.tensors.isEmpty)
        let fixture = try AttentionPacketFixture("segmented", dtype: .bfloat16)
        let (refused, forward, _, _, _, output) = try fixture.capture(qType: .float32, maximumBytes: 1)
        #expect(forward.evaluationTargets.isEmpty)
        eval(output)
        forward.materialize(discarded: false)
        let packet = refused.takeSnapshot()
        #expect(packet.evaluationStatus == "refused" && packet.tensors.isEmpty)
        #expect(packet.metadata.refusals["packet_byte_budget_exhausted"] == 1)
        #expect(fixture.rows[0]?.absoluteOffset == 33, "the original attention still executes once")
    }
}
