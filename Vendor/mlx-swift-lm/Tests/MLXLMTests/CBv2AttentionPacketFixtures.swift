import Foundation
import MLX
import Testing
@_spi(Diagnostics) @testable import MLXLMCommon

/// Native attention only; no model forward is added by this diagnostic fixture.
final class AttentionPacketFixture {
    let kind = CBv2LayerKind(
        attention: .full, headDim: 256, kvHeads: 2, queryHeads: 16, modelLayerIndex: 3)
    let backend: any CBv2KVBackend
    let cache: any CBv2AttendingLayerCache
    let rows: [CBv2SequenceKV?]
    let dtype: DType
    let historyKeys: MLXArray
    let historyValues: MLXArray
    private var released = false

    init(_ name: String, dtype: DType, history: Int = 32) throws {
        self.dtype = dtype
        if name == "contiguous" {
            backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 64 << 20, kvDType: dtype))
            cache = CBv2LayerCache(layerIndex: 3, kind: kind)
        } else {
            var config = PagedKVPoolConfig(capacityBytes: 64 << 20, dtype: dtype,
                maxPrefillChunk: 4_096, nominalMaxSequenceLength: history + 64)
            if name == "segmented" { config.segmentSizeBytes = 1 << 20 }
            let paged = try PagedKVBackend(layerKinds: [kind], config: config)
            backend = paged
            cache = paged.makeLayerCaches()[0]
        }
        rows = try backend.makeSequenceState(layerKinds: [kind], promptLength: history, maxLength: history + 64)
        let row = try #require(rows[0])
        cache.setRows([row])
        let count = 2 * history * 256
        historyKeys = MLXArray((0..<count).map { Float(($0 % 97) - 48) / 128 })
            .reshaped([1, 2, history, 256]).asType(dtype)
        historyValues = MLXArray((0..<count).map { Float(($0 % 53) - 26) / 64 })
            .reshaped([1, 2, history, 256]).asType(dtype)
        let stored = row.update(keys: historyKeys, values: historyValues)
        eval(stored.0, stored.1)
    }

    deinit { release() }
    func release() {
        guard !released else { return }
        cache.setRows([])
        backend.release(rows)
        released = true
    }

    func capture(qType: DType, maximumBytes: Int = CBv2AttentionPacketConfig.byteLimit)
        throws -> (CBv2AttentionPacketState, CBv2AttentionPacketForward, MLXArray, MLXArray, MLXArray, MLXArray) {
        let state = try CBv2AttentionPacketState(.init(
            requestID: 2, outputIndex: 62, storageLayerIndex: 0, maximumBytes: maximumBytes))
        let forward = try #require(state.select(requestID: 2, outputIndex: 62, batchSize: 1,
                                                phase: "chained_decode"))
        #expect(forward.metadata.bindOwner(cache: cache, storageLayerIndex: 0, kind: kind))
        let binding = try #require(cache as? any CBv2AttentionPacketBinding)
        #expect(binding.attentionPacket == nil)
        binding.attentionPacket = forward
        defer { binding.attentionPacket = nil }
        let q = MLXArray((0..<(16 * 256)).map { Float(($0 % 17) - 8) / 19 })
            .reshaped([1, 16, 1, 256]).asType(qType)
        let k = MLXArray((0..<(2 * 256)).map { Float(($0 % 29) - 14) / 32 })
            .reshaped([1, 2, 1, 256]).asType(dtype)
        let v = MLXArray((0..<(2 * 256)).map { Float(($0 % 41) - 20) / 16 })
            .reshaped([1, 2, 1, 256]).asType(dtype)
        let output = cache.updateAndAttend(queries: q, keys: k, values: v, scale: 0.0625, sinks: nil)
        forward.finish(succeeded: true)
        _ = state.takePendingForward()
        return (state, forward, q, k, v, output)
    }

    static func finalize(_ forward: CBv2AttentionPacketForward, output: MLXArray) {
        eval([output] + forward.evaluationTargets)
        forward.state.metadata.confirm(requestID: 2, outputIndex: 62, seed: 11346, target: 1928)
        forward.materialize(discarded: false)
    }
}
