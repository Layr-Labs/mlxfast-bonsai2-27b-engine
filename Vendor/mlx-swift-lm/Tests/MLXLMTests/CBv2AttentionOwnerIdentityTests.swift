import MLX
import Testing
@_spi(Diagnostics) @testable import MLXLMCommon

@Suite("Attention metadata bound owner identity", .serialized)
struct CBv2AttentionOwnerIdentityTests {
    private let kind = CBv2LayerKind(
        attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2, modelLayerIndex: 39)

    private func forward() throws -> CBv2AttentionMetadataForward {
        let state = CBv2AttentionMetadataState(
            try .init(requestID: 2, outputIndex: 62), expectedOwners: [9])
        return try #require(state.select(requestID: 2, outputIndex: 62, batchSize: 1, phase: "decode"))
    }

    @Test(arguments: [9, 39])
    func denseAndOriginalCacheIndicesRecordTheSameOwner(cacheIndex: Int) throws {
        let forward = try forward()
        defer { forward.state.discardPendingForward() }
        let row = CBv2FullSequenceKV(promptLength: 1, maxLength: 4, kvHeads: 1, headDim: 64)
        let cache = CBv2LayerCache(layerIndex: cacheIndex, kind: kind, rows: [row])
        #expect(forward.bindOwner(cache: cache, storageLayerIndex: 9, kind: kind))
        cache.attentionMetadata = forward
        let q = MLXArray.zeros([1, 2, 1, 64])
        let k = MLXArray.zeros([1, 1, 1, 64])
        let output = cache.updateAndAttend(queries: q, keys: k, values: k, scale: 0.125, sinks: nil)
        cache.attentionMetadata = nil
        forward.finish(succeeded: true)
        let record = try #require(forward.state.takeSnapshot().records.first)
        #expect(record.storageLayerIndex == 9 && record.modelLayerIndex == 39)
        #expect(cache.layerIndex == cacheIndex, "diagnostics must never rewrite backend cache indices")
        eval(output)
    }

    @Test func wrongDuplicateAndUnboundOwnersAreRefused() throws {
        for mode in ["wrong-kind", "wrong-cache-index", "wrong-storage", "duplicate-object", "duplicate-storage"] {
            let forward = try forward()
            defer { forward.state.discardPendingForward() }
            let cache = CBv2LayerCache(layerIndex: 39, kind: kind)
            var wrongKind = kind
            wrongKind.modelLayerIndex = 35
            switch mode {
            case "wrong-kind":
                #expect(!forward.bindOwner(cache: cache, storageLayerIndex: 9, kind: wrongKind))
            case "wrong-cache-index":
                let wrong = CBv2LayerCache(layerIndex: 35, kind: kind)
                #expect(!forward.bindOwner(cache: wrong, storageLayerIndex: 9, kind: kind))
            case "wrong-storage":
                #expect(!forward.bindOwner(cache: cache, storageLayerIndex: 39, kind: kind))
            default:
                #expect(forward.bindOwner(cache: cache, storageLayerIndex: 9, kind: kind))
                let duplicate = mode == "duplicate-object" ? cache : CBv2LayerCache(layerIndex: 9, kind: kind)
                #expect(!forward.bindOwner(cache: duplicate, storageLayerIndex: 9, kind: kind))
            }
            #expect(forward.state.takeSnapshot().refusals["invalid_or_repeated_attention_owner_binding"] == 1)
        }
        let forward = try forward()
        defer { forward.state.discardPendingForward() }
        let cache = CBv2LayerCache(layerIndex: 39, kind: kind)
        let q = MLXArray.zeros([1, 2, 1, 64])
        let k = MLXArray.zeros([1, 1, 1, 64])
        #expect(forward.begin(cache: cache, queries: q, keys: k, values: k,
                              scale: 0.125, sinks: nil, softcap: nil, spans: false) == nil)
        #expect(forward.state.takeSnapshot().refusals["unexpected_or_repeated_attention_owner"] == 1)
    }

    @Test func aSecondCallFromTheSameBoundOwnerCannotAddAnotherRecord() throws {
        let forward = try forward()
        defer { forward.state.discardPendingForward() }
        let row = CBv2FullSequenceKV(promptLength: 1, maxLength: 4, kvHeads: 1, headDim: 64)
        let cache = CBv2LayerCache(layerIndex: 39, kind: kind, rows: [row])
        #expect(forward.bindOwner(cache: cache, storageLayerIndex: 9, kind: kind))
        let q = MLXArray.zeros([1, 2, 1, 64])
        let k = MLXArray.zeros([1, 1, 1, 64])
        #expect(forward.begin(cache: cache, queries: q, keys: k, values: k,
                              scale: 0.125, sinks: nil, softcap: nil, spans: false) != nil)
        #expect(forward.begin(cache: cache, queries: q, keys: k, values: k,
                              scale: 0.125, sinks: nil, softcap: nil, spans: false) == nil)
        #expect(forward.state.takeSnapshot().refusals["unexpected_or_repeated_attention_owner"] == 1)
    }
}
