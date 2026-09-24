import MLX
import Testing
@_spi(Diagnostics) @testable import MLXLMCommon

@Suite("Bounded actual attention metadata", .serialized)
struct CBv2AttentionMetadataTests {
    private let kind = CBv2LayerKind(
        attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2, modelLayerIndex: 3)

    @Test func invalidConfigurationAndUnsupportedBatchAreExplicit() throws {
        for (request, output, records): (UInt64, Int, Int) in [
            (0, 62, 64), (2, 0, 64), (2, -1, 64), (2, 1_000_001, 64),
            (2, 62, 0), (2, 62, 129),
        ] {
            #expect(throws: CBv2AttentionMetadataError.self) {
                try CBv2AttentionMetadataConfig(
                    requestID: request, outputIndex: output, maximumRecords: records)
            }
        }
        let state = CBv2AttentionMetadataState(try .init(requestID: 2, outputIndex: 62), expectedOwners: [0])
        #expect(state.select(requestID: 1, outputIndex: 62, batchSize: 1, phase: "decode") == nil)
        #expect(state.select(requestID: 2, outputIndex: 61, batchSize: 1, phase: "decode") == nil)
        #expect(state.selectedForwards == 0)
        #expect(state.select(requestID: 2, outputIndex: 62, batchSize: 2, phase: "decode") == nil)
        #expect(state.takeSnapshot().refusals["ordinary_b1_decode_required"] == 1)
    }

    @Test(arguments: ["contiguous", "fixed", "segmented"],
          [DType.bfloat16, .float32])
    func nativeDTypesAndActualRoute(backendName: String, dtype: DType) throws {
        try observe(backendName: backendName, qType: dtype, kvType: dtype)
    }

    @Test(arguments: ["contiguous", "fixed", "segmented"])
    func widerQueryIsRecordedBeforeNativeKVConversion(backendName: String) throws {
        try observe(backendName: backendName, qType: .float32, kvType: .bfloat16)
    }

    private func observe(backendName: String, qType: DType, kvType: DType) throws {
        let backend: any CBv2KVBackend
        let cache: any CBv2AttendingLayerCache
        if backendName == "contiguous" {
            backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20))
            cache = CBv2LayerCache(layerIndex: 0, kind: kind)
        } else {
            var config = PagedKVPoolConfig(
                capacityBytes: 1 << 20, dtype: kvType, maxPrefillChunk: 16,
                nominalMaxSequenceLength: 64)
            if backendName == "segmented" { config.segmentSizeBytes = 16_384 }
            let paged = try PagedKVBackend(layerKinds: [kind], config: config)
            backend = paged
            cache = paged.makeLayerCaches()[0]
        }
        let rows = try backend.makeSequenceState(layerKinds: [kind], promptLength: 16, maxLength: 64)
        defer { cache.setRows([]); backend.release(rows) }
        cache.setRows([try #require(rows[0])])
        let binding = try #require(cache as? any CBv2AttentionMetadataBinding)
        #expect(binding.attentionMetadata == nil)
        let promptK = MLXArray.zeros([1, 1, 16, 64], dtype: kvType)
        let promptQ = MLXArray.zeros([1, 2, 16, 64], dtype: qType)
        let promptOutput = cache.updateAndAttend(
            queries: promptQ, keys: promptK, values: promptK, scale: 0.125, sinks: nil)
        eval(promptOutput)
        let state = CBv2AttentionMetadataState(try .init(requestID: 2, outputIndex: 62), expectedOwners: [0])
        let forward = try #require(state.select(
            requestID: 2, outputIndex: 62, batchSize: 1, phase: "chained_decode"))
        #expect(forward.bindOwner(cache: cache, storageLayerIndex: 0, kind: kind))
        binding.attentionMetadata = forward
        defer { binding.attentionMetadata = nil; state.discardPendingForward() }
        let q = MLXArray.zeros([1, 2, 1, 64], dtype: qType)
        let k = MLXArray.zeros([1, 1, 1, 64], dtype: kvType)
        let output = cache.updateAndAttend(queries: q, keys: k, values: k, scale: 0.125, sinks: nil)
        // Snapshot exists before eval: capture must inspect graph metadata only.
        forward.finish(succeeded: true)
        let snapshot = state.takeSnapshot()
        #expect(snapshot.records.count == 1)
        #expect(snapshot.sampleOutcome == "graph_constructed_unconfirmed")
        #expect(snapshot.targetToken == nil)
        #expect(snapshot.refusals.isEmpty)
        let record = try #require(snapshot.records.first)
        #expect(record.queries.dtype == String(describing: qType))
        #expect(record.incomingKeys.dtype == String(describing: kvType))
        #expect(record.incomingValues.dtype == String(describing: kvType))
        #expect(record.storage.values.allSatisfy { $0.dtype == String(describing: kvType) })
        #expect(!record.storage.isEmpty)
        #expect(record.queries.shape == [1, 2, 1, 64])
        #expect(record.incomingKeys.shape == [1, 1, 1, 64])
        #expect(record.queries.graphConstructionStrides.count == 4)
        #expect(record.storageLayerIndex == 0 && record.modelLayerIndex == 3)
        #expect(record.offsetBefore == 16 && record.offsetAfter == 17)
        #expect(record.scaleBits == Float(0.125).bitPattern)
        #expect(record.output.dtype == String(describing: qType))
        #expect(record.kernelOutputDType == String(describing: backendName == "contiguous" ? qType : kvType))
        #expect(record.dispatch == (backendName == "contiguous" ? "contiguous_sdpa" : "paged_\(backendName)_decode"))
        eval(output)
        #expect(state.takeSnapshot().records.isEmpty)
        #expect(state.select(requestID: 2, outputIndex: 62, batchSize: 1, phase: "decode") == nil,
                "drain does not refill the one-forward capture budget")
    }

    @Test(arguments: [5, 64])
    func allTenCompactOwnersKeepOriginalModelLayerIdentity(maximum: Int) throws {
        let state = CBv2AttentionMetadataState(try .init(requestID: 2, outputIndex: 62, maximumRecords: maximum),
                                              expectedOwners: Set(0..<10))
        let forward = try #require(state.select(
            requestID: 2, outputIndex: 62, batchSize: 1, phase: "decode"))
        defer { state.discardPendingForward() }
        var ownedCaches: [CBv2LayerCache] = []
        for index in 0..<10 {
            var kind = kind
            kind.modelLayerIndex = 3 + 4 * index
            let row = CBv2FullSequenceKV(promptLength: 1, maxLength: 8, kvHeads: 1, headDim: 64)
            let cache = CBv2LayerCache(layerIndex: index, kind: kind, rows: [row])
            ownedCaches.append(cache)
            #expect(forward.bindOwner(cache: cache, storageLayerIndex: index, kind: kind))
            cache.attentionMetadata = forward
            let q = MLXArray.zeros([1, 2, 1, 64])
            let k = MLXArray.zeros([1, 1, 1, 64])
            let output = cache.updateAndAttend(queries: q, keys: k, values: k, scale: 0.125, sinks: nil)
            cache.attentionMetadata = nil
            eval(output)
        }
        forward.finish(succeeded: true)
        #expect(ownedCaches.count == 10)
        let snapshot = state.takeSnapshot()
        let count = min(10, maximum)
        #expect(snapshot.records.map(\.storageLayerIndex) == Array(0..<count))
        #expect(snapshot.records.map(\.modelLayerIndex) == (0..<count).map { 3 + 4 * $0 })
        if maximum < 10 { #expect(snapshot.refusals["record_budget_exhausted"] == 10 - maximum) }
        else { #expect(snapshot.refusals.isEmpty) }
    }

    @Test func unsupportedGeometryRefusesMetadataWithoutDispatchingAdditionalWork() throws {
        let state = CBv2AttentionMetadataState(try .init(requestID: 2, outputIndex: 62), expectedOwners: [0])
        let forward = try #require(state.select(
            requestID: 2, outputIndex: 62, batchSize: 1, phase: "decode"))
        defer { state.discardPendingForward() }
        let row = CBv2FullSequenceKV(promptLength: 1, maxLength: 8, kvHeads: 1, headDim: 64)
        let cache = CBv2LayerCache(layerIndex: 0, kind: kind, rows: [row])
        #expect(forward.bindOwner(cache: cache, storageLayerIndex: 0, kind: kind))
        let q = MLXArray.zeros([1, 2, 2, 64])
        let k = MLXArray.zeros([1, 1, 2, 64])
        #expect(forward.begin(cache: cache, queries: q, keys: k, values: k,
                              scale: 0.125, sinks: nil, softcap: nil, spans: false) == nil)
        forward.finish(succeeded: true)
        #expect(row.absoluteOffset == 0, "refusal must not update storage or perform an extra forward")
        let snapshot = state.takeSnapshot()
        #expect(snapshot.records.isEmpty)
        #expect(snapshot.refusals["unsupported_attention_geometry"] == 1)
    }

    @Test func failedOrDiscardedForwardNeverBecomesConfirmed() throws {
        for succeeded in [false, true] {
            let state = CBv2AttentionMetadataState(try .init(requestID: 2, outputIndex: 62), expectedOwners: [0])
            var forward = state.select(requestID: 2, outputIndex: 62, batchSize: 1, phase: "decode")
            weak var released = forward
            forward?.finish(succeeded: succeeded)
            if succeeded {
                let inFlight = state.takePendingForward()
                inFlight?.state.retire(discarded: true)
            }
            forward = nil
            #expect(released == nil, "no diagnostic ownership cycle survives retirement")
            let snapshot = state.takeSnapshot()
            #expect(snapshot.sampleOutcome == (succeeded ? "discarded" : "forward_failed"))
            #expect(snapshot.targetToken == nil)
            #expect(snapshot.refusals["missing_or_unexpected_attention_owner"] == 1)
        }
    }
}
