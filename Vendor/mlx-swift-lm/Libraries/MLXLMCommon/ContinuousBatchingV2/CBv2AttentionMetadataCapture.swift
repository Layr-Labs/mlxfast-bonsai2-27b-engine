import MLX

/// Engine-queue owned; no MLXArray is stored anywhere in this state graph.
final class CBv2AttentionMetadataState {
    let configuration: CBv2AttentionMetadataConfig
    let expectedOwners: Set<Int>
    private(set) var selectedForwards = 0
    private(set) var records: [CBv2AttentionMetadataRecord] = []
    private(set) var forwardSucceeded = false
    private(set) var refusals: [String: Int] = [:]
    private(set) var sampleOutcome = "not_selected"
    private(set) var seedToken: Int?
    private(set) var targetToken: Int?
    private var pendingForward: CBv2AttentionMetadataForward?

    init(_ configuration: CBv2AttentionMetadataConfig, expectedOwners: Set<Int>) {
        self.configuration = configuration
        self.expectedOwners = expectedOwners
    }

    func refuse(_ reason: String) { refusals[reason, default: 0] += 1 }

    func select(requestID: UInt64, outputIndex: Int, batchSize: Int, phase: String)
        -> CBv2AttentionMetadataForward?
    {
        guard requestID == configuration.requestID, outputIndex == configuration.outputIndex else {
            return nil
        }
        guard selectedForwards == 0 else {
            refuse("selected_forward_repeated")
            return nil
        }
        selectedForwards = 1
        guard batchSize == 1, phase == "decode" || phase == "chained_decode" else {
            refuse("ordinary_b1_decode_required")
            return nil
        }
        let forward = CBv2AttentionMetadataForward(state: self, phase: phase)
        sampleOutcome = "graph_constructed_unconfirmed"
        pendingForward = forward
        return forward
    }

    func append(_ record: CBv2AttentionMetadataRecord) {
        guard records.count < configuration.maximumRecords else {
            refuse("record_budget_exhausted")
            return
        }
        records.append(record)
    }

    func finish(succeeded: Bool, owners: Set<Int>) {
        forwardSucceeded = succeeded
        if !succeeded {
            refuse("forward_failed")
            discardPendingForward()
        }
        if owners != expectedOwners { refuse("missing_or_unexpected_attention_owner") }
    }

    func takePendingForward() -> CBv2AttentionMetadataForward? {
        defer { pendingForward = nil }
        return pendingForward
    }

    func discardPendingForward() {
        if pendingForward != nil { sampleOutcome = "forward_failed" }
        pendingForward = nil
    }

    func confirm(requestID: UInt64, outputIndex: Int, seed: Int?, target: Int) {
        guard requestID == configuration.requestID, outputIndex == configuration.outputIndex else {
            refuse("finalized_sample_identity_mismatch")
            return
        }
        sampleOutcome = "confirmed"
        seedToken = seed
        targetToken = target
    }

    func retire(discarded: Bool) {
        if discarded { sampleOutcome = "discarded" }
        else if sampleOutcome != "confirmed" { sampleOutcome = "retired_unconfirmed" }
    }

    func takeSnapshot() -> CBv2AttentionMetadataSnapshot {
        let snapshot = CBv2AttentionMetadataSnapshot(
            configuration: configuration, records: records, selectedForwards: selectedForwards,
            expectedOwnerCount: expectedOwners.count, forwardSucceeded: forwardSucceeded,
            sampleOutcome: sampleOutcome, seedToken: seedToken, targetToken: targetToken,
            refusals: refusals)
        records.removeAll(keepingCapacity: false)
        return snapshot
    }
}

/// Bound to concrete cache objects only while the selected normal forward runs.
final class CBv2AttentionMetadataForward {
    let state: CBv2AttentionMetadataState
    let phase: String
    var boundOwners: [ObjectIdentifier: CBv2AttentionOwnerIdentity] = [:]
    private var owners: Set<Int> = []

    init(state: CBv2AttentionMetadataState, phase: String) {
        self.state = state
        self.phase = phase
    }

    func begin(
        cache: CBv2AttendingLayerCache, queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, softcap: Float?, spans: Bool
    ) -> CBv2AttentionMetadataObservation? {
        guard let owner = boundOwners[ObjectIdentifier(cache)], owner.matches(cache),
            owners.insert(owner.storageLayerIndex).inserted else {
            state.refuse("unexpected_or_repeated_attention_owner")
            return nil
        }
        let kind = owner.kind
        guard kind.attention == .full, kind.sharesKVWithLayer == nil,
            !kind.isBidirectional, !kind.hasSinks, sinks == nil, softcap == nil, !spans,
            cache.rows.count == 1,
            queries.shape == [1, kind.queryHeads, 1, kind.headDim],
            keys.shape == [1, kind.kvHeads, 1, kind.headDim], values.shape == keys.shape,
            [DType.float16, .bfloat16, .float32].contains(queries.dtype),
            [DType.float16, .bfloat16, .float32].contains(keys.dtype), values.dtype == keys.dtype,
            scale.isFinite, scale > 0 else {
            state.refuse("unsupported_attention_geometry")
            return nil
        }
        return CBv2AttentionMetadataObservation(
            forward: self, layer: owner.storageLayerIndex, kind: kind,
            row: cache.rows[0], queries: queries, keys: keys, values: values,
            scale: scale)
    }

    func finish(succeeded: Bool) { state.finish(succeeded: succeeded, owners: owners) }
}

/// Host metadata only: original post-RoPE inputs survive as scalars and arrays
/// of integers, not lazy tensor handles or additional dependencies.
final class CBv2AttentionMetadataObservation {
    let forward: CBv2AttentionMetadataForward
    let layer: Int
    let kind: CBv2LayerKind
    let offset: Int
    private weak var row: (any CBv2SequenceKV)?
    let queries: CBv2AttentionTensorMetadata
    let keys: CBv2AttentionTensorMetadata
    let values: CBv2AttentionTensorMetadata
    let scale: Float

    init(forward: CBv2AttentionMetadataForward, layer: Int, kind: CBv2LayerKind, row: CBv2SequenceKV,
         queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float) {
        self.forward = forward
        self.layer = layer
        self.kind = kind
        self.offset = row.absoluteOffset
        self.row = row
        self.queries = .init(queries)
        self.keys = .init(keys)
        self.values = .init(values)
        self.scale = scale
    }

    func finish(storage: [String: CBv2AttentionTensorMetadata], kernelOutputDType: DType,
                output: MLXArray, dispatch: String) {
        guard let row, row.absoluteOffset == offset + 1 else {
            forward.state.refuse("unexpected_post_update_offset")
            return
        }
        forward.state.append(CBv2AttentionMetadataRecord(
            requestID: forward.state.configuration.requestID,
            outputIndex: forward.state.configuration.outputIndex, phase: forward.phase,
            batchIndex: 0, batchSize: 1, inputWidth: 1,
            storageLayerIndex: layer, modelLayerIndex: kind.modelLayerIndex ?? layer,
            offsetBefore: offset, offsetAfter: row.absoluteOffset, scaleBits: scale.bitPattern,
            queries: queries, incomingKeys: keys, incomingValues: values, storage: storage,
            kernelOutputDType: String(describing: kernelOutputDType), output: .init(output),
            dispatch: dispatch, sinksPresent: false, softcapPresent: false, spansPresent: false))
    }
}

protocol CBv2AttentionMetadataBinding: AnyObject {
    var attentionMetadata: CBv2AttentionMetadataForward? { get set }
}

extension CBv2LayerCache: CBv2AttentionMetadataBinding {}
extension PagedLayerCache: CBv2AttentionMetadataBinding {}
