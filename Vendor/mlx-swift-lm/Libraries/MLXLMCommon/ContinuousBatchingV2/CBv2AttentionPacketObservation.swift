import MLX

extension CBv2AttentionPacketForward {
    func begin(cache: CBv2AttendingLayerCache, queries: MLXArray, keys: MLXArray,
               values: MLXArray, scale: Float, sinks: MLXArray?, softcap: Float?, spans: Bool)
        -> CBv2AttentionPacketObservation? {
        guard let observation = metadata.begin(
            cache: cache, queries: queries, keys: keys, values: values, scale: scale,
            sinks: sinks, softcap: softcap, spans: spans) else { return nil }
        let row = cache.rows[0]
        let end = row.absoluteOffset.addingReportingOverflow(1)
        guard !end.overflow, row.absoluteOffset >= 0,
            row.retainedCount == row.absoluteOffset else {
            state.refuse("packet_requires_complete_visible_history")
            return nil
        }
        guard state.reserve(queryHeads: cache.kind.queryHeads, kvHeads: cache.kind.kvHeads,
                            headDim: cache.kind.headDim, visibleTokens: end.partialValue)
        else { return nil }
        return .init(forward: self, metadata: observation, row: row,
                     queries: queries, keys: keys, values: values, visibleEnd: end.partialValue)
    }
}

/// Lives only during the selected ordinary forward. Native tensor handles move
/// into that exact in-flight step, never into the long-lived metadata snapshot.
final class CBv2AttentionPacketObservation {
    let forward: CBv2AttentionPacketForward
    let metadata: CBv2AttentionMetadataObservation
    private weak var row: (any CBv2SequenceKV)?
    private let queries: MLXArray
    private let keys: MLXArray
    private let values: MLXArray
    let visibleEnd: Int

    init(forward: CBv2AttentionPacketForward, metadata: CBv2AttentionMetadataObservation,
         row: CBv2SequenceKV, queries: MLXArray, keys: MLXArray, values: MLXArray, visibleEnd: Int) {
        self.forward = forward
        self.metadata = metadata
        self.row = row
        self.queries = queries
        self.keys = keys
        self.values = values
        self.visibleEnd = visibleEnd
    }

    func finishContiguous(keys: MLXArray, values: MLXArray, output: MLXArray) {
        metadata.finish(
            storage: ["visible_keys": .init(keys), "visible_values": .init(values)],
            kernelOutputDType: output.dtype, output: output, dispatch: "contiguous_sdpa")
        retain(keys: keys, values: values, output: output)
    }

    func retain(keys storedKeys: MLXArray, values storedValues: MLXArray, output: MLXArray) {
        guard let row, row.absoluteOffset == visibleEnd, row.retainedCount == visibleEnd,
            storedKeys.shape == [1, keys.dim(1), visibleEnd, keys.dim(3)],
            storedValues.shape == storedKeys.shape, output.shape == queries.shape,
            storedKeys.dtype == keys.dtype, storedValues.dtype == values.dtype,
            output.dtype == queries.dtype else {
            forward.state.refuse("packet_storage_shape_dtype_or_range_mismatch")
            return
        }
        guard forward.state.metadata.refusals.isEmpty,
            forward.state.metadata.records.count == 1 else { return }
        let tensors = ["queries": queries, "incomingKeys": keys, "incomingValues": values,
                       "storedKeys": storedKeys, "storedValues": storedValues, "output": output]
        let bytes = tensors.values.reduce(0) { $0 + $1.nbytes }
        guard bytes <= forward.state.reservedBytes else {
            forward.state.refuse("actual_packet_byte_budget_exhausted")
            return
        }
        forward.retain(tensors)
    }
}

protocol CBv2AttentionPacketBinding: AnyObject {
    var attentionPacket: CBv2AttentionPacketForward? { get set }
}

extension CBv2LayerCache: CBv2AttentionPacketBinding {}
extension PagedLayerCache: CBv2AttentionPacketBinding {}
