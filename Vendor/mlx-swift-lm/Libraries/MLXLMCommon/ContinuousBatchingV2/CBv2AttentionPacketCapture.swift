import Cmlx
import MLX

/// Engine-queue owner of one pending forward and, after retirement, host bytes.
final class CBv2AttentionPacketState {
    let configuration: CBv2AttentionPacketConfig
    let metadata: CBv2AttentionMetadataState
    private var pending: CBv2AttentionPacketForward?
    private(set) var reservedBytes = 0
    private(set) var evaluationStatus = "not_selected"
    private(set) var tensors: [String: CBv2AttentionPacketTensor] = [:]

    init(_ configuration: CBv2AttentionPacketConfig) throws {
        self.configuration = configuration
        // The public configuration has already validated these shared bounds.
        let selection = try CBv2AttentionMetadataConfig(
            requestID: configuration.requestID, outputIndex: configuration.outputIndex,
            maximumRecords: 1)
        metadata = CBv2AttentionMetadataState(
            selection, expectedOwners: [configuration.storageLayerIndex])
    }

    func select(requestID: UInt64, outputIndex: Int, batchSize: Int, phase: String)
        -> CBv2AttentionPacketForward? {
        guard let selected = metadata.select(requestID: requestID, outputIndex: outputIndex,
                                            batchSize: batchSize, phase: phase) else { return nil }
        let forward = CBv2AttentionPacketForward(state: self, metadata: selected)
        pending = forward
        evaluationStatus = "pending"
        return forward
    }

    func refuse(_ reason: String) {
        metadata.refuse(reason)
        evaluationStatus = "refused"
    }

    /// Reserve before retaining/copying tensor payloads. FP32 widths bound all
    /// supported native dtypes; this deliberately is not a context-length limit.
    func reserve(queryHeads: Int, kvHeads: Int, headDim: Int, visibleTokens: Int) -> Bool {
        guard queryHeads > 0, kvHeads > 0, headDim > 0, visibleTokens > 0,
            queryHeads % kvHeads == 0, reservedBytes == 0 else {
            refuse("invalid_packet_geometry_or_repeated_owner")
            return false
        }
        var bytes = 0
        for shape in [[2, queryHeads, headDim, 4], [2, kvHeads, headDim, 4],
                      [2, kvHeads, visibleTokens, headDim, 4]] {
            var product = 1
            for dimension in shape {
                let next = product.multipliedReportingOverflow(by: dimension)
                guard !next.overflow else { refuse("packet_byte_budget_exhausted"); return false }
                product = next.partialValue
            }
            let next = bytes.addingReportingOverflow(product)
            guard !next.overflow, next.partialValue <= configuration.maximumBytes else {
                refuse("packet_byte_budget_exhausted")
                return false
            }
            bytes = next.partialValue
        }
        reservedBytes = bytes
        return true
    }

    func takePendingForward() -> CBv2AttentionPacketForward? {
        defer { pending = nil }
        _ = metadata.takePendingForward()
        return pending
    }

    func discardPendingForward() {
        pending?.releaseArrays()
        pending = nil
        metadata.discardPendingForward()
        if evaluationStatus == "pending" { evaluationStatus = "forward_failed" }
    }

    func complete(_ values: [String: CBv2AttentionPacketTensor]) {
        tensors = values
        evaluationStatus = "completed"
    }

    func retireUnconfirmed() {
        if evaluationStatus == "pending" { evaluationStatus = metadata.sampleOutcome }
    }

    func takeSnapshot() -> CBv2AttentionPacketSnapshot {
        defer { tensors.removeAll(keepingCapacity: false) }
        return .init(configuration: configuration, metadata: metadata.takeSnapshot(),
                     evaluationStatus: evaluationStatus, reservedBytes: reservedBytes, tensors: tensors)
    }
}

final class CBv2AttentionPacketForward {
    let state: CBv2AttentionPacketState
    let metadata: CBv2AttentionMetadataForward
    private(set) var arrays: [String: MLXArray] = [:]

    init(state: CBv2AttentionPacketState, metadata: CBv2AttentionMetadataForward) {
        self.state = state
        self.metadata = metadata
    }

    var evaluationTargets: [MLXArray] { Array(arrays.values) }
    func retain(_ values: [String: MLXArray]) { arrays = values }
    func releaseArrays() { arrays.removeAll(keepingCapacity: false) }

    func finish(succeeded: Bool) {
        metadata.finish(succeeded: succeeded)
        if !succeeded { state.discardPendingForward(); releaseArrays() }
        else if arrays.count != 6 { state.refuse("incomplete_attention_packet") }
    }

    /// Called after this step's existing eval fence and normal cost stamp.
    /// asData copies evaluated native bytes into packed host-owned storage.
    func materialize(discarded: Bool) {
        defer { releaseArrays() }
        state.metadata.retire(discarded: discarded)
        guard !discarded, state.metadata.sampleOutcome == "confirmed",
            state.metadata.forwardSucceeded, state.metadata.refusals.isEmpty,
            arrays.count == 6 else { state.retireUnconfirmed(); return }
        // asData(.copy) calls eval internally. Prove every original handle is
        // already available without waiting or scheduling, so the copy cannot
        // evaluate a pending diagnostic graph after the normal step fence.
        for array in arrays.values {
            var available = false
            guard _mlx_array_is_available(&available, array.ctx) == 0, available else {
                state.refuse("packet_tensor_not_available_after_step_fence")
                return
            }
        }
        var result: [String: CBv2AttentionPacketTensor] = [:]
        var bytes = 0
        for (name, array) in arrays {
            let raw = array.asData(access: .copy)
            bytes += raw.data.count
            guard bytes <= state.reservedBytes, raw.data.count == array.nbytes,
                raw.shape == array.shape, raw.dType == array.dtype else {
                state.refuse("materialized_packet_shape_or_budget_mismatch")
                return
            }
            result[name] = .init(dtype: String(describing: raw.dType), shape: raw.shape,
                                 packedStrides: raw.strides, data: raw.data)
        }
        state.complete(result)
    }
}
