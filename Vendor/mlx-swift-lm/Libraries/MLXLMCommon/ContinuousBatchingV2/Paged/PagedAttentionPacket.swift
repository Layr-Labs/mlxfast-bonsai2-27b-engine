import MLX

extension CBv2AttentionPacketObservation {
    /// Ordinary gather orders after the fused write and publishes its normal
    /// read back-edge. The selected step also blocks a chained successor until
    /// all packet reads are retired. There is no private unfenced read path.
    func finishPaged(group: PagedKVGroup, row: PagedSequenceKV, kernelOutputDType: DType,
                     output: MLXArray, dispatch: String) {
        metadata.finishPaged(group: group, kernelOutputDType: kernelOutputDType,
                             output: output, dispatch: dispatch)
        guard forward.state.metadata.refusals.isEmpty,
            row.absoluteOffset == visibleEnd, row.retainedCount == visibleEnd else {
            forward.state.refuse("packet_paged_range_or_metadata_mismatch")
            return
        }
        let stored = row.gatherRange(start: 0, count: visibleEnd)
        retain(keys: stored.keys, values: stored.values, output: output)
    }
}
