import MLX

extension CBv2AttentionMetadataObservation {
    /// Called only by the branch that actually dispatched. Metadata describes
    /// native backing tensors, not a synthesized logical gather or new view.
    func finishPaged(group: PagedKVGroup, kernelOutputDType: DType,
                     output: MLXArray, dispatch: String) {
        var storage: [String: CBv2AttentionTensorMetadata] = [:]
        if group.segmentLayout != nil {
            guard group.segments.count <= 64 else {
                forward.state.refuse("storage_metadata_budget_exhausted")
                return
            }
            for (index, segment) in group.segments {
                storage["segment_\(index)_keys_and_values"] = .init(segment.storage)
            }
        } else {
            storage["key_slab"] = .init(group.kSlab)
            storage["value_slab"] = .init(group.vSlab)
        }
        finish(storage: storage, kernelOutputDType: kernelOutputDType,
               output: output, dispatch: dispatch)
    }
}
