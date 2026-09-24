import MLX
import MLXLMCommon

extension GPTOSSModel: IncrementalCheckpointMaterializing {
    public var needsIncrementalCheckpointMaterialization: Bool {
        loraLayers.contains { ($0 as? GPTOSSTransformerBlock)?.mlp.experts.hasFusedGateUp == true }
    }

    public func materializeCheckpointWeightsIncrementally() throws {
        // Only the fused projections have new concat buffers. The loader
        // has relinquished its original shard/staging references first, so
        // evaluating one projection can retire that pair of split inputs.
        for case let layer as GPTOSSTransformerBlock in loraLayers {
            guard let projection = layer.mlp.experts.gateUpProj else { continue }
            try MLX.checkedEval(projection)
            Memory.clearCache()
        }
    }
}
