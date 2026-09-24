import MLX

/// A loaded causal target whose complete state is attention KV. A per-round
/// assistant may rebuild from target rows; persistent assistants need their own
/// checkpoint codec and are deliberately excluded from this contract.
public protocol CBv2HistoricalAttentionCheckpointProviding {
    var cbv2SupportsHistoricalAttentionCheckpoint: Bool { get }
}

/// Canonical compact-layer mapping. Borrowers name an earlier owning row, never
/// another borrower. The same table drives disk identity, tensors and adoption.
public struct CBv2CheckpointAttentionLayer: Codable, Sendable, Equatable {
    public let modelLayer: Int
    public let owner: Int
    public let window: Int?
    public let kvHeads: Int
    public let headDim: Int
    public let queryHeads: Int
    public let hasSinks: Bool
    public let dtype: CBv2CheckpointDType

    /// Provider identity must use the same owner map as native staging.
    public static func resolve(layerKinds: [CBv2LayerKind], dtypes: [DType]) throws -> [Self] {
        try CBv2HistoricalAttentionLayout(layerKinds: layerKinds, dtypes: dtypes).layers
    }

    func tokenStart(at position: Int) -> Int { window.map { max(0, position - $0) } ?? 0 }
}

struct CBv2HistoricalAttentionLayout: Sendable {
    let layers: [CBv2CheckpointAttentionLayer]
    var owningIndices: [Int] { layers.indices.filter { layers[$0].owner == $0 } }

    init(layerKinds: [CBv2LayerKind], dtypes: [DType]) throws {
        guard !layerKinds.isEmpty, layerKinds.count <= 2048, layerKinds.count == dtypes.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var result: [CBv2CheckpointAttentionLayer] = []
        var modelIndices = Set<Int>()
        for (index, kind) in layerKinds.enumerated() {
            guard !kind.isBidirectional, kind.kvHeads > 0, kind.headDim >= 64,
                  kind.queryHeads > 0, kind.queryHeads % kind.kvHeads == 0,
                  (kind.modelLayerIndex ?? index) >= 0, let dtype = CBv2CheckpointDType(dtypes[index]), dtype != .int32,
                  modelIndices.insert(kind.modelLayerIndex ?? index).inserted
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            let window: Int?
            switch kind.attention {
            case .full: window = nil
            case .slidingWindow(let size):
                guard size > 0 else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                window = size
            }
            let owner = kind.sharesKVWithLayer ?? index
            if owner != index {
                guard owner >= 0, owner < index, result[owner].owner == owner,
                      result[owner].window == window, result[owner].kvHeads == kind.kvHeads,
                      result[owner].headDim == kind.headDim, result[owner].dtype == dtype
                else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            } else if kind.sharesKVWithLayer != nil {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            result.append(.init(modelLayer: kind.modelLayerIndex ?? index, owner: owner,
                                window: window, kvHeads: kind.kvHeads, headDim: kind.headDim,
                                queryHeads: kind.queryHeads, hasSinks: kind.hasSinks, dtype: dtype))
        }
        layers = result
    }

    func tensorDescriptors(position: Int) throws -> [CBv2CheckpointTensorDescriptor] {
        try owningIndices.flatMap { index in
            let layer = layers[index]
            return try [CBv2CheckpointTensorRole.keys, .values].map {
                try CBv2CheckpointTensorDescriptor(role: $0, layer: layer.modelLayer,
                          shape: [1, layer.kvHeads, position - layer.tokenStart(at: position), layer.headDim],
                          dtype: layer.dtype)
            }
        }
    }
}
