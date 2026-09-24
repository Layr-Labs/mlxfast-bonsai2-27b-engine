import MLX

final class CBv2CompleteCheckpointRetiredState: @unchecked Sendable {
    private var state: [CBv2SequenceKV?]
    init(state: [CBv2SequenceKV?]) { self.state = state }
    func release(backend: CBv2KVBackend) {
        backend.release(state)
        state.removeAll()
    }
}

extension EngineLoopV2 {
    func adoptCompleteCheckpoint(_ staged: CBv2StagedCompleteCheckpoint, requestID: CBv2RequestID)
        throws -> [CBv2SequenceKV?]
    {
        if staged.usesPagedBacking { return try adoptPagedCompleteCheckpoint(staged, requestID: requestID) }
        guard let contiguous = backend as? CBv2ContiguousKVBackend else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        return try staged.consumePreparedState { prepared in
            guard let checkpoint = prepared.checkpoint else {
                throw CBv2CompleteCheckpointError.incompleteTransfer
            }
            try contiguous.adoptPreparedCheckpoint(prepared.state)
            do {
                try adoptRecurrentCheckpoint(checkpoint, requestID: requestID)
            } catch {
                contiguous.release(prepared.state)
                throw error
            }
            return prepared.state
        }
    }
}
