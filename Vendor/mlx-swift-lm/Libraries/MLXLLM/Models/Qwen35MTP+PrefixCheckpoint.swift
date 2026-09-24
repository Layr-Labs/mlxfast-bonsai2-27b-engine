import Cmlx
import MLX
import MLXLMCommon

extension Qwen35InlineMTPAssistant: CBv2MTPPrefixCheckpointCoding {
    private final class PrefixCheckpoint: CBv2MTPPrefixCheckpoint {
        let owner: ObjectIdentifier
        let targetInputCount: Int
        let hidden: MLXArray
        let tokens: MLXArray
        let frontier: MLXArray

        init(owner: ObjectIdentifier, count: Int, hidden: MLXArray, tokens: MLXArray, frontier: MLXArray) {
            self.owner = owner
            self.targetInputCount = count
            self.hidden = hidden
            self.tokens = tokens
            self.frontier = frontier
        }

        var materializedBytes: Int { hidden.nbytes + tokens.nbytes + frontier.nbytes }
        var evaluationTargets: [MLXArray] { [hidden, tokens, frontier] }
    }

    public func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount: Int
    ) -> (any CBv2MTPPrefixCheckpoint)? {
        guard let state = requestState as? RequestState,
            !state.isReleased, !state.roundInFlight, state.cacheOffset == 0,
            targetInputCount > 1, state.committedInputCount == targetInputCount - 1,
            state.backlogHidden.count == state.backlogTokens.count,
            !state.backlogHidden.isEmpty, let frontier = state.targetHiddenFrontier
        else { return nil }
        // Hidden rows were normalized once when the target committed them.
        // Compact byte-preserving copies drop views of later prompt rows;
        // no assistant forward or normalization is repeated at restore.
        let hidden = concatenated(state.backlogHidden, axis: 1)
        let tokens = concatenated(state.backlogTokens, axis: 1)
        guard hidden.dim(1) == targetInputCount - 1,
            tokens.dim(1) == targetInputCount - 1
        else { return nil }
        return PrefixCheckpoint(
            owner: ObjectIdentifier(self), count: targetInputCount,
            hidden: MLX.where(MLXArray(true), hidden, hidden),
            tokens: MLX.where(MLXArray(true), tokens, tokens),
            frontier: MLX.where(MLXArray(true), frontier, frontier))
    }

    public func restorePrefixCheckpoint(
        _ checkpoint: any CBv2MTPPrefixCheckpoint
    ) -> (any CBv2MTPRequestState)? {
        guard let checkpoint = checkpoint as? PrefixCheckpoint,
            checkpoint.owner == ObjectIdentifier(self),
            let state = makeRequestState() as? RequestState
        else { return nil }
        // Each request gets fresh mutable head caches. Only immutable trusted
        // history is shared; its first draft still consumes the entire history.
        state.backlogHidden = [checkpoint.hidden]
        state.backlogTokens = [checkpoint.tokens]
        state.targetHiddenFrontier = checkpoint.frontier
        return state
    }

    public var prefixCheckpointCodecID: String {
        "qwen-trusted-normalized-history-v1:\(prefixCheckpointGeometry.verification)"
    }

    public func prefixCheckpointTensorDescriptors(targetInputCount: Int)
        -> [CBv2CheckpointTensorDescriptor]?
    {
        let geometry = prefixCheckpointGeometry
        guard targetInputCount > 1, let dtype = CBv2CheckpointDType(geometry.dtype), dtype != .int32 else {
            return nil
        }
        return try? [
            .init(role: .assistantHidden, shape: [1, targetInputCount - 1, geometry.width], dtype: dtype),
            .init(role: .assistantTokens, shape: [1, targetInputCount - 1], dtype: .int32),
            .init(role: .assistantFrontier, shape: [1, 1, geometry.width], dtype: dtype),
        ]
    }

    public func encodePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint) -> [MLXArray]? {
        guard let checkpoint = checkpoint as? PrefixCheckpoint,
            checkpoint.owner == ObjectIdentifier(self)
        else { return nil }
        return checkpoint.evaluationTargets
    }

    public func decodePrefixCheckpoint(tensors: [MLXArray], prefixTokens: [Int])
        -> (any CBv2MTPPrefixCheckpoint)?
    {
        guard let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: prefixTokens.count),
            tensors.count == descriptors.count,
            zip(tensors, descriptors).allSatisfy({ pair in
                pair.0.shape == pair.1.shape && pair.0.dtype == pair.1.dtype.mlxDType
            }),
            prefixTokens.allSatisfy({ $0 >= 0 && $0 < prefixCheckpointGeometry.vocabulary }),
            tokensMatch(tensors[1], prefixTokens: prefixTokens)
        else { return nil }
        // Disk identity and tensor geometry were verified by the loaded engine.
        // Rebind immutable trusted history to this assistant instance; mutable
        // draft caches are still created afresh by restorePrefixCheckpoint.
        return PrefixCheckpoint(
            owner: ObjectIdentifier(self), count: prefixTokens.count,
            hidden: tensors[0], tokens: tensors[1], frontier: tensors[2])
    }

    /// Imported tokens are already evaluated contiguous int32 storage. Compare
    /// in place while the array is retained; no M-sized Swift copies or casts
    /// may be constructed after the staging scratch owner has retired.
    private func tokensMatch(_ tokens: MLXArray, prefixTokens: [Int]) -> Bool {
        guard let info = try? tokens.evaluatedBufferInfo(), info.isRowContiguous,
              info.dataElements == tokens.size,
              let pointer = mlx_array_data_int32(tokens.ctx)
        else { return false }
        return withExtendedLifetime(tokens) {
            for (index, expected) in prefixTokens.dropFirst().enumerated() {
                if Int(pointer[index]) != expected { return false }
            }
            return true
        }
    }
}
