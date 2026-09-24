import Cmlx
import MLX
import MLXLMCommon

/// Stable prefill checkpoints contain immutable shifted target history only.
/// Mutable native pages and an in-flight speculative transaction are never
/// shared across requests; restore allocates fresh request-owned pages.
extension NemotronH35MTPAssistant: CBv2MTPPrefixCheckpointCoding {
    private final class PrefixCheckpoint: CBv2MTPPrefixCheckpoint {
        let owner: ObjectIdentifier
        let targetInputCount: Int
        let hidden: MLXArray
        let tokens: MLXArray
        let frontier: MLXArray

        init(owner: ObjectIdentifier, count: Int, arrays: [MLXArray]) {
            self.owner = owner
            targetInputCount = count
            hidden = arrays[0]
            tokens = arrays[1]
            frontier = arrays[2]
        }

        var evaluationTargets: [MLXArray] { [hidden, tokens, frontier] }
        var materializedBytes: Int {
            evaluationTargets.reduce(0) { total, array in
                let (sum, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : sum
            }
        }
    }

    public var prefixCheckpointCodecID: String {
        let a = target.configuration
        let hiddenDType = target.mtpEmbedding(MLXArray([Int32(0)]).reshaped([1, 1])).dtype
        let kvDType = module.projectedKVDType(hiddenDType: hiddenDType)
        return "nemotron35-mtp-paged-trusted-history-v1:h\(a.hiddenSize):v\(a.vocabSize):\(hiddenDType):kv\(kvDType)"
    }

    public func prefixCheckpointTensorDescriptors(targetInputCount count: Int)
        -> [CBv2CheckpointTensorDescriptor]?
    {
        let a = target.configuration
        let hiddenDType = target.mtpEmbedding(MLXArray([Int32(0)]).reshaped([1, 1])).dtype
        let activation = hiddenDType
        guard count > 1, let dtype = CBv2CheckpointDType(activation), dtype.isFloatingPoint
        else { return nil }
        return try? [
            .init(role: .assistantHidden, shape: [1, count - 1, a.hiddenSize], dtype: dtype),
            .init(role: .assistantTokens, shape: [1, count - 1], dtype: .int32),
            .init(role: .assistantFrontier, shape: [1, 1, a.hiddenSize], dtype: dtype),
        ]
    }

    public func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount count: Int
    ) -> (any CBv2MTPPrefixCheckpoint)? {
        guard let state = requestState as? State,
            state.owner == ObjectIdentifier(self), !state.released,
            state.stagedInputCount == 0, !state.started, state.cacheOffset == 0,
            state.committedInputCount == count, count > 1,
            state.pendingHidden.count == state.pendingTokens.count,
            !state.pendingHidden.isEmpty, let frontier = state.frontier,
            let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: count)
        else { return nil }
        let arrays = [concatenated(state.pendingHidden, axis: 1),
                      concatenated(state.pendingTokens, axis: 1), frontier]
        guard matches(arrays, descriptors: descriptors) else { return nil }
        return PrefixCheckpoint(owner: ObjectIdentifier(self), count: count,
            arrays: arrays.map { MLX.where(MLXArray(true), $0, $0) })
    }

    public func restorePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint)
        -> (any CBv2MTPRequestState)?
    {
        guard let checkpoint = checkpoint as? PrefixCheckpoint,
            checkpoint.owner == ObjectIdentifier(self),
            let state = makeRequestState() as? State
        else { return nil }
        state.pendingHidden = [checkpoint.hidden]
        state.pendingTokens = [checkpoint.tokens]
        state.frontier = checkpoint.frontier
        state.committedInputCount = checkpoint.targetInputCount
        return state
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
        guard let descriptors = prefixCheckpointTensorDescriptors(
                targetInputCount: prefixTokens.count),
            matches(tensors, descriptors: descriptors),
            prefixTokens.allSatisfy({ $0 >= 0 && $0 < target.configuration.vocabSize }),
            tokensMatch(tensors[1], prefixTokens: prefixTokens)
        else { return nil }
        return PrefixCheckpoint(owner: ObjectIdentifier(self), count: prefixTokens.count,
            arrays: tensors)
    }

    private func matches(
        _ arrays: [MLXArray], descriptors: [CBv2CheckpointTensorDescriptor]
    ) -> Bool {
        arrays.count == descriptors.count && zip(arrays, descriptors).allSatisfy {
            $0.shape == $1.shape && $0.dtype == $1.dtype.mlxDType
        }
    }

    private func tokensMatch(_ tokens: MLXArray, prefixTokens: [Int]) -> Bool {
        guard let info = try? tokens.evaluatedBufferInfo(), info.isRowContiguous,
            info.dataElements == tokens.size, let pointer = mlx_array_data_int32(tokens.ctx)
        else { return false }
        return withExtendedLifetime(tokens) {
            for (index, expected) in prefixTokens.dropFirst().enumerated() {
                if Int(pointer[index]) != expected { return false }
            }
            return true
        }
    }
}
