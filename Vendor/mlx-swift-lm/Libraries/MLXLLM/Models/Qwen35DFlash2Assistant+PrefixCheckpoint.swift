import MLX
import MLXLMCommon

/// Prefix checkpoints for the DFlash 2 block drafter, so the engine's hybrid
/// prefix cache can serve a request whose target runs Qwen35 + DFlash 2.
///
/// The engine captures a checkpoint only at a chunk-aligned prompt boundary
/// that is not the prompt's last chunk (`EngineLoopV2+HybridPrefix`). At that
/// point the drafter has observed the committed hidden rows but has not yet
/// absorbed them: its caches are unseeded and the rows sit in `pending`
/// (context prefetch only runs on the sampling chunk). The drafter's whole
/// history is therefore (pending rows, observedRows). Restoring a fresh state
/// with those two values reproduces cold execution: later prompt chunks append
/// to `pending` under the same row limit, and the first block seeds the caches
/// from the same retained rows.
extension Qwen35DFlash2Assistant: CBv2MTPPrefixCheckpointDrafter {
    private final class PrefixCheckpoint: CBv2MTPPrefixCheckpoint {
        let owner: ObjectIdentifier
        let targetInputCount: Int
        let rows: MLXArray

        init(owner: ObjectIdentifier, targetInputCount: Int, rows: MLXArray) {
            self.owner = owner
            self.targetInputCount = targetInputCount
            self.rows = rows
        }

        var materializedBytes: Int { rows.nbytes }
        var evaluationTargets: [MLXArray] { [rows] }
    }

    public func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount: Int
    ) -> (any CBv2MTPPrefixCheckpoint)? {
        guard let state = requestState as? RequestState,
            !state.isReleased, !state.cacheSeeded, !state.contextPrefetched,
            state.roots.isEmpty, !state.pending.isEmpty,
            state.observedRows == targetInputCount
        else { return nil }
        let rows = state.pending.count == 1 ? state.pending[0] : concatenated(state.pending, axis: 1)
        guard rows.dim(1) == state.pendingRows else { return nil }
        // A byte-preserving copy, so the checkpoint does not keep the live
        // chunk arrays (or views of the whole prefill chunk) alive.
        return PrefixCheckpoint(
            owner: ObjectIdentifier(self), targetInputCount: targetInputCount,
            rows: MLX.where(MLXArray(true), rows, rows))
    }

    public func restorePrefixCheckpoint(
        _ checkpoint: any CBv2MTPPrefixCheckpoint
    ) -> (any CBv2MTPRequestState)? {
        guard let checkpoint = checkpoint as? PrefixCheckpoint,
            checkpoint.owner == ObjectIdentifier(self),
            let state = makeRequestState() as? RequestState
        else { return nil }
        // Fresh caches per request; only the immutable committed rows are shared.
        state.pending = [checkpoint.rows]
        state.pendingRows = checkpoint.rows.dim(1)
        state.observedRows = checkpoint.targetInputCount
        return state
    }
}
