import MLX

// Engine-queue only: stage transfer and restoration must finish before any
// request reservation can detach or any target/assistant forward can run.
extension EngineLoopV2 {
    func adoptPagedCompleteCheckpoint(
        _ staged: CBv2StagedCompleteCheckpoint, requestID: CBv2RequestID
    ) throws -> [CBv2SequenceKV?] {
        guard let paged = backend as? PagedKVBackend,
              staged.codec === completeCheckpointCapture?.codec,
              recurrentStates[requestID] == nil
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        return try staged.consumePreparedState { prepared in
            guard let frame = prepared.pagedFrame else { throw CBv2CompleteCheckpointError.incompleteTransfer }
            prepared.pagedFrame = nil
            let adopted = try paged.pool.importCheckpoint(
                frame, admission: staged.codec.admission, requestID: requestID,
                layerKinds: layerKinds, maximumTokens: staged.maximumSequenceLength)
            // importCheckpoint has atomically replaced the stage destination
            // charge with physical backing + the full N request promise. No
            // generic capacity.reserve may precede or follow this handoff.
            return try adopted.moveToActiveRequest { auxiliary in
                do {
                    if staged.codec.historicalLayout != nil {
                        guard auxiliary.isEmpty, !(mtp?.tracksPersistentHistory ?? false) else {
                            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                        }
                        // Gemma's stateless drafter seeds from the next target
                        // round. No persistent assistant history is fabricated.
                        recurrentCheckpointGeometry[requestID] = .init(
                            position: staged.manifest.position, chunkSize: staged.manifest.chunkSize)
                        return
                    }
                    let checkpoint = try staged.codec.recurrentCheckpoint(
                        manifest: staged.manifest, auxiliary: auxiliary)
                    try adoptRecurrentCheckpoint(checkpoint, requestID: requestID)
                } catch {
                    // Restoration builds only candidate state; none has run a
                    // model forward. Remove every candidate alias before the
                    // move owner releases pages and its generation-bound charge.
                    mtp?.invalidateCarry(requestID)
                    if let recurrent = recurrentStates.removeValue(forKey: requestID) {
                        try recurrent.release()
                    }
                    recurrentCheckpointGeometry.removeValue(forKey: requestID)
                    throw error
                }
            }
        }
    }
}
