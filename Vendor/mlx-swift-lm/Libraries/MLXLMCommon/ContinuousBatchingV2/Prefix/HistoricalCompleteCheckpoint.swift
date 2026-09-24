import MLX

/// Request-local exact boundary. Window owners were copied at launch; full
/// pages remain with the donor until its existing retirement/donation barrier.
struct CBv2HistoricalCompleteCheckpoint {
    let position: Int
    let chunkSize: Int
    let windows: [Int: CBv2HistoricalWindow]
    var evaluationRoots: [MLXArray] { windows.values.compactMap(\.evaluationRoot) }
    func markSubmitted() { for window in windows.values { window.markSubmitted() } }
    func finishEvaluation() throws {
        for window in windows.values { try window.finishEvaluation() }
    }
}

extension CBv2CompleteCheckpointCodec {
    /// The stage transfers complete target state, including exact windows.
    /// This is a direct resume contract; attention-only replay capability is
    /// intentionally not used to infer a historical window that it never owns.
    func historicalReusePlan(position: Int, maximumSequenceLength: Int) throws -> CBv2PrefixReusePlan {
        guard let layout = historicalLayout, position > 1, maximumSequenceLength > position else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        var bytesPerToken = 0
        for (index, layer) in layout.layers.enumerated() where layer.owner == index && layer.window == nil {
            let bytes = try CBv2CheckpointTensorDescriptor.checkedByteCount(
                shape: [2, layer.kvHeads, layer.headDim], dtype: layer.dtype.mlxDType)
            let (next, overflow) = bytesPerToken.addingReportingOverflow(bytes)
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            bytesPerToken = next
        }
        let (fullBytes, overflow) = bytesPerToken.multipliedReportingOverflow(by: position)
        guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        return .init(
            backend: .pagedFP16, strategy: .direct, matchedBoundary: position,
            replayStart: position, replayTokens: 0, prefillTokensSaved: position,
            restoredFullTokens: position, capacityReservationTokens: maximumSequenceLength,
            nominalFullKVBytesPerToken: admission.fullKVBytesPerToken,
            fullKVBytesPerToken: bytesPerToken,
            additionalFullKVBytesPerToken: max(0, bytesPerToken - admission.fullKVBytesPerToken),
            initialAdditionalCapacityBytes: 0, fullCapacityTokensReserved: maximumSequenceLength,
            stagedFullKVBytes: fullBytes, residentFullKVBytes: fullBytes)
    }

    func exportHistorical(
        checkpoint: CBv2HistoricalCompleteCheckpoint, state: [CBv2SequenceKV?],
        tokens: [Int], cacheSalt: String?
    ) throws -> CBv2CompleteCheckpointExport {
        guard let layout = historicalLayout, checkpoint.position < tokens.count,
              state.count == layerKinds.count
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let permit = try CBv2CheckpointManifestMemory.Permit(admission: admission, position: checkpoint.position)
        return try withExtendedLifetime(permit) {
            try makeHistoricalExport(checkpoint: checkpoint, state: state, layout: layout,
                                     tokens: tokens, cacheSalt: cacheSalt, permit: permit)
        }
    }

    private func makeHistoricalExport(
        checkpoint: CBv2HistoricalCompleteCheckpoint, state: [CBv2SequenceKV?],
        layout: CBv2HistoricalAttentionLayout, tokens: [Int], cacheSalt: String?,
        permit: CBv2CheckpointManifestMemory.Permit
    ) throws -> CBv2CompleteCheckpointExport {
        let descriptors = try layout.tensorDescriptors(position: checkpoint.position)
        var sources: [CBv2CompleteCheckpointTensorSource] = []
        for index in layout.owningIndices {
            guard let row = state[index] as? PagedSequenceKV,
                  row.pool.layerKinds == layerKinds, row.groupKey.dtype == kvDTypes[index],
                  row.groupKey == row.pool.groupKey(forLayer: index)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            if layout.layers[index].window != nil {
                guard let window = checkpoint.windows[index], window.position == checkpoint.position,
                      window.start == layout.layers[index].tokenStart(at: checkpoint.position)
                else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                sources.append(.historicalWindow(.init(window: window, values: false)))
                sources.append(.historicalWindow(.init(window: window, values: true)))
            } else {
                let map = try CBv2PagedCheckpointPageMap(row: row, position: checkpoint.position, admission: admission)
                sources.append(.paged(try .init(pageMap: map, values: false)))
                sources.append(.paged(try .init(pageMap: map, values: true)))
            }
        }
        for index in layout.layers.indices where layout.layers[index].owner != index {
            guard state[index] == nil else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        }
        guard zip(sources, descriptors).allSatisfy({ $0.0.matches($0.1) }) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let manifest = CBv2CompleteCheckpointManifest(
            schemaVersion: CBv2CompleteCheckpointManifest.currentSchemaVersion, identity: identity,
            backendLayout: backendLayout, position: checkpoint.position, chunkSize: checkpoint.chunkSize,
            cacheSalt: cacheSalt, assistantCodecID: nil,
            metadata: .init(tokens: Array(tokens.prefix(checkpoint.position)), tensors: descriptors,
                            attentionLayers: layout.layers, permit: permit))
        _ = try manifest.validateStructure()
        return .init(manifest: manifest, sources: sources, usesProcessMemoryOwner: admission.hasProcessMemoryOwner)
    }
}
