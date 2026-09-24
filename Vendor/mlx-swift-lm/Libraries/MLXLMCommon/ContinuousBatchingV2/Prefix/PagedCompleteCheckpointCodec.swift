import MLX

extension CBv2CompleteCheckpointCodec {
    /// Page maps are captured on the engine queue while donor rows remain
    /// pinned. No whole-prefix snapshot/gather reaches the retirement queue.
    func exportPaged(
        checkpoint: CBv2RecurrentCheckpoint, state: [CBv2SequenceKV?],
        tokens: [Int], cacheSalt: String?
    ) throws -> CBv2CompleteCheckpointExport {
        guard checkpoint.position < tokens.count, state.count == layerKinds.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let metadataPermit = try CBv2CheckpointManifestMemory.Permit(admission: admission, position: checkpoint.position)
        return try withExtendedLifetime(metadataPermit) {
            try makePagedExport(checkpoint: checkpoint, state: state, tokens: tokens,
                                cacheSalt: cacheSalt, metadataPermit: metadataPermit)
        }
    }

    private func makePagedExport(
        checkpoint: CBv2RecurrentCheckpoint, state: [CBv2SequenceKV?], tokens: [Int], cacheSalt: String?,
        metadataPermit: CBv2CheckpointManifestMemory.Permit
    ) throws -> CBv2CompleteCheckpointExport {
        let descriptors = try tensorDescriptors(position: checkpoint.position)
        var sources: [CBv2CompleteCheckpointTensorSource] = []
        for (index, entry) in state.enumerated() {
            guard let row = entry as? PagedSequenceKV,
                  row.groupKey.dtype == kvDTypes[index], row.pool.layerKinds == layerKinds,
                  row.groupKey == row.pool.groupKey(forLayer: index)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            let pageMap = try CBv2PagedCheckpointPageMap(row: row, position: checkpoint.position, admission: admission)
            sources.append(.paged(try .init(pageMap: pageMap, values: false)))
            sources.append(.paged(try .init(pageMap: pageMap, values: true)))
        }
        for spec in recurrentSpec?.layers ?? [] {
            guard let layer = checkpoint.layers[spec.modelLayerIndex], let conv = layer.conv, let ssm = layer.ssm
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            sources.append(contentsOf: [.array(conv), .array(ssm)])
        }
        if let assistant {
            guard let state = checkpoint.assistant, let encoded = assistant.encodePrefixCheckpoint(state)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            sources.append(contentsOf: encoded.map { .array($0) })
        } else if checkpoint.assistant != nil {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        guard sources.count == descriptors.count,
              zip(sources, descriptors).allSatisfy({ $0.0.matches($0.1) })
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let manifest = CBv2CompleteCheckpointManifest(
            schemaVersion: CBv2CompleteCheckpointManifest.currentSchemaVersion, identity: identity,
            backendLayout: backendLayout, position: checkpoint.position, chunkSize: checkpoint.chunkSize,
            cacheSalt: cacheSalt, assistantCodecID: assistant?.prefixCheckpointCodecID,
            metadata: .init(tokens: Array(tokens.prefix(checkpoint.position)), tensors: descriptors, permit: metadataPermit))
        _ = try manifest.validateStructure()
        return .init(manifest: manifest, sources: sources, usesProcessMemoryOwner: admission.hasProcessMemoryOwner)
    }
}
