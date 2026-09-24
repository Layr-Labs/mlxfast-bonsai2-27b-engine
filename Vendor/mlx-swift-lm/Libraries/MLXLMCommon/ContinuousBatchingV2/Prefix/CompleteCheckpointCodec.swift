import Foundation
import MLX

/// Immutable loaded-model validation; safe to use from the provider's staging
/// queue. It never executes a model or changes assistant request state.
final class CBv2CompleteCheckpointCodec: @unchecked Sendable {
    let identity: CBv2CompleteCheckpointIdentity
    let layerKinds: [CBv2LayerKind]
    let recurrentSpec: CBv2RecurrentStateSpec?
    let kvDTypes: [DType]
    let assistant: (any CBv2MTPPrefixCheckpointCoding)?
    let admission: AdmissionV2
    /// Copied immutable layout only. Planning/staging never reads live pool maps.
    let pagedConfig: PagedKVPoolConfig?
    let historicalLayout: CBv2HistoricalAttentionLayout?
    var targetTensorCount: Int { (historicalLayout?.owningIndices.count ?? layerKinds.count) * 2 }
    var backendLayout: String {
        if recurrentSpec == nil { return CBv2CompleteCheckpointManifest.historicalAttentionLayout }
        return pagedConfig == nil ? CBv2CompleteCheckpointManifest.layout : CBv2CompleteCheckpointManifest.pagedLayout
    }
    var exportScratchBytes: Int {
        // Paged export copies shared native backing straight into the
        // provider-owned bounded Data; it creates no packing arrays.
        pagedConfig == nil ? CBv2CompleteCheckpointManifest.maximumSegmentBytes : 0
    }

    init(
        identity: CBv2CompleteCheckpointIdentity, layerKinds: [CBv2LayerKind],
        recurrentSpec: CBv2RecurrentStateSpec?, kvDTypes: [DType],
        assistant: (any CBv2MTPPrefixCheckpointCoding)?, admission: AdmissionV2,
        pagedConfig: PagedKVPoolConfig? = nil
    ) {
        self.identity = identity
        self.layerKinds = layerKinds
        self.recurrentSpec = recurrentSpec
        self.kvDTypes = kvDTypes
        self.assistant = assistant
        self.admission = admission
        self.pagedConfig = pagedConfig
        self.historicalLayout = recurrentSpec == nil && pagedConfig != nil && assistant == nil
            ? try? .init(layerKinds: layerKinds, dtypes: kvDTypes) : nil
    }

    func tensorDescriptors(position: Int) throws -> [CBv2CheckpointTensorDescriptor] {
        if let historicalLayout {
            guard identity.isValid, position > 1 else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            return try historicalLayout.tensorDescriptors(position: position)
        }
        guard kvDTypes.count == layerKinds.count, recurrentSpec?.layers.isEmpty == false,
            identity.isValid, position > 1
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        var result: [CBv2CheckpointTensorDescriptor] = []
        for (index, kind) in layerKinds.enumerated() {
            guard case .full = kind.attention, kind.sharesKVWithLayer == nil,
                kind.kvHeads > 0, kind.headDim > 0,
                let kvDType = CBv2CheckpointDType(kvDTypes[index]), kvDType != .int32
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            for role in [CBv2CheckpointTensorRole.keys, .values] {
                result.append(try .init(
                    role: role, layer: kind.modelLayerIndex ?? index,
                    shape: [1, kind.kvHeads, position, kind.headDim], dtype: kvDType))
            }
        }
        for spec in recurrentSpec?.layers ?? [] {
            guard let conv = CBv2CheckpointDType(spec.convDType),
                let ssm = CBv2CheckpointDType(spec.ssmDType)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            result.append(try .init(
                role: .convolution, layer: spec.modelLayerIndex, shape: spec.convShape, dtype: conv))
            result.append(try .init(
                role: .recurrent, layer: spec.modelLayerIndex, shape: spec.ssmShape, dtype: ssm))
        }
        if let assistant {
            guard let descriptors = assistant.prefixCheckpointTensorDescriptors(targetInputCount: position)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            result.append(contentsOf: descriptors)
        }
        return result
    }

    func plan(
        manifest: CBv2CompleteCheckpointManifest, request: CBv2Request,
        minimumChunkSize: Int, maximumChunkSize: Int
    ) throws -> CBv2CompleteCheckpointImportPlan {
        _ = try manifest.validateStructure()
        let (maximumLength, overflow) = request.promptTokens.count.addingReportingOverflow(max(1, request.maxTokens))
        guard !overflow, request.prefixCacheEnabled, request.multimodal == nil,
            request.positionState == nil, manifest.identity == identity,
            manifest.backendLayout == backendLayout,
            manifest.cacheSalt == request.cacheSalt,
            manifest.position < request.promptTokens.count,
            manifest.prefixTokens.elementsEqual(request.promptTokens.prefix(manifest.position)),
            manifest.chunkSize >= minimumChunkSize, manifest.chunkSize <= maximumChunkSize,
            CBv2AttentionV1.queryBlockSize <= 0 || manifest.chunkSize % CBv2AttentionV1.queryBlockSize == 0,
            manifest.assistantCodecID == assistant?.prefixCheckpointCodecID,
            manifest.attentionLayers == historicalLayout?.layers,
            manifest.tensors == (try tensorDescriptors(position: manifest.position))
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        return try .init(codec: self, manifest: manifest, maximumSequenceLength: maximumLength)
    }

    /// Build views only on the engine queue while retired source KV remains
    /// reserved. The export queue waits for those already-scheduled roots.
    func export(
        checkpoint: CBv2RecurrentCheckpoint,
        kv: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        tokens: [Int], cacheSalt: String?
    ) throws -> CBv2CompleteCheckpointExport {
        guard pagedConfig == nil, checkpoint.position < tokens.count, kv.count == layerKinds.count else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let metadataPermit = try CBv2CheckpointManifestMemory.Permit(admission: admission, position: checkpoint.position)
        return try withExtendedLifetime(metadataPermit) {
            try makeContiguousExport(checkpoint: checkpoint, kv: kv, tokens: tokens,
                                     cacheSalt: cacheSalt, metadataPermit: metadataPermit)
        }
    }

    private func makeContiguousExport(
        checkpoint: CBv2RecurrentCheckpoint, kv: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        tokens: [Int], cacheSalt: String?, metadataPermit: CBv2CheckpointManifestMemory.Permit
    ) throws -> CBv2CompleteCheckpointExport {
        let descriptors = try tensorDescriptors(position: checkpoint.position)
        var arrays: [MLXArray] = []
        for entry in kv {
            guard let entry, entry.offset >= checkpoint.position else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            arrays.append(entry.keys[.ellipsis, ..<checkpoint.position, 0...])
            arrays.append(entry.values[.ellipsis, ..<checkpoint.position, 0...])
        }
        for spec in recurrentSpec?.layers ?? [] {
            guard let state = checkpoint.layers[spec.modelLayerIndex],
                let conv = state.conv, let ssm = state.ssm
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            arrays.append(contentsOf: [conv, ssm])
        }
        if let assistant {
            guard let state = checkpoint.assistant,
                let encoded = assistant.encodePrefixCheckpoint(state)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            arrays.append(contentsOf: encoded)
        } else if checkpoint.assistant != nil {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        guard arrays.count == descriptors.count,
            zip(arrays, descriptors).allSatisfy({ pair in
                pair.0.shape == pair.1.shape && pair.0.dtype == pair.1.dtype.mlxDType
            })
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let manifest = CBv2CompleteCheckpointManifest(
            schemaVersion: CBv2CompleteCheckpointManifest.currentSchemaVersion, identity: identity,
            backendLayout: backendLayout, position: checkpoint.position, chunkSize: checkpoint.chunkSize,
            cacheSalt: cacheSalt, assistantCodecID: assistant?.prefixCheckpointCodecID,
            metadata: .init(tokens: Array(tokens.prefix(checkpoint.position)), tensors: descriptors, permit: metadataPermit))
        _ = try manifest.validateStructure()
        return .init(manifest: manifest, arrays: arrays, usesProcessMemoryOwner: admission.hasProcessMemoryOwner)
    }

    func export(
        checkpoint: CBv2RecurrentCheckpoint, state: [CBv2SequenceKV?],
        tokens: [Int], cacheSalt: String?
    ) throws -> CBv2CompleteCheckpointExport {
        if pagedConfig != nil {
            return try exportPaged(checkpoint: checkpoint, state: state, tokens: tokens, cacheSalt: cacheSalt)
        }
        return try export(checkpoint: checkpoint, kv: state.map { $0?.snapshot() },
                          tokens: tokens, cacheSalt: cacheSalt)
    }

    func preparedState(
        manifest: CBv2CompleteCheckpointManifest, arrays: [MLXArray], maximumSequenceLength: Int
    ) throws -> CBv2PreparedCompleteCheckpoint {
        guard pagedConfig == nil, arrays.count == manifest.tensors.count else {
            throw CBv2CompleteCheckpointError.incompleteTransfer
        }
        var rows: [CBv2SequenceKV?] = []
        for (index, kind) in layerKinds.enumerated() {
            rows.append(try CBv2FullSequenceKV(
                restoredKeys: arrays[index * 2], restoredValues: arrays[index * 2 + 1],
                offset: manifest.position, maxLength: maximumSequenceLength,
                kvHeads: kind.kvHeads, headDim: kind.headDim))
        }
        return .init(state: rows, checkpoint: try recurrentCheckpoint(
            manifest: manifest, auxiliary: Array(arrays.dropFirst(targetTensorCount))))
    }

    /// Decode only after the complete authenticated import owns evaluated
    /// destinations. These arrays become request state, never a resident bank.
    func recurrentCheckpoint(
        manifest: CBv2CompleteCheckpointManifest, auxiliary: [MLXArray]
    ) throws -> CBv2RecurrentCheckpoint {
        defer { withExtendedLifetime(manifest) {} }
        guard recurrentSpec != nil else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let descriptors = Array(manifest.tensors.dropFirst(targetTensorCount))
        guard manifest.tensors == (try tensorDescriptors(position: manifest.position)),
              auxiliary.count == descriptors.count,
              zip(auxiliary, descriptors).allSatisfy({ $0.0.shape == $0.1.shape && $0.0.dtype == $0.1.dtype.mlxDType })
        else { throw CBv2CompleteCheckpointError.incompleteTransfer }
        var layers: [Int: CBv2RecurrentLayerState] = [:]
        var cursor = 0, checkpointBytes = 0
        for spec in recurrentSpec?.layers ?? [] {
            layers[spec.modelLayerIndex] = .init(conv: auxiliary[cursor], ssm: auxiliary[cursor + 1])
            checkpointBytes += auxiliary[cursor].nbytes + auxiliary[cursor + 1].nbytes
            cursor += 2
        }
        var assistantCheckpoint: (any CBv2MTPPrefixCheckpoint)?
        if let assistant {
            guard let restored = assistant.decodePrefixCheckpoint(
                tensors: Array(auxiliary[cursor...]), prefixTokens: manifest.prefixTokens)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            assistantCheckpoint = restored
            checkpointBytes += restored.materializedBytes
        }
        return .init(position: manifest.position, chunkSize: manifest.chunkSize,
                     layers: layers, byteCount: checkpointBytes, assistant: assistantCheckpoint)
    }

}

final class CBv2PreparedCompleteCheckpoint {
    var state: [CBv2SequenceKV?]
    var checkpoint: CBv2RecurrentCheckpoint?
    var pagedFrame: CBv2PagedCheckpointFrame?

    init(state: [CBv2SequenceKV?], checkpoint: CBv2RecurrentCheckpoint) {
        self.state = state
        self.checkpoint = checkpoint
    }

    init(pagedFrame: CBv2PagedCheckpointFrame) {
        state = []
        self.pagedFrame = pagedFrame
    }

    func clear() {
        pagedFrame?.close()
        pagedFrame = nil
        state.removeAll()
        checkpoint = nil
    }
}
