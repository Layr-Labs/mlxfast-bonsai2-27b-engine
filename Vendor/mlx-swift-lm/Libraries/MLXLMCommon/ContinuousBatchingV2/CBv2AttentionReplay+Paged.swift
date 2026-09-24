import Foundation
import MLX

extension CBv2AttentionReplay {
    static func runPaged(_ input: Input, arm: Arm, geometry g: Geometry) throws -> Result {
        let kind = CBv2LayerKind(attention: .full, headDim: g.headDim,
            kvHeads: g.kvHeads, queryHeads: g.queryHeads)
        let config = PagedKVPoolConfig(pageSize: g.pageSize, capacityBytes: g.poolBudgetBytes,
            dtype: input.storedKeys.dtype, maxPrefillChunk: 128,
            nominalMaxSequenceLength: g.length, maxBufferLength: g.poolBudgetBytes,
            segmentSizeBytes: arm == .pagedSegmented ? g.segmentTargetBytes : nil)
        let backend = try PagedKVBackend(layerKinds: [kind], config: config)
        let states = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: g.length)
        defer { backend.release(states) }
        guard let row = states[0] as? PagedSequenceKV else { throw Failure.invalidInput }
        let cache = backend.makeLayerCaches()[0]
        cache.setRows([row])
        let group = backend.pool.group(row.groupKey)
        let k = input.storedKeys.array().squeezed(axis: 0)
        let v = input.storedValues.array().squeezed(axis: 0)
        // Deliberately omit the last token: updateAndAttend must write it itself.
        for start in stride(from: 0, to: g.length - 1, by: config.maxPrefillChunk) {
            let end = min(g.length - 1, start + config.maxPrefillChunk)
            row.write(keys: k[0..., start..<end, 0...], values: v[0..., start..<end, 0...])
            try backend.pool.writeValidation.check()
            eval(group.writeFence)
        }
        guard row.absoluteOffset == g.length - 1 else { throw Failure.storageMismatch }

        // Reuse the existing branch observation; this is never sample-confirmed
        // and is not exported as a model-forward metadata snapshot.
        let observer = CBv2AttentionMetadataState(
            try CBv2AttentionMetadataConfig(requestID: 1, outputIndex: 1, maximumRecords: 1),
            expectedOwners: [0])
        guard let forward = observer.select(requestID: 1, outputIndex: 1, batchSize: 1, phase: "decode"),
            forward.bindOwner(cache: cache, storageLayerIndex: 0, kind: kind)
        else { throw Failure.dispatchMismatch }
        cache.attentionMetadata = forward
        let output = cache.updateAndAttend(queries: input.queries.array(),
            keys: input.incomingKeys.array(), values: input.incomingValues.array(),
            scale: Float(bitPattern: input.scaleBits), sinks: nil)
        cache.attentionMetadata = nil
        forward.finish(succeeded: true)
        _ = observer.takePendingForward()  // Break the state/forward ownership cycle without confirming a sample.
        try backend.pool.writeValidation.check()
        eval(output)
        let metadata = observer.takeSnapshot()
        let expected = arm == .pagedFixed ? "paged_fixed_decode" : "paged_segmented_decode"
        guard metadata.refusals.isEmpty, metadata.records.count == 1,
            let record = metadata.records.first, record.dispatch == expected,
            record.offsetBefore == g.length - 1, record.offsetAfter == g.length,
            record.scaleBits == input.scaleBits,
            record.queries.dtype == String(describing: input.queries.dtype),
            metadata.sampleOutcome == "graph_constructed_unconfirmed"
        else { throw Failure.dispatchMismatch }
        guard arm != .pagedSegmented || g.length <= 4096 || group.segments.count > 1
        else { throw Failure.dispatchMismatch }
        let snapshot = row.snapshot()
        eval(snapshot.keys, snapshot.values)
        let readK = Tensor(snapshot.keys), readV = Tensor(snapshot.values)
        guard snapshot.offset == g.length, readK.shape == input.storedKeys.shape,
            readV.shape == input.storedValues.shape, readK.dtype == input.storedKeys.dtype,
            readV.dtype == input.storedValues.dtype
        else { throw Failure.storageMismatch }
        // Keep complete readback bytes even on an identity mismatch; the driver
        // reports the mismatch as inconclusive and retains evidence to inspect it.
        let gqa = g.queryHeads / g.kvHeads
        let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: g.headDim, gqa: gqa)
        // Geometry comes from the same pure pinned sizer used by both branches;
        // dispatch itself above is observed at the actual call site.
        let partition = PagedAttentionKernel.partitionTokensForDispatch(
            maxAttendLength: g.length, batch: 1, kvHeads: g.kvHeads,
            headSplits: gqa / hpt, pageSize: g.pageSize)
        return Result(output: Tensor(output), storedKeys: readK, storedValues: readV,
            dispatch: record.dispatch, kernelOutputDType: record.kernelOutputDType,
            offset: snapshot.offset, segmentCount: group.segments.count, pageTable: row.table,
            partitionTokens: partition, geometry: g)
    }
}
