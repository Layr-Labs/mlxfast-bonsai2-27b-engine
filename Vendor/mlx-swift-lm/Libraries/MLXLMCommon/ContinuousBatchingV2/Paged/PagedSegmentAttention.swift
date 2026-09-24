import Foundation
import MLX
import MLXFast

/// Decode over any number of native segments using bounded binding buckets.
/// Only address resolution differs from the direct-slab numerical reference.
enum PagedSegmentAttention {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    static func inputNames(bindings: Int) -> [String] {
        ["q", "knew", "vnew"] + (0 ..< bindings).map { "segment\($0)" }
            + ["seqinfo", "params", "previous", "value_offsets", "records", "partials", "meta"]
    }

    static func body(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const device int32_t* record = records + threadgroup_position_in_grid.y * STRIDE;
            const cbv2::PagedSegmentAccessor<T, SEGMENTS> cache{
                {\(pointers)}, value_offsets, record, (size_t)KVH * S * D};
            const uint3 position(threadgroup_position_in_grid.x, record[0], record[1]);
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            threadgroup float q_smem[HPT * D];
            threadgroup float red_smem[NSG * HPT * (D + 2)];
            cbv2::paged_attention_part_cached_impl<T, D, S, GQA, HPT, NSG, PTOK, HAS_WRITE, HAS_SOFTCAP>(
                q, knew, vnew, cache, seqinfo, seqinfo, params, KVH, 0, record[6],
                q_smem, red_smem, partials, meta,
                position, thread_position_in_threadgroup, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """
    }

    private static func kernel(key: PagedAttentionKernelKey, bindings: Int, kvHeads: Int,
                               source: String) -> MLXFast.MLXFastKernel {
        let name = key.kernelName + "_segments\(bindings)_kvh\(kvHeads)"
        return lock.withLock {
            if let existing = kernels[name] { return existing }
            let made = MLXFast.metalKernel(
                name: name, inputNames: inputNames(bindings: bindings), outputNames: ["fence"],
                source: body(bindings: bindings), header: source, ensureRowContiguous: true,
                mutableInputs: ["partials", "meta"]
                    + (key.hasWrite ? (0..<bindings).map { "segment\($0)" } : []))
            kernels[name] = made
            return made
        }
    }

    private static func mergeKernel(key: PagedAttentionKernelKey, source: String)
        -> MLXFast.MLXFastKernel
    {
        let name = key.kernelName + "_segments_ordered"
        return lock.withLock {
            if let existing = kernels[name] { return existing }
            // previous is a binding-only dependency. CustomKernel registers
            // every input with the Metal encoder, which inserts a full buffer
            // barrier when this prior kernel output is consumed. An alias-only
            // Depends node cannot provide that barrier for hidden writes.
            let made = MLXFast.metalKernel(
                name: name, inputNames: ["partials", "meta", "seqinfo", "sinks", "previous"],
                outputNames: ["out"], source: PagedAttentionMSL.mergeBody,
                header: source, ensureRowContiguous: true)
            kernels[name] = made
            return made
        }
    }

    static func decode(
        queries: MLXArray, newKeys: MLXArray?, newValues: MLXArray?,
        group: PagedKVGroup, rows: [PagedSegmentDispatchPlan.Row],
        sinks: MLXArray?, params: MLXArray, softcap: Bool, source: String,
        dispatchCache: PagedSegmentDispatchCache? = nil
    ) -> MLXArray {
        var q = queries.ndim == 4 ? queries.squeezed(axis: 2) : queries
        guard !group.writeValidation.isFaulted else { return q }
        if let newKeys, let newValues {
            guard group.writeValidation.validate(keys: newKeys, values: newValues, expected: group.dtype)
            else { return q }
        }
        precondition(q.ndim == 3 && q.dim(0) == rows.count)
        precondition((newKeys == nil) == (newValues == nil))
        let dtype = group.dtype
        let b = q.dim(0), qh = q.dim(1), d = q.dim(2), kvh = group.key.kvHeads
        precondition(d == group.key.headDim && qh % kvh == 0)
        let gqa = qh / kvh
        let nsg = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: d, gqa: gqa)!
        let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: d, gqa: gqa)
        let splits = gqa / hpt
        let (seqinfo, maxLength) = PagedAttentionKernel.seqinfo(rows.map(\.info))
        let ptok = PagedSegmentDispatchPlan.boundedPartitionTokens(
            PagedAttentionKernel.partitionTokensForDispatch(
                maxAttendLength: maxLength, batch: b, kvHeads: kvh,
                headSplits: splits, pageSize: group.pageSize), pageSize: group.pageSize)
        let prepared = dispatchCache?.prepare(
            rows: rows, group: group, partitionTokens: ptok, hasWrite: newKeys != nil)
            ?? PagedSegmentPreparedDispatch(
                rows: rows, group: group, partitionTokens: ptok, hasWrite: newKeys != nil)
        let plan = prepared.plan
        if q.dtype != dtype { q = q.asType(dtype) }
        let k = newKeys ?? q
        let v = newValues ?? q
        if newKeys != nil {
            precondition(k.shape == [b, kvh, d] && v.shape == k.shape)
        }
        let partials = MLXArray.zeros([b, qh, plan.maxPartitions, d], dtype: .float32)
        // Padding preserves a device pointer even for a one-head/one-partition probe.
        let meta = MLXArray.zeros([max(8, b * qh * plan.maxPartitions * 2)], dtype: .float32)
        let partKey = PagedAttentionKernelKey(
            pass: .part, dtype: dtype, headDim: d, pageSize: group.pageSize, gqa: gqa,
            simdgroups: nsg, hasSinks: false, hasSoftcap: softcap,
            partitionTokens: ptok, hasWrite: newKeys != nil)
        for (bucket, metadata) in zip(plan.buckets, prepared.metadata) {
            let segments = bucket.segmentIDs.map { group.segments[$0]! }
            let bindingCount = bucket.bindingClass
            var backing = segments.map(\.storage)
            // Empty records never select padded bindings. They alias an input,
            // never a separately declared writable output.
            backing.append(contentsOf: repeatElement(backing[0], count: bindingCount - backing.count))
            let inputs = [q, k, v] + backing + [
                seqinfo, params, group.writeFence, metadata.valueOffsets,
                metadata.records, partials, meta]
            group.writeFence = kernel(
                key: partKey, bindings: bindingCount, kvHeads: kvh, source: source)(
                inputs,
                template: [("T", dtype), ("D", d), ("S", group.pageSize),
                           ("KVH", kvh), ("GQA", gqa), ("HPT", hpt), ("NSG", nsg),
                           ("PTOK", ptok), ("HAS_WRITE", newKeys != nil),
                           ("HAS_SOFTCAP", softcap), ("SEGMENTS", bindingCount),
                           ("STRIDE", PagedSegmentDispatchPlan.recordStride)],
                grid: (kvh * splits * 32 * nsg, bucket.workCount, 1),
                threadGroup: (32 * nsg, 1, 1), outputShapes: [[1]], outputDTypes: [.int32])[0]
        }
        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: dtype, headDim: d, pageSize: group.pageSize, gqa: gqa,
            simdgroups: 1, hasSinks: sinks != nil, hasSoftcap: false, partitionTokens: ptok)
        let zeroSinks = MLXArray.zeros([max(8, qh)], dtype: .float32)
        return mergeKernel(key: mergeKey, source: source)(
            [partials, meta, seqinfo, sinks ?? zeroSinks, group.writeFence],
            template: [("T", dtype), ("D", d), ("PTOK", ptok), ("HAS_SINKS", sinks != nil)],
            grid: (qh * 32, b, 1), threadGroup: (32, 1, 1),
            outputShapes: [[b, qh, d]], outputDTypes: [dtype])[0]
    }
}

/// Experimental dispatch batching, not a new attention reduction. A single
/// global-cache sequence becomes one independent native decode row per query.
/// Fixed PTOK is required: adaptive batch sizing would change arithmetic.
enum PagedMTPBatchedColumns {
    static let enabled = ProcessInfo.processInfo.environment[
        "DARKBLOOM_NEMOTRON35_MTP_BATCHED_ATTENTION"] != "0"

    static func eligible(queries: MLXArray, row: PagedSequenceKV, group: PagedKVGroup) -> Bool {
        queries.dim(0) == 1 && (2...8).contains(queries.dim(2))
            && row.windowSize == nil && group.segmentLayout != nil
            && row.absoluteOffset >= row.frozenHighWater
            && PagedAttentionKernel.partitionTargetThreadgroups == 0
    }

    static func attend(queries: MLXArray, keys: MLXArray, values: MLXArray,
                       row: PagedSequenceKV, group: PagedKVGroup,
                       sinks: MLXArray?, params: MLXArray, softcap: Bool,
                       source: String, dispatchCache: PagedSegmentDispatchCache) -> MLXArray {
        precondition(eligible(queries: queries, row: row, group: group))
        let count = queries.dim(2), heads = queries.dim(1), width = queries.dim(3)
        // One ordinary chunk write, followed by independent causal query rows.
        row.write(keys: keys.squeezed(axis: 0), values: values.squeezed(axis: 0))
        let descriptors = (0..<count).map { position in
            let end = row.absoluteOffset - count + position + 1
            return PagedSegmentDispatchPlan.Row(pages: row.table,
                info: row.seqInfoRow(attending: (start: row.baseOffset,
                    length: end - row.baseOffset)),
                identity: .init(serial: row.serial, tableVersion: row.tableVersion))
        }
        let batch = queries.transposed(0, 2, 1, 3).reshaped(count, heads, width)
        let result = PagedSegmentAttention.decode(queries: batch, newKeys: nil, newValues: nil,
            group: group, rows: descriptors, sinks: sinks, params: params, softcap: softcap,
            source: source, dispatchCache: dispatchCache)
        let output = result.reshaped(1, count, heads, width).transposed(0, 2, 1, 3)
        return output.dtype == queries.dtype ? output : output.asType(queries.dtype)
    }
}
