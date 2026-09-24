import Foundation
import MLX
import MLXFast

/// Bounded page transfers. A dispatch binds one segment and the request's input
/// or final output, never a concatenation of the pool's native backing.
enum PagedSegmentTransfers {
    private static let writeBody = """
        const int h = int(thread_position_in_grid.y);
        const int d = int(thread_position_in_grid.x);
        const int r = int(thread_position_in_grid.z);
        const int token = records[r * 3];
        const int page = records[r * 3 + 1];
        const int slot = records[r * 3 + 2];
        const int n = keys_shape[1];
        if (page > 0) {
            const size_t source = ((size_t)h * n + token) * D + d;
            const size_t target = (((size_t)page * H + h) * S + slot) * D + d;
            device T* destination = storage;
            destination[target] = keys[source];
            destination[VBASE + target] = values[source];
        }
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """

    private static let readBody = """
        const int h = int(thread_position_in_grid.y);
        const int d = int(thread_position_in_grid.x);
        const int r = int(thread_position_in_grid.z);
        const int token = records[r * 3];
        const int page = records[r * 3 + 1];
        const int slot = records[r * 3 + 2];
        const int count = output_shape[3];
        const size_t source = (((size_t)page * H + h) * S + slot) * D + d;
        const size_t target = ((size_t)h * count + token) * D + d;
        device T* destination = output;
        destination[target] = storage[source];
        destination[(size_t)H * count * D + target] = storage[VBASE + source];
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """

    /// Depends is a buffer alias, not an encoded Metal dependency. A later
    /// Slice may copy that alias, so force one real GPU consumer of the final
    /// transfer fence first. Its read-after-write barrier covers the hidden
    /// in-place destination writes; only four bytes of new output are needed.
    private static let completionKernel = MLXFast.metalKernel(
        name: "cbv2_segment_transfer_complete", inputNames: ["previous"],
        outputNames: ["witness"], source: "witness[0] = previous[0] + 1;",
        ensureRowContiguous: true)

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    private static func kernel(reading: Bool, dtype: DType) -> MLXFast.MLXFastKernel {
        let key = "cbv2_segment_\(reading ? "read" : "write")_\(dtype)"
        return lock.withLock {
            if let existing = kernels[key] { return existing }
            let made = MLXFast.metalKernel(
                name: key,
                inputNames: reading
                    ? ["storage", "records", "output", "previous"]
                    : ["keys", "values", "storage", "records", "previous"],
                outputNames: ["fence"], source: reading ? readBody : writeBody,
                ensureRowContiguous: true,
                mutableInputs: [reading ? "output" : "storage"])
            kernels[key] = made
            return made
        }
    }

    private static func records(_ triples: [Int32]) -> MLXArray {
        precondition(triples.count % 3 == 0)
        return MLXArray(triples + Array(repeating: 0, count: max(0, 24 - triples.count)))
    }

    /// Each record is (input/output token index, local page, slot).
    private static func buckets(group: PagedKVGroup, slots: [Int32]) -> [(PagedKVSegment, [Int32])] {
        let layout = group.segmentLayout!
        var records: [Int: [Int32]] = [:]
        for (token, position) in slots.enumerated() {
            precondition(position >= 0)
            let page = position / Int32(group.pageSize)
            precondition(group.isAllocatable(page))
            let index = layout.segmentIndex(page: page)
            records[index, default: []].append(contentsOf: [
                Int32(token), Int32(layout.localPage(page)), position % Int32(group.pageSize)])
        }
        return records.keys.sorted().map { index in
            guard let segment = group.segments[index] else {
                preconditionFailure("transfer names an uncommitted segment")
            }
            return (segment, records[index]!)
        }
    }

    static func write(group: PagedKVGroup, slots: [Int32], keys: MLXArray, values: MLXArray) {
        guard group.writeValidation.validate(keys: keys, values: values, expected: group.dtype) else { return }
        precondition(keys.shape == [group.key.kvHeads, slots.count, group.key.headDim])
        precondition(values.shape == keys.shape)
        let k = keys
        let v = values
        for (segment, triples) in buckets(group: group, slots: slots) {
            group.writeFence = kernel(reading: false, dtype: group.dtype)(
                [k, v, segment.storage, records(triples), group.writeFence],
                template: [("T", group.dtype), ("H", group.key.kvHeads),
                           ("D", group.key.headDim), ("S", group.pageSize),
                           ("VBASE", segment.valueOffset)],
                grid: (group.key.headDim, group.key.kvHeads, triples.count / 3),
                threadGroup: (min(256, group.key.headDim), 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
        }
    }

    static func gather(group: PagedKVGroup, pages: [Int32], firstSlot: Int, count: Int)
        -> (keys: MLXArray, values: MLXArray)
    {
        let ordered = gatherCombined(group: group, pages: pages, firstSlot: firstSlot, count: count)
        return (ordered[0], ordered[1])
    }

    /// Historical owners retain this single output, avoiding two potentially
    /// materializing Slice views. The caller reserves every output/record/fence
    /// before calling and keeps its charge until the captured owner retires.
    /// `publishReadFence: false` requires the owning engine step to block all
    /// successor writes until the private read has completed or drained.
    static func gatherCombined(group: PagedKVGroup, pages: [Int32], firstSlot: Int, count: Int,
                               publishReadFence: Bool = true, stream: StreamOrDevice = .default) -> MLXArray {
        let h = group.key.kvHeads, d = group.key.headDim
        guard count > 0 else {
            return MLXArray.zeros([2, 1, h, 0, d], dtype: group.dtype, stream: stream)
        }
        precondition(firstSlot >= 0 && firstSlot < group.pageSize)
        precondition(pages.count * group.pageSize >= firstSlot + count)
        let slots = (0 ..< count).map { token -> Int32 in
            let offset = firstSlot + token
            return pages[offset / group.pageSize] * Int32(group.pageSize)
                + Int32(offset % group.pageSize)
        }
        let output = MLXArray.zeros([2, 1, h, count, d], dtype: group.dtype, stream: stream)
        var fence = group.writeFence
        for (segment, triples) in buckets(group: group, slots: slots) {
            fence = kernel(reading: true, dtype: group.dtype)(
                [segment.storage, records(triples), output, fence],
                template: [("T", group.dtype), ("H", h), ("D", d),
                           ("S", group.pageSize), ("VBASE", segment.valueOffset)],
                grid: (d, h, triples.count / 3), threadGroup: (min(256, d), 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32], stream: stream)[0]
        }
        fence = completionKernel(
            [fence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32], stream: stream)[0]
        // Keep the final destination; no second whole-gather allocation.
        if publishReadFence { group.writeFence = fence }
        // MLX Depends inherits its input primitive stream (ops.cpp); host
        // Int32 record arrays have no primitive or stream to drain. Source
        // storage/fence dependencies cross streams through MLX events.
        return depends(input: output, dependencies: [fence])
    }
}
