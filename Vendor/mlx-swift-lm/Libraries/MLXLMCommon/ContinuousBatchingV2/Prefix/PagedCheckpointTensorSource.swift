import Cmlx
import Foundation
import MLX

/// Immutable page-map capture, built on the engine queue while the donor row
/// remains charged and pinned. Export never reads mutable pool dictionaries on
/// the I/O queue. Full-attention prefix pages remain immutable during donation;
/// a rolling window requires a different boundary capture and is refused here.
final class CBv2PagedCheckpointTensorSource {
    private let key: PagedKVGroupKey
    private let pageSize: Int
    private let position: Int
    private let values: Bool
    private var pageMap: CBv2PagedCheckpointPageMap?
    let byteCount: Int

    convenience init(row: PagedSequenceKV, position: Int, values: Bool, admission: AdmissionV2) throws {
        try self.init(pageMap: .init(row: row, position: position, admission: admission), values: values)
    }

    convenience init(storage: CBv2PagedCheckpointStorage, layerIndex: Int, values: Bool,
                     admission: AdmissionV2) throws {
        guard storage.plan.layers.indices.contains(layerIndex) else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        let layer = storage.plan.layers[layerIndex]
        guard layer.ringPages == nil, layer.tokenStart == 0 else {
            throw CBv2CompleteCheckpointError.incompatibleCheckpoint
        }
        guard let group = storage.groups[layer.key] else { throw CBv2CompleteCheckpointError.closed }
        try self.init(pageMap: .init(
            key: layer.key, pageSize: storage.plan.pageSize, position: storage.plan.position,
            table: group.pages[layer.firstPage ..< layer.firstPage + layer.pageCount],
            layout: group.layout, segments: group.segments, previous: MLXArray.zeros([1], dtype: .int32),
            admission: admission), values: values)
    }

    init(pageMap: CBv2PagedCheckpointPageMap, values: Bool) throws {
        self.key = pageMap.key
        self.pageSize = pageMap.pageSize
        self.position = pageMap.position
        self.values = values
        self.pageMap = pageMap
        self.byteCount = try CBv2CheckpointTensorDescriptor.checkedByteCount(
            shape: [1, key.kvHeads, position, key.headDim], dtype: key.dtype)
    }

    func matches(_ descriptor: CBv2CheckpointTensorDescriptor) -> Bool {
        descriptor.byteCount == byteCount && descriptor.dtype.mlxDType == key.dtype
            && descriptor.shape == [1, key.kvHeads, position, key.headDim]
            && descriptor.role == (values ? .values : .keys)
    }

    /// The enclosing export serializes reads and close. The supported Metal
    /// allocator exposes shared CPU-readable storage, so only the returned
    /// provider-owned Data is allocated. No full backing view or GPU gather is
    /// constructed; the page map pins every source throughout the copy.
    func readSegment(byteOffset: Int, maximumBytes: Int) throws -> Data {
        guard let pageMap else { throw CBv2CompleteCheckpointError.closed }
        let width = key.dtype.size
        guard byteOffset >= 0, byteOffset < byteCount, byteOffset % width == 0,
            maximumBytes > 0, maximumBytes <= CBv2CompleteCheckpointManifest.maximumSegmentBytes
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        let count = min(byteCount - byteOffset, maximumBytes - maximumBytes % width)
        guard count > 0 else { throw CBv2CompleteCheckpointError.invalidSegment }
        try pageMap.prepareForReading()
        var result = Data(count: count)
        try result.withUnsafeMutableBytes { destination in
            try CBv2PagedCheckpointByteLayout.runs(
                headDim: key.headDim, position: position, pageSize: pageSize,
                itemSize: width, byteOffset: byteOffset, count: count
            ) { logicalPage, head, slot, feature, packedOffset, length in
                let page = pageMap[logicalPage]
                let segment = page.segment
                let source = ((page.localPage * key.kvHeads + head) * pageSize + slot)
                    * key.headDim + feature + (values ? segment.valueOffset : 0)
                guard let pointer = mlx_array_data_uint8(segment.storage.ctx) else {
                    throw CBv2CompleteCheckpointError.allocationFailed
                }
                destination.baseAddress!.advanced(by: packedOffset).copyMemory(
                    from: UnsafeRawPointer(pointer).advanced(by: source * width), byteCount: length)
            }
        }
        return result
    }

    func close() { pageMap = nil }
}
