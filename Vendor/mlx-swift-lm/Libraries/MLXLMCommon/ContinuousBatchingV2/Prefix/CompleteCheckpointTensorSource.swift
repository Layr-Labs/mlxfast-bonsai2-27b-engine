import Cmlx
import Foundation
import MLX

/// One logical tensor with a bounded read operation. Paged targets never become
/// contiguous whole-prefix views; recurrent/assistant arrays retain their native
/// layout and are packed only over the requested span.
enum CBv2CompleteCheckpointTensorSource {
    case array(MLXArray)
    case paged(CBv2PagedCheckpointTensorSource)
    case historicalWindow(CBv2HistoricalWindowTensorSource)

    func matches(_ descriptor: CBv2CheckpointTensorDescriptor) -> Bool {
        switch self {
        case .array(let array):
            array.shape == descriptor.shape && array.dtype == descriptor.dtype.mlxDType
        case .paged(let source): source.matches(descriptor)
        case .historicalWindow(let source): source.matches(descriptor)
        }
    }

    func readSegment(
        descriptor: CBv2CheckpointTensorDescriptor, byteOffset: Int, maximumBytes: Int
    ) throws -> Data {
        guard maximumBytes > 0, maximumBytes <= CBv2CompleteCheckpointManifest.maximumSegmentBytes else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        if case .historicalWindow(let source) = self {
            return try source.readSegment(byteOffset: byteOffset, maximumBytes: maximumBytes)
        }
        if case .paged(let source) = self {
            return try source.readSegment(byteOffset: byteOffset, maximumBytes: maximumBytes)
        }
        guard case .array(let array) = self else { throw CBv2CompleteCheckpointError.closed }
        let itemSize = descriptor.dtype.mlxDType.size
        guard byteOffset >= 0, byteOffset < descriptor.byteCount, byteOffset % itemSize == 0 else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let count = min(maximumBytes - maximumBytes % itemSize, descriptor.byteCount - byteOffset)
        guard count > 0 else { throw CBv2CompleteCheckpointError.invalidSegment }
        try withError { eval(array) }
        guard let pointer = mlx_array_data_uint8(array.ctx), let nativeStrides = mlx_array_strides(array.ctx) else {
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        let strides = descriptor.shape.indices.map { Int(nativeStrides[$0]) }
        var result = Data(count: count)
        result.withUnsafeMutableBytes { destination in
            CBv2CheckpointByteLayout.copy(
                shape: descriptor.shape, strides: strides, itemSize: itemSize,
                byteOffset: byteOffset, count: count
            ) { physicalOffset, packedOffset, length in
                destination.baseAddress!.advanced(by: packedOffset).copyMemory(
                    from: UnsafeRawPointer(pointer).advanced(by: physicalOffset), byteCount: length)
            }
        }
        return result
    }

    func close() {
        if case .historicalWindow(let source) = self { source.close() }
        if case .paged(let source) = self { source.close() }
    }
}
