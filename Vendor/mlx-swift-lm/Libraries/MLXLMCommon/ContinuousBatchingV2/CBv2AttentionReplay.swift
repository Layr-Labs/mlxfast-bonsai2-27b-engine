import Foundation
import MLX
import MLXFast

/// Standalone diagnostic inputs and outputs. No engine, model or key hierarchy.
@_spi(Benchmarking)
public enum CBv2AttentionReplay {
    public enum Arm: String, Codable, CaseIterable, Hashable, Sendable {
        case nativeSDPA, pagedFixed, pagedSegmented
    }

    public struct Tensor: Sendable {
        public let bytes: Data
        public let shape: [Int]
        public let dtype: DType

        public init(bytes: Data, shape: [Int], dtype: DType) {
            self.bytes = bytes; self.shape = shape; self.dtype = dtype
        }

        init(_ array: MLXArray) {
            bytes = array.asData(access: .copy).data
            shape = array.shape; dtype = array.dtype
        }

        func array() -> MLXArray { MLXArray(bytes, shape, dtype: dtype) }
    }

    public struct Input: Sendable {
        public let queries, storedKeys, storedValues, incomingKeys, incomingValues: Tensor
        public let scaleBits: UInt32

        public init(queries: Tensor, storedKeys: Tensor, storedValues: Tensor,
                    incomingKeys: Tensor, incomingValues: Tensor, scaleBits: UInt32) {
            self.queries = queries; self.storedKeys = storedKeys; self.storedValues = storedValues
            self.incomingKeys = incomingKeys; self.incomingValues = incomingValues
            self.scaleBits = scaleBits
        }
    }

    public struct Geometry: Sendable, Encodable {
        public let queryHeads, kvHeads, length, headDim, inputBytes: Int
        public let pageSize, poolBudgetBytes, segmentTargetBytes, allocationBoundBytes: Int
    }

    public struct Result: Sendable {
        public let output: Tensor
        public let storedKeys, storedValues: Tensor?
        public let dispatch, kernelOutputDType: String
        public let offset, segmentCount: Int
        public let pageTable: [Int32]
        public let partitionTokens: Int?
        public let geometry: Geometry
    }

    public enum Failure: Error { case invalidInput, allocationBudget, dispatchMismatch, storageMismatch }

    /// Host validation runs before constructing an MLXArray or a pool.
    public static func validate(_ input: Input) throws -> Geometry {
        let tensors = [input.queries, input.storedKeys, input.storedValues,
                       input.incomingKeys, input.incomingValues]
        var total = 0
        for tensor in tensors {
            guard [.float16, .bfloat16, .float32].contains(tensor.dtype), tensor.shape.count == 4,
                tensor.shape.allSatisfy({ (1...32_768).contains($0) })
            else { throw Failure.invalidInput }
            var bytes = tensor.dtype.size
            for dimension in tensor.shape {
                let (next, overflow) = bytes.multipliedReportingOverflow(by: dimension)
                guard !overflow, next <= 32 << 20 else { throw Failure.allocationBudget }
                bytes = next
            }
            guard tensor.bytes.count == bytes else { throw Failure.invalidInput }
            guard isFinite(tensor) else { throw Failure.invalidInput }
            total += bytes
        }
        let q = input.queries.shape, k = input.storedKeys.shape
        let qh = q[1], kh = k[1], length = k[2], d = q[3]
        guard total <= 32 << 20, qh <= 256, kh <= qh, qh % kh == 0,
            PagedAttentionKernel.supportedHeadDims.contains(d),
            q == [1, qh, 1, d], k == [1, kh, length, d],
            input.storedValues.shape == k,
            input.incomingKeys.shape == [1, kh, 1, d],
            input.incomingValues.shape == input.incomingKeys.shape,
            [input.storedValues, input.incomingKeys, input.incomingValues]
                .allSatisfy({ $0.dtype == input.storedKeys.dtype }),
            2 * qh * length * d <= 100_000_000,
            Float(bitPattern: input.scaleBits).isFinite, Float(bitPattern: input.scaleBits) > 0
        else { throw Failure.invalidInput }
        // Bound host/native copies, both pool slabs, readback and working buffers.
        // This is a conservative allocation plan, not a process-RSS guarantee.
        let pageBytes = 2 * kh * 16 * d * input.storedKeys.dtype.size
        let pages = (length + 15) / 16
        let poolBytes = max(8 << 20, (pages + 64) * pageBytes * 2)
        let bound = total * 6 + poolBytes * 2 + (16 << 20)
        guard bound <= 256 << 20 else { throw Failure.allocationBudget }
        try requireTail(input.storedKeys, equals: input.incomingKeys)
        try requireTail(input.storedValues, equals: input.incomingValues)
        return Geometry(queryHeads: qh, kvHeads: kh, length: length, headDim: d,
            inputBytes: total, pageSize: 16, poolBudgetBytes: poolBytes,
            segmentTargetBytes: pageBytes * 257, allocationBoundBytes: bound)
    }

    private static func isFinite(_ tensor: Tensor) -> Bool {
        tensor.bytes.withUnsafeBytes { raw in
            for offset in stride(from: 0, to: raw.count, by: tensor.dtype.size) {
                if tensor.dtype == .float32 {
                    let bits = UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
                    if bits & 0x7f80_0000 == 0x7f80_0000 { return false }
                } else {
                    let bits = UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
                    let mask: UInt16 = tensor.dtype == .bfloat16 ? 0x7f80 : 0x7c00
                    if bits & mask == mask { return false }
                }
            }
            return true
        }
    }

    private static func requireTail(_ stored: Tensor, equals incoming: Tensor) throws {
        let rowBytes = stored.shape[3] * stored.dtype.size
        for head in 0..<stored.shape[1] {
            let source = (head * stored.shape[2] + stored.shape[2] - 1) * rowBytes
            let target = head * rowBytes
            guard stored.bytes[source..<(source + rowBytes)] == incoming.bytes[target..<(target + rowBytes)]
            else { throw Failure.storageMismatch }
        }
    }

    public static func run(_ input: Input, arm: Arm) throws -> Result {
        let geometry = try validate(input)
        if arm != .nativeSDPA { return try runPaged(input, arm: arm, geometry: geometry) }
        let q = input.queries.array(), k = input.storedKeys.array(), v = input.storedValues.array()
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k.asType(q.dtype), values: v.asType(q.dtype),
            scale: Float(bitPattern: input.scaleBits), mask: .none)
        eval(output)
        return Result(output: Tensor(output), storedKeys: nil, storedValues: nil,
            dispatch: "contiguous_sdpa", kernelOutputDType: String(describing: output.dtype),
            offset: geometry.length, segmentCount: 0, pageTable: [], partitionTokens: nil,
            geometry: geometry)
    }
}
