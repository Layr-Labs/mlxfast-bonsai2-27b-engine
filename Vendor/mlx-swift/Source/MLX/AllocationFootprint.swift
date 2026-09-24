// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

extension Memory {
    /// Bound one buffer under the current allocator's alignment and cache-reuse
    /// rules. This performs no allocation, stream work, or cache inspection.
    /// Graph inputs and scratch are separate allocations and need separate bounds.
    public static func allocationFootprintUpperBound(byteCount: Int) throws -> Int {
        guard byteCount >= 0 else { throw MLXError.caught("negative allocation byte count") }
        return try withError {
            var result: size_t = 0
            guard mlx_get_allocation_size_upper_bound(&result, byteCount) == 0, result >= byteCount else {
                throw MLXError.caught("allocation footprint overflow")
            }
            return result
        }
    }
}

extension MLXArray {
    /// Observed metadata for an already evaluated buffer. Multiple arrays may
    /// share this allocation; this value does not grant ownership or credit.
    public struct BufferInfo: Sendable {
        public let allocatedBytes: Int
        public let dataOffset: Int
        public let dataElements: Int
        public let isRowContiguous: Bool
        public let isUnique: Bool
    }

    /// Returns nil until available. Never evaluates or waits for the array.
    public func evaluatedBufferInfo() throws -> BufferInfo? {
        try withError {
            var available = false, contiguous = false, unique = false
            var bytes: size_t = 0, offset: size_t = 0, elements: size_t = 0
            guard mlx_array_get_buffer_info(
                &available, &bytes, &offset, &elements, &contiguous, &unique, ctx) == 0 else {
                throw MLXError.caught("unable to inspect evaluated buffer")
            }
            guard available else { return nil }
            return BufferInfo(allocatedBytes: bytes, dataOffset: offset, dataElements: elements,
                              isRowContiguous: contiguous, isUnique: unique)
        }
    }
}

/// Immutable scalar allocator policy. Its bound calculations do no allocation,
/// error collection, waiting, or lock acquisition, including on overflow.
public struct AllocationFootprintPolicy: @unchecked Sendable {
    // This imported C value contains five immutable size_t scalars only.
    private let value: mlx_allocation_footprint_policy

    fileprivate init(_ value: mlx_allocation_footprint_policy) { self.value = value }

    public func upperBound(byteCount: Int) -> Int? {
        guard byteCount >= 0 else { return nil }
        var policy = value
        var result: size_t = 0
        guard mlx_allocation_footprint_policy_bound(&result, &policy, byteCount) == 0,
              result >= byteCount else { return nil }
        return result
    }

    public var maximumExtraBytes: Int? {
        var policy = value
        var result: size_t = 0
        guard mlx_allocation_footprint_policy_maximum_extra(&result, &policy) == 0,
              result >= 0 else { return nil }
        return result
    }
}

extension Memory {
    public static func allocationFootprintPolicy() -> AllocationFootprintPolicy? {
        var value = mlx_allocation_footprint_policy()
        guard mlx_get_allocation_footprint_policy(&value) == 0 else { return nil }
        return AllocationFootprintPolicy(value)
    }
}
