import MLX

/// Checkpoint destinations are fresh evaluated allocations. Their physical
/// charge is independent of logical tensor bytes and grants no M coverage.
enum CBv2CheckpointAllocationFootprint {
    static func bound(_ bytes: Int) throws -> Int {
        try Memory.allocationFootprintUpperBound(byteCount: bytes)
    }

    static func freshBytes(_ arrays: [MLXArray]) throws -> (bound: Int, actual: Int) {
        var bound = 0, actual = 0
        for array in arrays {
            let upper = try Self.bound(array.nbytes)
            guard let info = try array.evaluatedBufferInfo(), info.isUnique,
                  info.isRowContiguous, info.dataOffset == 0,
                  info.dataElements == array.size,
                  info.allocatedBytes >= array.nbytes, info.allocatedBytes <= upper
            else { throw CBv2CompleteCheckpointError.allocationFailed }
            bound = try add(bound, upper)
            actual = try add(actual, info.allocatedBytes)
        }
        return (bound, actual)
    }

    /// An immutable state view can retain a larger batch allocation. Charge
    /// that whole backing conservatively; aliases never mint M coverage.
    static func retainedBytes(_ array: MLXArray) throws -> Int {
        guard let info = try array.evaluatedBufferInfo(), info.allocatedBytes > 0 else {
            throw CBv2CompleteCheckpointError.allocationFailed
        }
        return info.allocatedBytes
    }

    static func captureBytes(
        _ descriptors: [CBv2CheckpointTensorDescriptor], layers: [Int: CBv2RecurrentLayerState]
    ) throws -> Int {
        let boolean = try bound(MemoryLayout<Bool>.size)
        var total = 0
        for descriptor in descriptors {
            let charge: Int
            switch descriptor.role {
            case .convolution, .assistantFrontier:
                charge = try add(bound(descriptor.byteCount), boolean)
            case .assistantHidden, .assistantTokens:
                // Concatenate and compact copy may coexist until evaluation.
                let copy = try bound(descriptor.byteCount)
                charge = try add(add(copy, copy), boolean)
            case .recurrent:
                guard let index = descriptor.layer, let ssm = layers[index]?.ssm else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                charge = try retainedBytes(ssm)
            case .keys, .values:
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            total = try add(total, charge)
        }
        return total
    }

    static func add(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        guard lhs >= 0, rhs >= 0, !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
        return sum
    }
}
