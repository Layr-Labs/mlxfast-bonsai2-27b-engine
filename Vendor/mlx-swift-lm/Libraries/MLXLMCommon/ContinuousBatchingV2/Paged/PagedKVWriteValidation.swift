import MLX

/// A runtime projection changed the native KV storage contract established at
/// engine construction. These are metadata only; no prompt or tensors escape.
public struct CBv2PagedKVWriteError: Error, Sendable, CustomStringConvertible {
    public let layerIndex: Int?
    public let expected: DType
    public let keys: DType
    public let values: DType

    public var description: String {
        "paged KV dtype mismatch at layer \(layerIndex.map(String.init) ?? "unknown"): expected \(expected), keys \(keys), values \(values)"
    }
}

/// Shared by all groups/caches of one pool, confined to its engine queue.
/// A nonthrowing model forward can finish constructing a shape-valid graph,
/// but no later attention operation may mutate/read cache after the first fault.
/// The engine checks this latch before sampling or evaluating that graph.
final class CBv2PagedKVWriteValidation {
    private(set) var fault: CBv2PagedKVWriteError?
    var isFaulted: Bool { fault != nil }

    @discardableResult
    func validate(keys: MLXArray, values: MLXArray, expected: DType, layerIndex: Int? = nil) -> Bool {
        guard fault == nil else { return false }
        guard keys.dtype == expected, values.dtype == expected else {
            fault = CBv2PagedKVWriteError(
                layerIndex: layerIndex, expected: expected, keys: keys.dtype, values: values.dtype)
            return false
        }
        return true
    }

    func record(_ error: CBv2PagedKVWriteError) { if fault == nil { fault = error } }
    func check() throws { if let fault { throw fault } }
    func clearAfterRetirement() { fault = nil }
}

/// Retain the last valid write-fence graph before building a step. On failure
/// no newly built fence may survive into another request's cache operation.
/// The affected rows are retired completely; this does not claim cursor-only
/// rollback can restore a window overwritten by an already evaluated MTP column.
final class CBv2PagedWriteBoundary {
    private var fences: [(PagedKVGroup, MLXArray)]

    init(pool: PagedKVPool) {
        fences = pool.groupKeys.map {
            let group = pool.group($0)
            return (group, group.writeFence)
        }
    }

    func discardFailedGraphAfterSynchronization() {
        for (group, fence) in fences { group.writeFence = fence }
        fences.removeAll()
    }
}
