import Foundation
import MLX

/// Process-local exact recurrent checkpoints. Constructed for one loaded
/// model; the factory reserves this budget inside that slot's KV allocation.
public struct CBv2HybridPrefixCacheConfig: Sendable, Equatable {
    public var maximumBytes: Int
    public var maximumEntries: Int
    public var maximumCheckpointsPerRequest: Int
    public var modelID: String
    public var promptContractID: String
    public var buildID: String

    public init(
        maximumBytes: Int, maximumEntries: Int = 32,
        maximumCheckpointsPerRequest: Int = 2,
        modelID: String, promptContractID: String, buildID: String
    ) {
        self.maximumBytes = maximumBytes
        self.maximumEntries = maximumEntries
        self.maximumCheckpointsPerRequest = maximumCheckpointsPerRequest
        self.modelID = modelID
        self.promptContractID = promptContractID
        self.buildID = buildID
    }

    var isValid: Bool {
        maximumBytes > 0 && maximumEntries > 0 && maximumCheckpointsPerRequest > 0
            && !modelID.isEmpty && !promptContractID.isEmpty && !buildID.isEmpty
    }
}

public struct CBv2HybridPrefixCacheStats: Sendable, Equatable {
    public var residentBytes: Int = 0
    public var stagedBytes: Int = 0
    public var publishingBytes: Int = 0
    public var entries: Int = 0
    public var checkpoints: Int = 0
    public var lookupMatches: Int = 0
    public var misses: Int = 0
    public var adoptions: Int = 0
    public var tokensSaved: Int = 0
    public var capacityRefusals: Int = 0
    public var evictions: Int = 0
    /// Cumulative compact-copy attempts and reserved destination bytes.
    public var kvCompactions: Int = 0
    public var kvCompactionBytes: Int = 0

    public var retainedBytes: Int { residentBytes + stagedBytes + publishingBytes }
}

struct CBv2RecurrentCheckpoint {
    let storageID = UUID()
    let position: Int
    let chunkSize: Int
    let layers: [Int: CBv2RecurrentLayerState]
    let byteCount: Int
    var assistant: (any CBv2MTPPrefixCheckpoint)? = nil

    var evaluationRoots: [MLXArray] {
        layers.values.flatMap { [$0.conv, $0.ssm].compactMap { $0 } }
            + (assistant?.evaluationTargets ?? [])
    }
}

struct CBv2HybridPrefixHit {
    let pin: UInt64
    let checkpoint: CBv2RecurrentCheckpoint
    let kvPrefix: [(keys: MLXArray, values: MLXArray, offset: Int)?]
    /// Actual backing bytes, including slack beyond the adopted prefix.
    let kvBackingBytes: Int
}

/// Exactness requires every donor chunk below a checkpoint to have identical
/// launch geometry. Packed rows, ragged chunks and preemption disarm capture.
struct CBv2RecurrentCheckpointGeometry {
    var position: Int = 0
    var chunkSize: Int?
    var isArmed = true

    mutating func record(range: Range<Int>, cap: Int, promptLength: Int, packed: Bool) -> Bool {
        guard isArmed else { return false }
        guard !packed, cap > 1, range.lowerBound == position,
            range.upperBound <= promptLength, range.count == cap,
            range.lowerBound % cap == 0, chunkSize == nil || chunkSize == cap
        else {
            isArmed = false
            return false
        }
        position = range.upperBound
        chunkSize = cap
        return true
    }
}
