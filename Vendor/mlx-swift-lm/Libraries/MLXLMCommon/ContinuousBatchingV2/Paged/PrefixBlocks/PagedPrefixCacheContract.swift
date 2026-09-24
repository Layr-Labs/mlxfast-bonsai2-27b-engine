// PagedPrefixCacheContract.swift
//
// Internal engine/backend seam for copy-free prefix sharing. The existing
// public CBv2PrefixCache snapshot contract remains unchanged so contiguous
// and durable SSD implementations stay source- and behavior-compatible.

import Foundation

public struct CBv2PagedPrefixCacheConfig: Sendable, Equatable {
    /// Tokens per indexed block. The default is one physical page, matching
    /// vLLM's page/block identity; larger page-aligned blocks trade lookup
    /// granularity for less hashing and metadata.
    public var blockSize: Int
    public var promptContractID: String
    public var scopeID: String

    public init(
        blockSize: Int = CBv2PagedDefaults.pageSize,
        promptContractID: String = "",
        scopeID: String = ""
    ) {
        self.blockSize = blockSize
        self.promptContractID = promptContractID
        self.scopeID = scopeID
    }
}

public struct CBv2PagedPrefixCacheStats: Sendable, Equatable {
    public var hits: Int
    public var misses: Int
    public var tokensSaved: Int
    public var blockCount: Int
    public var bytesIndexed: Int
    public var invalidations: Int
}

struct CBv2PagedPrefixProbe: Sendable {
    let hasher: CBv2BlockHasher
    let chainHashes: [Data]
    let maxLookupBlocks: Int

    var blockSize: Int { hasher.blockSize }
}

struct CBv2PagedPrefixCursor {
    let hasher: CBv2BlockHasher
    var chainHashes: [Data]
    var publishedBlockCount: Int
}

struct CBv2PagedPrefixMatch: Sendable {
    let chainHashes: [Data]
    let matchedTokens: Int
    let blockSize: Int
}

struct CBv2PagedSharedPrefix: Sendable {
    let matchedTokens: Int
    let layerPages: [[PagedKVPageHandle]?]
}

protocol CBv2PagedPrefixSharingBackend: CBv2KVBackend {
    func preparePrefixProbe(tokens: [Int], cacheSalt: String?) -> CBv2PagedPrefixProbe?
    func peekResidentPrefix(for probe: CBv2PagedPrefixProbe) -> CBv2PagedPrefixMatch?
    func longestResidentPrefix(for probe: CBv2PagedPrefixProbe) -> CBv2PagedPrefixMatch?
    func makeSequenceState(
        sharing match: CBv2PagedPrefixMatch,
        plan: CBv2PrefixReusePlan,
        layerKinds: [CBv2LayerKind],
        maxLength: Int
    ) throws -> [CBv2SequenceKV?]
    @discardableResult
    func publishResidentPrefixBlocks(
        state: [CBv2SequenceKV?],
        chainHashes: [Data],
        blockIndices: Range<Int>
    ) -> Int
    var residentPrefixCacheStats: CBv2PagedPrefixCacheStats? { get }
}
