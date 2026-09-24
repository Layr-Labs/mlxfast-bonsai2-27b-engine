// CBv2PagedPrefixBlockCacheTests.swift
//
// Physical-page lifecycle coverage for the resident (vLLM-style) prefix
// cache. These tests intentionally inspect page ids, generations, and pool
// ledgers: value-parity tests cannot distinguish sharing from a hidden copy.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2 resident paged prefix cache")
struct CBv2PagedPrefixBlockCacheTests {
    private static let blockSize = 32
    private static let pageSize = 16

    private func fullKind() -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
    }

    private func makeBackend(capacityBytes: Int = 512 << 10) throws -> PagedKVBackend {
        let kind = fullKind()
        return try PagedKVBackend(
            layerKinds: [kind],
            config: PagedKVPoolConfig(
                pageSize: Self.pageSize,
                capacityBytes: capacityBytes,
                maxPrefillChunk: Self.pageSize,
                nominalMaxSequenceLength: 128,
                // These tests exercise allocator/index metadata without
                // dispatching Metal. Avoid depending on GPU.deviceInfo(),
                // which reports no buffer limit under plain SwiftPM on hosts
                // without the separately installed Metal toolchain.
                maxBufferLength: Int.max),
            residentPrefixCache: CBv2PagedPrefixCacheConfig(
                blockSize: Self.blockSize,
                promptContractID: "resident-cache-tests",
                scopeID: "default-scope"))
    }

    /// Allocate page-table entries without dispatching a write kernel. The
    /// page-native cache is metadata-only, so page identity/refcounts are the
    /// exact seam under test; existing paged-kernel suites cover the bytes.
    private func makeState(
        backend: PagedKVBackend, tokenCount: Int, maxLength: Int
    ) throws -> (state: [CBv2SequenceKV?], row: PagedSequenceKV) {
        let kind = try #require(backend.layerKinds.first)
        let needs = backend.pageNeeds(layerKinds: backend.layerKinds, maxLength: maxLength)
        try backend.pool.reserve(needs)
        let row = PagedSequenceKV(
            pool: backend.pool,
            kind: kind,
            groupKey: backend.pool.groupKey(forLayer: 0),
            maxLength: maxLength,
            reservedPages: PagedKVPool.pageDemand(
                kind: kind, maxLength: maxLength, config: backend.pool.config))
        let state: [CBv2SequenceKV?] = [row]
        for _ in 0 ..< tokenCount {
            _ = row.prepareDecodeWrite()
        }
        return (state, row)
    }

    /// Exercise the same generation validation, all-or-nothing retain, and
    /// row page-table installation as the backend adoption path without
    /// materializing MLX slabs. `swift test` does not bundle the metallib;
    /// the project's xcodebuild suites cover slab commitment separately.
    private func adoptResolvedPages(
        backend: PagedKVBackend,
        match: CBv2PagedPrefixMatch,
        maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        let index = try #require(backend.residentPrefixIndex)
        let shared = try #require(index.resolve(match))
        let kind = try #require(backend.layerKinds.first)
        let pages = try #require(shared.layerPages.first ?? nil)
        let needs = backend.pageNeeds(layerKinds: backend.layerKinds, maxLength: maxLength)
        try backend.pool.reserve(needs)
        guard backend.pool.retainPages(pages) else {
            backend.pool.unreserve(needs)
            throw CBv2KVError.backendIneligible(
                reason: "resident pages changed generation during test adoption")
        }
        let row = PagedSequenceKV(
            pool: backend.pool,
            kind: kind,
            groupKey: backend.pool.groupKey(forLayer: 0),
            maxLength: maxLength,
            reservedPages: PagedKVPool.pageDemand(
                kind: kind, maxLength: maxLength, config: backend.pool.config))
        row.adoptSharedPages(
            pages,
            storedThrough: match.matchedTokens,
            cursor: match.matchedTokens,
            frozenThrough: 0)
        return [row]
    }

    private func probe(
        _ backend: PagedKVBackend, tokens: [Int], salt: String? = nil
    ) throws -> CBv2PagedPrefixProbe {
        try #require(backend.preparePrefixProbe(tokens: tokens, cacheSalt: salt))
    }

    @discardableResult
    private func publish(
        _ backend: PagedKVBackend,
        state: [CBv2SequenceKV?],
        tokens: [Int],
        salt: String? = nil
    ) throws -> CBv2PagedPrefixProbe {
        let probe = try probe(backend, tokens: tokens, salt: salt)
        #expect(
            backend.publishResidentPrefixBlocks(
                state: state,
                chainHashes: probe.chainHashes,
                blockIndices: 0 ..< probe.chainHashes.count)
                == probe.chainHashes.count)
        return probe
    }

    private func plan(
        for backend: PagedKVBackend,
        match: CBv2PagedPrefixMatch,
        maxLength: Int
    ) throws -> CBv2PrefixReusePlan {
        let capability = CBv2PrefixReuseCapability.derive(
            layerKinds: backend.layerKinds, backend: .pagedFP16)
        return try #require(
            capability.plan(
                matchedBoundary: match.matchedTokens,
                maximumSequenceLength: maxLength))
    }

    @Test func physicalPagesAreSharedAndZeroRefPagesCanBeResurrected() throws {
        let backend = try makeBackend()
        let tokens = Array(0 ..< 65) // two reusable blocks plus the required live token
        let donor = try makeState(backend: backend, tokenCount: 64, maxLength: 96)
        defer { backend.release(donor.state) }

        let donorPages = donor.row.table
        let handles = try #require(donor.row.prefixPageHandles(tokens: 0 ..< 64))
        #expect(handles.count == 4)
        _ = try publish(backend, state: donor.state, tokens: tokens)

        let firstProbe = try probe(backend, tokens: tokens)
        let firstMatch = try #require(backend.longestResidentPrefix(for: firstProbe))
        #expect(firstMatch.matchedTokens == 64)
        let firstPlan = try plan(for: backend, match: firstMatch, maxLength: 96)
        #expect(firstPlan.strategy == .direct)
        let adopter = try adoptResolvedPages(
            backend: backend, match: firstMatch, maxLength: 96)
        defer { backend.release(adopter) }
        let adopterRow = try #require(adopter[0] as? PagedSequenceKV)

        #expect(adopterRow.table == donorPages, "a hit must install the donor's page ids")
        for handle in handles {
            #expect(backend.pool.refCount(for: handle) == 2)
        }

        backend.release(donor.state)
        for handle in handles {
            #expect(backend.pool.refCount(for: handle) == 1)
        }

        // The cache is non-owning: releasing the last request puts its pages
        // back on the allocator queue while the generation-valid index stays.
        backend.release(adopter)
        for handle in handles {
            #expect(backend.pool.refCount(for: handle) == 0)
        }

        let secondProbe = try probe(backend, tokens: tokens)
        let secondMatch = try #require(backend.longestResidentPrefix(for: secondProbe))
        let secondPlan = try plan(for: backend, match: secondMatch, maxLength: 96)
        #expect(secondPlan.strategy == .direct)
        let resurrected = try adoptResolvedPages(
            backend: backend, match: secondMatch, maxLength: 96)
        defer { backend.release(resurrected) }
        let resurrectedRow = try #require(resurrected[0] as? PagedSequenceKV)
        #expect(resurrectedRow.table == donorPages)
        for handle in handles {
            #expect(backend.pool.refCount(for: handle) == 1)
        }
    }

    @Test func allocationReuseInvalidatesTheOldGenerationAndEveryBlockAlias() throws {
        let backend = try makeBackend(capacityBytes: 64 << 10)
        let tokens = Array(0 ..< 33)
        let donor = try makeState(backend: backend, tokenCount: 32, maxLength: 32)
        let handles = try #require(donor.row.prefixPageHandles(tokens: 0 ..< 32))
        _ = try publish(backend, state: donor.state, tokens: tokens)
        #expect(backend.longestResidentPrefix(for: try probe(backend, tokens: tokens)) != nil)

        backend.release(donor.state)
        #expect(handles.allSatisfy { backend.pool.refCount(for: $0) == 0 })

        let key = PagedKVGroupKey(fullKind())
        let oldPages = Set(handles.map(\.page))
        var allocated: [Int32] = []
        var recycled: Int32?
        for _ in 0 ..< backend.pool.usablePageCount(group: key) {
            let page = backend.pool.allocatePage(group: key)
            allocated.append(page)
            if oldPages.contains(page) {
                recycled = page
                break
            }
        }
        defer { backend.pool.freePages(group: key, pages: allocated) }

        let recycledPage = try #require(recycled)
        let oldHandle = try #require(handles.first { $0.page == recycledPage })
        let newHandle = backend.pool.currentHandle(group: key, page: recycledPage)
        #expect(newHandle.generation != oldHandle.generation)
        #expect(!backend.pool.isValid(oldHandle), "page-id reuse must not pass an ABA check")
        let orphanedPage = try #require(oldPages.subtracting([recycledPage]).first)
        #expect(
            backend.pool.group(key).freeList.first == orphanedPage,
            "the rest of an invalidated multi-page bundle must become immediately reusable")
        #expect(
            backend.longestResidentPrefix(for: try probe(backend, tokens: tokens)) == nil,
            "reusing any page in a block must remove the complete block bundle")
        #expect(backend.residentPrefixCacheStats?.invalidations == 1)
        #expect(backend.residentPrefixCacheStats?.blockCount == 0)
    }

    @Test func sameHashTailReplayReplacesTheOverlappingRealization() throws {
        let backend = try makeBackend(capacityBytes: 64 << 10)
        let tokens = Array(0 ..< 33)
        let donor = try makeState(backend: backend, tokenCount: 32, maxLength: 32)
        defer { backend.release(donor.state) }
        let originalPages = try #require(
            donor.row.prefixPageHandles(tokens: 0 ..< Self.blockSize))
        _ = try publish(backend, state: donor.state, tokens: tokens)

        // A tail replay can preserve the first allocator page of a logical
        // cache block while replacing its second page. Its full-block content
        // hash remains the same, so the new complete realization must replace
        // the overlapping old one instead of being treated as a conflict.
        donor.row.rollback(Self.pageSize)
        for _ in 0 ..< Self.pageSize { _ = donor.row.prepareDecodeWrite() }
        let replayedPages = try #require(
            donor.row.prefixPageHandles(tokens: 0 ..< Self.blockSize))
        #expect(replayedPages[0] == originalPages[0])
        #expect(replayedPages[1] != originalPages[1])

        _ = try publish(backend, state: donor.state, tokens: tokens)
        let match = try #require(
            backend.longestResidentPrefix(for: try probe(backend, tokens: tokens)))
        let resolved = try #require(backend.residentPrefixIndex?.resolve(match))
        #expect(try #require(resolved.layerPages.first ?? nil) == replayedPages)
        #expect(backend.residentPrefixCacheStats?.blockCount == 1)
        #expect(backend.residentPrefixCacheStats?.invalidations == 1)

        let key = PagedKVGroupKey(fullKind())
        #expect(
            backend.pool.group(key).freeList.first == originalPages[1].page,
            "the replaced tail must become immediately reusable")
    }

    @Test func lookupStopsAtTheFirstDivergenceAndCacheSaltIsolatesChains() throws {
        let backend = try makeBackend()
        let common = Array(0 ..< 32)
        let branchA = common + Array(1_000 ..< 1_033)
        let branchB = common + Array(2_000 ..< 2_033)
        let uncachedBranch = common + Array(3_000 ..< 3_033)
        let foreignRoot = Array(4_000 ..< 4_065)

        let donorA = try makeState(backend: backend, tokenCount: 64, maxLength: 96)
        defer { backend.release(donorA.state) }
        let donorB = try makeState(backend: backend, tokenCount: 64, maxLength: 96)
        defer { backend.release(donorB.state) }
        _ = try publish(backend, state: donorA.state, tokens: branchA)
        _ = try publish(backend, state: donorB.state, tokens: branchB)

        let matchA = try #require(
            backend.longestResidentPrefix(for: try probe(backend, tokens: branchA)))
        let matchB = try #require(
            backend.longestResidentPrefix(for: try probe(backend, tokens: branchB)))
        let commonOnly = try #require(
            backend.longestResidentPrefix(for: try probe(backend, tokens: uncachedBranch)))
        #expect(matchA.matchedTokens == 64)
        #expect(matchB.matchedTokens == 64)
        #expect(commonOnly.matchedTokens == 32)
        #expect(
            backend.longestResidentPrefix(for: try probe(backend, tokens: foreignRoot)) == nil)

        // Each authenticated scope computes its own physical realization; a
        // lookup can only traverse the chain for that scope's salt.
        let tenantDonor = try makeState(backend: backend, tokenCount: 64, maxLength: 96)
        defer { backend.release(tenantDonor.state) }
        _ = try publish(
            backend, state: tenantDonor.state, tokens: branchA, salt: "tenant-a")
        let tenantA = try #require(
            backend.longestResidentPrefix(
                for: try probe(backend, tokens: branchA, salt: "tenant-a")))
        #expect(tenantA.matchedTokens == 64)
        #expect(
            backend.longestResidentPrefix(
                for: try probe(backend, tokens: branchA, salt: "tenant-b")) == nil)

        let statsBeforePeek = try #require(backend.residentPrefixCacheStats)
        let advisory = backend.peekResidentPrefix(
            for: try probe(backend, tokens: branchA, salt: "tenant-a"))
        #expect(advisory?.matchedTokens == 64)
        #expect(
            backend.residentPrefixCacheStats == statsBeforePeek,
            "provider preflight must not count as an engine cache lookup")
    }

    @Test func failedAdoptionLeavesReservationsReferencesAndTablesUntouched() throws {
        let backend = try makeBackend(capacityBytes: 64 << 10)
        let tokens = Array(0 ..< 65)
        let donor = try makeState(backend: backend, tokenCount: 64, maxLength: 64)
        defer { backend.release(donor.state) }
        let handles = try #require(donor.row.prefixPageHandles(tokens: 0 ..< 64))
        _ = try publish(backend, state: donor.state, tokens: tokens)
        let match = try #require(
            backend.longestResidentPrefix(for: try probe(backend, tokens: tokens)))
        let impossibleLength = 10_000
        let reusePlan = try plan(
            for: backend, match: match, maxLength: impossibleLength)

        let key = PagedKVGroupKey(fullKind())
        let beforeReserved = backend.bytesReserved
        let beforeFreeList = backend.pool.group(key).freeList
        let beforeRefCounts = backend.pool.group(key).refCounts
        let beforeTable = donor.row.table

        do {
            let leaked = try backend.makeSequenceState(
                sharing: match,
                plan: reusePlan,
                layerKinds: backend.layerKinds,
                maxLength: impossibleLength)
            backend.release(leaked)
            Issue.record("expected resident adoption to fail capacity admission")
        } catch CBv2KVError.capacityExhausted {
            // Expected: pool.reserve validates the whole request before it
            // retains one shared page or constructs one row.
        } catch {
            Issue.record("unexpected adoption failure: \(error)")
        }

        #expect(backend.bytesReserved == beforeReserved)
        #expect(backend.pool.group(key).freeList == beforeFreeList)
        #expect(backend.pool.group(key).refCounts == beforeRefCounts)
        #expect(donor.row.table == beforeTable)
        for handle in handles {
            #expect(backend.pool.refCount(for: handle) == 1)
        }
    }

    @Test func finalizedPublicationDoesNotExposeTheOptimisticSuccessorFrontier() throws {
        let backend = try makeBackend()
        let tokens = Array(0 ..< 65)
        let donor = try makeState(backend: backend, tokenCount: 64, maxLength: 96)
        defer { backend.release(donor.state) }

        let id = CBv2RequestID(0xCA_C4_E0)
        let scheduler = SchedulerV2(
            config: CBv2SchedulerConfig(enablePrefixCache: true))
        let record = try scheduler.enqueue(
            CBv2Request(id: id, promptTokens: tokens, maxTokens: 1))
        // This is the chained-step hazard: while N is finalizing, planning
        // N+1 has already moved scheduler truth through the second block.
        record.numComputedTokens = 64

        let loop = EngineLoopV2(
            model: CBv2SchedScriptedModel(),
            layerKinds: backend.layerKinds,
            backend: backend,
            cacheProvider: CBv2SchedMockCacheProvider(layerKinds: backend.layerKinds),
            sampler: CBv2GreedySampler(),
            detokenizerFactory: CBv2SchedScriptedDetokFactory(),
            scheduler: scheduler,
            capacity: nil,
            residentPrefixBackend: backend,
            config: CBv2EngineLoopConfig(),
            gauges: CBv2EngineGauges(kvBytesCapacity: backend.bytesCapacity))
        loop.kvStates[id] = donor.state
        let requestProbe = try probe(backend, tokens: tokens)
        loop.residentPrefixCursorByID[id] = CBv2PagedPrefixCursor(
            hasher: requestProbe.hasher,
            chainHashes: [],
            publishedBlockCount: 0)

        loop.publishFinalizedResidentBlocks(requestID: id, safeComputedEnd: 32)
        let afterN = try #require(
            backend.longestResidentPrefix(for: try probe(backend, tokens: tokens)))
        #expect(afterN.matchedTokens == 32)
        #expect(
            backend.residentPrefixCacheStats?.blockCount == 1,
            "scheduler.numComputedTokens must not leak N+1 into N's publication")

        loop.publishFinalizedResidentBlocks(requestID: id, safeComputedEnd: 64)
        let afterSuccessor = try #require(
            backend.longestResidentPrefix(for: try probe(backend, tokens: tokens)))
        #expect(afterSuccessor.matchedTokens == 64)
        #expect(backend.residentPrefixCacheStats?.blockCount == 2)
    }
}
