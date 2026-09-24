// CBv2PagedPrefixLeakTests.swift
//
// Long-horizon metadata and allocator conservation tests for the resident
// physical-page prefix cache. No KV values are read or written: every
// assertion is over host-side page ids, generations, refcounts, and ledgers.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2 resident prefix cache leak resistance", .serialized)
struct CBv2PagedPrefixLeakTests {
    private static let blockSize = 32
    private static let pageSize = 16

    private func kind() -> CBv2LayerKind {
        CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
    }

    private func makeBackend(capacityBytes: Int = 64 << 10) throws -> PagedKVBackend {
        try PagedKVBackend(
            layerKinds: [kind()],
            config: PagedKVPoolConfig(
                pageSize: Self.pageSize,
                capacityBytes: capacityBytes,
                maxPrefillChunk: Self.pageSize,
                nominalMaxSequenceLength: 128,
                maxBufferLength: Int.max),
            residentPrefixCache: CBv2PagedPrefixCacheConfig(
                blockSize: Self.blockSize,
                promptContractID: "resident-leak-tests",
                scopeID: "default-scope"))
    }

    private func makeState(
        backend: PagedKVBackend, tokenCount: Int, maxLength: Int
    ) throws -> (state: [CBv2SequenceKV?], row: PagedSequenceKV) {
        let layer = kind()
        let needs = backend.pageNeeds(layerKinds: backend.layerKinds, maxLength: maxLength)
        try backend.pool.reserve(needs)
        let row = PagedSequenceKV(
            pool: backend.pool,
            kind: layer,
            groupKey: backend.pool.groupKey(forLayer: 0),
            maxLength: maxLength,
            reservedPages: PagedKVPool.pageDemand(
                kind: layer, maxLength: maxLength, config: backend.pool.config))
        for _ in 0 ..< tokenCount { _ = row.prepareDecodeWrite() }
        return ([row], row)
    }

    private func probe(
        _ backend: PagedKVBackend, tokens: [Int], salt: String? = nil
    ) throws -> CBv2PagedPrefixProbe {
        try #require(backend.preparePrefixProbe(tokens: tokens, cacheSalt: salt))
    }

    private func publish(
        _ backend: PagedKVBackend,
        state: [CBv2SequenceKV?],
        probe: CBv2PagedPrefixProbe
    ) {
        #expect(
            backend.publishResidentPrefixBlocks(
                state: state,
                chainHashes: probe.chainHashes,
                blockIndices: 0 ..< probe.chainHashes.count)
                == probe.chainHashes.count)
    }

    private func adopt(
        _ backend: PagedKVBackend,
        match: CBv2PagedPrefixMatch,
        maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        let index = try #require(backend.residentPrefixIndex)
        let shared = try #require(index.resolve(match))
        let pages = try #require(shared.layerPages.first ?? nil)
        let layer = kind()
        let needs = backend.pageNeeds(layerKinds: backend.layerKinds, maxLength: maxLength)
        try backend.pool.reserve(needs)
        guard backend.pool.retainPages(pages) else {
            backend.pool.unreserve(needs)
            throw CBv2KVError.backendIneligible(reason: "resident pages became stale")
        }
        let row = PagedSequenceKV(
            pool: backend.pool,
            kind: layer,
            groupKey: backend.pool.groupKey(forLayer: 0),
            maxLength: maxLength,
            reservedPages: PagedKVPool.pageDemand(
                kind: layer, maxLength: maxLength, config: backend.pool.config))
        row.adoptSharedPages(
            pages,
            storedThrough: match.matchedTokens,
            cursor: match.matchedTokens,
            frozenThrough: 0)
        return [row]
    }

    private func expectFullyConserved(
        _ backend: PagedKVBackend,
        key: PagedKVGroupKey,
        initialPages: Set<Int32>,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let group = backend.pool.group(key)
        #expect(
            group.freeList.count == group.usablePageCount,
            "a physical page leaked from the free queue",
            sourceLocation: sourceLocation)
        #expect(
            Set(group.freeList) == initialPages,
            "the free queue lost or duplicated a page id",
            sourceLocation: sourceLocation)
        #expect(
            group.freeList.count == Set(group.freeList).count,
            "a physical page appears twice in the free queue",
            sourceLocation: sourceLocation)
        #expect(group.pagesInUse == 0, sourceLocation: sourceLocation)
        #expect(group.pagesReserved == 0, sourceLocation: sourceLocation)
        #expect(backend.bytesInUse == 0, sourceLocation: sourceLocation)
        #expect(backend.bytesReserved == 0, sourceLocation: sourceLocation)
        for page in 1 ..< group.pageCount {
            #expect(group.refCounts[page] == 0, sourceLocation: sourceLocation)
        }
        #expect(group.refCounts[Int(group.poisonPage)] == 1, sourceLocation: sourceLocation)
    }

    @Test func repeatedPublishHitReleaseAndReuseCyclesConserveThePool() throws {
        let backend = try makeBackend()
        let key = PagedKVGroupKey(kind())
        let initialPages = Set(backend.pool.group(key).freeList)
        let usable = backend.pool.usablePageCount(group: key)
        let bundleBytes = backend.pool.group(key).pageBytes * 2
        let cycles = 128

        for cycle in 0 ..< cycles {
            let tokens = (0 ..< 33).map { cycle * 1_000 + $0 }
            let donor = try makeState(backend: backend, tokenCount: 32, maxLength: 32)
            let requestProbe = try probe(backend, tokens: tokens)

            // Repeated finalization/publication of the same realization is
            // idempotent: it must not accumulate duplicate metadata aliases.
            for _ in 0 ..< 8 { publish(backend, state: donor.state, probe: requestProbe) }
            #expect(backend.residentPrefixCacheStats?.blockCount == 1)
            #expect(backend.residentPrefixCacheStats?.bytesIndexed == bundleBytes)

            let match = try #require(backend.longestResidentPrefix(for: requestProbe))
            let adopter = try adopt(backend, match: match, maxLength: 32)
            #expect(
                backend.pool.group(key).pagesInUse == 2,
                "a zero-copy hit must not allocate another physical page bundle")
            #expect(
                backend.pool.group(key).pagesReserved == 4,
                "the current admission contract deliberately reserves donor and adopter rows")
            #expect(backend.bytesInUse == bundleBytes)
            #expect(backend.bytesReserved == 2 * bundleBytes)
            backend.release(donor.state)
            backend.release(adopter)
            #expect(backend.bytesInUse == 0)
            #expect(backend.bytesReserved == 0)

            // Drain the physical pool once. Reaching either cached page
            // invalidates its whole bundle before generation reuse.
            var allocated: [Int32] = []
            allocated.reserveCapacity(usable)
            for _ in 0 ..< usable { allocated.append(backend.pool.allocatePage(group: key)) }
            #expect(Set(allocated) == initialPages)
            #expect(backend.residentPrefixCacheStats?.blockCount == 0)
            #expect(backend.residentPrefixCacheStats?.bytesIndexed == 0)

            backend.pool.freePages(group: key, pages: allocated)
            expectFullyConserved(backend, key: key, initialPages: initialPages)
        }

        let stats = try #require(backend.residentPrefixCacheStats)
        #expect(stats.hits == cycles)
        #expect(stats.tokensSaved == cycles * Self.blockSize)
        #expect(stats.invalidations == cycles)
        #expect(stats.blockCount == 0)
        #expect(stats.bytesIndexed == 0)
    }

    @Test func duplicatePhysicalRealizationsReplaceIndependently() throws {
        let backend = try makeBackend(capacityBytes: 96 << 10)
        let key = PagedKVGroupKey(kind())
        let initialPages = Set(backend.pool.group(key).freeList)
        let usable = backend.pool.usablePageCount(group: key)
        let tokens = Array(0 ..< 33)
        let requestProbe = try probe(backend, tokens: tokens)

        let donorA = try makeState(backend: backend, tokenCount: 32, maxLength: 32)
        defer { backend.release(donorA.state) }
        let donorB = try makeState(backend: backend, tokenCount: 32, maxLength: 32)
        defer { backend.release(donorB.state) }
        let pagesA = try #require(donorA.row.prefixPageHandles(tokens: 0 ..< 32))
        let pagesB = try #require(donorB.row.prefixPageHandles(tokens: 0 ..< 32))
        #expect(Set(pagesA.map(\.page)).isDisjoint(with: Set(pagesB.map(\.page))))

        for _ in 0 ..< 8 {
            publish(backend, state: donorA.state, probe: requestProbe)
            publish(backend, state: donorB.state, probe: requestProbe)
        }
        #expect(backend.residentPrefixCacheStats?.blockCount == 2)
        backend.release(donorA.state)
        backend.release(donorB.state)

        var allocated: [Int32] = []
        while allocated.count < usable,
            backend.residentPrefixCacheStats?.blockCount == 2
        {
            allocated.append(backend.pool.allocatePage(group: key))
        }
        #expect(backend.residentPrefixCacheStats?.blockCount == 1)

        // Invalidating one physical realization must leave the duplicate hit
        // available, and resolution must select one complete unmixed bundle.
        let survivingMatch = try #require(backend.longestResidentPrefix(for: requestProbe))
        let surviving = try #require(backend.residentPrefixIndex?.resolve(survivingMatch))
        let survivingPages = try #require(surviving.layerPages.first ?? nil)
        #expect(survivingPages == pagesA || survivingPages == pagesB)

        while allocated.count < usable,
            backend.residentPrefixCacheStats?.blockCount != 0
        {
            allocated.append(backend.pool.allocatePage(group: key))
        }
        #expect(backend.residentPrefixCacheStats?.blockCount == 0)
        #expect(backend.residentPrefixCacheStats?.bytesIndexed == 0)
        while allocated.count < usable {
            allocated.append(backend.pool.allocatePage(group: key))
        }
        #expect(Set(allocated) == initialPages)

        backend.pool.freePages(group: key, pages: allocated)
        expectFullyConserved(backend, key: key, initialPages: initialPages)
        #expect(backend.residentPrefixCacheStats?.invalidations == 2)
    }

    @Test func transactionalRetainRejectsMixedGenerationsWithoutPartialIncrement() throws {
        let backend = try makeBackend()
        let key = PagedKVGroupKey(kind())
        let initialPages = Set(backend.pool.group(key).freeList)
        let first = backend.pool.allocatePage(group: key)
        let second = backend.pool.allocatePage(group: key)
        let valid = backend.pool.currentHandle(group: key, page: first)
        let soonStale = backend.pool.currentHandle(group: key, page: second)

        backend.pool.freePages(group: key, pages: [second])
        let recycled = backend.pool.allocatePage(group: key)
        #expect(recycled == second)
        let current = backend.pool.currentHandle(group: key, page: recycled)
        #expect(!backend.pool.isValid(soonStale))
        #expect(current.generation != soonStale.generation)

        let beforeFreeList = backend.pool.group(key).freeList
        let beforeRefCounts = backend.pool.group(key).refCounts
        #expect(!backend.pool.retainPages([valid, soonStale]))
        #expect(!backend.pool.retainPages([soonStale, valid]))
        #expect(backend.pool.group(key).freeList == beforeFreeList)
        #expect(backend.pool.group(key).refCounts == beforeRefCounts)
        #expect(backend.pool.refCount(for: valid) == 1)
        #expect(backend.pool.refCount(for: current) == 1)

        backend.pool.freePages(group: key, pages: [first, recycled])
        expectFullyConserved(backend, key: key, initialPages: initialPages)
    }

    @Test func conflictingIdentityIsQuarantinedUntilGenerationReuse() throws {
        let backend = try makeBackend()
        let key = PagedKVGroupKey(kind())
        let initialPages = Set(backend.pool.group(key).freeList)
        let usable = backend.pool.usablePageCount(group: key)
        let tokens = Array(0 ..< 33)
        let donor = try makeState(backend: backend, tokenCount: 32, maxLength: 32)
        let original = try probe(backend, tokens: tokens)
        publish(backend, state: donor.state, probe: original)
        #expect(backend.residentPrefixCacheStats?.blockCount == 1)

        // The same generation cannot truthfully represent another hash/scope.
        // The first conflict invalidates the old identity; repeated hostile
        // publications must remain metadata-bounded and fail closed.
        for salt in 0 ..< 256 {
            let conflicting = try probe(backend, tokens: tokens, salt: "tenant-\(salt)")
            #expect(
                backend.publishResidentPrefixBlocks(
                    state: donor.state,
                    chainHashes: conflicting.chainHashes,
                    blockIndices: 0 ..< conflicting.chainHashes.count)
                    == 0)
        }
        #expect(backend.residentPrefixCacheStats?.blockCount == 0)
        #expect(backend.residentPrefixCacheStats?.bytesIndexed == 0)
        #expect(backend.longestResidentPrefix(for: original) == nil)

        backend.release(donor.state)
        var allocated: [Int32] = []
        allocated.reserveCapacity(usable)
        for _ in 0 ..< usable { allocated.append(backend.pool.allocatePage(group: key)) }
        #expect(Set(allocated) == initialPages)
        backend.pool.freePages(group: key, pages: allocated)
        expectFullyConserved(backend, key: key, initialPages: initialPages)

        // Reuse advanced every generation and cleared the quarantine, so a
        // freshly computed realization can be indexed normally.
        let fresh = try makeState(backend: backend, tokenCount: 32, maxLength: 32)
        let freshProbe = try probe(backend, tokens: tokens, salt: "tenant-fresh")
        publish(backend, state: fresh.state, probe: freshProbe)
        #expect(backend.residentPrefixCacheStats?.blockCount == 1)
        backend.release(fresh.state)
    }
}
