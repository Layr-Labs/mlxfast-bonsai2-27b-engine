// PagedKVBackend+PrefixSharing.swift
//
// Transactional adoption/publication for the resident physical-page prefix
// cache. Snapshot adoption remains in PagedKVBackend.swift as the durable SSD
// and contiguous compatibility path.

import Foundation

extension PagedKVBackend: CBv2PagedPrefixSharingBackend {
    func preparePrefixProbe(tokens: [Int], cacheSalt: String?) -> CBv2PagedPrefixProbe? {
        residentPrefixIndex?.prepareProbe(tokens: tokens, cacheSalt: cacheSalt)
    }

    func longestResidentPrefix(for probe: CBv2PagedPrefixProbe) -> CBv2PagedPrefixMatch? {
        residentPrefixIndex?.longestResidentPrefix(for: probe)
    }

    func peekResidentPrefix(for probe: CBv2PagedPrefixProbe) -> CBv2PagedPrefixMatch? {
        residentPrefixIndex?.peekResidentPrefix(for: probe)
    }

    func makeSequenceState(
        sharing match: CBv2PagedPrefixMatch,
        plan: CBv2PrefixReusePlan,
        layerKinds: [CBv2LayerKind],
        maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        guard let residentPrefixIndex, let shared = residentPrefixIndex.resolve(match) else {
            throw CBv2KVError.backendIneligible(reason: "resident prefix pages became stale")
        }
        guard layerKinds == self.layerKinds, shared.layerPages.count == layerKinds.count else {
            throw CBv2KVError.backendIneligible(
                reason: "resident prefix layer layout does not match paged backend")
        }
        try cbv2ValidateResidentPagedPrefixPlan(
            plan, layerKinds: layerKinds, maxLength: maxLength,
            reserveFullSequenceTokens: pool.segmentGrant != nil)
        guard plan.matchedBoundary == shared.matchedTokens else {
            throw CBv2KVError.backendIneligible(
                reason: "resident prefix match does not match replay plan")
        }
        guard !plan.requiresExactWindowRestore else {
            // The resident index deliberately stores immutable full-attention
            // pages only. Exact sliding-window sidecars remain an SSD feature.
            throw CBv2KVError.backendIneligible(
                reason: "resident prefix cannot restore exact sliding-window sidecars")
        }

        let pageSize = pool.config.pageSize
        let storedThrough =
            plan.strategy == .frozenFullReplay ? plan.matchedBoundary : plan.replayStart
        guard storedThrough % pageSize == 0 else {
            // Sharing a partial frontier page would require copy-on-write.
            // Fail cold instead of silently rounding the replay boundary.
            throw CBv2KVError.backendIneligible(
                reason: "resident prefix restore boundary \(storedThrough) is not page-aligned")
        }
        let installedPageCount = storedThrough / pageSize
        var installed = [[PagedKVPageHandle]?](repeating: nil, count: layerKinds.count)
        var allHandles: [PagedKVPageHandle] = []
        var sawOwningFull = false

        // Validate every layer and generation before reserving or retaining a
        // single page. No partial row can escape on refusal.
        for (index, kind) in layerKinds.enumerated() {
            if kind.sharesKVWithLayer != nil {
                guard shared.layerPages[index] == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason: "resident prefix supplied pages for KV-shared layer \(index)")
                }
                continue
            }
            switch kind.attention {
            case .slidingWindow:
                guard shared.layerPages[index] == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason: "resident prefix supplied mutable sliding pages at layer \(index)")
                }
            case .full:
                guard let pages = shared.layerPages[index],
                    pages.count >= installedPageCount
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "resident prefix is missing full-layer pages at layer \(index)")
                }
                let prefix = Array(pages.prefix(installedPageCount))
                let expectedGroup = pool.groupKey(forLayer: index)
                guard prefix.allSatisfy({ $0.group == expectedGroup && pool.isValid($0) }) else {
                    throw CBv2KVError.backendIneligible(
                        reason: "resident prefix contains stale or foreign pages at layer \(index)")
                }
                installed[index] = prefix
                allHandles.append(contentsOf: prefix)
                sawOwningFull = true
            }
        }
        guard sawOwningFull else {
            throw CBv2KVError.backendIneligible(
                reason: "resident prefix has no storage-owning full layer")
        }

        // Keep the established conservative admission contract: every adopter
        // reserves its complete worst-case row even though the prefix pages are
        // shared. This under-realizes concurrency gains but preserves the
        // backend's nonthrowing-decode guarantee.
        let needs = pageNeeds(layerKinds: layerKinds, maxLength: maxLength)
        try pool.reserve(needs)
        do {
            try commitSlabs()
        } catch {
            pool.unreserve(needs)
            throw error
        }
        guard pool.retainPages(allHandles) else {
            pool.unreserve(needs)
            throw CBv2KVError.backendIneligible(
                reason: "resident prefix page generation changed during adoption")
        }

        // Everything below is nonthrowing. Each full row takes ownership of
        // the retains above; ordinary release decrements them transactionally.
        var states: [CBv2SequenceKV?] = []
        states.reserveCapacity(layerKinds.count)
        for (index, kind) in layerKinds.enumerated() {
            guard kind.sharesKVWithLayer == nil else {
                states.append(nil)
                continue
            }
            let reservedPages = PagedKVPool.pageDemand(
                kind: kind, maxLength: maxLength, config: pool.config)
            let row = PagedSequenceKV(
                pool: pool, kind: kind, groupKey: pool.groupKey(forLayer: index),
                maxLength: maxLength,
                reservedPages: reservedPages)
            switch kind.attention {
            case .slidingWindow:
                row.fastForward(to: plan.replayStart)
            case .full:
                let pages = installed[index]!
                let frozen = plan.strategy == .frozenFullReplay ? storedThrough : 0
                row.adoptSharedPages(
                    pages, storedThrough: storedThrough,
                    cursor: plan.replayStart, frozenThrough: frozen)
            }
            states.append(row)
        }
        return states
    }

    @discardableResult
    func publishResidentPrefixBlocks(
        state: [CBv2SequenceKV?], chainHashes: [Data], blockIndices: Range<Int>
    ) -> Int {
        residentPrefixIndex?.publish(
            state: state, chainHashes: chainHashes, blockIndices: blockIndices) ?? 0
    }

    public var residentPrefixCacheStats: CBv2PagedPrefixCacheStats? {
        residentPrefixIndex?.stats
    }
}
