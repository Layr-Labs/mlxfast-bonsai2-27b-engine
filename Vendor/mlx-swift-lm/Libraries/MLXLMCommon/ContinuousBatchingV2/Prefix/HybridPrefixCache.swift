import Foundation
import MLX

/// A bounded L1 for exact recurrent checkpoints plus their full-attention KV.
/// Staged, publishing and pinned entries share one hard budget. No disk state
/// or cross-model cache object is accepted; each instance belongs to one engine.
public final class CBv2HybridPrefixCache: @unchecked Sendable {
    private struct Scope: Hashable { let salt: String? }
    private struct Endpoint: Hashable {
        let entry: UInt64
        let checkpoint: Int
    }
    private final class Entry {
        let id: UInt64
        let scope: Scope
        let tokens: [Int]
        let checkpoints: [CBv2RecurrentCheckpoint]
        let kv: [(keys: MLXArray, values: MLXArray, offset: Int)?]
        let kvBytes: Int
        var lastUse: UInt64
        var pins = 0

        init(
            id: UInt64, scope: Scope, tokens: [Int],
            checkpoints: [CBv2RecurrentCheckpoint],
            kv: [(keys: MLXArray, values: MLXArray, offset: Int)?], kvBytes: Int,
            lastUse: UInt64
        ) {
            self.id = id
            self.scope = scope
            self.tokens = tokens
            self.checkpoints = checkpoints
            self.kv = kv
            self.kvBytes = kvBytes
            self.lastUse = lastUse
        }
    }

    public let config: CBv2HybridPrefixCacheConfig
    private let lock = NSLock()
    private let publicationQueue = DispatchQueue(label: "com.eigen.cbv2.hybrid-prefix")
    private var indexes: [Scope: CBv2TokenRadixIndex<Endpoint>] = [:]
    private var staged: [CBv2RequestID: [CBv2RecurrentCheckpoint]] = [:]
    private var inheritedFrom: [CBv2RequestID: UInt64] = [:]
    private var entries: [UInt64: Entry] = [:]
    private var pins: [UInt64: UInt64] = [:]
    private var nextID: UInt64 = 0
    private var clock: UInt64 = 0
    private var closed = false
    private var counters = CBv2HybridPrefixCacheStats()
    private var checkpointOwnership = CBv2HybridCheckpointOwnership()

    private var retainedBytesLocked: Int { counters.retainedBytes + checkpointOwnership.retainedBytes }
    private var byteLimit: Int
    private var publicationHandler: (@Sendable (CBv2RequestID, [Int]) -> Void)?

    public init(config: CBv2HybridPrefixCacheConfig) {
        self.config = config
        self.byteLimit = max(0, config.maximumBytes)
        closed = !config.isValid
    }

    public var stats: CBv2HybridPrefixCacheStats {
        lock.lock()
        defer { lock.unlock() }
        var result = counters
        result.stagedBytes += checkpointOwnership.stagedBytes
        result.publishingBytes += checkpointOwnership.publishingBytes
        result.residentBytes += checkpointOwnership.residentBytes
        return result
    }

    public func setPublicationHandler(_ handler: (@Sendable (CBv2RequestID, [Int]) -> Void)?) {
        lock.lock()
        publicationHandler = handler
        lock.unlock()
    }

    /// Outstanding GPU work and adoption pins remain charged after a shrink.
    /// The returned reservation keeps live-KV admission below the combined cap.
    func resizeReservation(to bytes: Int) -> Int {
        lock.lock()
        defer { lock.unlock() }
        byteLimit = min(max(0, bytes), max(0, config.maximumBytes))
        while retainedBytesLocked > byteLimit, let victim = evictionCandidateLocked() {
            removeLocked(victim)
        }
        return max(byteLimit, retainedBytesLocked)
    }

    /// Engine queue only. Reserve before building compact conv copies. Once
    /// returned, the engine must asyncEval these roots before drop or publish.
    func capture(
        requestID: CBv2RequestID, position: Int, chunkSize: Int,
        spec: CBv2RecurrentStateSpec, layers: [Int: CBv2RecurrentLayerState],
        assistant: (any CBv2MTPPrefixCheckpoint)? = nil
    ) -> [MLXArray] {
        guard position > 0, chunkSize > 1, position % chunkSize == 0 else { return [] }
        guard assistant == nil || assistant?.targetInputCount == position else { return [] }
        var bytes = assistant?.materializedBytes ?? 0
        for layer in spec.layers {
            guard let state = layers[layer.modelLayerIndex],
                let conv = state.conv, let ssm = state.ssm,
                conv.shape == layer.convShape, ssm.shape == layer.ssmShape,
                ssm.dtype == layer.ssmDType
            else { return [] }
            let (pair, pairOverflow) = conv.nbytes.addingReportingOverflow(ssm.nbytes)
            let (sum, sumOverflow) = bytes.addingReportingOverflow(pair)
            guard !pairOverflow, !sumOverflow else { return [] }
            bytes = sum
        }
        guard bytes > 0 else { return [] }
        lock.lock()
        defer { lock.unlock() }
        guard !closed,
            !(staged[requestID]?.contains { $0.position == position } ?? false)
        else { return [] }
        // Refusal must leave the previously useful endpoints intact. A new
        // copy and any retiring GPU copy must fit together before rolling.
        guard reserveLocked(bytes) else { return [] }
        if var checkpoints = staged[requestID], checkpoints.count >= config.maximumCheckpointsPerRequest {
            // Keep the first useful branch point and roll the newest endpoint
            // forward. The replaced copy stays charged until its GPU work ends.
            let replaced = checkpoints.removeLast()
            staged[requestID] = checkpoints
            checkpointOwnership.transfer(replaced, from: .staged, to: .publishing)
            let handoff = CBv2HybridHandoff(value: replaced)
            publicationQueue.async { [self] in
                handoff.consume { checkpoint in
                    eval(checkpoint.evaluationRoots)
                    lock.lock()
                    checkpointOwnership.release(checkpoint, from: .publishing)
                    lock.unlock()
                }
            }
        }
        var captured: [Int: CBv2RecurrentLayerState] = [:]
        for layer in spec.layers {
            let state = layers[layer.modelLayerIndex]!
            // Select copies bytes without arithmetic: preserve signed zeros
            // and denormals, while dropping the full convolution input view.
            let conv = MLX.where(MLXArray(true), state.conv!, state.conv!)
            captured[layer.modelLayerIndex] = .init(conv: conv, ssm: state.ssm)
        }
        let checkpoint = CBv2RecurrentCheckpoint(
            position: position, chunkSize: chunkSize, layers: captured, byteCount: bytes,
            assistant: assistant)
        staged[requestID, default: []].append(checkpoint)
        checkpointOwnership.retain(checkpoint, as: .staged)
        return checkpoint.evaluationRoots
    }

    /// Pins only a fully published endpoint. The radix path is exact tokens;
    /// a branch split without a checkpoint supplies no state. Always leaves a
    /// live prompt token for logits and preserves the donor's chunk geometry.
    func lookup(
        tokens: [Int], cacheSalt: String?, maximumChunkSize: Int
    ) -> CBv2HybridPrefixHit? {
        guard tokens.count > 1 else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return nil }
        let scope = Scope(salt: cacheSalt)
        let candidates = indexes[scope]?.matches(tokens: tokens.dropLast()) ?? []
        for match in candidates.reversed() {
            for endpoint in match.values {
                guard let entry = entries[endpoint.entry],
                    endpoint.checkpoint < entry.checkpoints.count
                else { continue }
                let checkpoint = entry.checkpoints[endpoint.checkpoint]
                guard checkpoint.chunkSize <= maximumChunkSize else { continue }
                let prefix = entry.kv.map { row -> (keys: MLXArray, values: MLXArray, offset: Int)? in
                    guard let row else { return nil }
                    return (
                        row.keys[.ellipsis, 0 ..< checkpoint.position, 0...],
                        row.values[.ellipsis, 0 ..< checkpoint.position, 0...],
                        checkpoint.position)
                }
                nextID += 1
                clock += 1
                entry.lastUse = clock
                entry.pins += 1
                pins[nextID] = entry.id
                counters.lookupMatches += 1
                return CBv2HybridPrefixHit(
                    pin: nextID, checkpoint: checkpoint,
                    kvPrefix: prefix, kvBackingBytes: entry.kvBytes)
            }
        }
        counters.misses += 1
        return nil
    }

    func candidate(tokens: [Int], cacheSalt: String?, maximumChunkSize: Int)
        -> CBv2ResidentPrefixCandidate?
    {
        guard tokens.count > 1 else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return nil }
        for match in (indexes[Scope(salt: cacheSalt)]?.matches(tokens: tokens.dropLast()) ?? []).reversed() {
            if match.values.contains(where: { endpoint in
                guard let entry = entries[endpoint.entry] else { return false }
                return entry.checkpoints[endpoint.checkpoint].chunkSize <= maximumChunkSize
            }) {
                return .init(matchedTokens: match.position, prefillTokensSaved: match.position)
            }
        }
        return nil
    }

    /// Retain the first branch point and the actual adopted endpoint. A short
    /// repeat must not replace its newest reusable boundary with an older one.
    func inheritCheckpoint(
        pin: UInt64, checkpoint adopted: CBv2RecurrentCheckpoint, requestID: CBv2RequestID
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard !closed, staged[requestID] == nil,
            let entryID = pins[pin], let entry = entries[entryID],
            entry.checkpoints.contains(where: { $0.storageID == adopted.storageID }),
            let first = entry.checkpoints.first
        else { return }
        let checkpoints = config.maximumCheckpointsPerRequest > 1 && first.position < adopted.position
            ? [first, adopted] : [adopted]
        staged[requestID] = checkpoints
        inheritedFrom[requestID] = entryID
        for checkpoint in checkpoints { checkpointOwnership.retain(checkpoint, as: .staged) }
    }

    func endAdoption(pin: UInt64, tokensSaved: Int = 0) {
        lock.lock()
        defer { lock.unlock() }
        guard let entryID = pins.removeValue(forKey: pin), let entry = entries[entryID] else { return }
        entry.pins -= 1
        if tokensSaved > 0 {
            counters.adoptions += 1
            counters.tokensSaved += tokensSaved
        }
        if closed && entry.pins == 0 { removeLocked(entry) }
        while retainedBytesLocked > byteLimit, let victim = evictionCandidateLocked() {
            removeLocked(victim)
        }
    }

    func hasStagedCheckpoints(requestID: CBv2RequestID) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return staged[requestID]?.isEmpty == false
    }

    /// Called after engine asyncEval detached every captured root. KV belongs
    /// to a retired request, so the publication queue is its sole evaluator.
    /// Completion runs after insertion or refusal; the engine must keep source
    /// KV reserved until then, including while an oversized donor is compacted.
    func publish(
        requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
        receiptID: CBv2RequestID? = nil,
        kv: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        backingBytes: Int,
        completion: @escaping @Sendable () -> Void
    ) {
        lock.lock()
        var checkpoints = staged.removeValue(forKey: requestID) ?? []
        let highest = checkpoints.last?.position ?? 0
        if let positions = refreshInheritedLocked(
            requestID: requestID, tokens: tokens, cacheSalt: cacheSalt, checkpoints: checkpoints)
        {
            // Drop local aliases before external callbacks can evict the entry.
            checkpoints.removeAll(keepingCapacity: false)
            let handler = publicationHandler
            lock.unlock()
            if let receiptID { handler?(receiptID, positions) }
            completion()
            return
        }
        for checkpoint in checkpoints {
            checkpointOwnership.transfer(checkpoint, from: .staged, to: .publishing)
        }
        let eligible = !closed && highest > 0 && highest <= tokens.count && backingBytes > 0
        let accepted = eligible && reserveLocked(backingBytes, recordRefusal: false)
        if accepted { counters.publishingBytes += backingBytes }
        let handoff = CBv2HybridHandoff(value: CBv2HybridPrefixPublication(
            checkpoints: checkpoints, kv: eligible ? kv : [],
            kvBytes: accepted ? backingBytes : 0, mayCompact: eligible && !accepted))
        checkpoints.removeAll(keepingCapacity: false)
        lock.unlock()

        publicationQueue.async { [self] in
            let (publishedPositions, handler) = handoff.consume { publication in
                // Captured roots were already detached on the engine thread;
                // this waits for their events without racing a live forward graph.
                eval(publication.checkpoints.flatMap(\.evaluationRoots))
                if publication.mayCompact {
                    lock.lock()
                    prepareCompactionLocked(publication)
                    lock.unlock()
                }
                if publication.kvBytes > 0 {
                    eval(publication.kv.flatMap { $0.map { [$0.keys, $0.values] } ?? [] })
                }
                lock.lock()
                var publishedPositions: [Int] = []
                let fitsCurrentBudget = retainedBytesLocked <= byteLimit
                counters.publishingBytes -= publication.kvBytes
                for checkpoint in publication.checkpoints {
                    checkpointOwnership.release(checkpoint, from: .publishing)
                }
                if publication.kvBytes > 0 && !closed && fitsCurrentBudget {
                    while entries.count >= config.maximumEntries,
                        let victim = evictionCandidateLocked()
                    {
                        removeLocked(victim)
                    }
                    if entries.count < config.maximumEntries {
                        nextID += 1
                        clock += 1
                        let entry = Entry(
                            id: nextID, scope: Scope(salt: cacheSalt),
                            tokens: Array(tokens.prefix(publication.checkpoints.last!.position)),
                            checkpoints: publication.checkpoints,
                            kv: publication.kv, kvBytes: publication.kvBytes, lastUse: clock)
                        entries[entry.id] = entry
                        counters.residentBytes += entry.kvBytes
                        for checkpoint in entry.checkpoints { checkpointOwnership.retain(checkpoint, as: .resident) }
                        counters.entries += 1
                        counters.checkpoints += entry.checkpoints.count
                        var index = indexes[entry.scope] ?? CBv2TokenRadixIndex<Endpoint>()
                        for (offset, checkpoint) in entry.checkpoints.enumerated() {
                            index.insert(
                                tokens: entry.tokens.prefix(checkpoint.position),
                                value: Endpoint(entry: entry.id, checkpoint: offset))
                        }
                        indexes[entry.scope] = index
                        publishedPositions = entry.checkpoints.map(\.position)
                    } else {
                        counters.capacityRefusals += 1
                    }
                }
                let handler = publicationHandler
                // Release uncharged aliases before another thread can reserve
                // their capacity. A successful entry owns its copies now.
                publication.checkpoints.removeAll(keepingCapacity: false)
                publication.kv.removeAll(keepingCapacity: false)
                lock.unlock()
                return (publishedPositions, handler)
            }
            // The queue handoff releases its roots before invoking external
            // callbacks; an eviction during a delayed callback cannot retain
            // uncharged checkpoint or KV arrays in this closure.
            if !publishedPositions.isEmpty, let receiptID { handler?(receiptID, publishedPositions) }
            completion()
        }
    }

    /// Roots have finished evaluating before this fallback runs. Retire only
    /// actual later checkpoints until a compact destination fits; never slice
    /// recurrent state or uncharge a pending GPU copy to make room.
    private func prepareCompactionLocked(_ publication: CBv2HybridPrefixPublication) {
        guard !closed else { return }
        while let position = publication.checkpoints.last?.position {
            if let bytes = publication.compactedBytes(through: position),
                reserveLocked(bytes, recordRefusal: false)
            {
                counters.publishingBytes += bytes
                publication.kvBytes = bytes
                publication.compactKV(through: position)
                counters.kvCompactions += 1
                counters.kvCompactionBytes += bytes
                return
            }
            releaseLastPublishingCheckpointLocked(publication)
        }
        counters.capacityRefusals += 1
    }

    private func releaseLastPublishingCheckpointLocked(_ publication: CBv2HybridPrefixPublication) {
        // The helper's local alias dies before another reservation can count
        // the removed checkpoint as available capacity.
        let checkpoint = publication.checkpoints.removeLast()
        checkpointOwnership.release(checkpoint, from: .publishing)
    }

    /// No new checkpoint: refresh an existing donor instead of duplicating its
    /// full KV on every repeat. Array aliases stay inside this locked helper.
    private func refreshInheritedLocked(
        requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
        checkpoints: [CBv2RecurrentCheckpoint]
    ) -> [Int]? {
        let source = inheritedFrom.removeValue(forKey: requestID)
        guard !closed, let highest = checkpoints.last?.position,
            let source, let entry = entries[source],
            entry.scope == Scope(salt: cacheSalt),
            tokens.starts(with: entry.tokens.prefix(highest)),
            checkpoints.allSatisfy({ checkpoint in
                entry.checkpoints.contains { $0.storageID == checkpoint.storageID }
            })
        else { return nil }
        clock += 1
        entry.lastUse = clock
        for checkpoint in checkpoints { checkpointOwnership.release(checkpoint, from: .staged) }
        return checkpoints.map(\.position)
    }

    /// Cancellation/preemption never publishes. Keep the charge until already
    /// submitted copies finish; dropping host references alone does not retire
    /// their GPU buffers.
    @discardableResult
    func dropStaged(
        requestID: CBv2RequestID, completion: (@Sendable () -> Void)? = nil
    ) -> Bool {
        lock.lock()
        inheritedFrom.removeValue(forKey: requestID)
        let checkpoints = staged.removeValue(forKey: requestID) ?? []
        for checkpoint in checkpoints {
            checkpointOwnership.transfer(checkpoint, from: .staged, to: .publishing)
        }
        lock.unlock()
        guard !checkpoints.isEmpty else { return false }
        let handoff = CBv2HybridHandoff(value: checkpoints)
        publicationQueue.async { [self] in
            handoff.consume { checkpoints in
                eval(checkpoints.flatMap(\.evaluationRoots))
                lock.lock()
                for checkpoint in checkpoints { checkpointOwnership.release(checkpoint, from: .publishing) }
                lock.unlock()
            }
            completion?()
        }
        return true
    }

    public func close() {
        lock.lock()
        closed = true
        let stagedIDs = Array(staged.keys)
        for entry in Array(entries.values) where entry.pins == 0 { removeLocked(entry) }
        lock.unlock()
        for id in stagedIDs { dropStaged(requestID: id) }
    }

    private func reserveLocked(_ bytes: Int, recordRefusal: Bool = true) -> Bool {
        guard bytes > 0, bytes <= byteLimit else {
            if recordRefusal { counters.capacityRefusals += 1 }
            return false
        }
        while retainedBytesLocked > byteLimit - bytes {
            guard let victim = evictionCandidateLocked() else {
                if recordRefusal { counters.capacityRefusals += 1 }
                return false
            }
            removeLocked(victim)
        }
        return true
    }

    private func evictionCandidateLocked() -> Entry? {
        entries.values.filter { $0.pins == 0 }.min { $0.lastUse < $1.lastUse }
    }

    private func removeLocked(_ entry: Entry) {
        guard entries.removeValue(forKey: entry.id) != nil else { return }
        if var index = indexes[entry.scope] {
            for (offset, checkpoint) in entry.checkpoints.enumerated() {
                index.remove(
                    tokens: entry.tokens.prefix(checkpoint.position),
                    value: Endpoint(entry: entry.id, checkpoint: offset))
            }
            indexes[entry.scope] = index.isEmpty ? nil : index
        }
        counters.residentBytes -= entry.kvBytes
        for checkpoint in entry.checkpoints { checkpointOwnership.release(checkpoint, from: .resident) }
        counters.entries -= 1
        counters.checkpoints -= entry.checkpoints.count
        counters.evictions += 1
    }
}

private final class CBv2HybridHandoff<Value>: @unchecked Sendable {
    private var value: Value?
    init(value: Value) { self.value = value }

    /// Single publication-queue consumer; discard references before returning
    /// to callbacks that may outlive the owning bank entry.
    func consume<Result>(_ body: (Value) -> Result) -> Result {
        defer { value = nil }
        return body(value!)
    }
}
