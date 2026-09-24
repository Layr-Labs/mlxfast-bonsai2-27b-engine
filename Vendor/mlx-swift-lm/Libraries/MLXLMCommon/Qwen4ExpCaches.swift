//
//  Qwen4ExpCaches.swift
//  mlx-swift-lm
//
//  Caches for the `qwen4_exp` / `qwen4_exp_text` model family
//  (Qwen 3.8 Flash-Next). Derived from the cache classes of the MIT-licensed
//  mlx-lm reference port (ml-explore/mlx-lm PR #1788, `mlx_lm/models/qwen4_exp.py`,
//  head c961f839): `_AttnCache` / `_IndexerCache` and `_LayerCache`.
//
//  These live in MLXLMCommon rather than next to the model because
//  `KVCacheSimple` and `ArraysCache` are `public` (not `open`) and can only be
//  subclassed inside their own module.
//

import Foundation
import MLX

/// KV cache for one Qwen4-Exp full-attention layer.
///
/// A full-attention layer runs a QSA indexer beside the ordinary attention. The
/// indexer keeps its own key tape: one raw key vector per token (never pooled,
/// never rotated), which the indexer pools into blocks on each call. The tape
/// has to live with the KV tape so that a trim, a copy or a state round-trip
/// keeps the two in step.
///
/// The indexer tape is EXACT — it holds `offset` rows and no reserve — while
/// `KVCacheSimple` over-allocates its KV buffers and tracks the live length in
/// `offset`. `trim` therefore slices the indexer tape and only moves the KV
/// offset.
public final class Qwen4ExpAttentionCache: KVCacheSimple {

    /// Raw indexer keys, shape `[B, offset, indexerHeadDim]`, or `nil` before
    /// the first update.
    public private(set) var indexerKeys: MLXArray?

    public override init() {
        super.init()
    }

    /// Append `keys` to the indexer tape and return the whole tape.
    public func updateIndexer(keys: MLXArray) -> MLXArray {
        let updated: MLXArray
        if let indexerKeys {
            updated = concatenated([indexerKeys, keys], axis: 1)
        } else {
            updated = keys
        }
        self.indexerKeys = updated
        return updated
    }

    public override func innerState() -> [MLXArray] {
        var inner = super.innerState()
        if let indexerKeys {
            inner.append(indexerKeys)
        }
        return inner
    }

    /// The indexer tape is serialized as the last entry. An empty array stands
    /// for "no tape yet", because `nil` is not serializable.
    public override var state: [MLXArray] {
        get {
            let kv = super.state
            let tape = indexerKeys ?? MLXArray([Float]())
            return kv.isEmpty ? [tape] : kv + [tape]
        }
        set {
            guard let tape = newValue.last else {
                fatalError("Qwen4ExpAttentionCache state must carry the indexer tape")
            }
            indexerKeys = tape.size > 0 ? tape : nil
            let kv = Array(newValue.dropLast())
            if !kv.isEmpty {
                super.state = kv
            }
        }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = super.trim(n)
        if let indexerKeys, indexerKeys.dim(1) > offset {
            self.indexerKeys = indexerKeys[0..., ..<offset, 0...]
        }
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = Qwen4ExpAttentionCache()
        new.step = self.step
        let s = self.state
        if s.count > 1 {
            new.state = s.map { $0[.ellipsis] }
        } else if let indexerKeys {
            new.indexerKeys = indexerKeys[.ellipsis]
        }
        return new
    }
}

/// Cache for one Qwen4-Exp linear-attention (gated deltanet) layer.
///
/// Four slots, because a linear layer can also carry the model's single PLE
/// layer:
///
/// - 0: gated deltanet short-convolution state
/// - 1: gated deltanet recurrent (SSM) state
/// - 2: PLE short-convolution state
/// - 3: the last `ngram_size - 1` token ids, the n-gram hash history
///
/// Slots 2 and 3 stay `nil` on every layer except the PLE layer.
public final class Qwen4ExpLayerCache: ArraysCache {

    public static let deltaConvSlot = 0
    public static let deltaStateSlot = 1
    public static let pleConvSlot = 2
    public static let ngramHistorySlot = 3

    public init(leftPadding: [Int]? = nil) {
        super.init(size: 4, leftPadding: leftPadding)
    }

    public override func copy() -> any KVCache {
        let new = Qwen4ExpLayerCache()
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        return new
    }

    public override func extract(_ idx: Int) -> ArraysCache {
        let extracted = Qwen4ExpLayerCache()
        extracted.state = state.map { $0[idx ..< (idx + 1)] }
        return extracted
    }
}

// MARK: - Speculative rollback

/// One layer's cache state at a draft boundary.
///
/// WHY A SNAPSHOT AND NOT A TRIM. A speculative round feeds the target tokens
/// it may have to take back. On the 12 full-attention layers that is a trim:
/// the key-value tape is append-only and `offset` says how much of it is real.
/// On the 36 gated-deltanet layers it is NOT. Their state is a RECURRENCE --
/// each forward overwrites the previous state -- so there is no row to drop and
/// no offset to move back. The state after the accepted prefix simply is not in
/// the cache any more once the rejected tokens have been folded in.
///
/// So the stack is snapshotted BEFORE the verify forward, and a rollback
/// restores that snapshot and replays the accepted tokens. The replay is one
/// forward of the accepted width, which is at most the draft depth, and it is
/// the only way to reach the recurrent state of a prefix that was overwritten.
public enum Qwen4ExpCacheSnapshot {
    /// A full-attention layer: how much of the tape was real.
    case attention(offset: Int)
    /// A linear-attention layer: every slot, plus the position count.
    case layer(slots: [MLXArray?], offset: Int)
}

extension Qwen4ExpAttentionCache {
    public func snapshot() -> Qwen4ExpCacheSnapshot { .attention(offset: offset) }

    /// Restore to `offset`. The key-value buffers keep their bytes past the
    /// offset and the next update overwrites them in place, so only the offset
    /// and the EXACT indexer tape have to move.
    public func restore(toOffset target: Int) {
        precondition(
            target >= 0 && target <= offset,
            "Qwen4ExpAttentionCache: cannot restore forward, from \(offset) to \(target)")
        offset = target
        if let indexerKeys, indexerKeys.dim(1) > target {
            self.indexerKeys = target == 0 ? nil : indexerKeys[0..., ..<target, 0...]
        }
    }
}

extension Qwen4ExpLayerCache {
    public func snapshot() -> Qwen4ExpCacheSnapshot {
        // The slot arrays are REPLACED, never written in place, by every writer
        // in this model (the deltanet's conv and state slots, the per-layer
        // embedding's conv slot, and the n-gram history), so holding the
        // references is a snapshot.
        .layer(slots: (0 ..< 4).map { self[$0] }, offset: offset)
    }

    public func restore(slots: [MLXArray?], offset target: Int) {
        precondition(slots.count == 4, "Qwen4ExpLayerCache has four slots")
        for index in 0 ..< 4 { self[index] = slots[index] }
        offset = target
    }
}

/// Snapshot a whole hybrid stack at a draft boundary.
public func qwen4ExpCheckpointCaches(_ caches: [KVCache]) -> [Qwen4ExpCacheSnapshot] {
    caches.map { cache in
        if let attention = cache as? Qwen4ExpAttentionCache {
            return attention.snapshot()
        }
        if let layer = cache as? Qwen4ExpLayerCache {
            return layer.snapshot()
        }
        preconditionFailure(
            "Qwen4Exp speculative rollback needs a Qwen4Exp cache stack, got "
                + "\(type(of: cache))")
    }
}

/// Restore a whole hybrid stack to a snapshot.
///
/// This returns the caches to the state they were in when the snapshot was
/// taken. It does NOT by itself put them at the accepted prefix: the caller
/// replays the accepted tokens after restoring. Splitting it that way keeps
/// the one thing that cannot be derived -- the pre-forward state -- separate
/// from the thing that can.
public func qwen4ExpRestoreCaches(
    _ caches: [KVCache], to snapshots: [Qwen4ExpCacheSnapshot]
) {
    precondition(
        caches.count == snapshots.count,
        "Qwen4Exp rollback: \(caches.count) caches against \(snapshots.count) snapshots")
    for (cache, snapshot) in zip(caches, snapshots) {
        switch snapshot {
        case .attention(let offset):
            guard let attention = cache as? Qwen4ExpAttentionCache else {
                preconditionFailure(
                    "Qwen4Exp rollback: an attention snapshot met \(type(of: cache))")
            }
            attention.restore(toOffset: offset)
        case .layer(let slots, let offset):
            guard let layer = cache as? Qwen4ExpLayerCache else {
                preconditionFailure(
                    "Qwen4Exp rollback: a layer snapshot met \(type(of: cache))")
            }
            layer.restore(slots: slots, offset: offset)
        }
    }
}
