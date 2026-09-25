// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - BatchPositionedKVCache

/// Protocol for KV caches that expose per-sequence RoPE offsets.
///
/// This is a forward-compatible hook for batched caches. Current scalar-cache
/// code paths continue using `KVCache.offset`.
public protocol BatchPositionedKVCache: KVCache {
    /// Per-sequence RoPE offsets with shape `[B]`.
    var batchOffset: MLXArray { get }
}

// MARK: - graphOffsetArray Helper

/// Returns a graph-visible cache offset when the cache exposes one.
///
/// `KVCache.offset` is an `Int` API for compatibility with upstream model
/// ports. Compile-safe caches keep the offset as an `MLXArray` so the value
/// can flow through `compile()` without an `.item()` readback. Model-side
/// helpers that need positions outside `applyRotaryPosition` should call this
/// before falling back to `cache.offset`.
public func graphOffsetArray(for cache: KVCache?) -> MLXArray? {
    // Snapshot with `+ 0` so cache.update() advancing offsetArray
    // doesn't shift the caller's RoPE position. Without this, the query
    // gets a position one step ahead of the keys in compiled decode.
    if let compilableRot = cache as? CompilableRotatingKVCache {
        return compilableRot.offsetArray + 0
    }
    if let compilable = cache as? CompilableKVCache {
        return compilable.offsetArray + 0
    }
    if let batchCache = cache as? BatchPositionedKVCache {
        return batchCache.batchOffset + 0
    }
    return nil
}

// MARK: - applyRotaryPosition Helper

/// Apply rotary position embeddings, using the cache offset when available.
///
/// This function enables models to use a single call site instead of
/// repeating conditional offset handling:
/// ```swift
/// queries = applyRotaryPosition(rope, to: queries, cache: cache)
/// keys = applyRotaryPosition(rope, to: keys, cache: cache)
/// ```
///
/// When the cache exposes an `offsetArray` (e.g. `CompilableKVCache`), the
/// offset is passed as an `MLXArray` so the compile tracer can track it
/// through the graph without triggering a synchronous GPU readback. For all
/// other cache types, the standard `Int`-based offset path is used.
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - x: The input tensor to apply RoPE to.
///   - cache: The KV cache (determines scalar or per-sequence offset), or `nil`
///     for offset 0.
///   - sharedOffset: Optional per-step snapshot from `graphOffsetArray(for:)`.
///     When non-`nil`, it is used directly and no per-call `+ 0` snapshot
///     dispatch is issued. Take the snapshot once per step — before any
///     `cache.update()` — and thread the same array through every Q/K RoPE
///     call across layers; offsets don't advance mid-step, so one snapshot
///     covers ~2 calls/layer. When `nil` (default), falls back to a per-call
///     `graphOffsetArray(for: cache)` snapshot, preserving existing behavior.
/// - Returns: The input with rotary positional encoding applied.
public func applyRotaryPosition<R: RoPELayer>(
    _ rope: R, to x: MLXArray, cache: KVCache?, sharedOffset: MLXArray? = nil
)
    -> MLXArray
{
    if let sharedOffset {
        return rope(x, offset: sharedOffset)
    }
    if let offsetArray = graphOffsetArray(for: cache) {
        return rope(x, offset: offsetArray)
    }
    return rope(x, offset: cache?.offset ?? 0)
}
