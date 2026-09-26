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

// MARK: - Fused QK Rotary Position (L3 prefill elementwise fusion)

/// Shared shape guard for the fused Q/K RoPE pair: post-transpose
/// `[B, H, L, D]` queries and `[B, KV, L, D]` keys with `qHeads` query
/// heads, so one concatenation along the head axis (1) followed by one
/// split at `qHeads` round-trips exactly.
private func fusedQKRotaryShapesMatch(
    queries: MLXArray, keys: MLXArray, qHeads: Int
) -> Bool {
    queries.ndim == 4 && keys.ndim == 4
        && qHeads > 0 && qHeads < queries.dim(1)
        && keys.dim(0) == queries.dim(0)
        && keys.dim(1) == queries.dim(1) - qHeads
        && keys.dim(2) == queries.dim(2)
        && keys.dim(3) == queries.dim(3)
}

/// Fuse one Q/K RoPE pair into a single kernel call: concatenate queries
/// and keys along the head axis, apply `rope` once with the shared `offset`,
/// split back. Q and K read the same positions (offsets don't advance
/// mid-step), so the angle table (`sin`/`cos` over positions x frequencies)
/// is computed once and broadcast over H+KV heads instead of twice. The
/// rotation itself is per-element over the last dimension, so the fused
/// pair is bit-identical to the two separate calls; nil on shape mismatch
/// keeps the caller's separate-call fallback.
///
/// - Parameters:
///   - rope: A RoPE layer conforming to both `OffsetLayer` and `ArrayOffsetLayer`.
///   - queries: Post-transpose `[B, H, L, D]` queries.
///   - keys: Post-transpose `[B, KV, L, D]` keys.
///   - qHeads: Query head count (the split point).
///   - offset: Shared per-step position offsets (scalar or `[B]`).
/// - Returns: The `(queries, keys)` pair with rotary embeddings applied,
///   or `nil` when the shapes cannot fuse.
public func fusedQKRotaryPosition<R: RoPELayer>(
    _ rope: R, queries: MLXArray, keys: MLXArray, qHeads: Int,
    offset: MLXArray
) -> (MLXArray, MLXArray)? {
    guard fusedQKRotaryShapesMatch(queries: queries, keys: keys, qHeads: qHeads)
    else { return nil }
    let fused = rope(concatenated([queries, keys], axis: 1), offset: offset)
    let parts = fused.split(indices: [qHeads], axis: 1)
    guard parts.count == 2 else { return nil }
    return (parts[0], parts[1])
}

/// Scalar-offset variant of ``fusedQKRotaryPosition(_:queries:keys:qHeads:offset:)``
/// for scalar/nil caches. Same single-call fusion, same `nil` fallback.
public func fusedQKRotaryPosition<R: RoPELayer>(
    _ rope: R, queries: MLXArray, keys: MLXArray, qHeads: Int,
    offset: Int
) -> (MLXArray, MLXArray)? {
    guard fusedQKRotaryShapesMatch(queries: queries, keys: keys, qHeads: qHeads)
    else { return nil }
    let fused = rope(concatenated([queries, keys], axis: 1), offset: offset)
    let parts = fused.split(indices: [qHeads], axis: 1)
    guard parts.count == 2 else { return nil }
    return (parts[0], parts[1])
}

/// Fused Q/K entry point mirroring ``applyRotaryPosition(_:to:cache:sharedOffset:)``:
/// resolves the offset exactly as the separate path does (`sharedOffset`,
/// then the per-call `graphOffsetArray` snapshot, then the scalar `offset`),
/// then serves the pair from one RoPE call when `enabled` and the shapes
/// fuse. Never fails: any guard miss returns the two separate calls.
public func applyRotaryPositionQK<R: RoPELayer>(
    _ rope: R, queries: MLXArray, keys: MLXArray, qHeads: Int,
    cache: KVCache?, sharedOffset: MLXArray? = nil, enabled: Bool = true
) -> (MLXArray, MLXArray) {
    if enabled {
        if let sharedOffset,
            let fused = fusedQKRotaryPosition(
                rope, queries: queries, keys: keys, qHeads: qHeads,
                offset: sharedOffset)
        {
            return fused
        } else if sharedOffset == nil,
            let offsetArray = graphOffsetArray(for: cache),
            let fused = fusedQKRotaryPosition(
                rope, queries: queries, keys: keys, qHeads: qHeads,
                offset: offsetArray)
        {
            return fused
        } else if sharedOffset == nil, graphOffsetArray(for: cache) == nil,
            let fused = fusedQKRotaryPosition(
                rope, queries: queries, keys: keys, qHeads: qHeads,
                offset: cache?.offset ?? 0)
        {
            return fused
        }
    }
    if let sharedOffset {
        return (rope(queries, offset: sharedOffset), rope(keys, offset: sharedOffset))
    }
    if let offsetArray = graphOffsetArray(for: cache) {
        return (rope(queries, offset: offsetArray), rope(keys, offset: offsetArray))
    }
    let scalar = cache?.offset ?? 0
    return (rope(queries, offset: scalar), rope(keys, offset: scalar))
}

/// Array-offset fused Q/K entry point for callers that already hold the
/// shared per-step offsets (e.g. CBv2 `sharedOffsets ?? cache.positionOffsets + 0`).
/// One RoPE call when `enabled` and the shapes fuse; the two separate
/// calls otherwise.
public func applyRotaryPositionQK<R: RoPELayer>(
    _ rope: R, queries: MLXArray, keys: MLXArray, qHeads: Int,
    offset: MLXArray, enabled: Bool = true
) -> (MLXArray, MLXArray) {
    if enabled,
        let fused = fusedQKRotaryPosition(
            rope, queries: queries, keys: keys, qHeads: qHeads, offset: offset)
    {
        return fused
    }
    return (rope(queries, offset: offset), rope(keys, offset: offset))
}
