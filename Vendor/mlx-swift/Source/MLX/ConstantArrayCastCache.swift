// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

/// Internal, bounded reuse of an exact widening constant conversion.
/// BF16 is the default; a model-specific caller may explicitly opt into FP16.
///
/// The retained context snapshot pins the backing descriptor, not the Swift
/// object. `Module.update`, indexed assignment, and compile-state replacement
/// can change a parameter's descriptor while preserving its Swift identity.
/// Retaining the old descriptor also prevents allocator address reuse (ABA).
/// This object is deliberately neither an MLXArray nor a Module: reflecting
/// its owner must not add cached constants to the model's parameter tree.
@_spi(QuantizedConstantCache)
public final class ConstantArrayCastCache {
    private struct Entry {
        let identity: UInt
        let stream: StreamOrDevice
        let sourceSnapshot: MLXArray
        let converted: MLXArray
    }

    private let lock = NSLock()
    private var entry: Entry?
    private let enabled: Bool

    /// Process-wide rollback, latched before inference. An explicit false
    /// value restores the original per-operation casts without changing math.
    private static let defaultEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLX_QUANTIZED_CONSTANT_CACHE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    public init(enabled: Bool? = nil) {
        self.enabled = enabled ?? Self.defaultEnabled
    }

    /// Return nil to keep the caller's original operation ordering during
    /// tracing or on any other dtype path, including packed MXFP4 U8 scales.
    public func cachedCast(
        _ source: MLXArray, to dtype: DType, allowFloat16: Bool = false
    ) -> MLXArray? {
        guard enabled, dtype == .float32,
            source.dtype == .bfloat16 || (allowFloat16 && source.dtype == .float16)
        else { return nil }
        return lock.withLock {
            var identity: UInt = 0
            var canCache = false
            guard _mlx_array_constant_cache_identity(&identity, &canCache, source.ctx) == 0,
                canCache
            else {
                // Do not retain a graph produced by compile/grad/vmap tracing.
                return nil
            }
            // Cast primitives belong to the execution stream selected at graph
            // construction. Reuse only within that same stream (and device).
            let stream = StreamOrDevice.default
            if let entry, entry.identity == identity, entry.stream == stream {
                return entry.converted
            }
            let snapshot = source.copyContext()
            let converted = snapshot.asType(.float32, stream: stream)
            entry = Entry(
                identity: identity, stream: stream, sourceSnapshot: snapshot, converted: converted)
            return converted
        }
    }

    public func clear() {
        lock.withLock { entry = nil }
    }
}
