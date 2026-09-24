import Foundation
import MLX

/// When native paged backing is evaluated into Metal residency.
public enum PagedKVSlabCommitment: String, Sendable, Equatable, CaseIterable {
    /// Eagerly allocate the complete configured pool. Intended for explicit
    /// profiling; idle backing counts against co-resident model headroom.
    case atConstruction

    /// Keep idle construction free of KV payload. A fixed pool commits its
    /// slabs on first admission; segmented pools commit only enough additional
    /// backing to honor outstanding reservations on each admission.
    case atFirstAdmission
}

extension PagedKVBackend {
    /// Actual native backing, including poison and reserved-but-untouched
    /// pages. Segmented ownership can shrink after request release. Fixed
    /// ownership persists and may be partial after an allocation failure.
    public var bytesWired: Int { pool.bytesMaterialized }

    /// Guarantee physical backing before publishing any request row. Fixed
    /// pools remember per-slab progress; segmented candidates stay private
    /// until every group succeeds, then publish together.
    ///
    /// Allocation runs under scoped MLX error handling. Refusal becomes the
    /// engine's retryable capacity error; callers unwind their page charge.
    /// Memory.memoryLimit is a throttle, not an allocator admission ceiling,
    /// so it is used only to describe an actual failure, never to predict one.
    /// Thread-affinity is the pool's: the engine queue.
    public func commitSlabs() throws {
        if pool.config.segmentSizeBytes != nil {
            do {
                try pool.materializeReservedSegments()
            } catch {
                throw CBv2KVError.capacityExhausted(
                    needed: pool.bytesReserved,
                    available: max(0, Memory.memoryLimit - Memory.activeMemory))
            }
            return
        }
        guard !slabsAreWired else { return }
        do {
            try pool.materializeSlabs()
        } catch {
            // `needed` is what is STILL missing after the partial progress
            // this attempt made; `available` is a diagnostic-only reading
            // of the throttle limit's remaining headroom (it is NOT what
            // admission decided on — the failed attempt is).
            throw CBv2KVError.capacityExhausted(
                needed: pool.bytesUnmaterialized,
                available: max(0, Memory.memoryLimit - Memory.activeMemory))
        }
        markSlabsWired()
    }
}
