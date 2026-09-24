// Host-only, non-overlapping committed-token accounting for stateless MTP.
// Every seed advances the clock and owns both its elapsed time and outputs;
// the subsequent verify claims that pair exactly once. Continued verification
// measures commit-to-commit wall time, including scheduling and overlapped draft
// construction, rather than the narrower isolated target-forward latency.

struct CBv2MTPCommittedGoodputClock {
    struct Sample: Equatable {
        let wallTimeNanos: UInt64
        let committedTokens: Int
    }

    private var previousRows: Set<CBv2RequestID> = []
    private var previousNanos: UInt64?
    private var pendingSeed: Sample?

    mutating func reset() {
        previousRows.removeAll(keepingCapacity: true)
        previousNanos = nil
        pendingSeed = nil
    }

    mutating func observe(
        measurement: CBv2MTPStepMeasurement,
        completedAtNanos: UInt64,
        isolatedWallTimeNanos: UInt64,
        rowIDs: [CBv2RequestID],
        committedTokens: Int
    ) -> Sample? {
        let rows = Set(rowIDs)
        let eligible = measurement.costEligible && completedAtNanos > 0
            && isolatedWallTimeNanos > 0 && !rows.isEmpty
            && rows.count == rowIDs.count && committedTokens > 0
            && CBv2MTPDepthController.decodeRowBucket(rows.count)
                == measurement.decision.decodeRowBucket
            && (measurement.actualDepth == 0 || !measurement.chained)
            && (measurement.seedOnly || measurement.actualDepth == measurement.decision.depth)
        guard eligible else { reset(); return nil }
        let sameRows = rows == previousRows
        let interval: UInt64
        if sameRows, let previousNanos, completedAtNanos > previousNanos {
            interval = completedAtNanos - previousNanos
        } else {
            interval = isolatedWallTimeNanos
            pendingSeed = nil
        }
        previousRows = rows
        previousNanos = completedAtNanos
        if measurement.seedOnly {
            let previous = pendingSeed
            pendingSeed = Sample(
                wallTimeNanos: interval &+ (previous?.wallTimeNanos ?? 0),
                committedTokens: committedTokens + (previous?.committedTokens ?? 0))
            return nil
        }
        guard measurement.actualDepth > 0 else {
            pendingSeed = nil
            return nil
        }
        defer { pendingSeed = nil }
        return Sample(
            wallTimeNanos: interval &+ (pendingSeed?.wallTimeNanos ?? 0),
            committedTokens: committedTokens + (pendingSeed?.committedTokens ?? 0))
    }
}
