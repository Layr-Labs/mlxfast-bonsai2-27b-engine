// Host-only steady decode intervals. Launch-to-readback timings overlap when
// ordinary decode is pipelined; consecutive commit timestamps do not.

struct CBv2MTPCommittedDecodeClock {
    private var previousRows: [CBv2RequestID] = []
    private var previousNanos: UInt64?

    mutating func observe(
        completedAtNanos: UInt64, rowIDs: [CBv2RequestID], eligible: Bool
    ) -> UInt64? {
        guard eligible, !rowIDs.isEmpty, completedAtNanos > 0 else {
            previousRows.removeAll(keepingCapacity: true)
            previousNanos = nil
            return nil
        }
        let start = previousNanos
        let sameRows = previousRows == rowIDs
        previousRows = rowIDs
        previousNanos = completedAtNanos
        // The first commit only anchors the interval, excluding pipeline fill.
        // Membership changes also re-anchor, even within the same size bucket.
        guard sameRows, let start, completedAtNanos > start else { return nil }
        return completedAtNanos - start
    }
}
