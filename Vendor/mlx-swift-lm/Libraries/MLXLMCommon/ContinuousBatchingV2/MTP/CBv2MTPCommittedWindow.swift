/// A bounded, contiguous observation of stateless speculation. A singleton
/// probe may pay a seed cost that obscures profitable retained-carry decode.
/// The window charges the complete observation's time and actual outputs,
/// while each engine round continues to stream and obey ordinary stop gates.
struct CBv2MTPCommittedWindow {
    static let maximumVerifiedRounds = 8

    let decision: CBv2MTPDepthDecision
    let rowCount: Int
    private(set) var verifiedRounds = 0
    private(set) var wallTimeNanos: UInt64 = 0
    private(set) var committedTokens = 0

    var isComplete: Bool { verifiedRounds >= Self.maximumVerifiedRounds }

    mutating func observe(wallTimeNanos: UInt64, committedTokens: Int, warmup: Bool) {
        guard !isComplete else { return }
        verifiedRounds += 1
        // JIT still consumes a bounded observation slot. The driver records
        // its full elapsed time in telemetry before discarding this sample.
        guard !warmup else { return }
        self.wallTimeNanos &+= wallTimeNanos
        self.committedTokens += committedTokens
    }
}
