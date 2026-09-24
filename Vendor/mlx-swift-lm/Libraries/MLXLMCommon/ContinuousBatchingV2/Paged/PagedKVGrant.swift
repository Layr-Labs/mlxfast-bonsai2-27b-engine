import Foundation

/// The only pool state touched by budget updates off the engine queue.
/// Metadata and GPU ownership are installed by the queue after an epoch check.
final class PagedKVGrant: @unchecked Sendable {
    struct Snapshot: Sendable, Equatable {
        let bytes: Int
        let epoch: UInt64
    }

    enum Publication: Equatable {
        case installed, staleEpoch, exceedsGrant
    }

    private let lock = NSLock()
    private var bytes: Int
    private var epoch: UInt64 = 0

    init(bytes: Int) { self.bytes = max(0, bytes) }

    func snapshot() -> Snapshot {
        lock.withLock { Snapshot(bytes: bytes, epoch: epoch) }
    }

    func update(bytes: Int) {
        lock.withLock {
            let value = max(0, bytes)
            guard value != self.bytes else { return }
            self.bytes = value
            epoch &+= 1
            precondition(epoch != 0, "paged grant epoch exhausted")
        }
    }

    /// The caller prepares complete replacement metadata and buffers before
    /// entering. The body may only swap ownership; no allocation, I/O or GPU
    /// execution runs under this lock. A stale epoch never publishes any part.
    func publish(expected: Snapshot, physicalBytes: Int, existingPhysicalBytes: Int = 0,
                 install: () -> Void) -> Publication {
        lock.withLock {
            guard epoch == expected.epoch else { return .staleEpoch }
            guard physicalBytes <= max(bytes, existingPhysicalBytes) else { return .exceedsGrant }
            install()
            return .installed
        }
    }
}

/// Physical ownership is distinct from the logical grant. After a shrink,
/// existing owners may temporarily exceed the grant without losing their pages.
public struct PagedKVStorageSnapshot: Sendable, Equatable {
    /// A new pool resets its counters and has a distinct generation. Sequence
    /// and monotonic time describe the allocator capture, not heartbeat polls.
    public let generation: UUID
    public let captureSequence: UInt64
    public let capturedUptimeNanoseconds: UInt64
    public var grantBytes: Int
    public let committedBytes: Int
    public let reservedPageBytes: Int
    public let livePageBytes: Int
    public let poisonBytes: Int
    public let slackBytes: Int
    /// Retained allocator padding beyond logical poison+usable segment bytes.
    public var allocatorPaddingBytes: Int = 0
    /// Last successful preparation's unused conservative allocation allowance.
    public var lastAllocationAllowanceBytes: Int = 0
    public let segmentCount: Int
    public let addressPages: Int
    /// nil only for a low-level pool without an AdmissionV2 physical owner.
    public let nominalKVBytes: Int?
    public let physicalFloorOverheadBytes: Int?
    public let allocationFailures: UInt64
    public let admissionRefusals: UInt64
    public let grantRefusals: UInt64
    public let grantEpochRetries: UInt64
    public var overGrantBytes: Int { max(0, committedBytes - grantBytes) }
}

/// Engine-queue-owned provenance and cumulative outcomes. Advisory can-fit
/// probes do not mutate these counters; one failed allocation transaction is
/// one event even when it unwinds several groups.
struct PagedKVStorageTelemetry {
    let generation = UUID()
    private(set) var captureSequence: UInt64 = 0
    var allocationFailures: UInt64 = 0
    var admissionRefusals: UInt64 = 0
    var grantRefusals: UInt64 = 0
    var grantEpochRetries: UInt64 = 0
    private(set) var lastAllocationAllowanceBytes = 0

    mutating func recordSettlement(bound: Int, actual: Int) {
        lastAllocationAllowanceBytes = max(0, bound - actual)
    }

    mutating func capture() -> UInt64 {
        Self.increment(&captureSequence)
        return DispatchTime.now().uptimeNanoseconds
    }

    mutating func record(_ result: PagedKVGrant.Publication) {
        switch result {
        case .installed: break
        case .staleEpoch: Self.increment(&grantEpochRetries)
        case .exceedsGrant: Self.increment(&grantRefusals)
        }
    }

    static func increment(_ counter: inout UInt64) {
        if counter < UInt64.max { counter += 1 }
    }
}
