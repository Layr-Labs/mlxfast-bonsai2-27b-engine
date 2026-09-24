import Foundation

/// One engine's immutable backend has one physical floor. Nominal target KV
/// from its own requests can replace this charge; auxiliary and exact external
/// storage never do. AdmissionV2 owns this value under its existing lock.
struct CBv2BackendPhysicalFloor {
    var isBound = false
    var physicalBytes = 0
    var nominalBytes = 0

    var overheadBytes: Int { max(0, physicalBytes - nominalBytes) }

    func chargedBytes(base: Int, nominal: Int? = nil, physical: Int? = nil) -> Int? {
        let overhead = max(0, (physical ?? physicalBytes) - (nominal ?? nominalBytes))
        let (value, overflow) = base.addingReportingOverflow(overhead)
        return overflow ? nil : value
    }
}

struct CBv2BackendPhysicalAccounting: Sendable {
    let nominalKVBytes: Int
    let physicalFloorOverheadBytes: Int
}

/// The pool retains this single owner while any native backing exists.
/// Growth reserves before allocation; shrink refunds only after aliases retire.
/// The resize callback acquires AdmissionV2's lock and changes no pool metadata.
final class CBv2BackendPhysicalLease: @unchecked Sendable {
    private let lock = NSLock()
    private var resizeBody: (@Sendable (Int) throws -> Void)?
    private var retainedBytes: Int
    private let admissionIdentity: ObjectIdentifier?
    private var onClose: (@Sendable () -> Void)?
    private var snapshotBody: (@Sendable () -> CBv2BackendPhysicalAccounting?)?

    init(bytes: Int, resize: @escaping @Sendable (Int) throws -> Void,
         onClose: @escaping @Sendable () -> Void,
         snapshot: (@Sendable () -> CBv2BackendPhysicalAccounting?)? = nil,
         admissionIdentity: ObjectIdentifier? = nil) {
        self.admissionIdentity = admissionIdentity
        self.retainedBytes = bytes
        self.resizeBody = resize
        self.onClose = onClose
        self.snapshotBody = snapshot
    }

    var bytes: Int { lock.withLock { retainedBytes } }
    var accountingSnapshot: CBv2BackendPhysicalAccounting? {
        lock.withLock { snapshotBody?() }
    }

    func resize(to bytes: Int) throws {
        precondition(bytes >= 0)
        lock.lock()
        defer { lock.unlock() }
        guard let resizeBody else {
            throw CBv2KVError.backendIneligible(reason: "paged physical lease is closed")
        }
        try resizeBody(bytes)
        retainedBytes = bytes
    }

    /// Update retained floor bytes together with the specialized Admission
    /// transaction. The body may only mutate accounting metadata, never allocate.
    func transferCheckpoint(
        to bytes: Int, admission: AdmissionV2,
        _ body: (Int) throws -> CBv2CheckpointAdoptionReservation
    ) throws -> CBv2CheckpointAdoptionReservation {
        try lock.withLock {
            guard resizeBody != nil, admissionIdentity == ObjectIdentifier(admission) else {
                throw CBv2CompleteCheckpointError.incompatibleCheckpoint
            }
            let previous = retainedBytes
            let reservation = try body(previous)
            retainedBytes = bytes
            return CBv2CheckpointAdoptionReservation { [self] in
                lock.withLock {
                    precondition(retainedBytes == bytes)
                    reservation.rollbackAfterDroppingOwners()
                    retainedBytes = previous
                }
            } onCommit: {
                reservation.commit()
            }
        }
    }

    /// Native owners have already been dropped. Lowering cannot exceed the
    /// admission ceiling and the pool never closes its lease while active.
    func release(to bytes: Int) {
        lock.lock()
        defer { lock.unlock() }
        precondition(bytes >= 0 && bytes <= retainedBytes)
        guard let resizeBody else {
            preconditionFailure("paged physical lease released after close")
        }
        do { try resizeBody(bytes) }
        catch { preconditionFailure("physical floor reduction refused: \(error)") }
        retainedBytes = bytes
    }

    func close() {
        lock.lock()
        let callback = onClose
        onClose = nil
        resizeBody = nil
        snapshotBody = nil
        retainedBytes = 0
        lock.unlock()
        callback?()
    }
    deinit { close() }
}
