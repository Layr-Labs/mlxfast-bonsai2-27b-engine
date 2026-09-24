import Foundation
import MLX

/// Private import owner, assembled after the final authenticated tensor byte.
/// The enclosing codec must move its arrays into this object, retaining no
/// writable source aliases. This mechanical frame is not cache eligibility.
final class CBv2PagedCheckpointFrame: @unchecked Sendable {
    private let lock = NSLock()
    private var owner: CBv2PagedCheckpointOwner?

    init(storage: CBv2PagedCheckpointStorage, auxiliary: [MLXArray],
         lease: CBv2CheckpointStageLease) throws {
        let bytes = try CBv2CheckpointAllocationFootprint.freshBytes(auxiliary)
        guard storage.plan.nativeBytes == lease.targetBytes, bytes.bound == lease.auxiliaryBytes,
            storage.allocatedBytes <= storage.plan.nativeBytes
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        try lease.settleDestinationAfterEvaluation(
            targetBytes: storage.allocatedBytes, auxiliaryBytes: bytes.actual)
        owner = CBv2PagedCheckpointOwner(storage: storage, auxiliary: auxiliary, lease: lease)
    }

    /// Empty the public closeable owner before engine work. A concurrent close
    /// can win or lose this move, but cannot refund arrays being adopted.
    func consume() throws -> CBv2PagedCheckpointOwner {
        try lock.withLock {
            guard let owner else { throw CBv2CompleteCheckpointError.closed }
            self.owner = nil
            return owner
        }
    }

    func close() {
        let owner = lock.withLock {
            let value = self.owner
            self.owner = nil
            return value
        }
        owner?.close()
    }

    deinit { close() }
}

/// Single engine-queue owner after consume. Its scratch reservation survives
/// the destination transfer until close; native/process charge is not doubled.
final class CBv2PagedCheckpointOwner {
    let storage: CBv2PagedCheckpointStorage
    var auxiliary: [MLXArray]
    let lease: CBv2CheckpointStageLease

    init(storage: CBv2PagedCheckpointStorage, auxiliary: [MLXArray], lease: CBv2CheckpointStageLease) {
        self.storage = storage
        self.auxiliary = auxiliary
        self.lease = lease
    }

    func close() {
        storage.close()
        auxiliary.removeAll()
        lease.closeAfterDroppingOwners()
    }

    deinit { close() }
}

/// Resources installed in ordinary pool ownership. The engine integration can
/// move them into its active state; this mechanical slice keeps an explicit
/// retirement boundary for rollback tests and never refunds before aliases.
final class CBv2PagedCheckpointAdoption {
    private(set) var rows: [PagedSequenceKV]
    private(set) var auxiliary: [MLXArray]
    private var releaseAdmission: (() -> Void)?
    private let modelIndices: [Int]
    private let stateCount: Int

    init(rows: [PagedSequenceKV], auxiliary: [MLXArray], modelIndices: [Int]? = nil,
         stateCount: Int? = nil, releaseAdmission: @escaping () -> Void) {
        self.modelIndices = modelIndices ?? Array(rows.indices)
        self.stateCount = stateCount ?? rows.count
        self.rows = rows
        self.auxiliary = auxiliary
        self.releaseAdmission = releaseAdmission
    }

    /// Restore only real recurrent/assistant request state from these private
    /// auxiliary arrays. On success ordinary EngineLoop rows own both the pool
    /// charge and its request reservation; the temporary result cannot refund it.
    /// A throwing restorer must drop every candidate state alias before returning.
    func moveToActiveRequest(_ restore: ([MLXArray]) throws -> Void) throws -> [CBv2SequenceKV?] {
        guard releaseAdmission != nil, !rows.isEmpty else { throw CBv2CompleteCheckpointError.closed }
        do { try restore(auxiliary) } catch {
            release()
            throw error
        }
        var state = [CBv2SequenceKV?](repeating: nil, count: stateCount)
        for (row, index) in zip(rows, modelIndices) { state[index] = row }
        rows.removeAll()
        auxiliary.removeAll()
        releaseAdmission = nil
        return state
    }

    func release() {
        for row in rows { row.releaseStorage() }
        rows.removeAll()
        auxiliary.removeAll()
        let release = releaseAdmission
        releaseAdmission = nil
        release?()
    }

    deinit { release() }
}
