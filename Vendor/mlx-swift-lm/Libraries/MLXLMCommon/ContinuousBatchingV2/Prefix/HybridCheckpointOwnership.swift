import Foundation

/// One physical charge for immutable checkpoints shared by a warm request and
/// older published entries. Ownership moves between phases under the bank lock.
struct CBv2HybridCheckpointOwnership {
    enum Phase: Int { case staged, publishing, resident }
    private struct Storage {
        let bytes: Int
        var references = [0, 0, 0]
        var phase: Phase? {
            if references[Phase.resident.rawValue] > 0 { return .resident }
            if references[Phase.publishing.rawValue] > 0 { return .publishing }
            if references[Phase.staged.rawValue] > 0 { return .staged }
            return nil
        }
    }
    private var storage: [UUID: Storage] = [:]
    private(set) var stagedBytes = 0
    private(set) var publishingBytes = 0
    private(set) var residentBytes = 0
    var retainedBytes: Int { stagedBytes + publishingBytes + residentBytes }

    mutating func retain(_ checkpoint: CBv2RecurrentCheckpoint, as phase: Phase) {
        update(checkpoint, phase: phase, delta: 1)
    }

    mutating func release(_ checkpoint: CBv2RecurrentCheckpoint, from phase: Phase) {
        update(checkpoint, phase: phase, delta: -1)
    }

    mutating func transfer(_ checkpoint: CBv2RecurrentCheckpoint, from: Phase, to: Phase) {
        retain(checkpoint, as: to)
        release(checkpoint, from: from)
    }

    private mutating func update(_ checkpoint: CBv2RecurrentCheckpoint, phase: Phase, delta: Int) {
        var item = storage[checkpoint.storageID] ?? Storage(bytes: checkpoint.byteCount)
        precondition(item.bytes == checkpoint.byteCount)
        adjust(item.phase, bytes: -item.bytes)
        item.references[phase.rawValue] += delta
        precondition(item.references[phase.rawValue] >= 0, "unbalanced hybrid checkpoint ownership")
        adjust(item.phase, bytes: item.bytes)
        storage[checkpoint.storageID] = item.phase == nil ? nil : item
    }

    private mutating func adjust(_ phase: Phase?, bytes: Int) {
        switch phase {
        case .staged: stagedBytes += bytes
        case .publishing: publishingBytes += bytes
        case .resident: residentBytes += bytes
        case nil: break
        }
    }
}
