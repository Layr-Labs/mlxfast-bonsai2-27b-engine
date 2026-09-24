import Foundation

/// One engine's synchronous connection to process-wide memory admission.
/// Admission supplies its complete native charge and exact evaluated coverage.
/// Implementations must not call an engine, await, allocate GPU buffers or do I/O.
public protocol CBv2ProcessMemoryOwner: AnyObject, Sendable {
    func replaceCharge(_ bytes: UInt64) throws
    func recordMaterialization(_ bytes: UInt64) throws
    func withdrawCoverage(_ bytes: UInt64) throws
    func retire()
}

/// Removing materialization credit is conservative; it never refunds charge.
/// The native caller refunds charge only after the actual buffer owners drain.
final class CBv2MemoryCoverage: @unchecked Sendable {
    private let lock = NSLock()
    private var withdraw: (@Sendable () -> Void)?

    init(withdraw: @escaping @Sendable () -> Void) { self.withdraw = withdraw }

    func invalidate() {
        lock.lock()
        defer { lock.unlock() }
        let callback = withdraw
        withdraw = nil
        callback?()
    }

    deinit { invalidate() }
}
