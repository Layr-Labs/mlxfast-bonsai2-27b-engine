import Darwin
import Foundation
import os

/// Keeps each round's host work on a performance core.
///
/// Measured on an M4 Max with per-thread, per-performance-level counters: the
/// round's host work is the same in every round (55 M instructions for the
/// verify build, the same allocator traffic), yet for whole windows the
/// scheduler runs the engine thread on the efficiency cluster (1.0-2.5 GHz,
/// IPC 2.7) instead of a performance core (3.7-4.5 GHz, IPC 4.1). The verify
/// build then takes 11.4 ms instead of 3.8 ms, and the committed recurrent
/// state takes 5.3 ms instead of 3.1 ms to encode. The thread sleeps on the GPU
/// for most of every round, so its thread group looks light and the scheduler
/// recommends the efficiency cluster for it. The queue's QoS (userInitiated or
/// userInteractive) does not change that, and a time-constraint policy is
/// silently ignored on a dispatch worker thread.
///
/// A work interval with a deadline changes it. From the readback to the end of
/// the step, the host work is one interval of the engine's own workgroup, due a
/// few milliseconds after the readback. The scheduler then runs it on a
/// performance core at full clock: the verify build took 3.1 ms in every
/// round, and GPU idle per round fell from 2.0 ms to 0.6 ms with unchanged GPU
/// busy time. The workgroup is an audio work interval (`AudioWorkIntervalCreate`,
/// public since macOS 11). Swift does not export it, so it is looked up at run
/// time, and it serves only as the scheduling hint. The thread's policy stays
/// untouched and nothing spins. Every failure leaves the plain dispatch thread.
/// `MLXFAST_ENGINE_WORK_INTERVAL=0` turns the hint off.
///
/// Engine-queue only: `stepBegan`, `hostWorkBegan` and `stepEnded` run on the
/// serial engine queue, one step at a time.
final class CBv2EngineWorkInterval: @unchecked Sendable {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_ENGINE_WORK_INTERVAL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The interval's deadline after the readback, in mach absolute time.
    private static let deadlineTicks: UInt64 = {
        var info = mach_timebase_info_data_t()
        guard mach_timebase_info(&info) == KERN_SUCCESS, info.numer > 0 else { return 96_000 }
        return 4_000_000 * UInt64(info.denom) / UInt64(info.numer)
    }()

    private typealias Create =
        @convention(c) (UnsafePointer<CChar>, UInt32, UnsafeMutableRawPointer?) ->
        UnsafeMutableRawPointer?

    private static let create: Create? = {
        guard enabled else { return nil }
        _ = dlopen("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox", RTLD_NOW)
        guard let symbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "AudioWorkIntervalCreate")
        else { return nil }
        return unsafeBitCast(symbol, to: Create.self)
    }()

    /// `OS_CLOCK_MACH_ABSOLUTE_TIME` from <os/clock.h>.
    private static let machAbsoluteClock: UInt32 = 32

    private let interval: WorkGroup?
    private var token: WorkGroup.JoinToken?
    private var joinedThread: pthread_t?
    private var started = false

    init() {
        interval = Self.create
            .flatMap { $0("com.eigen.cbv2.engine", Self.machAbsoluteClock, nil) }
            .map { Unmanaged<WorkGroup>.fromOpaque($0).takeRetainedValue() }
    }

    /// Joins the calling thread for one engine step. Returns whether it joined;
    /// only a step that joined calls `stepEnded`.
    func stepBegan() -> Bool {
        guard let interval, token == nil else { return false }
        token = interval.join()
        joinedThread = pthread_self()
        return true
    }

    /// The readback returned: the round's host work starts now.
    func hostWorkBegan() {
        guard let interval, token != nil, !started, let joinedThread,
            pthread_equal(joinedThread, pthread_self()) != 0
        else { return }
        let now = mach_absolute_time()
        interval.start(at: now, deadline: now &+ Self.deadlineTicks)
        started = true
    }

    func stepEnded() {
        guard let interval, let token else { return }
        if started {
            interval.finish()
            started = false
        }
        interval.leave(token: token)
        self.token = nil
        joinedThread = nil
    }
}
