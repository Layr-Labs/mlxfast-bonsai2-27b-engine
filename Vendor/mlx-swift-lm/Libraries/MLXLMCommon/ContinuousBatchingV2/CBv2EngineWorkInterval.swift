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
/// A prompt forward's host work outlives one deadline: building and encoding
/// a 512-row forward takes longer than 4 ms, and once the deadline passes the
/// thread drifts back to the efficiency cluster (the seed step's engine CPU
/// ran 94% there, the step's interval notwithstanding). So a prompt-width
/// forward inside an engine step renews the step's interval at its start and
/// at each early submission of its prompt plan (`promptForwardBegan`,
/// `promptSubmitted`): the running interval finishes and the next one starts,
/// due one deadline later (M4 Max: the seed step's engine CPU 78% on the
/// efficiency cluster, 59 ms instead of 82 ms). A forward on a thread that
/// joined no interval is left alone: the timed `prefill` verb's worker thread
/// runs its forward on the performance cluster by itself (13% efficiency),
/// and joining it to an interval of its own moved it there (76%). Rounds and
/// verify windows are untouched. `MLXFAST_PROMPT_WORK_INTERVAL=0` turns only
/// this prompt renewal off; a positive integer sets its deadline in
/// milliseconds (default 4).
///
/// `stepBegan`, `hostWorkBegan` and `stepEnded` run on the serial engine
/// queue, one step at a time; the prompt entry points act only on an interval
/// the calling thread itself joined.
public final class CBv2EngineWorkInterval: @unchecked Sendable {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_ENGINE_WORK_INTERVAL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `milliseconds` in mach absolute time.
    private static func ticks(milliseconds: UInt64) -> UInt64 {
        var info = mach_timebase_info_data_t()
        guard mach_timebase_info(&info) == KERN_SUCCESS, info.numer > 0 else {
            return 24_000 * milliseconds
        }
        return milliseconds * 1_000_000 * UInt64(info.denom) / UInt64(info.numer)
    }

    /// The interval's deadline after the readback, in mach absolute time.
    private static let deadlineTicks: UInt64 = ticks(milliseconds: 4)

    /// The deadline of a prompt forward's renewed intervals, or nil when the
    /// prompt coverage is off (`MLXFAST_PROMPT_WORK_INTERVAL`).
    private static let promptDeadlineTicks: UInt64? = {
        guard enabled else { return nil }
        let value = ProcessInfo.processInfo.environment["MLXFAST_PROMPT_WORK_INTERVAL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ["0", "false", "no", "off"].contains(value ?? "") { return nil }
        let milliseconds = value.flatMap { UInt64($0) }.flatMap { $0 > 0 ? $0 : nil } ?? 4
        return ticks(milliseconds: milliseconds)
    }()

    /// The interval the calling thread has joined (unretained; set by
    /// `stepBegan`, cleared by `stepEnded`).
    private static let currentKey: pthread_key_t? = {
        var key = pthread_key_t()
        return pthread_key_create(&key, nil) == 0 ? key : nil
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
        if let key = Self.currentKey {
            pthread_setspecific(key, Unmanaged.passUnretained(self).toOpaque())
        }
        return true
    }

    /// The readback returned: the round's host work starts now.
    func hostWorkBegan() {
        start(deadlineTicks: Self.deadlineTicks)
    }

    private var onJoinedThread: Bool {
        guard token != nil, let joinedThread else { return false }
        return pthread_equal(joinedThread, pthread_self()) != 0
    }

    private func start(deadlineTicks: UInt64) {
        guard let interval, !started, onJoinedThread else { return }
        let now = mach_absolute_time()
        interval.start(at: now, deadline: now &+ deadlineTicks)
        started = true
    }

    /// Finishes the running interval and starts the next, due `deadlineTicks`
    /// from now; starts one if none runs.
    private func renew(deadlineTicks: UInt64) {
        guard let interval, onJoinedThread else { return }
        if started {
            interval.finish()
            started = false
        }
        start(deadlineTicks: deadlineTicks)
    }

    /// The interval the calling thread has joined, if any.
    private static func current() -> CBv2EngineWorkInterval? {
        guard let key = currentKey, let pointer = pthread_getspecific(key) else { return nil }
        return Unmanaged<CBv2EngineWorkInterval>.fromOpaque(pointer).takeUnretainedValue()
    }

    /// A prompt-width forward starts building on the calling thread: renew
    /// the interval the thread joined for its engine step (the step finishes
    /// it). A thread that joined none is left alone.
    public static func promptForwardBegan() {
        guard let promptDeadlineTicks, let current = current() else { return }
        current.renew(deadlineTicks: promptDeadlineTicks)
    }

    /// A prompt forward submits the layers built so far: renew the calling
    /// thread's running interval, so the encoding and the layers built next
    /// run within a deadline.
    public static func promptSubmitted() {
        guard let promptDeadlineTicks, let current = current(), current.started else { return }
        current.renew(deadlineTicks: promptDeadlineTicks)
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
        if let key = Self.currentKey,
            pthread_getspecific(key) == Unmanaged.passUnretained(self).toOpaque()
        {
            pthread_setspecific(key, nil)
        }
    }
}
