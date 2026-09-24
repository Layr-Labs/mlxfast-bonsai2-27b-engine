// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — per-request timing stamps (`CBv2RequestTiming`).
//
// Every helper here is ENGINE-THREAD ONLY and writes plain integer fields on
// the `CBv2ScheduledRequest` already in hand: no allocation, no lock, no
// clock read. Instants come from the step's existing clock reads
// (`CBv2InFlightStep.wallStartedNanos` at launch, `readbackDoneNanos` set once
// in `finalize`) and are stored as offsets from `enqueuedNanos`. An observed
// offset is clamped to `>= 1` so `0` always means "not observed".

import Foundation

extension CBv2ScheduledRequest {

    /// Offset of `nowNanos` from the enqueue instant, never 0 once observed.
    @inline(__always)
    func timingOffset(_ nowNanos: UInt64) -> UInt64 {
        max(1, nowNanos &- enqueuedNanos)
    }

    /// First step whose plan included this row (`wallStartedNanos` of that
    /// step). Idempotent: re-admissions after preemption / capacity requeue
    /// keep the first stamp and are counted in `readmissions` instead.
    ///
    /// A prefix-cache hit allocates KV at enqueue-time adoption, BEFORE any
    /// plan includes the row; the exported contract requires
    /// `admitted <= kvAllocated`, so an earlier allocation stamp is raised
    /// to the admission instant here (once, on the first admission only).
    /// The adoption itself stays evidenced by `prefixAdoptionNanos > 0`.
    @inline(__always)
    func stampAdmission(launchNanos: UInt64) {
        if timing.admittedNanos == 0 {
            timing.admittedNanos = timingOffset(launchNanos)
            if timing.kvAllocatedNanos != 0, timing.kvAllocatedNanos < timing.admittedNanos {
                timing.kvAllocatedNanos = timing.admittedNanos
            }
        }
    }

    /// Per-layer KV state allocated. Idempotent (a preempted row
    /// re-allocates; the first allocation is the one that explains TTFT).
    @inline(__always)
    func stampKVAllocated(nowNanos: UInt64) {
        if timing.kvAllocatedNanos == 0 {
            timing.kvAllocatedNanos = timingOffset(nowNanos)
        }
    }

    /// One prefill chunk forward launched for this row this step.
    @inline(__always)
    func stampPrefillChunkLaunch(
        tokens: Int, packed: Bool, vision: Bool, stripe: Bool, launchNanos: UInt64
    ) {
        if timing.prefillFirstLaunchNanos == 0 {
            timing.prefillFirstLaunchNanos = timingOffset(launchNanos)
        }
        timing.prefillChunks &+= 1
        if packed { timing.packedPrefillChunks &+= 1 }
        if vision { timing.visionChunks &+= 1 }
        if stripe { timing.soloStripeChunks &+= 1 }
        let width = UInt32(clamping: tokens)
        if width > timing.prefillChunkTokensMax { timing.prefillChunkTokensMax = width }
    }

    /// The row confirmed token(s) in a finalized step: batch-size and
    /// launch→confirm latency aggregates. `batchRows` is the step's
    /// token-producing row count; the latency is the step's readback-done
    /// instant minus its launch instant (both already read).
    @inline(__always)
    func recordStepParticipation(step: CBv2InFlightStep, batchRows: Int) {
        let rows = UInt32(clamping: batchRows)
        timing.batchRowsSum &+= UInt64(rows)
        if timing.batchRowsMin == 0 || rows < timing.batchRowsMin {
            timing.batchRowsMin = rows
        }
        if rows > timing.batchRowsMax { timing.batchRowsMax = rows }
        let latency = step.readbackDoneNanos &- step.wallStartedNanos
        timing.stepLatencyNanosSum &+= latency
        if latency > timing.stepLatencyNanosMax { timing.stepLatencyNanosMax = latency }
        if step.chained { timing.chainedDecodeSteps &+= 1 }
    }

    /// The step that confirmed this row's FIRST generated token: the prompt
    /// is fully computed and the engine-side TTFT is observed (both at the
    /// readback-done instant).
    @inline(__always)
    func stampFirstToken(readbackDoneNanos: UInt64) {
        let offset = timingOffset(readbackDoneNanos)
        if timing.promptComputedNanos == 0 { timing.promptComputedNanos = offset }
        if timing.firstTokenNanos == 0 { timing.firstTokenNanos = offset }
    }

    /// One MTP verify round finalized for this row.
    @inline(__always)
    func recordMTPRound(drafted: Int, accepted: Int) {
        timing.mtpRounds &+= 1
        timing.mtpProposed &+= UInt32(clamping: drafted)
        timing.mtpAccepted &+= UInt32(clamping: accepted)
    }

    /// Backpressure pause / resume on the engine thread (monotonic instants
    /// from the loop's injectable clock; the paused interval is a duration,
    /// so the clock domain is irrelevant).
    func recordPaused(now: ContinuousClock.Instant) {
        guard pausedSince == nil else { return }
        pausedSince = now
        timing.pauseCount &+= 1
    }

    func recordResumed(now: ContinuousClock.Instant) {
        guard let since = pausedSince else { return }
        pausedSince = nil
        timing.pausedNanos &+= Self.nanoseconds(now - since)
    }

    /// Fold the terminal instant and the scheduler's own counters into the
    /// exported struct. Called once, in `finishRequest`.
    func exportTiming(finishedNanos: UInt64) -> CBv2RequestTiming {
        var exported = timing
        exported.finishedNanos = timingOffset(finishedNanos)
        exported.preemptions = UInt32(clamping: preemptionCount)
        return exported
    }

    private static func nanoseconds(_ duration: Duration) -> UInt64 {
        guard duration > .zero else { return 0 }
        let components = duration.components
        let seconds = UInt64(clamping: components.seconds)
        let (whole, overflow) = seconds.multipliedReportingOverflow(by: 1_000_000_000)
        guard !overflow else { return UInt64.max }
        let fractional = UInt64(clamping: components.attoseconds / 1_000_000_000)
        return whole &+ fractional
    }
}
