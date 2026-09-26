// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: the execution loop.
//
// Single engine thread (serial GCD queue, the EngineCore idiom): the loop
// does graph-build + `asyncEval` ONLY. Tokenization/detokenization state
// machines are pluggable (WS-E), prefix-cache donation and SSD I/O live
// elsewhere. Decode is rectangular [B, 1]; prefill runs per-request
// [1, chunk] under the shared token budget. There is no left padding, no
// shared frontier, and no batch-wide trim anywhere in this file.
//
// Chained async decode (SGLang-MLX pattern, report 09 §7): step N+1's [B, 1]
// forward is built ON TOP of step N's still-lazy sampled-token array and
// `asyncEval`ed BEFORE the loop blocks on step N's tokens for stop detection.
// Tokens are therefore inspected one step late; a finished request wastes at
// most one slot-step, and its extra token + KV tail are rolled back
// (`CBv2SequenceKV.rollback(1)`). The chain breaks on ANY membership change
// (prefill completion into a different set, finish, cancel, join, pause).

import Foundation
import MLX
import os

// MARK: - Model interface (WS-F adapters / WS-G fixtures conform)

/// Minimal steppable-model surface the loop drives. `tokens` is [B, L] int32
/// ([B, 1] decode, [1, chunk] prefill); `caches` has one entry per model
/// layer with rows matching batch rows. Returns logits [B, L, vocab].
public protocol CBv2SteppableModel: AnyObject {
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray
}

/// Builds per-layer batch-facing cache views for a set of rows
/// (`rowStates[b][layer]`, row order == batch row order). WS-A's
/// `LayerCacheV2` conforms; see CONTRACT-ISSUES-B-scheduler.md §1.
public protocol CBv2LayerCacheProvider: AnyObject {
    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache]
    /// True when EVERY cache this provider vends honors span-mask contexts
    /// (`CBv2SpanMaskBinding`), so vision prefill chunks can carry their
    /// causal-plus-bidirectional-within-span masks. Fail-safe default is
    /// false: providers that cannot vouch (paged backend, custom caches)
    /// reject multimodal requests at submit rather than silently serving
    /// them with plain causal masks.
    var supportsMultimodalSpans: Bool { get }
    /// True only when EVERY layer cache this provider vends keeps rows
    /// independent under a rectangular `[B > 1, L > 1]` prompt pass, so
    /// equal-length text chunks may be coalesced into one layer-major
    /// forward. Fail-safe default false (paged/custom providers).
    var supportsPackedPrefill: Bool { get }
    /// True only when every layer cache can bind one optional span context
    /// per rectangular row. This is stricter than single-row multimodal
    /// support and fails closed for paged/custom providers.
    var supportsPackedMultimodalSpans: Bool { get }
    /// True only when every layer cache can serialize the columns of one
    /// rectangular recurrent MTP verification window without cross-column
    /// attention. Stateful Qwen drafting fails closed without this seam.
    var supportsMTPRectangularVerification: Bool { get }
    /// True only when EVERY cache this provider vends honors the optional
    /// per-row keep mask on `updateAndAttend`. Sparse-attention families
    /// (Qwen 3.8 Flash-Next QSA) declare `cbv2RequiresKeepMask` and the
    /// engine refuses at construction when this is false, so a dense
    /// fallback can never be served silently. Fail-safe default false.
    var supportsKeepMask: Bool { get }
}

extension CBv2LayerCacheProvider {
    public var supportsMultimodalSpans: Bool { false }
    public var supportsPackedPrefill: Bool { false }
    public var supportsPackedMultimodalSpans: Bool { false }
    public var supportsMTPRectangularVerification: Bool { false }
    public var supportsKeepMask: Bool { false }
}

// MARK: - Sampler interface (WS-E's CBv2DefaultSampler is the production impl)

/// Samples next tokens from last-position logits [B, vocab] → lazy token
/// array [B] (int32). MUST NOT host-sync; see
/// CONTRACT-ISSUES-B-scheduler.md §2 and CONTRACT-DECISIONS.md.
///
/// Stateful samplers (penalties, keyed RNG) reconfigure on membership
/// change using `rowContext` and track per-request progress:
///  - `params`/`requestIDs`: per-row, row order == logits row order.
///  - `stepIndex`: global engine step (telemetry only — per-request RNG must
///    key on per-request progress, never on this).
///  - `pendingSampledTokens`: lazy [B] tokens sampled for exactly these rows
///    by the still-in-flight previous step (chained decode), row-aligned;
///    nil when every row's history is fully confirmed. A sampler that
///    reconfigures from `rowContext` (confirmed history only) must fold
///    these in on-device to stay exact.
///  - `rowContext`: materializes full per-row context (id, params, prompt,
///    CONFIRMED output tokens). Copies token arrays — only call it on
///    membership change, never on the chained fast path.
public protocol CBv2StepSampler: AnyObject {
    /// True only when this sampler implements the full token-constraint
    /// lifecycle: configure, hard-mask, confirm, failure reporting, and
    /// request-id retirement. EngineV2 rejects constrained requests unless
    /// the installed sampler opts in.
    var supportsTokenConstraints: Bool { get }

    func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray

    /// Consume the LAZY logprob gather built by the immediately preceding
    /// `sample` call (raw pre-transform logprobs; contract rule), or nil when
    /// no row of that call requested `topLogprobs > 0`. Graph-only handles:
    /// the loop adds them to the step's `asyncEval` set and materializes them
    /// at the finalize boundary alongside the sampled tokens — never an extra
    /// host sync on the chained decode path. Called at most once per `sample`
    /// (take semantics). Default: nil (samplers without logprob support).
    func takeStepLogprobs() -> CBv2StepLogprobs?

    /// The request left the engine for good (finish, cancel, error). Its id
    /// may legally be REUSED by a FUTURE request, so stateful samplers must
    /// drop any per-request configured state: without this, a later request
    /// whose row-id fingerprint matches the retired one exactly (e.g. the
    /// same id resubmitted solo) would skip reconfiguration and inherit the
    /// finished request's penalty counts and RNG step index (PR#62 review).
    /// NOT called on preemption — a preempted request is the SAME request
    /// and its sampler state remains a pure function of its history.
    /// Default: no-op (stateless samplers).
    func requestDidFinish(_ id: CBv2RequestID)

    /// Confirm host-materialized samples at the existing finalization
    /// boundary. Stateful token constraints advance here; unconstrained
    /// samplers use the no-op default.
    func confirmSampledTokens(_ tokens: [Int], requestIDs: [CBv2RequestID])

    /// Typed constraint failure latched while masking or advancing a row.
    func tokenConstraintFailure(for id: CBv2RequestID) -> String?

    /// True when `mtpVerifySample` reproduces this sampler's ordinary
    /// per-token draws (same transform pipeline, same keyed RNG stream) for
    /// MTP-eligible rows. Gates target-prefix acceptance: with it, verify
    /// acceptance compares drafts against genuine target samples, which is
    /// exact for the output distribution at any temperature.
    var supportsMTPTargetPrefix: Bool { get }

    /// Pre-sample the target token at every MTP verify position with each
    /// row's REAL sampler semantics. `logits` is `[B, W, vocab]` raw target
    /// logits for the verify window; `stepBases[r]` is row r's per-request
    /// output-step index at window position 0 (its confirmed generated-token
    /// count), so position j draws with the exact RNG key the ordinary path
    /// would use for output index `stepBases[r] + j`. Rows admitted here
    /// satisfy the MTP eligibility gates (no constraints/bias/penalties/
    /// logprobs), so the transform is the stateless temperature → top-k/
    /// top-p/min-p pipeline. Greedy rows yield plain argmax. Pure: MUST NOT
    /// mutate sampler state or host-sync; returns lazy `[B, W]` int32, or
    /// nil when unsupported (callers then use argmax acceptance).
    func mtpVerifySample(
        logits: MLXArray, params: [CBv2SamplingParams],
        requestIDs: [CBv2RequestID], stepBases: [Int]
    ) -> MLXArray?

    /// An MTP round confirmed tokens for `requestIDs` OUTSIDE `sample`
    /// (verify rows never pass through it). Stateful samplers must drop the
    /// affected row configuration so the next `sample` reconfigures from
    /// confirmed history — penalty counts and RNG step indices stay a pure
    /// function of each request's history. Default: no-op.
    func mtpRoundDidCommit(requestIDs: [CBv2RequestID])
}

extension CBv2StepSampler {
    public var supportsTokenConstraints: Bool { false }
    public func takeStepLogprobs() -> CBv2StepLogprobs? { nil }
    public func requestDidFinish(_ id: CBv2RequestID) {}
    public func confirmSampledTokens(_ tokens: [Int], requestIDs: [CBv2RequestID]) {}
    public func tokenConstraintFailure(for id: CBv2RequestID) -> String? { nil }
    public var supportsMTPTargetPrefix: Bool { false }
    public func mtpVerifySample(
        logits: MLXArray, params: [CBv2SamplingParams],
        requestIDs: [CBv2RequestID], stepBases: [Int]
    ) -> MLXArray? { nil }
    public func mtpRoundDidCommit(requestIDs: [CBv2RequestID]) {}
}

/// Greedy stub — vectorized argmax, batch-composition invariant by
/// construction (per-row reduction, no cross-row ops). Kept as the
/// deterministic fallback for scheduler/loop tests; production uses
/// `CBv2DefaultSampler`.
public final class CBv2GreedySampler: CBv2StepSampler {
    private let constraintSampler = CBv2TokenConstraintSampler()
    private var configuredIDs: [CBv2RequestID] = []

    public var supportsTokenConstraints: Bool { true }

    public init() {}
    public func sample(
        logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
        stepIndex: Int, pendingSampledTokens: MLXArray?,
        rowContext: () -> [CBv2SamplerRow]
    ) -> MLXArray {
        if requestIDs != configuredIDs {
            constraintSampler.configure(rowContext())
            configuredIDs = requestIDs
        }
        return argMax(
            constraintSampler.mask(logits, requestIDs: requestIDs),
            axis: -1
        ).asType(.int32)
    }

    public func requestDidFinish(_ id: CBv2RequestID) {
        constraintSampler.requestDidFinish(id)
        if configuredIDs.contains(id) {
            configuredIDs = []
        }
    }

    public func confirmSampledTokens(
        _ tokens: [Int], requestIDs: [CBv2RequestID]
    ) {
        constraintSampler.confirm(tokens: tokens, requestIDs: requestIDs)
    }

    public func tokenConstraintFailure(for id: CBv2RequestID) -> String? {
        constraintSampler.failure(for: id)
    }
}

// MARK: - Detokenizer interface (WS-E conforms; null stub until then)

/// Incremental per-request detokenizer with UTF-8 + stop-string holdback.
/// See CONTRACT-ISSUES-B-scheduler.md §3.
public protocol CBv2IncrementalDetokenizer: AnyObject {
    /// Append confirmed tokens; returns text now safe to emit.
    func push(_ tokens: [Int]) -> String
    /// True once a stop string has matched (engine finishes with `.stop`).
    var matchedStopString: Bool { get }
    /// Held-back text still emittable at finish (excludes matched stop text).
    func flush() -> String
}

public protocol CBv2DetokenizerFactory: AnyObject {
    func makeDetokenizer(stopStrings: [String]) -> CBv2IncrementalDetokenizer
}

/// Default until WS-E lands: deltas carry token ids with empty text, stop
/// strings never match (stop tokens / maxTokens / deadlines still work).
public final class CBv2NullDetokenizerFactory: CBv2DetokenizerFactory {
    final class NullDetokenizer: CBv2IncrementalDetokenizer {
        let matchedStopString = false
        func push(_ tokens: [Int]) -> String { "" }
        func flush() -> String { "" }
    }
    public init() {}
    public func makeDetokenizer(stopStrings: [String]) -> CBv2IncrementalDetokenizer {
        NullDetokenizer()
    }
}

// MARK: - Loop configuration

public struct CBv2EngineLoopConfig: Sendable {
    /// LEGACY single total-lifetime wall (seconds), used ONLY when
    /// `useLegacyRequestTimeout` is true (the rollback kill-switch). In the
    /// default new-lease behavior this value is inert — the monotonic
    /// admission/prefill/decode/backpressure leases and the safety ceiling
    /// below govern request lifetime instead. See `CBv2DeadlineLeases.swift`.
    public var requestTimeout: TimeInterval
    /// Kill-switch: when true, restore the legacy behavior — one
    /// `requestTimeout`-second wall over the entire engine lifetime, expiring
    /// with the original `.error("request exceeded Ns deadline")` string.
    /// Default false = the new independent monotonic leases.
    public var useLegacyRequestTimeout: Bool
    /// Admission-only lease (seconds): bounds time before the request begins
    /// engine work. Ends permanently at first admission; never re-arms after
    /// preemption. Expiry cause `.admissionTimeout`.
    public var admissionLease: TimeInterval
    /// Prefill progress lease (seconds): expires a prompt prefill that stops
    /// making confirmed finalized progress. Cause `.prefillStall`.
    public var prefillProgressLease: TimeInterval
    /// Decode progress lease (seconds): expires decode that stops producing
    /// confirmed token progress. A generation that keeps producing tokens
    /// NEVER expires. Cause `.decodeStall`.
    public var decodeProgressLease: TimeInterval
    /// Backpressure lease (seconds): bounds how long a request may sit paused
    /// on downstream buffer pressure. Health-neutral. Cause
    /// `.backpressureTimeout`.
    public var backpressureLease: TimeInterval
    /// Conservative decode throughput floor (tokens/second) for the absolute
    /// request-derived safety ceiling. Deliberately far below any real model
    /// (~30–120 tok/s) so the ceiling only catches pathology, never a healthy
    /// long generation. Cause `.safetyDeadline`.
    public var safetyCeilingDecodeFloorTPS: Double
    /// Injectable monotonic clock. Production uses `ContinuousClock`; tests
    /// inject a fake to drive lease expiry with no real sleeps. Never `Date`.
    public var clock: CBv2Clock
    /// Single-step watchdog: a step (graph build + blocking eval) exceeding
    /// this marks the engine unhealthy, terminal-finishes all live streams
    /// with cause `.watchdog` (carrying the reconciled usage observed before
    /// the wedge), and fires `onStepWedge`.
    public var stepTimeout: TimeInterval
    /// Watchdog polling interval.
    public var watchdogInterval: TimeInterval
    /// Idle re-check interval when there is no work.
    public var idleRecheckInterval: TimeInterval
    /// Per-request event buffer before backpressure pauses scheduling.
    public var eventBufferCapacity: Int
    /// Upper bound on a graceful drain. `shutdown()` waits for running
    /// requests to finish naturally; if the engine queue is wedged (a step
    /// blocked inside an eval), the drain would otherwise never complete —
    /// after this timeout every live stream is force-finished with
    /// `.error` and `shutdown()` returns. The wedged step may still be
    /// executing in the background; the process-level owner decides
    /// whether to exit.
    public var shutdownTimeout: TimeInterval

    public init(
        requestTimeout: TimeInterval = 120, stepTimeout: TimeInterval = 30,
        watchdogInterval: TimeInterval = 0.25, idleRecheckInterval: TimeInterval = 0.001,
        eventBufferCapacity: Int = 256, shutdownTimeout: TimeInterval = 10,
        useLegacyRequestTimeout: Bool = false,
        admissionLease: TimeInterval = 120,
        prefillProgressLease: TimeInterval = 120,
        decodeProgressLease: TimeInterval = 120,
        backpressureLease: TimeInterval = 120,
        safetyCeilingDecodeFloorTPS: Double = 5,
        clock: CBv2Clock = .continuous
    ) {
        self.requestTimeout = requestTimeout
        self.stepTimeout = stepTimeout
        self.watchdogInterval = watchdogInterval
        self.idleRecheckInterval = idleRecheckInterval
        self.eventBufferCapacity = eventBufferCapacity
        self.shutdownTimeout = shutdownTimeout
        self.useLegacyRequestTimeout = useLegacyRequestTimeout
        self.admissionLease = admissionLease
        self.prefillProgressLease = prefillProgressLease
        self.decodeProgressLease = decodeProgressLease
        self.backpressureLease = backpressureLease
        self.safetyCeilingDecodeFloorTPS = safetyCeilingDecodeFloorTPS
        self.clock = clock
    }
}

// MARK: - In-flight step

/// One launched-but-not-finalized step. Its sampled tokens are still lazy;
/// finalization materializes them (the ONE host sync per step, overlapped
/// with the next step's GPU work when chained).
final class CBv2InFlightStep {
    /// Exact scheduler work launched by this step. Scheduler records already
    /// contain these optimistic advances; deadline projection subtracts them
    /// to recover confirmed cursors, then charges the full work once.
    let assignments: [(id: CBv2RequestID, numTokens: Int)]
    /// Every request that computed anything this step (KV release for any of
    /// these must be deferred until finalization — see CONTRACT-ISSUES §4).
    let participants: Set<CBv2RequestID>
    /// Exact token ranges whose KV work THIS step launched. Scheduler records
    /// may already include a chained successor when this step finalizes, so
    /// prefix publication must use these immutable bounds instead.
    let computedRanges: [CBv2RequestID: Range<Int>]
    /// Rows that sampled a token, in plan order (== row order of
    /// `sampledTokens`).
    let sampledRows: [CBv2RequestID]
    /// Lazy [K] int32, or nil when no row sampled (all mid-prefill chunks).
    let sampledTokens: MLXArray?
    /// Cheap handles that force evaluation of non-sampling prefill chunks.
    let evalTargets: [MLXArray]
    /// Rows finished/cancelled AFTER launch: their sampled token is
    /// discarded at finalization (the ≤1 wasted slot-step).
    var discard: Set<CBv2RequestID> = []
    /// Lazy per-step logprob gathers (rows that requested topLogprobs > 0
    /// exist in the batch). Graph-only until finalization, where they are
    /// materialized at the SAME boundary as the sampled tokens.
    var logprobSegments: [CBv2StepLogprobs] = []
    /// Host wall clock captured before graph construction for controller
    /// attribution at the existing finalize boundary.
    let wallStartedNanos: UInt64
    /// The ONE unconditional timing read in `finalize`, taken right after
    /// the host readback; every per-row stamp of that finalize reuses it.
    var readbackDoneNanos: UInt64 = 0
    /// Launched on the chained-decode fast path (per-request
    /// `chainedDecodeSteps` telemetry).
    var chained = false
    /// Rows that confirm tokens when this step finalizes: sampled rows plus
    /// MTP verify rows (confirmed by the accept walk, not the sampler).
    var tokenProducingRows: Int { sampledRows.count + (mtpRound?.verify?.rows.count ?? 0) }
    var mtpMeasurement: CBv2MTPStepMeasurement?
    /// KV states whose release is fenced behind this step's completion.
    /// `rollbackOne` scrubs the wasted-token KV tail before release;
    /// `donation` (non-nil for natural finishes with prefix caching on)
    /// routes the retired state through the donation queue. Normal terminal
    /// delivery follows retirement so observing it also means the request ID
    /// is reusable; watchdog terminals remain the explicit exception.
    fileprivate var deferredReleases:
        [(
            id: CBv2RequestID, state: [CBv2SequenceKV?], rollbackOne: Bool,
            donation: CBv2DonationIntent?, recurrent: CBv2RecurrentRequestState?,
            hybridPublication: CBv2DonationIntent?,
            ownership: CBv2OutputStream?, terminalDelivery: (() -> Void)?
        )] = []
    /// One recurrent transaction per participating row. A normal finalized
    /// step commits it; a one-step-late discarded chained decode rolls it
    /// back before the request state is released.
    var recurrentEvaluations: [CBv2RequestID: CBv2RecurrentStateEvaluation] = [:]
    var forwardShapes: CBv2ForwardShapeStep?
    var recurrentCheckpointChunkSizes: [CBv2RequestID: Int] = [:]
    var historicalCheckpoints: [CBv2RequestID: CBv2CapturedCompleteCheckpoint] = [:]
    var permitsChainedSuccessor: Bool {
        mtpRound == nil && historicalCheckpoints.isEmpty && attentionPacket == nil
    }
    var packedPrefixRows: Set<CBv2RequestID> = []
    /// Non-nil marks this step as an MTP round (verify and/or seed work,
    /// finalized by `finalizeMTPRound`). MTP rounds NEVER chain: the
    /// chained path's finalize loop and `deferredReleases` assume exactly
    /// one sample per row, so `engineStep` guards on this before offering
    /// the step as a chain base.
    var mtpRound: CBv2MTPRoundInFlight?
    /// Offline-only compact observations, evaluated with this step's graph.
    var logitDiagnostics: [CBv2LogitDiagnosticPacket] = []
    var attentionMetadata: CBv2AttentionMetadataForward?
    var attentionPacket: CBv2AttentionPacketForward?

    init(
        assignments: [(id: CBv2RequestID, numTokens: Int)],
        participants: Set<CBv2RequestID>, sampledRows: [CBv2RequestID],
        sampledTokens: MLXArray?, evalTargets: [MLXArray],
        computedRanges: [CBv2RequestID: Range<Int>] = [:],
        wallStartedNanos: UInt64
    ) {
        self.assignments = assignments
        self.computedRanges = computedRanges
        self.participants = participants
        self.sampledRows = sampledRows
        self.sampledTokens = sampledTokens
        self.evalTargets = evalTargets
        self.wallStartedNanos = wallStartedNanos
    }
}

// MARK: - Prefix adoption handoff

/// A prefix-cache hit prepared on the submit thread (lookup + graph-only
/// slicing) and applied on the engine thread at enqueue. The typed plan owns
/// M/C/R, backend support, and accounting facts. Ordinary safe layouts carry
/// full snapshots through C; frozen-full layouts carry them through M while
/// sliding rows begin empty at C. The lookup's in-use pin is
/// balanced by exactly one `endAdoption(tokens:matched:)` when the adoption
/// is consumed or abandoned.
/// `@unchecked Sendable`: immutable value handed off exactly once, submit
/// thread → engine queue; the MLXArray views are graph-only until adopted.
struct CBv2PrefixAdoption: @unchecked Sendable {
    /// Correlation identity used by the lookup that owns this pin. This is
    /// the request's receipt id when supplied, otherwise its scheduler id.
    let requestID: CBv2RequestID
    let tokens: [Int]
    let matched: Int
    let plan: CBv2PrefixReusePlan
    let prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?]
    /// Request-scoped salt the lookup used (TB-007); the pin release and any
    /// fallback paths must carry the SAME salt or the pin leaks.
    let cacheSalt: String?
    var recurrentCheckpoint: CBv2RecurrentCheckpoint? = nil
    var hybridPin: UInt64? = nil
    var completeCheckpoint: CBv2StagedCompleteCheckpoint? = nil
}

/// Submit-thread lookup result handed to the engine queue. `outcome` is
/// provisional when `adoption != nil`; `applyAdoption` resolves it to a real
/// hit or a precise adoption failure before usage is emitted.
struct CBv2PrefixLookup: @unchecked Sendable {
    let adoption: CBv2PrefixAdoption?
    let outcome: CBv2PrefixCacheOutcome
    let matchedTokens: Int
    /// Submit-thread lookup duration (`CBv2RequestTiming.prefixLookupNanos`).
    var lookupNanos: UInt64 = 0
}

struct CBv2PrefixUsage {
    var outcome: CBv2PrefixCacheOutcome
    var tier: CBv2PrefixCacheTier?
    var matchedTokens: Int
    var prefillTokensSaved: Int
    var strategy: CBv2PrefixReuseStrategy?
    var replayTokens: Int
    var boundarySplits: Int

    static let disabled = CBv2PrefixUsage(
        outcome: .disabled,
        tier: nil,
        matchedTokens: 0,
        prefillTokensSaved: 0,
        strategy: nil,
        replayTokens: 0,
        boundarySplits: 0)
}

enum CBv2FirstTokenDeadlineEnqueueOutcome: Sendable {
    case admitted(
        CBv2FirstTokenProjectedWork,
        admittedAt: ContinuousClock.Instant
    )
    case deadlineUnreachable(CBv2FirstTokenProjectedWork)
    case cancelled
    case schedulerRejected(CBv2SchedulerError)
    case capacityRejected
}

/// Exactly-once handoff for async deadline submission. The engine queue,
/// caller cancellation, and watchdog/shutdown failure paths can race; the
/// winner resumes the caller and releases its pending-submit gauge.
private final class CBv2FirstTokenDeadlineWaiter: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation:
        CheckedContinuation<CBv2FirstTokenDeadlineEnqueueOutcome, Never>?
    private let onResume: @Sendable () -> Void

    init(
        _ continuation: CheckedContinuation<CBv2FirstTokenDeadlineEnqueueOutcome, Never>,
        onResume: @escaping @Sendable () -> Void
    ) {
        self.continuation = continuation
        self.onResume = onResume
    }

    @discardableResult
    func resume(returning outcome: CBv2FirstTokenDeadlineEnqueueOutcome) -> Bool {
        lock.lock()
        let continuation = continuation
        self.continuation = nil
        lock.unlock()
        guard let continuation else { return false }
        onResume()
        continuation.resume(returning: outcome)
        return true
    }
}

/// Lock-confined ownership state for one asynchronous deadline admission.
///
/// Cancellation is a requested transition, not completion: only the engine
/// queue may acknowledge it after discarding any scheduler/KV state it could
/// have created. Stream generation ownership stays live until that
/// acknowledgement, preventing immediate ID reuse from colliding with stale
/// queued work.
private final class CBv2FirstTokenDeadlineOperation: @unchecked Sendable {
    enum Phase: Equatable {
        case queued
        case executing
        case cancellationRequested
        /// Watchdog/shutdown already resumed the caller, but the engine queue
        /// still owns cleanup if it later makes progress.
        case failedExternally
        case completed
    }

    let generation: UInt64
    let waiter: CBv2FirstTokenDeadlineWaiter
    var phase: Phase = .queued

    init(generation: UInt64, waiter: CBv2FirstTokenDeadlineWaiter) {
        self.generation = generation
        self.waiter = waiter
    }
}

/// A request's donation intent: which exact token prefix to donate and under
/// which salt scope (TB-007 — the entry must be indexed with the same salt
/// its donor request carried, or a differently-salted request could hit it).
struct CBv2DonationIntent {
    let requestID: CBv2RequestID
    let tokens: [Int]
    let cacheSalt: String?
    var receiptID: CBv2RequestID? = nil
    var allowsCompletePublication = true
    var retiredReservation: CBv2CheckpointReservation? = nil
}

/// Single-consumer cross-queue handoff of engine-owned values (KV state,
/// donation snapshots). Safe by ownership transfer: the sender never
/// touches the value after enqueueing.
private struct CBv2Handoff<Value>: @unchecked Sendable {
    let value: Value
}

/// Resume-exactly-once wrapper for a drain continuation: the engine queue
/// (natural drain completion) and the shutdown-timeout timer race to
/// resume it; whichever wins consumes the continuation, the loser no-ops.
private final class CBv2DrainWaiter: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?

    init(_ continuation: CheckedContinuation<Void, Never>) {
        self.continuation = continuation
    }

    /// Resume if not already resumed; returns true when this call won.
    @discardableResult
    func resume() -> Bool {
        lock.lock()
        let c = continuation
        continuation = nil
        lock.unlock()
        c?.resume()
        return c != nil
    }
}

// MARK: - EngineLoopV2

/// The engine thread. All scheduler and MLX mutations happen on
/// `engineQueue` (serial, self-rescheduling — the EngineCore idiom that
/// keeps the GPU pipeline full with only GCD dispatch overhead between
/// steps). Cross-thread surface: `requestCancel`, `setPaused`, stream
/// registration, and the watchdog — all lock-protected.
public final class EngineLoopV2: @unchecked Sendable {
    let scheduler: SchedulerV2
    let capacity: CBv2StepCapacity?
    let backend: CBv2KVBackend
    let cacheProvider: CBv2LayerCacheProvider
    let model: CBv2SteppableModel
    let sampler: CBv2StepSampler
    let detokenizerFactory: CBv2DetokenizerFactory
    let layerKinds: [CBv2LayerKind]
    /// Non-nil only when prefix caching is active (instance supplied AND
    /// `CBv2SchedulerConfig.enablePrefixCache`).
    let prefixCache: CBv2PrefixCache?
    let residentPrefixBackend: (any CBv2PagedPrefixSharingBackend)?
    let prefixReuseCapability: CBv2PrefixReuseCapability
    let nominalFullKVBytesPerToken: Int
    /// MTP (speculative decoding) driver state, or nil (byte-identical
    /// plain-decode behavior). Round logic lives in EngineLoopV2+MTP.swift.
    let mtp: CBv2MTPRoundDriver?
    let config: CBv2EngineLoopConfig
    let gauges: CBv2EngineGauges

    private let engineQueue = DispatchQueue(
        label: "com.eigen.cbv2.engine", qos: .userInitiated)
    private let watchdogQueue = DispatchQueue(
        label: "com.eigen.cbv2.watchdog", qos: .utility)
    /// Prefix-cache donation runs here (hashing + indexing + optional device
    /// materialization) — never on the engine step thread (invariant 6).
    /// Safe to eval CONCURRENTLY with engine-step evals over the same paged
    /// slabs: the snapshot views were graph-built on the engine thread, so
    /// MLX array versioning pins them to the slab contents as of that step
    /// (later slab writes produce new versions, never mutating pinned
    /// ones), and the release-after-donate ordering in `retire` keeps the
    /// donor's pages out of the pool until the donation has materialized.
    private let donationQueue = DispatchQueue(
        label: "com.eigen.cbv2.donation", qos: .utility)
    /// Detokenization (Tokenizer.decode + UTF-8 holdback) + stream emission
    /// for PASSTHROUGH requests (no stop strings) runs here, OFF the serial
    /// step thread — that host string work is bounded per token but at B>1
    /// it otherwise sits on the critical path between GPU steps for every
    /// row (PR#62 review). A single serial queue preserves global FIFO
    /// order, so each request's deltas + trailing flush + terminal event
    /// stay ordered (per-request order is a subsequence of global order).
    /// Requests WITH stop strings keep the fully synchronous engine-thread
    /// path: their holdback scan gates the deterministic one-step-late
    /// stop-string finish, which cannot be deferred without generating text
    /// past the stop (contract) or breaking the parity suites.
    let detokQueue = DispatchQueue(
        label: "com.eigen.cbv2.detok", qos: .userInitiated)

    // Cross-thread state (stateLock).
    private let stateLock = NSLock()
    private var streams: [CBv2RequestID: CBv2OutputStream] = [:]
    private var streamGenerations: [CBv2RequestID: UInt64] = [:]
    private var nextStreamGeneration: UInt64 = 0
    /// Cancellation is bound to the monotonic token assigned to the current
    /// stream. An id may be reused after that stream retires; a late callback
    /// from the old stream must never cancel the new generation.
    private var pendingCancels: [CBv2RequestID: UInt64] = [:]
    private var deadlineAdmissionOperations:
        [CBv2RequestID: CBv2FirstTokenDeadlineOperation] = [:]
    private var stepStartedNanos: UInt64 = 0
    private var wedgeReported = false
    private var _healthy = true
    /// Reconciled per-request usage (prompt/completion counts) snapshotted at
    /// enqueue and refreshed at each finalize. Lock-protected so the watchdog
    /// thread — which fires while the engine thread is wedged inside a blocking
    /// eval and therefore cannot read engine-confined state — can attach the
    /// usage observed before the wedge instead of injecting raw zero usage.
    private var usageSnapshots: [CBv2RequestID: CBv2Usage] = [:]
    /// Prompt-frontier logit capture (`CBv2Engine.prefillLogitDigest`).
    /// Installed around ONE probe request and cleared immediately after, so
    /// every ordinary step pays a single uncontended lock read at the two
    /// sites where a prompt chunk actually samples. It is deliberately NOT
    /// a "digest is supported" flag: nothing reads it except the frontier
    /// itself, and when it never fires the digest call throws rather than
    /// reporting a capability.
    private var prefillFrontierCaptureHook: (@Sendable (CBv2RequestID, MLXArray) -> Void)?
    /// One-shot deterministic seam fired after a deadline closure has claimed
    /// its operation but before enqueue/adoption. Tests use it to place a
    /// cancellation in the exact formerly-orphaning window.
    private var deadlineAdmissionInitialGuardHookForTesting:
        (@Sendable (CBv2RequestID) -> Void)?
    /// One-shot seam after commit has made the scheduler row authoritative
    /// but before the admission waiter is resumed.
    private var deadlineAdmissionCommittedHookForTesting:
        (@Sendable (CBv2RequestID) -> Void)?

    // Engine-thread-confined state (internal, not private: the MTP round
    // driver in EngineLoopV2+MTP.swift is part of the loop).
    var detokenizers: [CBv2RequestID: CBv2IncrementalDetokenizer] = [:]
    var kvStates: [CBv2RequestID: [CBv2SequenceKV?]] = [:]
    var recurrentStates: [CBv2RequestID: CBv2RecurrentRequestState] = [:]
    var forwardShapeRecorder: CBv2ForwardShapeRecorder?
    var buildingForwardShapes: CBv2ForwardShapeStep?
    /// Tokens skipped via prefix-cache adoption, reported in usage.
    var prefixHitTokens: [CBv2RequestID: Int] = [:]
    /// Lookup/adoption outcome carried to terminal usage.
    var prefixUsageByID: [CBv2RequestID: CBv2PrefixUsage] = [:]
    /// Per-request incremental hash/publication state for the page-native L1.
    /// Entries exist only for eligible text requests and are engine-confined.
    var residentPrefixCursorByID: [CBv2RequestID: CBv2PagedPrefixCursor] = [:]
    let hybridPrefixCache: CBv2HybridPrefixCache?
    let completeCheckpointCapture: CBv2CompleteCheckpointCapture?
    var recurrentCheckpointGeometry: [CBv2RequestID: CBv2RecurrentCheckpointGeometry] = [:]
    /// Donation work that currently owns a retired backend state. Natural
    /// drain cannot complete until each terminal donation releases its state
    /// back on the engine queue.
    private var pendingDonationReleaseCount = 0
    /// Resolved vision inputs (validated spans + materialized embeddings),
    /// keyed by request. Kept across PREEMPTION (a full re-prefill replays
    /// the span chunks and needs the embeddings again); dropped at finish.
    var multimodalByID: [CBv2RequestID: CBv2ResolvedMultimodal] = [:]
    /// Monotonic per-request deadline leases (admission / prefill / decode /
    /// backpressure / safety, or the legacy single wall under the kill-switch).
    /// Engine-thread-confined. See `CBv2DeadlineLeases.swift`.
    var leasesByID: [CBv2RequestID: CBv2RequestLeaseState] = [:]
    /// Rollback-path preemption victims whose lease rewind (`markPreempted`)
    /// is deferred until after the pending finalize confirms their in-flight
    /// sample — finalization's progress refresh would otherwise flip the lease
    /// back to `.decode` and undo the rewind (PR#82 review). Engine-thread-
    /// confined; drained at the finalize site every step.
    var leasePreemptionsPendingFinalize: [CBv2RequestID] = []
    private var inFlight: CBv2InFlightStep?
    private var running = false
    /// Nil in production. Configuration and records are engine-queue confined.
    var logitDiagnostic: CBv2LogitDiagnosticState?
    var attentionMetadata: CBv2AttentionMetadataState?
    var attentionPacket: CBv2AttentionPacketState?
    private var draining = false
    private var drainWaiters: [CBv2DrainWaiter] = []
    /// True after a rejecting MTP round advanced rows OUTSIDE the eager
    /// provider's caches' host truth: the next eager bind must be forced to
    /// rebuild `positionOffsets` from host truth (see `eagerCaches`).
    /// Sole writer: `mtpFinalize` in EngineLoopV2+MTPFinalize.swift.
    var eagerCompositionStale = false

    /// Telemetry / test hooks.
    public private(set) var stepCount = 0
    public private(set) var chainedStepCount = 0
    public private(set) var preemptionCount = 0
    /// Cumulative step telemetry published on `CBv2CapacitySnapshot`
    /// (`publishGauges`): Σ launch→readback-done wall nanos over finalized
    /// steps, and Σ rows that confirmed a non-first token. Engine-thread
    /// owned, monotonic, two plain `+=` per finalize. (internal(set): the
    /// MTP accept walk in EngineLoopV2+MTPFinalize.swift counts here too.)
    public private(set) var stepWallNanosTotal: UInt64 = 0
    public internal(set) var decodeRowsTotal: UInt64 = 0
    /// Timing-read reuse windows (engine thread only). `launchClockNanos`
    /// is the in-progress launch's `wallStartedNanos` (so `ensureKVState`
    /// stamps KV allocation without its own read); `finalizeClockNanos` is
    /// the in-progress finalize's readback-done instant (so finishes
    /// driven by that finalize stamp `finishedNanos` without a read).
    /// Zero outside those windows, where a fresh read is taken instead.
    /// `boundaryClockNanos` covers step-boundary finishes (cancel / lease
    /// expiry): ONE lazily-taken read per step shared by every row finished
    /// there, none when nothing finishes; reset at the top of `engineStep`.
    var launchClockNanos: UInt64 = 0
    var finalizeClockNanos: UInt64 = 0
    var boundaryClockNanos: UInt64 = 0

    @inline(__always)
    private func boundaryFinishNanos() -> UInt64 {
        if boundaryClockNanos == 0 {
            boundaryClockNanos = DispatchTime.now().uptimeNanoseconds
        }
        return boundaryClockNanos
    }
    /// Instruments "step" intervals. Zero cost unless a signpost consumer is
    /// attached (`isEnabled` guards both ends).
    static let signposter = OSSignposter(subsystem: "com.darkbloom.engine", category: "cbv2")
    /// Packed-prefill EXECUTION evidence (`CBv2PackedPrefillActivity`).
    /// Incremented in `executeMixed` at the rectangular forward itself, so a
    /// capability that is claimed but never exercised leaves both at zero.
    /// Two plain `+=` per packed group — no allocation, no locking on the
    /// step path. Engine-thread owned, monotonic; read at quiescent points.
    public private(set) var packedPrefillRowsExecuted = 0
    public private(set) var packedPrefillGroupsExecuted = 0
    /// The two capability gates `executeMixed` consults before it packs —
    /// the caches vouch for per-row independence AND the model's prompt
    /// forward is batch-generic. Sole reader of the gates, so the flag the
    /// engine reports and the flag it acts on cannot drift. Configuration
    /// only: see the counters above for whether anything packed.
    var packedPrefillSupported: Bool {
        cacheProvider.supportsPackedPrefill
            && (model as? CBv2PackedPrefillSteppableModel)?.supportsPackedPrefill == true
    }

    /// Capability + cumulative execution evidence, as `EngineV2` republishes
    /// it to out-of-module callers.
    func packedPrefillActivity() -> CBv2PackedPrefillActivity {
        engineQueue.sync {
            CBv2PackedPrefillActivity(
                isSupported: packedPrefillSupported,
                rowsExecuted: packedPrefillRowsExecuted,
                groupsExecuted: packedPrefillGroupsExecuted)
        }
    }

    /// Teacher-forced scoring EXECUTION evidence (`teacherForcedTop1`).
    /// Incremented at the forwards themselves — the prompt's chunked
    /// prefill and the continuation's one-token decodes — so a witness that
    /// scored positions OUTSIDE the engine (batched offline logits, a
    /// reference implementation) leaves them at zero while still returning
    /// plausible argmaxes. The counters are the only proof the continuation
    /// really travelled the engine's own caches, chunking, and masks.
    /// Engine-thread owned, monotonic; read at quiescent points.
    private(set) var teacherForcedPrefillChunks = 0
    private(set) var teacherForcedDecodeForwards = 0

    /// Cumulative teacher-forced execution evidence, as `EngineV2`
    /// republishes it to out-of-module callers.
    func teacherForcedScoringActivity() -> CBv2TeacherForcedScoringActivity {
        engineQueue.sync {
            CBv2TeacherForcedScoringActivity(
                prefillChunksExecuted: teacherForcedPrefillChunks,
                decodeForwardsExecuted: teacherForcedDecodeForwards)
        }
    }

    /// Requests demoted back to waiting after a capacityExhausted at first
    /// allocation (test/telemetry hook), and the per-request attempt cap.
    private(set) var capacityRequeueCount = 0
    private var capacityRequeues: [CBv2RequestID: Int] = [:]
    static let maxCapacityRequeues = 64
    /// Steps that submitted eager layer-cache inner state (the offset chain +
    /// KV buffers) into the step's `asyncEval` set. The DAR-325 guard:
    /// evaluating the offset chain every eager step keeps its lazy `+ L`
    /// advance from accumulating O(steps) of graph. Zero for caches that
    /// vend no inner state (mocks). Test hook. (internal(set): MTP round
    /// steps in EngineLoopV2+MTP.swift count here too.)
    public internal(set) var offsetChainEvalSteps = 0
    /// Fired (from the watchdog thread) when a step exceeds `stepTimeout`.
    public var onStepWedge: (@Sendable (TimeInterval) -> Void)?
    /// Test hook: artificial delay at the START of the enqueue engine block,
    /// before the request reaches the scheduler. Widens the early-cancel race
    /// window (a cancel arriving after `submit` registered the stream but
    /// before enqueue ran) so the regression is deterministic. Zero in
    /// production. Set before submitting.
    var enqueueStartDelayForTesting: TimeInterval = 0
    /// Engine-queue test gate. Once `stepCount` reaches this value, scheduled
    /// step callbacks yield through the idle recheck path without finalizing
    /// or planning more work; admission closures on the same queue still run.
    /// nil in production. Set and clear only through `onEngineQueueSync`.
    var suspendStepExecutionAtCountForTesting: Int?

    public var isHealthy: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return _healthy
    }

    init(
        model: CBv2SteppableModel,
        layerKinds: [CBv2LayerKind],
        backend: CBv2KVBackend,
        cacheProvider: CBv2LayerCacheProvider,
        sampler: CBv2StepSampler,
        detokenizerFactory: CBv2DetokenizerFactory,
        scheduler: SchedulerV2,
        capacity: CBv2StepCapacity?,
        prefixCache: CBv2PrefixCache? = nil,
        residentPrefixBackend: (any CBv2PagedPrefixSharingBackend)? = nil,
        hybridPrefixCache: CBv2HybridPrefixCache? = nil,
        completeCheckpointCapture: CBv2CompleteCheckpointCapture? = nil,
        prefixReuseCapability: CBv2PrefixReuseCapability? = nil,
        nominalFullKVBytesPerToken: Int? = nil,
        mtp: CBv2MTPRoundDriver? = nil,
        config: CBv2EngineLoopConfig,
        gauges: CBv2EngineGauges
    ) {
        self.model = model
        self.layerKinds = layerKinds
        self.backend = backend
        self.cacheProvider = cacheProvider
        self.sampler = sampler
        self.detokenizerFactory = detokenizerFactory
        self.scheduler = scheduler
        self.capacity = capacity
        self.prefixCache = prefixCache
        self.residentPrefixBackend = residentPrefixBackend
        self.hybridPrefixCache = hybridPrefixCache
        self.completeCheckpointCapture = completeCheckpointCapture
        let resolvedPrefixCapability = prefixReuseCapability ?? CBv2PrefixReuseCapability.derive(
            layerKinds: layerKinds, backend: backend.prefixReuseBackend)
        self.prefixReuseCapability = resolvedPrefixCapability
        self.nominalFullKVBytesPerToken =
            nominalFullKVBytesPerToken ?? resolvedPrefixCapability.fullKVBytesPerToken
        self.mtp = mtp
        self.config = config
        self.gauges = gauges
        // MTP: the scheduler consults the loop for 1+k decode assignments.
        // `unowned` is safe (and cycle-free): the loop owns the scheduler
        // and both live exactly as long as the engine.
        if let mtp {
            scheduler.speculationPlanner = { [unowned self] rec in
                self.mtpPlanSpeculation(for: rec)
            }
            scheduler.speculationDraftTokenUpperBound = max(
                0,
                mtp.config.fixedDraftTokens
                    ?? mtp.config.maxDraftTokens)
        }
    }

    // MARK: Lifecycle

    func start() {
        engineQueue.async { [self] in
            guard !running else { return }
            running = true
            startWatchdog()
            engineQueue.async { [weak self] in self?.engineStep() }
        }
    }

    /// Graceful drain: waiting requests are cancelled, running requests
    /// finish naturally, then the loop stops. Idempotent.
    ///
    /// Bounded by `config.shutdownTimeout`: the drain waits on the engine
    /// queue, so a wedged queue (a step blocked inside an eval) would hang
    /// it forever. The timeout runs on the watchdog queue; when it wins,
    /// every live stream is force-finished with `.error`, live requests
    /// are marked for cancellation (cleaned up if the loop ever resumes),
    /// and `drain()` returns — the wedged step may still be executing.
    func drain() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            let waiter = CBv2DrainWaiter(c)
            watchdogQueue.asyncAfter(deadline: .now() + config.shutdownTimeout) { [weak self] in
                guard let self else {
                    waiter.resume()
                    return
                }
                if waiter.resume() {
                    self.forceFinishStreamsOnShutdownTimeout()
                }
            }
            engineQueue.async { [self] in
                guard running else {
                    waiter.resume()
                    return
                }
                draining = true
                // One read (per clock domain) for every waiting row the
                // drain cancels.
                let drainNanos = DispatchTime.now().uptimeNanoseconds
                let drainNow = config.clock.now()
                for rec in scheduler.waiting {
                    finishRequest(
                        rec.id, reason: .cancelled, nowNanos: drainNanos, now: drainNow)
                }
                publishGauges()
                drainWaiters.append(waiter)
                completeDrainIfReady()
            }
        }
    }

    /// Shutdown-timeout path (watchdog queue): the engine queue is not
    /// making progress, so touch ONLY lock-protected state and the
    /// (thread-safe) streams — the same discipline as `watchdogTick()`.
    /// Stream `finish` is idempotent, so a later natural finish (if the
    /// loop ever resumes) is a harmless no-op.
    private func forceFinishStreamsOnShutdownTimeout() {
        stateLock.lock()
        let liveStreams = streams
        let deadlineOperations = Array(deadlineAdmissionOperations.values)
        for operation in deadlineOperations where operation.phase != .completed {
            operation.phase = .failedExternally
        }
        for id in liveStreams.keys {
            if let generation = streamGenerations[id] {
                pendingCancels[id] = generation
            }
        }
        stateLock.unlock()
        for (_, stream) in liveStreams {
            stream.finish(
                reason: .error(
                    "engine shutdown timed out after \(Int(config.shutdownTimeout))s"),
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
        }
        for operation in deadlineOperations {
            operation.waiter.resume(returning: .capacityRejected)
        }
    }

    private func completeStop() {
        mtp?.removeAllRequestState()
        logitDiagnostic = nil
        attentionMetadata?.discardPendingForward()
        attentionMetadata = nil
        attentionPacket?.discardPendingForward()
        attentionPacket = nil
        running = false
        draining = false
        stopWatchdog()
    }

    /// Natural shutdown barrier. Donation completion hops back to the engine
    /// queue and calls this, so a drain with no scheduler work wakes promptly
    /// instead of waiting for another polling step.
    private func completeDrainIfReady() {
        guard draining, !scheduler.hasWork, inFlight == nil,
            pendingDonationReleaseCount == 0
        else { return }
        completeStop()
        let waiters = drainWaiters
        drainWaiters = []
        for waiter in waiters { waiter.resume() }
    }

    // MARK: Submission (from EngineV2)

    /// Register the stream for a new request. Fails (returns false, state
    /// untouched) when a stream with the same id is still live: a duplicate
    /// `submit` must never REPLACE the original request's stream — the
    /// scheduler would later reject the duplicate id, and the rejection
    /// path's `takeStream` would tear down the replacement, leaving the
    /// FIRST request's deltas and terminal event with nowhere to go
    /// (PR#62 review). Ids only become reusable after backend ownership
    /// retirement removes the previous stream generation.
    func register(stream: CBv2OutputStream) -> Bool {
        registerWithGeneration(stream: stream) != nil
    }

    func registerWithGeneration(stream: CBv2OutputStream) -> UInt64? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard streams[stream.id] == nil else { return nil }
        nextStreamGeneration &+= 1
        if nextStreamGeneration == 0 { nextStreamGeneration = 1 }
        streams[stream.id] = stream
        streamGenerations[stream.id] = nextStreamGeneration
        return nextStreamGeneration
    }

    /// Submit-side unwind: remove a stream registered by `submit` whose
    /// request never reached `enqueue` (multimodal materialization failed
    /// after registration — PR#63 review). Only legal BEFORE `enqueue`; a
    /// live request's stream is removed when backend ownership retires.
    func unregister(_ id: CBv2RequestID) {
        stateLock.lock()
        let stream = streams.removeValue(forKey: id)
        streamGenerations.removeValue(forKey: id)
        stateLock.unlock()
        stream?.releaseEngineOwnership()
    }

    @discardableResult
    private func registerDeadlineAdmissionOperation(
        _ operation: CBv2FirstTokenDeadlineOperation,
        for id: CBv2RequestID
    ) -> Bool {
        stateLock.lock()
        precondition(
            deadlineAdmissionOperations[id] == nil,
            "deadline admission operation already registered for \(id)")
        guard let currentStream = streams[id],
            streamGenerations[id] == operation.generation
        else {
            stateLock.unlock()
            operation.waiter.resume(returning: .cancelled)
            return false
        }
        if !_healthy {
            streams.removeValue(forKey: id)
            streamGenerations.removeValue(forKey: id)
            if pendingCancels[id] == operation.generation {
                pendingCancels.removeValue(forKey: id)
            }
            stateLock.unlock()
            currentStream.releaseEngineOwnership()
            currentStream.finish(
                reason: .error("engine is unhealthy"),
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
            operation.waiter.resume(returning: .capacityRejected)
            return false
        }
        if pendingCancels[id] == operation.generation {
            operation.phase = .cancellationRequested
        } else {
            // Discard cancellation state left by an older stream generation.
            pendingCancels.removeValue(forKey: id)
        }
        deadlineAdmissionOperations[id] = operation
        stateLock.unlock()
        return true
    }

    private enum DeadlineAdmissionStart: Equatable {
        case execute
        case cancellationRequested
        case failedExternally
        case notCurrent
    }

    private func beginDeadlineAdmission(
        _ operation: CBv2FirstTokenDeadlineOperation,
        for id: CBv2RequestID
    ) -> DeadlineAdmissionStart {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard deadlineAdmissionOperations[id] === operation,
            streamGenerations[id] == operation.generation
        else {
            return .notCurrent
        }
        switch operation.phase {
        case .queued:
            operation.phase = .executing
            return .execute
        case .cancellationRequested:
            return .cancellationRequested
        case .failedExternally:
            return .failedExternally
        case .executing, .completed:
            return .notCurrent
        }
    }

    private enum DeadlineAdmissionCommit {
        case admitted
        case cancellationRequested
        case failedExternally
        case notCurrent
    }

    /// Atomically publish admission while retaining stream ownership. If a
    /// cancellation won the lock first, the engine queue must clean the row
    /// and acknowledge cancellation instead.
    private func commitDeadlineAdmission(
        _ operation: CBv2FirstTokenDeadlineOperation,
        for id: CBv2RequestID
    ) -> DeadlineAdmissionCommit {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard deadlineAdmissionOperations[id] === operation,
            streamGenerations[id] == operation.generation
        else {
            return .notCurrent
        }
        switch operation.phase {
        case .executing:
            operation.phase = .completed
            deadlineAdmissionOperations.removeValue(forKey: id)
            return .admitted
        case .cancellationRequested:
            return .cancellationRequested
        case .failedExternally:
            return .failedExternally
        case .queued, .completed:
            return .notCurrent
        }
    }

    private func deadlineAdmissionInterruption(
        _ operation: CBv2FirstTokenDeadlineOperation,
        for id: CBv2RequestID
    ) -> DeadlineAdmissionStart? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard deadlineAdmissionOperations[id] === operation,
            streamGenerations[id] == operation.generation
        else {
            return .notCurrent
        }
        switch operation.phase {
        case .executing:
            return nil
        case .cancellationRequested:
            return .cancellationRequested
        case .failedExternally:
            return .failedExternally
        case .queued, .completed:
            return .notCurrent
        }
    }

    /// Complete a non-admitted operation after engine-owned state has already
    /// been discarded. Cancellation wins over a simultaneous policy outcome.
    /// Only here are the stream generation and pending-cancel token released.
    private func completeDeadlineAdmission(
        _ id: CBv2RequestID,
        operation: CBv2FirstTokenDeadlineOperation,
        outcome intendedOutcome: CBv2FirstTokenDeadlineEnqueueOutcome
    ) {
        stateLock.lock()
        guard deadlineAdmissionOperations[id] === operation else {
            stateLock.unlock()
            operation.waiter.resume(returning: intendedOutcome)
            return
        }
        let outcome: CBv2FirstTokenDeadlineEnqueueOutcome
        switch operation.phase {
        case .cancellationRequested:
            outcome = .cancelled
        case .failedExternally:
            outcome = .capacityRejected
        case .queued, .executing, .completed:
            outcome = intendedOutcome
        }
        operation.phase = .completed
        deadlineAdmissionOperations.removeValue(forKey: id)
        let releasedStream: CBv2OutputStream?
        if streamGenerations[id] == operation.generation {
            releasedStream = streams.removeValue(forKey: id)
            streamGenerations.removeValue(forKey: id)
        } else {
            releasedStream = nil
        }
        if pendingCancels[id] == operation.generation {
            pendingCancels.removeValue(forKey: id)
        }
        stateLock.unlock()
        releasedStream?.releaseEngineOwnership()
        if case .cancelled = outcome {
            releasedStream?.finish(
                reason: .cancelled,
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
        } else {
            // Rejected streams normally have no consumer, but terminal
            // acknowledgement is still required when task cancellation races
            // the rejection handoff.
            releasedStream?.finish(
                reason: .error("deadline admission rejected"),
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
        }
        operation.waiter.resume(returning: outcome)
    }

    func setDeadlineAdmissionInitialGuardHookForTesting(
        _ hook: (@Sendable (CBv2RequestID) -> Void)?
    ) {
        stateLock.lock()
        deadlineAdmissionInitialGuardHookForTesting = hook
        stateLock.unlock()
    }

    func setDeadlineAdmissionCommittedHookForTesting(
        _ hook: (@Sendable (CBv2RequestID) -> Void)?
    ) {
        stateLock.lock()
        deadlineAdmissionCommittedHookForTesting = hook
        stateLock.unlock()
    }

    private func invokeDeadlineAdmissionInitialGuardHookForTesting(
        _ id: CBv2RequestID
    ) {
        stateLock.lock()
        let hook = deadlineAdmissionInitialGuardHookForTesting
        deadlineAdmissionInitialGuardHookForTesting = nil
        stateLock.unlock()
        hook?(id)
    }

    private func invokeDeadlineAdmissionCommittedHookForTesting(
        _ id: CBv2RequestID
    ) {
        stateLock.lock()
        let hook = deadlineAdmissionCommittedHookForTesting
        deadlineAdmissionCommittedHookForTesting = nil
        stateLock.unlock()
        hook?(id)
    }

    /// Install (nil clears) the prompt-frontier logit capture. Cross-thread
    /// safe; the loop reads it on the engine queue at the two prefill
    /// sampling sites.
    func setPrefillFrontierCapture(
        _ capture: (@Sendable (CBv2RequestID, MLXArray) -> Void)?
    ) {
        stateLock.lock()
        prefillFrontierCaptureHook = capture
        stateLock.unlock()
    }

    /// Engine-queue read of the installed capture. nil on every ordinary
    /// step.
    func prefillFrontierCapture() -> (@Sendable (CBv2RequestID, MLXArray) -> Void)? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return prefillFrontierCaptureHook
    }

    /// Run `body` on the engine queue, serialized against steps. Callers
    /// holding a captured MLX handle use this so their `eval` never races a
    /// live step's graph build on another thread. MUST NOT be called from
    /// the engine queue itself.
    func onEngineQueueSync<T>(_ body: () throws -> T) rethrows -> T {
        try engineQueue.sync(execute: body)
    }

    func configureAttentionPacket(_ configuration: CBv2AttentionPacketConfig?) throws {
        try onEngineQueueSync {
            guard !scheduler.hasWork, inFlight == nil else {
                throw CBv2AttentionPacketError.engineBusy
            }
            if let configuration {
                guard mtp == nil, layerKinds.indices.contains(configuration.storageLayerIndex)
                else { throw CBv2AttentionPacketError.unsupportedModelOrMTP }
                let kind = layerKinds[configuration.storageLayerIndex]
                guard kind.attention == .full, kind.sharesKVWithLayer == nil,
                    !kind.hasSinks, !kind.isBidirectional else {
                    throw CBv2AttentionPacketError.unsupportedModelOrMTP
                }
                attentionPacket = try CBv2AttentionPacketState(configuration)
            } else {
                attentionPacket?.discardPendingForward()
                attentionPacket = nil
            }
        }
    }

    func takeAttentionPacketSnapshot() throws -> CBv2AttentionPacketSnapshot? {
        try onEngineQueueSync {
            guard !scheduler.hasWork, inFlight == nil else {
                throw CBv2AttentionPacketError.engineBusy
            }
            return attentionPacket?.takeSnapshot()
        }
    }

    func configureAttentionMetadata(_ configuration: CBv2AttentionMetadataConfig?) throws {
        try onEngineQueueSync {
            guard !scheduler.hasWork, inFlight == nil else {
                throw CBv2AttentionMetadataError.engineBusy
            }
            if let configuration {
                guard mtp == nil, !layerKinds.isEmpty,
                    layerKinds.count <= configuration.maximumRecords,
                    layerKinds.allSatisfy({
                        $0.attention == .full && $0.sharesKVWithLayer == nil
                            && !$0.hasSinks && !$0.isBidirectional
                    }) else { throw CBv2AttentionMetadataError.unsupportedModelOrMTP }
                attentionMetadata = CBv2AttentionMetadataState(
                    configuration, expectedOwners: Set(layerKinds.indices))
            } else {
                attentionMetadata?.discardPendingForward()
                attentionMetadata = nil
            }
        }
    }

    func takeAttentionMetadataSnapshot() throws -> CBv2AttentionMetadataSnapshot? {
        try onEngineQueueSync {
            guard !scheduler.hasWork, inFlight == nil else {
                throw CBv2AttentionMetadataError.engineBusy
            }
            return attentionMetadata?.takeSnapshot()
        }
    }

    /// A stopped scheduler may still own a discarded chained successor.
    /// The first observation scope has no recorder pending-count to expose it.
    func requireIdleForwardShapeBoundary() throws {
        guard !scheduler.hasWork, inFlight == nil, buildingForwardShapes == nil else {
            throw CBv2ForwardShapeError.engineBusy
        }
    }

    func configureLogitDiagnostic(_ configuration: CBv2LogitDiagnosticConfig?) throws {
        try onEngineQueueSync {
            guard !scheduler.hasWork, inFlight == nil else {
                throw CBv2LogitDiagnosticError.engineBusy
            }
            logitDiagnostic = configuration.map(CBv2LogitDiagnosticState.init)
        }
    }

    func takeLogitDiagnosticSnapshot() throws -> CBv2LogitDiagnosticSnapshot? {
        try onEngineQueueSync {
            guard !scheduler.hasWork, inFlight == nil else {
                throw CBv2LogitDiagnosticError.engineBusy
            }
            return logitDiagnostic?.takeSnapshot()
        }
    }

    /// Runs on the engine queue. The stream must already be registered.
    /// `multimodal` is the submit-thread resolution of the request's vision
    /// input (nil for text requests) — mutually exclusive with `adoption`
    /// (vision requests never do prefix-cache lookup).
    func enqueue(
        _ request: CBv2Request,
        prefixLookup: CBv2PrefixLookup = CBv2PrefixLookup(
            adoption: nil, outcome: .disabled, matchedTokens: 0),
        residentPrefixProbe: CBv2PagedPrefixProbe? = nil,
        multimodal: CBv2ResolvedMultimodal? = nil
    ) {
        engineQueue.async { [self] in
            defer {
                gauges.endSubmit()
                publishGauges()
            }
            if enqueueStartDelayForTesting > 0 {
                Thread.sleep(forTimeInterval: enqueueStartDelayForTesting)
            }
            prefixUsageByID[request.id] = CBv2PrefixUsage(
                outcome: prefixLookup.outcome,
                tier: nil,
                matchedTokens: prefixLookup.matchedTokens,
                prefillTokensSaved: 0,
                strategy: nil,
                replayTokens: 0,
                boundarySplits: 0)
            guard running, !draining else {
                releaseAbandonedAdoption(prefixLookup.adoption)
                takeStream(request.id)?.finish(
                    reason: .error("engine is shutting down"),
                    usage: takePrefixUsage(
                        requestID: request.id, promptTokens: request.promptTokens.count,
                        completionTokens: 0))
                return
            }
            // Early-cancel race: a cancel can arrive after `submit` registered
            // the stream but BEFORE this block runs — at which point the
            // scheduler has no record for it, so `processCancellations` cannot
            // act. A pending cancel is remembered until this point and
            // consumed here so the request is never started (PR#62 review).
            if consumeEarlyCancel(request.id) {
                releaseAbandonedAdoption(prefixLookup.adoption)
                takeStream(request.id)?.finish(
                    reason: .cancelled,
                    usage: takePrefixUsage(
                        requestID: request.id, promptTokens: request.promptTokens.count,
                        completionTokens: 0))
                return
            }
            do {
                let record = try scheduler.enqueue(request)
                record.timing.prefixLookupNanos = prefixLookup.lookupNanos
                armLease(for: request)
                detokenizers[request.id] =
                    detokenizerFactory.makeDetokenizer(stopStrings: request.stopStrings)
                if let multimodal {
                    multimodalByID[request.id] = multimodal
                }
                if let residentPrefixProbe {
                    residentPrefixCursorByID[request.id] = CBv2PagedPrefixCursor(
                        hasher: residentPrefixProbe.hasher,
                        chainHashes: residentPrefixProbe.chainHashes,
                        publishedBlockCount: 0)
                }
                let residentMatch = residentPrefixProbe.flatMap {
                    residentPrefixBackend?.longestResidentPrefix(for: $0)
                }
                let snapshotMatched = prefixLookup.adoption?.matched ?? 0
                if let residentMatch, residentMatch.matchedTokens >= snapshotMatched {
                    switch applyResidentAdoption(residentMatch, requestID: request.id) {
                    case .adopted:
                        // The snapshot/SSD lookup may own a staging pin. L1 won
                        // the tie, so balance the losing tier immediately.
                        releaseAbandonedAdoption(prefixLookup.adoption)
                    case .refused:
                        if let adoption = prefixLookup.adoption {
                            prefixUsageByID[request.id]?.matchedTokens = adoption.matched
                            applyAdoption(adoption, requestID: request.id)
                        }
                    }
                } else if let adoption = prefixLookup.adoption {
                    applyAdoption(adoption, requestID: request.id)
                }
            } catch let error as CBv2SchedulerError {
                releaseAbandonedAdoption(prefixLookup.adoption)
                // Contract violation (duplicate live request id) — surfaced
                // distinctly from capacity backpressure.
                takeStream(request.id)?.finish(
                    reason: .error("scheduler rejected request: \(error)"),
                    usage: takePrefixUsage(
                        requestID: request.id, promptTokens: request.promptTokens.count,
                        completionTokens: 0))
            } catch {
                releaseAbandonedAdoption(prefixLookup.adoption)
                // Precise maxWaiting enforcement (the submit-side gauge check
                // is the fast path; this is the authoritative one). The
                // MESSAGE IS A CONTRACT: it must classify exactly like
                // `EngineV2.submit`'s thrown queue-full sentinel
                // (`capacityExhausted(needed: 1, available: 0)`), which the
                // provider renders as this canonical string and maps to a
                // retryable 429 (`MultiModelBatchSchedulerEngineError
                // .fromSchedulerMessage` matches "queue full"). A divergent
                // wording here classified the SAME condition as a 500
                // (PR#62 review).
                takeStream(request.id)?.finish(
                    reason: .error("token_budget_exhausted: request queue full"),
                    usage: takePrefixUsage(
                        requestID: request.id, promptTokens: request.promptTokens.count,
                        completionTokens: 0))
            }
        }
    }

    /// Atomic first-token deadline admission.
    ///
    /// The queue position is load-bearing: enqueue establishes authoritative
    /// priority/FCFS order. Prefix metadata establishes the replay start for a
    /// non-materializing projection; paged slabs and KV reservations are
    /// touched only after that projection is accepted. A reachable row is
    /// activated before this block returns; an unreachable row is removed
    /// before any later `engineStep` can plan GPU work for it.
    func enqueueForFirstTokenDeadline(
        _ request: CBv2Request,
        prefixLookup: CBv2PrefixLookup,
        residentPrefixProbe: CBv2PagedPrefixProbe? = nil,
        admission: CBv2FirstTokenDeadlineAdmission
    ) async -> CBv2FirstTokenDeadlineEnqueueOutcome {
        guard let submissionGeneration = registeredStreamGeneration(for: request.id) else {
            releaseAbandonedAdoption(prefixLookup.adoption)
            gauges.endSubmit()
            return .cancelled
        }
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                let waiter = CBv2FirstTokenDeadlineWaiter(
                    continuation,
                    onResume: { [gauges] in gauges.endSubmit() })
                let operation = CBv2FirstTokenDeadlineOperation(
                    generation: submissionGeneration,
                    waiter: waiter)
                guard registerDeadlineAdmissionOperation(operation, for: request.id) else {
                    releaseAbandonedAdoption(prefixLookup.adoption)
                    return
                }

                engineQueue.async { [self] in
                    func complete(
                        _ outcome: CBv2FirstTokenDeadlineEnqueueOutcome
                    ) {
                        completeDeadlineAdmission(
                            request.id,
                            operation: operation,
                            outcome: outcome)
                        // The waiter synchronously retires pendingSubmits
                        // before resuming its caller. Publish only afterward,
                        // so a committed scheduler row is never double-counted
                        // as both pending and waiting.
                        publishGauges()
                    }

                    switch beginDeadlineAdmission(operation, for: request.id) {
                    case .execute:
                        break
                    case .cancellationRequested:
                        releaseAbandonedAdoption(prefixLookup.adoption)
                        complete(.cancelled)
                        return
                    case .failedExternally, .notCurrent:
                        releaseAbandonedAdoption(prefixLookup.adoption)
                        complete(.capacityRejected)
                        return
                    }

                    // The operation is now owned by this closure, but no
                    // scheduler/prefix state exists yet. Cancellation in this
                    // exact window used to remove the stream and let a reused
                    // ID race the stale closure.
                    invokeDeadlineAdmissionInitialGuardHookForTesting(request.id)
                    if let interruption = deadlineAdmissionInterruption(
                        operation, for: request.id)
                    {
                        releaseAbandonedAdoption(prefixLookup.adoption)
                        complete(
                            interruption == .cancellationRequested
                                ? .cancelled : .capacityRejected)
                        return
                    }

                    prefixUsageByID[request.id] = CBv2PrefixUsage(
                        outcome: prefixLookup.outcome,
                        tier: nil,
                        matchedTokens: prefixLookup.matchedTokens,
                        prefillTokensSaved: 0,
                        strategy: nil,
                        replayTokens: 0,
                        boundarySplits: 0)

                    guard running, !draining else {
                        releaseAbandonedAdoption(prefixLookup.adoption)
                        stream(for: request.id)?.finish(
                            reason: .error("engine is shutting down"),
                            usage: takePrefixUsage(
                                requestID: request.id,
                                promptTokens: request.promptTokens.count,
                                completionTokens: 0))
                        complete(.capacityRejected)
                        return
                    }
                    if let interruption = deadlineAdmissionInterruption(
                        operation, for: request.id)
                    {
                        releaseAbandonedAdoption(prefixLookup.adoption)
                        prefixUsageByID.removeValue(forKey: request.id)
                        complete(
                            interruption == .cancellationRequested
                                ? .cancelled : .capacityRejected)
                        return
                    }

                    do {
                        let record = try scheduler.enqueue(request)
                        record.timing.prefixLookupNanos = prefixLookup.lookupNanos
                        let residentMatch = residentPrefixProbe.flatMap {
                            residentPrefixBackend?.longestResidentPrefix(for: $0)
                        }
                        let residentPlan = residentMatch.flatMap {
                            residentReusePlan(for: $0, request: request)
                        }
                        let useResident = residentPlan != nil
                            && (residentMatch?.matchedTokens ?? 0)
                                >= (prefixLookup.adoption?.matched ?? 0)
                        let previewPlan = useResident ? residentPlan : prefixLookup.adoption?.plan
                        let hasPrefixPreview = previewPlan != nil
                        if let previewPlan {
                            // Projection needs only immutable plan metadata.
                            // Installing KV here would wire a deferred paged
                            // pool even when the deadline verdict rejects.
                            record.numComputedTokens = previewPlan.replayStart
                            record.prefixReusePlan = previewPlan
                        }

                        let projection = scheduler.firstTokenWorkProjection(
                            for: request.id,
                            inFlightAssignments: inFlight?.assignments ?? [],
                            inFlightAllowsChainedSuccessor:
                                inFlight?.permitsChainedSuccessor ?? true,
                            unmaterializedPrefixAdoption: hasPrefixPreview)
                        let projectedWork = firstTokenProjectedWork(
                            projection,
                            admission: admission)
                        if hasPrefixPreview {
                            // `applyAdoption` owns the real cursor transition.
                            // Clear the projection-only view before either
                            // materializing or discarding the scheduler row.
                            record.numComputedTokens = 0
                            record.prefixReusePlan = nil
                        }
                        let reachable: Bool
                        switch projectedWork {
                        case .bounded(_, let serviceDuration):
                            // Authoritative monotonic clock read at the atomic
                            // verdict. Queue wait and projection CPU time have
                            // consumed the original absolute budget.
                            let now = config.clock.now()
                            reachable =
                                now < admission.deadline
                                && serviceDuration <= admission.deadline - now
                        case .unbounded:
                            reachable = false
                        }

                        guard reachable else {
                            releaseAbandonedAdoption(prefixLookup.adoption)
                            discardDeadlineRejectedRequest(request.id)
                            complete(.deadlineUnreachable(projectedWork))
                            return
                        }

                        if let interruption = deadlineAdmissionInterruption(
                            operation, for: request.id)
                        {
                            releaseAbandonedAdoption(prefixLookup.adoption)
                            discardDeadlineRejectedRequest(request.id)
                            complete(
                                interruption == .cancellationRequested
                                    ? .cancelled : .capacityRejected)
                            return
                        }

                        if let residentPrefixProbe {
                            residentPrefixCursorByID[request.id] = CBv2PagedPrefixCursor(
                                hasher: residentPrefixProbe.hasher,
                                chainHashes: residentPrefixProbe.chainHashes,
                                publishedBlockCount: 0)
                        }
                        let adopted: Bool
                        if useResident, let residentMatch {
                            releaseAbandonedAdoption(prefixLookup.adoption)
                            switch applyResidentAdoption(residentMatch, requestID: request.id) {
                            case .adopted: adopted = true
                            case .refused: adopted = false
                            }
                        } else if let adoption = prefixLookup.adoption {
                            adopted = applyAdoption(adoption, requestID: request.id)
                        } else {
                            adopted = true
                        }
                        if !adopted {
                            // The deadline verdict was valid for the prefix
                            // metadata, but the backend could not realize it.
                            // Do not silently cold-prefill under a work bound
                            // that no longer describes this request.
                            discardDeadlineRejectedRequest(request.id)
                            stream(for: request.id)?.finish(
                                reason: .error("token_budget_exhausted: prefix adoption failed"),
                                usage: CBv2Usage(
                                    promptTokens: request.promptTokens.count,
                                    completionTokens: 0))
                            complete(.capacityRejected)
                            return
                        }

                        switch commitDeadlineAdmission(operation, for: request.id) {
                        case .admitted:
                            let admittedAt = config.clock.now()
                            invokeDeadlineAdmissionCommittedHookForTesting(request.id)
                            armLease(for: request)
                            detokenizers[request.id] =
                                detokenizerFactory.makeDetokenizer(
                                    stopStrings: request.stopStrings)
                            waiter.resume(
                                returning: .admitted(
                                    projectedWork,
                                    admittedAt: admittedAt))
                            publishGauges()
                        case .cancellationRequested:
                            discardDeadlineRejectedRequest(request.id)
                            complete(.cancelled)
                        case .failedExternally, .notCurrent:
                            discardDeadlineRejectedRequest(request.id)
                            complete(.capacityRejected)
                        }
                    } catch let error as CBv2SchedulerError {
                        releaseAbandonedAdoption(prefixLookup.adoption)
                        stream(for: request.id)?.finish(
                            reason: .error("scheduler rejected request: \(error)"),
                            usage: takePrefixUsage(
                                requestID: request.id,
                                promptTokens: request.promptTokens.count,
                                completionTokens: 0))
                        complete(.schedulerRejected(error))
                    } catch {
                        releaseAbandonedAdoption(prefixLookup.adoption)
                        stream(for: request.id)?.finish(
                            reason: .error("token_budget_exhausted: request queue full"),
                            usage: takePrefixUsage(
                                requestID: request.id,
                                promptTokens: request.promptTokens.count,
                                completionTokens: 0))
                        complete(.capacityRejected)
                    }
                }

                if Task.isCancelled {
                    requestCancel(
                        request.id,
                        streamGeneration: submissionGeneration)
                }
            }
        } onCancel: { [weak self] in
            self?.requestCancel(
                request.id,
                streamGeneration: submissionGeneration)
        }
    }

    func registeredStreamGeneration(
        for id: CBv2RequestID
    ) -> UInt64? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return streamGenerations[id]
    }

    private func firstTokenProjectedWork(
        _ projection: CBv2FirstTokenWorkProjection,
        admission policy: CBv2FirstTokenDeadlineAdmission
    ) -> CBv2FirstTokenProjectedWork {
        switch projection {
        case .bounded(let work, let capacityOperations):
            guard work.prefillTokens >= 0,
                work.decodeTokens >= 0,
                work.scheduledSteps >= 0,
                work.mixedSteps >= 0,
                work.mixedSteps <= work.scheduledSteps,
                work.mixedSteps == 0
                    || (work.prefillTokens > 0 && work.decodeTokens > 0)
            else {
                return .unbounded
            }
            if let capacity {
                let pool = (backend as? PagedKVBackend)?.pool
                let physicalProjection: (([CBv2RequestID: Int]) -> Int?)?
                if let pool, pool.segmentGrant != nil {
                    physicalProjection = { [layerKinds] tokens in
                        pool.projectedPhysicalBytes(reservedTokens: tokens, layerKinds: layerKinds)
                    }
                } else { physicalProjection = nil }
                guard let admission = capacity as? AdmissionV2,
                    admission.canGuarantee(
                        projectedOperations: capacityOperations,
                        projectedPhysicalBytes: physicalProjection)
                else {
                    return .unbounded
                }
            }

            func phaseSeconds(tokens: Int, rate: Double?) -> Double? {
                guard tokens > 0 else { return 0 }
                guard let rate, rate.isFinite, rate > 0 else { return nil }
                let seconds = Double(tokens) / rate
                guard seconds.isFinite, seconds > 0 else { return nil }
                return seconds
            }

            // Serial phase envelope. For a mixed step this charges decode and
            // prefill independently instead of treating radically different
            // work as one token currency. Missing either required lower-bound
            // rate makes the posture unbounded and enforcement fails closed.
            guard let prefillSeconds = phaseSeconds(
                tokens: work.prefillTokens,
                rate: policy.conservativePrefillTokensPerSecond),
                let decodeSeconds = phaseSeconds(
                    tokens: work.decodeTokens,
                    rate: policy.conservativeDecodeTokensPerSecond)
            else {
                return .unbounded
            }
            let seconds = prefillSeconds + decodeSeconds
            // Duration.seconds(_:) traps when its scaled Int128 conversion
            // overflows. Int64.max seconds is a deliberately narrower safe
            // bound; durations beyond it cannot be useful for admission.
            guard seconds.isFinite,
                seconds >= 0,
                seconds <= Double(Int64.max)
            else {
                return .unbounded
            }
            let serviceDuration = Duration.seconds(seconds)
            guard work.scheduledTokens == 0 || serviceDuration > .zero else {
                return .unbounded
            }
            return .bounded(
                work: work,
                serviceDuration: serviceDuration)
        case .unbounded:
            return .unbounded
        }
    }

    /// Undo the limited state installed before a deadline verdict. This path
    /// runs before activation, so no lease, detokenizer, sampler state, or
    /// multimodal payload exists and the row cannot participate in `inFlight`.
    private func discardDeadlineRejectedRequest(_ id: CBv2RequestID) {
        _ = scheduler.finish(id: id, reason: .cancelled)
        capacity?.releaseAll(id: id)
        if let state = kvStates.removeValue(forKey: id) {
            backend.release(state)
        }
        releaseRecurrentState(recurrentStates.removeValue(forKey: id))
        prefixHitTokens.removeValue(forKey: id)
        prefixUsageByID.removeValue(forKey: id)
        residentPrefixCursorByID.removeValue(forKey: id)
        recurrentCheckpointGeometry.removeValue(forKey: id)
        discardHybridCheckpoints(id)
        mtp?.requestDidFinish(id)
    }

    // MARK: Prefix-cache adoption (engine thread)

    /// Adopt a prepared prefix hit into fresh KV state and fast-forward the
    /// scheduler record. Best-effort: any failure (capacity, backend) falls
    /// back to a full prefill — the request itself never fails here. Always
    /// balances the lookup pin.
    @discardableResult
    private func applyAdoption(
        _ adoption: CBv2PrefixAdoption,
        requestID: CBv2RequestID
    ) -> Bool {
        let adoptionStartedNanos = DispatchTime.now().uptimeNanoseconds
        var adoptedHybridTokens = 0
        defer {
            if let checkpoint = adoption.completeCheckpoint {
                checkpoint.close()
            } else if let pin = adoption.hybridPin {
                hybridPrefixCache?.endAdoption(pin: pin, tokensSaved: adoptedHybridTokens)
            } else {
                prefixCache?.endAdoption(
                    requestID: adoption.requestID, tokens: adoption.tokens, matched: adoption.matched,
                    cacheSalt: adoption.cacheSalt)
            }
        }
        guard let rec = scheduler.record(for: requestID), kvStates[requestID] == nil else {
            markPrefixAdoptionFailed(requestID, outcome: .adoptionFailed)
            return false
        }
        // Once per adoption (not per step): the reserve + backend adoption
        // duration, success or fallback alike. A successful adoption IS the
        // row's KV allocation (`ensureKVState` finds it), stamped from the
        // same read — and raised to the admission instant by
        // `stampAdmission` at first launch, since adoption precedes the
        // first plan and the exported contract requires
        // `admitted <= kvAllocated`.
        defer {
            let adoptionEndedNanos = DispatchTime.now().uptimeNanoseconds
            rec.timing.prefixAdoptionNanos = max(
                1, adoptionEndedNanos &- adoptionStartedNanos)
            if kvStates[requestID] != nil {
                rec.stampKVAllocated(nowNanos: adoptionEndedNanos)
            }
        }
        let transfersPagedStage = adoption.completeCheckpoint?.usesPagedBacking == true
        if !transfersPagedStage, let capacity {
            do {
                // Frozen replay physically owns full K/V through M from the
                // instant adoption publishes, even though its logical cursor
                // starts at C. Reserving only C would understate residency.
                try capacity.reserve(
                    id: requestID,
                    additionalTokens: adoption.plan.capacityReservationTokens,
                    additionalBytes: adoption.plan.initialAdditionalCapacityBytes)
            } catch {
                markPrefixAdoptionFailed(requestID, outcome: .skippedCapacity)
                return false  // capacity tight — ordinary submit cold-prefills
            }
        }
        do {
            let maxLength = rec.request.promptTokens.count + max(rec.request.maxTokens, 1)
            let state: [CBv2SequenceKV?]
            if let checkpoint = adoption.completeCheckpoint {
                state = try adoptCompleteCheckpoint(checkpoint, requestID: requestID)
            } else {
                state = try backend.makeSequenceState(
                    adopting: adoption.prefix,
                    plan: adoption.plan,
                    layerKinds: layerKinds,
                    maxLength: maxLength)
            }
            if let checkpoint = adoption.recurrentCheckpoint {
                do {
                    try adoptRecurrentCheckpoint(checkpoint, requestID: requestID)
                } catch {
                    backend.release(state)
                    throw error
                }
            }
            kvStates[requestID] = state
            rec.numComputedTokens = adoption.plan.replayStart
            rec.prefixReusePlan = adoption.plan
            prefixHitTokens[requestID] = adoption.plan.prefillTokensSaved
            prefixUsageByID[requestID]?.outcome = .hit
            prefixUsageByID[requestID]?.tier = adoption.hybridPin == nil ? .snapshot : .resident
            prefixUsageByID[requestID]?.prefillTokensSaved = adoption.plan.prefillTokensSaved
            prefixUsageByID[requestID]?.strategy = adoption.plan.strategy
            prefixUsageByID[requestID]?.replayTokens = adoption.plan.replayTokens
            if let pin = adoption.hybridPin, let checkpoint = adoption.recurrentCheckpoint {
                hybridPrefixCache?.inheritCheckpoint(pin: pin, checkpoint: checkpoint, requestID: requestID)
                adoptedHybridTokens = adoption.plan.prefillTokensSaved
            }
            return true
        } catch {
            if !transfersPagedStage {
                capacity?.unreserve(
                    id: requestID,
                    tokens: adoption.plan.capacityReservationTokens,
                    bytes: adoption.plan.initialAdditionalCapacityBytes)
            }
            let outcome: CBv2PrefixCacheOutcome
            if let kvError = error as? CBv2KVError, case .capacityExhausted = kvError {
                outcome = .skippedCapacity
            } else {
                outcome = .adoptionFailed
            }
            markPrefixAdoptionFailed(requestID, outcome: outcome)
            return false
        }
    }

    func markPrefixAdoptionFailed(
        _ requestID: CBv2RequestID, outcome: CBv2PrefixCacheOutcome
    ) {
        prefixHitTokens.removeValue(forKey: requestID)
        prefixUsageByID[requestID]?.outcome = outcome
        prefixUsageByID[requestID]?.tier = nil
        prefixUsageByID[requestID]?.prefillTokensSaved = 0
        prefixUsageByID[requestID]?.strategy = nil
        prefixUsageByID[requestID]?.replayTokens = 0
        prefixUsageByID[requestID]?.boundarySplits = 0
    }

    /// Preemption discards adopted KV and forces a full recompute. Preserve
    /// genuine miss/policy outcomes; only an actual hit loses its credit.
    private func invalidateAdoptedPrefix(_ requestID: CBv2RequestID) {
        recurrentCheckpointGeometry.removeValue(forKey: requestID)
        discardHybridCheckpoints(requestID)
        guard prefixUsageByID[requestID]?.outcome == .hit else { return }
        markPrefixAdoptionFailed(requestID, outcome: .adoptionFailed)
    }

    /// Cancellation/preemption copies share the existing bounded drain
    /// barrier with publication. A final drop is queued after that request's
    /// rolling retirements, so shutdown cannot release its slot grant early.
    func discardHybridCheckpoints(_ id: CBv2RequestID) {
        if let capture = completeCheckpointCapture {
            pendingDonationReleaseCount += 1
            let queued = capture.drop(requestID: id) { [self] in
                engineQueue.async { [self] in
                    pendingDonationReleaseCount -= 1
                    publishGauges()
                    completeDrainIfReady()
                }
            }
            if !queued { pendingDonationReleaseCount -= 1 }
        }
        guard let cache = hybridPrefixCache else { return }
        pendingDonationReleaseCount += 1
        let queued = cache.dropStaged(requestID: id) { [self] in
            engineQueue.async { [self] in
                pendingDonationReleaseCount -= 1
                publishGauges()
                completeDrainIfReady()
            }
        }
        if !queued { pendingDonationReleaseCount -= 1 }
    }

    /// Balance a lookup pin for an adoption that never reached
    /// `applyAdoption` (shutdown, queue-full rejection).
    private func releaseAbandonedAdoption(_ adoption: CBv2PrefixAdoption?) {
        guard let adoption else { return }
        if let checkpoint = adoption.completeCheckpoint {
            checkpoint.close()
            return
        }
        if let pin = adoption.hybridPin {
            hybridPrefixCache?.endAdoption(pin: pin)
            return
        }
        prefixCache?.endAdoption(
            requestID: adoption.requestID, tokens: adoption.tokens, matched: adoption.matched,
            cacheSalt: adoption.cacheSalt)
    }

    // MARK: Cancellation & backpressure (any thread)

    /// O(1): marks the request; the row is dropped at the next step boundary.
    /// A pending deadline admission is different: cancellation records an
    /// explicit state transition, but the engine queue retains stream
    /// generation ownership until it acknowledges and cleans that operation.
    func requestCancel(
        _ id: CBv2RequestID,
        streamGeneration expectedGeneration: UInt64? = nil
    ) {
        stateLock.lock()
        guard streams[id] != nil, let currentGeneration = streamGenerations[id] else {
            stateLock.unlock()
            return
        }
        guard expectedGeneration == nil || expectedGeneration == currentGeneration else {
            stateLock.unlock()
            return
        }
        pendingCancels[id] = currentGeneration
        if let operation = deadlineAdmissionOperations[id],
            operation.generation == currentGeneration
        {
            switch operation.phase {
            case .queued, .executing:
                operation.phase = .cancellationRequested
            case .cancellationRequested, .failedExternally, .completed:
                break
            }
        }
        stateLock.unlock()
    }

    func setPaused(_ id: CBv2RequestID, _ paused: Bool) {
        engineQueue.async { [self] in
            let now = config.clock.now()
            if paused {
                scheduler.pause(id)
                scheduler.record(for: id)?.recordPaused(now: now)
                // Arm the backpressure lease and suspend the progress lease:
                // a request blocked on a slow consumer is not an engine stall.
                if var lease = leasesByID[id] {
                    lease.markPaused(now: now)
                    leasesByID[id] = lease
                }
            } else {
                scheduler.resume(id)
                scheduler.record(for: id)?.recordResumed(now: now)
                // Clear the backpressure lease and grant a fresh progress
                // window — the paused interval was not the engine's fault.
                if var lease = leasesByID[id] {
                    lease.markResumed(now: now)
                    leasesByID[id] = lease
                }
            }
        }
    }

    // MARK: The step loop

    private func engineStep() {
        guard running else { return }
        if let suspendedAt = suspendStepExecutionAtCountForTesting,
            stepCount >= suspendedAt
        {
            scheduleIdleRecheck()
            return
        }
        let stepStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        markStepStarted()
        // Instruments-only step interval: both ends gated on `isEnabled`, so
        // an unattached process pays one static bool read per step.
        let signposter = Self.signposter
        let signpostState = signposter.isEnabled ? signposter.beginInterval("step") : nil
        defer {
            if let signpostState { signposter.endInterval("step", signpostState) }
            markStepEnded()
            if CBv2StepProfiler.enabled {
                CBv2StepProfiler.record(
                    "v2.step.wall", seconds: CFAbsoluteTimeGetCurrent() - stepStart)
            }
        }

        // One monotonic clock read per step, reused for lease expiry, admission
        // stamping, and the finalize-time progress refresh — so lease gaps are
        // exactly one step apart (never widened by extra clock reads).
        let stepNow = config.clock.now()
        boundaryClockNanos = 0  // see `boundaryFinishNanos`
        processCancellations(now: stepNow)
        processLeaseExpiry(now: stepNow)

        // Chained decode fast path: build step N+1 on step N's lazy tokens.
        // MTP guards: an MTP round step is never a chain base (its finalize
        // confirms 1+k samples per row — the chained machinery assumes
        // exactly one), and the chain must BREAK whenever the MTP driver
        // wants the next step (seed or round) for any candidate row —
        // chained launches bypass hidden capture, so seeding could
        // otherwise never start.
        if let previous = inFlight,
            // A private historical read never becomes a serving write-fence
            // dependency. Finish its exact step before a successor can write.
            previous.permitsChainedSuccessor,
            previous.sampledTokens != nil,
            let ids = scheduler.chainCandidateIDs(),
            ids == previous.sampledRows,
            ids.allSatisfy({ kvStates[$0] != nil }),
            // Constraint state advances from the host-confirmed token. Do
            // not launch N+1 from N's lazy token before that transition.
            ids.allSatisfy({ scheduler.record(for: $0)?.request.tokenConstraint == nil }),
            !mtpWantsStep(ids: ids),
            capacity?.hasHeadroom(additionalTokens: ids.reduce(0) { total, id in
                guard let rec = scheduler.record(for: id) else { return total + 1 }
                return total + rec.capacityTokensForChunk(start: rec.numComputedTokens, count: 1)
            }) ?? true
        {
            beginMTPPlan()
            let plan = scheduler.plan()
            if isPureDecodePlan(plan, matching: ids) {
                if CBv2StepProfiler.enabled {
                    CBv2StepProfiler.record(
                        "v2.boundary", seconds: CFAbsoluteTimeGetCurrent() - stepStart)
                }
                let measurement = mtpMeasurement(for: plan)
                let boundary = (backend as? PagedKVBackend).map { CBv2PagedWriteBoundary(pool: $0.pool) }
                let next: CBv2InFlightStep
                do {
                    next = try launchChainedDecode(plan, feeding: previous.sampledTokens!)
                } catch {
                    handlePagedWriteFailure(error, plan: plan, boundary: boundary, now: stepNow)
                    publishGauges()
                    scheduleNextStep()
                    return
                }
                attachMTPMeasurement(measurement, to: next, chained: true)
                if var previousMeasurement = previous.mtpMeasurement {
                    // The previous step's finalize-to-launch interval now
                    // includes construction of this successor. It is no
                    // longer an isolated depth-zero cost sample either.
                    previousMeasurement.chained = true
                    previous.mtpMeasurement = previousMeasurement
                }
                inFlight = next
                chainedStepCount += 1
                stepCount += 1
                finalize(previous, now: stepNow)
                publishGauges()
                scheduleNextStep()
                return
            }
            // Defensive: chainCandidateIDs and plan() disagree (only
            // possible when the capacity oracle's `hasHeadroom` was
            // optimistic and `reserve` preempted mid-plan). Roll the
            // optimistic advance back and fall through to the general path.
            // Preemptions are NOT rolled back (the scheduler already
            // requeued the victims), so their KV must be released here —
            // fenced behind the still-in-flight step that references it.
            // The victims are NOT added to `discard`: their in-flight
            // sample must be recorded at finalization (preemption keeps
            // generated tokens, and an unconfirmed `pendingSamples` would
            // block their re-admission forever).
            scheduler.rollback(plan)
            // Lease rewind is DEFERRED past the finalize below: the victims'
            // in-flight sample is confirmed there (refreshProgressLeases would
            // see generated > watermark and flip the lease back to .decode,
            // undoing an earlier markPreempted — PR#82 review). The general
            // path already has the correct order because it finalizes before
            // planning; this defers the rollback path to match.
            leasePreemptionsPendingFinalize.append(contentsOf: plan.preemptions)
            for id in plan.preemptions {
                preemptionCount += 1
                // A preempted request recomputes from scratch — any adopted
                // prefix credit no longer describes work that was skipped.
                invalidateAdoptedPrefix(id)
                mtp?.invalidateCarry(id)
                guard let state = kvStates.removeValue(forKey: id) else { continue }
                if previous.participants.contains(id) {
                    previous.deferredReleases.append(
                        (
                            id: id, state: state, rollbackOne: false, donation: nil,
                            recurrent: recurrentStates.removeValue(forKey: id),
                            hybridPublication: nil,
                            ownership: nil, terminalDelivery: nil
                        ))
                } else {
                    backend.release(state)
                    releaseRecurrentState(recurrentStates.removeValue(forKey: id))
                }
            }
        }

        // Chain broken (or nothing chained): finalize before re-planning so
        // the plan sees confirmed tokens and post-stop membership.
        if let previous = inFlight {
            inFlight = nil
            finalize(previous, now: stepNow)
        }
        // Apply the rollback path's deferred lease rewinds now that the
        // victims' in-flight samples are confirmed: markPreempted must be the
        // LAST lease transition of the preemption instant so the fresh
        // prefill window and zeroed watermark survive into the re-prefill.
        if !leasePreemptionsPendingFinalize.isEmpty {
            for id in leasePreemptionsPendingFinalize {
                resetResidentPrefixPublication(id)
                if var lease = leasesByID[id] {
                    lease.markPreempted(now: stepNow)
                    leasesByID[id] = lease
                }
            }
            leasePreemptionsPendingFinalize.removeAll(keepingCapacity: true)
        }

        guard scheduler.hasWork else {
            // Idle: no future step will rebind the eager caches, so drop
            // their row bindings now — otherwise the last batch's (retired)
            // rows stay strongly retained by the provider until some future
            // rebind, pinning dead KV on an idle engine (PR#62 review).
            // No-op after the first call while idle.
            (cacheProvider as? CBv2CompositionInvalidating)?.releaseBoundRows()
            publishGauges()
            if draining {
                completeDrainIfReady()
                if !running { return }
            }
            scheduleIdleRecheck()
            return
        }

        beginMTPPlan()
        // Snapshot the waiting set BEFORE plan() so re-admissions (rows this
        // plan moves waiting→running) are distinguishable from continuing
        // running rows in markAdmitted below.
        let waitingBeforePlan = Set(scheduler.waiting.map(\.id))
        let plan = scheduler.plan()
        handlePreemptions(plan.preemptions, now: stepNow)
        guard !plan.assignments.isEmpty else {
            publishGauges()
            scheduleIdleRecheck()
            return
        }
        // First engine work for any newly-admitted row ends its admission lease
        // permanently and arms the prefill progress lease; a RE-admitted row
        // gets a fresh progress window (its demotion-time window may have
        // expired during the stall-exempt queue wait). Idempotent for
        // continuing rows (a preempted requeue never re-arms admission).
        markAdmitted(plan, now: stepNow, readmittedFrom: waitingBeforePlan)
        // MTP round steps (1+k verify assignments and/or seed decodes)
        // branch BEFORE executeMixed — its rec.tokens slicing and samples
        // predicate are structurally wrong for speculative tokens.
        let measurement = mtpMeasurement(for: plan)
        let boundary = (backend as? PagedKVBackend).map { CBv2PagedWriteBoundary(pool: $0.pool) }
        do {
            inFlight = try (mtpRoundNeeded(plan) ? executeMTPRound(plan) : executeMixed(plan))
        } catch {
            handlePagedWriteFailure(error, plan: plan, boundary: boundary, now: stepNow)
        }
        attachMTPMeasurement(measurement, to: inFlight, chained: false)
        stepCount += 1
        publishGauges()
        scheduleNextStep()
    }

    private func scheduleNextStep() {
        engineQueue.async { [weak self] in self?.engineStep() }
    }

    private func scheduleIdleRecheck() {
        engineQueue.asyncAfter(deadline: .now() + config.idleRecheckInterval) { [weak self] in
            self?.engineStep()
        }
    }

    /// Model protocols remain nonthrowing. A paged cache records a typed fault
    /// and returns shape-preserving placeholders so later model layers unwind
    /// normally; this boundary refuses those graphs before sampling/evaluation.
    func checkedModelForward<Result>(phase: CBv2ForwardPhase = .decode, _ body: () -> Result) throws -> Result {
        let validation = (backend as? PagedKVBackend)?.pool.writeValidation
        try validation?.check()
        let value = CBv2ForwardShapeObservation.dispatch(step: buildingForwardShapes, phase: phase, body)
        try validation?.check()
        return value
    }

    /// Rare runtime-contract failure. Model/recurrent/MTP graph construction may
    /// already have advanced different layers by different amounts, so retire
    /// the entire planned cohort rather than pretending cursor rewind repairs it.
    private func handlePagedWriteFailure(
        _ error: Error, plan: CBv2StepPlan, boundary: CBv2PagedWriteBoundary?,
        now: ContinuousClock.Instant
    ) {
        // MTP can submit assistant work before target verification, and chained
        // decode has a prior valid step in flight. Synchronize submitted work
        // only; never evaluate the failed forward's placeholders/write fences.
        Stream.gpu.synchronize()
        Stream.cpu.synchronize()
        boundary?.discardFailedGraphAfterSynchronization()
        attentionMetadata?.discardPendingForward()
        attentionPacket?.discardPendingForward()
        (cacheProvider as? CBv2CompositionInvalidating)?.releaseBoundRows()
        eagerCompositionStale = true
        scheduler.rollback(plan)
        if let previous = inFlight {
            previous.discard.formUnion(plan.assignments.map(\.id))
            inFlight = nil
            finalize(previous, now: now)
            // The chained caller also retains this step. Its transactions
            // retain their input arrays even after rollback; drain those
            // aliases before releasing request state and refunding capacity.
            previous.recurrentEvaluations.removeAll()
        }
        for assignment in plan.assignments {
            // Earlier successful forwards/serial target columns in this failed
            // step may have queued recurrent generations. Their GPU work is
            // now fenced and all model-call locals have unwound.
            mtp?.requestDidFinish(assignment.id)
            if let recurrent = recurrentStates.removeValue(forKey: assignment.id) {
                recurrent.discardPendingAfterSynchronization()
                releaseRecurrentState(recurrent)
            }
            if let state = kvStates.removeValue(forKey: assignment.id) {
                backend.release(state)
            }
            // finishRequest now sees no native payload; its capacity refund
            // follows actual target/recurrent/assistant alias retirement.
            finishRequest(assignment.id, reason: .error(String(describing: error)), now: now)
        }
        (backend as? PagedKVBackend)?.pool.writeValidation.clearAfterRetirement()
    }

    // MARK: Step execution

    private func isPureDecodePlan(_ plan: CBv2StepPlan, matching ids: [CBv2RequestID]) -> Bool {
        guard plan.preemptions.isEmpty, plan.assignments.count == ids.count else { return false }
        for (i, assignment) in plan.assignments.enumerated() {
            guard assignment.numTokens == 1, assignment.id == ids[i] else { return false }
        }
        return true
    }

    /// Eager layer caches, with the provider's composition fingerprint
    /// force-invalidated when an MTP round advanced rows behind its back.
    func eagerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        if eagerCompositionStale {
            (cacheProvider as? CBv2CompositionInvalidating)?.invalidateBoundComposition()
            eagerCompositionStale = false
        }
        let caches = cacheProvider.layerCaches(rowStates: rowStates)
        return caches
    }

    /// Inner-state arrays (lazily-mutated KV buffers + the on-device
    /// `positionOffsets` chain) of the eager layer caches, collected so the
    /// step's `asyncEval` collapses those lazy chains every step instead of
    /// letting `updateAndAttend`'s `+ L` offset advance accumulate O(steps)
    /// of unevaluated graph — the DAR-325 bug class (legacy `BatchKVCache`
    /// had exactly this). Empty for caches that vend no inner state (mocks).
    func eagerCacheInnerState(_ caches: [CBv2AttendingLayerCache]) -> [MLXArray] {
        caches.flatMap { ($0 as? KVCache)?.innerState() ?? [] }
    }

    /// Build one target forward with explicit row-aligned recurrent
    /// transactions when the model requires them. The returned arrays are
    /// appended to the step's eval set so conv/SSM state is materialized at
    /// the same boundary as KV and logits.
    /// `DARKBLOOM_CBV2_PREFILL_NARROWING=0` restores the engine-sliced full
    /// vocabulary projection (byte-old prompt behavior) as the A/B control
    /// and incident escape; any other value (or absence) keeps narrowing on.
    static let prefillNarrowingEnabled: Bool =
        ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_PREFILL_NARROWING"] != "0"

    /// `requirement` non-nil marks a PREFILL call: the returned `logits` are
    /// then already narrowed to the requirement (`[B,1]` handle or
    /// `[B, vocab]` frontier row) — by the model itself when it conforms to
    /// `CBv2RecurrentPrefillSteppableModel` (skipping the unused vocabulary
    /// projection), else by the engine's own slice. Decode and MTP callers
    /// pass nil and keep the full-logits contract.
    func targetForward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache], ids: [CBv2RequestID],
        positionIds: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        requirement: CBv2PrefillRequirement? = nil,
        phase: CBv2ForwardPhase? = nil
    ) throws -> (logits: MLXArray, recurrent: [CBv2RequestID: CBv2RecurrentStateEvaluation],
        innerState: [MLXArray])
    {
        let forwardPhase = phase ?? (requirement == nil ? .decode : .prefill)
        guard let recurrentModel = model as? any CBv2RecurrentSteppableModel,
            recurrentModel.recurrentStateSpec != nil
        else {
            if let inputEmbeddings {
                if positionIds != nil {
                    guard
                        let positioned = model as? any CBv2PositionedEmbeddingSteppableModel
                    else {
                        preconditionFailure(
                            "CBv2 positioned embedding forward reached an unsupported model")
                    }
                    let logits = try checkedModelForward(phase: forwardPhase) { positioned.forward(
                        tokens: tokens, inputEmbeddings: inputEmbeddings,
                        caches: caches, positionIds: positionIds) }
                    return (
                        requirement.map { narrowPrefillOutput(logits, requirement: $0) }
                            ?? logits,
                        [:], [])
                }
                guard let multimodal = model as? any CBv2MultimodalSteppableModel else {
                    preconditionFailure("CBv2 embedding forward reached an unsupported model")
                }
                let logits = try checkedModelForward(phase: forwardPhase) { multimodal.forward(
                    tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches) }
                return (
                    requirement.map { narrowPrefillOutput(logits, requirement: $0) } ?? logits,
                    [:], [])
            }
            if positionIds != nil {
                guard let positioned = model as? CBv2PositionedSteppableModel else {
                    preconditionFailure(
                        "CBv2 positioned attention forward reached an unsupported model")
                }
                let logits = try checkedModelForward(phase: forwardPhase) { positioned.forward(
                    tokens: tokens, caches: caches, positionIds: positionIds) }
                return (
                    requirement.map { narrowPrefillOutput(logits, requirement: $0) }
                        ?? logits,
                    [:], [])
            }
            let logits = try checkedModelForward(phase: forwardPhase) { model.forward(tokens: tokens, caches: caches) }
            return (
                requirement.map { narrowPrefillOutput(logits, requirement: $0) } ?? logits,
                [:], [])
        }
        let evaluations = ids.map { id -> CBv2RecurrentStateEvaluation in
            guard let state = recurrentStates[id] else {
                preconditionFailure("CBv2 recurrent state missing for \(id)")
            }
            do { return try state.bind() } catch {
                preconditionFailure("CBv2 recurrent bind failed for \(id): \(error)")
            }
        }
        let forward = try recurrentTargetForward(
            tokens: tokens, caches: caches, recurrentModel: recurrentModel,
            evaluations: evaluations, positionIds: positionIds,
            inputEmbeddings: inputEmbeddings, requirement: requirement, phase: forwardPhase)
        return (forward.logits, Dictionary(uniqueKeysWithValues: zip(ids, evaluations)), forward.innerState)
    }

    private func recurrentTargetForward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentModel: any CBv2RecurrentSteppableModel,
        evaluations: [CBv2RecurrentStateEvaluation], positionIds: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil, requirement: CBv2PrefillRequirement? = nil,
        phase: CBv2ForwardPhase? = nil
    ) throws -> (logits: MLXArray, innerState: [MLXArray]) {
        let forwardPhase = phase ?? (requirement == nil ? .decode : .prefill)
        let logits: MLXArray
        if let requirement, Self.prefillNarrowingEnabled,
            let prefillable = model as? any CBv2RecurrentPrefillSteppableModel
        {
            // Prefill seam: state already bound above, staging identical to
            // the full forward; the model returns only what the requirement
            // needs (no vocabulary projection on discarded rows).
            logits = try checkedModelForward(phase: forwardPhase) { prefillable.recurrentPrefill(
                tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches,
                recurrentState: evaluations, positionIds: positionIds,
                requirement: requirement) }
        } else if let inputEmbeddings {
            guard let positioned = model as? any CBv2PositionedMultimodalSteppableModel else {
                preconditionFailure(
                    "CBv2 recurrent embedding forward reached an unsupported model")
            }
            let full = try checkedModelForward(phase: forwardPhase) { positioned.forward(
                tokens: tokens,
                inputEmbeddings: inputEmbeddings,
                caches: caches,
                recurrentState: evaluations,
                positionIds: positionIds) }
            logits = requirement.map { narrowPrefillOutput(full, requirement: $0) } ?? full
        } else if positionIds != nil {
            guard let positioned = model as? any CBv2PositionedRecurrentSteppableModel else {
                preconditionFailure("CBv2 positioned forward reached an unsupported model")
            }
            let full = try checkedModelForward(phase: forwardPhase) { positioned.forward(
                tokens: tokens, caches: caches,
                recurrentState: evaluations, positionIds: positionIds) }
            logits = requirement.map { narrowPrefillOutput(full, requirement: $0) } ?? full
        } else {
            let full = try checkedModelForward(phase: forwardPhase) { recurrentModel.forward(
                tokens: tokens, caches: caches, recurrentState: evaluations) }
            logits = requirement.map { narrowPrefillOutput(full, requirement: $0) } ?? full
        }
        var arrays: [MLXArray] = []
        for evaluation in evaluations {
            do { arrays.append(contentsOf: try evaluation.evaluate()) } catch {
                preconditionFailure("CBv2 recurrent evaluation failed: \(error)")
            }
        }
        return (logits, arrays)
    }

    /// Last-position logits [B, vocab] for a rectangular [B, 1] decode
    /// batch. The second tuple element is the eager caches' inner state
    /// (offset chain + KV buffers) that must ride the step's `asyncEval`
    /// (DAR-325).
    private func decodeLogits(
        rowStates: [[CBv2SequenceKV?]], tokens: MLXArray, ids: [CBv2RequestID],
        chained: Bool = false, phase: CBv2ForwardPhase = .decode
    ) throws -> (logits: MLXArray, cacheInnerState: [MLXArray],
        recurrent: [CBv2RequestID: CBv2RecurrentStateEvaluation])
    {
        let caches = eagerCaches(rowStates: rowStates)
        let metadata = bindAttentionMetadata(caches: caches, ids: ids, chained: chained)
        let packet = bindAttentionPacket(caches: caches, ids: ids, chained: chained)
        var forwardSucceeded = false
        defer {
            finishAttentionMetadata(metadata, caches: caches, succeeded: forwardSucceeded)
            finishAttentionPacket(packet, caches: caches, succeeded: forwardSucceeded)
        }
        let positionIds = CBv2PositionState.decodePositionIds(
            states: ids.map { scheduler.record(for: $0)?.request.positionState },
            cacheOffsets: rowStates.map(Self.positionOffset))
        let forward = try targetForward(
            tokens: tokens, caches: caches, ids: ids, positionIds: positionIds, phase: phase)
        forwardSucceeded = true
        return (
            forward.logits[0..., -1, 0...],
            eagerCacheInnerState(caches) + forward.innerState,
            forward.recurrent)
    }

    /// Prompt-only output seam (see PrefillOutputV2.swift). Capable models
    /// skip the vocabulary projection for discarded prompt positions;
    /// everything else keeps the established full-logits forward and is
    /// sliced HERE, so the optimization is strictly opt-in and byte-neutral
    /// for non-conforming models.
    ///
    /// Returns `[B, 1]` for `.evaluationOnly` (a cheap handle that still
    /// forces the whole chunk's graph, including its KV writes) and
    /// `[B, vocab]` for `.lastPositionLogits`.
    func prefillOutput(
        tokens: MLXArray,
        inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) throws -> MLXArray {
        if let prefillModel = model as? CBv2PrefillSteppableModel {
            return try checkedModelForward(phase: .prefill) { prefillModel.prefill(
                tokens: tokens,
                inputEmbeddings: inputEmbeddings,
                caches: caches,
                requirement: requirement) }
        }

        let logits: MLXArray
        if let inputEmbeddings {
            guard let multimodalModel = model as? CBv2MultimodalSteppableModel else {
                preconditionFailure(
                    "CBv2 embedding prefill reached a model without embedding-forward support")
            }
            logits = try checkedModelForward(phase: .prefill) { multimodalModel.forward(
                tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches) }
        } else {
            logits = try checkedModelForward(phase: .prefill) { model.forward(tokens: tokens, caches: caches) }
        }
        switch requirement {
        case .evaluationOnly:
            return logits[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            return logits[0..., -1, 0...]
        }
    }

    // MARK: - Teacher-forced top-1 scoring (backend parity measurement)

    /// Drive `continuation` through the engine on a private KV row and
    /// return the ARGMAX at each continuation position.
    ///
    /// Free-running comparison between two engine arms is only valid up to
    /// the FIRST disagreement: past it the arms carry different contexts and
    /// every later position compares two unrelated conversations, so a
    /// harness can report a first-flip index but never an agreement RATE.
    /// Teacher forcing removes that: position `i` is always scored against
    /// `promptTokens + continuation[0..<i]`, identically in both arms,
    /// whatever either arm would have preferred. Agreement becomes a rate.
    ///
    /// The scoring row travels the ENGINE's own path, not a shortcut: the
    /// prompt goes through `prefillOutput` in the same chunks
    /// `SchedulerV2.plan()` would cut (`min(prefillChunkSize,
    /// maxBatchedTokensPerStep)`), and each forced token enters as its own
    /// `[1, 1]` forward through `eagerCaches` — byte-for-byte the decode
    /// seam. Scoring positions outside the engine would measure the MODEL,
    /// and the model is the one thing the two backends share.
    ///
    /// Deterministic by construction: `argMax` over the last-position
    /// logits. No sampler, no temperature, no top-k, no RNG.
    ///
    /// Result length == `continuation.count`. `result[0]` is the argmax
    /// after the prompt alone; `result[i]` the argmax after
    /// `continuation[i - 1]` is forced in. Forcing the model's own greedy
    /// continuation therefore returns that continuation unchanged.
    ///
    /// Runs on the engine queue, but NOT as a step: it never enters the
    /// scheduler, the chain, or the step watchdog. It refuses while OTHER
    /// requests are live — not for safety (the row is bound alone, so batch
    /// composition cannot reach its arithmetic) but for comparability: a
    /// contended pool hands the row different pages, and page order is
    /// precisely the drift a parity harness is trying to measure. A
    /// trailing in-flight step from a request that already finished is not
    /// contention and does not block scoring; its rows are re-derived and
    /// dropped around this call.
    func teacherForcedTop1(promptTokens: [Int], continuation: [Int]) throws -> [Int] {
        guard !promptTokens.isEmpty, !continuation.isEmpty else {
            throw CBv2TeacherForcingError.nothingToScore(
                promptTokens: promptTokens.count, continuation: continuation.count)
        }
        return try engineQueue.sync {
            try scoreTeacherForced(promptTokens: promptTokens, continuation: continuation)
        }
    }

    func teacherForcedScores(_ request: CBv2TeacherForcedScoreRequest) throws
        -> CBv2TeacherForcedScoreSnapshot
    {
        try engineQueue.sync {
            let collector = CBv2TeacherForcedScoreCollector(request)
            _ = try scoreTeacherForced(promptTokens: request.promptTokens,
                continuation: request.continuation, diagnostic: collector)
            guard let snapshot = collector.snapshot else {
                throw CBv2TeacherForcedScoreError.incompleteObservation
            }
            return snapshot
        }
    }

    /// Engine-queue body of `teacherForcedTop1`.
    private func scoreTeacherForced(
        promptTokens: [Int], continuation: [Int],
        diagnostic: CBv2TeacherForcedScoreCollector? = nil
    ) throws -> [Int] {
        guard running, !draining else { throw CBv2TeacherForcingError.engineNotRunning }
        guard !scheduler.hasWork else {
            throw CBv2TeacherForcingError.engineBusy(
                scheduledRequests: scheduler.running.count + scheduler.waiting.count)
        }

        // Same allocation the loop makes for a real request of this shape.
        let previousTargetBytes = backend.bytesReserved
        var state = try backend.makeSequenceState(
            layerKinds: layerKinds,
            promptLength: promptTokens.count,
            maxLength: promptTokens.count + continuation.count)
        let writeBoundary = (backend as? PagedKVBackend).map { CBv2PagedWriteBoundary(pool: $0.pool) }
        var recurrent: CBv2RecurrentRequestState?
        var recurrentEvaluation: CBv2RecurrentStateEvaluation?
        var recurrentReservation: CBv2CheckpointReservation?
        defer {
            if let recurrent {
                Stream.gpu.synchronize()
                Stream.cpu.synchronize()
                recurrentEvaluation = nil
                recurrent.discardPendingAfterSynchronization()
                releaseRecurrentState(recurrent)
            }
            if let pool = (backend as? PagedKVBackend)?.pool, pool.writeValidation.isFaulted {
                Stream.gpu.synchronize()
                Stream.cpu.synchronize()
                writeBoundary?.discardFailedGraphAfterSynchronization()
            }
            // Drop the caches' strong binding to this retired row BEFORE the
            // backend recycles its pages (PR#62), and force the next real
            // step to rebind its own rows.
            (cacheProvider as? CBv2CompositionInvalidating)?.releaseBoundRows()
            backend.release(state)
            state.removeAll()
            (backend as? PagedKVBackend)?.pool.writeValidation.clearAfterRetirement()
            recurrentReservation?.release()
        }
        if (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec != nil {
            guard let admission = capacity as? AdmissionV2 else {
                throw CBv2KVError.backendIneligible(
                    reason: "recurrent teacher scoring requires peak admission accounting")
            }
            // makeRecurrentRequestState is a one-generation allocation check;
            // ordinary admission owns the larger committed/pending peak. This
            // private call needs that same obligation without a scheduler ID.
            recurrentReservation = try admission.reserveUnscheduledRequest(
                maximumTokens: promptTokens.count + continuation.count,
                minimumTargetBytes: max(0, backend.bytesReserved - previousTargetBytes))
        }
        recurrent = try makeRecurrentRequestState()

        func forward(
            tokens: MLXArray, caches: [CBv2AttendingLayerCache],
            requirement: CBv2PrefillRequirement?
        ) throws -> (logits: MLXArray, innerState: [MLXArray]) {
            guard let recurrent else {
                if let requirement {
                    return (try prefillOutput(tokens: tokens, inputEmbeddings: nil,
                        caches: caches, requirement: requirement), [])
                }
                return (try checkedModelForward { model.forward(tokens: tokens, caches: caches) }, [])
            }
            guard let recurrentModel = model as? any CBv2RecurrentSteppableModel else {
                preconditionFailure("teacher recurrent state requires recurrent model")
            }
            recurrentEvaluation = try recurrent.bind()
            return try recurrentTargetForward(
                tokens: tokens, caches: caches, recurrentModel: recurrentModel,
                evaluations: [recurrentEvaluation!], requirement: requirement)
        }

        func finishForward(_ arrays: [MLXArray]) throws {
            if let evaluation = recurrentEvaluation {
                eval(arrays)
                StreamOrDevice.default.stream.synchronize()
                try evaluation.commit()
                recurrentEvaluation = nil
            } else {
                asyncEval(arrays)
            }
        }

        // One lazy [1] argmax per continuation position; the forwards never
        // read them back (the next input is the FORCED token, already on the
        // host), so the whole run needs exactly one readback at the end.
        var top1: [MLXArray] = []
        top1.reserveCapacity(continuation.count)

        // Prompt: chunked prefill, cut exactly as the scheduler would for a
        // solo row (`min(remaining, prefillChunkSize, budget)`).
        let chunkSize = max(
            1, min(scheduler.config.prefillChunkSize, scheduler.config.maxBatchedTokensPerStep))
        var index = 0
        while index < promptTokens.count {
            let count = min(chunkSize, promptTokens.count - index)
            let isFinalChunk = index + count == promptTokens.count
            let inputs = MLXArray(promptTokens[index ..< index + count].map(Int32.init))
                .reshaped([1, count])
            let caches = eagerCaches(rowStates: [state])
            let forwarded = try forward(
                tokens: inputs, caches: caches,
                requirement: isFinalChunk ? .lastPositionLogits : .evaluationOnly)
            let output = forwarded.logits
            teacherForcedPrefillChunks += 1
            var toEval = eagerCacheInnerState(caches) + forwarded.innerState
            if isFinalChunk {
                let argmax = argMax(output, axis: -1)  // [1]
                top1.append(argmax)
                toEval.append(argmax)
                if let diagnostic {
                    do { toEval += try diagnostic.capture(logits: output, index: 0) }
                    catch {
                        eval(toEval)
                        StreamOrDevice.default.stream.synchronize()
                        throw error
                    }
                }
            } else {
                // `.evaluationOnly` hands back a [1, 1] handle that still
                // forces the chunk's whole graph, KV writes included.
                toEval.append(output)
            }
            try finishForward(toEval)
            index += count
        }

        // Continuation: one [1, 1] decode forward per FORCED token. The last
        // continuation token is never fed — its logits would score a
        // position past the end of the range being measured.
        for forced in continuation.dropLast() {
            let inputs = MLXArray([Int32(forced)]).reshaped([1, 1])
            let caches = eagerCaches(rowStates: [state])
            let forwarded = try forward(tokens: inputs, caches: caches, requirement: nil)
            let logits = forwarded.logits[0..., -1, 0...]
            teacherForcedDecodeForwards += 1
            let argmax = argMax(logits, axis: -1)  // [1]
            top1.append(argmax)
            var toEval = eagerCacheInnerState(caches) + forwarded.innerState
            toEval.append(argmax)
            if let diagnostic {
                do { toEval += try diagnostic.capture(logits: logits, index: top1.count - 1) }
                catch {
                    eval(toEval)
                    StreamOrDevice.default.stream.synchronize()
                    throw error
                }
            }
            try finishForward(toEval)
        }

        let scored = top1.count == 1 ? top1[0] : concatenated(top1, axis: 0)
        let result = scored.asArray(Int32.self).map(Int.init)
        try diagnostic?.finish(top1: result)
        return result
    }

    /// Pure-decode step fed by the previous step's still-lazy tokens.
    private func launchChainedDecode(
        _ plan: CBv2StepPlan, feeding lazyTokens: MLXArray
    ) throws -> CBv2InFlightStep {
        let shapes = beginForwardShapeStep()
        defer { endForwardShapeStep(shapes) }
        let wallStartedNanos = DispatchTime.now().uptimeNanoseconds
        let buildStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let ids = plan.assignments.map(\.id)
        let rowStates = ids.map { kvStates[$0]! }  // presence pre-checked
        var params: [CBv2SamplingParams] = []
        params.reserveCapacity(ids.count)
        for id in ids { params.append(scheduler.record(for: id)!.request.sampling) }

        let inputs = lazyTokens.reshaped([ids.count, 1])
        let diagnosticOffsets = logitDiagnostic == nil ? nil : rowStates.map(Self.positionOffset)
        let forwardStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let (last, cacheInnerState, recurrent) = try decodeLogits(
            rowStates: rowStates, tokens: inputs, ids: ids, chained: true)  // [B, vocab]
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.forward.build", seconds: CFAbsoluteTimeGetCurrent() - forwardStart)
        }
        // `pendingSampledTokens` = the fed lazy tokens: each row has exactly
        // one launched-but-unconfirmed sample here (the chain invariant).
        let samplerStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let sampled = sampler.sample(
            logits: last, params: params, requestIDs: ids, stepIndex: stepCount,
            pendingSampledTokens: lazyTokens,
            rowContext: { [scheduler] in
                ids.map { Self.samplerRow(scheduler.record(for: $0)!) }
            })
        let stepLogprobs = sampler.takeStepLogprobs()
        var logitDiagnostics: [CBv2LogitDiagnosticPacket] = []
        if let diagnosticOffsets {
            for (index, id) in ids.enumerated() {
                let rec = scheduler.record(for: id)!
                // This forward consumes one not-yet-confirmed sample. Its
                // logits select the position AFTER that pending token.
                if let packet = makeLogitDiagnostic(
                    logits: last[index], requestID: id,
                    outputIndex: rec.generatedTokenCount + 1, phase: "chained_decode",
                    batchIndex: index, batchSize: ids.count, seedToken: nil,
                    cacheOffset: diagnosticOffsets[index])
                {
                    logitDiagnostics.append(packet)
                }
            }
        }
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.sampler.build", seconds: CFAbsoluteTimeGetCurrent() - samplerStart)
        }
        scheduler.markPendingSamples(ids: ids)
        let rawAttention = attentionPacket?.takePendingForward()
        var toEval = [sampled]
        if let rawAttention { toEval.append(contentsOf: rawAttention.evaluationTargets) }
        for packet in logitDiagnostics { toEval.append(contentsOf: packet.evaluationTargets) }
        if let stepLogprobs { toEval.append(contentsOf: stepLogprobs.evalTargets) }
        // Collapse the eager caches' lazy offset/KV chains this step (DAR-325).
        if !cacheInnerState.isEmpty {
            toEval.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }
        let evalStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        asyncEval(toEval)
        if CBv2StepProfiler.enabled {
            let now = CFAbsoluteTimeGetCurrent()
            CBv2StepProfiler.record("v2.asyncEval.submit", seconds: now - evalStart)
            CBv2StepProfiler.record("v2.launch.total", seconds: now - buildStart)
        }
        let step = CBv2InFlightStep(
            assignments: plan.assignments,
            participants: Set(ids), sampledRows: ids, sampledTokens: sampled, evalTargets: [],
            computedRanges: Dictionary(
                uniqueKeysWithValues: plan.assignments.map { assignment in
                    let end = scheduler.record(for: assignment.id)!.numComputedTokens
                    return (assignment.id, (end - assignment.numTokens) ..< end)
                }),
            wallStartedNanos: wallStartedNanos)
        if let stepLogprobs { step.logprobSegments = [stepLogprobs] }
        step.logitDiagnostics = logitDiagnostics
        step.attentionMetadata = attentionMetadata?.takePendingForward()
        step.attentionPacket = rawAttention
        step.recurrentEvaluations = recurrent
        step.chained = true
        step.forwardShapes = shapes
        shapes?.attach()
        return step
    }

    /// General step: decode batch [B, 1] + per-request [1, chunk] prefills,
    /// interleaved per the plan, ONE `asyncEval` for the whole step.
    /// NEVER extended for MTP speculation — 1+k assignments take
    /// `executeMTPRound` (this function's `rec.tokens` slicing and
    /// `samples` predicate are structurally wrong for speculative tokens).
    func executeMixed(_ plan: CBv2StepPlan) throws -> CBv2InFlightStep? {
        let shapes = beginForwardShapeStep()
        defer { endForwardShapeStep(shapes) }
        let wallStartedNanos = DispatchTime.now().uptimeNanoseconds
        launchClockNanos = wallStartedNanos
        defer { launchClockNanos = 0 }
        struct RowWork {
            let rec: CBv2ScheduledRequest
            let start: Int
            let count: Int
            let samples: Bool
            let isDecode: Bool
        }

        var work: [RowWork] = []
        work.reserveCapacity(plan.assignments.count)
        for (id, n) in plan.assignments {
            guard let rec = scheduler.record(for: id) else { continue }
            rec.stampAdmission(launchNanos: wallStartedNanos)
            guard ensureKVState(rec) != nil else { continue }  // error-finished
            let start = rec.numComputedTokens - n  // pre-optimistic-advance position
            // The step samples iff it computes through the last known token.
            // (`pendingSamples == 0` here: finalize always precedes
            // executeMixed, so every planned token value is host-visible.)
            let samples = rec.numComputedTokens == rec.effectiveTokenCount
            // A length-1 image span on the FINAL prompt token produces an
            // assignment shaped exactly like a decode row (n == 1, samples,
            // last known token) — but its placeholder token must take the
            // embedding-splice prefill path below, not the decode batch's
            // plain token-embedding path (PR#63 review). Genuine decode rows
            // never trip this: their positions are past every span.
            let finalTokenIsImageSpan =
                multimodalByID[id]?.containsSpan(at: rec.tokens.count - 1) ?? false
            let isDecode =
                n == 1 && samples && start == rec.tokens.count - 1 && !finalTokenIsImageSpan
            work.append(
                RowWork(rec: rec, start: start, count: n, samples: samples, isDecode: isDecode))
        }
        guard !work.isEmpty else { return nil }

        // Lazy offset/KV chains of every eager cache touched this step; ride
        // the step's asyncEval so the `+ L` offset advance can't accumulate
        // O(steps) of unevaluated graph (DAR-325).
        var cacheInnerState: [MLXArray] = []
        var recurrentEvaluations: [CBv2RequestID: CBv2RecurrentStateEvaluation] = [:]
        var logitDiagnostics: [CBv2LogitDiagnosticPacket] = []

        // Rectangular decode batch, in plan order.
        let decodeRows = work.filter(\.isDecode)
        var decodeSampled: MLXArray?
        var logprobSegments: [CBv2StepLogprobs] = []
        if !decodeRows.isEmpty {
            // A one-token prompt (or a one-token final prompt chunk) is
            // decode-SHAPED but still computes prompt tokens: count it as
            // the row's prefill chunk so `prefillChunks >= 1` holds for
            // every request. One compare per decode row, no allocation.
            for row in decodeRows where row.start < row.rec.request.promptTokens.count {
                row.rec.stampPrefillChunkLaunch(
                    tokens: 1, packed: false, vision: false, stripe: false,
                    launchNanos: wallStartedNanos)
            }
            let inputs = MLXArray(decodeRows.map { Int32($0.rec.tokens[$0.start]) })
                .reshaped([decodeRows.count, 1])
            let decodeIDs = decodeRows.map(\.rec.id)
            let diagnosticOffsets = logitDiagnostic == nil ? nil : decodeRows.map {
                Self.positionOffset(kvStates[$0.rec.id]!)
            }
            let (last, decodeInnerState, recurrent) = try decodeLogits(
                rowStates: decodeRows.map { kvStates[$0.rec.id]! }, tokens: inputs,
                ids: decodeIDs, phase: decodeRows.allSatisfy { $0.start < $0.rec.request.promptTokens.count }
                    ? .prefill : (decodeRows.allSatisfy { $0.start >= $0.rec.request.promptTokens.count }
                        ? .decode : .mixedFrontier))
            cacheInnerState.append(contentsOf: decodeInnerState)
            recurrentEvaluations.merge(recurrent) { _, _ in
                preconditionFailure("duplicate recurrent evaluation")
            }
            decodeSampled = sampler.sample(
                logits: last,
                params: decodeRows.map(\.rec.request.sampling),
                requestIDs: decodeRows.map(\.rec.id),
                stepIndex: stepCount,
                pendingSampledTokens: nil,  // finalize preceded: all confirmed
                rowContext: { decodeRows.map { Self.samplerRow($0.rec) } })
            if let stepLogprobs = sampler.takeStepLogprobs() {
                logprobSegments.append(stepLogprobs)
            }
            if let diagnosticOffsets {
                for (index, row) in decodeRows.enumerated() {
                    let isPromptFrontier = row.start < row.rec.request.promptTokens.count
                    if let packet = makeLogitDiagnostic(
                        logits: last[index], requestID: row.rec.id,
                        outputIndex: row.rec.generatedTokenCount,
                        phase: isPromptFrontier ? "prefill" : "decode",
                        batchIndex: index, batchSize: decodeRows.count,
                        seedToken: row.rec.tokens[row.start], cacheOffset: diagnosticOffsets[index])
                    {
                        logitDiagnostics.append(packet)
                    }
                }
            }
        }

        // Prompt chunks. Default shape is per-request [1, chunk]; when the
        // model AND the cache provider both prove rectangular per-row
        // semantics, equal-length chunks are coalesced into one layer-major
        // [B, chunk] forward so each layer's weights are read once for the
        // whole cohort. Span-bearing rows require the stronger model and
        // cache capabilities for row-local embeddings and attention masks.
        var prefillSampled: [CBv2RequestID: MLXArray] = [:]
        var evalTargets: [MLXArray] = []
        var packedIDs = Set<CBv2RequestID>()

        if packedPrefillSupported,
            let packedModel = model as? CBv2PackedPrefillSteppableModel
        {
            let canPackMultimodal =
                packedModel.supportsPackedMultimodalPrefill
                && cacheProvider.supportsPackedMultimodalSpans
            struct PackedGroup {
                let count: Int
                let samples: Bool
                var rows: [RowWork]
            }
            var groups: [PackedGroup] = []
            for row in work where !row.isDecode {
                guard row.rec.prefixReusePlan?.recurrentChunkSize == nil else { continue }
                // A multimodal request's text-only chunks remain packable.
                // A span-bearing chunk needs explicit rectangular embedding
                // and row-mask capability from both model and cache provider.
                // KNOWN LIMITATION (pre-existing; not changed by the timing
                // series): `chunkContext` is nil for `.causal` multimodal
                // input, so a causal span chunk classifies as text here —
                // packed without embedding splicing — and its launch stamp
                // records the path taken (`vision: false`). The shipped
                // causal model (Qwen) never reaches the packed cohort: the
                // recurrent/position gate below forces its solo path. Span
                // PRESENCE is `hasSpans(start:count:)` if this is revisited.
                let hasSpan =
                    multimodalByID[row.rec.id]?.chunkContext(
                        start: row.start, count: row.count) != nil
                if hasSpan && !canPackMultimodal { continue }
                // Recurrent packing v1 is text-only and position-free: rows
                // carrying explicit position state (vision MRoPE) take the
                // solo path rather than requiring stacked per-row positions.
                if (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec != nil,
                    hasSpan || row.rec.request.positionState != nil
                { continue }
                if let index = groups.firstIndex(where: {
                    $0.count == row.count && $0.samples == row.samples
                }) {
                    groups[index].rows.append(row)
                } else {
                    groups.append(
                        PackedGroup(count: row.count, samples: row.samples, rows: [row]))
                }
            }

            for group in groups where group.rows.count > 1 {
                var flatTokens: [Int32] = []
                flatTokens.reserveCapacity(group.rows.count * group.count)
                for row in group.rows {
                    flatTokens.append(
                        contentsOf: row.rec.tokens[row.start ..< row.start + row.count]
                            .map(Int32.init))
                }
                let inputs = MLXArray(flatTokens).reshaped([group.rows.count, group.count])
                let diagnosticOffsets = logitDiagnostic == nil ? nil : group.rows.map {
                    Self.positionOffset(kvStates[$0.rec.id]!)
                }
                let caches = eagerCaches(rowStates: group.rows.map { kvStates[$0.rec.id]! })
                let requirement: CBv2PrefillRequirement =
                    group.samples ? .lastPositionLogits : .evaluationOnly
                let spanContexts = group.rows.map {
                    multimodalByID[$0.rec.id]?.chunkContext(
                        start: $0.start, count: $0.count)
                }
                let output: MLXArray
                if spanContexts.contains(where: { $0 != nil }) {
                    output = try packedMultimodalChunksForward(
                        tokens: inputs,
                        starts: group.rows.map(\.start),
                        multimodal: group.rows.map { multimodalByID[$0.rec.id] },
                        spanContexts: spanContexts,
                        caches: caches,
                        requirement: requirement)
                } else if (model as? any CBv2RecurrentSteppableModel)?
                    .recurrentStateSpec != nil
                {
                    // Recurrent packed cohort: `targetForward` binds one
                    // recurrent state per row (row order == ids order), the
                    // prefill seam narrows, and each row's staged state
                    // evaluation is committed exactly as on the solo path.
                    let forward = try targetForward(
                        tokens: inputs, caches: caches,
                        ids: group.rows.map(\.rec.id),
                        requirement: requirement)
                    output = forward.logits
                    recurrentEvaluations.merge(forward.recurrent) { _, _ in
                        preconditionFailure("duplicate recurrent evaluation")
                    }
                    cacheInnerState.append(contentsOf: forward.innerState)
                } else {
                    output = try prefillOutput(
                        tokens: inputs, inputEmbeddings: nil, caches: caches,
                        requirement: requirement)
                }
                cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))

                if group.samples {
                    if let diagnosticOffsets {
                        for (index, row) in group.rows.enumerated() {
                            if let packet = makeLogitDiagnostic(
                                logits: output[index], requestID: row.rec.id,
                                outputIndex: row.rec.generatedTokenCount, phase: "prefill",
                                batchIndex: index, batchSize: group.rows.count,
                                seedToken: row.rec.tokens[row.start + row.count - 1],
                                cacheOffset: diagnosticOffsets[index])
                            {
                                logitDiagnostics.append(packet)
                            }
                        }
                    }
                    // Parity seam: the frontier logits, as this backend
                    // computed them, BEFORE the sampler touches them.
                    if let capture = prefillFrontierCapture() {
                        for (index, row) in group.rows.enumerated() {
                            capture(row.rec.id, output[index])
                        }
                    }
                    let sampled = sampler.sample(
                        logits: output,
                        params: group.rows.map(\.rec.request.sampling),
                        requestIDs: group.rows.map(\.rec.id),
                        stepIndex: stepCount,
                        pendingSampledTokens: nil,
                        rowContext: { group.rows.map { Self.samplerRow($0.rec) } })
                    for (index, row) in group.rows.enumerated() {
                        prefillSampled[row.rec.id] = sampled[index ..< index + 1]
                    }
                    if let stepLogprobs = sampler.takeStepLogprobs() {
                        logprobSegments.append(stepLogprobs)
                    }
                } else {
                    // One handle commits the whole rectangular trunk and
                    // every participating row's KV writes.
                    evalTargets.append(output)
                }
                packedIDs.formUnion(group.rows.map(\.rec.id))
                for (index, row) in group.rows.enumerated() {
                    row.rec.stampPrefillChunkLaunch(
                        tokens: row.count, packed: true, vision: spanContexts[index] != nil,
                        stripe: false, launchNanos: wallStartedNanos)
                }
                // Evidence, recorded where the rectangular forward actually
                // happened — never at the capability gate.
                packedPrefillRowsExecuted += group.rows.count
                packedPrefillGroupsExecuted += 1
            }
        }

        for row in work where !row.isDecode {
            let rec = row.rec
            if packedIDs.contains(rec.id) { continue }
            let slice = rec.tokens[row.start ..< row.start + row.count]
            let inputs = MLXArray(slice.map(Int32.init)).reshaped([1, row.count])
            let diagnosticOffset = logitDiagnostic == nil ? 0 : Self.positionOffset(kvStates[rec.id]!)
            let caches = eagerCaches(rowStates: [kvStates[rec.id]!])
            let requirement: CBv2PrefillRequirement =
                row.samples ? .lastPositionLogits : .evaluationOnly
            let output: MLXArray
            // ONE `multimodalByID` lookup per prefill row, reused below; the
            // span test iterates spans without allocating and runs only for
            // rows that carry multimodal input.
            let multimodal = multimodalByID[rec.id]
            let visionChunk = multimodal?.hasSpans(start: row.start, count: row.count) ?? false
            rec.stampPrefillChunkLaunch(
                tokens: row.count, packed: false, vision: visionChunk,
                stripe: !visionChunk && row.count > scheduler.config.prefillChunkSize,
                launchNanos: wallStartedNanos)
            if visionChunk, let multimodal {
                // Vision chunk (contains image spans): the NEW pinned path —
                // spliced input embeddings + span attention masks. Chunks of
                // the SAME request without spans fall through to the
                // untouched text path (pure function of has-spans).
                let forward = try multimodalChunkForward(
                    tokens: inputs, start: row.start, count: row.count,
                    id: rec.id, multimodal: multimodal, caches: caches,
                    requirement: requirement)
                output = forward.output
                recurrentEvaluations.merge(forward.recurrent) { _, _ in
                    preconditionFailure("duplicate recurrent evaluation")
                }
                cacheInnerState.append(contentsOf: forward.innerState)
            } else {
                let positions = rec.request.positionState?.promptSlice(
                    row.start ..< row.start + row.count)
                if (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec != nil
                    || positions != nil
                {
                    let forward = try targetForward(
                        tokens: inputs, caches: caches, ids: [rec.id],
                        positionIds: positions, requirement: requirement)
                    output = forward.logits  // already narrowed by targetForward
                    cacheInnerState.append(contentsOf: forward.innerState)
                    recurrentEvaluations.merge(forward.recurrent) { _, _ in
                        preconditionFailure("duplicate recurrent evaluation")
                    }
                } else {
                    output = try prefillOutput(
                        tokens: inputs, inputEmbeddings: nil, caches: caches,
                        requirement: requirement)
                }
            }
            cacheInnerState.append(contentsOf: eagerCacheInnerState(caches))
            if row.samples {
                if logitDiagnostic != nil,
                    let packet = makeLogitDiagnostic(
                        logits: output[0], requestID: rec.id,
                        outputIndex: rec.generatedTokenCount, phase: "prefill",
                        batchIndex: 0, batchSize: 1,
                        seedToken: rec.tokens[row.start + row.count - 1],
                        cacheOffset: diagnosticOffset)
                {
                    logitDiagnostics.append(packet)
                }
                if let capture = prefillFrontierCapture() {
                    capture(rec.id, output[0])
                }
                prefillSampled[rec.id] = sampler.sample(
                    logits: output,
                    params: [rec.request.sampling],
                    requestIDs: [rec.id],
                    stepIndex: stepCount,
                    pendingSampledTokens: nil,
                    rowContext: { [Self.samplerRow(rec)] })
                if let stepLogprobs = sampler.takeStepLogprobs() {
                    logprobSegments.append(stepLogprobs)
                }
            } else {
                // Cheap handle that forces this chunk's graph (incl. KV
                // writes) without materializing full logits on the host.
                evalTargets.append(output)
            }
        }

        // Assemble sampled tokens in plan order so a following pure-decode
        // plan (same membership, same order) can chain off this array.
        var pieces: [MLXArray] = []
        var sampledRows: [CBv2RequestID] = []
        var decodeIdx = 0
        for row in work {
            if row.isDecode {
                pieces.append(decodeSampled![decodeIdx ..< decodeIdx + 1])
                decodeIdx += 1
                sampledRows.append(row.rec.id)
            } else if let s = prefillSampled[row.rec.id] {
                pieces.append(s)
                sampledRows.append(row.rec.id)
            }
        }
        let sampledTokens: MLXArray? =
            pieces.isEmpty ? nil : (pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0))

        scheduler.markPendingSamples(ids: sampledRows)
        let rawAttention = attentionPacket?.takePendingForward()
        var toEval = evalTargets
        if let rawAttention { toEval.append(contentsOf: rawAttention.evaluationTargets) }
        for packet in logitDiagnostics { toEval.append(contentsOf: packet.evaluationTargets) }
        if let sampledTokens { toEval.append(sampledTokens) }
        for segment in logprobSegments { toEval.append(contentsOf: segment.evalTargets) }
        if !cacheInnerState.isEmpty {
            toEval.append(contentsOf: cacheInnerState)
            offsetChainEvalSteps += 1
        }
        let step = CBv2InFlightStep(
            assignments: work.map { (id: $0.rec.id, numTokens: $0.count) },
            participants: Set(work.map(\.rec.id)),
            sampledRows: sampledRows,
            sampledTokens: sampledTokens,
            evalTargets: evalTargets,
            computedRanges: Dictionary(
                uniqueKeysWithValues: work.map {
                    ($0.rec.id, $0.start ..< ($0.start + $0.count))
                }),
            wallStartedNanos: wallStartedNanos)
        step.logprobSegments = logprobSegments
        step.logitDiagnostics = logitDiagnostics
        step.attentionMetadata = attentionMetadata?.takePendingForward()
        step.attentionPacket = rawAttention
        step.recurrentEvaluations = recurrentEvaluations
        if hybridPrefixCache != nil || completeCheckpointCapture != nil {
            step.packedPrefixRows = packedIDs
            step.recurrentCheckpointChunkSizes = Dictionary(
                uniqueKeysWithValues: work.map { ($0.rec.id, $0.rec.plannedPrefillChunkSize) })
        }
        toEval.append(contentsOf: try prepareHistoricalCheckpoints(step))
        if step.historicalCheckpoints.isEmpty {
            asyncEval(toEval)
        } else {
            do { try withError { asyncEval(toEval) } }
            catch {
                // The step retains each capture permit until every borrowed
                // root on this launch stack has gone. Runtime failure then
                // retires KV through the existing cohort boundary.
                toEval.removeAll()
                throw error
            }
        }
        step.forwardShapes = shapes
        shapes?.attach()
        return step
    }

    func narrowPrefillOutput(
        _ logits: MLXArray, requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        switch requirement {
        case .evaluationOnly: return logits[0..., -1, 0 ..< 1]
        case .lastPositionLogits: return logits[0..., -1, 0...]
        }
    }

    // MARK: Vision prefill (span-containing chunks only)

    /// Forward one span-containing prefill chunk: splice the request's image
    /// embeddings over the scaled text embeddings, bind the chunk's span
    /// mask context to every layer cache for the duration of the graph
    /// build, and run the model's embedding forward. The context is unbound
    /// before returning, so every other computation (decode, text chunks,
    /// batchmates) sees nil — the mask can only ever apply to this one
    /// request's [1, chunk] rows.
    ///
    /// A 1-TOKEN chunk (`count == 1`) still splices its image embedding, but
    /// is NOT span-mask-bound: a chunk that small can only contain a
    /// length-1 block (blocks never split across chunks), and bidirectional
    /// attention within a single-token block is a no-op — the token attends
    /// only itself, identical under causal. Binding would trip the
    /// attention path's `L > 1` span precondition, so it is skipped
    /// (PR review: reachable via a length-1 span landing in a trailing
    /// 1-token prefill chunk).
    func multimodalChunkForward(
        tokens: MLXArray, start: Int, count: Int,
        id: CBv2RequestID, multimodal: CBv2ResolvedMultimodal,
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) throws -> (
        output: MLXArray,
        recurrent: [CBv2RequestID: CBv2RecurrentStateEvaluation],
        innerState: [MLXArray]
    ) {
        guard let mmModel = model as? CBv2MultimodalSteppableModel else {
            // Unreachable: EngineV2.submit gates multimodal requests on this
            // capability before they ever reach the scheduler.
            preconditionFailure(
                "CBv2 multimodal chunk reached a model without embedding-forward support")
        }
        let textEmbeddings = mmModel.embedPromptTokens(tokens)
        let spliced = CBv2MultimodalPlan.spliceEmbeddings(
            textEmbeddings: textEmbeddings,
            chunkStart: start,
            spans: multimodal.spansInChunk(start: start, count: count))
        let spanContext = multimodal.chunkContext(start: start, count: count)
        let bindables = spanContext != nil && count > 1
            ? caches.compactMap { $0 as? CBv2SpanMaskBinding } : []
        for bindable in bindables { bindable.bindSpanContext(spanContext) }
        defer { for bindable in bindables { bindable.bindSpanContext(nil) } }
        let positions = scheduler.record(for: id)?.request.positionState?.promptSlice(
            start ..< start + count)
        let deepstack = multimodal.deepstackInChunk(
            start: start, count: count, hidden: spliced.dim(-1), dtype: spliced.dtype)
        if !deepstack.isEmpty {
            precondition(
                (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec == nil,
                "CBv2 DeepStack currently supports attention-only models")
            guard let deepstackModel = model as? CBv2DeepstackMultimodalSteppableModel
            else {
                preconditionFailure(
                    "CBv2 DeepStack embeddings reached an unsupported model")
            }
            let full = try checkedModelForward(phase: .prefill) { deepstackModel.forward(
                tokens: tokens,
                inputEmbeddings: spliced,
                deepstackEmbeddings: deepstack,
                caches: caches,
                positionIds: positions) }
            return (
                narrowPrefillOutput(full, requirement: requirement),
                [:], [])
        }
        if (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec == nil,
            positions == nil
        {
            return (
                try prefillOutput(
                    tokens: tokens, inputEmbeddings: spliced, caches: caches,
                    requirement: requirement),
                [:], [])
        }
        let forward = try targetForward(
            tokens: tokens, caches: caches, ids: [id],
            positionIds: positions, inputEmbeddings: spliced,
            requirement: requirement)
        return (forward.logits, forward.recurrent, forward.innerState)
    }

    /// Rectangular counterpart of `multimodalChunkForward`. Each row keeps
    /// its own token embeddings, image splice coordinates, KV state, and
    /// optional span mask while the model traverses the cohort layer-major.
    func packedMultimodalChunksForward(
        tokens: MLXArray,
        starts: [Int],
        multimodal: [CBv2ResolvedMultimodal?],
        spanContexts: [CBv2SpanChunkContext?],
        caches: [CBv2AttendingLayerCache],
        requirement: CBv2PrefillRequirement
    ) throws -> MLXArray {
        guard let mmModel = model as? CBv2MultimodalSteppableModel else {
            preconditionFailure(
                "CBv2 packed multimodal chunk reached a model without embedding-forward support")
        }
        let batch = tokens.dim(0)
        let count = tokens.dim(1)
        precondition(
            starts.count == batch && multimodal.count == batch
                && spanContexts.count == batch,
            "CBv2 packed multimodal metadata must match batch \(batch)")

        let textEmbeddings = mmModel.embedPromptTokens(tokens)
        var embeddingRows: [MLXArray] = []
        embeddingRows.reserveCapacity(batch)
        for index in 0 ..< batch {
            let textRow = textEmbeddings[index ..< index + 1]
            if spanContexts[index] != nil, let rowMultimodal = multimodal[index] {
                embeddingRows.append(
                    CBv2MultimodalPlan.spliceEmbeddings(
                        textEmbeddings: textRow,
                        chunkStart: starts[index],
                        spans: rowMultimodal.spansInChunk(
                            start: starts[index], count: count)))
            } else {
                embeddingRows.append(textRow)
            }
        }
        let spliced = concatenated(embeddingRows, axis: 0)

        let bindables =
            count > 1 ? caches.compactMap { $0 as? CBv2PackedSpanMaskBinding } : []
        precondition(
            count == 1 || bindables.count == caches.count,
            "CBv2 packed multimodal prefill requires per-row span binding on every cache")
        for bindable in bindables { bindable.bindSpanContexts(spanContexts) }
        defer { for bindable in bindables { bindable.bindSpanContexts(nil) } }
        return try prefillOutput(
            tokens: tokens, inputEmbeddings: spliced, caches: caches,
            requirement: requirement)
    }

    // MARK: Finalization (deferred stop detection)

    private func finalize(_ step: CBv2InFlightStep, now: ContinuousClock.Instant) {
        // THE host sync — overlapped with the successor step's GPU work when
        // chained. All-prefill steps block on their eval targets instead so
        // graph pipelining stays bounded at two steps.
        let readbackStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        var host: [Int32] = []
        if let tokens = step.sampledTokens {
            host = tokens.asArray(Int32.self)
            CBv2CoreInstrumentation.recordHostSync()
        } else if !step.evalTargets.isEmpty {
            eval(step.evalTargets)
            CBv2CoreInstrumentation.recordHostSync()
        }
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.readback.wait", seconds: CFAbsoluteTimeGetCurrent() - readbackStart)
        }
        // THE one unconditional timing read of finalize: the readback-done
        // instant. Every per-row stamp below, the cumulative step wall, and
        // finishes driven by this finalize reuse it (no per-row reads).
        let readbackDoneNanos = DispatchTime.now().uptimeNanoseconds
        step.readbackDoneNanos = readbackDoneNanos
        stepWallNanosTotal = Self.saturatingAdd(
            stepWallNanosTotal, readbackDoneNanos &- step.wallStartedNanos)
        finalizeClockNanos = readbackDoneNanos
        defer { finalizeClockNanos = 0 }
        let tokenProducingRows = step.tokenProducingRows
        if !step.sampledRows.isEmpty {
            sampler.confirmSampledTokens(
                host.map(Int.init), requestIDs: step.sampledRows)
        }

        // Recurrent state follows the same one-step-late transaction as KV.
        // A discarded chained successor never becomes visible: rollback pops
        // exactly its newest generation, restoring the prior conv/SSM arrays.
        for (id, evaluation) in step.recurrentEvaluations {
            do {
                if step.discard.contains(id) {
                    try evaluation.rollback()
                } else {
                    try evaluation.commit()
                }
            } catch {
                preconditionFailure("CBv2 recurrent finalization failed for \(id): \(error)")
            }
        }

        let historicalFailure = commitHistoricalCheckpoints(step)
        captureRecurrentCheckpoints(step)

        // Publish only the exact work THIS finalized step launched. In the
        // chained path scheduler.numComputedTokens already includes N+1 here,
        // so consulting scheduler state would expose in-flight KV. MTP verify
        // rows wait for their accept/reject rollback below; every other row is
        // final after the host sync and recurrent commit above.
        let mtpVerifyIDs = Set(step.mtpRound?.verify?.rows.map(\.id) ?? [])
        for (id, range) in step.computedRanges
        where !step.discard.contains(id) && !mtpVerifyIDs.contains(id) {
            // A chained-plan preemption may already have detached this row
            // from `kvStates`, but the finalized step still owns the exact
            // state through `deferredReleases`. Publish before that fence
            // releases the pages so the completed prefix remains reusable.
            let detachedState = step.deferredReleases.first { $0.id == id }?.state
            publishFinalizedResidentBlocks(
                requestID: id, safeComputedEnd: range.upperBound,
                state: detachedState)
        }

        // Materialize any lazy logprob gathers at this same boundary (they
        // rode the step's asyncEval, so this reads back finished results —
        // no extra GPU work, and only when a row asked for logprobs).
        var logprobsByID: [CBv2RequestID: CBv2TokenLogprob] = [:]
        if !step.logprobSegments.isEmpty {
            var tokenByID: [CBv2RequestID: Int] = [:]
            for (i, id) in step.sampledRows.enumerated() { tokenByID[id] = Int(host[i]) }
            for segment in step.logprobSegments {
                let sampledTokens = segment.rows.map { tokenByID[$0] ?? -1 }
                let assembled = CBv2Logprobs.assemble(
                    segment.gathered, sampledTokens: sampledTokens,
                    topLogprobsPerRow: segment.topLogprobsPerRow)
                for (row, logprob) in zip(segment.rows, assembled) {
                    if let logprob { logprobsByID[row] = logprob }
                }
            }
        }

        var finalizedPlainWork = false
        var finalizedPlainRowCount = 0
        var committedPlainRows: [CBv2RequestID] = []
        let mtpSeedIDs = Set(step.mtpRound?.seedRows.map(\.id) ?? [])
        var deferredMTPFinishes: [CBv2RequestID: CBv2FinishReason] = [:]

        func finishPlainRow(_ id: CBv2RequestID, reason: CBv2FinishReason) {
            if mtpSeedIDs.contains(id) {
                deferredMTPFinishes[id] = reason
            } else {
                finishRequest(id, reason: reason)
            }
        }
        for (i, id) in step.sampledRows.enumerated() {
            if step.discard.contains(id) { continue }
            guard let rec = scheduler.record(for: id) else { continue }
            finalizedPlainWork = true
            finalizedPlainRowCount += 1
            let token = Int(host[i])
            for packet in step.logitDiagnostics where packet.requestID == id {
                packet.seedToken = rec.tokens.last
                packet.reconcile(accepted: 0, confirmed: 1, drafts: [], targets: [token])
            }
            if let metadata = step.attentionMetadata,
                metadata.state.configuration.requestID == id.raw
            {
                metadata.state.confirm(
                    requestID: id.raw, outputIndex: rec.generatedTokenCount,
                    seed: rec.tokens.last, target: token)
            }
            if let packet = step.attentionPacket,
                packet.state.configuration.requestID == id.raw {
                packet.state.metadata.confirm(
                    requestID: id.raw, outputIndex: rec.generatedTokenCount,
                    seed: rec.tokens.last, target: token)
            }
            let firstToken = rec.generatedTokenCount == 0
            scheduler.recordSampled(id: id, token: token)
            // Timing stamps on the record already in hand (field writes on
            // an existing object; the instant is the readback read above).
            rec.recordStepParticipation(step: step, batchRows: tokenProducingRows)
            if firstToken {
                rec.stampFirstToken(readbackDoneNanos: readbackDoneNanos)
            } else {
                rec.timing.decodeSteps &+= 1
                decodeRowsTotal = Self.saturatingAdd(decodeRowsTotal, 1)
            }
            if let constraintFailure = sampler.tokenConstraintFailure(for: id) {
                finishRequest(id, reason: .error(constraintFailure))
                continue
            }

            // A stop TOKEN's text is never emitted (OpenAI behavior: the
            // stop token terminates the stream and its rendering is
            // excluded from content). It still counts toward
            // usage.completionTokens (recordSampled above) and still rides
            // in the delta's raw `tokens` — see the CBv2Event.delta doc.
            // Skipping the detokenizer push keeps its held-back text
            // intact for the finish-time flush.
            let isStopToken = rec.request.stopTokens.contains(token)
            if !isStopToken { committedPlainRows.append(id) }
            let detokenizer = detokenizers[id]
            let logprobs = logprobsByID[id].map { [$0] }

            // Stop-string detection needs the holdback scan synchronously to
            // gate the deterministic one-step-late `.stop` finish; passthrough
            // requests can't stop on a string, so their decode + emit move
            // OFF the step thread (finding 2 — detokQueue). The buffer slot
            // is charged NOW, on the engine thread (`reserveEmission`), so
            // the backpressure pause still gates this request's scheduling
            // even though the emit itself is deferred — otherwise a slow
            // detokenizer/consumer could not fill the bounded buffer until
            // the queued blocks ran, and generation would run unbounded
            // ahead of it (PR#62 review).
            var matchedStopString = false
            if rec.request.stopStrings.isEmpty {
                let stream = stream(for: id)
                // First token only: arm the detok-delay probe under the
                // reservation's own lock acquisition (measured in `emit`).
                stream?.reserveEmission(
                    firstEmissionArmedNanos: firstToken ? readbackDoneNanos : 0)
                detokQueue.async {
                    let text = isStopToken ? "" : (detokenizer?.push([token]) ?? "")
                    stream?.emit(
                        .delta(text: text, tokens: [token], logprobs: logprobs),
                        consumingReservation: true)
                }
            } else {
                let detokStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
                let text = isStopToken ? "" : (detokenizer?.push([token]) ?? "")
                if CBv2StepProfiler.enabled {
                    CBv2StepProfiler.record(
                        "v2.detok.push", seconds: CFAbsoluteTimeGetCurrent() - detokStart)
                }
                let emitStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
                stream(for: id)?.emit(.delta(text: text, tokens: [token], logprobs: logprobs))
                if CBv2StepProfiler.enabled {
                    CBv2StepProfiler.record(
                        "v2.stream.emit", seconds: CFAbsoluteTimeGetCurrent() - emitStart)
                }
                matchedStopString = detokenizer?.matchedStopString ?? false
            }

            // Stop detection — one step late by construction. Deadlines are NO
            // LONGER checked here: a request that just produced a token is
            // making progress and must never be killed mid-generation. Lease
            // expiry is evaluated centrally in `processLeaseExpiry` at the top
            // of each step, and this confirmed token refreshes the decode lease
            // in `refreshProgressLeases` below.
            if isStopToken {
                finishPlainRow(id, reason: .stop)
            } else if matchedStopString {
                finishPlainRow(id, reason: .stop)
            } else if rec.generatedTokenCount >= rec.request.maxTokens {
                finishPlainRow(id, reason: .length)
            }
        }

        // MTP round steps: seed-carry capture + the verify accept-walk run
        // at this same host-sync boundary (their arrays rode the step's
        // asyncEval), AFTER the plain loop above (seed bonus tokens are
        // confirmed there) and BEFORE the fenced frees below.
        if step.mtpRound != nil {
            finalizeMTPRound(step)
        }
        mtp?.recordCommittedDecodeBaseline(
            measurement: step.mtpMeasurement, completedAtNanos: readbackDoneNanos,
            sampledRows: step.sampledRows, finalizedPlainRowCount: finalizedPlainRowCount,
            // Only steady pipeline commits qualify. The final draining step
            // has no successor; stopped/cancelled rows mark it discarded.
            hasChainedSuccessor: inFlight?.chained == true
                && inFlight?.sampledRows == step.sampledRows
                && inFlight?.discard.isEmpty == true)
        if let measurement = step.mtpMeasurement {
            let completedAtNanos = DispatchTime.now().uptimeNanoseconds
            let elapsed = completedAtNanos &- step.wallStartedNanos
            let verifiedRows = step.mtpRound?.finalizedVerifyIDs ?? []
            let actualCommittedRows = verifiedRows.isEmpty
                ? committedPlainRows : verifiedRows.sorted { $0.raw < $1.raw }
            let expectedRows = step.mtpRound?.verify?.rows.map(\.id) ?? step.sampledRows
            let committedRows = step.discard.isEmpty
                && Set(actualCommittedRows) == Set(expectedRows)
                ? actualCommittedRows : []
            mtp?.recordStepCost(
                measurement,
                wallTimeNanos: elapsed,
                finalizedPlainWork: finalizedPlainWork,
                finalizedSeedIDs: step.mtpRound?.finalizedSeedIDs ?? [],
                finalizedVerification: !(step.mtpRound?.finalizedVerifyIDs.isEmpty ?? true),
                claimedSeedCostNanos: step.mtpRound?.claimedSeedCostNanos ?? 0,
                completedAtNanos: completedAtNanos,
                committedRows: committedRows,
                committedTokenCount: verifiedRows.isEmpty
                    ? committedPlainRows.count : (step.mtpRound?.committedVerifyTokenCount ?? 0))
        }
        step.forwardShapes?.complete()
        // Preserve the normal adaptive-cost clock above. These compact arrays
        // already rode this step's fence; never launch another evaluation.
        materializeLogitDiagnostics(step)
        materializeAttentionPacket(step)
        if let metadata = step.attentionMetadata {
            metadata.state.retire(discarded: step.discard.contains(
                CBv2RequestID(metadata.state.configuration.requestID)))
            step.attentionMetadata = nil
        }
        for (id, reason) in deferredMTPFinishes {
            if let state = mtp?.takeAssistantState(for: id) {
                step.mtpRound?.deferredAssistantReleases.append(state)
            }
            finishRequest(id, reason: reason)
        }

        // Fenced frees: rows finished/cancelled while this step was in
        // flight. Scrub the wasted-token KV tail, then retire (donate to
        // the prefix cache when eligible, else release).
        for (id, state, rollbackOne, donation, recurrent, hybridPublication, ownership, terminalDelivery) in
            step.deferredReleases
        {
            if rollbackOne {
                for sequence in state { sequence?.rollback(1) }
            }
            let publicationDeferred = retire(
                state: state, donating: donation, hybridPublication: hybridPublication,
                afterPublication: { [self] in
                    if let ownership { retireStream(id, matching: ownership) }
                    terminalDelivery?()
                })
            releaseRecurrentState(recurrent)
            if !publicationDeferred, let ownership {
                retireStream(id, matching: ownership)
            }
            if !publicationDeferred { terminalDelivery?() }
        }
        if let mtp {
            for state in step.mtpRound?.deferredAssistantReleases ?? [] {
                mtp.releaseDetachedAssistantState(state)
            }
        }

        if let historicalFailure {
            // MLX supplies no typed stream-health distinction. Treat every
            // native copy error as a request failure, never as a healthy cache
            // miss. The cohort emitted no samples; private work and ordinary
            // deferred releases have drained before existing request teardown.
            for id in step.participants where scheduler.record(for: id) != nil {
                finishRequest(id, reason: .error(String(describing: historicalFailure)), now: now)
            }
        }

        // Refresh progress leases from CONFIRMED work this step — covers plain
        // decode, chained decode, prefill chunks, and MTP rounds uniformly
        // (MTP rows are `participants` and were confirmed above via
        // `finalizeMTPRound`). This is the single point where confirmed
        // finalized progress refreshes a lease, never optimistic planning.
        //
        // Read the clock FRESH here rather than reusing the caller's `now`:
        // that instant was captured before this function's blocking host
        // readback, so stamping progress with it backdates confirmation by
        // the sync duration — with a stepTimeout configured above a progress
        // lease, a healthy long readback could leave the just-refreshed lease
        // already expired at the next boundary (PR#82 review). Progress is
        // confirmed NOW, after the sync.
        refreshProgressLeases(step, now: config.clock.now())
    }

    /// Refresh each live participant's progress lease and its reconciled usage
    /// snapshot from CONFIRMED finalized state. A confirmed sampled/generated
    /// token refreshes the decode lease; a confirmed prefill-chunk advance
    /// refreshes the prefill lease.
    private func refreshProgressLeases(_ step: CBv2InFlightStep, now: ContinuousClock.Instant) {
        for id in step.participants {
            guard let rec = scheduler.record(for: id) else { continue }
            if var lease = leasesByID[id] {
                lease.recordProgress(
                    now: now,
                    computedTokens: rec.numComputedTokens,
                    generatedTokens: rec.generatedTokenCount)
                leasesByID[id] = lease
            }
            // Publish the reconciled usage once tokens start flowing (prompt
            // count with zero completion was already seeded at enqueue, so a
            // still-prefilling row needs no lock traffic here).
            if rec.generatedTokenCount > 0 {
                setUsageSnapshot(
                    id,
                    CBv2Usage(
                        promptTokens: rec.request.promptTokens.count,
                        completionTokens: rec.generatedTokenCount))
            }
        }
    }

    // MARK: Request completion

    /// `nowNanos`: the caller's already-taken timing instant (a step
    /// boundary finishing several rows shares ONE read). Absent, the
    /// in-progress finalize / launch read is reused; a fresh read is the
    /// last resort (direct callers outside any step). `now`: the step's
    /// boundary `ContinuousClock` read (`stepNow`), used only to close a
    /// paused row's backpressure interval — absent, one read per paused
    /// finish.
    func finishRequest(
        _ id: CBv2RequestID, reason: CBv2FinishReason, nowNanos: UInt64? = nil,
        now: ContinuousClock.Instant? = nil
    ) {
        // Ids are legally reusable after finish: drop the per-id capacity
        // requeue count on EVERY finish path (including the error-finish
        // that exhausted it), or a reused id inherits the previous
        // attempt tally and its first transient KV-capacity trip
        // error-finishes immediately instead of requeueing (PR#62 review).
        capacityRequeues.removeValue(forKey: id)
        residentPrefixCursorByID.removeValue(forKey: id)
        recurrentCheckpointGeometry.removeValue(forKey: id)
        multimodalByID.removeValue(forKey: id)
        // Ids are reusable after finish: drop the per-id lease and usage
        // snapshot so a reused id starts fresh (a stale lease would otherwise
        // expire the new request early).
        leasesByID.removeValue(forKey: id)
        clearUsageSnapshot(id)
        // MTP: ids are reusable, so every per-id trace (carry, marks) must
        // go on every finish path.
        mtp?.requestDidFinish(id)
        guard let rec = scheduler.finish(id: id, reason: reason) else {
            // A duplicate finish can arrive after the scheduler row was
            // removed but before an in-flight backend reference retires.
            // Preserve that generation fence; taking it here would
            // acknowledge ownership and permit unsafe immediate ID reuse.
            let ownsDeferredBackendState =
                inFlight?.deferredReleases.contains {
                    $0.id == id && $0.ownership != nil
                } ?? false
            if ownsDeferredBackendState {
                return
            }
            let usage = takePrefixUsage(requestID: id, promptTokens: 0, completionTokens: 0)
            takeStream(id)?.finish(reason: reason, usage: usage)
            return
        }
        var hybridPublication = hybridPublicationIntent(for: rec, reason: reason)
        let holdsCompleteExportReservation = completeCheckpointCapture != nil && hybridPublication != nil
        if holdsCompleteExportReservation {
            hybridPublication?.retiredReservation = completeCheckpointCapture?.codec.admission.detachReservation(id: id)
        } else {
            capacity?.releaseAll(id: id)
        }
        // Retire the id from the sampler's configured fingerprint: the id is
        // legally reusable after this finish, and a reused id with an
        // identical row set must reconfigure, not inherit stale penalties /
        // RNG progress (PR#62 review).
        sampler.requestDidFinish(id)

        // Keep the stream generation registered until every in-flight
        // backend reference is fenced. This blocks immediate ID reuse and
        // separates consumer terminal delivery from engine ownership.
        let stream = stream(for: id)
        let detokenizer = detokenizers.removeValue(forKey: id)
        prefixUsageByID[id]?.boundarySplits = rec.prefixReplayBoundarySplits
        var usage = takePrefixUsage(
            requestID: id, promptTokens: rec.request.promptTokens.count,
            completionTokens: rec.generatedTokenCount)
        // Fold the per-request timing in ONCE. The instant is, in order:
        // the caller's boundary read (cancel / lease expiry share one per
        // step), the in-progress finalize's readback-done read, the
        // in-progress launch's wall-start read (allocation failure), and
        // only for direct callers outside any step a fresh read.
        // Cost model: the `usage.timing` setter boxes the struct — ONE
        // `CBv2RequestTimingBox` allocation per request lifetime, at the
        // single assignment on whichever delivery path runs (synchronous
        // below, or async at delivery once the detok delay is folded in);
        // never on the step path. The raw value travels unboxed until then.
        if rec.pausedSince != nil { rec.recordResumed(now: now ?? config.clock.now()) }
        let exportedTiming = rec.exportTiming(
            finishedNanos: nowNanos
                ?? (finalizeClockNanos != 0
                    ? finalizeClockNanos
                    : launchClockNanos != 0
                        ? launchClockNanos : DispatchTime.now().uptimeNanoseconds))
        let usesAsyncDetokenization = rec.request.stopStrings.isEmpty
        if !usesAsyncDetokenization { usage.timing = exportedTiming }  // the ONE box
        let terminalQueue = detokQueue
        let terminalDelivery: () -> Void = { [usage, exportedTiming] in
            // Passthrough requests emit deltas on the detok queue; the
            // trailing flush + terminal MUST ride the same queue so they land
            // AFTER those deltas (FIFO ordering). Stop-string requests stay
            // synchronous. The first-token detok delay is read from the
            // stream (its own lock) at delivery time — after the deferred
            // first emit by FIFO — never from the scheduler record.
            if usesAsyncDetokenization {
                terminalQueue.async {
                    let trailing = detokenizer?.flush() ?? ""
                    if !trailing.isEmpty {
                        stream?.emit(.delta(text: trailing, tokens: [], logprobs: nil))
                    }
                    var timing = exportedTiming
                    timing.detokDelayFirstNanos = stream?.firstEmissionDelayNanos ?? 0
                    var delivered = usage
                    delivered.timing = timing  // the ONE box
                    stream?.finish(reason: reason, usage: delivered)
                }
            } else {
                let trailing = detokenizer?.flush() ?? ""
                if !trailing.isEmpty {
                    stream?.emit(.delta(text: trailing, tokens: [], logprobs: nil))
                }
                stream?.finish(reason: reason, usage: usage)
            }
        }

        var ownershipReleaseDeferred = false
        if hybridPublication == nil { discardHybridCheckpoints(id) }
        if let state = kvStates.removeValue(forKey: id) {
            let donation = donationIntent(for: rec, reason: reason, state: state)
            if let inFlight, inFlight.participants.contains(id) {
                // The in-flight step still references this state — fence the
                // free behind its completion; roll back the wasted token iff
                // that step sampled for this row.
                inFlight.discard.insert(id)
                inFlight.deferredReleases.append(
                    (
                        id: id, state: state,
                        rollbackOne: inFlight.sampledRows.contains(id),
                        donation: donation,
                        recurrent: recurrentStates.removeValue(forKey: id),
                        hybridPublication: hybridPublication,
                        ownership: stream,
                        terminalDelivery: terminalDelivery
                    ))
                ownershipReleaseDeferred = true
            } else {
                ownershipReleaseDeferred = retire(
                    state: state, donating: donation, hybridPublication: hybridPublication,
                    afterPublication: { [self] in
                        if let stream { retireStream(id, matching: stream) }
                        terminalDelivery()
                    })
                releaseRecurrentState(recurrentStates.removeValue(forKey: id))
            }
        } else {
            releaseRecurrentState(recurrentStates.removeValue(forKey: id))
            if holdsCompleteExportReservation {
                ownershipReleaseDeferred = retire(
                    state: [], donating: nil, hybridPublication: hybridPublication,
                    afterPublication: { [self] in
                        if let stream { retireStream(id, matching: stream) }
                        terminalDelivery()
                    })
            }
        }

        if !ownershipReleaseDeferred, let stream {
            retireStream(id, matching: stream)
        }
        if !ownershipReleaseDeferred {
            terminalDelivery()
        }
    }

    // MARK: Prefix-cache donation (engine thread → donation queue)

    /// Donation intent for a finished request, or nil when donation does
    /// not apply. Only NATURAL completions donate: their KV is a complete,
    /// confirmed prefix. The last confirmed token was sampled but never fed
    /// through the model, so it is dropped at donation time. The intent
    /// carries the request's cache salt so the entry lands in the donor's
    /// salt scope (TB-007).
    private func donationIntent(
        for rec: CBv2ScheduledRequest, reason: CBv2FinishReason, state: [CBv2SequenceKV?]
    ) -> CBv2DonationIntent? {
        guard prefixCache != nil else { return nil }
        guard rec.request.prefixCacheEnabled else { return nil }
        guard cbv2LayerKindsAllowPrefixReuse(layerKinds) else { return nil }
        // Vision requests NEVER donate (v1 policy, enforced in BOTH
        // directions — lookup is skipped in `EngineV2.makePrefixLookup`): the
        // prefix cache keys on token-id chain hashes, and an image span's
        // placeholder ids are identical for every image — a donated vision
        // prefix would be silently served to a request with different image
        // content. Image-digest extra keys in the block hash (vLLM-V1
        // `extra_keys`) are the documented follow-up.
        guard rec.request.multimodal == nil else { return nil }
        switch reason {
        case .stop, .length: break
        // A deadline/watchdog terminal aborts an incomplete request — its KV is
        // not a confirmed, complete prefix, so it never donates.
        case .cancelled, .error, .terminal: return nil
        }
        // Donation requires the full prompt to have been processed (at
        // least one sampled token) — mid-prefill finishes carry partial KV.
        guard rec.generatedTokenCount >= 1, rec.tokens.count > 1 else { return nil }
        return CBv2DonationIntent(
            requestID: rec.request.prefixCacheReceiptID ?? rec.id,
            tokens: Array(rec.tokens.dropLast()),
            cacheSalt: rec.request.cacheSalt)
    }

    /// Retire a finished request's KV state: donate to the prefix cache
    /// (snapshots graph-built HERE on the engine thread, so views over
    /// shared paged slabs are consistent with in-flight writes; hashing +
    /// indexing + optional materialization run on the donation queue), then
    /// release the storage back on the engine queue (the paged pool is
    /// engine-thread-affine).
    @discardableResult
    private func retire(
        state: [CBv2SequenceKV?], donating donation: CBv2DonationIntent?,
        hybridPublication: CBv2DonationIntent? = nil,
        afterPublication: @escaping () -> Void = {}
    ) -> Bool {
        if let capture = completeCheckpointCapture, let intent = hybridPublication {
            pendingDonationReleaseCount += 1
            let retired = CBv2CompleteCheckpointRetiredState(state: state)
            let handoff = CBv2Handoff(value: afterPublication)
            capture.publish(intent: intent, state: state) { [self] positions in
                engineQueue.async { [self] in
                    retired.release(backend: backend)
                    intent.retiredReservation?.release()
                    pendingDonationReleaseCount -= 1
                    capture.reportPublication(receiptID: intent.receiptID, positions: positions)
                    handoff.value()
                    publishGauges()
                    completeDrainIfReady()
                }
            }
            return true
        }
        if let cache = hybridPrefixCache, let intent = hybridPublication {
            var snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?] = []
            var bytes = 0
            var valid = state.count == layerKinds.count
            for row in state {
                guard let row else { valid = false; break }
                let (sum, overflow) = bytes.addingReportingOverflow(row.byteCount)
                guard !overflow, row.absoluteOffset >= intent.tokens.count else {
                    valid = false
                    break
                }
                bytes = sum
                snapshots.append(row.snapshot())
            }
            if valid {
                pendingDonationReleaseCount += 1
                let handoff = CBv2Handoff(value: (state, afterPublication))
                cache.publish(
                    requestID: intent.requestID, tokens: intent.tokens, cacheSalt: intent.cacheSalt,
                    receiptID: intent.receiptID,
                    kv: snapshots, backingBytes: bytes
                ) { [self] in
                    engineQueue.async { [self] in
                        backend.release(handoff.value.0)
                        pendingDonationReleaseCount -= 1
                        handoff.value.1()
                        publishGauges()
                        completeDrainIfReady()
                    }
                }
                return true
            }
            discardHybridCheckpoints(intent.requestID)
        }
        guard prefixCache != nil, let donation else {
            backend.release(state)
            return false
        }
        let queued = enqueueDonation(state: state, intent: donation)
        if !queued {
            backend.release(state)
        }
        return false
    }

    /// Build immutable-by-contract snapshot handles on the engine queue and
    /// hand all expensive work to the serial donation queue.
    private func enqueueDonation(
        state: [CBv2SequenceKV?], intent: CBv2DonationIntent
    ) -> Bool {
        guard let prefixCache else { return false }
        let tokenCount = intent.tokens.count
        guard stateCoversDonation(state, tokenCount: tokenCount) else { return false }
        // Opt-in sliding-row donation (`CBv2SlidingWindowDonating`). Asked
        // BEFORE anything is built: a sliding snapshot is `windowCount ×
        // window` positions of K/V — 200 MiB on gemma-4 — so a cache that
        // does not persist windows must not pay for the graph at all.
        let windowDonor = prefixCache as? any CBv2SlidingWindowDonating
        let donatesWindows = windowDonor?.wantsSlidingWindowDonation ?? false
        var built: [(keys: MLXArray, values: MLXArray, offset: Int)?] = []
        var sliding: [(keys: MLXArray, values: MLXArray, offset: Int)?] = []
        built.reserveCapacity(layerKinds.count)
        if donatesWindows { sliding.reserveCapacity(layerKinds.count) }
        for (i, kind) in layerKinds.enumerated() {
            var cacheable = kind.sharesKVWithLayer == nil
            var isSliding = false
            if case .slidingWindow = kind.attention {
                cacheable = false
                isSliding = true
            }
            if donatesWindows {
                // Storage-owning sliding rows only, and NEVER truncated to
                // `tokenCount`: the ring already holds exactly the last
                // `window` positions ending at the row's absolute offset,
                // which is the span the sidecar persists. A KV-shared row
                // borrows its source's storage, so donating it would
                // double-write the same bytes.
                sliding.append(
                    isSliding && kind.sharesKVWithLayer == nil
                        ? state[i]?.snapshot()
                        : nil)
            }
            guard cacheable, let seq = state[i] else {
                built.append(nil)
                continue
            }
            let snap = seq.snapshot()
            if snap.offset == tokenCount {
                built.append(snap)
            } else {
                built.append(
                    (
                        keys: snap.keys[.ellipsis, 0 ..< tokenCount, 0...],
                        values: snap.values[.ellipsis, 0 ..< tokenCount, 0...],
                        offset: tokenCount
                    ))
            }
        }
        let layerKinds = self.layerKinds
        let handoff = CBv2Handoff(
            value: (state: state, snapshots: built, sliding: sliding, intent: intent))
        pendingDonationReleaseCount += 1
        // Strong self on purpose: the deferred release is a pending
        // obligation of this loop — it must survive until the donation
        // lands, or the retired state leaks its pages (the paged pool is
        // engine-thread-affine, so the free hops back to the engine queue).
        // No cycle: the block releases its captures once it runs.
        donationQueue.async {
            if let windowDonor, donatesWindows {
                windowDonor.donate(
                    requestID: handoff.value.intent.requestID,
                    tokens: handoff.value.intent.tokens,
                    snapshots: handoff.value.snapshots,
                    slidingSnapshots: handoff.value.sliding,
                    layerKinds: layerKinds,
                    cacheSalt: handoff.value.intent.cacheSalt)
            } else {
                prefixCache.donate(
                    requestID: handoff.value.intent.requestID,
                    tokens: handoff.value.intent.tokens,
                    snapshots: handoff.value.snapshots, layerKinds: layerKinds,
                    cacheSalt: handoff.value.intent.cacheSalt)
            }
            self.releaseDonationStateOnEngineQueue(handoff.value.state)
        }
        return true
    }

    private func stateCoversDonation(
        _ state: [CBv2SequenceKV?], tokenCount: Int
    ) -> Bool {
        guard tokenCount > 1 else { return false }
        var cacheableLayers = 0
        for (i, kind) in layerKinds.enumerated() {
            var cacheable = kind.sharesKVWithLayer == nil
            if case .slidingWindow = kind.attention { cacheable = false }
            guard cacheable else { continue }
            guard let sequence = state[i], sequence.absoluteOffset >= tokenCount else { return false }
            cacheableLayers += 1
        }
        return cacheableLayers > 0
    }

    private func takePrefixUsage(
        requestID: CBv2RequestID, promptTokens: Int, completionTokens: Int
    ) -> CBv2Usage {
        let prefix = prefixUsageByID.removeValue(forKey: requestID) ?? .disabled
        let saved = prefixHitTokens.removeValue(forKey: requestID) ?? prefix.prefillTokensSaved
        return CBv2Usage(
            promptTokens: promptTokens, completionTokens: completionTokens,
            prefixCacheHitTokens: saved,
            prefixCacheOutcome: prefix.outcome,
            prefixCacheTier: prefix.tier,
            prefixCacheMatchedTokens: prefix.matchedTokens,
            prefixCachePrefillTokensSaved: saved,
            prefixCacheStrategy: prefix.strategy,
            prefixCacheReplayTokens: prefix.replayTokens,
            prefixCacheBoundarySplits: prefix.boundarySplits)
    }

    private func releaseDonationStateOnEngineQueue(_ state: [CBv2SequenceKV?]) {
        let handoff = CBv2Handoff(value: state)
        engineQueue.async { [self] in
            backend.release(handoff.value)
            assert(pendingDonationReleaseCount > 0, "unbalanced donation release completion")
            pendingDonationReleaseCount = max(0, pendingDonationReleaseCount - 1)
            publishGauges()
            completeDrainIfReady()
        }
    }

    // MARK: Sampler row context

    /// Full per-row sampler context (confirmed history only; in-flight
    /// chained samples travel separately as `pendingSampledTokens`).
    static func samplerRow(_ rec: CBv2ScheduledRequest) -> CBv2SamplerRow {
        CBv2SamplerRow(
            id: rec.id,
            params: rec.request.sampling,
            promptTokens: rec.request.promptTokens,
            outputTokens: Array(rec.tokens.dropFirst(rec.request.promptTokens.count)),
            tokenConstraint: rec.request.tokenConstraint,
            maxTokens: rec.request.maxTokens)
    }

    // MARK: Boundary housekeeping

    private func processCancellations(now: ContinuousClock.Instant) {
        stateLock.lock()
        let cancels = pendingCancels
        stateLock.unlock()

        // Which cancels are fully resolved this boundary (drop from the set).
        // A cancel with no scheduler record AND a still-registered stream is
        // an EARLY cancel racing `enqueue` — keep it so the enqueue block
        // refuses to start the request (PR#62 review). A cancel with neither
        // is stale (request already finished) — drop it.
        var resolved: [CBv2RequestID: UInt64] = [:]
        for (id, cancelledGeneration) in cancels {
            stateLock.lock()
            let currentGeneration = streamGenerations[id]
            stateLock.unlock()
            guard currentGeneration == cancelledGeneration else {
                resolved[id] = cancelledGeneration
                continue
            }
            if scheduler.record(for: id) != nil {
                finishRequest(
                    id, reason: .cancelled, nowNanos: boundaryFinishNanos(), now: now)
                resolved[id] = cancelledGeneration
            }
        }
        guard !resolved.isEmpty else { return }
        stateLock.lock()
        for (id, generation) in resolved where pendingCancels[id] == generation {
            pendingCancels.removeValue(forKey: id)
        }
        stateLock.unlock()
    }

    /// Consume a remembered early cancel for `id` at enqueue time. Returns
    /// true when the request was cancelled before it started (the caller
    /// must finish its stream and never enqueue it).
    private func consumeEarlyCancel(_ id: CBv2RequestID) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard let generation = pendingCancels[id],
            streams[id] != nil,
            generation == streamGenerations[id]
        else {
            pendingCancels.removeValue(forKey: id)
            return false
        }
        pendingCancels.removeValue(forKey: id)
        return true
    }

    // MARK: Deadline leases

    /// Arm the monotonic lease set for a freshly enqueued request and seed its
    /// reconciled-usage snapshot (prompt tokens known, zero completion). Under
    /// the kill-switch this is the legacy single total-lifetime wall.
    private func armLease(for request: CBv2Request) {
        let now = config.clock.now()
        let lease: CBv2RequestLeaseState
        if config.useLegacyRequestTimeout {
            lease = .legacy(now: now, wall: config.requestTimeout)
        } else {
            lease = CBv2RequestLeaseState(
                now: now,
                admissionLease: config.admissionLease,
                prefillLease: config.prefillProgressLease,
                decodeLease: config.decodeProgressLease,
                backpressureLease: config.backpressureLease,
                safety: CBv2SafetyCeiling.duration(
                    promptTokens: request.promptTokens.count,
                    maxTokens: request.maxTokens,
                    admissionLease: config.admissionLease,
                    decodeFloorTPS: config.safetyCeilingDecodeFloorTPS),
                computedTokens: 0,
                generatedTokens: 0)
        }
        leasesByID[request.id] = lease
        setUsageSnapshot(
            request.id,
            CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
    }

    /// End the admission lease for every row that began engine work this step,
    /// and grant a fresh progress window to RE-admitted rows (already-admitted
    /// rows crossing waiting→running again after preemption / capacity
    /// requeue): their demotion-time window kept ticking through the
    /// stall-exempt queue wait and may be pre-expired (PR#82 review).
    /// `waitingBefore` is the waiting set snapshotted BEFORE plan() — only
    /// rows that crossed out of it this step qualify; continuing running rows
    /// must NOT refresh here or genuine decode stalls would be masked
    /// (plan.assignments lists every scheduled row each step, not just
    /// admissions). Idempotent; a preempted requeue never re-arms admission.
    private func markAdmitted(
        _ plan: CBv2StepPlan, now: ContinuousClock.Instant,
        readmittedFrom waitingBefore: Set<CBv2RequestID>
    ) {
        for (id, _) in plan.assignments {
            guard var lease = leasesByID[id] else { continue }
            if !lease.isAdmitted {
                lease.markAdmitted(now: now)
                leasesByID[id] = lease
            } else if waitingBefore.contains(id) {
                lease.markReadmitted(now: now)
                leasesByID[id] = lease
                // Timing: a waiting→running crossing AFTER the first launch
                // stamp is a re-admission (preemption / capacity requeue).
                // The first admission itself is stamped at launch from
                // `wallStartedNanos`. Cost model: this `record(for:)` runs
                // only on a waiting→running crossing (an event), never per
                // row per steady-state step.
                if let rec = scheduler.record(for: id), rec.timing.admittedNanos != 0 {
                    rec.timing.readmissions &+= 1
                }
            }
        }
    }

    /// Central lease-expiry scan at the top of each step. Each expired request
    /// finishes with its TYPED cause (or, under the kill-switch, the legacy
    /// error string). Progress leases apply only to actively-RUNNING rows; a
    /// preempted row awaiting re-admission is bounded only by the absolute
    /// safety ceiling, never faulted as a stall.
    private func processLeaseExpiry(now: ContinuousClock.Instant) {
        var expired: [(CBv2RequestID, CBv2TerminalCause)] = []
        for rec in scheduler.running {
            if let cause = leasesByID[rec.id]?.expiredCause(
                now: now, isRunning: true, isPaused: rec.isPaused)
            {
                expired.append((rec.id, cause))
            }
        }
        for rec in scheduler.waiting {
            if let cause = leasesByID[rec.id]?.expiredCause(
                now: now, isRunning: false, isPaused: rec.isPaused)
            {
                expired.append((rec.id, cause))
            }
        }
        for (id, cause) in expired {
            finishRequest(
                id, reason: leaseFinishReason(for: cause), nowNanos: boundaryFinishNanos(),
                now: now)
        }
    }

    /// The wire finish reason for a lease cause. The legacy kill-switch keeps
    /// the exact original `.error` string for a behavior-compatible rollback
    /// (same single total-lifetime wall and wire terminal; timing now runs on
    /// the monotonic clock rather than `Date`, and expiry is observed at the
    /// central lease scan rather than the old inline checks — not literally
    /// bit-identical timing); every new-lease cause surfaces as a typed
    /// `.terminal`.
    private func leaseFinishReason(for cause: CBv2TerminalCause) -> CBv2FinishReason {
        switch cause {
        case .legacyRequestTimeout:
            return .error("request exceeded \(Int(config.requestTimeout))s deadline")
        default:
            return .terminal(cause: cause, message: cause.diagnostic)
        }
    }

    private func handlePreemptions(_ ids: [CBv2RequestID], now: ContinuousClock.Instant) {
        // Preemption only happens on the non-chained path, where the
        // previous step was finalized first — no in-flight step can
        // reference these states, so the release is immediate. The
        // scheduler already released the capacity reservations and requeued
        // the victims (generated tokens kept).
        assert(inFlight == nil, "preemption with a step in flight")
        // `now` is the step's boundary read (`stepNow`): no second clock
        // read for the preemption instant.
        for id in ids {
            preemptionCount += 1
            // A preempted request recomputes from scratch — any adopted
            // prefix credit no longer describes work that was skipped, so
            // usage.prefixCacheHitTokens must not over-credit at finish.
            invalidateAdoptedPrefix(id)
            resetResidentPrefixPublication(id)
            // A preempted row's drafter carry no longer describes its KV
            // (the structural fingerprint would catch it; drop eagerly).
            mtp?.invalidateCarry(id)
            // The scheduler rewound numComputedTokens to zero; reset the
            // lease's computed watermark and grant a fresh prefill window so
            // the re-prefill's confirmed chunks refresh the progress lease
            // (otherwise a long re-prefill is falsely killed as a stall).
            if var lease = leasesByID[id] {
                lease.markPreempted(now: now)
                leasesByID[id] = lease
            }
            if let state = kvStates.removeValue(forKey: id) {
                backend.release(state)
            }
            releaseRecurrentState(recurrentStates.removeValue(forKey: id))
        }
    }

    /// Create per-layer KV state on first execution. On `capacityExhausted`
    /// (several same-step admissions racing for the last bytes/pages — the
    /// backend's charge is atomic, so losers surface here) the request is
    /// requeued to waiting instead of error-finished: accepted requests wait
    /// for room. Bounded by `maxCapacityRequeues` (then error-finish) and by
    /// the request deadline. Other failures error-finish as before.
    func ensureKVState(_ rec: CBv2ScheduledRequest) -> [CBv2SequenceKV?]? {
        if let state = kvStates[rec.id] { return state }
        do {
            let maxLength = rec.request.promptTokens.count + max(rec.request.maxTokens, 1)
            let state = try backend.makeSequenceState(
                layerKinds: layerKinds, promptLength: rec.tokens.count, maxLength: maxLength)
            do {
                if let recurrent = try makeRecurrentRequestState() {
                    recurrentStates[rec.id] = recurrent
                }
            } catch {
                backend.release(state)
                if let kvError = error as? CBv2KVError { throw kvError }
                throw CBv2KVError.backendIneligible(
                    reason: "recurrent state allocation failed: \(error)")
            }
            kvStates[rec.id] = state
            capacityRequeues.removeValue(forKey: rec.id)
            // Inside a launch the step's `wallStartedNanos` is reused; the
            // fallback read covers direct (test) callers only.
            rec.stampKVAllocated(
                nowNanos: launchClockNanos != 0
                    ? launchClockNanos : DispatchTime.now().uptimeNanoseconds)
            return state
        } catch let kvError as CBv2KVError {
            if case .capacityExhausted = kvError {
                let attempts = capacityRequeues[rec.id, default: 0]
                if attempts < Self.maxCapacityRequeues, scheduler.requeueOnCapacity(rec.id) {
                    capacityRequeues[rec.id] = attempts + 1
                    capacityRequeueCount += 1
                    rec.timing.capacityRequeues &+= 1
                    mtp?.invalidateCarry(rec.id)  // preempted-style restart
                    // Preempted-style lease reset too: requeueOnCapacity
                    // rewound numComputedTokens to zero, so rewind the
                    // progress watermark and grant a fresh prefill window
                    // (admission stays permanently cleared). Without this a
                    // capacity wait longer than the prefill lease leaves the
                    // stale progress deadline expired and the newly
                    // re-admitted row is killed as .prefillStall before its
                    // first healthy chunk finalizes (PR#82 review). No
                    // pending finalize can include this row's sample here —
                    // it was being admitted this step, not running — so the
                    // rollback-path deferral does not apply.
                    if var lease = leasesByID[rec.id] {
                        lease.markPreempted(now: config.clock.now())
                        leasesByID[rec.id] = lease
                    }
                    return nil
                }
                // Terminal capacity exhaustion is retryable (the backend is
                // full, not broken): finish with the canonical prefix so
                // bridges surface a capacity error, never a server error.
                finishRequest(
                    rec.id,
                    reason: .error(
                        CBv2KVError.capacityExhaustedFinishPrefix + "\(kvError)"))
                return nil
            }
            finishRequest(rec.id, reason: .error("KV allocation failed: \(kvError)"))
            return nil
        } catch {
            finishRequest(rec.id, reason: .error("KV allocation failed: \(error)"))
            return nil
        }
    }

    private func makeRecurrentRequestState() throws -> CBv2RecurrentRequestState? {
        guard let spec = (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec else { return nil }
        let recurrent = try CBv2RecurrentRequestState(spec: spec)
        let existingRecurrentBytes = recurrentStates.values.reduce(0) { total, live in
            Self.saturatingAdd(total, live.byteCount)
        }
        let combined = Self.saturatingAdd(backend.bytesReserved,
            Self.saturatingAdd(existingRecurrentBytes, recurrent.byteCount))
        guard combined <= backend.bytesCapacity else {
            throw CBv2KVError.capacityExhausted(needed: recurrent.byteCount,
                available: max(0, backend.bytesCapacity - backend.bytesReserved - existingRecurrentBytes))
        }
        return recurrent
    }

    private func releaseRecurrentState(_ state: CBv2RecurrentRequestState?) {
        guard let state else { return }
        do { try state.release() } catch {
            preconditionFailure("CBv2 recurrent release failed: \(error)")
        }
    }

    // MARK: Streams

    func stream(for id: CBv2RequestID) -> CBv2OutputStream? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return streams[id]
    }

    private func takeStream(_ id: CBv2RequestID) -> CBv2OutputStream? {
        stateLock.lock()
        streamGenerations.removeValue(forKey: id)
        let stream = streams.removeValue(forKey: id)
        stateLock.unlock()
        stream?.releaseEngineOwnership()
        return stream
    }

    /// Remove and acknowledge exactly the stream generation whose backend
    /// ownership has retired. Identity matching prevents stale deferred work
    /// from removing a later generation if invariants are violated elsewhere.
    private func retireStream(
        _ id: CBv2RequestID,
        matching expected: CBv2OutputStream
    ) {
        stateLock.lock()
        let retired: CBv2OutputStream?
        if streams[id] === expected {
            streamGenerations.removeValue(forKey: id)
            retired = streams.removeValue(forKey: id)
        } else {
            retired = nil
        }
        stateLock.unlock()
        retired?.releaseEngineOwnership()
    }

    // MARK: Reconciled-usage snapshots (watchdog-readable)

    /// Publish the reconciled usage for `id` under `stateLock` so the watchdog
    /// thread can read it while the engine thread is wedged.
    private func setUsageSnapshot(_ id: CBv2RequestID, _ usage: CBv2Usage) {
        stateLock.lock()
        usageSnapshots[id] = usage
        stateLock.unlock()
    }

    private func clearUsageSnapshot(_ id: CBv2RequestID) {
        stateLock.lock()
        usageSnapshots.removeValue(forKey: id)
        stateLock.unlock()
    }

    // MARK: Test/telemetry introspection

    /// Engine-queue-synchronized snapshot of paused (backpressured) rows.
    /// Blocks until the current step boundary; test/telemetry use only.
    func pausedIDsSnapshot() -> Set<CBv2RequestID> {
        engineQueue.sync { Set(scheduler.running.filter(\.isPaused).map(\.id)) }
    }

    // MARK: Gauges

    func publishGauges() {
        // kvBytesCapacity carries the ADMISSION ceiling so a runtime
        // re-slice reads back consistently between the resize point-update
        // and per-step publishes (on the paged backend the two ledgers
        // diverge — the pool is construction-fixed). An INSTALLED ledger is
        // authoritative even at 0 (a legitimate zero re-slice must not be
        // overwritten with pool truth and re-advertise capacity that
        // admission rejects); backend truth is the fallback only for
        // ledger-less (bare-loop test) constructions. The installed admission
        // ledger includes target KV, recurrent fixed residency, assistant KV,
        // and external carve-outs. The backend can hold a larger target-only
        // promise (for example contiguous prompt + initial growth slack), so
        // reconcile that target promise with the ledger's non-backend portion
        // and take the larger complete obligation.
        let recurrentBytes = recurrentStates.values.reduce(0) { total, state in
            let (sum, overflow) = total.addingReportingOverflow(state.materializedByteCount)
            return overflow ? Int.max : sum
        }
        let detachedAssistantStates =
            (inFlight?.mtpRound?.verify?.rows.compactMap(\.assistantState) ?? [])
            + (inFlight?.mtpRound?.committedObservationRows.map(\.assistantState) ?? [])
            + (inFlight?.mtpRound?.deferredAssistantReleases ?? [])
        let assistantBytes = mtp?.materializedAssistantBytes(
            detachedStates: detachedAssistantStates) ?? 0
        let reservedBytes: Int
        if let admission = capacity as? AdmissionV2 {
            if (backend as? PagedKVBackend)?.pool.physicalLease != nil {
                // Segmented rows prepay their full logical promise. The ledger
                // already reconciles it atomically with physical slack, and
                // retains detached owners through asynchronous retirement.
                reservedBytes = Self.saturatingAdd(
                    admission.bytesReserved, admission.bytesExternallyReserved)
            } else {
                let target = admission.targetBytesReserved(
                    partitionedBy: Set(kvStates.keys))
                let materializedTarget = max(backend.bytesReserved, target.materialized)
                reservedBytes = Self.saturatingAdd(
                    Self.saturatingAdd(materializedTarget, target.unmaterialized),
                    admission.nonBackendBytesReserved)
            }
        } else {
            reservedBytes = Self.saturatingAdd(backend.bytesReserved, recurrentBytes)
        }
        gauges.update(
            CBv2CapacitySnapshot(
                activeRequests: scheduler.runningCount,
                waitingRequests: scheduler.waitingCount,
                kvBytesInUse: Self.saturatingAdd(
                    Self.saturatingAdd(backend.bytesInUse, recurrentBytes), assistantBytes),
                kvBytesCapacity: capacity?.bytesCapacity ?? backend.bytesCapacity,
                kvBytesBackendCapacity: backend.bytesCapacity,
                kvBytesReserved: reservedBytes,
                activeTokens: scheduler.activeTokens,
                stepsExecuted: stepCount,
                stepWallNanosTotal: stepWallNanosTotal,
                decodeRowsTotal: decodeRowsTotal,
                pagedStorage: (backend as? PagedKVBackend)?.pool.segmentStorageSnapshot))
    }

    static func saturatingAdd(_ lhs: Int, _ rhs: Int) -> Int {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? Int.max : value
    }

    /// Cumulative counters (`stepWallNanosTotal`, `decodeRowsTotal`) must
    /// never decrease: pin at `.max` instead of wrapping.
    static func saturatingAdd(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? UInt64.max : value
    }

    static func positionOffset(_ state: [CBv2SequenceKV?]) -> Int {
        guard let row = state.compactMap({ $0 }).first else {
            preconditionFailure("CBv2 position state has no owning attention row")
        }
        return row.absoluteOffset
    }

    // MARK: Watchdog (engine health signal)

    private var watchdogTimer: DispatchSourceTimer?

    private func markStepStarted() {
        stateLock.lock()
        stepStartedNanos = DispatchTime.now().uptimeNanoseconds
        stateLock.unlock()
    }

    private func markStepEnded() {
        stateLock.lock()
        stepStartedNanos = 0
        wedgeReported = false
        _healthy = true
        stateLock.unlock()
    }

    deinit {
        watchdogTimer?.cancel()
    }

    private func startWatchdog() {
        let timer = DispatchSource.makeTimerSource(queue: watchdogQueue)
        timer.schedule(
            deadline: .now() + config.watchdogInterval, repeating: config.watchdogInterval)
        timer.setEventHandler { [weak self] in self?.watchdogTick() }
        timer.resume()
        watchdogTimer = timer
    }

    private func stopWatchdog() {
        watchdogTimer?.cancel()
        watchdogTimer = nil
    }

    /// Runs on the watchdog queue. The engine thread may be blocked inside a
    /// wedged eval, so this touches ONLY lock-protected state and the
    /// (thread-safe) streams: consumers get a timely error, every live
    /// request is marked for cancellation (cleaned up if the loop ever
    /// resumes), and the health signal flips for provider heartbeats.
    private func watchdogTick() {
        stateLock.lock()
        let started = stepStartedNanos
        guard started != 0 else {
            stateLock.unlock()
            return
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds &- started) / 1_000_000_000
        guard elapsed > config.stepTimeout, !wedgeReported else {
            stateLock.unlock()
            return
        }
        wedgeReported = true
        _healthy = false
        let liveStreams = streams
        let deadlineOperations = Array(deadlineAdmissionOperations.values)
        for operation in deadlineOperations where operation.phase != .completed {
            operation.phase = .failedExternally
        }
        // Snapshot the reconciled usage observed before the wedge under the
        // SAME lock, so each terminal carries the tokens generated so far
        // instead of raw zero usage (the engine thread is wedged and cannot
        // reconcile now).
        let usageByID = usageSnapshots
        for id in liveStreams.keys {
            if let generation = streamGenerations[id] {
                pendingCancels[id] = generation
            }
        }
        stateLock.unlock()

        // Signal BEFORE erroring the streams: a consumer woken by the
        // terminal event must be able to observe the wedge side effects
        // (health metric, telemetry) immediately.
        onStepWedge?(elapsed)
        let message = "engine step exceeded \(Int(config.stepTimeout))s watchdog"
        for (id, stream) in liveStreams {
            stream.finish(
                reason: .terminal(cause: .watchdog, message: message),
                usage: usageByID[id]
                    ?? CBv2Usage(promptTokens: 0, completionTokens: 0))
        }
        for operation in deadlineOperations {
            operation.waiter.resume(returning: .capacityRejected)
        }
    }
}
