// Copyright © 2026 Eigen Labs.
//
// ContinuousBatchingV2 — WS-B: the public engine (`CBv2Engine`).
//
// Thin wiring layer: scheduler + admission + loop + sampler + model adapter.
// Tokenization happens on the caller's task (requests already carry token
// ids per the contract); the engine thread only ever does graph-build +
// asyncEval. Cancellation drops the row at the next step boundary, O(1).

import CryptoKit
import Foundation
import MLX
import os

private let log = Logger(subsystem: "darkbloom", category: "CBv2Engine")

/// Breaks the stream-initializer cycle while preserving the exact stream
/// generation in its abandonment callback. The identifier is installed
/// before the stream escapes `registerSubmission`.
private final class CBv2StreamGenerationBox: @unchecked Sendable {
    var generation: UInt64?
}

// MARK: - Shared gauges (submit-side admission ⇄ engine-thread truth)

/// Lock-protected counters bridging the caller-thread `submit`/`capacity`
/// surface and the engine thread. The loop publishes a full snapshot after
/// every step; `pendingSubmits` covers requests accepted on the caller
/// thread but not yet picked up by the engine queue (so the `maxWaiting`
/// bound holds even while a step blocks the queue).
final class CBv2EngineGauges: @unchecked Sendable {
    private let lock = NSLock()
    private var snapshot: CBv2CapacitySnapshot
    private var pendingSubmits = 0

    init(kvBytesCapacity: Int, kvBytesBackendCapacity: Int = 0, kvBytesReserved: Int = 0,
         pagedStorage: PagedKVStorageSnapshot? = nil) {
        // Seed backend truth at construction: heartbeats read `capacity()`
        // on IDLE engines (zero steps published), and a paged slot must
        // report its pool ceiling from the first beat, not after the
        // first request.
        self.snapshot = CBv2CapacitySnapshot(
            activeRequests: 0, waitingRequests: 0, kvBytesInUse: 0,
            kvBytesCapacity: kvBytesCapacity,
            kvBytesBackendCapacity: kvBytesBackendCapacity,
            kvBytesReserved: kvBytesReserved, activeTokens: 0, pagedStorage: pagedStorage)
    }

    func update(_ newValue: CBv2CapacitySnapshot) {
        lock.lock()
        snapshot = newValue
        lock.unlock()
    }

    func read() -> CBv2CapacitySnapshot {
        lock.lock()
        defer { lock.unlock() }
        var value = snapshot
        value.waitingRequests += pendingSubmits
        return value
    }

    /// Fast-path `maxWaiting` bound; the scheduler's own check (on the
    /// engine thread) is authoritative.
    func beginSubmit(maxWaiting: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard snapshot.waitingRequests + pendingSubmits < maxWaiting else { return false }
        pendingSubmits += 1
        return true
    }

    func endSubmit() {
        lock.lock()
        pendingSubmits = max(0, pendingSubmits - 1)
        lock.unlock()
    }

    /// Point update for runtime KV-capacity changes: an idle engine
    /// publishes no step snapshots, so `capacity()` must reflect a
    /// re-sliced ceiling and the backend's post-resize truth —
    /// without waiting for the next step. Refreshing only the ledger here
    /// would leave `kvBytesBackendCapacity` stale-small after an idle
    /// grow-back, and min-binding consumers (provider heartbeats) would
    /// under-advertise exactly the slots that just freed capacity. The
    /// loop's next full-snapshot publish carries the same live values.
    /// Paged storage remains the complete last queue capture: changing
    /// even its grant alone would mix two observations under one sequence.
    func updateKVBytesCapacity(_ bytes: Int, backendCapacity: Int) {
        lock.lock()
        snapshot.kvBytesCapacity = bytes
        snapshot.kvBytesBackendCapacity = backendCapacity
        lock.unlock()
    }
}

// MARK: - EngineV2

/// Production continuous-batching engine v2 (WS-B).
///
/// ```swift
/// let engine = EngineV2(
///     model: adapter, layerKinds: kinds, backend: kvBackend,
///     cacheProvider: layerCacheFactory)
/// let events = try engine.submit(request)
/// for await event in events { ... }
/// ```
/// `@unchecked Sendable` justification (contract `CBv2Engine: Sendable`):
/// all mutable state is either lock-protected (`stateLock`,
/// `CBv2EngineGauges`) or confined to the engine's serial dispatch queue
/// inside `EngineLoopV2`; the only cross-thread surfaces are `submit`
/// (lock + queue hop), resident-prefix preflight (index lock, advisory only),
/// `cancel` (lock), `capacity()` (lock), and `shutdown()`
/// (queue-synchronized drain).
public final class EngineV2: CBv2Engine, @unchecked Sendable {
    private let loop: EngineLoopV2
    private let admission: AdmissionV2
    /// Retained for runtime KV re-slicing (`updateKVBytesCapacity`); all
    /// step-path access goes through the loop.
    private let backend: CBv2KVBackend
    let schedulerConfig: CBv2SchedulerConfig
    private let loopConfig: CBv2EngineLoopConfig
    private let gauges: CBv2EngineGauges
    let layerKinds: [CBv2LayerKind]
    private let requiredPositionAxisCount: Int?
    private let supportsPositionedForwarding: Bool
    private let samplerSupportsTokenConstraints: Bool
    /// Non-nil only when active (instance supplied AND
    /// `schedulerConfig.enablePrefixCache`). Lookup + prefix slicing run on
    /// the submit thread (hashing is host work — never on the engine step
    /// thread); adoption and donation are handled by the loop.
    private let prefixCache: CBv2PrefixCache?
    /// Optional physical-page L1 discovered from the paged backend. Hash
    /// preparation is submit-thread safe; index lookup/claim stays on the
    /// serial engine queue.
    private let residentPrefixBackend: (any CBv2PagedPrefixSharingBackend)?
    public let prefixReuseCapability: CBv2PrefixReuseCapability
    public let modelCapabilities: CBv2ModelCapabilities
    public let hybridPrefixCache: CBv2HybridPrefixCache?
    public let completePrefixCache: (any CBv2CompletePrefixCache)?
    let completeCheckpointCodec: CBv2CompleteCheckpointCodec?
    let completeCheckpointCapture: CBv2CompleteCheckpointCapture?
    /// Exact fixed per-request charge after recurrent/MTP capability
    /// resolution. Shared/global budget bridges must use this value rather
    /// than recomputing the base recurrent peak from the model spec.
    public let resolvedFixedBytesPerRequest: Int

    private let stateLock = NSLock()
    private var rejectingSubmissions = false

    /// Engine health (step watchdog): false while a step exceeds the
    /// configured step timeout. Providers surface this in heartbeats.
    public var isHealthy: Bool { loop.isHealthy }
    /// Fired (once per wedge, from the watchdog thread) when a step exceeds
    /// the step timeout.
    public var onStepWedge: (@Sendable (TimeInterval) -> Void)? {
        get { loop.onStepWedge }
        set { loop.onStepWedge = newValue }
    }
    /// Telemetry/test hooks.
    public var stepCount: Int { loop.stepCount }
    public var chainedStepCount: Int { loop.chainedStepCount }
    public var preemptionCount: Int { loop.preemptionCount }
    /// Reset only at an idle benchmark boundary after warmup. Ordinary
    /// engines never allocate a recorder; snapshots add no device evaluation.
    @_spi(Benchmarking)
    public func beginForwardShapeObservation() throws -> CBv2ForwardShapeSnapshot {
        try loop.beginForwardShapeObservation()
    }

    @_spi(Benchmarking)
    public func forwardShapeSnapshot() -> CBv2ForwardShapeSnapshot {
        loop.forwardShapeSnapshot()
    }
    /// Steps that evaluated the eager caches' offset/KV inner state (DAR-325
    /// guard). Test hook; engine-thread owned, read at quiescent points.
    public var offsetChainEvalSteps: Int { loop.offsetChainEvalSteps }

    /// Packed-prefill capability AND cumulative execution evidence. The
    /// counters move only where `executeMixed` issues a rectangular
    /// `[B > 1, chunk]` forward, so a model that claims packing and never
    /// packs reads `isSupported == true, didExecute == false`. Plain counter
    /// reads (no lock, no step-path cost); poll from a benchmark harness or
    /// heartbeat, ideally at a quiescent point.
    public func packedPrefillActivity() -> CBv2PackedPrefillActivity {
        loop.packedPrefillActivity()
    }

    /// Cumulative MTP (speculative decoding) counters, or nil when MTP is
    /// inactive (no drafter, config/kill-switch off, or a model that cannot
    /// drive rounds). Lock-protected snapshot — safe to poll from any
    /// thread (provider heartbeats/telemetry).
    public func mtpMetricsSnapshot() -> CBv2MTPMetrics? { loop.mtp?.metricsSnapshot() }
    /// Construction-time reason MTP is inactive. nil means the driver is
    /// active. Providers can surface this alongside the nil metrics snapshot.
    public let mtpInactiveReason: String?
    /// Internal test hook (engine-queue synchronized).
    var loopForTesting: EngineLoopV2 { loop }
    /// Internal test hook: the admission ledger, for asserting byte
    /// reservation accounting at construction and across resizes.
    var admissionForTesting: AdmissionV2 { admission }

    public init(
        model: CBv2SteppableModel,
        layerKinds: [CBv2LayerKind],
        backend: CBv2KVBackend,
        cacheProvider: CBv2LayerCacheProvider,
        sampler: CBv2StepSampler = CBv2DefaultSampler(),
        detokenizerFactory: CBv2DetokenizerFactory = CBv2NullDetokenizerFactory(),
        schedulerConfig: CBv2SchedulerConfig = CBv2SchedulerConfig(),
        loopConfig: CBv2EngineLoopConfig = CBv2EngineLoopConfig(),
        admissionConfig: AdmissionV2.Config = AdmissionV2.Config(),
        prefixCache: CBv2PrefixCache? = nil,
        hybridPrefixCache: CBv2HybridPrefixCacheConfig? = nil,
        completePrefixCache: (any CBv2CompletePrefixCache)? = nil,
        mtpDrafter: (any CBv2MTPDrafter)? = nil,
        mtpConfig: CBv2MTPConfig = CBv2MTPConfig(),
        processMemoryOwner: (any CBv2ProcessMemoryOwner)? = nil
    ) {
        self.schedulerConfig = schedulerConfig
        self.loopConfig = loopConfig
        self.layerKinds = layerKinds
        self.requiredPositionAxisCount =
            (model as? any CBv2PositionAxisProviding)?.cbv2PositionAxisCount
        self.supportsPositionedForwarding = Self.modelSupportsPositionedForwarding(model)
        self.backend = backend
        self.samplerSupportsTokenConstraints = sampler.supportsTokenConstraints
        let modelCapabilities =
            (model as? any CBv2ModelCapabilityProviding)?.cbv2Capabilities ?? .attentionOnly
        self.modelCapabilities = modelCapabilities
        if let violation = Self.backendCapabilityViolation(
            capabilities: modelCapabilities, backend: backend)
        {
            preconditionFailure(violation)
        }
        if let violation = Self.keepMaskViolation(
            model: model, cacheProvider: cacheProvider)
        {
            preconditionFailure(violation)
        }
        let prefixReuseCapability = CBv2PrefixReuseCapability.derive(
            layerKinds: layerKinds,
            backend: backend.prefixReuseBackend,
            modelSupportsPrefixReuse: modelCapabilities.supportsPrefixReuse)
        self.prefixReuseCapability = prefixReuseCapability
        let activePrefixCache =
            schedulerConfig.enablePrefixCache && prefixReuseCapability.isSupported
            ? prefixCache : nil
        self.prefixCache = activePrefixCache
        let residentCandidate = backend as? any CBv2PagedPrefixSharingBackend
        self.residentPrefixBackend =
            schedulerConfig.enablePrefixCache && prefixReuseCapability.isSupported
                && residentCandidate?.residentPrefixCacheStats != nil
            ? residentCandidate : nil
        if let violation = Self.prefixCachePairingViolation(
            backend: backend, prefixCache: activePrefixCache)
        {
            preconditionFailure(violation)
        }
        // The ledger's token→byte conversion asks the BACKEND what a row
        // occupies (`CBv2KVResidencyPolicy`): contiguous windowed rows own
        // their whole fixed ring from the first write, paged rows own only
        // the pages `PagedKVPool.pageDemand` reserves.
        // MTP verification bypasses the sampler and emits raw target
        // argmaxes. Only the two known argmax-equivalent implementations may
        // activate it; custom samplers fail safe to ordinary target decode.
        let samplerSupportsMTP =
            sampler is CBv2DefaultSampler || sampler is CBv2GreedySampler
        let mtpDriver: CBv2MTPRoundDriver?
        if samplerSupportsMTP && modelCapabilities.supportsMTP {
            mtpDriver = CBv2MTPRoundDriver.build(
                model: model, drafter: mtpDrafter, config: mtpConfig,
                supportsRectangularCacheBank:
                    cacheProvider.supportsMTPRectangularVerification)
        } else {
            mtpDriver = nil
        }

        var admissionConfig = admissionConfig
        if let paged = backend as? PagedKVBackend {
            // Native storage and the token ledger consume the same resolved
            // table, including mixed BF16/FP32 layers after RoPE promotion.
            admissionConfig.layerElementBytes = paged.pool.layerDTypes.map(\.size)
        }
        // Capture allocator policy once, outside Admission. Failed projection
        // makes capacity unavailable instead of admitting logical-only bytes.
        let pricesAllocatorFootprints = (backend as? PagedKVBackend)?.pool.segmentGrant != nil
        let allocationPolicy = pricesAllocatorFootprints ? Memory.allocationFootprintPolicy() : nil
        if pricesAllocatorFootprints && allocationPolicy == nil {
            admissionConfig.fixedBytesPerRequest = Int.max
        } else if let recurrent = (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec {
            let perGeneration: Int
            if let allocationPolicy {
                perGeneration = (try? recurrent.allocationBytesPerGeneration(policy: allocationPolicy)) ?? Int.max
            } else {
                perGeneration = (try? recurrent.fixedBytesPerRequest()) ?? Int.max
            }
            let usesCompactRectangularReplay =
                modelCapabilities.supportsCompactRecurrentMTPReplay
                && (mtpDriver.map { $0.config.verificationMode != .serialTarget } ?? false)
            let recurrentDepth = mtpDriver?.config.fixedDraftTokens ?? mtpDriver?.config.maxDraftTokens ?? 0
            let compactContinuationHeadroom = usesCompactRectangularReplay && recurrentDepth >= 2
            let capturedContinuationHeadroom = !usesCompactRectangularReplay && recurrentDepth > 0
                && mtpDriver?.config.verificationMode != .serialTarget
                && (model as? any CBv2RecurrentMTPSteppableModel)?.supportsCapturedVerifyWindow == true
            let extraGenerations: Int
            if compactContinuationHeadroom {
                // One strict-prefix tape can survive into its successor.
                extraGenerations = 1
            } else if mtpDriver?.usesRequestStatefulDrafter == true && !usesCompactRectangularReplay {
                // The base three generations cover seed plus depth one.
                extraGenerations = max(0, recurrentDepth - 1)
            } else {
                extraGenerations = 0
            }
            let generations: Int
            let countOverflow: Bool
            if capturedContinuationHeadroom {
                // A committed prefix is a view retaining the previous whole
                // window. Its successor builds another whole captured stack.
                let (width, widthOverflow) = recurrentDepth.addingReportingOverflow(1)
                let (count, overlapOverflow) = width.multipliedReportingOverflow(by: 2)
                generations = count
                countOverflow = widthOverflow || overlapOverflow
            } else {
                (generations, countOverflow) = CBv2RecurrentStateSpec.maximumLiveGenerations
                    .addingReportingOverflow(extraGenerations)
            }
            let (bytes, byteOverflow) = perGeneration.multipliedReportingOverflow(by: generations)
            admissionConfig.fixedBytesPerRequest = countOverflow || byteOverflow ? Int.max : bytes
        }
        let fixedTargetOnly =
            mtpDriver?.config.fixedDraftTokens == 0
            || mtpDriver?.config.maxDraftTokens == 0
        admissionConfig.auxiliaryBytesPerToken =
            fixedTargetOnly ? 0 : (mtpDriver?.drafter.requestStateBytesPerToken ?? 0)
        admissionConfig.auxiliaryTokenGranularity =
            fixedTargetOnly ? 1 : (mtpDriver?.drafter.requestStateTokenGranularity ?? 1)
        admissionConfig.auxiliaryTokenAllocationPadding =
            fixedTargetOnly ? 0 : (mtpDriver?.drafter.requestStateTokenAllocationPadding ?? 0)
        if let allocationPolicy, admissionConfig.auxiliaryBytesPerToken > 0 {
            // Unknown stateful drafters get the conservative one-byte fragmented
            // envelope until they declare their independent allocation geometry.
            let specs = mtpDriver?.drafter.requestStateAllocationSpecs ?? [
                CBv2AuxiliaryAllocationSpec(
                    bytesPerToken: 1, allocationCount: admissionConfig.auxiliaryBytesPerToken,
                    tokenGranularity: admissionConfig.auxiliaryTokenGranularity,
                    tokenPadding: admissionConfig.auxiliaryTokenAllocationPadding, partitioned: true)
            ]
            if let projection = CBv2AuxiliaryAllocationProjection(policy: allocationPolicy, buffers: specs),
                !specs.isEmpty
            {
                admissionConfig.auxiliaryAllocationProjection = projection
            } else {
                admissionConfig.fixedBytesPerRequest = Int.max
            }
        }
        self.resolvedFixedBytesPerRequest = admissionConfig.fixedBytesPerRequest
        let admission = AdmissionV2(
            layerKinds: layerKinds, bytesCapacity: backend.bytesCapacity,
            config: admissionConfig, residency: backend.kvResidency,
            processMemoryOwner: processMemoryOwner)
        self.admission = admission
        let segmentedPool = (backend as? PagedKVBackend)?.pool
        if segmentedPool?.segmentGrant != nil {
            segmentedPool?.bindAdmission(admission)
        }
        let gauges = CBv2EngineGauges(
            kvBytesCapacity: backend.bytesCapacity,
            kvBytesBackendCapacity: backend.bytesCapacity,
            kvBytesReserved: segmentedPool?.segmentGrant != nil ? admission.bytesReserved : 0,
            pagedStorage: (backend as? PagedKVBackend)?.pool.segmentStorageSnapshot)
        self.gauges = gauges
        let mtpInactiveReason: String?
        if mtpDriver != nil {
            mtpInactiveReason = nil
        } else if mtpDrafter == nil {
            mtpInactiveReason = "no drafter supplied"
        } else if !mtpConfig.enabled {
            mtpInactiveReason = "configuration disabled"
        } else if !CBv2MTPConfig.envEnabled {
            mtpInactiveReason = "DARKBLOOM_CBV2_MTP kill switch"
        } else if !samplerSupportsMTP {
            mtpInactiveReason =
                "sampler \(type(of: sampler)) is not proven argmax-equivalent"
        } else {
            mtpInactiveReason =
                "model/drafter pair cannot prove matching MTP target identity or capture layers"
        }
        self.mtpInactiveReason = mtpInactiveReason
        if mtpDrafter != nil, mtpConfig.enabled, let mtpInactiveReason {
            log.info(
                "CBv2 MTP inactive despite a bound drafter: \(mtpInactiveReason, privacy: .public) — plain decode"
            )
        }
        // Mixed-step prefill quota (opt-in, default OFF ⇒ byte-identical
        // planning). When set, a step that also carries decode work admits
        // at most this many PROMPT tokens, so newly arriving prompts cannot
        // park an actively decoding request behind ~1.6s of prefill in one
        // asyncEval/readback boundary. Pure-prefill steps stay uncapped.
        let scheduler = SchedulerV2(config: schedulerConfig, capacity: admission)
        scheduler.reserveFullSequenceTokens = segmentedPool?.segmentGrant != nil
        if let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_MIXED_PREFILL_CAP"],
            let cap = Int(raw), cap >= 0
        {
            scheduler.mixedStepPrefillTokenCap = cap
            log.info(
                "CBv2 mixed-step prefill cap: \(cap, privacy: .public) tokens")
        }
        let fullRecurrentCheckpointEligible = modelCapabilities.supportsRecurrentCheckpointReuse
            && (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec != nil
            && !layerKinds.isEmpty
            && layerKinds.allSatisfy { kind in
                if case .full = kind.attention { return kind.sharesKVWithLayer == nil }
                return false
            }
            && (!(mtpDriver?.tracksPersistentHistory ?? false)
                || mtpDriver?.drafter is any CBv2MTPPrefixCheckpointDrafter)
            && schedulerConfig.enablePrefixCache
        let hybridEligible = fullRecurrentCheckpointEligible
            && backend.prefixReuseBackend == .contiguousUnquantized
            && !backend.requiresMaterializedSnapshots
        let pagedCheckpointConfig = segmentedPool.flatMap { pool -> PagedKVPoolConfig? in
            guard pool.segmentGrant != nil, pool.config.layerDTypes != nil else { return nil }
            return pool.config
        }
        self.hybridPrefixCache = hybridEligible
            ? hybridPrefixCache.flatMap { $0.isValid ? CBv2HybridPrefixCache(config: $0) : nil }
            : nil
        if fullRecurrentCheckpointEligible, hybridEligible || pagedCheckpointConfig != nil,
            let completePrefixCache,
            let recurrentSpec = (model as? any CBv2RecurrentSteppableModel)?.recurrentStateSpec,
            let kvDTypes = (model as? any CBv2CompleteCheckpointKVTypeProviding)?.cbv2CompleteCheckpointKVDTypes,
            kvDTypes.count == layerKinds.count,
            pagedCheckpointConfig == nil || kvDTypes == segmentedPool?.layerDTypes,
            !(mtpDriver?.tracksPersistentHistory ?? false)
                || mtpDriver?.drafter is any CBv2MTPPrefixCheckpointCoding
        {
            let codec = CBv2CompleteCheckpointCodec(
                identity: completePrefixCache.identity, layerKinds: layerKinds,
                recurrentSpec: recurrentSpec, kvDTypes: kvDTypes,
                assistant: mtpDriver?.tracksPersistentHistory == true
                    ? mtpDriver?.drafter as? any CBv2MTPPrefixCheckpointCoding : nil,
                admission: admission, pagedConfig: pagedCheckpointConfig)
            self.completePrefixCache = completePrefixCache
            self.completeCheckpointCodec = codec
            self.completeCheckpointCapture = .init(codec: codec, store: completePrefixCache)
        } else if schedulerConfig.enablePrefixCache,
            (model as? any CBv2HistoricalAttentionCheckpointProviding)?.cbv2SupportsHistoricalAttentionCheckpoint == true,
            !(mtpDriver?.tracksPersistentHistory ?? false),
            let completePrefixCache, let pagedCheckpointConfig, let segmentedPool,
            (try? CBv2HistoricalAttentionLayout(layerKinds: layerKinds, dtypes: segmentedPool.layerDTypes)) != nil
        {
            // The caller explicitly supplied a durable store. Observed native
            // storage dtypes, including borrower rows, define this loaded codec.
            let codec = CBv2CompleteCheckpointCodec(
                identity: completePrefixCache.identity, layerKinds: layerKinds,
                recurrentSpec: nil, kvDTypes: segmentedPool.layerDTypes,
                assistant: nil, admission: admission, pagedConfig: pagedCheckpointConfig)
            self.completePrefixCache = completePrefixCache
            self.completeCheckpointCodec = codec
            self.completeCheckpointCapture = .init(codec: codec, store: completePrefixCache)
        } else {
            self.completePrefixCache = nil
            self.completeCheckpointCodec = nil
            self.completeCheckpointCapture = nil
        }
        self.loop = EngineLoopV2(
            model: model,
            layerKinds: layerKinds,
            backend: backend,
            cacheProvider: cacheProvider,
            sampler: sampler,
            detokenizerFactory: detokenizerFactory,
            scheduler: scheduler,
            capacity: admission,
            prefixCache: activePrefixCache,
            residentPrefixBackend: residentPrefixBackend,
            hybridPrefixCache: self.hybridPrefixCache,
            completeCheckpointCapture: self.completeCheckpointCapture,
            prefixReuseCapability: prefixReuseCapability,
            nominalFullKVBytesPerToken: admission.fullKVBytesPerToken,
            mtp: mtpDriver,
            config: loopConfig,
            gauges: gauges)
        loop.start()
    }

    /// Structural safety check for the (backend, prefix-cache) pairing,
    /// enforced by `init` as a precondition: a backend whose donated
    /// snapshot views reference RECYCLABLE storage
    /// (`CBv2KVBackend.requiresMaterializedSnapshots`, e.g. the paged
    /// slabs) must never feed a `PrefixCacheV2` configured with
    /// `materializeOnDonate: false` — its entries would silently decay
    /// into other requests' bytes once the donor's pages are recycled.
    /// Returns the violation description, or nil when the pairing is safe.
    /// Custom `CBv2PrefixCache` implementations cannot be inspected
    /// structurally; they own this obligation themselves (see the
    /// `requiresMaterializedSnapshots` contract doc). Internal so tests
    /// can assert the rule without tripping the precondition.
    static func prefixCachePairingViolation(
        backend: CBv2KVBackend, prefixCache: CBv2PrefixCache?
    ) -> String? {
        guard backend.requiresMaterializedSnapshots,
            let concrete = prefixCache as? PrefixCacheV2,
            !concrete.config.materializeOnDonate
        else { return nil }
        return """
            EngineV2: \(type(of: backend)) donates snapshot views over recyclable \
            storage (requiresMaterializedSnapshots), but the prefix cache is \
            configured with materializeOnDonate: false — cached entries would \
            reference recycled pages. Use materializeOnDonate: true (the default).
            """
    }

    /// Production factories check this before the initializer's invariant.
    /// Recurrent models opt into paging only after their state transactions
    /// are proven, and can require the observed native segmented layout.
    public static func backendCapabilityViolation(
        capabilities: CBv2ModelCapabilities, backend: CBv2KVBackend
    ) -> String? {
        guard backend.prefixReuseBackend == .pagedFP16 else { return nil }
        guard capabilities.supportsPagedKV else {
            return "EngineV2: model capability vetoes paged KV for request-owned recurrent state"
        }
        if capabilities.requiresNativePagedKV {
            guard let paged = backend as? PagedKVBackend,
                paged.pool.config.segmentSizeBytes != nil,
                paged.pool.config.layerDTypes != nil
            else {
                return "EngineV2: model requires segmented paged KV with observed native layer dtypes"
            }
        }
        return nil
    }

    /// Construction-time keep-mask pairing. A model whose attention is sparse
    /// by construction (Qwen 3.8 Flash-Next QSA) declares
    /// `cbv2RequiresKeepMask`; serving it on a provider that drops the mask
    /// would answer with a DIFFERENT model, so the engine refuses to build.
    public static func keepMaskViolation(
        model: any CBv2SteppableModel, cacheProvider: CBv2LayerCacheProvider
    ) -> String? {
        guard (model as? any CBv2KeepMaskRequiringModel)?.cbv2RequiresKeepMask == true,
            !cacheProvider.supportsKeepMask
        else { return nil }
        return
            "EngineV2: \(type(of: model)) requires the attention keep mask, but "
            + "\(type(of: cacheProvider)) does not support it (supportsKeepMask == false)"
    }

    // MARK: Submission preparation

    private func normalizedRequest(_ request: CBv2Request) -> CBv2Request {
        var request = request
        if request.positionState == nil {
            request.positionState = request.multimodal?.positionState
        }
        return request
    }

    private func requireAcceptingSubmissions() throws {
        let rejecting = stateLock.withLock { rejectingSubmissions }
        guard !rejecting else {
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }
    }

    /// A positioned multimodal request exercises two seams: embedding
    /// prefill and token decode. Structural conformance must prove both, and
    /// runtime-capability adapters must additionally confirm that their
    /// wrapped model implements both. This makes the execution guards
    /// invariant assertions rather than request-dependent fatal paths.
    private static func modelSupportsPositionedForwarding(
        _ model: CBv2SteppableModel
    ) -> Bool {
        guard
            (model as? any CBv2PositionedForwardingCapabilityProviding)?
                .supportsPositionedForwarding == true
        else { return false }

        if let recurrent = model as? any CBv2RecurrentSteppableModel,
            recurrent.recurrentStateSpec != nil
        {
            return model is any CBv2PositionedRecurrentSteppableModel
                && model is any CBv2PositionedMultimodalSteppableModel
        }
        return model is any CBv2PositionedSteppableModel
            && model is any CBv2PositionedEmbeddingSteppableModel
    }

    private func validateRequest(_ request: CBv2Request) throws {
        if let positions = request.positionState {
            guard positions.promptLength == request.promptTokens.count else {
                throw CBv2MultimodalError.invalidSpans(
                    "position length \(positions.promptLength) != prompt length "
                        + "\(request.promptTokens.count)")
            }
            guard request.multimodal != nil else {
                throw CBv2MultimodalError.invalidSpans(
                    "explicit model positions require multimodal input")
            }
            if let requiredAxes = requiredPositionAxisCount,
                positions.axisCount != requiredAxes
            {
                throw CBv2MultimodalError.invalidSpans(
                    "position axes \(positions.axisCount) != model axes \(requiredAxes)")
            }
            guard supportsPositionedForwarding else {
                throw CBv2MultimodalError.unsupportedModel(
                    "request-owned positions require positioned model forwarding")
            }
        }
        if request.multimodal?.attention == .causal, request.positionState == nil {
            throw CBv2MultimodalError.invalidSpans(
                "causal multimodal input requires request-owned position state")
        }
        if let constraint = request.tokenConstraint {
            guard samplerSupportsTokenConstraints else {
                throw CBv2SchedulerError.tokenConstraintUnsupportedBySampler
            }
            guard constraint.maxTokens == request.maxTokens else {
                throw CBv2SchedulerError.tokenConstraintBudgetMismatch(
                    request: request.maxTokens,
                    constraint: constraint.maxTokens)
            }
        }
    }

    private func validatedMultimodalBlocks(
        for request: CBv2Request
    ) throws -> [CBv2ImageSpan] {
        guard let input = request.multimodal else { return [] }
        return try CBv2MultimodalPlan.validate(
            input,
            promptTokenCount: request.promptTokens.count,
            model: loop.model,
            cacheProvider: loop.cacheProvider,
            maxBatchedTokensPerStep: schedulerConfig.maxBatchedTokensPerStep)
    }

    private func requireRequestCanEverFit(_ request: CBv2Request) throws {
        let pool = (backend as? PagedKVBackend)?.pool
        let completion = pool?.segmentGrant != nil ? max(request.maxTokens, 1) : max(request.maxTokens, 0)
        let (tokens, overflow) = request.promptTokens.count.addingReportingOverflow(completion)
        let overhead: Int
        if let pool, pool.segmentGrant != nil {
            overhead = overflow ? Int.max : (pool.minimumSegmentedOverhead(
                tokens: tokens, layerKinds: layerKinds) ?? Int.max)
        } else { overhead = 0 }
        guard !overflow, admission.canEverFit(
            promptTokens: request.promptTokens.count,
            maxTokens: completion,
            additionalBackendBytes: overhead)
        else {
            let allocated = overflow ? Int.max : admission.allocatedBytes(forTokens: tokens)
            let (needed, sumOverflow) = allocated.addingReportingOverflow(overhead)
            throw CBv2KVError.capacityExhausted(
                needed: sumOverflow ? Int.max : needed,
                available: admission.admissibleBytesCapacity)
        }
    }

    private func registerSubmission(_ request: CBv2Request) throws -> CBv2OutputStream {
        guard gauges.beginSubmit(maxWaiting: schedulerConfig.maxWaiting) else {
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }
        let generation = CBv2StreamGenerationBox()
        let stream = CBv2OutputStream(
            id: request.id,
            capacity: loopConfig.eventBufferCapacity,
            onBackpressure: { [loop] id, paused in loop.setPaused(id, paused) },
            onAbandoned: { [loop, generation] id in
                guard let token = generation.generation else { return }
                loop.requestCancel(id, streamGeneration: token)
            })
        guard let streamGeneration = loop.registerWithGeneration(stream: stream) else {
            gauges.endSubmit()
            throw CBv2SchedulerError.duplicateRequestID(request.id)
        }
        generation.generation = streamGeneration
        return stream
    }

    // MARK: CBv2Engine

    /// Submit a request; events stream until `.finished`. Throws
    /// `CBv2KVError.capacityExhausted` when truthful admission fails (worst
    /// case could never fit), the waiting queue is full, or the engine is
    /// shutting down — the provider maps this to 429/503 exactly as today.
    /// Throws `CBv2SchedulerError.duplicateRequestID` when the id is still
    /// live: the duplicate is rejected BEFORE any stream registration, so
    /// the original request's stream is never touched (PR#62 review).
    public func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> {
        let request = normalizedRequest(request)
        try requireAcceptingSubmissions()
        // Degenerate requests: uniform event surface, no engine round-trip.
        if request.maxTokens <= 0 {
            return Self.immediateStream(
                reason: .length,
                usage: CBv2Usage(promptTokens: request.promptTokens.count, completionTokens: 0))
        }
        if request.promptTokens.isEmpty {
            return Self.immediateStream(
                reason: .error("empty prompt"),
                usage: CBv2Usage(promptTokens: 0, completionTokens: 0))
        }
        try validateRequest(request)

        // Vision requests, CHEAP half only: span structure, model/backend
        // capability, block-vs-budget — no provider call, no MLX graphs.
        // The heavy embedding materialization waits until EVERY cheap
        // admission gate (canEverFit, maxWaiting, duplicate id) has passed,
        // so a rejected request never runs the vision tower and heavy work
        // is always counted in `pendingSubmits` (PR#63 review). All
        // multimodal failures remain submit-time throws
        // (`CBv2MultimodalError`); nothing reaches the scheduler.
        let multimodalBlocks = try validatedMultimodalBlocks(for: request)

        // Truthful admission: reject what could NEVER fit; everything else
        // is admitted optimistically (preemption is the backstop).
        try requireRequestCanEverFit(request)
        // Registration doubles as the duplicate-id gate: it refuses to
        // replace a live stream, so the FIRST request keeps delivering and
        // the duplicate fails here — before the scheduler ever sees it.
        // (The scheduler's own `duplicateRequestID` rejection remains the
        // engine-thread backstop.)
        let stream = try registerSubmission(request)

        // Vision requests, HEAVY half: materialize the image embeddings ONCE,
        // here on the submit thread (the one provider call per request —
        // never on the engine step thread), now that every cheap gate has
        // passed. A failure must unwind the registration + pending-submit
        // count taken above (the stream was never handed to the caller).
        var multimodal: CBv2ResolvedMultimodal? = nil
        if let input = request.multimodal {
            do {
                multimodal = try CBv2MultimodalPlan.materialize(
                    input, blocks: multimodalBlocks, model: loop.model)
            } catch {
                loop.unregister(request.id)
                gauges.endSubmit()
                throw error
            }
        }

        loop.enqueue(
            request,
            prefixLookup: makePrefixLookup(for: request),
            residentPrefixProbe: makeResidentPrefixProbe(for: request),
            multimodal: multimodal)
        return stream.makeStream()
    }

    /// Submit only when the authoritative engine-queue projection can reach
    /// this request's first token by the absolute monotonic deadline.
    ///
    /// The caller owns the anchored deadline and conservative phase rates; the
    /// verdict subtracts elapsed time only inside the serialized engine-queue
    /// closure, so waiting for that closure never resets the budget.
    /// Multimodal requests fail closed before image materialization because
    /// image-tower work is not representable by scheduled token count.
    public func submit(
        _ request: CBv2Request,
        firstTokenDeadline: CBv2FirstTokenDeadlineAdmission
    ) async throws -> CBv2FirstTokenDeadlineResult {
        let request = normalizedRequest(request)
        try requireAcceptingSubmissions()

        let zeroWork = CBv2FirstTokenProjectedWork.bounded(
            work: CBv2FirstTokenScheduledWork(
                prefillTokens: 0,
                decodeTokens: 0,
                scheduledSteps: 0,
                mixedSteps: 0),
            serviceDuration: .zero)
        if request.maxTokens <= 0 {
            return .admitted(
                stream: Self.immediateStream(
                    reason: .length,
                    usage: CBv2Usage(
                        promptTokens: request.promptTokens.count,
                        completionTokens: 0)),
                projectedWork: zeroWork,
                admittedAt: .now,
                retirement: .acknowledged)
        }
        if request.promptTokens.isEmpty {
            return .admitted(
                stream: Self.immediateStream(
                    reason: .error("empty prompt"),
                    usage: CBv2Usage(promptTokens: 0, completionTokens: 0)),
                projectedWork: zeroWork,
                admittedAt: .now,
                retirement: .acknowledged)
        }
        try validateRequest(request)

        if request.multimodal != nil {
            _ = try validatedMultimodalBlocks(for: request)
            return .deadlineUnreachable(projectedWork: .unbounded)
        }
        guard loop.isHealthy else {
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }

        try requireRequestCanEverFit(request)
        let stream = try registerSubmission(request)

        let outcome = await loop.enqueueForFirstTokenDeadline(
            request,
            prefixLookup: makePrefixLookup(for: request),
            residentPrefixProbe: makeResidentPrefixProbe(for: request),
            admission: firstTokenDeadline)
        return try await withTaskCancellationHandler {
            switch outcome {
            case .admitted(let projectedWork, let admittedAt):
                let retirement = CBv2RequestRetirement(stream: stream)
                if Task.isCancelled {
                    loop.requestCancel(request.id)
                    throw CBv2FirstTokenAdmissionCancellation(
                        stream: stream.makeStream(),
                        retirement: retirement)
                }
                return .admitted(
                    stream: stream.makeStream(),
                    projectedWork: projectedWork,
                    admittedAt: admittedAt,
                    retirement: retirement)
            case .deadlineUnreachable(let projectedWork):
                return .deadlineUnreachable(projectedWork: projectedWork)
            case .cancelled:
                throw CancellationError()
            case .schedulerRejected(let error):
                throw error
            case .capacityRejected:
                throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
            }
        } onCancel: {
            // Never block a Swift cancellation handler: a permanently wedged
            // engine queue may never retire the row. The operation returns a
            // typed ownership-transfer error if commit already won.
            loop.requestCancel(request.id)
        }
    }

    /// Prefix-cache lookup on the SUBMIT thread (SHA-256 hashing is host
    /// work; slicing is graph-only). The capability derives the exact
    /// match-specific M/C/R plan. Safe layouts slice full rows to C; hybrid
    /// contiguous native-float layouts retain full rows through M for frozen replay.
    /// The lookup pin travels with the adoption; the loop balances it in
    /// every outcome.
    private func makePrefixLookup(for request: CBv2Request) -> CBv2PrefixLookup {
        // Submit-thread duration of hashing + lookup + plan derivation
        // (`CBv2RequestTiming.prefixLookupNanos`); rides the lookup into
        // `enqueue` so no engine-queue clock read is needed.
        guard prefixCache != nil || hybridPrefixCache != nil || completePrefixCache != nil else {
            return resolvePrefixLookup(for: request)
        }
        let started = DispatchTime.now().uptimeNanoseconds
        var lookup = resolvePrefixLookup(for: request)
        lookup.lookupNanos = max(1, DispatchTime.now().uptimeNanoseconds &- started)
        return lookup
    }

    private func resolvePrefixLookup(for request: CBv2Request) -> CBv2PrefixLookup {
        if completePrefixCache != nil {
            return completeCheckpointLookup(for: request)
        }
        if let hybridPrefixCache {
            return hybridPrefixLookup(for: request, cache: hybridPrefixCache)
        }
        guard prefixCache != nil || residentPrefixBackend != nil else {
            return CBv2PrefixLookup(adoption: nil, outcome: .disabled, matchedTokens: 0)
        }
        guard request.prefixCacheEnabled else {
            return CBv2PrefixLookup(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0)
        }
        guard cbv2LayerKindsAllowPrefixReuse(layerKinds) else {
            return CBv2PrefixLookup(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0)
        }
        // Vision requests NEVER look up the prefix cache (v1 policy — the
        // donation side is excluded symmetrically in
        // `EngineLoopV2.donationIntent`): token-id hashes cannot see image
        // content, so an adopted "hit" over another request's image span
        // would be silently wrong KV.
        guard request.multimodal == nil else {
            return CBv2PrefixLookup(adoption: nil, outcome: .skippedPolicy, matchedTokens: 0)
        }
        guard let prefixCache else {
            // The engine-queue resident lookup has not run yet.
            return CBv2PrefixLookup(adoption: nil, outcome: .miss, matchedTokens: 0)
        }
        let cacheRequestID = request.prefixCacheReceiptID ?? request.id
        guard
            let hit = prefixCache.lookup(
                requestID: cacheRequestID, tokens: request.promptTokens, layerKinds: layerKinds,
                cacheSalt: request.cacheSalt)
        else {
            return CBv2PrefixLookup(adoption: nil, outcome: .miss, matchedTokens: 0)
        }
        var exactStagedFullKVBytes = 0
        var byteOverflow = false
        for entry in hit.prefix {
            guard let entry else { continue }
            let (entryBytes, entryOverflow) = entry.keys.nbytes.addingReportingOverflow(
                entry.values.nbytes)
            let (sum, sumOverflow) = exactStagedFullKVBytes.addingReportingOverflow(entryBytes)
            if entryOverflow || sumOverflow {
                byteOverflow = true
                break
            }
            exactStagedFullKVBytes = sum
        }
        // The fixed sliding rings are deliberately NOT added on top of the
        // plan's capacity adjustment. Since Bug A (paged-KV plan §7 item
        // 0.1) the ledger charges every windowed layer its WHOLE ring
        // inside the token reservation itself
        // (`AdmissionV2.allocatedBytes(forTokens:)`), and `applyAdoption`
        // reserves `plan.capacityReservationTokens` — always > 0 for a
        // non-nil plan — before it touches the backend. Routing the
        // shortfall through `initialAdditionalCapacityBytes` as well would
        // charge the rings twice and starve adoption on Gemma-style
        // hybrids. Accounting overflow still fails cold: `reserve` throws
        // `capacityExhausted` and adoption falls back to a full prefill.
        guard !byteOverflow,
            let plan = prefixReuseCapability.plan(
                matchedBoundary: hit.matched,
                exactStagedFullKVBytes: exactStagedFullKVBytes,
                maximumSequenceLength:
                    request.promptTokens.count + max(request.maxTokens, 1),
                nominalFullKVBytesPerToken: admission.fullKVBytesPerToken,
                reserveFullSequenceTokens: (backend as? PagedKVBackend)?.pool.segmentGrant != nil)
        else {
            prefixCache.endAdoption(
                requestID: cacheRequestID, tokens: request.promptTokens, matched: hit.matched,
                cacheSalt: request.cacheSalt)
            return CBv2PrefixLookup(
                adoption: nil, outcome: .skippedPolicy, matchedTokens: hit.matched)
        }
        let prefix = hit.prefix.map { entry -> (keys: MLXArray, values: MLXArray, offset: Int)? in
            guard let entry else { return nil }
            let restored = plan.restoredFullTokens
            guard entry.offset >= restored else { return nil }
            guard entry.offset != restored else { return entry }
            return (
                keys: entry.keys[.ellipsis, 0 ..< restored, 0...],
                values: entry.values[.ellipsis, 0 ..< restored, 0...],
                offset: restored
            )
        }
        // Every storage-owning full row must survive the graph-only slice.
        for (index, kind) in layerKinds.enumerated()
        where kind.sharesKVWithLayer == nil {
            if case .full = kind.attention, prefix[index] == nil {
                prefixCache.endAdoption(
                    requestID: cacheRequestID,
                    tokens: request.promptTokens,
                    matched: hit.matched,
                    cacheSalt: request.cacheSalt)
                return CBv2PrefixLookup(
                    adoption: nil,
                    outcome: .adoptionFailed,
                    matchedTokens: hit.matched)
            }
        }
        return CBv2PrefixLookup(
            adoption: CBv2PrefixAdoption(
                requestID: cacheRequestID, tokens: request.promptTokens, matched: hit.matched,
                plan: plan, prefix: prefix, cacheSalt: request.cacheSalt),
            outcome: .adoptionFailed, matchedTokens: hit.matched)
    }

    /// Pure submit-thread hash preparation for the resident paged L1. The
    /// mutable index and physical page refs are intentionally untouched here.
    private func makeResidentPrefixProbe(for request: CBv2Request) -> CBv2PagedPrefixProbe? {
        guard let residentPrefixBackend,
            request.prefixCacheEnabled,
            cbv2LayerKindsAllowPrefixReuse(layerKinds),
            request.multimodal == nil
        else { return nil }
        return residentPrefixBackend.preparePrefixProbe(
            tokens: request.promptTokens, cacheSalt: request.cacheSalt)
    }

    /// Advisory only: no page reference is taken here. A generation-checked
    /// claim still runs on the serial engine queue after submission, so a page
    /// recycled in between produces a cold fallback rather than stale KV.
    public func residentPrefixCandidate(
        for request: CBv2Request
    ) -> CBv2ResidentPrefixCandidate? {
        if let hybridPrefixCache, request.prefixCacheEnabled,
            request.multimodal == nil, request.positionState == nil
        {
            return hybridPrefixCache.candidate(
                tokens: request.promptTokens, cacheSalt: request.cacheSalt,
                maximumChunkSize: max(
                    schedulerConfig.prefillChunkSize, schedulerConfig.soloPrefillStripeTokens ?? 0))
        }
        guard let residentPrefixBackend,
            let probe = makeResidentPrefixProbe(for: request),
            let match = residentPrefixBackend.peekResidentPrefix(for: probe)
        else { return nil }
        let completionBudget = max(request.maxTokens, 1)
        let (maxLength, overflow) = request.promptTokens.count.addingReportingOverflow(
            completionBudget)
        guard !overflow,
            let plan = prefixReuseCapability.plan(
                matchedBoundary: match.matchedTokens,
                maximumSequenceLength: maxLength,
                nominalFullKVBytesPerToken: admission.fullKVBytesPerToken,
                reserveFullSequenceTokens: (backend as? PagedKVBackend)?.pool.segmentGrant != nil),
            plan.prefillTokensSaved > 0
        else { return nil }
        return CBv2ResidentPrefixCandidate(
            matchedTokens: match.matchedTokens,
            prefillTokensSaved: plan.prefillTokensSaved)
    }

    /// Cancel promptly: the in-flight step completes, the row is dropped
    /// O(1) at the next step boundary.
    public func cancel(_ id: CBv2RequestID) {
        loop.requestCancel(id)
    }

    /// Truthful capacity snapshot (actual bytes/tokens, not worst case),
    /// published by the engine thread after every step.
    public func capacity() -> CBv2CapacitySnapshot {
        gauges.read()
    }

    /// Runtime KV-budget update (multi-model co-residency re-slicing): fans
    /// out to the admission ledger and the KV backend. Safe from any thread
    /// — both ledgers are lock-protected, and a step racing the update sees
    /// either the old or the new ceiling: a shrink that lands mid-step is
    /// absorbed by the loop's capacity-requeue path (backend admissions are
    /// atomic; losers requeue to waiting). Ordering keeps the soft gate at
    /// or under the hard gate throughout: on shrink the admission ledger
    /// tightens FIRST, on grow the backend widens FIRST, so admission never
    /// admits work the backend would still refuse at the old ceiling.
    /// In-flight reservations above a new lower ceiling are untouched; new
    /// admissions fail (`capacityExhausted`) until the pool drains below
    /// the new ceiling; grow admits immediately.
    public func updateKVBytesCapacity(_ bytes: Int) {
        let total = max(0, bytes)
        let hybridReservation = hybridPrefixCache?.resizeReservation(
            to: min(hybridPrefixCache?.config.maximumBytes ?? 0, total / 4)) ?? 0
        let clamped = max(0, total - hybridReservation)
        if clamped < admission.bytesCapacity {
            admission.updateBytesCapacity(clamped)
            backend.updateBytesCapacity(clamped)
        } else {
            backend.updateBytesCapacity(clamped)
            admission.updateBytesCapacity(clamped)
        }
        // Read the backend after resize. Segmented and contiguous backends
        // report the new grant; the fixed reference retains its ceiling.
        gauges.updateKVBytesCapacity(clamped, backendCapacity: backend.bytesCapacity)
    }

    /// Graceful drain: new submissions are rejected, waiting requests are
    /// cancelled, running requests finish naturally, then the loop stops.
    /// Bounded by `CBv2EngineLoopConfig.shutdownTimeout` (default 10 s): if
    /// the engine queue is wedged, live streams are force-finished with
    /// `.error` and this returns instead of hanging forever.
    public func shutdown() async {
        beginRejectingSubmissions()
        completeCheckpointCapture?.close()
        completePrefixCache?.close()
        await loop.drain()
        hybridPrefixCache?.close()
        admission.closeProcessMemoryOwner()
    }

    /// Synchronous helper: `NSLock` is not async-safe to hold across
    /// suspension points, so the flag flip lives outside the async context.
    private func beginRejectingSubmissions() {
        stateLock.lock()
        rejectingSubmissions = true
        stateLock.unlock()
    }

    // MARK: Helpers

    private static func immediateStream(
        reason: CBv2FinishReason, usage: CBv2Usage
    ) -> AsyncStream<CBv2Event> {
        AsyncStream { continuation in
            continuation.yield(.finished(reason: reason, usage: usage))
            continuation.finish()
        }
    }
}

// MARK: - Teacher-forced top-1 scoring (backend parity measurement)

extension EngineV2 {
    /// Force `continuation` through THIS engine and return the argmax at
    /// each continuation position (`CBv2Engine.teacherForcedTop1`).
    ///
    /// Runs on the engine queue against a private KV row allocated from the
    /// same backend as any request: the prompt through the loop's own
    /// `prefillOutput` in the chunks `SchedulerV2` would cut, then one
    /// `[1, 1]` forward per forced token through the same eager caches the
    /// decode step uses. Backend, cache provider, chunking and masks are
    /// the engine's, which is the entire point — scoring outside the engine
    /// would measure the model, and the two backends share the model.
    ///
    /// Never enters the scheduler, the decode chain or the step watchdog,
    /// and never samples: no sampler is consulted, so `CBv2SamplingParams`,
    /// temperature, seeds and token constraints cannot reach it. Repeated
    /// calls with the same arguments return the same ids.
    ///
    /// Throws `CBv2TeacherForcingError.engineBusy` rather than scoring
    /// beside an in-flight step, `.engineNotRunning` after `shutdown()`,
    /// `.nothingToScore` for an empty prompt or continuation, and
    /// `CBv2KVError.capacityExhausted` when the pool cannot seat the row.
    public func teacherForcedTop1(promptTokens: [Int], continuation: [Int]) throws -> [Int] {
        try loop.teacherForcedTop1(promptTokens: promptTokens, continuation: continuation)
    }

    /// Bounded numerical observation of ordinary teacher-forced target logits.
    /// This does not execute or certify speculative verification.
    @_spi(Diagnostics)
    public func teacherForcedScores(_ request: CBv2TeacherForcedScoreRequest) throws
        -> CBv2TeacherForcedScoreSnapshot
    {
        try loop.teacherForcedScores(request)
    }

    /// Cumulative evidence that the scoring above drove real engine
    /// forwards. Counters move at the forwards themselves, so a harness can
    /// assert `decodeForwardsExecuted` grew by exactly
    /// `continuation.count - 1` across a call instead of trusting that the
    /// witness exists.
    public func teacherForcedScoringActivity() -> CBv2TeacherForcedScoringActivity {
        loop.teacherForcedScoringActivity()
    }
}

// MARK: - Prefill logit digest (backend parity measurement)

/// One-shot delivery box for the captured frontier logits. The capture runs
/// on the engine queue; the digest caller blocks on the semaphore. Only the
/// FIRST delivery is kept — a probe has exactly one prompt frontier, and a
/// second one would mean the row was re-prefilled after preemption, in
/// which case the first (uninterrupted) vector is the one that matches a
/// quiescent arm.
private final class CBv2FrontierLogitBox: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var captured: MLXArray?

    func deliver(_ logits: MLXArray) {
        lock.lock()
        let isFirst = captured == nil
        if isFirst { captured = logits }
        lock.unlock()
        if isFirst { semaphore.signal() }
    }

    /// nil means the frontier never fired inside the window — the caller
    /// MUST throw rather than digest anything.
    func wait(seconds: TimeInterval) -> MLXArray? {
        guard semaphore.wait(timeout: .now() + seconds) == .success else { return nil }
        lock.lock()
        defer { lock.unlock() }
        return captured
    }
}

extension EngineV2 {
    /// Ceiling on how long a probe prefill may take before the digest call
    /// refuses. Generous: this is a once-per-arm measurement over a whole
    /// prompt on a cold engine, not a step-path budget.
    private static let digestProbeTimeout: TimeInterval = 120

    /// Serializes probes: the loop holds ONE capture hook, so two concurrent
    /// digests would fight over it.
    private static let digestProbeLock = NSLock()
    private nonisolated(unsafe) static var digestProbeCounter: UInt64 = .max

    private static func nextDigestProbeID() -> CBv2RequestID {
        digestProbeCounter &-= 1
        return CBv2RequestID(digestProbeCounter)
    }

    /// SHA-256 of the final-position prefill logit vector this engine
    /// produces for `promptTokens`, taken on the engine's OWN prefill path.
    ///
    /// A probe request (`maxTokens: 1`, greedy, prefix cache off) is
    /// submitted through the ordinary `submit` surface, so the vector is
    /// computed by the real scheduler, the real chunking, the real layer
    /// caches and the real KV backend. The capture sits at the prompt
    /// frontier BEFORE the sampler, so nothing downstream of the logits can
    /// launder a backend difference away. There is no second prefill
    /// implementation here to drift from the first.
    ///
    /// The prefix cache is disabled FOR THE PROBE ONLY (both directions):
    /// an adopted prefix would make the digest a function of some other
    /// request's donated KV, and a donation would make later traffic a
    /// function of the measurement. Neither is a property of the arm under
    /// test.
    ///
    /// Blocking and synchronous by design — the harness calls it once per
    /// arm at a quiescent point. Never call it from the engine queue.
    ///
    /// Throws `CBv2PrefillLogitDigestError.emptyPrompt`,
    /// `.prefillProducedNoLogits` when no frontier fired inside the window
    /// (wedged step, cancelled row), `.unexpectedLogitShape` when the
    /// frontier is not a single vector, and whatever `submit` throws
    /// (`CBv2KVError.capacityExhausted` when the probe cannot be admitted,
    /// including after `shutdown()`). It never returns a fabricated digest.
    public func prefillLogitDigest(_ promptTokens: [Int]) throws -> CBv2PrefillLogitDigest {
        guard !promptTokens.isEmpty else {
            throw CBv2PrefillLogitDigestError.emptyPrompt
        }
        Self.digestProbeLock.lock()
        defer { Self.digestProbeLock.unlock() }

        let probeID = Self.nextDigestProbeID()
        let box = CBv2FrontierLogitBox()
        loop.setPrefillFrontierCapture { id, logits in
            guard id == probeID else { return }
            box.deliver(logits)
        }
        defer { loop.setPrefillFrontierCapture(nil) }

        // The stream must outlive the wait: dropping it trips
        // `CBv2OutputStream.onAbandoned`, which cancels the row — possibly
        // before it ever prefills.
        let stream = try submit(
            CBv2Request(
                id: probeID,
                promptTokens: promptTokens,
                sampling: CBv2SamplingParams(temperature: 0),
                maxTokens: 1,
                prefixCacheEnabled: false))
        let captured = box.wait(seconds: Self.digestProbeTimeout)
        cancel(probeID)
        withExtendedLifetime(stream) {}

        guard let captured else {
            throw CBv2PrefillLogitDigestError.prefillProducedNoLogits(
                seconds: Self.digestProbeTimeout)
        }
        // Evaluate on the engine queue: the captured handle is a lazy graph
        // node over the same slabs a live step may still be writing, and MLX
        // graph work is not safe to run beside the step thread.
        return try loop.onEngineQueueSync {
            try Self.digest(frontierLogits: captured)
        }
    }

    /// Metadata-only benchmark diagnostic. Requires idle, MTP-off, full-attention owners.
    @_spi(Diagnostics)
    public func configureAttentionMetadata(_ configuration: CBv2AttentionMetadataConfig?) throws {
        try loop.configureAttentionMetadata(configuration)
    }

    @_spi(Diagnostics)
    public func takeAttentionMetadataSnapshot() throws -> CBv2AttentionMetadataSnapshot? {
        try loop.takeAttentionMetadataSnapshot()
    }

    /// One owner's native attention bytes, retained only through its existing step fence.
    @_spi(Diagnostics)
    public func configureAttentionPacket(_ configuration: CBv2AttentionPacketConfig?) throws {
        try loop.configureAttentionPacket(configuration)
    }

    @_spi(Diagnostics)
    public func takeAttentionPacketSnapshot() throws -> CBv2AttentionPacketSnapshot? {
        try loop.takeAttentionPacketSnapshot()
    }

    /// Offline-only, bounded capture. Install/clear only while the engine is
    /// idle; nil leaves the serving path without diagnostic tensor work.
    @_spi(Diagnostics)
    public func configureLogitDiagnostic(_ configuration: CBv2LogitDiagnosticConfig?) throws {
        try loop.configureLogitDiagnostic(configuration)
    }

    /// Drain immutable host records at an idle boundary. This never evaluates
    /// tensors or refills the configured record budget.
    @_spi(Diagnostics)
    public func takeLogitDiagnosticSnapshot() throws -> CBv2LogitDiagnosticSnapshot? {
        try loop.takeLogitDiagnosticSnapshot()
    }

    /// Hash the RAW bytes of the frontier vector in the dtype the model
    /// produced. No `asType`: an upcast here would make a bf16 arm and an
    /// fp32 arm hash identically for values that differ below fp16
    /// precision, which is exactly the drift this seam has to see.
    static func digest(frontierLogits: MLXArray) throws -> CBv2PrefillLogitDigest {
        let vector: MLXArray
        switch frontierLogits.ndim {
        case 1: vector = frontierLogits
        case 2 where frontierLogits.dim(0) == 1: vector = frontierLogits[0]
        default: throw CBv2PrefillLogitDigestError.unexpectedLogitShape(frontierLogits.shape)
        }
        let maxAbs = MLX.abs(vector).max().item(Float.self)
        let raw = vector.asData(access: .copy)
        var hasher = SHA256()
        hasher.update(data: raw.data)
        let hex = hasher.finalize().reduce(into: "") { $0 += String(format: "%02x", $1) }
        return CBv2PrefillLogitDigest(
            dtype: Self.dtypeName(raw.dType),
            count: vector.dim(0),
            sha256: hex,
            maxAbs: maxAbs)
    }

    /// Stable, MLX-canonical dtype spelling. A `String(describing:)` on
    /// `DType` would be a Swift enum case name and would silently change
    /// shape if mlx-swift renames one; the harness compares these across
    /// arms and across builds.
    static func dtypeName(_ dtype: DType) -> String {
        switch dtype {
        case .bool: return "bool"
        case .uint8: return "uint8"
        case .uint16: return "uint16"
        case .uint32: return "uint32"
        case .uint64: return "uint64"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .float16: return "float16"
        case .float32: return "float32"
        case .bfloat16: return "bfloat16"
        case .complex64: return "complex64"
        case .float64: return "float64"
        }
    }
}
