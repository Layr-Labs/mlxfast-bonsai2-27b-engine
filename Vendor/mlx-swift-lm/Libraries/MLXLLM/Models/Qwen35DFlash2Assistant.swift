// Copyright © 2026 Eigen Labs.
//
// The CBv2 block-drafter adapter over `DFlash2DraftModel`, bound to a Qwen 3.5
// / Ternary Bonsai 2 target.
//
// `Qwen35InlineMTPAssistant` is the CHAIN drafter for this target: the engine
// asks it for one token at a time and threads the previous step's hidden state
// back in. DFlash 2 is a BLOCK drafter: one forward over `1 + depth` positions
// returns the whole block, and its context is the TARGET's hidden state at five
// named layers rather than anything the drafter itself produced. This file is
// the seam between those two facts and `CBv2MTPBlockDrafter`.
//
// WHERE THE CONTEXT COMES FROM. The engine hands the drafter every position the
// target has COMMITTED, through the two seams a request-stateful drafter
// already has:
//
//   * `observeCommittedTarget` — the prompt's positions, chunk by chunk, and
//     every plain decode position;
//   * `finalizeRound` — a verified round's CONFIRMED columns, which is the
//     reference's `hidden = hidden[:, :accepted + 1, :]`.
//
// Those rows queue here until the next `proposeBlock` feeds them to the block,
// which is where they enter the drafter's own cache as projected keys and
// values. Nothing else reaches the drafter: the block itself carries no state
// between rounds.

import Foundation
import MLX
import MLXLMCommon

public enum Qwen35DFlash2Error: LocalizedError, Sendable, Equatable {
    case invalidTarget(String)
    case targetLayerCountMismatch(drafter: Int, target: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidTarget(let type):
            return "The target \(type) is not a Qwen 3.5 text model."
        case .targetLayerCountMismatch(let drafter, let target):
            return
                "The DFlash 2 drafter was trained against \(drafter) target layers; "
                + "this target has \(target)."
        }
    }
}

/// A DFlash 2 drafter bound to one Qwen 3.5 target, as the engine sees it.
public final class Qwen35DFlash2Assistant: CBv2MTPBlockLeadingSubmission, @unchecked Sendable {

    public let drafter: DFlash2DraftModel
    private let target: Qwen35TextModel

    /// How many context rows the drafter can hold. A drafter whose layers all
    /// slide keeps `sliding_window - 1` of them, so older rows are dropped
    /// here rather than carried to a forward that would drop them anyway.
    private let contextRowLimit: Int?

    public var targetIdentity: ObjectIdentifier { ObjectIdentifier(target) }
    public var mtpTargetIdentity: ObjectIdentifier? { targetIdentity }

    /// One drafter forward per round, over a rectangular `[1, 1 + k]` target
    /// window. Serial per-column verification would add k target forwards to a
    /// round whose whole point is that there is only one.
    public var requiredVerificationMode: CBv2MTPVerificationMode? { .rectangular }

    /// The engine's block ceiling. The drafter was trained at `block_size`,
    /// and the reference lets a caller ask for a larger block: the extra
    /// positions are legal and simply accept less. Capping at the trained
    /// depth would draft 7 while the echo named the depth that was asked for.
    public var maximumDraftTokens: Int? { CBv2MTPConfig.testedMaxBlockDraftTokens }

    /// Single stream. The block drafter keeps one context cache per request
    /// and the track measures one stream, so a wider rectangle is untested
    /// rather than declared.
    public var maximumSpeculativeBatch: Int? { 1 }

    /// Greedy only. The reference's rejection-sampling path is not ported, so
    /// this drafter never lifts the engine's `temperature == 0` gate.
    public var supportsTargetPrefixAcceptance: Bool { false }

    private init(drafter: DFlash2DraftModel, target: Qwen35TextModel) {
        self.drafter = drafter
        self.target = target
        self.contextRowLimit = drafter.contextRowLimit
    }

    // MARK: - Loading

    /// True when `directory` holds a DFlash 2 drafter. Read from the config's
    /// own declaration, never from the directory's name.
    public static func isDrafterDirectory(_ directory: URL) -> Bool {
        DFlash2DraftModel.isDFlash2Directory(directory)
    }

    /// Load the drafter and bind it to the target's tower.
    ///
    /// `bind` checks the vocabulary and the hidden size; the target's layer
    /// count is checked here, because a drafter trained against a different
    /// tower would read five layers that exist but mean something else, and
    /// the only symptom would be a poor accept rate.
    public static func load(
        from directory: URL, target: any LanguageModel
    ) throws -> Qwen35DFlash2Assistant {
        let text = try qwen35TextTarget(target)
        let drafter = try DFlash2DraftModel.load(from: directory)
        guard drafter.config.numTargetLayers == text.configuration.hiddenLayers else {
            throw Qwen35DFlash2Error.targetLayerCountMismatch(
                drafter: drafter.config.numTargetLayers,
                target: text.configuration.hiddenLayers)
        }
        try drafter.bind(target: text)
        let assistant = Qwen35DFlash2Assistant(drafter: drafter, target: text)
        assistant.warmSpeculativeShapes(serving: target)
        return assistant
    }

    // MARK: - Load-time warm

    /// Kill switch for the load-time drafter warm (default on).
    static let speculativeWarmEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_SPEC_WARM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The block the scored rounds draft: one anchor plus the declared depth.
    static let warmBlockSize = 16

    /// Runs the drafter's round once per shape, at load, on throwaway state,
    /// so the first timed round does not pay its custom-kernel compiles and
    /// pipeline builds.
    ///
    /// The drafter proposes over zero context rows through its own fresh
    /// caches: a prompt-sized first context, then each small context a round
    /// can hand it. Nothing here touches a request's cache, the engine, the
    /// target's tap or any random state; every result is evaluated and
    /// dropped, and the buffer cache is drained afterwards, as the resident's
    /// own warm does, so the served phases start from the footprint a cold
    /// load leaves.
    func warmSpeculativeShapes(serving: (any LanguageModel)? = nil) {
        guard Self.speculativeWarmEnabled else { return }
        warmTargetPrefill()
        warmDrafter()
        if let serving {
            warmEngineRound(serving: serving)
            // Once more after the runner has adopted the model, at the
            // resident's boot warm: locally the load-time engine round left
            // part of the first timed round's cost in place in some processes,
            // and a round run after the full load removed it in every one.
            CBv2DeferredLoadWarm.register { [weak self] in
                guard let self else { return }
                self.warmEngineRound(serving: serving)
                Stream().synchronize()
                Memory.clearCache()
            }
        }
        Stream().synchronize()
        Memory.clearCache()
    }

    /// `MLXFAST_ENGINE_ROUND_WARM=0` skips `warmEngineRound`.
    static let engineRoundWarmEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_ENGINE_ROUND_WARM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Runs one short request through a real `EngineV2` at load: the engine's
    /// seed step and its first speculative round, on throwaway state.
    ///
    /// The model-level warms above run the same target and drafter calls, but
    /// not the engine's own round machinery (planning, the round graph, the
    /// early block, finalize, the round journal). Nothing before the timed
    /// window runs that machinery either: the timed prefill goes through the
    /// teacher-forced stepper, and the seed window ends at the seed token. So
    /// a fresh process paid its first-use costs inside the window's first
    /// round: on the local M4 Max that round's early drafter submission took
    /// 143-208 ms instead of 36-41 ms and the round 464-534 ms instead of
    /// ~358 ms, with +1.2-1.8 G instructions, ~150 page faults and ~+0.13 s
    /// of system time in the window, and none of it in any later window of
    /// the same process. One engine request of 16 tokens at load removes all
    /// of it; a model-level round does not.
    ///
    /// The engine is built the way the benchmark worker builds a DFlash leg
    /// (contiguous KV, one stream, the scored seed width in one chunk, fixed
    /// depth 15, rectangular verify), over the SERVING model this drafter is
    /// bound to, so the same code runs. The prompt is the same fixed token
    /// pattern `warmTargetPrefill` uses, greedy, so nothing depends on any
    /// request's input; the engine is shut down, the tap restored, and
    /// `warmSpeculativeShapes` drains the buffer cache afterwards.
    private func warmEngineRound(serving: any LanguageModel) {
        guard Self.engineRoundWarmEnabled else { return }
        let layerKinds: [CBv2LayerKind]
        let caches: [any CBv2AttendingLayerCache]
        do {
            let make: (Int, CBv2LayerKind) throws -> any CBv2AttendingLayerCache = {
                index, kind in CBv2LayerCache(layerIndex: index, kind: kind)
            }
            if let model = serving as? Qwen35Model {
                layerKinds = model.cbv2LayerKinds
                caches = try model.newCacheV2(makeLayerCache: make)
            } else if let model = serving as? Qwen35TextModel {
                layerKinds = model.cbv2LayerKinds
                caches = try model.newCacheV2(makeLayerCache: make)
            } else {
                return
            }
        } catch {
            return
        }
        let previousTap = target.dFlash2TapLayerIds
        guard (try? setBlockContextArmed(true)) != nil else { return }
        defer {
            target.dFlash2TapLayerIds = previousTap
            target.model.dFlash2Tap.tappedHidden = nil
        }
        let depth = Self.warmBlockSize - 1
        let rows = Self.warmPromptRows
        weak var released: EngineV2?
        do {
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(serving),
                layerKinds: layerKinds,
                backend: CBv2ContiguousKVBackend(
                    config: CBv2ContiguousBackendConfig(bytesCapacity: 1 << 30)),
                cacheProvider: CBv2LayerCacheBank(caches: caches),
                sampler: CBv2DefaultSampler(),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: 1,
                    prefillChunkSize: max(CBv2SchedulerConfig().prefillChunkSize, rows),
                    maxWaiting: 1,
                    enablePrefixCache: false),
                mtpDrafter: self,
                mtpConfig: CBv2MTPConfig(
                    enabled: true,
                    maxDraftTokens: depth,
                    maxSpeculativeBatch: 1,
                    fixedDraftTokens: depth,
                    verificationMode: .automatic,
                    maxAutomaticRectangularTokens: 1 + depth,
                    draftTokenCeiling: CBv2MTPConfig.testedMaxBlockDraftTokens))
            released = engine
            var request = CBv2Request(
                id: CBv2RequestID(1),
                promptTokens: (0 ..< rows).map { 100 + ($0 &* 7919) % 20_000 },
                maxTokens: 1 + Self.warmBlockSize)
            request.sampling = CBv2SamplingParams(temperature: 0, topP: 1, topK: 0)
            request.stopTokens = []
            let warmRequest = request
            let done = DispatchSemaphore(value: 0)
            Task.detached {
                if let events = try? engine.submit(warmRequest) {
                    for await _ in events {}
                }
                await engine.shutdown()
                done.signal()
            }
            done.wait()
        }
        // The engine is out of scope here; its last references go as the
        // task above unwinds and its queues drain. Wait (bounded) until it is
        // gone, so its arrays are back in the cache before
        // `warmSpeculativeShapes` drains it, and the served phases start from
        // the footprint a cold load leaves.
        var waited = 0
        while released != nil, waited < 500 {
            usleep(1_000)
            waited += 1
        }
    }

    /// `MLXFAST_SEED_PREFILL_WARM=0` skips `warmTargetPrefill`.
    static let seedPrefillWarmEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_SEED_PREFILL_WARM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The scored seed width.
    static let warmPromptRows = 512

    /// Runs the ENGINE's prompt forward once, at load, on throwaway state.
    ///
    /// A decode window's seed prefill is the engine's prompt seam
    /// (`forwardWithHiddenForPrefill`: the DFlash 2 tap armed, the final layer
    /// narrowed to the last row, the tapped context cast for the drafter).
    /// Nothing before the timed decode phase runs that seam: the resident's
    /// boot warm and the benchmarker's warm-up prefill both go through the
    /// teacher-forced stepper, whose forward is full width with the tap off.
    /// So every scored window paid the seam's first-use costs inside the
    /// timed seed window. On every published leg the seed window reads
    /// 30-42 ms slower than the timed prefill of the same 512 tokens, even
    /// though it does less work (and ~36 ms on the serial control leg too).
    ///
    /// The prompt is a fixed token pattern (the resident warm's), the caches
    /// and recurrent state are fresh and released here, the tap is restored,
    /// and `warmSpeculativeShapes` drains the buffer cache afterwards, so the
    /// served phases start from the footprint a cold load leaves. Nothing here
    /// depends on any request's input.
    private func warmTargetPrefill() {
        guard Self.seedPrefillWarmEnabled else { return }
        let rows = Self.warmPromptRows
        let adapter = CBv2SteppableLanguageModelAdapter(target)
        guard let spec = adapter.recurrentStateSpec else { return }
        let backend = CBv2ContiguousKVBackend(
            config: CBv2ContiguousBackendConfig(bytesCapacity: 1 << 30))
        guard
            let caches = try? target.newCacheV2(makeLayerCache: { index, kind in
                CBv2LayerCache(layerIndex: index, kind: kind)
            }),
            let rowState = try? backend.makeSequenceState(
                layerKinds: target.cbv2LayerKinds, promptLength: 0, maxLength: rows + 32),
            let recurrent = try? CBv2RecurrentRequestState(spec: spec)
        else { return }
        let bank = CBv2LayerCacheBank(caches: caches)
        let previousTap = target.dFlash2TapLayerIds
        target.dFlash2TapLayerIds = drafter.config.targetLayerIds
        defer {
            target.dFlash2TapLayerIds = previousTap
            target.model.dFlash2Tap.tappedHidden = nil
            bank.releaseBoundRows()
            backend.release(rowState)
            if !recurrent.isReleased { try? recurrent.release() }
        }
        guard let evaluation = try? recurrent.bind() else { return }
        let tokens = MLXArray((0 ..< rows).map { Int32(100 + ($0 &* 7919) % 20_000) })
            .reshaped([1, rows])
        let forward = adapter.forwardWithHiddenForPrefill(
            tokens: tokens, caches: bank.layerCaches(rowStates: [rowState]),
            recurrentState: [evaluation], positionIds: nil,
            requirement: .lastPositionLogits)
        guard let roots = try? evaluation.evaluate() else { return }
        var targets = [argMax(forward.logits, axis: -1), forward.lastHidden] + roots
        if let tapped = target.dFlash2TappedHidden {
            targets.append(tapped.asType(drafter.dtype))
        }
        eval(targets)
        try? evaluation.commit()
    }

    private func warmDrafter() {
        let block = Self.warmBlockSize
        guard let caches = try? drafter.makeCache() else { return }
        let width = drafter.config.targetHiddenSize
        var offset = 0
        for rows in [512, 513] + Array(1 ... block) {
            let context = MLXArray.zeros([1, rows, width], dtype: drafter.dtype)
            guard
                let tokens = try? drafter.propose(
                    anchor: [0], targetHidden: context, cache: caches, blockSize: block)
            else { return }
            eval([tokens] + caches.flatMap { $0.innerState() })
            offset += rows
            drafter.trimCache(caches, toCommittedLength: offset)
        }
        warmContextPrefetch()
    }

    /// The prefetch path's own shapes, on fresh throwaway caches: a prompt's
    /// context absorbed without a block, then a block over the cache alone.
    private func warmContextPrefetch() {
        guard Self.contextPrefetchEnabled, let caches = try? drafter.makeCache() else { return }
        let block = Self.warmBlockSize
        let context = MLXArray.zeros(
            [1, Self.warmPromptRows, drafter.config.targetHiddenSize], dtype: drafter.dtype)
        guard (try? drafter.absorbContext(targetHidden: context, cache: caches)) == true
        else { return }
        eval(caches.flatMap { $0.innerState() })
        guard
            let tokens = try? drafter.propose(
                anchor: [0], targetHidden: nil, cache: caches, blockSize: block)
        else { return }
        eval([tokens] + caches.flatMap { $0.innerState() })
    }

    private static func qwen35TextTarget(
        _ target: any LanguageModel
    ) throws -> Qwen35TextModel {
        if let target = target as? Qwen35TextModel { return target }
        if let target = target as? Qwen35Model { return target.languageModel }
        throw Qwen35DFlash2Error.invalidTarget(String(describing: type(of: target)))
    }

    // MARK: - The context tap

    /// The tap is armed by the engine build that speculates with this drafter
    /// and disarmed by one that does not, so a serial leg in a process that
    /// also served a DFlash leg pays nothing for a resident drafter.
    public func setBlockContextArmed(_ armed: Bool) throws {
        if armed {
            try target.armDFlash2Tap(layerIds: drafter.config.targetLayerIds)
        } else {
            target.dFlash2TapLayerIds = nil
        }
    }

    public func blockContextHidden() -> MLXArray? {
        target.dFlash2TappedHidden
    }

    // MARK: - Request state

    /// One request's drafter cache and the committed context rows it has not
    /// absorbed yet.
    final class RequestState: CBv2MTPRequestState {
        var caches: [any KVCache]
        /// Committed context rows the next block will consume, oldest first.
        var pending: [MLXArray] = []
        var pendingRows = 0
        /// Absolute target position after the newest observed row; equal to
        /// the number of committed positions this state has ever seen.
        var observedRows = 0
        /// The caches start at the first pending row's position, which is not
        /// zero when the prompt was longer than the context window.
        var cacheSeeded = false
        /// True when every committed row was absorbed into the caches ahead
        /// of the next block (`prefetchCommittedContext`).
        var contextPrefetched = false
        /// Lazy proposals retained until the engine's finalize fence.
        var roots: [MLXArray] = []
        var isReleased = false

        init(caches: [any KVCache]) { self.caches = caches }

        /// Absolute position of the first pending row.
        var firstPendingPosition: Int { observedRows - pendingRows }

        /// Every committed position this drafter knows about. Nothing here is
        /// ever speculative: a block absorbs only rows the target committed.
        var committedInputCount: Int { observedRows }
        var stagedInputCount: Int { 0 }

        var materializedBytes: Int {
            let arrays = caches.flatMap { $0.innerState() } + pending + roots
            return arrays.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }

        func append(_ rows: MLXArray, limit: Int?) {
            let count = rows.dim(1)
            guard count > 0 else { return }
            pending.append(rows)
            pendingRows += count
            observedRows += count
            guard let limit, pendingRows > limit else { return }
            // Drop whole leading chunks, then the head of the next one. The
            // caches have not seen these rows, so dropping them only moves
            // where the block believes it sits.
            while let first = pending.first, pendingRows - first.dim(1) >= limit {
                pending.removeFirst()
                pendingRows -= first.dim(1)
            }
            if pendingRows > limit, let first = pending.first {
                let drop = pendingRows - limit
                pending[0] = first[0..., drop..., 0...]
                pendingRows -= drop
            }
        }

        func absorbPending() {
            pending.removeAll(keepingCapacity: true)
            pendingRows = 0
        }

        func clearAll() {
            caches.removeAll(keepingCapacity: false)
            pending.removeAll(keepingCapacity: false)
            roots.removeAll(keepingCapacity: false)
            pendingRows = 0
            contextPrefetched = false
            isReleased = true
        }
    }

    private func state(_ value: any CBv2MTPRequestState) -> RequestState {
        guard let state = value as? RequestState else {
            preconditionFailure("DFlash 2 request state of the wrong type")
        }
        return state
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        do {
            return RequestState(caches: try drafter.makeCache())
        } catch {
            preconditionFailure("DFlash 2 drafter cache construction failed: \(error)")
        }
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        state(requestState).clearAll()
    }

    // MARK: - Committed history

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        append(observation.hidden, to: requestState)
    }

    /// THE DTYPE CROSSING. The Bonsai trunk promotes activations to FP32 after
    /// its FP32 norms; the drafter is BF16. The cast happens HERE rather than
    /// at the drafter's `fc` input: it is the same one cast per context row,
    /// and it additionally DETACHES the row from the whole prefill chunk it
    /// was sliced out of, so a 2047-row window retains 2047 rows and not the
    /// prompt. `DFlash2DraftModel.hiddenStates` casts again and that stays a
    /// no-op for an already-BF16 tensor.
    private func append(_ hidden: MLXArray, to requestState: any CBv2MTPRequestState) {
        let state = self.state(requestState)
        guard hidden.dim(1) > 0 else { return }
        state.append(hidden.asType(drafter.dtype), limit: contextRowLimit)
    }

    // MARK: - Proposing

    public func proposeBlock(
        anchor: Int, depth: Int, requestState: any CBv2MTPRequestState
    ) throws -> MLXArray {
        try proposeBlock(
            anchor: anchor, depth: depth, requestState: requestState,
            submittingLeadingLayers: 0)
    }

    public func proposeBlock(
        anchor: Int, depth: Int, requestState: any CBv2MTPRequestState,
        submittingLeadingLayers leadingLayers: Int
    ) throws -> MLXArray {
        let state = self.state(requestState)
        // A state whose committed rows were all absorbed ahead of this round
        // (`prefetchCommittedContext`) proposes over its cache alone.
        guard !state.pending.isEmpty || state.contextPrefetched else {
            throw DFlash2Error.emptyBlockContext
        }
        let context: MLXArray? =
            state.pending.isEmpty
            ? nil
            : (state.pending.count == 1
                ? state.pending[0] : concatenated(state.pending, axis: 1))
        seedCacheOffsets(state)
        let tokens = try drafter.propose(
            anchor: [anchor], targetHidden: context, cache: state.caches,
            blockSize: depth + 1, submittingLeadingLayers: leadingLayers)
        state.absorbPending()
        state.contextPrefetched = false
        state.roots.append(tokens)
        return tokens
    }

    /// The cache must sit where the retained context actually starts. This is
    /// the reference's `cache.offset = prompt.size - rows`: a prompt longer
    /// than the context window leaves the first retained row at a positive
    /// position, and the block's rotations follow it.
    private func seedCacheOffsets(_ state: RequestState) {
        guard !state.cacheSeeded else { return }
        for cache in state.caches {
            guard let base = cache as? BaseKVCache else { continue }
            base.offset = state.firstPendingPosition
        }
        state.cacheSeeded = true
    }

    // MARK: - Context prefetch

    /// Kill switch for absorbing a prompt's context ahead of the first block
    /// (default on): `MLXFAST_DFLASH_CONTEXT_PREFETCH=0`.
    static let contextPrefetchEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_CONTEXT_PREFETCH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Only a prompt's worth of rows is worth a separate submission.
    static let contextPrefetchMinimumRows = 64

    /// Absorb the committed context rows this state holds into the drafter's
    /// cache now, instead of inside the next block's forward.
    ///
    /// A layer's context keys and values are a function of the context rows
    /// alone; only the block needs the anchor. The engine calls this right
    /// after the prompt forward that produced the rows, and evaluates the
    /// result in its own submission behind the prompt's, so the prompt's
    /// sampled token never waits for it and the first round's block forward
    /// covers the block rows only. Returns the arrays to evaluate, or nothing
    /// when the rows stay pending for the block as before.
    public func prefetchCommittedContext(
        requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        guard Self.contextPrefetchEnabled else { return [] }
        let state = self.state(requestState)
        guard !state.isReleased, state.pendingRows >= Self.contextPrefetchMinimumRows
        else { return [] }
        let context =
            state.pending.count == 1
            ? state.pending[0] : concatenated(state.pending, axis: 1)
        seedCacheOffsets(state)
        guard (try? drafter.absorbContext(targetHidden: context, cache: state.caches)) == true
        else { return [] }
        state.absorbPending()
        state.contextPrefetched = true
        return state.caches.flatMap { $0.innerState() }
    }

    public func trimBlockState(
        _ requestState: any CBv2MTPRequestState, toCommittedLength committed: Int
    ) {
        drafter.trimCache(state(requestState).caches, toCommittedLength: committed)
    }

    // MARK: - Round lifecycle

    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] {
        let state = self.state(requestState)
        return state.caches.flatMap { $0.innerState() } + state.roots
    }

    /// A round's confirmed columns become the next block's context.
    ///
    /// `confirmedInputTokens` and `committedDraftTokens` are the CHAIN seam's
    /// accounting: they describe which speculative head inputs survived. A
    /// block drafter stages nothing speculative — every row it absorbed was
    /// already committed target history — so it reads neither, and the
    /// confirmed hidden rows are the whole of what it needs.
    public func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        let state = self.state(requestState)
        state.roots.removeAll(keepingCapacity: true)
        append(committedTargetHidden, to: requestState)
    }

    /// Nothing to undo: the round wrote no speculative state here. The rows
    /// the proposal absorbed were committed before it ran, and the proposal
    /// itself is discarded with its graph.
    public func discardRound(requestState: any CBv2MTPRequestState) {
        state(requestState).roots.removeAll(keepingCapacity: true)
    }
}
