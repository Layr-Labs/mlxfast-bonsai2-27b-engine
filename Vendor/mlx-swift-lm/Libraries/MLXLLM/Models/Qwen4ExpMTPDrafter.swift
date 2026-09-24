//
//  Qwen4ExpMTPDrafter.swift
//  mlx-swift-lm
//
//  The embedded Qwen 3.8 Flash-Next MTP head as a ContinuousBatchingV2
//  request-stateful drafter.
//
//  The head lives in the TARGET checkpoint under `mtp.*`. It owns no
//  embedding table and no output head: it reads the target's `embed_tokens`
//  and writes through the target's `lm_head`. So there is no artifact to
//  stage and no compatibility matrix to check — the drafter is built from an
//  already-loaded target and can only ever match it.
//
//  WHAT THE HEAD CONSUMES, stated because it is the part that is easy to get
//  wrong: the target's PRE-final-mixer hyper-connection stream, `hc_count *
//  hidden` wide. `Qwen4ExpModel.cbv2ForwardWithHidden` returns exactly that
//  as `lastHidden`, and every draft round re-embeds it. The collapsed hidden
//  the head-facing path uses would be the wrong tensor and the wrong width.
//
//  DEPTH 1...3. The head is one hybrid layer applied to its own output, so a
//  deeper chain drifts further from the target with no measured acceptance to
//  pay for it. Three is the ruled ceiling for this track.
//
//  ARGMAX TIE-BREAK is the lowest token id, which is what `argMax` returns.
//  The target verifier uses the same rule, so a draft can never be rejected
//  over a tie the two sides broke differently.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Drives the target's own `mtp.*` head as a CBv2 drafter.
public final class Qwen4ExpInlineMTPAssistant {

    /// The largest draft depth this head is served at. A POLICY bound, not a
    /// structural one: the head is one layer applied once per draft step
    /// with its own key-value rows, so the chain length is the number of
    /// steps the engine asks for. Ruled 6 on both platforms (David: "mtp at
    /// 6 on both mlx and cuda"); the CUDA track serves the same head at
    /// depth <= 6 with per-depth oracles.
    public static let maximumDepth = 6

    private let target: Qwen4ExpModel
    private let mtp: Qwen4ExpMTPModule

    /// - Parameter target: an already-loaded model whose `mtp` head is
    ///   present. A serial-only load has none and cannot draft.
    public init?(target: Qwen4ExpModel) {
        guard let mtp = target.mtp else { return nil }
        self.target = target
        self.mtp = mtp
    }

    /// Head caches, one per head layer.
    func makeCache() -> [KVCache] { mtp.makeCache() }

    /// One head application over `[1, S]` inputs, returning the LAST
    /// position's draft id and multi stream.
    ///
    /// Only the last row reaches `lm_head`: the earlier rows exist to put the
    /// head's key-value history in place, and projecting them would read the
    /// whole output head for tokens nothing consumes.
    private func headStep(
        tokens: MLXArray, multiStream: MLXArray, cache: [KVCache], stepIndex: Int
    ) -> (draft: MLXArray, multi: MLXArray) {
        let step = mtp(
            nextTokenIds: tokens,
            multiStream: multiStream,
            embedTokens: target.model.embedTokens,
            cache: cache,
            stepIndex: stepIndex)
        let last = step.sample.dim(1) - 1
        let lastSample = step.sample[0..., last..., 0...]
        let lastMulti = step.multi[0..., last..., 0...]
        let draft = argMax(target.head(lastSample)[0..., -1, 0...], axis: -1).asType(.int32)
        return (draft, lastMulti)
    }
}

// MARK: - CBv2 drafter

extension Qwen4ExpInlineMTPAssistant: CBv2MTPRequestStatefulDrafter {

    /// Per-request head state: the head's own key-value caches plus the
    /// trusted target transitions not yet folded into them.
    private final class RequestState: CBv2MTPRequestState {
        var caches: [any KVCache]

        /// Trusted target transitions waiting to enter head history. The multi
        /// stream at position t pairs with the token at t+1.
        var backlogMulti: [MLXArray] = []
        var backlogTokens: [MLXArray] = []
        /// Last trusted multi stream of an observed chunk. It becomes the
        /// preceding row when the next observed chunk crosses the boundary.
        var multiFrontier: MLXArray?

        /// Cache geometry captured around the round's trusted flush. Every
        /// later head input is speculative and is trimmed at finalize.
        var roundBaseOffset = 0
        var roundValidHistoryOffset = 0
        var roundDraftSteps = 0
        var roundInFlight = false
        var isReleased = false

        /// Trusted inputs moved out of the backlog for this round. Holding
        /// their roots fences the lazy concatenation and lets discard restore
        /// them without a host read.
        var roundTrustedMulti: [MLXArray] = []
        var roundTrustedTokens: [MLXArray] = []
        /// Proposal rows that retain each lazy head-step graph until the
        /// engine's finalize synchronization.
        var roundRoots: [MLXArray] = []

        var cacheOffset: Int {
            guard let first = caches.first else { return 0 }
            precondition(
                caches.dropFirst().allSatisfy { $0.offset == first.offset },
                "Qwen4Exp MTP head cache offsets diverged")
            return first.offset
        }

        private var backlogInputCount: Int {
            backlogTokens.reduce(0) { $0 + $1.dim(1) }
        }

        var committedInputCount: Int {
            (roundInFlight ? roundValidHistoryOffset : cacheOffset) + backlogInputCount
        }

        var stagedInputCount: Int {
            guard roundInFlight else { return 0 }
            return max(0, cacheOffset - roundValidHistoryOffset)
        }

        var materializedBytes: Int {
            let arrays =
                caches.flatMap { $0.innerState() }
                + backlogMulti + backlogTokens
                + [multiFrontier].compactMap { $0 }
                + roundTrustedMulti + roundTrustedTokens + roundRoots
            return arrays.reduce(0) { total, array in
                let (next, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : next
            }
        }

        init(caches: [any KVCache]) { self.caches = caches }

        func clearRound() {
            roundBaseOffset = cacheOffset
            roundValidHistoryOffset = cacheOffset
            roundDraftSteps = 0
            roundInFlight = false
            roundTrustedMulti.removeAll(keepingCapacity: true)
            roundTrustedTokens.removeAll(keepingCapacity: true)
            roundRoots.removeAll(keepingCapacity: true)
        }

        func clearAll() {
            caches.removeAll(keepingCapacity: false)
            backlogMulti.removeAll(keepingCapacity: false)
            backlogTokens.removeAll(keepingCapacity: false)
            multiFrontier = nil
            roundTrustedMulti.removeAll(keepingCapacity: false)
            roundTrustedTokens.removeAll(keepingCapacity: false)
            roundRoots.removeAll(keepingCapacity: false)
            roundBaseOffset = 0
            roundValidHistoryOffset = 0
            roundDraftSteps = 0
            roundInFlight = false
            isReleased = true
        }
    }

    /// The frozen-KV capture seam this drafter does not use.
    private final class UnusedPreparedCapture: CBv2MTPPreparedCapture {}

    public var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }

    /// No required mode: the target implements the captured verify window
    /// (`cbv2ForwardWithHiddenCaptured`), so the engine's configured mode
    /// applies and a round verifies its `1 + k` candidates in one forward.
    /// Serial target scoring remains available by configuration.
    public var requiredVerificationMode: CBv2MTPVerificationMode? { nil }

    public var maximumDraftTokens: Int? { Self.maximumDepth }
    public var maximumSpeculativeBatch: Int? { 1 }

    /// Head key-value rows, the retained indexer tape, one multi-stream row
    /// and one token id, per input token.
    public var requestStateBytesPerToken: Int {
        let configuration = target.configuration
        let elementBytes = target.model.embedTokens.weight.dtype.size
        let perLayer =
            2 * configuration.kvHeads * configuration.headDim + configuration.indexerHeadDim
        let head = perLayer * mtp.layerCount * elementBytes
        let stream = configuration.hcCount * configuration.hiddenSize * elementBytes
        return head + stream + MemoryLayout<Int32>.stride
    }

    public func makeRequestState() -> any CBv2MTPRequestState {
        RequestState(caches: makeCache())
    }

    private func typed(_ requestState: any CBv2MTPRequestState) -> RequestState {
        guard let state = requestState as? RequestState else {
            preconditionFailure("Qwen4Exp MTP received foreign request state")
        }
        return state
    }

    public func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {
        let state = typed(requestState)
        precondition(!state.isReleased, "Qwen4Exp MTP observed released request state")
        precondition(!state.roundInFlight, "Qwen4Exp MTP observed target during a round")
        precondition(
            observation.tokens.ndim == 2 && observation.hidden.ndim == 3
                && observation.tokens.dim(0) == 1 && observation.hidden.dim(0) == 1
                && observation.tokens.dim(1) == observation.hidden.dim(1),
            "Qwen4Exp MTP target observation shape mismatch")

        let count = observation.tokens.dim(1)
        guard count > 0 else { return }

        // Cross-chunk transition: the preceding chunk's final multi stream
        // pairs with this chunk's first target input.
        if let frontier = state.multiFrontier {
            state.backlogMulti.append(frontier)
            state.backlogTokens.append(observation.tokens[0..., 0 ..< 1])
        }
        // Intra-chunk transitions: multi[t] conditions token[t+1].
        if count > 1 {
            state.backlogMulti.append(observation.hidden[0..., 0 ..< count - 1, 0...])
            state.backlogTokens.append(observation.tokens[0..., 1 ..< count])
        }
        state.multiFrontier = observation.hidden[0..., (count - 1) ..< count, 0...]
    }

    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        UnusedPreparedCapture()
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("Qwen4Exp MTP requires request-owned head state")
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        let state = typed(requestState)
        precondition(!state.isReleased, "Qwen4Exp MTP drafted with released request state")
        precondition(
            tokens.ndim == 2 && hidden.ndim == 3
                && tokens.dim(0) == 1 && tokens.dim(1) == 1
                && hidden.dim(0) == 1 && hidden.dim(1) == 1,
            "Qwen4Exp MTP draft input shape mismatch")
        precondition(
            hidden.dim(2) == target.configuration.hcCount * target.configuration.hiddenSize,
            "Qwen4Exp MTP draft hidden must be the pre-final-mixer multi stream")

        let isFirstStep = !state.roundInFlight
        let feed: (tokens: MLXArray, multi: MLXArray)
        if isFirstStep {
            feed = beginRound(tokens: tokens, multi: hidden, state: state)
        } else {
            precondition(
                state.roundDraftSteps < Self.maximumDepth,
                "Qwen4Exp MTP exceeded its \(Self.maximumDepth)-step draft chain")
            feed = (tokens, hidden)
        }

        let step = headStep(
            tokens: feed.tokens, multiStream: feed.multi, cache: state.caches,
            stepIndex: state.roundDraftSteps)
        state.roundRoots.append(contentsOf: [step.multi, step.draft])
        state.roundDraftSteps += 1

        if isFirstStep {
            // The engine submits this generation's evaluation targets before
            // it builds a deeper draft step, so the trusted flush ends here.
            state.roundValidHistoryOffset = state.cacheOffset
        }
        return (step.draft, step.multi)
    }

    private func beginRound(
        tokens: MLXArray, multi: MLXArray, state: RequestState
    ) -> (tokens: MLXArray, multi: MLXArray) {
        precondition(!state.roundInFlight, "Qwen4Exp MTP round already in flight")
        precondition(
            state.backlogMulti.count == state.backlogTokens.count,
            "Qwen4Exp MTP trusted backlog diverged")

        state.roundBaseOffset = state.cacheOffset
        state.roundValidHistoryOffset = state.cacheOffset
        state.roundDraftSteps = 0
        state.roundInFlight = true
        state.roundTrustedMulti = state.backlogMulti
        state.roundTrustedTokens = state.backlogTokens
        state.backlogMulti.removeAll(keepingCapacity: true)
        state.backlogTokens.removeAll(keepingCapacity: true)

        // The current target carry is trusted and completes the frontier
        // transition.
        state.roundTrustedMulti.append(multi)
        state.roundTrustedTokens.append(tokens)
        state.multiFrontier = nil

        if state.roundTrustedTokens.count == 1 {
            return (state.roundTrustedTokens[0], state.roundTrustedMulti[0])
        }
        return (
            concatenated(state.roundTrustedTokens, axis: 1),
            concatenated(state.roundTrustedMulti, axis: 1)
        )
    }

    public func evaluationTargets(
        for requestState: any CBv2MTPRequestState
    ) -> [MLXArray] {
        guard let state = requestState as? RequestState, !state.isReleased else { return [] }
        return state.caches.flatMap { $0.innerState() }
            + state.backlogMulti + state.backlogTokens
            + [state.multiFrontier].compactMap { $0 }
            + state.roundTrustedMulti + state.roundTrustedTokens + state.roundRoots
    }

    public func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray
    ) {
        let state = typed(requestState)
        precondition(!state.isReleased, "Qwen4Exp MTP finalized released request state")
        precondition(state.roundInFlight, "Qwen4Exp MTP finalized without a round")
        precondition(
            (0 ... state.roundDraftSteps + 1).contains(confirmedInputTokens),
            "Qwen4Exp MTP confirmed prefix exceeds the draft round")
        precondition(
            committedDraftTokens.ndim == 2 && committedTargetHidden.ndim == 3
                && committedDraftTokens.dim(0) == 1 && committedTargetHidden.dim(0) == 1
                && committedDraftTokens.dim(1) == committedTargetHidden.dim(1),
            "Qwen4Exp MTP committed target rows mismatch")
        let committedDraftCount = committedDraftTokens.dim(1)
        precondition(
            committedDraftCount <= state.roundDraftSteps
                && committedDraftCount <= max(0, confirmedInputTokens - 1),
            "Qwen4Exp MTP committed drafts exceed confirmed target inputs")

        trim(state: state, to: state.roundValidHistoryOffset)
        if committedDraftCount > 0 {
            // These are target verify multi streams, never speculative head
            // outputs. They flush with the next carry.
            state.backlogTokens.append(committedDraftTokens)
            state.backlogMulti.append(committedTargetHidden)
        }
        state.clearRound()
    }

    public func discardRound(requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState,
            !state.isReleased, state.roundInFlight
        else { return }

        trim(state: state, to: state.roundBaseOffset)
        // Restore every trusted transition the abandoned graph consumed, in
        // original order. No speculative head output enters the backlog.
        state.backlogMulti = state.roundTrustedMulti + state.backlogMulti
        state.backlogTokens = state.roundTrustedTokens + state.backlogTokens
        state.clearRound()
    }

    /// Roll the head's own caches back to `offset`.
    ///
    /// `Qwen4ExpAttentionCache.trim` moves the key-value offset AND slices the
    /// indexer tape, so the head's sparse attention sees the same history a
    /// round that never happened would have left.
    private func trim(state: RequestState, to offset: Int) {
        let rollback = state.cacheOffset - offset
        precondition(rollback >= 0, "Qwen4Exp MTP cache checkpoint moved forward")
        guard rollback > 0 else { return }
        for cache in state.caches {
            precondition(
                cache.trim(rollback) == rollback,
                "Qwen4Exp MTP head cache refused a \(rollback) token rollback")
        }
    }

    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let state = requestState as? RequestState, !state.isReleased else { return }
        state.clearAll()
    }
}
