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
public final class Qwen35DFlash2Assistant: CBv2MTPBlockDrafter, @unchecked Sendable {

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
        // Warmup gap #1 (behavior-neutral): the serial warm stepper never
        // drafts, so without this the first block propose at each width
        // compiles in-window on the candidate leg only. Outputs are
        // evaluated and discarded; neither the drafter nor the target is
        // mutated.
        try Self.warmBlockShapes(drafter)
        let assistant = Qwen35DFlash2Assistant(drafter: drafter, target: text)
        assistant.warmSpeculativeShapes()
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
    func warmSpeculativeShapes() {
        guard Self.speculativeWarmEnabled else { return }
        warmDrafter()
        Stream().synchronize()
        Memory.clearCache()
    }

    private func warmDrafter() {
        let block = Self.warmBlockSize
        guard let caches = try? drafter.makeCache() else { return }
        let width = drafter.config.targetHiddenSize
        var offset = 0
        for rows in [513] + Array(1 ... block) {
            let context = MLXArray.zeros([1, rows, width], dtype: drafter.dtype)
            guard
                let tokens = try? drafter.propose(
                    anchor: [0], targetHidden: context, cache: caches, blockSize: block)
            else { return }
            eval([tokens] + caches.flatMap { $0.innerState() })
            offset += rows
            drafter.trimCache(caches, toCommittedLength: offset)
        }
    }

    /// Eval+discard `propose` probes over every legal block size, at drafter
    /// load (outside both timed phases).
    ///
    /// One probe per block size warms the drafter trunk at M = blockSize,
    /// the target `lm_head` at M = blockSize - 1 (the prompt head projects
    /// the last row only and decode warms M = 1, so M 2...16 is cold), the
    /// fused logits epilogue, and the greedy top-16 candidate selector over
    /// full-vocabulary rows. The context is a short fixed slice: context
    /// length varies per round by construction, while the block path is the
    /// fixed-shape first fire this warms. The scratch cache is dropped with
    /// the probe.
    private static func warmBlockShapes(_ drafter: DFlash2DraftModel) throws {
        let context = MLXArray.zeros(
            [1, 8, drafter.config.targetHiddenSize], dtype: drafter.dtype)
        for blockSize in 2 ... (CBv2MTPConfig.testedMaxBlockDraftTokens + 1) {
            let tokens = try drafter.propose(
                anchor: [100], targetHidden: context,
                cache: try drafter.makeCache(), blockSize: blockSize)
            eval(tokens)
        }
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
    /// its FP32 norms; the drafter is BF16. Pending rows stay in the trunk's
    /// dtype here and cross in `proposeBlock` (see `fusedProposeContext`), so
    /// per-chunk casts never execute as their own elementwise passes at the
    /// draft round's eval fence on top of the full-context cast
    /// `DFlash2DraftModel.hiddenStates` already performs. Slice-then-cast
    /// casts the same elements as cast-then-slice, so every assembly below is
    /// bit-identical over the same elements.
    ///
    /// Fallback: `BONSAI_DFLASH2_EAGER_CAST=1` restores the legacy per-chunk
    /// cast in `append`; `proposeBlock` then sees rows already in the
    /// drafter's dtype and every cast below is a guarded no-op.
    private static let eagerPerChunkCast: Bool =
        ProcessInfo.processInfo.environment["BONSAI_DFLASH2_EAGER_CAST"] == "1"

    private func append(_ hidden: MLXArray, to requestState: any CBv2MTPRequestState) {
        let state = self.state(requestState)
        guard hidden.dim(1) > 0 else { return }
        let rows = Self.eagerPerChunkCast ? hidden.asType(drafter.dtype) : hidden
        state.append(rows, limit: contextRowLimit)
    }

    // MARK: - Proposing

    /// Force the wide assembly below for parity A/B.
    private static let forceWideConcat: Bool =
        ProcessInfo.processInfo.environment["BONSAI_DFLASH2_WIDE_CONCAT"] == "1"

    /// Force the narrow assembly below for parity A/B.
    private static let forceNarrowConcat: Bool =
        ProcessInfo.processInfo.environment["BONSAI_DFLASH2_NARROW_CONCAT"] == "1"

    /// The round's fused context in the drafter's dtype: the tap-gather rows
    /// (already fused across the tap layers upstream), concatenated along the
    /// sequence axis, cast exactly once per element.
    ///
    /// Large round contexts assemble NARROW: one lazy cast per shard, then a
    /// SINGLE `concatenated` in BF16. That moves ~10 bytes/element and never
    /// materializes the wide FP32 intermediate (~14 bytes/element for
    /// concat-then-cast at five tap layers). Small round contexts keep the
    /// WIDE path — two launches beat C + 1 when there is almost nothing to
    /// move. A lone pending chunk needs no concat at all: one guarded cast,
    /// zero-copy when the rows already arrived in the drafter's dtype. The
    /// choice reads shape metadata only, never a host sync, and the
    /// `hiddenStates` cast stays a same-dtype no-op downstream either way.
    private func fusedProposeContext(_ state: RequestState) -> MLXArray {
        let dtype = drafter.dtype
        if state.pending.count == 1 {
            let only = state.pending[0]
            return only.dtype == dtype ? only : only.asType(dtype)
        }
        let narrow: Bool
        if Self.forceNarrowConcat {
            narrow = true
        } else if Self.forceWideConcat {
            narrow = false
        } else {
            let elements = state.pendingRows * state.pending[0].dim(-1)
            narrow = elements >= (state.pending.count - 1) * 2_000_000
        }
        if narrow {
            return concatenated(
                state.pending.map { $0.dtype == dtype ? $0 : $0.asType(dtype) },
                axis: 1)
        }
        let fused = concatenated(state.pending, axis: 1)
        return fused.dtype == dtype ? fused : fused.asType(dtype)
    }

    public func proposeBlock(
        anchor: Int, depth: Int, requestState: any CBv2MTPRequestState
    ) throws -> MLXArray {
        let state = self.state(requestState)
        guard !state.pending.isEmpty else { throw DFlash2Error.emptyBlockContext }
        let context = fusedProposeContext(state)
        if !state.cacheSeeded {
            // The cache must sit where the retained context actually starts.
            // This is the reference's `cache.offset = prompt.size - rows`: a
            // prompt longer than the context window leaves the first retained
            // row at a positive position, and the block's rotations follow it.
            for cache in state.caches {
                guard let base = cache as? BaseKVCache else { continue }
                base.offset = state.firstPendingPosition
            }
            state.cacheSeeded = true
        }
        let tokens = try drafter.propose(
            anchor: [anchor], targetHidden: context, cache: state.caches,
            blockSize: depth + 1)
        state.absorbPending()
        state.roots.append(tokens)
        return tokens
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
