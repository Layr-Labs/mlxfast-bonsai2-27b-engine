// MTPContractsV2.swift
//
// ContinuousBatchingV2 — Gemma-4-style MTP (multi-token prediction /
// speculative decoding) integration contracts.
//
// Design (mirrors the KV-less frozen-KV drafter that vLLM and SGLang
// independently converged on for Gemma 4, and our own v1 engine shipped):
//
//   - The DRAFTER writes no KV. Each round it attends a snapshot of the
//     target's KV at exactly TWO layers (the last non-shared full-attention
//     layer and the last non-shared sliding-attention layer), with a
//     CONSTANT query RoPE position per round (the anchor = the absolute
//     position of the row's newest confirmed-but-unfed token).
//   - Per round, for each speculating row: chain k drafter forwards
//     (greedy argmax, seed = newest confirmed token + the target's pre-norm
//     hidden at the position before it), then verify [seed, d_1..d_k] in ONE
//     rectangular [B, 1+k] target forward. The accept-walk
//     (`Gemma4SpeculativeWalk` semantics: accept while target argmax ==
//     draft, always emit target argmax at the first divergence / bonus
//     position) emits a+1 tokens; the k−a rejected tokens are rolled back
//     per row (exact — see `CBv2SequenceKV.supportsSpeculativeWrites`).
//   - GREEDY-ONLY losslessness is the parity invariant: MTP-on output is
//     token-exact vs MTP-off for temperature-0 requests. Non-greedy rows
//     never speculate.
//
// Layering: everything here is model-family-agnostic. The Gemma-4 drafter
// module, its masks, and the accept-walk live in MLXLLM; they reach the
// engine through `CBv2MTPDrafter` / `CBv2MTPForwardable` (same pattern as
// `CBv2EmbeddingForwardable` for multimodal).

import Foundation
import MLX

// MARK: - Model seam (verify forward + capture geometry)

/// Additive target capability for extracting exact top-two policy evidence
/// from logits without forcing device evaluation.
public protocol CBv2MTPPolicyTopTwoProviding: AnyObject {
    func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray)
}

/// Runtime availability for type-erased model adapters whose static wrapper
/// type cannot express whether the wrapped target implements top-two.
public protocol CBv2MTPPolicyTopTwoCapabilityProviding: AnyObject {
    var cbv2MTPPolicyTopTwoAvailable: Bool { get }
}

/// Which layer indices the engine snapshots for the drafter's frozen KV.
/// Indices are MODEL layer indices (== positions in the engine's per-layer
/// caches array). Both referenced layers must OWN storage (non-KV-shared).
public struct CBv2MTPCaptureLayers: Sendable, Equatable {
    /// Last non-shared full-attention layer.
    public var full: Int
    /// Last non-shared sliding-attention layer.
    public var sliding: Int
    public init(full: Int, sliding: Int) {
        self.full = full
        self.sliding = sliding
    }
}

/// Model-level surface for `LanguageModel` conformers reached through
/// `CBv2SteppableLanguageModelAdapter` (Gemma4TextModel conforms): the
/// KVCache-shaped twin of the `CBv2MTPSteppableModel` requirements.
public protocol CBv2MTPForwardable: AnyObject {
    /// nil when this model cannot drive MTP (no capture layers).
    var cbv2MTPCaptureLayers: CBv2MTPCaptureLayers? { get }
    /// Forward returning (softcapped) logits [B, L, vocab] AND the pre-norm
    /// last-decoder-layer hidden [B, L, hidden] — the tensor the Gemma-4
    /// drafter was trained against. Must be numerically identical to the
    /// plain forward on the logits side.
    func cbv2ForwardWithHidden(_ tokens: MLXArray, caches: [KVCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
}

extension CBv2MTPForwardable {
    /// Identity may normalize an outer language-model wrapper to the inner
    /// text target that actually owns embeddings, logits, hidden, and KV.
    public var cbv2MTPTargetIdentity: ObjectIdentifier { ObjectIdentifier(self) }
}

/// Recurrent target counterpart to `CBv2MTPForwardable`. Hybrid targets must
/// receive the same request-owned transaction objects used by ordinary CBv2
/// decode; the assistant never owns or aliases these states.
public protocol CBv2RecurrentMTPForwardable:
    AnyObject, CBv2RecurrentLanguageModelForwardable
{
    var cbv2MTPTargetIdentity: ObjectIdentifier { get }
    func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray)
}

extension CBv2RecurrentMTPForwardable {
    public var cbv2MTPTargetIdentity: ObjectIdentifier { ObjectIdentifier(self) }
}

/// Capture-verify refinement of `CBv2RecurrentMTPForwardable` (MTPLX GDN
/// capture-commit pattern): ONE forward over the whole `[B, 1+k]` verify
/// window during which every recurrent layer stages per-position captured
/// conv/SSM stacks via `CBv2RecurrentStateEvaluation.stageCaptured`. Commit
/// selects the captured state at the accepted position on device; rollback
/// keeps the pre-verify committed state. Attention KV rolls back by trim as
/// usual, so no repair forward is ever needed.
public protocol CBv2RecurrentCaptureMTPForwardable: CBv2RecurrentMTPForwardable {
    /// Same contract as `cbv2ForwardWithHidden`, but each recurrent
    /// transaction receives captured `[L, ...]` per-position stacks
    /// (`L == tokens.dim(1)`) instead of a single final state.
    func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray)
}

/// Steppable models that can drive MTP rounds. Additive refinement of
/// `CBv2SteppableModel`; the engine speculates only when the bound model
/// conforms AND `mtpCaptureLayers` is non-nil AND a drafter is configured.
public protocol CBv2RecurrentPrefillHiddenForwardable: CBv2RecurrentMTPForwardable {
    /// Preserve every trusted hidden row needed by the assistant, while
    /// projecting vocabulary logits only at the positions the caller needs.
    func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray)
}

public protocol CBv2MTPSteppableModel: CBv2SteppableModel {
    /// nil when the underlying model cannot drive MTP (adapters over
    /// arbitrary models answer at runtime).
    var mtpCaptureLayers: CBv2MTPCaptureLayers? { get }
    /// True only when the target has the recurrent hidden/state transaction
    /// seam required by a request-stateful assistant.
    var supportsRequestStatefulMTP: Bool { get }
    /// Identity of the exact target instance that owns verification logits,
    /// hidden states, and KV. nil means compatibility cannot be proven and
    /// must fail safe to plain decode.
    var mtpTargetIdentity: ObjectIdentifier? { get }
    /// Forward returning logits [B, L, vocab] and pre-norm last hidden
    /// [B, L, hidden]. Same cache/attention semantics as `forward`.
    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
}

extension CBv2MTPSteppableModel {
    public var mtpTargetIdentity: ObjectIdentifier? { nil }
    public var supportsRequestStatefulMTP: Bool { false }
}

/// Engine-facing recurrent hidden-capture refinement. A recurrent MTP target
/// is only activated when this seam and request-owned recurrent state are both
/// present, so no call can silently fall through to the stateless forward.
public protocol CBv2RecurrentMTPSteppableModel:
    CBv2MTPSteppableModel, CBv2RecurrentSteppableModel
{
    func forwardWithHiddenForPrefill(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray)
    func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray)
    /// True only when `forwardWithHiddenCaptured` stages per-position
    /// captured recurrent stacks (MTP capture-verify). Without it,
    /// rectangular verification over a recurrent target is impossible and
    /// the driver falls back to the serial oracle.
    var supportsCapturedVerifyWindow: Bool { get }
    /// Capture-verify forward over the whole verify window. Only called when
    /// `supportsCapturedVerifyWindow == true`.
    func forwardWithHiddenCaptured(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray)
}

extension CBv2RecurrentMTPSteppableModel {
    public func forwardWithHiddenForPrefill(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        forwardWithHidden(tokens: tokens, caches: caches, recurrentState: recurrentState,
                          positionIds: positionIds)
    }

    /// Fail-safe defaults for first-generation recurrent targets: no
    /// captured-window support (the serial oracle remains the verify path).
    public var supportsCapturedVerifyWindow: Bool { false }

    public func forwardWithHiddenCaptured(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        preconditionFailure(
            "CBv2 capture-verify forward called on a model without captured-window support")
    }
}

// MARK: - Drafter seam

/// One speculating row's frozen-KV capture for a round: snapshot views of
/// the target's retained KV at the two capture layers, plus the row's
/// anchor geometry. Views are per-row (no padding — the drafter pads and
/// masks internally, so mixed retained lengths across rows are fine).
public struct CBv2MTPRowCapture {
    /// Full-attention capture layer: [1, kvHeads, Tfull, headDim], temporal
    /// order, post-RoPE (captured from storage the target attended).
    public var fullKeys: MLXArray
    public var fullValues: MLXArray
    /// Sliding-attention capture layer: [1, kvHeads, Tslide, headDim],
    /// temporal order. Window-limited by storage eviction.
    public var slidingKeys: MLXArray
    public var slidingValues: MLXArray
    /// Absolute position of the FIRST retained sliding entry (the sliding
    /// KV covers positions [slidingStart, anchor)). The full capture always
    /// starts at 0.
    public var slidingStart: Int
    /// The round's frozen query position: absolute position of the row's
    /// newest confirmed-but-unfed token (== the row's absoluteOffset).
    public var anchor: Int

    public init(
        fullKeys: MLXArray, fullValues: MLXArray,
        slidingKeys: MLXArray, slidingValues: MLXArray,
        slidingStart: Int, anchor: Int
    ) {
        self.fullKeys = fullKeys
        self.fullValues = fullValues
        self.slidingKeys = slidingKeys
        self.slidingValues = slidingValues
        self.slidingStart = slidingStart
        self.anchor = anchor
    }
}

/// Opaque round-scoped state a drafter builds once per round from the
/// per-row captures (padded/stacked batch KV, per-row masks, positions).
public protocol CBv2MTPPreparedCapture: AnyObject {}

/// The engine's view of a drafter. Implemented in MLXLLM by an adapter over
/// `Gemma4AssistantDraftModel` bound to the engine's target model (the
/// adapter owns target-embedding lookup, mask construction, and greedy
/// argmax). All methods are called on the engine thread while building the
/// step graph; they MUST NOT force evaluation (no host syncs).
public protocol CBv2MTPDrafter: AnyObject {
    /// Identity of the exact target instance whose embeddings and geometry
    /// this drafter consumes. nil means compatibility cannot be proven and
    /// must fail safe to plain decode.
    var mtpTargetIdentity: ObjectIdentifier? { get }
    /// A drafter may narrow target verification, depth, or batch policy for
    /// correctness. nil preserves the Gemma/default engine configuration.
    var requiredVerificationMode: CBv2MTPVerificationMode? { get }
    var maximumDraftTokens: Int? { get }
    var maximumSpeculativeBatch: Int? { get }
    /// True when this drafter's rounds may accept via target-prefix
    /// pre-sampling; see the extension default for the full contract.
    var supportsTargetPrefixAcceptance: Bool { get }
    /// The first stateless draft token may be submitted while target graph
    /// construction continues. Its graph must only read immutable weights
    /// and the engine's fenced captures, with no mutable assistant state.
    var supportsEarlyDraftSubmission: Bool { get }
    /// True when this model's paged full-attention layers may represent an
    /// exact rectangular window as independent native decode rows.
    var prefersBatchedRectangularAttention: Bool { get }
    /// Variable request-owned residency outside target KV. Admission charges
    /// this conservatively for every reserved token when the drafter is active.
    var requestStateBytesPerToken: Int { get }
    /// Physical allocation granularity of the request-owned state, in tokens.
    /// A value greater than one makes admission charge whole allocation blocks.
    var requestStateTokenGranularity: Int { get }
    /// Maximum physical high-water tokens retained beyond committed logical state.
    var requestStateTokenAllocationPadding: Int { get }
    /// Physical buffer families for allocator-aware request admission. A nil
    /// declaration uses the conservative fragmented-byte fallback.
    var requestStateAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? { get }
    /// Build round-scoped batch state from per-row captures. `rows` order
    /// == the round's speculating-row order.
    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture
    /// One draft-chain step over all speculating rows.
    ///  - tokens: [B, 1] int32 (lazy) — seed tokens (round start: each
    ///    row's newest confirmed token; later steps: previous draft).
    ///  - hidden: [B, 1, H] — round start: the target's pre-norm hidden at
    ///    the position BEFORE the seed token; later steps: the drafter's
    ///    own previous output hidden.
    ///  - Returns greedy next-token ids [B] int32 (lazy) and the drafter's
    ///    output hidden [B, 1, H] for chaining.
    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray)
}

extension CBv2MTPDrafter {
    public var mtpTargetIdentity: ObjectIdentifier? { nil }
    public var requiredVerificationMode: CBv2MTPVerificationMode? { nil }
    public var maximumDraftTokens: Int? { nil }
    public var maximumSpeculativeBatch: Int? { nil }
    public var requestStateBytesPerToken: Int { 0 }
    public var requestStateTokenGranularity: Int { 1 }
    public var requestStateTokenAllocationPadding: Int { 0 }
    public var requestStateAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? { nil }
    /// True when this drafter's rounds may accept via target-prefix
    /// pre-sampling (accept draft iff it equals a token pre-sampled from the
    /// target's real per-request sampler distribution; the committed token is
    /// always that target sample — exact for the output distribution at ANY
    /// temperature/top-p/top-k). Lifts the engine's `temperature == 0`
    /// eligibility gate when the installed sampler also supports MTP verify
    /// sampling. Default false: greedy argmax acceptance only.
    public var supportsTargetPrefixAcceptance: Bool { false }
    public var supportsEarlyDraftSubmission: Bool { false }
    public var prefersBatchedRectangularAttention: Bool { false }
}

/// Opaque, request-owned assistant state. It is deliberately distinct from
/// target recurrent state and target attention KV.
public protocol CBv2MTPRequestState: AnyObject {
    var committedInputCount: Int { get }
    var stagedInputCount: Int { get }
    /// Actual materialized device-array residency currently owned by this state.
    var materializedBytes: Int { get }
    /// Before drafting: true when queued history requires preparation beyond
    /// a normal steady decode round. Accounting only; never suppresses work.
    var hasPendingPrefillForCostAccounting: Bool { get }
}

extension CBv2MTPRequestState {
    public var materializedBytes: Int { 0 }
    public var hasPendingPrefillForCostAccounting: Bool { false }
}

/// Trusted target inputs and their corresponding pre-norm hidden rows.
/// Both arrays remain lazy and device-resident until the engine's existing
/// finalize fence.
public struct CBv2MTPCommittedTargetObservation {
    public let tokens: MLXArray
    public let hidden: MLXArray

    public init(tokens: MLXArray, hidden: MLXArray) {
        self.tokens = tokens
        self.hidden = hidden
    }
}

/// Alternate drafter seam for autoregressive assistants such as Qwen3.5/3.6.
/// Calls are row-local so histories and assistant-cache offsets may differ;
/// the target verifier may still batch the resulting columns as `[B,1]`.
public protocol CBv2MTPRequestStatefulDrafter: CBv2MTPDrafter {
    func makeRequestState() -> any CBv2MTPRequestState
    /// Bind the admitted request's complete logical bound before the state
    /// allocates request-owned storage. Stateful assistants whose allocator
    /// needs an up-front promise (notably native paged KV) use this instead
    /// of guessing a process-wide maximum. The call is idempotent.
    func configureRequestState(
        _ requestState: any CBv2MTPRequestState, maximumSequenceLength: Int) throws
    /// Queue trusted target inputs/hidden rows without evaluating them.
    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState)

    /// Draft-head shortlist opt-in. Non-nil K asks target verification to
    /// additionally surface each verify position's top-K token ids and
    /// their probability mass; finalize threads the accepted position's ids
    /// into the next round's carry when the mass clears the engine coverage
    /// threshold. nil keeps full-head draft scoring and adds zero verify
    /// work. Default nil.
    var draftShortlistSize: Int? { get }
    /// One request-local draft proposal.
    /// - shortlist: [K] int32 target top-K token ids captured at the carry
    ///   position, non-nil only when `draftShortlistSize` is set AND the
    ///   captured mass cleared the coverage threshold. When present the
    ///   drafter MUST propose a token from these ids (it may score only the
    ///   matching head rows instead of the full vocabulary).
    func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray)
    /// Device arrays that make assistant-cache mutation part of the round's
    /// evaluation fence.
    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray]
    /// Complete one staged round. The drafter stages the seed plus every
    /// proposal it consumes while chaining; `confirmedInputTokens` is the
    /// exact prefix of those inputs that became canonical target history.
    func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray)
    /// Reject/discard all assistant writes made by the current round.
    func discardRound(requestState: any CBv2MTPRequestState)
    /// Explicitly sever device-array ownership on finish/cancel/preemption.
    func releaseRequestState(_ requestState: any CBv2MTPRequestState)
}

extension CBv2MTPRequestStatefulDrafter {
    public var draftShortlistSize: Int? { nil }
    public func configureRequestState(
        _ requestState: any CBv2MTPRequestState, maximumSequenceLength: Int
    ) throws {}
}

// MARK: - Block drafter seam

/// A drafter that proposes a WHOLE BLOCK in one forward (DFlash 2), instead
/// of chaining `draftStep` k times.
///
/// THIS IS A SIBLING VERB, NOT AN ADAPTER. A block drafter's k draft tokens
/// come out of ONE forward over `1 + k` input positions — the last committed
/// token followed by k mask tokens — read at the mask positions. There is no
/// per-position seed token, no per-position hidden to chain, and no
/// intermediate draft state for the engine to carry between positions.
/// Spelling that as k `draftStep` calls would oblige the engine to invent k-1
/// inputs the drafter does not read, and would hide the one property the
/// block shape buys: a round costs ONE drafter forward at any depth.
///
/// It REFINES the request-stateful seam because everything AROUND the
/// proposal is the same: the same per-request state object and lifecycle, the
/// same committed-history observations, the same round finalization, the same
/// planning and the same carry. Only the proposal verb differs, so only the
/// proposal verb is new. The chain verbs are defaulted below to a refusal,
/// the way `CBv2RecurrentMTPSteppableModel.forwardWithHiddenCaptured` is.
///
/// CONTEXT. A block drafter reads the TARGET's hidden state at several named
/// layers, fused along the feature axis. Those rows reach it through the two
/// committed-history seams it already has: `observeCommittedTarget` for
/// prompt and plain-decode positions, and `finalizeRound`'s
/// `committedTargetHidden` for a verified round's accepted columns. The
/// engine fills both from `blockContextHidden()` rather than from the
/// target's pre-norm last hidden.
public protocol CBv2MTPBlockDrafter: CBv2MTPRequestStatefulDrafter {
    /// Turn the target's context tap on or off. The engine calls this exactly
    /// once per engine build, so a leg that does not speculate pays nothing
    /// for a drafter that is merely resident.
    func setBlockContextArmed(_ armed: Bool) throws

    /// The fused context rows the TARGET produced in its LAST forward,
    /// `[B, L, taps * hidden]`. The engine reads it immediately after each
    /// forward whose positions become committed history, while that forward
    /// is still the last one. nil means the tap is off.
    func blockContextHidden() -> MLXArray?

    /// One round, one proposal: `depth` draft token ids as `[1, depth]`
    /// int32, lazy. Row-local, like every other request-stateful call.
    func proposeBlock(
        anchor: Int, depth: Int, requestState: any CBv2MTPRequestState
    ) throws -> MLXArray

    /// Align the drafter's own context cache with the target's committed
    /// length once a round has finalized.
    func trimBlockState(
        _ requestState: any CBv2MTPRequestState, toCommittedLength committed: Int)
}

extension CBv2MTPBlockDrafter {
    /// The chain verbs of the seams this one refines. A block drafter
    /// proposes once per round through `proposeBlock`; the engine's block
    /// branch never reaches these, so a caller that does has taken the wrong
    /// branch rather than found a slower path.
    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        preconditionFailure("CBv2 block drafter: prepare is not the block seam")
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("CBv2 block drafter: draftStep is not the block seam")
    }

    public func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("CBv2 block drafter: draftStep is not the block seam")
    }
}

// MARK: - Config

/// How the target scores one MTP draft chain.
///
/// `serialTarget` is the correctness baseline: every column uses the same
/// `[B, 1]` eager target forward as ordinary decode. It works independently
/// of chip-specific multi-position kernel numerics. `rectangular` is an
/// explicit optimization that scores all `1+k` columns in one `[B, 1+k]`
/// forward and therefore requires separate numerical certification.
public enum CBv2MTPVerificationMode: String, Sendable, Equatable {
    case serialTarget = "serial_target"
    case rectangular
    /// One captured rectangular target transaction whose shape-sensitive
    /// arithmetic is explicitly M1-equivalent at the model boundary.
    case rectangularExact = "rectangular_exact"
    case automatic
}

/// Engine-level MTP configuration (parallel to `CBv2CompiledDecodeConfig`).
public struct CBv2MTPConfig: Sendable {
    /// The largest draft depth covered by the production rectangular-shape
    /// validation matrix (`verify width = 1 + k`, widths 1...8).
    public static let testedMaxDraftTokens = 7
    /// The largest draft depth a BLOCK drafter may ask for. A block proposes
    /// its whole depth in ONE drafter forward, so the depth that pays for
    /// itself is larger than a chain's: DFlash 2's own `block_size` is 16
    /// draft tokens after the committed anchor. This ceiling is reached only
    /// by a build that passes it explicitly, and it NEVER moves
    /// `testedMaxDraftTokens`.
    public static let testedMaxBlockDraftTokens = 16
    /// CBv2's production rectangular batch ceiling.
    public static let testedMaxSpeculativeBatch = 8

    /// Master switch. The engine also requires a drafter instance and a
    /// conforming model; `enabled == true` without both is inert.
    public var enabled: Bool
    /// Max draft tokens per round (k). Rounds verify 1+k and emit 1...k+1.
    /// Clamped to the production-tested `0...7` range.
    public var maxDraftTokens: Int
    /// Optional deterministic override. nil selects the adaptive controller;
    /// a value selects a fixed step-global depth, clamped to
    /// `0...maxDraftTokens`. Fixed zero is an explicit target-only mode that
    /// keeps MTP construction and metrics active for bring-up.
    public var fixedDraftTokens: Int?
    /// Hard operational gate on decode rows in one plan, clamped to 1...8.
    /// Together with the k<=7 bound this caps staged window-KV to 64 token
    /// rows per storage-owning layer and one in-flight step. The adaptive
    /// controller is separately keyed by a planned-decode-row bucket.
    public var maxSpeculativeBatch: Int
    /// Target scoring strategy. Automatic verification is the safe default:
    /// it uses rectangular scoring only within the configured work envelope
    /// and otherwise clamps depth before draft work. Serial target scoring
    /// remains an explicit correctness fallback.
    public var verificationMode: CBv2MTPVerificationMode
    /// Maximum `batch * (1+k)` target rows eligible for automatic
    /// rectangular verification. The planner clamps larger work to a safe
    /// depth, including ordinary target-only decode when no positive depth
    /// fits. Defaults to ZERO: a positive envelope is the integrator's
    /// explicit claim that rectangular target evaluation is argmax-exact for
    /// the deployed chip/OS/MLX/model tuple at every shape inside it. With
    /// no envelope, automatic mode performs no speculative work. Ignored by
    /// explicit serial/rectangular modes.
    public var maxAutomaticRectangularTokens: Int
    /// The ceiling `maxDraftTokens` is clamped to. It is a per-decoder FACT,
    /// not a tuning knob: a chain decoder leaves it at `testedMaxDraftTokens`
    /// and only a block decoder's build passes `testedMaxBlockDraftTokens`.
    /// Without it a depth of 8...16 would be clamped to 7 silently, and a leg
    /// would measure a depth it did not ask for.
    public let draftTokenCeiling: Int
    /// Process-level kill switch: `DARKBLOOM_CBV2_MTP=0/false/no/off`
    /// disables MTP even when the provider enables it (same convention as
    /// `DARKBLOOM_CBV2_COMPILED`). Unset or any other value: no override.
    public static let envEnabled: Bool = {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP"] {
            return !["0", "false", "no", "off"].contains(raw.lowercased())
        }
        return true
    }()

    public init(
        enabled: Bool = false,
        maxDraftTokens: Int = Self.testedMaxDraftTokens,
        maxSpeculativeBatch: Int = 8,
        fixedDraftTokens: Int? = nil,
        verificationMode: CBv2MTPVerificationMode = .automatic,
        maxAutomaticRectangularTokens: Int = 0,
        draftTokenCeiling: Int = Self.testedMaxDraftTokens
    ) {
        self.enabled = enabled
        let ceiling = max(draftTokenCeiling, 0)
        self.draftTokenCeiling = ceiling
        let resolvedMax = min(max(maxDraftTokens, 0), ceiling)
        self.maxDraftTokens = resolvedMax
        self.maxSpeculativeBatch = min(
            max(maxSpeculativeBatch, 1), Self.testedMaxSpeculativeBatch)
        self.fixedDraftTokens = fixedDraftTokens.map {
            min(max($0, 0), resolvedMax)
        }
        self.verificationMode = verificationMode
        self.maxAutomaticRectangularTokens = max(0, maxAutomaticRectangularTokens)
    }

    /// The effective on/off state (config AND env kill switch).
    public var effectiveEnabled: Bool { enabled && Self.envEnabled }
}

// MARK: - Metrics

/// One controller wall-cost input exposed in a lock-safe metrics snapshot.
/// Cost is measured at the existing finalize boundary; collecting it adds no
/// MLX evaluation or tensor readback.
public struct CBv2MTPCostInput: Sendable, Equatable {
    public var decodeRowBucket: Int
    public var depth: Int
    public var samples: Int
    /// Whole-window elapsed time for adaptive stateless MTP; per-step time
    /// for ordinary and legacy policies. Use normalized cadence to compare.
    public var ewmaWallTimeNanos: UInt64
    public var totalWallTimeNanos: UInt64
    /// Per-row committed-token cadence used by adaptive stateless MTP.
    public var ewmaNanosPerCommittedToken: UInt64?

    public init(
        decodeRowBucket: Int, depth: Int, samples: Int,
        ewmaWallTimeNanos: UInt64, totalWallTimeNanos: UInt64,
        ewmaNanosPerCommittedToken: UInt64? = nil
    ) {
        self.decodeRowBucket = decodeRowBucket
        self.depth = depth
        self.samples = samples
        self.ewmaWallTimeNanos = ewmaWallTimeNanos
        self.totalWallTimeNanos = totalWallTimeNanos
        self.ewmaNanosPerCommittedToken = ewmaNanosPerCommittedToken
    }
}

/// Cumulative MTP counters (engine-thread mutated, snapshot under the
/// engine's stats lock). Per-position acceptance is the tuning signal for
/// `maxDraftTokens`.
public struct CBv2MTPMetrics: Sendable {
    /// True for every non-nil `EngineV2.mtpMetricsSnapshot()`. Kept explicit
    /// so provider telemetry can serialize one stable shape.
    public var active: Bool = true
    /// Target scoring strategy used by every round in this engine.
    public var verificationMode: CBv2MTPVerificationMode = .automatic
    /// Configured automatic rectangular work cap, exposed so benchmark
    /// validators can distinguish intentional target-only fallback from a
    /// failure to run the requested fixed depth.
    public var maxAutomaticRectangularTokens: Int = 0
    /// Actual target-verification rounds by strategy. Automatic mode can use
    /// both across batch/depth regimes.
    public var rectangularVerificationRounds: Int = 0
    public var serialVerificationRounds: Int = 0
    /// Most recently selected step-global depth (zero means target-only).
    public var selectedDepth: Int = 0
    /// Planned decode-row bucket used for the most recent selection.
    public var decodeRowBucket: Int = 0
    /// Rounds that drafted (k ≥ 1) and verified.
    public var rounds: Int = 0
    /// Seed steps (eligible rows that decoded eagerly with hidden capture
    /// to establish the drafter carry — no drafts yet).
    public var seedSteps: Int = 0
    /// Nonblocking assistant graph submissions before target verification.
    public var earlyDraftSubmissions: Int = 0
    /// Total draft tokens proposed across all rounds.
    public var draftedTokens: Int = 0
    /// Total draft tokens accepted across all rounds.
    public var acceptedTokens: Int = 0
    /// Total tokens emitted by MTP rounds (accepted + bonus/correction).
    public var emittedTokens: Int = 0
    /// perPositionAccepted[i] = rounds in which draft position i (0-based)
    /// was accepted. Monotonically non-increasing over i within a run.
    public var perPositionAccepted: [Int] = []
    /// Rows that were round-eligible but clamped to plain decode, keyed by
    /// reason ("batch_gate", "kv_headroom", "carry_invalid", ...).
    public var skippedRows: [String: Int] = [:]
    /// Step selections by depth, including depth zero.
    public var depthSelections: [Int: Int] = [:]
    /// Stable controller/fallback reasons (warmup, exploration, hysteresis,
    /// unprofitable depth zero, batch gate, token/KV headroom, and so on).
    public var controllerFallbacks: [String: Int] = [:]
    /// Conditional acceptance rate at each draft position: P(position i is
    /// accepted | every earlier draft position was accepted).
    public var conditionalAcceptance: [Double] = []
    /// Outlier-clamped wall-cost EWMAs and raw cumulative inputs, sorted by
    /// decode-row bucket then depth in snapshots.
    public var costInputs: [CBv2MTPCostInput] = []
    /// Sum of measured wall time for cost-eligible speculative rounds.
    public var totalRoundWallTimeNanos: UInt64 = 0
    /// Per-VERIFY-round acceptance/rollback audit records, in finalize order,
    /// bounded at [`CBv2MTPRoundAuditRecord.retainedRecordCap`] (oldest
    /// dropped first). Observability seam: it lets a provider's session
    /// diagnostics OBSERVE, per round, the exact accept-walk inputs (draft
    /// ids vs target ids), the chosen acceptance boundary, and the
    /// post-rollback scheduler/KV accounting — so an on-box token-exactness
    /// divergence can be attributed to a wrong per-token reference (targets
    /// not matching the serial model) versus a wrong rollback boundary
    /// (accept/discard off-by-one), instead of assuming either. Snapshot-only
    /// like every other field; populated at the same finalize host-sync
    /// boundary as the counters above, so it adds no MLX evaluation.
    public var roundAudits: [CBv2MTPRoundAuditRecord] = []

    public init() {}

    /// Preferred proposal-count spelling. `draftedTokens` remains stored for
    /// compatibility with the first engine/provider seam.
    public var proposedTokens: Int { draftedTokens }

    /// Mean accepted drafts per round (nil before any round).
    public var meanAcceptedPerRound: Double? {
        rounds > 0 ? Double(acceptedTokens) / Double(rounds) : nil
    }
}

/// One finalized VERIFY round's acceptance/rollback audit, captured at the
/// finalize host-sync boundary (`EngineLoopV2+MTPFinalize.swift`) from values
/// already on the host — no extra readback. The record states, for one row:
/// what was drafted, what the target's authoritative per-column argmaxes
/// were, where the accept walk stopped, how many staged KV/scheduler
/// positions were rolled back, and the row's post-round accounting.
public struct CBv2MTPRoundAuditRecord: Sendable, Equatable {
    /// Bound on `CBv2MTPMetrics.roundAudits` (oldest dropped first). Sized
    /// so a full benchmark window's audits always fit with headroom: the
    /// widest admitted cohort (B=8) over a 128-token window finalizes at
    /// most ~8 x 128 verify-row records; consumers that RECONCILE audits
    /// against committed streams (the cohort assembler) refuse when the
    /// count reaches this cap, because a truncated head means coverage can
    /// no longer be proven.
    public static let retainedRecordCap = 8192

    /// The row's request id raw value (B > 1 disambiguation).
    public var requestID: UInt64
    /// Draft depth k this round.
    public var k: Int
    /// The k draft ids fed as verify input columns 1...k.
    public var draftTokens: [Int]
    /// The 1+k target argmaxes (verify outputs; `targets[i]` is the
    /// authoritative next token after input column i).
    public var targetTokens: [Int]
    /// Accept-walk result: number of leading drafts with
    /// `drafts[i] == targets[i]`.
    public var accepted: Int
    /// Tokens actually committed this round (`kept.count`): the accepted
    /// prefix plus the correction/bonus, clamped by the cross-row common
    /// width, stop tokens, and max_tokens.
    public var confirmed: Int
    /// Staged KV entries rolled back (`(1 + k) - confirmed`).
    public var rejected: Int
    /// `rec.tokens.count` AFTER this round's commits (prompt + emitted).
    public var tokensCountAfter: Int
    /// `rec.numComputedTokens` AFTER the rollback. Boundary invariant: must
    /// equal `tokensCountAfter - 1` (every token computed except the new
    /// carry).
    public var numComputedAfter: Int
    /// `rec.generatedTokenCount` AFTER this round's commits.
    public var generatedAfter: Int
    /// Terminal reason when this round finished the row ("stop"/"length"),
    /// else nil.
    public var finishReason: String?

    public init(
        requestID: UInt64, k: Int, draftTokens: [Int], targetTokens: [Int],
        accepted: Int, confirmed: Int, rejected: Int,
        tokensCountAfter: Int, numComputedAfter: Int, generatedAfter: Int,
        finishReason: String?
    ) {
        self.requestID = requestID
        self.k = k
        self.draftTokens = draftTokens
        self.targetTokens = targetTokens
        self.accepted = accepted
        self.confirmed = confirmed
        self.rejected = rejected
        self.tokensCountAfter = tokensCountAfter
        self.numComputedAfter = numComputedAfter
        self.generatedAfter = generatedAfter
        self.finishReason = finishReason
    }
}
