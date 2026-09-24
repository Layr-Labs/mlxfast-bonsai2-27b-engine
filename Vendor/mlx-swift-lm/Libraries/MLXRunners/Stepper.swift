// Copyright © 2026 Eigen Labs.
//
// MLXRunners — the teacher-forced stepper (Darkbloom runner contract §7).
//
// benchd's v1 timed verbs (`prefill`, `decode_begin`, `decode_step`,
// `correctness_begin`, `correctness_step`) are all ONE shape: force a token
// sequence through a single row and read the last position's logits. The
// stepper is that shape, and it is NOT a second implementation of the model:
// it drives the same `CBv2SteppableLanguageModelAdapter` forward the engine
// drives, over the same `newCacheV2` caches and the same layer kinds.
//
// The stepper holds NO timer and emits no timing. benchd times the round
// trip; the worker reports counters only.

import Foundation
import MLX
import MLXLMCommon

/// Last-position logits of one teacher-forced forward.
public struct StepOutput: Sendable {
    /// Greedy token. Ties break to the LOWEST token id — the same rule the
    /// engine's sampler uses, so stepper and engine cannot disagree on a tie.
    public let argmax: Int
    /// Top-k entries, sorted by descending logit then ascending token id.
    /// `k` is bench-protocol's `TOP_LOGITS_K` (8).
    public let topLogits: [(token: Int, logit: Double)]
    /// Top logit minus second logit.
    public let margin: Double

    public init(argmax: Int, topLogits: [(token: Int, logit: Double)], margin: Double) {
        self.argmax = argmax
        self.topLogits = topLogits
        self.margin = margin
    }
}

/// A single-row, teacher-forced forward over a runner's serving model.
public protocol TeacherForcedStepper: AnyObject {
    /// Fresh session. The caller drains the allocator before this.
    func begin() throws
    /// Forward `tokens` at the current position; return the LAST position's
    /// logits as a top-k list plus argmax.
    func forward(_ tokens: [Int]) throws -> StepOutput
    /// Forwards executed since `begin()`. bench-worker reports this as
    /// `completed_work`.
    var forwards: Int { get }
}

/// Refusals from the stepper. Each is a refusal to produce a token, never a
/// substitute answer.
public enum StepperError: Error, CustomStringConvertible, Equatable {
    case notBegun
    case emptyForward
    case unsupportedModel(String)
    case recurrentLifecycle(String)

    public var description: String {
        switch self {
        case .notBegun: return "stepper: forward before begin()"
        case .emptyForward: return "stepper: forward with no tokens"
        case .unsupportedModel(let type):
            return "stepper: \(type) has no CBv2 forward path"
        case .recurrentLifecycle(let detail):
            return "stepper: recurrent state lifecycle (\(detail))"
        }
    }
}

/// The generic one-row stepper. Every fork runner uses it; a family only
/// supplies its model, its layer kinds, and its cache constructor.
///
/// Construction mirrors what `RunnerEngineAssembly` does for the engine: one
/// contiguous backend, one row of per-layer sequence state, the model's own
/// `newCacheV2` caches wrapped in a `CBv2LayerCacheBank`, and the steppable
/// adapter over the serving model. Recurrent families additionally carry a
/// `CBv2RecurrentRequestState`, bound and committed per forward exactly as
/// `EngineLoopV2` does.
public final class CBv2SingleRowStepper: TeacherForcedStepper {

    /// Top-k depth. bench-protocol's `TOP_LOGITS_K`.
    public static let topLogitsK = 8

    private let model: any LanguageModel
    private let adapter: CBv2SteppableLanguageModelAdapter
    private let layerKinds: [CBv2LayerKind]
    private let newCaches:
        (
            (_ layerIndex: Int, _ kind: CBv2LayerKind) throws -> any CBv2AttendingLayerCache
        ) throws -> [any CBv2AttendingLayerCache]
    private let kvBytesCapacity: Int
    private let maxLength: Int

    private var backend: CBv2ContiguousKVBackend?
    private var bank: CBv2LayerCacheBank?
    private var rowState: [CBv2SequenceKV?] = []
    private var recurrentState: CBv2RecurrentRequestState?

    public private(set) var forwards = 0

    public init(
        model: any LanguageModel,
        layerKinds: [CBv2LayerKind],
        newCaches:
            @escaping (
                (_ layerIndex: Int, _ kind: CBv2LayerKind) throws -> any CBv2AttendingLayerCache
            ) throws -> [any CBv2AttendingLayerCache],
        kvBytesCapacity: Int,
        maxLength: Int
    ) {
        self.model = model
        self.adapter = CBv2SteppableLanguageModelAdapter(model)
        self.layerKinds = layerKinds
        self.newCaches = newCaches
        self.kvBytesCapacity = kvBytesCapacity
        self.maxLength = maxLength
    }

    // MARK: - TeacherForcedStepper

    public func begin() throws {
        try releaseSession()
        let backend = CBv2ContiguousKVBackend(
            config: CBv2ContiguousBackendConfig(bytesCapacity: kvBytesCapacity))
        let caches = try newCaches { index, kind in
            CBv2LayerCache(layerIndex: index, kind: kind)
        }
        self.backend = backend
        self.bank = CBv2LayerCacheBank(caches: caches)
        self.rowState = try backend.makeSequenceState(
            layerKinds: layerKinds, promptLength: 0, maxLength: maxLength)
        if let spec = adapter.recurrentStateSpec {
            self.recurrentState = try CBv2RecurrentRequestState(spec: spec)
        } else {
            self.recurrentState = nil
        }
        self.forwards = 0
    }

    public func forward(_ tokens: [Int]) throws -> StepOutput {
        Self.stepOutput(lastPositionLogits: try forwardLogits(tokens))
    }

    /// The last-position logits row `[1, vocab]` of one forward. What
    /// `forward` reduces to its top-k; exposed for side-by-side diagnostics.
    public func forwardLogits(_ tokens: [Int]) throws -> MLXArray {
        guard let bank else { throw StepperError.notBegun }
        guard !tokens.isEmpty else { throw StepperError.emptyForward }

        let input = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
        let caches = bank.layerCaches(rowStates: [rowState])

        let logits: MLXArray
        if let recurrentState {
            let evaluation = try recurrentState.bind()
            let full = adapter.forward(
                tokens: input, caches: caches, recurrentState: [evaluation])
            let roots = try evaluation.evaluate()
            logits = full[0..., -1, 0...]
            asyncEval([logits] + roots + innerState(caches))
            try evaluation.commit()
        } else {
            let full = adapter.forward(tokens: input, caches: caches)
            logits = full[0..., -1, 0...]
            asyncEval([logits] + innerState(caches))
        }
        forwards += 1
        return logits
    }

    // MARK: - Internals

    /// The caches' offset chain and KV buffers, ridden on the forward's
    /// `asyncEval` so the lazy `+ L` advance cannot grow O(steps) — the same
    /// DAR-325 discipline `EngineLoopV2.eagerCacheInnerState` applies.
    private func innerState(_ caches: [CBv2AttendingLayerCache]) -> [MLXArray] {
        caches.flatMap { ($0 as? KVCache)?.innerState() ?? [] }
    }

    private func releaseSession() throws {
        if let recurrentState, !recurrentState.isReleased {
            do { try recurrentState.release() } catch {
                throw StepperError.recurrentLifecycle("\(error)")
            }
        }
        recurrentState = nil
        bank?.releaseBoundRows()
        if let backend, !rowState.isEmpty {
            backend.release(rowState)
        }
        rowState = []
        bank = nil
        backend = nil
    }

    /// Top-k plus argmax from a `[1, vocab]` last-position logit row.
    ///
    /// `argPartition` narrows the vocabulary to k candidates on device (one
    /// small readback rather than a whole 262k-entry row per timed step),
    /// and the ordering is then decided on the host so the tie-break is
    /// EXPLICIT: descending logit, ascending token id. Ties for the maximum
    /// are always inside the k candidates unless more than k tokens share
    /// the maximum, which no real head produces.
    static func stepOutput(lastPositionLogits: MLXArray) -> StepOutput {
        let row = lastPositionLogits.reshaped([-1]).asType(.float32)
        let vocabulary = row.dim(0)
        let k = min(topLogitsK, vocabulary)
        let ids: MLXArray
        if k == vocabulary {
            ids = MLXArray(Array(Int32(0) ..< Int32(vocabulary)))
        } else {
            ids = argPartition(row, kth: vocabulary - k, axis: -1)[(vocabulary - k)...]
        }
        let values = takeAlong(row, ids, axis: -1)
        let hostIDs = ids.asType(.int32).asArray(Int32.self)
        let hostValues = values.asArray(Float.self)

        var candidates = zip(hostIDs, hostValues).map { (token: Int($0), logit: Double($1)) }
        candidates.sort {
            $0.logit == $1.logit ? $0.token < $1.token : $0.logit > $1.logit
        }
        let margin =
            candidates.count >= 2 ? candidates[0].logit - candidates[1].logit : 0
        return StepOutput(
            argmax: candidates[0].token, topLogits: candidates, margin: margin)
    }
}
