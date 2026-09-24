// RecurrentStateV2.swift
//
// Request-owned recurrent state for hybrid CBv2 target models. Recurrent
// state is deliberately separate from attention KV: it has fixed per-request
// residency, model-specific tensor shapes, and transactional step lifecycle.

import Foundation
import MLX

/// Explicit model/backend feature gates. Attention-only models inherit the
/// historical all-enabled defaults; first-generation recurrent adapters opt
/// out of paths whose state semantics have not been proven.
public struct CBv2ModelCapabilities: Sendable, Equatable, Codable {
    public var supportsPrefixReuse: Bool
    /// Exact committed recurrent state paired with its attention KV.
    public var supportsRecurrentCheckpointReuse: Bool
    public var supportsPagedKV: Bool
    /// Models validated only on observed native types and dynamic segmented
    /// ownership must not enter the historical fixed-precision pool.
    public var requiresNativePagedKV: Bool
    public var supportsCompiledDecode: Bool
    public var supportsPackedPrefill: Bool
    public var supportsMTP: Bool
    /// The model's rectangular MTP forward stages compact recurrence inputs
    /// and can reconstruct a strict accepted prefix without retaining one
    /// full recurrent state per verify position.
    public var supportsCompactRecurrentMTPReplay: Bool

    public init(
        supportsPrefixReuse: Bool = true,
        supportsRecurrentCheckpointReuse: Bool = false,
        supportsPagedKV: Bool = true,
        requiresNativePagedKV: Bool = false,
        supportsCompiledDecode: Bool = true,
        supportsPackedPrefill: Bool = true,
        supportsMTP: Bool = true,
        supportsCompactRecurrentMTPReplay: Bool = false
    ) {
        self.supportsPrefixReuse = supportsPrefixReuse
        self.supportsRecurrentCheckpointReuse = supportsRecurrentCheckpointReuse
        self.supportsPagedKV = supportsPagedKV
        self.requiresNativePagedKV = requiresNativePagedKV
        self.supportsCompiledDecode = supportsCompiledDecode
        self.supportsPackedPrefill = supportsPackedPrefill
        self.supportsMTP = supportsMTP
        self.supportsCompactRecurrentMTPReplay = supportsCompactRecurrentMTPReplay
    }

    enum CodingKeys: String, CodingKey {
        case supportsPrefixReuse
        case supportsRecurrentCheckpointReuse
        case supportsPagedKV
        case requiresNativePagedKV
        case supportsCompiledDecode
        case supportsPackedPrefill
        case supportsMTP
        case supportsCompactRecurrentMTPReplay
    }

    /// The Engine Protocol v1 manifest wire shape
    /// (`RunnerManifest.canonicalJSON`) froze the SIX flags it declared, and
    /// a checked-in v1 manifest carries only those. Flags added after that
    /// freeze decode to their initializer defaults instead of making the
    /// pinned file undecodable. The six frozen flags stay mandatory.
    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = CBv2ModelCapabilities()
        supportsPrefixReuse = try values.decode(Bool.self, forKey: .supportsPrefixReuse)
        supportsPagedKV = try values.decode(Bool.self, forKey: .supportsPagedKV)
        supportsCompiledDecode = try values.decode(Bool.self, forKey: .supportsCompiledDecode)
        supportsPackedPrefill = try values.decode(Bool.self, forKey: .supportsPackedPrefill)
        supportsMTP = try values.decode(Bool.self, forKey: .supportsMTP)
        supportsCompactRecurrentMTPReplay = try values.decode(
            Bool.self, forKey: .supportsCompactRecurrentMTPReplay)
        supportsRecurrentCheckpointReuse =
            try values.decodeIfPresent(Bool.self, forKey: .supportsRecurrentCheckpointReuse)
            ?? defaults.supportsRecurrentCheckpointReuse
        requiresNativePagedKV =
            try values.decodeIfPresent(Bool.self, forKey: .requiresNativePagedKV)
            ?? defaults.requiresNativePagedKV
    }

    public static let attentionOnly = CBv2ModelCapabilities()
    public static let initialRecurrentTarget = CBv2ModelCapabilities(
        supportsPrefixReuse: false,
        supportsPagedKV: false,
        supportsCompiledDecode: false,
        supportsPackedPrefill: false,
        supportsMTP: false,
        supportsCompactRecurrentMTPReplay: false)
}

public protocol CBv2ModelCapabilityProviding {
    var cbv2Capabilities: CBv2ModelCapabilities { get }
}

/// Tensor shapes for one recurrent model layer. Shapes include the singleton
/// request row so `elements` is also the exact per-request element count.
public struct CBv2RecurrentLayerStateSpec: Sendable, Equatable {
    public let modelLayerIndex: Int
    public let convShape: [Int]
    public let convDType: DType
    public let ssmShape: [Int]
    public let ssmDType: DType

    public init(
        modelLayerIndex: Int,
        convShape: [Int], convDType: DType,
        ssmShape: [Int], ssmDType: DType = .float32
    ) {
        self.modelLayerIndex = modelLayerIndex
        self.convShape = convShape
        self.convDType = convDType
        self.ssmShape = ssmShape
        self.ssmDType = ssmDType
    }
}

public enum CBv2RecurrentStateError: Error, Equatable {
    case invalidSpec(String)
    case byteCountOverflow
    case lifecycleViolation(String)
}

/// Complete fixed-residency description for one request.
public struct CBv2RecurrentStateSpec: Sendable, Equatable {
    /// One committed generation plus the two pending generations retained by
    /// chained decode and serial MTP verification before finalization.
    public static let maximumLiveGenerations = 3

    public let layers: [CBv2RecurrentLayerStateSpec]

    public init(layers: [CBv2RecurrentLayerStateSpec]) {
        self.layers = layers
    }

    /// Exact fixed bytes for one live request. Every multiplication and sum
    /// is checked; malformed or hostile configurations fail instead of
    /// wrapping into an artificially small admission charge.
    public func fixedBytesPerRequest() throws -> Int {
        var total = 0
        var seen = Set<Int>()
        for layer in layers {
            guard seen.insert(layer.modelLayerIndex).inserted else {
                throw CBv2RecurrentStateError.invalidSpec(
                    "duplicate recurrent model layer \(layer.modelLayerIndex)")
            }
            let conv = try Self.byteCount(shape: layer.convShape, dtype: layer.convDType)
            let ssm = try Self.byteCount(shape: layer.ssmShape, dtype: layer.ssmDType)
            let (layerBytes, layerOverflow) = conv.addingReportingOverflow(ssm)
            let (newTotal, totalOverflow) = total.addingReportingOverflow(layerBytes)
            guard !layerOverflow, !totalOverflow else {
                throw CBv2RecurrentStateError.byteCountOverflow
            }
            total = newTotal
        }
        return total
    }

    public func peakBytesPerRequest() throws -> Int {
        let (bytes, overflow) = try fixedBytesPerRequest().multipliedReportingOverflow(
            by: Self.maximumLiveGenerations)
        guard !overflow else { throw CBv2RecurrentStateError.byteCountOverflow }
        return bytes
    }

    /// Per-generation allocator bound, including a request row that aliases
    /// a larger batched allocation. This runs only during engine construction.
    func allocationBytesPerGeneration(policy: AllocationFootprintPolicy) throws -> Int {
        _ = try fixedBytesPerRequest()
        var total = 0
        for layer in layers {
            for (shape, dtype) in [(layer.convShape, layer.convDType), (layer.ssmShape, layer.ssmDType)] {
                let logical = try Self.byteCount(shape: shape, dtype: dtype)
                guard let bytes = CBv2AuxiliaryAllocationProjection.maximumBytesPerUnit(logical, policy: policy) else {
                    throw CBv2RecurrentStateError.byteCountOverflow
                }
                let (next, overflow) = total.addingReportingOverflow(bytes)
                guard !overflow else { throw CBv2RecurrentStateError.byteCountOverflow }
                total = next
            }
        }
        return total
    }

    public var modelLayerIndices: [Int] { layers.map(\.modelLayerIndex) }

    private static func byteCount(shape: [Int], dtype: DType) throws -> Int {
        guard !shape.isEmpty, shape.allSatisfy({ $0 >= 0 }) else {
            throw CBv2RecurrentStateError.invalidSpec("recurrent shapes must be nonempty and nonnegative")
        }
        var elements = 1
        for dimension in shape {
            let (next, overflow) = elements.multipliedReportingOverflow(by: dimension)
            guard !overflow else { throw CBv2RecurrentStateError.byteCountOverflow }
            elements = next
        }
        let (bytes, overflow) = elements.multipliedReportingOverflow(by: dtype.size)
        guard !overflow else { throw CBv2RecurrentStateError.byteCountOverflow }
        return bytes
    }
}

public struct CBv2RecurrentLayerState {
    public let conv: MLXArray?
    public let ssm: MLXArray?

    public init(conv: MLXArray?, ssm: MLXArray?) {
        self.conv = conv
        self.ssm = ssm
    }
}

/// One layer's compact representation of a rectangular recurrent verify.
/// The replay closure is invoked only for a strict accepted prefix. Full
/// acceptance may lazily materialize an exact final state; rollback discards
/// both commit paths.
public struct CBv2RecurrentPrefixReplayStage {
    public let positions: Int
    public let finalState: CBv2RecurrentLayerState
    public let materializedByteCount: Int
    public let evaluationRoots: [MLXArray]
    /// Additional materialized roots retained only after a strict-prefix
    /// commit. They stay charged until a later recurrent generation commits.
    public let strictReplayRetainedByteCount: Int
    public let strictReplayRetainedRoots: [MLXArray]
    /// Roots needed by a lazy full-accept materialization until its successor
    /// evaluates and commits. The final state itself remains covered by the
    /// request's ordinary one-generation charge.
    public let fullAcceptanceRetainedByteCount: Int
    public let fullAcceptanceRetainedRoots: [MLXArray]
    /// Optional commit-time materialization for full acceptance. This keeps
    /// rejectable exact-tail copies out of the verify graph.
    fileprivate let fullAcceptance: (() -> CBv2RecurrentLayerState)?
    fileprivate let replay: (Int) -> CBv2RecurrentLayerState

    public init(
        positions: Int,
        finalConv: MLXArray,
        finalSSM: MLXArray,
        materializedByteCount: Int,
        evaluationRoots: [MLXArray],
        strictReplayRetainedByteCount: Int,
        strictReplayRetainedRoots: [MLXArray],
        fullAcceptanceRetainedByteCount: Int = 0,
        fullAcceptanceRetainedRoots: [MLXArray] = [],
        fullAcceptance: (() -> CBv2RecurrentLayerState)? = nil,
        replay: @escaping (Int) -> CBv2RecurrentLayerState
    ) throws {
        guard positions >= 2 else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "prefix replay requires at least two positions")
        }
        guard materializedByteCount >= 0,
              strictReplayRetainedByteCount >= 0,
              fullAcceptanceRetainedByteCount >= 0
        else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "prefix replay materialized bytes must be nonnegative")
        }
        self.positions = positions
        self.finalState = CBv2RecurrentLayerState(conv: finalConv, ssm: finalSSM)
        self.materializedByteCount = materializedByteCount
        self.evaluationRoots = evaluationRoots
        self.strictReplayRetainedByteCount = strictReplayRetainedByteCount
        self.strictReplayRetainedRoots = strictReplayRetainedRoots
        self.fullAcceptanceRetainedByteCount = fullAcceptanceRetainedByteCount
        self.fullAcceptanceRetainedRoots = fullAcceptanceRetainedRoots
        self.fullAcceptance = fullAcceptance
        self.replay = replay
    }
}

/// One request's recurrent state. Mutated only by the engine thread.
public final class CBv2RecurrentRequestState {
    public let spec: CBv2RecurrentStateSpec
    public let byteCount: Int
    public var materializedByteCount: Int {
        var total = committed.isEmpty ? 0 : byteCount
        let (retainedTotal, retainedOverflow) = total.addingReportingOverflow(
            committedTransitionRetainedByteCount)
        if retainedOverflow { return Int.max }
        total = retainedTotal
        for generation in pending {
            let generationBytes: Int
            if let replay = generation.prefixReplay {
                generationBytes = replay.values.reduce(0) { partial, stage in
                    let (sum, overflow) = partial.addingReportingOverflow(
                        stage.materializedByteCount)
                    return overflow ? Int.max : sum
                }
            } else {
                let generations = generation.capturedPositions ?? 1
                let (bytes, overflow) = byteCount.multipliedReportingOverflow(
                    by: generations)
                generationBytes = overflow ? Int.max : bytes
            }
            let (sum, overflow) = total.addingReportingOverflow(generationBytes)
            if overflow { return Int.max }
            total = sum
        }
        return total
    }

    private struct Generation {
        let id: UInt64
        let layers: [Int: CBv2RecurrentLayerState]
        /// Non-nil marks a CAPTURED verify-window generation: every layer's
        /// conv/ssm array carries a leading per-position axis of this length
        /// (`[S, ...]` instead of the spec's `[1, ...]`). Such a generation
        /// can only leave `pending` through `commit(keepPositions:)` (which
        /// selects one position) or `rollback()`.
        let capturedPositions: Int?
        /// Non-nil marks a compact replay verify generation. It is mutually
        /// exclusive with `capturedPositions`; `layers` contains its final
        /// full-window states.
        let prefixReplay: [Int: CBv2RecurrentPrefixReplayStage]?
    }

    private var committed: [Int: CBv2RecurrentLayerState] = [:]
    private var committedTransitionGeneration: UInt64?
    private var committedTransitionRetainedByteCount = 0
    private var committedTransitionRetainedRoots: [MLXArray] = []
    private var pending: [Generation] = []
    private var nextGeneration: UInt64 = 0
    private var bindingOpen = false
    public private(set) var isReleased = false

    public init(spec: CBv2RecurrentStateSpec) throws {
        self.spec = spec
        self.byteCount = try spec.fixedBytesPerRequest()
    }

    public init(
        spec: CBv2RecurrentStateSpec,
        adoptedCommitted layers: [Int: CBv2RecurrentLayerState]
    ) throws {
        self.spec = spec
        self.byteCount = try spec.fixedBytesPerRequest()
        var adopted: [Int: CBv2RecurrentLayerState] = [:]
        for layer in spec.layers {
            guard let state = layers[layer.modelLayerIndex] else {
                throw CBv2RecurrentStateError.invalidSpec(
                    "adopted recurrent state is missing layer \(layer.modelLayerIndex)")
            }
            guard let conv = state.conv, let ssm = state.ssm else {
                throw CBv2RecurrentStateError.invalidSpec(
                    "adopted recurrent layer \(layer.modelLayerIndex) is incomplete")
            }
            guard conv.shape == layer.convShape else {
                throw CBv2RecurrentStateError.invalidSpec(
                    "adopted conv shape \(conv.shape) != spec \(layer.convShape) "
                        + "at layer \(layer.modelLayerIndex)")
            }
            guard ssm.shape == layer.ssmShape, ssm.dtype == layer.ssmDType else {
                throw CBv2RecurrentStateError.invalidSpec(
                    "adopted ssm \(ssm.shape)/\(ssm.dtype) != spec "
                        + "\(layer.ssmShape)/\(layer.ssmDType) "
                        + "at layer \(layer.modelLayerIndex)")
            }
            adopted[layer.modelLayerIndex] = state
        }
        guard adopted.count == spec.layers.count else {
            throw CBv2RecurrentStateError.invalidSpec(
                "adopted recurrent state layer count mismatch")
        }
        self.committed = adopted
    }

    /// The latest CONFIRMED per-layer state, or nil when none exists.
    ///
    /// This is deliberately NOT `state(modelLayerIndex:)`, which returns
    /// the newest VISIBLE state — `pending.last` when a speculative
    /// generation exists (a chained-decode successor, a pipelined prefill
    /// successor, an unresolved MTP verify window). Those are positions
    /// the engine has not confirmed and may roll back, so a boundary
    /// snapshot must never come from one.
    ///
    /// **Caller contract:** which POSITION this state belongs to is the
    /// caller's knowledge, not this type's. `committed` advances exactly
    /// once per `commit()`, so a caller that commits a generation and then
    /// reads this — as `EngineLoopV2.finalize` does — sees the state after
    /// that generation's tokens, even when a successor is already pending.
    /// A caller that reads it at an arbitrary time learns only "the last
    /// confirmed state", which is why the prefix bank tracks the position
    /// itself in `CBv2RecurrentCaptureGeometry` rather than inferring it
    /// from the scheduler's optimistically advanced token cursor.
    public func confirmedStateSnapshot() -> [Int: CBv2RecurrentLayerState]? {
        guard !isReleased, !committed.isEmpty else { return nil }
        return committed
    }

    /// Bind the latest visible generation for a target forward. A chained
    /// decode therefore reads the previous still-pending generation without
    /// prematurely committing it.
    public func bind() throws -> CBv2RecurrentStateEvaluation {
        guard !isReleased else {
            throw CBv2RecurrentStateError.lifecycleViolation("bind after release")
        }
        guard !bindingOpen else {
            throw CBv2RecurrentStateError.lifecycleViolation("overlapping recurrent bindings")
        }
        let last = pending.last
        if let captured = last?.capturedPositions {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "bind over an unresolved captured verify window (\(captured) positions)")
        }
        if last?.prefixReplay != nil {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "bind over an unresolved compact prefix replay window")
        }
        bindingOpen = true
        let input = last?.layers ?? committed
        let evaluation = CBv2RecurrentStateEvaluation(
            owner: self, generation: nextGeneration, input: input,
            requiredLayers: Set(spec.modelLayerIndices))
        nextGeneration &+= 1
        return evaluation
    }

    fileprivate func evaluate(
        generation: UInt64, layers: [Int: CBv2RecurrentLayerState],
        capturedPositions: Int?,
        prefixReplay: [Int: CBv2RecurrentPrefixReplayStage]?
    ) throws {
        guard bindingOpen, !isReleased else {
            throw CBv2RecurrentStateError.lifecycleViolation("evaluate without an active binding")
        }
        guard capturedPositions == nil || prefixReplay == nil else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "cannot mix captured and compact replay generations")
        }
        pending.append(
            Generation(
                id: generation, layers: layers,
                capturedPositions: capturedPositions, prefixReplay: prefixReplay))
        bindingOpen = false
    }

    fileprivate func abandonBinding() {
        bindingOpen = false
    }

    fileprivate func commit(generation: UInt64, keepPositions: Int? = nil) throws {
        guard !isReleased, let first = pending.first, first.id == generation else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "recurrent commits must follow evaluation order")
        }
        func clearOlderTransitionRetention() {
            guard let retainedGeneration = committedTransitionGeneration,
                generation > retainedGeneration
            else { return }
            committedTransitionGeneration = nil
            committedTransitionRetainedByteCount = 0
            committedTransitionRetainedRoots.removeAll(keepingCapacity: false)
        }
        if let captured = first.capturedPositions {
            guard let keep = keepPositions, (1 ... captured).contains(keep) else {
                throw CBv2RecurrentStateError.lifecycleViolation(
                    "captured commit requires keepPositions in 1...\(captured) "
                        + "(got \(String(describing: keepPositions)))")
            }
            clearOlderTransitionRetention()
            committed = first.layers.mapValues { layer in
                CBv2RecurrentLayerState(
                    conv: layer.conv.map { $0[(keep - 1) ..< keep] },
                    ssm: layer.ssm.map { $0[(keep - 1) ..< keep] })
            }
            // MLX slices share their allocation: selecting one prefix does
            // not release the other positions in the captured backing stack.
            // Preserve that obligation through rollback of later work, until
            // a newer committed generation replaces these views or release.
            if captured > 1 {
                let (retained, overflow) = byteCount.multipliedReportingOverflow(by: captured - 1)
                committedTransitionGeneration = generation
                committedTransitionRetainedByteCount = overflow ? Int.max : retained
                committedTransitionRetainedRoots = first.layers.values.flatMap {
                    [$0.conv, $0.ssm].compactMap { $0 }
                }
            }
        } else if let replay = first.prefixReplay {
            guard let positions = replay.values.first?.positions,
                  let keep = keepPositions, (1 ... positions).contains(keep)
            else {
                throw CBv2RecurrentStateError.lifecycleViolation(
                    "prefix replay commit requires a valid keepPositions")
            }
            clearOlderTransitionRetention()
            let fullAcceptance = keep == positions
            if fullAcceptance {
                committed = replay.mapValues { stage in
                    stage.fullAcceptance?() ?? stage.finalState
                }
            } else {
                committed = replay.mapValues { $0.replay(keep) }
            }
            var retainedBytes = 0
            var retainedRoots: [MLXArray] = []
            for index in replay.keys.sorted() {
                guard let stage = replay[index] else { continue }
                let stageBytes = fullAcceptance
                    ? stage.fullAcceptanceRetainedByteCount
                    : stage.strictReplayRetainedByteCount
                let (sum, overflow) = retainedBytes.addingReportingOverflow(stageBytes)
                retainedBytes = overflow ? Int.max : sum
                retainedRoots.append(
                    contentsOf: fullAcceptance
                        ? stage.fullAcceptanceRetainedRoots
                        : stage.strictReplayRetainedRoots)
            }
            if retainedBytes > 0 || !retainedRoots.isEmpty {
                committedTransitionGeneration = generation
                committedTransitionRetainedByteCount = retainedBytes
                committedTransitionRetainedRoots = retainedRoots
            }
        } else {
            guard keepPositions == nil else {
                throw CBv2RecurrentStateError.lifecycleViolation(
                    "keepPositions is only valid for captured or prefix replay verify windows")
            }
            clearOlderTransitionRetention()
            committed = first.layers
        }
        pending.removeFirst()
    }

    fileprivate func rollback(generation: UInt64) throws {
        guard !isReleased, let last = pending.last, last.id == generation else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "recurrent rollback must target the newest pending generation")
        }
        pending.removeLast()
    }

    /// Test/integration inspection of the latest visible state.
    public func state(modelLayerIndex: Int) -> CBv2RecurrentLayerState? {
        pending.last?.layers[modelLayerIndex] ?? committed[modelLayerIndex]
    }

    /// The engine has stopped and synchronized a failed planned cohort. No
    /// forward/evaluation object may remain reachable when this is called.
    /// Committed state is released by the ordinary request retirement next.
    func discardPendingAfterSynchronization() {
        bindingOpen = false
        pending.removeAll()
    }

    public func release() throws {
        guard !bindingOpen, pending.isEmpty else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "release with recurrent work still in flight")
        }
        committed.removeAll(keepingCapacity: false)
        committedTransitionGeneration = nil
        committedTransitionRetainedByteCount = 0
        committedTransitionRetainedRoots.removeAll(keepingCapacity: false)
        isReleased = true
    }
}

/// A single forward's recurrent transaction. The model reads its row-local
/// input and stages one output for every declared recurrent layer. The engine
/// then calls `evaluate()` before submitting the arrays to MLX and later
/// chooses exactly one of `commit()` or `rollback()` at finalization.
public final class CBv2RecurrentStateEvaluation {
    private unowned let owner: CBv2RecurrentRequestState
    private let generation: UInt64
    private let input: [Int: CBv2RecurrentLayerState]
    private let requiredLayers: Set<Int>
    private var staged: [Int: CBv2RecurrentLayerState] = [:]
    private var stagedCapturedPositions: Int?
    private var stagedPrefixReplay: [Int: CBv2RecurrentPrefixReplayStage] = [:]
    private var evaluated = false
    private var finalized = false

    /// True when this transaction staged per-position captured stacks
    /// (MTP capture-verify) and must be finalized via
    /// `commit(keepPositions:)` or `rollback()`.
    public var isCaptured: Bool {
        stagedCapturedPositions != nil || !stagedPrefixReplay.isEmpty
    }

    fileprivate init(
        owner: CBv2RecurrentRequestState, generation: UInt64,
        input: [Int: CBv2RecurrentLayerState], requiredLayers: Set<Int>
    ) {
        self.owner = owner
        self.generation = generation
        self.input = input
        self.requiredLayers = requiredLayers
    }

    deinit {
        if !evaluated { owner.abandonBinding() }
    }

    public func inputState(modelLayerIndex: Int) -> CBv2RecurrentLayerState? {
        input[modelLayerIndex]
    }

    public func stage(modelLayerIndex: Int, conv: MLXArray, ssm: MLXArray) throws {
        guard stagedCapturedPositions == nil, stagedPrefixReplay.isEmpty else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "cannot mix captured, compact replay, and plain recurrent stages")
        }
        try stageRaw(modelLayerIndex: modelLayerIndex, conv: conv, ssm: ssm)
    }

    private func stageRaw(modelLayerIndex: Int, conv: MLXArray, ssm: MLXArray) throws {
        guard !evaluated, requiredLayers.contains(modelLayerIndex) else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "stage for undeclared or already-evaluated recurrent layer \(modelLayerIndex)")
        }
        guard staged[modelLayerIndex] == nil else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "duplicate recurrent stage for layer \(modelLayerIndex)")
        }
        staged[modelLayerIndex] = CBv2RecurrentLayerState(conv: conv, ssm: ssm)
    }

    /// Stage one layer's CAPTURED per-position states for an MTP verify
    /// window: `conv`/`ssm` carry a leading position axis of length
    /// `positions` (`[S, ...]`) where index `s` is the state after consuming
    /// window token `s`. Index `positions - 1` must equal the state a plain
    /// `stage` would have produced for the whole window. Mixing captured and
    /// plain stages in one transaction is a lifecycle violation.
    public func stageCaptured(
        modelLayerIndex: Int, conv: MLXArray, ssm: MLXArray, positions: Int
    ) throws {
        guard positions >= 1 else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "captured stage requires at least one position")
        }
        if let existing = stagedCapturedPositions, existing != positions {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "captured position count changed mid-transaction "
                    + "(\(existing) -> \(positions))")
        }
        if stagedCapturedPositions == nil,
           (!staged.isEmpty || !stagedPrefixReplay.isEmpty)
        {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "cannot mix captured, compact replay, and plain recurrent stages")
        }
        guard conv.dim(0) == positions, ssm.dim(0) == positions else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "captured stacks must carry the position axis first "
                    + "(conv \(conv.shape), ssm \(ssm.shape), positions \(positions))")
        }
        try stageRaw(modelLayerIndex: modelLayerIndex, conv: conv, ssm: ssm)
        stagedCapturedPositions = positions
    }

    /// Stage compact recurrence inputs for one layer. Every layer in the
    /// transaction must use this mode and the same verify width.
    public func stagePrefixReplay(
        modelLayerIndex: Int,
        positions: Int,
        finalConv: MLXArray,
        finalSSM: MLXArray,
        materializedByteCount: Int,
        evaluationRoots: [MLXArray],
        strictReplayRetainedByteCount: Int,
        strictReplayRetainedRoots: [MLXArray],
        fullAcceptanceRetainedByteCount: Int = 0,
        fullAcceptanceRetainedRoots: [MLXArray] = [],
        fullAcceptance: (() -> CBv2RecurrentLayerState)? = nil,
        replay: @escaping (Int) -> CBv2RecurrentLayerState
    ) throws {
        guard !evaluated, stagedCapturedPositions == nil, staged.isEmpty else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "cannot mix compact replay with captured or plain recurrent stages")
        }
        guard requiredLayers.contains(modelLayerIndex),
              stagedPrefixReplay[modelLayerIndex] == nil
        else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "compact replay stage for undeclared, duplicate, or evaluated layer "
                    + "\(modelLayerIndex)")
        }
        if let width = stagedPrefixReplay.values.first?.positions, width != positions {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "compact replay position count changed mid-transaction "
                    + "(\(width) -> \(positions))")
        }
        stagedPrefixReplay[modelLayerIndex] = try CBv2RecurrentPrefixReplayStage(
            positions: positions,
            finalConv: finalConv,
            finalSSM: finalSSM,
            materializedByteCount: materializedByteCount,
            evaluationRoots: evaluationRoots,
            strictReplayRetainedByteCount: strictReplayRetainedByteCount,
            strictReplayRetainedRoots: strictReplayRetainedRoots,
            fullAcceptanceRetainedByteCount: fullAcceptanceRetainedByteCount,
            fullAcceptanceRetainedRoots: fullAcceptanceRetainedRoots,
            fullAcceptance: fullAcceptance,
            replay: replay)
    }

    @discardableResult
    public func evaluate() throws -> [MLXArray] {
        let replayMode = !stagedPrefixReplay.isEmpty
        let stagedLayers =
            replayMode
            ? stagedPrefixReplay.mapValues(\.finalState)
            : staged
        guard !evaluated, Set(stagedLayers.keys) == requiredLayers else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "recurrent evaluation did not stage every declared layer")
        }
        try owner.evaluate(
            generation: generation, layers: stagedLayers,
            capturedPositions: stagedCapturedPositions,
            prefixReplay: replayMode ? stagedPrefixReplay : nil)
        evaluated = true
        var roots = stagedLayers.keys.sorted().flatMap { index in
            [stagedLayers[index]!.conv, stagedLayers[index]!.ssm].compactMap { $0 }
        }
        for index in stagedPrefixReplay.keys.sorted() {
            roots.append(contentsOf: stagedPrefixReplay[index]!.evaluationRoots)
        }
        return roots
    }

    public func commit() throws {
        guard evaluated, !finalized, stagedCapturedPositions == nil,
              stagedPrefixReplay.isEmpty
        else {
            throw CBv2RecurrentStateError.lifecycleViolation("invalid recurrent commit")
        }
        try owner.commit(generation: generation)
        finalized = true
        staged.removeAll(keepingCapacity: false)
    }

    /// Finalize a captured verify window by committing the state after
    /// `keepPositions` consumed window tokens (1-based; the accepted prefix
    /// length including the seed column).
    public func commit(keepPositions: Int) throws {
        guard evaluated, !finalized,
              stagedCapturedPositions != nil || !stagedPrefixReplay.isEmpty
        else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "captured commit on a non-captured recurrent transaction")
        }
        try owner.commit(generation: generation, keepPositions: keepPositions)
        finalized = true
        staged.removeAll(keepingCapacity: false)
        stagedPrefixReplay.removeAll(keepingCapacity: false)
    }

    public func rollback() throws {
        guard evaluated, !finalized else {
            throw CBv2RecurrentStateError.lifecycleViolation("invalid recurrent rollback")
        }
        try owner.rollback(generation: generation)
        finalized = true
        staged.removeAll(keepingCapacity: false)
        stagedPrefixReplay.removeAll(keepingCapacity: false)
    }
}

/// Model-level target seam used by `CBv2SteppableLanguageModelAdapter`.
public protocol CBv2RecurrentLanguageModelForwardable: CBv2ModelCapabilityProviding {
    var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec { get }
    func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray
}

public protocol CBv2PositionedRecurrentLanguageModelForwardable:
    CBv2RecurrentLanguageModelForwardable
{
    func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray
}

/// Engine-facing optional refinement. The optional spec keeps the generic
/// language-model adapter source-compatible for attention-only models.
public protocol CBv2RecurrentSteppableModel: CBv2SteppableModel, CBv2ModelCapabilityProviding {
    var recurrentStateSpec: CBv2RecurrentStateSpec? { get }
    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray
}

/// Optional recurrent forward with explicit request-owned model positions.
public protocol CBv2PositionedRecurrentSteppableModel:
    CBv2RecurrentSteppableModel, CBv2PositionedForwardingCapabilityProviding
{
    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray
}
