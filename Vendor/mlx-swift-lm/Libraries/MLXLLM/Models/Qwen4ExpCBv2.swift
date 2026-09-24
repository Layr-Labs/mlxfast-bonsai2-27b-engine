//
//  Qwen4ExpCBv2.swift
//  mlx-swift-lm
//
//  ContinuousBatchingV2 conformances for Qwen 3.8 Flash-Next ("qwen4_exp").
//
//  The legacy `LLMModel` / `KVCache` path in `Qwen4Exp.swift` is untouched.
//  This file adds the second, request-owned path: a compact attention-storage
//  layout for the 12 full-attention layers, a recurrent-state spec for the 36
//  gated-deltanet layers and for the PLE layer's own state, and the forwards
//  that drive them.
//
//  THREE THINGS ARE FAMILY-SPECIFIC HERE.
//
//   1. The QSA indexer keeps a second per-row tape beside the key-value tape.
//      `Qwen4ExpCBv2LayerCache` owns it and re-synchronizes it to the row's
//      own `absoluteOffset` before every append, so an engine-driven rollback
//      of the key-value tape moves the indexer tape with it.
//   2. The keep mask the indexer produces reaches attention through the
//      `keepMask` seam on `CBv2AttendingLayerCache.updateAndAttend`.
//   3. The PLE layer's short-convolution state and its n-gram token history
//      ride the recurrent-state spec under a SYNTHETIC layer index, past the
//      last real layer. Spec keys are opaque, so this carries the two extra
//      pieces of per-request state through machinery the engine already has
//      for commit, rollback and byte accounting. Read the indices from
//      `cbv2AuxiliaryStateLayerIndices`; do not rebuild the convention.
//
//  ONE ROW PER CALL. The QSA indexer scores a single tape, so this path
//  serves batch one and refuses wider batches by name. That matches the
//  single-stream regime this family is served under.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Layer cache with the indexer tape

/// A `CBv2LayerCache` plus the QSA indexer's raw key tape.
///
/// The tape holds one un-rotated, un-pooled indexer key per token, which is
/// what the indexer pools into blocks on each call. It is EXACT -- one row per
/// processed token, no reserve -- and it is truncated to the sequence row's
/// own `absoluteOffset` before every append. That single rule keeps it in step
/// with the key-value tape: the engine rolls a speculative round back by
/// calling `rollback` on the row, and the next append sees the shorter offset.
public final class Qwen4ExpCBv2LayerCache: CBv2AttendingLayerCache {

    private let base: CBv2LayerCache
    private var tapes: [ObjectIdentifier: MLXArray] = [:]

    public init(layerIndex: Int, kind: CBv2LayerKind) {
        precondition(
            kind.sharesKVWithLayer == nil, "Qwen4Exp has no cross-layer KV sharing")
        self.base = CBv2LayerCache(layerIndex: layerIndex, kind: kind)
    }

    public var layerIndex: Int { base.layerIndex }
    public var kind: CBv2LayerKind { base.kind }
    public var rows: [CBv2SequenceKV] { base.rows }
    public var positionOffsets: MLXArray { base.positionOffsets }

    public func setRows(_ rows: [CBv2SequenceKV]) {
        base.setRows(rows)
        // A row that left the batch keeps no tape: its storage may be
        // recycled and its object identity reused.
        let live = Set(rows.map(ObjectIdentifier.init))
        tapes = tapes.filter { live.contains($0.key) }
    }

    /// Length of the bound row's tape, for tests and diagnostics.
    public var indexerTapeLength: Int {
        guard let row = rows.first else { return 0 }
        return tapes[ObjectIdentifier(row)]?.dim(1) ?? 0
    }

    /// Append this step's raw indexer keys `[1, S, indexerHeadDim]` to the
    /// single bound row's tape and return the whole tape.
    ///
    /// Call BEFORE `updateAndAttend`: the truncation reads the row's
    /// pre-update `absoluteOffset`, which is where the tape must start.
    public func updateIndexerTape(keys: MLXArray) -> MLXArray {
        precondition(
            rows.count == 1, "Qwen4Exp QSA serves one row per call, got \(rows.count)")
        let row = rows[0]
        let identity = ObjectIdentifier(row)
        let committed = row.absoluteOffset
        var tape = tapes[identity]
        if let existing = tape, existing.dim(1) > committed {
            tape = committed == 0 ? nil : existing[0..., ..<committed, 0...]
        }
        let updated = tape.map { concatenated([$0, keys], axis: 1) } ?? keys
        tapes[identity] = updated
        return updated
    }

    public func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        base.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: sinks)
    }

    public func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?, keepMask: MLXArray?
    ) -> MLXArray {
        base.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: sinks,
            keepMask: keepMask)
    }

    public func attendBorrowing(
        source: CBv2AttendingLayerCache,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        preconditionFailure("Qwen4Exp has no KV-shared layers to borrow from")
    }
}

extension Qwen4ExpCBv2LayerCache: CBv2KeepMaskCapableCache {
    public var honorsKeepMask: Bool { true }
}

/// Rectangular MTP verification: while set, the wrapped cache attends one
/// query position at a time, so a verify window's attention is arithmetically
/// the serial decode path's. The keep mask carries one row per query and is
/// sliced per position by the serialized path.
extension Qwen4ExpCBv2LayerCache: CBv2MTPRectangularSerializing {
    public var mtpSerializesRectangularAttention: Bool {
        get { base.mtpSerializesRectangularAttention }
        set { base.mtpSerializesRectangularAttention = newValue }
    }
    public var mtpBatchesRectangularAttention: Bool {
        get { base.mtpBatchesRectangularAttention }
        set { base.mtpBatchesRectangularAttention = newValue }
    }
}

extension Qwen4ExpCBv2LayerCache: KVCache {
    public var offset: Int { base.offset }
    public var maxSize: Int? { base.maxSize }

    public func innerState() -> [MLXArray] {
        var arrays = base.innerState()
        for row in rows {
            if let tape = tapes[ObjectIdentifier(row)] { arrays.append(tape) }
        }
        return arrays
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        base.update(keys: keys, values: values)
    }

    public var state: [MLXArray] {
        get { base.state }
        set { base.state = newValue }
    }

    public var metaState: [String] {
        get { base.metaState }
        set { base.metaState = newValue }
    }

    public var isTrimmable: Bool { base.isTrimmable }

    @discardableResult
    public func trim(_ n: Int) -> Int { base.trim(n) }

    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        base.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache { base.copy() }
}

// MARK: - Configuration derivation

extension Qwen4ExpTextConfiguration {

    /// Model layer indices that run full attention.
    public var fullAttentionLayerIndices: [Int] {
        layerTypes.enumerated().compactMap { $0.element == "full_attention" ? $0.offset : nil }
    }

    /// Model layer indices that run the gated-deltanet recurrence.
    public var recurrentLayerIndices: [Int] {
        layerTypes.enumerated().compactMap { $0.element != "full_attention" ? $0.offset : nil }
    }

    /// Recurrent-state key of the `ordinal`-th PLE layer's own state.
    ///
    /// THE CONVENTION, IN ONE PLACE. Keys are `hiddenLayers + ordinal`, past
    /// the last real decoder layer, so they can never collide with a layer's
    /// own key. A runner reads ``cbv2AuxiliaryStateLayerIndices`` instead of
    /// rebuilding this.
    public func pleStateLayerIndex(ordinal: Int) -> Int { hiddenLayers + ordinal }

    /// Recurrent-state keys that carry state belonging to no decoder layer.
    /// Today that is one key per PLE layer, in `ple_layer_ids` order.
    public var cbv2AuxiliaryStateLayerIndices: [Int] {
        pleLayerIndices.indices.map { pleStateLayerIndex(ordinal: $0) }
    }

    /// Compact CBv2 attention-storage layout: one row per full-attention
    /// layer, with `modelLayerIndex` mapping it back to its decoder layer.
    /// The recurrent layers own no key-value tape and are absent by design.
    public var cbv2LayerKinds: [CBv2LayerKind] {
        fullAttentionLayerIndices.map { modelLayerIndex in
            CBv2LayerKind(
                attention: .full,
                headDim: headDim,
                kvHeads: kvHeads,
                queryHeads: attentionHeads,
                modelLayerIndex: modelLayerIndex)
        }
    }

    /// Request-owned recurrent state: one entry per gated-deltanet layer, plus
    /// one auxiliary entry per PLE layer.
    ///
    /// A PLE entry carries the short-convolution state in `conv` and the
    /// `ngramSize - 1` trailing token ids in `ssm`. The n-gram history is an
    /// integer slot, not a float state, and its dtype says so.
    public func cbv2RecurrentStateSpec(
        activationDType: DType = .bfloat16
    ) -> CBv2RecurrentStateSpec {
        let keyDim = linearNumKeyHeads * linearKeyHeadDim
        let valueDim = linearNumValueHeads * linearValueHeadDim
        let convDim = 2 * keyDim + valueDim
        var layers = recurrentLayerIndices.map { modelLayerIndex in
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: modelLayerIndex,
                convShape: [1, max(0, linearConvKernelDim - 1), convDim],
                convDType: activationDType,
                ssmShape: [1, linearNumValueHeads, linearValueHeadDim, linearKeyHeadDim],
                ssmDType: .float32)
        }
        for ordinal in pleLayerIndices.indices {
            layers.append(
                CBv2RecurrentLayerStateSpec(
                    modelLayerIndex: pleStateLayerIndex(ordinal: ordinal),
                    convShape: [
                        1, max(0, (pleConvKernelSize - 1) * ngramSize), hcCount * hiddenSize,
                    ],
                    convDType: activationDType,
                    ssmShape: [1, max(1, ngramSize - 1)],
                    ssmDType: .int32))
        }
        return CBv2RecurrentStateSpec(layers: layers)
    }

    /// Every claim is explicit, and every optimization stays off until it is
    /// proven for this family.
    public var cbv2Capabilities: CBv2ModelCapabilities {
        CBv2ModelCapabilities(
            supportsPrefixReuse: false,
            supportsPagedKV: false,
            supportsCompiledDecode: false,
            supportsPackedPrefill: false,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: false)
    }
}

// MARK: - QSA indexer, CBv2 form

extension Qwen4ExpQSAIndexer {

    /// Keep mask over the row's whole indexer tape, or nil while the visible
    /// context still fits the budget.
    ///
    /// Same computation as the legacy `callAsFunction`, reading the tape from
    /// the CBv2 layer cache instead of a `Qwen4ExpAttentionCache`, and taking
    /// query positions as an array so the row's absolute history is honored.
    ///
    /// - Returns: `[1, 1, S, kvLength]` boolean, true == attend.
    func cbv2KeepMask(
        _ x: MLXArray,
        rope: Qwen4ExpRotary,
        cache: Qwen4ExpCBv2LayerCache,
        positions: MLXArray
    ) -> MLXArray? {
        let B = x.dim(0)
        let S = x.dim(1)
        precondition(B == 1, "Qwen4Exp QSA serves one row per call, got batch \(B)")

        let qk = indexQKProj(x)
        let split = heads * headDim
        var q = qk[.ellipsis, ..<split].reshaped(B, S, heads, headDim)
        let rawK = cache.updateIndexerTape(
            keys: qk[.ellipsis, split...].reshaped(B, S, headDim))

        let kvLength = rawK.dim(1)
        if kvLength <= tokenBudget { return nil }

        let blocks = kvLength / compressRatio
        var pooled = rawK[0..., ..<(blocks * compressRatio), 0...]
            .reshaped(B, blocks, compressRatio, headDim)
        pooled = kLayerNorm(pooled.asType(.float32).mean(axis: 2).asType(rawK.dtype))

        // Block n holds the logical positions n * compressRatio and up.
        let blockStarts = MLXArray(Int32(0) ..< Int32(blocks)) * Int32(compressRatio)
        let (cosK, sinK) = rope.cosSin(blockStarts[.newAxis])
        pooled = qwen4ExpRopePartial(pooled, cos: cosK, sin: sinK)

        let qPos = positions.asType(.int32)
        let (cosQ, sinQ) = rope.cosSin(qPos)
        q = qLayerNorm(q)
        q = qwen4ExpRopePartial(
            q, cos: cosQ[0..., 0..., .newAxis, 0...], sin: sinQ[0..., 0..., .newAxis, 0...])

        var scores = qwen4ExpIndexerBlockScores(
            q: q.asType(.float32), pooled: pooled.asType(.float32))
        scores = maximum(scores, MLXArray(Float(0))).sum(axis: -1) / Foundation.sqrt(Float(headDim))

        // Integer block COUNT. See the legacy path for why floor division is
        // load-bearing on both uses below.
        let complete = maximum(qPos + Int32(1), MLXArray(Int32(0)))
            .floorDivide(Int32(compressRatio))
        let blockIds = MLXArray(Int32(0) ..< Int32(blocks))
        var visible = blockIds[.newAxis, .newAxis, 0...] .< complete[.ellipsis, .newAxis]
        visible = MLX.broadcast(visible, to: [B, S, blocks])
        scores = MLX.where(visible, scores, MLXArray(-Float.infinity))

        let k = Swift.min(blockTopK, blocks)
        let top = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        let picked = takeAlong(visible, top, axis: -1)

        let sentinel = MLXArray(Int32(blocks))
        let keepBlocks = putAlong(
            MLXArray.zeros([B, S, blocks + 1], dtype: .bool),
            MLX.where(picked, top, sentinel),
            values: MLXArray(true),
            axis: -1
        )[.ellipsis, ..<blocks]
        var keep = repeated(keepBlocks, count: compressRatio, axis: -1)
        let rest = kvLength - blocks * compressRatio
        if rest > 0 {
            keep = concatenated([keep, MLXArray.zeros([B, S, rest], dtype: .bool)], axis: -1)
        }

        // Each query also keeps the partial block it sits in. Tape column c
        // holds the token at absolute position c, so this comparison is in
        // tape coordinates and the query's own position bounds it.
        let ownStart = complete * Int32(compressRatio)
        let tokens = MLXArray(Int32(0) ..< Int32(kvLength))
        let own =
            (tokens[.newAxis, .newAxis, 0...] .>= ownStart[.ellipsis, .newAxis])
            & (tokens[.newAxis, .newAxis, 0...] .<= qPos[.ellipsis, .newAxis])

        return expandedDimensions(keep | own, axis: 1)
    }
}

// MARK: - Full attention, CBv2 form

extension Qwen4ExpAttention {
    func cbv2Forward(
        _ x: MLXArray,
        rope: Qwen4ExpRotary,
        cache: Qwen4ExpCBv2LayerCache,
        positions: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        precondition(B == 1, "Qwen4Exp CBv2 attention serves one row per call, got batch \(B)")

        let keepMask = indexer.cbv2KeepMask(x, rope: rope, cache: cache, positions: positions)

        let projected = qProj(x).reshaped(B, S, heads, -1).split(parts: 2, axis: -1)
        var queries = projected[0]
        let gate = projected[1].reshaped(B, S, -1)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, S, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, S, kvHeads, -1).transposed(0, 2, 1, 3)

        let (cos, sin) = rope.cosSin(positions)
        queries = qwen4ExpRopePartial(queries, cos: cos[0..., .newAxis], sin: sin[0..., .newAxis])
        keys = qwen4ExpRopePartial(keys, cos: cos[0..., .newAxis], sin: sin[0..., .newAxis])

        let out = cache.updateAndAttend(
            queries: queries, keys: keys, values: values,
            scale: scale, sinks: nil, keepMask: keepMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, S, -1)

        return oProj(out * sigmoid(gate))
    }
}

// MARK: - Gated deltanet, CBv2 form

extension Qwen4ExpGatedDeltaNet {
    func cbv2Forward(
        _ x: MLXArray,
        modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        precondition(recurrentState.count == B, "Qwen4Exp CBv2 recurrent row count mismatch")

        let mixedQKV = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, valueHeads, valueHeadDim)
        let b = inProjB(x)
        let a = inProjA(x)

        var convRows: [MLXArray] = []
        var ssmRows: [MLXArray] = []
        convRows.reserveCapacity(B)
        ssmRows.reserveCapacity(B)
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: modelLayerIndex)
            convRows.append(
                state?.conv
                    ?? MLXArray.zeros([1, convKernelSize - 1, convDim], dtype: x.dtype))
            ssmRows.append(
                state?.ssm
                    ?? MLXArray.zeros(
                        [1, valueHeads, valueHeadDim, keyHeadDim], dtype: .float32))
        }
        let convState = convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        let ssmState = ssmRows.count == 1 ? ssmRows[0] : concatenated(ssmRows, axis: 0)

        // No conv mask: a CBv2 row is never a padded batch member, so every
        // position in the rectangle is a real token of this request.
        let convInput = concatenated([convState, mixedQKV], axis: 1)
        let newConvState = contiguous(convInput[0..., (1 - convKernelSize)..., 0...])
        let convOut = silu(conv1d(convInput))
        let parts = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)

        var q = parts[0].reshaped(B, S, keyHeads, keyHeadDim)
        var k = parts[1].reshaped(B, S, keyHeads, keyHeadDim)
        let v = parts[2].reshaped(B, S, valueHeads, valueHeadDim)

        let invScale = Foundation.pow(Float(keyHeadDim), -0.5)
        q =
            MLXArray(invScale * invScale).asType(x.dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        k =
            MLXArray(invScale).asType(x.dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, newSsmState) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: ssmState, mask: nil)

        for (row, evaluation) in recurrentState.enumerated() {
            do {
                try evaluation.stage(
                    modelLayerIndex: modelLayerIndex,
                    conv: newConvState[row ..< row + 1],
                    ssm: newSsmState[row ..< row + 1])
            } catch {
                preconditionFailure(
                    "Qwen4Exp CBv2 recurrent stage failed at layer \(modelLayerIndex): \(error)")
            }
        }
        return outProj(norm(out, gate: z).reshaped(B, S, -1))
    }

    /// MTP capture-verify: the same layer over a window of `S` tokens, with
    /// the recurrent state after EVERY position staged (`stageCaptured`), so
    /// finalize can keep the accepted prefix on device.
    ///
    /// The convolution is causal, so one call over the padded window gives
    /// every position the output a single-token step would; the delta rule
    /// runs one position at a time, which is exactly the serial decode
    /// arithmetic, so a verify window and the same tokens fed one by one
    /// produce the same logits and the same states.
    func cbv2ForwardCaptured(
        _ x: MLXArray,
        modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        precondition(recurrentState.count == B, "Qwen4Exp CBv2 recurrent row count mismatch")
        precondition(S >= 1, "Qwen4Exp capture-verify window must be non-empty")

        let mixedQKV = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, valueHeads, valueHeadDim)
        let b = inProjB(x)
        let a = inProjA(x)

        var convRows: [MLXArray] = []
        var ssmRows: [MLXArray] = []
        convRows.reserveCapacity(B)
        ssmRows.reserveCapacity(B)
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: modelLayerIndex)
            convRows.append(
                state?.conv
                    ?? MLXArray.zeros([1, convKernelSize - 1, convDim], dtype: x.dtype))
            ssmRows.append(
                state?.ssm
                    ?? MLXArray.zeros(
                        [1, valueHeads, valueHeadDim, keyHeadDim], dtype: .float32))
        }
        let convState = convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        let ssmState = ssmRows.count == 1 ? ssmRows[0] : concatenated(ssmRows, axis: 0)

        let nKeep = convKernelSize - 1
        let convInput = concatenated([convState, mixedQKV], axis: 1)
        let convOut = silu(conv1d(convInput))
        let parts = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)

        var q = parts[0].reshaped(B, S, keyHeads, keyHeadDim)
        var k = parts[1].reshaped(B, S, keyHeads, keyHeadDim)
        let v = parts[2].reshaped(B, S, valueHeads, valueHeadDim)

        let invScale = Foundation.pow(Float(keyHeadDim), -0.5)
        q =
            MLXArray(invScale * invScale).asType(x.dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        k =
            MLXArray(invScale).asType(x.dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        var outs: [MLXArray] = []
        var ssmStates: [MLXArray] = []
        outs.reserveCapacity(S)
        ssmStates.reserveCapacity(S)
        var state = ssmState
        for s in 0 ..< S {
            let (stepOut, next) = gatedDeltaUpdate(
                q: q[0..., s ..< (s + 1)],
                k: k[0..., s ..< (s + 1)],
                v: v[0..., s ..< (s + 1)],
                a: a[0..., s ..< (s + 1)],
                b: b[0..., s ..< (s + 1)],
                aLog: aLog, dtBias: dtBias,
                state: state, mask: nil)
            outs.append(stepOut)
            ssmStates.append(next)
            state = next
        }

        for (row, evaluation) in recurrentState.enumerated() {
            // After consuming position s the retained conv tail is the
            // nKeep inputs ending at s: convInput[s + 1 ..< s + 1 + nKeep].
            let convStack = concatenated(
                (0 ..< S).map { s in
                    convInput[row ..< (row + 1), (s + 1) ..< (s + 1 + nKeep), 0...]
                }, axis: 0)
            let ssmStack = concatenated(ssmStates.map { $0[row ..< (row + 1)] }, axis: 0)
            do {
                try evaluation.stageCaptured(
                    modelLayerIndex: modelLayerIndex,
                    conv: convStack, ssm: ssmStack, positions: S)
            } catch {
                preconditionFailure(
                    "Qwen4Exp CBv2 captured stage failed at layer \(modelLayerIndex): \(error)")
            }
        }
        let out = outs.count == 1 ? outs[0] : concatenated(outs, axis: 1)
        return outProj(norm(out, gate: z).reshaped(B, S, -1))
    }
}

// MARK: - PLE layer, CBv2 form

extension Qwen4ExpPLELayer {

    /// Trailing token ids the n-gram hash carries between forwards.
    var cbv2ContextLength: Int { max(1, dilation - 1) }

    /// One PLE block over request-owned state.
    ///
    /// Reads the n-gram history and the short-convolution state from the
    /// auxiliary recurrent slot, stages both updated pieces back, and returns
    /// the block output. On a fresh request the history is filled with the
    /// tower's end-of-text id, which is what makes the hash agree with the
    /// legacy path.
    func cbv2Forward(
        _ hidden: MLXArray,
        ids: MLXArray,
        stateLayerIndex: Int,
        eosTokenId: Int,
        recurrentState: [CBv2RecurrentStateEvaluation],
        captureRecurrentWindow: Bool = false
    ) -> MLXArray {
        let B = hidden.dim(0)
        precondition(recurrentState.count == B, "Qwen4Exp CBv2 recurrent row count mismatch")
        let contextLength = cbv2ContextLength

        var contextRows: [MLXArray] = []
        var convRows: [MLXArray] = []
        contextRows.reserveCapacity(B)
        convRows.reserveCapacity(B)
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: stateLayerIndex)
            contextRows.append(
                state?.ssm
                    ?? MLXArray.full(
                        [1, contextLength], values: MLXArray(Int32(eosTokenId)), dtype: .int32))
            convRows.append(
                state?.conv
                    ?? MLXArray.zeros(
                        [1, shortConvStateLength, hiddenSize * hcCount], dtype: hidden.dtype))
        }
        let previousContext =
            (contextRows.count == 1 ? contextRows[0] : concatenated(contextRows, axis: 0))
            .asType(ids.dtype)
        let convState = convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)

        let block = cbv2Block(
            hidden, ids: ids, previousContext: previousContext, convState: convState)

        let history = concatenated([previousContext, ids], axis: 1)
        if captureRecurrentWindow {
            // MTP capture-verify: stage the state after EVERY window position.
            // Both states are windows over a causal sequence, so position s's
            // state is a slice: the conv tail ending at s, and the trailing
            // context ending at s. Same bytes a single-token step would keep.
            let S = ids.dim(1)
            let n = shortConvStateLength
            for (row, evaluation) in recurrentState.enumerated() {
                let convStack = concatenated(
                    (0 ..< S).map { s in
                        block.convWindow[row ..< (row + 1), (s + 1) ..< (s + 1 + n), 0...]
                    }, axis: 0)
                let contextStack = concatenated(
                    (0 ..< S).map { s in
                        history[row ..< (row + 1), (s + 1) ..< (s + 1 + contextLength)]
                    }, axis: 0
                ).asType(.int32)
                do {
                    try evaluation.stageCaptured(
                        modelLayerIndex: stateLayerIndex,
                        conv: convStack, ssm: contextStack, positions: S)
                } catch {
                    preconditionFailure(
                        "Qwen4Exp CBv2 PLE captured stage failed at slot \(stateLayerIndex): \(error)")
                }
            }
            return block.value
        }
        let newContext = history[0..., (-contextLength)...].asType(.int32)
        for (row, evaluation) in recurrentState.enumerated() {
            do {
                try evaluation.stage(
                    modelLayerIndex: stateLayerIndex,
                    conv: block.convState[row ..< row + 1],
                    ssm: newContext[row ..< row + 1])
            } catch {
                preconditionFailure(
                    "Qwen4Exp CBv2 PLE stage failed at slot \(stateLayerIndex): \(error)")
            }
        }
        return block.value
    }

    /// The block itself, with the convolution state passed in and out instead
    /// of read from a cache. Numerically identical to the legacy path.
    private func cbv2Block(
        _ hidden: MLXArray,
        ids: MLXArray,
        previousContext: MLXArray,
        convState: MLXArray
    ) -> (value: MLXArray, convState: MLXArray, convWindow: MLXArray) {
        let embedded = pleEmbedding(ids, previousContext: previousContext).asType(hidden.dtype)
        let keyFlat = normKey(keyProj(embedded))
        let key = keyFlat.reshaped(keyFlat.shape.dropLast() + [hcCount, hiddenSize])
        let value = valueProj(embedded)
        let queryFlat = normQuery(hidden)
        let query = queryFlat.reshaped(queryFlat.shape.dropLast() + [hcCount, hiddenSize])

        var gate = (key * query).sum(axis: -1, keepDims: true) / Foundation.sqrt(Float(hiddenSize))
        // The floor is a scalar IN THE GATE'S DTYPE. `MLXArray(Float(1e-6))` is
        // a float32 array, not a weak scalar, and `maximum` promotes: the gate,
        // the gated value and the PLE output become float32, and `stream + ple`
        // then carries the whole hyper stream in float32 for every layer after
        // this one. The reference (`gate.abs().clamp_min(1e-6)`) keeps the
        // model dtype.
        let floor = MLXArray(Float(1e-6), dtype: gate.dtype)
        gate = MLX.sqrt(maximum(MLX.abs(gate), floor)) * MLX.sign(gate)

        var gated = sigmoid(gate) * value[.ellipsis, .newAxis, 0...]
        gated = gated.reshaped(gated.shape.dropLast(2) + [-1])

        let normed = normConv(gated)
        let S = normed.dim(1)
        let n = shortConvStateLength
        let full = concatenated([convState, normed], axis: 1)
        let newConvState = contiguous(full[0..., (-n)..., 0...])
        let convolved = silu(conv1d(full[0..., (-(n + S))..., 0...]))
        return (gated + convolved, newConvState, full)
    }
}

// MARK: - Decoder layer and tower, CBv2 form

extension Qwen4ExpDecoderLayer {
    func cbv2Forward(
        _ hyper: MLXArray,
        rope: Qwen4ExpRotary,
        modelLayerIndex: Int,
        attentionCache: Qwen4ExpCBv2LayerCache?,
        pleStateLayerIndex: Int?,
        eosTokenId: Int,
        recurrentState: [CBv2RecurrentStateEvaluation],
        ids: MLXArray,
        positions: MLXArray,
        captureRecurrentWindow: Bool = false
    ) -> MLXArray {
        var stream = hyper

        if let ple, let pleStateLayerIndex {
            stream =
                stream
                + ple.cbv2Forward(
                    stream, ids: ids, stateLayerIndex: pleStateLayerIndex,
                    eosTokenId: eosTokenId, recurrentState: recurrentState,
                    captureRecurrentWindow: captureRecurrentWindow)
        }

        var (input, residual, inject) = attnHyperConnection.mixWithInject(stream)
        let attended: MLXArray
        if isLinear {
            precondition(attentionCache == nil, "Qwen4Exp recurrent layer received attention KV")
            if captureRecurrentWindow {
                attended = linearAttn!.cbv2ForwardCaptured(
                    input, modelLayerIndex: modelLayerIndex, recurrentState: recurrentState)
            } else {
                attended = linearAttn!.cbv2Forward(
                    input, modelLayerIndex: modelLayerIndex, recurrentState: recurrentState)
            }
        } else {
            guard let attentionCache else {
                preconditionFailure("Qwen4Exp full-attention layer is missing its CBv2 cache")
            }
            attended = selfAttn!.cbv2Forward(
                input, rope: rope, cache: attentionCache, positions: positions)
        }
        stream = qwen4ExpInject(residual: residual, output: attended, inject: inject)

        (input, residual, inject) = mlpHyperConnection.mixWithInject(stream)
        return qwen4ExpInject(residual: residual, output: mlp(input), inject: inject)
    }
}

extension Qwen4ExpTower {
    /// Both tower streams over request-owned state.
    ///
    /// `caches` is the COMPACT attention layout: one entry per full-attention
    /// layer, in model-layer order.
    func cbv2Streams(
        _ ids: MLXArray,
        inputEmbeddings: MLXArray?,
        caches: [Qwen4ExpCBv2LayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?,
        captureRecurrentWindow: Bool = false
    ) -> (mixed: MLXArray, multi: MLXArray) {
        precondition(
            caches.count == args.fullAttentionLayerIndices.count,
            "Qwen4Exp CBv2 needs one cache per full-attention layer")
        var hidden = inputEmbeddings ?? embedTokens(ids)
        let positions = cbv2Positions(
            positionIds: positionIds, caches: caches,
            batch: hidden.dim(0), length: hidden.dim(1))

        var attentionIndex = 0
        hidden = tiled(hidden, repetitions: [1, 1, args.hcCount])
        for (modelLayerIndex, layer) in layers.enumerated() {
            let attentionCache: Qwen4ExpCBv2LayerCache?
            if layer.isLinear {
                attentionCache = nil
            } else {
                attentionCache = caches[attentionIndex]
                precondition(
                    attentionCache!.kind.modelLayerIndex == modelLayerIndex,
                    "Qwen4Exp CBv2 attention cache mapped to the wrong model layer")
                attentionIndex += 1
            }
            hidden = layer.cbv2Forward(
                hidden,
                rope: rope,
                modelLayerIndex: modelLayerIndex,
                attentionCache: attentionCache,
                pleStateLayerIndex: pleLayerIndices.firstIndex(of: modelLayerIndex)
                    .map { args.pleStateLayerIndex(ordinal: $0) },
                eosTokenId: args.eosTokenId,
                recurrentState: recurrentState,
                ids: ids,
                positions: positions,
                captureRecurrentWindow: captureRecurrentWindow)
        }
        return (hyperConnectionMixer(hidden), hidden)
    }

    /// Absolute positions `[B, S]` for this forward.
    ///
    /// Request-owned position ids win when the engine supplies them. Without
    /// them the positions come from the bound row's own absolute offset, which
    /// is a host integer and costs no synchronization.
    private func cbv2Positions(
        positionIds: MLXArray?, caches: [Qwen4ExpCBv2LayerCache],
        batch: Int, length: Int
    ) -> MLXArray {
        if let positionIds {
            precondition(
                positionIds.ndim == 3 && positionIds.dim(0) >= 1,
                "Qwen4Exp CBv2 positions must be [axes, B, L]")
            return positionIds[0].asType(.int32)
        }
        precondition(batch == 1, "Qwen4Exp CBv2 serves one row per call, got batch \(batch)")
        guard let cache = caches.first, cache.rows.count == 1 else {
            preconditionFailure("Qwen4Exp CBv2 forward needs exactly one bound row")
        }
        let offset = cache.rows[0].absoluteOffset
        return MLXArray(Int32(offset) ..< Int32(offset + length))[.newAxis]
    }
}

// MARK: - Model conformances

extension Qwen4ExpModel {
    public var cbv2LayerKinds: [CBv2LayerKind] { configuration.cbv2LayerKinds }

    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        configuration.cbv2RecurrentStateSpec(activationDType: model.embedTokens.weight.dtype)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities { configuration.cbv2Capabilities }

    /// Recurrent-state keys that belong to no decoder layer. A runner reads
    /// this instead of rebuilding the convention.
    public var cbv2AuxiliaryStateLayerIndices: [Int] {
        configuration.cbv2AuxiliaryStateLayerIndices
    }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        // The vended cache is DISCARDED: this family owns its layer cache
        // because the QSA indexer keeps a second per-row tape beside the
        // key-value tape. The closure still runs so a caller that primes
        // per-layer state at build time still sees every layer.
        try cbv2LayerKinds.enumerated().map { storageIndex, kind in
            _ = try makeLayerCache(kind.modelLayerIndex ?? storageIndex, kind)
            return Qwen4ExpCBv2LayerCache(
                layerIndex: kind.modelLayerIndex ?? storageIndex, kind: kind)
        }
    }

    /// The caches this family requires, without a vending closure.
    public func newCacheV2() -> [any CBv2AttendingLayerCache] {
        cbv2LayerKinds.enumerated().map { storageIndex, kind in
            Qwen4ExpCBv2LayerCache(
                layerIndex: kind.modelLayerIndex ?? storageIndex, kind: kind)
        }
    }

    func cbv2Caches(_ caches: [KVCache]) -> [Qwen4ExpCBv2LayerCache] {
        caches.map { cache in
            guard let typed = cache as? Qwen4ExpCBv2LayerCache else {
                preconditionFailure(
                    """
                    Qwen4Exp CBv2 requires Qwen4ExpCBv2LayerCache: the QSA indexer \
                    keeps its own key tape and emits a keep mask. Got \(type(of: cache)).
                    """)
            }
            return typed
        }
    }

    /// Both CBv2 tower streams: `mixed` feeds the head, `multi` feeds the
    /// native MTP head.
    func cbv2Streams(
        _ inputs: MLXArray, inputEmbeddings: MLXArray?, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        captureRecurrentWindow: Bool = false
    ) -> (mixed: MLXArray, multi: MLXArray) {
        model.cbv2Streams(
            inputs, inputEmbeddings: inputEmbeddings, caches: cbv2Caches(caches),
            recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrentWindow: captureRecurrentWindow)
    }
}

extension Qwen4ExpModel: CBv2PositionAxisProviding {
    /// Text-only partial rope over one position per token.
    public var cbv2PositionAxisCount: Int? { 1 }
}

extension Qwen4ExpModel: CBv2KeepMaskRequiringModel {
    /// The QSA indexer is not an optimization: without the keep mask every
    /// full-attention layer attends densely and the model answers differently.
    public var cbv2RequiresKeepMask: Bool { true }
}

extension Qwen4ExpModel: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable
{
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        cbv2Forward(tokens, caches: caches, recurrentState: recurrentState, positionIds: nil)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        head(
            cbv2Streams(
                tokens, inputEmbeddings: nil, caches: caches,
                recurrentState: recurrentState, positionIds: positionIds
            ).mixed)
    }

    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { false }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        model.embedTokens(inputs)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        preconditionFailure("Qwen4Exp embedding prefill requires request-owned recurrent state")
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        head(
            cbv2Streams(
                inputs, inputEmbeddings: inputEmbedding, caches: cache ?? [],
                recurrentState: recurrentState, positionIds: positionIds
            ).mixed)
    }
}

extension Qwen4ExpModel: CBv2RecurrentLanguageModelPrefillForwardable {
    /// One row per call: the QSA indexer scores a single tape.
    public var cbv2SupportsPackedPrefill: Bool { false }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        let mixed = cbv2Streams(
            inputs, inputEmbeddings: inputEmbedding, caches: cache ?? [],
            recurrentState: recurrentState, positionIds: positionIds
        ).mixed
        switch requirement {
        case .evaluationOnly:
            // A small handle whose graph depends on the whole trunk: forcing
            // it commits every key-value write and every recurrent stage.
            return mixed[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            return head(mixed[0..., -1, 0...])
        }
    }
}

extension Qwen4ExpModel: CBv2RecurrentMTPForwardable {
    /// `lastHidden` is the PRE-final-mixer hyper-connection stream, which is
    /// `hc_count * hidden` wide. That is the tensor the native MTP head reads;
    /// the collapsed hidden would be the wrong tensor and the wrong width.
    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let streams = cbv2Streams(
            tokens, inputEmbeddings: nil, caches: caches,
            recurrentState: recurrentState, positionIds: positionIds)
        return (head(streams.mixed), streams.multi)
    }
}

extension Qwen4ExpModel: CBv2MTPPolicyTopTwoProviding {
    /// Exact top-two ids and values per verify position. Rectangular
    /// verification reads the target's argmax from `ids[..., 0]`; the kernel
    /// orders by value descending then lowest id, which is the drafter's
    /// `argMax` tie-break, so a draft is never rejected over a tie.
    public func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        precondition(logits.ndim == 3, "Qwen4Exp MTP policy logits must be [B,L,V]")
        let batch = logits.dim(0)
        let length = logits.dim(1)
        let vocabularySize = logits.dim(2)
        precondition(batch > 0 && length > 0 && vocabularySize >= 2)
        let topTwo = qwen35MTPTopTwoRows(logits.reshaped([1, batch * length, vocabularySize]))
        return (
            topTwo.ids.reshaped([batch, length, 2]),
            topTwo.values.reshaped([batch, length, 2]))
    }
}

extension Qwen4ExpModel: CBv2RecurrentCaptureMTPForwardable {
    /// MTP capture-verify: one forward over the whole verify window with the
    /// recurrent state after every position staged, so the engine verifies
    /// `1 + k` candidates in ONE target forward and commits the accepted
    /// prefix on device. Attention serialises per query for the round
    /// (`CBv2MTPRectangularSerializing`), so the arithmetic is the serial
    /// decode path's at every position.
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let streams = cbv2Streams(
            tokens, inputEmbeddings: nil, caches: caches,
            recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrentWindow: true)
        return (head(streams.mixed), streams.multi)
    }
}
