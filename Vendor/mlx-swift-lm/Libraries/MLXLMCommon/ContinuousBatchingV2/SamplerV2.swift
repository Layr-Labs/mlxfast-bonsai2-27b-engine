// SamplerV2.swift
//
// Vectorized token selection for ContinuousBatchingV2 (workstream E).
// Consumes the transformed [B, vocab] logits produced by
// `LogitsPipelineV2.process` and returns one token per row [B].
//
//  - All-greedy fast path: a single `argMax` when every row is greedy
//    (temperature < 1e-5), matching the legacy vectorized greedy path
//    bit for bit.
//  - Mixed batches: Gumbel-max via the exponential-noise race
//    (`argmax(probs / e)`, e ~ Exp(1)) so there is no multinomial and no
//    host sync. The tail (softmax, keyed RNG, sampled argmax) runs on the
//    stochastic-row subset only and scatters back over the greedy argmax
//    base; all-stochastic batches run full-width with no merge.
//  - Per-row keyed RNG: each row's noise is generated from its own key,
//    derived ONLY from (seed, requestID, per-request step index). A row's
//    random stream therefore never depends on its batchmates or on when it
//    joined the batch — the batch-composition-invariance requirement
//    (research report 12, item 5). Reproducibility is best-effort under
//    batching, per the contract: fixed (seed, requestID) reproduces the
//    same stream across runs given the same per-row logits.
//
// Threading: `sample` builds graph nodes only (no eval, no `.item()`).
// `setRows` runs at batch-membership change; `commit` is O(B) host counter
// bookkeeping (no device work).

import Foundation
import MLX
import MLXRandom

public final class SamplerV2 {

    /// Temperatures below this are treated as greedy (matches vLLM and
    /// `LogitsPipelineV2.greedyEpsilon`).
    public static let greedyEpsilon: Float = LogitsPipelineV2.greedyEpsilon

    /// Engine-level fallback used for rows that did not supply a seed.
    /// Fixed at init so nil-seed rows are still batch-invariant within a
    /// process; across runs they are intentionally non-deterministic.
    private let fallbackSeed: UInt64

    private struct RowState {
        var id: CBv2RequestID
        var seed: UInt64
        var greedy: Bool
        /// Per-request decode step index (number of tokens sampled so far
        /// for this request). Keying noise to the per-request index — not
        /// the global engine step — is what keeps a row's stream
        /// independent of when it joined the batch.
        var step: UInt64
    }

    private var rows: [RowState] = []
    private var allGreedy = true
    /// Non-greedy rows only — the sole consumers of the stochastic tail
    /// (softmax + keyed RNG + sampled argmax). Fixed at membership change,
    /// so the step path never scans for them.
    private var sampledRows: [RowState] = []
    /// Ascending int32 row indices of `sampledRows`, present only for mixed
    /// batches (0 < S < B). Nil when all-greedy (tail fully skipped) or
    /// when every row is stochastic (full-width path needs no merge).
    private var sampledIndices: MLXArray?

    public init(fallbackSeed: UInt64? = nil) {
        self.fallbackSeed = fallbackSeed ?? UInt64.random(in: .min ... .max)
    }

    // MARK: Membership change

    /// Rebuild per-row sampling state. Row order must match the logits row
    /// order passed to `sample`. Mid-flight rows resume their step index
    /// from `outputTokens.count`, so re-vectorization after a membership
    /// change does not perturb their random streams.
    public func setRows(_ rows: [CBv2SamplerRow]) {
        self.rows = rows.map { row in
            RowState(
                id: row.id,
                seed: row.params.seed ?? fallbackSeed,
                greedy: row.params.temperature < Self.greedyEpsilon,
                step: UInt64(row.outputTokens.count)
            )
        }
        allGreedy = self.rows.allSatisfy(\.greedy)
        sampledRows = []
        var sampledIdx = [Int32]()
        for (i, row) in self.rows.enumerated() where !row.greedy {
            sampledRows.append(row)
            sampledIdx.append(Int32(i))
        }
        // Indices exist only when a proper subset needs the tail: all-greedy
        // returns before any tail work, and all-stochastic runs full-width
        // with no merge.
        sampledIndices =
            (!allGreedy && sampledRows.count < self.rows.count) ? MLXArray(sampledIdx) : nil
    }

    // MARK: Step path

    /// Select one token per row from transformed logits [B, vocab].
    /// Returns [B] int32. Pure graph construction — call `commit()` once
    /// per step afterwards to advance the per-row step indices.
    public func sample(from logits: MLXArray) -> MLXArray {
        precondition(
            logits.dim(0) == rows.count,
            "logits rows (\(logits.dim(0))) != configured rows (\(rows.count)) — call setRows")

        // Fast path: every row greedy ⇒ one argMax, bit-identical to the
        // legacy vectorized greedy decode.
        let greedyTokens = argMax(logits, axis: -1).asType(.int32)
        if allGreedy {
            return greedyTokens
        }

        // Mixed batch: exponential-race Gumbel-max. probs/e keeps masked
        // tokens (-inf logits → probability 0) unreachable, and argmax of
        // the ratio is an exact categorical draw.
        let vocab = logits.dim(-1)
        if sampledRows.count == rows.count {
            // No greedy rows: full-width stochastic path, no merge dispatch.
            let probs = softmax(logits, axis: -1)
            let noise = exponentialNoise(vocab: vocab)
            return argMax(probs / noise, axis: -1).asType(.int32)
        }
        // Proper subset: the tail (softmax, keyed RNG, sampled argmax) runs
        // on stochastic rows only; picks scatter back over the greedy argmax
        // base. Per-row math is row-independent, so both sides are bitwise
        // the full-width result.
        guard let sampledIndices else { return greedyTokens }
        let sub = take(logits, sampledIndices, axis: 0)
        let probs = softmax(sub, axis: -1)
        let noise = exponentialNoiseSubset(vocab: vocab)
        let sampledSub = argMax(probs / noise, axis: -1).asType(.int32)
        return putAlong(
            greedyTokens.expandedDimensions(axis: 1),
            sampledIndices.expandedDimensions(axis: 1),
            values: sampledSub.expandedDimensions(axis: 1),
            axis: 0
        ).squeezed(axis: 1)
    }

    /// Advance every row's per-request step index. Call exactly once per
    /// sampled step, after `sample` (alongside `LogitsPipelineV2.commit`).
    public func commit() {
        for i in rows.indices {
            rows[i].step &+= 1
        }
    }

    // MARK: - Per-row keyed noise

    /// Exp(1) noise [B, vocab]; row r is generated from key
    /// mix(seed_r, id_r, step_r) and stacked, so each row's draw is a pure
    /// function of that row's identity — never of batchmates. Greedy rows
    /// receive a constant placeholder (their sampled pick is discarded by
    /// the merge).
    private func exponentialNoise(vocab: Int) -> MLXArray {
        var perRow = [MLXArray]()
        perRow.reserveCapacity(rows.count)
        for row in rows {
            if row.greedy {
                perRow.append(MLXArray.full([1, vocab], values: MLXArray(Float(1))))
                continue
            }
            perRow.append(Self.expNoiseRow(seed: row.seed, id: row.id.raw, step: row.step, vocab: vocab))
        }
        return concatenated(perRow, axis: 0)
    }

    /// Exp(1) noise [S, vocab] over `sampledRows` only, with the EXACT same
    /// keying and uniform→Exp transform as `exponentialNoise`, so each
    /// stochastic row's draw is bitwise the full-width draw. No placeholders:
    /// every row here is sampled, and picks scatter back by index.
    private func exponentialNoiseSubset(vocab: Int) -> MLXArray {
        var perRow = [MLXArray]()
        perRow.reserveCapacity(sampledRows.count)
        for row in sampledRows {
            perRow.append(Self.expNoiseRow(seed: row.seed, id: row.id.raw, step: row.step, vocab: vocab))
        }
        return concatenated(perRow, axis: 0)
    }

    /// One Exp(1) noise row [1, vocab] from key mix(seed, id, step).
    /// Single definition of the keyed-RNG transform shared by the full-width
    /// and subset tails (the MTP mirror `verifyExponentialNoise` below keeps
    /// its own inline copy deliberately — it is the independent audit trail
    /// for the verify-path contract).
    private static func expNoiseRow(seed: UInt64, id: UInt64, step: UInt64, vocab: Int) -> MLXArray {
        let key = MLXRandom.key(Self.mix(seed: seed, id: id, step: step))
        let u = MLXRandom.uniform(
            low: Float(0), high: Float(1), [1, vocab], type: Float.self, key: key)
        // e = -log(1 - u) ~ Exp(1); u ∈ [0, 1) keeps the log argument in
        // (0, 1]. Clamp away e == 0 (u == 0) so probs/e never divides
        // by zero.
        return maximum(-log(1 - u), MLXArray(Float(1e-20)))
    }

    /// MTP verify pre-sampling noise: one Exp(1) row per (request, window
    /// position), keyed EXACTLY like the step path —
    /// `mix(seed ?? fallbackSeed, id, step)` with the same uniform→-log(1-u)
    /// transform and zero clamp — so a verify position's draw is bitwise the
    /// draw the ordinary path would have made at that per-request step.
    /// Greedy rows receive the same constant placeholder as `sample`.
    func verifyExponentialNoise(
        rows: [(seed: UInt64?, id: UInt64, step: UInt64, greedy: Bool)], vocab: Int
    ) -> MLXArray {
        var perRow = [MLXArray]()
        perRow.reserveCapacity(rows.count)
        for row in rows {
            if row.greedy {
                perRow.append(MLXArray.full([1, vocab], values: MLXArray(Float(1))))
                continue
            }
            let key = MLXRandom.key(
                Self.mix(seed: row.seed ?? fallbackSeed, id: row.id, step: row.step))
            let u = MLXRandom.uniform(
                low: Float(0), high: Float(1), [1, vocab], type: Float.self, key: key)
            let e = maximum(-log(1 - u), MLXArray(Float(1e-20)))
            perRow.append(e)
        }
        return concatenated(perRow, axis: 0)
    }

    /// SplitMix64-style mix of (seed, requestID, step) into one RNG key
    /// seed. Deterministic across processes (no `Hasher`), and a pure
    /// function of exactly the three inputs the contract allows.
    static func mix(seed: UInt64, id: UInt64, step: UInt64) -> UInt64 {
        func splitmix(_ value: UInt64) -> UInt64 {
            var z = value &+ 0x9E37_79B9_7F4A_7C15
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        return splitmix(splitmix(splitmix(seed) ^ id) ^ step)
    }

    // MARK: - Short-width warmup (speculative verify/block widths)

    /// Process-wide pin set for the short-width sampler probes: one entry
    /// per warmed vocab size, so repeated pipeline creations (tests,
    /// multi-engine processes) pay the probe cost exactly once. A member so
    /// the warmup lives and dies with the sampler file.
    private final class ShortWidthWarmPin: @unchecked Sendable {
        static let shared = ShortWidthWarmPin()
        private let lock = NSLock()
        private var warmed: Set<Int> = []

        func claim(_ vocabSize: Int) -> Bool {
            lock.withLock { warmed.insert(vocabSize).inserted }
        }
    }

    /// Compile + warm the sampler kernels over the speculative short widths
    /// the serial warm stepper never touches, before any timed window, with
    /// no behavior change.
    ///
    /// Warmup gap: the resident warm stepper is serial-only ([1, 1, vocab]
    /// single-token decodes), so the first MTP target-prefix verify
    /// ([1, S, vocab], S = 1 + draft, draft 1...7) and the first DFlash
    /// block verify (S = block size 2...17) compile their sampler-side Metal
    /// pipelines in-window, on the candidate leg only. The earlier
    /// speculative-shape probes covered the matmul side; the sampler side —
    /// greedy `argMax`, the `applyTopKTopPMinP` descending sort + cumsum,
    /// the stochastic softmax + keyed-noise + sampled-argmax tail, and the
    /// subset `take`/`putAlong` + `which` merges — stayed cold.
    ///
    /// For every width S in 2...17 this builds the exact verify-shaped probe
    /// ([1, S, vocab] flattened to [S, vocab], mirroring
    /// `CBv2DefaultSampler.mtpVerifySample`) through BOTH tails, evaluates
    /// (forcing compile + one warm execution), and discards the outputs: no
    /// sampler state is read or written, numerics are untouched, and the
    /// greedy contract (all-greedy single argmax, bit-identical) is
    /// unaffected — this adds entries to the Metal kernel cache only.
    ///
    /// Idempotent per `vocabSize` (process-wide pin set). Called from
    /// `LogitsPipelineV2.init` — pipeline creation is the vocab-known
    /// load/warm boundary reachable without touching the step path (which
    /// must stay graph-build-only) or any file outside the sampler pair.
    public static func warmShortWidths(vocabSize: Int) {
        precondition(vocabSize > 0, "vocabSize must be positive")
        guard ShortWidthWarmPin.shared.claim(vocabSize) else { return }
        let kEff = Int32(max(1, min(64, vocabSize)))
        for s in 2 ... 17 {
            // Verify-shaped probe: [1, S, vocab] flattened to [S, vocab],
            // exactly like mtpVerifySample's `reshaped([b * w, vocab])`.
            let flat = MLXArray.zeros([1, s, vocabSize], dtype: .float32)
                .reshaped([s, vocabSize]).asType(.float32)
            // Greedy tail: the all-greedy single-argmax fast path, in the
            // working dtype and in the raw model dtypes (the no-upcast fast
            // path passes those straight to argmax).
            let greedy = argMax(flat, axis: -1).asType(.int32)
            let greedyF16 = argMax(
                MLXArray.zeros([s, vocabSize], dtype: .float16), axis: -1
            ).asType(.int32)
            let greedyBF16 = argMax(
                MLXArray.zeros([s, vocabSize], dtype: .bfloat16), axis: -1
            ).asType(.int32)
            // Pipeline filter side: descending sort + cumsum top-k over the
            // short width (top-k-only row: topP/minP carry the disabled
            // sentinels, exactly like a temperature+top-k row's tensors).
            let filtered = LogitsPipelineV2.applyTopKTopPMinP(
                flat,
                topK: MLXArray(Array(repeating: kEff, count: s)).reshaped([s, 1]),
                topP: MLXArray(Array(repeating: Float(2.0), count: s)).reshaped([s, 1]),
                minP: MLXArray.zeros([s, 1], dtype: .float32))
            // Stochastic tail: softmax + keyed Exp(1) noise race + sampled
            // argmax, reusing the step path's own noise helper so the warmed
            // uniform→-log(1-u) kernels are bitwise the production ones.
            let probs = softmax(filtered, axis: -1)
            var noiseRows = [MLXArray]()
            noiseRows.reserveCapacity(s)
            for r in 0 ..< s {
                noiseRows.append(
                    expNoiseRow(seed: 0, id: UInt64(r), step: 0, vocab: vocabSize))
            }
            let noise = concatenated(noiseRows, axis: 0)
            let sampled = argMax(probs / noise, axis: -1).asType(.int32)
            // MTP verify merge: which() over the short width.
            let merged = which(
                MLXArray(Array(repeating: false, count: s)), greedy, sampled)
            // Step-path subset merge: take S rows of an [S+1, vocab] batch
            // and scatter back via putAlong.
            let base = argMax(
                MLXArray.zeros([s + 1, vocabSize], dtype: .float32), axis: -1
            ).asType(.int32)
            let takeIdx = MLXArray(Array(0 ..< Int32(s)))
            let gathered = take(base, takeIdx, axis: 0)
            let scattered = putAlong(
                base.expandedDimensions(axis: 1),
                takeIdx.expandedDimensions(axis: 1),
                values: gathered.expandedDimensions(axis: 1),
                axis: 0
            ).squeezed(axis: 1)
            eval(
                greedy, greedyF16, greedyBF16, filtered, probs, noise,
                sampled, merged, gathered, scattered)
        }
    }
}
