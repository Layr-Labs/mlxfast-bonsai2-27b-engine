// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Exact top-2 token ids and logit values for every row of `[1, rows, vocab]`.
///
/// The Qwen policy entry point keeps its existing shape and lazy reduction and
/// forwards to the shared CBv2 reduction. It is `public` because the Qwen 3.8
/// Flash-Next model files call it from an editable copy outside this fork.
public func qwen35MTPTopTwoRows(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
    cbv2TopTwoRows(logits)
}

// Moved verbatim from `Qwen35.swift` (per-file static review limit).
/// The verify window's producer chains on the int8 narrow route (16 rows):
/// the SwiGLU product, the GDN output's gated norm and the attention output
/// gate are formed in the quantizing rotation's read (the prompt route's
/// `..._q8p`) instead of by their own launches ahead of
/// `bonsai_signed_hadamard_1024_q8`. Per verify window of the 27B: 64
/// compiled SwiGLU launches, 48 norms and 48 compiled gated tails, and 16
/// compiled gates with the 32 copies that flattened their operands fewer.
///
/// Each kind is self-tested once per width and dtypes on the running GPU at
/// 16 rows, bit for bit against the composed ops production runs (the
/// compiled chain with the signs, then the pre-signed `forwardInt8`), on
/// operands laid out as production lays them out; a mismatch keeps the
/// composed path. `BONSAI_VERIFY_SWIGLU_Q8=0`, `BONSAI_VERIFY_GATED_NORM_Q8=0`
/// and `BONSAI_VERIFY_ATTN_GATE_Q8=0` keep it per kind.
enum Qwen35VerifyProducerQ8 {
    private static func on(_ name: String) -> Bool {
        let value = ProcessInfo.processInfo.environment[name]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }
    static let swiglu = on("BONSAI_VERIFY_SWIGLU_Q8")
    static let gatedNorm = on("BONSAI_VERIFY_GATED_NORM_Q8")
    static let attentionGate = on("BONSAI_VERIFY_ATTN_GATE_Q8")

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]

    /// `HadamardQuantizedLinear.narrowProducerApproves`: the kind's switch,
    /// the operand dtypes the composed chain computes in FP32, and the kind's
    /// verdict (its self-test runs on first use).
    static func approves(
        _ producer: SignedBlockHadamard.Int8Producer, _ transform: SignedBlockHadamard
    ) -> Bool {
        // The composed chains compared against carry the signs (the default fold).
        guard Qwen35FusedElementwise.foldsHadamardSigns, transform.blockSize == 1024
        else { return false }
        let key: String
        switch producer {
        case .swiglu(let gate, let up):
            guard swiglu, gate.dtype == up.dtype, [DType.float16, .float32].contains(gate.dtype)
            else { return false }
            key = "swiglu \(transform.width) \(gate.dtype)"
        case .gatedRMSNorm(let x, let gate, let weight, _):
            guard gatedNorm, x.dtype == .float32, x.ndim == 4, x.dim(3) == 128,
                [DType.float16, .float32].contains(gate.dtype), weight.dtype == .float32
            else { return false }
            key = "gated norm \(transform.width) \(gate.dtype)"
        case .sigmoidGate(let x, let gate):
            guard attentionGate, x.dtype == .float32, gate.dtype == .float32, x.ndim == 4
            else { return false }
            key = "attention gate \(transform.width) \(x.dim(2))x\(x.dim(3))"
        }
        lock.lock()
        defer { lock.unlock() }
        if let verdict = verdicts[key] { return verdict }
        let report = selfTest(producer, transform)
        verdicts[key] = report.passed
        FileHandle.standardError.write(
            ("bonsai verify producer q8 (\(key)): " + report.summary
                + (report.passed ? "; fused\n" : "; composed path kept\n")).data(using: .utf8)!)
        return report.passed
    }

    /// Sixteen rows with per-row scales from 0.05 to 30 (the sigmoid saturates
    /// both ways) and one zero row (all-zero groups, the norm's `rsqrt(eps)`),
    /// in production's layouts (SwiGLU's halves, the GDN gate and the attention
    /// gate are column slices of a stacked product, the attention output is
    /// head-transposed). Outputs compared as unsigned integers.
    private static func selfTest(
        _ producer: SignedBlockHadamard.Int8Producer, _ transform: SignedBlockHadamard
    ) -> Qwen35FusedBoundaryQ8.SelfTestReport {
        var report = Qwen35FusedBoundaryQ8.SelfTestReport()
        let rows = 16
        let width = transform.width
        let signs = transform.signVector
        do {
            try withError { error in
                for seed in [61, 62] {
                    let scale = MLXRandom.uniform(
                        Float(0.05) ..< Float(30), [1, rows, 1], key: MLXRandom.key(UInt64(seed)))
                        * (MLXArray(0 ..< rows) .!= MLXArray(Int32(rows / 3)))
                            .asType(.float32).reshaped(1, rows, 1)
                    func normal(_ n: Int, _ salt: Int) -> MLXArray {
                        MLXRandom.normal([1, rows, n], key: MLXRandom.key(UInt64(seed * 8 + salt)))
                            * scale
                    }
                    let signed: MLXArray
                    let fused: SignedBlockHadamard.Int8Producer
                    switch producer {
                    case .swiglu(let gate, _):
                        let halves = split(normal(2 * width, 1).asType(gate.dtype), parts: 2, axis: -1)
                        signed = Qwen35FusedElementwise.swigluSigned(halves[0], halves[1], signs)
                        fused = .swiglu(gate: halves[0], up: halves[1])
                    case .gatedRMSNorm(let x, let gate, let weight, let eps):
                        let shape = [1, rows, x.dim(2), x.dim(3)]
                        let out = normal(width, 2).reshaped(shape)
                        let z = split(normal(2 * width, 3).asType(gate.dtype), parts: 2, axis: -1)[1]
                            .reshaped(shape)
                        let normed = MLXFast.rmsNorm(out, weight: weight, eps: eps)
                        signed = Qwen35FusedElementwise.gatedNormTailSigned(
                            normed, z.asType(.float32), signs.reshaped(x.dim(2), x.dim(3)))
                        fused = .gatedRMSNorm(x: out, gate: z, weight: weight, eps: eps)
                    case .sigmoidGate(let x, _):
                        // The head-transposed attention output and the gate
                        // half of each q|gate head.
                        let (heads, dim) = (x.dim(2), x.dim(3))
                        let xs = (MLXRandom.normal(
                            [1, heads, rows, dim], key: MLXRandom.key(UInt64(seed * 8 + 4)))
                            * scale.reshaped(1, 1, rows, 1)).transposed(0, 2, 1, 3)
                        let gs = normal(2 * width, 5).reshaped(1, rows, heads, 2 * dim)
                            .split(parts: 2, axis: -1)[1]
                        signed = Qwen35FusedElementwise.sigmoidGateSigned(
                            xs.reshaped(1, rows, -1), gs.reshaped(1, rows, -1), signs)
                        fused = .sigmoidGate(x: xs, gate: gs)
                    }
                    guard
                        let a0 = transform.forwardInt8(
                            signed.reshaped(rows, width), gdnLayout: nil, preSigned: true,
                            groupSize: 128),
                        let a1 = transform.forwardInt8(
                            producer: fused, gdnLayout: nil, groupSize: 128)
                    else {
                        report.passed = false
                        report.error = "a quantizing rotation is not installed"
                        return
                    }
                    report.cases += 1
                    for (a, b) in [
                        (a0.codes, a1.codes), (a0.scales, a1.scales),
                        (a0.scaledSums, a1.scaledSums),
                    ] {
                        guard a.dtype == b.dtype, a.shape == b.shape else {
                            report.passed = false
                            report.error = "output \(b.dtype) \(b.shape) vs \(a.dtype) \(a.shape)"
                            return
                        }
                        let bits: DType = a.dtype == .float32 ? .uint32 : a.dtype
                        let differ = (a.view(dtype: bits) .!= b.view(dtype: bits))
                            .asType(.int32).sum()
                        eval(differ)
                        try error.check()
                        let count = Int(differ.item(Int32.self))
                        report.values += a.size
                        report.mismatches += count
                        if count != 0 { report.passed = false }
                    }
                }
            }
        } catch {
            report.passed = false
            report.error = "\(error)"
        }
        return report
    }
}

/// The GDN output's gated per-head RMSNorm and the output projection's
/// Hadamard signs at verify width as ONE launch. The composed path runs
/// `MLXFast.rmsNorm` over each 128-wide head (`rms_single_row`) and the
/// compiled `gatedNormTailSigned` (`(z * sigmoid(z)) * normed * signs`): two
/// launches per GDN layer. The arithmetic here is the int8 producer kernel's
/// PROD 3 read, which already matches that composed chain bit for bit: per
/// head, lane l squares elements 4l .. 4l + 3 in order, `simd_sum`,
/// `precise::rsqrt(acc / 128 + eps)`, `w[d] * (x * inv)`, then `(z *
/// sigmoid(z)) * xn` with MLX's `Sigmoid` and the signs. One simdgroup per
/// (row, head), as `rms_single_row`: the launch keeps the norm's parallelism
/// and the quantizing rotation stays its own launch. z is read through its
/// strides (a slice of the qkv|z product), so it is not copied first.
///
/// Before first use a self-test on the running GPU compares the output bit for
/// bit against the composed ops (FP32 and FP16 z, a strided z); a mismatch or
/// any MLX error keeps the composed path. `BONSAI_VERIFY_GATEDNORM=0` keeps it
/// too.
enum Qwen35GatedNormTail {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_VERIFY_GATEDNORM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let header = """
        // MLX `Sigmoid` (unary_ops.h), verbatim.
        METAL_FUNC float bgn_sigmoid(float x) {
          auto y = 1 / (1 + metal::exp(metal::abs(x)));
          return (x < 0) ? y : 1 - y;
        }
        // element (row, c) of a [B, L, heads, 128] view, through its strides
        inline int64_t bgn_row(const constant int* shape, const constant int64_t* st, uint row) {
          const uint L = uint(shape[1]);
          return int64_t(row / L) * st[0] + int64_t(row % L) * st[1];
        }
        inline int64_t bgn_col(const constant int64_t* st, uint c) {
          return int64_t(c / 128u) * st[2] + int64_t(c % 128u) * st[3];
        }

        """

    // grid (32 * rows * H, 1, 1), threadgroup (256, 1, 1): one simdgroup per
    // (row, head). Inputs: x float [B, L, H, 128], z float|half [B, L, H,
    // 128] (any strides), w float [128], eps float [1], signs float [H * 128].
    // Template: H, InZ. Output: out float [rows, H * 128].
    private static let source = """
        const uint gidx = thread_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const uint hr = gidx / 32u;
        const uint row = hr / uint(H);
        const uint head = hr % uint(H);
        const int64_t xrow = bgn_row(x_shape, x_strides, row);
        const int64_t zrow = bgn_row(z_shape, z_strides, row);
        const uint c0 = head * 128u + lane * 4u;
        float xv[4];
        float acc = 0.0f;
        #pragma clang loop unroll(full)
        for (int r = 0; r < 4; r++) {
          xv[r] = float(x[xrow + bgn_col(x_strides, c0 + uint(r))]);
          acc += xv[r] * xv[r];
        }
        acc = simd_sum(acc);
        const float inv = metal::precise::rsqrt(acc / float(128) + eps[0]);
        #pragma clang loop unroll(full)
        for (int r = 0; r < 4; r++) {
          const uint col = c0 + uint(r);
          const float bv = float(z[zrow + bgn_col(z_strides, col)]);
          const float xn = w[col % 128u] * (xv[r] * inv);
          const float v = (bv * bgn_sigmoid(bv)) * xn;
          out[size_t(row) * size_t(H * 128) + col] = v * signs[col];
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_gdn_gated_norm_signed",
        inputNames: ["x", "z", "w", "eps", "signs"],
        outputNames: ["out"],
        source: source,
        header: header,
        ensureRowContiguous: false)

    /// `gatedNormTailSigned(rmsNorm(x, weight, eps), z, signs)` flattened to
    /// `[rows, H * 128]`, or nil when it does not apply.
    static func apply(
        _ x: MLXArray, gate z: MLXArray, weight: MLXArray, eps: Float, signs: MLXArray
    ) -> MLXArray? {
        guard enabled, x.dtype == .float32, z.dtype == .float32 || z.dtype == .float16,
            x.ndim == 4, z.shape == x.shape, x.dim(3) == 128,
            (x.dim(0) * x.dim(1) * x.dim(2)) % 8 == 0,
            weight.dtype == .float32, weight.ndim == 1, weight.dim(0) == 128,
            signs.dtype == .float32, signs.size == x.dim(2) * 128,
            verified(eps: eps)
        else { return nil }
        return launch(x, z, weight: weight, eps: eps, signs: signs)
    }

    private static func launch(
        _ x: MLXArray, _ z: MLXArray, weight: MLXArray, eps: Float, signs: MLXArray
    ) -> MLXArray {
        let rows = x.dim(0) * x.dim(1)
        let heads = x.dim(2)
        return kernel(
            [x, z, weight, MLXArray([eps]), signs.reshaped(-1)],
            template: [("H", heads), ("InZ", z.dtype)],
            grid: (32 * rows * heads, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, heads * 128]], outputDTypes: [.float32])[0]
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdict: Bool?

    private static func verified(eps: Float) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if let verdict { return verdict }
        let report = selfTest(eps: eps)
        verdict = report.passed
        FileHandle.standardError.write(
            ("bonsai verify gated norm: " + report.summary
                + (report.passed ? "; fused\n" : "; composed path kept\n")).data(using: .utf8)!)
        return report.passed
    }

    static func selfTest(eps: Float) -> Qwen35FusedBoundaryQ8.SelfTestReport {
        var report = Qwen35FusedBoundaryQ8.SelfTestReport()
        do {
            try withError { error in
                let rows = 16
                let heads = 48
                let width = heads * 128
                let signs = which(
                    MLXRandom.uniform(Float(0) ..< Float(1), [width], key: MLXRandom.key(121))
                        .< Float(0.5), MLXArray(Float(-1)), MLXArray(Float(1)))
                let weight = MLXRandom.uniform(
                    Float(0.5) ..< Float(1.5), [128], key: MLXRandom.key(122))
                let scale = MLXRandom.uniform(
                    Float(0.01) ..< Float(20), [1, rows, heads, 1], key: MLXRandom.key(123))
                let x = MLXRandom.normal([1, rows, heads, 128], key: MLXRandom.key(124)) * scale
                // z as the qkv|z product's slice: a strided view.
                let wide = MLXRandom.normal([1, rows, 10240 + width], key: MLXRandom.key(125))
                    * Float(3)
                for zType in [DType.float32, .float16] {
                    let z = wide.asType(zType)[0..., 0..., 10240...].reshaped(1, rows, heads, 128)
                    let zw = z.dtype == x.dtype ? z : z.asType(x.dtype)
                    let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
                    let reference = Qwen35FusedElementwise.gatedNormTailSigned(
                        normed, zw, signs.reshaped(heads, 128)
                    ).asType(x.dtype).reshaped(rows, width)
                    let fused = launch(x, z, weight: weight, eps: eps, signs: signs)
                    report.cases += 1
                    guard fused.shape == reference.shape, fused.dtype == reference.dtype else {
                        report.passed = false
                        report.error = "output \(fused.dtype) \(fused.shape)"
                        return
                    }
                    let differ = (fused.view(dtype: .uint32) .!= reference.view(dtype: .uint32))
                        .asType(.int32).sum()
                    eval(differ)
                    try error.check()
                    let count = Int(differ.item(Int32.self))
                    report.values += fused.size
                    report.mismatches += count
                    if count != 0 { report.passed = false }
                }
            }
        } catch {
            report.passed = false
            report.error = "\(error)"
        }
        return report
    }
}
