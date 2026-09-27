// CBv2VerifyValueProduct.swift
//
// The verify block's p·v as a split-K kernel pair (see the enum below).

import Foundation
import MLX
import MLXFast

/// `p·v` of a verify window's causal block (`CBv2PromptCausalAttention`,
/// `verify: true`) as a split-K kernel pair.
///
/// The block's composition hands the product to MLX's GEMM as a batch of
/// `kvHeads x repeats` products of `[L, kL] x [kL, D]` (16 x ~600 x 256 at
/// depth 15, 24 of them). On a device with NAX that is the regular NAX GEMM:
/// `D / 128` column tiles per product, so 48 threadgroups that each walk the
/// whole key axis for a 16-row tile of a 64-row pipeline, re-reading each KV
/// head's values once per query head that shares it. That is the shape of
/// the GDN a|b projection MLX ran as a handful of threadgroups (Subflatus3,
/// `Qwen35SmallNMatmul`), and FP32 there goes through the tensor unit's
/// reduced-precision path.
///
/// Here the key axis is split into `chunk`-wide pieces: one threadgroup per
/// (KV head, 32 value columns, key chunk), 256 threads, each thread holding
/// `rows / 8` rows (all query heads of the KV head, all block rows) of one
/// column in FP32, so each value element is read once per chunk and shared by
/// every query head of its KV head. A second kernel adds the chunks in chunk
/// order. The result is the FP32 product up to rounding (the partial sums
/// round differently from the GEMM's), checked at load against a CPU FP32
/// product; if that check fails the GEMM stays.
/// `BONSAI_VERIFY_SPLITK_PV=0` keeps MLX's GEMM.
package enum CBv2VerifyValueProduct {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_VERIFY_SPLITK_PV"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Keys per partial.
    static let chunk = 64

    // grid (D / 32 * 256, chunks, kvHeads), threadgroup (256, 1, 1). Thread t:
    // column cb * 32 + (t & 31), rows RPT * (t >> 5) ..< + RPT of the head's
    // `rows` (= repeats * L) probability rows. `v` is read through its strides
    // (the cache's row view, never copied); its last axis is contiguous.
    private static let partialSource = """
        const int kL = dims[0]; const int R = dims[1]; const int D = dims[3];
        const int H = dims[4];
        const int cb = int(threadgroup_position_in_grid.x);
        const int ch = int(threadgroup_position_in_grid.y);
        const int h = int(threadgroup_position_in_grid.z);
        const uint t = thread_position_in_threadgroup.x;
        const int col = cb * 32 + int(t & 31);
        const int r0 = int(t >> 5) * RPT;
        const int k0 = ch * KC;
        const int k1 = min(k0 + KC, kL);
        const device float* vp = v + (size_t)h * (size_t)v_strides[1] + col;
        const size_t vs = (size_t)v_strides[2];
        const device float* pp = p + ((size_t)h * R + r0) * (size_t)kL;
        float acc[RPT];
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPT; r++) { acc[r] = 0.0f; }
        for (int k = k0; k < k1; k++) {
          const float vv = vp[(size_t)k * vs];
          #pragma clang loop unroll(full)
          for (int r = 0; r < RPT; r++) { acc[r] = fma(pp[(size_t)r * kL + k], vv, acc[r]); }
        }
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPT; r++) {
          part[(((size_t)ch * H + h) * R + r0 + r) * D + col] = acc[r];
        }
        """

    private static let reduceSource = """
        const int NCH = dims[2]; const uint total = uint(dims[1] * dims[3] * dims[4]);
        const uint i = thread_position_in_grid.x;
        if (i >= total) { return; }
        float s = 0.0f;
        for (int c = 0; c < NCH; c++) { s += part[(size_t)c * total + i]; }
        out[i] = s;
        """

    private static let partialKernel = MLXFast.metalKernel(
        name: "bonsai_verify_pv_splitk_partial", inputNames: ["p", "v", "dims"],
        outputNames: ["part"], source: partialSource, ensureRowContiguous: false)
    private static let reduceKernel = MLXFast.metalKernel(
        name: "bonsai_verify_pv_splitk_reduce", inputNames: ["part", "dims"],
        outputNames: ["out"], source: reduceSource, ensureRowContiguous: true)

    /// `probabilities` `[1, H, R / L, L, kL]` (or `[1, H, L, kL]`),
    /// row-contiguous FP32; `values` `[1, H, kL, D]` FP32 with a contiguous
    /// last axis. Returns `[H * R, D]` in the GEMM's output order. No checks.
    private static func run(_ probabilities: MLXArray, _ values: MLXArray) -> MLXArray {
        let H = values.dim(1)
        let kL = values.dim(2)
        let D = values.dim(3)
        let R = probabilities.size / (H * kL)
        let chunks = (kL + chunk - 1) / chunk
        let dims = MLXArray([Int32(kL), Int32(R), Int32(chunks), Int32(D), Int32(H)])
        let part = partialKernel(
            [probabilities, values, dims], template: [("KC", chunk), ("RPT", R / 8)],
            grid: (D / 32 * 256, chunks, H), threadGroup: (256, 1, 1),
            outputShapes: [[chunks, H, R, D]], outputDTypes: [.float32])[0]
        let total = H * R * D
        return reduceKernel(
            [part, dims], template: [("KC", chunk)],
            grid: ((total + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[H * R, D]], outputDTypes: [.float32])[0]
    }

    private static func admits(_ probabilities: MLXArray, _ values: MLXArray) -> Bool {
        guard probabilities.dtype == .float32, values.dtype == .float32,
            values.ndim == 4, values.dim(0) == 1, probabilities.dim(0) == 1,
            probabilities.dim(1) == values.dim(1), probabilities.dim(-1) == values.dim(2),
            values.dim(3) % 32 == 0, values.dim(2) >= 1, values.strides[3] == 1
        else { return false }
        let rows = probabilities.size / (values.dim(1) * values.dim(2))
        return rows % 8 == 0 && rows / 8 <= 16
    }

    /// The load-time check: two KV heads of 96 rows over 530 keys (a partial
    /// last chunk) read through a strided view of a 600-row cache, against a
    /// CPU FP32 product. Host-built operands (no GPU random source needed).
    static let verdict: Bool = {
        let H = 2, rep = 6, L = 16, kL = 530, capacity = 600, D = 256
        var logits = [Float](repeating: 0, count: H * rep * L * kL)
        for i in logits.indices { logits[i] = Float(sin(Double(i) * 0.7131) * 3) }
        var cache = [Float](repeating: 0, count: H * capacity * D)
        for i in cache.indices { cache[i] = Float(cos(Double(i) * 0.3977) * 2) }
        let p = softmax(
            MLXArray(logits, [1, H, rep, L, kL]), axis: -1, precise: true, stream: .cpu)
        let v = MLXArray(cache, [1, H, capacity, D])[.ellipsis, ..<kL, 0...]
        eval(p, v)
        guard admits(p, v) else { return false }
        let y = run(p, v)
        let reference = matmul(p, v.expandedDimensions(axis: 2), stream: .cpu)
            .reshaped([H * rep * L, D])
        let diff = abs(y - reference, stream: .cpu).max(stream: .cpu).item(Float.self)
        let passed = diff.isFinite && diff <= 1e-4
        FileHandle.standardError.write(
            ("bonsai verify split-K p·v: max abs error \(diff) against CPU FP32"
                + (passed ? "; split-K\n" : "; GEMM kept\n")).data(using: .utf8)!)
        return passed
    }()

    /// `p·v` reshaped to `[1, H * repeats, L, D]` (the block's output), or nil
    /// when the kernel does not apply (the caller keeps the GEMM).
    static func apply(_ probabilities: MLXArray, values: MLXArray, queryHeads: Int, rows: Int)
        -> MLXArray?
    {
        guard enabled, admits(probabilities, values), verdict else { return nil }
        return run(probabilities, values).reshaped([1, queryHeads, rows, values.dim(3)])
    }
}
