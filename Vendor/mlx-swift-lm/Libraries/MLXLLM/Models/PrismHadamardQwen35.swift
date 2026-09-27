// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLMCommon

/// Text factory for Prism's folded Qwen3.8-27B pack.
///
/// The pack declares a vision tower. This track never serves an image, so the
/// load filter drops the 333 `vision_tower.*` tensors and the tower is never
/// built. `Qwen35Runner` declares `multimodal: false` for the same reason, so
/// the LLM factory is the only factory that resolves this model type.
///
/// The pack carries no MTP head (`mtp_num_hidden_layers: 0`). The track
/// attaches a separate, published head instead, so the speculative capability
/// stays as `Qwen35Model` declares it and the runner decides whether a head is
/// present. See `docs/bonsai2.md`.
public final class PrismHadamardQwen35TextModel: Qwen35Model, PrismHadamardLoading,
    WeightNameFiltering
{
    public let prismCheckpoint: PrismHadamardCheckpointConfiguration

    public init(configurationData: Data) throws {
        prismCheckpoint = try JSONDecoder().decode(PrismHadamardCheckpointConfiguration.self, from: configurationData)
        super.init(try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configurationData))
    }
    public func shouldLoadWeight(named name: String) -> Bool {
        !name.hasPrefix("vision_tower.")
    }
}

/// `x @ w.T` for a verify-width FP32 `x` (at most 16 rows) and a narrow FP32
/// `w` `[N, K]` (N a multiple of 32, K of 128): the GDN layers' stacked
/// `in_proj_b | in_proj_a` (N = 96, K = 5120).
///
/// MLX's GEMM gives this shape three threadgroups (N / 32 by one row tile)
/// that each walk all of K, so on the verify window it costs about as much
/// as a packed projection twenty times its size; it also runs FP32 through
/// the tensor unit's reduced-precision path (max error ~7e-4 of the output
/// range against a CPU FP32 product on random operands). Here K is split
/// into 128-wide chunks, one threadgroup per (32 columns, chunk), each thread
/// accumulating four rows of one column in FP32, and a second kernel adds the
/// chunks in chunk order. The result is the FP32 product (max error ~4e-4
/// absolute on outputs of magnitude ~260, i.e. rounding).
/// `DARKBLOOM_QWEN35_SPLITK_BA=0` keeps MLX's GEMM.
enum Qwen35SmallNMatmul {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN35_SPLITK_BA"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static let chunk = 128

    // grid (N / 32 * 128, K / 128, 1), threadgroup (128, 1, 1). The chunk's
    // x rows [M x 128] are staged in threadgroup memory; thread t takes column
    // nb + (t & 31) and k sub-range 32 * (t >> 5) .. + 31 for every row (eight
    // independent float4 weight loads), and the four sub-range sums of each
    // (row, column) are added in sub-range order.
    private static let partialSource = """
        const int K = dims[0]; const int M = dims[1]; const int N = dims[2];
        const int nb = int(threadgroup_position_in_grid.x) * 32;
        const int kc = int(threadgroup_position_in_grid.y);
        const int k0 = kc * 128;
        const uint t = thread_position_in_threadgroup.x;
        threadgroup float4 xs[16 * 32];
        threadgroup float red[4 * 16 * 33];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++) {
          const uint i = t + 128 * j;
          const int m = int(i >> 5); const int q = int(i & 31);
          xs[i] = m < M ? *(const device float4*)(x + (size_t)m * K + k0 + 4 * q) : float4(0.0f);
        }
        const int c = int(t & 31);
        const int s = int(t >> 5);
        const device float4* wp = (const device float4*)(w + (size_t)(nb + c) * K + k0 + 32 * s);
        float4 wv[8];
        #pragma clang loop unroll(full)
        for (int j = 0; j < 8; j++) { wv[j] = wp[j]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int m = 0; m < 16; m++) {
          float acc = 0.0f;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 8; j++) { acc += dot(xs[m * 32 + 8 * s + j], wv[j]); }
          red[(s * 16 + m) * 33 + c] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++) {
          const uint o = t + 128 * j;
          const int m = int(o >> 5); const int cc = int(o & 31);
          if (m < M) {
            const float v = ((red[(0 * 16 + m) * 33 + cc] + red[(1 * 16 + m) * 33 + cc])
                + red[(2 * 16 + m) * 33 + cc]) + red[(3 * 16 + m) * 33 + cc];
            part[((size_t)kc * M + m) * N + nb + cc] = v;
          }
        }
        """

    // One thread per output: the KS chunk partials loaded together (unrolled),
    // then added in chunk order.
    private static let reduceSource = """
        const int M = dims[1]; const int N = dims[2];
        const uint i = thread_position_in_grid.x;
        if (i >= uint(M * N)) { return; }
        float v[KS];
        #pragma clang loop unroll(full)
        for (int s = 0; s < KS; s++) { v[s] = part[(size_t)s * M * N + i]; }
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (int s = 0; s < KS; s++) { sum += v[s]; }
        out[i] = sum;
        """

    private static let partialKernel = MLXFast.metalKernel(
        name: "qwen35_splitk_partial", inputNames: ["x", "w", "dims"], outputNames: ["part"],
        source: partialSource, ensureRowContiguous: true)
    private static let reduceKernel = MLXFast.metalKernel(
        name: "qwen35_splitk_reduce", inputNames: ["part", "dims"], outputNames: ["out"],
        source: reduceSource, ensureRowContiguous: true)

    static func apply(_ x: MLXArray, _ w: MLXArray) -> MLXArray? {
        guard enabled, x.dtype == .float32, w.dtype == .float32, w.ndim == 2 else { return nil }
        let k = x.dim(-1)
        let n = w.dim(0)
        let rows = x.size / k
        guard rows >= 1, w.dim(1) == k, n % 32 == 0, k % chunk == 0 else { return nil }
        // Prompt width: the 64-row simdgroup-matrix split-K (`Qwen35WideNMatmul`).
        if rows > 16 { return Qwen35WideNMatmul.apply(x, w, rows: rows, k: k, n: n) }
        let dims = MLXArray([Int32(k), Int32(rows), Int32(n)])
        let part = partialKernel(
            [x.reshaped(rows, k), w, dims],
            grid: (n / 32 * 128, k / chunk, 1), threadGroup: (128, 1, 1),
            outputShapes: [[k / chunk, rows, n]], outputDTypes: [.float32])[0]
        let y = reduceKernel(
            [part, dims], template: [("KS", k / chunk)],
            grid: ((rows * n + 31) / 32 * 32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[rows, n]], outputDTypes: [.float32])[0]
        return y.reshaped(Array(x.shape.dropLast()) + [n])
    }
}
