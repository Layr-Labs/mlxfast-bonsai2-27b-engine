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

/// The prompt-width packed matmul on the tensor unit over the raw 2-bit codes
/// (MetalPerformancePrimitives `matmul2d`, `uint8 x uint2b_format -> int32`),
/// applying each 128-group's FP16 scale and offset in FP32:
/// `y = sum_g as[m,g] * (s[n,g] * (C[m,n,g] - 128 * colsum[n,g]) + b[n,g] * rs[m,g])`
/// where `C` is the integer product of the shifted codes `q + 128` with the
/// weight codes, `colsum` the per-group sum of the weight codes, `as` and
/// `rs` the activation's per-group scale and code sum. The weights are read
/// as stored; the scales, offsets and folded code sums are read through a
/// per-layer derived-constant cache. Installed into
/// `HadamardQuantizedLinear.tensorPackedMatmul` when the model loads;
/// `DARKBLOOM_BONSAI_TENSOR_ROUTE=0` keeps the dequantizing kernels.
enum Qwen35TensorPackedMatmul {
    private static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // The trailing newline matters: the JIT appends the kernel signature
    // directly after the header text.
    private static let header = """
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """

    // grid: (N / 64 * 128, M / 64, 1), threadgroup (128, 1, 1). Inputs: xq
    // uint8 [M, K], w uint32 [N, K / 16], scalesT / biasesT half [K / 128, N],
    // uT float [K / 128, N] (= -128 * s * colsum), ascale / rsb float [M, K /
    // 128], ksz int32 [K, M, N]. Template: OutT.
    private static let source = """
        const int K = ksz[0]; const int M = ksz[1]; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 64;
        const int m0 = int(threadgroup_position_in_grid.y) * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            64, 64, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
        tensor<device uint8_t, dextents<int, 2>, tensor_inline> A((device uint8_t*)xq, dextents<int, 2>(K, M));
        tensor<device uint2b_format, dextents<int, 2>, tensor_inline> B((device uchar*)w, dextents<int, 2>(K, N));
        auto tA0 = A.template slice<128, 64>(0, m0);
        auto tB0 = B.template slice<128, 64>(0, n0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB0)>, int32_t>();
        constexpr int CAP = 32;
        // Destination layout (the NAX fragment layout): element i ->
        //   n = n0 + 16 * (sg & 1) + fn + (i & 3) + 32 * ((i >> 3) & 1)
        //   m = m0 + 16 * (sg >> 1) + fm + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1)
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        const int nb = n0 + 16 * int(sg & 1) + fn;
        const int mb = m0 + 16 * int(sg >> 1) + fm;
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 32);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 32);
        const device float4* up0 = (const device float4*)(uT + nb);
        const device float4* up1 = (const device float4*)(uT + nb + 32);
        const int NQ = N / 4;
        const size_t mrow[4] = {(size_t)mb, (size_t)(mb + 8), (size_t)(mb + 32), (size_t)(mb + 40)};
        // Row-tiled constants: this lane's four rows are adjacent in the tile.
        const size_t tbase = (size_t)(m0 / 64) * (size_t)Kg * 64 + (size_t)((8 * int(sg >> 1) + fm) * 4);
        for (int g = 0; g < Kg; g++) {
          auto tA = A.template slice<128, 64>(g * 128, m0);
          auto tB = B.template slice<128, 64>(g * 128, n0);
          op.run(tA, tB, cT);
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = float4(bp0[g * NQ]), b1 = float4(bp1[g * NQ]);
          const float4 u0 = up0[g * NQ], u1 = up1[g * NQ];
          float as[4], rb[4];
          if (MPERM) {
            const float4 as4 = *(const device float4*)(ascale + tbase + (size_t)g * 64);
            const float4 rb4 = *(const device float4*)(rsb + tbase + (size_t)g * 64);
            as[0] = as4.x; as[1] = as4.y; as[2] = as4.z; as[3] = as4.w;
            rb[0] = rb4.x; rb[1] = rb4.y; rb[2] = rb4.z; rb[3] = rb4.w;
          } else {
            #pragma clang loop unroll(full)
            for (int q = 0; q < 4; q++) { as[q] = ascale[mrow[q] * Kg + g]; rb[q] = rsb[mrow[q] * Kg + g]; }
          }
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1;
            const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            const float u = nh ? u1[c] : u0[c];
            const float t = fma(s, float(cT[i]), u);
            acc[i] = fma(b, rb[mh], fma(as[mh], t, acc[i]));
          }
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) {
          const int c = i & 3; const int nh = (i >> 3) & 1;
          const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
          out[(size_t)mm * N + nb + c + 32 * nh] = OutT(acc[i]);
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_q8",
        inputNames: ["xq", "w", "scalesT", "biasesT", "uT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: source,
        header: header,
        ensureRowContiguous: true)

    // Verify width (`half x uint2b_format -> float`, 16 rows): each threadgroup
    // owns TN (64 or 32) output columns; its four simdgroups split K into four
    // contiguous quarters (one 16 x 32 x 128 op per 128-group) and their
    // partials are summed through threadgroup memory. grid: (N / 32 * 128, 1,
    // 1), threadgroup (128, 1, 1). Inputs: x half [16, K], w uint32 [N, K / 16],
    // scalesT / biasesT half [K / 128, N], rowsum float [16, K / 128], ksz.
    private static let sourceNarrow = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * TN;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int gper = Kg / SG;
        const int g0 = int(sg) * gper;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, TN, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device half, dextents<int, 2>, tensor_inline> A((device half*)x, dextents<int, 2>(K, M));
        tensor<device uint2b_format, dextents<int, 2>, tensor_inline> B((device uchar*)w, dextents<int, 2>(K, N));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto tB0 = B.template slice<128, TN>(0, n0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB0)>, float>();
        constexpr int CAP = TN / 2;
        // Destination layout: element i -> n = n0 + fn + (i & 3) + 16 * (i >> 3),
        // m = fm + 8 * ((i >> 2) & 1).
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        for (int g = g0; g < g0 + gper; g++) {
          auto tA = A.template slice<128, 16>(g * 128, 0);
          auto tB = B.template slice<128, TN>(g * 128, n0);
          op.run(tA, tB, cT);
          float4 sv[TN / 16], bv[TN / 16];
          #pragma clang loop unroll(full)
          for (int q = 0; q < TN / 16; q++) {
            sv[q] = float4(*(const device half4*)(scalesT + (size_t)g * N + n0 + fn + 16 * q));
            bv[q] = float4(*(const device half4*)(biasesT + (size_t)g * N + n0 + fn + 16 * q));
          }
          const float rs0 = rowsum[(size_t)fm * Kg + g];
          const float rs1 = rowsum[(size_t)(fm + 8) * Kg + g];
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
            acc[i] = fma(sv[nq][c], cT[i], fma(bv[nq][c], mh ? rs1 : rs0, acc[i]));
          }
        }
        threadgroup float red[SG - 1][CAP * 32];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { red[sg - 1][i * 32 + lane] = acc[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            float v = acc[i];
            #pragma clang loop unroll(full)
            for (int q = 0; q < SG - 1; q++) { v += red[q][i * 32 + lane]; }
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
            out[(size_t)(fm + 8 * mh) * N + n0 + fn + c + 16 * nq] = OutT(v);
          }
        }
        """

    private static let kernelNarrow = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16",
        inputNames: ["x", "w", "scalesT", "biasesT", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrow,
        header: header,
        ensureRowContiguous: true)

    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY=0` keeps the record's verify-width
    /// kernels (the few-row core and the split-K bodies) with the tensor route
    /// still on at prompt width.
    /// Columns per threadgroup at verify width: 32 (measured 1.5% faster on
    /// the decode window than 64, which the isolated kernel preferred);
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_TN=64` selects 64.
    private static let narrowTileColumns: Int = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_TN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return value == "64" ? 64 : 32
    }()

    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_SG8=0` keeps four simdgroups per
    /// threadgroup for every verify-width shape.
    private static let narrowDeepSplit: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_SG8"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let verifyEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // MARK: - The vocabulary head at verify width

    /// The vocabulary head (`n >= 65536`) at verify width (<= 16 rows): the
    /// core's few-row packed matmul body (`qmm_m16_block`, half operands
    /// dequantized straight into the tensor unit's right-operand fragment)
    /// with an FP32 output store, so the logits keep their FP32 accumulation
    /// instead of the core's FP16 rounding, and eight simdgroups per
    /// threadgroup (four 32-column blocks, each split in two over K). The
    /// head's rotated activation is read in FP16, the read every tower
    /// projection already takes at this width. Needs no packed operand
    /// format, so it runs on a toolchain without `uint2b_format`; probed once
    /// at init against the core's own product and declined on any error.
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_HEAD=0` keeps the core's dispatch.
    private static let headEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_HEAD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// 32-column blocks per threadgroup (`DARKBLOOM_BONSAI_TENSOR_ROUTE_HEAD_CB`,
    /// 1 to 8) and simdgroups splitting K per block (`..._HEAD_KS`, 2 or 4).
    private static let headColumnBlocks: Int = {
        let value = Int(ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_HEAD_CB"] ?? "") ?? 4
        return min(max(value, 1), 8)
    }()
    private static let headKSplit: Int = {
        let value = Int(ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_HEAD_KS"] ?? "") ?? 2
        return value == 4 ? 4 : 2
    }()

    private static let headHeader = """
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;

        // The core's few-row packed matmul (quantized_utils.h qmm_m16_block):
        // one 16 x 32 output block over the K partition [k0, k0 + Kp), KS
        // simdgroups splitting the 128-groups round-robin, weights read as
        // 16-byte lines and dequantized straight into the right-operand
        // fragment, partials summed through threadgroup memory. Same
        // arithmetic as the core's body; the store is OutT instead of T.
        template <typename T, int KS, typename OutT>
        METAL_FUNC void bonsai_head_block(
            const device uint32_t* w, const device T* scales, const device T* biases,
            const device T* x, device OutT* y, const int K, const int N, const int rows,
            const int col0, const int k0, const int Kp, const uint ks, const uint simd_lid,
            threadgroup float* red0, threadgroup float* red1) {
          constexpr int GS = 128;
          typedef vec<T, 8> frag_t;
          typedef vec<float, 8> cfrag_t;
          const int K_w = K / 16;
          const int K_g = K / GS;
          const short qid = simd_lid >> 2;
          const short fm = ((qid & 4) | ((simd_lid >> 1) & 3));
          const short fn = ((qid & 2) | (simd_lid & 1)) * 4;
          const ushort bsh = ushort(8 * (fn >> 2));
          int wrow[4];
          #pragma unroll
          for (int j = 0; j < 4; j++) { wrow[j] = min(col0 + int(fm) + 8 * j, N - 1); }
          const device T* xa0 = x + min(int(fm), rows - 1) * K + fn;
          const device T* xa1 = x + min(int(fm) + 8, rows - 1) * K + fn;
          constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
              16, 32, 16, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
          mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
          auto ct_a = gemm_op.template get_left_input_cooperative_tensor<T, T, float>();
          auto ct_b = gemm_op.template get_right_input_cooperative_tensor<T, T, float>();
          auto ct_c = gemm_op.template get_destination_cooperative_tensor<
              metal::remove_addrspace_t<decltype(ct_a)>, metal::remove_addrspace_t<decltype(ct_b)>, float>();
          #pragma unroll
          for (short i = 0; i < 16; i++) { ct_c[i] = 0.0f; }
          const int g_begin = k0 / GS;
          const int n_groups = Kp / GS;
          const ushort wq = ushort(fn >> 2);
          ushort qlane[4];
          #pragma unroll
          for (int st = 0; st < 4; st++) {
            qlane[st] = ushort((simd_lid & ~0x9u) | uint(st & 1) | (uint(st >> 1) << 3));
          }
          const int n_blocks_total = (n_groups + 1) / 2;
          const int my_blocks = max((n_blocks_total - int(ks) + KS - 1) / KS, 0);
          auto block_line = [&](int i, int j) -> uint4 {
            const int gb = g_begin + 2 * (int(ks) + i * KS);
            return *((const device uint4*)(w + wrow[j] * K_w + gb * 8) + wq);
          };
          uint4 ring[2][4];
          #pragma unroll
          for (int r = 0; r < 2; r++) {
            if (r < my_blocks) {
              #pragma unroll
              for (int j = 0; j < 4; j++) { ring[r][j] = block_line(r, j); }
            }
          }
          for (int i = 0; i < my_blocks; i++) {
            const int gb = g_begin + 2 * (int(ks) + i * KS);
            volatile int compiler_barrier;
            #pragma unroll
            for (int gh = 0; gh < 2; gh++) {
              const int g = gb + gh;
              if (g - g_begin >= n_groups) { break; }
              float s0[4]; float s1[4]; float s2[4]; float s3[4]; float b[4];
              #pragma unroll
              for (int j = 0; j < 4; j++) {
                const float s = float(scales[wrow[j] * K_g + g]);
                b[j] = float(biases[wrow[j] * K_g + g]);
                s0[j] = s; s1[j] = s * 0.25f; s2[j] = s * 0.0625f; s3[j] = s * 0.015625f;
              }
              #pragma unroll
              for (int st8 = 0; st8 < 8; st8++) {
                const int st = gh * 8 + st8;
                const int k = g * GS + st8 * 16;
                frag_t B0; frag_t B1;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                  const uint word = simd_shuffle(ring[0][j][st & 3], qlane[st >> 2]);
                  const uint by = (word >> bsh) & 0xffu;
                  const float v0 = s0[j] * float(by & 0x03u) + b[j];
                  const float v1 = s1[j] * float(by & 0x0cu) + b[j];
                  const float v2 = s2[j] * float(by & 0x30u) + b[j];
                  const float v3 = s3[j] * float(by & 0xc0u) + b[j];
                  if (j < 2) {
                    B0[4 * j + 0] = T(v0); B0[4 * j + 1] = T(v1); B0[4 * j + 2] = T(v2); B0[4 * j + 3] = T(v3);
                  } else {
                    B1[4 * (j - 2) + 0] = T(v0); B1[4 * (j - 2) + 1] = T(v1); B1[4 * (j - 2) + 2] = T(v2); B1[4 * (j - 2) + 3] = T(v3);
                  }
                }
                #pragma unroll
                for (int q = 0; q < 4; q++) { ct_a[q] = xa0[k + q]; ct_a[4 + q] = xa1[k + q]; }
                #pragma unroll
                for (short q = 0; q < 8; q++) { ct_b[q] = B0[q]; ct_b[8 + q] = B1[q]; }
                gemm_op.run(ct_a, ct_b, ct_c);
              }
            }
            (void)compiler_barrier;
            #pragma unroll
            for (int j = 0; j < 4; j++) { ring[0][j] = ring[1][j]; }
            if (i + 2 < my_blocks) {
              #pragma unroll
              for (int j = 0; j < 4; j++) { ring[1][j] = block_line(i + 2, j); }
            }
          }
          cfrag_t C0; cfrag_t C1;
          #pragma unroll
          for (short i = 0; i < 8; i++) { C0[i] = ct_c[i]; C1[i] = ct_c[8 + i]; }
          {
            threadgroup float* red = (ks & 2) ? red1 : red0;
            if (ks & 1) {
              #pragma unroll
              for (int i = 0; i < 8; i++) { red[i * 32 + simd_lid] = C0[i]; red[(8 + i) * 32 + simd_lid] = C1[i]; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!(ks & 1)) {
              #pragma unroll
              for (int i = 0; i < 8; i++) { C0[i] += red[i * 32 + simd_lid]; C1[i] += red[(8 + i) * 32 + simd_lid]; }
            }
            if (KS == 4) {
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (ks == 2) {
                #pragma unroll
                for (int i = 0; i < 8; i++) { red0[i * 32 + simd_lid] = C0[i]; red0[(8 + i) * 32 + simd_lid] = C1[i]; }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (ks == 0) {
                #pragma unroll
                for (int i = 0; i < 8; i++) { C0[i] += red0[i * 32 + simd_lid]; C1[i] += red0[(8 + i) * 32 + simd_lid]; }
              }
            }
          }
          if (ks != 0) { return; }
          #pragma unroll
          for (int i = 0; i < 8; i++) {
            const int v = int(fm) + (i / 4) * 8;
            const int c = col0 + int(fn) + (i % 4);
            if (v < rows) {
              if (c < N) { y[v * N + c] = static_cast<OutT>(C0[i]); }
              if (c + 16 < N) { y[v * N + c + 16] = static_cast<OutT>(C1[i]); }
            }
          }
        }

        """

    // grid: (ceil(N / 32 / CB) * 32 * CB * KS, 1, 1), threadgroup (32 * CB * KS,
    // 1, 1): CB 32-column blocks per threadgroup, KS simdgroups splitting K per
    // block. Inputs: x half [16, K], w uint32 [N, K / 16], scales / biases half
    // [N, K / 128], ksz int32 [K, M, N]. Template: OutT, KS, CB.
    private static let sourceHead = """
        const int K = ksz[0]; const int N = ksz[2];
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint cb = sg / KS;
        const uint ks = sg % KS;
        const int col0 = (int(threadgroup_position_in_grid.x) * CB + int(cb)) * 32;
        threadgroup float red[CB][2][16 * 32];
        if (col0 >= N) { return; }
        bonsai_head_block<half, KS, OutT>(w, scales, biases, x, out, K, N, 16, col0, 0, K, ks, lane, red[cb][0], red[cb][1]);
        """

    private static let kernelHead = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_head",
        inputNames: ["x", "w", "scales", "biases", "ksz"],
        outputNames: ["out"],
        source: sourceHead,
        header: headHeader,
        ensureRowContiguous: true)

    private static func runHead(
        _ x: MLXArray, _ weight: MLXArray, _ scales: MLXArray, _ biases: MLXArray,
        k: Int, n: Int, outputDType: DType
    ) -> MLXArray {
        let cb = headColumnBlocks
        let ks = headKSplit
        let blocks = (n / 32 + cb - 1) / cb
        return kernelHead(
            [x, weight, scales, biases, dimsArray(k: k, m: 16, n: n)],
            template: [("OutT", outputDType), ("KS", ks), ("CB", cb)],
            grid: (blocks * 32 * cb * ks, 1, 1), threadGroup: (32 * cb * ks, 1, 1),
            outputShapes: [[16, n]], outputDTypes: [outputDType])[0]
    }

    /// Compiles and runs the head kernel on a small random product and checks
    /// it against the core's own matmul; false on a JIT error or a mismatch.
    nonisolated(unsafe) private static var headAnnounced = false

    static let headAvailable: Bool = {
        guard headEnabled, support != .none else { return false }
        // `DARKBLOOM_BONSAI_TENSOR_ROUTE_HEAD_PROBE_FAIL=1` compiles a broken
        // kernel in place of the probe, to exercise the decline path.
        if ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_HEAD_PROBE_FAIL"] == "1" {
            return probe("bonsai_probe_head_broken", "this is not a kernel;")
        }
        probeFailed = false
        var ok = false
        withErrorHandler({ _ in Qwen35TensorPackedMatmul.probeFailed = true }) {
            let k = 512, n = 256
            let x = MLXRandom.normal([16, k], key: MLXRandom.key(11)).asType(.float16)
            let w = MLXRandom.randInt(low: 0, high: 1 << 30, [n, k / 16], key: MLXRandom.key(12)).asType(.uint32)
            let s = MLXRandom.uniform(low: 0.002, high: 0.02, [n, k / 128], key: MLXRandom.key(13)).asType(.float16)
            let b = MLXRandom.uniform(low: -0.02, high: 0.0, [n, k / 128], key: MLXRandom.key(14)).asType(.float16)
            let reference = quantizedMM(
                x.asType(.float32), w, scales: s.asType(.float32), biases: b.asType(.float32),
                transpose: true, groupSize: 128, bits: 2)
            let scale = abs(reference).max().item(Float.self)
            // Both output forms the routes take, so the timed window compiles
            // nothing: FP32 logits for the target's read, FP16 for the drafter's.
            let y32 = runHead(x, w, s, b, k: k, n: n, outputDType: .float32)
            let y16 = runHead(x, w, s, b, k: k, n: n, outputDType: .float16)
            let error32 = abs(y32 - reference).max().item(Float.self)
            let error16 = abs(y16.asType(.float32) - reference).max().item(Float.self)
            ok = error32.isFinite && error32 <= 1e-2 * max(scale, 1e-3)
                && error16.isFinite && error16 <= 2e-2 * max(scale, 1e-3)
        }
        return ok && !probeFailed
    }()

    // MARK: - Packed-operand support on this box's Metal toolchain

    /// How the tensor unit reads the 2-bit codes: `native2b` as a
    /// `uint2b_format` tensor over the stored bytes; `staged4b` on a toolchain
    /// without the 2-bit format (the ranked box's, September 2026): the same
    /// bytes expanded to `uint4b_format` in threadgroup memory per 128-group;
    /// `none` when neither compiles (every route then stays off).
    enum PackedOperandSupport { case native2b, staged8, staged4b, none }

    private static let probeHeader = header

    private static let probeSourceNative = """
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device half, dextents<int, 2>, tensor_inline> A((device half*)a, dextents<int, 2>(128, 16));
        tensor<device uint2b_format, dextents<int, 2>, tensor_inline> B((device uchar*)w, dextents<int, 2>(128, 32));
        auto tA = A.template slice<128, 16>(0, 0);
        auto tB = B.template slice<128, 32>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>, metal::remove_addrspace_t<decltype(tB)>, float>();
        op.run(tA, tB, cT);
        out[thread_position_in_grid.x] = cT[0];
        """

    private static let probeSourceStaged = """
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device half, dextents<int, 2>, tensor_inline> A((device half*)a, dextents<int, 2>(128, 16));
        threadgroup uint32_t bs[32 * 128 / 8];
        const uint lane = thread_index_in_simdgroup;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 16; j++) { bs[lane * 16 + j] = w[lane * 16 + j]; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B((threadgroup uchar*)bs, dextents<int, 2>(128, 32));
        auto tA = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>, metal::remove_addrspace_t<decltype(B)>, float>();
        op.run(tA, B, cT);
        out[thread_position_in_grid.x] = cT[0];
        """

    private static let probeSourceStaged8 = """
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device uint8_t, dextents<int, 2>, tensor_inline> A((device uint8_t*)a, dextents<int, 2>(128, 16));
        threadgroup uint32_t bs[32 * 128 / 4];
        const uint lane = thread_index_in_simdgroup;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 32; j++) { bs[lane * 32 + j] = w[(lane * 32 + j) & 255]; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tensor<threadgroup uint8_t, dextents<int, 2>, tensor_inline> B((threadgroup uint8_t*)bs, dextents<int, 2>(128, 32));
        auto tA = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>, metal::remove_addrspace_t<decltype(B)>, int32_t>();
        op.run(tA, B, cT);
        out[thread_position_in_grid.x] = float(cT[0]);
        """

    nonisolated(unsafe) private static var probeFailed = false

    /// Compiles and runs one tiny kernel; false when the toolchain rejects it
    /// (the JIT error is caught by a scoped MLX error handler instead of
    /// ending the process).
    private static func probe(_ name: String, _ source: String, aDType: DType = .float16) -> Bool {
        probeFailed = false
        let kernel = MLXFast.metalKernel(
            name: name, inputNames: ["a", "w"], outputNames: ["out"], source: source,
            header: probeHeader, ensureRowContiguous: true)
        withErrorHandler({ _ in Qwen35TensorPackedMatmul.probeFailed = true }) {
            let a = MLXArray.zeros([16, 128], dtype: aDType)
            let w = MLXArray.zeros([32 * 128 / 16], dtype: .uint32)
            let y = kernel(
                [a, w], template: [], grid: (32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[32]], outputDTypes: [.float32])[0]
            eval(y)
        }
        return !probeFailed
    }

    /// Decided once at model init. `DARKBLOOM_BONSAI_TENSOR_ROUTE_PACKED`
    /// = `uint2b` / `uint4b` / `off` forces a form (for A/B on a box that has
    /// both); otherwise the probes decide.
    static let support: PackedOperandSupport = {
        let forced = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_PACKED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch forced {
        case "uint2b", "native": return probe("bonsai_probe_uint2b", probeSourceNative) ? .native2b : .none
        case "uint8", "staged8": return probe("bonsai_probe_uint8", probeSourceStaged8, aDType: .uint8) ? .staged8 : .none
        case "uint4b", "staged": return probe("bonsai_probe_uint4b", probeSourceStaged) ? .staged4b : .none
        case "off", "none", "0": return .none
        default: break
        }
        // Preferred order: the int8-staged form (fastest measured, no packed
        // format needed), then the native 2-bit operand, then the 4-bit-staged form.
        if probe("bonsai_probe_uint8", probeSourceStaged8, aDType: .uint8) { return .staged8 }
        if probe("bonsai_probe_uint2b", probeSourceNative) { return .native2b }
        if probe("bonsai_probe_uint4b", probeSourceStaged) { return .staged4b }
        return .none
    }()

    /// True when the toolchain takes `tensor` operands at all (either form).
    static var tensorOperandsAvailable: Bool { support != .none }

    /// The verify-width route's operand form. `staged8`: the activation
    /// quantized to signed int8 per 128-group (the prompt route's rotation)
    /// and `int8 x int8 -> int32` ops over threadgroup-staged int8 weight
    /// slices, which run at the tensor unit's int8 rate (about twice its
    /// FP16 rate: 116 against 58 TF/s on an M5 Max); it beats both the
    /// native 2-bit operand with an FP16 activation (-3% decode window) and
    /// the record's verify kernels (-5%) and needs no packed format, so it is
    /// the form wherever the int8 op compiles. `native2b`: the FP16
    /// activation against the 2-bit operand, where the toolchain has it.
    /// `none`: the record's verify kernels.
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_FORM=staged8|native|off` forces one.
    static let verifyForm: PackedOperandSupport = {
        let forced = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_FORM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch forced {
        case "native", "uint2b": return probe("bonsai_probe_uint2b", probeSourceNative) ? .native2b : .none
        case "staged8", "uint8", "int8": return signedCodes ? .staged8 : .none
        case "off", "none", "0": return .none
        default: break
        }
        if signedCodes { return .staged8 }
        if support == .native2b || probe("bonsai_probe_uint2b", probeSourceNative) { return .native2b }
        return .none
    }()

    static var verifyNativeAvailable: Bool { verifyForm == .native2b }

    /// The int8-staged prompt kernel reads the activation codes as signed
    /// int8 (`q`, not `q + 128`): its products then carry no `128 * colsum`
    /// offset, so the epilogue's folded-offset term and its two 16-byte loads
    /// per lane and group go away. Same values bit for bit: `fma(s, C, -128 s
    /// colsum)` and `s * (C - 128 colsum)` each round once to the same
    /// number (the offset is exact in FP32). Only that form takes it; the
    /// native and 4-bit-staged forms keep the unsigned codes.
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_SIGNED=0` keeps the unsigned codes.
    static let signedCodes: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_SIGNED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if ["0", "false", "no", "off"].contains(value ?? "") { return false }
        guard support == .staged8 else { return false }
        return probe(
            "bonsai_probe_int8",
            probeSourceStaged8.replacingOccurrences(of: "uint8_t", with: "int8_t"), aDType: .int8)
    }()

    /// The dtype of the activation codes the quantizing rotations write.
    static var codesDType: DType { signedCodes ? .int8 : .uint8 }

    /// The int8-staged prompt kernel's per-group epilogue in factored form when
    /// the offsets are the negated scales and the codes are signed:
    /// `s * (as * C - rsb)` instead of `as * (s * C) + (-s) * rsb`, one FMA
    /// fewer per output element and 128-group (1-1.5% on the prompt-width
    /// matmuls on an M5 Max). The values differ only by FP32 rounding.
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_FACTORED_EPILOGUE=0` keeps the unfactored form.
    static let factoredPromptEpilogue: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_FACTORED_EPILOGUE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The widest projection the verify-width route takes: every tower
    /// projection and the vocabulary head (n = 248320) by default. Excluding
    /// gate|up (n = 34816) measured 3% slower in situ although the record's
    /// few-row core streams it faster in isolation; the head on the route
    /// measured 1.2% faster on the decode window.
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_MAXN` overrides it.
    static let verifyMaximumColumns: Int = {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_MAXN"],
            let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)), value > 0
        {
            return value
        }
        return 262144
    }()

    /// Row-tiled activation constants for the prompt route: the quantizing
    /// rotations store each 64-row tile's scales and scaled sums as
    /// `[tile][group][64]` in the packed kernels' lane order, so a lane reads
    /// its four rows' constants as one `float4` per group instead of eight
    /// scalar loads. `DARKBLOOM_BONSAI_TENSOR_ROUTE_MPERM=0` keeps `[rows,
    /// groups]`.
    static let rowTiledConstants: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_MPERM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // Verify width without the 2-bit operand: the native kernel's structure
    // (32 columns per threadgroup, four simdgroups splitting K, a threadgroup
    // reduce) with each simdgroup's 128-group slice of the 32 columns staged
    // as int8 in its own threadgroup buffer (2-bit codes expanded, K
    // permuted, simdgroup barriers only). The activation is read in the
    // permuted K order the staged prompt kernel uses.
    private static let sourceNarrowStaged8 = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int gper = Kg / 4;
        const int g0 = int(sg) * gper;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device half, dextents<int, 2>, tensor_inline> A((device half*)x, dextents<int, 2>(K, M));
        threadgroup uint32_t bs[4][1][32 * 128 / 4];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)bs[sg][0], dextents<int, 2>(128, 32));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B1((threadgroup int8_t*)bs[sg][1 - 1], dextents<int, 2>(128, 32));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, float>();
        constexpr int CAP = 32 / 2;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        // staging: 32 columns x 8 words per group over 32 lanes: lane -> column lane % 32, word part lane / 32
        constexpr int PARTS = 32 / 32;             // lanes per column (1 for 32=32, 2 for 32=16)
        constexpr int WPP = 8 / PARTS;             // 2-bit words per lane per group
        const int sc = int(lane) % 32; const int sp = int(lane) / 32;
        const device uint32_t* wrow = w + (size_t)(n0 + sc) * (K / 16) + sp * WPP;
        auto stage = [&](int g, int buf) {
          threadgroup uint32_t* dst = bs[sg][buf] + sc * 32 + sp * (WPP * 4);
          // One word's four planes are four contiguous uint32s at a 4-word
          // aligned offset: one uint4 store, same values and positions.
          #pragma clang loop unroll(full)
          for (int j = 0; j < WPP; j++) {
            const uint32_t wv = wrow[g * 8 + j];
            *(threadgroup uint4*)(dst + 4 * j) = uint4(
                wv & 0x03030303u, (wv >> 2) & 0x03030303u,
                (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
          }
        };
        stage(g0, 0);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = g0; g < g0 + gper; g++) {
          const int cur = (1 == 2) ? ((g - g0) & 1) : 0;
          if (1 == 2 && g + 1 < g0 + gper) { stage(g + 1, cur ^ 1); }
          auto tA = A.template slice<128, 16>(g * 128, 0);
          if (cur == 0) { op.run(tA, B0, cT); } else { op.run(tA, B1, cT); }
          float4 sv[32 / 16], bv[32 / 16];
          #pragma clang loop unroll(full)
          for (int q = 0; q < 32 / 16; q++) {
            sv[q] = float4(*(const device half4*)(scalesT + (size_t)g * N + n0 + fn + 16 * q));
            bv[q] = float4(*(const device half4*)(biasesT + (size_t)g * N + n0 + fn + 16 * q));
          }
          const float rs0 = rowsum[(size_t)fm * Kg + g];
          const float rs1 = rowsum[(size_t)(fm + 8) * Kg + g];
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
            acc[i] = fma(sv[nq][c], cT[i], fma(bv[nq][c], mh ? rs1 : rs0, acc[i]));
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (1 == 1 && g + 1 < g0 + gper) { stage(g + 1, 0); simdgroup_barrier(mem_flags::mem_threadgroup); }
        }
        threadgroup float red[4 - 1][CAP * 32];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { red[sg - 1][i * 32 + lane] = acc[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            float v = acc[i];
            #pragma clang loop unroll(full)
            for (int q = 0; q < 4 - 1; q++) { v += red[q][i * 32 + lane]; }
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
            out[(size_t)(fm + 8 * mh) * N + n0 + fn + c + 16 * nq] = OutT(v);
          }
        }
        """

    // Verify width over the quantized rotation (signed int8 codes, as the
    // prompt route's int8-staged kernel reads them): the same 32-column,
    // four-simdgroup split-K structure, each simdgroup staging its
    // 128-group slice of the weight tile as int8 in its own threadgroup
    // buffer, the op `int8 x int8 -> int32` at the int8 rate of the tensor
    // unit (about twice the FP16 rate), the affine map in FP32 from the
    // activation's per-group scale and scaled sum. Template `NEG` takes the
    // offset as `-scale` (proven per constant pair; no offset load) and `F32S`
    // reads FP32-widened scales; see `NarrowEpilogue`. Both are exact.
    private static let sourceNarrowInt8 = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int gper = Kg / 4;
        const int g0 = int(sg) * gper;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)x, dextents<int, 2>(K, M));  // SIGNED codes only
        threadgroup uint32_t bs[4][1][32 * 128 / 4];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)bs[sg][0], dextents<int, 2>(128, 32));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B1((threadgroup int8_t*)bs[sg][1 - 1], dextents<int, 2>(128, 32));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        constexpr int CAP = 32 / 2;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        // staging: 32 columns x 8 words per group over 32 lanes: each lane stages 32/32 columns (all 8 words each)
        constexpr int CPL = 32 / 32;
        auto stage = [&](int g, int buf) {
          #pragma clang loop unroll(full)
          for (int cc = 0; cc < CPL; cc++) {
            const int sc = int(lane) + 32 * cc;
            // TILED: the tiled copy (`narrowTiledWeight`) holds each 32-column
            // block's 8 words per column of a group contiguously, so the 32
            // lanes read 1 KB in one run instead of 32 B from each of 32 rows.
            const device uint32_t* wrow = TILED
                ? w + ((size_t)(n0 / 32) * Kg + (size_t)g) * 256 + (size_t)sc * 8
                : w + (size_t)(n0 + sc) * (K / 16) + (size_t)g * 8;
            threadgroup uint32_t* dst = bs[sg][buf] + sc * 32;
            // As in the prompt kernel's staging: one uint4 store per word.
            #pragma clang loop unroll(full)
            for (int j = 0; j < 8; j++) {
              const uint32_t wv = wrow[j];
              *(threadgroup uint4*)(dst + 4 * j) = uint4(
                  wv & 0x03030303u, (wv >> 2) & 0x03030303u,
                  (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
            }
          }
        };
        stage(g0, 0);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = g0; g < g0 + gper; g++) {
          const int cur = (1 == 2) ? ((g - g0) & 1) : 0;
          if (1 == 2 && g + 1 < g0 + gper) { stage(g + 1, cur ^ 1); }
          auto tA = A.template slice<128, 16>(g * 128, 0);
          if (cur == 0) { op.run(tA, B0, cT); } else { op.run(tA, B1, cT); }
          float4 sv[32 / 16], bv[32 / 16];
          #pragma clang loop unroll(full)
          for (int q = 0; q < 32 / 16; q++) {
            // F32S: scalesT holds the FP16 scales widened to FP32 at load
            // (exact), so `sv` is the same value without the conversion.
            if constexpr (F32S) {
              sv[q] = *(const device float4*)(scalesT + (size_t)g * N + n0 + fn + 16 * q);
            } else {
              sv[q] = float4(*(const device half4*)(scalesT + (size_t)g * N + n0 + fn + 16 * q));
            }
            // NEG: every offset is the negated scale (FP16 bits proven at
            // load), so `-sv` is the offset's exact FP32 value; no load.
            if constexpr (NEG) {
              bv[q] = -sv[q];
            } else {
              bv[q] = float4(*(const device half4*)(biasesT + (size_t)g * N + n0 + fn + 16 * q));
            }
          }
          const float as0 = ascale[(size_t)fm * Kg + g], as1 = ascale[(size_t)(fm + 8) * Kg + g];
          const float rs0 = rowsum[(size_t)fm * Kg + g];
          const float rs1 = rowsum[(size_t)(fm + 8) * Kg + g];
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
            acc[i] = fma(mh ? as1 : as0, sv[nq][c] * float(cT[i]), fma(bv[nq][c], mh ? rs1 : rs0, acc[i]));
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (1 == 1 && g + 1 < g0 + gper) { stage(g + 1, 0); simdgroup_barrier(mem_flags::mem_threadgroup); }
        }
        // The reduction reuses the staging buffers (free after the K loop):
        // 16 KB of threadgroup memory in all, two threadgroups per core.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float (*red)[CAP * 32] = (threadgroup float (*)[CAP * 32])&bs[0][0][0];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { red[sg - 1][i * 32 + lane] = acc[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          // i = 0, 4, 8, 12. mh and nq are constant on each group and c is
          // 0, 1, 2, 3, so the four outputs are consecutive columns. The sum
          // is still acc, then red[0], red[1], red[2], each folded on its own
          // partial. OutT is half or float; both stores are 4-element aligned
          // because fn, n0 and N are multiples of 4.
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i += 4) {
            const int mh = (i >> 2) & 1;
            const int nq = i >> 3;
            float v0 = acc[i];
            float v1 = acc[i + 1];
            float v2 = acc[i + 2];
            float v3 = acc[i + 3];
            #pragma clang loop unroll(full)
            for (int q = 0; q < 4 - 1; q++) {
              v0 += red[q][i * 32 + lane];
              v1 += red[q][(i + 1) * 32 + lane];
              v2 += red[q][(i + 2) * 32 + lane];
              v3 += red[q][(i + 3) * 32 + lane];
            }
            const size_t base = (size_t)(fm + 8 * mh) * N + n0 + fn + 16 * nq;
            if constexpr (sizeof(OutT) == sizeof(float)) {
              *(device float4*)(out + base) = float4(v0, v1, v2, v3);
            } else {
              *(device half4*)(out + base) = half4(half(v0), half(v1), half(v2), half(v3));
            }
          }
        }
        """

    // The verify int8 kernel software-pipelined through registers (K2, K3):
    // the same threadgroup (four simdgroups splitting K into contiguous
    // quarters, the 16 x 32 int8 products of each group and 32-column half,
    // the partials summed in simdgroup order) and the same arithmetic in the
    // same order, so the output is bitwise that of `sourceNarrowInt8`
    // (self-tested at load). What changes is when the loads are issued and how
    // much threadgroup memory a threadgroup holds:
    // - PD (1..4): a static register ring of the 2-bit words of the next PD
    //   groups, refilled as each group is staged, and a matching ring of
    //   epilogue constants (depth max(PD, 2), loaded that many groups minus
    //   one ahead). The group loop is unrolled by the ring depth, so every ring
    //   index is a constant; remainder groups are guarded (10 groups per
    //   simdgroup at K = 5120, 34 at down_proj).
    // - KH (128 or 64): K per tensor op. 64 stages each group in two halves
    //   (stage -> barrier -> op 16 x 32 x 64, accumulating into the group's
    //   zeroed int32 tile, twice) and runs the epilogue once per group: 2 KB of
    //   staging per simdgroup and half, 8 KB per threadgroup at TN = 32 (four
    //   threadgroups per core instead of two). The integer sum of a group is
    //   exact under any split, so the FP32 sequence is unchanged.
    // - TN (32 or 64): 32-column halves per threadgroup (two ops per step).
    // Templates: OutT, NEG, F32S (as `sourceNarrowInt8`), PD, TN, KH.
    // grid (N / TN * 128, 1, 1), threadgroup (128, 1, 1).
    private static let sourceNarrowInt8Pipelined = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * TN;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int gper = Kg / 4;
        const int g0 = int(sg) * gper;
        const int g1 = g0 + gper;
        constexpr int NH = TN / 32;
        constexpr int KW = KH / 16;           // 2-bit words per column per staged step
        constexpr int NQ = KH / 64;           // uint4 word quads per staged step
        constexpr int CD = PD < 2 ? 2 : PD;   // constants ring depth = group-loop unroll
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, KH, false, true, false, KH == 64 ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate : mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)x, dextents<int, 2>(K, M));  // SIGNED codes only
        threadgroup uint32_t bs[4][NH][32 * KH / 4];
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)bs[sg][0], dextents<int, 2>(KH, 32));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B1((threadgroup int8_t*)bs[sg][NH - 1], dextents<int, 2>(KH, 32));
        auto tA0 = A.template slice<KH, 16>(0, 0);
        auto cT0 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        auto cT1 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        constexpr int CAP = 32 / 2;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[NH][CAP];
        #pragma clang loop unroll(full)
        for (int h = 0; h < NH; h++) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { acc[h][i] = 0.0f; }
        }
        // lane -> column lane of each 32-column half, 8 words (two quads) per group
        // TILED: see `sourceNarrowInt8`; the next 32-column half is the next
        // block, and a group's words sit 256 words after the previous group's.
        const device uint32_t* wrow = TILED
            ? w + (size_t)(n0 / 32) * (K / 128) * 256 + (size_t)int(lane) * 8
            : w + (size_t)(n0 + int(lane)) * (K / 16);
        const size_t hstride = TILED ? (size_t)(K / 128) * 256 : (size_t)32 * (K / 16);
        const size_t gstride = TILED ? 256 : 8;
        // quads [q0, q1) of group gg's words into v
        auto getw = [&](int gg, thread uint32_t (&v)[NH][8], int q0, int q1) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            const device uint4* src = (const device uint4*)(wrow + h * hstride + (size_t)gg * gstride);
            #pragma clang loop unroll(full)
            for (int q = 0; q < 2; q++) {
              if (q < q0 || q >= q1) { continue; }
              const uint4 u = src[q];
              v[h][4 * q + 0] = u.x; v[h][4 * q + 1] = u.y; v[h][4 * q + 2] = u.z; v[h][4 * q + 3] = u.w;
            }
          }
        };
        // stages quads [q0, q1) of v: word j of the step at 4 * (j % KW) of its column
        auto putw = [&](thread const uint32_t (&v)[NH][8], int q0, int q1) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            threadgroup uint32_t* dst = bs[sg][h] + int(lane) * (KH / 4);
            #pragma clang loop unroll(full)
            for (int j = 0; j < 8; j++) {
              if (j < 4 * q0 || j >= 4 * q1) { continue; }
              // As in the prompt kernel's staging: one uint4 store per word.
              const uint32_t wv = v[h][j];
              *(threadgroup uint4*)(dst + 4 * (j % KW)) = uint4(
                  wv & 0x03030303u, (wv >> 2) & 0x03030303u,
                  (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
            }
          }
        };
        // epilogue constants of one group, kept in their stored types until use
        auto getc = [&](int g, thread half4 (&sh)[NH][2], thread half4 (&bh)[NH][2],
                        thread float4 (&sf)[NH][2], thread float (&c)[4]) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            #pragma clang loop unroll(full)
            for (int q = 0; q < 2; q++) {
              const size_t o = (size_t)g * N + n0 + 32 * h + fn + 16 * q;
              if constexpr (F32S) { sf[h][q] = *(const device float4*)(scalesT + o); }
              else { sh[h][q] = *(const device half4*)(scalesT + o); }
              if constexpr (!NEG) { bh[h][q] = *(const device half4*)(biasesT + o); }
            }
          }
          c[0] = ascale[(size_t)fm * Kg + g]; c[1] = ascale[(size_t)(fm + 8) * Kg + g];
          c[2] = rowsum[(size_t)fm * Kg + g]; c[3] = rowsum[(size_t)(fm + 8) * Kg + g];
        };
        // Word ring: before group g runs, slot (g - g0 - 1) % PD holds group g + PD
        // (KH = 64: its lower quad, the upper quad still holding group g's) and the
        // other slots groups g + 1 .. g + PD - 1. Constants ring: slot (g - g0) % CD
        // holds group g's, the next CD - 2 slots the following groups'.
        uint32_t wr[PD][NH][8];
        half4 shr[CD][NH][2], bhr[CD][NH][2];
        float4 sfr[CD][NH][2];
        float cr[CD][4];
        // one K step of group g at offset ko: KH x 32 staged codes per half
        auto mm = [&](int g, int ko) {
          auto tA = A.template slice<KH, 16>(g * 128 + ko, 0);
          op.run(tA, B0, cT0);
          if constexpr (NH == 2) {
          op.run(tA, B1, cT1);
          }
        };
        // group g at unrolled position j (a constant): ring slots are static
        auto body = [&](int g, int j) {
          const int cs = j % CD;
          if (g + CD - 1 < g1) {
            const int cn = (j + CD - 1) % CD;
            getc(g + CD - 1, shr[cn], bhr[cn], sfr[cn], cr[cn]);
          }
          if constexpr (KH == 64) {
            const int wp = (j + PD - 1) % PD;  // slot holding group g's upper quad
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i++) { cT0[i] = 0; cT1[i] = 0; }
            mm(g, 0);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            putw(wr[wp], 1, 2);
            if (g + PD < g1) { getw(g + PD, wr[wp], 1, 2); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            mm(g, 64);
          } else {
            mm(g, 0);
          }
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            float4 sv[2], bv[2];
            #pragma clang loop unroll(full)
            for (int q = 0; q < 2; q++) {
              if constexpr (F32S) { sv[q] = sfr[cs][h][q]; } else { sv[q] = float4(shr[cs][h][q]); }
              if constexpr (NEG) { bv[q] = -sv[q]; } else { bv[q] = float4(bhr[cs][h][q]); }
            }
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i++) {
              const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
              const int32_t ci = (h == 0) ? cT0[i] : cT1[i];
              acc[h][i] = fma(mh ? cr[cs][1] : cr[cs][0], sv[nq][c] * float(ci), fma(bv[nq][c], mh ? cr[cs][3] : cr[cs][2], acc[h][i]));
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (g + 1 < g1) {
            const int wn = j % PD;             // slot holding group g + 1 (its lower quad at KH = 64)
            putw(wr[wn], 0, NQ);
            if (g + 1 + PD < g1) { getw(g + 1 + PD, wr[wn], 0, NQ); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
          }
        };
        {
          uint32_t w0[NH][8];
          getw(g0, w0, 0, NQ);
          #pragma clang loop unroll(full)
          for (int i = 1; i < PD; i++) {
            if (g0 + i < g1) { getw(g0 + i, wr[i - 1], 0, 2); }
          }
          if constexpr (KH == 64) { getw(g0, wr[PD - 1], 1, 2); }
          if (g0 + PD < g1) { getw(g0 + PD, wr[PD - 1], 0, NQ); }
          #pragma clang loop unroll(full)
          for (int i = 0; i < CD - 1; i++) {
            if (g0 + i < g1) { getc(g0 + i, shr[i], bhr[i], sfr[i], cr[i]); }
          }
          putw(w0, 0, NQ);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = g0; g < g1; g += CD) {
          #pragma clang loop unroll(full)
          for (int j = 0; j < CD; j++) {
            if (g + j < g1) { body(g + j, j); }
          }
        }
        // the reduction reuses the staging buffers, in simdgroup order
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* red = (threadgroup float*)&bs[0][0][0];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i++) { red[((int(sg) - 1) * NH + h) * (CAP * 32) + i * 32 + int(lane)] = acc[h][i]; }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          // As in `sourceNarrowInt8` (fkiene 98f554ad): i = 0, 4, 8, 12. mh
          // and nq are constant on each group and c is 0, 1, 2, 3, so the four
          // outputs are consecutive columns. The sum is still acc, then
          // red[0], red[1], red[2], each folded on its own partial. OutT is
          // half or float; both stores are 4-element aligned because fn, n0,
          // 32 * h and N are multiples of 4.
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i += 4) {
              const int mh = (i >> 2) & 1;
              const int nq = i >> 3;
              float v0 = acc[h][i];
              float v1 = acc[h][i + 1];
              float v2 = acc[h][i + 2];
              float v3 = acc[h][i + 3];
              #pragma clang loop unroll(full)
              for (int q = 0; q < 4 - 1; q++) {
                v0 += red[(q * NH + h) * (CAP * 32) + i * 32 + int(lane)];
                v1 += red[(q * NH + h) * (CAP * 32) + (i + 1) * 32 + int(lane)];
                v2 += red[(q * NH + h) * (CAP * 32) + (i + 2) * 32 + int(lane)];
                v3 += red[(q * NH + h) * (CAP * 32) + (i + 3) * 32 + int(lane)];
              }
              const size_t base = (size_t)(fm + 8 * mh) * N + n0 + 32 * h + fn + 16 * nq;
              if constexpr (sizeof(OutT) == sizeof(float)) {
                *(device float4*)(out + base) = float4(v0, v1, v2, v3);
              } else {
                *(device half4*)(out + base) = half4(half(v0), half(v1), half(v2), half(v3));
              }
            }
          }
        }
        """

    // Tiled zoo (`DARKBLOOM_BONSAI_TENSOR_ROUTE_TZOO`, on the tiled layout of
    // Subflatus3 bb781255): more verify int8 bodies that keep every output's
    // reduction order, so every one is bitwise that of `sourceNarrowInt8`
    // (self-tested at load against `original` on the stored words). Per output
    // the FP32 value is fixed by (a) the exact int32 16 x 32 product of each
    // 128-group (any K split inside a group, any operand path: integers), (b)
    // the four contiguous K quarters, each folded group by group in ascending
    // order with the same two FMAs, and (c) quarter 0 + 1 + 2 + 3. The bodies
    // move only what (a)-(c) leave free:
    // - `sourceNarrowInt8Zoo`: KH 32 / 64 / 128 per op (staging 1 / 2 / 4 KB
    //   per simdgroup and 32-column half), TN 32 / 64, a PD-deep register ring
    //   of whole groups' words (slot freed when its last K step is staged),
    //   AM = 2 A as a left-input cooperative tensor loaded one group ahead
    //   (one load for both halves at TN = 64). Threadgroup memory is the larger
    //   of the staging and the reduction (6 KB at TN = 32, 12 KB at TN = 64).
    //   Templates: OutT, NEG, F32S, PD, TN, KH, AM. grid (N / TN * 128), (128).
    // - `sourceNarrowInt8Pair`: eight simdgroups per 32 columns; quarter q's
    //   owner (simdgroup q) multiplies its even groups and folds the chain,
    //   quarter q's helper (simdgroup q + 4) multiplies the odd groups and hands
    //   each int32 tile over in threadgroup memory before the owner folds it,
    //   so the owner's sequence is (b) exactly with twice the loads in flight
    //   per column block (for the N = 5120 shapes: 160 threadgroups). KH 32 / 64
    //   (8 / 16 KB staging + 8 KB exchange). Templates: OutT, NEG, F32S, PD
    //   (1 or 2), KH. grid (N / 32 * 256), (256).
    // Both read the tiled copy only.
    private static let sourceNarrowInt8Zoo = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * TN;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int gper = Kg / 4;
        const int g0 = int(sg) * gper;
        const int g1 = g0 + gper;
        constexpr int NH = TN / 32;
        constexpr int NP = 128 / KH;          // K steps per 128-group
        constexpr int KW = KH / 16;           // 2-bit words per column per step
        constexpr int CD = PD < 2 ? 2 : PD;   // constants ring depth = group-loop unroll
        constexpr int CAP = 32 / 2;
        constexpr int SWS = NH * 32 * KH / 4; // staging words per simdgroup
        constexpr int RDW = 3 * NH * CAP * 32; // reduction words
        static_assert(AM == 0 || NP <= 2, "AM = 2 takes KH = 64 or 128");
        // staging, reused by the reduction: sized for whichever is larger
        threadgroup uint32_t bs[4 * SWS > RDW ? 4 * SWS : RDW];
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, KH, false, true, false, NP > 1 ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate : mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)x, dextents<int, 2>(K, M));  // SIGNED codes only
        threadgroup uint32_t* sb = bs + int(sg) * SWS;
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)sb, dextents<int, 2>(KH, 32));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B1((threadgroup int8_t*)(sb + (NH - 1) * (32 * KH / 4)), dextents<int, 2>(KH, 32));
        auto tA0 = A.template slice<KH, 16>(0, 0);
        auto cT0 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        auto cT1 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        // AM = 2: A as the op's left-input cooperative tensor, one per K step of a
        // group, refilled with the next group's slice right after the op that read it
        // (one load for both 32-column halves at TN = 64). The toolchain must have the
        // int8 cooperative-input load; where it does not, AM = 2 fails to compile.
        #if defined(__TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2) && __TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2
        auto cA0 = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        auto cA1 = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        #else
        static_assert(AM == 0, "no int8 cooperative left input on this toolchain");
        #endif
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[NH][CAP];
        #pragma clang loop unroll(full)
        for (int h = 0; h < NH; h++) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { acc[h][i] = 0.0f; }
        }
        // Tiled copy: block n0 / 32 + h, group gg, column lane: 8 contiguous words.
        const device uint32_t* wrow = w + (size_t)(n0 / 32) * (size_t)Kg * 256 + (size_t)int(lane) * 8;
        const size_t hstride = (size_t)Kg * 256;
        // all 8 words of group gg (both 32-column halves)
        auto getw = [&](int gg, thread uint32_t (&v)[NH][8]) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            const device uint4* src = (const device uint4*)(wrow + h * hstride + (size_t)gg * 256);
            const uint4 u0 = src[0];
            const uint4 u1 = src[1];
            v[h][0] = u0.x; v[h][1] = u0.y; v[h][2] = u0.z; v[h][3] = u0.w;
            v[h][4] = u1.x; v[h][5] = u1.y; v[h][6] = u1.z; v[h][7] = u1.w;
          }
        };
        // stages K step p of v (words p * KW .. p * KW + KW - 1): one uint4 store per word
        auto putw = [&](thread const uint32_t (&v)[NH][8], int p) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            threadgroup uint32_t* dst = sb + h * (32 * KH / 4) + int(lane) * (KH / 4);
            #pragma clang loop unroll(full)
            for (int jj = 0; jj < KW; jj++) {
              const uint32_t wv = v[h][p * KW + jj];
              *(threadgroup uint4*)(dst + 4 * jj) = uint4(
                  wv & 0x03030303u, (wv >> 2) & 0x03030303u,
                  (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
            }
          }
        };
        // epilogue constants of one group, kept in their stored types until use
        auto getc = [&](int g, thread half4 (&sh)[NH][2], thread half4 (&bh)[NH][2],
                        thread float4 (&sf)[NH][2], thread float (&c)[4]) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            #pragma clang loop unroll(full)
            for (int q = 0; q < 2; q++) {
              const size_t o = (size_t)g * N + n0 + 32 * h + fn + 16 * q;
              if constexpr (F32S) { sf[h][q] = *(const device float4*)(scalesT + o); }
              else { sh[h][q] = *(const device half4*)(scalesT + o); }
              if constexpr (!NEG) { bh[h][q] = *(const device half4*)(biasesT + o); }
            }
          }
          c[0] = ascale[(size_t)fm * Kg + g]; c[1] = ascale[(size_t)(fm + 8) * Kg + g];
          c[2] = rowsum[(size_t)fm * Kg + g]; c[3] = rowsum[(size_t)(fm + 8) * Kg + g];
        };
        // Word ring: group g0 + i sits in slot i % PD from its load until its last K
        // step is staged; the slot is then refilled with group g0 + i + PD. Constants
        // ring: slot (g - g0) % CD holds group g's, loaded CD - 1 groups ahead.
        uint32_t wr[PD][NH][8];
        half4 shr[CD][NH][2], bhr[CD][NH][2];
        float4 sfr[CD][NH][2];
        float cr[CD][4];
        // K step p of group g: the staged KH x 32 codes of each half
        auto mm = [&](int g, int p) {
          #if defined(__TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2) && __TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2
          if constexpr (AM == 2) {
            if (p == 0) {
              op.run(cA0, B0, cT0);
              if constexpr (NH == 2) { op.run(cA0, B1, cT1); }
              if (g + 1 < g1) { cA0.load(A.template slice<KH, 16>((g + 1) * 128, 0)); }
            } else {
              op.run(cA1, B0, cT0);
              if constexpr (NH == 2) { op.run(cA1, B1, cT1); }
              if (g + 1 < g1) { cA1.load(A.template slice<KH, 16>((g + 1) * 128 + KH, 0)); }
            }
            return;
          }
          #endif
          auto tA = A.template slice<KH, 16>(g * 128 + p * KH, 0);
          op.run(tA, B0, cT0);
          if constexpr (NH == 2) { op.run(tA, B1, cT1); }
        };
        // group g at unrolled position j (a constant): ring slots are static
        auto body = [&](int g, int j) {
          const int cs = j % CD;
          const int ws = j % PD;
          if (g + CD - 1 < g1) {
            const int cn = (j + CD - 1) % CD;
            getc(g + CD - 1, shr[cn], bhr[cn], sfr[cn], cr[cn]);
          }
          if constexpr (NP > 1) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i++) { cT0[i] = 0; cT1[i] = 0; }
          }
          #pragma clang loop unroll(full)
          for (int p = 0; p < NP; p++) {
            if (p > 0) {
              simdgroup_barrier(mem_flags::mem_threadgroup);
              putw(wr[ws], p);
              simdgroup_barrier(mem_flags::mem_threadgroup);
              if (p == NP - 1 && g + PD < g1) { getw(g + PD, wr[ws]); }
            }
            mm(g, p);
          }
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            float4 sv[2], bv[2];
            #pragma clang loop unroll(full)
            for (int q = 0; q < 2; q++) {
              if constexpr (F32S) { sv[q] = sfr[cs][h][q]; } else { sv[q] = float4(shr[cs][h][q]); }
              if constexpr (NEG) { bv[q] = -sv[q]; } else { bv[q] = float4(bhr[cs][h][q]); }
            }
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i++) {
              const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
              const int32_t ci = (h == 0) ? cT0[i] : cT1[i];
              acc[h][i] = fma(mh ? cr[cs][1] : cr[cs][0], sv[nq][c] * float(ci), fma(bv[nq][c], mh ? cr[cs][3] : cr[cs][2], acc[h][i]));
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (g + 1 < g1) {
            const int wn = (j + 1) % PD;       // slot holding group g + 1
            putw(wr[wn], 0);
            if (NP == 1 && g + 1 + PD < g1) { getw(g + 1 + PD, wr[wn]); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
          }
        };
        #pragma clang loop unroll(full)
        for (int i = 0; i < PD; i++) {
          if (g0 + i < g1) { getw(g0 + i, wr[i]); }
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < CD - 1; i++) {
          if (g0 + i < g1) { getc(g0 + i, shr[i], bhr[i], sfr[i], cr[i]); }
        }
        #if defined(__TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2) && __TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2
        if constexpr (AM == 2) {
          cA0.load(A.template slice<KH, 16>(g0 * 128, 0));
          if constexpr (NP == 2) { cA1.load(A.template slice<KH, 16>(g0 * 128 + KH, 0)); }
        }
        #endif
        putw(wr[0], 0);
        if (NP == 1 && g0 + PD < g1) { getw(g0 + PD, wr[0]); }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = g0; g < g1; g += CD) {
          #pragma clang loop unroll(full)
          for (int j = 0; j < CD; j++) {
            if (g + j < g1) { body(g + j, j); }
          }
        }
        // the reduction reuses the staging buffers, in simdgroup order
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* red = (threadgroup float*)bs;
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i++) { red[((int(sg) - 1) * NH + h) * (CAP * 32) + i * 32 + int(lane)] = acc[h][i]; }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i += 4) {
              const int mh = (i >> 2) & 1;
              const int nq = i >> 3;
              float v0 = acc[h][i];
              float v1 = acc[h][i + 1];
              float v2 = acc[h][i + 2];
              float v3 = acc[h][i + 3];
              #pragma clang loop unroll(full)
              for (int q = 0; q < 4 - 1; q++) {
                v0 += red[(q * NH + h) * (CAP * 32) + i * 32 + int(lane)];
                v1 += red[(q * NH + h) * (CAP * 32) + (i + 1) * 32 + int(lane)];
                v2 += red[(q * NH + h) * (CAP * 32) + (i + 2) * 32 + int(lane)];
                v3 += red[(q * NH + h) * (CAP * 32) + (i + 3) * 32 + int(lane)];
              }
              const size_t base = (size_t)(fm + 8 * mh) * N + n0 + 32 * h + fn + 16 * nq;
              if constexpr (sizeof(OutT) == sizeof(float)) {
                *(device float4*)(out + base) = float4(v0, v1, v2, v3);
              } else {
                *(device half4*)(out + base) = half4(half(v0), half(v1), half(v2), half(v3));
              }
            }
          }
        }
        """

    private static let sourceNarrowInt8Pair = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int qd = int(sg) & 3;           // K quarter, as in `sourceNarrowInt8`
        const int role = int(sg) >> 2;        // 0: owner (even groups, the FP32 sum), 1: helper (odd groups)
        const int gper = Kg / 4;
        const int g0 = qd * gper;
        const int g1 = g0 + gper;
        const int npair = (gper + 1) / 2;
        constexpr int NP = 128 / KH;          // K steps per 128-group
        constexpr int KW = KH / 16;           // 2-bit words per column per step
        constexpr int CAP = 32 / 2;
        constexpr int SWS = 32 * KH / 4;      // staging words per simdgroup
        constexpr int RDW = 3 * CAP * 32;     // reduction words
        static_assert(8 * SWS >= RDW, "the reduction reuses the staging");
        threadgroup uint32_t bs[8 * SWS];
        threadgroup int32_t xch[4 * CAP * 32]; // each quarter's helper tile, one pair step at a time
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, KH, false, true, false, NP > 1 ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate : mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)x, dextents<int, 2>(K, M));  // SIGNED codes only
        threadgroup uint32_t* sb = bs + int(sg) * SWS;
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)sb, dextents<int, 2>(KH, 32));
        auto tA0 = A.template slice<KH, 16>(0, 0);
        auto cT0 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        // Tiled copy: block n0 / 32, group gg, column lane: 8 contiguous words.
        const device uint32_t* wrow = w + (size_t)(n0 / 32) * (size_t)Kg * 256 + (size_t)int(lane) * 8;
        auto getw = [&](int gg, thread uint32_t (&v)[8]) {
          const device uint4* src = (const device uint4*)(wrow + (size_t)gg * 256);
          const uint4 u0 = src[0];
          const uint4 u1 = src[1];
          v[0] = u0.x; v[1] = u0.y; v[2] = u0.z; v[3] = u0.w;
          v[4] = u1.x; v[5] = u1.y; v[6] = u1.z; v[7] = u1.w;
        };
        auto putw = [&](thread const uint32_t (&v)[8], int p) {
          threadgroup uint32_t* dst = sb + int(lane) * (KH / 4);
          #pragma clang loop unroll(full)
          for (int jj = 0; jj < KW; jj++) {
            const uint32_t wv = v[p * KW + jj];
            *(threadgroup uint4*)(dst + 4 * jj) = uint4(
                wv & 0x03030303u, (wv >> 2) & 0x03030303u,
                (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
          }
        };
        auto getc = [&](int g, thread half4 (&sh)[2], thread half4 (&bh)[2], thread float4 (&sf)[2],
                        thread float (&c)[4]) {
          #pragma clang loop unroll(full)
          for (int q = 0; q < 2; q++) {
            const size_t o = (size_t)g * N + n0 + fn + 16 * q;
            if constexpr (F32S) { sf[q] = *(const device float4*)(scalesT + o); }
            else { sh[q] = *(const device half4*)(scalesT + o); }
            if constexpr (!NEG) { bh[q] = *(const device half4*)(biasesT + o); }
          }
          c[0] = ascale[(size_t)fm * Kg + g]; c[1] = ascale[(size_t)(fm + 8) * Kg + g];
          c[2] = rowsum[(size_t)fm * Kg + g]; c[3] = rowsum[(size_t)(fm + 8) * Kg + g];
        };
        // one group's epilogue, the FP32 sequence of `sourceNarrowInt8`
        auto epi = [&](thread const half4 (&sh)[2], thread const half4 (&bh)[2], thread const float4 (&sf)[2],
                       thread const float (&c)[4], thread const int32_t (&ci)[CAP]) {
          float4 sv[2], bv[2];
          #pragma clang loop unroll(full)
          for (int q = 0; q < 2; q++) {
            if constexpr (F32S) { sv[q] = sf[q]; } else { sv[q] = float4(sh[q]); }
            if constexpr (NEG) { bv[q] = -sv[q]; } else { bv[q] = float4(bh[q]); }
          }
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int cc = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
            acc[i] = fma(mh ? c[1] : c[0], sv[nq][cc] * float(ci[i]), fma(bv[nq][cc], mh ? c[3] : c[2], acc[i]));
          }
        };
        // This simdgroup's groups: g0 + role + 2 t, t = 0 .. npair - 1 (when < g1).
        // Word ring: its group of pair t in slot t % PD until the last K step is
        // staged, then that of pair t + PD. The owner's constants ring: pair t's two
        // groups in slot t % 2, loaded one pair ahead.
        uint32_t wr[PD][8];
        half4 shr[2][2][2], bhr[2][2][2];
        float4 sfr[2][2][2];
        float cr[2][2][4];
        auto pairc = [&](int t, int s) {
          #pragma clang loop unroll(full)
          for (int e = 0; e < 2; e++) {
            if (g0 + 2 * t + e < g1) { getc(g0 + 2 * t + e, shr[s][e], bhr[s][e], sfr[s][e], cr[s][e]); }
          }
        };
        // pair t at unrolled position u (a constant)
        auto step = [&](int t, int u) {
          const int gm = g0 + role + 2 * t;
          const int ws = u % PD;
          const int cs = u % 2;
          if (role == 0 && t + 1 < npair) { pairc(t + 1, (u + 1) % 2); }
          if (gm < g1) {
            if constexpr (NP > 1) {
              #pragma clang loop unroll(full)
              for (int i = 0; i < CAP; i++) { cT0[i] = 0; }
            }
            #pragma clang loop unroll(full)
            for (int p = 0; p < NP; p++) {
              if (p > 0) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
                putw(wr[ws], p);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (p == NP - 1 && gm + 2 * PD < g1) { getw(gm + 2 * PD, wr[ws]); }
              }
              auto tA = A.template slice<KH, 16>(gm * 128 + p * KH, 0);
              op.run(tA, B0, cT0);
            }
            if (role == 1) {
              #pragma clang loop unroll(full)
              for (int i = 0; i < CAP; i++) { xch[(qd * CAP + i) * 32 + int(lane)] = cT0[i]; }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (role == 0) {
            int32_t ci[CAP];
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i++) { ci[i] = cT0[i]; }
            epi(shr[cs][0], bhr[cs][0], sfr[cs][0], cr[cs][0], ci);
            if (gm + 1 < g1) {
              #pragma clang loop unroll(full)
              for (int i = 0; i < CAP; i++) { ci[i] = xch[(qd * CAP + i) * 32 + int(lane)]; }
              epi(shr[cs][1], bhr[cs][1], sfr[cs][1], cr[cs][1], ci);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (gm + 2 < g1) {
            const int wn = (u + 1) % PD;       // slot holding this simdgroup's group of pair t + 1
            putw(wr[wn], 0);
            if (NP == 1 && gm + 2 + 2 * PD < g1) { getw(gm + 2 + 2 * PD, wr[wn]); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
          }
        };
        #pragma clang loop unroll(full)
        for (int i = 0; i < PD; i++) {
          if (g0 + role + 2 * i < g1) { getw(g0 + role + 2 * i, wr[i]); }
        }
        if (role == 0) { pairc(0, 0); }
        if (g0 + role < g1) {
          putw(wr[0], 0);
          if (NP == 1 && g0 + role + 2 * PD < g1) { getw(g0 + role + 2 * PD, wr[0]); }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < npair; t += 2) {
          #pragma clang loop unroll(full)
          for (int u = 0; u < 2; u++) {
            if (t + u < npair) { step(t + u, u); }
          }
        }
        // the owners' partials, reduced as in `sourceNarrowInt8` (the staging is free)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* red = (threadgroup float*)bs;
        if (role == 0 && qd > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { red[(qd - 1) * (CAP * 32) + i * 32 + int(lane)] = acc[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i += 4) {
            const int mh = (i >> 2) & 1;
            const int nq = i >> 3;
            float v0 = acc[i];
            float v1 = acc[i + 1];
            float v2 = acc[i + 2];
            float v3 = acc[i + 3];
            #pragma clang loop unroll(full)
            for (int q = 0; q < 4 - 1; q++) {
              v0 += red[q * (CAP * 32) + i * 32 + int(lane)];
              v1 += red[q * (CAP * 32) + (i + 1) * 32 + int(lane)];
              v2 += red[q * (CAP * 32) + (i + 2) * 32 + int(lane)];
              v3 += red[q * (CAP * 32) + (i + 3) * 32 + int(lane)];
            }
            const size_t base = (size_t)(fm + 8 * mh) * N + n0 + fn + 16 * nq;
            if constexpr (sizeof(OutT) == sizeof(float)) {
              *(device float4*)(out + base) = float4(v0, v1, v2, v3);
            } else {
              *(device half4*)(out + base) = half4(half(v0), half(v1), half(v2), half(v3));
            }
          }
        }
        """

    // Zoo 2: more bodies under the zoo's rule (a)-(c) above, so every one is
    // bitwise that of `sourceNarrowInt8` (self-tested at load against
    // `original` on the stored words, one error scope per body):
    // - `sourceNarrowInt8Zoo2`: the zoo body over "units" of GS consecutive
    //   groups of a quarter. A K step runs E = GS x TN / 32 ops (group
    //   GS u + e / NH, 32-column half e % NH), each into its own int32 tile;
    //   the epilogue then folds the unit's groups in ascending order, each over
    //   its halves, with the two FMAs of `sourceNarrowInt8`. KH 16 / 32 / 64 /
    //   128, TN 32 / 64 / 128, GS 1 / 2 / 4 (E <= 4), a PD-deep ring of whole
    //   units' words, a CD-deep ring of their constants (CD = 1: loaded at the
    //   unit's start, NP K steps before the epilogue reads them), and the
    //   reduction in passes of RC halves (the same sums in the same order, less
    //   threadgroup memory at TN 128).
    //   Templates: OutT, NEG, F32S, PD, TN, KH, CD, GS, RC.
    //   grid (N / TN * 128), (128).
    // - `sourceNarrowInt8PairR`: the pair body with R simdgroups per K quarter
    //   (R = 2 .. 5): simdgroup q + 4 r takes groups g0 + r + R t; at step t
    //   the helpers hand their int32 tiles over through XS slots per quarter
    //   (ceil((R - 1) / XS) rounds of two barriers) and the owner folds the
    //   step's groups g0 + R t, g0 + R t + 1, .. in ascending order, so the
    //   owner's sequence is (b) exactly with R loads in flight per column block.
    //   The reduction reuses the exchange. Templates: OutT, NEG, F32S, PD, KH,
    //   R, XS, CD. grid (N / 32 * 128 R), (128 R).
    // Both read the tiled copy only. The zoo 2 reduction keeps one `if (sg ==
    // 0)` block and one store, so the fused head top two (`headTop2Source`)
    // applies to it as to the others (its running state at `HT2 STATE`).
    private static let sourceNarrowInt8Zoo2 = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * TN;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int gper = Kg / 4;
        const int g0 = int(sg) * gper;
        const int g1 = g0 + gper;
        const int nu = (gper + GS - 1) / GS;   // units: GS consecutive groups of this quarter
        constexpr int NH = TN / 32;
        constexpr int E = GS * NH;             // ops per K step: op e takes group GS u + e / NH, half e % NH
        constexpr int NP = 128 / KH;           // K steps per 128-group
        constexpr int KW = KH / 16;            // 2-bit words per column per step
        constexpr int CDD = CD > 0 ? CD : (PD < 2 ? 2 : PD);  // constants ring depth (units)
        constexpr int UR = CDD % PD == 0 ? CDD : (PD % CDD == 0 ? PD : PD * CDD);  // unit-loop unroll
        constexpr int CAP = 32 / 2;
        constexpr int STW = 32 * KH / 4;       // staging words per op
        constexpr int SWS = E * STW;           // staging words per simdgroup
        constexpr int RH = RC > 0 ? RC : NH;   // 32-column halves per reduction pass
        constexpr int RDW = 3 * RH * CAP * 32; // reduction words
        static_assert(E <= 4 && NH % RH == 0, "at most four ops per step");
        // staging, reused by the reduction: sized for whichever is larger
        threadgroup uint32_t bs[4 * SWS > RDW ? 4 * SWS : RDW];
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, KH, false, true, false, NP > 1 ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate : mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)x, dextents<int, 2>(K, M));  // SIGNED codes only
        threadgroup uint32_t* sb = bs + int(sg) * SWS;
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)sb, dextents<int, 2>(KH, 32));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B1((threadgroup int8_t*)(sb + (E > 1 ? 1 : 0) * STW), dextents<int, 2>(KH, 32));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B2((threadgroup int8_t*)(sb + (E > 2 ? 2 : 0) * STW), dextents<int, 2>(KH, 32));
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B3((threadgroup int8_t*)(sb + (E > 3 ? 3 : 0) * STW), dextents<int, 2>(KH, 32));
        auto tA0 = A.template slice<KH, 16>(0, 0);
        auto cT0 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        auto cT1 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        auto cT2 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        auto cT3 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[NH][CAP];
        #pragma clang loop unroll(full)
        for (int h = 0; h < NH; h++) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { acc[h][i] = 0.0f; }
        }
        // Tiled copy: block n0 / 32 + h, group gg, column lane: 8 contiguous words.
        const device uint32_t* wrow = w + (size_t)(n0 / 32) * (size_t)Kg * 256 + (size_t)int(lane) * 8;
        const size_t hstride = (size_t)Kg * 256;
        // all 8 words of every op of unit u (a missing last group is skipped)
        auto getw = [&](int u, thread uint32_t (&v)[E][8]) {
          #pragma clang loop unroll(full)
          for (int e = 0; e < E; e++) {
            const int g = g0 + GS * u + e / NH;
            if (GS > 1 && g >= g1) { continue; }
            const device uint4* src = (const device uint4*)(wrow + (e % NH) * hstride + (size_t)g * 256);
            const uint4 u0 = src[0];
            const uint4 u1 = src[1];
            v[e][0] = u0.x; v[e][1] = u0.y; v[e][2] = u0.z; v[e][3] = u0.w;
            v[e][4] = u1.x; v[e][5] = u1.y; v[e][6] = u1.z; v[e][7] = u1.w;
          }
        };
        // stages K step p of unit u's words (words p * KW .. p * KW + KW - 1): one uint4 store per word
        auto putw = [&](thread const uint32_t (&v)[E][8], int u, int p) {
          #pragma clang loop unroll(full)
          for (int e = 0; e < E; e++) {
            if (GS > 1 && g0 + GS * u + e / NH >= g1) { continue; }
            threadgroup uint32_t* dst = sb + e * STW + int(lane) * (KH / 4);
            #pragma clang loop unroll(full)
            for (int jj = 0; jj < KW; jj++) {
              const uint32_t wv = v[e][p * KW + jj];
              *(threadgroup uint4*)(dst + 4 * jj) = uint4(
                  wv & 0x03030303u, (wv >> 2) & 0x03030303u,
                  (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
            }
          }
        };
        // Word ring: unit i sits in slot i % PD from its load until its last K step is
        // staged; the slot is then refilled with unit i + PD. Constants ring: slot
        // (u % CDD) holds unit u's groups' constants, loaded CDD - 1 units ahead, kept
        // in their stored types until use.
        uint32_t wr[PD][E][8];
        half4 shr[CDD][GS][NH][2], bhr[CDD][GS][NH][2];
        float4 sfr[CDD][GS][NH][2];
        float cr[CDD][GS][4];
        auto getc = [&](int u, int s) {
          #pragma clang loop unroll(full)
          for (int ge = 0; ge < GS; ge++) {
            const int g = g0 + GS * u + ge;
            if (GS > 1 && g >= g1) { continue; }
            #pragma clang loop unroll(full)
            for (int h = 0; h < NH; h++) {
              #pragma clang loop unroll(full)
              for (int q = 0; q < 2; q++) {
                const size_t o = (size_t)g * N + n0 + 32 * h + fn + 16 * q;
                if constexpr (F32S) { sfr[s][ge][h][q] = *(const device float4*)(scalesT + o); }
                else { shr[s][ge][h][q] = *(const device half4*)(scalesT + o); }
                if constexpr (!NEG) { bhr[s][ge][h][q] = *(const device half4*)(biasesT + o); }
              }
            }
            cr[s][ge][0] = ascale[(size_t)fm * Kg + g]; cr[s][ge][1] = ascale[(size_t)(fm + 8) * Kg + g];
            cr[s][ge][2] = rowsum[(size_t)fm * Kg + g]; cr[s][ge][3] = rowsum[(size_t)(fm + 8) * Kg + g];
          }
        };
        // K step p of unit u: every op's staged KH x 32 codes
        auto mm = [&](int u, int p) {
          #pragma clang loop unroll(full)
          for (int e = 0; e < E; e++) {
            const int g = g0 + GS * u + e / NH;
            if (GS > 1 && g >= g1) { continue; }
            auto tA = A.template slice<KH, 16>(g * 128 + p * KH, 0);
            if (e == 0) { op.run(tA, B0, cT0); }
            else if (e == 1) { op.run(tA, B1, cT1); }
            else if (e == 2) { op.run(tA, B2, cT2); }
            else { op.run(tA, B3, cT3); }
          }
        };
        // unit u at unrolled position j (a constant): ring slots are static
        auto body = [&](int u, int j) {
          const int cs = j % CDD;
          const int ws = j % PD;
          if (u + CDD - 1 < nu) { getc(u + CDD - 1, (j + CDD - 1) % CDD); }
          if constexpr (NP > 1) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i++) { cT0[i] = 0; cT1[i] = 0; cT2[i] = 0; cT3[i] = 0; }
          }
          #pragma clang loop unroll(full)
          for (int p = 0; p < NP; p++) {
            if (p > 0) {
              simdgroup_barrier(mem_flags::mem_threadgroup);
              putw(wr[ws], u, p);
              simdgroup_barrier(mem_flags::mem_threadgroup);
              if (p == NP - 1 && u + PD < nu) { getw(u + PD, wr[ws]); }
            }
            mm(u, p);
          }
          // the unit's groups in ascending order, each over its halves: every output
          // still folds its quarter's groups one by one in ascending order
          #pragma clang loop unroll(full)
          for (int ge = 0; ge < GS; ge++) {
            if (GS > 1 && g0 + GS * u + ge >= g1) { continue; }
            #pragma clang loop unroll(full)
            for (int h = 0; h < NH; h++) {
              const int e = ge * NH + h;
              float4 sv[2], bv[2];
              #pragma clang loop unroll(full)
              for (int q = 0; q < 2; q++) {
                if constexpr (F32S) { sv[q] = sfr[cs][ge][h][q]; } else { sv[q] = float4(shr[cs][ge][h][q]); }
                if constexpr (NEG) { bv[q] = -sv[q]; } else { bv[q] = float4(bhr[cs][ge][h][q]); }
              }
              #pragma clang loop unroll(full)
              for (int i = 0; i < CAP; i++) {
                const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
                const int32_t ci = e == 0 ? cT0[i] : (e == 1 ? cT1[i] : (e == 2 ? cT2[i] : cT3[i]));
                acc[h][i] = fma(mh ? cr[cs][ge][1] : cr[cs][ge][0], sv[nq][c] * float(ci), fma(bv[nq][c], mh ? cr[cs][ge][3] : cr[cs][ge][2], acc[h][i]));
              }
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (u + 1 < nu) {
            const int wn = (j + 1) % PD;       // slot holding unit u + 1
            putw(wr[wn], u + 1, 0);
            if (NP == 1 && u + 1 + PD < nu) { getw(u + 1 + PD, wr[wn]); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
          }
        };
        #pragma clang loop unroll(full)
        for (int i = 0; i < PD; i++) {
          if (i < nu) { getw(i, wr[i]); }
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < CDD - 1; i++) {
          if (i < nu) { getc(i, i); }
        }
        putw(wr[0], 0, 0);
        if (NP == 1 && PD < nu) { getw(PD, wr[0]); }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int u = 0; u < nu; u += UR) {
          #pragma clang loop unroll(full)
          for (int j = 0; j < UR; j++) {
            if (u + j < nu) { body(u + j, j); }
          }
        }
        // the reduction reuses the staging buffers, in simdgroup order, RH halves a pass
        threadgroup float* red = (threadgroup float*)bs;
        /* HT2 STATE */
        #pragma clang loop unroll(full)
        for (int rp = 0; rp < NH / RH; rp++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (sg > 0) {
            #pragma clang loop unroll(full)
            for (int hh = 0; hh < RH; hh++) {
              #pragma clang loop unroll(full)
              for (int i = 0; i < CAP; i++) { red[((int(sg) - 1) * RH + hh) * (CAP * 32) + i * 32 + int(lane)] = acc[rp * RH + hh][i]; }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (sg == 0) {
            #pragma clang loop unroll(full)
            for (int hh = 0; hh < RH; hh++) {
              const int h = rp * RH + hh;
              #pragma clang loop unroll(full)
              for (int i = 0; i < CAP; i += 4) {
                const int mh = (i >> 2) & 1;
                const int nq = i >> 3;
                float v0 = acc[h][i];
                float v1 = acc[h][i + 1];
                float v2 = acc[h][i + 2];
                float v3 = acc[h][i + 3];
                #pragma clang loop unroll(full)
                for (int q = 0; q < 4 - 1; q++) {
                  v0 += red[(q * RH + hh) * (CAP * 32) + i * 32 + int(lane)];
                  v1 += red[(q * RH + hh) * (CAP * 32) + (i + 1) * 32 + int(lane)];
                  v2 += red[(q * RH + hh) * (CAP * 32) + (i + 2) * 32 + int(lane)];
                  v3 += red[(q * RH + hh) * (CAP * 32) + (i + 3) * 32 + int(lane)];
                }
                const size_t base = (size_t)(fm + 8 * mh) * N + n0 + 32 * h + fn + 16 * nq;
                if constexpr (sizeof(OutT) == sizeof(float)) {
                  *(device float4*)(out + base) = float4(v0, v1, v2, v3);
                } else {
                  *(device half4*)(out + base) = half4(half(v0), half(v1), half(v2), half(v3));
                }
              }
            }
          }
        }
        """

    private static let sourceNarrowInt8PairR = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int qd = int(sg) & 3;           // K quarter, as in `sourceNarrowInt8`
        const int role = int(sg) >> 2;        // 0: owner (the FP32 sum), 1 .. R - 1: helpers
        const int gper = Kg / 4;
        const int g0 = qd * gper;
        const int g1 = g0 + gper;
        const int ns = (gper + R - 1) / R;    // steps: step t holds groups g0 + R t .. g0 + R t + R - 1
        constexpr int NP = 128 / KH;          // K steps per 128-group
        constexpr int KW = KH / 16;           // 2-bit words per column per step
        constexpr int CAP = 32 / 2;
        constexpr int SWS = 32 * KH / 4;      // staging words per simdgroup
        constexpr int XN = XS > 0 ? XS : R - 1;  // exchange slots per quarter
        constexpr int XR = (R - 2) / XN + 1;     // exchange rounds per step
        constexpr int CDD = CD > 0 ? CD : 2;     // the owner's constants ring depth (steps)
        constexpr int UT = PD > CDD ? PD : CDD;  // step-loop unroll
        static_assert(R >= 2 && R <= 5 && XN <= R - 1 && UT % PD == 0 && UT % CDD == 0, "two to five simdgroups per quarter");
        threadgroup uint32_t bs[4 * R * SWS];
        // each quarter's helper tiles, XN at a time; the reduction reuses it
        threadgroup int32_t xch[XN * 4 * CAP * 32];
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, KH, false, true, false, NP > 1 ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate : mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)x, dextents<int, 2>(K, M));  // SIGNED codes only
        threadgroup uint32_t* sb = bs + int(sg) * SWS;
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)sb, dextents<int, 2>(KH, 32));
        auto tA0 = A.template slice<KH, 16>(0, 0);
        auto cT0 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        // Tiled copy: block n0 / 32, group gg, column lane: 8 contiguous words.
        const device uint32_t* wrow = w + (size_t)(n0 / 32) * (size_t)Kg * 256 + (size_t)int(lane) * 8;
        auto getw = [&](int gg, thread uint32_t (&v)[8]) {
          const device uint4* src = (const device uint4*)(wrow + (size_t)gg * 256);
          const uint4 u0 = src[0];
          const uint4 u1 = src[1];
          v[0] = u0.x; v[1] = u0.y; v[2] = u0.z; v[3] = u0.w;
          v[4] = u1.x; v[5] = u1.y; v[6] = u1.z; v[7] = u1.w;
        };
        auto putw = [&](thread const uint32_t (&v)[8], int p) {
          threadgroup uint32_t* dst = sb + int(lane) * (KH / 4);
          #pragma clang loop unroll(full)
          for (int jj = 0; jj < KW; jj++) {
            const uint32_t wv = v[p * KW + jj];
            *(threadgroup uint4*)(dst + 4 * jj) = uint4(
                wv & 0x03030303u, (wv >> 2) & 0x03030303u,
                (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
          }
        };
        // The owner's constants ring: step t's R groups in slot t % CDD, loaded CDD - 1
        // steps ahead, kept in their stored types until use.
        half4 shr[CDD][R][2], bhr[CDD][R][2];
        float4 sfr[CDD][R][2];
        float cr[CDD][R][4];
        auto stepc = [&](int t, int s) {
          #pragma clang loop unroll(full)
          for (int e = 0; e < R; e++) {
            const int g = g0 + R * t + e;
            if (g >= g1) { continue; }
            #pragma clang loop unroll(full)
            for (int q = 0; q < 2; q++) {
              const size_t o = (size_t)g * N + n0 + fn + 16 * q;
              if constexpr (F32S) { sfr[s][e][q] = *(const device float4*)(scalesT + o); }
              else { shr[s][e][q] = *(const device half4*)(scalesT + o); }
              if constexpr (!NEG) { bhr[s][e][q] = *(const device half4*)(biasesT + o); }
            }
            cr[s][e][0] = ascale[(size_t)fm * Kg + g]; cr[s][e][1] = ascale[(size_t)(fm + 8) * Kg + g];
            cr[s][e][2] = rowsum[(size_t)fm * Kg + g]; cr[s][e][3] = rowsum[(size_t)(fm + 8) * Kg + g];
          }
        };
        // one group's epilogue, the FP32 sequence of `sourceNarrowInt8`
        auto epi = [&](int s, int e, thread const int32_t (&ci)[CAP]) {
          float4 sv[2], bv[2];
          #pragma clang loop unroll(full)
          for (int q = 0; q < 2; q++) {
            if constexpr (F32S) { sv[q] = sfr[s][e][q]; } else { sv[q] = float4(shr[s][e][q]); }
            if constexpr (NEG) { bv[q] = -sv[q]; } else { bv[q] = float4(bhr[s][e][q]); }
          }
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int cc = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
            acc[i] = fma(mh ? cr[s][e][1] : cr[s][e][0], sv[nq][cc] * float(ci[i]), fma(bv[nq][cc], mh ? cr[s][e][3] : cr[s][e][2], acc[i]));
          }
        };
        // This simdgroup's groups: g0 + role + R t (when < g1). Word ring: its group of
        // step t in slot t % PD until the last K step is staged, then that of step t + PD.
        uint32_t wr[PD][8];
        // step t at unrolled position u (a constant)
        auto step = [&](int t, int u) {
          const int gm = g0 + role + R * t;
          const int ws = u % PD;
          const int cs = u % CDD;
          if (role == 0 && t + CDD - 1 < ns) { stepc(t + CDD - 1, (u + CDD - 1) % CDD); }
          if (gm < g1) {
            if constexpr (NP > 1) {
              #pragma clang loop unroll(full)
              for (int i = 0; i < CAP; i++) { cT0[i] = 0; }
            }
            #pragma clang loop unroll(full)
            for (int p = 0; p < NP; p++) {
              if (p > 0) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
                putw(wr[ws], p);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (p == NP - 1 && gm + R * PD < g1) { getw(gm + R * PD, wr[ws]); }
              }
              auto tA = A.template slice<KH, 16>(gm * 128 + p * KH, 0);
              op.run(tA, B0, cT0);
            }
          }
          // XR rounds: helpers 1 + XN x .. XN + XN x hand their tiles over in round x,
          // and the owner folds the step's groups in ascending order: its own first,
          // then the helpers'
          #pragma clang loop unroll(full)
          for (int xr = 0; xr < XR; xr++) {
            if (role > 0 && (role - 1) / XN == xr && gm < g1) {
              #pragma clang loop unroll(full)
              for (int i = 0; i < CAP; i++) { xch[((((role - 1) % XN) * 4 + qd) * CAP + i) * 32 + int(lane)] = cT0[i]; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (role == 0) {
              int32_t ci[CAP];
              if (xr == 0) {
                #pragma clang loop unroll(full)
                for (int i = 0; i < CAP; i++) { ci[i] = cT0[i]; }
                epi(cs, 0, ci);
              }
              #pragma clang loop unroll(full)
              for (int x = 0; x < XN; x++) {
                const int e = 1 + xr * XN + x;
                if (e < R && gm + e < g1) {
                  #pragma clang loop unroll(full)
                  for (int i = 0; i < CAP; i++) { ci[i] = xch[((x * 4 + qd) * CAP + i) * 32 + int(lane)]; }
                  epi(cs, e, ci);
                }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
          if (gm + R < g1) {
            const int wn = (u + 1) % PD;       // slot holding this simdgroup's group of step t + 1
            putw(wr[wn], 0);
            if (NP == 1 && gm + R + R * PD < g1) { getw(gm + R + R * PD, wr[wn]); }
            simdgroup_barrier(mem_flags::mem_threadgroup);
          }
        };
        #pragma clang loop unroll(full)
        for (int i = 0; i < PD; i++) {
          if (g0 + role + R * i < g1) { getw(g0 + role + R * i, wr[i]); }
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < CDD - 1; i++) {
          if (role == 0 && i < ns) { stepc(i, i); }
        }
        if (g0 + role < g1) {
          putw(wr[0], 0);
          if (NP == 1 && g0 + role + R * PD < g1) { getw(g0 + role + R * PD, wr[0]); }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < ns; t += UT) {
          #pragma clang loop unroll(full)
          for (int u = 0; u < UT; u++) {
            if (t + u < ns) { step(t + u, u); }
          }
        }
        // the owners' partials, reduced as in `sourceNarrowInt8` (the exchange is free)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* red = (threadgroup float*)xch;
        if (role == 0 && qd > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { red[(qd - 1) * (CAP * 32) + i * 32 + int(lane)] = acc[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i += 4) {
            const int mh = (i >> 2) & 1;
            const int nq = i >> 3;
            float v0 = acc[i];
            float v1 = acc[i + 1];
            float v2 = acc[i + 2];
            float v3 = acc[i + 3];
            #pragma clang loop unroll(full)
            for (int q = 0; q < 4 - 1; q++) {
              v0 += red[q * (CAP * 32) + i * 32 + int(lane)];
              v1 += red[q * (CAP * 32) + (i + 1) * 32 + int(lane)];
              v2 += red[q * (CAP * 32) + (i + 2) * 32 + int(lane)];
              v3 += red[q * (CAP * 32) + (i + 3) * 32 + int(lane)];
            }
            const size_t base = (size_t)(fm + 8 * mh) * N + n0 + fn + 16 * nq;
            if constexpr (sizeof(OutT) == sizeof(float)) {
              *(device float4*)(out + base) = float4(v0, v1, v2, v3);
            } else {
              *(device half4*)(out + base) = half4(half(v0), half(v1), half(v2), half(v3));
            }
          }
        }
        """

    private static let kernelNarrowInt8Zoo = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_i8z",
        inputNames: ["x", "w", "scalesT", "biasesT", "ascale", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrowInt8Zoo,
        header: header,
        ensureRowContiguous: true)

    private static let kernelNarrowInt8Pair = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_i8x",
        inputNames: ["x", "w", "scalesT", "biasesT", "ascale", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrowInt8Pair,
        header: header,
        ensureRowContiguous: true)

    private static let kernelNarrowInt8Zoo2 = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_i8z2",
        inputNames: ["x", "w", "scalesT", "biasesT", "ascale", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrowInt8Zoo2,
        header: header,
        ensureRowContiguous: true)

    private static let kernelNarrowInt8PairR = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_i8r",
        inputNames: ["x", "w", "scalesT", "biasesT", "ascale", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrowInt8PairR,
        header: header,
        ensureRowContiguous: true)

    // Zoo 3a (`NarrowVariant.xtg`): the zoo, zoo 2 or R-pair body with its
    // four K quarters spread across threadgroups, QT per threadgroup and
    // 4 / QT threadgroups per column block (grid.y), so an N = 5120 shape runs
    // 4 / QT times the threadgroups (640 at QT = 1) at QT / 4 of the
    // threadgroup memory. `narrowXTGSource` derives the text: the quarter
    // index takes grid.y, the staging (and exchange) arrays shrink to QT
    // quarters, and the reduction becomes a store of each quarter's FP32 fold
    // to `part` [4, 16, N]. Each quarter runs the base body's own text, so
    // (a) and (b) of the zoo's rule hold as written; `sourceNarrowXTGSum`
    // then adds the planes 0 + 1 + 2 + 3 per output (the reduction's order)
    // and stores as `sourceNarrowInt8` does: (c). Bitwise by the zoo's proof,
    // self-tested at load against `original` on every production shape.
    // Templates: NEG, F32S, the base's, QT. grid (N / 32 * 32 QT R, 4 / QT).
    static func narrowXTGSource(_ text: String, pair: Bool) -> String? {
        let y = "int(threadgroup_position_in_grid.y) * QT"
        let edits: [(String, String, Int)] = pair
            ? [
                ("const int qd = int(sg) & 3;", "const int qd = int(sg) % QT;", 1),
                ("const int role = int(sg) >> 2;", "const int role = int(sg) / QT;", 1),
                ("const int g0 = qd * gper;", "const int g0 = (\(y) + qd) * gper;", 1),
                ("bs[4 * R * SWS]", "bs[QT * R * SWS]", 1),
                ("xch[XN * 4 * CAP * 32]", "xch[XN * QT * CAP * 32]", 1),
                (" * 4 + qd) * CAP", " * QT + qd) * CAP", 2),
            ]
            : [
                ("const int g0 = int(sg) * gper;", "const int g0 = (\(y) + int(sg)) * gper;", 1),
                ("bs[4 * SWS > RDW ? 4 * SWS : RDW]", "bs[QT * SWS]", 1),
            ]
        var t = text
        for (from, to, count) in edits {
            guard t.components(separatedBy: from).count == count + 1 else { return nil }
            t = t.replacingOccurrences(of: from, with: to)
        }
        let cut = pair ? "// the owners' partials" : "// the reduction reuses the staging"
        guard t.components(separatedBy: cut).count == 2, let r = t.range(of: cut) else { return nil }
        let (who, q, nh, a) = pair ? ("role == 0", "qd", "1", "acc") : ("true", "int(sg)", "NH", "acc[h]")
        return String(t[..<r.lowerBound]) + """
            // zoo 3a: this quarter's FP32 fold to its plane of `part`
            device float* pq = part + (size_t)(\(y) + \(q)) * 16 * N;
            if (\(who)) {
              #pragma clang loop unroll(full)
              for (int h = 0; h < \(nh); h++) {
                #pragma clang loop unroll(full)
                for (int i = 0; i < CAP; i += 4) {
                  const size_t base = (size_t)(fm + 8 * ((i >> 2) & 1)) * N + n0 + 32 * h + fn + 16 * (i >> 3);
                  *(device float4*)(pq + base) = float4(\(a)[i], \(a)[i + 1], \(a)[i + 2], \(a)[i + 3]);
                }
              }
            }

            """
    }

    /// Zoo 3a's sum: four consecutive outputs a thread. grid (16 N / 4), (256).
    private static let sourceNarrowXTGSum = """
        const size_t plane = (size_t)16 * ksz[2];
        const size_t e = (size_t)thread_position_in_grid.x * 4;
        if (e >= plane) { return; }
        float4 v = *(const device float4*)(part + e);
        v += *(const device float4*)(part + plane + e);
        v += *(const device float4*)(part + 2 * plane + e);
        v += *(const device float4*)(part + 3 * plane + e);
        if constexpr (sizeof(OutT) == sizeof(float)) {
          *(device float4*)(out + e) = v;
        } else {
          *(device half4*)(out + e) = half4(half(v.x), half(v.y), half(v.z), half(v.w));
        }
        """

    /// Zoo 3a's bodies from the zoo, zoo 2 and R-pair texts (nil: an anchor
    /// moved; the variants on it are then not offered).
    private static let kernelNarrowXTG: [MLXFast.MLXFastKernel?] = [
        (sourceNarrowInt8Zoo, false), (sourceNarrowInt8Zoo2, false), (sourceNarrowInt8PairR, true),
    ].enumerated().map { base in
        narrowXTGSource(base.element.0, pair: base.element.1).map {
            MLXFast.metalKernel(
                name: "bonsai_tensor_packed_matmul_m16_i8q\(base.offset)",
                inputNames: ["x", "w", "scalesT", "biasesT", "ascale", "rowsum", "ksz"],
                outputNames: ["part"], source: $0, header: header, ensureRowContiguous: true)
        }
    }

    private static let kernelNarrowXTGSum = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_i8q_sum", inputNames: ["part", "ksz"],
        outputNames: ["out"], source: sourceNarrowXTGSum, ensureRowContiguous: true)

    private static let kernelNarrowInt8Pipelined = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_i8p",
        inputNames: ["x", "w", "scalesT", "biasesT", "ascale", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrowInt8Pipelined,
        header: header,
        ensureRowContiguous: true)

    private static let kernelNarrowInt8 = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_i8",
        inputNames: ["x", "w", "scalesT", "biasesT", "ascale", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrowInt8,
        header: header,
        ensureRowContiguous: true)

    private static let kernelNarrowStaged8 = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_s8",
        inputNames: ["x", "w", "scalesT", "biasesT", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrowStaged8,
        header: header,
        ensureRowContiguous: true)

    // The prompt-width kernel for a toolchain without `uint2b_format`: the
    // same math, the 64 x 128 weight tile of each 128-group expanded from the
    // stored 2-bit words to 4-bit codes in threadgroup memory (double-buffered,
    // one barrier per group) and read by the op as a threadgroup
    // `uint4b_format` tensor. Same inputs as `source`.
    private static let sourceStaged = """
        const int K = ksz[0]; const int M = ksz[1]; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 64;
        const int m0 = int(threadgroup_position_in_grid.y) * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 64, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
        tensor<device uint8_t, dextents<int, 2>, tensor_inline> A((device uint8_t*)xq, dextents<int, 2>(K, M));
        // staged B: 64 columns x 128 codes as 4-bit, 2 per byte, k inner: byte index (n * 128 + k) / 2
        threadgroup uint32_t bs[2][64 * 128 / 8];
        tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B0((threadgroup uchar*)bs[0], dextents<int, 2>(128, 64));
        tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B1((threadgroup uchar*)bs[1], dextents<int, 2>(128, 64));
        auto tA0 = A.template slice<128, 64>(0, m0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        constexpr int CAP = 32;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        const int nb = n0 + 16 * int(sg & 1) + fn;
        const int mb = m0 + 16 * int(sg >> 1) + fm;
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 32);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 32);
        const device float4* up0 = (const device float4*)(uT + nb);
        const device float4* up1 = (const device float4*)(uT + nb + 32);
        const int NQ = N / 4;
        const size_t mrow[4] = {(size_t)mb, (size_t)(mb + 8), (size_t)(mb + 32), (size_t)(mb + 40)};
        // Row-tiled constants: this lane's four rows are adjacent in the tile.
        const size_t tbase = (size_t)(m0 / 64) * (size_t)Kg * 64 + (size_t)((8 * int(sg >> 1) + fm) * 4);
        // staging assignment: thread t -> column c = t >> 1, K half h = t & 1 (64 codes = 4 words -> 8 words of nibbles)
        const int sc = int(tid >> 1); const int sh = int(tid & 1);
        const device uint32_t* wrow = w + (size_t)(n0 + sc) * (K / 16) + sh * 4;
        auto stage = [&](int g, int buf) {
          const device uint4* src = (const device uint4*)(wrow + g * 8);
          const uint4 v = *src;
          threadgroup uint32_t* dst = bs[buf] + sc * 16 + sh * 8;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; j++) {
            const uint32_t wv = v[j];
            uint32_t lo = wv & 0xFFFFu; uint32_t hi = wv >> 16;
            lo = (lo | (lo << 8)) & 0x00FF00FFu; lo = (lo | (lo << 4)) & 0x0F0F0F0Fu; lo = (lo | (lo << 2)) & 0x33333333u;
            hi = (hi | (hi << 8)) & 0x00FF00FFu; hi = (hi | (hi << 4)) & 0x0F0F0F0Fu; hi = (hi | (hi << 2)) & 0x33333333u;
            dst[2 * j] = lo; dst[2 * j + 1] = hi;
          }
        };
        stage(0, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < Kg; g++) {
          const int cur = g & 1;
          if (g + 1 < Kg) { stage(g + 1, cur ^ 1); }
          auto tA = A.template slice<128, 64>(g * 128, m0);
          if (cur == 0) { op.run(tA, B0, cT); } else { op.run(tA, B1, cT); }
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = float4(bp0[g * NQ]), b1 = float4(bp1[g * NQ]);
          const float4 u0 = up0[g * NQ], u1 = up1[g * NQ];
          float as[4], rb[4];
          if (MPERM) {
            const float4 as4 = *(const device float4*)(ascale + tbase + (size_t)g * 64);
            const float4 rb4 = *(const device float4*)(rsb + tbase + (size_t)g * 64);
            as[0] = as4.x; as[1] = as4.y; as[2] = as4.z; as[3] = as4.w;
            rb[0] = rb4.x; rb[1] = rb4.y; rb[2] = rb4.z; rb[3] = rb4.w;
          } else {
            #pragma clang loop unroll(full)
            for (int q = 0; q < 4; q++) { as[q] = ascale[mrow[q] * Kg + g]; rb[q] = rsb[mrow[q] * Kg + g]; }
          }
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1; const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            const float u = nh ? u1[c] : u0[c];
            const float t = fma(s, float(cT[i]), u);
            acc[i] = fma(b, rb[mh], fma(as[mh], t, acc[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Groups of four consecutive i share mm and nh with c=0..3, so the
        // four outputs are consecutive columns at nb + 32*nh. Same values as
        // the scalar loop; float4/half4 stores match OutT. nb = n0 + 16*(sg&1)
        // + fn with fn in {0,4,8,12} and N multiple of 64, so the base is
        // 4-element aligned. Hot path: support==staged8 (signed + FACTORED).
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i += 4) {
          const int nh = (i >> 3) & 1;
          const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
          const float v0 = acc[i];
          const float v1 = acc[i + 1];
          const float v2 = acc[i + 2];
          const float v3 = acc[i + 3];
          const size_t base = (size_t)mm * N + nb + 32 * nh;
          if constexpr (sizeof(OutT) == sizeof(float)) {
            *(device float4*)(out + base) = float4(v0, v1, v2, v3);
          } else {
            *(device half4*)(out + base) = half4(half(v0), half(v1), half(v2), half(v3));
          }
        }
        """

    // The prompt-width kernel with the 2-bit words expanded to int8 codes in
    // threadgroup memory (`uint8 x uint8 -> int32`; no packed format needed):
    // four AND/shift ops per word, the codes landing in a permuted K order
    // inside each 16-block (position p holds code 4 * (p % 4) + p / 4), which
    // the quantizing rotation writes its codes in (`PERM`), so the integer
    // dot product is unchanged. Same inputs as `source`.
    private static let sourceStaged8 = """
        const int K = ksz[0]; const int M = ksz[1]; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 64;
        const int m0 = int(threadgroup_position_in_grid.y) * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 64, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
        // SIGNED: the codes are int8 (q) and the products carry no offset.
        typedef typename metal::conditional<SIGNED != 0, int8_t, uint8_t>::type CodeT;
        tensor<device CodeT, dextents<int, 2>, tensor_inline> A((device CodeT*)xq, dextents<int, 2>(K, M));
        // staged B: 64 columns x 128 codes as int8 bytes, k inner
        threadgroup uint32_t bs[2][64 * 128 / 4];
        tensor<threadgroup CodeT, dextents<int, 2>, tensor_inline> B0((threadgroup CodeT*)bs[0], dextents<int, 2>(128, 64));
        tensor<threadgroup CodeT, dextents<int, 2>, tensor_inline> B1((threadgroup CodeT*)bs[1], dextents<int, 2>(128, 64));
        auto tA0 = A.template slice<128, 64>(0, m0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        constexpr int CAP = 32;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        const int nb = n0 + 16 * int(sg & 1) + fn;
        const int mb = m0 + 16 * int(sg >> 1) + fm;
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 32);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 32);
        const device float4* up0 = (const device float4*)(uT + nb);
        const device float4* up1 = (const device float4*)(uT + nb + 32);
        const int NQ = N / 4;
        const size_t mrow[4] = {(size_t)mb, (size_t)(mb + 8), (size_t)(mb + 32), (size_t)(mb + 40)};
        // Row-tiled constants: this lane's four rows are adjacent in the tile.
        const size_t tbase = (size_t)(m0 / 64) * (size_t)Kg * 64 + (size_t)((8 * int(sg >> 1) + fm) * 4);
        // staging assignment: thread t -> column c = t >> 1, K half h = t & 1 (64 codes = 4 words -> 8 words of nibbles)
        const int sc = int(tid >> 1); const int sh = int(tid & 1);
        // TILED: the tiled copy (`narrowTiledWeight`): column c of the tile
        // is column c & 31 of block (n0 + c) / 32, whose group g sits at
        // (block * Kg + g) * 256 words; 64 threads read each block's 1 KB.
        const device uint32_t* wrow = TILED
            ? w + (size_t)((n0 + sc) >> 5) * (size_t)Kg * 256 + (size_t)((n0 + sc) & 31) * 8 + sh * 4
            : w + (size_t)(n0 + sc) * (K / 16) + sh * 4;
        auto stage = [&](int g, int buf) {
          const device uint4* src = (const device uint4*)(wrow + (size_t)g * (TILED ? 256 : 8));
          const uint4 v = *src;
          threadgroup uint32_t* dst = bs[buf] + sc * 32 + sh * 16;
          // One word's four planes are four contiguous uint32s (16 codes).
          // The base is 16-uint32 aligned, so each plane group is one uint4
          // store. Values and positions match the four scalar stores.
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; j++) {
            const uint32_t wv = v[j];
            const uint4 codes = uint4(
                wv & 0x03030303u,
                (wv >> 2) & 0x03030303u,
                (wv >> 4) & 0x03030303u,
                (wv >> 6) & 0x03030303u);
            *(threadgroup uint4*)(dst + 4 * j) = codes;
          }
        };
        stage(0, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < Kg; g++) {
          const int cur = g & 1;
          if (g + 1 < Kg) { stage(g + 1, cur ^ 1); }
          auto tA = A.template slice<128, 64>(g * 128, m0);
          if (cur == 0) { op.run(tA, B0, cT); } else { op.run(tA, B1, cT); }
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          float4 b0, b1;
          if constexpr (NEGATIVE_SCALE_BIAS) {
            b0 = -s0; b1 = -s1;
          } else {
            b0 = float4(bp0[g * NQ]); b1 = float4(bp1[g * NQ]);
          }
          float4 u0 = 0.0f, u1 = 0.0f;
          if (!SIGNED) { u0 = up0[g * NQ]; u1 = up1[g * NQ]; }
          float as[4], rb[4];
          if (MPERM) {
            const float4 as4 = *(const device float4*)(ascale + tbase + (size_t)g * 64);
            const float4 rb4 = *(const device float4*)(rsb + tbase + (size_t)g * 64);
            as[0] = as4.x; as[1] = as4.y; as[2] = as4.z; as[3] = as4.w;
            rb[0] = rb4.x; rb[1] = rb4.y; rb[2] = rb4.z; rb[3] = rb4.w;
          } else {
            #pragma clang loop unroll(full)
            for (int q = 0; q < 4; q++) { as[q] = ascale[mrow[q] * Kg + g]; rb[q] = rsb[mrow[q] * Kg + g]; }
          }
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1; const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            const float u = nh ? u1[c] : u0[c];
            if constexpr (FACTORED != 0 && NEGATIVE_SCALE_BIAS != 0 && SIGNED != 0) {
              // offset = -scale: as*(s*C) + (-s)*rb == s*(as*C - rb), one
              // FMA fewer per element and group.
              acc[i] = fma(s, fma(as[mh], float(cT[i]), -rb[mh]), acc[i]);
            } else {
              const float t = SIGNED ? s * float(cT[i]) : fma(s, float(cT[i]), u);
              acc[i] = fma(b, rb[mh], fma(as[mh], t, acc[i]));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Groups of four consecutive i share mm and nh with c=0..3, so the
        // four outputs are consecutive columns at nb + 32*nh. Same values as
        // the scalar loop; float4/half4 stores match OutT. Hot path:
        // support==staged8 (signed + FACTORED). Alignment under tip N/nb guards.
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i += 4) {
          const int nh = (i >> 3) & 1;
          const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
          const float v0 = acc[i];
          const float v1 = acc[i + 1];
          const float v2 = acc[i + 2];
          const float v3 = acc[i + 3];
          const size_t base = (size_t)mm * N + nb + 32 * nh;
          if constexpr (sizeof(OutT) == sizeof(float)) {
            *(device float4*)(out + base) = float4(v0, v1, v2, v3);
          } else {
            *(device half4*)(out + base) = half4(half(v0), half(v1), half(v2), half(v3));
          }
        }
        """

    private static let kernelStaged8 = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_q8_u8",
        inputNames: ["xq", "w", "scalesT", "biasesT", "uT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: sourceStaged8,
        header: header,
        ensureRowContiguous: true)

    private static let kernelStaged = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_q8_u4",
        inputNames: ["xq", "w", "scalesT", "biasesT", "uT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: sourceStaged,
        header: header,
        ensureRowContiguous: true)

    // The verify-width kernel for the same toolchain: each simdgroup expands
    // its own 32 x 128 tile per group into its threadgroup region
    // (double-buffered, simdgroup-scoped sync). 32 columns, 4 simdgroups.
    private static let sourceNarrowStaged = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int gper = Kg / 4;
        const int g0 = int(sg) * gper;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device half, dextents<int, 2>, tensor_inline> A((device half*)x, dextents<int, 2>(K, M));
        // per-simdgroup staged tile: 32 columns x 128 codes as nibbles = 2 KB; double-buffered
        threadgroup uint32_t bs[4][2][32 * 128 / 8];
        tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B0((threadgroup uchar*)bs[sg][0], dextents<int, 2>(128, 32));
        tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B1((threadgroup uchar*)bs[sg][1], dextents<int, 2>(128, 32));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, float>();
        constexpr int CAP = 16;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        const device half4* sp0 = (const device half4*)(scalesT + n0 + fn);
        const device half4* sp1 = (const device half4*)(scalesT + n0 + fn + 16);
        const device half4* bp0 = (const device half4*)(biasesT + n0 + fn);
        const device half4* bp1 = (const device half4*)(biasesT + n0 + fn + 16);
        const int NQ = N / 4;
        // staging: lane l -> column l (32 columns), all 128 codes of the group = 8 words -> 16 nibble words
        const device uint32_t* wrow = w + (size_t)(n0 + int(lane)) * (K / 16);
        auto stage = [&](int g, int buf) {
          const device uint4* src = (const device uint4*)(wrow + g * 8);
          const uint4 v0 = src[0]; const uint4 v1 = src[1];
          threadgroup uint32_t* dst = bs[sg][buf] + int(lane) * 16;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 8; j++) {
            const uint32_t wv = (j < 4) ? v0[j] : v1[j - 4];
            uint32_t lo = wv & 0xFFFFu; uint32_t hi = wv >> 16;
            lo = (lo | (lo << 8)) & 0x00FF00FFu; lo = (lo | (lo << 4)) & 0x0F0F0F0Fu; lo = (lo | (lo << 2)) & 0x33333333u;
            hi = (hi | (hi << 8)) & 0x00FF00FFu; hi = (hi | (hi << 4)) & 0x0F0F0F0Fu; hi = (hi | (hi << 2)) & 0x33333333u;
            dst[2 * j] = lo; dst[2 * j + 1] = hi;
          }
        };
        stage(g0, 0);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = g0; g < g0 + gper; g++) {
          const int cur = (g - g0) & 1;
          if (g + 1 < g0 + gper) { stage(g + 1, cur ^ 1); }
          auto tA = A.template slice<128, 16>(g * 128, 0);
          if (cur == 0) { op.run(tA, B0, cT); } else { op.run(tA, B1, cT); }
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = float4(bp0[g * NQ]), b1 = float4(bp1[g * NQ]);
          const float rs0 = rowsum[(size_t)fm * Kg + g];
          const float rs1 = rowsum[(size_t)(fm + 8) * Kg + g];
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            acc[i] = fma(s, cT[i], fma(b, mh ? rs1 : rs0, acc[i]));
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup float red[3][16 * 32];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { red[sg - 1][i * 32 + lane] = acc[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const float v = acc[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
            out[(size_t)(fm + 8 * mh) * N + n0 + fn + c + 16 * nh] = OutT(v);
          }
        }
        """

    private static let kernelNarrowStaged = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_u4",
        inputNames: ["x", "w", "scalesT", "biasesT", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrowStaged,
        header: header,
        ensureRowContiguous: true)

    private static let dimsLock = NSLock()
    nonisolated(unsafe) private static var dims: [[Int]: MLXArray] = [:]
    private static func dimsArray(k: Int, m: Int, n: Int) -> MLXArray {
        dimsLock.withLock {
            if let cached = dims[[k, m, n]] { return cached }
            let array = MLXArray([Int32(k), Int32(m), Int32(n)])
            dims[[k, m, n]] = array
            return array
        }
    }

    /// The per-group sums of the 2-bit codes of `weight` (`[n, k / 16]`
    /// UInt32, 16 codes per word, LSB first), `[n, k / 128]` FP32: the count
    /// of set low bits plus twice the count of set high bits of each word.
    private static func codeSums(_ weight: MLXArray, k: Int) -> MLXArray {
        let n = weight.dim(0)
        func popcount(_ v: MLXArray) -> MLXArray {
            var x = v - ((v >> 1) & MLXArray(UInt32(0x5555_5555)))
            x = (x & MLXArray(UInt32(0x3333_3333))) + ((x >> 2) & MLXArray(UInt32(0x3333_3333)))
            x = (x + (x >> 4)) & MLXArray(UInt32(0x0F0F_0F0F))
            return (x * MLXArray(UInt32(0x0101_0101))) >> 24
        }
        let low = popcount(weight & MLXArray(UInt32(0x5555_5555)))
        let high = popcount((weight >> 1) & MLXArray(UInt32(0x5555_5555)))
        let perWord = low + high * MLXArray(UInt32(2))
        return perWord.reshaped(n, k / 128, 8).sum(axis: -1).asType(.float32)
    }

    // MARK: - Verify int8 kernel choice (epilogue form x pipeline)

    /// The epilogue of the verify int8 kernel. `base` loads the FP16 scale and
    /// offset of every group. `negativeBias` loads the scale only and takes
    /// `-scale` as the offset: exact wherever every offset's FP16 bits are its
    /// scale's with the sign flipped, which
    /// `HadamardConstantLayoutCache.biasesAreNegativeScales` proves per
    /// constant pair (all 402 pairs of this checkpoint). `negativeBiasF32Scales`
    /// also reads the scales pre-widened to FP32 (exact), one 16-byte load
    /// per four columns and no conversion. The products are the same FP32
    /// values bit for bit in every form.
    enum NarrowEpilogue: Int {
        case base = 0
        case negativeBias = 1
        case negativeBiasF32Scales = 2
    }

    /// The kernel body. `v0` is `sourceNarrowInt8` as recorded; `pdN` is
    /// `sourceNarrowInt8Pipelined` with a register ring of the next N groups'
    /// words (and a matching constants ring) loaded ahead of the op; `k64pdN`
    /// is the same with each group staged and multiplied in two K = 64 halves
    /// (half the threadgroup memory, twice the threadgroups per core); `tn64`
    /// is `pd1` over 64 columns per threadgroup (twice the threadgroup
    /// memory). All bitwise identical.
    enum NarrowVariant: Int, CaseIterable {
        case v0 = 0
        case pd1 = 1
        case pd2 = 2
        case tn64 = 3
        case pd3 = 4
        case pd4 = 5
        case k64pd1 = 6
        case k64pd2 = 7
        case k64pd3 = 8
        case k64pd4 = 9
        // Tiled zoo (`sourceNarrowInt8Zoo` / `sourceNarrowInt8Pair`), by family:
        // k32 (1 KB staging per simdgroup), wide (TN 64), acoop (A cooperative,
        // one group ahead), pair (eight simdgroups).
        case k32pd1 = 20
        case k32pd2 = 21
        case k32pd4 = 22
        case w64k64pd1 = 23
        case w64k32pd1 = 24
        case a128pd2 = 25
        case a64pd2 = 26
        case aw64pd1 = 27
        case pk32pd1 = 28
        case pk32pd2 = 29
        case pk64pd2 = 30
        // Zoo 2 (`sourceNarrowInt8Zoo2` / `sourceNarrowInt8PairR`): k16 (K 16
        // per op), a four-group ring over K 16 steps, TN 128, dual (two
        // consecutive groups per step, folded in order), pair with 16 (one
        // helper), 32 (three helpers; `x1`: one exchange slot, three rounds)
        // or 40 simdgroups (four helpers, two slots).
        case k16pd2 = 31
        case k16pd4 = 32
        case w128k32pd1 = 33
        case g2k32pd1 = 34
        case pk16pd2 = 35
        case p4k16pd1 = 36
        case p4k16x1 = 37
        case p5k16pd1 = 38
        // Zoo 3a (`narrowXTGSource`): the quarters across threadgroups, `x4`
        // one quarter per threadgroup, `x2` two; from the zoo body (k32, k128),
        // zoo 2 (k16) or the R-pair body (`p2`, `p4`: R simdgroups a quarter).
        case x4k32pd2 = 40
        case x4k128pd2 = 41
        case x2k32pd2 = 42
        case x4k16pd4 = 43
        case x4p2k32pd2 = 44
        case x4p4k16pd1 = 45
        case x2p2k32pd2 = 46

        /// Words ring depth, columns per threadgroup, K per op.
        var pd: Int {
            switch self {
            case .v0, .pd1, .tn64, .k64pd1, .k32pd1, .w64k64pd1, .w64k32pd1, .aw64pd1, .pk32pd1,
                .w128k32pd1, .g2k32pd1, .p4k16pd1, .p4k16x1, .p5k16pd1, .x4p4k16pd1:
                return 1
            case .pd2, .k64pd2, .k32pd2, .a128pd2, .a64pd2, .pk32pd2, .pk64pd2, .k16pd2, .pk16pd2,
                .x4k32pd2, .x4k128pd2, .x2k32pd2, .x4p2k32pd2, .x2p2k32pd2:
                return 2
            case .pd3, .k64pd3: return 3
            case .pd4, .k64pd4, .k32pd4, .k16pd4, .x4k16pd4: return 4
            }
        }
        var tn: Int {
            switch self {
            case .tn64, .w64k64pd1, .w64k32pd1, .aw64pd1: return 64
            case .w128k32pd1: return 128
            default: return 32
            }
        }
        var kh: Int {
            switch self {
            case .k64pd1, .k64pd2, .k64pd3, .k64pd4, .w64k64pd1, .a64pd2, .aw64pd1, .pk64pd2: return 64
            case .k32pd1, .k32pd2, .k32pd4, .w64k32pd1, .pk32pd1, .pk32pd2, .w128k32pd1, .g2k32pd1:
                return 32
            case .k16pd2, .k16pd4, .pk16pd2, .p4k16pd1, .p4k16x1, .p5k16pd1: return 16
            default: return 128
            }
        }
        /// The zoo family (nil: the record's bodies and K3).
        var family: String? {
            switch self {
            case .k32pd1, .k32pd2, .k32pd4, .k16pd2, .k16pd4: return "k32"
            case .w64k64pd1, .w64k32pd1, .w128k32pd1: return "wide"
            case .a128pd2, .a64pd2, .aw64pd1: return "acoop"
            case .pk32pd1, .pk32pd2, .pk64pd2, .pk16pd2, .p4k16pd1, .p4k16x1, .p5k16pd1: return "pair"
            case .g2k32pd1: return "dual"
            default: return xtg.map { $0.body == 2 ? "xtgp" : "xtg" }
            }
        }
        /// Zoo 3a: the base body (0 zoo, 1 zoo 2, 2 R-pair), quarters per
        /// threadgroup, simdgroups per quarter and the base's templates.
        var xtg: (body: Int, qt: Int, r: Int, t: [(String, Int)])? {
            let zoo: [(String, Int)] = [("PD", 2), ("TN", 32), ("KH", 32), ("AM", 0)]
            let pair: [(String, Int)] = [("PD", 2), ("KH", 32), ("R", 2), ("XS", 0), ("CD", 0)]
            switch self {
            case .x4k32pd2: return (0, 1, 1, zoo)
            case .x4k128pd2: return (0, 1, 1, [("PD", 2), ("TN", 32), ("KH", 128), ("AM", 0)])
            case .x2k32pd2: return (0, 2, 1, zoo)
            case .x4k16pd4: return (1, 1, 1, [("PD", 4), ("TN", 32), ("KH", 16), ("CD", 1), ("GS", 1), ("RC", 0)])
            case .x4p2k32pd2: return (2, 1, 2, pair)
            case .x4p4k16pd1: return (2, 1, 4, [("PD", 1), ("KH", 16), ("R", 4), ("XS", 0), ("CD", 1)])
            case .x2p2k32pd2: return (2, 2, 2, pair)
            default: return nil
            }
        }
        var am: Int { [.a128pd2, .a64pd2, .aw64pd1].contains(self) ? 2 : 0 }

        /// Zoo 2 bodies: `sourceNarrowInt8PairR` (true) or `sourceNarrowInt8Zoo2`
        /// (false); nil for every other body.
        var zoo2Pair: Bool? {
            switch self {
            case .k16pd2, .k16pd4, .w128k32pd1, .g2k32pd1: return false
            case .pk16pd2, .p4k16pd1, .p4k16x1, .p5k16pd1: return true
            default: return nil
            }
        }
        /// A zoo 2 body's templates after OutT / NEG / F32S: PD, TN, KH, CD
        /// (constants ring depth, 0: the words ring's), GS (groups per unit),
        /// RC (halves per reduction pass, 0: all) for `sourceNarrowInt8Zoo2`;
        /// PD, KH, R (simdgroups per quarter), XS (exchange slots, 0: R - 1),
        /// CD for `sourceNarrowInt8PairR`.
        var zoo2Template: [(String, Int)] {
            switch self {
            case .k16pd2: return [("PD", 2), ("TN", 32), ("KH", 16), ("CD", 0), ("GS", 1), ("RC", 0)]
            case .k16pd4: return [("PD", 4), ("TN", 32), ("KH", 16), ("CD", 1), ("GS", 1), ("RC", 0)]
            case .w128k32pd1:
                return [("PD", 1), ("TN", 128), ("KH", 32), ("CD", 1), ("GS", 1), ("RC", 2)]
            case .g2k32pd1: return [("PD", 1), ("TN", 32), ("KH", 32), ("CD", 1), ("GS", 2), ("RC", 0)]
            case .pk16pd2: return [("PD", 2), ("KH", 16), ("R", 2), ("XS", 0), ("CD", 0)]
            case .p4k16pd1: return [("PD", 1), ("KH", 16), ("R", 4), ("XS", 0), ("CD", 1)]
            case .p4k16x1: return [("PD", 1), ("KH", 16), ("R", 4), ("XS", 1), ("CD", 1)]
            case .p5k16pd1: return [("PD", 1), ("KH", 16), ("R", 5), ("XS", 2), ("CD", 1)]
            default: return []
            }
        }
        /// Threads per threadgroup: 32 per simdgroup; a pair body runs R
        /// simdgroups per K quarter (the zoo's pair bodies two).
        var threads: Int {
            switch self {
            case .pk32pd1, .pk32pd2, .pk64pd2, .pk16pd2: return 256
            case .p4k16pd1, .p4k16x1: return 512
            case .p5k16pd1: return 640
            default: return 128
            }
        }
        /// Shape classes a zoo body is tried on (`narrowShapeClass`): the new
        /// pair, dual and deep-ring bodies target the N = 5120 shapes only.
        var zooClasses: Set<Int> {
            switch self {
            case .k16pd4, .g2k32pd1, .pk16pd2, .p4k16pd1, .p4k16x1, .p5k16pd1, .x4p2k32pd2, .x4p4k16pd1,
                .x2p2k32pd2:
                return [1]
            default: return [1, 2]
            }
        }

        init?(name: String) {
            guard let v = Self.allCases.first(where: { "\($0)" == name }) else { return nil }
            self = v
        }
    }

    struct NarrowKernel: Hashable, CustomStringConvertible {
        var variant: NarrowVariant
        var form: NarrowEpilogue
        static let original = NarrowKernel(variant: .v0, form: .base)
        var description: String { "\(variant)/\(form)" }
    }

    /// The load-time choice (see `chooseNarrowKernels`): per production shape
    /// `[k, n]`, and a default for every other shape. `original` until then
    /// and wherever the int8 verify kernel is not installed.
    nonisolated(unsafe) static var narrowDefault = NarrowKernel.original
    nonisolated(unsafe) static var narrowByShape: [[Int]: NarrowKernel] = [:]
    /// Whether any chosen kernel needs the per-projection proof / FP32 scales.
    nonisolated(unsafe) static var narrowNeedsProof = false
    nonisolated(unsafe) static var narrowNeedsF32 = false

    /// One choice: a default kernel and per-shape kernels.
    typealias NarrowChoice = (NarrowKernel, [[Int]: NarrowKernel])

    /// The record's (dcaf489's) kernel bodies: its own autotune chooses among
    /// these (and `original`), so the pick over them is the record's pick.
    static let narrowRecordVariants: Set<NarrowVariant> = [.v0, .pd1, .pd2, .tn64]

    /// Sets `narrowNeedsProof` / `narrowNeedsF32` for these choices.
    static func setNarrowOperandNeeds(_ choices: [NarrowChoice]) {
        let all = choices.flatMap { [$0.0] + Array($0.1.values) }
        narrowNeedsProof = all.contains { $0.form != .base }
        narrowNeedsF32 = all.contains { $0.form == .negativeBiasF32Scales }
    }

    /// The in-situ choice of the verify int8 kernels: candidate choices
    /// (shortlisted by the synthetic self-test and timing at load) run in turn
    /// on real speculative rounds of the load-time engine warm, and a
    /// candidate replaces the record's pick only when its median round beats
    /// the record's pick's by more than 0.5 %. The synthetic timing streams
    /// each kernel for 0.5-0.7 ms bursts in which the GPU never reaches the
    /// clock state of the decode window, so it mispicks; a whole round does
    /// reach it. Every candidate passed the bitwise self-test at the output
    /// types its shapes take, so the rounds and their tokens are the same
    /// whichever runs.
    ///
    /// Up to `singleStageMax` sets (the base's trial, e.g. without the zoo):
    /// `roundsPerSet` rounds each, interleaved. More sets (the zoo's family
    /// sets): two stages in one request. Stage 1 runs every set for
    /// `stage1Rounds` rounds, interleaved. Stage 2 then takes up to
    /// `stage2Sets` candidates by their stage-1 median, where a set combining
    /// the best N = 5120-class set's bodies with the best wide-class set's is
    /// estimated as the sum of their gains, and runs them and the record's pick
    /// for `roundsPerSet` rounds each, interleaved; only stage 2's rounds
    /// decide. A candidate whose stage-1 estimate is more than `stage2Slack`
    /// slower than the record's pick is not taken; with none left, the
    /// record's pick is kept after stage 1.
    ///
    /// Per shape, between the stages (`perShapeKeys`): each production shape
    /// is one factor whose levels are the record's kernel on it and the two
    /// kernels the fastest stage-1 sets (within `stage2Slack`) put on it; the
    /// 18 runs of an L18 orthogonal array over the five factors, one round
    /// each, give every kernel's per-shape effect (its mean round minus the
    /// record kernel's; the array is balanced, so these are the additive
    /// model's least-squares effects). Stage 2 then runs the mix of each
    /// shape's best kernel first and ranks the next mix with the other
    /// candidates; the head follows the gate|up level's set. The adoption
    /// rule is unchanged. `DARKBLOOM_BONSAI_TENSOR_ROUTE_TZOO_SHAPE=0` skips
    /// this step (the trial of zoo 3a).
    ///
    /// `roundBoundary()` runs at the top of every block proposal and of every
    /// adopted speculative block (one static bool check when no trial is
    /// active): round r's time is the host time from its boundary to the next
    /// one, and the choice installed at a boundary is what the next verify
    /// graph build reads. The first round after the seed is discarded. Runs
    /// only inside the deferred load warm (`Qwen35DFlash2Assistant`), never in
    /// a served request. One stderr line names every set's medians and the
    /// set installed. `DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_INSITU=off` keeps
    /// the record's pick without a trial.
    enum NarrowInSituTrial {
        nonisolated(unsafe) static var active = false
        /// The candidate sets (the first is the record's pick), their names,
        /// and their scopes (0: any shape, 1: the zoo's bodies on the N = 5120
        /// class only, 2: on the wide class only; `narrowShapeClass`).
        nonisolated(unsafe) static var sets: [NarrowChoice] = []
        nonisolated(unsafe) static var labels: [String] = []
        nonisolated(unsafe) static var scopes: [Int] = []
        nonisolated(unsafe) static var roundIndex = 0
        nonisolated(unsafe) static var lastBoundary: UInt64 = 0
        nonisolated(unsafe) static var onEnough: (() -> Void)?
        // The running trial: `pool` holds the sets, then stage 2's combined
        // sets; a timing slot is one set in one stage; `schedule` holds the
        // slot of each timed round (round 1 on).
        nonisolated(unsafe) private static var pool: [NarrowChoice] = []
        nonisolated(unsafe) private static var poolLabels: [String] = []
        nonisolated(unsafe) private static var slotSet: [Int] = []
        nonisolated(unsafe) private static var slotTimes: [[UInt64]] = []
        nonisolated(unsafe) private static var schedule: [Int] = []
        nonisolated(unsafe) private static var stage1Slots = 0
        nonisolated(unsafe) private static var stage2Built = false
        nonisolated(unsafe) private static var stage2First = -1
        nonisolated(unsafe) private static var current = -1
        // Per shape: each factor's level kernels (the record's first), the
        // head's kernel per gate|up level, the first L18 slot, the log part.
        nonisolated(unsafe) private static var perShapeBuilt = false
        nonisolated(unsafe) private static var levels: [[NarrowKernel]] = []
        nonisolated(unsafe) private static var headLevels: [NarrowKernel] = []
        nonisolated(unsafe) private static var perShapeFirst = -1
        nonisolated(unsafe) private static var perShapeLog = ""

        /// The per-shape factors: attention qkv, GDN qkv|z, gate|up, o/out,
        /// down (the head follows gate|up).
        static let perShapeKeys: [[Int]] = [[5120, 14336], [5120, 16384], [5120, 34816], [6144, 5120], [17408, 5120]]
        static let perShapeNames = ["attn", "qkv|z", "gate|up", "o", "down"]
        static let headKey = [5120, 248320]
        /// L18: OA(18, 3^5) (columns 2-5 and 7 of the standard table; every
        /// two columns hold each level pair twice), rows ordered so no
        /// column's levels trend with the run order.
        static let perShapeArray: [[Int]] =
            "21010 02121 10202 10021 21102 02210 22222 00000 11111 22001 11220 00112 20120 12012 01201 12100 01022 20211"
            .split(separator: " ").map { $0.compactMap(\.wholeNumberValue) }
        static let perShapeEnabled: Bool = {
            let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_TZOO_SHAPE"]?
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            return !["0", "false", "no", "off"].contains(value ?? "")
        }()

        /// Timed rounds per candidate (one stage, and stage 2).
        static let roundsPerSet = 8
        /// Timed rounds per set in stage 1.
        static let stage1Rounds = 3
        /// Candidates stage 2 runs beside the record's pick.
        static let stage2Sets = 3
        /// Up to this many sets the trial is one stage (the base's).
        static let singleStageMax = 4
        /// A candidate must beat the record's pick by more than this.
        static let adoptMargin = 0.005
        /// Stage 2 takes a candidate only if its stage-1 estimate is at most
        /// this much slower than the record's pick's median.
        static let stage2Slack = 0.02
        /// Rounds above this multiple of their candidate's median are dropped.
        static let outlierFactor = 1.5
        /// A candidate is adopted only with at least this many kept rounds.
        static let minimumRounds = 4

        static let enabled: Bool = {
            let value = ProcessInfo.processInfo.environment[
                "DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_INSITU"]?
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            return !["0", "false", "no", "off"].contains(value ?? "")
        }()

        /// Whether a trial is set up (two or more distinct candidates; the
        /// first is the record's pick).
        static var armed: Bool { enabled && sets.count >= 2 }
        static var twoStage: Bool { sets.count > singleStageMax }

        /// Proposals the trial can need (the warm request's budget): the
        /// discarded round, the seed-side boundary, then every timed round.
        static var roundsNeeded: Int {
            twoStage
                ? 2 + stage1Rounds * sets.count + (perShapeEnabled ? perShapeArray.count : 0)
                    + roundsPerSet * (1 + stage2Sets)
                : 2 + roundsPerSet * sets.count
        }

        @inline(__always) static func roundBoundary() {
            guard active else { return }
            boundary()
        }

        private static func install(_ set: NarrowChoice) {
            narrowDefault = set.0
            narrowByShape = set.1
        }

        /// New timing slots for these pool sets, `rounds` each, interleaved
        /// (the first set's rounds last, as the base's rotation).
        private static func appendRounds(_ poolIndices: [Int], rounds: Int) {
            let first = slotSet.count
            for index in poolIndices {
                slotSet.append(index)
                slotTimes.append([])
            }
            let count = poolIndices.count
            schedule += (1 ... rounds * count).map { first + $0 % count }
        }

        private static func boundary() {
            let now = DispatchTime.now().uptimeNanoseconds
            if roundIndex >= 2, current >= 0 { slotTimes[current].append(now - lastBoundary) }
            lastBoundary = now
            if roundIndex >= 1, roundIndex - 1 >= schedule.count, twoStage {
                if !perShapeBuilt { buildPerShape() }
                if roundIndex - 1 >= schedule.count, !stage2Built { buildStage2() }
            }
            if roundIndex >= 1, roundIndex - 1 >= schedule.count {
                // every scheduled round is timed
                roundIndex += 1
                active = false
                let enough = onEnough
                onEnough = nil
                enough?()
                return
            }
            current = roundIndex == 0 ? -1 : schedule[roundIndex - 1]
            install(pool[current < 0 ? 0 : slotSet[current]])
            roundIndex += 1
        }

        /// Arms the round hook. `onEnough` runs (on the engine's thread) once
        /// every candidate has its rounds.
        static func begin(onEnough: @escaping () -> Void) {
            guard armed else { return }
            pool = sets
            poolLabels = sets.indices.map { $0 < labels.count ? labels[$0] : "set \($0)" }
            slotSet = []
            slotTimes = []
            schedule = []
            appendRounds(Array(sets.indices), rounds: twoStage ? stage1Rounds : roundsPerSet)
            stage1Slots = sets.count
            stage2Built = false
            stage2First = -1
            perShapeBuilt = false
            levels = []
            headLevels = []
            perShapeFirst = -1
            perShapeLog = ""
            current = -1
            roundIndex = 0
            lastBoundary = 0
            self.onEnough = onEnough
            active = true
        }

        private static func median(_ values: [UInt64]) -> Double? {
            guard !values.isEmpty else { return nil }
            let sorted = values.sorted()
            let mid = sorted.count / 2
            return sorted.count % 2 == 1
                ? Double(sorted[mid]) : (Double(sorted[mid - 1]) + Double(sorted[mid])) / 2
        }

        /// A slot's median over its rounds within `outlierFactor` of the raw
        /// median, the rounds kept and the rounds run.
        private static func summary(_ slot: Int) -> (Double?, Int, Int) {
            let times = slot < slotTimes.count ? slotTimes[slot] : []
            guard let first = median(times) else { return (nil, 0, times.count) }
            let kept = times.filter { Double($0) <= outlierFactor * first }
            return (median(kept), kept.count, times.count)
        }

        /// The kernel each production shape and the head resolve to.
        static func effective(_ choice: NarrowChoice) -> [NarrowKernel] {
            [choice.0] + (narrowTunedShapes + narrowZooShapes).map { choice.1[[$0.0, $0.1]] ?? choice.0 }
        }

        /// A kernel's name in the logs.
        static func name(_ kernel: NarrowKernel) -> String {
            kernel.variant.family == nil ? "\(kernel)" : "\(kernel.variant)"
        }

        /// A set's kernel on each per-shape factor and the head.
        static func mapping(_ set: NarrowChoice) -> String {
            zip(perShapeNames + ["head"], perShapeKeys + [headKey])
                .map { "\($0.0)=" + name(set.1[$0.1] ?? set.0) }.joined(separator: " ")
        }

        /// The level factor `f` takes at array entry `level` (a factor with
        /// fewer than three kernels repeats its last).
        private static func levelIndex(_ f: Int, _ level: Int) -> Int { min(level, levels[f].count - 1) }

        /// The record's pick with factor f at `row[f]` (the head per gate|up).
        private static func perShapeSet(_ row: [Int]) -> NarrowChoice {
            var map = pool[0].1
            for (f, key) in perShapeKeys.enumerated() { map[key] = levels[f][levelIndex(f, row[f])] }
            map[headKey] = headLevels[levelIndex(2, row[2])]
            return (pool[0].0, map)
        }

        /// The per-shape levels from stage 1's ranking and the L18 runs, one
        /// round each (see the type's notes). Nothing when no shape has a
        /// second kernel.
        private static func buildPerShape() {
            perShapeBuilt = true
            perShapeLog = "; per shape " + (perShapeEnabled ? "none (no second kernel on any shape)" : "off")
            guard perShapeEnabled, let record = summary(0).0 else { return }
            let ranked = (1 ..< stage1Slots).compactMap { slot in summary(slot).0.map { (slot, $0) } }
                .filter { $0.1 < record * (1 + stage2Slack) }.sorted { $0.1 < $1.1 }
            levels = []
            for (f, key) in perShapeKeys.enumerated() {
                var list = [pool[0].1[key] ?? pool[0].0]
                var heads = [pool[0].1[headKey] ?? pool[0].0]
                for (slot, _) in ranked where list.count < 3 {
                    let set = pool[slotSet[slot]]
                    let kernel = set.1[key] ?? set.0
                    if !list.contains(kernel) {
                        list.append(kernel)
                        heads.append(set.1[headKey] ?? set.0)
                    }
                }
                levels.append(list)
                if f == 2 { headLevels = heads }
            }
            guard levels.contains(where: { $0.count > 1 }) else { return }
            perShapeFirst = slotSet.count
            for row in perShapeArray {
                pool.append(perShapeSet(row))
                poolLabels.append("L18 " + row.map(String.init).joined())
                slotSet.append(pool.count - 1)
                slotTimes.append([])
            }
            schedule += Array(perShapeFirst ..< perShapeFirst + perShapeArray.count)
            perShapeLog = "; per shape (L18) not finished"
        }

        /// Per factor, each level's effect in ns per round against the
        /// record's kernel (nil: no kept run), from the L18 rounds within
        /// `outlierFactor` of their median; and the rounds kept.
        private static func perShapeEffects() -> ([[Double?]], Int) {
            guard perShapeFirst >= 0 else { return ([], 0) }
            let runs = perShapeArray.indices.compactMap { r -> (Int, UInt64)? in
                let slot = perShapeFirst + r
                return slot < slotTimes.count ? slotTimes[slot].first.map { (r, $0) } : nil
            }
            guard let mid = median(runs.map(\.1)) else { return ([], 0) }
            let kept = runs.filter { Double($0.1) <= outlierFactor * mid }
            let effects = levels.indices.map { f -> [Double?] in
                var sums = [Double](repeating: 0, count: levels[f].count)
                var counts = [Int](repeating: 0, count: levels[f].count)
                for (r, t) in kept {
                    let j = levelIndex(f, perShapeArray[r][f])
                    sums[j] += Double(t)
                    counts[j] += 1
                }
                guard counts[0] > 0 else { return levels[f].map { _ in nil } }
                return levels[f].indices.map { counts[$0] > 0 ? sums[$0] / Double(counts[$0]) - sums[0] / Double(counts[0]) : nil }
            }
            return (effects, kept.count)
        }

        /// Stage 2's candidates from stage 1's medians and the per-shape
        /// effects (see the type's notes).
        private static func buildStage2() {
            stage2Built = true
            let medians = (0 ..< stage1Slots).map { summary($0).0 }
            guard let record = medians[0] else { return }
            var candidates: [(Int, Double)] = []
            for index in 1 ..< stage1Slots {
                if let m = medians[index] { candidates.append((index, m)) }
            }
            func best(_ scope: Int) -> [(Int, Double)] {
                Array(
                    candidates.filter { $0.0 < scopes.count && scopes[$0.0] == scope }
                        .sorted { $0.1 < $1.1 }.prefix(2))
            }
            let narrow = best(1)
            let wide = best(2)
            for a in narrow {
                for b in wide {
                    var map = pool[a.0].1
                    for (key, kernel) in pool[b.0].1 where narrowShapeClass(key) == 2 { map[key] = kernel }
                    pool.append((pool[a.0].0, map))
                    poolLabels.append(poolLabels[a.0] + "+" + poolLabels[b.0])
                    candidates.append((pool.count - 1, a.1 + b.1 - record))
                }
            }
            // Per shape: the mix of each factor's best level, and the mix one
            // step from it (the swap that costs least), when predicted faster.
            var firstMix: Int?
            let (effects, kept) = perShapeEffects()
            if !effects.isEmpty {
                func gain(_ picks: [Int]) -> Double {
                    picks.enumerated().reduce(0) { $0 + (effects[$1.0][$1.1] ?? 0) }
                }
                let picks = effects.map { row -> Int in
                    var pick = 0
                    for (j, e) in row.enumerated() where j > 0 {
                        if let e, e < (row[pick] ?? 0) { pick = j }
                    }
                    return pick
                }
                var next: [Int]?
                for f in picks.indices {
                    for j in levels[f].indices where j != picks[f] && effects[f][j] != nil {
                        var alt = picks
                        alt[f] = j
                        if next == nil || gain(alt) < gain(next!) { next = alt }
                    }
                }
                for (label, mix) in [("shape1", picks), ("shape2", next)] {
                    guard let mix, gain(mix) < 0 else { continue }
                    pool.append(perShapeSet(mix))
                    poolLabels.append(label)
                    if label == "shape1" { firstMix = pool.count - 1 }
                    else { candidates.append((pool.count - 1, record + gain(mix))) }
                }
                var parts: [String] = []
                for f in levels.indices {
                    var cells: [String] = []
                    for (j, kernel) in levels[f].enumerated() {
                        let effect = effects[f][j].map { String(format: " %+.3f", $0 / 1e6) } ?? " -"
                        cells.append(name(kernel) + (j == 0 ? "" : effect))
                    }
                    parts.append(perShapeNames[f] + ": " + cells.joined(separator: " | "))
                }
                perShapeLog = "; per shape (L18, \(kept)/\(perShapeArray.count) rounds kept, ms vs the record's kernel) ["
                    + parts.joined(separator: "; ") + "]"
                if let firstMix {
                    perShapeLog += String(format: ", best mix %+.3f = ", gain(picks) / 1e6) + mapping(pool[firstMix])
                } else {
                    perShapeLog += ", no faster mix"
                }
            }
            candidates.sort { $0.1 < $1.1 }
            var chosen: [Int] = []
            let recordKernels = effective(pool[0])
            func take(_ index: Int) {
                let kernels = effective(pool[index])
                if kernels != recordKernels, !chosen.contains(where: { effective(pool[$0]) == kernels }) {
                    chosen.append(index)
                }
            }
            if let firstMix { take(firstMix) }
            for (index, estimate) in candidates
            where chosen.count < stage2Sets && estimate < record * (1 + stage2Slack) {
                take(index)
            }
            guard !chosen.isEmpty else { return }
            stage2First = slotSet.count
            appendRounds([0] + chosen, rounds: roundsPerSet)
        }

        static func describe(_ set: NarrowChoice) -> String {
            guard !set.1.isEmpty else { return name(set.0) }
            // the zoo's extra keys (o/out, head), named, when a set has them
            let extra = narrowZooShapes.compactMap { shape in
                set.1[[shape.0, shape.1]].map { "\(shape.0)x\(shape.1)=\(name($0))" }
            }
            return "\(name(set.0)){"
                + (narrowTunedShapes.map { name(set.1[[$0.0, $0.1]] ?? set.0) } + extra)
                .joined(separator: ",") + "}"
        }

        /// Ends the trial: installs the winner (or the record's pick), logs
        /// one line and disarms. Safe to call when nothing ran.
        static func finish(elapsedNanoseconds: UInt64) {
            active = false
            onEnough = nil
            guard armed else { return }
            if pool.isEmpty {
                pool = sets
                poolLabels = sets.indices.map { $0 < labels.count ? labels[$0] : "set \($0)" }
            }
            // The deciding slots: stage 2's (the record's pick first) when the
            // trial has two stages, else the single stage's.
            let deciding: [Int] = twoStage
                ? (stage2First >= 0 ? Array(stage2First ..< slotSet.count) : [])
                : Array(0 ..< min(stage1Slots, slotSet.count))
            var chosen = 0
            if let first = deciding.first, let record = summary(first).0 {
                var best = record
                for slot in deciding.dropFirst() {
                    let (m, kept, _) = summary(slot)
                    if let m, kept >= minimumRounds, m < best { best = m; chosen = slotSet[slot] }
                }
                if chosen != 0, !(best < record * (1 - adoptMargin)) { chosen = 0 }
            }
            let set = pool[chosen]
            install(set)
            setNarrowOperandNeeds([set])
            func line(_ slots: Range<Int>) -> String {
                slots.map { slot in
                    let (m, kept, total) = summary(slot)
                    return poolLabels[slotSet[slot]] + " "
                        + (m.map { String(format: "%.2f", $0 / 1e6) } ?? "-") + " (\(kept)/\(total))"
                }.joined(separator: " | ")
            }
            var log = "bonsai verify int8 in-situ: \(roundIndex) proposals; "
            if twoStage {
                log += "stage 1 (\(stage1Slots) sets x \(stage1Rounds)) median ms/round ["
                    + line(0 ..< min(stage1Slots, slotSet.count)) + "]" + perShapeLog + "; stage 2 ("
                    + (deciding.isEmpty
                        ? "none within \(Int(stage2Slack * 100)) % of the record's pick)"
                        : "\(deciding.count) sets x \(roundsPerSet)) [" + line(stage2First ..< slotSet.count) + "]")
            } else {
                log += "median ms/round [" + line(0 ..< min(stage1Slots, slotSet.count)) + "]"
            }
            log += "; "
            if chosen == 0 {
                log += "keeping the record's pick " + describe(set)
            } else if let first = deciding.first, let record = summary(first).0,
                let slot = deciding.first(where: { slotSet[$0] == chosen }), let m = summary(slot).0
            {
                log += "adopted \(poolLabels[chosen]) = " + describe(set)
                    + String(format: " (%+.2f%%)", (m / record - 1) * 100)
            }
            log += "; per-shape mapping [" + mapping(set) + "]"
            log += String(format: "; %.0f ms\n", Double(elapsedNanoseconds) / 1e6)
            FileHandle.standardError.write(log.data(using: .utf8)!)
            sets = []
            labels = []
            scopes = []
            pool = []
            poolLabels = []
            slotSet = []
            slotTimes = []
            schedule = []
            levels = []
            headLevels = []
            perShapeFirst = -1
            perShapeLog = ""
        }
    }

    /// Zoo bodies that passed the bitwise self-test with FP16 / FP32 output.
    /// A zoo body runs only at an output type it passed at (`narrowKernel`).
    nonisolated(unsafe) static var zooExact16 = Set<NarrowKernel>()
    nonisolated(unsafe) static var zooExact32 = Set<NarrowKernel>()
    nonisolated(unsafe) private static var zooDeclineAnnounced = false

    /// The kernel for this projection: the chosen one for its shape when its
    /// constants pass the proof (or it needs none) and, for a zoo body, when
    /// it passed the self-test at this output type, else `original`.
    /// `materialize` evaluates the FP32 scales now (the prompt route's
    /// load-time call); the verify route leaves them lazy.
    static func narrowKernel(
        _ cache: HadamardConstantLayoutCache, _ scales: MLXArray, _ biases: MLXArray,
        k: Int, n: Int, outputDType: DType, materialize: Bool
    ) -> NarrowKernel {
        let choice = narrowByShape[[k, n]] ?? narrowDefault
        if n % choice.variant.tn != 0 { return .original }
        // zoo 3a: only on the production shapes its self-test ran (not the head)
        if choice.variant.xtg != nil, !narrowXTGShapes.contains([k, n]) { return .original }
        if choice.variant.family != nil,
            !(outputDType == .float32 ? zooExact32 : zooExact16).contains(choice)
        {
            if !zooDeclineAnnounced {
                zooDeclineAnnounced = true
                FileHandle.standardError.write(
                    Data(
                        ("bonsai verify int8: \(choice.variant) is not self-tested at \(outputDType) "
                            + "(k \(k), n \(n)); original used there\n").utf8))
            }
            return .original
        }
        guard choice.form != .base else { return choice }
        guard cache.biasesAreNegativeScales(scales, biases) else { return .original }
        if choice.form == .negativeBiasF32Scales {
            _ = narrowScalesF32(cache, scales, materialize: materialize)
        }
        return choice
    }

    /// The prompt route's load-time preparation of the verify operands: the
    /// tiled weight copy, the proof and, when any chosen kernel reads them, the
    /// FP32 scales.
    static func prepareNarrowOperands(
        _ cache: HadamardConstantLayoutCache, _ weight: MLXArray, _ scales: MLXArray,
        _ biases: MLXArray
    ) {
        if narrowTiled { _ = narrowTiledWeight(cache, weight, materialize: true) }
        guard narrowNeedsProof, cache.biasesAreNegativeScales(scales, biases) else { return }
        if narrowNeedsF32 { _ = narrowScalesF32(cache, scales, materialize: true) }
    }

    /// The verify int8 kernels read a tiled copy of each projection's packed
    /// words (`tileNarrowWeight`). The copy holds the stored 32-bit words
    /// unchanged, only reordered, so every code, scale and offset the kernel
    /// applies is the stored one and the output is bitwise that of the stored
    /// layout (the self-test runs every candidate on the copy against
    /// `original` on the stored words). What changes is the load: 32 lanes
    /// read a group's 1 KB tile in one contiguous run. The int8-staged prompt
    /// kernel reads the same copy, after its own bitwise self-test, so the
    /// copy is in use in every phase and the stored words are read only at
    /// load. `DARKBLOOM_BONSAI_TENSOR_ROUTE_TILED=0` reads the stored layout on
    /// both routes.
    static let narrowTiled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_TILED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !["0", "false", "no", "off"].contains(value ?? ""), support == .staged8 else {
            return false
        }
        return promptTiledSelfTest()
    }()

    /// The int8-staged prompt kernel on synthetic operands, once on the stored
    /// words and once on the tiled copy: every output bit must match (FP32
    /// output, signed zeros included), or neither route reads the copy.
    private static func promptTiledSelfTest() -> Bool {
        let m = 128, k = 1024, n = 192, kg = k / 128
        let codeType: DType = signedCodes ? .int8 : .uint8
        let codes = MLXRandom.randInt(
            signedCodes ? Int32(-127) ..< Int32(128) : Int32(0) ..< Int32(256), [m, k],
            key: MLXRandom.key(71)
        ).asType(codeType)
        let weight = MLXRandom.randInt(
            Int32(0) ..< Int32(65536), [n, k / 8], key: MLXRandom.key(72)
        ).asType(.uint16).view(dtype: .uint32)
        let scalesT = MLXRandom.uniform(
            Float(-0.05) ..< Float(0.05), [kg, n], key: MLXRandom.key(73)
        ).asType(.float16)
        let biasesT = (scalesT.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
        let folded = MLXRandom.normal([kg, n], key: MLXRandom.key(74))
        let ascale = MLXRandom.uniform(Float(0.0001) ..< Float(0.05), [m, kg], key: MLXRandom.key(75))
        let asums = MLXRandom.normal([m, kg], key: MLXRandom.key(76)) * Float(50)
        let tiled = tileNarrowWeight(weight, n: n, k: k)
        func run(_ words: MLXArray, _ tiledFlag: Int) -> MLXArray {
            kernelStaged8(
                [codes, words, scalesT, biasesT, folded, ascale, asums, dimsArray(k: k, m: m, n: n)],
                template: [
                    ("OutT", DType.float32), ("MPERM", rowTiledConstants ? 1 : 0),
                    ("SIGNED", signedCodes ? 1 : 0), ("NEGATIVE_SCALE_BIAS", 1),
                    ("FACTORED", factoredPromptEpilogue ? 1 : 0), ("TILED", tiledFlag),
                ],
                grid: (n / 64 * 128, m / 64, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [.float32])[0]
        }
        let stored = run(weight, 0)
        let copy = run(tiled, 1)
        let same = (stored.view(dtype: .uint32) .== copy.view(dtype: .uint32)).all().item(Bool.self)
        FileHandle.standardError.write(
            Data(
                ("bonsai tiled weights: prompt kernel self-test "
                    + (same ? "passed; both routes read the tiled copy\n"
                        : "FAILED; the stored layout is kept\n")).utf8))
        return same
    }

    /// `[N, K/16]` packed words reordered to `[N/32, K/128, 32, 8]`: for each
    /// 32-column block and 128-group, the 32 columns' 8 words in column order.
    static func tileNarrowWeight(_ weight: MLXArray, n: Int, k: Int) -> MLXArray {
        weight.reshaped([n / 32, 32, k / 128, 8]).transposed(0, 2, 1, 3).contiguous()
            .reshaped([n, k / 16])
    }

    /// The tiled copy of a projection's words, built once per weight array.
    static func narrowTiledWeight(
        _ cache: HadamardConstantLayoutCache, _ weight: MLXArray, materialize: Bool
    ) -> MLXArray {
        cache.derived(weight, tag: 5) { w in
            let tiled = tileNarrowWeight(w, n: w.dim(0), k: w.dim(1) * 16)
            if materialize { eval(tiled) }
            return tiled
        }
    }

    /// The FP16 scales widened to FP32 and transposed to `[groups, rows]`.
    static func narrowScalesF32(
        _ cache: HadamardConstantLayoutCache, _ scales: MLXArray, materialize: Bool
    ) -> MLXArray {
        cache.derived(scales, tag: 4) { s in
            let widened = s.asType(.float32).transposed(1, 0).contiguous()
            if materialize { eval(widened) }
            return widened
        }
    }

    /// One launch of the verify int8 kernel. For the negated-offset forms
    /// `biasesT` is not read (callers pass `scalesT`).
    static func launchNarrowInt8(
        _ codes: MLXArray, _ weight: MLXArray, _ scalesT: MLXArray, _ biasesT: MLXArray,
        _ ascale: MLXArray, _ rowsum: MLXArray, k: Int, n: Int, outputDType: DType,
        kernel: NarrowKernel, tiled: Bool = false
    ) -> MLXArray {
        let m = 16
        let inputs = [codes, weight, scalesT, biasesT, ascale, rowsum, dimsArray(k: k, m: m, n: n)]
        let template: [(String, any KernelTemplateArg)] = [
            ("OutT", outputDType), ("NEG", kernel.form == .base ? 0 : 1),
            ("F32S", kernel.form == .negativeBiasF32Scales ? 1 : 0), ("TILED", tiled ? 1 : 0),
        ]
        let v = kernel.variant
        // Zoo 3a: each quarter's fold to its plane, then the planes' sum.
        if let x = v.xtg, tiled, let body = kernelNarrowXTG[x.body] {
            let threads = 32 * x.qt * x.r
            let t = (x.t + [("QT", x.qt)]).map { ($0.0, $0.1 as any KernelTemplateArg) }
            let part = body(
                inputs, template: Array(template[1 ..< 3]) + t,
                grid: (n / 32 * threads, 4 / x.qt, 1), threadGroup: (threads, 1, 1),
                outputShapes: [[4, m, n]], outputDTypes: [.float32])[0]
            return kernelNarrowXTGSum(
                [part, inputs[6]], template: [("OutT", outputDType)], grid: (m * n / 4, 1, 1),
                threadGroup: (256, 1, 1), outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
        }
        // The zoo bodies read the tiled copy only (`original` otherwise).
        if let family = v.family, v.xtg == nil, tiled {
            let zooTemplate = Array(template.prefix(3))
            if let pairR = v.zoo2Pair {
                let t = zooTemplate + v.zoo2Template.map { ($0.0, $0.1 as any KernelTemplateArg) }
                if pairR {
                    return kernelNarrowInt8PairR(
                        inputs, template: t, grid: (n / 32 * v.threads, 1, 1),
                        threadGroup: (v.threads, 1, 1),
                        outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
                }
                return kernelNarrowInt8Zoo2(
                    inputs, template: t, grid: (n / v.tn * 128, 1, 1), threadGroup: (128, 1, 1),
                    outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
            }
            if family == "pair" {
                return kernelNarrowInt8Pair(
                    inputs, template: zooTemplate + [("PD", v.pd), ("KH", v.kh)],
                    grid: (n / 32 * 256, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
            }
            return kernelNarrowInt8Zoo(
                inputs, template: zooTemplate + [("PD", v.pd), ("TN", v.tn), ("KH", v.kh), ("AM", v.am)],
                grid: (n / v.tn * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
        }
        switch v.family == nil ? v : .v0 {
        case .v0:
            return kernelNarrowInt8(
                inputs, template: template,
                grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
        default:
            return kernelNarrowInt8Pipelined(
                inputs, template: template + [("PD", v.pd), ("TN", v.tn), ("KH", v.kh)],
                grid: (n / v.tn * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
        }
    }

    /// Synthetic operands for the verify int8 kernel: signed codes, random
    /// 2-bit words, FP16 scales of both signs (zeros and signed zeros
    /// included) with offsets that are their FP16 negations bit for bit,
    /// FP32 activation scales and scaled sums. Nothing depends on a request.
    fileprivate struct NarrowOperands {
        let k: Int, n: Int
        let codes: MLXArray, weight: MLXArray, tiledWeight: MLXArray
        let scalesT: MLXArray, biasesT: MLXArray, scalesT32: MLXArray
        let ascale: MLXArray, rowsum: MLXArray

        init(k: Int, n: Int, seed: UInt64) {
            self.k = k
            self.n = n
            let kg = k / 128
            codes = MLXRandom.randInt(
                Int32(-127) ..< Int32(128), [16, k], key: MLXRandom.key(seed)
            ).asType(.int8)
            weight = MLXRandom.randInt(
                Int32(0) ..< Int32(65536), [n, k / 8], key: MLXRandom.key(seed + 1)
            ).asType(.uint16).view(dtype: .uint32)
            var s = MLXRandom.uniform(
                Float(-0.05) ..< Float(0.05), [n, kg], key: MLXRandom.key(seed + 2))
            let pick = MLXRandom.randInt(Int32(0) ..< Int32(64), [n, kg], key: MLXRandom.key(seed + 3))
            s = which(pick .== MLXArray(Int32(0)), MLXArray(Float(0)), s)
            s = which(pick .== MLXArray(Int32(1)), MLXArray(Float(-0.0)), s)
            let scales = s.asType(.float16)
            let biases = (scales.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000))).view(dtype: .float16)
            scalesT = scales.transposed(1, 0).contiguous()
            biasesT = biases.transposed(1, 0).contiguous()
            scalesT32 = scales.asType(.float32).transposed(1, 0).contiguous()
            ascale = MLXRandom.uniform(
                Float(0.0001) ..< Float(0.05), [16, kg], key: MLXRandom.key(seed + 4))
            rowsum = MLXRandom.normal([16, kg], key: MLXRandom.key(seed + 5)) * Float(50)
            tiledWeight = narrowTiled ? tileNarrowWeight(weight, n: n, k: k) : weight
            eval(codes, weight, tiledWeight, scalesT, biasesT, scalesT32, ascale, rowsum)
        }

        /// `tiled` (default: the route's setting) reads the tiled copy; the
        /// self-test's reference passes `false`, so every tiled candidate is
        /// checked bit for bit against `original` on the stored layout.
        func run(_ kernel: NarrowKernel, _ outputDType: DType, tiled: Bool = narrowTiled) -> MLXArray {
            let (s, b): (MLXArray, MLXArray)
            switch kernel.form {
            case .base: (s, b) = (scalesT, biasesT)
            case .negativeBias: (s, b) = (scalesT, scalesT)
            case .negativeBiasF32Scales: (s, b) = (scalesT32, scalesT32)
            }
            return launchNarrowInt8(
                codes, tiled ? tiledWeight : weight, s, b, ascale, rowsum, k: k, n: n,
                outputDType: outputDType, kernel: kernel, tiled: tiled)
        }
    }

    /// The verify window's production shapes `(k, n)`: qkv|z, gate|up, down
    /// and attention qkv, 16 rows each.
    static let narrowTunedShapes = [(5120, 16384), (5120, 34816), (17408, 5120), (5120, 14336)]
    /// The shapes the zoo sets also key: o/out (timed with the tuned four) and
    /// the head (untimed at load: a wide-class zoo set puts its gate|up body on
    /// it; the fused head top two (`Qwen35HeadTopTwo`) has a form for every
    /// zoo body, self-tested on the head's final kernel after the trial).
    static let narrowZooShapes = [(6144, 5120), (5120, 248320)]

    /// A production shape's class: 1 for the N = 5120 shapes (down, o/out: 160
    /// threadgroups at TN = 32, FP16 outputs on the verify route), 2 for the
    /// wide ones (qkv|z and attention qkv with FP32 outputs, gate|up with FP16,
    /// the head with FP32).
    static func narrowShapeClass(_ key: [Int]) -> Int { key.count == 2 && key[1] == 5120 ? 1 : 2 }

    /// The tiled zoo (`NarrowVariant.family != nil`): its own self-test (one
    /// error scope per body, so a body the toolchain cannot compile is dropped
    /// alone), timing on the tuned shapes and o/out against the record's picks
    /// (each body on the shapes of its classes), and trial sets: per family,
    /// the record's pick with the family's fastest body on each N = 5120 shape
    /// (k32, wide, acoop, pair, dual, xtg, xtgp), and the record's pick with
    /// the family's fastest body over the wide shapes on all of them and the
    /// head (k32, wide, acoop; xtg without the head). Every body on a wide
    /// shape passes the FP32 self-test too.
    /// With the record's own candidates that is more than four sets, so the
    /// in-situ trial runs in two stages (`NarrowInSituTrial`).
    /// Only with the tiled copy. `DARKBLOOM_BONSAI_TENSOR_ROUTE_TZOO=0` leaves
    /// the record's candidates alone (and the trial one stage);
    /// `..._TZOO_FORCE=<body>` (e.g. `k32pd2`, `p4k16pd1`) installs that body
    /// on every shape (a zoo 3a body: but the head), without a trial, once it
    /// passes; `..._TZOO_FORCE=down=x4k32pd2,o=pk32pd2,...` installs one body
    /// per named shape over the record's pick (validation).
    static let narrowZoo: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_TZOO"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "") && narrowTiled
    }()

    /// Zoo bodies in self-test order (the deadline cuts the last): the zoo's
    /// eleven, then zoo 2's eight.
    static let narrowZooVariants: [NarrowVariant] = [
        .k32pd2, .pk32pd2, .w64k64pd1, .a128pd2, .k32pd4, .pk64pd2, .w64k32pd1, .a64pd2,
        .aw64pd1, .k32pd1, .pk32pd1,
        .p4k16pd1, .g2k32pd1, .k16pd2, .w128k32pd1, .p5k16pd1, .p4k16x1, .pk16pd2, .k16pd4,
    ]

    /// Zoo 3a's bodies (`NarrowVariant.xtg`), self-tested after the zoo's
    /// with a budget of their own, and also on every production shape but the
    /// head (`narrowXTGShapes`, the only shapes they run on). Families `xtg`
    /// (zoo / zoo 2 bases: N = 5120 and wide sets) and `xtgp` (R-pair bases:
    /// N = 5120 sets). `DARKBLOOM_BONSAI_TENSOR_ROUTE_TZOO_XTG=0` drops them.
    static let narrowXTGVariants: [NarrowVariant] = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_TZOO_XTG"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !["0", "false", "no", "off"].contains(value ?? "") else { return [] }
        return [.x4k32pd2, .x4p2k32pd2, .x4k128pd2, .x2k32pd2, .x4p4k16pd1, .x4k16pd4, .x2p2k32pd2]
            .filter { kernelNarrowXTG[$0.xtg!.body] != nil }
    }()
    static let narrowXTGShapes = Set((narrowTunedShapes + [narrowZooShapes[0]]).map { [$0.0, $0.1] })

    /// Chooses the verify int8 kernels once, at load, on the running GPU.
    ///
    /// Self-test: every candidate runs against `original` on synthetic
    /// operands (gate-, down- and two odd-quarter widths, so every ring's
    /// remainder guards run) and must match every output bit (FP16 for all;
    /// FP32 as well for each kernel that is then chosen); a mismatch or any
    /// MLX error (a body the toolchain cannot compile) drops it. Candidates:
    /// every body (`v0` and each pipelined variant) in each epilogue form.
    /// Timing: the survivors and `original` run alternately on the four
    /// production shapes, each over distinct weight sets of >= 96 MB (so
    /// every launch streams its weights), best of five trials. The record's
    /// pick: over the record's bodies (`narrowRecordVariants`), each shape
    /// keeps its fastest kernel and every other shape takes the fastest in
    /// total. That is what is installed; the timing only shortlists the
    /// in-situ candidates (`NarrowInSituTrial`): the record's pick, `v0` with
    /// the negated offset, the tiled zoo's family sets (`zooSets`), and the
    /// two fastest K3 bodies in total, each after its FP32 self-test (a zoo
    /// body where its shapes take FP32), and every candidate kernel is then
    /// launched once per production shape at each output type it may run at. A candidate not
    /// started within 4 s of the self-test is skipped (load time is not
    /// timed). Runs at model init, before any timed phase, and builds every
    /// candidate's pipelines, so no verify round (nor any trial round)
    /// compiles anything.
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_EPILOGUE=off` keeps `original`
    /// (master kill switch); `neg` / `f32` force that epilogue.
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_PIPELINE=off` keeps the recorded
    /// body (`v0`); a comma-separated list of variant names (`pd1` .. `pd4`,
    /// `tn64`, `k64pd1` .. `k64pd4`) limits the pipelined candidates to those.
    private static func chooseNarrowKernels() -> (NarrowChoice, [(NarrowChoice, String, Int)]) {
        let environment = ProcessInfo.processInfo.environment
        func knob(_ name: String) -> String? {
            environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }
        let forms: [NarrowEpilogue]
        switch knob("DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_EPILOGUE") {
        case "off", "0", "false", "no", "base": return ((.original, [:]), [])
        case "neg": forms = [.negativeBias]
        case "f32": forms = [.negativeBiasF32Scales]
        default: forms = [.negativeBias, .negativeBiasF32Scales]
        }
        // Default order = self-test order (what the deadline would cut last).
        let defaultVariants: [NarrowVariant] = [
            .pd1, .pd2, .k64pd1, .k64pd2, .pd3, .k64pd3, .pd4, .k64pd4, .tn64,
        ]
        var variants = defaultVariants
        switch knob("DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_PIPELINE") {
        case "off", "0", "false", "no", "v0": variants = []
        case .some(let value):
            let named = value.split(separator: ",").map {
                NarrowVariant(name: $0.trimmingCharacters(in: .whitespaces))
            }
            if !named.isEmpty, named.allSatisfy({ $0 != nil }) {
                variants = named.compactMap { $0 }.filter { $0 != .v0 && $0.family == nil }
            }
        case nil: break
        }
        let candidates = forms.map { NarrowKernel(variant: .v0, form: $0) }
            + variants.flatMap { v in forms.map { NarrowKernel(variant: v, form: $0) } }

        let start = DispatchTime.now().uptimeNanoseconds
        func elapsedMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 }
        var log = "bonsai verify int8 kernels:"
        var passed: [NarrowKernel] = []
        var failedF32 = Set<NarrowKernel>()
        var timings: [NarrowKernel: [Double]] = [:]
        var byShape: [[Int]: NarrowKernel] = [:]
        var fallback = NarrowKernel.original
        var trial: [(NarrowChoice, String, Int)] = []
        do {
            try withError { error in
                let testShapes = [
                    (5120, 4096, UInt64(71)), (17408, 1024, UInt64(72)), (2560, 512, UInt64(73)),
                    (3584, 512, UInt64(74)),
                ]
                let testOps = testShapes.map { NarrowOperands(k: $0.0, n: $0.1, seed: $0.2) }
                func matches(_ kernel: NarrowKernel, _ outputDType: DType) throws -> Bool {
                    let bits: DType = outputDType == .float16 ? .uint16 : .uint32
                    var same = true
                    for ops in testOps {
                        let reference = ops.run(.original, outputDType, tiled: false)
                        let y = ops.run(kernel, outputDType)
                        let differ = (y.view(dtype: bits) .!= reference.view(dtype: bits))
                            .asType(.int32).sum()
                        eval(differ)
                        try error.check()
                        if differ.item(Int32.self) != 0 { same = false }
                    }
                    return same
                }
                var skipped: [NarrowKernel] = []
                for kernel in candidates {
                    if elapsedMs() > 4000 { skipped.append(kernel); continue }
                    if try matches(kernel, .float16) { passed.append(kernel) }
                }
                log += " self-test passed [" + passed.map(\.description).joined(separator: " ") + "]"
                if !skipped.isEmpty {
                    log += " skipped [" + skipped.map(\.description).joined(separator: " ") + "]"
                }
                guard !passed.isEmpty else { return }

                let kernels = [NarrowKernel.original] + passed
                let sets = narrowTunedShapes.enumerated().map { (index, shape) -> [NarrowOperands] in
                    let bytes = shape.0 * shape.1 / 4
                    let copies = min(6, max(2, (96 << 20) / bytes + 1))
                    return (0 ..< copies).map {
                        NarrowOperands(k: shape.0, n: shape.1, seed: 100 + UInt64(index * 8 + $0))
                    }
                }
                for kernel in kernels { eval(sets.flatMap { $0.map { $0.run(kernel, .float16) } }) }
                try error.check()
                for kernel in kernels { timings[kernel] = Array(repeating: .infinity, count: sets.count) }
                for _ in 0 ..< 5 {
                    for (index, shapeSets) in sets.enumerated() {
                        for kernel in kernels {
                            let outs = shapeSets.map { $0.run(kernel, .float16) }
                            let t0 = DispatchTime.now().uptimeNanoseconds
                            eval(outs)
                            let us = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000
                                / Double(outs.count)
                            timings[kernel]![index] = min(timings[kernel]![index], us)
                        }
                    }
                }
                try error.check()

                // The record's picks, then the FP32 self-test of each picked
                // kernel; a kernel failing it is dropped and the picks redone.
                var checkedF32 = Set<NarrowKernel>()
                func exactF32(_ kernel: NarrowKernel) throws -> Bool {
                    if kernel == .original || checkedF32.contains(kernel) {
                        return !failedF32.contains(kernel)
                    }
                    checkedF32.insert(kernel)
                    if try matches(kernel, .float32) { return true }
                    failedF32.insert(kernel)
                    return false
                }
                while true {
                    let usable = kernels.filter {
                        !failedF32.contains($0) && narrowRecordVariants.contains($0.variant)
                    }
                    func fastest(_ cost: (NarrowKernel) -> Double) -> NarrowKernel {
                        usable.min { cost($0) < cost($1) } ?? .original
                    }
                    fallback = fastest { timings[$0]!.reduce(0, +) }
                    byShape = [:]
                    for (index, shape) in narrowTunedShapes.enumerated() {
                        byShape[[shape.0, shape.1]] = fastest { timings[$0]![index] }
                    }
                    let picked = Set([fallback] + Array(byShape.values)).subtracting([.original])
                    var clean = true
                    for kernel in picked {
                        if try !exactF32(kernel) { clean = false }
                    }
                    if clean { break }
                }

                // Tiled zoo: each body in its own error scope (a body that
                // fails to compile is dropped alone), bit for bit against
                // `original` on the stored words, FP16 and FP32 outputs.
                let zooForm: NarrowEpilogue = forms.contains(.negativeBias) ? .negativeBias : forms[0]
                var xtgOps: [NarrowOperands] = []
                func zooExact(_ kernel: NarrowKernel, _ outputDType: DType) -> Bool {
                    let bits: DType = outputDType == .float16 ? .uint16 : .uint32
                    // zoo 3a: every production shape but the head as well
                    if kernel.variant.xtg != nil, xtgOps.isEmpty {
                        xtgOps = sets.map { $0[0] } + [NarrowOperands(k: 6144, n: 5120, seed: 200)]
                    }
                    let same = try? withError { scoped -> Bool in
                        var same = true
                        for ops in testOps + (kernel.variant.xtg != nil ? xtgOps : []) {
                            let reference = ops.run(.original, outputDType, tiled: false)
                            let y = ops.run(kernel, outputDType)
                            let differ = (y.view(dtype: bits) .!= reference.view(dtype: bits))
                                .asType(.int32).sum()
                            eval(differ)
                            try scoped.check()
                            if differ.item(Int32.self) != 0 { same = false }
                        }
                        return same
                    }
                    return same ?? false
                }
                if narrowZoo, let name = knob("DARKBLOOM_BONSAI_TENSOR_ROUTE_TZOO_FORCE"),
                    let variant = NarrowVariant(name: name), variant.family != nil,
                    variant.xtg == nil || narrowXTGVariants.contains(variant)
                {
                    let forced = NarrowKernel(variant: variant, form: zooForm)
                    if zooExact(forced, .float16), zooExact(forced, .float32) {
                        zooExact16 = [forced]
                        zooExact32 = [forced]
                        fallback = forced
                        byShape = [:]
                        log += "; tzoo forced \(forced) on every shape"
                        return
                    }
                    log += "; tzoo forced \(forced) FAILED its self-test, record's pick kept"
                }
                // A per-shape map (`attn=`, `qkvz=`, `gateup=`, `o=`, `down=`,
                // `head=` a zoo body each, comma-separated) over the record's
                // pick; each body passes FP16, and FP32 on a wide shape.
                if narrowZoo, let value = knob("DARKBLOOM_BONSAI_TENSOR_ROUTE_TZOO_FORCE"), value.contains("=") {
                    let names = NarrowInSituTrial.perShapeNames.map { $0.replacingOccurrences(of: "|", with: "") } + ["head"]
                    let keys = NarrowInSituTrial.perShapeKeys + [NarrowInSituTrial.headKey]
                    var map = byShape, exact16 = Set<NarrowKernel>(), exact32 = Set<NarrowKernel>()
                    var ok = true
                    for entry in value.split(separator: ",") {
                        let part = entry.split(separator: "=").map { $0.trimmingCharacters(in: .whitespaces) }
                        guard part.count == 2, let index = names.firstIndex(of: part[0]),
                            let variant = NarrowVariant(name: part[1]), variant.family != nil,
                            variant.xtg == nil || (narrowXTGVariants.contains(variant) && index < 5)
                        else { ok = false; break }
                        let kernel = NarrowKernel(variant: variant, form: zooForm)
                        if !exact16.contains(kernel) {
                            guard zooExact(kernel, .float16) else { ok = false; break }
                            exact16.insert(kernel)
                        }
                        if narrowShapeClass(keys[index]) == 2, !exact32.contains(kernel) {
                            guard zooExact(kernel, .float32) else { ok = false; break }
                            exact32.insert(kernel)
                        }
                        map[keys[index]] = kernel
                    }
                    if ok {
                        zooExact16 = exact16
                        zooExact32 = exact32
                        byShape = map
                        log += "; tzoo forced per-shape mapping [" + NarrowInSituTrial.mapping((fallback, map)) + "]"
                        return
                    }
                    log += "; tzoo forced map \(value) FAILED (a name or a self-test), record's pick kept"
                }

                // The in-situ candidates: the record's pick, v0 with the
                // negated offset, the zoo's family sets, and the two fastest
                // K3 bodies in total, each exact at FP32 as well (a zoo body at
                // the output types of its shapes). Choices equal on every
                // shape collapse.
                guard NarrowInSituTrial.enabled else { return }
                var shortlist: [(NarrowChoice, String, Int)] = [((fallback, byShape), "record", 0)]
                let v0Negative = NarrowKernel(variant: .v0, form: .negativeBias)
                if passed.contains(v0Negative), try exactF32(v0Negative) {
                    shortlist.append(((v0Negative, [:]), "v0neg", 0))
                }
                let k3 = passed.filter { !narrowRecordVariants.contains($0.variant) }
                    .sorted { timings[$0]!.reduce(0, +) < timings[$1]!.reduce(0, +) }
                var k3Sets: [(NarrowChoice, String, Int)] = []
                for kernel in k3 where k3Sets.count < 2 {
                    if try exactF32(kernel) { k3Sets.append(((kernel, [:]), "k3 \(kernel)", 0)) }
                }
                if narrowZoo { shortlist += zooSets(fallback, byShape, sets, zooForm, zooExact, &log) }
                shortlist += k3Sets
                for candidate in shortlist
                where !trial.contains(where: {
                    NarrowInSituTrial.effective($0.0) == NarrowInSituTrial.effective(candidate.0)
                }) {
                    trial.append(candidate)
                }
                guard trial.count >= 2 else { trial = []; return }
                // Every candidate kernel once per production shape at each
                // output type it may run at (a zoo body at FP32 only where it
                // passed it), so no pipeline compiles inside a trial round.
                let trialKernels = Set(trial.flatMap { [$0.0.0] + Array($0.0.1.values) })
                for kernel in trialKernels {
                    for outputDType in [DType.float16, .float32]
                    where kernel.variant.family == nil
                        || (outputDType == .float16 ? zooExact16 : zooExact32).contains(kernel)
                    {
                        eval(sets.map { $0[0].run(kernel, outputDType) })
                    }
                }
                try error.check()
            }
        } catch {
            passed = []
            byShape = [:]
            fallback = .original
            trial = []
            log += " error \(error)"
        }
        if !timings.isEmpty {
            log += "; us/launch per shape (qkv|z gate|up down attn):"
            for kernel in [NarrowKernel.original] + passed {
                guard let row = timings[kernel] else { continue }
                log += " \(kernel)=" + row.map { String(format: "%.1f", $0) }.joined(separator: ",")
            }
        }
        if !failedF32.isEmpty {
            log += "; FP32 self-test failed [" + failedF32.map(\.description).joined(separator: " ") + "]"
        }
        Memory.clearCache()
        log += "; using default \(fallback), per shape ["
            + narrowTunedShapes.map { "\($0.0)x\($0.1)=\(byShape[[$0.0, $0.1]] ?? fallback)" }
            .joined(separator: " ") + "]"
        if !trial.isEmpty {
            log += "; in-situ candidates [record" + trial.dropFirst().map {
                " \($0.1)=" + NarrowInSituTrial.describe($0.0)
            }.joined() + "]"
        }
        log += "; \(String(format: "%.0f", elapsedMs())) ms\n"
        FileHandle.standardError.write(log.data(using: .utf8)!)
        return ((fallback, byShape), trial)
    }

    /// The tiled zoo's trial sets (see `narrowZoo`): the zoo bodies' FP16
    /// self-test (9 s budget: a cold toolchain compiles each body at
    /// 0.2-0.9 s), their timing against the record's picks on the
    /// tuned shapes and o/out (the record's method, best of three, each body
    /// on the shapes of its classes), then per family the record's pick with
    /// the family's fastest body on each N = 5120 shape (scope 1), and with
    /// the family's fastest body over the wide shapes (launches per round
    /// times time) on all of them and the head (scope 2; not for pair, dual
    /// and xtgp, which target the N = 5120 shapes; xtg leaves the head's pick).
    /// Zoo 3a (`narrowXTGVariants`) has 4 s more. Every zoo body in a scope-2
    /// set also passes the FP32 self-test (qkv|z, attention qkv and the head
    /// take FP32 outputs); a body failing it leaves the sets.
    private static func zooSets(
        _ fallback: NarrowKernel, _ byShape: [[Int]: NarrowKernel], _ sets: [[NarrowOperands]],
        _ form: NarrowEpilogue, _ exact: (NarrowKernel, DType) -> Bool, _ log: inout String
    ) -> [(NarrowChoice, String, Int)] {
        let start = DispatchTime.now().uptimeNanoseconds
        var passed: [NarrowKernel] = [], failed: [NarrowKernel] = [], skipped: [NarrowKernel] = []
        for variant in narrowZooVariants + narrowXTGVariants {
            let kernel = NarrowKernel(variant: variant, form: form)
            let budget: Double = variant.xtg == nil ? 9000 : 13000
            if Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 > budget {
                skipped.append(kernel)
            } else if exact(kernel, .float16) {
                passed.append(kernel)
            } else {
                failed.append(kernel)
            }
        }
        zooExact16 = Set(passed)
        func names(_ kernels: [NarrowKernel]) -> String { kernels.map { "\($0.variant)" }.joined(separator: " ") }
        log += "; tzoo self-test passed [\(names(passed))]"
        if !failed.isEmpty { log += " failed [\(names(failed))]" }
        if !skipped.isEmpty { log += " skipped [\(names(skipped))]" }
        guard !passed.isEmpty, sets.count == narrowTunedShapes.count else { return [] }

        let shapes = narrowTunedShapes + [narrowZooShapes[0]]
        let keys = shapes.map { [$0.0, $0.1] }
        let classes = keys.map { narrowShapeClass($0) }
        let head = [narrowZooShapes[1].0, narrowZooShapes[1].1]
        let o = narrowZooShapes[0]
        let oSets = (0 ..< min(6, max(2, (96 << 20) / (o.0 * o.1 / 4) + 1))).map {
            NarrowOperands(k: o.0, n: o.1, seed: 200 + UInt64($0))
        }
        let allSets = sets + [oSets]
        var record: [NarrowKernel] = []
        for kernel in [fallback] + narrowTunedShapes.compactMap({ byShape[[$0.0, $0.1]] })
        where !record.contains(kernel) {
            record.append(kernel)
        }
        let kernels = record + passed
        func timed(_ kernel: NarrowKernel, _ index: Int) -> Bool {
            kernel.variant.family == nil || kernel.variant.zooClasses.contains(classes[index])
        }
        for kernel in kernels {
            eval(allSets.indices.filter { timed(kernel, $0) }.flatMap { allSets[$0].map { $0.run(kernel, .float16) } })
        }
        var t: [NarrowKernel: [Double]] = [:]
        for kernel in kernels { t[kernel] = Array(repeating: .infinity, count: shapes.count) }
        for _ in 0 ..< 3 {
            for (index, shapeSets) in allSets.enumerated() {
                for kernel in kernels where timed(kernel, index) {
                    let outs = shapeSets.map { $0.run(kernel, .float16) }
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    eval(outs)
                    let us = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000 / Double(outs.count)
                    t[kernel]![index] = min(t[kernel]![index], us)
                }
            }
        }
        log += "; tzoo us/launch (qkv|z gate|up down attn o):"
        for kernel in kernels {
            let name = kernel.variant.family == nil ? "\(kernel)" : "\(kernel.variant)"
            log += " \(name)=" + t[kernel]!.map { $0.isFinite ? String(format: "%.1f", $0) : "-" }
                .joined(separator: ",")
        }

        // launches per round: qkv|z 48, gate|up 64, down 64, attn 16, o/out 64
        let perRound: [Double] = [48, 64, 64, 16, 64]
        func build(_ usable: [NarrowKernel]) -> [(NarrowChoice, String, Int)] {
            var choices: [(NarrowChoice, String, Int)] = []
            for family in ["k32", "wide", "acoop", "pair", "dual", "xtg", "xtgp"] {
                let members = usable.filter { $0.variant.family == family }
                var map = byShape
                var changed = false
                for index in shapes.indices where classes[index] == 1 {
                    let fit = members.filter { t[$0]![index].isFinite }
                    if let best = fit.min(by: { t[$0]![index] < t[$1]![index] }) {
                        map[keys[index]] = best
                        changed = true
                    }
                }
                if changed { choices.append(((fallback, map), "\(family)/5120", 1)) }
            }
            for family in ["k32", "wide", "acoop", "xtg"] {
                func cost(_ kernel: NarrowKernel) -> Double {
                    shapes.indices.filter { classes[$0] == 2 }.reduce(0) { $0 + perRound[$1] * t[kernel]![$1] }
                }
                let fit = usable.filter { $0.variant.family == family && cost($0).isFinite }
                guard let best = fit.min(by: { cost($0) < cost($1) }) else { continue }
                var map = byShape
                for index in shapes.indices where classes[index] == 2 { map[keys[index]] = best }
                if best.variant.xtg == nil { map[head] = best }
                choices.append(((fallback, map), "\(family)/wide", 2))
            }
            return choices
        }
        var usable = passed
        var checked = Set<NarrowKernel>()
        var failedF32: [NarrowKernel] = []
        var choices: [(NarrowChoice, String, Int)] = []
        while true {
            choices = build(usable)
            var clean = true
            for choice in choices where choice.2 == 2 {
                for kernel in Set(choice.0.1.values)
                where kernel.variant.family != nil && !checked.contains(kernel) {
                    checked.insert(kernel)
                    if exact(kernel, .float32) {
                        zooExact32.insert(kernel)
                    } else {
                        usable.removeAll { $0 == kernel }
                        failedF32.append(kernel)
                        clean = false
                    }
                }
            }
            if clean { break }
        }
        if !failedF32.isEmpty { log += "; tzoo FP32 self-test failed [\(names(failedF32))]" }
        return choices
    }

    /// Installs the record's pick and sets up the in-situ trial. The operands
    /// every candidate reads (the per-projection proof, the FP32 scales) are
    /// prepared at the load-time prompt forward, before any trial round.
    private static func installNarrowChoice() {
        let (choice, trial) = chooseNarrowKernels()
        narrowDefault = choice.0
        narrowByShape = choice.1
        NarrowInSituTrial.sets = trial.map(\.0)
        NarrowInSituTrial.labels = trial.map(\.1)
        NarrowInSituTrial.scopes = trial.map(\.2)
        setNarrowOperandNeeds([choice] + trial.map(\.0))
    }

    nonisolated(unsafe) private static var installed = false

    static func installIfNeeded() {
        guard enabled, !installed else { return }
        installed = true
        guard support != .none else { return }
        HadamardQuantizedLinear.tensorPackedMatmulApplies = { rows, n, k in
            rows % 64 == 0 && n % 64 == 0 && k % 512 == 0
        }
        if headAvailable {
            HadamardQuantizedLinear.tensorPackedMatmulHead = {
                rotated, weight, scales, biases, groupSize, outputDType in
                guard groupSize == 128, rotated.dtype == .float16, rotated.ndim == 2,
                    weight.dtype == .uint32, scales.dtype == .float16, biases.dtype == .float16,
                    [DType.float16, .float32].contains(outputDType)
                else { return nil }
                let m = rotated.dim(0)
                let k = rotated.dim(1)
                let n = weight.dim(0)
                guard m == 16, n >= 65536, n % 32 == 0, k % 256 == 0, weight.dim(1) == k / 16,
                    scales.shape == [n, k / 128], biases.shape == [n, k / 128]
                else { return nil }
                if !headAnnounced {
                    headAnnounced = true
                    FileHandle.standardError.write(
                        "bonsai head kernel: in use (k \(k), n \(n), \(outputDType))\n"
                            .data(using: .utf8)!)
                }
                return runHead(rotated, weight, scales, biases, k: k, n: n, outputDType: outputDType)
            }
        }
        if verifyEnabled, verifyForm != .none {
            HadamardQuantizedLinear.tensorPackedMatmulNarrowApplies = { rows, n, k in
                rows <= 16 && n % 64 == 0 && k % 512 == 0 && n <= verifyMaximumColumns
            }
        }
        if verifyEnabled, verifyForm == .staged8, signedCodes {
            HadamardQuantizedLinear.tensorPackedMatmulNarrowInt8 = {
                activation, weight, scales, biases, groupSize, outputDType, cache in
                let codes = activation.codes
                guard groupSize == 128, codes.dtype == .int8, codes.ndim == 2,
                    activation.scales.dtype == .float32, activation.scaledSums.dtype == .float32,
                    weight.dtype == .uint32, scales.dtype == .float16, biases.dtype == .float16,
                    [DType.float16, .float32].contains(outputDType)
                else { return nil }
                let m = codes.dim(0)
                let k = codes.dim(1)
                let n = weight.dim(0)
                guard m == 16, n % 32 == 0, k % 512 == 0, weight.dim(1) == k / 16,
                    activation.scales.shape == [m, k / 128],
                    activation.scaledSums.shape == [m, k / 128],
                    scales.shape == [n, k / 128], biases.shape == [n, k / 128]
                else { return nil }
                let choice = narrowKernel(
                    cache, scales, biases, k: k, n: n, outputDType: outputDType, materialize: false)
                let scalesT: MLXArray
                let biasesT: MLXArray
                switch choice.form {
                case .negativeBiasF32Scales:
                    scalesT = narrowScalesF32(cache, scales, materialize: false)
                    biasesT = scalesT
                case .negativeBias:
                    scalesT = cache.derived(scales, tag: 1) { $0.transposed(1, 0).contiguous() }
                    biasesT = scalesT
                case .base:
                    scalesT = cache.derived(scales, tag: 1) { $0.transposed(1, 0).contiguous() }
                    biasesT = cache.derived(biases, tag: 2) { $0.transposed(1, 0).contiguous() }
                }
                let words = narrowTiled
                    ? narrowTiledWeight(cache, weight, materialize: false) : weight
                // The capture verify's head (`Qwen35HeadTopTwo.capture`): its
                // shape and kernel are recorded for the load-time self-test,
                // and where the fused form is on and passed it, the fused top
                // two of the same launch (same words, tiled or not) rides
                // beside the lazy logits.
                if Qwen35HeadTopTwo.capturing, outputDType == .float32 {
                    if headTop2Site == nil {
                        headTop2Site = (k, n, {
                            narrowKernel(
                                cache, scales, biases, k: k, n: n, outputDType: .float32,
                                materialize: false)
                        })
                    }
                    if Qwen35HeadTopTwo.on, headTop2Applies(k: k, n: n, kernel: choice) {
                        Qwen35HeadTopTwo.captured = launchNarrowInt8Top2(
                            codes, words, scalesT, biasesT, activation.scales,
                            activation.scaledSums, k: k, n: n, kernel: choice, tiled: narrowTiled)
                        Qwen35HeadTopTwo.announce(
                            "\(choice)\(narrowTiled ? " tiled" : ""), k \(k), n \(n)")
                    }
                }
                return launchNarrowInt8(
                    codes, words, scalesT, biasesT, activation.scales, activation.scaledSums,
                    k: k, n: n, outputDType: outputDType, kernel: choice, tiled: narrowTiled)
            }
            installNarrowChoice()
        } else if verifyEnabled, verifyForm != .none {
            HadamardQuantizedLinear.tensorPackedMatmulNarrow = {
                rotated, sums, weight, scales, biases, groupSize, outputDType, cache in
                guard groupSize == 128, rotated.dtype == .float16, rotated.ndim == 2,
                    sums.dtype == .float32, weight.dtype == .uint32,
                    scales.dtype == .float16, biases.dtype == .float16,
                    [DType.float16, .float32].contains(outputDType)
                else { return nil }
                let m = rotated.dim(0)
                let k = rotated.dim(1)
                let n = weight.dim(0)
                guard m == 16, n % 32 == 0, k % 512 == 0, weight.dim(1) == k / 16,
                    sums.shape == [m, k / 128], scales.shape == [n, k / 128],
                    biases.shape == [n, k / 128]
                else { return nil }
                let scalesT = cache.derived(scales, tag: 1) { $0.transposed(1, 0).contiguous() }
                let biasesT = cache.derived(biases, tag: 2) { $0.transposed(1, 0).contiguous() }
                if verifyForm == .staged8 {
                    return kernelNarrowStaged8(
                        [rotated, weight, scalesT, biasesT, sums, dimsArray(k: k, m: m, n: n)],
                        template: [("OutT", outputDType)],
                        grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
                        outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
                }
                let tn = (narrowTileColumns == 64 && n % 64 == 0) ? 64 : 32
                // Eight simdgroups split a long K (down_proj); four otherwise.
                let sg = (narrowDeepSplit && k >= 8192 && (k / 128) % 8 == 0) ? 8 : 4
                return kernelNarrow(
                    [rotated, weight, scalesT, biasesT, sums, dimsArray(k: k, m: m, n: n)],
                    template: [("OutT", outputDType), ("TN", tn), ("SG", sg)],
                    grid: (n / tn * 32 * sg, 1, 1), threadGroup: (32 * sg, 1, 1),
                    outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
            }
        }
        HadamardQuantizedLinear.tensorPackedMatmul = {
            activation, weight, scales, biases, groupSize, outputDType, cache in
            let codes = activation.codes
            guard groupSize == 128, codes.dtype == codesDType, codes.ndim == 2,
                activation.scales.dtype == .float32, activation.scaledSums.dtype == .float32,
                weight.dtype == .uint32, scales.dtype == .float16, biases.dtype == .float16,
                [DType.float16, .float32].contains(outputDType)
            else { return nil }
            let m = codes.dim(0)
            let k = codes.dim(1)
            let n = weight.dim(0)
            guard m % 64 == 0, n % 64 == 0, k % 512 == 0, weight.dim(1) == k / 16,
                activation.scales.shape == [m, k / 128],
                activation.scaledSums.shape == [m, k / 128],
                scales.shape == [n, k / 128], biases.shape == [n, k / 128]
            else { return nil }
            // The verify int8 kernel's per-projection proof and FP32 scales are
            // built here, at the first prompt forward (the load-time warm), so
            // no verify round pays the readback or the widening.
            prepareNarrowOperands(cache, weight, scales, biases)
            let scalesT = cache.derived(scales, tag: 1) { $0.transposed(1, 0).contiguous() }
            let biasesT = cache.derived(biases, tag: 2) { $0.transposed(1, 0).contiguous() }
            let foldedSums = cache.derived(scales, tag: 3) { s in
                (s.asType(.float32) * codeSums(weight, k: k) * MLXArray(Float(-128)))
                    .transposed(1, 0).contiguous()
            }
            let packedKernel: MLXFast.MLXFastKernel
            var template: [(String, any KernelTemplateArg)] = [
                ("OutT", outputDType), ("MPERM", rowTiledConstants ? 1 : 0),
                ("SIGNED", signedCodes ? 1 : 0),
            ]
            switch support {
            case .native2b: packedKernel = kernel
            case .staged8:
                packedKernel = kernelStaged8
                template.append(("NEGATIVE_SCALE_BIAS",
                    cache.biasesAreNegativeScales(scales, biases) ? 1 : 0))
                template.append(("FACTORED", factoredPromptEpilogue ? 1 : 0))
                template.append(("TILED", narrowTiled ? 1 : 0))
            default: packedKernel = kernelStaged
            }
            // The prompt route reads the verify route's tiled copy too, so the
            // copy is in use in every phase (a copy only the verify window
            // read could lose its GPU residency between windows).
            let words = support == .staged8 && narrowTiled
                ? narrowTiledWeight(cache, weight, materialize: true) : weight
            return packedKernel(
                [codes, words, scalesT, biasesT, foldedSums, activation.scales,
                 activation.scaledSums, dimsArray(k: k, m: m, n: n)],
                template: template,
                grid: (n / 64 * 128, m / 64, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
        }
    }
}

// MARK: - The verify head's top two, fused into the int8 head launch

/// The capture verify's vocabulary head without its FP32 logits store (D2),
/// behind an in-situ timed trial.
///
/// On the int8 verify route the head's launch (`launchNarrowInt8`, n = 248320)
/// stores `[16, 248320]` FP32 logits, 15.9 MB a round, and the policy top two
/// (`qwen35MTPTopTwoRows`) reads them all back; the acceptance packet takes
/// only each row's top-1 id and top-two values. The fused form keeps, per
/// threadgroup and row, the top two (value, column) of its columns after its
/// K-split reduction in place of the logit stores
/// (`Qwen35TensorPackedMatmul.headTop2Source`, `[16, blocks, 2]`), and one
/// simdgroup per row merges the blocks (`headTop2MergeKernel`). The order is
/// `cbv2TopTwoRows`'s: value descending, the lower id on exact ties (signed
/// zeros tie), NaN last; the top two under that total order does not depend
/// on how the candidates are grouped, and each candidate is the value the
/// stock launch would have stored, so ids and values are the same bits.
///
/// Routing: only the capture verify's head call (`capture`), only where the
/// int8 route takes the 16 rows with the kernel whose fused form passed the
/// load-time self-test, and only while `on`. The FP32 logits stay in the graph
/// as the lazy stock launch: nothing on the greedy round evaluates them, and
/// any caller that asks gets them as before. `cbv2MTPTopTwo` takes the fused
/// pair only for the logits array this capture returned (`lookup`, by
/// identity), so a round reads the pair its own head launch built.
///
/// `on` is OFF by default. After the verify kernels' in-situ trial (so the
/// head's kernel is final) the deferred load warm self-tests the fused form
/// of that kernel (`prepareHeadTop2`) and times real engine rounds with it
/// off and on, interleaved (`Trial`); `on` is adopted only if its median
/// round beats off's by more than 0.5 %. `MLXFAST_HEAD_TOP2=1` turns it on
/// after the self-test without a trial, `=0` keeps it off with neither.
enum Qwen35HeadTopTwo {
    /// `MLXFAST_HEAD_TOP2`: true / false force the choice (no trial), nil
    /// leaves it to the trial.
    static let forced: Bool? = {
        switch ProcessInfo.processInfo.environment["MLXFAST_HEAD_TOP2"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        {
        case "1", "true", "yes", "on": return true
        case "0", "false", "no", "off": return false
        default: return nil
        }
    }()

    /// Whether the head launch takes its fused form: read at every capture
    /// verify's head launch (graph build), set by the trial (or `forced`).
    nonisolated(unsafe) static var on = false

    /// Set while the capture verify builds its head call; the int8 head
    /// launch then leaves its fused pair in `captured`.
    nonisolated(unsafe) static var capturing = false
    nonisolated(unsafe) static var captured: (ids: MLXArray, values: MLXArray)?
    /// Rounds whose policy top two came from the fused pair (`lookup` hits).
    nonisolated(unsafe) static var hits = 0

    private final class Entry {
        weak var logits: MLXArray?
        let ids: MLXArray
        let values: MLXArray
        init(logits: MLXArray, ids: MLXArray, values: MLXArray) {
            self.logits = logits
            self.ids = ids
            self.values = values
        }
    }

    nonisolated(unsafe) private static var last: Entry?
    nonisolated(unsafe) private static var announced = false

    /// One line at the first head launch that takes the fused form.
    static func announce(_ what: String) {
        guard !announced else { return }
        announced = true
        FileHandle.standardError.write("bonsai head top-2: in use (\(what))\n".data(using: .utf8)!)
    }

    /// `head()` (the capture verify's `lmHead(normalized)`), remembering the
    /// fused top two of the logits it returns when the int8 head launch
    /// produced one.
    static func capture(_ head: () -> MLXArray) -> MLXArray {
        guard forced != false else { return head() }
        captured = nil
        capturing = true
        let logits = head()
        capturing = false
        last = captured.map { Entry(logits: logits, ids: $0.ids, values: $0.values) }
        captured = nil
        return logits
    }

    /// The fused `[rows, 2]` ids and values of `logits` when they are the
    /// array the last `capture` returned (`[B, L, V]`, `B * L <= 16` rows).
    static func lookup(_ logits: MLXArray, rows: Int) -> (ids: MLXArray, values: MLXArray)? {
        guard let entry = last, let held = entry.logits, held === logits,
            rows >= 1, rows <= entry.ids.dim(0)
        else { return nil }
        hits += 1
        if rows == entry.ids.dim(0) { return (entry.ids, entry.values) }
        return (entry.ids[0 ..< rows], entry.values[0 ..< rows])
    }

    /// The in-situ trial (as `Qwen35TensorPackedMatmul.NarrowInSituTrial`):
    /// one load-time engine request whose block proposals switch `on` off and
    /// on in turn (`roundBoundary`, from `proposeBlock`); round r's time is
    /// the host time from its proposal to the next. After the discarded first
    /// round, `roundsPerArm` timed rounds per arm; rounds above 1.5x their
    /// arm's median are dropped; on is adopted only if its median beats off's
    /// by more than `adoptMargin`. Every round's tokens are the same either
    /// way (the fused pair is bitwise the stock top two).
    enum Trial {
        nonisolated(unsafe) static var active = false
        nonisolated(unsafe) static var roundTimes: [[UInt64]] = [[], []]
        nonisolated(unsafe) static var roundIndex = 0
        nonisolated(unsafe) static var lastBoundary: UInt64 = 0
        nonisolated(unsafe) static var onEnough: (() -> Void)?

        static let roundsPerArm = 8
        static let adoptMargin = 0.005
        static var roundsNeeded: Int { 2 + 2 * roundsPerArm }

        @inline(__always) static func roundBoundary() {
            guard active else { return }
            boundary()
        }

        private static func boundary() {
            let now = DispatchTime.now().uptimeNanoseconds
            if roundIndex >= 2 { roundTimes[(roundIndex - 1) & 1].append(now - lastBoundary) }
            lastBoundary = now
            Qwen35HeadTopTwo.on = roundIndex & 1 == 1
            roundIndex += 1
            if roundIndex >= roundsNeeded {
                active = false
                let enough = onEnough
                onEnough = nil
                enough?()
            }
        }

        static func begin(onEnough: @escaping () -> Void) {
            roundTimes = [[], []]
            roundIndex = 0
            lastBoundary = 0
            Qwen35HeadTopTwo.hits = 0
            self.onEnough = onEnough
            active = true
        }

        private static func median(_ values: [UInt64]) -> Double? {
            guard !values.isEmpty else { return nil }
            let sorted = values.sorted()
            let mid = sorted.count / 2
            return sorted.count % 2 == 1
                ? Double(sorted[mid]) : (Double(sorted[mid - 1]) + Double(sorted[mid])) / 2
        }

        /// Ends the trial: sets `on`, logs one line. Safe when nothing ran.
        static func finish(elapsedNanoseconds: UInt64) {
            active = false
            onEnough = nil
            let arms: [(Double?, Int)] = roundTimes.map { times in
                guard let first = median(times) else { return (nil, 0) }
                let kept = times.filter { Double($0) <= 1.5 * first }
                return (median(kept), kept.count)
            }
            var adopt = false
            if let off = arms[0].0, let fused = arms[1].0 { adopt = fused < off * (1 - adoptMargin) }
            Qwen35HeadTopTwo.on = adopt
            Qwen35HeadTopTwo.last = nil
            func ms(_ value: Double?) -> String { value.map { String(format: "%.2f", $0 / 1e6) } ?? "-" }
            let log = "bonsai head top-2 trial: off \(ms(arms[0].0)) ms, on \(ms(arms[1].0)) ms, adopted "
                + (adopt ? "on" : "off")
                + " (kept \(arms[0].1)/\(roundTimes[0].count) off, \(arms[1].1)/\(roundTimes[1].count) on;"
                + " fused pair read \(Qwen35HeadTopTwo.hits)x; "
                + String(format: "%.0f ms)\n", Double(elapsedNanoseconds) / 1e6)
            FileHandle.standardError.write(log.data(using: .utf8)!)
            roundTimes = [[], []]
        }
    }
}

extension Qwen35TensorPackedMatmul {
    /// `cbv2TopTwoRows`'s running top two and its order (CBv2TopTwo.swift):
    /// value descending, token id ascending on exact ties, NaNs last; an id
    /// already held is not inserted again. `merge` inserts another state's
    /// entries; `shuffle_xor` reads a lane's state across the simdgroup. The
    /// trailing newline matters (the JIT appends the kernel signature).
    static let headTop2Header = """
        struct bonsai_head_top2 {
          float first_value;
          float second_value;
          uint first_id;
          uint second_id;
          uint count;
        };

        inline bonsai_head_top2 bonsai_head_top2_empty() {
          bonsai_head_top2 state;
          state.first_value = 0.0f;
          state.second_value = 0.0f;
          state.first_id = 0;
          state.second_id = 0;
          state.count = 0;
          return state;
        }

        inline bool bonsai_head_top2_better(
            float candidate_value, uint candidate_id, float current_value, uint current_id) {
          bool candidate_nan = isnan(candidate_value);
          bool current_nan = isnan(current_value);
          if (candidate_nan != current_nan) {
            return !candidate_nan;
          }
          if (candidate_value > current_value) {
            return true;
          }
          if (candidate_value < current_value) {
            return false;
          }
          return candidate_id < current_id;
        }

        inline void bonsai_head_top2_insert(
            thread bonsai_head_top2 &state, float value, uint id) {
          if (state.count > 0 && state.first_id == id) {
            return;
          }
          if (state.count > 1 && state.second_id == id) {
            return;
          }
          if (state.count == 0
              || bonsai_head_top2_better(value, id, state.first_value, state.first_id)) {
            if (state.count > 0) {
              state.second_value = state.first_value;
              state.second_id = state.first_id;
            }
            state.first_value = value;
            state.first_id = id;
            state.count = min(state.count + 1, 2u);
            return;
          }
          if (state.count == 1
              || bonsai_head_top2_better(value, id, state.second_value, state.second_id)) {
            state.second_value = value;
            state.second_id = id;
            state.count = 2;
          }
        }

        inline void bonsai_head_top2_merge(
            thread bonsai_head_top2 &state, bonsai_head_top2 other) {
          if (other.count > 0) {
            bonsai_head_top2_insert(state, other.first_value, other.first_id);
          }
          if (other.count > 1) {
            bonsai_head_top2_insert(state, other.second_value, other.second_id);
          }
        }

        inline bonsai_head_top2 bonsai_head_top2_shuffle_xor(bonsai_head_top2 s, ushort mask) {
          bonsai_head_top2 o;
          o.first_value = simd_shuffle_xor(s.first_value, mask);
          o.second_value = simd_shuffle_xor(s.second_value, mask);
          o.first_id = simd_shuffle_xor(s.first_id, mask);
          o.second_id = simd_shuffle_xor(s.second_id, mask);
          o.count = simd_shuffle_xor(s.count, mask);
          return o;
        }

        """

    /// The stock stores of the int8 verify kernels' reduction (simdgroup 0,
    /// four consecutive columns of row `fm + 8 mh` per step, `base` their
    /// offset in the `[16, N]` output).
    private static let headTop2StorePattern =
        #"if constexpr \(sizeof\(OutT\) == sizeof\(float\)\) \{\s*\*\(device float4\*\)\(out \+ base\) = float4\(v0, v1, v2, v3\);\s*\} else \{\s*\*\(device half4\*\)\(out \+ base\) = half4\(half\(v0\), half\(v1\), half\(v2\), half\(v3\)\);\s*\}"#

    /// The four summed values enter this lane's running top two of the row
    /// instead of being stored; the column is `base` less the row offset.
    private static let headTop2Insert = """
        {
                      // TOP2: the four columns enter this lane's top two of the row.
                      const uint ht2col = uint(base - (size_t)(fm + 8 * mh) * (size_t)N);
                      bonsai_head_top2_insert(ht2[mh], v0, ht2col);
                      bonsai_head_top2_insert(ht2[mh], v1, ht2col + 1u);
                      bonsai_head_top2_insert(ht2[mh], v2, ht2col + 2u);
                      bonsai_head_top2_insert(ht2[mh], v3, ht2col + 3u);
                    }
        """

    /// After the reduction: the four lanes holding one row pair's columns
    /// (lanes differing in bits 0 and 3 of the fragment layout) merge their
    /// states, and one of them stores the block's top two of rows `fm` and
    /// `fm + 8` at `[row, block, 0 ..< 2]`. `HT2COLS` is the kernel's columns
    /// per threadgroup.
    private static let headTop2Tail = """

        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int mh = 0; mh < 2; mh++) {
            bonsai_head_top2_merge(ht2[mh], bonsai_head_top2_shuffle_xor(ht2[mh], 1));
            bonsai_head_top2_merge(ht2[mh], bonsai_head_top2_shuffle_xor(ht2[mh], 8));
          }
          if ((lane & 9u) == 0u) {
            const int ht2blocks = N / (HT2COLS);
            const int ht2block = n0 / (HT2COLS);
            #pragma clang loop unroll(full)
            for (int mh = 0; mh < 2; mh++) {
              const size_t o = ((size_t)(fm + 8 * mh) * (size_t)ht2blocks + (size_t)ht2block) * 2;
              top_ids[o] = int(ht2[mh].first_id);
              top_ids[o + 1] = int(ht2[mh].second_id);
              top_values[o] = ht2[mh].first_value;
              top_values[o + 1] = ht2[mh].second_value;
            }
          }
        }

        """

    /// `text` (one of the int8 verify kernel bodies) with its logit stores
    /// replaced by the running top two (`headTop2Insert`) and the block's
    /// merge and store appended (`headTop2Tail`). Everything up to the
    /// summed values is the stock text, so each candidate is the value the
    /// stock kernel stores. Nil if the text no longer has the anchors.
    /// A body whose `if (sg == 0)` block sits in a loop (the zoo 2 reduction
    /// passes) marks where the running state is declared, outside it, with
    /// `/* HT2 STATE */`; the others declare it just before the block.
    static func headTop2Source(_ text: String, columns: String) -> String? {
        let open = "if (sg == 0) {"
        let marker = "/* HT2 STATE */"
        guard text.components(separatedBy: open).count == 2, !text.contains("ht2"),
            text.components(separatedBy: marker).count <= 2,
            let regex = try? NSRegularExpression(pattern: headTop2StorePattern)
        else { return nil }
        let state = "bonsai_head_top2 ht2[2] = {bonsai_head_top2_empty(), bonsai_head_top2_empty()};\n        "
        var t = text.contains(marker)
            ? text.replacingOccurrences(of: marker, with: state)
            : text.replacingOccurrences(of: open, with: state + open)
        let range = NSRange(t.startIndex..., in: t)
        guard regex.numberOfMatches(in: t, range: range) == 1,
            let match = regex.firstMatch(in: t, range: range),
            let swiftRange = Range(match.range, in: t)
        else { return nil }
        t.replaceSubrange(swiftRange, with: headTop2Insert)
        t += headTop2Tail.replacingOccurrences(of: "HT2COLS", with: columns)
        guard !t.contains("out + base"), !t.contains("out["), !t.contains("OutT(") else { return nil }
        return t
    }

    private static func headTop2Kernel(
        _ name: String, _ text: String, columns: String
    ) -> MLXFast.MLXFastKernel? {
        guard let source = headTop2Source(text, columns: columns) else { return nil }
        return MLXFast.metalKernel(
            name: name,
            inputNames: ["x", "w", "scalesT", "biasesT", "ascale", "rowsum", "ksz"],
            outputNames: ["top_ids", "top_values"],
            source: source,
            header: header + headTop2Header,
            ensureRowContiguous: true)
    }

    private static let kernelNarrowInt8Top2 = headTop2Kernel(
        "bonsai_tensor_packed_matmul_m16_i8_top2", sourceNarrowInt8, columns: "32")
    private static let kernelNarrowInt8PipelinedTop2 = headTop2Kernel(
        "bonsai_tensor_packed_matmul_m16_i8p_top2", sourceNarrowInt8Pipelined, columns: "TN")
    // The zoo bodies' fused forms (the head on a zoo body, zoo 2).
    private static let kernelNarrowInt8ZooTop2 = headTop2Kernel(
        "bonsai_tensor_packed_matmul_m16_i8z_top2", sourceNarrowInt8Zoo, columns: "TN")
    private static let kernelNarrowInt8PairTop2 = headTop2Kernel(
        "bonsai_tensor_packed_matmul_m16_i8x_top2", sourceNarrowInt8Pair, columns: "32")
    private static let kernelNarrowInt8Zoo2Top2 = headTop2Kernel(
        "bonsai_tensor_packed_matmul_m16_i8z2_top2", sourceNarrowInt8Zoo2, columns: "TN")
    private static let kernelNarrowInt8PairRTop2 = headTop2Kernel(
        "bonsai_tensor_packed_matmul_m16_i8r_top2", sourceNarrowInt8PairR, columns: "32")

    /// One simdgroup per row merges the blocks' pairs (`DFlash2TopK`'s merge
    /// pattern): each lane takes every 32nd block, then five butterfly steps.
    /// grid (32, rows, 1), threadgroup (32, 1, 1).
    private static let headTop2MergeKernel = MLXFast.metalKernel(
        name: "bonsai_head_top2_merge",
        inputNames: ["pid", "pval"],
        outputNames: ["top_ids", "top_values"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint row = threadgroup_position_in_grid.y;
            const uint blocks = uint(pid_shape[1]);
            bonsai_head_top2 st = bonsai_head_top2_empty();
            for (uint b = lane; b < blocks; b += 32) {
              const size_t o = (size_t(row) * size_t(blocks) + size_t(b)) * 2;
              bonsai_head_top2_insert(st, pval[o], uint(pid[o]));
              bonsai_head_top2_insert(st, pval[o + 1], uint(pid[o + 1]));
            }
            for (ushort m = 16; m > 0; m >>= 1) {
              bonsai_head_top2_merge(st, bonsai_head_top2_shuffle_xor(st, m));
            }
            if (lane == 0) {
              top_ids[row * 2] = int(st.first_id);
              top_ids[row * 2 + 1] = int(st.second_id);
              top_values[row * 2] = st.first_value;
              top_values[row * 2 + 1] = st.second_value;
            }
            """,
        header: headTop2Header,
        ensureRowContiguous: true)

    /// The fused form of `launchNarrowInt8` with FP32 output: the same
    /// kernel body, template and grid, returning each of the 16 rows' top
    /// two ids (`int32`) and values (`float32`) as `[16, 2]`, the layout of
    /// `qwen35MTPTopTwoRows`. Nil when the variant's fused body is missing.
    /// `tiled`: `weight` is the tiled copy (`narrowTiledWeight`), as for the
    /// stock launch; the fused body keeps the stock body's `TILED` loads.
    static func launchNarrowInt8Top2(
        _ codes: MLXArray, _ weight: MLXArray, _ scalesT: MLXArray, _ biasesT: MLXArray,
        _ ascale: MLXArray, _ rowsum: MLXArray, k: Int, n: Int, kernel: NarrowKernel,
        tiled: Bool
    ) -> (ids: MLXArray, values: MLXArray)? {
        let m = 16
        let inputs = [codes, weight, scalesT, biasesT, ascale, rowsum, dimsArray(k: k, m: m, n: n)]
        let template: [(String, any KernelTemplateArg)] = [
            ("OutT", DType.float32), ("NEG", kernel.form == .base ? 0 : 1),
            ("F32S", kernel.form == .negativeBiasF32Scales ? 1 : 0), ("TILED", tiled ? 1 : 0),
        ]
        let v = kernel.variant
        let columns = v == .v0 ? 32 : v.tn
        guard n % columns == 0, v.xtg == nil else { return nil }
        let shapes = [[m, n / columns, 2], [m, n / columns, 2]]
        let dtypes: [DType] = [.int32, .float32]
        let partial: [MLXArray]
        if let family = v.family {
            // A zoo body's fused form: the same body text, template and grid
            // as its stock launch (`launchNarrowInt8`), on the tiled copy only.
            guard tiled else { return nil }
            let zooTemplate = Array(template.prefix(3))
            let launch: MLXFast.MLXFastKernel?
            let t: [(String, any KernelTemplateArg)]
            var grid = (n / v.tn * 128, 1, 1)
            var threads = 128
            if let pairR = v.zoo2Pair {
                launch = pairR ? kernelNarrowInt8PairRTop2 : kernelNarrowInt8Zoo2Top2
                t = zooTemplate + v.zoo2Template.map { ($0.0, $0.1 as any KernelTemplateArg) }
                if pairR { (grid, threads) = ((n / 32 * v.threads, 1, 1), v.threads) }
            } else if family == "pair" {
                launch = kernelNarrowInt8PairTop2
                t = zooTemplate + [("PD", v.pd), ("KH", v.kh)]
                (grid, threads) = ((n / 32 * 256, 1, 1), 256)
            } else {
                launch = kernelNarrowInt8ZooTop2
                t = zooTemplate + [("PD", v.pd), ("TN", v.tn), ("KH", v.kh), ("AM", v.am)]
            }
            guard let launch else { return nil }
            partial = launch(
                inputs, template: t, grid: grid, threadGroup: (threads, 1, 1),
                outputShapes: shapes, outputDTypes: dtypes)
        } else {
            switch v {
            case .v0:
                guard let launch = kernelNarrowInt8Top2 else { return nil }
                partial = launch(
                    inputs, template: template,
                    grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
                    outputShapes: shapes, outputDTypes: dtypes)
            default:
                guard let launch = kernelNarrowInt8PipelinedTop2 else { return nil }
                partial = launch(
                    inputs, template: template + [("PD", v.pd), ("TN", v.tn), ("KH", v.kh)],
                    grid: (n / v.tn * 128, 1, 1), threadGroup: (128, 1, 1),
                    outputShapes: shapes, outputDTypes: dtypes)
            }
        }
        let merged = headTop2MergeKernel(
            partial, grid: (32, m, 1), threadGroup: (32, 1, 1),
            outputShapes: [[m, 2], [m, 2]], outputDTypes: [.int32, .float32])
        return (merged[0], merged[1])
    }

    /// The capture verify's head launch: `(k, n)` and the kernel the route
    /// resolves for it (the current choice and the head constants' proof),
    /// recorded at the first capture (the load-time verify warm).
    nonisolated(unsafe) static var headTop2Site: (k: Int, n: Int, kernel: () -> NarrowKernel)?
    /// The head shape the self-test ran on and the kernel whose fused form
    /// passed it there.
    nonisolated(unsafe) private static var headTop2Shape: [Int] = []
    nonisolated(unsafe) private static var headTop2Verified: Set<NarrowKernel> = []

    /// Whether the int8 head launch of `kernel` at `[k, n]` may take its
    /// fused form (the caller checks `Qwen35HeadTopTwo.on`).
    static func headTop2Applies(k: Int, n: Int, kernel: NarrowKernel) -> Bool {
        headTop2Shape == [k, n] && headTop2Verified.contains(kernel)
    }

    /// The load-time self-test of the fused head, after the verify kernels'
    /// in-situ trial, on the kernel the head's launch now resolves to (final
    /// for the process). On the head shape: 16 random rows over random
    /// weights; every row with exact ties at its maximum (each column's
    /// weights and scales repeated every 24 columns, so ties fall inside a
    /// block and across blocks); and repeats every 4099 columns with infinite
    /// and NaN scales in some columns (NaN and infinite logits). The fused ids
    /// and values are compared with `qwen35MTPTopTwoRows` over the stock FP32
    /// launch, as unsigned integers; any mismatch or MLX error keeps the
    /// two-kernel path. Builds the fused pipelines on the way. True when the
    /// fused form passed. Nothing runs where the head is not on the int8
    /// route or under `MLXFAST_HEAD_TOP2=0`.
    @discardableResult
    static func prepareHeadTop2() -> Bool {
        guard Qwen35HeadTopTwo.forced != false, let site = headTop2Site else { return false }
        let (k, n) = (site.k, site.n)
        let start = DispatchTime.now().uptimeNanoseconds
        let kernels = [site.kernel()]
        var log = "bonsai head top-2: "
        let base = NarrowOperands(k: k, n: n, seed: 0x7432_6865)
        func repeated(_ period: Int, specials: Bool) -> NarrowOperands {
            let columns = MLXArray((0 ..< n).map { Int32($0 % period) })
            var s = base.scalesT[0..., columns]
            var b = base.biasesT[0..., columns]
            var s32 = base.scalesT32[0..., columns]
            if specials {
                let col = MLXArray((0 ..< n).map { Int32($0) }).reshaped([1, n])
                let inf = col % 997 .== MLXArray(Int32(5))
                let nan = col % 991 .== MLXArray(Int32(7))
                let infH = MLXArray(Float16.infinity), nanH = MLXArray(Float16.nan)
                s = which(inf, infH, which(nan, nanH, s))
                b = which(inf, -infH, which(nan, (-nanH), b))
                s32 = which(inf, MLXArray(Float.infinity), which(nan, MLXArray(Float.nan), s32))
            }
            let words = base.weight[columns]
            let operands = NarrowOperands(
                k: k, n: n, codes: base.codes, weight: words,
                tiledWeight: narrowTiled ? tileNarrowWeight(words, n: n, k: k) : words,
                scalesT: s.contiguous(), biasesT: b.contiguous(), scalesT32: s32.contiguous(),
                ascale: base.ascale, rowsum: base.rowsum)
            eval(
                operands.weight, operands.tiledWeight, operands.scalesT, operands.biasesT,
                operands.scalesT32)
            return operands
        }
        var mismatchesByKernel = [NarrowKernel: Int]()
        var failed = [NarrowKernel: String]()
        var values = 0
        let cases: [() -> NarrowOperands] = [
            { base }, { repeated(24, specials: false) }, { repeated(4099, specials: true) },
        ]
        for makeOperands in cases {
            let operands = makeOperands()
            for kernel in kernels where failed[kernel] == nil {
                do {
                    try withError { error in
                        let stock = qwen35MTPTopTwoRows(
                            operands.run(kernel, .float32).reshaped([1, 16, n]))
                        guard let fused = operands.runTop2(kernel) else {
                            throw MLXFastHeadTop2Failure.message("no fused body")
                        }
                        guard fused.ids.shape == stock.ids.shape,
                            fused.values.shape == stock.values.shape,
                            fused.ids.dtype == stock.ids.dtype,
                            fused.values.dtype == stock.values.dtype
                        else { throw MLXFastHeadTop2Failure.message("shape or dtype mismatch") }
                        let count =
                            (fused.ids.view(dtype: .uint32) .!= stock.ids.view(dtype: .uint32))
                            .asType(.int32).sum()
                            + (fused.values.view(dtype: .uint32)
                                .!= stock.values.view(dtype: .uint32)).asType(.int32).sum()
                        eval(count)
                        try error.check()
                        mismatchesByKernel[kernel, default: 0] += Int(count.item(Int32.self))
                        values += 2 * stock.ids.size
                    }
                } catch {
                    failed[kernel] = "\(error)"
                }
            }
        }
        let passed = kernels.filter { failed[$0] == nil && mismatchesByKernel[$0] == 0 }
        headTop2Shape = [k, n]
        headTop2Verified = Set(passed)
        Memory.clearCache()
        let total = mismatchesByKernel.values.reduce(0, +)
        log += (passed.count == kernels.count ? "self-test passed" : "self-test FAILED")
            + " [" + kernels.map { "\($0)" + (passed.contains($0) ? "" : "!") }.joined(separator: " ")
            + (narrowTiled ? " tiled" : "")
            + "] (\(cases.count) cases x 16 rows on \(k)x\(n): random, ties every 24 columns, "
            + "ties every 4099 with inf/NaN scales; \(values) ids+values compared bitwise, "
            + "\(total) mismatches"
            + (failed.isEmpty ? "" : "; errors " + failed.map { "\($0.key): \($0.value)" }
                .joined(separator: ", "))
            + "); "
            + (passed.isEmpty ? "two-kernel path kept" : "fused head top-2 available")
            + String(
                format: "; %.0f ms\n", Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        FileHandle.standardError.write(log.data(using: .utf8)!)
        return !passed.isEmpty
    }
}

private enum MLXFastHeadTop2Failure: Error {
    case message(String)
}

extension Qwen35TensorPackedMatmul.NarrowOperands {
    init(
        k: Int, n: Int, codes: MLXArray, weight: MLXArray, tiledWeight: MLXArray,
        scalesT: MLXArray, biasesT: MLXArray, scalesT32: MLXArray, ascale: MLXArray,
        rowsum: MLXArray
    ) {
        self.k = k
        self.n = n
        self.codes = codes
        self.weight = weight
        self.tiledWeight = tiledWeight
        self.scalesT = scalesT
        self.biasesT = biasesT
        self.scalesT32 = scalesT32
        self.ascale = ascale
        self.rowsum = rowsum
    }

    /// `run` in its fused head form (FP32 values), on the words the route
    /// reads (the tiled copy where `narrowTiled`).
    func runTop2(
        _ kernel: Qwen35TensorPackedMatmul.NarrowKernel
    ) -> (ids: MLXArray, values: MLXArray)? {
        let tiled = Qwen35TensorPackedMatmul.narrowTiled
        let (s, b): (MLXArray, MLXArray)
        switch kernel.form {
        case .base: (s, b) = (scalesT, biasesT)
        case .negativeBias: (s, b) = (scalesT, scalesT)
        case .negativeBiasF32Scales: (s, b) = (scalesT32, scalesT32)
        }
        return Qwen35TensorPackedMatmul.launchNarrowInt8Top2(
            codes, tiled ? tiledWeight : weight, s, b, ascale, rowsum, k: k, n: n,
            kernel: kernel, tiled: tiled)
    }
}
