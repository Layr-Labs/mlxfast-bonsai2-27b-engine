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
        // Groups of four consecutive i share mm and nh with c=0..3, so the
        // four outputs are consecutive columns at nb + 32*nh. Same values as
        // the scalar loop; float4/half4 stores match OutT. Alignment holds
        // under tip N/nb guards (N multiple of 64; nb 4-element aligned).
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

    /// The DFlash 2 drafter's head read (BF16, 16 rows, the vocabulary head)
    /// on the int8 verify-width route, like the target's verify head, instead
    /// of the matrix route: the BF16 hidden widened to FP32 exactly, rotated
    /// and quantized per 128-group, FP16 logits (Subflatus3 aa6a540a: 855 us
    /// against 1160 us per read on an M5 Max). Only when the verify route
    /// runs its int8 form. It changes the drafter's numerics (its proposals,
    /// so acceptance), never an emitted token: the target decides every one.
    /// Off by default; `MLXFAST_DRAFT_HEAD_INT8=1` enables it.
    static let drafterHeadInt8: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DRAFT_HEAD_INT8"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard ["1", "true", "yes", "on"].contains(value ?? "") else { return false }
        return verifyEnabled && verifyForm == .staged8
    }()

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

    /// `sourceStaged8`'s final write as one 4-wide store per group of four
    /// accumulators (dukemawex `a64c663b`): accumulators `i..i+3` (i % 4 == 0)
    /// share the row `mm` and the column half `nh`, with `c = 0..3`, so they
    /// are four consecutive columns at `nb + 32 * nh`. Each lane converts
    /// with `OutT(acc)` exactly as the scalar loop does and writes the same
    /// address, so every output bit is unchanged. `nb` is a multiple of 4
    /// (`n0 % 64 == 0`, `fn` in {0, 4, 8, 12}) and `N % 64 == 0` (the route's
    /// guard), so each store is 4-element aligned. `MLXFAST_RIDER_STAGED8_VEC=0`
    /// keeps the scalar stores.
    static let staged8VectorStores: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_RIDER_STAGED8_VEC"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The verify int8 epilogue in the prompt kernel's factored form when the
    /// offset is the negated scale: `s * (as * C - rowsum)` instead of
    /// `as * (s * C) + (-s) * rowsum`, one FMA fewer per output element and
    /// group (fkiene 1e156535). Same identity the prompt kernel uses; the two
    /// orders can differ by one FP32 rounding, so it is NOT bit-identical to
    /// the record's verify. Every verify body takes it (`v0`, the pipelined
    /// K3 bodies and the K5 producer / consumer bodies), only in the negated-
    /// offset forms; `original` (independent offsets) keeps the two-FMA form.
    /// The load-time self-test then compares each body against `v0` in the
    /// same (factored) form. Off by default; `MLXFAST_VERIFY_FACTORED=1`
    /// enables it.
    static let factoredVerifyEpilogue: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_VERIFY_FACTORED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
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
    // `FACTORED` (with `NEG` only) takes the per-group update as
    // `s * (as * C - rs)`, one FMA fewer, up to one FP32 rounding away from
    // the two-FMA form (`factoredVerifyEpilogue`, fkiene 1e156535).
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
            const device uint32_t* wrow = w + (size_t)(n0 + sc) * (K / 16) + (size_t)g * 8;
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
            if constexpr (FACTORED != 0 && NEG) {
              const float as = mh ? as1 : as0;
              const float rs = mh ? rs1 : rs0;
              acc[i] = fma(sv[nq][c], fma(as, float(cT[i]), -rs), acc[i]);
            } else {
              acc[i] = fma(mh ? as1 : as0, sv[nq][c] * float(cT[i]), fma(bv[nq][c], mh ? rs1 : rs0, acc[i]));
            }
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
    // Templates: OutT, NEG, F32S, FACTORED (as `sourceNarrowInt8`), PD, TN, KH.
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
        const device uint32_t* wrow = w + (size_t)(n0 + int(lane)) * (K / 16);
        const size_t hstride = (size_t)32 * (K / 16);
        // quads [q0, q1) of group gg's words into v
        auto getw = [&](int gg, thread uint32_t (&v)[NH][8], int q0, int q1) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NH; h++) {
            const device uint4* src = (const device uint4*)(wrow + h * hstride + (size_t)gg * 8);
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
              if constexpr (FACTORED != 0 && NEG) {
                const float as = mh ? cr[cs][1] : cr[cs][0];
                const float rs = mh ? cr[cs][3] : cr[cs][2];
                acc[h][i] = fma(sv[nq][c], fma(as, float(ci), -rs), acc[h][i]);
              } else {
                acc[h][i] = fma(mh ? cr[cs][1] : cr[cs][0], sv[nq][c] * float(ci), fma(bv[nq][c], mh ? cr[cs][3] : cr[cs][2], acc[h][i]));
              }
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

    // The verify int8 kernel as producer / consumer pairs (K5). Same
    // threadgroup shape (128 threads, 32 columns) and the same arithmetic in
    // the same order as `sourceNarrowInt8`, so the output is bitwise that
    // kernel's (self-tested at load); what changes is who does what:
    // - Simdgroups 2 + p (producers) expand the 2-bit words of their pair's
    //   groups into a ring of RD staging slots in threadgroup memory, each
    //   slot one K = KH step of the 32 columns in the base layout plus, for a
    //   group's first step, its epilogue constants as FP32 (the FP16 values
    //   widened, exact). The next group's words and constants are loaded
    //   into registers while the current one is stored.
    // - Simdgroups p (consumers) run the tensor op on filled slots and the
    //   epilogue from the slot's constants. Pair p covers K quarters 2p and
    //   2p + 1, each in its own accumulator in the same sequential group order
    //   as the base kernel's simdgroup of that quarter.
    // - Sync per slot: a `produced` / `consumed` counter per pair in
    //   threadgroup memory (fence, simdgroup barrier, one lane stores; the
    //   reader spins on the counter, then fences). No threadgroup barrier in
    //   the K loop; a slot is released right after its op, before the
    //   epilogue. The spins are bounded: a timed-out wait turns every output
    //   of the threadgroup into NaN, so it can never pass the self-test or go
    //   unnoticed.
    // - The final reduction is the base kernel's, in quarter order:
    //   ((q0 + q1) + q2) + q3, with q0 and q1 in consumer 0's registers,
    //   and its four-column vector stores (fkiene 98f554ad).
    // Templates: OutT, NEG, F32S, FACTORED (as `sourceNarrowInt8`), RD (slots
    // per pair), KH (128 or 64: K per op and slot; 64 takes two slots per group
    // into the group's zeroed int32 tile, exact as in K3).
    // grid (N / 32 * 128, 1, 1), threadgroup (128, 1, 1).
    private static let sourceNarrowInt8PC = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int gper = Kg / 4;
        const int pr = int(sg) & 1;           // pair: consumer sg pr, producer sg 2 + pr
        const int gb = pr * 2 * gper;         // the pair's groups: quarters 2 pr, 2 pr + 1
        const int ge = gb + 2 * gper;
        constexpr int HPG = 128 / KH;         // slots (K steps) per group
        constexpr int KW = KH / 16;           // 2-bit words per column per slot
        constexpr int TW = 32 * KH / 4;       // words per slot
        constexpr uint SPIN = 1u << 18;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, KH, false, true, false, KH == 64 ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate : mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)x, dextents<int, 2>(K, M));  // SIGNED codes only
        threadgroup uint4 ring[2][RD][TW / 4];
        threadgroup float4 cst[2][RD][24];    // per slot: scales[32], offsets[32], ascale[16], rowsum[16]
        threadgroup atomic_uint flags[5];     // produced[2], consumed[2], timed out
        tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> B0((threadgroup int8_t*)ring[0][0], dextents<int, 2>(KH, 32));
        auto tA0 = A.template slice<KH, 16>(0, 0);
        auto cT0 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        constexpr int CAP = 32 / 2;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[2][CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[0][i] = 0.0f; acc[1][i] = 0.0f; }
        if (sg == 0 && lane < 5) { atomic_store_explicit(&flags[lane], 0u, memory_order_relaxed); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool late = false;
        // Waits until *f >= target (one value for the whole simdgroup), bounded.
        auto waitge = [&](threadgroup atomic_uint* f, uint target) {
          for (uint it = 0; ; it++) {
            const uint v = simd_broadcast_first(atomic_load_explicit(f, memory_order_relaxed));
            if (v >= target) { break; }
            if (it >= SPIN) { late = true; break; }
          }
          atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
        };
        // Stores *f = value after every lane's prior threadgroup accesses.
        auto publish = [&](threadgroup atomic_uint* f, uint value) {
          atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if (lane == 0) { atomic_store_explicit(f, value, memory_order_relaxed); }
        };
        if (sg >= 2) {
          // producer: lane -> column lane; group g's 8 words, its column's scale
          // (and offset) and one row constant (lanes 0-15 ascale, 16-31 rowsum)
          const device uint4* wsrc = (const device uint4*)(w + (size_t)(n0 + int(lane)) * (K / 16));
          auto load = [&](int g, thread uint4 (&v)[2], thread float (&c)[3]) {
            v[0] = wsrc[2 * g]; v[1] = wsrc[2 * g + 1];
            const size_t o = (size_t)g * N + n0 + int(lane);
            if constexpr (F32S) { c[0] = ((const device float*)scalesT)[o]; }
            else { c[0] = float(((const device half*)scalesT)[o]); }
            if constexpr (!NEG) { c[1] = float(((const device half*)biasesT)[o]); } else { c[1] = 0.0f; }
            c[2] = lane < 16 ? ascale[(size_t)lane * Kg + g] : rowsum[(size_t)(lane - 16) * Kg + g];
          };
          auto put = [&](int g, thread const uint4 (&v)[2], thread const float (&c)[3]) {
            #pragma clang loop unroll(full)
            for (int h = 0; h < HPG; h++) {
              const int u = (g - gb) * HPG + h;
              const int slot = u % RD;
              if (u >= RD) { waitge(&flags[2 + pr], uint(u - RD + 1)); }
              threadgroup uint32_t* dst = (threadgroup uint32_t*)ring[pr][slot] + int(lane) * (KH / 4);
              #pragma clang loop unroll(full)
              for (int j = 0; j < KW; j++) {
                // As in the base kernel's staging: one uint4 store per word.
                const uint32_t wv = v[(h * KW + j) >> 2][(h * KW + j) & 3];
                *(threadgroup uint4*)(dst + 4 * j) = uint4(
                    wv & 0x03030303u, (wv >> 2) & 0x03030303u,
                    (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
              }
              if (h == 0) {
                threadgroup float* cs = (threadgroup float*)cst[pr][slot];
                cs[lane] = c[0];
                if constexpr (!NEG) { cs[32 + lane] = c[1]; }
                cs[64 + lane] = c[2];
              }
              publish(&flags[pr], uint(u + 1));
            }
          };
          uint4 wa[2], wb[2];
          float ca[3], cb[3];
          load(gb, wa, ca);
          for (int g = gb; g < ge; g += 2) {
            if (g + 1 < ge) { load(g + 1, wb, cb); }
            put(g, wa, ca);
            if (g + 1 < ge) {
              if (g + 2 < ge) { load(g + 2, wa, ca); }
              put(g + 1, wb, cb);
            }
          }
        } else {
          // consumer: group gb + t into accumulator a
          auto step = [&](int t, thread float (&a)[CAP]) {
            const int g = gb + t;
            float4 sv[2], bv[2];
            float as0 = 0.0f, as1 = 0.0f, rs0 = 0.0f, rs1 = 0.0f;
            if constexpr (KH == 64) {
              #pragma clang loop unroll(full)
              for (int i = 0; i < CAP; i++) { cT0[i] = 0; }
            }
            #pragma clang loop unroll(full)
            for (int h = 0; h < HPG; h++) {
              const int u = t * HPG + h;
              const int slot = u % RD;
              waitge(&flags[pr], uint(u + 1));
              if (h == 0) {
                threadgroup const float* cs = (threadgroup const float*)cst[pr][slot];
                #pragma clang loop unroll(full)
                for (int q = 0; q < 2; q++) {
                  sv[q] = *(threadgroup const float4*)(cs + fn + 16 * q);
                  if constexpr (NEG) { bv[q] = -sv[q]; }
                  else { bv[q] = *(threadgroup const float4*)(cs + 32 + fn + 16 * q); }
                }
                as0 = cs[64 + fm]; as1 = cs[64 + fm + 8];
                rs0 = cs[80 + fm]; rs1 = cs[80 + fm + 8];
              }
              const int ko = h * KH;
              threadgroup int8_t* tile = (threadgroup int8_t*)ring[pr][slot];
              tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline> Bt(tile, dextents<int, 2>(KH, 32));
              auto tA = A.template slice<KH, 16>(g * 128 + ko, 0);
              op.run(tA, Bt, cT0);
              publish(&flags[2 + pr], uint(u + 1));
            }
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i++) {
              const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
              if constexpr (FACTORED != 0 && NEG) {
                const float as = mh ? as1 : as0;
                const float rs = mh ? rs1 : rs0;
                a[i] = fma(sv[nq][c], fma(as, float(cT0[i]), -rs), a[i]);
              } else {
                a[i] = fma(mh ? as1 : as0, sv[nq][c] * float(cT0[i]), fma(bv[nq][c], mh ? rs1 : rs0, a[i]));
              }
            }
          };
          for (int t = 0; t < gper; t++) { step(t, acc[0]); }
          for (int t = gper; t < 2 * gper; t++) { step(t, acc[1]); }
        }
        if (late) { atomic_store_explicit(&flags[4], 1u, memory_order_relaxed); }
        // the reduction reuses the ring, in quarter order
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* red = (threadgroup float*)&ring[0][0][0];
        if (sg == 1) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            red[i * 32 + int(lane)] = acc[0][i];
            red[CAP * 32 + i * 32 + int(lane)] = acc[1][i];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          const bool bad = atomic_load_explicit(&flags[4], memory_order_relaxed) != 0u;
          // As in `sourceNarrowInt8` (fkiene 98f554ad): i = 0, 4, 8, 12. mh
          // and nq are constant on each group and c is 0, 1, 2, 3, so the four
          // outputs are consecutive columns. Each is still ((q0 + q1) + q2) +
          // q3 folded on its own partial: acc[0], acc[1] (this consumer's
          // quarters), then consumer 1's two from `red`. OutT is half or
          // float; both stores are 4-element aligned because fn, n0 and N are
          // multiples of 4.
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i += 4) {
            const int mh = (i >> 2) & 1;
            const int nq = i >> 3;
            float v0 = acc[0][i];
            float v1 = acc[0][i + 1];
            float v2 = acc[0][i + 2];
            float v3 = acc[0][i + 3];
            v0 += acc[1][i];
            v1 += acc[1][i + 1];
            v2 += acc[1][i + 2];
            v3 += acc[1][i + 3];
            #pragma clang loop unroll(full)
            for (int q = 0; q < 2; q++) {
              v0 += red[q * (CAP * 32) + i * 32 + int(lane)];
              v1 += red[q * (CAP * 32) + (i + 1) * 32 + int(lane)];
              v2 += red[q * (CAP * 32) + (i + 2) * 32 + int(lane)];
              v3 += red[q * (CAP * 32) + (i + 3) * 32 + int(lane)];
            }
            if (bad) {
              v0 = as_type<float>(0x7fc00000u);
              v1 = v0;
              v2 = v0;
              v3 = v0;
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

    private static let kernelNarrowInt8PC = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_i8pc",
        inputNames: ["x", "w", "scalesT", "biasesT", "ascale", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrowInt8PC,
        header: header,
        ensureRowContiguous: true)

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
        const device uint32_t* wrow = w + (size_t)(n0 + sc) * (K / 16) + sh * 4;
        auto stage = [&](int g, int buf) {
          const device uint4* src = (const device uint4*)(wrow + g * 8);
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
        if constexpr (VEC != 0) {
          // `staged8VectorStores`: accumulators i..i+3 are columns c = 0..3
          // of one row and half, one 4-wide store of the same OutT values.
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i += 4) {
            const int nh = (i >> 3) & 1;
            const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
            *(device vec<OutT, 4>*)(out + (size_t)mm * N + nb + 32 * nh) =
                vec<OutT, 4>(OutT(acc[i]), OutT(acc[i + 1]), OutT(acc[i + 2]), OutT(acc[i + 3]));
          }
        } else {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1;
            const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
            out[(size_t)mm * N + nb + c + 32 * nh] = OutT(acc[i]);
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

    // `sourceStaged8` over a TM x TN output tile (TM, TN in {64, 128}):
    // TM / 64 x TN / 64 ops of 64 x 64 x 128 per group. Every column tile
    // re-reads the whole activation, so the two ops of a 128-column tile
    // reading one A slice halve that traffic; the two ops of a 128-row tile
    // share each staged weight slice (half the 2-bit staging). DB = 2
    // double-buffers the staged slices as `sourceStaged8` does; DB = 1 keeps
    // one buffer (half the threadgroup memory, so a 128-column tile still fits
    // two threadgroups per core) and holds the next group's words in registers
    // across the ops. Each output element is still the same 64 x 64 x 128 op
    // over the same codes, then `sourceStaged8`'s epilogue (the factored form
    // under the same FACTORED condition) in the same group order: bitwise
    // identical to `sourceStaged8`, which the load-time self-test checks per
    // variant (`PromptInSituTrial`). grid: (N / TN * 128, M / TM, 1),
    // threadgroup (128, 1, 1); same inputs and templates plus TM, TN, DB.
    private static let sourceStaged8Tiled = """
        const int K = ksz[0]; const int M = ksz[1]; const int N = ksz[2];
        const int Kg = K / 128;
        constexpr int MT = TM / 64;
        constexpr int NT = TN / 64;
        const int n0 = int(threadgroup_position_in_grid.x) * TN;
        const int m0 = int(threadgroup_position_in_grid.y) * TM;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 64, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
        typedef typename metal::conditional<SIGNED != 0, int8_t, uint8_t>::type CodeT;
        tensor<device CodeT, dextents<int, 2>, tensor_inline> A((device CodeT*)xq, dextents<int, 2>(K, M));
        // staged B: TN columns x 128 codes as int8 bytes, k inner; column half
        // h (64 columns, one op's operand) at word 2048 * h of a buffer
        threadgroup uint32_t bs[DB][TN * 32];
        tensor<threadgroup CodeT, dextents<int, 2>, tensor_inline> B00((threadgroup CodeT*)bs[0], dextents<int, 2>(128, 64));
        tensor<threadgroup CodeT, dextents<int, 2>, tensor_inline> B01((threadgroup CodeT*)(bs[0] + 2048 * (NT - 1)), dextents<int, 2>(128, 64));
        tensor<threadgroup CodeT, dextents<int, 2>, tensor_inline> B10((threadgroup CodeT*)bs[DB - 1], dextents<int, 2>(128, 64));
        tensor<threadgroup CodeT, dextents<int, 2>, tensor_inline> B11((threadgroup CodeT*)(bs[DB - 1] + 2048 * (NT - 1)), dextents<int, 2>(128, 64));
        auto tA0 = A.template slice<128, 64>(0, m0);
        // cT<r><h>: the op over row tile r and column half h
        auto cT00 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B00)>, int32_t>();
        auto cT01 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B00)>, int32_t>();
        auto cT10 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B00)>, int32_t>();
        auto cT11 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B00)>, int32_t>();
        constexpr int CAP = 32;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        const int nb = n0 + 16 * int(sg & 1) + fn;
        const int mb = m0 + 16 * int(sg >> 1) + fm;
        float acc[MT * NT][CAP];
        #pragma clang loop unroll(full)
        for (int t = 0; t < MT * NT; t++) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { acc[t][i] = 0.0f; }
        }
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 32);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 32);
        const device float4* up0 = (const device float4*)(uT + nb);
        const device float4* up1 = (const device float4*)(uT + nb + 32);
        const int NQ = N / 4;
        const size_t mrow[4] = {(size_t)mb, (size_t)(mb + 8), (size_t)(mb + 32), (size_t)(mb + 40)};
        // Row-tiled constants: this lane's four rows of row tile r are adjacent
        // in 64-row tile m0 / 64 + r.
        const size_t tbase = (size_t)(m0 / 64) * (size_t)Kg * 64 + (size_t)((8 * int(sg >> 1) + fm) * 4);
        // staging: thread t -> column t >> 1 of each column half, K half t & 1
        const int sc = int(tid >> 1); const int sh = int(tid & 1);
        const device uint32_t* wrow = w + (size_t)(n0 + sc) * (K / 16) + sh * 4;
        const size_t wstep = (size_t)64 * (size_t)(K / 16);
        uint4 wq[NT];
        auto fetch = [&](int g) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NT; h++) { wq[h] = *(const device uint4*)(wrow + h * wstep + g * 8); }
        };
        // One word's four planes are four contiguous uint32s (16 codes), one
        // uint4 store each: `sourceStaged8`'s staging values and positions.
        auto put = [&](int buf) {
          #pragma clang loop unroll(full)
          for (int h = 0; h < NT; h++) {
            threadgroup uint32_t* dst = bs[buf] + 2048 * h + sc * 32 + sh * 16;
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; j++) {
              const uint32_t wv = wq[h][j];
              *(threadgroup uint4*)(dst + 4 * j) = uint4(
                  wv & 0x03030303u, (wv >> 2) & 0x03030303u,
                  (wv >> 4) & 0x03030303u, (wv >> 6) & 0x03030303u);
            }
          }
        };
        fetch(0);
        put(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < Kg; g++) {
          const int cur = DB == 2 ? (g & 1) : 0;
          if (g + 1 < Kg) {
            fetch(g + 1);
            if (DB == 2) { put(cur ^ 1); }
          }
          auto tAr0 = A.template slice<128, 64>(g * 128, m0);
          auto tAr1 = A.template slice<128, 64>(g * 128, m0 + 64 * (MT - 1));
          if (cur == 0) {
            op.run(tAr0, B00, cT00);
            if (NT == 2) { op.run(tAr0, B01, cT01); }
            if (MT == 2) { op.run(tAr1, B00, cT10); }
            if (MT == 2 && NT == 2) { op.run(tAr1, B01, cT11); }
          } else {
            op.run(tAr0, B10, cT00);
            if (NT == 2) { op.run(tAr0, B11, cT01); }
            if (MT == 2) { op.run(tAr1, B10, cT10); }
            if (MT == 2 && NT == 2) { op.run(tAr1, B11, cT11); }
          }
          float4 s0[NT], s1[NT], b0[NT], b1[NT], u0[NT], u1[NT];
          #pragma clang loop unroll(full)
          for (int h = 0; h < NT; h++) {
            s0[h] = float4(sp0[g * NQ + 16 * h]); s1[h] = float4(sp1[g * NQ + 16 * h]);
            if constexpr (NEGATIVE_SCALE_BIAS) {
              b0[h] = -s0[h]; b1[h] = -s1[h];
            } else {
              b0[h] = float4(bp0[g * NQ + 16 * h]); b1[h] = float4(bp1[g * NQ + 16 * h]);
            }
            u0[h] = 0.0f; u1[h] = 0.0f;
            if (!SIGNED) { u0[h] = up0[g * NQ + 16 * h]; u1[h] = up1[g * NQ + 16 * h]; }
          }
          float as[MT][4], rb[MT][4];
          #pragma clang loop unroll(full)
          for (int r = 0; r < MT; r++) {
            if (MPERM) {
              const size_t tb = tbase + (size_t)r * (size_t)Kg * 64 + (size_t)g * 64;
              const float4 as4 = *(const device float4*)(ascale + tb);
              const float4 rb4 = *(const device float4*)(rsb + tb);
              as[r][0] = as4.x; as[r][1] = as4.y; as[r][2] = as4.z; as[r][3] = as4.w;
              rb[r][0] = rb4.x; rb[r][1] = rb4.y; rb[r][2] = rb4.z; rb[r][3] = rb4.w;
            } else {
              #pragma clang loop unroll(full)
              for (int q = 0; q < 4; q++) {
                as[r][q] = ascale[(mrow[q] + 64 * r) * Kg + g]; rb[r][q] = rsb[(mrow[q] + 64 * r) * Kg + g];
              }
            }
          }
          #pragma clang loop unroll(full)
          for (int t = 0; t < MT * NT; t++) {
            const int r = t / NT; const int h = t % NT;
            #pragma clang loop unroll(full)
            for (int i = 0; i < CAP; i++) {
              const int c = i & 3; const int nh = (i >> 3) & 1; const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
              const int ci = r == 0 ? (h == 0 ? cT00[i] : cT01[i]) : (h == 0 ? cT10[i] : cT11[i]);
              const float s = nh ? s1[h][c] : s0[h][c];
              const float b = nh ? b1[h][c] : b0[h][c];
              const float u = nh ? u1[h][c] : u0[h][c];
              if constexpr (FACTORED != 0 && NEGATIVE_SCALE_BIAS != 0 && SIGNED != 0) {
                // `sourceStaged8`'s factored form: s * (as * C - rb).
                acc[t][i] = fma(s, fma(as[r][mh], float(ci), -rb[r][mh]), acc[t][i]);
              } else {
                const float tv = SIGNED ? s * float(ci) : fma(s, float(ci), u);
                acc[t][i] = fma(b, rb[r][mh], fma(as[r][mh], tv, acc[t][i]));
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (DB == 1 && g + 1 < Kg) {
            put(0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
        }
        #pragma clang loop unroll(full)
        for (int t = 0; t < MT * NT; t++) {
          const int r = t / NT; const int h = t % NT;
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1;
            const int mm = mb + 64 * r + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
            out[(size_t)mm * N + nb + 64 * h + c + 32 * nh] = OutT(acc[t][i]);
          }
        }
        """

    private static let kernelStaged8Tiled = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_q8_u8_tiled",
        inputNames: ["xq", "w", "scalesT", "biasesT", "uT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: sourceStaged8Tiled,
        header: header,
        ensureRowContiguous: true)

    // MARK: - Prompt int8 kernel tiles, chosen in situ

    /// A prompt-width int8-staged kernel: `original` is `sourceStaged8` (64 x
    /// 64, double-buffered; the record's kernel); any other value is
    /// `sourceStaged8Tiled` over a `tm` x `tn` output tile with `db` staging
    /// buffers.
    struct PromptTile: Hashable, CustomStringConvertible {
        var tm: Int, tn: Int, db: Int
        static let original = PromptTile(tm: 64, tn: 64, db: 0)
        var description: String { self == .original ? "base" : "\(tm)x\(tn)x\(db)" }
        /// True when this is a tiled kernel whose tile divides `[m, n]`.
        func takes(m: Int, n: Int) -> Bool { self != .original && m % tm == 0 && n % tn == 0 }
    }

    /// The candidates, in the order the self-test tries them under its
    /// deadline: the 128-column tile first (half the activation traffic), with
    /// one buffer (two threadgroups per core) and with two (overlapped
    /// staging); then the 128-row tile, the 128 x 128 tile and the one-buffer
    /// 64 x 64 tile. `128x128x2` (32 KB of staging) runs only on request.
    static let promptTileCandidates: [PromptTile] = [
        PromptTile(tm: 64, tn: 128, db: 1), PromptTile(tm: 64, tn: 128, db: 2),
        PromptTile(tm: 128, tn: 64, db: 2), PromptTile(tm: 128, tn: 64, db: 1),
        PromptTile(tm: 128, tn: 128, db: 1), PromptTile(tm: 64, tn: 64, db: 1),
    ]

    /// The prompt window's production shapes `(k, n)`, timed at 512 rows by
    /// the shortlist: qkv|z, attention qkv, o, gate|up, down.
    static let promptTunedShapes = [
        (5120, 16384), (5120, 14336), (6144, 5120), (5120, 34816), (17408, 5120),
    ]

    /// The installed prompt kernel: `original` (the record's) unless the
    /// in-situ trial adopted a tile. Read at graph build, per launch.
    nonisolated(unsafe) static var promptTileInstalled = PromptTile.original

    /// The kernel for an `[m, k] x [k, n]` prompt matmul: the installed tile
    /// where it divides `[m, n]`, else `original`.
    static func promptTile(m: Int, n: Int) -> PromptTile {
        promptTileInstalled.takes(m: m, n: n) ? promptTileInstalled : .original
    }

    /// One launch of the int8-staged prompt kernel `tile`. `template` carries
    /// OutT, MPERM, SIGNED, NEGATIVE_SCALE_BIAS and FACTORED; the tiled kernel
    /// appends its tile. `kernelStaged8` (the record's launch, unchanged)
    /// wherever the tile does not divide `[m, n]`.
    static func launchStaged8(
        _ inputs: [MLXArray], template: [(String, any KernelTemplateArg)], m: Int, n: Int,
        outputDType: DType, tile: PromptTile
    ) -> MLXArray {
        if tile.takes(m: m, n: n),
            let y = kernelStaged8Tiled(
                inputs, template: template + [("TM", tile.tm), ("TN", tile.tn), ("DB", tile.db)],
                grid: (n / tile.tn * 128, m / tile.tm, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [outputDType]
            ).first
        {
            return y
        }
        return kernelStaged8(
            inputs, template: template + [("VEC", staged8VectorStores ? 1 : 0)],
            grid: (n / 64 * 128, m / 64, 1), threadGroup: (128, 1, 1),
            outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
    }

    /// Synthetic operands for the prompt kernel self-test and shortlist: `m`
    /// rows of codes in the route's form (signed int8 when `signedCodes`),
    /// random 2-bit words, FP16 scales of both signs (zeros and signed zeros
    /// included) with offsets that are their FP16 negations bit for bit (the
    /// pattern NEGATIVE_SCALE_BIAS = 1 relies on), independent offsets for
    /// NEGATIVE_SCALE_BIAS = 0, folded code sums, and FP32 activation scales
    /// and scaled sums (read row-tiled when `rowTiledConstants`, so every
    /// 64-row tile's constants differ). Nothing depends on a request.
    private struct PromptOperands {
        let k: Int, n: Int, m: Int
        let codes: MLXArray, weight: MLXArray
        let scalesT: MLXArray, negatedT: MLXArray, biasesT: MLXArray, foldedSums: MLXArray
        let ascale: MLXArray, rsb: MLXArray

        init(k: Int, n: Int, m: Int, seed: UInt64) {
            self.k = k
            self.n = n
            self.m = m
            let kg = k / 128
            func key(_ i: UInt64) -> MLXArray { MLXRandom.key(seed &* 16 &+ i) }
            codes =
                signedCodes
                ? MLXRandom.randInt(Int32(-127) ..< Int32(128), [m, k], key: key(0)).asType(.int8)
                : MLXRandom.randInt(Int32(0) ..< Int32(256), [m, k], key: key(0)).asType(.uint8)
            weight = MLXRandom.randInt(Int32(0) ..< Int32(65536), [n, k / 8], key: key(1))
                .asType(.uint16).view(dtype: .uint32)
            var s = MLXRandom.uniform(Float(-0.05) ..< Float(0.05), [n, kg], key: key(2))
            let pick = MLXRandom.randInt(Int32(0) ..< Int32(64), [n, kg], key: key(3))
            s = which(pick .== MLXArray(Int32(0)), MLXArray(Float(0)), s)
            s = which(pick .== MLXArray(Int32(1)), MLXArray(Float(-0.0)), s)
            let scales = s.asType(.float16)
            scalesT = scales.transposed(1, 0).contiguous()
            negatedT = (scales.view(dtype: .uint16) ^ MLXArray(UInt16(0x8000)))
                .view(dtype: .float16).transposed(1, 0).contiguous()
            biasesT = MLXRandom.uniform(Float(-0.05) ..< Float(0.05), [kg, n], key: key(4))
                .asType(.float16)
            foldedSums = MLXRandom.normal([kg, n], key: key(5)) * Float(100)
            ascale = MLXRandom.uniform(Float(0.0001) ..< Float(0.05), [m, kg], key: key(6))
            rsb = MLXRandom.normal([m, kg], key: key(7)) * Float(50)
            eval(codes, weight, scalesT, negatedT, biasesT, foldedSums, ascale, rsb)
        }

        /// The production template (this box's MPERM, SIGNED and FACTORED).
        func run(_ tile: PromptTile, _ outputDType: DType, negativeScaleBias: Bool) -> MLXArray {
            let template: [(String, any KernelTemplateArg)] = [
                ("OutT", outputDType), ("MPERM", rowTiledConstants ? 1 : 0),
                ("SIGNED", signedCodes ? 1 : 0),
                ("NEGATIVE_SCALE_BIAS", negativeScaleBias ? 1 : 0),
                ("FACTORED", factoredPromptEpilogue ? 1 : 0),
            ]
            return launchStaged8(
                [codes, weight, scalesT, negativeScaleBias ? negatedT : biasesT, foldedSums,
                 ascale, rsb, dimsArray(k: k, m: m, n: n)],
                template: template, m: m, n: n, outputDType: outputDType, tile: tile)
        }
    }

    /// The in-situ choice of the prompt int8 kernel: the record's
    /// `sourceStaged8` or a `sourceStaged8Tiled` tile, chosen by timing the
    /// real 512-row prompt forward of the load-time prompt warm
    /// (`Qwen35DFlash2Assistant.warmTargetPrefill`).
    ///
    /// Self-test: each candidate tile runs against `original` on synthetic
    /// 512-row operands (three shapes, `PromptOperands`) in the production
    /// template (this box's MPERM, SIGNED and FACTORED) and must match every
    /// output bit with FP16 output and NEGATIVE_SCALE_BIAS = 1; a mismatch or
    /// any MLX error (a failed compile included) drops it, and a candidate not
    /// started within 1.5 s is skipped. Shortlist: the survivors run
    /// alternately with `original` on the five production shapes (best of
    /// five); the fastest in total, in order, each after it also matches with
    /// FP32 output and with NEGATIVE_SCALE_BIAS = 0 (independent offsets, both
    /// outputs), up to `tiledCandidates` of them. The synthetic timing only
    /// ranks: its short bursts never reach the prefill's steady GPU state.
    ///
    /// Trial: the record's kernel and each shortlisted tile are installed in
    /// turn for one whole prompt forward (fresh throwaway state each time, the
    /// warm's fixed prompt), `1 + timedRuns` forwards per candidate in
    /// serpentine order (forward, then reversed, so a clock ramp over the
    /// trial does not favour any position); each forward is timed on the host
    /// from its graph build to the end of its evaluation, and the first run of
    /// each candidate is discarded. A tile is adopted only when its median
    /// forward beats the record's kernel's by more than 0.5 %; otherwise the
    /// record's kernel stays. One stderr line, then `Memory.clearCache()`.
    /// Every candidate is the same arithmetic bit for bit, so the forwards
    /// and every served result are the same whichever wins; nothing runs in a
    /// served request.
    ///
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_PROMPT_INSITU=off` keeps the record's
    /// kernel with no self-test and no trial; a number (1 to 6) sets
    /// `tiledCandidates`. `DARKBLOOM_BONSAI_TENSOR_ROUTE_PROMPT_TILE=off`
    /// also keeps the record's kernel; a comma list of `TMxTNxDB` (for
    /// example `64x128x1,128x64x2`) replaces the candidates.
    enum PromptInSituTrial {
        /// Timed forwards per candidate (after the discarded first).
        static let timedRuns = 3
        /// A tile must beat the record's kernel by more than this.
        static let adoptMargin = 0.005

        private static func knob(_ name: String) -> String {
            ProcessInfo.processInfo.environment[name]?
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
        }
        private static let off: Set<String> = ["off", "0", "false", "no", "base"]

        /// Shortlisted tiles per trial (the record's kernel comes on top).
        static let tiledCandidates: Int = {
            if let value = Int(knob("DARKBLOOM_BONSAI_TENSOR_ROUTE_PROMPT_INSITU")), value > 0 {
                return min(value, 6)
            }
            return 2
        }()

        /// The candidate tiles: `promptTileCandidates`, or the
        /// `..._PROMPT_TILE` list; empty under either kill switch.
        static let candidates: [PromptTile] = {
            if off.contains(knob("DARKBLOOM_BONSAI_TENSOR_ROUTE_PROMPT_INSITU")) { return [] }
            let raw = knob("DARKBLOOM_BONSAI_TENSOR_ROUTE_PROMPT_TILE")
            if off.contains(raw) { return [] }
            let requested = raw.split(separator: ",").compactMap { item -> PromptTile? in
                let v = item.split(separator: "x").compactMap { Int($0) }
                guard v.count == 3, [64, 128].contains(v[0]), [64, 128].contains(v[1]),
                    [1, 2].contains(v[2])
                else { return nil }
                return PromptTile(tm: v[0], tn: v[1], db: v[2])
            }
            return requested.isEmpty ? promptTileCandidates : requested
        }()

        /// Whether a trial can run: the int8-staged prompt route is installed
        /// and a candidate is left after the kill switches.
        static var armed: Bool {
            enabled && installed && support == .staged8 && !candidates.isEmpty
        }

        nonisolated(unsafe) private static var done = false

        /// The self-test and the shortlist: the trial's candidates, the
        /// record's kernel first; `log` gets their outcome.
        private static func shortlist(_ log: inout String) -> [PromptTile] {
            let start = DispatchTime.now().uptimeNanoseconds
            func elapsedMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 }
            var passed: [PromptTile] = []
            var failed: [PromptTile] = []
            var skipped: [PromptTile] = []
            var failedLate: [PromptTile] = []
            var totals: [PromptTile: Double] = [:]
            var sets: [PromptTile] = [.original]
            do {
                try withError { error in
                    let testOps = [(5120, 1024, UInt64(91)), (17408, 256, UInt64(92)), (1536, 384, UInt64(93))]
                        .map { PromptOperands(k: $0.0, n: $0.1, m: 512, seed: $0.2) }
                    try error.check()
                    // True when `tile` matches `original` bit for bit in every
                    // `(output dtype, NEGATIVE_SCALE_BIAS)` form; false on a
                    // mismatch or on any MLX error inside (scoped to this call).
                    func matches(_ tile: PromptTile, _ forms: [(DType, Bool)]) -> Bool {
                        do {
                            return try withError { scoped in
                                for (outputDType, negative) in forms {
                                    let bits: DType = outputDType == .float16 ? .uint16 : .uint32
                                    for ops in testOps {
                                        let reference = ops.run(.original, outputDType, negativeScaleBias: negative)
                                        let y = ops.run(tile, outputDType, negativeScaleBias: negative)
                                        let differ = (y.view(dtype: bits) .!= reference.view(dtype: bits))
                                            .asType(.int32).sum()
                                        eval(differ)
                                        try scoped.check()
                                        if differ.item(Int32.self) != 0 { return false }
                                    }
                                }
                                return true
                            }
                        } catch {
                            return false
                        }
                    }
                    for tile in candidates {
                        if elapsedMs() > 1500 { skipped.append(tile); continue }
                        if matches(tile, [(.float16, true)]) { passed.append(tile) } else { failed.append(tile) }
                    }
                    guard !passed.isEmpty else { return }

                    let kernels = [PromptTile.original] + passed
                    let shapeSets = promptTunedShapes.enumerated().map { (index, shape) in
                        PromptOperands(k: shape.0, n: shape.1, m: 512, seed: 100 + UInt64(index))
                    }
                    for kernel in kernels {
                        eval(shapeSets.map { $0.run(kernel, .float16, negativeScaleBias: true) })
                    }
                    try error.check()
                    var best: [PromptTile: [Double]] = [:]
                    for kernel in kernels { best[kernel] = Array(repeating: .infinity, count: shapeSets.count) }
                    for _ in 0 ..< 5 {
                        for (index, ops) in shapeSets.enumerated() {
                            for kernel in kernels {
                                let outs = (0 ..< 2).map { _ in
                                    ops.run(kernel, .float16, negativeScaleBias: true)
                                }
                                let t0 = DispatchTime.now().uptimeNanoseconds
                                eval(outs)
                                let us = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000
                                    / Double(outs.count)
                                best[kernel]![index] = min(best[kernel]![index], us)
                            }
                        }
                    }
                    try error.check()
                    for kernel in kernels { totals[kernel] = best[kernel]!.reduce(0, +) }
                    for tile in passed.sorted(by: { totals[$0]! < totals[$1]! })
                    where sets.count <= tiledCandidates {
                        if matches(tile, [(.float32, true), (.float16, false), (.float32, false)]) {
                            sets.append(tile)
                        } else {
                            failedLate.append(tile)
                        }
                    }
                }
            } catch {
                sets = [.original]
                log += " error \(error);"
            }
            log += " self-test passed [" + passed.map(\.description).joined(separator: " ") + "]"
            if !failed.isEmpty { log += " failed [" + failed.map(\.description).joined(separator: " ") + "]" }
            if !skipped.isEmpty { log += " skipped [" + skipped.map(\.description).joined(separator: " ") + "]" }
            if !failedLate.isEmpty {
                log += " failed FP32/offset forms [" + failedLate.map(\.description).joined(separator: " ") + "]"
            }
            if !totals.isEmpty {
                log += "; synthetic us/launch over the 5 shapes [" + ([PromptTile.original] + passed)
                    .compactMap { k in totals[k].map { "\(k)=" + String(format: "%.0f", $0) } }
                    .joined(separator: " ") + "]"
            }
            log += String(format: " (%.0f ms)", elapsedMs())
            return sets
        }

        private static func median(_ values: [UInt64]) -> Double? {
            guard !values.isEmpty else { return nil }
            let sorted = values.sorted()
            let mid = sorted.count / 2
            return sorted.count % 2 == 1
                ? Double(sorted[mid]) : (Double(sorted[mid - 1]) + Double(sorted[mid])) / 2
        }

        /// Runs the self-test, the shortlist and the trial once. `forward`
        /// runs one whole prompt forward on fresh throwaway state and returns
        /// its host nanoseconds (nil when it could not run, which ends the
        /// trial with the record's kernel). Leaves the choice installed.
        static func run(forward: () -> UInt64?) {
            guard armed, !done else { return }
            done = true
            let start = DispatchTime.now().uptimeNanoseconds
            promptTileInstalled = .original
            var log = "bonsai prompt int8 in-situ:"
            let sets = shortlist(&log)
            Memory.clearCache()
            var times: [[UInt64]] = Array(repeating: [], count: sets.count)
            var complete = sets.count >= 2
            if complete {
                trial: for rep in 0 ... timedRuns {
                    let order = rep % 2 == 0 ? Array(sets.indices) : Array(sets.indices.reversed())
                    for index in order {
                        promptTileInstalled = sets[index]
                        guard let ns = forward() else { complete = false; break trial }
                        if rep > 0 { times[index].append(ns) }
                    }
                }
            }
            promptTileInstalled = .original
            let medians = times.map { median($0) }
            var chosen = 0
            if complete, let record = medians[0] {
                var best = record
                for (index, m) in medians.enumerated().dropFirst() {
                    if let m, m < best { best = m; chosen = index }
                }
                if chosen != 0, !(best < record * (1 - adoptMargin)) { chosen = 0 }
            }
            promptTileInstalled = sets[chosen]
            if sets.count >= 2 {
                log += "; median ms/forward (512 rows, \(timedRuns) of \(timedRuns + 1) runs)"
                for (index, tile) in sets.enumerated() {
                    log += (index == 0 ? " [record " : " | ") + "\(tile) "
                        + (medians[index].map { String(format: "%.2f", $0 / 1e6) } ?? "-")
                }
                log += "]"
            }
            if !complete, sets.count >= 2 {
                log += "; a forward did not run, keeping the record's kernel"
            } else if chosen == 0 {
                log += "; keeping the record's kernel"
            } else if let record = medians[0], let m = medians[chosen] {
                log += "; adopted \(sets[chosen])" + String(format: " (%+.2f%%)", (m / record - 1) * 100)
            }
            log += String(format: "; %.0f ms\n", Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            FileHandle.standardError.write(log.data(using: .utf8)!)
            Memory.clearCache()
        }
    }

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
    /// memory); `pcN` / `k64pcN` are `sourceNarrowInt8PC` (K5: two producer
    /// simdgroups expanding codes into N staging slots per pair, two consumer
    /// simdgroups running the op, synchronised by threadgroup-memory
    /// counters) with K = 128 / 64 per slot. All bitwise identical.
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
        case pc2 = 10
        case k64pc2 = 11
        case k64pc4 = 12

        /// Words ring depth (K5: staging slots per pair), columns per
        /// threadgroup, K per op.
        var pd: Int {
            switch self {
            case .v0, .pd1, .tn64, .k64pd1: return 1
            case .pd2, .k64pd2, .pc2, .k64pc2: return 2
            case .pd3, .k64pd3: return 3
            case .pd4, .k64pd4, .k64pc4: return 4
            }
        }
        var tn: Int { self == .tn64 ? 64 : 32 }
        var kh: Int {
            [.k64pd1, .k64pd2, .k64pd3, .k64pd4, .k64pc2, .k64pc4].contains(self) ? 64 : 128
        }
        /// The producer / consumer body (K5).
        var producerConsumer: Bool { [.pc2, .k64pc2, .k64pc4].contains(self) }

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

    /// The in-situ choice of the verify int8 kernels: a few candidate choices
    /// (shortlisted by the synthetic self-test and timing at load) run in turn
    /// on real speculative rounds of the load-time engine warm, and a
    /// candidate replaces the record's pick only when its median round beats
    /// the record's pick's by more than 0.5 %. The synthetic timing streams
    /// each kernel for 0.5-0.7 ms bursts in which the GPU never reaches the
    /// clock state of the decode window, so it mispicks; a whole round does
    /// reach it. Every candidate passed the bitwise self-test (FP16 and FP32
    /// outputs), so the rounds and their tokens are the same whichever runs.
    ///
    /// `roundBoundary()` runs at the top of every block proposal (one static
    /// bool check when no trial is active): round r's time is the host time
    /// from its proposal to the next one, and the choice installed at a
    /// boundary is what the next verify graph build reads. The first round
    /// after the seed is discarded. Runs only inside the deferred load warm
    /// (`Qwen35DFlash2Assistant`), never in a served request.
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_INSITU=off` keeps the record's
    /// pick without a trial.
    enum NarrowInSituTrial {
        nonisolated(unsafe) static var active = false
        nonisolated(unsafe) static var sets: [NarrowChoice] = []
        nonisolated(unsafe) static var roundTimes: [[UInt64]] = []
        nonisolated(unsafe) static var roundIndex = 0
        nonisolated(unsafe) static var lastBoundary: UInt64 = 0
        nonisolated(unsafe) static var onEnough: (() -> Void)?

        /// Timed rounds per candidate.
        static let roundsPerSet = 8
        /// A candidate must beat the record's pick by more than this.
        static let adoptMargin = 0.005
        /// Rounds above this multiple of their candidate's median are dropped.
        static let outlierFactor = 1.5

        static let enabled: Bool = {
            let value = ProcessInfo.processInfo.environment[
                "DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_INSITU"]?
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            return !["0", "false", "no", "off"].contains(value ?? "")
        }()

        /// Whether a trial is set up (two or more distinct candidates; the
        /// first is the record's pick).
        static var armed: Bool { enabled && sets.count >= 2 }

        /// Proposals the trial needs: the discarded round, the seed-side
        /// boundary, then `roundsPerSet` per candidate.
        static var roundsNeeded: Int { 2 + roundsPerSet * sets.count }

        @inline(__always) static func roundBoundary() {
            guard active else { return }
            boundary()
        }

        private static func boundary() {
            let now = DispatchTime.now().uptimeNanoseconds
            let count = sets.count
            if roundIndex >= 2 {
                roundTimes[(roundIndex - 1) % count].append(now - lastBoundary)
            }
            lastBoundary = now
            let set = sets[roundIndex % count]
            narrowDefault = set.0
            narrowByShape = set.1
            roundIndex += 1
            if roundIndex >= roundsNeeded {
                active = false
                let enough = onEnough
                onEnough = nil
                enough?()
            }
        }

        /// Arms the round hook. `onEnough` runs (on the engine's thread) once
        /// every candidate has its rounds.
        static func begin(onEnough: @escaping () -> Void) {
            guard armed else { return }
            roundTimes = Array(repeating: [], count: sets.count)
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

        static func describe(_ set: NarrowChoice) -> String {
            guard !set.1.isEmpty else { return "\(set.0)" }
            return "\(set.0){"
                + narrowTunedShapes.map { "\(set.1[[$0.0, $0.1]] ?? set.0)" }
                .joined(separator: ",") + "}"
        }

        /// Ends the trial: installs the winner (or the record's pick), logs
        /// one line and disarms. Safe to call when nothing ran.
        static func finish(elapsedNanoseconds: UInt64) {
            active = false
            onEnough = nil
            guard armed else { return }
            if roundTimes.count != sets.count {
                roundTimes = Array(repeating: [], count: sets.count)
            }
            let medians: [(Double?, Int, Int)] = roundTimes.map { times in
                guard let first = median(times) else { return (nil, 0, 0) }
                let kept = times.filter { Double($0) <= outlierFactor * first }
                return (median(kept), kept.count, times.count)
            }
            var chosen = 0
            if let record = medians[0].0 {
                var best = record
                for (index, entry) in medians.enumerated().dropFirst() {
                    if let m = entry.0, m < best { best = m; chosen = index }
                }
                if chosen != 0, !(best < record * (1 - adoptMargin)) { chosen = 0 }
            }
            let set = sets[chosen]
            narrowDefault = set.0
            narrowByShape = set.1
            setNarrowOperandNeeds([set])
            var log = "bonsai verify int8 in-situ: \(roundIndex) proposals; median ms/round"
            for (index, candidate) in sets.enumerated() {
                let (m, kept, total) = medians[index]
                log += (index == 0 ? " [record " : " | ") + describe(candidate) + " "
                    + (m.map { String(format: "%.2f", $0 / 1e6) } ?? "-") + " (\(kept)/\(total))"
            }
            log += "]; "
            if chosen == 0 {
                log += "keeping the record's pick"
            } else if let record = medians[0].0, let m = medians[chosen].0 {
                log += "adopted " + describe(set)
                    + String(format: " (%+.2f%%)", (m / record - 1) * 100)
            }
            log += String(format: "; %.0f ms\n", Double(elapsedNanoseconds) / 1e6)
            FileHandle.standardError.write(log.data(using: .utf8)!)
            sets = []
            roundTimes = []
        }
    }

    /// The kernel for this projection: the chosen one for its shape when its
    /// constants pass the proof (or it needs none), else `original`.
    /// `materialize` evaluates the FP32 scales now (the prompt route's
    /// load-time call); the verify route leaves them lazy.
    static func narrowKernel(
        _ cache: HadamardConstantLayoutCache, _ scales: MLXArray, _ biases: MLXArray,
        k: Int, n: Int, materialize: Bool
    ) -> NarrowKernel {
        let choice = narrowByShape[[k, n]] ?? narrowDefault
        if n % choice.variant.tn != 0 { return .original }
        guard choice.form != .base else { return choice }
        guard cache.biasesAreNegativeScales(scales, biases) else { return .original }
        if choice.form == .negativeBiasF32Scales {
            _ = narrowScalesF32(cache, scales, materialize: materialize)
        }
        return choice
    }

    /// The prompt route's load-time preparation of the verify operands: the
    /// proof and, when any chosen kernel reads them, the FP32 scales.
    static func prepareNarrowOperands(
        _ cache: HadamardConstantLayoutCache, _ scales: MLXArray, _ biases: MLXArray
    ) {
        guard narrowNeedsProof, cache.biasesAreNegativeScales(scales, biases) else { return }
        if narrowNeedsF32 { _ = narrowScalesF32(cache, scales, materialize: true) }
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
    /// `biasesT` is not read (callers pass `scalesT`). `factored` takes the
    /// factored epilogue (`factoredVerifyEpilogue`) in the negated-offset
    /// forms; `.base` always keeps the two-FMA form. Production and the
    /// load-time self-test, timing and pipeline builds all pass the switch.
    static func launchNarrowInt8(
        _ codes: MLXArray, _ weight: MLXArray, _ scalesT: MLXArray, _ biasesT: MLXArray,
        _ ascale: MLXArray, _ rowsum: MLXArray, k: Int, n: Int, outputDType: DType,
        kernel: NarrowKernel, factored: Bool
    ) -> MLXArray {
        let m = 16
        let inputs = [codes, weight, scalesT, biasesT, ascale, rowsum, dimsArray(k: k, m: m, n: n)]
        let template: [(String, any KernelTemplateArg)] = [
            ("OutT", outputDType), ("NEG", kernel.form == .base ? 0 : 1),
            ("F32S", kernel.form == .negativeBiasF32Scales ? 1 : 0),
            ("FACTORED", factored && kernel.form != .base ? 1 : 0),
        ]
        switch kernel.variant {
        case .v0:
            return kernelNarrowInt8(
                inputs, template: template,
                grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
        case let v where v.producerConsumer:
            return kernelNarrowInt8PC(
                inputs, template: template + [("RD", v.pd), ("KH", v.kh)],
                grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
        default:
            let v = kernel.variant
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
        let codes: MLXArray, weight: MLXArray
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
            eval(codes, weight, scalesT, biasesT, scalesT32, ascale, rowsum)
        }

        /// In the production epilogue mode (`factoredVerifyEpilogue`) unless
        /// `factored` says otherwise, so the timed and pre-built pipelines are
        /// the ones production launches.
        func run(
            _ kernel: NarrowKernel, _ outputDType: DType, factored: Bool = factoredVerifyEpilogue
        ) -> MLXArray {
            let (s, b): (MLXArray, MLXArray)
            switch kernel.form {
            case .base: (s, b) = (scalesT, biasesT)
            case .negativeBias: (s, b) = (scalesT, scalesT)
            case .negativeBiasF32Scales: (s, b) = (scalesT32, scalesT32)
            }
            return launchNarrowInt8(
                codes, weight, s, b, ascale, rowsum, k: k, n: n, outputDType: outputDType,
                kernel: kernel, factored: factored)
        }
    }

    /// The record's (dcaf489's) tuned shapes `(k, n)`: qkv|z, gate|up, down
    /// and attention qkv, 16 rows each. The record's pick is taken over these
    /// alone, as the record takes it.
    static let narrowRecordShapes = [(5120, 16384), (5120, 34816), (17408, 5120), (5120, 14336)]

    /// The verify window's production shapes `(k, n)`: the record's, then
    /// attention o_proj | GDN out_proj and the vocabulary head. The two added
    /// shapes are timed by the synthetic stage and tuned per shape, but only
    /// as input to the in-situ shortlist (the K5 set); they never change the
    /// record's pick.
    static let narrowTunedShapes = narrowRecordShapes + [(6144, 5120), (5120, 248320)]
    static let narrowTunedLabels = ["qkv|z", "gate|up", "down", "attn", "o|out", "head"]

    /// Whether a tuned shape's synthetic operands fit now. A shape whose
    /// weight set alone reaches 96 MB (the head: 318 MB, about four times
    /// that while generated) is timed on one copy, and only when the GPU
    /// working set has that plus 2 GB free; every other shape always fits.
    static func narrowTunedShapeFits(k: Int, n: Int) -> Bool {
        let bytes = k * n / 4
        guard bytes >= 96 << 20 else { return true }
        guard let working = GPU.maxRecommendedWorkingSetBytes() else { return false }
        let limit = min(working, Memory.memoryLimit)
        return limit - Memory.activeMemory - Memory.cacheMemory >= 4 * bytes + (2 << 30)
    }

    /// Chooses the verify int8 kernels once, at load, on the running GPU.
    ///
    /// Self-test: every candidate runs against `original` on synthetic
    /// operands (gate-, down- and two odd-quarter widths, so every ring's
    /// remainder guards run) and must match every output bit (FP16 for all;
    /// FP32 as well for each kernel that is then chosen); a mismatch or any
    /// MLX error (a body the toolchain cannot compile) drops it. With
    /// `MLXFAST_VERIFY_FACTORED=1` the reference is `v0` with the negated
    /// offset, factored like every candidate (like for like), after it stays
    /// within FP32 rounding of `original`. Candidates:
    /// every body (`v0`, each pipelined K3 variant and each K5 producer /
    /// consumer variant) in each epilogue form.
    /// Timing: the survivors and `original` run alternately on the six
    /// production shapes, each over distinct weight sets of >= 96 MB (so
    /// every launch streams its weights; the head is timed on one copy and
    /// only when it fits, see `narrowTunedShapeFits`), best of five trials.
    /// The record's pick: over the record's bodies (`narrowRecordVariants`)
    /// and shapes (`narrowRecordShapes`), each shape keeps its fastest kernel
    /// and every other shape takes the fastest in total. That is what is
    /// installed; the timing only shortlists the in-situ candidates
    /// (`NarrowInSituTrial`): the record's pick, `v0` with the negated
    /// offset, the two fastest K3 bodies in total (over the record's shapes)
    /// and the K5 set (per production shape, o|out and head included, the
    /// fastest K5 kernel; for every other shape the K5 kernel fastest in
    /// total relative to `original`), each kernel after its FP32 self-test,
    /// and every candidate kernel is then launched once per production shape
    /// with FP16 and FP32 outputs. A candidate not started within 4 s of the
    /// self-test is skipped (load time is not timed); the record's bodies
    /// come first in that order, then the K5 bodies. Runs at model init,
    /// before any timed phase, and builds every candidate's pipelines, so no
    /// verify round (nor any trial round) compiles anything.
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_EPILOGUE=off` keeps `original`
    /// (master kill switch); `neg` / `f32` force that epilogue.
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_PIPELINE=off` keeps the recorded
    /// body (`v0`); a comma-separated list of variant names (`pd1` .. `pd4`,
    /// `tn64`, `k64pd1` .. `k64pd4`, `pc2`, `k64pc2`, `k64pc4`) limits the
    /// pipelined candidates to those.
    private static func chooseNarrowKernels() -> (NarrowChoice, [NarrowChoice]) {
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
        // Default order = self-test order (what the deadline would cut last):
        // the record's bodies (so its pick is never cut), the K5 bodies, then
        // the K3 ones.
        let defaultVariants: [NarrowVariant] = [
            .pd1, .pd2, .tn64, .pc2, .k64pc2, .k64pc4,
            .k64pd1, .k64pd2, .pd3, .k64pd3, .pd4, .k64pd4,
        ]
        var variants = defaultVariants
        switch knob("DARKBLOOM_BONSAI_TENSOR_ROUTE_NARROW_PIPELINE") {
        case "off", "0", "false", "no", "v0": variants = []
        case .some(let value):
            let named = value.split(separator: ",").map {
                NarrowVariant(name: $0.trimmingCharacters(in: .whitespaces))
            }
            if !named.isEmpty, named.allSatisfy({ $0 != nil }) {
                variants = named.compactMap { $0 }.filter { $0 != .v0 }
            }
        case nil: break
        }
        let candidates = forms.map { NarrowKernel(variant: .v0, form: $0) }
            + variants.flatMap { v in forms.map { NarrowKernel(variant: v, form: $0) } }

        let start = DispatchTime.now().uptimeNanoseconds
        func elapsedMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 }
        // The self-test's reference: `original` (the recorded body and
        // epilogue), or with the factored epilogue on, `v0` with the negated
        // offset in that same factored form, so every body is compared with
        // the base body in the mode production runs it (`original` itself has
        // independent offsets and cannot take the factored form).
        let selfTestReference: NarrowKernel =
            factoredVerifyEpilogue ? NarrowKernel(variant: .v0, form: .negativeBias) : .original
        var log = "bonsai verify int8 kernels"
            + (factoredVerifyEpilogue ? " (factored epilogue, reference \(selfTestReference))" : "") + ":"
        var passed: [NarrowKernel] = []
        var failedF32 = Set<NarrowKernel>()
        var timings: [NarrowKernel: [Double]] = [:]
        var byShape: [[Int]: NarrowKernel] = [:]
        var fallback = NarrowKernel.original
        var trial: [NarrowChoice] = []
        var tuned = narrowTunedShapes
        do {
            try withError { error in
                let testShapes = [
                    (5120, 4096, UInt64(71)), (17408, 1024, UInt64(72)), (2560, 512, UInt64(73)),
                    (3584, 512, UInt64(74)),
                ]
                let testOps = testShapes.map { NarrowOperands(k: $0.0, n: $0.1, seed: $0.2) }
                // Bit for bit against `selfTestReference`, both in the production
                // epilogue mode (factored or not).
                func matches(_ kernel: NarrowKernel, _ outputDType: DType) throws -> Bool {
                    let bits: DType = outputDType == .float16 ? .uint16 : .uint32
                    var same = true
                    for ops in testOps {
                        let expected = ops.run(selfTestReference, outputDType)
                        let y = ops.run(kernel, outputDType)
                        let differ = (y.view(dtype: bits) .!= expected.view(dtype: bits))
                            .asType(.int32).sum()
                        eval(differ)
                        try error.check()
                        if differ.item(Int32.self) != 0 { same = false }
                    }
                    return same
                }
                // The factored reference has no bitwise twin; it must stay
                // within FP32 rounding of `original` (FP32 outputs, max error
                // <= 1e-4 of the largest output), so a broken factored body
                // cannot pass as its own reference. Failing keeps `original`.
                if factoredVerifyEpilogue {
                    for ops in testOps {
                        let exact = ops.run(.original, .float32, factored: false)
                        let y = ops.run(selfTestReference, .float32)
                        let worst = abs(y - exact).max()
                        let scale = abs(exact).max()
                        eval(worst, scale)
                        try error.check()
                        let w = worst.item(Float.self)
                        let limit = 1e-4 * scale.item(Float.self)
                        guard w <= limit else {
                            log += " factored reference off original by \(w) (limit \(limit));"
                                + " keeping original"
                            return
                        }
                    }
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
                tuned = narrowTunedShapes.filter { narrowTunedShapeFits(k: $0.0, n: $0.1) }
                let sets = tuned.enumerated().map { (index, shape) -> [NarrowOperands] in
                    let bytes = shape.0 * shape.1 / 4
                    let copies = bytes >= 96 << 20 ? 1 : min(6, max(2, (96 << 20) / bytes + 1))
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
                // Timing indices of the record's shapes (always timed: none
                // reaches 96 MB).
                let recordIndices = narrowRecordShapes.compactMap { shape in
                    tuned.firstIndex { $0 == shape }
                }
                func recordTotal(_ kernel: NarrowKernel) -> Double {
                    recordIndices.reduce(0) { $0 + timings[kernel]![$1] }
                }
                while true {
                    let usable = kernels.filter {
                        !failedF32.contains($0) && narrowRecordVariants.contains($0.variant)
                    }
                    func fastest(_ cost: (NarrowKernel) -> Double) -> NarrowKernel {
                        usable.min { cost($0) < cost($1) } ?? .original
                    }
                    fallback = fastest(recordTotal)
                    byShape = [:]
                    for index in recordIndices {
                        byShape[[tuned[index].0, tuned[index].1]] = fastest { timings[$0]![index] }
                    }
                    let picked = Set([fallback] + Array(byShape.values)).subtracting([.original])
                    var clean = true
                    for kernel in picked {
                        if try !exactF32(kernel) { clean = false }
                    }
                    if clean { break }
                }

                // The in-situ candidates: the record's pick, v0 with the
                // negated offset, the two fastest K3 bodies in total (over the
                // record's shapes, as before K5) and the K5 set, each kernel
                // exact at FP32 as well. Choices equal on every shape collapse.
                guard NarrowInSituTrial.enabled else { return }
                var shortlist: [NarrowChoice] = [(fallback, byShape)]
                let v0Negative = NarrowKernel(variant: .v0, form: .negativeBias)
                if passed.contains(v0Negative), try exactF32(v0Negative) {
                    shortlist.append((v0Negative, [:]))
                }
                let k3 = passed.filter {
                    !narrowRecordVariants.contains($0.variant) && !$0.variant.producerConsumer
                }.sorted { recordTotal($0) < recordTotal($1) }
                var survivors = 0
                for kernel in k3 where survivors < 2 {
                    if try exactF32(kernel) {
                        shortlist.append((kernel, [:]))
                        survivors += 1
                    }
                }
                // The K5 set: each timed production shape (o|out and the head
                // included) its fastest K5 kernel; every other shape the K5
                // kernel fastest in total relative to `original` per shape, so
                // the head's long launch does not decide it alone. A kernel
                // failing the FP32 self-test is dropped and the set redone.
                let reference = timings[.original]!
                while true {
                    let k5 = passed.filter { $0.variant.producerConsumer && !failedF32.contains($0) }
                    guard !k5.isEmpty else { break }
                    func fastest(_ cost: (NarrowKernel) -> Double) -> NarrowKernel {
                        k5.min { cost($0) < cost($1) }!
                    }
                    let k5Default = fastest { k in
                        zip(timings[k]!, reference).reduce(0) { $0 + $1.0 / $1.1 }
                    }
                    var k5ByShape: [[Int]: NarrowKernel] = [:]
                    for (index, shape) in tuned.enumerated() {
                        k5ByShape[[shape.0, shape.1]] = fastest { timings[$0]![index] }
                    }
                    var clean = true
                    for kernel in Set([k5Default] + Array(k5ByShape.values)) {
                        if try !exactF32(kernel) { clean = false }
                    }
                    if clean {
                        shortlist.append((k5Default, k5ByShape))
                        break
                    }
                }
                func effective(_ choice: NarrowChoice) -> [NarrowKernel] {
                    [choice.0] + narrowTunedShapes.map { choice.1[[$0.0, $0.1]] ?? choice.0 }
                }
                for choice in shortlist
                where !trial.contains(where: { effective($0) == effective(choice) }) {
                    trial.append(choice)
                }
                guard trial.count >= 2 else { trial = []; return }
                // Every candidate kernel once per production shape, FP16 and
                // FP32 outputs, so no pipeline compiles inside a trial round.
                let trialKernels = Set(trial.flatMap { [$0.0] + Array($0.1.values) })
                for kernel in trialKernels {
                    for outputDType in [DType.float16, .float32] {
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
            log += "; us/launch per shape ("
                + tuned.map { shape in
                    narrowTunedShapes.firstIndex { $0 == shape }.map { narrowTunedLabels[$0] } ?? "?"
                }.joined(separator: " ") + "):"
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
                " " + NarrowInSituTrial.describe($0)
            }.joined() + "]"
        }
        log += "; \(String(format: "%.0f", elapsedMs())) ms\n"
        FileHandle.standardError.write(log.data(using: .utf8)!)
        return ((fallback, byShape), trial)
    }

    /// Installs the record's pick and sets up the in-situ trial. The operands
    /// every candidate reads (the per-projection proof, the FP32 scales) are
    /// prepared at the load-time prompt forward, before any trial round.
    private static func installNarrowChoice() {
        let (choice, trial) = chooseNarrowKernels()
        narrowDefault = choice.0
        narrowByShape = choice.1
        NarrowInSituTrial.sets = trial
        setNarrowOperandNeeds([choice] + trial)
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
            HadamardQuantizedLinear.narrowProducerApproves = Qwen35VerifyProducerQ8.approves
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
                let choice = narrowKernel(cache, scales, biases, k: k, n: n, materialize: false)
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
                // The capture verify's head (`Qwen35HeadTopTwo.capture`): the
                // fused top two of the same launch, beside its lazy logits.
                if Qwen35HeadTopTwo.capturing, outputDType == .float32,
                    headTop2Applies(k: k, n: n, kernel: choice)
                {
                    Qwen35HeadTopTwo.captured = launchNarrowInt8Top2(
                        codes, weight, scalesT, biasesT, activation.scales,
                        activation.scaledSums, k: k, n: n, kernel: choice,
                        factored: factoredVerifyEpilogue)
                    Qwen35HeadTopTwo.announce("\(choice), k \(k), n \(n)")
                }
                return launchNarrowInt8(
                    codes, weight, scalesT, biasesT, activation.scales, activation.scaledSums,
                    k: k, n: n, outputDType: outputDType, kernel: choice,
                    factored: factoredVerifyEpilogue)
            }
            installNarrowChoice()
            prepareHeadTop2()
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
            prepareNarrowOperands(cache, scales, biases)
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
                template.append(("NEGATIVE_SCALE_BIAS",
                    cache.biasesAreNegativeScales(scales, biases) ? 1 : 0))
                template.append(("FACTORED", factoredPromptEpilogue ? 1 : 0))
                // The tile the in-situ trial installed (`PromptInSituTrial`);
                // the record's `kernelStaged8` launch where none is or the
                // tile does not divide [m, n].
                return launchStaged8(
                    [codes, weight, scalesT, biasesT, foldedSums, activation.scales,
                     activation.scaledSums, dimsArray(k: k, m: m, n: n)],
                    template: template, m: m, n: n, outputDType: outputDType,
                    tile: promptTile(m: m, n: n))
            default: packedKernel = kernelStaged
            }
            return packedKernel(
                [codes, weight, scalesT, biasesT, foldedSums, activation.scales,
                 activation.scaledSums, dimsArray(k: k, m: m, n: n)],
                template: template,
                grid: (n / 64 * 128, m / 64, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
        }
    }
}

// MARK: - The verify head's top two, fused into the int8 head launch

/// The capture verify's vocabulary head never stores its FP32 logits
/// (`MLXFAST_HEAD_TOP2=0` stores them).
///
/// On the int8 verify route the head's launch (`launchNarrowInt8`, n = 248320)
/// stored `[16, 248320]` FP32 logits, 15.9 MB a round, and the policy top two
/// (`qwen35MTPTopTwoRows`) read them all back; the acceptance packet takes
/// only each row's top-1 id and top-two values. Here the head launch keeps,
/// per threadgroup and row, the top two (value, column) of its columns after
/// its K-split reduction, in place of the logit stores
/// (`Qwen35TensorPackedMatmul.headTop2Source`, `[16, blocks, 2]`), and one
/// simdgroup per row merges the blocks (`headTop2MergeKernel`). The order is
/// `cbv2TopTwoRows`'s: value descending, the lower id on exact ties (signed
/// zeros tie), NaN last; the top two under that total order does not depend
/// on how the candidates are grouped, and each candidate is the value the
/// stock launch would have stored, so ids and values are the same bits.
///
/// Routing: only the capture verify's head call (`capture`), only where the
/// int8 route takes the 16 rows with a kernel whose fused form passed the
/// load-time self-test. The FP32 logits stay in the graph as the lazy stock
/// launch: nothing on the greedy round evaluates them, and any caller that
/// asks gets them as before. `cbv2MTPTopTwo` takes the fused pair only for
/// the logits array this capture returned (`lookup`, by identity).
enum Qwen35HeadTopTwo {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_HEAD_TOP2"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Set while the capture verify builds its head call; the int8 head
    /// launch then leaves its fused pair in `captured`.
    nonisolated(unsafe) static var capturing = false
    nonisolated(unsafe) static var captured: (ids: MLXArray, values: MLXArray)?

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

    /// One line at the first head launch that takes the fused form (the
    /// load-time verify warm).
    static func announce(_ what: String) {
        guard !announced else { return }
        announced = true
        FileHandle.standardError.write("bonsai head top-2: in use (\(what))\n".data(using: .utf8)!)
    }

    /// `head()` (the capture verify's `lmHead(normalized)`), remembering the
    /// fused top two of the logits it returns when the int8 head launch
    /// produced one.
    static func capture(_ head: () -> MLXArray) -> MLXArray {
        guard enabled else { return head() }
        captured = nil
        capturing = true
        let logits = head()
        capturing = false
        if let pair = captured {
            last = Entry(logits: logits, ids: pair.ids, values: pair.values)
        } else {
            last = nil
        }
        captured = nil
        return logits
    }

    /// The fused `[rows, 2]` ids and values of `logits` when they are the
    /// array the last `capture` returned (`[B, L, V]`, `B * L <= 16` rows).
    static func lookup(_ logits: MLXArray, rows: Int) -> (ids: MLXArray, values: MLXArray)? {
        guard enabled, let entry = last, let captured = entry.logits, captured === logits,
            rows >= 1, rows <= entry.ids.dim(0)
        else { return nil }
        if rows == entry.ids.dim(0) { return (entry.ids, entry.values) }
        return (entry.ids[0 ..< rows], entry.values[0 ..< rows])
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
    static func headTop2Source(_ text: String, columns: String) -> String? {
        let open = "if (sg == 0) {"
        guard text.components(separatedBy: open).count == 2, !text.contains("ht2"),
            let regex = try? NSRegularExpression(pattern: headTop2StorePattern)
        else { return nil }
        var t = text.replacingOccurrences(
            of: open,
            with: "bonsai_head_top2 ht2[2] = {bonsai_head_top2_empty(), bonsai_head_top2_empty()};\n        "
                + open)
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
    private static let kernelNarrowInt8PCTop2 = headTop2Kernel(
        "bonsai_tensor_packed_matmul_m16_i8pc_top2", sourceNarrowInt8PC, columns: "32")

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
    static func launchNarrowInt8Top2(
        _ codes: MLXArray, _ weight: MLXArray, _ scalesT: MLXArray, _ biasesT: MLXArray,
        _ ascale: MLXArray, _ rowsum: MLXArray, k: Int, n: Int,
        kernel: NarrowKernel, factored: Bool
    ) -> (ids: MLXArray, values: MLXArray)? {
        let m = 16
        let inputs = [codes, weight, scalesT, biasesT, ascale, rowsum, dimsArray(k: k, m: m, n: n)]
        let template: [(String, any KernelTemplateArg)] = [
            ("OutT", DType.float32), ("NEG", kernel.form == .base ? 0 : 1),
            ("F32S", kernel.form == .negativeBiasF32Scales ? 1 : 0),
            ("FACTORED", factored && kernel.form != .base ? 1 : 0),
        ]
        let v = kernel.variant
        let columns = v == .v0 || v.producerConsumer ? 32 : v.tn
        guard n % columns == 0 else { return nil }
        let blocks = n / columns
        let shapes = [[m, blocks, 2], [m, blocks, 2]]
        let dtypes: [DType] = [.int32, .float32]
        let partial: [MLXArray]
        switch v {
        case .v0:
            guard let launch = kernelNarrowInt8Top2 else { return nil }
            partial = launch(
                inputs, template: template,
                grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: shapes, outputDTypes: dtypes)
        case let v where v.producerConsumer:
            guard let launch = kernelNarrowInt8PCTop2 else { return nil }
            partial = launch(
                inputs, template: template + [("RD", v.pd), ("KH", v.kh)],
                grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: shapes, outputDTypes: dtypes)
        default:
            guard let launch = kernelNarrowInt8PipelinedTop2 else { return nil }
            partial = launch(
                inputs, template: template + [("PD", v.pd), ("TN", v.tn), ("KH", v.kh)],
                grid: (n / v.tn * 128, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: shapes, outputDTypes: dtypes)
        }
        let merged = headTop2MergeKernel(
            partial, grid: (32, m, 1), threadGroup: (32, 1, 1),
            outputShapes: [[m, 2], [m, 2]], outputDTypes: [.int32, .float32])
        return (merged[0], merged[1])
    }

    /// The head shape `(k, n)` the self-test ran on and the kernels whose
    /// fused form passed it there.
    nonisolated(unsafe) private static var headTop2Shape: [Int] = []
    nonisolated(unsafe) private static var headTop2Verified: Set<NarrowKernel> = []

    /// Whether the int8 head launch of `kernel` at `[k, n]` may take its
    /// fused form.
    static func headTop2Applies(k: Int, n: Int, kernel: NarrowKernel) -> Bool {
        Qwen35HeadTopTwo.enabled && headTop2Shape == [k, n] && headTop2Verified.contains(kernel)
    }

    /// The load-time self-test of the fused head (after the verify kernel
    /// choice, so every kernel the head can take is known: `original`, the
    /// record's pick and each in-situ candidate's). On the head shape, per
    /// kernel: 16 random rows over random weights; every row with exact ties
    /// at its maximum (each column's weights and scales repeated every 24
    /// columns, so ties fall inside a block and across blocks); and repeats
    /// every 4099 columns with infinite and NaN scales in some columns (NaN
    /// and infinite logits). The fused ids and values are compared with
    /// `qwen35MTPTopTwoRows` over the stock FP32 launch, as unsigned
    /// integers; a kernel with any mismatch or MLX error keeps the
    /// two-kernel path. Builds the fused pipelines on the way.
    static func prepareHeadTop2() {
        guard Qwen35HeadTopTwo.enabled,
            let index = narrowTunedLabels.firstIndex(of: "head")
        else { return }
        let (k, n) = narrowTunedShapes[index]
        let start = DispatchTime.now().uptimeNanoseconds
        var kernels: [NarrowKernel] = [.original]
        for choice in [(narrowDefault, narrowByShape)] + NarrowInSituTrial.sets {
            let pick = choice.1[[k, n]] ?? choice.0
            let effective = n % pick.variant.tn == 0 ? pick : .original
            if !kernels.contains(effective) { kernels.append(effective) }
        }
        var log = "bonsai head top-2: "
        guard narrowTunedShapeFits(k: k, n: n) else {
            log += "self-test skipped (the head operands do not fit); two-kernel path kept\n"
            FileHandle.standardError.write(log.data(using: .utf8)!)
            return
        }
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
            let operands = NarrowOperands(
                k: k, n: n, codes: base.codes, weight: base.weight[columns],
                scalesT: s.contiguous(), biasesT: b.contiguous(), scalesT32: s32.contiguous(),
                ascale: base.ascale, rowsum: base.rowsum)
            eval(operands.weight, operands.scalesT, operands.biasesT, operands.scalesT32)
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
        log += (passed.count == kernels.count ? "self-test passed" : "self-test FAILED for some kernels")
            + " [" + kernels.map { "\($0)" + (passed.contains($0) ? "" : "!") }.joined(separator: " ")
            + "] (\(cases.count) cases x 16 rows on \(k)x\(n): random, ties every 24 columns, "
            + "ties every 4099 with inf/NaN scales; \(values) ids+values compared bitwise, "
            + "\(total) mismatches"
            + (failed.isEmpty ? "" : "; errors " + failed.map { "\($0.key): \($0.value)" }
                .joined(separator: ", "))
            + "); "
            + (passed.isEmpty ? "two-kernel path kept" : "fused head top-2 for the passed kernels")
            + String(
                format: "; %.0f ms\n", Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        FileHandle.standardError.write(log.data(using: .utf8)!)
    }
}

private enum MLXFastHeadTop2Failure: Error {
    case message(String)
}

extension Qwen35TensorPackedMatmul.NarrowOperands {
    init(
        k: Int, n: Int, codes: MLXArray, weight: MLXArray, scalesT: MLXArray,
        biasesT: MLXArray, scalesT32: MLXArray, ascale: MLXArray, rowsum: MLXArray
    ) {
        self.k = k
        self.n = n
        self.codes = codes
        self.weight = weight
        self.scalesT = scalesT
        self.biasesT = biasesT
        self.scalesT32 = scalesT32
        self.ascale = ascale
        self.rowsum = rowsum
    }

    /// `run` in its fused head form (FP32 values, production epilogue mode).
    func runTop2(
        _ kernel: Qwen35TensorPackedMatmul.NarrowKernel,
        factored: Bool = Qwen35TensorPackedMatmul.factoredVerifyEpilogue
    ) -> (ids: MLXArray, values: MLXArray)? {
        let (s, b): (MLXArray, MLXArray)
        switch kernel.form {
        case .base: (s, b) = (scalesT, biasesT)
        case .negativeBias: (s, b) = (scalesT, scalesT)
        case .negativeBiasF32Scales: (s, b) = (scalesT32, scalesT32)
        }
        return Qwen35TensorPackedMatmul.launchNarrowInt8Top2(
            codes, weight, s, b, ascale, rowsum, k: k, n: n, kernel: kernel, factored: factored)
    }
}
