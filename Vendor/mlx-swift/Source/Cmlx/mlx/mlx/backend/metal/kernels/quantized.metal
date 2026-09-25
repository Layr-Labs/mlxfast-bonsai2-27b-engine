// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define instantiate_quantized(name, type, group_size, bits)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits,                    \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits)

#define instantiate_quantized_batched(name, type, group_size, bits, batched)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits,                                                              \
      batched)

#define instantiate_quantized_aligned(name, type, group_size, bits, aligned)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, aligned, batched)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned,                                                               \
      batched)

#define instantiate_quantized_quad(name, type, group_size, bits, D, batched)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      D,                                                            \
      batched)

#define instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, batched)               \
  instantiate_kernel(                                                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched,   \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      vecs_per_tg,                                                  \
      k_lanes,                                                      \
      batched)

#define instantiate_quantized_split_k(name, type, group_size, bits, split_k)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      split_k)

#define instantiate_gather_qmm_rhs(func, name, type, group_size, bits, bm, bn, bk, wm, wn, transpose)        \
  instantiate_kernel(                                                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      bm,                                                           \
      bn,                                                           \
      bk,                                                           \
      wm,                                                           \
      wn,                                                           \
      transpose)

#define instantiate_quantized_batched_wrap(name, type, group_size, bits) \
  instantiate_quantized_batched(name, type, group_size, bits, 1)      \
  instantiate_quantized_batched(name, type, group_size, bits, 0)

#define instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_batched_wrap(affine_qmv_fast, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qmv, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qvm, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_single(type, group_size, bits) \
  instantiate_quantized(affine_quantize, type, group_size, bits) \
  instantiate_quantized(affine_dequantize, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmv_fast, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmv, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qvm, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_aligned(type, group_size, bits)   \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, true) \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, false) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 0) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 0)

#define instantiate_quantized_all_quad(type, group_size, bits)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 1)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 0)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 1)  \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 0)

// vecs_per_tg (input-vector tile) 2..5; affine uses k_lanes=8 (more rows per
// simdgroup) where the fp path uses 16.
#define instantiate_quantized_wide_wrap(name, type, group_size, bits, vecs_per_tg, k_lanes) \
  instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 0)         \
  instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 1)

#define instantiate_quantized_all_wide(type, group_size, bits) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 2, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 3, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 4, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 5, 8)

#define instantiate_quantized_all_splitk(type, group_size, bits)   \
  instantiate_quantized_split_k(affine_qvm_split_k, type, group_size, bits, 8)   \
  instantiate_quantized_split_k(affine_qvm_split_k, type, group_size, bits, 32)  \

#define instantiate_quantized_splitk_qmm(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                           \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,         \
      name,                                                                     \
      type,                                                                     \
      group_size,                                                               \
      bits,                                                                     \
      aligned)

#define instantiate_quantized_all_splitk_qmm(type, group_size, bits)                    \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, true)  \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, false)

#define instantiate_quantized_all_rhs(type, group_size, bits) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nt, type, group_size, bits, 16, 32, 32, 1, 2, true) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nn, type, group_size, bits, 16, 32, 32, 1, 2, false)

#define instantiate_quantized_funcs(type, group_size, bits) \
  instantiate_quantized_all_single(type, group_size, bits)  \
  instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_all_aligned(type, group_size, bits) \
  instantiate_quantized_all_quad(type, group_size, bits)    \
  instantiate_quantized_all_wide(type, group_size, bits)    \
  instantiate_quantized_all_splitk(type, group_size, bits)  \
  instantiate_quantized_all_splitk_qmm(type, group_size, bits) \
  instantiate_quantized_all_rhs(type, group_size, bits)

#define instantiate_quantized_types(group_size, bits)       \
  instantiate_quantized_funcs(float, group_size, bits)      \
  instantiate_quantized_funcs(float16_t, group_size, bits)  \
  instantiate_quantized_funcs(bfloat16_t, group_size, bits)

#define instantiate_quantized_groups(bits) \
  instantiate_quantized_types(128, bits)   \
  instantiate_quantized_types(64, bits)    \
  instantiate_quantized_types(32, bits)

#define instantiate_quantized_all() \
  instantiate_quantized_groups(2) \
  instantiate_quantized_groups(3) \
  instantiate_quantized_groups(4) \
  instantiate_quantized_groups(5) \
  instantiate_quantized_groups(6) \
  instantiate_quantized_groups(8)

instantiate_quantized_all()

// AOT 4-row NSG=1 qmv_fast (32 threads/TG). Same body as JIT _nsg1; host
// loads these from mlx.metallib so decode does not JIT a second library.
#define instantiate_qmv_fast_nsg1(type, group_size, bits)                      \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_nsg1_batch_0", \
      affine_qmv_fast,                                                         \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      false,                                                                   \
      false,                                                                   \
      4,                                                                       \
      false,                                                                   \
      false,                                                                   \
      0,                                                                       \
      1)                                                                       \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_nsg1_batch_1", \
      affine_qmv_fast,                                                         \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      true,                                                                    \
      false,                                                                   \
      4,                                                                       \
      false,                                                                   \
      false,                                                                   \
      0,                                                                       \
      1)

instantiate_qmv_fast_nsg1(float, 128, 2)
instantiate_qmv_fast_nsg1(float16_t, 128, 2)
instantiate_qmv_fast_nsg1(bfloat16_t, 128, 2)

// AOT 4-row NSG=4 qmv_fast (128 threads/TG). Same body as JIT _nsg4.
#define instantiate_qmv_fast_nsg4(type, group_size, bits)                      \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_nsg4_batch_0", \
      affine_qmv_fast,                                                         \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      false,                                                                   \
      false,                                                                   \
      4,                                                                       \
      false,                                                                   \
      false,                                                                   \
      0,                                                                       \
      4)                                                                       \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_nsg4_batch_1", \
      affine_qmv_fast,                                                         \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      true,                                                                    \
      false,                                                                   \
      4,                                                                       \
      false,                                                                   \
      false,                                                                   \
      0,                                                                       \
      4)

instantiate_qmv_fast_nsg4(float, 128, 2)
instantiate_qmv_fast_nsg4(float16_t, 128, 2)
instantiate_qmv_fast_nsg4(bfloat16_t, 128, 2)

// AOT 8-row nsg=2 qmv_fast (64 threads/TG, 16 rows). Same body as JIT _r_8.
#define instantiate_qmv_fast_rps8(type, group_size, bits)                      \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_r_8_batch_0", \
      affine_qmv_fast,                                                         \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      false,                                                                   \
      false,                                                                   \
      8)                                                                       \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_r_8_batch_1", \
      affine_qmv_fast,                                                         \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      true,                                                                    \
      false,                                                                   \
      8)

instantiate_qmv_fast_rps8(float, 128, 2)
instantiate_qmv_fast_rps8(float16_t, 128, 2)
instantiate_qmv_fast_rps8(bfloat16_t, 128, 2)

// AOT 2-row nsg=2 qmv_fast (64 threads/TG, 4 rows). bn=4 divides this
// pack's N. Distinct from NSG=1 RPS=2 (PB closed D 0.92) and RPS=8.
#define instantiate_qmv_fast_rps2(type, group_size, bits)                      \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_r_2_batch_0", \
      affine_qmv_fast,                                                         \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      false,                                                                   \
      false,                                                                   \
      2)                                                                       \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_r_2_batch_1", \
      affine_qmv_fast,                                                         \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      true,                                                                    \
      false,                                                                   \
      2)

instantiate_qmv_fast_rps2(float, 128, 2)
instantiate_qmv_fast_rps2(float16_t, 128, 2)
instantiate_qmv_fast_rps2(bfloat16_t, 128, 2)

// AOT 64-thread occupancy-hint qmv_fast (stock nsg=2, 4 rows). Same body
// as JIT _tg64; host loads these from mlx.metallib.
#define instantiate_qmv_fast_tg64(type, group_size, bits)                      \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_tg64_batch_0", \
      affine_qmv_fast_tg64,                                                    \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      false,                                                                   \
      false,                                                                   \
      4)                                                                       \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_tg64_batch_1", \
      affine_qmv_fast_tg64,                                                    \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      true,                                                                    \
      false,                                                                   \
      4)

instantiate_qmv_fast_tg64(float, 128, 2)
instantiate_qmv_fast_tg64(float16_t, 128, 2)
instantiate_qmv_fast_tg64(bfloat16_t, 128, 2)

// AOT 8-tile thread-private x-reuse qmv_fast (no smem). Host loads these
// from mlx.metallib so decode does not JIT a second library.
#define instantiate_qmv_fast_xr8(type, group_size, bits)                       \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_xr8_batch_0", \
      affine_qmv_fast_xr8,                                                     \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      false)                                                                   \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_xr8_batch_1", \
      affine_qmv_fast_xr8,                                                     \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      true)

instantiate_qmv_fast_xr8(float, 128, 2)
instantiate_qmv_fast_xr8(float16_t, 128, 2)
instantiate_qmv_fast_xr8(bfloat16_t, 128, 2)

#define instantiate_qmv_fast_xr2(type, group_size, bits)                       \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_xr2_batch_0", \
      affine_qmv_fast_xr2,                                                     \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      false)                                                                   \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_xr2_batch_1", \
      affine_qmv_fast_xr2,                                                     \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      true)

instantiate_qmv_fast_xr2(float, 128, 2)
instantiate_qmv_fast_xr2(float16_t, 128, 2)
instantiate_qmv_fast_xr2(bfloat16_t, 128, 2)

#define instantiate_qmv_fast_xr4(type, group_size, bits)                       \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_xr4_batch_0", \
      affine_qmv_fast_xr4,                                                     \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      false)                                                                   \
  instantiate_kernel(                                                          \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits "_xr4_batch_1", \
      affine_qmv_fast_xr4,                                                     \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      true)

instantiate_qmv_fast_xr4(float, 128, 2)
instantiate_qmv_fast_xr4(float16_t, 128, 2)
instantiate_qmv_fast_xr4(bfloat16_t, 128, 2)

instantiate_kernel(
    "affine_gather_qmm_gemma4_expert_tiles_bfloat16_t_gs_64_b_4_alN_true_bm_32_bn_32_bk_32",
    affine_gather_qmm_gemma4_expert_tiles,
    bfloat16_t,
    64,
    4,
    true,
    32,
    32,
    32)

// Sorted expert-tile descriptor builders. The E=128 instantiation keeps the
// historical Gemma 4 host name; E=256 serves Qwen 3.5/3.6 MoE. The tile
// kernel instantiation above is expert-count agnostic (K/N are runtime
// arguments) and is shared by both routes.
instantiate_kernel(
    "build_gemma4_sorted_expert_tiles_bm32",
    build_sorted_expert_tiles_bm32,
    128)

instantiate_kernel(
    "build_sorted_expert_tiles_bm32_e256",
    build_sorted_expert_tiles_bm32,
    256)

    // clang-format on
