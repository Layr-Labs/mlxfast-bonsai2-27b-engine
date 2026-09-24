// Copyright © 2026 Apple Inc.

#ifndef MLX_GEMMA4_EXPERT_QMM_H
#define MLX_GEMMA4_EXPERT_QMM_H

#include <stddef.h>
#include <stdint.h>

#if defined(__APPLE__)
#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_metal_gemma4_expert_qmm_diagnostics {
  uint8_t requested;
  uint8_t aot_available;
  uint8_t nax_available;
  uint8_t armed;
  uint64_t attempts;
  uint64_t hits;
  uint64_t fallback_nax;
  uint64_t fallback_outer_route;
  uint64_t fallback_quantization;
  uint64_t fallback_topology;
  uint64_t fallback_assignment_count;
  uint64_t fallback_geometry;
  uint64_t fallback_metallib_unavailable;
  uint64_t fallback_sortedness_retracted;
} mlx_metal_gemma4_expert_qmm_diagnostics;

void mlx_metal_gemma4_expert_qmm_diagnostics_snapshot(
    mlx_metal_gemma4_expert_qmm_diagnostics* diagnostics);
void mlx_metal_gemma4_expert_qmm_diagnostics_reset(void);
void mlx_metal_gemma4_expert_qmm_diagnostics_clear_and_arm(void);
void mlx_metal_gemma4_expert_qmm_diagnostics_snapshot_and_disarm(
    mlx_metal_gemma4_expert_qmm_diagnostics* diagnostics);

#ifdef __cplusplus
}
#endif

// ABI drift pins. The layout is 4 x uint8 at offsets 0-3, 4 bytes of
// alignment padding, then 8-aligned uint64 counters; the Swift
// GPU.Gemma4ExpertQMMDiagnostics mapping and the Cmlx C++ facade mirror both
// depend on these exact values.
_Static_assert(
    sizeof(mlx_metal_gemma4_expert_qmm_diagnostics) == 88,
    "mlx_metal_gemma4_expert_qmm_diagnostics ABI drift: expected 88 bytes");
_Static_assert(
    offsetof(mlx_metal_gemma4_expert_qmm_diagnostics, armed) == 3,
    "mlx_metal_gemma4_expert_qmm_diagnostics.armed offset drift: expected 3");
_Static_assert(
    offsetof(mlx_metal_gemma4_expert_qmm_diagnostics, attempts) == 8,
    "mlx_metal_gemma4_expert_qmm_diagnostics.attempts offset drift: expected 8");
_Static_assert(
    offsetof(mlx_metal_gemma4_expert_qmm_diagnostics, hits) == 16,
    "mlx_metal_gemma4_expert_qmm_diagnostics.hits offset drift: expected 16");
_Static_assert(
    offsetof(
        mlx_metal_gemma4_expert_qmm_diagnostics,
        fallback_metallib_unavailable) == 72,
    "mlx_metal_gemma4_expert_qmm_diagnostics.fallback_metallib_unavailable "
    "offset drift: expected 72");
_Static_assert(
    offsetof(
        mlx_metal_gemma4_expert_qmm_diagnostics,
        fallback_sortedness_retracted) == 80,
    "mlx_metal_gemma4_expert_qmm_diagnostics.fallback_sortedness_retracted "
    "offset drift: expected 80");

#endif

#endif
