// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <stdint.h>

#include <Cmlx/mlx-api.h>

// `mlx-api.h` only defines MLX_API under `__cplusplus`; the extern-C block
// below is also parsed in C mode (Swift / Objective-C consumers of the Cmlx
// Clang module), where the macro would otherwise be an unknown type name.
#ifndef MLX_API
#define MLX_API
#endif

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

MLX_API void mlx_metal_gemma4_expert_qmm_diagnostics_snapshot(
    mlx_metal_gemma4_expert_qmm_diagnostics* diagnostics);
MLX_API void mlx_metal_gemma4_expert_qmm_diagnostics_reset(void);
MLX_API void mlx_metal_gemma4_expert_qmm_diagnostics_clear_and_arm(void);
MLX_API void mlx_metal_gemma4_expert_qmm_diagnostics_snapshot_and_disarm(
    mlx_metal_gemma4_expert_qmm_diagnostics* diagnostics);

#ifdef __cplusplus
}
#endif
#endif

#ifdef __cplusplus

#include <atomic>

namespace mlx::core::metal {

enum class Gemma4ExpertQMMRoute : uint8_t {
  not_requested,
  hit,
  fallback_nax,
  fallback_outer_route,
  fallback_quantization,
  fallback_topology,
  fallback_assignment_count,
  fallback_geometry,
  fallback_metallib_unavailable,
  fallback_sortedness_retracted,
};

struct Gemma4ExpertQMMRouteInput {
  bool requested{false};
  bool aot_available{false};
  bool nax_available{false};
  bool outer_route{false};

  bool affine{false};
  bool transpose{false};
  bool has_bias{false};
  bool indices_uint32{false};
  bool indices_contiguous{false};
  bool x_bfloat16{false};
  bool x_contiguous{false};
  bool w_uint32{false};
  bool w_contiguous{false};
  bool scales_bfloat16{false};
  bool scales_contiguous{false};
  bool biases_bfloat16{false};
  bool biases_contiguous{false};

  int group_size{0};
  int bits{0};
  int expert_count{0};
  int assignments{0};
  int index_count{0};
  int k{0};
  int n{0};

  int x_rank{0};
  int x_dim0{0};
  int x_dim1{0};
  int x_dim2{0};
  int w_rank{0};
  int w_dim0{0};
  int w_dim1{0};
  int w_dim2{0};
  int scales_rank{0};
  int scales_dim0{0};
  int scales_dim1{0};
  int scales_dim2{0};
  int biases_rank{0};
  int biases_dim0{0};
  int biases_dim1{0};
  int biases_dim2{0};
};

inline Gemma4ExpertQMMRoute classify_gemma4_expert_qmm(
    const Gemma4ExpertQMMRouteInput& input) {
  if (!input.requested) {
    return Gemma4ExpertQMMRoute::not_requested;
  }
  if (!input.outer_route) {
    return Gemma4ExpertQMMRoute::fallback_outer_route;
  }
  // The existing NAX route owns every supported BF16/transposed RHS call and
  // must win before the Gemma 4 specialization or its AOT capability matters.
  if (input.nax_available) {
    return Gemma4ExpertQMMRoute::fallback_nax;
  }
  if (!input.affine || !input.transpose || !input.has_bias ||
      !input.indices_uint32 || !input.indices_contiguous ||
      !input.x_bfloat16 || !input.x_contiguous || !input.w_uint32 ||
      !input.w_contiguous || !input.scales_bfloat16 ||
      !input.scales_contiguous || !input.biases_bfloat16 ||
      !input.biases_contiguous || input.group_size != 64 || input.bits != 4) {
    return Gemma4ExpertQMMRoute::fallback_quantization;
  }
  const bool gemma4 = input.expert_count == 128;
  const bool qwen36 = input.expert_count == 256;
  if ((!gemma4 && !qwen36) || input.x_rank != 3 ||
      input.x_dim0 != input.assignments || input.x_dim1 != 1 ||
      input.x_dim2 != input.k || input.w_rank != 3 ||
      input.w_dim0 != input.expert_count || input.scales_rank != 3 ||
      input.scales_dim0 != input.expert_count || input.biases_rank != 3 ||
      input.biases_dim0 != input.expert_count ||
      input.index_count != input.assignments) {
    return Gemma4ExpertQMMRoute::fallback_topology;
  }
  if (input.assignments != 4096 && input.assignments != 8192 &&
      input.assignments != 16384) {
    return Gemma4ExpertQMMRoute::fallback_assignment_count;
  }

  // Whole-projection geometry for one expert matrix [E, n, k] at W4/g64:
  // packed weight columns k/8 (eight 4-bit values per uint32) and one
  // scale/bias per 64-wide group, k/64 columns. The quantization gate above
  // guarantees bits==4 and group_size==64, so the divisions are exact.
  auto projection = [&input](int k, int n) {
    return input.k == k && input.n == n && input.w_dim1 == n &&
        input.w_dim2 == k / 8 && input.scales_dim1 == n &&
        input.scales_dim2 == k / 64 && input.biases_dim1 == n &&
        input.biases_dim2 == k / 64;
  };
  // Gemma 4 26B-A4B (E=128): gate/up [128,1408,2816] and down [128,2816,704].
  // Qwen 3.5/3.6 35B-A3B (E=256): fused gate_up [256,1024,2048], split
  // gate/up [256,512,2048], and down [256,2048,512]. The tile kernel itself
  // is expert-count agnostic; only the descriptor builder instantiation
  // differs (one thread per expert).
  const bool hit_geometry = gemma4
      ? (projection(2816, 1408) || projection(704, 2816))
      : (projection(2048, 1024) || projection(2048, 512) ||
         projection(512, 2048));
  if (!hit_geometry) {
    return Gemma4ExpertQMMRoute::fallback_geometry;
  }
  if (!input.aot_available) {
    return Gemma4ExpertQMMRoute::fallback_metallib_unavailable;
  }
  return Gemma4ExpertQMMRoute::hit;
}

struct Gemma4ExpertQMMCounterSnapshot {
  uint64_t hits{0};
  uint64_t fallback_nax{0};
  uint64_t fallback_outer_route{0};
  uint64_t fallback_quantization{0};
  uint64_t fallback_topology{0};
  uint64_t fallback_assignment_count{0};
  uint64_t fallback_geometry{0};
  uint64_t fallback_metallib_unavailable{0};
  uint64_t fallback_sortedness_retracted{0};
  bool armed{false};

  uint64_t attempts() const {
    return hits + fallback_nax + fallback_outer_route +
        fallback_quantization + fallback_topology +
        fallback_assignment_count + fallback_geometry +
        fallback_metallib_unavailable + fallback_sortedness_retracted;
  }
};

class Gemma4ExpertQMMCounters {
 public:
  bool armed() const {
    return armed_.load(std::memory_order_relaxed);
  }

  // Recording is called only after the caller's relaxed-atomic armed branch.
  // Keeping the branch at that boundary makes the unarmed inference path free
  // of counter atomic operations while the engine-idle arm/disarm contract
  // makes access to armed_ well-defined.
  void record(Gemma4ExpertQMMRoute route) {
    std::atomic<uint64_t>* counter = nullptr;
    switch (route) {
      case Gemma4ExpertQMMRoute::not_requested:
        return;
      case Gemma4ExpertQMMRoute::hit:
        counter = &hits_;
        break;
      case Gemma4ExpertQMMRoute::fallback_nax:
        counter = &fallback_nax_;
        break;
      case Gemma4ExpertQMMRoute::fallback_outer_route:
        counter = &fallback_outer_route_;
        break;
      case Gemma4ExpertQMMRoute::fallback_quantization:
        counter = &fallback_quantization_;
        break;
      case Gemma4ExpertQMMRoute::fallback_topology:
        counter = &fallback_topology_;
        break;
      case Gemma4ExpertQMMRoute::fallback_assignment_count:
        counter = &fallback_assignment_count_;
        break;
      case Gemma4ExpertQMMRoute::fallback_geometry:
        counter = &fallback_geometry_;
        break;
      case Gemma4ExpertQMMRoute::fallback_metallib_unavailable:
        counter = &fallback_metallib_unavailable_;
        break;
      case Gemma4ExpertQMMRoute::fallback_sortedness_retracted:
        counter = &fallback_sortedness_retracted_;
        break;
    }
    counter->fetch_add(1, std::memory_order_relaxed);
  }

  Gemma4ExpertQMMCounterSnapshot snapshot() const {
    return {
        hits_.load(std::memory_order_relaxed),
        fallback_nax_.load(std::memory_order_relaxed),
        fallback_outer_route_.load(std::memory_order_relaxed),
        fallback_quantization_.load(std::memory_order_relaxed),
        fallback_topology_.load(std::memory_order_relaxed),
        fallback_assignment_count_.load(std::memory_order_relaxed),
        fallback_geometry_.load(std::memory_order_relaxed),
        fallback_metallib_unavailable_.load(std::memory_order_relaxed),
        fallback_sortedness_retracted_.load(std::memory_order_relaxed),
        armed_.load(std::memory_order_relaxed),
    };
  }

  Gemma4ExpertQMMCounterSnapshot snapshot_and_disarm() {
    const bool was_armed = armed_.load(std::memory_order_relaxed);
    armed_.store(false, std::memory_order_relaxed);
    auto result = snapshot();
    result.armed = was_armed;
    return result;
  }

  void reset() {
    hits_.store(0, std::memory_order_relaxed);
    fallback_nax_.store(0, std::memory_order_relaxed);
    fallback_outer_route_.store(0, std::memory_order_relaxed);
    fallback_quantization_.store(0, std::memory_order_relaxed);
    fallback_topology_.store(0, std::memory_order_relaxed);
    fallback_assignment_count_.store(0, std::memory_order_relaxed);
    fallback_geometry_.store(0, std::memory_order_relaxed);
    fallback_metallib_unavailable_.store(0, std::memory_order_relaxed);
    fallback_sortedness_retracted_.store(0, std::memory_order_relaxed);
  }

  void clear_and_arm() {
    reset();
    armed_.store(true, std::memory_order_relaxed);
  }

 private:
  std::atomic<bool> armed_{false};
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> fallback_nax_{0};
  std::atomic<uint64_t> fallback_outer_route_{0};
  std::atomic<uint64_t> fallback_quantization_{0};
  std::atomic<uint64_t> fallback_topology_{0};
  std::atomic<uint64_t> fallback_assignment_count_{0};
  std::atomic<uint64_t> fallback_geometry_{0};
  std::atomic<uint64_t> fallback_metallib_unavailable_{0};
  std::atomic<uint64_t> fallback_sortedness_retracted_{0};
};

} // namespace mlx::core::metal

#endif
