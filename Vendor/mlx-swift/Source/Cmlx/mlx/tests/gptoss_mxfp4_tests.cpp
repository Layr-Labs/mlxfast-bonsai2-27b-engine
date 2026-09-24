// Copyright © 2026 Eigen Labs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {
struct ScopedPrefillTile {
  std::optional<std::string> previous;
  explicit ScopedPrefillTile(const char* value) {
    if (const auto* old = std::getenv("MLX_GPTOSS_MXFP4_PREFILL_TILE")) previous = old;
    setenv("MLX_GPTOSS_MXFP4_PREFILL_TILE", value, 1);
  }
  ~ScopedPrefillTile() {
    if (previous) setenv("MLX_GPTOSS_MXFP4_PREFILL_TILE", previous->c_str(), 1);
    else unsetenv("MLX_GPTOSS_MXFP4_PREFILL_TILE");
  }
};
}

TEST_CASE("gptoss mxfp4 prefill tiles preserve expert boundaries and partial rows") {
  constexpr int E = 32, N = 2880, K = 2880;
  auto packed = reshape(
      multiply(arange(E * N * K / 8, uint32, Device::gpu), array(uint32_t{2654435761}), Device::gpu),
      {E, N, K / 8});
  auto scales = full({E, N, K / 32}, array(uint8_t{126}), Device::gpu);
  eval(packed, scales);

  for (const auto dtype : {float32, bfloat16}) {
    for (int rows : {128, 131, 257}) {
      std::vector<uint32_t> ids(rows);
      for (int i = 0; i < rows; ++i) ids[i] = (i * E) / rows;
      const array indices(ids.data(), {rows});
      auto x = astype(reshape(sin(arange(rows * K, float32, Device::gpu) * array(0.017f), Device::gpu),
                              {rows, 1, K}), dtype, Device::gpu);
      array reference(0.0f);
      {
        ScopedPrefillTile tile("legacy");
        reference = gather_qmm(x, packed, scales, std::nullopt, std::nullopt,
                               indices, true, 32, 4, "mxfp4", true, Device::gpu);
        eval(reference);
      }
      for (const auto* name : {"m32n32k32"}) {
        ScopedPrefillTile tile(name);
        const auto actual = gather_qmm(x, packed, scales, std::nullopt, std::nullopt,
                                       indices, true, 32, 4, "mxfp4", true, Device::gpu);
        eval(actual);
        INFO("tile=", name, " rows=", rows, " dtype=", dtype);
        const auto delta = abs(astype(actual, float32) - astype(reference, float32));
        const float worst = max(delta).item<float>();
        const float magnitude = max(abs(astype(reference, float32))).item<float>();
        CHECK(std::isfinite(worst));
        CHECK(worst <= (dtype == float32 ? 1e-4f : 0.016f) * std::max(1.0f, magnitude));
      }
    }
  }
}

namespace {
struct ScopedDecodeFastTail {
  std::optional<std::string> previous;
  explicit ScopedDecodeFastTail(const char* value) {
    if (const auto* old = std::getenv("MLX_GPTOSS_MXFP4_DECODE_FAST_TAIL")) previous = old;
    setenv("MLX_GPTOSS_MXFP4_DECODE_FAST_TAIL", value, 1);
  }
  ~ScopedDecodeFastTail() {
    if (previous) setenv("MLX_GPTOSS_MXFP4_DECODE_FAST_TAIL", previous->c_str(), 1);
    else unsetenv("MLX_GPTOSS_MXFP4_DECODE_FAST_TAIL");
  }
};
}

TEST_CASE("gptoss mxfp4 fast gather covers the final 320 input values") {
  constexpr int E = 32, N = 2880, K = 2880;
  auto packed = reshape(
      multiply(arange(E * N * K / 8, uint32, Device::gpu), array(uint32_t{2654435761}), Device::gpu),
      {E, N, K / 8});
  auto scales = full({E, N, K / 32}, array(uint8_t{126}), Device::gpu);
  eval(packed, scales);
  for (const auto dtype : {float32, bfloat16}) {
    for (int batch : {1, 2, 4, 8}) {
      std::vector<uint32_t> lhs_ids(batch * 4), rhs_ids(batch * 4);
      for (int row = 0; row < batch; ++row) {
        for (int expert = 0; expert < 4; ++expert) {
          lhs_ids[row * 4 + expert] = row;
          rhs_ids[row * 4 + expert] = (row * 7 + expert * 5) % E;
        }
      }
      const array lhs(lhs_ids.data(), {batch, 4});
      const array rhs(rhs_ids.data(), {batch, 4});
      for (bool tail_edges_only : {true, false}) {
        std::vector<float> values(batch * K, 0.0f);
        for (int row = 0; row < batch; ++row) {
          if (tail_edges_only) {
            values[row * K + 2559] = 0.75f;
            values[row * K + 2560] = 0.25f;
            values[row * K + 2879] = -0.5f;
          } else {
            for (int k = 0; k < K; ++k) values[row * K + k] = std::sin((row * K + k) * 0.017f);
          }
        }
        auto x = astype(array(values.data(), {batch, 1, K}), dtype, Device::gpu);
        array reference(0.0f);
        {
          ScopedDecodeFastTail flag("0");
          reference = gather_qmm(x, packed, scales, std::nullopt, lhs, rhs,
                                 true, 32, 4, "mxfp4", false, Device::gpu);
          eval(reference);
        }
        ScopedDecodeFastTail flag("1");
        const auto actual = gather_qmm(x, packed, scales, std::nullopt, lhs, rhs,
                                       true, 32, 4, "mxfp4", false, Device::gpu);
        eval(actual);
        INFO("batch=", batch, " dtype=", dtype, " tail_edges_only=", tail_edges_only);
        if (tail_edges_only) {
          CHECK(array_equal(actual, reference).item<bool>());
        } else {
          const auto delta = abs(astype(actual, float32) - astype(reference, float32));
          const float worst = max(delta).item<float>();
          const float magnitude = max(abs(astype(reference, float32))).item<float>();
          CHECK(std::isfinite(worst));
          CHECK(worst <= (dtype == float32 ? 1e-4f : 0.016f) * std::max(1.0f, magnitude));
        }
      }
    }
  }
}
