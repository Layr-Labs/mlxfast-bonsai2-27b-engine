// Copyright © 2026 Eigen Labs.
#pragma once

#include <cstdlib>
#include <string_view>

namespace mlx::core::metal {

struct GPTOSSMXFP4PrefillTile {
  int bm = 16;
  int bn = 32;
  int bk = 32;
  int wm = 1;
  int wn = 2;
};

inline GPTOSSMXFP4PrefillTile gptoss_mxfp4_prefill_tile(const char* option) {
  const std::string_view value = option ? option : "";
  if (value == "m32n32k32") return {32, 32, 32, 2, 2};
  return {};
}

} // namespace mlx::core::metal
