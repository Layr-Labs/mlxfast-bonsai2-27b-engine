/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/memory.h"
#include "mlx/memory.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" int mlx_clear_cache(void) {
  try {
    mlx::core::clear_cache();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_active_memory(size_t* res) {
  try {
    *res = mlx::core::get_active_memory();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_cache_memory(size_t* res) {
  try {
    *res = mlx::core::get_cache_memory();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_memory_limit(size_t* res) {
  try {
    *res = mlx::core::get_memory_limit();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_memory_snapshot(size_t* active, size_t* cache, size_t* peak) {
  try {
    auto snapshot = mlx::core::get_memory_snapshot();
    *active = snapshot.active_memory;
    *cache = snapshot.cache_memory;
    *peak = snapshot.peak_memory;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_get_num_resources(size_t* res) {
  try {
    *res = mlx::core::get_num_resources();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_peak_memory(size_t* res) {
  try {
    *res = mlx::core::get_peak_memory();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_get_resource_limit(size_t* res) {
  try {
    *res = mlx::core::get_resource_limit();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_reset_peak_memory(void) {
  try {
    mlx::core::reset_peak_memory();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_set_cache_limit(size_t* res, size_t limit) {
  try {
    *res = mlx::core::set_cache_limit(limit);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_set_memory_limit(size_t* res, size_t limit) {
  try {
    *res = mlx::core::set_memory_limit(limit);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_set_wired_limit(size_t* res, size_t limit) {
  try {
    *res = mlx::core::set_wired_limit(limit);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_get_allocation_size_upper_bound(size_t* res, size_t size) {
  try {
    *res = mlx::core::get_allocation_size_upper_bound(size);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_get_allocation_footprint_policy(mlx_allocation_footprint_policy* res) {
  if (!res) { return 1; }
  auto p = mlx::core::get_allocation_footprint_policy();
  *res = {p.alignment, p.rounding_threshold, p.minimum_allocation,
      p.power_of_two_below, p.cache_page_size};
  return 0;
}

static mlx::core::AllocationFootprintPolicy core_policy(
    const mlx_allocation_footprint_policy& p) noexcept {
  return {p.alignment, p.rounding_threshold, p.minimum_allocation,
      p.power_of_two_below, p.cache_page_size};
}

extern "C" int mlx_allocation_footprint_policy_bound(size_t* res,
    const mlx_allocation_footprint_policy* policy, size_t size) {
  return res && policy && core_policy(*policy).upper_bound(size, *res) ? 0 : 1;
}

extern "C" int mlx_allocation_footprint_policy_maximum_extra(size_t* res,
    const mlx_allocation_footprint_policy* policy) {
  return res && policy && core_policy(*policy).maximum_extra_bytes(*res) ? 0 : 1;
}
