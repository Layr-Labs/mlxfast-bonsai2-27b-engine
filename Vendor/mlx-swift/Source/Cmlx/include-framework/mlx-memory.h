#ifdef __cplusplus
// Copyright © 2025 Apple Inc.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <limits>

#include <Cmlx/mlx-api.h>

namespace mlx::core {

struct MemorySnapshot {
  size_t active_memory;
  size_t cache_memory;
  size_t peak_memory;
};

/* Read allocator accounting under one lock. This does not synchronize streams
 * or include allocations that have not entered allocator accounting yet. */
MLX_API MemorySnapshot get_memory_snapshot();

/* Bound one allocator buffer, including alignment and larger cache reuse.
 * This does not allocate, synchronize, or inspect the live buffer cache. */
MLX_API size_t get_allocation_size_upper_bound(size_t size);

// A detached value: evaluating bounds requires no allocator, error callback,
// exception, lock, or allocation. Capture once before entering admission locks.
struct AllocationFootprintPolicy {
  size_t alignment;
  size_t rounding_threshold;
  size_t minimum_allocation;
  size_t power_of_two_below;
  size_t cache_page_size;

  bool upper_bound(size_t size, size_t& result) const noexcept {
    if (size == 0) { result = 0; return true; }
    if (alignment == 0 || cache_page_size == 0) { return false; }
    size = std::max(size, minimum_allocation);
    if (power_of_two_below && size < power_of_two_below) {
      size_t rounded = 1;
      while (rounded < size) {
        if (!add(rounded, rounded, rounded)) { return false; }
      }
      size = rounded;
    } else if (size > rounding_threshold) {
      auto remainder = size % alignment;
      if (remainder && !add(size, alignment - remainder, size)) { return false; }
    }
    size_t two_pages;
    return add(cache_page_size, cache_page_size, two_pages) &&
        add(size, std::min(size - 1, two_pages - 1), result);
  }

  // For any positive n with a valid bound, B(n) <= n + this overhead.
  bool maximum_extra_bytes(size_t& result) const noexcept {
    if (alignment == 0 || cache_page_size == 0) { return false; }
    auto normalization = std::max({alignment - 1,
        minimum_allocation ? minimum_allocation - 1 : 0,
        power_of_two_below ? power_of_two_below - 1 : 0});
    size_t two_pages;
    return add(cache_page_size, cache_page_size, two_pages) &&
        add(normalization, two_pages - 1, result);
  }

 private:
  static bool add(size_t a, size_t b, size_t& result) noexcept {
    if (b > std::numeric_limits<size_t>::max() - a) { return false; }
    result = a + b;
    return true;
  }
};

MLX_API AllocationFootprintPolicy get_allocation_footprint_policy() noexcept;

/* Get the actively used memory in bytes.
 *
 * Note, this will not always match memory use reported by the system because
 * it does not include cached memory buffers.
 * */
MLX_API size_t get_active_memory();

/* Get the peak amount of used memory in bytes.
 *
 * The maximum memory used recorded from the beginning of the program
 * execution or since the last call to reset_peak_memory.
 * */
MLX_API size_t get_peak_memory();

/* Reset the peak memory to zero.
 * */
MLX_API void reset_peak_memory();

/* Get the cache size in bytes.
 *
 * The cache includes memory not currently used that has not been returned
 * to the system allocator.
 * */
MLX_API size_t get_cache_memory();

/* Get the number of live Metal resources (buffers).
 *
 * This is a COUNT, independent of byte usage. The Metal backend throws when it
 * reaches the resource limit (see get_resource_limit). Many small cached
 * buffers can push this count high while byte usage stays low.
 * */
MLX_API size_t get_num_resources();

/* Get the hard ceiling on the number of live Metal resources (buffers).
 *
 * Defaults to the iogpu.rsrc_limit sysctl (~499000 when unset). Allocation
 * throws once get_num_resources() reaches this value.
 * */
MLX_API size_t get_resource_limit();

/* Set the memory limit.
 * The memory limit is a guideline for the maximum amount of memory to use
 * during graph evaluation. If the memory limit is exceeded and there is no
 * more RAM (including swap when available) allocations will result in an
 * exception.
 *
 * When Metal is available the memory limit defaults to 1.5 times the maximum
 * recommended working set size reported by the device.
 *
 * Returns the previous memory limit.
 * */
MLX_API size_t set_memory_limit(size_t limit);

/* Get the current memory limit. */
MLX_API size_t get_memory_limit();

/* Set the cache limit.
 * If using more than the given limit, free memory will be reclaimed
 * from the cache on the next allocation. To disable the cache,
 * set the limit to 0.
 *
 * The cache limit defaults to the memory limit.
 *
 * Returns the previous cache limit.
 * */
MLX_API size_t set_cache_limit(size_t limit);

/* Clear the memory cache. */
MLX_API void clear_cache();

/* Set the wired size limit.
 *
 * Note, this function is only useful when using the Metal backend with
 * macOS 15.0 or higher.
 *
 * The wired limit is the total size in bytes of memory that will be kept
 * resident. The default value is ``0``.
 *
 * Setting a wired limit larger than system wired limit is an error.
 *
 * Returns the previous wired limit.
 * */
MLX_API size_t set_wired_limit(size_t limit);

} // namespace mlx::core

#endif
