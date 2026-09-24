// Copyright © 2026 Apple Inc.

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace mlx::core::allocator {

inline size_t checked_allocation_add(size_t size, size_t extra) {
  if (extra > std::numeric_limits<size_t>::max() - size) {
    throw std::overflow_error("allocation footprint overflow");
  }
  return size + extra;
}

inline size_t round_allocation_size(size_t size, size_t alignment) {
  if (alignment == 0) {
    throw std::invalid_argument("allocation alignment is zero");
  }
  auto remainder = size % alignment;
  return checked_allocation_add(size, remainder ? alignment - remainder : 0);
}

// Inclusive cache-reuse bound. A fresh buffer never exceeds this bound.
inline size_t maximum_reuse_size(size_t size, size_t page_size) {
  if (size == 0) {
    return 0;
  }
  auto two_pages = checked_allocation_add(page_size, page_size);
  if (two_pages == 0) {
    throw std::invalid_argument("cache page size is zero");
  }
  return checked_allocation_add(size, std::min(size - 1, two_pages - 1));
}

} // namespace mlx::core::allocator
