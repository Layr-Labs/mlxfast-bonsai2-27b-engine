/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_MEMORY_H
#define MLX_MEMORY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include <Cmlx/mlx-c-array.h>
#include <Cmlx/mlx-c-closure.h>
#include <Cmlx/mlx-c-distributed_group.h>
#include <Cmlx/mlx-c-io_types.h>
#include <Cmlx/mlx-c-map.h>
#include <Cmlx/mlx-c-stream.h>
#include <Cmlx/mlx-c-string.h>
#include <Cmlx/mlx-c-vector.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup memory Memory operations
 */
/**@{*/

int mlx_clear_cache(void);
int mlx_get_active_memory(size_t* res);
int mlx_get_allocation_size_upper_bound(size_t* res, size_t size);

/* Immutable allocator geometry. Bound functions return 1 on invalid policy or
 * overflow without invoking the error callback or allocating an exception. */
typedef struct mlx_allocation_footprint_policy {
  size_t alignment;
  size_t rounding_threshold;
  size_t minimum_allocation;
  size_t power_of_two_below;
  size_t cache_page_size;
} mlx_allocation_footprint_policy;

int mlx_get_allocation_footprint_policy(mlx_allocation_footprint_policy* res);
int mlx_allocation_footprint_policy_bound(size_t* res,
    const mlx_allocation_footprint_policy* policy, size_t size);
int mlx_allocation_footprint_policy_maximum_extra(size_t* res,
    const mlx_allocation_footprint_policy* policy);

int mlx_get_cache_memory(size_t* res);
int mlx_get_memory_limit(size_t* res);
int mlx_get_memory_snapshot(size_t* active, size_t* cache, size_t* peak);

int mlx_get_num_resources(size_t* res);
int mlx_get_peak_memory(size_t* res);
int mlx_get_resource_limit(size_t* res);
int mlx_reset_peak_memory(void);
int mlx_set_cache_limit(size_t* res, size_t limit);
int mlx_set_memory_limit(size_t* res, size_t limit);
int mlx_set_wired_limit(size_t* res, size_t limit);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
