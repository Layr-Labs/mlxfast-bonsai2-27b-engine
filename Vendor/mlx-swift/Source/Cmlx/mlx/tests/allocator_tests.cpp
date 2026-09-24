// Copyright © 2023-2026 Apple Inc.

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <limits>
#include <stdexcept>
#include <thread>

#include "doctest/doctest.h"

#include "mlx/allocator.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "mlx/scheduler.h"
#include "mlx/stream.h"

using namespace mlx::core;

TEST_CASE("test simple allocations") {
  {
    auto buffer = allocator::malloc(sizeof(float));
    auto fptr = static_cast<float*>(buffer.raw_ptr());
    *fptr = 0.5f;
    CHECK_EQ(*fptr, 0.5f);
    allocator::free(buffer);
  }

  {
    auto buffer = allocator::malloc(128 * sizeof(int));
    int* ptr = static_cast<int*>(buffer.raw_ptr());
    for (int i = 0; i < 128; ++i) {
      ptr[i] = i;
    }
    allocator::free(buffer);
  }

  {
    auto buffer = allocator::malloc(0);
    allocator::free(buffer);
  }
}

TEST_CASE("test large allocations") {
  size_t size = 1 << 30;
  for (int i = 0; i < 100; ++i) {
    auto buffer = allocator::malloc(size);
    allocator::free(buffer);
  }
}

TEST_CASE("test cached allocation keeps capacity") {
  auto old_limit = set_cache_limit(1 << 20);
  clear_cache();

  auto large = allocator::malloc(8192);
  allocator::free(large);
  auto cached = get_cache_memory();
  CHECK_GE(cached, 8192);

  auto small = allocator::malloc(6000);
  CHECK_GE(allocator::allocator().size(small), cached);
  allocator::free(small);
  CHECK_GE(get_cache_memory(), cached);

  clear_cache();
  set_cache_limit(old_limit);
}

TEST_CASE("test clear cache synchronizes cpu streams") {
  if (is_available(Device{Device::gpu})) {
    return;
  }

  auto old_limit = set_cache_limit(1 << 20);
  clear_cache();

  auto cached = allocator::malloc(8192);
  allocator::free(cached);
  CHECK_GE(get_cache_memory(), 8192);

  auto task_started = std::make_shared<std::promise<void>>();
  auto task_started_future = task_started->get_future();
  auto release_task = std::make_shared<std::promise<void>>();
  auto release_task_future = release_task->get_future().share();
  auto task_finished = std::make_shared<std::promise<void>>();
  auto task_finished_future = task_finished->get_future();
  auto clear_finished = std::make_shared<std::promise<void>>();
  auto clear_finished_future = clear_finished->get_future();

  auto stream = new_stream(Device{Device::cpu});
  scheduler::enqueue(
      stream, [task_started, release_task_future, task_finished] {
        task_started->set_value();
        release_task_future.wait();
        task_finished->set_value();
      });

  task_started_future.wait();

  std::thread clear_thread([clear_finished] {
    clear_cache();
    clear_finished->set_value();
  });

  CHECK_EQ(
      clear_finished_future.wait_for(std::chrono::milliseconds(50)),
      std::future_status::timeout);

  release_task->set_value();

  CHECK_EQ(
      task_finished_future.wait_for(std::chrono::seconds(10)),
      std::future_status::ready);
  CHECK_EQ(
      clear_finished_future.wait_for(std::chrono::seconds(10)),
      std::future_status::ready);
  clear_thread.join();

  set_cache_limit(old_limit);
}

TEST_CASE("test coherent memory snapshot during cache transfers") {
  auto old_limit = set_cache_limit(1 << 20);
  clear_cache();
  auto live = allocator::malloc(32768);
  auto moving = allocator::malloc(65536);
  auto moving_bytes = allocator::allocator().size(moving);
  auto active = get_memory_snapshot();
  allocator::free(moving);
  auto cached = get_memory_snapshot();
  CHECK_EQ(active.active_memory, cached.active_memory + moving_bytes);
  CHECK_EQ(active.cache_memory + moving_bytes, cached.cache_memory);
  CHECK_EQ(active.peak_memory, cached.peak_memory);
  auto total = cached.active_memory + cached.cache_memory;

  std::atomic<bool> stop{false};
  std::promise<void> started;
  auto started_future = started.get_future();
  auto churn = std::async(std::launch::async, [&] {
    started.set_value();
    size_t transfers = 0;
    do {
      auto buffer = allocator::malloc(65536);
      allocator::free(buffer);
      ++transfers;
    } while (!stop.load(std::memory_order_relaxed));
    return transfers;
  });
  started_future.wait();

  size_t inconsistent = 0;
  for (int i = 0; i < 20000; ++i) {
    auto snapshot = get_memory_snapshot();
    inconsistent += snapshot.active_memory + snapshot.cache_memory != total;
    inconsistent += snapshot.active_memory > snapshot.peak_memory;
  }
  stop.store(true, std::memory_order_relaxed);
  CHECK_GT(churn.get(), 0);
  CHECK_EQ(inconsistent, 0);

  allocator::free(live);
  clear_cache();
  set_cache_limit(old_limit);
}

TEST_CASE("test memory snapshot does not wait for stream work") {
  std::promise<void> started;
  auto started_future = started.get_future();
  std::promise<void> release;
  auto release_future = release.get_future().share();
  auto stream = new_stream(Device{Device::cpu});
  scheduler::enqueue(stream, [&] {
    started.set_value();
    release_future.wait();
  });
  started_future.wait();

  auto snapshot = std::async(std::launch::async, [] {
    return get_memory_snapshot();
  });
  auto status = snapshot.wait_for(std::chrono::seconds(1));
  release.set_value();
  synchronize(stream);
  snapshot.get();
  CHECK_EQ(status, std::future_status::ready);
}

TEST_CASE("allocation footprint bounds fresh and cached buffer owners") {
  auto old_limit = set_cache_limit(1 << 20);
  clear_cache();
  for (size_t size : {size_t(4), size_t(6000), size_t(24576), size_t(65536)}) {
    auto bound = get_allocation_size_upper_bound(size);
    CHECK_GE(bound, size);
    auto buffer = allocator::malloc(size);
    CHECK_LE(allocator::allocator().size(buffer), bound);
    allocator::free(buffer);
  }
  clear_cache();
  auto large = allocator::malloc(49152);
  auto larger_bytes = allocator::allocator().size(large);
  allocator::free(large);
  auto reused = allocator::malloc(32768);
  CHECK_LE(allocator::allocator().size(reused),
           get_allocation_size_upper_bound(32768));
  CHECK_GE(larger_bytes, 49152);
  allocator::free(reused);
  clear_cache();
  set_cache_limit(old_limit);
}

TEST_CASE("allocation prediction does not change allocator counters") {
  auto before = get_memory_snapshot();
  CHECK_GE(get_allocation_size_upper_bound(24576), 24576);
  CHECK_EQ(get_allocation_size_upper_bound(0), 0);
  CHECK_THROWS(get_allocation_size_upper_bound(std::numeric_limits<size_t>::max()));
  auto after = get_memory_snapshot();
  CHECK_EQ(after.active_memory, before.active_memory);
  CHECK_EQ(after.cache_memory, before.cache_memory);
  CHECK_EQ(after.peak_memory, before.peak_memory);
}

TEST_CASE("detached allocation policies preserve backend size classes without exceptions") {
  const AllocationFootprintPolicy cpu{1, 0, 0, 0, 4096};
  const AllocationFootprintPolicy metal{16384, 16384, 0, 0, 16384};
  const AllocationFootprintPolicy cuda{16384, 0, 8, 16384, 16384};
  size_t result = 0;
  CHECK(cpu.upper_bound(24576, result));
  CHECK_EQ(result, 32767);
  CHECK(metal.upper_bound(24576, result));
  CHECK_EQ(result, 65535);
  CHECK(cuda.upper_bound(1, result));
  CHECK_EQ(result, 15);
  CHECK(cuda.upper_bound(8193, result));
  CHECK_EQ(result, 32767);
  for (auto p : {cpu, metal, cuda}) {
    size_t extra = 0;
    REQUIRE(p.maximum_extra_bytes(extra));
    for (size_t n : {1, 4, 8191, 8192, 8193, 16383, 16384, 16385, 24576, 65537}) {
      REQUIRE(p.upper_bound(n, result));
      CHECK_GE(result, n);
      CHECK_LE(result - n, extra);
    }
    CHECK_FALSE(p.upper_bound(std::numeric_limits<size_t>::max(), result));
  }
}
