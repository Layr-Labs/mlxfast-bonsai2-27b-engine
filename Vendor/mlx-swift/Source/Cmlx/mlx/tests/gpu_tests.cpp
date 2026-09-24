// Copyright © 2023-2024 Apple Inc.

#include <array>
#include <chrono>
#include <cmath>
#include <future>

#include "doctest/doctest.h"
#include "mlx/backend/common/gemma4_expert_qmm.h"
#include "mlx/mlx.h"

using namespace mlx::core;

static const std::array<Dtype, 5> types =
    {bool_, uint32, int32, int64, float32};

TEST_CASE("test gpu arange") {
  for (auto t : types) {
    if (t == bool_) {
      continue;
    }
    auto out_cpu = arange(1, 100, 2, t, Device::cpu);
    auto out_gpu = arange(1, 100, 2, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());

    out_cpu = arange(1, 5, 0.25, t, Device::cpu);
    out_gpu = arange(1, 5, 0.25, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }
}

TEST_CASE("test gpu full") {
  for (auto t : types) {
    auto out_cpu = full({4, 4}, 2, t, Device::cpu);
    auto out_gpu = full({4, 4}, 2, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  // Check broadcasting works
  {
    auto x = full({2, 2}, array({3, 4}, {2, 1}), Device::gpu);
    CHECK(
        array_equal(x, array({3, 3, 4, 4}, {2, 2}), Device::cpu).item<bool>());
    x = full({2, 2}, array({3, 4}, {1, 2}), Device::gpu);
    CHECK(
        array_equal(x, array({3, 4, 3, 4}, {2, 2}), Device::cpu).item<bool>());
  }

  // Check zeros and ones
  {
    auto x = zeros({2, 2}, float32, Device::gpu);
    auto y = array({0.0, 0.0, 0.0, 0.0}, {2, 2});
    CHECK(array_equal(x, y, Device::cpu).item<bool>());

    x = ones({2, 2}, float32, Device::gpu);
    y = array({1.0, 1.0, 1.0, 1.0}, {2, 2});
    CHECK(array_equal(x, y, Device::cpu).item<bool>());
  }
}

TEST_CASE("test gpu astype") {
  array x = array({-4, -3, -2, -1, 0, 1, 2, 3});
  // Check all types work
  for (auto t : types) {
    auto out_cpu = astype(x, t, Device::cpu);
    auto out_gpu = astype(x, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  x = transpose(reshape(x, {2, 2, 2}), {1, 2, 0});
  for (auto t : types) {
    auto out_cpu = astype(x, t, Device::cpu);
    auto out_gpu = astype(x, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }
}

TEST_CASE("test gpu reshape") {
  array x = array({0, 1, 2, 3, 4, 5, 6, 7});
  auto out_cpu = reshape(x, {2, 2, 2});
  auto out_gpu = reshape(x, {2, 2, 2}, Device::gpu);
  CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());

  x = transpose(reshape(x, {2, 2, 2}), {1, 2, 0});
  out_cpu = reshape(x, {4, 2});
  out_gpu = reshape(x, {4, 2}, Device::gpu);
  CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());

  out_cpu = reshape(x, {8});
  out_gpu = reshape(x, {8}, Device::gpu);
  CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
}

TEST_CASE("test gpu reduce") {
  {
    array a(true);
    CHECK_EQ(all(a, Device::gpu).item<bool>(), true);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), true);

    a = array(std::initializer_list<bool>{});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), true);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), false);
  }

  {
    std::vector<int> vals(33, 1);
    array a(vals.data(), {33});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), true);

    vals[32] = 0;
    a = array(vals.data(), {33});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), false);
  }

  {
    std::vector<int> vals(33, 0);
    array a(vals.data(), {33});
    CHECK_EQ(any(a, Device::gpu).item<bool>(), false);

    vals[32] = 1;
    a = array(vals.data(), {33});
    CHECK_EQ(any(a, Device::gpu).item<bool>(), true);
  }

  {
    std::vector<int> vals(1 << 14, 0);
    array a(vals.data(), {1 << 14});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), false);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), false);

    vals[4] = 1;
    vals[999] = 1;
    vals[2000] = 1;
    a = array(vals.data(), {1 << 14});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), false);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), true);
  }

  // sum and prod
  {
    array a = array({true, false, true});
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 2);
    CHECK_EQ(prod(a, Device::gpu).item<bool>(), false);

    a = array({true, true, true});
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 3);
    CHECK_EQ(prod(a, Device::gpu).item<bool>(), true);

    a = full({2, 2, 2}, 2.0f);
    CHECK_EQ(sum(a, Device::gpu).item<float>(), 16.0f);
    CHECK_EQ(prod(a, Device::gpu).item<float>(), 256.0f);

    a = full({500, 2, 2}, 1u);
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 2000);
    CHECK_EQ(prod(a, Device::gpu).item<uint32_t>(), 1u);

    a = full({500, 2, 2}, 1);
    CHECK_EQ(sum(a, Device::gpu).item<int32_t>(), 2000);
    CHECK_EQ(prod(a, Device::gpu).item<int32_t>(), 1);
  }

  // sum and prod overflow
  {
    auto a = full({256, 2, 2}, 1u, uint8);
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 256 * 4);
    CHECK_EQ(prod(a, Device::gpu).item<uint32_t>(), 1);

    a = full({65535, 2, 2}, 1u, uint16);
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 65535 * 4);
    CHECK_EQ(prod(a, Device::gpu).item<uint32_t>(), 1);
  }
}

TEST_CASE("test gpu reduce with axes") {
  // reducing only some axes and irregular layouts
  {
    array a(1.0f);
    a = broadcast_to(a, {2, 2, 2});
    CHECK_EQ(sum(a, Device::gpu).item<float>(), 8.0f);

    a = ones({2, 4, 8, 16});
    for (auto ax : {0, 1, 2, 3}) {
      auto out_gpu = sum(a, ax, false, Device::gpu);
      auto out_cpu = sum(a, ax, false, Device::cpu);
      CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    }

    for (auto ax : {1, 2, 3}) {
      auto out_gpu = sum(a, {0, ax}, false, Device::gpu);
      auto out_cpu = sum(a, {0, ax}, false, Device::cpu);
      CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    }
    for (auto ax : {2, 3}) {
      auto out_gpu = sum(a, {0, 1, ax}, false, Device::gpu);
      auto out_cpu = sum(a, {0, 1, ax}, false, Device::cpu);
      CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    }
  }
}

TEST_CASE("test gpu binary ops") {
  // scalar-scalar
  {
    array a(2.0f);
    array b(4.0f);
    auto out = add(a, b, Device::gpu);
    CHECK_EQ(out.item<float>(), 6.0f);
  }

  // scalar-vector and vector-scalar
  {
    array a(2.0f);
    array b({2.0f, 4.0f, 6.0f});
    auto out = add(a, b, Device::gpu);
    auto expected = array({4.0f, 6.0f, 8.0f});
    CHECK(array_equal(out, expected, Device::cpu).item<bool>());
    out = add(b, a, Device::gpu);
    CHECK(array_equal(out, expected, Device::cpu).item<bool>());
  }

  // vector-vector
  {
    array a({0.0f, 1.0f, 2.0f});
    array b({3.0f, 4.0f, 5.0f});
    auto out = add(a, b, Device::gpu);
    auto expected = array({3.0f, 5.0f, 7.0f});
    CHECK(array_equal(out, expected, Device::cpu).item<bool>());
  }

  // general
  {
    array a({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, {2, 2, 2});
    array b({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, {2, 2, 2});
    a = transpose(a, {0, 2, 1});
    b = transpose(b, {1, 0, 2});
    auto out_gpu = add(a, b, Device::gpu);
    auto out_cpu = add(a, b, Device::cpu);
    auto expected =
        array({0.0f, 3.0f, 5.0f, 8.0f, 6.0f, 9.0f, 11.0f, 14.0f}, {2, 2, 2});
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    CHECK(array_equal(out_gpu, expected, Device::cpu).item<bool>());
  }

  // Check all types work
  for (auto t : types) {
    auto a = astype(array({0, 1, 2}), t);
    auto b = astype(array({3, 4, 5}), t);
    auto out_cpu = add(a, b, Device::cpu);
    auto out_gpu = add(a, b, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  // Check subtraction
  {
    auto a = array({3, 2, 1});
    auto b = array({1, 1, 1});
    auto out = subtract(a, b, Device::gpu);
    CHECK(array_equal(out, array({2, 1, 0}), Device::cpu).item<bool>());
  }

  // Check multiplication
  {
    auto a = array({1, 2, 3});
    auto b = array({2, 2, 2});
    auto out = multiply(a, b, Device::gpu);
    CHECK(array_equal(out, array({2, 4, 6}), Device::cpu).item<bool>());
  }

  // Check division
  {
    auto x = array(1.0f);
    auto y = array(1.0f);
    CHECK_EQ(divide(x, y, Device::gpu).item<float>(), 1.0f);

    x = array(1.0f);
    y = array(0.5);
    CHECK_EQ(divide(x, y, Device::gpu).item<float>(), 2.0f);

    x = array(1.0f);
    y = array(0.0f);
    CHECK(std::isinf(divide(x, y, Device::gpu).item<float>()));

    x = array(0.0f);
    y = array(0.0f);
    CHECK(std::isnan(divide(x, y, Device::gpu).item<float>()));
  }

  // Check maximum and minimum
  {
    auto x = array(1.0f);
    auto y = array(0.0f);
    CHECK_EQ(maximum(x, y, Device::gpu).item<float>(), 1.0f);
    CHECK_EQ(minimum(x, y, Device::gpu).item<float>(), 0.0f);
    y = array(2.0f);
    CHECK_EQ(maximum(x, y, Device::gpu).item<float>(), 2.0f);
    CHECK_EQ(minimum(x, y, Device::gpu).item<float>(), 1.0f);
  }

  // Check equal
  {
    array x(1.0f);
    array y(1.0f);
    CHECK(equal(x, y, Device::gpu).item<bool>());
    x = array(0.0f);
    CHECK(!equal(x, y, Device::gpu).item<bool>());
  }

  // Greater and less
  {
    array x(1.0f);
    array y(0.0f);
    CHECK(greater(x, y, Device::gpu).item<bool>());
    CHECK(greater_equal(x, y, Device::gpu).item<bool>());
    CHECK(!greater(y, x, Device::gpu).item<bool>());
    CHECK(!greater_equal(y, x, Device::gpu).item<bool>());
    y = array(1.0f);
    CHECK(!greater(x, y, Device::gpu).item<bool>());
    CHECK(greater_equal(x, y, Device::gpu).item<bool>());

    x = array(0.0f);
    y = array(1.0f);
    CHECK(less(x, y, Device::gpu).item<bool>());
    CHECK(less_equal(x, y, Device::gpu).item<bool>());
    CHECK(!less(y, x, Device::gpu).item<bool>());
    CHECK(!less_equal(y, x, Device::gpu).item<bool>());
    y = array(0.0f);
    CHECK(!less(x, y, Device::gpu).item<bool>());
    CHECK(less_equal(x, y, Device::gpu).item<bool>());
  }

  // Check logaddexp
  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    array x(inf);
    array y(2.0f);
    auto out = logaddexp(x, y, Device::gpu);
    CHECK_EQ(out.item<float>(), inf);

    x = array(-inf);
    out = logaddexp(x, y, Device::gpu);
    CHECK_EQ(out.item<float>(), 2.0f);

    y = array(-inf);
    out = logaddexp(x, y, Device::gpu);
    CHECK_EQ(out.item<float>(), -inf);
  }
}

TEST_CASE("test gpu unary ops") {
  // contiguous
  {
    array x({-1.0f, 0.0f, 1.0f});
    auto expected = array({1.0f, 0.0f, 1.0f});
    CHECK(array_equal(abs(x, Device::gpu), expected, Device::cpu).item<bool>());
  }

  // general
  {
    array x({-1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 3.0f, -3.0f});
    auto y = slice(x, {0}, {8}, {2});
    auto expected = array({1.0f, 1.0f, 1.0f, 3.0f});
    CHECK(array_equal(abs(y, Device::gpu), expected, Device::cpu).item<bool>());

    y = slice(x, {4}, {8});
    expected = array({1.0f, 1.0f, 3.0f, 3.0f});
    CHECK(array_equal(abs(y, Device::gpu), expected, Device::cpu).item<bool>());
  }

  // Test negative
  {
    array x(1.0f);
    CHECK_EQ(negative(x, Device::gpu).item<float>(), -1.0f);
  }

  // Check all types work
  for (auto t : types) {
    if (t == bool_) {
      continue;
    }
    auto in = astype(array({1}), t);
    auto out_cpu = negative(in, Device::cpu);
    auto out_gpu = negative(in, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  // Test log1p
  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    array x(-1.0f);
    CHECK_EQ(log1p(x, Device::gpu).item<float>(), -inf);

    x = array(0.0f);
    CHECK_EQ(log1p(x, Device::gpu).item<float>(), 0.0f);

    x = array(1e-9f);
    CHECK_EQ(log1p(x, Device::gpu).item<float>(), 1e-9f);

    x = array(-2.0f);
    CHECK(std::isnan(log1p(x, Device::gpu).item<float>()));
  }
}

TEST_CASE("test gpu random") {
  {
    auto key = random::key(0);
    auto x = random::bits({}, 4, key, Device::gpu);
    auto y = random::bits({}, 4, key, Device::gpu);
    CHECK_EQ(x.item<uint32_t>(), 1797259609u);
    CHECK_EQ(x.item<uint32_t>(), y.item<uint32_t>());
  }

  {
    auto key = random::key(1);
    auto x = random::bits({}, 4, key, Device::gpu);
    CHECK_EQ(x.item<uint32_t>(), 507451445u);
  }

  {
    auto key = random::key(0);
    auto x = random::bits({3, 1}, 4, key, Device::gpu);
    auto expected = array({4146024105u, 1351547692u, 2718843009u}, {3, 1});
    CHECK(array_equal(x, expected, Device::cpu).item<bool>());
  }
}

TEST_CASE("test gpu matmul") {
  {
    auto a = ones({2, 2});
    auto b = ones({2, 2});
    auto out = matmul(a, b, Device::gpu);
    CHECK(array_equal(out, full({2, 2}, 2.0f), Device::cpu).item<bool>());
  }

  // Batched matmul
  {
    auto a = ones({3, 2, 2});
    auto b = ones({3, 2, 2});
    auto out = matmul(a, b, Device::gpu);
    CHECK(array_equal(out, full({3, 2, 2}, 2.0f), Device::cpu).item<bool>());
  }

  // Broadcast batched matmul
  {
    auto a = ones({1, 3, 2, 2});
    auto b = ones({3, 1, 2, 2});
    auto out = matmul(a, b, Device::gpu);
    CHECK(array_equal(out, full({3, 3, 2, 2}, 2.0f), Device::cpu).item<bool>());
  }
}

TEST_CASE("test gpu validation") {
  // Run this test with Metal validation enabled
  // METAL_DEVICE_WRAPPER_TYPE=1 METAL_DEBUG_ERROR_MODE=0 ./tests/tests \
  //     -tc="test metal validation"

  auto x = array({});
  eval(exp(x));

  auto y = array({});
  eval(add(x, y));

  eval(sum(x));

  x = array({1, 2, 3});
  y = array(0);
  eval(gather(x, y, 0, {0}));
  eval(gather(x, y, 0, {2}));

  eval(gather(x, y, 0, {0}));
  eval(gather(x, y, 0, {2}));

  eval(scatter(x, y, array({2}), 0));

  x = arange(0, -3, 1);
  eval(x);
  array_equal(x, array({})).item<bool>();

  x = array({1.0, 0.0});
  eval(argmax(x));

  eval(scatter_max(array(1), {}, array(2), std::vector<int>{}));
}

TEST_CASE("test dynamic slice update waits for its start") {
  // Regression for #3880: a donated array-valued start could be read
  // stale when a command-buffer boundary lands between its producer
  // and the dynamic slice. The boundary occurs when the buffer's op
  // or memory limits split the graph (or with MLX_MAX_OPS_PER_BUFFER
  // set low); without a split the checks still assert the correct
  // update position.
  auto source = ones({2, 1 << 26}, int32);
  auto target = zeros({4, 4}, int32);
  auto update = full({1, 1}, 7, int32);
  eval(source, target, update);

  {
    auto recycled = zeros({2}, int32);
    eval(recycled);
  }

  auto out = [&] {
    auto start = max(source, 1, false);
    return slice_update(target, update, start, {0, 1});
  }();

  CHECK_EQ(slice(out, {1, 1}, {2, 2}).item<int>(), 7);
  CHECK_EQ(slice(out, {0, 0}, {1, 1}).item<int>(), 0);
}

TEST_CASE("test gpu int32 shape overflow errors") {
  // (2^30, 2).flatten() — product 2^31 doesn't fit in ShapeElem.
  // Issue #2681 reported wrapped shape (-2147483648,) and a
  // 2^64 - X reported size. The lazy graph is never evaluated.
  auto a = zeros({1 << 30, 2});
  CHECK_THROWS_AS(flatten(a), std::overflow_error);

  // conv_general output > 2^31 elements with each per-dim < 2^31.
  // Total elements 524290 * 64 * 64 = 2,147,491,840.
  int n = static_cast<int>((int64_t{1} << 31) / (64 * 64) + 2);
  auto x = ones({n, 8, 8, 1}, float16);
  auto w = ones({1, 1, 1, 1}, float16);
  auto y = conv_general(
      /* input = */ x,
      /* weight = */ w,
      /* stride = */ {1, 1},
      /* padding_lo = */ {0, 0},
      /* padding_hi = */ {0, 0},
      /* kernel_dilation = */ {1, 1},
      /* input_dilation = */ {9, 9},
      /* groups = */ 1,
      /* flip = */ false);
  CHECK_EQ(y.shape(), Shape{n, 64, 64, 1});

  // reshape with inferred dim that won't fit in ShapeElem — issue #3327.
  CHECK_THROWS_AS(reshape(y, {-1}), std::overflow_error);

  // take(a, idx) routes through an internal flatten — overflows on flatten.
  auto idx = array({0u}, uint32);
  CHECK_THROWS_AS(take(y, idx), std::overflow_error);
}

TEST_CASE("test memory info") {
  // Test cache limits
  {
    auto old_limit = set_cache_limit(0);
    {
      auto a = zeros({4096});
      eval(a);
    }
    CHECK_EQ(get_cache_memory(), 0);
    CHECK_EQ(set_cache_limit(old_limit), 0);
    CHECK_EQ(set_cache_limit(old_limit), old_limit);
  }

  // Test memory limits
  {
    auto old_limit = set_memory_limit(10);
    CHECK_EQ(set_memory_limit(old_limit), 10);
    CHECK_EQ(set_memory_limit(old_limit), old_limit);
  }

  // Query active and peak memory
  {
    auto a = zeros({4096});
    eval(a);
    synchronize();
    auto active_mem = get_active_memory();
    CHECK(active_mem >= 4096 * 4);
    {
      auto b = zeros({4096});
      eval(b);
    }
    synchronize();
    auto new_active_mem = get_active_memory();
    CHECK_EQ(new_active_mem, active_mem);
    auto peak_mem = get_peak_memory();
    CHECK(peak_mem >= 4096 * 8);

    auto cache_mem = get_cache_memory();
    CHECK(cache_mem >= 4096 * 4);
  }

  clear_cache();
  CHECK_EQ(get_cache_memory(), 0);
}

TEST_CASE("test scatter_prod with NaN does not hang") {
  // Regression test for the NaN CAS workaround in atomic.h. The std::async
  // timeout converts a wedged GPU into a test failure.
  auto run_with_timeout = [](auto fn) {
    auto fut = std::async(std::launch::async, std::move(fn));
    REQUIRE(
        fut.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    return fut.get();
  };

  float nan = std::nanf("");

  // NaN-valued update colliding with a normal update on the same slot.
  {
    auto out = run_with_timeout([&] {
      auto x = array({1.0f, 1.0f, 1.0f, 1.0f});
      auto idx = array({0, 0});
      auto upd = array({nan, 2.0f}, {2, 1});
      auto y = scatter_prod(x, idx, upd, 0, Device::gpu);
      eval(y);
      return y;
    });
    CHECK(std::isnan(out.data<float>()[0]));
  }

  // NaN already in memory before any update lands.
  {
    auto out = run_with_timeout([&] {
      auto x = array({nan, 1.0f, 1.0f, 1.0f});
      auto idx = array({0, 0});
      auto upd = array({2.0f, 3.0f}, {2, 1});
      auto y = scatter_prod(x, idx, upd, 0, Device::gpu);
      eval(y);
      return y;
    });
    CHECK(std::isnan(out.data<float>()[0]));
  }
}

TEST_CASE("test gpu depthwise conv2d non-mod-8 spatial") {
  // Depthwise Metal kernel is gated on C_per_group == 1, C == O, C % 16 == 0,
  // kernel <= 7, stride <= 2. Previously the dispatch also required the
  // output spatial dims to be multiples of 8; non-multiples fell back to the
  // gemm path. These cases exercise the tail-tile dispatch now handled by
  // the depthwise kernel itself and verify GPU ≈ CPU.
  struct Case {
    int N, H, W, C, kH, kW;
    std::pair<int, int> stride;
    std::pair<int, int> padding;
  };
  std::vector<Case> cases = {
      // Matches the issue reproducer resolutions, reduced batch.
      {1, 15, 15, 16, 3, 3, {1, 1}, {1, 1}},
      {1, 30, 30, 32, 3, 3, {1, 1}, {1, 1}},
      {1, 60, 60, 16, 3, 3, {1, 1}, {1, 1}},
      // Non-square and non-matching alignment on the two spatial axes.
      {1, 17, 30, 16, 5, 5, {1, 1}, {2, 2}},
      {1, 8, 13, 16, 3, 3, {1, 1}, {1, 1}},
      // Stride 2 with non-mod-8 output.
      {1, 19, 19, 32, 3, 3, {2, 2}, {1, 1}},
  };

  auto key = random::key(42);
  for (const auto& c : cases) {
    auto in = random::normal(
        {c.N, c.H, c.W, c.C}, float32, 0.0f, 1.0f, key, Device::cpu);
    auto wt = random::normal(
        {c.C, c.kH, c.kW, 1}, float32, 0.0f, 1.0f, key, Device::cpu);
    eval(in);
    eval(wt);

    auto out_cpu =
        conv2d(in, wt, c.stride, c.padding, {1, 1}, c.C, Device::cpu);
    auto out_gpu =
        conv2d(in, wt, c.stride, c.padding, {1, 1}, c.C, Device::gpu);
    CHECK(allclose(out_cpu, out_gpu, 1e-4, 1e-4).item<bool>());
  }
}

TEST_CASE("test layer norm vjp bias grad race") {
  // Regression test for a write-after-read (WAR) hazard in
  // LayerNormVJP::eval_gpu (mlx/backend/metal/normalization.cpp).
  //
  // The bias-gradient reduction reads the cotangent `g` and is dispatched
  // before the main vjp kernel. When the cotangent is donatable the kernel
  // overwrites `g`'s buffer in place (gx / gw_temp alias g), a WAR hazard.
  // The Metal command encoder uses concurrent dispatch and only auto-inserts
  // barriers for read-after-write, so without an explicit barrier the reduction
  // races the kernel and the bias gradient is intermittently wrong (error on
  // the order of the value magnitude). The gradients for x and w are
  // unaffected.
  //
  // A batched input makes the cotangent donatable, which is required to trigger
  // the aliasing. It is a race, so we loop; pre-fix this trips within a couple
  // thousand iterations, post-fix the GPU result matches the CPU reference
  // exactly on every iteration.
  auto x = random::normal({2, 4, 8}, float32, 0.0f, 1.0f, random::key(0));
  auto w = random::normal({8}, float32, 0.0f, 1.0f, random::key(1));
  auto b = random::normal({8}, float32, 0.0f, 1.0f, random::key(2));
  eval(x, w, b);

  auto loss = [](const std::vector<array>& p, Device dev) {
    return mean(fast::layer_norm(p[0], p[1], p[2], 1e-5f, dev), dev);
  };
  auto gb_fn = [&](Device dev) {
    return grad(
        [&loss, dev](const std::vector<array>& p) { return loss(p, dev); },
        std::vector<int>{0, 1, 2});
  };

  // CPU reference for d/db (ground truth). Verify it is deterministic: a
  // CPU-vs-CPU self-check must be exactly zero before trusting it.
  auto gb_ref = gb_fn(Device::cpu)({x, w, b})[2];
  auto gb_ref2 = gb_fn(Device::cpu)({x, w, b})[2];
  eval(gb_ref, gb_ref2);
  CHECK_EQ(max(abs(gb_ref - gb_ref2)).item<float>(), 0.0f);

  auto gpu_grad = gb_fn(Device::gpu);
  float worst = 0.0f;
  for (int i = 0; i < 3000; ++i) {
    auto gb_gpu = gpu_grad({x, w, b})[2];
    float diff = max(abs(gb_gpu - gb_ref), Device::cpu).item<float>();
    worst = std::max(worst, diff);
    if (diff > 1e-5) {
      break; // Fail fast once the race is observed.
    }
  }
  CHECK(worst <= 1e-5);
}


TEST_CASE("test Gemma 4 expert QMM pure route table") {
  using metal::Gemma4ExpertQMMRoute;
  using metal::Gemma4ExpertQMMRouteInput;
  using metal::classify_gemma4_expert_qmm;

  auto gate_up = [](int assignments) {
    Gemma4ExpertQMMRouteInput input;
    input.requested = true;
    input.aot_available = true;
    input.outer_route = true;
    input.affine = true;
    input.transpose = true;
    input.has_bias = true;
    input.indices_uint32 = true;
    input.indices_contiguous = true;
    input.x_bfloat16 = true;
    input.x_contiguous = true;
    input.w_uint32 = true;
    input.w_contiguous = true;
    input.scales_bfloat16 = true;
    input.scales_contiguous = true;
    input.biases_bfloat16 = true;
    input.biases_contiguous = true;
    input.group_size = 64;
    input.bits = 4;
    input.expert_count = 128;
    input.assignments = assignments;
    input.index_count = assignments;
    input.k = 2816;
    input.n = 1408;
    input.x_rank = 3;
    input.x_dim0 = assignments;
    input.x_dim1 = 1;
    input.x_dim2 = 2816;
    input.w_rank = 3;
    input.w_dim0 = 128;
    input.w_dim1 = 1408;
    input.w_dim2 = 352;
    input.scales_rank = 3;
    input.scales_dim0 = 128;
    input.scales_dim1 = 1408;
    input.scales_dim2 = 44;
    input.biases_rank = 3;
    input.biases_dim0 = 128;
    input.biases_dim1 = 1408;
    input.biases_dim2 = 44;
    return input;
  };
  auto down = [&gate_up](int assignments) {
    auto input = gate_up(assignments);
    input.k = 704;
    input.n = 2816;
    input.x_dim2 = 704;
    input.w_dim1 = 2816;
    input.w_dim2 = 88;
    input.scales_dim1 = 2816;
    input.scales_dim2 = 11;
    input.biases_dim1 = 2816;
    input.biases_dim2 = 11;
    return input;
  };

  for (int assignments : {4096, 8192, 16384}) {
    CHECK(
        classify_gemma4_expert_qmm(gate_up(assignments)) ==
        Gemma4ExpertQMMRoute::hit);
    CHECK(
        classify_gemma4_expert_qmm(down(assignments)) ==
        Gemma4ExpertQMMRoute::hit);
  }

  auto exact = gate_up(4096);
  auto check_miss = [&exact](
                        auto mutate, Gemma4ExpertQMMRoute expected) {
    auto input = exact;
    mutate(input);
    CHECK(classify_gemma4_expert_qmm(input) == expected);
  };
  check_miss(
      [](auto& x) { x.requested = false; },
      Gemma4ExpertQMMRoute::not_requested);
  check_miss(
      [](auto& x) { x.nax_available = true; },
      Gemma4ExpertQMMRoute::fallback_nax);
  check_miss(
      [](auto& x) { x.outer_route = false; },
      Gemma4ExpertQMMRoute::fallback_outer_route);
  check_miss(
      [](auto& x) { x.affine = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.transpose = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.has_bias = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.group_size = 32; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.bits = 8; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.indices_uint32 = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.indices_contiguous = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.x_bfloat16 = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.x_contiguous = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.w_uint32 = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.w_contiguous = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.scales_bfloat16 = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.scales_contiguous = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.biases_bfloat16 = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.biases_contiguous = false; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.expert_count = 127; },
      Gemma4ExpertQMMRoute::fallback_topology);
  check_miss(
      [](auto& x) { x.x_rank = 4; },
      Gemma4ExpertQMMRoute::fallback_topology);
  check_miss(
      [](auto& x) { x.w_rank = 2; },
      Gemma4ExpertQMMRoute::fallback_topology);
  check_miss(
      [](auto& x) { x.scales_rank = 2; },
      Gemma4ExpertQMMRoute::fallback_topology);
  check_miss(
      [](auto& x) { x.biases_rank = 2; },
      Gemma4ExpertQMMRoute::fallback_topology);
  check_miss(
      [](auto& x) { x.index_count -= 1; },
      Gemma4ExpertQMMRoute::fallback_topology);
  for (int assignments : {8, 16, 32, 4095, 4097}) {
    check_miss(
        [assignments](auto& x) {
          x.assignments = assignments;
          x.index_count = assignments;
          x.x_dim0 = assignments;
        },
        Gemma4ExpertQMMRoute::fallback_assignment_count);
  }
  check_miss(
      [](auto& x) { x.w_dim2 = 176; },
      Gemma4ExpertQMMRoute::fallback_geometry);
  check_miss(
      [](auto& x) { x.w_dim1 += 1; },
      Gemma4ExpertQMMRoute::fallback_geometry);
  check_miss(
      [](auto& x) {
        x.k += 32;
        x.x_dim2 = x.k;
      },
      Gemma4ExpertQMMRoute::fallback_geometry);
  check_miss(
      [](auto& x) { x.n -= 32; },
      Gemma4ExpertQMMRoute::fallback_geometry);
  check_miss(
      [](auto& x) { x.aot_available = false; },
      Gemma4ExpertQMMRoute::fallback_metallib_unavailable);

  auto nax_without_aot = exact;
  nax_without_aot.nax_available = true;
  nax_without_aot.aot_available = false;
  CHECK(
      classify_gemma4_expert_qmm(nax_without_aot) ==
      Gemma4ExpertQMMRoute::fallback_nax);
}

TEST_CASE("test Qwen 3.6 expert QMM pure route table") {
  using metal::Gemma4ExpertQMMRoute;
  using metal::Gemma4ExpertQMMRouteInput;
  using metal::classify_gemma4_expert_qmm;

  // Base input: Qwen 3.5/3.6 35B-A3B expert projection at W4/g64,
  // parametrized by whole-projection [E=256, n, k].
  auto qwen = [](int assignments, int k, int n) {
    Gemma4ExpertQMMRouteInput input;
    input.requested = true;
    input.aot_available = true;
    input.outer_route = true;
    input.affine = true;
    input.transpose = true;
    input.has_bias = true;
    input.indices_uint32 = true;
    input.indices_contiguous = true;
    input.x_bfloat16 = true;
    input.x_contiguous = true;
    input.w_uint32 = true;
    input.w_contiguous = true;
    input.scales_bfloat16 = true;
    input.scales_contiguous = true;
    input.biases_bfloat16 = true;
    input.biases_contiguous = true;
    input.group_size = 64;
    input.bits = 4;
    input.expert_count = 256;
    input.assignments = assignments;
    input.index_count = assignments;
    input.k = k;
    input.n = n;
    input.x_rank = 3;
    input.x_dim0 = assignments;
    input.x_dim1 = 1;
    input.x_dim2 = k;
    input.w_rank = 3;
    input.w_dim0 = 256;
    input.w_dim1 = n;
    input.w_dim2 = k / 8;
    input.scales_rank = 3;
    input.scales_dim0 = 256;
    input.scales_dim1 = n;
    input.scales_dim2 = k / 64;
    input.biases_rank = 3;
    input.biases_dim0 = 256;
    input.biases_dim1 = n;
    input.biases_dim2 = k / 64;
    return input;
  };

  // Fused gate_up, split gate/up, and down projections hit at the chunked
  // prefill assignment counts (T x top-8 for T in {512, 1024, 2048}).
  for (int assignments : {4096, 8192, 16384}) {
    CHECK(
        classify_gemma4_expert_qmm(qwen(assignments, 2048, 1024)) ==
        Gemma4ExpertQMMRoute::hit);
    CHECK(
        classify_gemma4_expert_qmm(qwen(assignments, 2048, 512)) ==
        Gemma4ExpertQMMRoute::hit);
    CHECK(
        classify_gemma4_expert_qmm(qwen(assignments, 512, 2048)) ==
        Gemma4ExpertQMMRoute::hit);
  }

  auto exact = qwen(4096, 2048, 1024);
  auto check_miss = [&exact](
                        auto mutate, Gemma4ExpertQMMRoute expected) {
    auto input = exact;
    mutate(input);
    CHECK(classify_gemma4_expert_qmm(input) == expected);
  };
  // Expert counts other than the two instantiated builders miss on topology.
  check_miss(
      [](auto& x) {
        x.expert_count = 255;
        x.w_dim0 = 255;
        x.scales_dim0 = 255;
        x.biases_dim0 = 255;
      },
      Gemma4ExpertQMMRoute::fallback_topology);
  // E=256 with Gemma geometry (and vice versa) must miss on geometry: the
  // shape table is tied to the expert count, never mixed.
  check_miss(
      [](auto& x) {
        x.k = 2816;
        x.n = 1408;
        x.x_dim2 = 2816;
        x.w_dim1 = 1408;
        x.w_dim2 = 352;
        x.scales_dim1 = 1408;
        x.scales_dim2 = 44;
        x.biases_dim1 = 1408;
        x.biases_dim2 = 44;
      },
      Gemma4ExpertQMMRoute::fallback_geometry);
  check_miss(
      [](auto& x) { x.w_dim2 = 128; },
      Gemma4ExpertQMMRoute::fallback_geometry);
  check_miss(
      [](auto& x) { x.n -= 32; },
      Gemma4ExpertQMMRoute::fallback_geometry);
  // T=128 chunks (1024 assignments) intentionally stay on the legacy path.
  for (int assignments : {8, 1024, 4095, 4097}) {
    check_miss(
        [assignments](auto& x) {
          x.assignments = assignments;
          x.index_count = assignments;
          x.x_dim0 = assignments;
        },
        Gemma4ExpertQMMRoute::fallback_assignment_count);
  }
  check_miss(
      [](auto& x) { x.bits = 8; },
      Gemma4ExpertQMMRoute::fallback_quantization);
  check_miss(
      [](auto& x) { x.aot_available = false; },
      Gemma4ExpertQMMRoute::fallback_metallib_unavailable);

  // The Gemma table must also reject Qwen geometry under E=128.
  auto gemma_with_qwen_geometry = exact;
  gemma_with_qwen_geometry.expert_count = 128;
  gemma_with_qwen_geometry.w_dim0 = 128;
  gemma_with_qwen_geometry.scales_dim0 = 128;
  gemma_with_qwen_geometry.biases_dim0 = 128;
  CHECK(
      classify_gemma4_expert_qmm(gemma_with_qwen_geometry) ==
      Gemma4ExpertQMMRoute::fallback_geometry);
}

TEST_CASE("test Gemma 4 expert QMM counter invariant") {
  metal::Gemma4ExpertQMMCounters counters;
  using Route = metal::Gemma4ExpertQMMRoute;
  counters.record(Route::not_requested);
  counters.record(Route::hit);
  counters.record(Route::fallback_nax);
  counters.record(Route::fallback_outer_route);
  counters.record(Route::fallback_quantization);
  counters.record(Route::fallback_topology);
  counters.record(Route::fallback_assignment_count);
  counters.record(Route::fallback_geometry);
  counters.record(Route::fallback_metallib_unavailable);
  counters.record(Route::fallback_sortedness_retracted);

  auto snapshot = counters.snapshot();
  CHECK(snapshot.hits == 1);
  CHECK(snapshot.fallback_nax == 1);
  CHECK(snapshot.fallback_outer_route == 1);
  CHECK(snapshot.fallback_quantization == 1);
  CHECK(snapshot.fallback_topology == 1);
  CHECK(snapshot.fallback_assignment_count == 1);
  CHECK(snapshot.fallback_geometry == 1);
  CHECK(snapshot.fallback_metallib_unavailable == 1);
  CHECK(snapshot.fallback_sortedness_retracted == 1);
  CHECK(snapshot.attempts() == 9);

  counters.reset();
  snapshot = counters.snapshot();
  CHECK(snapshot.attempts() == 0);
  CHECK(snapshot.attempts() == snapshot.hits + snapshot.fallback_nax +
          snapshot.fallback_outer_route + snapshot.fallback_quantization +
          snapshot.fallback_topology +
          snapshot.fallback_assignment_count + snapshot.fallback_geometry +
          snapshot.fallback_metallib_unavailable +
          snapshot.fallback_sortedness_retracted);
}

TEST_CASE("test Gemma 4 expert QMM arm disarm cycle") {
  metal::Gemma4ExpertQMMCounters counters;
  using Route = metal::Gemma4ExpertQMMRoute;

  // Counters start disarmed with an empty interval.
  CHECK(!counters.armed());

  // Arm: the interval opens with zeroed counters.
  counters.clear_and_arm();
  CHECK(counters.armed());
  CHECK(counters.snapshot().attempts() == 0);

  // Record across the measured interval, including the retract class the
  // sortedness fail-safe attributes mis-sorted indices to.
  counters.record(Route::hit);
  counters.record(Route::fallback_sortedness_retracted);
  counters.record(Route::fallback_metallib_unavailable);

  // Disarm snapshots the interval and reports the previous armed state.
  auto interval = counters.snapshot_and_disarm();
  CHECK(interval.armed);
  CHECK(!counters.armed());

  // The attempts == hits + sum(fallback classes) invariant holds across the
  // cycle, with the sortedness-retract class included in the sum.
  CHECK(interval.attempts() == 3);
  CHECK(interval.hits == 1);
  CHECK(interval.fallback_sortedness_retracted == 1);
  CHECK(interval.fallback_metallib_unavailable == 1);
  CHECK(interval.attempts() == interval.hits + interval.fallback_nax +
          interval.fallback_outer_route + interval.fallback_quantization +
          interval.fallback_topology + interval.fallback_assignment_count +
          interval.fallback_geometry + interval.fallback_metallib_unavailable +
          interval.fallback_sortedness_retracted);

  // The snapshot stays readable while disarmed.
  CHECK(!counters.snapshot().armed);
  CHECK(counters.snapshot().attempts() == 3);

  // Re-arming clears the interval again, and disarming it reports armed.
  counters.clear_and_arm();
  CHECK(counters.armed());
  auto reopened = counters.snapshot_and_disarm();
  CHECK(reopened.armed);
  CHECK(reopened.attempts() == 0);
  CHECK(!counters.armed());
}