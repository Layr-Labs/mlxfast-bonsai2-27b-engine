#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
configuration="${CONFIGURATION:-release}"
metal_build="$repo_root/.build/qwen36-a3b-mlx-metal"
mlx_source="$repo_root/.build/checkouts/mlx-swift/Source/Cmlx/mlx"

swift build \
    --package-path "$repo_root" \
    --configuration "$configuration" \
    --product BenchCBv2

cmake \
    -S "$mlx_source" \
    -B "$metal_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_BUILD_METAL=ON \
    -DMLX_BUILD_TESTS=OFF
cmake --build "$metal_build" --target mlx-metallib --parallel

bin_directory="$(swift build \
    --package-path "$repo_root" \
    --configuration "$configuration" \
    --show-bin-path)"
cp \
    "$metal_build/mlx/backend/metal/kernels/mlx.metallib" \
    "$bin_directory/mlx.metallib"

echo "$bin_directory/BenchCBv2"
