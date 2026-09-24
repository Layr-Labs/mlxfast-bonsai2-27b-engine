#!/usr/bin/env bash
# test-build-cache.sh -- prove tools/build-cache.sh without a toolchain build.
#   1. the key is stable across two calls and changes when a tracked source changes
#   2. restore misses on an empty cache; save then restore hits and restores byte-identical artefacts
#   3. a corrupted cached artefact is a miss, and nothing is restored
# Hermetic: runs in a scratch git clone of this repository with stub artefacts.
set -uo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT
fails=0; pass() { printf 'test-build-cache: PASS -- %s\n' "$*"; }; fail() { printf 'test-build-cache: FAIL -- %s\n' "$*" >&2; fails=$((fails + 1)); }
git clone -q --no-hardlinks "${ROOT_DIR}" "${WORK}/repo" || { fail "cannot clone"; exit 1; }
cd "${WORK}/repo"
export MLXFAST_BUILD_CACHE_DIR="${WORK}/cache"
k1="$(tools/build-cache.sh key)"; k2="$(tools/build-cache.sh key)"
[[ -n "${k1}" && "${k1}" == "${k2}" ]] && pass "the key is stable (${k1:0:12})" || fail "unstable key: ${k1} vs ${k2}"
printf '\n// touched\n' >> Package.swift; k3="$(tools/build-cache.sh key)"; git checkout -q -- Package.swift
[[ "${k3}" != "${k1}" ]] && pass "a tracked source change changes the key" || fail "the key ignored a source change"
out="$(tools/build-cache.sh restore 2>&1)"; rc=$?
[[ "${rc}" -eq 1 && "${out}" == *"miss"* ]] && pass "an empty cache is a miss" || fail "empty cache did not miss (rc ${rc}): ${out}"
mkdir -p .build-worker/release .build/release
printf 'worker' > .build-worker/release/bench-worker; printf 'lib' > .build-worker/release/mlx.metallib; printf 'mlxfast-metallib-fingerprint-v1 abc\n' > .build-worker/release/mlx.metallib.fingerprint; printf 'cli' > .build/release/mlxfast-swift
tools/build-cache.sh save >/dev/null || fail "save failed"
rm -rf .build-worker .build
out="$(tools/build-cache.sh restore 2>&1)"; rc=$?
if [[ "${rc}" -eq 0 && "$(cat .build-worker/release/bench-worker)" == "worker" && "$(cat .build/release/mlxfast-swift)" == "cli" && -f .build-worker/release/mlx.metallib.fingerprint ]]; then pass "save then restore hits and restores the artefacts"; else fail "restore did not reproduce the artefacts (rc ${rc}): ${out}"; fi
printf 'tampered' > "${MLXFAST_BUILD_CACHE_DIR}/${k1}/.build-worker/release/bench-worker"; rm -rf .build-worker .build
out="$(tools/build-cache.sh restore 2>&1)"; rc=$?
[[ "${rc}" -eq 1 && "${out}" == *"does not match"* && ! -e .build/release/mlxfast-swift ]] && pass "a corrupted cached artefact is a miss and nothing is restored" || fail "corruption not refused (rc ${rc}): ${out}"
kA="$(tools/build-cache.sh key)"; cp -R "${WORK}/repo" "${WORK}/repo2"; kB="$(cd "${WORK}/repo2" && tools/build-cache.sh key)"
[[ -n "${kA}" && "${kA}" != "${kB}" ]] && pass "the same tree under another root keys differently (products embed their build path)" || fail "two roots share a key: ${kA} ${kB}"
[[ "${fails}" -eq 0 ]] && { printf 'test-build-cache: all cases pass\n'; exit 0; } || { printf 'test-build-cache: %s failed\n' "${fails}" >&2; exit 1; }
