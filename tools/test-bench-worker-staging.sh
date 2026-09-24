#!/usr/bin/env bash
# Revert-proof for tools/stage-bench-worker.sh and its wiring into setup.sh.
#
# benchd resolves the scored worker at .build/release/bench-worker and
# loads mlx.metallib from that SAME directory. The worker is built in an isolated
# scratch root (.build-worker/release) with its metallib beside it, so setup.sh
# stages the finished set into .build/release. This suite asserts that staging
# actually lands that set -- binary, its sibling metallib AND the metallib's
# fingerprint sidecar -- at the benchd-resolved path, and that setup.sh invokes
# the staging step on both of its exit paths. It goes RED if the staging fix is
# reverted: the copy stops, the metallib or its sidecar is dropped, or the
# setup.sh call is removed.
#
# The sidecar matters because the harness reads the fingerprint record at
# <staged metallib>.fingerprint. A metallib staged without it fails that check
# with "no fingerprint record", which an official run treats as fatal, so a
# metallib present with no sidecar is a refusal here rather than a skip.
#
# Hermetic: no weights, no GPU, no network, no toolchain, no box. Each case copies
# the REAL tool into a throwaway repo root so its ROOT_DIR resolves there, then
# drives it with stub files under $TMPDIR.
#
# Usage: tools/test-bench-worker-staging.sh [-v]
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
STAGE_TOOL="${REPO_ROOT}/tools/stage-bench-worker.sh"
SETUP_SH="${REPO_ROOT}/setup.sh"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

# Non-vacuity floor: a case deleted or short-circuited leaves the survivors green,
# so exit status alone cannot notice the suite shrinking. Raise this in the same
# commit that adds assertions.
EXPECTED_MIN_ASSERTIONS=14

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rtw-staging.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

PASSED=0
FAILED=0
FAILURES=()

pass() {
  PASSED=$((PASSED + 1))
  [[ "${VERBOSE}" == "1" ]] && echo "ok    $1"
  return 0
}

fail() {
  FAILED=$((FAILED + 1))
  FAILURES+=("$1")
  echo "FAIL  $1" >&2
}

assert_file() {
  if [[ -f "$1" ]]; then pass "$2"; else fail "$2 (missing: $1)"; fi
}

assert_executable() {
  if [[ -x "$1" ]]; then pass "$2"; else fail "$2 (not executable: $1)"; fi
}

assert_absent() {
  if [[ ! -e "$1" ]]; then pass "$2"; else fail "$2 (unexpectedly present: $1)"; fi
}

# Build a throwaway repo root containing a copy of the real staging tool, so its
# ROOT_DIR resolves into the sandbox. Prints the root path.
make_fake_root() {
  local root
  root="$(mktemp -d "${WORK}/root.XXXXXX")"
  mkdir -p "${root}/tools"
  cp "${STAGE_TOOL}" "${root}/tools/stage-bench-worker.sh"
  chmod +x "${root}/tools/stage-bench-worker.sh"
  # The vendored paged-attention Metal source the Nemotron MTP assistant
  # compiles at runtime; staging copies it beside the worker.
  mkdir -p "${root}/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Paged"
  printf '// fake pagedattention.metal\n' \
    > "${root}/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/pagedattention.metal"
  printf '%s\n' "${root}"
}

# Create the SOURCE set the build would have produced under the scratch root:
# the worker, its metallib, and the fingerprint sidecar build-mlx-metallib.sh
# publishes beside every metallib it writes.
FAKE_FINGERPRINT='mlxfast-metallib-fingerprint-v1 0000000000000000000000000000000000000000000000000000000000000000'
seed_source_pair() {
  local root="$1"
  mkdir -p "${root}/.build-worker/release"
  printf '#!/bin/sh\nexit 0\n' > "${root}/.build-worker/release/bench-worker"
  chmod +x "${root}/.build-worker/release/bench-worker"
  printf 'FAKE-METALLIB\n' > "${root}/.build-worker/release/mlx.metallib"
  printf '%s\n' "${FAKE_FINGERPRINT}" > "${root}/.build-worker/release/mlx.metallib.fingerprint"
}

# ---------------------------------------------------------------------------
# Case 1: the happy path stages the pair at the benchd-resolved location.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
seed_source_pair "${root}"
if "${root}/tools/stage-bench-worker.sh" >/dev/null 2>&1; then
  pass "staging exits 0 when the source pair is present"
else
  fail "staging exits 0 when the source pair is present"
fi
staged_bin="${root}/.build/release/bench-worker"
staged_metallib="${root}/.build/release/mlx.metallib"
assert_file "${staged_bin}" "worker staged at benchd path .build/release/bench-worker"
assert_executable "${staged_bin}" "staged worker keeps its execute bit"
# The metallib-adjacency check run-parity.sh enforces: metallib beside the binary.
assert_file "${staged_metallib}" "mlx.metallib staged as the worker's sibling"
assert_file "${root}/.build/release/pagedattention.metal" \
  "pagedattention.metal staged beside the worker (the MTP assistant compiles it at runtime)"
if cmp -s "${root}/.build/release/pagedattention.metal" \
     "${root}/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/pagedattention.metal"; then
  pass "staged pagedattention.metal is the vendored source, byte for byte"
else
  fail "staged pagedattention.metal is the vendored source, byte for byte"
fi
if [[ "$(dirname "${staged_bin}")" == "$(dirname "${staged_metallib}")" ]]; then
  pass "staged worker and mlx.metallib share one directory (run-parity adjacency)"
else
  fail "staged worker and mlx.metallib share one directory (run-parity adjacency)"
fi
# The fingerprint sidecar travels with the metallib: the harness reads the
# record at <staged metallib>.fingerprint, so it must be beside the STAGED one.
staged_fingerprint="${staged_metallib}.fingerprint"
assert_file "${staged_fingerprint}" \
  "mlx.metallib.fingerprint staged beside the staged metallib"
if [[ -f "${staged_fingerprint}" ]] && [[ "$(cat "${staged_fingerprint}")" == "${FAKE_FINGERPRINT}" ]]; then
  pass "the staged fingerprint record carries the source sidecar's bytes"
else
  fail "the staged fingerprint record carries the source sidecar's bytes (got: $(cat "${staged_fingerprint}" 2>/dev/null))"
fi
# The source scratch root is untouched (isolation preserved).
assert_file "${root}/.build-worker/release/bench-worker" \
  "scratch-root worker is left in place (build isolation preserved)"

# ---------------------------------------------------------------------------
# Case 2: fail-closed when the worker was never built.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
if "${root}/tools/stage-bench-worker.sh" >/dev/null 2>&1; then
  fail "staging fails when the worker binary is absent"
else
  pass "staging fails when the worker binary is absent"
fi
assert_absent "${root}/.build/release/bench-worker" \
  "nothing is staged when the worker binary is absent"

# ---------------------------------------------------------------------------
# Case 2b: fail-closed when the paged-attention Metal source is missing from
# the vendored fork, so a worker can never be staged without the resource its
# MTP leg needs.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
seed_source_pair "${root}"
rm -f "${root}/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/pagedattention.metal"
if "${root}/tools/stage-bench-worker.sh" >/dev/null 2>&1; then
  fail "staging fails when pagedattention.metal is missing from the vendored fork"
else
  pass "staging fails when pagedattention.metal is missing from the vendored fork"
fi

# ---------------------------------------------------------------------------
# Case 3: fail-closed when the metallib is missing (and skip is not set), so a
# worker can never be staged without its metallib.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
mkdir -p "${root}/.build-worker/release"
printf '#!/bin/sh\nexit 0\n' > "${root}/.build-worker/release/bench-worker"
chmod +x "${root}/.build-worker/release/bench-worker"
if "${root}/tools/stage-bench-worker.sh" >/dev/null 2>&1; then
  fail "staging fails when mlx.metallib is missing"
else
  pass "staging fails when mlx.metallib is missing"
fi

# ---------------------------------------------------------------------------
# Case 4: fail-closed when the metallib is there but its fingerprint sidecar is
# not. A missing sidecar is indistinguishable at the harness check from a
# tampered one, so staging refuses rather than staging a pair that will fail
# the check on the box.
# ---------------------------------------------------------------------------
root="$(make_fake_root)"
seed_source_pair "${root}"
rm -f "${root}/.build-worker/release/mlx.metallib.fingerprint"
out="$("${root}/tools/stage-bench-worker.sh" 2>&1)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  pass "staging refuses when the metallib fingerprint sidecar is missing"
else
  fail "staging refuses when the metallib fingerprint sidecar is missing (exited 0)"
fi
if [[ "${out}" == *"fingerprint"* ]]; then
  pass "the refusal names the fingerprint sidecar"
else
  fail "the refusal names the fingerprint sidecar (got: ${out})"
fi
assert_absent "${root}/.build/release/mlx.metallib" \
  "no metallib is staged when its fingerprint sidecar is missing"

# ---------------------------------------------------------------------------
# Case 5: setup.sh invokes the staging step on BOTH exit paths, after the
# metallib wait. Catches a revert that drops the wiring while the tool survives.
# ---------------------------------------------------------------------------
# Count bare invocations only: the `()` definition line and the `tools/...` call
# inside the function body do not match this anchored, argument-free pattern.
invocations="$(grep -cE '^[[:space:]]*stage_bench_worker_for_benchd$' "${SETUP_SH}" || true)"
if [[ "${invocations}" -ge 2 ]]; then
  pass "setup.sh invokes stage_bench_worker_for_benchd on both exit paths"
else
  fail "setup.sh invokes stage_bench_worker_for_benchd on both exit paths (found ${invocations})"
fi
if grep -q 'tools/stage-bench-worker.sh' "${SETUP_SH}"; then
  pass "setup.sh's staging function calls tools/stage-bench-worker.sh"
else
  fail "setup.sh's staging function calls tools/stage-bench-worker.sh"
fi

# ---------------------------------------------------------------------------
# Trailer + non-vacuity floor.
# ---------------------------------------------------------------------------
echo "bench-worker-staging: ${PASSED} passed, ${FAILED} failed"
if [[ "${FAILED}" -ne 0 ]]; then
  echo "bench-worker-staging: FAILURES:" >&2
  printf '  - %s\n' "${FAILURES[@]}" >&2
  exit 1
fi
if [[ "${PASSED}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]]; then
  echo "bench-worker-staging: ran only ${PASSED} assertions, expected at least ${EXPECTED_MIN_ASSERTIONS} -- the suite shrank" >&2
  exit 1
fi
exit 0
