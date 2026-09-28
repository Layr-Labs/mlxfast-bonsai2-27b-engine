#!/usr/bin/env bash
#
# test-sandboxed-cli.sh -- the transform confinement starts outside the CLI,
# and the CLI refuses an official run that is not confined.
#
# WHAT IS UNDER TEST.
#   tools/seatbelt-profile.py  the one profile generator (the transform
#                              wrapper, the resident and preflight 7e use it)
#   tools/sandboxed-cli.sh     the trusted wrapper that starts the CLI under
#                              sandbox-exec
#   mlxfast-swift              its check that an official run is confined
#
# Part 1 is hermetic and runs on any host: the profile rules, and the
# wrapper's refusals before it starts anything. Part 2 needs macOS
# (/usr/bin/sandbox-exec) and a built CLI (MLXFAST_SWIFT_BIN, default
# .build/release/mlxfast-swift). It runs the real CLI with canary evaluator
# paths in a temporary directory. When either is missing, part 2 is reported
# as not run. No weights, no GPU, no network.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WRAPPER="${REPO_ROOT}/tools/sandboxed-cli.sh"
GENERATOR="${REPO_ROOT}/tools/seatbelt-profile.py"
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'chmod -R u+w "${WORK}" 2>/dev/null; rm -rf "${WORK}"' EXIT

passes=0
failures=0
pass() { passes=$((passes + 1)); echo "ok    $*"; }
fail() { failures=$((failures + 1)); echo "FAIL  $*" >&2; }

# The canary evaluator paths. Every one exists, so a deny is observable.
C="${WORK}/canary"
mkdir -p "${C}/goldens" "${C}/benchd-bin" "${C}/baseline" "${C}/private" "${C}/runner/_work/repo"
printf '{"weight_map":{"a":"model.safetensors"}}\n' > "${C}/goldens/canary.golden.json"
printf 'benchd\n' > "${C}/benchd-bin/benchd"
printf '{}\n' > "${C}/calibration.json"
canary_env=(
  MLXFAST_QWEN38_GOLDEN_DIR="${C}/goldens"
  MLXFAST_BASELINE_WORKSPACE="${C}/baseline"
  MLXFAST_BASELINE_CALIBRATION="${C}/calibration.json"
  BENCHD_BIN_DIR="${C}/benchd-bin"
  MLXFAST_PRIVATE_DIR="${C}/private"
  RUNNER_WORKSPACE="${C}/runner/_work/repo"
)

# --- part 1: the profile generator -------------------------------------------
profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" "${canary_env[@]}" python3 "${GENERATOR}" \
  --exec /bin/bash --tree "${WORK}/tree" --write-subpath "${WORK}/out" --write-prefix "${WORK}/.out." --official)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  fail "generator: exit ${rc} with every evaluator path set"
else
  for path in "${C}/goldens" "${C}/baseline" "${C}/calibration.json" "${C}/benchd-bin" "${C}/private" \
    "${WORK}/home/.cache/mlxfast-engine-build" "${C}/runner/.runner" "${C}/runner/.credentials"; do
    grep -qF "(deny file-read* file-write* (subpath \"${path}\"))" <<< "${profile}" \
      || fail "generator: no read and write deny for ${path}"
  done
  # The generator resolves the program path, and /bin is a symlink on Linux.
  bash_path="$(python3 -c 'import os; print(os.path.realpath("/bin/bash"))')"
  grep -qF "(allow process-exec (literal \"${bash_path}\"))" <<< "${profile}" \
    || fail "generator: the named program is not the one exec allowed"
  grep -qF "(allow file-write* (subpath \"${WORK}/out\"))" <<< "${profile}" \
    || fail "generator: the output tree is not writable"
  grep -qF "(allow file-write* (regex #\"^${WORK//./\\.}/\\.out\\.\"))" <<< "${profile}" \
    || fail "generator: the staging prefix is not an escaped regex"
  [[ "$(tail -n 1 <<< "${profile}")" == "(deny file-read* file-write* "* ]] \
    || fail "generator: the evaluator denies are not the last rules"
  for rule in "(deny network*)" "(deny process-fork)" "(deny process-exec*)" "(deny file-write*)"; do
    grep -qF "${rule}" <<< "${profile}" || fail "generator: no ${rule}"
  done
  pass "generator: denies network, fork, foreign exec and all writes; allows the named tree and prefix; denies every evaluator path last"
fi

out="$(env -i PATH="${PATH}" HOME="${WORK}/home" "${canary_env[@]}" BENCHD_BIN_DIR= \
  python3 "${GENERATOR}" --exec /bin/bash --official 2>&1)"
rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"unset: BENCHD_BIN_DIR"* ]]; then
  pass "generator: an official profile with BENCHD_BIN_DIR unset refuses by name"
else
  fail "generator: an official profile with BENCHD_BIN_DIR unset did not refuse (rc ${rc}): ${out}"
fi

profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" "${canary_env[@]}" python3 "${GENERATOR}" \
  --exec /bin/bash --tree "${C}/baseline/checkout")"
if grep -qF "(subpath \"${C}/baseline\"))" <<< "${profile}"; then
  fail "generator: the reference workspace is denied to a program from inside it"
else
  pass "generator: a program from inside the reference workspace can read that workspace"
fi

# --- part 1: the wrapper refuses before it starts anything ------------------
expect_wrapper_refusal() {
  local label="$1" needle="$2"
  shift 2
  out="$(env -i PATH="${PATH}" HOME="${WORK}/home" MLXFAST_SWIFT_BIN="${WORK}/no-cli" "${WRAPPER}" "$@" 2>&1)"
  rc=$?
  if [[ "${rc}" -eq 2 && "${out}" == *"${needle}"* ]]; then
    pass "wrapper: ${label}"
  else
    fail "wrapper: ${label}: rc ${rc}, wanted 2 and '${needle}'; got: ${out}"
  fi
}
expect_wrapper_refusal "an unknown verb refuses" "unknown verb 'benchmark'" benchmark --weights w
expect_wrapper_refusal "an unknown option refuses" "unknown option '--golden'" transform --reference r --output w --golden g
expect_wrapper_refusal "the --name=value form refuses" "unknown option '--output=w'" transform --reference r --output=w
expect_wrapper_refusal "a repeated option refuses" "--output is given twice" transform --reference r --output a --output b
expect_wrapper_refusal "transform without --output refuses" "transform: --output is required" transform --reference r
expect_wrapper_refusal "verify-transform without --weights refuses" "verify-transform: --weights is required" verify-transform --reference r
expect_wrapper_refusal "an option without a value refuses" "--index has no value" checkpoint-shards --index

# --- part 2: the real CLI under the real Seatbelt ---------------------------
CLI="${MLXFAST_SWIFT_BIN:-${REPO_ROOT}/.build/release/mlxfast-swift}"
if [[ ! -x /usr/bin/sandbox-exec ]]; then
  echo "test-sandboxed-cli.sh: part 2 NOT RUN: /usr/bin/sandbox-exec is not on this host"
elif [[ ! -x "${CLI}" ]]; then
  echo "test-sandboxed-cli.sh: part 2 NOT RUN: no CLI at ${CLI} (set MLXFAST_SWIFT_BIN)"
else
  official=(env -i PATH="${PATH}" HOME="${WORK}/home" TMPDIR="${WORK}/tmp" "${canary_env[@]}"
    MLXFAST_OFFICIAL_BENCHMARK_RUN=1 MLXFAST_SWIFT_BIN="${CLI}")
  mkdir -p "${WORK}/tmp"

  out="$("${official[@]}" "${CLI}" transform --reference "${WORK}/no-reference" --output "${WORK}/direct" 2>&1)"
  if [[ "${out}" == *"transform is not confined"* && ! -e "${WORK}/direct" ]]; then
    pass "cli: an official transform started without the wrapper refuses before it runs"
  else
    fail "cli: an official transform started without the wrapper did not refuse: ${out}"
  fi

  out="$("${official[@]}" "${CLI}" verify-transform --reference "${WORK}/no-reference" --weights "${WORK}/w" 2>&1)"
  if [[ "${out}" == *"verify-transform is not confined"* ]]; then
    pass "cli: an official verify-transform started without the wrapper refuses before it runs"
  else
    fail "cli: an official verify-transform started without the wrapper did not refuse: ${out}"
  fi

  mkdir -p "${WORK}/read-only-bin"
  cp "${CLI}" "${WORK}/read-only-bin/mlxfast-swift"
  chmod a-w "${WORK}/read-only-bin"
  out="$("${official[@]}" "${WORK}/read-only-bin/mlxfast-swift" transform --reference "${WORK}/no-reference" --output "${WORK}/eacces" 2>&1)"
  if [[ "${out}" == *"failed with EACCES"*"not EPERM"* ]]; then
    pass "cli: a denied write that fails with EACCES, not EPERM, is refused and the errno is named"
  else
    fail "cli: a non-EPERM failure of the denied write was not refused by name: ${out}"
  fi

  chmod a-w "${WORK}/tmp"
  out="$("${official[@]}" "${CLI}" transform --reference "${WORK}/no-reference" --output "${WORK}/control" 2>&1)"
  chmod u+w "${WORK}/tmp"
  if [[ "${out}" == *"the positive control could not create"*"EACCES"* ]]; then
    pass "cli: when the positive control cannot write TMPDIR, the run refuses"
  else
    fail "cli: a failed positive control did not refuse by name: ${out}"
  fi

  out="$("${official[@]}" "${WRAPPER}" transform --reference "${WORK}/no-reference" --output "${WORK}/wrapped" 2>&1)"
  if [[ "${out}" == *"no config.json found under ${WORK}/no-reference"* && "${out}" != *"confine"* ]]; then
    pass "wrapper: an official transform under the wrapper passes the confinement check (EPERM outside, TMPDIR writable)"
  else
    fail "wrapper: an official transform under the wrapper did not pass the confinement check: ${out}"
  fi

  cp "${C}/goldens/canary.golden.json" "${WORK}/control-index.json"
  out="$("${official[@]}" "${WRAPPER}" checkpoint-shards --index "${WORK}/control-index.json" 2>/dev/null)"
  if [[ "${out}" == "model.safetensors" ]]; then
    pass "wrapper: the CLI reads an index outside the evaluator paths"
  else
    fail "wrapper: the CLI could not read the control index: ${out}"
  fi
  out="$("${official[@]}" "${WRAPPER}" checkpoint-shards --index "${C}/goldens/canary.golden.json" 2>&1)"
  if [[ $? -ne 0 && "${out}" == *"Operation not permitted"* ]]; then
    pass "wrapper: the CLI cannot read the same bytes in the golden directory (EPERM)"
  else
    fail "wrapper: the CLI read a canary golden: ${out}"
  fi

  before="$(find "${C}/benchd-bin" -mindepth 1 | sort)"
  "${official[@]}" "${WRAPPER}" transform --reference "${REPO_ROOT}" --output "${C}/benchd-bin/weights" > /dev/null 2>&1
  if [[ "$(find "${C}/benchd-bin" -mindepth 1 | sort)" == "${before}" ]]; then
    pass "wrapper: a transform whose --output is inside benchd-bin cannot write there"
  else
    fail "wrapper: a transform wrote into benchd-bin: $(find "${C}/benchd-bin" -mindepth 1 | tr '\n' ' ')"
  fi
fi

echo "test-sandboxed-cli.sh: ${passes} passed, ${failures} failed"
[[ "${failures}" -eq 0 ]]
