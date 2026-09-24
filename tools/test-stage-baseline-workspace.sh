#!/usr/bin/env bash
#
# test-stage-baseline-workspace.sh -- the reference-tree stager prints the
# documented build and refuses a bad request, off the box.
#
# WHY A DRY-RUN SUITE. tools/stage-baseline-workspace.sh builds Swift and Metal
# and runs the weight transform, so its real path is BOX-ONLY. What can be held
# anywhere is the SHAPE of what it would do: the commands, in order, with the
# pinned commit in them -- and every refusal that happens before the first
# command runs. --dry-run exists for exactly that, and this suite is its
# consumer.
#
# HERMETIC. Nothing clones, builds, transforms, or reaches the network.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${REPO_ROOT}/tools/stage-baseline-workspace.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

command -v jq >/dev/null 2>&1 || { echo "test-stage-baseline-workspace.sh: jq is required" >&2; exit 1; }

REFERENCE_COMMIT="$(jq -r '.baseline_reference_commit' "${REPO_ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json")"
TARGET="${WORK}/reference"

run_stage() {
  local case_name="$1"
  shift
  local extra=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do extra+=("$1"); shift; done
  env -u MLXFAST_BASELINE_SOURCE -u MLXFAST_METALLIB_STAGE \
      -u MLXFAST_REFERENCE_DIR \
    ${extra[@]+"${extra[@]}"} \
    "${SCRIPT}" "$@" > "${WORK}/${case_name}.out" 2>&1
  rc=$?
}

# --- case 1: the dry run prints the documented build, in order --------------
run_stage case1 "${TARGET}" --dry-run
if [[ "${rc}" -ne 0 ]]; then
  fail "case 1: --dry-run exited ${rc}: $(cat "${WORK}/case1.out")"
else
  # The pinned commit reaches both the fetch and the checkout: a stager that
  # cloned a branch tip would stage whatever moved there.
  grep -q "fetch --quiet --no-tags origin ${REFERENCE_COMMIT}" "${WORK}/case1.out" \
    || fail "case 1: the dry run does not fetch the pinned commit ${REFERENCE_COMMIT}"
  grep -q "checkout --quiet --detach ${REFERENCE_COMMIT}" "${WORK}/case1.out" \
    || fail "case 1: the dry run does not check out the pinned commit"
  # The documented build: both swift builds, the metallib, the stager, the
  # tree's own transform.
  for needle in \
    "swift build -c release --force-resolved-versions --product mlxfast-swift" \
    "swift build -c release --force-resolved-versions --scratch-path .build-worker --product bench-worker" \
    "tools/build-mlx-metallib.sh" \
    "tools/stage-bench-worker.sh" \
    "mlxfast-swift transform --reference"
  do
    grep -qF -- "${needle}" "${WORK}/case1.out" \
      || fail "case 1: the dry run does not run '${needle}'"
  done
  # ORDER MATTERS: the metallib is published next to the worker the second
  # build produced, the stager copies the finished set, and only a staged tree
  # can transform. A dry run that printed these out of order would document a
  # build that cannot work.
  order_ok="$(awk '
    /--product bench-worker/ { worker = NR }
    /tools\/build-mlx-metallib.sh/ { metallib = NR }
    /tools\/stage-bench-worker.sh/ { stage = NR }
    /mlxfast-swift transform/ { transform = NR }
    END { print (worker && metallib && stage && transform \
                 && worker < metallib && metallib < stage && stage < transform) ? "ok" : "bad" }
  ' "${WORK}/case1.out")"
  [[ "${order_ok}" == "ok" ]] \
    || fail "case 1: the build steps are out of order (worker, metallib, stage, transform)"
  grep -q "nothing was run" "${WORK}/case1.out" \
    || fail "case 1: the dry run does not say it ran nothing"
  # The engine fork is a VENDORED TREE, so the clone already carries it. A
  # submodule update here would mean the gitlink came back.
  grep -q "submodule update" "${WORK}/case1.out" \
    && fail "case 1: the dry run still resolves a submodule; the engine fork is a vendored tree"
  [[ ! -e "${TARGET}" ]] || fail "case 1: --dry-run created ${TARGET}; it must run nothing"
fi

# --- case 2: --prestaged-metallib swaps the Metal build for the fingerprint check
run_stage case2 "${TARGET}" --prestaged-metallib --dry-run
if [[ "${rc}" -ne 0 ]]; then
  fail "case 2: --prestaged-metallib --dry-run exited ${rc}: $(cat "${WORK}/case2.out")"
else
  grep -q "verify .*mlx.metallib.fingerprint ==" "${WORK}/case2.out" \
    || fail "case 2: the pre-staged path does not verify the fingerprint against the checkout"
  grep -qE '^ *[0-9]+ +env -C .* tools/build-mlx-metallib.sh$' "${WORK}/case2.out" \
    && fail "case 2: the pre-staged path still compiles Metal"
  grep -q "tools/stage-bench-worker.sh" "${WORK}/case2.out" \
    || fail "case 2: the pre-staged path does not stage the finished set"
fi

# --- case 3: no target directory --------------------------------------------
run_stage case3 --dry-run
if [[ "${rc}" -eq 0 ]]; then
  fail "case 3: the stager ran with no target directory"
elif ! grep -q "usage:" "${WORK}/case3.out"; then
  fail "case 3: the refusal does not print usage: $(cat "${WORK}/case3.out")"
fi

# --- case 4: an unknown option ----------------------------------------------
run_stage case4 "${TARGET}" --rebuild-everything --dry-run
if [[ "${rc}" -eq 0 ]]; then
  fail "case 4: the stager accepted an unknown option"
elif ! grep -q "unrecognized option" "${WORK}/case4.out"; then
  fail "case 4: the refusal does not name the unknown option: $(cat "${WORK}/case4.out")"
fi

# --- case 5: a real run needs a clone source --------------------------------
# No credential exists on the box, so the source is a staged mirror or bundle.
run_stage case5 "${TARGET}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 5: the stager ran a real stage with no clone source"
elif ! grep -q "MLXFAST_BASELINE_SOURCE is unset" "${WORK}/case5.out"; then
  fail "case 5: the refusal does not name the variable: $(cat "${WORK}/case5.out")"
fi

# --- case 6: a real run needs the box's reference checkpoint ----------------
# The reference tree transforms its OWN weights; borrowing the candidate's
# would put a submission's transform on both sides of the ratio.
MIRROR="${WORK}/mirror"
mkdir -p "${MIRROR}"
run_stage case6 "MLXFAST_BASELINE_SOURCE=${MIRROR}" "${TARGET}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 6: the stager ran a real stage with no reference checkpoint"
elif ! grep -q "MLXFAST_REFERENCE_DIR is unset" "${WORK}/case6.out"; then
  fail "case 6: the refusal does not name the variable: $(cat "${WORK}/case6.out")"
fi

# --- case 6b: a real stage needs the box's verified MTP head -----------------
HEADLESS="${WORK}/headless-checkpoint"
mkdir -p "${HEADLESS}"
run_stage case6b "MLXFAST_BASELINE_SOURCE=${MIRROR}" "MLXFAST_REFERENCE_DIR=${HEADLESS}" "${TARGET}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 6b: the stager ran a real stage with no MTP head"
elif ! grep -q "MLXFAST_MTP_HEAD_REFERENCE_DIR" "${WORK}/case6b.out"; then
  fail "case 6b: the refusal does not name the variable: $(cat "${WORK}/case6b.out")"
fi

# --- case 6c: a real stage needs the box's verified DFlash 2 drafter ---------
# A DFlash pair loads the drafter on BOTH legs, so a reference tree without it
# would be a different residency from the candidate.
run_stage case6c "MLXFAST_BASELINE_SOURCE=${MIRROR}" "MLXFAST_REFERENCE_DIR=${HEADLESS}" \
  "MLXFAST_MTP_HEAD_REFERENCE_DIR=${HEADLESS}" "${TARGET}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 6c: the stager ran a real stage with no DFlash 2 drafter"
elif ! grep -q "MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR" "${WORK}/case6c.out"; then
  fail "case 6c: the refusal does not name the variable: $(cat "${WORK}/case6c.out")"
fi

# --- case 7: it never writes into a tree it did not make --------------------
CHECKPOINT="${WORK}/checkpoint"
mkdir -p "${CHECKPOINT}"
OCCUPIED="${WORK}/occupied"
mkdir -p "${OCCUPIED}"
echo "someone else's tree" > "${OCCUPIED}/README.md"
run_stage case7 "MLXFAST_BASELINE_SOURCE=${MIRROR}" "MLXFAST_REFERENCE_DIR=${CHECKPOINT}" "${OCCUPIED}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 7: the stager wrote into a non-empty directory"
elif ! grep -q "not empty" "${WORK}/case7.out"; then
  fail "case 7: the refusal does not name the occupied directory: $(cat "${WORK}/case7.out")"
fi

# --- case 8: --prestaged-metallib without a stage ---------------------------
run_stage case8 "MLXFAST_BASELINE_SOURCE=${MIRROR}" "MLXFAST_REFERENCE_DIR=${CHECKPOINT}" \
  "MLXFAST_MTP_HEAD_REFERENCE_DIR=${CHECKPOINT}" \
  "MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR=${CHECKPOINT}" "${TARGET}" --prestaged-metallib
if [[ "${rc}" -eq 0 ]]; then
  fail "case 8: --prestaged-metallib was accepted with no staged pair"
elif ! grep -q "MLXFAST_METALLIB_STAGE" "${WORK}/case8.out"; then
  fail "case 8: the refusal does not name the variable: $(cat "${WORK}/case8.out")"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "test-stage-baseline-workspace.sh: all 9 cases passed"
  exit 0
fi
echo "test-stage-baseline-workspace.sh: ${failures} case(s) failed" >&2
exit 1
