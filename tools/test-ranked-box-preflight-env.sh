#!/usr/bin/env bash
#
# test-ranked-box-preflight-env.sh -- the ranked preflight refuses a box whose
# paired-leg staging is wrong, and names which part is wrong.
#
# WHAT IS UNDER TEST. tools/ranked-box-preflight.sh section 6, the gate the
# paired design added (David ruling 2026-09-08): the serial-control leg runs on
# an organizer-staged REFERENCE tree, and this box's calibration file is the
# health band that leg must land inside. Neither is a thing the job can fetch or
# repair, so the only correct behaviour on a bad stage is a refusal that names
# the fault -- and that is what these cases pin.
#
# HERMETIC. Every case builds a SYNTHETIC box: a copy of the real script beside
# a copy of the real contract and the real goldens, a stub temperature reader, a
# throwaway git repository standing in for the reference tree, and a calibration
# file written per case. Nothing loads a model, spawns an engine, reaches the
# network, or touches the real box. The synthetic contract's
# baseline_reference_commit is rewritten to the synthetic repository's own HEAD,
# because a fixed sha cannot be reproduced in a fresh repository.
#
# Each case asserts the exit status AND that the refusal names the failing
# thing: a gate that refuses for the wrong reason sends an operator to the wrong
# file, which costs a box slot.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

command -v jq >/dev/null 2>&1 || { echo "test-ranked-box-preflight-env.sh: jq is required" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "test-ranked-box-preflight-env.sh: git is required" >&2; exit 1; }

TRACK_ID="$(jq -r '.track_id' "${REPO_ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json")"
BOX_NAME="synthetic-ranked-box"

# --- the synthetic root: the real script, the real contract, the real goldens
ROOT="${WORK}/root"
mkdir -p "${ROOT}/tools" "${ROOT}/fixtures"
cp "${REPO_ROOT}/tools/ranked-box-preflight.sh" "${ROOT}/tools/"
chmod +x "${ROOT}/tools/ranked-box-preflight.sh"
cp "${REPO_ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" "${ROOT}/fixtures/"

# The staged golden pool: the pinned cohort and nothing else (the preflight
# refuses an unpinned *.json beside it, so the staging is pin-driven).
#
# THE REAL TAPES ARE NOT IN ANY TREE. They are organizer material published in
# R2 and staged on the box, so this suite stages a SYNTHETIC pool and RE-PINS
# the copy of the contract in ${ROOT} to it. Only sha256 and bytes move; every
# other field is the shipped fixture's, so the depth envelope and
# baseline_reference_commit stay under test, and so does the whole
# byte-then-sha verification the preflight performs.
#
# A FRESHLY STAMPED TRACK HAS NO POOL AT ALL. Its live_golden is empty, its
# timed_prompt_pool is empty and its hidden correctness golden is still an
# organizer sentinel, because the tapes do not exist yet. There is nothing to
# re-pin in that state, so the block below AUTHORS the cohort instead: the 8
# pool entries this track scores, a live_golden naming one of them, a per-depth
# oracle for every contract-permitted draft depth, and a real digest for the
# hidden golden. It also arms the copy, because every check after section 5
# is unreachable on an unarmed contract (the arm gate keeps its own case
# below, against a second root that is left unarmed). The shipped fixture is
# never written.
GOLDEN_DIR="${WORK}/goldens"
mkdir -p "${GOLDEN_DIR}"
python3 - "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" "${GOLDEN_DIR}" <<'REPINEOF'
import hashlib, json, os, sys

contract_path, golden_dir = sys.argv[1:3]
contract = json.load(open(contract_path, encoding="utf-8"))

# The cohort size the preflight enforces, and the size this track scores.
POOL_SIZE = 8


def stage(r2_path):
    base = r2_path.rsplit("/", 1)[-1]
    path = os.path.join(golden_dir, base)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"synthetic_golden": base}) + "\n")
    data = open(path, "rb").read()
    return hashlib.sha256(data).hexdigest(), len(data)


def entry(name):
    r2_path = "correctness_prompts/%s/%s.golden.json" % (contract["track_id"], name)
    sha256, size = stage(r2_path)
    return {"r2_path": r2_path, "sha256": sha256, "bytes": size}


if not contract.get("timed_prompt_pool"):
    live = "synthetic-live"
    names = [live] + ["synthetic-pool-%d" % n for n in range(1, POOL_SIZE)]
    contract["timed_prompt_pool"] = [entry(name) for name in names]
    contract["live_golden"] = live
    contract["live_golden_speculative"] = {
        "mtp%d" % depth: entry("%s.mtp%d" % (live, depth))
        for depth in contract.get("mtp_head", {}).get("permitted_draft_depths", [])
    }
    # The hidden oracle is pinned by digest only and never staged here, so any
    # real digest over known bytes arms it.
    hidden = json.dumps({"synthetic_hidden_golden": contract["track_id"]}).encode()
    contract["hidden_correctness_golden"] = {
        "sha256": hashlib.sha256(hidden).hexdigest(),
        "bytes": len(hidden),
    }
    contract["official_scoring_enabled"] = True
else:
    for pool_entry in contract["timed_prompt_pool"]:
        pool_entry["sha256"], pool_entry["bytes"] = stage(pool_entry["r2_path"])
    for spec_entry in (contract.get("live_golden_speculative") or {}).values():
        spec_entry["sha256"], spec_entry["bytes"] = stage(spec_entry["r2_path"])
with open(contract_path, "w", encoding="utf-8") as fh:
    json.dump(contract, fh, indent=2)
    fh.write("\n")
REPINEOF
compgen -G "${GOLDEN_DIR}/*.json" >/dev/null \
  || { echo "FAIL: could not stage the synthetic golden pool" >&2; exit 1; }

# A temperature reader that moves, so the frozen-sensor guard passes without
# asserting anything about a real sensor.
MACMON="${WORK}/macmon-stub"
cat > "${MACMON}" <<'MACMONEOF'
#!/usr/bin/env bash
counter_file="${MACMON_STUB_COUNTER:-/tmp/macmon-stub-counter}"
n=$(( $( { cat "${counter_file}" 2>/dev/null || echo 0; } ) + 1 ))
printf '%s' "${n}" > "${counter_file}"
printf '{"temp":{"gpu_temp_avg":%s},"gpu_usage":[0,0.1]}\n' "$(( 30 + n % 5 ))"
MACMONEOF
chmod +x "${MACMON}"

# --- the synthetic reference tree ------------------------------------------
# A real (tiny) git repository: the preflight reads its HEAD and the commit's
# date, so it has to be git, not a directory that looks like one.
REF_WS="${WORK}/reference"
mkdir -p "${REF_WS}"
git -C "${REF_WS}" init --quiet
git -C "${REF_WS}" config user.email "test@example.invalid"
git -C "${REF_WS}" config user.name "preflight test"
echo "reference tree" > "${REF_WS}/README.md"
git -C "${REF_WS}" add README.md
git -C "${REF_WS}" commit --quiet --no-gpg-sign -m "reference"
REF_COMMIT="$(git -C "${REF_WS}" rev-parse HEAD)"
REF_DATE="$(git -C "${REF_WS}" show -s --format=%cI HEAD)"

stage_reference_worker() {
  mkdir -p "${REF_WS}/.build/release" "${REF_WS}/weights"
  printf '#!/bin/sh\nexit 0\n' > "${REF_WS}/.build/release/bench-worker"
  chmod +x "${REF_WS}/.build/release/bench-worker"
  printf 'metallib\n' > "${REF_WS}/.build/release/mlx.metallib"
  printf 'mlxfast-metallib-fingerprint-v1 deadbeef\n' > "${REF_WS}/.build/release/mlx.metallib.fingerprint"
  # The control leg runs on the pinned commit end to end, weights included.
  printf '{}\n' > "${REF_WS}/weights/config.json"
}
stage_reference_worker

# The synthetic contract pins the synthetic tree.
jq --arg c "${REF_COMMIT}" '.baseline_reference_commit = $c' \
  "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" > "${WORK}/contract.tmp"
mv "${WORK}/contract.tmp" "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json"

# --- the calibration file ---------------------------------------------------
# A healthy file, and a mutator so each case states its ONE difference.
CAPTURED_AT="$(python3 -c "
import datetime, sys
ref = datetime.datetime.fromisoformat(sys.argv[1])
print((ref + datetime.timedelta(hours=1)).isoformat())
" "${REF_DATE}")"

write_calibration() {
  # write_calibration <path> [jq filter]
  local path="$1" filter="${2:-.}"
  jq -n \
    --arg track "${TRACK_ID}" \
    --arg box "${BOX_NAME}" \
    --arg commit "${REF_COMMIT}" \
    --arg captured "${CAPTURED_AT}" \
    '{
      version: 1,
      track_id: $track,
      box: $box,
      reference_commit: $commit,
      prompt: "pool-1",
      passes: 4,
      prefill_seconds_per_token_mean: 0.0006282488193359375,
      decode_seconds_per_token_mean: 0.0329116748046875,
      prefill_cv: 0.004,
      decode_cv: 0.002,
      prefill_band_low: 0.95,
      prefill_band_high: 1.05,
      decode_band_low: 0.98,
      decode_band_high: 1.02,
      captured_at: $captured,
      benchd_source_commit: "0123456789abcdef0123456789abcdef01234567"
    }' | jq "${filter}" > "${path}"
}

CALIBRATION="${WORK}/baseline-calibration.json"
write_calibration "${CALIBRATION}"

# run_preflight [ENV=VAL...] -- drives the REAL preflight in the synthetic root
# with a clean environment. Output lands in ${WORK}/out; sets rc.
run_preflight() {
  local extra=("$@")
  env -i \
    PATH="${PATH}" \
    HOME="${HOME}" \
    MACMON_STUB_COUNTER="${WORK}/macmon.counter" \
    MLXFAST_MACMON="${MACMON}" \
    MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
    RUNNER_NAME="${BOX_NAME}" \
    MLXFAST_BASELINE_WORKSPACE="${REF_WS}" \
    MLXFAST_BASELINE_CALIBRATION="${CALIBRATION}" \
    ${extra[@]+"${extra[@]}"} \
    "${ROOT}/tools/ranked-box-preflight.sh" > "${WORK}/out" 2>&1
  rc=$?
}

# expect_refusal <label> <needle> [ENV=VAL...]
expect_refusal() {
  local label="$1" needle="$2"
  shift 2
  run_preflight "$@"
  if [[ "${rc}" -eq 0 ]]; then
    fail "${label}: the preflight PASSED; it must refuse"
    return
  fi
  if ! grep -qi -- "${needle}" "${WORK}/out"; then
    fail "${label}: the refusal does not name '${needle}'; got: $(tail -3 "${WORK}/out" | tr '\n' ' ')"
  fi
}

# --- case 1: a correctly staged box passes ---------------------------------
run_preflight
if [[ "${rc}" -ne 0 ]]; then
  fail "case 1 (healthy box): the preflight refused a correctly staged box: $(tail -5 "${WORK}/out" | tr '\n' ' ')"
elif ! grep -q "reference workspace is a git checkout at baseline_reference_commit" "${WORK}/out"; then
  fail "case 1 (healthy box): the reference-workspace check did not run"
elif ! grep -q "baseline calibration parses and names this track, this box" "${WORK}/out"; then
  fail "case 1 (healthy box): the calibration check did not run against RUNNER_NAME"
elif ! grep -q "reference workspace carries its own transformed weights" "${WORK}/out"; then
  fail "case 1 (healthy box): the reference-weights check did not run"
fi

# --- cases 2-3: the two names are required ---------------------------------
run_preflight_unset() {
  # env -i plus an explicit unset: the variable is genuinely absent.
  local drop="$1"
  env -i \
    PATH="${PATH}" HOME="${HOME}" \
    MACMON_STUB_COUNTER="${WORK}/macmon.counter" \
    MLXFAST_MACMON="${MACMON}" \
    MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
    RUNNER_NAME="${BOX_NAME}" \
    MLXFAST_BASELINE_WORKSPACE="${REF_WS}" \
    MLXFAST_BASELINE_CALIBRATION="${CALIBRATION}" \
    env -u "${drop}" \
    "${ROOT}/tools/ranked-box-preflight.sh" > "${WORK}/out" 2>&1
  rc=$?
}

run_preflight_unset MLXFAST_BASELINE_WORKSPACE
if [[ "${rc}" -eq 0 ]]; then
  fail "case 2 (workspace unset): the preflight passed with no reference tree"
elif ! grep -q "MLXFAST_BASELINE_WORKSPACE is unset" "${WORK}/out"; then
  fail "case 2 (workspace unset): the refusal does not name the variable: $(tail -3 "${WORK}/out" | tr '\n' ' ')"
fi

run_preflight_unset MLXFAST_BASELINE_CALIBRATION
if [[ "${rc}" -eq 0 ]]; then
  fail "case 3 (calibration unset): the preflight passed with no health band"
elif ! grep -q "MLXFAST_BASELINE_CALIBRATION is unset" "${WORK}/out"; then
  fail "case 3 (calibration unset): the refusal does not name the variable: $(tail -3 "${WORK}/out" | tr '\n' ' ')"
fi

# --- case 4: the workspace is not a git checkout ---------------------------
NOT_GIT="${WORK}/not-a-checkout"
mkdir -p "${NOT_GIT}/.build/release"
expect_refusal "case 4 (not a checkout)" "not a git checkout" \
  "MLXFAST_BASELINE_WORKSPACE=${NOT_GIT}"

# --- case 5: the workspace is at another commit ----------------------------
OTHER_WS="${WORK}/other-commit"
mkdir -p "${OTHER_WS}"
git -C "${OTHER_WS}" init --quiet
git -C "${OTHER_WS}" config user.email "test@example.invalid"
git -C "${OTHER_WS}" config user.name "preflight test"
echo "a different tree" > "${OTHER_WS}/README.md"
git -C "${OTHER_WS}" add README.md
git -C "${OTHER_WS}" commit --quiet --no-gpg-sign -m "different"
mkdir -p "${OTHER_WS}/.build/release"
expect_refusal "case 5 (wrong commit)" "baseline_reference_commit" \
  "MLXFAST_BASELINE_WORKSPACE=${OTHER_WS}"

# --- cases 6-8: the staged worker set ---------------------------------------
mv "${REF_WS}/.build/release/bench-worker" "${WORK}/worker.hidden"
expect_refusal "case 6 (no worker)" "no executable worker"
mv "${WORK}/worker.hidden" "${REF_WS}/.build/release/bench-worker"

mv "${REF_WS}/.build/release/mlx.metallib" "${WORK}/metallib.hidden"
expect_refusal "case 7 (no metallib)" "no sibling mlx.metallib"
mv "${WORK}/metallib.hidden" "${REF_WS}/.build/release/mlx.metallib"

mv "${REF_WS}/.build/release/mlx.metallib.fingerprint" "${WORK}/fingerprint.hidden"
expect_refusal "case 8 (no fingerprint sidecar)" "no fingerprint sidecar"
mv "${WORK}/fingerprint.hidden" "${REF_WS}/.build/release/mlx.metallib.fingerprint"

# --- case 8b: the reference tree must carry its own transformed weights -----
# Borrowing the candidate's transform would put a submission on both sides of
# the ratio, so an untransformed reference tree is a refusal, not a fallback.
mv "${REF_WS}/weights/config.json" "${WORK}/weights-config.hidden"
expect_refusal "case 8b (no reference weights)" "no transformed weights"
mv "${WORK}/weights-config.hidden" "${REF_WS}/weights/config.json"

# --- cases 9-16: the calibration file ---------------------------------------
BAD="${WORK}/bad-calibration.json"

printf 'not json at all\n' > "${BAD}"
expect_refusal "case 9 (unparseable)" "does not parse as JSON" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

write_calibration "${BAD}" '.version = 2'
expect_refusal "case 10 (wrong version)" "version is 2" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

write_calibration "${BAD}" '.track_id = "some-other-track-mlx-v9"'
expect_refusal "case 11 (wrong track)" "belongs to another track" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

write_calibration "${BAD}" '.box = "some-other-box"'
expect_refusal "case 12 (wrong box)" "measured on another machine" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

write_calibration "${BAD}" '.reference_commit = "0000000000000000000000000000000000000000"'
expect_refusal "case 13 (wrong reference commit)" "measured a different reference tree" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

write_calibration "${BAD}" '.decode_seconds_per_token_mean = -1'
expect_refusal "case 14 (negative measurement)" "must be positive" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

write_calibration "${BAD}" '.prefill_band_low = 1.02'
expect_refusal "case 15 (band does not straddle 1)" "must straddle" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

write_calibration "${BAD}" '.captured_at = "2001-01-01T00:00:00+00:00"'
expect_refusal "case 16 (captured before the reference commit)" "did not exist yet" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

# --- case 18: an inherited resident socket is refused ------------------------
# The paired run boots ONE resident PER LEG and benchd boots each from that
# leg's own tree. A socket in the runner service environment would attach every
# phase of BOTH legs to one already-loaded resident, so the reference leg would
# run on the candidate's weights -- public run 34230122059, arriving through the
# box instead of through the measure script.
expect_refusal "case 18 (inherited resident socket)" "BENCH_WORKER_RESIDENT_SOCKET is set" \
  "BENCH_WORKER_RESIDENT_SOCKET=${WORK}/inherited.sock"

# A box name is only checked when the runner names itself: a hand run off
# Actions asserts nothing about it rather than inventing a name.
write_calibration "${BAD}" '.box = "some-other-box"'
env -i PATH="${PATH}" HOME="${HOME}" \
  MACMON_STUB_COUNTER="${WORK}/macmon.counter" \
  MLXFAST_MACMON="${MACMON}" \
  MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
  MLXFAST_BASELINE_WORKSPACE="${REF_WS}" \
  MLXFAST_BASELINE_CALIBRATION="${BAD}" \
  "${ROOT}/tools/ranked-box-preflight.sh" > "${WORK}/out" 2>&1
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  fail "case 17 (RUNNER_NAME unset): the preflight refused a foreign box name off Actions, where there is no runner name to compare against: $(tail -3 "${WORK}/out" | tr '\n' ' ')"
elif ! grep -q "RUNNER_NAME unset" "${WORK}/out"; then
  fail "case 17 (RUNNER_NAME unset): the pass line does not record that the box name went unchecked"
fi

# --- case 19: an unarmed contract refuses -----------------------------------
# Section 5 is the arm gate: benchd refuses to seal an official artifact unless
# the contract declares official_scoring_enabled: true, and this gate refuses
# first so an unarmed dispatch dies before setup rather than after the GPU
# window. The root above is armed so the checks past section 5 stay reachable,
# so the gate gets a root of its own.
UNARMED_ROOT="${WORK}/unarmed"
mkdir -p "${UNARMED_ROOT}/tools" "${UNARMED_ROOT}/fixtures"
cp "${ROOT}/tools/ranked-box-preflight.sh" "${UNARMED_ROOT}/tools/"
chmod +x "${UNARMED_ROOT}/tools/ranked-box-preflight.sh"
jq '.official_scoring_enabled = false' \
  "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" \
  > "${UNARMED_ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json"
env -i PATH="${PATH}" HOME="${HOME}" \
  MACMON_STUB_COUNTER="${WORK}/macmon.counter" \
  MLXFAST_MACMON="${MACMON}" \
  MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
  RUNNER_NAME="${BOX_NAME}" \
  MLXFAST_BASELINE_WORKSPACE="${REF_WS}" \
  MLXFAST_BASELINE_CALIBRATION="${CALIBRATION}" \
  "${UNARMED_ROOT}/tools/ranked-box-preflight.sh" > "${WORK}/out" 2>&1
rc=$?
if [[ "${rc}" -eq 0 ]]; then
  fail "case 19 (unarmed contract): the preflight passed a contract that is not armed for official scoring"
elif ! grep -q "official_scoring_enabled" "${WORK}/out"; then
  fail "case 19 (unarmed contract): the refusal does not name the arm field: $(tail -3 "${WORK}/out" | tr '\n' ' ')"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "test-ranked-box-preflight-env.sh: all 20 cases passed"
  exit 0
fi
echo "test-ranked-box-preflight-env.sh: ${failures} case(s) failed" >&2
exit 1
