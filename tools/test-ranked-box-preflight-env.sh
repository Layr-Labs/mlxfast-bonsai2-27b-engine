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
# The official-run cases keep the evaluator trees in a read-only directory
# directly under /tmp. /tmp has the sticky bit, which check 7a accepts above a
# path that the job account does not own; every other directory above a
# protected path must not be writable, and the directories above ${WORK} are.
RO="$(cd "$(mktemp -d /tmp/ranked-preflight-ro.XXXXXX)" && pwd -P)"
# The official-run cases make read-only trees; make them writable to remove them.
trap 'chmod -R u+w "${WORK}" "${RO}" 2>/dev/null; rm -rf "${WORK}" "${RO}"' EXIT

# The preflight waits between temperature samples (2 s and 5 s). The stub
# reader below moves on every call, so the waits prove nothing here and only
# make each run take 4 s or more. A `sleep` that returns at once
# comes first on PATH for every run.
mkdir -p "${WORK}/fast-sleep"
printf '#!/bin/sh\nexit 0\n' > "${WORK}/fast-sleep/sleep"
chmod +x "${WORK}/fast-sleep/sleep"
export PATH="${WORK}/fast-sleep:${PATH}"

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
# A FRESHLY STAMPED TRACK HAS NO POOL AT ALL. Its live_goldens is empty, its
# timed_prompt_pool is empty and its hidden correctness golden is still an
# organizer sentinel, because the tapes do not exist yet. There is nothing to
# re-pin in that state, so the block below AUTHORS the cohort instead: a
# synthetic pool of POOL_SIZE entries, a live_goldens list naming one of them, a
# per-depth oracle for every contract-permitted draft depth, and a real digest
# for the hidden golden. It also arms the copy, because every check after
# section 5 is unreachable on an unarmed contract (the arm gate keeps its own
# case below, against a second root that is left unarmed). The shipped fixture
# is never written.
GOLDEN_DIR="${WORK}/goldens"
mkdir -p "${GOLDEN_DIR}"
python3 - "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" "${GOLDEN_DIR}" <<'REPINEOF'
import hashlib, json, os, sys

contract_path, golden_dir = sys.argv[1:3]
contract = json.load(open(contract_path, encoding="utf-8"))

# Any size of one or more passes the preflight.
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
    contract["live_goldens"] = [live]
    contract["live_golden_speculative"] = {
        "mtp%d" % depth: {live: entry("%s.mtp%d" % (live, depth))}
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
    for per_prompt in (contract.get("live_golden_speculative") or {}).values():
        for spec_entry in per_prompt.values():
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
    --argjson live "$(jq -c '.live_goldens' "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json")" \
    '{
      version: 2,
      track_id: $track,
      box: $box,
      reference_commit: $commit,
      captured_at: $captured,
      benchd_source_commit: "0123456789abcdef0123456789abcdef01234567",
      prompts: [$live[] | {
        prompt: .,
        passes: 4,
        prefill_seconds_per_token_mean: 0.0006282488193359375,
        decode_seconds_per_token_mean: 0.0329116748046875,
        prefill_cv: 0.004,
        decode_cv: 0.002,
        prefill_band_low: 0.95,
        prefill_band_high: 1.05,
        decode_band_low: 0.98,
        decode_band_high: 1.02
      }]
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

write_calibration "${BAD}" '.version = 3'
expect_refusal "case 10 (wrong version)" "version is 3" \
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

write_calibration "${BAD}" '.prompts[-1].decode_seconds_per_token_mean = -1'
expect_refusal "case 14 (negative measurement)" "must be positive" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

write_calibration "${BAD}" '.prompts[-1].prefill_band_low = 1.02'
expect_refusal "case 15 (band does not straddle 1)" "must straddle" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

write_calibration "${BAD}" 'del(.prompts[-1])'
expect_refusal "case 15b (a live prompt has no entry)" "has no entry for live prompt" \
  "MLXFAST_BASELINE_CALIBRATION=${BAD}"

write_calibration "${BAD}" '.version = 1 | . + .prompts[0] | del(.prompts)'
expect_refusal "case 15c (a version 1 file holds one prompt only)" "has no entry for live prompt" \
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

# --- cases 22-35: the account boundary on an official run -------------------
# GHSA-rc55-jfmg-gvc9, GHSA-2j7x-cjrv-43wv. On a self-hosted runner the
# preflight refuses a box whose job account can change the evaluator material
# or has privilege. The job account is simulated: a stub `id`
# gives the job a uid that owns nothing here (570), and a stub `sudo` fails.
# The evaluator trees are copies under a read-only directory (${RO}, made at
# the top), so the real access test fails for them as it does on the box.
OFF="${WORK}/official"
mkdir -p "${OFF}/stubs" "${OFF}/home" "${OFF}/tmp"
cp -R "${GOLDEN_DIR}" "${RO}/goldens"
cp -R "${REF_WS}" "${RO}/reference"
cp "${CALIBRATION}" "${RO}/baseline-calibration.json"
mkdir -p "${RO}/benchd-bin" "${RO}/reference-checkpoint" "${RO}/bin" "${RO}/metallib-stage"
printf 'benchd\n' > "${RO}/benchd-bin/benchd"
printf '{}\n' > "${RO}/benchd-bin/benchd.manifest.json"
printf '{}\n' > "${RO}/reference-checkpoint/config.json"
cp "${MACMON}" "${RO}/bin/macmon"
printf 'metallib\n' > "${RO}/metallib-stage/mlx.metallib"
# A build tree with more items than the old 64-file sample, and a second
# temperature reader two directories below ${RO}, for cases 33-35.
mkdir -p "${RO}/reference/.build/many" "${RO}/hidden/deep"
for n in $(seq 1 130); do
  printf '%s\n' "${n}" > "${RO}/reference/.build/many/f${n}"
done
cp "${MACMON}" "${RO}/hidden/deep/macmon"
chmod -R a-w "${RO}"
cat > "${OFF}/stubs/id" <<'IDEOF'
#!/bin/sh
case "$1" in
  -u) echo "${STUB_ID_UID}" ;;
  -Gn) echo "${STUB_ID_GROUPS}" ;;
  *) exec /usr/bin/id "$@" ;;
esac
IDEOF
cat > "${OFF}/stubs/sudo" <<'SUDOEOF'
#!/bin/sh
exit "${STUB_SUDO_RC}"
SUDOEOF
chmod +x "${OFF}/stubs/id" "${OFF}/stubs/sudo"

# run_official [ENV=VAL...] -- the REAL preflight as a simulated job account on a
# self-hosted runner. Output lands in ${WORK}/out; sets rc.
run_official() {
  env -i \
    PATH="${OFF}/stubs:${PATH}" \
    HOME="${OFF}/home" \
    TMPDIR="${OFF}/tmp" \
    STUB_ID_UID=570 \
    STUB_ID_GROUPS="bench everyone" \
    STUB_SUDO_RC=1 \
    MACMON_STUB_COUNTER="${WORK}/macmon.counter" \
    MLXFAST_MACMON="${RO}/bin/macmon" \
    MLXFAST_QWEN38_GOLDEN_DIR="${RO}/goldens" \
    MLXFAST_BASELINE_WORKSPACE="${RO}/reference" \
    MLXFAST_BASELINE_CALIBRATION="${RO}/baseline-calibration.json" \
    BENCHD_BIN_DIR="${RO}/benchd-bin" \
    MLXFAST_REFERENCE_DIR="${RO}/reference-checkpoint" \
    MLXFAST_METALLIB_STAGE="${RO}/metallib-stage" \
    RUNNER_NAME="${BOX_NAME}" \
    RUNNER_ENVIRONMENT=self-hosted \
    "$@" \
    "${ROOT}/tools/ranked-box-preflight.sh" > "${WORK}/out" 2>&1
  rc=$?
}

# expect_official_refusal <label> <needle> [ENV=VAL...]
expect_official_refusal() {
  local label="$1" needle="$2"
  shift 2
  run_official "$@"
  if [[ "${rc}" -eq 0 ]]; then
    fail "${label}: the preflight PASSED; it must refuse"
  elif ! grep -qF -- "${needle}" "${WORK}/out"; then
    fail "${label}: the refusal does not name '${needle}'; got: $(tail -3 "${WORK}/out" | tr '\n' ' ')"
  fi
}

# Case 22: a converged box passes every boundary check.
run_official
if [[ "${rc}" -ne 0 ]]; then
  fail "case 22 (converged box): the preflight refused: $(tail -3 "${WORK}/out" | tr '\n' ' ')"
else
  for check in 7a 7b; do
    grep -q "ok    account boundary ${check}:" "${WORK}/out" \
      || fail "case 22 (converged box): no pass line for account boundary ${check}"
  done
  grep -Eq "ok    account boundary 7a: .*checked [0-9]+ items in 7 protected trees .* and [0-9]+ directories above them up to /" "${WORK}/out" \
    || fail "case 22 (converged box): the 7a pass line does not give what it checked: $(grep 'account boundary 7a' "${WORK}/out")"
fi

# Case 24: MLXFAST_OFFICIAL_BENCHMARK_RUN alone marks the run as official.
expect_official_refusal "case 24 (official flag, job in admin)" "account boundary check 7b: the job account is in the admin group" \
  RUNNER_ENVIRONMENT= MLXFAST_OFFICIAL_BENCHMARK_RUN=1 STUB_ID_GROUPS="bench admin"

# Case 25: the job account owns the evaluator material (the operator runner).
expect_official_refusal "case 25 (job owns the goldens)" "account boundary check 7a: the job account (uid $(/usr/bin/id -u)) owns ${RO}/goldens" \
  STUB_ID_UID="$(/usr/bin/id -u)"

# Case 26: one file in benchd-bin is writable.
chmod u+w "${RO}/benchd-bin/benchd"
expect_official_refusal "case 26 (writable benchd)" "account boundary check 7a: the job account can write ${RO}/benchd-bin/benchd"
chmod a-w "${RO}/benchd-bin/benchd"

# Case 27: the parent directory of the evaluator trees is writable.
chmod u+w "${RO}"
expect_official_refusal "case 27 (writable parent)" "account boundary check 7a: the job account can write ${RO}, a directory above the golden directory"
chmod a-w "${RO}"

# Cases 28-30: privilege.
expect_official_refusal "case 28 (job in admin)" "account boundary check 7b: the job account is in the admin group" \
  STUB_ID_GROUPS="bench admin"
expect_official_refusal "case 29 (sudo works)" "account boundary check 7b: sudo -n true succeeds" \
  STUB_SUDO_RC=0
expect_official_refusal "case 30 (root)" "account boundary check 7b: the job runs as root" \
  STUB_ID_UID=0

# Case 33: one writable file deep in a large tree. The check is not a sample:
# the last file that find lists is writable, and the refusal names it.
deep_file="$(find "${RO}/reference/.build/many" -type f | tail -n 1)"
chmod u+w "${deep_file}"
expect_official_refusal "case 33 (one writable file deep in a large tree)" "account boundary check 7a: the job account can write ${deep_file}, which is part of the reference workspace"
chmod a-w "${deep_file}"

# Case 34: a file whose mode is read-only but whose ACL lets the job write it.
# The access test applies the ACL. The case runs on macOS only (`chmod +a`). On
# Linux an ACL entry for the owner of a file has no effect, because the mode of
# the owner applies, and this suite has one account. There the case is reported
# as not run.
acl_file="$(find "${RO}/goldens" -type f -name '*.json' | head -n 1)"
acl_set=0
chmod u+w "${RO}/goldens"
if [[ "$(uname -s)" == "Darwin" ]] && chmod +a "user:$(/usr/bin/id -un) allow write" "${acl_file}" 2>/dev/null; then
  acl_set=1
fi
chmod a-w "${RO}/goldens"
if [[ "${acl_set}" == "1" ]]; then
  [[ -z "$(find "${acl_file}" -perm -u+w)" ]] \
    || fail "case 34 (write by ACL only): the fixture file ${acl_file} has the owner write bit in its mode"
  expect_official_refusal "case 34 (write by ACL only)" "account boundary check 7a: the job account can write ${acl_file}, which is part of the golden directory"
  chmod u+w "${RO}/goldens"
  chmod -N "${acl_file}"
  chmod a-w "${RO}/goldens"
else
  echo "test-ranked-box-preflight-env.sh: case 34 (write by ACL only) NOT RUN: this host is not macOS"
fi

# Case 35: a writable directory two levels above a protected path. The parent
# of the temperature reader is read-only; the directory above it is not.
chmod u+w "${RO}/hidden"
expect_official_refusal "case 35 (writable directory above the parent)" "account boundary check 7a: the job account can write ${RO}/hidden, a directory above the temperature reader" \
  MLXFAST_MACMON="${RO}/hidden/deep/macmon"
chmod a-w "${RO}/hidden"

# --- cases 36-37: the live_goldens list ------------------------------------
# Every name in live_goldens must name a pinned pool entry, and the list must
# not be empty. Each case gets a root of its own, so the root above stays
# healthy.
# run_live_goldens_case <label> <needle> <jq filter>
run_live_goldens_case() {
  local label="$1" needle="$2" filter="$3" case_root="${WORK}/$1"
  mkdir -p "${case_root}/tools" "${case_root}/fixtures"
  cp "${ROOT}/tools/ranked-box-preflight.sh" "${case_root}/tools/"
  chmod +x "${case_root}/tools/ranked-box-preflight.sh"
  jq "${filter}" "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" \
    > "${case_root}/fixtures/bonsai2_27b_mlx_v1_track.json"
  env -i PATH="${PATH}" HOME="${HOME}" \
    MACMON_STUB_COUNTER="${WORK}/macmon.counter" \
    MLXFAST_MACMON="${MACMON}" \
    MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
    RUNNER_NAME="${BOX_NAME}" \
    MLXFAST_BASELINE_WORKSPACE="${REF_WS}" \
    MLXFAST_BASELINE_CALIBRATION="${CALIBRATION}" \
    "${case_root}/tools/ranked-box-preflight.sh" > "${WORK}/out" 2>&1
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    fail "${label}: the preflight PASSED; it must refuse"
  elif ! grep -q -- "${needle}" "${WORK}/out"; then
    fail "${label}: the refusal does not name '${needle}'; got: $(tail -3 "${WORK}/out" | tr '\n' ' ')"
  fi
}
run_live_goldens_case case36 "declares no live_goldens" '.live_goldens = []'
run_live_goldens_case case37 "live golden 'no-such-prompt' names no timed_prompt_pool entry" \
  '.live_goldens += ["no-such-prompt"]'

# --- cases 38-39: the pool size is not fixed, and its entries are distinct --
run_live_goldens_case case38 "two entries with the same sha256" \
  '.timed_prompt_pool += [.timed_prompt_pool[0]]'

# A pool with one more entry than the fixture passes. The extra tape is staged
# in a directory of its own, so the directory above stays healthy.
case_root="${WORK}/case39"
mkdir -p "${case_root}/tools" "${case_root}/fixtures" "${case_root}/goldens"
cp "${ROOT}/tools/ranked-box-preflight.sh" "${case_root}/tools/"
chmod +x "${case_root}/tools/ranked-box-preflight.sh"
cp "${GOLDEN_DIR}"/*.json "${case_root}/goldens/"
extra_tape="${case_root}/goldens/synthetic-extra.golden.json"
printf '{"synthetic_golden":"synthetic-extra"}\n' > "${extra_tape}"
jq --arg path "correctness_prompts/${TRACK_ID}/synthetic-extra.golden.json" \
  --arg sha "$(shasum -a 256 "${extra_tape}" | awk '{print $1}')" \
  --argjson bytes "$(wc -c < "${extra_tape}" | tr -d '[:space:]')" \
  '.timed_prompt_pool += [{r2_path: $path, sha256: $sha, bytes: $bytes}]' \
  "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" \
  > "${case_root}/fixtures/bonsai2_27b_mlx_v1_track.json"
pool_size="$(jq '.timed_prompt_pool | length' "${case_root}/fixtures/bonsai2_27b_mlx_v1_track.json")"
env -i PATH="${PATH}" HOME="${HOME}" \
  MACMON_STUB_COUNTER="${WORK}/macmon.counter" \
  MLXFAST_MACMON="${MACMON}" \
  MLXFAST_QWEN38_GOLDEN_DIR="${case_root}/goldens" \
  RUNNER_NAME="${BOX_NAME}" \
  MLXFAST_BASELINE_WORKSPACE="${REF_WS}" \
  MLXFAST_BASELINE_CALIBRATION="${CALIBRATION}" \
  "${case_root}/tools/ranked-box-preflight.sh" > "${WORK}/out" 2>&1
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  fail "case 39 (pool of ${pool_size}): the preflight refused a correctly staged pool: $(tail -3 "${WORK}/out" | tr '\n' ' ')"
elif ! grep -q "timed pool armed: ${pool_size} pinned tapes" "${WORK}/out"; then
  fail "case 39 (pool of ${pool_size}): the pass line does not state the pool size"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "test-ranked-box-preflight-env.sh: all 37 cases passed"
  exit 0
fi
echo "test-ranked-box-preflight-env.sh: ${failures} case(s) failed" >&2
exit 1
