#!/usr/bin/env bash
#
# ranked-box-preflight.sh -- the fail-closed gate the ranked job runs BEFORE
# ./setup.sh and before any measurement, for track bonsai2-27b-mlx-v1.
#
# WHAT IT IS FOR. The ranked job holds NO CREDENTIAL by design (bundles-no-keys:
# a box receives staged bundles, never keys), so the hidden material this track
# measures against cannot be fetched by the job itself -- it is staged onto the
# box out of band by the organizer. That moves the whole question from "can we
# download it" to "is what is on this box the pinned material". This script
# answers that question, and refuses when the answer is anything other than
# "yes, exactly".
#
# Every check below aborts before a score.json could exist. There is no
# degraded mode: the alternative to a verified staged asset is a non-zero exit,
# never a substituted, defaulted, or re-fetched one.
#
# THE STAGING CONVENTION IS THE EXISTING ONE, not a new one. The golden path
# comes from the runner process environment under the name
# tools/bonsai2-27b-measure-and-score.sh already reads:
#
#   MLXFAST_QWEN38_GOLDEN_DIR          directory holding the 8 timed-pool tapes
#                                      (the live golden this leg scores over
#                                      among them). The name is the FLEET's
#                                      golden-dir contract, which every box
#                                      script exports; it is not this track's
#                                      name and it is not renamed here.
#
# On a self-hosted runner it is set in the runner service environment by whoever
# stages the box; a `run:` step inherits it. Nothing here reads a GitHub secret,
# and nothing here reaches the network.
#
# PAIRED, WITH A PER-BOX BAND (David 2026-09-08). This track scores TWO legs on
# the same box in the same job, over the ONE live golden: a SERIAL-CONTROL leg
# on the organizer-staged reference tree and the CANDIDATE leg on the submission
# tree at its declared depth. The score is the LIVE ratio. Nothing stores a
# pair -- not the constants, not the fixture, not the golden. So this preflight
# verifies the staged pool tapes (and the live golden among them), the fixture
# arm state, AND the two names the paired path needs on this box:
#
#   MLXFAST_BASELINE_WORKSPACE     the built reference tree, at the fixture's
#                                  baseline_reference_commit;
#   MLXFAST_BASELINE_CALIBRATION   this box's health band for the control leg.
#
# Section 6 holds both. Nothing here fetches or builds the reference tree: it is
# staged on the box out of band, the same way the goldens are.
#
# WHAT THE PINS ARE. fixtures/bonsai2_27b_mlx_v1_track.json is trusted-side (not an
# editable path), and its timed_prompt_pool[] carries {r2_path, sha256, bytes}
# per tape plus hidden_correctness_golden's {sha256, bytes}. A staged file is
# accepted only when its byte count AND its sha256 equal the contract's -- byte
# count first, because a truncated stage is the common failure and naming it
# precisely is worth one `wc -c` (the same order tools/fetch-goldens.sh and
# tools/fetch-benchd.sh verify in).
#
# Usage:  tools/ranked-box-preflight.sh
# Exit:   0 every check passed
#         1 a check failed (message on stderr names which)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONTRACT="${REPO_ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json"

fail() {
  echo "ranked-box-preflight: REFUSING -- $*" >&2
  exit 1
}

ok() {
  echo "ok    $*"
}

# --- 0. tools ---------------------------------------------------------------
# jq is already a hard requirement of tools/bonsai2-27b-measure-and-score.sh
# (it reads the fixture pins and trackId), so a box that cannot run this cannot
# finish a ranked run either.
command -v jq >/dev/null 2>&1 || fail "jq is required to read the track contract"
command -v shasum >/dev/null 2>&1 || fail "shasum is required to verify staged assets"
[[ -r "${CONTRACT}" ]] || fail "cannot read the track contract at ${CONTRACT}"
jq -e . >/dev/null 2>&1 < "${CONTRACT}" || fail "the track contract is not valid JSON: ${CONTRACT}"
ok "track contract readable and parses: fixtures/bonsai2_27b_mlx_v1_track.json"

# --- 1. the job holds no credential -----------------------------------------
# The workflow references no secret (tools/ci-workflow-egress-scan.sh is the
# static half of that). This is the runtime half: a credential reaching the job
# through the RUNNER's environment would defeat the same invariant without ever
# appearing in the workflow file. R2 keys and a signer are what would let this
# job pull hidden material itself instead of measuring the staged bundle.
for var in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY MLXFAST_QWEN38_R2_DOWNLOADER BENCHD_DIST_TOKEN; do
  eval "value=\${${var}:-}"
  # shellcheck disable=SC2154  # assigned by the eval above
  [[ -z "${value}" ]] || fail "${var} is set in the ranked job's environment; this job must hold no credential (boxes get staged bundles, never keys)"
done
ok "no R2 credential, signer, or dist token in the job environment"

# --- 2. no measurement-weakening override -----------------------------------
# Each of these is a real, local-debugging-only switch in this tree. On a
# ranked box any of them would silently change what is measured or what is
# accepted, so their presence is a refusal rather than a warning.
#
#   BENCHD                          tools/bonsai2-27b-measure-and-score.sh honours a
#                                     caller-supplied benchd WITHOUT the
#                                     channel-manifest hash check (deliberate, for
#                                     benchd development) -- so on a ranked run
#                                     it is a way to measure against unpinned
#                                     scoring code.
#   MLXFAST_SKIP_WEIGHTS_DOWNLOAD /   setup.sh builds tools only and never
#   SKIP_MODEL_DOWNLOAD               obtains or verifies the checkpoint.
#   MLXFAST_LOCAL_COOL_GATE           disables the thermal gate.
#   MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT  publishes a timing estimate past a failed
#                                     public correctness gate.
#   MLXFAST_GPU_TEMP_CMD              displaces macmon as benchd's temperature
#                                     reader with an arbitrary shell command --
#                                     it is the FIRST branch of benchd's reader
#                                     discovery, ahead of MLXFAST_MACMON_BIN, so
#                                     `echo 20` set here would make every cool
#                                     gate pass instantly on a hot GPU.
#   BENCH_WORKER_RESIDENT_SOCKET      names an ALREADY-LOADED resident. The
#                                     paired run boots ONE resident PER LEG, and
#                                     benchd boots each from that leg's own
#                                     tree. A socket inherited from the runner
#                                     service environment would attach every
#                                     phase of BOTH legs to one resident, so the
#                                     reference leg would run on the candidate's
#                                     weights -- the exact failure of public run
#                                     34230122059, arriving through the box
#                                     instead of through the measure script.
for var in BENCHD BENCHCTL MLXFAST_SKIP_WEIGHTS_DOWNLOAD SKIP_MODEL_DOWNLOAD MLXFAST_LOCAL_COOL_GATE MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT MLXFAST_GPU_TEMP_CMD BENCH_WORKER_RESIDENT_SOCKET; do
  eval "value=\${${var}:-}"
  [[ -z "${value}" ]] || fail "${var} is set in the ranked job's environment; it weakens or bypasses what the ranked run measures"
done
ok "no measurement-weakening override in the job environment"

# --- 2b. the GPU temperature reader exists ----------------------------------
# DAVID'S RULING 2026-08-26: no calibration without thermal control, and a
# missing reader REFUSES rather than silently self-disabling.
#
# This is the challenger's host-preflight line, same form, same variable:
#   test -x "${MLXFAST_MACMON}" || { echo "::error::macmon missing"; exit 1; }
# (Layr-Labs/qwen-3.8-mtp-challenge .github/workflows/qwen-mtp-ranked-benchmark.yml
# :1049, with the job-env pin at :218). The PATH differs on purpose: the
# challenger pins /opt/bench-runner/bin/macmon, its box's sudo-gated operator
# tree. This box has no /opt/bench-runner and needs none.
#
# WHY IT MUST BE LOUD HERE. The pinned benchd's own cool gate SKIPS with a
# warning when no reader resolves -- it returns GateState::SkippedNoReader
# (mlxfast-bench crates/benchd/src/coolgate.rs:253) rather than failing. That
# is correct for a participant's laptop and catastrophic for a ranked run: every
# timed leg would proceed ungated and the seal would look normal. Refusing here
# makes that path unreachable on this box.
#
# The default below is the same literal .github/workflows/benchmark.yml pins into
# MLXFAST_MACMON and MLXFAST_MACMON_BIN; it exists so a manual run of this script
# on the box checks the same file the ranked job will use.
MLXFAST_MACMON="${MLXFAST_MACMON:-${MLXFAST_MACMON_BIN:-/opt/homebrew/bin/macmon}}"
test -x "${MLXFAST_MACMON}" \
  || fail "macmon missing at ${MLXFAST_MACMON}: no GPU temperature reader, so the 40C cool-down gate would silently skip on every timed leg (benchd coolgate.rs:253). Install macmon or point MLXFAST_MACMON_BIN at it; this run measures nothing without thermal control"
ok "GPU temperature reader present and executable: ${MLXFAST_MACMON}"

# --- 2c. the reader is not frozen -------------------------------------------
# A reader that ALWAYS returns the same number passes every cool gate instantly,
# including on a hot GPU, and leaves a seal that looks perfect ("waited 0s" on
# every phase). Presence is therefore not enough: the sample has to be plausible
# and it has to be capable of moving. Mirrors the challenger's own implausible-
# reading guard (benchmark.sh:445-454, :871-886) with its <=5C floor.
#
# ORDER, AND WHY IT IS NOT A FLAKE: three quick samples first; a constant reading
# there is common on a genuinely idle box, so it is not by itself a refusal. Only
# if all three agree does it take three more, spread wider. Six identical
# readings across ~20s is a stuck sensor, not an idle one.
read_gpu_temp() {
  "${MLXFAST_MACMON}" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty' | head -1
}

first_temp="$(read_gpu_temp || true)"
[[ -n "${first_temp}" ]] \
  || fail "the temperature reader at ${MLXFAST_MACMON} produced no .temp.gpu_temp_avg sample; a reader that cannot be read is a missing reader"
[[ "$(jq -n --argjson t "${first_temp}" '$t > 5')" == "true" ]] \
  || fail "GPU temperature reads ${first_temp}C, at or below the 5C implausibility floor; the sensor is broken or frozen, and a broken sensor passes every cool gate"

distinct_temp_count() {
  printf '%s\n' "$@" | sort -u | wc -l | tr -d ' '
}

temps=("${first_temp}")
for _ in 1 2; do
  sleep 2
  temps+=("$(read_gpu_temp || true)")
done
if [[ "$(distinct_temp_count "${temps[@]}")" == "1" ]]; then
  for _ in 1 2 3; do
    sleep 5
    temps+=("$(read_gpu_temp || true)")
  done
  if [[ "$(distinct_temp_count "${temps[@]}")" == "1" ]]; then
    fail "the temperature reader at ${MLXFAST_MACMON} returned the identical value ${first_temp}C on ${#temps[@]} samples across ~20s; treating it as a frozen sensor, because a stuck reading passes every 40C cool gate on an arbitrarily hot GPU"
  fi
fi
ok "temperature reader is plausible and moving (samples: ${temps[*]})"

# --- 3. the timed pool is armed ---------------------------------------------
# "Armed" is a property of the CONTRACT, checked before anything on disk is
# looked at: a sentinel entry has no digest to verify a staged file against, so
# a staged file would be accepted on its name alone.
if grep -q 'PENDING-ORGANIZER' "${CONTRACT}"; then
  fail "the track contract still carries PENDING-ORGANIZER sentinels; the timed pool is unarmed and nothing can be pin-verified against it"
fi

pool_count="$(jq -r '.timed_prompt_pool | length' "${CONTRACT}")"
[[ "${pool_count}" == "8" ]] || fail "timed_prompt_pool has ${pool_count} entries, expected 8 (the cohort size this track scores)"

# A pin is {sha256, bytes} together; neither half alone is one. An entry that
# fails this is unarmed no matter what it is called.
malformed="$(jq -r '
  .timed_prompt_pool
  | to_entries
  | map(select(
      (.value.r2_path | type != "string" or length == 0)
      or (.value.sha256 | type != "string" or test("^[0-9a-f]{64}$") | not)
      or (.value.bytes | type != "number" or . <= 0)
    ))
  | map("timed_prompt_pool[" + (.key | tostring) + "]")
  | join(", ")
' "${CONTRACT}")"
[[ -z "${malformed}" ]] || fail "unarmed or malformed pool pin(s): ${malformed}"

hidden_sha="$(jq -r '.hidden_correctness_golden.sha256 // ""' "${CONTRACT}")"
hidden_bytes="$(jq -r '.hidden_correctness_golden.bytes // 0' "${CONTRACT}")"
printf '%s' "${hidden_sha}" | grep -Eq '^[0-9a-f]{64}$' \
  || fail "hidden_correctness_golden.sha256 is not a 64-hex digest; the token-fidelity oracle is unarmed"
printf '%s' "${hidden_bytes}" | grep -Eq '^[1-9][0-9]*$' \
  || fail "hidden_correctness_golden.bytes is not a positive integer"
ok "timed pool armed: 8 pinned tapes + a pinned hidden correctness golden"

# --- 4. the staged tapes match the pins -------------------------------------
GOLDEN_DIR="${MLXFAST_QWEN38_GOLDEN_DIR:-}"
[[ -n "${GOLDEN_DIR}" ]] \
  || fail "MLXFAST_QWEN38_GOLDEN_DIR is unset; the 8 timed-pool tapes are staged onto the box out of band and this job holds no credential to fetch them"
[[ -d "${GOLDEN_DIR}" ]] \
  || fail "MLXFAST_QWEN38_GOLDEN_DIR does not exist or is not a directory: ${GOLDEN_DIR}"

verify_pin() {
  # verify_pin <path> <want_sha256> <want_bytes> <label>
  local path="$1" want_sha="$2" want_bytes="$3" label="$4" got_bytes got_sha
  [[ -f "${path}" ]] || fail "${label}: staged file is missing: ${path}"
  got_bytes="$(wc -c < "${path}" | tr -d '[:space:]')"
  [[ "${got_bytes}" == "${want_bytes}" ]] \
    || fail "${label}: byte-count mismatch (staged ${got_bytes}, pinned ${want_bytes}): ${path}"
  got_sha="$(shasum -a 256 "${path}" | awk '{print $1}')"
  [[ "${got_sha}" == "${want_sha}" ]] \
    || fail "${label}: sha256 mismatch (staged ${got_sha}, pinned ${want_sha}): ${path}"
}

expected_list=""
while IFS='	' read -r r2_path want_sha want_bytes; do
  [[ -n "${r2_path}" ]] || continue
  name="${r2_path##*/}"
  verify_pin "${GOLDEN_DIR}/${name}" "${want_sha}" "${want_bytes}" "timed-pool tape ${name}"
  expected_list="${expected_list}${name}
"
done <<EOF
$(jq -r '.timed_prompt_pool[] | [.r2_path, .sha256, (.bytes | tostring)] | @tsv' "${CONTRACT}")
EOF
ok "all 8 staged timed-pool tapes match their contract pins (bytes then sha256)"

# Per-depth oracles (David ruling 2026-09-07): every live_golden_speculative
# entry must be a well-formed pin AND staged, so a speculative submission never
# reaches the GPU to find its oracle missing. Absent map = serial-only track.
spec_malformed="$(jq -r '
  (.live_golden_speculative // {})
  | to_entries
  | map(select(
      (.value.r2_path | type != "string" or length == 0)
      or (.value.sha256 | type != "string" or test("^[0-9a-f]{64}$") | not)
      or (.value.bytes | type != "number" or . <= 0)
    ))
  | map("live_golden_speculative[" + .key + "]")
  | join(", ")
' "${CONTRACT}")"
[[ -z "${spec_malformed}" ]] || fail "unarmed or malformed per-depth oracle pin(s): ${spec_malformed}"
spec_count=0
while IFS='	' read -r r2_path want_sha want_bytes; do
  [[ -n "${r2_path}" ]] || continue
  name="${r2_path##*/}"
  verify_pin "${GOLDEN_DIR}/${name}" "${want_sha}" "${want_bytes}" "per-depth oracle ${name}"
  # A per-depth oracle is a pinned cohort member: the staging-directory check
  # below must accept it beside the timed-pool tapes.
  if ! printf '%s' "${expected_list}" | grep -Fxq "${name}"; then
    expected_list="${expected_list}${name}
"
  fi
  spec_count=$((spec_count + 1))
done < <(jq -r '(.live_golden_speculative // {}) | to_entries[] | [.value.r2_path, .value.sha256, (.value.bytes|tostring)] | @tsv' "${CONTRACT}")
ok "${spec_count} per-depth oracle(s) pinned and staged"

# The staging directory must hold the cohort and NOTHING ELSE.
# tools/bonsai2-27b-measure-and-score.sh passes EVERY *.json in this directory as a
# --golden, so an extra file there is an extra cohort member -- it would change
# what is measured without failing anything downstream.
unexpected=""
for staged in "${GOLDEN_DIR}"/*.json; do
  [[ -e "${staged}" ]] || continue
  staged_name="${staged##*/}"
  if ! printf '%s' "${expected_list}" | grep -Fxq "${staged_name}"; then
    unexpected="${unexpected} ${staged_name}"
  fi
done
[[ -z "${unexpected}" ]] \
  || fail "MLXFAST_QWEN38_GOLDEN_DIR holds *.json file(s) that are not in the pinned cohort:${unexpected} (every *.json there is passed as a --golden, so an extra file silently changes the measured cohort)"
ok "no unpinned *.json in the staging directory"

# The hidden correctness oracle is pinned by digest only -- the contract gives
# it no r2_path -- and benchd resolves it itself. If the box names one
# through the existing MLXFAST_CORRECTNESS_GOLDEN_PATH convention
# (Sources/MLXFastCLI/main.swift), it must be the pinned bytes; if it names
# none, this asserts nothing about it rather than inventing a location.
if [[ -n "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
  verify_pin "${MLXFAST_CORRECTNESS_GOLDEN_PATH}" "${hidden_sha}" "${hidden_bytes}" "hidden correctness golden"
  ok "MLXFAST_CORRECTNESS_GOLDEN_PATH matches hidden_correctness_golden"
else
  ok "MLXFAST_CORRECTNESS_GOLDEN_PATH unset; benchd resolves the oracle from the contract"
fi

# --- 4b. the live golden the single-leg run scores over is staged -----------
# Single-leg reads exactly ONE golden: the fixture's live_golden, resolved by
# tools/bonsai2-27b-measure-and-score.sh as <live_golden>.golden.json. The
# loop above already pin-verified it AS a pool member; this asserts the
# fixture's live_golden actually NAMES a pinned pool entry and is staged, so a
# live_golden rotation that points at a golden absent from the pool -- or a box
# that staged the pool but not the live golden -- is caught here, pre-GPU,
# rather than at measure time.
LIVE_GOLDEN_NAME="$(jq -r '.live_golden // ""' "${CONTRACT}")"
[[ -n "${LIVE_GOLDEN_NAME}" ]] \
  || fail "the fixture declares no live_golden; there is no golden for the single-leg run to score over"
live_golden_base="${LIVE_GOLDEN_NAME}.golden.json"
printf '%s' "${expected_list}" | grep -Fxq "${live_golden_base}" \
  || fail "live_golden '${LIVE_GOLDEN_NAME}' names no timed_prompt_pool entry (looked for ${live_golden_base}); it carries no pin and cannot be pin-verified"
[[ -f "${GOLDEN_DIR}/${live_golden_base}" ]] \
  || fail "the live golden ${live_golden_base} is not staged in ${GOLDEN_DIR}; it is the one golden the single-leg run scores over"
ok "live golden ${live_golden_base} is pinned and staged (the single-leg scored golden)"

# --- 5. the fixture is armed for official scoring ---------------------------
# benchd refuses, pre-GPU, to seal an official artifact unless the fixture
# declares official_scoring_enabled: true (enforce_official_scoring_enabled).
# false AND absent both refuse -- an absent arm state is not an armed one. The
# measure wrapper's --preflight-only mirrors this; refusing here makes an
# unarmed ranked dispatch fail before setup rather than after the GPU window.
armed="$(jq -r '.official_scoring_enabled // false' "${CONTRACT}")"
[[ "${armed}" == "true" ]] \
  || fail "the track contract does not declare official_scoring_enabled: true (got '${armed}'); benchd refuses to seal an official score for an unarmed track, and so does this gate"
ok "fixture is armed for official scoring (official_scoring_enabled: true)"

# --- 6. the paired legs: the reference workspace and this box's band --------
# THE RUN IS PAIRED, AND THE PAIR IS PER BOX (David ruling 2026-09-08). A ranked
# run measures TWO legs on the same box in the same job: a SERIAL-CONTROL leg on
# the organizer-staged reference tree and the CANDIDATE leg on the submission
# tree. The score is the LIVE ratio of the two. Nothing stores a pair.
#
# So two things must be true of this box before a window opens:
#
#   MLXFAST_BASELINE_WORKSPACE   a BUILT checkout of this repository at the
#                                fixture's baseline_reference_commit. The
#                                serial-control leg runs there. A tree at a
#                                different commit is a different denominator,
#                                so the HEAD check is exact, not a prefix.
#   MLXFAST_BASELINE_CALIBRATION this box's baseline-calibration.json: the
#                                HEALTH BAND leg 1 must land inside. It is
#                                NEVER the denominator. benchd kills the run by
#                                name when leg 1 falls outside the band, so a
#                                file naming another box or another reference
#                                commit would judge this leg against the wrong
#                                machine.
#
# Both are refused by NAME. There is no default and no fallback: a box that
# cannot present a verified reference tree measures nothing.
BASELINE_WORKSPACE="${MLXFAST_BASELINE_WORKSPACE:-}"
BASELINE_CALIBRATION="${MLXFAST_BASELINE_CALIBRATION:-}"

[[ -n "${BASELINE_WORKSPACE}" ]] \
  || fail "MLXFAST_BASELINE_WORKSPACE is unset; the paired run has no reference tree for the serial-control leg, so it has no denominator"
[[ -d "${BASELINE_WORKSPACE}" ]] \
  || fail "MLXFAST_BASELINE_WORKSPACE does not exist or is not a directory: ${BASELINE_WORKSPACE}"
[[ -n "${BASELINE_CALIBRATION}" ]] \
  || fail "MLXFAST_BASELINE_CALIBRATION is unset; the serial-control leg would have no health band and a box that was not itself would still seal a score"
[[ -f "${BASELINE_CALIBRATION}" ]] \
  || fail "MLXFAST_BASELINE_CALIBRATION does not name a file: ${BASELINE_CALIBRATION}"

REFERENCE_COMMIT="$(jq -r '.baseline_reference_commit // ""' "${CONTRACT}")"
printf '%s' "${REFERENCE_COMMIT}" | grep -Eq '^[0-9a-f]{40}$' \
  || fail "the track contract declares no 40-hex baseline_reference_commit (got '${REFERENCE_COMMIT}'); there is nothing to hold the reference tree to"

command -v git >/dev/null 2>&1 || fail "git is required to verify the reference workspace's HEAD"
[[ -e "${BASELINE_WORKSPACE}/.git" ]] \
  || fail "MLXFAST_BASELINE_WORKSPACE is not a git checkout (no .git at ${BASELINE_WORKSPACE}); its commit cannot be verified, so the serial-control leg would run on an unidentified tree"
workspace_head="$(git -C "${BASELINE_WORKSPACE}" rev-parse HEAD 2>/dev/null || true)"
[[ "${workspace_head}" == "${REFERENCE_COMMIT}" ]] \
  || fail "the reference workspace is at '${workspace_head:-nothing}' but the contract pins baseline_reference_commit ${REFERENCE_COMMIT}; re-stage it with tools/stage-baseline-workspace.sh"
ok "reference workspace is a git checkout at baseline_reference_commit ${REFERENCE_COMMIT}"

# The reference tree must be BUILT, in the layout tools/stage-bench-worker.sh
# writes: benchd resolves the engine at the fixed workspace-relative path
# .build/release/bench-worker, and Metal loads mlx.metallib from the SAME
# directory as the running binary. The fingerprint sidecar travels with the
# metallib -- the harness reads it at <metallib>.fingerprint, and a metallib
# without one fails that check exactly as a tampered one does.
REFERENCE_WORKER="${BASELINE_WORKSPACE}/.build/release/bench-worker"
REFERENCE_METALLIB="${BASELINE_WORKSPACE}/.build/release/mlx.metallib"
[[ -x "${REFERENCE_WORKER}" ]] \
  || fail "the reference workspace has no executable worker at ${REFERENCE_WORKER}; build and stage it on the box (tools/stage-baseline-workspace.sh), because nothing in the ranked job builds it"
[[ -f "${REFERENCE_METALLIB}" ]] \
  || fail "the reference worker has no sibling mlx.metallib at ${REFERENCE_METALLIB}; Metal loads it from the running binary's own directory, so the serial-control leg would die at its first MLXArray"
[[ -f "${REFERENCE_METALLIB}.fingerprint" ]] \
  || fail "the reference metallib has no fingerprint sidecar at ${REFERENCE_METALLIB}.fingerprint; the harness reads it there, and a metallib without one fails the check the same way a tampered one does"
ok "reference worker staged with its mlx.metallib and fingerprint sidecar"

# The control leg runs on the pinned commit END TO END, which includes the
# weights the reference tree's OWN transform produced. Sharing the candidate's
# transformed tree would put a submission's transform on both sides of the
# ratio, which is the one thing a control leg exists to prevent.
[[ -f "${BASELINE_WORKSPACE}/weights/config.json" ]] \
  || fail "the reference workspace has no transformed weights at ${BASELINE_WORKSPACE}/weights (no config.json); tools/stage-baseline-workspace.sh runs the tree's own transform, and the control leg must not borrow the candidate's"
ok "reference workspace carries its own transformed weights"

# The commit's own date bounds the calibration from below: a calibration
# captured BEFORE the reference tree existed cannot have measured that tree.
reference_commit_date="$(git -C "${BASELINE_WORKSPACE}" show -s --format=%cI "${REFERENCE_COMMIT}" 2>/dev/null || true)"
[[ -n "${reference_commit_date}" ]] \
  || fail "cannot read the commit date of ${REFERENCE_COMMIT} from ${BASELINE_WORKSPACE}; the calibration's captured_at cannot be bounded"

# The calibration file. python3 parses it (the fixture's own tools already
# require python3), checks every field the ranked path depends on, and prints
# ONE refusal naming the first thing that is wrong.
CALIBRATION_TRACK_ID="$(jq -r '.track_id' "${CONTRACT}")"
command -v python3 >/dev/null 2>&1 || fail "python3 is required to validate the baseline calibration file"
calibration_error="$(
  MLXFAST_CAL_PATH="${BASELINE_CALIBRATION}" \
  MLXFAST_CAL_TRACK_ID="${CALIBRATION_TRACK_ID}" \
  MLXFAST_CAL_REF_COMMIT="${REFERENCE_COMMIT}" \
  MLXFAST_CAL_REF_DATE="${reference_commit_date}" \
  MLXFAST_CAL_BOX="${RUNNER_NAME:-}" \
  python3 - <<'PYEOF'
import datetime
import json
import math
import os
import sys

path = os.environ["MLXFAST_CAL_PATH"]


def refuse(message):
    sys.stdout.write(message)
    raise SystemExit(0)


try:
    with open(path, encoding="utf-8") as handle:
        cal = json.load(handle)
except (OSError, json.JSONDecodeError) as exc:
    refuse(f"the baseline calibration file does not parse as JSON ({exc}): {path}")

if not isinstance(cal, dict):
    refuse(f"the baseline calibration file is not a JSON object: {path}")

if cal.get("version") != 1:
    refuse(
        f"baseline calibration version is {cal.get('version')!r}, expected 1; "
        "this box's file was written by a different calibrator"
    )

want_track = os.environ["MLXFAST_CAL_TRACK_ID"]
if cal.get("track_id") != want_track:
    refuse(
        f"baseline calibration track_id is {cal.get('track_id')!r} but this track is "
        f"{want_track!r}; the band belongs to another track"
    )

want_box = os.environ["MLXFAST_CAL_BOX"]
if want_box:
    if cal.get("box") != want_box:
        refuse(
            f"baseline calibration box is {cal.get('box')!r} but this runner is "
            f"{want_box!r}; the band was measured on another machine and says nothing "
            "about this one"
        )

want_commit = os.environ["MLXFAST_CAL_REF_COMMIT"]
if cal.get("reference_commit") != want_commit:
    refuse(
        f"baseline calibration reference_commit is {cal.get('reference_commit')!r} but "
        f"the contract pins {want_commit!r}; the band measured a different reference tree"
    )

numeric_fields = (
    "passes",
    "prefill_seconds_per_token_mean",
    "decode_seconds_per_token_mean",
    "prefill_cv",
    "decode_cv",
    "prefill_band_low",
    "prefill_band_high",
    "decode_band_low",
    "decode_band_high",
)
for field in numeric_fields:
    value = cal.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        refuse(f"baseline calibration {field} is {value!r}, which is not a number")
    if not math.isfinite(value):
        refuse(f"baseline calibration {field} is {value!r}, which is not finite")
    if value <= 0:
        refuse(
            f"baseline calibration {field} is {value!r}; every measured value and every "
            "band edge must be positive (a zero CV across repeated passes is a frozen "
            "clock, not a stable box)"
        )

for axis in ("prefill", "decode"):
    low = cal[f"{axis}_band_low"]
    high = cal[f"{axis}_band_high"]
    if not low < 1 < high:
        refuse(
            f"baseline calibration {axis} band is [{low}, {high}]; a band must straddle "
            "1 (low < 1 < high), or the leg it judges can never be healthy"
        )

captured_at = cal.get("captured_at")
if not isinstance(captured_at, str) or not captured_at:
    refuse(f"baseline calibration captured_at is {captured_at!r}, expected an RFC 3339 timestamp")
try:
    captured = datetime.datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
except ValueError:
    refuse(f"baseline calibration captured_at {captured_at!r} is not an RFC 3339 timestamp")
if captured.tzinfo is None:
    refuse(f"baseline calibration captured_at {captured_at!r} carries no timezone offset")
reference_date = datetime.datetime.fromisoformat(os.environ["MLXFAST_CAL_REF_DATE"])
if captured <= reference_date:
    refuse(
        f"baseline calibration captured_at {captured_at} is not after the reference "
        f"commit's date {reference_date.isoformat()}; it cannot have measured a tree "
        "that did not exist yet"
    )
PYEOF
)" || fail "the baseline calibration validator failed to run against ${BASELINE_CALIBRATION}"
[[ -z "${calibration_error}" ]] || fail "${calibration_error} (${BASELINE_CALIBRATION})"
if [[ -n "${RUNNER_NAME:-}" ]]; then
  ok "baseline calibration parses and names this track, this box (${RUNNER_NAME}) and the reference commit"
else
  ok "baseline calibration parses and names this track and the reference commit (RUNNER_NAME unset, so the box name is not checked)"
fi

echo "ranked-box-preflight: all checks passed"
