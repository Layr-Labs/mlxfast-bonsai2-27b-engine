#!/usr/bin/env bash
#
# test-calibrate-box.sh -- tools/calibrate-box.sh drives the contract's
# calibrate-baseline invocation, holds the GPU lock, and refuses by name.
#
# WHAT IS UNDER TEST. tools/calibrate-box.sh records THIS box's health band for
# the serial-control leg of the paired ranked run (David ruling 2026-09-08). It
# is a box-only script -- the real one loads an 18.5 GB checkpoint -- so what a
# hermetic suite can hold is everything AROUND the measurement: the argv it
# builds, the environment it builds it from, the lock it takes, and every
# refusal that has to happen BEFORE the GPU is touched.
#
# HERMETIC. A STUB benchd records its argv and writes a calibration file; a
# synthetic git repository stands in for the reference tree; the GPU lock is a
# throwaway path. Nothing loads a model, spawns an engine, or reaches the
# network.
#
# IT BOOTS NO RESIDENT. Each pass is a full serial-control leg and benchd boots
# that leg's resident from the REFERENCE tree, so this script must hand benchd
# no socket -- a socket from here would name an already-loaded engine that is
# not the reference tree's, which is the defect public run 34230122059 hit on
# the ranked path. Cases 10 and 11 hold that.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

command -v jq >/dev/null 2>&1 || { echo "test-calibrate-box.sh: jq is required" >&2; exit 1; }

BOX_NAME="synthetic-ranked-box"

# --- the synthetic root -----------------------------------------------------
ROOT="${WORK}/root"
mkdir -p "${ROOT}/tools" "${ROOT}/fixtures"
cp "${REPO_ROOT}/tools/calibrate-box.sh" "${ROOT}/tools/"
chmod +x "${ROOT}/tools/calibrate-box.sh"
cp "${REPO_ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" "${ROOT}/fixtures/"
cp "${REPO_ROOT}/benchmark.json" "${ROOT}/benchmark.json"

TRACK_ID="$(jq -r '.trackId' "${ROOT}/benchmark.json")"

# A FRESHLY STAMPED TRACK HAS NO LIVE GOLDEN. The real tapes are organizer
# material, published in R2 and staged on the box, and they do not exist for
# this track yet, so calibrate-box.sh refuses before it builds any argv. The
# copy in ${ROOT} therefore gets a synthetic live_golden and the pool entry
# that pins it; the shipped fixture is never written, and a fixture that
# already carries a pool is left exactly as it is.
python3 - "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" <<'SYNTHEOF'
import json, sys

contract_path = sys.argv[1]
contract = json.load(open(contract_path, encoding="utf-8"))
if not contract.get("live_golden"):
    live = "synthetic-live"
    contract["live_golden"] = live
    contract["timed_prompt_pool"] = [{
        "r2_path": "correctness_prompts/%s/%s.golden.json" % (contract["track_id"], live),
        "sha256": "0" * 64,
        "bytes": 1,
    }]
    with open(contract_path, "w", encoding="utf-8") as fh:
        json.dump(contract, fh, indent=2)
        fh.write("\n")
SYNTHEOF

LIVE_GOLDEN_NAME="$(jq -r '.live_golden' "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json")"

# --- the synthetic reference tree, and the contract that pins it ------------
REF_WS="${WORK}/reference"
mkdir -p "${REF_WS}"
git -C "${REF_WS}" init --quiet
git -C "${REF_WS}" config user.email "test@example.invalid"
git -C "${REF_WS}" config user.name "calibrate test"
echo "reference tree" > "${REF_WS}/README.md"
git -C "${REF_WS}" add README.md
git -C "${REF_WS}" commit --quiet --no-gpg-sign -m "reference"
REF_COMMIT="$(git -C "${REF_WS}" rev-parse HEAD)"
# The reference tree's own build and its own transformed weights: the control
# leg is the pinned commit end to end, and the script refuses without either.
mkdir -p "${REF_WS}/.build/release" "${REF_WS}/weights"
printf '#!/bin/sh\nexit 0\n' > "${REF_WS}/.build/release/bench-worker"
chmod +x "${REF_WS}/.build/release/bench-worker"
printf '{}\n' > "${REF_WS}/weights/config.json"
jq --arg c "${REF_COMMIT}" '.baseline_reference_commit = $c' \
  "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" > "${WORK}/contract.tmp"
mv "${WORK}/contract.tmp" "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json"

GOLDEN_DIR="${WORK}/goldens"
mkdir -p "${GOLDEN_DIR}"
echo '{}' > "${GOLDEN_DIR}/${LIVE_GOLDEN_NAME}.golden.json"

# --- the stub benchd --------------------------------------------------------
# Records its argv and the track id it was handed, then writes the calibration
# file at the --out path unless told to write nothing (the CV-refusal case).
STUB="${WORK}/benchd-stub"
cat > "${STUB}" <<'STUBEOF'
#!/usr/bin/env bash
printf '%s\n' "$@" > "${STUB_CAPTURE_ARGV}"
printf '%s\n' "${MLXFAST_QWEN_MTP_TRACK_ID-__UNSET__}" > "${STUB_CAPTURE_ENV}"
printf '%s\n' "${BENCH_WORKER_RESIDENT_SOCKET-__UNSET__}" > "${STUB_CAPTURE_SOCKET}"
if [[ "${STUB_WRITE_OUT:-1}" == "1" ]]; then
  out=""
  prev=""
  for arg in "$@"; do
    if [[ "${prev}" == "--out" ]]; then out="${arg}"; fi
    prev="${arg}"
  done
  [[ -z "${out}" ]] || echo '{"version":1}' > "${out}"
fi
exit "${STUB_EXIT:-0}"
STUBEOF
chmod 755 "${STUB}"

# The channel manifest beside the pinned binary. calibrate-box.sh records its
# source_commit in the band, so a band states which benchmarker measured it.
BENCHD_BIN="${WORK}/benchd-bin"
mkdir -p "${BENCHD_BIN}"
BENCHD_SOURCE_COMMIT="89abcdef0123456789abcdef0123456789abcdef"
printf '{"branch":"main","source_commit":"%s","sha256":"%s","bytes":1}\n' \
  "${BENCHD_SOURCE_COMMIT}" "$(printf '0%.0s' {1..64})" > "${BENCHD_BIN}/benchd.manifest.json"

# run_calibrate CASE [ENV=VAL...] -- [args...]
run_calibrate() {
  local case_name="$1"
  shift
  rm -f "${WORK}/${case_name}.argv" "${WORK}/${case_name}.env" "${WORK}/${case_name}.out.json"
  local extra=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do extra+=("$1"); shift; done
  [[ $# -eq 0 ]] || shift
  env -u MLXFAST_QWEN_MTP_TRACK_ID -u RUNNER_NAME -u BENCH_WORKER_RESIDENT_SOCKET \
    STUB_CAPTURE_ARGV="${WORK}/${case_name}.argv" \
    STUB_CAPTURE_ENV="${WORK}/${case_name}.env" \
    STUB_CAPTURE_SOCKET="${WORK}/${case_name}.socket-env" \
    BENCHD="${STUB}" \
    BENCHD_BIN_DIR="${BENCHD_BIN}" \
    MLXFAST_BASELINE_WORKSPACE="${REF_WS}" \
    MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
    RESIDENT_UP_LOCK_PATH="${WORK}/${case_name}.gpu.lock" \
    MLXFAST_GPU_LOCK_TIMEOUT_S=30 \
    ${extra[@]+"${extra[@]}"} \
    "${ROOT}/tools/calibrate-box.sh" "$@" > "${WORK}/${case_name}.log" 2>&1
  rc=$?
}

argv_has_pair() {
  # argv_has_pair <file> <flag> <value>
  python3 - "$1" "$2" "$3" <<'PYEOF'
import sys
path, flag, value = sys.argv[1], sys.argv[2], sys.argv[3]
args = open(path, encoding="utf-8").read().splitlines()
for i, arg in enumerate(args):
    if arg == flag and i + 1 < len(args) and args[i + 1] == value:
        raise SystemExit(0)
raise SystemExit(1)
PYEOF
}

# --- case 1: the contract invocation ---------------------------------------
run_calibrate case1 -- "${BOX_NAME}" "${WORK}/case1.out.json"
if [[ "${rc}" -ne 0 ]]; then
  fail "case 1: calibrate-box.sh exited ${rc}; output: $(cat "${WORK}/case1.log")"
elif [[ ! -f "${WORK}/case1.argv" ]]; then
  fail "case 1: benchd was never spawned; output: $(cat "${WORK}/case1.log")"
else
  head -n 1 "${WORK}/case1.argv" | grep -qx 'calibrate-baseline' \
    || fail "case 1: the subcommand is not calibrate-baseline: $(head -n 1 "${WORK}/case1.argv")"
  argv_has_pair "${WORK}/case1.argv" --baseline-workspace "${REF_WS}" \
    || fail "case 1: argv carries no --baseline-workspace ${REF_WS}"
  # THE ENGINE PATH IS RELATIVE TO THE WORKSPACE. benchd re-roots it there; an
  # absolute path would name this checkout's binary and measure the candidate
  # as its own control.
  argv_has_pair "${WORK}/case1.argv" --engine .build/release/bench-worker \
    || fail "case 1: argv does not carry the workspace-relative --engine .build/release/bench-worker"
  # THE WEIGHTS ARE THE WORKSPACE'S OWN, not this checkout's.
  argv_has_pair "${WORK}/case1.argv" --weights "${REF_WS}/weights" \
    || fail "case 1: argv does not carry the reference tree's own --weights"
  argv_has_pair "${WORK}/case1.argv" --track "${TRACK_ID}" \
    || fail "case 1: argv does not carry --track ${TRACK_ID}"
  argv_has_pair "${WORK}/case1.argv" --prompt "${LIVE_GOLDEN_NAME}" \
    || fail "case 1: argv does not carry --prompt ${LIVE_GOLDEN_NAME}"
  argv_has_pair "${WORK}/case1.argv" --reference-commit "${REF_COMMIT}" \
    || fail "case 1: argv does not carry --reference-commit ${REF_COMMIT}"
  argv_has_pair "${WORK}/case1.argv" --benchd-source-commit "${BENCHD_SOURCE_COMMIT}" \
    || fail "case 1: argv does not carry the channel manifest's --benchd-source-commit"
  argv_has_pair "${WORK}/case1.argv" --golden "${GOLDEN_DIR}/${LIVE_GOLDEN_NAME}.golden.json" \
    || fail "case 1: argv does not name the fixture's live golden"
  argv_has_pair "${WORK}/case1.argv" --passes 4 \
    || fail "case 1: argv does not carry --passes 4 (the contract's pass count)"
  argv_has_pair "${WORK}/case1.argv" --contract "${ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json" \
    || fail "case 1: argv does not carry --contract (benchd reads the model shape and the window from the fixture)"
  argv_has_pair "${WORK}/case1.argv" --box "${BOX_NAME}" \
    || fail "case 1: argv does not carry --box ${BOX_NAME}"
  argv_has_pair "${WORK}/case1.argv" --out "${WORK}/case1.out.json" \
    || fail "case 1: argv does not carry the requested --out"
  [[ -f "${WORK}/case1.out.json" ]] || fail "case 1: no calibration file was written"
  grep -q "wrote ${WORK}/case1.out.json" "${WORK}/case1.log" \
    || fail "case 1: the script does not print the file it wrote"
  [[ "$(cat "${WORK}/case1.env")" == "${TRACK_ID}" ]] \
    || fail "case 1: benchd saw MLXFAST_QWEN_MTP_TRACK_ID='$(cat "${WORK}/case1.env")', expected '${TRACK_ID}'"
fi

# --- case 2: the box name defaults to RUNNER_NAME ---------------------------
run_calibrate case2 "RUNNER_NAME=${BOX_NAME}" -- "" "${WORK}/case2.out.json"
if [[ "${rc}" -ne 0 ]]; then
  fail "case 2: calibrate-box.sh refused an empty box name with RUNNER_NAME exported: $(cat "${WORK}/case2.log")"
elif ! argv_has_pair "${WORK}/case2.argv" --box "${BOX_NAME}"; then
  fail "case 2: argv does not carry the RUNNER_NAME box name"
fi

# --- case 3: no box name anywhere -------------------------------------------
run_calibrate case3 -- "" "${WORK}/case3.out.json"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 3: calibrate-box.sh ran with no box name"
elif ! grep -q "box name" "${WORK}/case3.log"; then
  fail "case 3: the refusal does not name the missing box name: $(cat "${WORK}/case3.log")"
fi

# --- case 4: no output path --------------------------------------------------
run_calibrate case4 -- "${BOX_NAME}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 4: calibrate-box.sh ran with no output path"
elif ! grep -q "no output path" "${WORK}/case4.log"; then
  fail "case 4: the refusal does not name the missing output path: $(cat "${WORK}/case4.log")"
fi

# --- case 5: the reference tree is required ---------------------------------
run_calibrate case5 "MLXFAST_BASELINE_WORKSPACE=" -- "${BOX_NAME}" "${WORK}/case5.out.json"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 5: calibrate-box.sh ran with no reference tree"
elif ! grep -q "MLXFAST_BASELINE_WORKSPACE is unset" "${WORK}/case5.log"; then
  fail "case 5: the refusal does not name the variable: $(cat "${WORK}/case5.log")"
fi
if [[ -f "${WORK}/case5.argv" ]]; then
  fail "case 5: benchd was spawned despite the missing reference tree"
fi

# --- case 6: the reference tree must be at the pinned commit ----------------
OTHER_WS="${WORK}/other"
mkdir -p "${OTHER_WS}"
git -C "${OTHER_WS}" init --quiet
git -C "${OTHER_WS}" config user.email "test@example.invalid"
git -C "${OTHER_WS}" config user.name "calibrate test"
echo "different" > "${OTHER_WS}/README.md"
git -C "${OTHER_WS}" add README.md
git -C "${OTHER_WS}" commit --quiet --no-gpg-sign -m "different"
run_calibrate case6 "MLXFAST_BASELINE_WORKSPACE=${OTHER_WS}" -- "${BOX_NAME}" "${WORK}/case6.out.json"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 6: calibrate-box.sh calibrated a tree at the wrong commit"
elif ! grep -q "${REF_COMMIT}" "${WORK}/case6.log"; then
  fail "case 6: the refusal does not name the pinned commit: $(cat "${WORK}/case6.log")"
fi

# --- case 6b: the reference tree must be built -------------------------------
mv "${REF_WS}/.build/release/bench-worker" "${WORK}/worker.hidden"
run_calibrate case6b -- "${BOX_NAME}" "${WORK}/case6b.out.json"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 6b: calibrate-box.sh calibrated a reference tree with no worker"
elif ! grep -q "no executable worker" "${WORK}/case6b.log"; then
  fail "case 6b: the refusal does not name the missing worker: $(cat "${WORK}/case6b.log")"
fi
mv "${WORK}/worker.hidden" "${REF_WS}/.build/release/bench-worker"

# --- case 6c: the reference tree must carry its own transformed weights -----
mv "${REF_WS}/weights" "${WORK}/weights.hidden"
run_calibrate case6c -- "${BOX_NAME}" "${WORK}/case6c.out.json"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 6c: calibrate-box.sh calibrated a reference tree with no transformed weights"
elif ! grep -q "no transformed weights" "${WORK}/case6c.log"; then
  fail "case 6c: the refusal does not name the missing weights: $(cat "${WORK}/case6c.log")"
fi
mv "${WORK}/weights.hidden" "${REF_WS}/weights"

# --- case 7: the staged golden is required ----------------------------------
run_calibrate case7 "MLXFAST_QWEN38_GOLDEN_DIR=" -- "${BOX_NAME}" "${WORK}/case7.out.json"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 7: calibrate-box.sh ran with no staged golden pool"
elif ! grep -q "MLXFAST_QWEN38_GOLDEN_DIR is unset" "${WORK}/case7.log"; then
  fail "case 7: the refusal does not name the variable: $(cat "${WORK}/case7.log")"
fi

# --- case 8: benchd writing nothing is a refusal, not a silent success ------
# A CV above 1 % on either axis makes benchd write no file. The script must say
# so rather than exit 0 on an absent band.
run_calibrate case8 STUB_WRITE_OUT=0 -- "${BOX_NAME}" "${WORK}/case8.out.json"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 8: calibrate-box.sh exited 0 with no calibration file written"
elif ! grep -q "CV above 1 %" "${WORK}/case8.log"; then
  fail "case 8: the refusal does not explain the missing file: $(cat "${WORK}/case8.log")"
fi

# --- case 9: the GPU lock is held for the window ----------------------------
# A second owner holding the lock must make the calibration refuse rather than
# measure beside it. The holder is a python process holding flock on the same
# path the script takes.
HOLD_LOCK="${WORK}/case9.gpu.lock"
python3 -c "
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o666)
fcntl.flock(fd, fcntl.LOCK_EX)
sys.stderr.write('held\n')
sys.stderr.flush()
time.sleep(30)
" "${HOLD_LOCK}" 2>"${WORK}/holder.err" &
holder_pid=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
  grep -q held "${WORK}/holder.err" 2>/dev/null && break
  sleep 0.3
done
run_calibrate case9 "RESIDENT_UP_LOCK_PATH=${HOLD_LOCK}" MLXFAST_GPU_LOCK_TIMEOUT_S=1 -- "${BOX_NAME}" "${WORK}/case9.out.json"
kill "${holder_pid}" 2>/dev/null
wait "${holder_pid}" 2>/dev/null
if [[ "${rc}" -eq 0 ]]; then
  fail "case 9: calibrate-box.sh calibrated while another owner held the GPU lock"
elif ! grep -q "refusing to calibrate beside another GPU owner" "${WORK}/case9.log"; then
  fail "case 9: the refusal does not name the lock contention: $(cat "${WORK}/case9.log")"
fi
if [[ -f "${WORK}/case9.argv" ]]; then
  fail "case 9: benchd was spawned while another owner held the GPU lock"
fi

# --- case 10: no socket reaches benchd, and no resident is booted -----------
# benchd boots each pass's resident from the reference tree. A socket handed in
# from here would send every pass to an already-loaded engine that is not that
# tree's.
if [[ ! -f "${WORK}/case1.socket-env" ]]; then
  fail "case 10: the stub recorded no socket environment"
elif [[ "$(cat "${WORK}/case1.socket-env")" != "__UNSET__" ]]; then
  fail "case 10: benchd saw BENCH_WORKER_RESIDENT_SOCKET=$(cat "${WORK}/case1.socket-env"); calibrate-box.sh must hand benchd no socket"
fi
if [[ -e "${WORK}/case1.gpu.lock" ]]; then
  : # the GPU lock is still taken -- the box needs one loader whoever boots it
else
  fail "case 10: the GPU lock was never taken; every per-pass resident boot refuses when nobody holds it"
fi

# --- case 11: an inherited socket is refused by name, before the lock -------
run_calibrate case11 "BENCH_WORKER_RESIDENT_SOCKET=${WORK}/inherited.sock" -- "${BOX_NAME}" "${WORK}/case11.out.json"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 11: calibrate-box.sh ran with an inherited BENCH_WORKER_RESIDENT_SOCKET"
elif ! grep -q "BENCH_WORKER_RESIDENT_SOCKET is set" "${WORK}/case11.log"; then
  fail "case 11: the refusal does not name the variable: $(cat "${WORK}/case11.log")"
fi
if [[ -f "${WORK}/case11.argv" ]]; then
  fail "case 11: benchd was spawned despite the inherited socket"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "test-calibrate-box.sh: all 13 cases passed"
  exit 0
fi
echo "test-calibrate-box.sh: ${failures} case(s) failed" >&2
exit 1
