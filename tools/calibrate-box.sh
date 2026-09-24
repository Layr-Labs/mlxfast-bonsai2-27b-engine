#!/usr/bin/env bash
#
# calibrate-box.sh -- record THIS box's health band for the serial-control leg.
#
# WHAT A CALIBRATION IS, AND WHAT IT IS NOT. A ranked run on this track is
# PAIRED (David ruling 2026-09-08): it measures a serial-control leg on the
# organizer-staged reference tree and a candidate leg on the submission tree, in
# the same job, on the same box, and the score is the LIVE ratio of the two. The
# calibration file this script writes is NEVER that denominator. It is a HEALTH
# BAND: it records what this box's serial-control leg costs when the box is
# itself, so benchd can kill a run by name when leg 1 lands outside the band
# instead of sealing a score measured on a machine that was thermally throttled,
# shared, or otherwise not the machine that was calibrated.
#
# So a stale calibration file cannot move a score. It can only stop one.
#
# WHAT IT MEASURES. `benchd calibrate-baseline` runs the serial-control leg
# --passes times under the FULL official methodology -- the quiescence gate and
# the cool gate per pass,
# one resident worker per pass, the same live golden the ranked run scores over
# -- and writes the mean, the coefficient of variation and the band on both
# axes. It REFUSES to write a file when the CV exceeds 1 % on either axis,
# because a box that cannot repeat itself has no band worth recording. That
# refusal is the point of the four passes, and this script does not soften it.
#
# THE LEG RUNS ENTIRELY INSIDE THE REFERENCE TREE. --engine is a path RELATIVE
# to the workspace (.build/release/bench-worker, where tools/stage-bench-worker.sh
# puts it) and --weights is the workspace's OWN transformed weights, not this
# checkout's. That is the whole point of a reference tree: the control leg's
# engine, its Metal library and its weights all come from the pinned commit, so
# nothing this submission changed can move the denominator or the band.
#
# THIS SCRIPT RUNS ON THE RANKED BOX ONLY. It loads the checkpoint, so it takes
# the fleet GPU lock (/tmp/mtplx-gpu-exclusive.lock) FIRST and holds it for the
# whole calibration -- the same lock, taken the same way, as the ranked
# measurement in tools/bonsai2-27b-measure-and-score.sh. A calibration
# measured beside another GPU owner is a calibration of the wrong machine.
#
# IT BOOTS NO RESIDENT AND EXPORTS NO SOCKET. Each pass is a full serial-control
# leg under the official methodology, and benchd boots that leg's resident
# itself by calling `tools/resident-up.sh --boot` in the reference tree, then
# `--stop` when the pass ends. A resident this script booted would be the
# candidate tree's, and every pass would then measure the wrong engine -- the
# same defect public run 34230122059 hit on the ranked path. An inherited
# BENCH_WORKER_RESIDENT_SOCKET is refused for the same reason.
#
# Usage:
#   tools/calibrate-box.sh <box name> <output file>
#
#   <box name>     the runner name this box runs under (Actions RUNNER_NAME).
#                  tools/ranked-box-preflight.sh refuses a calibration whose
#                  `box` does not equal it, so it is an argument rather than a
#                  guess. Defaults to RUNNER_NAME when that is exported.
#   <output file>  where to write baseline-calibration.json. Export its path as
#                  MLXFAST_BASELINE_CALIBRATION in the runner service
#                  environment afterwards.
#
# Env (the SAME names the ranked job runs under):
#   MLXFAST_BASELINE_WORKSPACE   the built reference tree the control leg runs
#                                 on. Required; there is no default, because a
#                                 guessed tree is a different denominator.
#   MLXFAST_QWEN38_GOLDEN_DIR    the staged golden pool. The live golden is
#                                 resolved from it as <live_golden>.golden.json,
#                                 exactly as the ranked path resolves it.
#   BENCHD_BIN_DIR               where tools/fetch-benchd.sh keeps the pinned
#                                 binary and its benchd.manifest.json. The
#                                 manifest's `source_commit` is recorded in the
#                                 calibration as benchd_source_commit, so a band
#                                 states which benchmarker measured it.
#   MLXFAST_BENCHD_SOURCE_COMMIT an explicit benchd source commit, used when the
#                                 manifest cannot be read (benchd reads the same
#                                 name itself when the flag is absent).
#   BENCHD                       an explicit benchd, NOT hash-checked (benchd
#                                 development). Same meaning as in the measure
#                                 script.
#   MLXFAST_CALIBRATION_PASSES   pass count. Default 4, the contract's value.
#   RESIDENT_UP_LOCK_PATH        the GPU lock. Default
#                                 /tmp/mtplx-gpu-exclusive.lock.
#   MLXFAST_GPU_LOCK_TIMEOUT_S   ceiling on the wait for it (default 1800).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

die() {
  echo "calibrate-box.sh: $*" >&2
  exit 1
}

BOX_NAME="${1:-${RUNNER_NAME:-}}"
OUT_PATH="${2:-}"
[[ -n "${BOX_NAME}" ]] \
  || die "usage: tools/calibrate-box.sh <box name> <output file> -- the box name must equal this runner's RUNNER_NAME, because the preflight refuses a calibration that names another machine"
[[ -n "${OUT_PATH}" ]] \
  || die "usage: tools/calibrate-box.sh <box name> <output file> -- no output path given"
if [[ $# -gt 2 ]]; then
  die "unrecognized argument: $3"
fi

command -v jq >/dev/null 2>&1 || die "jq is required (it reads the fixture's live_golden and track id)"

# AN INHERITED SOCKET IS A REFUSAL. Every calibration pass is a control leg on
# the REFERENCE tree, and benchd boots that leg's resident from that tree. A
# socket in the environment would send every pass to an already-loaded resident
# -- in practice the candidate tree's -- and the band would then describe the
# wrong engine.
if [[ -n "${BENCH_WORKER_RESIDENT_SOCKET:-}" ]]; then
  die "BENCH_WORKER_RESIDENT_SOCKET is set (${BENCH_WORKER_RESIDENT_SOCKET}); each calibration pass is a control leg whose resident benchd boots from the reference tree, so a socket in the environment would measure an already-loaded engine that is not that tree's. Unset it."
fi
command -v python3 >/dev/null 2>&1 || die "python3 is required to hold the GPU lock for the calibration window"

CONTRACT="${SCRIPT_DIR}/fixtures/bonsai2_27b_mlx_v1_track.json"
[[ -f "${CONTRACT}" ]] || die "track contract fixture missing: ${CONTRACT}"

# The reference tree. No default: the control leg measured on a guessed tree is
# a band for a machine-and-tree pair nobody chose.
BASELINE_WORKSPACE="${MLXFAST_BASELINE_WORKSPACE:-}"
[[ -n "${BASELINE_WORKSPACE}" ]] \
  || die "MLXFAST_BASELINE_WORKSPACE is unset; the serial-control leg runs on the organizer-staged reference tree and there is nothing to calibrate without it"
[[ -d "${BASELINE_WORKSPACE}" ]] \
  || die "MLXFAST_BASELINE_WORKSPACE is not a directory: ${BASELINE_WORKSPACE}"

# The reference tree must be the one the contract pins: a band recorded against
# another commit is refused by tools/ranked-box-preflight.sh anyway, so refuse
# here rather than after the GPU window.
REFERENCE_COMMIT="$(jq -r '.baseline_reference_commit // ""' "${CONTRACT}")"
printf '%s' "${REFERENCE_COMMIT}" | grep -Eq '^[0-9a-f]{40}$' \
  || die "the track contract declares no 40-hex baseline_reference_commit (got '${REFERENCE_COMMIT}')"
workspace_head="$(git -C "${BASELINE_WORKSPACE}" rev-parse HEAD 2>/dev/null || true)"
[[ "${workspace_head}" == "${REFERENCE_COMMIT}" ]] \
  || die "the reference workspace is at '${workspace_head:-nothing}' but the contract pins ${REFERENCE_COMMIT}; re-stage it with tools/stage-baseline-workspace.sh"

# The track id. `benchd` requires it in every mode, and it is benchmark.json's
# value, never a guess -- the same resolution the measure script performs.
MANIFEST_TRACK_ID="$(jq -r '.trackId // empty' "${SCRIPT_DIR}/benchmark.json" 2>/dev/null || true)"
[[ -n "${MANIFEST_TRACK_ID}" ]] || die "benchmark.json carries no trackId"
if [[ -n "${MLXFAST_QWEN_MTP_TRACK_ID:-}" && "${MLXFAST_QWEN_MTP_TRACK_ID}" != "${MANIFEST_TRACK_ID}" ]]; then
  die "MLXFAST_QWEN_MTP_TRACK_ID is '${MLXFAST_QWEN_MTP_TRACK_ID}' but benchmark.json trackId is '${MANIFEST_TRACK_ID}'; the track id is ONE value"
fi
export MLXFAST_QWEN_MTP_TRACK_ID="${MANIFEST_TRACK_ID}"

# The live golden -- the ONE prompt both legs run, read FROM the fixture.
LIVE_GOLDEN_NAME="$(jq -r '.live_golden // empty' "${CONTRACT}")"
[[ -n "${LIVE_GOLDEN_NAME}" ]] || die "the fixture declares no live_golden"
GOLDEN_DIR="${MLXFAST_QWEN38_GOLDEN_DIR:-}"
[[ -n "${GOLDEN_DIR}" ]] \
  || die "MLXFAST_QWEN38_GOLDEN_DIR is unset; the live golden is staged on the box out of band and this script fetches nothing"
LIVE_GOLDEN_PATH="${GOLDEN_DIR}/${LIVE_GOLDEN_NAME}.golden.json"
[[ -f "${LIVE_GOLDEN_PATH}" ]] \
  || die "the live golden is not staged at ${LIVE_GOLDEN_PATH}"

# The PINNED benchd, resolved exactly as the ranked path resolves it.
BENCHD="${BENCHD:-${BENCHCTL:-}}"
if [[ -z "${BENCHD:-}" ]]; then
  BENCHD="$("${SCRIPT_DIR}/tools/fetch-benchd.sh")"
fi
[[ -x "${BENCHD}" ]] || die "benchd not found at ${BENCHD}; run ./tools/fetch-benchd.sh or set BENCHD"

# THE REFERENCE TREE'S OWN BUILD. benchd re-roots the relative engine path
# under the workspace, so the flag value is the workspace-relative path
# tools/stage-bench-worker.sh writes -- never an absolute one, and never this
# checkout's.
REFERENCE_ENGINE_REL=".build/release/bench-worker"
[[ -x "${BASELINE_WORKSPACE}/${REFERENCE_ENGINE_REL}" ]] \
  || die "the reference workspace has no executable worker at ${BASELINE_WORKSPACE}/${REFERENCE_ENGINE_REL}; build and stage it with tools/stage-baseline-workspace.sh"

# THE REFERENCE TREE'S OWN WEIGHTS, not this checkout's. The control leg is the
# pinned commit end to end: its engine, its Metal library and the tree its own
# transform produced.
REFERENCE_WEIGHTS="${BASELINE_WORKSPACE}/weights"
[[ -d "${REFERENCE_WEIGHTS}" ]] \
  || die "the reference workspace has no transformed weights at ${REFERENCE_WEIGHTS}; tools/stage-baseline-workspace.sh runs the transform there, exactly as the ranked job runs it for the candidate"

# The benchmarker's own identity, recorded in the band. The channel manifest
# beside the pinned binary is the source of truth; the environment name benchd
# itself reads is the fallback; neither present means the flag is omitted and
# benchd resolves it however it does.
BENCHD_MANIFEST="${BENCHD_BIN_DIR:-${SCRIPT_DIR}/benchd-bin}/benchd.manifest.json"
BENCHD_SOURCE_COMMIT="${MLXFAST_BENCHD_SOURCE_COMMIT:-}"
if [[ -z "${BENCHD_SOURCE_COMMIT}" && -f "${BENCHD_MANIFEST}" ]]; then
  BENCHD_SOURCE_COMMIT="$(jq -r '.source_commit // empty' "${BENCHD_MANIFEST}" 2>/dev/null || true)"
fi
BENCHD_SOURCE_ARGS=()
if [[ -n "${BENCHD_SOURCE_COMMIT}" ]]; then
  printf '%s' "${BENCHD_SOURCE_COMMIT}" | grep -Eq '^[0-9a-f]{40}$' \
    || die "the benchd source commit '${BENCHD_SOURCE_COMMIT}' is not 40 hex characters (from ${BENCHD_MANIFEST} or MLXFAST_BENCHD_SOURCE_COMMIT)"
  BENCHD_SOURCE_ARGS=(--benchd-source-commit "${BENCHD_SOURCE_COMMIT}")
fi

PASSES="${MLXFAST_CALIBRATION_PASSES:-4}"
printf '%s' "${PASSES}" | grep -Eq '^[1-9][0-9]*$' \
  || die "MLXFAST_CALIBRATION_PASSES must be a positive integer (got '${PASSES}')"

mkdir -p "$(dirname "${OUT_PATH}")"

GPU_LOCK_PATH="${RESIDENT_UP_LOCK_PATH:-/tmp/mtplx-gpu-exclusive.lock}"
GPU_LOCK_TIMEOUT_S="${MLXFAST_GPU_LOCK_TIMEOUT_S:-1800}"
printf '%s' "${GPU_LOCK_TIMEOUT_S}" | grep -Eq '^[1-9][0-9]*$' \
  || die "MLXFAST_GPU_LOCK_TIMEOUT_S must be a positive integer (got '${GPU_LOCK_TIMEOUT_S}')"

# resident-up.sh reads this name to check the lock is HELD before it boots a
# pass's resident, so the lock this script takes is the lock those boots check.
export RESIDENT_UP_LOCK_PATH="${GPU_LOCK_PATH}"

echo "calibrate-box.sh: taking the GPU lock ${GPU_LOCK_PATH}, then calibrating ${PASSES} pass(es) of the serial-control leg on ${BASELINE_WORKSPACE} over ${LIVE_GOLDEN_NAME}. benchd boots ONE resident per pass from the reference tree; this script boots none." >&2

# THE LOCK HOLDER IS THE OUTERMOST PROCESS of the calibration window. It holds an
# exclusive flock on an inheritable descriptor and then becomes benchd through
# execv, so the lock lives exactly as long as the measurement. A bounded wait,
# then a refusal -- never a calibration beside another GPU owner. Same shape,
# same lock and same timeout name as the ranked measurement.
python3 -c "
import fcntl, os, signal, sys
lock_path = sys.argv[1]
timeout_s = int(sys.argv[2])
argv = sys.argv[3:]
fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
os.set_inheritable(fd, True)
def on_timeout(*_):
    sys.stderr.write(
        'calibrate-box.sh: another owner has held ' + lock_path + ' for '
        + str(timeout_s) + 's; refusing to calibrate beside another GPU owner. Nothing has been loaded.\n')
    sys.exit(1)
signal.signal(signal.SIGALRM, on_timeout)
signal.alarm(timeout_s)
fcntl.flock(fd, fcntl.LOCK_EX)
signal.alarm(0)
os.execv(argv[0], argv)
" "${GPU_LOCK_PATH}" "${GPU_LOCK_TIMEOUT_S}" \
  "${BENCHD}" calibrate-baseline \
  --contract "${CONTRACT}" \
  --baseline-workspace "${BASELINE_WORKSPACE}" \
  --engine "${REFERENCE_ENGINE_REL}" \
  --weights "${REFERENCE_WEIGHTS}" \
  --golden "${LIVE_GOLDEN_PATH}" \
  --passes "${PASSES}" \
  --box "${BOX_NAME}" \
  --track "${MANIFEST_TRACK_ID}" \
  --prompt "${LIVE_GOLDEN_NAME}" \
  --reference-commit "${REFERENCE_COMMIT}" \
  ${BENCHD_SOURCE_ARGS[@]+"${BENCHD_SOURCE_ARGS[@]}"} \
  --out "${OUT_PATH}"

[[ -f "${OUT_PATH}" ]] \
  || die "benchd wrote no calibration file at ${OUT_PATH}; a CV above 1 % on either axis is a refusal, not a warning -- the box did not repeat itself"

echo "calibrate-box.sh: wrote ${OUT_PATH}"
echo "calibrate-box.sh: export it as MLXFAST_BASELINE_CALIBRATION in the runner service environment."
cat "${OUT_PATH}"
