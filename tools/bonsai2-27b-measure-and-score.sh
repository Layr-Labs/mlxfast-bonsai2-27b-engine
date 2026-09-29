#!/usr/bin/env bash
#
# bonsai2-27b-measure-and-score.sh -- benchmark.json's benchmarkCommand /
# preSubmitCommand entry point for track bonsai2-27b-mlx-v1.
#
# It drives `benchd iterate --mode official` -- the SOLE scored path -- against
# the track's LIVE goldens. `iterate --mode official` is timed-first, spawns the
# sandboxed bench-worker engine, runs the full correctness set, gates on the
# official floor/bands, and SEALS the artifact itself: it writes score.json in
# the {score, metrics} shape Yukon's ScoreFileSchema reads, its `.sha256`
# sidecar, and the per-mode benchmark-integrity sidecar. There is NO results.json
# to convert -- benchd is the sole writer of the sealed score, so this script no
# longer post-processes one. This script is TRUSTED-side tooling: it is NOT in
# editablePaths, so a submission cannot rewrite the measurement pipeline from
# inside its own archive.
#
# THE RUN IS PAIRED, AND THE PAIR IS PER BOX (David ruling 2026-09-08). A ranked
# run measures TWO legs on the same box in the same job, over each live
# golden:
#
#   1. the SERIAL-CONTROL leg, on the organizer-staged REFERENCE tree
#      (MLXFAST_BASELINE_WORKSPACE), with no speculation;
#   2. the CANDIDATE leg, on this submission tree at its declared draft depth.
#
# composite = (ref_prefill_spt / cand_prefill_spt)^0.25
#           * (ref_decode_spt  / cand_decode_spt )^0.75
#
# so the denominator is MEASURED on this box, in this job, minutes apart from
# the numerator. NOTHING STORES A PAIR: not the scoring constants, not the
# fixture, not the golden. A golden that carries
# benchmark.baseline_*_seconds_per_token is refused on the ranked path
# (tools/lint-benchmark-manifest.py check 5b keeps the field out of the tree).
#
# THE ENGINE PATH IS RELATIVE, AND THAT IS LOAD-BEARING. benchd boots nothing
# extra on MLX: it RE-ROOTS the candidate's `--engine` path under the reference
# workspace to find the control leg's worker, so `--engine
# .build/release/bench-worker` on this checkout means
# <baseline workspace>/.build/release/bench-worker on the reference tree. An
# absolute `--engine` has nothing to re-root and would run the CANDIDATE binary
# for both legs, which is a score of 1 by construction. So this script passes
# the engine RELATIVE to the checkout root, and refuses when the resolved engine
# lies outside that root.
#
# MLXFAST_BASELINE_CALIBRATION is a HEALTH BAND for leg 1, NEVER the
# denominator: this box's calibrator recorded what its own serial-control leg
# costs, and benchd dies by name when leg 1 lands outside that band rather than
# sealing a score against a box that was not itself. tools/calibrate-box.sh
# writes the file; tools/ranked-box-preflight.sh verifies both variables before
# any measurement.
#
# WHY iterate, not measure-job. `benchd measure-job` is RETIRED. `benchd iterate
# --mode official` drives both legs itself from --baseline-workspace and seals
# the artifact. This is the SAME methodology the vendored facade
# tools/benchmark.sh runs its --official path through.
#
# ONE RESIDENT bench-worker PER LEG, AND benchd BOOTS IT (weights load once per
# leg; David 2026-08-30 as the paired design realises it). benchd spawns
# `bench-worker runtime-worker` once per phase -- warmup, timed prefill, timed
# decode, correctness -- and an in-process spawn loads the whole checkpoint each
# time, so the weights need an OWNER. Under the paired design that owner is
# PER LEG, because the two legs run on two different trees with two different
# weight directories.
#
# THE PREVIOUS SHAPE WAS WRONG, and public run 34230122059 proved it. This
# script used to boot ONE resident from the CANDIDATE tree and export
# BENCH_WORKER_RESIDENT_SOCKET into benchd; the reference leg attached to that
# resident and benchd refused it -- "resident holds <candidate>/weights but this
# phase asked for <baseline-workspace>/weights". Correct refusal, wrong
# topology.
#
# So this script boots NOTHING and exports NO SOCKET. benchd calls
# `tools/resident-up.sh --boot --spec <serial|mtp> --draft-len <N>
# --socket-out <file>` in the leg's OWN tree, measures the leg against the
# socket that boot reports, and then calls `--stop --socket <path>`. That is
# the same convention the CUDA track's serve-up follows, with the same argv.
#
# AN INHERITED BENCH_WORKER_RESIDENT_SOCKET IS A REFUSAL here, and
# tools/ranked-box-preflight.sh refuses it earlier still: a socket arriving
# through the runner service environment would reproduce the same failure
# without this script ever booting anything.
#
# THIS SCRIPT STILL OWNS THE GPU WINDOW. A resident holds ~18.5 GB of unified
# memory whoever booted it, so the box must have exactly one loader for the
# whole measurement. This script takes the fleet GPU lock
# (/tmp/mtplx-gpu-exclusive.lock) FIRST and re-executes itself inside it, and
# resident-up.sh REFUSES to boot when nobody holds that lock -- so every per-leg
# boot benchd makes happens inside this window. resident-up.sh never takes the
# lock itself, because a lock it took would end when it exits, which is not the
# window.
#
# The LIVE goldens are read FROM the fixture, never hardcoded: the fixture's
# `live_goldens` lists them and each name's `timed_prompt_pool[]` entry pins it
# ({sha256, bytes}). A change to the list is picked up here with no edit to
# this script. A ranked run measures each live golden once, one pair per
# golden, and benchd scores the lower-median pair. The pins are forwarded to
# benchd as --golden-sha256/--golden-bytes, once per golden, and benchd
# re-verifies the raw bytes BEFORE parse and refuses on any mismatch (the
# integrity pin). --contract carries the arm
# gate: benchd refuses, pre-GPU, to seal an official artifact unless the fixture
# declares official_scoring_enabled: true.
#
# The design rationale lives in fixtures/bonsai2_27b_mlx_v1_track.json
# scoring_semantics and docs/bonsai2-27b-port-notes.md section 4.
#
# Usage:
#   ./tools/bonsai2-27b-measure-and-score.sh                 # full measure + seal
#   ./tools/bonsai2-27b-measure-and-score.sh --preflight-only # pre-GPU dry-run, no engine
#
# Env:
#   MLXFAST_SCORE_PATH               Where benchd seals the {score, metrics} JSON
#                                     (--score-path). Defaults to score.json
#                                     (benchmark.json always sets this explicitly).
#                                     The .sha256 sidecar and benchmark-integrity
#                                     sidecar are sealed beside it by benchd.
#   MLXFAST_BASELINE_WORKSPACE       The organizer-staged REFERENCE tree the
#                                     serial-control leg runs on -- a built
#                                     checkout of THIS repository at the
#                                     fixture's baseline_reference_commit.
#                                     REQUIRED on a real run; this script
#                                     refuses by name when it is unset, because
#                                     an absent reference tree has no control
#                                     leg and therefore no denominator.
#                                     tools/stage-baseline-workspace.sh builds
#                                     it on the box.
#   MLXFAST_BASELINE_CALIBRATION     This box's baseline-calibration.json --
#                                     the HEALTH BAND for the serial-control
#                                     leg, written by tools/calibrate-box.sh.
#                                     REQUIRED on a real run, same refusal.
#                                     benchd fails the run by name when leg 1
#                                     falls outside the band; it never uses the
#                                     band as a denominator.
#   MLXFAST_QWEN38_GOLDEN_DIR         Directory holding the staged timed-pool
#                                     golden files. Each LIVE golden is resolved
#                                     from it as <name>.golden.json (the
#                                     basename of the fixture entry's r2_path).
#                                     Box-only, staged out of band and
#                                     pin-verified by tools/ranked-box-preflight.sh.
#                                     The name is the FLEET's golden-dir contract,
#                                     which every box script exports; it is not
#                                     this track's name and it is not renamed here.
#   MLXFAST_ENGINE_BIN               The Engine Protocol v1 engine benchd spawns
#                                     (`--engine`): the fork's generic
#                                     bench-worker. Default: the fixed staged
#                                     path tools/stage-bench-worker.sh writes,
#                                     .build/release/bench-worker (./setup.sh
#                                     stages it there). benchd spawns the engine
#                                     directly -- there is no separate serve.
#   MLXFAST_WEIGHTS_PATH             Transformed weights directory, passed as
#                                     `--weights`. Default: ./weights -- the SAME
#                                     convention tools/benchmark.sh uses and the
#                                     directory ./setup.sh transforms into.
#   MLXFAST_GPU_WINDOW_HELD          Set to 1 by this script when it re-executes
#                                     itself inside the GPU lock it just took.
#                                     It marks the inner run, so the script does
#                                     not wait for a lock it already holds. Do
#                                     not set it by hand: a hand-set 1 means the
#                                     measurement runs with NO lock held, and
#                                     every per-leg resident boot then refuses.
#   RESIDENT_UP_LOCK_PATH            The GPU lock this script holds for the
#                                     whole measurement window. Default:
#                                     /tmp/mtplx-gpu-exclusive.lock. Same name
#                                     and same default tools/resident-up.sh
#                                     reads, so the lock this script holds is
#                                     the lock that script checks.
#   MLXFAST_GPU_LOCK_TIMEOUT_S       Ceiling on the wait for that lock
#                                     (default 1800). On the ceiling the run
#                                     refuses; it never measures beside
#                                     another owner of the GPU.
#   BENCHD                         Path to the benchd binary. Default: the
#                                     binary ./tools/fetch-benchd.sh resolves from
#                                     the dist channel and verifies against the
#                                     channel's benchd.manifest.json
#                                     (benchd-bin/benchd). No cargo build: the
#                                     ranked box has no Rust toolchain, so benchd
#                                     ships prebuilt and pinned by sha256.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

PREFLIGHT_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *)
      echo "bonsai2-27b-measure-and-score.sh: unrecognized argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

# jq is required for reading the fixture (live_goldens + pins) and the trackId.
if ! command -v jq >/dev/null 2>&1; then
  echo "bonsai2-27b-measure-and-score.sh: jq is required (it reads the fixture pins and trackId)." >&2
  exit 1
fi

# Resolve the PINNED benchd. benchd ships as a channel PREBUILT
# (./tools/fetch-benchd.sh, verified against the channel's
# benchd.manifest.json; the sha pin is retired -- David ruling 2026-08-27, so
# measurement fixes ship bench-side with no engine commit). fetch-benchd.sh
# accepts an already-present benchd-bin/benchd whose sha256 and bytes match the
# manifest beside it -- the offline path on the ranked box, which has no Rust
# toolchain -- and otherwise downloads and verifies it. It never yields an
# unverified binary, so this refuses rather than measuring against unpinned
# scoring code.
#
# BENCHD= from the caller is honoured and NOT hash-checked: that is a
# deliberate "use this other binary" for benchd development.
# BENCHCTL is the pre-rename spelling; honoured so a box .env written before
# the benchd rename keeps working.
BENCHD="${BENCHD:-${BENCHCTL:-}}"
if [[ -z "${BENCHD:-}" ]]; then
  BENCHD="$("${SCRIPT_DIR}/tools/fetch-benchd.sh")"
fi
if [[ ! -x "${BENCHD}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: benchd not found at ${BENCHD}." >&2
  echo "  fetch it: ./tools/fetch-benchd.sh   (resolves the dist channel, verifies the manifest)" >&2
  echo "  or set BENCHD to an existing binary." >&2
  exit 1
fi

CONTRACT="${SCRIPT_DIR}/fixtures/bonsai2_27b_mlx_v1_track.json"
if [[ ! -f "${CONTRACT}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: track contract fixture missing: ${CONTRACT}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# THE TRACK ID. `benchd iterate` REQUIRES MLXFAST_QWEN_MTP_TRACK_ID in every
# mode -- its `-{platform}-v{N}` suffix keys the OFFICIAL_BASELINE pair and the
# acceptance bands, and the official path resolves it BEFORE the timed run. The
# value is benchmark.json's `trackId` (the PLATFORM id). benchd treats
# env != contract as a hard error, so a caller that pre-set a DIFFERENT value is
# refused here rather than silently overridden.
#
# THE VARIABLE NAME IS THE HARNESS PROTOCOL'S, NOT THIS TRACK'S. benchd reads
# this exact spelling on every track it scores, so it stays as it is.
MANIFEST_TRACK_ID="$(jq -r '.trackId // empty' "${SCRIPT_DIR}/benchmark.json" 2>/dev/null || true)"
if [[ -z "${MANIFEST_TRACK_ID}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: benchmark.json carries no trackId; benchd iterate requires MLXFAST_QWEN_MTP_TRACK_ID and this script will not guess one." >&2
  exit 1
fi
if [[ -n "${MLXFAST_QWEN_MTP_TRACK_ID:-}" && "${MLXFAST_QWEN_MTP_TRACK_ID}" != "${MANIFEST_TRACK_ID}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: MLXFAST_QWEN_MTP_TRACK_ID is set to '${MLXFAST_QWEN_MTP_TRACK_ID}' but benchmark.json trackId is '${MANIFEST_TRACK_ID}'; the track id is ONE value and this script will not override either." >&2
  exit 1
fi
export MLXFAST_QWEN_MTP_TRACK_ID="${MANIFEST_TRACK_ID}"

# ---------------------------------------------------------------------------
# THE LIVE GOLDENS + THEIR PINS, read FROM the fixture (never hardcoded). The
# fixture's `live_goldens` lists the goldens this track scores over, in order.
# Each name's timed_prompt_pool[] entry -- matched by the basename of its
# r2_path (<name>.golden.json) -- carries the {sha256, bytes} pin. A change to
# the list needs no edit here.
#
# benchd measures one pair per golden: pair k measures golden (k - 1) mod N,
# and the score is the lower-median pair. Every golden flag below is repeated
# once per live golden, in fixture order, and benchd matches the flags by
# position.
LIVE_GOLDEN_NAMES=()
while IFS= read -r name; do
  LIVE_GOLDEN_NAMES+=("${name}")
done < <(jq -r '(.live_goldens // [])[]' "${CONTRACT}")
if [[ ${#LIVE_GOLDEN_NAMES[@]} -eq 0 ]]; then
  echo "bonsai2-27b-measure-and-score.sh: fixture declares no live_goldens; there is no golden to score over." >&2
  exit 1
fi

# PER-DEPTH ORACLE (David ruling 2026-09-07, the CUDA track's shape). A window
# verified in one target forward does not reproduce the serial tape token for
# token (the kernels differ at M > 1), so a speculative declaration scores
# against the oracle RECORDED AT THAT DEPTH: fixture live_golden_speculative
# maps "mtpN" to one pinned oracle per live golden, keyed by the golden's name.
# A declared depth with no authored oracle for a live golden is refused below
# rather than scored against the serial tape.
SPEC_DESC="$("${SCRIPT_DIR}/tools/spec-declaration.sh" describe)"

# The goldens are box-only, staged out of band into MLXFAST_QWEN38_GOLDEN_DIR
# and pin-verified by tools/ranked-box-preflight.sh.
GOLDEN_DIR="${MLXFAST_QWEN38_GOLDEN_DIR:-}"

# For each live golden: the timed oracle the candidate leg verifies against
# (the serial tape, or the per-depth oracle), and the serial tape the
# serial-control leg verifies against. The serial-control leg (leg 1) is
# serial by construction, so benchd verifies it against the serial tape
# (--control-golden); the per-depth tape is the candidate leg's oracle only. On
# the MLX engine the depth-1 tape forks from the serial tape at step 1, so a
# control leg checked against it dies at step 1.
GOLDEN_BASENAMES=()
GOLDEN_SHA256S=()
GOLDEN_BYTES_LIST=()
SERIAL_GOLDEN_BASENAMES=()
SERIAL_GOLDEN_SHA256S=()
SERIAL_GOLDEN_BYTES_LIST=()
for name in "${LIVE_GOLDEN_NAMES[@]}"; do
  serial_base="${name}.golden.json"
  # The pool entry whose r2_path ends in /<name>.golden.json. Its pin is the
  # integrity pin forwarded to benchd.
  pool_entry="$(jq -c --arg base "/${serial_base}" \
    'first(.timed_prompt_pool[] | select(.r2_path | endswith($base)))' "${CONTRACT}")"
  if [[ -z "${pool_entry}" || "${pool_entry}" == "null" ]]; then
    echo "bonsai2-27b-measure-and-score.sh: live golden '${name}' has no timed_prompt_pool entry (looked for an r2_path ending in /${serial_base})." >&2
    exit 1
  fi
  serial_sha="$(printf '%s' "${pool_entry}" | jq -r '.sha256 // empty')"
  serial_bytes="$(printf '%s' "${pool_entry}" | jq -r '.bytes // empty')"
  timed_base="${serial_base}"
  timed_sha="${serial_sha}"
  timed_bytes="${serial_bytes}"
  if [[ "${SPEC_DESC}" != "serial" ]]; then
    spec_entry="$(jq -c --arg k "${SPEC_DESC}" --arg n "${name}" '.live_golden_speculative[$k][$n] // empty' "${CONTRACT}")"
    if [[ -z "${spec_entry}" || "${spec_entry}" == "null" ]]; then
      echo "bonsai2-27b-measure-and-score.sh: declared spec '${SPEC_DESC}' has no live_golden_speculative entry for live golden '${name}' in ${CONTRACT}; no timed oracle is authored for that prompt at that draft depth. Refusing rather than scoring it against the serial oracle." >&2
      exit 1
    fi
    timed_base="$(printf '%s' "${spec_entry}" | jq -r '.r2_path // empty')"
    timed_base="${timed_base##*/}"
    timed_sha="$(printf '%s' "${spec_entry}" | jq -r '.sha256 // empty')"
    timed_bytes="$(printf '%s' "${spec_entry}" | jq -r '.bytes // empty')"
    if [[ -z "${timed_base}" || "${timed_base}" != *.golden.json ]]; then
      echo "bonsai2-27b-measure-and-score.sh: live_golden_speculative['${SPEC_DESC}']['${name}'] names no *.golden.json (r2_path='${timed_base}')." >&2
      exit 1
    fi
    echo "bonsai2-27b-measure-and-score.sh: declared spec ${SPEC_DESC}; timed oracle for ${name} is ${timed_base}" >&2
  fi
  if ! printf '%s' "${timed_sha}" | grep -Eq '^[0-9a-f]{64}$' \
    || ! printf '%s' "${timed_bytes}" | grep -Eq '^[1-9][0-9]*$'; then
    echo "bonsai2-27b-measure-and-score.sh: live golden '${name}' is unarmed or malformed (sha256='${timed_sha}', bytes='${timed_bytes}'); nothing can be pin-verified against it." >&2
    exit 1
  fi
  GOLDEN_BASENAMES+=("${timed_base}")
  GOLDEN_SHA256S+=("${timed_sha}")
  GOLDEN_BYTES_LIST+=("${timed_bytes}")
  SERIAL_GOLDEN_BASENAMES+=("${serial_base}")
  SERIAL_GOLDEN_SHA256S+=("${serial_sha}")
  SERIAL_GOLDEN_BYTES_LIST+=("${serial_bytes}")
done

# ---------------------------------------------------------------------------
# --preflight-only: a pre-GPU DRY RUN that exercises the arm gate and (when the
# goldens are staged) the integrity pins, WITHOUT spawning the engine or loading
# the model. No score is written. The full run below enforces both refusals
# through benchd itself; this is the loud early gate.
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  # ARM GATE (mirror of benchd's enforce_official_scoring_enabled): an official
  # run refuses unless the fixture declares official_scoring_enabled: true. false
  # and ABSENT both refuse -- an absent arm state is not an armed one. benchd
  # enforces this itself on the real run; mirroring it here makes the pre-GPU
  # dry-run honest rather than a rubber stamp.
  armed="$(jq -r '.official_scoring_enabled // false' "${CONTRACT}")"
  if [[ "${armed}" != "true" ]]; then
    echo "bonsai2-27b-measure-and-score.sh: preflight REFUSING -- fixtures/bonsai2_27b_mlx_v1_track.json does not declare official_scoring_enabled: true (got '${armed}')." >&2
    echo "  benchd refuses to seal an official scoring artifact for an unarmed track; this dry-run mirrors that refusal." >&2
    exit 1
  fi
  echo "bonsai2-27b-measure-and-score.sh: preflight -- arm gate OK (official_scoring_enabled: true), live goldens ${LIVE_GOLDEN_NAMES[*]}" >&2

  # INTEGRITY PIN: for each staged live golden, validate-golden re-verifies its
  # raw bytes against the pin AND load-validates it (reference-model pin from
  # --contract), with NO engine spawned. When one is not staged (e.g. an off-box
  # pre-submit), the pin is still enforced by benchd on the real run.
  for i in "${!GOLDEN_BASENAMES[@]}"; do
    if [[ -n "${GOLDEN_DIR}" && -f "${GOLDEN_DIR}/${GOLDEN_BASENAMES[$i]}" ]]; then
      "${BENCHD}" validate-golden \
        --golden "${GOLDEN_DIR}/${GOLDEN_BASENAMES[$i]}" \
        --golden-sha256 "${GOLDEN_SHA256S[$i]}" \
        --golden-bytes "${GOLDEN_BYTES_LIST[$i]}" \
        --contract "${CONTRACT}"
    else
      echo "bonsai2-27b-measure-and-score.sh: preflight -- live golden not staged (MLXFAST_QWEN38_GOLDEN_DIR unset or ${GOLDEN_BASENAMES[$i]} absent); benchd re-verifies the {sha256, bytes} pin on the real run." >&2
    fi
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# A REAL RUN needs every staged live golden.
if [[ -z "${GOLDEN_DIR}" || ! -d "${GOLDEN_DIR}" ]]; then
  cat >&2 <<EOF
bonsai2-27b-measure-and-score.sh: MLXFAST_QWEN38_GOLDEN_DIR is unset or missing.
  The live goldens (${GOLDEN_BASENAMES[*]}) are staged onto the box out of band
  (docs/bonsai2-27b-port-notes.md section 5) and this job holds no credential
  to fetch them. There is nothing this script can do here except refuse.
EOF
  exit 1
fi
GOLDEN_ARGS=()
GOLDEN_PIN_ARGS=()
for i in "${!GOLDEN_BASENAMES[@]}"; do
  if [[ ! -f "${GOLDEN_DIR}/${GOLDEN_BASENAMES[$i]}" ]]; then
    echo "bonsai2-27b-measure-and-score.sh: live golden not found at ${GOLDEN_DIR}/${GOLDEN_BASENAMES[$i]}" >&2
    echo "  live_goldens names '${LIVE_GOLDEN_NAMES[$i]}' (fixtures/bonsai2_27b_mlx_v1_track.json); stage ${GOLDEN_BASENAMES[$i]} into MLXFAST_QWEN38_GOLDEN_DIR." >&2
    exit 1
  fi
  GOLDEN_ARGS+=(--golden "${GOLDEN_DIR}/${GOLDEN_BASENAMES[$i]}")
done
for sha in "${GOLDEN_SHA256S[@]}"; do
  GOLDEN_PIN_ARGS+=(--golden-sha256 "${sha}")
done
for bytes in "${GOLDEN_BYTES_LIST[@]}"; do
  GOLDEN_PIN_ARGS+=(--golden-bytes "${bytes}")
done

# ---------------------------------------------------------------------------
# THE PAIRED LEG PAIR. Both names are REQUIRED on a real run and neither has a
# default: a missing reference tree means no serial-control leg and so no
# denominator, and a missing calibration file means leg 1 has no band to be
# judged healthy against. Refuse by name, here, before the GPU lock is taken --
# not deeper, where the refusal would cost a window.
#
# --preflight-only does NOT require them. That path is benchmark.json's
# preSubmitCommand: a participant runs it off the box, where no reference tree
# is staged and none can be.
BASELINE_WORKSPACE="${MLXFAST_BASELINE_WORKSPACE:-}"
BASELINE_CALIBRATION="${MLXFAST_BASELINE_CALIBRATION:-}"
if [[ -z "${BASELINE_WORKSPACE}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: MLXFAST_BASELINE_WORKSPACE is unset." >&2
  echo "  The ranked run is PAIRED: the serial-control leg runs on the organizer-staged reference tree at the fixture's baseline_reference_commit, and its measured spt is the denominator. Without that tree there is no control leg and no score." >&2
  echo "  Stage it with tools/stage-baseline-workspace.sh and export MLXFAST_BASELINE_WORKSPACE in the runner service environment." >&2
  exit 1
fi
if [[ ! -d "${BASELINE_WORKSPACE}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: MLXFAST_BASELINE_WORKSPACE is not a directory: ${BASELINE_WORKSPACE}" >&2
  exit 1
fi
if [[ -z "${BASELINE_CALIBRATION}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: MLXFAST_BASELINE_CALIBRATION is unset." >&2
  echo "  It names this box's baseline-calibration.json, the health band the serial-control leg must land inside. Without it a leg-1 measurement from a box that was not itself would seal a score anyway." >&2
  echo "  Write it with tools/calibrate-box.sh and export MLXFAST_BASELINE_CALIBRATION in the runner service environment." >&2
  exit 1
fi
if [[ ! -f "${BASELINE_CALIBRATION}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: MLXFAST_BASELINE_CALIBRATION is not a file: ${BASELINE_CALIBRATION}" >&2
  exit 1
fi

# --engine. `iterate` spawns the engine (Engine Protocol v1) directly as
# `<engine> runtime-worker --weights <dir>`. The engine is the fork's generic
# bench-worker. Default to the fixed staged path tools/stage-bench-worker.sh
# writes, so the documented invocation is self-sufficient; MLXFAST_ENGINE_BIN
# overrides it for engine development.
ENGINE_BIN="${MLXFAST_ENGINE_BIN:-${SCRIPT_DIR}/.build/release/bench-worker}"
if [[ ! -x "${ENGINE_BIN}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: engine binary missing or not executable: ${ENGINE_BIN}" >&2
  echo "  build + stage it (./setup.sh, or tools/stage-bench-worker.sh), or set MLXFAST_ENGINE_BIN." >&2
  exit 1
fi

# THE RELATIVE FORM benchd is handed. benchd re-roots this path under the
# reference workspace to locate the control leg's worker, so it must be
# workspace-relative AND must resolve inside this checkout. An engine somewhere
# else on the box cannot be re-rooted, and passing it would silently run the
# CANDIDATE binary as the control -- a score of 1 by construction. Refuse.
ENGINE_BIN_ABS="$(cd "$(dirname "${ENGINE_BIN}")" && pwd -P)/$(basename "${ENGINE_BIN}")"
SCRIPT_DIR_ABS="$(cd "${SCRIPT_DIR}" && pwd -P)"
case "${ENGINE_BIN_ABS}" in
  "${SCRIPT_DIR_ABS}"/*)
    ENGINE_BIN_REL="${ENGINE_BIN_ABS#"${SCRIPT_DIR_ABS}"/}"
    ;;
  *)
    echo "bonsai2-27b-measure-and-score.sh: the engine at ${ENGINE_BIN_ABS} is outside this checkout (${SCRIPT_DIR_ABS})." >&2
    echo "  The paired run hands benchd a checkout-RELATIVE engine path and benchd re-roots it under MLXFAST_BASELINE_WORKSPACE to find the serial-control leg's worker. An engine outside the checkout has no relative form, so the control leg would run this same binary and the score would be 1 by construction." >&2
    exit 1
    ;;
esac

# --weights. benchd resolves the transformed weights from `--weights`. Default to
# ./weights -- the convention tools/benchmark.sh uses and the directory ./setup.sh
# transforms into; MLXFAST_WEIGHTS_PATH overrides.
WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-${SCRIPT_DIR}/weights}"
if [[ ! -d "${WEIGHTS_PATH}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: the transformed weights directory is missing: ${WEIGHTS_PATH}" >&2
  echo "  Run ./setup.sh, or set MLXFAST_WEIGHTS_PATH to the transformed weights directory." >&2
  exit 1
fi

# THE WEIGHTS ARE RE-ROOTED THE SAME WAY THE ENGINE IS, so they carry the same
# constraint. The control leg runs on the reference tree's OWN transform, which
# benchd finds by re-rooting this path under MLXFAST_BASELINE_WORKSPACE. A
# weights directory somewhere else on the box has no checkout-relative form, so
# the control leg would read the CANDIDATE's transformed tree -- a submission's
# own transform on both sides of the ratio, which is the one thing a control leg
# exists to prevent. Refuse, by name, exactly as for the engine.
WEIGHTS_PATH_ABS="$(cd "${WEIGHTS_PATH}" && pwd -P)"
case "${WEIGHTS_PATH_ABS}" in
  "${SCRIPT_DIR_ABS}"/*)
    : # inside the checkout, so it has a relative form to re-root
    ;;
  *)
    echo "bonsai2-27b-measure-and-score.sh: the transformed weights at ${WEIGHTS_PATH_ABS} are outside this checkout (${SCRIPT_DIR_ABS})." >&2
    echo "  benchd re-roots this path under MLXFAST_BASELINE_WORKSPACE to find the serial-control leg's own transformed weights. A weights directory outside the checkout has no relative form, so the control leg would read this submission's transform and the ratio would carry it on both sides." >&2
    exit 1
    ;;
esac

SCORE_PATH="${MLXFAST_SCORE_PATH:-score.json}"

# ---------------------------------------------------------------------------
# THE GPU WINDOW. Everything above resolved WITHOUT loading anything: the
# pinned benchd, the arm gate, the goldens and their pins, the engine binary, the
# weights. This is the last point before benchd starts
# spawning workers, so this is where the window opens.
#
# THIS SCRIPT NO LONGER BOOTS A RESIDENT, AND THAT IS THE FIX (public run
# 34230122059). It used to boot ONE resident from the CANDIDATE tree and export
# BENCH_WORKER_RESIDENT_SOCKET into benchd. Under the paired design the
# REFERENCE leg then attached to the candidate's resident and was refused:
# "resident holds <candidate>/weights but this phase asked for
# <baseline-workspace>/weights". One resident per WINDOW is wrong when a window
# has two legs on two trees.
#
# ONE RESIDENT PER LEG, BOOTED BY benchd. benchd knows where a leg begins and
# ends, so benchd boots that leg's resident by calling
# `tools/resident-up.sh --boot` IN THAT LEG'S OWN TREE and stops it with
# `--stop` when the leg is done. This script hands benchd no socket at all.
#
# WHAT THIS SCRIPT STILL OWNS IS THE LOCK. A resident holds ~18.5 GB of unified
# memory whoever boots it, so the box GPU window must still be exclusive for the
# whole measurement. This script takes /tmp/mtplx-gpu-exclusive.lock and
# re-executes itself inside it; resident-up.sh REFUSES to boot when nobody
# holds that lock, so every per-leg boot benchd makes is inside this window.
#
# AN INHERITED SOCKET IS A REFUSAL. A BENCH_WORKER_RESIDENT_SOCKET in the
# environment would send every phase of BOTH legs to one already-loaded
# resident -- exactly the failure above, arriving through the runner service
# environment instead of through this script. tools/ranked-box-preflight.sh
# refuses it before the job reaches here; this is the second line.
if [[ -n "${BENCH_WORKER_RESIDENT_SOCKET:-}" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: BENCH_WORKER_RESIDENT_SOCKET is set (${BENCH_WORKER_RESIDENT_SOCKET})." >&2
  echo "  The paired run boots ONE resident PER LEG, and benchd boots each from that leg's own tree. A socket in the environment would attach every phase of both legs to one already-loaded resident, so the reference leg would run on the candidate's weights. Unset it." >&2
  exit 1
fi

# MLXFAST_GPU_WINDOW_HELD marks the inner run: this process is already inside
# the lock it took below. Without the marker the script would re-take a lock it
# already holds and wait for itself.
if [[ "${MLXFAST_GPU_WINDOW_HELD:-0}" != "1" ]]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "bonsai2-27b-measure-and-score.sh: python3 is required to hold the GPU lock for the measurement window." >&2
    exit 1
  fi
  GPU_LOCK_PATH="${RESIDENT_UP_LOCK_PATH:-/tmp/mtplx-gpu-exclusive.lock}"
  GPU_LOCK_TIMEOUT_S="${MLXFAST_GPU_LOCK_TIMEOUT_S:-1800}"
  if ! printf '%s' "${GPU_LOCK_TIMEOUT_S}" | grep -Eq '^[1-9][0-9]*$'; then
    echo "bonsai2-27b-measure-and-score.sh: MLXFAST_GPU_LOCK_TIMEOUT_S must be a positive integer (got '${GPU_LOCK_TIMEOUT_S}')." >&2
    exit 1
  fi
  # resident-up.sh reads this name to check the lock is HELD before it boots a
  # leg's resident, so the lock this script takes is the lock those boots check.
  export RESIDENT_UP_LOCK_PATH="${GPU_LOCK_PATH}"
  export MLXFAST_GPU_WINDOW_HELD=1
  echo "bonsai2-27b-measure-and-score.sh: taking the GPU lock ${GPU_LOCK_PATH} for the whole measurement. benchd boots ONE resident per leg inside it (tools/resident-up.sh --boot), from that leg's own tree." >&2
  # The lock holder is the OUTERMOST process of the window: it holds an
  # exclusive flock on an inheritable descriptor and then becomes this script
  # again through execv, so the lock lives exactly as long as the measurement. A
  # bounded wait, then a refusal -- never a measurement beside another owner.
  exec python3 -c "
import fcntl, os, signal, sys
lock_path = sys.argv[1]
timeout_s = int(sys.argv[2])
argv = sys.argv[3:]
fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
os.set_inheritable(fd, True)
def on_timeout(*_):
    sys.stderr.write(
        \"bonsai2-27b-measure-and-score.sh: another owner has held \"
        + lock_path
        + \" for \" + str(timeout_s) + \"s; refusing to measure beside another GPU owner. Nothing has been loaded.\\n\")
    sys.exit(1)
signal.signal(signal.SIGALRM, on_timeout)
signal.alarm(timeout_s)
fcntl.flock(fd, fcntl.LOCK_EX)
signal.alarm(0)
os.execv(argv[0], argv)
" "${GPU_LOCK_PATH}" "${GPU_LOCK_TIMEOUT_S}" \
    "${SCRIPT_DIR}/tools/bonsai2-27b-measure-and-score.sh" "$@"
fi

# THE SOLE SCORED PATH: benchd iterate --mode official over the live goldens.
# --baseline-workspace names the reference tree benchd runs the serial-control
# leg on, and --baseline-calibration names this box's health band for that leg.
# The score is the composite prefill_gain^0.25 * decode_gain^0.75 over the two
# LIVE legs. benchd SEALS score.json (the
# {score, metrics} shape Yukon reads), score.json.sha256, and the
# benchmark-integrity sidecar itself; this script does no post-conversion.
# benchd spawns the bench-worker engine directly via --engine; there is no serve.
#
# --golden-sha256/--golden-bytes are the INTEGRITY PINS, one per --golden and
# in the same order (benchd re-verifies the raw bytes before parse and refuses
# on mismatch). --contract carries the ARM
# GATE (benchd refuses, pre-GPU, unless official_scoring_enabled: true).
# THE DECLARED DECODER AND DRAFT DEPTH. tools/spec-declaration.sh is the single
# trusted reader of the participant's mtp-head.manifest.json `spec` block; it
# refuses (exit 1, before any engine spawns) on a malformed or out-of-envelope
# declaration. A serial declaration (absent, disabled, or 0) sends NO spec, and
# the engine runs depth 0 -- the baseline validation's leg.
#
# An `mtp` depth N sends `--mtp-depth N`. A `dflash` depth N sends the explicit
# candidate spec, because benchd's depth convenience flag builds an MTP spec
# only. Either way the engine echoes effective_spec and the seal carries it.
DECLARED_DECODER="$("${SCRIPT_DIR}/tools/spec-declaration.sh" decoder)"
DECLARED_DEPTH="$("${SCRIPT_DIR}/tools/spec-declaration.sh" draft-len)"
SPEC_ARGS=()
if [[ "${DECLARED_DEPTH}" != "0" ]]; then
  case "${DECLARED_DECODER}" in
    mtp)
      if "${BENCHD}" iterate --help 2>&1 | grep -q -- '--mtp-depth'; then
        SPEC_ARGS=(--mtp-depth "${DECLARED_DEPTH}")
        echo "bonsai2-27b-measure-and-score.sh: the declaration requests MTP depth ${DECLARED_DEPTH}; benchd sends it on the wire and the engine's effective_spec echo is sealed." >&2
      else
        echo "bonsai2-27b-measure-and-score.sh: REFUSING -- the declaration requests MTP depth ${DECLARED_DEPTH} but this benchd has no --mtp-depth; the leg would run serial against a speculative declaration." >&2
        exit 1
      fi
      ;;
    dflash)
      if "${BENCHD}" iterate --help 2>&1 | grep -q -- '--candidate-spec'; then
        SPEC_ARGS=(--candidate-spec "{\"mode\":\"dflash\",\"dflash\":{\"depth\":${DECLARED_DEPTH}}}")
        echo "bonsai2-27b-measure-and-score.sh: the declaration requests DFlash 2 depth ${DECLARED_DEPTH}; benchd sends the candidate spec on the wire and the engine's effective_spec echo is sealed." >&2
      else
        echo "bonsai2-27b-measure-and-score.sh: REFUSING -- the declaration requests DFlash 2 depth ${DECLARED_DEPTH} but this benchd has no --candidate-spec; the leg would run serial against a speculative declaration." >&2
        exit 1
      fi
      # THE RESIDENT'S DRAFTER IS NOT SET HERE. Each leg's resident boot reads
      # ITS OWN tree's mtp-head.manifest.json through tools/spec-declaration.sh
      # and loads the drafter that declaration names, so the candidate leg holds
      # the DFlash 2 drafter whatever label benchd puts on the boot. The control
      # leg holds what the ORGANIZER's reference tree declares, which is the MTP
      # head; the two legs therefore differ in drafter residency by the size
      # difference of the two exports. Nothing here can fix that from this side:
      # benchd derives the boot label from `spec.mtp.depth` alone
      # (crates/benchd/src/legserve.rs:135-141), so it cannot tell the control
      # leg which drafter the pair is about.
      ;;
    *)
      echo "bonsai2-27b-measure-and-score.sh: REFUSING -- the declaration names decoder '${DECLARED_DECODER}', which this script cannot request." >&2
      exit 1
      ;;
  esac
else
  echo "bonsai2-27b-measure-and-score.sh: the declaration is serial; benchd sends no spec (depth 0)." >&2
fi

# THE RESIDENT LOG DIRECTORY IS OUTSIDE BOTH TREES. benchd boots each leg's
# resident with that leg's own tools/resident-up.sh. That script keeps its
# pidfile, identity file, log and sandbox profile in RESIDENT_UP_LOG_DIR, which
# is <tree>/.build/resident by default. The ranked job account cannot write the
# reference workspace, so the control leg must not write there. One directory
# in this job's temporary directory serves both legs. The legs run one after
# the other, inside the GPU lock that this script holds.
if [[ -z "${RESIDENT_UP_LOG_DIR:-}" ]]; then
  RESIDENT_UP_LOG_DIR="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/resident-up.XXXXXX")"
  export RESIDENT_UP_LOG_DIR
fi

# --box, AND THE CONDITION IS INVERTED FROM THE OBVIOUS ONE. benchd lets
# RUNNER_NAME win over the flag, so passing --box on a runner would be dead
# argv: the flag can never change what benchd checks the calibration band
# against. The case the flag DOES serve is the operator hand-run, where there is
# no RUNNER_NAME at all and benchd would otherwise have no box identity to match
# the band's own `box` against.
#
# The value comes from the calibration file itself. That is not a guess: the
# file names the box it was measured on, tools/ranked-box-preflight.sh has
# already refused it if that name is not this runner's, and any other source
# here would be inventing a name for a machine.
BOX_ARGS=()
if [[ -z "${RUNNER_NAME:-}" ]]; then
  CALIBRATION_BOX="$(jq -r '.box // empty' "${BASELINE_CALIBRATION}" 2>/dev/null || true)"
  if [[ -n "${CALIBRATION_BOX}" ]]; then
    BOX_ARGS=(--box "${CALIBRATION_BOX}")
    echo "bonsai2-27b-measure-and-score.sh: RUNNER_NAME is unset (hand run); naming the box '${CALIBRATION_BOX}' from the calibration file, so benchd matches the band to the machine it was measured on." >&2
  else
    echo "bonsai2-27b-measure-and-score.sh: RUNNER_NAME is unset and the calibration file names no box; benchd resolves the box identity itself." >&2
  fi
fi


# --control-golden: the serial tape for the serial-control leg, one per live
# golden, in the same order as --golden. A benchd that does not know the flag
# verifies leg 1 against the candidate's per-depth tape, which is only the same
# tape at depth 0; a speculative declaration on such a benchd is refused rather
# than measured against the wrong oracle.
CONTROL_ARGS=()
if "${BENCHD}" iterate --help 2>&1 | grep -q -- '--control-golden'; then
  CONTROL_PIN_ARGS=()
  for i in "${!SERIAL_GOLDEN_BASENAMES[@]}"; do
    control_path="${GOLDEN_DIR}/${SERIAL_GOLDEN_BASENAMES[$i]}"
    if [[ ! -f "${control_path}" ]]; then
      echo "bonsai2-27b-measure-and-score.sh: serial golden not found at ${control_path}; the serial-control leg has no tape to verify against." >&2
      exit 1
    fi
    CONTROL_ARGS+=(--control-golden "${control_path}")
    echo "bonsai2-27b-measure-and-score.sh: serial-control leg verifies against ${SERIAL_GOLDEN_BASENAMES[$i]} (sha256 ${SERIAL_GOLDEN_SHA256S[$i]}, ${SERIAL_GOLDEN_BYTES_LIST[$i]} bytes)" >&2
  done
  for sha in "${SERIAL_GOLDEN_SHA256S[@]}"; do
    CONTROL_PIN_ARGS+=(--control-golden-sha256 "${sha}")
  done
  for bytes in "${SERIAL_GOLDEN_BYTES_LIST[@]}"; do
    CONTROL_PIN_ARGS+=(--control-golden-bytes "${bytes}")
  done
  CONTROL_ARGS+=("${CONTROL_PIN_ARGS[@]}")
elif [[ "${SPEC_DESC}" != "serial" ]]; then
  echo "bonsai2-27b-measure-and-score.sh: REFUSING -- the declaration is ${SPEC_DESC} but this benchd has no --control-golden; the serial-control leg would be verified against the ${SPEC_DESC} tape." >&2
  exit 1
fi
exec "${BENCHD}" iterate \
  --engine "${ENGINE_BIN_REL}" \
  ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
  --weights "${WEIGHTS_PATH}" \
  "${GOLDEN_ARGS[@]}" \
  --mode official \
  --baseline-workspace "${BASELINE_WORKSPACE}" \
  --baseline-calibration "${BASELINE_CALIBRATION}" \
  ${CONTROL_ARGS[@]+"${CONTROL_ARGS[@]}"} \
  ${BOX_ARGS[@]+"${BOX_ARGS[@]}"} \
  --score-path "${SCORE_PATH}" \
  "${GOLDEN_PIN_ARGS[@]}" \
  --contract "${CONTRACT}"
