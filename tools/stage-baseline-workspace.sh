#!/usr/bin/env bash
#
# stage-baseline-workspace.sh -- build the organizer's REFERENCE tree on a
# ranked box.
#
# WHAT THE REFERENCE TREE IS. A ranked run on this track is PAIRED (David ruling
# 2026-09-08): it measures a SERIAL-CONTROL leg on the organizer's reference
# tree and a CANDIDATE leg on the submission tree, in the same job, on the same
# box. The score is the LIVE ratio of the two. The reference tree is therefore
# the denominator's engine, and it must be ONE known thing: a checkout of this
# repository at the fixture's baseline_reference_commit, built the documented
# way. This script makes that tree, and nothing else does.
#
# NOTHING IN THE RANKED JOB RUNS THIS. The job holds no credential and builds no
# reference tree (bundles-no-keys, ruled 2026-08-21): it only VERIFIES that the
# tree an operator staged is at the pinned commit with its worker staged
# (tools/ranked-box-preflight.sh section 6, and the workflow's own reference
# check). An operator runs this script by hand when the organizer re-baselines.
#
# NO CREDENTIAL HERE EITHER. The clone source is a box-staged bare mirror or a
# git bundle, never github.com: this box has no key and needs none. The commit
# is checked out by sha, and the resulting HEAD is verified against the fixture
# before the script reports success, so a mirror that lacks the pinned commit
# fails closed rather than staging a different tree under the right name.
#
# THE BUILD IS THE DOCUMENTED ONE, not a shortcut: the two `swift build`
# invocations setup.sh runs (the trusted CLI, then the worker under its own
# scratch root), then the Metal library, then tools/stage-bench-worker.sh, which
# copies the finished set -- the worker, its mlx.metallib and the metallib's
# fingerprint sidecar -- into .build/release, where benchd resolves the engine.
# A box with the Command Line Tools only cannot compile Metal; on such a box set
# MLXFAST_METALLIB_STAGE and pass --prestaged-metallib, which adopts a
# pre-built pair after checking its fingerprint against this checkout's own
# vendored sources.
#
# THE TREE TRANSFORMS ITS OWN WEIGHTS. The control leg runs on the pinned commit
# END TO END -- its engine, its Metal library AND the weights its own transform
# produced -- so this script runs the transform in the reference tree exactly as
# the ranked job runs it for the candidate:
#
#   .build/release/mlxfast-swift transform --reference "${MLXFAST_REFERENCE_DIR}" --output weights
#
# Sharing the candidate's transformed tree instead would put a submission's
# transform on both sides of the ratio, which is the one thing a control leg
# exists to prevent. MLXFAST_REFERENCE_DIR is the box's verified checkpoint, the
# same name the ranked job reads.
#
# Usage:
#   tools/stage-baseline-workspace.sh <dir> [--prestaged-metallib] [--dry-run]
#
#   <dir>                   where the reference tree is created. It must not
#                           already exist, or must be empty: this script never
#                           writes into a tree it did not make, because an
#                           existing tree may be the one a run is using.
#   --prestaged-metallib    adopt MLXFAST_METALLIB_STAGE's mlx.metallib pair
#                           instead of compiling Metal here.
#   --dry-run               print every command in order and run none of them.
#                           This is what CI and the offline test exercise; the
#                           real path is box-only.
#
# Env:
#   MLXFAST_BASELINE_SOURCE  the clone source: a bare mirror directory or a
#                            .bundle file of THIS repository, staged on the box.
#                            Required unless --dry-run.
#   MLXFAST_METALLIB_STAGE   directory holding mlx.metallib and
#                            mlx.metallib.fingerprint, for --prestaged-metallib.
#   MLXFAST_MTP_HEAD_REFERENCE_DIR  the box's verified copy of the separate MTP
#                            head. It is copied into the reference tree and
#                            checked against that tree's own pin. Required
#                            unless --dry-run.
#   MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR  the box's verified copy of the
#                            separate DFlash 2 drafter, staged the same way.
#                            Required unless --dry-run, because a DFlash pair
#                            loads it on BOTH legs.
#   MLXFAST_REFERENCE_DIR    the box's verified reference checkpoint. The
#                            reference tree's own transform reads it. Required
#                            unless --dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACT="${SCRIPT_DIR}/fixtures/bonsai2_27b_mlx_v1_track.json"

die() {
  echo "stage-baseline-workspace.sh: $*" >&2
  exit 1
}

TARGET=""
DRY_RUN=0
PRESTAGED_METALLIB=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --prestaged-metallib) PRESTAGED_METALLIB=1 ;;
    -*) die "unrecognized option: $1" ;;
    *)
      [[ -z "${TARGET}" ]] || die "only one target directory is accepted (got '${TARGET}' and '$1')"
      TARGET="$1"
      ;;
  esac
  shift
done
[[ -n "${TARGET}" ]] || die "usage: tools/stage-baseline-workspace.sh <dir> [--prestaged-metallib] [--dry-run]"

command -v jq >/dev/null 2>&1 || die "jq is required (it reads the fixture's baseline_reference_commit)"
[[ -f "${CONTRACT}" ]] || die "track contract fixture missing: ${CONTRACT}"

REFERENCE_COMMIT="$(jq -r '.baseline_reference_commit // ""' "${CONTRACT}")"
printf '%s' "${REFERENCE_COMMIT}" | grep -Eq '^[0-9a-f]{40}$' \
  || die "the track contract declares no 40-hex baseline_reference_commit (got '${REFERENCE_COMMIT}'); there is no commit to stage"

SOURCE="${MLXFAST_BASELINE_SOURCE:-}"
if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -n "${SOURCE}" ]] \
    || die "MLXFAST_BASELINE_SOURCE is unset; the reference tree is cloned from a box-staged bare mirror or bundle, and this box holds no credential to reach github.com"
  [[ -e "${SOURCE}" ]] || die "MLXFAST_BASELINE_SOURCE does not exist: ${SOURCE}"
  if [[ -e "${TARGET}" ]]; then
    [[ -d "${TARGET}" ]] || die "the target exists and is not a directory: ${TARGET}"
    [[ -z "$(ls -A "${TARGET}" 2>/dev/null)" ]] \
      || die "the target directory is not empty: ${TARGET}; this script never writes into a tree it did not make, because a run may be using it"
  fi
  [[ -n "${MLXFAST_REFERENCE_DIR:-}" ]] \
    || die "MLXFAST_REFERENCE_DIR is unset; the reference tree transforms its OWN weights from the box's verified checkpoint, and sharing the candidate's transformed tree would put a submission's transform on both sides of the ratio"
  [[ -d "${MLXFAST_REFERENCE_DIR}" ]] \
    || die "MLXFAST_REFERENCE_DIR is not a directory: ${MLXFAST_REFERENCE_DIR}"
  [[ -d "${MLXFAST_MTP_HEAD_REFERENCE_DIR:-}" ]] \
    || die "MLXFAST_MTP_HEAD_REFERENCE_DIR is not a directory: '${MLXFAST_MTP_HEAD_REFERENCE_DIR:-}'. It is the box's verified copy of the separate MTP head (./setup-mtp-head.sh stages one at reference_weights/Qwen3.8-27B-MTP-4bit); the control leg loads the head from its own tree"
  [[ -d "${MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR:-}" ]] \
    || die "MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR is not a directory: '${MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR:-}'. It is the box's verified copy of the separate DFlash 2 drafter (./setup-dflash-drafter.sh stages one at reference_weights/Qwen3.8-27B-DFlash2); a DFlash pair loads it on BOTH legs, so the control leg needs its own copy"
  if [[ "${PRESTAGED_METALLIB}" == "1" ]]; then
    stage="${MLXFAST_METALLIB_STAGE:-}"
    [[ -n "${stage}" && -f "${stage}/mlx.metallib" && -f "${stage}/mlx.metallib.fingerprint" ]] \
      || die "--prestaged-metallib was requested but MLXFAST_METALLIB_STAGE does not name a directory holding mlx.metallib and mlx.metallib.fingerprint"
  fi
else
  SOURCE="${SOURCE:-<MLXFAST_BASELINE_SOURCE>}"
fi

# run -- execute a command, or print it under --dry-run. Every step of the
# documented build goes through this, so the dry run prints exactly the sequence
# the real run performs, in order, and nothing else.
step=0
run() {
  step=$((step + 1))
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%2d  %s\n' "${step}" "$*"
    return 0
  fi
  "$@"
}

# git in the reference tree, as one word so --dry-run prints it legibly.
git_target() {
  run git -C "${TARGET}" "$@"
}

echo "stage-baseline-workspace.sh: staging the reference tree at ${REFERENCE_COMMIT} into ${TARGET}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "stage-baseline-workspace.sh: --dry-run, printing the commands only. The real path is BOX-ONLY: it builds Swift and Metal."
fi

# 1. The clone. --no-checkout first, because the pinned commit need not be any
#    branch's tip in the mirror; the checkout is by sha immediately after.
run mkdir -p "${TARGET}"
git_target init --quiet
git_target remote add origin "${SOURCE}"
git_target fetch --quiet --no-tags origin "${REFERENCE_COMMIT}"
git_target checkout --quiet --detach "${REFERENCE_COMMIT}"

# 2. The documented build: the trusted CLI, then the worker under its OWN
#    scratch root (.build-worker), so a participant-code compile can never write
#    into the trusted CLI's .build tree. These are setup.sh's own two lines.
run env -C "${TARGET}" swift build -c release --force-resolved-versions --product mlxfast-swift
run env -C "${TARGET}" swift build -c release --force-resolved-versions --scratch-path .build-worker --product bench-worker

# 3. The Metal library, published next to the worker where Cmlx searches for it.
if [[ "${PRESTAGED_METALLIB}" == "1" ]]; then
  STAGE="${MLXFAST_METALLIB_STAGE:-<MLXFAST_METALLIB_STAGE>}"
  # The pre-staged pair must be THIS checkout's vendored-source fingerprint. A
  # mismatch is fatal, never a fallback: the alternative is a metallib built
  # from other Metal sources measuring the control leg.
  if [[ "${DRY_RUN}" == "1" ]]; then
    run "verify ${STAGE}/mlx.metallib.fingerprint == \$(${TARGET}/tools/build-mlx-metallib.sh --print-fingerprint)"
  else
    want="$(env -C "${TARGET}" tools/build-mlx-metallib.sh --print-fingerprint)"
    have="$(awk '/^mlxfast-metallib-fingerprint-v1 /{print $2}' "${STAGE}/mlx.metallib.fingerprint")"
    [[ -n "${have}" && "${have}" == "${want}" ]] \
      || die "the pre-staged mlx.metallib.fingerprint (${have:-none}) is not this checkout's vendored-source fingerprint (${want}); rebuild the library from the reference tree"
    echo "stage-baseline-workspace.sh: pre-staged mlx.metallib adopted, fingerprint ${want}"
  fi
  # SwiftPM replaces .build-worker/release with a symlink into
  # arm64-apple-macosx/ on a fresh scratch path, which discards anything placed
  # there BEFORE the build -- so the copy happens after it, never before.
  run mkdir -p "${TARGET}/.build-worker/release"
  run cp "${STAGE}/mlx.metallib" "${TARGET}/.build-worker/release/mlx.metallib"
  run cp "${STAGE}/mlx.metallib.fingerprint" "${TARGET}/.build-worker/release/mlx.metallib.fingerprint"
else
  run env -C "${TARGET}" tools/build-mlx-metallib.sh
fi

# 4. Stage the finished set into .build/release, the FIXED path benchd resolves
#    the engine at, with the metallib and its fingerprint sidecar as siblings.
run env -C "${TARGET}" tools/stage-bench-worker.sh

# 5. The reference tree's OWN transformed weights, from the box's verified
#    checkpoint. This is the ranked job's own transform line, run here.
REFERENCE_DIR="${MLXFAST_REFERENCE_DIR:-<MLXFAST_REFERENCE_DIR>}"
run env -C "${TARGET}" .build/release/mlxfast-swift transform --reference "${REFERENCE_DIR}" --output weights

# 5b. The separate MTP head. tools/resident-up.sh loads the head on BOTH legs,
#     so the two legs hold the same bytes, and in --boot it reads the head from
#     ITS OWN tree. The head is copied from the box's verified copy, and the
#     reference tree's own ./setup-mtp-head.sh then checks every file against
#     the reference tree's own pin. A file that fails the pin is refused there.
HEAD_SOURCE_DIR="${MLXFAST_MTP_HEAD_REFERENCE_DIR:-<MLXFAST_MTP_HEAD_REFERENCE_DIR>}"
run mkdir -p "${TARGET}/reference_weights"
run cp -R "${HEAD_SOURCE_DIR}" "${TARGET}/reference_weights/Qwen3.8-27B-MTP-4bit"
run env -C "${TARGET}" ./setup-mtp-head.sh

# 5c. The separate DFlash 2 drafter, staged exactly as the head is and for the
#     same reason. A DFlash pair holds the drafter on BOTH legs -- the control
#     leg loads it and never calls it -- so a reference tree without it would be
#     a different residency from the candidate, and residency would bias the
#     ratio. It costs 3.85 GB in the reference tree, which is the price of a
#     symmetric denominator.
DFLASH_SOURCE_DIR="${MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR:-<MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR>}"
run cp -R "${DFLASH_SOURCE_DIR}" "${TARGET}/reference_weights/Qwen3.8-27B-DFlash2"
run env -C "${TARGET}" ./setup-dflash-drafter.sh

# 6. Verify what was staged. HEAD must be the pinned commit, the worker must be
#    there with its pair, and the weights must be transformed, which is exactly
#    what tools/ranked-box-preflight.sh checks on every ranked run -- proving it
#    here means the operator learns of a bad stage now rather than at dispatch.
if [[ "${DRY_RUN}" == "1" ]]; then
  run "verify $(printf '%s' "${TARGET}")/.git HEAD == ${REFERENCE_COMMIT}"
  run "verify ${TARGET}/.build/release/bench-worker, mlx.metallib and mlx.metallib.fingerprint exist"
  run "verify ${TARGET}/weights/config.json exists"
  echo "stage-baseline-workspace.sh: --dry-run complete, ${step} command(s) printed, nothing was run"
  exit 0
fi

head="$(git -C "${TARGET}" rev-parse HEAD)"
[[ "${head}" == "${REFERENCE_COMMIT}" ]] \
  || die "the staged tree is at ${head}, not the pinned ${REFERENCE_COMMIT}; the mirror at ${SOURCE} does not carry that commit"
for artifact in bench-worker mlx.metallib mlx.metallib.fingerprint; do
  [[ -e "${TARGET}/.build/release/${artifact}" ]] \
    || die "the build produced no ${TARGET}/.build/release/${artifact}; the reference tree is not usable as the control leg"
done
[[ -x "${TARGET}/.build/release/bench-worker" ]] \
  || die "${TARGET}/.build/release/bench-worker is not executable"
[[ -f "${TARGET}/weights/config.json" ]] \
  || die "the reference tree's transform wrote no ${TARGET}/weights/config.json; the control leg has no weights of its own"

echo "stage-baseline-workspace.sh: reference tree staged at ${REFERENCE_COMMIT} in ${TARGET}"
echo "stage-baseline-workspace.sh: export it as MLXFAST_BASELINE_WORKSPACE in the runner service environment."
