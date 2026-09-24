#!/usr/bin/env bash
# Provision the organizer-pinned DFLASH 2 DRAFTER for track bonsai2-27b-mlx-v1.
#
# This script provisions the DRAFTER and nothing else. The TARGET is the Ternary
# Bonsai 2 27B pack that ./setup.sh pins, downloads and verifies against
# fixtures/reference_bonsai2_27b_2bit.sha256, so there is no second target
# download here and no way for this script to point the track at a different
# pack.
#
# WHY THE DRAFTER IS A SEPARATE ARTIFACT. The pinned pack carries no drafter of
# any kind: it declares `mtp_num_hidden_layers: 0` and ships no `mtp.*` tensor.
# Both of this track's speculative decoders are therefore their own published
# exports with their own pin files. This one stages the second.
#
# WHAT IT STAGES. z-lab/Qwen3.8-27B-DFlash2, 81 BF16 tensors in one shard,
# 3,848,808,960 tensor bytes. The drafter owns no embedding table and no output
# projection: it binds the target's, which on this pack are packed Hadamard
# modules. It stays BF16 as published; nothing re-quantizes it.
#
# WHY THIS IS A WRAPPER AND NOT A DOWNLOADER. setup.sh already holds the
# resumable, hash-verified, stall-detecting download logic, and it is fully
# parameterised by environment (MLXFAST_REFERENCE_MODEL_REPO / _REVISION /
# _MANIFEST_PATH / _BASE_URL / _CACHE_DIR / _DIR / _COMPAT_LINK). This script
# points those at the drafter and delegates. One downloader, one verification
# path, one set of stall and resume semantics. ./setup-mtp-head.sh is the same
# wrapper around the same downloader for the other decoder.
#
# The delegated run does NOT rebuild the Swift products: it passes
# MLXFAST_SKIP_SWIFT_BUILD=1, so an existing pair is reused. Nothing about the
# products can have changed between the two commands. The tool-installing legs
# and the metallib build are skipped for the same reason in the other
# direction: ./setup.sh owns those and this script must not mutate global state
# a second time.
#
# ./setup.sh runs this script for you, after the MTP head. Run it by hand only
# to re-verify the drafter, or after MLXFAST_SKIP_DFLASH_DRAFTER=1 skipped it.
#
# DISK. The drafter is 3.85 GB. The free-space floor is raised to match.
#
# Usage: ./setup-dflash-drafter.sh
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${ROOT_DIR}"

TRACK_ID="bonsai2-27b-mlx-v1"
DRAFTER_REPO="${MLXFAST_DFLASH_DRAFTER_REPO:-z-lab/Qwen3.8-27B-DFlash2}"
DRAFTER_REVISION="${MLXFAST_DFLASH_DRAFTER_REVISION:-50307d4c4cde6860d4eee73e2547cd786fe8e8a4}"
DRAFTER_MANIFEST="${MLXFAST_DFLASH_DRAFTER_MANIFEST_PATH:-fixtures/reference_bonsai2_27b_dflash2_drafter.sha256}"
DRAFTER_DIR="${MLXFAST_DFLASH_DRAFTER_DIR:-reference_weights/Qwen3.8-27B-DFlash2}"

[[ -f "${DRAFTER_MANIFEST}" ]] || {
  echo "setup-dflash-drafter.sh: pin file is missing: ${DRAFTER_MANIFEST}" >&2
  exit 1
}

echo "setup-dflash-drafter.sh: track ${TRACK_ID}; drafter ${DRAFTER_REPO}@${DRAFTER_REVISION} -> ${DRAFTER_DIR}"

# The drafter has no compatibility link of its own: nothing resolves it by a
# legacy path. Point the link at the drafter directory so setup.sh does not
# create or check one for the target while provisioning the drafter.
MLXFAST_SKIP_SWIFT_BUILD=1 \
MLXFAST_SKIP_MACMON_INSTALL=1 \
MLXFAST_SKIP_MLX_METALLIB=1 \
MLXFAST_SKIP_MTP_HEAD=1 \
MLXFAST_SKIP_DFLASH_DRAFTER=1 \
MLXFAST_REFERENCE_MODEL_REPO="${DRAFTER_REPO}" \
MLXFAST_REFERENCE_REVISION="${DRAFTER_REVISION}" \
MLXFAST_REFERENCE_MANIFEST_PATH="${DRAFTER_MANIFEST}" \
MLXFAST_REFERENCE_DIR="${DRAFTER_DIR}" \
MLXFAST_REFERENCE_COMPAT_LINK="${DRAFTER_DIR}" \
MLXFAST_REFERENCE_MIN_FREE_GIB="${MLXFAST_DFLASH_DRAFTER_MIN_FREE_GIB:-8}" \
  ./setup.sh "$@"

echo "setup-dflash-drafter.sh: the drafter is staged and verified at ${DRAFTER_DIR}"
echo "setup-dflash-drafter.sh: the engine reads it with --drafter ${DRAFTER_DIR}"
