#!/usr/bin/env bash
# Provision the organizer-pinned MTP HEAD for track bonsai2-27b-mlx-v1.
#
# This script provisions the HEAD and nothing else. The TARGET is the Ternary
# Bonsai 2 27B pack that ./setup.sh pins, downloads and verifies against
# fixtures/reference_bonsai2_27b_2bit.sha256, so there is no second target
# download here and no way for this script to point the track at a different
# pack.
#
# WHY THE HEAD IS A SEPARATE ARTIFACT. The pinned pack declares
# `mtp_num_hidden_layers: 0` and ships no `mtp.*` tensor. The transform refuses
# a pack that carries one. The head is therefore its own published export, and
# its own pin file.
#
# WHY THIS IS A WRAPPER AND NOT A DOWNLOADER. setup.sh already holds the
# resumable, hash-verified, stall-detecting download logic, and it is fully
# parameterised by environment (MLXFAST_REFERENCE_MODEL_REPO / _REVISION /
# _MANIFEST_PATH / _BASE_URL / _CACHE_DIR / _DIR / _COMPAT_LINK). This script
# points those at the head and delegates. One downloader, one verification
# path, one set of stall and resume semantics.
#
# The delegated run does NOT rebuild the Swift products: it passes
# MLXFAST_SKIP_SWIFT_BUILD=1, so an existing pair is reused. Nothing about the
# products can have changed between the two commands. The tool-installing legs
# and the metallib build are skipped for the same reason in the other
# direction: ./setup.sh owns those and this script must not mutate global state
# a second time.
#
# ./setup.sh runs this script for you as its last step. Run it by hand only to
# re-verify the head, or after MLXFAST_SKIP_MTP_HEAD=1 skipped it.
#
# Usage: ./setup-mtp-head.sh
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${ROOT_DIR}"

TRACK_ID="bonsai2-27b-mlx-v1"
HEAD_REPO="${MLXFAST_MTP_HEAD_REPO:-EigenLabs/Qwen3.8-27B-MTP-4bit}"
HEAD_REVISION="${MLXFAST_MTP_HEAD_REVISION:-329261c5e0b3f9c233485e682cb3b67b88c20a55}"
HEAD_MANIFEST="${MLXFAST_MTP_HEAD_MANIFEST_PATH:-fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256}"
HEAD_DIR="${MLXFAST_MTP_HEAD_DIR:-reference_weights/Qwen3.8-27B-MTP-4bit}"

[[ -f "${HEAD_MANIFEST}" ]] || {
  echo "setup-mtp-head.sh: pin file is missing: ${HEAD_MANIFEST}" >&2
  exit 1
}

echo "setup-mtp-head.sh: track ${TRACK_ID}; head ${HEAD_REPO}@${HEAD_REVISION} -> ${HEAD_DIR}"

# The head has no compatibility link of its own: nothing resolves it by a
# legacy path. Point the link at the head directory so setup.sh does not
# create or check one for the target while provisioning the head.
MLXFAST_SKIP_SWIFT_BUILD=1 \
MLXFAST_SKIP_MACMON_INSTALL=1 \
MLXFAST_SKIP_MLX_METALLIB=1 \
MLXFAST_SKIP_MTP_HEAD=1 \
MLXFAST_REFERENCE_MODEL_REPO="${HEAD_REPO}" \
MLXFAST_REFERENCE_REVISION="${HEAD_REVISION}" \
MLXFAST_REFERENCE_MANIFEST_PATH="${HEAD_MANIFEST}" \
MLXFAST_REFERENCE_DIR="${HEAD_DIR}" \
MLXFAST_REFERENCE_COMPAT_LINK="${HEAD_DIR}" \
MLXFAST_REFERENCE_MIN_FREE_GIB="${MLXFAST_MTP_HEAD_MIN_FREE_GIB:-2}" \
  ./setup.sh "$@"

echo "setup-mtp-head.sh: the head is staged and verified at ${HEAD_DIR}"
echo "setup-mtp-head.sh: the engine reads it with --drafter ${HEAD_DIR}"
