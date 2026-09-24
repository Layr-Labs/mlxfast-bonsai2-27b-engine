#!/usr/bin/env bash
# Stage the scored engine -- the fork's generic `bench-worker` -- and its
# mlx.metallib into the location benchd resolves.
#
# WHY THIS EXISTS. benchd resolves the scored engine at a FIXED
# workspace-relative path -- <workspace>/.build/release/bench-worker -- and
# Metal loads mlx.metallib from the SAME directory as the running binary (Cmlx
# searches next to the executable). The worker is BUILT under its own SwiftPM
# scratch root (.build-worker) so a participant-code compile can never write
# into the trusted CLI's .build tree, and mlx.metallib is emitted next to that
# scratch binary (.build-worker/release/mlx.metallib). The build output and
# benchd's resolver therefore land in different directories, and the metallib
# is not beside the binary benchd launches -- so a run either fails to find the
# engine or finds one with no metallib and dies at its first MLXArray.
#
# This step copies the FINISHED SET -- the engine binary, its mlx.metallib and
# the metallib's fingerprint sidecar -- from the scratch root into
# .build/release, as siblings. Only this trusted step performs the copy, and it
# runs AFTER the build, so the scratch-root build isolation is preserved.
#
# THE FINGERPRINT SIDECAR TRAVELS WITH THE METALLIB. tools/build-mlx-metallib.sh
# writes mlx.metallib.fingerprint beside every metallib it publishes, and the
# harness check (Sources/MLXFastTrustedHarness/VendoredMetalFingerprint.swift,
# verifyMetallibFingerprintRecord) reads it at <metallib path>.fingerprint --
# i.e. next to the metallib the worker actually loads, which is the STAGED one.
# Staging the metallib without its sidecar therefore turns a passing check into
# a `mismatch` ("no fingerprint record at ..."), which an official run treats as
# fatal. So the sidecar is copied with the pair, and a metallib that arrives
# without one is a REFUSAL, not a skip: a silently absent sidecar is
# indistinguishable at the check from a tampered one.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"

repository_path() {
  local candidate="$1"
  if [[ "${candidate}" == /* ]]; then
    printf '%s\n' "${candidate}"
  else
    printf '%s/%s\n' "${ROOT_DIR}" "${candidate#./}"
  fi
}

BUILD_CONFIGURATION="${MLXFAST_SWIFT_CONFIGURATION:-release}"

# SOURCE pair -- the participant engine's scratch root. Resolved exactly as
# setup.sh and tools/build-mlx-metallib.sh resolve them, so an operator
# override is honoured identically across all three.
BENCH_WORKER_BIN="$(repository_path \
  "${MLXFAST_BENCH_WORKER_EXECUTABLE:-.build-worker/${BUILD_CONFIGURATION}/bench-worker}")"
MLX_METALLIB="$(repository_path \
  "${MLXFAST_MLX_METALLIB:-$(dirname "${BENCH_WORKER_BIN}")/mlx.metallib}")"

# DESTINATION pair -- the FIXED location benchd resolves. Not env-overridable:
# it is a contract with the benchmarker, not a preference.
STAGED_WORKER_BIN="$(repository_path ".build/${BUILD_CONFIGURATION}/bench-worker")"
STAGED_METALLIB="$(dirname "${STAGED_WORKER_BIN}")/mlx.metallib"
# THE PAGED-ATTENTION METAL SOURCE. The Nemotron 3.5 MTP assistant keeps its
# own paged KV state and compiles pagedattention.metal at runtime from a
# SwiftPM resource. The resolver (PagedAttentionResources.swift) accepts the
# bare file beside the executable or inside a sibling *.bundle, never the
# Contents/Resources layout the build system writes, so the bare file is
# staged beside the worker. Without it every MTP leg ends with
# "backendIneligible: paged-attention runtime resource unavailable".
PAGED_ATTENTION_SOURCE="$(repository_path \
  "Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/pagedattention.metal")"
STAGED_PAGED_ATTENTION="$(dirname "${STAGED_WORKER_BIN}")/pagedattention.metal"
MLX_METALLIB_FINGERPRINT="${MLX_METALLIB}.fingerprint"
STAGED_METALLIB_FINGERPRINT="${STAGED_METALLIB}.fingerprint"

if [[ ! -x "${BENCH_WORKER_BIN}" ]]; then
  echo "stage-bench-worker.sh: scored engine missing or not executable: ${BENCH_WORKER_BIN}" >&2
  echo "stage-bench-worker.sh: build it first (setup.sh, or swift build -c ${BUILD_CONFIGURATION} --scratch-path .build-worker --product bench-worker)" >&2
  exit 1
fi

mkdir -p "$(dirname "${STAGED_WORKER_BIN}")"

# `-ef` is true only when both paths already resolve to the same file, which
# happens when an operator override points the source straight at the benchd
# path -- then the copy is a no-op (and cp would error copying a file onto
# itself).
if [[ ! "${BENCH_WORKER_BIN}" -ef "${STAGED_WORKER_BIN}" ]]; then
  cp -f "${BENCH_WORKER_BIN}" "${STAGED_WORKER_BIN}"
fi
# benchd checks the execute bit on the resolved binary; guarantee it survives
# the copy regardless of the caller's umask.
chmod +x "${STAGED_WORKER_BIN}"
if [[ ! -f "${PAGED_ATTENTION_SOURCE}" ]]; then
  echo "stage-bench-worker.sh: paged-attention Metal source missing from the vendored fork: ${PAGED_ATTENTION_SOURCE}" >&2
  exit 1
fi
cp -f "${PAGED_ATTENTION_SOURCE}" "${STAGED_PAGED_ATTENTION}"

if [[ "${MLXFAST_SKIP_MLX_METALLIB:-0}" == "1" ]]; then
  # The metallib build was explicitly skipped; there is nothing to stage and
  # the operator has accepted that a real GPU run cannot succeed without it.
  metallib_staged=0
else
  if [[ ! -f "${MLX_METALLIB}" ]]; then
    echo "stage-bench-worker.sh: mlx.metallib missing next to the scored engine: ${MLX_METALLIB}" >&2
    echo "stage-bench-worker.sh: build it first (tools/build-mlx-metallib.sh)" >&2
    exit 1
  fi
  if [[ ! -f "${MLX_METALLIB_FINGERPRINT}" ]]; then
    echo "stage-bench-worker.sh: the metallib fingerprint sidecar is missing beside the scored engine: ${MLX_METALLIB_FINGERPRINT}" >&2
    echo "stage-bench-worker.sh: the harness reads it at <staged metallib>.fingerprint, and a metallib with no sidecar fails that check; rebuild the metallib (tools/build-mlx-metallib.sh), which publishes both" >&2
    exit 1
  fi
  if [[ ! "${MLX_METALLIB}" -ef "${STAGED_METALLIB}" ]]; then
    cp -f "${MLX_METALLIB}" "${STAGED_METALLIB}"
    cp -f "${MLX_METALLIB_FINGERPRINT}" "${STAGED_METALLIB_FINGERPRINT}"
  fi
  metallib_staged=1
fi

# ONE closing line, and it reports what this run actually did to the pair. A
# skip does not imply the sibling is absent: a previous run usually left one in
# place, and that is the case worth distinguishing from the one where a GPU run
# will die at its first MLXArray.
if [[ "${metallib_staged}" == "1" ]]; then
  echo "stage-bench-worker.sh: staged ${STAGED_WORKER_BIN} with sibling mlx.metallib, its fingerprint sidecar and pagedattention.metal for benchd"
elif [[ -f "${STAGED_METALLIB}" ]]; then
  echo "stage-bench-worker.sh: staged ${STAGED_WORKER_BIN}; MLXFAST_SKIP_MLX_METALLIB=1, kept the mlx.metallib already beside it"
else
  echo "stage-bench-worker.sh: staged ${STAGED_WORKER_BIN}; MLXFAST_SKIP_MLX_METALLIB=1 and no mlx.metallib is beside it, so a GPU run cannot succeed yet"
fi
