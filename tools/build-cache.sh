#!/usr/bin/env bash
# build-cache.sh -- keep the ranked job's worker build between dispatches.
#
# WHY. actions/checkout cleans the workspace, so every ranked dispatch rebuilt
# bench-worker, the trusted CLI and (with Xcode) mlx.metallib from cold: minutes
# of the job that measure nothing. The build is a pure function of the tracked
# sources (the vendored engine fork included), the resolved packages, the
# vendored Metal sources
# and the toolchain, so it is cached under a content key and restored when the
# key matches. A miss builds as before; a hit skips the build and the metallib
# and lets setup.sh verify the restored pair.
#
# Usage:
#   tools/build-cache.sh key             print the content key
#   tools/build-cache.sh restore         restore the artefacts for the key; exit 0 on a hit, 1 on a miss
#   tools/build-cache.sh save            save the artefacts under the key (after a build)
#
# Environment:
#   MLXFAST_BUILD_CACHE_DIR   cache root (default ~/.cache/mlxfast-engine-build); the box owns it
#
# The artefacts are verified by sha256 on restore against the MANIFEST the save
# wrote; any mismatch is a miss, never a partial restore.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"
log() { printf 'build-cache.sh: %s\n' "$*"; }
die() { printf 'build-cache.sh: %s\n' "$*" >&2; exit 2; }
CACHE_ROOT="${MLXFAST_BUILD_CACHE_DIR:-${HOME}/.cache/mlxfast-engine-build}"
ARTEFACTS=(
  ".build-worker/release/bench-worker"
  ".build-worker/release/mlx.metallib"
  ".build-worker/release/mlx.metallib.fingerprint"
  ".build/release/mlxfast-swift"
)
sha256_of_file() { shasum -a 256 -- "$1" | cut -d ' ' -f 1; }
sha256_of_stdin() { shasum -a 256 | cut -d ' ' -f 1; }
content_key() {
  local paths dirty
  # Vendor/mlx-swift-lm is a VENDORED TREE, so the fork's sources are hashed
  # here like every other tracked source. It used to be a gitlink, which the key
  # carried as one commit id.
  paths=('Sources/*' 'Vendor/mlx-swift/*' 'Vendor/mlx-swift-lm/*' 'Package.swift' 'Package.resolved' 'tools/build-mlx-metallib.sh' 'tools/stage-bench-worker.sh')
  # Blob ids from the index are the content hash of every tracked source; a
  # path modified in the working tree is re-hashed so a dirty tree never
  # keys as its committed state.
  dirty="$(git diff --name-only -- "${paths[@]}")"
  {
    git ls-files -s -- "${paths[@]}" | awk '{print $4 "\t" $2}' | LC_ALL=C sort
    if [[ -n "${dirty}" ]]; then
      while IFS= read -r rel; do
        [[ -n "${rel}" ]] || continue
        [[ -f "${rel}" ]] || die "tracked source ${rel} is missing from the working tree; refusing to key a build on it"
        printf 'dirty\t%s\t%s\n' "${rel}" "$(git hash-object -- "${rel}")"
      done <<< "${dirty}"
    fi
    printf 'metal\t%s\n' "$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null || echo none)"
    printf 'swift\t%s\n' "$(swift --version 2>/dev/null | head -1)"
    printf 'host\t%s\n' "$(uname -m)"
    # The products embed their build directory (SwiftPM resource bundles resolve
    # mlx.metallib from an absolute path baked in at build time), so a product
    # built under one root does not run under another: run 34030992573 restored
    # a worker built elsewhere and the resident died with "Failed to load the
    # default metallib". The root is therefore part of the key.
    printf 'root\t%s\n' "${ROOT_DIR}"
  } | sha256_of_stdin
}
case "${1:-}" in
  key) content_key ;;
  restore)
    key="$(content_key)"; dir="${CACHE_ROOT}/${key}"
    [[ -f "${dir}/MANIFEST" ]] || { log "miss: no cache for key ${key:0:12}"; exit 1; }
    for a in "${ARTEFACTS[@]}"; do
      want="$(awk -v a="${a}" '$1==a{print $2}' "${dir}/MANIFEST")"
      [[ -n "${want}" && -f "${dir}/${a}" ]] || { log "miss: ${a} absent from the cache for key ${key:0:12}"; exit 1; }
      [[ "$(sha256_of_file "${dir}/${a}")" == "${want}" ]] || { log "miss: ${a} in the cache does not match its manifest; refusing a partial restore"; exit 1; }
    done
    for a in "${ARTEFACTS[@]}"; do mkdir -p "$(dirname "${a}")"; cp -p "${dir}/${a}" "${a}"; done
    log "hit: restored ${#ARTEFACTS[@]} artefacts for key ${key:0:12}"
    ;;
  save)
    key="$(content_key)"; dir="${CACHE_ROOT}/${key}"; tmp="${dir}.tmp.$$"
    for a in "${ARTEFACTS[@]}"; do [[ -f "${a}" ]] || die "cannot save: ${a} is missing (build first)"; done
    rm -rf "${tmp}"; mkdir -p "${tmp}"
    : > "${tmp}/MANIFEST"
    for a in "${ARTEFACTS[@]}"; do mkdir -p "${tmp}/$(dirname "${a}")"; cp -p "${a}" "${tmp}/${a}"; printf '%s %s\n' "${a}" "$(sha256_of_file "${a}")" >> "${tmp}/MANIFEST"; done
    rm -rf "${dir}"; mv "${tmp}" "${dir}"
    log "saved ${#ARTEFACTS[@]} artefacts under key ${key:0:12} in ${CACHE_ROOT}"
    ;;
  *) die "usage: tools/build-cache.sh key|restore|save" ;;
esac
