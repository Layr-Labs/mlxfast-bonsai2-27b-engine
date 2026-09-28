#!/usr/bin/env bash
#
# sandboxed-build.sh -- run one compile command of the build under Seatbelt.
#
# The build compiles files from editablePaths (C++, Metal and Swift). A
# compiler reads every file that a source names (#include, #embed, .incbin).
# On an official run, this script writes a build profile
# (tools/seatbelt-profile.py --build) and starts the command under
# /usr/bin/sandbox-exec. The command and every process that it starts:
#   * cannot read or write the evaluator-only paths that the environment
#     names, the build cache root, or the runner registration files;
#   * can write only the --write trees and a private TMPDIR that this script
#     makes and removes;
#   * cannot use the network.
# setup.sh and tools/build-mlx-metallib.sh call this script for each compile
# command. The dependency fetch (swift package resolve) runs before it, outside
# the sandbox, and compiles no editable file.
#
# Usage:
#   tools/sandboxed-build.sh --write DIR [--write DIR ...] -- COMMAND [ARG ...]
#
# A swift command in the sandbox must have --disable-sandbox, because a
# process in a sandbox cannot start a nested sandbox.
#
# Environment:
#   An official run (RUNNER_ENVIRONMENT=self-hosted or
#   MLXFAST_OFFICIAL_BENCHMARK_RUN=1) refuses when the runner environment does
#   not name the evaluator-only paths. On other runs, this script runs the
#   command without a sandbox.
#
# Exit codes: the command's exit code; 2 on a refusal before the command starts.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
refuse() { printf 'sandboxed-build.sh: REFUSED: %s\n' "$*" >&2; exit 2; }

writes=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
  case "$1" in
    --write)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || refuse "--write needs a directory"
      writes+=("$2")
      shift 2
      ;;
    *) refuse "unknown option '$1'; usage: tools/sandboxed-build.sh --write DIR ... -- COMMAND [ARG ...]" ;;
  esac
done
[[ $# -ge 2 && "$1" == "--" ]] || refuse "no command; usage: tools/sandboxed-build.sh --write DIR ... -- COMMAND [ARG ...]"
shift
[[ ${#writes[@]} -gt 0 ]] || refuse "no --write directory; the build must write somewhere"

if [[ "${RUNNER_ENVIRONMENT:-}" != "self-hosted" && "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" != "1" ]]; then
  exec "$@"
fi

temp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
WORK_DIR="$(mktemp -d "${temp_root%/}/mlxfast-sandboxed-build.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT
chmod 700 "${WORK_DIR}"
PRIVATE_TMP="${WORK_DIR}/tmp"
mkdir -m 700 "${PRIVATE_TMP}"
PROFILE="${WORK_DIR}/profile.sb"
profile_args=()
# A build tool that SwiftPM starts gets the temporary and cache directories of
# the account, not TMPDIR. These directories belong to the job account only.
# /bin/sh writes the temporary file of a here document in /var/tmp. It does
# not use TMPDIR for it. Build scripts use here documents.
profile_args+=(--write-subpath /private/var/tmp)
for name in DARWIN_USER_TEMP_DIR DARWIN_USER_CACHE_DIR; do
  user_dir="$(/usr/bin/getconf "${name}" 2>/dev/null || true)"
  [[ -n "${user_dir}" && -d "${user_dir}" ]] && profile_args+=(--write-subpath "${user_dir%/}")
done
for dir in "${writes[@]}"; do
  mkdir -p "${dir}"
  profile_args+=(--write-subpath "${dir}")
done
python3 "${REPO_ROOT}/tools/seatbelt-profile.py" --build --official --tree "${REPO_ROOT}" \
  --write-subpath "${PRIVATE_TMP}" "${profile_args[@]}" > "${PROFILE}" \
  || refuse "could not write the Seatbelt profile; the build was not started"

SANDBOX_EXEC=/usr/bin/sandbox-exec
[[ -x "${SANDBOX_EXEC}" ]] || refuse "${SANDBOX_EXEC} is not available; an official build must run under it"

printf 'sandboxed-build.sh: %s runs under sandbox-exec with %s (TMPDIR %s)\n' "$1" "${PROFILE}" "${PRIVATE_TMP}" >&2
rc=0
TMPDIR="${PRIVATE_TMP}/" "${SANDBOX_EXEC}" -f "${PROFILE}" "$@" || rc=$?
exit "${rc}"
