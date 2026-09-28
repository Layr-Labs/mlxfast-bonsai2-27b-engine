#!/usr/bin/env bash
#
# sandboxed-cli.sh -- run a verb of the trusted CLI under Seatbelt.
#
# The CLI (.build/release/mlxfast-swift) links the editable transform module.
# Code in a linked module can run before main(), so the CLI cannot confine
# itself. This script is trusted and links nothing: it writes the profile
# (tools/seatbelt-profile.py) and starts the CLI under /usr/bin/sandbox-exec,
# so the confinement is in force before the first instruction of the CLI runs.
# tools/resident-up.sh starts the resident bench-worker the same way.
#
# Usage:
#   tools/sandboxed-cli.sh transform --reference DIR --output DIR
#   tools/sandboxed-cli.sh verify-transform --reference DIR --weights DIR
#                                           [--tmp-parent DIR] [--max-bytes N]
#   tools/sandboxed-cli.sh checkpoint-shards --index PATH
#
# Each option is `--name value`, and each option that sets a writable path is
# required, so this script and the CLI read the same paths. The CLI can write:
#   transform          the --output tree and its `.<output name>.*` staging
#                      siblings;
#   verify-transform   the --tmp-parent tree, or, without --tmp-parent, the
#                      `.mlxfast-transform-verify-*` and
#                      `..mlxfast-transform-verify-*` trees beside --weights;
#   checkpoint-shards  nothing;
# and, for every verb, a private TMPDIR that this script makes and removes.
# The CLI cannot use the network, cannot start another program, and cannot
# read or write the evaluator-only paths the environment names.
#
# Environment:
#   MLXFAST_SWIFT_BIN   the CLI (default <repo>/.build/release/mlxfast-swift)
#   An official run (RUNNER_ENVIRONMENT=self-hosted or
#   MLXFAST_OFFICIAL_BENCHMARK_RUN=1) refuses when the runner environment does
#   not name the evaluator-only paths.
#
# Exit codes: the CLI's exit code; 2 on a refusal before the CLI starts.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
refuse() { printf 'sandboxed-cli.sh: REFUSED: %s\n' "$*" >&2; exit 2; }

[[ $# -ge 1 ]] || refuse "usage: tools/sandboxed-cli.sh transform|verify-transform|checkpoint-shards --name value ..."
VERB="$1"
shift
case "${VERB}" in
  transform) allowed=(--reference --output) ;;
  verify-transform) allowed=(--reference --weights --tmp-parent --max-bytes) ;;
  checkpoint-shards) allowed=(--index) ;;
  *) refuse "unknown verb '${VERB}'; this script runs transform, verify-transform and checkpoint-shards" ;;
esac

output="" weights="" tmp_parent="" index=""
seen=" "
args=("$@")
i=0
while (( i < ${#args[@]} )); do
  name="${args[i]}"
  known=0
  for option in "${allowed[@]}"; do
    [[ "${name}" == "${option}" ]] && known=1
  done
  [[ "${known}" == "1" ]] || refuse "${VERB}: unknown option '${name}'; use --name value with one of: ${allowed[*]}"
  (( i + 1 < ${#args[@]} )) || refuse "${VERB}: ${name} has no value"
  value="${args[i + 1]}"
  [[ -n "${value}" && "${value}" != --* ]] || refuse "${VERB}: ${name} needs a value"
  [[ "${seen}" != *" ${name} "* ]] || refuse "${VERB}: ${name} is given twice"
  seen+="${name} "
  case "${name}" in
    --output) output="${value}" ;;
    --weights) weights="${value}" ;;
    --tmp-parent) tmp_parent="${value}" ;;
    --index) index="${value}" ;;
  esac
  i=$(( i + 2 ))
done

profile_args=()
case "${VERB}" in
  transform)
    [[ -n "${output}" ]] || refuse "transform: --output is required"
    [[ "${seen}" == *" --reference "* ]] || refuse "transform: --reference is required"
    output_abs="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "${output}")"
    profile_args+=(--write-subpath "${output_abs}"
      --write-prefix "$(dirname "${output_abs}")/.$(basename "${output_abs}").")
    ;;
  verify-transform)
    [[ -n "${weights}" ]] || refuse "verify-transform: --weights is required"
    [[ "${seen}" == *" --reference "* ]] || refuse "verify-transform: --reference is required"
    if [[ -n "${tmp_parent}" ]]; then
      profile_args+=(--write-subpath "${tmp_parent}")
    else
      weights_parent="$(python3 -c 'import os, sys; print(os.path.dirname(os.path.abspath(sys.argv[1])))' "${weights}")"
      profile_args+=(--write-prefix "${weights_parent}/.mlxfast-transform-verify-"
        --write-prefix "${weights_parent}/..mlxfast-transform-verify-")
    fi
    ;;
  checkpoint-shards)
    [[ -n "${index}" ]] || refuse "checkpoint-shards: --index is required"
    ;;
esac

SANDBOX_EXEC=/usr/bin/sandbox-exec
[[ -x "${SANDBOX_EXEC}" ]] || refuse "${SANDBOX_EXEC} is not available; the CLI must run under it"
CLI="${MLXFAST_SWIFT_BIN:-${REPO_ROOT}/.build/release/mlxfast-swift}"
[[ -x "${CLI}" ]] || refuse "the CLI is not executable at ${CLI}; build it first"
CLI="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${CLI}")"

official=()
if [[ "${RUNNER_ENVIRONMENT:-}" == "self-hosted" || "${MLXFAST_OFFICIAL_BENCHMARK_RUN:-0}" == "1" ]]; then
  official=(--official)
fi

temp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
WORK_DIR="$(mktemp -d "${temp_root%/}/mlxfast-sandboxed-cli.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT
chmod 700 "${WORK_DIR}"
PRIVATE_TMP="${WORK_DIR}/tmp"
mkdir -m 700 "${PRIVATE_TMP}"
PROFILE="${WORK_DIR}/profile.sb"
python3 "${REPO_ROOT}/tools/seatbelt-profile.py" \
  --exec "${CLI}" --tree "${REPO_ROOT}" --write-subpath "${PRIVATE_TMP}" \
  "${profile_args[@]+"${profile_args[@]}"}" "${official[@]+"${official[@]}"}" > "${PROFILE}" \
  || refuse "could not write the Seatbelt profile; the CLI was not started"

printf 'sandboxed-cli.sh: %s runs under sandbox-exec with %s (TMPDIR %s)\n' "${VERB}" "${PROFILE}" "${PRIVATE_TMP}" >&2
rc=0
TMPDIR="${PRIVATE_TMP}/" "${SANDBOX_EXEC}" -f "${PROFILE}" "${CLI}" "${VERB}" "$@" || rc=$?
exit "${rc}"
