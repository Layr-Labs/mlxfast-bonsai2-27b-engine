#!/usr/bin/env bash
# test-fetch-benchd.sh -- fetch-benchd.sh resolves the renamed channel pair.
#
# Hermetic: a stub `benchd` and a manifest in the post-rename dist format
# (six top-level fields describing benchd, then the `binaries` map with a
# sha256/bytes entry per binary) are handed to fetch-benchd.sh through
# BENCHD_DIST_LOCAL. The script must install benchd + benchd.manifest.json into
# BENCHD_BIN_DIR, print the binary path, and refuse a pair whose bytes moved.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
fail() { echo "test-fetch-benchd: FAIL -- $*" >&2; exit 1; }

# The channel name is an input of fetch-benchd.sh (BENCHD_BRANCH); the test
# names its own so the manifest and the resolve agree by construction.
BRANCH="test-channel-v0"
export BENCHD_BRANCH="${BRANCH}"

DIST="${WORK}/dist"; mkdir -p "${DIST}"
printf '#!/bin/sh\necho stub-benchd\n' > "${DIST}/benchd"; chmod 755 "${DIST}/benchd"
printf '#!/bin/sh\necho stub-recorder\n' > "${DIST}/record-correctness-golden"; chmod 755 "${DIST}/record-correctness-golden"
sha() { shasum -a 256 "$1" | awk '{print $1}'; }
bytes() { wc -c < "$1" | tr -d '[:space:]'; }
{
  printf '{\n'
  printf '  "version": "0.0.0",\n'
  printf '  "branch": "%s",\n' "${BRANCH}"
  printf '  "source_commit": "0123456789abcdef0123456789abcdef01234567",\n'
  printf '  "target_triple": "aarch64-apple-darwin",\n'
  printf '  "sha256": "%s",\n' "$(sha "${DIST}/benchd")"
  printf '  "bytes": %s,\n' "$(bytes "${DIST}/benchd")"
  printf '  "binaries": {\n'
  printf '    "benchd": {"sha256": "%s", "bytes": %s},\n' "$(sha "${DIST}/benchd")" "$(bytes "${DIST}/benchd")"
  printf '    "record-correctness-golden": {"sha256": "%s", "bytes": %s}\n' "$(sha "${DIST}/record-correctness-golden")" "$(bytes "${DIST}/record-correctness-golden")"
  printf '  }\n'
  printf '}\n'
} > "${DIST}/benchd.manifest.json"

BIN="${WORK}/benchd-bin"
if ! out="$(BENCHD_DIST_LOCAL="${DIST}" BENCHD_BIN_DIR="${BIN}" bash "${ROOT}/tools/fetch-benchd.sh" 2>"${WORK}/err")"; then
  fail "resolve refused a good post-rename pair: $(cat "${WORK}/err")"
fi
[[ "${out}" == "${BIN}/benchd" ]] || fail "printed '${out}', expected ${BIN}/benchd"
[[ -x "${BIN}/benchd" && -f "${BIN}/benchd.manifest.json" ]] || fail "pair not installed under ${BIN}"
grep -q "source_commit=0123456789abcdef0123456789abcdef01234567" "${WORK}/err" || fail "identity line missing: $(cat "${WORK}/err")"
echo "test-fetch-benchd: PASS -- post-rename pair resolves (binaries map tolerated)"

# A binary whose bytes moved after the manifest was written is refused.
printf '#!/bin/sh\necho tampered\n' > "${DIST}/benchd"
BIN2="${WORK}/benchd-bin-2"
if BENCHD_DIST_LOCAL="${DIST}" BENCHD_BIN_DIR="${BIN2}" bash "${ROOT}/tools/fetch-benchd.sh" >/dev/null 2>"${WORK}/err2"; then
  fail "a tampered binary was accepted"
fi
[[ ! -e "${BIN2}/benchd" ]] || fail "a tampered binary was installed"
echo "test-fetch-benchd: PASS -- a mismatching binary is refused and nothing is installed"
echo "test-fetch-benchd: all cases pass"
