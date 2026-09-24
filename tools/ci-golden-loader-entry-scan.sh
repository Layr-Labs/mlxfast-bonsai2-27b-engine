#!/usr/bin/env bash
#
# Trip when production code reaches the golden loader around its Qwen wrapper.
#
# `loadQwenGoldenFixture` is where a golden is held to the Qwen `model_type`
# and where `model_provenance` becomes REQUIRED (see the doc comment on that
# function in Sources/MLXFastCore/Golden.swift). Those two rules bind ONLY
# because that wrapper is the single production entry point: the loader it
# calls, `loadGoldenFixture`, is deliberately model-agnostic and takes both
# rules as opt-in arguments. A new call site that reaches `loadGoldenFixture`
# or `loadGoldenCases` directly therefore does not fail -- it silently loads a
# golden with no model check and no provenance check.
#
# That regression is invisible in review of the new file alone, so it is
# checked here instead of trusted. The invariant, stated exactly:
#
#   NO occurrence of `loadGoldenFixture` or `loadGoldenCases` in tracked Swift
#   sources under Sources/ or Vendor/, outside Sources/MLXFastCore/Golden.swift
#   where both are defined.
#
# WHAT THIS IS, PRECISELY: a NAMED-IDENTIFIER TRIPWIRE. It matches the two
# names as text and nothing else, so it also fails on a mention in a comment
# and it cannot see a call made through some renamed alias. Both are accepted:
# the first is a one-word edit to the comment, and the second is not a shape
# this package writes.
#
# Tests/ is deliberately NOT scanned. The loader tests call `loadGoldenFixture`
# on purpose -- one of them asserts the model-agnostic loader ACCEPTS what the
# Qwen wrapper refuses, which is what makes the wrapper's rules a real delta.
#
# Usage:  tools/ci-golden-loader-entry-scan.sh
# Exit:   0 no direct call, 1 at least one.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Where both functions are defined, and the only file allowed to name them.
home_file='Sources/MLXFastCore/Golden.swift'

hits="$(
  git grep -nE 'loadGoldenFixture|loadGoldenCases' -- 'Sources/*.swift' 'Vendor/*.swift' \
    | grep -v "^${home_file}:" \
    || true
)"

if [[ -n "${hits}" ]]; then
  echo "::error::production code reaches the golden loader directly; every Qwen-facing entry point must call loadQwenGoldenFixture" >&2
  echo "ci-golden-loader-entry-scan: loadGoldenFixture / loadGoldenCases may be named only in ${home_file}:" >&2
  printf '%s\n' "${hits}" | sed 's/^/  /' >&2
  echo "ci-golden-loader-entry-scan: loadQwenGoldenFixture is what pins model_type and REQUIRES model_provenance; a direct call skips both." >&2
  exit 1
fi

echo "ok: loadQwenGoldenFixture is the only golden entry point under Sources/ and Vendor/"
