#!/usr/bin/env bash
# Unit test for tools/golden-arming-patch.py. Offline and hermetic: synthetic
# golden files in a temporary directory, a copy of the contract fixture for the
# --apply case, no R2, no box, no real golden.
#
# The synthetic set has the shape the per-depth rule must handle: the seven
# MTP oracles are identical to each other and differ from the serial tape,
# dflash1 is byte-identical to the serial tape, and dflash2..16 are identical
# to each other. So 23 keys need 2 new objects, and 8 tapes + 2 oracles + 2
# public captures = 12 distinct uploads.
#
# Usage: tools/test-golden-arming-patch.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TOOL="${SCRIPT_DIR}/tools/golden-arming-patch.py"
CONTRACT="${SCRIPT_DIR}/fixtures/bonsai2_27b_mlx_v1_track.json"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() { echo "FAIL: $1" >&2; failures=$((failures + 1)); }
pass() { echo "ok: $1"; }

# The marker is in every synthetic file. It must never reach the tool's output.
MARKER="SYNTHETIC-CONTENT-MARKER-7f3a"
golden() { printf '{"version":1,"marker":"%s","case":"%s"}\n' "${MARKER}" "$2" > "$1"; }

make_set() { # make_set DIR
  mkdir -p "$1"
  local n
  for n in alpha bravo charlie delta echo foxtrot golf hotel; do
    golden "$1/${n}.golden.json" "${n}"
  done
  for n in 1 2 3 4 5 6 7; do golden "$1/alpha.mtp${n}.golden.json" "speculative-mtp"; done
  cp "$1/alpha.golden.json" "$1/alpha.dflash1.golden.json"
  for n in $(seq 2 16); do golden "$1/alpha.dflash${n}.golden.json" "speculative-dflash"; done
  golden "$1/public-local-iterate.golden.json" "public-short"
  golden "$1/public-local-submit.golden.json" "public-long"
}

make_set "${WORK}/set"
q() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$@"; }

# --- case 1: the patch arms every key, and identical bytes share one object --
if python3 "${TOOL}" --dir "${WORK}/set" --live alpha > "${WORK}/patch.json" 2> "${WORK}/case1.err"; then
  P="${WORK}/patch.json"
  pre="correctness_prompts/bonsai2-27b-mlx-v1"
  ok1=1
  [[ "$(q "${P}" 'd["official_scoring_enabled"]')" == "True" ]] || { fail "case 1: official_scoring_enabled is not true"; ok1=0; }
  [[ "$(q "${P}" 'len(d["timed_prompt_pool"])')" == "8" ]] || { fail "case 1: the pool is not 8 tapes"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden"]')" == "alpha" ]] || { fail "case 1: live_golden is not alpha"; ok1=0; }
  [[ "$(q "${P}" '[e for e in d["timed_prompt_pool"] if e["r2_path"].endswith("/alpha.golden.json")][0]["sha256"] == d["hidden_correctness_golden"]["sha256"]')" == "True" ]] \
    || { fail "case 1: the hidden oracle is not the live golden's own pin"; ok1=0; }
  [[ "$(q "${P}" 'len(d["live_golden_speculative"])')" == "23" ]] || { fail "case 1: live_golden_speculative does not carry 23 keys"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden_speculative"]["mtp7"]["r2_path"]')" == "${pre}/alpha.mtp1.golden.json" ]] \
    || { fail "case 1: mtp7 does not share the mtp1 object"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden_speculative"]["dflash1"]["r2_path"]')" == "${pre}/alpha.golden.json" ]] \
    || { fail "case 1: dflash1 (serial-identical) does not point at the live golden"; ok1=0; }
  [[ "$(q "${P}" 'd["live_golden_speculative"]["dflash16"]["r2_path"]')" == "${pre}/alpha.dflash2.golden.json" ]] \
    || { fail "case 1: dflash16 does not share the dflash2 object"; ok1=0; }
  [[ "$(q "${P}" 'd["public_captures"]["local_submit"]["r2_path"]')" == "${pre}/public-local-submit.golden.json" ]] \
    || { fail "case 1: the local_submit capture does not keep its r2_path"; ok1=0; }
  grep -q "12 distinct object(s); 21 of 23 per-depth key(s) share an earlier object" "${WORK}/case1.err" \
    || { fail "case 1: the upload summary is wrong ($(tail -1 "${WORK}/case1.err"))"; ok1=0; }
  if grep -q "${MARKER}" "${P}" "${WORK}/case1.err"; then
    fail "case 1: the output carries golden CONTENT"; ok1=0
  fi
  [[ "${ok1}" == "1" ]] && pass "the patch arms all keys, dedupes identical bytes, and prints no content"
else
  fail "case 1: the tool refused a complete set ($(cat "${WORK}/case1.err"))"
fi

# --- case 2: a missing per-depth oracle is refused by key --------------------
make_set "${WORK}/missing"
rm "${WORK}/missing/alpha.dflash9.golden.json"
if python3 "${TOOL}" --dir "${WORK}/missing" --live alpha > /dev/null 2> "${WORK}/case2.err"; then
  fail "case 2: the tool accepted a set with no dflash9 oracle"
elif grep -q "no per-depth oracle for: dflash9" "${WORK}/case2.err"; then
  pass "a missing per-depth oracle is refused by key"
else
  fail "case 2: refused for the wrong reason ($(cat "${WORK}/case2.err"))"
fi

# --- case 3: a public capture with a hidden tape's bytes is refused ----------
make_set "${WORK}/leak"
cp "${WORK}/leak/charlie.golden.json" "${WORK}/leak/public-local-iterate.golden.json"
if python3 "${TOOL}" --dir "${WORK}/leak" --live alpha > /dev/null 2> "${WORK}/case3.err"; then
  fail "case 3: the tool accepted a public capture that is a hidden tape"
elif grep -q "same bytes as a hidden tape" "${WORK}/case3.err"; then
  pass "a public capture with a hidden tape's bytes is refused"
else
  fail "case 3: refused for the wrong reason ($(cat "${WORK}/case3.err"))"
fi

# --- case 4: a stored baseline pair is refused -------------------------------
make_set "${WORK}/pair"
printf '{"version":1,"benchmark":{"baseline_decode_seconds_per_token":0.03}}\n' > "${WORK}/pair/delta.golden.json"
if python3 "${TOOL}" --dir "${WORK}/pair" --live alpha > /dev/null 2> "${WORK}/case4.err"; then
  fail "case 4: the tool accepted a tape that stores a baseline pair"
elif grep -q "stores a baseline pair" "${WORK}/case4.err"; then
  pass "a tape that stores a baseline pair is refused"
else
  fail "case 4: refused for the wrong reason ($(cat "${WORK}/case4.err"))"
fi

# --- case 5: --apply leaves no pending sentinel ------------------------------
# The ranked preflight refuses any PENDING-ORGANIZER in the contract, so an
# applied patch must leave none.
cp "${CONTRACT}" "${WORK}/contract.json"
if python3 "${TOOL}" --dir "${WORK}/set" --live alpha --apply --contract "${WORK}/contract.json" 2> "${WORK}/case5.err"; then
  if grep -q "PENDING-ORGANIZER" "${WORK}/contract.json"; then
    fail "case 5: the applied contract still carries a pending sentinel"
  elif [[ "$(q "${WORK}/contract.json" 'd["track_id"]')" != "bonsai2-27b-mlx-v1" ]]; then
    fail "case 5: --apply lost a fixture key it does not own"
  else
    pass "--apply arms the contract and leaves no pending sentinel"
  fi
else
  fail "case 5: --apply refused ($(cat "${WORK}/case5.err"))"
fi

echo
if (( failures > 0 )); then
  echo "${failures} case(s) failed" >&2
  exit 1
fi
echo "test-golden-arming-patch.sh: all cases passed"
