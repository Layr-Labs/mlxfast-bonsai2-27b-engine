#!/usr/bin/env bash
# Unit test for tools/spec-declaration.sh. Offline, no GPU, no serve -- it drives
# the trusted declaration->serve-spec deriver against the REAL contract fixture
# with a series of throwaway manifests, so the envelope it enforces is the one
# that actually ships.
#
# WHAT THIS PROVES, and it is the part worth having: the envelope REFUSALS. The
# decoder and the draft depth become benchd's candidate-spec request, and this
# script is the one trusted bridge that turns the participant's declaration into
# that request -- so an enabled depth outside the declared decoder's
# permitted_draft_depths must be REFUSED here, before any engine spawns. The
# accept cases pin every permitted depth of both decoders; the refuse cases pin
# the first depth above each contract set and the first depth above each
# structural ceiling. If a future fixture edit ever WIDENS
# mtp_head.permitted_draft_depths past 7, or
# dflash_drafter.permitted_draft_depths past 16, the matching case below turns
# this test red -- that is the tripwire.
#
# THE TWO ENVELOPES ARE INDEPENDENT. An mtp declaration is bounded by 1..7 and a
# dflash declaration by 1..16. A depth that one decoder permits and the other
# refuses is pinned in both directions below.
#
# Usage: tools/test-spec-declaration.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DECL="${SCRIPT_DIR}/tools/spec-declaration.sh"
CONTRACT="${SCRIPT_DIR}/fixtures/bonsai2_27b_mlx_v1_track.json"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

fails=0
mani=0

# Write a throwaway manifest and echo its path.
manifest() {
  mani=$((mani + 1))
  local path="${WORK}/manifest-${mani}.json"
  printf '%s' "$1" > "${path}"
  printf '%s' "${path}"
}

# expect_ok <label> <manifest-json> <subcommand> <expected-stdout>
expect_ok() {
  local label="$1" json="$2" sub="$3" want="$4" got rc
  got="$(SPEC_DECLARATION_MANIFEST="$(manifest "${json}")" SPEC_DECLARATION_CONTRACT="${CONTRACT}" \
    "${DECL}" "${sub}" 2>/dev/null)"
  rc=$?
  if [[ ${rc} -eq 0 && "${got}" == "${want}" ]]; then
    echo "PASS ${label} (${sub} => ${got})"
  else
    echo "FAIL ${label}: ${sub} rc=${rc} got='${got}' want rc=0 '${want}'"
    fails=$((fails + 1))
  fi
}

# expect_refuse <label> <manifest-json> <needle-in-stderr>
expect_refuse() {
  local label="$1" json="$2" needle="$3" err rc
  err="$(SPEC_DECLARATION_MANIFEST="$(manifest "${json}")" SPEC_DECLARATION_CONTRACT="${CONTRACT}" \
    "${DECL}" draft-len 2>&1 >/dev/null)"
  rc=$?
  if [[ ${rc} -ne 0 && "${err}" == *"${needle}"* ]]; then
    echo "PASS ${label} (refused: ${err##*REFUSING -- })"
  else
    echo "FAIL ${label}: expected refusal containing '${needle}', got rc=${rc} err='${err}'"
    fails=$((fails + 1))
  fi
}

# --- the contract this test pins --------------------------------------------
# The whole point is to read the SHIPPING envelope, not a copy. Assert it up
# front so a reader knows exactly what the accept/refuse split below rests on.
permitted="$(jq -c '.mtp_head.permitted_draft_depths' "${CONTRACT}")"
if [[ "${permitted}" != "[1,2,3,4,5,6,7]" ]]; then
  echo "FAIL contract envelope: mtp_head.permitted_draft_depths is ${permitted}, expected [1,2,3,4,5,6,7] -- the track's draft-depth envelope changed; update this test deliberately or revert the fixture"
  fails=$((fails + 1))
fi
permitted_dflash="$(jq -c '.dflash_drafter.permitted_draft_depths' "${CONTRACT}")"
if [[ "${permitted_dflash}" != "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]" ]]; then
  echo "FAIL contract envelope: dflash_drafter.permitted_draft_depths is ${permitted_dflash}, expected [1..16] -- the track's DFlash draft-depth envelope changed; update this test deliberately or revert the fixture"
  fails=$((fails + 1))
fi

# --- serial (the no-op) -----------------------------------------------------
expect_ok  "absent-spec-is-serial"        '{}' describe serial
expect_ok  "disabled-is-serial"           '{"spec":{"enabled":false,"num_speculative_tokens":0}}' describe serial
expect_ok  "enabled-zero-is-serial"       '{"spec":{"enabled":true,"num_speculative_tokens":0}}'  describe serial
expect_ok  "dflash-disabled-is-serial"    '{"spec":{"decoder":"dflash","enabled":false,"num_speculative_tokens":7}}' describe serial

# --- the decoder defaults to mtp, so an older declaration keeps its meaning ---
expect_ok  "absent-decoder-is-mtp"        '{}' decoder mtp
expect_ok  "explicit-mtp-decoder"         '{"spec":{"decoder":"mtp","enabled":true,"num_speculative_tokens":3}}' decoder   mtp
expect_ok  "explicit-mtp-describe"        '{"spec":{"decoder":"mtp","enabled":true,"num_speculative_tokens":3}}' describe  mtp3
expect_ok  "explicit-dflash-decoder"      '{"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":7}}' decoder dflash

# --- the seven permitted mtp depths -----------------------------------------
expect_ok  "mtp1-permitted"  '{"spec":{"enabled":true,"num_speculative_tokens":1}}' describe  mtp1
expect_ok  "mtp1-draftlen"   '{"spec":{"enabled":true,"num_speculative_tokens":1}}' draft-len 1
expect_ok  "mtp2-permitted"  '{"spec":{"enabled":true,"num_speculative_tokens":2}}' describe  mtp2
expect_ok  "mtp3-permitted"  '{"spec":{"enabled":true,"num_speculative_tokens":3}}' describe  mtp3
expect_ok  "mtp3-speculative" '{"spec":{"enabled":true,"num_speculative_tokens":3}}' speculative 1
expect_ok  "mtp4-permitted"  '{"spec":{"enabled":true,"num_speculative_tokens":4}}' describe  mtp4
expect_ok  "mtp5-permitted"  '{"spec":{"enabled":true,"num_speculative_tokens":5}}' describe  mtp5
expect_ok  "mtp6-permitted"  '{"spec":{"enabled":true,"num_speculative_tokens":6}}' describe  mtp6
expect_ok  "mtp7-permitted"  '{"spec":{"enabled":true,"num_speculative_tokens":7}}' describe  mtp7
expect_ok  "mtp7-draftlen"   '{"spec":{"enabled":true,"num_speculative_tokens":7}}' draft-len 7

# --- the restriction: depths ABOVE 7 are refused, never clamped -------------
# 8 is inside the structural 0..8 ceiling but OUTSIDE permitted_draft_depths,
# so it must be refused by the contract-membership check. This is the tripwire
# for "cap mtp at 7": widening permitted_draft_depths reddens it.
expect_refuse "mtp8-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":8}}' "not a contract-permitted mtp draft depth"
# 9 is beyond the structural ceiling and refuses there, before the membership check.
expect_refuse "mtp9-refused"  '{"spec":{"enabled":true,"num_speculative_tokens":9}}' "outside the permitted range 0..8"

# --- the sixteen permitted dflash depths ------------------------------------
# The DFlash 2 drafter proposes a whole block per round, so its depth is the
# block size minus one. The drafter was trained at block 8 (depth 7); a larger
# block is legal and simply accepts less.
for depth in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
  expect_ok "dflash${depth}-permitted" \
    "{\"spec\":{\"decoder\":\"dflash\",\"enabled\":true,\"num_speculative_tokens\":${depth}}}" \
    describe "dflash${depth}"
done
expect_ok  "dflash7-draftlen"    '{"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":7}}' draft-len   7
expect_ok  "dflash7-speculative" '{"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":7}}' speculative 1

# --- the two envelopes are independent, in both directions ------------------
# Depth 16 is permitted for dflash and refused for mtp; the mtp refusal is the
# structural ceiling, which is 8 there and 16 here.
expect_refuse "mtp16-refused" '{"spec":{"enabled":true,"num_speculative_tokens":16}}' "outside the permitted range 0..8 for decoder mtp"
expect_refuse "dflash17-refused" '{"spec":{"decoder":"dflash","enabled":true,"num_speculative_tokens":17}}' "outside the permitted range 0..16 for decoder dflash"

# --- malformed declarations refuse ------------------------------------------
expect_refuse "negative-refused"    '{"spec":{"enabled":true,"num_speculative_tokens":-1}}' "outside the permitted range 0..8"
expect_refuse "non-integer-refused" '{"spec":{"enabled":true,"num_speculative_tokens":1.5}}' "must be an integer"
expect_refuse "unknown-key-refused" '{"spec":{"enabled":true,"num_speculative_tokens":1,"foo":1}}' "unknown key"
expect_refuse "bad-enabled-refused" '{"spec":{"enabled":"yes","num_speculative_tokens":1}}' "must be a boolean"
expect_refuse "bad-decoder-refused" '{"spec":{"decoder":"dspark","enabled":true,"num_speculative_tokens":1}}' "is not a declarable decoder"
expect_refuse "bad-decoder-type-refused" '{"spec":{"decoder":3,"enabled":true,"num_speculative_tokens":1}}' "spec.decoder must be a string"

if [[ ${fails} -eq 0 ]]; then
  echo "OK: all spec-declaration envelope cases passed"
  exit 0
fi
echo "FAILED: ${fails} case(s)"
exit 1
