#!/usr/bin/env bash
#
# spec-declaration.sh -- the SINGLE trusted source that reads the participant's
# speculative-decode DECLARATION and derives the decoder and the draft depth the
# ranked run requests from it. tools/bonsai2-27b-measure-and-score.sh resolves
# both THROUGH this one script, so the fail-closed validation and the derivation
# can never drift between the arm gate and the benchd request.
#
# NOT AN EDITABLE PATH. The DECLARATION (mtp-head.manifest.json) is editable --
# a submission EXPRESSES a value there -- but this DERIVATION is trusted, so a
# submission cannot rewrite how the value is interpreted or relax the envelope.
#
# HOW THE DECLARATION REACHES THE ENGINE ON THIS TRACK. The runtime worker is
# in-process, so the decoder and the draft depth ride the Engine Protocol wire
# per request. benchd sends the candidate spec on free_decode_begin and the
# engine echoes `effective_spec`, which the sealed score carries:
#
#   mtp     {"mode":"mtp","mtp":{"depth":N}}          (`benchd iterate --mtp-depth N`)
#   dflash  {"mode":"dflash","dflash":{"depth":N}}    (`benchd iterate --candidate-spec ...`)
#
# With no declaration, or with `enabled: false`, benchd sends no spec and the
# engine runs SERIAL (depth 0) -- the baseline validation's leg.
#
# DECLARATION SHAPE (mtp-head.manifest.json, optional key):
#
#   "spec": { "decoder": "mtp", "enabled": true, "num_speculative_tokens": N }
#
#   decoder absent         => "mtp", so an older declaration keeps its meaning
#   decoder "mtp"          => the pinned MTP head, depth 1..7
#   decoder "dflash"       => the pinned DFlash 2 drafter, depth 1..16
#   enabled=false or N=0   => serial (no spec on the wire), whatever the decoder
#   anything else          => REFUSED here, before any engine spawns
#
# The permitted depth set of each decoder is the contract fixture's, and the two
# sets are independent: `mtp_head.permitted_draft_depths` and
# `dflash_drafter.permitted_draft_depths`.
#
# VERBS
#   speculative  prints 1 when a depth is requested, else 0
#   decoder      prints the declared decoder ("mtp" or "dflash")
#   draft-len    prints the requested depth (0 when serial)
#   describe     prints "serial", "mtpN" or "dflashN"
#   validate     exits 0 when the declaration is well-formed and in-envelope
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MANIFEST="${SPEC_DECLARATION_MANIFEST:-${REPO_ROOT}/mtp-head.manifest.json}"
CONTRACT="${SPEC_DECLARATION_CONTRACT:-${REPO_ROOT}/fixtures/bonsai2_27b_mlx_v1_track.json}"

fail() {
  echo "spec-declaration.sh: REFUSING -- $*" >&2
  exit 1
}

command -v jq >/dev/null 2>&1 || fail "jq is required to read the declaration"

# --- read the declaration ---------------------------------------------------
# An absent manifest OR an absent `spec` block is SERIAL: the pure no-op. It is
# resolved WITHOUT requiring the file to exist, so the stock-repo derivation
# never depends on a spec block being present.
decoder="mtp"
enabled="false"
raw_tokens="0"
if [[ -f "${MANIFEST}" ]]; then
  jq -e . >/dev/null 2>&1 < "${MANIFEST}" || fail "mtp-head.manifest.json is not valid JSON"
  if [[ "$(jq -r 'has("spec")' "${MANIFEST}")" == "true" ]]; then
    [[ "$(jq -r '.spec | type' "${MANIFEST}")" == "object" ]] \
      || fail "the \"spec\" declaration must be an object with keys {decoder, enabled, num_speculative_tokens}"
    # A typo'd key must not silently read as its default: config drift is a
    # refusal, matching the sibling's rejectUnknownSpecKeys.
    unknown="$(jq -r '.spec | keys[] | select(. != "decoder" and . != "enabled" and . != "num_speculative_tokens")' "${MANIFEST}")"
    [[ -z "${unknown}" ]] \
      || fail "the \"spec\" declaration carries unknown key(s): $(printf '%s' "${unknown}" | tr '\n' ' '); allowed keys are decoder, enabled, num_speculative_tokens"
    if [[ "$(jq -r '.spec | has("decoder")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.decoder | type' "${MANIFEST}")" == "string" ]] \
        || fail "spec.decoder must be a string (got $(jq -r '.spec.decoder | type' "${MANIFEST}"))"
      decoder="$(jq -r '.spec.decoder' "${MANIFEST}")"
    fi
    if [[ "$(jq -r '.spec | has("enabled")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.enabled | type' "${MANIFEST}")" == "boolean" ]] \
        || fail "spec.enabled must be a boolean (got $(jq -r '.spec.enabled | type' "${MANIFEST}"))"
      enabled="$(jq -r '.spec.enabled' "${MANIFEST}")"
    fi
    if [[ "$(jq -r '.spec | has("num_speculative_tokens")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}")" == "number" ]] \
        || fail "spec.num_speculative_tokens must be an integer (got $(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}"))"
      raw_tokens="$(jq -r '.spec.num_speculative_tokens' "${MANIFEST}")"
    fi
  fi
fi

# --- resolve the decoder's envelope -----------------------------------------
# Each decoder has its OWN structural ceiling and its OWN contract block. The
# structural ceiling is the outer type/range guard (the knob accepts a declared
# integer, not a hardcoded value); the contract's permitted set is the tighter,
# AUTHORITATIVE one enforced below when the value is enabled.
case "${decoder}" in
  mtp)
    SPEC_MAX_TOKENS=8
    CONTRACT_DEPTHS=".mtp_head.permitted_draft_depths"
    ;;
  dflash)
    SPEC_MAX_TOKENS=16
    CONTRACT_DEPTHS=".dflash_drafter.permitted_draft_depths"
    ;;
  *)
    fail "spec.decoder='${decoder}' is not a declarable decoder; this track declares mtp or dflash"
    ;;
esac

# integer + structural 0..SPEC_MAX_TOKENS ceiling
printf '%s' "${raw_tokens}" | grep -Eq '^-?[0-9]+$' \
  || fail "spec.num_speculative_tokens must be an integer (got '${raw_tokens}')"
if (( raw_tokens < 0 || raw_tokens > SPEC_MAX_TOKENS )); then
  fail "spec.num_speculative_tokens=${raw_tokens} is outside the permitted range 0..${SPEC_MAX_TOKENS} for decoder ${decoder}"
fi

# --- derive the effective serve spec ----------------------------------------
# enabled:false OR num_speculative_tokens 0 => serial (the no-op). enabled:true
# with N>0 must be a CONTRACT-permitted depth for the declared decoder.
spec="0"
draft="0"
if [[ "${enabled}" == "true" && "${raw_tokens}" -gt 0 ]]; then
  # The fixture's permitted set is the authority. Enforce membership so the
  # serve can never boot a depth the scored envelope would refuse. When the
  # contract declares no such set, the structural ceiling above stands alone.
  if jq -e "${CONTRACT_DEPTHS} | arrays" >/dev/null 2>&1 < "${CONTRACT}"; then
    if [[ "$(jq -r --argjson n "${raw_tokens}" "(${CONTRACT_DEPTHS} | index(\$n)) != null" "${CONTRACT}")" != "true" ]]; then
      permitted="$(jq -r "${CONTRACT_DEPTHS} | map(tostring) | join(\", \")" "${CONTRACT}")"
      fail "spec.num_speculative_tokens=${raw_tokens} is not a contract-permitted ${decoder} draft depth (permitted_draft_depths: ${permitted})"
    fi
  fi
  spec="1"
  draft="${raw_tokens}"
fi

case "${1:-}" in
  speculative) echo "${spec}" ;;
  decoder)     echo "${decoder}" ;;
  draft-len)   echo "${draft}" ;;
  describe)    if [[ "${spec}" == "1" ]]; then echo "${decoder}${draft}"; else echo "serial"; fi ;;
  validate)    : ;;  # validation already ran above; a clean exit means valid
  *)
    echo "spec-declaration.sh: usage: spec-declaration.sh {speculative|decoder|draft-len|describe|validate}" >&2
    exit 2
    ;;
esac
