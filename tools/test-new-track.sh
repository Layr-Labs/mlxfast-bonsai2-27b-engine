#!/usr/bin/env bash
# test-new-track.sh -- tools/new-track.sh stamps a seeded repo into a new track.
#
# Offline, no GPU, no toolchain, no Hugging Face. A throwaway python3
# http.server on 127.0.0.1 stands in for the Hugging Face tree API
# (HF_API_BASE_URL), so the checkpoint-pinning path -- URL assembly, the
# LFS/non-LFS split, the omission rule -- runs against real bytes over a real
# socket without any credential or any network egress.
#
# The subject is a SEED: the tracked tree of this repository (git archive), in
# its own git repository with one commit, exactly the shape
# docs/new-track-repo-procedure.md section 1 describes. The stamp runs there,
# never here.
#
# WHAT THIS PROVES: the stamped tree is internally consistent -- the manifest,
# the contract fixture, the runner label, the fork revision and the channel all name
# the NEW track -- and the repository's own linter still passes on it. Plus the
# refusals, which are the part worth having: a dirty tree, a malformed track id,
# and a checkpoint file the tree API publishes no sha256 for.
#
# Usage: tools/test-new-track.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -rf "${WORK}"
}
trap cleanup EXIT

failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
pass() { echo "ok: $*"; }
# A group of assertions reports ONE line, and it reports "ok" only if every
# assertion inside it held -- an unconditional pass after inline checks would
# print "ok" over its own failures.
group_mark=0
group() { group_mark="${failures}"; }
group_ok() { [[ "${failures}" -eq "${group_mark}" ]] && pass "$*"; return 0; }

# --- the stamp's inputs ------------------------------------------------------
NEW_TRACK="demo1.0-9b-a1b-mlx-v1"
NEW_STEM="demo1_0_9b_a1b_mlx_v1_track"
NEW_NAME="mlxfast-demo10-9b-a1b"
FORK_SHA="1111111111111111111111111111111111111111"
CKPT_REPO="org/model"
CKPT_REV="2222222222222222222222222222222222222222"
BAD_REV="3333333333333333333333333333333333333333"

OLD_TRACK="$(python3 -c 'import json;print(json.load(open("'"${ROOT}"'/benchmark.json"))["trackId"])')"
OLD_CONTRACT="$(python3 -c 'import json;print(json.load(open("'"${ROOT}"'/benchmark.json"))["contractPath"])')"

# THE EXEMPTION LIST -- files allowed to keep naming the SOURCE track after a
# stamp. Each is a RECORD of the source track, so rewriting the id inside it
# would falsify the record rather than update it:
#
#   docs/bonsai2-27b-port-notes.md       the source track's porting history and
#                                        its citations. History, not identity.
#   tools/lint-benchmark-manifest.py     the per-track scoring registry pins
#                                        David's rulings BY TRACK ID. The new
#                                        track legitimately has no entry yet,
#                                        and the linter reports that absence as
#                                        a visible gap; renaming the qwen keys
#                                        would both falsify the citations and
#                                        silently un-register the real track.
#   docs/new-track-repo-procedure.md     the cross-track procedure, which cites
#                                        precedent track ids by name.
#   tools/new-track.sh                   the stamping tool itself. Its examples
#                                        and citations describe the TEMPLATE, and
#                                        rewriting the RUNNING script would be
#                                        unsafe besides: bash reads a script
#                                        incrementally, so an in-place edit
#                                        shifts the byte offsets of every step it
#                                        has not read yet. This was a real defect
#                                        -- it silently skipped the --os rewrite.
#   tools/test-new-track.sh              this file, for the same reason.
#
# This list must match EXEMPT in tools/new-track.sh.
EXEMPTIONS=(
  "docs/bonsai2-27b-port-notes.md"
  "tools/lint-benchmark-manifest.py"
  "docs/new-track-repo-procedure.md"
  "tools/new-track.sh"
  "tools/test-new-track.sh"
)

# --- the stand-in Hugging Face tree API --------------------------------------
# SimpleHTTPRequestHandler strips the query string, so ?recursive=true resolves
# to the plain path and one file per revision is enough.
HF="${WORK}/hf/api/models/${CKPT_REPO}/tree"
mkdir -p "${HF}"

# The good tree. Every pinnable file is LFS-backed (lfs.oid IS the sha256), and
# it also carries the three kinds the manifest deliberately omits --
# .gitattributes, LICENSE and an image -- which must NOT trip the non-LFS
# refusal, because they are never pinned in the first place.
cat > "${HF}/${CKPT_REV}" <<'JSONEOF'
[
  {"type": "directory", "path": "subdir", "oid": "aaaa"},
  {"type": "file", "path": ".gitattributes", "oid": "b0", "size": 1570},
  {"type": "file", "path": "LICENSE", "oid": "b1", "size": 3235},
  {"type": "file", "path": "logo.png", "oid": "b2", "size": 11203},
  {"type": "file", "path": "config.json", "oid": "b3", "size": 5361,
   "lfs": {"oid": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "size": 5361}},
  {"type": "file", "path": "model.safetensors.index.json", "oid": "b4", "size": 420813,
   "lfs": {"oid": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "size": 420813}},
  {"type": "file", "path": "model-00001-of-00001.safetensors", "oid": "b5", "size": 5283195887,
   "lfs": {"oid": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", "size": 5283195887}},
  {"type": "file", "path": "tokenizer.json", "oid": "b6", "size": 12809320,
   "lfs": {"oid": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "size": 12809320}}
]
JSONEOF

# The bad tree: one file the API publishes NO sha256 for (a plain git blob --
# `oid` there is the git sha1, not the content digest the manifest pins).
cat > "${HF}/${BAD_REV}" <<'JSONEOF'
[
  {"type": "file", "path": "config.json", "oid": "b3", "size": 5361,
   "lfs": {"oid": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "size": 5361}},
  {"type": "file", "path": "generation_config.json", "oid": "deadbeef", "size": 202}
]
JSONEOF

PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
( cd "${WORK}/hf" && exec python3 -m http.server "${PORT}" --bind 127.0.0.1 >/dev/null 2>&1 ) &
SERVER_PID=$!

ready=0
for _ in $(seq 1 50); do
  if curl --fail --silent --max-time 1 -o /dev/null "http://127.0.0.1:${PORT}/api/models/${CKPT_REPO}/tree/${CKPT_REV}"; then
    ready=1; break
  fi
  sleep 0.2
done
[[ "${ready}" == "1" ]] || { echo "FAIL: the stand-in tree API did not come up on 127.0.0.1:${PORT}" >&2; exit 1; }
export HF_API_BASE_URL="http://127.0.0.1:${PORT}"

# --- seeding -----------------------------------------------------------------
# seed <dir> -- a fresh single-commit repo holding this repository's TRACKED
# tree. Vendor/mlx-swift-lm is a VENDORED TREE of plain files, so git archive
# carries it like any other source and there is no gitlink to re-create. The
# -f on the add is what a fresh cut of the tree itself needs: the fork's own
# .gitignore has a `mlx-swift-lm/` rule that would drop skills/mlx-swift-lm/.
seed() {
  local dir="$1"
  mkdir -p "${dir}"
  git -C "${ROOT}" archive HEAD | tar -x -C "${dir}"
  git -C "${dir}" init -q
  git -C "${dir}" add -A -f
  git -C "${dir}" -c user.name=seed -c user.email=seed@example.com \
    commit -q --no-gpg-sign -m "seed" >/dev/null
}

# --- case 1: the happy path --------------------------------------------------
SEED="${WORK}/seed"
seed "${SEED}"
if ( cd "${SEED}" && tools/new-track.sh \
      --track-id "${NEW_TRACK}" \
      --fork-sha "${FORK_SHA}" \
      --checkpoint "${CKPT_REPO}@${CKPT_REV}" ) > "${WORK}/stamp.out" 2>&1; then
  pass "the stamp ran and the stamped tree passes tools/lint-benchmark-manifest.py"
else
  fail "the stamp refused a good invocation; output:"
  sed 's/^/    /' "${WORK}/stamp.out" >&2
fi

jqf() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps(eval('d'+sys.argv[2])))" "$@"; }

# benchmark.json
group
M="${SEED}/benchmark.json"
[[ "$(jqf "${M}" '["name"]')"                    == "\"${NEW_NAME}\""  ]] || fail "benchmark.json name: $(jqf "${M}" '["name"]')"
[[ "$(jqf "${M}" '["trackId"]')"                 == "\"${NEW_TRACK}\"" ]] || fail "benchmark.json trackId: $(jqf "${M}" '["trackId"]')"
[[ "$(jqf "${M}" '["staticReviewTrackId"]')"     == "\"${NEW_TRACK}\"" ]] || fail "benchmark.json staticReviewTrackId: $(jqf "${M}" '["staticReviewTrackId"]')"
[[ "$(jqf "${M}" '["leaderboard"]["namespace"]')" == "\"${NEW_TRACK}\"" ]] || fail "benchmark.json leaderboard.namespace: $(jqf "${M}" '["leaderboard"]["namespace"]')"
[[ "$(jqf "${M}" '["contractPath"]')"            == "\"fixtures/${NEW_STEM}.json\"" ]] || fail "benchmark.json contractPath: $(jqf "${M}" '["contractPath"]')"
group_ok "benchmark.json names the new track (name, trackId, staticReviewTrackId, namespace, contractPath)"

# The manifest's harness and ruled scoring constants are NOT identity and must
# not move.
group
for key in editablePaths optionalEditablePaths editableSurfaceByteBudget scoring \
           setupCommand preSubmitCommand benchmarkCommand; do
  if [[ "$(jqf "${M}" "[\"${key}\"]")" != "$(jqf "${ROOT}/benchmark.json" "[\"${key}\"]")" ]]; then
    fail "benchmark.json ${key} changed; the stamp must leave the harness and the scoring constants alone"
  fi
done
group_ok "benchmark.json editablePaths, budget, scoring and the three commands are untouched"

# the contract fixture
group
F="${SEED}/fixtures/${NEW_STEM}.json"
if [[ -f "${F}" ]]; then
  [[ ! -f "${SEED}/${OLD_CONTRACT}" ]] || fail "the old fixture ${OLD_CONTRACT} still exists"
  [[ "$(jqf "${F}" '["track_id"]')"      == "\"${NEW_TRACK}\"" ]] || fail "fixture track_id: $(jqf "${F}" '["track_id"]')"
  [[ "$(jqf "${F}" '["benchmark_name"]')" == "\"${NEW_NAME}\"" ]] || fail "fixture benchmark_name: $(jqf "${F}" '["benchmark_name"]')"
  [[ "$(jqf "${F}" '["official_scoring_enabled"]')" == "false" ]] || fail "fixture official_scoring_enabled is not false"
  [[ "$(jqf "${F}" '.get("official_baseline","__ABSENT__")')" == "\"__ABSENT__\"" ]] || fail "fixture still carries official_baseline; a new track has no measured baseline"
  [[ "$(jqf "${F}" '["mlx_swift_lm_revision"]')" == "\"${FORK_SHA}\"" ]] || fail "fixture mlx_swift_lm_revision: $(jqf "${F}" '["mlx_swift_lm_revision"]')"
  [[ "$(jqf "${F}" '["live_golden"]')"           == '""' ]] || fail "fixture live_golden is not empty"
  [[ "$(jqf "${F}" '["timed_prompt_pool"]')"     == "[]" ]] || fail "fixture timed_prompt_pool is not empty"
  [[ "$(jqf "${F}" '["live_golden_speculative"]')" == "{}" ]] || fail "fixture live_golden_speculative is not empty"
  [[ "$(jqf "${F}" '["hidden_correctness_golden"]["sha256"]')" == '"DEMO1-0-9B-A1B-MLX-V1-PENDING-ORGANIZER"' ]] \
    || fail "fixture hidden_correctness_golden sentinel: $(jqf "${F}" '["hidden_correctness_golden"]["sha256"]')"
  for role in local_iterate local_submit; do
    [[ "$(jqf "${F}" "[\"public_captures\"][\"${role}\"][\"sha256\"]")" == '"DEMO1-0-9B-A1B-MLX-V1-PENDING-ORGANIZER"' ]] \
      || fail "fixture public_captures.${role} sentinel: $(jqf "${F}" "[\"public_captures\"][\"${role}\"]")"
    [[ "$(jqf "${F}" "[\"public_captures\"][\"${role}\"][\"r2_path\"]")" == "\"correctness_prompts/${NEW_TRACK}/public-${role//_/-}.golden.json\"" ]] \
      || fail "fixture public_captures.${role} r2_path: $(jqf "${F}" "[\"public_captures\"][\"${role}\"][\"r2_path\"]")"
  done
  [[ "$(jqf "${F}" '["target"]["upstream_model_id"]')" == "\"${CKPT_REPO}\"" ]] || fail "fixture target.upstream_model_id: $(jqf "${F}" '["target"]["upstream_model_id"]')"
  [[ "$(jqf "${F}" '["target"]["upstream_revision"]')" == "\"${CKPT_REV}\"" ]] || fail "fixture target.upstream_revision: $(jqf "${F}" '["target"]["upstream_revision"]')"
  group_ok "the new fixture carries the stamped values and the old one is gone"
else
  fail "the new fixture fixtures/${NEW_STEM}.json was not written"
fi

# the checkpoint file list
group
MANIFEST="${SEED}/$(jqf "${F}" '["target"]["manifest_path"]' 2>/dev/null | tr -d '"')"
if [[ -f "${MANIFEST}" ]]; then
  records="$(grep -cE '^[0-9a-f]{64} [0-9]+ ' "${MANIFEST}")"
  [[ "${records}" == "4" ]] || fail "the checkpoint manifest pinned ${records} files, expected 4 (the LFS entries only)"
  grep -q "^aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 5361 config.json$" "${MANIFEST}" \
    || fail "the checkpoint manifest does not pin config.json at its lfs.oid"
  for omitted in .gitattributes LICENSE logo.png; do
    grep -qE "^[0-9a-f]{64} [0-9]+ ${omitted}$" "${MANIFEST}" \
      && fail "the checkpoint manifest pinned ${omitted}; the template rule omits it"
  done
  grep -q "Revision: ${CKPT_REV}" "${MANIFEST}" || fail "the checkpoint manifest does not name the pinned revision"
  group_ok "the checkpoint file list is the tree's LFS entries at their lfs.oid, with .gitattributes/LICENSE/images omitted"
else
  fail "the checkpoint manifest was not written"
fi

# the ranked runner label
if grep -qF "runs-on: [self-hosted, macOS, ${NEW_TRACK}]" "${SEED}/.github/workflows/benchmark.yml"; then
  pass "runs-on is [self-hosted, macOS, ${NEW_TRACK}]"
else
  fail "runs-on label: $(grep -n 'runs-on: \[' "${SEED}/.github/workflows/benchmark.yml" | tr '\n' ' ')"
fi

# the benchd channel
if grep -qF 'BRANCH="${BENCHD_BRANCH:-main}"' "${SEED}/tools/fetch-benchd.sh"; then
  pass "tools/fetch-benchd.sh defaults BENCHD_BRANCH to main (benchd is published from main)"
else
  fail "BENCHD_BRANCH default: $(grep -n '^BRANCH=' "${SEED}/tools/fetch-benchd.sh")"
fi

# the engine fork pin. The fork is a vendored tree, so the pin the stamp moves
# is the contract fixture's mlx_swift_lm_revision, not a gitlink.
fork_rev="$(jqf "${F}" '["mlx_swift_lm_revision"]')"
if [[ "${fork_rev}" == "\"${FORK_SHA}\"" ]]; then
  pass "the contract fixture's mlx_swift_lm_revision is the fork sha"
else
  fail "mlx_swift_lm_revision is ${fork_rev}, expected \"${FORK_SHA}\""
fi

# the goldens
# They are never in git: a track's goldens are published in R2 and staged on its
# own box. So the stamped seed must carry NEITHER track's goldens, and there is
# no directory for the tool to rename.
group
if [[ -d "${SEED}/correctness_prompts/${NEW_TRACK}" || -d "${SEED}/correctness_prompts/${OLD_TRACK}" ]]; then
  fail "the stamped seed carries a track goldens directory: $(ls "${SEED}/correctness_prompts")"
else
  group_ok "the stamped seed carries no track goldens (they live in R2 and on the box)"
fi

# the source track id is gone everywhere but the exemptions
mapfile -t leftovers < <(cd "${SEED}" && git grep -l -F "${OLD_TRACK}" -- . 2>/dev/null || true)
unexpected=()
for path in ${leftovers[@]+"${leftovers[@]}"}; do
  keep=0
  for ex in "${EXEMPTIONS[@]}"; do [[ "${path}" == "${ex}" ]] && keep=1; done
  [[ "${keep}" == "0" ]] && unexpected+=("${path}")
done
if [[ "${#unexpected[@]}" -eq 0 ]]; then
  pass "'${OLD_TRACK}' survives only in the ${#EXEMPTIONS[@]} exempted records"
else
  fail "'${OLD_TRACK}' still appears in un-exempted files: ${unexpected[*]}"
fi

# the old fixture stem is gone everywhere but the same exempted records
mapfile -t stem_left < <(cd "${SEED}" && git grep -l -F "$(basename "${OLD_CONTRACT}" .json)" -- . 2>/dev/null || true)
stem_unexpected=()
for path in ${stem_left[@]+"${stem_left[@]}"}; do
  keep=0
  for ex in "${EXEMPTIONS[@]}"; do [[ "${path}" == "${ex}" ]] && keep=1; done
  [[ "${keep}" == "0" ]] && stem_unexpected+=("${path}")
done
if [[ "${#stem_unexpected[@]}" -eq 0 ]]; then
  pass "no tool, test or workflow outside the exemptions still opens the old contract fixture by name"
else
  fail "the old fixture stem still appears in un-exempted files: ${stem_unexpected[*]}"
fi

# --- case 1b: --os moves the OS token ---------------------------------------
# The mlx default is macOS, which is what the template already carries, so the
# happy path above cannot tell a working rewrite from a no-op. A cuda track
# defaults to Linux and does.
OSSEED="${WORK}/osseed"
seed "${OSSEED}"
if ( cd "${OSSEED}" && tools/new-track.sh --track-id "qwen4.0-200b-a8b-cuda-v2" \
      --fork-sha "${FORK_SHA}" --checkpoint "${CKPT_REPO}@${CKPT_REV}" ) > "${WORK}/os.out" 2>&1; then
  group
  grep -qF "runs-on: [self-hosted, Linux, qwen4.0-200b-a8b-cuda-v2]" "${OSSEED}/.github/workflows/benchmark.yml" \
    || fail "a cuda track did not get the Linux runner label: $(grep -n 'runs-on: \[self-hosted' "${OSSEED}/.github/workflows/benchmark.yml" | tr '\n' ' ')"
  [[ "$(jqf "${OSSEED}/benchmark.json" '["name"]')" == '"cudafast-qwen40-200b-a8b"' ]] \
    || fail "a cuda track's manifest name is not cudafast-*: $(jqf "${OSSEED}/benchmark.json" '["name"]')"
  group_ok "a cuda track stamps cudafast-* and [self-hosted, Linux, <track>]"
else
  fail "the stamp refused a cuda track; output: $(tail -5 "${WORK}/os.out")"
fi

# --- case 2: a dirty tree refuses -------------------------------------------
DIRTY="${WORK}/dirty"
seed "${DIRTY}"
echo "# uncommitted" >> "${DIRTY}/README.md"
if ( cd "${DIRTY}" && tools/new-track.sh --track-id "${NEW_TRACK}" --fork-sha "${FORK_SHA}" \
      --checkpoint "${CKPT_REPO}@${CKPT_REV}" ) > "${WORK}/dirty.out" 2>&1; then
  fail "the stamp ran on a dirty worktree"
else
  group
  grep -qi "dirty" "${WORK}/dirty.out" || fail "the dirty-tree refusal does not say so: $(cat "${WORK}/dirty.out")"
  [[ "$(jqf "${DIRTY}/benchmark.json" '["trackId"]')" == "\"${OLD_TRACK}\"" ]] \
    || fail "the dirty-tree refusal still rewrote benchmark.json"
  group_ok "a dirty worktree is refused and nothing is rewritten"
fi

# --- case 3: a malformed track id refuses ------------------------------------
BADID="${WORK}/badid"
seed "${BADID}"
group
for bad in "demo1.0-9b-a1b-mlx" "demo1.0-9b-a1b-metal-v1" "demo1.0-mlx-v1" "DEMO1.0-9b-a1b-mlx-v1"; do
  if ( cd "${BADID}" && tools/new-track.sh --track-id "${bad}" --fork-sha "${FORK_SHA}" \
        --checkpoint "${CKPT_REPO}@${CKPT_REV}" ) > "${WORK}/badid.out" 2>&1; then
    fail "the stamp accepted the malformed track id '${bad}'"
  elif ! grep -qF -- "${bad}" "${WORK}/badid.out"; then
    fail "the refusal of '${bad}' does not name it: $(cat "${WORK}/badid.out")"
  fi
done
[[ -n "$(cd "${BADID}" && git status --porcelain)" ]] && fail "a malformed track id left the tree modified"
group_ok "a malformed track id is refused by name and nothing is written"

# --- case 4: a checkpoint file with no sha256 refuses by name ----------------
NOLFS="${WORK}/nolfs"
seed "${NOLFS}"
if ( cd "${NOLFS}" && tools/new-track.sh --track-id "${NEW_TRACK}" --fork-sha "${FORK_SHA}" \
      --checkpoint "${CKPT_REPO}@${BAD_REV}" ) > "${WORK}/nolfs.out" 2>&1; then
  fail "the stamp pinned a checkpoint whose tree publishes no sha256 for a file"
else
  group
  grep -qF "generation_config.json" "${WORK}/nolfs.out" \
    || fail "the non-LFS refusal does not name the file: $(cat "${WORK}/nolfs.out")"
  group_ok "a checkpoint file the tree API publishes no sha256 for is refused BY NAME"
fi

# --- verdict -----------------------------------------------------------------
if [[ "${failures}" -eq 0 ]]; then
  echo "test-new-track.sh: all cases pass"
  exit 0
fi
echo "test-new-track.sh: ${failures} case(s) failed" >&2
exit 1
