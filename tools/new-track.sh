#!/usr/bin/env bash
#
# new-track.sh -- stamp a NEW track into a freshly seeded engine repository.
#
# Run this from the ROOT of a fresh seed: a tree-identical copy of this
# repository's tip, made the way docs/new-track-repo-procedure.md section 1
# describes (one signed commit, no history). The seed still carries the SOURCE
# track's identity everywhere -- the manifest, the contract fixture, the runner
# label, the docs, the checkpoint pin. This script replaces that identity with
# the new track's, in one pass, so the stamped tree is internally consistent
# and the repository's own checks still pass.
#
# Usage:
#   tools/new-track.sh --track-id <{model}{ver}-{params}-{platform}-v{N}> \
#                      --fork-sha <40 hex> \
#                      --checkpoint <hf_repo>@<40 hex revision> \
#                      [--bench-commit <40 hex>] \
#                      [--os macOS|Linux] \
#                      [--hash-non-lfs]
#
# WHAT IT DOES NOT DO: it never commits, never pushes, and never touches the
# network except for the Hugging Face tree read that pins the checkpoint. The
# operator reviews the working tree and commits.
#
# WHAT IT CANNOT DO: it copies MODEL FACTS from the template. The new fixture's
# layer counts, head widths, expert counts, MoE geometry and n-gram shape are
# the SOURCE model's, and they are wrong for any different model family. The
# script prints this loudly at the end; re-authoring them is the operator's
# first job after stamping.
#
# Env:
#   HF_API_BASE_URL      Hugging Face API base. Default https://huggingface.co.
#                        Overridden by tools/test-new-track.sh, which serves a
#                        stub tree from 127.0.0.1.
#   HF_RESOLVE_BASE_URL  Base for --hash-non-lfs downloads. Default
#                        https://huggingface.co.
set -euo pipefail

REPO_ROOT="$(pwd -P)"
SCRIPT_NAME="new-track.sh"

die() { echo "${SCRIPT_NAME}: ${*}" >&2; exit 1; }
note() { echo "${SCRIPT_NAME}: ${*}" >&2; }

usage() {
  sed -n '15,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# --- arguments ---------------------------------------------------------------
TRACK_ID=""
FORK_SHA=""
CHECKPOINT=""
BENCH_COMMIT=""
OS_TOKEN=""
HASH_NON_LFS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track-id)     TRACK_ID="${2:-}"; shift 2 ;;
    --fork-sha)     FORK_SHA="${2:-}"; shift 2 ;;
    --checkpoint)   CHECKPOINT="${2:-}"; shift 2 ;;
    --bench-commit) BENCH_COMMIT="${2:-}"; shift 2 ;;
    --os)           OS_TOKEN="${2:-}"; shift 2 ;;
    --hash-non-lfs) HASH_NON_LFS=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) die "unknown argument '$1'. Run with --help." ;;
  esac
done

[[ -n "${TRACK_ID}" ]]   || die "--track-id is required."
[[ -n "${FORK_SHA}" ]]   || die "--fork-sha is required."
[[ -n "${CHECKPOINT}" ]] || die "--checkpoint is required."

# Track id shape. Three groups plus the ruled suffix: a model+version part, a
# params part, then -{platform}-v{N}. The platform is closed (mlx|cuda) because
# it selects the manifest name prefix and the default runner OS below.
if [[ ! "${TRACK_ID}" =~ ^[a-z0-9]+[a-z0-9.]*-[a-z0-9]+[a-z0-9-]*-(mlx|cuda)-v[0-9]+$ ]]; then
  die "--track-id '${TRACK_ID}' is not {model}{ver}-{params}-{platform}-v{N} with platform mlx or cuda (e.g. qwen3.8-125b-a6b-mlx-v1)."
fi
PLATFORM="$(printf '%s' "${TRACK_ID}" | sed -E 's/^.*-(mlx|cuda)-v[0-9]+$/\1/')"

if [[ "${#FORK_SHA}" -ne 40 || -n "${FORK_SHA//[0-9a-f]/}" ]]; then
  die "--fork-sha '${FORK_SHA}' is not 40 lowercase hex characters."
fi

if [[ ! "${CHECKPOINT}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+@[0-9a-f]{40}$ ]]; then
  die "--checkpoint '${CHECKPOINT}' is not <hf_repo>@<40 hex revision>, e.g. org/model@0123456789abcdef0123456789abcdef01234567."
fi
CKPT_REPO="${CHECKPOINT%@*}"
CKPT_REV="${CHECKPOINT##*@}"

if [[ -n "${BENCH_COMMIT}" ]]; then
  if [[ "${#BENCH_COMMIT}" -ne 40 || -n "${BENCH_COMMIT//[0-9a-f]/}" ]]; then
    die "--bench-commit '${BENCH_COMMIT}' is not 40 lowercase hex characters."
  fi
fi

if [[ -z "${OS_TOKEN}" ]]; then
  if [[ "${PLATFORM}" == "mlx" ]]; then OS_TOKEN="macOS"; else OS_TOKEN="Linux"; fi
fi
if [[ "${OS_TOKEN}" != "macOS" && "${OS_TOKEN}" != "Linux" ]]; then
  die "--os '${OS_TOKEN}' is not macOS or Linux."
fi

# --- preconditions -----------------------------------------------------------
command -v git >/dev/null 2>&1     || die "git is required."
command -v python3 >/dev/null 2>&1 || die "python3 is required."
command -v curl >/dev/null 2>&1    || die "curl is required to read the Hugging Face tree."

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not a git work tree: ${REPO_ROOT}"
[[ -f "${REPO_ROOT}/benchmark.json" ]] || die "benchmark.json not found; run this from the repository ROOT."

# A dirty tree is refused: the stamp rewrites files across the whole repository
# and the operator's review is a `git diff` against the seed. Uncommitted work
# would be indistinguishable from what the stamp did.
if [[ -n "$(git status --porcelain)" ]]; then
  die "the worktree is dirty. Commit or stash first -- the stamp's review surface IS the diff against the seed."
fi

OLD_TRACK_ID="$(python3 -c 'import json;print(json.load(open("benchmark.json"))["trackId"])')"
OLD_CONTRACT="$(python3 -c 'import json;print(json.load(open("benchmark.json"))["contractPath"])')"
OLD_CONTRACT_STEM="$(basename "${OLD_CONTRACT}" .json)"

[[ "${OLD_TRACK_ID}" != "${TRACK_ID}" ]] || die "benchmark.json already carries trackId '${TRACK_ID}'; this tree is already stamped."

# --- derived names -----------------------------------------------------------
# The contract fixture stem: the track id with '-' and '.' folded to '_'.
NEW_CONTRACT_STEM="$(printf '%s' "${TRACK_ID}" | tr '.-' '__')_track"
NEW_CONTRACT="fixtures/${NEW_CONTRACT_STEM}.json"

# The manifest name: {mlxfast|cudafast} + the track id with the -{platform}-v{N}
# suffix cut and the dots removed (qwen3.8-125b-a6b-mlx-v1 -> qwen38-125b-a6b).
MODEL_PART="$(printf '%s' "${TRACK_ID}" | sed -E "s/-${PLATFORM}-v[0-9]+$//" | tr -d '.')"
if [[ "${PLATFORM}" == "mlx" ]]; then NEW_NAME="mlxfast-${MODEL_PART}"; else NEW_NAME="cudafast-${MODEL_PART}"; fi

# The pending-golden sentinel, in the form this repository's own history used
# (the source track's contract fixture used the same construction).
SENTINEL="$(printf '%s' "${TRACK_ID}" | tr '[:lower:]' '[:upper:]' | tr '.' '-')-PENDING-ORGANIZER"

note "stamping ${OLD_TRACK_ID} -> ${TRACK_ID}"
note "  manifest name    ${NEW_NAME}"
note "  contract fixture ${NEW_CONTRACT}"
note "  runner label     [self-hosted, ${OS_TOKEN}, ${TRACK_ID}]"

# Files this pass must NOT rewrite. Each is a record of the SOURCE track, and
# renaming the id inside it would falsify that record:
#   * the port notes are the source track's porting history and citations;
#   * the linter's per-track scoring registry pins David's rulings BY TRACK ID
#     -- the new track legitimately has no entry yet, and the linter reports
#     that as a visible gap rather than a silent pass;
#   * the new-track procedure cites precedent track ids by name;
#   * this script and its test are the STAMPING TOOLS. Their examples and
#     citations describe the TEMPLATE, so rewriting them would make the
#     citations false -- and rewriting THIS FILE would be actively unsafe:
#     bash reads a script incrementally, so an in-place edit of the running
#     script shifts every later byte offset and garbles the steps that have
#     not been read yet.
EXEMPT=(
  "docs/bonsai2-27b-port-notes.md"
  "tools/lint-benchmark-manifest.py"
  "docs/new-track-repo-procedure.md"
  "tools/new-track.sh"
  "tools/test-new-track.sh"
)

# --- 1. benchmark.json -------------------------------------------------------
# Commands, editablePaths, budget and scoring are deliberately untouched: they
# are the track's harness and its ruled scoring constants, not its identity.
python3 - <<PYEOF || die "rewriting benchmark.json failed."
import json
path = "benchmark.json"
with open(path, encoding="utf-8") as fh:
    m = json.load(fh)
m["name"] = "${NEW_NAME}"
m["trackId"] = "${TRACK_ID}"
m["staticReviewTrackId"] = "${TRACK_ID}"
m["leaderboard"]["namespace"] = "${TRACK_ID}"
m["contractPath"] = "${NEW_CONTRACT}"
with open(path, "w", encoding="utf-8") as fh:
    json.dump(m, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
PYEOF
note "rewrote benchmark.json (name, trackId, staticReviewTrackId, leaderboard.namespace, contractPath)"

# --- 2. the contract fixture -------------------------------------------------
python3 - <<PYEOF || die "writing ${NEW_CONTRACT} failed."
import json
with open("${OLD_CONTRACT}", encoding="utf-8") as fh:
    c = json.load(fh)

c["track_id"] = "${TRACK_ID}"
c["track_name"] = "${TRACK_ID} -- composite cohort speedup"
c["benchmark_name"] = "${NEW_NAME}"
c["official_scoring_enabled"] = False
c.pop("official_baseline", None)
c["mlx_swift_lm_revision"] = "${FORK_SHA}"
c["live_golden"] = ""
c["timed_prompt_pool"] = []
c["hidden_correctness_golden"] = {"sha256": "${SENTINEL}", "bytes": 0}
c["live_golden_speculative"] = {}

target = c.get("target", {})
target["upstream_model_id"] = "${CKPT_REPO}"
target["upstream_revision"] = "${CKPT_REV}"

with open("${NEW_CONTRACT}", "w", encoding="utf-8") as fh:
    json.dump(c, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
PYEOF
rm -f "${OLD_CONTRACT}"
note "wrote ${NEW_CONTRACT} and removed ${OLD_CONTRACT}"

# --- 3. the checkpoint file list --------------------------------------------
# The checkpoint's file list is the reference manifest the fixture's
# target.manifest_path names; setup.sh derives BOTH what it pins and what it
# downloads from that one file, so regenerating it here is what actually
# re-aims provisioning at the new checkpoint.
MANIFEST_REL="$(python3 -c 'import json;print(json.load(open("'"${NEW_CONTRACT}"'"))["target"]["manifest_path"])')"
[[ -n "${MANIFEST_REL}" ]] || die "the fixture's target.manifest_path is empty; nothing to write the checkpoint file list to."

HF_API_BASE_URL="${HF_API_BASE_URL:-https://huggingface.co}"
HF_RESOLVE_BASE_URL="${HF_RESOLVE_BASE_URL:-https://huggingface.co}"
TREE_URL="${HF_API_BASE_URL}/api/models/${CKPT_REPO}/tree/${CKPT_REV}?recursive=true"
TREE_JSON="$(mktemp)"
trap 'rm -f "${TREE_JSON}"' EXIT

note "reading the checkpoint tree: ${TREE_URL}"
curl -fsSL --retry 3 --retry-delay 2 -o "${TREE_JSON}" "${TREE_URL}" \
  || die "could not read the Hugging Face tree for ${CKPT_REPO}@${CKPT_REV} (${TREE_URL}). A private repo needs a token; this script takes none by design."

python3 - "${TREE_JSON}" "${MANIFEST_REL}" "${HASH_NON_LFS}" "${HF_RESOLVE_BASE_URL}" <<PYEOF || die "pinning the checkpoint file list failed."
import hashlib, json, os, subprocess, sys

tree_path, manifest_path, hash_non_lfs, resolve_base = sys.argv[1:5]
hash_non_lfs = hash_non_lfs == "1"
repo, rev, track = "${CKPT_REPO}", "${CKPT_REV}", "${TRACK_ID}"

with open(tree_path, encoding="utf-8") as fh:
    tree = json.load(fh)
if not isinstance(tree, list):
    sys.exit("new-track.sh: the tree endpoint did not return a list; got %s" % type(tree).__name__)

# The inclusion rule, read off the template it replaces: the manifest pins
# every published file EXCEPT .gitattributes, LICENSE and the repo logo -- none
# is engine-loaded, and setup.sh's downloader derives its metadata list from
# this manifest, so an unpinned-but-published file is simply not fetched.
# (the reference manifest's own header; setup.sh
# reference_manifest_metadata_files.)
OMIT_NAMES = {".gitattributes", "license", "license.md", "license.txt", "notice"}
OMIT_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

def omitted(path):
    base = os.path.basename(path).lower()
    return base in OMIT_NAMES or os.path.splitext(base)[1] in OMIT_EXTS

records, unhashable = [], []
for entry in tree:
    if entry.get("type") != "file":
        continue
    path = entry.get("path", "")
    if not path or omitted(path):
        continue
    lfs = entry.get("lfs") or {}
    oid = lfs.get("oid")
    if oid:
        records.append((oid, int(lfs.get("size", entry.get("size", 0))), path))
    else:
        unhashable.append((path, int(entry.get("size", 0))))

if unhashable and not hash_non_lfs:
    lines = "\n".join("    %s (%d bytes)" % (p, n) for p, n in unhashable)
    sys.exit(
        "new-track.sh: the checkpoint tree carries files with NO sha256. The tree\n"
        "endpoint publishes a sha256 only for LFS-backed entries (lfs.oid); for a\n"
        "plain git blob it publishes the git sha1, which is NOT the content digest\n"
        "the manifest pins. Refusing to write an unpinnable manifest. Files:\n"
        + lines + "\n"
        "  Re-run with --hash-non-lfs to download and hash exactly these files,\n"
        "  which is how the template manifest's non-LFS records were made."
    )

if unhashable:
    for path, _size in unhashable:
        url = "%s/%s/resolve/%s/%s" % (resolve_base, repo, rev, path)
        sys.stderr.write("new-track.sh:   hashing non-LFS %s\n" % path)
        blob = subprocess.run(
            ["curl", "-fsSL", "--retry", "3", "--retry-delay", "2", url],
            capture_output=True, check=False)
        if blob.returncode != 0:
            sys.exit("new-track.sh: could not download %s to hash it (%s)" % (path, url))
        records.append((hashlib.sha256(blob.stdout).hexdigest(), len(blob.stdout), path))

if not records:
    sys.exit("new-track.sh: the checkpoint tree yielded no pinnable file; refusing to write an empty manifest.")

records.sort(key=lambda r: r[2])
total = sum(r[1] for r in records)

with open(manifest_path, "w", encoding="utf-8") as fh:
    fh.write("# SHA256 manifest for %s.\n" % repo)
    fh.write("# Revision: %s\n" % rev)
    fh.write("#\n")
    fh.write("# The target checkpoint for track %s. Written by\n" % track)
    fh.write("# tools/new-track.sh from the published tree. LFS-backed entries use the\n")
    fh.write("# LFS object's own sha256 (lfs.oid).\n")
    fh.write("#\n")
    fh.write("# .gitattributes, LICENSE and image files are DELIBERATELY NOT pinned: none\n")
    fh.write("# is engine-loaded, and setup.sh derives what it downloads from this file.\n")
    fh.write("#\n")
    fh.write("#   MLXFAST_REFERENCE_MANIFEST_RECORDS: %d\n" % len(records))
    fh.write("#   MLXFAST_REFERENCE_MANIFEST_BYTES:   %d\n" % total)
    fh.write("#\n")
    fh.write("# Format: <sha256> <byte_count> <relative_path>\n")
    for sha, size, path in records:
        fh.write("%s %d %s\n" % (sha, size, path))

sys.stderr.write("new-track.sh: pinned %d checkpoint files (%d bytes) in %s\n"
                 % (len(records), total, manifest_path))
PYEOF

# --- 4. the benchd channel ---------------------------------------------------
# David ruling 2026-09-07: benchd is published from `main`. There is no
# per-track bench release branch any more, so a freshly stamped track resolves
# the same channel every other track does.
FETCH="tools/fetch-benchd.sh"
if [[ -f "${FETCH}" ]]; then
  python3 - <<'PYEOF' || die "rewriting tools/fetch-benchd.sh failed."
import re
path = "tools/fetch-benchd.sh"
src = open(path, encoding="utf-8").read()
new, n = re.subn(r'^BRANCH="\$\{BENCHD_BRANCH:-[^}]*\}"$',
                 'BRANCH="${BENCHD_BRANCH:-main}"', src, flags=re.M)
if n != 1:
    raise SystemExit("new-track.sh: expected exactly one BENCHD_BRANCH default in %s, found %d" % (path, n))
open(path, "w", encoding="utf-8").write(new)
PYEOF
  note "tools/fetch-benchd.sh: BENCHD_BRANCH default set to main"

  if [[ -n "${BENCH_COMMIT}" ]]; then
    if grep -q 'BENCHD_COMMIT' "${FETCH}"; then
      python3 - "${BENCH_COMMIT}" <<'PYEOF' || die "setting the BENCHD_COMMIT default failed."
import re, sys
commit = sys.argv[1]
path = "tools/fetch-benchd.sh"
src = open(path, encoding="utf-8").read()
new, n = re.subn(r'^(COMMIT=)"\$\{BENCHD_COMMIT:-[^}]*\}"$',
                 r'\1"${BENCHD_COMMIT:-%s}"' % commit, src, flags=re.M)
if n != 1:
    raise SystemExit("new-track.sh: expected exactly one BENCHD_COMMIT default, found %d" % n)
open(path, "w", encoding="utf-8").write(new)
PYEOF
      note "tools/fetch-benchd.sh: BENCHD_COMMIT default set to ${BENCH_COMMIT}"
    else
      note "--bench-commit ${BENCH_COMMIT} IGNORED: tools/fetch-benchd.sh has no BENCHD_COMMIT variable."
      note "  This template resolves the channel TIP and pins nothing by commit; a commit pin"
      note "  is not supported here yet. Nothing was changed for it."
    fi
  fi
fi

# --- 5. the identity strings -------------------------------------------------
# One pass over every tracked TEXT file. Both strings are replaced: the track id
# (runner labels, R2 prefixes, workflow comments, prose) and the contract
# fixture stem (every tool and test that opens it by name). Binary files and the
# exemptions above are skipped.
python3 - <<PYEOF || die "the identity replacement pass failed."
import os, subprocess, sys

old_track, new_track = "${OLD_TRACK_ID}", "${TRACK_ID}"
old_stem, new_stem = "${OLD_CONTRACT_STEM}", "${NEW_CONTRACT_STEM}"
exempt = set("""${EXEMPT[@]}""".split())

tracked = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True)
changed = []
for path in tracked.stdout.decode().split("\0"):
    if not path or path in exempt or not os.path.isfile(path) or os.path.islink(path):
        continue
    with open(path, "rb") as fh:
        raw = fh.read()
    if b"\0" in raw:
        continue
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        continue
    out = text.replace(old_track, new_track).replace(old_stem, new_stem)
    if out != text:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
        changed.append(path)

sys.stderr.write("new-track.sh: rewrote the track identity in %d tracked files\n" % len(changed))
for path in sorted(changed):
    sys.stderr.write("new-track.sh:   %s\n" % path)
PYEOF

# --- 6. the ranked runner label ----------------------------------------------
# The track label was replaced by the pass above; the OS token is this step.
WORKFLOW=".github/workflows/benchmark.yml"
if [[ -f "${WORKFLOW}" ]]; then
  python3 - "${OS_TOKEN}" "${TRACK_ID}" <<'PYEOF' || die "rewriting the runs-on label failed."
import re, sys
os_token, track = sys.argv[1], sys.argv[2]
path = ".github/workflows/benchmark.yml"
src = open(path, encoding="utf-8").read()
# One rule covers both spellings: the live `runs-on:` list and the same list
# quoted in the header comment. The track label was already replaced by the
# identity pass; only the OS token moves here.
new, n = re.subn(r"(\[\s*self-hosted\s*,\s*)(?:macOS|Linux)(\s*,)",
                 lambda m: m.group(1) + os_token + m.group(2), src)
if n == 0:
    raise SystemExit("new-track.sh: no [self-hosted, <os>, ...] label found in %s" % path)
open(path, "w", encoding="utf-8").write(new)
sys.stderr.write("new-track.sh: runs-on -> [self-hosted, %s, %s] (%d occurrence(s))\n"
                 % (os_token, track, n))
PYEOF
fi

# --- 7. the goldens ----------------------------------------------------------
# A track's hidden goldens are recorded on ITS OWN BOX, published to R2 under
# correctness_prompts/<track id>/ and staged on the ranked box as
# MLXFAST_QWEN38_GOLDEN_DIR. They are never in git, so there is nothing to
# rename. The source track's PUBLIC captures do ship in git, at the same path,
# and they are the source track's: the new track records its own on its own
# box. Drop them, so the stamped tree carries no other track's golden.
if [[ -d "correctness_prompts/${OLD_TRACK_ID}" ]]; then
  rm -rf "correctness_prompts/${OLD_TRACK_ID}"
  note "removed correctness_prompts/${OLD_TRACK_ID}/ (the source track's public captures); record this track's own on its box and commit them at correctness_prompts/${TRACK_ID}/"
fi

# --- 8. the engine fork pin --------------------------------------------------
# NOTHING TO STAMP HERE. Vendor/mlx-swift-lm is a VENDORED TREE of plain files,
# not a gitlink, so the pin is the CONTRACT FIXTURE's mlx_swift_lm_revision
# (section 2 already wrote ${FORK_SHA} there) plus the bytes in the tree itself.
# The tree is edited in the engine repository; nothing goes upstream. Cutting
# a fresh tree from a new upstream base is a separate, deliberate step that
# replaces the whole directory and is reviewed on its own.
# docs/new-track-repo-procedure.md section 6 holds the command.
note "Vendor/mlx-swift-lm is a vendored tree edited in this repository; mlx_swift_lm_revision is ${FORK_SHA} (the upstream base). A fresh cut is a separate step (docs/new-track-repo-procedure.md section 6)."

# --- 9. report and verify ----------------------------------------------------
echo
echo "${SCRIPT_NAME}: files changed"
git status --porcelain | sed 's/^/  /'
echo

cat <<WARNEOF
${SCRIPT_NAME}: RE-AUTHOR THE MODEL FACTS BEFORE ANY MEASUREMENT.
  ${NEW_CONTRACT} target.* -- layer counts, attention geometry, head dims,
  expert counts, MoE widths, n-gram shape, token ids, tensor counts -- and the
  fixtures/ config, tensor-inventory and norm-convention files it names were
  COPIED FROM THE TEMPLATE. They describe the SOURCE model. They are wrong for
  any different model family, and nothing here can derive them.
  benchmark.json description, track_name and the README prose are the template's
  too. Rewrite them for this track.
WARNEOF
echo

# --gitlink-targets report matches what CI runs. This repository has no
# submodule, so nothing is downgraded; the flag is passed for parity with CI.
python3 tools/lint-benchmark-manifest.py --manifest benchmark.json --gitlink-targets report \
  || die "the stamped benchmark.json does NOT pass tools/lint-benchmark-manifest.py (above). The tree is left as stamped for inspection; nothing was committed."

echo
echo "${SCRIPT_NAME}: stamped ${TRACK_ID}. Nothing was committed -- review the diff and commit."
