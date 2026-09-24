# New-track repo & branch procedure

How a new model track is stood up across the engine and bench repos. This is the
procedure this repo itself was created by; it applies to every future track family.
Ruled 2026-08-22 (engine fork = new repo). Ruled 2026-09-07 (benchd is published from `main`).

## 1. Engine fork = a NEW repo per track family

- Create a fresh org repo with a **decoder-neutral** name: `mlxfast-{model}{ver}-{params}-engine`
  (no spec-decoder kind — mtp/dflash/dspark — in repo, track, or branch names; model-facts inside
  code/config are fine).
- **Fresh-seed, do not fork-push.** The org ruleset requires verified commit signatures and the
  source engine's early history contains unsigned commits, so a history-carrying push is rejected.
  Seed one signed commit whose tree is identical to the source engine's `main` tip, and name the
  source commit in the seed message. History stays in the source repo.
  - Precedent: this repo's `main` root `30e5104` = tree-identical seed of
    `mlxfast-qwen38-125b-a6b-engine-dev @ 0dd0a2a3`.
- The repo is created **private**; the visibility flip to internal is org-owner-gated.

## 2. Bench side = the `main` branch

benchd is published from `main`. There is no bench release branch for a track
(ruled 2026-09-07). `main` is the channel every track resolves, and the
channel's own `benchd.manifest.json` names the bytes. The track id still names
the leaderboard namespace, the runner label and the R2 key prefix. It no longer
names a bench branch.

## 3. Engine ↔ bench binding: the resolved binary

- benchd is a **prebuilt binary**, not a source dependency. The engine carries no
  submodule and no gitlink for it.
- `tools/fetch-benchd.sh` reads `dist/benchd.manifest.json` at the tip of the channel
  branch, verifies the binary against the `{sha256, bytes}` that manifest names, and
  installs both into `benchd-bin/`. The manifest is the pin.
- Verify what a box resolved by reading the identity line the script prints, and the
  manifest it installed beside the binary. Do not quote a `{source_commit, sha256,
  bytes}` triple in a document: it goes stale at the next republish.
- `.github/scripts/overlay-editable-paths.sh` refuses an editable entry at `benchd-bin`,
  or at the retired `benchd` and `benchd.pin` spellings, so a submission can never
  reach its own scorer.

## 4. Re-baseline discipline

The channel tip moves when the organizer republishes dist. Before any new measurement
logic lands:

1. Resolve the current channel pair and read the identity the script prints.
2. Only then build track measurement features on top of what that benchd supports.

## 5. Goldens for the new track

- Authored **on the track's designated benchmark hardware**, never a development laptop
  (greedy-decode argmax near-ties differ across silicon).
- **A≡B double-generated** — two independent generations, byte-identical asserted **before**
  pinning. Non-identical = STOP; it is also the determinism tripwire for the track's pinned
  runtime configuration.
- Identity = **sha256 + bytes**, never name/path/location. The gates-bound golden pin is the
  oracle-carrying file's hash; the oracle is mandatory on the timed path.
- Uploaded to R2 under `correctness_prompts/{track_id}/`, append-only, per-instance
  authorization, operator-workstation credentials only, GET + sha + bytes round-trip
  verified after upload. The bucket is part of the endpoint, never part of the object
  key. `tools/fetch-goldens.sh` documents the convention and is the reader.
- Upload happens **once the track is stable and ready for testing** — after the engine port and
  measurement stack are proven on-box, before the first scored window.
  `official_scoring_enabled` flips true LAST, in its own PR, after one clean scored window.

## 6. Stamping with tools/new-track.sh

The seed from section 1 still carries the SOURCE track's identity in every file.
`tools/new-track.sh` replaces that identity in one pass. Run it from the root of
the fresh seed, on a clean worktree:

```sh
tools/new-track.sh --track-id <{model}{ver}-{params}-{platform}-v{N}> \
                   --fork-sha <40 hex> \
                   --checkpoint <hf_repo>@<40 hex revision> \
                   [--bench-commit <40 hex>] \
                   [--os macOS|Linux]
```

What it changes:

- `benchmark.json`: `name`, `trackId`, `staticReviewTrackId`,
  `leaderboard.namespace` and `contractPath`. The commands, the editable paths,
  the byte budget and the scoring constants stay as they are.
- The contract fixture: a copy of the current one under the new name, with the
  new track id, the fork revision, `official_scoring_enabled: false`, an empty
  timed pool, an empty live golden, and the pending-organizer sentinel. The old
  fixture is deleted.
- The checkpoint file list: rewritten from the Hugging Face tree of
  `--checkpoint`. LFS entries use the LFS object's own sha256. A file the tree
  publishes no sha256 for is refused by name.
- `tools/fetch-benchd.sh`: `BENCHD_BRANCH` defaults to `main`.
- `.github/workflows/benchmark.yml`: the `runs-on` list becomes
  `[self-hosted, <os>, <track id>]`. The default OS is macOS for `mlx` and Linux
  for `cuda`.
- The contract fixture's `mlx_swift_lm_revision`: the upstream commit the
  vendored tree was cut from. `Vendor/mlx-swift-lm` is a vendored tree of plain
  files, not a submodule, so the stamp writes no gitlink and touches no bytes
  in the tree. Cutting a fresh tree is a separate step; see below.
- Every other tracked text file: the old track id and the old fixture name become
  the new ones. The source track's port notes, the manifest linter, this procedure
  and the two new-track scripts keep the old names, because they record the source
  track. `tools/new-track.sh` holds that exemption list, and
  `tools/test-new-track.sh` holds the same list and proves it.
- The goldens: nothing to do. A track's goldens are recorded on its own box,
  published to R2 under `correctness_prompts/<track id>/` and staged on the
  ranked box as `MLXFAST_QWEN38_GOLDEN_DIR`. They are never in git, so the new
  track carries none of the source track's and there is no directory to rename.

The script never commits. Review the diff, then commit.

### `Vendor/mlx-swift-lm` is edited here

`Vendor/mlx-swift-lm` is a COPY of `Layr-Labs/mlx-swift-lm`, cut once from an
upstream commit. It is a copy and not a submodule because an editable path
names bytes in this tree, and a gitlink names a commit: the track's Nemotron
3.5 Lightning model files are an editable surface. `Vendor/mlx-swift` is a copy
for the same reason.

After the cut, this repository is the source of truth for the tree. Merges,
runners and fixes are commits here. Nothing is pushed to the fork and no fork
PR is opened for track work (David ruling 2026-09-16). The fixture's
`mlx_swift_lm_revision` names the upstream commit the tree was cut from; the
engine's own history holds everything on top of it.

Cut a fresh tree only when the whole tree moves to a new upstream base. That is
ONE command, from the repository root, with `$FORK` set to a clone of the fork
and `$SHA` set to the upstream commit; it discards the engine-local changes,
which must then be re-applied from this repository's history:

```sh
rm -rf Vendor/mlx-swift-lm && mkdir -p Vendor/mlx-swift-lm \
  && git -C "$FORK" archive "$SHA" | tar -x -C Vendor/mlx-swift-lm \
  && git add -A -f Vendor/mlx-swift-lm
```

The `-f` on `git add` is load bearing. The fork's own `.gitignore` carries an
`mlx-swift-lm/` rule, which would otherwise drop the files under
`skills/mlx-swift-lm/` that the fork itself tracks.

Then prove the copy is exact. The two hashes must be equal:

```sh
git rev-parse "$(git write-tree):Vendor/mlx-swift-lm"
git -C "$FORK" rev-parse "$SHA^{tree}"
```

A difference means the copy dropped or changed a file. Do not commit it. The
check holds only for a fresh cut; once engine-local commits sit on top, the
tree is not expected to equal any upstream commit.

Last, set the contract fixture's `mlx_swift_lm_revision` to `$SHA` in full
40-character form, and update the `Vendor/mlx-swift-lm` row of
`THIRD_PARTY_NOTICES.md` and the dependency comment in `Package.swift`.

The model facts do not change. The layer counts, the attention geometry and the
expert counts in the new fixture are the source model's. The script prints this
at the end. Re-author them before any measurement.

`tools/test-new-track.sh` proves all of the above offline, in CI.
