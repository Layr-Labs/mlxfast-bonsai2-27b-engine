# CI coverage: what the hosted pipeline gates, and what it cannot

`.github/workflows/ci.yml` is the only automated gate in this repository. This
document says what that pipeline proves. More importantly, it says what the
pipeline does not prove, because this package's most interesting tests need
hardware that GitHub does not rent.

This document exists to enforce one rule: **a check that CI cannot run is
named here, never silently skipped.** `tools/ci-box-only-inventory.sh --check`
runs in CI. It fails the build if the inventory below drifts from the test
sources.

## Why the split exists

The engine targets Apple Silicon through MLX. Two resources decide whether a
test can run on a hosted runner:

- **A Metal GPU with the kernels built for it.** GitHub-hosted macOS runners
  are virtualised. `./setup.sh` never runs there, so nothing ever builds
  `mlx.metallib` (`tools/build-mlx-metallib.sh` needs the Metal toolchain
  component). Nothing in this pipeline relies on MLX executing on a GPU, and
  no timing claim may ever be made from it.
- **The reference checkpoint.** The pinned target is about 17.3 GiB across 4
  safetensors shards, and it carries the MTP head inside it. It is not in the
  repository. This pipeline deliberately does not perform that large network
  fetch.

Everything that needs neither resource compiles and runs on hosted hardware,
and that is what CI gates. This covers protocol encoding and decoding, config
and artifact contracts, safetensors header parsing, scoring arithmetic, chat
templating, wire fixtures, and request validation.

## CI-covered

| Job | Runner | Check |
| --- | --- | --- |
| `lint` | `ubuntu-latest` | `tools/ci-workflow-egress-scan.sh` — nothing in `.github/` matches a credential-or-egress tripwire pattern (see [Deliberate non-goals](#deliberate-non-goals) for the exact set) |
| `lint` | `ubuntu-latest` | `actionlint` over `.github/workflows/`, with the ranked runner's label declared in `.github/actionlint.yaml` |
| `lint` | `ubuntu-latest` | shell syntax (`bash -n` / `sh -n`) over every tracked `*.sh` |
| `lint` | `ubuntu-latest` | `tools/lint-benchmark-manifest.py --gitlink-targets report` — the Yukon track manifest at rest |
| `lint` | `ubuntu-latest` | `tools/ci-box-only-inventory.sh --check` — the inventory below matches the sources |
| `lint` | `ubuntu-latest` | `tools/test-benchmark-facade-env.sh` — the real `tools/benchmark.sh`, driven with a stub `benchd`, exports `MLXFAST_QWEN_MTP_TRACK_ID` equal to `benchmark.json`'s `trackId` in every mode, and refuses when the manifest carries none |
| `lint` | `ubuntu-latest` | `tools/test-new-track.sh` — the real `tools/new-track.sh` stamps a seeded copy of this tree into a new track against a stub Hugging Face tree API on `127.0.0.1`: the manifest, the contract fixture, the checkpoint file list, the runner label, the fork revision and the benchd channel all name the new track, the result still passes the manifest linter, and a dirty tree, a malformed track id and a checkpoint file with no published sha256 are refused |
| `swift` | `macos-26` | toolchain preflight: a full Xcode is selected, Swift >= 6.3 |
| `swift` | `macos-26` | `tools/test-submission-security.sh` — the submission-restriction HARD GATE, every hostile-archive assertion in the suite (the declared floor is 236). It sits in the `swift` job because it compiles the real byte-budget enforcer with `swiftc`. Nothing in it is box-dependent, so no assertion is gated off. See [What "hard gate" does and does not mean](#what-hard-gate-does-and-does-not-mean) |
| `swift` | `macos-26` | the suite's **non-vacuity floor**, restated workflow-side against its printed trailer — the run reds if the suite stopped short, reported any failure, or ran fewer assertions than `CI_MIN_ASSERTIONS`, and reds again if that number has drifted from the suite's own `EXPECTED_MIN_ASSERTIONS` |
| `swift` | `macos-26` | `swift build --build-tests --force-resolved-versions` — `MLXFastCore`/`Transform`/`Model`/`Harness`, both executables, and the test bundle |
| `swift` | `macos-26` | `tools/ci-swift-warning-gate.sh` — a warning in `Sources/` or `Tests/` fails the run |
| `swift` | `macos-26` | `git diff --exit-code -- Package.swift Package.resolved` — the frozen dependency graph was not re-resolved |
| `swift` | `macos-26` | `swift test --force-resolved-versions --no-parallel` with `MLXFAST_RUN_MLX_RUNTIME_TESTS=0` (serial on purpose: parallel `swift test` wedges on this runner image -- see the step comment in `ci.yml`) |

### What "hard gate" does and does not mean

**This pipeline is ADVISORY, and that is the accepted posture** (ruled by David
2026-08-20: the two-session red-team is the de facto merge gate). CI runs
`tools/test-submission-security.sh` on every pull request and every push to
`main`, and reds the run on a regression. That is the whole of it. **No status
check is required anywhere in this repository, and a red run blocks nothing
mechanically** — not a merge, not a dispatch:

- **there are no required status checks.** Branch protection is unavailable on
  this repository's plan — the REST endpoint answers `403 Upgrade to GitHub Pro
  or make this repository public` — so no required-status-check rule exists, or
  can be configured, for any job in this file;
- there is no `CODEOWNERS` in this tree, so nothing mechanically requires a
  review either;
- no ranked workflow invokes the suite: `.github/workflows/benchmark.yml`
  measures and scores, and
  `.github/workflows/bonsai2-27b-ranked-benchmark.yml` is a fail-closed
  stub;
- a **ranked** `workflow_dispatch` does not consult `ci.yml`, so dispatching a
  benchmark does not consult the gate. (This file carries its own
  `workflow_dispatch` trigger, for re-running CI by hand — a different thing.)

None of that is a defect to be worked around. What holds the line is a person
who reads the run, plus the independent red-team passes that gate a merge.
Dispatch-time gating is separate work that lands with the ranked runner: see
`docs/submission-restriction-spec.md` §9 (not ported) and §10 (the hard gate).

Do not describe any check in this file as "required". Do not add prose implying
that it blocks a merge or a dispatch. If anyone ever configures required
checks, change this section first.

### On the non-vacuity floor

The suite's exit status is not the whole gate. A section can stop running: it
can be deleted, renamed out of the flow, or skipped by an early `continue`.
Such a section leaves every remaining assertion green. The suite also runs
`set -uo pipefail` *without* `-e`. The suite carries `EXPECTED_MIN_ASSERTIONS` for exactly this. That check lives
in the suite's own trailer, so truncating the file removes the floor along with
the assertions it guarded.

So the workflow also states the floor from outside, against the trailer the
suite prints. `tools/ci-swift-warning-gate.sh` uses the same separation to
guard the build from a different file than the build.
`tools/ci-workflow-egress-scan.sh` uses it to keep its patterns outside the
directory it scans. The workflow's `CI_MIN_ASSERTIONS` and the suite's
`EXPECTED_MIN_ASSERTIONS` are cross-checked against each other, so they cannot
drift apart in silence. Raise both in the commit that adds assertions.

### On warnings-as-errors

The gate is a log scan (`tools/ci-swift-warning-gate.sh`), not
`-Xswiftc -warnings-as-errors`, for two reasons. That flag is global, and this
package's graph is mostly vendored third-party code whose warnings belong to
upstream. The per-target alternative, `swiftSettings` in `Package.swift`, is
unavailable because `Package.swift` sits inside the frozen trusted-harness
source scope. So the gate binds where we own the code — `Sources/` and
`Tests/` — and reports vendored warnings as a count.

Widening the gate's exclusion list to make a first-party warning pass is not an
accepted fix. Fix the cause.

## Box-only: real gates that CI does NOT run

### Test files behind an opt-in environment gate

A gated test runs on an Apple Silicon box. Two resources decide which box:
some suites need the checkpoint staged (`./setup.sh`), and some need only the
built MLX runtime (`tools/build-mlx-metallib.sh`). Hosted runners can supply
neither. An MLX evaluation without a published metallib ABORTS the process
rather than failing the test, which is why the runtime class is gated at all.
The gate variable is one switch for both classes; set it to `1`:

    MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test --force-resolved-versions

THE INVENTORY IS EMPTY, and that is the correct state after the Darkbloom
CBv2 re-base. Every GPU-resident and checkpoint-resident test in this tree
tested the private engine. That engine is deleted, and the tests went with it.
The model, the runner and their GPU tests are in the vendored
`Vendor/mlx-swift-lm` tree, and the fork's own suite runs them. This repository
keeps only tests
that need no GPU and no checkpoint.

The inventory below is generated. Regenerate it with
`tools/ci-box-only-inventory.sh`. Its shape is `<gate> <file> <gate-expressions>` (counting guard
expressions, not comment mentions):

<!-- BEGIN box-only-inventory (generated by tools/ci-box-only-inventory.sh) -->
```
```
<!-- END box-only-inventory -->

`MLXFAST_RUN_QWEN_REFERENCE_PARITY` additionally needs
`MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH` pointed at a transformed weights tree. It
is checkpoint-resident, not just GPU-resident. It currently gates no test in
this tree. The `GATES` array keeps the name so that a returning parity test is
inventoried rather than running unseen.

`MLXFAST_RUN_GEMMA4_UPSTREAM_EQUIVALENCE` is the third name the `GATES` array
keeps. It gates no test in this tree either. The array holds all three names so
that a returning gated test is inventoried rather than running unseen.

### Other gates CI does not attempt

| Not run in CI | Why | Where it runs |
| --- | --- | --- |
| ~~The two `benchd/scripts/benchmark.sh` command targets in `benchmark.json`~~ | **No longer applicable.** benchd stopped being a source dependency; every `benchmark.json` command target now lives in this repository (`./setup.sh`, `./tools/fetch-benchd.sh`, `./tools/bonsai2-27b-measure-and-score.sh`), so CI verifies all three and the linter downgrades nothing | CI, fully verified |
| `tools/build-mlx-metallib.sh` (the AOT `mlx.metallib`) | needs the Metal toolchain component and produces a GPU artifact CI cannot exercise | box, via `./setup.sh` |
| `swift build -c release` into the two production scratch roots (`.build`, `.build-worker`) | the scored binary layout; CI proves the code compiles, not that the release layout is staged | box, before a benchd pass |
| Correctness gates, token fidelity, the timed paired measurement, scoring | measurement authority is benchd's, on the ranked M5 box, behind a thermal gate | the ranked pipeline; nothing in this repository times or scores anything |
| Hidden-golden fidelity | hidden material, R2-provisioned, sha256+bytes pinned | ranked box only |

## Deliberate non-goals

- **No secrets.** Not a token, not a deploy key, not an R2 credential.
- **No artifact upload from CI.** This repository can receive organizer-material
  fixtures, and the runner must not be a way out for them. The one exception is
  the ranked pipeline's score file, which is the product Yukon reads back:
  `.github/workflows/benchmark.yml` uploads `score.json` as
  `benchmark-results-<run_id>`. The scan checks that upload rather than
  exempting it. **Exactly one** upload reference may exist in that file. That
  one step's own block must carry a 40-hex-pinned action, `path: score.json`,
  and `if-no-files-found: error`. A second upload step fails the scan however
  well-formed it is. The three property checks are evaluated inside the checked
  step, so another step cannot vouch for it.
- **Both are tripwired, not proven.** `tools/ci-workflow-egress-scan.sh` scans
  all of `.github/` — workflows and any composite action beside them — and
  fails the run on a named set of patterns: a `secrets` reference
  (`secrets.NAME`, `secrets['NAME']`, `toJSON(secrets)`), `secrets: inherit`,
  any `write` permission (blanket or single-scope, `id-token: write`
  included), and an artifact upload by action or by expression anywhere other
  than the checked ranked-score upload above. That set is
  the whole of what it detects. It cannot see a `run:` step that curls, a
  third-party action that uploads on its own, or an `actions/cache` entry used
  as a side channel. Those remain review's job. The scan is the floor that
  stops the common accidental regressions landing silently. It is not a proof
  that nothing leaves the runner.
- **No ranked measurement.** The ranked pipeline is
  `.github/workflows/benchmark.yml`. It is `workflow_dispatch` only and
  self-hosted, and CI never dispatches it.
  `.github/workflows/bonsai2-27b-ranked-benchmark.yml` is a fail-closed
  stub and stays one: a `workflow_dispatch` of it exits 1 by design.
- **No submodule.** This repository has none. Both vendored trees are plain
  files that an ordinary checkout carries, and benchd is a fetched binary. See
  the table above.
