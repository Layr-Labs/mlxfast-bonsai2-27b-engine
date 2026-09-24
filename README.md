# mlxfast — Ternary Bonsai 2 27B MLX

This repository is the engine for the Ternary Bonsai 2 27B MLX
speedup benchmark. The track identifier is `bonsai2-27b-mlx-v1`.

## What this repository is

This repository holds the engine. The engine is the Swift and Metal code that
runs the target model on Apple Silicon. You optimize the engine. You make the
model do the same work in less time.

The benchmarker measures the engine. The benchmarker is a separate program
called `benchd`. It arrives as a verified prebuilt binary. It owns all
timing, all scoring, and all gates. Nothing in this repository measures or
scores anything.

The ranked run runs ONE STREAM AT A TIME. The ranked run measures your engine
and a serial control engine in the same session, on the same box. The score
compares the two. The control engine is the organizer's reference tree, and its
cost is measured in that same session. No file stores it.

> **NOTE — the track goldens are not in this repository.**
> A track's timed tapes and per-depth oracles are organizer material. They are
> published in R2 at the `r2_path` keys the contract pins. The ranked box
> stages them out of band into the directory its runner service exports as
> `MLXFAST_QWEN38_GOLDEN_DIR`. That environment name keeps its Qwen spelling
> because it is the fleet's contract with the runner service, not this track's
> own name. `tools/ranked-box-preflight.sh` verifies every file there against
> the contract's `{sha256, bytes}` and refuses an extra `*.json`. They are
> never in git, so your clone does not carry them.
>
> This track has no goldens yet. Section [Current status](#current-status)
> states what that blocks.
>
> The organizer stages them with the signer this repository vendors:
>
> ```bash
> R2_BUCKET_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
>   tools/fetch-goldens.sh --all --out "$MLXFAST_QWEN38_GOLDEN_DIR"
> tools/ranked-box-preflight.sh
> ```

> Official scoring is NOT armed. Section [Current status](#current-status)
> states each fact. `docs/bonsai2-27b-port-notes.md` is the engineering
> record.

### Lineage

This repository descends from the Qwen 3.8 125B A6B MLX engine, which descends
from `Layr-Labs/mlxfast-challenge-dev`. Those repositories rank different
models under different rules. Only this track's rules apply here.

The `.gemma4` transform family in `Sources/MLXFastTransform/` is not a
leftover. It is the synthetic-fixture substrate the transform pipeline tests
run on. Those tests exercise the transform, not a Gemma model.

## Requirements

- An Apple Silicon Mac.
- Enough unified memory for the target model and its working set. The target
  pack is 8,608,721,032 bytes across 19 pinned files, of which one is the
  single safetensors shard. The MTP head adds 238,942,843 bytes across 4 more.
  The DFlash 2 drafter adds 3,848,824,532 bytes across 3 more. A run loads the
  target and ONE drafter. Add the KV cache, the recurrent state and the decode
  buffers on top of that.
- At least 22 GiB of free disk space before the download starts. Change this
  limit with `MLXFAST_REFERENCE_MIN_FREE_GIB`. The DFlash 2 drafter asks for
  8 GiB of free space of its own. Change that limit with
  `MLXFAST_DFLASH_DRAFTER_MIN_FREE_GIB`.
- macOS 14 or later. `Package.swift` sets that platform floor. CI builds on a
  `macos-26` runner, and the Metal toolchain policy in `setup.sh` treats
  macOS 26 as its own case.
- Swift 6, through Xcode or through the Xcode Command Line Tools.
  `Package.swift` declares `swift-tools-version: 6.3`.
- The Xcode Metal Toolchain. `./setup.sh` tries to download it. The Command
  Line Tools alone carry no `metal` compiler, so a box that builds the metallib
  needs full Xcode. Install it, open it once, and accept the license with
  `sudo xcodebuild -license accept`.
- CMake. `./setup.sh` installs it through Homebrew when it is missing.
- Git.

You do not need Rust. The benchmarker arrives as a prebuilt binary.
`./tools/fetch-benchd.sh` resolves it from the bench repository's `main` dist
channel and verifies it against that channel's `benchd.manifest.json`. The
channel is public. No token is needed.

## Quickstart

Run these commands in order. One sentence describes each command.

```bash
git clone <repository-url> mlxfast-bonsai2-27b-engine
```

This command copies the repository to your machine.

```bash
cd mlxfast-bonsai2-27b-engine
```

This command makes the repository your working directory.

```bash
./tools/fetch-benchd.sh
```

This command resolves the benchmarker binary from the dist channel into
`benchd-bin/` and verifies its sha256 and its byte count against the channel's
`benchd.manifest.json` (installed beside the binary).

```bash
./setup.sh
```

This command checks your toolchain, builds the two Swift binaries, builds
`mlx.metallib`, and downloads and verifies the target model.

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/Ternary-Bonsai-2-27B-mlx-2bit \
  --output weights
```

This command converts the downloaded checkpoint into the `weights/` tree that
the engine loads.

```bash
R2_BUCKET_ENDPOINT=... tools/fetch-goldens.sh --public
```

This command fetches the two public captures from R2 into
`correctness_prompts/bonsai2-27b-mlx-v1/`. Get the value of
`R2_BUCKET_ENDPOINT` from the organizer. See [The public
captures](#the-public-captures).

```bash
MLXFAST_ENGINE_BIN=.build/release/bench-worker \
MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/bonsai2-27b-mlx-v1/public-local-iterate.golden.json \
  ./benchmark.sh --local-iterate
```

This command runs the local test against the public local-iterate capture.

Both speculative decoders are separate artifacts. `./setup.sh` stages them for
you: it runs `./setup-mtp-head.sh` for the MTP head, then
`./setup-dflash-drafter.sh` for the DFlash 2 drafter. See [The speculative
decoders are separate
exports](#the-speculative-decoders-are-separate-exports).

> **WARNING — set `MLXFAST_CORRECTNESS_GOLDEN_PATH` yourself.**
> The local test has no default golden. It stops with an error when the
> variable is empty.

> **WARNING — do not pass `--golden`, `--weights`, or `--score-path` to
> `./benchmark.sh`.**
> The script rejects these flags. Use the environment variables instead.
> `MLXFAST_WEIGHTS_PATH` defaults to `weights`.

### The public captures

The local test checks correctness against a public capture. The two public
captures are R2 objects. They are not in this repository.
`fixtures/bonsai2_27b_mlx_v1_track.json` pins each one in `public_captures`,
with its R2 key (`r2_path`), its sha256 and its byte count.

| Capture | Local path after the fetch | Use |
|---|---|---|
| `local_iterate` | `correctness_prompts/bonsai2-27b-mlx-v1/public-local-iterate.golden.json` | `--local-iterate`. The drift tripwire. |
| `local_submit` | `correctness_prompts/bonsai2-27b-mlx-v1/public-local-submit.golden.json` | `--local-submit`. The long capture. |

`tools/fetch-goldens.sh --public` reads the pins, fetches each capture to its
`r2_path` under the repository root, and verifies the byte count and then the
sha256. It deletes a file that does not match. It keeps a file that already
matches. Git ignores `correctness_prompts/`, so a fetched capture never goes
into a commit.

The fetch needs `R2_BUCKET_ENDPOINT`. Get the value from the organizer, and
keep it in your `.env`. Do not put it in a file that you commit.

> **WARNING — the public captures are not recorded yet.**
> The fixture pins both captures with the pending sentinel
> `BONSAI2-27B-MLX-V1-PENDING-ORGANIZER`. `tools/fetch-goldens.sh --public`
> refuses until the organizer records the captures on the track's box and pins
> them. Until then, the local test cannot check correctness.

> **NOTE — a golden must name the checkpoint it came from.**
> Each `.json` golden must carry a `model_provenance` block. The block names
> the repository and the revision of the pinned model. The loader refuses a
> golden that carries no block. The error message names `model_provenance`.
> The loader does not skip the check when the block is absent.

## Repository structure

| Path | What it holds | Status |
|---|---|---|
| `Sources/MLXFastTransform/` | The offline transform that writes `weights/`. | Editable |
| `Sources/MLXFastCLI/` | The trusted CLI, `mlxfast-swift`. | Trusted |
| `Sources/MLXFastCore/` | Shared constants and contracts. | Trusted |
| `Sources/MLXFastTrustedHarness/` | The editable-surface budget, the head declaration reader, transform verification, and the metallib fingerprint. | Trusted |
| `Vendor/mlx-swift/` | The pinned MLX fork, a vendored tree. The listed Metal kernel sources are editable. | Mixed |
| `Vendor/mlx-swift-lm/` | The engine fork, a vendored tree. It holds the model, the runner, and `bench-worker`. The Bonsai 2 and Qwen 3.5 model files are editable. | Mixed |
| `fixtures/` | The track contract and the pinned checkpoint manifests. | Trusted |
| `tools/` | Setup, build, lint, and measurement scripts. | Trusted |
| `benchd-bin/` | Where `./tools/fetch-benchd.sh` installs the verified binary. Git ignores it. | Fetched |
| `mtp-head.manifest.json` | The speculative-decoder declaration. It names the decoder and the draft depth. It declares; it carries no weights. | Editable, optional |
| `correctness_prompts/` | Not in git. `tools/fetch-goldens.sh --public` writes the public captures here. The goldens live in R2 and on the ranked box. | Fetched |
| `weights/` | The transformed weights the engine loads. | Generated |
| `benchmark.json` | The Yukon track manifest. It lists every editable path. | Trusted |

### Starting a new track from this repository

This repository is the template for the next track. Seed a new repository with a
copy of this tree, then run `tools/new-track.sh` in it. The script stamps the new
identity into the manifest, the contract fixture, the checkpoint file list, the
runner label, the engine pin and the docs. It touches no golden: a track's
goldens live in R2 and on its own box, never in git. It never commits.
`docs/new-track-repo-procedure.md` holds the full procedure and the usage line.

### The engine is a vendored tree

`Vendor/mlx-swift-lm` is a copy of `Layr-Labs/mlx-swift-lm`, as plain files.
That fork holds the three parts of the engine:

* the model, including the packed Bonsai 2 tower, the MTP assistant that binds
  to it and the DFlash 2 block drafter;
* the runner, which loads the checkpoint and builds the engine and the one-row
  stepper;
* `bench-worker`, the Engine Protocol v1 server. The benchmarker starts this
  binary. One binary serves every model family, and it selects the runner from
  the checkpoint's own `model_type`.

A plain `git clone` carries the whole engine. This repository has no submodule.

The tree was cut from `Layr-Labs/mlx-swift-lm` `feat/qwen38-flash-next-runner`
at `859f4d9`, with fork `main` `0607011` merged in. The merge, two build fixes
and the Bonsai 2 support files taken from fork `main` `fd0eaac` (pull requests
154 and 155) are commits of THIS repository. Nothing goes upstream. Edit the
tree in place, here. The contract fixture's `mlx_swift_lm_revision` records the
upstream base only.

The fork is a copy and not a submodule for one reason: its Bonsai 2 and Qwen
3.5 model files are an editable surface. An editable path names bytes in this
tree, and a gitlink names a commit.

`Vendor/mlx-swift` is a vendored tree for the same reason: its Metal kernel
sources are editable. The fork asks for `mlx-swift` from the network. SwiftPM
uses this package's local `Vendor/mlx-swift` instead, because a local path
dependency of the root package wins. SwiftPM prints a "conflicting identity"
warning when it does this. The warning is expected.

The copy is `Layr-Labs/mlx-swift` at `6d6796d7`. It holds the two nested
submodules of that repository as plain files: `Source/Cmlx/mlx` at `3fa8f25e`
and `Source/Cmlx/mlx-c` at `02cf6f4d`. To move the pin, copy the whole tree
again from a fresh checkout of the new commit, with its submodules, then
rebuild the metallib.

### The speculative decoders are separate exports

This track has two declarable speculative decoders. The pack declares
`mtp_num_hidden_layers: 0` and carries no drafter of any kind, so each decoder
is its own published export, used unmodified. A submission names the one it
arms in `mtp-head.manifest.json`.

The MTP head:

| Property | Value |
|---|---|
| Repository | `EigenLabs/Qwen3.8-27B-MTP-4bit` @ `329261c5e0b3f9c233485e682cb3b67b88c20a55` |
| Model type | `qwen3_5_mtp` |
| Tensors | 31, at the top level. There is no tensor prefix. |
| Hidden layers | 1, full attention |
| Quantization | Affine, group size 64, 4 bits |
| Embeddings | None of its own. It rides the target's embeddings and output head. |
| Permitted draft depths | 1 to 7 |
| Maximum speculative batch | 1. The arm is single-stream. |

The DFlash 2 drafter:

| Property | Value |
|---|---|
| Repository | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Architecture | `DFlash2DraftModel`, over a `qwen3` configuration |
| Tensors | 81, at the top level, all BF16, in one shard. 3,848,808,960 tensor bytes. |
| Hidden layers | 5, sliding attention, sliding window 2048 |
| Attention | 32 query heads, 8 key/value heads, head dimension 128 |
| Shape | Hidden 5120, intermediate 17408, vocabulary 248320, rope base 1e7 |
| Embeddings | None of its own. It rides the target's embeddings and output head. |
| Permitted draft depths | 1 to 16 |
| Maximum speculative batch | 1. The arm is single-stream. |
| License | Apache-2.0 |

A DFlash 2 depth is the drafter's BLOCK SIZE minus one. The drafter proposes a
whole block in one forward pass. The block is the last committed token followed
by that many mask tokens. The drafter was trained at block 8, which is depth 7.
A larger block is permitted, and it simply accepts less.

The two depth ranges are independent. Neither decoder can borrow the other's
range.

No submission carries drafter weights. `./setup-mtp-head.sh` stages the head
into `reference_weights/Qwen3.8-27B-MTP-4bit`. `./setup-dflash-drafter.sh`
stages the drafter into `reference_weights/Qwen3.8-27B-DFlash2`. `./setup.sh`
runs both, the head first. `tools/resident-up.sh` names one drafter directory
to the worker with `--drafter`, on both legs.

> **WARNING — the DFlash 2 accept rate is not measured.**
> The drafter was trained against a BF16 Qwen3.8-27B trunk. This trunk is the
> same architecture at 2 bits with a folded Hadamard transform. The drafter
> only proposes, so the output cannot move. The accept rate can move, and
> nobody has measured it. Measure it on the box before the first scored window.
> Read a low rate as a fact about the pairing, not as a defect.

> **NOTE — both drafters borrow packed modules.**
> The target's embedding and output projection are packed Hadamard modules on
> this pack. Each drafter calls them as modules, so the forward transform
> applies to the drafter input and the inverse transform applies to the
> embedding output. Code that reads their raw `.weight` skips both.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 97 entries. The
rule behind the list is simple. Code that **proposes** tokens or computes the
forward pass is editable. Code that **verifies**, **measures**, or **ledgers**
stays trusted.

The editable surface has five groups.

1. The decoder declaration. `mtp-head.manifest.json`. The declaration file
   only. It names the decoder and the draft depth.
2. The offline transform. `Sources/MLXFastTransform/`.
3. The model files. Most are in `Vendor/mlx-swift-lm/`:
   `Libraries/MLXLMCommon/PrismHadamardCheckpoint.swift`,
   `Libraries/MLXLLM/Models/PrismHadamardQwen35.swift`, the nine
   `Libraries/MLXLLM/Models/Qwen35*.swift` tower and MTP files, and
   `Libraries/MLXLLM/Models/DFlash2Draft.swift`, the DFlash 2 block drafter,
   and `Libraries/MLXLLM/Models/Qwen35DFlash2Assistant.swift`, its adapter.
   The two DFlash 2 files are editable for the reason `Qwen35MTP.swift` is:
   they propose tokens, and the target decides every emitted token. One more file
   is in the other vendored tree,
   `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift`, which holds the signed
   block transform and the two packed layer types every projection runs.
4. The batching engine and the cache layer: the whole
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/` directory
   (the CBv2 engine, the MTP round driver, the paged KV pool and its kernels),
   twelve `Libraries/MLXLMCommon/` cache, attention, RoPE and config files, and
   `Libraries/MLXLMServer/Runtime/ToolStreamHandler.swift`. This is the shape
   the Gemma 4 track opened.
5. The 68 vendored MLX Metal kernel files the forward pass dispatches. These
   are the quantized matmul, the gather-GEMM, SDPA and steel attention,
   RoPE, RMSNorm, softmax, sort, reduce, copy, elementwise, `arg_reduce`, and
   gather indexing.

The rest of the engine fork is trusted: the runner and its registry,
`bench-worker`, the server, and every `Libraries/MLXLMCommon/` file not named
above.

### The decoder declaration

`mtp-head.manifest.json` is one declaration file for two decoders. It stays
editable and optional. `spec.decoder` names the decoder the submission arms:

```json
"spec": { "decoder": "mtp",    "enabled": true, "num_speculative_tokens": 3 }
```

```json
"spec": { "decoder": "dflash", "enabled": true, "num_speculative_tokens": 7 }
```

An absent `decoder` key reads as `"mtp"`, so a declaration written before the
DFlash 2 arm existed keeps its meaning.

The declaration accepts `"source": "pinned"` only. On this track that means the
pinned export of the DECLARED decoder:
`fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256` for the MTP head, and
`fixtures/reference_bonsai2_27b_dflash2_drafter.sha256` for the DFlash 2
drafter. `"source": "remote"` and `"source": "in_branch"` are refused by name.

Each decoder carries its OWN size cap, because one cap cannot bound both. The
MTP head's cap is 2 GiB (`max_bytes` = 2147483648). The DFlash 2 drafter's cap
is 4 GiB (`max_bytes` = 4294967296), because the drafter is 3.85 GB of BF16. A
declaration may lower the cap of the decoder it names. It may not raise it.

An absent declaration selects the pinned MTP head. That is the normal case. A
declaration that is present but broken is a refusal. The runner never falls
back silently.

The size cap is the only gate on a declaration. A declared `sha256` is
optional, and the runner does not verify it against the drafter bytes.
`docs/participant-contract.md` section 4.3 states that limit plainly.

The declaration also decides which drafter a leg's resident HOLDS.
`tools/resident-up.sh` reads the leg's own declaration through
`tools/spec-declaration.sh` for that one choice. It decides nothing else: the
worker takes the decoder and the depth from the wire, per request.

The MTP head is the organizer's pinned weights. You may re-quantize it. You may
not replace it, and you may not upload head weights of your own. Custom head
weights are not accepted on this track. A re-quantization happens ON LOAD, in
memory. Nothing on disk changes. The head loader calls `quantize(model:)` while
it binds the head, and the file that holds that call is an editable path:
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`.
Change the geometry that call selects. `docs/participant-contract.md` section
4.4 is the authority.

The re-quantization exception names the MTP head. The DFlash 2 drafter is
staged and loaded as published, in BF16.

A drafter only **proposes** tokens. The pinned target model decides every
emitted token. Both legs load a drafter; only the candidate leg drafts with
it.

### The draft depth

> **NOTE — the decoder and the draft depth are free levers.**
> The scored stream count is 1 and you may not tune it. The decoder and the
> draft depth are yours. You declare both in `mtp-head.manifest.json`, which is
> editable:
>
> ```json
> "spec": { "decoder": "mtp", "enabled": true, "num_speculative_tokens": 3 }
> ```
>
> The MTP head permits a depth from 1 to 7
> (`mtp_head.permitted_draft_depths`). The DFlash 2 drafter permits a depth
> from 1 to 16 (`dflash_drafter.permitted_draft_depths`). The two sets are
> independent, and neither decoder can borrow the other's range. With no
> `spec` block, `enabled: false`, or `0`, the run is serial (depth 0),
> whatever the decoder. A depth outside the declared decoder's list is refused
> before the engine starts.
> `tools/spec-declaration.sh describe` prints what your tree declares.
> The ranked run sends the decoder and the depth to the engine on each request
> and seals the engine's `effective_spec` echo.
>
> A declared depth is scored against the oracle recorded AT THAT DEPTH
> (`fixtures/bonsai2_27b_mlx_v1_track.json` `live_golden_speculative`).
> The runner verifies a draft window in one target forward, and the multi-row
> kernels round differently from the single-row ones, so a speculative run is
> not token-identical to the serial tape; it is token-identical to its own
> depth's tape, which the organizer records on the pinned runner. Correctness
> at a depth means matching that tape.
>
> Every run seals what actually ran: `effective_spec` for the declared depth,
> `effective_mean_draft_len` for the realized draft length.

Try a decoder and a depth locally with one flag. The local flag names the leg's
decoder directly, so you do not have to edit the declaration to try a value.

```bash
./benchmark.sh --local-iterate
```

This command runs the local test with no speculation.

```bash
./benchmark.sh --local-iterate --mtp-depth 3
```

This command runs the local test with the MTP head at depth 3.

```bash
./benchmark.sh --local-iterate --dflash-depth 7
```

This command runs the local test with the DFlash 2 drafter at depth 7.

`--mtp-depth` and `--dflash-depth` name the same leg's decoder, so pass exactly
one of them. Each flag also exports the matching drafter directory to the
worker in `BENCH_WORKER_DRAFTER`. `--dflash-depth N` reaches the benchmarker as
`--candidate-spec '{"mode":"dflash","dflash":{"depth":N}}'`, because the
benchmarker takes `--mtp-depth` or `--candidate-spec` and never both.

### The byte budget

`benchmark.json` `editableSurfaceByteBudget` caps the enforced editable
surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 7949663 |
| `maxFileBytes` | 524288 |
| `maxGrowthBytes` | 262144 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is absent. The exemption existed to let head weights ride in a
submission outside the source budget. A submission carries no head weights, so
there is nothing to exempt. The two exempt caps stay declared because both
enforcers carry the same numbers as compiled-in fallbacks and this manifest is
what holds them to a reviewed value.

### The target quantization is frozen

The target model's quantization is frozen as shipped. Do not re-quantize a
target weight. Do not re-represent one. Do not change the numerical format of
one. This holds even when the result passes every correctness gate.

`Sources/MLXFastTransform/` is editable. That does not license a change of
target format. A lossier target substitutes a degraded model instead of
optimizing the accepted one.

The MTP head is a narrow exception, and the exception is re-quantization only.
You may re-quantize the head. You may not replace it. The head stays within its
2 GiB declaration cap. The head only proposes tokens, and the pinned target
decides every emitted token.

The exception names the MTP head. The DFlash 2 drafter is staged and loaded as
published, in BF16, within its own 4 GiB declaration cap.

### What you must not change

- Everything in `Sources/` that `editablePaths` does not list.
- `Package.swift` and `Package.resolved`. The dependency graph is frozen.
- Everything in `Vendor/` that `editablePaths` does not list.
- `fixtures/`, `benchmark.json`, the scripts, the tests, and the
  documents.
- `weights/`, the reference checkpoints, the scores, and the goldens.

Do not hardcode hidden prompts. Do not hardcode hidden token identifiers. Do
not use timing shortcuts, protocol injection, network access, or filesystem
exfiltration.

Do not add a cache keyed on a request's input tokens whose only possible hit is
the harness repeating one identical computation. The benchmark measures
single-pass inference. Input-independent caches stay legal. These are weights,
dequantized tensors, and RoPE or mask tables keyed on shapes and offsets.
Within-request KV reuse also stays legal.

## Local testing vs the ranked run

The local test and the ranked run are different by design. Read this section
before you tune.

The local test runs a **single stream**. It uses a public golden. It prints a
single-stream estimate.

### What each local mode checks

Both local modes run one fused checked-timing pass. The pass teacher-forces the
golden's expected tokens and times the wall clock. It judges correctness from
that same pass. A mismatch is reported as a teacher-forced token mismatch.

A correctness failure does not discard the timing. The benchmarker reruns the
timing phase in a mismatch-tolerant form. It then reports the correctness
failure together with real timing numbers.

| Mode | Decode steps | Expected tokens the golden must hold | Cool gate |
|---|---|---|---|
| `--local-iterate` | 128 | 129 | On, because `./benchmark.sh` always passes `--cool-gate` |
| `--local-submit` | 1023 | 1024 | On |

The gate is on because `./benchmark.sh` arms it. Driving the Swift CLI directly
skips it and times a hot GPU. Use `./benchmark.sh`. See AGENTS.md, "The
cool-down gate".

The two public captures differ in length for this reason. Use the 256-token
capture for `--local-iterate`. Use the 1024-token capture for
`--local-submit`.

> **NOTE — both local modes check correctness and speed.**
> Neither local mode is a speed-only signal. Both apply the teacher-forced
> check. Neither one runs the ranked gates.

The ranked run also runs one stream at a time, so the two shapes agree. A local
number is still only a smoke signal: your machine is not the ranked box.

> **WARNING — a local score is directional, not predictive.**
> Treat a local score as a smoke signal for speed and correctness. Do not treat
> it as a prediction of the ranked composite. The ranked M5 run is the
> authority.

## Scoring and gates

### The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_leg_seconds_per_token / candidate_leg_seconds_per_token
```

The score is serial-anchored. A faster candidate scores above 1.

### The pair is measured, not stored

A ranked run measures TWO legs. It measures them on the SAME box, in the SAME
job, over the ONE prompt the fixture names in `live_golden`:

1. The **serial-control leg**. It runs on the organizer's reference tree. That
   tree is a build of this repository at the commit the fixture names in
   `baseline_reference_commit`. This leg uses no speculation. Its tokens are
   checked against the serial tape, never against a per-depth tape.
2. The **candidate leg**. It runs on your tree, at the draft depth you declare.

The score is the ratio of the two measurements. Both numbers come from the same
machine, minutes apart.

**NO FILE HOLDS A BASELINE PAIR.** The scoring constants hold none. The fixture
holds none. The goldens hold none. A golden that carries
`benchmark.baseline_prefill_seconds_per_token` or
`benchmark.baseline_decode_seconds_per_token` is refused on the ranked path.
`tools/lint-benchmark-manifest.py` keeps the two fields out of the tree.

The organizer stages the reference tree on each ranked box. The ranked job
verifies that tree. It does not fetch it and it does not build it.

`baseline_reference_commit` names `fffde01`, the merge of the DFlash 2 decoder
(PR #3). The control leg runs the ported engine. Re-pin it, and
re-stage the reference tree on every box, whenever a trusted-side code change
merges that the control leg must carry.

### Each box has its own calibration

Each ranked box records what its own serial-control leg costs. The record is a
file. `MLXFAST_BASELINE_CALIBRATION` names it.

**THE FILE IS A HEALTH BAND. IT IS NEVER A DENOMINATOR.** The benchmarker
compares the measured control leg against the band. The run stops by name when
the leg falls outside the band. A stale calibration file can stop a run. It can
never move a score.

An operator writes the file on the box:

```bash
tools/calibrate-box.sh "<runner name>" /path/to/baseline-calibration.json
```

The command takes the box GPU lock. It then runs the serial-control leg four
times under the full official methodology: the quiescence gate and the cool gate
before each pass, one
resident worker for each pass, and the same live golden the ranked run scores
over. It writes the mean, the coefficient of variation and the band for prefill
and for decode. It writes no file when the coefficient of variation is more
than 1 percent on either axis. A box that cannot repeat itself has no band.

The `box` value in the file must equal the runner name. The `reference_commit`
value must equal the fixture's `baseline_reference_commit`.
`tools/ranked-box-preflight.sh` refuses the run when either differs.

### The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 512 |
| Checked decode steps | 128 |
| Golden shape | 512 prompt tokens and 129 expected tokens |
| Streams per window | 1 |
| Timed prompts per leg | 1 (the fixture's `live_golden`) |
| Legs per ranked job | 2 (serial control, then candidate) |

The two legs run one after the other. Each leg loads the weights once. The
unmeasured warm-up prefill pass stays at 1 pass, and it applies to both legs in
the same way.

The serial-control leg runs entirely inside the reference tree. It uses that
tree's worker, that tree's Metal library and that tree's own transformed
weights. Nothing you change can move it.

### One resident worker per LEG, booted by the benchmarker

A ranked job has two legs on two trees. The weights load ONCE per leg.

The benchmarker starts `bench-worker runtime-worker` once for each phase: the
warmup, the timed prefill, the timed decode and the correctness pass. Each start
used to load the whole checkpoint again. It no longer does: a resident owns the
weights for the leg, and each per-phase worker attaches to it.

**THE RESIDENT BELONGS TO THE LEG, NOT TO THE WINDOW.** The two legs run
different trees with different weights, so one resident cannot serve both. The
benchmarker knows where a leg begins and ends, so the benchmarker boots it. For
each leg it calls that leg's OWN copy of `tools/resident-up.sh`:

```bash
tools/resident-up.sh --boot --spec <serial|mtp|dflash> --draft-len <N> --socket-out <file>
tools/resident-up.sh --stop --socket <path>
```

`--boot` loads that tree's own `.build/release/bench-worker`, that tree's own
`weights/` and that tree's own drafter directory, waits for a healthy hello,
writes the socket path as the first line
of the `--socket-out` file, and exits 0 with the resident still running. A
`<socket>.pid` sidecar and a `<socket>.ready` marker sit beside the socket.
`--stop` ends that resident and removes all three files. A second `--stop` is a
no-op.

`--draft-len` is 0 with `serial`, 1 to 7 with `mtp`, and 1 to 16 with
`dflash`. A mismatch is a refusal.

`--spec` is authoritative over what the leg DOES. It is the label the leg
records, so a reference tree that declares a draft depth still boots a serial
control leg when the benchmarker says serial. The worker takes the decoder and
the depth from the wire, per request, and never from the tree.

`--spec` does NOT choose the drafter. The boot reads the leg's OWN
`mtp-head.manifest.json`, through `tools/spec-declaration.sh`, for that one
decision: which of the two staged drafters this leg's resident holds. Each leg
therefore holds the drafter its own tree declares, and nothing crosses between
the two trees.

`tools/stage-baseline-workspace.sh` stages both drafters into the reference
tree. It reads the DFlash 2 drafter from
`MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR`, and that copy costs 3.85 GB there.

`tools/bonsai2-27b-measure-and-score.sh` boots nothing. It takes the box
GPU lock `/tmp/mtplx-gpu-exclusive.lock` and holds it for the whole
measurement, because a resident holds about 8.6 GB of unified memory whoever
booted it, and the box needs exactly one loader. `tools/resident-up.sh` refuses
to boot when nobody holds that lock, so every per-leg boot happens inside that
window.

| Name | What it is |
|---|---|
| `BENCH_WORKER_RESIDENT_SOCKET` | A leg's resident Unix socket. A per-phase worker attaches to it instead of loading. The measure script never sets it, and REFUSES when the environment does. |
| `RESIDENT_IDENTITY_FILE` | A JSON file that records the resident, its socket, its declared leg and its hello. |

> **WARNING — do not set `BENCH_WORKER_RESIDENT_SOCKET` on a ranked box.**
> It names one already-loaded resident. Both legs would attach to it, so the
> serial-control leg would run on the candidate's weights. The measure script
> and `tools/ranked-box-preflight.sh` both refuse it.

The wrapper form of `tools/resident-up.sh` stays for local, unscored use:

```bash
tools/resident-up.sh --weights <dir> -- <command>
```

It boots one resident, exports `BENCH_WORKER_RESIDENT_SOCKET` to the command,
and halts the resident when the command ends. The ranked path does not use it.

### The parameters

| Parameter | Value |
|---|---|
| `scoredBatchSize` | 1 |
| `prefillGainExponent` | 0.25 |
| `decodeGainExponent` | 0.75 |
| `pairsPerCohort` | 3 |
| `minPairsPerCohort` | 3 |
| `decodeSpeedupFloor` | 0.95 |
| `decodeSpeedupCeiling` | 5.0 |
| `kvBackend` | `contiguous` |

The scored width is fixed. A width the benchmarker has not certified has no
series tag, and the benchmarker refuses that width rather than run it.

The fixture's `official_pairs` sets how many pairs one ranked run measures.
Each pair scores on its own control leg. The run scores the pair whose
composite is the lower median over the pairs. Pairs are never averaged. The
other pairs stay in the artifact as measured.

`kvBackend` is pinned `contiguous` on both legs. The benchmarker refuses when
it cannot honour the pinned backend. It does not degrade to another backend.

### Token fidelity

The benchmarker applies a per-stream token-tolerance gate with a **10% budget**.

> **WARNING — the gate accepts similar output, not identical output.**
> This track does not require your output to match the serial trajectory token
> for token. A speculative forward verifies a whole draft chain in one pass, and
> that pass diverges from a one-token-at-a-time decode at near-tie argmaxes. The
> gate prices that divergence against the 10% budget. Do not read the gate as
> lossless.

A near-tie argmax can diverge across Apple Silicon generations, even for
correct code. Before you treat a local failure as your own regression, check
whether an unmodified `main` fails at the same token position on your machine.

### Current status

Five statements are true right now.

1. `fixtures/bonsai2_27b_mlx_v1_track.json` sets
   `official_scoring_enabled` to `false`. That flag is the single authority on
   the arm state, and the benchmarker enforces it. It refuses to seal an
   official scoring artifact while the flag is false.
2. The track has no goldens. `timed_prompt_pool` is empty, `live_golden` is
   empty, and `hidden_correctness_golden` carries the pending sentinel
   `BONSAI2-27B-MLX-V1-PENDING-ORGANIZER`.
3. No runner advertises the ranked label set
   `[self-hosted, macOS, bonsai2-27b-mlx-v1]`, and no box is staged
   with the goldens, the reference tree or a calibration file.
   `tools/stage-baseline-workspace.sh` builds the reference tree, and
   `tools/calibrate-box.sh` writes the calibration file, when a box is ready.
4. Scoring is single-stream and paired. `scored_batch_size` is `1`. The
   candidate leg runs one stream at its declared decoder and depth. The
   serial-control leg runs on the organizer's reference tree at the fixture's
   `baseline_reference_commit`. The score is the live ratio. No golden carries
   a baseline pair.
5. The bench channel is the bench repository's `main` dist channel.
   `./tools/fetch-benchd.sh` reads that channel's `dist/benchd.manifest.json`,
   verifies the binary against the `sha256` and `bytes` the manifest names, and
   prints the identity it resolved. THE MANIFEST IS THE SOURCE OF TRUTH for
   which `benchd` measures a run, and this document pins nothing: read the
   identity off the script, not off this page.

The model port has landed as SOURCE. The engine constructs and gates
`prism_hadamard_qwen35`, and `Sources/MLXFastCore/Constants.swift` carries this
target's geometry. NOTHING IN THE PORT HAS BEEN COMPILED OR RUN: build it and
run the local test on a box before you trust it.
`docs/bonsai2-27b-port-notes.md` holds the detail and the open items.

## Submitting

Use the Yukon CLI for every account operation and every submission operation.

```bash
export PATH="${HOME}/.local/bin:${PATH}"
```

This command puts `yukon` on your path.

```bash
yukon login <api-key> --api <url>
```

This command authenticates you.

```bash
yukon clone <benchmark-id-or-name>
```

This command clones the benchmark repository.

```bash
yukon submit --model "<exact model name>" --note-file submission-note.md
```

This command uploads your editable-path archive.

```bash
yukon submissions
```

This command lists your submissions.

A submission archive replaces the editable paths. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first.
No local run blocks the upload. Run the local test yourself before you submit.

## The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ `3f926b415992eaa2ae9dd7b573706494d6bbf787` |
| Target manifest | `fixtures/reference_bonsai2_27b_2bit.sha256` (19 records, 8,608,721,032 bytes) |
| MTP head | `EigenLabs/Qwen3.8-27B-MTP-4bit` @ `329261c5e0b3f9c233485e682cb3b67b88c20a55`, pinned by `fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256` (4 records, 238,942,843 bytes) |
| DFlash 2 drafter | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, Apache-2.0, pinned by `fixtures/reference_bonsai2_27b_dflash2_drafter.sha256` (3 records, 3,848,824,532 bytes) |
| Benchmarker | The bench repository's `main` dist channel, resolved at run time (`benchd.manifest.json` is the authority for the bytes; `tools/fetch-benchd.sh` enforces it and logs the resolved `source_commit`/sha256) |
| Track id | `bonsai2-27b-mlx-v1` — the PLATFORM name: leaderboard namespace, runner labels, R2 prefix |
| Engine fork upstream base | `859f4d97e0d1ae9cce0883b044f764283841edfb` (+ fork `main` `0607011` merged; engine-local changes on top) |

The model repository is public. It downloads without a token. There is no
organizer-hosted mirror for this checkpoint, so
`MLXFAST_REFERENCE_FALLBACK_BASE_URL` is empty by default.

### The target model

| Property | Value |
|---|---|
| Architecture | The model type is `prism_hadamard_qwen35`, over a `qwen3_5` backbone. Schema 2. |
| Hidden layers | 64. Every fourth layer is full attention; the other three are gated-delta-net linear attention. |
| Layer counts | 16 full attention, 48 linear attention |
| Attention layers | At indices 3, 7, 11, … 63 |
| Attention | 24 query heads, 4 KV heads, head dimension 256, no bias, output gate on |
| Rotary | Partial (`partial_rotary_factor` 0.25), `rope_theta` 10000000, mRoPE sections 11/11/10 |
| Linear attention | 16 key heads and 48 value heads of dimension 128, conv kernel 4, FP32 state |
| MLP | Dense. A gate and an up projection of width 17408 into a down projection. No experts. |
| Hidden size | 5120 |
| Vocabulary | 248320 |
| Embeddings | Untied |
| Context | 262144 |
| Tokens | bos 248044, eos 248044 |
| Quantization | Affine, group size 128, 2 bits, no mixed precision. FP16 scales and biases. |
| Transform | A signed normalized block Walsh-Hadamard transform of width 1024, folded into all 402 packed modules. `hadamard.json` is the manifest. |
| Raw tensors | 2390 in one shard, 8,595,174,880 bytes of tensors |
| Tensor types | 1137 F16, 851 F32, 402 U32 |
| Vision | The pack declares a tower and ships its 333 tensors. This track serves text only and never builds it. |
| Base model | A 2-bit MLX pack of `Qwen/Qwen3.8-27B` |

Only the 16 full-attention layers carry a KV cache. The 48 linear-attention
layers carry a constant-size recurrent state instead.

Every packed projection applies the forward transform to its input, and the
embedding applies the inverse transform to its output. Both run in FP32. Call
the module; do not read its raw `.weight`.

## Building after an edit

Two build forms matter, because the vendored MLX package builds in JIT mode.

Kernel families with an `mlx-generated/*.cpp` twin compile at runtime from the
C++ source strings inside those files. For these families the twin is the
runtime-effective source. Edit the twin. Keep the readable `.metal` and `.h`
pair in step with it.

RoPE, RMSNorm, the SDPA vector kernel, and `arg_reduce` load ahead of time from
`mlx.metallib`. After you edit one of those sources, run
`tools/build-mlx-metallib.sh`. `./setup.sh` runs that script for you.

`_nax` names are the M5-generation kernel variants. The ranked runner selects
them. Tune the `_nax` twin as well as the plain one.

```bash
swift build -c release --force-resolved-versions
```

This command builds the trusted CLI into `.build/release`.

```bash
swift build -c release --force-resolved-versions --scratch-path .build-worker \
  --product bench-worker
```

This command builds the scored engine into `.build-worker/release`. The engine
is the fork's generic `bench-worker`, and this package builds it as a
dependency product, so the build reads `Vendor/mlx-swift` and your kernel edits
reach it.

```bash
tools/stage-bench-worker.sh
```

This command copies the engine and its `mlx.metallib` into `.build/release`,
where the benchmarker resolves them.

> **WARNING — always pass `--force-resolved-versions`.**
> The dependency graph is frozen. A bare `swift build` or `swift test` can
> rewrite `Package.resolved`. `./setup.sh` then refuses to run. Restore the
> file with `git checkout -- Package.resolved`.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on every push to
`main`. It runs repository hygiene checks on `ubuntu-latest`. It runs
`swift build --build-tests` and `swift test` on a hosted `macos-26` runner. It
treats first-party warnings as errors.

CI is advisory. No status check is required. A red run blocks neither a merge
nor a dispatch. `docs/ci-coverage.md` holds the detail.

CI never measures and never scores. CI holds no secret, downloads no weights,
and runs no GPU test. The GPU tests and the checkpoint tests are box-only.
`docs/ci-coverage.md` lists them, and CI fails when that list drifts.

`.github/workflows/benchmark.yml` is the ranked pipeline. It triggers on
`workflow_dispatch` only. It holds no secret. Its ranked job runs on
`[self-hosted, macOS, bonsai2-27b-mlx-v1]`. Before it measures, it verifies
every box-staged asset against the contract's `{sha256, bytes}` pins. It
publishes no score until the runner is registered, the box is staged, and the
benchmarker emits a composite. Each of those gaps gives a non-zero exit and no
artifact.

## Where to get help

| Question | Authority |
|---|---|
| What the track measures, path by path | `benchmark.json` |
| Pins, the timed pool, scoring values | `fixtures/bonsai2_27b_mlx_v1_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| The engineering record for this port | `docs/bonsai2-27b-port-notes.md` |
| What CI covers | `docs/ci-coverage.md` |
| Agent and contributor guidance | `AGENTS.md` |

> **NOTE — the order of authority.**
> The ranked M5 run is the authority on any score. The contract fixture
> `fixtures/bonsai2_27b_mlx_v1_track.json` wins over this document. This document
> only explains; it never overrides. If either disagrees with the benchmarker
> about measurement, the benchmarker wins.

## License and attribution

This repository's harness code is licensed per [LICENSE](LICENSE). The pinned
checkpoint carries OpenMDW-1.1. The terms ship with the checkpoint at its
pinned revision. This repository distributes no model weights.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) holds the full third-party
attribution.
