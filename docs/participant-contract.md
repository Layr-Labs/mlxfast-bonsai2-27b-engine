# Ternary Bonsai 2 27B — participant contract

This document states the terms that bind a submission to track
`bonsai2-27b-mlx-v1`.

`benchmark.json` is the Yukon track manifest.
`fixtures/bonsai2_27b_mlx_v1_track.json` is the track contract fixture. Both files
carry pure configuration: values, paths, commands, and pins. They carry no
prose. This document explains those files. It never overrides them.

## 1. Order of authority

Apply these in order. The higher entry wins.

1. The ranked run on the official runner. It is the authority on any score.
2. `fixtures/bonsai2_27b_mlx_v1_track.json` and `benchmark.json`.
3. This document.
4. `README.md` and `TASK.md`.

If either configuration file disagrees with this document on a plain value, the
configuration file wins. If either disagrees with the benchmarker about
measurement, the benchmarker wins.

The benchmarker is a prebuilt `benchd` binary resolved from the bench
repository's `main` dist channel. The channel publishes
`benchd.manifest.json` (`{branch, source_commit, sha256, bytes}`) beside the
binary; `./tools/fetch-benchd.sh` verifies the binary against that manifest,
installs both into `benchd-bin/`, and logs the resolved identity. The harness is
trusted-side: a submission cannot change what measures it. This repository
carries no submodule and no sha pin.

Both vendored trees, `Vendor/mlx-swift` and `Vendor/mlx-swift-lm`, are plain
files in this repository. Neither is a submodule. A fresh `git clone` carries
everything a build needs.

## 2. What the track measures

The track measures Ternary Bonsai 2 27B MLX inference speed.

You optimize the MLX runner, the offline transform, and the vendored MLX Metal
kernel families that the forward pass dispatches. You also optimize the
speculative-decode arm.

The target model is `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` at revision
`3f926b415992eaa2ae9dd7b573706494d6bbf787`. It is a 2-bit packed build of a
dense Qwen3.8-27B backbone: gated-delta-net recurrence on three layers out of
every four, and full attention on the fourth.

| Property | Value |
|---|---|
| Architecture | The model type is `prism_hadamard_qwen35`, over a `qwen3_5` backbone. Schema 2. |
| Hidden layers | 64. Every fourth layer is full attention. |
| Layer counts | 16 full attention, 48 linear attention |
| Attention layers | At indices 3, 7, 11, ... 63 |
| Attention | 24 query heads, 4 KV heads, head dimension 256, no attention bias, output gate on |
| Rotary | Partial (`partial_rotary_factor` 0.25), `rope_theta` 10000000, mRoPE sections 11/11/10 |
| Linear attention | 16 key heads and 48 value heads of dimension 128, convolution kernel 4, FP32 state, grouped GDN layout |
| MLP | Dense: a gate and an up projection of width 17408 into a down projection. No experts. |
| Hidden size | 5120 |
| Vocabulary | 248320, embeddings untied |
| Context | `max_position_embeddings` 262144 |
| Tokens | bos 248044, eos 248044 |
| Quantization | Affine, group size 128, 2 bits, no mixed precision. FP16 scales and biases. |
| Transform | A signed normalized block Walsh-Hadamard transform of width 1024, folded into all 402 packed modules. `hadamard.json` is the manifest. |
| Raw tensors | 2390 in one shard, 8,595,174,880 bytes of tensors: 1137 F16, 851 F32, 402 U32 |
| Namespace split | 2057 `language_model.` tensors and a 333-tensor vision tower |

Only the 16 full-attention layers carry a key-value cache. The 48
linear-attention layers carry a constant-size recurrent state instead.

EVERY PACKED PROJECTION CARRIES THE TRANSFORM. The forward transform applies to
the input of each packed linear and the inverse transform applies to the output
of the embedding. Both run in FP32 and restore the incoming dtype. Call the
module. A path that reads a packed module's raw `.weight` skips the transform
and returns wrong numbers while it still compiles and still runs.

THE PACK DECLARES A VISION TOWER and ships its 333 tensors. This track serves
text only. The transform drops that namespace, the runner declares
`multimodal: false`, and the model filters the same prefix at load.

The speculative-decode arm has TWO declarable decoders. The pack declares
`mtp_num_hidden_layers: 0` and carries no drafter of any kind, so each decoder
is a SEPARATE published export.

The first is an MTP head: `EigenLabs/Qwen3.8-27B-MTP-4bit` at
`329261c5e0b3f9c233485e682cb3b67b88c20a55`, model type `qwen3_5_mtp`, 31
tensors, 1 full-attention hidden layer, affine 4-bit at group 64.

The second is a DFlash 2 block drafter: `z-lab/Qwen3.8-27B-DFlash2` at
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, architecture `DFlash2DraftModel`,
81 BF16 tensors, 5 sliding-attention hidden layers. Section 4.5 states its pin,
its cap and its depth range.

Neither has an embedding table or an `lm_head` of its own. Both ride the
target's, which is what `use_dedicated_embeddings: false` means here.

The arm is single-stream. The runner's MTP assistant declares
`maximumSpeculativeBatch == 1`, so the runner declares single-stream regimes
only.

`kv_backend` is pinned `contiguous` on both legs. The benchmarker refuses when
it cannot honour the pinned backend. It does not degrade to another backend.

## 3. What you may edit

`benchmark.json` `editablePaths` is the authority. It lists 97 entries.

The rule behind the list: anything that only **proposes** tokens or computes
the forward pass is editable. Anything that **verifies**, **measures**, or
**ledgers** stays trusted.

The editable surface has five groups.

1. **The decoder declaration.** `mtp-head.manifest.json`. The declaration
   file only. It names the decoder and the draft depth. It carries no weights,
   and no drafter weights directory exists. See section 4.
2. **The offline transform.** `Sources/MLXFastTransform/`.
3. **The model files.** Under `Vendor/mlx-swift-lm/`, unless the line says
   otherwise:

   - `Libraries/MLXLMCommon/PrismHadamardCheckpoint.swift`
   - `Libraries/MLXLLM/Models/PrismHadamardQwen35.swift`
   - `Libraries/MLXLLM/Models/Qwen35.swift`
   - `Libraries/MLXLLM/Models/Qwen35+CompleteCheckpoint.swift`
   - `Libraries/MLXLLM/Models/Qwen35A3BOptimization.swift`
   - `Libraries/MLXLLM/Models/Qwen35A3BTargetVerify.swift`
   - `Libraries/MLXLLM/Models/Qwen35MTP.swift`
   - `Libraries/MLXLLM/Models/Qwen35MTP+PrefixCheckpoint.swift`
   - `Libraries/MLXLLM/Models/Qwen35MTPTopTwo.swift`
   - `Libraries/MLXLLM/Models/Qwen35MoE.swift`
   - `Libraries/MLXLLM/Models/DFlash2Draft.swift`
   - `Libraries/MLXLLM/Models/Qwen35DFlash2Assistant.swift`
   - `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift` (the OTHER vendored tree)

   `DFlash2Draft.swift` is the DFlash 2 block drafter, and
   `Qwen35DFlash2Assistant.swift` is its adapter: it feeds the drafter the
   target rows and trims the drafter's cache. Both are editable for the reason
   `Qwen35MTP.swift` is, which holds the MTP head and its adapter in one file:
   they PROPOSE tokens, and the pinned target decides every emitted token.

4. **The batching engine and the cache layer.** The whole
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/` directory
   (the CBv2 engine, the MTP round driver and depth controller, the paged KV
   pool and its Metal kernels), and these files under
   `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/`: `AttentionUtils.swift`,
   `BaseConfiguration.swift`, `CompilableKVCache.swift`,
   `CompilableRotatingKVCache.swift`, `DynamicSlice.swift`, `Evaluate.swift`,
   `JSONDecodingTypes.swift`, `KVCache.swift`, `LanguageModel.swift`,
   `RoPEApplication.swift`, `RoPEUtils.swift`, `SwitchLayers.swift`; plus
   `Vendor/mlx-swift-lm/Libraries/MLXLMServer/Runtime/ToolStreamHandler.swift`.
   This is the same shape the Gemma 4 track opened (David ruling 2026-09-16).
5. **The vendored MLX Metal kernels.** The 68 files the forward pass
   dispatches: the quantized matmul, the gather-GEMM, SDPA
   and steel attention, RoPE, RMSNorm, softmax, sort, reduce, copy,
   elementwise, `arg_reduce`, and gather indexing.

THE ENGINE FORK IS A VENDORED TREE. `Vendor/mlx-swift-lm` is a copy of
`Layr-Labs/mlx-swift-lm`, as plain files, with this repository's own commits on
top. It is a copy and not a submodule for one reason: an editable path names
bytes in this tree, and a gitlink names a commit. The upstream base the tree
was cut from is in this contract's `mlx_swift_lm_revision`.

THE MODEL FILES, THE BATCHING ENGINE AND THE CACHE LAYER ARE EDITABLE. The
rest of the fork is trusted: the runner and its registry, `bench-worker`, the
server, and every `Libraries/MLXLMCommon/` file not named above. A submission
that changes any of them is refused by the surface gate.

### 3.1 Optional paths

`optionalEditablePaths` lists `mtp-head.manifest.json`.

A submission archive has REPLACE semantics over `editablePaths`. An absent head
declaration means the pinned head. The overlay therefore skips a missing
optional path instead of failing closed.
`.github/scripts/overlay-editable-paths.sh` reads this list from the trusted
contract, never from the submission.

### 3.2 The byte budget

`editableSurfaceByteBudget` caps the editable surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 7949663 |
| `maxFileBytes` | 524288 |
| `maxGrowthBytes` | 262144 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is **absent** since 2026-08-26. The exemption existed for one
reason: to let head weights ride in a submission outside the source budget. A
submission carries no head weights any more, so there is nothing to exempt.

The two exempt caps stay declared. They cannot bind while `exemptPaths` is
absent. They stay because both enforcers carry the same two numbers as
compiled-in fallbacks, and this manifest is what holds those constants to a
reviewed value. `tools/lint-benchmark-manifest.py` check 3b enforces that
equality.

No head weight file is staged, so this budget never meets one. The head is part
of the pinned target checkpoint, which is outside the editable surface. What
the runner LOADS is bounded instead by the 2 GiB declaration cap in section 4.

### 3.3 What you may not edit

You may not edit anything that verifies, measures, or ledgers. This covers the
trusted harness, the target weights, the transform contract, the tokenizer, the
goldens, the gates, and the timing and telemetry code. `fixtures/` is outside
the editable surface. The scoring step reads the contract from the trusted
checkout for that reason.

### 3.4 The target quantization is frozen

The target model's quantization is frozen as shipped.

A submission must not re-quantize any target weight. It must not re-represent a
target weight. It must not change the numerical format of a target weight. This
holds even when the result passes every correctness gate.

`Sources/MLXFastTransform/` is editable. That does not license a change of
target format. A lossier target substitutes a degraded model. It does not
optimize the accepted one.

The MTP head is a narrow exception, and the exception is re-quantization only.

You may re-quantize the MTP head. You may **not** replace it. You may **not**
upload head weights of your own. Custom head weights are not accepted on this
track.

This is the 2026-08-26 ruling. It replaces the earlier bring-your-own-head
design, under which a participant could declare and ship a head of their own
choosing. That design is retired.

The head is the organizer's pinned weights, because it is part of the pinned
target checkpoint. `fixtures/bonsai2_27b_mlx_v1_track.json` names the repository
and revision, and `fixtures/reference_bonsai2_27b_2bit.sha256` carries the
per-file digests that `./setup.sh` verifies every downloaded byte against. Both
files live in `fixtures/`, which is outside the editable surface.

Three things enforce this, and section 4 states each one:

1. No head weights directory exists and no submission path can hold head
   weights. A submission that carries a weight file is refused.
2. The head declaration accepts `"source": "pinned"` only. `"remote"` and
   `"in_branch"` are refused by name.
3. A re-quantization runs on load, in memory, on the benchmark machine.
   No re-quantized file is made, so there is no artifact to travel in a
   submission. Section 4.4 states the mechanism.

The loader reads a `quantization` block in the shape an MLX conversion writes.
That block selects which modules load quantized and at what geometry. The
accepted parameters are `group_size` (positive, at most 65536), `bits` between
2 and 8, and optional per-layer overrides (at most 8192 entries). A value
outside those bounds is refused by name.

The MTP head loader does not check a declare-versus-carry mismatch. An absent
declaration skips quantization, and packed weights then fail later inside the
weight bind with a shape error. A declaration with no packed tensor quantizes
nothing, silently. That limit is stated here rather than promised away.

The reason for the whole exception is the propose-and-decide split. The head
only proposes tokens. The pinned target model decides every emitted token.

## 4. The speculative decoders

The track carries TWO declarable speculative decoders and ONE declaration file.
Both are the organizer's weights. The pinned pack carries no drafter of any
kind, so each decoder is its own separate pinned export.

`mtp-head.manifest.json` `spec.decoder` names the decoder a submission arms:
`"mtp"` for the MTP head, `"dflash"` for the DFlash 2 drafter. The key is
optional and reads as `"mtp"` when it is absent, so a declaration written
before the DFlash 2 arm existed keeps its meaning.

| Item | The MTP head | The DFlash 2 drafter |
|---|---|---|
| Declaration | `mtp-head.manifest.json` | `mtp-head.manifest.json` |
| Where the weights are | A separate pinned export, `reference_weights/Qwen3.8-27B-MTP-4bit` | A separate pinned export, `reference_weights/Qwen3.8-27B-DFlash2` |
| Organizer pin | `EigenLabs/Qwen3.8-27B-MTP-4bit@329261c5`, `fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256` | `z-lab/Qwen3.8-27B-DFlash2@50307d4c`, `fixtures/reference_bonsai2_27b_dflash2_drafter.sha256` |
| Shape | 31 tensors, 1 full-attention hidden layer, affine 4-bit group 64 | 81 tensors, 5 sliding-attention hidden layers, BF16 |
| Declaration cap | 2 GiB | 4 GiB |
| Permitted draft depths | 1 to 7 | 1 to 16 |
| Staged by | `./setup-mtp-head.sh` | `./setup-dflash-drafter.sh` |

The declaration file is editable. Neither pinned export is.

No submission carries a drafter weight file. `./setup.sh` stages both exports,
the head first.

Sections 4.1 to 4.4 hold the MTP head's rules, and those rules are unchanged.
Section 4.5 holds the DFlash 2 drafter's own pin, its own cap and its own depth
range.

### 4.1 What you may declare

`"source": "pinned"` is the only accepted source. On this track it means the
separate head export the organizer pins. A declaration may also state
`max_bytes` (it may lower the 2 GiB track cap and may not raise it), a `bytes`
count, and an optional `sha256`. It carries no `arm` key: `spec.decoder` is the
selection, and it sits beside the depth it goes with.

`"source": "remote"` is refused by name. `"source": "in_branch"` is refused by
name. Both were accepted before the 2026-08-26 ruling and both meant "load
weights the participant chose". The refusal names the retired source and names
`pinned` as what replaced it.

An absent declaration selects the pinned head. A declaration that is present
but broken is a refusal. The runner never falls back silently.

### 4.2 What you may not do

You may not ship head weights. No path in the editable surface can hold them,
so a submission that carries a weight file is refused before any measurement.
`.github/scripts/enforce-modifiable-surface.sh` names the file and refuses.
`.github/scripts/overlay-editable-paths.sh` never copies it. The benchmarker's
own write-divergence gate refuses any content that differs from the trusted
baseline outside the editable surface.

You may not edit the checkpoint's head tensors. The checkpoint is not an
editable path, so any change to it is outside the surface.

### 4.3 What the size cap does and does not do

The 2 GiB declaration cap (`max_bytes` = 2147483648) bounds what the runner
loads.

The size cap is the only gate on the declaration. A declared `sha256` is
optional, and the runner does not verify it against the head bytes. It treats a
wrong digest and an absent digest alike. That is stated here plainly because it
is a real limit, not a detail: **nothing in this repository binds the loaded
head bytes to the organizer's pinned digests at run time.** The harness
computes a head digest that it reports and never compares.

What does bind at run time is relative, not absolute: the benchmarker compares
the candidate workspace against the trusted baseline workspace and refuses any
divergence outside the editable surface. A correctly provisioned baseline is
therefore load-bearing for the whole property.

The head bytes themselves are bound one level up. They are part of the pinned
target checkpoint, and `./setup.sh` verifies every downloaded byte against the
per-file digests in `fixtures/reference_bonsai2_27b_2bit.sha256`. Both
legs load the head out of that one verified checkpoint.

### 4.4 How a re-quantization reaches the box

A re-quantization happens ON LOAD, in memory. Nothing on disk changes.

You do not make a re-quantized checkpoint. Your code quantizes the head's
parameters in memory, in the same pass that binds them. The staged bytes are
only read.

#### What to edit

The head loader calls `quantize(model:)` while it binds the checkpoint. That
call is the seam, and the file that holds it is an editable path:
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`.

Change the geometry that call selects. The default reads the checkpoint's own
per-layer quantization for the matching `mtp.` key. Your code may select a
different geometry instead.

Section 3.4 states the bounds the loader accepts: `group_size` positive and at
most 65536, `bits` between 2 and 8, and at most 8192 per-layer overrides. A
value outside those bounds is refused by name. The loader does not check a
declare-versus-carry mismatch; section 3.4 states that limit.

#### Why nothing is written

Two properties follow from the in-memory rule, and both are why this mechanism
is the safe one.

1. The benchmarker compares the candidate workspace against the trusted
   baseline workspace and refuses any change outside the editable surface. That
   comparison reads the disk. A re-quantization on load is not a disk
   operation, so there is nothing for the gate to see.
2. The ranked worker runs under a sandbox profile that denies file writes. Code
   that tried to rewrite a staged head would fail there.

Do not rewrite the head tensors on disk. Do not rewrite them from `setup.sh` or
from the `mlxfast-swift transform` command. Each of those runs before the
workspace comparison, and the benchmarker refuses the change.

#### What changes in the record

The worker reports the digest of the head it loaded. That digest is the digest
of the ORGANIZER's checkpoint bytes, before and after a re-quantization,
because the bytes do not change. The geometry you selected is not visible in
that digest.

#### What this does not permit

The exception is for the HEAD's OWN weights only. The target model's
quantization stays frozen, as section 3.4 states.

**The head is a separate module tree, and the rule is drawn by module path.**
David ruling 2026-08-27, relayed by orchestrator: "Exempt mtp.* from the
freeze." The head binds to the loaded target, so the loaded-target check has to
say which side of the line each module is on. It says it this way:

| Module path | Treatment |
|---|---|
| The head's own modules | EXEMPT. Re-quantize them on load. |
| A head module naming `embed_tokens` or `lm_head` | REFUSED BY NAME. |
| Everything else, `model.embed_tokens` and `lm_head` included | FROZEN. |

**The shared tensors are the middle row, and they are shared for a real
reason.** The head owns no embedding table and no output projection. It READS
the target's embedding table for its next-token vectors and the target's output
projection for its logits. On this pack both are PACKED HADAMARD modules, so
they are read through the module and the transform applies. Those two tensors
decide the TARGET's tokens. Coarsening one of them is a target re-quantization
whatever path it is spelled under, so a quantized module inside the head
subtree that names either one is refused, and the refusal says which shared
tensor it reached.

The target is verified TWICE, and both checks read the loaded model, not the
declaration:

1. At worker startup, immediately after the target is loaded.
2. Again at the top of each window that gets measured, immediately before the
   measured work starts.

The second check exists because the first one alone verifies a model that code
can still change afterwards. Both refuse by name, and a refusal stops the worker
before any measurement.

The head only **proposes** tokens. The organizer-pinned target model decides
every emitted token. Both legs load the pinned head; only the candidate leg
drafts with it.

### 4.5 The DFlash 2 drafter

The DFlash 2 drafter is the track's second declarable decoder. It is the
organizer's pinned weights, exactly as the MTP head is, and it is used
unmodified.

| Item | Value |
|---|---|
| Organizer pin | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Pin file | `fixtures/reference_bonsai2_27b_dflash2_drafter.sha256`, 3 records, 3,848,824,532 bytes |
| License | Apache-2.0 |
| Architecture | `DFlash2DraftModel` |
| Tensors | 81, all BF16, one shard, 3,848,808,960 tensor bytes |
| Hidden layers | 5, sliding attention, sliding window 2048 |
| Attention | 32 query heads, 8 key/value heads, head dimension 128 |
| Shape | Hidden 5120, intermediate 17408, vocabulary 248320, rope base 1e7 |
| Embeddings | None of its own. It binds the target's. |
| Declaration cap | 4 GiB (`max_bytes` = 4294967296) |
| Permitted draft depths | 1 to 16 (`dflash_drafter.permitted_draft_depths`) |
| Staged by | `./setup-dflash-drafter.sh`, which `./setup.sh` runs |

**THE CAP IS ITS OWN.** The MTP head's 2 GiB cap does not move for this
decoder. One cap cannot bound both drafters: the head is 239 MB of 4-bit
weights and the drafter is 3.85 GB of BF16. `max_bytes` may lower the cap of
the decoder a declaration names, and it may not raise it.

**THE DEPTH RANGE IS ITS OWN.** A DFlash 2 depth is the drafter's BLOCK SIZE
minus one. The drafter proposes a whole block in one forward pass, and the
block is the last committed token followed by that many mask tokens. The
drafter was trained at block 8, which is depth 7. A larger block is permitted,
and it simply accepts less. The contract sets the ceiling at 16. The MTP head's
1-to-7 range is unchanged, and neither decoder can borrow the other's range.

**THE SHARED TENSORS STAY THE TARGET'S.** The drafter owns no embedding table
and no output projection. It binds the target's, which on this pack are packed
Hadamard modules, so the drafter calls the modules and never reads a raw
`.weight`. The module-path table in section 4.4 is unchanged and holds here
too: those two tensors decide the TARGET's tokens, and a coarsened one is a
target re-quantization whatever path it is spelled under.

`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/DFlash2Draft.swift` is the Swift
drafter, and it is one of the editable paths. It is editable for the reason
`Qwen35MTP.swift` is: it PROPOSES tokens. The re-quantization exception of
section 3.4 names the MTP head. The DFlash 2 drafter is staged and loaded as
published, in BF16.

**THE ACCEPT RATE IS NOT MEASURED.** The drafter was trained against a BF16
Qwen3.8-27B trunk. This trunk is the same architecture at 2 bits with a folded
Hadamard transform. The drafter can only propose, so the output cannot move.
The accept rate can move, and nobody has measured it. It must be measured on
the box before the first scored window. Read a low rate as a fact about the
pairing rather than as a defect.

The drafter only **proposes** tokens. The organizer-pinned target model decides
every emitted token. Both legs of a pair load a drafter; only the candidate leg
drafts with it.

## 5. Scoring

### 5.1 The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_leg_seconds_per_token / candidate_leg_seconds_per_token
```

The score is serial-anchored. A faster candidate scores above 1.

### 5.1.0 The pair is measured on the box, and no file stores it

**DAVID RULING 2026-09-08: EACH RANKED MACHINE HAS ITS OWN BASELINE.** A ranked
run measures TWO legs. It measures them on the SAME box, in the SAME job, over
the ONE prompt the fixture names in `live_golden`:

1. The **serial-control leg**. It runs on the organizer's reference tree, which
   `MLXFAST_BASELINE_WORKSPACE` names. That tree is a build of this repository
   at the commit the fixture names in `baseline_reference_commit`. The leg uses
   no speculation. The measure script passes the serial tape to benchd as
   `--control-golden`, so this leg is verified against the serial tape and the
   candidate leg against the tape recorded at its declared depth.
2. The **candidate leg**. It runs on the submission tree, at its declared draft
   depth.

```text
score = (ref_prefill_spt / cand_prefill_spt) ^ 0.25
      * (ref_decode_spt  / cand_decode_spt ) ^ 0.75
```

The floors do not change.

**NO STORED PAIR EXISTS ANYWHERE.** Not in the scoring constants. Not in the
fixture. Not in a golden. A golden that carries
`benchmark.baseline_prefill_seconds_per_token` or
`benchmark.baseline_decode_seconds_per_token` is REFUSED on the ranked path.
`tools/lint-benchmark-manifest.py` check 5b keeps both fields out of this
repository.

The organizer stages the reference tree on each ranked box.
`tools/stage-baseline-workspace.sh` builds it there from a staged mirror or
bundle. The ranked job verifies the tree. It never fetches or builds it, because
the job holds no credential.

The reference tree carries its own engine, its own Metal library and its own
transformed weights. The candidate cannot move the control leg.

`baseline_reference_commit` names `fffde01`, the merge of the DFlash 2 decoder,
so the control leg runs the ported engine. A trusted-side code change that the
control leg must carry re-pins it, and every box re-stages its reference tree
at the new commit before its next ranked run.

### 5.1.0.1 The per-box calibration is a health band

Each ranked box records what its own serial-control leg costs.
`MLXFAST_BASELINE_CALIBRATION` names the file.

**THE FILE IS A HEALTH BAND. IT IS NEVER THE DENOMINATOR.** The benchmarker
compares the measured control leg against the band. It stops the run by name
when the leg falls outside the band, and it seals no score. A stale calibration
file can stop a run. It can never move a score.

An operator writes the file on the box:

```bash
tools/calibrate-box.sh "<runner name>" /path/to/baseline-calibration.json
```

The command takes the box GPU lock. It then runs the serial-control leg four
times under the full official methodology: the cool gate before each pass, one
resident worker for each pass, and the same live golden the ranked run scores
over. The file it writes carries the values only:

```json
{
  "version": 1,
  "track_id": "bonsai2-27b-mlx-v1",
  "box": "<the runner name>",
  "reference_commit": "<the fixture's baseline_reference_commit>",
  "prompt": "<the fixture's live_golden>",
  "passes": 4,
  "prefill_seconds_per_token_mean": 0.0,
  "decode_seconds_per_token_mean": 0.0,
  "prefill_cv": 0.0,
  "decode_cv": 0.0,
  "prefill_band_low": 0.95,
  "prefill_band_high": 1.05,
  "decode_band_low": 0.98,
  "decode_band_high": 1.02,
  "captured_at": "<ISO 8601 UTC>",
  "benchd_source_commit": "<40 hex>"
}
```

The means and the coefficients of variation above are placeholders. Only a
measured box can supply them, and this track has no calibrated box yet.

The calibrator writes no file when the coefficient of variation is more than
1 percent on either axis. A box that cannot repeat itself has no band worth
recording.

`tools/ranked-box-preflight.sh` refuses the run before any measurement when:

- either variable is absent;
- the workspace is not a git checkout at `baseline_reference_commit`;
- the workspace has no staged worker, no `mlx.metallib`, no fingerprint sidecar,
  or no transformed weights of its own;
- the calibration file does not parse, or its `version` is not 1;
- its `track_id` is not this track;
- its `box` is not this runner's `RUNNER_NAME`;
- its `reference_commit` is not the fixture's `baseline_reference_commit`;
- its `captured_at` is not later than the reference commit's date;
- any numeric value is not finite and positive;
- a band does not straddle 1 (`low < 1 < high`).

Each refusal names the failing thing.

**THE SCORED SHAPE IS SINGLE-STREAM.** The fixture pins `scored_batch_size` 1.
Each leg runs one stream, and the engine agrees: the runner declares
single-stream regimes only. A scored run times the ONE prompt `live_golden`
names, on both legs.

Where a run times a pool of prompts rather than one, `aggregate` is the
**per-prompt sum**. Run each pool prompt in its own single-stream window and
add the elapsed times together. Do this for prefill and for decode separately.
Do it on the baseline leg and on the candidate leg, over the same prompts.

One ranked run measures the pairs the fixture's `official_pairs` declares, and
scores ONE of them. Each pair has its own composite, from its own control leg.
The run scores the pair whose composite is the lower median over the pairs:
the middle pair on an odd count, the lower of the two central pairs on an even
count. Pairs are never averaged. Every enforced figure in the artifact is that
one pair's, and the other pairs stay in `metrics.paired_legs` as measured.

The fixture carries the formula as `scoring_semantics.formula`, and
`benchmark.json` carries the same string as `scoring.formula`.

`scoring.mode` is `bonsai2-native-mtp-paired-composite`. It names the
measurement methodology, not the formula.

### 5.1.1 Where the prefill window is, and why you cannot move work out of it

**THE VERBS DO NOT CHANGE.** The single-stream pair is
`free_decode_begin` followed by `free_decode_run`. There is no new message and
no new field. The benchmarker splits its OWN parent clock at the verb boundary:

| Window | From | To |
|---|---|---|
| prefill | `free_decode_begin` sent | the validated `seed_token` comes back |
| decode | there | `free_decode_run(N)` returns |

`elapsed = prefill + decode`, and `seconds_per_token` is unchanged. Both legs
are bracketed identically.

**WHAT THE ENGINE OWES, and it is not optional:**

1. `free_decode_begin` runs the FULL seed prefill -- the golden's
   `decode_seed_tokens`, all 512 of them, with the requested spec resolved --
   and replies only after that work has COMPLETED. The reply is the seed token
   (the greedy argmax after the whole seed) and the echoed `effective_spec`.
2. `free_decode_run` does NOT prefill and does not re-run any part of the seed.
   It decodes from the state `begin` left.
3. Nothing prefills before `free_decode_begin` arrives.
4. The hello advertises `free_run_decode` only.
5. Units are unchanged.

**MOVING PREFILL WORK INTO THE RUN IS NOT AN OPTIMISATION. IT IS A SCORING
DEFECT.** Deferred seed work does not disappear: it leaves the prefill window
and lands in the decode window. The whole window is unchanged, so `elapsed` and
`seconds_per_token` look identical -- but the composite weights the two windows
0.25 and 0.75, so shrinking prefill and growing decode by the same amount MOVES
THE COMPOSITE, and it moves it against you. The same applies in reverse to a
leg that did decode work early.

This is engine-side and unobservable on the wire. Nothing on the wire can catch
it, so it is a rule of the contract rather than a gate, and the static review
reads a change against it.

**NO SCORED RUN IS POSSIBLE ON THIS TRACK TODAY**, and section 11 states why.
The short form is the arm state and the goldens: `official_scoring_enabled` is
`false` and no organizer tape exists.

### 5.2 The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 512 |
| Checked decode steps | 128 |
| Golden shape | 512 `prompt_tokens` and 129 `expected_tokens` |
| Streams per window | 1 |
| Timed prompts per leg | 1 (the fixture's `live_golden`) |
| Legs per ranked job | 2 (serial control, then candidate) |

The two legs run one after the other, in one job. Each leg loads the weights
once, and each leg gets its own worker residency. The unmeasured warm-up prefill
pass stays at 1 pass on MLX, and it applies to both legs in the same way.

`MLXFastConstants.correctnessPromptTokens`, `benchmarkPrefillPromptTokens`, and
`benchmarkDecodeSeedTokens` all equal 512. `benchmarkDecodeSteps` is 128.

Every timed leg runs on a cool, quiescent box. The benchmarker holds each timed
phase of each leg behind two fixed gates, in this order: the quiescence gate (the
1-minute load average is below 2.0 and GPU utilization is below 10 %, polled
every 15 s, refused after 900 s) and the cool-down gate (GPU at or below 40 C,
refused after 900 s). Calibration passes run behind the same two gates. The job
refuses to measure at all when the box has no GPU temperature reader, or when
that reader returns a frozen or implausible value. Only pairs accepted under
both gates feed the composite.

### 5.3 The parameters

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

The fixture's own floors are `decode_speedup_floor` and
`prefill_speedup_floor`, both 0.95. They gate the two gains. The manifest's
`decodeSpeedupFloor` states the same 0.95, and `pairsPerCohort` states the
fixture's `official_pairs`. The benchmarker reads the fixture; the manifest
block restates it, and `tools/lint-benchmark-manifest.py` keeps the two equal.

There is no sweep and no per-run choice of width. A width the benchmarker has
not certified has no series tag, and the benchmarker refuses that width rather
than run it.

The ranked run scores one pair, the lower median over the `official_pairs`
pairs by composite (section 5.1). No mean over pairs enters the published
number.

### 5.4 Token fidelity

The benchmarker applies a per-stream token-tolerance gate with a **10%
budget**.

This track does not require token-for-token equality with the serial
trajectory. A speculative round verifies a whole draft chain in one target
forward, and a multi-row forward rounds differently from a single-row one, so
it can diverge from the serial forward at a near-tie argmax. The gate prices
that divergence against the 10% budget. The gate accepts similar output. It
does not certify lossless output.

### 5.5 Arming

`fixtures/bonsai2_27b_mlx_v1_track.json` sets `official_scoring_enabled` to
`false`. That flag is the SINGLE authority on this track's arm state, and it is
load-bearing. The pinned benchmarker reads it from the `--contract` fixture. It
refuses to seal an official scoring artifact while the flag is `false`. It also
refuses while the flag is absent, because it treats an absent flag as unarmed
rather than armed. No submission can publish an official score until that flag
flips.

The benchmarker, not the engine, produces the composite. It computes the
composite from benchd's own parent-clocked prefill and decode windows, summed
over the accepted pairs, at the certified exponent pair. No engine-reported
value feeds it, and it does not depend on per-stream instrumentation. Each
record seals exactly one of `composite` and `composite_absent_reason`. A
composite is absent only when the record accepted no pair, or when a window is
degenerate, and the reason names which.

What is missing is the arm state and the goldens, not this repository's score
path. `tools/bonsai2-27b-measure-and-score.sh` refuses with a non-zero
exit rather than emit a score. It does not substitute a diagnostic for the
ruled composite formula. Refuse, not degrade, is the standing posture for this
track. The `kv_backend` check and the byte-budget check use it too.

The timed prompt pool is EMPTY. `timed_prompt_pool` holds no entry,
`live_golden` is the empty string, `live_golden_speculative` holds no per-depth
oracle, and `hidden_correctness_golden` carries the pending sentinel
`BONSAI2-27B-MLX-V1-PENDING-ORGANIZER`. An empty pool is legal only
while the track is unarmed: `tools/lint-benchmark-manifest.py` requires 8
pinned pool prompts before `official_scoring_enabled` may be `true`.
`tools/ranked-box-preflight.sh` refuses a contract that still carries that
sentinel. The sentinel is matched exactly; it is never a prefix test.

### 5.6 Which goldens you can hold

| Object | Where it lives | Can you have it? |
|---|---|---|
| The public captures (`public_captures`) | R2, at the `r2_path` keys the fixture pins. `tools/fetch-goldens.sh --public` fetches them into `correctness_prompts/bonsai2-27b-mlx-v1/`. | **Yes.** They are pending today. See section 11.3. |
| `timed_prompt_pool[]` tapes | R2, at the `r2_path` keys the fixture pins. The ranked box stages them out of band into `MLXFAST_QWEN38_GOLDEN_DIR`. | **No.** They are organizer material and they are never in git. |
| `live_golden_speculative{}` per-depth oracles | The same: R2 keys, staged on the box. | **No.** Same material, same handling. |
| `hidden_correctness_golden` | The live golden, pinned by digest only. It is one of the staged files. | **No.** It is the token-fidelity oracle and it stays on the box. |
| The reference tree (`MLXFAST_BASELINE_WORKSPACE`) | Built on the ranked box at `baseline_reference_commit`. | **No.** It is the serial-control leg's engine. Its commit is public: the fixture names it. |

`MLXFAST_QWEN38_GOLDEN_DIR` keeps its Qwen spelling because it is the fleet's
contract with the box's runner service, not this track's own name. The same
holds for `MLXFAST_QWEN38_R2_DOWNLOADER`, the override for the R2 request
signer.

`tools/fetch-goldens.sh` is the organizer-side, pin-verified fetcher for R2
objects. It reads the R2 base from the environment variable
`R2_BUCKET_ENDPOINT` only. That value is secret-tier and is absent from this
repository. The script verifies the byte count first, then the sha256, and
deletes the file on either mismatch. It refuses to fetch anything the contract
declares hidden, and that guard fails closed when it cannot read the contract.

> **NOTE — this repository pins no golden for that tool to fetch yet.**
> The pool is empty, so `--all` has nothing to stage today. The public
> captures carry the pending sentinel, so `--public` refuses today.

The organizer stages the whole pinned set on a ranked box with the same tool.
`--all` reads the fixture, fetches every tape and every per-depth oracle, and
verifies each one against its `{sha256, bytes}` pin. It signs the requests with
the signer vendored at `tools/download-r2-object.sh`, so it needs R2 credentials
and refuses without them. A file that already matches its pin is left alone, so
the command is safe to re-run:

```bash
R2_BUCKET_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
  tools/fetch-goldens.sh --all --out "$MLXFAST_QWEN38_GOLDEN_DIR"
tools/ranked-box-preflight.sh
```

The preflight then verifies the staged directory against the fixture again and
refuses an extra `*.json` in it.

## 6. Running the benchmark

`benchmarkCommand` targets `benchd iterate --mode official` through
`tools/bonsai2-27b-measure-and-score.sh`. That script is trusted-side
tooling. It is not an editable path, so a submission cannot rewrite the
measurement pipeline from inside its own archive.

The wrapped invocation is:

```text
benchd iterate --mode official \
  --engine .build/release/bench-worker \
  --weights <transformed weights> \
  --golden $MLXFAST_QWEN38_GOLDEN_DIR/<live golden>.golden.json \
  --contract fixtures/bonsai2_27b_mlx_v1_track.json \
  --baseline-workspace $MLXFAST_BASELINE_WORKSPACE \
  --baseline-calibration $MLXFAST_BASELINE_CALIBRATION \
  [--box <this box>] [--mtp-depth N | --candidate-spec <json>] \
  --score-path score.json
```

`--engine` and `--weights` are RELATIVE to the checkout root. The benchmarker
re-roots both under `--baseline-workspace` to find the serial-control leg's own
worker and its own transformed weights. The script refuses an engine or a
weights directory outside the checkout: such a path has no relative form, and
the control leg would then run the candidate's own binary or read the
candidate's own transform.

`--mtp-depth` carries the depth the tree declares when the declared decoder is
the MTP head. A DFlash 2 depth travels as the explicit candidate spec,
`--candidate-spec '{"mode":"dflash","dflash":{"depth":N}}'`, because the
benchmarker's depth convenience flag builds an MTP spec only.
`tools/spec-declaration.sh decoder` and `tools/spec-declaration.sh draft-len`
derive both values from `mtp-head.manifest.json`, and the script refuses rather
than run a serial leg against a speculative declaration when the resolved
benchd has neither flag.

`--box` is passed only on a hand run. On a runner the benchmarker reads
`RUNNER_NAME` itself and lets it win over the flag, so passing the flag there
would be argv that cannot matter. A hand run has no `RUNNER_NAME`, and the value
then comes from the calibration file's own `box`.

The measure script refuses by name when `MLXFAST_BASELINE_WORKSPACE` or
`MLXFAST_BASELINE_CALIBRATION` is absent on a real run. `--preflight-only`
requires neither: a participant runs it off the box.

benchd seals `score.json` itself, in the `{score, metrics}` shape the scorer
reads. This script writes no score and converts nothing.

**WHERE THE COMPOSITE LIVES ON THIS TRACK.** A single-stream run has no cohort
record, so benchd seals the composite ONE LEVEL UP: `results.json` carries
`composite` (`{composite_score, composite_speedup_floor,
composite_speedup_floor_met, decode_gain, prefill_gain}`) beside
`composite_scored_exponents`, and exactly one of `composite` /
`composite_absent_reason` is present. The published score is
`composite.composite_score`.

The decode-only median is NOT the score on this track. It is a different
formula -- decode only, no prefill component, no exponents -- and the emitter
refuses a single-stream record that seals no composite rather than publishing
the median in its place. The median is forwarded in `metrics` as a diagnostic.

Two drift tripwires run at that seam, because benchd reports a floor and an
exponent pair but wires neither to an exit code: the emitter REFUSES a run
whose `composite_speedup_floor_met` is false, and refuses a run whose sealed
`composite_scored_exponents` differ from `benchmark.json`
`scoring.scoredExponents`.

`preSubmitCommand` runs
`./tools/bonsai2-27b-measure-and-score.sh --preflight-only`.
That runs the arm gate and, when the live golden is staged, the integrity pin.
It exits without measuring, and it needs no reference tree.

The ranked pipeline is `.github/workflows/benchmark.yml`, which `benchmark.json`
`runner.workflow` names. It triggers on `workflow_dispatch` only. Its hosted
surface-check job gates its ranked job, which runs on the self-hosted labels
`[self-hosted, macOS, bonsai2-27b-mlx-v1]` — the third label is the track id.
The ranked job holds no credential. The organizer stages the timed-pool tapes
onto the box out of band, from R2; they are never in the checkout. Before
`./setup.sh` runs, `tools/ranked-box-preflight.sh` verifies each tape against
this track's `{sha256, bytes}` pins. One ranked run occupies the box at a time.
A second dispatch queues rather than cancelling the first.

**ONE RESIDENT WORKER PER LEG, AND THE BENCHMARKER BOOTS IT.** A ranked job has
two legs on two trees, so one resident cannot serve both: the reference leg's
weights are not the candidate's. The benchmarker knows where a leg begins and
ends, so for each leg it calls that leg's OWN copy of `tools/resident-up.sh`:

```text
tools/resident-up.sh --boot --spec <serial|mtp|dflash> --draft-len <N> --socket-out <file>
tools/resident-up.sh --stop --socket <path>
```

`--boot` loads that tree's own `.build/release/bench-worker` and that tree's own
`weights/`, waits for a healthy hello, writes the socket path as the first line
of the `--socket-out` file, and exits 0 with the resident still running. A
`<socket>.pid` sidecar and a `<socket>.ready` marker sit beside the socket, and
`--stop` ends the resident and removes all three. Every per-phase
`bench-worker runtime-worker` the benchmarker starts attaches to that leg's
socket instead of loading the checkpoint again.

`--spec` is authoritative over what the leg DOES. It is the label the leg
records, so a reference tree that declares a draft depth still boots a SERIAL
control leg when the benchmarker says serial, and the worker takes the decoder
and the depth from the wire per request.

`--spec` does NOT choose the drafter. The boot reads the leg's OWN
`mtp-head.manifest.json`, through `tools/spec-declaration.sh`, for that one
decision: which of the two staged drafters this leg's resident holds. Each leg
holds the drafter its own tree declares, and nothing crosses between the two
trees.

`tools/bonsai2-27b-measure-and-score.sh` boots no resident and exports no
socket. What it still owns is the window: it takes the box GPU lock
`/tmp/mtplx-gpu-exclusive.lock` and holds it for the whole measurement, because
a resident holds about 18.5 GB of unified memory whoever booted it and the box
needs exactly one loader. `tools/resident-up.sh` refuses to boot when nobody
holds that lock, so every per-leg boot happens inside that window.

**A `BENCH_WORKER_RESIDENT_SOCKET` IN THE ENVIRONMENT IS REFUSED.** It names one
already-loaded resident, so both legs would attach to it and the serial-control
leg would run on the candidate's weights. Both
`tools/ranked-box-preflight.sh` and the measure script refuse an inherited
socket by name.

None of this changes what is scored. It removes repeated loads, not measured
work.

The wrapper form of `tools/resident-up.sh` (`--weights <dir> -- <command>`) is
kept for LOCAL, UNSCORED use. The ranked path has no caller for it.

`setupCommand` is `./tools/fetch-benchd.sh && ./setup.sh`. It chains no drafter
stager of its own, because `./setup.sh` runs both: `./setup-mtp-head.sh` for
the MTP head, then `./setup-dflash-drafter.sh` for the DFlash 2 drafter. The
checked-in `mtp-head.manifest.json` declares `"source": "pinned"`, and each
stager points the one verified downloader at its own decoder's pin file.

## 7. The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ `3f926b415992eaa2ae9dd7b573706494d6bbf787` |
| Target manifest | `fixtures/reference_bonsai2_27b_2bit.sha256` |
| MTP head | `EigenLabs/Qwen3.8-27B-MTP-4bit` @ `329261c5e0b3f9c233485e682cb3b67b88c20a55` |
| DFlash 2 drafter | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Engine fork upstream base | `859f4d97e0d1ae9cce0883b044f764283841edfb` (+ fork `main` `0607011` merged; engine-local changes on top) |

`fixtures/reference_bonsai2_27b_2bit.sha256` pins 19 files totalling
8,608,721,032 bytes, of which one is the single safetensors shard. The pack
holds 2390 raw tensors, 8,595,174,880 bytes of them.
`fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256` pins the head's 4 files,
238,942,843 bytes.
`fixtures/reference_bonsai2_27b_dflash2_drafter.sha256` pins the DFlash 2
drafter's 3 files, 3,848,824,532 bytes, of which one is the single safetensors
shard holding 81 BF16 tensors and 3,848,808,960 tensor bytes. Its
`.gitattributes` and its figure are deliberately not pinned, for the reason the
head's are not: neither is engine-loaded.
`Sources/MLXFastCore/Constants.swift` mirrors the same repository and revision
pin. `fixtures/bonsai2_27b_config.json` is the pack's own `config.json`, and
`fixtures/bonsai2_27b_tensor_inventory.json` is the pinned tensor inventory
that `Sources/MLXFastTransform/PrismHadamardCheckpointValidation.swift` checks
the headers against.

The checkpoint's `.gitattributes`, `LICENSE` and image files are deliberately
not pinned. None is engine-loaded, and `setup.sh` derives what it downloads
from the manifest.

The model repository is public and downloads without a token. There is no
organizer-hosted mirror for this checkpoint, so
`MLXFAST_REFERENCE_FALLBACK_BASE_URL` is empty by default.

Participants never supply the target weights. Substituting or re-deriving the
target is a failure.

### 7.1 The decoder and the draft depth

You select the decoder and the draft depth. The depth is not pinned at 1.

The declaration sets both, and `mtp-head.manifest.json` is an editable path, so
both are free levers:

```json
"spec": { "decoder": "mtp",    "enabled": true, "num_speculative_tokens": 3 }
```

```json
"spec": { "decoder": "dflash", "enabled": true, "num_speculative_tokens": 7 }
```

Each decoder has its own permitted range, and the two are independent.

The MTP head permits 1 to 7. Three layers agree on that number:

| Layer | Where |
|---|---|
| The contract | `mtp_head.permitted_draft_depths` is `[1 … 7]` |
| The protocol envelope | `protocol.mtp_envelope_constants_from_darkbloom.max_draft_tokens` is 7 |
| The engine | the runner declares `depth: 1 ... CBv2MTPConfig.testedMaxDraftTokens`, which is 7 |

The DFlash 2 drafter permits 1 to 16. Two layers agree on that number:

| Layer | Where |
|---|---|
| The contract | `dflash_drafter.permitted_draft_depths` is `[1 … 16]` |
| The protocol envelope | `protocol.dflash_envelope_constants.max_draft_tokens` is 16 |

A DFlash 2 depth is the drafter's block size minus one. Section 4.5 states what
that means for acceptance.

`tools/spec-declaration.sh` is the single trusted reader of the declaration. It
refuses a decoder the track does not declare, and a depth that is not in the
DECLARED decoder's list, so a serve can never boot a decoder or a depth the
contract does not permit. With no `spec` block, `enabled: false`, or `0`, the
declaration resolves to depth 0 and the leg runs serial, whatever the
decoder.

The same envelope pins the speculative batch at 1 and the verification mode at
`rectangular_exact`. The arm is single-stream on both sides: the runner's
assistant declares `maximumSpeculativeBatch == 1`.

Every run seals the depth that operated. Read `effective_spec` for the depth the
run declared. Read `effective_mean_draft_len` for the draft length that the run
realized. The two can differ: a run can declare depth 2 and realize a mean draft
length near 1.

## 8. Prohibited techniques

A submission that uses any of these fails the static review.

- A cache or memo keyed on a request's input tokens whose only possible hit is
  the harness repeating one identical computation. Bit-identical output does
  not make it legitimate. The benchmark measures single-pass inference. An
  optimization must save work that recurs in single-pass production inference.
- Hardcoded hidden prompts, hidden token identifiers, or answers.
- Timing shortcuts, protocol injection, network access, and filesystem
  exfiltration.
- Any change outside `editablePaths`.

Input-independent caching stays legal. This covers weights, dequantized
tensors, and RoPE or mask tables keyed on shapes and offsets. Within-request KV
reuse stays legal.

Keep every change prompt-independent and model-general. The hidden prompts
differ from the public fixtures.

## 9. Submitting

Use the Yukon CLI for every account operation and every submission operation.
`README.md` holds the commands.

A submission archive packages only `editablePaths`. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first,
and no local run blocks the upload.

The ranked run on the official runner is the gate that ranks a submission.

## 10. License

The pinned pack carries Apache-2.0. The terms ship with the pack at its pinned
revision.

The pack is a 2-bit MLX build of `Qwen/Qwen3.8-27B`.

This repository distributes no model weights.

## 11. What is not in place yet

Read this section before you conclude that something is broken.

### 11.1 The track is not armed

Section 5.5 states the arm state. `official_scoring_enabled` is `false`, the
timed prompt pool and the live golden are empty, and the hidden correctness
oracle is the pending sentinel. No runner advertises the ranked label set
`[self-hosted, macOS, bonsai2-27b-mlx-v1]`, and the box is not staged.

THE CHANNEL RESOLVES FROM `main`. benchd is published from the bench
repository's `main` branch; there is no per-track bench branch (ruled
2026-09-07). `./tools/fetch-benchd.sh` reads `dist/benchd.manifest.json` at the
tip of branch `BENCHD_BRANCH` (default `main`), checks the manifest's `branch`
against that channel, and installs the binary only when it matches the `sha256`
and `bytes` the manifest names. The script then prints the identity it resolved
and keeps the manifest beside the binary in `benchd-bin/`.

THAT MANIFEST IS THE SOURCE OF TRUTH for which `benchd` measures your run,
and it moves when the organizer republishes dist. This document therefore names
no `{source_commit, sha256, bytes}` triple as current: run the script and read
the identity line.

The channel host is the public bench repository, so `./tools/fetch-benchd.sh`
needs no token. An air-gapped box can pass a verified pair through
`BENCHD_DIST_LOCAL`.

### 11.2 The model port HAS landed, as SOURCE

The engine constructs and gates `prism_hadamard_qwen35`. The geometry in
`Sources/MLXFastCore/Constants.swift`, the checkpoint validator in
`Sources/MLXFastTransform` and this contract's `target.*` block are all this
target's, and they move as ONE SET: a gate holding some fields of one model and
some of another rejects every checkpoint and explains none of them.

NOTHING IN THE PORT HAS BEEN COMPILED OR RUN. Build it and run the local test
on a box before you trust it.

The model, the runner and the MTP assistant are in the vendored
`Vendor/mlx-swift-lm` tree. The Bonsai 2 and Qwen 3.5 model files in it are
editable paths, and so is `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift`. The
runner is not; see section 3.
`bench-worker` selects the runner from the checkpoint's own `model_type`, and
this track passes it no `--resource` argument: the model loads no out-of-band
row source.

`docs/bonsai2-27b-port-notes.md` holds what the port changed, what is
verified and what is still open.

### 11.3 The public captures

The public captures are the goldens that `--local-iterate` and
`--local-submit` check against. They are R2 objects under
`correctness_prompts/bonsai2-27b-mlx-v1/`. No golden, capture or prompt file is
in git: `tools/lint-benchmark-manifest.py` check 5a3 refuses a tracked file
under `correctness_prompts/`.

The fixture pins each capture in `public_captures` as `{r2_path, sha256,
bytes}`. `local_iterate` is the short capture and `local_submit` is the long
one. The organizer records both on the track's box, in one recording, so the
short capture is the long capture truncated. Both carry the pending sentinel
today.

Fetch the captures with this command:

```bash
R2_BUCKET_ENDPOINT=... tools/fetch-goldens.sh --public
```

The command writes each capture to its `r2_path` under the repository root. It
verifies the byte count first, then the sha256. It refuses while a pin carries
the pending sentinel. `R2_BUCKET_ENDPOINT` is secret-tier: get it from the
organizer and keep it in your `.env`. The R2 credentials are optional for this
mode.

The ranked preflight refuses a contract that carries any pending sentinel, the
public captures included. The track therefore does not open while participants
have no local golden.

**A GOLDEN MUST CARRY `model_provenance`.** The block names the repository and
the revision of the pinned model. `loadQwenGoldenFixture` is the loader that
every golden consumer in `Sources/` calls, and it refuses a golden that carries
no block. The error message names `model_provenance`. The model-agnostic
`loadGoldenFixture` keeps the reference schema, which has no such key.

### 11.4 The batched cohort path is not part of this track

The scored shape is single-stream, and the engine declares nothing else. The
runner's MTP assistant declares `maximumSpeculativeBatch == 1`, so the runner
declares only the two single-stream regimes: free-run and teacher-forced. The
benchmarker therefore refuses a batched request at its pre-measurement
capability check rather than after it has spent box time.

That agrees with the fixture, which pins `scored_batch_size` 1 and
`scoring.mode` `bonsai2-native-mtp-paired-composite`. Section 5 describes the
single-stream paired series that follows.

### 11.5 The verify runs at the draft depth

A speculative round verifies its whole draft chain in ONE target forward. The
verify width is the resolved draft depth on the mtp leg; the serial control
still runs one token at a time. The envelope pins the verification mode
`rectangular_exact`.

**WHAT THE WIDE VERIFY RESTS ON.** Not bit-identity -- MLX dispatches a
different kernel at one row than at several, by design, so the logits differ in
their last bits. It rests on ARGMAX AGREEMENT: the wide forward picking the
same tokens. A committed token is correct when it matches what that forward
says, not when it matches what a one-token-at-a-time decode would have said.
Section 5.4 is the gate that prices any resulting difference in emitted tokens.

**WHAT THIS MEANS FOR YOU.** The arm is not strictly more work than serial for
the same output: a round pays one target forward for its whole chain instead of
one per committed token. It still pays the head forwards that proposed the
chain. Whether that becomes a speedup on the ranked box is a measurement, not a
promise, and it depends on the drafter -- an accepted draft is what buys the
forward back. Nothing on this track has been measured yet.

### 11.6 The KV backend is contiguous, and paged is not built

The model supports paged KV and sets `requiresNativePagedKV`, but
`RunnerEngineAssembly`'s paged branch builds neither the segment-size nor the
layer-dtype declaration that `EngineV2` demands. The runner therefore declares
the `contiguous` backend only. Declaring the paged one today would halt the
process instead of refusing the build, so the narrow declaration is the correct
state and not an oversight.
