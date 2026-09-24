# Ternary Bonsai 2 27B — port notes

This document is the engineering record of the port. It says what this
repository inherited, what the port changed, what is verified, and what is
still open.

It is a record, not a contract. `benchmark.json` and
`fixtures/bonsai2_27b_mlx_v1_track.json` hold the values.
`docs/participant-contract.md` explains them. Where this document and those
files disagree, those files win.

## 1. What this repository is

This repository is the engine for track `bonsai2-27b-mlx-v1`. The target model
is `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` at revision
`3f926b415992eaa2ae9dd7b573706494d6bbf787`. The manifest name is
`mlxfast-bonsai2-27b`.

## 2. What the track inherited

The tree is a seed of the Nemotron 3.5 Lightning 30B A3B MLX engine
(`648495f`, tree-identical to `mlxfast-nemotron35-30b-a3b-engine-dev @
d6e2d399`). `tools/new-track.sh` then stamped the new identity into it
(`5f4b611`).

These parts carried over unchanged:

- the submission-restriction stack: the byte-budget enforcer, the overlay, the
  modifiable-surface gate and `tools/test-submission-security.sh`;
- the measurement topology: a paired run of a serial-control leg and a
  candidate leg on one box, one resident worker per leg, the box GPU lock and
  the cool-down gate;
- the benchd channel discipline: a prebuilt binary resolved and pin-verified by
  `tools/fetch-benchd.sh`, never a source dependency;
- the golden discipline: goldens are R2 objects staged on the track's own box,
  never in git;
- the scoring shape: single-stream, paired, `composite =
  prefill_gain^0.25 * decode_gain^0.75` over a lower-median of 3 pairs;
- the byte budget: `maxTotalBytes` 7,949,663, unchanged by this port.

## 3. The model

### 3.1 The pack

The target is a 2-bit MLX pack of a dense Qwen3.8-27B backbone. It is not the
native Qwen4 Flash-Next model.

| Fact | Value |
|---|---|
| `model_type` | `prism_hadamard_qwen35`, schema 2 |
| `base_model_type` | `qwen3_5`; `text_config.model_type` is `qwen3_5_text` |
| Layers | 64. Every fourth layer is full attention: 16 attention, 48 linear |
| Hidden / intermediate | 5120 / 17408 |
| Attention | 24 query heads, 4 key/value heads, head dim 256, output gate on |
| Linear attention | 16 key heads, 48 value heads, head dim 128, conv kernel 4 |
| Vocabulary | 248320, untied head |
| Rotary | partial factor 0.25 over theta 1e7, mRoPE sections 11/11/10 |
| Quantization | affine, 2 bits, group 128, FP16 scales and biases |
| Tensors | 2390: 2057 under `language_model.`, 333 in the vision tower |

The pack ships one `model.safetensors` and NO `model.safetensors.index.json`.
The index is optional for a single-shard checkpoint, and both the transform and
`setup.sh` already allowed that case.

### 3.2 The signed Hadamard transform

402 of the pack's linear modules are packed. Each one carries four tensors:
`weight` (U32 codes), `scales` and `biases` (FP16), and `signs` (FP32). The
pack folds a normalized block Walsh-Hadamard transform of width 1024 into each
packed weight, and `hadamard.json` is the manifest for it.

At load, `PrismHadamardCheckpoint` replaces each declared module with a
`HadamardQuantizedLinear` or, for `model.embed_tokens`, a
`HadamardQuantizedEmbedding`. The forward transform applies to the INPUT of
every packed linear. The inverse transform applies to the OUTPUT of the
embedding. Both run in FP32 and restore the incoming dtype.

THE CONSEQUENCE FOR ANY CODE THAT TOUCHES A PACKED MODULE: call the module.
A path that reads the module's raw `.weight` skips the transform and returns
wrong numbers while it still type-checks and still runs.
`HadamardQuantizedLinear` IS a `QuantizedLinear`, so a `as?` cast to that type
succeeds. `HadamardQuantizedEmbedding` is NOT a `QuantizedEmbedding`, so a
cast to that type fails and a fall-through reads packed integers as floats.
Section 6 lists the one place in the tree where both of those happened.

### 3.3 The vision tower

The pack declares a vision tower and ships its 333 tensors. This track serves
text only. The transform drops the `vision_tower.` namespace,
`Qwen35Runner` declares `multimodal: false` so the LLM factory resolves the
model type, and `PrismHadamardQwen35TextModel` filters the same prefix at
load. The tower is never built.

### 3.4 The MTP head

The pack declares `mtp_num_hidden_layers: 0` and carries no `mtp.*` tensor.
The transform refuses a pack that carries one.

David ruled MTP enabled for this track. The head is therefore a SEPARATE
published export, used unmodified: `EigenLabs/Qwen3.8-27B-MTP-4bit` at
`329261c5e0b3f9c233485e682cb3b67b88c20a55`. It is `qwen3_5_mtp`, affine 4-bit
at group 64, one layer, 31 tensors, 238,930,944 tensor bytes.

The head owns no embedding and no output projection. It borrows the target's,
which on this pack are packed Hadamard modules.
`Qwen35InlineMTPAssistant.load` checks the head's `text_config` against the
loaded target on hidden size, vocabulary, attention heads, key/value heads,
head dim and expert counts. All of those agree, because both sides are
Qwen3.8-27B geometry.

`fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256` pins the head's four
files. `./setup-mtp-head.sh` stages it, and `./setup.sh` runs that script.
`tools/resident-up.sh` passes `--drafter` to the worker on BOTH legs, so the
serial control leg and the candidate leg load the same bytes and differ only in
the decoder benchd selects.

### 3.5 The DFlash 2 drafter

The track has a SECOND declarable speculative decoder. It is a separate
published export too, and it is used unmodified: `z-lab/Qwen3.8-27B-DFlash2` at
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, Apache-2.0. It is 81 BF16 tensors
in one shard, 3,848,808,960 tensor bytes.
`fixtures/reference_bonsai2_27b_dflash2_drafter.sha256` pins its 3 files at
3,848,824,532 bytes. `./setup-dflash-drafter.sh` stages it into
`reference_weights/Qwen3.8-27B-DFlash2`, and `./setup.sh` runs that script
after `./setup-mtp-head.sh`.

The drafter is five sliding-attention layers of sliding window 2048, with 32
query heads and 8 key/value heads of dimension 128, hidden size 5120,
intermediate size 17408, vocabulary 248320 and rope base 1e7. It owns no
embedding table and no output projection. It binds the TARGET's, which on this
pack are packed Hadamard modules, so the code calls the modules and never reads
a raw `.weight`. Section 6.1 is the defect that rule exists to prevent.

**THE DECLARATION.** `mtp-head.manifest.json` `spec` gained `decoder`, which is
`"mtp"` or `"dflash"`. An absent key reads as `"mtp"`, so an older declaration
keeps its meaning. The MTP head's depths stay 1 to 7 and its 2 GiB declaration
cap is unchanged. The drafter's depths are 1 to 16, which is David's limit, and
its declaration cap is its own, 4 GiB, because one cap cannot bound a 239 MB
export and a 3.85 GB one. `max_bytes` may lower the cap of the declared decoder
and may not raise it.

**THE PORT.** `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/DFlash2Draft.swift`
is a port of `dflash/model_mlx.py` of `z-lab/dflash` at `07ebd93`:
`DFlashAttention`, `GroupedDynamicCausalConv`, `DFlash2DecoderLayer`,
`CandidateSelector` and `DFlash2DraftModel`. This track is greedy, so the
reference's rejection-sampling path is not ported.

**THE SHAPE IS A BLOCK, NOT A CHAIN.** One `propose` call takes the last
committed token followed by `depth` mask tokens, runs the five layers once, and
returns `depth` draft tokens. A depth is therefore the block size minus one.
The block attends to itself with no causal mask, because the drafter's config
sets `is_causal` false. Context reaches the block only through attention: the
layers project the fused target hidden state into context keys and values and
cache those. The drafter was trained at block 8, which is depth 7. A larger
block is legal and simply accepts less.

**THE CONTEXT COMES FROM THE TARGET.** The drafter reads the OUTPUT hidden
state of five named target layers, `[5, 19, 33, 47, 61]` on this pack, and its
`fc` consumes them fused along the feature axis.
`Qwen35TextModelInner.cbv2Forward` is the one layer loop every CBv2 path takes,
so the tap sits there and covers the prefill forward and the rectangular verify
forward alike. The tap is held in `DFlash2TapSlot`, a plain class, because
`Module` reflection classifies a stored `MLXArray` as a PARAMETER, and an
unexpected parameter key would fail the loaded-target re-verification at the
top of a measured window.

**THE TWO DTYPE CROSSINGS ARE NAMED.** The Bonsai trunk runs its norms in FP32
and hands out FP32 activations. The drafter is BF16. Exactly two tensors cross:
the embedded block, and the fused target hidden state. `hiddenStates` casts
both, at the embedding and at the `fc` input. The fused hidden is cast once
more, and earlier, where the adapter takes the rows from the target
(`Qwen35DFlash2Assistant.append`). That earlier cast is the same one cast per
context row, and it also DETACHES the row from the whole prefill chunk it was
sliced out of, so a 2047-row window holds 2047 rows and not the prompt. The
`hiddenStates` cast then does nothing to an already-BF16 tensor. Everything
after that is the drafter's own dtype.

**THE ENGINE SEAM.** `CBv2MTPBlockDrafter` is the block verb, a sibling of the
chain verb and not an adapter over it. It REFINES
`CBv2MTPRequestStatefulDrafter`, because everything around the proposal is the
same object and the same lifecycle; only `proposeBlock` is new, and the three
chain verbs default to a refusal. A round calls `proposeBlock` ONCE and fills
all `depth` draft columns from it, so the depth costs one drafter forward
rather than `depth` of them.

Committed target rows reach the drafter through the two seams a
request-stateful drafter already has. `observeCommittedTarget` takes the
prompt's positions, chunk by chunk, and every plain decode position; that is
how PREFILL seeds the drafter, and the adapter keeps the newest
`sliding_window - 1` of them and starts its cache where those rows actually
are. `finalizeRound` takes a verified round's CONFIRMED columns, which is the
reference's `hidden = hidden[:, :accepted + 1, :]`. The drafter's cache is
trimmed to the row's committed length right after the proposal absorbs the
rows, which is where the reference trims; under the block protocol it normally
trims nothing.

The tap is armed by `RunnerEngineAssembly.makeEngine` for a build that
speculates and DISARMED for one that does not, so a serial leg in a process
that also served a DFlash leg pays nothing for a resident drafter.

The gated-delta-net state needs nothing new.
`Qwen35GatedDeltaNet.cbv2ForwardCaptured` commits by position and is
width-generic: widths 1 and 2 stack per position (`stageCaptured`), width 3 and
above run one recurrence over the whole window and replay the accepted prefix
(`stagePrefixReplay`), and the recurrent transaction bounds the position count
only below. A 17-wide window, which is depth 16 plus the anchor, is legal
there. Contiguous KV carries no speculative-span cap.

**THE DEPTH CEILING IS A DECODER FACT.** `CBv2MTPConfig` carries
`draftTokenCeiling`. It defaults to the chain's tested 7, so the MTP head's
clamp is unchanged, and only a `dflash` build passes the block ceiling of 16.
The runner declares the two ranges as separate constants for the same reason.

**THE EDITABLE SURFACE.** `DFlash2Draft.swift` is on the editable surface, for
the reason `Qwen35MTP.swift` is: it PROPOSES tokens, and the rule this track
draws is that code which proposes is editable while code that verifies,
measures or ledgers stays trusted. Its CBv2 adapter,
`Qwen35DFlash2Assistant.swift`, is editable too (David ruling 2026-09-20: the
source files of both decoders are editable). The MTP head's adapter is already
editable, because it is in `Qwen35MTP.swift`. The adapter only feeds and trims
the drafter; the engine's verification and its ledgers are in other files.

**RESIDENCY.** `tools/resident-up.sh` passes one `--drafter` directory to the
worker on both legs, and a `--boot` picks WHICH drafter from the leg's own
`mtp-head.manifest.json`, read through `tools/spec-declaration.sh`. `--spec`
cannot pick it; section 7 records why.
`tools/stage-baseline-workspace.sh` stages the drafter into the reference tree
as well, from `MLXFAST_DFLASH_DRAFTER_REFERENCE_DIR`, which costs 3.85 GB
there.

## 4. What the port changed

### 4.1 The vendored trees

`Vendor/mlx-swift` moves from `Layr-Labs/mlx-swift` `6d6796d7` to `70052b2`.
That range is two commits: `Source/MLXNN/Hadamard.swift`, which holds the
signed block transform and the two packed layer types, and an FP16 option on
the widening constant cache. The move is REQUIRED, not optional: the Bonsai
loader needs those three types and the previous core did not have them.

`Vendor/mlx-swift-lm` takes the Bonsai support from fork `main` `fd0eaac`
(pull requests 154 and 155). The vendored tree is cut from
`feat/qwen38-flash-next-runner` `859f4d9`, which is a different line of the
fork — it carries `Libraries/MLXRunners` and `bench-worker`, and fork `main`
does not — so the full range diff is 289 files and does not apply. The port
took the closed set the two pull requests need. Three parts were NOT taken:

- `PrismHadamardPrefillCarry.swift` needs `CBv2DeferredHostFill` and a paged
  write-fault accessor that this base does not have. It is an opt-in
  experiment, off unless `DARKBLOOM_BONSAI_PREFILL_CARRY_ASYNC` is set, so the
  base keeps its scheduling. The storage half of that change WAS taken: the
  packed prefill conv tail is copied instead of sliced, so a three-row carry
  stops holding the whole prefill chunk alive. That moves bytes and changes no
  bit.
- `Libraries/MLXVLM/Models/PrismHadamardQwen35.swift`, because the runner
  declares `multimodal: false` and the vision tower is never built.
- the `MLXLMServer` request-validation changes, which serve the fork's HTTP
  server. The scored path does not use it.

### 4.2 The contract and the fixtures

`fixtures/bonsai2_27b_mlx_v1_track.json` keeps the stamp's identity and its
model facts are re-authored from the pack's own config.json. A `hadamard`
sub-block records the transform, and the `mtp_head` block records a separate
pinned export instead of an embedded block.

| File | What it is |
|---|---|
| `fixtures/bonsai2_27b_config.json` | the pack's own `config.json`, verbatim |
| `fixtures/bonsai2_27b_tensor_inventory.json` | the pinned tensor inventory, built from the safetensors header |
| `fixtures/reference_bonsai2_27b_2bit.sha256` | the 19 pinned target files and their digests |
| `fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256` | the 4 pinned head files and their digests |

`golden_model_type` is `prism_hadamard_qwen35`, the id the weights declare.
benchd reads that string from the fixture, so benchd needs no change.

### 4.3 The harness

`NemotronHCheckpointValidation` is replaced by
`PrismHadamardCheckpointValidation`. It derives the whole 2,057-tensor
inventory from a pinned geometry, and it adds one check the Nemotron validator
had no reason to have: a packed module must ship its `.signs`, and that vector
must cover the whole contracted axis in whole 1024-wide blocks. A vector one
block wide would rotate the first block and leave the rest alone, which reads
as a subtly wrong model rather than a failure.

`Sources/MLXFastCore/Constants.swift` carries the new pin, the new geometry
and the new `requiredGoldenModelType`. The transform family is renamed to
`.prismHadamardQwen35`, selects `language_model.` and emits the source config
whole, because the packed loader reads that config as the artifact contract.

### 4.4 The editable surface

The ten `NemotronH*.swift` files leave `editablePaths`. Eleven take their
place: `PrismHadamardCheckpoint.swift`, `PrismHadamardQwen35.swift`, the nine
`Qwen35*.swift` tower and MTP files, and
`Vendor/mlx-swift/Source/MLXNN/Hadamard.swift`. That last one is in the other
vendored tree. It belongs on the surface for the same reason the Metal kernels
beside it do: it is arithmetic this track runs on every projection of every
layer.

94 entries become 95. At rest the surface is 4,019,431 bytes over 266 files
against an unchanged 7,949,663 cap. No cap moved.

## 5. The goldens

The track has none yet. No golden, capture or prompt file is in git. The
three Nemotron-era files that the seed carried (a 1024-token prompt and its
256-step and 1024-step captures) are removed: both loaders refused them, and
this track's seed is 512 tokens.

A golden is hardware-generated. The track's goldens must be recorded on this
track's own box, against this pack, A≡B double-generated, and published to R2
under `correctness_prompts/bonsai2-27b-mlx-v1/`. That covers the ranked
material (the 8 pool tapes, the per-depth oracles) and the two public captures
for local runs. The fixture pins the public captures in `public_captures`, and
`tools/fetch-goldens.sh --public` fetches them. The local public drift gate
cannot pass until they are recorded, and it should not.

### 5.1 The arming shape

The arming commit fills these fixture keys from the recorded files:
`timed_prompt_pool` (8 pins), `live_golden`, `hidden_correctness_golden` (the
live golden's own pin), `live_golden_speculative`, `public_captures` and
`official_scoring_enabled`.

`tools/bonsai2-27b-measure-and-score.sh` refuses a declared decoder and depth
that has no `live_golden_speculative` entry. The keys are `mtp1` to `mtp7` and
`dflash1` to `dflash16`, so the arming commit needs all 23.

A per-depth oracle can differ from the serial tape. A speculative round verifies its
draft in one forward at M > 1, and MLX dispatches a different kernel there, so
the free-run tape can fork from the serial tape at a near-tie (section 11.5 of
the participant contract). benchd states the same for MLX and verifies the
serial-control leg against the serial tape (`--control-golden`). On the Qwen
3.8 125B MLX track the six per-depth oracles are byte-identical to each other
and differ from the serial tape. Record an oracle at every depth, then let the
bytes decide the shape.

`tools/golden-arming-patch.py` writes the patch from a directory of recorded
files. It computes each pin, and it maps every key whose bytes are identical
to one R2 object. The organizer uploads each distinct object once. The tool
reads the files; it never prints their contents.

```bash
python3 tools/golden-arming-patch.py --dir DIR --live NAME
```

This command prints the patch and, on stderr, the list of objects to upload.
Add `--apply` to write the values into the fixture.

## 6. Correctness findings

### 6.1 The shortlist path scored folded rows against an unrotated hidden state

`Qwen35InlineMTPAssistant.shortlistLogits` gathers rows of the target's output
projection and multiplies them itself. `HadamardQuantizedLinear` IS a
`QuantizedLinear`, so the cast succeeded and the forward transform was never
applied. On the tied branch, `HadamardQuantizedEmbedding` is NOT a
`QuantizedEmbedding`, so the code fell through to a float gather of packed
integers.

Both are fixed. The head now rotates the hidden state before the gathered
matmul, and the packed embedding has its own branch.

The path is off unless `DARKBLOOM_QWEN_MTP_SHORTLIST` is set, so this was a
latent defect and not an observed one. Every other use of the target's
embedding and output projection goes through the module, so the transforms
apply.

### 6.2 The exact-verify lane assumes 4-bit group-64 and no rotation

`qwen35A3BExactW4G64Projection` and its pair and quad variants call
`unsafeDowncast` to `QuantizedLinear` and dispatch a hand-written Metal kernel
templated on 4-bit group-64 packing. They apply no Hadamard rotation. On this
pack they would be wrong in two independent ways.

The lane is UNREACHABLE under `bench-worker`. It is selected by
`Qwen35A3BConstructionContext.installation`, a task-local that nothing in
`MLXRunners` sets, so `exactTargetVerify` is false and the MTP assistant
resolves `rectangular` rather than `rectangularExact`.

It is left in place, and it is recorded here because `Qwen35A3BOptimization.swift`
and `Qwen35A3BTargetVerify.swift` are editable paths. A participant who turns
that lane on will fail correctness.

### 6.3 The fork doc says not to attach a head

`Vendor/mlx-swift-lm/docs/bonsai2.md` said "do not attach a similarly named
checkpoint's head or advertise MTP". David's ruling overrides that sentence for
this track, and the file now records what this repository does and why. The
underlying caution is real and is restated in section 7.

## 7. What is pending

- **The goldens.** Section 5. `official_scoring_enabled` stays false, the
  timed pool stays empty, and `hidden_correctness_golden` keeps the
  `BONSAI2-27B-MLX-V1-PENDING-ORGANIZER` sentinel until the box records them.
- **The build.** Nothing in this port has been compiled. The development
  laptop has Xcode installed but its licence is not accepted and its Metal
  toolchain component is not downloaded, so `metal` refuses and the vendored
  MLX C target cannot build. Every Swift change here is unverified by a
  compiler. Build it on a box before you trust it.
- **Head acceptance.** The head was trained against a Qwen3.8-27B trunk at its
  own precision. This trunk is the same architecture at 2 bits with a folded
  transform. The geometry check passes and the head can only propose, so
  output cannot move; the ACCEPT RATE can. Nobody has measured it. Measure it
  on the box before the first scored window, and treat a low rate as a fact
  about the pairing rather than a bug.
- **DFlash 2 acceptance.** The drafter was trained against a bf16 Qwen3.8-27B
  trunk. This trunk is the same architecture at 2 bits with a folded transform.
  The drafter can only propose, so output cannot move; the ACCEPT RATE can.
  Nobody has measured it. Measure it on the box before the first scored window,
  and treat a low rate as a fact about the pairing rather than a bug. This is
  the same caution the MTP head carries above, for the same reason.
- **benchd labels a DFlash 2 candidate leg `serial 0`.** benchd's
  `legserve::spec_arguments` derives the resident boot's `--spec` and
  `--draft-len` pair from `spec.mtp.depth` ALONE. It never reads `spec.mode`
  and never reads the DFlash depth, so a DFlash 2 candidate boots its resident
  as `--spec serial --draft-len 0`. Those two flags never reach the worker: the
  worker learns the decoder per request from the wire, and the boot picks the
  drafter from the leg's own declaration rather than from the label. The
  MEASUREMENT is therefore correct. What is wrong is the resident's own BOOT
  RECORD: the `<socket>.ready` marker and the identity JSON label a DFlash 2
  candidate leg `serial 0`. Record it as a known inaccuracy in the ledger, not
  as a measurement error. The fix is one line in `spec_arguments`, on the
  benchd side, and this repository cannot make it. The engine-decides MTP form
  `{"mode":"mtp","mtp":{}}` has the same cause and also boots serial. The same
  cause leaves a DFlash 2 pair asymmetric in drafter RESIDENCY: the control leg
  boots from the organizer's reference tree, which declares the MTP head, so
  the two legs hold a 239 MB export and a 3.85 GB one. benchd would have to
  pass the candidate's decoder to both boots for the two legs to match.
- **A ranked runner.** None is registered. `.github/workflows/benchmark.yml`
  targets `[self-hosted, macOS, bonsai2-27b-mlx-v1]`.
