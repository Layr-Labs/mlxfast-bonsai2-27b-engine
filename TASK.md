# The task — Ternary Bonsai 2 27B MLX

Make the Ternary Bonsai 2 27B model run faster on Apple Silicon.

The ranked track is `bonsai2-27b-mlx-v1`. Read [README.md](README.md) for the
setup steps and the repository structure. Read
[`docs/participant-contract.md`](docs/participant-contract.md) for the reasons
behind the rules.

## What you optimize

You optimize the engine. The engine is the MLX runner, the offline transform,
and the vendored MLX Metal kernels that the forward pass dispatches. You also
optimize the speculative-decode arm.

The target model is `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`. It is a hybrid
model. The tower is 64 layers. Every fourth layer is full attention, so 16
layers carry a key-value cache and the other 48 are gated-delta-net recurrent
layers. The MLP is dense: a gate and an up projection into a down projection,
with no experts. The embeddings are untied.

The pack is packed at 2 bits. Every one of its 402 linear modules folds a
signed block Walsh-Hadamard transform into its weights, and `hadamard.json`
declares that transform. The runtime applies the forward transform to the
input of each packed linear and the inverse transform to the output of the
embedding. Read
[`Vendor/mlx-swift-lm/docs/bonsai2.md`](Vendor/mlx-swift-lm/docs/bonsai2.md)
before you change a projection.

The pack also declares a vision tower and ships its 333 tensors. This track
serves text only. The transform drops that namespace and the model never
builds the tower.

The track has TWO declarable speculative decoders. Each one proposes tokens,
and the target model decides every emitted token. The pack carries no drafter
of any kind: it declares `mtp_num_hidden_layers: 0`. Each decoder is therefore
a separate published export, staged beside the target.

| Decoder | Export | Staged by | Depths |
|---|---|---|---|
| MTP head | `EigenLabs/Qwen3.8-27B-MTP-4bit` | `./setup-mtp-head.sh` | 1 to 7 |
| DFlash 2 drafter | `z-lab/Qwen3.8-27B-DFlash2` | `./setup-dflash-drafter.sh` | 1 to 16 |

The MTP head proposes one token at a time. The DFlash 2 drafter is a BLOCK
drafter: it proposes a whole block in one forward pass, and the block is the
last committed token followed by the depth in mask tokens. A DFlash 2 depth is
the block size minus one. The drafter was trained at block 8, which is depth 7.
A larger block is permitted, and it simply accepts less.

The two depth ranges are independent. Neither decoder can borrow the other's
range.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 97 entries in five
groups.

| Group | Paths |
|---|---|
| The decoder declaration | `mtp-head.manifest.json` (the declaration file only) |
| The offline transform | `Sources/MLXFastTransform/` |
| The model files | `PrismHadamardCheckpoint.swift`, `PrismHadamardQwen35.swift`, the 9 `Qwen35*.swift` tower and MTP files, `DFlash2Draft.swift`, `Qwen35DFlash2Assistant.swift`, and `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift` |
| The batching engine and cache layer | `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/` and 12 `MLXLMCommon` cache, attention, RoPE and config files, plus `MLXLMServer/Runtime/ToolStreamHandler.swift` (the Gemma 4 shape) |
| The vendored kernels | The 68 MLX Metal files the forward pass dispatches |

The engine fork `Vendor/mlx-swift-lm` is a vendored tree of plain files. Its
Bonsai 2 and Qwen 3.5 model files, the batching engine and the cache layer are
editable. The runner, `bench-worker` and the server in the same fork stay
trusted.

The rule behind the list is simple. Code that **proposes** tokens or computes
the forward pass is editable. Code that **verifies**, **measures**, or
**ledgers** stays trusted. `DFlash2Draft.swift` and its adapter
`Qwen35DFlash2Assistant.swift` are on the list for that reason, the same reason
`Qwen35MTP.swift` is: they propose tokens.

The MTP head is the organizer's pinned weights. You may re-quantize it. You
may not replace it, and you may not upload head weights of your own. The
exception names the MTP head. The DFlash 2 drafter is staged and loaded as
published, in BF16.

No submission carries a drafter weight file. Nothing stages one.
`mtp-head.manifest.json` stays editable and optional, and it accepts
`"source": "pinned"` only, which on this track means the pinned export of the
decoder the declaration names:
`fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256` for the MTP head, and
`fixtures/reference_bonsai2_27b_dflash2_drafter.sha256` for the DFlash 2
drafter. `"source": "remote"` and `"source": "in_branch"` are refused by name.

Each decoder has its OWN declaration cap, because one cap cannot bound both.
The MTP head's cap is 2 GiB. The DFlash 2 drafter's cap is 4 GiB, because the
drafter is 3.85 GB of BF16. `max_bytes` may lower the cap of the declared
decoder and may not raise it. The size cap is the only gate on a declaration. A
declared `sha256` is optional, and the runner does not verify it.

A re-quantization happens ON LOAD, in memory. Nothing on disk changes, and no
artifact travels in a submission.

The head loader calls `quantize(model:)` while it binds the head. That call is
the seam, and the file that holds it is an editable path:
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift`.
Change the geometry that call selects. `docs/participant-contract.md` section
4.4 is the authority.

> **WARNING — the target quantization is frozen.**
> Do not re-quantize any target weight. Do not re-represent one. Do not change
> the numerical format of one. This holds even when the result passes every
> correctness gate. An editable transform does not license the change. The MTP
> head is a narrow exception, and the exception is re-quantization only. You may
> re-quantize the head within its 2 GiB declaration cap. You may not replace it.

> **NOTE — the stream count is locked. The decoder and the depth are not.**
> The ranked run scores one stream. You may not tune that.
>
> The decoder and the draft depth are free levers, declared in
> `mtp-head.manifest.json`, which is editable. The depth is not pinned at 1.
>
> ```json
> "spec": { "decoder": "mtp",    "enabled": true, "num_speculative_tokens": 3 }
> ```
>
> ```json
> "spec": { "decoder": "dflash", "enabled": true, "num_speculative_tokens": 7 }
> ```
>
> An absent `decoder` key reads as `"mtp"`, so a declaration written before the
> DFlash 2 arm existed keeps its meaning.
>
> Select a depth from 1 to 7 for the MTP head
> (`mtp_head.permitted_draft_depths`), or from 1 to 16 for the DFlash 2 drafter
> (`dflash_drafter.permitted_draft_depths`). The two sets are independent. A
> depth outside the declared decoder's list is refused before the engine
> starts.
>
> Each run seals `effective_spec` and `effective_mean_draft_len`, so the depth
> that ran and the draft length it realized are both visible afterwards.

You may not change anything that verifies, measures, or ledgers. This covers the
trusted harness, the target weights, the transform contract, the tokenizer, the
goldens, the gates, and the timing code.

## How to run it

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary.

```bash
./setup.sh
```

This command builds the Swift binaries and downloads the target model. It then
stages both drafters beside it: `./setup-mtp-head.sh` for the MTP head, then
`./setup-dflash-drafter.sh` for the DFlash 2 drafter.

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/Ternary-Bonsai-2-27B-mlx-2bit \
  --output weights
```

This command writes the `weights/` tree that the engine loads.

```bash
./benchmark.sh --local-iterate
```

This command runs the local test against the shipped local-iterate capture,
with no speculation. It needs no environment variable: the engine is the one
`./setup.sh` built, and the golden is the shipped capture.

```bash
./benchmark.sh --local-iterate --mtp-depth 3
```

This command runs the same local test with the MTP head at depth 3.

```bash
./benchmark.sh --local-iterate --dflash-depth 7
```

This command runs the same local test with the DFlash 2 drafter at depth 7.

`--mtp-depth` and `--dflash-depth` name the same leg's decoder, so pass exactly
one of them. Each flag also exports the matching drafter directory to the
worker in `BENCH_WORKER_DRAFTER`.

> **WARNING — the two checked-in `.json` captures do not load yet.**
> They are captures from the QWEN tokenizer, so the model-identity loader
> refuses them. README.md, "The public prompt and its two captures", states the
> detail. The prompt text file is fine and does not change.

## How it scores

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain      = baseline_aggregate / candidate_aggregate
```

The score is serial-anchored. A faster candidate scores above 1.

The ranked run measures two legs on the same box, in the same job: a
serial-control leg on the organizer's reference tree, then your candidate leg
at the decoder and draft depth you declare. Each leg times one stream over a 512-token
seed and a 128-step decode window. It runs 4 pairs. The floor is 0.90. The
ceiling is 5.0. The KV backend is pinned `contiguous`.

The benchmarker applies a per-stream token-tolerance gate with a 10% budget.

> **WARNING — the gate accepts similar output, not identical output.**
> This track does not require token-for-token equality with the serial
> trajectory. The gate prices divergence against the 10% budget.

## The current state

> **NOTE — the track is NOT armed.**
> `fixtures/bonsai2_27b_mlx_v1_track.json` sets `official_scoring_enabled` to
> `false`, and the benchmarker refuses to seal an official scoring artifact
> while it is. The timed prompt pool and the live golden are empty, and the
> hidden correctness oracle carries the pending sentinel
> `BONSAI2-27B-MLX-V1-PENDING-ORGANIZER`. No ORGANIZER goldens exist
> for this track, and no runner advertises the ranked label.

> **NOTE — the engine fork is a vendored tree, edited here.**
> `Vendor/mlx-swift-lm` was cut from `Layr-Labs/mlx-swift-lm`
> `feat/qwen38-flash-next-runner` `859f4d9` with fork `main` `0607011` merged in;
> the merge, the Bonsai 2 support files and two build fixes are this repository's
> commits. Nothing goes upstream.
> `docs/bonsai2-27b-port-notes.md` holds the detail.

The repositories stay private until launch.

## Local runs are directional

The local test runs one stream and so does the ranked run, so the two shapes
agree. Your machine is still not the ranked box: its memory budget, its thermal
behavior and its kernel selection all differ. Treat a local score as a smoke
signal, not as a prediction. The ranked M5 run is the authority.

## Authorities

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, the timed pool, scoring semantics | `fixtures/bonsai2_27b_mlx_v1_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`, verified against the dist `benchd.manifest.json`) |

Where this document and the contract fixture disagree, the fixture wins. Where
either disagrees with the benchmarker about measurement, the benchmarker wins.
