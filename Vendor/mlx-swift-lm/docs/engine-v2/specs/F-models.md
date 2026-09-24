# WS-F: Model adaptation — Gemma 4 + GPT-OSS 20B on the v2 attention path

You own the two production models' integration. EXCLUSIVE ownership of
`Libraries/MLXLLM/Models/Gemma4Text.swift` and `Libraries/MLXLLM/Models/GPTOSS.swift`.

## Deliverables

1. `Libraries/MLXLMCommon/ContinuousBatchingV2/LayerKindDerivation.swift` —
   `[CBv2LayerKind]` derivation from model configs:
   - Gemma 4: `layer_types` / slidingWindowPattern 5 (4×sliding(512) + 1×
     full), trailing `num_kv_shared_layers` (20 of 35) with
     `sharesKVWithLayer` = last non-shared layer OF THE SAME TYPE (mirror
     the existing `previousKvs` map, Gemma4Text.swift:877-893), asymmetric
     head dims (sliding 256 / global 512), K-eq-V variants, hasSinks=false.
   - GPT-OSS: alternating sliding(128)/full, hasSinks=true, headDim 64.
   Pure functions + unit tests against both configs (construct configs in
   code; no weight downloads).
2. Gemma4Text.swift: add a v2 attention branch — when the layer's cache is
   a `CBv2AttendingLayerCache`, route through `updateAndAttend` /
   `attendBorrowing` (KV-shared layers) instead of the manual mask/SDPA
   forks. Requirements:
   - Per-row RoPE via `cache.positionOffsets` (array-offset RoPE overload);
     KV-shared layers reuse the SOURCE layer's captured offsets (invariant
     1, report 10). Snapshot offsets BEFORE update.
   - Dual RoPE thetas and QK-norm flow unchanged.
   - The legacy (B=1 / old engine) paths must remain byte-identical — v2 is
     a new branch, not a rewrite. `gemma4AttentionFallback` and mask logic
     stay for legacy callers.
3. GPTOSS.swift: same v2 branch; sinks passed into `updateAndAttend`
   (contract guarantees denominator-only handling); kill the per-request
   `sinksActive` `.item()` host readback on the v2 path (compute once at
   load, cache the Bool).
4. `newCacheV2(backend:)` (or equivalent factory) on both models producing
   `[CBv2AttendingLayerCache]` via the contract types + LayerKindDerivation
   (WS-A implements the concrete classes; code against the protocols and a
   trivial in-file mock for tests — integration wires the real ones).

## Tests (`Tests/MLXLMTests/CBv2ModelTests.swift`)
- LayerKindDerivation: exact expected [CBv2LayerKind] for Gemma-4 (35
  layers, shared map matches previousKvs semantics) and GPT-OSS (24
  layers alternating), including K-eq-V and head-dim asymmetry.
- Tiny-config forward smoke test: construct Gemma4Text and GPTOSS with
  2–4 layer random-weight tiny configs; run a 5-token prompt + 3 decode
  steps through the v2 branch with a mock AttendingLayerCache; assert
  shapes, offset progression, and that the shared-layer branch calls
  `attendBorrowing` with the right source layer.
- Legacy-path regression: existing model tests still pass untouched.

## References
Report 10 §1 (exact model structures, file:line), §4 invariants 1, 2, 5,
9, 11. Corpus: `~/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/`.

## Qwen 3.8 Flash-Next (`qwen4_exp`, `qwen4_exp_text`)

This family has 48 layers. 12 layers use full attention. 36 layers use a
gated-deltanet recurrence. The recurrent layers keep no key-value tape.
`cbv2LayerKinds` therefore gives 12 rows. Each row has a `modelLayerIndex`
that points to its decoder layer.

Each full-attention layer has a QSA indexer. The indexer keeps a budget of
2048 visible tokens. It pools the keys into blocks of 4 tokens. It scores the
blocks and keeps the best blocks for each query. The result is a keep mask.
The model sends the mask to attention through the `keepMask` parameter of
`CBv2AttendingLayerCache.updateAndAttend`. The model declares
`CBv2KeepMaskRequiringModel`. If the cache provider cannot apply the mask, the
engine refuses to start.

The indexer keeps its own key tape. `Qwen4ExpCBv2LayerCache` holds this tape
for each row. Before each append, the cache cuts the tape to the length that
the row's `absoluteOffset` gives. A rollback of the key-value tape thus also
moves the indexer tape.

The residual stream is `hc_count` streams side by side. There is no final norm
tensor. The last mixer does this function. The MTP head reads the stream
before the last mixer. This stream is `hc_count * hidden` wide.
`cbv2ForwardWithHidden` gives this stream as `lastHidden`.

### Auxiliary recurrent-state indices

The PLE layer keeps two more pieces of state: a short-convolution state and
the n-gram token history. These do not belong to a decoder layer. They ride
the recurrent-state spec under synthetic index values. The values are
`hiddenLayers + ordinal`, one for each PLE layer. `conv` holds the
short-convolution state. `ssm` holds the history token ids as `int32`. Read
the values from `cbv2AuxiliaryStateLayerIndices`. Do not build them again.
Do not compare a recurrent-state index with the layer count: these indices are
larger than the last layer index on purpose.

This family serves one row for each call. The QSA indexer scores one tape.
Packed prefill, paged key-value storage, prefix reuse and compiled decode stay
off. MTP draft depth is 1 to 3.

### Runner (`Libraries/MLXRunners/Qwen4ExpRunner.swift`)

`Qwen4ExpRunner` puts this family behind the runner contract. It claims the
`qwen4_exp` and `qwen4_exp_text` model types in `RunnerRegistry`. Its manifest
declares contiguous key-value storage only, one serial decoder, one MTP decoder
with the embedded head at depth 1 to 3, single-stream free-run and
teacher-forced regimes, recurrent layers, and `requiresKeepMask`. The manifest
digest is `474efd99…`. Both this repo and benchd pin that value.

The runner loads the checkpoint one time. It reads `cbv2LayerKinds` from the
model. It reports the `mtp` decoder only when the checkpoint holds the `mtp.*`
head. It builds the engine and the one-row stepper over the same model
instance, so both drive one forward pass.

### The n-gram row source

The n-gram table is 29.8 GiB and is never model parameters. The model asks a
`Qwen4ExpNGramRowSource` for the rows. `Qwen4ExpNGram.swift` holds the
protocol. `Qwen4ExpNGramTable.swift` holds one conformer, which reads the rows
from the disk and caches a bounded number of them.

Add a new conformer beside that one. Give it to
`Qwen4ExpNGramRowSourceLoader`, which is the one construction entry point. Do
not change the runner: the runner names the protocol and the loader only.

The caller gives the source in `RunnerLoadOptions.resources`, under the name
`Qwen4ExpRunner.ngramRowSourceResource`. The value has one of two shapes:

- a path, as a `URL` or a `String`. This is the DIRECTORY of n-gram shard
  files that the offline transform writes. bench-worker gives its
  `--resource <name>=<path>` value in this shape. A path to one file is
  refused, and the refusal names the directory shape.
- an already built `Qwen4ExpNGramRowSource`, for a caller in the same process
  that holds one.

A checkpoint with PLE layers and no row source is refused at load.

The cache ceiling is a byte count. The default is 1 gibibyte. The environment
variable `MLXFAST_NGRAM_CACHE_LIMIT` sets it. Zero stops the cache. The
ceiling changes the memory only. It can never change a row that the source
gives back.

The cache holds hot rows for decode steps only. A gather of 4096 different
rows or fewer uses the cache. A larger gather is a prefill gather. It reads
the rows from the memory maps and it does not use the cache. Before a large
gather the source tests a sample of the rows with `mincore`. If the pages are
not in memory, the source reads the rows with `pread` first.

At load the source reads part of the table into the page cache. The
environment variable `MLXFAST_NGRAM_PREWARM` sets this behaviour. The value is
`auto`, `off` or `on`. The default is `auto`, which reads what the free memory
allows, less a margin of 6 gibibytes. The value `on` reads the whole table,
but only on a machine with 160 gibibytes of memory or more. If the model
directory holds a file `ngram-hotness.npy`, the source reads the rows of that
file first, in the order of that file. The file holds 64-bit integer row ids,
and the most used row is first. If the file is absent or unreadable, the
source reads the start of the table instead. A bad file is never a load
failure. The read changes the memory only. It can never change a row that the
source gives back.

The three tests in `Qwen4ExpForwardParityTests` need a full ahead-of-time
`mlx.metallib`, because a decode through the mixture-of-experts shared expert
gate uses the `dot_product` kernel that the standard build does not supply.
Build the metallib with `cmake -DMLX_METAL_JIT=OFF --target mlx-metallib` (see
`fetch-metallib.sh` in d-inference), copy it over every `default.metallib`
under `Build/Products/Debug`, including the copies inside each `.xctest`
bundle, and then set `MLXLM_FULL_AOT_METALLIB=1`. Without the variable the
three tests skip. For `swift test`, copy the same file to
`mlx.metallib` in `.build/<triple>/debug/mlx-swift-lmPackageTests.xctest/Contents/MacOS`.
The same gate holds for `Qwen4ExpRunnerEngineTests`.
