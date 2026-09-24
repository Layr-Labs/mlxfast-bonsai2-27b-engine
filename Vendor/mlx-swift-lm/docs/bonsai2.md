# Ternary Bonsai 2 27B

The `prism_hadamard_qwen35` factories load the published
[Prism 2-bit MLX artifact](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/tree/3f926b415992eaa2ae9dd7b573706494d6bbf787)
using the existing dense Qwen3.8-27B / `qwen3_5` backbone. This is not the native
Qwen4 Flash-Next model.

## Artifact contract

- Schema 2, affine 2-bit/group 128 with FP16 packed scales. Published language
  normalizers remain FP32; native normalization promotes downstream activations
  and KV/recurrent convolution rows to FP32. Do not infer state precision from
  embedding or packed scale storage alone.
- Packed recurrent prefill materializes the small convolution carry before
  retaining it. A slice otherwise owns the entire prefill-chunk allocation;
  this is a bit-preserving storage change, not a precision or recurrence change.
  Other model families and single-token decode keep their existing path.
- Signed block-Hadamard metadata in `hadamard.json`; FP32 transform arithmetic
  followed by restoration of the incoming dtype. Generic BF16 weight conversion
  is not applied to this artifact.
- Explicit grouped GDN layout, without a second output-head permutation.
- The selected weights contain 2,390 tensors: 402 sign vectors, 333 vision
  tensors and no MTP/draft tensors. The config declares vision support and
  `mtp_num_hidden_layers: 0`. A text-generation pipeline tag is not evidence
  that the vision tower is absent.
- No conversion or requantization occurs. The loader validates the transform
  manifest, duplicated sign vectors, packed dtypes/shapes and module mapping
  before returning the model. Unsupported packing contracts fail recoverably.

## Serving

`PrismHadamardQwen35TextModel` supports the text factory without loading vision
weights. `PrismHadamardQwen35` retains the vision tower and delegates native
CBv2 state to the existing Qwen text model; image/video preprocessing uses the
Qwen3-VL processor and native causal vision prefill. Generic VLM media generation
is rejected explicitly; use the native provider route for media.

The artifact itself has no MTP assistant.

This repository attaches one. The track declares a separate, published
`qwen3_5_mtp` export as the drafter, and `Qwen35Runner` binds it to the loaded
pack. That is a decision of this track and not a property of the artifact, and
it is why `Qwen35Runner` declares `assistantCheckpoint` rather than
`embeddedHead`. The head is a Qwen3.8-27B head over a Qwen3.8-27B trunk with
the same hidden size, vocabulary and attention geometry, which is what
`Qwen35InlineMTPAssistant.load` checks. The head proposes tokens only; the pack
decides every emitted token. Acceptance on this pack is unmeasured until the
track records its goldens.

The assistant borrows the pack's `embed_tokens` and `lm_head`. Both are packed
Hadamard modules here, so every use goes through the module: the embedding
output carries the inverse transform and the head input carries the forward
one. A path that reads their raw `.weight` skips both.

Native maximum context is 262144; physical memory and request admission remain
separate from that architectural limit.

## Qualification

### Opt-in performance qualification profile

This performance follow-up leaves the published weights unchanged. Both
switches below are **off by default** and must be set
before starting the provider:

```sh
DARKBLOOM_BONSAI_PREFILL_CARRY_ASYNC=1
DARKBLOOM_BONSAI_F16_CONSTANT_CACHE=1
```

The first submits already-compacted recurrent carry arrays earlier, only during
eligible packed text prefill. Native arithmetic, deferred-host-input handling,
paged write validation, final evaluation and request retirement stay intact.
Short/decode, media-position and captured-verification paths retain the previous
scheduling. The second reuses the exact FP16-to-FP32 quantization-constant
conversion the native packed operator already performs. It does not change
weights, quantization, activation/KV precision, gates or model architecture.
Descriptor and execution-stream changes invalidate that bounded reuse; tracing
falls back to the native operation. The generic `MLX_QUANTIZED_CONSTANT_CACHE=0`
rollback remains effective.

Constant reuse retains approximately 1.60 GB (decimal) of additional converted
constants for this artifact. This is a real residency cost, not a smaller model
or a free cache. Earlier carry evaluation reduces temporary prefill peaks; report
MLX active/cache and whole-process footprint separately. Neither mechanism is
MTP: the artifact has no assistant heads.

Independent frozen raw-logit, all-layer recurrent-state and paged-KV comparisons
cover both M3 Ultra and M5 Max at batch sizes 1 and 2. Native long-context,
media/cohort, cancellation/readmission, reload and encrypted-fixture checks pass
within the dependent provider's recorded scope. Ordinary API serving, lifecycle
and late-error framing pass; the original strict tool-success matrix remains
24/28, with four baseline-reproduced copy-value failures. Signed persistence and
hosted qualification remain separate open gates, not production sign-off.
Rejected or low-impact gather/sign-fusion prototypes are
not part of this candidate. No new numerical baseline or tolerance is accepted.

Focused configuration/shape, transform and HTTP regressions are included. The
dependent provider PR records full-model/API/image execution, 1K/10K/20K/50K
benchmarks, memory tradeoffs and open gates. Large decode improvements do not
imply a large or universal prefill improvement. Build success alone is not a
production, full-context, hosted OpenRouter or model-quality pass.

The wire parser uses Qwen structured tool frames. Tokenizer/template bytes and
the published quantization stay unchanged. The pack runtime's older text-only
note and stale README entry in its file manifest are superseded for identity by
the pinned config, actual tensor inventory and Hugging Face Git/LFS hashes.

## Composition and attribution

The public baseline uses the runtime sources approved in Layr-Labs/mlx-swift
PR #27 (`97b7f13830baa0aa761c979bcb4e94fc86b20cb6`, merged as
`35e55a6b9f53db22a01fa0b6fb38afef8dd3100d`). This follow-up pins the public
`Layr-Labs/mlx-swift` performance commit
`70052b2147f828e6b9bcc2dce599017415387377` for exact constant reuse. Merge the
dependency first, then repin to its observed merged commit and recheck the
composed build. No floating branch or inaccessible private dependency is used.
The required Hadamard/2-bit primitives
already exist in the pinned MLX core; no whole-fork substitution is required.
The signed-transform layers are adapted from Prism's MIT-licensed Swift work
at `6d3a84de28225d1f5bc0a56f5c781596997242f9`; pack semantics are checked against
the runtime distributed with the immutable model revision above.
