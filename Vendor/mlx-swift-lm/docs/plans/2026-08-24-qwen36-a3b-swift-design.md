# Qwen3.6 35B-A3B Swift Performance Port Design

## Objective

Branch from `ab73a827c9dde6f8802507003aa0be71605aab8e` and run
`EigenLabs/Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8` through a Swift/MLX text
path that exceeds 200 decode tokens/second on an approximately 100-token
Python coding prompt with exactly 100 generated tokens. Prefill is an equal
performance objective and is measured independently at 128, 1K, 8K, and 32K
tokens. The combined VLM artifact remains the sole weight source.

## Premise check

The work is justified, but only after separating three effects:

1. The pinned commit already contains opt-in 2048-token solo stripes,
   recurrent prompt-output narrowing, and packed recurrent prefill. Its stock
   scheduler still defaults to 512-token prompt chunks, so slow prefill can be
   a construction/configuration failure rather than a missing kernel.
2. The prior MTPLX target used the same 40-layer, hidden-2048, E256/top-8,
   intermediate-512 target geometry with W4/g64 routed weights and W8/g64
   routers. That makes the target-side arithmetic a valid source of hypotheses.
3. The EigenLabs inline MTP subtree is MXFP8/g32. Prior affine MTP kernels are
   not eligible by topology alone. Their ownership and scheduling ideas may be
   right-shaped for MXFP8/g32, but their packing and dequantization may not be
   copied.

The proportional solution is therefore a matched campaign: enable and measure
the facilities already present, profile the remaining real shapes, and install
only independently profitable candidates. The cost of doing nothing is slow
TTFT even if speculative decode reaches the requested headline.

## Immutable model contract

Construction reads and validates the checkpoint once. The installed route is
valid only for the exact following contract:

- model type `qwen3_5_moe`, 40 decoder layers, hidden size 2048;
- 30 recurrent GatedDelta layers and 10 full-attention layers;
- 256 routed experts, top-8, routed intermediate width 512, shared width 512;
- target routed matrices use affine W4/g64 and routers/shared gates use W8/g64;
- inline MTP has one layer and MXFP8/g32 matrix packing;
- BF16 activations and the checkpoint's existing FP32 accumulation/state rules;
- target embedding and language head are shared with the inline assistant.

If any invariant fails, optimized construction fails before generation. There
is no runtime `eligible-or-stock`, exception fallback, environment read, or
engagement counter in an enabled hot path. Stock and optimized operation are
separate construction profiles.

## Construction profiles

`Qwen35A3BOptimizationProfile` has four explicit values:

- `stock`: unchanged pinned behavior and the control for every comparison;
- `prefill`: accepted prefill routes only, no speculative decode;
- `decode`: accepted target/MTP decode routes, stock prompt processing;
- `full`: the retained winner stack from both campaigns.

The benchmark executable chooses the profile once, loads the model, validates
the manifest and realized modules, installs fixed callables/routes, runs exact
self-checks, and only then starts measured generation. Environment variables
may select a process profile before construction; model execution never reads
them.

## Benchmark surface

Extend `BenchCBv2`, rather than create a disconnected micro-runner, so the
measurement exercises the real tokenizer, request-owned recurrent state,
continuous-batching engine, caches, and inline assistant.

The runner adds:

- Qwen35/Qwen35MoE v2 hooks and cache construction;
- exact inline MTP loading from the same indexed checkpoint;
- construction profile and fixed MTP depth selection;
- a UTF-8 prompt file option, with the resulting token count recorded;
- separate prompt-evaluation wall time, first-token latency, and decode time;
- JSON receipts containing model SHA, package revisions, build revision,
  prompt SHA, token IDs, timings, route profile, and peak memory.

The approximately 100-token Python prompt is frozen by content SHA and actual
tokenizer count, not hand-counted words. The decode metric excludes the first
token and covers exactly 100 emitted tokens. Greedy sampling is mandatory.

## Prefill campaign

Prefill is never measured with a decode QMV kernel pretending to represent a
prompt GEMM. The campaign proceeds in this order:

1. Unchanged 512-token control.
2. Existing recurrent output narrowing and 2048 solo stripe, with a
   512/1024/2048/4096 chunk sweep. The winner is selected outside generation
   and fixed at construction.
3. Construction-time four-in-one GDN input projection, preserving the exact
   quantization mode, group size, output-row order, and checkpoint topology.
4. Token-major direct expert reduction, preserving sorted expert ownership and
   the exact score application order.
5. Profile real gather-QMM production shapes at each context length. A custom
   prefill gather route is permitted only when it uses matrix/tile geometry
   appropriate to the prompt rows and beats unchanged MLX Steel gather-QMM.
6. Profile full-attention and GatedDelta work separately. Attention changes
   are admitted only for the actual 10 full-attention layers and real context
   lengths.
7. Measure text-only prompt processing and image-span prefill separately. A
   text win cannot be labeled a VLM prefill win.

Each candidate must pass token/checksum parity and an interleaved matched A/B
gate. A first authentic neutral or losing result rejects it.

## Decode and MTP campaign

Every prior MTPLX optimization is re-derived for the real Swift path:

1. Fixed-K1 target-prefix cycle: one request-owned proposal transition and one
   rectangular target M2 verification, with rejection repair and cache commit
   compiled around stable shapes.
2. Target whole-MoE M2: route, target W4/g64 gate/up/down computation, shared
   expert, and reduction are shaped for M=2, E=256, top-8, I=512, H=2048.
3. Target M1 route for rejection repair and non-speculative seeds.
4. Row-owned top-8 router with score/tie semantics identical to stock.
5. GatedDelta post-convolution C1 route for M1/M2 with the target's 30-layer
   recurrent state layout.
6. Combine tail and packed gate/up are measured as independent structural
   candidates; the existing packed gate/up path is retained only if the exact
   artifact installs it.
7. Inline MTP projection and MoE routes are separately derived for MXFP8/g32.
   Dequantization, scale addressing, ownership, and tile widths follow those
   tensors. No affine MTP packing code is reused.
8. Proposal-head persistent history, target-prefix acceptance, and captured
   recurrent commit are retained only when exact state transitions and cache
   ownership are proven.

Whole-MoE reductions may change floating-point association. Such a route must
keep route IDs and normalized scores exact, remain within a declared BF16
output bound (`maxAbs <= 0.002` and `maxRel <= 0.01`), and produce identical
greedy tokens on the frozen parity suite. Exact candidates remain bitwise.

## Gates and receipts

All timing runs use `/tmp/mtplx-gpu-exclusive.lock`, a quiet host, identical
model residency, identical package revisions, release binaries, warmup, timing
boundaries, prompt, output length, cache geometry, and sampling. The raw
receipt records rejected candidates as well as the retained stack.

Required final evidence:

- exact model revision `73a03825c2226177f3e679210965dba3508cdee8`;
- Swift source base `ab73a827c9dde6f8802507003aa0be71605aab8e`;
- resolved `mlx-swift` revision recorded for both arms;
- exact prompt SHA and actual tokenizer count near 100;
- exactly 100 generated tokens, greedy, decode throughput greater than 200;
- text prefill results at 128, 1K, 8K, and 32K;
- image-prefix prefill result reported separately;
- exact token parity against the unchanged target path;
- construction self-check output and complete candidate ledger.

## Failure behavior

- Manifest, dtype, layout, topology, or self-check mismatch: fail construction.
- Candidate parity failure: reject before timing.
- Candidate performance loss or wash: reject at the first matched gate.
- GPU lock held by another owner: wait; never interrupt the owner.
- Existing service must be stopped and restored only inside a guarded benchmark
  bracket, with its command and health state captured.

## Non-goals

The >200 tok/s requirement applies to text generation from the exact combined
artifact, not image generation. Image prefill is still measured and optimized,
but it receives its own claim. No server cutover, push, PR, or remote promotion
is part of this local campaign without separate authorization.
