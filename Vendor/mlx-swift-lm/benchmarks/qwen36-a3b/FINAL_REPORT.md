# EigenLabs Qwen3.6 35B-A3B Swift Optimization Report

## Source-final context matrix

Each cell uses a non-repeating CPython 3.12 repository context with the coding
task at the end, followed by exactly 1,024 generated tokens. Stock has one run
per context. Optimized AR, fast K2, and rectangular-exact K2 are medians of two
paired runs. All runs held `/tmp/mtplx-gpu-exclusive.lock`.

| context | stock TTFT | optimized AR TTFT | reduction | stock prefill | optimized AR prefill | stock AR decode | optimized AR decode | fast K2 decode | exact K2 decode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 293.2 ms | 233.7 ms | 20.3% | 3492.2 tok/s | 4381.6 tok/s | 141.6 tok/s | 147.7 tok/s | 211.6 tok/s | 191.1 tok/s |
| 16,384 | 5793.1 ms | 3926.7 ms | 32.2% | 2828.2 tok/s | 4173.8 tok/s | 125.8 tok/s | 131.2 tok/s | 177.9 tok/s | 165.0 tok/s |
| 32,768 | 13427.4 ms | 9524.7 ms | 29.1% | 2440.4 tok/s | 3440.3 tok/s | 116.0 tok/s | 121.2 tok/s | 150.9 tok/s | 140.2 tok/s |
| 65,536 | 33361.5 ms | 26191.6 ms | 21.5% | 1964.4 tok/s | 2502.2 tok/s | 96.7 tok/s | 100.1 tok/s | 112.1 tok/s | 107.4 tok/s |

Fast K2 improves decode over optimized AR by 43.3%, 35.5%, 24.5%, and
12.0%. Rectangular-exact improves it by 29.4%, 25.7%, 15.8%, and 7.3%.
End-to-end request wall time for fast K2 is approximately 29.1%, 16.2%, and
6.1% faster at 1K, 16K, and 32K; at 64K it is 0.6% slower. Exact K2 is about
21.8%, 11.2%, and 3.3% faster through 32K, then 1.6% slower at 64K.

Peak memory is not neutral:

| context | stock | optimized AR | fast/exact K2 | K2 vs stock |
|---:|---:|---:|---:|---:|
| 1,024 | 19.29 GiB | 19.46 GiB | 20.64 GiB | +7.0% |
| 16,384 | 19.70 GiB | 27.84 GiB | 32.03 GiB | +62.6% |
| 32,768 | 20.22 GiB | 29.09 GiB | 32.40 GiB | +60.2% |
| 65,536 | 21.31 GiB | 32.30 GiB | 33.34 GiB | +56.5% |

Decode throughput counts the 1,023 inter-token intervals after the first
token. The optimized prefill route uses an 8,192-token chunk and solo stripe.

## Output parity

The retained exact mode is `.rectangularExact`, recorded as
`rectangular-timewise-byte-exact`. Its generated-token arrays match the paired
optimized autoregressive receipts for all 1,024 tokens in both repetitions at
all four contexts (8/8 comparisons). Fast K2 is target-authoritative and has a
different stream.

This is exactness against the optimized AR target route, not against the stock
construction: the committed stock and optimized-AR streams differ. The earlier
serial candidate and its token-13/0/62 failures are rejected historical
evidence; they are not the implementation summarized here.

## Evidence and pins

The source-final JSON and Markdown receipts are in
[`context-matrix`](context-matrix). They record exact prompt/generated token
IDs, phase timestamps, requested and actual decode length, installed routes,
MTP totals, revisions, lock ownership, and peak memory.

| item | value |
|---|---|
| model | `EigenLabs/Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8` |
| model revision | `73a03825c2226177f3e679210965dba3508cdee8` |
| measured benchmark source | `3b20ab1` |
| mlx-swift dependency | `6b0505cc790f512ae49d740b21e13f80802946bd` |
| chip | Apple M5 Max, 128 GiB unified memory |

These receipts predate the rebase onto main's Qwen fused-GDN/direct-reduction
changes. They accurately describe source `3b20ab1`, but they are not evidence
for the rebased merge result; the parity and performance gates must be rerun on
that integrated source before a current merge claim.

## Construction boundary

Artifact inspection fixes H2048, E256/top8, I512, 40 layers, affine W4/g64
target projections, W8/g64 routers/shared gates, and MXFP8/g32 inline-MTP
matrices. Every explicit per-layer packing used by the exact kernels is checked.
After weights load, the exact route validates every concrete projection,
affine bias buffer, dtype, and output tiling before model execution.

The inspected contract produces an immutable, task-scoped construction
installation. Concurrent loads cannot share or overwrite route selection.
Enabled hot paths execute their installed lane directly and route only on
logical M; they do not re-read metadata or the environment, count engagement,
or silently fall back after an enabled-lane failure.

## Disposition

Retained on the measured source:

- 8,192-token prefill chunk and solo stripe;
- fixed-K2 rectangular target-authoritative fast decode;
- fixed-K2 rectangular-exact decode relative to optimized AR;
- exact-artifact row-owned E256/top8 router, composed with main's direct expert
  reduction rather than replacing it.

Rejected:

- the old serial exact candidate;
- a 16K prefill stripe;
- fixed K1/K3/K4;
- target affine kernels applied to the differently packed MXFP8 assistant.

See [`OPTIMIZATION_LEDGER.md`](OPTIMIZATION_LEDGER.md) for candidate history.
