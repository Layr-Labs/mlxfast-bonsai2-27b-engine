# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 |
| Label | qwen36-a3b-full-k2-c8192-byte-exact-rep1-20260825 |
| Output parity | byte-exact |
| Verification route | serial-byte-exact |
| Profile | full |
| Prefill construction | chunk=8192, soloStripe=8192 |
| Chip | Apple M5 Max |
| RAM | 128 GB |
| OS | Version 26.5.2 (Build 25F84) |
| Host at start | load avg (1m) 2.9 / 18 cores; no darkbloom process |
| Invocation | `~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/.build/arm64-apple-macosx/release/BenchCBv2 --model ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 --mode perf --engines v2 --batches 1 --steps 100 --profile full --prefill-chunk 8192 --solo-stripe 8192 --mtp-depth 2 --output-parity byte-exact --prompt-file benchmarks/qwen36-a3b/python-prompt.txt --model-revision 73a03825c2226177f3e679210965dba3508cdee8 --mlx-swift-revision 606d28cfa8c1d66b2975d3162a4aac9756835c5f --gpu-lock-owner codex-mlx-swift-lm-eigenlabs-qwen36-a3b-byte-exact --receipt benchmarks/qwen36-a3b/full-k2-c8192-byte-exact-rep1-20260825.json --label qwen36-a3b-full-k2-c8192-byte-exact-rep1-20260825 --out benchmarks/qwen36-a3b/full-k2-c8192-byte-exact-rep1-20260825.md` |
| Prompt contract | measured campaign prompt; 129 tokens; SHA-256 ef6da4e0964d59b4f3099c3925d2ea98dc72a9608df4749806ab3950a20825de |
| Paged nominalMaxSeqLen | not used by one-prompt contiguous campaign |
| mlx-swift-lm (build) | ab73a82-dirty |
| Date | 2026-08-25T09:39:57Z |

model class: Qwen35MoEModel; layers: 10
- optimization model: layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false; safeR1(requested=false, effective=false, aot=true, nax=true)
- optimization-run-json: {"model":{"layer18Effective":false,"layer18Requested":false,"safeR1GeometryEligible":false,"weightedUnsortEffective":false,"weightedUnsortRequested":false},"safeR1AOTAvailable":true,"safeR1Effective":false,"safeR1NAXAvailable":true,"safeR1Requested":false}
vocabSize=248320

## Exact Qwen campaign

- profile: full
- output parity: byte-exact
- verification route: serial-byte-exact
- prompt: 129 tokens; sha256 ef6da4e0964d59b4f3099c3925d2ea98dc72a9608df4749806ab3950a20825de
- prefill: 0.103s, 1247.0 tok/s
- decode: 109.9 tok/s (99 tokens after first token)
- generated: 100 greedy tokens; finish length
- route: ["target": "row-owned-E256-K8+combine-M1M2", "engine": "v2-contiguous", "decode": "inline-mtp", "verificationRoute": "serial-byte-exact", "mtp": "inline-fixed-k2,verify=serial_target,rounds=35,proposed=70,accepted=63,emitted=98", "outputParity": "byte-exact", "artifactContract": "H2048/E256/K8/I512/L40; target=affine-w4-g64; mtp=mxfp8-g32", "modelOptimizations": "layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false", "prefill": "chunk=8192,solo-stripe=8192"]
- receipt: benchmarks/qwen36-a3b/full-k2-c8192-byte-exact-rep1-20260825.json
