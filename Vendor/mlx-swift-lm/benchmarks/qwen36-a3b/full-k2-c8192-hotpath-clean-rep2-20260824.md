# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 |
| Label | qwen36-a3b-full-k2-c8192-hotpath-clean-rep2 |
| Profile | full |
| Prefill construction | chunk=8192, soloStripe=8192 |
| Chip | Apple M5 Max |
| RAM | 128 GB |
| OS | Version 26.5.2 (Build 25F84) |
| Host at start | load avg (1m) 1.6 / 18 cores; no darkbloom process |
| Invocation | `.build/arm64-apple-macosx/release/BenchCBv2 --model ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 --mode perf --engines v2 --batches 1 --steps 100 --profile full --prefill-chunk 8192 --solo-stripe 8192 --mtp-depth 2 --prompt-file benchmarks/qwen36-a3b/python-prompt.txt --model-revision 73a03825c2226177f3e679210965dba3508cdee8 --mlx-swift-revision 606d28cfa8c1d66b2975d3162a4aac9756835c5f --gpu-lock-owner codex-mlx-swift-lm-eigenlabs-qwen36-a3b-audit --receipt benchmarks/qwen36-a3b/full-k2-c8192-hotpath-clean-rep2-20260824.json --label qwen36-a3b-full-k2-c8192-hotpath-clean-rep2 --out benchmarks/qwen36-a3b/full-k2-c8192-hotpath-clean-rep2-20260824.md` |
| Prompt contract | measured campaign prompt; 129 tokens; SHA-256 ef6da4e0964d59b4f3099c3925d2ea98dc72a9608df4749806ab3950a20825de |
| Paged nominalMaxSeqLen | not used by one-prompt contiguous campaign |
| mlx-swift-lm (build) | ab73a82-dirty |
| Date | 2026-08-25T03:17:42Z |

model class: Qwen35MoEModel; layers: 10
- optimization model: layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false; safeR1(requested=false, effective=false, aot=true, nax=true)
- optimization-run-json: {"model":{"layer18Effective":false,"layer18Requested":false,"safeR1GeometryEligible":false,"weightedUnsortEffective":false,"weightedUnsortRequested":false},"safeR1AOTAvailable":true,"safeR1Effective":false,"safeR1NAXAvailable":true,"safeR1Requested":false}
vocabSize=248320

## Exact Qwen campaign

- profile: full
- prompt: 129 tokens; sha256 ef6da4e0964d59b4f3099c3925d2ea98dc72a9608df4749806ab3950a20825de
- prefill: 0.103s, 1248.4 tok/s
- decode: 232.8 tok/s (99 tokens after first token)
- generated: 100 greedy tokens; finish length
- route: ["prefill": "chunk=8192,solo-stripe=8192", "mtp": "inline-fixed-k2,verify=rectangular,rounds=34,proposed=68,accepted=64,emitted=98", "engine": "v2-contiguous", "modelOptimizations": "layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false", "target": "row-owned-E256-K8+combine-M1M2", "artifactContract": "H2048/E256/K8/I512/L40; target=affine-w4-g64; mtp=mxfp8-g32", "decode": "inline-mtp"]
- receipt: benchmarks/qwen36-a3b/full-k2-c8192-hotpath-clean-rep2-20260824.json
