# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 |
| Label | qwen36-a3b-decode-mtp-k4 |
| Profile | decode |
| Prefill construction | chunk=512, soloStripe=off |
| Chip | Apple M5 Max |
| RAM | 128 GB |
| OS | Version 26.5.2 (Build 25F84) |
| Host at start | load avg (1m) 1.3 / 18 cores; no darkbloom process |
| Invocation | `.build/arm64-apple-macosx/release/BenchCBv2 --model ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 --mode perf --engines v2 --batches 1 --steps 100 --profile decode --mtp-depth 4 --prompt-file benchmarks/qwen36-a3b/python-prompt.txt --model-revision 73a03825c2226177f3e679210965dba3508cdee8 --mlx-swift-revision 606d28cfa8c1d66b2975d3162a4aac9756835c5f --gpu-lock-owner codex-mlx-swift-lm-eigenlabs-qwen36-a3b-vl --receipt benchmarks/qwen36-a3b/decode-mtp-k4-20260824.json --label qwen36-a3b-decode-mtp-k4 --out benchmarks/qwen36-a3b/decode-mtp-k4-20260824.md` |
| Prompt lengths | default mix (B=1 500; B=2 100,1500; B=4 100,500,1500,500; else 500 x B) |
| Paged nominalMaxSeqLen | 4096 |
| mlx-swift-lm (build) | ab73a82-dirty |
| Date | 2026-08-25T02:52:45Z |

model class: Qwen35MoEModel; layers: 10
- optimization model: layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false; safeR1(requested=false, effective=false, aot=true, nax=true)
- optimization-run-json: {"model":{"layer18Effective":false,"layer18Requested":false,"safeR1GeometryEligible":false,"weightedUnsortEffective":false,"weightedUnsortRequested":false},"safeR1AOTAvailable":true,"safeR1Effective":false,"safeR1NAXAvailable":true,"safeR1Requested":false}
vocabSize=248320

## Exact Qwen campaign

- profile: decode
- prompt: 129 tokens; sha256 ef6da4e0964d59b4f3099c3925d2ea98dc72a9608df4749806ab3950a20825de
- prefill: 0.103s, 1246.9 tok/s
- decode: 178.7 tok/s (99 tokens after first token)
- generated: 100 greedy tokens; finish length
- route: ["prefill": "chunk=512,solo-stripe=off", "target": "pinned-default", "modelOptimizations": "layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false", "decode": "inline-mtp", "engine": "v2-contiguous", "mtp": "inline-fixed-k4,verify=rectangular,rounds=24,proposed=94,accepted=74,emitted=98", "artifactContract": "H2048/E256/K8/I512/L40; target=affine-w4-g64; mtp=mxfp8-g32"]
- receipt: benchmarks/qwen36-a3b/decode-mtp-k4-20260824.json
