# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 |
| Label | context-L65536-stock-rep2-20260825 |
| Profile | stock |
| Prefill construction | chunk=512, soloStripe=off |
| Chip | Apple M5 Max |
| RAM | 128 GB |
| OS | Version 26.5.2 (Build 25F84) |
| Host at start | load avg (1m) 1.9 / 18 cores; no darkbloom process |
| Invocation | `~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/.build/arm64-apple-macosx/release/BenchCBv2 --model ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 --mode perf --engines v2 --batches 1 --steps 1024 --model-revision 73a03825c2226177f3e679210965dba3508cdee8 --mlx-swift-revision 606d28cfa8c1d66b2975d3162a4aac9756835c5f --gpu-lock-owner codex-qwen36-realistic-context-matrix --prompt-file ~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/benchmarks/qwen36-a3b/contexts/python-coding-context-65536.txt --receipt ~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/benchmarks/qwen36-a3b/context-matrix/context-L65536-stock-rep2-20260825.json --label context-L65536-stock-rep2-20260825 --out ~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/benchmarks/qwen36-a3b/context-matrix/context-L65536-stock-rep2-20260825.md --profile stock` |
| Prompt contract | measured campaign prompt; 65536 tokens; SHA-256 f21b3a6bdba10ab5e35ad8c968e15ea2566fc122b5e8c1aaa4e01207d659eb44 |
| Paged nominalMaxSeqLen | not used by one-prompt contiguous campaign |
| mlx-swift-lm (build) | aa5c23d |
| Date | 2026-08-25T10:46:00Z |

model class: Qwen35MoEModel; layers: 10
- optimization model: layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false; safeR1(requested=false, effective=false, aot=true, nax=true)
- optimization-run-json: {"model":{"layer18Effective":false,"layer18Requested":false,"safeR1GeometryEligible":false,"weightedUnsortEffective":false,"weightedUnsortRequested":false},"safeR1AOTAvailable":true,"safeR1Effective":false,"safeR1NAXAvailable":true,"safeR1Requested":false}
vocabSize=248320

## Exact Qwen campaign

- profile: stock
- prompt: 65536 tokens; sha256 f21b3a6bdba10ab5e35ad8c968e15ea2566fc122b5e8c1aaa4e01207d659eb44
- prefill: 39.857s, 1644.3 tok/s
- decode: 95.3 tok/s (1023 tokens after first token)
- generated: 1024 greedy tokens; finish length
- route: ["decode": "pinned-default", "target": "pinned-default", "artifactContract": "H2048/E256/K8/I512/L40; target=affine-w4-g64; mtp=mxfp8-g32", "modelOptimizations": "layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false", "mtp": "disabled", "engine": "v2-contiguous", "prefill": "chunk=512,solo-stripe=off"]
- receipt: ~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/benchmarks/qwen36-a3b/context-matrix/context-L65536-stock-rep2-20260825.json
