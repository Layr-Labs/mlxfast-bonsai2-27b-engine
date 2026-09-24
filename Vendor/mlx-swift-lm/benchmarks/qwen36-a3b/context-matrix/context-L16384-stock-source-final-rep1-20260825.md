# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 |
| Label | stock-source-final-L16384-rep1 |
| Profile | stock |
| Prefill construction | chunk=512, soloStripe=off |
| Chip | Apple M5 Max |
| RAM | 128 GB |
| OS | Version 26.5.2 (Build 25F84) |
| Host at start | load avg (1m) 2.3 / 18 cores; no darkbloom process |
| Invocation | `~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/.build/arm64-apple-macosx/release/BenchCBv2 --model ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 --mode perf --engines v2 --batches 1 --steps 1024 --model-revision 73a03825c2226177f3e679210965dba3508cdee8 --mlx-swift-revision 6b0505cc790f512ae49d740b21e13f80802946bd --profile stock --mtp-depth 0 --gpu-lock-owner codex-a3b-exact-dispatch-fusion --prompt-file benchmarks/qwen36-a3b/contexts/python-coding-context-16384.txt --receipt benchmarks/qwen36-a3b/context-matrix/context-L16384-stock-source-final-rep1-20260825.json --label stock-source-final-L16384-rep1 --out benchmarks/qwen36-a3b/context-matrix/context-L16384-stock-source-final-rep1-20260825.md` |
| Prompt contract | measured campaign prompt; 16384 tokens; SHA-256 00abbf748514d7ff8609d5871156c8a8dea8e8eb5f355393c94296c846d0d1f9 |
| Paged nominalMaxSeqLen | not used by one-prompt contiguous campaign |
| mlx-swift-lm (build) | 3b20ab1 |
| Date | 2026-08-25T14:47:51Z |

model class: Qwen35MoEModel; layers: 10
- optimization model: layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false; safeR1(requested=false, effective=false, aot=true, nax=true)
- optimization-run-json: {"model":{"layer18Effective":false,"layer18Requested":false,"safeR1GeometryEligible":false,"weightedUnsortEffective":false,"weightedUnsortRequested":false},"safeR1AOTAvailable":true,"safeR1Effective":false,"safeR1NAXAvailable":true,"safeR1Requested":false}
vocabSize=248320

## Exact Qwen campaign

- profile: stock
- prompt: 16384 tokens; sha256 00abbf748514d7ff8609d5871156c8a8dea8e8eb5f355393c94296c846d0d1f9
- prefill: 5.793s, 2828.2 tok/s
- decode: 125.8 tok/s (1023 tokens after first token)
- generated: 1024 greedy tokens; finish length
- route: ["mtp": "disabled", "prefill": "chunk=512,solo-stripe=off", "modelOptimizations": "layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false", "artifactContract": "H2048/E256/K8/I512/L40; target=affine-w4-g64; mtp=mxfp8-g32", "target": "pinned-default", "engine": "v2-contiguous", "decode": "pinned-default"]
- receipt: benchmarks/qwen36-a3b/context-matrix/context-L16384-stock-source-final-rep1-20260825.json
