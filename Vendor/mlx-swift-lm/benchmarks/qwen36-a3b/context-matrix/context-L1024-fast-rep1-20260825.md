# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 |
| Label | context-L1024-fast-rep1-20260825 |
| Output parity | fast |
| Verification route | rectangular-target-authoritative |
| Profile | full |
| Prefill construction | chunk=8192, soloStripe=8192 |
| Chip | Apple M5 Max |
| RAM | 128 GB |
| OS | Version 26.5.2 (Build 25F84) |
| Host at start | load avg (1m) 2.4 / 18 cores; no darkbloom process |
| Invocation | `~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/.build/arm64-apple-macosx/release/BenchCBv2 --model ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 --mode perf --engines v2 --batches 1 --steps 1024 --model-revision 73a03825c2226177f3e679210965dba3508cdee8 --mlx-swift-revision 606d28cfa8c1d66b2975d3162a4aac9756835c5f --gpu-lock-owner codex-qwen36-realistic-context-matrix --prompt-file ~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/benchmarks/qwen36-a3b/contexts/python-coding-context-1024.txt --receipt ~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/benchmarks/qwen36-a3b/context-matrix/context-L1024-fast-rep1-20260825.json --label context-L1024-fast-rep1-20260825 --out ~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/benchmarks/qwen36-a3b/context-matrix/context-L1024-fast-rep1-20260825.md --profile full --prefill-chunk 8192 --solo-stripe 8192 --mtp-depth 2 --output-parity fast` |
| Prompt contract | measured campaign prompt; 1024 tokens; SHA-256 75fa9037d8a6ee6e539ff35ce706f695d7bc3a093b9f040b51a9bfff1e179ecc |
| Paged nominalMaxSeqLen | not used by one-prompt contiguous campaign |
| mlx-swift-lm (build) | aa5c23d |
| Date | 2026-08-25T10:27:48Z |

model class: Qwen35MoEModel; layers: 10
- optimization model: layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false; safeR1(requested=false, effective=false, aot=true, nax=true)
- optimization-run-json: {"model":{"layer18Effective":false,"layer18Requested":false,"safeR1GeometryEligible":false,"weightedUnsortEffective":false,"weightedUnsortRequested":false},"safeR1AOTAvailable":true,"safeR1Effective":false,"safeR1NAXAvailable":true,"safeR1Requested":false}
vocabSize=248320

## Exact Qwen campaign

- profile: full
- output parity: fast
- verification route: rectangular-target-authoritative
- prompt: 1024 tokens; sha256 75fa9037d8a6ee6e539ff35ce706f695d7bc3a093b9f040b51a9bfff1e179ecc
- prefill: 0.282s, 3636.3 tok/s
- decode: 197.2 tok/s (1023 tokens after first token)
- generated: 1024 greedy tokens; finish length
- route: ["modelOptimizations": "layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false", "outputParity": "fast", "prefill": "chunk=8192,solo-stripe=8192", "artifactContract": "H2048/E256/K8/I512/L40; target=affine-w4-g64; mtp=mxfp8-g32", "engine": "v2-contiguous", "target": "row-owned-E256-K8+combine-M1M2", "decode": "inline-mtp", "verificationRoute": "rectangular-target-authoritative", "mtp": "inline-fixed-k2,verify=rectangular,rounds=404,proposed=808,accepted=618,emitted=1022"]
- receipt: ~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/benchmarks/qwen36-a3b/context-matrix/context-L1024-fast-rep1-20260825.json
