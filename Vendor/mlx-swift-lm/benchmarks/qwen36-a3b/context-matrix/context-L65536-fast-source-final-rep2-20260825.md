# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 |
| Label | fast-source-final-L65536-rep2 |
| Output parity | fast |
| Verification route | rectangular-target-authoritative |
| Profile | full |
| Prefill construction | chunk=8192, soloStripe=8192 |
| Chip | Apple M5 Max |
| RAM | 128 GB |
| OS | Version 26.5.2 (Build 25F84) |
| Host at start | load avg (1m) 1.7 / 18 cores; no darkbloom process |
| Invocation | `~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/.build/arm64-apple-macosx/release/BenchCBv2 --model ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 --mode perf --engines v2 --batches 1 --steps 1024 --model-revision 73a03825c2226177f3e679210965dba3508cdee8 --mlx-swift-revision 6b0505cc790f512ae49d740b21e13f80802946bd --profile full --prefill-chunk 8192 --solo-stripe 8192 --gpu-lock-owner codex-a3b-exact-dispatch-fusion --mtp-depth 2 --output-parity fast --prompt-file benchmarks/qwen36-a3b/contexts/python-coding-context-65536.txt --receipt benchmarks/qwen36-a3b/context-matrix/context-L65536-fast-source-final-rep2-20260825.json --label fast-source-final-L65536-rep2 --out benchmarks/qwen36-a3b/context-matrix/context-L65536-fast-source-final-rep2-20260825.md` |
| Prompt contract | measured campaign prompt; 65536 tokens; SHA-256 f21b3a6bdba10ab5e35ad8c968e15ea2566fc122b5e8c1aaa4e01207d659eb44 |
| Paged nominalMaxSeqLen | not used by one-prompt contiguous campaign |
| mlx-swift-lm (build) | 3b20ab1 |
| Date | 2026-08-25T14:46:02Z |

model class: Qwen35MoEModel; layers: 10
- optimization model: layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false; safeR1(requested=false, effective=false, aot=true, nax=true)
- optimization-run-json: {"model":{"layer18Effective":false,"layer18Requested":false,"safeR1GeometryEligible":false,"weightedUnsortEffective":false,"weightedUnsortRequested":false},"safeR1AOTAvailable":true,"safeR1Effective":false,"safeR1NAXAvailable":true,"safeR1Requested":false}
vocabSize=248320

## Exact Qwen campaign

- profile: full
- output parity: fast
- verification route: rectangular-target-authoritative
- prompt: 65536 tokens; sha256 f21b3a6bdba10ab5e35ad8c968e15ea2566fc122b5e8c1aaa4e01207d659eb44
- prefill: 27.588s, 2375.6 tok/s
- decode: 112.4 tok/s (1023 tokens after first token)
- generated: 1024 greedy tokens; finish length
- route: ["artifactContract": "H2048/E256/K8/I512/L40; target=affine-w4-g64; mtp=mxfp8-g32", "target": "row-owned-E256-K8+combine-M1M2", "mtp": "inline-fixed-k2,verify=rectangular,rounds=381,proposed=762,accepted=641,emitted=1022", "verificationRoute": "rectangular-target-authoritative", "engine": "v2-contiguous", "prefill": "chunk=8192,solo-stripe=8192", "decode": "inline-mtp", "modelOptimizations": "layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false", "outputParity": "fast"]
- receipt: benchmarks/qwen36-a3b/context-matrix/context-L65536-fast-source-final-rep2-20260825.json
