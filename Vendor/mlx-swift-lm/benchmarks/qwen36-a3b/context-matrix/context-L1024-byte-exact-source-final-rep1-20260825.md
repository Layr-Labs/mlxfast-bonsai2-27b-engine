# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 |
| Label | byte-exact-source-final-L1024-rep1 |
| Output parity | byte-exact |
| Verification route | rectangular-timewise-byte-exact |
| Profile | full |
| Prefill construction | chunk=8192, soloStripe=8192 |
| Chip | Apple M5 Max |
| RAM | 128 GB |
| OS | Version 26.5.2 (Build 25F84) |
| Host at start | load avg (1m) 2.7 / 18 cores; no darkbloom process |
| Invocation | `~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/.build/arm64-apple-macosx/release/BenchCBv2 --model ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 --mode perf --engines v2 --batches 1 --steps 1024 --model-revision 73a03825c2226177f3e679210965dba3508cdee8 --mlx-swift-revision 6b0505cc790f512ae49d740b21e13f80802946bd --profile full --prefill-chunk 8192 --solo-stripe 8192 --gpu-lock-owner codex-a3b-exact-dispatch-fusion --mtp-depth 2 --output-parity byte-exact --prompt-file benchmarks/qwen36-a3b/contexts/python-coding-context-1024.txt --receipt benchmarks/qwen36-a3b/context-matrix/context-L1024-byte-exact-source-final-rep1-20260825.json --label byte-exact-source-final-L1024-rep1 --out benchmarks/qwen36-a3b/context-matrix/context-L1024-byte-exact-source-final-rep1-20260825.md` |
| Prompt contract | measured campaign prompt; 1024 tokens; SHA-256 75fa9037d8a6ee6e539ff35ce706f695d7bc3a093b9f040b51a9bfff1e179ecc |
| Paged nominalMaxSeqLen | not used by one-prompt contiguous campaign |
| mlx-swift-lm (build) | 3b20ab1 |
| Date | 2026-08-25T14:27:39Z |

model class: Qwen35MoEModel; layers: 10
- optimization model: layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false; safeR1(requested=false, effective=false, aot=true, nax=true)
- optimization-run-json: {"model":{"layer18Effective":false,"layer18Requested":false,"safeR1GeometryEligible":false,"weightedUnsortEffective":false,"weightedUnsortRequested":false},"safeR1AOTAvailable":true,"safeR1Effective":false,"safeR1NAXAvailable":true,"safeR1Requested":false}
vocabSize=248320

## Exact Qwen campaign

- profile: full
- output parity: byte-exact
- verification route: rectangular-timewise-byte-exact
- prompt: 1024 tokens; sha256 75fa9037d8a6ee6e539ff35ce706f695d7bc3a093b9f040b51a9bfff1e179ecc
- prefill: 0.246s, 4168.9 tok/s
- decode: 191.4 tok/s (1023 tokens after first token)
- generated: 1024 greedy tokens; finish length
- route: ["outputParity": "byte-exact", "target": "row-owned-E256-K8+combine-M1M2", "decode": "inline-mtp", "prefill": "chunk=8192,solo-stripe=8192", "verificationRoute": "rectangular-timewise-byte-exact", "modelOptimizations": "layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false", "mtp": "inline-fixed-k2,verify=rectangular_exact,rounds=383,proposed=765,accepted=639,emitted=1022", "artifactContract": "H2048/E256/K8/I512/L40; target=affine-w4-g64; mtp=mxfp8-g32", "engine": "v2-contiguous"]
- receipt: benchmarks/qwen36-a3b/context-matrix/context-L1024-byte-exact-source-final-rep1-20260825.json
