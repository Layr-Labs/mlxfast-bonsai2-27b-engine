# BenchCBv2RealModel report

| | |
|---|---|
| Model | ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 |
| Label | byte-exact-dispatch-fused-L16384 |
| Output parity | byte-exact |
| Verification route | rectangular-timewise-byte-exact |
| Profile | full |
| Prefill construction | chunk=8192, soloStripe=8192 |
| Chip | Apple M5 Max |
| RAM | 128 GB |
| OS | Version 26.5.2 (Build 25F84) |
| Host at start | load avg (1m) 1.8 / 18 cores; no darkbloom process |
| Invocation | `~/projects/OpenSourceWTF/.worktrees/mlx-swift-lm-eigenlabs-a3b/.build/arm64-apple-macosx/release/BenchCBv2 --model ~/.cache/huggingface/hub/models--EigenLabs--Qwen3.6-35B-A3B-MLX-VL-4bit-g64-router8/snapshots/73a03825c2226177f3e679210965dba3508cdee8 --mode perf --engines v2 --batches 1 --steps 1024 --model-revision 73a03825c2226177f3e679210965dba3508cdee8 --mlx-swift-revision 6b0505cc790f512ae49d740b21e13f80802946bd --profile full --prefill-chunk 8192 --solo-stripe 8192 --mtp-depth 2 --output-parity byte-exact --gpu-lock-owner codex-a3b-exact-dispatch-fusion --prompt-file benchmarks/qwen36-a3b/contexts/python-coding-context-16384.txt --receipt benchmarks/qwen36-a3b/context-matrix/context-L16384-byte-exact-dispatch-fused-rep1-20260825.json --label byte-exact-dispatch-fused-L16384 --out benchmarks/qwen36-a3b/context-matrix/context-L16384-byte-exact-dispatch-fused-rep1-20260825.md` |
| Prompt contract | measured campaign prompt; 16384 tokens; SHA-256 00abbf748514d7ff8609d5871156c8a8dea8e8eb5f355393c94296c846d0d1f9 |
| Paged nominalMaxSeqLen | not used by one-prompt contiguous campaign |
| mlx-swift-lm (build) | 904d495-dirty |
| Date | 2026-08-25T14:11:47Z |

model class: Qwen35MoEModel; layers: 10
- optimization model: layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false; safeR1(requested=false, effective=false, aot=true, nax=true)
- optimization-run-json: {"model":{"layer18Effective":false,"layer18Requested":false,"safeR1GeometryEligible":false,"weightedUnsortEffective":false,"weightedUnsortRequested":false},"safeR1AOTAvailable":true,"safeR1Effective":false,"safeR1NAXAvailable":true,"safeR1Requested":false}
vocabSize=248320

## Exact Qwen campaign

- profile: full
- output parity: byte-exact
- verification route: rectangular-timewise-byte-exact
- prompt: 16384 tokens; sha256 00abbf748514d7ff8609d5871156c8a8dea8e8eb5f355393c94296c846d0d1f9
- prefill: 4.291s, 3818.6 tok/s
- decode: 165.9 tok/s (1023 tokens after first token)
- generated: 1024 greedy tokens; finish length
- route: ["engine": "v2-contiguous", "target": "row-owned-E256-K8+combine-M1M2", "modelOptimizations": "layer18(requested=false, effective=false, interval=n/a); weightedUnsort(requested=false, effective=false); safeR1GeometryEligible=false", "prefill": "chunk=8192,solo-stripe=8192", "verificationRoute": "rectangular-timewise-byte-exact", "outputParity": "byte-exact", "mtp": "inline-fixed-k2,verify=rectangular_exact,rounds=377,proposed=754,accepted=645,emitted=1022", "artifactContract": "H2048/E256/K8/I512/L40; target=affine-w4-g64; mtp=mxfp8-g32", "decode": "inline-mtp"]
- receipt: benchmarks/qwen36-a3b/context-matrix/context-L16384-byte-exact-dispatch-fused-rep1-20260825.json
