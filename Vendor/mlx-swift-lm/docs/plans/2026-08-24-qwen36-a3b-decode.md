# Qwen3.6 35B-A3B Decode and MTP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. GPU work is serial and each candidate stops at its first authentic matched gate.

**Goal:** Right-shape the complete prior A3B decode stack, including the EigenLabs MXFP8/g32 inline assistant, and exceed 200 decode tok/s on the frozen approximately 100-token Python prompt with exactly 100 greedy output tokens.

**Architecture:** The prefill plan supplies the exact-model runner and construction route table. Decode construction binds fixed M1 and M2 target routes, a fixed-K1 request-owned MTP cycle, and independently shaped MXFP8/g32 assistant routes. Each candidate first passes kernel/state parity, then an isolated timing gate, then a natural matched end-to-end gate; only retained callables enter the `full` profile.

**Tech Stack:** Swift 6.3, SwiftPM, MLX Swift, MLX compile/custom Metal kernels, XCTest/Swift Testing, JSON benchmark receipts.

**Assumptions:**

- Assumes the prefill plan's harness and construction contract are complete — this plan will NOT create a second benchmark path.
- Assumes fixed greedy K1 is the primary >200 lane — adaptive or deeper speculation may be measured but cannot replace the K1 acceptance contract.
- Assumes bounded BF16 reassociation is allowed for whole-MoE after route exactness and token parity — it will NOT claim bitwise equality for reordered reductions.

---

## File structure

- `Libraries/MLXLLM/Models/Qwen35A3BOptimization.swift`: retained route table and construction-time self-checks.
- `Libraries/MLXLLM/Models/Qwen35A3BKernels.swift`: target router, target/MTP MoE, GDN C1, combine-tail, and helper kernels.
- `Libraries/MLXLLM/Models/Qwen35A3BCompiledK1.swift`: request-owned fixed-K1 graph inputs/outputs and commit transitions.
- `Libraries/MLXLLM/Models/Qwen35.swift`: bound target route invocation.
- `Libraries/MLXLLM/Models/Qwen35MTP.swift`: bound MXFP8 assistant projections and request-owned proposal state.
- `Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/CBv2MTPRoundDriver.swift`: construction-selected compiled K1 driver, with stock driver remaining a distinct profile.
- `Tests/MLXLMTests/Qwen35A3BRouterTests.swift`: top-8/tie parity.
- `Tests/MLXLMTests/Qwen35A3BWholeMoETests.swift`: target and MXFP8 assistant arithmetic/shape parity.
- `Tests/MLXLMTests/Qwen35A3BGDNC1Tests.swift`: recurrent state and capture parity.
- `Tests/MLXLMTests/Qwen35A3BCompiledK1Tests.swift`: acceptance, rejection, cache, and state-transition parity.
- `benchmarks/qwen36-a3b/OPTIMIZATION_LEDGER.md`: complete candidate ledger.

### Task 1: Right-shape target and MTP module contracts

**Files:**
- Modify: `Libraries/MLXLLM/Models/Qwen35A3BOptimization.swift`
- Modify: `Libraries/MLXLLM/Models/Qwen35MTP.swift`
- Create: `Tests/MLXLMTests/Qwen35A3BWholeMoETests.swift`

**Security flag:** none

**Does NOT cover:** Timing or compiled request transitions; this task proves physical tensor interpretation and arithmetic references.

- [ ] **Step 1: Write failing target/MTP packing tests**

```swift
func testTargetAndMTPUseDifferentPhysicalReaders() throws {
    let routes = try Qwen35A3BRouteTable.make(fixture: .eigenLabsRouter8, profile: .decode)
    XCTAssertEqual(routes.targetMoE.packing, .affine(bits: 4, groupSize: 64))
    XCTAssertEqual(routes.mtpMoE.packing, .mxfp8(groupSize: 32))
    XCTAssertFalse(routes.targetMoE.readerIdentity == routes.mtpMoE.readerIdentity)
}

func testMXFP8GroupAddressingMatchesStockAssistant() {
    // M=1 and M=2, I=512, H=2048, topK=8; include exponent edge values.
    assertAllClose(candidate, stock, rtol: 0, atol: 0)
}
```

- [ ] **Step 2: Verify RED**

Run: `swift test --filter Qwen35A3BWholeMoETests`

Expected: compile failure because target/MTP physical reader identities and assistant routes do not exist.

- [ ] **Step 3: Implement separate immutable target and MXFP8 readers**

The target reader addresses affine W4 nibbles plus scale/bias groups of 64. The MTP reader addresses MXFP8 bytes plus exponent groups of 32 exactly as the realized `QuantizedLinear` module does. Both expose read-only typed descriptors containing logical dimensions, physical strides, group size, and accumulation dtype. Construction validates every routed gate/up/down path and publishes descriptors only after exact micro-self-checks against stock projections.

- [ ] **Step 4: Verify GREEN for M1/M2 projection fixtures**

Run: `swift test --filter Qwen35A3BWholeMoETests`

Expected: exact per-projection parity for both packings; no optimized whole-MoE route is installed yet.

### Task 2: Implement and gate the target M1/M2 decode kernels

**Files:**
- Modify: `Libraries/MLXLLM/Models/Qwen35A3BKernels.swift`
- Modify: `Libraries/MLXLLM/Models/Qwen35.swift`
- Create: `Tests/MLXLMTests/Qwen35A3BRouterTests.swift`
- Modify: `Tests/MLXLMTests/Qwen35A3BWholeMoETests.swift`
- Create: `Tests/MLXLMTests/Qwen35A3BGDNC1Tests.swift`
- Modify: `benchmarks/qwen36-a3b/OPTIMIZATION_LEDGER.md`

**Security flag:** none

**Does NOT cover:** Prompt M>2 matrices or the MTP assistant's MXFP8 matrices; target decode routes are fixed to M1/M2.

- [ ] **Step 1: Write failing router, whole-MoE, and GDN parity tests**

```swift
func testRowOwnedRouterMatchesStockTop8AndTieOrder() {
    XCTAssertEqual(candidate.ids, stock.ids)
    assertAllClose(candidate.scores, stock.scores, rtol: 0, atol: 0)
}

func testTargetWholeMoEM2BoundAndRoutes() {
    XCTAssertEqual(candidate.routeIDs, stock.routeIDs)
    assertAllClose(candidate.routeScores, stock.routeScores, rtol: 0, atol: 0)
    assertAllClose(candidate.output, stock.output, rtol: 0.01, atol: 0.002)
}

func testGDNC1M1M2CommitsExactConvAndSSMState() {
    assertArrayEqual(candidate.convState, stock.convState)
    assertArrayEqual(candidate.ssmState, stock.ssmState)
}
```

- [ ] **Step 2: Verify RED**

Run: `swift test --filter 'Qwen35A3B(Router|WholeMoE|GDNC1)Tests'`

Expected: compile failure because the candidate routes are absent.

- [ ] **Step 3: Implement row-owned router**

One threadgroup owns each M1/M2 row, streams all 256 W8/g64 router rows, keeps top-8 with value-descending/token-ID-ascending tie order, and normalizes selected scores in stock order. The result is `(ids:[M,8], scores:[M,8])`; it performs no host readback.

- [ ] **Step 4: Implement target whole-MoE M2 and M1 repair routes**

Use the measured target geometry H2048/E256/top-8/I512. For M2, pair both rows under one fixed-weight owner where one W4/g64 read serves both rows. Stage 1 produces exact route IDs/scores; stage 2 owns expert-slot/intermediate tiles for the eight routed plus shared expert paths; stage 3 owns output columns and reduces in a declared deterministic order. M1 uses geometry measured independently rather than pretending to be the first row of M2. Include shared-expert sigmoid gating in the installed whole-MoE callable.

- [ ] **Step 5: Implement GDN post-convolution C1 for M1/M2**

For each of the 30 recurrent layers, one owner holds a value-head/state-quarter, applies q/k normalization, gates, decay/delta update, and capture using the actual `[1,32,128,128]` FP32 state and BF16 conv tail. M1 and M2 are separate fixed entry points. State self-checks are exact before installation.

- [ ] **Step 6: Measure packed gate/up and combine tail independently**

Confirm the exact checkpoint installed the existing packed gate/up matrices. Measure combine-tail as a separate callable that applies route scores while reducing intermediate owners, with no extra assignment tensor. Retain only repeatable >=1% wins; record washes or regressions immediately.

- [ ] **Step 7: Verify retained target routes**

Run: `swift test --filter 'Qwen35A3B(Router|WholeMoE|GDNC1|Construction)Tests'`

Expected: all tests pass; target route and score parity are exact, and any reordered MoE output stays within the declared bound.

### Task 3: Implement the MXFP8 assistant and compiled fixed-K1 cycle

**Files:**
- Modify: `Libraries/MLXLLM/Models/Qwen35A3BKernels.swift`
- Create: `Libraries/MLXLLM/Models/Qwen35A3BCompiledK1.swift`
- Modify: `Libraries/MLXLLM/Models/Qwen35MTP.swift`
- Modify: `Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/CBv2MTPRoundDriver.swift`
- Create: `Tests/MLXLMTests/Qwen35A3BCompiledK1Tests.swift`
- Modify: `Tests/MLXLMTests/Qwen35A3BWholeMoETests.swift`
- Modify: `benchmarks/qwen36-a3b/OPTIMIZATION_LEDGER.md`

**Security flag:** none

**Does NOT cover:** Adaptive K2-K4 policy; success is fixed K1 and adaptive depth cannot substitute for it.

- [ ] **Step 1: Write failing MXFP8 assistant and transition tests**

```swift
func testMXFP8AssistantM1MatchesStockDraftTokenAndHidden() {
    XCTAssertEqual(candidate.token, stock.token)
    assertAllClose(candidate.hidden, stock.hidden, rtol: 0, atol: 0)
}

func testCompiledK1FullAcceptCommitsSecondTargetState() {
    XCTAssertEqual(result.emitted, reference.emitted)
    assertStateEqual(result.committed, reference.committed)
}

func testCompiledK1RejectRepairsFromFirstTargetState() {
    XCTAssertEqual(result.emitted, reference.emitted)
    assertStateEqual(result.committed, reference.committed)
}
```

- [ ] **Step 2: Verify RED**

Run: `swift test --filter 'Qwen35A3B(WholeMoE|CompiledK1)Tests'`

Expected: compile failure because MXFP8 assistant routes and compiled cycle state do not exist.

- [ ] **Step 3: Implement MXFP8/g32 assistant projections and MoE**

Build M1 and, only if the assistant actually receives M2, M2 kernels from the MXFP8 descriptors. Use group-32 exponent addressing and assistant-specific physical strides. Gate/up/down, router, shared expert, and language-head candidates each compare against the stock assistant and receive independent isolated gates. A losing component stays stock in a construction-fixed hybrid route; there is no per-token fallback.

- [ ] **Step 4: Implement request-owned fixed-K1 graph state**

Define fixed graph inputs for the committed token/hidden, proposal-head history, target KV handles, recurrent state, and one draft slot. The graph executes one assistant proposal, one target M2 verification, target-prefix comparison, and produces both full-accept and reject-repair candidate states. Host finalization selects the already-computed transition without cache snapshot/rollback reconstruction. No graph is created in the cycle.

- [ ] **Step 5: Bind compiled K1 at engine construction**

`CBv2MTPRoundDriver` receives a `Qwen35A3BCompiledK1` only from the decode/full construction profile. The driver calls it directly for fixed depth 1. Stock/adaptive drivers are separate construction choices; there is no `compiled-or-stock` branch inside a K1 round.

- [ ] **Step 6: Run exact transition and 100-token parity suites**

Run: `swift test --filter Qwen35A3BCompiledK1Tests`

Expected: exact emitted token IDs, exact cache offsets, exact committed conv/SSM state, and no state leak across accept, reject, cancellation, or request restart.

### Task 4: Cross 200 tok/s with a complete matched receipt

**Files:**
- Modify: `benchmarks/qwen36-a3b/OPTIMIZATION_LEDGER.md`
- Create: `benchmarks/qwen36-a3b/decode-control-<timestamp>.json`
- Create: `benchmarks/qwen36-a3b/decode-full-<timestamp>.json`

**Security flag:** none

**Does NOT cover:** Server deployment, remote publication, or a claim that every prompt exceeds 200 tok/s.

- [ ] **Step 1: Establish unchanged stock and stock-MTP controls**

Under `/tmp/mtplx-gpu-exclusive.lock`, stop the existing model service inside a restore trap, verify the owner PID, use the exact checkpoint snapshot, and run release profile `stock` both target-only and with the pinned stock fixed-K1 assistant. Use the frozen Python prompt, greedy sampling, exactly 100 emitted tokens, and at least one warmup plus three measured repetitions.

- [ ] **Step 2: Gate each candidate marginally**

For router, target M2 whole-MoE, target M1, GDN C1, combine-tail, MXFP8 assistant projections/MoE, and compiled K1, run interleaved unchanged-control/candidate repetitions. Reject at the first parity failure, neutral result, or regression. Append every verdict and raw receipt path to the ledger.

- [ ] **Step 3: Run the retained full stack**

Use profile `full`, identical model residency and dependency/source revisions, the actual tokenizer count near 100, exactly 100 generated tokens, and greedy output. Decode timing excludes the first token. Require every measured repetition to exceed 200 tok/s, not only the maximum; report median and range.

- [ ] **Step 4: Re-run prefill non-regression matrix**

Run 128/1K/8K/32K text prefill with the full profile and compare it with the retained prefill profile. Reject any decode addition that regresses prefill beyond the accepted 2% short-context tolerance or invalidates the long-context winner.

- [ ] **Step 5: Verify the final receipt and restore the service**

Confirm model SHA, base/dependency revisions, prompt SHA/token count, exact 100-token output, token identity, route table, acceptance histogram, phase timings, memory, host state, and lock ownership are present. Restore the previous service command and health-check it before releasing the lock.

- [ ] **Step 6: Run final source verification**

Run:

```bash
swift test --filter 'Qwen35A3B'
swift test --filter BenchCBv2
git diff --check
git status --short
```

Expected: all focused tests pass, no whitespace errors, and only intended source, test, plan, ledger, prompt, and receipt files are modified.
