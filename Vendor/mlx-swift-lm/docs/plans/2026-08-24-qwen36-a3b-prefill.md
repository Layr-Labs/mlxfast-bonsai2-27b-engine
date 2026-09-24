# Qwen3.6 35B-A3B Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:executing-plans to implement this plan task-by-task. GPU measurements remain serial under the canonical lock.

**Goal:** Establish a trustworthy exact-model harness and make both short and long text prefill materially faster without contaminating decode or silently changing arithmetic.

**Architecture:** `BenchCBv2` loads the combined checkpoint through the text factory, constructs a stock or optimized Qwen route once, and emits phase-separated JSON receipts. Existing stripe/narrowing/packing facilities are measured first; GDN input fusion, expert reduction, gather geometry, and attention are admitted individually. The installed profile contains fixed callables and chunk geometry, never runtime eligibility or fallback checks.

**Tech Stack:** Swift 6.3, SwiftPM, MLX Swift at the resolved revision in `Package.resolved`, Metal/MLX custom kernels, XCTest/Swift Testing, JSON receipts.

**Assumptions:**

- Assumes the exact EigenLabs checkpoint revision and manifest described in the design — it will NOT install on another quantization or MTP layout.
- Assumes text prefill can use the LLM factory while reading the combined indexed artifact — it does NOT claim image-span prefill from text-only results.
- Assumes `/tmp/mtplx-gpu-exclusive.lock` is held by this campaign — timing is invalid otherwise.

---

## File structure

- `Sources/BenchCBv2Core/BenchCBv2RealModel.swift`: Qwen hooks, CLI options, phase metrics, receipt serialization, and exact prompt input.
- `Tests/BenchCBv2Tests/BenchCBv2HarnessTests.swift`: CLI and timing-contract tests.
- `Tests/BenchCBv2Tests/BenchCBv2QwenTests.swift`: Qwen hook and receipt identity tests.
- `Libraries/MLXLLM/Models/Qwen35A3BOptimization.swift`: construction contract, profile, installer, and immutable route table.
- `Libraries/MLXLLM/Models/Qwen35A3BKernels.swift`: Qwen-specific prefill/decode kernel wrappers and Metal sources.
- `Libraries/MLXLLM/Models/Qwen35.swift`: construction hook points only; hot forwards call an already-bound route.
- `Libraries/MLXLMCommon/SwitchLayers.swift`: installed expert-reduction callable seam.
- `Tests/MLXLMTests/Qwen35A3BConstructionTests.swift`: exact eligibility and fail-before-run tests.
- `Tests/MLXLMTests/Qwen35A3BPrefillParityTests.swift`: per-candidate numerical and shape parity.
- `benchmarks/qwen36-a3b/`: frozen prompt, JSON receipts, and append-only candidate ledger.

### Task 1: Add exact Qwen benchmark and receipt surface

**Files:**
- Modify: `Package.swift`
- Modify: `Sources/BenchCBv2Core/BenchCBv2RealModel.swift`
- Modify: `Tests/BenchCBv2Tests/BenchCBv2HarnessTests.swift`
- Create: `Tests/BenchCBv2Tests/BenchCBv2QwenTests.swift`
- Create: `benchmarks/qwen36-a3b/python-prompt.txt`
- Create: `benchmarks/qwen36-a3b/OPTIMIZATION_LEDGER.md`

**Security flag:** none

**Does NOT cover:** Kernel or model arithmetic changes; this task only makes the existing stock and configured paths measurable.

- [ ] **Step 1: Write failing option and phase-metric tests**

```swift
@Test func qwenCampaignOptionsAreStrictAndConstructionScoped() throws {
    let o = try BenchOptions.parse([
        "--model", "/m", "--profile", "stock", "--prompt-file", "/p",
        "--steps", "100", "--prefill-chunk", "2048", "--solo-stripe", "2048",
        "--mtp-depth", "1", "--receipt", "/r.json",
    ])
    #expect(o.profile == .stock)
    #expect(o.steps == 100)
    #expect(o.prefillChunk == 2048)
    #expect(o.soloStripe == 2048)
    #expect(o.mtpDepth == 1)
}

@Test func phaseMetricsExcludeFirstTokenFromDecode() {
    let m = PhaseMetrics(
        promptTokens: 100, submittedAt: 10, firstTokenAt: 10.2,
        lastTokenAt: 10.695, generatedTokens: 100)
    #expect(m.prefillSeconds == 0.2)
    #expect(abs(m.decodeTPS - 200.0) < 0.001)
}
```

- [ ] **Step 2: Verify RED**

Run: `swift test --filter 'BenchCBv2.*(qwenCampaignOptions|phaseMetrics)'`

Expected: compile failure because `profile`, `promptFile`, `prefillChunk`, `soloStripe`, `mtpDepth`, `receipt`, and `PhaseMetrics` do not exist.

- [ ] **Step 3: Add strict options, Qwen hooks, prompt tokenization, and JSON schema**

Add a `BenchProfile: String, Codable { case stock, prefill, decode, full }`, strict positive option parsing, and this phase calculation:

```swift
struct PhaseMetrics: Codable, Equatable {
    let promptTokens: Int
    let submittedAt: Double
    let firstTokenAt: Double
    let lastTokenAt: Double
    let generatedTokens: Int
    var prefillSeconds: Double { firstTokenAt - submittedAt }
    var prefillTPS: Double { Double(promptTokens) / prefillSeconds }
    var decodeTPS: Double {
        Double(generatedTokens - 1) / (lastTokenAt - firstTokenAt)
    }
}
```

Add `Qwen35Model`, `Qwen35MoEModel`, and their text towers to `v2Hooks(for:)`, deriving `cbv2LayerKinds` and `newCacheV2` from the same instance passed to the engine. Read `--prompt-file`, tokenize it through the loaded `ModelContext`, and refuse to run if `--steps != 100` in campaign receipt mode. Record the real token count rather than padding or truncating it silently.

- [ ] **Step 4: Verify GREEN and harness regression suite**

Run: `swift test --filter BenchCBv2`

Expected: all BenchCBv2 tests pass; existing engine labels and provenance remain unchanged.

- [ ] **Step 5: Freeze the Python prompt and ledger header**

Use a deterministic coding request asking for a Python implementation plus tests, long enough to tokenize near 100 tokens. Record its SHA-256, but make the actual tokenizer count authoritative in receipts. Initialize ledger columns `candidate`, `source`, `shapes`, `parity`, `control`, `candidate`, `verdict`, `receipt`.

### Task 2: Install immutable exact-artifact optimization profiles

**Files:**
- Create: `Libraries/MLXLLM/Models/Qwen35A3BOptimization.swift`
- Modify: `Libraries/MLXLLM/Models/Qwen35.swift`
- Create: `Tests/MLXLMTests/Qwen35A3BConstructionTests.swift`

**Security flag:** none

**Does NOT cover:** Any model whose target is not W4/g64 plus W8/g64 routers or whose inline assistant is not MXFP8/g32.

- [ ] **Step 1: Write failing construction-contract tests**

```swift
func testExactEigenLabsContractSelectsSeparateTargetAndMTPPacking() throws {
    let c = try Qwen35A3BArtifactContract.inspect(fixture: .eigenLabsRouter8)
    XCTAssertEqual(c.target, .affine(bits: 4, groupSize: 64, routerBits: 8))
    XCTAssertEqual(c.mtp, .mxfp8(groupSize: 32))
    XCTAssertEqual(c.geometry, .init(h: 2048, e: 256, k: 8, i: 512, layers: 40))
}

func testWrongMTPPackingFailsBeforeInstallation() throws {
    var f = Qwen35A3BArtifactFixture.eigenLabsRouter8
    f.mtpGroupSize = 64
    XCTAssertThrowsError(try Qwen35A3BArtifactContract.inspect(fixture: f))
}
```

- [ ] **Step 2: Verify RED**

Run: `swift test --filter Qwen35A3BConstructionTests`

Expected: compile failure because the contract, profile, and route table are absent.

- [ ] **Step 3: Implement construction inspection and one-time installation**

Define immutable `Qwen35A3BArtifactContract`, `Qwen35A3BOptimizationProfile`, and `Qwen35A3BRouteTable`. Inspection consumes decoded configuration plus realized module types/weights after load. `install(profile:)` binds direct function values for GDN projection, expert reduction, router, target MoE M1/M2, and MTP M1/M2. A stock table binds only stock functions. Optimized forwards call the bound function directly.

Do not add `if eligible`, `try custom`, environment reads, or counters inside layer calls. Reject any mismatched layer before publishing the route table.

- [ ] **Step 4: Verify GREEN and existing Qwen configuration tests**

Run: `swift test --filter 'Qwen35(A3BConstruction|CBv2Configuration)Tests'`

Expected: all tests pass and the pre-existing Qwen capabilities remain unchanged.

### Task 3: Right-shape and gate prefill candidates

**Files:**
- Create: `Libraries/MLXLLM/Models/Qwen35A3BKernels.swift`
- Modify: `Libraries/MLXLLM/Models/Qwen35.swift`
- Modify: `Libraries/MLXLMCommon/SwitchLayers.swift`
- Create: `Tests/MLXLMTests/Qwen35A3BPrefillParityTests.swift`
- Modify: `benchmarks/qwen36-a3b/OPTIMIZATION_LEDGER.md`

**Security flag:** none

**Does NOT cover:** Decode M1/M2 kernels or MTP proposal kernels; prompt matrices use their real M>1 geometry and MLX Steel remains the control.

- [ ] **Step 1: Write failing exact-shape parity tests**

```swift
func testFusedGDNInputProjectionPreservesOutputOrder() {
    // [q, k, v, z, beta, decay] row ranges are asserted explicitly.
    assertAllClose(candidate, stock, rtol: 0, atol: 4.1e-6)
}

func testDirectExpertReductionPreservesSortedOwnership() {
    // B=1, L=512/2048, topK=8, H=2048; duplicate expert IDs included.
    assertAllClose(candidate, stock, rtol: 0, atol: 0)
}

func testPrefillRouteRejectsDecodeGeometryAtConstruction() {
    XCTAssertThrowsError(try route.install(logicalM: 1))
}
```

- [ ] **Step 2: Verify RED**

Run: `swift test --filter Qwen35A3BPrefillParityTests`

Expected: compile failure because the prefill kernels and installed route are absent.

- [ ] **Step 3: Implement and verify the GDN projection candidate**

Concatenate the four compatible quantized projections along output rows in the exact order `qkv,z,b,a`, share their W4/g64 scale/bias layout, realize the fused arrays at construction, and bind a single QMM callable. Slice views only after the projection. Run the parity test, then a matched 8K TTFT bracket. Retain only if the median improves by at least 1% with no token divergence.

- [ ] **Step 4: Implement and verify direct expert reduction**

Flatten `[B,L,topK,H]` to sorted assignment rows, use the inverse permutation to accumulate each score into its token-major owner in stock order, and return `[B,L,H]` without materializing an unsorted assignment tensor. Bind only at construction for E256/top-8/H2048. Run parity and an isolated production-shape bracket before end-to-end timing.

- [ ] **Step 5: Sweep existing chunk/stripe geometry before new gather code**

Run stock arithmetic with construction-fixed `(chunk,stripe)` pairs `(512,nil)`, `(1024,1024)`, `(2048,2048)`, `(4096,4096)` at 128/1024/8192/32768 prompt tokens. Use release builds, B=1, three interleaved repetitions, identical caches, and no decode beyond the first token. Record TTFT, prefill TPS, and peak memory. Select one fixed text-prefill geometry only if it wins the long-prompt aggregate and does not regress 128/1K by more than 2%.

- [ ] **Step 6: Profile real gather and attention shapes and conditionally implement only a proven gap**

Capture dispatch shapes outside the measured path. Compare MLX Steel gather-QMM with a candidate that owns prompt tiles (never scalar decode QMV). The candidate must use the observed rows-per-expert distribution, E256/top-8, I512/H2048, W4/g64 metadata, and matrix tiles. Reject it immediately if isolated performance is neutral or worse. Apply the same rule to the ten full-attention layers; do not alter the 30 GDN layers based on attention results.

- [ ] **Step 7: Verify the retained prefill stack**

Run: `swift test --filter 'Qwen35(A3BPrefillParity|A3BConstruction|CBv2Configuration)Tests'`

Expected: all tests pass; every retained candidate has a ledger row and every rejection records its matched receipt.

### Task 4: Publish matched prefill receipts without broadening claims

**Files:**
- Modify: `Sources/BenchCBv2Core/BenchCBv2RealModel.swift`
- Modify: `benchmarks/qwen36-a3b/OPTIMIZATION_LEDGER.md`
- Create: `benchmarks/qwen36-a3b/prefill-<timestamp>.json`

**Security flag:** none

**Does NOT cover:** Decode >200 tok/s or image generation throughput; image-prefix prefill receives a separate receipt.

- [ ] **Step 1: Run the unchanged stock matrix under the canonical lock**

Run the release `BenchCBv2` binary with profile `stock`, B=1, greedy one-token output, and prompt lengths `128,1024,8192,32768`. Capture source/dependency revisions, model SHA, prompt/token IDs, host state, warmups, timing boundaries, and peak memory.

- [ ] **Step 2: Run the full prefill profile with identical inputs**

Repeat the exact matrix with profile `prefill`; reject the stack if any input identity or build provenance differs. Require at least a 1% repeatable improvement for every retained marginal candidate and no greater than 2% short-prompt regression.

- [ ] **Step 3: Run image-span prefill separately**

Use a fixed local image and fixed text prompt through the VLM factory. Record vision-encoder time, language-prefix time, and TTFT separately. Do not merge this number into text prefill throughput.

- [ ] **Step 4: Verify receipt completeness**

Run a JSON-schema test that rejects missing model SHA, Swift revisions, profile, prompt SHA/token count, generated token IDs, phase times, route table, memory, or lock ownership. Append the winner stack and rejected candidates to the ledger.
