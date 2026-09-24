# Qwen3.6 A3B Output-Parity Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a construction-time `--output-parity fast|byte-exact` benchmark flag and publish byte-exact decode as a third measured datapoint beside stock and optimized-fast.

**Architecture:** A typed `BenchOutputParity` value owns both the installed `CBv2MTPVerificationMode` and its stable receipt label. Parsing records separately whether the user explicitly supplied the flag so stock and prefill defaults remain compatible while meaningless explicit combinations fail before construction. The chosen route is written to JSON, Markdown, and route metadata; execution receives an already-specialized MTP configuration and adds no parity branch to the measured hot path.

**Tech Stack:** Swift 6, Swift Testing, MLX Swift, BenchCBv2, Markdown and JSON benchmark receipts.

**Assumptions:**

- Assumes `CBv2MTPVerificationMode.serialTarget` preserves serial target evaluation order — this will NOT claim byte exactness if a locked-GPU token comparison disagrees with the stock receipt.
- Assumes the pinned EigenLabs snapshot and mlx-swift revision remain available locally — benchmark numbers will NOT be substituted from another Qwen checkpoint or dependency revision.
- Assumes `/tmp/mtplx-gpu-exclusive.lock` can be acquired without displacing another owner — timing will NOT run while another owner holds the GPU lane.
- Assumes the existing stock and optimized-fast receipts remain the unchanged matched controls — the report will explicitly show their sample counts rather than imply all three rows have equal repetitions.

---

## File structure

- `Sources/BenchCBv2Core/BenchCBv2RealModel.swift`: define and parse the typed parity option, validate construction combinations, install the selected MTP verification route, and record it in receipts and Markdown.
- `Tests/BenchCBv2Tests/BenchCBv2HarnessTests.swift`: lock parser, construction mapping, receipt schema, and report-label behavior.
- `benchmarks/qwen36-a3b/`: store three new byte-exact Markdown/JSON receipts produced from the pinned model under the GPU lock.
- `benchmarks/qwen36-a3b/FINAL_REPORT.md`: add the third decode datapoint and link its raw receipts.
- `benchmarks/qwen36-a3b/OPTIMIZATION_LEDGER.md`: change serial verification from a rejected hidden candidate to an explicit measured parity mode while preserving its performance disposition.

### Task 1: Parse and validate the output-parity contract

**Files:**
- Modify: `Tests/BenchCBv2Tests/BenchCBv2HarnessTests.swift`
- Modify: `Sources/BenchCBv2Core/BenchCBv2RealModel.swift`

**Security flag:** none

**Does NOT cover:** This flag does not apply to stock, prefill-only, or MTP-depth-zero routes; those explicit combinations fail before model construction.

- [x] **Step 1: Write failing parser tests**

Add tests that exercise the public CLI contract rather than mutating `BenchOptions` directly:

```swift
@Test("output parity defaults to fast and parses byte exact")
func outputParityParsesStrictly() throws {
    let defaults = try BenchOptions.parse(["--model", "/m"])
    #expect(defaults.outputParity == .fast)
    #expect(defaults.outputParityWasSpecified == false)

    let exact = try BenchOptions.parse([
        "--model", "/m", "--profile", "decode", "--mtp-depth", "2",
        "--output-parity", "byte-exact",
    ])
    #expect(exact.outputParity == .byteExact)
    #expect(exact.outputParityWasSpecified)

    #expect(throws: BenchOptionError.self) {
        _ = try BenchOptions.parse([
            "--model", "/m", "--profile", "decode", "--mtp-depth", "2",
            "--output-parity", "approximate",
        ])
    }
}

@Test("explicit output parity requires a decode capable MTP route")
func outputParityRejectsMeaninglessRoutes() {
    for arguments in [
        ["--model", "/m", "--profile", "stock", "--output-parity", "fast"],
        ["--model", "/m", "--profile", "prefill", "--output-parity", "byte-exact"],
        ["--model", "/m", "--profile", "decode", "--mtp-depth", "0",
         "--output-parity", "byte-exact"],
    ] {
        #expect(throws: BenchOptionError.self) { _ = try BenchOptions.parse(arguments) }
    }
}
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```sh
swift test --filter BenchCBv2HarnessTests.outputParity
```

Expected: compilation fails because `BenchOutputParity`, `outputParity`, and `outputParityWasSpecified` do not exist.

- [x] **Step 3: Implement the typed parser and construction-time validation**

Add the enum beside `BenchProfile`:

```swift
enum BenchOutputParity: String, Codable, CaseIterable, Sendable {
    case fast
    case byteExact = "byte-exact"

    var verificationMode: CBv2MTPVerificationMode {
        switch self {
        case .fast: .rectangular
        case .byteExact: .serialTarget
        }
    }

    var verificationRoute: String {
        switch self {
        case .fast: "rectangular-target-authoritative"
        case .byteExact: "serial-byte-exact"
        }
    }
}
```

Add `outputParity = BenchOutputParity.fast` and
`outputParityWasSpecified = false` to `BenchOptions`, list
`[--output-parity fast|byte-exact]` in usage, parse the value strictly, and set
the supplied marker. After all arguments are parsed, validate an explicitly
supplied value with:

```swift
if options.outputParityWasSpecified {
    guard options.profile == .decode || options.profile == .full else {
        throw BenchOptionError.invalidValue(
            option: "--output-parity", value: options.outputParity.rawValue,
            requirement: "requires profile decode or full")
    }
    guard options.mtpDepth > 0 else {
        throw BenchOptionError.invalidValue(
            option: "--output-parity", value: options.outputParity.rawValue,
            requirement: "requires --mtp-depth greater than zero")
    }
}
```

- [x] **Step 4: Run the parser tests and verify GREEN**

Run:

```sh
swift test --filter BenchCBv2HarnessTests.outputParity
```

Expected: both output-parity tests pass.

### Task 2: Install and report the selected route

**Files:**
- Modify: `Tests/BenchCBv2Tests/BenchCBv2HarnessTests.swift`
- Modify: `Sources/BenchCBv2Core/BenchCBv2RealModel.swift`

**Security flag:** none

**Does NOT cover:** Route selection does not change target kernels, MTP draft width, prefill geometry, output arithmetic within either verification implementation, or runtime fallback behavior.

- [x] **Step 1: Write failing construction and receipt tests**

Extend the MTP construction test with both mappings:

```swift
stock.outputParity = .fast
#expect(campaignMTPConfig(stock).verificationMode == .rectangular)
stock.outputParity = .byteExact
#expect(campaignMTPConfig(stock).verificationMode == .serialTarget)
#expect(BenchOutputParity.fast.verificationRoute == "rectangular-target-authoritative")
#expect(BenchOutputParity.byteExact.verificationRoute == "serial-byte-exact")
```

Construct a decode receipt with `outputParity: .byteExact` and
`verificationRoute: BenchOutputParity.byteExact.verificationRoute`; assert the
encoded JSON contains exactly:

```swift
#expect(json.contains("\"outputParity\":\"byte-exact\""))
#expect(json.contains("\"verificationRoute\":\"serial-byte-exact\""))
```

Add a pure report-header assertion for a decode/MTP option:

```swift
#expect(header.contains("| Output parity | byte-exact |"))
#expect(header.contains("| Verification route | serial-byte-exact |"))
```

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```sh
swift test --filter BenchCBv2HarnessTests
```

Expected: the serial mapping and new receipt/report fields fail because the production path still hard-codes rectangular verification and omits parity metadata.

- [x] **Step 3: Implement construction and reporting**

Change `campaignMTPConfig` to install the already-selected route:

```swift
verificationMode: options.outputParity.verificationMode,
```

When MTP is enabled, add these construction-derived keys to the returned route
summary:

```swift
"outputParity": options.outputParity.rawValue,
"verificationRoute": options.outputParity.verificationRoute,
```

Add `outputParity: BenchOutputParity?` and `verificationRoute: String?` to
`CampaignReceipt`. Initialize them only when `mtpDepth > 0`; use `nil` for
stock and prefill receipts. Add matching `Output parity` and `Verification
route` rows to `reportHeader` only when `mtpDepth > 0`, and emit the same two
values in the exact-campaign Markdown block before the decode metric.

- [x] **Step 4: Run focused suites and verify GREEN**

Run:

```sh
swift test --filter BenchCBv2HarnessTests
swift test --filter Qwen35A3BConstructionTests
swift test --filter BenchCBv2QwenTests
```

Expected: all focused suites pass, including the pre-existing structural test
that forbids invariant validation and silent fallback in optimized closures.

### Task 3: Build and verify the release artifact

**Files:**
- Modify: `benchmarks/qwen36-a3b/FINAL_REPORT.md`

**Security flag:** none

- [x] **Step 1: Build the exact release benchmark**

Run:

```sh
scripts/build-qwen36-a3b-bench.sh
.build/arm64-apple-macosx/release/BenchCBv2 --print-revision
shasum -a 256 .build/arm64-apple-macosx/release/BenchCBv2 \
  .build/arm64-apple-macosx/release/mlx.metallib
```

Expected: the build succeeds, the revision identifies this branch state, and
both hashes are captured for the report.

- [x] **Step 2: Run the relevant non-GPU verification**

Run:

```sh
swift test --filter BenchCBv2HarnessTests
swift test --filter Qwen35A3BConstructionTests
swift test --filter BenchCBv2QwenTests
```

Expected: all focused tests pass with zero failures.

### Task 4: Measure byte-exact mode and publish the third datapoint

**Files:**
- Create: `benchmarks/qwen36-a3b/full-k2-c8192-byte-exact-rep1-20260825.json`
- Create: `benchmarks/qwen36-a3b/full-k2-c8192-byte-exact-rep1-20260825.md`
- Create: `benchmarks/qwen36-a3b/full-k2-c8192-byte-exact-rep2-20260825.json`
- Create: `benchmarks/qwen36-a3b/full-k2-c8192-byte-exact-rep2-20260825.md`
- Create: `benchmarks/qwen36-a3b/full-k2-c8192-byte-exact-rep3-20260825.json`
- Create: `benchmarks/qwen36-a3b/full-k2-c8192-byte-exact-rep3-20260825.md`
- Modify: `benchmarks/qwen36-a3b/FINAL_REPORT.md`
- Modify: `benchmarks/qwen36-a3b/OPTIMIZATION_LEDGER.md`

**Security flag:** none

**Does NOT cover:** A byte-exact result below the fast or stock throughput is recorded honestly; it is not promoted as the default and does not trigger another optimization campaign in this PR.

- [x] **Step 1: Acquire the GPU lock without displacing another owner**

Inspect `/tmp/mtplx-gpu-exclusive.lock`. If it names a live unrelated owner,
wait and retry without stopping that process. Once the lane is free, acquire
the lock for `codex-mlx-swift-lm-eigenlabs-qwen36-a3b-byte-exact`, stop only the
known Qwen service inside a restoration trap, and verify no other benchmark or
model process is using Metal. Keep the lock through all timing and token
comparisons.

- [x] **Step 2: Run three matched byte-exact repetitions**

For repetitions 1 through 3, run the release binary with the exact same model,
prompt, 100-token boundary, full profile, 8192-token prefill stripe, and fixed
K2 route as the optimized-fast control, adding only:

```sh
--output-parity byte-exact
```

Write each run to the exact JSON and Markdown filenames listed above. The full
invocation must retain these pins:

```text
model revision: 73a03825c2226177f3e679210965dba3508cdee8
mlx-swift revision: 606d28cfa8c1d66b2975d3162a4aac9756835c5f
prompt: benchmarks/qwen36-a3b/python-prompt.txt
profile: full
prefill chunk and solo stripe: 8192
mtp depth: 2
generated tokens: 100
```

Expected: every receipt says `outputParity=byte-exact`,
`verificationRoute=serial-byte-exact`, and `verify=serialTarget`; all three
runs generate the same 100 token IDs.

- [x] **Step 3: Prove byte parity against stock**

Use `jq -c '.generatedTokenIDs'` on each new JSON receipt and
`decode-stock-20260824T2137.json`, then compare the byte streams with `cmp`.

Expected: all three new generated-token arrays are byte-for-byte identical to
the stock array. If any differ, do not label the mode byte-exact or update the
headline table; preserve the receipts as a failed gate and diagnose before PR.

- [x] **Step 4: Calculate and add the third decode datapoint**

Extract each `phaseMetrics` object and compute steady decode as
`(generatedTokens - 1) / (lastTokenAt - firstTokenAt)`. Sort the three values
and use the middle value as the byte-exact median. Replace the existing decode
summary with a three-row comparison table containing:

```text
stock serial control | byte-exact | 140.0 tok/s | n=1 | 1.000x
optimized byte-exact K2 | byte-exact | measured median | n=3 | median / 140.0
optimized fast K2 | target-authoritative | 232.8 tok/s | n=3 | 1.663x
```

Also retain the per-repetition fast table, add a per-repetition byte-exact
table, link all six new raw receipts, and state explicitly that the byte-exact
mode is opt-in while `fast` remains the default.

- [x] **Step 5: Update the candidate ledger and restore the machine**

Record the byte-exact route as retained as an opt-in correctness datapoint but
not the throughput winner. Restore the exact pre-benchmark Qwen service,
verify its health endpoint and configured model, then release
`/tmp/mtplx-gpu-exclusive.lock`.

- [x] **Step 6: Final verification and commit**

Run the verification-before-completion checklist, inspect `git diff --check`,
run the focused tests again, verify every report path exists, and confirm the
JSON route fields with `jq`. Then commit the source, tests, spec, plan, reports,
and receipts as one reviewable change.
