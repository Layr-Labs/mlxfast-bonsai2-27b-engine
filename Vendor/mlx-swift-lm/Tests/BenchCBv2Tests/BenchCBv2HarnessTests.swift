// BenchCBv2HarnessTests.swift
//
// Integrity gates for the BenchCBv2 harness itself. These numbers feed the
// v0.8.0 release gates, so the harness must never produce a plausible-looking
// report for a different experiment than the one requested:
//
//   * a measured row's engine label always names the path that produced it —
//     a `v2-compiled` cell cannot be served by some other backend,
//   * a malformed option is a hard error, not a silent fallback,
//   * the recorded invocation reproduces the original argv exactly,
//   * the recorded revision is the one the binary was built from.
//
// Model-free by construction: nothing here loads weights or touches Metal.

import Foundation
import MLXLMCommon
import Testing

@testable import BenchCBv2Core

// `.serialized`: one case chdirs to prove the build stamp is not derived from
// the process's working directory.
@Suite("BenchCBv2 harness integrity", .serialized)
struct BenchCBv2HarnessTests {

    // MARK: - Engine labelling

    /// Split a markdown table row into its cells the way a GFM renderer does:
    /// on *unescaped* pipes only, dropping the empty leading and trailing
    /// fields produced by the outer pipes.
    private func cells(_ row: String) -> [String] {
        let escapedPipe = "\u{FFFF}"
        return row
            .replacingOccurrences(of: "\\|", with: escapedPipe)
            .components(separatedBy: "|")
            .dropFirst().dropLast()
            .map {
                $0.replacingOccurrences(of: escapedPipe, with: "\\|")
                    .trimmingCharacters(in: .whitespaces)
            }
    }

    private var headerColumnCount: Int {
        cells(CellResult.markdownHeader.split(separator: "\n").first.map(String.init) ?? "").count
    }

    @Test("a v2-compiled cell is refused, not silently served by another backend")
    func compiledCellIsRefusedAtEveryPromptLength() throws {
        // The exact command the review calls out: a prompt far past the 4096
        // capacity the compiled executor used to fall back to eager decode at.
        for length in [100, 4_096, 10_000, 1_000_000] {
            let options = try BenchOptions.parse([
                "--model", "/tmp/model",
                "--engines", "v2-compiled",
                "--prompt-lengths", "\(length)",
            ])
            #expect(options.engines == ["v2-compiled"])

            guard case .refuse(let reason) = resolveEngine("v2-compiled") else {
                Issue.record("v2-compiled resolved to a runnable backend at \(length) tokens")
                return
            }
            #expect(reason.contains("removed in v0.8.0"))

            // The emitted row must be unmistakably a non-measurement: same
            // column count as the table header, and every metric column `-`.
            let row = cells(refusalRow(engine: "v2-compiled", reason: reason))
            #expect(row.count == headerColumnCount)
            #expect(row[0] == "v2-compiled")
            for metric in row.dropFirst() where metric != "-" {
                #expect(
                    Double(metric) == nil,
                    "refused cell carries a number in a metric column: \(metric)")
            }
        }
    }

    @Test("a measured row is always labelled with the backend that produced it")
    func measuredRowLabelMatchesBackend() {
        for name in ["v2", "v2-paged", "v2-compiled", "legacy", "v2-Paged", "", "v3"] {
            switch resolveEngine(name) {
            case .run(let backend):
                // `runV2Cell` labels the row `backend.rawValue`; if that ever
                // diverges from the requested name the report lies.
                #expect(backend.rawValue == name)
            case .refuse(let reason):
                #expect(reason.hasPrefix("refused: "))
            }
        }
    }

    @Test("retired and unknown engines both refuse rather than fall through")
    func retiredEnginesRefuse() {
        #expect(resolveEngine("legacy") == .refuse(
            reason: "refused: the legacy engine was removed in v0.8.0 — use v2"))
        #expect(resolveEngine("nonsense") == .refuse(reason: "refused: unknown engine"))
        #expect(resolveEngine("v2") == .run(.contiguous))
        #expect(resolveEngine("v2-paged") == .run(.paged))
    }

    @Test("a refusal reason cannot forge extra table columns")
    func refusalReasonIsEscaped() {
        let row = refusalRow(
            engine: "v2-compiled", reason: "skipped: bad | 999.9 | 999.9\nsecond line")
        #expect(cells(row).count == headerColumnCount)
        #expect(!row.contains("\n"))
    }

    // MARK: - Option parsing

    @Test("Qwen campaign options are strict and construction scoped")
    func qwenCampaignOptionsAreStrictAndConstructionScoped() throws {
        let options = try BenchOptions.parse([
            "--model", "/m",
            "--profile", "prefill",
            "--prompt-file", "/p",
            "--steps", "1024",
            "--prefill-chunk", "2048",
            "--solo-stripe", "2048",
            "--model-revision", "model-sha",
            "--mlx-swift-revision", "mlx-sha",
            "--gpu-lock-owner", "campaign-owner",
            "--receipt", "/r.json",
        ])

        #expect(options.profile == .prefill)
        #expect(options.promptFile == "/p")
        #expect(options.steps == 1_024)
        #expect(options.prefillChunk == 2_048)
        #expect(options.soloStripe == 2_048)
        #expect(options.mtpDepth == 0)
        #expect(options.modelRevision == "model-sha")
        #expect(options.mlxSwiftRevision == "mlx-sha")
        #expect(options.gpuLockOwner == "campaign-owner")
        #expect(options.receiptPath == "/r.json")

        #expect(throws: BenchOptionError.self) {
            _ = try BenchOptions.parse([
                "--model", "/m", "--profile", "stock", "--prompt-file", "/p",
                "--steps", "1024", "--prefill-chunk", "2048",
                "--model-revision", "m", "--mlx-swift-revision", "s",
                "--gpu-lock-owner", "o", "--receipt", "/r.json",
            ])
        }
        #expect(throws: BenchOptionError.self) {
            _ = try BenchOptions.parse([
                "--model", "/m", "--profile", "prefill", "--prompt-file", "/p",
                "--steps", "1024", "--mtp-depth", "1",
                "--model-revision", "m", "--mlx-swift-revision", "s",
                "--gpu-lock-owner", "o", "--receipt", "/r.json",
            ])
        }
    }

    @Test("output parity defaults to fast and parses byte exact")
    func outputParityParsesStrictly() throws {
        let defaults = try BenchOptions.parse(["--model", "/m"])
        #expect(defaults.outputParity == .fast)
        #expect(defaults.outputParityWasSpecified == false)

        let exact = try BenchOptions.parse([
            "--model", "/m", "--profile", "decode", "--mtp-depth", "2",
            "--output-parity", "byte-exact",
            "--prompt-file", "/p", "--steps", "1024", "--receipt", "/r.json",
            "--model-revision", "m", "--mlx-swift-revision", "s",
            "--gpu-lock-owner", "o",
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
            [
                "--model", "/m", "--profile", "decode", "--mtp-depth", "0",
                "--output-parity", "byte-exact",
            ],
        ] {
            #expect(throws: BenchOptionError.self) { _ = try BenchOptions.parse(arguments) }
        }
    }

    @Test("optimization flags require the one-prompt campaign route")
    func optimizationFlagsRejectOrdinaryMatrixRuns() {
        for arguments in [
            ["--model", "/m", "--profile", "prefill"],
            ["--model", "/m", "--profile", "decode", "--mtp-depth", "2"],
            ["--model", "/m", "--profile", "full", "--mtp-depth", "2"],
        ] {
            #expect(throws: BenchOptionError.self) {
                _ = try BenchOptions.parse(arguments)
            }
        }
    }

    @Test("campaign receipts accept the realistic 1024-token coding decode")
    func campaignReceiptModeIsExact() throws {
        #expect(throws: BenchOptionError.self) {
            _ = try BenchOptions.parse([
                "--model", "/m", "--steps", "100", "--prompt-file", "/p",
                "--model-revision", "m", "--mlx-swift-revision", "s",
                "--gpu-lock-owner", "o", "--receipt", "/r.json",
            ])
        }
        let options = try BenchOptions.parse([
            "--model", "/m", "--steps", "1024", "--prompt-file", "/p",
            "--model-revision", "m", "--mlx-swift-revision", "s",
            "--gpu-lock-owner", "o", "--receipt", "/r.json",
        ])
        #expect(options.steps == 1024)
        #expect(throws: BenchOptionError.self) {
            _ = try BenchOptions.parse([
                "--model", "/m", "--steps", "100", "--receipt", "/r.json",
            ])
        }
        for missing in ["--model-revision", "--mlx-swift-revision", "--gpu-lock-owner"] {
            var arguments = [
                "--model", "/m", "--steps", "100", "--prompt-file", "/p",
                "--model-revision", "m", "--mlx-swift-revision", "s",
                "--gpu-lock-owner", "o", "--receipt", "/r.json",
            ]
            let index = try #require(arguments.firstIndex(of: missing))
            arguments.removeSubrange(index ... index + 1)
            #expect(throws: BenchOptionError.self) { _ = try BenchOptions.parse(arguments) }
        }
    }

    @Test("campaign scheduler geometry is fixed at construction")
    func campaignSchedulerGeometryIsConstructionScoped() throws {
        let stock = try BenchOptions.parse(["--model", "/m"])
        let stockConfig = campaignSchedulerConfig(stock)
        #expect(stockConfig.maxConcurrentRequests == 1)
        #expect(stockConfig.prefillChunkSize == 512)
        #expect(stockConfig.soloPrefillStripeTokens == nil)

        var prefill = stock
        prefill.profile = .prefill
        prefill.prefillChunk = 2_048
        prefill.soloStripe = 2_048
        let tuned = campaignSchedulerConfig(prefill)
        #expect(tuned.maxConcurrentRequests == 1)
        #expect(tuned.maxBatchedTokensPerStep == 2_048)
        #expect(tuned.prefillChunkSize == 2_048)
        #expect(tuned.soloPrefillStripeTokens == 2_048)
    }

    @Test("campaign MTP depth is fixed at construction")
    func campaignMTPDepthIsConstructionScoped() throws {
        var stock = try BenchOptions.parse(["--model", "/m"])
        let disabled = campaignMTPConfig(stock)
        #expect(disabled.enabled == false)
        #expect(disabled.fixedDraftTokens == nil)

        stock.profile = .decode
        stock.mtpDepth = 3
        let decode = campaignMTPConfig(stock)
        #expect(decode.enabled)
        #expect(decode.maxDraftTokens == 3)
        #expect(decode.fixedDraftTokens == 3)
        #expect(decode.maxSpeculativeBatch == 1)
        #expect(decode.verificationMode == .rectangular)

        stock.outputParity = .byteExact
        let exact = campaignMTPConfig(stock)
        #expect(exact.verificationMode == .rectangularExact)
        #expect(BenchOutputParity.fast.verificationRoute
            == "rectangular-target-authoritative")
        #expect(BenchOutputParity.byteExact.verificationRoute
            == "rectangular-timewise-byte-exact")
    }

    @Test("byte exact preserves the installed fast construction profile")
    func byteExactStartsFromFastConstruction() throws {
        let fast = try BenchOptions.parse([
            "--model", "/m/qwen", "--profile", "full", "--mtp-depth", "2",
            "--output-parity", "fast",
            "--prompt-file", "/p", "--steps", "1024", "--receipt", "/r-fast.json",
            "--model-revision", "m", "--mlx-swift-revision", "s",
            "--gpu-lock-owner", "o",
        ])
        let exactFull = try BenchOptions.parse([
            "--model", "/m/qwen", "--profile", "full", "--mtp-depth", "2",
            "--output-parity", "byte-exact",
            "--prompt-file", "/p", "--steps", "1024", "--receipt", "/r-full.json",
            "--model-revision", "m", "--mlx-swift-revision", "s",
            "--gpu-lock-owner", "o",
        ])
        let exactDecode = try BenchOptions.parse([
            "--model", "/m/qwen", "--profile", "decode", "--mtp-depth", "2",
            "--output-parity", "byte-exact",
            "--prompt-file", "/p", "--steps", "1024", "--receipt", "/r-decode.json",
            "--model-revision", "m", "--mlx-swift-revision", "s",
            "--gpu-lock-owner", "o",
        ])

        #expect(qwen35CampaignConstructionProfile(fast) == .full)
        #expect(qwen35CampaignConstructionProfile(exactFull) == .full)
        #expect(qwen35CampaignConstructionProfile(exactDecode) == .decode)
    }

    @Test("campaign construction rejects installed verification drift")
    func campaignConstructionRejectsVerificationDrift() {
        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: 2, maxSpeculativeBatch: 1,
            fixedDraftTokens: 2, verificationMode: .serialTarget)
        var installed = CBv2MTPMetrics()
        installed.verificationMode = .rectangular

        #expect(throws: CampaignExecutionError.self) {
            try validateCampaignMTPInstallation(config: config, metrics: installed)
        }
    }

    @Test("campaign construction rejects a drafter-clamped fixed depth")
    func campaignConstructionRejectsDepthDrift() {
        #expect(throws: CampaignExecutionError.self) {
            try validateCampaignMTPDepth(requested: 2, drafterMaximum: 1)
        }
        #expect(throws: Never.self) {
            try validateCampaignMTPDepth(requested: 2, drafterMaximum: 4)
        }
    }

    @Test("campaign prompt loading preserves text bytes and records SHA-256")
    func campaignPromptIsExact() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("python.txt")
        let text = "Write a Python function that returns the nth Fibonacci number.\n"
        try Data(text.utf8).write(to: url)

        let prompt = try CampaignPrompt.load(from: url)
        #expect(prompt.text == text)
        #expect(prompt.sha256 == "cfcd09297db017afcada064c952b4625d20d1275dab4d58264cfd8f7bfa3a147")
    }

    @Test("campaign receipt records its requested decode length and accepts long contexts")
    func campaignReceiptIsStrict() throws {
        let metrics = PhaseMetrics(
            promptTokens: 65_536, submittedAt: 10, firstTokenAt: 10.1,
            lastTokenAt: 15.215, generatedTokens: 1_024)
        let receipt = CampaignReceipt(
            modelPath: "/models/qwen",
            modelRevision: String(repeating: "a", count: 40),
            sourceRevision: "abcdef0",
            mlxSwiftRevision: String(repeating: "b", count: 40),
            profile: .decode, promptSHA256: String(repeating: "c", count: 64),
            promptTokenIDs: Array(repeating: 7, count: 65_536),
            generatedTokenIDs: Array(0 ..< 1_024), requestedGeneratedTokens: 1_024,
            generatedText: "output", phaseMetrics: metrics, prefillChunk: 512,
            soloStripe: nil, mtpDepth: 2, outputParity: .byteExact,
            verificationRoute: BenchOutputParity.byteExact.verificationRoute,
            routeSummary: [
                "target": "optimized", "outputParity": "byte-exact",
                "verificationRoute": "rectangular-timewise-byte-exact",
                "mtp": "inline-fixed-k2,verify=rectangular_exact,rounds=35",
            ],
            gpuLockOwner: "owner", peakMemoryBytes: 123)

        try receipt.validate()
        var unsupported = receipt
        unsupported.schemaVersion = 999
        #expect(throws: CampaignReceiptError.self) { try unsupported.validate() }
        let json = try benchmarkJSONString(receipt)
        for key in [
            "modelRevision", "sourceRevision", "mlxSwiftRevision", "promptSHA256",
            "promptTokenIDs", "generatedTokenIDs", "phaseMetrics", "routeSummary",
            "gpuLockOwner", "peakMemoryBytes", "outputParity", "verificationRoute",
            "requestedGeneratedTokens",
        ] {
            #expect(json.contains("\"\(key)\""), "missing receipt key \(key)")
        }
        #expect(json.contains("\"outputParity\":\"byte-exact\""))
        #expect(json.contains(
            "\"verificationRoute\":\"rectangular-timewise-byte-exact\""))

        let mislabeled = CampaignReceipt(
            modelPath: receipt.modelPath, modelRevision: receipt.modelRevision,
            sourceRevision: receipt.sourceRevision, mlxSwiftRevision: receipt.mlxSwiftRevision,
            profile: receipt.profile, promptSHA256: receipt.promptSHA256,
            promptTokenIDs: receipt.promptTokenIDs,
            generatedTokenIDs: receipt.generatedTokenIDs,
            requestedGeneratedTokens: receipt.requestedGeneratedTokens,
            generatedText: receipt.generatedText, phaseMetrics: metrics,
            prefillChunk: receipt.prefillChunk, soloStripe: receipt.soloStripe,
            mtpDepth: receipt.mtpDepth, outputParity: .byteExact,
            verificationRoute: "serial-byte-exact",
            routeSummary: [
                "outputParity": "byte-exact", "verificationRoute": "serial-byte-exact",
                "mtp": "inline-fixed-k2,verify=rectangular,rounds=34",
            ],
            gpuLockOwner: receipt.gpuLockOwner, peakMemoryBytes: receipt.peakMemoryBytes)
        #expect(throws: CampaignReceiptError.self) { try mislabeled.validate() }

        let prefixCollision = CampaignReceipt(
            modelPath: receipt.modelPath, modelRevision: receipt.modelRevision,
            sourceRevision: receipt.sourceRevision, mlxSwiftRevision: receipt.mlxSwiftRevision,
            profile: receipt.profile, promptSHA256: receipt.promptSHA256,
            promptTokenIDs: receipt.promptTokenIDs,
            generatedTokenIDs: receipt.generatedTokenIDs,
            requestedGeneratedTokens: receipt.requestedGeneratedTokens,
            generatedText: receipt.generatedText, phaseMetrics: metrics,
            prefillChunk: receipt.prefillChunk, soloStripe: receipt.soloStripe,
            mtpDepth: receipt.mtpDepth, outputParity: .fast,
            verificationRoute: BenchOutputParity.fast.verificationRoute,
            routeSummary: [
                "outputParity": "fast",
                "verificationRoute": BenchOutputParity.fast.verificationRoute,
                "mtp": "inline-fixed-k2,verify=rectangular_exact,rounds=34",
            ],
            gpuLockOwner: receipt.gpuLockOwner, peakMemoryBytes: receipt.peakMemoryBytes)
        #expect(throws: CampaignReceiptError.self) { try prefixCollision.validate() }

        let short = CampaignReceipt(
            modelPath: receipt.modelPath, modelRevision: receipt.modelRevision,
            sourceRevision: receipt.sourceRevision, mlxSwiftRevision: receipt.mlxSwiftRevision,
            profile: receipt.profile, promptSHA256: receipt.promptSHA256,
            promptTokenIDs: receipt.promptTokenIDs,
            generatedTokenIDs: Array(receipt.generatedTokenIDs.dropLast()),
            requestedGeneratedTokens: receipt.requestedGeneratedTokens,
            generatedText: receipt.generatedText, phaseMetrics: metrics,
            prefillChunk: receipt.prefillChunk, soloStripe: receipt.soloStripe,
            mtpDepth: receipt.mtpDepth, outputParity: receipt.outputParity,
            verificationRoute: receipt.verificationRoute, routeSummary: receipt.routeSummary,
            gpuLockOwner: receipt.gpuLockOwner, peakMemoryBytes: receipt.peakMemoryBytes)
        #expect(throws: CampaignReceiptError.self) { try short.validate() }
    }

    @Test("phase metrics exclude the first token from decode")
    func phaseMetricsExcludeFirstTokenFromDecode() {
        let metrics = PhaseMetrics(
            promptTokens: 100,
            submittedAt: 10,
            firstTokenAt: 10.2,
            lastTokenAt: 10.695,
            generatedTokens: 100)

        #expect(abs(metrics.prefillSeconds - 0.2) < 1e-12)
        #expect(abs(metrics.decodeTPS - 200.0) < 0.001)
    }

    @Test("malformed prompt-length lists are rejected, never silently dropped")
    func malformedPromptLengthsAreRejected() {
        // `100O0` is the review's example: a capital O for a zero. The old
        // compactMap/filter pair dropped it and ran the default 500/1500 mix
        // under a header that still claimed the requested axis.
        for raw in ["100O0", "0", "-5", "", "500,", ",500", "500,,1500", "500,abc", "1 000", "1_000"] {
            #expect(throws: BenchOptionError.self) {
                _ = try BenchOptions.parse(["--model", "/tmp/m", "--prompt-lengths", raw])
            }
        }
    }

    @Test("a rejected prompt-length list never degrades to the default mix")
    func rejectedListDoesNotBecomeTheDefaultAxis() throws {
        let defaults = try BenchOptions.parse(["--model", "/tmp/m"])
        #expect(defaults.promptLengths.isEmpty)
        #expect(defaults.promptAxisDescription.hasPrefix("default mix"))

        // The failure mode being gated: parsing must not *return* the default
        // axis for a malformed request.
        let parsed = try? BenchOptions.parse(["--model", "/tmp/m", "--prompt-lengths", "100O0"])
        #expect(parsed == nil)
    }

    @Test("a valid prompt-length list is preserved and sizes the paged pool")
    func validPromptLengthsSizeThePool() throws {
        let options = try BenchOptions.parse([
            "--model", "/tmp/m", "--prompt-lengths", "500, 10000", "--steps", "128",
        ])
        #expect(options.promptLengths == [500, 10_000])
        #expect(options.longestPrompt == 10_000)
        #expect(options.pagedNominalMaxSequenceLength == 10_128)
        #expect(options.promptAxisDescription == "500,10000")

        // No new flags → the historical 4096 sizing, unchanged.
        let legacyDefault = try BenchOptions.parse(["--model", "/tmp/m"])
        #expect(legacyDefault.pagedNominalMaxSequenceLength == 4_096)
    }

    @Test("other numeric options are equally strict")
    func otherNumericOptionsAreStrict() {
        for argument in [
            ["--batches", "1,2,x"], ["--batches", "0"], ["--steps", "12S"],
            ["--steps", "0"], ["--kv-gb", "-1"], ["--max-seq-len", "abc"],
        ] {
            #expect(throws: BenchOptionError.self) {
                _ = try BenchOptions.parse(["--model", "/tmp/m"] + argument)
            }
        }
    }

    @Test("a missing option value is an error, not a silent default")
    func missingValueIsAnError() {
        #expect(throws: BenchOptionError.missingValue(option: "--prompt-lengths")) {
            _ = try BenchOptions.parse(["--model", "/tmp/m", "--prompt-lengths"])
        }
        #expect(throws: BenchOptionError.unknownOption("--engnies")) {
            _ = try BenchOptions.parse(["--model", "/tmp/m", "--engnies", "v2"])
        }
        #expect(throws: BenchOptionError.missingModel) {
            _ = try BenchOptions.parse(["--steps", "8"])
        }
        // A mistyped mode used to run neither correctness nor perf and emit an
        // empty but successful-looking report.
        #expect(throws: BenchOptionError.self) {
            _ = try BenchOptions.parse(["--model", "/tmp/m", "--mode", "pref"])
        }
    }

    @Test("--print-revision needs no model")
    func printRevisionNeedsNoModel() throws {
        let options = try BenchOptions.parse(["--print-revision"])
        #expect(options.printRevisionOnly)
    }

    // MARK: - Invocation provenance

    /// Split the argv `/bin/sh` recovers from a recorded invocation line.
    private func shellSplit(_ commandLine: String) throws -> [String] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", "printf '%s\\0' \(commandLine)"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        var fields = data.split(separator: 0, omittingEmptySubsequences: false)
            .map { String(decoding: $0, as: UTF8.self) }
        // Every element is NUL-terminated, so the split leaves a trailing "".
        if fields.last == "" { fields.removeLast() }
        return fields
    }

    @Test("an argv containing whitespace round-trips through the recorded invocation")
    func invocationRoundTripsWhitespace() throws {
        let argv = [
            "/tmp/build dir/BenchCBv2",
            "--model", "/Volumes/My Disk/models/gemma 4 e4b",
            "--label", "paged vs contiguous",
            "--prompt-lengths", "500,10000",
            "--out", "report 2026-07-25.md",
        ]
        let recorded = shellQuotedInvocation(argv)
        #expect(try shellSplit(recorded) == argv)
        // The old `joined(separator: " ")` form is exactly what this replaces.
        #expect(try shellSplit(argv.joined(separator: " ")) != argv)
    }

    @Test("hostile argv elements round-trip too")
    func invocationRoundTripsHostileArguments() throws {
        let argv = [
            "BenchCBv2",
            "--label", "it's a \"quoted\" tag",
            "--label", "$HOME `id` $(rm -rf /) ${x}",
            "--label", "tab\there",
            "--label", "line\nbreak",
            "--label", "back\\slash",
            "--label", "glob*?[]",
            "--label", "",
            "--label", "semi;colon&amp|pipe",
        ]
        #expect(try shellSplit(shellQuotedInvocation(argv)) == argv)
    }

    @Test("plain arguments stay unquoted so the common report is readable")
    func plainArgumentsAreNotQuoted() {
        #expect(shellQuotedInvocation(["BenchCBv2", "--steps", "128"])
            == "BenchCBv2 --steps 128")
        #expect(shellQuote("/tmp/models/gemma-4-e4b") == "/tmp/models/gemma-4-e4b")
        #expect(shellQuote("") == "''")
    }

    @Test("the report records the shell-escaped invocation, not a flattened one")
    func headerRecordsEscapedInvocation() throws {
        let argv = ["BenchCBv2", "--model", "/m/gemma 4", "--label", "a b"]
        let options = try BenchOptions.parse(Array(argv.dropFirst()))
        let header = reportHeader(
            options: options, argv: argv, chip: "Apple M4 Max", ramGB: 128,
            osVersion: "Version 26.0", hostLine: "load avg (1m) 0.4 / 16 cores",
            revision: "abc1234", date: Date(timeIntervalSince1970: 0))
        let invocationRow = try #require(
            header.split(separator: "\n").first { $0.hasPrefix("| Invocation |") })
        let recorded = String(invocationRow)
            .replacingOccurrences(of: "| Invocation | `", with: "")
            .replacingOccurrences(of: "` |", with: "")
        #expect(try shellSplit(recorded) == argv)
        #expect(header.contains("| Label | a b |"))
    }

    @Test("the report records the installed output parity route")
    func headerRecordsOutputParityRoute() throws {
        let argv = [
            "BenchCBv2", "--model", "/m/qwen", "--profile", "decode",
            "--mtp-depth", "2", "--output-parity", "byte-exact",
            "--prompt-file", "/p", "--steps", "1024", "--receipt", "/r.json",
            "--model-revision", "m", "--mlx-swift-revision", "s",
            "--gpu-lock-owner", "o",
        ]
        let options = try BenchOptions.parse(Array(argv.dropFirst()))
        let header = reportHeader(
            options: options, argv: argv, chip: "Apple M5 Max", ramGB: 128,
            osVersion: "Version 26.0", hostLine: "load avg (1m) 0.4 / 16 cores",
            revision: "abc1234", date: Date(timeIntervalSince1970: 0))

        #expect(header.contains("| Output parity | byte-exact |"))
        #expect(header.contains(
            "| Verification route | rectangular-timewise-byte-exact |"))
    }

    // MARK: - Build revision provenance

    @Test("the report stamps the injected build revision verbatim")
    func headerStampsInjectedRevision() throws {
        let options = try BenchOptions.parse(["--model", "/tmp/m"])
        let header = reportHeader(
            options: options, argv: ["BenchCBv2", "--model", "/tmp/m"],
            chip: "Apple M4 Max", ramGB: 128, osVersion: "Version 26.0",
            hostLine: "load avg (1m) 0.4 / 16 cores", revision: "deadbee-dirty",
            date: Date(timeIntervalSince1970: 0))
        #expect(header.contains("| mlx-swift-lm (build) | deadbee-dirty |"))
        // Nothing in the header may re-derive a revision of its own.
        #expect(!header.contains("| mlx-swift-lm |"))
    }

    @Test("the build revision is a compile-time constant, not a runtime git query")
    func buildRevisionIsStampedAtBuildTime() {
        let revision = buildRevision()
        #expect(revision == BenchBuildRevision.value)
        #expect(!revision.isEmpty)

        // Well-formed: `unknown`, or a short SHA optionally marked dirty. The
        // stamp is produced by scripts/stamp-bench-revision.sh at build time.
        let pattern = /^(unknown|[0-9a-f]{7,40}(-dirty)?)$/
        #expect(
            revision.wholeMatch(of: pattern) != nil,
            "unexpected build revision stamp: \(revision)")

        // Invariant to the process's working directory: the value is baked in,
        // so it cannot pick up the git state of wherever the binary is run.
        let original = FileManager.default.currentDirectoryPath
        defer { FileManager.default.changeCurrentDirectoryPath(original) }
        #expect(FileManager.default.changeCurrentDirectoryPath("/tmp"))
        #expect(buildRevision() == revision)
    }

    @Test("Qwen campaign evidence does not publish a macOS operator home")
    func qwenEvidenceScrubsOperatorHomePaths() throws {
        let repository = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let roots = [
            repository.appendingPathComponent("benchmarks/qwen36-a3b"),
            repository.appendingPathComponent("docs/plans"),
            repository.appendingPathComponent("docs/specs"),
            repository.appendingPathComponent("scripts"),
        ]
        let plainPrefix = Data("/Users/".utf8)
        let escapedPrefix = Data("\\/Users\\/".utf8)
        for root in roots {
            guard let files = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: [.isRegularFileKey])
            else { continue }
            for case let file as URL in files {
                guard try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
                else { continue }
                if root.lastPathComponent != "qwen36-a3b",
                    !file.lastPathComponent.lowercased().contains("qwen36")
                {
                    continue
                }
                let data = try Data(contentsOf: file, options: .mappedIfSafe)
                #expect(
                    data.range(of: plainPrefix) == nil
                        && data.range(of: escapedPrefix) == nil,
                    "operator home path leaked in \(file.path)")
            }
        }
    }
}
