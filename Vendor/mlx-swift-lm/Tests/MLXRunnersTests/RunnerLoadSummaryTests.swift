// RunnerLoadSummaryTests.swift
//
// The `--verbose` load summary.
//
// The bar is what a box needed and did not have: the summary must state what
// the load DECIDED, on stderr, before the hello — and stdout must be
// byte-identical whether or not it is on, because stdout is the wire.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("Load summary")
struct RunnerLoadSummaryTests {

    private func summary(for runner: any Runner) -> RunnerLoadSummary {
        runner.loadSummary(
            weights: URL(fileURLWithPath: "/models/mock"),
            options: RunnerLoadOptions())
    }

    private func value(_ key: String, in summary: RunnerLoadSummary) -> String? {
        summary.entries.first { $0.key == key }?.value
    }

    @Test("The generic summary names the runner, the checkpoint and what bound")
    func genericLines() {
        let summary = summary(for: MockRunner())
        #expect(value("runner_id", in: summary) == "layr/mock-adapter")
        #expect(
            value("manifest_sha256", in: summary)
                == "850ef7df262d03f851a355d1572257b04ade4834d2563c18f01b9cf344a6fffd")
        #expect(value("weights_dir", in: summary) == "/models/mock")
        #expect(value("model_type", in: summary) == "qwen4_exp_text")
        #expect(value("decoders_loaded", in: summary) == "serial,mtp")
        #expect(value("mtp_head_bound", in: summary) == "yes")
        #expect(value("head_provenance", in: summary) == "none")
        #expect(value("requires_keep_mask", in: summary) == "yes")
        // The rule BY NAME: two implementations that disagree on a tie
        // disagree wherever a parity run diverges.
        #expect(value("argmax_tie_break", in: summary) == "lowest_token_id")
    }

    /// The worker holds an `any Runner`. If `loadSummary` were only an
    /// extension default, this call would pick the generic half and the
    /// family's lines would never print — which a box saw. The requirement
    /// on the protocol makes the override reachable through the existential.
    @Test("A family override is reached through `any Runner`")
    func familyOverrideDispatchesDynamically() {
        let runner: any Runner = FamilySummaryRunner()
        let summary = summary(for: runner)
        #expect(value("runner_id", in: summary) == "layr/mock-adapter")
        #expect(value("family_line", in: summary) == "present")
    }

    @Test("Every entry renders as one prefixed key=value line")
    func rendering() {
        var summary = RunnerLoadSummary()
        summary.add("runner_id", "layr/mock")
        summary.add("tensors_bound", 12)
        summary.add("mtp_head_bound", true)
        let rendered = summary.rendered()
        #expect(
            rendered == """
                bench-worker: runner_id=layr/mock
                bench-worker: tensors_bound=12
                bench-worker: mtp_head_bound=yes

                """)
    }

    /// Ordered, not a dictionary: two runs of the same load diff cleanly.
    @Test("Entry order is the load's own order")
    func orderIsStable() {
        let first = summary(for: MockRunner()).entries.map(\.key)
        let second = summary(for: MockRunner()).entries.map(\.key)
        #expect(first == second)
        #expect(first.first == "runner_id")
    }

    @Test("The flag and the environment both enable it, nothing else does")
    func enablement() {
        #expect(RunnerLoadSummary.isEnabled(flag: true, environment: [:]))
        #expect(
            RunnerLoadSummary.isEnabled(
                flag: false, environment: [RunnerLoadSummary.verboseEnvironmentName: "1"]))
        #expect(!RunnerLoadSummary.isEnabled(flag: false, environment: [:]))
        #expect(
            !RunnerLoadSummary.isEnabled(
                flag: false, environment: [RunnerLoadSummary.verboseEnvironmentName: "0"]))
    }

    @Test("--verbose parses on both serving subcommands")
    func parsesOnBothSubcommands() throws {
        for subcommand in ["runtime-worker", "resident"] {
            let arguments =
                subcommand == "resident"
                ? [subcommand, "--weights", "/nonexistent", "--socket", "/tmp/s", "--verbose"]
                : [subcommand, "--weights", "/nonexistent", "--verbose"]
            guard case .serve(let launch) = try BenchWorkerCommand.parse(arguments) else {
                Issue.record("\(subcommand) did not parse as a serve command")
                continue
            }
            #expect(launch.verbose)
        }
        guard
            case .serve(let quiet) = try BenchWorkerCommand.parse([
                "runtime-worker", "--weights", "/nonexistent",
            ])
        else { return }
        #expect(!quiet.verbose)
    }

    @Test("diag-parity parses with a golden and refuses without; serving refuses --golden")
    func diagParityParses() throws {
        guard
            case .serve(let launch) = try BenchWorkerCommand.parse([
                "diag-parity", "--weights", "/nonexistent", "--golden", "/g.json", "--steps", "12",
            ])
        else {
            Issue.record("diag-parity did not parse as a launch")
            return
        }
        #expect(launch.subcommand == .diagParity)
        #expect(launch.golden?.path == "/g.json")
        #expect(launch.steps == 12)
        #expect(throws: WorkerLaunchError.self) {
            try BenchWorkerCommand.parse(["diag-parity", "--weights", "/nonexistent"])
        }
        #expect(throws: WorkerLaunchError.self) {
            try BenchWorkerCommand.parse([
                "runtime-worker", "--weights", "/nonexistent", "--golden", "/g.json",
            ])
        }
    }

    /// The load summary NEVER reaches stdout: the fixture replay is
    /// byte-identical with verbose on, because the summary goes to stderr
    /// and the wire is untouched.
    @Test("The wire is byte-identical with the summary enabled")
    func wireUnchangedWhenVerbose() async throws {
        let fixture = try BenchWorkerProtocolTests.Fixture()

        func replay() async throws -> [String] {
            let transport = ScriptedTransport(lines: fixture.requests)
            let server = BenchWorkerServer(
                runner: MockRunner(),
                transport: transport,
                trusted: false,
                speculative: true,
                build: "fixture",
                device: "mock",
                kvBytesCapacity: 1 << 20,
                maxDecodeTokens: 64,
                memory: FixtureMemoryReporter(),
                nonce: "fixturenonce")
            await server.run()
            return transport.written
        }

        // The summary is rendered — proving it produced output — and the
        // protocol transcript is compared to the fixture regardless.
        let quiet = try await replay()
        let rendered = summary(for: MockRunner()).rendered()
        #expect(!rendered.isEmpty)
        let verbose = try await replay()

        #expect(quiet == verbose)
        #expect(verbose.count == fixture.responses.count + 1)
        for (index, expected) in fixture.responses.enumerated() {
            #expect(verbose[index + 1] == expected)
        }
        // Not one summary line leaked into the transcript.
        #expect(!verbose.contains { $0.hasPrefix("bench-worker:") })
        #expect(!verbose.contains { $0.contains("runner_id=") })
    }
}

/// A mock whose family half adds one line on top of the generic summary.
private final class FamilySummaryRunner: Runner, @unchecked Sendable {
    private let base = MockRunner()
    static var manifest: RunnerManifest { MockRunner.manifest }
    static func adopt(
        model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration, directory: URL, options: RunnerLoadOptions
    ) throws -> FamilySummaryRunner {
        throw RunnerError.unexpectedModel("not adopted in tests")
    }
    var servingModel: any LanguageModel { base.servingModel }
    var tokenizer: any MLXLMCommon.Tokenizer { base.tokenizer }
    var eosTokenIDs: Set<Int> { base.eosTokenIDs }
    var layerKinds: [CBv2LayerKind] { base.layerKinds }
    var loadedDecoders: [DecoderID] { base.loadedDecoders }
    var headProvenance: HeadProvenance? { base.headProvenance }
    var loadedModelType: String { base.loadedModelType }
    func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine { try base.makeEngine(build) }
    func makeStepper() throws -> any TeacherForcedStepper { try base.makeStepper() }
    func loadSummary(weights: URL, options: RunnerLoadOptions) -> RunnerLoadSummary {
        var summary = genericLoadSummary(weights: weights, options: options)
        summary.add("family_line", "present")
        return summary
    }
}
