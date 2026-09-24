// MockRunner.swift
//
// A scripted `Runner` for the protocol tests: no weights, no Metal, no MLX
// allocator. Its token outputs are a pure function of the input, chosen to
// reproduce the shared conformance fixture
// (`Resources/engine-wire-v1-adapter.ndjson`, pinned on both sides).

import Foundation
import MLX
import MLXLMCommon
import MLXRunners

// MARK: - Token rules

enum MockTokens {
    /// Which answer block a prompt lands in.
    ///
    /// In the fixture, a prompt whose first token is in the 20s or 50s is a
    /// DECODE seed (`decode_begin`, `free_decode_begin`) and answers in the
    /// 200000 block; every other prompt is a prefill/correctness anchor and
    /// answers in the 100000 block.
    static func block(_ prompt: [Int]) -> Int {
        guard let first = prompt.first else { return 100_000 }
        return [2, 5].contains(first / 10) ? 200_000 : 100_000
    }

    /// A whole-prompt forward answers the block base plus the prompt length.
    static func promptAnswer(_ prompt: [Int]) -> Int {
        block(prompt) + prompt.count
    }

    /// A single forced token answers itself plus one.
    static func stepAnswer(_ token: Int) -> Int { token + 1 }

    /// First token an engine free-run emits.
    ///
    /// A decode window has already answered its seed forward
    /// (`promptAnswer`), so its stream continues from there; a `correctness`
    /// window has no seed response and starts at the block base.
    static func freeRunStart(_ prompt: [Int]) -> Int {
        block(prompt) == 200_000 ? promptAnswer(prompt) : block(prompt)
    }
}

// MARK: - Stepper

final class MockStepper: TeacherForcedStepper {
    private(set) var forwards = 0
    private var begun = false

    func begin() throws {
        begun = true
        forwards = 0
    }

    func forward(_ tokens: [Int]) throws -> StepOutput {
        guard begun else { throw StepperError.notBegun }
        guard !tokens.isEmpty else { throw StepperError.emptyForward }
        forwards += 1
        let argmax =
            tokens.count == 1
            ? MockTokens.stepAnswer(tokens[0])
            : MockTokens.promptAnswer(tokens)
        // A descending ladder from 10.0, so the top-logit margin is 1.0 and
        // the ordering is unambiguous.
        let top = (0 ..< 8).map { index in
            (token: argmax + index, logit: 10.0 - Double(index))
        }
        return StepOutput(argmax: argmax, topLogits: top, margin: 1.0)
    }
}

// MARK: - Engine

/// The per-round journal a scripted engine replays for one free-run window.
///
/// Written as records, not as summary numbers: the worker derives every
/// audit field from `CBv2MTPMetrics.roundAudits`, so the mock has to exercise
/// that same path or the test proves nothing about it.
struct MockRoundScript: Sendable {
    /// One round: draft depth, the pre-min accept-walk length, and the
    /// committed width.
    struct Round: Sendable {
        var k: Int
        var accepted: Int
        var confirmed: Int
    }

    var rounds: [Round]
    /// Rows the rounds are attributed to, in SLOT ORDER. bench-worker submits
    /// slot s as request id s + 1.
    var requestIDs: [UInt64] = [1]

    /// The fixture's window: three rounds at depth 2, committing 3, 1 and 2
    /// tokens, with accept walks of 2, 1 and 1. That gives the fixture's
    /// `acceptance_lengths [3,1,2]`, `drafted_total 6`, `accepted_total 4`
    /// (the sum of `min(accepted, confirmed)`) and `committed_total 6`.
    static let fixtureWindow = MockRoundScript(rounds: [
        Round(k: 2, accepted: 2, confirmed: 3),
        Round(k: 2, accepted: 1, confirmed: 1),
        Round(k: 2, accepted: 1, confirmed: 2),
    ])

    /// The journal in finalize order: one record per (row, round).
    var journal: [CBv2MTPRoundAuditRecord] {
        rounds.enumerated().flatMap { roundIndex, round in
            requestIDs.map { requestID in
                CBv2MTPRoundAuditRecord(
                    requestID: requestID,
                    k: round.k,
                    draftTokens: Array(repeating: 0, count: round.k),
                    targetTokens: Array(repeating: 0, count: 1 + round.k),
                    accepted: round.accepted,
                    confirmed: round.confirmed,
                    rejected: (1 + round.k) - round.confirmed,
                    tokensCountAfter: roundIndex + 1,
                    numComputedAfter: roundIndex,
                    generatedAfter: roundIndex + 1,
                    finishReason: nil)
            }
        }
    }
}

final class MockEngine: CBv2Engine, CBv2MTPCountersReporting, @unchecked Sendable {
    private let script: MockRoundScript?
    private let lock = NSLock()
    private var reads = 0

    init(script: MockRoundScript?) {
        self.script = script
    }

    func submit(_ request: CBv2Request) throws -> AsyncStream<CBv2Event> {
        let start = MockTokens.freeRunStart(request.promptTokens)
        let maxTokens = request.maxTokens
        return AsyncStream { continuation in
            for index in 0 ..< maxTokens {
                continuation.yield(
                    .delta(text: "", tokens: [start + index], logprobs: nil))
            }
            continuation.yield(
                .finished(
                    reason: .length,
                    usage: CBv2Usage(
                        promptTokens: request.promptTokens.count,
                        completionTokens: maxTokens)))
            continuation.finish()
        }
    }

    func cancel(_ id: CBv2RequestID) {}

    func capacity() -> CBv2CapacitySnapshot {
        CBv2CapacitySnapshot(
            activeRequests: 0, waitingRequests: 0, kvBytesInUse: 0,
            kvBytesCapacity: 0, kvBytesBackendCapacity: 0, kvBytesReserved: 0,
            activeTokens: 0)
    }

    func shutdown() async {}

    /// Cumulative and monotonic, like the real engine's: the FIRST read is
    /// the baseline `free_decode_begin` takes after its seed forward, when no
    /// verify round has finalized yet; every read after it is the drained
    /// window's journal.
    func mtpMetricsSnapshot() -> CBv2MTPMetrics? {
        guard let script else { return nil }
        lock.lock()
        defer { lock.unlock() }
        defer { reads += 1 }
        var metrics = CBv2MTPMetrics()
        guard reads > 0 else { return metrics }
        metrics.roundAudits = script.journal
        metrics.rounds = script.rounds.count
        return metrics
    }
}

// MARK: - Runner

/// Manifests the CONTRACT itself pins, built here from the document.
///
/// The section 11 manifest for Qwen 3.8 Flash-Next is the cross-repo digest
/// vector, and the shared conformance fixture's hello carries its runner
/// identity. It is the runner's own manifest: the fixture is driven over the
/// value the fork serves, not over a copy of it.
enum ContractManifests {
    /// The runner's own manifest. One declaration: a test that pins the
    /// digest and a fixture hello driven over this value cannot drift apart
    /// from what the fork actually serves.
    static let sectionEleven = Qwen4ExpRunner.manifest
}

/// The mock adapter's manifest, DECODED from the file benchd checked in
/// beside the fixture (`Resources/engine-wire-v1-adapter.manifest.json`).
///
/// Decoded rather than declared in Swift on purpose: both sides load the
/// SAME bytes, so the hello's `manifest_sha256` is a digest this repo
/// actually computed over the other repo's file, not two declarations that
/// happen to agree today.
enum FixtureManifest {
    static let mockAdapter: RunnerManifest = {
        guard
            let url = Bundle.module.url(
                forResource: "engine-wire-v1-adapter.manifest", withExtension: "json"),
            let data = try? Data(contentsOf: url),
            let manifest = try? JSONDecoder().decode(RunnerManifest.self, from: data)
        else {
            fatalError("the shared mock-adapter manifest is missing or undecodable")
        }
        return manifest
    }()
}

/// The scripted runner the conformance fixture is driven over.
///
/// Its manifest is the shared mock-adapter file, so `hello.backend` and
/// `hello.runner.manifest_sha256` are two readings of ONE declaration —
/// which is what §6.1 requires and what a hand-written hello field would
/// break.
final class MockRunner: Runner, @unchecked Sendable {

    static var manifest: RunnerManifest { FixtureManifest.mockAdapter }

    /// The window audit the scripted engine replays.
    let script: MockRoundScript?

    var servingModel: any LanguageModel {
        preconditionFailure("mock runner holds no model")
    }
    var tokenizer: any MLXLMCommon.Tokenizer {
        preconditionFailure("mock runner holds no tokenizer")
    }
    let eosTokenIDs: Set<Int> = []
    let layerKinds: [CBv2LayerKind] = []
    let loadedDecoders: [DecoderID] = [.serial, .mtp]
    let headProvenance: HeadProvenance? = nil
    let loadedModelType = "qwen4_exp_text"

    init(script: MockRoundScript? = .fixtureWindow) {
        self.script = script
    }

    /// The mock holds no module, so adoption is the whole of its
    /// construction — but it still reads the two checkpoint facts every real
    /// runner reads, so a directory holding only those two files is enough
    /// for it too.
    static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> MockRunner {
        _ = try RunnerCheckpoint.modelType(at: directory)
        return MockRunner()
    }

    func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        guard loadedDecoders.contains(build.decoder) else {
            throw RunnerError.decoderNotLoaded(
                requested: build.decoder.rawValue,
                loaded: loadedDecoders.map(\.rawValue))
        }
        return MockEngine(script: build.decoder == .serial ? nil : script)
    }

    func makeStepper() throws -> any TeacherForcedStepper { MockStepper() }
}

/// The same mock with a COHORT free-run regime added, so the batched verbs
/// and the cohort-only audit fields can be driven. Its manifest is not the
/// shared fixture's, and nothing in the byte-for-byte replay uses it.
final class CohortMockRunner: Runner, @unchecked Sendable {

    static let manifest: RunnerManifest = {
        let base = MockRunner.manifest
        return RunnerManifest(
            schemaVersion: base.schemaVersion,
            runnerID: "layr/mock-cohort",
            modelTypes: ["mock-cohort"],
            backend: base.backend,
            engine: base.engine,
            kvBackends: base.kvBackends,
            decoders: base.decoders,
            regimes: base.regimes + [
                RegimeDeclaration(batch: .upTo(8), timing: .freeRun, perStreamTiming: false)
            ],
            multimodal: base.multimodal,
            recurrentLayers: base.recurrentLayers,
            requiresKeepMask: base.requiresKeepMask)
    }()

    private let inner: MockRunner

    init(script: MockRoundScript?) {
        self.inner = MockRunner(script: script)
    }

    var servingModel: any LanguageModel { inner.servingModel }
    var tokenizer: any MLXLMCommon.Tokenizer { inner.tokenizer }
    var eosTokenIDs: Set<Int> { inner.eosTokenIDs }
    var layerKinds: [CBv2LayerKind] { inner.layerKinds }
    var loadedDecoders: [DecoderID] { inner.loadedDecoders }
    var headProvenance: HeadProvenance? { inner.headProvenance }
    var loadedModelType: String { inner.loadedModelType }

    static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> CohortMockRunner {
        CohortMockRunner(script: .fixtureWindow)
    }

    func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        try inner.makeEngine(build)
    }
    func makeStepper() throws -> any TeacherForcedStepper { try inner.makeStepper() }
}

/// The same mock with every FREE-RUN regime removed. A worker over this one
/// must REFUSE `free_decode_begin` rather than serve a regime the manifest
/// does not declare (contract §6.2 rule 3).
final class TeacherForcedOnlyMockRunner: Runner, @unchecked Sendable {

    static let manifest = RunnerManifest(
        runnerID: "layr/mock-teacher-forced",
        modelTypes: ["mock-teacher-forced"],
        engine: MockRunner.manifest.engine,
        kvBackends: [.contiguous],
        decoders: MockRunner.manifest.decoders,
        regimes: [
            RegimeDeclaration(batch: .single, timing: .teacherForced, perStreamTiming: false)
        ],
        multimodal: false,
        recurrentLayers: false,
        requiresKeepMask: false)

    private let inner = MockRunner()

    var servingModel: any LanguageModel { inner.servingModel }
    var tokenizer: any MLXLMCommon.Tokenizer { inner.tokenizer }
    var eosTokenIDs: Set<Int> { inner.eosTokenIDs }
    var layerKinds: [CBv2LayerKind] { inner.layerKinds }
    var loadedDecoders: [DecoderID] { inner.loadedDecoders }
    var headProvenance: HeadProvenance? { inner.headProvenance }
    var loadedModelType: String { inner.loadedModelType }

    static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> TeacherForcedOnlyMockRunner {
        TeacherForcedOnlyMockRunner()
    }

    func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        try inner.makeEngine(build)
    }
    func makeStepper() throws -> any TeacherForcedStepper { try inner.makeStepper() }
}

// MARK: - Transport

/// Scripted transport: the request lines go in, the response lines come out.
final class ScriptedTransport: BenchWorkerTransport {
    private var pending: [String]
    private(set) var written: [String] = []

    init(lines: [String]) {
        self.pending = lines
    }

    func readLine() -> String? {
        pending.isEmpty ? nil : pending.removeFirst()
    }

    func write(line: String) {
        written.append(line)
    }
}

/// A worker with no MLX allocator: it reports the peak RSS the fixture
/// pins and nothing else. Absence is a real answer — the schema reads an
/// absent `cache_memory` as "not asserted", which is a different claim
/// from zero.
struct FixtureMemoryReporter: WorkerMemoryReporter {
    func preDrainSnapshot() -> (active: Int, cache: Int, peak: Int)? { nil }
    func drain() {}
    func cacheMemoryAfterDrain() -> Int? { nil }
    func peakRAMGB() -> Double? { 18.5 }
}

// MARK: - Stand-ins for the adoption seam

/// A tokenizer that answers only what adoption asks of it.
struct StubTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [[String: any Sendable]], chatTemplate: String
    ) throws -> [Int] { [] }
    func applyChatTemplate(
        messages: [[String: any Sendable]], chatTemplate: String,
        tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// A resident drafter: identity is all the adoption seam needs of it. Every
/// engine-facing member traps, because adoption binds a drafter and never
/// drives one.
final class StubDrafter: CBv2MTPDrafter, @unchecked Sendable {
    var mtpTargetIdentity: ObjectIdentifier? { nil }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        preconditionFailure("the stub drafter never drafts")
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("the stub drafter never drafts")
    }
}
