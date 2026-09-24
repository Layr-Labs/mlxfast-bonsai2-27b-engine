// BenchWorkerProtocolTests.swift
//
// Engine Protocol v1 conformance for `BenchWorkerServer`, driven in process
// over a scripted runner. No weights, no Metal, no MLX allocator.
//
// The load-bearing case replays the SHARED fixture
// (`Resources/engine-wire-v1-adapter.ndjson`, pinned identically on the
// benchd side) and compares every response line BYTE FOR BYTE, key order
// included. A wire that merely parses the same is not the same wire: key
// order, omitted-vs-null, and the `0.0`/`10.0` rendering of a JSON number
// are all things one side can get wrong while still round-tripping.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("bench-worker Engine Protocol v1")
struct BenchWorkerProtocolTests {

    /// The fixture split into its request lines and its expected response
    /// lines. Line 0 is the unsolicited hello; the rest alternate.
    struct Fixture {
        let hello: String
        let requests: [String]
        let responses: [String]

        init() throws {
            let url = try #require(
                Bundle.module.url(
                    forResource: "engine-wire-v1-adapter", withExtension: "ndjson"))
            let lines = try String(contentsOf: url, encoding: .utf8)
                .split(separator: "\n", omittingEmptySubsequences: true)
                .map(String.init)
            hello = lines[0]
            var requests: [String] = []
            var responses: [String] = []
            for (index, line) in lines.dropFirst().enumerated() {
                if index % 2 == 0 { requests.append(line) } else { responses.append(line) }
            }
            self.requests = requests
            self.responses = responses
        }
    }

    /// The fixture's hello carries `capabilities` and `spec_modes`, which
    /// are v1.1 additions, so the spawn it records IS the official one —
    /// `--speculative-protocol v1.1` present. Passed explicitly rather than
    /// left to the default, because that is a property of the fixture and
    /// not of this helper.
    private func makeServer(
        runner: any Runner, transport: ScriptedTransport, trusted: Bool = false
    ) -> BenchWorkerServer {
        BenchWorkerServer(
            runner: runner,
            transport: transport,
            trusted: trusted,
            speculative: true,
            build: "fixture",
            device: "mock",
            kvBytesCapacity: 1 << 20,
            maxDecodeTokens: 64,
            memory: FixtureMemoryReporter(),
            nonce: "fixturenonce")
    }

    // MARK: - Shared conformance fixture

    @Test("Every fixture response line is reproduced byte for byte")
    func fixtureBytes() async throws {
        let fixture = try Fixture()
        let transport = ScriptedTransport(lines: fixture.requests)
        let server = makeServer(runner: MockRunner(), transport: transport)
        await server.run()

        #expect(transport.written.count == fixture.responses.count + 1)

        // The hello like every other line: no substitution, no special
        // case. Both sides now derive it from the same checked-in manifest,
        // so `backend` and `runner.manifest_sha256` are two readings of one
        // declaration rather than two facts that can disagree.
        #expect(transport.written[0] == fixture.hello, "the hello diverged from the fixture")

        for (index, expected) in fixture.responses.enumerated() {
            #expect(
                transport.written[index + 1] == expected,
                "response \(index + 1) diverged from the fixture")
        }
    }

    /// The fixture's own phase-close barriers, called out explicitly: the
    /// parent fails a run whose `completed_work` disagrees with the timed
    /// steps it issued.
    @Test("completed_work equals the timed steps issued in each phase")
    func completedWorkBarrier() async throws {
        let fixture = try Fixture()
        let transport = ScriptedTransport(lines: fixture.requests)
        let server = makeServer(runner: MockRunner(), transport: transport)
        await server.run()

        let decoder = JSONDecoder()
        let completed = try transport.written
            .map { try decoder.decode(WorkerResponse.self, from: Data($0.utf8)) }
            .compactMap(\.completedWork)
        // Phases in fixture order: prefill only (no timed steps);
        // decode_begin + decode_step; correctness only; correctness_begin +
        // correctness_step; free_decode_begin + three verify rounds.
        #expect(completed == [0, 2, 0, 2, 4])
    }

    // MARK: - Refusals

    @Test("free_decode_begin is refused when no free-run regime is declared")
    func refusesUnadvertisedFreeRun() async throws {
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53]}"
        ])
        let server = makeServer(runner: TeacherForcedOnlyMockRunner(), transport: transport)
        await server.run()

        let decoder = JSONDecoder()
        let response = try decoder.decode(
            WorkerResponse.self, from: Data(transport.written[1].utf8))
        #expect(response.id == 1)
        #expect(response.ok == false)
        #expect(response.nonce == "fixturenonce")
        #expect(response.error == "free_run_decode is not served by this adapter")
    }

    @Test("An unparseable line answers id -1 without ending the session")
    func refusesUnparseableLine() async throws {
        let transport = ScriptedTransport(lines: [
            "{ not json",
            "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[11,12,13]}",
        ])
        let server = makeServer(runner: MockRunner(), transport: transport)
        await server.run()

        let decoder = JSONDecoder()
        let refusal = try decoder.decode(
            WorkerResponse.self, from: Data(transport.written[1].utf8))
        #expect(refusal.id == -1)
        #expect(refusal.ok == false)
        let served = try decoder.decode(
            WorkerResponse.self, from: Data(transport.written[2].utf8))
        #expect(served.id == 1)
        #expect(served.token == 100_003)
    }

    @Test("A spec mode the runner did not load is refused, never resolved")
    func refusesUnloadedMode() async throws {
        let transport = ScriptedTransport(lines: [
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51],"
                + "\"spec\":{\"mode\":\"dflash\"}}"
        ])
        let server = makeServer(runner: MockRunner(), transport: transport)
        await server.run()

        let response = try JSONDecoder().decode(
            WorkerResponse.self, from: Data(transport.written[1].utf8))
        #expect(response.ok == false)
        #expect(response.error == "spec mode dflash was not advertised")
    }

    // MARK: - Hello derivation (§6.1)

    @Test("Hello derives every field from the manifest and the loaded state")
    func helloDerivation() async throws {
        let transport = ScriptedTransport(lines: [])
        let server = makeServer(runner: MockRunner(), transport: transport)
        let hello = server.hello()

        #expect(hello.id == 0)
        #expect(hello.ok)
        #expect(hello.nonce == "fixturenonce")
        #expect(hello.protocolVersion == 1)
        // From the manifest, never from a constant here.
        #expect(hello.backend == "mock")
        #expect(hello.device == "mock")
        // Every declared regime is single-stream and free-run or
        // teacher-forced, so exactly one capability, and no max_batch_size.
        #expect(hello.capabilities == ["free_run_decode"])
        #expect(hello.maxBatchSize == nil)
        #expect(hello.specModes == ["serial", "mtp"])
        #expect(hello.headProvenance == nil)
        #expect(hello.runner?.id == "layr/mock-adapter")
        #expect(hello.runner?.modelType == "qwen4_exp_text")
        #expect(hello.runner?.manifestSHA256 == MockRunner.manifest.sha256Digest())
        #expect(hello.runner?.build == "fixture")
    }

    @Test("The trusted build is the only one advertising cohort_reference_replay")
    func helloTrustedCapability() {
        let untrusted = makeServer(
            runner: MockRunner(), transport: ScriptedTransport(lines: []))
        #expect(untrusted.hello().capabilities?.contains("cohort_reference_replay") == false)

        let trusted = makeServer(
            runner: MockRunner(), transport: ScriptedTransport(lines: []), trusted: true)
        #expect(trusted.hello().capabilities?.contains("cohort_reference_replay") == true)
    }

    @Test("A cohort regime yields the batched capability and max_batch_size")
    func helloCohortDerivation() {
        // Gemma 4 declares an `upTo(8)` free-run regime; the derivation must
        // read the width off the manifest rather than a constant here.
        let widths = Gemma4TextRunner.manifest.regimes.map(\.batch.maxWidth)
        #expect(widths.max() == 8)
        #expect(
            Gemma4TextRunner.manifest.regimes.contains {
                $0.batch.maxWidth > 1 && $0.timing == .freeRun
            })
        #expect(Qwen3VLRunner.manifest.regimes.allSatisfy { $0.batch.maxWidth == 1 })
    }

    // MARK: - Wire round trip

    @Test("Every request kind decodes from its schema spelling")
    func requestKindsDecode() throws {
        let lines: [(String, WorkerRequest.Kind)] = [
            ("{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[1]}", .prefill),
            ("{\"id\":2,\"kind\":\"decode_begin\",\"seed_tokens\":[1]}", .decodeBegin),
            ("{\"id\":3,\"kind\":\"decode_step\",\"token\":1}", .decodeStep),
            ("{\"id\":4,\"kind\":\"correctness\",\"prompt_tokens\":[1],\"steps\":2}", .correctness),
            (
                "{\"id\":5,\"kind\":\"correctness_begin\",\"prompt_tokens\":[1]}",
                .correctnessBegin
            ),
            ("{\"id\":6,\"kind\":\"correctness_step\",\"token\":1}", .correctnessStep),
            ("{\"id\":7,\"kind\":\"phase_diagnostics\"}", .phaseDiagnostics),
            (
                "{\"id\":8,\"kind\":\"free_decode_begin\",\"seed_tokens_by_stream\":[[1],[2]],"
                    + "\"batch_size\":2,\"spec\":{\"mode\":\"mtp\",\"mtp\":{\"depth\":2}}}",
                .freeDecodeBegin
            ),
            (
                "{\"id\":9,\"kind\":\"free_decode_run\",\"count\":4,\"batch_size\":2}",
                .freeDecodeRun
            ),
            (
                "{\"id\":10,\"kind\":\"cohort_reference_replay\","
                    + "\"replay_seeds_by_stream\":[[1]],\"committed_by_stream\":[[2]],"
                    + "\"logit_top_k\":16,\"rel_envelope\":0.05,\"replay_width\":\"cohort\"}",
                .cohortReferenceReplay
            ),
        ]
        for (line, kind) in lines {
            let request = try JSONDecoder().decode(
                WorkerRequest.self, from: Data(line.utf8))
            #expect(request.kind == kind)
        }
    }

    @Test("Every response field round-trips through the wire writer")
    func responseRoundTrip() throws {
        var response = WorkerResponse(id: 7, ok: true, nonce: "n")
        response.token = 1
        response.topLogits = [CorrectnessTraceLogit(token: 1, logit: 10)]
        response.seedToken = 2
        response.tokens = [3, 4]
        response.expertStats = ExpertStreamingStats()
        response.peakRAMGB = 18.5
        response.protocolVersion = 1
        response.backend = "mlx"
        response.device = "m5"
        response.completedWork = 2
        response.cacheMemory = 0
        response.capabilities = ["free_run_decode"]
        response.acceptanceLengths = [1, 2]
        response.draftedTotal = 6
        response.acceptedTotal = 4
        response.committedTotal = 6
        response.effectiveSpec = SpecConfig(mode: "mtp", mtp: MtpSpec(depth: 2))
        response.specModes = ["serial", "mtp"]
        response.headProvenance = WireHeadProvenance(
            HeadProvenance(sha256: "abc", bytes: 12, fileCount: 1))
        response.mlxActiveMemoryBytes = 1
        response.mlxCacheMemoryBytes = 2
        response.mlxPeakMemoryBytes = 3
        response.topLogitMargin = 1.5
        response.expectedTokenLogit = 2.5
        response.expectedTokenRank = 0
        response.specDecoder = "mtp"
        response.maxBatchSize = 8
        response.seedTokenByStream = [1, 2]
        response.effectiveBatchSize = 2
        response.tokensByStream = [[1], [2]]
        response.naturalAcceptedByStream = [[1], [2]]
        response.rounds = 2
        response.activeStreamsByRound = [2, 1]
        response.depthClampReasons = ["batch_gate": 1]
        response.prefillNsByStream = [1, 2]
        response.decodeNsByStream = [3, 4]
        response.verifyReplayDisagreements = 0
        response.runner = RunnerIdentity(
            id: "layr/mock", modelType: "mock", manifestSHA256: "d", build: "b")

        let line = response.jsonLine()
        // `runner` LAST, per contract §6.0's appended-last convention.
        #expect(
            line.hasSuffix(
                ",\"runner\":{\"id\":\"layr/mock\",\"model_type\":\"mock\","
                    + "\"manifest_sha256\":\"d\",\"build\":\"b\"}}"))

        let decoded = try JSONDecoder().decode(WorkerResponse.self, from: Data(line.utf8))
        #expect(decoded.id == 7)
        #expect(decoded.ok)
        #expect(decoded.tokensByStream == [[1], [2]])
        #expect(decoded.effectiveSpec == SpecConfig(mode: "mtp", mtp: MtpSpec(depth: 2)))
        #expect(decoded.depthClampReasons == ["batch_gate": 1])
        #expect(decoded.runner?.manifestSHA256 == "d")

        // Doubles keep their decimal point: a JSON `0` where the wire says
        // `0.0` is a different byte string, and both sides compare bytes.
        #expect(line.contains("\"expert_read_seconds\":0.0"))
        #expect(line.contains("\"logit\":10.0"))
        #expect(line.contains("\"peak_ram_gb\":18.5"))
    }

    @Test("An omitted optional is absent, never null")
    func omittedNotNull() {
        let line = WorkerResponse(id: 3, ok: true, nonce: "n").jsonLine()
        #expect(line == "{\"id\":3,\"nonce\":\"n\",\"ok\":true}")
    }
}

// MARK: - Free-run audit from the engine's round journal

@Suite("Free-run round audit")
struct FreeRunRoundAuditTests {

    private func makeServer(
        runner: any Runner, transport: ScriptedTransport
    ) -> BenchWorkerServer {
        BenchWorkerServer(
            runner: runner,
            transport: transport,
            trusted: false,
            speculative: true,
            build: "fixture",
            device: "mock",
            kvBytesCapacity: 1 << 20,
            maxDecodeTokens: 64,
            memory: FixtureMemoryReporter(),
            nonce: "fixturenonce")
    }

    private func freeRun(
        script: MockRoundScript?, batch: Int = 1, count: Int = 6, serial: Bool = false
    ) async throws -> WorkerResponse {
        let spec =
            serial
            ? "\"spec\":{\"mode\":\"serial\"}"
            : "\"spec\":{\"mode\":\"mtp\",\"mtp\":{\"depth\":2}}"
        let begin: String
        let run: String
        if batch > 1 {
            let seeds = (0 ..< batch).map { "[5\($0),52,53]" }.joined(separator: ",")
            begin =
                "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens_by_stream\":[\(seeds)],"
                + "\"batch_size\":\(batch),\(spec)}"
            run =
                "{\"id\":2,\"kind\":\"free_decode_run\",\"count\":\(count),"
                + "\"batch_size\":\(batch)}"
        } else {
            begin =
                "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53],"
                + "\(spec)}"
            run = "{\"id\":2,\"kind\":\"free_decode_run\",\"count\":\(count)}"
        }
        let transport = ScriptedTransport(lines: [begin, run])
        let runner: any Runner =
            batch > 1 ? CohortMockRunner(script: script) : MockRunner(script: script)
        let server = makeServer(runner: runner, transport: transport)
        await server.run()
        return try JSONDecoder().decode(
            WorkerResponse.self, from: Data(transport.written[2].utf8))
    }

    /// The journal states the window; the worker only groups it.
    @Test("A three-round journal yields its own acceptance lengths and totals")
    func journalDrivesTheAudit() async throws {
        let response = try await freeRun(script: .fixtureWindow)
        #expect(response.ok)
        #expect(response.acceptanceLengths == [3, 1, 2])
        // drafted: 3 rounds at k = 2. accepted: min(accepted, confirmed) per
        // round — 2, 1, 1 — which is what the engine's own cumulative
        // `observedAccepted` counter accumulates.
        #expect(response.draftedTotal == 6)
        #expect(response.acceptedTotal == 4)
        #expect(response.committedTotal == 6)
        // Single-stream: the v1.2 cohort trio is absent, and
        // `acceptance_lengths.count` already IS the round count.
        #expect(response.rounds == nil)
        #expect(response.naturalAcceptedByStream == nil)
        #expect(response.activeStreamsByRound == nil)
        #expect(response.acceptanceLengths?.count == 3)
    }

    /// The cohort form carries the round count and the per-row pre-min walk.
    @Test("The cohort form reports rounds, natural accepts and active streams")
    func cohortAuditFields() async throws {
        var script = MockRoundScript.fixtureWindow
        script.requestIDs = [1, 2]
        let response = try await freeRun(script: script, batch: 2)
        #expect(response.ok)
        #expect(response.rounds == 3)
        #expect(response.acceptanceLengths == [3, 1, 2])
        #expect(response.activeStreamsByRound == [2, 2, 2])
        // PRE-min: round 2 committed 1 while the walk accepted 1, and round 1
        // committed 3 on a walk of 2 — the raw walk, not the clamped count.
        #expect(response.naturalAcceptedByStream == [[2, 1, 1], [2, 1, 1]])
        #expect(response.draftedTotal == 12)
        #expect(response.acceptedTotal == 8)
    }

    /// A row that finishes early leaves no record past its last round, so the
    /// active-stream curve falls out of the journal rather than being counted
    /// somewhere else.
    @Test("A row that stops early makes active_streams_by_round fall")
    func stragglerRow() async throws {
        var script = MockRoundScript.fixtureWindow
        script.requestIDs = [1, 2]
        // Drop row 2's last round.
        var records = script.journal
        records.removeLast()
        let assembled = try FreeRunRoundAudit.assemble(
            records: records, slotForRequestID: [1: 0, 2: 1], streamCount: 2)
        #expect(assembled?.activeStreamsByRound == [2, 2, 1])
        #expect(assembled?.acceptanceLengths == [3, 1, 2])
        #expect(assembled?.naturalAcceptedByStream == [[2, 1, 1], [2, 1, 0]])
    }

    /// The engine runs ahead: a round commits up to depth + 1 tokens, and the
    /// journal can hold rounds whose tokens are still in the stream. The
    /// window states exactly the tokens it returned; the rest carries.
    @Test("A round straddling the drain boundary is clipped, the remainder carries")
    func straddlingRoundIsClipped() async throws {
        // Fixture rounds commit 3, 1, 2; drain 5: the third round is cut to 1.
        let response = try await freeRun(script: .fixtureWindow, count: 5)
        #expect(response.ok)
        #expect(response.acceptanceLengths == [3, 1, 1])
        // Drafts of every round that started are stated now (3 × k=2);
        // accepts that fit: 2, 1, and min(1, 1 - 1) = 0 for the clipped round.
        #expect(response.draftedTotal == 6)
        #expect(response.acceptedTotal == 3)
        #expect(response.committedTotal == 5)
    }

    @Test("Rounds past the boundary carry intact into the next window")
    func runAheadRoundsCarry() async throws {
        // Two rounds journaled at once (4 then 2 tokens); the caller drains 4,
        // then 2. Window one is the first round alone; window two is the
        // second round, drafts and accepts intact — stated once, in the
        // window that returned its tokens.
        let script = MockRoundScript(rounds: [
            MockRoundScript.Round(k: 3, accepted: 3, confirmed: 4),
            MockRoundScript.Round(k: 3, accepted: 1, confirmed: 2),
        ])
        let begin =
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53],"
            + "\"spec\":{\"mode\":\"mtp\",\"mtp\":{\"depth\":3}}}"
        let transport = ScriptedTransport(lines: [
            begin,
            "{\"id\":2,\"kind\":\"free_decode_run\",\"count\":4}",
            "{\"id\":3,\"kind\":\"free_decode_run\",\"count\":2}",
        ])
        let server = makeServer(runner: MockRunner(script: script), transport: transport)
        await server.run()
        let first = try JSONDecoder().decode(
            WorkerResponse.self, from: Data(transport.written[2].utf8))
        let second = try JSONDecoder().decode(
            WorkerResponse.self, from: Data(transport.written[3].utf8))
        #expect(first.ok && second.ok)
        #expect(first.acceptanceLengths == [4])
        #expect(first.draftedTotal == 3)
        #expect(first.acceptedTotal == 3)
        #expect(first.committedTotal == 4)
        #expect(second.acceptanceLengths == [2])
        #expect(second.draftedTotal == 3)
        #expect(second.acceptedTotal == 1)
        #expect(second.committedTotal == 2)
    }

    /// The echo is the SERVED depth: the engine clamps to the manifest's
    /// ceiling, so the echo must too, or benchd seals a depth that never ran.
    @Test("A depth above the manifest range echoes the ceiling")
    func depthEchoIsClamped() async throws {
        let begin =
            "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53],"
            + "\"spec\":{\"mode\":\"mtp\",\"mtp\":{\"depth\":7}}}"
        let transport = ScriptedTransport(lines: [begin])
        let server = makeServer(runner: MockRunner(script: nil), transport: transport)
        await server.run()
        let response = try JSONDecoder().decode(
            WorkerResponse.self, from: Data(transport.written[1].utf8))
        #expect(response.ok)
        // The mock adapter manifest serves mtp at depth [1, 3].
        #expect(response.effectiveSpec?.depth == 3)
    }

    /// A truncated head cannot prove coverage of the window, so the response
    /// is refused rather than assembled from what survived.
    @Test("A journal at its retention cap refuses the response")
    func cappedJournalRefuses() async throws {
        let capped = MockRoundScript(
            rounds: Array(
                repeating: MockRoundScript.Round(k: 2, accepted: 1, confirmed: 1),
                count: CBv2MTPRoundAuditRecord.retainedRecordCap))
        let response = try await freeRun(script: capped, count: 1)
        #expect(response.ok == false)
        #expect(
            response.error
                == "mtp round journal truncated at its 8192-record cap "
                + "(8192 retained): coverage cannot be proven")
    }

    /// Rows disagreeing on a round's committed width means the grouping or
    /// the window boundary is wrong; a cohort commits ONE common width.
    @Test("Disagreeing committed widths in one round are refused")
    func widthDisagreementRefused() {
        let wide = CBv2MTPRoundAuditRecord(
            requestID: 1, k: 2, draftTokens: [], targetTokens: [], accepted: 1,
            confirmed: 3, rejected: 0, tokensCountAfter: 1, numComputedAfter: 0,
            generatedAfter: 1, finishReason: nil)
        let narrow = CBv2MTPRoundAuditRecord(
            requestID: 2, k: 2, draftTokens: [], targetTokens: [], accepted: 1,
            confirmed: 2, rejected: 1, tokensCountAfter: 1, numComputedAfter: 0,
            generatedAfter: 1, finishReason: nil)
        let records = [wide, narrow]
        #expect(
            throws: FreeRunRoundAudit.AuditError.widthDisagreement(
                round: 0, widths: [3, 2])
        ) {
            _ = try FreeRunRoundAudit.assemble(
                records: records, slotForRequestID: [1: 0, 2: 1], streamCount: 2)
        }
    }

    /// An MTP window whose engine journaled nothing reports no audit fields
    /// — absent is not the same claim as zero, and benchd refuses the reply,
    /// which is the right outcome for a drafting engine that did not journal.
    @Test("An MTP window with no journal reports no audit fields")
    func noJournalNoAudit() async throws {
        let response = try await freeRun(script: nil)
        #expect(response.ok)
        #expect(response.acceptanceLengths == nil)
        #expect(response.draftedTotal == nil)
        #expect(response.acceptedTotal == nil)
        #expect(response.verifyReplayDisagreements == nil)
        #expect(response.specDecoder == nil)
        // `committed_total` is the drained count and is always reported.
        #expect(response.committedTotal == 6)
    }

    /// A serial window has no verify rounds, but the protocol carries the
    /// audit triple on EVERY free_decode_run: benchd refused the official
    /// warmup leg on a reply without `acceptance_lengths`. Serial states
    /// itself: N rounds of width one, nothing drafted, nothing accepted.
    @Test("A serial window reports N rounds of width one and zero drafts")
    func serialWindowStatesItself() async throws {
        let response = try await freeRun(script: nil, serial: true)
        #expect(response.ok)
        #expect(response.acceptanceLengths == Array(repeating: 1, count: 6))
        #expect(response.draftedTotal == 0)
        #expect(response.acceptedTotal == 0)
        #expect(response.committedTotal == 6)
        // Single-stream form: the cohort trio stays off the wire.
        #expect(response.rounds == nil)
        #expect(response.naturalAcceptedByStream == nil)
        #expect(response.activeStreamsByRound == nil)
        #expect(response.verifyReplayDisagreements == nil)
    }

    @Test("A serial cohort window reports the trio with every stream active")
    func serialCohortStatesItself() async throws {
        let response = try await freeRun(script: nil, batch: 2, count: 4, serial: true)
        #expect(response.ok)
        #expect(response.acceptanceLengths == [1, 1, 1, 1])
        #expect(response.rounds == 4)
        #expect(response.activeStreamsByRound == [2, 2, 2, 2])
        #expect(response.naturalAcceptedByStream == [[0, 0, 0, 0], [0, 0, 0, 0]])
        #expect(response.draftedTotal == 0)
        #expect(response.acceptedTotal == 0)
        #expect(response.committedTotal == 8)
    }
}

// MARK: - Speculative protocol gate

@Suite("Speculative protocol gate")
struct SpeculativeProtocolTests {

    private func makeServer(
        transport: ScriptedTransport, speculative: Bool, trusted: Bool = false
    ) -> BenchWorkerServer {
        BenchWorkerServer(
            runner: MockRunner(),
            transport: transport,
            trusted: trusted,
            speculative: speculative,
            build: "fixture",
            device: "mock",
            kvBytesCapacity: 1 << 20,
            maxDecodeTokens: 64,
            memory: FixtureMemoryReporter(),
            nonce: "fixturenonce")
    }

    private func responses(
        _ lines: [String], speculative: Bool
    ) async throws -> [WorkerResponse] {
        let transport = ScriptedTransport(lines: lines)
        let server = makeServer(transport: transport, speculative: speculative)
        await server.run()
        let decoder = JSONDecoder()
        return try transport.written.map {
            try decoder.decode(WorkerResponse.self, from: Data($0.utf8))
        }
    }

    // MARK: Argument

    @Test("Only v1.1 is accepted, and a wrong value refuses rather than degrades")
    func argumentParsing() throws {
        #expect(try SpeculativeProtocol.isEnabled("v1.1"))
        // Absent is plain v1, not a refusal.
        #expect(try SpeculativeProtocol.isEnabled(nil) == false)
        for raw in ["v1", "v1.2", "V1.1", "1.1", ""] {
            #expect(throws: SpeculativeProtocol.ArgumentError.unsupportedVersion(raw)) {
                _ = try SpeculativeProtocol.isEnabled(raw)
            }
        }
    }

    // MARK: Flag present

    @Test("With the flag, the hello advertises the manifest's speculative surface")
    func helloWithFlag() async throws {
        let hello = try await responses([], speculative: true)[0]
        #expect(hello.specModes == ["serial", "mtp"])
        #expect(hello.capabilities == ["free_run_decode"])
    }

    @Test("With the flag, decode_begin echoes the effective spec")
    func decodeBeginEchoesSpec() async throws {
        let answers = try await responses(
            [
                "{\"id\":1,\"kind\":\"decode_begin\",\"seed_tokens\":[21,22],"
                    + "\"spec\":{\"mode\":\"mtp\",\"mtp\":{\"depth\":2}}}"
            ],
            speculative: true)
        #expect(answers[1].ok)
        #expect(answers[1].effectiveSpec == SpecConfig(mode: "mtp", mtp: MtpSpec(depth: 2)))
    }

    // MARK: Flag absent

    @Test("Without the flag, the hello is plain v1")
    func helloWithoutFlag() async throws {
        let hello = try await responses([], speculative: false)[0]
        // Absent, not empty: an empty capability array is the claim "I
        // support none of these", while absence is the v1 wire.
        #expect(hello.capabilities == nil)
        #expect(hello.specModes == nil)
        #expect(hello.maxBatchSize == nil)
        // Everything a v1 hello does carry is unchanged.
        #expect(hello.protocolVersion == 1)
        #expect(hello.backend == "mock")
        #expect(hello.runner?.id == "layr/mock-adapter")
    }

    @Test("Without the flag, a spec on decode_begin is refused by name")
    func decodeBeginRefusesSpec() async throws {
        let answers = try await responses(
            [
                "{\"id\":1,\"kind\":\"decode_begin\",\"seed_tokens\":[21,22],"
                    + "\"spec\":{\"mode\":\"mtp\",\"mtp\":{\"depth\":2}}}"
            ],
            speculative: false)
        #expect(answers[1].ok == false)
        #expect(answers[1].error == "spec requires --speculative-protocol v1.1")
    }

    @Test("Without the flag, decode_begin with no spec still works and echoes nothing")
    func decodeBeginWithoutSpec() async throws {
        let answers = try await responses(
            ["{\"id\":1,\"kind\":\"decode_begin\",\"seed_tokens\":[21,22]}"],
            speculative: false)
        #expect(answers[1].ok)
        #expect(answers[1].seedToken == 200_002)
        #expect(answers[1].effectiveSpec == nil)
    }

    @Test("Without the flag, free_decode_begin refuses a spec by name")
    func freeDecodeBeginRefusesSpec() async throws {
        let answers = try await responses(
            [
                "{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53],"
                    + "\"spec\":{\"mode\":\"mtp\",\"mtp\":{\"depth\":2}}}"
            ],
            speculative: false)
        #expect(answers[1].ok == false)
        // The SPEC refusal, not the capability one: the parent asked for a
        // speculative surface by name, and pointing it at the manifest would
        // send it looking in the wrong place.
        #expect(answers[1].error == "spec requires --speculative-protocol v1.1")
    }

    @Test("Without the flag, free_decode_begin is not served at all")
    func freeDecodeBeginUnadvertised() async throws {
        let answers = try await responses(
            ["{\"id\":1,\"kind\":\"free_decode_begin\",\"seed_tokens\":[51,52,53]}"],
            speculative: false)
        #expect(answers[1].ok == false)
        #expect(answers[1].error == "free_run_decode is not served by this adapter")
    }

    @Test("Without the flag, the teacher-forced verbs are untouched")
    func teacherForcedVerbsUnaffected() async throws {
        let answers = try await responses(
            [
                "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[11,12,13]}",
                "{\"id\":2,\"kind\":\"correctness_begin\",\"prompt_tokens\":[41,42,43]}",
                "{\"id\":3,\"kind\":\"correctness_step\",\"token\":5001}",
            ],
            speculative: false)
        #expect(answers[1].token == 100_003)
        #expect(answers[2].token == 100_003)
        #expect(answers[3].token == 5002)
        #expect(answers[3].topLogits?.count == 8)
    }
}
