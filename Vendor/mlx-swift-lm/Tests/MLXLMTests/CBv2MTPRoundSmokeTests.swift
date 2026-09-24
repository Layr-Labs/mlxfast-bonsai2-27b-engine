// CBv2MTPRoundSmokeTests.swift
//
// MTP round-driver smoke tests through the REAL engine: EngineV2 over a
// tiny random-init Gemma-4 target + assistant drafter (the same weight-free
// fixtures as CBv2MTPModelSeamTests), contiguous KV backend,
// CBv2LayerCacheBank, CBv2DefaultSampler. Compiled decode is disabled so
// both legs run the eager paths the MTP round reuses.
//
// The acceptance invariant is GREEDY LOSSLESSNESS: an MTP-on engine emits
// token-exactly what an MTP-off engine emits, for every scenario —
// including sliding-window wrap during rounds (window 16 << generation
// length), mixed batches (verify rows + prefill neighbors), mid-round
// stop-token and maxTokens truncation, and batched [2, 1+k] verify rounds.
// Plus liveness: a cancelled MTP request must not leak pendingSamples
// (a leak blocks the ENTIRE waiting-admission loop, so a follow-up request
// would never complete).

import Foundation
import MLX
@_spi(Benchmarking) @testable import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXLLM

@Suite("CBv2MTPRoundSmoke", .serialized)
struct CBv2MTPRoundSmokeTests {

    private let vocabSize = 256
    private let hiddenSize = 64
    /// Small window so decode + verify rounds wrap the sliding ring early.
    private let slidingWindow = 16

    // MARK: - Fixtures (mirrors CBv2MTPModelSeamTests)

    /// 6-layer target, last 2 KV-shared; capture layers full=2, sliding=3.
    private func targetConfig(
        tieWordEmbeddings: Bool = true
    ) throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": \(hiddenSize),
                "num_hidden_layers": 6,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "head_dim": 32,
                "global_head_dim": 32,
                "num_key_value_heads": 1,
                "num_kv_shared_layers": 2,
                "layer_types": ["sliding_attention", "full_attention",
                                "full_attention", "sliding_attention",
                                "sliding_attention", "full_attention"],
                "sliding_window": \(slidingWindow),
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": \(tieWordEmbeddings),
                "vocab_size": \(vocabSize),
                "vocab_size_per_layer_input": \(vocabSize),
                "rms_norm_eps": 1e-6,
                "hidden_size_per_layer_input": 0,
                "use_double_wide_mlp": false
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8))
    }

    /// 2-layer fully-KV-shared drafter matching the target's hidden/vocab.
    private func drafterConfig() throws -> Gemma4AssistantConfiguration {
        let json = """
            {
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": \(hiddenSize),
                "use_ordered_embeddings": false,
                "num_centroids": 16,
                "centroid_intermediate_top_k": 4,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "head_dim": 32,
                    "global_head_dim": 32,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 2,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "sliding_window": \(slidingWindow),
                    "final_logit_softcapping": null,
                    "tie_word_embeddings": true,
                    "vocab_size": \(vocabSize),
                    "vocab_size_per_layer_input": \(vocabSize),
                    "rms_norm_eps": 1e-6,
                    "hidden_size_per_layer_input": 0,
                    "use_double_wide_mlp": false
                }
            }
            """
        return try JSONDecoder.json5().decode(
            Gemma4AssistantConfiguration.self, from: Data(json.utf8))
    }

    private struct Fixture {
        let target: Gemma4TextModel
        let drafter: Gemma4AssistantDraftModel
    }

    private func makeFixture(
        seed: UInt64 = 0x5EED, deterministicTarget: Bool = false
    ) throws -> Fixture {
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(
            try targetConfig(tieWordEmbeddings: !deterministicTarget))
        let drafter = try Gemma4AssistantDraftModel(config: drafterConfig())
        if deterministicTarget { stabilizeCBv2MTPGreedyCycleTarget(target) }
        eval(target, drafter)
        return Fixture(target: target, drafter: drafter)
    }

    /// The real engine over the fixture. `mtp: false` builds the identical
    /// engine WITHOUT a drafter (the MTP-off baseline). Compiled decode is
    /// off: MTP never touches it, and the parity legs should compare the
    /// same eager paths.
    private func makeEngine(
        _ fixture: Fixture, mtp: Bool, maxDraftTokens: Int = 2, maxSpeculativeBatch: Int = 2,
        maxConcurrent: Int = 4,
        verificationMode: CBv2MTPVerificationMode = .serialTarget
    ) throws -> EngineV2 {
        let kinds = fixture.target.cbv2LayerKinds
        let mtpDrafter: Gemma4CBv2MTPDrafter? =
            mtp
            ? try Gemma4CBv2MTPDrafter(drafter: fixture.drafter, target: fixture.target)
            : nil
        let mtpConfig = CBv2MTPConfig(
            enabled: mtp, maxDraftTokens: maxDraftTokens,
            maxSpeculativeBatch: maxSpeculativeBatch,
            fixedDraftTokens: maxDraftTokens,
            verificationMode: verificationMode,
            maxAutomaticRectangularTokens: 8)
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(fixture.target),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            mtpDrafter: mtpDrafter,
            mtpConfig: mtpConfig)
    }

    private func greedyRequest(
        id: UInt64, prompt: [Int], maxTokens: Int, stopTokens: Set<Int> = [],
        stopStrings: [String] = [], temperature: Float = 0
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: temperature),
            maxTokens: maxTokens, stopTokens: stopTokens,
            stopStrings: stopStrings)
    }

    private func run(
        _ engine: EngineV2, _ request: CBv2Request
    ) async throws -> CBv2SchedCollected {
        await cbv2SchedCollect(try engine.submit(request))
    }

    private func completedForwardShapes(
        _ engine: EngineV2, since before: CBv2ForwardShapeSnapshot,
        requireVerification: Bool = true
    ) -> CBv2ForwardShapeDelta {
        let after = engine.forwardShapeSnapshot(), delta = after.delta(since: before)
        #expect(after.pendingSteps == 0 && after.abandonedSteps == 0)
        #expect(after.unobservedDispatches == 0 && after.droppedCalls == 0)
        #expect(delta.complete)
        #expect(delta.entries.allSatisfy { $0.submittedCalls == $0.completedCalls })
        let targets = delta.entries.filter { $0.axes.kind == .target }
        #expect(!targets.isEmpty)
        if requireVerification { #expect(targets.contains { $0.axes.phase == .mtpVerification }) }
        return delta
    }

    // MARK: - (1) Solo greedy parity through window wraps + metrics sanity

    @Test func soloGreedyTokenExactWithWindowWrap() async throws {
        let fixture = try makeFixture()
        // Prompt 24 > window 16 and 40 generated tokens: the sliding ring
        // wraps repeatedly THROUGH verify rounds (staged-write edge).
        let prompt = makePromptTokens(length: 24, seed: 11, vocabSize: vocabSize)

        let off = try makeEngine(fixture, mtp: false)
        let baseline = try await run(off, greedyRequest(id: 1, prompt: prompt, maxTokens: 40))
        await off.shutdown()
        #expect(baseline.finishReason == .length)
        #expect(baseline.tokens.count == 40)
        #expect(off.mtpMetricsSnapshot() == nil, "MTP-off engine must report no MTP state")

        let on = try makeEngine(fixture, mtp: true, verificationMode: .automatic)
        let speculative = try await run(on, greedyRequest(id: 1, prompt: prompt, maxTokens: 40))
        let metrics = try #require(on.mtpMetricsSnapshot())
        await on.shutdown()

        #expect(speculative.finishReason == .length)
        #expect(
            speculative.tokens == baseline.tokens,
            "MTP-on output diverged: on=\(speculative.tokens) off=\(baseline.tokens)")

        // The round loop really ran: one seed step to establish the carry,
        // then rounds emitting 1..1+k tokens each.
        #expect(metrics.seedSteps >= 1)
        #expect(metrics.rounds >= 1)
        // Terminal-depth clamping may run the final rounds at k=1 even though
        // the configured ceiling is k=2.
        #expect(metrics.draftedTokens >= metrics.rounds)
        #expect(metrics.draftedTokens <= metrics.rounds * 2)
        #expect(metrics.controllerFallbacks["tail_depth", default: 0] > 0)
        #expect(metrics.emittedTokens >= metrics.rounds)
        #expect(metrics.emittedTokens <= metrics.rounds * 3)
        #expect(metrics.acceptedTokens <= metrics.draftedTokens)
        // Per-position acceptance is monotonically non-increasing.
        if metrics.perPositionAccepted.count == 2 {
            #expect(metrics.perPositionAccepted[0] >= metrics.perPositionAccepted[1])
        }
        // Every generated token comes from the prefill bonus (1), seed
        // steps (1 each), round emissions, or plain decode steps (near the
        // length cap / after a carry loss) — never more than the total.
        #expect(1 + metrics.seedSteps + metrics.emittedTokens <= 40)
    }

    // MARK: - (1b) Per-request timing sees the rounds

    @Test func perRequestTimingCountsRounds() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 24, seed: 12, vocabSize: vocabSize)
        let on = try makeEngine(fixture, mtp: true, verificationMode: .automatic)
        CBv2CoreInstrumentation.countingEnabled = true
        defer { CBv2CoreInstrumentation.countingEnabled = false }
        let syncsBefore = CBv2CoreInstrumentation.hostSyncs
        let speculative = try await run(on, greedyRequest(id: 1, prompt: prompt, maxTokens: 40))
        let metrics = try #require(on.mtpMetricsSnapshot())
        await on.shutdown()  // the drain completes only with no step in flight
        let syncs = CBv2CoreInstrumentation.hostSyncs - syncsBefore
        let stepsExecuted = on.capacity().stepsExecuted

        // Host-sync multiplier on the MTP path: every executed step performs
        // its ONE finalize readback; an MTP-round finalize adds up to three
        // more (seed policy margin, acceptance packet, verify policy margin);
        // serial target verification adds one blocking eval per verify
        // column at launch; a round whose capture could not be fenced adds
        // one blocking eval (never on the contiguous/paged backends that
        // exist today); a logprob segment adds three readbacks (none here —
        // no row asks for logprobs). So
        //   steps ≤ syncs ≤ steps + seedSteps + 2 × rounds + serialColumns
        //                  + captureFallbackRounds.
        // Only the lower bound is asserted here: the counter is
        // process-global and swift-testing runs other engine suites
        // concurrently (the exact per-step equality is asserted by the
        // serial XCTest timing suite).
        #expect(syncs >= stepsExecuted, "one finalize readback per executed step")

        #expect(speculative.finishReason == .length)
        let t = try #require(speculative.usage).timing
        #expect(t.mtpRounds > 0, "the round loop must be visible per request")
        #expect(t.mtpAccepted <= t.mtpProposed)
        // Alone in the engine, the per-request tallies ARE the engine's
        // cumulative round metrics (both recorded in the same verify walk).
        #expect(t.mtpRounds == UInt32(metrics.rounds))
        #expect(t.mtpProposed == UInt32(metrics.draftedTokens))
        #expect(t.mtpAccepted == UInt32(metrics.acceptedTokens))
        // The exported phase order holds on the MTP launch path too.
        #expect(t.admittedNanos > 0)
        #expect(t.admittedNanos <= t.kvAllocatedNanos)
        #expect(t.kvAllocatedNanos <= t.prefillFirstLaunchNanos)
        #expect(t.prefillFirstLaunchNanos <= t.promptComputedNanos)
        #expect(t.promptComputedNanos <= t.firstTokenNanos)
        #expect(t.firstTokenNanos <= t.finishedNanos)
    }

    // MARK: - (2) Mixed batch: verify rows + a prefilling neighbor

    @Test func mixedBatchWithPrefillNeighborStaysTokenExact() async throws {
        let fixture = try makeFixture()
        let promptA = makePromptTokens(length: 20, seed: 21, vocabSize: vocabSize)
        // Long prompt = several [1, 16] prefill chunks riding beside A's
        // seed/verify steps.
        let promptB = makePromptTokens(length: 44, seed: 22, vocabSize: vocabSize)

        var baselines: [CBv2SchedCollected] = []
        for (index, (prompt, maxTokens)) in [(promptA, 28), (promptB, 20)].enumerated() {
            let off = try makeEngine(fixture, mtp: false)
            baselines.append(
                try await run(
                    off, greedyRequest(id: UInt64(index + 1), prompt: prompt, maxTokens: maxTokens))
            )
            await off.shutdown()
        }

        let on = try makeEngine(fixture, mtp: true)
        async let a = run(on, greedyRequest(id: 1, prompt: promptA, maxTokens: 28))
        // Give A a head start so it is decoding (seeding/rounds) while B
        // prefills its chunks — the mixed-plan shape the driver must handle.
        try await Task.sleep(nanoseconds: 100_000_000)
        async let b = run(on, greedyRequest(id: 2, prompt: promptB, maxTokens: 20))
        let (collectedA, collectedB) = try await (a, b)
        await on.shutdown()

        #expect(collectedA.tokens == baselines[0].tokens, "row A diverged in the mixed batch")
        #expect(collectedB.tokens == baselines[1].tokens, "row B diverged in the mixed batch")
    }

    // MARK: - (3) Two verify rows in one rectangular round

    @Test func twoSpeculatingRowsStayTokenExact() async throws {
        let fixture = try makeFixture(deterministicTarget: true)
        let promptA = makePromptTokens(length: 12, seed: 31, vocabSize: vocabSize)
        let promptB = makePromptTokens(length: 18, seed: 32, vocabSize: vocabSize)

        var baselines: [CBv2SchedCollected] = []
        for (index, prompt) in [promptA, promptB].enumerated() {
            let off = try makeEngine(fixture, mtp: false)
            baselines.append(
                try await run(off, greedyRequest(id: UInt64(index + 1), prompt: prompt, maxTokens: 24)))
            await off.shutdown()
        }
        let expectedA = cbv2MTPExpectedGreedyCycle(
            after: promptA.last!, count: 24, vocabularySize: vocabSize)
        let expectedB = cbv2MTPExpectedGreedyCycle(
            after: promptB.last!, count: 24, vocabularySize: vocabSize)
        #expect(baselines[0].tokens == expectedA)
        #expect(baselines[1].tokens == expectedB)

        // Both rows greedy and running (batch gate 2): exercise the universal
        // serial target verifier and the explicit rectangular optimization.
        for mode in [CBv2MTPVerificationMode.serialTarget, .rectangular] {
            let on = try makeEngine(fixture, mtp: true, verificationMode: mode)
            let before = try on.beginForwardShapeObservation()
            let streams = try on.loopForTesting.onEngineQueueSync {
                (try on.submit(greedyRequest(id: 1, prompt: promptA, maxTokens: 24)),
                 try on.submit(greedyRequest(id: 2, prompt: promptB, maxTokens: 24)))
            }
            async let a = cbv2SchedCollect(streams.0)
            async let b = cbv2SchedCollect(streams.1)
            let (collectedA, collectedB) = await (a, b)
            let metrics = try #require(on.mtpMetricsSnapshot())
            await on.shutdown()
            let shapes = completedForwardShapes(on, since: before)
            let verification = shapes.entries.filter { $0.axes.kind == .target && $0.axes.phase == .mtpVerification }
            #expect(verification.contains { $0.axes.liveBatchRows == 2 && $0.completedCalls > 0 })
            #expect(verification.allSatisfy { $0.axes.liveBatchRows <= 2 })
            if mode == .serialTarget {
                #expect(verification.allSatisfy { $0.axes.sequenceWidth == 1 })
            } else {
                #expect(verification.contains { $0.axes.sequenceWidth > 1 })
            }

            #expect(
                collectedA.tokens == baselines[0].tokens,
                "row A diverged in \(mode.rawValue) mode")
            #expect(
                collectedB.tokens == baselines[1].tokens,
                "row B diverged in \(mode.rawValue) mode")
            #expect(metrics.rounds >= 1)
            #expect(metrics.acceptedTokens < metrics.draftedTokens, "fixture must exercise rejected speculative work")
        }
    }

    // MARK: - (4) Mid-round stop-token truncation

    @Test func stopTokenMidRoundTruncatesExactly() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 20, seed: 41, vocabSize: vocabSize)

        // Baseline without stops, to pick a token the stream really emits
        // somewhere mid-generation.
        let probe = try makeEngine(fixture, mtp: false)
        let unstopped = try await run(probe, greedyRequest(id: 9, prompt: prompt, maxTokens: 32))
        await probe.shutdown()
        try #require(unstopped.tokens.count == 32)
        // A token from the middle of the stream: rounds emit up to 3
        // tokens, so an odd index regularly lands mid-round.
        let stopToken = unstopped.tokens[17]
        let stops: Set<Int> = [stopToken]

        let off = try makeEngine(fixture, mtp: false)
        let baseline = try await run(
            off, greedyRequest(id: 1, prompt: prompt, maxTokens: 32, stopTokens: stops))
        await off.shutdown()
        #expect(baseline.finishReason == .stop)

        let on = try makeEngine(fixture, mtp: true)
        let speculative = try await run(
            on, greedyRequest(id: 1, prompt: prompt, maxTokens: 32, stopTokens: stops))
        await on.shutdown()

        #expect(speculative.finishReason == .stop)
        #expect(
            speculative.tokens == baseline.tokens,
            "stop-token truncation diverged: on=\(speculative.tokens) off=\(baseline.tokens)")
    }

    // MARK: - (5) maxTokens truncation lands exactly

    @Test func maxTokensTruncationStaysExact() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 20, seed: 51, vocabSize: vocabSize)
        // Odd cap: with k=2 the last round would overshoot without the
        // mid-scan maxTokens truncation + planner max_tokens clamp.
        for maxTokens in [3, 7] {
            let off = try makeEngine(fixture, mtp: false)
            let baseline = try await run(
                off, greedyRequest(id: 1, prompt: prompt, maxTokens: maxTokens))
            await off.shutdown()

            let on = try makeEngine(fixture, mtp: true)
            let before = try on.beginForwardShapeObservation()
            let speculative = try await run(
                on, greedyRequest(id: 1, prompt: prompt, maxTokens: maxTokens))
            await on.shutdown()
            _ = completedForwardShapes(on, since: before, requireVerification: false)

            #expect(baseline.finishReason == .length)
            #expect(speculative.finishReason == .length)
            #expect(speculative.tokens.count == maxTokens)
            #expect(
                speculative.tokens == baseline.tokens,
                "maxTokens=\(maxTokens) diverged: on=\(speculative.tokens) off=\(baseline.tokens)")
        }
    }

    // MARK: - (6) Stateless target-prefix sampling

    @Test(arguments: [Float(0.7), Float(1.1)])
    func stochasticGemmaUsesActualAdapterAndTargetSamples(temperature: Float) async throws {
        let fixture = try makeFixture()
        let adapter = try Gemma4CBv2MTPDrafter(drafter: fixture.drafter, target: fixture.target)
        #expect(adapter.supportsTargetPrefixAcceptance)
        let prompt = makePromptTokens(length: 24, seed: 61, vocabSize: vocabSize)
        let request = CBv2Request(id: .init(601), promptTokens: prompt,
            sampling: .init(temperature: temperature, topP: 0.9, topK: 24, minP: 0.02, seed: 312),
            maxTokens: 32)
        let off = try makeEngine(fixture, mtp: false)
        let expected = try await run(off, request)
        await off.shutdown()
        // Serial target geometry isolates acceptance/RNG semantics from the
        // independent rectangular arithmetic policy. Actual Gemma drafting,
        // wrapped-window state and the shared accept/rollback walk still run.
        let on = try makeEngine(fixture, mtp: true, verificationMode: .serialTarget)
        let actual = try await run(on, request)
        let metrics = try #require(on.mtpMetricsSnapshot())
        await on.shutdown()
        #expect(actual.finishReason == .length && actual.tokens.count == 32)
        #expect(actual.tokens == expected.tokens)
        #expect(metrics.rounds > 0 && metrics.seedSteps > 0)
        #expect(metrics.proposedTokens > metrics.acceptedTokens,
                "the fixture must exercise discarded speculative suffixes")
        #expect(on.capacity().activeRequests == 0 && on.capacity().kvBytesReserved == 0)
    }

    @Test(arguments: [Float(0.7), Float(1.1)])
    func stochasticGemmaRectangularUsesTargetSamplesThroughWindowWrap(temperature: Float)
        async throws
    {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 24, seed: 61, vocabSize: vocabSize)
        let request = CBv2Request(id: .init(603), promptTokens: prompt,
            sampling: .init(temperature: temperature, topP: 0.9, topK: 24, minP: 0.02, seed: 312),
            maxTokens: 32)
        let off = try makeEngine(fixture, mtp: false)
        let expected = try await run(off, request)
        await off.shutdown()

        // Exercise the real Gemma adapter and automatic rectangular selection
        // at the release draft depth. Exact seeded output is a tiny-fixture
        // oracle for target-prefix sampling and rollback, not a requirement
        // that full-size BF16 models use identical arithmetic across widths.
        let on = try makeEngine(fixture, mtp: true, maxDraftTokens: 1,
            verificationMode: .automatic)
        let before = try on.beginForwardShapeObservation()
        let actual = try await run(on, request)
        let metrics = try #require(on.mtpMetricsSnapshot())
        await on.shutdown()
        _ = completedForwardShapes(on, since: before)

        #expect(expected.finishReason == .length && expected.tokens.count == 32)
        #expect(actual.finishReason == expected.finishReason)
        #expect(actual.tokens == expected.tokens)
        #expect(metrics.seedSteps > 0 && metrics.rectangularVerificationRounds > 0)
        #expect(metrics.proposedTokens > metrics.acceptedTokens,
                "the fixture must exercise discarded rectangular suffixes")
        #expect(on.capacity().activeRequests == 0 && on.capacity().kvBytesReserved == 0)
        #expect(off.capacity().activeRequests == 0 && off.capacity().kvBytesReserved == 0)
    }

    private final class ExclusionConstraint: CBv2TokenConstraint {
        let mode: CBv2TokenConstraintMode = .none
        let initialState = 0
        let maxTokens = 8
        let fallbackTokenID = 0
        func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] { [0] }
        func nextState(state: Int, tokenID: Int) -> Int? { 0 }
    }

    @Test func stochasticOptInPreservesUnsupportedTransformExclusions() async throws {
        let fixture = try makeFixture()
        let engine = try makeEngine(fixture, mtp: true)
        let base = CBv2Request(id: .init(602), promptTokens: [1, 2, 3],
            sampling: .init(temperature: 0.7, topP: 0.9, topK: 12, minP: 0.05, seed: 42),
            maxTokens: 8)
        func eligible(_ request: CBv2Request) -> Bool {
            engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.mtpBasicEligible(CBv2ScheduledRequest(
                    request: request, arrivalSeq: 1, submittedAt: Date()))
            }
        }
        #expect(eligible(base))
        var requests: [CBv2Request] = []
        var changed = base; changed.sampling.logitBias = [1: 0.5]; requests.append(changed)
        changed = base; changed.sampling.repetitionPenalty = 1.1; requests.append(changed)
        changed = base; changed.sampling.frequencyPenalty = 0.1; requests.append(changed)
        changed = base; changed.sampling.presencePenalty = 0.1; requests.append(changed)
        changed = base; changed.sampling.topLogprobs = 1; requests.append(changed)
        changed = base; changed.stopStrings = ["stop"]; requests.append(changed)
        changed = base; changed.tokenConstraint = ExclusionConstraint(); requests.append(changed)
        for request in requests { #expect(!eligible(request)) }
        await engine.shutdown()
    }

    @Test func stopStringRowsFailOpenToPlainDecode() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 16, seed: 62, vocabSize: vocabSize)
        let engine = try makeEngine(fixture, mtp: true)
        let collected = try await run(
            engine,
            greedyRequest(
                id: 1, prompt: prompt, maxTokens: 8,
                stopStrings: ["never-matched-by-null-detokenizer"]))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()
        #expect(collected.finishReason == .length)
        #expect(metrics.rounds == 0)
        #expect(metrics.seedSteps == 0)
    }

    // MARK: - (7) Cancel mid-generation leaks nothing (admission liveness)

    @Test func cancelMidRoundDoesNotBlockAdmission() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 20, seed: 71, vocabSize: vocabSize)

        let on = try makeEngine(fixture, mtp: true, verificationMode: .automatic)
        let before = try on.beginForwardShapeObservation()
        let victim = greedyRequest(id: 1, prompt: prompt, maxTokens: 512)
        let stream = try on.submit(victim)
        // Let it get well into MTP rounds, then cancel mid-flight.
        var seen = 0
        var finish: CBv2FinishReason?
        for await event in stream {
            switch event {
            case .delta(_, let tokens, _):
                seen += tokens.count
                if seen >= 8 { on.cancel(victim.id) }
            case .finished(let reason, _):
                finish = reason
            }
        }
        #expect(finish == .cancelled)

        // A leaked pendingSamples would wedge the waiting-admission loop:
        // this follow-up request must run to completion.
        let follower = try await run(on, greedyRequest(id: 2, prompt: prompt, maxTokens: 8))
        await on.shutdown()
        _ = completedForwardShapes(on, since: before)
        #expect(follower.finishReason == .length)
        #expect(follower.tokens.count == 8)
    }

    @Test func requestIDReuseAndShutdownClearMTPState() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 20, seed: 72, vocabSize: vocabSize)
        let engine = try makeEngine(fixture, mtp: true)
        let reusedID = CBv2RequestID(44)

        let first = try engine.submit(
            CBv2Request(
                id: reusedID, promptTokens: prompt,
                sampling: .init(temperature: 0), maxTokens: 256))
        var iterator = first.makeAsyncIterator()
        var seen = 0
        while let event = await iterator.next() {
            switch event {
            case .delta(_, let tokens, _):
                seen += tokens.count
                if seen >= 8 { engine.cancel(reusedID) }
            case .finished(let reason, _):
                #expect(reason == .cancelled)
                break
            }
            if seen >= 8, engine.capacity().activeRequests == 0 { break }
        }

        let reused = try await run(
            engine,
            CBv2Request(
                id: reusedID, promptTokens: prompt,
                sampling: .init(temperature: 0), maxTokens: 8))
        #expect(reused.finishReason == .length)
        await engine.shutdown()
        #expect(engine.loopForTesting.mtp?.requestStateCountForTesting == 0)
    }

    @Test func fixedDepthZeroProbesOnceThenKeepsNormalChaining() async throws {
        let fixture = try makeFixture()
        let prompt = makePromptTokens(length: 16, seed: 73, vocabSize: vocabSize)
        let engine = try makeEngine(fixture, mtp: true, maxDraftTokens: 0)
        let result = try await run(
            engine, greedyRequest(id: 1, prompt: prompt, maxTokens: 20))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        let baseline = try #require(
            metrics.costInputs.first {
                $0.decodeRowBucket == 1 && $0.depth == 0
            })
        await engine.shutdown()

        #expect(result.finishReason == .length)
        #expect(engine.chainedStepCount > 0)
        #expect(baseline.samples == 1)
        #expect(metrics.rounds == 0)
    }
}
