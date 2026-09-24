// CBv2MTPRoundAuditTests.swift
//
// Per-round acceptance/rollback audit records. The free run proves the
// records reconcile with the committed stream and with the counters the
// same finalize boundary reports; the unit test proves the retention cap.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

private final class CBv2AuditCapPrepared: CBv2MTPPreparedCapture {}

private final class CBv2AuditCapDrafter: CBv2MTPDrafter {
    let mtpTargetIdentity: ObjectIdentifier?

    init(target: CBv2AuditCapModel) {
        self.mtpTargetIdentity = ObjectIdentifier(target)
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        CBv2AuditCapPrepared()
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        (tokens, hidden)
    }
}

private final class CBv2AuditCapModel: CBv2MTPSteppableModel {
    let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 0)
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        fatalError("audit cap test does not execute model graphs")
    }

    func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        fatalError("audit cap test does not execute model graphs")
    }
}

@Suite("CBv2MTPRoundAudit", .serialized)
struct CBv2MTPRoundAuditTests {
    private let vocabSize = 256
    private let hiddenSize = 64
    private let slidingWindow = 16
    private let k = 2

    private func targetConfig() throws -> Gemma4TextConfiguration {
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
                "tie_word_embeddings": true,
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

    private func makeTarget(seed: UInt64 = 0x9A7E) throws -> Gemma4TextModel {
        MLXRandom.seed(seed)
        let target = Gemma4TextModel(try targetConfig())
        eval(target)
        return target
    }

    private func makeEngine(
        target: Gemma4TextModel, drafter: (any CBv2MTPDrafter)?
    ) -> EngineV2 {
        let kinds = target.cbv2LayerKinds
        let mtpConfig = CBv2MTPConfig(
            enabled: drafter != nil, maxDraftTokens: k,
            maxSpeculativeBatch: 4,
            fixedDraftTokens: k,
            verificationMode: .serialTarget,
            maxAutomaticRectangularTokens: 8)
        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(target),
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2DefaultSampler(),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 16),
            mtpDrafter: drafter,
            mtpConfig: mtpConfig)
    }

    private func request(prompt: [Int], maxTokens: Int) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(1), promptTokens: prompt,
            sampling: CBv2SamplingParams(temperature: 0), maxTokens: maxTokens)
    }

    private func run(
        _ engine: EngineV2, _ request: CBv2Request
    ) async throws -> CBv2SchedCollected {
        await cbv2SchedCollect(try engine.submit(request))
    }

    /// Depth-2 free run over the deterministic fixture. Every finalized
    /// verify round must leave exactly one audit record that reconciles with
    /// the committed token stream and with the counters the same boundary
    /// reports.
    @Test func freeRunAuditsReconcileWithCommittedStream() async throws {
        let target = try makeTarget()
        let prompt = makePromptTokens(length: 21, seed: 403, vocabSize: vocabSize)

        let baselineEngine = makeEngine(target: target, drafter: nil)
        let probe = try await run(
            baselineEngine, request(prompt: prompt, maxTokens: 32))
        await baselineEngine.shutdown()

        let maxTokens = 24
        let engine = makeEngine(
            target: target,
            drafter: CBv2ParityScriptedDrafter(
                script: probe.tokens, promptLength: prompt.count, offset: 0,
                vocabSize: vocabSize, target: target,
                offsetsByStep: [0, 1]))
        let on = try await run(engine, request(prompt: prompt, maxTokens: maxTokens))
        let metrics = try #require(engine.mtpMetricsSnapshot())
        await engine.shutdown()

        #expect(on.tokens == Array(probe.tokens.prefix(maxTokens)))
        #expect(metrics.rounds > 2)

        // (a) One record per verify round the counters already report.
        #expect(metrics.roundAudits.count == metrics.rounds)
        #expect(metrics.roundAudits.count < CBv2MTPRoundAuditRecord.retainedRecordCap)

        // (b/c) Per-record shape, boundary invariant, and commit reconciliation.
        var confirmedTotal = 0
        var acceptedTotal = 0
        var previousGenerated = 0
        for audit in metrics.roundAudits {
            #expect(audit.requestID == 1)
            #expect(audit.k == k)
            #expect(audit.draftTokens.count == k)
            #expect(audit.targetTokens.count == 1 + k)
            #expect(audit.confirmed >= 1)
            #expect(audit.confirmed <= 1 + k)
            #expect(audit.rejected == (1 + k) - audit.confirmed)

            var walk = 0
            while walk < k, audit.draftTokens[walk] == audit.targetTokens[walk] {
                walk += 1
            }
            #expect(audit.accepted == walk)

            // Boundary invariant: every token computed except the new carry.
            #expect(audit.numComputedAfter == audit.tokensCountAfter - 1)
            #expect(audit.tokensCountAfter == prompt.count + audit.generatedAfter)

            // `confirmed` is exactly what this round committed to the stream:
            // the record's own emitted prefix, anchored at `generatedAfter`.
            // Rounds advance the stream by at least their own commits; seed
            // steps may emit between two rounds.
            #expect(audit.generatedAfter >= previousGenerated + audit.confirmed)
            let start = audit.generatedAfter - audit.confirmed
            #expect(
                Array(audit.targetTokens.prefix(audit.confirmed))
                    == Array(on.tokens[start ..< audit.generatedAfter]))
            previousGenerated = audit.generatedAfter

            confirmedTotal += audit.confirmed
            acceptedTotal += min(audit.accepted, audit.confirmed)
        }
        #expect(confirmedTotal == metrics.emittedTokens)
        #expect(acceptedTotal == metrics.acceptedTokens)

        let finished = metrics.roundAudits.filter { $0.finishReason != nil }
        #expect(finished.count <= 1)
    }

    /// Retention is bounded and drops the oldest record first.
    @Test func retainedRecordCapEvictsOldestFirst() throws {
        let model = CBv2AuditCapModel()
        let driver = try #require(
            CBv2MTPRoundDriver.build(
                model: model,
                drafter: CBv2AuditCapDrafter(target: model),
                config: CBv2MTPConfig(
                    enabled: true, maxDraftTokens: k,
                    maxSpeculativeBatch: 1, fixedDraftTokens: k)))

        let cap = CBv2MTPRoundAuditRecord.retainedRecordCap
        let overflow = 16
        for index in 0 ..< (cap + overflow) {
            driver.recordRound(
                drafted: k, accepted: 0, emitted: 1,
                audit: CBv2MTPRoundAuditRecord(
                    requestID: UInt64(index), k: k,
                    draftTokens: [0, 0], targetTokens: [1, 1, 1],
                    accepted: 0, confirmed: 1, rejected: k,
                    tokensCountAfter: index + 1, numComputedAfter: index,
                    generatedAfter: index + 1, finishReason: nil))
        }

        let audits = driver.metricsSnapshot().roundAudits
        #expect(audits.count == cap)
        #expect(audits.first?.requestID == UInt64(overflow))
        #expect(audits.last?.requestID == UInt64(cap + overflow - 1))
    }
}
