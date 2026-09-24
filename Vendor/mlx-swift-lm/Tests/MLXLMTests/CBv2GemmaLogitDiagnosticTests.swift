import Foundation
import MLX
import MLXRandom
import Testing
@_spi(Diagnostics) @testable import MLXLMCommon
@testable import MLXLLM

@Suite("CBv2 Gemma logit diagnostics without policy capability", .serialized)
struct CBv2GemmaLogitDiagnosticTests {
    private struct Run {
        let tokens: [Int]
        let diagnostic: CBv2LogitDiagnosticSnapshot?
        let metrics: CBv2MTPMetrics?
        let policyAvailable: Bool
        let marginalPolicy: Bool
    }

    private func target() throws -> Gemma4TextModel {
        let json = """
            {"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":6,
             "intermediate_size":128,"num_attention_heads":2,"head_dim":32,
             "global_head_dim":32,"num_key_value_heads":1,"num_kv_shared_layers":2,
             "layer_types":["sliding_attention","full_attention","full_attention",
                            "sliding_attention","sliding_attention","full_attention"],
             "sliding_window":16,"final_logit_softcapping":30.0,"tie_word_embeddings":false,
             "vocab_size":256,"vocab_size_per_layer_input":256,"rms_norm_eps":1e-6,
             "hidden_size_per_layer_input":0,"use_double_wide_mlp":false}
            """
        MLXRandom.seed(0x6E_44_A)
        let target = Gemma4TextModel(try JSONDecoder.json5().decode(
            Gemma4TextConfiguration.self, from: Data(json.utf8)))
        stabilizeCBv2MTPGreedyCycleTarget(target)
        eval(target)
        return target
    }

    private func run(mtp: Bool, capture: Bool, mode: CBv2MTPVerificationMode) async throws -> Run {
        let model = try target()
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let available = adapter.cbv2MTPPolicyTopTwoAvailable
        #expect(!available, "Gemma must not gain a Qwen policy capability to support diagnostics")
        let prompt = [3, 7, 11, 19, 23]
        let expected = cbv2MTPExpectedGreedyCycle(after: prompt.last!, count: 14, vocabularySize: 256)
        let drafter: (any CBv2MTPDrafter)? = mtp ? CBv2ParityScriptedDrafter(
            script: expected, promptLength: prompt.count, offset: 0, vocabSize: 256, target: model) : nil
        let engine = EngineV2(
            model: adapter, layerKinds: model.cbv2LayerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.cbv2LayerKinds),
            sampler: CBv2GreedySampler(),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 16, maxWaiting: 4, enablePrefixCache: false),
            mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: mtp, maxDraftTokens: 2, maxSpeculativeBatch: 1,
                fixedDraftTokens: 2, verificationMode: mode))
        do {
            // Match the actual benchmark trigger: configure only after an ordinary warmup retires.
            let warmup = await cbv2SchedCollect(try engine.submit(CBv2Request(
                id: .init(500), promptTokens: [1, 2, 3], sampling: .init(temperature: 0),
                maxTokens: 1, prefixCacheEnabled: false)))
            #expect(warmup.tokens == [4])
            let idle = await cbv2SchedWait {
                do { _ = try engine.takeLogitDiagnosticSnapshot(); return true }
                catch { return false }
            }
            #expect(idle)
            let before = engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.mtp?.usesMarginalPolicy ?? false
            }
            if capture {
                try engine.configureLogitDiagnostic(.init(
                    requestID: 501, outputIndex: 3, candidateIDs: [1, 2], maximumRecords: 1))
            }
            #expect(adapter.cbv2MTPPolicyTopTwoAvailable == available)
            let after = engine.loopForTesting.onEngineQueueSync {
                engine.loopForTesting.mtp?.usesMarginalPolicy ?? false
            }
            #expect(before == after && !after)
            let output = await cbv2SchedCollect(try engine.submit(CBv2Request(
                id: .init(501), promptTokens: prompt, sampling: .init(temperature: 0),
                maxTokens: 14, prefixCacheEnabled: false)))
            #expect(output.tokens == expected)
            var snapshot: CBv2LogitDiagnosticSnapshot?
            if capture {
                let drained = await cbv2SchedWait {
                    do {
                        snapshot = try engine.takeLogitDiagnosticSnapshot()
                        return snapshot != nil
                    } catch { return false }
                }
                #expect(drained)
            }
            let metrics = engine.mtpMetricsSnapshot()
            await engine.shutdown()
            return Run(tokens: output.tokens, diagnostic: snapshot, metrics: metrics,
                       policyAvailable: available, marginalPolicy: after)
        } catch {
            await engine.shutdown()
            throw error
        }
    }

    @Test func ordinaryNonQwenAdapterCapturesConfirmedDecode() async throws {
        let control = try await run(mtp: false, capture: false, mode: .serialTarget)
        let captured = try await run(mtp: false, capture: true, mode: .serialTarget)
        #expect(control.tokens == captured.tokens)
        #expect(control.diagnostic == nil)
        #expect(!captured.policyAvailable && !captured.marginalPolicy)
        let snapshot = try #require(captured.diagnostic)
        #expect(snapshot.records.count == 1)
        #expect(snapshot.omittedRecords == 0 && snapshot.invalidVocabularyRecords == 0)
        let record = try #require(snapshot.records.first)
        #expect(record.phase == "chained_decode" && record.outcome == "confirmed")
        #expect(record.outputIndex == 3 && record.seedToken == captured.tokens[2])
        #expect(record.targetToken == captured.tokens[3] && record.argMaxID == captured.tokens[3])
        #expect(record.topTwoIDs.first == captured.tokens[3])
        #expect(record.cacheOffset == 7 && record.nanCount == 0 && record.infiniteCount == 0)
    }

    @Test(arguments: [CBv2MTPVerificationMode.serialTarget, .rectangular])
    func confirmedNonQwenMTPKeepsPolicyAndVerificationGeometry(_ mode: CBv2MTPVerificationMode) async throws {
        let control = try await run(mtp: true, capture: false, mode: mode)
        let captured = try await run(mtp: true, capture: true, mode: mode)
        #expect(control.tokens == captured.tokens)
        #expect(!captured.policyAvailable && !captured.marginalPolicy)
        let old = try #require(control.metrics), new = try #require(captured.metrics)
        #expect(old.active && new.active && old.rounds > 0)
        #expect(old.rounds == new.rounds && old.draftedTokens == new.draftedTokens)
        #expect(old.acceptedTokens == new.acceptedTokens && old.emittedTokens == new.emittedTokens)
        #expect(old.depthSelections == new.depthSelections)
        #expect(old.serialVerificationRounds == new.serialVerificationRounds)
        #expect(old.rectangularVerificationRounds == new.rectangularVerificationRounds)
        let snapshot = try #require(captured.diagnostic)
        #expect(snapshot.records.count == 1 && snapshot.omittedRecords == 0)
        let record = try #require(snapshot.records.first)
        #expect(record.phase == (mode == .serialTarget ? "serial_verify" : "rectangular_verify"))
        #expect(record.outcome == "confirmed" && record.outputIndex == 3)
        #expect(record.outputBase == 2 && record.column == 1 && record.verificationWidth == 3)
        #expect(record.draftDepth == 2 && record.acceptedDrafts == 2 && record.confirmedWidth == 3)
        #expect(record.seedToken == captured.tokens[1] && record.draftPrefix == [captured.tokens[2]])
        #expect(record.targetToken == captured.tokens[3] && record.argMaxID == captured.tokens[3])
        #expect(record.cacheOffset == 6 && record.nanCount == 0 && record.infiniteCount == 0)
    }
}
