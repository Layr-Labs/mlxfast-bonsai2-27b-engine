// Qwen35MTPDraftTrimTests.swift
//
// Persistent-history and rollback coverage for the inline Qwen3.5/3.6 MTP assistant:
//  - prompt/plain target observations preserve cross-chunk transition order,
//  - k=1...4 rounds trim every speculative head row and flush trusted target
//    rows exactly once,
//  - request-owned bytes, release, discard, and isolation are exact,
//  - shortlisted draft head evaluation matches the full head on both float
//    and quantized (gathered-row qmv) paths.
//
// The assistant is built through the production `load` path from a
// fabricated inline artifact (real safetensors + config), never via
// test-only constructors.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

private func draftTrimConfigJSON(hiddenSize: Int, mtpLayers: Int = 1, experts: Int = 0) -> Data {
    Data(
        """
        {
          "model_type": "qwen3_5_moe",
          "mtplx_mtp": {
            "included": true,
            "prefix": "mtp.",
            "block_size": 3
          },
          "mtplx_mtp_quantization": {},
          "text_config": {
            "model_type": "qwen3_5_moe",
            "hidden_size": \(hiddenSize),
            "num_hidden_layers": 4,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "linear_num_value_heads": 1,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 32,
            "head_dim": 8,
            "full_attention_interval": 4,
            "num_experts": \(experts),
            "num_experts_per_tok": \(experts > 0 ? 2 : 0),
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "mtp_num_hidden_layers": \(mtpLayers)
          }
        }
        """.utf8)
}

/// Build a REAL loadable assistant: decode the text config, materialize a
/// randomly initialized `Qwen35MTPModule`, save its exact parameter set as
/// an indexed `mtp.*` shard, and load through the production path.
private func withDraftTrimAssistant(
    hiddenSize: Int = 8,
    mtpLayers: Int = 1,
    experts: Int = 0,
    _ body: (Qwen35InlineMTPAssistant, Qwen35TextModel) throws -> Void
) throws {
    let configData = draftTrimConfigJSON(
        hiddenSize: hiddenSize, mtpLayers: mtpLayers, experts: experts)
    let root = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
    let textData = try JSONSerialization.data(withJSONObject: root["text_config"]!)
    let configuration = try JSONDecoder().decode(
        Qwen35TextConfiguration.self, from: textData)

    MLXRandom.seed(1913)
    let donor = Qwen35MTPModule(configuration)
    var weights: [String: MLXArray] = [:]
    for (key, value) in donor.parameters().flattened() {
        weights["mtp." + key] = value
    }

    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("qwen-mtp-draft-trim-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try configData.write(to: directory.appendingPathComponent("config.json"))
    let shardName = "model-00001-of-00001.safetensors"
    try save(arrays: weights, url: directory.appendingPathComponent(shardName))
    let index = [
        "weight_map": Dictionary(uniqueKeysWithValues: weights.keys.map { ($0, shardName) })
    ]
    try JSONSerialization.data(withJSONObject: index)
        .write(to: directory.appendingPathComponent("model.safetensors.index.json"))

    let target = Qwen35TextModel(configuration)
    eval(target)
    let assistant = try Qwen35InlineMTPAssistant.load(from: directory, target: target)
    try body(assistant, target)
}

/// The removed double-forward staging path, reproduced verbatim as the
/// parity oracle: proposal forward + full staging re-forward, rejection
/// trims the draft column.
private struct LegacyDoubleForwardOracle {
    let assistant: Qwen35InlineMTPAssistant
    let caches: [any KVCache]

    init(_ assistant: Qwen35InlineMTPAssistant) {
        self.assistant = assistant
        self.caches = assistant.makeCache()
    }

    func draft(seedToken: Int32, carryHidden: MLXArray) -> Int32 {
        let tokens = MLXArray([seedToken]).reshaped([1, 1])
        let output = assistant.forward(
            hidden: assistant.targetFinalNorm(carryHidden), tokens: tokens, cache: caches)
        let draft = argMax(output.logits[0..., -1, 0...], axis: -1).asType(.int32)
        eval([draft, output.hidden] + caches.flatMap { $0.innerState() })
        _ = assistant.forward(
            hidden: output.hidden, tokens: draft.reshaped([1, 1]), cache: caches)
        eval(caches.flatMap { $0.innerState() })
        return draft.item(Int32.self)
    }

    func finalize(confirmed: Int) {
        let rollback = 2 - confirmed
        if rollback > 0 {
            for cache in caches {
                precondition(cache.trim(rollback) == rollback)
            }
        }
    }

    var offset: Int { caches.first?.offset ?? 0 }
}

@Suite("Qwen inline MTP draft trim", .serialized)
struct Qwen35MTPDraftTrimTests {

    private func drafterResult(
        _ assistant: Qwen35InlineMTPAssistant,
        state: any CBv2MTPRequestState,
        seedToken: Int32, carryHidden: MLXArray,
        shortlist: MLXArray? = nil
    ) -> (token: Int32, tokenArray: MLXArray, hidden: MLXArray) {
        let tokens = MLXArray([seedToken]).reshaped([1, 1])
        let result = assistant.draftStep(
            tokens: tokens, hidden: carryHidden, shortlist: shortlist,
            requestState: state)
        eval([result.tokens] + assistant.evaluationTargets(for: state))
        return (result.tokens.item(Int32.self), result.tokens.reshaped([1, 1]), result.hidden)
    }

    private func drafterDraft(
        _ assistant: Qwen35InlineMTPAssistant,
        state: any CBv2MTPRequestState,
        seedToken: Int32, carryHidden: MLXArray,
        shortlist: MLXArray? = nil
    ) -> Int32 {
        drafterResult(
            assistant, state: state, seedToken: seedToken,
            carryHidden: carryHidden, shortlist: shortlist).token
    }

    private func emptyTokens() -> MLXArray {
        MLXArray([Int32]()).reshaped([1, 0])
    }

    private func emptyHidden(_ width: Int) -> MLXArray {
        MLXArray([Float]()).reshaped([1, 0, width])
    }

    @Test("single-forward reject rounds match the double-forward staging oracle")
    func singleForwardMatchesDoubleForwardOracle() throws {
        try withDraftTrimAssistant { assistant, _ in
            MLXRandom.seed(7)
            let state = assistant.makeRequestState()
            let oracle = LegacyDoubleForwardOracle(assistant)

            // Rejection keeps both implementations on identical canonical
            // history, so each next proposal isolates single- vs double-forward
            // staging without substituting assistant hidden for target hidden.
            let outcomes: [Bool] = Array(repeating: false, count: 7)
            var seed: Int32 = 5
            for (round, accepted) in outcomes.enumerated() {
                let carryHidden = MLXRandom.normal([1, 1, 8]) * 0.3
                let result = drafterResult(
                    assistant, state: state, seedToken: seed, carryHidden: carryHidden)
                let draftNew = result.token
                let draftOracle = oracle.draft(seedToken: seed, carryHidden: carryHidden)
                #expect(draftNew == draftOracle, "round \(round)")

                let confirmed = accepted ? 2 : 1
                assistant.finalizeRound(
                    requestState: state,
                    confirmedInputTokens: confirmed,
                    committedDraftTokens: accepted ? result.tokenArray : emptyTokens(),
                    committedTargetHidden: accepted ? result.hidden : emptyHidden(8))
                oracle.finalize(confirmed: confirmed)
                // Next seed: bonus after an accept, replacement after a
                // reject — arbitrary but deterministic and shared.
                seed = accepted ? (draftNew + 7) % 32 : (draftNew + 11) % 32
            }

            // Terminal reject leaves only the same canonical logical history
            // represented by the eager staging oracle.
            #expect(state.committedInputCount == oracle.offset)
            #expect(state.stagedInputCount == 0)
        }
    }

    @Test("reject-0 leaves canonical head KV intact")
    func rejectionRestoresCarry() throws {
        try withDraftTrimAssistant { assistant, _ in
            MLXRandom.seed(11)
            let state = assistant.makeRequestState()
            let oracle = LegacyDoubleForwardOracle(assistant)

            let hidden0 = MLXRandom.normal([1, 1, 8]) * 0.3
            let draft0 = drafterDraft(
                assistant, state: state, seedToken: 3, carryHidden: hidden0)
            let oracleDraft0 = oracle.draft(seedToken: 3, carryHidden: hidden0)
            #expect(draft0 == oracleDraft0)
            // Only the trusted seed pair entered assistant KV; no deeper
            // speculative input is staged at k=1.
            #expect(state.committedInputCount == 1)
            #expect(state.stagedInputCount == 0)

            // Reject-0 trims every speculative row and queues no draft pair.
            assistant.finalizeRound(
                requestState: state, confirmedInputTokens: 1,
                committedDraftTokens: emptyTokens(),
                committedTargetHidden: emptyHidden(8))
            oracle.finalize(confirmed: 1)
            #expect(state.committedInputCount == 1)
            #expect(state.committedInputCount == oracle.offset)

            // The next round must be indistinguishable from one that never
            // proposed draft0: the independent double-forward oracle is the
            // ground truth for "no residue of the rejected pair".
            let replacement = (draft0 + 5) % 32
            let hidden1 = MLXRandom.normal([1, 1, 8]) * 0.3
            let next = drafterDraft(
                assistant, state: state, seedToken: replacement, carryHidden: hidden1)
            let control = oracle.draft(seedToken: replacement, carryHidden: hidden1)
            #expect(next == control)
        }
    }

    @Test("acceptance queues trusted target rows for one-time next-round flush")
    func acceptanceAppendsTrustedPair() throws {
        try withDraftTrimAssistant { assistant, _ in
            MLXRandom.seed(13)
            let state = assistant.makeRequestState()
            let hidden0 = MLXRandom.normal([1, 1, 8]) * 0.3
            let result = drafterResult(
                assistant, state: state, seedToken: 9, carryHidden: hidden0)
            #expect(state.committedInputCount == 1)
            assistant.finalizeRound(
                requestState: state, confirmedInputTokens: 2,
                committedDraftTokens: result.tokenArray,
                committedTargetHidden: result.hidden)
            // The accepted target transition is logically committed but
            // remains device-resident backlog until the next head forward.
            #expect(state.committedInputCount == 2)

            let bonus = (result.token + 7) % 32
            let hidden1 = MLXRandom.normal([1, 1, 8]) * 0.3
            _ = drafterDraft(
                assistant, state: state, seedToken: bonus, carryHidden: hidden1)
            // [trusted accepted draft, current carry] flush exactly once.
            #expect(state.committedInputCount == 3)
            assistant.finalizeRound(
                requestState: state, confirmedInputTokens: 1,
                committedDraftTokens: emptyTokens(),
                committedTargetHidden: emptyHidden(8))
            _ = drafterDraft(
                assistant, state: state, seedToken: (bonus + 1) % 32,
                carryHidden: MLXRandom.normal([1, 1, 8]) * 0.3)
            #expect(state.committedInputCount == 4)
        }
    }

    @Test("discard restores trusted round inputs")
    func discardRestoresTrustedInputs() throws {
        try withDraftTrimAssistant { assistant, _ in
            MLXRandom.seed(17)
            let state = assistant.makeRequestState()
            _ = drafterDraft(
                assistant, state: state, seedToken: 4,
                carryHidden: MLXRandom.normal([1, 1, 8]) * 0.3)
            assistant.discardRound(requestState: state)
            #expect(state.stagedInputCount == 0)
            // A discarded round leaves canonical history usable: the next
            // draft appends exactly one seed pair.
            _ = drafterDraft(
                assistant, state: state, seedToken: 6,
                carryHidden: MLXRandom.normal([1, 1, 8]) * 0.3)
            #expect(state.committedInputCount == 2)
        }
    }

    @Test("target final norm is applied exactly once at trusted-history ingress")
    func targetFinalNormIngress() throws {
        try withDraftTrimAssistant { assistant, target in
            target.model.norm.update(
                parameters: ModuleParameters.unflattened([
                    "weight": MLXArray([
                        Float(0.5), 0.75, 1.0, 1.5, 2.0, 2.5, 3.25, 4.0,
                    ])
                ]))
            let rawTargetHidden = MLXArray([
                Float(1), -2, 3, -4, 5, -6, 7, -8,
            ]).reshaped([1, 1, 8])
            let token = MLXArray([Int32(7)]).reshaped([1, 1])

            let state = assistant.makeRequestState()
            let actual = assistant.draftStep(
                tokens: token, hidden: rawTargetHidden, shortlist: nil,
                requestState: state)
            let normalized = assistant.targetFinalNorm(rawTargetHidden)
            let expected = assistant.moduleForward(
                hidden: normalized, tokens: token, cache: assistant.makeCache())
            let rawDirection = assistant.moduleForward(
                hidden: rawTargetHidden, tokens: token, cache: assistant.makeCache())
            let twiceNormalized = assistant.moduleForward(
                hidden: assistant.targetFinalNorm(normalized), tokens: token,
                cache: assistant.makeCache())
            eval(actual.hidden, expected, rawDirection, twiceNormalized)

            #expect(allClose(actual.hidden, expected, atol: 1e-5).item(Bool.self))
            #expect(!allClose(actual.hidden, rawDirection, atol: 1e-5).item(Bool.self))
            #expect(!allClose(actual.hidden, twiceNormalized, atol: 1e-5).item(Bool.self))

            let rawAcceptedHidden = MLXArray([
                Float(-8), 7, -6, 5, -4, 3, -2, 1,
            ]).reshaped([1, 1, 8])
            let nextRawCarry = MLXArray([
                Float(2), 3, 5, 7, 11, 13, 17, 19,
            ]).reshaped([1, 1, 8])
            let nextToken = MLXArray([Int32(11)]).reshaped([1, 1])
            assistant.finalizeRound(
                requestState: state, confirmedInputTokens: 2,
                committedDraftTokens: actual.tokens.reshaped([1, 1]),
                committedTargetHidden: rawAcceptedHidden)
            let actualNext = assistant.draftStep(
                tokens: nextToken, hidden: nextRawCarry, shortlist: nil,
                requestState: state)

            func referenceNext(
                acceptedHidden: MLXArray, carryHidden: MLXArray
            ) -> MLXArray {
                let cache = assistant.makeCache()
                let first = assistant.moduleForward(
                    hidden: normalized, tokens: token, cache: cache)
                eval([first] + cache.flatMap { $0.innerState() })
                let output = assistant.moduleForward(
                    hidden: concatenated([acceptedHidden, carryHidden], axis: 1),
                    tokens: concatenated(
                        [actual.tokens.reshaped([1, 1]), nextToken], axis: 1),
                    cache: cache)
                return output[0..., (output.dim(1) - 1)..., 0...]
            }

            let expectedNext = referenceNext(
                acceptedHidden: assistant.targetFinalNorm(rawAcceptedHidden),
                carryHidden: assistant.targetFinalNorm(nextRawCarry))
            let rawHistoryNext = referenceNext(
                acceptedHidden: rawAcceptedHidden, carryHidden: nextRawCarry)
            let doubleNormNext = referenceNext(
                acceptedHidden: assistant.targetFinalNorm(
                    assistant.targetFinalNorm(rawAcceptedHidden)),
                carryHidden: assistant.targetFinalNorm(
                    assistant.targetFinalNorm(nextRawCarry)))
            eval(actualNext.hidden, expectedNext, rawHistoryNext, doubleNormNext)
            #expect(allClose(actualNext.hidden, expectedNext, atol: 1e-5).item(Bool.self))
            #expect(!allClose(actualNext.hidden, rawHistoryNext, atol: 1e-5).item(Bool.self))
            #expect(!allClose(actualNext.hidden, doubleNormNext, atol: 1e-5).item(Bool.self))
        }
    }

    @Test("prompt observations preserve cross-chunk order and flush once")
    func promptPrimingAndCrossChunkOrder() throws {
        try withDraftTrimAssistant { assistant, _ in
            let state = assistant.makeRequestState()
            let firstTokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
            let firstHidden = MLXArray((0 ..< 32).map(Float.init)).reshaped([1, 4, 8])
            assistant.observeCommittedTarget(
                .init(tokens: firstTokens, hidden: firstHidden), requestState: state)
            #expect(state.committedInputCount == 3)

            let secondTokens = MLXArray([Int32(5), 6]).reshaped([1, 2])
            let secondHidden = MLXArray((32 ..< 48).map(Float.init)).reshaped([1, 2, 8])
            assistant.observeCommittedTarget(
                .init(tokens: secondTokens, hidden: secondHidden), requestState: state)
            #expect(state.committedInputCount == 5)

            let primed = drafterResult(
                assistant, state: state, seedToken: 7,
                carryHidden: secondHidden[0..., 1 ..< 2, 0...])
            let oracleOutput = assistant.forward(
                hidden: assistant.targetFinalNorm(
                    concatenated([firstHidden, secondHidden], axis: 1)),
                tokens: MLXArray([Int32(2), 3, 4, 5, 6, 7]).reshaped([1, 6]),
                cache: assistant.makeCache())
            let oracleLastHidden = oracleOutput.hidden[
                0..., (oracleOutput.hidden.dim(1) - 1)..., 0...]
            let oracleDraft = argMax(
                oracleOutput.logits[0..., -1, 0...], axis: -1
            ).item(Int32.self)
            #expect(primed.token == oracleDraft)
            #expect(allClose(primed.hidden, oracleLastHidden, atol: 1e-5).item(Bool.self))
            #expect(state.committedInputCount == 6)
            assistant.finalizeRound(
                requestState: state, confirmedInputTokens: 1,
                committedDraftTokens: emptyTokens(),
                committedTargetHidden: emptyHidden(8))

            _ = drafterDraft(
                assistant, state: state, seedToken: 8,
                carryHidden: MLXRandom.normal([1, 1, 8]))
            // No duplicate prompt/backlog rows on the second head forward.
            #expect(state.committedInputCount == 7)
        }
    }

    @Test("prefix restoration preserves full trusted history and exact draft chains", arguments: [0, 4])
    func exactPrefixCheckpointRestoration(experts: Int) throws {
        try withDraftTrimAssistant(experts: experts) { assistant, _ in
            let firstTokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
            let firstHidden = MLXRandom.normal([1, 4, 8])
            let secondTokens = MLXArray([Int32(5), 6, 7, 8]).reshaped([1, 4])
            let secondHidden = MLXRandom.normal([1, 4, 8])
            let donor = assistant.makeRequestState()
            assistant.observeCommittedTarget(
                .init(tokens: firstTokens, hidden: firstHidden), requestState: donor)
            let checkpoint = try #require(assistant.capturePrefixCheckpoint(
                requestState: donor, targetInputCount: 4))
            eval(checkpoint.evaluationTargets)
            #expect(checkpoint.materializedBytes == 4 * 8 * 4 + 3 * 4)
            #expect(assistant.capturePrefixCheckpoint(requestState: donor, targetInputCount: 3) == nil)
            assistant.releaseRequestState(donor)

            for accepted in 0 ... 4 {
                let cold = assistant.makeRequestState()
                assistant.observeCommittedTarget(
                    .init(tokens: firstTokens, hidden: firstHidden), requestState: cold)
                let warm = try #require(assistant.restorePrefixCheckpoint(checkpoint))
                #expect(warm.committedInputCount == 3)
                #expect(warm.stagedInputCount == 0)
                for state in [cold, warm] {
                    assistant.observeCommittedTarget(
                        .init(tokens: secondTokens, hidden: secondHidden), requestState: state)
                }
                var coldToken = MLXArray([Int32(9)]).reshaped([1, 1])
                var warmToken = coldToken
                var coldHidden = secondHidden[0..., 3 ..< 4, 0...]
                var warmHidden = coldHidden
                var drafts: [MLXArray] = []
                for _ in 0 ..< 4 {
                    let c = assistant.draftStep(
                        tokens: coldToken, hidden: coldHidden, shortlist: nil, requestState: cold)
                    let w = assistant.draftStep(
                        tokens: warmToken, hidden: warmHidden, shortlist: nil, requestState: warm)
                    eval([c.tokens, c.hidden, w.tokens, w.hidden]
                        + assistant.evaluationTargets(for: cold) + assistant.evaluationTargets(for: warm))
                    #expect(c.tokens.asArray(Int32.self) == w.tokens.asArray(Int32.self))
                    #expect(c.hidden.asArray(Float.self) == w.hidden.asArray(Float.self))
                    coldToken = c.tokens.reshaped([1, 1]); warmToken = w.tokens.reshaped([1, 1])
                    coldHidden = c.hidden; warmHidden = w.hidden
                    drafts.append(coldToken)
                }
                #expect(assistant.capturePrefixCheckpoint(requestState: warm, targetInputCount: 8) == nil)
                let acceptedTokens = accepted == 0 ? emptyTokens() : concatenated(Array(drafts.prefix(accepted)), axis: 1)
                let acceptedHidden = accepted == 0 ? emptyHidden(8) : MLXRandom.normal([1, accepted, 8])
                for state in [cold, warm] {
                    assistant.finalizeRound(
                        requestState: state, confirmedInputTokens: 1 + accepted,
                        committedDraftTokens: acceptedTokens, committedTargetHidden: acceptedHidden)
                }
                let carry = MLXRandom.normal([1, 1, 8])
                let c = drafterResult(assistant, state: cold, seedToken: 10, carryHidden: carry)
                let w = drafterResult(assistant, state: warm, seedToken: 10, carryHidden: carry)
                #expect(c.token == w.token)
                #expect(c.hidden.asArray(Float.self) == w.hidden.asArray(Float.self))
                assistant.releaseRequestState(cold)
                assistant.releaseRequestState(warm)
                #expect(cold.materializedBytes == 0 && warm.materializedBytes == 0)
            }
            // A fresh assistant instance cannot consume another loaded
            // model's otherwise shape-compatible checkpoint.
            try withDraftTrimAssistant(experts: experts) { other, _ in
                #expect(other.restorePrefixCheckpoint(checkpoint) == nil)
            }
        }
    }

    @Test("disk codec rebinds exact trusted history to a fresh loaded assistant", arguments: [0, 4])
    func diskCheckpointCodecRestoration(experts: Int) throws {
        try withDraftTrimAssistant(experts: experts) { donorAssistant, _ in
            let tokens = [1, 2, 3, 4]
            let tokenArray = MLXArray(tokens.map(Int32.init)).reshaped([1, 4])
            let hidden = MLXRandom.normal([1, 4, 8])
            let donor = donorAssistant.makeRequestState()
            donorAssistant.observeCommittedTarget(.init(tokens: tokenArray, hidden: hidden), requestState: donor)
            let checkpoint = try #require(donorAssistant.capturePrefixCheckpoint(requestState: donor, targetInputCount: 4))
            let arrays = try #require(donorAssistant.encodePrefixCheckpoint(checkpoint))
            let payload = arrays.map { ($0.asData().data, $0.shape, $0.dtype) }
            donorAssistant.releaseRequestState(donor)
            try withDraftTrimAssistant(experts: experts) { restoredAssistant, _ in
                #expect(restoredAssistant.restorePrefixCheckpoint(checkpoint) == nil)
                #expect(restoredAssistant.prefixCheckpointCodecID == donorAssistant.prefixCheckpointCodecID)
                let restoredArrays = payload.map { MLXArray($0.0, $0.1, dtype: $0.2) }
                let restored = try #require(restoredAssistant.decodePrefixCheckpoint(
                    tensors: restoredArrays, prefixTokens: tokens))
                let warm = try #require(restoredAssistant.restorePrefixCheckpoint(restored))
                let cold = restoredAssistant.makeRequestState()
                restoredAssistant.observeCommittedTarget(.init(tokens: tokenArray, hidden: hidden), requestState: cold)
                let carry = hidden[0..., 3 ..< 4, 0...]
                let expected = drafterResult(restoredAssistant, state: cold, seedToken: 5, carryHidden: carry)
                let actual = drafterResult(restoredAssistant, state: warm, seedToken: 5, carryHidden: carry)
                #expect(actual.token == expected.token)
                #expect(actual.hidden.asArray(Float.self).map(\.bitPattern) == expected.hidden.asArray(Float.self).map(\.bitPattern))
                let lazyTokens = MLX.where(MLXArray(true), restoredArrays[1], restoredArrays[1])
                #expect(try lazyTokens.evaluatedBufferInfo() == nil)
                #expect(restoredAssistant.decodePrefixCheckpoint(
                    tensors: [restoredArrays[0], lazyTokens, restoredArrays[2]], prefixTokens: tokens) == nil)
                #expect(try lazyTokens.evaluatedBufferInfo() == nil, "validation must not evaluate or allocate a token copy")
                #expect(restoredAssistant.decodePrefixCheckpoint(tensors: Array(restoredArrays.dropLast()), prefixTokens: tokens) == nil)
                #expect(restoredAssistant.decodePrefixCheckpoint(tensors: restoredArrays, prefixTokens: [1, 2, 3, 9]) == nil)
                #expect(restoredAssistant.decodePrefixCheckpoint(
                    tensors: [restoredArrays[0], restoredArrays[1].asType(.float32), restoredArrays[2]],
                    prefixTokens: tokens) == nil)
                #expect(restoredAssistant.decodePrefixCheckpoint(
                    tensors: [restoredArrays[0][0..., 0 ..< 2, 0...], restoredArrays[1], restoredArrays[2]],
                    prefixTokens: tokens) == nil)
                restoredAssistant.releaseRequestState(cold)
                restoredAssistant.releaseRequestState(warm)
                #expect(cold.materializedBytes == 0 && warm.materializedBytes == 0)
            }
        }
    }

    @Test("depths one through four trim reject, partial, and full prefixes")
    func chainedDepthTrimMatrix() throws {
        try withDraftTrimAssistant { assistant, _ in
            for depth in 1 ... 4 {
                for accepted in 0 ... depth {
                    let state = assistant.makeRequestState()
                    var inputToken = MLXArray([Int32(depth + accepted + 1)])
                        .reshaped([1, 1])
                    var inputHidden = MLXRandom.normal([1, 1, 8]) * 0.3
                    var draftTokens: [MLXArray] = []

                    for _ in 0 ..< depth {
                        let result = assistant.draftStep(
                            tokens: inputToken, hidden: inputHidden, shortlist: nil,
                            requestState: state)
                        let shaped = result.tokens.reshaped([1, 1])
                        draftTokens.append(shaped)
                        inputToken = shaped
                        inputHidden = result.hidden
                    }
                    eval(assistant.evaluationTargets(for: state))
                    #expect(state.stagedInputCount == max(0, depth - 1))

                    let trustedTokens =
                        accepted == 0
                        ? emptyTokens()
                        : concatenated(Array(draftTokens.prefix(accepted)), axis: 1)
                    let trustedHidden =
                        accepted == 0
                        ? emptyHidden(8)
                        : MLXRandom.normal([1, accepted, 8])
                    let roundHighWater = state.materializedBytes
                    assistant.finalizeRound(
                        requestState: state,
                        confirmedInputTokens: accepted + 1,
                        committedDraftTokens: trustedTokens,
                        committedTargetHidden: trustedHidden)

                    #expect(state.stagedInputCount == 0)
                    #expect(state.committedInputCount == 1 + accepted)
                    #expect(state.materializedBytes < roundHighWater)
                }
            }
        }
    }

    @Test("discard restores trusted inputs, release drops every owned byte")
    func discardAndReleaseOwnership() throws {
        try withDraftTrimAssistant { assistant, _ in
            let state = assistant.makeRequestState()
            let observedTokens = MLXArray([Int32(2), 3, 4]).reshaped([1, 3])
            let observedHidden = MLXRandom.normal([1, 3, 8])
            assistant.observeCommittedTarget(
                .init(tokens: observedTokens, hidden: observedHidden),
                requestState: state)
            #expect(state.committedInputCount == 2)

            var token = MLXArray([Int32(5)]).reshaped([1, 1])
            var hidden = observedHidden[0..., 2 ..< 3, 0...]
            for _ in 0 ..< 4 {
                let result = assistant.draftStep(
                    tokens: token, hidden: hidden, shortlist: nil, requestState: state)
                token = result.tokens.reshaped([1, 1])
                hidden = result.hidden
            }
            eval(assistant.evaluationTargets(for: state))
            let ownedBytes = assistant.evaluationTargets(for: state)
                .reduce(0) { $0 + $1.nbytes }
            #expect(state.materializedBytes == ownedBytes)
            #expect(state.committedInputCount == 3)
            #expect(state.stagedInputCount == 3)
            assistant.discardRound(requestState: state)
            #expect(state.committedInputCount == 3)
            #expect(state.stagedInputCount == 0)

            assistant.releaseRequestState(state)
            #expect(state.committedInputCount == 0)
            #expect(state.stagedInputCount == 0)
            #expect(state.materializedBytes == 0)
            #expect(assistant.evaluationTargets(for: state).isEmpty)
            assistant.releaseRequestState(state)
            #expect(state.materializedBytes == 0)
        }
    }

    @Test("auxiliary accounting includes hidden/token history and isolates requests")
    func auxiliaryAccountingAndIsolation() throws {
        try withDraftTrimAssistant { assistant, _ in
            #expect(assistant.requestStateBytesPerToken == 100)
            #expect(assistant.requestStateTokenGranularity == 256)
            #expect(assistant.requestStateTokenAllocationPadding == 4)
            #expect(assistant.maximumDraftTokens == 4)
            let specs = try #require(assistant.requestStateAllocationSpecs)
            #expect(specs.map(\.bytesPerToken) == [32, 32, 4])
            #expect(specs.map(\.allocationCount) == [2, 1, 1])
            #expect(specs.map(\.partitioned) == [false, true, true])


            let first = assistant.makeRequestState()
            let second = assistant.makeRequestState()
            assistant.observeCommittedTarget(
                .init(
                    tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4]),
                    hidden: MLXRandom.normal([1, 4, 8])),
                requestState: first)
            #expect(first.committedInputCount == 3)
            #expect(second.committedInputCount == 0)

            _ = drafterDraft(
                assistant, state: second, seedToken: 9,
                carryHidden: MLXRandom.normal([1, 1, 8]))
            #expect(second.committedInputCount == 1)
            #expect(first.committedInputCount == 3)
            assistant.releaseRequestState(first)
            #expect(first.materializedBytes == 0)
            #expect(second.committedInputCount == 1)
            #expect(second.materializedBytes > 0)
        }
    }

    @Test("first history flush stays within advertised rounded residency")
    func firstFlushResidencyBound() throws {
        try withDraftTrimAssistant { assistant, _ in
            let state = assistant.makeRequestState()
            let observedCount = 301
            let observedTokens = MLXArray(
                (1 ... observedCount).map(Int32.init)
            ).reshaped([1, observedCount])
            let observedHidden = MLXRandom.normal([1, observedCount, 8])
            assistant.observeCommittedTarget(
                .init(tokens: observedTokens, hidden: observedHidden),
                requestState: state)

            _ = drafterDraft(
                assistant, state: state, seedToken: Int32(observedCount + 1),
                carryHidden: observedHidden[
                    0..., (observedCount - 1) ..< observedCount, 0...])
            let logicalTokens = state.committedInputCount
            let chargedTokens = logicalTokens + assistant.requestStateTokenAllocationPadding
            let granularity = assistant.requestStateTokenGranularity
            let roundedTokens =
                ((chargedTokens + granularity - 1) / granularity) * granularity
            let advertisedBound = roundedTokens * assistant.requestStateBytesPerToken

            #expect(logicalTokens == observedCount)
            #expect(state.materializedBytes <= advertisedBound)
            assistant.discardRound(requestState: state)
            #expect(state.committedInputCount == observedCount)
            #expect(state.stagedInputCount == 0)
            assistant.releaseRequestState(state)
            #expect(state.materializedBytes == 0)
        }
    }
    @Test("multi-layer KV-only history fails closed before cache mutation")
    func multiLayerHistoryFailsClosed() throws {
        try withDraftTrimAssistant(mtpLayers: 2) { assistant, _ in
            let caches = assistant.makeCache()
            #expect(caches.allSatisfy { $0.offset == 0 })
            let result = assistant.moduleLastHiddenWithKVOnlyHistory(
                hidden: MLXRandom.normal([1, 2, 8]),
                tokens: MLXArray([Int32(3), 4]).reshaped([1, 2]),
                cache: caches)
            #expect(result?.shape == nil)
            #expect(caches.allSatisfy { $0.offset == 0 })
        }
    }


    @Test("full-coverage shortlist reproduces the full-head draft (float path)")
    func shortlistMatchesFullHeadFloat() throws {
        try withDraftTrimAssistant { assistant, _ in
            MLXRandom.seed(19)
            let hidden = MLXRandom.normal([1, 1, 8]) * 0.3
            let full = drafterDraft(
                assistant, state: assistant.makeRequestState(),
                seedToken: 12, carryHidden: hidden)
            // A permutation of the whole vocabulary must select the exact
            // same token through the gathered-row path AND map the local
            // argmax back to the true id.
            let permutation = MLXArray((0 ..< 32).shuffled().map(Int32.init))
            let shortlisted = drafterDraft(
                assistant, state: assistant.makeRequestState(),
                seedToken: 12, carryHidden: hidden, shortlist: permutation)
            #expect(shortlisted == full)

            // A shortlist that excludes the true argmax constrains the
            // draft to the shortlist (coverage-miss behavior: the draft is
            // simply a worse proposal, never an out-of-list token).
            let excluding = MLXArray(
                (0 ..< 32).map(Int32.init).filter { $0 != full }.prefix(5).map { $0 })
            let constrained = drafterDraft(
                assistant, state: assistant.makeRequestState(),
                seedToken: 12, carryHidden: hidden, shortlist: excluding)
            #expect(excluding.asArray(Int32.self).contains(constrained))
            #expect(constrained != full)
        }
    }

    @Test("shortlisted rows equal the full quantized head rows (gathered qmv)")
    func shortlistMatchesFullHeadQuantized() throws {
        // hidden 64 so the 4-bit group-32 quantized lm_head is legal.
        try withDraftTrimAssistant(hiddenSize: 64) { assistant, target in
            // Quantize ONLY the shared output head — exactly the production
            // shape (Qwen3.6 serves a 4-bit lm_head; the MTP module keeps
            // its own dtype).
            quantize(model: target, groupSize: 32, bits: 4) { path, module in
                path == "lm_head" && module is Linear
            }
            #expect(target.lmHead is QuantizedLinear)

            MLXRandom.seed(23)
            let hidden = MLXRandom.normal([1, 1, 64]) * 0.3

            // Row-level check: gathered-row quantized logits must equal the
            // full quantized head at those ids.
            let ids = MLXArray([3, 31, 7, 19, 0, 27].map(Int32.init))
            let gathered = assistant.shortlistLogits(hidden: hidden, ids: ids)
            let full = assistant.headLogits(hidden)
            let reference = takeAlong(full, ids.reshaped([1, 1, -1]), axis: -1)
            #expect(allClose(gathered, reference, atol: 1e-5).item(Bool.self))

            // Token-level check through draftStep.
            let fullDraft = drafterDraft(
                assistant, state: assistant.makeRequestState(),
                seedToken: 21, carryHidden: hidden)
            let permutation = MLXArray((0 ..< 32).shuffled().map(Int32.init))
            let shortlisted = drafterDraft(
                assistant, state: assistant.makeRequestState(),
                seedToken: 21, carryHidden: hidden, shortlist: permutation)
            #expect(shortlisted == fullDraft)
        }
    }
}
