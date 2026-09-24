// Qwen4ExpTestSupport.swift
//
// Tiny fixtures for the Qwen 3.8 Flash-Next port. Everything here is built
// from seeded random weights: no model is downloaded and nothing reads the
// real checkpoint.
//
// The configuration is deliberately small AND complete: four layers, two of
// them full attention with a QSA indexer, two gated-deltanet, and one PLE
// layer riding a recurrent layer, which is the production arrangement. The
// indexer budget is eight tokens so the keep mask actually fires inside a
// test-sized prompt.
//
// THE GATED-DELTANET HEAD DIMENSION IS NOT FREE. The shared recurrence kernel
// (`GatedDelta.swift`) computes `n_per_t = Dk / 32` and declares
// `float state[n_per_t]`, so a key head dimension below 32, or one that is not
// a multiple of it, makes the Metal shader fail to compile with a zero-length
// array. 32 is the smallest legal value; production uses 128.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

enum Qwen4ExpFixture {

    /// Indexer budget of the tiny configuration. A context longer than this
    /// makes the QSA keep mask fire.
    static let indexerBudget = 8

    static let configurationJSON = """
        {
          "model_type": "qwen4_exp_text",
          "hidden_size": 32,
          "num_hidden_layers": 4,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "vocab_size": 64,
          "rms_norm_eps": 1e-6,
          "full_attention_interval": 2,
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 16,
          "shared_expert_intermediate_size": 16,
          "linear_num_key_heads": 2,
          "linear_num_value_heads": 4,
          "linear_key_head_dim": 32,
          "linear_value_head_dim": 32,
          "linear_conv_kernel_dim": 4,
          "hc_count": 2,
          "hc_lowrank": 8,
          "indexer_n_heads": 2,
          "indexer_kv_heads": 1,
          "indexer_head_dim": 8,
          "indexer_budget": \(indexerBudget),
          "indexer_compress_ratio": 2,
          "ngram_size": 3,
          "heads_per_ngram": 2,
          "ngram_vocab_size_base": 1024,
          "make_ngram_vocab_size_divisible_by": 8,
          "split_ngram_parts": 4,
          "ple_embed_dim": 8,
          "ple_layer_ids": [1],
          "ple_conv_kernel_size": 4,
          "eos_token_id": 5,
          "partial_rotary_factor": 0.5,
          "rope_theta": 10000,
          "max_position_embeddings": 4096,
          "tie_word_embeddings": false
        }
        """

    static func configuration() throws -> Qwen4ExpTextConfiguration {
        try JSONDecoder().decode(
            Qwen4ExpTextConfiguration.self, from: Data(configurationJSON.utf8))
    }

    /// A model with seeded random weights and the n-gram row source installed.
    static func model(withMTP: Bool = true, seed: UInt64 = 7) throws -> Qwen4ExpModel {
        MLXRandom.seed(seed)
        let model = Qwen4ExpModel(text: try configuration(), withMTP: withMTP)
        eval(model)
        model.install(ngramRowSource: DeterministicNGramRowSource(rowDimensions: 2))
        return model
    }
}

/// An n-gram row source that computes rows instead of reading a table.
///
/// The contract asks only that the same ids always produce the same rows. A
/// pure function of the id satisfies it exactly and needs no fixture file.
final class DeterministicNGramRowSource: Qwen4ExpNGramHostRowSource {
    let rowDimensions: Int
    /// Serve the host-id form (the production path) unless a test wants
    /// the device-id form to compare against.
    var hostPath = true

    init(rowDimensions: Int, hostPath: Bool = true) {
        self.rowDimensions = rowDimensions
        self.hostPath = hostPath
    }

    func rows(globalIds: MLXArray) -> MLXArray {
        let ids = globalIds.asType(.float32)[.ellipsis, .newAxis]
        let lanes = MLXArray((0 ..< rowDimensions).map { Float($0) + 1 })
        return MLX.sin(ids * Float(0.017) * lanes) * Float(0.5)
    }

    func rows(globalIds: [Int], shape: [Int]) -> MLXArray {
        rows(globalIds: MLXArray(globalIds.map { Int32($0) }).reshaped(shape))
    }
}

/// The device-id path only, for host-vs-device comparisons.
final class DeviceOnlyNGramRowSource: Qwen4ExpNGramRowSource {
    let inner: DeterministicNGramRowSource
    init(rowDimensions: Int) { inner = DeterministicNGramRowSource(rowDimensions: rowDimensions) }
    var rowDimensions: Int { inner.rowDimensions }
    func rows(globalIds: MLXArray) -> MLXArray { inner.rows(globalIds: globalIds) }
}

/// One request's CBv2 state: the family's layer caches bound to contiguous
/// rows, plus the request-owned recurrent state.
final class Qwen4ExpCBv2Session {
    let model: Qwen4ExpModel
    let backend: CBv2ContiguousKVBackend
    let caches: [Qwen4ExpCBv2LayerCache]
    let rows: [CBv2SequenceKV?]
    let recurrent: CBv2RecurrentRequestState

    init(model: Qwen4ExpModel, maxLength: Int = 256) throws {
        self.model = model
        self.backend = CBv2ContiguousKVBackend(
            config: CBv2ContiguousBackendConfig(bytesCapacity: 1 << 26, kvDType: .float32))
        let kinds = model.cbv2LayerKinds
        self.rows = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: maxLength)
        let caches = model.newCacheV2().map { $0 as! Qwen4ExpCBv2LayerCache }
        for (index, cache) in caches.enumerated() {
            guard let row = rows[index] else {
                throw XCTSkip("contiguous backend vended no row for a full-attention layer")
            }
            cache.setRows([row])
        }
        self.caches = caches
        self.recurrent = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
    }

    var kvCaches: [KVCache] { caches.map { $0 as KVCache } }

    /// One committed forward. Returns the last position's logits and the
    /// whole window's multi stream.
    func forward(_ tokens: [Int]) throws -> (logits: MLXArray, multi: MLXArray) {
        let ids = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
        let evaluation = try recurrent.bind()
        let out = model.cbv2ForwardWithHidden(
            ids, caches: kvCaches, recurrentState: [evaluation], positionIds: nil)
        try evaluation.evaluate()
        try evaluation.commit()
        eval(out.logits, out.lastHidden)
        // `lastHidden` IS the pre-final-mixer multi stream for this family.
        return (out.logits[0..., -1, 0...], out.lastHidden)
    }

    /// Greedy next token, lowest id on a tie.
    func greedy(_ logits: MLXArray) -> Int {
        argMax(logits, axis: -1).item(Int.self)
    }
}

/// Greedy generation through the legacy `LLMModel` + `KVCache` path.
func qwen4ExpLegacyGreedy(
    model: Qwen4ExpModel, prompt: [Int], count: Int
) -> [Int] {
    let caches = model.makeCache()
    var ids = MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])
    var produced: [Int] = []
    for _ in 0 ..< count {
        let logits = model(ids, cache: caches)
        let next = argMax(logits[0..., -1, 0...], axis: -1)
        eval(next)
        produced.append(next.item(Int.self))
        ids = next.reshaped([1, 1]).asType(.int32)
    }
    return produced
}
