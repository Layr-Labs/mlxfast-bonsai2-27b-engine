// Qwen4ExpAdoptTests.swift
//
// `Qwen4ExpRunner.adopt` — the seam Darkbloom uses, over a module it already
// holds. Everything the runner reports is derived from that module plus
// `config.json` and the safetensors index: no factory, no second read of the
// checkpoint, and the embedded head is BOUND to the module in memory rather
// than reloaded.
//
// No GPU and no checkpoint. The module is the tiny seeded configuration
// below, built but never evaluated, and the checkpoint directory is the same
// two-file fixture the shared adoption tests use.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("Qwen 3.8 Flash-Next adoption")
struct Qwen4ExpAdoptTests {

    // One PLE layer, and one full-attention layer of two, so `layerKinds` is
    // the COMPACT attention storage layout and the n-gram resource is
    // required. Same geometry as `Qwen4ExpNGramResourceTests`.
    private static let configurationJSON = """
        {
          "model_type": "qwen4_exp_text",
          "hidden_size": 32,
          "num_hidden_layers": 2,
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
          "indexer_budget": 8,
          "indexer_compress_ratio": 2,
          "ngram_size": 3,
          "heads_per_ngram": 2,
          "ngram_vocab_size_base": 1024,
          "make_ngram_vocab_size_divisible_by": 8,
          "split_ngram_parts": 4,
          "ple_embed_dim": 256,
          "ple_layer_ids": [1],
          "ple_conv_kernel_size": 4,
          "eos_token_id": 5,
          "partial_rotary_factor": 0.5,
          "rope_theta": 10000,
          "max_position_embeddings": 4096,
          "tie_word_embeddings": false
        }
        """

    private func makeModel(withMTP: Bool) throws -> Qwen4ExpModel {
        let configuration = try JSONDecoder().decode(
            Qwen4ExpTextConfiguration.self, from: Data(Self.configurationJSON.utf8))
        return Qwen4ExpModel(text: configuration, withMTP: withMTP)
    }

    /// The n-gram rows an in-process caller already holds. Adoption takes an
    /// already built source as it is, so this needs no file.
    private final class StubNGramRowSource: Qwen4ExpNGramRowSource {
        let rowDimensions = 64
        func rows(globalIds: MLXArray) -> MLXArray {
            MLXArray.zeros(globalIds.shape + [rowDimensions])
        }
    }

    private func options(
        ngram: Bool, preloadedDrafter: (any CBv2MTPDrafter)? = nil
    ) -> RunnerLoadOptions {
        var resources = RunnerResources()
        if ngram {
            resources[Qwen4ExpRunner.ngramRowSourceResource] = StubNGramRowSource()
        }
        return RunnerLoadOptions(resources: resources, preloadedDrafter: preloadedDrafter)
    }

    private func adopt(
        _ model: any LanguageModel,
        checkpoint: RunnerAdoptTests.MinimalCheckpoint,
        options: RunnerLoadOptions
    ) throws -> Qwen4ExpRunner {
        try Qwen4ExpRunner.adopt(
            model: model,
            tokenizer: StubTokenizer(),
            configuration: ModelConfiguration(directory: checkpoint.directory),
            directory: checkpoint.directory,
            options: options)
    }

    @Test("Adoption derives the layer kinds, the model type and the manifest")
    func adoptionDerivesTheDeclaration() throws {
        let checkpoint = try RunnerAdoptTests.MinimalCheckpoint(
            modelType: "qwen4_exp_text", eosTokenID: 5)
        defer { checkpoint.cleanUp() }
        let model = try makeModel(withMTP: false)

        let runner = try adopt(model, checkpoint: checkpoint, options: options(ngram: true))

        // Model-owned, not re-derived: the compact attention storage layout.
        #expect(runner.layerKinds == model.cbv2LayerKinds)
        // The ADOPTED module is what serves; nothing was constructed for it.
        #expect(ObjectIdentifier(runner.servingModel) == ObjectIdentifier(model))
        // Read from `config.json`, which is the authority for the hello.
        #expect(runner.loadedModelType == "qwen4_exp_text")
        #expect(runner.eosTokenIDs == [5])
        // The declaration is the static one, byte for byte: the digest the
        // fork and benchd both pin has not moved.
        #expect(runner.manifest.runnerID == "layr/qwen4exp-125b-a6b")
        #expect(runner.manifest.requiresKeepMask)
        #expect(runner.manifest.engine.supportsMTP)
        #expect(runner.manifest.engine.supportsPagedKV == false)
        #expect(
            runner.manifest.sha256Digest()
                == "0430b22f8325c9c9371910d1e14eb3c78b235932bf35fa6623a4c511dd68e180")
    }

    @Test("mtp is loaded only when the adopted module carries the head")
    func loadedDecodersFollowTheModule() throws {
        let checkpoint = try RunnerAdoptTests.MinimalCheckpoint(modelType: "qwen4_exp_text")
        defer { checkpoint.cleanUp() }

        // §6.2 rule 1: a mode is advertised only if its drafter is resident,
        // and the drafter is resident exactly when the module carries `mtp`.
        let withHead = try adopt(
            try makeModel(withMTP: true), checkpoint: checkpoint,
            options: options(ngram: true))
        #expect(withHead.loadedDecoders == [.serial, .mtp])

        let withoutHead = try adopt(
            try makeModel(withMTP: false), checkpoint: checkpoint,
            options: options(ngram: true))
        #expect(withoutHead.loadedDecoders == [.serial])
        // No head means no artifact to describe.
        #expect(withoutHead.headProvenance == nil)
    }

    @Test("A preloaded drafter is refused by name")
    func preloadedDrafterIsRefused() throws {
        let checkpoint = try RunnerAdoptTests.MinimalCheckpoint(modelType: "qwen4_exp_text")
        defer { checkpoint.cleanUp() }

        do {
            _ = try adopt(
                try makeModel(withMTP: true), checkpoint: checkpoint,
                options: options(ngram: true, preloadedDrafter: StubDrafter()))
            Issue.record("qwen4_exp accepted a drafter it does not serve")
        } catch let error as RunnerError {
            guard case .drafterUnavailable(let detail) = error else {
                Issue.record("refused with \(error), not the drafter")
                return
            }
            #expect(detail.contains("qwen4_exp"))
            #expect(detail.contains("preloaded drafter"))
        }
    }

    @Test("A PLE checkpoint with no n-gram resource is refused")
    func missingNGramResourceIsRefused() throws {
        let checkpoint = try RunnerAdoptTests.MinimalCheckpoint(modelType: "qwen4_exp_text")
        defer { checkpoint.cleanUp() }
        let model = try makeModel(withMTP: false)
        #expect(!model.pleEmbeddings.isEmpty)

        do {
            _ = try adopt(model, checkpoint: checkpoint, options: options(ngram: false))
            Issue.record("adoption ran without the n-gram table")
        } catch let error as RunnerError {
            guard case .resourceMissing(let detail) = error else {
                Issue.record("refused with \(error), not the resource")
                return
            }
            #expect(detail.hasPrefix(Qwen4ExpRunner.ngramRowSourceResource))
        }
    }

    @Test("A module of another family is refused, and nothing else was read")
    func foreignModuleIsRefused() throws {
        // The checkpoint holds ONLY config.json and the index, so reaching the
        // module check proves adoption read nothing else.
        let checkpoint = try RunnerAdoptTests.MinimalCheckpoint(modelType: "qwen4_exp")
        defer { checkpoint.cleanUp() }
        do {
            _ = try adopt(
                RunnerAdoptTests.ForeignModel(), checkpoint: checkpoint,
                options: options(ngram: true))
            Issue.record("qwen4_exp adopted a module of another family")
        } catch let error as RunnerError {
            guard case .unexpectedModel = error else {
                Issue.record("refused with \(error), not the module")
                return
            }
        }
    }
}
