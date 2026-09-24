// NemotronH35AdoptTests.swift
//
// `NemotronH35Runner.adopt` — the seam Darkbloom uses, over a module it
// already holds. Everything the runner reports is derived from that module
// plus `config.json` and the safetensors index: no factory and no second read
// of the checkpoint.
//
// No GPU and no checkpoint. The modules are the tiny seeded configurations
// below, built but never evaluated, and the checkpoint directory is the same
// two-file fixture the shared adoption tests use. Anything that needs the
// real `mtp.*` block — loading the assistant, hashing its shards — needs
// weights and is not tested here.

import Foundation
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("Nemotron 3.5 Lightning adoption")
struct NemotronH35AdoptTests {

    /// One attention block and one Mamba block, so `layerKinds` is the
    /// COMPACT attention storage layout and `modelLayerIndex` is not the
    /// storage position.
    private static let lightningConfigurationJSON = """
        {
          "model_type": "nemotron_h",
          "layers_block_type": ["attention", "mamba"],
          "vocab_size": 64,
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "mamba_num_heads": 2,
          "mamba_head_dim": 8,
          "ssm_state_size": 8,
          "conv_kernel": 4,
          "n_groups": 1,
          "intermediate_size": 32,
          "moe_intermediate_size": 16,
          "moe_shared_expert_intermediate_size": 16,
          "n_routed_experts": 2,
          "num_experts_per_tok": 1,
          "num_nextn_predict_layers": 1,
          "mtp_layers_block_type": ["attention", "moe"]
        }
        """

    private func makeLightningModel() throws -> NemotronH35Model {
        NemotronH35Model(
            try JSONDecoder().decode(
                NemotronH35Configuration.self,
                from: Data(Self.lightningConfigurationJSON.utf8)))
    }

    /// The LEGACY Nemotron Nano contract: the same `model_type`, no
    /// `layers_block_type`, so the factory builds `NemotronHModel` and this
    /// runner must not serve it.
    private func makeLegacyModel() -> NemotronHModel {
        NemotronHModel(
            NemotronHConfiguration(
                vocabSize: 64, hiddenSize: 32, numHiddenLayers: 2,
                numAttentionHeads: 4, numKeyValueHeads: 2,
                mambaNumHeads: 2, mambaHeadDim: 8, ssmStateSize: 8,
                convKernel: 4, nGroups: 1, intermediateSize: 32,
                moeIntermediateSize: 16, moeSharedExpertIntermediateSize: 16,
                nRoutedExperts: 2, numExpertsPerTok: 1,
                hybridOverridePattern: "*M", headDim: 8))
    }

    private func adopt(
        _ model: any LanguageModel,
        checkpoint: RunnerAdoptTests.MinimalCheckpoint,
        options: RunnerLoadOptions = RunnerLoadOptions()
    ) throws -> NemotronH35Runner {
        try NemotronH35Runner.adopt(
            model: model,
            tokenizer: StubTokenizer(),
            configuration: ModelConfiguration(directory: checkpoint.directory),
            directory: checkpoint.directory,
            options: options)
    }

    @Test("Adoption derives the layer kinds, the model type and the manifest")
    func adoptionDerivesTheDeclaration() throws {
        let checkpoint = try RunnerAdoptTests.MinimalCheckpoint(
            modelType: "nemotron_h", eosTokenID: 5)
        defer { checkpoint.cleanUp() }
        let model = try makeLightningModel()

        let runner = try adopt(model, checkpoint: checkpoint)

        // Model-owned, not re-derived: only the attention block owns KV.
        #expect(runner.layerKinds == model.cbv2LayerKinds)
        #expect(runner.layerKinds.map(\.modelLayerIndex) == [0])
        // The ADOPTED module is what serves; nothing was constructed for it.
        #expect(ObjectIdentifier(runner.servingModel) == ObjectIdentifier(model))
        // Read from `config.json`, which is the authority for the hello.
        #expect(runner.loadedModelType == "nemotron_h")
        #expect(runner.eosTokenIDs == [5])
        // The declaration is the static one: contiguous only, MTP declared.
        #expect(runner.manifest.runnerID == "layr/nemotron35-lightning")
        #expect(runner.manifest.kvBackends == [.contiguous])
        #expect(runner.manifest.engine.supportsMTP)
        #expect(runner.manifest.recurrentLayers)
        #expect(
            runner.manifest.sha256Digest()
                == "10b8718181feb354dd6b0bd75e6a0df554f23313d3dfe3d018ccf91f77b74c73")
    }

    @Test("mtp is loaded only when a drafter is resident")
    func loadedDecodersFollowTheDrafter() throws {
        let checkpoint = try RunnerAdoptTests.MinimalCheckpoint(modelType: "nemotron_h")
        defer { checkpoint.cleanUp() }

        // §6.2 rule 1: a mode is advertised only if its drafter is resident.
        // The head is read by `loadDrafter`, which needs the `mtp.*` block;
        // adoption itself only binds what it was handed.
        let serialOnly = try adopt(try makeLightningModel(), checkpoint: checkpoint)
        #expect(serialOnly.loadedDecoders == [.serial])
        #expect(serialOnly.headProvenance == nil)

        let withDrafter = try adopt(
            try makeLightningModel(), checkpoint: checkpoint,
            options: RunnerLoadOptions(preloadedDrafter: StubDrafter()))
        #expect(withDrafter.loadedDecoders == [.serial, .mtp])
        // The fixture index carries no `mtp.*` tensor, so the checkpoint has
        // no embedded head to attribute — nil, not a digest of the target.
        #expect(withDrafter.headProvenance == nil)
    }

    @Test("A legacy nemotron_h checkpoint without layers_block_type is refused")
    func legacyNemotronIsRefused() throws {
        // The Lightning configuration REQUIRES `layers_block_type`, so a
        // checkpoint without it decodes to the legacy module — which claims
        // the same `model_type` and is not this runner's.
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(
                NemotronH35Configuration.self,
                from: Data(
                    """
                    {"model_type":"nemotron_h","vocab_size":64,"hidden_size":32,
                     "num_hidden_layers":2,"num_attention_heads":4,
                     "num_key_value_heads":2,"mamba_num_heads":2,"mamba_head_dim":8,
                     "ssm_state_size":8,"conv_kernel":4,"n_groups":1,
                     "intermediate_size":32,"moe_intermediate_size":16,
                     "moe_shared_expert_intermediate_size":16,"n_routed_experts":2,
                     "num_experts_per_tok":1,"hybrid_override_pattern":"*M"}
                    """.utf8))
        }

        let checkpoint = try RunnerAdoptTests.MinimalCheckpoint(modelType: "nemotron_h")
        defer { checkpoint.cleanUp() }
        do {
            _ = try adopt(makeLegacyModel(), checkpoint: checkpoint)
            Issue.record("the Lightning runner adopted a legacy nemotron_h module")
        } catch let error as RunnerError {
            guard case .unexpectedModel(let detail) = error else {
                Issue.record("refused with \(error), not the module")
                return
            }
            #expect(detail.contains("layers_block_type"))
        }
    }

    @Test("A module of another family is refused, and nothing else was read")
    func foreignModuleIsRefused() throws {
        // The checkpoint holds ONLY config.json and the index, so reaching the
        // module check proves adoption read nothing else.
        let checkpoint = try RunnerAdoptTests.MinimalCheckpoint(modelType: "nemotron_h")
        defer { checkpoint.cleanUp() }
        do {
            _ = try adopt(RunnerAdoptTests.ForeignModel(), checkpoint: checkpoint)
            Issue.record("the Lightning runner adopted a module of another family")
        } catch let error as RunnerError {
            guard case .unexpectedModel = error else {
                Issue.record("refused with \(error), not the module")
                return
            }
        }
    }
}
