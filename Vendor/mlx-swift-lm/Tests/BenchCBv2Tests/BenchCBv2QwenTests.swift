import Foundation
import MLXLLM
import MLXLMCommon
import Testing

@testable import BenchCBv2Core

@Suite("BenchCBv2 Qwen3.6 hooks")
struct BenchCBv2QwenTests {
    private func textConfiguration() throws -> Qwen35TextConfiguration {
        let json = """
            {
              "model_type": "qwen3_5_moe",
              "hidden_size": 64,
              "num_hidden_layers": 8,
              "intermediate_size": 128,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 16,
              "linear_num_value_heads": 2,
              "linear_num_key_heads": 1,
              "linear_key_head_dim": 4,
              "linear_value_head_dim": 4,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 64,
              "full_attention_interval": 4,
              "num_experts": 8,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 16,
              "shared_expert_intermediate_size": 16
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    private func outerConfiguration() throws -> Qwen35Configuration {
        let text = try textConfiguration()
        let textObject = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(text)) as? [String: Any])
        return try JSONDecoder().decode(
            Qwen35Configuration.self,
            from: JSONSerialization.data(withJSONObject: [
                "model_type": "qwen3_5_moe",
                "text_config": textObject,
            ]))
    }

    @Test("text model hooks preserve model identity and original layer indices")
    func textModelHooksPreserveIdentityAndLayerIndices() throws {
        let model = Qwen35TextModel(try textConfiguration())
        let hooks = try #require(v2Hooks(for: model))
        let hookedModel = try #require(hooks.model as? Qwen35TextModel)

        #expect(ObjectIdentifier(hookedModel) == ObjectIdentifier(model))
        #expect(hooks.layerKinds == model.cbv2LayerKinds)
        #expect(hooks.layerKinds.compactMap(\.modelLayerIndex) == [3, 7])

        let caches = try hooks.buildCaches { index, kind in
            CBv2LayerCache(layerIndex: index, kind: kind)
        }
        #expect(caches.map(\.layerIndex) == [3, 7])
    }

    @Test("outer MoE model hooks preserve wrapper identity")
    func outerMoEModelHooksPreserveWrapperIdentity() throws {
        let model = Qwen35MoEModel(try outerConfiguration())
        let hooks = try #require(v2Hooks(for: model))
        let hookedModel = try #require(hooks.model as? Qwen35MoEModel)

        #expect(ObjectIdentifier(hookedModel) == ObjectIdentifier(model))
        #expect(hooks.layerKinds == model.cbv2LayerKinds)
        #expect(hooks.optimizations == ModelOptimizationProvenance(
            layer18Requested: false,
            layer18Effective: false,
            layer18Interval: nil,
            weightedUnsortRequested: false,
            weightedUnsortEffective: false,
            safeR1GeometryEligible: false))
    }
}
