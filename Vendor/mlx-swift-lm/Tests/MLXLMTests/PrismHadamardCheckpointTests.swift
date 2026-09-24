import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class PrismHadamardCheckpointTests: XCTestCase {
    private func fixture() -> [String: Any] {
        [
            "schema_version": 2, "model_type": "prism_hadamard_qwen35",
            "base_model_type": "qwen3_5", "gdn_activation_layout": "grouped",
            "tensor_namespace": "mlx-vlm-qwen3_5", "hadamard_config": "hadamard.json",
            "components": ["text": true, "vision": true, "mtp": false],
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "text_config": ["mtp_num_hidden_layers": 0],
            "modules": [
                ["path": "model.embed_tokens", "block": 1024, "embedding": true, "dtype": "float16"],
                ["path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"],
            ],
        ]
    }
    private func decode(_ value: [String: Any]) throws -> PrismHadamardCheckpointConfiguration {
        try JSONDecoder().decode(PrismHadamardCheckpointConfiguration.self,
            from: JSONSerialization.data(withJSONObject: value))
    }
    func testExplicitVisionWithoutMTPContract() throws {
        let value = try decode(fixture())
        XCTAssertTrue(value.hasVision)
        XCTAssertEqual(value.modules.count, 2)
    }
    func testRejectsDifferentPackingLayoutAndInventedHeads() throws {
        for (key, replacement) in [
            ("schema_version", 1), ("base_model_type", "qwen4_exp"),
            ("gdn_activation_layout", "tiled"), ("hadamard_config", "../hadamard.json"),
            ("quantization", ["bits": 4, "group_size": 128, "mode": "affine"]),
            ("components", ["text": true, "vision": true, "mtp": true]),
            ("text_config", ["mtp_num_hidden_layers": 1]),
        ] as [(String, Any)] {
            var value = fixture()
            value[key] = replacement
            XCTAssertThrowsError(try decode(value), key)
        }
    }
    func testRejectsDuplicateAndUnsafeModulePaths() {
        var value = fixture()
        var modules = value["modules"] as! [[String: Any]]
        modules.append(modules[0])
        value["modules"] = modules
        XCTAssertThrowsError(try decode(value))
        modules.removeLast()
        modules[1]["path"] = "../lm_head"
        value["modules"] = modules
        XCTAssertThrowsError(try decode(value))
    }

    func testPackedEmbeddingAdvertisesFloatingCacheState() throws {
        let config = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data("""
            {"model_type":"qwen3_5_text","hidden_size":512,"num_hidden_layers":4,
             "intermediate_size":512,"num_attention_heads":8,"num_key_value_heads":2,
             "head_dim":64,"linear_num_value_heads":1,"linear_num_key_heads":1,
             "linear_key_head_dim":64,"linear_value_head_dim":64,"linear_conv_kernel_dim":4,
             "full_attention_interval":4,"vocab_size":4,"mtp_num_hidden_layers":0}
            """.utf8))
        let model = Qwen35TextModel(config)
        let embedding = try HadamardQuantizedEmbedding(
            weight: .zeros([4, 32], dtype: .uint32),
            scales: .ones([4, 4], dtype: .float16), biases: .zeros([4, 4], dtype: .float16),
            groupSize: 128, bits: 2,
            transform: SignedBlockHadamard(blockSize: 512, signs: Array(repeating: 1, count: 512)))
        model.update(modules: ModuleChildren.unflattened([("model.embed_tokens", embedding)]))
        XCTAssertEqual(model.cbv2CheckpointActivationDType, .float32)
        XCTAssertEqual(model.cbv2CompleteCheckpointKVDTypes, [.float32])
        XCTAssertEqual(model.cbv2RecurrentStateSpec.layers.map(\.convDType), [.float32, .float32, .float32])
    }
}
