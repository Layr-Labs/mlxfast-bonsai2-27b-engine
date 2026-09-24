import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class NemotronH35GraphCacheTests: XCTestCase {
    private func fixture(_ dtype: DType) throws -> NemotronHMoE {
        let data = Data("""
        {"model_type":"nemotron_h","vocab_size":128,"hidden_size":64,"num_hidden_layers":2,
         "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
         "mamba_num_heads":4,"mamba_head_dim":16,"ssm_state_size":16,
         "conv_kernel":4,"n_groups":2,"intermediate_size":128,
         "moe_intermediate_size":64,"moe_shared_expert_intermediate_size":128,
         "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
         "layers_block_type":["mamba","moe"]}
        """.utf8)
        MLXRandom.seed(35003516)
        let module = NemotronHMoE(try JSONDecoder().decode(NemotronHConfiguration.self, from: data))
        module.update(parameters: ModuleParameters.unflattened(module.parameters().flattened().map {
            ($0.0, $0.1.asType(dtype))
        }))
        quantize(model: module, groupSize: 32, bits: 4)
        eval(module)
        return module
    }

    func testDynamicInputsWeightUpdateAndModuleReplacement() throws {
        for dtype: DType in [.bfloat16, .float32] {
            let owner = try fixture(dtype)
            let cache = owner.mtpCompiledRowsCache
            let x = MLXRandom.normal([1, 3, 64]).asType(dtype)
            func check(_ input: MLXArray) -> Data {
                let expected = owner.mtpForwardRowsUncompiled(input)
                let actual = cache(input, owner: owner)
                eval(expected, actual)
                XCTAssertEqual(actual.dtype, expected.dtype)
                let bytes = actual.asData(access: .copy).data
                XCTAssertEqual(bytes, expected.asData(access: .copy).data)
                return bytes
            }
            let original = check(x)
            _ = check(x * 0.7)
            XCTAssertEqual(cache.generationCount, 1)
            let key = "switch_mlp.fc2.scales"
            let scales = try XCTUnwrap(owner.parameters().flattened().first { $0.0 == key }?.1)
            let changed = scales * 1.25
            eval(changed)
            owner.update(parameters: ModuleParameters.unflattened([(key, changed)]))
            XCTAssertNotEqual(check(x), original)
            XCTAssertEqual(cache.generationCount, 1, "In-place parameters must remain explicit graph inputs")
            let replacementOwner = try fixture(dtype)
            owner.update(modules: ModuleChildren.unflattened([("switch_mlp", replacementOwner.switchMLP)]))
            _ = check(x)
            XCTAssertEqual(cache.generationCount, 2, "Module replacement must rebuild the graph")
        }
    }

    func testCacheDoesNotRetainItsModuleOwner() throws {
        var owner: NemotronHMoE? = try fixture(.bfloat16)
        weak var weakOwner = owner
        let cache = try XCTUnwrap(owner?.mtpCompiledRowsCache)
        let output = cache(MLXArray.ones([1, 2, 64], dtype: .bfloat16), owner: try XCTUnwrap(owner))
        eval(output)
        owner = nil
        XCTAssertNil(weakOwner, "Compiled closure/state inputs must not form a Module retain cycle")
        XCTAssertEqual(cache.generationCount, 1)
    }
}
