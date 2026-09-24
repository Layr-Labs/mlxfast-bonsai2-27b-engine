import Foundation
import MLX
import MLXNN
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class PrismRecurrentOwnershipTests: XCTestCase {
    func testPackedPrefillRetainsOnlyCompactConvolutionState() throws {
        let config = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data("""
            {"model_type":"qwen3_5_text","hidden_size":512,"num_hidden_layers":1,
             "intermediate_size":512,"num_attention_heads":8,"num_key_value_heads":2,
             "head_dim":64,"linear_num_value_heads":1,"linear_num_key_heads":1,
             "linear_key_head_dim":64,"linear_value_head_dim":64,"linear_conv_kernel_dim":4,
             "full_attention_interval":4,"vocab_size":4,"mtp_num_hidden_layers":0}
            """.utf8))
        let layer = Qwen35GatedDeltaNet(config)
        let packed = try HadamardQuantizedLinear(
            weight: .zeros([192, 32], dtype: .uint32),
            scales: .ones([192, 4], dtype: .float16), biases: .ones([192, 4], dtype: .float16),
            groupSize: 128, bits: 2,
            transform: SignedBlockHadamard(blockSize: 512, signs: Array(repeating: 1, count: 512)))
        layer.update(modules: ModuleChildren.unflattened([("in_proj_qkv", packed)]))
        for batch in [1, 2] {
            let owners = try (0..<batch).map { _ in
                try CBv2RecurrentRequestState(spec: config.cbv2RecurrentStateSpec(activationDType: .float32))
            }
            defer { for owner in owners { try? owner.release() } }
            let transactions = try owners.map { try $0.bind() }
            let input = MLXArray.ones([batch, 512, 512], dtype: .float32)
            let output = layer.cbv2Forward(input, modelLayerIndex: 0, recurrentState: transactions)
            let roots = try transactions.flatMap { try $0.evaluate() }
            eval([output] + roots)
            for transaction in transactions { try transaction.commit() }
            let projected = packed(input)
            eval(projected)
            for (row, owner) in owners.enumerated() {
                let conv = try XCTUnwrap(owner.state(modelLayerIndex: 0)?.conv)
                let info = try XCTUnwrap(conv.evaluatedBufferInfo())
                XCTAssertEqual(conv.shape, [1, 3, 192])
                XCTAssertEqual(conv.dtype, .float32)
                // MLX can share a small contiguous allocation within one page;
                // it must not retain the 512-row parent for a three-row carry.
                XCTAssertLessThanOrEqual(info.allocatedBytes, batch * conv.nbytes + 16384)
                XCTAssertEqual(conv.asArray(Float.self).map(\.bitPattern),
                    projected[row..<row+1, 509..<512].asArray(Float.self).map(\.bitPattern))
            }
        }
    }
}
