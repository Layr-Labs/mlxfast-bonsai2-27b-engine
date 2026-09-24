import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class Qwen4ExpPLEDtypeTests: XCTestCase {
    /// The PLE block must hand back the model dtype. A float32 scalar in the
    /// gate promoted the output, and with it the hyper stream of every later
    /// layer, to float32.
    func testPLEOutputKeepsTheModelDtype() throws {
        let config = try Qwen4ExpFixture.configuration()
        let ngramHeads = (config.ngramSize - 1) * config.headsPerNGram
        let layer = Qwen4ExpPLELayer(config, pleLayerIndex: 0)
        layer.pleEmbedding.install(
            rowSource: DeterministicNGramRowSource(rowDimensions: config.pleEmbedDim / ngramHeads))
        for dtype in [DType.bfloat16, DType.float16] {
            layer.update(parameters: layer.parameters().mapValues { $0.asType(dtype) })
            let hidden = MLXArray.ones([1, 1, config.hcCount * config.hiddenSize]).asType(dtype)
            let ids = MLXArray([Int32(14)]).reshaped([1, 1])
            let context = MLXArray([Int32(3), Int32(config.eosTokenId)]).reshaped([1, 2])
            let out = layer(hidden, ids: ids, previousContext: context, cache: nil)
            eval(out)
            XCTAssertEqual(out.dtype, dtype, "PLE output must stay \(dtype)")
        }
    }
}
