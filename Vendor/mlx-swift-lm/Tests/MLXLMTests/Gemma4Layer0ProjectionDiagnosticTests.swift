import Foundation
import MLX
@_spi(Diagnostics) @testable import MLXLLM
import MLXNN
import MLXRandom
import Testing

@Suite("Gemma layer-zero actual projection diagnostic", .serialized)
struct Gemma4Layer0ProjectionDiagnosticTests {
    private func model(perLayerInput: Int = 0) throws -> Gemma4TextModel {
        let json = """
        {"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":2,
         "intermediate_size":128,"num_attention_heads":2,"head_dim":64,
         "global_head_dim":64,"num_key_value_heads":1,"num_kv_shared_layers":0,
         "layer_types":["sliding_attention","full_attention"],"sliding_window":16,
         "vocab_size":128,"vocab_size_per_layer_input":128,"hidden_size_per_layer_input":\(perLayerInput)}
        """
        MLXRandom.seed(37)
        return Gemma4TextModel(try JSONDecoder().decode(Gemma4TextConfiguration.self, from: Data(json.utf8)))
    }

    @Test func refusesUnquantizedAndInvalidInputsWithoutAForward() throws {
        let target = try model()
        for tokens in [[], [3], [3, 7, 11], [-1, 3], [128, 3]] {
            #expect(throws: (any Error).self) { try target.diagnosticLayer0Projections(tokens: tokens) }
        }
        #expect(throws: (any Error).self) { try target.diagnosticLayer0Projections(tokens: [3, 7]) }
        let perLayer = try model(perLayerInput: 64)
        #expect(throws: (any Error).self) { try perLayer.diagnosticLayer0Projections(tokens: [3, 7]) }
    }

    @Test func capturesBothNativeShapesWithTheLoadedParameters() throws {
        let target = try model()
        target.update(parameters: ModuleParameters.unflattened(
            target.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        quantize(model: target, groupSize: 64, bits: 4)
        eval(target)
        let before = Dictionary(uniqueKeysWithValues: target.parameters().flattened().map {
            ($0.0, $0.1.asData(access: .copy).data)
        })
        let capture = try target.diagnosticLayer0Projections(tokens: [3, 7])
        #expect(capture.tensors.count == 10)
        for name in ["embedding", "normalized", "q", "k", "v"] {
            let single = try #require(capture.tensors["m1." + name])
            let rectangular = try #require(capture.tensors["m2." + name])
            #expect(single.dim(0) == 1 && single.dim(1) == 1)
            #expect(rectangular.dim(0) == 1 && rectangular.dim(1) == 2)
            #expect(single.dtype == .bfloat16 && rectangular.dtype == .bfloat16)
            if ["embedding", "normalized"].contains(name) {
                #expect(single.asData(access: .copy).data
                    == rectangular[0..., 0..<1, 0...].asData(access: .copy).data)
            }
        }
        for name in ["q", "k", "v"] {
            #expect(capture.quantization[name]?.bits == 4 && capture.quantization[name]?.groupSize == 64)
            for parameter in ["weight", "scales", "biases"] {
                let actual = try #require(capture.parameters[name + "." + parameter])
                #expect(actual.asData(access: .copy).data
                    == before["model.layers.0.self_attn.\(name)_proj.\(parameter)"])
            }
        }
        for (name, parameter) in target.parameters().flattened() {
            #expect(parameter.asData(access: .copy).data == before[name])
        }
    }
}
