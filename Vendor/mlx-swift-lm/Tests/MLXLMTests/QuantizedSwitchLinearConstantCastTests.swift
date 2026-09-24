import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite("QuantizedSwitchLinear constant casts", .serialized)
struct QuantizedSwitchLinearConstantCastTests {
    private func makeLayer(_ mode: QuantizationMode) -> QuantizedSwitchLinear {
        let weights = (MLXArray(0 ..< 8192, [4, 32, 64]).asType(.float32) / 8192).asType(.bfloat16)
        let bias = (MLXArray(0 ..< 128, [4, 32]).asType(.float32) / 100).asType(.bfloat16)
        return QuantizedSwitchLinear(
            SwitchLinear(inputDims: 64, outputDims: 32, numExperts: 4, weight: weights, bias: bias),
            groupSize: mode == .affine ? 64 : 32, bits: mode == .affine ? 8 : 4, mode: mode)
    }

    private func reference(_ layer: QuantizedSwitchLinear, _ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
        let output = gatherQuantizedMM(
            x, parameters["weight"]!, scales: parameters["scales"]!, biases: parameters["biases"],
            rhsIndices: indices, transpose: true, groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode)
        return output + expandedDimensions(parameters["bias"]![indices], axis: -2)
    }

    @Test func exactMathParameterUpdatesAndPackedScaleTypes() {
        let indices = MLXArray([UInt32(0), 2, 1, 2], [2, 2])
        for mode in [QuantizationMode.affine, .mxfp4] {
            let layer = makeLayer(mode)
            let names = layer.parameters().flattened().map(\.0).sorted()
            for dtype in [DType.bfloat16, .float32, .float32] {
                let x = MLXArray.ones([2, 1, 1, 64], dtype: dtype)
                let actual = layer(x, indices), expected = reference(layer, x, indices)
                #expect(actual.dtype == expected.dtype)
                #expect(arrayEqual(actual, expected).item(Bool.self))
            }
            #expect(layer.parameters().flattened().map(\.0).sorted() == names)
            let parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
            #expect(parameters["scales"]!.dtype == (mode == .affine ? .bfloat16 : .uint8))
            layer.update(parameters: ModuleParameters.unflattened([("bias", parameters["bias"]! + 1)]))
            let x = MLXArray.ones([2, 1, 1, 64], dtype: .float32)
            #expect(arrayEqual(layer(x, indices), reference(layer, x, indices)).item(Bool.self))
            parameters["bias"]!._updateInternal(parameters["bias"]! + 2)
            #expect(arrayEqual(layer(x, indices), reference(layer, x, indices)).item(Bool.self))
        }
    }

    @Test func compiledParameterReplacementUsesFreshConstants() {
        let layer = makeLayer(.affine)
        let x = MLXArray.ones([1, 1, 1, 64], dtype: .float32)
        let indices = MLXArray([UInt32(1), 1], [1, 2])
        eval(layer(x, indices))
        let compiled = compile(inputs: [layer]) { (input: MLXArray) in layer(input, indices) }
        #expect(arrayEqual(compiled(x), reference(layer, x, indices)).item(Bool.self))
        let parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
        layer.update(parameters: ModuleParameters.unflattened([("scales", parameters["scales"]! * 2)]))
        #expect(arrayEqual(compiled(x), reference(layer, x, indices)).item(Bool.self))
        #expect(arrayEqual(layer(x, indices), reference(layer, x, indices)).item(Bool.self))
    }
}
