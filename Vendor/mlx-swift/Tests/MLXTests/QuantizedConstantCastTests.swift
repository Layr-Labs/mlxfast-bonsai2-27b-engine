import MLX
import MLXNN
import XCTest

final class QuantizedConstantCastTests: XCTestCase {
    private func makeLayer(mode: QuantizationMode = .affine) -> QuantizedLinear {
        let weights = (MLXArray(0 ..< 2048, [32, 64]).asType(.float32) / 2048).asType(.bfloat16)
        let bias = (MLXArray(0 ..< 32).asType(.float32) / 100).asType(.bfloat16)
        return QuantizedLinear(
            weight: weights, bias: bias, groupSize: mode == .affine ? 64 : 32,
            bits: mode == .affine ? 8 : 4, mode: mode)
    }

    private func reference(_ layer: QuantizedLinear, _ x: MLXArray) -> MLXArray {
        var result = quantizedMM(
            x, layer.weight, scales: layer.scales, biases: layer.biases,
            transpose: true, groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode)
        if let bias = layer.bias { result = result + bias }
        return result
    }

    func testExactOutputDtypesAndParameterTreeAcrossRepeatedCallsAndUpdates() {
        final class Parent: Module {
            let child: QuantizedLinear
            init(_ child: QuantizedLinear) { self.child = child; super.init() }
        }
        let layer = makeLayer()
        let names = layer.parameters().flattened().map(\.0).sorted()
        for dtype in [DType.bfloat16, .float32, .bfloat16, .float32] {
            let x = MLXArray.ones([2, 64], dtype: dtype)
            let actual = layer(x), expected = reference(layer, x)
            XCTAssertEqual(actual.dtype, expected.dtype)
            XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
        }
        XCTAssertEqual(layer.parameters().flattened().map(\.0).sorted(), names)
        XCTAssertEqual(layer.scales.dtype, .bfloat16)
        XCTAssertEqual(layer.biases?.dtype, .bfloat16)
        let x = MLXArray.ones([1, 64], dtype: .float32)
        eval(layer(x))
        Parent(layer).update(parameters: ModuleParameters.unflattened([
            ("child.scales", layer.scales * 2), ("child.bias", layer.bias! + 1)
        ]))
        XCTAssertTrue(arrayEqual(layer(x), reference(layer, x)).item(Bool.self))
        layer.scales._updateInternal(layer.scales * 3)
        layer.bias!._updateInternal(layer.bias! + 2)
        XCTAssertTrue(arrayEqual(layer(x), reference(layer, x)).item(Bool.self))
        XCTAssertEqual(layer.parameters().flattened().map(\.0).sorted(), names)
    }

    func testCompileStateReplacementAndMXFP4ScalesStayCorrect() {
        let layer = makeLayer()
        let x = MLXArray.ones([1, 64], dtype: .float32)
        eval(layer(x))
        let compiled = compile(inputs: [layer]) { (x: MLXArray) in layer(x) }
        XCTAssertTrue(arrayEqual(compiled(x), reference(layer, x)).item(Bool.self))
        layer.update(parameters: ModuleParameters.unflattened([("scales", layer.scales * 2)]))
        XCTAssertTrue(arrayEqual(compiled(x), reference(layer, x)).item(Bool.self))
        XCTAssertTrue(arrayEqual(layer(x), reference(layer, x)).item(Bool.self))
        let mxfp4 = makeLayer(mode: .mxfp4)
        XCTAssertEqual(mxfp4.scales.dtype, .uint8)
        XCTAssertNil(mxfp4.biases)
        XCTAssertTrue(arrayEqual(mxfp4(x), reference(mxfp4, x)).item(Bool.self))
        XCTAssertEqual(mxfp4.scales.dtype, .uint8)
    }
}
