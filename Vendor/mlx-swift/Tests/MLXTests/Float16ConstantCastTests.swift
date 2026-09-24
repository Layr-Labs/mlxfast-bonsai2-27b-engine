import Foundation
@_spi(QuantizedConstantCache) import MLX
@testable import MLXNN
import XCTest

final class Float16ConstantCastTests: XCTestCase {
    func testExplicitOptInReusesExactWideningForEveryHalfBitPattern() throws {
        let cache = ConstantArrayCastCache(enabled: true)
        let source = MLXArray(Array(UInt16.min...UInt16.max)).view(dtype: .float16)
        eval(source)
        XCTAssertNil(cache.cachedCast(source, to: .float32))
        let first = try XCTUnwrap(cache.cachedCast(source, to: .float32, allowFloat16: true))
        XCTAssertTrue(first === cache.cachedCast(source, to: .float32, allowFloat16: true))
        XCTAssertEqual(first.asData(access: .copy).data,
            source.asType(.float32).asData(access: .copy).data,
            "Includes signed zero, subnormals, infinities and NaN payloads")
        XCTAssertEqual(source.dtype, .float16)
        XCTAssertNil(cache.cachedCast(source, to: .float16, allowFloat16: true))
        XCTAssertNil(cache.cachedCast(source, to: .bfloat16, allowFloat16: true))
        XCTAssertNil(ConstantArrayCastCache(enabled: false)
            .cachedCast(source, to: .float32, allowFloat16: true))
        for dtype in [DType.uint8, .float32] {
            XCTAssertNil(cache.cachedCast(MLXArray([1, 2]).asType(dtype),
                to: .float32, allowFloat16: true))
        }
        cache.clear()
        XCTAssertFalse(first === cache.cachedCast(source, to: .float32, allowFloat16: true))
    }

    func testHalfDescriptorUpdatesAndStreamChangesInvalidate() throws {
        let cache = ConstantArrayCastCache(enabled: true)
        let source = MLXArray([Float(1.25), -2.5]).asType(.float16)
        let old = try XCTUnwrap(cache.cachedCast(source, to: .float32, allowFloat16: true))
        eval(old)
        source._updateInternal(MLXArray([Float(3), 4]).asType(.float16))
        let changed = try XCTUnwrap(cache.cachedCast(source, to: .float32, allowFloat16: true))
        XCTAssertFalse(old === changed)
        XCTAssertEqual(changed.asArray(Float.self), [3, 4])
        XCTAssertEqual(old.asArray(Float.self), [1.25, -2.5])
        source[0] = MLXArray(Float(7)).asType(.float16)
        XCTAssertEqual(cache.cachedCast(source, to: .float32, allowFloat16: true)!
            .asArray(Float.self), [7, 4])
        Device.withDefaultDevice(.cpu) {
            let cpu = cache.cachedCast(source, to: .float32, allowFloat16: true)!
            XCTAssertEqual(cpu.asArray(Float.self), [7, 4])
            Stream.withNewDefaultStream(device: .cpu) {
                let other = cache.cachedCast(source, to: .float32, allowFloat16: true)!
                XCTAssertFalse(cpu === other)
                XCTAssertEqual(other.asArray(Float.self), [7, 4])
            }
        }
    }

    func testHalfCompileAndGradientInputsAreNotRetained() {
        let cache = ConstantArrayCastCache(enabled: true)
        let f: (MLXArray) -> MLXArray = { x in
            (cache.cachedCast(x, to: .float32, allowFloat16: true)
                ?? x.asType(.float32)).square().sum()
        }
        let compiled = compile(f)
        for value in [Float(2), 3, -4] {
            let x = MLXArray([value, value]).asType(.float16)
            XCTAssertEqual(compiled(x).item(Float.self), value * value * 2)
            XCTAssertEqual(grad(f)(x).asArray(Float.self), [value * 2, value * 2])
        }
    }

    private func layer(bits: Int = 2, group: Int = 128, block: Int = 1024)
        throws -> HadamardQuantizedLinear {
        let width = 1024
        let transform = try SignedBlockHadamard(blockSize: block,
            signs: (0..<width).map { $0 % 3 == 0 ? -1 : 1 })
        let weight = MLXArray((0..<32 * width).map { Float(($0 * 13) % 31 - 15) / 16 },
            [32, width]).asType(.float16)
        let (packed, scales, offsets) = quantized(weight, groupSize: group, bits: bits)
        let bias = (MLXArray(0..<32).asType(.float32) / Float(100)).asType(.float16)
        return try HadamardQuantizedLinear(weight: packed, bias: bias, scales: scales,
            biases: offsets, groupSize: group, bits: bits, transform: transform,
            gdnLayout: HadamardGDNLayout(width: width, keyHeads: 2, valueHeads: 4))
    }

    private func reference(_ layer: HadamardQuantizedLinear, _ x: MLXArray) -> MLXArray {
        var output = quantizedMM(layer.transform(layer.gdnLayout.map { $0(x) } ?? x),
            layer.weight, scales: layer.scales, biases: layer.biases,
            groupSize: layer.groupSize, bits: layer.bits)
        if let bias = layer.bias { output = output + bias }
        return output
    }

    func testPackedLayerExactRepeatedCallsParameterTreeAndMutation() throws {
        final class Parent: Module {
            let child: HadamardQuantizedLinear
            init(_ child: HadamardQuantizedLinear) { self.child = child; super.init() }
        }
        let layer = try layer()
        XCTAssertEqual(layer.permitsFloat16ConstantReuse,
            ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_F16_CONSTANT_CACHE"] == "1")
        let names = layer.parameters().flattened().map(\.0).sorted()
        for dtype in [DType.float32, .float16, .bfloat16, .float32] {
            for rows in [1, 129] {
                let input = (MLXArray(0..<rows * 1024).reshaped(1, rows, 1024)
                    .asType(.float32) / Float(1024)).asType(dtype)
                for _ in 0..<2 {
                    let actual = layer(input), expected = reference(layer, input)
                    XCTAssertEqual(actual.dtype, expected.dtype)
                    XCTAssertEqual(actual.asData(access: .copy).data,
                        expected.asData(access: .copy).data)
                }
            }
        }
        XCTAssertEqual(layer.parameters().flattened().map(\.0).sorted(), names)
        let input = MLXArray.ones([1, 1024], dtype: .float32)
        Parent(layer).update(parameters: ModuleParameters.unflattened([
            ("child.scales", (layer.scales * 2).asType(.float16)),
            ("child.bias", (layer.bias! + 1).asType(.float16)),
        ]))
        XCTAssertEqual(layer(input).asData(access: .copy).data,
            reference(layer, input).asData(access: .copy).data)
        layer.scales._updateInternal((layer.scales * 3).asType(.float16))
        layer.biases![0, 0] = MLXArray(Float(2)).asType(.float16)
        XCTAssertEqual(layer(input).asData(access: .copy).data,
            reference(layer, input).asData(access: .copy).data)
        XCTAssertEqual(layer.scales.dtype, .float16)
        XCTAssertEqual(layer.parameters().flattened().map(\.0).sorted(), names)
        XCTAssertTrue(layer.trainableParameters().flattened().isEmpty)
    }

    func testPackedCompileStateReplacementAndIneligibleShapes() throws {
        let layer = try layer()
        let input = MLXArray.ones([1, 1024], dtype: .float32)
        eval(layer(input))
        let compiled = compile(inputs: [layer]) { (x: MLXArray) in layer(x) }
        for multiplier in [Float(1), 2] {
            layer.update(parameters: ModuleParameters.unflattened([
                ("scales", (layer.scales * multiplier).asType(.float16))]))
            XCTAssertEqual(compiled(input).asData(access: .copy).data,
                reference(layer, input).asData(access: .copy).data)
        }
        for other in [try self.layer(bits: 4), try self.layer(group: 64), try self.layer(block: 128)] {
            XCTAssertFalse(other.permitsFloat16ConstantReuse)
            XCTAssertEqual(other(input).asData(access: .copy).data,
                reference(other, input).asData(access: .copy).data)
        }
    }
}
