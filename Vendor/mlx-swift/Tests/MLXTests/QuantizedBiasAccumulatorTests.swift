import MLX
import XCTest

#if canImport(Metal)

final class QuantizedBiasAccumulatorTests: XCTestCase {
    func testBFloat16AffineBiasAccumulatesInFloat() throws {
        try checkBiasSums(dtype: .bfloat16, epsilon: 1.0 / 256.0)
    }

    func testFloat16AffineBiasAccumulatesInFloat() throws {
        try checkBiasSums(dtype: .float16, epsilon: 1.0 / 2048.0)
    }

    func testFloat32AffineBiasControl() throws {
        try checkBiasSums(dtype: .float32, epsilon: 1.0 / 256.0)
    }

    private func checkBiasSums(dtype: DType, epsilon: Float) throws {
        let quartet: [Float] = [1, epsilon, epsilon, -1]
        var cases = 0
        try MLX.Device.withDefaultDevice(.gpu) {
            try withError {
                for bits in [2, 3, 4, 5, 6, 8] {
                    for inputDimensions in [2816, 2880, 4096] {
                        for outputDimensions in [64, 35] {
                            let weight = MLXArray.zeros(
                                [outputDimensions, inputDimensions * bits / 32], dtype: .uint32)
                            let scales = MLXArray.ones(
                                [outputDimensions, inputDimensions / 64], dtype: dtype)
                            let biases = MLXArray.ones(
                                [outputDimensions, inputDimensions / 64], dtype: dtype)
                            for rows in [1, 2] {
                                let values = (0 ..< rows * inputDimensions).map { quartet[$0 % 4] }
                                let input = MLXArray(values, [rows, inputDimensions]).asType(dtype)
                                let output = quantizedMM(
                                    input, weight, scales: scales, biases: biases,
                                    groupSize: 64, bits: bits, mode: .affine)
                                eval(output)
                                let expected = Float(inputDimensions / 4) * 2 * epsilon
                                let actual = output.asArray(Float.self)
                                let label = "dtype=\(dtype) bits=\(bits) K=\(inputDimensions) N=\(outputDimensions) M=\(rows)"
                                XCTAssertEqual(output.dtype, dtype, label)
                                XCTAssertEqual(actual.count, rows * outputDimensions, label)
                                XCTAssertTrue(
                                    actual.allSatisfy { $0 == expected },
                                    "\(label): expected exact affine bias sum \(expected), got \(actual.first ?? .nan)")
                                cases += 1
                            }
                        }
                    }
                }
            }
        }
        XCTAssertEqual(cases, 72)
    }
}

#endif
