import MLX
import MLXRandom
import XCTest
@testable import MLXLLM

final class NemotronH35SSMWindowTests: XCTestCase {
    func testNativeActivationAndEveryFP32PrefixMatchOneTokenKernelBytes() {
        MLXRandom.seed(35003509)
        for dtype: DType in [.bfloat16, .float16, .float32] {
            for (heads, headDim, stateDim, groups) in [(8, 8, 32, 2), (4, 5, 7, 2), (64, 64, 128, 8)] {
                for length in 1...8 {
                    check(dtype: dtype, heads: heads, headDim: headDim,
                          stateDim: stateDim, groups: groups, length: length)
                }
            }
        }
    }

    private func check(dtype: DType, heads: Int, headDim: Int,
                       stateDim: Int, groups: Int, length: Int) {
        func random(_ shape: [Int], _ type: DType) -> MLXArray {
            (MLXRandom.normal(shape) * 0.1).asType(type)
        }
        let x = random([1, length, heads, headDim], dtype)
        let a = random([heads], .float32)
        let b = random([1, length, groups, stateDim], dtype)
        let c = random([1, length, groups, stateDim], dtype)
        let d = random([heads], dtype)
        let dt = random([1, length, heads], dtype)
        let bias = random([heads], dtype)
        let initial = random([1, heads, headDim, stateDim], .float32)
        let result = nemotronMTPSSMWindow(hiddenStates: x, ALog: a, B: b, C: c,
            D: d, dt: dt, dtBias: bias, state: initial, timeStepLimit: (0.001, 100))
        eval(result.output, result.capturedStates)
        XCTAssertEqual(result.output.dtype, dtype)
        XCTAssertEqual(result.capturedStates.dtype, .float32)
        var state = initial
        for position in 0..<length {
            let span = position..<(position + 1)
            let (y, next) = ssmUpdate(hiddenStates: x[0..., span, 0..., 0...],
                ALog: a, B: b[0..., span, 0..., 0...], C: c[0..., span, 0..., 0...],
                D: d, dt: dt[0..., span, 0...], dtBias: bias,
                state: state, timeStepLimit: (0.001, 100))
            eval(y, next)
            let label = "dtype=\(dtype), H=\(heads), Dh=\(headDim), Ds=\(stateDim), L=\(length), position=\(position)"
            XCTAssertEqual(result.output[0..., span, 0..., 0...].asData(access: .copy).data,
                           y.asData(access: .copy).data, "output: \(label)")
            XCTAssertEqual(result.capturedStates[span, 0..., 0..., 0...].asData(access: .copy).data,
                           next.asData(access: .copy).data, "state: \(label)")
            state = next
        }
    }
}
