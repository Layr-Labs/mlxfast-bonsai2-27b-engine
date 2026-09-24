// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import XCTest

final class GPTOSSMXFP4DecodeTailTests: XCTestCase {
    private func withFlag<T>(_ flag: String, _ body: () -> T) -> T {
        let key = "MLX_GPTOSS_MXFP4_DECODE_FAST_TAIL"
        let previous = getenv(key).map { String(cString: $0) }
        setenv(key, flag, 1)
        defer {
            if let previous { setenv(key, previous, 1) }
            else { unsetenv(key) }
        }
        return body()
    }

    func testFinal320ValuesAndUnsortedGatherRows() {
        let k = 2880
        for n in [2880, 5760] {
            let packed = (arange(32 * n * k / 8, dtype: .uint32)
                * MLXArray(UInt32(2654435761))).reshaped(32, n, k / 8)
            let scales = (remainder(arange(32 * n * k / 32, dtype: .int32), 5) + 124)
                .asType(.uint8).reshaped(32, n, k / 32)
            eval(packed, scales)
            for dtype in [DType.float32, .bfloat16] {
                for batch in [1, 2, 4, 8] {
                    let lhs = MLXArray((0..<(batch * 4)).map { UInt32($0 / 4) })
                        .reshaped(batch, 4)
                    var expertIDs: [UInt32] = []
                    for index in 0..<(batch * 4) {
                        let expert = (index / 4) * 7 + (index % 4) * 5
                        expertIDs.append(UInt32(expert % 32))
                    }
                    let rhs = MLXArray(expertIDs).reshaped(batch, 4)
                    for tailOnly in [true, false] {
                        var values = [Float](repeating: 0, count: batch * k)
                        for row in 0..<batch {
                            if tailOnly {
                                values[row * k + 2559] = 0.75
                                values[row * k + 2560] = 0.25
                                values[row * k + 2879] = -0.5
                            } else {
                                for i in 0..<k { values[row * k + i] = sin(Float(row * k + i) * 0.017) }
                            }
                        }
                        let x = MLXArray(values).reshaped(batch, 1, k).asType(dtype)
                        eval(x, lhs, rhs)
                        func forward() -> MLXArray {
                            let output = gatherQuantizedMM(
                                x, packed, scales: scales, biases: nil,
                                lhsIndices: lhs, rhsIndices: rhs,
                                transpose: true, groupSize: 32, bits: 4,
                                mode: .mxfp4, sortedIndices: false)
                            eval(output)
                            return output
                        }
                        let reference = withFlag("0", forward)
                        let actual = withFlag("1", forward)
                        XCTAssertEqual(actual.shape, reference.shape)
                        if tailOnly {
                            XCTAssertTrue((actual .== reference).all().item(Bool.self),
                                "tail edges differ: batch=\(batch),n=\(n),dtype=\(dtype)")
                        } else {
                            let worst = abs(actual.asType(.float32) - reference.asType(.float32))
                                .max().item(Float.self)
                            let scale = abs(reference.asType(.float32)).max().item(Float.self)
                            let tolerance: Float = dtype == .float32 ? 1e-4 : 0.016
                            XCTAssertTrue(worst.isFinite)
                            XCTAssertLessThanOrEqual(worst, tolerance * max(1, scale),
                                "dense difference: batch=\(batch),n=\(n),dtype=\(dtype)")
                        }
                    }
                }
            }
        }
    }
    func testOtherInputWidthKeepsLegacyGatheredMatvec() {
        let k = 2816, n = 2880
        let packed = (arange(32 * n * k / 8, dtype: .uint32)
            * MLXArray(UInt32(2654435761))).reshaped(32, n, k / 8)
        let scales = (remainder(arange(32 * n * k / 32, dtype: .int32), 5) + 124)
            .asType(.uint8).reshaped(32, n, k / 32)
        let lhs = MLXArray([UInt32(0), 0, 0, 0, 1, 1, 1, 1]).reshaped(2, 4)
        let rhs = MLXArray([UInt32(31), 2, 17, 4, 3, 0, 19, 7]).reshaped(2, 4)
        eval(packed, scales, lhs, rhs)
        for dtype in [DType.float32, .bfloat16] {
            let x = sin(arange(2 * k, dtype: .float32) * 0.017)
                .asType(dtype).reshaped(2, 1, k)
            eval(x)
            func forward() -> MLXArray {
                let output = gatherQuantizedMM(
                    x, packed, scales: scales, biases: nil,
                    lhsIndices: lhs, rhsIndices: rhs,
                    transpose: true, groupSize: 32, bits: 4,
                    mode: .mxfp4, sortedIndices: false)
                eval(output)
                return output
            }
            let reference = withFlag("0", forward)
            let actual = withFlag("1", forward)
            XCTAssertTrue((actual .== reference).all().item(Bool.self))
        }
    }

}
