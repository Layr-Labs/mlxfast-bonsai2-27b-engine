// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import XCTest

final class GPTOSSMXFP4KernelTests: XCTestCase {
    private let key = "MLX_GPTOSS_MXFP4_PREFILL_TILE"

    private func withTile<T>(_ tile: String, _ body: () throws -> T) rethrows -> T {
        let previous = getenv(key).map { String(cString: $0) }
        setenv(key, tile, 1)
        defer {
            if let previous { setenv(key, previous, 1) }
            else { unsetenv(key) }
        }
        return try body()
    }

    func testPrefillTilesPreserveFragmentedExpertsAndPartialRows() {
        let k = 2880
        for n in [2880, 5760] {
            // Legal finite MXFP4 payloads, distinct across experts and columns.
            let packed = (arange(32 * n * k / 8, dtype: .uint32)
                * MLXArray(UInt32(2654435761))).reshaped(32, n, k / 8)
            let scales = (remainder(arange(32 * n * k / 32, dtype: .int32), 5) + 124)
                .asType(.uint8).reshaped(32, n, k / 32)
            eval(packed, scales)
            for dtype in [DType.float32, .bfloat16] {
                for rows in (n == 2880 ? [128, 131, 257] : [131]) {
                    // Native sorted-RHS route requires M=1 and B/E >= 4.
                    XCTAssertGreaterThanOrEqual(rows / 32, 4)
                    let indices = MLXArray((0..<rows).map { UInt32($0 * 32 / rows) })
                    let x = sin(arange(rows * k, dtype: .float32) * 0.017)
                        .reshaped(rows, 1, k).asType(dtype)
                    eval(x, indices)
                    func forward() -> MLXArray {
                        let output = gatherQuantizedMM(
                            x, packed, scales: scales, biases: nil,
                            rhsIndices: indices, transpose: true,
                            groupSize: 32, bits: 4, mode: .mxfp4,
                            sortedIndices: true)
                        eval(output)
                        return output
                    }
                    let reference = withTile("legacy", forward)
                    let magnitude = abs(reference.asType(.float32)).max().item(Float.self)
                    for tile in ["m32n32k32"] {
                        let actual = withTile(tile, forward)
                        XCTAssertEqual(actual.shape, reference.shape)
                        let worst = abs(actual.asType(.float32) - reference.asType(.float32))
                            .max().item(Float.self)
                        let tolerance: Float = dtype == .float32 ? 1e-4 : 0.016
                        XCTAssertTrue(worst.isFinite)
                        XCTAssertLessThanOrEqual(
                            worst, tolerance * max(1, magnitude),
                            "tile=\(tile), rows=\(rows), n=\(n), dtype=\(dtype)")
                    }
                }
            }
        }
    }
}
