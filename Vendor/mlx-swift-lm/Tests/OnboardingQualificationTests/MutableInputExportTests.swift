import Foundation
import MLX
import XCTest

final class MutableInputExportTests: XCTestCase {
    func testReadonlyCustomKernelExportRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("readonly-kernel-\(UUID().uuidString).mlxfn")
        defer { try? FileManager.default.removeItem(at: url) }
        let kernel = MLXFast.metalKernel(
            name: "qualification_readonly_export",
            inputNames: ["source"], outputNames: ["output"],
            source: "output[thread_position_in_grid.x] = source[thread_position_in_grid.x] + 7;")
        let function: ([MLXArray]) -> [MLXArray] = { inputs in
            kernel(inputs, grid: (4, 1, 1), threadGroup: (4, 1, 1),
                   outputShapes: [[4]], outputDTypes: [.int32])
        }
        try exportFunction(to: url, function)(MLXArray([Int32(1), 2, 3, 4]))
        let imported = try importFunction(from: url)
        let result = try imported(MLXArray([Int32(10), 20, 30, 40]))
        XCTAssertEqual(result[0].asArray(Int32.self), [17, 27, 37, 47])
    }

    func testMutableCustomKernelExportRetainsWriteDependencies() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mutable-kernel-\(UUID().uuidString).mlxfn")
        defer { try? FileManager.default.removeItem(at: url) }
        let kernel = MLXFast.metalKernel(
            name: "qualification_mutable_export",
            inputNames: ["owned", "source", "previous"], outputNames: ["fence"],
            source: """
                const uint i = thread_position_in_grid.x;
                owned[i] = source[i];
                if (i == 0) fence[0] = previous[0] + 1;
                """, mutableInputs: ["owned"])
        let function: ([MLXArray]) -> [MLXArray] = { inputs in
            let fence = kernel(inputs, grid: (4, 1, 1), threadGroup: (4, 1, 1),
                               outputShapes: [[1]], outputDTypes: [.int32])[0]
            return [depends(input: inputs[0], dependencies: [fence]), fence]
        }
        try exportFunction(to: url, function)(
            MLXArray.zeros([4], dtype: .int32), MLXArray([Int32(1), 2, 3, 4]), MLXArray([Int32(0)]))
        let imported = try importFunction(from: url)
        for start: Int32 in [10, 30, 50] {
            let owned = MLXArray.zeros([4], dtype: .int32)
            let expected = [start, start + 1, start + 2, start + 3]
            let result = try imported(owned, MLXArray(expected), MLXArray([Int32(5)]))
            XCTAssertEqual(result[0].asArray(Int32.self), expected)
            XCTAssertEqual(result[1].item(Int32.self), 6)
            XCTAssertEqual(owned.asArray(Int32.self), expected)
        }
    }
}
