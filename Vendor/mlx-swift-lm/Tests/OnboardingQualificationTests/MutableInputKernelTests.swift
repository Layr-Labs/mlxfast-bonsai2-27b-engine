import MLX
import XCTest

final class MutableInputKernelTests: XCTestCase {
    private func writer() -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "qualification_owned_buffer_write",
            inputNames: ["owned", "source", "previous"], outputNames: ["fence"],
            source: """
                const uint i = thread_position_in_grid.x;
                owned[i] = source[i];
                if (i == 0) fence[0] = previous[0] + 1;
                """, mutableInputs: ["owned"])
    }

    func testOwnedWritesIncludeSmallDeviceBuffersAndRetainValues() throws {
        let kernel = writer()
        for size in [1, 4, 128, 4096] {
            let owned = MLXArray.zeros([size], dtype: .int32)
            let source = MLXArray((0..<size).map { Int32($0 + 7) })
            let fence = kernel([owned, source, MLXArray([Int32(0)])],
                grid: (size, 1, 1), threadGroup: (min(size, 256), 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
            let ready = depends(input: owned, dependencies: [fence])
            XCTAssertEqual(ready.asArray(Int32.self), source.asArray(Int32.self))
            XCTAssertEqual(fence.item(Int32.self), 1)
        }
    }

    func testInvalidOrDuplicateMutableInputNamesFailBeforeExecution() throws {
        for names in [["missing"], ["owned", "owned"]] {
            XCTAssertThrowsError(try MLX.withError {
                _ = MLXFast.metalKernel(name: "qualification_bad_mutable_name",
                    inputNames: ["owned"], outputNames: ["out"],
                    source: "out[0] = owned[0];", mutableInputs: names)
            })
        }
    }

    func testMutableViewsCannotSilentlyWriteAnImplicitContiguousCopy() throws {
        let source = MLXArray((0..<8).map(Int32.init)).reshaped(2, 4).transposed()
        eval(source)
        let kernel = MLXFast.metalKernel(name: "qualification_bad_mutable_layout",
            inputNames: ["owned"], outputNames: ["fence"],
            source: "owned[0] = 99; fence[0] = 1;", mutableInputs: ["owned"])
        XCTAssertThrowsError(try MLX.withError {
            let output = kernel([source], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                                outputShapes: [[1]], outputDTypes: [.int32])
            eval(output)
        })
        XCTAssertEqual(source.asArray(Int32.self), [0, 4, 1, 5, 2, 6, 3, 7])
    }

    func testExistingReadOnlyKernelStillUsesItsOrdinaryContract() {
        let kernel = MLXFast.metalKernel(name: "qualification_readonly_control",
            inputNames: ["source"], outputNames: ["out"],
            source: "out[thread_position_in_grid.x] = source[thread_position_in_grid.x];")
        let input = MLXArray([Int32(4), 7, 11, 13])
        let output = kernel([input], grid: (4, 1, 1), threadGroup: (4, 1, 1),
                            outputShapes: [[4]], outputDTypes: [.int32])[0]
        XCTAssertEqual(output.asArray(Int32.self), input.asArray(Int32.self))
    }

    func testExplicitStrideAwareMutationKeepsTheOriginalOwner() {
        let parent = MLXArray((0..<8).map(Int32.init)).reshaped(2, 4)
        let view = parent.transposed()
        let kernel = MLXFast.metalKernel(name: "qualification_mutable_strides",
            inputNames: ["owned"], outputNames: ["fence"], source: """
                const uint row = thread_position_in_grid.y;
                const uint col = thread_position_in_grid.x;
                owned[row * owned_strides[0] + col * owned_strides[1]] = int(99 + row * 2 + col);
                if (row == 0 && col == 0) fence[0] = 1;
                """, ensureRowContiguous: false, mutableInputs: ["owned"])
        let fence = kernel([view], grid: (2, 4, 1), threadGroup: (2, 4, 1),
                            outputShapes: [[1]], outputDTypes: [.int32])[0]
        let ready = depends(input: parent, dependencies: [fence])
        XCTAssertEqual(ready.asArray(Int32.self), [99, 101, 103, 105, 100, 102, 104, 106])
    }
}
