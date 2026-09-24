// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import MLX
import XCTest

final class AllocationFootprintTests: XCTestCase {
    func testUnscheduledMetadataQueryDoesNotEvaluate() throws {
        let allocationStream = StreamOrDevice.default
        let array = MLXArray.zeros([24_576], dtype: .uint8, stream: allocationStream)
        let before = Memory.snapshot()
        XCTAssertNil(try array.evaluatedBufferInfo())
        let after = Memory.snapshot()
        XCTAssertEqual(after.activeMemory, before.activeMemory)
        XCTAssertEqual(after.cacheMemory, before.cacheMemory)
        try withError { eval(array); allocationStream.stream.synchronize() }
        let info = try XCTUnwrap(array.evaluatedBufferInfo())
        XCTAssertTrue(info.isUnique)
        XCTAssertTrue(info.isRowContiguous)
        XCTAssertEqual(info.dataElements, array.size)
        XCTAssertEqual(info.dataOffset, 0)
        XCTAssertGreaterThanOrEqual(info.allocatedBytes, array.nbytes)
        XCTAssertLessThanOrEqual(info.allocatedBytes,
            try Memory.allocationFootprintUpperBound(byteCount: array.nbytes))
    }

    func testViewsReportSharedFullBackingWithoutBecomingOwners() throws {
        let allocationStream = StreamOrDevice.default
        let array = MLXArray.zeros([24_576], dtype: .uint8, stream: allocationStream)
        try withError { eval(array) }
        let original = try XCTUnwrap(array.evaluatedBufferInfo())
        let view = array[1024 ..< 2048]
        try withError { eval(view); allocationStream.stream.synchronize() }
        let sliced = try XCTUnwrap(view.evaluatedBufferInfo())
        XCTAssertEqual(sliced.allocatedBytes, original.allocatedBytes)
        XCTAssertGreaterThan(sliced.allocatedBytes, view.nbytes)
        XCTAssertEqual(sliced.dataOffset, 1024)
        XCTAssertFalse(sliced.isUnique)
        XCTAssertFalse(try XCTUnwrap(array.evaluatedBufferInfo()).isUnique)
        withExtendedLifetime((array, view)) {}
    }

    func testBoundCoversFreshAndLargerCachedBuffers() throws {
        let previousLimit = Memory.cacheLimit
        Memory.cacheLimit = 2 << 20
        defer { Memory.clearCache(); Memory.cacheLimit = previousLimit }
        for (cachedBytes, requestedBytes) in [(48 << 10, 24_576), (8192, 6000), (64 << 10, 48 << 10)] {
            Memory.clearCache()
            try autoreleasepool {
                let cached = MLXArray.zeros([cachedBytes], dtype: .uint8)
                try withError { eval(cached) }
            }
            let array = MLXArray.zeros([requestedBytes], dtype: .uint8)
            try withError { eval(array) }
            let info = try XCTUnwrap(array.evaluatedBufferInfo())
            XCTAssertGreaterThanOrEqual(info.allocatedBytes, array.nbytes)
            XCTAssertLessThanOrEqual(info.allocatedBytes,
                try Memory.allocationFootprintUpperBound(byteCount: requestedBytes))
        }
    }

    func testPredictionRejectsNegativeAndOverflowWithoutAllocation() throws {
        XCTAssertEqual(try Memory.allocationFootprintUpperBound(byteCount: 0), 0)
        XCTAssertThrowsError(try Memory.allocationFootprintUpperBound(byteCount: -1))
        XCTAssertThrowsError(try Memory.allocationFootprintUpperBound(byteCount: Int.max))
    }

    func testDetachedPolicyMatchesAllocatorAtSizeClassBoundaries() throws {
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        let extra = try XCTUnwrap(policy.maximumExtraBytes)
        for bytes in [0, 1, 2, 4, 7, 8, 9, 4095, 4096, 4097, 8191, 8192, 8193,
                      16_383, 16_384, 16_385, 24_576, 32_767, 32_768, (64 << 20) + 1] {
            let bound = try XCTUnwrap(policy.upperBound(byteCount: bytes))
            XCTAssertEqual(bound, try Memory.allocationFootprintUpperBound(byteCount: bytes))
            XCTAssertLessThanOrEqual(bound - bytes, extra)
        }
    }

    func testDetachedPolicyOverflowDoesNotInvokeMLXErrorHandler() throws {
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        try withError {
            XCTAssertNil(policy.upperBound(byteCount: -1))
            XCTAssertNil(policy.upperBound(byteCount: Int.max))
            XCTAssertEqual(policy.upperBound(byteCount: 0), 0)
        }
    }
}
