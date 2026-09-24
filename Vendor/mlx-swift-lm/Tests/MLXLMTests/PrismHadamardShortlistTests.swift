// Copyright © 2026 Eigen Labs.
//
// The draft head's shortlist path scores GATHERED rows of the target's output
// projection. `HadamardQuantizedLinear` IS a `QuantizedLinear`, so a gather
// that skips the signed Hadamard rotation still type-checks and still runs —
// it simply returns the wrong numbers. This pins the rotation.
//
// No weights and no checkpoint: the packed module is synthetic.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class PrismHadamardShortlistTests: XCTestCase {

    private func packedHead(outputs: Int, width: Int) throws -> HadamardQuantizedLinear {
        var signs = [Float](repeating: 1, count: width)
        for index in stride(from: 1, to: width, by: 3) { signs[index] = -1 }
        return try HadamardQuantizedLinear(
            weight: MLXArray(
                (0 ..< outputs * width / 16).map { UInt32(truncatingIfNeeded: $0 &* 2_654_435_761) }
            ).reshaped([outputs, width / 16]),
            scales: MLXArray((0 ..< outputs * width / 128).map { Float($0 % 7 + 1) / 64 })
                .reshaped([outputs, width / 128]).asType(.float16),
            biases: MLXArray((0 ..< outputs * width / 128).map { Float($0 % 5) / 128 })
                .reshaped([outputs, width / 128]).asType(.float16),
            groupSize: 128, bits: 2,
            transform: try SignedBlockHadamard(blockSize: width, signs: signs))
    }

    /// The gathered shortlist equals the full projection restricted to the
    /// same rows — but only when the hidden state carries the transform.
    func testGatheredRowsMatchTheFullProjectionOnlyWithTheTransform() throws {
        let width = 512
        let head = try packedHead(outputs: 64, width: width)
        let hidden = MLXArray((0 ..< 2 * width).map { Float(($0 % 11)) / 16 - 0.25 })
            .reshaped([1, 2, width])
        let ids = MLXArray([Int32(3), 17, 40, 61])

        let full = head(hidden)[0..., 0..., ids]

        let gathered = quantizedMM(
            head.transform(hidden), head.weight[ids],
            scales: head.scales[ids], biases: head.biases.map { $0[ids] },
            transpose: true, groupSize: head.groupSize, bits: head.bits, mode: head.mode)
        XCTAssertTrue(allClose(gathered, full, atol: 1e-5).item(Bool.self))

        let unrotated = quantizedMM(
            hidden, head.weight[ids],
            scales: head.scales[ids], biases: head.biases.map { $0[ids] },
            transpose: true, groupSize: head.groupSize, bits: head.bits, mode: head.mode)
        XCTAssertFalse(allClose(unrotated, full, atol: 1e-3).item(Bool.self))
    }

    /// The packed embedding is NOT a `QuantizedEmbedding`, so a tied shortlist
    /// must take its own branch rather than fall through to a float gather of
    /// packed integers.
    func testPackedEmbeddingIsNotAQuantizedEmbedding() throws {
        let width = 512
        let embedding = try HadamardQuantizedEmbedding(
            weight: .zeros([8, width / 16], dtype: .uint32),
            scales: .ones([8, width / 128], dtype: .float16),
            biases: .zeros([8, width / 128], dtype: .float16),
            groupSize: 128, bits: 2,
            transform: try SignedBlockHadamard(
                blockSize: width, signs: Array(repeating: 1, count: width)))
        XCTAssertNil(embedding as Embedding as? QuantizedEmbedding)
        XCTAssertEqual(embedding.shape.1, width)
    }
}
