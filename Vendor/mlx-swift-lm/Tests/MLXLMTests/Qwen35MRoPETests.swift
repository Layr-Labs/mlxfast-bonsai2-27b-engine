// Copyright © 2026 Apple Inc.

import MLX
import MLXFast
import MLXLMCommon
import XCTest

@testable import MLXLLM
@testable import MLXVLM

/// The interleaved M-RoPE rewrite replaces a per-frequency slice loop
/// (~dims lazy ops per attention layer per forward) with one precomputed
/// takeAlong. These tests pin the new path to a slice-loop reference that
/// mirrors the replaced implementation exactly.
final class Qwen35MRoPETests: XCTestCase {
    func testDefaultSectionsMatchSliceReference() {
        assertMatchesReference(dim: 64, sections: [11, 11, 10])
    }

    func testTruncatedSectionsMatchSliceReference() {
        assertMatchesReference(dim: 16, sections: [2, 1, 1])
    }

    func testMissingSectionsUseDefaults() {
        assertMatchesReference(dim: 64, sections: [])
    }

    private func assertMatchesReference(dim: Int, sections: [Int]) {
        let embedding = Qwen35Language.RotaryEmbedding(
            dim: dim, base: 100_000, mropeSection: sections)
        let positionIds = MLXArray((0 ..< 24).map(Int32.init)).reshaped(3, 2, 4)
        let x = MLXArray.zeros([1], dtype: .float32)

        let actual = embedding(x: x, positionIds: positionIds)
        let expected = referenceInterleavedMRope(
            dim: dim, base: 100_000, sections: sections,
            fallbackSections: [11, 11, 10], positionIds: positionIds)

        eval(actual.0, actual.1, expected.0, expected.1)
        XCTAssertTrue(allClose(actual.0, expected.0, atol: 1e-6).item(Bool.self))
        XCTAssertTrue(allClose(actual.1, expected.1, atol: 1e-6).item(Bool.self))
    }
}

final class Qwen3VLMRoPETests: XCTestCase {
    func testDefaultSectionsMatchSliceReference() {
        assertMatchesReference(headDim: 128, sections: nil)
    }

    func testCustomSectionsMatchSliceReference() {
        assertMatchesReference(headDim: 16, sections: [2, 1, 1])
    }

    private func assertMatchesReference(headDim: Int, sections: [Int]?) {
        let scaling = sections.map {
            Qwen3VLConfiguration.RoPEScaling(mropeSection: $0)
        }
        let embedding = Qwen3VLLanguage.RotaryEmbedding(
            headDim: headDim, base: 100_000, ropeScaling: scaling)
        let positionIds = MLXArray((0 ..< 24).map(Int32.init)).reshaped(3, 2, 4)

        let actual = embedding(positionIds: positionIds, dtype: .float32)
        let expected = referenceInterleavedMRope(
            dim: headDim, base: 100_000, sections: sections ?? [24, 20, 20],
            fallbackSections: [24, 20, 20], positionIds: positionIds)

        eval(actual.0, actual.1, expected.0, expected.1)
        XCTAssertTrue(allClose(actual.0, expected.0, atol: 1e-6).item(Bool.self))
        XCTAssertTrue(allClose(actual.1, expected.1, atol: 1e-6).item(Bool.self))
    }
}

/// Full-output parity for the MLXLLM `Qwen35MRoPE` default path (the one the
/// production text-extraction tower drives): the takeAlong frequency selection
/// must leave `apply(queries:keys:positionIds:)` bit-for-bit interchangeable
/// with the replaced per-frequency slice loop.
final class Qwen35LLMMRoPEParityTests: XCTestCase {
    func testDefaultPathMatchesSliceReference() {
        let rotaryDim = 64
        let base: Float = 5_000_000
        let sections = [11, 11, 10]
        let (B, H, kvH, L) = (2, 4, 2, 5)
        let headDim = rotaryDim  // full-rotary, matches production partialRotaryFactor 1.0

        let rope = initializeRope(
            dims: rotaryDim, base: base, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: 4096)
        let mrope = Qwen35MRoPE(
            rope: rope, dim: rotaryDim, base: base,
            scalingConfig: nil, sections: sections)

        let queries = MLXRandom.normal([B, H, L, headDim], dtype: .float32)
        let keys = MLXRandom.normal([B, kvH, L, headDim], dtype: .float32)
        let positionIds = MLXArray((0 ..< (3 * B * L)).map { Int32($0 % 97) })
            .reshaped(3, B, L)

        let (actualQ, actualK) = mrope.apply(
            queries: queries, keys: keys, positionIds: positionIds)

        // Reference: the replaced implementation, per-frequency slice selection
        // included, applied to the same inputs.
        let frequencyCount = rotaryDim / 2
        var frequency = MLXArray(stride(from: 0, to: rotaryDim, by: 2)).asType(.float32)
        frequency = frequency / Float(rotaryDim)
        let invFreq = 1 / pow(MLXArray(base), frequency)
        let all = positionIds.asType(.float32)[0..., 0..., 0..., .newAxis]
            * invFreq[.newAxis, .newAxis, .newAxis, 0...]
        func referenceAxis(_ index: Int) -> Int {
            for (axis, offset) in [(1, 1), (2, 2)] {
                let length = min(sections[axis] * 3, frequencyCount)
                if index >= offset && index < length && (index - offset) % 3 == 0 {
                    return axis
                }
            }
            return 0
        }
        var selected: [MLXArray] = []
        for index in 0 ..< frequencyCount {
            selected.append(all[referenceAxis(index), 0..., 0..., index])
        }
        let referenceFrequency = stacked(selected, axis: -1)
        let angles = concatenated([referenceFrequency, referenceFrequency], axis: -1)
        let cosine = cos(angles).expandedDimensions(axis: 1)
        let sine = sin(angles).expandedDimensions(axis: 1)
        func applyReference(_ value: MLXArray) -> MLXArray {
            let half = rotaryDim / 2
            let rotatedHalf = concatenated(
                [-value[.ellipsis, half...], value[.ellipsis, ..<half]], axis: -1)
            return value * cosine + rotatedHalf * sine
        }
        let expectedQ = applyReference(queries)
        let expectedK = applyReference(keys)

        eval(actualQ, actualK, expectedQ, expectedK)
        XCTAssertTrue(allClose(actualQ, expectedQ, atol: 1e-5).item(Bool.self))
        XCTAssertTrue(allClose(actualK, expectedK, atol: 1e-5).item(Bool.self))
    }
}

private func referenceInterleavedMRope(
    dim: Int, base: Float, sections: [Int], fallbackSections: [Int], positionIds: MLXArray
) -> (MLXArray, MLXArray) {
    let sections = sections.count >= 3 ? sections : fallbackSections
    let safeDim = max(1, dim)
    var frequency = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
    frequency = frequency / Float(safeDim)
    var inverseFrequency = 1.0 / pow(MLXArray(base), frequency)
    inverseFrequency = inverseFrequency[.newAxis, .newAxis, .newAxis, 0...]

    let positions = positionIds.asType(.float32)
    let frequencies = positions[0..., 0..., 0..., .newAxis] * inverseFrequency
    let temporal = frequencies[0, 0..., 0..., 0...]
    var slices: [MLXArray] = []
    slices.reserveCapacity(temporal.dim(-1))

    for index in 0 ..< temporal.dim(-1) {
        var slice = temporal[0..., 0..., index]
        for (dimension, offset) in [(1, 1), (2, 2)] {
            let end = min(sections[dimension] * 3, temporal.dim(-1))
            if index >= offset && index < end && (index - offset) % 3 == 0 {
                slice = frequencies[dimension, 0..., 0..., index]
                break
            }
        }
        slices.append(slice)
    }

    let interleaved = stacked(slices, axis: -1)
    let embedding = concatenated([interleaved, interleaved], axis: -1)
    return (cos(embedding), sin(embedding))
}
