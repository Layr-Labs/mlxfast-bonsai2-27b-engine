// Copyright © 2026 Apple Inc.

import MLX
import MLXFast
import XCTest

@testable import MLXVLM

/// The vision tower's attention is block-diagonal over the images in
/// `cuSeqlens`. The per-image fused-SDPA path must be numerically
/// interchangeable with the dense joint-mask formulation it replaced, and the
/// pad-to-80 trick must be exact math, not an approximation.
final class Qwen3VLVisionAttentionTests: XCTestCase {

    /// Pad-to-80 exactness: head dim 72 (the production ViT's 1152/16) through
    /// `ensureFusedSDPA` must match a composed-op reference at the same scale.
    func testEnsureFusedSDPAMatchesComposedReference() {
        let (H, L, D) = (4, 33, 72)
        let scale = Float(pow(Double(D), -0.5))
        let queries = MLXRandom.normal([1, H, L, D], dtype: .float32)
        let keys = MLXRandom.normal([1, H, L, D], dtype: .float32)
        let values = MLXRandom.normal([1, H, L, D], dtype: .float32)

        let actual = Qwen3VLVision.ensureFusedSDPA(
            queries: queries, keys: keys, values: values, scale: scale)
        let scores = matmul(queries * scale, keys.transposed(0, 1, 3, 2))
        let expected = matmul(softmax(scores, axis: -1), values)

        eval(actual, expected)
        XCTAssertEqual(actual.shape, [1, H, L, D])
        XCTAssertTrue(allClose(actual, expected, atol: 1e-4).item(Bool.self))
    }

    /// Block-diagonal parity: attending each `cuSeqlens` segment independently
    /// must match the dense [L, L] additive-mask formulation over the joint
    /// sequence.
    func testPerImageSegmentsMatchDenseJointMask() {
        let (H, D) = (4, 72)
        let segments = [0, 5, 12, 20]
        let L = segments.last!
        let scale = Float(pow(Double(D), -0.5))
        let queries = MLXRandom.normal([1, H, L, D], dtype: .float32)
        let keys = MLXRandom.normal([1, H, L, D], dtype: .float32)
        let values = MLXRandom.normal([1, H, L, D], dtype: .float32)

        var parts: [MLXArray] = []
        for idx in 1 ..< segments.count {
            let (start, end) = (segments[idx - 1], segments[idx])
            parts.append(
                Qwen3VLVision.ensureFusedSDPA(
                    queries: queries[0..., 0..., start ..< end, 0...],
                    keys: keys[0..., 0..., start ..< end, 0...],
                    values: values[0..., 0..., start ..< end, 0...],
                    scale: scale))
        }
        let actual = concatenated(parts, axis: 2)

        var mask = ones([1, L, L], dtype: queries.dtype) * MLXArray(-1e9, dtype: queries.dtype)
        for idx in 1 ..< segments.count {
            let (start, end) = (segments[idx - 1], segments[idx])
            mask[0..., start ..< end, start ..< end] = MLXArray(0, dtype: queries.dtype)
        }
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .array(mask))

        eval(actual, expected)
        XCTAssertTrue(allClose(actual, expected, atol: 1e-4).item(Bool.self))
    }

    /// The default image budget resolves to 1,280 vision tokens' worth of
    /// pixels under the production factor, clamps to the config ceiling, and
    /// degrades to the ceiling on overflow instead of trapping.
    func testDefaultVisionTokenBudgetPixels() {
        // qwen3_5 ViT: patch 16, merge 2 → factor 32 → 1280 * 32² pixels.
        XCTAssertEqual(
            Qwen3VLProcessor.defaultVisionTokenBudgetPixels(factor: 32, ceiling: 12_845_056),
            1280 * 32 * 32)
        // A config ceiling below the budget wins.
        XCTAssertEqual(
            Qwen3VLProcessor.defaultVisionTokenBudgetPixels(factor: 32, ceiling: 100_000),
            100_000)
        // Pathological factor overflows to the ceiling, not a trap.
        XCTAssertEqual(
            Qwen3VLProcessor.defaultVisionTokenBudgetPixels(factor: Int.max, ceiling: 42), 42)
    }
}
