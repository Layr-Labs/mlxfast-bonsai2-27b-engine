import MLX
import XCTest

@testable import MLXLMCommon

/// The target-prefix acceptance rule must draw from the same target sampler,
/// including ragged output offsets. This is a synthetic sampler oracle, not
/// a claim about whole-model logits or arbitrary model correctness.
final class CBv2MTPTargetSamplingMatrixTests: XCTestCase {
    func testRaggedVerifyWindowsMatchOrdinarySampling() throws {
        let vocab = 257
        let bases = [0, 1, 37, 4097]
        let ids = (0..<4).map { CBv2RequestID(UInt64(41 + $0 * 17)) }
        for dtype in [DType.float32, .float16, .bfloat16] {
            for seed in [UInt64(0), 7, 1234, UInt64.max - 1] {
                let params: [CBv2SamplingParams] = [
                    .init(temperature: 0, seed: seed),
                    .init(temperature: 0.7, topP: 0.9, topK: 12, minP: 0.05, seed: seed),
                    .init(temperature: 1, topP: 1, topK: 0, seed: nil),
                    .init(temperature: 1.3, topP: 0.6, topK: 31, minP: 0.1, seed: seed),
                ]
                for width in [1, 2, 6] {
                    let values: [Float] = (0..<(4 * width * vocab)).map {
                        Float(($0 * 73 + 19) % 509 - 254) / 47
                    }
                    let logits = MLXArray(values, [4, width, vocab]).asType(dtype)
                    let verifier = CBv2DefaultSampler(fallbackSeed: 991)
                    let verified = try XCTUnwrap(verifier.mtpVerifySample(
                        logits: logits, params: params, requestIDs: ids, stepBases: bases))
                    eval(verified)
                    let ordinary = CBv2DefaultSampler(fallbackSeed: 991)
                    let rows = (0..<4).map { row in
                        CBv2SamplerRow(id: ids[row], params: params[row], promptTokens: [1, 2],
                            outputTokens: Array(repeating: 3, count: bases[row]))
                    }
                    for column in 0..<width {
                        let sampled = ordinary.sample(
                            logits: logits[0..., column, 0...], params: params,
                            requestIDs: ids, stepIndex: column, pendingSampledTokens: nil,
                            rowContext: { rows })
                        eval(sampled)
                        XCTAssertEqual(sampled.asArray(Int32.self),
                            verified[0..., column].asArray(Int32.self),
                            "dtype=\(dtype) seed=\(seed) width=\(width) column=\(column)")
                    }

                    // Batching and order must not change any row's draw.
                    for row in (0..<4).reversed() {
                        let solo = try XCTUnwrap(verifier.mtpVerifySample(
                            logits: logits[row..<(row + 1), 0..., 0...],
                            params: [params[row]], requestIDs: [ids[row]], stepBases: [bases[row]]))
                        eval(solo)
                        XCTAssertEqual(solo.asArray(Int32.self), verified[row].asArray(Int32.self))
                    }
                }
            }
        }
    }
}
