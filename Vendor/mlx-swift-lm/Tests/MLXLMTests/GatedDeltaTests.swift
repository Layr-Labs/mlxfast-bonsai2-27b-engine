// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

public class GatedDeltaTests: XCTestCase {

    private struct Inputs {
        let q, k, v, a, b, aLog, dtBias: MLXArray
    }

    /// Build deterministic bf16 inputs shaped for the GDN entry points.
    /// Hk/Hv/Dk/Dv stay tiny so the kernel dispatches but the test runs in ms.
    private func makeInputs(
        B: Int = 1, T: Int = 16, Hk: Int = 2, Dk: Int = 32,
        Hv: Int = 4, Dv: Int = 16, seed: UInt64 = 42
    ) -> Inputs {
        MLXRandom.seed(seed)
        let dtype = DType.bfloat16
        let q = MLXRandom.normal([B, T, Hk, Dk]).asType(dtype)
        let k = MLXRandom.normal([B, T, Hk, Dk]).asType(dtype)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(dtype)
        let a = MLXRandom.normal([B, T, Hv]).asType(dtype)
        let b = MLXRandom.normal([B, T, Hv]).asType(dtype)
        let aLog = (MLXRandom.normal([Hv]) * MLXArray(0.1)).asType(dtype)
        let dtBias = MLXRandom.normal([Hv]).asType(dtype)
        return Inputs(q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias)
    }

    /// Multi-chunk prefill must match single-chunk prefill at the same total T.
    ///
    /// Regression for the fp32-state fix. Pre-fix, `gatedDeltaKernel` wrote
    /// `state_out` as `InT` (bf16) and `gatedDeltaUpdate` defaulted state to
    /// `q.dtype` (bf16). When a second-chunk prefill reloaded that state, it
    /// arrived bf16-quantized; the fp32 scratch recurrence then ran from a
    /// degraded starting point. With this test's inputs the cross-chunk drift
    /// vs a single full-length prefill is >10 max abs. Post-fix, state crosses
    /// the chunk boundary as fp32 and the two paths match within bf16 input
    /// quantization noise.
    func testGatedDeltaMultiChunkMatchesSingleChunk() throws {
        let T = 16
        let inputs = makeInputs(T: T)

        let (ySingle, _) = gatedDeltaUpdate(
            q: inputs.q, k: inputs.k, v: inputs.v,
            a: inputs.a, b: inputs.b,
            aLog: inputs.aLog, dtBias: inputs.dtBias
        )
        eval(ySingle)

        let mid = T / 2
        let (y1, state1) = gatedDeltaUpdate(
            q: inputs.q[0..., ..<mid], k: inputs.k[0..., ..<mid], v: inputs.v[0..., ..<mid],
            a: inputs.a[0..., ..<mid], b: inputs.b[0..., ..<mid],
            aLog: inputs.aLog, dtBias: inputs.dtBias
        )
        let (y2, _) = gatedDeltaUpdate(
            q: inputs.q[0..., mid...], k: inputs.k[0..., mid...], v: inputs.v[0..., mid...],
            a: inputs.a[0..., mid...], b: inputs.b[0..., mid...],
            aLog: inputs.aLog, dtBias: inputs.dtBias,
            state: state1
        )
        let yMulti = concatenated([y1, y2], axis: 1)
        eval(yMulti)

        let diff = abs(ySingle.asType(.float32) - yMulti.asType(.float32)).max()
        eval(diff)
        let maxDiff = diff.item(Float.self)

        // Pre-fix: bf16-cast at chunk boundary diverges by >10 max abs.
        // Post-fix: fp32 state across the boundary leaves only bf16 input noise.
        XCTAssertLessThan(
            maxDiff, 1e-2,
            "Multi-chunk GDN prefill diverged from single-chunk by \(maxDiff) max abs. "
                + "Cross-chunk state must persist in fp32; bf16 cast loses precision."
        )
    }

    /// The recurrent state update must retain low-order terms in its k-state dot product.
    ///
    /// Each SIMD lane accumulates one large value followed by five unit values. A naive
    /// FP32 sum drops all five units, shifting the recurrent state by one ULP. Compensated
    /// summation recovers them and matches the correctly rounded reference result.
    /// Upstream d5d8b29.
    func testGatedDeltaCompensatesRecurrentDotProduct() throws {
        let keyDimension = 192
        let laneState: [Float] = [100_000_000, 1, 1, 1, 1, 1]
        let stateValues = (0 ..< 32).flatMap { _ in laneState }

        let q = MLXArray.zeros([1, 1, 1, keyDimension], dtype: .bfloat16)
        let k = MLXArray.ones([1, 1, 1, keyDimension], dtype: .bfloat16)
        let v = MLXArray.zeros([1, 1, 1, 1], dtype: .bfloat16)
        let a = MLXArray.zeros([1, 1, 1], dtype: .bfloat16)
        let b = MLXArray.zeros([1, 1, 1], dtype: .bfloat16)
        let aLog = MLXArray([-100] as [Float])
        let dtBias = MLXArray.zeros([1], dtype: .bfloat16)
        let state = MLXArray(stateValues).reshaped(1, 1, 1, keyDimension)

        let (_, nextState) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b,
            aLog: aLog, dtBias: dtBias, state: state)

        let actual = nextState[0, 0, 0, 0].item(Float.self)
        let exactDotProduct = 32.0 * (100_000_000.0 + 5.0)
        let expected = Float(100_000_000.0 - 0.5 * exactDotProduct)

        XCTAssertEqual(
            actual.bitPattern, expected.bitPattern,
            "GDN recurrent dot product rounded to \(actual), expected \(expected). "
                + "Naive FP32 accumulation loses the unit terms and returns -1.5e9."
        )
    }

}
