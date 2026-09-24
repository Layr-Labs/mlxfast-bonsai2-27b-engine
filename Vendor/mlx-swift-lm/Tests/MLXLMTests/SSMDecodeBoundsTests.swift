import MLX
import XCTest
@testable import MLXLLM

final class SSMDecodeBoundsTests: XCTestCase {
    func testShortAndTailStateWidthsWriteEveryElementExactly() {
        for dtype in [DType.float32, .bfloat16] {
            for (dh, ds) in [(1, 1), (7, 16), (16, 16), (10, 33), (16, 48), (8, 32), (64, 128)] {
                let x = MLXArray.ones([2, 1, 4, dh], dtype: dtype)
                let b = MLXArray.full([2, 1, 2, ds], values: MLXArray(Float(0.5)), dtype: dtype)
                let c = MLXArray.full([2, 1, 2, ds], values: MLXArray(Float(0.25)), dtype: dtype)
                let old = MLXArray.zeros([2, 4, dh, ds], dtype: .float32)
                let (output, state) = ssmUpdate(hiddenStates: x,
                    ALog: MLXArray.zeros([4], dtype: .float32), B: b, C: c,
                    D: MLXArray.full([4], values: MLXArray(Float(0.75)), dtype: .float32),
                    dt: MLXArray.zeros([2, 1, 4], dtype: dtype),
                    dtBias: MLXArray.zeros([4], dtype: .float32), state: old,
                    timeStepLimit: (1, 1))
                eval(output, state)
                // Exact dyadic oracle: dt=1, prior=0 => each state=0.5;
                // sum(state*C)+X*D = Ds/8+3/4. No tolerance is involved.
                XCTAssertEqual(MLX.abs(state - Float(0.5)).max().item(Float.self), 0,
                               "unwritten or wrong state for Dh=\(dh) Ds=\(ds) dtype=\(dtype)")
                XCTAssertEqual(MLX.abs(output - (Float(ds) / 8 + 0.75)).max().item(Float.self), 0,
                               "wrong output for Dh=\(dh) Ds=\(ds) dtype=\(dtype)")
                XCTAssertEqual(state.dtype, DType.float32)
                XCTAssertEqual(output.dtype, dtype)
            }
        }
    }

    func testEstablishedAlignedWidthsRemainBitIdenticalToPriorKernel() {
        let historical = MLXFast.metalKernel(name: "ssm_historical_aligned_control",
            inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
            outputNames: ["out", "state_out"], source: Self.historicalSource)
        for dtype in [DType.float32, .bfloat16] {
            for (dh, ds) in [(8, 32), (64, 128)] {
                func values(_ shape: [Int], dtype: DType) -> MLXArray {
                    let count = shape.reduce(1, *)
                    return (MLXArray(0..<count).asType(.float32) / Float(count + 1) - 0.25)
                        .reshaped(shape).asType(dtype)
                }
                let x = values([2, 1, 4, dh], dtype: dtype)
                let b = values([2, 1, 2, ds], dtype: dtype)
                let c = values([2, 1, 2, ds], dtype: dtype)
                let old = values([2, 4, dh, ds], dtype: .float32)
                let a = values([4], dtype: .float32), d = values([4], dtype: .float32)
                let dt = values([2, 1, 4], dtype: dtype), bias = values([4], dtype: .float32)
                let actual = ssmUpdateKernel(hiddenStates: x, ALog: a, B: b, C: c, D: d,
                    dt: dt, dtBias: bias, state: old, timeStepLimit: (0, 4))
                let expected = historical([x, a, b, c, d, computeDt(dt, bias, (0, 4)), old],
                    template: [("T", dtype), ("U", DType.float32), ("Dh", dh), ("Ds", ds), ("H", 4), ("G", 2)],
                    grid: (32, dh, 8), threadGroup: (32, 8, 1),
                    outputShapes: [[2, 1, 4, dh], old.shape], outputDTypes: [dtype, .float32])
                eval([actual.0, actual.1] + expected)
                XCTAssertTrue(actual.0.asData(access: .copy).data == expected[0].asData(access: .copy).data,
                              "aligned output changed for \(dtype), Dh=\(dh), Ds=\(ds)")
                XCTAssertTrue(actual.1.asData(access: .copy).data == expected[1].asData(access: .copy).data,
                              "aligned recurrent state changed for \(dtype), Dh=\(dh), Ds=\(ds)")
            }
        }
    }

    // Frozen pre-tail SSM.swift kernel (John Mai / MLX-LM port), used only
    // for its established divisible-by-32 state/multiple-of-8 head geometry.
    private static let historicalSource = """
        auto n = thread_position_in_grid.z;
        auto h_idx = n % H;
        auto g_idx = n / G;
        constexpr int n_per_t = Ds / 32;
        auto x = X + n * Dh;
        out += n * Dh;
        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;
        auto C_ = C + g_idx * Ds;
        auto B_ = B + g_idx * Ds;
        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;
        auto dt_ = static_cast<float>(dt[n]);
        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        auto dA = fast::exp(A * dt_);
        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<U>(state);
            acc += state * C_[s_idx];
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
        }
        """
}
