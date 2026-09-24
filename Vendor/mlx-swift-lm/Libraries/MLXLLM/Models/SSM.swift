//
//  SSM.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/10/01.
//

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/ssm.py

import Foundation
import MLX
import MLXNN

public func computeDt(_ dt: MLXArray, _ dtBias: MLXArray, _ timeStepLimit: (Float, Float))
    -> MLXArray
{
    // mlx-lm performs the timestep transform in fp32 even when the model
    // activations and checkpoint tensors are bf16.
    let dt = softplus(dt.asType(.float32) + dtBias)
    return MLX.clip(dt, min: timeStepLimit.0, max: timeStepLimit.1)
}

private func makeSSMKernel() -> MLXFast.MLXFastKernel? {
    let source = """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            // Ceiling division covers short/tail state widths. For the
            // production widths divisible by 32 this is the original loop.
            constexpr int n_per_t = (Ds + 31) / 32;

            auto x = X + n * Dh;
            out += n * Dh;
            auto i_state = state_in + n * Dh * Ds;
            auto o_state = state_out + n * Dh * Ds;

            // C and B have shape [batch, group, state_dim]
            // C and B need to be offset by group size
            auto C_ = C + g_idx * Ds;
            auto B_ = B + g_idx * Ds;

            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;

            // A partial final y threadgroup must not read/write past Dh.
            // Compile the guard away for the established multiple-of-8 path.
            if constexpr (Dh % 8 != 0) {
                if (d_idx >= Dh) return;
            }

            auto dt_ = static_cast<float>(dt[n]);
            auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
            auto dA = fast::exp(A * dt_);

            float acc = 0.0;
            auto x_ = static_cast<float>(x[d_idx]);

            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                if constexpr (Ds % 32 != 0) {
                    if (s_idx >= Ds) continue;
                }
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

    return MLXFast.metalKernel(
        name: "ssm_kernel",
        inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
        outputNames: ["out", "state_out"],
        source: source
    )
}

private final class SSMKernelManager: Sendable {
    static let shared = SSMKernelManager()

    let ssmKernel: MLXFast.MLXFastKernel?

    private init() {
        ssmKernel = makeSSMKernel()
    }
}

func ssmUpdateKernel(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray,
    timeStepLimit: (Float, Float)
) -> (MLXArray, MLXArray) {
    let (n, _, h, d) = hiddenStates.shape4
    let inputType = hiddenStates.dtype
    let stateType = state.dtype
    let (hb, ds) = (B.dim(-2), B.dim(-1))

    let dt = computeDt(dt, dtBias, timeStepLimit)

    guard let kernel = SSMKernelManager.shared.ssmKernel else {
        fatalError("SSM kernel not available")
    }

    let outputs = kernel(
        [hiddenStates, ALog, B, C, D, dt, state],
        template: [
            ("T", inputType),
            ("U", stateType),
            ("Dh", d),
            ("Ds", ds),
            ("H", h),
            ("G", h / hb),
        ],
        grid: (32, d, h * n),
        threadGroup: (32, 8, 1),
        outputShapes: [[n, 1, h, d], state.shape],
        outputDTypes: [inputType, stateType]
    )

    return (outputs[0], outputs[1])
}

public func segsum(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let l = x.dim(-1)
    var x = x

    if let mask = mask {
        let mask = MLX.expandedDimensions(mask, axis: 1)
        x = x * mask
    }

    x = MLX.repeated(x[.ellipsis, .newAxis], count: l, axis: -1)
    x = MLX.tril(x, k: -1)
    var xSegsum = MLX.cumsum(x, axis: -2)

    if let mask = mask {
        xSegsum = which(
            mask[.ellipsis, .newAxis, 0...] * mask[.ellipsis, .newAxis],
            xSegsum,
            MLXArray(Float(-Float.infinity), dtype: xSegsum.dtype)
        )
    }

    return xSegsum
}

public func ssmAttn(
    x: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    timeStepLimit: (Float, Float) = (0.001, 100.0),
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let (b, l, h, dh) = x.shape4
    let (_, _, g, d) = B.shape4

    let dt = computeDt(dt, dtBias, timeStepLimit)
    let repeats = h / g
    let A = -MLX.exp(ALog).asType(dt.dtype)
    let dtA = dt * A.reshaped(1, 1, -1)
    let dtx = dt.reshaped(b, l, h, 1) * x

    // mlx-lm bounds the quadratic surrogate-attention scan to 256-token
    // windows and threads the fp32 recurrent state between them. Besides
    // bounding prefill memory, the window boundary is numerically observable
    // on long prompts and therefore part of the serial oracle.
    let step = 256
    var currentState = state
    var outputs = [MLXArray]()
    outputs.reserveCapacity((l + step - 1) / step)
    for start in stride(from: 0, to: l, by: step) {
        let end = min(start + step, l)
        let stepDtx = dtx[0..., start ..< end, 0..., 0...]
        let stepDtA = dtA[0..., start ..< end, 0...]
        var stepB = B[0..., start ..< end, 0..., 0...]
        let stepC = C[0..., start ..< end, 0..., 0...]
        let stepMask = mask.map { $0[.ellipsis, start ..< end] }

        stepB = MLX.transposed(stepB, axes: [0, 2, 3, 1])
        var CB = MLX.swappedAxes(stepC, 1, 2).matmul(stepB)
        CB = MLX.repeated(CB, count: repeats, axis: 1)

        var decay = MLX.exp(
            segsum(stepDtA.swappedAxes(1, 2), mask: stepMask))
        let surrogateAttentionMatrix = MLX.tril(CB * decay, k: 0)
        var y = surrogateAttentionMatrix.matmul(stepDtx.swappedAxes(1, 2))
        y = MLX.swappedAxes(y, 1, 2)

        decay = decay[0..., 0..., (-1)..., 0...].transposed(0, 3, 1, 2)
        stepB = MLX.repeated(stepB, count: repeats, axis: 1).swappedAxes(2, 3)
        var dtxdecay = stepDtx * decay
        dtxdecay = dtxdecay.swappedAxes(1, 2).swappedAxes(2, 3)
        var nextState = dtxdecay.matmul(stepB)

        if let previousState = currentState {
            let expDtACumsum = MLX.exp(MLX.cumsum(stepDtA, axis: -2))
            nextState =
                nextState
                + expDtACumsum[0..., -1, 0..., .newAxis, .newAxis] * previousState
            let reshapedState = previousState.reshaped(
                b, 1, g, repeats, dh, d)
            let reshapedC = stepC.reshaped(
                b, end - start, g, 1, d, 1)
            let yPrev = (reshapedState.matmul(reshapedC))
                .squeezed(axis: -1)
                .flattened(start: 2, end: 3)
            y = y + expDtACumsum[.ellipsis, .newAxis] * yPrev
        }

        outputs.append(y.asType(x.dtype))
        currentState = nextState
    }

    let y = MLX.concatenated(outputs, axis: 1) + x * D.reshaped(1, 1, h, 1)
    return (y, currentState!)
}

public func ssmUpdate(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    timeStepLimit: (Float, Float) = (0.001, 100.0),
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let seqLen = hiddenStates.dim(1)

    if seqLen == 1,
        let state = state,
        SSMKernelManager.shared.ssmKernel != nil
    {
        return ssmUpdateKernel(
            hiddenStates: hiddenStates,
            ALog: ALog,
            B: B,
            C: C,
            D: D,
            dt: dt,
            dtBias: dtBias,
            state: state,
            timeStepLimit: timeStepLimit
        )
    } else {
        return ssmAttn(
            x: hiddenStates,
            ALog: ALog,
            B: B,
            C: C,
            D: D,
            dt: dt,
            dtBias: dtBias,
            state: state,
            timeStepLimit: timeStepLimit,
            mask: mask
        )
    }
}
