import Cmlx
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Exact M1 arithmetic over a short verify window without rereading each W4
/// matrix once per position. One SIMD lane owns the same quantized values and
/// reduction order as the stock one-row QMV; the verify positions share the
/// weight load but retain independent accumulators.
private let qwen35A3BExactW4G64VerifyKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_exact_w4_g64_verify_narrow2",
    inputNames: ["x", "w", "scales", "biases"],
    outputNames: ["y"],
    source: """
        uint n_tile = threadgroup_position_in_grid.y;
        uint batch = threadgroup_position_in_grid.z;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        constexpr int PACK_FACTOR = 8;
        constexpr int VALUES_PER_THREAD = 16;
        constexpr int BLOCK_SIZE = 512;
        constexpr int RESULTS_PER_SIMDGROUP = 2;
        constexpr int OUTPUTS_PER_THREADGROUP = 4;

        int output_row = int(n_tile) * OUTPUTS_PER_THREADGROUP
            + int(simd_group) * RESULTS_PER_SIMDGROUP;
        int weight_row_bytes = K_SIZE / 2;
        int groups_per_row = K_SIZE / 64;

        const device uint8_t* weight_base =
            (const device uint8_t*)w + output_row * weight_row_bytes
            + int(lane) * 8;
        const device T* scale_base =
            scales + output_row * groups_per_row + int(lane) / 4;
        const device T* bias_base =
            biases + output_row * groups_per_row + int(lane) / 4;
        const device T* input_base =
            x + int(batch) * VERIFY_T * K_SIZE + int(lane) * VALUES_PER_THREAD;

        float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
        float input_values[VERIFY_T][VALUES_PER_THREAD];
        for (int t = 0; t < VERIFY_T; ++t) {
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                result[t][row] = 0.0f;
            }
        }

        const device uint8_t* weight_block = weight_base;
        const device T* scale_block = scale_base;
        const device T* bias_block = bias_base;
        const device T* input_block = input_base;

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
            float sums[VERIFY_T];
            for (int t = 0; t < VERIFY_T; ++t) {
                const device T* input = input_block + t * K_SIZE;
                float sum = 0.0f;
                for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
                    sum += input[i] + input[i + 1] + input[i + 2] + input[i + 3];
                    input_values[t][i] = input[i];
                    input_values[t][i + 1] = input[i + 1] / 16.0f;
                    input_values[t][i + 2] = input[i + 2] / 256.0f;
                    input_values[t][i + 3] = input[i + 3] / 4096.0f;
                }
                sums[t] = sum;
            }

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                const device uint16_t* packed =
                    (const device uint16_t*)(weight_block + row * weight_row_bytes);
                const device T* row_scales = scale_block + row * groups_per_row;
                const device T* row_biases = bias_block + row * groups_per_row;
                float scale = float(row_scales[0]);
                float bias = float(row_biases[0]);
                for (int t = 0; t < VERIFY_T; ++t) {
                    float dot = 0.0f;
                    for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
                        dot +=
                            input_values[t][4 * i] * (packed[i] & 0x000f)
                            + input_values[t][4 * i + 1] * (packed[i] & 0x00f0)
                            + input_values[t][4 * i + 2] * (packed[i] & 0x0f00)
                            + input_values[t][4 * i + 3] * (packed[i] & 0xf000);
                    }
                    result[t][row] += scale * dot + sums[t] * bias;
                }
            }

            weight_block += BLOCK_SIZE / 2;
            scale_block += BLOCK_SIZE / 64;
            bias_block += BLOCK_SIZE / 64;
            input_block += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int output = output_row + row;
            for (int t = 0; t < VERIFY_T; ++t) {
                float reduced = simd_sum(result[t][row]);
                if (lane == 0) {
                    y[(int(batch) * VERIFY_T + t) * N_SIZE + output] = T(reduced);
                }
            }
        }
    """,
    header: "using namespace metal;",
    ensureRowContiguous: true)

/// Two same-shape exact projections sharing one launch. Matrix ownership is
/// uniform per threadgroup; lane/K ownership and every arithmetic operation
/// remain identical to the single-projection M1-ordered kernel above.
private let qwen35A3BExactW4G64PairVerifyKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_exact_w4_g64_verify_pair_narrow2",
    inputNames: [
        "x", "w0", "scales0", "biases0", "w1", "scales1", "biases1",
    ],
    outputNames: ["y0", "y1"],
    source: """
        uint combined_tile = threadgroup_position_in_grid.y;
        uint matrix = combined_tile / TILES_PER_MATRIX;
        uint n_tile = combined_tile - matrix * TILES_PER_MATRIX;
        uint batch = threadgroup_position_in_grid.z;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        constexpr int VALUES_PER_THREAD = 16;
        constexpr int BLOCK_SIZE = 512;
        constexpr int RESULTS_PER_SIMDGROUP = 2;
        constexpr int OUTPUTS_PER_THREADGROUP = 4;

        int output_row = int(n_tile) * OUTPUTS_PER_THREADGROUP
            + int(simd_group) * RESULTS_PER_SIMDGROUP;
        int weight_row_bytes = K_SIZE / 2;
        int groups_per_row = K_SIZE / 64;

        const device uint8_t* selected_weights = matrix == 0
            ? (const device uint8_t*)w0 : (const device uint8_t*)w1;
        const device T* selected_scales = matrix == 0 ? scales0 : scales1;
        const device T* selected_biases = matrix == 0 ? biases0 : biases1;
        device T* selected_output = matrix == 0 ? y0 : y1;

        const device uint8_t* weight_base =
            selected_weights + output_row * weight_row_bytes + int(lane) * 8;
        const device T* scale_base =
            selected_scales + output_row * groups_per_row + int(lane) / 4;
        const device T* bias_base =
            selected_biases + output_row * groups_per_row + int(lane) / 4;
        const device T* input_base =
            x + int(batch) * VERIFY_T * K_SIZE + int(lane) * VALUES_PER_THREAD;

        float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
        float input_values[VERIFY_T][VALUES_PER_THREAD];
        for (int t = 0; t < VERIFY_T; ++t) {
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                result[t][row] = 0.0f;
            }
        }

        const device uint8_t* weight_block = weight_base;
        const device T* scale_block = scale_base;
        const device T* bias_block = bias_base;
        const device T* input_block = input_base;

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
            float sums[VERIFY_T];
            for (int t = 0; t < VERIFY_T; ++t) {
                const device T* input = input_block + t * K_SIZE;
                float sum = 0.0f;
                for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
                    sum += input[i] + input[i + 1] + input[i + 2] + input[i + 3];
                    input_values[t][i] = input[i];
                    input_values[t][i + 1] = input[i + 1] / 16.0f;
                    input_values[t][i + 2] = input[i + 2] / 256.0f;
                    input_values[t][i + 3] = input[i + 3] / 4096.0f;
                }
                sums[t] = sum;
            }

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                const device uint16_t* packed =
                    (const device uint16_t*)(weight_block + row * weight_row_bytes);
                const device T* row_scales = scale_block + row * groups_per_row;
                const device T* row_biases = bias_block + row * groups_per_row;
                float scale = float(row_scales[0]);
                float bias = float(row_biases[0]);
                for (int t = 0; t < VERIFY_T; ++t) {
                    float dot = 0.0f;
                    for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
                        dot +=
                            input_values[t][4 * i] * (packed[i] & 0x000f)
                            + input_values[t][4 * i + 1] * (packed[i] & 0x00f0)
                            + input_values[t][4 * i + 2] * (packed[i] & 0x0f00)
                            + input_values[t][4 * i + 3] * (packed[i] & 0xf000);
                    }
                    result[t][row] += scale * dot + sums[t] * bias;
                }
            }

            weight_block += BLOCK_SIZE / 2;
            scale_block += BLOCK_SIZE / 64;
            bias_block += BLOCK_SIZE / 64;
            input_block += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int output = output_row + row;
            for (int t = 0; t < VERIFY_T; ++t) {
                float reduced = simd_sum(result[t][row]);
                if (lane == 0) {
                    selected_output[
                        (int(batch) * VERIFY_T + t) * N_SIZE + output] = T(reduced);
                }
            }
        }
    """,
    header: "using namespace metal;",
    ensureRowContiguous: true)

/// Four exact projections sharing one launch. Each threadgroup selects one
/// matrix uniformly; the selected projection retains the single-projection
/// kernel's row ownership, K traversal, lane mapping, and reduction order.
private let qwen35A3BExactW4G64QuadVerifyKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_exact_w4_g64_verify_quad_narrow2",
    inputNames: [
        "x",
        "w0", "scales0", "biases0",
        "w1", "scales1", "biases1",
        "w2", "scales2", "biases2",
        "w3", "scales3", "biases3",
    ],
    outputNames: ["y0", "y1", "y2", "y3"],
    source: """
        uint combined_tile = threadgroup_position_in_grid.y;
        uint matrix;
        uint n_tile;
        if (combined_tile < TILES0) {
            matrix = 0;
            n_tile = combined_tile;
        } else if (combined_tile < TILES01) {
            matrix = 1;
            n_tile = combined_tile - TILES0;
        } else if (combined_tile < TILES012) {
            matrix = 2;
            n_tile = combined_tile - TILES01;
        } else {
            matrix = 3;
            n_tile = combined_tile - TILES012;
        }
        uint batch = threadgroup_position_in_grid.z;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        constexpr int VALUES_PER_THREAD = 16;
        constexpr int BLOCK_SIZE = 512;
        constexpr int RESULTS_PER_SIMDGROUP = 2;
        constexpr int OUTPUTS_PER_THREADGROUP = 4;

        int output_row = int(n_tile) * OUTPUTS_PER_THREADGROUP
            + int(simd_group) * RESULTS_PER_SIMDGROUP;
        int weight_row_bytes = K_SIZE / 2;
        int groups_per_row = K_SIZE / 64;

        const device uint8_t* selected_weights = (const device uint8_t*)w0;
        const device T* selected_scales = scales0;
        const device T* selected_biases = biases0;
        device T* selected_output = y0;
        int output_size = N0_SIZE;
        if (matrix == 1) {
            selected_weights = (const device uint8_t*)w1;
            selected_scales = scales1;
            selected_biases = biases1;
            selected_output = y1;
            output_size = N1_SIZE;
        } else if (matrix == 2) {
            selected_weights = (const device uint8_t*)w2;
            selected_scales = scales2;
            selected_biases = biases2;
            selected_output = y2;
            output_size = N2_SIZE;
        } else if (matrix == 3) {
            selected_weights = (const device uint8_t*)w3;
            selected_scales = scales3;
            selected_biases = biases3;
            selected_output = y3;
            output_size = N3_SIZE;
        }

        const device uint8_t* weight_base =
            selected_weights + output_row * weight_row_bytes + int(lane) * 8;
        const device T* scale_base =
            selected_scales + output_row * groups_per_row + int(lane) / 4;
        const device T* bias_base =
            selected_biases + output_row * groups_per_row + int(lane) / 4;
        const device T* input_base =
            x + int(batch) * VERIFY_T * K_SIZE + int(lane) * VALUES_PER_THREAD;

        float result[VERIFY_T][RESULTS_PER_SIMDGROUP];
        float input_values[VERIFY_T][VALUES_PER_THREAD];
        for (int t = 0; t < VERIFY_T; ++t) {
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                result[t][row] = 0.0f;
            }
        }

        const device uint8_t* weight_block = weight_base;
        const device T* scale_block = scale_base;
        const device T* bias_block = bias_base;
        const device T* input_block = input_base;

        for (int k = 0; k < K_SIZE; k += BLOCK_SIZE) {
            float sums[VERIFY_T];
            for (int t = 0; t < VERIFY_T; ++t) {
                const device T* input = input_block + t * K_SIZE;
                float sum = 0.0f;
                for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
                    sum += input[i] + input[i + 1] + input[i + 2] + input[i + 3];
                    input_values[t][i] = input[i];
                    input_values[t][i + 1] = input[i + 1] / 16.0f;
                    input_values[t][i + 2] = input[i + 2] / 256.0f;
                    input_values[t][i + 3] = input[i + 3] / 4096.0f;
                }
                sums[t] = sum;
            }

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
                const device uint16_t* packed =
                    (const device uint16_t*)(weight_block + row * weight_row_bytes);
                const device T* row_scales = scale_block + row * groups_per_row;
                const device T* row_biases = bias_block + row * groups_per_row;
                float scale = float(row_scales[0]);
                float bias = float(row_biases[0]);
                for (int t = 0; t < VERIFY_T; ++t) {
                    float dot = 0.0f;
                    for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {
                        dot +=
                            input_values[t][4 * i] * (packed[i] & 0x000f)
                            + input_values[t][4 * i + 1] * (packed[i] & 0x00f0)
                            + input_values[t][4 * i + 2] * (packed[i] & 0x0f00)
                            + input_values[t][4 * i + 3] * (packed[i] & 0xf000);
                    }
                    result[t][row] += scale * dot + sums[t] * bias;
                }
            }

            weight_block += BLOCK_SIZE / 2;
            scale_block += BLOCK_SIZE / 64;
            bias_block += BLOCK_SIZE / 64;
            input_block += BLOCK_SIZE;
        }

        for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            int output = output_row + row;
            for (int t = 0; t < VERIFY_T; ++t) {
                float reduced = simd_sum(result[t][row]);
                if (lane == 0) {
                    selected_output[
                        (int(batch) * VERIFY_T + t) * output_size + output] = T(reduced);
                }
            }
        }
    """,
    header: "using namespace metal;",
    ensureRowContiguous: true)

/// Pure shape transform used by the exact fallback and its construction test.
/// The projection is built once per verify column, so every call has logical
/// M1 even though the results are returned as the original rectangle.
func qwen35A3BTimewiseProjection(
    _ input: MLXArray, projection: (MLXArray) -> MLXArray
) -> MLXArray {
    precondition(input.ndim == 3 && input.dim(1) > 1)
    return concatenated(
        (0 ..< input.dim(1)).map { position in
            projection(input[0..., position ..< (position + 1), 0...])
        }, axis: 1)
}

func qwen35A3BExactW4G64Projection(
    _ linear: Linear, _ input: MLXArray
) -> MLXArray {
    let quantized = unsafeDowncast(linear, to: QuantizedLinear.self)
    let quantizationBiases = quantized.biases!
    let batch = input.dim(0)
    let width = input.dim(1)
    let inputSize = input.dim(2)
    let outputSize = quantized.weight.dim(0)
    return qwen35A3BExactW4G64VerifyKernel(
        [input, quantized.weight, quantized.scales, quantizationBiases],
        template: [
            ("T", input.dtype),
            ("VERIFY_T", width),
            ("K_SIZE", inputSize),
            ("N_SIZE", outputSize),
        ],
        grid: (32, 2 * (outputSize / 4), batch),
        threadGroup: (32, 2, 1),
        outputShapes: [[batch, width, outputSize]],
        outputDTypes: [input.dtype])[0]
}

func qwen35A3BExactW4G64ProjectionPair(
    _ first: Linear, _ second: Linear, _ input: MLXArray
) -> (MLXArray, MLXArray) {
    let firstQuantized = unsafeDowncast(first, to: QuantizedLinear.self)
    let secondQuantized = unsafeDowncast(second, to: QuantizedLinear.self)
    let batch = input.dim(0)
    let width = input.dim(1)
    let inputSize = input.dim(2)
    let outputSize = firstQuantized.weight.dim(0)
    let outputs = qwen35A3BExactW4G64PairVerifyKernel(
        [
            input,
            firstQuantized.weight, firstQuantized.scales, firstQuantized.biases!,
            secondQuantized.weight, secondQuantized.scales, secondQuantized.biases!,
        ],
        template: [
            ("T", input.dtype),
            ("VERIFY_T", width),
            ("K_SIZE", inputSize),
            ("N_SIZE", outputSize),
            ("TILES_PER_MATRIX", outputSize / 4),
        ],
        grid: (32, 4 * (outputSize / 4), batch),
        threadGroup: (32, 2, 1),
        outputShapes: [
            [batch, width, outputSize], [batch, width, outputSize],
        ],
        outputDTypes: [input.dtype, input.dtype])
    return (outputs[0], outputs[1])
}

func qwen35A3BExactW4G64ProjectionQuad(
    _ first: Linear, _ second: Linear, _ third: Linear, _ fourth: Linear,
    _ input: MLXArray
) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    let q0 = unsafeDowncast(first, to: QuantizedLinear.self)
    let q1 = unsafeDowncast(second, to: QuantizedLinear.self)
    let q2 = unsafeDowncast(third, to: QuantizedLinear.self)
    let q3 = unsafeDowncast(fourth, to: QuantizedLinear.self)
    let batch = input.dim(0)
    let width = input.dim(1)
    let inputSize = input.dim(2)
    let n0 = q0.weight.dim(0)
    let n1 = q1.weight.dim(0)
    let n2 = q2.weight.dim(0)
    let n3 = q3.weight.dim(0)
    let tiles0 = n0 / 4
    let tiles01 = tiles0 + n1 / 4
    let tiles012 = tiles01 + n2 / 4
    let outputs = qwen35A3BExactW4G64QuadVerifyKernel(
        [
            input,
            q0.weight, q0.scales, q0.biases!,
            q1.weight, q1.scales, q1.biases!,
            q2.weight, q2.scales, q2.biases!,
            q3.weight, q3.scales, q3.biases!,
        ],
        template: [
            ("T", input.dtype),
            ("VERIFY_T", width),
            ("K_SIZE", inputSize),
            ("N0_SIZE", n0),
            ("N1_SIZE", n1),
            ("N2_SIZE", n2),
            ("N3_SIZE", n3),
            ("TILES0", tiles0),
            ("TILES01", tiles01),
            ("TILES012", tiles012),
        ],
        grid: (32, 2 * (tiles012 + n3 / 4), batch),
        threadGroup: (32, 2, 1),
        outputShapes: [
            [batch, width, n0],
            [batch, width, n1],
            [batch, width, n2],
            [batch, width, n3],
        ],
        outputDTypes: [input.dtype, input.dtype, input.dtype, input.dtype])
    return (outputs[0], outputs[1], outputs[2], outputs[3])
}

/// Installed W8/unquantized route. Its matrices are small enough that explicit
/// M1 projection calls are faster than rereading every W4 target matrix.
func qwen35A3BExactTimewiseProjection(
    _ linear: Linear, _ input: MLXArray
) -> MLXArray {
    qwen35A3BTimewiseProjection(input) { linear($0) }
}

/// The strict-prefix commit replay fused into the next verify's scan
/// (`BONSAI_GDN_REPLAY_FUSED=0` keeps the replay).
///
/// The replay (`Qwen35GDNReplayBatch`) reads each GDN layer's pre-verify state
/// and writes the committed state, which the next verify's scan reads again:
/// 3.1 MB per layer each way. For a single row the commit stays deferred
/// (`CBv2DeferredRecurrentReplay`, its conv rows a lazy slice of the tape),
/// and the next verify runs one launch per layer: phase 1 is the replay's step
/// loop over the kept rows (the batched replay's text, no output) from the
/// previous pre-verify state, then the stock store writes the committed state
/// (this round's pre-verify state), then phase 2 is the output-only verify's
/// step loop, text and template unchanged, over this round's rows. Between the
/// phases the state stays in registers, where an FP32 store and reload would
/// give the same bits. Any other reader of a deferred state builds the replay.
///
/// At model construction a self-test compares the output rows and the
/// committed state, as unsigned integers, with the per-layer replay followed by
/// `runOutputOnly`, and the state with the batched replay, for 0, 1, 7 and 16
/// kept rows of four synthetic tapes; a mismatch or MLX error keeps the replay.
/// (A Qwen35.swift verify kernel, kept here: that file is near its byte cap.)
enum Qwen35GDNReplayFused {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_REPLAY_FUSED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// A deferred commit's replay inputs: the verify tape and its layer.
    final class Inputs {
        let layer: ObjectIdentifier
        let tape: ArraysCache.PrefixReplayTape
        init(layer: Qwen35GatedDeltaNet, tape: ArraysCache.PrefixReplayTape) {
            self.layer = ObjectIdentifier(layer)
            self.tape = tape
        }
    }

    /// `Qwen35GatedDeltaV3.source` loading the previous pre-verify state `ps`,
    /// with a nested block before its step loop: that loop over the previous
    /// tape's `KP` rows with the batched replay's replacements and
    /// `OUTPUT_NEEDED` false, then the stock state store.
    static let source: String? = {
        guard let parts = Qwen35GatedDeltaV3.sourceParts else { return nil }
        let load = "state[d][i] = state_in[(n * Dv + dvbase + d) * Dk + dk0 + i];"
        guard parts.head.components(separatedBy: load).count == 2 else { return nil }
        let head = parts.head.replacingOccurrences(
            of: load, with: "state[d][i] = ps[(n * Dv + dvbase + d) * Dk + dk0 + i];")
        var replay = parts.loop
        for (target, replacement) in [
            ("for (int t = 0; t < T; ++t) {", "for (int t = 0; t < KP; ++t) {"),
            (
                "const float gt = g_[0];",
                "const float g_sp = qwen35_replay_logaddexp(a_[0] + g_dtb, 0.0f);\n"
                    + "const float gt = metal::precise::exp(g_nexp * g_sp);"
            ),
            ("const float bt = beta_[0];", "const float bt = qwen35_replay_sigmoid(b_[0]);"),
            (
                "k_ += Hk * Dk; v_ += Hv * Dv; g_ += Hv; beta_ += Hv;",
                "k_ += Hk * Dk; v_ += Hv * Dv; a_ += a_rs; b_ += b_rs;"
            ),
        ] {
            guard replay.components(separatedBy: target).count == 2 else { return nil }
            replay = replay.replacingOccurrences(of: target, with: replacement)
        }
        replay = replay.replacingOccurrences(of: "OUTPUT_NEEDED", with: "false")
        guard !replay.contains("g_["), !replay.contains("beta_"), !head.contains("state_in")
        else { return nil }
        return head + """
            {
              const device float* k_ = pk + hk_idx * Dk + dk0;
              const device float* v_ = pv + hv_idx * Dv + dvbase;
              const device float* a_ = pa + hv_idx;
              const device float* b_ = pb + hv_idx;
              const int a_rs = ab_rows[0];
              const int b_rs = ab_rows[1];
              const float g_nexp = -metal::precise::exp(alog[hv_idx]);
              const float g_dtb = dtb[hv_idx];

            """ + replay + "\n}\n" + parts.store + "\n" + parts.loop + "\n"
    }()

    private static let kernel: MLXFast.MLXFastKernel? = source.map {
        MLXFast.metalKernel(
            name: "qwen35_gdn_replay_fused",
            inputNames: [
                "q", "k", "v", "g", "beta", "T", "ps", "pk", "pv", "pa", "pb", "alog", "dtb",
                "ab_rows", "KP",
            ],
            outputNames: ["y", "state_out"],
            source: $0,
            header: Qwen35GDNReplayBatch.header,
            ensureRowContiguous: false)
    }

    /// `BONSAI_REPLAY_EVAL_SKIP=0` keeps the replays' stride waits on
    /// inputs that are already available.
    static let evalSkipEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_REPLAY_EVAL_SKIP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `eval(arrays)` unless every array is already available (evaluated,
    /// its event signaled): that eval would wait on nothing and change no
    /// value or stride. Any other array is evaluated and waited for.
    static func evalUnlessAvailable(_ arrays: [MLXArray]) {
        if evalSkipEnabled, arrays.allSatisfy({
            var available = false
            return _mlx_array_is_available(&available, $0.ctx) == 0 && available
        }) { return }
        eval(arrays)
    }

    /// One row's `(y, committed state)`: `keep` rows of the previous `tape`
    /// from its pre-verify state, then this verify's rows (`q` ... `beta`,
    /// the prework's row-contiguous FP32 outputs). Nil when it does not fit.
    static func launch(
        tape: ArraysCache.PrefixReplayTape, keep: Int, aLog: MLXArray, dtBias: MLXArray,
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
        staged: Bool? = nil, storeFinal: Bool = false
    ) -> (y: MLXArray, state: MLXArray, final: MLXArray?)? {
        guard let kernel, Qwen35GatedDeltaV3.enabled, let ps = tape.ssmPre, tape.mask == nil,
            k.ndim == 4, v.ndim == 4
        else { return nil }
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Dk = k.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        let P = tape.rowCount
        let dvpl = Qwen35GatedDeltaV3.rowsPerLane
        let previous = [ps, tape.k, tape.v, tape.a, tape.b, aLog, dtBias]
        // The replay's own routing: from `minRows` kept rows it is chunked.
        guard ([q, k, v, g, beta] + previous).allSatisfy({ $0.dtype == .float32 }),
            !(Qwen35GatedDeltaChunked.enabled && keep >= Qwen35GatedDeltaChunked.minRows
                && keep >= Qwen35GatedDeltaChunked.chunk),
            Dk == 128, Dv % (16 * dvpl) == 0, Hv % Hk == 0, T > 0, keep >= 0, keep <= P,
            q.shape == [1, T, Hk, Dk], k.shape == q.shape, v.shape == [1, T, Hv, Dv],
            g.shape == [1, T, Hv], beta.shape == [1, T, Hv], ps.shape == [1, Hv, Dv, Dk],
            tape.k.shape == [1, P, Hk, Dk], tape.v.shape == [1, P, Hv, Dv],
            tape.a.shape == [1, P, Hv], tape.b.shape == [1, P, Hv],
            aLog.shape == [Hv], dtBias.shape == [Hv]
        else { return nil }
        // Read in place, as the batched replay reads them: evaluated with
        // their verify, so this wait is a no-op.
        evalUnlessAvailable(previous)
        guard Qwen35GDNReplayBatch.rowContiguousAfterLeading(ps),
            Qwen35GDNReplayBatch.rowContiguousAfterLeading(tape.k),
            Qwen35GDNReplayBatch.rowContiguousAfterLeading(tape.v),
            aLog.strides == [1], dtBias.strides == [1],
            let aRows = Qwen35GDNReplayBatch.gateRowStride(tape.a),
            let bRows = Qwen35GDNReplayBatch.gateRowStride(tape.b)
        else { return nil }
        // Both windows fit the staging buffers (and the verify window is a
        // full one, where staging pays): the staged form (same values).
        let useStaged =
            (staged ?? stagedActive) && T >= stagedRows / 2 && T <= stagedRows
            && P <= stagedRows
        var inputs = [q, k, v, g, beta, MLXArray(Int32(T))] + previous
            + [MLXArray([aRows, bRows]), MLXArray(Int32(keep))]
        // Reuse this tape's verify gates when their layout fits the staged read.
        var storedGates = false
        if useStaged, let pg = tape.g, let pb = tape.beta,
            pg.dtype == .float32, pb.dtype == .float32,
            pg.shape == [1, P, Hv], pb.shape == [1, P, Hv],
            pg.strides == [P * Hv, Hv, 1], pb.strides == [P * Hv, Hv, 1]
        {
            inputs[9] = pg
            inputs[10] = pb
            inputs[13] = MLXArray([Int32(Hv), Int32(Hv)])
            storedGates = true
        }
        let template: [(String, any KernelTemplateArg)] = [
            ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("OUTPUT_NEEDED", true),
            ("DVPL", dvpl),
        ]
        if useStaged {
            // The committed state, and (`storeFinal`) the window's final state.
            let out = stagedKernel(
                inputs, template: template + [
                    ("SC", true), ("SF", storeFinal), ("GATES_STORED", storedGates),
                    ("REPLAY_NEEDED", keep > 0),
                ],
                grid: (128, Dv / (16 * dvpl), Hv), threadGroup: (128, 1, 1),
                outputShapes: [[1, T, Hv, Dv], ps.shape, storeFinal ? ps.shape : [1]],
                outputDTypes: [.float32, .float32, .float32])
            return (out[0], out[1], storeFinal ? out[2] : nil)
        }
        let out = kernel(
            inputs, template: template,
            grid: (128, Dv / (16 * dvpl), Hv), threadGroup: (128, 1, 1),
            outputShapes: [[1, T, Hv, Dv], ps.shape],
            outputDTypes: [.float32, .float32])
        return (out[0], out[1], nil)
    }

    /// `BONSAI_GDN_PLAIN_STAGED=0` keeps the stock output-only scan for a
    /// verify from a committed (not deferred) state that stores nothing.
    static let plainStagedEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_PLAIN_STAGED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// This verify's rows from a committed (not deferred) `state` (and, with
    /// `storeFinal`, the window's final state): the staged kernel without the
    /// replay (`KP` 0) and without the committed-state store. The same step
    /// arithmetic as `Qwen35GatedDeltaV3.run` / `runOutputOnly`, whose rows
    /// and final state it matches bit for bit (`Qwen35GDNFullAcceptStore`'s
    /// self-test). Nil when it does not fit (the caller takes the stock path).
    static func plainScan(
        pre: Qwen35GDNPrework.Outputs, state: MLXArray, aLog: MLXArray, dtBias: MLXArray,
        storeFinal: Bool = true
    ) -> (y: MLXArray, final: MLXArray?)? {
        let (q, k, v, g, beta) = (pre.q, pre.k, pre.v, pre.g, pre.beta)
        guard stagedActive, Qwen35GatedDeltaV3.enabled, k.ndim == 4, v.ndim == 4 else {
            return nil
        }
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Dk = k.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        let dvpl = Qwen35GatedDeltaV3.rowsPerLane
        guard [q, k, v, g, beta, state, aLog, dtBias].allSatisfy({ $0.dtype == .float32 }),
            Dk == 128, Dv % (16 * dvpl) == 0, Hv % Hk == 0,
            T >= stagedRows / 2, T <= stagedRows,
            q.shape == [1, T, Hk, Dk], k.shape == q.shape, v.shape == [1, T, Hv, Dv],
            g.shape == [1, T, Hv], beta.shape == [1, T, Hv], state.shape == [1, Hv, Dv, Dk],
            aLog.shape == [Hv], dtBias.shape == [Hv],
            Qwen35GDNReplayBatch.rowContiguousAfterLeading(state),
            aLog.strides == [1], dtBias.strides == [1]
        else { return nil }
        // KP = 0: the replay inputs are never read; any float arrays bind.
        let out = stagedKernel(
            [q, k, v, g, beta, MLXArray(Int32(T)), state, k, v, g, beta, aLog, dtBias,
             MLXArray([Int32(Hv), Int32(Hv)]), MLXArray(Int32(0))],
            template: [
                ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("OUTPUT_NEEDED", true),
                ("DVPL", dvpl), ("SC", false), ("SF", storeFinal),
                ("GATES_STORED", false), ("REPLAY_NEEDED", false),
            ],
            grid: (128, Dv / (16 * dvpl), Hv), threadGroup: (128, 1, 1),
            outputShapes: [[1, T, Hv, Dv], [1], storeFinal ? state.shape : [1]],
            outputDTypes: [.float32, .float32, .float32])
        return (out[0], storeFinal ? out[2] : nil)
    }

    /// The fused scan of a single-row verify whose input SSM is `layer`'s
    /// pending deferred replay, which it resolves with the committed state.
    static func run(
        layer: Qwen35GatedDeltaNet, input: CBv2RecurrentLayerState?,
        pre: Qwen35GDNPrework.Outputs, storeFinal: Bool = false
    ) -> (y: MLXArray, state: MLXArray, final: MLXArray?)? {
        guard let deferred = input?.deferredReplay, deferred.isPending,
            let inputs = deferred.inputs as? Inputs, inputs.layer == ObjectIdentifier(layer),
            applies(to: layer),
            let fused = launch(
                tape: inputs.tape, keep: deferred.keep, aLog: layer.aLog,
                dtBias: layer.dtBias, q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta,
                storeFinal: storeFinal),
            deferred.resolve(fused.state)
        else { return nil }
        return fused
    }

    private struct Geometry: Hashable {
        let hk: Int, dk: Int, hv: Int, dv: Int, dvpl: Int
    }

    private static func geometry(_ layer: Qwen35GatedDeltaNet) -> Geometry {
        Geometry(
            hk: layer.numKHeads, dk: layer.headKDim, hv: layer.numVHeads, dv: layer.headVDim,
            dvpl: Qwen35GatedDeltaV3.rowsPerLane)
    }

    private static let verdictLock = NSLock()
    nonisolated(unsafe) private static var verdicts: [Geometry: Bool] = [:]

    /// Whether `layer`'s verify takes the output-only scan this kernel extends
    /// (state skip on, chunked verify off) and its self-test passed.
    static func applies(to layer: Qwen35GatedDeltaNet) -> Bool {
        guard enabled, !Qwen35GatedDeltaChunked.verifyEnabled,
            Qwen35GDNVerifyStateSkip.applies(to: layer)
        else { return false }
        let key = geometry(layer)
        return verdictLock.withLock { verdicts[key] ?? false }
    }

    private enum SelfTestFailure: Error {
        case message(String)
    }

    /// Run the bitwise self-test once per geometry at model construction,
    /// after `Qwen35GDNVerifyStateSkip.prepare` (compiling the kernel).
    static func prepare(layer: Qwen35GatedDeltaNet) {
        guard enabled, Qwen35GatedDeltaV3.enabled else { return }
        let key = geometry(layer)
        verdictLock.lock()
        defer { verdictLock.unlock() }
        guard verdicts[key] == nil else { return }
        var (passed, detail) = selfTest(layer: layer)
        if !passed, stagedActive {
            // The staged form failed: retest (and keep) the stock fused kernel.
            stagedLive = false
            let first = detail
            (passed, detail) = selfTest(layer: layer)
            detail += "; staged form FAILED (\(first)), stock fused kernel"
        }
        verdicts[key] = passed
        Memory.clearCache()
        FileHandle.standardError.write(
            ("qwen35 GDN replay fused: self-test " + (passed ? "passed" : "FAILED") + " ("
                + detail + ")"
                + (passed ? "; the next verify replays the committed prefix" : "; replay kept")
                + (passed && stagedActive ? ", rows staged in threadgroup memory\n" : "\n"))
                .data(using: .utf8)!)
    }

    /// The batch self-test's tapes (a/b column slices of one product, one row
    /// of saturating and infinite gate inputs) and a following window whose
    /// gates `Qwen35FusedElementwise.gatedDeltaGates` forms from such inputs.
    private static func selfTest(layer: Qwen35GatedDeltaNet) -> (Bool, String) {
        let G = Qwen35GDNReplayBatch.layersPerLaunch
        let S = Qwen35GDNReplayBatch.selfTestRows
        let Hk = layer.numKHeads
        let Dk = layer.headKDim
        let Hv = layer.numVHeads
        let Dv = layer.headVDim
        let NK = layer.convKernelSize - 1
        let batched = Qwen35GDNReplayBatch.isVerified(layer: layer)
        let keys = MLXRandom.split(key: MLXRandom.key(0x4644_5250), into: 16 * G)
        let specials: [Float] = [60, -60, 25, -25, .infinity, -.infinity, 1e-8, -1e-8]
        let marks = MLXArray((0 ..< (2 * Hv)).map { specials[$0 % specials.count] })
            .reshaped([1, 1, 2 * Hv])
        var comparisons = 0
        var values = 0
        var mismatches = 0
        do {
            try withError { error in
                var differ: [MLXArray] = []
                func compare(_ a: MLXArray?, _ b: MLXArray?) throws {
                    guard let a, let b, a.shape == b.shape, a.dtype == .float32,
                        b.dtype == .float32
                    else { throw SelfTestFailure.message("shape or dtype mismatch") }
                    differ.append(
                        (a.view(dtype: .uint32) .!= b.view(dtype: .uint32)).asType(.int32).sum())
                    values += a.size
                    comparisons += 1
                }
                var operands: [Qwen35GDNReplayBatch.Operand] = []
                var windows: [[MLXArray]] = []
                for j in 0 ..< G {
                    func key(_ i: Int) -> MLXArray { keys[16 * j + i] }
                    func gatePair(_ i: Int) -> MLXArray {
                        let row = MLXArray((0 ..< S).map { $0 == (j + i) % 3 ? Float(1) : 0 })
                        return MLX.where(
                            row.reshaped([1, S, 1]) .> 0, marks,
                            MLXRandom.normal([1, S, 2 * Hv], key: key(i)) * 4)
                    }
                    let spread = exp(MLXRandom.normal([1, Hv, Dv, Dk], key: key(0)))
                    let ssmPre = MLXRandom.normal([1, Hv, Dv, Dk], key: key(1)) * spread * 0.05
                    let pair = gatePair(2)
                    let next = gatePair(3)
                    func rows(_ i: Int, _ h: Int, _ d: Int, _ s: Float) -> MLXArray {
                        MLXRandom.normal([1, S, h, d], key: key(i)) * s
                    }
                    let v = rows(4, Hv, Dv, 1) * exp(rows(5, Hv, Dv, 1))
                    let nextV = rows(6, Hv, Dv, 1) * exp(rows(7, Hv, Dv, 1))
                    let convInput = MLXRandom.normal([1, NK + S, layer.convDim], key: key(8))
                    let aLog = log(MLXRandom.uniform(Float(1) ..< Float(16), [Hv], key: key(9)))
                    let dtBias = MLXRandom.normal([Hv], key: key(10))
                    let gates = Qwen35FusedElementwise.gatedDeltaGates(
                        [next[0..., 0..., Hv...], next[0..., 0..., ..<Hv], aLog, dtBias])
                    let window = [
                        rows(11, Hk, Dk, 0.09), rows(12, Hk, Dk, 0.09), nextV, gates[0], gates[1],
                    ]
                    let previousGates = Qwen35FusedElementwise.gatedDeltaGates(
                        [pair[0..., 0..., Hv...], pair[0..., 0..., ..<Hv], aLog, dtBias])
                    let tape = ArraysCache.PrefixReplayTape(
                        convInput: convInput, q: rows(13, Hk, Dk, 0.09), k: rows(14, Hk, Dk, 0.09),
                        v: v, a: pair[0..., 0..., Hv...], b: pair[0..., 0..., ..<Hv],
                        ssmPre: ssmPre, mask: nil, rowCount: S, convStateRows: NK,
                        g: previousGates[0], beta: previousGates[1])
                    eval(window + previousGates
                        + [ssmPre, convInput, tape.q, tape.k, v, tape.a, tape.b, aLog, dtBias])
                    operands.append(
                        Qwen35GDNReplayBatch.Operand(tape: tape, aLog: aLog, dtBias: dtBias))
                    windows.append(window)
                }
                for keep in [0, 1, 7, S] {
                    let states =
                        batched && keep > 0
                        ? Qwen35GDNReplayBatch.launch(
                            operands, keep: keep,
                            alog: concatenated(operands.map(\.aLog), axis: 0),
                            dtb: concatenated(operands.map(\.dtBias), axis: 0),
                            verifiedOnly: false)
                        : nil
                    if batched && keep > 0 && states == nil {
                        throw SelfTestFailure.message("no batched replay at \(keep) rows")
                    }
                    for (j, operand) in operands.enumerated() {
                        let w = windows[j]
                        let committed =
                            keep == 0
                            ? operand.tape.ssmPre
                            : layer.replayedPrefixState(
                                tape: operand.tape, committedRows: keep, aLog: operand.aLog,
                                dtBias: operand.dtBias, fullWindow: true
                            ).ssm
                        guard let committed,
                            let y = Qwen35GatedDeltaV3.runOutputOnly(
                                q: w[0], k: w[1], v: w[2], g: w[3], beta: w[4], state: committed),
                            let fused = launch(
                                tape: operand.tape, keep: keep, aLog: operand.aLog,
                                dtBias: operand.dtBias, q: w[0], k: w[1], v: w[2], g: w[3],
                                beta: w[4])
                        else { throw SelfTestFailure.message("no launch at \(keep) rows") }
                        try compare(fused.y, y)
                        try compare(fused.state, committed)
                        if let states { try compare(fused.state, states[j].ssm) }
                        if stagedActive {
                            guard
                                let stock = launch(
                                    tape: operand.tape, keep: keep, aLog: operand.aLog,
                                    dtBias: operand.dtBias, q: w[0], k: w[1], v: w[2], g: w[3],
                                    beta: w[4], staged: false)
                            else { throw SelfTestFailure.message("no stock launch at \(keep) rows") }
                            try compare(fused.y, stock.y)
                            try compare(fused.state, stock.state)
                        }
                    }
                    let count = stacked(differ).sum()
                    eval(count)
                    try error.check()
                    mismatches += Int(count.item(Int32.self))
                    differ.removeAll()
                }
            }
        } catch {
            return (false, "\(error)")
        }
        let expected = 4 * G * 2 + (batched ? 3 * G : 0) + (stagedActive ? 4 * G * 2 : 0)
        let passed = mismatches == 0 && comparisons == expected
        return (
            passed,
            "\(G) tapes at 0, 1, 7 and \(S) kept rows, \(comparisons) comparisons, "
                + "\(values) values, \(mismatches) mismatches"
                + (batched ? ", batched replay included" : "")
                + (stagedActive ? ", staged form vs stock fused kernel included" : ""))
    }
}

/// `qwen35_gdn_replay_fused` with each window's rows staged in threadgroup
/// memory (`BONSAI_GDN_REPLAY_STAGED=0` keeps the stock fused kernel).
///
/// The stock kernel reads every step's inputs from device memory inside its
/// sequential step loop: per replayed row the previous tape's k and v rows and
/// a/b, from which each of the 128 threads computes the same gates, and per
/// output row q, k, v, g and beta. The previous tape is long out of the
/// caches by the next verify, so each replayed step waits for a DRAM round
/// trip, and the output steps wait for their own loads. Here each window is
/// read once, coalesced, into threadgroup memory before its loop (the replay
/// window at the start, next to the state load; the output window after the
/// committed-state store, reusing the replay's k/v space): the k rows as
/// float4 (swizzled so the eight lanes of a row read different banks), the
/// threadgroup's 32 v columns, and the per-step gates (thread t computes step
/// t's gates with the stock expressions). The loops then read those values
/// from threadgroup memory: every step does the stock arithmetic in the stock
/// order (the output pass reads the updated state exactly as the stock loop's
/// interleaved `o` accumulation does), so y and the committed state are the
/// stock kernel's, bit for bit. Both windows must hold at most 16 rows (the
/// verify window is 16); otherwise the stock kernel runs.
///
/// The fused replay's self-test runs through this form, compared both with
/// the unfused replay and output-only scan (as for the stock kernel) and with
/// the stock fused kernel on the same inputs; on a mismatch the stock fused
/// kernel is retested and kept.
extension Qwen35GDNReplayFused {
    static let stagedEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_REPLAY_STAGED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Cleared when the staged form fails its self-test.
    nonisolated(unsafe) static var stagedLive = true
    static var stagedActive: Bool { stagedEnabled && stagedLive }

    /// Rows per staged window (the threadgroup buffers' size).
    static let stagedRows = 16

    private static let stagedHeader = Qwen35GDNReplayBatch.header + """
        // float4 j of the 16-float k (or q) slice c of a staged row; the
        // swizzle spreads the eight slices of one read over the banks.
        inline uint qwen35_staged_slot(uint c, uint j) {
          return c * 4 + (j ^ ((c >> 1) & 3));
        }

        """

    // The stock kernel's inputs, outputs, template and grid.
    private static let stagedSource = """
        constexpr int R = 16;
        constexpr int LPD = Dk / R;
        constexpr int DVPS = (32 / LPD) * DVPL;
        constexpr int DVPT = DVPS * 4;
        constexpr uint NT = 128;
        static_assert(Dk == 128 && LPD == 8, "16-float k slices, 8 lanes per row");
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = sg * 32 + lane;
        const uint c = lane % LPD;
        const uint dk0 = c * R;
        const uint row0 = threadgroup_position_in_grid.y * DVPT;
        const uint rbase = sg * DVPS + (lane / LPD) * DVPL;
        const uint dvbase = row0 + rbase;
        threadgroup float4 tk[16 * 32];
        threadgroup float4 tq[16 * 32];
        threadgroup float tv[16 * DVPT];
        threadgroup float tgate[32];

        float state[DVPL][R];
        #pragma clang loop unroll(full)
        for (int d = 0; d < DVPL; ++d) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < R; i += 4) {
            const uint base = (n * Dv + dvbase + d) * Dk + dk0 + i;
            const float4 x = *(const device float4*)(ps + base);
            state[d][i] = x.x;
            state[d][i + 1] = x.y;
            state[d][i + 2] = x.z;
            state[d][i + 3] = x.w;
          }
        }

        // Phase 1: the previous tape's KP rows, the batched replay's step.
        // A zero-row replay has no previous staging or recurrence.
        if (REPLAY_NEEDED) {
          const int a_rs = ab_rows[0];
          const int b_rs = ab_rows[1];
          const float g_nexp = -metal::precise::exp(alog[hv_idx]);
          const float g_dtb = dtb[hv_idx];
          const device float4* k4src = (const device float4*)(pk + hk_idx * Dk);
          for (uint e = tid; e < uint(KP) * 32u; e += NT) {
            const uint t = e >> 5, f = e & 31u;
            tk[t * 32u + qwen35_staged_slot(f >> 2, f & 3u)] = k4src[t * uint(Hk * Dk / 4) + f];
          }
          for (uint e = tid; e < uint(KP) * uint(DVPT); e += NT) {
            const uint t = e / uint(DVPT), r = e % uint(DVPT);
            tv[e] = pv[(t * Hv + hv_idx) * Dv + row0 + r];
          }
          if (tid < uint(KP)) {
            if (GATES_STORED) {
              tgate[tid] = pa[hv_idx + tid * a_rs];
              tgate[16 + tid] = pb[hv_idx + tid * b_rs];
            } else {
              const float g_sp = qwen35_replay_logaddexp(pa[hv_idx + tid * a_rs] + g_dtb, 0.0f);
              tgate[tid] = metal::precise::exp(g_nexp * g_sp);
              tgate[16 + tid] = qwen35_replay_sigmoid(pb[hv_idx + tid * b_rs]);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          for (int t = 0; t < KP; ++t) {
            const float gt = tgate[t];
            const float bt = tgate[16 + t];
            const threadgroup float4* kt = tk + t * 32;
            float kv[DVPL];
            #pragma clang loop unroll(full)
            for (int d = 0; d < DVPL; ++d) {
              float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
              #pragma clang loop unroll(full)
              for (int j = 0; j < R / 4; ++j) {
                const float4 k4 = kt[qwen35_staged_slot(c, j)];
                state[d][4 * j] *= gt; state[d][4 * j + 1] *= gt;
                state[d][4 * j + 2] *= gt; state[d][4 * j + 3] *= gt;
                a0 = fma(state[d][4 * j], k4.x, a0);
                a1 = fma(state[d][4 * j + 1], k4.y, a1);
                a2 = fma(state[d][4 * j + 2], k4.z, a2);
                a3 = fma(state[d][4 * j + 3], k4.w, a3);
              }
              kv[d] = (a0 + a1) + (a2 + a3);
            }
            #pragma clang loop unroll(full)
            for (int o = LPD / 2; o > 0; o >>= 1) {
              #pragma clang loop unroll(full)
              for (int d = 0; d < DVPL; ++d) {
                kv[d] += simd_shuffle_xor(kv[d], o);
              }
            }
            #pragma clang loop unroll(full)
            for (int d = 0; d < DVPL; ++d) {
              const float delta = (tv[t * DVPT + rbase + d] - kv[d]) * bt;
              #pragma clang loop unroll(full)
              for (int j = 0; j < R / 4; ++j) {
                const float4 k4 = kt[qwen35_staged_slot(c, j)];
                state[d][4 * j] = fma(k4.x, delta, state[d][4 * j]);
                state[d][4 * j + 1] = fma(k4.y, delta, state[d][4 * j + 1]);
                state[d][4 * j + 2] = fma(k4.z, delta, state[d][4 * j + 2]);
                state[d][4 * j + 3] = fma(k4.w, delta, state[d][4 * j + 3]);
              }
            }
          }
        }

        // The committed state (SC; the plain scan of a committed state skips it).
        if (SC) {
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < R; i += 4) {
              *((device float4*)(state_out + (n * Dv + dvbase + d) * Dk + dk0 + i)) =
                  float4(state[d][i], state[d][i + 1], state[d][i + 2], state[d][i + 3]);
            }
          }
        }

        // Phase 2: this verify's T rows, the output-only scan's step.
        {
          if (REPLAY_NEEDED) threadgroup_barrier(mem_flags::mem_threadgroup);
          const device float4* k4src = (const device float4*)(k + (b_idx * T * Hk + hk_idx) * Dk);
          const device float4* q4src = (const device float4*)(q + (b_idx * T * Hk + hk_idx) * Dk);
          for (uint e = tid; e < uint(T) * 32u; e += NT) {
            const uint t = e >> 5, f = e & 31u;
            const uint slot = t * 32u + qwen35_staged_slot(f >> 2, f & 3u);
            tk[slot] = k4src[t * uint(Hk * Dk / 4) + f];
            tq[slot] = q4src[t * uint(Hk * Dk / 4) + f];
          }
          for (uint e = tid; e < uint(T) * uint(DVPT); e += NT) {
            const uint t = e / uint(DVPT), r = e % uint(DVPT);
            tv[e] = v[((b_idx * T + t) * Hv + hv_idx) * Dv + row0 + r];
          }
          if (tid < uint(T)) {
            tgate[tid] = g[(b_idx * T + tid) * Hv + hv_idx];
            tgate[16 + tid] = beta[(b_idx * T + tid) * Hv + hv_idx];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          device float* y_ = y + (b_idx * T * Hv + hv_idx) * Dv + dvbase;
          for (int t = 0; t < T; ++t) {
            const float gt = tgate[t];
            const float bt = tgate[16 + t];
            const threadgroup float4* kt = tk + t * 32;
            const threadgroup float4* qt = tq + t * 32;
            float kv[DVPL];
            #pragma clang loop unroll(full)
            for (int d = 0; d < DVPL; ++d) {
              float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
              #pragma clang loop unroll(full)
              for (int j = 0; j < R / 4; ++j) {
                const float4 k4 = kt[qwen35_staged_slot(c, j)];
                state[d][4 * j] *= gt; state[d][4 * j + 1] *= gt;
                state[d][4 * j + 2] *= gt; state[d][4 * j + 3] *= gt;
                a0 = fma(state[d][4 * j], k4.x, a0);
                a1 = fma(state[d][4 * j + 1], k4.y, a1);
                a2 = fma(state[d][4 * j + 2], k4.z, a2);
                a3 = fma(state[d][4 * j + 3], k4.w, a3);
              }
              kv[d] = (a0 + a1) + (a2 + a3);
            }
            #pragma clang loop unroll(full)
            for (int o = LPD / 2; o > 0; o >>= 1) {
              #pragma clang loop unroll(full)
              for (int d = 0; d < DVPL; ++d) {
                kv[d] += simd_shuffle_xor(kv[d], o);
              }
            }
            #pragma clang loop unroll(full)
            for (int d = 0; d < DVPL; ++d) {
              const float delta = (tv[t * DVPT + rbase + d] - kv[d]) * bt;
              #pragma clang loop unroll(full)
              for (int j = 0; j < R / 4; ++j) {
                const float4 k4 = kt[qwen35_staged_slot(c, j)];
                state[d][4 * j] = fma(k4.x, delta, state[d][4 * j]);
                state[d][4 * j + 1] = fma(k4.y, delta, state[d][4 * j + 1]);
                state[d][4 * j + 2] = fma(k4.z, delta, state[d][4 * j + 2]);
                state[d][4 * j + 3] = fma(k4.w, delta, state[d][4 * j + 3]);
              }
            }
            float out[DVPL];
            #pragma clang loop unroll(full)
            for (int d = 0; d < DVPL; ++d) {
              float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
              #pragma clang loop unroll(full)
              for (int j = 0; j < R / 4; ++j) {
                const float4 q4 = qt[qwen35_staged_slot(c, j)];
                o0 = fma(state[d][4 * j], q4.x, o0);
                o1 = fma(state[d][4 * j + 1], q4.y, o1);
                o2 = fma(state[d][4 * j + 2], q4.z, o2);
                o3 = fma(state[d][4 * j + 3], q4.w, o3);
              }
              out[d] = (o0 + o1) + (o2 + o3);
            }
            #pragma clang loop unroll(full)
            for (int o = LPD / 2; o > 0; o >>= 1) {
              #pragma clang loop unroll(full)
              for (int d = 0; d < DVPL; ++d) {
                out[d] += simd_shuffle_xor(out[d], o);
              }
            }
            if (lane % LPD == 0) {
              #pragma clang loop unroll(full)
              for (int d = 0; d < DVPL; ++d) {
                y_[d] = out[d];
              }
            }
            y_ += Hv * Dv;
          }
        }

        // The window's final state (SF: a full-accept store).
        if (SF) {
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < R; i += 4) {
              *((device float4*)(state_final + (n * Dv + dvbase + d) * Dk + dk0 + i)) =
                  float4(state[d][i], state[d][i + 1], state[d][i + 2], state[d][i + 3]);
            }
          }
        }
        """

    static let stagedKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_replay_fused_staged",
        inputNames: [
            "q", "k", "v", "g", "beta", "T", "ps", "pk", "pv", "pa", "pb", "alog", "dtb",
            "ab_rows", "KP",
        ],
        outputNames: ["y", "state_out", "state_final"],
        source: stagedSource,
        header: stagedHeader,
        ensureRowContiguous: false)
}

/// The full-accept state store (`BONSAI_GDN_FULLACCEPT_STORE=0` keeps the
/// state skip in every round).
///
/// The verify stores no final state (`Qwen35GDNVerifyStateSkip`): a fully
/// accepted window's committed state is replayed from its tape by the next
/// verify (`Qwen35GDNReplayFused`, phase 1 over all 16 rows, then the
/// committed-state store). While the output quotes the prompt, nearly every
/// round accepts its whole window, so that replay runs in almost every round.
/// In a round whose proposal is a host lookup's continuation (ids read from
/// the prompt on the host, whether the drafter was skipped or overridden:
/// `CBv2VerifyRoundHint.proposalFromPrompt`, set by the engine around the
/// single-row verify build; a device-side splice does not count, since the
/// host cannot tell whether it took the prompt's ids), the verify stores the
/// window's final state
/// instead: the staged kernel's `SF` store after its output loop, or, from a
/// committed (not deferred) state, the staged kernel without the replay and
/// without the committed-state store (`plainScan`). That state is the prefix
/// replay stage's `finalSSM`: a fully accepted window commits it directly (no
/// replay, no committed-state store in the next verify); any other outcome
/// commits the deferred replay from the pre-verify state exactly as before,
/// and the stored state is dropped with the stage (it is its own buffer and
/// never aliases the pre-verify state the replay reads). The committed conv
/// rows of a stored full acceptance stay the window's slice of the conv input,
/// as a deferred commit keeps them (no detaching copy). Drafter rounds, the
/// seed, and every load-time trial round (their proposals are the drafter's,
/// and a running trial turns the store off) keep today's path, except that a
/// verify from a committed (not deferred) state, which ran the stock
/// output-only scan, takes the staged kernel's plain form without a store
/// (`BONSAI_GDN_PLAIN_STAGED=0` keeps the stock scan).
///
/// The stored state is the replay's, bit for bit: the verify's gates (the
/// prework's `exp(decay * softplus(a + dt_bias))` and `sigmoid(b)`, decay
/// from `Qwen35GDNDerived`) and the replay's in-kernel gates agree bit for bit
/// (the state skip's own self-test compares the stored final state of
/// `Qwen35GatedDeltaV3.run` with the full-window replay, special gate inputs
/// included). At model construction this self-test checks, on four synthetic
/// verify windows from the verify prework, that the plain scan's rows and
/// final state equal `Qwen35GatedDeltaV3.run`'s and the full-window replay's,
/// and that the fused kernel's final state after 0, 7 and 16 replayed rows
/// equals `Qwen35GatedDeltaV3.run` from its committed state; every value is
/// compared as an unsigned integer. A mismatch, or a failed state-skip or
/// fused-replay check, keeps the store off.
enum Qwen35GDNFullAcceptStore {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_FULLACCEPT_STORE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private struct Geometry: Hashable {
        let hk: Int, dk: Int, hv: Int, dv: Int, cd: Int, ks: Int, dvpl: Int
    }

    private static func geometry(_ layer: Qwen35GatedDeltaNet) -> Geometry {
        Geometry(
            hk: layer.numKHeads, dk: layer.headKDim, hv: layer.numVHeads, dv: layer.headVDim,
            cd: layer.convDim, ks: layer.convKernelSize, dvpl: Qwen35GatedDeltaV3.rowsPerLane)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [Geometry: Bool] = [:]

    /// Whether this verify of `layer` stores its window's final state.
    static func wanted(layer: Qwen35GatedDeltaNet) -> Bool {
        guard enabled, CBv2VerifyRoundHint.proposalFromPrompt,
            !Qwen35NarrowProducerTrial.active, !Qwen35TensorPackedMatmul.NarrowInSituTrial.active,
            !Qwen35HeadTopTwo.Trial.active, !DFlash2KernelTrial.active,
            !Qwen35ExactFormTrial.active, Qwen35GDNReplayFused.stagedActive,
            Qwen35GDNVerifyStateSkip.applies(to: layer), Qwen35GDNReplayFused.applies(to: layer)
        else { return false }
        let key = geometry(layer)
        return lock.withLock { verdicts[key] ?? false }
    }

    private enum SelfTestFailure: Error {
        case message(String)
    }

    /// The verdict for `layer`'s geometry at the current rows per lane.
    static func verdict(layer: Qwen35GatedDeltaNet) -> Bool? {
        let key = geometry(layer)
        return lock.withLock { verdicts[key] }
    }

    /// Once per geometry at model construction, after the state skip's and
    /// the fused replay's checks.
    static func prepare(layer: Qwen35GatedDeltaNet) {
        Qwen35ExactFormTrial.gdnLayer = layer
        guard enabled, Qwen35GDNReplayFused.stagedActive,
            Qwen35GDNVerifyStateSkip.applies(to: layer), Qwen35GDNReplayFused.applies(to: layer)
        else { return }
        let key = geometry(layer)
        lock.lock()
        defer { lock.unlock() }
        guard verdicts[key] == nil else { return }
        let (passed, detail) = selfTest(layer: layer)
        verdicts[key] = passed
        Memory.clearCache()
        FileHandle.standardError.write(
            ("qwen35 GDN full-accept state store: self-test " + (passed ? "passed" : "FAILED")
                + " (" + detail + ")"
                + (passed
                    ? "; prompt rounds store the window's final state\n"
                    : "; the state skip stays on\n")).data(using: .utf8)!)
    }

    private static func selfTest(layer: Qwen35GatedDeltaNet) -> (Bool, String) {
        let G = 4
        let S = Qwen35GDNVerifyStateSkip.selfTestRows
        let Hk = layer.numKHeads
        let Dk = layer.headKDim
        let Hv = layer.numVHeads
        let Dv = layer.headVDim
        let CD = layer.convDim
        let KS = layer.convKernelSize
        let NK = KS - 1
        guard KS == 4, Dk == 128, Dv == 128, Hv % Hk == 0, CD == 2 * Hk * Dk + Hv * Dv else {
            return (false, "geometry outside the verify prework")
        }
        let derived = Qwen35GDNDerived()
        let keys = MLXRandom.split(key: MLXRandom.key(0x4641_5354), into: 8 * G)
        var comparisons = 0
        var values = 0
        var mismatches = 0
        do {
            try withError { error in
                var differ: [MLXArray] = []
                func compare(_ a: MLXArray?, _ b: MLXArray?, _ what: String) throws {
                    guard let a, let b, a.shape == b.shape, a.dtype == .float32,
                        b.dtype == .float32
                    else { throw SelfTestFailure.message("\(what): shape or dtype mismatch") }
                    differ.append(
                        (a.view(dtype: .uint32) .!= b.view(dtype: .uint32)).asType(.int32).sum())
                    values += a.size
                    comparisons += 1
                }
                for j in 0 ..< G {
                    func key(_ i: Int) -> MLXArray { keys[8 * j + i] }
                    let qkvz = MLXRandom.normal([1, S, CD + Hv * Dv], key: key(0))
                    let qkv = qkvz[0..., 0..., ..<CD]
                    let convState = MLXRandom.normal([1, NK, CD], key: key(1))
                    let convWeight = MLXRandom.normal([CD, KS, 1], key: key(2)) * 0.5
                    var gatePair = MLXRandom.normal([1, S, 2 * Hv], key: key(3)) * 4
                    let specials: [Float] = [60, -60, 25, -25, .infinity, -.infinity, 1e-8, -1e-8]
                    let marks = MLXArray((0 ..< (2 * Hv)).map { specials[$0 % specials.count] })
                    let rowMask = MLXArray((0 ..< S).map { $0 == j % 3 ? Float(1) : 0 })
                        .reshaped([1, S, 1])
                    gatePair = MLX.where(rowMask .> 0, marks.reshaped([1, 1, 2 * Hv]), gatePair)
                    let aLog = log(MLXRandom.uniform(Float(1) ..< Float(16), [Hv], key: key(4)))
                    let dtBias = MLXRandom.normal([Hv], key: key(5))
                    let spread = exp(MLXRandom.normal([1, Hv, Dv, Dk], key: key(6)))
                    let ssmPre = MLXRandom.normal([1, Hv, Dv, Dk], key: key(7)) * spread * 0.05
                    eval(qkvz, convState, convWeight, gatePair, aLog, dtBias, ssmPre)
                    let b = gatePair[0..., 0..., ..<Hv]
                    let a = gatePair[0..., 0..., Hv...]
                    guard
                        let pre = Qwen35GDNPrework.run(
                            qkv: qkv, convState: convState, convWeight: convWeight, a: a, b: b,
                            aDecay: derived.decay(aLog), dtBias: dtBias,
                            normScales: derived.normScales(headKDim: Dk, dtype: .float32),
                            keyHeads: Hk, valueHeads: Hv, headKDim: Dk, headVDim: Dv,
                            writeConvInput: true,
                            stridedReads: Qwen35GDNPrework.verifyStridedReads),
                        let convInput = pre.convInput
                    else { throw SelfTestFailure.message("no verify prework") }
                    guard
                        let (yStored, finalStored) = Qwen35GatedDeltaV3.run(
                            q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta,
                            state: ssmPre),
                        let plain = Qwen35GDNReplayFused.plainScan(
                            pre: pre, state: ssmPre, aLog: aLog, dtBias: dtBias),
                        let plainFinal = plain.final,
                        let plainRows = Qwen35GDNReplayFused.plainScan(
                            pre: pre, state: ssmPre, aLog: aLog, dtBias: dtBias,
                            storeFinal: false)
                    else { throw SelfTestFailure.message("no plain scan") }
                    let tape = ArraysCache.PrefixReplayTape(
                        convInput: convInput, q: pre.q, k: pre.k, v: pre.v, a: a, b: b,
                        ssmPre: ssmPre, mask: nil, rowCount: S, convStateRows: NK,
                        g: pre.g, beta: pre.beta)
                    guard layer.canReplayPrefix(tape: tape, committedRows: S, fullWindow: true)
                    else { throw SelfTestFailure.message("tape rejected") }
                    let replayed = layer.replayedPrefixState(
                        tape: tape, committedRows: S, aLog: aLog, dtBias: dtBias,
                        fullWindow: true)
                    try compare(plain.y, yStored, "plain rows")
                    try compare(plainFinal, finalStored, "plain final state")
                    try compare(plainFinal, replayed.ssm, "plain final state vs replay")
                    try compare(plainRows.y, yStored, "plain rows without the store")
                    // The fused kernel with the store: this window replayed as
                    // the previous tape (0, 7, 16 rows), then its rows again.
                    for keep in [0, 7, S] {
                        guard
                            let fused = Qwen35GDNReplayFused.launch(
                                tape: tape, keep: keep, aLog: aLog, dtBias: dtBias,
                                q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta,
                                storeFinal: true),
                            let final = fused.final,
                            let reference = Qwen35GatedDeltaV3.run(
                                q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta,
                                state: fused.state)
                        else { throw SelfTestFailure.message("no fused store at \(keep) rows") }
                        try compare(final, reference.1, "fused final state at \(keep) rows")
                    }
                }
                let count = stacked(differ).sum()
                eval(count)
                try error.check()
                mismatches += Int(count.item(Int32.self))
            }
        } catch {
            return (false, "\(error)")
        }
        let passed = mismatches == 0 && comparisons == G * 7
        return (
            passed,
            "\(G) windows of \(S) rows, \(comparisons) comparisons, \(values) values, "
                + "\(mismatches) mismatches")
    }
}

// MARK: - Row-tiled fresh strided GDN prompt prework (znan2)

/// Kept beside the verify helpers so `Qwen35.swift` stays under the
/// per-file size cap; `Qwen35GDNPrework.runFreshState` calls it.
extension Qwen35GDNPrework {
    // MARK: Row-tiled fresh strided prework

    /// `BONSAI_GDN_PREWORK_ROWS=0` keeps `freshStridedKernel` (one row per
    /// threadgroup) for every prompt chunk.
    static let rowTiledEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_PREWORK_ROWS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Rows per threadgroup of `freshStridedRowsKernel`: one fixed value, not
    /// chosen per chip or at run time. A chunk whose row count it does not
    /// divide takes `freshStridedKernel`.
    static let rowTile = 4

    /// The tile the prompt prework takes when it divides the chunk and passed
    /// its check (`checkRowTile`); else `rowTile`. Set by the load-time trial
    /// (`Qwen35ExactFormTrial`) or `BONSAI_GDN_PREWORK_ROW_TILE=<n>`.
    nonisolated(unsafe) static var rowTileChoice = rowTile
    static let rowTileForced: Int? = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_PREWORK_ROW_TILE"]
            .flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        return value.flatMap { [2, 8, 16].contains($0) ? $0 : nil }
    }()
    private struct RowTileAlternative: Hashable {
        let geometry: RowTileGeometry, rows: Int
    }
    nonisolated(unsafe) private static var rowTileAlternatives: [RowTileAlternative: Bool] = [:]
    nonisolated(unsafe) private static var preparedShape: [Int]?
    nonisolated(unsafe) private static var rowTileForcedChecked = false

    /// `freshStridedSource` with one threadgroup per (key head, `RW`
    /// consecutive rows) instead of per (key head, row). The stock launch reads
    /// every chunk element from the threadgroups of four neighbouring rows and
    /// a column's conv taps from every row's threadgroup; here each column's
    /// `RW + NK` chunk rows and `KS` taps are read once per threadgroup.
    /// Thread c still owns channel c, so every simdgroup holds the same
    /// channels in the same lanes and `simd_sum` sees the same operands; row
    /// t0 + i accumulates the same taps in the same order (`fma` over
    /// j = 0..KS-1 on chunk row t0 + i + j - NK, the stock kernel's r); each
    /// row's norms reduce through the same `(r0 + r1) + (r2 + r3)` tree. Every
    /// formula (the chunk-row load, the tap load and `fma`, SiLU, the norms,
    /// the gates, the tail) is cut from `freshStridedSource` by checked spans,
    /// so the arithmetic is the stock kernel's text and follows it; only the
    /// row bookkeeping around it is new. Checked bit for bit against the stock
    /// launch at model construction (`prepare`).
    private static let freshStridedRowsSource: String = {
        let src = freshStridedSource
        func fail(_ what: String) -> Never {
            preconditionFailure(
                "Qwen35 GDN prework rows: the strided fresh source no longer matches (\(what))")
        }
        func occurrences(_ s: String, in text: String) -> Int {
            text.components(separatedBy: s).count - 1
        }
        func once(_ s: String) -> String {
            if occurrences(s, in: src) != 1 { fail(s) }
            return s
        }
        // The text of `src` from the unique `start` through the first `end` after it.
        func span(_ start: String, through end: String) -> String {
            let head = src.range(of: once(start))!
            guard let stop = src.range(of: end, range: head.upperBound ..< src.endIndex)
            else { fail(end) }
            return String(src[head.lowerBound ..< stop.upperBound])
        }
        func replacing(
            _ text: String, _ target: String, _ replacement: String, count: Int = 1
        ) -> String {
            if occurrences(target, in: text) != count { fail(target) }
            return text.replacingOccurrences(of: target, with: replacement)
        }

        // Constants, thread ids, strides; the row id becomes the tile's first
        // row and the row-dependent a/b bases move into the gates' row scope.
        let abBase = span("const int64_t ab = ", through: ";")
        let bBase = span("const int64_t bbase = ", through: ";")
        var header = span("constexpr int GRP", through: "threadgroup float red[8];")
        header = replacing(
            header, "const uint t = threadgroup_position_in_grid.y;",
            "const uint t0 = threadgroup_position_in_grid.y * uint(RW);")
        header = replacing(header, abBase, "")
        header = replacing(header, bBase, "")
        header = replacing(header, "threadgroup float red[8];", "threadgroup float red[8 * RW];")
        if header.range(of: "\\bt\\b", options: .regularExpression) != nil { fail("row id") }

        let silu = span("// MLX's silu", through: "return acc * sig;")
        let chunkLoad = span("const float xv = (r < 0)", through: ";")
        let tap = span("acc = fma(xv, ", through: ";")
        guard tap.hasSuffix(", acc);") else { fail(tap) }
        let tapWeight = String(tap.dropFirst("acc = fma(xv, ".count).dropLast(", acc);".count))
        let tapFma = replacing(tap, tapWeight, "wt[j]")

        let barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
        var normPartial = span("float sq = simd_sum(xq * xq);", through: barrier)
        normPartial = replacing(String(normPartial.dropLast(barrier.count)), "red[", "rd[", count: 2)
        let normApply = replacing(
            span("sq = (red[0]", through: "* wk[c];"), "red[", "rd[", count: 8)

        let vStore = replacing(
            once("v[vrow * size_t(DV) + c] = conv_silu(colv);"), "conv_silu(colv)", "xvs[ri]")
        let tailMarker = once("// Next convolution tail")
        let tail = String(src[src.range(of: tailMarker)!.lowerBound...])

        var text = """
            @HEADER@

            auto silu = [&](float acc) -> float {
              @SILU@
            };

            // Column `col` for rows t0 .. t0+RW-1: window element m is chunk row
            // t0 + m - NK (the stock kernel's r), row t0 + i takes tap j on
            // window element i + j.
            auto conv_silu_rows = [&](uint col, thread float* out) {
              float wt[KS];
              #pragma clang loop unroll(full)
              for (int j = 0; j < KS; j++) {
                wt[j] = @TAPWEIGHT@;
              }
              float xw[RW + NK];
              #pragma clang loop unroll(full)
              for (int m = 0; m < RW + NK; m++) {
                const int r = int(t0) + m - NK;
                @CHUNKLOAD@
                xw[m] = xv;
              }
              #pragma clang loop unroll(full)
              for (int i = 0; i < RW; i++) {
                float acc = 0.0f;
                #pragma clang loop unroll(full)
                for (int j = 0; j < KS; j++) {
                  const float xv = xw[i + j];
                  @TAPFMA@
                }
                out[i] = silu(acc);
              }
            };

            // q and k channel c of key head h, rows t0 .. t0+RW-1.
            @COLQ@
            @COLK@
            float xqs[RW];
            float xks[RW];
            conv_silu_rows(colq, xqs);
            conv_silu_rows(colk, xks);
            #pragma clang loop unroll(full)
            for (int ri = 0; ri < RW; ri++) {
              const float xq = xqs[ri];
              const float xk = xks[ri];
              threadgroup float* rd = red + 8 * ri;
              @NORMPARTIAL@
            }

            // The GRP value heads of this key head (they do not read the norms).
            #pragma clang loop unroll(full)
            for (int i = 0; i < GRP; i++) {
              @VHEAD@
              @COLV@
              float xvs[RW];
              conv_silu_rows(colv, xvs);
              #pragma clang loop unroll(full)
              for (int ri = 0; ri < RW; ri++) {
                const uint t = t0 + uint(ri);
                @VROW@
                @VSTORE@
              }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (int ri = 0; ri < RW; ri++) {
              const uint t = t0 + uint(ri);
              const float xq = xqs[ri];
              const float xk = xks[ri];
              threadgroup float* rd = red + 8 * ri;
              float sq;
              float sk;
              @NORMAPPLY@
            }

            // Gates: thread c < GRP * RW takes row t0 + c / GRP, value head c % GRP.
            if (c < uint(GRP * RW)) {
              const uint t = t0 + c / uint(GRP);
              const uint hv = h * GRP + c % uint(GRP);
              @ABBASE@
              @BBASE@
              @GROW@
              @GATES@
            }

            #pragma clang loop unroll(full)
            for (int ri = 0; ri < RW; ri++) {
              const uint t = t0 + uint(ri);
              @TAIL@
            }
            """
        for (placeholder, piece) in [
            ("@HEADER@", header),
            ("@SILU@", silu),
            ("@TAPWEIGHT@", tapWeight),
            ("@CHUNKLOAD@", chunkLoad),
            ("@TAPFMA@", tapFma),
            ("@COLQ@", span("const uint colq = ", through: ";")),
            ("@COLK@", span("const uint colk = ", through: ";")),
            ("@NORMPARTIAL@", normPartial),
            ("@VHEAD@", once("const uint hv = h * GRP + uint(i);")),
            ("@COLV@", once("const uint colv = VOFF + hv * DV + c;")),
            ("@VROW@", span("const size_t vrow = ", through: ";")),
            ("@VSTORE@", vStore),
            ("@NORMAPPLY@", normApply),
            ("@ABBASE@", abBase),
            ("@BBASE@", bBase),
            ("@GROW@", span("const size_t grow = ", through: ";")),
            ("@GATES@", span(
                "// g = exp(-exp(A_log)",
                through: "beta[grow] = (bv < 0.0f) ? by : 1.0f - by;")),
            ("@TAIL@", tail),
        ] {
            text = replacing(text, placeholder, piece)
        }
        if text.contains("@") || text.contains("conv_silu(") || text.contains("red[sg]") {
            fail("assembly")
        }
        return text
    }()

    private static let freshStridedRowsKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework_fresh_strided_rows",
        inputNames: ["qkv", "w", "a", "b", "decay", "dtb", "wq", "wk", "S"],
        outputNames: ["q", "k", "v", "g", "beta", "tail"],
        source: freshStridedRowsSource,
        ensureRowContiguous: false)

    /// `BONSAI_GDN_PREWORK_NARROW=0` keeps 64-bit offsets in the row-tiled launch.
    static let narrowOffsetsEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_PREWORK_NARROW"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `BONSAI_GDN_PREWORK_FUSED_PREP=0` keeps the chunked scan's prep launch.
    static let fusedPrepEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_PREWORK_FUSED_PREP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `BONSAI_GDN_PREWORK_SPLIT=0` keeps the value columns in the q/k launch.
    static let splitValuesEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_PREWORK_SPLIT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `freshStridedRowsSource` with 32-bit element offsets. The row-tiled
    /// launch is bound by its index arithmetic (every load and store forms
    /// 64-bit products; M4 Max, 48 back-to-back layers: 135 -> 108 us). Here
    /// the strides are read as `int` and every `int64_t` / `size_t` index is
    /// `int` / `uint`: the same elements are read and written, and no
    /// arithmetic on values changes. Taken only where every offset fits in 31
    /// bits (`narrowFits`).
    private static let narrowRowsSource: String = {
        var text = freshStridedRowsSource
        for name in ["qkv", "w", "a", "b"] {
            let target = "\(name)_strides["
            precondition(
                text.contains(target) && !text.contains("(int)" + target),
                "Qwen35 GDN prework rows: the narrow source no longer matches")
            text = text.replacingOccurrences(of: target, with: "(int)" + target)
        }
        return text.replacingOccurrences(of: "int64_t", with: "int")
            .replacingOccurrences(of: "size_t", with: "uint")
    }()

    /// The value columns' span of `narrowRowsSource` (the GRP value heads'
    /// conv, SiLU and store), which reads no norm and nothing the q/k block
    /// writes.
    private static func valueSpan(_ text: String) -> Range<String.Index> {
        let marker = "// The GRP value heads of this key head"
        guard text.components(separatedBy: marker).count == 2,
            let start = text.range(of: marker),
            let end = text.range(
                of: "threadgroup_barrier(mem_flags::mem_threadgroup);",
                range: start.upperBound ..< text.endIndex)
        else { preconditionFailure("Qwen35 GDN prework split: the row source no longer matches") }
        return start.lowerBound ..< end.lowerBound
    }

    /// `narrowRowsSource` without its value span: q, k, the gates and the tail.
    private static let splitQKSource: String = {
        var text = narrowRowsSource
        text.removeSubrange(valueSpan(text))
        precondition(!text.contains("v[vrow"))
        return text
    }()

    /// The value span alone behind the header and conv helpers (the text
    /// before the q/k block), one launch of the same grid. The q/k launch's
    /// consumer (the chunk prep) is enqueued before it, so the two overlap.
    private static let splitValueSource: String = {
        let text = narrowRowsSource
        guard let qk = text.range(of: "// q and k channel c of key head h") else {
            preconditionFailure("Qwen35 GDN prework split: the row source no longer matches")
        }
        let value = String(text[..<qk.lowerBound]) + String(text[valueSpan(text)])
        precondition(!value.contains("q[qkrow") && !value.contains("tail[") && !value.contains("g[grow"))
        return value
    }()

    /// `splitQKSource` launched one chunk per threadgroup (RW = C) with the
    /// chunked scan's prep (`Qwen35GatedDeltaChunked.prepSource`, verbatim
    /// but for the (b, key head) id and its three barriers, which only its own
    /// simdgroup's lanes cross) run by simdgroup 0 behind a device barrier:
    /// it reads the chunk's q, k, g and beta this threadgroup just stored,
    /// the values its own launch would have read. Saves the prep launch and
    /// its latency (M4 Max, 48 dependent layers: -20 us per layer).
    private static let fusedPrepSource: String = {
        var prep = Qwen35GatedDeltaChunked.prepSource
        for (target, replacement, count) in [
            ("const int bk = int(thread_position_in_grid.z);",
             "const int bk = int(bb) * Hk + int(h);", 1),
            ("threadgroup_barrier(mem_flags::mem_threadgroup);",
             "simdgroup_barrier(mem_flags::mem_threadgroup);", 3),
        ] {
            precondition(
                prep.components(separatedBy: target).count == count + 1,
                "Qwen35 GDN prework fused prep: the prep source no longer matches")
            prep = prep.replacingOccurrences(of: target, with: replacement)
        }
        precondition(!prep.contains("threadgroup_barrier"))
        return splitQKSource + """

            threadgroup_barrier(mem_flags::mem_device);
            if (simdgroup_index_in_threadgroup == 0) {
            constexpr int C = RW;
            constexpr int Dk = DK;
            constexpr int Hk = HK;
            constexpr int Hv = HV;
            const int T = Sn;
            \(prep)
            }

            """
    }()

    private static let fusedPrepKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework_fresh_rows_qk_prep",
        inputNames: ["qkv", "w", "a", "b", "decay", "dtb", "wq", "wk", "S"],
        outputNames: ["q", "k", "g", "beta", "tail", "tp", "pm", "gf"],
        source: Qwen35IO32.narrow(fusedPrepSource, count: 6, "qwen35_gdn_prework_fresh_rows_qk_prep"),
        ensureRowContiguous: false)

    private static let narrowRowsKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework_fresh_strided_rows_n",
        inputNames: ["qkv", "w", "a", "b", "decay", "dtb", "wq", "wk", "S"],
        outputNames: ["q", "k", "v", "g", "beta", "tail"],
        source: narrowRowsSource, ensureRowContiguous: false)

    private static let splitQKKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework_fresh_rows_qk",
        inputNames: ["qkv", "w", "a", "b", "decay", "dtb", "wq", "wk", "S"],
        outputNames: ["q", "k", "g", "beta", "tail"],
        source: splitQKSource, ensureRowContiguous: false)

    private static let splitValueKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework_fresh_rows_v",
        inputNames: ["qkv", "w", "S"], outputNames: ["v"],
        source: splitValueSource, ensureRowContiguous: false)

    /// The value launch alone (`freshStridedRows` forms 2 and 3): the value
    /// columns' conv and SiLU, FP32 `[B, S, HV, DV]`. The v-fold scan form's
    /// check and trial (`Qwen35GatedDeltaChunked.prepareVFold`) compare against it.
    static func valueLaunch(
        qkv: MLXArray, convWeight: MLXArray, keyHeads: Int, valueHeads: Int,
        headKDim: Int, headVDim: Int, rows: Int
    ) -> MLXArray {
        let B = qkv.dim(0)
        let S = qkv.dim(1)
        let CD = qkv.dim(2)
        let KS = convWeight.dim(1)
        return splitValueKernel(
            [qkv, convWeight, MLXArray(Int32(S))],
            template: [
                ("InT", qkv.dtype), ("HK", keyHeads), ("HV", valueHeads), ("DK", headKDim),
                ("DV", headVDim), ("CD", CD), ("KS", KS), ("RW", rows),
            ],
            grid: (128 * keyHeads, S / rows, B), threadGroup: (128, 1, 1),
            outputShapes: [[B, S, valueHeads, headVDim]], outputDTypes: [.float32])[0]
    }

    /// Every element offset below 2^31 with each input row at most 4 * CD
    /// elements apart (`qkv`, `a` and `b` are column slices of the qkv|z and
    /// b|a stacks, or of one stack of all four, each narrower; a lazy slice
    /// does not report its parent's strides before it is evaluated). Every
    /// output has at most B * S * CD elements.
    static func narrowFits(batch: Int, rows: Int, convDim: Int) -> Bool {
        batch * max(rows, 4) * 4 * convDim < Int(Int32.max)
    }

    /// The row-tiled launch on `runFreshState`'s arguments (after its guards
    /// and dtype conversions), `rows` rows per threadgroup; `rows` must divide
    /// the chunk. The outputs have `freshStridedKernel`'s shapes and dtypes.
    /// `form` 0 is the row-tiled kernel, 1 its narrow offsets, 2 those as a
    /// q/k launch and a value launch, 3 that with the chunked scan's prep in
    /// the q/k launch (`Outputs.prepared`); nil takes the verified form.
    static func freshStridedRows(
        qkv: MLXArray, convWeight: MLXArray, a: MLXArray, b: MLXArray,
        decay: MLXArray, dtb: MLXArray, normScales: (q: MLXArray, k: MLXArray),
        keyHeads: Int, valueHeads: Int, headKDim: Int, headVDim: Int, rows: Int,
        form: Int? = nil
    ) -> Outputs {
        let B = qkv.dim(0)
        let S = qkv.dim(1)
        let CD = qkv.dim(2)
        let KS = convWeight.dim(1)
        precondition(
            headKDim == 128 && rows > 0 && S % rows == 0
                && (valueHeads / keyHeads) * rows <= headKDim,
            "Qwen35 GDN prework rows: unsupported launch")
        let form =
            form
            ?? (narrowFits(batch: B, rows: S, convDim: CD)
                ? rowForm(
                    RowTileGeometry(
                        hk: keyHeads, hv: valueHeads, cd: CD, ks: KS, dtype: "\(qkv.dtype)"))
                : 0)
        if form >= 2 {
            let C = Qwen35GatedDeltaChunked.chunk
            let fused = form >= 3 && S % C == 0 && (valueHeads / keyHeads) * C <= headKDim
            let qkRows = fused ? C : rows
            let qk = (fused ? fusedPrepKernel : splitQKKernel)(
                [qkv, convWeight, a, b, decay, dtb, normScales.q, normScales.k,
                 MLXArray(Int32(S))],
                template: [
                    ("InT", qkv.dtype), ("HK", keyHeads), ("HV", valueHeads), ("DK", headKDim),
                    ("DV", headVDim), ("CD", CD), ("KS", KS), ("RW", qkRows),
                ],
                grid: (128 * keyHeads, S / qkRows, B), threadGroup: (128, 1, 1),
                outputShapes: [
                    [B, S, keyHeads, headKDim], [B, S, keyHeads, headKDim],
                    [B, S, valueHeads], [B, S, valueHeads], [B, KS - 1, CD],
                ] + (fused
                    ? [[B, valueHeads, S / C, C, C], [B, valueHeads, S / C, C, C],
                       [B, valueHeads, S / C, 2, C]] : []),
                outputDTypes: [DType](repeating: .float32, count: fused ? 8 : 5))
            let v = splitValueKernel(
                [qkv, convWeight, MLXArray(Int32(S))],
                template: [
                    ("InT", qkv.dtype), ("HK", keyHeads), ("HV", valueHeads), ("DK", headKDim),
                    ("DV", headVDim), ("CD", CD), ("KS", KS), ("RW", rows),
                ],
                grid: (128 * keyHeads, S / rows, B), threadGroup: (128, 1, 1),
                outputShapes: [[B, S, valueHeads, headVDim]], outputDTypes: [.float32])
            return Outputs(
                q: qk[0], k: qk[1], v: v[0], g: qk[2], beta: qk[3], tail: qk[4],
                prepared: fused ? Array(qk[5 ..< 8]) : nil)
        }
        let outputs = (form == 1 ? narrowRowsKernel : freshStridedRowsKernel)(
            [qkv, convWeight, a, b, decay, dtb, normScales.q, normScales.k,
             MLXArray(Int32(S))],
            template: [
                ("InT", qkv.dtype), ("HK", keyHeads), ("HV", valueHeads), ("DK", headKDim),
                ("DV", headVDim), ("CD", CD), ("KS", KS), ("RW", rows),
            ],
            grid: (128 * keyHeads, S / rows, B), threadGroup: (128, 1, 1),
            outputShapes: [
                [B, S, keyHeads, headKDim], [B, S, keyHeads, headKDim],
                [B, S, valueHeads, headVDim], [B, S, valueHeads], [B, S, valueHeads],
                [B, KS - 1, CD],
            ],
            outputDTypes: [.float32, .float32, .float32, .float32, .float32, .float32])
        return Outputs(
            q: outputs[0], k: outputs[1], v: outputs[2], g: outputs[3], beta: outputs[4],
            tail: outputs[5])
    }

    private struct RowTileGeometry: Hashable {
        let hk: Int, hv: Int, cd: Int, ks: Int, dtype: String
    }

    private static let rowTileLock = NSLock()
    nonisolated(unsafe) private static var rowTileVerdicts: [RowTileGeometry: Bool] = [:]
    /// The highest `freshStridedRows` form that matched, per geometry and dtype.
    nonisolated(unsafe) private static var rowFormVerdicts: [RowTileGeometry: Int] = [:]
    private static var rowForms: Int {
        !narrowOffsetsEnabled ? 1 : !splitValuesEnabled ? 2 : !fusedPrepEnabled ? 3 : 4
    }

    private static func rowForm(_ geometry: RowTileGeometry) -> Int {
        min(rowTileLock.withLock { rowFormVerdicts[geometry] ?? 0 }, rowForms - 1)
    }

    /// Verdict lookup only (the check runs in `prepare`, never inside a
    /// forward); a geometry or qkv dtype that was not prepared, or failed its
    /// check, keeps the stock kernel.
    static func rowTileVerified(
        keyHeads: Int, valueHeads: Int, convDim: Int, taps: Int, dtype: DType, rows S: Int
    ) -> Int? {
        guard rowTiledEnabled else { return nil }
        let geometry = RowTileGeometry(
            hk: keyHeads, hv: valueHeads, cd: convDim, ks: taps, dtype: "\(dtype)")
        lastPromptDType = dtype
        return rowTileLock.withLock { () -> Int? in
            guard rowTileVerdicts[geometry] ?? false else { return nil }
            let choice = rowTileChoice
            if choice != rowTile, S % choice == 0,
                rowTileAlternatives[RowTileAlternative(geometry: geometry, rows: choice)] ?? false
            {
                return choice
            }
            return S % rowTile == 0 ? rowTile : nil
        }
    }

    /// Checks `rows` rows per threadgroup bit for bit against the stock launch
    /// for the prepared geometry and every qkv dtype, through every launch
    /// form the record's tile passed; true when all match (then `rowTileChoice`
    /// may name it).
    static func checkRowTile(_ rows: Int) -> Bool {
        guard rows != rowTile, let shape = rowTileLock.withLock({ preparedShape }) else {
            return false
        }
        let (hk, dk, hv, dv, ks) = (shape[0], shape[1], shape[2], shape[3], shape[4])
        guard (hv / hk) * rows <= dk, 512 % rows == 0 else { return false }
        let cd = 2 * hk * dk + hv * dv
        var all = true
        for dtype in [DType.float16, .bfloat16, .float32] {
            let geometry = RowTileGeometry(hk: hk, hv: hv, cd: cd, ks: ks, dtype: "\(dtype)")
            let key = RowTileAlternative(geometry: geometry, rows: rows)
            if let known = rowTileLock.withLock({ rowTileAlternatives[key] }) {
                all = all && known
                continue
            }
            guard let record = rowTileLock.withLock({ () -> Int? in
                (rowTileVerdicts[geometry] ?? false) ? (rowFormVerdicts[geometry] ?? 0) : nil
            }) else { continue }
            let passed = rowTileSelfCheck(
                hk: hk, dk: dk, hv: hv, dv: dv, ks: ks, dtype: dtype, forms: record + 1,
                rows: rows)
            let verdict = passed == record + 1
            rowTileLock.withLock { rowTileAlternatives[key] = verdict }
            all = all && verdict
        }
        return all
    }

    /// The prompt qkv dtype last seen by `rowTileVerified`.
    nonisolated(unsafe) static var lastPromptDType: DType?

    /// Real-shape launch race of `tiles` rows per threadgroup (512 rows, the
    /// prepared geometry, the prompt qkv dtype, the verified launch form):
    /// per tile the median wall time of a burst of launches, in us. Empty on
    /// an MLX error or without a prepared geometry.
    static func raceRowTiles(_ tiles: [Int]) -> [Double] {
        guard let shape = rowTileLock.withLock({ preparedShape }) else { return [] }
        let (hk, dk, hv, dv, ks) = (shape[0], shape[1], shape[2], shape[3], shape[4])
        let T = 512
        let dtype = lastPromptDType ?? .float16
        let cd = 2 * hk * dk + hv * dv
        let width = cd + hv * dv
        let keys = MLXRandom.split(key: MLXRandom.key(0x7261_6365), into: 6)
        let stack = (MLXRandom.normal([1, T, width], key: keys[0])
            * exp(MLXRandom.normal([1, T, width], key: keys[1]))).asType(dtype)
        let qkv = stack[.ellipsis, ..<cd]
        let ba = MLXRandom.normal([1, T, 2 * hv], key: keys[2]) * 2
        let b = ba[.ellipsis, ..<hv]
        let a = ba[.ellipsis, hv...]
        let convWeight = MLXRandom.normal([cd, ks, 1], key: keys[3]) * 0.5
        let aDecay = Qwen35GDNDerived().decay(MLXRandom.normal([hv], key: keys[4]) * 0.5)
        let dtBias = MLXRandom.normal([hv], key: keys[5])
        let normScales = (q: MLXArray.ones([dk]), k: MLXArray.ones([dk]))
        eval(stack, ba, convWeight, aDecay, dtBias, normScales.q, normScales.k)
        let builders: [() -> [MLXArray]] = tiles.map { rows in
            {
                let o = freshStridedRows(
                    qkv: qkv, convWeight: convWeight, a: a, b: b, decay: aDecay, dtb: dtBias,
                    normScales: normScales, keyHeads: hk, valueHeads: hv, headKDim: dk,
                    headVDim: dv, rows: rows)
                return [o.q, o.k, o.v, o.g, o.beta, o.tail] + (o.prepared ?? [])
            }
        }
        return DFlash2LaunchTrial.race(builders, copies: 8, samples: 11)
    }

    /// Compile the row-tiled kernel for this geometry and check it bit for bit
    /// against the stock launch (`runFreshState` itself, which takes the stock
    /// kernel while no verdict exists), once per process, at model
    /// construction, for every qkv dtype `runFreshState` accepts. A mismatch
    /// prints one line and keeps the stock kernel. Called from the layer's init.
    static func prepare(hk: Int, dk: Int, hv: Int, dv: Int, ks: Int) {
        guard enabled, freshStridedReads, rowTiledEnabled, dk == 128, dv == 128, hk > 0,
            hv % hk == 0, (hv / hk) * rowTile <= dk, ks > 1
        else { return }
        let cd = 2 * hk * dk + hv * dv
        rowTileLock.withLock { preparedShape = [hk, dk, hv, dv, ks] }
        defer {
            if let forced = rowTileForced, rowTileChoice == rowTile, !rowTileForcedChecked {
                rowTileForcedChecked = true
                let passed = checkRowTile(forced)
                if passed { rowTileChoice = forced }
                FileHandle.standardError.write(
                    "qwen35 GDN prompt prework: \(forced) rows per threadgroup forced: bitwise check \(passed ? "passed, installed" : "FAILED, \(rowTile) kept")\n"
                        .data(using: .utf8)!)
            }
        }
        for dtype in [DType.float16, .bfloat16, .float32] {
            let geometry = RowTileGeometry(hk: hk, hv: hv, cd: cd, ks: ks, dtype: "\(dtype)")
            if rowTileLock.withLock({ rowTileVerdicts[geometry] != nil }) { continue }
            let passed = rowTileSelfCheck(
                hk: hk, dk: dk, hv: hv, dv: dv, ks: ks, dtype: dtype, forms: rowForms)
            let verdict = passed > 0
            let recorded = rowTileLock.withLock { () -> Bool in
                guard rowTileVerdicts[geometry] == nil else { return false }
                rowTileVerdicts[geometry] = verdict
                rowFormVerdicts[geometry] = max(passed - 1, 0)
                return true
            }
            if recorded && verdict && rowForms > 1 {
                FileHandle.standardError.write(
                    "qwen35 GDN prompt prework (\(dtype)): \(passed - 1) of \(rowForms - 1) forms (narrow offsets, split value launch, fused chunk prep) match the row-tiled kernel bit for bit\n"
                        .data(using: .utf8)!)
            }
            if recorded && !verdict {
                FileHandle.standardError.write(
                    "qwen35: GDN row-tiled prework kernel disagrees with the stock kernel on this device (\(dtype)); using the stock kernel\n"
                        .data(using: .utf8)!)
            }
        }
    }

    /// How many of the first `forms` launch forms match the stock launch bit
    /// for bit, stopping at the first that does not (0: the row-tiled kernel
    /// does not).
    private static func rowTileSelfCheck(
        hk: Int, dk: Int, hv: Int, dv: Int, ks: Int, dtype: DType, forms: Int,
        rows rowTile: Int = Qwen35GDNPrework.rowTile
    ) -> Int {
        var passed = forms
        let cd = 2 * hk * dk + hv * dv
        // qkv is a column slice of a wider stack, as the model's qkv|z product.
        let width = cd + hv * dv
        let keys = MLXRandom.split(key: MLXRandom.key(0x7277_7469), into: 8)
        for T in [64, 512] where T % rowTile == 0 {
            // A wide magnitude spread; rows 0..<ks of key head 0's q channels
            // are zero, so those rows' q norm reduces exact zeros (eps only).
            let spread = MLXRandom.normal([1, T, width], key: keys[0])
                * exp(MLXRandom.normal([1, T, width], key: keys[1]))
            let zero = (MLXArray.arange(T).reshaped(1, T, 1) .< ks)
                .&& (MLXArray.arange(width).reshaped(1, 1, width) .< dk)
            let stack = which(zero, Float(0), spread).asType(dtype)
            let qkv = stack[.ellipsis, ..<cd]
            let ba = MLXRandom.normal([1, T, 2 * hv], key: keys[2]) * 2
            let b = ba[.ellipsis, ..<hv]
            let a = ba[.ellipsis, hv...]
            let convWeight = MLXRandom.normal([cd, ks, 1], key: keys[3]) * 0.5
            // The layer's decay coefficient, formed as the layer forms it.
            let aDecay = Qwen35GDNDerived().decay(MLXRandom.normal([hv], key: keys[4]) * 0.5)
            let dtBias = MLXRandom.normal([hv], key: keys[5])
            let normScales = (
                q: MLXRandom.normal([dk], key: keys[6]), k: MLXRandom.normal([dk], key: keys[7])
            )
            guard
                let stock = runFreshState(
                    qkv: qkv, convStateShape: [1, ks - 1, cd], convWeight: convWeight,
                    a: a, b: b, aDecay: aDecay, dtBias: dtBias, normScales: normScales,
                    keyHeads: hk, valueHeads: hv, headKDim: dk, headVDim: dv)
            else { return 0 }
            for form in 0 ..< passed {
                let tiled = freshStridedRows(
                    qkv: qkv, convWeight: convWeight, a: a, b: b, decay: aDecay, dtb: dtBias,
                    normScales: normScales, keyHeads: hk, valueHeads: hv, headKDim: dk,
                    headVDim: dv, rows: rowTile, form: form)
                var pairs = [
                    (stock.q, tiled.q), (stock.k, tiled.k), (stock.v, tiled.v),
                    (stock.g, tiled.g), (stock.beta, tiled.beta), (stock.tail, tiled.tail),
                ]
                if form == 3 {
                    guard let fused = tiled.prepared, fused.count == 3 else { passed = form; break }
                    let prep = Qwen35GatedDeltaChunked.prep(
                        q: stock.q, k: stock.k, g: stock.g, beta: stock.beta)
                    pairs += zip(prep, fused).map { ($0, $1) }
                }
                var same = MLXArray(true)
                for (x, y) in pairs {
                    same = same .&& all(x.view(dtype: .uint32) .== y.view(dtype: .uint32))
                }
                if !same.item(Bool.self) {
                    passed = form
                    break
                }
            }
            if passed == 0 { return 0 }
        }
        return passed
    }
}

// MARK: - Verify-window GDN prework, every read first

/// The verify window's GDN prework (`qwen35_gdn_prework_ci_strided`) with
/// every device read issued before any store. The stock kernel stores q and
/// k, then reads the v columns' taps; stores v, then reads the gate inputs;
/// then re-reads this row's qkv columns for the conv input and, for the last
/// NK rows, again for the conv tail. Here each thread reads its q, k and v
/// columns' taps and weights, the norm scales and the gate inputs up front,
/// computes, and stores last. The conv input is written from the tap values
/// (row NK + t is tap NK; at t == 0, rows r < NK are taps r, the state's
/// rows), and the conv tail, which the verify never reads, is not stored: it
/// is returned as the conv input's last NK rows (the same rows). Every
/// formula is the stock kernel's, in its order, so the outputs are the same
/// bits; checked against the stock launch at model construction
/// (`prepareVerify`). `BONSAI_PREWORK_VERIFY_LF=0` keeps the stock kernel.
extension Qwen35GDNPrework {
    static let verifyLoadsFirstEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_PREWORK_VERIFY_LF"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // grid (128 * HK, S, B), threadgroup (128, 1, 1); inputs and template as
    // `stridedSource`; outputs q, k, v, g, beta and ci (`[B, KS - 1 + S, CD]`).
    // Not private: `Qwen35SplitKFold` derives its reads-first variant from it.
    static let verifyLoadsFirstSource = """
        constexpr int GRP = HV / HK;
        constexpr int KEY = HK * DK;
        constexpr int VOFF = 2 * KEY;
        constexpr int NK = KS - 1;
        constexpr int NC = 2 + GRP;
        const uint c = thread_position_in_threadgroup.x;
        const uint h = threadgroup_position_in_grid.x;
        const uint t = threadgroup_position_in_grid.y;
        const uint bb = threadgroup_position_in_grid.z;
        const int Sn = S;
        const int64_t qb = int64_t(bb) * qkv_strides[0];
        const int64_t qs1 = qkv_strides[1];
        const int64_t qs2 = qkv_strides[2];
        const int64_t cb = int64_t(bb) * cs_strides[0];
        const int64_t cs1 = cs_strides[1];
        const int64_t cs2 = cs_strides[2];
        threadgroup float red[8];
        uint col[NC];
        col[0] = h * DK + c;
        col[1] = KEY + h * DK + c;
        #pragma clang loop unroll(full)
        for (int i = 0; i < GRP; i++) { col[2 + i] = VOFF + (h * GRP + uint(i)) * DV + c; }

        // Reads: the KS taps and weights of the q, k and v columns, the norm
        // scales, and (c < GRP) the gate inputs.
        float xt[NC][KS];
        float wt[NC][KS];
        #pragma clang loop unroll(full)
        for (int n = 0; n < NC; n++) {
          #pragma clang loop unroll(full)
          for (int j = 0; j < KS; j++) {
            const int r = int(t) + j - NK;
            xt[n][j] = (r < 0)
                ? cs[cb + int64_t(r + NK) * cs1 + int64_t(col[n]) * cs2]
                : float(qkv[qb + int64_t(r) * qs1 + int64_t(col[n]) * qs2]);
            wt[n][j] = w[int64_t(col[n]) * w_strides[0] + int64_t(j) * w_strides[1]];
          }
        }
        const float wqc = wq[int64_t(c) * wq_strides[0]];
        const float wkc = wk[int64_t(c) * wk_strides[0]];
        const bool gate = c < uint(GRP);
        const uint hv = h * GRP + (gate ? c : 0u);
        float av = 0.0f;
        float bv = 0.0f;
        float dcy = 0.0f;
        if (gate) {
          const int64_t ab = int64_t(bb) * a_strides[0] + int64_t(t) * a_strides[1];
          const int64_t bbase = int64_t(bb) * b_strides[0] + int64_t(t) * b_strides[1];
          av = a[ab + int64_t(hv) * a_strides[2]] + dtb[int64_t(hv) * dtb_strides[0]];
          bv = b[bbase + int64_t(hv) * b_strides[2]];
          dcy = decay[int64_t(hv) * decay_strides[0]];
        }

        // Conv + SiLU of each column (the stock `conv_silu`).
        float xs[NC];
        #pragma clang loop unroll(full)
        for (int n = 0; n < NC; n++) {
          float acc = 0.0f;
          #pragma clang loop unroll(full)
          for (int j = 0; j < KS; j++) {
            acc = fma(xt[n][j], wt[n][j], acc);
          }
          const float sy = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
          const float sig = (acc < 0.0f) ? sy : 1.0f - sy;
          xs[n] = acc * sig;
        }
        const float xq = xs[0];
        const float xk = xs[1];
        float sq = simd_sum(xq * xq);
        float sk = simd_sum(xk * xk);
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        if (lane == 0) {
          red[sg] = sq;
          red[4 + sg] = sk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sq = (red[0] + red[1]) + (red[2] + red[3]);
        sk = (red[4] + red[5]) + (red[6] + red[7]);
        const float invq = metal::precise::rsqrt(sq / float(DK) + 1e-6f);
        const float invk = metal::precise::rsqrt(sk / float(DK) + 1e-6f);
        float gv = 0.0f;
        float betav = 0.0f;
        if (gate) {
          const float mx = metal::max(av, 0.0f);
          const float mn = metal::min(av, 0.0f);
          const float sp = mx + log1p(metal::exp(mn - mx));
          gv = metal::precise::exp(dcy * sp);
          const float by = 1.0f / (1.0f + metal::exp(metal::abs(bv)));
          betav = (bv < 0.0f) ? by : 1.0f - by;
        }

        // Stores.
        const size_t qkrow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HK) + size_t(h);
        q[qkrow * size_t(DK) + c] = (xq * invq) * wqc;
        k[qkrow * size_t(DK) + c] = (xk * invk) * wkc;
        #pragma clang loop unroll(full)
        for (int i = 0; i < GRP; i++) {
          const size_t vrow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HV)
              + size_t(h * GRP + uint(i));
          v[vrow * size_t(DV) + c] = xs[2 + i];
        }
        if (gate) {
          const size_t grow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HV) + size_t(hv);
          g[grow] = gv;
          beta[grow] = betav;
        }
        const size_t cirow = (size_t(bb) * size_t(Sn + NK) + size_t(NK) + size_t(t)) * size_t(CD);
        #pragma clang loop unroll(full)
        for (int n = 0; n < NC; n++) { ci[cirow + col[n]] = xt[n][NK]; }
        if (t == 0) {
          #pragma clang loop unroll(full)
          for (int r = 0; r < NK; r++) {
            const size_t cirow0 = (size_t(bb) * size_t(Sn + NK) + size_t(r)) * size_t(CD);
            #pragma clang loop unroll(full)
            for (int n = 0; n < NC; n++) { ci[cirow0 + col[n]] = xt[n][r]; }
          }
        }
        """

    private static let verifyLoadsFirstKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework_verify_lf",
        inputNames: ["qkv", "cs", "w", "a", "b", "decay", "dtb", "wq", "wk", "S"],
        outputNames: ["q", "k", "v", "g", "beta", "ci"],
        source: Qwen35IO32.narrow(verifyLoadsFirstSource, count: 32, "qwen35_gdn_prework_verify_lf"),
        ensureRowContiguous: false)

    private struct LoadsFirstGeometry: Hashable {
        let hk: Int, hv: Int, cd: Int, ks: Int, dtype: String
    }

    private static let loadsFirstLock = NSLock()
    nonisolated(unsafe) private static var loadsFirstVerdicts: [LoadsFirstGeometry: Bool] = [:]

    /// Whether `run(..., writeConvInput: true, stridedReads: true)` takes the
    /// loads-first kernel for this geometry and qkv dtype (the switch on and
    /// the verdict recorded by `prepareVerify`). `Qwen35SplitKFold` mirrors
    /// the pick so its folded launch is derived from the kernel the verify
    /// path takes.
    static func verifyLoadsFirstVerified(
        keyHeads: Int, valueHeads: Int, cd: Int, ks: Int, dtype: DType
    ) -> Bool {
        guard verifyLoadsFirstEnabled else { return false }
        let geometry = LoadsFirstGeometry(
            hk: keyHeads, hv: valueHeads, cd: cd, ks: ks, dtype: "\(dtype)")
        return loadsFirstLock.withLock { loadsFirstVerdicts[geometry] ?? false }
    }

    /// `run(..., writeConvInput: true, stridedReads: true)` by the loads-first
    /// kernel, or nil when this geometry and dtype were not verified (the
    /// check runs in `prepareVerify`, never inside a forward).
    static func verifyLoadsFirst(
        qkv: MLXArray, convState: MLXArray, convWeight: MLXArray, a: MLXArray, b: MLXArray,
        aDecay: MLXArray, dtb: MLXArray, normScales: (q: MLXArray, k: MLXArray),
        keyHeads: Int, valueHeads: Int, headKDim: Int, headVDim: Int
    ) -> Outputs? {
        guard verifyLoadsFirstEnabled else { return nil }
        let B = qkv.dim(0)
        let S = qkv.dim(1)
        let CD = qkv.dim(2)
        let KS = convWeight.dim(1)
        let geometry = LoadsFirstGeometry(
            hk: keyHeads, hv: valueHeads, cd: CD, ks: KS, dtype: "\(qkv.dtype)")
        guard loadsFirstLock.withLock({ loadsFirstVerdicts[geometry] ?? false }) else {
            return nil
        }
        let outputs = verifyLoadsFirstKernel(
            [qkv, convState, convWeight, a, b, aDecay, dtb, normScales.q, normScales.k,
             MLXArray(Int32(S))],
            template: [
                ("InT", qkv.dtype), ("HK", keyHeads), ("HV", valueHeads), ("DK", headKDim),
                ("DV", headVDim), ("CD", CD), ("KS", KS),
            ],
            grid: (128 * keyHeads, S, B), threadGroup: (128, 1, 1),
            outputShapes: [
                [B, S, keyHeads, headKDim], [B, S, keyHeads, headKDim],
                [B, S, valueHeads, headVDim], [B, S, valueHeads], [B, S, valueHeads],
                [B, KS - 1 + S, CD],
            ],
            outputDTypes: [.float32, .float32, .float32, .float32, .float32, .float32])
        return Outputs(
            q: outputs[0], k: outputs[1], v: outputs[2], g: outputs[3], beta: outputs[4],
            tail: outputs[5][0..., S..., 0...], convInput: outputs[5])
    }

    /// Check the loads-first kernel bit for bit against the stock verify
    /// launch (`run` itself, which takes the stock kernel while no verdict
    /// exists) for each qkv dtype the verify window passes, once per process,
    /// at model construction. A mismatch or MLX error keeps the stock kernel.
    static func prepareVerify(hk: Int, dk: Int, hv: Int, dv: Int, ks: Int) {
        guard enabled, verifyStridedReads, verifyLoadsFirstEnabled, dk == 128, dv == 128,
            hk > 0, hv % hk == 0, ks > 1
        else { return }
        let cd = 2 * hk * dk + hv * dv
        for dtype in [DType.float32, .float16] {
            let geometry = LoadsFirstGeometry(hk: hk, hv: hv, cd: cd, ks: ks, dtype: "\(dtype)")
            if loadsFirstLock.withLock({ loadsFirstVerdicts[geometry] != nil }) { continue }
            let (verdict, detail) = loadsFirstSelfCheck(
                hk: hk, dk: dk, hv: hv, dv: dv, ks: ks, dtype: dtype)
            let recorded = loadsFirstLock.withLock { () -> Bool in
                guard loadsFirstVerdicts[geometry] == nil else { return false }
                loadsFirstVerdicts[geometry] = verdict
                return true
            }
            if recorded {
                FileHandle.standardError.write(
                    ("qwen35 GDN verify prework (reads first, \(dtype)): self-test "
                        + (verdict ? "passed" : "FAILED") + " (" + detail + ")"
                        + (verdict ? "\n" : "; stock kernel kept\n")).data(using: .utf8)!)
            }
        }
    }

    private static func loadsFirstSelfCheck(
        hk: Int, dk: Int, hv: Int, dv: Int, ks: Int, dtype: DType
    ) -> (Bool, String) {
        let cd = 2 * hk * dk + hv * dv
        let nk = ks - 1
        // qkv a column slice of a wider stack (the qkv|z product), a and b
        // column slices of one product, the conv state a row slice.
        let width = cd + hv * dv
        let keys = MLXRandom.split(key: MLXRandom.key(0x6C66_7672), into: 10)
        let specials: [Float] = [60, -60, 25, -25, .infinity, -.infinity, 1e-8, -1e-8]
        var values = 0
        var mismatches = 0
        do {
            try withError { error in
                for T in [16, 5] {
                    let spread = MLXRandom.normal([1, T, width], key: keys[0])
                        * exp(MLXRandom.normal([1, T, width], key: keys[1]))
                    // Key head 0's q channels are zero in row 1: an eps-only norm.
                    let zero = (MLXArray.arange(T).reshaped(1, T, 1) .== 1)
                        .&& (MLXArray.arange(width).reshaped(1, 1, width) .< dk)
                    let stack = which(zero, Float(0), spread).asType(dtype)
                    let qkv = stack[.ellipsis, ..<cd]
                    let states = MLXRandom.normal([1, nk + 2, cd], key: keys[2])
                    let convState = states[0..., 1 ..< (nk + 1), 0...]
                    var ba = MLXRandom.normal([1, T, 2 * hv], key: keys[3]) * 4
                    let marks = MLXArray((0 ..< (2 * hv)).map { specials[$0 % specials.count] })
                    ba = MLX.where(
                        (MLXArray.arange(T) .== 2).reshaped([1, T, 1]),
                        marks.reshaped([1, 1, 2 * hv]), ba)
                    let b = ba[.ellipsis, ..<hv]
                    let a = ba[.ellipsis, hv...]
                    let convWeight = MLXRandom.normal([cd, ks, 1], key: keys[4]) * 0.5
                    let aDecay = Qwen35GDNDerived().decay(
                        MLXRandom.normal([hv], key: keys[5]) * 0.5)
                    let dtBias = MLXRandom.normal([hv], key: keys[6])
                    let normScales = (
                        q: MLXRandom.normal([dk], key: keys[7]),
                        k: MLXRandom.normal([dk], key: keys[8])
                    )
                    eval(qkv, convState, a, b, convWeight, aDecay, dtBias, normScales.q,
                        normScales.k)
                    guard
                        let stock = run(
                            qkv: qkv, convState: convState, convWeight: convWeight, a: a, b: b,
                            aDecay: aDecay, dtBias: dtBias, normScales: normScales,
                            keyHeads: hk, valueHeads: hv, headKDim: dk, headVDim: dv,
                            writeConvInput: true, stridedReads: true),
                        let stockCI = stock.convInput
                    else { throw SelfTestFailure.message("no stock launch") }
                    let lf = verifyLoadsFirstKernel(
                        [qkv, convState, convWeight, a, b, aDecay, dtBias, normScales.q,
                         normScales.k, MLXArray(Int32(T))],
                        template: [
                            ("InT", dtype), ("HK", hk), ("HV", hv), ("DK", dk), ("DV", dv),
                            ("CD", cd), ("KS", ks),
                        ],
                        grid: (128 * hk, T, 1), threadGroup: (128, 1, 1),
                        outputShapes: [
                            [1, T, hk, dk], [1, T, hk, dk], [1, T, hv, dv], [1, T, hv], [1, T, hv],
                            [1, nk + T, cd],
                        ],
                        outputDTypes: [.float32, .float32, .float32, .float32, .float32, .float32])
                    var differ: [MLXArray] = []
                    for (x, y) in [
                        (stock.q, lf[0]), (stock.k, lf[1]), (stock.v, lf[2]), (stock.g, lf[3]),
                        (stock.beta, lf[4]), (stockCI, lf[5]), (stock.tail, lf[5][0..., T..., 0...]),
                    ] {
                        guard x.shape == y.shape else {
                            throw SelfTestFailure.message("shape mismatch")
                        }
                        differ.append(
                            (x.view(dtype: .uint32) .!= y.view(dtype: .uint32)).asType(.int32).sum())
                        values += x.size
                    }
                    let count = stacked(differ).sum()
                    eval(count)
                    try error.check()
                    mismatches += Int(count.item(Int32.self))
                }
            }
        } catch {
            return (false, "\(error)")
        }
        return (mismatches == 0, "rows 16 and 5, \(values) values, \(mismatches) mismatches")
    }

    private enum SelfTestFailure: Error {
        case message(String)
    }
}

/// `bonsai_signed_hadamard_1024_q8` (the quantizing rotation, one 64-thread
/// threadgroup per 1024-block) at verify width with 128 or 256 threads per
/// block. Thread t reads the stock operand of block elements EPT t ... EPT t +
/// EPT - 1 (EPT = 1024 / threads, the same `inp` element, signs and head remap
/// per element); the transform applies `hadamard_n<float, 1024, 16, 4>`'s
/// butterfly levels in its order (strides below EPT in registers, the next five
/// across the simdgroup by `simd_shuffle_xor`, each pair still `a + b` /
/// `a - b` with `a` the lower element, the rest as one radix pass through
/// threadgroup memory); each 128-group is quantized by one simdgroup with the
/// stock expressions and lane layout (lane l holds elements 4l .. 4l + 3). So
/// every code, scale and scaled sum is the stock kernel's, bit for bit.
///
/// Each template form (width, presigned, head remap, code order) is compared
/// bit for bit with the stock kernel on synthetic rows (an all-zero group,
/// outliers, sign flips) before its first use; a mismatch or MLX error keeps
/// the stock kernel for that form. `BONSAI_ROTATION_Q8_BLOCKS=0` keeps it
/// everywhere.
enum Qwen35RotationQ8Blocks {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_ROTATION_Q8_BLOCKS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let header = """
        // Thread-local Hadamard butterfly for 2^R values, as in
        // mlx/backend/metal/kernels/hadamard.h (radix_func).
        template <short R>
        inline void bonsai_hadamard_radix(thread float* x) {
          constexpr short logR = __builtin_ctz(R);
          short h = 1;
          #pragma clang loop unroll(full)
          for (short s = 0; s < logR; s++) {
            #pragma clang loop unroll(full)
            for (short i = 0; i < R / 2; i++) {
              short k = i & (h - 1);
              short j = ((i - k) << 1) + k;
              float a = x[j];
              float b = x[j + h];
              x[j] = a + b;
              x[j + h] = a - b;
            }
            h <<= 1;
          }
        }

        """

    // grid (TPB * rows * BPR, 1, 1), threadgroup (TPB, 1, 1); the stock
    // kernel's template plus TPB (128 or 256).
    private static let source = """
        constexpr short N = 1024;
        constexpr uint EPT = uint(N) / uint(TPB);
        static_assert(TPB == 128 || TPB == 256, "threads per block");
        const uint blk = threadgroup_position_in_grid.x;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint row = blk / uint(BPR);
        const uint bcol = (blk % uint(BPR)) * uint(N);
        const uint rowbase = uint(row) * uint(W);
        alignas(16) threadgroup float buf[N];
        float x[EPT];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < EPT; r++) {
          const uint col = bcol + EPT * tid + r;
          uint src = col;
          if (GR > 1) {
            const uint d = col % uint(GD);
            const uint hr = col / uint(GD);
            const uint h = hr / uint(GR);
            const uint rr = hr % uint(GR);
            src = (rr * uint(GKH) + h) * uint(GD) + d;
          }
          float v = float(inp[rowbase + src]);
          if (!PRESIGNED) {
            v = v * signs[col];
          }
          x[r] = v;
        }
        bonsai_hadamard_radix<short(EPT)>(x);
        #pragma clang loop unroll(full)
        for (ushort s = 1; s < 32; s <<= 1) {
          const bool upper = (lane & s) != 0;
          #pragma clang loop unroll(full)
          for (uint r = 0; r < EPT; r++) {
            const float o = simd_shuffle_xor(x[r], s);
            x[r] = upper ? (o - x[r]) : (x[r] + o);
          }
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < EPT; r += 4) {
          *(threadgroup float4*)(buf + EPT * tid + r) = float4(x[r], x[r + 1], x[r + 2], x[r + 3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (TPB == 256) {
          // strides 128, 256, 512: radix 8 over elements e + 128 k
          if (tid < 128) {
            float y[8];
            #pragma clang loop unroll(full)
            for (short k = 0; k < 8; k++) {
              y[k] = buf[tid + 128 * k];
            }
            bonsai_hadamard_radix<8>(y);
            #pragma clang loop unroll(full)
            for (short k = 0; k < 8; k++) {
              buf[tid + 128 * k] = y[k];
            }
          }
        } else {
          // strides 256, 512: radix 4 over elements e + 256 k
          #pragma clang loop unroll(full)
          for (uint u = 0; u < 2; u++) {
            const uint e = tid + 128 * u;
            float y[4];
            #pragma clang loop unroll(full)
            for (short k = 0; k < 4; k++) {
              y[k] = buf[e + 256 * k];
            }
            bonsai_hadamard_radix<4>(y);
            #pragma clang loop unroll(full)
            for (short k = 0; k < 4; k++) {
              buf[e + 256 * k] = y[k];
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint gi = sg; gi < 8; gi += uint(TPB) / 32) {
          const short index = short(gi * 128 + 4 * lane);
          const float4 shared = *(const threadgroup float4*)(buf + index);
          float v[4];
          float amax = 0.0f;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            v[r] = shared[r] * 0.03125f;
            amax = max(amax, fabs(v[r]));
          }
          amax = simd_max(amax);
          const float qs = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
          const float iqs = amax > 0.0f ? 127.0f / amax : 0.0f;
          float part = 0.0f;
          uchar4 packed;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            const float q = rint(v[r] * iqs);
            part += q;
            packed[r] = SIGNED ? as_type<uchar>(int8_t(q)) : uint8_t(int(q) + 128);
          }
          if (PERM) {
            uint word = as_type<uint>(packed);
            uint other = simd_shuffle_xor(word, 1);
            word = (lane & 1u)
                ? ((word & 0xff00ff00u) | ((other & 0xff00ff00u) >> 8))
                : ((word & 0x00ff00ffu) | ((other & 0x00ff00ffu) << 8));
            other = simd_shuffle_xor(word, 2);
            word = (lane & 2u)
                ? ((word & 0xffff0000u) | ((other & 0xffff0000u) >> 16))
                : ((word & 0x0000ffffu) | ((other & 0x0000ffffu) << 16));
            packed = as_type<uchar4>(word);
          }
          *(device uchar4*)(out + rowbase + bcol + uint(index)) = packed;
          part = simd_sum(part);
          if (lane == 0) {
            const uint g = uint(bcol / 128) + uint(gi);
            const uint ml = row & 63u;
            const uint qidx = MPERM
              ? (uint(row >> 6) * uint(W / 128) * 64 + g * 64
                 + uint(((ml >> 4) & 1u) * 32u + (ml & 7u) * 4u + ((ml >> 5) & 1u) * 2u + ((ml >> 3) & 1u)))
              : (uint(row) * uint(W / 128) + g);
            qscale[qidx] = qs;
            qsum[qidx] = qs * part;
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_signed_hadamard_1024_q8_blocks",
        inputNames: ["inp", "signs"],
        outputNames: ["out", "qscale", "qsum"],
        source: source,
        header: header,
        ensureRowContiguous: true)

    /// Threads per block: 256 (the 17408-wide form's 272 blocks as well: its
    /// isolated launch finishes sooner than with 128, the others were already
    /// at 256). `BONSAI_ROTATION_Q8_TPB=128` keeps 128 above 128 blocks.
    static let wideThreads: Int = {
        let value = ProcessInfo.processInfo.environment["BONSAI_ROTATION_Q8_TPB"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return value == "128" ? 128 : 256
    }()
    /// `BONSAI_ROTATION_Q8_TPB_SMALL=128` takes 128 at most 128 blocks too.
    static let smallThreadsForced: Int? = {
        let value = ProcessInfo.processInfo.environment["BONSAI_ROTATION_Q8_TPB_SMALL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return value == "128" ? 128 : nil
    }()
    /// The threads per block in use (the load-time trial may change them,
    /// `Qwen35ExactFormTrial`); every form is self-tested at its first use.
    nonisolated(unsafe) static var smallThreads = smallThreadsForced ?? 256
    nonisolated(unsafe) static var wideThreadsInUse = wideThreads
    static func threads(blocks: Int) -> Int { blocks <= 128 ? smallThreads : wideThreadsInUse }

    private struct Form: Hashable {
        let width: Int, presigned: Bool, gr: Int, gkh: Int, gd: Int
        let perm: Bool, mperm: Bool, signed: Bool, dtype: String, tpb: Int
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [Form: Bool] = [:]

    /// The stock kernel's outputs for `x` from this kernel, or nil (off, not
    /// a verify-width launch, or a form that failed its self-test). `stock`
    /// runs the stock kernel with the same template (for the self-test).
    static func launch(
        _ x: MLXArray, _ signs: MLXArray, template: [(String, any KernelTemplateArg)],
        rows: Int, width: Int, groupShape: [Int], codesDType: DType,
        stock: ([MLXArray], [(String, any KernelTemplateArg)]) -> [MLXArray]
    ) -> SignedBlockHadamard.Int8Activation? {
        guard enabled, rows > 0, rows < BonsaiPromptWidth.minimumRows, width > 0,
            width % 1024 == 0, rows <= Int(Int32.max) / width
        else { return nil }
        func value(_ name: String) -> Int? {
            guard let arg = template.first(where: { $0.0 == name })?.1 else { return nil }
            if let v = arg as? Int { return v }
            if let v = arg as? Bool { return v ? 1 : 0 }
            return nil
        }
        guard let presigned = value("PRESIGNED"), let gr = value("GR"), let gkh = value("GKH"),
            let gd = value("GD"), let perm = value("PERM"), let mperm = value("MPERM"),
            let signed = value("SIGNED"), value("QSIM") == 0
        else { return nil }
        let blocks = rows * (width / 1024)
        let tpb = threads(blocks: blocks)
        let form = Form(
            width: width, presigned: presigned != 0, gr: gr, gkh: gkh, gd: gd, perm: perm != 0,
            mperm: mperm != 0, signed: signed != 0, dtype: "\(x.dtype)", tpb: tpb)
        let tmpl = template + [("TPB", tpb)]
        guard verified(form, x: x, signs: signs, template: template, tmpl: tmpl, rows: rows,
            width: width, groupShape: groupShape, codesDType: codesDType, stock: stock)
        else { return nil }
        let outs = run(x, signs, tmpl: tmpl, rows: rows, width: width, tpb: tpb,
            groupShape: groupShape, codesDType: codesDType)
        return SignedBlockHadamard.Int8Activation(codes: outs[0], scales: outs[1], scaledSums: outs[2])
    }

    private static func run(
        _ x: MLXArray, _ signs: MLXArray, tmpl: [(String, any KernelTemplateArg)], rows: Int,
        width: Int, tpb: Int, groupShape: [Int], codesDType: DType
    ) -> [MLXArray] {
        kernel(
            [x, signs], template: tmpl,
            grid: (tpb * rows * (width / 1024), 1, 1), threadGroup: (tpb, 1, 1),
            outputShapes: [x.shape, groupShape, groupShape],
            outputDTypes: [codesDType, .float32, .float32])
    }

    private enum SelfTestFailure: Error {
        case message(String)
    }

    /// Once per form: synthetic operands of `x`'s shape and dtype (per-row
    /// scales, outlier channels, an all-zero 128-group, a row of opposite
    /// signs), both kernels, every output compared as unsigned integers.
    private static func verified(
        _ form: Form, x: MLXArray, signs: MLXArray, template: [(String, any KernelTemplateArg)],
        tmpl: [(String, any KernelTemplateArg)], rows: Int, width: Int, groupShape: [Int],
        codesDType: DType, stock: ([MLXArray], [(String, any KernelTemplateArg)]) -> [MLXArray]
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if let verdict = verdicts[form] { return verdict }
        var values = 0
        var mismatches = 0
        var detail = ""
        do {
            try withError { error in
                for seed in [51, 52] {
                    let key = MLXRandom.key(UInt64(seed))
                    let scale = MLXRandom.uniform(
                        Float(0.01) ..< Float(40), [rows, 1], key: MLXRandom.split(key: key, into: 3)[0])
                    let outlier = MLXArray(
                        (0 ..< width).map { $0 % 331 == 17 ? Float(90) : Float(1) })
                    var v = MLXRandom.normal(
                        [rows, width], key: MLXRandom.split(key: key, into: 3)[1]) * scale * outlier
                    // an all-zero group in row 0 (amax 0) and a negated copy of row 0 in the last row
                    let cols = MLXArray(0 ..< width).reshaped(1, width)
                    let zeroGroup = (cols .>= MLXArray(Int32(256))) .&& (cols .< MLXArray(Int32(384)))
                    let rowIds = MLXArray(0 ..< rows).reshaped(rows, 1)
                    v = which(zeroGroup .&& (rowIds .== MLXArray(Int32(0))), MLXArray(Float(0)), v)
                    if rows > 1 {
                        v = which(rowIds .== MLXArray(Int32(rows - 1)), -v[0 ..< 1], v)
                    }
                    let operand = v.asType(x.dtype).reshaped(x.shape)
                    let a = stock([operand, signs], template)
                    let b = run(operand, signs, tmpl: tmpl, rows: rows, width: width,
                        tpb: form.tpb, groupShape: groupShape, codesDType: codesDType)
                    guard a.count == 3, b.count == 3 else {
                        throw SelfTestFailure.message("output count")
                    }
                    var differ: [MLXArray] = []
                    for (p, q) in zip(a, b) {
                        guard p.shape == q.shape, p.dtype == q.dtype else {
                            throw SelfTestFailure.message("shape or dtype mismatch")
                        }
                        let bits: DType = p.dtype == .float32 ? .uint32 : p.dtype
                        differ.append((p.view(dtype: bits) .!= q.view(dtype: bits)).asType(.int32).sum())
                        values += p.size
                    }
                    let count = stacked(differ).sum()
                    eval(count)
                    try error.check()
                    mismatches += Int(count.item(Int32.self))
                }
            }
        } catch {
            detail = " (\(error))"
            mismatches = max(mismatches, 1)
        }
        let passed = mismatches == 0
        verdicts[form] = passed
        FileHandle.standardError.write(
            ("bonsai rotation q8 per block (\(rows)x\(width), \(form.tpb) threads, presigned "
                + "\(form.presigned ? 1 : 0), heads \(form.gr)): self-test "
                + (passed ? "passed" : "FAILED") + ": \(values) values, \(mismatches) mismatches"
                + detail + (passed ? "\n" : "; stock kernel kept\n")).data(using: .utf8)!)
        return passed
    }
}

// MARK: - The drafter head's BF16 rows, padded in the read

/// The drafter's shared-head read (`HadamardQuantizedLinear.drafterHeadInt8`)
/// hands the verify route's int8 head its BF16 rows (15 at depth 15). The
/// record widens them to FP32 (a copy kernel, 154 KB read and 307 KB written)
/// and pads them with an FP32 zero row to the 16 rows the narrow body reads
/// (`concatenated`: two more copy kernels, 328 KB read and written), then
/// rotates and quantizes the 16 FP32 rows here. This form reads the BF16 rows
/// where they are and forms the zero rows in its read: the per-block launch's
/// own text with one change, a row at or past the input's row count loads
/// 0.0f instead of an element. Everything after the load (the signs, the
/// butterflies, the group absmax, the codes, scales and scaled sums) is the
/// same text. Exact: BF16 to FP32 is exact, so every real row loads the value
/// the copy wrote, and every pad row loads the +0.0f the zero row held, then
/// takes the same arithmetic. One pipeline serves 1 to 16 rows (the row count
/// is read from the input's shape, not compiled in).
///
/// At first use, per thread count, a self-test runs this launch against the
/// record's chain itself (the FP32 cast, the zero-row concatenation and
/// `SignedBlockHadamard.fusedTransformInt8`) on synthetic BF16 rows at the
/// real width with the real signs: 16, 15, 8 and 1 rows; random rows of a
/// wide scale spread with outlier channels, an all-zero group, an all-zero
/// row and a negated row; every BF16 bit pattern (NaN and infinities
/// included); every finite pattern. Codes, scales and scaled sums are compared
/// as unsigned integers, pad rows included. A mismatch or an MLX error keeps
/// the record's chain. `BONSAI_HEAD_ROT_BF16=0` keeps it too.
extension Qwen35RotationQ8Blocks {
    private static let paddedSource: String? = {
        let load = "float v = float(inp[rowbase + src]);"
        let bounded = "float v = 0.0f;\n          if (row < uint(inp_shape[0])) { v = float(inp[rowbase + src]); }"
        let text = source.replacingOccurrences(of: load, with: bounded)
        return text == source ? nil : text
    }()

    private static let paddedKernel: MLXFast.MLXFastKernel? = paddedSource.map {
        MLXFast.metalKernel(
            name: "bonsai_signed_hadamard_1024_q8_blocks_padrows",
            inputNames: ["inp", "signs"],
            outputNames: ["out", "qscale", "qsum"],
            source: $0,
            header: header,
            ensureRowContiguous: true)
    }

    private static let paddedLock = NSLock()
    nonisolated(unsafe) private static var paddedVerdicts: [String: Bool] = [:]

    /// `x` (`[rows, width]`, BF16, `rows <= paddedRows`) rotated and quantized
    /// as `paddedRows` rows, the rows past `rows` as zero rows: the outputs of
    /// `SignedBlockHadamard.fusedTransformInt8` over `x` widened to FP32 and
    /// concatenated with zero rows. Nil when off or not verified (the caller
    /// then takes that chain).
    static func launchPadded(
        _ x: MLXArray, _ signs: MLXArray, template: [(String, any KernelTemplateArg)],
        paddedRows: Int, width: Int, codesDType: DType, preSigned: Bool
    ) -> SignedBlockHadamard.Int8Activation? {
        guard enabled, let kernel = paddedKernel, x.ndim == 2, x.dtype == .bfloat16,
            x.dim(1) == width, x.dim(0) >= 1, x.dim(0) <= paddedRows,
            paddedRows < BonsaiPromptWidth.minimumRows, width % 1024 == 0,
            paddedRows <= Int(Int32.max) / width
        else { return nil }
        let tpb = threads(blocks: paddedRows * (width / 1024))
        guard paddedVerified(
            tpb: tpb, kernel: kernel, signs: signs, template: template, paddedRows: paddedRows,
            width: width, codesDType: codesDType, preSigned: preSigned)
        else { return nil }
        return runPadded(
            kernel, x, signs, tmpl: template + [("TPB", tpb)], paddedRows: paddedRows,
            width: width, tpb: tpb, codesDType: codesDType)
    }

    private static func runPadded(
        _ kernel: MLXFast.MLXFastKernel, _ x: MLXArray, _ signs: MLXArray,
        tmpl: [(String, any KernelTemplateArg)], paddedRows: Int, width: Int, tpb: Int,
        codesDType: DType
    ) -> SignedBlockHadamard.Int8Activation {
        let groupShape = [paddedRows, width / 128]
        let outs = kernel(
            [x, signs], template: tmpl,
            grid: (tpb * paddedRows * (width / 1024), 1, 1), threadGroup: (tpb, 1, 1),
            outputShapes: [[paddedRows, width], groupShape, groupShape],
            outputDTypes: [codesDType, .float32, .float32])
        return SignedBlockHadamard.Int8Activation(codes: outs[0], scales: outs[1], scaledSums: outs[2])
    }

    /// Once per process, for both thread counts (the load-time trial may
    /// switch them): this launch against the record's chain, bit for bit.
    private static func paddedVerified(
        tpb: Int, kernel: MLXFast.MLXFastKernel, signs: MLXArray,
        template: [(String, any KernelTemplateArg)], paddedRows: Int, width: Int,
        codesDType: DType, preSigned: Bool
    ) -> Bool {
        paddedLock.lock()
        defer { paddedLock.unlock() }
        let form = { (t: Int) in "\(width) \(paddedRows) \(preSigned) \(codesDType) \(t)" }
        if let verdict = paddedVerdicts[form(tpb)] { return verdict }
        var values = 0
        var mismatches: [Int: Int] = [128: 0, 256: 0]
        var detail = ""
        do {
            try withError { error in
                guard let chain = SignedBlockHadamard.fusedTransformInt8 else {
                    throw SelfTestFailure.message("no FP32 chain")
                }
                let finite: (UInt16) -> UInt16 = { ($0 & 0x7F80) == 0x7F80 ? $0 & 0xFF7F : $0 }
                for rows in [paddedRows, paddedRows - 1, 8, 1] where rows >= 1 && rows <= paddedRows {
                    let count = rows * width
                    let key = MLXRandom.key(UInt64(60 + rows))
                    let parts = MLXRandom.split(key: key, into: 2)
                    let scale = MLXRandom.uniform(Float(0.001) ..< Float(60), [rows, 1], key: parts[0])
                    let outlier = MLXArray((0 ..< width).map { $0 % 331 == 17 ? Float(90) : Float(1) })
                    var v = MLXRandom.normal([rows, width], key: parts[1]) * scale * outlier
                    let cols = MLXArray(0 ..< width).reshaped(1, width)
                    let rowIds = MLXArray(0 ..< rows).reshaped(rows, 1)
                    let zeroGroup = (cols .>= MLXArray(Int32(256))) .&& (cols .< MLXArray(Int32(384)))
                    v = which(zeroGroup .&& (rowIds .== MLXArray(Int32(0))), MLXArray(Float(0)), v)
                    if rows > 2 { v = which(rowIds .== MLXArray(Int32(1)), MLXArray(Float(0)), v) }
                    if rows > 1 { v = which(rowIds .== MLXArray(Int32(rows - 1)), -v[0 ..< 1], v) }
                    let patterns = (0 ..< count).map { UInt16(truncatingIfNeeded: $0 &* 40503) }
                    let operands = [
                        v.asType(.bfloat16),
                        MLXArray(patterns, [rows, width]).view(dtype: .bfloat16),
                        MLXArray(patterns.map(finite), [rows, width]).view(dtype: .bfloat16),
                    ]
                    for operand in operands {
                        let wide = operand.asType(.float32)
                        let padded =
                            rows < paddedRows
                            ? concatenated(
                                [wide, MLXArray.zeros([paddedRows - rows, width], dtype: .float32)],
                                axis: 0)
                            : wide
                        guard let a = chain(padded, signs, 1024, preSigned, nil, 128) else {
                            throw SelfTestFailure.message("FP32 chain declined")
                        }
                        for t in [128, 256] {
                            let b = runPadded(
                                kernel, operand, signs, tmpl: template + [("TPB", t)],
                                paddedRows: paddedRows, width: width, tpb: t,
                                codesDType: codesDType)
                            var differ: [MLXArray] = []
                            for (p, q) in zip(
                                [a.codes, a.scales, a.scaledSums], [b.codes, b.scales, b.scaledSums])
                            {
                                guard p.shape == q.shape, p.dtype == q.dtype else {
                                    throw SelfTestFailure.message("shape or dtype mismatch")
                                }
                                let bits: DType = p.dtype == .float32 ? .uint32 : .uint8
                                differ.append(
                                    (p.view(dtype: bits) .!= q.view(dtype: bits)).asType(.int32).sum())
                                values += p.size
                            }
                            let n = stacked(differ).sum()
                            eval(n)
                            try error.check()
                            mismatches[t, default: 0] += Int(n.item(Int32.self))
                        }
                    }
                }
            }
        } catch {
            detail = " (\(error))"
            mismatches[128, default: 0] += 1
            mismatches[256, default: 0] += 1
        }
        for (t, m) in mismatches { paddedVerdicts[form(t)] = m == 0 }
        let passed = paddedVerdicts[form(tpb)] ?? false
        FileHandle.standardError.write(
            ("bonsai head rotation from BF16 rows (\(width) wide, padded to \(paddedRows) in the read, "
                + "presigned \(preSigned ? 1 : 0)): self-test "
                + (passed ? "passed" : "FAILED")
                + " against the FP32 cast-and-pad chain (rows \(paddedRows)/\(paddedRows - 1)/8/1, "
                + "random, every BF16 pattern, every finite pattern; 128 and 256 threads): "
                + "\(values) values bitwise, mismatches 128: \(mismatches[128] ?? 0), "
                + "256: \(mismatches[256] ?? 0)" + detail
                + (passed ? "; \(tpb) threads in use\n" : "; FP32 cast-and-pad chain kept\n"))
                .data(using: .utf8)!)
        return passed
    }
}

// MARK: - Verify prework with the value columns in their own threadgroups

/// The folded verify-window GDN prework (`Qwen35SplitKFold`'s reads-first
/// `_bafold` launch: one 128-thread threadgroup per key head and row, each
/// thread its q, k and GRP value columns, the gate threads also summing the
/// b|a chunk partials one after another) as a launch of two threadgroup
/// kinds on one grid: per key head and row, a q/k threadgroup (the q and k
/// columns, the per-head RMS, the gates) and a value threadgroup (the key
/// head's GRP value columns: conv, SiLU, `v` and the conv input), so each
/// thread keeps fewer loads in flight and the value work no longer waits
/// behind the RMS barrier. The b|a partials of the row's GRP value heads are
/// read by all 128 threads at once into threadgroup memory before the
/// barrier (one read each, no chain), and each gate thread adds its a and b
/// sums from there in the reduce kernel's order (from 0.0f, chunk 0 first).
/// The conv weights are read as one float4 per column when they are laid out
/// `[CD, 4, 1]` with unit tap stride (else the strided scalar reads). Every
/// output is formed by the stock expressions in the stock order, so all eight
/// outputs (q, k, v, g, beta, the conv input, a, b) are the stock launch's,
/// bit for bit.
///
/// At model construction a self-test runs both launches on synthetic
/// verify-window operands (FP32 and FP16 qkv as a column slice of a wider
/// stack, 16 and 3 rows, contiguous and strided conv weights, partials of a
/// wide magnitude spread) and compares every output as unsigned integers; a
/// mismatch or an MLX error keeps the stock launch. The fold's own self-test
/// then runs through this launch too. `BONSAI_PREWORK_VSPLIT=0` keeps the
/// stock launch.
///
/// The launch takes this kernel's 32-bit twin (`fastSource`) where its own
/// self-test passed for the dtype: the same text with 32-bit element offsets,
/// every expression and order unchanged (M4, T 16, serialized: 29.3 -> 23.4
/// us). `BONSAI_GDN_PREWORK_FAST=0` keeps the 64-bit text.
enum Qwen35PreworkSplit {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_PREWORK_VSPLIT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // grid (128 * 2 * HK, S, B), threadgroup (128, 1, 1): blocks [0, HK) are
    // the q/k threadgroups of key head blk, blocks [HK, 2 HK) the value
    // threadgroups of key head blk - HK. Inputs, outputs and template as the
    // folded reads-first launch.
    private static let source = """
        constexpr int GRP = HV / HK;
        constexpr int KEY = HK * DK;
        constexpr int VOFF = 2 * KEY;
        constexpr int NK = KS - 1;
        static_assert(DK == 128 && DV == 128, "one 128-thread threadgroup per head");
        const uint c = thread_position_in_threadgroup.x;
        const uint blk = threadgroup_position_in_grid.x;
        const uint t = threadgroup_position_in_grid.y;
        const uint bb = threadgroup_position_in_grid.z;
        const int Sn = S;
        const int64_t qb = int64_t(bb) * qkv_strides[0];
        const int64_t qs1 = qkv_strides[1];
        const int64_t qs2 = qkv_strides[2];
        const int64_t cb = int64_t(bb) * cs_strides[0];
        const int64_t cs1 = cs_strides[1];
        const int64_t cs2 = cs_strides[2];
        const bool wvec = (KS == 4) && (w_strides[1] == 1) && (w_strides[0] == 4);
        const size_t cirow = (size_t(bb) * size_t(Sn + NK) + size_t(NK) + size_t(t)) * size_t(CD);

        if (blk >= uint(HK)) {
          // The value threadgroup: key head blk - HK's GRP value columns.
          const uint hv0 = (blk - uint(HK)) * uint(GRP);
          uint col[GRP];
          #pragma clang loop unroll(full)
          for (int i = 0; i < GRP; i++) { col[i] = VOFF + (hv0 + uint(i)) * DV + c; }
          float xt[GRP][KS];
          float wt[GRP][KS];
          #pragma clang loop unroll(full)
          for (int n = 0; n < GRP; n++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < KS; j++) {
              const int r = int(t) + j - NK;
              xt[n][j] = (r < 0)
                  ? cs[cb + int64_t(r + NK) * cs1 + int64_t(col[n]) * cs2]
                  : float(qkv[qb + int64_t(r) * qs1 + int64_t(col[n]) * qs2]);
            }
            if (wvec) {
              const float4 w4 = *(const device float4*)(w + int64_t(col[n]) * 4);
              wt[n][0] = w4.x; wt[n][1] = w4.y; wt[n][2] = w4.z; wt[n][3] = w4.w;
            } else {
              #pragma clang loop unroll(full)
              for (int j = 0; j < KS; j++) {
                wt[n][j] = w[int64_t(col[n]) * w_strides[0] + int64_t(j) * w_strides[1]];
              }
            }
          }
          #pragma clang loop unroll(full)
          for (int n = 0; n < GRP; n++) {
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < KS; j++) {
              acc = fma(xt[n][j], wt[n][j], acc);
            }
            const float sy = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
            const float sig = (acc < 0.0f) ? sy : 1.0f - sy;
            const size_t vrow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HV)
                + size_t(hv0 + uint(n));
            v[vrow * size_t(DV) + c] = acc * sig;
          }
          #pragma clang loop unroll(full)
          for (int n = 0; n < GRP; n++) { ci[cirow + col[n]] = xt[n][NK]; }
          if (t == 0) {
            #pragma clang loop unroll(full)
            for (int r = 0; r < NK; r++) {
              const size_t cirow0 = (size_t(bb) * size_t(Sn + NK) + size_t(r)) * size_t(CD);
              #pragma clang loop unroll(full)
              for (int n = 0; n < GRP; n++) { ci[cirow0 + col[n]] = xt[n][r]; }
            }
          }
          return;
        }

        // The q/k threadgroup of key head h: q and k columns, RMS, gates.
        const uint h = blk;
        threadgroup float red[8];
        threadgroup float tgp[2 * GRP * KSP];
        uint col[2];
        col[0] = h * DK + c;
        col[1] = KEY + h * DK + c;
        float xt[2][KS];
        float wt[2][KS];
        #pragma clang loop unroll(full)
        for (int n = 0; n < 2; n++) {
          #pragma clang loop unroll(full)
          for (int j = 0; j < KS; j++) {
            const int r = int(t) + j - NK;
            xt[n][j] = (r < 0)
                ? cs[cb + int64_t(r + NK) * cs1 + int64_t(col[n]) * cs2]
                : float(qkv[qb + int64_t(r) * qs1 + int64_t(col[n]) * qs2]);
          }
          if (wvec) {
            const float4 w4 = *(const device float4*)(w + int64_t(col[n]) * 4);
            wt[n][0] = w4.x; wt[n][1] = w4.y; wt[n][2] = w4.z; wt[n][3] = w4.w;
          } else {
            #pragma clang loop unroll(full)
            for (int j = 0; j < KS; j++) {
              wt[n][j] = w[int64_t(col[n]) * w_strides[0] + int64_t(j) * w_strides[1]];
            }
          }
        }
        const float wqc = wq[int64_t(c) * wq_strides[0]];
        const float wkc = wk[int64_t(c) * wk_strides[0]];
        const bool gate = c < uint(GRP);
        const uint hv = h * GRP + (gate ? c : 0u);
        float dtbv = 0.0f;
        float dcy = 0.0f;
        if (gate) {
          dtbv = dtb[int64_t(hv) * dtb_strides[0]];
          dcy = decay[int64_t(hv) * decay_strides[0]];
        }
        // The b|a chunk partials of this row's GRP value heads, one read per
        // element: element e is (sum e / KSP, chunk e % KSP), sums 0 .. GRP - 1
        // the a's, GRP .. 2 GRP - 1 the b's.
        {
          const int64_t prow = (int64_t(bb) * int64_t(Sn) + int64_t(t)) * abp_strides[1];
          for (uint e = c; e < uint(2 * GRP * KSP); e += 128u) {
            const uint which = e / uint(KSP);
            const uint s = e % uint(KSP);
            const int colp = (which < uint(GRP) ? AOFF : BOFF) + int(h * GRP + which % uint(GRP));
            tgp[e] = abp[int64_t(s) * abp_strides[0] + prow + int64_t(colp) * abp_strides[2]];
          }
        }
        float xs[2];
        #pragma clang loop unroll(full)
        for (int n = 0; n < 2; n++) {
          float acc = 0.0f;
          #pragma clang loop unroll(full)
          for (int j = 0; j < KS; j++) {
            acc = fma(xt[n][j], wt[n][j], acc);
          }
          const float sy = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
          const float sig = (acc < 0.0f) ? sy : 1.0f - sy;
          xs[n] = acc * sig;
        }
        const float xq = xs[0];
        const float xk = xs[1];
        float sq = simd_sum(xq * xq);
        float sk = simd_sum(xk * xk);
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        if (lane == 0) {
          red[sg] = sq;
          red[4 + sg] = sk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sq = (red[0] + red[1]) + (red[2] + red[3]);
        sk = (red[4] + red[5]) + (red[6] + red[7]);
        const float invq = metal::precise::rsqrt(sq / float(DK) + 1e-6f);
        const float invk = metal::precise::rsqrt(sk / float(DK) + 1e-6f);
        float gv = 0.0f;
        float betav = 0.0f;
        float asum = 0.0f;
        float bsum = 0.0f;
        if (gate) {
          // The reduce kernel's order: from 0.0f, chunk 0 first.
          #pragma clang loop unroll(full)
          for (int s = 0; s < KSP; s++) {
            asum += tgp[c * KSP + s];
            bsum += tgp[(GRP + c) * KSP + s];
          }
          const float av = asum + dtbv;
          const float bv = bsum;
          const float mx = metal::max(av, 0.0f);
          const float mn = metal::min(av, 0.0f);
          const float sp = mx + log1p(metal::exp(mn - mx));
          gv = metal::precise::exp(dcy * sp);
          const float by = 1.0f / (1.0f + metal::exp(metal::abs(bv)));
          betav = (bv < 0.0f) ? by : 1.0f - by;
        }
        const size_t qkrow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HK) + size_t(h);
        q[qkrow * size_t(DK) + c] = (xq * invq) * wqc;
        k[qkrow * size_t(DK) + c] = (xk * invk) * wkc;
        if (gate) {
          const size_t grow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HV) + size_t(hv);
          g[grow] = gv;
          beta[grow] = betav;
          ao[grow] = asum;
          bo[grow] = bsum;
        }
        #pragma clang loop unroll(full)
        for (int n = 0; n < 2; n++) { ci[cirow + col[n]] = xt[n][NK]; }
        if (t == 0) {
          #pragma clang loop unroll(full)
          for (int r = 0; r < NK; r++) {
            const size_t cirow0 = (size_t(bb) * size_t(Sn + NK) + size_t(r)) * size_t(CD);
            #pragma clang loop unroll(full)
            for (int n = 0; n < 2; n++) { ci[cirow0 + col[n]] = xt[n][r]; }
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework_verify_lf_bafold_vsplit",
        inputNames: ["qkv", "cs", "w", "abp", "decay", "dtb", "wq", "wk", "S"],
        outputNames: ["q", "k", "v", "g", "beta", "ci", "ao", "bo"],
        source: Qwen35IO32.narrow(source, count: 37, "qwen35_gdn_prework_verify_lf_bafold_vsplit"),
        ensureRowContiguous: false)

    static let fastEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_PREWORK_FAST"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // `source` with its 37 size_t and 32 int64_t as uint and its 17 stride
    // reads (not the float4 test's) as uint(...); nil if a count moved. It
    // runs only where every element offset is below 2^31: B and S at most 16
    // (the launch checks), KS at most 16, each stride, as unsigned (a
    // negative one fails), under its cap. Else the launch runs `source`.
    private static let fastSource: String? = {
        guard let reads = try? NSRegularExpression(pattern: #"(\w+_strides\[\d\])(?! ==)"#)
        else { return nil }
        let all = NSRange(source.startIndex..., in: source)
        guard source.components(separatedBy: "size_t").count == 38,
            source.components(separatedBy: "int64_t").count == 33,
            reads.numberOfMatches(in: source, range: all) == 17
        else { return nil }
        let narrow = reads.stringByReplacingMatches(
            in: source, range: all, withTemplate: "uint($1)"
        ).replacingOccurrences(of: "size_t", with: "uint")
            .replacingOccurrences(of: "int64_t", with: "uint")
        return """
            const bool fits32 = (KS <= 16)
                & (ulong(qkv_strides[0]) <= (1ul << 26)) & (ulong(qkv_strides[1]) <= (1ul << 26))
                & (ulong(qkv_strides[2]) <= (1ul << 26) / CD)
                & (ulong(cs_strides[0]) <= (1ul << 26)) & (ulong(cs_strides[1]) <= (1ul << 26))
                & (ulong(cs_strides[2]) <= (1ul << 26) / CD)
                & (ulong(w_strides[0]) <= (1ul << 29) / CD) & (ulong(w_strides[1]) <= (1ul << 29) / KS)
                & (ulong(abp_strides[0]) <= (1ul << 29) / KSP) & (ulong(abp_strides[1]) <= (1ul << 26))
                & (ulong(abp_strides[2]) <= (1ul << 29) / (AOFF + BOFF + HV))
                & (ulong(wq_strides[0]) <= (1ul << 30) / DK) & (ulong(wk_strides[0]) <= (1ul << 30) / DK)
                & (ulong(dtb_strides[0]) <= (1ul << 30) / HV) & (ulong(decay_strides[0]) <= (1ul << 30) / HV);
            if (fits32) {
            \(narrow)
            } else {
            \(source)
            }
            """
    }()

    private static let fastKernel: MLXFast.MLXFastKernel? = fastSource.map {
        MLXFast.metalKernel(
            name: "qwen35_gdn_prework_verify_lf_bafold_vsplit32",
            inputNames: ["qkv", "cs", "w", "abp", "decay", "dtb", "wq", "wk", "S"],
            outputNames: ["q", "k", "v", "g", "beta", "ci", "ao", "bo"],
            source: $0, ensureRowContiguous: false)
    }

    /// Fewer than 2^31 elements, checked by division.
    private static func fits32(_ shape: [Int]) -> Bool {
        var n = 1
        for d in shape {
            guard d >= 0, d == 0 || n <= Int(Int32.max) / d else { return false }
            n *= d
        }
        return true
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]
    nonisolated(unsafe) private static var fastVerdicts: [String: Bool] = [:]
    /// Set while the twin's self-test runs the 64-bit text.
    nonisolated(unsafe) private static var forceWide = false
    nonisolated(unsafe) private static var prepared: Set<String> = []
    /// Set while the self-test runs the stock launch through the fold.
    nonisolated(unsafe) private static var forceStock = false

    /// The folded reads-first launch's outputs from this kernel, or nil (off,
    /// in the self-test's stock pass, or a qkv dtype that did not pass).
    static func launch(
        _ inputs: [MLXArray], template: [(String, any KernelTemplateArg)], keyHeads: Int,
        valueHeads: Int, S: Int, B: Int, outputShapes: [[Int]], dtype: DType
    ) -> [MLXArray]? {
        guard enabled, !forceStock, keyHeads > 0, valueHeads % keyHeads == 0,
            lock.withLock({ verdicts["\(dtype)"] ?? false })
        else { return nil }
        let fast =
            !forceWide && B * S <= 16 && lock.withLock({ fastVerdicts["\(dtype)"] ?? false })
            && outputShapes.allSatisfy(fits32)
        return ((fast ? fastKernel : nil) ?? kernel)(
            inputs, template: template,
            grid: (128 * 2 * keyHeads, S, B), threadGroup: (128, 1, 1),
            outputShapes: outputShapes,
            outputDTypes: Array(repeating: DType.float32, count: outputShapes.count))
    }

    private enum SelfTestFailure: Error {
        case message(String)
    }

    /// Check this kernel against the folded reads-first launch, bit for bit,
    /// once per GDN geometry, at model construction (before the fold's own
    /// self-test, so that one runs through this kernel as well).
    static func prepare(hk: Int, dk: Int, hv: Int, dv: Int, ks: Int, hidden: Int) {
        guard enabled, Qwen35SplitKFold.enabled, dk == 128, dv == 128, hk > 0, hv % hk == 0,
            ks == 4, hidden % Qwen35SmallNMatmul.chunk == 0,
            hidden / Qwen35SmallNMatmul.chunk <= Qwen35SplitKFold.maximumChunks
        else { return }
        let geometry = "\(hk)/\(dk)/\(hv)/\(dv)/\(ks)/\(hidden)"
        let first = lock.withLock { () -> Bool in
            guard !prepared.contains(geometry) else { return false }
            prepared.insert(geometry)
            return true
        }
        guard first else { return }
        let cd = 2 * hk * dk + hv * dv
        var report: [String] = []
        for dtype in [DType.float32, .float16] {
            var values = 0
            var mismatches = 0
            var detail = ""
            // The stock launch the fold takes for this dtype (reads-first with
            // strided reads and the conv input) must exist and be verified by
            // its own check.
            guard Qwen35SplitKFold.takesLoadsFirst(
                dtype: dtype, strided: true, convInput: true, hk: hk, hv: hv, cd: cd, ks: ks)
            else {
                report.append("\(dtype): no reads-first launch")
                continue
            }
            // The check runs with this kernel enabled for the dtype.
            lock.withLock { verdicts["\(dtype)"] = true }
            do {
                try withError { error in
                    let keys = MLXRandom.split(key: MLXRandom.key(0x7673_706c), into: 12)
                    for S in [16, 3] {
                        for stridedWeight in [false, true] {
                            let n = 2 * hv
                            let x = MLXRandom.normal([1, S, hidden], key: keys[0])
                                * exp(MLXRandom.normal([1, S, hidden], key: keys[1]))
                            let wBA = MLXRandom.normal([n, hidden], key: keys[2]) * Float(0.02)
                                * exp(MLXRandom.normal([n, hidden], key: keys[3]) * Float(0.5))
                            guard let p = Qwen35SmallNMatmul.partials(x, wBA) else {
                                throw SelfTestFailure.message("no partials")
                            }
                            let width = cd + hv * dv
                            let stack = (MLXRandom.normal([1, S, width], key: keys[4])
                                * exp(MLXRandom.normal([1, S, width], key: keys[5]))).asType(dtype)
                            let qkv = stack[.ellipsis, ..<cd]
                            let convState = MLXRandom.normal([1, ks - 1, cd], key: keys[6])
                            // Contiguous [CD, 4, 1] (the float4 reads) or a
                            // view with a tap stride of 2 (the scalar reads).
                            let convWeight =
                                stridedWeight
                                ? (MLXRandom.normal([cd, 2 * ks, 1], key: keys[7]) * Float(0.5))[
                                    0..., .stride(by: 2), 0...]
                                : MLXRandom.normal([cd, ks, 1], key: keys[7]) * Float(0.5)
                            let aDecay = Qwen35GDNDerived().decay(
                                MLXRandom.normal([hv], key: keys[8]) * Float(0.5))
                            let dtBias = MLXRandom.normal([hv], key: keys[9])
                            let normScales = (
                                q: MLXRandom.normal([dk], key: keys[10]) * Float(0.3) + Float(1),
                                k: MLXRandom.normal([dk], key: keys[11]) * Float(0.3) + Float(1)
                            )
                            func run(stock: Bool) -> Qwen35GDNPrework.Outputs? {
                                forceStock = stock
                                defer { forceStock = false }
                                return Qwen35GDNPrework.runFoldedUnchecked(
                                    qkv: qkv, convState: convState, convWeight: convWeight,
                                    abPartials: p.part, aOffset: hv, bOffset: 0,
                                    aDecay: aDecay, dtBias: dtBias, normScales: normScales,
                                    keyHeads: hk, valueHeads: hv, headKDim: dk, headVDim: dv,
                                    writeConvInput: true, stridedReads: true)
                            }
                            guard let ref = run(stock: true), let new = run(stock: false),
                                let rci = ref.convInput, let nci = new.convInput,
                                let ra = ref.a, let na = new.a, let rb = ref.b, let nb = new.b
                            else { throw SelfTestFailure.message("a launch declined") }
                            var differ: [MLXArray] = []
                            for (lhs, rhs) in [
                                (ref.q, new.q), (ref.k, new.k), (ref.v, new.v), (ref.g, new.g),
                                (ref.beta, new.beta), (rci, nci), (ra, na), (rb, nb),
                            ] {
                                guard lhs.shape == rhs.shape, lhs.dtype == .float32,
                                    rhs.dtype == .float32
                                else { throw SelfTestFailure.message("shape or dtype mismatch") }
                                differ.append(
                                    (lhs.view(dtype: .uint32) .!= rhs.view(dtype: .uint32))
                                        .asType(.int32).sum())
                                values += lhs.size
                            }
                            let count = stacked(differ).sum()
                            eval(count)
                            try error.check()
                            mismatches += Int(count.item(Int32.self))
                        }
                    }
                }
            } catch {
                detail = " (\(error))"
                mismatches = max(mismatches, 1)
            }
            let passed = mismatches == 0
            lock.withLock { verdicts["\(dtype)"] = passed }
            report.append(
                "\(dtype): " + (passed ? "passed" : "FAILED") + " (\(values) values, "
                    + "\(mismatches) mismatches\(detail))")
        }
        Memory.clearCache()
        FileHandle.standardError.write(
            ("qwen35 GDN verify prework, value threadgroups apart: self-test "
                + report.joined(separator: "; ") + "\n").data(using: .utf8)!)
        fastSelfTest(hk: hk, dk: dk, hv: hv, dv: dv, ks: ks, hidden: hidden)
    }

    /// The twin against the 64-bit text, every output bit, per dtype that
    /// passed above, at 16, 9 and 3 rows on four operand forms: fresh; a tail
    /// view conv state, strided conv weights and partials sliced from a wider
    /// product; qkv with a negative column stride (the 64-bit branch); edge
    /// values (+-inf partials, no sum mixing signs; 6e4 and zero qkv blocks).
    /// A mismatch or an MLX error keeps the 64-bit text for the dtype. The
    /// fold's self-test then runs through the twin too.
    private static func fastSelfTest(hk: Int, dk: Int, hv: Int, dv: Int, ks: Int, hidden: Int) {
        guard fastEnabled, fastKernel != nil else { return }
        var report: [String] = []
        for dtype in [DType.float32, .float16] where lock.withLock({ verdicts["\(dtype)"] ?? false }) {
            var n = [0, 0, 0, 0]  // values, mismatches, inf, NaN
            var detail = ""
            lock.withLock { fastVerdicts["\(dtype)"] = true }
            do {
                try withError { error in
                    for S in [16, 9, 3] {
                        for form in 0 ..< 4 {
                            let (counts, values) = try fastCase(
                                hk: hk, dk: dk, hv: hv, dv: dv, ks: ks, hidden: hidden, S: S, form: form,
                                dtype: dtype)
                            eval(counts)
                            try error.check()
                            let c = counts.asArray(Int32.self)
                            n = [n[0] + values, n[1] + Int(c[0]), n[2] + Int(c[1]), n[3] + Int(c[2])]
                        }
                    }
                }
            } catch {
                detail = " (\(error))"
                n[1] = max(n[1], 1)
            }
            lock.withLock { fastVerdicts["\(dtype)"] = n[1] == 0 }
            report.append(
                "\(dtype): \(n[1] == 0 ? "passed" : "FAILED") (\(n[0]) values, \(n[1]) mismatches, "
                    + "\(n[2]) inf, \(n[3]) NaN\(detail))")
        }
        Memory.clearCache()
        FileHandle.standardError.write(
            ("qwen35 GDN verify prework, 32-bit offsets: self-test " + report.joined(separator: "; ")
                + "\n").data(using: .utf8)!)
    }

    /// One `fastSelfTest` case: [mismatches, inf, NaN] (of the 64-bit
    /// outputs) and the value count.
    private static func fastCase(
        hk: Int, dk: Int, hv: Int, dv: Int, ks: Int, hidden: Int, S: Int, form: Int, dtype: DType
    ) throws -> (MLXArray, Int) {
        let cd = 2 * hk * dk + hv * dv
        let width = cd + hv * dv
        let nab = 2 * hv
        let keys = MLXRandom.split(key: MLXRandom.key(0x7077_3332), into: 8)
        func normal(_ shape: [Int], _ i: Int) -> MLXArray { MLXRandom.normal(shape, key: keys[i]) }
        let x = normal([1, S, hidden], 0)
        guard var abp = Qwen35SmallNMatmul.partials(x, normal([nab, hidden], 1) * Float(0.02))?.part
        else { throw SelfTestFailure.message("no partials") }
        var stack = normal([1, S, width], 2) * exp(normal([1, S, width], 3))
        if form == 1 { abp = concatenated([abp, abp], axis: 2)[0..., 0..., ..<nab] }
        if form == 3 {
            let flat = MLXArray(0 ..< abp.size).reshaped(abp.shape)
            for (f, v) in [
                (hv, Float.infinity), (5 * S * nab + nab + hv + 1, -Float.infinity),
                (2 * S * nab + 2 * nab + 2, Float.infinity), (7 * S * nab + 3, -Float.infinity),
            ] {
                abp = which(flat .== MLXArray(f), MLXArray(v), abp)
            }
            let col = MLXArray(0 ..< width).reshaped([1, 1, width])
            func band(_ lo: Int, _ n: Int) -> MLXArray { (col .>= MLXArray(lo)) .&& (col .< MLXArray(lo + n)) }
            stack = which(
                band(dk, dk) .|| band(hk * dk + dk, dk), MLXArray(Float(0)),
                which(band(2 * hk * dk, dv) .|| band(2 * dk, dk), sign(stack) * Float(6e4), stack))
        }
        let typed = stack.asType(dtype)
        let norms = normal([2, dk], 7) * Float(0.3) + Float(1)
        let inputs = [
            (form == 2 ? typed[0..., 0..., .stride(by: -1)] : typed)[.ellipsis, ..<cd],
            form == 1 ? normal([1, S + ks - 1, cd], 4)[0..., S..., 0...] : normal([1, ks - 1, cd], 4),
            form == 1
                ? (normal([cd, 2 * ks, 1], 5) * Float(0.5))[0..., .stride(by: 2), 0...]
                : normal([cd, ks, 1], 5) * Float(0.5),
            abp, Qwen35GDNDerived().decay(normal([hv], 6) * Float(0.5)), normal([hv], 7), norms[0], norms[1],
            MLXArray(Int32(S)),
        ]
        let template: [(String, any KernelTemplateArg)] = [
            ("InT", dtype), ("HK", hk), ("HV", hv), ("DK", dk), ("DV", dv), ("CD", cd), ("KS", ks),
            ("KSP", abp.dim(0)), ("AOFF", hv), ("BOFF", 0),
        ]
        let shapes = [
            [1, S, hk, dk], [1, S, hk, dk], [1, S, hv, dv], [1, S, hv], [1, S, hv], [1, ks - 1 + S, cd],
            [1, S, hv], [1, S, hv],
        ]
        func run(wide: Bool) -> [MLXArray]? {
            forceWide = wide
            defer { forceWide = false }
            return launch(
                inputs, template: template, keyHeads: hk, valueHeads: hv, S: S, B: 1, outputShapes: shapes,
                dtype: dtype)
        }
        guard let ref = run(wide: true), let new = run(wide: false) else {
            throw SelfTestFailure.message("a launch declined")
        }
        let counts = [
            zip(ref, new).map { ($0.view(dtype: .uint32) .!= $1.view(dtype: .uint32)).asType(.int32).sum() },
            ref.map { isInf($0).asType(.int32).sum() }, ref.map { isNaN($0).asType(.int32).sum() },
        ]
        return (stacked(counts.map { stacked($0).sum() }), ref.reduce(0) { $0 + $1.size })
    }
}

/// The fused verify boundary (`Qwen35FusedBoundaryQ8`) at verify width with one
/// threadgroup per (row, 1024-block) instead of one per row: 80 threadgroups of
/// 256 threads for a 16-row window instead of 16 of 1024 (a 40-core GPU leaves
/// 24 cores idle under the stock grid). Each threadgroup recomputes its row's
/// sum of squares with `rms_looped`'s lane structure: the 1024 lanes are
/// virtual, lane vl = tid + 256 m accumulates p = 0, 1 and i = 0 ... 3 in the
/// stock order, a real `simd_sum` over each m-accumulator is that virtual
/// simdgroup's (same 32 lane positions), and every simdgroup takes the second
/// `simd_sum` of the 32 partials in order and `precise::rsqrt` itself (one
/// barrier, as the one-barrier boundary does). It normalizes, rotates
/// and quantizes only its own block: `w * (h * inv)` and the signs on the
/// thread's four elements (gain and signs read before the reduction), the
/// transform's butterfly levels in the stock order (strides 1 and 2 in
/// registers, 4 ... 64 across the simdgroup by `simd_shuffle_xor`, each pair
/// still `a + b` / `a - b` with `a` the lower element, 128 ... 512 as a radix-8
/// in threadgroup memory), and the stock quantization per 128-group (lane l
/// holds elements 4l .. 4l + 3). Every output is bit-identical.
///
/// The fused boundary's 16-row self-test runs through this kernel (bitwise
/// against the composed ops, the reference the stock kernel is verified
/// against); on a mismatch the stock kernel is retested and kept.
/// `BONSAI_BOUNDARY_Q8_BLOCKS=0` keeps the stock kernel.
enum Qwen35BoundaryBlocks {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_BOUNDARY_Q8_BLOCKS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Cleared when the 16-row self-test fails through this kernel.
    nonisolated(unsafe) static var live = true
    /// Set when the per-row kernel is in use (the load-time trial's
    /// alternative, `Qwen35ExactFormTrial`); `stockVerified` is its own 16-row
    /// self-test through the same test.
    nonisolated(unsafe) static var stock = false
    nonisolated(unsafe) static var stockVerified: Bool?
    static var active: Bool { enabled && live && !stock }

    private static let threads = 256

    private static let header = """
        #define BONSAI_UNROLL _Pragma("clang loop unroll(full)")

        // Thread-local Hadamard butterfly for 2^R values, as in
        // mlx/backend/metal/kernels/hadamard.h (radix_func).
        template <short R>
        inline void bonsai_hadamard_radix(thread float* x) {
          constexpr short logR = __builtin_ctz(R);
          short h = 1;
          BONSAI_UNROLL for (short s = 0; s < logR; s++) {
            BONSAI_UNROLL for (short i = 0; i < R / 2; i++) {
              short k = i & (h - 1);
              short j = ((i - k) << 1) + k;
              float a = x[j];
              float b = x[j + h];
              x[j] = a + b;
              x[j + h] = a - b;
            }
            h <<= 1;
          }
        }

        """

    // grid (256 * rows * W / 1024, 1, 1), threadgroup (256, 1, 1). Inputs and
    // outputs as `Qwen35FusedBoundaryQ8`'s kernel.
    private static let source = """
        constexpr uint NR = 4;
        constexpr uint LS = 1024;
        constexpr uint BT = 256;
        constexpr uint NP = (uint(W) + LS * NR - 1) / (LS * NR);
        constexpr uint NB = uint(W) / 1024;
        constexpr uint NG = uint(W) / 128;
        constexpr uint VM = LS / BT;
        static_assert(W % 1024 == 0 && W > 4096 && W <= 7168, "rms_looped width with 1024-blocks");
        const uint tid = thread_position_in_threadgroup.x;
        const uint row = threadgroup_position_in_grid.x / NB;
        const uint blk = threadgroup_position_in_grid.x % NB;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint base = uint(row) * uint(W);

        alignas(16) threadgroup float buf[1024];
        threadgroup float local_sums[32];

        // Gain and signs of this thread's four block elements, read early.
        const uint el = 4 * tid;
        const uint ec = blk * 1024 + el;
        const float4 wv = *(const device float4*)(w + ec);
        float4 sv = float4(1.0f);
        if (!PRESIGNED) {
          sv = *(const device float4*)(signs + ec);
        }

        // The FP16 residual add and rms_looped's per-lane sum of squares for
        // virtual lanes vl = tid + BT * m; this block's four elements are kept.
        float acc[VM];
        float hb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        BONSAI_UNROLL for (uint m = 0; m < VM; m++) {
          acc[m] = 0;
        }
        BONSAI_UNROLL for (uint m = 0; m < VM; m++) {
          const uint vl = tid + BT * m;
          BONSAI_UNROLL for (uint p = 0; p < NP; p++) {
            const uint r0 = p * LS * NR;
            if (r0 + vl * NR + NR <= uint(W)) {
              const uint e0 = r0 + vl * NR;
              const half4 hs = *(const device half4*)(xa + base + e0)
                  + *(const device half4*)(xb + base + e0);
              const bool mine = (e0 / 1024) == blk;
              if (mine) {
                *(device half4*)(hout + base + e0) = hs;
              }
              BONSAI_UNROLL for (uint i = 0; i < NR; i++) {
                const float hvf = float(hs[i]);
                acc[m] += hvf * hvf;
                if (mine) {
                  hb[i] = hvf;
                }
              }
            }
          }
        }
        BONSAI_UNROLL for (uint m = 0; m < VM; m++) {
          const float s = simd_sum(acc[m]);
          if (lane == 0) {
            local_sums[sg + (BT / 32) * m] = s;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Every simdgroup forms the same simd_sum of the same 32 partials in
        // the same lanes and the same rsqrt (the one-barrier reduction).
        const float t = simd_sum(local_sums[lane]);
        const float inv = metal::precise::rsqrt(t / axis_size + eps);

        // rms_looped's output `w * (x * inv)`, then the signs.
        float x[4];
        {
          const float n0 = wv[0] * (hb[0] * inv);
          const float n1 = wv[1] * (hb[1] * inv);
          const float n2 = wv[2] * (hb[2] * inv);
          const float n3 = wv[3] * (hb[3] * inv);
          x[0] = n0;
          x[1] = n1;
          x[2] = n2;
          x[3] = n3;
          if (!PRESIGNED) {
            x[0] = n0 * sv[0];
            x[1] = n1 * sv[1];
            x[2] = n2 * sv[2];
            x[3] = n3 * sv[3];
          }
          BONSAI_STORE_NORMED(ec + 0, n0);
          BONSAI_STORE_NORMED(ec + 1, n1);
          BONSAI_STORE_NORMED(ec + 2, n2);
          BONSAI_STORE_NORMED(ec + 3, n3);
        }

        // hadamard_n<float, 1024, 16, 4>'s levels in its order: strides 1, 2
        // in registers, 4 ... 64 across the simdgroup, 128 ... 512 in memory.
        bonsai_hadamard_radix<4>(x);
        BONSAI_UNROLL for (ushort s = 1; s < 32; s <<= 1) {
          const bool upper = (lane & s) != 0;
          BONSAI_UNROLL for (short r = 0; r < 4; r++) {
            const float o = simd_shuffle_xor(x[r], s);
            x[r] = upper ? (o - x[r]) : (x[r] + o);
          }
        }
        *(threadgroup float4*)(buf + el) = float4(x[0], x[1], x[2], x[3]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 128) {
          float y8[8];
          BONSAI_UNROLL for (short k = 0; k < 8; k++) {
            y8[k] = buf[tid + 128 * k];
          }
          bonsai_hadamard_radix<8>(y8);
          BONSAI_UNROLL for (short k = 0; k < 8; k++) {
            buf[tid + 128 * k] = y8[k];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The quantizing rotation's tail, one 128-group per simdgroup.
        {
          const uint g = blk * 8 + sg;
          const uint g0 = sg * 128 + lane * 4;
          const float4 shared = *(const threadgroup float4*)(buf + g0);
          float v[4];
          float amax = 0.0f;
          BONSAI_UNROLL for (short r = 0; r < 4; r++) {
            v[r] = shared[r] * 0.03125f;
            amax = max(amax, fabs(v[r]));
          }
          amax = simd_max(amax);
          const float qs = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
          const float iqs = amax > 0.0f ? 127.0f / amax : 0.0f;
          float part = 0.0f;
          uchar4 packed;
          BONSAI_UNROLL for (short r = 0; r < 4; r++) {
            const float q = rint(v[r] * iqs);
            part += q;
            packed[r] = SIGNED ? as_type<uchar>(int8_t(q)) : uint8_t(int(q) + 128);
          }
          if (PERM) {
            uint word = as_type<uint>(packed);
            uint other = simd_shuffle_xor(word, 1);
            word = (lane & 1u)
                ? ((word & 0xff00ff00u) | ((other & 0xff00ff00u) >> 8))
                : ((word & 0x00ff00ffu) | ((other & 0x00ff00ffu) << 8));
            other = simd_shuffle_xor(word, 2);
            word = (lane & 2u)
                ? ((word & 0xffff0000u) | ((other & 0xffff0000u) >> 16))
                : ((word & 0x0000ffffu) | ((other & 0x0000ffffu) << 16));
            packed = as_type<uchar4>(word);
          }
          *(device uchar4*)(codes + base + uint(g) * 128 + lane * 4) = packed;
          part = simd_sum(part);
          if (lane == 0) {
            const uint ml = row & 63u;
            const uint qidx = MPERM
              ? (uint(row >> 6) * uint(NG) * 64 + uint(g) * 64
                 + uint(((ml >> 4) & 1u) * 32u + (ml & 7u) * 4u + ((ml >> 5) & 1u) * 2u + ((ml >> 3) & 1u)))
              : (uint(row) * NG + g);
            qscale[qidx] = qs;
            qsum[qidx] = qs * part;
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_boundary_q8_blocks",
        inputNames: ["xa", "xb", "w", "signs", "eps", "axis_size"],
        outputNames: ["hout", "codes", "qscale", "qsum"],
        source: "#define BONSAI_STORE_NORMED(e, n)\n" + source,
        header: header,
        ensureRowContiguous: true)

    private static let kernelNormed = MLXFast.metalKernel(
        name: "bonsai_boundary_q8_blocks_normed",
        inputNames: ["xa", "xb", "w", "signs", "eps", "axis_size"],
        outputNames: ["hout", "codes", "qscale", "qsum", "nout"],
        source: "#define BONSAI_STORE_NORMED(e, n) nout[base + (e)] = (n)\n" + source,
        header: header,
        ensureRowContiguous: true)

    nonisolated(unsafe) private static let axisSize = MLXArray(UInt32(Qwen35FusedBoundaryQ8.width))

    /// `Qwen35FusedBoundaryQ8`'s outputs for a verify-width window (fewer than
    /// `BonsaiPromptWidth.minimumRows` rows), or nil when this kernel is off.
    static func launch(
        _ x: MLXArray, _ r: MLXArray, gain: MLXArray, signs: MLXArray, eps: Float,
        gainSigned: Bool, writeNormed: Bool, perm: Bool
    ) -> Qwen35FusedBoundaryQ8.Output? {
        let width = Qwen35FusedBoundaryQ8.width
        guard active, width % 1024 == 0, width > 4096, width <= 7168, x.size % width == 0,
            x.size <= Int(Int32.max)
        else { return nil }
        let rows = x.size / width
        guard rows > 0, rows < BonsaiPromptWidth.minimumRows else { return nil }
        let codesShape = [rows, width]
        let groupShape = [rows, width / 128]
        let template: [(String, any KernelTemplateArg)] = [
            ("W", width), ("PRESIGNED", gainSigned), ("PERM", perm),
            ("MPERM", Qwen35TensorPackedMatmul.rowTiledConstants && rows % 64 == 0),
            ("SIGNED", Qwen35TensorPackedMatmul.signedCodes),
        ]
        let inputs = [x, r, gain, signs, MLXArray(eps), axisSize]
        let grid = (threads * rows * (width / 1024), 1, 1)
        if writeNormed {
            let outs = kernelNormed(
                inputs, template: template, grid: grid, threadGroup: (threads, 1, 1),
                outputShapes: [x.shape, codesShape, groupShape, groupShape, x.shape],
                outputDTypes: [
                    .float16, Qwen35TensorPackedMatmul.codesDType, .float32, .float32, .float32,
                ])
            return Qwen35FusedBoundaryQ8.Output(
                h: outs[0], normed: outs[4],
                activation: SignedBlockHadamard.Int8Activation(
                    codes: outs[1], scales: outs[2], scaledSums: outs[3]))
        }
        let outs = kernel(
            inputs, template: template, grid: grid, threadGroup: (threads, 1, 1),
            outputShapes: [x.shape, codesShape, groupShape, groupShape],
            outputDTypes: [.float16, Qwen35TensorPackedMatmul.codesDType, .float32, .float32])
        return Qwen35FusedBoundaryQ8.Output(
            h: outs[0], normed: nil,
            activation: SignedBlockHadamard.Int8Activation(
                codes: outs[1], scales: outs[2], scaledSums: outs[3]))
    }
}

/// The producer form of the quantizing rotation (`bonsai_signed_hadamard_1024_q8p`
/// / its vector-read twin `_q8pv`: SwiGLU, the attention output gate or the
/// GDN gated norm formed in the rotation's read) at verify width, with the
/// per-block layout of `Qwen35RotationQ8Blocks`: 256 threads per 1024-block
/// (at most 128 blocks) or 128. Thread t reads its four (or eight) block
/// elements as the vector-read form reads four consecutive columns (the same
/// source columns, operand strides and FP32 producer expressions, times the
/// signs); for the gated norm (256 threads) simdgroup s holds head s of the
/// block with lane l on its elements 4l .. 4l + 3, the stock RMS pass's lanes
/// and elements, so the head's sum of squares (r = 0 .. 3), `simd_sum` and
/// `precise::rsqrt` are the same operations on the same values. Then the
/// butterflies and the quantization of `Qwen35RotationQ8Blocks`, so every code,
/// scale and scaled sum is the stock producer rotation's, bit for bit.
///
/// Each template form is compared bit for bit with the stock producer kernel
/// at its first use (outside any timed round: the verify graph's first build),
/// on that call's operands and on two synthetic operand sets with the same
/// shapes, strides and dtypes (wide magnitude spreads, a zero stretch); a
/// mismatch or an MLX error keeps the stock kernel for that form.
/// `BONSAI_ROTATION_Q8P_BLOCKS=0` keeps the stock kernel everywhere.
extension Qwen35RotationQ8Blocks {
    static let producerEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_ROTATION_Q8P_BLOCKS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The gated-norm producer (PROD 3: the GDN output into the quantizing
    /// rotation of its output projection) takes this kernel at prompt width
    /// too: 256 threads per 1024-block, each lane's four elements consecutive,
    /// the butterflies up to stride 64 by `simd_shuffle_xor` (no threadgroup
    /// passes), the per-head sum of squares from the same lanes and elements
    /// as the stock kernel's. Measured on the M4 (6144 wide): -8% on the
    /// kernel alone with cold operands, -3 us per call (about -4%) in the
    /// one-op census of the prompt forward; the SwiGLU and attention-gate
    /// producers stay on the stock kernel there (the SwiGLU one read 19%
    /// slower).
    /// Self-tested at the prompt-width form (its own `ProducerForm`, MPERM on)
    /// against the stock kernel before first use; a failure keeps the stock
    /// kernel. `BONSAI_ROTATION_Q8P_BLOCKS_PROMPT=0` keeps the stock kernel at
    /// prompt width.
    static let producerPromptGatedNorm: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_ROTATION_Q8P_BLOCKS_PROMPT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Set while a form's self-test runs: the stock kernel it compares with
    /// can run its own first-use self-test through the production closure,
    /// which must not come back here (the lock is held).
    nonisolated(unsafe) private static var producerVerifying = false

    private static let producerHeader = """
        // MLX `Sigmoid` (unary_ops.h), verbatim.
        METAL_FUNC float bonsai_sigmoid(float x) {
          auto y = 1 / (1 + metal::exp(metal::abs(x)));
          return (x < 0) ? y : 1 - y;
        }

        // Thread-local Hadamard butterfly for 2^R values, as in
        // mlx/backend/metal/kernels/hadamard.h (radix_func).
        template <short R>
        inline void bonsai_hadamard_radix(thread float* x) {
          constexpr short logR = __builtin_ctz(R);
          short h = 1;
          #pragma clang loop unroll(full)
          for (short s = 0; s < logR; s++) {
            #pragma clang loop unroll(full)
            for (short i = 0; i < R / 2; i++) {
              short k = i & (h - 1);
              short j = ((i - k) << 1) + k;
              float a = x[j];
              float b = x[j + h];
              x[j] = a + b;
              x[j + h] = a - b;
            }
            h <<= 1;
          }
        }

        // The producer rotation's operand addressing and four-wide reads,
        // verbatim (`Qwen35FusedHadamard` headerProducer / producerVecHeader).
        template <int HD>
        inline int64_t bonsai_q8p_row(
            const constant int* shape, const constant int64_t* st, uint row) {
          if (HD == 0) {
            return int64_t(row) * st[0];
          }
          const uint L = uint(shape[1]);
          return int64_t(row / L) * st[0] + int64_t(row % L) * st[1];
        }
        template <int HD>
        inline int64_t bonsai_q8p_col(const constant int64_t* st, uint c) {
          if (HD == 0) {
            return int64_t(c) * st[1];
          }
          return int64_t(c / uint(HD)) * st[2] + int64_t(c % uint(HD)) * st[3];
        }
        inline float4 bonsai_ld4(const device float* p) {
          return *(const device float4*)p;
        }
        inline float4 bonsai_ld4(const device half* p) {
          return float4(*(const device half4*)p);
        }
        template <typename T>
        inline float4 bonsai_ld4(const device T* p) {
          return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
        }
        // Same row and column formulas, in 32-bit. The host sets FIT32 only
        // when every stride and every corner offset fits in int, so the casts
        // do not truncate. Pointer + int is the 32-bit offset add; the loaded
        // elements are the 64-bit path's elements. Prompt width keeps int64.
        template <int HD>
        inline int bonsai_q8p_row32(
            const constant int* shape, const constant int64_t* st, uint row) {
          if (HD == 0) {
            return int(row) * int(st[0]);
          }
          const uint L = uint(shape[1]);
          return int(row / L) * int(st[0]) + int(row % L) * int(st[1]);
        }
        template <int HD>
        inline int bonsai_q8p_col32(const constant int64_t* st, uint c) {
          if (HD == 0) {
            return int(c) * int(st[1]);
          }
          return int(c / uint(HD)) * int(st[2]) + int(c % uint(HD)) * int(st[3]);
        }
        template <int HD, typename T>
        inline float4 bonsai_q8p_ld4_32(
            const device T* p, int rowoff, const constant int64_t* st, uint c, bool vec) {
          if (vec) {
            return bonsai_ld4(p + (rowoff + bonsai_q8p_col32<HD>(st, c)));
          }
          float4 v;
          #pragma clang loop unroll(full)
          for (int r = 0; r < 4; r++) {
            v[r] = float(p[rowoff + bonsai_q8p_col32<HD>(st, c + uint(r))]);
          }
          return v;
        }
        template <int HD, typename T>
        inline float4 bonsai_q8p_ld4(
            const device T* p, int64_t rowoff, const constant int64_t* st, uint c, bool vec) {
          if (vec) {
            return bonsai_ld4(p + rowoff + bonsai_q8p_col<HD>(st, c));
          }
          float4 v;
          #pragma clang loop unroll(full)
          for (int r = 0; r < 4; r++) {
            v[r] = float(p[rowoff + bonsai_q8p_col<HD>(st, c + uint(r))]);
          }
          return v;
        }

        template <int FIT, int HD, typename T>
        inline float4 bonsai_q8p_load(
            const device T* p, const constant int* shape, const constant int64_t* st,
            uint row, uint c, bool vec) {
          if (FIT) {
            return bonsai_q8p_ld4_32<HD>(p, bonsai_q8p_row32<HD>(shape, st, row), st, c, vec);
          }
          return bonsai_q8p_ld4<HD>(p, bonsai_q8p_row<HD>(shape, st, row), st, c, vec);
        }

        """

    // grid (TPB * rows * BPR, 1, 1), threadgroup (TPB, 1, 1); the stock
    // producer kernel's template plus TPB.
    private static let producerSource = """
        constexpr short N = 1024;
        constexpr uint EPT = uint(N) / uint(TPB);
        static_assert(TPB == 128 || TPB == 256, "threads per block");
        static_assert(GR == 1 || GD % 4 == 0, "four-column runs share a head");
        static_assert(PROD != 3 || (GD == 128 && TPB == 256), "one head per simdgroup");
        const uint blk = threadgroup_position_in_grid.x;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint row = blk / uint(BPR);
        const uint bcol = (blk % uint(BPR)) * uint(N);
        const uint rowbase = uint(row) * uint(W);
        const bool AV = (AHD == 0 ? a_strides[1] : a_strides[3]) == 1;
        const bool BV = (BHD == 0 ? b_strides[1] : b_strides[3]) == 1;
        alignas(16) threadgroup float buf[N];
        float x[EPT];
        #pragma clang loop unroll(full)
        for (uint u = 0; u < EPT / 4; u++) {
          const uint index = EPT * tid + 4 * u;
          const uint col = bcol + index;
          uint src = col;
          if (GR > 1) {
            const uint d = col % uint(GD);
            const uint hr = col / uint(GD);
            const uint h = hr / uint(GR);
            const uint rr = hr % uint(GR);
            src = (rr * uint(GKH) + h) * uint(GD) + d;
          }
          const float4 a4 = bonsai_q8p_load<FIT32, AHD>(a, a_shape, a_strides, row, src, AV);
          const float4 b4 = bonsai_q8p_load<FIT32, BHD>(b, b_shape, b_strides, row, src, BV);
          const float4 s4 = bonsai_ld4(signs + col);
          float inv = 0.0f;
          if (PROD == 3) {
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (int r = 0; r < 4; r++) {
              const float tx = a4[r];
              acc += tx * tx;
            }
            acc = simd_sum(acc);
            inv = metal::precise::rsqrt(acc / float(GD) + eps[0]);
          }
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            const float av = a4[r];
            const float bv = b4[r];
            float v;
            if (PROD == 1) {
              v = (av * bonsai_sigmoid(av)) * bv;
            } else if (PROD == 2) {
              v = av * bonsai_sigmoid(bv);
            } else {
              const float xn = w[src % uint(GD) + uint(r)] * (av * inv);
              v = (bv * bonsai_sigmoid(bv)) * xn;
            }
            x[4 * u + uint(r)] = v * s4[r];
          }
        }
        bonsai_hadamard_radix<short(EPT)>(x);
        #pragma clang loop unroll(full)
        for (ushort s = 1; s < 32; s <<= 1) {
          const bool upper = (lane & s) != 0;
          #pragma clang loop unroll(full)
          for (uint r = 0; r < EPT; r++) {
            const float o = simd_shuffle_xor(x[r], s);
            x[r] = upper ? (o - x[r]) : (x[r] + o);
          }
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < EPT; r += 4) {
          *(threadgroup float4*)(buf + EPT * tid + r) = float4(x[r], x[r + 1], x[r + 2], x[r + 3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (TPB == 256) {
          if (tid < 128) {
            float y[8];
            #pragma clang loop unroll(full)
            for (short k = 0; k < 8; k++) {
              y[k] = buf[tid + 128 * k];
            }
            bonsai_hadamard_radix<8>(y);
            #pragma clang loop unroll(full)
            for (short k = 0; k < 8; k++) {
              buf[tid + 128 * k] = y[k];
            }
          }
        } else {
          #pragma clang loop unroll(full)
          for (uint u = 0; u < 2; u++) {
            const uint e = tid + 128 * u;
            float y[4];
            #pragma clang loop unroll(full)
            for (short k = 0; k < 4; k++) {
              y[k] = buf[e + 256 * k];
            }
            bonsai_hadamard_radix<4>(y);
            #pragma clang loop unroll(full)
            for (short k = 0; k < 4; k++) {
              buf[e + 256 * k] = y[k];
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint gi = sg; gi < 8; gi += uint(TPB) / 32) {
          const short index = short(gi * 128 + 4 * lane);
          const float4 shared = *(const threadgroup float4*)(buf + index);
          float v[4];
          float amax = 0.0f;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            v[r] = shared[r] * 0.03125f;
            amax = max(amax, fabs(v[r]));
          }
          amax = simd_max(amax);
          const float qs = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
          const float iqs = amax > 0.0f ? 127.0f / amax : 0.0f;
          float part = 0.0f;
          uchar4 packed;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            const float q = rint(v[r] * iqs);
            part += q;
            packed[r] = SIGNED ? as_type<uchar>(int8_t(q)) : uint8_t(int(q) + 128);
          }
          if (PERM) {
            uint word = as_type<uint>(packed);
            uint other = simd_shuffle_xor(word, 1);
            word = (lane & 1u)
                ? ((word & 0xff00ff00u) | ((other & 0xff00ff00u) >> 8))
                : ((word & 0x00ff00ffu) | ((other & 0x00ff00ffu) << 8));
            other = simd_shuffle_xor(word, 2);
            word = (lane & 2u)
                ? ((word & 0xffff0000u) | ((other & 0xffff0000u) >> 16))
                : ((word & 0x0000ffffu) | ((other & 0x0000ffffu) << 16));
            packed = as_type<uchar4>(word);
          }
          *(device uchar4*)(out + rowbase + bcol + uint(index)) = packed;
          part = simd_sum(part);
          if (lane == 0) {
            const uint g = uint(bcol / 128) + uint(gi);
            const uint ml = row & 63u;
            const uint qidx = MPERM
              ? (uint(row >> 6) * uint(W / 128) * 64 + g * 64
                 + uint(((ml >> 4) & 1u) * 32u + (ml & 7u) * 4u + ((ml >> 5) & 1u) * 2u + ((ml >> 3) & 1u)))
              : (uint(row) * uint(W / 128) + g);
            qscale[qidx] = qs;
            qsum[qidx] = qs * part;
          }
        }
        """

    private static let producerKernel = MLXFast.metalKernel(
        name: "bonsai_signed_hadamard_1024_q8p_blocks",
        inputNames: ["a", "b", "w", "eps", "signs"],
        outputNames: ["out", "qscale", "qsum"],
        source: producerSource,
        header: producerHeader,
        ensureRowContiguous: false)

    private struct ProducerForm: Hashable {
        let width: Int, prod: Int, gr: Int, gkh: Int, gd: Int, ahd: Int, bhd: Int
        let perm: Int, mperm: Int, signed: Int, adtype: String, bdtype: String, tpb: Int
        let fit32: Int
    }

    private static let producerLock = NSLock()
    nonisolated(unsafe) private static var producerVerdicts: [ProducerForm: Bool] = [:]

    /// The stock producer kernel's outputs for these operands from this
    /// kernel, or nil (off, not a verify-width launch, or a form that failed
    /// or cannot take its self-test). `stock` launches the stock producer
    /// kernel with the same template on given operands.

    /// True when every stride and every corner offset of this launch fits in
    /// a signed 32-bit int. Negative strides are included. A false answer
    /// keeps the int64 address path.
    private static func producerOffsetsFit32(
        _ x: MLXArray, rows: Int, width: Int, head: Int
    ) -> Bool {
        let st = x.strides
        func fits(_ v: Int) -> Bool { v >= Int(Int32.min) && v <= Int(Int32.max) }
        guard rows > 0, width > 0, st.allSatisfy(fits) else { return false }
        func corner(_ parts: [(Int, Int)]) -> Bool {
            var acc = 0
            for (n, s) in parts {
                let p = n.multipliedReportingOverflow(by: s)
                if p.overflow { return false }
                let q = acc.addingReportingOverflow(p.partialValue)
                if q.overflow { return false }
                acc = q.partialValue
            }
            return fits(acc)
        }
        if head == 0 {
            guard st.count >= 2 else { return false }
            for row in [0, rows - 1] {
                for col in [0, width - 1] {
                    if !corner([(row, st[0]), (col, st[1])]) { return false }
                }
            }
            return true
        }
        guard x.ndim == 4, st.count >= 4, head > 0 else { return false }
        let L = x.dim(1)
        guard L > 0 else { return false }
        for row in [0, rows - 1] {
            for col in [0, width - 1] {
                if !corner([
                    (row / L, st[0]), (row % L, st[1]),
                    (col / head, st[2]), (col % head, st[3]),
                ]) { return false }
            }
        }
        return true
    }

    static func launchProducer(
        a: MLXArray, b: MLXArray, w: MLXArray, eps: MLXArray, signs: MLXArray,
        template: [(String, any KernelTemplateArg)], rows: Int, width: Int,
        outShape: [Int], groupShape: [Int], codesDType: DType,
        stock: ([MLXArray], [(String, any KernelTemplateArg)]) -> [MLXArray]
    ) -> SignedBlockHadamard.Int8Activation? {
        guard producerEnabled, rows > 0, width > 0, width % 1024 == 0,
            rows <= Int(Int32.max) / width, !producerVerifying
        else { return nil }
        func value(_ name: String) -> Int? {
            guard let arg = template.first(where: { $0.0 == name })?.1 else { return nil }
            if let v = arg as? Int { return v }
            if let v = arg as? Bool { return v ? 1 : 0 }
            return nil
        }
        guard let prod = value("PROD"), let gr = value("GR"), let gkh = value("GKH"),
            let gd = value("GD"), let ahd = value("AHD"), let bhd = value("BHD"),
            let perm = value("PERM"), let mperm = value("MPERM"), let signed = value("SIGNED")
        else { return nil }
        guard rows < BonsaiPromptWidth.minimumRows || (prod == 3 && producerPromptGatedNorm)
        else { return nil }
        let blocks = rows * (width / 1024)
        // The gated norm holds one head per simdgroup: always 256 threads.
        let tpb = prod == 3 ? 256 : threads(blocks: blocks)
        guard (1 ... 3).contains(prod), gr == 1 || gd % 4 == 0,
            prod != 3 || (gd == 128 && tpb == 256)
        else { return nil }
        // Verify width only. Prompt-width 32-bit offsets have regressed the
        // scoring box on another kernel; this producer stays int64 there.
        let fit32 = rows < BonsaiPromptWidth.minimumRows
            && producerOffsetsFit32(a, rows: rows, width: width, head: ahd)
            && producerOffsetsFit32(b, rows: rows, width: width, head: bhd) ? 1 : 0
        let form = ProducerForm(
            width: width, prod: prod, gr: gr, gkh: gkh, gd: gd, ahd: ahd, bhd: bhd, perm: perm,
            mperm: mperm, signed: signed, adtype: "\(a.dtype)", bdtype: "\(b.dtype)", tpb: tpb,
            fit32: fit32)
        let tmpl = template + [("TPB", tpb), ("FIT32", fit32)]
        func run(_ inputs: [MLXArray]) -> [MLXArray] {
            producerKernel(
                inputs, template: tmpl,
                grid: (tpb * blocks, 1, 1), threadGroup: (tpb, 1, 1),
                outputShapes: [outShape, groupShape, groupShape],
                outputDTypes: [codesDType, .float32, .float32])
        }
        guard producerVerified(form, a: a, b: b, w: w, eps: eps, signs: signs,
            template: template, run: run, stock: stock)
        else { return nil }
        let outs = run([a, b, w, eps, signs])
        return SignedBlockHadamard.Int8Activation(codes: outs[0], scales: outs[1], scaledSums: outs[2])
    }

    /// A synthetic operand with `like`'s shape, strides and dtype: a fresh
    /// base covering its strided extent (wide magnitude spreads; `zeros`
    /// clears the first eighth of the base), viewed through the same strides.
    private static func synthetic(like x: MLXArray, seed: UInt64, zeros: Bool) -> MLXArray {
        let shape = x.shape
        let strides = x.strides
        var extent = 1
        for (n, s) in zip(shape, strides) where n > 1 { extent += (n - 1) * max(s, 0) }
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 2)
        var base = MLXRandom.normal([extent], key: keys[0])
            * exp(MLXRandom.normal([extent], key: keys[1]) * Float(1.5))
        if zeros {
            base = which(MLXArray(0 ..< extent) .< MLXArray(Int32(max(extent / 8, 1))), Float(0), base)
        }
        return asStrided(base.asType(x.dtype), shape, strides: strides, offset: 0)
    }

    private static func producerVerified(
        _ form: ProducerForm, a: MLXArray, b: MLXArray, w: MLXArray, eps: MLXArray,
        signs: MLXArray, template: [(String, any KernelTemplateArg)],
        run: ([MLXArray]) -> [MLXArray],
        stock: ([MLXArray], [(String, any KernelTemplateArg)]) -> [MLXArray]
    ) -> Bool {
        producerLock.lock()
        defer { producerLock.unlock() }
        if let verdict = producerVerdicts[form] { return verdict }
        producerVerifying = true
        defer { producerVerifying = false }
        var values = 0
        var mismatches = 0
        var detail = ""
        do {
            try withError { error in
                // The first call's operands (evaluated here once, at the verify
                // graph's first build), then two synthetic sets in their layout.
                eval(a, b, w, eps, signs)
                let sets: [[MLXArray]] = [
                    [a, b, w, eps, signs],
                    [synthetic(like: a, seed: 61, zeros: false), synthetic(like: b, seed: 62, zeros: false),
                     w, eps, signs],
                    [synthetic(like: a, seed: 63, zeros: true), synthetic(like: b, seed: 64, zeros: true),
                     w, eps, signs],
                ]
                for inputs in sets {
                    let p = stock(inputs, template)
                    let q = run(inputs)
                    guard p.count == 3, q.count == 3 else { throw SelfTestFailure.message("output count") }
                    var differ: [MLXArray] = []
                    for (x, y) in zip(p, q) {
                        guard x.shape == y.shape, x.dtype == y.dtype else {
                            throw SelfTestFailure.message("shape or dtype mismatch")
                        }
                        let bits: DType = x.dtype == .float32 ? .uint32 : x.dtype
                        differ.append((x.view(dtype: bits) .!= y.view(dtype: bits)).asType(.int32).sum())
                        values += x.size
                    }
                    let count = stacked(differ).sum()
                    eval(count)
                    try error.check()
                    mismatches += Int(count.item(Int32.self))
                }
            }
        } catch {
            detail = " (\(error))"
            mismatches = max(mismatches, 1)
        }
        let passed = mismatches == 0
        producerVerdicts[form] = passed
        FileHandle.standardError.write(
            ("bonsai producer rotation q8 per block (prod \(form.prod), width \(form.width), "
                + "\(form.tpb) threads, fit32 \(form.fit32), a \(form.adtype)/\(form.ahd), b \(form.bdtype)/\(form.bhd)): "
                + "self-test " + (passed ? "passed" : "FAILED") + ": \(values) values, "
                + "\(mismatches) mismatches" + detail + (passed ? "\n" : "; stock kernel kept\n"))
                .data(using: .utf8)!)
        return passed
    }
}

// MARK: - Chunked GDN fresh scan: exact forms and a load-time trial

/// Exact forms of the prompt-width fresh scan (`freshChunks`' kernel), and a
/// load-time trial that keeps the fastest on this device. Every form runs the
/// record's text (its prefetch staging when active) with the same
/// `simdgroup_multiply_accumulate` sequence into every accumulator and the
/// same elementwise operations; what changes:
/// - `kt`: each chunk's K is also staged transposed (`KTsh[Dk][9]`), written
///   from the registers that stage `Ksh`, so the state update's K^T tiles are
///   plain `simdgroup_load`s holding the same 64 values the stock transposed
///   load of `Ksh` returns; and the staging reads and per-chunk v / y / gf
///   pointers use 32-bit offsets (the same elements; taken only when every
///   offset is below 2^31).
/// - `/8`: eight 8-row state blocks per threadgroup instead of four (each
///   simdgroup still owns the same 8 rows; only how many share a chunk's
///   staging changes).
/// Which form is fastest depends on the GPU (cores; registers and threadgroup
/// memory per core; the K^T tile adds 4.6 KB per threadgroup). M4 Max, 32
/// cores, 48 value heads (192 threadgroups of 4, 6 per core): record 324 us,
/// `kt/4` 300 us (-8%), `kt/8` 289 us (-11%); the per-core load of a 40-core
/// part (40 heads on 32 cores, 5 threadgroups of 4 per core): record 264 us,
/// `kt/4` 250 us (-6%), `kt/8` 287 us (+9%). So each form is checked bit for
/// bit against the stock scan (`chunks` from a zero state; outputs and final
/// state; 8, 9, 64 and 65 chunks) at model construction, the passing forms are
/// timed against the record's launch on this device (serialized launches in
/// interleaved rounds, the median ratio), and the fastest is installed only
/// when it beats the record by `scanTrialMargin` and a confirmation run
/// agrees; otherwise the record's launch stays. Double-buffered staging (one
/// barrier per chunk) and v / decay-factor prefetch were measured and lose
/// (M4 Max: +28% and +0-7%), so they are not offered. `BONSAI_GDN_SCAN_FORMS=0`
/// keeps the record's launch; `BONSAI_GDN_SCAN_FORM=<name>` installs that form
/// after its bitwise check, without timing.
extension Qwen35GatedDeltaChunked {
    static let scanFormsEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_SCAN_FORMS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let scanFormForced: String? = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_SCAN_FORM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return (value?.isEmpty ?? true) ? nil : value
    }()

    /// The installed form's kernel, simdgroups per threadgroup and checked
    /// geometry [Hk, Dk, Hv, Dv] (set once, at model construction, before any
    /// forward); nil keeps the record's launch.
    nonisolated(unsafe) static var installedScanForm:
        (kernel: MLXFast.MLXFastKernel, simdgroups: Int, geometry: [Int])? = nil

    /// Relative gain a form needs over the record's launch to be installed.
    private static let scanTrialMargin = 0.02

    /// `text` with K also staged transposed and 32-bit offsets (`kt`), by
    /// checked replacements on the record's fresh scan text, prefetch staging
    /// (`prefetch`) or stock; nil when a target moved.
    private static func ktSource(_ text: String, prefetch: Bool) -> String? {
        var text = text
        let kStore = prefetch
            ? "*(threadgroup float4*)(Ksh + row * LK + c4) = pk_[i];"
            : "*(threadgroup float4*)(Ksh + row * LK + c4) = *(const device float4*)(kbase + src);"
        let kValue = prefetch ? "pk_[i]" : "kv4"
        let kStaged = (prefetch ? "" : "const float4 kv4 = *(const device float4*)(kbase + src);\n")
            + "*(threadgroup float4*)(Ksh + row * LK + c4) = \(kValue);\n"
            + "KTsh[(c4 + 0) * LT + row] = \(kValue).x;\n"
            + "KTsh[(c4 + 1) * LT + row] = \(kValue).y;\n"
            + "KTsh[(c4 + 2) * LT + row] = \(kValue).z;\n"
            + "KTsh[(c4 + 3) * LT + row] = \(kValue).w;"
        let t = prefetch ? "tt0" : "t0"
        let n = prefetch ? "nn" : "n"
        for (target, replacement) in [
            ("threadgroup float Ksh[C * LK];",
             "threadgroup float Ksh[C * LK];\nconstexpr int LT = 9;\nthreadgroup float KTsh[Dk * LT];"),
            (kStore, kStaged),
            ("simdgroup_load(kt, Ksh + (ti * 8) * LK + d * 8, LK, ulong2(0, 0), true);",
             "simdgroup_load(kt, KTsh + (d * 8) * LT + ti * 8, LT);"),
            ("const size_t src = (size_t)(\(t) + row) * ks + c4;",
             "const int src = (\(t) + row) * ks + c4;"),
            ("(which == 0 ? tbase : pbase) + (size_t)\(n) * C * C;",
             "(which == 0 ? tbase : pbase) + \(n) * C * C;"),
            ("const device float* v_ = v + ((size_t)b_idx * T_ + t0) * vs + hv * Dv + r0;",
             "const device float* v_ = vb0 + t0 * vs;"),
            ("device float* y_ = y + ((size_t)b_idx * T_ + t0) * vs + hv * Dv + r0;",
             "device float* y_ = yb0 + t0 * vs;"),
            ("const device float* gf_ = gf + ((size_t)bh * NC + n) * 2 * C;",
             "const device float* gf_ = gfb0 + n * 2 * C;"),
            ("simdgroup_float8x8 St[DT];",
             "const device float* vb0 = v + ((size_t)b_idx * T_) * vs + hv * Dv + r0;\n"
                + "device float* yb0 = y + ((size_t)b_idx * T_) * vs + hv * Dv + r0;\n"
                + "const device float* gfb0 = gf + (size_t)bh * NC * 2 * C;\n"
                + "simdgroup_float8x8 St[DT];"),
        ] {
            guard text.components(separatedBy: target).count == 2 else { return nil }
            text = text.replacingOccurrences(of: target, with: replacement)
        }
        return text
    }

    private static let ktKernel: MLXFast.MLXFastKernel? = {
        let prefetch = scanPrefetchActive && scanFreshPrefetchSource != nil
        guard let text = ktSource(
            prefetch ? scanFreshPrefetchSource! : scanFreshSource, prefetch: prefetch)
        else {
            FileHandle.standardError.write(
                "qwen35: chunked GDN scan forms: the scan text moved; the record's launch kept\n"
                    .data(using: .utf8)!)
            return nil
        }
        return MLXFast.metalKernel(
            name: prefetch ? "bonsai_gated_delta_chunk_scan_fresh_pf_kt"
                : "bonsai_gated_delta_chunk_scan_fresh_kt",
            inputNames: ["q", "k", "v", "tp", "pm", "gf", "T"],
            outputNames: ["y", "state_out"],
            source: Qwen35IO32.narrow(text, count: 8, "bonsai_gated_delta_chunk_scan_fresh_kt"))
    }()

    private struct ScanForm {
        let name: String
        let kernel: MLXFast.MLXFastKernel
        let simdgroups: Int
    }

    private static func scanFormLaunch(
        _ form: ScanForm, q: MLXArray, k: MLXArray, v: MLXArray, prepared: [MLXArray],
        stateShape: [Int]
    ) -> (MLXArray, MLXArray) {
        let (B, T, Hk, Dk, Hv, Dv) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3), v.dim(2), v.dim(3))
        let outputs = form.kernel(
            [q, k, v, prepared[0], prepared[1], prepared[2], MLXArray(Int32(T))],
            template: [
                ("C", chunk), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
                ("NS", form.simdgroups),
            ],
            grid: (32, Dv / 8, B * Hv),
            threadGroup: (32, form.simdgroups, 1),
            outputShapes: [[B, T, Hv, Dv], stateShape],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }

    /// Check the forms, time the passing ones, install the fastest (see the
    /// type's notes). Called by `prepareFresh` once its verdict passed.
    static func prepareScanForms(hk: Int, dk: Int, hv: Int, dv: Int) {
        guard scanFormsEnabled, installedScanForm == nil, chunk == 8, dk == 128,
            dv % 64 == 0, 520 * max(hk * dk, hv * dv) < Int(Int32.max)
        else { return }
        let record = recordFreshScanKernel
        var forms = [ScanForm(name: "record/4", kernel: record, simdgroups: scanSimdgroups)]
        if let ktKernel {
            forms += [
                ScanForm(name: "kt/4", kernel: ktKernel, simdgroups: 4),
                ScanForm(name: "kt/8", kernel: ktKernel, simdgroups: 8),
                // The same K^T body with two and with sixteen 8-row state blocks per
                // threadgroup (dv / 8 = 16 rows' blocks per head admit both): the
                // optimum moves with the core count (see above), and a 40-core part
                // was never offered these two. Each still passes the bitwise check
                // below before it is timed.
                ScanForm(name: "kt/2", kernel: ktKernel, simdgroups: 2),
                ScanForm(name: "kt/16", kernel: ktKernel, simdgroups: 16),
            ]
        }
        // The record's own text at two, eight and sixteen blocks per threadgroup
        // (only the staging's sharing changes, as for kt): never offered before.
        forms += [2, 8, 16].map { ScanForm(name: "record/\($0)", kernel: record, simdgroups: $0) }
        forms = forms.filter { (dv / 8) % $0.simdgroups == 0 }
        guard forms.count > 1, forms[0].simdgroups == 4 else { return }
        let start = DispatchTime.now().uptimeNanoseconds
        let keys = MLXRandom.split(key: MLXRandom.key(0x6b74_7363), into: 8)
        var passing = [0]
        var failed: [String] = []
        var timingSet: (q: MLXArray, k: MLXArray, v: MLXArray, p: [MLXArray], s: [Int])? = nil
        do {
            try withError { error in
                var verdict = [Bool](repeating: true, count: forms.count)
                for T in [64, 72, 512, 520] {
                    // The fresh check's operands: a wide magnitude spread, rows
                    // 8..15 with g == 0 (the prep's underflow mask), rows 16..23
                    // with a subnormal g.
                    func spread(_ shape: [Int], _ i: Int) -> MLXArray {
                        MLXRandom.normal(shape, key: keys[i])
                            * exp(MLXRandom.normal(shape, key: keys[i + 3]))
                    }
                    let q = spread([1, T, hk, dk], 0) * 0.1
                    let k = spread([1, T, hk, dk], 1) * 0.1
                    let v = spread([1, T, hv, dv], 2)
                    let rowIndex = MLXArray.arange(T).reshaped(1, T, 1)
                    let g0 = MLXRandom.uniform(0.5 ..< 1.0, [1, T, hv], key: keys[6])
                    let g = which(
                        (rowIndex .>= 8) .&& (rowIndex .< 16), Float(0),
                        which((rowIndex .>= 16) .&& (rowIndex .< 24), Float(1e-39), g0))
                    let beta = MLXRandom.uniform(0.0 ..< 1.0, [1, T, hv], key: keys[7])
                    let stateShape = [1, hv, dv, dk]
                    let (yRef, sRef) = chunks(
                        q: q, k: k, v: v, g: g, beta: beta,
                        state: MLXArray.zeros(stateShape, dtype: .float32))
                    let p = prep(q: q, k: k, g: g, beta: beta)
                    eval([yRef, sRef] + p)
                    try error.check()
                    for f in 1 ..< forms.count where verdict[f] {
                        let (y, s) = scanFormLaunch(
                            forms[f], q: q, k: k, v: v, prepared: p, stateShape: stateShape)
                        let differ = (yRef.view(dtype: .uint32) .!= y.view(dtype: .uint32))
                            .asType(.int32).sum()
                            + (sRef.view(dtype: .uint32) .!= s.view(dtype: .uint32))
                            .asType(.int32).sum()
                        eval(differ)
                        try error.check()
                        if differ.item(Int32.self) != 0 { verdict[f] = false }
                    }
                    if T == 512 { timingSet = (q, k, v, p, stateShape) }
                }
                for f in 1 ..< forms.count {
                    if verdict[f] { passing.append(f) } else { failed.append(forms[f].name) }
                }
            }
        } catch {
            FileHandle.standardError.write(
                "qwen35 GDN scan forms: self-test error (\(error)); the record's launch kept\n"
                    .data(using: .utf8)!)
            return
        }
        var line = "qwen35 GDN scan forms: bitwise vs the stock scan (8, 9, 64, 65 chunks): "
            + "passed [" + passing.dropFirst().map { forms[$0].name }.joined(separator: " ") + "]"
            + (failed.isEmpty ? "" : " FAILED [" + failed.joined(separator: " ") + "]")
        func finish(_ text: String) {
            let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
            FileHandle.standardError.write(
                (line + "; " + text + String(format: " (%.0f ms)\n", ms)).data(using: .utf8)!)
        }
        if let forced = scanFormForced {
            if let f = passing.first(where: { forms[$0].name == forced }), f != 0 {
                installedScanForm = (forms[f].kernel, forms[f].simdgroups, [hk, dk, hv, dv])
                finish("forced \(forced), installed")
            } else {
                finish("forced \(forced) not available; the record's launch kept")
            }
            return
        }
        guard passing.count > 1, let set = timingSet else {
            finish("the record's launch kept")
            return
        }
        // Interleaved rounds of bursts (the first round warms up and is not
        // kept); each form's score is its median time ratio to the record's.
        // A burst's launches are serialized as in the forward (each takes the
        // previous launch's output rows as its v): independent launches would
        // overlap on the GPU and time throughput instead of latency.
        func sample(_ f: Int, burst: Int) -> Double {
            var outputs: [MLXArray] = []
            var rows = set.v
            for _ in 0 ..< burst {
                let (y, s) = scanFormLaunch(
                    forms[f], q: set.q, k: set.k, v: rows, prepared: set.p, stateShape: set.s)
                outputs.append(s)
                rows = y
            }
            outputs.append(rows)
            let begin = DispatchTime.now().uptimeNanoseconds
            eval(outputs)
            return Double(DispatchTime.now().uptimeNanoseconds - begin) / Double(burst)
        }
        let burst = 6
        var times = [[Double]](repeating: [], count: forms.count)
        for round in 0 ..< 6 {
            for j in passing.indices {
                let f = passing[(j + round) % passing.count]
                let t = sample(f, burst: burst)
                if round > 0 { times[f].append(t) }
            }
        }
        func median(_ x: [Double]) -> Double {
            let s = x.sorted()
            return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        }
        func score(_ f: Int) -> Double {
            median(zip(times[f], times[0]).map { $0 / $1 }) - 1
        }
        line += String(format: "; record/4 %.1f us", median(times[0]) / 1e3)
        var best = passing[1]
        for f in passing.dropFirst() {
            line += " | \(forms[f].name) " + String(format: "%+.1f%%", score(f) * 100)
            if score(f) < score(best) { best = f }
        }
        guard score(best) < -scanTrialMargin else {
            finish("the record's launch kept")
            return
        }
        // Confirmation: alternating pairs, the same median ratio over all rounds.
        for round in 0 ..< 6 {
            let first = round % 2 == 0 ? 0 : best
            let a = sample(first, burst: burst)
            let b = sample(first == 0 ? best : 0, burst: burst)
            times[0].append(first == 0 ? a : b)
            times[best].append(first == 0 ? b : a)
        }
        let confirmed = score(best)
        if confirmed < -scanTrialMargin {
            installedScanForm = (forms[best].kernel, forms[best].simdgroups, [hk, dk, hv, dv])
            finish("\(forms[best].name) confirmed " + String(format: "%+.1f%%", confirmed * 100)
                + ", installed")
        } else {
            finish("\(forms[best].name) not confirmed " + String(format: "(%+.1f%%)", confirmed * 100)
                + "; the record's launch kept")
        }
    }
}


// MARK: - Chunked GDN fresh scan: v from the conv input (v-fold)

/// The prompt-width fresh scan reading the value columns' conv input instead
/// of the value launch's FP32 `v` (`freshStridedRows` forms 2 and 3 launch it
/// beside the q/k launch; it writes `[S, Hv, Dv]` FP32 that only this scan
/// reads). Each thread forms the two `v` elements it would have loaded, for
/// its row and its two columns, with the value launch's own arithmetic: the
/// taps before the prompt read 0.0f (the fresh state), `acc = fma(x, w, acc)`
/// for j = 0..KS-1 from 0.0f, then MLX's SiLU in its stable functor form,
/// all in FP32 from `float(qkv[...])`, exactly as `conv_silu_rows` computes
/// them. Every later operation is the record scan's text. The value launch is
/// then never evaluated (its output has no other reader) and its store and
/// reload of `v` go away. Two forms: the taps read where `v` was read, and the
/// next chunk's taps loaded into registers during this chunk. Each form is
/// checked bit for bit against the value launch followed by the scan
/// `freshChunks` would otherwise launch (y and the final state; FP32, FP16 and
/// BF16 conv inputs; 8 and 64 chunks), the passing forms are timed against
/// that pair on this device (interleaved rounds, the median ratio), and the
/// fastest is installed only when it beats the pair by `scanTrialMargin` and
/// a confirmation run agrees. Taken only after a q/k launch that formed the
/// chunk prep (form 3, whose value launch is a separate launch).
/// `BONSAI_GDN_SCAN_VFOLD=0` keeps the value launch.
extension Qwen35GatedDeltaChunked {
    static let vfoldEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_SCAN_VFOLD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The installed v-fold form's kernel and checked geometry [Hk, Dk, Hv, Dv]
    /// (set once, at model construction); nil keeps the value launch.
    nonisolated(unsafe) static var installedVFold:
        (kernel: MLXFast.MLXFastKernel, geometry: [Int])? = nil

    private static let vfoldPrelude = """
        // v from the conv input: the value launch's conv (taps from 0.0f before
        // the prompt, fma(x, w, acc) for j = 0..KS-1) and MLX's SiLU, for this
        // thread's two columns of its 8 state rows.
        constexpr int VOFF = 2 * Hk * Dk;
        constexpr int NK = KS - 1;
        const int vqb = b_idx * (int)qkv_strides[0];
        const int vqs1 = (int)qkv_strides[1];
        const int vqs2 = (int)qkv_strides[2];
        const int vcol0 = VOFF + hv * Dv + r0 + fn;
        float vw0[KS], vw1[KS];
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < KS; j++) {
          vw0[j] = w[vcol0 * (int)w_strides[0] + j * (int)w_strides[1]];
          vw1[j] = w[(vcol0 + 1) * (int)w_strides[0] + j * (int)w_strides[1]];
        }
        auto vsilu = [&](float acc) -> float {
          const float sy = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
          const float sig = (acc < 0.0f) ? sy : 1.0f - sy;
          return acc * sig;
        };
        auto vconv = [&](int t, int col, thread const float* wt) -> float {
          float acc = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < KS; j++) {
            const int r = t + j - NK;
            const float xv = (r < 0) ? 0.0f : float(qkv[vqb + r * vqs1 + col * vqs2]);
            acc = fma(xv, wt[j], acc);
          }
          return vsilu(acc);
        };

        """

    private static let vfoldTapRegisters = """
        float vx0[KS], vx1[KS];
        auto vtaps = [&](int t) {
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < KS; j++) {
            const int r = t + j - NK;
            vx0[j] = (r < 0) ? 0.0f : float(qkv[vqb + r * vqs1 + vcol0 * vqs2]);
            vx1[j] = (r < 0) ? 0.0f : float(qkv[vqb + r * vqs1 + (vcol0 + 1) * vqs2]);
          }
        };
        auto vconv_regs = [&](thread const float* x, thread const float* wt) -> float {
          float acc = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < KS; j++) {
            acc = fma(x[j], wt[j], acc);
          }
          return vsilu(acc);
        };
        vtaps(fm);

        """

    /// `text` (the record's fresh scan, prefetch staging or stock) reading the
    /// conv input for `v`, its taps in registers when `registers`; nil when a
    /// target moved.
    private static func vfoldSource(_ text: String, registers: Bool) -> String? {
        var text = text
        let vRead = "const float2 vv = *(const device float2*)(v_ + row * vs + fn);"
        var pairs: [(String, String)] = [
            ("simdgroup_float8x8 St[DT];",
             vfoldPrelude + (registers ? vfoldTapRegisters : "") + "simdgroup_float8x8 St[DT];"),
            ("const device float* v_ = v + ((size_t)b_idx * T_ + t0) * vs + hv * Dv + r0;", ""),
            (vRead, registers
                ? "const float2 vv = float2(vconv_regs(vx0, vw0), vconv_regs(vx1, vw1));"
                : "const float2 vv = float2(vconv(t0 + row, vcol0, vw0), vconv(t0 + row, vcol0 + 1, vw1));"),
        ]
        if registers {
            pairs.append(
                ("// Delta = T' Z (T' lower triangular)",
                 "if (n + 1 < NC) { vtaps((n + 1) * C + fm); }\n// Delta = T' Z (T' lower triangular)"))
        }
        for (target, replacement) in pairs {
            guard text.components(separatedBy: target).count == 2 else { return nil }
            text = text.replacingOccurrences(of: target, with: replacement)
        }
        // No read of the value launch's output is left.
        guard !text.contains("(v_ + "), !text.contains("v_ = v + ") else { return nil }
        return text
    }

    private struct VFoldForm {
        let name: String
        let kernel: MLXFast.MLXFastKernel
    }

    private static let vfoldForms: [VFoldForm] = {
        let prefetch = scanPrefetchActive && scanFreshPrefetchSource != nil
        let base = prefetch ? scanFreshPrefetchSource! : scanFreshSource
        var forms: [VFoldForm] = []
        for registers in [false, true] {
            let name = "vfold" + (registers ? "-regs" : "")
            guard let text = vfoldSource(base, registers: registers) else {
                FileHandle.standardError.write(
                    "qwen35: chunked GDN v-fold scan (\(name)): the scan text moved; not offered\n"
                        .data(using: .utf8)!)
                continue
            }
            let kernelName = "bonsai_gated_delta_chunk_scan_fresh_vfold"
                + (registers ? "_regs" : "") + (prefetch ? "_pf" : "")
            forms.append(VFoldForm(
                name: name,
                kernel: MLXFast.metalKernel(
                    name: kernelName,
                    inputNames: ["q", "k", "qkv", "w", "tp", "pm", "gf", "T"],
                    outputNames: ["y", "state_out"],
                    source: Qwen35IO32.narrow(text, count: 10, kernelName),
                    ensureRowContiguous: false)))
        }
        return forms
    }()

    /// One v-fold launch: `freshChunks`' geometry, the conv input in place of `v`.
    static func vfoldLaunch(
        _ kernel: MLXFast.MLXFastKernel, q: MLXArray, k: MLXArray, qkv: MLXArray,
        convWeight: MLXArray, prepared: [MLXArray], valueHeads Hv: Int, headVDim Dv: Int
    ) -> (MLXArray, MLXArray) {
        let (B, T, Hk, Dk) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3))
        let outputs = kernel(
            [q, k, qkv, convWeight, prepared[0], prepared[1], prepared[2], MLXArray(Int32(T))],
            template: [
                ("C", chunk), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
                ("NS", scanSimdgroups), ("KS", convWeight.dim(1)), ("InT", qkv.dtype),
            ],
            grid: (32, Dv / 8, B * Hv),
            threadGroup: (32, scanSimdgroups, 1),
            outputShapes: [[B, T, Hv, Dv], [B, Hv, Dv, Dk]],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }

    /// Whether `freshChunks` may take the installed v-fold form for this
    /// launch: the checked geometry, the record's conv input layout, and every
    /// element offset below 2^31 (`Qwen35GDNPrework.narrowFits`).
    static func vfoldApplies(
        _ geometry: [Int], qkv: MLXArray, convWeight: MLXArray, rows T: Int
    ) -> Bool {
        guard let installed = installedVFold, installed.geometry == geometry,
            qkv.ndim == 3, qkv.dim(1) == T, convWeight.ndim == 3, convWeight.dim(2) == 1,
            convWeight.dtype == .float32,
            [DType.float32, .float16, .bfloat16].contains(qkv.dtype)
        else { return false }
        let CD = qkv.dim(2)
        return CD == 2 * geometry[0] * geometry[1] + geometry[2] * geometry[3]
            && convWeight.dim(0) == CD
            && Qwen35GDNPrework.narrowFits(batch: qkv.dim(0), rows: T, convDim: CD)
    }

    /// Check the v-fold forms, time the passing ones against the value launch
    /// and the scan it feeds, install the fastest (see the type's notes).
    /// Called by `prepareFresh` after `prepareScanForms`.
    static func prepareVFold(hk: Int, dk: Int, hv: Int, dv: Int) {
        guard vfoldEnabled, installedVFold == nil, chunk == 8, dk == 128, dv == 128,
            hk > 0, hv % hk == 0, (dv / 8) % scanSimdgroups == 0,
            Qwen35GDNPrework.narrowFits(batch: 1, rows: 520, convDim: 2 * hk * dk + hv * dv)
        else { return }
        let forms = vfoldForms
        guard !forms.isEmpty else { return }
        let start = DispatchTime.now().uptimeNanoseconds
        let cd = 2 * hk * dk + hv * dv
        let ks = 4
        let geometry = [hk, dk, hv, dv]
        // The scan `freshChunks` launches without a v-fold form.
        let installedForm = installedScanForm.flatMap {
            $0.geometry == geometry ? ($0.kernel, $0.simdgroups) : nil
        }
        let (scanKernel, ns) = installedForm ?? (recordFreshScanKernel, scanSimdgroups)
        func scanPair(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ p: [MLXArray], _ T: Int)
            -> (MLXArray, MLXArray)
        {
            let outputs = scanKernel(
                [q, k, v, p[0], p[1], p[2], MLXArray(Int32(T))],
                template: [
                    ("C", chunk), ("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv), ("NS", ns),
                ],
                grid: (32, dv / 8, hv), threadGroup: (32, ns, 1),
                outputShapes: [[1, T, hv, dv], [1, hv, dv, dk]],
                outputDTypes: [.float32, .float32])
            return (outputs[0], outputs[1])
        }
        func value(_ qkv: MLXArray, _ w: MLXArray) -> MLXArray {
            Qwen35GDNPrework.valueLaunch(
                qkv: qkv, convWeight: w, keyHeads: hk, valueHeads: hv, headKDim: dk,
                headVDim: dv, rows: Qwen35GDNPrework.rowTile)
        }
        let keys = MLXRandom.split(key: MLXRandom.key(0x7666_6f6c), into: 10)
        var verdict = [Bool](repeating: true, count: forms.count)
        var timingSet: (q: MLXArray, k: MLXArray, qkv: MLXArray, w: MLXArray, p: [MLXArray])? = nil
        do {
            try withError { error in
                for T in [64, 512] {
                    func spread(_ shape: [Int], _ i: Int) -> MLXArray {
                        MLXRandom.normal(shape, key: keys[i])
                            * exp(MLXRandom.normal(shape, key: keys[i + 5]))
                    }
                    let q = spread([1, T, hk, dk], 0) * 0.1
                    let k = spread([1, T, hk, dk], 1) * 0.1
                    // The conv input as at prompt width: a column slice of a wider stack.
                    let stack = spread([1, T, cd + hv * dv], 2)
                    let w = spread([cd, ks, 1], 3) * 0.5
                    let rowIndex = MLXArray.arange(T).reshaped(1, T, 1)
                    let g0 = MLXRandom.uniform(0.5 ..< 1.0, [1, T, hv], key: keys[4])
                    let g = which(
                        (rowIndex .>= 8) .&& (rowIndex .< 16), Float(0),
                        which((rowIndex .>= 16) .&& (rowIndex .< 24), Float(1e-39), g0))
                    let beta = MLXRandom.uniform(0.0 ..< 1.0, [1, T, hv], key: keys[9])
                    let p = prep(q: q, k: k, g: g, beta: beta)
                    for dtype in [DType.float32, .float16, .bfloat16] {
                        let qkv = stack.asType(dtype)[0..., 0..., 0 ..< cd]
                        let (yRef, sRef) = scanPair(q, k, value(qkv, w), p, T)
                        eval([yRef, sRef] + p)
                        try error.check()
                        for f in forms.indices where verdict[f] {
                            let (y, s) = vfoldLaunch(
                                forms[f].kernel, q: q, k: k, qkv: qkv, convWeight: w,
                                prepared: p, valueHeads: hv, headVDim: dv)
                            let differ = (yRef.view(dtype: .uint32) .!= y.view(dtype: .uint32))
                                .asType(.int32).sum()
                                + (sRef.view(dtype: .uint32) .!= s.view(dtype: .uint32))
                                .asType(.int32).sum()
                            eval(differ)
                            try error.check()
                            if differ.item(Int32.self) != 0 { verdict[f] = false }
                        }
                        if T == 512 && dtype == .float32 { timingSet = (q, k, qkv, w, p) }
                    }
                }
            }
        } catch {
            FileHandle.standardError.write(
                "qwen35 GDN v-fold scan: self-test error (\(error)); the value launch kept\n"
                    .data(using: .utf8)!)
            return
        }
        let passing = forms.indices.filter { verdict[$0] }
        var line = "qwen35 GDN v-fold scan: bitwise vs the value launch + scan (8, 64 chunks; "
            + "FP32/FP16/BF16): passed [" + passing.map { forms[$0].name }.joined(separator: " ")
            + "]" + (passing.count == forms.count ? "" : " FAILED ["
                + forms.indices.filter { !verdict[$0] }.map { forms[$0].name }
                .joined(separator: " ") + "]")
        func finish(_ text: String) {
            let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
            FileHandle.standardError.write(
                (line + "; " + text + String(format: " (%.0f ms)\n", ms)).data(using: .utf8)!)
        }
        guard !passing.isEmpty, let set = timingSet else {
            finish("the value launch kept")
            return
        }
        // Arm 0: the value launch and the scan; arm 1 + f: v-fold form f. Bursts
        // of independent launches (both arms saturate the GPU, as at prompt
        // width), interleaved rounds, the first round discarded.
        let arms = [-1] + passing
        func sample(_ arm: Int, burst: Int) -> Double {
            var outputs: [MLXArray] = []
            for _ in 0 ..< burst {
                let (y, s) = arm < 0
                    ? scanPair(set.q, set.k, value(set.qkv, set.w), set.p, 512)
                    : vfoldLaunch(
                        forms[arm].kernel, q: set.q, k: set.k, qkv: set.qkv, convWeight: set.w,
                        prepared: set.p, valueHeads: hv, headVDim: dv)
                outputs += [y, s]
            }
            let begin = DispatchTime.now().uptimeNanoseconds
            eval(outputs)
            return Double(DispatchTime.now().uptimeNanoseconds - begin) / Double(burst)
        }
        let burst = 6
        var times = [[Double]](repeating: [], count: arms.count)
        for round in 0 ..< 6 {
            for j in arms.indices {
                let a = (j + round) % arms.count
                let t = sample(arms[a], burst: burst)
                if round > 0 { times[a].append(t) }
            }
        }
        func median(_ x: [Double]) -> Double {
            let s = x.sorted()
            return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        }
        func score(_ a: Int) -> Double {
            median(zip(times[a], times[0]).map { $0 / $1 }) - 1
        }
        line += String(format: "; value + scan %.1f us", median(times[0]) / 1e3)
        var best = 1
        for a in 1 ..< arms.count {
            line += " | \(forms[arms[a]].name) " + String(format: "%+.1f%%", score(a) * 100)
            if score(a) < score(best) { best = a }
        }
        guard score(best) < -scanTrialMargin else {
            finish("the value launch kept")
            return
        }
        for round in 0 ..< 6 {
            let firstIsPair = round % 2 == 0
            let a = sample(firstIsPair ? -1 : arms[best], burst: burst)
            let b = sample(firstIsPair ? arms[best] : -1, burst: burst)
            times[0].append(firstIsPair ? a : b)
            times[best].append(firstIsPair ? b : a)
        }
        let confirmed = score(best)
        if confirmed < -scanTrialMargin {
            installedVFold = (forms[arms[best]].kernel, geometry)
            finish("\(forms[arms[best]].name) confirmed "
                + String(format: "%+.1f%%", confirmed * 100) + ", installed")
        } else {
            finish("\(forms[arms[best]].name) not confirmed "
                + String(format: "(%+.1f%%)", confirmed * 100) + "; the value launch kept")
        }
    }
}

// MARK: - The prompt embedding: host ids for the stepper, built early for the seed

/// Two exact extensions of `Qwen35PromptEmbeddingHostGather` (the prompt's
/// packed embedding rows gathered on the host, so the prompt forward's first
/// command buffer never binds the 358 MB table with nothing queued ahead).
///
/// LAZY IDS (the timed prefill). The teacher-forced stepper hands the model its
/// prompt ids as a lazy reshape of a host array, and the host gather reads
/// only arrays already on the host, so the timed prefill kept the GPU gather
/// and its first command buffer bound the whole table right after the phase's
/// idle gate. The ids are evaluated first (a reshape of a host array: no
/// kernel, one empty command buffer's round trip, which after an idle is the
/// GPU's wake-up that the first real buffer would otherwise pay), then the same
/// host gather, the same dequantization and the same unfold run on them.
/// `BONSAI_PROMPT_EMBED_LAZY_IDS=0` leaves lazy ids to the GPU gather.
///
/// EARLY (the seed). The engine builds and submits the prompt's embedding at
/// the top of the round graph (`cbv2PrefetchPromptEmbedding`), before the
/// prompt row's cache and state binds and the model entry, so the seed's first
/// command buffer (and with it the GPU's wake-up) goes out that much sooner.
/// The forward then takes the embedding it would have built: the same ids
/// (compared element for element), the same array. Switched by the engine
/// (`MLXFAST_PROMPT_EMBED_EARLY`).
enum Qwen35PromptEmbeddingEarly {
    static let lazyIdsEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_PROMPT_EMBED_LAZY_IDS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Makes `inputs` host-readable when it is a lazy array (the stepper's
    /// reshaped ids).
    static func makeHostAvailable(_ inputs: MLXArray) {
        guard lazyIdsEnabled, Qwen35PromptEmbeddingHostGather.hostBytes(inputs) == nil else { return }
        eval(inputs)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var slot: (ids: [Int32], embedding: MLXArray)?

    /// Builds and submits the embedding of `ids` (`[1, rows]`, host int32) now,
    /// for the prompt forward that follows to take.
    static func prefetch(_ module: Embedding, _ ids: MLXArray) {
        lock.withLock { slot = nil }
        guard let embedding = Qwen35PromptEmbeddingHostGather.embed(module, ids),
            let bytes = Qwen35PromptEmbeddingHostGather.hostBytes(ids)
        else { return }
        asyncEval([embedding])
        let copy = Array(UnsafeBufferPointer(start: bytes.assumingMemoryBound(to: Int32.self), count: ids.size))
        lock.withLock { slot = (copy, embedding) }
    }

    /// The prefetched embedding when `inputs` holds exactly its ids; the slot
    /// is emptied either way.
    static func take(_ inputs: MLXArray) -> MLXArray? {
        guard let held = lock.withLock({ () -> (ids: [Int32], embedding: MLXArray)? in
            defer { slot = nil }
            return slot
        }), held.ids.count == inputs.size,
            let bytes = Qwen35PromptEmbeddingHostGather.hostBytes(inputs)
        else { return nil }
        let same = held.ids.withUnsafeBytes { memcmp($0.baseAddress!, bytes, $0.count) == 0 }
        return same ? held.embedding : nil
    }
}

extension Qwen35TextModel: CBv2PromptEmbeddingPrefetching {
    public func cbv2PrefetchPromptEmbedding(_ tokens: MLXArray) {
        Qwen35PromptEmbeddingEarly.prefetch(model.embedTokens, tokens)
    }
}

extension Qwen35Model: CBv2PromptEmbeddingPrefetching {
    public func cbv2PrefetchPromptEmbedding(_ tokens: MLXArray) {
        languageModel.cbv2PrefetchPromptEmbedding(tokens)
    }
}

// MARK: - Exact kernel forms chosen on the device at load

/// Launch parameters that were fixed from M4 measurements, or never timed on
/// the ranked device, where a bit-identical alternative exists; after the
/// load, outside any timed window, this device times them and keeps the
/// faster. Every alternative computes the same values (checked bit for bit
/// before it is offered), so tokens, rounds and acceptance never change.
///
/// Verify rounds (`runVerify` hooks, in the deferred boot warm after the
/// record's trials, one engine request as `Qwen35NarrowProducerTrial`):
/// - `dvpl4`: the GDN recurrence with four dv rows per lane (template `DVPL`,
///   `Qwen35GatedDeltaV3.rowsPerLane`) instead of two, for the verify scan,
///   the fused replay and the state stores. Rows never mix; checked here:
///   `run` / `runOutputOnly` / `runFreshState` at 2 and 4 rows per lane on
///   the same operands (1, 3 and 16 rows) bit for bit, then the derived
///   kernels' own self-tests (state skip, fused replay, full-accept store)
///   at 4 rows per lane, which must reach the verdicts they reached at 2.
/// - `tpb128`, `wide128`: the per-block quantizing rotations
///   (`Qwen35RotationQ8Blocks`, plain and SwiGLU / attention-gate producer)
///   with 128 threads per 1024-block instead of 256, at most 128 blocks
///   (6144 wide) and above (17408 wide). Each form is compared bit for bit
///   with the stock kernel before its first use (the arm's first round,
///   untimed); a failing form keeps the stock kernel, the same values.
/// - `rowkernel`: the verify boundary on the per-row kernel instead of
///   `Qwen35BoundaryBlocks` (80 threadgroups of 256); offered only when the
///   16-row self-test passed through both.
/// Round 0 and each arm's first round warm up; then the arms rotate for
/// `roundsPerArm` rounds each, rounds above 1.5x their arm's median dropped,
/// and an alternative is kept only when its median round beats the record's
/// by more than `adoptMargin`.
///
/// Prompt (`runPromptRows`): the GDN prompt prework with 8 or 2 rows per
/// threadgroup instead of 4 (`checkRowTile`: every launch form the record's
/// tile passed, bit for bit against the stock launch, three dtypes), raced
/// at the real prompt shape. (Whole prompt forwards are too coarse for it:
/// 2.36 s each on the M4 Max, arms within 0.02 %.)
///
/// `BONSAI_EXACT_TRIALS=0` keeps every record form (no trial, no extra
/// self-test). Per item (default on): `BONSAI_TRIAL_GDN_DVPL`,
/// `BONSAI_TRIAL_ROTATION_TPB`, `BONSAI_TRIAL_BOUNDARY`,
/// `BONSAI_TRIAL_PROMPT_ROWS` `=0`. A forced form skips its item:
/// `BONSAI_GDN_V3_DVPL=2|4`, `BONSAI_ROTATION_Q8_TPB=128`,
/// `BONSAI_ROTATION_Q8_TPB_SMALL=128`, `BONSAI_BOUNDARY_Q8_BLOCKS=0`,
/// `BONSAI_GDN_PREWORK_ROW_TILE=2|8|16`.
enum Qwen35ExactFormTrial {
    private static func on(_ name: String) -> Bool {
        let value = ProcessInfo.processInfo.environment[name]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }

    static let enabled = on("BONSAI_EXACT_TRIALS")
    static let dvplWanted = enabled && on("BONSAI_TRIAL_GDN_DVPL")
    static let rotationWanted = enabled && on("BONSAI_TRIAL_ROTATION_TPB")
    static let boundaryWanted = enabled && on("BONSAI_TRIAL_BOUNDARY")
    static let promptRowsWanted = enabled && on("BONSAI_TRIAL_PROMPT_ROWS")

    /// Whether the fused boundary's 16-row self-test also checks the per-row kernel.
    static var boundaryOffered: Bool { boundaryWanted && Qwen35BoundaryBlocks.enabled }

    /// A GDN layer of the model (its geometry and parameters for the checks).
    nonisolated(unsafe) static weak var gdnLayer: Qwen35GatedDeltaNet?

    /// One alternative: `set(true)` installs it, `set(false)` the record's form.
    struct Arm {
        let name: String
        let knob: String
        let set: (Bool) -> Void
    }

    static let roundsPerArm = 8
    static let adoptMargin = 0.005

    nonisolated(unsafe) static var active = false
    nonisolated(unsafe) private static var arms: [Arm] = []
    nonisolated(unsafe) private static var roundTimes: [[UInt64]] = []
    nonisolated(unsafe) private static var roundIndex = 0
    nonisolated(unsafe) private static var lastBoundary: UInt64 = 0
    nonisolated(unsafe) private static var onEnough: (() -> Void)?
    nonisolated(unsafe) private static var checks: [String] = []

    /// Boundaries the request needs: round 0 and one warm round per arm
    /// (untimed), `roundsPerArm` timed rounds per arm, and the closing one.
    static var roundsNeeded: Int { 2 + arms.count * (1 + roundsPerArm) }

    private static func install(_ arms: [Arm], _ chosen: Int?) {
        for arm in arms.dropFirst() { arm.set(false) }
        if let chosen, chosen > 0 { arms[chosen].set(true) }
    }

    // MARK: Verify rounds

    /// Builds the verify arms (running their checks); true when any is offered.
    static func prepareVerify() -> Bool {
        arms = []
        checks = []
        guard enabled else { return false }
        var list = [Arm(name: "record", knob: "", set: { _ in })]
        if dvplWanted, gdnLayer == nil { checks.append("dvpl4 not offered (no GDN layer)") }
        if dvplWanted, Qwen35GatedDeltaV3.enabled, Qwen35GatedDeltaV3.rowsPerLaneForced == nil,
            Qwen35GatedDeltaV3.rowsPerLane == 2, !Qwen35GDNReplayBatch.enabled,
            let layer = gdnLayer
        {
            let (passed, detail) = dvplCheck(layer: layer)
            checks.append("dvpl4 " + (passed ? "passed" : "FAILED") + " (\(detail))")
            if passed {
                list.append(Arm(name: "dvpl4", knob: "dvpl") {
                    Qwen35GatedDeltaV3.rowsPerLane = $0 ? 4 : 2
                })
            }
        }
        if rotationWanted, Qwen35RotationQ8Blocks.enabled {
            if Qwen35RotationQ8Blocks.smallThreadsForced == nil {
                list.append(Arm(name: "tpb128", knob: "small") {
                    Qwen35RotationQ8Blocks.smallThreads = $0 ? 128 : 256
                })
            }
            if Qwen35RotationQ8Blocks.wideThreads == 256 {
                list.append(Arm(name: "wide128", knob: "wide") {
                    Qwen35RotationQ8Blocks.wideThreadsInUse = $0 ? 128 : 256
                })
            }
        }
        if boundaryWanted, Qwen35BoundaryBlocks.active {
            let verdict = Qwen35BoundaryBlocks.stockVerified
            checks.append(
                "rowkernel " + (verdict == true ? "passed" : verdict == false ? "FAILED" : "untested"))
            if verdict == true {
                list.append(Arm(name: "rowkernel", knob: "boundary") {
                    Qwen35BoundaryBlocks.stock = $0
                })
            }
        }
        guard list.count > 1 else {
            if !checks.isEmpty || enabled {
                FileHandle.standardError.write(
                    ("bonsai exact forms trial (verify rounds): nothing offered"
                        + (checks.isEmpty ? "" : "; checks: " + checks.joined(separator: ", "))
                        + "\n").data(using: .utf8)!)
            }
            return false
        }
        arms = list
        return true
    }

    @inline(__always) static func roundBoundary() {
        guard active else { return }
        boundary()
    }

    private static func arm(round: Int) -> Int { round == 0 ? 0 : (round - 1) % arms.count }

    private static func boundary() {
        let now = DispatchTime.now().uptimeNanoseconds
        let ended = roundIndex - 1
        if ended > arms.count { roundTimes[arm(round: ended)].append(now - lastBoundary) }
        lastBoundary = now
        install(arms, arm(round: roundIndex))
        roundIndex += 1
        if roundIndex >= roundsNeeded {
            active = false
            install(arms, nil)
            let enough = onEnough
            onEnough = nil
            enough?()
        }
    }

    static func begin(onEnough: @escaping () -> Void) {
        roundTimes = [[UInt64]](repeating: [], count: arms.count)
        roundIndex = 0
        lastBoundary = 0
        self.onEnough = onEnough
        active = true
    }

    private static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }

    /// Per arm: the median after dropping samples above 1.5x the median, and
    /// how many were kept.
    private static func filtered(_ times: [[Double]]) -> [(Double?, Int)] {
        times.map { t in
            guard let first = median(t) else { return (nil, 0) }
            let kept = t.filter { $0 <= 1.5 * first }
            return (median(kept), kept.count)
        }
    }

    /// Per knob, the arm with the lowest median among those beating the
    /// record's by more than `margin` (with at least `minKept` samples each).
    private static func choose(
        _ arms: [Arm], _ stats: [(Double?, Int)], margin: Double, minKept: Int
    ) -> [Int] {
        guard let base = stats[0].0, stats[0].1 >= minKept else { return [] }
        var best: [String: Int] = [:]
        for i in arms.indices.dropFirst() {
            guard let t = stats[i].0, stats[i].1 >= minKept, t < base * (1 - margin) else { continue }
            if let j = best[arms[i].knob], let tj = stats[j].0, tj <= t { continue }
            best[arms[i].knob] = i
        }
        return best.values.sorted()
    }

    private static func report(
        _ title: String, _ arms: [Arm], _ stats: [(Double?, Int)], _ adopted: [Int],
        elapsed: UInt64, unit: Double = 1e6
    ) {
        func ms(_ v: Double?) -> String { v.map { String(format: "%.2f", $0 / unit) } ?? "-" }
        var parts: [String] = []
        for (i, arm) in arms.enumerated() {
            var text = "\(arm.name) \(ms(stats[i].0)) ms (\(stats[i].1))"
            if i > 0, let t = stats[i].0, let b = stats[0].0 {
                text += String(format: " %+.2f%%", (t / b - 1) * 100)
            }
            parts.append(text)
        }
        let line = "bonsai exact forms trial (\(title)): " + parts.joined(separator: " | ")
            + "; adopted " + (adopted.isEmpty ? "none (record)" : adopted.map { arms[$0].name }.joined(separator: ", "))
            + (checks.isEmpty ? "" : "; checks: " + checks.joined(separator: ", "))
            + String(format: " (%.0f ms)\n", Double(elapsed) / 1e6)
        FileHandle.standardError.write(line.data(using: .utf8)!)
    }

    /// Ends the verify trial: installs the choice, logs one line.
    static func finishVerify(elapsedNanoseconds: UInt64) {
        active = false
        onEnough = nil
        guard !arms.isEmpty else { return }
        let stats = filtered(roundTimes.map { $0.map { Double($0) } })
        let adopted = choose(arms, stats, margin: adoptMargin, minKept: 4)
        install(arms, nil)
        for i in adopted { arms[i].set(true) }
        report("verify rounds", arms, stats, adopted, elapsed: elapsedNanoseconds)
        roundTimes = []
        arms = []
        checks = []
    }

    /// GDN recurrence at 2 and 4 rows per lane, bit for bit, then the derived
    /// kernels' self-tests at 4 (restoring 2 whatever happens).
    private static func dvplCheck(layer: Qwen35GatedDeltaNet) -> (Bool, String) {
        let hk = layer.numKHeads
        let dk = layer.headKDim
        let hv = layer.numVHeads
        let dv = layer.headVDim
        guard dk == 128, dv % 64 == 0, hv % hk == 0 else { return (false, "geometry") }
        defer { Qwen35GatedDeltaV3.rowsPerLane = 2 }
        let before = (
            Qwen35GDNVerifyStateSkip.applies(to: layer), Qwen35GDNReplayFused.applies(to: layer),
            Qwen35GDNFullAcceptStore.verdict(layer: layer))
        let stagedBefore = Qwen35GDNReplayFused.stagedLive
        var values = 0
        var mismatches = 0
        do {
            try withError { error in
                let keys = MLXRandom.split(key: MLXRandom.key(0x4456_504c), into: 6)
                for T in [1, 3, 16] {
                    func spread(_ shape: [Int], _ i: Int) -> MLXArray {
                        MLXRandom.normal(shape, key: keys[i])
                            * exp(MLXRandom.normal(shape, key: keys[(i + 3) % 6]))
                    }
                    let q = spread([1, T, hk, dk], 0) * 0.1
                    let k = spread([1, T, hk, dk], 1) * 0.1
                    let v = spread([1, T, hv, dv], 2)
                    let g = MLXRandom.uniform(0.5 ..< 1.0, [1, T, hv], key: keys[3])
                    let beta = MLXRandom.uniform(0.0 ..< 1.0, [1, T, hv], key: keys[4])
                    let state = spread([1, hv, dv, dk], 5)
                    var outs: [[MLXArray]] = []
                    for rows in [2, 4] {
                        Qwen35GatedDeltaV3.rowsPerLane = rows
                        var o: [MLXArray] = []
                        if let r = Qwen35GatedDeltaV3.run(
                            q: q, k: k, v: v, g: g, beta: beta, state: state)
                        {
                            o += [r.0, r.1]
                        }
                        if let y = Qwen35GatedDeltaV3.runOutputOnly(
                            q: q, k: k, v: v, g: g, beta: beta, state: state)
                        {
                            o.append(y)
                        }
                        if let r = Qwen35GatedDeltaV3.runFreshState(
                            q: q, k: k, v: v, g: g, beta: beta, stateShape: state.shape)
                        {
                            o += [r.0, r.1]
                        }
                        eval(o)
                        outs.append(o)
                    }
                    Qwen35GatedDeltaV3.rowsPerLane = 2
                    guard outs[0].count == 5, outs[1].count == 5 else {
                        throw SelfTestFailure.message("a launch declined")
                    }
                    var differ: [MLXArray] = []
                    for (a, b) in zip(outs[0], outs[1]) {
                        differ.append(
                            (a.view(dtype: .uint32) .!= b.view(dtype: .uint32)).asType(.int32).sum())
                        values += a.size
                    }
                    let count = stacked(differ).sum()
                    eval(count)
                    try error.check()
                    mismatches += Int(count.item(Int32.self))
                }
            }
        } catch {
            Qwen35GatedDeltaV3.rowsPerLane = 2
            return (false, "\(error)")
        }
        guard mismatches == 0 else { return (false, "\(values) values, \(mismatches) mismatches") }
        // The derived kernels at 4 rows per lane (each against the stock
        // kernel at 4, now equal to it at 2 bit for bit).
        Qwen35GatedDeltaV3.rowsPerLane = 4
        Qwen35GDNVerifyStateSkip.prepare(layer: layer)
        Qwen35GDNReplayFused.prepare(layer: layer)
        Qwen35GDNFullAcceptStore.prepare(layer: layer)
        let after = (
            Qwen35GDNVerifyStateSkip.applies(to: layer), Qwen35GDNReplayFused.applies(to: layer),
            Qwen35GDNFullAcceptStore.verdict(layer: layer))
        Qwen35GatedDeltaV3.rowsPerLane = 2
        let stagedSame = Qwen35GDNReplayFused.stagedLive == stagedBefore
        if !stagedSame { Qwen35GDNReplayFused.stagedLive = stagedBefore }
        Memory.clearCache()
        let derivedSame = after.0 == before.0 && after.1 == before.1 && after.2 == before.2
            && stagedSame
        return (
            derivedSame,
            "\(values) values bitwise, 0 mismatches; derived kernels "
                + (derivedSame ? "same verdicts" : "DIFFERENT verdicts"))
    }

    private enum SelfTestFailure: Error {
        case message(String)
    }

    // MARK: Prompt prework tiles

    static let rowsMargin = 0.02

    /// The prompt prework's rows per threadgroup: 8 and 2 are checked bit for
    /// bit (`checkRowTile`), then raced against 4 at the real prompt shape
    /// (512 rows, the model's geometry and prompt qkv dtype;
    /// `Qwen35GDNPrework.raceRowTiles`), and the fastest is installed only
    /// when it beats 4 by more than `rowsMargin` in the race and in a
    /// confirmation race. One stderr line.
    static func runPromptRows() {
        guard promptRowsWanted, Qwen35GDNPrework.rowTiledEnabled,
            Qwen35GDNPrework.rowTileForced == nil,
            Qwen35GDNPrework.rowTileChoice == Qwen35GDNPrework.rowTile
        else { return }
        let start = DispatchTime.now().uptimeNanoseconds
        var tiles = [Qwen35GDNPrework.rowTile]
        var notes: [String] = []
        for rows in [8, 2] {
            let passed = Qwen35GDNPrework.checkRowTile(rows)
            notes.append("\(rows) rows " + (passed ? "passed" : "FAILED"))
            if passed { tiles.append(rows) }
        }
        var line = "bonsai exact forms trial (prompt prework rows): bitwise " + notes.joined(separator: ", ")
        defer {
            line += String(format: " (%.0f ms)\n", Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            FileHandle.standardError.write(line.data(using: .utf8)!)
        }
        guard tiles.count > 1 else {
            line += "; 4 kept"
            return
        }
        let first = Qwen35GDNPrework.raceRowTiles(tiles)
        guard first.count == tiles.count else {
            line += "; race failed, 4 kept"
            return
        }
        line += "; race us:" + zip(tiles, first).map { " \($0) " + String(format: "%.1f", $1) }.joined()
        var best = 0
        for i in tiles.indices.dropFirst() where first[i] < first[best] { best = i }
        guard best > 0, first[best] < first[0] * (1 - rowsMargin) else {
            line += "; 4 kept"
            return
        }
        let again = Qwen35GDNPrework.raceRowTiles([tiles[0], tiles[best]])
        guard again.count == 2, again[1] < again[0] * (1 - rowsMargin) else {
            line += "; \(tiles[best]) not confirmed" + (again.count == 2
                ? String(format: " (%.1f vs %.1f)", again[1], again[0]) : "") + ", 4 kept"
            return
        }
        Qwen35GDNPrework.rowTileChoice = tiles[best]
        line += String(format: "; confirmed %.1f vs %.1f; ", again[1], again[0]) + "\(tiles[best]) installed"
    }
}

// MARK: - Layer 0's input norm on the FP16 embedding

/// `RMSNorm` (FP32 weight) of FP16 rows in one launch. MLX's `rms_norm`
/// promotes FP16 rows to the weight's FP32 with an `astype` node, a copy
/// launch that writes the rows as FP32 (5 MB read and 10 MB written at 512
/// rows), and `rms_looped` then reads that copy. On the tensor route this
/// runs once per forward, at layer 0's input norm on the FP16 embedding
/// (every later input norm is inside the fused boundary): in each prompt
/// forward and in each verify window. This kernel is
/// `rms_looped` (1024 lanes x 4 reads, two passes at W = 5120) reading the
/// FP16 rows and widening each element at its read, as the fused boundary
/// reads its FP16 sum (`Qwen35FusedBoundaryQ8`): the same FP32 values summed
/// in the same order, the same `simd_sum` and threadgroup reduction,
/// `precise::rsqrt(t / axis_size + eps)` and `w * (x * inv)`, so every output
/// has the chain's bits and dtype, with one launch and 20 MB of traffic less.
///
/// Before first use a self-test compares it with the norm's own call (the
/// chain, cast included) on FP16 rows with the production weight and eps at
/// 1, 16, 128 and 512 rows (per-row scales from 0.05 to 30, outlier channels
/// 100x larger, an all-zero row from 16 rows up: `rsqrt(eps)`), bit for bit;
/// a mismatch or an MLX error keeps the chain. The kernel is per row, so it
/// takes any row count: a prompt's, a verify window's, a serial step's.
/// `BONSAI_HALF_INPUT_NORM=0` keeps the chain.
enum Qwen35HalfInputNorm {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_HALF_INPUT_NORM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The width `rms_looped` covers in two passes of 1024 lanes x 4.
    private static let width = 5120
    /// `rms_looped`'s threadgroup: its pipeline's maximum, 1024.
    private static let lanes = 1024
    nonisolated(unsafe) private static let axisSize = MLXArray(UInt32(width))

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdict: Bool?

    /// `norm(x)` for FP16 rows, or nil for the chain.
    static func apply(_ x: MLXArray, _ norm: RMSNorm) -> MLXArray? {
        guard enabled, ObjectIdentifier(type(of: norm)) == ObjectIdentifier(RMSNorm.self),
            x.dtype == .float16, x.ndim >= 2, x.dim(-1) == width, x.size > 0,
            norm.weight.dtype == .float32, norm.weight.shape == [width],
            verified(norm)
        else { return nil }
        return launch(x, norm)
    }

    private static func launch(_ x: MLXArray, _ norm: RMSNorm) -> MLXArray {
        kernel(
            [x, norm.weight, MLXArray(norm.eps), axisSize], template: [("W", width)],
            grid: (lanes * (x.size / width), 1, 1), threadGroup: (lanes, 1, 1),
            outputShapes: [x.shape], outputDTypes: [.float32])[0]
    }

    private static func verified(_ norm: RMSNorm) -> Bool {
        lock.withLock {
            if let verdict { return verdict }
            let (passed, summary) = selfTest(norm)
            verdict = passed
            FileHandle.standardError.write(
                ("bonsai half-input norm: " + summary
                    + (passed ? "; one launch\n" : "; chain kept\n")).data(using: .utf8)!)
            return passed
        }
    }

    private static func selfTest(_ norm: RMSNorm) -> (Bool, String) {
        var values = 0
        var mismatches = 0
        var failure: String? = nil
        do {
            try withError { error in
                let outlier = MLXArray(
                    (0 ..< width).map { $0 % 509 == 7 ? Float(100) : Float(1) })
                for (rows, seed) in [(1, 63), (16, 64), (128, 61), (512, 62)] {
                    let scale = MLXRandom.uniform(
                        Float(0.05) ..< Float(30), [1, rows, 1], key: MLXRandom.key(UInt64(seed)))
                    let zeroRow = (MLXArray(0 ..< rows) .== MLXArray(Int32(rows >= 16 ? rows / 3 : -1)))
                        .reshaped(1, rows, 1)
                    let x = which(
                        zeroRow, Float(0),
                        MLXRandom.normal([1, rows, width], key: MLXRandom.key(UInt64(seed + 100)))
                            * scale * outlier
                    ).asType(.float16)
                    let chain = norm(x)
                    let fused = launch(x, norm)
                    guard chain.dtype == fused.dtype, chain.shape == fused.shape else {
                        failure = "output \(fused.dtype) \(fused.shape) vs \(chain.dtype) \(chain.shape)"
                        return
                    }
                    let differ = (chain.view(dtype: .uint32) .!= fused.view(dtype: .uint32))
                        .asType(.int32).sum()
                    eval(differ)
                    try error.check()
                    values += chain.size
                    mismatches += Int(differ.item(Int32.self))
                }
            }
        } catch {
            failure = "\(error)"
        }
        if let failure { return (false, "self-test error: \(failure)") }
        let passed = mismatches == 0 && values > 0
        return (
            passed,
            "self-test \(passed ? "passed" : "FAILED"): \(values) values compared bitwise, "
                + "\(mismatches) mismatches")
    }

    // grid (1024 * rows, 1, 1), threadgroup (1024, 1, 1): one threadgroup per
    // row. Inputs: x half [rows, W], w float [W], eps, axis_size. Output: out
    // float [rows, W].
    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_rmsnorm_half_input",
        inputNames: ["x", "w", "eps", "axis_size"],
        outputNames: ["out"],
        source: """
            constexpr uint NR = 4;
            constexpr uint LS = 1024;
            constexpr uint NP = (uint(W) + LS * NR - 1) / (LS * NR);
            static_assert(W % 4 == 0 && W > 4096 && W <= 8192, "rms_looped width, two passes");
            const uint lid = thread_position_in_threadgroup.x;
            const uint row = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const size_t base = size_t(row) * size_t(W);

            threadgroup float local_sums[32];
            threadgroup float local_inv[1];

            // rms_looped's sum of squares of the promoted row: pass p covers
            // elements p * 4096 + 4 * lid + i, each widened at its read.
            float hv[NP * NR];
            float acc = 0;
            BONSAI_UNROLL for (uint p = 0; p < NP; p++) {
              const uint e0 = p * LS * NR + lid * NR;
              if (e0 + NR <= uint(W)) {
                const half4 xs = *(const device half4*)(x + base + e0);
                BONSAI_UNROLL for (uint i = 0; i < NR; i++) {
                  hv[p * NR + i] = float(xs[i]);
                  acc += hv[p * NR + i] * hv[p * NR + i];
                }
              }
            }
            acc = simd_sum(acc);
            if (sg == 0) {
              local_sums[lane] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              local_sums[sg] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float t = simd_sum(local_sums[lane]);
              if (lane == 0) {
                local_inv[0] = metal::precise::rsqrt(t / axis_size + eps);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv = local_inv[0];

            // rms_looped's output `w * (x * inv)`.
            BONSAI_UNROLL for (uint p = 0; p < NP; p++) {
              const uint e0 = p * LS * NR + lid * NR;
              if (e0 + NR <= uint(W)) {
                float4 o;
                BONSAI_UNROLL for (uint i = 0; i < NR; i++) {
                  o[i] = w[e0 + i] * (hv[p * NR + i] * inv);
                }
                *(device float4*)(out + base + e0) = o;
              }
            }
            """,
        header: "#define BONSAI_UNROLL _Pragma(\"clang loop unroll(full)\")\n",
        ensureRowContiguous: true)
}

// MARK: - The verify window's final add and norm in one launch

/// `(h + p, norm(h + p))` for a verify window's FP16 rows in one launch. The
/// pending path leaves the last layer's residual add for the forward's end
/// (`h + p`, a binary launch), and the final `RMSNorm` (FP32 weight) promotes
/// that sum to FP32 with an `astype` copy launch before `rms_looped` reads it:
/// three launches in every decode round. This kernel is
/// `Qwen35HalfInputNorm`'s `rms_looped` (1024 lanes x 4 reads, two passes at
/// W = 5120) reading the sum as the fused boundary forms it
/// (`Qwen35FusedBoundaryQ8`): the binary kernel's FP16 `half + half`, stored
/// as the FP16 sum, widened at its read. The same FP32 values are summed in
/// the same order, with the same `simd_sum` and threadgroup reduction,
/// `precise::rsqrt(t / axis_size + eps)` and `w * (x * inv)`, so both outputs
/// have the chain's bits and dtypes, from one launch.
///
/// Before first use a self-test compares both outputs with the chain (`h + p`
/// and the norm's own call on it, cast included) at 1, 16 and 64 rows with
/// the production weight and eps (per-row scales from 0.05 to 30, outlier
/// channels 100x larger, an all-zero row, and FP16 pairs whose sum overflows
/// to infinity), bit for bit; a mismatch or an MLX error keeps the chain.
/// `MLXFAST_ROUND_GLUE3=0` keeps the chain.
enum Qwen35FinalNormGlue {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_ROUND_GLUE3"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let width = 5120
    private static let lanes = 1024
    nonisolated(unsafe) private static let axisSize = MLXArray(UInt32(width))

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdict: Bool?

    /// `(h + p, norm(h + p))` for FP16 rows of one shape, or nil for the chain.
    static func apply(_ h: MLXArray, _ p: MLXArray, _ norm: RMSNorm)
        -> (sum: MLXArray, normed: MLXArray)?
    {
        guard enabled, ObjectIdentifier(type(of: norm)) == ObjectIdentifier(RMSNorm.self),
            h.dtype == .float16, p.dtype == .float16, h.shape == p.shape,
            h.ndim >= 2, h.dim(-1) == width, h.size > 0,
            norm.weight.dtype == .float32, norm.weight.shape == [width],
            verified(norm)
        else { return nil }
        return launch(h, p, norm)
    }

    private static func launch(_ h: MLXArray, _ p: MLXArray, _ norm: RMSNorm)
        -> (sum: MLXArray, normed: MLXArray)
    {
        let out = kernel(
            [h, p, norm.weight, MLXArray(norm.eps), axisSize], template: [("W", width)],
            grid: (lanes * (h.size / width), 1, 1), threadGroup: (lanes, 1, 1),
            outputShapes: [h.shape, h.shape], outputDTypes: [.float16, .float32])
        return (out[0], out[1])
    }

    private static func verified(_ norm: RMSNorm) -> Bool {
        lock.withLock {
            if let verdict { return verdict }
            let (passed, summary) = selfTest(norm)
            verdict = passed
            FileHandle.standardError.write(
                ("qwen35 verify final add and norm (GLUE3): " + summary
                    + (passed ? "; one launch\n" : "; chain kept\n")).data(using: .utf8)!)
            return passed
        }
    }

    private static func selfTest(_ norm: RMSNorm) -> (Bool, String) {
        var values = 0
        var mismatches = 0
        var failure: String? = nil
        do {
            try withError { error in
                let column = MLXArray(0 ..< width).reshaped(1, 1, width)
                let outlier = which(column % 509 .== 7, Float(100), Float(1))
                for (rows, seed) in [(1, 71), (16, 72), (64, 73)] {
                    let row = MLXArray(0 ..< rows).reshaped(1, rows, 1)
                    let scale = MLXRandom.uniform(
                        Float(0.05) ..< Float(30), [1, rows, 1], key: MLXRandom.key(UInt64(seed)))
                    let zeroRow = row .== MLXArray(Int32(rows >= 16 ? rows / 3 : -1))
                    // FP16 pairs whose sum overflows: two channels of one row.
                    let overflow = (row .== MLXArray(Int32(rows >= 16 ? 5 : -1)))
                        .&& ((column .== MLXArray(Int32(11))) .|| (column .== MLXArray(Int32(4100))))
                    let a = MLXRandom.normal([1, rows, width], key: MLXRandom.key(UInt64(seed + 100)))
                        * scale * outlier
                    let b = MLXRandom.normal([1, rows, width], key: MLXRandom.key(UInt64(seed + 200)))
                        * scale
                    let h = which(overflow, Float(65504), which(zeroRow, Float(0), a)).asType(.float16)
                    let p = which(overflow, Float(65504), which(zeroRow, Float(0), b)).asType(.float16)
                    let chainSum = h + p
                    let chain = norm(chainSum)
                    let fused = launch(h, p, norm)
                    guard chainSum.dtype == fused.sum.dtype, chainSum.shape == fused.sum.shape,
                        chain.dtype == fused.normed.dtype, chain.shape == fused.normed.shape
                    else {
                        failure = "outputs \(fused.sum.dtype) \(fused.normed.dtype) \(fused.normed.shape)"
                            + " vs \(chainSum.dtype) \(chain.dtype) \(chain.shape)"
                        return
                    }
                    let differ =
                        (chainSum.view(dtype: .uint16) .!= fused.sum.view(dtype: .uint16))
                        .asType(.int32).sum()
                        + (chain.view(dtype: .uint32) .!= fused.normed.view(dtype: .uint32))
                        .asType(.int32).sum()
                    eval(differ)
                    try error.check()
                    values += chainSum.size + chain.size
                    mismatches += Int(differ.item(Int32.self))
                }
            }
        } catch {
            failure = "\(error)"
        }
        if let failure { return (false, "self-test error: \(failure)") }
        let passed = mismatches == 0 && values > 0
        return (
            passed,
            "self-test \(passed ? "passed" : "FAILED"): \(values) values compared bitwise "
                + "(FP16 sums and FP32 norms), \(mismatches) mismatches")
    }

    // grid (1024 * rows, 1, 1), threadgroup (1024, 1, 1): one threadgroup per
    // row. Inputs: xa, xb half [rows, W], w float [W], eps, axis_size.
    // Outputs: sum half [rows, W], out float [rows, W].
    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_final_add_norm",
        inputNames: ["xa", "xb", "w", "eps", "axis_size"],
        outputNames: ["sum", "out"],
        source: """
            constexpr uint NR = 4;
            constexpr uint LS = 1024;
            constexpr uint NP = (uint(W) + LS * NR - 1) / (LS * NR);
            static_assert(W % 4 == 0 && W > 4096 && W <= 8192, "rms_looped width, two passes");
            const uint lid = thread_position_in_threadgroup.x;
            const uint row = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const size_t base = size_t(row) * size_t(W);

            threadgroup float local_sums[32];
            threadgroup float local_inv[1];

            // The binary kernel's FP16 add, stored, and rms_looped's sum of
            // squares of its promotion: pass p covers elements
            // p * 4096 + 4 * lid + i, each widened at its read.
            float hv[NP * NR];
            float acc = 0;
            BONSAI_UNROLL for (uint p = 0; p < NP; p++) {
              const uint e0 = p * LS * NR + lid * NR;
              if (e0 + NR <= uint(W)) {
                const half4 hs = *(const device half4*)(xa + base + e0)
                    + *(const device half4*)(xb + base + e0);
                *(device half4*)(sum + base + e0) = hs;
                BONSAI_UNROLL for (uint i = 0; i < NR; i++) {
                  hv[p * NR + i] = float(hs[i]);
                  acc += hv[p * NR + i] * hv[p * NR + i];
                }
              }
            }
            acc = simd_sum(acc);
            if (sg == 0) {
              local_sums[lane] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
              local_sums[sg] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              const float t = simd_sum(local_sums[lane]);
              if (lane == 0) {
                local_inv[0] = metal::precise::rsqrt(t / axis_size + eps);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv = local_inv[0];

            // rms_looped's output `w * (x * inv)`.
            BONSAI_UNROLL for (uint p = 0; p < NP; p++) {
              const uint e0 = p * LS * NR + lid * NR;
              if (e0 + NR <= uint(W)) {
                float4 o;
                BONSAI_UNROLL for (uint i = 0; i < NR; i++) {
                  o[i] = w[e0 + i] * (hv[p * NR + i] * inv);
                }
                *(device float4*)(out + base + e0) = o;
              }
            }
            """,
        header: "#define BONSAI_UNROLL _Pragma(\"clang loop unroll(full)\")\n",
        ensureRowContiguous: true)
}

// MARK: - The head's input norm, rotation and quantization in one launch

/// The capture verify's last residual add with the vocabulary head's whole
/// input in one launch. With GLUE3 the round ends in the add-and-norm kernel
/// (the FP16 sum and the FP32 norm) and the head's quantizing rotation
/// (`bonsai_signed_hadamard_1024_q8_blocks`: signs, 1024-block Hadamard,
/// q8 per 128-group), whose codes, scales and scaled sums the int8 head
/// matmul reads; the FP32 norm has no other reader. The fused verify
/// boundary's per-block kernel (`Qwen35BoundaryBlocks`, which every layer
/// boundary of the window runs) is that chain's text: the FP16 add and
/// `rms_looped`'s reduction, `w * (h * inv)`, the signs multiplied after the
/// norm (`PRESIGNED` 0, the head's own form), the butterflies in the stock
/// order and the stock quantization, with the same `PERM`, `SIGNED` and
/// codes dtype as the head's rotation. Here it takes the final norm's weight
/// and eps and the head's signs, and the head reads its activation through
/// the same narrow int8 matmul (`sharedHadamardProjectionsQuantized`), inside
/// the head's top-two capture as before. Only a full 16-row window on that
/// route takes it (the only rows the route takes); every other forward keeps
/// the chain.
///
/// Before first use a self-test runs the live chain (GLUE3 where it applies,
/// the head's quantizing rotation, the head call) and this launch on the
/// production weight, eps and signs, 16 rows with per-row scales from 0.001
/// to 60, outlier channels, a zero row and an FP16 sum that overflows, and
/// compares the sums, codes, scales, scaled sums and logits bit for bit; a
/// mismatch or an MLX error keeps the chain. `MLXFAST_ROUND_GLUE4=0` keeps it.
enum Qwen35HeadInputGlue {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_ROUND_GLUE4"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let width = 5120
    private static let rows = 16

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdict: Bool?

    /// The window's FP16 sum and the head's quantized input.
    struct Input {
        let sum: MLXArray
        let activation: SignedBlockHadamard.Int8Activation
        let head: HadamardQuantizedLinear

        /// The head's logits: its route's matmul over `activation`, or its own
        /// call on `normed()` where that route declines.
        func logits(_ normed: () -> MLXArray) -> MLXArray {
            sharedHadamardProjectionsQuantized(
                activation, leading: Array(sum.shape.dropLast()), [head], widenOutput: true)?.first
                ?? head(normed())
        }
    }

    /// `h + p` and the quantized input of `lmHead(norm(h + p))`, or nil for the chain.
    static func apply(_ h: MLXArray, _ p: MLXArray, _ norm: RMSNorm, _ lmHead: Linear?) -> Input? {
        guard enabled, let head = lmHead as? HadamardQuantizedLinear,
            ObjectIdentifier(type(of: norm)) == ObjectIdentifier(RMSNorm.self),
            norm.weight.dtype == .float32, norm.weight.shape == [width],
            head.transform.width == width, head.transform.blockSize == 1024,
            head.transform.signVector.dtype == .float32,
            h.dtype == .float16, p.dtype == .float16, h.shape == p.shape, h.ndim >= 2,
            h.dim(-1) == width, h.size == rows * width,
            sharedHadamardTensorRouteTakesNarrowInt8([head], rows: rows),
            verified(norm, head), let out = launch(h, p, norm, head)
        else { return nil }
        return Input(sum: out.h, activation: out.activation, head: head)
    }

    private static func launch(
        _ h: MLXArray, _ p: MLXArray, _ norm: RMSNorm, _ head: HadamardQuantizedLinear
    ) -> Qwen35FusedBoundaryQ8.Output? {
        Qwen35BoundaryBlocks.launchUnsigned(
            h, p, gain: norm.weight, signs: head.transform.signVector, eps: norm.eps)
    }

    private static func verified(_ norm: RMSNorm, _ head: HadamardQuantizedLinear) -> Bool {
        lock.withLock {
            if let verdict { return verdict }
            let (passed, summary) = selfTest(norm, head)
            verdict = passed
            FileHandle.standardError.write(
                ("qwen35 verify head input norm, rotation and quantization (GLUE4): " + summary
                    + (passed ? "; one launch\n" : "; chain kept\n")).data(using: .utf8)!)
            return passed
        }
    }

    private static func selfTest(_ norm: RMSNorm, _ head: HadamardQuantizedLinear) -> (Bool, String) {
        var values = 0
        var mismatches = 0
        var failure: String? = nil
        do {
            try withError { error in
                let column = MLXArray(0 ..< width).reshaped(1, 1, width)
                let row = MLXArray(0 ..< rows).reshaped(1, rows, 1)
                let outlier = which(column % 509 .== 7, Float(100), Float(1))
                for seed in [91, 92] {
                    let scale = MLXRandom.uniform(
                        Float(0.001) ..< Float(60), [1, rows, 1], key: MLXRandom.key(UInt64(seed)))
                    let zeroRow = row .== MLXArray(Int32(seed % 7))
                    // FP16 pairs whose sum overflows: two channels of one row.
                    let overflow = (row .== MLXArray(Int32(11)))
                        .&& ((column .== MLXArray(Int32(11))) .|| (column .== MLXArray(Int32(4100))))
                    let a = MLXRandom.normal([1, rows, width], key: MLXRandom.key(UInt64(seed + 100)))
                        * scale * outlier
                    let b = MLXRandom.normal([1, rows, width], key: MLXRandom.key(UInt64(seed + 200)))
                        * scale
                    let h = which(overflow, Float(65504), which(zeroRow, Float(0), a)).asType(.float16)
                    let p = which(overflow, Float(65504), which(zeroRow, Float(0), b)).asType(.float16)
                    // The live chain: the forward's final add, the head's call.
                    let glued = Qwen35FinalNormGlue.apply(h, p, norm)
                    let chainSum = glued?.sum ?? (h + p)
                    let chainNormed = glued?.normed ?? norm(chainSum)
                    guard
                        let chain = head.transform.forwardInt8(
                            chainNormed.reshaped(rows, width), gdnLayout: nil, preSigned: false,
                            groupSize: 128),
                        let fused = launch(h, p, norm, head)
                    else {
                        failure = "a launch declined"
                        return
                    }
                    let input = Input(sum: fused.h, activation: fused.activation, head: head)
                    let pairs = [
                        (chainSum, fused.h), (chain.codes, fused.activation.codes),
                        (chain.scales, fused.activation.scales),
                        (chain.scaledSums, fused.activation.scaledSums),
                        (head(chainNormed), input.logits { chainNormed }),
                    ]
                    var differ: [MLXArray] = []
                    for (x, y) in pairs {
                        guard x.dtype == y.dtype, x.shape == y.shape else {
                            failure = "output \(y.dtype) \(y.shape) vs \(x.dtype) \(x.shape)"
                            return
                        }
                        let bits: DType =
                            x.dtype == .float16 ? .uint16 : x.dtype == .float32 ? .uint32 : x.dtype
                        differ.append((x.view(dtype: bits) .!= y.view(dtype: bits)).asType(.int32).sum())
                        values += x.size
                    }
                    let count = stacked(differ).sum()
                    eval(count)
                    try error.check()
                    mismatches += Int(count.item(Int32.self))
                }
            }
        } catch {
            failure = "\(error)"
        }
        if let failure { return (false, "self-test error: \(failure)") }
        let passed = mismatches == 0 && values > 0
        return (
            passed,
            "self-test \(passed ? "passed" : "FAILED"): \(values) values compared bitwise "
                + "(FP16 sums, codes, scales, scaled sums, logits), \(mismatches) mismatches")
    }
}

extension Qwen35BoundaryBlocks {
    /// `launch` for an unsigned gain, the signs multiplied after the norm: the
    /// same kernel and grid whichever form the load-time trial leaves the
    /// layer boundaries on; nil when it is off or failed its self-test.
    static func launchUnsigned(
        _ x: MLXArray, _ r: MLXArray, gain: MLXArray, signs: MLXArray, eps: Float
    ) -> Qwen35FusedBoundaryQ8.Output? {
        let width = Qwen35FusedBoundaryQ8.width
        guard enabled, live, x.size % width == 0 else { return nil }
        let rows = x.size / width
        guard rows > 0, rows < BonsaiPromptWidth.minimumRows else { return nil }
        let groupShape = [rows, width / 128]
        let template: [(String, any KernelTemplateArg)] = [
            ("W", width), ("PRESIGNED", false),
            ("PERM", Qwen35TensorPackedMatmul.support == .staged8),
            ("MPERM", Qwen35TensorPackedMatmul.rowTiledConstants && rows % 64 == 0),
            ("SIGNED", Qwen35TensorPackedMatmul.signedCodes),
        ]
        let outs = kernel(
            [x, r, gain, signs, MLXArray(eps), axisSize], template: template,
            grid: (threads * rows * (width / 1024), 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [x.shape, [rows, width], groupShape, groupShape],
            outputDTypes: [.float16, Qwen35TensorPackedMatmul.codesDType, .float32, .float32])
        return Qwen35FusedBoundaryQ8.Output(
            h: outs[0], normed: nil,
            activation: SignedBlockHadamard.Int8Activation(
                codes: outs[1], scales: outs[2], scaledSums: outs[3]))
    }
}

// MARK: - Layer 0's input norm, rotation and quantization in one launch

/// Layer 0's input in a verify window in one launch. The window's FP16
/// embedding rows go through the half-input norm (`Qwen35HalfInputNorm`:
/// `rms_looped`, each element widened at its read), then through the input
/// projections' quantizing rotation (`bonsai_signed_hadamard_1024_q8_blocks`
/// at width 5120: signs, 1024-block Hadamard, q8 per 128-group), whose codes,
/// scales and scaled sums the qkv|z int8 matmul reads; the GDN's b|a read the
/// FP32 norm. The verify boundary's per-block kernel (`Qwen35BoundaryBlocks`,
/// its `_normed` form at every other GDN boundary of the window) is that chain
/// after its FP16 residual add: `rms_looped`'s reduction over the widened row,
/// `w * (h * inv)`, the signs multiplied after the norm, the butterflies in
/// the stock order and the stock quantization. Here it runs without the add:
/// the residual read is the row read alone and the sum's store is dropped (a
/// zero addend would turn -0.0 into +0.0), so every element is widened from
/// the embedding's own bits. One launch writes the codes, scales, scaled sums
/// and the FP32 norm, with layer 0's weight and eps and qkv|z's signs. Only a
/// full 16-row window on the int8 narrow route takes it (the only rows that
/// route takes); every other forward keeps the chain.
///
/// Before first use a self-test runs the live chain (the half-input norm where
/// it applies, the quantizing rotation, the qkv|z matmul) and this launch on
/// layer 0's weight, eps and signs: 16 rows with per-row scales from 1e-5 (FP16
/// subnormals) to 60, outlier channels, a zero row, a row of -0.0 and scattered
/// -0.0 elements; it compares the FP32 norm, codes, scales, scaled sums and
/// both projections bit for bit. A mismatch or an MLX error keeps the chain.
/// `MLXFAST_ROUND_LAYER0=0` keeps it.
enum Qwen35Layer0InputGlue {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_ROUND_LAYER0"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let width = 5120
    private static let rows = 16

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdict: Bool?

    /// Layer 0's FP32 norm of `x` and qkv|z's quantized rotation of it (`h` is
    /// `x` itself), or nil for the chain.
    static func apply(
        _ x: MLXArray, _ norm: RMSNorm, _ siblings: [HadamardQuantizedLinear]?
    ) -> Qwen35FusedBoundaryQ8.Output? {
        guard enabled, let siblings, let transform = siblings.first?.transform,
            ObjectIdentifier(type(of: norm)) == ObjectIdentifier(RMSNorm.self),
            norm.weight.dtype == .float32, norm.weight.shape == [width],
            transform.width == width, transform.blockSize == 1024,
            transform.signVector.dtype == .float32,
            x.dtype == .float16, x.ndim >= 2, x.dim(-1) == width, x.size == rows * width,
            sharedHadamardTensorRouteTakesNarrowInt8(siblings, rows: rows),
            verified(norm, siblings), let out = launch(x, norm, transform)
        else { return nil }
        return out
    }

    private static func launch(
        _ x: MLXArray, _ norm: RMSNorm, _ transform: SignedBlockHadamard
    ) -> Qwen35FusedBoundaryQ8.Output? {
        Qwen35BoundaryBlocks.launchRows(
            x, gain: norm.weight, signs: transform.signVector, eps: norm.eps)
    }

    private static func verified(_ norm: RMSNorm, _ siblings: [HadamardQuantizedLinear]) -> Bool {
        lock.withLock {
            if let verdict { return verdict }
            let (passed, summary) = selfTest(norm, siblings)
            verdict = passed
            FileHandle.standardError.write(
                ("qwen35 verify layer 0 input norm, rotation and quantization (LAYER0): " + summary
                    + (passed ? "; one launch\n" : "; chain kept\n")).data(using: .utf8)!)
            return passed
        }
    }

    private static func selfTest(_ norm: RMSNorm, _ siblings: [HadamardQuantizedLinear])
        -> (Bool, String)
    {
        let transform = siblings[0].transform
        var values = 0
        var mismatches = 0
        var negativeZeros = 0
        var normedNegativeZeros = 0
        var failure: String? = nil
        do {
            try withError { error in
                let column = MLXArray(0 ..< width).reshaped(1, 1, width)
                let row = MLXArray(0 ..< rows).reshaped(1, rows, 1)
                let outlier = which(column % 509 .== 7, Float(100), Float(1))
                for seed in [95, 96] {
                    // Per-row scales from 0.001 to 60, one row at 1e-5 (FP16
                    // subnormals), a zero row, a row of -0.0, and -0.0 scattered
                    // over every block and group of the even rows.
                    let scale = which(
                        row .== MLXArray(Int32(seed % 5)), Float(1e-5),
                        MLXRandom.uniform(
                            Float(0.001) ..< Float(60), [1, rows, 1], key: MLXRandom.key(UInt64(seed))))
                    let zeroRow = row .== MLXArray(Int32(seed % 7 + 6))
                    let negativeZero =
                        (row .== MLXArray(Int32(seed % 3 + 13)))
                        .|| ((column % 37 .== MLXArray(Int32(seed % 37))) .&& (row % 2 .== 0))
                    let a = MLXRandom.normal([1, rows, width], key: MLXRandom.key(UInt64(seed + 100)))
                        * scale * outlier
                    let x = which(negativeZero, -Float(0), which(zeroRow, Float(0), a))
                        .asType(.float16)
                    // The live chain: layer 0's norm, the route's quantizing
                    // rotation, the qkv|z matmul.
                    let chainNormed = Qwen35HalfInputNorm.apply(x, norm) ?? norm(x)
                    guard
                        let chain = transform.forwardInt8(
                            chainNormed.reshaped(rows, width), gdnLayout: nil, preSigned: false,
                            groupSize: 128),
                        let chainProjections = sharedHadamardProjections(chainNormed, siblings),
                        let fused = launch(x, norm, transform), let fusedNormed = fused.normed,
                        let fusedProjections = sharedHadamardProjectionsQuantized(
                            fused.activation, leading: Array(x.shape.dropLast()), siblings),
                        chainProjections.count == fusedProjections.count
                    else {
                        failure = "a launch declined"
                        return
                    }
                    let pairs =
                        [
                            (chainNormed, fusedNormed), (chain.codes, fused.activation.codes),
                            (chain.scales, fused.activation.scales),
                            (chain.scaledSums, fused.activation.scaledSums),
                        ] + Array(zip(chainProjections, fusedProjections))
                    var differ: [MLXArray] = []
                    for (c, f) in pairs {
                        guard c.dtype == f.dtype, c.shape == f.shape else {
                            failure = "output \(f.dtype) \(f.shape) vs \(c.dtype) \(c.shape)"
                            return
                        }
                        let bits: DType =
                            c.dtype == .float16 ? .uint16 : c.dtype == .float32 ? .uint32 : c.dtype
                        differ.append((c.view(dtype: bits) .!= f.view(dtype: bits)).asType(.int32).sum())
                        values += c.size
                    }
                    let count = stacked(differ).sum()
                    let zeros = (x.view(dtype: .uint16) .== MLXArray(UInt16(0x8000))).asType(.int32).sum()
                    let normedZeros = (chainNormed.view(dtype: .uint32) .== MLXArray(UInt32(0x8000_0000)))
                        .asType(.int32).sum()
                    eval(count, zeros, normedZeros)
                    try error.check()
                    mismatches += Int(count.item(Int32.self))
                    negativeZeros += Int(zeros.item(Int32.self))
                    normedNegativeZeros += Int(normedZeros.item(Int32.self))
                }
            }
        } catch {
            failure = "\(error)"
        }
        if let failure { return (false, "self-test error: \(failure)") }
        let passed = mismatches == 0 && values > 0 && negativeZeros > 0
        return (
            passed,
            "self-test \(passed ? "passed" : "FAILED"): \(values) values compared bitwise "
                + "(FP32 norms, codes, scales, scaled sums, projections; \(negativeZeros) -0.0 "
                + "inputs, \(normedNegativeZeros) -0.0 norms), \(mismatches) mismatches")
    }
}

extension Qwen35BoundaryBlocks {
    /// The `_normed` kernel's text without the residual add: the row read
    /// alone (`xa`) and no sum store. Nil (the caller keeps the chain) unless
    /// the add and the store it removes occur exactly once.
    private static let rowsKernel: MLXFast.MLXFastKernel? = {
        let add = "+ *(const device half4*)(xb + base + e0)"
        let store = "*(device half4*)(hout + base + e0) = hs;"
        guard source.components(separatedBy: add).count == 2,
            source.components(separatedBy: store).count == 2
        else { return nil }
        let text = source.replacingOccurrences(of: add, with: "")
            .replacingOccurrences(of: store, with: "")
        guard !text.contains("xb"), !text.contains("hout") else { return nil }
        return MLXFast.metalKernel(
            name: "bonsai_boundary_q8_blocks_rows_normed",
            inputNames: ["xa", "w", "signs", "eps", "axis_size"],
            outputNames: ["codes", "qscale", "qsum", "nout"],
            source: "#define BONSAI_STORE_NORMED(e, n) nout[base + (e)] = (n)\n" + text,
            header: header,
            ensureRowContiguous: true)
    }()

    /// `launchUnsigned` for rows normed as they are (no residual add), the
    /// FP32 norm written too; `h` is `x` itself. Nil when it is off or failed
    /// its self-test.
    static func launchRows(
        _ x: MLXArray, gain: MLXArray, signs: MLXArray, eps: Float
    ) -> Qwen35FusedBoundaryQ8.Output? {
        let width = Qwen35FusedBoundaryQ8.width
        guard enabled, live, let rowsKernel, x.size % width == 0 else { return nil }
        let rows = x.size / width
        guard rows > 0, rows < BonsaiPromptWidth.minimumRows else { return nil }
        let groupShape = [rows, width / 128]
        let template: [(String, any KernelTemplateArg)] = [
            ("W", width), ("PRESIGNED", false),
            ("PERM", Qwen35TensorPackedMatmul.support == .staged8),
            ("MPERM", Qwen35TensorPackedMatmul.rowTiledConstants && rows % 64 == 0),
            ("SIGNED", Qwen35TensorPackedMatmul.signedCodes),
        ]
        let outs = rowsKernel(
            [x, gain, signs, MLXArray(eps), axisSize], template: template,
            grid: (threads * rows * (width / 1024), 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, width], groupShape, groupShape, x.shape],
            outputDTypes: [Qwen35TensorPackedMatmul.codesDType, .float32, .float32, .float32])
        return Qwen35FusedBoundaryQ8.Output(
            h: x, normed: outs[3],
            activation: SignedBlockHadamard.Int8Activation(
                codes: outs[0], scales: outs[1], scaledSums: outs[2]))
    }
}

// MARK: - The packed embedding lookup in one launch

/// A lookup's embedding rows in one launch. The packed lookup is a chain of
/// three: the row gather of the codes, scales and biases
/// (`bonsai_embedding_row_gather`), MLX's `affine_dequantize` (2 bits, group
/// 128, FP16) and the inverse signed rotation
/// (`bonsai_signed_hadamard_1024_inv`). Every speculative round runs it
/// twice: the drafter block's anchor row and the 16-row verify window. This
/// kernel is the inverse rotation's text with one load replaced. Each thread
/// reads its token's packed byte, scale and bias where the table holds them.
/// It forms each value as `affine_dequantize` does: the byte as `uint val`,
/// `uint8_t d = (val >> (bits * i)) & 0x03` and `scale * d + bias`, all in
/// FP16. It then widens the value, as the rotation's load widens the stored
/// FP16 row. The butterflies and the signed store are the inverse rotation's
/// own text, so the output has the chain's bits.
///
/// Before first use a self-test runs the lookup's own chain on the production
/// table and compares every bit: 1-row and 16-row lookups (the first, middle
/// and last ids, negative ids, random ids), a 17-row lookup, and every
/// vocabulary row in chunks. A mismatch or an MLX error keeps the chain.
/// Lookups of prompt width keep the chain. `MLXFAST_EMB_ROWS=0` keeps it.
enum Qwen35EmbeddingRows {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_EMB_ROWS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let chunk = 4096
    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]

    static func install() {
        guard enabled, kernel != nil else { return }
        SignedBlockHadamard.fusedLookup = {
            codes, scales, biases, ids, signs, blockSize, groupSize, bits, chain in
            guard blockSize == 1024, groupSize == 128, bits == 2, ids.ndim == 1, ids.size > 0,
                ids.size < BonsaiPromptWidth.minimumRows, ids.dtype == .int32 || ids.dtype == .uint32,
                codes.dtype == .uint32, codes.ndim == 2, codes.dim(0) <= Int(Int32.max),
                scales.dtype == .float16, biases.dtype == .float16, scales.ndim == 2,
                biases.shape == scales.shape, scales.dim(0) == codes.dim(0),
                signs.dtype == .float32, signs.ndim == 1, signs.size % 1024 == 0,
                codes.dim(1) * 16 == signs.size, scales.dim(1) * 128 == signs.size,
                verified(codes, scales, biases, ids.dtype, signs, chain)
            else { return nil }
            return launch(codes, scales, biases, ids, signs)
        }
    }

    private static func launch(
        _ codes: MLXArray, _ scales: MLXArray, _ biases: MLXArray, _ ids: MLXArray,
        _ signs: MLXArray
    ) -> MLXArray {
        let (rows, width) = (ids.size, signs.size)
        return kernel!(
            [codes, scales, biases, ids, signs],
            template: [
                ("OutT", DType.float16), ("W", width), ("BPR", width / 1024),
                ("WC", codes.dim(1)), ("GC", scales.dim(1)), ("V", codes.dim(0)),
                ("PRESIGNED", 1), ("GR", 1), ("GKH", 1), ("GD", 1), ("QSIM", 0),
            ],
            grid: (64 * rows * (width / 1024), 1, 1), threadGroup: (64, 1, 1),
            outputShapes: [[rows, width]], outputDTypes: [.float16])[0]
    }

    private static func verified(
        _ codes: MLXArray, _ scales: MLXArray, _ biases: MLXArray, _ idType: DType,
        _ signs: MLXArray, _ chain: (MLXArray) -> MLXArray
    ) -> Bool {
        lock.withLock {
            let key = "\(idType) \(codes.shape) \(scales.shape) \(signs.size)"
            if let verdict = verdicts[key] { return verdict }
            let (passed, summary) = selfTest(codes, scales, biases, idType, signs, chain)
            verdicts[key] = passed
            FileHandle.standardError.write(
                ("qwen35 embedding rows (EMBROWS): " + summary
                    + (passed ? "; one launch\n" : "; chain kept\n")).data(using: .utf8)!)
            return passed
        }
    }

    private static func selfTest(
        _ codes: MLXArray, _ scales: MLXArray, _ biases: MLXArray, _ idType: DType,
        _ signs: MLXArray, _ chain: (MLXArray) -> MLXArray
    ) -> (Bool, String) {
        let vocab = codes.dim(0)
        var values = 0
        var mismatches = 0
        var failure: String? = nil
        let start = Date()
        do {
            try withError { error in
                // The FP16 values in which the kernel and the chain differ.
                func differ(_ ids: MLXArray) -> MLXArray {
                    let ids = ids.asType(idType)
                    let chained = chain(ids)
                    let fused = launch(codes, scales, biases, ids, signs)
                    guard chained.dtype == fused.dtype, chained.shape == fused.shape else {
                        failure = "output \(fused.dtype) \(fused.shape) vs \(chained.dtype) \(chained.shape)"
                        return MLXArray(Int32(0))
                    }
                    values += chained.size
                    return (chained.view(dtype: .uint16) .!= fused.view(dtype: .uint16))
                        .asType(.int32).sum()
                }
                func count(_ differences: [MLXArray]) throws {
                    let total = differences.dropFirst().reduce(differences[0], +)
                    eval(total)
                    try error.check()
                    mismatches += Int(total.item(Int32.self))
                }
                let v = Int32(vocab)
                let random = MLXRandom.randInt(
                    Int32(0) ..< v, [64 + 65 * 16 + 17], key: MLXRandom.key(0x656d_6272))
                var edge: [Int32] = [0, v - 1, v / 2, 1, v - 2]
                if idType == .int32 { edge += [-1, -v] }
                var differences = [Int32(0), v - 1, v / 2].map { differ(MLXArray([$0])) }
                for k in 0 ..< 64 { differences.append(differ(random[k ..< (k + 1)])) }
                differences.append(
                    differ(concatenated([MLXArray(edge), random[64 ..< (80 - edge.count)]], axis: 0)))
                for k in 1 ... 64 { differences.append(differ(random[(64 + 16 * k) ..< (80 + 16 * k)])) }
                differences.append(differ(random[(64 + 65 * 16)...]))
                guard failure == nil else { return }
                try count(differences)
                // Every vocabulary row, in chunks.
                var first = 0
                while first < vocab, failure == nil {
                    let n = min(chunk, vocab - first)
                    let ids = MLXArray((first ..< (first + n)).map { Int32($0) })
                    try count([differ(ids)])
                    first += n
                }
            }
        } catch {
            failure = "\(error)"
        }
        if let failure { return (false, "self-test error: \(failure)") }
        let passed = mismatches == 0 && values > 0
        return (
            passed,
            "self-test \(passed ? "passed" : "FAILED"): \(values) values compared bitwise "
                + "(1-row, 16-row and 17-row lookups, every vocabulary row), \(mismatches) mismatches, "
                + String(format: "%.0f ms", Date().timeIntervalSince(start) * 1000))
    }

    /// MLX `affine_dequantize` (quantized.h) at 2 bits for the value in
    /// column `col` of one packed row, with the kernel's own types and
    /// expression: the byte as `uint val`, `uint8_t d`, `T scale`, `T bias`,
    /// and `scale * d + bias` in `T`.
    private static let dequantizeHeader = """
        template <typename T>
        METAL_FUNC T bonsai_affine_dequantize_2(
            const device uint8_t* w, const device T* scales, const device T* biases, uint col) {
          constexpr int bits = 2;
          const int i = int(col % 4);
          uint val = w[col / 4];
          uint8_t d = (val >> (bits * i)) & 0x03;
          T scale = scales[col / 128];
          T bias = biases[col / 128];
          return scale * d + bias;
        }

        """

    // grid (64 * rows * BPR, 1, 1), threadgroup (64, 1, 1): one threadgroup
    // per 1024-wide block of a row, as the inverse rotation. Inputs: codes
    // uint32 [V, WC], scale_table and bias_table half [V, GC], ids [rows],
    // signs float [W]. Output: out half [rows, W].
    private static let kernel: MLXFast.MLXFastKernel? = {
        let source = Qwen35FusedHadamard.source
        let rowStart = "threadgroup float buf[N];"
        let load = "float v = float(inp[rowbase + src]);"
        let store = "out[rowbase + bcol + uint(index + r)] = OutT(buf[index + r] * 0.03125f);"
        guard [rowStart, load, store].allSatisfy({ source.components(separatedBy: $0).count == 2 })
        else { return nil }
        let text =
            source
            .replacingOccurrences(
                of: rowStart,
                with: """
                    long eid = long(ids[row]);
                    if (eid < 0) eid += long(V);
                    const device uint8_t* erow = (const device uint8_t*)codes + ulong(eid) * ulong(4 * WC);
                    const device half* escale = scale_table + ulong(eid) * ulong(GC);
                    const device half* ebias = bias_table + ulong(eid) * ulong(GC);
                    \(rowStart)
                    """)
            .replacingOccurrences(
                of: load, with: "float v = float(bonsai_affine_dequantize_2(erow, escale, ebias, col));")
            .replacingOccurrences(
                of: store,
                with: "out[rowbase + bcol + uint(index + r)] = "
                    + "OutT((buf[index + r] * 0.03125f) * signs[bcol + uint(index + r)]);")
        guard !text.contains("inp[") else { return nil }
        return MLXFast.metalKernel(
            name: "bonsai_embedding_rows_hadamard_1024_inv",
            inputNames: ["codes", "scale_table", "bias_table", "ids", "signs"],
            outputNames: ["out"],
            source: Qwen35IO32.narrow(text, count: 3, "bonsai_embedding_rows_hadamard_1024_inv"),
            header: Qwen35FusedHadamard.header + dequantizeHeader,
            ensureRowContiguous: true)
    }()
}

// MARK: - The GDN output's gated norm in the out_proj rotation's read

/// Each GDN layer of a verify window ends with two launches before its
/// out_proj matmul: the gated output norm (`Qwen35GatedNormTail`,
/// `bonsai_gdn_gated_norm_signed`: per row and value head the RMSNorm of the
/// 128 outputs with the norm weight, `(z * sigmoid(z)) * normed` and
/// out_proj's Hadamard signs, stored FP32) and out_proj's quantizing rotation
/// (`bonsai_signed_hadamard_1024_q8_blocks` at width 6,144), which reads the
/// signed product back. The producer rotation's gated-norm form
/// (`SignedBlockHadamard.fusedTransformInt8Producer` with `.gatedRMSNorm`:
/// PROD 3 of `bonsai_signed_hadamard_1024_q8p_blocks`, 256 threads per
/// 1024-block, one value head per simdgroup) forms that product in its read
/// with the gated norm kernel's expressions: lane l squares elements 4l ..<
/// 4l + 4 of its head in order, `simd_sum`, `precise::rsqrt(acc / 128 +
/// eps)`, `w * (x * inv)`, `(z * sigmoid(z)) * xn` with MLX's `Sigmoid`, then
/// the signs; the butterflies and the quantization are the plain rotation's.
/// At verify width it ran only when the load-time producer trial adopted the
/// three producers together (`narrowProducerActive`). Here a full 16-row
/// window on the int8 narrow route takes it for every GDN output, and
/// out_proj reads the activation through the same narrow int8 matmul
/// (`sharedHadamardProjectionsQuantized`): one launch, and the FP32 product's
/// store and re-read, less per GDN layer. Every other forward keeps the chain.
/// (The replay kernel that produces the output cannot take the norm: a
/// head's 128 outputs come from two or four of its threadgroups.)
///
/// Before first use a self-test runs the live chain (the gated norm, the
/// quantizing rotation, the out_proj matmul) and this path for every GDN layer
/// with its production norm weight, eps, signs and out_proj: 16 rows of output
/// with per-row scales from 1e-4 to 60, head scales spread over e^±3, outlier
/// channels, a zero row and -0.0 elements, and z a column slice of a wider
/// FP32 product. Codes, scales, scaled sums and the FP16 projection are
/// compared bit for bit; a mismatch or an MLX error keeps the chain, and so
/// does a layer the test did not cover. `MLXFAST_GDN_NORM_FUSED=0` keeps it.
enum Qwen35GDNNormFold {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_GDN_NORM_FUSED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let rows = 16

    private final class Entry {
        weak var layer: Qwen35GatedDeltaNet?
        init(_ layer: Qwen35GatedDeltaNet) { self.layer = layer }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var entries: [Entry] = []
    nonisolated(unsafe) private static var verdict: Bool?
    nonisolated(unsafe) private static var covered: Set<ObjectIdentifier> = []

    /// Each GDN layer at construction: the self-test covers all of them.
    static func register(_ layer: Qwen35GatedDeltaNet) {
        guard enabled else { return }
        lock.withLock { entries.append(Entry(layer)) }
    }

    /// `packed.forwardPreSigned(gatedNormTailSigned(rmsNorm(out), gate, signs),
    /// widenOutput: false)` for a full verify window, or nil for the chain.
    static func apply(
        _ layer: Qwen35GatedDeltaNet, _ packed: HadamardQuantizedLinear, _ out: MLXArray,
        gate: MLXArray
    ) -> MLXArray? {
        guard enabled, takes(layer, packed, out, gate), verified(layer) else { return nil }
        return launch(layer, packed, out, gate)?.y
    }

    private static func takes(
        _ layer: Qwen35GatedDeltaNet, _ packed: HadamardQuantizedLinear, _ out: MLXArray,
        _ gate: MLXArray
    ) -> Bool {
        let weight = layer.norm.weight
        let transform = packed.transform
        return out.dtype == .float32 && gate.dtype == .float32 && out.ndim == 4
            && gate.shape == out.shape && out.dim(0) * out.dim(1) == rows && out.dim(3) == 128
            && out.dim(2) * 128 == transform.width && packed.gdnLayout == nil
            && transform.blockSize == 1024 && transform.width % 1024 == 0
            && transform.signVector.dtype == .float32
            && weight.dtype == .float32 && weight.shape == [128]
            && sharedHadamardTensorRouteTakesNarrowInt8([packed], rows: rows)
    }

    private static func launch(
        _ layer: Qwen35GatedDeltaNet, _ packed: HadamardQuantizedLinear, _ out: MLXArray,
        _ gate: MLXArray
    ) -> (activation: SignedBlockHadamard.Int8Activation, y: MLXArray)? {
        guard
            let activation = packed.transform.forwardInt8(
                producer: .gatedRMSNorm(
                    x: out, gate: gate, weight: layer.norm.weight, eps: layer.norm.eps),
                gdnLayout: nil, groupSize: 128),
            let y = sharedHadamardProjectionsQuantized(
                activation, leading: [out.dim(0), out.dim(1)], [packed], widenOutput: false)?
                .first
        else { return nil }
        return (activation, y)
    }

    private static func verified(_ layer: Qwen35GatedDeltaNet) -> Bool {
        lock.withLock {
            if verdict == nil {
                let (passed, summary) = selfTest()
                verdict = passed
                FileHandle.standardError.write(
                    ("qwen35 verify GDN gated norm in the out_proj rotation (GDNNORM): " + summary
                        + (passed ? "; one launch\n" : "; chain kept\n")).data(using: .utf8)!)
            }
            return verdict == true && covered.contains(ObjectIdentifier(layer))
        }
    }

    /// The live chain and this path on every registered layer (the lock is
    /// held).
    private static func selfTest() -> (Bool, String) {
        var layers = 0
        var values = 0
        var mismatches = 0
        var negativeZeros = 0
        var passedLayers: Set<ObjectIdentifier> = []
        var failure: String? = nil
        do {
            try withError { error in
                let row = MLXArray(0 ..< rows).reshaped(1, rows, 1, 1)
                let column = MLXArray(0 ..< 128).reshaped(1, 1, 1, 128)
                for (index, entry) in entries.enumerated() {
                    guard let layer = entry.layer,
                        let packed = layer.outProj as? HadamardQuantizedLinear
                    else { continue }
                    let heads = layer.numVHeads
                    let width = heads * 128
                    let keys = MLXRandom.split(
                        key: MLXRandom.key(UInt64(0x4744_4e00 + index)), into: 5)
                    // Per-row scales from 1e-4 to 60, head scales over e^±3,
                    // outlier channels, a zero row, -0.0 on every fifth
                    // channel of another row.
                    let scale = exp(
                        MLXRandom.uniform(
                            Float(-9.21) ..< Float(4.09), [1, rows, 1, 1], key: keys[0]))
                    let headScale = exp(
                        MLXRandom.uniform(Float(-3) ..< Float(3), [1, 1, heads, 1], key: keys[1]))
                    let outlier = which(
                        column % 29 .== MLXArray(Int32(index % 29)), Float(40), Float(1))
                    let zeroRow = row .== MLXArray(Int32(index % rows))
                    let negativeZero =
                        (row .== MLXArray(Int32((index + 5) % rows))) .&& (column % 5 .== 0)
                    let x = which(
                        negativeZero, -Float(0),
                        which(
                            zeroRow, Float(0),
                            MLXRandom.normal([1, rows, heads, 128], key: keys[2]) * scale
                                * headScale * outlier))
                    // z as the qkv|z product's column slice: a strided view.
                    let product =
                        MLXRandom.normal([1, rows, 10240 + width], key: keys[3])
                        * exp(MLXRandom.normal([1, rows, 10240 + width], key: keys[4]))
                    let z = product[0..., 0..., 10240...].reshaped(1, rows, heads, 128)
                    guard takes(layer, packed, x, z) else { continue }
                    let weight = layer.norm.weight
                    let eps = layer.norm.eps
                    let signs = packed.transform.signVector
                    // The live chain: the gated norm (its launch, or the
                    // composed ops where that declines), out_proj's quantizing
                    // rotation and its matmul.
                    let signed =
                        Qwen35GatedNormTail.apply(x, gate: z, weight: weight, eps: eps, signs: signs)
                        ?? Qwen35FusedElementwise.gatedNormTailSigned(
                            MLXFast.rmsNorm(x, weight: weight, eps: eps), z,
                            signs.reshaped(heads, 128)
                        ).asType(x.dtype)
                    let chainY = packed.forwardPreSigned(
                        signed.reshaped(1, rows, -1), widenOutput: false)
                    guard
                        let chain = packed.transform.forwardInt8(
                            signed.reshaped(rows, width), gdnLayout: nil, preSigned: true,
                            groupSize: 128),
                        let fused = launch(layer, packed, x, z)
                    else {
                        failure = "a launch declined at layer \(index)"
                        return
                    }
                    let pairs = [
                        (chain.codes, fused.activation.codes),
                        (chain.scales, fused.activation.scales),
                        (chain.scaledSums, fused.activation.scaledSums), (chainY, fused.y),
                    ]
                    var differ: [MLXArray] = []
                    for (c, f) in pairs {
                        guard c.dtype == f.dtype, c.shape == f.shape else {
                            failure = "output \(f.dtype) \(f.shape) vs \(c.dtype) \(c.shape)"
                            return
                        }
                        let bits: DType =
                            c.dtype == .float16 ? .uint16 : c.dtype == .float32 ? .uint32 : c.dtype
                        differ.append(
                            (c.view(dtype: bits) .!= f.view(dtype: bits)).asType(.int32).sum())
                        values += c.size
                    }
                    let count = stacked(differ).sum()
                    let zeros = (x.view(dtype: .uint32) .== MLXArray(UInt32(0x8000_0000)))
                        .asType(.int32).sum()
                    eval(count, zeros)
                    try error.check()
                    let layerMismatches = Int(count.item(Int32.self))
                    mismatches += layerMismatches
                    negativeZeros += Int(zeros.item(Int32.self))
                    layers += 1
                    if layerMismatches == 0 { passedLayers.insert(ObjectIdentifier(layer)) }
                }
            }
        } catch {
            failure = "\(error)"
        }
        if let failure { return (false, "self-test error: \(failure)") }
        let passed = mismatches == 0 && layers > 0 && negativeZeros > 0
        if passed { covered = passedLayers }
        return (
            passed,
            "self-test \(passed ? "passed" : "FAILED"): \(layers) layers, \(values) values "
                + "compared bitwise (codes, scales, scaled sums, projections; \(negativeZeros) "
                + "-0.0 inputs), \(mismatches) mismatches")
    }
}

// MARK: - The verify window's SwiGLU and attention-gate producers in the rotation's read

/// Two more of a verify window's output projections run an elementwise
/// launch before their quantizing rotation: the MLP's compiled `(silu(gate) *
/// up) * signs` (`Qwen35FusedElementwise.swigluSigned`, stored FP32, 16 x
/// 17,408, in each of the 64 layers) ahead of down_proj's, and the attention's
/// compiled `(x * sigmoid(gate)) * signs` (`sigmoidGateSigned`, stored FP32,
/// 16 x 6,144, in each of the 16 full-attention layers) ahead of o_proj's
/// (`bonsai_signed_hadamard_1024_q8_blocks`), which reads the product back.
/// The record's producer rotation forms either product in its read
/// (`SignedBlockHadamard.fusedTransformInt8Producer`: PROD 1 `(a *
/// sigmoid(a)) * b` and PROD 2 `a * sigmoid(b)` of
/// `bonsai_signed_hadamard_1024_q8p_blocks`, MLX's `Sigmoid` verbatim, in
/// FP32, then the signs; the butterflies and the quantization are the plain
/// rotation's), but at verify width only when the load-time producer trial
/// adopts the three producers together (`narrowProducerActive`). Here, where
/// the record's path declines, a full 16-row window on the int8 narrow route
/// takes the producer, and the projection reads the activation through the
/// same narrow int8 matmul (`sharedHadamardProjectionsQuantized`): one
/// launch, and the FP32 product's store and re-read, less per projection.
/// Every other forward keeps the chain.
///
/// Before first use a self-test per form runs the live chain (the compiled
/// elementwise launch, the quantizing rotation, the projection's matmul) and
/// this path on every layer with its production signs and projection: 16
/// rows with per-row scales from 1e-4 up (to 8 for the FP16 gate|up, to 60
/// with head scales over e^+-3 for the attention output), outlier channels,
/// a zero row and -0.0 elements, in the call's layouts: gate and up the
/// column halves of one FP16 gate|up product; the attention output
/// head-transposed (and row-contiguous) with the gate half of each q|gate
/// head of a wider FP32 q|k|v product. Codes, scales, scaled sums and the
/// FP16 projection are compared bit for bit; a mismatch or an MLX error keeps
/// that form's chain, and so does a layer the test did not cover.
/// `MLXFAST_VERIFY_PRODUCERS_FORCED=0` keeps both chains.
enum Qwen35ProducerFold {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_VERIFY_PRODUCERS_FORCED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let rows = 16

    private enum Form: Hashable {
        case swiglu, gate
        var label: String {
            self == .swiglu
                ? "SwiGLU in the down_proj rotation" : "attention gate in the o_proj rotation"
        }
    }

    private final class Entry {
        weak var layer: Qwen35DecoderLayer?
        init(_ layer: Qwen35DecoderLayer) { self.layer = layer }
    }

    /// One layer's test operands: the projection, the input whose -0.0 and
    /// finiteness are counted, the live chain's activation and product, and
    /// this path's.
    private struct Case {
        let packed: HadamardQuantizedLinear
        let input: MLXArray
        let chain: SignedBlockHadamard.Int8Activation
        let chainY: MLXArray
        let fused: (activation: SignedBlockHadamard.Int8Activation, y: MLXArray)
    }

    private struct Declined: Error {
        let message: String
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var entries: [Entry] = []
    nonisolated(unsafe) private static var verdicts: [Form: Bool] = [:]
    nonisolated(unsafe) private static var covered: [Form: Set<ObjectIdentifier>] = [:]

    /// Each decoder layer at construction: the self-tests cover all of them.
    static func register(_ layer: Qwen35DecoderLayer) {
        guard enabled else { return }
        lock.withLock { entries.append(Entry(layer)) }
    }

    /// `down.forwardPreSigned(swigluSigned(gate, up, signs), widenOutput:
    /// false)` for a full verify window, or nil for the chain.
    static func swiglu(_ down: HadamardQuantizedLinear, _ gate: MLXArray, _ up: MLXArray)
        -> MLXArray?
    {
        guard enabled, swigluTakes(down, gate, up), verified(.swiglu, down) else { return nil }
        return launch(down, .swiglu(gate: gate, up: up), leading: [gate.dim(0), gate.dim(1)])?.y
    }

    /// `oProj.forwardPreSigned(sigmoidGateSigned(x, gate, signs).reshaped(B, L,
    /// -1), widenOutput: false)` for a full verify window's `[B, L, heads,
    /// headDim]` operands, or nil for the chain.
    static func sigmoidGate(_ oProj: Linear, _ x: MLXArray, _ gate: MLXArray) -> MLXArray? {
        guard enabled, let packed = oProj as? HadamardQuantizedLinear, gateTakes(packed, x, gate),
            verified(.gate, packed)
        else { return nil }
        return launch(packed, .sigmoidGate(x: x, gate: gate), leading: [x.dim(0), x.dim(1)])?.y
    }

    private static func routeTakes(_ packed: HadamardQuantizedLinear) -> Bool {
        let transform = packed.transform
        return packed.gdnLayout == nil && transform.blockSize == 1024
            && transform.width % 1024 == 0 && transform.signVector.dtype == .float32
            && sharedHadamardTensorRouteTakesNarrowInt8([packed], rows: rows)
    }

    private static func swigluTakes(
        _ down: HadamardQuantizedLinear, _ gate: MLXArray, _ up: MLXArray
    ) -> Bool {
        gate.dtype == .float16 && up.dtype == .float16 && gate.ndim == 3 && gate.shape == up.shape
            && gate.dim(0) * gate.dim(1) == rows && gate.dim(2) == down.transform.width
            && routeTakes(down)
    }

    private static func gateTakes(
        _ packed: HadamardQuantizedLinear, _ x: MLXArray, _ gate: MLXArray
    ) -> Bool {
        Qwen35FusedElementwise.foldsHadamardSigns && x.dtype == .float32
            && gate.dtype == .float32 && x.ndim == 4 && x.shape == gate.shape
            && x.dim(0) * x.dim(1) == rows && x.dim(2) * x.dim(3) == packed.transform.width
            && routeTakes(packed)
    }

    private static func launch(
        _ packed: HadamardQuantizedLinear, _ producer: SignedBlockHadamard.Int8Producer,
        leading: [Int]
    ) -> (activation: SignedBlockHadamard.Int8Activation, y: MLXArray)? {
        guard
            let activation = packed.transform.forwardInt8(
                producer: producer, gdnLayout: nil, groupSize: 128),
            let y = sharedHadamardProjectionsQuantized(
                activation, leading: leading, [packed], widenOutput: false)?.first
        else { return nil }
        return (activation, y)
    }

    private static func verified(_ form: Form, _ packed: HadamardQuantizedLinear) -> Bool {
        lock.withLock {
            if verdicts[form] == nil {
                let (passed, summary) = selfTest(form)
                verdicts[form] = passed
                FileHandle.standardError.write(
                    ("qwen35 verify \(form.label) (PRODALL): " + summary
                        + (passed ? "; one launch\n" : "; chain kept\n")).data(using: .utf8)!)
            }
            return verdicts[form] == true
                && covered[form, default: []].contains(ObjectIdentifier(packed))
        }
    }

    /// The live chain and this path for one registered layer; none when the
    /// layer has no such projection or the route does not take it.
    private static func cases(_ form: Form, _ layer: Qwen35DecoderLayer, _ index: Int) throws
        -> [Case]
    {
        let keys = MLXRandom.split(
            key: MLXRandom.key(UInt64((form == .swiglu ? 0x5357_0000 : 0x4741_0000) + index)),
            into: 5)
        if form == .swiglu {
            guard let mlp = layer.mlp as? Qwen3NextMLP,
                let down = mlp.downProj as? HadamardQuantizedLinear
            else { return [] }
            let width = down.transform.width
            let row = MLXArray(0 ..< rows).reshaped(1, rows, 1)
            let column = MLXArray(0 ..< 2 * width).reshaped(1, 1, 2 * width)
            // Per-row scales from 1e-4 to 8 and x8 outlier channels (FP16
            // stays finite), a zero row, -0.0 on every fifth channel of
            // another row; gate and up the column halves of one product.
            let scale = exp(
                MLXRandom.uniform(Float(-9.21) ..< Float(2.08), [1, rows, 1], key: keys[0]))
            let outlier = which(column % 29 .== MLXArray(Int32(index % 29)), Float(8), Float(1))
            let zeroRow = row .== MLXArray(Int32(index % rows))
            let negativeZero = (row .== MLXArray(Int32((index + 5) % rows))) .&& (column % 5 .== 0)
            let gateUp = which(
                negativeZero, -Float(0),
                which(
                    zeroRow, Float(0),
                    MLXRandom.normal([1, rows, 2 * width], key: keys[1]) * scale * outlier)
            ).asType(.float16)
            let halves = MLX.split(gateUp, indices: [width], axis: -1)
            guard swigluTakes(down, halves[0], halves[1]) else { return [] }
            let signed = Qwen35FusedElementwise.swigluSigned(
                halves[0], halves[1], down.transform.signVector)
            guard
                let chain = down.transform.forwardInt8(
                    signed.reshaped(rows, width), gdnLayout: nil, preSigned: true, groupSize: 128),
                let fused = launch(
                    down, .swiglu(gate: halves[0], up: halves[1]), leading: [1, rows])
            else { throw Declined(message: "a launch declined at layer \(index)") }
            return [
                Case(
                    packed: down, input: gateUp, chain: chain,
                    chainY: down.forwardPreSigned(signed, widenOutput: false), fused: fused)
            ]
        }
        guard let attention = layer.selfAttn,
            let packed = attention.oProj as? HadamardQuantizedLinear
        else { return [] }
        let width = packed.transform.width
        let heads = attention.attentionHeads
        guard heads > 0, width % heads == 0 else { return [] }
        let headDim = width / heads
        let row = MLXArray(0 ..< rows).reshaped(1, 1, rows, 1)
        let column = MLXArray(0 ..< headDim).reshaped(1, 1, 1, headDim)
        // The attention's [1, heads, rows, headDim] output: per-row scales
        // from 1e-4 to 60, head scales over e^+-3, outlier channels, a zero
        // row, -0.0 on every fifth channel of another row.
        let scale = exp(
            MLXRandom.uniform(Float(-9.21) ..< Float(4.09), [1, 1, rows, 1], key: keys[0]))
        let headScale = exp(
            MLXRandom.uniform(Float(-3) ..< Float(3), [1, heads, 1, 1], key: keys[1]))
        let outlier = which(column % 29 .== MLXArray(Int32(index % 29)), Float(40), Float(1))
        let zeroRow = row .== MLXArray(Int32(index % rows))
        let negativeZero = (row .== MLXArray(Int32((index + 5) % rows))) .&& (column % 5 .== 0)
        let output = which(
            negativeZero, -Float(0),
            which(
                zeroRow, Float(0),
                MLXRandom.normal([1, heads, rows, headDim], key: keys[2]) * scale * headScale
                    * outlier))
        // The gate half of each q|gate head of the stacked q|k|v product, split
        // as the call splits it.
        let stackedWidth = 2 * width + 2048
        let product =
            MLXRandom.normal([1, rows, stackedWidth], key: keys[3])
            * exp(MLXRandom.normal([1, rows, stackedWidth], key: keys[4]) * Float(1.5))
        let gate = MLX.split(product, indices: [2 * width], axis: -1)[0]
            .reshaped(1, rows, heads, -1).split(parts: 2, axis: -1)[1]
        // The head-transposed output, and the same values row-contiguous.
        let transposed = output.transposed(0, 2, 1, 3)
        let contiguous = transposed.reshaped(1, rows, width).reshaped(1, rows, heads, headDim)
        guard gateTakes(packed, transposed, gate) else { return [] }
        return try [transposed, contiguous].map { (x: MLXArray) throws -> Case in
            let signed = Qwen35FusedElementwise.sigmoidGateSigned(
                x, gate, packed.transform.signVector.reshaped(heads, headDim))
            guard
                let chain = packed.transform.forwardInt8(
                    signed.reshaped(rows, width), gdnLayout: nil, preSigned: true, groupSize: 128),
                let fused = launch(packed, .sigmoidGate(x: x, gate: gate), leading: [1, rows])
            else { throw Declined(message: "a launch declined at layer \(index)") }
            return Case(
                packed: packed, input: output, chain: chain,
                chainY: packed.forwardPreSigned(signed.reshaped(1, rows, -1), widenOutput: false),
                fused: fused)
        }
    }

    /// The live chain and this path on every registered layer (the lock is
    /// held).
    private static func selfTest(_ form: Form) -> (Bool, String) {
        var layers = 0
        var values = 0
        var mismatches = 0
        var negativeZeros = 0
        var passedLayers: Set<ObjectIdentifier> = []
        var failure: String? = nil
        do {
            try withError { error in
                for (index, entry) in entries.enumerated() {
                    guard let layer = entry.layer else { continue }
                    let tests = try cases(form, layer, index)
                    guard let packed = tests.first?.packed else { continue }
                    var layerMismatches = 0
                    for test in tests {
                        let pairs = [
                            (test.chain.codes, test.fused.activation.codes),
                            (test.chain.scales, test.fused.activation.scales),
                            (test.chain.scaledSums, test.fused.activation.scaledSums),
                            (test.chainY, test.fused.y),
                        ]
                        var differ: [MLXArray] = []
                        for (c, f) in pairs {
                            guard c.dtype == f.dtype, c.shape == f.shape else {
                                throw Declined(
                                    message:
                                        "output \(f.dtype) \(f.shape) vs \(c.dtype) \(c.shape)")
                            }
                            let bits: DType =
                                c.dtype.size == 4 ? .uint32 : c.dtype.size == 2 ? .uint16 : .uint8
                            differ.append(
                                (c.view(dtype: bits) .!= f.view(dtype: bits)).asType(.int32).sum())
                            values += c.size
                        }
                        let count = stacked(differ).sum()
                        let half = test.input.dtype == .float16
                        let zeros = (half
                            ? test.input.view(dtype: .uint16) .== MLXArray(UInt16(0x8000))
                            : test.input.view(dtype: .uint32) .== MLXArray(UInt32(0x8000_0000)))
                            .asType(.int32).sum()
                        let finite = (abs(test.input.asType(.float32)) .< Float.infinity)
                            .asType(.int32).sum()
                        eval(count, zeros, finite)
                        try error.check()
                        guard Int(finite.item(Int32.self)) == test.input.size else {
                            throw Declined(message: "non-finite operands at layer \(index)")
                        }
                        layerMismatches += Int(count.item(Int32.self))
                        negativeZeros += Int(zeros.item(Int32.self))
                    }
                    mismatches += layerMismatches
                    layers += 1
                    if layerMismatches == 0 { passedLayers.insert(ObjectIdentifier(packed)) }
                }
            }
        } catch let declined as Declined {
            failure = declined.message
        } catch {
            failure = "\(error)"
        }
        if let failure { return (false, "self-test error: \(failure)") }
        let passed = mismatches == 0 && layers > 0 && negativeZeros > 0
        if passed { covered[form] = passedLayers }
        return (
            passed,
            "self-test \(passed ? "passed" : "FAILED"): \(layers) layers, \(values) values "
                + "compared bitwise (codes, scales, scaled sums, projections; \(negativeZeros) "
                + "-0.0 inputs), \(mismatches) mismatches")
    }
}
