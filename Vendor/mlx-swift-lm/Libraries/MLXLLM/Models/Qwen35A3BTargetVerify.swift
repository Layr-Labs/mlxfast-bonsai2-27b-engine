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

    /// One row's `(y, committed state)`: `keep` rows of the previous `tape`
    /// from its pre-verify state, then this verify's rows (`q` ... `beta`,
    /// the prework's row-contiguous FP32 outputs). Nil when it does not fit.
    static func launch(
        tape: ArraysCache.PrefixReplayTape, keep: Int, aLog: MLXArray, dtBias: MLXArray,
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
        staged: Bool? = nil
    ) -> (y: MLXArray, state: MLXArray)? {
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
        eval(previous)
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
        let out = (useStaged ? stagedKernel : kernel)(
            [q, k, v, g, beta, MLXArray(Int32(T))] + previous
                + [MLXArray([aRows, bRows]), MLXArray(Int32(keep))],
            template: [
                ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("OUTPUT_NEEDED", true),
                ("DVPL", dvpl),
            ],
            grid: (128, Dv / (16 * dvpl), Hv), threadGroup: (128, 1, 1),
            outputShapes: [[1, T, Hv, Dv], ps.shape],
            outputDTypes: [.float32, .float32])
        return (out[0], out[1])
    }

    /// The fused scan of a single-row verify whose input SSM is `layer`'s
    /// pending deferred replay, which it resolves with the committed state.
    static func run(
        layer: Qwen35GatedDeltaNet, input: CBv2RecurrentLayerState?,
        pre: Qwen35GDNPrework.Outputs
    ) -> (y: MLXArray, state: MLXArray)? {
        guard let deferred = input?.deferredReplay, deferred.isPending,
            let inputs = deferred.inputs as? Inputs, inputs.layer == ObjectIdentifier(layer),
            applies(to: layer),
            let fused = launch(
                tape: inputs.tape, keep: deferred.keep, aLog: layer.aLog,
                dtBias: layer.dtBias, q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta),
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
                    let tape = ArraysCache.PrefixReplayTape(
                        convInput: convInput, q: rows(13, Hk, Dk, 0.09), k: rows(14, Hk, Dk, 0.09),
                        v: v, a: pair[0..., 0..., Hv...], b: pair[0..., 0..., ..<Hv],
                        ssmPre: ssmPre, mask: nil, rowCount: S, convStateRows: NK)
                    eval(window + [ssmPre, convInput, tape.q, tape.k, v, tape.a, tape.b, aLog, dtBias])
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
          for (int i = 0; i < R; ++i) {
            state[d][i] = ps[(n * Dv + dvbase + d) * Dk + dk0 + i];
          }
        }

        // Phase 1: the previous tape's KP rows, the batched replay's step.
        {
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
            const float g_sp = qwen35_replay_logaddexp(pa[hv_idx + tid * a_rs] + g_dtb, 0.0f);
            tgate[tid] = metal::precise::exp(g_nexp * g_sp);
            tgate[16 + tid] = qwen35_replay_sigmoid(pb[hv_idx + tid * b_rs]);
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

        // The committed state.
        #pragma clang loop unroll(full)
        for (int d = 0; d < DVPL; ++d) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < R; ++i) {
            state_out[(n * Dv + dvbase + d) * Dk + dk0 + i] = state[d][i];
          }
        }

        // Phase 2: this verify's T rows, the output-only scan's step.
        {
          threadgroup_barrier(mem_flags::mem_threadgroup);
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
        """

    static let stagedKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_replay_fused_staged",
        inputNames: [
            "q", "k", "v", "g", "beta", "T", "ps", "pk", "pv", "pa", "pb", "alog", "dtb",
            "ab_rows", "KP",
        ],
        outputNames: ["y", "state_out"],
        source: stagedSource,
        header: stagedHeader,
        ensureRowContiguous: false)
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
        source: fusedPrepSource, ensureRowContiguous: false)

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
        keyHeads: Int, valueHeads: Int, convDim: Int, taps: Int, dtype: DType
    ) -> Bool {
        guard rowTiledEnabled else { return false }
        let geometry = RowTileGeometry(
            hk: keyHeads, hv: valueHeads, cd: convDim, ks: taps, dtype: "\(dtype)")
        return rowTileLock.withLock { rowTileVerdicts[geometry] ?? false }
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
        hk: Int, dk: Int, hv: Int, dv: Int, ks: Int, dtype: DType, forms: Int
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
        source: verifyLoadsFirstSource,
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
        const size_t rowbase = size_t(row) * size_t(W);
        threadgroup float buf[N];
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
        for (uint r = 0; r < EPT; r++) {
          buf[EPT * tid + r] = x[r];
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
          float v[4];
          float amax = 0.0f;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            v[r] = buf[index + r] * 0.03125f;
            amax = max(amax, fabs(v[r]));
          }
          amax = simd_max(amax);
          const float qs = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
          const float iqs = amax > 0.0f ? 127.0f / amax : 0.0f;
          float part = 0.0f;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            const float q = rint(v[r] * iqs);
            part += q;
            const uint kk = uint(index + r);
            const uint kp = PERM ? ((kk & ~15u) | (4u * (kk & 3u) + ((kk >> 2) & 3u))) : kk;
            if (SIGNED) { out[rowbase + bcol + kp] = int8_t(q); } else { out[rowbase + bcol + kp] = uint8_t(int(q) + 128); }
          }
          part = simd_sum(part);
          if (lane == 0) {
            const size_t g = size_t(bcol / 128) + size_t(gi);
            const uint ml = row & 63u;
            const size_t qidx = MPERM
              ? (size_t(row >> 6) * size_t(W / 128) * 64 + g * 64
                 + size_t(((ml >> 4) & 1u) * 32u + (ml & 7u) * 4u + ((ml >> 5) & 1u) * 2u + ((ml >> 3) & 1u)))
              : (size_t(row) * size_t(W / 128) + g);
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
    static func threads(blocks: Int) -> Int { blocks <= 128 ? 256 : wideThreads }

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
        guard enabled, rows > 0, rows < BonsaiPromptWidth.minimumRows, width % 1024 == 0
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
        source: source,
        ensureRowContiguous: false)

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]
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
        return kernel(
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
    }
}

// MARK: - Tree speculative verify: the GDN tree block

/// Host helpers for token-tree layouts (rows in topological order, `parent[0]
/// == -1`, `parent[i] < i`); the self-tests use them.
enum Qwen35TreeLayout {
    static let maxRows = 16

    static func isValid(_ parent: [Int]) -> Bool {
        guard !parent.isEmpty, parent.count <= maxRows, parent[0] == -1 else { return false }
        for i in 1 ..< parent.count where parent[i] < 0 || parent[i] >= i { return false }
        return true
    }

    static func depths(_ parent: [Int]) -> [Int] {
        var depth = [Int](repeating: 0, count: parent.count)
        for i in 1 ..< max(parent.count, 1) { depth[i] = depth[parent[i]] + 1 }
        return depth
    }

    /// Rows root ... `node`.
    static func path(to node: Int, parent: [Int]) -> [Int] {
        var rows: [Int] = []
        var n = node
        while n >= 0 {
            rows.append(n)
            n = parent[n]
        }
        return rows.reversed()
    }

    static func leaves(_ parent: [Int]) -> [Int] {
        var hasChild = [Bool](repeating: false, count: parent.count)
        for i in 1 ..< max(parent.count, 1) { hasChild[parent[i]] = true }
        return (0 ..< parent.count).filter { !hasChild[$0] }
    }
}

/// The gated-delta recurrence of one speculative tree block (T <= 16 rows in
/// topological order, one launch, output rows only): every row's output is
/// the output that row gets with its root-to-row path run as a chain from the
/// block's input state. Chunkwise (WY) form with one chunk, restricted to
/// ancestors: with `anc(i)` the ancestor mask of row i (itself included) and
/// `Gam_i` the sum of `log g` along its path,
///
///     A_ij  = beta_i (k_i . k_j) exp(Gam_i - Gam_j)   (j in anc(i), j != i)
///     P_ij  = (q_i . k_j) exp(Gam_i - Gam_j)          (j in anc(i))
///     T'    = (I + A)^-1 diag(beta)                  (forward substitution)
///     Delta = T' (V - diag(exp Gam) K S0^T)
///     Y     = diag(exp Gam) Q S0^T + P Delta
///
/// A is strictly lower triangular and ancestry is transitive, so row i of
/// (I + A)^-1 is supported on anc(i) and siblings never interact. Masks and
/// path sums come from `parent` by pointer jumping on device; non-ancestor
/// terms are skipped by a branch (their exponents are unbounded), never
/// multiplied by zero. grid (32, Dv / 8, Hv), threadgroup (32, NS, 1). No
/// state output: the committed state is the replay's job.
enum Qwen35GatedDeltaTree {
    /// Simdgroups per threadgroup (`MLXFAST_TREE_GDN_NS`: 1, 2, 4, 8 or 16).
    static let simdgroups: Int = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_TREE_GDN_NS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let raw, let value = Int(raw), [1, 2, 4, 8, 16].contains(value) { return value }
        return 8
    }()

    static let source = """
        // grid (32, Dv / 8, Hv), threadgroup (32, NS, 1); B == 1; T <= C rows.
        constexpr int CT = C / 8;
        constexpr int DT = Dk / 8;
        constexpr int LC = C + 8;
        constexpr int LD = C + 1;
        constexpr int NJ = CT * (CT + 1) / 2;
        static_assert(C == 8 || C == 16, "one chunk of 8 or 16 rows");
        threadgroup float KK[C * LD];
        threadgroup float QK[C * LD];
        threadgroup float Ah[C * LD];
        threadgroup float Mh[C * LD];
        threadgroup float TPsh[2 * C * LC];
        threadgroup float EG[C];
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int T_ = T;
        const int r0 = int(thread_position_in_grid.y) * 8;
        const int hv = int(thread_position_in_grid.z);
        const int hk = hv / (Hv / Hk);
        const int ks = Hk * Dk;
        const int vs = Hv * Dv;
        const short qid = lane / 4;
        const short fm = (qid & 4) + ((lane / 2) % 4);
        const short fn = (qid & 2) * 2 + (lane % 2) * 2;
        const device float* kbase = k + hk * Dk;
        const device float* qbase = q + hk * Dk;

        // This simdgroup's 8 state rows (S0^T columns), loaded first.
        simdgroup_float8x8 St[DT];
        _Pragma("clang loop unroll(full)")
        for (int d = 0; d < DT; ++d)
          simdgroup_load(St[d], state_in + ((size_t)hv * Dv + r0) * Dk + d * 8, Dk, ulong2(0, 0), true);

        // Lower tiles of K K^T and Q K^T, one tile per simdgroup; q and k
        // hold C rows (zero padding past T).
        for (int job = int(sg); job < NJ; job += NS) {
          int ti = 0;
          int tj = job;
          while (tj > ti) { tj -= ti + 1; ++ti; }
          simdgroup_float8x8 akk = simdgroup_float8x8(0);
          simdgroup_float8x8 aqk = simdgroup_float8x8(0);
          _Pragma("clang loop unroll(full)")
          for (int d = 0; d < Dk / 8; ++d) {
            simdgroup_float8x8 ka, qa, kb;
            simdgroup_load(ka, kbase + (ti * 8) * ks + d * 8, ks);
            simdgroup_load(qa, qbase + (ti * 8) * ks + d * 8, ks);
            simdgroup_load(kb, kbase + (tj * 8) * ks + d * 8, ks, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(akk, ka, kb, akk);
            simdgroup_multiply_accumulate(aqk, qa, kb, aqk);
          }
          simdgroup_store(akk, KK + (ti * 8) * LD + tj * 8, LD);
          simdgroup_store(aqk, QK + (ti * 8) * LD + tj * 8, LD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Row work in simdgroup 0, lane = row: ancestor masks and path sums of
        // log g by pointer jumping, decay factors, P and A rows.
        float bet = 0.0f;
        if (sg == 0) {
          const int row = int(lane);
          const bool live = row < C;
          const bool real = row < T_;
          int jump = -1;
          uint anc = live ? (1u << row) : 0u;
          float gam = 0.0f;
          bool zero = false;
          if (live) {
            // Padding rows (>= T) continue a chain from row - 1.
            const int p = real ? parent[row] : row - 1;
            jump = (p >= 0 && p < row) ? p : -1;
            const float gv = real ? g[(size_t)row * Hv + hv] : 1.0f;
            zero = !(gv >= FLT_MIN);
            gam = zero ? 0.0f : metal::precise::log(zero ? 1.0f : gv);
            bet = real ? beta[(size_t)row * Hv + hv] : 0.0f;
          }
          const uint zr = uint(static_cast<simd_vote::vote_t>(simd_ballot(zero))) & ((1u << C) - 1u);
          _Pragma("clang loop unroll(full)")
          for (int off = 1; off < C; off <<= 1) {
            const ushort src = ushort(jump >= 0 ? jump : int(lane));
            const float up = simd_shuffle(gam, src);
            const uint upa = simd_shuffle(anc, src);
            const int upj = simd_shuffle(jump, src);
            gam += jump >= 0 ? up : 0.0f;
            if (jump >= 0) {
              anc |= upa;
              jump = upj;
            }
          }
          // exp(Gam_i) unless i's path crosses an underflowed row.
          if (live) EG[row] = (zr & anc) == 0u ? metal::precise::exp(gam) : 0.0f;
          threadgroup float* P_ = TPsh + C * LC;
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < C; ++j) {
            const float gj = simd_shuffle(gam, ushort(j));
            const uint aj = simd_shuffle(anc, ushort(j));
            if (live) {
              float pv = 0.0f;
              float av = 0.0f;
              if (((anc >> j) & 1u) != 0u) {
                // rows (j, i] of i's path
                const uint span = anc & ~aj;
                const float e = (zr & span) == 0u ? metal::precise::exp(gam - gj) : 0.0f;
                pv = QK[row * LD + j] * e;
                if (j != row) av = bet * KK[row * LD + j] * e;
              }
              P_[row * LC + j] = pv;
              Ah[row * LD + j] = av;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // T = (I + A)^-1 by forward substitution over the topological order,
        // column m = row per lane (stored transposed in Mh[m][i]); T' = T diag(beta).
        if (sg == 0 && int(lane) < C) {
          const int m = int(lane);
          for (int i = 0; i < C; ++i) Mh[m * LD + i] = (i == m) ? 1.0f : 0.0f;
          for (int i = m + 1; i < C; ++i) {
            float s0 = 0.0f, s1 = 0.0f;
            int j = m;
            for (; j + 1 < i; j += 2) {
              s0 = metal::fma(Ah[i * LD + j], Mh[m * LD + j], s0);
              s1 = metal::fma(Ah[i * LD + j + 1], Mh[m * LD + j + 1], s1);
            }
            if (j < i) s0 = metal::fma(Ah[i * LD + j], Mh[m * LD + j], s0);
            Mh[m * LD + i] = -(s0 + s1);
          }
          for (int i = 0; i < C; ++i) TPsh[i * LC + m] = Mh[m * LD + i] * bet;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // X = K S0^T, Xq = Q S0^T (C x 8 each)
        simdgroup_float8x8 Xk[CT];
        simdgroup_float8x8 Xq[CT];
        _Pragma("clang loop unroll(full)")
        for (int ti = 0; ti < CT; ++ti) {
          Xk[ti] = simdgroup_float8x8(0);
          Xq[ti] = simdgroup_float8x8(0);
        }
        _Pragma("clang loop unroll(full)")
        for (int d = 0; d < DT; ++d) {
          _Pragma("clang loop unroll(full)")
          for (int ti = 0; ti < CT; ++ti) {
            simdgroup_float8x8 ka, qa;
            simdgroup_load(ka, kbase + (ti * 8) * ks + d * 8, ks);
            simdgroup_load(qa, qbase + (ti * 8) * ks + d * 8, ks);
            simdgroup_multiply_accumulate(Xk[ti], ka, St[d], Xk[ti]);
            simdgroup_multiply_accumulate(Xq[ti], qa, St[d], Xq[ti]);
          }
        }
        // Z = V - diag(exp Gam) Xk (in Xk); Xq <- diag(exp Gam) Xq
        const device float* v_ = v + hv * Dv + r0;
        device float* y_ = y + hv * Dv + r0;
        _Pragma("clang loop unroll(full)")
        for (int ti = 0; ti < CT; ++ti) {
          const int row = ti * 8 + fm;
          const float eg = EG[row];
          thread auto& zk = Xk[ti].thread_elements();
          thread auto& zq = Xq[ti].thread_elements();
          const float2 vv = row < T_ ? *(const device float2*)(v_ + row * vs + fn) : float2(0.0f);
          zk[0] = vv.x - eg * zk[0];
          zk[1] = vv.y - eg * zk[1];
          zq[0] = eg * zq[0];
          zq[1] = eg * zq[1];
        }
        // Delta = T' Z (T' lower triangular)
        simdgroup_float8x8 Dl[CT];
        _Pragma("clang loop unroll(full)")
        for (int ti = 0; ti < CT; ++ti) {
          Dl[ti] = simdgroup_float8x8(0);
          _Pragma("clang loop unroll(full)")
          for (int tj = 0; tj <= ti; ++tj) {
            simdgroup_float8x8 ta;
            simdgroup_load(ta, TPsh + (ti * 8) * LC + tj * 8, LC);
            simdgroup_multiply_accumulate(Dl[ti], ta, Xk[tj], Dl[ti]);
          }
        }
        // Y = diag(exp Gam) Q S0^T + P Delta
        _Pragma("clang loop unroll(full)")
        for (int ti = 0; ti < CT; ++ti) {
          _Pragma("clang loop unroll(full)")
          for (int tj = 0; tj <= ti; ++tj) {
            simdgroup_float8x8 pa;
            simdgroup_load(pa, TPsh + C * LC + (ti * 8) * LC + tj * 8, LC);
            simdgroup_multiply_accumulate(Xq[ti], pa, Dl[tj], Xq[ti]);
          }
          const int row = ti * 8 + fm;
          thread auto& ye = Xq[ti].thread_elements();
          if (row < T_) *(device float2*)(y_ + row * vs + fn) = float2(ye[0], ye[1]);
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_tree_block",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "parent", "T"],
        outputNames: ["y"],
        source: source)

    static func supports(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray,
        parent: MLXArray, simdgroups ns: Int
    ) -> Bool {
        guard q.ndim == 4, k.ndim == 4, v.ndim == 4, q.shape == k.shape, k.dim(0) == 1
        else { return false }
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        return T >= 1 && T <= Qwen35TreeLayout.maxRows && k.dim(3) == 128 && Dv % 8 == 0
            && (Dv / 8) % ns == 0 && Hv % Hk == 0 && v.dim(1) == T
            && g.shape == [1, T, Hv] && beta.shape == [1, T, Hv]
            && state.shape == [1, Hv, Dv, 128] && parent.shape == [T]
            && parent.dtype == .int32
            && [q, k, v, g, beta, state].allSatisfy { $0.dtype == .float32 }
    }

    /// The output rows `[1, T, Hv, Dv]` of the tree block `q ... beta` (FP32,
    /// rows in topological order) from `state` `[1, Hv, Dv, Dk]`; `parent`
    /// int32 `[T]` (an out-of-range parent is a root). Nil when unfit.
    static func run(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray,
        parent: MLXArray, simdgroups: Int? = nil
    ) -> MLXArray? {
        let ns = simdgroups ?? Self.simdgroups
        guard supports(
            q: q, k: k, v: v, g: g, beta: beta, state: state, parent: parent, simdgroups: ns)
        else { return nil }
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        // The tiles read whole 8-row tiles of q and k: a short block pads them
        // with zero rows (read, never stored).
        let C = Qwen35TreeLayout.maxRows
        let pad = { (x: MLXArray) -> MLXArray in
            T == C ? x : concatenated([x, MLXArray.zeros([1, C - T, Hk, 128], dtype: .float32)], axis: 1)
        }
        return kernel(
            [pad(q), pad(k), v, g, beta, state, parent, MLXArray(Int32(T))],
            template: [("C", C), ("Dk", 128), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("NS", ns)],
            grid: (32, Dv / 8, Hv), threadGroup: (32, ns, 1),
            outputShapes: [[1, T, Hv, Dv]],
            outputDTypes: [.float32])[0]
    }
}

/// The verify prework of a tree block: the verify's prework launch (the
/// reads-first kernel where its check passed, else `qwen35_gdn_prework_ci_strided`)
/// with row t's conv tap j read from row `tap[t * KS + j] - NK` of
/// `[conv state; qkv]` instead of `t + j - NK` (`CBv2TreeVerifyLayout.convTapTable`).
/// One checked text replacement, so every path's rows get the chain launch's
/// values bit for bit. The conv input it writes is `[convState; qkv]` in row
/// order (row 0 is the anchor, whose taps are the row-order ones); its `tail`
/// is the last rows' (unused: the commit takes the accepted path's rows).
extension Qwen35GDNPrework {
    static let rowOrderTap = "const int r = int(t) + j - NK;"

    private static func withTreeTaps(_ text: String) -> String {
        precondition(
            text.components(separatedBy: rowOrderTap).count == 2 && !text.contains("tap["),
            "Qwen35 GDN prework: the tap loop no longer matches the stock kernel")
        return text.replacingOccurrences(
            of: rowOrderTap,
            with: "const int r = int(tap[(int64_t(t) * KS + j) * tap_strides[0]]) - NK;")
    }

    private static let treeKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework_ci_strided_tree",
        inputNames: ["qkv", "cs", "w", "a", "b", "decay", "dtb", "wq", "wk", "S", "tap"],
        outputNames: ["q", "k", "v", "g", "beta", "tail", "ci"],
        source: withTreeTaps(withConvInput(stridedSource)),
        ensureRowContiguous: false)

    private static let treeLoadsFirstKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework_verify_lf_tree",
        inputNames: ["qkv", "cs", "w", "a", "b", "decay", "dtb", "wq", "wk", "S", "tap"],
        outputNames: ["q", "k", "v", "g", "beta", "ci"],
        source: withTreeTaps(verifyLoadsFirstSource),
        ensureRowContiguous: false)

    /// `run(..., writeConvInput: true, stridedReads: true)` with tree taps
    /// (`tapTable` int32 `[S * KS]`); B == 1. Nil exactly when unfit.
    static func runTree(
        qkv: MLXArray, convState: MLXArray, convWeight: MLXArray, a: MLXArray, b: MLXArray,
        aDecay: MLXArray, dtBias: MLXArray, normScales: (q: MLXArray, k: MLXArray),
        keyHeads: Int, valueHeads: Int, headKDim: Int, headVDim: Int, tapTable: MLXArray
    ) -> Outputs? {
        guard enabled, qkv.ndim == 3, convState.ndim == 3, convWeight.ndim == 3, qkv.dim(0) == 1
        else { return nil }
        let S = qkv.dim(1)
        let CD = qkv.dim(2)
        let KS = convWeight.dim(1)
        guard headKDim == 128, headVDim == 128, valueHeads % keyHeads == 0,
            CD == 2 * keyHeads * headKDim + valueHeads * headVDim,
            convState.shape == [1, KS - 1, CD], convWeight.shape == [CD, KS, 1],
            [DType.float32, .float16, .bfloat16].contains(qkv.dtype),
            convState.dtype == .float32, convWeight.dtype == .float32,
            a.dtype == .float32, b.dtype == .float32,
            a.shape == [1, S, valueHeads], b.shape == [1, S, valueHeads],
            aDecay.shape == [valueHeads], aDecay.dtype == .float32,
            dtBias.shape == [valueHeads],
            normScales.q.dtype == .float32, normScales.k.dtype == .float32,
            normScales.q.shape == [headKDim], normScales.k.shape == [headKDim],
            tapTable.shape == [S * KS], tapTable.dtype == .int32,
            S > 0, S <= Qwen35TreeLayout.maxRows
        else { return nil }
        let dtb = dtBias.dtype == .float32 ? dtBias : dtBias.asType(.float32)
        let geometry = LoadsFirstGeometry(
            hk: keyHeads, hv: valueHeads, cd: CD, ks: KS, dtype: "\(qkv.dtype)")
        let loadsFirst =
            verifyLoadsFirstEnabled && loadsFirstLock.withLock { loadsFirstVerdicts[geometry] ?? false }
        let shapes: [[Int]] = [
            [1, S, keyHeads, headKDim], [1, S, keyHeads, headKDim],
            [1, S, valueHeads, headVDim], [1, S, valueHeads], [1, S, valueHeads],
        ]
        let outputs = (loadsFirst ? treeLoadsFirstKernel : treeKernel)(
            [qkv, convState, convWeight, a, b, aDecay, dtb, normScales.q, normScales.k,
             MLXArray(Int32(S)), tapTable],
            template: [
                ("InT", qkv.dtype), ("HK", keyHeads), ("HV", valueHeads), ("DK", headKDim),
                ("DV", headVDim), ("CD", CD), ("KS", KS),
            ],
            grid: (128 * keyHeads, S, 1), threadGroup: (128, 1, 1),
            outputShapes: shapes + (loadsFirst ? [] : [[1, KS - 1, CD]]) + [[1, KS - 1 + S, CD]],
            outputDTypes: [DType](repeating: .float32, count: loadsFirst ? 6 : 7))
        let ci = outputs[outputs.count - 1]
        return Outputs(
            q: outputs[0], k: outputs[1], v: outputs[2], g: outputs[3], beta: outputs[4],
            tail: loadsFirst ? ci[0..., S..., 0...] : outputs[5], convInput: ci)
    }
}

/// The tree verify's load-time kernel checks (also `MLXFAST_TREE_SELFTEST=1`),
/// once per process at the first GDN layer's construction: (a) every tree
/// row against its root-to-row path run as a chain on the verify's
/// sequential kernel from S0 (tolerance 1e-3); (c) the tap-table prework
/// against each path's stock verify prework, bit for bit. Tree rounds need
/// `passed`. One stderr line per check.
enum Qwen35TreeVerifySelfTest {
    static let requested: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_TREE_SELFTEST"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var done = false
    nonisolated(unsafe) static var passed = false
    nonisolated(unsafe) private static var passA = false
    nonisolated(unsafe) private static var passC = false

    static var wanted: Bool { requested || CBv2TreeVerify.requested }

    private static func report(_ line: String) {
        FileHandle.standardError.write(Data(("qwen35 tree verify self-test: " + line + "\n").utf8))
    }

    /// Deterministic tree shapes of `T` rows.
    static func shapes(T: Int, seed: UInt64) -> [(String, [Int])] {
        var state = seed &* 0x9E37_79B9_7F4A_7C15 | 1
        func next(_ n: Int) -> Int {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Int((state >> 33) % UInt64(n))
        }
        return [
            ("chain", (0 ..< T).map { $0 - 1 }),
            ("star", (0 ..< T).map { $0 == 0 ? -1 : 0 }),
            ("binary", (0 ..< T).map { $0 == 0 ? -1 : ($0 - 1) / 2 }),
            ("random", (0 ..< T).map { $0 == 0 ? -1 : next($0) }),
            ("deep", (0 ..< T).map { $0 == 0 ? -1 : (next(4) == 0 ? next($0) : $0 - 1) }),
            ("twochains", (0 ..< T).map { $0 == 0 ? -1 : max($0 - 2, 0) }),
            ("chain+sib", (0 ..< T).map { $0 == 0 ? -1 : ($0 == T - 1 ? 0 : $0 - 1) }),
        ]
    }

    static func runIfRequested(hk: Int, dk: Int, hv: Int, dv: Int, ks: Int) {
        guard wanted, dk == 128, dv == 128, hk > 0, hv % hk == 0, ks > 1 else { return }
        let first = lock.withLock { () -> Bool in
            if done { return false }
            done = true
            return true
        }
        guard first else { return }
        gdnCheck(hk: hk, hv: hv, dv: dv)
        convCheck(hk: hk, hv: hv, ks: ks)
        passed = passA && passC
        Memory.clearCache()
    }

    struct Tape {
        let q, k, v, g, beta, state: MLXArray
    }

    static func tape(_ seed: UInt64, hk: Int, hv: Int, dv: Int, T: Int) -> Tape {
        let dk = 128
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 10)
        let specials: [Float] = [60, -60, 25, -25, .infinity, -.infinity, 1e-8, -1e-8]
        let marks = MLXArray((0 ..< (2 * hv)).map { specials[$0 % specials.count] })
            .reshaped([1, 1, 2 * hv])
        // One row of saturating and infinite gate inputs (underflowed decays).
        let special = Int(seed % UInt64(T))
        var ab = MLXRandom.normal([1, T, 2 * hv], key: keys[0]) * 4
        ab = MLX.where((MLXArray.arange(T) .== special).reshaped([1, T, 1]), marks, ab)
        let aLog = log(MLXRandom.uniform(Float(1) ..< Float(16), [hv], key: keys[1]))
        let dtBias = MLXRandom.normal([hv], key: keys[2])
        let gates = Qwen35FusedElementwise.gatedDeltaGates(
            [ab[0..., 0..., hv...], ab[0..., 0..., ..<hv], aLog, dtBias])
        let spread = exp(MLXRandom.normal([1, hv, dv, dk], key: keys[3]))
        let state = MLXRandom.normal([1, hv, dv, dk], key: keys[4]) * spread * 0.05
        let q = MLXRandom.normal([1, T, hk, dk], key: keys[5]) * 0.09
        let k = MLXRandom.normal([1, T, hk, dk], key: keys[6]) * 0.09
        let v = MLXRandom.normal([1, T, hv, dv], key: keys[7])
            * exp(MLXRandom.normal([1, T, hv, dv], key: keys[8]))
        let t = Tape(q: q, k: k, v: v, g: gates[0], beta: gates[1], state: state)
        eval(t.q, t.k, t.v, t.g, t.beta, t.state)
        return t
    }

    static func rows(_ x: MLXArray, _ idx: [Int]) -> MLXArray {
        take(x, MLXArray(idx.map { Int32($0) }), axis: 1)
    }

    /// The verify's chain recurrence output rows (`qwen35GatedDelta`'s kernel).
    private static func chain(_ t: Tape, _ path: [Int]) -> MLXArray {
        let (q, k, v, g, b) = (rows(t.q, path), rows(t.k, path), rows(t.v, path), rows(t.g, path), rows(t.beta, path))
        return (Qwen35GatedDeltaV3.run(q: q, k: k, v: v, g: g, beta: b, state: t.state)
            ?? gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: b, state: t.state, mask: nil)).0
    }

    /// (a) every row vs its path as a chain.
    private static func gdnCheck(hk: Int, hv: Int, dv: Int) {
        var (maxAbs, maxMixed, trees, paths, control): (Float, Float, Int, Int, Float) = (0, 0, 0, 0, 0)
        var failure: String? = nil
        do {
            try withError { error in
                // Negative control: a star tree's path against a chain-parent
                // launch (siblings interacting) must differ.
                let t0 = tape(0x636F_6E74, hk: hk, hv: hv, dv: dv, T: 16)
                let star = shapes(T: 16, seed: 1)[1].1
                if let wrong = Qwen35GatedDeltaTree.run(
                    q: t0.q, k: t0.k, v: t0.v, g: t0.g, beta: t0.beta, state: t0.state,
                    parent: MLXArray((0 ..< 16).map { Int32($0 - 1) }))
                {
                    let d = abs(rows(wrong, Qwen35TreeLayout.path(to: 5, parent: star)) - chain(t0, [0, 5])).max()
                    control = d.item(Float.self)
                }
                for tapeIndex in 0 ..< 6 {
                    let T = tapeIndex == 5 ? 11 : 16
                    let tp = tape(0x7472_6565 &+ UInt64(tapeIndex) &* 7919, hk: hk, hv: hv, dv: dv, T: T)
                    for (name, parent) in shapes(T: T, seed: UInt64(tapeIndex + 1)) {
                        guard Qwen35TreeLayout.isValid(parent),
                            let y = Qwen35GatedDeltaTree.run(
                                q: tp.q, k: tp.k, v: tp.v, g: tp.g, beta: tp.beta, state: tp.state,
                                parent: MLXArray(parent.map { Int32($0) }))
                        else {
                            failure = "no tree launch (\(name), T \(T))"
                            return
                        }
                        trees += 1
                        var diffs: [MLXArray] = []
                        var mixed: [MLXArray] = []
                        for leaf in Qwen35TreeLayout.leaves(parent) {
                            let path = Qwen35TreeLayout.path(to: leaf, parent: parent)
                            let reference = chain(tp, path)
                            let d = abs(rows(y, path) - reference)
                            diffs.append(d.max())
                            mixed.append((d / maximum(abs(reference), MLXArray(Float(1)))).max())
                            paths += 1
                        }
                        let stats = [stacked(diffs).max(), stacked(mixed).max()]
                        eval(stats)
                        try error.check()
                        let a = stats[0].item(Float.self)
                        let m = stats[1].item(Float.self)
                        if !(a.isFinite && m.isFinite) { failure = "non-finite (\(name))" }
                        maxAbs = max(maxAbs, a.isNaN ? .infinity : a)
                        maxMixed = max(maxMixed, m.isNaN ? .infinity : m)
                    }
                }
            }
        } catch {
            failure = "\(error)"
        }
        let tol: Float = 1e-3
        passA = failure == nil && maxMixed <= tol && control > tol
        report(
            "(a) tree rows vs root-to-row paths as chains on the sequential kernel: "
                + (passA ? "PASSED" : "FAILED") + " (\(trees) trees, \(paths) paths, max abs \(maxAbs), "
                + "max abs/max(1,|ref|) \(maxMixed), tolerance \(tol); control \(control)"
                + (failure.map { "; \($0)" } ?? "") + ")")
    }

    /// (c) the tap-table prework vs each path's stock verify prework, bitwise.
    private static func convCheck(hk: Int, hv: Int, ks: Int) {
        let (dk, dv, nk, T) = (128, 128, ks - 1, 16)
        let cd = 2 * hk * dk + hv * dv
        let width = cd + hv * dv
        var (values, mismatches, trees, paths) = (0, 0, 0, 0)
        var failure: String? = nil
        do {
            try withError { error in
                for (di, dtype) in [DType.float32, .float16].enumerated() {
                    let keys = MLXRandom.split(key: MLXRandom.key(0x7461_7073 &+ UInt64(di)), into: 10)
                    let spread = MLXRandom.normal([1, T, width], key: keys[0])
                        * exp(MLXRandom.normal([1, T, width], key: keys[1]))
                    // A column slice of a wider product, as the verify's qkv.
                    let qkv = spread.asType(dtype)[.ellipsis, ..<cd]
                    let convState = MLXRandom.normal([1, nk, cd], key: keys[2])
                    let ba = MLXRandom.normal([1, T, 2 * hv], key: keys[3]) * 4
                    let b = ba[.ellipsis, ..<hv]
                    let a = ba[.ellipsis, hv...]
                    let convWeight = MLXRandom.normal([cd, ks, 1], key: keys[4]) * 0.5
                    let aDecay = -exp(MLXRandom.normal([hv], key: keys[5]) * 0.5)
                    let dtb = MLXRandom.normal([hv], key: keys[6])
                    let norms = (q: MLXRandom.normal([dk], key: keys[7]), k: MLXRandom.normal([dk], key: keys[8]))
                    eval(qkv, convState, a, b, convWeight, aDecay, dtb, norms.q, norms.k)
                    for (name, parent) in shapes(T: T, seed: UInt64(11 + di)) {
                        guard
                            let table = CBv2TreeVerifyLayout(parents: parent).convTapTable(kernelSize: ks),
                            let tree = Qwen35GDNPrework.runTree(
                                qkv: qkv, convState: convState, convWeight: convWeight, a: a, b: b,
                                aDecay: aDecay, dtBias: dtb, normScales: norms, keyHeads: hk,
                                valueHeads: hv, headKDim: dk, headVDim: dv, tapTable: table),
                            let treeCI = tree.convInput
                        else {
                            failure = "no tree prework launch (\(name))"
                            return
                        }
                        trees += 1
                        var counts: [MLXArray] = []
                        func compare(_ x: MLXArray, _ y: MLXArray) {
                            counts.append((x.view(dtype: .uint32) .!= y.view(dtype: .uint32)).asType(.int32).sum())
                            values += x.size
                        }
                        for leaf in Qwen35TreeLayout.leaves(parent) {
                            let path = Qwen35TreeLayout.path(to: leaf, parent: parent)
                            guard
                                let chain = Qwen35GDNPrework.run(
                                    qkv: rows(qkv, path), convState: convState, convWeight: convWeight,
                                    a: rows(a, path), b: rows(b, path), aDecay: aDecay, dtBias: dtb,
                                    normScales: norms, keyHeads: hk, valueHeads: hv, headKDim: dk,
                                    headVDim: dv, writeConvInput: true, stridedReads: true),
                                let chainCI = chain.convInput
                            else {
                                failure = "no stock prework launch"
                                return
                            }
                            compare(rows(tree.q, path), chain.q)
                            compare(rows(tree.k, path), chain.k)
                            compare(rows(tree.v, path), chain.v)
                            compare(rows(tree.g, path), chain.g)
                            compare(rows(tree.beta, path), chain.beta)
                            compare(rows(treeCI, Array(0 ..< nk) + path.map { nk + $0 }), chainCI)
                            paths += 1
                        }
                        let count = stacked(counts).sum()
                        eval(count)
                        try error.check()
                        mismatches += Int(count.item(Int32.self))
                    }
                }
            }
        } catch {
            failure = "\(error)"
        }
        passC = failure == nil && mismatches == 0 && trees > 0
        report(
            "(c) conv tap table vs each path's stock verify prework (q, k, v, g, beta, conv input), bitwise: "
                + (passC ? "PASSED" : "FAILED")
                + " (FP32 and FP16 qkv, \(trees) trees, \(paths) paths, \(values) values, \(mismatches) mismatches"
                + (failure.map { "; \($0)" } ?? "") + ")")
    }
}

// MARK: - Tree speculative verify: a GDN layer's token-tree window

/// A GDN layer's token-tree verify window (`CBv2TreeVerifyLayout` bound on
/// the row's recurrent transaction): the tap-table prework, S0 = the
/// committed state (the previous commit's replay), the tree block from S0.
/// The staged replay tape is the window's; the commit compacts it to the
/// accepted path (`Qwen35TreePathTape`) and the chain replay runs unchanged.
enum Qwen35TreeVerifyGDN {
    /// In the load-time warm, a declined window sets `declined` (tree rounds stay off).
    nonisolated(unsafe) static var warming = false
    nonisolated(unsafe) static var declined = false
    nonisolated(unsafe) static var taken = 0

    /// Output rows `[1, S, Hv, Dv]`, replay staged; nil (nothing staged) if unfit.
    static func forward(
        layer: Qwen35GatedDeltaNet, tree: CBv2TreeVerifyLayout, qkv: MLXArray, a: MLXArray,
        b: MLXArray, convState: MLXArray, aDecay: MLXArray,
        normScales: (q: MLXArray, k: MLXArray), evaluation: CBv2RecurrentStateEvaluation,
        modelLayerIndex: Int
    ) -> MLXArray? {
        let S = qkv.dim(1)
        let nKeep = layer.convKernelSize - 1
        guard qkv.ndim == 3, qkv.dim(0) == 1, S == tree.count, S >= 2, nKeep >= 1,
            let parents = tree.deviceParents,
            let taps = tree.convTapTable(kernelSize: layer.convKernelSize),
            let pre = Qwen35GDNPrework.runTree(
                qkv: qkv, convState: convState, convWeight: layer.conv1d.weight, a: a, b: b,
                aDecay: aDecay, dtBias: layer.dtBias, normScales: normScales,
                keyHeads: layer.numKHeads, valueHeads: layer.numVHeads,
                headKDim: layer.headKDim, headVDim: layer.headVDim, tapTable: taps),
            let convInput = pre.convInput
        else { return nil }
        let inputSSM = evaluation.inputState(modelLayerIndex: modelLayerIndex)?.ssm
        let s0 =
            inputSSM
            ?? MLXArray.zeros([1, layer.numVHeads, layer.headVDim, layer.headKDim], dtype: .float32)
        guard
            let y = Qwen35GatedDeltaTree.run(
                q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta, state: s0, parent: parents)
        else { return nil }
        let tape = ArraysCache.PrefixReplayTape(
            convInput: convInput, q: pre.q, k: pre.k, v: pre.v, a: a, b: b, ssmPre: s0,
            mask: nil, rowCount: S, convStateRows: nKeep)
        let path = Qwen35TreePathTape(tape: tape, layout: tree)
        var roots = [convInput, pre.q, pre.k, pre.v, a, b]
        if inputSSM == nil { roots.append(s0) }
        let strictRoots = inputSSM.map { roots + [$0] } ?? roots
        func bytes(_ arrays: [MLXArray]) -> Int {
            arrays.reduce(0) { total, array in
                let (sum, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : sum
            }
        }
        do {
            try evaluation.stagePrefixReplay(
                modelLayerIndex: modelLayerIndex,
                positions: S,
                finalConv: convInput[0..., S ..< (S + nKeep), 0...],
                finalSSM: nil,
                materializedByteCount: bytes(roots + [s0]),
                evaluationRoots: strictRoots,
                strictReplayRetainedByteCount: bytes(strictRoots),
                strictReplayRetainedRoots: strictRoots,
                fullAcceptanceRetainedByteCount: bytes(strictRoots),
                fullAcceptanceRetainedRoots: strictRoots,
                fullAcceptance: { [unowned layer] in
                    layer.replayedPrefixState(tape: path.compacted, committedRows: S, fullWindow: true)
                },
                replay: { [unowned layer] keep in
                    layer.replayedPrefixState(tape: path.compacted, committedRows: keep)
                })
        } catch {
            preconditionFailure(
                "Qwen35 tree verify: replay stage failed at layer \(modelLayerIndex): \(error)")
        }
        taken += 1
        return y
    }
}

/// A tree window's replay tape compacted to `acceptedRows` (path rows first,
/// then the rest; the replay never reads q). A path that is the window's
/// leading chain is the tape itself.
final class Qwen35TreePathTape {
    let tape: ArraysCache.PrefixReplayTape
    let layout: CBv2TreeVerifyLayout
    private var built: ArraysCache.PrefixReplayTape?

    init(tape: ArraysCache.PrefixReplayTape, layout: CBv2TreeVerifyLayout) {
        (self.tape, self.layout) = (tape, layout)
    }

    var compacted: ArraysCache.PrefixReplayTape {
        if let built { return built }
        let S = tape.rowCount
        guard let rows = layout.acceptedRows, rows.allSatisfy({ $0 >= 0 && $0 < S }),
            Set(rows).count == rows.count
        else {
            preconditionFailure("Qwen35 tree verify: the commit read a tape with no accepted path")
        }
        let chosen = Set(rows)
        let order = rows + (0 ..< S).filter { !chosen.contains($0) }
        let result: ArraysCache.PrefixReplayTape
        if order == Array(0 ..< S) {
            result = tape
        } else {
            let nk = tape.convStateRows
            let index = MLXArray(order.map { Int32($0) })
            let convIndex = MLXArray((0 ..< nk).map { Int32($0) } + order.map { Int32(nk + $0) })
            result = ArraysCache.PrefixReplayTape(
                convInput: take(tape.convInput, convIndex, axis: 1), q: tape.q,
                k: take(tape.k, index, axis: 1), v: take(tape.v, index, axis: 1),
                a: take(tape.a, index, axis: 1), b: take(tape.b, index, axis: 1),
                ssmPre: tape.ssmPre, mask: nil, rowCount: S, convStateRows: nk)
        }
        built = result
        return result
    }
}

// MARK: - Tree speculative verify: the attention check

/// The fused attention prework's token-tree form: row t rotates at
/// `offs + tdep[t]` (int32 `[L]`, the rows' depths) instead of `offs + t`. One
/// edit keeping the types (`uint + int`, then to float), so a chain's depths
/// give the same bits.
extension Qwen35AttentionPrework {
    private static let treeKernel: MLXFast.MLXFastKernel = {
        let pattern = "static_cast<float>(t + off)"
        precondition(
            source.components(separatedBy: pattern).count == 2,
            "qwen35 tree rotary: the rotary position must occur once in the source")
        return MLXFast.metalKernel(
            name: "bonsai_attn_prework_tree",
            inputNames: ["q", "k", "wq", "wk", "offs", "epsq", "epsk", "axis", "lbase", "scale", "tdep"],
            outputNames: ["qo", "ko"],
            source: source.replacingOccurrences(
                of: pattern, with: "static_cast<float>(uint(tdep[int64_t(t) * tdep_strides[0]]) + off)"),
            ensureRowContiguous: false)
    }()

    /// `run` for a token-tree window (one row); the same verdict.
    static func runTree(
        q: MLXArray, k: MLXArray, qNorm: RMSNorm, kNorm: RMSNorm,
        offsets: MLXArray, depths: MLXArray, ropeDims: Int, ropeBase: Float
    ) -> (MLXArray, MLXArray)? {
        guard enabled, q.ndim == 4, k.ndim == 4,
            verified(
                Geometry(hq: q.dim(2), hk: k.dim(2), d: q.dim(3), rd: ropeDims, dtype: "\(q.dtype)"))
        else { return nil }
        return runTreeUnchecked(
            q: q, k: k, wq: qNorm.weight, wk: kNorm.weight, epsQ: qNorm.eps, epsK: kNorm.eps,
            offsets: offsets, depths: depths, ropeDims: ropeDims, ropeBase: ropeBase)
    }

    static func runTreeUnchecked(
        q: MLXArray, k: MLXArray, wq: MLXArray, wk: MLXArray, epsQ: Float, epsK: Float,
        offsets: MLXArray, depths: MLXArray, ropeDims: Int, ropeBase: Float
    ) -> (MLXArray, MLXArray)? {
        let (L, HQ, HK, D) = (q.dim(1), q.dim(2), k.dim(2), q.dim(3))
        guard q.dim(0) == 1, k.dim(0) == 1, k.dim(1) == L, k.dim(3) == D,
            q.dtype == k.dtype, [DType.float32, .float16, .bfloat16].contains(q.dtype),
            wq.dtype == .float32, wk.dtype == .float32, wq.shape == [D], wk.shape == [D],
            offsets.dtype == .int32, offsets.ndim <= 1, offsets.size == 1,
            depths.dtype == .int32, depths.shape == [L], L > 0, L <= CBv2TreeVerifyLayout.maximumRows
        else { return nil }
        let offs = offsets.ndim == 1 ? offsets : offsets.reshaped([1])
        let outputs = treeKernel(
            [q, k, wq, wk, offs, MLXArray(epsQ), MLXArray(epsK), MLXArray(UInt32(D)),
             MLXArray(log2(ropeBase)), MLXArray(Float(1)), depths],
            template: [("D", D), ("RD", ropeDims), ("HQ", HQ), ("HK", HK), ("OB", 1)],
            grid: ((D / 4) * (HQ + HK), L, 1), threadGroup: (D / 4, 1, 1),
            outputShapes: [[1, HQ, L, D], [1, HK, L, D]],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }
}

/// The tree verify's attention check (also `MLXFAST_TREE_ATTN_SELFTEST=1`),
/// once per attention geometry at model construction, FP32 at the layer's
/// real shapes, through both verify routes (the fused prework with the
/// cache's slice updates, and the op chain) over the verify attention's
/// three key-count forms (the verify block's fused softmax, its mask kernel,
/// SDPA over the tree's boolean mask): a chain-shaped tree against the
/// unbound window, bit for bit (outputs, stored K/V, offsets); 7 trees, each
/// row against the last row of its root-to-node path run as an unbound chain
/// (max |diff| <= 1e-3 * max(1, max |ref|)) and each stored K/V row against
/// that chain's row at its depth, bit for bit; a control (a star tree's rows
/// differ from the causal window's). Tree rounds need `verdict == true`.
enum Qwen35TreeAttentionSelfTest {
    static let requested: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_TREE_ATTN_SELFTEST"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var tested: Set<String> = []
    /// Every geometry checked passed (nil: none checked).
    nonisolated(unsafe) static var verdict: Bool?

    private enum Route: String, CaseIterable {
        case slices = "prework + slice updates"
        case opChain = "op chain"
    }

    private struct Geometry {
        let hq: Int, hk: Int, d: Int, rd: Int
        let ropeBase: Float, epsQ: Float, epsK: Float
    }

    private typealias Operands = (q: MLXArray, k: MLXArray, v: MLXArray)

    static func runIfRequested(
        hq: Int, hk: Int, d: Int, ropeDims rd: Int, ropeBase: Float, epsQ: Float, epsK: Float
    ) {
        guard requested || CBv2TreeVerify.requested,
            lock.withLock({ tested.insert("\(hq) \(hk) \(d) \(rd)").inserted })
        else { return }
        let geo = Geometry(hq: hq, hk: hk, d: d, rd: rd, ropeBase: ropeBase, epsQ: epsQ, epsK: epsK)
        var (failures, ran) = (0, 0)
        var details: [String] = []
        for route in Route.allCases {
            for prefix in [611, 4200, 4201] {
                let result = check(route, geo, prefix: prefix)
                if result.ran { ran += 1 }
                if result.ran && !result.passed { failures += 1 }
                details.append(
                    "[\(route.rawValue); \(prefix)+16 keys] \(result.detail): "
                        + (result.ran ? (result.passed ? "ok" : "FAILED") : "not run"))
            }
        }
        let passed = failures == 0 && ran > 0
        verdict = (verdict ?? true) && passed
        FileHandle.standardError.write(
            Data(("qwen35 tree attention self-test (hq \(hq), hk \(hk), d \(d), rotary \(rd)): "
                + "\(ran) checks, \(failures) failed; " + (passed ? "PASSED" : "FAILED")
                + (passed && !requested ? "" : "\n  " + details.joined(separator: "\n  ")) + "\n").utf8))
        Memory.clearCache()
    }

    private static func trees() -> [[Int]] {
        var trees: [[Int]] = [
            (0 ..< 16).map { $0 - 1 },
            [-1] + [Int](repeating: 0, count: 15),
            (0 ..< 16).map { $0 == 0 ? -1 : ($0 - 1) / 2 },
            [-1, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 8, 9, 12, 4],
        ]
        var state: UInt64 = 0x7472_6565_5f61_7474
        for _ in 0 ..< 3 {
            var parents = [-1]
            for i in 1 ..< 16 {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                parents.append(Int((state >> 33) % UInt64(i)))
            }
            trees.append(parents)
        }
        return trees
    }

    private static func check(_ route: Route, _ geo: Geometry, prefix: Int)
        -> (ran: Bool, passed: Bool, detail: String)
    {
        if route == .slices, !Qwen35AttentionPrework.enabled {
            return (false, false, "fused prework off")
        }
        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x7472_6565 + prefix)), into: 6)
        let wq = 1 + 0.25 * MLXRandom.normal([geo.d], key: keys[0])
        let wk = 1 + 0.25 * MLXRandom.normal([geo.d], key: keys[1])
        let scale = pow(Float(geo.d), -0.5)
        let rows = 16
        let pk = MLXRandom.normal([1, geo.hk, prefix, geo.d], key: keys[2])
        let pv = MLXRandom.normal([1, geo.hk, prefix, geo.d], key: keys[3])
        let kind = CBv2LayerKind(attention: .full, headDim: geo.d, kvHeads: geo.hk, queryHeads: geo.hq)
        func makeLayer() -> CBv2LayerCache {
            let row = CBv2FullSequenceKV(
                promptLength: prefix, maxLength: 1 << 16, kvHeads: geo.hk, headDim: geo.d)
            _ = row.update(keys: pk, values: pv)
            return CBv2LayerCache(layerIndex: 0, kind: kind, rows: [row])
        }
        func rollback(_ layer: CBv2LayerCache, _ n: Int) {
            let bound = layer.rows
            for row in bound { row.rollback(n) }
            layer.setRows(bound)
        }
        let width = geo.hq * 2 * geo.d + 2 * geo.hk * geo.d
        let wide = MLXRandom.normal([1, rows, width], key: keys[4])
            * exp(MLXRandom.normal([1, rows, width], key: keys[5]))
        let parts = MLX.split(
            wide, indices: [geo.hq * 2 * geo.d, geo.hq * 2 * geo.d + geo.hk * geo.d], axis: -1)
        let o: Operands = (
            parts[0].reshaped(1, rows, geo.hq, -1).split(parts: 2, axis: -1)[0],
            parts[1].reshaped(1, rows, geo.hk, -1),
            parts[2].reshaped(1, rows, geo.hk, -1)
        )
        func differing(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            (a.view(dtype: .uint32) .!= b.view(dtype: .uint32)).asType(.int32).sum()
        }
        var (chainMismatches, kvMismatches, comparedRows) = (0, 0, 0)
        var (worst, control): (Float, Float) = (0, 0)
        do {
            try withError { error in
                // 1. A chain-shaped tree against the unbound window.
                let plainLayer = makeLayer()
                let chainLayer = makeLayer()
                let plain = forward(route, plainLayer, o, tree: nil, geo, wq: wq, wk: wk, scale: scale)
                let chained = forward(
                    route, chainLayer, o, tree: .chain(rows), geo, wq: wq, wk: wk, scale: scale)
                let a = plainLayer.rows[0].snapshot()
                let b = chainLayer.rows[0].snapshot()
                let count = stacked([
                    differing(plain, chained), differing(a.keys, b.keys), differing(a.values, b.values),
                    (plainLayer.positionOffsets .!= chainLayer.positionOffsets).asType(.int32).sum(),
                ]).sum()
                eval(count)
                try error.check()
                chainMismatches = Int(count.item(Int32.self))
                // 2. Trees against each row's root-to-node path as a plain chain.
                let treeLayer = makeLayer()
                let pathLayer = makeLayer()
                for (treeIndex, parents) in trees().enumerated() {
                    let out = forward(
                        route, treeLayer, o, tree: CBv2TreeVerifyLayout(parents: parents), geo,
                        wq: wq, wk: wk, scale: scale)
                    let tree = treeLayer.rows[0].snapshot()
                    eval(out, tree.keys, tree.values)
                    try error.check()
                    if treeIndex == 1 {
                        let causal = plain[0..., 0..., 2..., 0...]
                        let apart: MLXArray = abs(out[0..., 0..., 2..., 0...] - causal).max()
                            / maximum(abs(causal).max(), MLXArray(Float(1)))
                        control = apart.item(Float.self)
                    }
                    for i in 0 ..< rows {
                        let path = Qwen35TreeLayout.path(to: i, parent: parents)
                        let depth = path.count - 1
                        let index = MLXArray(path.map { Int32($0) })
                        let po: Operands = (
                            o.q.take(index, axis: 1), o.k.take(index, axis: 1), o.v.take(index, axis: 1)
                        )
                        let reference = forward(
                            route, pathLayer, po, tree: nil, geo, wq: wq, wk: wk, scale: scale)
                        let chain = pathLayer.rows[0].snapshot()
                        let expected = reference[0..., 0..., depth ..< (depth + 1), 0...]
                        let actual = out[0..., 0..., i ..< (i + 1), 0...]
                        let (slot, chainSlot) = (prefix + i, prefix + depth)
                        let kvDiff =
                            differing(
                                tree.keys[0..., 0..., slot ..< (slot + 1), 0...],
                                chain.keys[0..., 0..., chainSlot ..< (chainSlot + 1), 0...])
                            + differing(
                                tree.values[0..., 0..., slot ..< (slot + 1), 0...],
                                chain.values[0..., 0..., chainSlot ..< (chainSlot + 1), 0...])
                        let metrics = stacked([
                            abs(actual - expected).max(), abs(expected).max(), kvDiff.asType(.float32),
                        ])
                        eval(metrics)
                        try error.check()
                        let m = metrics.asArray(Float.self)
                        let normalized = m[0] / max(1, m[1])
                        worst = normalized.isFinite ? max(worst, normalized) : .infinity
                        kvMismatches += Int(m[2])
                        comparedRows += 1
                        rollback(pathLayer, depth + 1)
                    }
                    rollback(treeLayer, rows)
                }
            }
        } catch {
            return (true, false, "\(error)")
        }
        let passed = chainMismatches == 0 && kvMismatches == 0 && worst <= 1e-3
            && comparedRows == 7 * rows && control > 1e-2
        return (
            true, passed,
            "chain tree vs unbound \(chainMismatches) differ; \(comparedRows) rows vs path chains max err "
                + String(format: "%.3g", worst) + ", K/V at depth \(kvMismatches) differ; control "
                + String(format: "%.3g", control))
    }

    /// One attention layer's verify window through `route`, `tree` bound for the call.
    private static func forward(
        _ route: Route, _ layer: CBv2LayerCache, _ o: Operands, tree: CBv2TreeVerifyLayout?,
        _ geo: Geometry, wq: MLXArray, wk: MLXArray, scale: Float
    ) -> MLXArray {
        layer.bindTreeVerify(tree)
        defer { layer.bindTreeVerify(nil) }
        let values = o.v.transposed(0, 2, 1, 3)
        let offsets = layer.positionOffsets
        switch route {
        case .slices:
            let rotated =
                tree.flatMap {
                    Qwen35AttentionPrework.runTreeUnchecked(
                        q: o.q, k: o.k, wq: wq, wk: wk, epsQ: geo.epsQ, epsK: geo.epsK,
                        offsets: offsets, depths: $0.depths, ropeDims: geo.rd, ropeBase: geo.ropeBase)
                }
                ?? Qwen35AttentionPrework.runUnchecked(
                    q: o.q, k: o.k, wq: wq, wk: wk, epsQ: geo.epsQ, epsK: geo.epsK,
                    offsets: offsets, ropeDims: geo.rd, ropeBase: geo.ropeBase)!
            return layer.updateAndAttend(
                queries: rotated.0, keys: rotated.1, values: values, scale: scale, sinks: nil)
        case .opChain:
            // As `Qwen35Attention.cbv2Forward` composes it.
            func rotate(_ x: MLXArray, _ w: MLXArray, _ eps: Float) -> MLXArray {
                let normed = MLXFast.rmsNorm(x, weight: w, eps: eps).transposed(0, 2, 1, 3)
                guard let tree else {
                    return MLXFast.RoPE(
                        normed, dimensions: geo.rd, traditional: false, base: geo.ropeBase,
                        scale: 1, offset: offsets + 0)
                }
                return MLXFast.RoPE(
                    normed.transposed(2, 1, 0, 3), dimensions: geo.rd, traditional: false,
                    base: geo.ropeBase, scale: 1, offset: (offsets + 0) + tree.depths
                ).transposed(2, 1, 0, 3)
            }
            return layer.updateAndAttend(
                queries: rotate(o.q, wq, geo.epsQ), keys: rotate(o.k, wk, geo.epsK),
                values: values, scale: scale, sinks: nil)
        }
    }
}
