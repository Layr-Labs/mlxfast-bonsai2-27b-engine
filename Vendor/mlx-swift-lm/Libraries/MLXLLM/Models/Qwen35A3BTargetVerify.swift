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
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray
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
        let out = kernel(
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
        let (passed, detail) = selfTest(layer: layer)
        verdicts[key] = passed
        Memory.clearCache()
        FileHandle.standardError.write(
            ("qwen35 GDN replay fused: self-test " + (passed ? "passed" : "FAILED") + " ("
                + detail + ")"
                + (passed ? "; the next verify replays the committed prefix\n" : "; replay kept\n"))
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
        let expected = 4 * G * 2 + (batched ? 3 * G : 0)
        let passed = mismatches == 0 && comparisons == expected
        return (
            passed,
            "\(G) tapes at 0, 1, 7 and \(S) kept rows, \(comparisons) comparisons, "
                + "\(values) values, \(mismatches) mismatches"
                + (batched ? ", batched replay included" : ""))
    }
}
