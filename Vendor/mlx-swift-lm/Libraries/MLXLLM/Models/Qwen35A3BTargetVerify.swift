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

    /// The row-tiled launch on `runFreshState`'s arguments (after its guards
    /// and dtype conversions), `rows` rows per threadgroup; `rows` must divide
    /// the chunk. The outputs have `freshStridedKernel`'s shapes and dtypes.
    static func freshStridedRows(
        qkv: MLXArray, convWeight: MLXArray, a: MLXArray, b: MLXArray,
        decay: MLXArray, dtb: MLXArray, normScales: (q: MLXArray, k: MLXArray),
        keyHeads: Int, valueHeads: Int, headKDim: Int, headVDim: Int, rows: Int
    ) -> Outputs {
        let B = qkv.dim(0)
        let S = qkv.dim(1)
        let CD = qkv.dim(2)
        let KS = convWeight.dim(1)
        precondition(
            headKDim == 128 && rows > 0 && S % rows == 0
                && (valueHeads / keyHeads) * rows <= headKDim,
            "Qwen35 GDN prework rows: unsupported launch")
        let outputs = freshStridedRowsKernel(
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
            let verdict = rowTileSelfCheck(hk: hk, dk: dk, hv: hv, dv: dv, ks: ks, dtype: dtype)
            let recorded = rowTileLock.withLock { () -> Bool in
                guard rowTileVerdicts[geometry] == nil else { return false }
                rowTileVerdicts[geometry] = verdict
                return true
            }
            if recorded && !verdict {
                FileHandle.standardError.write(
                    "qwen35: GDN row-tiled prework kernel disagrees with the stock kernel on this device (\(dtype)); using the stock kernel\n"
                        .data(using: .utf8)!)
            }
        }
    }

    private static func rowTileSelfCheck(
        hk: Int, dk: Int, hv: Int, dv: Int, ks: Int, dtype: DType
    ) -> Bool {
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
            else { return false }
            let tiled = freshStridedRows(
                qkv: qkv, convWeight: convWeight, a: a, b: b, decay: aDecay, dtb: dtBias,
                normScales: normScales, keyHeads: hk, valueHeads: hv, headKDim: dk,
                headVDim: dv, rows: rowTile)
            var same = MLXArray(true)
            for (x, y) in [
                (stock.q, tiled.q), (stock.k, tiled.k), (stock.v, tiled.v),
                (stock.g, tiled.g), (stock.beta, tiled.beta), (stock.tail, tiled.tail),
            ] {
                same = same .&& all(x.view(dtype: .uint32) .== y.view(dtype: .uint32))
            }
            if !same.item(Bool.self) { return false }
        }
        return true
    }
}
