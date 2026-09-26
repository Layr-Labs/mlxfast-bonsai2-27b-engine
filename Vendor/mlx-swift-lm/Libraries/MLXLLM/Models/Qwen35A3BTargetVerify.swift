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

// Moved from Qwen35.swift unchanged (same module) to keep that file under the
// editable surface's per-file byte cap.

/// The verify window's producer chains on the int8 narrow route (16 rows):
/// the SwiGLU product, the GDN output's gated norm and the attention output
/// gate are formed in the quantizing rotation's read (the prompt route's
/// `..._q8p`) instead of by their own launches ahead of
/// `bonsai_signed_hadamard_1024_q8`. Per verify window of the 27B: 64
/// compiled SwiGLU launches, 48 norms and 48 compiled gated tails, and 16
/// compiled gates with the 32 copies that flattened their operands fewer.
///
/// Each kind is self-tested once per width and dtypes on the running GPU at
/// 16 rows, bit for bit against the composed ops production runs (the
/// compiled chain with the signs, then the pre-signed `forwardInt8`), on
/// operands laid out as production lays them out; a mismatch keeps the
/// composed path. `BONSAI_VERIFY_SWIGLU_Q8=0`, `BONSAI_VERIFY_GATED_NORM_Q8=0`
/// and `BONSAI_VERIFY_ATTN_GATE_Q8=0` keep it per kind.
enum Qwen35VerifyProducerQ8 {
    private static func on(_ name: String) -> Bool {
        let value = ProcessInfo.processInfo.environment[name]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }
    static let swiglu = on("BONSAI_VERIFY_SWIGLU_Q8")
    static let gatedNorm = on("BONSAI_VERIFY_GATED_NORM_Q8")
    static let attentionGate = on("BONSAI_VERIFY_ATTN_GATE_Q8")

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [String: Bool] = [:]

    /// `HadamardQuantizedLinear.narrowProducerApproves`: the kind's switch,
    /// the operand dtypes the composed chain computes in FP32, and the kind's
    /// verdict (its self-test runs on first use).
    static func approves(
        _ producer: SignedBlockHadamard.Int8Producer, _ transform: SignedBlockHadamard
    ) -> Bool {
        // The composed chains compared against carry the signs (the default fold).
        guard Qwen35FusedElementwise.foldsHadamardSigns, transform.blockSize == 1024
        else { return false }
        let key: String
        switch producer {
        case .swiglu(let gate, let up):
            guard swiglu, gate.dtype == up.dtype, [DType.float16, .float32].contains(gate.dtype)
            else { return false }
            key = "swiglu \(transform.width) \(gate.dtype)"
        case .gatedRMSNorm(let x, let gate, let weight, _):
            guard gatedNorm, x.dtype == .float32, x.ndim == 4, x.dim(3) == 128,
                [DType.float16, .float32].contains(gate.dtype), weight.dtype == .float32
            else { return false }
            key = "gated norm \(transform.width) \(gate.dtype)"
        case .sigmoidGate(let x, let gate):
            guard attentionGate, x.dtype == .float32, gate.dtype == .float32, x.ndim == 4
            else { return false }
            key = "attention gate \(transform.width) \(x.dim(2))x\(x.dim(3))"
        }
        lock.lock()
        defer { lock.unlock() }
        if let verdict = verdicts[key] { return verdict }
        let report = selfTest(producer, transform)
        verdicts[key] = report.passed
        FileHandle.standardError.write(
            ("bonsai verify producer q8 (\(key)): " + report.summary
                + (report.passed ? "; fused\n" : "; composed path kept\n")).data(using: .utf8)!)
        return report.passed
    }

    /// Sixteen rows with per-row scales from 0.05 to 30 (the sigmoid saturates
    /// both ways) and one zero row (all-zero groups, the norm's `rsqrt(eps)`),
    /// in production's layouts (SwiGLU's halves, the GDN gate and the attention
    /// gate are column slices of a stacked product, the attention output is
    /// head-transposed). Outputs compared as unsigned integers.
    private static func selfTest(
        _ producer: SignedBlockHadamard.Int8Producer, _ transform: SignedBlockHadamard
    ) -> Qwen35FusedBoundaryQ8.SelfTestReport {
        var report = Qwen35FusedBoundaryQ8.SelfTestReport()
        let rows = 16
        let width = transform.width
        let signs = transform.signVector
        do {
            try withError { error in
                for seed in [61, 62] {
                    let scale = MLXRandom.uniform(
                        Float(0.05) ..< Float(30), [1, rows, 1], key: MLXRandom.key(UInt64(seed)))
                        * (MLXArray(0 ..< rows) .!= MLXArray(Int32(rows / 3)))
                            .asType(.float32).reshaped(1, rows, 1)
                    func normal(_ n: Int, _ salt: Int) -> MLXArray {
                        MLXRandom.normal([1, rows, n], key: MLXRandom.key(UInt64(seed * 8 + salt)))
                            * scale
                    }
                    let signed: MLXArray
                    let fused: SignedBlockHadamard.Int8Producer
                    switch producer {
                    case .swiglu(let gate, _):
                        let halves = split(normal(2 * width, 1).asType(gate.dtype), parts: 2, axis: -1)
                        signed = Qwen35FusedElementwise.swigluSigned(halves[0], halves[1], signs)
                        fused = .swiglu(gate: halves[0], up: halves[1])
                    case .gatedRMSNorm(let x, let gate, let weight, let eps):
                        let shape = [1, rows, x.dim(2), x.dim(3)]
                        let out = normal(width, 2).reshaped(shape)
                        let z = split(normal(2 * width, 3).asType(gate.dtype), parts: 2, axis: -1)[1]
                            .reshaped(shape)
                        let normed = MLXFast.rmsNorm(out, weight: weight, eps: eps)
                        signed = Qwen35FusedElementwise.gatedNormTailSigned(
                            normed, z.asType(.float32), signs.reshaped(x.dim(2), x.dim(3)))
                        fused = .gatedRMSNorm(x: out, gate: z, weight: weight, eps: eps)
                    case .sigmoidGate(let x, _):
                        // The head-transposed attention output and the gate
                        // half of each q|gate head.
                        let (heads, dim) = (x.dim(2), x.dim(3))
                        let xs = (MLXRandom.normal(
                            [1, heads, rows, dim], key: MLXRandom.key(UInt64(seed * 8 + 4)))
                            * scale.reshaped(1, 1, rows, 1)).transposed(0, 2, 1, 3)
                        let gs = normal(2 * width, 5).reshaped(1, rows, heads, 2 * dim)
                            .split(parts: 2, axis: -1)[1]
                        signed = Qwen35FusedElementwise.sigmoidGateSigned(
                            xs.reshaped(1, rows, -1), gs.reshaped(1, rows, -1), signs)
                        fused = .sigmoidGate(x: xs, gate: gs)
                    }
                    guard
                        let a0 = transform.forwardInt8(
                            signed.reshaped(rows, width), gdnLayout: nil, preSigned: true,
                            groupSize: 128),
                        let a1 = transform.forwardInt8(
                            producer: fused, gdnLayout: nil, groupSize: 128)
                    else {
                        report.passed = false
                        report.error = "a quantizing rotation is not installed"
                        return
                    }
                    report.cases += 1
                    for (a, b) in [
                        (a0.codes, a1.codes), (a0.scales, a1.scales),
                        (a0.scaledSums, a1.scaledSums),
                    ] {
                        guard a.dtype == b.dtype, a.shape == b.shape else {
                            report.passed = false
                            report.error = "output \(b.dtype) \(b.shape) vs \(a.dtype) \(a.shape)"
                            return
                        }
                        let bits: DType = a.dtype == .float32 ? .uint32 : a.dtype
                        let differ = (a.view(dtype: bits) .!= b.view(dtype: bits))
                            .asType(.int32).sum()
                        eval(differ)
                        try error.check()
                        let count = Int(differ.item(Int32.self))
                        report.values += a.size
                        report.mismatches += count
                        if count != 0 { report.passed = false }
                    }
                }
            }
        } catch {
            report.passed = false
            report.error = "\(error)"
        }
        return report
    }
}

/// The GDN output's gated per-head RMSNorm and the output projection's
/// Hadamard signs at verify width as ONE launch. The composed path runs
/// `MLXFast.rmsNorm` over each 128-wide head (`rms_single_row`) and the
/// compiled `gatedNormTailSigned` (`(z * sigmoid(z)) * normed * signs`): two
/// launches per GDN layer. The arithmetic here is the int8 producer kernel's
/// PROD 3 read, which already matches that composed chain bit for bit: per
/// head, lane l squares elements 4l .. 4l + 3 in order, `simd_sum`,
/// `precise::rsqrt(acc / 128 + eps)`, `w[d] * (x * inv)`, then `(z *
/// sigmoid(z)) * xn` with MLX's `Sigmoid` and the signs. One simdgroup per
/// (row, head), as `rms_single_row`: the launch keeps the norm's parallelism
/// and the quantizing rotation stays its own launch. z is read through its
/// strides (a slice of the qkv|z product), so it is not copied first.
///
/// Before first use a self-test on the running GPU compares the output bit for
/// bit against the composed ops (FP32 and FP16 z, a strided z); a mismatch or
/// any MLX error keeps the composed path. `BONSAI_VERIFY_GATEDNORM=0` keeps it
/// too.
enum Qwen35GatedNormTail {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_VERIFY_GATEDNORM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let header = """
        // MLX `Sigmoid` (unary_ops.h), verbatim.
        METAL_FUNC float bgn_sigmoid(float x) {
          auto y = 1 / (1 + metal::exp(metal::abs(x)));
          return (x < 0) ? y : 1 - y;
        }
        // element (row, c) of a [B, L, heads, 128] view, through its strides
        inline int64_t bgn_row(const constant int* shape, const constant int64_t* st, uint row) {
          const uint L = uint(shape[1]);
          return int64_t(row / L) * st[0] + int64_t(row % L) * st[1];
        }
        inline int64_t bgn_col(const constant int64_t* st, uint c) {
          return int64_t(c / 128u) * st[2] + int64_t(c % 128u) * st[3];
        }

        """

    // grid (32 * rows * H, 1, 1), threadgroup (256, 1, 1): one simdgroup per
    // (row, head). Inputs: x float [B, L, H, 128], z float|half [B, L, H,
    // 128] (any strides), w float [128], eps float [1], signs float [H * 128].
    // Template: H, InZ. Output: out float [rows, H * 128].
    private static let source = """
        const uint gidx = thread_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const uint hr = gidx / 32u;
        const uint row = hr / uint(H);
        const uint head = hr % uint(H);
        const int64_t xrow = bgn_row(x_shape, x_strides, row);
        const int64_t zrow = bgn_row(z_shape, z_strides, row);
        const uint c0 = head * 128u + lane * 4u;
        float xv[4];
        float acc = 0.0f;
        #pragma clang loop unroll(full)
        for (int r = 0; r < 4; r++) {
          xv[r] = float(x[xrow + bgn_col(x_strides, c0 + uint(r))]);
          acc += xv[r] * xv[r];
        }
        acc = simd_sum(acc);
        const float inv = metal::precise::rsqrt(acc / float(128) + eps[0]);
        #pragma clang loop unroll(full)
        for (int r = 0; r < 4; r++) {
          const uint col = c0 + uint(r);
          const float bv = float(z[zrow + bgn_col(z_strides, col)]);
          const float xn = w[col % 128u] * (xv[r] * inv);
          const float v = (bv * bgn_sigmoid(bv)) * xn;
          out[size_t(row) * size_t(H * 128) + col] = v * signs[col];
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_gdn_gated_norm_signed",
        inputNames: ["x", "z", "w", "eps", "signs"],
        outputNames: ["out"],
        source: source,
        header: header,
        ensureRowContiguous: false)

    /// `gatedNormTailSigned(rmsNorm(x, weight, eps), z, signs)` flattened to
    /// `[rows, H * 128]`, or nil when it does not apply.
    static func apply(
        _ x: MLXArray, gate z: MLXArray, weight: MLXArray, eps: Float, signs: MLXArray
    ) -> MLXArray? {
        guard enabled, x.dtype == .float32, z.dtype == .float32 || z.dtype == .float16,
            x.ndim == 4, z.shape == x.shape, x.dim(3) == 128,
            (x.dim(0) * x.dim(1) * x.dim(2)) % 8 == 0,
            weight.dtype == .float32, weight.ndim == 1, weight.dim(0) == 128,
            signs.dtype == .float32, signs.size == x.dim(2) * 128,
            verified(eps: eps)
        else { return nil }
        return launch(x, z, weight: weight, eps: eps, signs: signs)
    }

    private static func launch(
        _ x: MLXArray, _ z: MLXArray, weight: MLXArray, eps: Float, signs: MLXArray
    ) -> MLXArray {
        let rows = x.dim(0) * x.dim(1)
        let heads = x.dim(2)
        return kernel(
            [x, z, weight, MLXArray([eps]), signs.reshaped(-1)],
            template: [("H", heads), ("InZ", z.dtype)],
            grid: (32 * rows * heads, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[rows, heads * 128]], outputDTypes: [.float32])[0]
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdict: Bool?

    private static func verified(eps: Float) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if let verdict { return verdict }
        let report = selfTest(eps: eps)
        verdict = report.passed
        FileHandle.standardError.write(
            ("bonsai verify gated norm: " + report.summary
                + (report.passed ? "; fused\n" : "; composed path kept\n")).data(using: .utf8)!)
        return report.passed
    }

    static func selfTest(eps: Float) -> Qwen35FusedBoundaryQ8.SelfTestReport {
        var report = Qwen35FusedBoundaryQ8.SelfTestReport()
        do {
            try withError { error in
                let rows = 16
                let heads = 48
                let width = heads * 128
                let signs = which(
                    MLXRandom.uniform(Float(0) ..< Float(1), [width], key: MLXRandom.key(121))
                        .< Float(0.5), MLXArray(Float(-1)), MLXArray(Float(1)))
                let weight = MLXRandom.uniform(
                    Float(0.5) ..< Float(1.5), [128], key: MLXRandom.key(122))
                let scale = MLXRandom.uniform(
                    Float(0.01) ..< Float(20), [1, rows, heads, 1], key: MLXRandom.key(123))
                let x = MLXRandom.normal([1, rows, heads, 128], key: MLXRandom.key(124)) * scale
                // z as the qkv|z product's slice: a strided view.
                let wide = MLXRandom.normal([1, rows, 10240 + width], key: MLXRandom.key(125))
                    * Float(3)
                for zType in [DType.float32, .float16] {
                    let z = wide.asType(zType)[0..., 0..., 10240...].reshaped(1, rows, heads, 128)
                    let zw = z.dtype == x.dtype ? z : z.asType(x.dtype)
                    let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
                    let reference = Qwen35FusedElementwise.gatedNormTailSigned(
                        normed, zw, signs.reshaped(heads, 128)
                    ).asType(x.dtype).reshaped(rows, width)
                    let fused = launch(x, z, weight: weight, eps: eps, signs: signs)
                    report.cases += 1
                    guard fused.shape == reference.shape, fused.dtype == reference.dtype else {
                        report.passed = false
                        report.error = "output \(fused.dtype) \(fused.shape)"
                        return
                    }
                    let differ = (fused.view(dtype: .uint32) .!= reference.view(dtype: .uint32))
                        .asType(.int32).sum()
                    eval(differ)
                    try error.check()
                    let count = Int(differ.item(Int32.self))
                    report.values += fused.size
                    report.mismatches += count
                    if count != 0 { report.passed = false }
                }
            }
        } catch {
            report.passed = false
            report.error = "\(error)"
        }
        return report
    }
}
