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
        guard let parts = Qwen35GatedDeltaV3.sourceParts, let replayBlock else { return nil }
        let load = "state[d][i] = state_in[(n * Dv + dvbase + d) * Dk + dk0 + i];"
        guard parts.head.components(separatedBy: load).count == 2 else { return nil }
        let head = parts.head.replacingOccurrences(
            of: load, with: "state[d][i] = ps[(n * Dv + dvbase + d) * Dk + dk0 + i];")
        guard !head.contains("state_in") else { return nil }
        return head + replayBlock + parts.store + "\n" + parts.loop + "\n"
    }()

    /// The nested replay block of `source` (the previous tape's `KP` rows
    /// from device pointers `pk`, `pv`, `pa`, `pb`, gates from `alog`/`dtb`),
    /// also the replay phase of `Qwen35GDNVerifyFused`. Nil when the stock
    /// loop no longer carries the replaced lines.
    static let replayBlock: String? = {
        guard let parts = Qwen35GatedDeltaV3.sourceParts else { return nil }
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
        guard !replay.contains("g_["), !replay.contains("beta_") else { return nil }
        return """
            {
              const device float* k_ = pk + hk_idx * Dk + dk0;
              const device float* v_ = pv + hv_idx * Dv + dvbase;
              const device float* a_ = pa + hv_idx;
              const device float* b_ = pb + hv_idx;
              const int a_rs = ab_rows[0];
              const int b_rs = ab_rows[1];
              const float g_nexp = -metal::precise::exp(alog[hv_idx]);
              const float g_dtb = dtb[hv_idx];

            """ + replay + "\n}\n"
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

// MARK: - Verify GDN chain: the prework (and the b|a reduce) folded into the recurrence launch

/// The verify window's GDN prework folded into its recurrence launch, with
/// the b|a split-K reduce folded in as well where the load-time trial finds
/// that fastest (`BONSAI_GDN_VERIFY_FUSED=0` keeps the separate launches).
///
/// At verify width every GDN layer ran, per round, the b|a split-K partial
/// kernel, its reduce (encoded after the qkv|z product through an unread
/// `dep` input), `Qwen35GDNPrework` (one launch: the causal conv over the
/// retained rows and the window, SiLU, the head split, the q/k norms, the
/// gates, the tape's conv input; q, k, v, g, beta written to memory) and then
/// the recurrence launch (`Qwen35GDNReplayFused`, or the store-free
/// `Qwen35GatedDeltaV3.runOutputOnly` in a round without a pending replay),
/// which read q, k, v, g and beta back. Here each recurrence threadgroup
/// (`SL` dv slabs of one value head, 128 threads each) first computes its key
/// head's q and k channels for the window's rows with the prework kernel's
/// own text (thread c owns channel c as there, so the `simd_sum` lanes and
/// the `(r0 + r1) + (r2 + r3)` tree see the same operands), its own dv rows'
/// v and its value head's gates, into threadgroup memory; the replay block,
/// the state store and the step loop follow with their text unchanged,
/// reading q, k, v, g and beta from threadgroup memory instead of device
/// memory (an FP32 value stored and reloaded is the same bits). The tape's
/// q, k, v and conv input are still written to device memory, each element
/// by exactly one threadgroup. In the `_bafold` variant the gate threads
/// also do what `Qwen35SplitKFold`'s prework does: they sum the b|a chunk
/// partials in the reduce kernel's order (from 0.0f, chunk 0 first) in place
/// of reading the reduced a and b, and write the sums out for the tape, so
/// the reduce launch leaves the round too. Every arithmetic expression is
/// cut from the stock kernels' text by checked spans (the fold by the same
/// anchors `Qwen35SplitKFold` applies to the prework source); only the
/// bookkeeping around them is new.
///
/// The kernel indexes the conv weight, the gate constants and the norm scales
/// plainly (row-major, checked once per array object on the Swift side) so
/// each replay variant binds 28 of Metal's 31 buffer slots; `kernel` counts
/// the slots as MLX assigns them and yields nil above the limit.
///
/// Four chains are on the table for a layer's verify (`Chain`): A, the
/// record's (partial, reduce, prework reading first, recurrence); B, the
/// prework summing the partials (`Qwen35SplitKFold`); C, the reduce then the
/// fused launch; D, the fused launch summing the partials. At model
/// construction `trial` times all of them as serialized dependent chains on
/// the verify's shapes and picks the fastest when it is at least
/// `trialMargin` under A whose self-test passed; that self-test compares
/// every launched chain's outputs (q, k, v, the conv input, the summed a and
/// b, the output rows and the committed state, and the fold's g, beta and
/// tail), as unsigned integers, with A on synthetic windows of 16 and 5 rows
/// (FP16 qkv) and 3 rows (FP32) at 0, 1, 7 and all kept rows. A chain with a
/// mismatch, a missing launch or an MLX error is excluded; with none left, A
/// runs. One line on stderr reports every time, every verdict and the pick.
enum Qwen35GDNVerifyFused {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_VERIFY_FUSED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `BONSAI_GDN_VERIFY_FUSED=force` takes the fused launch at the largest
    /// slab count (summing the partials where the fold is available) on a
    /// passed self-test without `trial` (the default lets the trial decide).
    static let forced: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_VERIFY_FUSED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return value == "force"
    }()

    /// A chain other than A is taken only when `trial` reads it at least this
    /// fraction faster per round than A.
    static let trialMargin: Double = 0.03
    /// Dependent rounds per timed chain and timed chains per path.
    static let trialChain = 24
    static let trialRepeats = 7

    /// The dv slabs (of `16 * DVPL` rows) one threadgroup covers: 128, 256
    /// or 512 threads. The trial picks; nothing is fixed for one chip.
    static let slabOptions: [Int] = [1, 2, 4]

    /// The widest window the kernel serves: its threadgroup arrays are sized
    /// for `maxRows` rows (one anchor plus depth 15) and a launch's `T` may be
    /// any 1...maxRows (a round clamped to the remaining tokens is shorter).
    /// Wider windows take the separate launches.
    static let maxRows = 16

    /// How a launch receives the gate inputs a and b.
    enum Gates {
        /// The reduced b|a product's column slices (the record's reduce ran).
        case reduced(a: MLXArray, b: MLXArray)
        /// The b|a split-K chunk partials `[KSP, S, NAB]` (`a` at columns
        /// `aOffset ..< aOffset + HV`, `b` at `bOffset ...`): the launch's gate
        /// threads sum them in the reduce kernel's order and write the sums.
        case partials(abp: MLXArray, aOffset: Int, bOffset: Int)

        var folded: Bool {
            if case .partials = self { return true }
            return false
        }
    }

    struct Result {
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        /// `[cs; qkv]` in FP32, the tape's conv input.
        let convInput: MLXArray
        let y: MLXArray
        /// The window's pre-verify state: the committed state the pending
        /// replay resolved to, or the input state as it was passed.
        let state: MLXArray
        /// The summed a and b `[1, S, Hv]` when the launch formed them from
        /// the chunk partials (the tape keeps them); nil for reduced gates.
        let a: MLXArray?
        let b: MLXArray?
    }

    /// The GDN chain a layer's verify takes.
    enum Chain: Hashable {
        /// A: partial, reduce (after the qkv|z product), prework, recurrence.
        case record
        /// B: partial, prework summing the partials, recurrence
        /// (`Qwen35SplitKFold` as it stands on its own).
        case fold
        /// C (`folded` false): partial, reduce, the fused launch. D (`folded`
        /// true): partial, the fused launch summing the partials.
        case fused(slabs: Int, folded: Bool)

        /// Whether the chain reads the b|a chunk partials itself (B, D): the
        /// forward then captures them and the reduce node is never evaluated.
        var usesPartials: Bool {
            switch self {
            case .record: return false
            case .fold: return true
            case .fused(_, let folded): return folded
            }
        }

        var label: String {
            switch self {
            case .record: return "A record"
            case .fold: return "B fold"
            case .fused(let slabs, let folded):
                return (folded ? "D fused+fold x" : "C fused x") + String(slabs)
            }
        }
    }

    // MARK: Kernel text

    /// The stock strided prework's text between `start` and the first `end`
    /// after it, or nil when either is not found exactly once.
    private static func span(_ text: String, _ start: String, through end: String) -> String? {
        guard text.components(separatedBy: start).count == 2,
            let head = text.range(of: start),
            let stop = text.range(of: end, range: head.upperBound ..< text.endIndex)
        else { return nil }
        return String(text[head.lowerBound ..< stop.upperBound])
    }

    private static func once(_ text: String, _ literal: String) -> Bool {
        text.components(separatedBy: literal).count == 2
    }

    /// The prework preamble: the stock strided prework's expressions (the
    /// conv taps' `fma` in the stock order, MLX's SiLU, the norms' `simd_sum`
    /// and `(r0 + r1) + (r2 + r3)` tree, the gates) around this threadgroup's
    /// rows. A threadgroup holds `SL` dv slabs of one value head (128 threads
    /// each); its `SL` simdgroup quads split the key head's rows for q and k
    /// (quad `slab` takes rows `slab * TT/SL ...`), so the q/k work a key
    /// head's twelve recurrence threadgroups repeated is repeated 12/SL
    /// times. Each column's conv window (`NK` state rows plus the quad's
    /// rows) and taps are loaded once into registers; row t then accumulates
    /// `fma(window[t + j], w[j], acc)` for j = 0..KS-1, the stock lambda's
    /// taps in its order. With `folded` the gate threads form a and b from
    /// the chunk partials `abp` by `Qwen35SplitKFold.foldBlock` (the text the
    /// fold's prework variants carry) and write them to `ao` / `bo`. Runs
    /// before the recurrence's pointers are set; ends on a threadgroup
    /// barrier.
    private static func preworkText(folded: Bool) -> String? {
        let stock = Qwen35GDNPrework.stridedSource
        guard
            let lambda = span(stock, "auto conv_silu = [&](uint col) -> float {", through: "return acc * sig;"),
            let silu = span(stock, "// MLX's silu: x * sigmoid(x), sigmoid in its stable functor form.", through: "return acc * sig;"),
            let gates = span(
                stock, "// g = exp(-exp(A_log) * softplus(a + dt_bias)), softplus as MLX's",
                through: "beta[grow] = (bv < 0.0f) ? by : 1.0f - by;")
        else { return nil }
        // The stock lambda's loads and tap, quoted as it spells them (checked
        // below to still be its text). The conv weight, the gate constants
        // and the norm scales are indexed plainly, as the stock row-contiguous
        // prework indexes them (`launch` admits only arrays with that
        // layout): the same elements enter the same arithmetic, and five
        // `<name>_strides` buffer slots are not bound (Metal binds at most 31
        // buffers per launch; see `bufferSlots`).
        let stateLoad = "cs[cb + int64_t(r + NK) * cs1 + int64_t(col) * cs2]"
        let chunkLoad = "float(qkv[qb + int64_t(r) * qs1 + int64_t(col) * qs2])"
        let stridedW = "w[int64_t(col) * w_strides[0] + int64_t(j) * w_strides[1]]"
        let tap = "acc = fma(xv, " + stridedW + ", acc);"
        let stridedDtb = "dtb[int64_t(hv) * dtb_strides[0]]"
        let stridedDecay = "decay[int64_t(hv) * decay_strides[0]]"
        let readA = "const float av = a[ab + int64_t(hv) * a_strides[2]] + " + stridedDtb + ";"
        let readB = "const float bv = b[bbase + int64_t(hv) * b_strides[2]];"
        let growLine = "const size_t grow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HV) + size_t(hv);"
        guard once(lambda, "const int r = int(t) + j - NK;"), once(lambda, "? " + stateLoad),
            once(lambda, ": " + chunkLoad), once(lambda, tap), once(lambda, "float acc = 0.0f;"),
            once(lambda, "for (int j = 0; j < KS; j++) {"),
            once(gates, "g[grow] = "), once(gates, "beta[grow] = "),
            once(gates, readA), once(gates, readB),
            once(gates, stridedDtb), once(gates, stridedDecay), once(stock, growLine),
            !gates.contains("dtb[hv]"), !gates.contains("decay[hv]"),
            silu.contains("const float sy = "), silu.contains("const float sig = ")
        else { return nil }
        var gateBlock = gates
        if folded {
            // `Qwen35SplitKFold`'s replacements on the gate reads: the ordered
            // sums of the chunk partials in place of the reduced values, then
            // the sums out for the tape (once per value head and row: the
            // threadgroups of one value head read the same partials).
            // Its `pa` / `pb` row offsets are spelled `fpa` / `fpb` here: the
            // replay variant's kernel arguments carry those names.
            var fold = Qwen35SplitKFold.foldBlock
            guard once(fold, "ao[grow] = asum;"), once(fold, "bo[grow] = bsum;"),
                once(fold, "const int64_t pa = prow + "), once(fold, "const int64_t pb = prow + "),
                once(fold, "asum += abp[pa + pi];"), once(fold, "bsum += abp[pb + pi];"),
                fold.contains("float asum = 0.0f;"), fold.contains("float bsum = 0.0f;")
            else { return nil }
            fold =
                fold
                .replacingOccurrences(of: "const int64_t pa = prow + ", with: "const int64_t fpa = prow + ")
                .replacingOccurrences(of: "const int64_t pb = prow + ", with: "const int64_t fpb = prow + ")
                .replacingOccurrences(of: "asum += abp[pa + pi];", with: "asum += abp[fpa + pi];")
                .replacingOccurrences(of: "bsum += abp[pb + pi];", with: "bsum += abp[fpb + pi];")
            guard !fold.contains(" pa "), !fold.contains(" pb "), !fold.contains("[pa "),
                !fold.contains("[pb ")
            else { return nil }
            gateBlock =
                gateBlock
                .replacingOccurrences(
                    of: readA, with: fold + "          const float av = asum + " + stridedDtb + ";")
                .replacingOccurrences(of: readB, with: "const float bv = bsum;")
                .replacingOccurrences(of: "ao[grow] = asum;", with: "if (owner_y) { ao[grow] = asum; }")
                .replacingOccurrences(of: "bo[grow] = bsum;", with: "if (owner_y) { bo[grow] = bsum; }")
            // (`dtb_strides` is still in the text here: the strided anchors
            // are named with their multiplication, as the fold names them.)
            guard !gateBlock.contains("a[ab"), !gateBlock.contains("b[bbase"),
                !gateBlock.contains("* a_strides"), !gateBlock.contains("* b_strides"),
                once(gateBlock, "asum += abp[fpa + pi];"), once(gateBlock, "bsum += abp[fpb + pi];"),
                once(gateBlock, "const float av = asum + "),
                once(gateBlock, "const float bv = bsum;")
            else { return nil }
        }
        gateBlock =
            gateBlock
            .replacingOccurrences(of: "g[grow] = ", with: "pw_g[t] = ")
            .replacingOccurrences(of: "beta[grow] = ", with: "pw_b[t] = ")
            .replacingOccurrences(of: stridedDtb, with: "dtb[hv]")
            .replacingOccurrences(of: stridedDecay, with: "decay[hv]")
        // The stock norm and store expressions, quoted here as the stock
        // kernel spells them (checked below to still be its text).
        let sumQ = "float sq = simd_sum(xq * xq);"
        let sumK = "float sk = simd_sum(xk * xk);"
        let treeQ = "sq = (red[0] + red[1]) + (red[2] + red[3]);"
        let treeK = "sk = (red[4] + red[5]) + (red[6] + red[7]);"
        let invQ = "const float invq = metal::precise::rsqrt(sq / float(DK) + 1e-6f);"
        let invK = "const float invk = metal::precise::rsqrt(sk / float(DK) + 1e-6f);"
        let stridedStoreQ = "(xq * invq) * wq[int64_t(c) * wq_strides[0]]"
        let stridedStoreK = "(xk * invk) * wk[int64_t(c) * wk_strides[0]]"
        let storeQ = "(xq * invq) * wq[c]"
        let storeK = "(xk * invk) * wk[c]"
        for literal in [sumQ, sumK, treeQ, treeK, invQ, invK, stridedStoreQ, stridedStoreK]
        where !once(stock, literal) {
            return nil
        }
        guard !gateBlock.contains("dtb_strides"), !gateBlock.contains("decay_strides"),
            !silu.contains("_strides[")
        else { return nil }
        let treeQRow = treeQ.replacingOccurrences(of: "red[", with: "red[t * 8 + ")
        let treeKRow = treeK.replacingOccurrences(of: "red[", with: "red[t * 8 + ")
        // The gate threads' row bases: the reduced arrays' (strided) or, for
        // the fold, the stock `grow` of the summed arrays and the writer flag.
        let gateHeader: String
        if folded {
            gateHeader = """
                            const int Sn = T;
                            const bool owner_y = threadgroup_position_in_grid.y == 0;
                            \(growLine)
            """
        } else {
            gateHeader = """
                            const int64_t ab = int64_t(bb) * a_strides[0] + int64_t(t) * a_strides[1];
                            const int64_t bbase = int64_t(bb) * b_strides[0] + int64_t(t) * b_strides[1];
            """
        }
        return """
            // ---- the GDN prework (Qwen35GDNPrework's expressions) for this threadgroup ----
            constexpr int DK = Dk;
            constexpr int DV = Dv;
            constexpr int HK = Hk;
            constexpr int HV = Hv;
            constexpr int CD = 2 * Hk * Dk + Hv * Dv;
            constexpr int GRP = HV / HK;
            constexpr int KEY = HK * DK;
            constexpr int VOFF = 2 * KEY;
            constexpr int NK = KS - 1;
            constexpr int NDV = DVPT * SL;
            constexpr int TPQ = TT / SL;
            constexpr int RPT = 128 / DVPT;
            constexpr int VPT = TT / RPT;
            constexpr int CPT = ((NK + TT) * NDV + 128 * SL - 1) / (128 * SL);
            threadgroup float4 pw_qk4[TT * 2 * Dk / 4];
            threadgroup float pw_v[TT * NDV];
            threadgroup float pw_g[TT];
            threadgroup float pw_b[TT];
            threadgroup float red[8 * TT];
            threadgroup float* pw_qk = (threadgroup float*)pw_qk4;
            {
              const uint tid = thread_position_in_threadgroup.x;
              const uint c = tid % 128u;
              const uint h = hk_idx;
              const uint bb = 0;
              const int64_t qb = int64_t(bb) * qkv_strides[0];
              const int64_t qs1 = qkv_strides[1];
              const int64_t qs2 = qkv_strides[2];
              const int64_t cb = int64_t(bb) * cs_strides[0];
              const int64_t cs1 = cs_strides[1];
              const int64_t cs2 = cs_strides[2];
              // Row m of the concatenated conv input [cs; qkv] in column col
              // (the stock lambda's r is m - NK): the state's row, the
              // widened chunk row, or 0 past the window (never a tap of a
              // row inside it).
              auto win_load = [&](uint col, int m) -> float {
                const int r = m - NK;
                return (r < 0)
                    ? \(stateLoad)
                    : ((r < T) ? \(chunkLoad) : 0.0f);
              };
              auto silu = [&](float acc) -> float {
                \(silu)
              };
              const uint colq = h * DK + c;
              const uint colk = KEY + h * DK + c;
              const uint dvT0 = threadgroup_position_in_grid.y * uint(NDV);
              const bool qk_owner = (threadgroup_position_in_grid.y == 0) && (hv_idx % uint(GRP) == 0);
              // q and k channel c of key head h, this quad's TPQ rows: the
              // column windows and taps once, then per row the stock taps in
              // the stock order, SiLU, and the norms' partial sums.
              const int t0 = int(slab) * TPQ;
              float winq[TPQ + NK];
              float wink[TPQ + NK];
              float wtq[KS];
              float wtk[KS];
              #pragma clang loop unroll(full)
              for (int m = 0; m < TPQ + NK; ++m) {
                winq[m] = win_load(colq, t0 + m);
                wink[m] = win_load(colk, t0 + m);
              }
              #pragma clang loop unroll(full)
              for (int j = 0; j < KS; j++) {
                wtq[j] = w[size_t(colq) * size_t(KS) + size_t(j)];
                wtk[j] = w[size_t(colk) * size_t(KS) + size_t(j)];
              }
              float xqr[TPQ];
              float xkr[TPQ];
              #pragma clang loop unroll(full)
              for (int tl = 0; tl < TPQ; ++tl) {
                const int t = t0 + tl;
                if (t < T) {
                  float accq = 0.0f;
                  float acck = 0.0f;
                  #pragma clang loop unroll(full)
                  for (int j = 0; j < KS; j++) {
                    accq = fma(winq[tl + j], wtq[j], accq);
                    acck = fma(wink[tl + j], wtk[j], acck);
                  }
                  const float xq = silu(accq);
                  const float xk = silu(acck);
                  xqr[tl] = xq;
                  xkr[tl] = xk;
                  \(sumQ)
                  \(sumK)
                  if (lane == 0) {
                    red[t * 8 + sg] = sq;
                    red[t * 8 + 4 + sg] = sk;
                  }
                } else {
                  xqr[tl] = 0.0f;
                  xkr[tl] = 0.0f;
                }
              }
              // v of this threadgroup's NDV dv rows of value head hv_idx:
              // each thread one dv row over VPT consecutive rows.
              {
                const int dvr = int(tid % uint(NDV));
                const int tv0 = int(tid / uint(NDV)) * VPT;
                const uint colv = uint(VOFF) + hv_idx * uint(DV) + dvT0 + uint(dvr);
                float winv[VPT + NK];
                float wtv[KS];
                #pragma clang loop unroll(full)
                for (int m = 0; m < VPT + NK; ++m) {
                  winv[m] = win_load(colv, tv0 + m);
                }
                #pragma clang loop unroll(full)
                for (int j = 0; j < KS; j++) {
                  wtv[j] = w[size_t(colv) * size_t(KS) + size_t(j)];
                }
                #pragma clang loop unroll(full)
                for (int i = 0; i < VPT; ++i) {
                  const int t = tv0 + i;
                  if (t < T) {
                    float acc = 0.0f;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < KS; j++) {
                      acc = fma(winv[i + j], wtv[j], acc);
                    }
                    const float xv = silu(acc);
                    pw_v[t * NDV + dvr] = xv;
                    v[(size_t(t) * size_t(HV) + size_t(hv_idx)) * size_t(DV) + size_t(dvT0) + size_t(dvr)] = xv;
                  }
                }
              }
              // Gates of value head hv_idx, one row per thread.
              if (tid < uint(TT) && int(tid) < T) {
                const uint t = tid;
                const uint hv = hv_idx;
            \(gateHeader)
                \(gateBlock)
              }
              // The tape's conv input [cs; qkv] in FP32: q/k columns by the
              // key head's owner threadgroup (a row per quad), v columns by
              // their dv rows.
              if (qk_owner) {
                #pragma clang loop unroll(full)
                for (int r = 0; r < NK + TT; ++r) {
                  if (r < NK + T && (r % SL) == int(slab)) {
                    const size_t cirow = size_t(r) * size_t(CD);
                    ci[cirow + colq] = win_load(colq, r);
                    ci[cirow + colk] = win_load(colk, r);
                  }
                }
              }
              #pragma clang loop unroll(full)
              for (int i = 0; i < CPT; ++i) {
                const int idx = int(tid) + 128 * SL * i;
                const int r = idx / NDV;
                const int dvr = idx % NDV;
                if (r < NK + T) {
                  const uint colv = uint(VOFF) + hv_idx * uint(DV) + dvT0 + uint(dvr);
                  ci[size_t(r) * size_t(CD) + colv] = win_load(colv, r);
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              // The norms' totals in the stock tree, the scaled q and k
              // channels into threadgroup memory (and the tape, once).
              #pragma clang loop unroll(full)
              for (int tl = 0; tl < TPQ; ++tl) {
                const int t = t0 + tl;
                if (t < T) {
                  const float xq = xqr[tl];
                  const float xk = xkr[tl];
                  float sq;
                  float sk;
                  \(treeQRow)
                  \(treeKRow)
                  \(invQ)
                  \(invK)
                  const float qn = \(storeQ);
                  const float kn = \(storeK);
                  pw_qk[t * (2 * DK) + int(c)] = qn;
                  pw_qk[t * (2 * DK) + DK + int(c)] = kn;
                  if (qk_owner) {
                    q[(size_t(t) * size_t(HK) + size_t(h)) * size_t(DK) + size_t(c)] = qn;
                    k[(size_t(t) * size_t(HK) + size_t(h)) * size_t(DK) + size_t(c)] = kn;
                  }
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            // ---- the recurrence, its rows read from threadgroup memory ----

            """
    }

    /// `Qwen35GatedDeltaV3.sourceParts` with the state loaded from `ps`, the
    /// simdgroup's slab (`SL` slabs per threadgroup) in its dv base, the
    /// prework preamble before the row pointers, and those pointers plus the
    /// step loop's row loads and advances moved to threadgroup memory. The
    /// step loop's arithmetic is the stock text.
    private static func partsText(folded: Bool) -> (head: String, loop: String, store: String)? {
        guard let stock = Qwen35GatedDeltaV3.sourceParts, let prework = preworkText(folded: folded)
        else { return nil }
        var head = stock.head
        let anchor = "const device float* q_ = q;"
        for (target, replacement) in [
            ("state[d][i] = state_in[(n * Dv + dvbase + d) * Dk + dk0 + i];",
             "state[d][i] = ps[(n * Dv + dvbase + d) * Dk + dk0 + i];"),
            ("const uint sg = simdgroup_index_in_threadgroup;",
             "const uint sg = simdgroup_index_in_threadgroup % (128 / 32);\n"
                 + "    const uint slab = simdgroup_index_in_threadgroup / (128 / 32);"),
            ("const uint dvbase = threadgroup_position_in_grid.y * DVPT + sg * DVPS + (lane / LPD) * DVPL;",
             "const uint dvbase = (threadgroup_position_in_grid.y * SL + slab) * DVPT + sg * DVPS + (lane / LPD) * DVPL;"),
            (anchor, prework + "const threadgroup float* q_ = pw_qk + dk0;"),
            ("const device float* k_ = k + (b_idx * T * Hk + hk_idx) * Dk + dk0;",
             "const threadgroup float* k_ = pw_qk + Dk + dk0;"),
            ("const device float* v_ = v + (b_idx * T * Hv + hv_idx) * Dv + dvbase;",
             "const threadgroup float* v_ = pw_v + (dvbase - threadgroup_position_in_grid.y * (DVPT * SL));"),
            ("const device float* g_ = g + b_idx * T * Hv + hv_idx;",
             "const threadgroup float* g_ = pw_g;"),
            ("const device float* beta_ = beta + b_idx * T * Hv + hv_idx;",
             "const threadgroup float* beta_ = pw_b;"),
            ("q_ += (b_idx * T * Hk + hk_idx) * Dk + dk0;", ""),
        ] {
            guard head.components(separatedBy: target).count == 2 else { return nil }
            head = head.replacingOccurrences(of: target, with: replacement)
        }
        guard !head.contains("state_in"), !head.contains("const device float*") else { return nil }
        var loop = stock.loop
        for (target, replacement) in [
            ("((const device float4*)k_)[j]", "((const threadgroup float4*)k_)[j]"),
            ("((const device float4*)q_)[j]", "((const threadgroup float4*)q_)[j]"),
            ("q_ += Hk * Dk;", "q_ += 2 * Dk;"),
            ("k_ += Hk * Dk; v_ += Hv * Dv; g_ += Hv; beta_ += Hv;",
             "k_ += 2 * Dk; v_ += DVPT * SL; g_ += 1; beta_ += 1;"),
        ] {
            guard loop.components(separatedBy: target).count == 2 else { return nil }
            loop = loop.replacingOccurrences(of: target, with: replacement)
        }
        guard !loop.contains("const device float4*") else { return nil }
        return (head, loop, stock.store)
    }

    private static let plainParts = partsText(folded: false)
    private static let foldedParts = partsText(folded: true)

    /// The inputs before the recurrence's: the reduced `a` and `b`, or the
    /// chunk partials `abp` in their place.
    private static func preworkInputs(folded: Bool) -> [String] {
        let gateNames: [String] = folded ? ["abp"] : ["a", "b"]
        return ["qkv", "cs", "w"] + gateNames + ["decay", "dtb", "wq", "wk", "T", "ps"]
    }

    /// Metal binds at most 31 buffers per launch.
    static let maxBufferSlots = 31

    /// The buffer slots `MLXFast.metalKernel` binds for `source`, counted as
    /// MLX's `metal_kernel.cpp` assigns them: one per input, one per
    /// `<input>_shape`, `<input>_strides` and `<input>_ndim` the source text
    /// contains (a substring search, as there), one per output.
    static func bufferSlots(source: String, inputNames: [String], outputNames: [String]) -> Int {
        var slots: Int = inputNames.count + outputNames.count
        for name in inputNames {
            for suffix in ["_shape", "_strides", "_ndim"] where source.contains(name + suffix) {
                slots += 1
            }
        }
        return slots
    }

    /// `MLXFast.metalKernel` for `source`, or nil when it would bind more
    /// buffers than Metal's argument table holds (a kernel that fails to
    /// build must leave the separate launches in place, not fail at load).
    private static func kernel(
        name: String, inputNames: [String], outputNames: [String], source: String, header: String = ""
    ) -> MLXFast.MLXFastKernel? {
        let slots = bufferSlots(source: source, inputNames: inputNames, outputNames: outputNames)
        guard slots <= maxBufferSlots else {
            FileHandle.standardError.write(
                "qwen35 GDN verify fused: \(name) would bind \(slots) buffers (limit \(maxBufferSlots)); separate prework launch kept\n"
                    .data(using: .utf8)!)
            return nil
        }
        return MLXFast.metalKernel(
            name: name, inputNames: inputNames, outputNames: outputNames, source: source,
            header: header, ensureRowContiguous: false)
    }

    /// Prework, then the store-free step loop from `ps` (no pending replay):
    /// 11 inputs, 4 stride tables (qkv, cs, a, b), 5 outputs; the fold 10
    /// inputs, 3 stride tables (qkv, cs, abp), 7 outputs.
    private static func makeOutputOnlyKernel(folded: Bool) -> MLXFast.MLXFastKernel? {
        guard let parts = folded ? foldedParts : plainParts else { return nil }
        let sums: [String] = folded ? ["ao", "bo"] : []
        return kernel(
            name: "qwen35_gdn_verify_fused_output_only" + (folded ? "_bafold" : ""),
            inputNames: preworkInputs(folded: folded),
            outputNames: ["y", "q", "k", "v", "ci"] + sums,
            source: parts.head + parts.loop + "\n")
    }

    /// Prework, the previous tape's `KP` kept rows from `ps`, the committed
    /// state's store, then the step loop (`Qwen35GDNReplayFused` fused):
    /// 18 inputs, 4 stride tables, 6 outputs, 28 of the 31 slots; the fold
    /// 17 inputs, 3 stride tables, 8 outputs, also 28.
    private static func makeReplayKernel(folded: Bool) -> MLXFast.MLXFastKernel? {
        guard let parts = folded ? foldedParts : plainParts,
            let stockReplay = Qwen35GDNReplayFused.replayBlock
        else { return nil }
        // The replay block is the stock step loop with `OUTPUT_NEEDED` spelled
        // `false`; its discarded `if constexpr` branch still names the outer
        // `q_`, here a threadgroup pointer, and Metal type-checks a discarded
        // branch outside a template (the round-1 build failed on exactly that
        // cast). Its own `k_` is a device pointer local to the block.
        let deviceQ = "((const device float4*)q_)[j]"
        guard stockReplay.components(separatedBy: deviceQ).count == 2 else { return nil }
        let replayBlock = stockReplay.replacingOccurrences(
            of: deviceQ, with: "((const threadgroup float4*)q_)[j]")
        guard !replayBlock.contains("const device float4*)q_") else { return nil }
        let sums: [String] = folded ? ["ao", "bo"] : []
        return kernel(
            name: "qwen35_gdn_verify_fused_replay" + (folded ? "_bafold" : ""),
            inputNames: preworkInputs(folded: folded) + ["pk", "pv", "pa", "pb", "alog", "ab_rows", "KP"],
            outputNames: ["y", "state_out", "q", "k", "v", "ci"] + sums,
            source: parts.head + replayBlock + parts.store + "\n" + parts.loop + "\n",
            header: Qwen35GDNReplayBatch.header)
    }

    private static let plainOutputOnlyKernel = makeOutputOnlyKernel(folded: false)
    private static let foldedOutputOnlyKernel = makeOutputOnlyKernel(folded: true)
    private static let plainReplayKernel = makeReplayKernel(folded: false)
    private static let foldedReplayKernel = makeReplayKernel(folded: true)

    // MARK: Constant-vector layouts

    private static let layoutLock = NSLock()
    /// Per array object (retained, so its identity stays valid): whether it
    /// has the row-major layout the kernel indexes plainly. The arrays are the
    /// layer's parameters and its cached derived vectors, so each is checked
    /// once; the check evaluates the array first (strides are stable only
    /// then), a no-op after load.
    nonisolated(unsafe) private static var layouts: [ObjectIdentifier: (array: MLXArray, plain: Bool)] = [:]

    /// Whether `array` (evaluated here if it is not yet) has strides
    /// `expected` on the axes of size > 1.
    @available(*, deprecated, message: "reads strides; evaluates the array first")
    private static func plainLayout(_ array: MLXArray, expected: [Int]) -> Bool {
        let id = ObjectIdentifier(array)
        if let known = layoutLock.withLock({ layouts[id] }), known.array === array {
            return known.plain
        }
        eval(array)
        let strides: [Int] = array.strides
        let shape: [Int] = array.shape
        var plain: Bool = strides.count == expected.count && shape.count == expected.count
        if plain {
            for axis in 0 ..< expected.count where shape[axis] != 1 && strides[axis] != expected[axis] {
                plain = false
            }
        }
        layoutLock.withLock {
            if layouts.count >= 4096 { layouts.removeAll() }
            layouts[id] = (array, plain)
        }
        return plain
    }

    /// The loaded parameters the kernel indexes plainly (`w[col * KS + j]`,
    /// `dtb[hv]`) in row-major layout. `decay` and the norm scales are not
    /// checked: `Qwen35GDNDerived` builds them as a row-contiguous kernel
    /// output and from a Swift array, so their layout is row-major by
    /// construction. A parameter object is checked (evaluated, as it already
    /// is after the load) the first time it is seen, during the load-time
    /// verify warm; a timed round looks the same object up.
    @available(*, deprecated, message: "reads strides; evaluates the arrays first")
    private static func plainConstants(convWeight: MLXArray, dtb: MLXArray) -> Bool {
        let KS = convWeight.dim(1)
        return plainLayout(convWeight, expected: [KS, 1, 1]) && plainLayout(dtb, expected: [1])
    }

    // MARK: Launch

    /// One row's window: the prework of `qkv` (`[1, S, CD]`, a column slice
    /// read in place) over `convState` with the gate inputs `gates`, then the
    /// recurrence from `state` (`[1, Hv, Dv, Dk]` FP32, row-major), with
    /// `replay` (the previous tape, its kept rows and the layer's `A_log`)
    /// applied first when given. Nil when the window does not fit.
    static func launch(
        qkv: MLXArray, convState: MLXArray, convWeight: MLXArray, gates: Gates,
        aDecay: MLXArray, dtBias: MLXArray, normScales: (q: MLXArray, k: MLXArray),
        keyHeads: Int, valueHeads: Int, headKDim: Int, headVDim: Int,
        state: MLXArray, replay: (tape: ArraysCache.PrefixReplayTape, keep: Int, aLog: MLXArray)?,
        slabs: Int
    ) -> Result? {
        guard enabled, Qwen35GatedDeltaV3.enabled, qkv.ndim == 3, convState.ndim == 3,
            convWeight.ndim == 3, slabOptions.contains(slabs)
        else { return nil }
        let S = qkv.dim(1)
        let CD = qkv.dim(2)
        let KS = convWeight.dim(1)
        let Hk = keyHeads
        let Hv = valueHeads
        let Dk = headKDim
        let Dv = headVDim
        let dvpl = Qwen35GatedDeltaV3.rowsPerLane
        let dvpt: Int = 16 * dvpl
        guard qkv.dim(0) == 1, S >= 1, S <= maxRows, Dk == 128, Dv % (dvpt * slabs) == 0,
            maxRows % slabs == 0, 128 % dvpt == 0, maxRows % (128 / dvpt) == 0,
            Hk > 0, Hv % Hk == 0, KS >= 2,
            CD == 2 * Hk * Dk + Hv * Dv,
            [DType.float32, .float16].contains(qkv.dtype),
            convState.shape == [1, KS - 1, CD], convState.dtype == .float32,
            convWeight.shape == [CD, KS, 1], convWeight.dtype == .float32,
            aDecay.shape == [Hv], aDecay.dtype == .float32, dtBias.shape == [Hv],
            normScales.q.shape == [Dk], normScales.k.shape == [Dk],
            normScales.q.dtype == .float32, normScales.k.dtype == .float32,
            state.shape == [1, Hv, Dv, Dk], state.dtype == .float32,
            // The constant vectors are indexed plainly: FP32 (`dt_bias` as the
            // fused replay already requires) in row-major layout.
            dtBias.dtype == .float32,
            plainConstants(convWeight: convWeight, dtb: dtBias)
        else { return nil }
        let dtb = dtBias
        var template: [(String, any KernelTemplateArg)] = [
            ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("OUTPUT_NEEDED", true),
            ("DVPL", dvpl), ("KS", KS), ("TT", maxRows), ("SL", slabs),
        ]
        let gateOperands: [MLXArray]
        let folded: Bool
        switch gates {
        case .reduced(let a, let b):
            guard a.shape == [1, S, Hv], b.shape == [1, S, Hv], a.dtype == .float32,
                b.dtype == .float32
            else { return nil }
            gateOperands = [a, b]
            folded = false
        case .partials(let abp, let aOffset, let bOffset):
            guard Qwen35SplitKFold.enabled, abp.ndim == 3, abp.dtype == .float32,
                abp.dim(1) == S, abp.dim(0) >= 1, abp.dim(0) <= Qwen35SplitKFold.maximumChunks,
                aOffset >= 0, bOffset >= 0, aOffset + Hv <= abp.dim(2), bOffset + Hv <= abp.dim(2)
            else { return nil }
            template.append(("KSP", abp.dim(0)))
            template.append(("AOFF", aOffset))
            template.append(("BOFF", bOffset))
            gateOperands = [abp]
            folded = true
        }
        let grid = (128 * slabs, Dv / (dvpt * slabs), Hv)
        let threadGroup = (128 * slabs, 1, 1)
        let sumShapes: [[Int]] = folded ? [[1, S, Hv], [1, S, Hv]] : []
        let rowShapes: [[Int]] =
            [[1, S, Hk, Dk], [1, S, Hk, Dk], [1, S, Hv, Dv], [1, KS - 1 + S, CD]] + sumShapes
        let preworkOperands: [MLXArray] =
            [qkv, convState, convWeight] + gateOperands
            + [aDecay, dtb, normScales.q, normScales.k, MLXArray(Int32(S))]
        guard let replay else {
            guard let outputOnlyKernel = folded ? foldedOutputOnlyKernel : plainOutputOnlyKernel
            else { return nil }
            // The state is read with row-major indexing (no copy when it
            // already is; the stock launch ensures the same).
            let ps = contiguous(state)
            let out = outputOnlyKernel(
                preworkOperands + [ps],
                template: template, grid: grid, threadGroup: threadGroup,
                outputShapes: [[1, S, Hv, Dv]] + rowShapes,
                outputDTypes: Array(repeating: DType.float32, count: 1 + rowShapes.count))
            return Result(
                q: out[1], k: out[2], v: out[3], convInput: out[4], y: out[0], state: state,
                a: folded ? out[5] : nil, b: folded ? out[6] : nil)
        }
        let tape = replay.tape
        let keep = replay.keep
        let aLog = replay.aLog
        guard let replayKernel = folded ? foldedReplayKernel : plainReplayKernel,
            let ps = tape.ssmPre, tape.mask == nil, ps === state
        else { return nil }
        let P = tape.rowCount
        let previous = [ps, tape.k, tape.v, tape.a, tape.b, aLog, dtb]
        // The replay's own routing: from `minRows` kept rows it is chunked.
        guard previous.allSatisfy({ $0.dtype == .float32 }),
            !(Qwen35GatedDeltaChunked.enabled && keep >= Qwen35GatedDeltaChunked.minRows
                && keep >= Qwen35GatedDeltaChunked.chunk),
            keep >= 0, keep <= P,
            tape.k.shape == [1, P, Hk, Dk], tape.v.shape == [1, P, Hv, Dv],
            tape.a.shape == [1, P, Hv], tape.b.shape == [1, P, Hv], aLog.shape == [Hv]
        else { return nil }
        // Read in place, as the fused replay reads them: evaluated with their
        // verify, so this wait is a no-op.
        eval(previous)
        guard Qwen35GDNReplayBatch.rowContiguousAfterLeading(ps),
            Qwen35GDNReplayBatch.rowContiguousAfterLeading(tape.k),
            Qwen35GDNReplayBatch.rowContiguousAfterLeading(tape.v),
            aLog.strides == [1], dtb.strides == [1],
            let aRows = Qwen35GDNReplayBatch.gateRowStride(tape.a),
            let bRows = Qwen35GDNReplayBatch.gateRowStride(tape.b)
        else { return nil }
        let out = replayKernel(
            preworkOperands + [ps, tape.k, tape.v, tape.a, tape.b, aLog]
                + [MLXArray([aRows, bRows]), MLXArray(Int32(keep))],
            template: template, grid: grid, threadGroup: threadGroup,
            outputShapes: [[1, S, Hv, Dv], ps.shape] + rowShapes,
            outputDTypes: Array(repeating: DType.float32, count: 2 + rowShapes.count))
        return Result(
            q: out[2], k: out[3], v: out[4], convInput: out[5], y: out[0], state: out[1],
            a: folded ? out[6] : nil, b: folded ? out[7] : nil)
    }

    /// The fused prework and scan of `layer`'s single-row verify whose input
    /// SSM is `input` (a pending deferred replay is computed inside the launch
    /// and resolved with the committed state), at the trial's `slabs`. Nil
    /// when the window does not fit: the caller then takes the separate
    /// launches (the fold's prework where it holds the partials, else the
    /// record's).
    static func run(
        layer: Qwen35GatedDeltaNet, input: CBv2RecurrentLayerState?,
        qkv: MLXArray, convState: MLXArray, convWeight: MLXArray, gates: Gates,
        aDecay: MLXArray, dtBias: MLXArray, normScales: (q: MLXArray, k: MLXArray), slabs: Int
    ) -> Result? {
        guard enabled, Qwen35GDNReplayFused.applies(to: layer) else { return nil }
        func launchWindow(state: MLXArray, replay: (tape: ArraysCache.PrefixReplayTape, keep: Int, aLog: MLXArray)?)
            -> Result?
        {
            launch(
                qkv: qkv, convState: convState, convWeight: convWeight, gates: gates,
                aDecay: aDecay, dtBias: dtBias, normScales: normScales,
                keyHeads: layer.numKHeads, valueHeads: layer.numVHeads,
                headKDim: layer.headKDim, headVDim: layer.headVDim, state: state, replay: replay,
                slabs: slabs)
        }
        if let deferred = input?.deferredReplay, deferred.isPending {
            guard let inputs = deferred.inputs as? Qwen35GDNReplayFused.Inputs,
                inputs.layer == ObjectIdentifier(layer), let ps = inputs.tape.ssmPre,
                let fused = launchWindow(state: ps, replay: (inputs.tape, deferred.keep, layer.aLog)),
                deferred.resolve(fused.state)
            else { return nil }
            return fused
        }
        let ssm =
            input?.ssm
            ?? MLXArray.zeros([1, layer.numVHeads, layer.headVDim, layer.headKDim], dtype: .float32)
        return launchWindow(state: ssm, replay: nil)
    }

    // MARK: Verdict

    private struct Geometry: Hashable {
        let hk: Int, dk: Int, hv: Int, dv: Int, cd: Int, ks: Int, dvpl: Int, hidden: Int
    }

    private static func geometry(_ layer: Qwen35GatedDeltaNet) -> Geometry {
        Geometry(
            hk: layer.numKHeads, dk: layer.headKDim, hv: layer.numVHeads, dv: layer.headVDim,
            cd: layer.convDim, ks: layer.convKernelSize, dvpl: Qwen35GatedDeltaV3.rowsPerLane,
            hidden: layer.hiddenSize)
    }

    private static let verdictLock = NSLock()
    /// Per geometry: the chain the trial picked and the self-test passed.
    nonisolated(unsafe) private static var verdicts: [Geometry: Chain] = [:]

    /// The chain `layer`'s verify takes. With this item off, or where the
    /// fused replay's conditions do not hold (state skip on, chunked verify
    /// off, its self-test passed) so no trial ran, B: the fold as it stands
    /// on its own (its own switch and verdict decide in the forward). After
    /// a trial, its pick; A when the trial or the self-test failed.
    static func chain(for layer: Qwen35GatedDeltaNet) -> Chain {
        guard enabled, Qwen35GDNReplayFused.applies(to: layer) else { return .fold }
        let key = geometry(layer)
        return verdictLock.withLock { verdicts[key] } ?? .fold
    }

    private enum SelfTestFailure: Error {
        case message(String)
    }

    /// Run the trial and the bitwise self-test once per geometry at model
    /// construction, after `Qwen35GDNReplayFused.prepare` (whose launch is
    /// the reference), `Qwen35GDNPrework.prepareVerify` (the record's verify
    /// prework) and `Qwen35SplitKFold.prepare` (the fold's verdict). Every
    /// chain the trial launched is self-tested; the pick is the fastest of
    /// those at least `trialMargin` under A whose self-test passed (a chain
    /// that fails is excluded, not taken), else A.
    static func prepare(layer: Qwen35GatedDeltaNet) {
        guard enabled, Qwen35GatedDeltaV3.enabled, Qwen35GDNReplayFused.applies(to: layer) else {
            return
        }
        let key = geometry(layer)
        verdictLock.lock()
        defer { verdictLock.unlock() }
        guard verdicts[key] == nil else { return }
        var pick: Chain = .record
        var timing = ""
        var launched: [(chain: Chain, time: Double)] = []
        var record: Double? = nil
        if forced {
            launched = [
                (
                    .fused(
                        slabs: slabOptions[slabOptions.count - 1],
                        folded: Qwen35SplitKFold.active(rows: maxRows)), 0
                )
            ]
        } else if let times = trial(layer: layer) {
            record = times.record
            launched = times.candidates
            timing = "; trial us/round: A record " + String(format: "%.1f", times.record)
            for candidate in times.candidates {
                timing += ", " + candidate.chain.label + String(format: " %.1f", candidate.time)
            }
            timing += String(
                format: " (%ld dependent rounds, median of %ld)", trialChain, trialRepeats)
            if !times.dropped.isEmpty {
                timing +=
                    "; not launched: "
                    + times.dropped.map { $0.chain.label + " (" + $0.reason + ")" }
                    .joined(separator: ", ")
            }
        } else {
            timing = "; trial did not launch"
        }
        // The bitwise self-test of every chain that launched, against A.
        let test = selfTest(layer: layer, chains: launched.map { $0.chain })
        let passedChains = Set(test.passed)
        var pickNote = ""
        if forced {
            if let first = launched.first, passedChains.contains(first.chain) {
                pick = first.chain
                pickNote = "; " + first.chain.label + " forced"
            }
        } else if let record {
            let fastest = launched.filter { passedChains.contains($0.chain) }
                .min { $0.time < $1.time }
            if let fastest, fastest.time <= record * (1 - trialMargin) {
                pick = fastest.chain
                pickNote =
                    "; " + fastest.chain.label
                    + String(format: " picked (%.1f%% under A)", (1 - fastest.time / record) * 100)
            } else if let fastest {
                pickNote =
                    "; none " + String(format: "%.0f%%", trialMargin * 100) + " under A ("
                    + fastest.chain.label
                    + String(format: " %.1f%% under)", (1 - fastest.time / record) * 100)
            } else {
                pickNote = launched.isEmpty ? "; no other chain launched" : "; no chain passed"
            }
        }
        verdicts[key] = pick
        // The self-test's synthetic constants leave the layout cache.
        layoutLock.withLock { layouts.removeAll() }
        Memory.clearCache()
        let verdict: String
        if test.failed.isEmpty {
            verdict = test.passed.isEmpty ? "skipped" : "passed"
        } else {
            verdict =
                (test.passed.isEmpty ? "FAILED" : "passed for " + test.passed.map { $0.label }.joined(separator: ", ") + "; FAILED")
                + " for " + test.failed.map { $0.chain.label + " (" + $0.reason + ")" }.joined(separator: ", ")
        }
        let outcome: String
        switch pick {
        case .record:
            outcome = "; the record's chain kept (reduce, prework and recurrence launches)"
        case .fold:
            outcome = "; the prework launch sums the b|a chunk partials (Qwen35SplitKFold)"
        case .fused(let slabs, let folded):
            outcome =
                folded
                ? "; the verify's GDN prework and the b|a reduce run inside the recurrence launch (\(slabs) slabs per threadgroup)"
                : "; the verify's GDN prework runs inside the recurrence launch (\(slabs) slabs per threadgroup; the reduce launch kept)"
        }
        FileHandle.standardError.write(
            ("qwen35 GDN verify fused: self-test " + verdict + " (" + test.detail + ")" + timing
                + pickNote + outcome + "\n").data(using: .utf8)!)
    }

    private struct TrialTimes {
        /// Microseconds per round of chain A.
        let record: Double
        /// The other chains that launched, with their microseconds per round.
        let candidates: [(chain: Chain, time: Double)]
        /// The chains that did not launch, with why (a declined launch, or
        /// the MLX error's first line: a variant failing to compile).
        let dropped: [(chain: Chain, reason: String)]
    }

    /// Wall time per round of `trialChain` dependent rounds at the verify's
    /// shape (a 16-row FP16 window, the b|a product of the layer's hidden
    /// width in 128-wide split-K chunks, a previous tape with 10 kept rows):
    /// chain A (the partial kernel, the reduce after the qkv|z product, the
    /// prework as `Qwen35GDNPrework.run` picks it, `Qwen35GDNReplayFused
    /// .launch`) against B (the fold's prework), C and D at each slab count
    /// of `slabOptions`. Round i's normed input carries a zero-weighted term
    /// of round i-1's output rows, so each round waits for the one before it
    /// as the layers of a verify wait for each other (the two tiny launches
    /// that form it are the same on every path); every input is evaluated
    /// first, so nothing waits on the host inside a chain. Each path is timed
    /// `trialRepeats` times, interleaved, after one untimed pass; the
    /// medians, in microseconds per round. Nil when chain A does not launch;
    /// a chain that does not launch (or raises an MLX error, e.g. a variant
    /// failing to compile) is left out.
    private static func trial(layer: Qwen35GatedDeltaNet) -> TrialTimes? {
        let Hk = layer.numKHeads
        let Dk = layer.headKDim
        let Hv = layer.numVHeads
        let Dv = layer.headVDim
        let CD = layer.convDim
        let KS = layer.convKernelSize
        let NK = KS - 1
        let hidden = layer.hiddenSize
        let S = maxRows
        let keep = min(S, 10)
        let derived = Qwen35GDNDerived()
        let keys = MLXRandom.split(key: MLXRandom.key(0x5646_5452), into: 14)
        let qkvz = (MLXRandom.normal([1, S, CD + Hv * Dv], key: keys[0]) * 0.5).asType(.float16)
        let qkv = qkvz[0..., 0..., ..<CD]
        let convState = MLXRandom.normal([1, NK, CD], key: keys[1])
        let convWeight = MLXRandom.normal([CD, KS, 1], key: keys[2]) * 0.5
        // The normed FP32 input and the b|a weight at the layer's scale.
        let x0 = MLXRandom.normal([1, S, hidden], key: keys[3])
        let w = MLXRandom.normal([2 * Hv, hidden], key: keys[12]) * Float(0.02)
        let aLog = log(MLXRandom.uniform(Float(1) ..< Float(16), [Hv], key: keys[4]))
        let dtBias = MLXRandom.normal([Hv], key: keys[5])
        let prevPair = MLXRandom.normal([1, S, 2 * Hv], key: keys[6])
        let prevA = prevPair[0..., 0..., Hv...]
        let prevB = prevPair[0..., 0..., ..<Hv]
        let tapeQ = MLXRandom.normal([1, S, Hk, Dk], key: keys[7]) * 0.09
        let tapeK = MLXRandom.normal([1, S, Hk, Dk], key: keys[8]) * 0.09
        let tapeV = MLXRandom.normal([1, S, Hv, Dv], key: keys[9])
        let prevConv = MLXRandom.normal([1, NK + S, CD], key: keys[10])
        let state = MLXRandom.normal([1, Hv, Dv, Dk], key: keys[11]) * 0.05
        let aDecay = derived.decay(aLog)
        let scales = derived.normScales(headKDim: Dk, dtype: .float32)
        eval(
            qkvz, qkv, convState, convWeight, x0, w, aLog, dtBias, prevA, prevB, tapeQ, tapeK,
            tapeV, prevConv, state, aDecay, scales.q, scales.k)
        let tape = ArraysCache.PrefixReplayTape(
            convInput: prevConv, q: tapeQ, k: tapeK, v: tapeV, a: prevA, b: prevB,
            ssmPre: state, mask: nil, rowCount: S, convStateRows: NK)
        // The recurrence from the prework's rows (chains A and B).
        func recurrence(_ pre: Qwen35GDNPrework.Outputs) -> MLXArray? {
            Qwen35GDNReplayFused.launch(
                tape: tape, keep: keep, aLog: aLog, dtBias: dtBias,
                q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta)?.y
        }
        // One round of a chain from this round's normed input; its output rows.
        func record(_ x: MLXArray) -> MLXArray? {
            guard let p = Qwen35SmallNMatmul.partials(x, w) else { return nil }
            let ab = Qwen35SmallNMatmul.reduce(p, after: qkv)
            guard
                let pre = Qwen35GDNPrework.run(
                    qkv: qkv, convState: convState, convWeight: convWeight,
                    a: ab[0..., 0..., Hv...], b: ab[0..., 0..., ..<Hv],
                    aDecay: aDecay, dtBias: dtBias, normScales: scales,
                    keyHeads: Hk, valueHeads: Hv, headKDim: Dk, headVDim: Dv,
                    writeConvInput: true, stridedReads: Qwen35GDNPrework.verifyStridedReads)
            else { return nil }
            return recurrence(pre)
        }
        func fold(_ x: MLXArray) -> MLXArray? {
            guard let p = Qwen35SmallNMatmul.partials(x, w),
                let pre = Qwen35GDNPrework.runFolded(
                    qkv: qkv, convState: convState, convWeight: convWeight,
                    abPartials: p.part, aOffset: Hv, bOffset: 0,
                    aDecay: aDecay, dtBias: dtBias, normScales: scales,
                    keyHeads: Hk, valueHeads: Hv, headKDim: Dk, headVDim: Dv,
                    writeConvInput: true, stridedReads: Qwen35GDNPrework.verifyStridedReads)
            else { return nil }
            return recurrence(pre)
        }
        func fused(_ slabs: Int, folded: Bool) -> (MLXArray) -> MLXArray? {
            { x in
                guard let p = Qwen35SmallNMatmul.partials(x, w) else { return nil }
                let gates: Gates
                if folded {
                    gates = .partials(abp: p.part, aOffset: Hv, bOffset: 0)
                } else {
                    let ab = Qwen35SmallNMatmul.reduce(p, after: qkv)
                    gates = .reduced(a: ab[0..., 0..., Hv...], b: ab[0..., 0..., ..<Hv])
                }
                return launch(
                    qkv: qkv, convState: convState, convWeight: convWeight, gates: gates,
                    aDecay: aDecay, dtBias: dtBias, normScales: scales,
                    keyHeads: Hk, valueHeads: Hv, headKDim: Dk, headVDim: Dv,
                    state: state, replay: (tape, keep, aLog), slabs: slabs)?.y
            }
        }
        // `trialChain` rounds, each depending on the previous one's output rows.
        func chain(_ round: (MLXArray) -> MLXArray?) -> Double? {
            let start = DispatchTime.now().uptimeNanoseconds
            var x = x0
            var last: MLXArray?
            for _ in 0 ..< trialChain {
                guard let y = round(x) else { return nil }
                x = x0 + y[0, 0, 0, ..<1] * 0
                last = y
            }
            guard let last else { return nil }
            eval(last, x)
            return Double(DispatchTime.now().uptimeNanoseconds - start) / Double(trialChain) / 1000
        }
        var rounds: [(chain: Chain, round: (MLXArray) -> MLXArray?)] = [(.fold, fold)]
        for folded in [false, true] {
            for slabs in slabOptions {
                rounds.append((.fused(slabs: slabs, folded: folded), fused(slabs, folded: folded)))
            }
        }
        // The untimed pass compiles each variant; a chain that raises an MLX
        // error (or does not launch) is dropped, the others are timed.
        var candidates: [(chain: Chain, round: (MLXArray) -> MLXArray?)] = []
        var dropped: [(chain: Chain, reason: String)] = []
        for candidate in rounds {
            var reason = "declined"
            let launched: Bool
            do {
                launched = try withError { error in
                    guard chain(candidate.round) != nil else { return false }
                    try error.check()
                    return true
                }
            } catch {
                launched = false
                let text = "\(error)".replacingOccurrences(of: "\n", with: " ")
                reason = String(text.prefix(240))
            }
            if launched {
                candidates.append(candidate)
            } else {
                dropped.append((candidate.chain, reason))
            }
        }
        guard chain(record) != nil else { return nil }
        var recordTimes: [Double] = []
        var candidateTimes: [[Double]] = Array(repeating: [], count: candidates.count)
        do {
            try withError { error in
                for _ in 0 ..< trialRepeats {
                    guard let s = chain(record) else { return }
                    recordTimes.append(s)
                    for (index, candidate) in candidates.enumerated() {
                        guard let f = chain(candidate.round) else { return }
                        candidateTimes[index].append(f)
                    }
                }
                try error.check()
            }
        } catch {
            return nil
        }
        guard recordTimes.count == trialRepeats else { return nil }
        let recordMedian = recordTimes.sorted()[trialRepeats / 2]
        var timed: [(chain: Chain, time: Double)] = []
        for (index, candidate) in candidates.enumerated()
        where candidateTimes[index].count == trialRepeats {
            timed.append((candidate.chain, candidateTimes[index].sorted()[trialRepeats / 2]))
        }
        return TrialTimes(record: recordMedian, candidates: timed, dropped: dropped)
    }

    private struct SelfTestResult {
        let passed: [Chain]
        let failed: [(chain: Chain, reason: String)]
        let detail: String
    }

    /// Synthetic windows shaped as the capture verify stages them (qkv a
    /// column slice of a qkv|z product, the gate inputs the b|a split-K's
    /// chunk partials of a normed input with a wide magnitude spread, one row
    /// of them carrying saturating and infinite values, a previous tape with
    /// such gate inputs too): chain A's q, k, v, g, beta, tail, conv input,
    /// reduced a and b, output rows and committed state (the strided prework
    /// launch as `run` picks it, the reduce kernel, `Qwen35GDNReplayFused
    /// .launch` at 0, 1, 7 and all kept rows, `Qwen35GatedDeltaV3
    /// .runOutputOnly` from the window's own pre-verify state) are compared,
    /// as unsigned integers, with each of `chains`: a fused launch's q, k, v,
    /// conv input, its summed a and b when it folds, its output rows and
    /// committed state; the fold's prework's q, k, v, g, beta, tail, conv
    /// input, a and b and the recurrences from them, wherever its verdict
    /// admits the window. A chain fails on any mismatch, a declined launch,
    /// or a comparison count other than its loops' structure gives.
    private static func selfTest(layer: Qwen35GatedDeltaNet, chains: [Chain]) -> SelfTestResult {
        struct Tally {
            var comparisons: Int = 0
            var values: Int = 0
            var mismatches: Int = 0
            var expected: Int = 0
            var admitted: Int = 0
            var failure: String? = nil
            var differ: [MLXArray] = []
        }
        let G: Int = 3
        let Hk = layer.numKHeads
        let Dk = layer.headKDim
        let Hv = layer.numVHeads
        let Dv = layer.headVDim
        let CD = layer.convDim
        let KS = layer.convKernelSize
        let NK = KS - 1
        let hidden = layer.hiddenSize
        // The windows: the full 16-row window and a clamped last round of 5
        // rows in FP16 (the verify's qkv dtype), and the narrowest FP32 window
        // the layer admits (3 rows).
        let windows: [(rows: Int, dtype: DType)] = [(maxRows, .float16), (5, .float16), (3, .float32)]
        func keepsFor(_ S: Int) -> [Int] { Array(Set([0, 1, min(7, S - 1), S])).sorted() }
        var tallies: [Chain: Tally] = [:]
        for chain in chains where chain != .record {
            var tally = Tally()
            // The comparisons the loops below make, from their structure
            // alone: per window and generation, for a fused chain (P = 4
            // arrays per launch, 6 when it folds) P + 2 per kept-row count
            // and P + 1 for the no-replay launch; the fold's are counted where
            // its verdict admits the window (9 arrays, 2 per kept-row count
            // and 1 for the no-replay scan).
            if case .fused(_, let folded) = chain {
                let P: Int = folded ? 6 : 4
                tally.expected = windows.reduce(into: 0) { total, window in
                    total += G * (keepsFor(window.rows).count * (P + 2) + (P + 1))
                }
            }
            tallies[chain] = tally
        }
        let order: [Chain] = chains.filter { $0 != .record }
        guard Dk == 128, Dv == 128, Hv % Hk == 0, CD == 2 * Hk * Dk + Hv * Dv, KS >= 2 else {
            return SelfTestResult(
                passed: [], failed: order.map { ($0, "geometry outside the verify prework") },
                detail: "geometry outside the verify prework")
        }
        let derived = Qwen35GDNDerived()
        let specials: [Float] = [60, -60, 25, -25, .infinity, -.infinity, 1e-8, -1e-8]
        let marks = MLXArray((0 ..< (2 * Hv)).map { specials[$0 % specials.count] })
            .reshaped([1, 1, 2 * Hv])
        var tested: [String] = []
        func message(_ error: Error) -> String {
            if let failure = error as? SelfTestFailure, case .message(let text) = failure {
                return text
            }
            return String("\(error)".replacingOccurrences(of: "\n", with: " ").prefix(240))
        }
        do {
            try withError { error in
                func compare(_ chain: Chain, _ a: MLXArray?, _ b: MLXArray?, _ what: String) throws {
                    guard let a, let b, a.shape == b.shape, a.dtype == .float32,
                        b.dtype == .float32
                    else { throw SelfTestFailure.message("\(what): shape or dtype mismatch") }
                    tallies[chain]!.differ.append(
                        (a.view(dtype: .uint32) .!= b.view(dtype: .uint32)).asType(.int32).sum())
                    tallies[chain]!.values += a.size
                    tallies[chain]!.comparisons += 1
                }
                for (round, window) in windows.enumerated() {
                    let S = window.rows
                    let keeps: [Int] = keepsFor(S)
                    let keys = MLXRandom.split(
                        key: MLXRandom.key(UInt64(0x5646_5553) &+ UInt64(round)), into: 20 * G)
                    for j in 0 ..< G {
                        func key(_ i: Int) -> MLXArray { keys[20 * j + i] }
                        func gatePair(_ i: Int) -> MLXArray {
                            let row = MLXArray((0 ..< S).map { $0 == (j + i) % 3 ? Float(1) : 0 })
                            return MLX.where(
                                row.reshaped([1, S, 1]) .> 0, marks,
                                MLXRandom.normal([1, S, 2 * Hv], key: key(i)) * 4)
                        }
                        func rows(_ i: Int, _ h: Int, _ d: Int, _ s: Float) -> MLXArray {
                            MLXRandom.normal([1, S, h, d], key: key(i)) * s
                        }
                        // The previous round's tape.
                        let spread = exp(MLXRandom.normal([1, Hv, Dv, Dk], key: key(0)))
                        let prevPre = MLXRandom.normal([1, Hv, Dv, Dk], key: key(1)) * spread * 0.05
                        let prevPair = gatePair(2)
                        let prevV = rows(4, Hv, Dv, 1) * exp(rows(5, Hv, Dv, 1))
                        let prevConv = MLXRandom.normal([1, NK + S, CD], key: key(8))
                        let tape = ArraysCache.PrefixReplayTape(
                            convInput: prevConv, q: rows(13, Hk, Dk, 0.09), k: rows(14, Hk, Dk, 0.09),
                            v: prevV, a: prevPair[0..., 0..., Hv...], b: prevPair[0..., 0..., ..<Hv],
                            ssmPre: prevPre, mask: nil, rowCount: S, convStateRows: NK)
                        let aLog = log(MLXRandom.uniform(Float(1) ..< Float(16), [Hv], key: key(9)))
                        let dtBias = MLXRandom.normal([Hv], key: key(10))
                        // This round's window, as the layer stages it.
                        let qkvz = (MLXRandom.normal([1, S, CD + Hv * Dv], key: key(3))
                            * exp(MLXRandom.normal([1, S, CD + Hv * Dv], key: key(6)) * 0.5))
                            .asType(window.dtype)
                        let qkv = qkvz[0..., 0..., ..<CD]
                        let convState = MLXRandom.normal([1, NK, CD], key: key(7))
                        let convWeight = MLXRandom.normal([CD, KS, 1], key: key(11)) * 0.5
                        // The b|a product's chunk partials of a normed input with
                        // a wide magnitude spread (the sums see real rounding),
                        // one row replaced by the saturating and infinite values
                        // in chunk 0 (zeros in the others: the sums are those
                        // values), and their reduce: the record's a and b.
                        let x = MLXRandom.normal([1, S, hidden], key: key(16))
                            * exp(MLXRandom.normal([1, S, hidden], key: key(17)))
                        let w = MLXRandom.normal([2 * Hv, hidden], key: key(18)) * Float(0.02)
                            * exp(MLXRandom.normal([2 * Hv, hidden], key: key(19)) * Float(0.5))
                        guard let split = Qwen35SmallNMatmul.partials(x, w) else {
                            throw SelfTestFailure.message("no b|a split-K at \(S) rows")
                        }
                        let markRow = MLXArray((0 ..< S).map { $0 == (j + 12) % 3 ? Float(1) : 0 })
                            .reshaped([1, S, 1]) .> 0
                        let chunk0 = (MLXArray(0 ..< split.chunks) .== 0).reshaped([split.chunks, 1, 1])
                        let abp = MLX.where(
                            markRow, MLX.where(chunk0, marks, MLXArray(Float(0))), split.part)
                        let partials = Qwen35SmallNMatmul.Partials(
                            part: abp, rows: split.rows, n: split.n, chunks: split.chunks,
                            leading: split.leading, dims: split.dims)
                        let ab = Qwen35SmallNMatmul.reduce(partials)
                        let b = ab[0..., 0..., ..<Hv]
                        let a = ab[0..., 0..., Hv...]
                        let ownPre = MLXRandom.normal([1, Hv, Dv, Dk], key: key(15)) * spread * 0.05
                        eval(
                            prevPre, prevConv, tape.q, tape.k, prevV, tape.a, tape.b, aLog, dtBias,
                            qkvz, convState, convWeight, abp, ab, ownPre)
                        let aDecay = derived.decay(aLog)
                        let scales = derived.normScales(headKDim: Dk, dtype: .float32)
                        // Chain A: the references.
                        guard
                            let pre = Qwen35GDNPrework.run(
                                qkv: qkv, convState: convState, convWeight: convWeight, a: a, b: b,
                                aDecay: aDecay, dtBias: dtBias, normScales: scales,
                                keyHeads: Hk, valueHeads: Hv, headKDim: Dk, headVDim: Dv,
                                writeConvInput: true,
                                stridedReads: Qwen35GDNPrework.verifyStridedReads),
                            let convInput = pre.convInput
                        else { throw SelfTestFailure.message("no verify prework at \(S) rows") }
                        var references: [Int: (y: MLXArray, state: MLXArray)] = [:]
                        for keep in keeps {
                            guard
                                let reference = Qwen35GDNReplayFused.launch(
                                    tape: tape, keep: keep, aLog: aLog, dtBias: dtBias,
                                    q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta)
                            else { throw SelfTestFailure.message("no fused replay at \(keep) kept rows") }
                            references[keep] = reference
                        }
                        guard
                            let yOnly = Qwen35GatedDeltaV3.runOutputOnly(
                                q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta, state: ownPre)
                        else { throw SelfTestFailure.message("no store-free scan at \(S) rows") }
                        // Chain B where the fold's verdict admits this window.
                        func checkFold(_ chain: Chain) throws {
                            guard
                                let folded = Qwen35GDNPrework.runFolded(
                                    qkv: qkv, convState: convState, convWeight: convWeight,
                                    abPartials: abp, aOffset: Hv, bOffset: 0,
                                    aDecay: aDecay, dtBias: dtBias, normScales: scales,
                                    keyHeads: Hk, valueHeads: Hv, headKDim: Dk, headVDim: Dv,
                                    writeConvInput: true,
                                    stridedReads: Qwen35GDNPrework.verifyStridedReads)
                            else { return }
                            tallies[chain]!.admitted += 1
                            tallies[chain]!.expected += 9 + keeps.count * 2 + 1
                            try compare(chain, folded.q, pre.q, "fold q")
                            try compare(chain, folded.k, pre.k, "fold k")
                            try compare(chain, folded.v, pre.v, "fold v")
                            try compare(chain, folded.g, pre.g, "fold g")
                            try compare(chain, folded.beta, pre.beta, "fold beta")
                            try compare(chain, folded.tail, pre.tail, "fold tail")
                            try compare(chain, folded.convInput, convInput, "fold conv input")
                            try compare(chain, folded.a, a, "fold a")
                            try compare(chain, folded.b, b, "fold b")
                            for keep in keeps {
                                guard
                                    let out = Qwen35GDNReplayFused.launch(
                                        tape: tape, keep: keep, aLog: aLog, dtBias: dtBias,
                                        q: folded.q, k: folded.k, v: folded.v, g: folded.g,
                                        beta: folded.beta), let reference = references[keep]
                                else {
                                    throw SelfTestFailure.message(
                                        "no fused replay after the fold at \(keep) kept rows")
                                }
                                try compare(chain, out.y, reference.y, "fold output rows")
                                try compare(chain, out.state, reference.state, "fold committed state")
                            }
                            guard
                                let y = Qwen35GatedDeltaV3.runOutputOnly(
                                    q: folded.q, k: folded.k, v: folded.v, g: folded.g,
                                    beta: folded.beta, state: ownPre)
                            else {
                                throw SelfTestFailure.message("no store-free scan after the fold at \(S) rows")
                            }
                            try compare(chain, y, yOnly, "fold output rows (no replay)")
                        }
                        // Chains C and D: the fused launch at its slab count.
                        func checkFused(_ chain: Chain, slabs: Int, folded: Bool) throws {
                            let gates: Gates =
                                folded
                                ? .partials(abp: abp, aOffset: Hv, bOffset: 0) : .reduced(a: a, b: b)
                            func launchFused(_ replay: (tape: ArraysCache.PrefixReplayTape, keep: Int, aLog: MLXArray)?, _ state: MLXArray)
                                throws -> Result
                            {
                                guard
                                    let fused = launch(
                                        qkv: qkv, convState: convState, convWeight: convWeight,
                                        gates: gates, aDecay: aDecay, dtBias: dtBias, normScales: scales,
                                        keyHeads: Hk, valueHeads: Hv, headKDim: Dk, headVDim: Dv,
                                        state: state, replay: replay, slabs: slabs)
                                else { throw SelfTestFailure.message("no fused launch at \(S) rows") }
                                try compare(chain, fused.q, pre.q, "q")
                                try compare(chain, fused.k, pre.k, "k")
                                try compare(chain, fused.v, pre.v, "v")
                                try compare(chain, fused.convInput, convInput, "conv input")
                                if folded {
                                    try compare(chain, fused.a, a, "summed a")
                                    try compare(chain, fused.b, b, "summed b")
                                } else if fused.a != nil || fused.b != nil {
                                    throw SelfTestFailure.message("reduced gates returned sums")
                                }
                                return fused
                            }
                            for keep in keeps {
                                guard let reference = references[keep] else {
                                    throw SelfTestFailure.message("no reference at \(keep) kept rows")
                                }
                                let fused = try launchFused((tape, keep, aLog), prevPre)
                                try compare(chain, fused.y, reference.y, "output rows")
                                try compare(chain, fused.state, reference.state, "committed state")
                            }
                            let fused = try launchFused(nil, ownPre)
                            try compare(chain, fused.y, yOnly, "output rows (no replay)")
                            guard fused.state === ownPre else {
                                throw SelfTestFailure.message("the input state is not passed through")
                            }
                        }
                        for chain in order where tallies[chain]!.failure == nil {
                            do {
                                switch chain {
                                case .record:
                                    break
                                case .fold:
                                    try checkFold(chain)
                                case .fused(let slabs, let folded):
                                    try checkFused(chain, slabs: slabs, folded: folded)
                                }
                            } catch let failure as SelfTestFailure {
                                tallies[chain]!.failure = message(failure)
                            }
                        }
                    }
                    for chain in order where !tallies[chain]!.differ.isEmpty {
                        let count = stacked(tallies[chain]!.differ).sum()
                        eval(count)
                        try error.check()
                        tallies[chain]!.mismatches += Int(count.item(Int32.self))
                        tallies[chain]!.differ.removeAll()
                    }
                    tested.append("\(S) (\(window.dtype))")
                }
            }
        } catch {
            // An MLX error is not one chain's: every chain still standing fails.
            let text = message(error)
            for chain in order where tallies[chain]!.failure == nil {
                tallies[chain]!.failure = text
            }
        }
        var passed: [Chain] = []
        var failed: [(chain: Chain, reason: String)] = []
        var comparisons: Int = 0
        var values: Int = 0
        var mismatches: Int = 0
        for chain in order {
            let tally = tallies[chain]!
            comparisons += tally.comparisons
            values += tally.values
            mismatches += tally.mismatches
            if let failure = tally.failure {
                failed.append((chain, failure))
            } else if tally.mismatches != 0 {
                failed.append((chain, "\(tally.mismatches) mismatches"))
            } else if tally.comparisons != tally.expected || tally.comparisons == 0 {
                failed.append((chain, "\(tally.comparisons) comparisons, \(tally.expected) expected"))
            } else if chain == .fold, tally.admitted < windows.count * G {
                // The fold's verdict admits the verify's FP16 windows at least
                // (its own self-test covers each dtype); fewer cases are noted.
                passed.append(chain)
            } else {
                passed.append(chain)
            }
        }
        var detail: String
        if passed.isEmpty {
            detail = order.isEmpty ? "no chain launched" : "no chain passed"
        } else {
            detail = passed.map { $0.label }.joined(separator: ", ") + " against A"
        }
        if let foldTally = tallies[.fold], order.contains(.fold) {
            detail += "; B admitted in \(foldTally.admitted) of \(windows.count * G) cases"
        }
        detail +=
            ", \(G) windows of \(tested.joined(separator: ", ")) rows, \(comparisons) comparisons, "
            + "\(values) values, \(mismatches) mismatches"
        return SelfTestResult(passed: passed, failed: failed, detail: detail)
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
