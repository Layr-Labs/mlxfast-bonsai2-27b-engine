//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4
    var mropeSection: [Int] = [11, 11, 10]

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    // MTP — number of Multi-Token Prediction head layers.
    // Port of omlx commit 696d90a: patches/mlx_lm_mtp/qwen35_model.py
    // `_patch_text_model_args` attaches this from config.json at runtime.
    var mtpNumHiddenLayers: Int = 0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.mtpNumHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
            self.mropeSection = ropeParameters["mrope_section"]?.asInts() ?? [11, 11, 10]
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
            self.mropeSection = self.ropeScaling?["mrope_section"]?.asInts() ?? [11, 11, 10]
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }

    /// Compact CBv2 attention-storage layout. Recurrent layers are absent by
    /// design; `modelLayerIndex` maps each of the 10 full-attention KV rows
    /// back to its original transformer layer.
    public var cbv2LayerKinds: [CBv2LayerKind] {
        precondition(fullAttentionInterval > 0, "full_attention_interval must be positive")
        let dimension = headDim ?? (hiddenSize / attentionHeads)
        return (0 ..< hiddenLayers).compactMap { modelLayerIndex in
            guard (modelLayerIndex + 1) % fullAttentionInterval == 0 else { return nil }
            return CBv2LayerKind(
                attention: .full,
                headDim: dimension,
                kvHeads: kvHeads,
                queryHeads: attentionHeads,
                modelLayerIndex: modelLayerIndex)
        }
    }

    public func cbv2RecurrentStateSpec(
        activationDType: DType = .bfloat16
    ) -> CBv2RecurrentStateSpec {
        precondition(fullAttentionInterval > 0, "full_attention_interval must be positive")
        let keyDim = linearNumKeyHeads * linearKeyHeadDim
        let valueDim = linearNumValueHeads * linearValueHeadDim
        let convDim = 2 * keyDim + valueDim
        let layers: [CBv2RecurrentLayerStateSpec] = (0 ..< hiddenLayers).compactMap {
            modelLayerIndex -> CBv2RecurrentLayerStateSpec? in
            guard (modelLayerIndex + 1) % fullAttentionInterval != 0 else { return nil }
            return CBv2RecurrentLayerStateSpec(
                modelLayerIndex: modelLayerIndex,
                convShape: [1, max(0, linearConvKernelDim - 1), convDim],
                convDType: activationDType,
                ssmShape: [1, linearNumValueHeads, linearValueHeadDim, linearKeyHeadDim],
                ssmDType: .float32)
        }
        return CBv2RecurrentStateSpec(layers: layers)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities {
        var capabilities = CBv2ModelCapabilities.initialRecurrentTarget
        capabilities.supportsMTP = true
        capabilities.supportsPagedKV = true
        capabilities.requiresNativePagedKV = true
        capabilities.supportsCompactRecurrentMTPReplay = true
        // Rectangular [B, L] prompt cohorts: one recurrent state row per
        // batch row; each packed row attends its own KV (see
        // `cbv2SupportsPackedPrefill` on the prefill conformance).
        capabilities.supportsPackedPrefill = true
        // Dense and MoE layers share the same attention/recurrent state.
        // Routed and shared experts are token-local and add no checkpoint state.
        capabilities.supportsRecurrentCheckpointReuse = true
        return capabilities
    }
}

// MARK: - GatedDeltaNet

/// Elementwise chains of the Bonsai 2 forward that MLX `compile` fuses into
/// one kernel each. Every function here is pure elementwise arithmetic in the
/// same order as the ops it replaces; fusion changes the dispatch count, not
/// the operands or their precision. Shapeless, so one trace serves every
/// window width.
enum Qwen35FusedElementwise {
    /// `silu(gate) * up` in FP32, times the down projection's Hadamard signs.
    static let swigluSigned: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: true) { gate, up, signs in
            (silu(gate.asType(.float32)) * up.asType(.float32)) * signs
        }

    /// `[g, beta]` of the gated delta rule: `exp(-exp(A_log) * softplus(a + dt_bias))`
    /// and `sigmoid(b)`, both FP32, exactly as `gatedDeltaUpdate` forms them.
    static let gatedDeltaGates: @Sendable ([MLXArray]) -> [MLXArray] =
        compile(shapeless: true) { inputs in
            let a = inputs[0]
            let b = inputs[1]
            let aLog = inputs[2]
            let dtBias = inputs[3]
            let g = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
            let beta = sigmoid(b).asType(.float32)
            return [g, beta]
        }

    /// The gated norm's tail: `silu(gate) * normed`, in FP32.
    static let gatedNormTail: @Sendable (MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: true) { normed, gate in
            silu(gate.asType(.float32)) * normed.asType(.float32)
        }

    /// On unless explicitly disabled (`DARKBLOOM_BONSAI_FOLD_SIGNS=0`, fkiene
    /// `74bd8112`): a packed projection whose input comes out of a norm or an
    /// elementwise op receives that input with its Hadamard signs already
    /// applied, and its rotation skips the multiply. On this tree the fused
    /// boundary, gate and SwiGLU rotations apply the signs in-kernel first,
    /// so the fold is only a fallback.
    static let foldsHadamardSigns: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_FOLD_SIGNS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `gatedNormTail` times the output projection's Hadamard signs.
    static let gatedNormTailSigned: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: true) { normed, gate, signs in
            (silu(gate.asType(.float32)) * normed.asType(.float32)) * signs
        }

    /// The attention output gate `x * sigmoid(gate)` times the output
    /// projection's Hadamard signs, returned in the gate product's dtype.
    static let sigmoidGateSigned: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: true) { x, gate, signs in
            ((x * sigmoid(gate)) * signs).asType(x.dtype)
        }
}

/// Input-independent constants a GDN layer derives from its geometry, held
/// outside the parameter tree (a plain class, so Module reflection sees
/// `.other`).
private final class Qwen35GDNDerived {
    private let lock = NSLock()
    private var qScale: MLXArray?
    private var kScale: MLXArray?

    /// Per-dimension weights for the q and k norms that carry the head-scale
    /// factors: `rmsNorm(x, weight: w)` computes `w * (x * inv)` and a
    /// separate scalar multiply computes `(x * inv) * s`; with `w` filled with
    /// `s` the two are the same product, so the multiply dispatch disappears.
    func normScales(headKDim: Int, dtype: DType) -> (q: MLXArray, k: MLXArray) {
        lock.withLock {
            if let qScale, let kScale, qScale.dtype == dtype {
                return (qScale, kScale)
            }
            let invScale = pow(Float(headKDim), -0.5)
            let q = MLXArray(Array(repeating: pow(invScale, 2), count: headKDim)).asType(dtype)
            let k = MLXArray(Array(repeating: invScale, count: headKDim)).asType(dtype)
            qScale = q
            kScale = k
            return (q, k)
        }
    }
}

/// A norm gain with its consumer's Hadamard signs folded in, derived once from
/// the loaded gain and held outside the parameter tree (a plain class, so
/// Module reflection sees `.other`). The norm writes `w * y` per element; with
/// `w * s` it writes `(w * s) * y`, which is `(w * y) * s` exactly: the signs
/// are ±1, and negating a factor negates a rounded product without changing
/// its magnitude.
fileprivate final class Qwen35SignedGain {
    private let lock = NSLock()
    private var source: MLXArray?
    private var signs: MLXArray?
    private var folded: MLXArray?

    func gain(_ source: MLXArray, signs: MLXArray) -> MLXArray {
        lock.withLock {
            if let folded, self.source === source, self.signs === signs {
                return folded
            }
            let value = (source * signs).asType(source.dtype)
            self.source = source
            self.signs = signs
            self.folded = value
            return value
        }
    }

    func clear() {
        lock.withLock {
            source = nil
            signs = nil
            folded = nil
        }
    }
}

/// The gated delta recurrence with its gates formed by one fused kernel and
/// the state kept in FP32, matching `gatedDeltaUpdate` op for op. A
/// prompt-sized window (see `Qwen35GatedDeltaChunked`) takes the chunked form.
func qwen35GatedDelta(
    q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
    aLog: MLXArray, dtBias: MLXArray, state: MLXArray?, mask: MLXArray?,
    outputNeeded: Bool = true
) -> (MLXArray, MLXArray) {
    let gates = Qwen35FusedElementwise.gatedDeltaGates([a, b, aLog, dtBias])
    return qwen35GatedDelta(
        q: q, k: k, v: v, g: gates[0], beta: gates[1], state: state, mask: mask,
        outputNeeded: outputNeeded)
}

/// `qwen35GatedDelta` on gates already formed. The verify window's prework
/// kernel forms `g` and `beta` with `gatedDeltaGates`' formulas op for op
/// (`precise::exp`, MLX's `LogAddExp` softplus, the `Sigmoid` functor), so its
/// accepted-prefix replay slices them instead of recomputing them from the
/// taped a and b (DPZZxlz `5f72492`).
func qwen35GatedDelta(
    q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
    state: MLXArray?, mask: MLXArray?, outputNeeded: Bool = true
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    var ssm = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if ssm.dtype != .float32 {
        ssm = ssm.asType(.float32)
    }
    if mask == nil,
        let chunked = Qwen35GatedDeltaChunked.run(
            q: q, k: k, v: v, g: g, beta: beta, state: ssm)
    {
        return chunked
    }
    if mask == nil,
        let fast = Qwen35GatedDeltaV3.run(
            q: q, k: k, v: v, g: g, beta: beta, state: ssm,
            outputNeeded: outputNeeded)
    {
        return fast
    }
    if mask == nil, Qwen35GDNPrefillKernel.applies(q: q, v: v) {
        return Qwen35GDNPrefillKernel.run(
            q: q, k: k, v: v, g: g, beta: beta, state: ssm)
    }
    return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: ssm, mask: mask)
}

/// The gated-delta recurrence with a register-resident state layout: each
/// lane owns 16 consecutive dk of two dv rows (32 state floats), eight lanes
/// cover a dv row, one 128-thread threadgroup covers 32 dv rows, and the grid
/// is (128, Dv / 32, B * Hv). Per step the kv and out dot products run as
/// four independent FMA chains per row, reduced across the eight lanes with
/// three simd_shuffle_xor steps; the state update is one FMA per element. The
/// stock kernel (GatedDelta.swift) walks the same recurrence with each lane
/// owning 4 dk of ONE row, two 32-lane simd_sum reductions and a Kahan
/// compensation per step, and is ALU-bound: on the M5 Max this layout runs
/// the 512-row prefill recurrence 2.4x faster with a ~1e-7 relative
/// deviation of y and state (FP32 reassociation only; g < 1 keeps the
/// recurrence contractive). `DARKBLOOM_QWEN35_GDN_KERNEL=v1` keeps the stock
/// kernel; the masked (chain-verify) path always does.
enum Qwen35GatedDeltaV3 {
    private static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN35_GDN_KERNEL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return value != "v1" && !["0", "off", "false", "no"].contains(value ?? "")
    }()

    private static let source = """
        constexpr int R = 16;
        constexpr int DVPL = 2;
        constexpr int LPD = Dk / R;
        constexpr int DVPS = (32 / LPD) * DVPL;
        constexpr int DVPT = DVPS * 4;
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint dk0 = (lane % LPD) * R;
        const uint dvbase = threadgroup_position_in_grid.y * DVPT + sg * DVPS + (lane / LPD) * DVPL;
        const device float* q_ = q;
        const device float* k_ = k + (b_idx * T * Hk + hk_idx) * Dk + dk0;
        const device float* v_ = v + (b_idx * T * Hv + hv_idx) * Dv + dvbase;
        const device float* g_ = g + b_idx * T * Hv + hv_idx;
        const device float* beta_ = beta + b_idx * T * Hv + hv_idx;
        device float* y_ = y;
        if constexpr (OUTPUT_NEEDED) {
          q_ += (b_idx * T * Hk + hk_idx) * Dk + dk0;
          y_ += (b_idx * T * Hv + hv_idx) * Dv + dvbase;
        } else if (n == 0 && dvbase == 0 && lane == 0) {
          y[0] = 0.f;
        }
        float state[DVPL][R];
        #pragma clang loop unroll(full)
        for (int d = 0; d < DVPL; ++d) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < R; ++i) {
            state[d][i] = state_in[(n * Dv + dvbase + d) * Dk + dk0 + i];
          }
        }
        float kr[R];
        float qr[R];
        for (int t = 0; t < T; ++t) {
          const float gt = g_[0];
          const float bt = beta_[0];
          #pragma clang loop unroll(full)
          for (int j = 0; j < R / 4; ++j) {
            const float4 k4 = ((const device float4*)k_)[j];
            kr[4 * j] = k4.x; kr[4 * j + 1] = k4.y; kr[4 * j + 2] = k4.z; kr[4 * j + 3] = k4.w;
            if constexpr (OUTPUT_NEEDED) {
              const float4 q4 = ((const device float4*)q_)[j];
              qr[4 * j] = q4.x; qr[4 * j + 1] = q4.y; qr[4 * j + 2] = q4.z; qr[4 * j + 3] = q4.w;
            }
          }
          float kv[DVPL];
          float vt[DVPL];
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            vt[d] = v_[d];
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < R / 4; ++j) {
              state[d][4 * j] *= gt; state[d][4 * j + 1] *= gt;
              state[d][4 * j + 2] *= gt; state[d][4 * j + 3] *= gt;
              a0 = fma(state[d][4 * j], kr[4 * j], a0);
              a1 = fma(state[d][4 * j + 1], kr[4 * j + 1], a1);
              a2 = fma(state[d][4 * j + 2], kr[4 * j + 2], a2);
              a3 = fma(state[d][4 * j + 3], kr[4 * j + 3], a3);
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
          float out[DVPL];
          #pragma clang loop unroll(full)
          for (int d = 0; d < DVPL; ++d) {
            const float delta = (vt[d] - kv[d]) * bt;
            float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
            #pragma clang loop unroll(full)
            for (int j = 0; j < R / 4; ++j) {
              state[d][4 * j] = fma(kr[4 * j], delta, state[d][4 * j]);
              state[d][4 * j + 1] = fma(kr[4 * j + 1], delta, state[d][4 * j + 1]);
              state[d][4 * j + 2] = fma(kr[4 * j + 2], delta, state[d][4 * j + 2]);
              state[d][4 * j + 3] = fma(kr[4 * j + 3], delta, state[d][4 * j + 3]);
              if constexpr (OUTPUT_NEEDED) {
                o0 = fma(state[d][4 * j], qr[4 * j], o0);
                o1 = fma(state[d][4 * j + 1], qr[4 * j + 1], o1);
                o2 = fma(state[d][4 * j + 2], qr[4 * j + 2], o2);
                o3 = fma(state[d][4 * j + 3], qr[4 * j + 3], o3);
              }
            }
            if constexpr (OUTPUT_NEEDED) {
              out[d] = (o0 + o1) + (o2 + o3);
            }
          }
          if constexpr (OUTPUT_NEEDED) {
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
            q_ += Hk * Dk;
            y_ += Hv * Dv;
          }
          k_ += Hk * Dk; v_ += Hv * Dv; g_ += Hv; beta_ += Hv;
        }
        #pragma clang loop unroll(full)
        for (int d = 0; d < DVPL; ++d) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < R; ++i) {
            state_out[(n * Dv + dvbase + d) * Dk + dk0 + i] = state[d][i];
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gated_delta_v3",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: source,
        ensureRowContiguous: true)

    static func run(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray,
        outputNeeded: Bool = true
    ) -> (MLXArray, MLXArray)? {
        guard enabled, q.dtype == .float32, k.dtype == .float32, v.dtype == .float32,
            g.dtype == .float32, beta.dtype == .float32, state.dtype == .float32,
            q.ndim == 4, k.ndim == 4, v.ndim == 4
        else { return nil }
        let B = k.dim(0)
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Dk = k.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        guard Dk == 128, Dv % 32 == 0, Hv % Hk == 0, T > 0,
            q.shape == k.shape, state.shape == [B, Hv, Dv, Dk],
            g.shape == [B, T, Hv], beta.shape == [B, T, Hv]
        else { return nil }
        // Prefix replay consumes only state_out. Avoid copying its sliced q
        // input or allocating a full y tensor when no output row is needed.
        // The unused q slot aliases the already-contiguous gate buffer.
        let outputs = kernel(
            [outputNeeded ? q : g, k, v, g, beta, state, MLXArray(Int32(T))],
            template: [
                ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
                ("OUTPUT_NEEDED", outputNeeded),
            ],
            grid: (128, Dv / 32, B * Hv), threadGroup: (128, 1, 1),
            outputShapes: [outputNeeded ? [B, T, Hv, Dv] : [1], state.shape],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }
}

/// The gated-delta recurrence of a prompt-sized window in chunkwise-parallel
/// (WY) form, FP32 throughout. The window is cut into chunks of `C` rows; per
/// value head and chunk, with `gam_i` the in-chunk prefix sum of `log g`:
///
///     A_ij  = beta_i (k_i . k_j) exp(gam_i - gam_j)     (i > j)
///     T'    = (I + A)^-1 diag(beta)
///     P_ij  = (q_i . k_j) exp(gam_i - gam_j)            (i >= j)
///     Z     = V - diag(exp gam) K S^T
///     Delta = T' Z                                     (the per-row deltas)
///     Y     = diag(exp gam) Q S^T + P Delta
///     S^T  <- exp(gam_C) S^T + K^T diag(exp(gam_C - gam)) Delta
///
/// which is the sequential recurrence regrouped exactly; only the rounding
/// differs (FP32 8x8 simdgroup MMAs instead of per-step dot products), so it
/// is not bit-identical to `Qwen35GatedDeltaV3`. `prep` forms K K^T and Q K^T
/// once per (key head, chunk) and builds T', P and the decay factors of the
/// value heads sharing that key head; `scan` carries the state across the
/// chunks, one simdgroup per 8 state rows with its slice of the state held in
/// registers, the threadgroup sharing each chunk's K, Q, T' and P through
/// threadgroup memory. Both are ordinary lazy graph ops, evaluated with the
/// forward. Other windows under `minRows` (replay, decode) never take this
/// path; a capture-verify window of whole chunks does (`runVerify`). A prompt
/// window that is not a whole number of chunks runs its remainder rows on the
/// sequential kernel from the chunked state. `MLXFAST_GDN_CHUNKED=0` turns
/// the path off; `MLXFAST_GDN_CHUNK` picks the chunk (8, the default, or 16).
enum Qwen35GatedDeltaChunked {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_GDN_CHUNKED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static let chunk: Int = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_GDN_CHUNK"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let raw, let value = Int(raw), [8, 16].contains(value) { return value }
        return 8
    }()

    /// Shortest window that takes the chunked path.
    static let minRows = 64

    /// Simdgroups per scan threadgroup (each owns 8 state rows of one head).
    static let scanSimdgroups = 4

    static func supports(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
    ) -> Bool {
        guard q.ndim == 4, k.ndim == 4, v.ndim == 4, q.shape == k.shape else { return false }
        let B = k.dim(0)
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        return k.dim(3) == 128 && Dv % 8 == 0 && Hv % Hk == 0 && v.dim(1) == T
            && g.shape == [B, T, Hv] && beta.shape == [B, T, Hv]
            && state.shape == [B, Hv, Dv, 128]
            && q.dtype == .float32 && k.dtype == .float32 && v.dtype == .float32
            && g.dtype == .float32 && beta.dtype == .float32 && state.dtype == .float32
    }

    static func run(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
    ) -> (MLXArray, MLXArray)? {
        let T = k.dim(1)
        guard enabled, T >= minRows, T >= chunk,
            supports(q: q, k: k, v: v, g: g, beta: beta, state: state)
        else { return nil }
        return window(q: q, k: k, v: v, g: g, beta: beta, state: state)
    }

    /// Capture-verify windows (the DFlash block, 16 rows) of at least two
    /// whole chunks also take the chunked kernels. The verify only needs the
    /// rows' outputs and the window's final state (the replay of a strict
    /// prefix re-runs the sequential kernel from the retained pre-verify state
    /// and inputs), which is exactly what `chunks` returns. A window that is
    /// not a whole number of chunks stays on the sequential kernel: a
    /// sequential tail costs more than it saves at these widths.
    /// Off by default (`MLXFAST_GDN_CHUNKED_VERIFY=1` turns it on). On the
    /// ranked M5 every promoted tree that ran the chunked kernels at the
    /// 16-row verify had all three paired decode windows on the slow side
    /// (~52 ms per round), and the tree that turned them off with the rest
    /// of the verify-width changes had two of three on the fast side
    /// (~49.4 ms per round), so the verify keeps the sequential kernel.
    static let verifyEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_GDN_CHUNKED_VERIFY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return enabled && ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    static func runVerify(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
    ) -> (MLXArray, MLXArray)? {
        let T = k.dim(1)
        guard verifyEnabled, T >= 2 * chunk, T % chunk == 0,
            supports(q: q, k: k, v: v, g: g, beta: beta, state: state)
        else { return nil }
        return chunks(q: q, k: k, v: v, g: g, beta: beta, state: state)
    }

    /// Whole chunks on the chunked kernels, any remainder on the sequential one.
    private static func window(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let T = k.dim(1)
        let head = (T / chunk) * chunk
        if head == T {
            return chunks(q: q, k: k, v: v, g: g, beta: beta, state: state)
        }
        let rows = 0 ..< head
        let (yHead, sHead) = chunks(
            q: q[0..., rows], k: k[0..., rows], v: v[0..., rows], g: g[0..., rows],
            beta: beta[0..., rows], state: state)
        let tq = q[0..., head...]
        let tk = k[0..., head...]
        let tv = v[0..., head...]
        let tg = g[0..., head...]
        let tb = beta[0..., head...]
        let tail =
            Qwen35GatedDeltaV3.run(q: tq, k: tk, v: tv, g: tg, beta: tb, state: sHead)
            ?? gatedDeltaKernel(q: tq, k: tk, v: tv, g: tg, beta: tb, state: sHead, mask: nil)
        return (concatenated([yHead, tail.0], axis: 1), tail.1)
    }

    private static func chunks(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let B = k.dim(0)
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Dk = k.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        let C = chunk
        let NC = T / C
        let rowCount = MLXArray(Int32(T))
        let prepared = prepKernel(
            [q, k, g, beta, rowCount],
            template: [("C", C), ("Dk", Dk), ("Hk", Hk), ("Hv", Hv)],
            grid: (32, NC, B * Hk),
            threadGroup: (32, 1, 1),
            outputShapes: [[B, Hv, NC, C, C], [B, Hv, NC, C, C], [B, Hv, NC, 2, C]],
            outputDTypes: [.float32, .float32, .float32])
        let outputs = scanKernel(
            [q, k, v, prepared[0], prepared[1], prepared[2], state, rowCount],
            template: [
                ("C", C), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
                ("NS", scanSimdgroups),
            ],
            grid: (32, Dv / 8, B * Hv),
            threadGroup: (32, scanSimdgroups, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var prepared = Set<[Int]>()

    /// Builds both kernels' pipelines (and the remainder path's) for this
    /// geometry once per process, on throwaway inputs, so the first prompt
    /// window does not pay their compiles. Called from the layer's init.
    static func prepare(hk: Int, dk: Int, hv: Int, dv: Int) {
        guard enabled, hk > 0, hv % hk == 0 else { return }
        lock.withLock {
            guard prepared.insert([hk, dk, hv, dv]).inserted else { return }
            let rows = chunk + 3
            let q = MLXArray.zeros([1, rows, hk, dk], dtype: .float32)
            let v = MLXArray.zeros([1, rows, hv, dv], dtype: .float32)
            let g = MLXArray.ones([1, rows, hv], dtype: .float32)
            let beta = MLXArray.zeros([1, rows, hv], dtype: .float32)
            let state = MLXArray.zeros([1, hv, dv, dk], dtype: .float32)
            guard supports(q: q, k: q, v: v, g: g, beta: beta, state: state) else { return }
            let (y, s) = window(q: q, k: q, v: v, g: g, beta: beta, state: state)
            eval(y, s)
        }
    }

    private static let prepKernel = MLXFast.metalKernel(
        name: "bonsai_gated_delta_chunk_prep",
        inputNames: ["q", "k", "g", "beta", "T"],
        outputNames: ["tp", "pm", "gf"],
        source: prepSource)

    private static let scanKernel = MLXFast.metalKernel(
        name: "bonsai_gated_delta_chunk_scan",
        inputNames: ["q", "k", "v", "tp", "pm", "gf", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: scanSource)

    // BEGIN GENERATED CHUNKED GDN SOURCES
    private static let prepSource = """
            // grid: (32, NC, B * Hk). One simdgroup per (b, key head, chunk): K K^T and
            // Q K^T are formed once and serve the Hv / Hk value heads of this key
            // head, which run side by side in lane groups of C lanes (lane = head
            // group * C + row), GP heads per pass.
            constexpr int HR = Hv / Hk;
            constexpr int GP = (32 / C) < HR ? (32 / C) : HR;
            constexpr int CT = C / 8;
            constexpr int LD = C + 1;
            threadgroup float KK[C * LD];
            threadgroup float QK[C * LD];
            threadgroup float Ah[GP * C * LD];
            threadgroup float Th[GP * C * LD];
            const uint lane = thread_index_in_simdgroup;
            const int T_ = T;
            const int NC = T_ / C;
            const int n = int(thread_position_in_grid.y);
            const int bk = int(thread_position_in_grid.z);
            const int b_idx = bk / Hk;
            const int hk = bk % Hk;
            const int t0 = n * C;
            const int ks = Hk * Dk;
            const device float* k_ = k + ((size_t)b_idx * T_ + t0) * ks + hk * Dk;
            const device float* q_ = q + ((size_t)b_idx * T_ + t0) * ks + hk * Dk;

            // lower tiles of K K^T and Q K^T
            _Pragma("clang loop unroll(full)")
            for (int ti = 0; ti < CT; ++ti) {
              _Pragma("clang loop unroll(full)")
              for (int tj = 0; tj <= ti; ++tj) {
                simdgroup_float8x8 akk = simdgroup_float8x8(0);
                simdgroup_float8x8 aqk = simdgroup_float8x8(0);
                _Pragma("clang loop unroll(full)")
                for (int d = 0; d < Dk / 8; ++d) {
                  simdgroup_float8x8 ka, qa, kb;
                  simdgroup_load(ka, k_ + (ti * 8) * ks + d * 8, ks);
                  simdgroup_load(qa, q_ + (ti * 8) * ks + d * 8, ks);
                  simdgroup_load(kb, k_ + (tj * 8) * ks + d * 8, ks, ulong2(0, 0), true);
                  simdgroup_multiply_accumulate(akk, ka, kb, akk);
                  simdgroup_multiply_accumulate(aqk, qa, kb, aqk);
                }
                simdgroup_store(akk, KK + (ti * 8) * LD + tj * 8, LD);
                simdgroup_store(aqk, QK + (ti * 8) * LD + tj * 8, LD);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const int grp = int(lane) / C;
            const int row = int(lane) % C;
            const int gbase = grp * C;
            for (int h0 = 0; h0 < HR; h0 += GP) {
              const int h = h0 + grp;
              const bool live = grp < GP && h < HR;
              const int hv = hk * HR + (live ? h : 0);
              const size_t slot = ((size_t)b_idx * Hv + hv) * NC + n;
              // in-chunk inclusive prefix of log g within each lane group. A row
              // whose g underflowed (g == 0, or subnormal) has no finite log: it
              // zeroes every decay product spanning it, so it adds 0 to the prefix
              // and `zr` (the group's underflowed rows, one bit per row) masks
              // the products that span it.
              float gam = 0.0f;
              float bet = 0.0f;
              bool zero = false;
              if (live) {
                const float gv = g[((size_t)b_idx * T_ + t0 + row) * Hv + hv];
                zero = !(gv >= FLT_MIN);
                gam = zero ? 0.0f : metal::precise::log(zero ? 1.0f : gv);
                bet = beta[((size_t)b_idx * T_ + t0 + row) * Hv + hv];
              }
              const uint zr = uint(static_cast<simd_vote::vote_t>(simd_ballot(zero)) >> gbase) & ((1u << C) - 1u);
              _Pragma("clang loop unroll(full)")
              for (int off = 1; off < C; off <<= 1) {
                const float up = simd_shuffle_up(gam, ushort(off));
                gam += row >= off ? up : 0.0f;
              }
              const float gam_last = simd_shuffle(gam, ushort(gbase + C - 1));
              // rows (j, i] as a bit mask: ((2 << i) - 1) & ~((2 << j) - 1)
              const uint upto = (2u << row) - 1u;
              // decay factors for the scan: exp(gam_i), exp(gam_last - gam_i)
              if (live) {
                device float* gf_ = gf + slot * 2 * C;
                gf_[row] = (zr & upto) == 0u ? metal::precise::exp(gam) : 0.0f;
                gf_[C + row] = (zr & ~upto) == 0u ? metal::precise::exp(gam_last - gam) : 0.0f;
              }
              // P row (lower incl. diagonal) to device; A row (strict lower) to Ah
              threadgroup float* A_ = Ah + grp * C * LD;
              threadgroup float* M_ = Th + grp * C * LD;
              device float* pm_ = pm + slot * C * C;
              _Pragma("clang loop unroll(full)")
              for (int j = 0; j < C; ++j) {
                const float gj = simd_shuffle(gam, ushort(gbase + j));
                if (live) {
                  float pv = 0.0f;
                  float av = 0.0f;
                  if (j <= row) {
                    const uint span = upto & ~((2u << j) - 1u);
                    const float e = (zr & span) == 0u ? metal::precise::exp(gam - gj) : 0.0f;
                    pv = QK[row * LD + j] * e;
                    if (j < row) av = bet * KK[row * LD + j] * e;
                  }
                  pm_[row * C + j] = pv;
                  A_[row * LD + j] = av;
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              // T = (I + A)^-1, column m = row per lane, stored transposed in M_[m][i]
              if (live) {
                const int m = row;
                for (int i = 0; i < C; ++i) M_[m * LD + i] = (i == m) ? 1.0f : 0.0f;
                for (int i = m + 1; i < C; ++i) {
                  float s0 = 0.0f, s1 = 0.0f;
                  int j = m;
                  for (; j + 1 < i; j += 2) {
                    s0 = metal::fma(A_[i * LD + j], M_[m * LD + j], s0);
                    s1 = metal::fma(A_[i * LD + j + 1], M_[m * LD + j + 1], s1);
                  }
                  if (j < i) s0 = metal::fma(A_[i * LD + j], M_[m * LD + j], s0);
                  M_[m * LD + i] = -(s0 + s1);
                }
                // T' = T diag(beta): T'[i][m] = T[i][m] * beta_m
                device float* tp_ = tp + slot * C * C;
                for (int i = 0; i < C; ++i) tp_[i * C + m] = M_[m * LD + i] * bet;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        """

    private static let scanSource = """
            // grid: (32, Dv / 8, B * Hv), threadgroup (32, NS, 1). One simdgroup per
            // 8 state rows (its S^T columns held in registers across the chunks); the
            // threadgroup shares each chunk's K, Q, T' and P through threadgroup
            // memory.
            constexpr int CT = C / 8;
            constexpr int DT = Dk / 8;
            constexpr int LK = Dk + 8;
            constexpr int LC = C + 8;
            constexpr int NT = 32 * NS;
            constexpr int KQ4 = C * (Dk / 4);        // float4s of one chunk of K (or Q)
            constexpr int TP4 = C * (C / 4);         // float4s of one chunk of T' (or P)
            threadgroup float Ksh[C * LK];
            threadgroup float Qsh[C * LK];
            threadgroup float TPsh[2 * C * LC];
            const uint lane = thread_index_in_simdgroup;
            const int tid = int(thread_index_in_threadgroup);
            const int T_ = T;
            const int NC = T_ / C;
            const int r0 = int(thread_position_in_grid.y) * 8;
            const int bh = int(thread_position_in_grid.z);
            const int b_idx = bh / Hv;
            const int hv = bh % Hv;
            const int hk = hv / (Hv / Hk);
            const int ks = Hk * Dk;
            const int vs = Hv * Dv;
            const short qid = lane / 4;
            const short fm = (qid & 4) + ((lane / 2) % 4);
            const short fn = (qid & 2) * 2 + (lane % 2) * 2;
            const device float* kbase = k + (size_t)b_idx * T_ * ks + hk * Dk;
            const device float* qbase = q + (size_t)b_idx * T_ * ks + hk * Dk;
            const device float* tbase = tp + (size_t)bh * NC * C * C;
            const device float* pbase = pm + (size_t)bh * NC * C * C;

            simdgroup_float8x8 St[DT];
            _Pragma("clang loop unroll(full)")
            for (int d = 0; d < DT; ++d)
              simdgroup_load(St[d], state_in + ((size_t)bh * Dv + r0) * Dk + d * 8, Dk, ulong2(0, 0), true);

            for (int n = 0; n < NC; ++n) {
              const int t0 = n * C;
              const device float* v_ = v + ((size_t)b_idx * T_ + t0) * vs + hv * Dv + r0;
              device float* y_ = y + ((size_t)b_idx * T_ + t0) * vs + hv * Dv + r0;
              const device float* gf_ = gf + ((size_t)bh * NC + n) * 2 * C;
              // stage this chunk's K, Q, T', P (the previous chunk's readers are done)
              threadgroup_barrier(mem_flags::mem_threadgroup);
              for (int e = tid; e < KQ4; e += NT) {
                const int row = e / (Dk / 4);
                const int c4 = (e % (Dk / 4)) * 4;
                const size_t src = (size_t)(t0 + row) * ks + c4;
                *(threadgroup float4*)(Ksh + row * LK + c4) = *(const device float4*)(kbase + src);
                *(threadgroup float4*)(Qsh + row * LK + c4) = *(const device float4*)(qbase + src);
              }
              for (int e = tid; e < 2 * TP4; e += NT) {
                const int which = e / TP4;
                const int f = e % TP4;
                const int row = f / (C / 4);
                const int c4 = (f % (C / 4)) * 4;
                const device float* src = (which == 0 ? tbase : pbase) + (size_t)n * C * C;
                *(threadgroup float4*)(TPsh + which * C * LC + row * LC + c4) = *(const device float4*)(src + f * 4);
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);

              // X = K S^T, Xq = Q S^T (C x 8 each)
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
                  simdgroup_load(ka, Ksh + (ti * 8) * LK + d * 8, LK);
                  simdgroup_load(qa, Qsh + (ti * 8) * LK + d * 8, LK);
                  simdgroup_multiply_accumulate(Xk[ti], ka, St[d], Xk[ti]);
                  simdgroup_multiply_accumulate(Xq[ti], qa, St[d], Xq[ti]);
                }
              }
              // Z = V - diag(exp gam) Xk (in Xk); Xq <- diag(exp gam) Xq
              _Pragma("clang loop unroll(full)")
              for (int ti = 0; ti < CT; ++ti) {
                const int row = ti * 8 + fm;
                const float eg = gf_[row];
                thread auto& zk = Xk[ti].thread_elements();
                thread auto& zq = Xq[ti].thread_elements();
                const float2 vv = *(const device float2*)(v_ + row * vs + fn);
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
              // Y = diag(exp gam) Q S^T + P Delta
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
                *(device float2*)(y_ + row * vs + fn) = float2(ye[0], ye[1]);
              }
              // Delta~ = diag(exp(gam_C - gam)) Delta
              _Pragma("clang loop unroll(full)")
              for (int ti = 0; ti < CT; ++ti) {
                const int row = ti * 8 + fm;
                const float r = gf_[C + row];
                thread auto& de = Dl[ti].thread_elements();
                de[0] = de[0] * r;
                de[1] = de[1] * r;
              }
              // S^T <- exp(gam_C) S^T + K^T Delta~
              const float eC = gf_[C - 1];
              _Pragma("clang loop unroll(full)")
              for (int d = 0; d < DT; ++d) {
                thread auto& se = St[d].thread_elements();
                se[0] = se[0] * eC;
                se[1] = se[1] * eC;
                _Pragma("clang loop unroll(full)")
                for (int ti = 0; ti < CT; ++ti) {
                  simdgroup_float8x8 kt;
                  simdgroup_load(kt, Ksh + (ti * 8) * LK + d * 8, LK, ulong2(0, 0), true);
                  simdgroup_multiply_accumulate(St[d], kt, Dl[ti], St[d]);
                }
              }
            }
            _Pragma("clang loop unroll(full)")
            for (int d = 0; d < DT; ++d)
              simdgroup_store(St[d], state_out + ((size_t)bh * Dv + r0) * Dk + d * 8, Dk, ulong2(0, 0), true);
        """
    // END GENERATED CHUNKED GDN SOURCES
}

/// Wide-window (prefill) variant of the unmasked gated-delta kernel.
///
/// Same recurrence, same operands and the same per-element arithmetic as
/// `gatedDeltaKernel`: each lane keeps the same `Dk / 32` contiguous state
/// entries, the Kahan-compensated `kv_mem` partial runs in the same order under
/// the same fp pragmas, and every cross-lane total is the same XOR butterfly
/// over lane offsets 1, 2, 4, 8, 16 that `simd_sum` performs. What changes is
/// the layout: one simdgroup carries four `Dv` rows of a head (four
/// independent chains sharing each step's q/k/g/beta loads), and the four
/// rows' butterflies are interleaved (the first two levels exchange two and
/// one values instead of four, a transposed reduction) so a step issues 13
/// shuffles instead of 40. Pairwise sums are identical, so the outputs are
/// bit-identical; this is checked once per process on this device against
/// the stock kernel (random operands, both outputs compared bit for bit), and
/// the stock kernel is used if the check fails. Only windows of at least
/// `minimumT` rows use it; `MLXFAST_GDN_PREFILL_KERNEL=0` disables it.
enum Qwen35GDNPrefillKernel {
    static let minimumT = 64
    private static let rows = 4

    static let enabled: Bool =
        ProcessInfo.processInfo.environment["MLXFAST_GDN_PREFILL_KERNEL"] != "0"

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "qwen35_gated_delta_rows4",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: """
            constexpr int R = 4;
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]; v, y: [B, T, Hv, Dv]; g, beta: [B, T, Hv]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv0 = thread_position_in_grid.y * R;
            const uint lane = thread_index_in_simdgroup;
            const bool b0 = (lane & 1) != 0;
            const bool b1 = (lane & 2) != 0;

            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv0) * Dk;
            auto o_state = state_out + (n * Dv + dv0) * Dk;

            float state[R][n_per_t];
            for (int r = 0; r < R; ++r)
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[r][i] = static_cast<float>(i_state[r * Dk + s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              float kk[n_per_t];
              float qq[n_per_t];
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                kk[i] = static_cast<float>(k_[s_idx]);
                qq[i] = static_cast<float>(q_[s_idx]);
              }
              float gg = g_[hv_idx];
              float bb = beta_[hv_idx];
              float kv_mem[R];
              {
                // Preserve Kahan summation under Metal's default fast math.
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                for (int r = 0; r < R; ++r) {
                  float acc = 0.0f;
                  float kv_compensation = 0.0f;
                  for (int i = 0; i < n_per_t; ++i) {
                    state[r][i] = state[r][i] * gg;
                    auto product = state[r][i] * kk[i];
                    auto corrected = product - kv_compensation;
                    auto next_sum = acc + corrected;
                    kv_compensation = (next_sum - acc) - corrected;
                    acc = next_sum;
                  }
                  kv_mem[r] = acc;
                }
              }

              // Transposed butterfly: after offsets 1 and 2 lane (b0, b1)
              // holds row 2*b0 + b1; offsets 4, 8, 16 finish that row.
              float km[R];
              {
                float p0 = (b0 ? kv_mem[2] : kv_mem[0]) + simd_shuffle_xor(b0 ? kv_mem[0] : kv_mem[2], 1);
                float p1 = (b0 ? kv_mem[3] : kv_mem[1]) + simd_shuffle_xor(b0 ? kv_mem[1] : kv_mem[3], 1);
                float x = (b1 ? p1 : p0) + simd_shuffle_xor(b1 ? p0 : p1, 2);
                x = x + simd_shuffle_xor(x, 4);
                x = x + simd_shuffle_xor(x, 8);
                x = x + simd_shuffle_xor(x, 16);
                // Every lane needs all four totals.
                float x2 = simd_shuffle_xor(x, 2);
                float lo = b1 ? x2 : x;   // row 2*b0
                float hi = b1 ? x : x2;   // row 2*b0 + 1
                float lo1 = simd_shuffle_xor(lo, 1);
                float hi1 = simd_shuffle_xor(hi, 1);
                km[0] = b0 ? lo1 : lo;
                km[1] = b0 ? hi1 : hi;
                km[2] = b0 ? lo : lo1;
                km[3] = b0 ? hi : hi1;
              }

              float outp[R];
              for (int r = 0; r < R; ++r) {
                auto delta = (static_cast<float>(v_[dv0 + r]) - km[r]) * bb;
                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  state[r][i] = state[r][i] + kk[i] * delta;
                  out += state[r][i] * qq[i];
                }
                outp[r] = out;
              }
              {
                float p0 = (b0 ? outp[2] : outp[0]) + simd_shuffle_xor(b0 ? outp[0] : outp[2], 1);
                float p1 = (b0 ? outp[3] : outp[1]) + simd_shuffle_xor(b0 ? outp[1] : outp[3], 1);
                float x = (b1 ? p1 : p0) + simd_shuffle_xor(b1 ? p0 : p1, 2);
                x = x + simd_shuffle_xor(x, 4);
                x = x + simd_shuffle_xor(x, 8);
                x = x + simd_shuffle_xor(x, 16);
                if (lane < R) {
                  y[dv0 + (b0 ? 2 : 0) + (b1 ? 1 : 0)] = static_cast<InT>(x);
                }
              }
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int r = 0; r < R; ++r)
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[r * Dk + s_idx] = static_cast<StT>(state[r][i]);
            }
            """
    )

    static func applies(q: MLXArray, v: MLXArray) -> Bool {
        guard enabled, q.dim(1) >= minimumT, q.dim(3) % 32 == 0, v.dim(3) % rows == 0
        else { return false }
        return verified(q: q, v: v)
    }

    static func run(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let B = k.dim(0)
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Dk = k.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        let outputs = kernel(
            [q, k, v, g, beta, state, MLXArray(T)],
            template: [
                ("InT", q.dtype),
                ("StT", state.dtype),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
            ],
            grid: (32, Dv / rows, B * Hv),
            threadGroup: (32, 1, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape],
            outputDTypes: [q.dtype, state.dtype]
        )
        return (outputs[0], outputs[1])
    }

    private struct Geometry: Hashable {
        let hk: Int, dk: Int, hv: Int, dv: Int, dtype: String
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [Geometry: Bool] = [:]

    /// One-time, per-geometry device check: both kernels on the same random
    /// operands must agree bit for bit in `y` and in the final state.
    /// Verdict lookup only: the check itself runs at model construction
    /// (`prepare`), never inside a forward. A geometry or activation dtype
    /// that was not prepared uses the stock kernel.
    private static func verified(q: MLXArray, v: MLXArray) -> Bool {
        let geometry = Geometry(
            hk: q.dim(2), dk: q.dim(3), hv: v.dim(2), dv: v.dim(3), dtype: "\(q.dtype)")
        return lock.withLock { verdicts[geometry] ?? false }
    }

    /// Compile both kernels and run the bit-identity check for one head
    /// geometry, once per process, at model construction (before any timed
    /// forward). The window length is a runtime argument of both kernels, not
    /// a template parameter, so one pipeline serves every prefill width.
    /// Activations reach the recurrence in FP32 on the packed checkpoint;
    /// BF16 and FP16 are prepared too so no width or dtype compiles lazily.
    static func prepare(hk: Int, dk: Int, hv: Int, dv: Int) {
        guard enabled, dk % 32 == 0, dv % rows == 0, hv % hk == 0 else { return }
        lock.withLock {
            for dtype in [DType.float32, .bfloat16, .float16] {
                let geometry = Geometry(hk: hk, dk: dk, hv: hv, dv: dv, dtype: "\(dtype)")
                if verdicts[geometry] != nil { continue }
                let verdict = selfCheck(geometry, dtype: dtype)
                verdicts[geometry] = verdict
                if !verdict {
                    FileHandle.standardError.write(
                        "qwen35: GDN prefill kernel disagrees with the stock kernel on this device (\(dtype)); using the stock kernel\n"
                            .data(using: .utf8)!)
                }
            }
        }
    }

    private static func selfCheck(_ geo: Geometry, dtype: DType) -> Bool {
        let T = minimumT
        let keys = MLXRandom.split(key: MLXRandom.key(0x6d6c_7866), into: 9)
        // Wide magnitude spread so the pairwise sums actually differ by order.
        func spread(_ shape: [Int], _ i: Int) -> MLXArray {
            MLXRandom.normal(shape, key: keys[i])
                * exp(MLXRandom.normal(shape, key: keys[i + 3]))
        }
        let q = (spread([1, T, geo.hk, geo.dk], 0) * 0.1).asType(dtype)
        let k = (spread([1, T, geo.hk, geo.dk], 1) * 0.1).asType(dtype)
        let v = spread([1, T, geo.hv, geo.dv], 2).asType(dtype)
        let g = MLXRandom.uniform(0.5 ..< 1.0, [1, T, geo.hv], key: keys[6])
        let beta = MLXRandom.uniform(0.0 ..< 1.0, [1, T, geo.hv], key: keys[7])
        let state = MLXRandom.normal([1, geo.hv, geo.dv, geo.dk], key: keys[8])
        let (yRef, sRef) = gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state)
        let (yNew, sNew) = run(q: q, k: k, v: v, g: g, beta: beta, state: state)
        let bits: DType = dtype.size == 2 ? .uint16 : .uint32
        let same = all(yRef.view(dtype: bits) .== yNew.view(dtype: bits))
            .&& all(sRef.view(dtype: .uint32) .== sNew.view(dtype: .uint32))
        return same.item(Bool.self)
    }
}


/// Two dense sibling projections of the same input, stacked along the output
/// axis into one matmul (the same bytes, concatenated on first use, held in a
/// plain class). `DARKBLOOM_QWEN35_STACK_BA=0` keeps the two matmuls.
final class Qwen35DenseSiblingStack {
    private static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN35_STACK_BA"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
    private var weight: MLXArray?
    private var boundary = 0

    func clear() {
        weight = nil
        boundary = 0
    }

    /// `(b(x), a(x))` from one matmul, or nil when the stack does not apply
    /// (only plain, unquantized, bias-free `Linear` siblings of one dtype).
    func apply(_ x: MLXArray, b: Linear, a: Linear) -> (MLXArray, MLXArray)? {
        guard Self.enabled,
            ObjectIdentifier(type(of: b)) == ObjectIdentifier(Linear.self),
            ObjectIdentifier(type(of: a)) == ObjectIdentifier(Linear.self),
            b.bias == nil, a.bias == nil, b.weight.ndim == 2, a.weight.ndim == 2,
            b.weight.dtype == a.weight.dtype, b.weight.dim(1) == a.weight.dim(1),
            x.dtype == b.weight.dtype
        else { return nil }
        if weight == nil {
            weight = concatenated([b.weight, a.weight], axis: 0)
            boundary = b.weight.dim(0)
        }
        let y = matmul(x, weight!.T)
        return (y[.ellipsis, ..<boundary], y[.ellipsis, boundary...])
    }
}

final class Qwen35GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    // Inference-only cache. It is intentionally not registered in the module
    // topology, so checkpoint and adapter paths remain stable.
    private var fusedInProj: Linear?
    private var fusedInputSourceSignature: [MLXArray]?
    private var fusedInputPermanentlyIneligible = false

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    /// Derived norm weights; a plain box, never a parameter.
    private let derived = Qwen35GDNDerived()
    /// The dense `in_proj_b` and `in_proj_a` weights stacked along the output
    /// axis (same bytes, concatenated once, held outside the parameter
    /// tree): one N = 96 GEMM over the normed input instead of two N = 48
    /// GEMMs that each re-read it.
    private let baStack = Qwen35DenseSiblingStack()

    /// The gated output norm with its `silu(gate) * x` tail fused into one
    /// kernel; the same FP32 arithmetic as `norm(out, gate:)`.
    private func gatedNorm(_ out: MLXArray, gate: MLXArray) -> MLXArray {
        let normed = MLXFast.rmsNorm(out, weight: norm.weight, eps: norm.eps)
        return Qwen35FusedElementwise.gatedNormTail(normed, gate).asType(out.dtype)
    }

    /// `out_proj` on the normed output. When the projection is packed on the
    /// matrix route the FP16 product is left for the residual add to widen.
    private func projectOut(_ x: MLXArray) -> MLXArray {
        if let packed = outProj as? HadamardQuantizedLinear {
            return packed.forwardUnwidened(x)
        }
        return outProj(x)
    }

    /// `projectOut(gatedNorm(out, gate:))` with `out_proj`'s Hadamard signs
    /// multiplied in by the gated tail's kernel, so the packed projection's
    /// rotation skips its own sign multiply. Multiplying by ±1 is exact, so
    /// the rotation reads the same values either way.
    private func projectGatedOut(_ out: MLXArray, gate: MLXArray, B: Int, S: Int) -> MLXArray {
        // The per-head norm, the gated tail, the signs, the transform and the
        // route dtype's rounding in one kernel (ercumentyildirim, `ade7529`).
        if let packed = outProj as? HadamardQuantizedLinear,
            let y = packed.applyAfterGatedRMSNorm(
                out, gate: gate, weight: norm.weight, eps: norm.eps, widenOutput: false)
        {
            return y
        }
        // A narrow (FP16) qkv|z stack's gate is widened here for the op chain.
        let gate = gate.dtype == out.dtype ? gate : gate.asType(out.dtype)
        if Qwen35FusedElementwise.foldsHadamardSigns,
            let packed = outProj as? HadamardQuantizedLinear, packed.gdnLayout == nil,
            packed.transform.width == numVHeads * headVDim
        {
            let normed = MLXFast.rmsNorm(out, weight: norm.weight, eps: norm.eps)
            let signs = packed.transform.signVector.reshaped(numVHeads, headVDim)
            let signed = Qwen35FusedElementwise.gatedNormTailSigned(normed, gate, signs)
                .asType(out.dtype)
            return packed.forwardPreSigned(signed.reshaped(B, S, -1), widenOutput: false)
        }
        return projectOut(gatedNorm(out, gate: gate).reshaped(B, S, -1))
    }

    init(_ args: Qwen35TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        self.fusedInProj = nil
        self.fusedInputSourceSignature = nil

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()

        Qwen35GDNPrefillKernel.prepare(
            hk: numKHeads, dk: headKDim, hv: numVHeads, dv: headVDim)
        Qwen35GatedDeltaChunked.prepare(
            hk: numKHeads, dk: headKDim, hv: numVHeads, dv: headVDim)
    }

    private func exactQuantizedInputProjections() -> (
        qkv: QuantizedLinear, z: QuantizedLinear,
        b: QuantizedLinear, a: QuantizedLinear
    )? {
        guard let qkv = inProjQKV as? QuantizedLinear,
            let z = inProjZ as? QuantizedLinear,
            let b = inProjB as? QuantizedLinear,
            let a = inProjA as? QuantizedLinear,
            ObjectIdentifier(type(of: qkv)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: z)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: b)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: a)) == ObjectIdentifier(QuantizedLinear.self),
            qkv.bias == nil, z.bias == nil, b.bias == nil, a.bias == nil,
            qkv.bits == z.bits, qkv.bits == b.bits, qkv.bits == a.bits,
            qkv.groupSize == z.groupSize,
            qkv.groupSize == b.groupSize,
            qkv.groupSize == a.groupSize,
            qkv.mode == z.mode, qkv.mode == b.mode, qkv.mode == a.mode
        else { return nil }
        return (qkv, z, b, a)
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        baStack.clear()
        let prefixes = ["in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a."]
        let replacesInputProjection = parameters.flattened().contains { key, _ in
            prefixes.contains(where: key.hasPrefix)
        }
        let result = try super.update(
            parameters: parameters, verify: verify,
            path: path, modulePath: modulePath)
        if replacesInputProjection {
            fusedInProj = nil
            fusedInputSourceSignature = nil
            fusedInputPermanentlyIneligible = false
        }
        return result
    }

    override func updateModule(key: String, _ value: Any) throws {
        try super.updateModule(key: key, value)
        if key == "in_proj_qkv" || key == "in_proj_z"
            || key == "in_proj_b" || key == "in_proj_a"
        {
            fusedInProj = nil
            fusedInputSourceSignature = nil
            fusedInputPermanentlyIneligible = false
        }
    }

    var hasFusedInputProjection: Bool { fusedInProj != nil }

    private func inputProjectionSourceSignature(
        _ projections: (
            qkv: QuantizedLinear, z: QuantizedLinear,
            b: QuantizedLinear, a: QuantizedLinear
        )
    ) -> [MLXArray] {
        [
            projections.qkv.weight, projections.qkv.scales,
            projections.z.weight, projections.z.scales,
            projections.b.weight, projections.b.scales,
            projections.a.weight, projections.a.scales,
        ] + [
            projections.qkv.biases, projections.z.biases,
            projections.b.biases, projections.a.biases,
        ].compactMap { $0 }
    }

    private func sourceSignatureMatches(_ current: [MLXArray], _ cached: [MLXArray]) -> Bool {
        current.count == cached.count
            && zip(current, cached).allSatisfy { $0 === $1 }
    }

    private func exactFrozenQuantizedInputProjections() -> (
        qkv: QuantizedLinear, z: QuantizedLinear,
        b: QuantizedLinear, a: QuantizedLinear
    )? {
        // Reject ineligible module types and policies before traversing the
        // trainable-parameter tree on hot decode forwards.
        guard let qkv = inProjQKV as? QuantizedLinear,
            let z = inProjZ as? QuantizedLinear,
            let b = inProjB as? QuantizedLinear,
            let a = inProjA as? QuantizedLinear,
            ObjectIdentifier(type(of: qkv)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: z)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: b)) == ObjectIdentifier(QuantizedLinear.self),
            ObjectIdentifier(type(of: a)) == ObjectIdentifier(QuantizedLinear.self),
            qkv.bias == nil, z.bias == nil, b.bias == nil, a.bias == nil,
            qkv.bits == z.bits, qkv.bits == b.bits, qkv.bits == a.bits,
            qkv.groupSize == z.groupSize,
            qkv.groupSize == b.groupSize,
            qkv.groupSize == a.groupSize,
            qkv.mode == z.mode, qkv.mode == b.mode, qkv.mode == a.mode
        else {
            fusedInputPermanentlyIneligible = true
            return nil
        }
        let prefixes = ["in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a."]
        guard !trainableParameters().flattened().contains(where: { key, _ in
            prefixes.contains(where: key.hasPrefix)
        }) else { return nil }
        return (qkv, z, b, a)
    }

    @discardableResult
    func prepareFusedInputProjection() -> Bool {
        if fusedInputPermanentlyIneligible { return false }
        if let fusedInputSourceSignature {
            guard let projections = exactFrozenQuantizedInputProjections(),
                sourceSignatureMatches(inputProjectionSourceSignature(projections), fusedInputSourceSignature)
            else {
                fusedInProj = nil
                self.fusedInputSourceSignature = nil
                return false
            }
            return fusedInProj != nil
        }
        guard let projections = exactFrozenQuantizedInputProjections() else { return false }
        let fusedBiases: MLXArray?
        switch (
            projections.qkv.biases, projections.z.biases,
            projections.b.biases, projections.a.biases
        ) {
        case let (.some(qkv), .some(z), .some(b), .some(a)):
            fusedBiases = concatenated([qkv, z, b, a], axis: 0)
        case (nil, nil, nil, nil):
            fusedBiases = nil
        default:
            return false
        }
        let fusedWeight = concatenated([
            projections.qkv.weight, projections.z.weight,
            projections.b.weight, projections.a.weight,
        ], axis: 0)
        let fusedScales = concatenated([
            projections.qkv.scales, projections.z.scales,
            projections.b.scales, projections.a.scales,
        ], axis: 0)
        eval(fusedWeight, fusedScales)
        if let fusedBiases { eval(fusedBiases) }

        let fused = QuantizedLinear(
            weight: fusedWeight, bias: nil,
            scales: fusedScales, biases: fusedBiases,
            groupSize: projections.qkv.groupSize,
            bits: projections.qkv.bits,
            mode: projections.qkv.mode)
        fused.freeze()

        let qkvRows = keyDim * 2 + valueDim
        let zRows = valueDim
        let bRows = numVHeads
        let ranges = [
            0 ..< qkvRows,
            qkvRows ..< (qkvRows + zRows),
            (qkvRows + zRows) ..< (qkvRows + zRows + bRows),
            (qkvRows + zRows + bRows) ..< (qkvRows + zRows + 2 * bRows),
        ]
        func sourceView(_ rows: Range<Int>) -> QuantizedLinear {
            let view = QuantizedLinear(
                weight: fusedWeight[rows], bias: nil,
                scales: fusedScales[rows],
                biases: fusedBiases.map { $0[rows] },
                groupSize: projections.qkv.groupSize,
                bits: projections.qkv.bits,
                mode: projections.qkv.mode)
            view.freeze()
            return view
        }
        // Preserve checkpoint/adaptor-facing module names as views into the
        // one fused physical allocation. A later module replacement invalidates
        // `fusedInProj` through updateModule before the next forward.
        try! update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(sourceView(ranges[0])),
                "in_proj_z": .value(sourceView(ranges[1])),
                "in_proj_b": .value(sourceView(ranges[2])),
                "in_proj_a": .value(sourceView(ranges[3])),
            ]), verify: [])
        // updateModule invalidates on source replacement; assign only after
        // the stable named views have been installed.
        fusedInProj = fused
        guard let current = exactFrozenQuantizedInputProjections() else {
            fusedInProj = nil
            return false
        }
        fusedInputSourceSignature = inputProjectionSourceSignature(current)
        return true
    }

    /// The packed qkv and z projections when they share one input transform
    /// (the rotation `projectInputs` computes once for both); nil otherwise.
    var inputRotationSiblings: [HadamardQuantizedLinear]? {
        sharedHadamardSiblings([inProjQKV, inProjZ])
    }

    /// `preRotated`, when given, is `inputs` already rotated by the shared
    /// qkv|z transform and stored in the dtype their stacked matmul reads
    /// (see `Qwen35DecoderLayer.cbv2ForwardPending`); b and a still read
    /// `inputs`.
    /// `narrowStack`: the packed qkv|z product stays in the route's output
    /// dtype (FP16) instead of being widened to FP32 as one [rows, 16384]
    /// cast. Its CBv2 consumers widen exactly at the read: the prework kernel
    /// (`InT`), the conv fallback's concatenation with the FP32 conv state,
    /// and the gated-norm rotation's `z` (other `projectGatedOut` paths
    /// widen `z` first). Same values; one fewer [rows, 16384] FP32 pass.
    static let narrowStackEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_NARROW_STACK"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private func projectInputs(
        _ inputs: MLXArray, B: Int, S: Int, preRotated: MLXArray? = nil,
        narrowStack: Bool = false, preQuantized: SignedBlockHadamard.Int8Activation? = nil
    ) -> (
        qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray
    ) {
        guard prepareFusedInputProjection(), let fusedInProj else {
            // The prompt-width tensor route on the boundary kernel's codes
            // (`preQuantized`): the qkv|z product as `sharedHadamardProjections`
            // returns it on that route.
            if let preQuantized, let siblings = inputRotationSiblings, siblings.count == 2,
                let shared = siblings[0].tensorRouteForwardQuantized(
                    preQuantized, siblings: siblings, leading: [B, S], widenOutput: !narrowStack),
                shared.count == 2
            {
                if let (bOut, aOut) = baStack.apply(inputs, b: inProjB, a: inProjA) {
                    return (shared[0], shared[1].reshaped(B, S, numVHeads, headVDim), bOut, aOut)
                }
                return (
                    shared[0],
                    shared[1].reshaped(B, S, numVHeads, headVDim),
                    inProjB(inputs),
                    inProjA(inputs)
                )
            }
            // Packed qkv and z read the same activation through the same
            // transform; rotate it once. b and a stay full precision.
            if let preRotated, let siblings = inputRotationSiblings,
                let (wide, boundaries) = sharedHadamardStackOnRotated(
                    preRotated, siblings, widenOutput: !narrowStack),
                boundaries.count == 2
            {
                let shared = MLX.split(wide, indices: [boundaries[0]], axis: -1)
                if let (bOut, aOut) = baStack.apply(inputs, b: inProjB, a: inProjA) {
                    return (shared[0], shared[1].reshaped(B, S, numVHeads, headVDim), bOut, aOut)
                }
                return (
                    shared[0],
                    shared[1].reshaped(B, S, numVHeads, headVDim),
                    inProjB(inputs),
                    inProjA(inputs)
                )
            }
            if let shared = sharedHadamardProjections(
                inputs, [inProjQKV, inProjZ], widenOutput: !narrowStack)
            {
                if let (bOut, aOut) = baStack.apply(inputs, b: inProjB, a: inProjA) {
                    return (shared[0], shared[1].reshaped(B, S, numVHeads, headVDim), bOut, aOut)
                }
                return (
                    shared[0],
                    shared[1].reshaped(B, S, numVHeads, headVDim),
                    inProjB(inputs),
                    inProjA(inputs)
                )
            }
            return (
                inProjQKV(inputs),
                inProjZ(inputs).reshaped(B, S, numVHeads, headVDim),
                inProjB(inputs),
                inProjA(inputs)
            )
        }
        let outFused = fusedInProj(inputs)
        let qkvDim = keyDim * 2 + valueDim
        let zDim = valueDim
        let bDim = numVHeads
        let total = qkvDim + zDim + 2 * bDim
        return (
            outFused[0..., 0..., 0 ..< qkvDim],
            outFused[0..., 0..., qkvDim ..< (qkvDim + zDim)].reshaped(
                B, S, numVHeads, headVDim),
            outFused[0..., 0..., (qkvDim + zDim) ..< (qkvDim + zDim + bDim)],
            outFused[0..., 0..., (qkvDim + zDim + bDim) ..< total]
        )
    }

    /// The retained convolution tail is three rows, but a slice aliases the
    /// whole chunk. A packed checkpoint carries that parent into every
    /// recurrent transaction, which costs gigabytes over a prefill. Copying
    /// the tail drops the parent. It moves bytes; it changes no bit, no
    /// precision and no recurrence, and it is skipped for single-token decode
    /// and for every other checkpoint format.
    private func retainedConvTail(
        of convInput: MLXArray, keeping nKeep: Int, chunkWidth: Int
    ) -> MLXArray {
        let tail = convInput[0..., (convInput.dim(1) - nKeep)...]
        guard chunkWidth > 1, inProjQKV is HadamardQuantizedLinear else { return tail }
        return contiguous(tail)
    }

    // MARK: - _processChunk (MTP helper)

    /// Process one time-chunk of the linear-attention layer.
    ///
    /// Extracted from `callAsFunction` so the MTP verify cycle can run the prefix
    /// (n_confirmed tokens) and draft suffix separately, snapshotting the SSM/conv
    /// state in between for rollback on draft rejection.
    ///
    /// Port of omlx commit 696d90a:
    ///   patches/mlx_lm_mtp/qwen35_model.py `GatedDeltaNet._process_chunk`
    ///
    /// - Parameters:
    ///   - qkv: Already-masked QKV for this chunk [B, S_chunk, conv_dim]
    ///   - a, b: Input projections for this chunk [B, S_chunk, ...]
    ///   - convState: Initial conv state [B, conv_kernel_size-1, conv_dim]
    ///   - ssmState: Initial SSM state (nil on first token)
    ///   - mask: SSM mask for `gatedDeltaUpdate` (optional)
    /// - Returns: `(out, newConvState, newSsmState)`
    private func processChunk(
        qkv: MLXArray,
        a: MLXArray,
        b: MLXArray,
        convState: MLXArray,
        ssmState: MLXArray?,
        mask: MLXArray?
    ) -> (out: MLXArray, newConvState: MLXArray, newSsmState: MLXArray) {
        let B = qkv.dim(0)
        let S = qkv.dim(1)

        if mask == nil, convKernelSize == 4,
            let pre = Qwen35GDNPrework.run(
                qkv: qkv, convState: convState, convWeight: conv1d.weight, a: a, b: b,
                aLog: aLog, dtBias: dtBias,
                normScales: derived.normScales(headKDim: headKDim, dtype: .float32),
                keyHeads: numKHeads, valueHeads: numVHeads, headKDim: headKDim,
                headVDim: headVDim)
        {
            var ssm = ssmState ?? MLXArray.zeros([B, numVHeads, headVDim, headKDim], dtype: .float32)
            if ssm.dtype != .float32 { ssm = ssm.asType(.float32) }
            let (out, newSsmState) =
                Qwen35GatedDeltaChunked.run(
                    q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta, state: ssm)
                ?? Qwen35GatedDeltaV3.run(
                    q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta, state: ssm)
                ?? gatedDeltaKernel(
                    q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta, state: ssm, mask: nil)
            return (out, pre.tail, newSsmState)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = retainedConvTail(of: convInput, keeping: nKeep, chunkWidth: S)
        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let dtype = q.dtype
        let scales = derived.normScales(headKDim: headKDim, dtype: dtype)
        let qNormed = MLXFast.rmsNorm(q, weight: scales.q, eps: 1e-6)
        let kNormed = MLXFast.rmsNorm(k, weight: scales.k, eps: 1e-6)

        let (out, newSsmState) = qwen35GatedDelta(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: ssmState,
            mask: mask
        )
        return (out, newConvState, newSsmState)
    }

    /// Run one legacy-cache verify chunk while retaining only the transformed
    /// recurrence inputs needed to rebuild a shorter committed prefix. The
    /// CBv2 rectangular path below uses the same tape and replay math.
    private func processChunkStashingPrefix(
        qkv: MLXArray,
        a: MLXArray,
        b: MLXArray,
        convState: MLXArray,
        ssmState: MLXArray?,
        mask: MLXArray?
    ) -> (
        out: MLXArray, newConvState: MLXArray, newSsmState: MLXArray,
        tape: ArraysCache.PrefixReplayTape
    ) {
        let B = qkv.dim(0)
        let S = qkv.dim(1)
        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = retainedConvTail(of: convInput, keeping: nKeep, chunkWidth: S)
        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let recurrence = gatedDeltaUpdate(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: ssmState,
            mask: mask
        )
        let tape = ArraysCache.PrefixReplayTape(
            convInput: convInput,
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            ssmPre: ssmState.map { $0[.ellipsis] },
            mask: mask.map { $0[.ellipsis] },
            rowCount: S,
            convStateRows: nKeep
        )
        return (recurrence.0, newConvState, recurrence.1, tape)
    }

    private func canReplayPrefix(
        tape: ArraysCache.PrefixReplayTape, committedRows: Int
    ) -> Bool {
        guard committedRows > 0,
              committedRows < tape.rowCount,
              tape.convStateRows == convKernelSize - 1,
              tape.convInput.ndim == 3,
              tape.q.ndim == 4,
              tape.k.ndim == 4,
              tape.v.ndim == 4,
              tape.a.ndim == 3,
              tape.b.ndim == 3
        else { return false }

        let batch = tape.q.dim(0)
        guard tape.convInput.shape
                  == [batch, tape.convStateRows + tape.rowCount, convDim],
              tape.q.shape == [batch, tape.rowCount, numKHeads, headKDim],
              tape.k.shape == [batch, tape.rowCount, numKHeads, headKDim],
              tape.v.shape == [batch, tape.rowCount, numVHeads, headVDim],
              tape.a.shape == [batch, tape.rowCount, numVHeads],
              tape.b.shape == [batch, tape.rowCount, numVHeads]
        else { return false }

        if let ssmPre = tape.ssmPre,
           ssmPre.shape != [batch, numVHeads, headVDim, headKDim]
        {
            return false
        }
        if let mask = tape.mask,
           mask.shape != [batch, tape.rowCount]
        {
            return false
        }
        return true
    }

    fileprivate func canReplayPrefix(
        cache: MambaCache, committedRows: Int
    ) -> Bool {
        guard let tape = cache.prefixReplayTape else { return false }
        return canReplayPrefix(tape: tape, committedRows: committedRows)
    }

    private func replayedPrefixState(
        tape: ArraysCache.PrefixReplayTape, committedRows: Int,
        gates: (g: MLXArray, beta: MLXArray)? = nil
    ) -> CBv2RecurrentLayerState {
        precondition(
            canReplayPrefix(tape: tape, committedRows: committedRows),
            "Qwen35 invalid compact recurrent prefix replay")
        let rows = 0 ..< committedRows
        let boundarySsm: MLXArray
        if let gates, tape.mask == nil, Self.replayReusesGates {
            // The verify window's own gates, sliced: no recompute of g and
            // beta from the taped a and b.
            boundarySsm = qwen35GatedDelta(
                q: tape.q[0..., rows, 0...],
                k: tape.k[0..., rows, 0...],
                v: tape.v[0..., rows, 0...],
                g: gates.g[0..., rows, 0...],
                beta: gates.beta[0..., rows, 0...],
                state: tape.ssmPre,
                mask: nil,
                outputNeeded: false
            ).1
        } else {
            boundarySsm = qwen35GatedDelta(
                q: tape.q[0..., rows, 0...],
                k: tape.k[0..., rows, 0...],
                v: tape.v[0..., rows, 0...],
                a: tape.a[0..., rows, 0...],
                b: tape.b[0..., rows, 0...],
                aLog: aLog,
                dtBias: dtBias,
                state: tape.ssmPre,
                mask: tape.mask.map { $0[0..., rows] },
                outputNeeded: false
            ).1
        }
        let boundaryConvView = tape.convInput[
            0...,
            committedRows ..< (committedRows + tape.convStateRows),
            0...]
        // A contiguous copy detaches the three retained rows from the whole
        // window's conv input, as the zero-add did, in one dispatch.
        let boundaryConv = contiguous(boundaryConvView)
        return CBv2RecurrentLayerState(conv: boundaryConv, ssm: boundarySsm)
    }

    /// `BONSAI_PREWORK_CONV_INPUT=0` concatenates the tape's conv input with
    /// ops (a cast and two copies) instead of writing it from the prework kernel.
    static let preworkWritesConvInput: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_PREWORK_CONV_INPUT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Default OFF (`BONSAI_REPLAY_REUSES_GATES=1` slices the verify window's
    /// prework gates into the replay). Bit-identical either way, but every
    /// ranked M5 bundle that carried it measured a longer decode window
    /// (DPZZxlz `5f72492` +2.9%, ercumentyildirim `94da3f5` +1.4%, our v6
    /// +2.9%), and the one sliced-verify bundle without it measured shorter.
    static let replayReusesGates: Bool =
        ProcessInfo.processInfo.environment["BONSAI_REPLAY_REUSES_GATES"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "1"

    /// Reconstruct the fp32 recurrent state after `committedRows` verify rows
    /// from the exact pre-verify state and transformed recurrence inputs.
    /// Call only after every GDN layer has passed `canReplayPrefix`.
    fileprivate func replayPrefix(
        cache: MambaCache, committedRows: Int
    ) -> Bool {
        guard canReplayPrefix(cache: cache, committedRows: committedRows),
              let tape = cache.prefixReplayTape
        else { return false }

        let state = replayedPrefixState(tape: tape, committedRows: committedRows)
        cache[0] = state.conv
        cache[1] = state.ssm
        cache.clearMTPTransientState()
        return true
    }

    // MARK: - callAsFunction

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil,
        nConfirmed: Int = 0
    ) -> MLXArray {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py GatedDeltaNet.__call__
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        // A verify tape belongs to exactly one forward. Starting another
        // forward consumes neither its values nor its state, so release it
        // before constructing this forward's transient rollback data.
        cache?.clearMTPTransientState()

        var (qkv, z, b, a) = projectInputs(inputs, B: B, S: S)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        // Apply mask to full qkv before any chunking.
        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let ssmState = cache?[1]
        let out: MLXArray
        let finalConvState: MLXArray
        let finalSsmState: MLXArray
        var pendingPrefixTape: ArraysCache.PrefixReplayTape?

        if nConfirmed == 1 && S >= 3 && mask == nil {
            // A wider MTP verify stays as one recurrence. Retain transformed
            // inputs instead of one full fp32 SSM checkpoint per boundary.
            let (o, c, s, tape) = processChunkStashingPrefix(
                qkv: qkv, a: a, b: b,
                convState: convState, ssmState: ssmState, mask: mask)
            out = o
            finalConvState = c
            finalSsmState = s
            pendingPrefixTape = tape
        } else if nConfirmed > 0 && nConfirmed < S {
            // Width-2 and masked verification retain the established rollback
            // snapshot path. CBv2 capture uses its separate request-state API.
            let maskC = mask.map { $0[0..., 0..<nConfirmed] }
            let maskD = mask.map { $0[0..., nConfirmed...] }

            let (outC, convC, ssmC) = processChunk(
                qkv: qkv[0..., 0..<nConfirmed, 0...],
                a: a[0..., 0..<nConfirmed, 0...],
                b: b[0..., 0..<nConfirmed, 0...],
                convState: convState,
                ssmState: ssmState,
                mask: maskC
            )
            cache?.rollbackState = (convC, ssmC)

            let (outD, convF, ssmF) = processChunk(
                qkv: qkv[0..., nConfirmed..., 0...],
                a: a[0..., nConfirmed..., 0...],
                b: b[0..., nConfirmed..., 0...],
                convState: convC,
                ssmState: ssmC,
                mask: maskD
            )
            out = concatenated([outC, outD], axis: 1)
            finalConvState = convF
            finalSsmState = ssmF
        } else {
            let (o, c, s) = processChunk(
                qkv: qkv, a: a, b: b,
                convState: convState,
                ssmState: ssmState,
                mask: mask
            )
            out = o
            finalConvState = c
            finalSsmState = s
        }

        if let cache {
            cache[0] = finalConvState
            cache[1] = finalSsmState
            cache.prefixReplayTape = pendingPrefixTape
        }

        let normedOut = norm(out, gate: z)
        return outProj(normedOut.reshaped(B, S, -1))
    }

    /// CBv2 target path. Request-owned conv/SSM rows are gathered into the
    /// active rectangle, evaluated once, then split back into their owning
    /// transactions. No recurrent tensor is represented as attention KV.
    func cbv2Forward(
        _ inputs: MLXArray,
        modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation],
        preRotated: MLXArray? = nil,
        preQuantized: SignedBlockHadamard.Int8Activation? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)
        precondition(recurrentState.count == B, "Qwen35 CBv2 recurrent row count mismatch")

        let (qkv, z, b, a) = projectInputs(
            inputs, B: B, S: S, preRotated: preRotated,
            narrowStack: Self.narrowStackEnabled
                && B * S >= HadamardQuantizedLinear.promptKernelsMinimumRows,
            preQuantized: preQuantized)

        var convRows: [MLXArray] = []
        var ssmRows: [MLXArray] = []
        convRows.reserveCapacity(B)
        ssmRows.reserveCapacity(B)
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: modelLayerIndex)
            convRows.append(
                state?.conv
                    ?? MLXArray.zeros(
                        [1, convKernelSize - 1, convDim], dtype: inputs.dtype))
            ssmRows.append(
                state?.ssm
                    ?? MLXArray.zeros(
                        [1, numVHeads, headVDim, headKDim], dtype: .float32))
        }

        let convState = convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        let ssmState = ssmRows.count == 1 ? ssmRows[0] : concatenated(ssmRows, axis: 0)
        let (out, newConvState, newSsmState) = processChunk(
            qkv: qkv, a: a, b: b,
            convState: convState, ssmState: ssmState, mask: nil)

        for (row, evaluation) in recurrentState.enumerated() {
            do {
                try evaluation.stage(
                    modelLayerIndex: modelLayerIndex,
                    conv: newConvState[row ..< row + 1],
                    ssm: newSsmState[row ..< row + 1])
            } catch {
                preconditionFailure(
                    "Qwen35 CBv2 recurrent stage failed at layer \(modelLayerIndex): \(error)")
            }
        }

        return projectGatedOut(out, gate: z, B: B, S: S)
    }

    /// CBv2 MTP rectangular verify path. Widths one and two retain the
    /// established captured-boundary implementation. Wider windows run the
    /// recurrence once over the whole window and retain compact transformed
    /// inputs, allowing a strict accepted prefix to be replayed lazily without
    /// another target-model forward or a per-position fp32 SSM stack.
    func cbv2ForwardCaptured(
        _ inputs: MLXArray,
        modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation],
        exactTargetVerify: Bool = false,
        preRotated: MLXArray? = nil,
        preQuantized: SignedBlockHadamard.Int8Activation? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)
        precondition(recurrentState.count == B, "Qwen35 CBv2 recurrent row count mismatch")
        precondition(S >= 1, "Qwen35 capture-verify window must be non-empty")

        let qkv: MLXArray
        let z: MLXArray
        let b: MLXArray
        let a: MLXArray
        if exactTargetVerify {
            let exact = qwen35A3BExactW4G64ProjectionQuad(
                inProjQKV, inProjZ, inProjB, inProjA, inputs)
            qkv = exact.0
            z = exact.1.reshaped(B, S, numVHeads, headVDim)
            b = exact.2
            a = exact.3
        } else {
            // Preserve main's fused GDN projection construction and graph.
            (qkv, z, b, a) = projectInputs(
                inputs, B: B, S: S, preRotated: preRotated,
                narrowStack: Self.narrowStackEnabled
                    && B * S >= HadamardQuantizedLinear.promptKernelsMinimumRows,
                preQuantized: preQuantized)
        }

        var convRows: [MLXArray] = []
        var ssmRows: [MLXArray] = []
        convRows.reserveCapacity(B)
        ssmRows.reserveCapacity(B)
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: modelLayerIndex)
            convRows.append(
                state?.conv
                    ?? MLXArray.zeros(
                        [1, convKernelSize - 1, convDim], dtype: inputs.dtype))
            ssmRows.append(
                state?.ssm
                    ?? MLXArray.zeros(
                        [1, numVHeads, headVDim, headKDim], dtype: .float32))
        }
        let convState = convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        let ssmState = ssmRows.count == 1 ? ssmRows[0] : concatenated(ssmRows, axis: 0)

        // Conv over the whole window in one call (same as processChunk); the
        // per-position conv tail is a free slice of the padded input: after
        // consuming position s, the retained tail is convInput[:, s+1 ..<
        // s+1+nKeep].
        let nKeep = convKernelSize - 1
        // The fused prework kernel (conv, SiLU, split, q/k norms, gates, tail)
        // serves the wide verify window too; the replay tape keeps the
        // concatenated conv input for its boundary rows, which the prework
        // kernel also writes (the same FP32 values as the concatenation of the
        // state with the widened qkv, without its cast and copy launches).
        let pre: Qwen35GDNPrework.Outputs? =
            (!exactTargetVerify && S >= 3 && convKernelSize == 4)
            ? Qwen35GDNPrework.run(
                qkv: qkv, convState: convState, convWeight: conv1d.weight, a: a, b: b,
                aLog: aLog, dtBias: dtBias,
                normScales: derived.normScales(headKDim: headKDim, dtype: .float32),
                keyHeads: numKHeads, valueHeads: numVHeads, headKDim: headKDim,
                headVDim: headVDim,
                writeConvInput: Self.preworkWritesConvInput
                    && qkv.dtype != .bfloat16 && convState.dtype == .float32
                    && B * S >= HadamardQuantizedLinear.promptKernelsMinimumRows)
            : nil
        let convInput = pre?.convInput ?? concatenated([convState, qkv], axis: 1)
        let qNormed: MLXArray
        let kNormed: MLXArray
        let v: MLXArray
        let convOutBackingAll: MLXArray
        if let pre {
            qNormed = pre.q
            kNormed = pre.k
            v = pre.v
            convOutBackingAll = pre.v
        } else {
            let convOut: MLXArray
            if exactTargetVerify, S > 1 {
                convOut = silu(concatenated(
                    (0 ..< S).map { position in
                        conv1d(convInput[
                            0..., position ..< (position + convKernelSize), 0...])
                    }, axis: 1))
            } else {
                convOut = silu(conv1d(convInput))
            }

            let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
            let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
            v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

            let dtype = q.dtype
            let scales = derived.normScales(headKDim: headKDim, dtype: dtype)
            qNormed = MLXFast.rmsNorm(q, weight: scales.q, eps: 1e-6)
            kNormed = MLXFast.rmsNorm(k, weight: scales.k, eps: 1e-6)
            convOutBackingAll = convOut
        }

        let out: MLXArray
        if S >= 3 {
            // The prework kernel's gates, kept for the accepted-prefix replay.
            let windowGates: (g: MLXArray, beta: MLXArray)? = pre.map { (g: $0.g, beta: $0.beta) }
            let recurrence: (MLXArray, MLXArray)
            if let pre {
                recurrence =
                    Qwen35GatedDeltaChunked.runVerify(
                        q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta, state: ssmState)
                    ?? Qwen35GatedDeltaV3.run(
                        q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta, state: ssmState)
                    ?? gatedDeltaKernel(
                        q: pre.q, k: pre.k, v: pre.v, g: pre.g, beta: pre.beta, state: ssmState,
                        mask: nil)
            } else {
                recurrence = qwen35GatedDelta(
                    q: qNormed,
                    k: kNormed,
                    v: v,
                    a: a,
                    b: b,
                    aLog: aLog,
                    dtBias: dtBias,
                    state: ssmState,
                    mask: nil)
            }
            out = recurrence.0
            let finalSsmState = recurrence.1

            for (row, evaluation) in recurrentState.enumerated() {
                let rowRange = row ..< (row + 1)
                let finalConv = convInput[rowRange, S ..< (S + nKeep), 0...]
                let finalSSM = finalSsmState[rowRange]
                let tape = ArraysCache.PrefixReplayTape(
                    convInput: convInput[rowRange],
                    q: qNormed[rowRange],
                    k: kNormed[rowRange],
                    v: v[rowRange],
                    a: a[rowRange],
                    b: b[rowRange],
                    ssmPre: ssmState[rowRange],
                    mask: nil,
                    rowCount: S,
                    convStateRows: nKeep)
                // Count unique additional buffers retained by this stage.
                // `finalConv` aliases `convInput` until full acceptance detaches
                // its exact tail at commit. A one-row `ssmPre` aliases the
                // already-accounted committed generation when it existed before
                // this verify. `v` retains the whole conv output backing.
                let convOutBacking = convOutBackingAll[rowRange]
                var roots = [
                    tape.convInput, tape.q, tape.k, convOutBacking, tape.a, tape.b,
                ]
                let inputSSM =
                    evaluation.inputState(modelLayerIndex: modelLayerIndex)?.ssm
                if B > 1 || inputSSM == nil {
                    if let ssmPre = tape.ssmPre { roots.append(ssmPre) }
                }
                var strictReplayRoots = roots
                if B == 1, let inputSSM {
                    // The pending baseline already charges this state. After
                    // strict acceptance it becomes an additional replay input.
                    strictReplayRoots.append(inputSSM)
                }

                func checkedByteCount(_ arrays: [MLXArray]) -> Int {
                    var total = 0
                    for array in arrays {
                        let (bytes, multiplyOverflow) =
                            array.size.multipliedReportingOverflow(by: array.dtype.size)
                        let (sum, addOverflow) = total.addingReportingOverflow(bytes)
                        guard !multiplyOverflow, !addOverflow else {
                            preconditionFailure(
                                "Qwen35 compact recurrent replay byte accounting overflow")
                        }
                        total = sum
                    }
                    return total
                }
                let materializedBytes = checkedByteCount(roots + [finalSSM])
                let strictReplayRetainedBytes = checkedByteCount(strictReplayRoots)
                let fullAcceptanceRetainedBytes = checkedByteCount([tape.convInput])
                do {
                    try evaluation.stagePrefixReplay(
                        modelLayerIndex: modelLayerIndex,
                        positions: S,
                        finalConv: finalConv,
                        finalSSM: finalSSM,
                        materializedByteCount: materializedBytes,
                        evaluationRoots: strictReplayRoots,
                        strictReplayRetainedByteCount: strictReplayRetainedBytes,
                        strictReplayRetainedRoots: strictReplayRoots,
                        fullAcceptanceRetainedByteCount: fullAcceptanceRetainedBytes,
                        fullAcceptanceRetainedRoots: [tape.convInput],
                        fullAcceptance: {
                            let detachedConv = finalConv + MLXArray.zeros(
                                finalConv.shape, dtype: finalConv.dtype)
                            return CBv2RecurrentLayerState(
                                conv: detachedConv, ssm: finalSSM)
                        },
                        replay: { [unowned self] keepPositions in
                            self.replayedPrefixState(
                                tape: tape, committedRows: keepPositions,
                                gates: windowGates.map {
                                    (g: $0.g[rowRange], beta: $0.beta[rowRange])
                                })
                        })
                } catch {
                    preconditionFailure(
                        "Qwen35 CBv2 compact replay stage failed at layer "
                            + "\(modelLayerIndex): \(error)")
                }
            }
        } else {
            var outs: [MLXArray] = []
            var ssmStates: [MLXArray] = []
            outs.reserveCapacity(S)
            ssmStates.reserveCapacity(S)
            var state = ssmState
            for s in 0 ..< S {
                let (stepOut, next) = gatedDeltaUpdate(
                    q: qNormed[0..., s ..< (s + 1)],
                    k: kNormed[0..., s ..< (s + 1)],
                    v: v[0..., s ..< (s + 1)],
                    a: a[0..., s ..< (s + 1)],
                    b: b[0..., s ..< (s + 1)],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: state,
                    mask: nil)
                outs.append(stepOut)
                ssmStates.append(next)
                state = next
            }

            for (row, evaluation) in recurrentState.enumerated() {
                let convStack = concatenated(
                    (0 ..< S).map { s in
                        convInput[row ..< (row + 1), (s + 1) ..< (s + 1 + nKeep)]
                    }, axis: 0)
                let ssmStack = concatenated(
                    ssmStates.map { $0[row ..< (row + 1)] }, axis: 0)
                do {
                    try evaluation.stageCaptured(
                        modelLayerIndex: modelLayerIndex,
                        conv: convStack, ssm: ssmStack, positions: S)
                } catch {
                    preconditionFailure(
                        "Qwen35 CBv2 captured stage failed at layer "
                            + "\(modelLayerIndex): \(error)")
                }
            }
            out = outs.count == 1 ? outs[0] : concatenated(outs, axis: 1)
        }
        if exactTargetVerify {
            return qwen35A3BExactW4G64Projection(
                outProj, gatedNorm(out, gate: z).reshaped(B, S, -1))
        }
        return projectGatedOut(out, gate: z, B: B, S: S)
    }
}

// MARK: - Attention

final class Qwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer
    let mrope: Qwen35MRoPE
    /// The plain RoPE's parameters, for the fused q/k prep (nil when the
    /// configured RoPE is not the plain scale-1 `RoPE`).
    private let plainRope: (dims: Int, base: Float)?

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )
        self.mrope = Qwen35MRoPE(
            rope: self.rope, dim: max(1, ropeDims), base: args.ropeTheta,
            scalingConfig: args.ropeScaling,
            sections: args.mropeSection)
        let ropeType: String = {
            if let config = args.ropeScaling,
                let value = config["type"] ?? config["rope_type"], case .string(let s) = value
            {
                return s
            }
            return "default"
        }()
        self.plainRope =
            (self.rope is RoPE && ropeType == "default" && ropeDims >= 2 && ropeDims % 2 == 0
                && ropeDims <= headDim)
            ? (dims: ropeDims, base: args.ropeTheta) : nil

        super.init()
    }

    /// q, k and v read the same activation. On a packed Hadamard checkpoint
    /// they share one input transform, so it is computed once.
    /// The packed q, k and v projections when they share one input transform;
    /// nil otherwise.
    var inputRotationSiblings: [HadamardQuantizedLinear]? {
        sharedHadamardSiblings([qProj, kProj, vProj])
    }

    /// `preRotated`, when given, is `x` already rotated by the shared transform
    /// and stored in the dtype the q|k|v stack reads.
    private func projectQKV(
        _ x: MLXArray, preRotated: MLXArray? = nil,
        preQuantized: SignedBlockHadamard.Int8Activation? = nil
    ) -> (
        MLXArray, MLXArray, MLXArray
    ) {
        // The prompt-width tensor route on the boundary kernel's codes.
        if let preQuantized, let siblings = inputRotationSiblings,
            let shared = siblings[0].tensorRouteForwardQuantized(
                preQuantized, siblings: siblings, leading: Array(x.shape.dropLast()),
                widenOutput: true),
            shared.count == 3
        {
            return (shared[0], shared[1], shared[2])
        }
        if let preRotated, let siblings = inputRotationSiblings,
            let (wide, boundaries) = sharedHadamardStackOnRotated(preRotated, siblings),
            boundaries.count == 3
        {
            let parts = MLX.split(wide, indices: Array(boundaries.dropLast()), axis: -1)
            return (parts[0], parts[1], parts[2])
        }
        if let shared = sharedHadamardProjections(x, [qProj, kProj, vProj]) {
            return (shared[0], shared[1], shared[2])
        }
        return (qProj(x), kProj(x), vProj(x))
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let (qProjOutput, kProjOutput, vProjOutput) = projectQKV(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProjOutput
        var values = vProjOutput

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(sigmoidMultiply(output, gate))
    }

    func cbv2Forward(
        _ x: MLXArray, cache: any CBv2AttendingLayerCache,
        positionIds: MLXArray? = nil,
        exactTargetVerify: Bool = false,
        lastQueryOnly: Bool = false,
        preRotated: MLXArray? = nil,
        preQuantized: SignedBlockHadamard.Int8Activation? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        // Final-layer prompt narrowing: commit every position's K/V, attend
        // only the newest query (see LastQueryPrefillV2: the newest causal
        // query sees every key the chunk wrote, so its row is the same
        // attention the full chunk would compute for it), and project only
        // that row. Returns [B, 1, hidden].
        let narrowsToLastQuery =
            lastQueryOnly && L > 1 && positionIds == nil && !exactTargetVerify
            && cache is any CBv2LastQueryPrefillLayerCache

        let projected: (MLXArray, MLXArray, MLXArray)
        if exactTargetVerify {
            projected = (
                qwen35A3BExactW4G64Projection(qProj, x),
                qwen35A3BExactW4G64Projection(kProj, x),
                qwen35A3BExactW4G64Projection(vProj, x)
            )
        } else {
            projected = projectQKV(x, preRotated: preRotated, preQuantized: preQuantized)
        }
        let qProjOutput = projected.0
        let kProjection = projected.1
        let vProjection = projected.2
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        let gate = qSplit[1].reshaped(B, L, -1)
        let values = vProjection.reshaped(B, L, kvHeads, -1)
            .transposed(0, 2, 1, 3)
        var queries: MLXArray
        var keys: MLXArray
        if positionIds == nil, !exactTargetVerify, let plainRope,
            ObjectIdentifier(type(of: qNorm)) == ObjectIdentifier(RMSNorm.self),
            ObjectIdentifier(type(of: kNorm)) == ObjectIdentifier(RMSNorm.self),
            let offsets = Optional(cache.positionOffsets + 0),
            let q = Qwen35QKPrep.run(
                qSplit[0], weight: qNorm.weight, eps: qNorm.eps, offsets: offsets,
                ropeDims: plainRope.dims, ropeBase: plainRope.base),
            let k = Qwen35QKPrep.run(
                kProjection.reshaped(B, L, kvHeads, -1), weight: kNorm.weight, eps: kNorm.eps,
                offsets: offsets, ropeDims: plainRope.dims, ropeBase: plainRope.base)
        {
            // Norm, head transpose and RoPE in one launch per tensor, reading
            // the q|gate and k views through their strides.
            queries = q
            keys = k
        } else {
            queries = qNorm(qSplit[0]).transposed(0, 2, 1, 3)
            keys = kNorm(kProjection.reshaped(B, L, kvHeads, -1))
                .transposed(0, 2, 1, 3)

            // Text-only Qwen positions are ordinary scalar-equivalent positions,
            // but histories differ across rows. Capture the per-row device offsets
            // before the cache advances and use the array RoPE overload.
            if let positionIds {
                (queries, keys) = mrope.apply(
                    queries: queries, keys: keys, positionIds: positionIds)
            } else {
                let offsets = cache.positionOffsets + 0
                queries = rope(queries, offset: offsets)
                keys = rope(keys, offset: offsets)
            }
        }

        let output: MLXArray
        let attendedGate: MLXArray
        if narrowsToLastQuery, let lastQuery = cache as? any CBv2LastQueryPrefillLayerCache {
            output = lastQuery.updateAndAttendLastQuery(
                queries: queries[0..., 0..., (L - 1)..., 0...], keys: keys, values: values,
                scale: scale, sinks: nil)
                .transposed(0, 2, 1, 3)
                .reshaped(B, 1, -1)
            attendedGate = gate[0..., (L - 1)..., 0...]
        } else {
            let attended = cache.updateAndAttend(
                queries: queries, keys: keys, values: values,
                scale: scale, sinks: nil)
                .transposed(0, 2, 1, 3)
            // The gate, the signs, the transform and the route dtype's rounding
            // read the head-transposed output and the gate half of the q|gate
            // projection through their strides: neither is reshaped into a copy.
            if !exactTargetVerify, let packed = oProj as? HadamardQuantizedLinear,
                let y = packed.applyAfterSigmoidGateHeads(
                    attended, gate: qSplit[1], widenOutput: false)
            {
                return y
            }
            output = attended.reshaped(B, L, -1)
            attendedGate = gate
        }
        if exactTargetVerify {
            return qwen35A3BExactW4G64Projection(oProj, sigmoidMultiply(output, attendedGate))
        }
        if let packed = oProj as? HadamardQuantizedLinear {
            // The gate product, the signs, the transform and the route
            // dtype's rounding in one kernel (ercumentyildirim, `ade7529`).
            if let y = packed.applyAfterSigmoidGate(
                output, gate: attendedGate, widenOutput: false)
            {
                return y
            }
            // The output gate and the projection's Hadamard signs share one
            // kernel; the residual add widens the FP16 product itself.
            if Qwen35FusedElementwise.foldsHadamardSigns, packed.gdnLayout == nil,
                output.dtype == attendedGate.dtype
            {
                let signed = Qwen35FusedElementwise.sigmoidGateSigned(
                    output, attendedGate, packed.transform.signVector)
                return packed.forwardPreSigned(signed, widenOutput: false)
            }
            return packed.forwardUnwidened(sigmoidMultiply(output, attendedGate))
        }
        return oProj(sigmoidMultiply(output, attendedGate))
    }
}

/// Qwen3.5 interleaved 3-axis M-RoPE. Request positions arrive as function
/// inputs; the module retains configuration only.
final class Qwen35MRoPE {
    private let rope: RoPELayer
    private let rotaryDim: Int
    private let defaultInvFreq: MLXArray?
    private let sections: [Int]
    // Which of the three position planes (t/h/w) owns each frequency,
    // precomputed as [1, 1, 1, rotaryDim/2] so the default path interleaves
    // with one takeAlong instead of a per-frequency slice loop.
    private let mropeIndices: MLXArray

    init(
        rope: RoPELayer, dim: Int, base: Float,
        scalingConfig: [String: StringOrNumber]?, sections: [Int]
    ) {
        self.rope = rope
        self.rotaryDim = max(1, dim)
        let ropeType: String = {
            if let value = scalingConfig?["type"] ?? scalingConfig?["rope_type"],
                case .string(let type) = value
            {
                return type
            }
            return "default"
        }()
        if ropeType == "default" || ropeType == "mrope" {
            let exponents = MLXArray(stride(from: 0, to: self.rotaryDim, by: 2))
                .asType(.float32) / Float(self.rotaryDim)
            self.defaultInvFreq = 1 / pow(MLXArray(base), exponents)
        } else {
            self.defaultInvFreq = nil
        }
        let resolvedSections = sections.count >= 3 ? sections : [11, 11, 10]
        self.sections = resolvedSections
        let frequencyCount = max(1, self.rotaryDim / 2)
        var indices = [Int32](repeating: 0, count: frequencyCount)
        for (dimension, offset) in [(1, 1), (2, 2)] {
            let end = min(resolvedSections[dimension] * 3, frequencyCount)
            for index in stride(from: offset, to: end, by: 3) {
                indices[index] = Int32(dimension)
            }
        }
        self.mropeIndices = MLXArray(indices).reshaped(1, 1, 1, -1)
    }

    private func axis(forFrequency index: Int, frequencyCount: Int) -> Int {
        for (axis, offset) in [(1, 1), (2, 2)] {
            let length = min(sections[axis] * 3, frequencyCount)
            if index >= offset && index < length && (index - offset) % 3 == 0 {
                return axis
            }
        }
        return 0
    }

    func apply(
        queries: MLXArray, keys: MLXArray, positionIds: MLXArray
    ) -> (MLXArray, MLXArray) {
        var positions = positionIds
        if positions.ndim == 2 {
            positions = broadcast(
                positions[.newAxis, 0..., 0...],
                to: [3, positions.dim(0), positions.dim(1)])
        }
        precondition(positions.ndim == 3 && positions.dim(0) == 3)
        precondition(rotaryDim % 2 == 0 && rotaryDim <= queries.dim(-1))

        if let defaultInvFreq {
            let all = positions.asType(.float32)[0..., 0..., 0..., .newAxis]
                * defaultInvFreq[.newAxis, .newAxis, .newAxis, 0...]
            let frequency = takeAlong(all, mropeIndices, axis: 0).squeezed(axis: 0)
            let angles = concatenated([frequency, frequency], axis: -1)
            let cosine = cos(angles).asType(queries.dtype).expandedDimensions(axis: 1)
            let sine = sin(angles).asType(queries.dtype).expandedDimensions(axis: 1)
            func applyDefault(_ value: MLXArray) -> MLXArray {
                let rotating = value[.ellipsis, ..<rotaryDim]
                let half = rotating.dim(-1) / 2
                let rotatedHalf = concatenated(
                    [-rotating[.ellipsis, half...], rotating[.ellipsis, ..<half]], axis: -1)
                let rotated = rotating * cosine + rotatedHalf * sine
                return rotaryDim < value.dim(-1)
                    ? concatenated([rotated, value[.ellipsis, rotaryDim...]], axis: -1)
                    : rotated
            }
            return (applyDefault(queries), applyDefault(keys))
        }

        let queryHeads = queries.dim(1)
        let combined = concatenated([queries, keys], axis: 1)
        let batch = combined.dim(0)
        let heads = combined.dim(1)
        let length = combined.dim(2)
        let headDim = combined.dim(3)
        let flattened = combined.transposed(0, 2, 1, 3)
            .reshaped([batch * length, heads, 1, headDim])
        let byAxis = (0 ..< 3).map { axis in
            rope(flattened, offset: positions[axis, 0..., 0...].flattened())
                .reshaped([batch, length, heads, headDim])
                .transposed(0, 2, 1, 3)
        }
        let frequencyCount = rotaryDim / 2
        var firstHalf: [MLXArray] = []
        var secondHalf: [MLXArray] = []
        firstHalf.reserveCapacity(frequencyCount)
        secondHalf.reserveCapacity(frequencyCount)
        for index in 0 ..< frequencyCount {
            let rotated = byAxis[axis(forFrequency: index, frequencyCount: frequencyCount)]
            firstHalf.append(rotated[.ellipsis, index ..< index + 1])
            secondHalf.append(
                rotated[.ellipsis, frequencyCount + index ..< frequencyCount + index + 1])
        }
        var result = concatenated(firstHalf + secondHalf, axis: -1)
        if rotaryDim < combined.dim(-1) {
            result = concatenated([result, combined[.ellipsis, rotaryDim...]], axis: -1)
        }
        return (
            result[0..., ..<queryHeads, 0..., 0...],
            result[0..., queryHeads..., 0..., 0...])
    }
}

func qwen35FlattenMoEInputs(
    x: MLXArray, indices: MLXArray, scores: MLXArray
) -> (x: MLXArray, indices: MLXArray, scores: MLXArray) {
    precondition(x.ndim >= 2 && indices.ndim >= 2 && scores.shape == indices.shape)
    return (
        x.reshaped([-1, x.dim(-1)]),
        indices.reshaped([-1, indices.dim(-1)]),
        scores.reshaped([-1, scores.dim(-1)])
    )
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int
    private let routerFinalizer: Qwen35A3BRouterFinalizer

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    /// - Parameter fuseGateUp: when true the routed experts use one fused
    ///   `gate_up_proj` SwitchLinear (one gather_qmm serves gate+up; the
    ///   sanitizers concatenate split checkpoints into the fused layout).
    ///   The inline MTP head passes false: its weights load through
    ///   `Qwen35InlineMTPAssistant` whose per-path quantization table is
    ///   keyed on the split `gate_proj`/`up_proj` module paths.
    init(_ args: Qwen35TextConfiguration, fuseGateUp: Bool = true) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok
        self.routerFinalizer = qwen35A3BRouterFinalizer(
            hidden: args.hiddenSize, experts: args.numExperts,
            topK: args.numExpertsPerTok, normalize: args.normTopkProb)

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts,
            fuseGateUp: fuseGateUp,
            weightedReductionProfile: .qwen35ProductionSwiGLU
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, exactTargetVerify: false)
    }

    func callAsFunction(
        _ x: MLXArray, exactTargetVerify: Bool
    ) -> MLXArray {
        var gates = exactTargetVerify
            ? qwen35A3BExactTimewiseProjection(gate, x) : gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let (inds, scores) = routerFinalizer(gates)

        let tokenShape = x.shape
        let flattened = qwen35FlattenMoEInputs(x: x, indices: inds, scores: scores)
        let flatX = flattened.x
        let flatIndices = flattened.indices
        let flatScores = flattened.scores
        let combined = switchMLP.callAndWeightedReduce(
            flatX, flatIndices, weights: flatScores.asType(x.dtype),
            fuseSortedReduction: true, isProductionPrefill: true
        ).reshaped(tokenShape)

        var sharedY = sharedExpert.qwen35TargetVerify(
            x, exact: exactTargetVerify)
        let sharedGate = exactTargetVerify
            ? qwen35A3BExactTimewiseProjection(sharedExpertGate, x)
            : sharedExpertGate(x)
        sharedY = sigmoid(sharedGate) * sharedY

        return combined + sharedY
    }
}

extension Qwen3NextMLP {
    func qwen35TargetVerify(_ x: MLXArray, exact: Bool) -> MLXArray {
        guard exact else { return qwen35Forward(x) }
        let (gate, up) = qwen35A3BExactW4G64ProjectionPair(
            gateProj, upProj, x)
        return qwen35A3BExactW4G64Projection(downProj, silu(gate) * up)
    }

    /// `callAsFunction` with gate and up sharing one packed input transform.
    ///
    /// On a packed down projection the tail is one fused kernel: the gate and
    /// up products stay FP16, `silu(gate) * up` is formed in FP32 together with
    /// the down projection's Hadamard signs, and the down product is left for
    /// the residual add to widen. Same arithmetic, four fewer dispatches.
    func qwen35Forward(_ x: MLXArray) -> MLXArray {
        if let down = downProj as? HadamardQuantizedLinear, down.gdnLayout == nil,
            let shared = sharedHadamardProjections(x, [gateProj, upProj], widenOutput: false)
        {
            if let y = down.applyAfterSwiGLU(
                gate: shared[0], up: shared[1], widenOutput: false)
            {
                return y
            }
            let signed = Qwen35FusedElementwise.swigluSigned(
                shared[0], shared[1], down.transform.signVector)
            return down.forwardPreSigned(signed, widenOutput: false)
        }
        guard let shared = sharedHadamardProjections(x, [gateProj, upProj]) else {
            return self(x)
        }
        return downProj(silu(shared[0]) * shared[1])
    }

    /// `qwen35ForwardSignedNorm(x + r, ...)` with the residual add, the norm and
    /// the gate|up rotation in one kernel (`SignedBlockHadamard.residualNormRotated`
    /// computes `h = x + r`, `rmsNorm(h)`, the signs and the transform, storing
    /// the dtype the gate|up stack reads). Returns `h` and the MLP output, the
    /// same values as the composed path. Nil when it does not apply.
    fileprivate func qwen35ForwardResidualNorm(
        _ x: MLXArray, _ r: MLXArray, norm: RMSNorm
    ) -> (h: MLXArray, out: MLXArray)? {
        if let quantized = qwen35ForwardResidualNormQuantized(x, r, norm: norm) {
            return quantized
        }
        guard ObjectIdentifier(type(of: norm)) == ObjectIdentifier(RMSNorm.self),
            let down = downProj as? HadamardQuantizedLinear, down.gdnLayout == nil,
            let siblings = sharedHadamardSiblings([gateProj, upProj]),
            let transform = siblings.first?.transform,
            norm.weight.ndim == 1, norm.weight.dim(0) == transform.width,
            x.ndim >= 2, x.dim(-1) == transform.width,
            let readDType = sharedHadamardStackReadDType(siblings, rows: x.size / transform.width),
            let (h, _, rotated) = transform.residualNormRotated(
                x, r, weight: norm.weight, eps: norm.eps, outputDType: readDType),
            let (wide, boundaries) = sharedHadamardStackOnRotated(
                rotated, siblings, widenOutput: false),
            boundaries.count == 2, boundaries[1] == 2 * boundaries[0],
            boundaries[0] == down.transform.width,
            let y = down.applyAfterSwiGLUStacked(
                wide, gateOffset: 0, upOffset: boundaries[0], widenOutput: false)
        else { return nil }
        return (h, y)
    }

    /// `qwen35ForwardResidualNorm` where the prompt-width tensor route takes
    /// the gate|up stack: the boundary kernel quantizes the rotated norm
    /// output for the route (`residualNormQuantized`) instead of the route's
    /// own rotation re-reading a stored norm output, the route's matmul reads
    /// those codes, and the down projection's SwiGLU producer reads the gate
    /// and up halves of its FP16 product. The composed path's values.
    fileprivate func qwen35ForwardResidualNormQuantized(
        _ x: MLXArray, _ r: MLXArray, norm: RMSNorm
    ) -> (h: MLXArray, out: MLXArray)? {
        guard ObjectIdentifier(type(of: norm)) == ObjectIdentifier(RMSNorm.self),
            let down = downProj as? HadamardQuantizedLinear, down.gdnLayout == nil,
            let siblings = sharedHadamardSiblings([gateProj, upProj]), siblings.count == 2,
            let transform = siblings.first?.transform,
            norm.weight.ndim == 1, norm.weight.dim(0) == transform.width,
            x.ndim >= 2, x.dim(-1) == transform.width,
            siblings[0].tensorRouteTakesPrompt(siblings, rows: x.size / transform.width),
            let (h, _, activation) = transform.residualNormQuantized(
                x, r, weight: norm.weight, eps: norm.eps,
                permuted: Qwen35TensorPackedMatmul.support == .staged8),
            let shared = siblings[0].tensorRouteForwardQuantized(
                activation, siblings: siblings, leading: Array(x.shape.dropLast()),
                widenOutput: false),
            shared.count == 2
        else { return nil }
        if let y = down.applyAfterSwiGLU(gate: shared[0], up: shared[1], widenOutput: false) {
            return (h, y)
        }
        let signed = Qwen35FusedElementwise.swigluSigned(
            shared[0], shared[1], down.transform.signVector)
        return (h, down.forwardPreSigned(signed, widenOutput: false))
    }

    /// `qwen35Forward(norm(h))` with gate and up's Hadamard signs folded into
    /// the norm's gain, so their shared rotation skips its sign multiply. The
    /// signed gain yields the plain norm's output times the signs exactly (see
    /// `Qwen35SignedGain`). Nil when the fold does not apply.
    fileprivate func qwen35ForwardSignedNorm(
        _ h: MLXArray, norm: RMSNorm, gain: Qwen35SignedGain
    ) -> MLXArray? {
        guard Qwen35FusedElementwise.foldsHadamardSigns,
            ObjectIdentifier(type(of: norm)) == ObjectIdentifier(RMSNorm.self),
            let down = downProj as? HadamardQuantizedLinear, down.gdnLayout == nil,
            let siblings = sharedHadamardSiblings([gateProj, upProj]),
            let transform = siblings.first?.transform,
            norm.weight.ndim == 1, norm.weight.dim(0) == transform.width
        else { return nil }
        let signedInput = MLXFast.rmsNorm(
            h, weight: gain.gain(norm.weight, signs: transform.signVector), eps: norm.eps)
        // The gate|up stack's product goes to the down projection's SwiGLU
        // rotation unsplit: the kernel reads both halves at their columns, so
        // neither half is split off into a strided view and copied.
        if let (wide, boundaries) = sharedHadamardProjectionsPreSignedWide(
            signedInput, siblings, widenOutput: false),
            boundaries.count == 2, boundaries[1] == 2 * boundaries[0],
            boundaries[0] == down.transform.width,
            let y = down.applyAfterSwiGLUStacked(
                wide, gateOffset: 0, upOffset: boundaries[0], widenOutput: false)
        {
            return y
        }
        guard
            let shared = sharedHadamardProjectionsPreSigned(
                signedInput, siblings, widenOutput: false)
        else { return nil }
        if let y = down.applyAfterSwiGLU(gate: shared[0], up: shared[1], widenOutput: false) {
            return y
        }
        let signed = Qwen35FusedElementwise.swigluSigned(
            shared[0], shared[1], down.transform.signVector)
        return down.forwardPreSigned(signed, widenOutput: false)
    }
}

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    /// The post-attention gain with the MLP's gate/up signs folded in.
    private let signedGain = Qwen35SignedGain()

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        defer { signedGain.clear() }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    @discardableResult
    override func update(
        modules: ModuleChildren, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        defer { signedGain.clear() }
        return try super.update(modules: modules, verify: verify, path: path, modulePath: modulePath)
    }

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
        nConfirmed: Int = 0
    ) -> MLXArray {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py DecoderLayer.__call__
        // Passes nConfirmed through to the linear-attention sublayer.
        let r: MLXArray
        if isLinear {
            r = linearAttn!(
                inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                nConfirmed: nConfirmed)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        if let dense = mlp as? Qwen3NextMLP {
            return h + dense.qwen35Forward(postAttentionLayerNorm(h))
        }
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }

    func cbv2Forward(
        _ x: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil,
        captureRecurrentWindow: Bool = false,
        exactTargetVerify: Bool = false,
        lastRowOnly: Bool = false
    ) -> MLXArray {
        // A full-attention layer whose output is read at the last position
        // only (the final layer of a prompt-width forward): attend and
        // project that row alone. Returns [B, 1, hidden]; recurrent layers
        // and every other case keep the full width.
        if lastRowOnly, !isLinear, x.dim(1) > 1, positionIds == nil, !exactTargetVerify,
            let attentionCache, attentionCache is any CBv2LastQueryPrefillLayerCache
        {
            let r = selfAttn!.cbv2Forward(
                inputLayerNorm(x), cache: attentionCache, positionIds: nil,
                exactTargetVerify: false, lastQueryOnly: true)
            let last = x.dim(1) - 1
            let h = x[0..., last..., 0...] + r
            let normalized = postAttentionLayerNorm(h)
            let feedForward: MLXArray
            if let sparse = mlp as? Qwen35SparseMoeBlock {
                feedForward = sparse(normalized, exactTargetVerify: false)
            } else if let dense = mlp as? Qwen3NextMLP {
                feedForward = dense.qwen35TargetVerify(normalized, exact: false)
            } else {
                preconditionFailure("Qwen35 decoder has an unsupported MLP module")
            }
            return h + feedForward
        }
        let r: MLXArray
        if isLinear {
            precondition(attentionCache == nil, "Qwen35 recurrent layer received attention KV")
            if captureRecurrentWindow {
                r = linearAttn!.cbv2ForwardCaptured(
                    inputLayerNorm(x), modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState,
                    exactTargetVerify: exactTargetVerify)
            } else {
                r = linearAttn!.cbv2Forward(
                    inputLayerNorm(x), modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState)
            }
        } else {
            guard let attentionCache else {
                preconditionFailure("Qwen35 full-attention layer is missing its CBv2 cache")
            }
            r = selfAttn!.cbv2Forward(
                inputLayerNorm(x), cache: attentionCache, positionIds: positionIds,
                exactTargetVerify: exactTargetVerify)
        }
        if !exactTargetVerify, let dense = mlp as? Qwen3NextMLP,
            let (h, feedForward) = dense.qwen35ForwardResidualNorm(
                x, r, norm: postAttentionLayerNorm)
        {
            return h + feedForward
        }
        let h = x + r
        let feedForward: MLXArray
        if let sparse = mlp as? Qwen35SparseMoeBlock {
            feedForward = sparse(
                postAttentionLayerNorm(h), exactTargetVerify: exactTargetVerify)
        } else if let dense = mlp as? Qwen3NextMLP {
            if !exactTargetVerify,
                let folded = dense.qwen35ForwardSignedNorm(
                    h, norm: postAttentionLayerNorm, gain: signedGain)
            {
                feedForward = folded
            } else {
                feedForward = dense.qwen35TargetVerify(
                    postAttentionLayerNorm(h), exact: exactTargetVerify)
            }
        } else {
            preconditionFailure("Qwen35 decoder has an unsupported MLP module")
        }
        return h + feedForward
    }

    /// The input-side layer boundary fusion (`BONSAI_FUSED_RESNORM_INPUT=0`
    /// materializes every residual add as before).
    static let pendingResidualEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_FUSED_RESNORM_INPUT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `x + pending` (the previous layer's last residual add), `inputLayerNorm`
    /// of the sum and the input projections' shared rotation in one kernel
    /// (`SignedBlockHadamard.residualNormRotated`): the sum, the FP32 norm
    /// output when the layer also reads it unrotated (the GDN b|a
    /// projections), and the rotation in the dtype the input stack reads.
    private func fusedInputBoundary(_ x: MLXArray, _ pending: MLXArray)
        -> (sum: MLXArray, normed: MLXArray?, rotated: MLXArray)?
    {
        let siblings =
            isLinear ? linearAttn?.inputRotationSiblings : selfAttn?.inputRotationSiblings
        guard ObjectIdentifier(type(of: inputLayerNorm)) == ObjectIdentifier(RMSNorm.self),
            let siblings, let transform = siblings.first?.transform,
            x.ndim >= 2, x.dim(-1) == transform.width,
            inputLayerNorm.weight.ndim == 1, inputLayerNorm.weight.dim(0) == transform.width,
            let readDType = sharedHadamardStackReadDType(
                siblings, rows: x.size / transform.width),
            let out = transform.residualNormRotated(
                x, pending, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                outputDType: readDType, keepNormed: isLinear)
        else { return nil }
        return (out.h, out.normed, out.rotated)
    }

    /// `fusedInputBoundary` where the prompt-width tensor route takes the
    /// layer's input stack: the kernel quantizes the rotated norm output for
    /// the route (`residualNormQuantized`) instead of storing the rotation.
    private func fusedInputBoundaryQuantized(_ x: MLXArray, _ pending: MLXArray)
        -> (sum: MLXArray, normed: MLXArray?, activation: SignedBlockHadamard.Int8Activation)?
    {
        let siblings =
            isLinear ? linearAttn?.inputRotationSiblings : selfAttn?.inputRotationSiblings
        guard ObjectIdentifier(type(of: inputLayerNorm)) == ObjectIdentifier(RMSNorm.self),
            let siblings, let transform = siblings.first?.transform,
            x.ndim >= 2, x.dim(-1) == transform.width,
            inputLayerNorm.weight.ndim == 1, inputLayerNorm.weight.dim(0) == transform.width,
            siblings[0].tensorRouteTakesPrompt(siblings, rows: x.size / transform.width),
            let out = transform.residualNormQuantized(
                x, pending, weight: inputLayerNorm.weight, eps: inputLayerNorm.eps,
                permuted: Qwen35TensorPackedMatmul.support == .staged8, keepNormed: isLinear)
        else { return nil }
        return (out.h, out.normed, out.activation)
    }

    /// `cbv2Forward(x + pending)` for a non-exact, full-width forward, with the
    /// pending residual add fused into this layer's input norm and rotation
    /// and this layer's own last residual add left pending. Returns the layer
    /// input (`x + pending`, as the kernel stored it) and the output as
    /// `h + f`; every value is the one `cbv2Forward` computes.
    func cbv2ForwardPending(
        _ x: MLXArray, pending: MLXArray?,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?,
        captureRecurrentWindow: Bool
    ) -> (input: MLXArray, h: MLXArray, f: MLXArray) {
        var input = x
        var normed: MLXArray? = nil
        var rotated: MLXArray? = nil
        var quantized: SignedBlockHadamard.Int8Activation? = nil
        if let pending {
            if let fused = fusedInputBoundaryQuantized(x, pending) {
                input = fused.sum
                normed = fused.normed
                quantized = fused.activation
            } else if let fused = fusedInputBoundary(x, pending) {
                input = fused.sum
                normed = fused.normed
                rotated = fused.rotated
            } else {
                input = x + pending
            }
        }
        // With a rotated input the attention takes only the norm's shape (the
        // node is never evaluated); the GDN's b|a read the kernel's norm.
        let layerInput = normed ?? inputLayerNorm(input)
        let r: MLXArray
        if isLinear {
            precondition(attentionCache == nil, "Qwen35 recurrent layer received attention KV")
            if captureRecurrentWindow {
                r = linearAttn!.cbv2ForwardCaptured(
                    layerInput, modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState, exactTargetVerify: false,
                    preRotated: rotated, preQuantized: quantized)
            } else {
                r = linearAttn!.cbv2Forward(
                    layerInput, modelLayerIndex: modelLayerIndex,
                    recurrentState: recurrentState, preRotated: rotated,
                    preQuantized: quantized)
            }
        } else {
            guard let attentionCache else {
                preconditionFailure("Qwen35 full-attention layer is missing its CBv2 cache")
            }
            r = selfAttn!.cbv2Forward(
                layerInput, cache: attentionCache, positionIds: positionIds,
                exactTargetVerify: false, preRotated: rotated, preQuantized: quantized)
        }
        if let dense = mlp as? Qwen3NextMLP,
            let (h, feedForward) = dense.qwen35ForwardResidualNorm(
                input, r, norm: postAttentionLayerNorm)
        {
            return (input, h, feedForward)
        }
        let h = input + r
        let feedForward: MLXArray
        if let sparse = mlp as? Qwen35SparseMoeBlock {
            feedForward = sparse(postAttentionLayerNorm(h), exactTargetVerify: false)
        } else if let dense = mlp as? Qwen3NextMLP {
            if let folded = dense.qwen35ForwardSignedNorm(
                h, norm: postAttentionLayerNorm, gain: signedGain)
            {
                feedForward = folded
            } else {
                feedForward = dense.qwen35TargetVerify(postAttentionLayerNorm(h), exact: false)
            }
        } else {
            preconditionFailure("Qwen35 decoder has an unsupported MLP module")
        }
        return (input, h, feedForward)
    }
}

// MARK: - Text Model

public class Qwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    /// Metadata-only inspection for packed variants whose normalizers promote
    /// activations. No tensor evaluation or serving arithmetic changes.
    var cbv2UniformLayerNormDType: DType? {
        guard let dtype = layers.first?.inputLayerNorm.weight.dtype,
            layers.allSatisfy({
                $0.inputLayerNorm.weight.dtype == dtype
                    && $0.postAttentionLayerNorm.weight.dtype == dtype
            })
        else { return nil }
        return dtype
    }

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int
    let exactTargetVerify: Bool

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1
        self.exactTargetVerify =
            Qwen35A3BConstructionContext.targetVerifyArithmetic == .exactM1

        super.init()
    }

    /// Returns the pre-norm hidden state from the final layer.
    ///
    /// The caller (`Qwen35TextModel`) applies `norm` and the LM head on top.
    /// This split returns logits and raw capture rows in one pass; the bound
    /// Qwen MTP adapter applies this same final norm at trusted-history ingress.
    ///
    /// Port of omlx commit 696d90a:
    ///   patches/mlx_lm_mtp/qwen35_model.py `_patch_qwen3_5_text_model`
    ///   (returns hidden_states before self.model.norm so TextModel can apply it)
    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache?]? = nil,
        nConfirmed: Int = 0
    ) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask,
                cache: cacheArray?[i], nConfirmed: nConfirmed)
        }

        // Return pre-norm hidden states. Norm is applied by Qwen35TextModel.
        return hiddenStates
    }

    /// Rebuild every recurrent layer at the same committed verify boundary.
    /// Eligibility is checked for every layer before persistent state changes.
    /// Attention caches are intentionally untouched; the caller trims their
    /// rejected suffix after recurrent replay succeeds.
    func replayRecurrentPrefix(
        cache: [KVCache?], committedRows: Int
    ) -> Bool {
        guard cache.count == layers.count, committedRows > 0 else {
            clearRecurrentPrefixReplay(cache: cache)
            return false
        }

        var foundRecurrentLayer = false
        for (i, layer) in layers.enumerated() where layer.isLinear {
            foundRecurrentLayer = true
            guard let mamba = cache[i] as? MambaCache,
                  let linear = layer.linearAttn,
                  linear.canReplayPrefix(
                    cache: mamba, committedRows: committedRows)
            else {
                clearRecurrentPrefixReplay(cache: cache)
                return false
            }
        }
        guard foundRecurrentLayer else {
            clearRecurrentPrefixReplay(cache: cache)
            return false
        }

        for (i, layer) in layers.enumerated() where layer.isLinear {
            guard let mamba = cache[i] as? MambaCache,
                  let linear = layer.linearAttn,
                  linear.replayPrefix(
                    cache: mamba, committedRows: committedRows)
            else {
                clearRecurrentPrefixReplay(cache: cache)
                return false
            }
        }
        return true
    }

    /// Discard verify-only recurrent data without changing committed cache
    /// state. Used after full acceptance and by fallback/reset paths.
    func clearRecurrentPrefixReplay(cache: [KVCache?]) {
        for case let arrays as ArraysCache in cache.compactMap({ $0 }) {
            arrays.clearMTPTransientState()
        }
    }

    /// The DFlash 2 layer tap, held in a plain box so `Module` reflection
    /// classifies it as `.other` and never walks into it. A bare `MLXArray?`
    /// stored here would be discovered as a PARAMETER; `DFlash2TapSlot`
    /// documents why that matters.
    let dFlash2Tap = DFlash2TapSlot()

    /// `DARKBLOOM_BONSAI_LAST_ROW_FINAL=0` keeps the final layer full width.
    static let lastRowNarrowingEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_LAST_ROW_FINAL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private func tapLayerIdsForNarrowing() -> [Int]? { dFlash2Tap.layerIds }

    /// Submission slices (`BONSAI_VERIFY_SLICES=0` / `BONSAI_PROMPT_SLICES=0`
    /// submit each forward as one graph). The host builds a 64-layer graph in
    /// a few milliseconds while the GPU waits for it: behind the drafter's
    /// block on a verify, from idle on a prompt. Committing the tower every
    /// sixteen layers (a prompt also after its fourth, as nothing runs ahead
    /// of it) lets the GPU run the front while the host builds the rest. Same
    /// kernels, same inputs, same order; only command-buffer boundaries move
    /// (Meganpark980320 `72b3b48`, DPZZxlz `5f72492`, DrCleverHans `db2d22e`).
    /// Verify slices (`BONSAI_VERIFY_SLICES=0` off). The host does NOT build
    /// the verify while the drafter runs: every op that reads a weight larger
    /// than MLX's per-command-buffer byte budget (40-50 MB) commits its own
    /// buffer, the drafter's block commits ~27, and `eval_impl` stalls the
    /// submitting thread in `wait_for_one` while more than 10 are in flight.
    /// The drafter's `asyncEval` therefore returns only when the GPU is ten
    /// buffers from its end, and without slices the GPU then idles while the
    /// host builds the whole 64-layer verify graph and its tape. A short first
    /// slice restarts the GPU after `verifySliceFirst` layers; later slices
    /// every `verifySliceEvery` layers keep it fed.
    ///
    /// Off by default (`BONSAI_VERIFY_SLICES=1` turns them on). On the ranked
    /// M5 Max box the verify round measured ~3 ms LONGER with verify and
    /// drafter slices than without, on polymorf's `5fbfa003` tree: 52.7 ms
    /// (this tree's v8, `ec9b754d`, slices 4/8) and 52.4 ms (i34-9
    /// `a255ab16`, slices every 2) against 49.4 ms (`5fbfa003`) and 49.5 ms
    /// (`1bcbdf0c`), neither of which slices. Prompt slices stay on: the
    /// same box measured the prompt forward shorter with them.
    static let verifySlices: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_VERIFY_SLICES"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()
    static let verifySliceFirst: Int = sliceCount("BONSAI_VERIFY_SLICE_FIRST", 4)
    static let verifySliceEvery: Int = sliceCount("BONSAI_VERIFY_SLICE_EVERY", 8)

    private static func sliceCount(_ name: String, _ fallback: Int) -> Int {
        let raw = ProcessInfo.processInfo.environment[name]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return raw.flatMap { Int($0) }.map { max(1, $0) } ?? fallback
    }
    static let promptSlices = sliceFlag("BONSAI_PROMPT_SLICES")

    private static func sliceFlag(_ name: String) -> Bool {
        let value = ProcessInfo.processInfo.environment[name]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }

    private func submitsSlice(
        after modelLayerIndex: Int, verify: Bool, width: Int
    ) -> Bool {
        guard modelLayerIndex + 1 < layers.count else { return false }
        if verify {
            guard Self.verifySlices else { return false }
            let done = modelLayerIndex + 1
            return done == Self.verifySliceFirst
                || (done > Self.verifySliceFirst
                    && (done - Self.verifySliceFirst) % Self.verifySliceEvery == 0)
        }
        return Self.promptSlices && width > 32
            && (modelLayerIndex == 3 || (modelLayerIndex + 1) % 16 == 0)
    }

    func cbv2Forward(
        _ inputs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil,
        captureRecurrentWindow: Bool = false,
        lastRowOnly: Bool = false
    ) -> MLXArray {
        precondition(
            caches.count == layers.filter({ !$0.isLinear }).count,
            "Qwen35 CBv2 requires only full-attention caches")
        // Prompt-width forwards whose consumer reads the last position only
        // let the FINAL layer attend and project that row alone; the tap
        // layers (all earlier) and every K/V write stay full width.
        let lastLayerIndex = layers.count - 1
        let narrowFinalLayer =
            lastRowOnly && Self.lastRowNarrowingEnabled && !captureRecurrentWindow
            && (tapLayerIdsForNarrowing().map { $0.allSatisfy { $0 < lastLayerIndex } } ?? true)
        let shapeCall = CBv2ForwardShapeObservation.isActive
            ? CBv2ForwardShapeObservation.beginTarget(liveBatchRows: inputs.dim(0), sequenceWidth: inputs.dim(1)) : nil
        defer { shapeCall?.end() }
        var hiddenStates = inputEmbeddings ?? embedTokens(inputs)
        // Read the tap ONCE. A nil list costs one comparison per layer and
        // allocates nothing; the drafter is not attached on a serial leg.
        let tapLayerIds = dFlash2Tap.layerIds
        var tapped = [MLXArray?](
            repeating: nil, count: tapLayerIds?.count ?? 0)
        var attentionIndex = 0
        // Non-exact forwards leave each layer's last residual add pending and
        // fuse it into the next layer's input norm and rotation. A tap reads
        // the materialized sum (the next layer's kernel stores it).
        let pendingPath =
            Qwen35DecoderLayer.pendingResidualEnabled
            && !(captureRecurrentWindow && exactTargetVerify)
        var pending: MLXArray? = nil
        var pendingTapSlot: Int? = nil
        for (modelLayerIndex, layer) in layers.enumerated() {
            let attentionCache: (any CBv2AttendingLayerCache)?
            if layer.isLinear {
                attentionCache = nil
            } else {
                attentionCache = caches[attentionIndex]
                precondition(
                    attentionCache!.kind.modelLayerIndex == nil
                        || attentionCache!.kind.modelLayerIndex == modelLayerIndex,
                    "Qwen35 CBv2 attention cache mapped to the wrong model layer")
                attentionIndex += 1
            }
            let narrowsThisLayer = narrowFinalLayer && modelLayerIndex == lastLayerIndex
            if pendingPath && !narrowsThisLayer {
                let out = layer.cbv2ForwardPending(
                    hiddenStates, pending: pending,
                    modelLayerIndex: modelLayerIndex,
                    attentionCache: attentionCache,
                    recurrentState: recurrentState,
                    positionIds: positionIds,
                    captureRecurrentWindow: captureRecurrentWindow)
                if let slot = pendingTapSlot {
                    tapped[slot] = out.input
                    pendingTapSlot = nil
                }
                hiddenStates = out.h
                pending = out.f
                if let tapLayerIds, let slot = tapLayerIds.firstIndex(of: modelLayerIndex) {
                    pendingTapSlot = slot
                }
                if submitsSlice(
                    after: modelLayerIndex, verify: captureRecurrentWindow,
                    width: hiddenStates.dim(1))
                {
                    asyncEval([out.h, out.f])
                }
                continue
            }
            if let p = pending {
                hiddenStates = hiddenStates + p
                pending = nil
            }
            if let slot = pendingTapSlot {
                tapped[slot] = hiddenStates
                pendingTapSlot = nil
            }
            hiddenStates = layer.cbv2Forward(
                hiddenStates,
                modelLayerIndex: modelLayerIndex,
                attentionCache: attentionCache,
                recurrentState: recurrentState,
                positionIds: positionIds,
                captureRecurrentWindow: captureRecurrentWindow,
                exactTargetVerify: captureRecurrentWindow && exactTargetVerify,
                lastRowOnly: narrowFinalLayer && modelLayerIndex == lastLayerIndex)
            // `hiddenStates` here IS the OUTPUT hidden state of this layer,
            // which is what the reference taps (`_LayerHook` wraps the layer and
            // keeps what it returned).
            if let tapLayerIds, let slot = tapLayerIds.firstIndex(of: modelLayerIndex) {
                tapped[slot] = hiddenStates
            }
            if submitsSlice(
                after: modelLayerIndex, verify: captureRecurrentWindow,
                width: hiddenStates.dim(1))
            {
                asyncEval([hiddenStates])
            }
        }
        if let p = pending {
            hiddenStates = hiddenStates + p
        }
        if let slot = pendingTapSlot {
            tapped[slot] = hiddenStates
        }
        if tapLayerIds == nil {
            dFlash2Tap.tappedHidden = nil
        } else {
            dFlash2Tap.tappedHidden = concatenated(tapped.map { $0! }, axis: -1)
        }
        return hiddenStates
    }
}



// MARK: - Fused attention q/k prep

/// The attention's per-head q/k RMSNorm, head transpose and partial RoPE in
/// one launch per tensor. The op chain ran a row-contiguous copy of the
/// strided q (k) view, `rms_single_row`, a copy for the RoPE's untouched
/// dims and the in-place `rope` kernel: four launches per tensor. Here one
/// 64-thread threadgroup per (batch, position, head) row keeps every
/// operation's arithmetic: `rms_single_row`'s 64 threads x 4 reads, `acc +=
/// xi * xi`, `simd_sum`, the 32-slot pass, `precise::rsqrt(acc / D + eps)`,
/// `w * (x * inv)`; then MLX's `rope` for the first `RD` dims (non-traditional
/// pairs `(p, p + RD/2)`, `inv_freq = exp2(-(p / (RD/2)) * log2(base))`,
/// `theta = float(pos + offset) * inv_freq`, `fast::cos` / `fast::sin`, the
/// same two products per output), the rest copied. Output `[B, H, L, D]`
/// FP32 as the chain's. `DARKBLOOM_QWEN35_QK_PREP=0` keeps the chain.
enum Qwen35QKPrep {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN35_QK_PREP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `x`: `[B, L, H, D]` (any strides), FP32; `weight`: `[D]` FP32;
    /// `offsets`: `[B]` int32 position offsets.
    static func run(
        _ x: MLXArray, weight: MLXArray, eps: Float, offsets: MLXArray,
        ropeDims: Int, ropeBase: Float
    ) -> MLXArray? {
        guard enabled, x.ndim == 4, x.dtype == .float32, weight.dtype == .float32,
            weight.ndim == 1, x.dim(3) == 256, weight.dim(0) == 256,
            ropeDims <= 256, ropeDims % 2 == 0,
            offsets.dtype == .int32, offsets.size == x.dim(0)
        else { return nil }
        let B = x.dim(0)
        let L = x.dim(1)
        let H = x.dim(2)
        let D = x.dim(3)
        return kernel(
            [x, weight, offsets.reshaped([-1]), MLXArray(eps), MLXArray(Foundation.log2(ropeBase))],
            template: [("H", H), ("L", L), ("D", D), ("RD", ropeDims)],
            grid: (64, B * L * H, 1), threadGroup: (64, 1, 1),
            outputShapes: [[B, H, L, D]], outputDTypes: [.float32])[0]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_qk_norm_rope",
        inputNames: ["x", "w", "offsets", "eps", "base"],
        outputNames: ["out"],
        source: """
            const uint lid = thread_position_in_threadgroup.x;
            const uint row = threadgroup_position_in_grid.y;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint h = row % uint(H);
            const uint l = (row / uint(H)) % uint(L);
            const uint bb = row / (uint(H) * uint(L));
            threadgroup float local_sums[32];
            threadgroup float local_inv[1];
            threadgroup float buf[D];

            const int64_t src = int64_t(bb) * x_strides[0] + int64_t(l) * x_strides[1]
                + int64_t(h) * x_strides[2];
            float xv[4];
            float acc = 0.0f;
            for (int i = 0; i < 4; i++) {
              xv[i] = x[src + int64_t(lid * 4 + uint(i)) * x_strides[3]];
              acc += xv[i] * xv[i];
            }
            acc = simd_sum(acc);
            if (sg == 0) { local_sums[lane] = 0; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) { local_sums[sg] = acc; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              float t = simd_sum(local_sums[lane]);
              if (lane == 0) { local_inv[0] = metal::precise::rsqrt(t / float(D) + eps); }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv = local_inv[0];
            for (int i = 0; i < 4; i++) {
              const uint e = lid * 4 + uint(i);
              buf[e] = w[e] * static_cast<float>(xv[i] * inv);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            constexpr uint HALF = uint(RD) / 2;
            const float L_pos = 1.0f * static_cast<float>(int(l) + offsets[bb]);
            const size_t dst = ((size_t(bb) * size_t(H) + size_t(h)) * size_t(L) + size_t(l)) * size_t(D);
            for (int i = 0; i < 4; i++) {
              const uint e = lid * 4 + uint(i);
              float v = buf[e];
              if (e < uint(RD)) {
                const uint p = e % HALF;
                const float d = static_cast<float>(p) / static_cast<float>(HALF);
                const float inv_freq = metal::exp2(-d * base);
                const float theta = L_pos * inv_freq;
                const float costheta = metal::fast::cos(theta);
                const float sintheta = metal::fast::sin(theta);
                const float x1 = buf[p];
                const float x2 = buf[p + HALF];
                if (e < HALF) {
                  v = x1 * costheta - x2 * sintheta;
                } else {
                  v = x1 * sintheta + x2 * costheta;
                }
              }
              out[dst + e] = v;
            }
            """,
        ensureRowContiguous: false)
}

// MARK: - Fused gated-delta prework

/// The gated-delta layer's prework between the input projection and the
/// recurrence as ONE launch: the depthwise causal convolution over the
/// retained 3-row state and the chunk, SiLU, the head split, the q/k RMSNorm
/// with the folded head scales, the gates `g = exp(-exp(A_log) *
/// softplus(a + dt_bias))` and `beta = sigmoid(b)`, and the next 3-row
/// convolution tail. One 128-thread threadgroup per (row, key head) computes
/// that head's q and k channel, the three value heads that share it, and
/// their gates. The op chain ran this as ~10 launches per layer over FP32
/// [S, 10240] intermediates (concat, conv1d, silu, three split copies, two
/// norms, the gates, the tail copy); here the only traffic is the chunk read
/// (four neighbouring rows per output row, cache-resident) and the
/// per-head outputs. Same FP32 formulas on the same inputs; the two norms'
/// mean-of-squares reduce in a different order than MLX's kernel (last-ulp).
/// `DARKBLOOM_QWEN35_GDN_PREWORK=0` restores the op chain.
enum Qwen35GDNPrework {
    struct Outputs {
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        let g: MLXArray
        let beta: MLXArray
        let tail: MLXArray
        /// `concatenated([convState, qkv], axis: 1)` in FP32, when requested.
        let convInput: MLXArray?
    }

    private static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN35_GDN_PREWORK"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // grid (128 * HK, S, B), threadgroup (128, 1, 1).
    // Template: InT, HK, HV, DK, DV, CD (conv channels), KS (taps). Inputs:
    // qkv [B, S, CD], cs [B, KS-1, CD], w [CD, KS, 1], a/b [B, S, HV],
    // alog/dtb [HV], wq/wk [DK], S (scalar). qkv, cs, w, a and b are read
    // through their strides: qkv and a/b are column slices of the stacked
    // qkv|z and b|a products, which a row-contiguous launch would copy first
    // (three extra dispatches per layer).
    private static let source = """
        constexpr int GRP = HV / HK;
        constexpr int KEY = HK * DK;
        constexpr int VOFF = 2 * KEY;
        constexpr int NK = KS - 1;
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
        const int64_t ab = int64_t(bb) * a_strides[0] + int64_t(t) * a_strides[1];
        const int64_t bbase = int64_t(bb) * b_strides[0] + int64_t(t) * b_strides[1];
        threadgroup float red[8];

        auto conv_silu = [&](uint col) -> float {
          float acc = 0.0f;
          #pragma clang loop unroll(full)
          for (int j = 0; j < KS; j++) {
            const int r = int(t) + j - NK;
            const float xv = (r < 0)
                ? cs[cb + int64_t(r + NK) * cs1 + int64_t(col) * cs2]
                : float(qkv[qb + int64_t(r) * qs1 + int64_t(col) * qs2]);
            acc = fma(xv, w[int64_t(col) * w_strides[0] + int64_t(j) * w_strides[1]], acc);
          }
          // MLX's silu: x * sigmoid(x), sigmoid in its stable functor form.
          const float sy = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
          const float sig = (acc < 0.0f) ? sy : 1.0f - sy;
          return acc * sig;
        };

        // q and k channel c of key head h.
        const uint colq = h * DK + c;
        const uint colk = KEY + h * DK + c;
        const float xq = conv_silu(colq);
        const float xk = conv_silu(colk);
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
        const size_t qkrow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HK) + size_t(h);
        q[qkrow * size_t(DK) + c] = (xq * invq) * wq[c];
        k[qkrow * size_t(DK) + c] = (xk * invk) * wk[c];

        // The GRP value heads of this key head.
        #pragma clang loop unroll(full)
        for (int i = 0; i < GRP; i++) {
          const uint hv = h * GRP + uint(i);
          const uint colv = VOFF + hv * DV + c;
          const size_t vrow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HV) + size_t(hv);
          v[vrow * size_t(DV) + c] = conv_silu(colv);
        }
        // Gates for those heads.
        if (c < uint(GRP)) {
          const uint hv = h * GRP + c;
          const size_t grow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HV) + size_t(hv);
          // g = exp(-exp(A_log) * softplus(a + dt_bias)), softplus as MLX's
          // logaddexp(x, 0); beta = sigmoid(b) in MLX's functor form.
          const float av = a[ab + int64_t(hv) * a_strides[2]] + dtb[hv];
          const float mx = metal::max(av, 0.0f);
          const float mn = metal::min(av, 0.0f);
          const float sp = mx + log1p(metal::exp(mn - mx));
          g[grow] = metal::precise::exp(-metal::precise::exp(alog[hv]) * sp);
          const float bv = b[bbase + int64_t(hv) * b_strides[2]];
          const float by = 1.0f / (1.0f + metal::exp(metal::abs(bv)));
          beta[grow] = (bv < 0.0f) ? by : 1.0f - by;
        }
        // The concatenated conv input [cs; qkv] (FP32), when requested: row
        // NK + t from this row's columns, rows 0..NK-1 from the state (t == 0).
        if (WCI) {
          const size_t cirow = (size_t(bb) * size_t(Sn + NK) + size_t(NK) + size_t(t)) * size_t(CD);
          const int64_t qsrc = qb + int64_t(t) * qs1;
          ci[cirow + colq] = float(qkv[qsrc + int64_t(colq) * qs2]);
          ci[cirow + colk] = float(qkv[qsrc + int64_t(colk) * qs2]);
          #pragma clang loop unroll(full)
          for (int i = 0; i < GRP; i++) {
            const uint colv = VOFF + (h * GRP + uint(i)) * DV + c;
            ci[cirow + colv] = float(qkv[qsrc + int64_t(colv) * qs2]);
          }
          if (t == 0) {
            #pragma clang loop unroll(full)
            for (int r = 0; r < NK; r++) {
              const size_t cirow0 = (size_t(bb) * size_t(Sn + NK) + size_t(r)) * size_t(CD);
              const int64_t csrc = cb + int64_t(r) * cs1;
              ci[cirow0 + colq] = cs[csrc + int64_t(colq) * cs2];
              ci[cirow0 + colk] = cs[csrc + int64_t(colk) * cs2];
              #pragma clang loop unroll(full)
              for (int i = 0; i < GRP; i++) {
                const uint colv = VOFF + (h * GRP + uint(i)) * DV + c;
                ci[cirow0 + colv] = cs[csrc + int64_t(colv) * cs2];
              }
            }
          }
        }
        // Next convolution tail: rows S-NK..S-1 of the concatenated input.
        #pragma clang loop unroll(full)
        for (int r = 0; r < NK; r++) {
          const int src = Sn + r - NK; // chunk row feeding tail row r (< 0: from cs)
          if (src >= 0 && int(t) == src) {
            const size_t trow = size_t(bb) * size_t(NK) * size_t(CD) + size_t(r) * size_t(CD);
            const int64_t qrow = qb + int64_t(t) * qs1;
            tail[trow + colq] = float(qkv[qrow + int64_t(colq) * qs2]);
            tail[trow + colk] = float(qkv[qrow + int64_t(colk) * qs2]);
            #pragma clang loop unroll(full)
            for (int i = 0; i < GRP; i++) {
              const uint colv = VOFF + (h * GRP + uint(i)) * DV + c;
              tail[trow + colv] = float(qkv[qrow + int64_t(colv) * qs2]);
            }
          } else if (src < 0 && t == 0) {
            const size_t trow = size_t(bb) * size_t(NK) * size_t(CD) + size_t(r) * size_t(CD);
            const int64_t crow = cb + int64_t(src + NK) * cs1;
            tail[trow + colq] = cs[crow + int64_t(colq) * cs2];
            tail[trow + colk] = cs[crow + int64_t(colk) * cs2];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GRP; i++) {
              const uint colv = VOFF + (h * GRP + uint(i)) * DV + c;
              tail[trow + colv] = cs[crow + int64_t(colv) * cs2];
            }
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework",
        inputNames: ["qkv", "cs", "w", "a", "b", "alog", "dtb", "wq", "wk", "S"],
        outputNames: ["q", "k", "v", "g", "beta", "tail", "ci"],
        source: source,
        ensureRowContiguous: !stridedReads)


    // The record's prework (row-contiguous inputs), which the 16-row verify
    // keeps: this tree's strided reads measured a longer verify round on the
    // ranked M5 (pratikgx `6b57e531`, the strided prework alone at verify
    // width: +6.5% per round against the record on the same box).
    // grid (128 * HK, S, B), threadgroup (128, 1, 1).
    // Template: InT, HK, HV, DK, DV, CD (conv channels), KS (taps). Inputs:
    // qkv [B, S, CD], cs [B, KS-1, CD], w [CD, KS, 1], a/b [B, S, HV],
    // alog/dtb [HV], wq/wk [DK], S (scalar).
    private static let recordSource = """
        constexpr int GRP = HV / HK;
        constexpr int KEY = HK * DK;
        constexpr int VOFF = 2 * KEY;
        constexpr int NK = KS - 1;
        const uint c = thread_position_in_threadgroup.x;
        const uint h = threadgroup_position_in_grid.x;
        const uint t = threadgroup_position_in_grid.y;
        const uint bb = threadgroup_position_in_grid.z;
        const int Sn = S;
        const size_t rowbase = (size_t(bb) * size_t(Sn)) * size_t(CD);
        const size_t csbase = size_t(bb) * size_t(NK) * size_t(CD);
        threadgroup float red[8];

        auto conv_silu = [&](uint col) -> float {
          float acc = 0.0f;
          #pragma clang loop unroll(full)
          for (int j = 0; j < KS; j++) {
            const int r = int(t) + j - NK;
            const float xv = (r < 0)
                ? cs[csbase + size_t(r + NK) * size_t(CD) + col]
                : float(qkv[rowbase + size_t(r) * size_t(CD) + col]);
            acc = fma(xv, w[size_t(col) * size_t(KS) + size_t(j)], acc);
          }
          // MLX's silu: x * sigmoid(x), sigmoid in its stable functor form.
          const float sy = 1.0f / (1.0f + metal::exp(metal::abs(acc)));
          const float sig = (acc < 0.0f) ? sy : 1.0f - sy;
          return acc * sig;
        };

        // q and k channel c of key head h.
        const uint colq = h * DK + c;
        const uint colk = KEY + h * DK + c;
        const float xq = conv_silu(colq);
        const float xk = conv_silu(colk);
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
        const size_t qkrow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HK) + size_t(h);
        q[qkrow * size_t(DK) + c] = (xq * invq) * wq[c];
        k[qkrow * size_t(DK) + c] = (xk * invk) * wk[c];

        // The GRP value heads of this key head.
        #pragma clang loop unroll(full)
        for (int i = 0; i < GRP; i++) {
          const uint hv = h * GRP + uint(i);
          const uint colv = VOFF + hv * DV + c;
          const size_t vrow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HV) + size_t(hv);
          v[vrow * size_t(DV) + c] = conv_silu(colv);
        }
        // Gates for those heads.
        if (c < uint(GRP)) {
          const uint hv = h * GRP + c;
          const size_t grow = (size_t(bb) * size_t(Sn) + size_t(t)) * size_t(HV) + size_t(hv);
          // g = exp(-exp(A_log) * softplus(a + dt_bias)), softplus as MLX's
          // logaddexp(x, 0); beta = sigmoid(b) in MLX's functor form.
          const float av = a[grow] + dtb[hv];
          const float mx = metal::max(av, 0.0f);
          const float mn = metal::min(av, 0.0f);
          const float sp = mx + log1p(metal::exp(mn - mx));
          g[grow] = metal::precise::exp(-metal::precise::exp(alog[hv]) * sp);
          const float bv = b[grow];
          const float by = 1.0f / (1.0f + metal::exp(metal::abs(bv)));
          beta[grow] = (bv < 0.0f) ? by : 1.0f - by;
        }
        // Next convolution tail: rows S-NK..S-1 of the concatenated input.
        #pragma clang loop unroll(full)
        for (int r = 0; r < NK; r++) {
          const int src = Sn + r - NK; // chunk row feeding tail row r (< 0: from cs)
          if (src >= 0 && int(t) == src) {
            const size_t trow = size_t(bb) * size_t(NK) * size_t(CD) + size_t(r) * size_t(CD);
            tail[trow + colq] = float(qkv[rowbase + size_t(t) * size_t(CD) + colq]);
            tail[trow + colk] = float(qkv[rowbase + size_t(t) * size_t(CD) + colk]);
            #pragma clang loop unroll(full)
            for (int i = 0; i < GRP; i++) {
              const uint colv = VOFF + (h * GRP + uint(i)) * DV + c;
              tail[trow + colv] = float(qkv[rowbase + size_t(t) * size_t(CD) + colv]);
            }
          } else if (src < 0 && t == 0) {
            const size_t trow = size_t(bb) * size_t(NK) * size_t(CD) + size_t(r) * size_t(CD);
            const size_t crow = csbase + size_t(src + NK) * size_t(CD);
            tail[trow + colq] = cs[crow + colq];
            tail[trow + colk] = cs[crow + colk];
            #pragma clang loop unroll(full)
            for (int i = 0; i < GRP; i++) {
              const uint colv = VOFF + (h * GRP + uint(i)) * DV + c;
              tail[trow + colv] = cs[crow + colv];
            }
          }
        }
        """

    private static let recordKernel = MLXFast.metalKernel(
        name: "qwen35_gdn_prework_rc",
        inputNames: ["qkv", "cs", "w", "a", "b", "alog", "dtb", "wq", "wk", "S"],
        outputNames: ["q", "k", "v", "g", "beta", "tail"],
        source: recordSource,
        ensureRowContiguous: true)

    /// `DARKBLOOM_QWEN35_GDN_PREWORK_STRIDED=0` copies the strided inputs to
    /// row-contiguous buffers before the launch, as before.
    private static let stridedReads: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN35_GDN_PREWORK_STRIDED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static func run(
        qkv: MLXArray, convState: MLXArray, convWeight: MLXArray, a: MLXArray, b: MLXArray,
        aLog: MLXArray, dtBias: MLXArray, normScales: (q: MLXArray, k: MLXArray),
        keyHeads: Int, valueHeads: Int, headKDim: Int, headVDim: Int,
        writeConvInput: Bool = false
    ) -> Outputs? {
        guard enabled, qkv.ndim == 3, convState.ndim == 3, convWeight.ndim == 3 else { return nil }
        let B = qkv.dim(0)
        let S = qkv.dim(1)
        let CD = qkv.dim(2)
        let KS = convWeight.dim(1)
        guard headKDim == 128, headVDim == 128, valueHeads % keyHeads == 0,
            CD == 2 * keyHeads * headKDim + valueHeads * headVDim,
            convState.shape == [B, KS - 1, CD], convWeight.shape == [CD, KS, 1],
            [DType.float32, .float16, .bfloat16].contains(qkv.dtype),
            convState.dtype == .float32, convWeight.dtype == .float32,
            a.dtype == .float32, b.dtype == .float32,
            a.shape == [B, S, valueHeads], b.shape == [B, S, valueHeads],
            aLog.shape == [valueHeads], dtBias.shape == [valueHeads],
            normScales.q.dtype == .float32, normScales.k.dtype == .float32,
            normScales.q.shape == [headKDim], normScales.k.shape == [headKDim],
            S > 0, S < 65536
        else { return nil }
        let alog = aLog.dtype == .float32 ? aLog : aLog.asType(.float32)
        let dtb = dtBias.dtype == .float32 ? dtBias : dtBias.asType(.float32)
        if B * S < HadamardQuantizedLinear.promptKernelsMinimumRows, !writeConvInput {
            let outputs = recordKernel(
                [qkv, convState, convWeight, a, b, alog, dtb, normScales.q, normScales.k,
                 MLXArray(Int32(S))],
                template: [
                    ("InT", qkv.dtype), ("HK", keyHeads), ("HV", valueHeads), ("DK", headKDim),
                    ("DV", headVDim), ("CD", CD), ("KS", KS),
                ],
                grid: (128 * keyHeads, S, B), threadGroup: (128, 1, 1),
                outputShapes: [
                    [B, S, keyHeads, headKDim], [B, S, keyHeads, headKDim],
                    [B, S, valueHeads, headVDim], [B, S, valueHeads], [B, S, valueHeads],
                    [B, KS - 1, CD],
                ],
                outputDTypes: [.float32, .float32, .float32, .float32, .float32, .float32])
            return Outputs(
                q: outputs[0], k: outputs[1], v: outputs[2], g: outputs[3], beta: outputs[4],
                tail: outputs[5], convInput: nil)
        }
        let outputs = kernel(
            [qkv, convState, convWeight, a, b, alog, dtb, normScales.q, normScales.k,
             MLXArray(Int32(S))],
            template: [
                ("InT", qkv.dtype), ("HK", keyHeads), ("HV", valueHeads), ("DK", headKDim),
                ("DV", headVDim), ("CD", CD), ("KS", KS), ("WCI", writeConvInput ? 1 : 0),
            ],
            grid: (128 * keyHeads, S, B), threadGroup: (128, 1, 1),
            outputShapes: [
                [B, S, keyHeads, headKDim], [B, S, keyHeads, headKDim],
                [B, S, valueHeads, headVDim], [B, S, valueHeads], [B, S, valueHeads],
                [B, KS - 1, CD], writeConvInput ? [B, KS - 1 + S, CD] : [1],
            ],
            outputDTypes: [.float32, .float32, .float32, .float32, .float32, .float32, .float32])
        return Outputs(
            q: outputs[0], k: outputs[1], v: outputs[2], g: outputs[3], beta: outputs[4],
            tail: outputs[5], convInput: writeConvInput ? outputs[6] : nil)
    }
}

// MARK: - One-launch signed block Hadamard rotation

/// The packed projections' input rotation as ONE Metal launch: the FP32 sign
/// multiply, the 1024-wide block Walsh-Hadamard transform and the output cast
/// (and, for a GDN output projection, the value-head layout gather) that the
/// op chain runs as three to five launches. The arithmetic is the stock
/// `hadamard_n<float, 1024, 16, 4>` kernel's, stage for stage: 64 threads per
/// block, sixteen FP32 values per thread through the same radix-16, radix-16,
/// radix-4 butterflies in the same order, scaled by 1/32 and rounded once to
/// the output dtype, so the rotated values are bit-identical to the chain.
/// Installed into `SignedBlockHadamard.fusedTransform` when the model loads;
/// `DARKBLOOM_BONSAI_FUSED_ROTATE=0` keeps the op chain.
enum Qwen35FusedHadamard {
    private static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_FUSED_ROTATE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Numerics probe only (off by default): `DARKBLOOM_BONSAI_Q8SIM=1` makes
    /// every prompt-width (>= 64 rows) FP16 rotation round-trip through a
    /// per-128-group symmetric int8 quantization before the packed matmul.
    private static let q8Probe: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_Q8SIM"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["1", "true", "yes", "on"].contains(value ?? "")
    }()

    private static let header = """
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
        """

    // grid: (64 * blocks, 1, 1), threadgroup (64, 1, 1); one threadgroup per
    // 1024-wide block. Template: InT, OutT, W (row width), BPR (blocks per
    // row), PRESIGNED, GR (GDN repeats, 1 = identity), GKH, GD.
    private static let source = """
        constexpr short N = 1024;
        constexpr short NT = 64;
        const uint blk = threadgroup_position_in_grid.x;
        const short i = short(thread_position_in_threadgroup.x);
        const uint row = blk / uint(BPR);
        const uint bcol = (blk % uint(BPR)) * uint(N);
        const size_t rowbase = size_t(row) * size_t(W);
        threadgroup float buf[N];
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
          const short index = j * 4 * NT + i * 4;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            const uint col = bcol + uint(index + r);
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
            buf[index + r] = v;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float x[16];
        short h = 1;
        #pragma clang loop unroll(full)
        for (short s = 0; s < 2; s++) {
          short k = i & (h - 1);
          short j = ((i - k) << 4) + k;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 16; r++) {
            x[r] = buf[j + h * r];
          }
          bonsai_hadamard_radix<16>(x);
          #pragma clang loop unroll(full)
          for (short r = 0; r < 16; r++) {
            buf[j + h * r] = x[r];
          }
          h <<= 4;
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (int t = 0; t < 4; t++) {
          short index = i + t * NT;
          short k = index & (h - 1);
          short j = ((index - k) << 2) + k;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            x[r] = buf[j + h * r];
          }
          bonsai_hadamard_radix<4>(x);
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            buf[j + h * r] = x[r];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
          const short index = j * 4 * NT + i * 4;
          if (QSIM) {
            // Emulate a per-128-group symmetric int8 quantization of the
            // rotated activation (numerics probe for the int8 matmul path):
            // one 128-group is the 4 values of each lane of one simdgroup.
            float amax = 0.0f;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) {
              amax = max(amax, fabs(buf[index + r] * 0.03125f));
            }
            amax = simd_max(amax);
            const float qs = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
            const float iqs = amax > 0.0f ? 127.0f / amax : 0.0f;
            #pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) {
              const float q = rint(buf[index + r] * 0.03125f * iqs);
              out[rowbase + bcol + uint(index + r)] = OutT(q * qs);
            }
          } else {
            #pragma clang loop unroll(full)
            for (short r = 0; r < 4; r++) {
              out[rowbase + bcol + uint(index + r)] = OutT(buf[index + r] * 0.03125f);
            }
          }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_signed_hadamard_1024",
        inputNames: ["inp", "signs"],
        outputNames: ["out"],
        source: source,
        header: header,
        ensureRowContiguous: true)

    // The same rotation quantized per 128-group for the tensor route: three
    // outputs, the UInt8 codes `round(v / scale) + 128`, the FP32 scale
    // (absmax / 127, one per row and 128-group: the four values of each lane
    // of one simdgroup) and the FP32 scaled sum `scale * sum(round(v / scale))`.
    private static let sourceInt8: String = {
        let tail = """
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const size_t gbase = size_t(row) * size_t(W / 128) + size_t(bcol / 128) + size_t(i >> 5);
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
          const short index = j * 4 * NT + i * 4;
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
            out[rowbase + bcol + kp] = uint8_t(int(q) + 128);
          }
          part = simd_sum(part);
          if ((i & 31) == 0) {
            qscale[gbase + size_t(2 * j)] = qs;
            qsum[gbase + size_t(2 * j)] = qs * part;
          }
        }
        """
        // The multi-line literal strips its closing delimiter's indentation.
        guard let cut = source.range(of: "threadgroup_barrier(mem_flags::mem_threadgroup);\n#pragma clang loop unroll(full)\nfor (short j = 0; j < 4; j++) {\n  const short index = j * 4 * NT + i * 4;\n  if (QSIM) {")
        else { preconditionFailure("fused rotation source changed") }
        return String(source[source.startIndex ..< cut.lowerBound]) + tail
    }()

    private static let kernelInt8 = MLXFast.metalKernel(
        name: "bonsai_signed_hadamard_1024_q8",
        inputNames: ["inp", "signs"],
        outputNames: ["out", "qscale", "qsum"],
        source: sourceInt8,
        header: header,
        ensureRowContiguous: true)

    // Operand addressing for the producer: element (row, c) of a [rows, W]
    // view (HD 0) or of a [B, L, heads, HD] view, through its strides.
    private static let headerProducer = header + """
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

        """

    // The quantizing rotation with the projection's input producer formed in
    // its read (the same FP32 arithmetic as the model's compiled chains):
    // PROD 1 `(a * sigmoid(a)) * b` (SwiGLU), 2 `a * sigmoid(b)` (the
    // attention output gate), 3 `(b * sigmoid(b)) * (w[d] * a * rsqrt(mean(a^2)
    // + eps))` per 128-wide head (the GDN output's gated norm), each times
    // the signs; then the transform and the quantization.
    private static let sourceInt8Producer: String = {
        let read = """
        constexpr short N = 1024;
        constexpr short NT = 64;
        const uint blk = threadgroup_position_in_grid.x;
        const short i = short(thread_position_in_threadgroup.x);
        const uint row = blk / uint(BPR);
        const uint bcol = (blk % uint(BPR)) * uint(N);
        // a and b are read through their strides (bonsai_q8p_row/col): the
        // SwiGLU halves are column slices of the stacked gate|up product, the
        // attention output is head-transposed and its gate is the second half
        // of each q|gate head, and the GDN z is a slice of qkv|z. None is
        // copied into a row-contiguous array first.
        const size_t rowbase = size_t(row) * size_t(W);
        const int64_t arow = bonsai_q8p_row<AHD>(a_shape, a_strides, row);
        const int64_t brow = bonsai_q8p_row<BHD>(b_shape, b_strides, row);
        threadgroup float buf[N];
        threadgroup float inv_rms[8];
        if (PROD == 3) {
          // Per-head RMS as rms_single_row: lane l sums elements 4l..4l+3 of
          // the head in order, then simd_sum; heads of this block in output
          // order, read from their source head.
          const uint lane = uint(i) & 31u;
          const uint sgi = uint(i) >> 5;
          for (uint hh = sgi; hh < 8u; hh += 2u) {
            const uint p0 = bcol + hh * uint(GD);
            const uint kh = p0 / uint(GR * GD);
            const uint rep = (p0 % uint(GR * GD)) / uint(GD);
            const uint src_head = rep * uint(GKH) + kh;
            const uint c0 = src_head * uint(GD) + lane * 4;
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (int r = 0; r < 4; r++) {
              const float tx = float(a[arow + bonsai_q8p_col<AHD>(a_strides, c0 + uint(r))]);
              acc += tx * tx;
            }
            acc = simd_sum(acc);
            if (lane == 0) {
              inv_rms[hh] = metal::precise::rsqrt(acc / float(GD) + eps[0]);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
          const short index = j * 4 * NT + i * 4;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            const uint col = bcol + uint(index + r);
            uint src = col;
            if (GR > 1) {
              const uint d = col % uint(GD);
              const uint hr = col / uint(GD);
              const uint h = hr / uint(GR);
              const uint rr = hr % uint(GR);
              src = (rr * uint(GKH) + h) * uint(GD) + d;
            }
            const float av = float(a[arow + bonsai_q8p_col<AHD>(a_strides, src)]);
            const float bv = float(b[brow + bonsai_q8p_col<BHD>(b_strides, src)]);
            float v;
            if (PROD == 1) {
              v = (av * bonsai_sigmoid(av)) * bv;
            } else if (PROD == 2) {
              v = av * bonsai_sigmoid(bv);
            } else {
              const float xn = w[src % uint(GD)] * (av * inv_rms[(index + r) / GD]);
              v = (bv * bonsai_sigmoid(bv)) * xn;
            }
            buf[index + r] = v * signs[col];
          }
        }
        """
        guard let cut = sourceInt8.range(of: "threadgroup_barrier(mem_flags::mem_threadgroup);\nfloat x[16];")
        else { preconditionFailure("quantizing rotation source changed") }
        return read + String(sourceInt8[cut.lowerBound...])
    }()

    private static let kernelInt8Producer = MLXFast.metalKernel(
        name: "bonsai_signed_hadamard_1024_q8p",
        inputNames: ["a", "b", "w", "eps", "signs"],
        outputNames: ["out", "qscale", "qsum"],
        source: sourceInt8Producer,
        header: headerProducer,
        ensureRowContiguous: false)

    nonisolated(unsafe) private static let unusedWeight = MLXArray.zeros([128], dtype: .float32)
    nonisolated(unsafe) private static let unusedEps = MLXArray([Float(0)])

    // The same rotation stored in OutT with a second output: the FP32 sum of
    // every 128 consecutive rounded outputs, for the verify-width tensor route.
    private static let sourceWithGroupSums: String = {
        let tail = """
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const size_t gbase = size_t(row) * size_t(W / 128) + size_t(bcol / 128) + size_t(i >> 5);
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; j++) {
          const short index = j * 4 * NT + i * 4;
          float part = 0.0f;
          #pragma clang loop unroll(full)
          for (short r = 0; r < 4; r++) {
            const OutT o = OutT(buf[index + r] * 0.03125f);
            out[rowbase + bcol + uint(index + r)] = o;
            part += float(o);
          }
          part = simd_sum(part);
          if ((i & 31) == 0) {
            gsum[gbase + size_t(2 * j)] = part;
          }
        }
        """
        guard let cut = source.range(of: "threadgroup_barrier(mem_flags::mem_threadgroup);\n#pragma clang loop unroll(full)\nfor (short j = 0; j < 4; j++) {\n  const short index = j * 4 * NT + i * 4;\n  if (QSIM) {")
        else { preconditionFailure("fused rotation source changed") }
        return String(source[source.startIndex ..< cut.lowerBound]) + tail
    }()

    private static let kernelWithGroupSums = MLXFast.metalKernel(
        name: "bonsai_signed_hadamard_1024_gsum",
        inputNames: ["inp", "signs"],
        outputNames: ["out", "gsum"],
        source: sourceWithGroupSums,
        header: header,
        ensureRowContiguous: true)

    nonisolated(unsafe) private static var installed = false

    static func installIfNeeded() {
        guard enabled, !installed else { return }
        installed = true
        SignedBlockHadamard.fusedTransform = { x, signs, blockSize, preSigned, gdnLayout, outputDType in
            guard blockSize == 1024, x.ndim >= 1,
                [DType.float32, .float16, .bfloat16].contains(x.dtype),
                [DType.float32, .float16, .bfloat16].contains(outputDType),
                signs.dtype == .float32
            else { return nil }
            let width = x.dim(-1)
            guard width % 1024 == 0, signs.size == width else { return nil }
            var repeats = 1
            var keyHeads = 1
            var headDim = 1
            if let gdnLayout {
                guard gdnLayout.width == width, gdnLayout.valueHeads % gdnLayout.keyHeads == 0,
                    width % gdnLayout.valueHeads == 0
                else { return nil }
                repeats = gdnLayout.valueHeads / gdnLayout.keyHeads
                keyHeads = gdnLayout.keyHeads
                headDim = width / gdnLayout.valueHeads
            }
            let rows = x.size / width
            guard rows > 0 else { return nil }
            let blocksPerRow = width / 1024
            let template: [(String, any KernelTemplateArg)] = [
                ("InT", x.dtype), ("OutT", outputDType), ("W", width), ("BPR", blocksPerRow),
                ("PRESIGNED", preSigned ? 1 : 0), ("GR", repeats), ("GKH", keyHeads), ("GD", headDim),
                ("QSIM", (q8Probe && rows >= 64 && outputDType == .float16) ? 1 : 0),
            ]
            return kernel(
                [x, signs], template: template,
                grid: (64 * rows * blocksPerRow, 1, 1), threadGroup: (64, 1, 1),
                outputShapes: [x.shape], outputDTypes: [outputDType])[0]
        }
        SignedBlockHadamard.fusedTransformInt8 = {
            x, signs, blockSize, preSigned, gdnLayout, groupSize in
            guard groupSize == 128, blockSize == 1024, x.ndim >= 1,
                [DType.float32, .float16, .bfloat16].contains(x.dtype),
                signs.dtype == .float32
            else { return nil }
            let width = x.dim(-1)
            guard width % 1024 == 0, signs.size == width else { return nil }
            var repeats = 1
            var keyHeads = 1
            var headDim = 1
            if let gdnLayout {
                guard gdnLayout.width == width, gdnLayout.valueHeads % gdnLayout.keyHeads == 0,
                    width % gdnLayout.valueHeads == 0
                else { return nil }
                repeats = gdnLayout.valueHeads / gdnLayout.keyHeads
                keyHeads = gdnLayout.keyHeads
                headDim = width / gdnLayout.valueHeads
            }
            let rows = x.size / width
            guard rows > 0 else { return nil }
            let blocksPerRow = width / 1024
            let template: [(String, any KernelTemplateArg)] = [
                ("InT", x.dtype), ("OutT", DType.uint8), ("W", width), ("BPR", blocksPerRow),
                ("PRESIGNED", preSigned ? 1 : 0), ("GR", repeats), ("GKH", keyHeads), ("GD", headDim),
                ("QSIM", 0), ("PERM", Qwen35TensorPackedMatmul.support == .staged8 ? 1 : 0),
            ]
            let groupShape = Array(x.shape.dropLast()) + [width / 128]
            let outputs = kernelInt8(
                [x, signs], template: template,
                grid: (64 * rows * blocksPerRow, 1, 1), threadGroup: (64, 1, 1),
                outputShapes: [x.shape, groupShape, groupShape],
                outputDTypes: [.uint8, .float32, .float32])
            return SignedBlockHadamard.Int8Activation(
                codes: outputs[0], scales: outputs[1], scaledSums: outputs[2])
        }
        SignedBlockHadamard.fusedTransformWithGroupSums = {
            x, signs, blockSize, preSigned, gdnLayout, outputDType, groupSize in
            guard groupSize == 128, blockSize == 1024, x.ndim >= 1,
                [DType.float32, .float16, .bfloat16].contains(x.dtype),
                outputDType == .float16, signs.dtype == .float32
            else { return nil }
            let width = x.dim(-1)
            guard width % 1024 == 0, signs.size == width else { return nil }
            var repeats = 1
            var keyHeads = 1
            var headDim = 1
            if let gdnLayout {
                guard gdnLayout.width == width, gdnLayout.valueHeads % gdnLayout.keyHeads == 0,
                    width % gdnLayout.valueHeads == 0
                else { return nil }
                repeats = gdnLayout.valueHeads / gdnLayout.keyHeads
                keyHeads = gdnLayout.keyHeads
                headDim = width / gdnLayout.valueHeads
            }
            let rows = x.size / width
            guard rows > 0 else { return nil }
            let blocksPerRow = width / 1024
            let template: [(String, any KernelTemplateArg)] = [
                ("InT", x.dtype), ("OutT", outputDType), ("W", width), ("BPR", blocksPerRow),
                ("PRESIGNED", preSigned ? 1 : 0), ("GR", repeats), ("GKH", keyHeads), ("GD", headDim),
                ("QSIM", 0),
            ]
            let outputs = kernelWithGroupSums(
                [x, signs], template: template,
                grid: (64 * rows * blocksPerRow, 1, 1), threadGroup: (64, 1, 1),
                outputShapes: [x.shape, Array(x.shape.dropLast()) + [width / 128]],
                outputDTypes: [outputDType, .float32])
            return (outputs[0], outputs[1])
        }
        SignedBlockHadamard.fusedTransformInt8Producer = {
            producer, signs, blockSize, gdnLayout, groupSize in
            guard groupSize == 128, blockSize == 1024, signs.dtype == .float32 else { return nil }
            let a: MLXArray
            let b: MLXArray
            var w = unusedWeight
            var eps = unusedEps
            let prod: Int
            switch producer {
            case .swiglu(let gate, let up):
                guard gdnLayout == nil, gate.shape == up.shape else { return nil }
                a = gate; b = up; prod = 1
            case .sigmoidGate(let x, let gate):
                guard gdnLayout == nil, x.shape == gate.shape else { return nil }
                a = x; b = gate; prod = 2
            case .gatedRMSNorm(let x, let gate, let weight, let epsilon):
                guard x.shape == gate.shape, weight.dtype == .float32, weight.ndim == 1,
                    weight.dim(0) == 128
                else { return nil }
                a = x; b = gate; w = weight; eps = MLXArray([epsilon]); prod = 3
            }
            // Each operand is widened at its own read: the z of an FP16 qkv|z
            // stack gates an FP32 GDN output.
            guard [DType.float32, .float16].contains(a.dtype),
                [DType.float32, .float16].contains(b.dtype), a.ndim >= 1
            else { return nil }
            let width = signs.size
            guard width % 1024 == 0, a.size % width == 0 else { return nil }
            var repeats = 1
            var keyHeads = 1
            var headDim = 1
            if let gdnLayout {
                guard gdnLayout.width == width, gdnLayout.valueHeads % gdnLayout.keyHeads == 0,
                    width % gdnLayout.valueHeads == 0
                else { return nil }
                repeats = gdnLayout.valueHeads / gdnLayout.keyHeads
                keyHeads = gdnLayout.keyHeads
                headDim = width / gdnLayout.valueHeads
            }
            if prod == 3 {
                // The per-head norm needs the 128-wide head of this pack.
                guard gdnLayout != nil ? headDim == 128 : width % 128 == 0 else { return nil }
                if gdnLayout == nil { headDim = 128 }
            }
            let rows = a.size / width
            guard rows > 0 else { return nil }
            let blocksPerRow = width / 1024
            // 4-D operands are read as [B, L, heads, headDim] (the head
            // transpose of the attention output and the gate half of each
            // q|gate head flatten only through a copy); the rest as
            // [rows, width] views, which a column slice reshapes to.
            func operand(_ v: MLXArray) -> (MLXArray, Int) {
                if v.ndim == 4, v.dim(0) * v.dim(1) == rows, v.dim(2) * v.dim(3) == width {
                    return (v, v.dim(3))
                }
                return (v.reshaped(rows, width), 0)
            }
            let (aView, aHead) = operand(a)
            let (bView, bHead) = operand(b)
            let template: [(String, any KernelTemplateArg)] = [
                ("InT", a.dtype), ("W", width), ("BPR", blocksPerRow),
                ("GR", repeats), ("GKH", keyHeads), ("GD", headDim), ("PROD", prod),
                ("PERM", Qwen35TensorPackedMatmul.support == .staged8 ? 1 : 0),
                ("AHD", aHead), ("BHD", bHead),
            ]
            let outShape = [rows, width]
            let groupShape = [rows, width / 128]
            let outputs = kernelInt8Producer(
                [aView, bView, w, eps, signs], template: template,
                grid: (64 * rows * blocksPerRow, 1, 1), threadGroup: (64, 1, 1),
                outputShapes: [outShape, groupShape, groupShape],
                outputDTypes: [.uint8, .float32, .float32])
            return SignedBlockHadamard.Int8Activation(
                codes: outputs[0], scales: outputs[1], scaledSums: outputs[2])
        }
    }
}

/// The prompt-width packed matmul on the tensor unit over the raw 2-bit codes
/// (MetalPerformancePrimitives `matmul2d`, `uint8 x uint2b_format -> int32`),
/// applying each 128-group's FP16 scale and offset in FP32:
/// `y = sum_g as[m,g] * (s[n,g] * (C[m,n,g] - 128 * colsum[n,g]) + b[n,g] * rs[m,g])`
/// where `C` is the integer product of the shifted codes `q + 128` with the
/// weight codes, `colsum` the per-group sum of the weight codes, `as` and
/// `rs` the activation's per-group scale and code sum. The weights are read
/// as stored; the scales, offsets and folded code sums are read through a
/// per-layer derived-constant cache. Installed into
/// `HadamardQuantizedLinear.tensorPackedMatmul` when the model loads;
/// `DARKBLOOM_BONSAI_TENSOR_ROUTE=0` keeps the dequantizing kernels.
enum Qwen35TensorPackedMatmul {
    private static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // The trailing newline matters: the JIT appends the kernel signature
    // directly after the header text.
    private static let header = """
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

        """

    // grid: (N / 64 * 128, M / 64, 1), threadgroup (128, 1, 1). Inputs: xq
    // uint8 [M, K], w uint32 [N, K / 16], scalesT / biasesT half [K / 128, N],
    // uT float [K / 128, N] (= -128 * s * colsum), ascale / rsb float [M, K /
    // 128], ksz int32 [K, M, N]. Template: OutT.
    private static let source = """
        const int K = ksz[0]; const int M = ksz[1]; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 64;
        const int m0 = int(threadgroup_position_in_grid.y) * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            64, 64, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
        tensor<device uint8_t, dextents<int, 2>, tensor_inline> A((device uint8_t*)xq, dextents<int, 2>(K, M));
        tensor<device uint2b_format, dextents<int, 2>, tensor_inline> B((device uchar*)w, dextents<int, 2>(K, N));
        auto tA0 = A.template slice<128, 64>(0, m0);
        auto tB0 = B.template slice<128, 64>(0, n0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB0)>, int32_t>();
        constexpr int CAP = 32;
        // Destination layout (the NAX fragment layout): element i ->
        //   n = n0 + 16 * (sg & 1) + fn + (i & 3) + 32 * ((i >> 3) & 1)
        //   m = m0 + 16 * (sg >> 1) + fm + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1)
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        const int nb = n0 + 16 * int(sg & 1) + fn;
        const int mb = m0 + 16 * int(sg >> 1) + fm;
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 32);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 32);
        const device float4* up0 = (const device float4*)(uT + nb);
        const device float4* up1 = (const device float4*)(uT + nb + 32);
        const int NQ = N / 4;
        const size_t mrow[4] = {(size_t)mb, (size_t)(mb + 8), (size_t)(mb + 32), (size_t)(mb + 40)};
        for (int g = 0; g < Kg; g++) {
          auto tA = A.template slice<128, 64>(g * 128, m0);
          auto tB = B.template slice<128, 64>(g * 128, n0);
          op.run(tA, tB, cT);
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = float4(bp0[g * NQ]), b1 = float4(bp1[g * NQ]);
          const float4 u0 = up0[g * NQ], u1 = up1[g * NQ];
          float as[4], rb[4];
          #pragma clang loop unroll(full)
          for (int q = 0; q < 4; q++) { as[q] = ascale[mrow[q] * Kg + g]; rb[q] = rsb[mrow[q] * Kg + g]; }
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1;
            const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            const float u = nh ? u1[c] : u0[c];
            const float t = fma(s, float(cT[i]), u);
            acc[i] = fma(b, rb[mh], fma(as[mh], t, acc[i]));
          }
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) {
          const int c = i & 3; const int nh = (i >> 3) & 1;
          const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
          out[(size_t)mm * N + nb + c + 32 * nh] = OutT(acc[i]);
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_q8",
        inputNames: ["xq", "w", "scalesT", "biasesT", "uT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: source,
        header: header,
        ensureRowContiguous: true)

    // Verify width (`half x uint2b_format -> float`, 16 rows): each threadgroup
    // owns TN (64 or 32) output columns; its four simdgroups split K into four
    // contiguous quarters (one 16 x 32 x 128 op per 128-group) and their
    // partials are summed through threadgroup memory. grid: (N / 32 * 128, 1,
    // 1), threadgroup (128, 1, 1). Inputs: x half [16, K], w uint32 [N, K / 16],
    // scalesT / biasesT half [K / 128, N], rowsum float [16, K / 128], ksz.
    private static let sourceNarrow = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * TN;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int gper = Kg / SG;
        const int g0 = int(sg) * gper;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, TN, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device half, dextents<int, 2>, tensor_inline> A((device half*)x, dextents<int, 2>(K, M));
        tensor<device uint2b_format, dextents<int, 2>, tensor_inline> B((device uchar*)w, dextents<int, 2>(K, N));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto tB0 = B.template slice<128, TN>(0, n0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(tB0)>, float>();
        constexpr int CAP = TN / 2;
        // Destination layout: element i -> n = n0 + fn + (i & 3) + 16 * (i >> 3),
        // m = fm + 8 * ((i >> 2) & 1).
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        for (int g = g0; g < g0 + gper; g++) {
          auto tA = A.template slice<128, 16>(g * 128, 0);
          auto tB = B.template slice<128, TN>(g * 128, n0);
          op.run(tA, tB, cT);
          float4 sv[TN / 16], bv[TN / 16];
          #pragma clang loop unroll(full)
          for (int q = 0; q < TN / 16; q++) {
            sv[q] = float4(*(const device half4*)(scalesT + (size_t)g * N + n0 + fn + 16 * q));
            bv[q] = float4(*(const device half4*)(biasesT + (size_t)g * N + n0 + fn + 16 * q));
          }
          const float rs0 = rowsum[(size_t)fm * Kg + g];
          const float rs1 = rowsum[(size_t)(fm + 8) * Kg + g];
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
            acc[i] = fma(sv[nq][c], cT[i], fma(bv[nq][c], mh ? rs1 : rs0, acc[i]));
          }
        }
        threadgroup float red[SG - 1][CAP * 32];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { red[sg - 1][i * 32 + lane] = acc[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            float v = acc[i];
            #pragma clang loop unroll(full)
            for (int q = 0; q < SG - 1; q++) { v += red[q][i * 32 + lane]; }
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nq = i >> 3;
            out[(size_t)(fm + 8 * mh) * N + n0 + fn + c + 16 * nq] = OutT(v);
          }
        }
        """

    private static let kernelNarrow = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16",
        inputNames: ["x", "w", "scalesT", "biasesT", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrow,
        header: header,
        ensureRowContiguous: true)

    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY=0` keeps the record's verify-width
    /// kernels (the few-row core and the split-K bodies) with the tensor route
    /// still on at prompt width.
    /// Columns per threadgroup at verify width: 32 (measured 1.5% faster on
    /// the decode window than 64, which the isolated kernel preferred);
    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_TN=64` selects 64.
    private static let narrowTileColumns: Int = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_TN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return value == "64" ? 64 : 32
    }()

    /// `DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_SG8=0` keeps four simdgroups per
    /// threadgroup for every verify-width shape.
    private static let narrowDeepSplit: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY_SG8"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let verifyEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_VERIFY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // MARK: - Packed-operand support on this box's Metal toolchain

    /// How the tensor unit reads the 2-bit codes: `native2b` as a
    /// `uint2b_format` tensor over the stored bytes; `staged4b` on a toolchain
    /// without the 2-bit format (the ranked box's, September 2026): the same
    /// bytes expanded to `uint4b_format` in threadgroup memory per 128-group;
    /// `none` when neither compiles (every route then stays off).
    enum PackedOperandSupport { case native2b, staged8, staged4b, none }

    private static let probeHeader = header

    private static let probeSourceNative = """
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device half, dextents<int, 2>, tensor_inline> A((device half*)a, dextents<int, 2>(128, 16));
        tensor<device uint2b_format, dextents<int, 2>, tensor_inline> B((device uchar*)w, dextents<int, 2>(128, 32));
        auto tA = A.template slice<128, 16>(0, 0);
        auto tB = B.template slice<128, 32>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>, metal::remove_addrspace_t<decltype(tB)>, float>();
        op.run(tA, tB, cT);
        out[thread_position_in_grid.x] = cT[0];
        """

    private static let probeSourceStaged = """
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device half, dextents<int, 2>, tensor_inline> A((device half*)a, dextents<int, 2>(128, 16));
        threadgroup uint32_t bs[32 * 128 / 8];
        const uint lane = thread_index_in_simdgroup;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 16; j++) { bs[lane * 16 + j] = w[lane * 16 + j]; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B((threadgroup uchar*)bs, dextents<int, 2>(128, 32));
        auto tA = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>, metal::remove_addrspace_t<decltype(B)>, float>();
        op.run(tA, B, cT);
        out[thread_position_in_grid.x] = cT[0];
        """

    private static let probeSourceStaged8 = """
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device uint8_t, dextents<int, 2>, tensor_inline> A((device uint8_t*)a, dextents<int, 2>(128, 16));
        threadgroup uint32_t bs[32 * 128 / 4];
        const uint lane = thread_index_in_simdgroup;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 32; j++) { bs[lane * 32 + j] = w[(lane * 32 + j) & 255]; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tensor<threadgroup uint8_t, dextents<int, 2>, tensor_inline> B((threadgroup uint8_t*)bs, dextents<int, 2>(128, 32));
        auto tA = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(tA)>, metal::remove_addrspace_t<decltype(B)>, int32_t>();
        op.run(tA, B, cT);
        out[thread_position_in_grid.x] = float(cT[0]);
        """

    nonisolated(unsafe) private static var probeFailed = false

    /// Compiles and runs one tiny kernel; false when the toolchain rejects it
    /// (the JIT error is caught by a scoped MLX error handler instead of
    /// ending the process).
    private static func probe(_ name: String, _ source: String, aDType: DType = .float16) -> Bool {
        probeFailed = false
        let kernel = MLXFast.metalKernel(
            name: name, inputNames: ["a", "w"], outputNames: ["out"], source: source,
            header: probeHeader, ensureRowContiguous: true)
        withErrorHandler({ _ in Qwen35TensorPackedMatmul.probeFailed = true }) {
            let a = MLXArray.zeros([16, 128], dtype: aDType)
            let w = MLXArray.zeros([32 * 128 / 16], dtype: .uint32)
            let y = kernel(
                [a, w], template: [], grid: (32, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[32]], outputDTypes: [.float32])[0]
            eval(y)
        }
        return !probeFailed
    }

    /// Decided once at model init. `DARKBLOOM_BONSAI_TENSOR_ROUTE_PACKED`
    /// = `uint2b` / `uint4b` / `off` forces a form (for A/B on a box that has
    /// both); otherwise the probes decide.
    static let support: PackedOperandSupport = {
        let forced = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_PACKED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch forced {
        case "uint2b", "native": return probe("bonsai_probe_uint2b", probeSourceNative) ? .native2b : .none
        case "uint8", "staged8": return probe("bonsai_probe_uint8", probeSourceStaged8, aDType: .uint8) ? .staged8 : .none
        case "uint4b", "staged": return probe("bonsai_probe_uint4b", probeSourceStaged) ? .staged4b : .none
        case "off", "none", "0": return .none
        default: break
        }
        // Preferred order: the int8-staged form (fastest measured, no packed
        // format needed), then the native 2-bit operand, then the 4-bit-staged form.
        if probe("bonsai_probe_uint8", probeSourceStaged8, aDType: .uint8) { return .staged8 }
        if probe("bonsai_probe_uint2b", probeSourceNative) { return .native2b }
        if probe("bonsai_probe_uint4b", probeSourceStaged) { return .staged4b }
        return .none
    }()

    /// True when the toolchain takes `tensor` operands at all (either form).
    static var tensorOperandsAvailable: Bool { support != .none }

    /// The verify-width route needs the native 2-bit operand: its staged
    /// forms measured no faster than the record's verify kernels, so without
    /// `uint2b_format` that route stays off. Probed separately (once).
    static let verifyNativeAvailable: Bool = {
        if support == .native2b { return true }
        let forced = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_PACKED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if forced == "uint4b" || forced == "staged" || forced == "off" || forced == "none" || forced == "0" {
            return false
        }
        return probe("bonsai_probe_uint2b", probeSourceNative)
    }()

    // The prompt-width kernel for a toolchain without `uint2b_format`: the
    // same math, the 64 x 128 weight tile of each 128-group expanded from the
    // stored 2-bit words to 4-bit codes in threadgroup memory (double-buffered,
    // one barrier per group) and read by the op as a threadgroup
    // `uint4b_format` tensor. Same inputs as `source`.
    private static let sourceStaged = """
        const int K = ksz[0]; const int M = ksz[1]; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 64;
        const int m0 = int(threadgroup_position_in_grid.y) * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 64, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
        tensor<device uint8_t, dextents<int, 2>, tensor_inline> A((device uint8_t*)xq, dextents<int, 2>(K, M));
        // staged B: 64 columns x 128 codes as 4-bit, 2 per byte, k inner: byte index (n * 128 + k) / 2
        threadgroup uint32_t bs[2][64 * 128 / 8];
        tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B0((threadgroup uchar*)bs[0], dextents<int, 2>(128, 64));
        tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B1((threadgroup uchar*)bs[1], dextents<int, 2>(128, 64));
        auto tA0 = A.template slice<128, 64>(0, m0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        constexpr int CAP = 32;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        const int nb = n0 + 16 * int(sg & 1) + fn;
        const int mb = m0 + 16 * int(sg >> 1) + fm;
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 32);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 32);
        const device float4* up0 = (const device float4*)(uT + nb);
        const device float4* up1 = (const device float4*)(uT + nb + 32);
        const int NQ = N / 4;
        const size_t mrow[4] = {(size_t)mb, (size_t)(mb + 8), (size_t)(mb + 32), (size_t)(mb + 40)};
        // staging assignment: thread t -> column c = t >> 1, K half h = t & 1 (64 codes = 4 words -> 8 words of nibbles)
        const int sc = int(tid >> 1); const int sh = int(tid & 1);
        const device uint32_t* wrow = w + (size_t)(n0 + sc) * (K / 16) + sh * 4;
        auto stage = [&](int g, int buf) {
          const device uint4* src = (const device uint4*)(wrow + g * 8);
          const uint4 v = *src;
          threadgroup uint32_t* dst = bs[buf] + sc * 16 + sh * 8;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; j++) {
            const uint32_t wv = v[j];
            uint32_t lo = wv & 0xFFFFu; uint32_t hi = wv >> 16;
            lo = (lo | (lo << 8)) & 0x00FF00FFu; lo = (lo | (lo << 4)) & 0x0F0F0F0Fu; lo = (lo | (lo << 2)) & 0x33333333u;
            hi = (hi | (hi << 8)) & 0x00FF00FFu; hi = (hi | (hi << 4)) & 0x0F0F0F0Fu; hi = (hi | (hi << 2)) & 0x33333333u;
            dst[2 * j] = lo; dst[2 * j + 1] = hi;
          }
        };
        stage(0, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < Kg; g++) {
          const int cur = g & 1;
          if (g + 1 < Kg) { stage(g + 1, cur ^ 1); }
          auto tA = A.template slice<128, 64>(g * 128, m0);
          if (cur == 0) { op.run(tA, B0, cT); } else { op.run(tA, B1, cT); }
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = float4(bp0[g * NQ]), b1 = float4(bp1[g * NQ]);
          const float4 u0 = up0[g * NQ], u1 = up1[g * NQ];
          float as[4], rb[4];
          #pragma clang loop unroll(full)
          for (int q = 0; q < 4; q++) { as[q] = ascale[mrow[q] * Kg + g]; rb[q] = rsb[mrow[q] * Kg + g]; }
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1; const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            const float u = nh ? u1[c] : u0[c];
            const float t = fma(s, float(cT[i]), u);
            acc[i] = fma(b, rb[mh], fma(as[mh], t, acc[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) {
          const int c = i & 3; const int nh = (i >> 3) & 1;
          const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
          out[(size_t)mm * N + nb + c + 32 * nh] = OutT(acc[i]);
        }
        """

    // The prompt-width kernel with the 2-bit words expanded to int8 codes in
    // threadgroup memory (`uint8 x uint8 -> int32`; no packed format needed):
    // four AND/shift ops per word, the codes landing in a permuted K order
    // inside each 16-block (position p holds code 4 * (p % 4) + p / 4), which
    // the quantizing rotation writes its codes in (`PERM`), so the integer
    // dot product is unchanged. Same inputs as `source`.
    private static let sourceStaged8 = """
        const int K = ksz[0]; const int M = ksz[1]; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 64;
        const int m0 = int(threadgroup_position_in_grid.y) * 64;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_position_in_threadgroup.x;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 64, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
        tensor<device uint8_t, dextents<int, 2>, tensor_inline> A((device uint8_t*)xq, dextents<int, 2>(K, M));
        // staged B: 64 columns x 128 codes as 4-bit, 2 per byte, k inner: byte index (n * 128 + k) / 2
        threadgroup uint32_t bs[2][64 * 128 / 4];
        tensor<threadgroup uint8_t, dextents<int, 2>, tensor_inline> B0((threadgroup uint8_t*)bs[0], dextents<int, 2>(128, 64));
        tensor<threadgroup uint8_t, dextents<int, 2>, tensor_inline> B1((threadgroup uint8_t*)bs[1], dextents<int, 2>(128, 64));
        auto tA0 = A.template slice<128, 64>(0, m0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, int32_t>();
        constexpr int CAP = 32;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        const int nb = n0 + 16 * int(sg & 1) + fn;
        const int mb = m0 + 16 * int(sg >> 1) + fm;
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        const device half4* sp0 = (const device half4*)(scalesT + nb);
        const device half4* sp1 = (const device half4*)(scalesT + nb + 32);
        const device half4* bp0 = (const device half4*)(biasesT + nb);
        const device half4* bp1 = (const device half4*)(biasesT + nb + 32);
        const device float4* up0 = (const device float4*)(uT + nb);
        const device float4* up1 = (const device float4*)(uT + nb + 32);
        const int NQ = N / 4;
        const size_t mrow[4] = {(size_t)mb, (size_t)(mb + 8), (size_t)(mb + 32), (size_t)(mb + 40)};
        // staging assignment: thread t -> column c = t >> 1, K half h = t & 1 (64 codes = 4 words -> 8 words of nibbles)
        const int sc = int(tid >> 1); const int sh = int(tid & 1);
        const device uint32_t* wrow = w + (size_t)(n0 + sc) * (K / 16) + sh * 4;
        auto stage = [&](int g, int buf) {
          const device uint4* src = (const device uint4*)(wrow + g * 8);
          const uint4 v = *src;
          threadgroup uint32_t* dst = bs[buf] + sc * 32 + sh * 16;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 4; j++) {
            const uint32_t wv = v[j];
            dst[4 * j + 0] = wv & 0x03030303u;
            dst[4 * j + 1] = (wv >> 2) & 0x03030303u;
            dst[4 * j + 2] = (wv >> 4) & 0x03030303u;
            dst[4 * j + 3] = (wv >> 6) & 0x03030303u;
          }
        };
        stage(0, 0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = 0; g < Kg; g++) {
          const int cur = g & 1;
          if (g + 1 < Kg) { stage(g + 1, cur ^ 1); }
          auto tA = A.template slice<128, 64>(g * 128, m0);
          if (cur == 0) { op.run(tA, B0, cT); } else { op.run(tA, B1, cT); }
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = float4(bp0[g * NQ]), b1 = float4(bp1[g * NQ]);
          const float4 u0 = up0[g * NQ], u1 = up1[g * NQ];
          float as[4], rb[4];
          #pragma clang loop unroll(full)
          for (int q = 0; q < 4; q++) { as[q] = ascale[mrow[q] * Kg + g]; rb[q] = rsb[mrow[q] * Kg + g]; }
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int nh = (i >> 3) & 1; const int mh = ((i >> 2) & 1) | (((i >> 4) & 1) << 1);
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            const float u = nh ? u1[c] : u0[c];
            const float t = fma(s, float(cT[i]), u);
            acc[i] = fma(b, rb[mh], fma(as[mh], t, acc[i]));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) {
          const int c = i & 3; const int nh = (i >> 3) & 1;
          const int mm = mb + 8 * ((i >> 2) & 1) + 32 * ((i >> 4) & 1);
          out[(size_t)mm * N + nb + c + 32 * nh] = OutT(acc[i]);
        }
        """

    private static let kernelStaged8 = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_q8_u8",
        inputNames: ["xq", "w", "scalesT", "biasesT", "uT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: sourceStaged8,
        header: header,
        ensureRowContiguous: true)

    private static let kernelStaged = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_q8_u4",
        inputNames: ["xq", "w", "scalesT", "biasesT", "uT", "ascale", "rsb", "ksz"],
        outputNames: ["out"],
        source: sourceStaged,
        header: header,
        ensureRowContiguous: true)

    // The verify-width kernel for the same toolchain: each simdgroup expands
    // its own 32 x 128 tile per group into its threadgroup region
    // (double-buffered, simdgroup-scoped sync). 32 columns, 4 simdgroups.
    private static let sourceNarrowStaged = """
        const int K = ksz[0]; const int M = 16; const int N = ksz[2];
        const int Kg = K / 128;
        const int n0 = int(threadgroup_position_in_grid.x) * 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int gper = Kg / 4;
        const int g0 = int(sg) * gper;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 128, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        tensor<device half, dextents<int, 2>, tensor_inline> A((device half*)x, dextents<int, 2>(K, M));
        // per-simdgroup staged tile: 32 columns x 128 codes as nibbles = 2 KB; double-buffered
        threadgroup uint32_t bs[4][2][32 * 128 / 8];
        tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B0((threadgroup uchar*)bs[sg][0], dextents<int, 2>(128, 32));
        tensor<threadgroup uint4b_format, dextents<int, 2>, tensor_inline> B1((threadgroup uchar*)bs[sg][1], dextents<int, 2>(128, 32));
        auto tA0 = A.template slice<128, 16>(0, 0);
        auto cT = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tA0)>, metal::remove_addrspace_t<decltype(B0)>, float>();
        constexpr int CAP = 16;
        const int fm = int(((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
        const int fn = int((((lane >> 3) & 1) * 2 + (lane & 1)) * 4);
        float acc[CAP];
        #pragma clang loop unroll(full)
        for (int i = 0; i < CAP; i++) { acc[i] = 0.0f; }
        const device half4* sp0 = (const device half4*)(scalesT + n0 + fn);
        const device half4* sp1 = (const device half4*)(scalesT + n0 + fn + 16);
        const device half4* bp0 = (const device half4*)(biasesT + n0 + fn);
        const device half4* bp1 = (const device half4*)(biasesT + n0 + fn + 16);
        const int NQ = N / 4;
        // staging: lane l -> column l (32 columns), all 128 codes of the group = 8 words -> 16 nibble words
        const device uint32_t* wrow = w + (size_t)(n0 + int(lane)) * (K / 16);
        auto stage = [&](int g, int buf) {
          const device uint4* src = (const device uint4*)(wrow + g * 8);
          const uint4 v0 = src[0]; const uint4 v1 = src[1];
          threadgroup uint32_t* dst = bs[sg][buf] + int(lane) * 16;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 8; j++) {
            const uint32_t wv = (j < 4) ? v0[j] : v1[j - 4];
            uint32_t lo = wv & 0xFFFFu; uint32_t hi = wv >> 16;
            lo = (lo | (lo << 8)) & 0x00FF00FFu; lo = (lo | (lo << 4)) & 0x0F0F0F0Fu; lo = (lo | (lo << 2)) & 0x33333333u;
            hi = (hi | (hi << 8)) & 0x00FF00FFu; hi = (hi | (hi << 4)) & 0x0F0F0F0Fu; hi = (hi | (hi << 2)) & 0x33333333u;
            dst[2 * j] = lo; dst[2 * j + 1] = hi;
          }
        };
        stage(g0, 0);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = g0; g < g0 + gper; g++) {
          const int cur = (g - g0) & 1;
          if (g + 1 < g0 + gper) { stage(g + 1, cur ^ 1); }
          auto tA = A.template slice<128, 16>(g * 128, 0);
          if (cur == 0) { op.run(tA, B0, cT); } else { op.run(tA, B1, cT); }
          const float4 s0 = float4(sp0[g * NQ]), s1 = float4(sp1[g * NQ]);
          const float4 b0 = float4(bp0[g * NQ]), b1 = float4(bp1[g * NQ]);
          const float rs0 = rowsum[(size_t)fm * Kg + g];
          const float rs1 = rowsum[(size_t)(fm + 8) * Kg + g];
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
            const float s = nh ? s1[c] : s0[c];
            const float b = nh ? b1[c] : b0[c];
            acc[i] = fma(s, cT[i], fma(b, mh ? rs1 : rs0, acc[i]));
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup float red[3][16 * 32];
        if (sg > 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) { red[sg - 1][i * 32 + lane] = acc[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
          #pragma clang loop unroll(full)
          for (int i = 0; i < CAP; i++) {
            const float v = acc[i] + red[0][i * 32 + lane] + red[1][i * 32 + lane] + red[2][i * 32 + lane];
            const int c = i & 3; const int mh = (i >> 2) & 1; const int nh = (i >> 3) & 1;
            out[(size_t)(fm + 8 * mh) * N + n0 + fn + c + 16 * nh] = OutT(v);
          }
        }
        """

    private static let kernelNarrowStaged = MLXFast.metalKernel(
        name: "bonsai_tensor_packed_matmul_m16_u4",
        inputNames: ["x", "w", "scalesT", "biasesT", "rowsum", "ksz"],
        outputNames: ["out"],
        source: sourceNarrowStaged,
        header: header,
        ensureRowContiguous: true)

    private static let dimsLock = NSLock()
    nonisolated(unsafe) private static var dims: [[Int]: MLXArray] = [:]
    private static func dimsArray(k: Int, m: Int, n: Int) -> MLXArray {
        dimsLock.withLock {
            if let cached = dims[[k, m, n]] { return cached }
            let array = MLXArray([Int32(k), Int32(m), Int32(n)])
            dims[[k, m, n]] = array
            return array
        }
    }

    /// The per-group sums of the 2-bit codes of `weight` (`[n, k / 16]`
    /// UInt32, 16 codes per word, LSB first), `[n, k / 128]` FP32: the count
    /// of set low bits plus twice the count of set high bits of each word.
    private static func codeSums(_ weight: MLXArray, k: Int) -> MLXArray {
        let n = weight.dim(0)
        func popcount(_ v: MLXArray) -> MLXArray {
            var x = v - ((v >> 1) & MLXArray(UInt32(0x5555_5555)))
            x = (x & MLXArray(UInt32(0x3333_3333))) + ((x >> 2) & MLXArray(UInt32(0x3333_3333)))
            x = (x + (x >> 4)) & MLXArray(UInt32(0x0F0F_0F0F))
            return (x * MLXArray(UInt32(0x0101_0101))) >> 24
        }
        let low = popcount(weight & MLXArray(UInt32(0x5555_5555)))
        let high = popcount((weight >> 1) & MLXArray(UInt32(0x5555_5555)))
        let perWord = low + high * MLXArray(UInt32(2))
        return perWord.reshaped(n, k / 128, 8).sum(axis: -1).asType(.float32)
    }

    nonisolated(unsafe) private static var installed = false

    static func installIfNeeded() {
        guard enabled, !installed else { return }
        installed = true
        guard support != .none else { return }
        HadamardQuantizedLinear.tensorPackedMatmulApplies = { rows, n, k in
            rows % 64 == 0 && n % 64 == 0 && k % 512 == 0
        }
        if verifyEnabled, verifyNativeAvailable {
            HadamardQuantizedLinear.tensorPackedMatmulNarrowApplies = { rows, n, k in
                rows <= 16 && n % 32 == 0 && k % 512 == 0 && n < 65536
            }
            HadamardQuantizedLinear.tensorPackedMatmulNarrow = {
                rotated, sums, weight, scales, biases, groupSize, outputDType, cache in
                guard groupSize == 128, rotated.dtype == .float16, rotated.ndim == 2,
                    sums.dtype == .float32, weight.dtype == .uint32,
                    scales.dtype == .float16, biases.dtype == .float16,
                    [DType.float16, .float32].contains(outputDType)
                else { return nil }
                let m = rotated.dim(0)
                let k = rotated.dim(1)
                let n = weight.dim(0)
                guard m == 16, n % 32 == 0, k % 512 == 0, weight.dim(1) == k / 16,
                    sums.shape == [m, k / 128], scales.shape == [n, k / 128],
                    biases.shape == [n, k / 128]
                else { return nil }
                let scalesT = cache.derived(scales, tag: 1) { $0.transposed(1, 0).contiguous() }
                let biasesT = cache.derived(biases, tag: 2) { $0.transposed(1, 0).contiguous() }
                if !verifyNativeAvailable {
                    return kernelNarrowStaged(
                        [rotated, weight, scalesT, biasesT, sums, dimsArray(k: k, m: m, n: n)],
                        template: [("OutT", outputDType)],
                        grid: (n / 32 * 128, 1, 1), threadGroup: (128, 1, 1),
                        outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
                }
                let tn = (narrowTileColumns == 64 && n % 64 == 0) ? 64 : 32
                // Eight simdgroups split a long K (down_proj); four otherwise.
                let sg = (narrowDeepSplit && k >= 8192 && (k / 128) % 8 == 0) ? 8 : 4
                return kernelNarrow(
                    [rotated, weight, scalesT, biasesT, sums, dimsArray(k: k, m: m, n: n)],
                    template: [("OutT", outputDType), ("TN", tn), ("SG", sg)],
                    grid: (n / tn * 32 * sg, 1, 1), threadGroup: (32 * sg, 1, 1),
                    outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
            }
        }
        HadamardQuantizedLinear.tensorPackedMatmul = {
            activation, weight, scales, biases, groupSize, outputDType, cache in
            let codes = activation.codes
            guard groupSize == 128, codes.dtype == .uint8, codes.ndim == 2,
                activation.scales.dtype == .float32, activation.scaledSums.dtype == .float32,
                weight.dtype == .uint32, scales.dtype == .float16, biases.dtype == .float16,
                [DType.float16, .float32].contains(outputDType)
            else { return nil }
            let m = codes.dim(0)
            let k = codes.dim(1)
            let n = weight.dim(0)
            guard m % 64 == 0, n % 64 == 0, k % 512 == 0, weight.dim(1) == k / 16,
                activation.scales.shape == [m, k / 128],
                activation.scaledSums.shape == [m, k / 128],
                scales.shape == [n, k / 128], biases.shape == [n, k / 128]
            else { return nil }
            let scalesT = cache.derived(scales, tag: 1) { $0.transposed(1, 0).contiguous() }
            let biasesT = cache.derived(biases, tag: 2) { $0.transposed(1, 0).contiguous() }
            let foldedSums = cache.derived(scales, tag: 3) { s in
                (s.asType(.float32) * codeSums(weight, k: k) * MLXArray(Float(-128)))
                    .transposed(1, 0).contiguous()
            }
            let packedKernel: MLXFast.MLXFastKernel
            switch support {
            case .native2b: packedKernel = kernel
            case .staged8: packedKernel = kernelStaged8
            default: packedKernel = kernelStaged
            }
            return packedKernel(
                [codes, weight, scalesT, biasesT, foldedSums, activation.scales,
                 activation.scaledSums, dimsArray(k: k, m: m, n: n)],
                template: [("OutT", outputDType)],
                grid: (n / 64 * 128, m / 64, 1), threadGroup: (128, 1, 1),
                outputShapes: [[m, n]], outputDTypes: [outputDType])[0]
        }
    }
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// MTP head. Non-nil only when `_qwen35MTPEnabled == true` at init time
    /// AND `args.mtpNumHiddenLayers > 0`.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.__init__ (MTPModule attachment)
    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPModule?

    /// Checkpoint quantization policy staged by `loadWeights` (via
    /// `QuantizationPolicyReceiving`) before `sanitize` runs; drives the
    /// per-layer decision whether routed-expert gate/up halves may fuse.
    /// `nil` for unquantized checkpoints.
    public var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization?

    public init(_ args: Qwen35TextConfiguration) {
        Qwen35FusedHadamard.installIfNeeded()
        Qwen35TensorPackedMatmul.installIfNeeded()
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }

        // Attach MTP head only when enabled and config declares MTP layers.
        // omlx: `if n_mtp > 0 and is_mtp_active(): self.mtp = q35.MTPModule(args)`
        if args.mtpNumHiddenLayers > 0 && _qwen35MTPEnabled {
            _mtp.wrappedValue = Qwen35MTPModule(args)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Inner model now returns pre-norm hidden; apply norm + lm_head here.
        // omlx: TextModel.__call__ (normed = self.model.norm(hidden); out = lm_head(normed))
        let hidden = model(inputs, cache: cache)
        var out = model.norm(hidden)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    public var cbv2LayerKinds: [CBv2LayerKind] { configuration.cbv2LayerKinds }

    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        configuration.cbv2RecurrentStateSpec(
            activationDType: cbv2CheckpointActivationDType ?? model.embedTokens.weight.dtype)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities { configuration.cbv2Capabilities }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { storageIndex, kind in
            try makeLayerCache(kind.modelLayerIndex ?? storageIndex, kind)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Port of omlx commit 696d90a:
        //   patches/mlx_lm_mtp/qwen35_model.py TextModel.sanitize
        //
        // Key differences from stock mlx-lm:
        //  1. Gate norm shift on unsanitized conv1d shape ONLY (not on MTP key presence).
        //     Stock code uses `hasMTPWeights || hasUnsanitizedConv1d`, which double-shifts
        //     already-converted MLX checkpoints that have mtp.* keys.
        //  2. Keep mtp.* keys when the MTP head is attached; strip them otherwise.
        //  3. Extend norm-shift key set with MTP-specific norm names.

        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasUnsanitizedConv1d  // NOT hasMTPWeights

        var weights = weights

        // Routed experts are built fused (`SwitchGLU(fuseGateUp: true)`), and
        // the `qwen3_5_text` registry entry reaches this sanitizer directly —
        // without the MoE/VLM wrappers that call the fusion helper. Apply it
        // here so raw stacked `experts.gate_up_proj` exports and converted
        // split `switch_mlp.{gate,up}_proj.*` checkpoints both match the
        // module tree. Idempotent, so the wrapper paths that already fused
        // are unaffected; `mtp.*` keys stay split.
        if configuration.numExperts > 0 {
            weights = qwen35FuseSwitchMLPGateUp(
                weights: weights,
                perLayerQuantization: checkpointPerLayerQuantization,
                setFused: { qwen35SetSwitchGLUGateUpFused($1, at: $0, in: self) })
        }

        // Keep mtp.* keys if the head is attached; strip them otherwise.
        // omlx: `if not hasattr(self, "mtp"): weights = {k:v if "mtp." not in k}`
        if mtp == nil {
            weights = weights.filter { !$0.key.contains("mtp.") }
        } else if !weights.keys.contains(where: { $0.contains("mtp.") }) {
            // MTP enabled but no mtp.* keys in checkpoint → needs re-conversion.
            // omlx: raises ValueError with "weights are missing the mtp.* tensors"
            print(
                "[WARNING] Qwen35TextModel.sanitize: MTP head is enabled but no mtp.* "
                + "weights found. Load will likely fail or produce garbage. "
                + "Re-convert the checkpoint with a converter that preserves MTP weights.")
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // Extended norm key set includes MTP-specific names.
        // omlx: norm_keys tuple with ".pre_fc_norm_hidden.weight" etc.
        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }
}

extension Qwen35TextModel: CBv2PositionAxisProviding {
    public var cbv2PositionAxisCount: Int? { 3 }
}

extension Qwen35TextModel: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable
{
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        // Prompt-width windows on the unpositioned seam project the vocabulary
        // at the LAST position only. Every caller of this seam reads row -1
        // (the worker's teacher-forced stepper, `narrowPrefillOutput`,
        // `decodeLogits`), so the [B, L, 248320] projection of the other rows
        // was pure discarded work. The trunk, every K/V write and every
        // recurrent stage are unchanged; final RMSNorm is row-independent, so
        // norm-after-slice equals slice-after-norm for the surviving row.
        // Single-token windows keep the original path.
        guard tokens.dim(1) > 1 else {
            return positionedForward(
                tokens, inputEmbedding: nil, cache: caches,
                recurrentState: recurrentState, positionIds: nil)
        }
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: nil, lastRowOnly: true)
        let last = model.norm(hidden[0..., (hidden.dim(1) - 1)..., 0...])
        return lmHead.map { $0(last) } ?? model.embedTokens.asLinear(last)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        positionedForward(
            tokens, inputEmbedding: nil, cache: caches,
            recurrentState: recurrentState, positionIds: positionIds)
    }

    public var supportsVisionSpanPrefill: Bool { false }
    public var supportsCausalVisionPrefill: Bool { true }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        model.embedTokens(inputs)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        preconditionFailure("Qwen35 embedding prefill requires request-owned recurrent state")
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        positionedForward(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds)
    }

    private func positionedForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        let caches = cache ?? []
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            inputs, inputEmbeddings: inputEmbedding, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds)
        let rows = hidden.dim(1)
        let output = Self.narrowPromptRows && rows > 32
            ? hidden[0..., (rows - 1)..., 0...] : hidden
        let normalized = model.norm(output)
        return lmHead.map { $0(normalized) } ?? model.embedTokens.asLinear(normalized)
    }

    /// Prompt-width forwards through this seam are read at their final
    /// position only.
    static let narrowPromptRows: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_PROMPT_LAST_ROW"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
}

// MARK: - ContinuousBatchingV2 prompt-only output narrowing

/// CBv2 consumes only the final prompt position, so the prompt path skips
/// the [B, L, 248320] vocabulary projection for discarded rows: intermediate
/// chunks return a one-element hidden handle, the frontier chunk projects
/// exactly one row. The trunk — every K/V write, every recurrent-state
/// stage, positions, embeddings — is byte-identical to `positionedForward`;
/// final RMSNorm is row-independent, so norm-after-slice equals
/// slice-after-norm for the surviving row. Decode, MTP draft/verify, and
/// capture paths keep their existing full contracts.
extension Qwen35TextModel: CBv2RecurrentLanguageModelPrefillForwardable {
    /// The Qwen trunk is shape-generic over `[B, L]` with one recurrent
    /// state row per batch row (`recurrentState.count == B` precondition),
    /// and CBv2 attention attends each packed row against its OWN KV — a
    /// packed row is semantically identical to running alone.
    public var cbv2SupportsPackedPrefill: Bool { true }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        let caches = cache ?? []
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            inputs, inputEmbeddings: inputEmbedding, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            lastRowOnly: positionIds == nil)
        switch requirement {
        case .evaluationOnly:
            // Small handle whose graph depends on the whole trunk — forcing
            // it commits every layer's K/V write and recurrent stage.
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            let last = model.norm(hidden[0..., -1, 0...])
            return lmHead.map { $0(last) } ?? model.embedTokens.asLinear(last)
        }
    }
}

extension Qwen35TextModel: CBv2RecurrentMTPForwardable {
    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 MTP target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds)
        let normalized = model.norm(hidden)
        let logits = lmHead.map { $0(normalized) } ?? model.embedTokens.asLinear(normalized)
        return (logits, hidden)
    }
}

/// The prompt forward of a speculative leg. The engine keeps one row of the
/// prompt's logits (`narrowPrefillOutput`) but needs every trusted hidden row
/// for the drafter, so this projects the vocabulary at the last position only
/// and returns the full pre-norm hidden. The serial prefill path above already
/// does the same narrowing; without this seam the speculative leg paid a
/// `[B, 512, 248320]` head projection it then discarded. Final RMSNorm is
/// row-independent, so norm-after-slice equals slice-after-norm for the
/// surviving row; the trunk, every K/V write, every recurrent stage and the
/// DFlash 2 tap are the same as in `cbv2ForwardWithHidden`.
extension Qwen35TextModel: CBv2RecurrentPrefillHiddenForwardable {
    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 MTP target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            lastRowOnly: positionIds == nil)
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        switch requirement {
        case .evaluationOnly:
            // A small handle whose graph depends on the whole trunk.
            return (last[0..., 0..., 0 ..< 1], hidden)
        case .lastPositionLogits:
            let normalized = model.norm(last)
            let logits = lmHead.map { $0(normalized) } ?? model.embedTokens.asLinear(normalized)
            return (logits, hidden)
        }
    }
}

extension Qwen35TextModel: CBv2RecurrentCaptureMTPForwardable {
    /// MTP capture-verify: identical to `cbv2ForwardWithHidden` except each
    /// GatedDeltaNet layer stages per-position captured conv/SSM stacks so
    /// finalize can commit the accepted position on device. Attention and
    /// MoE/projection layers are stateless over the window and run batched.
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("Qwen35 CBv2 MTP target received a legacy KV cache")
            }
            return attending
        }
        let hidden = model.cbv2Forward(
            tokens, inputEmbeddings: nil, caches: attending,
            recurrentState: recurrentState, positionIds: positionIds,
            captureRecurrentWindow: true)
        let normalized = model.norm(hidden)
        let logits: MLXArray
        if let lmHead {
            logits = model.exactTargetVerify
                ? qwen35A3BExactW4G64Projection(lmHead, normalized)
                : lmHead(normalized)
        } else if model.exactTargetVerify, normalized.dim(1) > 1 {
            logits = qwen35A3BTimewiseProjection(normalized) {
                model.embedTokens.asLinear($0)
            }
        } else {
            logits = model.embedTokens.asLinear(normalized)
        }
        return (logits, hidden)
    }
}

// MARK: - Qwen35TextModel + DFlash 2

/// The target surface the DFlash 2 block drafter binds to, and the layer tap
/// the engine reads between rounds.
///
/// BOTH SHARED TENSORS GO THROUGH THE MODULE. On this pack `embed_tokens` and
/// `lm_head` are packed Hadamard modules: `HadamardQuantizedLinear` IS a
/// `QuantizedLinear` and `HadamardQuantizedEmbedding` is NOT a
/// `QuantizedEmbedding`, so a path that casts and reads `.weight` type-checks,
/// runs, and returns numbers with the folded transform left unapplied. The two
/// calls below are plain module calls for exactly that reason; see
/// `Qwen35InlineMTPAssistant.headLogits` and `docs/bonsai2-27b-port-notes.md`
/// section 6.1.
extension Qwen35TextModel: DFlash2TapTarget {
    public var dFlash2VocabularySize: Int { vocabularySize }
    public var dFlash2HiddenSize: Int { configuration.hiddenSize }
    public var dFlash2LayerCount: Int { configuration.hiddenLayers }

    public var dFlash2TapLayerIds: [Int]? {
        get { model.dFlash2Tap.layerIds }
        set { model.dFlash2Tap.layerIds = newValue }
    }

    public var dFlash2TappedHidden: MLXArray? { model.dFlash2Tap.tappedHidden }

    public func embedTokensForDFlash2(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func logitsForDFlash2Hidden(_ hidden: MLXArray) -> MLXArray {
        // The drafter reads the head in FP16 and keeps the FP16 logits: its
        // top-k reads them directly (see `HadamardQuantizedLinear.drafterHeadFloat16`).
        if HadamardQuantizedLinear.drafterHeadFloat16,
            let head = lmHead as? HadamardQuantizedLinear
        {
            return head.forwardUnwidened(hidden)
        }
        return lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
    }
}

extension Qwen35TextModel: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(
        _ logits: MLXArray
    ) -> (ids: MLXArray, values: MLXArray) {
        precondition(logits.ndim == 3, "Qwen35 MTP policy logits must be [B,L,V]")
        let batch = logits.dim(0)
        let length = logits.dim(1)
        let vocabularySize = logits.dim(2)
        precondition(batch > 0 && length > 0 && vocabularySize >= 2)
        let rows = batch * length
        let topTwo = qwen35MTPTopTwoRows(
            logits.reshaped([1, rows, vocabularySize]))
        return (
            topTwo.ids.reshaped([batch, length, 2]),
            topTwo.values.reshaped([batch, length, 2]))
    }
}

// MARK: - Qwen35TextModel + MTPCapable

extension Qwen35TextModel: MTPCapable {
    public var hasMTPHead: Bool { mtp != nil }

    /// Run a backbone forward that also returns pre-final-norm hidden states.
    ///
    /// Returns `(logits, preNormHidden)` where `preNormHidden` is the raw
    /// backbone output before `model.norm`. Qwen MTP applies this exact final
    /// norm once at the target-to-assistant boundary, followed by the head's
    /// separately trained `pre_fc_norm_hidden`; recursive assistant hidden
    /// bypasses the target final norm.
    ///
    /// PR #990: `return out, hidden  # pre-norm hidden for MTP head`
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.__call__ with return_hidden=True
    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        let cacheOpt: [KVCache?] = cache.map { Optional($0) }
        let hidden = model(input.tokens, cache: cacheOpt, nConfirmed: nConfirmed)
        let normed = model.norm(hidden)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(normed)
        } else {
            logits = model.embedTokens.asLinear(normed)
        }
        // Return raw capture hidden. The bound Qwen MTP adapter owns the exact
        // target final-norm application when these rows enter trusted history.
        return (logits, hidden)
    }

    /// Commit a shorter accepted recurrent prefix from the most recent wide
    /// verify. On false, transient rollback data is cleared so a fallback
    /// repair cannot accidentally reuse an incompatible tape.
    public func replayRecurrentPrefix(
        cache: [any KVCache], committedRows: Int
    ) -> Bool {
        model.replayRecurrentPrefix(
            cache: cache.map { Optional($0) },
            committedRows: committedRows)
    }

    /// Release a verify tape after full acceptance or an external fallback.
    /// Persistent recurrent state and attention offsets are unchanged.
    public func clearRecurrentPrefixReplay(cache: [any KVCache]) {
        model.clearRecurrentPrefixReplay(cache: cache.map { Optional($0) })
    }

    /// Run the MTP head forward.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.mtp_forward
    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        guard let mtp else {
            fatalError("mtpForward called but MTP head is not attached. "
                + "Set _qwen35MTPEnabled = true before loading the model.")
        }
        let mtpOut = mtp(
            hidden: hidden,
            nextTokenIds: nextTokenIds,
            embedTokens: model.embedTokens,
            cache: cache)
        if configuration.tieWordEmbeddings {
            return model.embedTokens.asLinear(mtpOut)
        }
        return lmHead!(mtpOut)
    }

    /// Allocate a fresh KV cache for the MTP head layers.
    /// omlx: patches/mlx_lm_mtp/qwen35_model.py TextModel.make_mtp_cache
    public func makeMTPCache() -> [any KVCache] {
        guard let mtp else { return [] }
        return mtp.layers.map { _ in KVCacheSimple() as any KVCache }
    }
}

extension Qwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level Model

public class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    public init(_ args: Qwen35Configuration) {
        let textModel = Qwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public var cbv2LayerKinds: [CBv2LayerKind] { languageModel.cbv2LayerKinds }
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        languageModel.cbv2RecurrentStateSpec
    }
    public var cbv2Capabilities: CBv2ModelCapabilities { languageModel.cbv2Capabilities }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try languageModel.newCacheV2(makeLayerCache: makeLayerCache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen35Model: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable
{
    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        languageModel.cbv2Forward(
            tokens, caches: caches, recurrentState: recurrentState)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        languageModel.cbv2Forward(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }

    public var supportsVisionSpanPrefill: Bool { languageModel.supportsVisionSpanPrefill }
    public var supportsCausalVisionPrefill: Bool { languageModel.supportsCausalVisionPrefill }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        languageModel.scaledInputEmbeddings(inputs)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        languageModel.embeddingForward(inputs, inputEmbedding: inputEmbedding, cache: cache)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        languageModel.embeddingForward(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds)
    }
}

extension Qwen35Model: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool {
        languageModel.cbv2SupportsPackedPrefill
    }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        languageModel.cbv2RecurrentPrefill(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds,
            requirement: requirement)
    }
}

extension Qwen35Model: CBv2RecurrentMTPForwardable {
    public var cbv2MTPTargetIdentity: ObjectIdentifier {
        ObjectIdentifier(languageModel)
    }

    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHidden(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }
}

extension Qwen35Model: CBv2RecurrentPrefillHiddenForwardable {
    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHiddenForPrefill(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, requirement: requirement)
    }
}

extension Qwen35Model: CBv2RecurrentCaptureMTPForwardable {
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        languageModel.cbv2ForwardWithHiddenCaptured(
            tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
    }
}

extension Qwen35Model: DFlash2TapTarget {
    public var dFlash2VocabularySize: Int { languageModel.dFlash2VocabularySize }
    public var dFlash2HiddenSize: Int { languageModel.dFlash2HiddenSize }
    public var dFlash2LayerCount: Int { languageModel.dFlash2LayerCount }

    public var dFlash2TapLayerIds: [Int]? {
        get { languageModel.dFlash2TapLayerIds }
        set { languageModel.dFlash2TapLayerIds = newValue }
    }

    public var dFlash2TappedHidden: MLXArray? { languageModel.dFlash2TappedHidden }

    public func embedTokensForDFlash2(_ tokens: MLXArray) -> MLXArray {
        languageModel.embedTokensForDFlash2(tokens)
    }

    public func logitsForDFlash2Hidden(_ hidden: MLXArray) -> MLXArray {
        languageModel.logitsForDFlash2Hidden(hidden)
    }
}

extension Qwen35Model: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(
        _ logits: MLXArray
    ) -> (ids: MLXArray, values: MLXArray) {
        languageModel.cbv2MTPTopTwo(logits)
    }
}

extension Qwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - Qwen35Model + MTPCapable

/// VLM-outer-wrapper pass-through for MTPCapable.
/// Forwards all MTP calls to the inner `languageModel` (a Qwen35TextModel).
/// omlx: patches/mlx_lm_mtp/qwen35_model.py `_patch_outer_model`
extension Qwen35Model: MTPCapable {
    public var hasMTPHead: Bool { languageModel.hasMTPHead }

    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray) {
        languageModel.callWithHidden(input: input, cache: cache, nConfirmed: nConfirmed)
    }

    public func replayRecurrentPrefix(
        cache: [any KVCache], committedRows: Int
    ) -> Bool {
        languageModel.replayRecurrentPrefix(
            cache: cache, committedRows: committedRows)
    }

    public func clearRecurrentPrefixReplay(cache: [any KVCache]) {
        languageModel.clearRecurrentPrefixReplay(cache: cache)
    }

    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray {
        languageModel.mtpForward(hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    public func makeMTPCache() -> [any KVCache] {
        languageModel.makeMTPCache()
    }
}
