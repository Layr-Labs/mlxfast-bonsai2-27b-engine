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
import MLXFast
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

/// The gated delta recurrence with its gates formed by one fused kernel and
/// the state kept in FP32, matching `gatedDeltaUpdate` op for op.
func qwen35GatedDelta(
    q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
    aLog: MLXArray, dtBias: MLXArray, state: MLXArray?, mask: MLXArray?
) -> (MLXArray, MLXArray) {
    let gates = Qwen35FusedElementwise.gatedDeltaGates([a, b, aLog, dtBias])
    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    var ssm = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if ssm.dtype != .float32 {
        ssm = ssm.asType(.float32)
    }
    return gatedDeltaKernel(q: q, k: k, v: v, g: gates[0], beta: gates[1], state: ssm, mask: mask)
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

    /// The gated output norm with its `silu(gate) * x` tail fused into one
    /// kernel; the same FP32 arithmetic as `norm(out, gate:)`.
    private func gatedNorm(_ out: MLXArray, gate: MLXArray) -> MLXArray {
        if let fused = Qwen35GdnT1Mega.gatedNorm(
            out, gate: gate, weight: norm.weight, eps: norm.eps)
        {
            return fused
        }
        if let wide = Bonsai2GdnGnorm.apply(
            out, gate: gate, weight: norm.weight, eps: norm.eps)
        {
            return wide
        }
        if Qwen35T1Compile.isArmed(), Qwen35T1Compile.isDecodeShape(out) {
            let normed = MLXFast.rmsNorm(out, weight: norm.weight, eps: norm.eps)
            return Qwen35T1Compile.gatedSilu(normed, gate)
        }
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

    /// Depthwise conv then SiLU. `BONSAI2_GDN_CONV=1` fuses them. `ALL=off`
    /// stays on `silu(conv1d)`.
    private func convThenSilu(_ convInput: MLXArray) -> MLXArray {
        qwen35GdnConvSilu(convInput, weight: conv1d.weight, groups: conv1d.groups)
    }

    /// Dense `in_proj_b` and `in_proj_a`. `BONSAI2_GDN_AB=1` uses one matmul.
    private func projectedBA(_ inputs: MLXArray) -> (MLXArray, MLXArray) {
        if let pair = Bonsai2GdnAB.bAndA(b: inProjB, a: inProjA, x: inputs) {
            return pair
        }
        return (inProjB(inputs), inProjA(inputs))
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

    private func projectInputs(_ inputs: MLXArray, B: Int, S: Int) -> (
        qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray
    ) {
        guard prepareFusedInputProjection(), let fusedInProj else {
            // Packed qkv and z read the same activation through the same
            // transform; rotate it once. b and a stay full precision.
            if let shared = sharedHadamardProjections(inputs, [inProjQKV, inProjZ]) {
                let ba = projectedBA(inputs)
                return (
                    shared[0],
                    shared[1].reshaped(B, S, numVHeads, headVDim),
                    ba.0,
                    ba.1
                )
            }
            if let (qkvY, zY) = Bonsai2GdnQkvzOnce.qkvAndZ(
                qkv: inProjQKV, z: inProjZ, x: inputs, xIsHat: false)
            {
                let ba = projectedBA(inputs)
                return (
                    qkvY,
                    zY.reshaped(B, S, numVHeads, headVDim),
                    ba.0,
                    ba.1
                )
            }
            if let (qkvY, zY) = Bonsai2GdnQkvz.qkvAndZ(
                qkv: inProjQKV, z: inProjZ, x: inputs, xIsHat: false)
            {
                let ba = projectedBA(inputs)
                return (
                    qkvY,
                    zY.reshaped(B, S, numVHeads, headVDim),
                    ba.0,
                    ba.1
                )
            }
            let ba = projectedBA(inputs)
            return (
                inProjQKV(inputs),
                inProjZ(inputs).reshaped(B, S, numVHeads, headVDim),
                ba.0,
                ba.1
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

        if S == 1, mask == nil,
            let pref = Qwen35GdnT1Mega.prefix(
                qkv: qkv, a: a, b: b, convState: convState,
                convWeight: conv1d.weight, aLog: aLog, dtBias: dtBias,
                numKHeads: numKHeads, numVHeads: numVHeads,
                headKDim: headKDim, headVDim: headVDim,
                convKernelSize: convKernelSize)
        {
            var state = ssmState ?? MLXArray.zeros(
                [B, numVHeads, headVDim, headKDim], dtype: .float32)
            if state.dtype != .float32 { state = state.asType(.float32) }
            let (out, newSsm) = gatedDeltaKernel(
                q: pref.q, k: pref.k, v: pref.v,
                g: pref.g, beta: pref.beta, state: state)
            return (out: out, newConvState: pref.newConv, newSsmState: newSsm)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        let nKeep = convKernelSize - 1
        let newConvState = retainedConvTail(of: convInput, keeping: nKeep, chunkWidth: S)
        let convOut = convThenSilu(convInput)

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed: MLXArray
        let kNormed: MLXArray
        if let fused = Bonsai2QkRms.apply(q: q, k: k, invScale: invScale) {
            qNormed = fused.0
            kNormed = fused.1
        } else {
            let scales = derived.normScales(headKDim: headKDim, dtype: dtype)
            qNormed = MLXFast.rmsNorm(q, weight: scales.q, eps: 1e-6)
            kNormed = MLXFast.rmsNorm(k, weight: scales.k, eps: 1e-6)
        }

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
        let convOut = convThenSilu(convInput)

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
        tape: ArraysCache.PrefixReplayTape, committedRows: Int
    ) -> CBv2RecurrentLayerState {
        precondition(
            canReplayPrefix(tape: tape, committedRows: committedRows),
            "Qwen35 invalid compact recurrent prefix replay")
        let rows = 0 ..< committedRows
        let boundarySsm = qwen35GatedDelta(
            q: tape.q[0..., rows, 0...],
            k: tape.k[0..., rows, 0...],
            v: tape.v[0..., rows, 0...],
            a: tape.a[0..., rows, 0...],
            b: tape.b[0..., rows, 0...],
            aLog: aLog,
            dtBias: dtBias,
            state: tape.ssmPre,
            mask: tape.mask.map { $0[0..., rows] }
        ).1
        let boundaryConvView = tape.convInput[
            0...,
            committedRows ..< (committedRows + tape.convStateRows),
            0...]
        // A contiguous copy detaches the three retained rows from the whole
        // window's conv input, as the zero-add did, in one dispatch.
        let boundaryConv = contiguous(boundaryConvView)
        return CBv2RecurrentLayerState(conv: boundaryConv, ssm: boundarySsm)
    }

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
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)
        precondition(recurrentState.count == B, "Qwen35 CBv2 recurrent row count mismatch")

        let (qkv, z, b, a) = projectInputs(inputs, B: B, S: S)

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

        let normedOut = gatedNorm(out, gate: z)
        return projectOut(normedOut.reshaped(B, S, -1))
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
        exactTargetVerify: Bool = false
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
            (qkv, z, b, a) = projectInputs(inputs, B: B, S: S)
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
        let convInput = concatenated([convState, qkv], axis: 1)
        let convOut: MLXArray
        if exactTargetVerify, S > 1 {
            convOut = silu(concatenated(
                (0 ..< S).map { position in
                    conv1d(convInput[
                        0..., position ..< (position + convKernelSize), 0...])
                }, axis: 1))
        } else {
            convOut = convThenSilu(convInput)
        }

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let dtype = q.dtype
        let scales = derived.normScales(headKDim: headKDim, dtype: dtype)
        let qNormed = MLXFast.rmsNorm(q, weight: scales.q, eps: 1e-6)
        let kNormed = MLXFast.rmsNorm(k, weight: scales.k, eps: 1e-6)

        let out: MLXArray
        if S >= 3 {
            let recurrence = qwen35GatedDelta(
                q: qNormed,
                k: kNormed,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: ssmState,
                mask: nil)
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
                let convOutBacking = convOut[rowRange]
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
                                tape: tape, committedRows: keepPositions)
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
        let normedOut = gatedNorm(out, gate: z)
        let projectionInput = normedOut.reshaped(B, S, -1)
        if exactTargetVerify {
            return qwen35A3BExactW4G64Projection(outProj, projectionInput)
        }
        return projectOut(projectionInput)
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

        super.init()
    }

    /// q, k and v read the same activation. On a packed Hadamard checkpoint
    /// they share one input transform, so it is computed once.
    private func projectQKV(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
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

        return oProj(qwen35SigmoidGate(output, gate))
    }

    func cbv2Forward(
        _ x: MLXArray, cache: any CBv2AttendingLayerCache,
        positionIds: MLXArray? = nil,
        exactTargetVerify: Bool = false
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let projected: (MLXArray, MLXArray, MLXArray)
        if exactTargetVerify {
            projected = (
                qwen35A3BExactW4G64Projection(qProj, x),
                qwen35A3BExactW4G64Projection(kProj, x),
                qwen35A3BExactW4G64Projection(vProj, x)
            )
        } else {
            projected = projectQKV(x)
        }
        let qProjOutput = projected.0
        let kProjection = projected.1
        let vProjection = projected.2
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qNorm(qSplit[0]).transposed(0, 2, 1, 3)
        let gate = qSplit[1].reshaped(B, L, -1)
        var keys = kNorm(kProjection.reshaped(B, L, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let values = vProjection.reshaped(B, L, kvHeads, -1)
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

        let output = cache.updateAndAttend(
            queries: queries, keys: keys, values: values,
            scale: scale, sinks: nil)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
        let projectionInput = qwen35SigmoidGate(output, gate)
        if exactTargetVerify {
            return qwen35A3BExactW4G64Projection(oProj, projectionInput)
        }
        if let packed = oProj as? HadamardQuantizedLinear {
            // The residual add widens the FP16 product itself.
            return packed.forwardUnwidened(projectionInput)
        }
        return oProj(projectionInput)
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
            let signed = Qwen35FusedElementwise.swigluSigned(
                shared[0], shared[1], down.transform.signVector)
            return down.forwardPreSigned(signed, widenOutput: false)
        }
        guard let shared = sharedHadamardProjections(x, [gateProj, upProj]) else {
            return self(x)
        }
        return downProj(silu(shared[0]) * shared[1])
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

        if let fused = Bonsai2AddRms.sumAndNorm(x, r, norm: postAttentionLayerNorm) {
            if let dense = mlp as? Qwen3NextMLP {
                return fused.0 + dense.qwen35Forward(fused.1)
            }
            return fused.0 + (mlp as! UnaryLayer)(fused.1)
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
        exactTargetVerify: Bool = false
    ) -> MLXArray {
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
        let h = x + r
        let normalized = postAttentionLayerNorm(h)
        let feedForward: MLXArray
        if let sparse = mlp as? Qwen35SparseMoeBlock {
            feedForward = sparse(
                normalized, exactTargetVerify: exactTargetVerify)
        } else if let dense = mlp as? Qwen3NextMLP {
            feedForward = dense.qwen35TargetVerify(
                normalized, exact: exactTargetVerify)
        } else {
            preconditionFailure("Qwen35 decoder has an unsupported MLP module")
        }
        return h + feedForward
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

    func cbv2Forward(
        _ inputs: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray? = nil,
        captureRecurrentWindow: Bool = false
    ) -> MLXArray {
        precondition(
            caches.count == layers.filter({ !$0.isLinear }).count,
            "Qwen35 CBv2 requires only full-attention caches")
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
            hiddenStates = layer.cbv2Forward(
                hiddenStates,
                modelLayerIndex: modelLayerIndex,
                attentionCache: attentionCache,
                recurrentState: recurrentState,
                positionIds: positionIds,
                captureRecurrentWindow: captureRecurrentWindow,
                exactTargetVerify: captureRecurrentWindow && exactTargetVerify)
            // `hiddenStates` here IS the OUTPUT hidden state of this layer,
            // which is what the reference taps (`_LayerHook` wraps the layer and
            // keeps what it returned).
            if let tapLayerIds, let slot = tapLayerIds.firstIndex(of: modelLayerIndex) {
                tapped[slot] = hiddenStates
            }
        }
        if tapLayerIds == nil {
            dFlash2Tap.tappedHidden = nil
        } else {
            dFlash2Tap.tappedHidden = concatenated(tapped.map { $0! }, axis: -1)
        }
        return hiddenStates
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
        Bonsai2QmvS16Phase.arm(sequenceLength: Bonsai2QmvS16Phase.sequenceLength(inputs))
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
        positionedForward(
            tokens, inputEmbedding: nil, cache: caches,
            recurrentState: recurrentState, positionIds: nil)
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
        let normalized = model.norm(hidden)
        return lmHead.map { $0(normalized) } ?? model.embedTokens.asLinear(normalized)
    }
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
            recurrentState: recurrentState, positionIds: positionIds)
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
            recurrentState: recurrentState, positionIds: positionIds)
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
        lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
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

private func bonsai2ValveOff() -> Bool {
    guard let raw = getenv("BONSAI2_VALVE") else { return false }
    return String(cString: raw) == "ALL=off"
}

/// Full-attention `x * sigmoid(gate)` in one kernel. `ALL=off` keeps
/// `sigmoidMultiply`.
func qwen35SigmoidGate(_ x: MLXArray, _ gate: MLXArray) -> MLXArray {
    if let fused = Bonsai2AttnSigmoid.apply(x, gate) {
        return fused
    }
    return sigmoidMultiply(x, gate)
}

private final class Bonsai2AttnSigmoidKernel: Sendable {
    static let shared = Bonsai2AttnSigmoidKernel()
    let kernel: MLXFast.MLXFastKernel
    private init() {
        kernel = MLXFast.metalKernel(
            name: "bonsai2_attn_sigmoid",
            inputNames: ["x", "gate"],
            outputNames: ["y"],
            source: """
                uint i = thread_position_in_grid.x;
                float g = float(gate[i]);
                float s;
                if (g >= 0.0f) {
                    s = 1.0f / (1.0f + exp(-g));
                } else {
                    float e = exp(g);
                    s = e / (1.0f + e);
                }
                y[i] = static_cast<InT>(float(x[i]) * s);
                """,
            ensureRowContiguous: true)
    }
}

enum Bonsai2AttnSigmoid {
    nonisolated(unsafe) static var calls = 0
    static func isArmed() -> Bool {
        if bonsai2ValveOff() { return false }
        guard let raw = getenv("BONSAI2_ATTN_SIGMOID") else { return false }
        return String(cString: raw) == "1"
    }
    static func apply(_ x: MLXArray, _ gate: MLXArray) -> MLXArray? {
        guard isArmed(), x.shape == gate.shape, x.dtype == gate.dtype,
            x.dtype == .float32 || x.dtype == .bfloat16
        else { return nil }
        let n = x.size
        guard n > 0, n % 32 == 0 else { return nil }
        calls += 1
        return Bonsai2AttnSigmoidKernel.shared.kernel(
            [x, gate],
            template: [("InT", x.dtype)],
            grid: (n, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [x.dtype])[0]
    }
}

private final class Bonsai2GdnConvSiluKernel: Sendable {
    static let shared = Bonsai2GdnConvSiluKernel()
    let kernel: MLXFast.MLXFastKernel
    private init() {
        kernel = MLXFast.metalKernel(
            name: "bonsai2_gdn_conv_silu",
            inputNames: ["x", "weight"],
            outputNames: ["y"],
            source: """
                uint i = thread_position_in_grid.x;
                uint c = i % uint(C);
                uint t = (i / uint(C)) % uint(OutLen);
                uint b = i / (uint(C) * uint(OutLen));
                const device float* row = x + ((b * uint(InLen) + t) * uint(C) + c);
                const device float* w = weight + c * 4;
                float acc = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    acc += row[k * int(C)] * w[k];
                }
                float s;
                if (acc >= 0.0f) {
                    s = 1.0f / (1.0f + exp(-acc));
                } else {
                    float e = exp(acc);
                    s = e / (1.0f + e);
                }
                y[i] = acc * s;
                """,
            ensureRowContiguous: true)
    }
}

enum Bonsai2GdnConvSilu {
    static func isArmed() -> Bool {
        if bonsai2ValveOff() { return false }
        guard let raw = getenv("BONSAI2_GDN_CONV") else { return false }
        return String(cString: raw) == "1"
    }
    static func apply(_ input: MLXArray, weight: MLXArray, groups: Int) -> MLXArray? {
        guard isArmed(),
            input.ndim == 3, weight.ndim == 3,
            input.dtype == .float32, weight.dtype == .float32
        else { return nil }
        let channels = input.dim(2)
        let inLen = input.dim(1)
        guard channels > 0, channels % 32 == 0, channels == groups,
            weight.dim(0) == channels, weight.dim(1) == 4, weight.dim(2) == 1,
            inLen >= 4
        else { return nil }
        let outLen = inLen - 3
        let batch = input.dim(0)
        guard batch > 0, outLen > 0 else { return nil }
        return Bonsai2GdnConvSiluKernel.shared.kernel(
            [input, weight],
            template: [("C", channels), ("InLen", inLen), ("OutLen", outLen)],
            grid: (batch * outLen * channels, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[batch, outLen, channels]],
            outputDTypes: [.float32])[0]
    }
}

func qwen35GdnConvSilu(_ input: MLXArray, weight: MLXArray, groups: Int) -> MLXArray {
    if let fused = Bonsai2GdnConvSilu.apply(input, weight: weight, groups: groups) {
        return fused
    }
    return silu(conv1d(input, weight, stride: 1, padding: 0, dilation: 1, groups: groups))
}


// Darkbloom serving fusions. ALL=off stands each one down.

enum Bonsai2MlpCat {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_MLP_CAT") else { return false }
        return String(cString: raw) == "1"
    }

    static func rows(_ x: MLXArray) -> Int {
        if x.ndim >= 2 { return x.dim(x.ndim - 2) }
        return 1
    }

    static func gateUp(
        gate: HadamardQuantizedLinear, up: HadamardQuantizedLinear, hat: MLXArray
    ) -> (MLXArray, MLXArray)? {
        guard isArmed(), rows(hat) >= 64,
            gate.groupSize == up.groupSize, gate.bits == up.bits,
            gate.weight.dim(1) == up.weight.dim(1),
            let gateBias = gate.biases, let upBias = up.biases
        else { return nil }
        let both = quantizedMM(
            hat,
            concatenated([gate.weight, up.weight], axis: 0),
            scales: concatenated([gate.scales, up.scales], axis: 0),
            biases: concatenated([gateBias, upBias], axis: 0),
            transpose: true, groupSize: gate.groupSize, bits: gate.bits, mode: .affine)
        let half = both.dim(-1) / 2
        let flat = both.reshaped([-1, both.dim(-1)])
        var gateShape = both.shape
        var upShape = both.shape
        gateShape[gateShape.count - 1] = half
        upShape[upShape.count - 1] = both.dim(-1) - half
        return (
            flat[0..., 0..<half].reshaped(gateShape),
            flat[0..., half...].reshaped(upShape))
    }
}

/// One prefill qmm for gate and up, with the concatenated weight built once.
/// Decode rows stay on two qmv calls. Does not stack with `BONSAI2_MLP_CAT`.
/// Opt-in. `ALL=off` restores the two matmuls. Set `BONSAI2_MLP_CAT_ONCE=1`.

enum Bonsai2MlpCatOnce {
    private struct Packed {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var packed: [ObjectIdentifier: Packed] = [:]
    nonisolated(unsafe) static var buildCount = 0

    static func reset() {
        lock.withLock {
            packed.removeAll()
            buildCount = 0
        }
    }

    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_MLP_CAT_ONCE") else { return false }
        return String(cString: raw) == "1"
    }

    static func gateUp(
        gate: HadamardQuantizedLinear, up: HadamardQuantizedLinear, hat: MLXArray
    ) -> (MLXArray, MLXArray)? {
        guard isArmed(), !Bonsai2MlpCat.isArmed(), Bonsai2MlpCat.rows(hat) >= 64,
            gate.groupSize == up.groupSize, gate.bits == up.bits,
            gate.weight.dim(1) == up.weight.dim(1),
            let gateBias = gate.biases, let upBias = up.biases
        else { return nil }
        let key = ObjectIdentifier(gate)
        let hit = lock.withLock { packed[key] }
        let fused: Packed
        if let hit {
            fused = hit
        } else {
            let weight = concatenated([gate.weight, up.weight], axis: 0)
            let scales = concatenated([gate.scales, up.scales], axis: 0)
            let biases = concatenated([gateBias, upBias], axis: 0)
            eval(weight, scales, biases)
            let made = Packed(weight: weight, scales: scales, biases: biases)
            lock.withLock {
                if packed[key] == nil {
                    packed[key] = made
                    buildCount += 1
                }
            }
            fused = lock.withLock { packed[key] ?? made }
        }
        let both = quantizedMM(
            hat, fused.weight, scales: fused.scales, biases: fused.biases,
            transpose: true, groupSize: gate.groupSize, bits: gate.bits, mode: .affine)
        let half = both.dim(-1) / 2
        let flat = both.reshaped([-1, both.dim(-1)])
        var gateShape = both.shape
        var upShape = both.shape
        gateShape[gateShape.count - 1] = half
        upShape[upShape.count - 1] = both.dim(-1) - half
        return (
            flat[0..., 0..<half].reshaped(gateShape),
            flat[0..., half...].reshaped(upShape))
    }
}

/// Decode-only gate+up concat. One qmv keeps float16 scales when the fused
/// row count is 34816. Prefill stays two matmuls. The lm_head stays on the
/// cast. Opt-in. `ALL=off` restores the two matmuls. Set `BONSAI2_MLP_DEC_S16=1`.

enum Bonsai2MlpCatBoth {
    private struct Packed {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var packed: [ObjectIdentifier: Packed] = [:]
    nonisolated(unsafe) static var buildCount = 0

    static func reset() {
        lock.withLock {
            packed.removeAll()
            buildCount = 0
        }
    }

    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        if let raw = getenv("BONSAI2_MLP_CAT_BOTH"), String(cString: raw) == "1" {
            return true
        }
        if let raw = getenv("BONSAI2_MLP_CAT_S16"), String(cString: raw) == "1" {
            return true
        }
        return false
    }

    static func gateUp(
        gate: HadamardQuantizedLinear, up: HadamardQuantizedLinear, hat: MLXArray
    ) -> (MLXArray, MLXArray)? {
        let decodeOnly = Bonsai2MlpDecS16.isArmed() && Bonsai2MlpCat.rows(hat) == 1
        guard isArmed() || decodeOnly, !Bonsai2MlpCat.isArmed(), !Bonsai2MlpCatOnce.isArmed(),
            !Qwen35MlpGateUp.isArmed(), Bonsai2MlpCat.rows(hat) >= 1,
            gate.groupSize == up.groupSize, gate.bits == up.bits, gate.mode == up.mode,
            gate.weight.dim(0) == up.weight.dim(0),
            gate.weight.dim(1) == up.weight.dim(1),
            let gateBias = gate.biases, let upBias = up.biases
        else { return nil }
        let key = ObjectIdentifier(gate)
        let hit = lock.withLock { packed[key] }
        let fused: Packed
        if let hit {
            fused = hit
        } else {
            let weight = concatenated([gate.weight, up.weight], axis: 0)
            let scales = concatenated([gate.scales, up.scales], axis: 0)
            let biases = concatenated([gateBias, upBias], axis: 0)
            eval(weight, scales, biases)
            let made = Packed(weight: weight, scales: scales, biases: biases)
            lock.withLock {
                if packed[key] == nil {
                    packed[key] = made
                    buildCount += 1
                }
            }
            fused = lock.withLock { packed[key] ?? made }
        }
        let both = quantizedMM(
            hat, fused.weight, scales: fused.scales, biases: fused.biases,
            transpose: true, groupSize: gate.groupSize, bits: gate.bits, mode: gate.mode)
        let half = both.dim(-1) / 2
        let flat = both.reshaped([-1, both.dim(-1)])
        var gateShape = both.shape
        var upShape = both.shape
        gateShape[gateShape.count - 1] = half
        upShape[upShape.count - 1] = both.dim(-1) - half
        return (
            flat[0..., 0..<half].reshaped(gateShape),
            flat[0..., half...].reshaped(upShape))
    }
}

/// One prefill qmm for GDN `in_proj_qkv` and `in_proj_z`. The concatenated
/// weight is built once. Dense `in_proj_a` and `in_proj_b` stay separate.
/// Decode rows stay on two qmv calls. Does not stack with `BONSAI2_GDN_FUSE`.
/// Opt-in. `ALL=off` restores the two matmuls. Set `BONSAI2_GDN_QKVZ_ONCE=1`.

enum Bonsai2GdnQkvzOnce {
    private struct Packed {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var packed: [ObjectIdentifier: Packed] = [:]
    nonisolated(unsafe) static var buildCount = 0

    static func reset() {
        lock.withLock {
            packed.removeAll()
            buildCount = 0
        }
    }

    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_GDN_QKVZ_ONCE") else { return false }
        return String(cString: raw) == "1"
    }

    static func qkvAndZ(
        qkv qkvLinear: Linear, z zLinear: Linear, x: MLXArray, xIsHat: Bool
    ) -> (MLXArray, MLXArray)? {
        guard isArmed(), !Qwen35GdnInputFuse.isArmed(),
            Bonsai2MlpCat.rows(x) >= 64,
            let qkv = qkvLinear as? HadamardQuantizedLinear,
            let z = zLinear as? HadamardQuantizedLinear,
            ObjectIdentifier(type(of: qkv))
                == ObjectIdentifier(HadamardQuantizedLinear.self),
            ObjectIdentifier(type(of: z))
                == ObjectIdentifier(HadamardQuantizedLinear.self),
            qkv.gdnLayout == nil, z.gdnLayout == nil,
            qkv.groupSize == z.groupSize, qkv.bits == z.bits, qkv.mode == z.mode,
            qkv.weight.dim(1) == z.weight.dim(1),
            qkv.transform.width == z.transform.width,
            qkv.transform.blockSize == z.transform.blockSize,
            let qkvBias = qkv.biases, let zBias = z.biases
        else { return nil }
        let key = ObjectIdentifier(qkv)
        let hit = lock.withLock { packed[key] }
        let fused: Packed
        if let hit {
            fused = hit
        } else {
            let weight = concatenated([qkv.weight, z.weight], axis: 0)
            let scales = concatenated([qkv.scales, z.scales], axis: 0)
            let biases = concatenated([qkvBias, zBias], axis: 0)
            eval(weight, scales, biases)
            let made = Packed(weight: weight, scales: scales, biases: biases)
            lock.withLock {
                if packed[key] == nil {
                    packed[key] = made
                    buildCount += 1
                }
            }
            fused = lock.withLock { packed[key] ?? made }
        }
        let hat = xIsHat ? Bonsai2HatF32.share(x) : Bonsai2HatF32.share(qkv.rotate(x))
        let both = quantizedMM(
            hat, fused.weight, scales: fused.scales, biases: fused.biases,
            transpose: true, groupSize: qkv.groupSize, bits: qkv.bits, mode: qkv.mode)
        let qkvN = qkv.weight.dim(0)
        let flat = both.reshaped([-1, both.dim(-1)])
        var qkvShape = both.shape
        var zShape = both.shape
        qkvShape[qkvShape.count - 1] = qkvN
        zShape[zShape.count - 1] = both.dim(-1) - qkvN
        return (
            flat[0..., 0..<qkvN].reshaped(qkvShape),
            flat[0..., qkvN...].reshaped(zShape))
    }
}

/// Decode-only concat of GDN `in_proj_qkv` and `in_proj_z`. Prefill stays
/// two projections. Dense `in_proj_a` and `in_proj_b` stay separate.
/// Opt-in. `ALL=off` restores the two matmuls. Set `BONSAI2_GDN_QKVZ_DEC=1`.

enum Bonsai2GdnQkvzDec {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        if Bonsai2GdnQkvz.isArmed() || Bonsai2GdnQkvzOnce.isArmed() { return false }
        guard let raw = getenv("BONSAI2_GDN_QKVZ_DEC") else { return false }
        return String(cString: raw) == "1"
    }
}

/// One qmm or qmv for GDN `in_proj_qkv` and `in_proj_z` at every row count.
/// Prefill is one qmm. Decode is one qmv. The concatenated weight is built
/// once. Dense `in_proj_a` and `in_proj_b` stay separate. Does not stack
/// with `BONSAI2_GDN_FUSE` or `BONSAI2_GDN_QKVZ_ONCE`. Opt-in. `ALL=off`
/// restores the two matmuls. Set `BONSAI2_GDN_QKVZ=1`.

enum Bonsai2GdnQkvz {
    private struct Packed {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var packed: [ObjectIdentifier: Packed] = [:]
    nonisolated(unsafe) static var buildCount = 0

    static func reset() {
        lock.withLock {
            packed.removeAll()
            buildCount = 0
        }
    }

    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_GDN_QKVZ") else { return false }
        return String(cString: raw) == "1"
    }

    static func qkvAndZ(
        qkv qkvLinear: Linear, z zLinear: Linear, x: MLXArray, xIsHat: Bool
    ) -> (MLXArray, MLXArray)? {
        let decodeOnly = Bonsai2GdnQkvzDec.isArmed() && Bonsai2MlpCat.rows(x) == 1
        guard isArmed() || decodeOnly, !Qwen35GdnInputFuse.isArmed(),
            !Bonsai2GdnQkvzOnce.isArmed(),
            Bonsai2MlpCat.rows(x) >= 1,
            let qkv = qkvLinear as? HadamardQuantizedLinear,
            let z = zLinear as? HadamardQuantizedLinear,
            ObjectIdentifier(type(of: qkv))
                == ObjectIdentifier(HadamardQuantizedLinear.self),
            ObjectIdentifier(type(of: z))
                == ObjectIdentifier(HadamardQuantizedLinear.self),
            qkv.gdnLayout == nil, z.gdnLayout == nil,
            qkv.groupSize == z.groupSize, qkv.bits == z.bits, qkv.mode == z.mode,
            qkv.weight.dim(1) == z.weight.dim(1),
            qkv.transform.width == z.transform.width,
            qkv.transform.blockSize == z.transform.blockSize,
            let qkvBias = qkv.biases, let zBias = z.biases
        else { return nil }
        let key = ObjectIdentifier(qkv)
        let hit = lock.withLock { packed[key] }
        let fused: Packed
        if let hit {
            fused = hit
        } else {
            let weight = concatenated([qkv.weight, z.weight], axis: 0)
            let scales = concatenated([qkv.scales, z.scales], axis: 0)
            let biases = concatenated([qkvBias, zBias], axis: 0)
            eval(weight, scales, biases)
            let made = Packed(weight: weight, scales: scales, biases: biases)
            lock.withLock {
                if packed[key] == nil {
                    packed[key] = made
                    buildCount += 1
                }
            }
            fused = lock.withLock { packed[key] ?? made }
        }
        let hat = xIsHat ? Bonsai2HatF32.share(x) : Bonsai2HatF32.share(qkv.rotate(x))
        let both = quantizedMM(
            hat, fused.weight, scales: fused.scales, biases: fused.biases,
            transpose: true, groupSize: qkv.groupSize, bits: qkv.bits, mode: qkv.mode)
        let qkvN = qkv.weight.dim(0)
        let flat = both.reshaped([-1, both.dim(-1)])
        var qkvShape = both.shape
        var zShape = both.shape
        qkvShape[qkvShape.count - 1] = qkvN
        zShape[zShape.count - 1] = both.dim(-1) - qkvN
        return (
            flat[0..., 0..<qkvN].reshaped(qkvShape),
            flat[0..., qkvN...].reshaped(zShape))
    }
}

/// Float16-scale qmv on decode steps only. Prefill leaves the live flag
/// unset, so the 1024-token forward, including its one-row lm-head, keeps
/// the float cast. Opt-in. `ALL=off` restores the cast. Set
/// `BONSAI2_QMV_S16_PHASE=1`.

enum Bonsai2QmvS16Phase {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_QMV_S16_PHASE") else { return false }
        return String(cString: raw) == "1"
    }

    static func sequenceLength(_ inputs: MLXArray) -> Int {
        if inputs.ndim >= 2 { return inputs.dim(1) }
        return inputs.dim(0)
    }

    static func arm(sequenceLength: Int) {
        if isArmed(), sequenceLength == 1 {
            setenv("BONSAI2_QMV_S16_NOW", "1", 1)
        } else {
            unsetenv("BONSAI2_QMV_S16_NOW")
        }
    }
}

/// One snapshot of the C++ `BONSAI2_*` flags per forward.
/// `getenv` stays the stock path. Opt-in. `ALL=off` clears the snapshot.
/// Set `BONSAI2_ENV_CACHE=1`.

enum Bonsai2Valve {
    static func allOff() -> Bool {
        guard let raw = getenv("BONSAI2_VALVE") else { return false }
        return String(cString: raw) == "ALL=off"
    }

    static func envOff(_ name: String) -> Bool {
        guard let raw = getenv(name) else { return false }
        return String(cString: raw) == "0"
    }
}

/// Slice hidden to the last token before the vocab GEMM.
/// Default on; `ALL=off` or `BONSAI2_LAST_TOKEN_HEAD=0` restores full-row
/// RMS + lm-head. CBv2 prefill already narrows via lastPositionLogits;
/// this covers generate() and positionedForward when S>1. T=1 is a no-op.

public enum Bonsai2B1RecurrentRow {
    public static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        return true
    }

    public static func row(_ state: MLXArray, index: Int, batch: Int) -> MLXArray {
        if isArmed(), batch == 1 { return state }
        return state[index ..< index + 1]
    }
}

/// Same-input Hadamard (one FWHT then N qmm) for every sequence length.
/// Default on; `ALL=off` or `BONSAI2_SHARED_HAT=0` restores split FWHT.

public enum Bonsai2SharedHat {
    public static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        if Bonsai2Valve.envOff("BONSAI2_SHARED_HAT") { return false }
        return true
    }
}

/// Cmlx Kconst qmv_fast: prefetch next weight K-block while accumulating.
/// Opt-in: quiet pair-sum P 0.94 D 0.81. Extra-template `_wp` plus
/// extra registers. `ALL=off` restores the rolled K-loop.

enum Qwen35GdnInputFuse {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_GDN_FUSE") else { return false }
        return String(cString: raw) == "1"
    }
}

/// One dense matmul for GDN `in_proj_b` and `in_proj_a` at every row count.
/// The concatenated weight is built once. Quantized `in_proj_qkv` and
/// `in_proj_z` stay on their own path. Does not stack with `BONSAI2_GDN_FUSE`.
/// Opt-in. `ALL=off` restores the two matmuls. Set `BONSAI2_GDN_AB=1`.

enum Bonsai2GdnAB {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var packed: [ObjectIdentifier: MLXArray] = [:]
    nonisolated(unsafe) static var buildCount = 0

    static func reset() {
        lock.withLock {
            packed.removeAll()
            buildCount = 0
        }
    }

    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_GDN_AB") else { return false }
        return String(cString: raw) == "1"
    }

    static func bAndA(b: Linear, a: Linear, x: MLXArray) -> (MLXArray, MLXArray)? {
        guard isArmed(), !Qwen35GdnInputFuse.isArmed(), Bonsai2MlpCat.rows(x) >= 1,
            !(b is QuantizedLinear), !(a is QuantizedLinear),
            b.bias == nil, a.bias == nil,
            b.weight.dim(1) == a.weight.dim(1),
            b.weight.dtype == a.weight.dtype
        else { return nil }
        let key = ObjectIdentifier(b)
        let hit = lock.withLock { packed[key] }
        let weight: MLXArray
        if let hit {
            weight = hit
        } else {
            let made = concatenated([b.weight, a.weight], axis: 0)
            eval(made)
            lock.withLock {
                if packed[key] == nil {
                    packed[key] = made
                    buildCount += 1
                }
            }
            weight = lock.withLock { packed[key] ?? made }
        }
        let both = matmul(x, weight.T)
        let bN = b.weight.dim(0)
        let flat = both.reshaped([-1, both.dim(-1)])
        var bShape = both.shape
        var aShape = both.shape
        bShape[bShape.count - 1] = bN
        aShape[aShape.count - 1] = both.dim(-1) - bN
        return (
            flat[0..., 0..<bN].reshaped(bShape),
            flat[0..., bN...].reshaped(aShape))
    }
}

/// Decode-only float16-scale gate+up concat. Opt-in. `ALL=off` stands down.
enum Bonsai2MlpDecS16 {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        if Bonsai2MlpCatBoth.isArmed() { return false }
        guard let raw = getenv("BONSAI2_MLP_DEC_S16") else { return false }
        return String(cString: raw) == "1"
    }
}

/// One fused gate+up linear. Opt-in. `ALL=off` stands down.
enum Qwen35MlpGateUp {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_MLP_FUSE") else { return false }
        return String(cString: raw) == "1"
    }
}

/// One q/k/v attention projection. Opt-in. `ALL=off` stands down.
enum Qwen35AttnQkvFuse {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_ATTN_FUSE") else { return false }
        return String(cString: raw) == "1"
    }
}

/// Concatenate attention q/k/v Hadamard linears (Bonsai N=14336).
/// Opt-in: quiet pair-sum P 0.91 D 0.96. ALL=off restores split projections.

enum Bonsai2AttnKvDec {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        if Bonsai2AttnKV.isArmed() { return false }
        if Qwen35AttnQkvFuse.isArmed() { return false }
        guard let raw = getenv("BONSAI2_ATTN_KV_DEC") else { return false }
        return String(cString: raw) == "1"
    }
}

/// One qmm or qmv for attention `k_proj` and `v_proj` at every row count.
/// Prefill is one qmm. Decode is one qmv. The concatenated weight is built
/// once. `q_proj` stays a separate matmul. Does not stack with
/// `BONSAI2_ATTN_FUSE`. Opt-in. `ALL=off` restores the two matmuls.
/// Set `BONSAI2_ATTN_KV=1`.

enum Bonsai2AttnKV {
    private struct Packed {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var packed: [ObjectIdentifier: Packed] = [:]
    nonisolated(unsafe) static var buildCount = 0

    static func reset() {
        lock.withLock {
            packed.removeAll()
            buildCount = 0
        }
    }

    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_ATTN_KV") else { return false }
        return String(cString: raw) == "1"
    }

    static func kv(
        k kLinear: HadamardQuantizedLinear, v vLinear: HadamardQuantizedLinear, hat: MLXArray
    ) -> (MLXArray, MLXArray)? {
        let decodeOnly = Bonsai2AttnKvDec.isArmed() && Bonsai2MlpCat.rows(hat) == 1
        guard isArmed() || decodeOnly, !Qwen35AttnQkvFuse.isArmed(),
            Bonsai2MlpCat.rows(hat) >= 1,
            kLinear.gdnLayout == nil, vLinear.gdnLayout == nil,
            kLinear.groupSize == vLinear.groupSize, kLinear.bits == vLinear.bits,
            kLinear.mode == vLinear.mode,
            kLinear.weight.dim(1) == vLinear.weight.dim(1),
            kLinear.transform.width == vLinear.transform.width,
            kLinear.transform.blockSize == vLinear.transform.blockSize,
            let kBias = kLinear.biases, let vBias = vLinear.biases
        else { return nil }
        let key = ObjectIdentifier(kLinear)
        let hit = lock.withLock { packed[key] }
        let fused: Packed
        if let hit {
            fused = hit
        } else {
            let weight = concatenated([kLinear.weight, vLinear.weight], axis: 0)
            let scales = concatenated([kLinear.scales, vLinear.scales], axis: 0)
            let biases = concatenated([kBias, vBias], axis: 0)
            eval(weight, scales, biases)
            let made = Packed(weight: weight, scales: scales, biases: biases)
            lock.withLock {
                if packed[key] == nil {
                    packed[key] = made
                    buildCount += 1
                }
            }
            fused = lock.withLock { packed[key] ?? made }
        }
        let both = quantizedMM(
            hat, fused.weight, scales: fused.scales, biases: fused.biases,
            transpose: true, groupSize: kLinear.groupSize, bits: kLinear.bits,
            mode: kLinear.mode)
        let kN = kLinear.weight.dim(0)
        let flat = both.reshaped([-1, both.dim(-1)])
        var kShape = both.shape
        var vShape = both.shape
        kShape[kShape.count - 1] = kN
        vShape[vShape.count - 1] = both.dim(-1) - kN
        return (
            flat[0..., 0..<kN].reshaped(kShape),
            flat[0..., kN...].reshaped(vShape))
    }
}

/// NSG=1 qmv_fast with 1 row/TG on large-N nconst kernels (N≥4096).
/// Opt-in: quiet pair-sum P 0.92 D 0.73. Occupancy ladder is 4 > 2 > 1
/// rows/TG on this pack. `ALL=off` restores 4 rows.

public enum Bonsai2GdnBlocked {
    public static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_GDN_BLOCKED") else { return false }
        return String(cString: raw) == "1"
    }
}

/// Prefill chunk size. The scored POC also sets `soloPrefillStripeTokens`
/// to 4096, and that stripe wins whenever it is larger than this chunk.
/// A 1024-token prompt is therefore one forward at chunk 512 and at chunk
/// 1024. Opt-in. `ALL=off` keeps 512.
/// Set `BONSAI2_PREFILL_CHUNK=1024`.

public enum Bonsai2LayerAsync {
    public static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_LAYER_ASYNC") else { return false }
        return String(cString: raw) == "1"
    }

    @discardableResult
    public static func submit(_ hidden: MLXArray) -> MLXArray {
        if isArmed(), hidden.ndim >= 2, hidden.dim(1) == 1 {
            asyncEval(hidden)
        }
        return hidden
    }
}


public enum Bonsai2AddRms {
    public static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_ADD_RMS") else { return false }
        return String(cString: raw) == "1"
    }

    /// Residual sum and the RMSNorm of that sum. Nil keeps the stock pair.
    public static func sumAndNorm(
        _ x: MLXArray, _ residual: MLXArray, norm: RMSNorm
    ) -> (MLXArray, MLXArray)? {
        guard isArmed() else { return nil }
        return evaluate(x, residual, norm: norm)
    }

    /// Same kernel as `sumAndNorm`, without the `BONSAI2_ADD_RMS` check.
    static func evaluate(
        _ x: MLXArray, _ residual: MLXArray, norm: RMSNorm
    ) -> (MLXArray, MLXArray)? {
        guard x.shape == residual.shape,
            x.dtype == residual.dtype,
            x.dtype == .float32 || x.dtype == .bfloat16,
            norm.weight.dtype == .float32 || norm.weight.dtype == x.dtype,
            x.ndim >= 1
        else { return nil }
        let axis = x.dim(-1)
        guard axis > 4096, axis % 4 == 0, norm.weight.dim(0) == axis, axis > 0,
            x.size % axis == 0
        else { return nil }
        let rows = x.size / axis
        guard rows > 0 else { return nil }
        let outType: DType = (x.dtype == .bfloat16 && norm.weight.dtype == .float32)
            ? .float32 : x.dtype
        let weight = norm.weight.asType(outType)
        let outputs = Bonsai2AddRmsKernel.shared.kernel(
            [x, residual, weight, MLXArray([norm.eps]), MLXArray([Int32(axis)])],
            template: [("InT", x.dtype), ("OutT", outType)],
            grid: (1024, rows, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [x.shape, x.shape],
            outputDTypes: [x.dtype, outType])
        return (outputs[0], outputs[1])
    }
}

/// Fuse one decoder layer's `h + feedForward` with the next layer's input
/// RMSNorm. Both the prefill rows and the decode row take the same kernel.
/// `ALL=off` keeps the separate add and RMSNorm. Set `BONSAI2_OUT_ADD_RMS=1`.

public enum Bonsai2OutAddRms {
    public static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_OUT_ADD_RMS") else { return false }
        return String(cString: raw) == "1"
    }

    nonisolated(unsafe) private static var stash: (MLXArray, MLXArray)?

    public static func reset() {
        stash = nil
    }

    public static func remember(_ h: MLXArray, _ feedForward: MLXArray) {
        guard isArmed() else {
            stash = nil
            return
        }
        stash = (h, feedForward)
    }

    public static func take() -> (MLXArray, MLXArray)? {
        guard isArmed() else {
            stash = nil
            return nil
        }
        let held = stash
        stash = nil
        return held
    }

    /// `(h + feedForward, rms(h + feedForward))`. Nil keeps the stock pair.
    public static func boundary(
        _ h: MLXArray, _ feedForward: MLXArray, norm: RMSNorm
    ) -> (MLXArray, MLXArray)? {
        guard isArmed() else { return nil }
        return Bonsai2AddRms.evaluate(h, feedForward, norm: norm)
    }
}

private final class Bonsai2AddRmsKernel: Sendable {
    static let shared = Bonsai2AddRmsKernel()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "bonsai2_add_rms_looped",
            inputNames: ["x", "res", "w", "eps", "axisBuf"],
            outputNames: ["h", "out"],
            source: bonsai2AddRmsLoopedSource,
            ensureRowContiguous: true)
    }
}

/// `rms_looped` with the residual add on the load. 1024 threads, 4-wide
/// reads, 32 simdgroups. Squares use the rounded sum, then the same
/// precise rsqrt and weight multiply as the stock kernel.
private let bonsai2AddRmsLoopedSource = """
    constexpr int N_READS = 4;
    constexpr int SIMD_SIZE = 32;
    uint gid = threadgroup_position_in_grid.y;
    uint lid = thread_index_in_threadgroup;
    uint lsize = threads_per_threadgroup.x;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;
    uint axis = uint(axisBuf[0]);
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];

    float acc = 0;
    const device InT* xp = x + gid * size_t(axis) + lid * N_READS;
    const device InT* rp = res + gid * size_t(axis) + lid * N_READS;
    device InT* hp = h + gid * size_t(axis) + lid * N_READS;
    for (uint r = 0; r < axis; r += lsize * N_READS) {
      if (r + lid * N_READS + N_READS <= axis) {
        for (int i = 0; i < N_READS; i++) {
          InT hi = static_cast<InT>(float(xp[i + r]) + float(rp[i + r]));
          hp[i + r] = hi;
          float xi = float(hi);
          acc += xi * xi;
        }
      } else {
        for (int i = 0; i < N_READS; i++) {
          if ((r + lid * N_READS + i) < axis) {
            InT hi = static_cast<InT>(float(xp[i + r]) + float(rp[i + r]));
            hp[i + r] = hi;
            float xi = float(hi);
            acc += xi * xi;
          }
        }
      }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
      local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
      local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / float(axis) + eps[0]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device OutT* op = out + gid * size_t(axis) + lid * N_READS;
    const device OutT* wp = w + lid * N_READS;
    for (uint r = 0; r < axis; r += lsize * N_READS) {
      if (r + lid * N_READS + N_READS <= axis) {
        for (int i = 0; i < N_READS; i++) {
          op[r + i] = wp[r + i] * static_cast<OutT>(hp[r + i] * local_inv_mean[0]);
        }
      } else {
        for (int i = 0; i < N_READS; i++) {
          if ((r + lid * N_READS + i) < axis) {
            op[r + i] = wp[r + i] * static_cast<OutT>(hp[r + i] * local_inv_mean[0]);
          }
        }
      }
    }
    """

/// Concatenate dense MLP gate+up Hadamard linears. ALL=off stands down.
/// Default off: N=2×intermediate (34816) page-faults qmv on this pack.

enum Qwen35GdnT1Mega {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        if Bonsai2Valve.envOff("BONSAI2_GDN_MEGA") { return false }
        return true
    }

    static func prefix(
        qkv: MLXArray,
        a: MLXArray,
        b: MLXArray,
        convState: MLXArray,
        convWeight: MLXArray,
        aLog: MLXArray,
        dtBias: MLXArray,
        numKHeads: Int,
        numVHeads: Int,
        headKDim: Int,
        headVDim: Int,
        convKernelSize: Int
    ) -> (
        q: MLXArray, k: MLXArray, v: MLXArray, newConv: MLXArray,
        g: MLXArray, beta: MLXArray
    )? {
        guard isArmed(),
            qkv.ndim == 3, qkv.dim(1) == 1,
            convKernelSize == 4,
            headKDim >= 32, headKDim % 32 == 0, headKDim <= 128,
            headVDim >= 4, headVDim % 4 == 0,
            numVHeads > 0, numKHeads > 0, numVHeads % numKHeads == 0
        else { return nil }

        let B = qkv.dim(0)
        let keyDim = headKDim * numKHeads
        let valueDim = headVDim * numVHeads
        let convDim = keyDim * 2 + valueDim
        guard qkv.dim(2) == convDim,
            qkv.dtype == convState.dtype,
            convState.shape == [B, convKernelSize - 1, convDim],
            convWeight.dim(0) == convDim, convWeight.dim(1) == convKernelSize
        else { return nil }

        let weight2 = convWeight.ndim >= 3 && convWeight.dim(2) == 1
            ? convWeight.reshaped([convDim, convKernelSize]) : convWeight
        let a2 = a.reshaped([B, numVHeads]).asType(.float32)
        let b2 = b.reshaped([B, numVHeads]).asType(.float32)
        let inType = qkv.dtype
        let outputs = Qwen35GdnT1PrefixManager.shared.kernel(
            [
                qkv, convState, weight2, a2, b2,
                aLog.asType(.float32), dtBias.asType(.float32),
            ],
            template: [
                ("InT", inType),
                ("Dk", headKDim),
                ("Dv", headVDim),
                ("Hk", numKHeads),
                ("Hv", numVHeads),
                ("Ksize", convKernelSize),
            ],
            grid: (128, numKHeads, B),
            threadGroup: (128, 1, 1),
            outputShapes: [
                [B, 1, numKHeads, headKDim],
                [B, 1, numKHeads, headKDim],
                [B, 1, numVHeads, headVDim],
                [B, convKernelSize - 1, convDim],
                [B, 1, numVHeads],
                [B, 1, numVHeads],
            ],
            outputDTypes: [inType, inType, inType, convState.dtype, .float32, .float32])
        return (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5])
    }

    static func gatedNorm(
        _ out: MLXArray, gate: MLXArray, weight: MLXArray, eps: Float
    ) -> MLXArray? {
        guard isArmed(),
            out.ndim == 4, out.dim(1) == 1,
            gate.shape == out.shape,
            weight.ndim == 1, weight.dim(0) == out.dim(3)
        else { return nil }
        let B = out.dim(0)
        let Hv = out.dim(2)
        let Dv = out.dim(3)
        guard Dv >= 32, Dv % 32 == 0, Dv <= 128 else { return nil }
        let outputs = Qwen35GdnT1GnormManager.shared.kernel(
            [out, gate, weight.asType(.float32), MLXArray([eps])],
            template: [("InT", out.dtype), ("Dv", Dv), ("Hv", Hv)],
            grid: (32, B * Hv, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [out.shape],
            outputDTypes: [out.dtype])
        return outputs[0]
    }

    static func siluProduct(_ gate: MLXArray, _ up: MLXArray) -> MLXArray? {
        guard isArmed(), gate.shape == up.shape, gate.dtype == up.dtype else { return nil }
        if gate.ndim >= 3, gate.dim(gate.ndim - 2) != 1 { return nil }
        if gate.ndim == 2, gate.dim(0) != 1 { return nil }
        let n = gate.size
        guard n > 0, n % 32 == 0 else { return nil }
        let outputs = Qwen35T1SiluManager.shared.kernel(
            [gate, up],
            template: [("InT", gate.dtype)],
            grid: (n, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [gate.shape],
            outputDTypes: [gate.dtype])
        return outputs[0]
    }
}

private final class Qwen35GdnT1PrefixManager: Sendable {
    static let shared = Qwen35GdnT1PrefixManager()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "bonsai2_gdn_t1_prefix_gates",
            inputNames: ["qkv", "convState", "convW", "a", "b", "aLog", "dtBias"],
            outputNames: ["q", "k", "v", "conv_out", "g", "beta"],
            source: qwen35GdnT1PrefixSource,
            ensureRowContiguous: true)
    }
}

private final class Qwen35GdnT1GnormManager: Sendable {
    static let shared = Qwen35GdnT1GnormManager()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "bonsai2_gdn_t1_gnorm",
            inputNames: ["yIn", "gate", "normW", "eps"],
            outputNames: ["yOut"],
            source: qwen35GdnT1GnormSource,
            ensureRowContiguous: true)
    }
}

private final class Qwen35T1SiluManager: Sendable {
    static let shared = Qwen35T1SiluManager()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "bonsai2_t1_silu_prod",
            inputNames: ["gate", "up"],
            outputNames: ["y"],
            source: """
                uint i = thread_position_in_grid.x;
                float gv = float(gate[i]);
                float uv = float(up[i]);
                float silu = gv * (1.0f / (1.0f + exp(-gv)));
                y[i] = static_cast<InT>(silu * uv);
                """,
            ensureRowContiguous: true)
    }
}

/// One threadgroup per key-head. Depthwise T=1 conv+SiLU, QK RMS, and
/// logaddexp GDN gates. Same mean-of-squares + precise rsqrt as rms_norm.
private let qwen35GdnT1PrefixSource = """
    constexpr int nKeep = Ksize - 1;
    const int convDim = 2 * Hk * Dk + Hv * Dv;
    const int keyDim = Hk * Dk;
    const int repeat = Hv / Hk;

    auto tid = thread_index_in_threadgroup;
    auto hk = threadgroup_position_in_grid.y;
    auto b_idx = threadgroup_position_in_grid.z;
    auto simd_gid = simdgroup_index_in_threadgroup;
    auto simd_lid = thread_index_in_simdgroup;

    const size_t bConv = (size_t)b_idx * nKeep * convDim;
    const size_t bQkv = (size_t)b_idx * convDim;

    if (hk == 0) {
      for (int c = tid; c < convDim; c += 128) {
        for (int r = 0; r < nKeep - 1; r++) {
          conv_out[bConv + (size_t)r * convDim + c] =
              convState[bConv + (size_t)(r + 1) * convDim + c];
        }
        conv_out[bConv + (size_t)(nKeep - 1) * convDim + c] = qkv[bQkv + c];
      }
    }

    threadgroup float q_s[128];
    threadgroup float k_s[128];
    threadgroup float rms_scratch[8];
    threadgroup float qk_inv[2];

    for (int i = tid; i < Dk; i += 128) {
      int qch = int(hk) * Dk + i;
      int kch = keyDim + int(hk) * Dk + i;
      float qacc = 0.0f;
      float kacc = 0.0f;
      for (int t = 0; t < Ksize; t++) {
        float qx = t < nKeep
            ? float(convState[bConv + (size_t)t * convDim + qch])
            : float(qkv[bQkv + qch]);
        float kx = t < nKeep
            ? float(convState[bConv + (size_t)t * convDim + kch])
            : float(qkv[bQkv + kch]);
        qacc += float(convW[(size_t)qch * Ksize + t]) * qx;
        kacc += float(convW[(size_t)kch * Ksize + t]) * kx;
      }
      q_s[i] = qacc * (1.0f / (1.0f + exp(-qacc)));
      k_s[i] = kacc * (1.0f / (1.0f + exp(-kacc)));
    }
    for (int r = 0; r < repeat; r++) {
      int hv = int(hk) * repeat + r;
      for (int d = tid; d < Dv; d += 128) {
        int vch = 2 * keyDim + hv * Dv + d;
        float vacc = 0.0f;
        for (int t = 0; t < Ksize; t++) {
          float vx = t < nKeep
              ? float(convState[bConv + (size_t)t * convDim + vch])
              : float(qkv[bQkv + vch]);
          vacc += float(convW[(size_t)vch * Ksize + t]) * vx;
        }
        float silu = vacc * (1.0f / (1.0f + exp(-vacc)));
        v[(size_t)b_idx * Hv * Dv + hv * Dv + d] = static_cast<InT>(silu);
      }
      if (tid == 0) {
        float a_h = float(a[(size_t)b_idx * Hv + hv]);
        float b_h = float(b[(size_t)b_idx * Hv + hv]);
        float alog = float(aLog[hv]);
        float dt = float(dtBias[hv]);
        float xdt = a_h + dt;
        float mx = xdt > 0.0f ? xdt : 0.0f;
        float sp = mx + log(exp(xdt - mx) + exp(-mx));
        g[(size_t)b_idx * Hv + hv] = exp(-exp(alog) * sp);
        beta[(size_t)b_idx * Hv + hv] = 1.0f / (1.0f + exp(-b_h));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qacc = 0.0f;
    float kacc = 0.0f;
    for (int i = tid; i < Dk; i += 128) {
      float qv = q_s[i];
      float kv = k_s[i];
      qacc += qv * qv;
      kacc += kv * kv;
    }
    qacc = simd_sum(qacc);
    kacc = simd_sum(kacc);
    if (simd_lid == 0) {
      rms_scratch[simd_gid] = qacc;
      rms_scratch[simd_gid + 4] = kacc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float qt = rms_scratch[0] + rms_scratch[1] + rms_scratch[2] + rms_scratch[3];
      float kt = rms_scratch[4] + rms_scratch[5] + rms_scratch[6] + rms_scratch[7];
      qk_inv[0] = metal::precise::rsqrt(qt / float(Dk) + 1e-6f);
      qk_inv[1] = metal::precise::rsqrt(kt / float(Dk) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_scale = qk_inv[0] / float(Dk);
    float k_scale = qk_inv[1] * metal::precise::rsqrt(float(Dk));
    for (int i = tid; i < Dk; i += 128) {
      q[(size_t)b_idx * Hk * Dk + hk * Dk + i] =
          static_cast<InT>(q_s[i] * q_scale);
      k[(size_t)b_idx * Hk * Dk + hk * Dk + i] =
          static_cast<InT>(k_s[i] * k_scale);
    }
    """

/// T=1 gated RMSNorm + SiLU. One simdgroup per value-head. RMS over Dv
/// matches mlx_fast.rms_norm (mean of squares, precise rsqrt, times weight).
private let qwen35GdnT1GnormSource = """
    auto tid = thread_index_in_threadgroup;
    auto n = threadgroup_position_in_grid.y;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    const size_t base = ((size_t)b_idx * Hv + hv_idx) * Dv;
    threadgroup float inv[1];

    float acc = 0.0f;
    for (int d = tid; d < Dv; d += 32) {
      float v = float(yIn[base + d]);
      acc += v * v;
    }
    acc = simd_sum(acc);
    if (tid == 0) {
      inv[0] = metal::precise::rsqrt(acc / float(Dv) + float(eps[0]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = tid; d < Dv; d += 32) {
      float yv = float(yIn[base + d]);
      float nrm = yv * inv[0] * float(normW[d]);
      float gz = float(gate[base + d]);
      float silu = gz * (1.0f / (1.0f + exp(-gz)));
      yOut[base + d] = static_cast<InT>(silu * nrm);
    }
    """

/// Prefill gated RMSNorm + SiLU. One simdgroup per `[B, T, Hv]` row.
/// Same mean-of-squares, precise rsqrt, weight, and SiLU as the T=1 kernel.
/// T=1 stays on that kernel. Opt-in. `ALL=off` restores `norm(out, gate:)`.
/// Set `BONSAI2_GDN_GNORM=1`.
enum Bonsai2GdnGnorm {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_GDN_GNORM") else { return false }
        return String(cString: raw) == "1"
    }

    static func apply(
        _ out: MLXArray, gate: MLXArray, weight: MLXArray, eps: Float
    ) -> MLXArray? {
        guard isArmed(),
            out.ndim == 4, out.dim(1) > 1,
            gate.shape == out.shape,
            weight.ndim == 1, weight.dim(0) == out.dim(3)
        else { return nil }
        let rows = out.dim(0) * out.dim(1) * out.dim(2)
        let dv = out.dim(3)
        guard dv >= 32, dv % 32 == 0, dv <= 128, rows > 0 else { return nil }
        let outputs = Bonsai2GdnGnormManager.shared.kernel(
            [out, gate, weight.asType(.float32), MLXArray([eps])],
            template: [("InT", out.dtype), ("Dv", dv)],
            grid: (32, rows, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [out.shape],
            outputDTypes: [out.dtype])
        return outputs[0]
    }
}

/// Prefill GDN QK RMS. One simdgroup per `[B, S, Hk]` row. Same mean of
/// squares and precise rsqrt as `mlx_fast.rms_norm`, then the Qwen3.5
/// scales `Dk^-1` on q and `Dk^-0.5` on k. T=1 stays on the mega prefix.
/// Opt-in. `ALL=off` restores the two RMS norms. Set `BONSAI2_QK_RMS=1`.
enum Bonsai2QkRms {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_QK_RMS") else { return false }
        return String(cString: raw) == "1"
    }

    static func apply(q: MLXArray, k: MLXArray, invScale: Float) -> (MLXArray, MLXArray)? {
        guard isArmed(),
            q.dtype == .float32, k.dtype == .float32,
            q.ndim == 4, k.shape == q.shape, q.dim(1) > 1
        else { return nil }
        let rows = q.dim(0) * q.dim(1) * q.dim(2)
        let d = q.dim(3)
        guard d >= 32, d <= 128, d % 32 == 0, rows > 0, rows <= 65535 else { return nil }
        let qScale = invScale * invScale
        let outputs = Bonsai2QkRmsManager.shared.kernel(
            [q, k, MLXArray([qScale, invScale]), MLXArray([Float(1e-6)])],
            template: [("D", d)],
            grid: (32, rows, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [q.shape, k.shape],
            outputDTypes: [q.dtype, k.dtype])
        return (outputs[0], outputs[1])
    }
}

private final class Bonsai2QkRmsManager: Sendable {
    static let shared = Bonsai2QkRmsManager()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "bonsai2_qk_rms",
            inputNames: ["qIn", "kIn", "scales", "eps"],
            outputNames: ["qOut", "kOut"],
            source: """
                auto tid = thread_index_in_threadgroup;
                const size_t base = (size_t)threadgroup_position_in_grid.y * D;
                threadgroup float inv[2];

                float accQ = 0.0f;
                float accK = 0.0f;
                for (int d = tid; d < D; d += 32) {
                  float qv = qIn[base + d];
                  float kv = kIn[base + d];
                  accQ += qv * qv;
                  accK += kv * kv;
                }
                accQ = simd_sum(accQ);
                accK = simd_sum(accK);
                if (tid == 0) {
                  inv[0] = metal::precise::rsqrt(accQ / float(D) + eps[0]);
                  inv[1] = metal::precise::rsqrt(accK / float(D) + eps[0]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float qScale = scales[0];
                float kScale = scales[1];
                for (int d = tid; d < D; d += 32) {
                  qOut[base + d] = qIn[base + d] * inv[0] * qScale;
                  kOut[base + d] = kIn[base + d] * inv[1] * kScale;
                }
                """,
            ensureRowContiguous: true)
    }
}

private final class Bonsai2GdnGnormManager: Sendable {
    static let shared = Bonsai2GdnGnormManager()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "bonsai2_gdn_gnorm_rows",
            inputNames: ["yIn", "gate", "normW", "eps"],
            outputNames: ["yOut"],
            source: """
                auto tid = thread_index_in_threadgroup;
                const size_t base = (size_t)threadgroup_position_in_grid.y * Dv;
                threadgroup float inv[1];

                float acc = 0.0f;
                for (int d = tid; d < Dv; d += 32) {
                  float v = float(yIn[base + d]);
                  acc += v * v;
                }
                acc = simd_sum(acc);
                if (tid == 0) {
                  inv[0] = metal::precise::rsqrt(acc / float(Dv) + float(eps[0]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int d = tid; d < Dv; d += 32) {
                  float yv = float(yIn[base + d]);
                  float nrm = yv * inv[0] * float(normW[d]);
                  float gz = float(gate[base + d]);
                  float silu = gz * (1.0f / (1.0f + exp(-gz)));
                  yOut[base + d] = static_cast<InT>(silu * nrm);
                }
                """,
            ensureRowContiguous: true)
    }
}

enum Qwen35T1Compile {
    static func isArmed() -> Bool {
        if Bonsai2Valve.allOff() { return false }
        guard let raw = getenv("BONSAI2_T1_COMPILE") else { return false }
        return String(cString: raw) == "1"
    }

    static func isDecodeShape(_ x: MLXArray) -> Bool {
        if x.ndim >= 3 { return x.dim(x.ndim - 2) == 1 }
        if x.ndim == 2 { return x.dim(0) == 1 }
        return true
    }

    static func siluProduct(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
        guard isArmed(), isDecodeShape(gate) else { return silu(gate) * up }
        return siluProductCompiled(gate, up)
    }

    static func gatedSilu(_ normed: MLXArray, _ gate: MLXArray) -> MLXArray {
        guard isArmed(), isDecodeShape(normed) else {
            return (silu(gate.asType(.float32)) * normed.asType(.float32)).asType(normed.dtype)
        }
        return gatedSiluCompiled(normed, gate).asType(normed.dtype)
    }

    static func gdnGates(a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray)
        -> (g: MLXArray, beta: MLXArray)
    {
        guard isArmed(), isDecodeShape(a) else {
            return (
                computeGatedDeltaG(aLog, a, dtBias),
                sigmoid(b).asType(.float32))
        }
        let out = gdnGatesCompiled([a, b, aLog, dtBias])
        return (out[0], out[1])
    }

    private static let siluProductCompiled: @Sendable (MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: false) { gate, up in
            silu(gate) * up
        }

    private static let gatedSiluCompiled: @Sendable (MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: false) { normed, gate in
            (silu(gate.asType(.float32)) * normed.asType(.float32)).asType(normed.dtype)
        }

    private static let gdnGatesCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(
        shapeless: false
    ) { args in
        let a = args[0]
        let b = args[1]
        let aLog = args[2]
        let dtBias = args[3]
        let beta = sigmoid(b).asType(.float32)
        let g = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
        return [g, beta]
    }
}
