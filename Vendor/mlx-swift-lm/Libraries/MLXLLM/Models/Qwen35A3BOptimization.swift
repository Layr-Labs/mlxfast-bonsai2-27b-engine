// Exact-artifact construction contract for the EigenLabs Qwen3.6 35B-A3B
// campaign. This file owns only load-time inspection and immutable route
// selection. Model forwards never parse configuration, read the environment,
// or recover from an ineligible optimized dispatch.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum Qwen35A3BTargetPacking: Equatable, Sendable {
    case affine(bits: Int, groupSize: Int, routerBits: Int, routerGroupSize: Int)
}

public enum Qwen35A3BMTPPacking: Equatable, Sendable {
    case mxfp8(bits: Int, groupSize: Int)
}

public struct Qwen35A3BGeometry: Equatable, Sendable {
    public let hidden: Int
    public let experts: Int
    public let topK: Int
    public let expertIntermediate: Int
    public let sharedIntermediate: Int
    public let layers: Int
    public let recurrentLayers: Int
    public let fullAttentionLayers: Int

    public init(
        hidden: Int, experts: Int, topK: Int, expertIntermediate: Int,
        sharedIntermediate: Int, layers: Int, recurrentLayers: Int,
        fullAttentionLayers: Int
    ) {
        self.hidden = hidden
        self.experts = experts
        self.topK = topK
        self.expertIntermediate = expertIntermediate
        self.sharedIntermediate = sharedIntermediate
        self.layers = layers
        self.recurrentLayers = recurrentLayers
        self.fullAttentionLayers = fullAttentionLayers
    }
}

public enum Qwen35A3BArtifactError: Error, Equatable, CustomStringConvertible {
    case mismatch(field: String, expected: String, actual: String)
    case malformed(field: String)

    public var description: String {
        switch self {
        case .mismatch(let field, let expected, let actual):
            return "Qwen35 A3B artifact mismatch at \(field): expected \(expected), got \(actual)"
        case .malformed(let field):
            return "Qwen35 A3B artifact is missing or malformed at \(field)"
        }
    }
}

/// Model-free fixture used by construction tests. Production inspection builds
/// the same value from config.json before any optimized route can be installed.
struct Qwen35A3BArtifactFixture: Equatable, Sendable {
    var rootModelType: String
    var textModelType: String
    var hidden: Int
    var experts: Int
    var topK: Int
    var expertIntermediate: Int
    var sharedIntermediate: Int
    var layers: Int
    var recurrentLayers: Int
    var fullAttentionLayers: Int
    var targetMode: String
    var targetBits: Int
    var targetGroupSize: Int
    var routerBits: Int
    var routerGroupSize: Int
    var routerLayerEntries: Int
    var sharedGateLayerEntries: Int
    var mtpIncluded: Bool
    var mtpLayers: Int
    var mtpMode: String
    var mtpBits: Int
    var mtpGroupSize: Int

    static let eigenLabsRouter8 = Qwen35A3BArtifactFixture(
        rootModelType: "qwen3_5_moe", textModelType: "qwen3_5_moe_text",
        hidden: 2_048, experts: 256, topK: 8, expertIntermediate: 512,
        sharedIntermediate: 512, layers: 40, recurrentLayers: 30,
        fullAttentionLayers: 10, targetMode: "affine", targetBits: 4,
        targetGroupSize: 64, routerBits: 8, routerGroupSize: 64,
        routerLayerEntries: 40, sharedGateLayerEntries: 40,
        mtpIncluded: true, mtpLayers: 1, mtpMode: "mxfp8", mtpBits: 8,
        mtpGroupSize: 32)
}

public struct Qwen35A3BArtifactContract: Equatable, Sendable {
    public let target: Qwen35A3BTargetPacking
    public let mtp: Qwen35A3BMTPPacking
    public let geometry: Qwen35A3BGeometry
    public let targetReaderIdentity: String
    public let mtpReaderIdentity: String

    public var summary: String {
        "H\(geometry.hidden)/E\(geometry.experts)/K\(geometry.topK)/I"
            + "\(geometry.expertIntermediate)/L\(geometry.layers); target="
            + targetReaderIdentity + "; mtp=" + mtpReaderIdentity
    }

    static func inspect(fixture: Qwen35A3BArtifactFixture) throws -> Self {
        let expected = Qwen35A3BArtifactFixture.eigenLabsRouter8
        func require<T: Equatable>(
            _ field: String, _ actual: T, _ wanted: T
        ) throws {
            guard actual == wanted else {
                throw Qwen35A3BArtifactError.mismatch(
                    field: field, expected: String(describing: wanted),
                    actual: String(describing: actual))
            }
        }

        try require("model_type", fixture.rootModelType, expected.rootModelType)
        try require("text_config.model_type", fixture.textModelType, expected.textModelType)
        try require("text_config.hidden_size", fixture.hidden, expected.hidden)
        try require("text_config.num_experts", fixture.experts, expected.experts)
        try require("text_config.num_experts_per_tok", fixture.topK, expected.topK)
        try require(
            "text_config.moe_intermediate_size", fixture.expertIntermediate,
            expected.expertIntermediate)
        try require(
            "text_config.shared_expert_intermediate_size", fixture.sharedIntermediate,
            expected.sharedIntermediate)
        try require("text_config.num_hidden_layers", fixture.layers, expected.layers)
        try require("layer_types.recurrent", fixture.recurrentLayers, expected.recurrentLayers)
        try require(
            "layer_types.full_attention", fixture.fullAttentionLayers,
            expected.fullAttentionLayers)
        try require("quantization.mode", fixture.targetMode, expected.targetMode)
        try require("quantization.bits", fixture.targetBits, expected.targetBits)
        try require(
            "quantization.group_size", fixture.targetGroupSize, expected.targetGroupSize)
        try require("router.bits", fixture.routerBits, expected.routerBits)
        try require("router.group_size", fixture.routerGroupSize, expected.routerGroupSize)
        try require(
            "router.layer_entries", fixture.routerLayerEntries, expected.routerLayerEntries)
        try require(
            "shared_gate.layer_entries", fixture.sharedGateLayerEntries,
            expected.sharedGateLayerEntries)
        try require("mtplx_mtp.included", fixture.mtpIncluded, expected.mtpIncluded)
        try require("text_config.mtp_num_hidden_layers", fixture.mtpLayers, expected.mtpLayers)
        try require("mtp.mode", fixture.mtpMode, expected.mtpMode)
        try require("mtp.bits", fixture.mtpBits, expected.mtpBits)
        try require("mtp.group_size", fixture.mtpGroupSize, expected.mtpGroupSize)

        return Self(
            target: .affine(
                bits: fixture.targetBits, groupSize: fixture.targetGroupSize,
                routerBits: fixture.routerBits, routerGroupSize: fixture.routerGroupSize),
            mtp: .mxfp8(bits: fixture.mtpBits, groupSize: fixture.mtpGroupSize),
            geometry: Qwen35A3BGeometry(
                hidden: fixture.hidden, experts: fixture.experts, topK: fixture.topK,
                expertIntermediate: fixture.expertIntermediate,
                sharedIntermediate: fixture.sharedIntermediate, layers: fixture.layers,
                recurrentLayers: fixture.recurrentLayers,
                fullAttentionLayers: fixture.fullAttentionLayers),
            targetReaderIdentity: "affine-w4-g64",
            mtpReaderIdentity: "mxfp8-g32")
    }

    public static func inspect(configurationURL: URL) throws -> Self {
        let data = try Data(contentsOf: configurationURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = root["text_config"] as? [String: Any],
            let target = root["quantization"] as? [String: Any],
            let inline = root["mtplx_mtp"] as? [String: Any],
            let mtp = root["mtplx_mtp_quantization"] as? [String: Any],
            let layerTypes = text["layer_types"] as? [String]
        else {
            throw Qwen35A3BArtifactError.malformed(field: "config.json")
        }

        func string(_ table: [String: Any], _ key: String) throws -> String {
            guard let value = table[key] as? String else {
                throw Qwen35A3BArtifactError.malformed(field: key)
            }
            return value
        }
        func int(_ table: [String: Any], _ key: String) throws -> Int {
            guard let value = table[key] as? Int else {
                throw Qwen35A3BArtifactError.malformed(field: key)
            }
            return value
        }

        let layers = try int(text, "num_hidden_layers")
        let targetMode = try string(target, "mode")
        let targetBits = try int(target, "bits")
        let targetGroupSize = try int(target, "group_size")

        func requirePacking(
            _ path: String, _ table: [String: Any], bits: Int, groupSize: Int
        ) throws {
            let actualMode = (table["mode"] as? String) ?? targetMode
            let actualBits = (table["bits"] as? Int) ?? targetBits
            let actualGroupSize = (table["group_size"] as? Int) ?? targetGroupSize
            for (field, actual, expected) in [
                ("mode", actualMode, "affine"),
                ("bits", String(actualBits), String(bits)),
                ("group_size", String(actualGroupSize), String(groupSize)),
            ] where actual != expected {
                throw Qwen35A3BArtifactError.mismatch(
                    field: "quantization.\(path).\(field)", expected: expected,
                    actual: actual)
            }
        }

        let exactProjectionSuffixes = [
            ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
            ".linear_attn.in_proj_b", ".linear_attn.in_proj_a",
            ".linear_attn.out_proj", ".self_attn.q_proj", ".self_attn.k_proj",
            ".self_attn.v_proj", ".self_attn.o_proj",
            ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj",
            ".mlp.shared_expert.down_proj",
        ]
        for (path, value) in target where
            path == "lm_head" || exactProjectionSuffixes.contains(where: path.hasSuffix)
        {
            guard let override = value as? [String: Any] else {
                throw Qwen35A3BArtifactError.malformed(field: "quantization.\(path)")
            }
            try requirePacking(path, override, bits: 4, groupSize: 64)
        }

        var routerEntries = 0
        var sharedGateEntries = 0
        var routerBits: Int?
        var routerGroupSize: Int?
        for layer in 0 ..< layers {
            let gateKey = "language_model.model.layers.\(layer).mlp.gate"
            let sharedKey = "language_model.model.layers.\(layer).mlp.shared_expert_gate"
            guard let gate = target[gateKey] as? [String: Any] else {
                throw Qwen35A3BArtifactError.malformed(field: "quantization.\(gateKey)")
            }
            try requirePacking(gateKey, gate, bits: 8, groupSize: 64)
            routerEntries += 1
            routerBits = routerBits ?? ((gate["bits"] as? Int) ?? targetBits)
            routerGroupSize = routerGroupSize
                ?? ((gate["group_size"] as? Int) ?? targetGroupSize)

            guard let sharedGate = target[sharedKey] as? [String: Any] else {
                throw Qwen35A3BArtifactError.malformed(field: "quantization.\(sharedKey)")
            }
            try requirePacking(sharedKey, sharedGate, bits: 8, groupSize: 64)
            sharedGateEntries += 1
        }

        return try inspect(fixture: Qwen35A3BArtifactFixture(
            rootModelType: try string(root, "model_type"),
            textModelType: try string(text, "model_type"),
            hidden: try int(text, "hidden_size"), experts: try int(text, "num_experts"),
            topK: try int(text, "num_experts_per_tok"),
            expertIntermediate: try int(text, "moe_intermediate_size"),
            sharedIntermediate: try int(text, "shared_expert_intermediate_size"),
            layers: layers,
            recurrentLayers: layerTypes.filter { $0 == "linear_attention" }.count,
            fullAttentionLayers: layerTypes.filter { $0 == "full_attention" }.count,
            targetMode: targetMode, targetBits: targetBits,
            targetGroupSize: targetGroupSize,
            routerBits: routerBits ?? -1, routerGroupSize: routerGroupSize ?? 64,
            routerLayerEntries: routerEntries, sharedGateLayerEntries: sharedGateEntries,
            mtpIncluded: (inline["included"] as? Bool) ?? false,
            mtpLayers: try int(text, "mtp_num_hidden_layers"),
            mtpMode: try string(mtp, "mode"), mtpBits: try int(mtp, "bits"),
            mtpGroupSize: try int(mtp, "group_size")))
    }
}

public enum Qwen35A3BOptimizationProfile: String, Equatable, Sendable {
    case stock
    case prefill
    case decode
    case full
}

public enum Qwen35A3BTargetVerifyArithmetic: Equatable, Sendable {
    case rectangular
    case exactM1
}

/// Immutable construction capability. The only public constructor requires a
/// previously inspected artifact contract, so an optimized lane cannot be
/// selected independently of the checkpoint that proved its invariants.
public struct Qwen35A3BConstructionInstallation: Sendable {
    let profile: Qwen35A3BOptimizationProfile
    let targetVerifyArithmetic: Qwen35A3BTargetVerifyArithmetic
    public let contract: Qwen35A3BArtifactContract

    public static func install(
        contract: Qwen35A3BArtifactContract,
        profile: Qwen35A3BOptimizationProfile,
        targetVerifyArithmetic: Qwen35A3BTargetVerifyArithmetic = .rectangular
    ) throws -> Self {
        _ = try Qwen35A3BRouteTable.install(contract: contract, profile: profile)
        if targetVerifyArithmetic == .exactM1,
            profile != .decode && profile != .full
        {
            throw Qwen35A3BArtifactError.mismatch(
                field: "target_verify_arithmetic", expected: "decode-capable profile",
                actual: profile.rawValue)
        }
        return Self(
            profile: profile, targetVerifyArithmetic: targetVerifyArithmetic,
            contract: contract)
    }
}

/// Task-scoped construction state. Model initializers capture these values in
/// immutable callables and stored properties; forwards never read this state.
/// Task locality also prevents simultaneous model loads from mixing routes.
public enum Qwen35A3BConstructionContext {
    @TaskLocal public static var installation: Qwen35A3BConstructionInstallation?

    static var profile: Qwen35A3BOptimizationProfile {
        installation?.profile ?? .stock
    }

    static var targetVerifyArithmetic: Qwen35A3BTargetVerifyArithmetic {
        installation?.targetVerifyArithmetic ?? .rectangular
    }

    public static func withInstallation<T>(
        _ installation: Qwen35A3BConstructionInstallation,
        operation: () throws -> T
    ) rethrows -> T {
        try $installation.withValue(installation, operation: operation)
    }

    public static func withInstallation<T>(
        _ installation: Qwen35A3BConstructionInstallation,
        operation: () async throws -> T
    ) async rethrows -> T {
        try await $installation.withValue(installation, operation: operation)
    }
}

/// Validate the concrete modules and buffers used by the unchecked exact
/// kernels after weight loading and before the first model execution.
func qwen35A3BValidateExactProjection(
    _ linear: Linear, path: String
) throws -> QuantizedLinear {
    guard let quantized = linear as? QuantizedLinear else {
        throw Qwen35A3BArtifactError.mismatch(
            field: "loaded.\(path).type", expected: "QuantizedLinear",
            actual: String(describing: type(of: linear)))
    }
    guard quantized.bits == 4 else {
        throw Qwen35A3BArtifactError.mismatch(
            field: "loaded.\(path).bits", expected: "4",
            actual: String(quantized.bits))
    }
    guard quantized.groupSize == 64 else {
        throw Qwen35A3BArtifactError.mismatch(
            field: "loaded.\(path).group_size", expected: "64",
            actual: String(quantized.groupSize))
    }
    guard quantized.mode == .affine, let biases = quantized.biases else {
        throw Qwen35A3BArtifactError.mismatch(
            field: "loaded.\(path).mode", expected: "affine with biases",
            actual: quantized.mode.rawValue)
    }
    guard quantized.scales.dtype == .bfloat16, biases.dtype == .bfloat16 else {
        throw Qwen35A3BArtifactError.mismatch(
            field: "loaded.\(path).scale_dtype", expected: "bfloat16",
            actual: "\(quantized.scales.dtype)/\(biases.dtype)")
    }
    guard quantized.shape.0.isMultiple(of: 4) else {
        throw Qwen35A3BArtifactError.mismatch(
            field: "loaded.\(path).output_tiling", expected: "multiple of 4",
            actual: String(quantized.shape.0))
    }
    return quantized
}

public func qwen35A3BValidateLoadedExactTarget(
    _ target: Qwen35TextModel, contract: Qwen35A3BArtifactContract
) throws {
    guard target.model.embedTokens.weight.dtype == .bfloat16 else {
        throw Qwen35A3BArtifactError.mismatch(
            field: "loaded.embed_tokens.dtype", expected: "bfloat16",
            actual: String(describing: target.model.embedTokens.weight.dtype))
    }
    let exactProjectionSuffixes = [
        ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
        ".linear_attn.in_proj_b", ".linear_attn.in_proj_a",
        ".linear_attn.out_proj", ".self_attn.q_proj", ".self_attn.k_proj",
        ".self_attn.v_proj", ".self_attn.o_proj",
        ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj",
        ".mlp.shared_expert.down_proj",
    ]
    var validated = 0
    for (path, module) in target.namedModules() where
        path == "lm_head" || exactProjectionSuffixes.contains(where: path.hasSuffix)
    {
        guard let linear = module as? Linear else {
            throw Qwen35A3BArtifactError.mismatch(
                field: "loaded.\(path).type", expected: "Linear",
                actual: String(describing: type(of: module)))
        }
        _ = try qwen35A3BValidateExactProjection(linear, path: path)
        validated += 1
    }
    let expected = contract.geometry.recurrentLayers * 5
        + contract.geometry.fullAttentionLayers * 4
        + contract.geometry.layers * 3 + 1
    guard validated == expected else {
        throw Qwen35A3BArtifactError.mismatch(
            field: "loaded.exact_projection_count", expected: String(expected),
            actual: String(validated))
    }
}

enum Qwen35A3BRouteLane: String, Equatable, Sendable {
    case disabled
    case stock
    case rightShaped = "right-shaped"
}

/// Construction result. Later kernel tasks replace each `rightShaped` lane's
/// implementation atomically before this table is attached to the model; the
/// table itself never mutates once published.
struct Qwen35A3BRouteTable: Equatable, Sendable {
    let profile: Qwen35A3BOptimizationProfile
    let contract: Qwen35A3BArtifactContract
    let prefill: Qwen35A3BRouteLane
    let targetDecode: Qwen35A3BRouteLane
    let mtpDecode: Qwen35A3BRouteLane

    static func install(
        contract: Qwen35A3BArtifactContract,
        profile: Qwen35A3BOptimizationProfile
    ) throws -> Self {
        switch profile {
        case .stock:
            return Self(
                profile: profile, contract: contract, prefill: .stock,
                targetDecode: .stock, mtpDecode: .disabled)
        case .prefill:
            return Self(
                profile: profile, contract: contract, prefill: .rightShaped,
                targetDecode: .stock, mtpDecode: .disabled)
        case .decode:
            return Self(
                profile: profile, contract: contract, prefill: .stock,
                targetDecode: .rightShaped, mtpDecode: .rightShaped)
        case .full:
            return Self(
                profile: profile, contract: contract, prefill: .rightShaped,
                targetDecode: .rightShaped, mtpDecode: .rightShaped)
        }
    }
}

// MARK: - Exact E256/K8 row-owned router finalizer

/// One threadgroup owns one complete BF16 probability row. The two-stage
/// tournament preserves MLX argPartition's ascending selected order and its
/// BF16 denominator accumulation. This primitive is unchecked by design: the
/// artifact contract and model construction route prove E256/K8/BF16 once.
private let qwen35A3BRowOwnedRouterKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_row_owned_top8_bf16_exact",
    inputNames: ["probabilities"],
    outputNames: ["expert_ids", "route_scores"],
    source: """
        constexpr int N = 256;
        constexpr int TOPK = 8;
        constexpr int SIMD_GROUPS = 8;
        constexpr int LOCAL_CANDIDATES = SIMD_GROUPS * TOPK;

        uint row = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        threadgroup float local_probabilities[LOCAL_CANDIDATES];
        threadgroup int local_indices[LOCAL_CANDIDATES];
        threadgroup float merged_probabilities[TOPK];
        threadgroup int merged_indices[TOPK];

        int expert = int(simd_gid) * 32 + int(lane);
        float candidate_probability = float(probabilities[row * N + expert]);
        int candidate_index = expert;

        _Pragma("unroll")
        for (int rank = 0; rank < TOPK; ++rank) {
            float winner_probability = simd_max(candidate_probability);
            float winner_index_value = simd_max(
                candidate_probability == winner_probability
                    ? float(candidate_index) : -1.0f);
            int winner_index = int(winner_index_value);
            if (lane == 0) {
                int destination = int(simd_gid) * TOPK + rank;
                local_probabilities[destination] = winner_probability;
                local_indices[destination] = winner_index;
            }
            if (candidate_index == winner_index) {
                candidate_probability = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {
            int slot0 = int(lane);
            int slot1 = int(lane) + 32;
            float probability0 = local_probabilities[slot0];
            float probability1 = local_probabilities[slot1];
            int index0 = local_indices[slot0];
            int index1 = local_indices[slot1];

            _Pragma("unroll")
            for (int rank = 0; rank < TOPK; ++rank) {
                bool take1 = probability1 > probability0
                    || (probability1 == probability0 && index1 > index0);
                float lane_probability = take1 ? probability1 : probability0;
                int lane_index = take1 ? index1 : index0;
                float winner_probability = simd_max(lane_probability);
                float winner_index_value = simd_max(
                    lane_probability == winner_probability
                        ? float(lane_index) : -1.0f);
                int winner_index = int(winner_index_value);
                if (lane == 0) {
                    merged_probabilities[rank] = winner_probability;
                    merged_indices[rank] = winner_index;
                }
                if (lane_index == winner_index) {
                    if (take1) { probability1 = -INFINITY; }
                    else { probability0 = -INFINITY; }
                }
            }

            if (lane == 0) {
                bfloat rounded_denominator = bfloat(0.0f);
                _Pragma("unroll")
                for (int output = 0; output < TOPK; ++output) {
                    rounded_denominator = bfloat(
                        float(rounded_denominator)
                            + merged_probabilities[TOPK - 1 - output]);
                }
                _Pragma("unroll")
                for (int output = 0; output < TOPK; ++output) {
                    int source = TOPK - 1 - output;
                    int destination = int(row) * TOPK + output;
                    expert_ids[destination] = uint(merged_indices[source]);
                    route_scores[destination] = bfloat(
                        merged_probabilities[source]
                            / float(rounded_denominator));
                }
            }
        }
    """,
    ensureRowContiguous: true)

func qwen35A3BRowOwnedRoute(
    _ probabilities: MLXArray, rows: Int
) -> (expertIDs: MLXArray, routeScores: MLXArray) {
    let outputs = qwen35A3BRowOwnedRouterKernel(
        [probabilities.reshaped([rows, 256])],
        grid: (rows * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[rows, 8], [rows, 8]],
        outputDTypes: [.uint32, .bfloat16])
    let outputShape = Array(probabilities.shape.dropLast()) + [8]
    return (
        outputs[0].reshaped(outputShape),
        outputs[1].reshaped(outputShape))
}

typealias Qwen35A3BRouterFinalizer = (MLXArray) -> (
    expertIDs: MLXArray, routeScores: MLXArray
)

private func qwen35A3BStockRoute(
    _ probabilities: MLXArray, topK: Int, normalize: Bool
) -> (expertIDs: MLXArray, routeScores: MLXArray) {
    let kth = probabilities.dim(-1) - topK
    let ids = argPartition(probabilities, kth: kth, axis: -1)[.ellipsis, kth...]
    var scores = takeAlong(probabilities, ids, axis: -1)
    if normalize { scores = scores / scores.sum(axis: -1, keepDims: true) }
    return (ids, scores)
}

/// Bind either the unchanged stock finalizer or the exact row-owned decode
/// route. The only execution-time decision in the optimized closure is logical
/// M: rows 1...16 are the installed Metal lane; all wider values are the
/// explicit prefill route selected by construction.
func qwen35A3BRouterFinalizer(
    hidden: Int, experts: Int, topK: Int, normalize: Bool
) -> Qwen35A3BRouterFinalizer {
    let stock: Qwen35A3BRouterFinalizer = { probabilities in
        qwen35A3BStockRoute(
            probabilities, topK: topK, normalize: normalize)
    }
    guard (Qwen35A3BConstructionContext.profile == .decode
        || Qwen35A3BConstructionContext.profile == .full),
        hidden == 2_048, experts == 256, topK == 8, normalize
    else { return stock }

    return { probabilities in
        let rows = probabilities.size / 256
        if rows >= 1 && rows <= 16 {
            return qwen35A3BRowOwnedRoute(probabilities, rows: rows)
        }
        return stock(probabilities)
    }
}

private let qwen35A3BCombineKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_combine_bf16_exact",
    inputNames: ["routed", "scores"],
    outputNames: ["combined"],
    source: """
        constexpr int TOPK = 8;
        constexpr int HIDDEN = 2048;
        uint output_index = thread_position_in_grid.x;
        uint row = output_index / HIDDEN;
        uint column = output_index - row * HIDDEN;
        bfloat accumulator = bfloat(0.0f);
        _Pragma("unroll")
        for (int expert = 0; expert < TOPK; ++expert) {
            uint routed_index = (row * TOPK + uint(expert)) * HIDDEN + column;
            uint score_index = row * TOPK + uint(expert);
            bfloat product = bfloat(
                float(routed[routed_index]) * float(scores[score_index]));
            accumulator = bfloat(float(accumulator) + float(product));
        }
        combined[output_index] = accumulator;
    """,
    ensureRowContiguous: true)

typealias Qwen35A3BExpertCombiner = (MLXArray, MLXArray) -> MLXArray

func qwen35A3BExpertCombiner(
    hidden: Int, topK: Int
) -> Qwen35A3BExpertCombiner {
    let stock: Qwen35A3BExpertCombiner = { routed, scores in
        weightedExpertSum(routed, scores.asType(routed.dtype))
    }
    guard (Qwen35A3BConstructionContext.profile == .decode
        || Qwen35A3BConstructionContext.profile == .full),
        hidden == 2_048, topK == 8
    else { return stock }

    return { routed, scores in
        let rows = routed.size / (8 * 2_048)
        guard rows == 1 || rows == 2 else { return stock(routed, scores) }
        let output = qwen35A3BCombineKernel(
            [routed, scores],
            grid: (rows * 2_048, 1, 1),
            threadGroup: (rows == 1 ? 128 : 64, 1, 1),
            outputShapes: [[rows, 2_048]],
            outputDTypes: [.bfloat16])[0]
        let shape = Array(routed.shape.dropLast(2)) + [2_048]
        return output.reshaped(shape)
    }
}

// MARK: - Chunked GDN with prepared fresh-state scan
// Kept here to respect the model source per-file byte limit.
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
    /// `MLXFAST_GDN_CHUNKED_VERIFY=0` keeps verify on the sequential kernel.
    /// Off by default here: every verify-width change on this lineage that
    /// the ranked box measured lengthened the decode window, and this record's
    /// own window is 5.9% longer than its parent's. `MLXFAST_GDN_CHUNKED_VERIFY=1`
    /// takes the chunked kernels at verify width.
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

    // MARK: Fresh-state scan

    /// `BONSAI_GDN_CHUNKED_FRESH=0` keeps a new request's first prompt chunk
    /// on `run` from a materialized zeros state.
    static let freshEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_GDN_CHUNKED_FRESH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `scanSource` for a state known to be all zeros (a new request's first
    /// prompt chunk, see `Qwen35GatedDeltaNet.freshPromptChunk`): each
    /// simdgroup's state registers start as a zero matrix instead of loading a
    /// zeros array, so the `[B, Hv, Dv, Dk]` zeros array is never
    /// materialized. The registers hold the same values (+0.0f) the load would
    /// have put there, and every later operation is the stock text. Derived
    /// from `scanSource` by one checked replacement, so the stock kernel stays
    /// as it is and this one follows it.
    private static let scanFreshSource: String = {
        let target =
            "simdgroup_load(St[d], state_in + ((size_t)bh * Dv + r0) * Dk + d * 8, Dk, ulong2(0, 0), true);"
        precondition(
            scanSource.components(separatedBy: target).count == 2,
            "Qwen35 chunked GDN: the fresh-state scan source no longer matches the stock kernel")
        let text = scanSource.replacingOccurrences(
            of: target, with: "St[d] = simdgroup_float8x8(0);")
        precondition(!text.contains("state_in"))
        return text
    }()

    private static let scanFreshKernel = MLXFast.metalKernel(
        name: "bonsai_gated_delta_chunk_scan_fresh",
        inputNames: ["q", "k", "v", "tp", "pm", "gf", "T"],
        outputNames: ["y", "state_out"],
        source: scanFreshSource)

    /// `run` from an all-zero FP32 state of `stateShape`, which is not passed,
    /// for a window of whole chunks (a remainder's sequential tail reads the
    /// state array, so such a window stays on `run`); nil when this does not
    /// apply or the geometry did not pass its check, and the caller takes the
    /// stock path.
    static func runFresh(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, stateShape: [Int]
    ) -> (MLXArray, MLXArray)? {
        guard enabled, freshEnabled, q.ndim == 4, k.ndim == 4, v.ndim == 4 else { return nil }
        let B = k.dim(0)
        let T = k.dim(1)
        guard T >= minRows, T >= chunk, T % chunk == 0,
            q.shape == k.shape, k.dim(3) == 128, v.dim(1) == T, v.dim(3) % 8 == 0,
            v.dim(2) % k.dim(2) == 0,
            g.shape == [B, T, v.dim(2)], beta.shape == [B, T, v.dim(2)],
            stateShape == [B, v.dim(2), v.dim(3), 128],
            q.dtype == .float32, k.dtype == .float32, v.dtype == .float32,
            g.dtype == .float32, beta.dtype == .float32,
            freshVerified(hk: k.dim(2), dk: k.dim(3), hv: v.dim(2), dv: v.dim(3))
        else { return nil }
        return freshChunks(q: q, k: k, v: v, g: g, beta: beta, stateShape: stateShape)
    }

    /// `chunks` with `scanFreshKernel` in place of `scanKernel`: the same prep
    /// launch, the same scan launch geometry, no state input.
    private static func freshChunks(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, stateShape: [Int]
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
        let outputs = scanFreshKernel(
            [q, k, v, prepared[0], prepared[1], prepared[2], rowCount],
            template: [
                ("C", C), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
                ("NS", scanSimdgroups),
            ],
            grid: (32, Dv / 8, B * Hv),
            threadGroup: (32, scanSimdgroups, 1),
            outputShapes: [[B, T, Hv, Dv], stateShape],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }

    private static let freshLock = NSLock()
    nonisolated(unsafe) private static var freshVerdicts: [[Int]: Bool] = [:]

    /// Verdict lookup only (the check runs in `prepareFresh`, never inside a
    /// forward); a geometry that was not prepared, or failed, keeps `run`.
    private static func freshVerified(hk: Int, dk: Int, hv: Int, dv: Int) -> Bool {
        freshLock.withLock { freshVerdicts[[hk, dk, hv, dv, chunk]] ?? false }
    }

    /// Compile the fresh-state scan for this geometry and check it bit for bit
    /// against `chunks` from a zeros state, once per process, at model
    /// construction. A mismatch prints one line and keeps `run`. Called from
    /// the layer's init.
    static func prepareFresh(hk: Int, dk: Int, hv: Int, dv: Int) {
        guard enabled, freshEnabled, hk > 0, dk == 128, dv % 8 == 0, hv % hk == 0
        else { return }
        let key = [hk, dk, hv, dv, chunk]
        if freshLock.withLock({ freshVerdicts[key] != nil }) { return }
        let verdict = freshSelfCheck(hk: hk, dk: dk, hv: hv, dv: dv)
        let recorded = freshLock.withLock { () -> Bool in
            guard freshVerdicts[key] == nil else { return false }
            freshVerdicts[key] = verdict
            return true
        }
        if recorded && !verdict {
            FileHandle.standardError.write(
                "qwen35: chunked GDN fresh-state scan disagrees with the stock scan on this device; using the stock scan\n"
                    .data(using: .utf8)!)
        }
    }

    private static func freshSelfCheck(hk: Int, dk: Int, hv: Int, dv: Int) -> Bool {
        let keys = MLXRandom.split(key: MLXRandom.key(0x6673_7368), into: 8)
        for T in [minRows, 512] where T % chunk == 0 && T >= chunk {
            // Wide magnitude spread; rows 8..15 of every head have g == 0 (the
            // prep's underflow mask), rows 16..23 a subnormal g.
            func spread(_ shape: [Int], _ i: Int) -> MLXArray {
                MLXRandom.normal(shape, key: keys[i]) * exp(MLXRandom.normal(shape, key: keys[i + 3]))
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
            let (yNew, sNew) = freshChunks(
                q: q, k: k, v: v, g: g, beta: beta, stateShape: stateShape)
            let same = all(yRef.view(dtype: .uint32) .== yNew.view(dtype: .uint32))
                .&& all(sRef.view(dtype: .uint32) .== sNew.view(dtype: .uint32))
            if !same.item(Bool.self) { return false }
        }
        return true
    }
}
