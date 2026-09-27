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


// Moved verbatim from Qwen35.swift (per-file byte cap); same module.
/// The full-attention layer's pre-SDPA work as ONE launch (polymorf's "fused
/// attention prework", `d4ed6585` notes; this is an independent
/// implementation, that one was not shipped): for each (row, head) the q or k
/// head is read through its strides from the q|gate and k projections, then
/// MLX's `rms_single_row` over the head (D/4 lanes x 4 reads, the
/// `acc += x * x` chain, `simd_sum`, the zeroed 32-slot threadgroup pass,
/// `precise::rsqrt(acc / axis + eps)`, `w * (x * inv)`), then MLX's `rope`
/// kernel on the first RD channels (`exp2(-(i / (RD/2)) * log2(base))`,
/// `fast::cos`/`fast::sin` of `scale * float(t + offset) * inv_freq`,
/// `x1 * cos - x2 * sin`, `x1 * sin + x2 * cos`), written head-major
/// ([B, H, L, D], row-contiguous) as the rope's partial-dims copy lays it out.
/// The op chain runs eight launches per attention layer for this (a strided
/// copy and an `rms` launch per norm, a copy and a `rope` launch per partial
/// rotation) plus the offsets copy; the kernel reads the cache's offsets array
/// before the cache advances, which is what the copy captured. Same
/// expressions in the same order, all compiled without fast math; verified
/// bit for bit against the op chain once per geometry at model construction
/// (a disagreeing device keeps the op chain). `BONSAI_FUSED_ATTN_PREWORK=0`
/// keeps the op chain.
enum Qwen35AttentionPrework {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_FUSED_ATTN_PREWORK"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // grid (TPG * (HQ + HK), L, B), threadgroup (TPG, 1, 1), TPG = D / 4.
    // Inputs: q [B, L, HQ, D] and k [B, L, HK, D] (any strides, same dtype),
    // wq/wk [D] FP32, offs [1] or [B] int32 (any stride), epsq/epsk/scale/
    // lbase (log2 of the rope base) FP32 scalars, axis (= D) uint32 scalar.
    // Outputs qo [B, HQ, L, D], ko [B, HK, L, D] FP32.
    private static let source = """
        constexpr int NR = 4;
        constexpr int HALF = RD / 2;
        const uint lid = thread_position_in_threadgroup.x;
        const uint hh = threadgroup_position_in_grid.x;
        const uint t = threadgroup_position_in_grid.y;
        const uint bb = threadgroup_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int Ln = int(q_shape[1]);
        const bool isq = hh < uint(HQ);
        const uint h = isq ? hh : hh - uint(HQ);

        threadgroup float local_sums[32];
        threadgroup float local_inv[1];
        threadgroup float rot[RD];

        // rms_single_row: lane lid holds channels NR*lid .. NR*lid+NR-1.
        const int64_t base = isq
            ? int64_t(bb) * q_strides[0] + int64_t(t) * q_strides[1] + int64_t(h) * q_strides[2]
            : int64_t(bb) * k_strides[0] + int64_t(t) * k_strides[1] + int64_t(h) * k_strides[2];
        const int64_t cs = isq ? q_strides[3] : k_strides[3];
        auto src = isq ? q : k;
        float acc = 0;
        float thread_x[NR];
        for (int i = 0; i < NR; i++) {
          thread_x[i] = static_cast<float>(src[base + int64_t(lid * NR + i) * cs]);
          acc += thread_x[i] * thread_x[i];
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
          acc = simd_sum(local_sums[lane]);
          if (lane == 0) {
            const float eps = isq ? epsq : epsk;
            local_inv[0] = metal::precise::rsqrt(acc / axis + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto w = isq ? wq : wk;
        auto dst = isq ? qo : ko;
        const size_t obase =
            ((size_t(bb) * size_t(isq ? HQ : HK) + size_t(h)) * size_t(Ln) + size_t(t)) * size_t(D);
        const float inv = local_inv[0];
        for (int i = 0; i < NR; i++) {
          const uint c = lid * NR + uint(i);
          const float n = w[c] * static_cast<float>(thread_x[i] * inv);
          if (c < uint(RD)) {
            rot[c] = n;
          } else {
            dst[obase + c] = n;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // rope on channels [0, RD): pair (j, j + HALF), as MLX's rope kernel.
        if (lid < uint(HALF)) {
          const int off = offs[OB ? 0 : int64_t(bb) * offs_strides[0]];
          float d = static_cast<float>(lid) / static_cast<float>(HALF);
          float inv_freq = metal::exp2(-d * lbase);
          float L = scale * static_cast<float>(t + off);
          float theta = L * inv_freq;
          float costheta = metal::fast::cos(theta);
          float sintheta = metal::fast::sin(theta);
          float x1 = rot[lid];
          float x2 = rot[lid + HALF];
          float rx1 = x1 * costheta - x2 * sintheta;
          float rx2 = x1 * sintheta + x2 * costheta;
          dst[obase + lid] = rx1;
          dst[obase + lid + HALF] = rx2;
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_attn_prework",
        inputNames: ["q", "k", "wq", "wk", "offs", "epsq", "epsk", "axis", "lbase", "scale"],
        outputNames: ["qo", "ko"],
        source: source,
        ensureRowContiguous: false)

    /// `(rope(qNorm(q).transposed(0, 2, 1, 3)), rope(kNorm(k).transposed(0, 2, 1,
    /// 3)))` at `offsets` for `q` [B, L, HQ, D] and `k` [B, L, HK, D]; nil when
    /// it does not apply.
    static func run(
        q: MLXArray, k: MLXArray, qNorm: RMSNorm, kNorm: RMSNorm,
        offsets: MLXArray, ropeDims: Int, ropeBase: Float
    ) -> (MLXArray, MLXArray)? {
        guard enabled, q.ndim == 4, k.ndim == 4,
            verified(
                Geometry(
                    hq: q.dim(2), hk: k.dim(2), d: q.dim(3), rd: ropeDims, dtype: "\(q.dtype)"))
        else { return nil }
        return runUnchecked(
            q: q, k: k, wq: qNorm.weight, wk: kNorm.weight, epsQ: qNorm.eps, epsK: kNorm.eps,
            offsets: offsets, ropeDims: ropeDims, ropeBase: ropeBase)
    }

    private static func runUnchecked(
        q: MLXArray, k: MLXArray, wq: MLXArray, wk: MLXArray, epsQ: Float, epsK: Float,
        offsets: MLXArray, ropeDims: Int, ropeBase: Float
    ) -> (MLXArray, MLXArray)? {
        let B = q.dim(0)
        let L = q.dim(1)
        let HQ = q.dim(2)
        let HK = k.dim(2)
        let D = q.dim(3)
        guard k.dim(0) == B, k.dim(1) == L, k.dim(3) == D,
            q.dtype == k.dtype, [DType.float32, .float16, .bfloat16].contains(q.dtype),
            wq.dtype == .float32, wk.dtype == .float32, wq.shape == [D], wk.shape == [D],
            offsets.dtype == .int32, offsets.ndim <= 1,
            offsets.size == 1 || offsets.size == B,
            L > 0, L < 65536
        else { return nil }
        let offs = offsets.ndim == 1 ? offsets : offsets.reshaped([1])
        let outputs = kernel(
            [q, k, wq, wk, offs, MLXArray(epsQ), MLXArray(epsK), MLXArray(UInt32(D)),
             MLXArray(log2(ropeBase)), MLXArray(Float(1))],
            template: [
                ("D", D), ("RD", ropeDims), ("HQ", HQ), ("HK", HK),
                ("OB", offs.size == 1 ? 1 : 0),
            ],
            grid: ((D / 4) * (HQ + HK), L, B), threadGroup: (D / 4, 1, 1),
            outputShapes: [[B, HQ, L, D], [B, HK, L, D]],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }

    private struct Geometry: Hashable {
        let hq: Int, hk: Int, d: Int, rd: Int, dtype: String
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [Geometry: Bool] = [:]

    private static func verified(_ geometry: Geometry) -> Bool {
        lock.withLock { verdicts[geometry] ?? false }
    }

    /// Compile the kernel and check it bit for bit against the op chain for one
    /// attention geometry, once per process, at model construction (before
    /// any timed forward). A geometry or dtype that was not prepared, or that
    /// disagrees, keeps the op chain.
    static func prepare(
        hq: Int, hk: Int, d: Int, ropeDims rd: Int, ropeBase: Float, epsQ: Float, epsK: Float
    ) {
        guard enabled, d % 128 == 0, d <= 4096, rd > 0, rd % 4 == 0, rd <= d,
            rd / 2 <= d / 4
        else { return }
        lock.withLock {
            for dtype in [DType.float32] {
                let geometry = Geometry(hq: hq, hk: hk, d: d, rd: rd, dtype: "\(dtype)")
                if verdicts[geometry] != nil { continue }
                let verdict = selfCheck(
                    geometry, dtype: dtype, ropeBase: ropeBase, epsQ: epsQ, epsK: epsK)
                verdicts[geometry] = verdict
                if !verdict {
                    FileHandle.standardError.write(
                        "qwen35: fused attention prework disagrees with the op chain on this device (\(dtype)); using the op chain\n"
                            .data(using: .utf8)!)
                }
            }
        }
    }

    private static func selfCheck(
        _ geo: Geometry, dtype: DType, ropeBase: Float, epsQ: Float, epsK: Float
    ) -> Bool {
        let keys = MLXRandom.split(key: MLXRandom.key(0x6174_746e), into: 6)
        let wq = 1 + 0.25 * MLXRandom.normal([geo.d], key: keys[0])
        let wk = 1 + 0.25 * MLXRandom.normal([geo.d], key: keys[1])
        // The op chain as `Qwen35Attention.cbv2Forward` composes it (RMSNorm and
        // RoPE modules call exactly these).
        func chain(_ x: MLXArray, _ w: MLXArray, _ eps: Float, _ offsets: MLXArray) -> MLXArray {
            MLXFast.RoPE(
                MLXFast.rmsNorm(x, weight: w, eps: eps).transposed(0, 2, 1, 3),
                dimensions: geo.rd, traditional: false, base: ropeBase, scale: 1,
                offset: offsets + 0)
        }
        var same = MLXArray(true)
        for (index, (rows, offset)) in [(16, 611), (1, 4093), (37, 0), (5, 200_003)].enumerated() {
            // The q|gate, k and v projections stacked as the verify produces
            // them; wide magnitude spread so the reductions see real rounding.
            let width = geo.hq * 2 * geo.d + 2 * geo.hk * geo.d
            let wide = (MLXRandom.normal([1, rows, width], key: keys[2 + index % 4])
                * exp(MLXRandom.normal([1, rows, width], key: keys[(3 + index) % 6])))
                .asType(dtype)
            let parts = MLX.split(
                wide, indices: [geo.hq * 2 * geo.d, geo.hq * 2 * geo.d + geo.hk * geo.d],
                axis: -1)
            let q = parts[0].reshaped(1, rows, geo.hq, -1).split(parts: 2, axis: -1)[0]
            let k = parts[1].reshaped(1, rows, geo.hk, -1)
            let offsets = MLXArray([Int32(offset)])
            let refQ = chain(q, wq, epsQ, offsets)
            let refK = chain(k, wk, epsK, offsets)
            guard
                let (newQ, newK) = runUnchecked(
                    q: q, k: k, wq: wq, wk: wk, epsQ: epsQ, epsK: epsK, offsets: offsets,
                    ropeDims: geo.rd, ropeBase: ropeBase),
                refQ.shape == newQ.shape, refK.shape == newK.shape,
                refQ.dtype == newQ.dtype, refK.dtype == newK.dtype
            else { return false }
            same = same .&& all(refQ.view(dtype: .uint32) .== newQ.view(dtype: .uint32))
                .&& all(refK.view(dtype: .uint32) .== newK.view(dtype: .uint32))
        }
        return same.item(Bool.self)
    }
}
