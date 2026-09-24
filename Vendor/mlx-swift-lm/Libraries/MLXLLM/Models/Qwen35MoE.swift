//
//  Qwen35MoE.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5_moe.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen35Configuration: Codable, Sendable {
    public var modelType: String
    public var textConfig: Qwen35TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)

        if let textConfig = try container.decodeIfPresent(
            Qwen35TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
        } else {
            self.textConfig = try Qwen35TextConfiguration(from: decoder)
        }
    }

    public var cbv2LayerKinds: [CBv2LayerKind] { textConfig.cbv2LayerKinds }

    public func cbv2RecurrentStateSpec(
        activationDType: DType = .bfloat16
    ) -> CBv2RecurrentStateSpec {
        textConfig.cbv2RecurrentStateSpec(activationDType: activationDType)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities { textConfig.cbv2Capabilities }
}

/// Fuse split routed-expert projections into the fused `gate_up_proj` layout
/// that `SwitchGLU(fuseGateUp: true)` expects (one gather_qmm serving
/// gate+up).
///
/// Two checkpoint families are handled, prefix-agnostically:
/// - Raw HF exports carry one stacked tensor `<mlp>.experts.gate_up_proj`
///   (`[E, 2*ffn, hidden]`, gate rows first — exactly the order
///   `SwitchGLU` slices) plus `<mlp>.experts.down_proj`; both map directly
///   onto `<mlp>.switch_mlp.{gate_up_proj,down_proj}.weight` without the
///   historical split.
/// - MLX-converted checkpoints (including quantized ones, e.g. the
///   production W4/g64 artifact) carry split `<mlp>.switch_mlp.gate_proj.*`
///   and `up_proj.*` tensors. Concatenating along the output-row axis is
///   exact for any per-output-row grouped quantization (affine or mxfp8):
///   rows quantize independently, so `weight`, `scales`, and `biases` all
///   concatenate on axis -2 and an unquantized `bias` on axis -1. The fused
///   module path resolves its quantization through
///   `qwen35GateUpQuantizationAliases`: explicit per-layer overrides written
///   against the split paths keep applying to the fused projection, and
///   default-quantized checkpoints resolve the default as before.
///
/// Fusing is a PER-LAYER, PER-LOAD decision: it is only representable when
/// both halves resolve to one quantization policy, because a single
/// quantized projection has one bits/group_size/mode. For every pair whose
/// halves differ — in resolved policy (`perLayerQuantization`, explicit
/// entries first, default fallback), in carried tensor sets (e.g. affine
/// `biases` vs mxfp without), or in packed shapes/dtypes (bits or
/// group_size mismatch, caught even without a policy table) — the split
/// tensors are kept verbatim, the reason is logged, and `setFused` is
/// invoked with (`<mlp>.switch_mlp`, false). Every pair that IS fused (and
/// every already-fused checkpoint tensor) reports (path, true) instead, so
/// the caller can reshape that layer's `SwitchGLU` in either direction
/// (`qwen35SetSwitchGLUGateUpFused`) — the topology follows each load
/// rather than ratcheting one way. Split loading is bit-exact: the tensors
/// are untouched and each half quantizes through its own explicit table
/// entry.
///
/// `mtp.*` keys are never touched: the inline MTP head keeps split modules
/// (`Qwen35MTPDecoderLayer` passes `fuseGateUp: false`) because its
/// quantization table (`mtplx_mtp_quantization`) is keyed on the split
/// module paths.
///
/// A split checkpoint missing one half of a gate/up pair (corrupt, partial
/// download, bad conversion) is left untouched — with a warning naming the
/// layer — so the strict update reports a catchable `UpdateError` (the
/// orphaned keys are unhandled and the fused parameters stay unset), exactly
/// like other malformed checkpoints. `loadWeights` stays a throwing API;
/// nothing here terminates the host process.
public func qwen35FuseSwitchMLPGateUp(
    weights: [String: MLXArray],
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    setFused: ((String, Bool) -> Void)? = nil
) -> [String: MLXArray] {
    var adjusted = weights

    // Raw HF stacked exports: already fused, just re-keyed.
    let rawSuffix = ".experts.gate_up_proj"
    for key in Array(adjusted.keys)
    where key.hasSuffix(rawSuffix) && !key.contains("mtp.") {
        guard let gateUp = adjusted.removeValue(forKey: key) else { continue }
        let prefix = String(key.dropLast(rawSuffix.count))
        adjusted["\(prefix).switch_mlp.gate_up_proj.weight"] = gateUp
        if let downProj = adjusted.removeValue(forKey: "\(prefix).experts.down_proj") {
            adjusted["\(prefix).switch_mlp.down_proj.weight"] = downProj
        }
    }

    return fuseSwitchGLUGateUpWeights(
        weights: adjusted,
        perLayerQuantization: perLayerQuantization,
        quantizationAliases: {
            Array(qwen35PolicyPathCandidates(for: $0).dropFirst())
        },
        shouldProcess: { !$0.contains("mtp.") },
        setFused: setFused)
}


/// Config-table path candidates for one checkpoint module path, bridging
/// every key space the sanitizers and configs are known to use:
/// `language_model.model.*` (wrapper module tree), `model.language_model.*`
/// (raw HF VLM exports — the VLM sanitizer fuses BEFORE remapping, so raw
/// paths reach policy lookups directly), `model.*` (bare text-model
/// checkpoints and bare-keyed mixed-precision tables), and
/// `language_model.*`. The caller's exact path is always the first
/// candidate so a same-space entry wins; the remaining spaces follow in a
/// fixed order. Raw, wrapper, and bare inputs normalize symmetrically —
/// each produces the full candidate set.
private func qwen35PolicyPathCandidates(for path: String) -> [String] {
    // Longest prefix first so "model.language_model." is not consumed by
    // the bare "model." arm.
    let prefixes = [
        "language_model.model.", "model.language_model.", "language_model.", "model.",
    ]
    var suffix = path
    for prefix in prefixes where path.hasPrefix(prefix) {
        suffix = String(path.dropFirst(prefix.count))
        break
    }
    var candidates = [path]
    for candidate in [
        "language_model.model." + suffix,
        "model.language_model." + suffix,
        "model." + suffix,
        "language_model." + suffix,
        suffix,
    ] where !candidates.contains(candidate) {
        candidates.append(candidate)
    }
    return candidates
}


/// Reshape the `SwitchGLU` at `path` (checkpoint key space) to the fused or
/// split gate/up topology, whichever the current load requires. Intended as
/// the `setFused` callback of `qwen35FuseSwitchMLPGateUp`: heterogeneous
/// pairs load through split `gate_proj`/`up_proj` modules, and a later
/// homogeneous load on the same instance restores the fused layout. No-op
/// when the module is absent or already has the requested topology. The
/// swap goes through `Module.update(modules:)` so the module cache sees the
/// new children.
public func qwen35SetSwitchGLUGateUpFused(_ fused: Bool, at path: String, in root: Module) {
    setSwitchGLUGateUpFused(
        fused,
        at: path,
        aliases: Array(qwen35PolicyPathCandidates(for: path).dropFirst()),
        in: root)
}

/// Per-layer quantization aliases for the routed-expert projections, across
/// every checkpoint key space.
///
/// Mixed-precision configs are written against the checkpoint layout, which
/// differs from the post-sanitize module tree in two independent ways:
/// - `qwen35FuseSwitchMLPGateUp` erases the split `…switch_mlp.gate_proj` /
///   `…switch_mlp.up_proj` names when it fuses a pair, and
/// - the wrappers remap the namespace itself (raw VLM checkpoints key
///   entries on `model.language_model.*` while the module tree is
///   `language_model.model.*`; text-only checkpoints use bare `model.*`).
///
/// `loadWeights` consults these aliases whenever the module path has no
/// entry of its own, so both remappings must be bridged: a fused
/// `gate_up_proj` module aliases to its own name in the other key spaces
/// first, then to the split halves in every key space (gate first; a pair
/// is only fused when both halves resolved to one quantization policy, so
/// whichever half carries an entry describes the fused tensor). A split
/// `gate_proj`/`up_proj` module kept by the per-layer policy decision
/// aliases to its own name in the other key spaces.
///
/// `mtp.*` paths return no aliases: the inline MTP head keeps split modules
/// with its own quantization table.
public func qwen35GateUpQuantizationAliases(for path: String) -> [String] {
    guard !path.contains("mtp.") else { return [] }

    var aliases: [String] = []
    func appendCandidates(for name: String) {
        for candidate in qwen35PolicyPathCandidates(for: name)
        where candidate != path && !aliases.contains(candidate) {
            aliases.append(candidate)
        }
    }

    let fusedSuffix = ".gate_up_proj"
    if path.hasSuffix(fusedSuffix) {
        let base = String(path.dropLast(fusedSuffix.count))
        appendCandidates(for: path)
        appendCandidates(for: "\(base).gate_proj")
        appendCandidates(for: "\(base).up_proj")
    } else if path.hasSuffix(".switch_mlp.gate_proj") || path.hasSuffix(".switch_mlp.up_proj")
        || path.hasSuffix(".switch_mlp.down_proj")
    {
        appendCandidates(for: path)
    }
    return aliases
}

extension Qwen35TextModel: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen35GateUpQuantizationAliases(for: path)
    }
}

extension Qwen35TextModel: QuantizationPolicyReceiving {}

// Covers `Qwen35MoEModel` through inheritance.
extension Qwen35Model: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen35GateUpQuantizationAliases(for: path)
    }
}

// Covers `Qwen35MoEModel` through inheritance. Storage lives on the wrapped
// text model so its own sanitizer (which the wrapper's sanitize dispatches
// into) makes the identical per-layer fuse decision.
extension Qwen35Model: QuantizationPolicyReceiving {
    public var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization? {
        get { languageModel.checkpointPerLayerQuantization }
        set { languageModel.checkpointPerLayerQuantization = newValue }
    }
}

public class Qwen35MoEModel: Qwen35Model {

    override public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = [String: MLXArray]()
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
            newWeights[key] = value
        }

        newWeights = qwen35FuseSwitchMLPGateUp(
            weights: newWeights,
            perLayerQuantization: checkpointPerLayerQuantization,
            setFused: { qwen35SetSwitchGLUGateUpFused($1, at: $0, in: self) })

        return languageModel.sanitize(weights: newWeights)
    }
}
