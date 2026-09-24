//
//  Qwen35MoE.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/25.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_5_moe
//

import MLX
import MLXLLM
import MLXLMCommon

public final class Qwen35MoE: Qwen35 {
    // Routed experts run with `SwitchGLU(fuseGateUp: true)`; the shared
    // helper maps raw HF stacked `experts.gate_up_proj` exports and
    // MLX-converted split `switch_mlp.{gate,up}_proj.*` checkpoints
    // (quantized included) onto the fused `switch_mlp.gate_up_proj.*`
    // layout. Suffix matching covers both the `model.language_model.*`
    // and `language_model.model.*` key spaces. The fuse decision is per
    // layer AND per load: pairs whose gate/up halves resolve to different
    // quantization policies stay split, and the `setFused` callback
    // reshapes that layer's `SwitchGLU` in either direction so a later
    // homogeneous load restores the fused layout.
    //
    // The metadata variant MUST be overridden too: for `format == "mlx"`
    // checkpoints (the production artifact) the base implementation
    // bypasses `sanitize(weights:)` entirely, and those checkpoints carry
    // the split quantized tensors that need fusing. The helper is
    // idempotent, so the double application on the non-MLX path (base
    // dispatches back through the subclass `sanitize(weights:)`) is safe.
    public override func sanitize(
        weights: [String: MLXArray], metadata: [String: String]
    ) -> [String: MLXArray] {
        qwen35FuseSwitchMLPGateUp(
            weights: super.sanitize(weights: weights, metadata: metadata),
            perLayerQuantization: checkpointPerLayerQuantization,
            setFused: { qwen35SetSwitchGLUGateUpFused($1, at: $0, in: self) })
    }

    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        super.sanitize(
            weights: qwen35FuseSwitchMLPGateUp(
                weights: weights,
                perLayerQuantization: checkpointPerLayerQuantization,
                setFused: { qwen35SetSwitchGLUGateUpFused($1, at: $0, in: self) }))
    }
}

// On the base class so both the dense-registry and MoE-registry entries
// resolve fused `gate_up_proj` quantization through the split-path
// overrides that mixed-precision configs are keyed on.
extension Qwen35: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        qwen35GateUpQuantizationAliases(for: path)
    }
}

extension Qwen35: QuantizationPolicyReceiving {}
