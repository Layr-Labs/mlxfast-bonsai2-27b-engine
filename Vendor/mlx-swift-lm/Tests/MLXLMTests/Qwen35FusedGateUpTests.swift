import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

// Serialized for the same reason as SwitchGLUTests: SwitchGLU forwards run
// compiled GLU kernels, and parallel tracing can deadlock the per-
// CompiledFunction locks.
@Suite(.serialized)
struct Qwen35FusedGateUpTests {

    /// Fused (`gate_up_proj`) and split (`gate_proj`/`up_proj`) SwitchGLU
    /// must be numerically identical when the fused weights are the
    /// row-concatenation [gate; up] — the exact layout
    /// `qwen35FuseSwitchMLPGateUp` produces.
    @Test func fusedForwardMatchesSplitForwardFloat() {
        let inputDims = 16
        let hiddenDims = 8
        let numExperts = 4
        let split = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        let fused = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            fuseGateUp: true)

        let splitParams = Dictionary(
            uniqueKeysWithValues: split.parameters().flattened())
        let fusedWeights = concatenated(
            [splitParams["gate_proj.weight"]!, splitParams["up_proj.weight"]!],
            axis: -2)
        try! fused.update(
            parameters: ModuleParameters.unflattened([
                "gate_up_proj.weight": fusedWeights,
                "down_proj.weight": splitParams["down_proj.weight"]!,
            ]),
            verify: [.all])

        // Decode-shaped (tiny) and prefill-shaped (sorted path, >= 64
        // assignments) calls.
        for tokens in [1, 16] {
            let x = MLXRandom.normal([tokens, inputDims])
            let indices = MLXRandom.randInt(0 ..< numExperts, [tokens, 8])
                .asType(.uint32)
            let expected = split(x, indices)
            let actual = fused(x, indices)
            eval(expected, actual)
            #expect(expected.shape == actual.shape)
            let maxErr = (actual - expected).abs().max().item(Float.self)
            #expect(
                actual.allClose(expected, rtol: 1e-5, atol: 1e-6).item(Bool.self),
                Comment(rawValue: "fused/split divergence at tokens=\(tokens): max abs err \(maxErr)"))
        }
    }

    /// The sanitize helper must map an MLX split quantized checkpoint onto
    /// the fused layout such that the fused quantized forward reproduces the
    /// split quantized forward exactly (same codes, same scales — row
    /// concatenation cannot change any dequantized value).
    @Test func quantizedSplitCheckpointFusesExactly() {
        let inputDims = 128  // group size 64 needs K % 64 == 0
        let hiddenDims = 64
        let numExperts = 4
        let groupSize = 64
        let bits = 4
        let prefix = "language_model.model.layers.0.mlp.switch_mlp"

        func randomQuantized(_ rows: Int, _ cols: Int) -> (MLXArray, MLXArray, MLXArray) {
            let w = MLXRandom.normal([numExperts, rows, cols]).asType(.bfloat16)
            let q = quantized(w, groupSize: groupSize, bits: bits, mode: .affine)
            return (q.wq, q.scales, q.biases!)
        }
        let gate = randomQuantized(hiddenDims, inputDims)
        let up = randomQuantized(hiddenDims, inputDims)
        let down = randomQuantized(inputDims, hiddenDims)

        var weights: [String: MLXArray] = [
            "\(prefix).gate_proj.weight": gate.0,
            "\(prefix).gate_proj.scales": gate.1,
            "\(prefix).gate_proj.biases": gate.2,
            "\(prefix).up_proj.weight": up.0,
            "\(prefix).up_proj.scales": up.1,
            "\(prefix).up_proj.biases": up.2,
            "\(prefix).down_proj.weight": down.0,
            "\(prefix).down_proj.scales": down.1,
            "\(prefix).down_proj.biases": down.2,
        ]
        weights = qwen35FuseSwitchMLPGateUp(weights: weights)

        // Split keys gone, fused keys present with concatenated shapes.
        #expect(weights["\(prefix).gate_proj.weight"] == nil)
        #expect(weights["\(prefix).up_proj.scales"] == nil)
        let fusedWeight = weights["\(prefix).gate_up_proj.weight"]!
        let fusedScales = weights["\(prefix).gate_up_proj.scales"]!
        let fusedBiases = weights["\(prefix).gate_up_proj.biases"]!
        #expect(fusedWeight.shape == [numExperts, 2 * hiddenDims, inputDims * bits / 32])
        #expect(fusedScales.shape == [numExperts, 2 * hiddenDims, inputDims / groupSize])
        // Idempotence: a second pass is a no-op.
        let again = qwen35FuseSwitchMLPGateUp(weights: weights)
        #expect(again.keys.sorted() == weights.keys.sorted())

        // Forward equivalence, split vs fused, on the quantized modules.
        let tokens = 16
        let x = MLXRandom.normal([tokens, inputDims]).asType(.bfloat16)
        let indices = MLXRandom.randInt(0 ..< numExperts, [tokens, 8]).asType(.uint32)

        let splitGLU = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        let fusedGLU = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            fuseGateUp: true)
        quantize(model: splitGLU) { _, _ in (groupSize, bits, .affine) }
        quantize(model: fusedGLU) { _, _ in (groupSize, bits, .affine) }
        try! splitGLU.update(
            parameters: ModuleParameters.unflattened([
                "gate_proj.weight": gate.0, "gate_proj.scales": gate.1,
                "gate_proj.biases": gate.2,
                "up_proj.weight": up.0, "up_proj.scales": up.1,
                "up_proj.biases": up.2,
                "down_proj.weight": down.0, "down_proj.scales": down.1,
                "down_proj.biases": down.2,
            ]), verify: [.all])
        try! fusedGLU.update(
            parameters: ModuleParameters.unflattened([
                "gate_up_proj.weight": fusedWeight,
                "gate_up_proj.scales": fusedScales,
                "gate_up_proj.biases": fusedBiases,
                "down_proj.weight": down.0, "down_proj.scales": down.1,
                "down_proj.biases": down.2,
            ]), verify: [.all])

        let expected = splitGLU(x, indices)
        let actual = fusedGLU(x, indices)
        eval(expected, actual)
        #expect(expected.shape == actual.shape)
        let maxErr = (actual - expected).abs().max().item(Float.self)
        #expect(
            actual.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            Comment(rawValue: "quantized fused/split divergence: max abs err \(maxErr)"))
    }

    /// Raw HF stacked exports map directly onto the fused module keys with
    /// gate rows first, and `mtp.*` trees are never rewritten.
    @Test func rawStackedExportMapsDirectlyAndMTPKeysAreUntouched() {
        let e = 2
        let ffn = 4
        let hidden = 8
        let stacked = MLXRandom.normal([e, 2 * ffn, hidden])
        let down = MLXRandom.normal([e, hidden, ffn])
        let mtpGate = MLXRandom.normal([e, ffn, hidden])
        var weights: [String: MLXArray] = [
            "language_model.model.layers.0.mlp.experts.gate_up_proj": stacked,
            "language_model.model.layers.0.mlp.experts.down_proj": down,
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight": mtpGate,
            "mtp.layers.0.mlp.switch_mlp.up_proj.weight": mtpGate,
        ]
        weights = qwen35FuseSwitchMLPGateUp(weights: weights)

        let fused = weights["language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.weight"]
        #expect(fused != nil)
        #expect(fused!.shape == [e, 2 * ffn, hidden])
        // Gate rows first: identical array, not re-ordered.
        eval(fused!, stacked)
        #expect(fused!.allClose(stacked).item(Bool.self))
        #expect(
            weights["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"]
                != nil)
        #expect(weights["language_model.model.layers.0.mlp.experts.gate_up_proj"] == nil)
        // MTP stays split.
        #expect(weights["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(weights["mtp.layers.0.mlp.switch_mlp.up_proj.weight"] != nil)
        #expect(weights["mtp.layers.0.mlp.switch_mlp.gate_up_proj.weight"] == nil)
    }

    /// PR #107 P1 comment scenario: a mixed-precision checkpoint whose
    /// per-layer quantization table names the split `gate_proj`/`up_proj`
    /// module paths — with NO default quantization — must still quantize the
    /// fused `gate_up_proj` module. `resolveQuantization` consults the
    /// split-path aliases exactly the way `loadWeights` does, the strict
    /// update then accepts the renamed `.scales`/`.biases` tensors, and the
    /// fused forward reproduces the split forward.
    @Test func heterogeneousPerLayerQuantizationLoadsStrictlyThroughAliases() throws {
        struct GateUpAliases: QuantizationPathAliasing {
            func quantizationPathAliases(for path: String) -> [String] {
                qwen35GateUpQuantizationAliases(for: path)
            }
        }

        let inputDims = 128
        let hiddenDims = 64
        let numExperts = 4
        let prefix = "model.layers.0.mlp.switch_mlp"

        // Heterogeneous: W4/g64 experts, W8/g32 down projection, no default.
        let expertQuant = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let downQuant = BaseConfiguration.Quantization(groupSize: 32, bits: 8)
        let table = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(prefix).gate_proj": .quantize(expertQuant),
                "\(prefix).up_proj": .quantize(expertQuant),
                "\(prefix).down_proj": .quantize(downQuant),
            ])

        func randomQuantized(
            _ rows: Int, _ cols: Int, _ quant: BaseConfiguration.Quantization
        ) -> (MLXArray, MLXArray, MLXArray) {
            let w = MLXRandom.normal([numExperts, rows, cols]).asType(.bfloat16)
            let q = quantized(w, groupSize: quant.groupSize, bits: quant.bits, mode: .affine)
            return (q.wq, q.scales, q.biases!)
        }
        let gate = randomQuantized(hiddenDims, inputDims, expertQuant)
        let up = randomQuantized(hiddenDims, inputDims, expertQuant)
        let down = randomQuantized(inputDims, hiddenDims, downQuant)

        var weights: [String: MLXArray] = [
            "\(prefix).gate_proj.weight": gate.0,
            "\(prefix).gate_proj.scales": gate.1,
            "\(prefix).gate_proj.biases": gate.2,
            "\(prefix).up_proj.weight": up.0,
            "\(prefix).up_proj.scales": up.1,
            "\(prefix).up_proj.biases": up.2,
            "\(prefix).down_proj.weight": down.0,
            "\(prefix).down_proj.scales": down.1,
            "\(prefix).down_proj.biases": down.2,
        ]
        weights = qwen35FuseSwitchMLPGateUp(weights: weights)

        // Without aliasing the fused path resolves to nothing — the exact
        // pre-fix failure (module left unquantized, strict update rejects the
        // renamed scales/biases).
        #expect(
            resolveQuantization(
                path: "\(prefix).gate_up_proj", perLayerQuantization: table,
                aliasing: nil) == nil)
        // With aliasing the split experts' policy applies to the fused path;
        // paths with their own entries resolve directly.
        #expect(
            resolveQuantization(
                path: "\(prefix).gate_up_proj", perLayerQuantization: table,
                aliasing: GateUpAliases()) == expertQuant)
        #expect(
            resolveQuantization(
                path: "\(prefix).down_proj", perLayerQuantization: table,
                aliasing: GateUpAliases()) == downQuant)

        // Replicate the loadWeights flow on the fused module: quantize with
        // alias resolution, then a STRICT update with the sanitized weights.
        let fusedGLU = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            fuseGateUp: true)
        quantize(model: fusedGLU) { path, _ in
            let full = "\(prefix).\(path)"
            guard weights["\(full).scales"] != nil else { return nil }
            return resolveQuantization(
                path: full, perLayerQuantization: table,
                aliasing: GateUpAliases())?.asTuple
        }
        let bare = Dictionary(
            uniqueKeysWithValues: weights.map {
                (String($0.key.dropFirst(prefix.count + 1)), $0.value)
            })
        try fusedGLU.update(parameters: ModuleParameters.unflattened(bare), verify: [.all])

        // Split reference carrying the identical quantized codes.
        let splitGLU = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        quantize(model: splitGLU) { path, _ in
            path == "down_proj" ? downQuant.asTuple : expertQuant.asTuple
        }
        try splitGLU.update(
            parameters: ModuleParameters.unflattened([
                "gate_proj.weight": gate.0, "gate_proj.scales": gate.1,
                "gate_proj.biases": gate.2,
                "up_proj.weight": up.0, "up_proj.scales": up.1,
                "up_proj.biases": up.2,
                "down_proj.weight": down.0, "down_proj.scales": down.1,
                "down_proj.biases": down.2,
            ]), verify: [.all])

        let tokens = 16
        let x = MLXRandom.normal([tokens, inputDims]).asType(.bfloat16)
        let indices = MLXRandom.randInt(0 ..< numExperts, [tokens, 8]).asType(.uint32)
        let expected = splitGLU(x, indices)
        let actual = fusedGLU(x, indices)
        eval(expected, actual)
        #expect(expected.shape == actual.shape)
        let maxErr = (actual - expected).abs().max().item(Float.self)
        #expect(
            actual.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            Comment(rawValue: "heterogeneous fused/split divergence: max abs err \(maxErr)"))
    }

    /// PR #107 P2 comment scenario: the `qwen3_5_text` registry entry
    /// constructs `Qwen35TextModel` directly, so its own sanitizer must fuse
    /// routed experts — both converted split checkpoints and raw stacked HF
    /// exports — for the strict update to match the fused module tree.
    @Test func textModelSanitizeFusesExpertsOnDirectLoadPath() throws {
        let json = """
            {
              "model_type": "qwen3_5_text",
              "hidden_size": 32,
              "num_hidden_layers": 2,
              "intermediate_size": 64,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 8,
              "linear_num_value_heads": 2,
              "linear_num_key_heads": 1,
              "linear_key_head_dim": 32,
              "linear_value_head_dim": 32,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 64,
              "full_attention_interval": 2,
              "num_experts": 4,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 16,
              "shared_expert_intermediate_size": 16
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        let model = Qwen35TextModel(config)
        let reference = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect(reference.keys.contains { $0.hasSuffix(".switch_mlp.gate_up_proj.weight") })

        // Converted split checkpoint: gate/up halves instead of the fused tensor.
        var splitCheckpoint: [String: MLXArray] = [:]
        for (key, value) in reference {
            if key.hasSuffix(".switch_mlp.gate_up_proj.weight") {
                let base = String(key.dropLast("gate_up_proj.weight".count))
                let halves = split(value, parts: 2, axis: -2)
                splitCheckpoint["\(base)gate_proj.weight"] = halves[0]
                splitCheckpoint["\(base)up_proj.weight"] = halves[1]
            } else {
                splitCheckpoint[key] = value
            }
        }
        let sanitizedSplit = model.sanitize(weights: splitCheckpoint)
        try model.update(
            parameters: ModuleParameters.unflattened(sanitizedSplit), verify: [.all])

        // Raw stacked HF export: `<mlp>.experts.gate_up_proj` + `.experts.down_proj`.
        var rawCheckpoint: [String: MLXArray] = [:]
        for (key, value) in reference {
            if key.hasSuffix(".switch_mlp.gate_up_proj.weight") {
                let base = String(key.dropLast(".switch_mlp.gate_up_proj.weight".count))
                rawCheckpoint["\(base).experts.gate_up_proj"] = value
            } else if key.hasSuffix(".switch_mlp.down_proj.weight") {
                let base = String(key.dropLast(".switch_mlp.down_proj.weight".count))
                rawCheckpoint["\(base).experts.down_proj"] = value
            } else {
                rawCheckpoint[key] = value
            }
        }
        let sanitizedRaw = model.sanitize(weights: rawCheckpoint)
        try model.update(
            parameters: ModuleParameters.unflattened(sanitizedRaw), verify: [.all])

        // The direct-path model advertises the split-path quantization
        // aliases that `loadWeights` consults for fused modules. Contract
        // (not an exact list — the candidate matrix spans every key space):
        // the fused name in the OTHER key spaces comes first, then the
        // split halves with gate before up, and the path itself is never
        // its own alias.
        let aliasing = try #require(model as Any as? QuantizationPathAliasing)
        let fusedPath = "model.layers.0.mlp.switch_mlp.gate_up_proj"
        let aliases = aliasing.quantizationPathAliases(for: fusedPath)
        #expect(!aliases.contains(fusedPath))
        #expect(aliases.contains("language_model.model.layers.0.mlp.switch_mlp.gate_up_proj"))
        let gateIdx = try #require(
            aliases.firstIndex(of: "model.layers.0.mlp.switch_mlp.gate_proj"))
        let upIdx = try #require(
            aliases.firstIndex(of: "model.layers.0.mlp.switch_mlp.up_proj"))
        #expect(gateIdx < upIdx)
        let lastFusedIdx = try #require(
            aliases.lastIndex(where: { $0.hasSuffix(".gate_up_proj") }))
        #expect(lastFusedIdx < gateIdx, "fused-name aliases precede split halves")
        // MTP trees stay split and get no aliases; unrelated paths get none.
        #expect(
            aliasing.quantizationPathAliases(
                for: "mtp.layers.0.mlp.switch_mlp.gate_up_proj").isEmpty)
        #expect(aliasing.quantizationPathAliases(for: "model.layers.0.mlp.gate").isEmpty)
    }

    /// PR #107 P2 comment scenario, policy arm: a heterogeneous checkpoint
    /// assigning different quantization policies to `gate_proj` and
    /// `up_proj` (4-bit vs 8-bit here) must NOT fuse that pair. The split
    /// tensors stay verbatim, the `unfuse` callback names the layer, and a
    /// homogeneous pair in the same checkpoint still fuses — the decision is
    /// per layer, not global.
    @Test func heterogeneousGateUpPolicyKeepsPairSplitPerLayer() {
        let inputDims = 128
        let hiddenDims = 64
        let numExperts = 4
        let mixed = "model.layers.0.mlp.switch_mlp"
        let uniform = "model.layers.1.mlp.switch_mlp"

        let gateQuant = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let upQuant = BaseConfiguration.Quantization(groupSize: 64, bits: 8)
        let table = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(mixed).gate_proj": .quantize(gateQuant),
                "\(mixed).up_proj": .quantize(upQuant),
                "\(uniform).gate_proj": .quantize(gateQuant),
                "\(uniform).up_proj": .quantize(gateQuant),
            ])

        func randomQuantized(
            _ quant: BaseConfiguration.Quantization
        ) -> (MLXArray, MLXArray, MLXArray) {
            let w = MLXRandom.normal([numExperts, hiddenDims, inputDims]).asType(.bfloat16)
            let q = quantized(w, groupSize: quant.groupSize, bits: quant.bits, mode: .affine)
            return (q.wq, q.scales, q.biases!)
        }
        var weights: [String: MLXArray] = [:]
        for (prefix, gq, uq) in [(mixed, gateQuant, upQuant), (uniform, gateQuant, gateQuant)] {
            let gate = randomQuantized(gq)
            let up = randomQuantized(uq)
            weights["\(prefix).gate_proj.weight"] = gate.0
            weights["\(prefix).gate_proj.scales"] = gate.1
            weights["\(prefix).gate_proj.biases"] = gate.2
            weights["\(prefix).up_proj.weight"] = up.0
            weights["\(prefix).up_proj.scales"] = up.1
            weights["\(prefix).up_proj.biases"] = up.2
        }
        let original = weights

        var keptSplit: [String] = []
        let sanitized = qwen35FuseSwitchMLPGateUp(
            weights: weights, perLayerQuantization: table,
            setFused: { path, fused in if !fused { keptSplit.append(path) } })

        // Mixed layer: split tensors kept verbatim, nothing fused, callback
        // named exactly this layer's switch_mlp.
        #expect(keptSplit == [mixed])
        #expect(sanitized["\(mixed).gate_up_proj.weight"] == nil)
        for suffix in ["weight", "scales", "biases"] {
            #expect(sanitized["\(mixed).gate_proj.\(suffix)"] === original["\(mixed).gate_proj.\(suffix)"])
            #expect(sanitized["\(mixed).up_proj.\(suffix)"] === original["\(mixed).up_proj.\(suffix)"])
        }
        // Uniform layer: still fused (regression guard on the fusion win).
        #expect(sanitized["\(uniform).gate_up_proj.weight"] != nil)
        #expect(sanitized["\(uniform).gate_proj.weight"] == nil)
        #expect(sanitized["\(uniform).up_proj.scales"] == nil)

        // Even without a policy table the packed-shape backstop keeps the
        // 4-bit/8-bit pair split (packed columns differ).
        var backstop: [String] = []
        let noTable = qwen35FuseSwitchMLPGateUp(
            weights: original,
            setFused: { path, fused in if !fused { backstop.append(path) } })
        #expect(backstop == [mixed])
        #expect(noTable["\(mixed).gate_up_proj.weight"] == nil)
        #expect(noTable["\(uniform).gate_up_proj.weight"] != nil)
    }

    /// PR #107 P2 comment scenario, mode arm: gate/up halves whose packed
    /// `weight` tensors have identical shapes but whose policies differ only
    /// in mode (affine vs mxfp4) must not silently fuse — a fused projection
    /// would reinterpret the up rows under the gate half's mode.
    @Test func sameShapeDifferentModeDoesNotFuse() {
        let inputDims = 128
        let hiddenDims = 64
        let numExperts = 4
        let prefix = "model.layers.0.mlp.switch_mlp"

        let affine = BaseConfiguration.Quantization(groupSize: 32, bits: 4, mode: .affine)
        let mxfp4 = BaseConfiguration.Quantization(groupSize: 32, bits: 4, mode: .mxfp4)
        let table = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(prefix).gate_proj": .quantize(affine),
                "\(prefix).up_proj": .quantize(mxfp4),
            ])

        let w = MLXRandom.normal([numExperts, hiddenDims, inputDims]).asType(.bfloat16)
        let gate = quantized(w, groupSize: 32, bits: 4, mode: .affine)
        let up = quantized(w, groupSize: 32, bits: 4, mode: .mxfp4)
        #expect(up.biases == nil)
        // The trap: packed weight shapes are identical, so only the resolved
        // policies (or the differing tensor sets) reveal the mismatch.
        #expect(gate.wq.shape == up.wq.shape)
        #expect(gate.scales.shape == up.scales.shape)

        var keptSplit: [String] = []
        let sanitized = qwen35FuseSwitchMLPGateUp(
            weights: [
                "\(prefix).gate_proj.weight": gate.wq,
                "\(prefix).gate_proj.scales": gate.scales,
                "\(prefix).gate_proj.biases": gate.biases!,
                "\(prefix).up_proj.weight": up.wq,
                "\(prefix).up_proj.scales": up.scales,
            ],
            perLayerQuantization: table,
            setFused: { path, fused in if !fused { keptSplit.append(path) } })
        #expect(keptSplit == [prefix])
        #expect(sanitized["\(prefix).gate_up_proj.weight"] == nil)
        #expect(sanitized["\(prefix).gate_proj.weight"] != nil)
        #expect(sanitized["\(prefix).up_proj.weight"] != nil)

        // Pure policy arm: even when both halves carry bitwise-identical
        // affine tensors (same shapes, dtypes, and tensor sets), a table
        // declaring different modes must block the fuse — nothing about the
        // tensors themselves can.
        var policyOnly: [String] = []
        let policySanitized = qwen35FuseSwitchMLPGateUp(
            weights: [
                "\(prefix).gate_proj.weight": gate.wq,
                "\(prefix).gate_proj.scales": gate.scales,
                "\(prefix).gate_proj.biases": gate.biases!,
                "\(prefix).up_proj.weight": gate.wq,
                "\(prefix).up_proj.scales": gate.scales,
                "\(prefix).up_proj.biases": gate.biases!,
            ],
            perLayerQuantization: table,
            setFused: { path, fused in if !fused { policyOnly.append(path) } })
        #expect(policyOnly == [prefix])
        #expect(policySanitized["\(prefix).gate_up_proj.weight"] == nil)

        // Same tensors with a homogeneous table fuse as before.
        let homogeneous = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(prefix).gate_proj": .quantize(affine),
                "\(prefix).up_proj": .quantize(affine),
            ])
        let fused = qwen35FuseSwitchMLPGateUp(
            weights: [
                "\(prefix).gate_proj.weight": gate.wq,
                "\(prefix).gate_proj.scales": gate.scales,
                "\(prefix).gate_proj.biases": gate.biases!,
                "\(prefix).up_proj.weight": gate.wq,
                "\(prefix).up_proj.scales": gate.scales,
                "\(prefix).up_proj.biases": gate.biases!,
            ],
            perLayerQuantization: homogeneous)
        #expect(fused["\(prefix).gate_up_proj.weight"] != nil)
        #expect(fused["\(prefix).gate_proj.weight"] == nil)
    }

    @Test func implicitAndExplicitAffinePoliciesFuse() {
        let prefix = "model.layers.0.mlp.switch_mlp"
        let implicitAffine = BaseConfiguration.Quantization(groupSize: 32, bits: 4)
        let explicitAffine =
            BaseConfiguration.Quantization(groupSize: 32, bits: 4, mode: .affine)
        #expect(implicitAffine.mode == explicitAffine.mode)
        #expect(implicitAffine != explicitAffine)

        let table = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(prefix).gate_proj": .quantize(implicitAffine),
                "\(prefix).up_proj": .quantize(explicitAffine),
            ])
        let source = MLXRandom.normal([4, 64, 128]).asType(.bfloat16)
        let half = quantized(source, groupSize: 32, bits: 4, mode: .affine)
        let fused = qwen35FuseSwitchMLPGateUp(
            weights: [
                "\(prefix).gate_proj.weight": half.wq,
                "\(prefix).gate_proj.scales": half.scales,
                "\(prefix).gate_proj.biases": half.biases!,
                "\(prefix).up_proj.weight": half.wq,
                "\(prefix).up_proj.scales": half.scales,
                "\(prefix).up_proj.biases": half.biases!,
            ],
            perLayerQuantization: table)

        #expect(fused["\(prefix).gate_up_proj.weight"] != nil)
        #expect(fused["\(prefix).gate_proj.weight"] == nil)
        #expect(fused["\(prefix).up_proj.weight"] == nil)
    }

    /// PR #107 P2 comment scenario, end to end: a Qwen35TextModel loading a
    /// mixed checkpoint (layer 0 heterogeneous 4-bit gate / 8-bit up, layer 1
    /// homogeneous 4-bit) reshapes ONLY layer 0's SwitchGLU to split modules,
    /// quantizes each half under its own policy, passes the strict update,
    /// and reproduces a split reference forward bit-exactly.
    @Test func textModelKeepsHeterogeneousLayerSplitAndLoadsStrictly() throws {
        let json = """
            {
              "model_type": "qwen3_5_text",
              "hidden_size": 64,
              "num_hidden_layers": 2,
              "intermediate_size": 64,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 8,
              "linear_num_value_heads": 2,
              "linear_num_key_heads": 1,
              "linear_key_head_dim": 32,
              "linear_value_head_dim": 32,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 64,
              "full_attention_interval": 2,
              "num_experts": 4,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 64,
              "shared_expert_intermediate_size": 16
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        let model = Qwen35TextModel(config)
        let reference = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        let gateQuant = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let upQuant = BaseConfiguration.Quantization(groupSize: 64, bits: 8)
        let table = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "model.layers.0.mlp.switch_mlp.gate_proj": .quantize(gateQuant),
                "model.layers.0.mlp.switch_mlp.up_proj": .quantize(upQuant),
                "model.layers.1.mlp.switch_mlp.gate_proj": .quantize(gateQuant),
                "model.layers.1.mlp.switch_mlp.up_proj": .quantize(gateQuant),
            ])
        // Stage the policy exactly the way `loadWeights` does.
        let receiver = try #require(model as Any as? QuantizationPolicyReceiving)
        receiver.checkpointPerLayerQuantization = table

        // Split quantized checkpoint: per-layer policies from the table.
        var checkpoint: [String: MLXArray] = [:]
        for (key, value) in reference {
            guard key.hasSuffix(".switch_mlp.gate_up_proj.weight") else {
                checkpoint[key] = value
                continue
            }
            let base = String(key.dropLast("gate_up_proj.weight".count))
            let modulePath = String(base.dropLast(1))
            let halves = split(value, parts: 2, axis: -2)
            for (half, name) in [(halves[0], "gate_proj"), (halves[1], "up_proj")] {
                let quant = table.quantization(layer: "\(modulePath).\(name)")!
                let q = quantized(
                    half.asType(.bfloat16), groupSize: quant.groupSize, bits: quant.bits,
                    mode: .affine)
                checkpoint["\(base)\(name).weight"] = q.wq
                checkpoint["\(base)\(name).scales"] = q.scales
                checkpoint["\(base)\(name).biases"] = q.biases!
            }
        }

        let sanitized = model.sanitize(weights: checkpoint)

        // Layer 0 stays split; layer 1 fused — in weights AND module tree.
        #expect(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(sanitized["model.layers.0.mlp.switch_mlp.gate_up_proj.weight"] == nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_up_proj.weight"] != nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"] == nil)

        func switchGLU(_ layer: Int) throws -> SwitchGLU {
            try #require(
                model.namedModules().first {
                    $0.0 == "model.layers.\(layer).mlp.switch_mlp"
                }?.1 as? SwitchGLU)
        }
        #expect(try switchGLU(0).hasFusedGateUp == false)
        #expect(try switchGLU(1).hasFusedGateUp == true)

        // Quantize + strict update exactly the way `loadWeights` does.
        let aliasing = model as Any as? QuantizationPathAliasing
        quantize(model: model) { path, _ in
            guard sanitized["\(path).scales"] != nil else { return nil }
            return resolveQuantization(
                path: path, perLayerQuantization: table, aliasing: aliasing)?.asTuple
        }
        try model.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.all])

        // Split reference carrying the identical quantized codes: layer 0's
        // forward must be bit-exact against it (same ops, same tensors).
        let inputDims = 64
        let hiddenDims = 64
        let numExperts = 4
        let splitGLU = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        quantize(model: splitGLU) { path, _ in
            switch path {
            case "gate_proj": return gateQuant.asTuple
            case "up_proj": return upQuant.asTuple
            default: return nil
            }
        }
        let layer0 = "model.layers.0.mlp.switch_mlp"
        try splitGLU.update(
            parameters: ModuleParameters.unflattened([
                "gate_proj.weight": sanitized["\(layer0).gate_proj.weight"]!,
                "gate_proj.scales": sanitized["\(layer0).gate_proj.scales"]!,
                "gate_proj.biases": sanitized["\(layer0).gate_proj.biases"]!,
                "up_proj.weight": sanitized["\(layer0).up_proj.weight"]!,
                "up_proj.scales": sanitized["\(layer0).up_proj.scales"]!,
                "up_proj.biases": sanitized["\(layer0).up_proj.biases"]!,
                "down_proj.weight": sanitized["\(layer0).down_proj.weight"]!,
            ]), verify: [.all])

        let tokens = 16
        let x = MLXRandom.normal([tokens, inputDims]).asType(.bfloat16)
        let indices = MLXRandom.randInt(0 ..< numExperts, [tokens, 8]).asType(.uint32)
        let expected = splitGLU(x, indices)
        let actual = try switchGLU(0)(x, indices)
        eval(expected, actual)
        #expect(expected.shape == actual.shape)
        let maxErr = (actual - expected).abs().max().item(Float.self)
        #expect(
            actual.allClose(expected, rtol: 0, atol: 0).item(Bool.self),
            Comment(rawValue: "split-path load diverged from reference: max abs err \(maxErr)"))
    }

    /// PR #107 P1 follow-up: Qwen VLM mixed-precision configs key their
    /// per-module entries on RAW checkpoint paths
    /// (`model.language_model.layers.*`) while sanitization remaps modules
    /// to `language_model.model.layers.*`. Alias resolution must bridge the
    /// namespaces for the fused name, the split halves, and split modules
    /// kept by the per-layer policy decision — otherwise the fused module
    /// falls back to the default (or nothing) and strict loading rejects the
    /// scale tensors.
    @Test func rawCheckpointKeySpaceAliasesResolvePolicies() throws {
        struct GateUpAliases: QuantizationPathAliasing {
            func quantizationPathAliases(for path: String) -> [String] {
                qwen35GateUpQuantizationAliases(for: path)
            }
        }
        let aliasing = GateUpAliases()

        let inputDims = 128
        let hiddenDims = 64
        let numExperts = 4
        let raw = "model.language_model.layers.0.mlp.switch_mlp"
        let module = "language_model.model.layers.0.mlp.switch_mlp"

        let expertQuant = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let upQuant = BaseConfiguration.Quantization(groupSize: 64, bits: 8)
        let downQuant = BaseConfiguration.Quantization(groupSize: 32, bits: 8)

        // Homogeneous VLM config keyed on raw split paths, NO default: the
        // fused module path must resolve the experts' policy through the
        // cross-namespace aliases.
        let rawSplitTable = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(raw).gate_proj": .quantize(expertQuant),
                "\(raw).up_proj": .quantize(expertQuant),
                "\(raw).down_proj": .quantize(downQuant),
            ])
        #expect(
            resolveQuantization(
                path: "\(module).gate_up_proj", perLayerQuantization: rawSplitTable,
                aliasing: nil) == nil)
        #expect(
            resolveQuantization(
                path: "\(module).gate_up_proj", perLayerQuantization: rawSplitTable,
                aliasing: aliasing) == expertQuant)
        #expect(
            resolveQuantization(
                path: "\(module).down_proj", perLayerQuantization: rawSplitTable,
                aliasing: aliasing) == downQuant)

        // Raw-keyed FUSED entry resolves too (same-name alias in the raw
        // namespace wins before the split-half fallbacks).
        let rawFusedTable = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: ["\(raw).gate_up_proj": .quantize(expertQuant)])
        #expect(
            resolveQuantization(
                path: "\(module).gate_up_proj", perLayerQuantization: rawFusedTable,
                aliasing: aliasing) == expertQuant)

        // Fused strict load driven entirely by the raw-keyed table.
        func randomQuantized(
            _ rows: Int, _ cols: Int, _ quant: BaseConfiguration.Quantization
        ) -> (MLXArray, MLXArray, MLXArray) {
            let w = MLXRandom.normal([numExperts, rows, cols]).asType(.bfloat16)
            let q = quantized(w, groupSize: quant.groupSize, bits: quant.bits, mode: .affine)
            return (q.wq, q.scales, q.biases!)
        }
        let gate = randomQuantized(hiddenDims, inputDims, expertQuant)
        let up = randomQuantized(hiddenDims, inputDims, expertQuant)
        let down = randomQuantized(inputDims, hiddenDims, downQuant)
        var weights: [String: MLXArray] = [
            "\(module).gate_proj.weight": gate.0, "\(module).gate_proj.scales": gate.1,
            "\(module).gate_proj.biases": gate.2,
            "\(module).up_proj.weight": up.0, "\(module).up_proj.scales": up.1,
            "\(module).up_proj.biases": up.2,
            "\(module).down_proj.weight": down.0, "\(module).down_proj.scales": down.1,
            "\(module).down_proj.biases": down.2,
        ]
        weights = qwen35FuseSwitchMLPGateUp(
            weights: weights, perLayerQuantization: rawSplitTable)
        #expect(weights["\(module).gate_up_proj.weight"] != nil)

        let fusedGLU = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            fuseGateUp: true)
        quantize(model: fusedGLU) { path, _ in
            let full = "\(module).\(path)"
            guard weights["\(full).scales"] != nil else { return nil }
            return resolveQuantization(
                path: full, perLayerQuantization: rawSplitTable,
                aliasing: aliasing)?.asTuple
        }
        let bare = Dictionary(
            uniqueKeysWithValues: weights.map {
                (String($0.key.dropFirst(module.count + 1)), $0.value)
            })
        try fusedGLU.update(parameters: ModuleParameters.unflattened(bare), verify: [.all])

        // Heterogeneous raw-keyed config: the pair stays split (the policy
        // candidates bridge the namespaces), and each kept-split module path
        // resolves its own raw entry through the aliases.
        let heterogeneous = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(raw).gate_proj": .quantize(expertQuant),
                "\(raw).up_proj": .quantize(upQuant),
            ])
        var keptSplit: [String] = []
        let sanitized = qwen35FuseSwitchMLPGateUp(
            weights: [
                "\(module).gate_proj.weight": gate.0,
                "\(module).gate_proj.scales": gate.1,
                "\(module).gate_proj.biases": gate.2,
                "\(module).up_proj.weight": up.0,
                "\(module).up_proj.scales": up.1,
                "\(module).up_proj.biases": up.2,
            ],
            perLayerQuantization: heterogeneous,
            setFused: { path, fused in if !fused { keptSplit.append(path) } })
        #expect(keptSplit == [module])
        #expect(sanitized["\(module).gate_up_proj.weight"] == nil)
        #expect(
            resolveQuantization(
                path: "\(module).gate_proj", perLayerQuantization: heterogeneous,
                aliasing: aliasing) == expertQuant)
        #expect(
            resolveQuantization(
                path: "\(module).up_proj", perLayerQuantization: heterogeneous,
                aliasing: aliasing) == upQuant)
    }

    /// PR #107 P1 follow-up (bare candidate for raw VLM paths): the VLM
    /// sanitizer fuses on RAW checkpoint keys (`model.language_model.*`)
    /// BEFORE the namespace remap, while mixed-precision tables may key the
    /// equivalent bare `model.*` paths. The policy candidates must bridge
    /// raw→bare directly — otherwise both halves fall to the fallback
    /// policy, a heterogeneous pair fuses, and load-time aliasing applies
    /// the gate policy to the up rows (silent dequant corruption).
    @Test func bareKeyedTableResolvesAgainstRawVlmPaths() throws {
        struct GateUpAliases: QuantizationPathAliasing {
            func quantizationPathAliases(for path: String) -> [String] {
                qwen35GateUpQuantizationAliases(for: path)
            }
        }
        let aliasing = GateUpAliases()

        let raw = "model.language_model.layers.0.mlp.switch_mlp"
        let bare = "model.layers.0.mlp.switch_mlp"
        let gateQuant = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let upQuant = BaseConfiguration.Quantization(groupSize: 64, bits: 8)

        // Heterogeneous table keyed on BARE paths, weights arriving RAW
        // (the VLM pre-remap fuse call): the pair must stay split.
        let bareHeterogeneous = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(bare).gate_proj": .quantize(gateQuant),
                "\(bare).up_proj": .quantize(upQuant),
            ])
        let inputDims = 128
        let hiddenDims = 64
        let numExperts = 4
        func randomQuantized(
            _ rows: Int, _ cols: Int, _ quant: BaseConfiguration.Quantization
        ) -> (MLXArray, MLXArray, MLXArray) {
            let w = MLXRandom.normal([numExperts, rows, cols]).asType(.bfloat16)
            let q = quantized(w, groupSize: quant.groupSize, bits: quant.bits, mode: .affine)
            return (q.wq, q.scales, q.biases!)
        }
        let gate = randomQuantized(hiddenDims, inputDims, gateQuant)
        let up = randomQuantized(hiddenDims, inputDims, upQuant)
        var keptSplit: [String] = []
        let sanitized = qwen35FuseSwitchMLPGateUp(
            weights: [
                "\(raw).gate_proj.weight": gate.0,
                "\(raw).gate_proj.scales": gate.1,
                "\(raw).gate_proj.biases": gate.2,
                "\(raw).up_proj.weight": up.0,
                "\(raw).up_proj.scales": up.1,
                "\(raw).up_proj.biases": up.2,
            ],
            perLayerQuantization: bareHeterogeneous,
            setFused: { path, fused in if !fused { keptSplit.append(path) } })
        #expect(keptSplit == [raw])
        #expect(sanitized["\(raw).gate_up_proj.weight"] == nil)
        #expect(sanitized["\(raw).gate_proj.weight"] != nil)

        // Each kept-split RAW module path resolves its own BARE entry.
        #expect(
            resolveQuantization(
                path: "\(raw).gate_proj", perLayerQuantization: bareHeterogeneous,
                aliasing: aliasing) == gateQuant)
        #expect(
            resolveQuantization(
                path: "\(raw).up_proj", perLayerQuantization: bareHeterogeneous,
                aliasing: aliasing) == upQuant)

        // Homogeneous bare-keyed table: the same raw-keyed pair fuses, and
        // the fused module path (in ANY key space) resolves the policy.
        let bareHomogeneous = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(bare).gate_proj": .quantize(gateQuant),
                "\(bare).up_proj": .quantize(gateQuant),
            ])
        let up2 = randomQuantized(hiddenDims, inputDims, gateQuant)
        var fusedPaths: [String] = []
        let fusedSanitized = qwen35FuseSwitchMLPGateUp(
            weights: [
                "\(raw).gate_proj.weight": gate.0,
                "\(raw).gate_proj.scales": gate.1,
                "\(raw).gate_proj.biases": gate.2,
                "\(raw).up_proj.weight": up2.0,
                "\(raw).up_proj.scales": up2.1,
                "\(raw).up_proj.biases": up2.2,
            ],
            perLayerQuantization: bareHomogeneous,
            setFused: { path, fused in if fused { fusedPaths.append(path) } })
        #expect(fusedPaths == [raw])
        #expect(fusedSanitized["\(raw).gate_up_proj.weight"] != nil)
        for space in [
            raw, "language_model.model.layers.0.mlp.switch_mlp", bare,
        ] {
            #expect(
                resolveQuantization(
                    path: "\(space).gate_up_proj", perLayerQuantization: bareHomogeneous,
                    aliasing: aliasing) == gateQuant,
                "fused policy must resolve from key space \(space)")
        }
    }

    /// PR #107 P2 follow-up: a half-split checkpoint (gate tensors without
    /// their up counterparts) must surface as a catchable load error, not a
    /// preconditionFailure that kills the host process from inside the
    /// throwing `loadWeights` API. The sanitizer leaves the orphaned tensors
    /// untouched and the strict update rejects them by name.
    @Test func halfSplitCheckpointSurfacesCatchableError() throws {
        let json = """
            {
              "model_type": "qwen3_5_text",
              "hidden_size": 32,
              "num_hidden_layers": 2,
              "intermediate_size": 64,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 8,
              "linear_num_value_heads": 2,
              "linear_num_key_heads": 1,
              "linear_key_head_dim": 32,
              "linear_value_head_dim": 32,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 64,
              "full_attention_interval": 2,
              "num_experts": 4,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 16,
              "shared_expert_intermediate_size": 16
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        let model = Qwen35TextModel(config)
        let reference = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        // Corrupt checkpoint: layer 0 carries gate_proj but its up_proj half
        // is missing (partial download / bad conversion).
        var checkpoint: [String: MLXArray] = [:]
        for (key, value) in reference {
            if key.hasSuffix(".switch_mlp.gate_up_proj.weight") {
                let base = String(key.dropLast("gate_up_proj.weight".count))
                let halves = split(value, parts: 2, axis: -2)
                checkpoint["\(base)gate_proj.weight"] = halves[0]
                if !key.contains("layers.0.") {
                    checkpoint["\(base)up_proj.weight"] = halves[1]
                }
            } else {
                checkpoint[key] = value
            }
        }

        // Sanitize must not trap; the orphaned half stays for verification.
        let sanitized = model.sanitize(weights: checkpoint)
        #expect(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(sanitized["model.layers.0.mlp.switch_mlp.gate_up_proj.weight"] == nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_up_proj.weight"] != nil)

        // The strict update throws a catchable error naming the bad tensors;
        // the process stays alive.
        var caught: Error?
        do {
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        } catch {
            caught = error
        }
        let error = try #require(caught)
        #expect(
            String(describing: error).contains("gate"),
            Comment(rawValue: "error should name the offending tensors: \(error)"))
    }

    /// PR #107 P2 follow-up: the per-layer topology decision must follow
    /// EACH load, not ratchet one way. Three-phase round-trip on one model
    /// instance: homogeneous load (fused) → heterogeneous load (split,
    /// strict-green) → homogeneous load again (re-fused, strict-green,
    /// forward parity with the first load).
    @Test func topologyRoundTripsAcrossReloads() throws {
        let json = """
            {
              "model_type": "qwen3_5_text",
              "hidden_size": 64,
              "num_hidden_layers": 2,
              "intermediate_size": 64,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 8,
              "linear_num_value_heads": 2,
              "linear_num_key_heads": 1,
              "linear_key_head_dim": 32,
              "linear_value_head_dim": 32,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 64,
              "full_attention_interval": 2,
              "num_experts": 4,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 64,
              "shared_expert_intermediate_size": 16
            }
            """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        let model = Qwen35TextModel(config)
        let reference = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        let fourBit = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let eightBit = BaseConfiguration.Quantization(groupSize: 64, bits: 8)
        func table(upBitsLayer0 upQuant: BaseConfiguration.Quantization)
            -> BaseConfiguration.PerLayerQuantization
        {
            BaseConfiguration.PerLayerQuantization(
                quantization: nil,
                perLayerQuantization: [
                    "model.layers.0.mlp.switch_mlp.gate_proj": .quantize(fourBit),
                    "model.layers.0.mlp.switch_mlp.up_proj": .quantize(upQuant),
                    "model.layers.1.mlp.switch_mlp.gate_proj": .quantize(fourBit),
                    "model.layers.1.mlp.switch_mlp.up_proj": .quantize(fourBit),
                ])
        }
        let homogeneous = table(upBitsLayer0: fourBit)
        let heterogeneous = table(upBitsLayer0: eightBit)

        // Split quantized checkpoint for a given policy table, derived from
        // the reference float weights.
        func checkpoint(for policy: BaseConfiguration.PerLayerQuantization)
            -> [String: MLXArray]
        {
            var checkpoint: [String: MLXArray] = [:]
            for (key, value) in reference {
                guard key.hasSuffix(".switch_mlp.gate_up_proj.weight") else {
                    checkpoint[key] = value
                    continue
                }
                let base = String(key.dropLast("gate_up_proj.weight".count))
                let modulePath = String(base.dropLast(1))
                let halves = split(value, parts: 2, axis: -2)
                for (half, name) in [(halves[0], "gate_proj"), (halves[1], "up_proj")] {
                    let quant = policy.quantization(layer: "\(modulePath).\(name)")!
                    let q = quantized(
                        half.asType(.bfloat16), groupSize: quant.groupSize,
                        bits: quant.bits, mode: .affine)
                    checkpoint["\(base)\(name).weight"] = q.wq
                    checkpoint["\(base)\(name).scales"] = q.scales
                    checkpoint["\(base)\(name).biases"] = q.biases!
                }
            }
            return checkpoint
        }

        // One full loadWeights-shaped pass: stage policy, sanitize, quantize
        // with alias resolution, strict update.
        let aliasing = model as Any as? QuantizationPathAliasing
        func load(_ policy: BaseConfiguration.PerLayerQuantization) throws {
            (model as Any as? QuantizationPolicyReceiving)?
                .checkpointPerLayerQuantization = policy
            let sanitized = model.sanitize(weights: checkpoint(for: policy))
            quantize(model: model) { path, _ in
                guard sanitized["\(path).scales"] != nil else { return nil }
                return resolveQuantization(
                    path: path, perLayerQuantization: policy, aliasing: aliasing)?.asTuple
            }
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
        }
        func layer0GLU() throws -> SwitchGLU {
            try #require(
                model.namedModules().first {
                    $0.0 == "model.layers.0.mlp.switch_mlp"
                }?.1 as? SwitchGLU)
        }

        let tokens = 16
        let x = MLXRandom.normal([tokens, 64]).asType(.bfloat16)
        let indices = MLXRandom.randInt(0 ..< 4, [tokens, 8]).asType(.uint32)

        // Phase 1: homogeneous → fused.
        try load(homogeneous)
        #expect(try layer0GLU().hasFusedGateUp == true)
        let phase1 = try layer0GLU()(x, indices)
        eval(phase1)

        // Phase 2: heterogeneous on the SAME instance → split, strict-green.
        try load(heterogeneous)
        #expect(try layer0GLU().hasFusedGateUp == false)
        eval(try layer0GLU()(x, indices))

        // Phase 3: homogeneous again → re-fused, strict-green, and forward
        // parity with phase 1 (identical checkpoint, identical codes).
        try load(homogeneous)
        #expect(try layer0GLU().hasFusedGateUp == true)
        let phase3 = try layer0GLU()(x, indices)
        eval(phase3)
        #expect(phase1.shape == phase3.shape)
        let maxErr = (phase3 - phase1).abs().max().item(Float.self)
        #expect(
            phase3.allClose(phase1, rtol: 0, atol: 0).item(Bool.self),
            Comment(rawValue: "re-fused forward diverged from first fused load: max abs err \(maxErr)"))
    }
}
