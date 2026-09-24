import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXLLM
@testable import MLXLMCommon

@Suite("GPTOSS gate/up fusion", .serialized)
struct GPTOSSGateUpFusionTests {
    private let prefix = "model.layers.0.mlp.experts"

    @Test("row concatenation preserves biased GPTOSS SwiGLU for float, affine, and MXFP4")
    func expertParity() throws {
        let config = try JSONDecoder().decode(GPTOSSConfiguration.self, from: Data("""
        {"model_type":"gpt_oss","num_hidden_layers":2,"num_local_experts":4,
         "num_experts_per_tok":2,"vocab_size":64,"rms_norm_eps":0.00001,
         "hidden_size":64,"intermediate_size":64,"head_dim":16,
         "num_attention_heads":4,"num_key_value_heads":2,"sliding_window":8}
        """.utf8))
        let normalizer = GPTOSSModel(config)
        for mode in [nil, QuantizationMode.affine, .mxfp4] as [QuantizationMode?] {
            MLXRandom.seed(0x475054)
            let split = SwiGLUSwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 4, bias: true)
            let fused = SwiGLUSwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 4,
                                        bias: true, fusedGateUp: true)
            if let mode {
                quantize(model: split, groupSize: 32, bits: 4, mode: mode)
                quantize(model: fused, groupSize: 32, bits: 4, mode: mode)
            }
            eval(split)
            let source = Dictionary(uniqueKeysWithValues: split.parameters().flattened().map {
                (prefix + "." + $0.0, $0.1)
            })
            let adjusted = fuseSwitchGLUGateUpWeights(weights: source, moduleName: "experts")
            let restored = normalizer.splitSavedGateUpWeights(adjusted)
            #expect(restored.keys.sorted() == source.keys.sorted())
            if let mode {
                let policy = BaseConfiguration.Quantization(groupSize: 32, bits: 4, mode: mode)
                let other = BaseConfiguration.Quantization(groupSize: 64, bits: 8, mode: .affine)
                let table = BaseConfiguration.PerLayerQuantization(
                    quantization: other,
                    perLayerQuantization: [prefix + ".gate_up_proj": .quantize(policy)])
                for half in ["gate_proj", "up_proj"] {
                    let resolved = try #require(resolveQuantization(
                        path: prefix + "." + half,
                        perLayerQuantization: table, aliasing: normalizer))
                    #expect(resolved.bits == 4 && resolved.groupSize == 32 && resolved.mode == mode)
                }
            }

            for (key, expected) in source {
                let actual = try #require(restored[key])
                #expect(actual.dtype == expected.dtype && actual.shape == expected.shape)
                #expect((actual .== expected).all().item(Bool.self))
            }
            let bare = adjusted.map { (String($0.key.dropFirst(prefix.count + 1)), $0.value) }
            try fused.update(parameters: ModuleParameters.unflattened(bare), verify: [.all])
            eval(fused)
            for suffix in ["weight", "scales", "biases", "bias"] {
                guard let gate = source[prefix + ".gate_proj." + suffix],
                      let up = source[prefix + ".up_proj." + suffix] else { continue }
                let packed = try #require(adjusted[prefix + ".gate_up_proj." + suffix])
                let expected = concatenated([gate, up], axis: suffix == "bias" ? -1 : -2)
                #expect((packed .== expected).all().item(Bool.self))
            }
            for length in [1, 4, 64] {
                let x = MLXRandom.normal([1, length, 64]) * 0.5
                let ids = MLXArray((0..<(length * 4)).map { UInt32($0 % 4) }).reshaped(1, length, 4)
                let a = split(x, ids), b = fused(x, ids)
                eval(a, b)
                #expect(a.shape == b.shape)
                let error = abs(a - b).max().item(Float.self)
                #expect(error.isFinite && error <= 1e-4,
                        "mode=\(String(describing: mode)), length=\(length), max error=\(error)")
            }
        }
    }

    @Test("mixed policies stay split even when packed shapes match")
    func mixedPolicy() {
        let affine = BaseConfiguration.Quantization(groupSize: 32, bits: 4, mode: .affine)
        let mxfp4 = BaseConfiguration.Quantization(groupSize: 32, bits: 4, mode: .mxfp4)
        let table = BaseConfiguration.PerLayerQuantization(quantization: nil, perLayerQuantization: [
            prefix + ".gate_proj": .quantize(affine),
            prefix + ".up_proj": .quantize(mxfp4),
        ])
        let source = [prefix + ".gate_proj.weight": MLXArray.zeros([4, 64, 8], dtype: .uint32),
                      prefix + ".up_proj.weight": MLXArray.ones([4, 64, 8], dtype: .uint32)]
        var layout: Bool?
        let adjusted = fuseSwitchGLUGateUpWeights(
            weights: source, perLayerQuantization: table, moduleName: "experts",
            setFused: { _, fused in layout = fused })
        #expect(layout == false)
        #expect(adjusted.keys.sorted() == source.keys.sorted())
        for (key, value) in source {
            #expect((adjusted[key]! .== value).all().item(Bool.self))
        }
    }

    @Test("explicit fused policy cannot override the packed split-half format")
    func fusedPolicyConflict() {
        let q4 = BaseConfiguration.Quantization(groupSize: 32, bits: 4, mode: .mxfp4)
        let q8 = BaseConfiguration.Quantization(groupSize: 64, bits: 8, mode: .affine)
        let source = [prefix + ".gate_proj.weight": MLXArray.zeros([4, 64, 8], dtype: .uint32),
                      prefix + ".up_proj.weight": MLXArray.ones([4, 64, 8], dtype: .uint32)]
        for override in [BaseConfiguration.QuantizationOption.quantize(q8), .skip] {
            let table = BaseConfiguration.PerLayerQuantization(quantization: q4, perLayerQuantization: [
                prefix + ".gate_up_proj": override
            ])
            var layout: Bool?
            let result = fuseSwitchGLUGateUpWeights(
                weights: source, perLayerQuantization: table, moduleName: "experts",
                setFused: { _, value in layout = value })
            #expect(layout == false)
            #expect(result.keys.sorted() == source.keys.sorted())
        }
        // No explicit fused entry: the loader aliases to the split paths,
        // so a different default must not spuriously prohibit fusion.
        let aliasTable = BaseConfiguration.PerLayerQuantization(quantization: q8, perLayerQuantization: [
            prefix + ".gate_proj": .quantize(q4), prefix + ".up_proj": .quantize(q4)
        ])
        let aliased = fuseSwitchGLUGateUpWeights(
            weights: source, perLayerQuantization: aliasTable, moduleName: "experts")
        #expect(aliased[prefix + ".gate_up_proj.weight"] != nil)
    }

    @Test("model layout follows split/fused loads and aliases preserve checkpoint policies")
    func modelLayout() throws {
        let config = try JSONDecoder().decode(GPTOSSConfiguration.self, from: Data("""
        {"model_type":"gpt_oss","num_hidden_layers":2,"num_local_experts":4,
         "num_experts_per_tok":2,"vocab_size":64,"rms_norm_eps":0.00001,
         "hidden_size":64,"intermediate_size":64,"head_dim":16,
         "num_attention_heads":4,"num_key_value_heads":2,"sliding_window":8}
        """.utf8))
        let model = GPTOSSModel(config)
        let original = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let adjusted = model.fuseGateUpWeights(original, enabled: true)
        try model.update(parameters: ModuleParameters.unflattened(adjusted), verify: [.all])
        let experts = model.namedModules().compactMap { $0.1 as? SwiGLUSwitchGLU }
        #expect(experts.count == 2 && experts.allSatisfy(\.hasFusedGateUp))
        #expect(model.quantizationPathAliases(for: prefix + ".gate_up_proj") == [
            prefix + ".gate_proj", prefix + ".up_proj"
        ])
        // A saved fused module uses canonical .gate_up_proj.weight keys.
        // Normalize and reload it without interpreting those contiguous
        // halves as the upstream interleaved checkpoint representation.
        let restored = model.sanitize(weights: adjusted)
        try model.update(parameters: ModuleParameters.unflattened(restored), verify: [.all])
        for (key, expected) in original {
            let actual = try #require(restored[key])
            #expect(actual.shape == expected.shape)
            #expect((actual .== expected).all().item(Bool.self))
        }
        let again = model.fuseGateUpWeights(restored, enabled: true)
        let rolledBack = model.fuseGateUpWeights(again, enabled: false)
        try model.update(parameters: ModuleParameters.unflattened(rolledBack), verify: [.all])
        for (key, expected) in original {
            let actual = try #require(rolledBack[key])
            #expect((actual .== expected).all().item(Bool.self))
        }
        let q = BaseConfiguration.Quantization(groupSize: 32, bits: 4, mode: .affine)
        model.checkpointPerLayerQuantization = .init(quantization: nil, perLayerQuantization: [
            prefix + ".gate_proj": .quantize(q), prefix + ".up_proj": .skip
        ])
        let mixed = model.fuseGateUpWeights(original, enabled: true)
        try model.update(parameters: ModuleParameters.unflattened(mixed), verify: [.all])
        let layer0 = try #require(model.namedModules().first { $0.0 == prefix }?.1 as? SwiGLUSwitchGLU)
        #expect(!layer0.hasFusedGateUp)
    }
}
