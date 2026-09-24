import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

private func qwenInlineMTPConfig(blockSize: Int = 3, prefix: String = "mtp.") -> Data {
    Data(
        """
        {
          "model_type": "qwen3_5_moe",
          "mtplx_mtp": {
            "included": true,
            "prefix": "\(prefix)",
            "block_size": \(blockSize)
          },
          "mtplx_mtp_quantization": {
            "group_size": 32,
            "bits": 8,
            "mode": "mxfp8",
            "layers.0.self_attn.q_proj": {
              "group_size": 32,
              "bits": 8,
              "mode": "mxfp8"
            }
          },
          "text_config": {
            "model_type": "qwen3_5_moe",
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "linear_num_value_heads": 1,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 32,
            "head_dim": 8,
            "full_attention_interval": 4,
            "num_experts": 0,
            "num_experts_per_tok": 0,
            "mtp_num_hidden_layers": 1
          }
        }
        """.utf8)
}

private func qwenStandaloneMTPConfig(blockSize: Int = 3) -> Data {
    Data(
        """
        {
          "block_size": \(blockSize),
          "model_type": "qwen3_5_mtp",
          "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine"
          },
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "intermediate_size": 17408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_value_heads": 48,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 248320,
            "full_attention_interval": 4,
            "max_position_embeddings": 262144,
            "mtp_num_hidden_layers": 1
          }
        }
        """.utf8)
}

private func qwenStandaloneMTPConfig(quantization: Any?) throws -> Data {
    try qwenStandaloneMTPConfig(
        quantization: quantization,
        quantizationConfig: nil)
}

private func qwenStandaloneMTPConfig(
    quantization: Any?,
    quantizationConfig: Any?
) throws -> Data {
    var root = try #require(
        try JSONSerialization.jsonObject(with: qwenStandaloneMTPConfig()) as? [String: Any])
    if let quantization {
        root["quantization"] = quantization
    } else {
        root.removeValue(forKey: "quantization")
    }
    if let quantizationConfig {
        root["quantization_config"] = quantizationConfig
    } else {
        root.removeValue(forKey: "quantization_config")
    }
    return try JSONSerialization.data(withJSONObject: root)
}

private func qwenSmallStandaloneMTPConfig(quantization: Any?) throws -> Data {
    var root = try #require(
        try JSONSerialization.jsonObject(with: qwenInlineMTPConfig()) as? [String: Any])
    var text = try #require(root["text_config"] as? [String: Any])
    text["model_type"] = "qwen3_5_text"
    root["model_type"] = "qwen3_5_mtp"
    root["block_size"] = 3
    root["text_config"] = text
    root.removeValue(forKey: "mtplx_mtp")
    root.removeValue(forKey: "mtplx_mtp_quantization")
    if let quantization {
        root["quantization"] = quantization
    }
    return try JSONSerialization.data(withJSONObject: root)
}

private func qwenTextConfiguration(from data: Data) throws -> Qwen35TextConfiguration {
    let root = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any])
    return try JSONDecoder.json5().decode(
        Qwen35TextConfiguration.self,
        from: try JSONSerialization.data(withJSONObject: root["text_config"]!))
}

private func withStandaloneMTPDirectory(
    config: Data,
    weights: [String: MLXArray],
    _ body: (URL) throws -> Void
) throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("qwen-standalone-mtp-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try config.write(to: directory.appendingPathComponent("config.json"))
    try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
    try body(directory)
}

private func withInlineMTPDirectory(
    blockSize: Int = 3,
    weights: [String: MLXArray] = ["mtp.unexpected.weight": MLXArray([Float(1)])],
    _ body: (URL) throws -> Void
) throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("qwen-inline-mtp-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try qwenInlineMTPConfig(blockSize: blockSize)
        .write(to: directory.appendingPathComponent("config.json"))
    let shardName = "model-00001-of-00001.safetensors"
    try save(arrays: weights, url: directory.appendingPathComponent(shardName))
    let index = ["weight_map": Dictionary(uniqueKeysWithValues: weights.keys.map { ($0, shardName) })]
    try JSONSerialization.data(withJSONObject: index)
        .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
    try body(directory)
}

@Suite("Qwen inline MTP loader")
struct Qwen35InlineMTPLoaderTests {
    @Test("explicit verification mode overrides the assistant default")
    func explicitVerificationModeWinsAtConstruction() {
        #expect(
            Qwen35InlineMTPAssistant.resolvedVerificationMode(
                requested: .serialTarget, forceSerialEnvironment: false) == .serialTarget)
        #expect(
            Qwen35InlineMTPAssistant.resolvedVerificationMode(
                requested: .rectangular, forceSerialEnvironment: true) == .serialTarget)
        #expect(
            Qwen35InlineMTPAssistant.resolvedVerificationMode(
                requested: nil, forceSerialEnvironment: false) == .rectangular)
        #expect(
            Qwen35InlineMTPAssistant.resolvedVerificationMode(
                requested: nil, forceSerialEnvironment: true) == .serialTarget)
    }

    @Test("exact verification rejects a target built with ordinary arithmetic")
    func exactVerificationRequiresExactTargetArithmetic() throws {
        try withInlineMTPDirectory { directory in
            let root = try #require(
                try JSONSerialization.jsonObject(with: qwenInlineMTPConfig())
                    as? [String: Any])
            let textData = try JSONSerialization.data(withJSONObject: root["text_config"]!)
            let configuration = try JSONDecoder.json5().decode(
                Qwen35TextConfiguration.self, from: textData)
            let target = Qwen35TextModel(configuration)

            do {
                _ = try Qwen35InlineMTPAssistant.load(
                    from: directory, target: target,
                    verificationMode: .rectangularExact)
                Issue.record("non-exact target accepted rectangular-exact verification")
            } catch let error as Qwen35InlineMTPError {
                #expect(error == .invalidConfiguration(
                    "rectangular_exact verification requires exact target arithmetic"))
            }
        }
    }

    @Test("loader is artifact-scoped and does not mutate the legacy process flag")
    func artifactScopedFlag() throws {
        _qwen35MTPEnabled = false
        try withInlineMTPDirectory { directory in
            let config = try JSONDecoder.json5().decode(
                Qwen35TextConfiguration.self,
                from: try JSONSerialization.data(
                    withJSONObject: (try JSONSerialization.jsonObject(
                        with: qwenInlineMTPConfig()) as! [String: Any])["text_config"]!))
            let target = Qwen35TextModel(config)
            #expect(throws: (any Error).self) {
                _ = try Qwen35InlineMTPAssistant.load(from: directory, target: target)
            }
            #expect(_qwen35MTPEnabled == false)
        }
    }

    @Test("block size is bounded before weight loading")
    func boundedBlockSize() throws {
        try withInlineMTPDirectory(blockSize: 9) { directory in
            let config = try JSONDecoder.json5().decode(
                Qwen35TextConfiguration.self,
                from: try JSONSerialization.data(
                    withJSONObject: (try JSONSerialization.jsonObject(
                        with: qwenInlineMTPConfig()) as! [String: Any])["text_config"]!))
            let target = Qwen35TextModel(config)
            #expect(throws: Qwen35InlineMTPError.self) {
                _ = try Qwen35InlineMTPAssistant.load(from: directory, target: target)
            }
        }
    }

    @Test("top-level quantization is retained as the default")
    func defaultQuantization() throws {
        try withInlineMTPDirectory { directory in
            let metadata = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
            let fallback = metadata.resolvedQuantization(for: "fc")
            #expect(fallback?.groupSize == 32)
            #expect(fallback?.bits == 8)
            #expect(fallback?.mode == .mxfp8)
            #expect(
                metadata.resolvedQuantization(
                    for: "layers.0.self_attn.q_proj")?.mode == .mxfp8)
        }
    }

    @Test("per-layer-only quantization resolves without a global fallback")
    func perLayerOnlyQuantization() throws {
        let path = "layers.0.self_attn.q_proj"
        var root = try #require(
            try JSONSerialization.jsonObject(with: qwenInlineMTPConfig())
                as? [String: Any])
        root["mtplx_mtp_quantization"] = [
            "quant_method": "mlx",
            "linear_class": "QuantizedLinear",
            "quantization_mode": "mixed",
            path: [
                "group_size": 32,
                "bits": 8,
                "mode": "mxfp8",
            ],
        ]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-inline-mtp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: root)
            .write(to: directory.appendingPathComponent("config.json"))

        let metadata = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
        #expect(metadata.quantization?.quantization == nil)
        #expect(
            metadata.resolvedQuantization(for: path)
                == BaseConfiguration.Quantization(
                    groupSize: 32, bits: 8, mode: .mxfp8))
        #expect(metadata.resolvedQuantization(for: "fc") == nil)
    }

    @Test("per-layer quantization overrides the global fallback")
    func perLayerQuantizationOverridesGlobal() throws {
        let path = "layers.0.self_attn.q_proj"
        let config = try qwenStandaloneMTPConfig(quantization: [
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            path: [
                "group_size": 32,
                "bits": 8,
                "mode": "mxfp8",
            ],
        ])
        try withStandaloneMTPDirectory(
            config: config, weights: ["placeholder": MLXArray([Float(0)])]
        ) { directory in
            let metadata = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
            let override = metadata.resolvedQuantization(for: path)
            #expect(override?.groupSize == 32)
            #expect(override?.bits == 8)
            #expect(override?.mode == .mxfp8)
        }
    }

    @Test("incomplete global quantization metadata is rejected")
    func incompleteGlobalQuantizationIsRejected() throws {
        let incomplete: [[String: Any]] = [
            ["group_size": 64],
            ["bits": 4],
            ["mode": "mxfp8"],
        ]
        for quantization in incomplete {
            let config = try qwenStandaloneMTPConfig(quantization: quantization)
            try withStandaloneMTPDirectory(
                config: config, weights: ["placeholder": MLXArray([Float(0)])]
            ) { directory in
                #expect(throws: (any Error).self) {
                    _ = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
                }
            }
        }
    }

    @Test("standalone Qwen3.8 MTP metadata uses unprefixed weights")
    func standaloneMetadata() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-standalone-mtp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try qwenStandaloneMTPConfig()
            .write(to: directory.appendingPathComponent("config.json"))

        let metadata = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
        #expect(metadata.prefix.isEmpty)
        #expect(metadata.blockSize == 3)
        let fallback = metadata.quantization?.quantization(layer: "fc")
        #expect(fallback?.groupSize == 64)
        #expect(fallback?.bits == 4)
        #expect(fallback?.mode == .affine)
        #expect(metadata.textConfiguration.modelType == "qwen3_5_text")
        #expect(metadata.textConfiguration.hiddenLayers == 64)
        #expect(metadata.textConfiguration.mtpNumHiddenLayers == 1)
    }

    @Test("dotted false quantization entries suppress the global fallback")
    func dottedFalseQuantizationEntryIsSkip() throws {
        let path = "layers.0.self_attn.q_norm"
        let config = try qwenStandaloneMTPConfig(quantization: [
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            path: false,
        ])
        try withStandaloneMTPDirectory(
            config: config, weights: ["placeholder": MLXArray([Float(0)])]
        ) { directory in
            let metadata = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
            #expect(metadata.resolvedQuantization(for: path) == nil)
            let fallback = metadata.resolvedQuantization(for: "fc")
            #expect(fallback?.groupSize == 64)
            #expect(fallback?.bits == 4)
            #expect(fallback?.mode == .affine)
        }
    }

    @Test("undotted false quantization entries suppress the global fallback")
    func undottedFalseQuantizationEntryIsSkip() throws {
        let config = try qwenStandaloneMTPConfig(quantization: [
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "fc": false,
        ])
        try withStandaloneMTPDirectory(
            config: config, weights: ["placeholder": MLXArray([Float(0)])]
        ) { directory in
            let metadata = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
            #expect(metadata.resolvedQuantization(for: "fc") == nil)
            let fallback = metadata.resolvedQuantization(
                for: "layers.0.self_attn.q_proj")
            #expect(fallback?.groupSize == 64)
            #expect(fallback?.bits == 4)
            #expect(fallback?.mode == .affine)
        }
    }

    @Test("dotted true, string, and number quantization entries are rejected")
    func malformedDottedQuantizationEntriesAreRejected() throws {
        let path = "layers.0.self_attn.q_norm"
        for malformed in [true as Any, "false" as Any, 0 as Any] {
            let config = try qwenStandaloneMTPConfig(quantization: [
                "group_size": 64,
                "bits": 4,
                path: malformed,
            ])
            try withStandaloneMTPDirectory(
                config: config, weights: ["placeholder": MLXArray([Float(0)])]
            ) { directory in
                #expect(throws: (any Error).self) {
                    _ = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
                }
            }
        }
    }

    @Test("undotted fc true, string, number, array, and null are rejected")
    func malformedUndottedFCQuantizationEntriesAreRejected() throws {
        let malformed: [Any] = [true, "false", 0, [false], NSNull()]
        for value in malformed {
            let config = try qwenStandaloneMTPConfig(quantization: [
                "group_size": 64,
                "bits": 4,
                "fc": value,
            ])
            try withStandaloneMTPDirectory(
                config: config, weights: ["placeholder": MLXArray([Float(0)])]
            ) { directory in
                do {
                    _ = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
                    Issue.record("malformed fc quantization entry was accepted")
                } catch let error as Qwen35InlineMTPError {
                    #expect(
                        error == .invalidConfiguration(
                            "quantization entry fc must be false or an object"))
                } catch {
                    Issue.record("unexpected error: \(error)")
                }
            }
        }
    }

    @Test("null primary quantization selects quantization_config")
    func nullPrimaryQuantizationUsesFallback() throws {
        let config = try qwenStandaloneMTPConfig(
            quantization: NSNull(),
            quantizationConfig: [
                "group_size": 32,
                "bits": 8,
                "mode": "mxfp8",
            ])
        try withStandaloneMTPDirectory(
            config: config, weights: ["placeholder": MLXArray([Float(0)])]
        ) { directory in
            let metadata = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
            let fallback = metadata.resolvedQuantization(for: "fc")
            #expect(fallback?.groupSize == 32)
            #expect(fallback?.bits == 8)
            #expect(fallback?.mode == .mxfp8)
        }
    }

    @Test("null primary and fallback quantization mean unquantized")
    func nullStandaloneQuantizationIsUnquantized() throws {
        let config = try qwenStandaloneMTPConfig(
            quantization: NSNull(),
            quantizationConfig: NSNull())
        try withStandaloneMTPDirectory(
            config: config, weights: ["placeholder": MLXArray([Float(0)])]
        ) { directory in
            let metadata = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
            #expect(metadata.quantization == nil)
        }
    }

    @Test("malformed primary quantization does not select a valid fallback")
    func malformedPrimaryQuantizationIsRejectedBeforeFallback() throws {
        let config = try qwenStandaloneMTPConfig(
            quantization: "invalid",
            quantizationConfig: [
                "group_size": 32,
                "bits": 8,
                "mode": "mxfp8",
            ])
        try withStandaloneMTPDirectory(
            config: config, weights: ["placeholder": MLXArray([Float(0)])]
        ) { directory in
            do {
                _ = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
                Issue.record("malformed primary quantization selected the fallback")
            } catch let error as Qwen35InlineMTPError {
                #expect(
                    error == .invalidConfiguration(
                        "standalone MTP quantization must be an object"))
            } catch {
                Issue.record("unexpected error: \(error)")
            }
        }
    }

    @Test("valid primary quantization takes precedence over fallback")
    func validPrimaryQuantizationTakesPrecedence() throws {
        let config = try qwenStandaloneMTPConfig(
            quantization: [
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
            ],
            quantizationConfig: [
                "group_size": 32,
                "bits": 8,
                "mode": "mxfp8",
            ])
        try withStandaloneMTPDirectory(
            config: config, weights: ["placeholder": MLXArray([Float(0)])]
        ) { directory in
            let metadata = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
            let primary = metadata.resolvedQuantization(for: "fc")
            #expect(primary?.groupSize == 64)
            #expect(primary?.bits == 4)
            #expect(primary?.mode == .affine)
        }
    }

    @Test("absent and empty standalone quantization mean unquantized")
    func standaloneQuantizationMayBeAbsentOrEmpty() throws {
        let configurations = [
            try qwenStandaloneMTPConfig(quantization: nil),
            try qwenStandaloneMTPConfig(quantization: [String: Any]()),
        ]
        for config in configurations {
            try withStandaloneMTPDirectory(
                config: config, weights: ["placeholder": MLXArray([Float(0)])]
            ) { directory in
                let metadata = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
                #expect(metadata.quantization == nil)
            }
        }
    }

    @Test("standalone BF16 tensors load without quantization metadata")
    func standaloneBF16LoadsWithoutQuantization() throws {
        let configData = try qwenSmallStandaloneMTPConfig(quantization: nil)
        let configuration = try qwenTextConfiguration(from: configData)
        let donor = Qwen35MTPModule(configuration)
        let weights = Dictionary(
            uniqueKeysWithValues: donor.parameters().flattened().map {
                ($0.0, $0.1.asType(.bfloat16))
            })

        try withStandaloneMTPDirectory(config: configData, weights: weights) { directory in
            let target = Qwen35TextModel(configuration)
            _ = try Qwen35InlineMTPAssistant.load(from: directory, target: target)
        }
    }

    @Test("scaled standalone tensors require quantization metadata")
    func scaledStandaloneWithoutQuantizationIsRejected() throws {
        let configurations = [
            try qwenSmallStandaloneMTPConfig(quantization: nil),
            try qwenSmallStandaloneMTPConfig(quantization: [String: Any]()),
        ]
        let weights = [
            "fc.weight": MLXArray([UInt32(0)]),
            "fc.scales": MLXArray([Float(1)]),
            "fc.biases": MLXArray([Float(0)]),
        ]
        for configData in configurations {
            let configuration = try qwenTextConfiguration(from: configData)
            try withStandaloneMTPDirectory(config: configData, weights: weights) { directory in
                let target = Qwen35TextModel(configuration)
                do {
                    _ = try Qwen35InlineMTPAssistant.load(from: directory, target: target)
                    Issue.record("scaled standalone artifact loaded without quantization metadata")
                } catch let error as Qwen35InlineMTPError {
                    #expect(error == .missingQuantization("fc"))
                }
            }
        }
    }

    @Test("scaled tensors on explicitly skipped paths are rejected")
    func scaledStandaloneOnSkippedPathIsRejected() throws {
        let configData = try qwenSmallStandaloneMTPConfig(quantization: [
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "fc": false,
        ])
        let configuration = try qwenTextConfiguration(from: configData)
        let weights = [
            "fc.weight": MLXArray([UInt32(0)]),
            "fc.scales": MLXArray([Float(1)]),
            "fc.biases": MLXArray([Float(0)]),
        ]
        try withStandaloneMTPDirectory(config: configData, weights: weights) { directory in
            let target = Qwen35TextModel(configuration)
            do {
                _ = try Qwen35InlineMTPAssistant.load(from: directory, target: target)
                Issue.record("scaled standalone artifact loaded on an explicitly skipped path")
            } catch let error as Qwen35InlineMTPError {
                #expect(error == .missingQuantization("fc"))
            }
        }
    }

    @Test("the published 31-key quantized path layout maps unchanged")
    func standaloneQuantizedPathLayoutMapsUnchanged() throws {
        let paths: Set<String> = [
            "pre_fc_norm_hidden.weight",
            "pre_fc_norm_embedding.weight",
            "fc.weight", "fc.scales", "fc.biases",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.q_proj.scales",
            "layers.0.self_attn.q_proj.biases",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.k_proj.scales",
            "layers.0.self_attn.k_proj.biases",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.v_proj.scales",
            "layers.0.self_attn.v_proj.biases",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.o_proj.scales",
            "layers.0.self_attn.o_proj.biases",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.gate_proj.scales",
            "layers.0.mlp.gate_proj.biases",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.up_proj.scales",
            "layers.0.mlp.up_proj.biases",
            "layers.0.mlp.down_proj.weight",
            "layers.0.mlp.down_proj.scales",
            "layers.0.mlp.down_proj.biases",
            "norm.weight",
        ]
        #expect(paths.count == 31)
        let sortedPaths = paths.sorted()
        let weights = Dictionary(
            uniqueKeysWithValues: sortedPaths.enumerated().map {
                ($0.element, MLXArray([Float($0.offset)]))
            })
        try withStandaloneMTPDirectory(
            config: qwenStandaloneMTPConfig(), weights: weights
        ) { directory in
            let loaded = try Qwen35InlineMTPAssistant.loadStandaloneWeights(from: directory)
            #expect(Set(loaded.keys) == paths)
            for (offset, path) in sortedPaths.enumerated() {
                #expect(loaded[path]?.item(Float.self) == Float(offset))
            }
        }
    }

    @Test("assistant state accounting includes KV, target hidden, and token storage")
    func assistantStateAccounting() throws {
        let root = try #require(
            try JSONSerialization.jsonObject(with: qwenInlineMTPConfig()) as? [String: Any])
        let textData = try JSONSerialization.data(withJSONObject: root["text_config"]!)
        let configuration = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: textData)

        #expect(Qwen35InlineMTPAssistant.cacheAllocationStep == 256)
        #expect(
            Qwen35InlineMTPAssistant.cacheBytesPerToken(
                configuration: configuration, layerCount: 1, elementBytes: 2) == 32)
        #expect(
            Qwen35InlineMTPAssistant.cacheBytesPerToken(
                configuration: configuration, layerCount: 1, elementBytes: 4) == 64)
        #expect(
            Qwen35InlineMTPAssistant.stateBytesPerToken(
                configuration: configuration, layerCount: 1,
                cacheElementBytes: 2, hiddenElementBytes: 2) == 52)
        #expect(
            Qwen35InlineMTPAssistant.stateBytesPerToken(
                configuration: configuration, layerCount: 1,
                cacheElementBytes: 4, hiddenElementBytes: 4) == 100)
        #expect(
            Qwen35InlineMTPAssistant.stateBytesPerToken(
                configuration: configuration, layerCount: Int.max,
                cacheElementBytes: Int.max, hiddenElementBytes: Int.max) == Int.max)
    }

    @Test("custom assistant prefixes fail before target loading")
    func customPrefixRejected() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-inline-mtp-prefix-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try qwenInlineMTPConfig(prefix: "assistant.")
            .write(to: directory.appendingPathComponent("config.json"))

        #expect(throws: Qwen35InlineMTPError.self) {
            _ = try Qwen35InlineMTPAssistant.loadMetadata(from: directory)
        }
    }
}
