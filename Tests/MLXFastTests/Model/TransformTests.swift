import Foundation
import Testing
@testable import MLXFastCore
@testable import MLXFastHarness
@testable import MLXFastTransform

@Test
func transformSelectsTextTowerTensorsAndDropsVisionTensors() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)

    try gemmaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try #"{"tokenizer":"fixture"}"#.write(
        to: reference.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.layers.0.self_attn.q_proj.weight"
    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let embedVisionName = "embed_vision.embedding_projection.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [4], data: Data([1, 2, 3, 4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [3], data: Data([9, 8, 7])),
            TensorFixture(name: embedVisionName, dtype: "U8", shape: [2], data: Data([5, 6])),
        ]
    )
    try """
    {
      "metadata": {"total_size": 9},
      "weight_map": {
        "\(textName)": "\(shardName)",
        "\(visionName)": "\(shardName)",
        "\(embedVisionName)": "\(shardName)"
      }
    }
    """.write(
        to: reference.appendingPathComponent("model.safetensors.index.json"),
        atomically: true,
        encoding: .utf8
    )

    let report = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    #expect(report.denseTensorCount == 1)
    #expect(report.denseShardCount == 1)
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("config.json").path))
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("tokenizer.json").path))
    #expect(!FileManager.default.fileExists(atPath: output.appendingPathComponent("experts").path))

    let outputShard = output.appendingPathComponent(shardName)
    let outputHeader = try Safetensors.readHeader(outputShard)
    #expect(outputHeader.tensors.keys.sorted() == [textName])
    #expect(try tensorBytes(outputShard, header: outputHeader, name: textName) == Data([1, 2, 3, 4]))

    let strippedIndexData = try Data(
        contentsOf: output.appendingPathComponent("model.safetensors.index.json")
    )
    let strippedIndex = try JSONSerialization.jsonObject(with: strippedIndexData) as? [String: Any]
    let weightMap = try #require(strippedIndex?["weight_map"] as? [String: String])
    #expect(weightMap == [textName: shardName])
    let metadata = try #require(strippedIndex?["metadata"] as? [String: Any])
    #expect(metadata["total_size"] as? Int == 4)
}

@Test
func transformWritesFlattenedRuntimeConfigFromTextConfig() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try gemmaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.embed_tokens.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [TensorFixture(name: textName, dtype: "U8", shape: [1], data: Data([1]))]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [textName: shardName]
    )

    _ = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    let configData = try Data(contentsOf: output.appendingPathComponent("config.json"))
    let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
    #expect(config?["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
    #expect(config?["vocab_size"] as? Int == MLXFastConstants.vocabSize)
    #expect(config?["text_config"] == nil)
    let quantization = try #require(config?["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == 64)
    #expect(quantization["bits"] as? Int == 4)
}

@Test
func transformDetectsModelFamilyFromSourceConfig() throws {
    let bonsai = try #require(
        try JSONSerialization.jsonObject(
            with: Data(bonsai2ReferenceConfigJSON().utf8)
        ) as? [String: Any]
    )
    #expect(
        try SwiftTransform.detectModelFamily(sourceConfigRoot: bonsai)
            == .prismHadamardQwen35
    )
    // The pack DOES carry a `text_config`, so the packed test must run before
    // the text_config-means-legacy-Gemma fallthrough or the pack would be
    // routed to the wrong family.
    #expect(bonsai["text_config"] is [String: Any])

    let gemma = try #require(
        try JSONSerialization.jsonObject(
            with: Data(gemmaReferenceConfigJSON().utf8)
        ) as? [String: Any]
    )
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: gemma) == .gemma4)

    // A point revision of the same family still routes to the packed path
    // rather than being refused: the prefix is matched, not the exact string.
    #expect(
        try SwiftTransform.detectModelFamily(
            sourceConfigRoot: ["model_type": "prism_hadamard_qwen35_v2"]
        ) == .prismHadamardQwen35
    )
    // And a text_config that is NOT this family keeps the legacy Gemma route.
    #expect(
        try SwiftTransform.detectModelFamily(
            sourceConfigRoot: ["text_config": ["model_type": "gemma4_text"]]
        ) == .gemma4
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.detectModelFamily(sourceConfigRoot: ["model_type": "prism"])
    }
}

@Test
func transformKeySelectionKeepsTheLanguageModelNamespaceOnly() {
    for kept in [
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.linear_attn.in_proj_qkv.signs",
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    ] {
        #expect(
            SwiftTransform.isSelectedTextTowerKey(kept, family: .prismHadamardQwen35),
            Comment(rawValue: kept)
        )
    }
    // The pack PUBLISHES a vision tower. This track serves text only, so the
    // tower is dropped here the same way the model's own weight filter drops
    // it at load.
    for dropped in [
        "vision_tower.blocks.0.attn.qkv.weight",
        "embed_vision.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.fc.weight",
    ] {
        #expect(
            !SwiftTransform.isSelectedTextTowerKey(dropped, family: .prismHadamardQwen35),
            Comment(rawValue: dropped)
        )
    }
    // The legacy Gemma family keeps its own prefix.
    #expect(
        SwiftTransform.isSelectedTextTowerKey(
            "language_model.model.layers.3.self_attn.q_proj.weight",
            family: .gemma4
        )
    )
}

/// The emitted runtime config is the source config PASSED THROUGH. It is the
/// artifact contract the packed loader reads -- `modules`, `components`,
/// `hadamard_config`, `gdn_activation_layout`, `tensor_namespace` -- so a
/// dropped field refuses the load; the only edit is the duplicate
/// quantization block, which this pack does not publish.
@Test
func transformBuildsBonsaiRuntimeConfigByPassingTheSourceThrough() throws {
    let configData = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigPath: bonsai2ConfigFixtureURL
    )
    let config = try #require(
        try JSONSerialization.jsonObject(with: configData) as? [String: Any]
    )
    let source = try bonsai2ConfigObject()

    #expect(config["model_type"] as? String == "prism_hadamard_qwen35")
    #expect(config["base_model_type"] as? String == "qwen3_5")
    #expect(config["hadamard_config"] as? String == "hadamard.json")
    #expect(config["gdn_activation_layout"] as? String == "grouped")
    #expect(config["tie_word_embeddings"] as? Bool == false)
    let text = try #require(config["text_config"] as? [String: Any])
    #expect(text["vocab_size"] as? Int == MLXFastConstants.vocabSize)
    #expect(text["hidden_size"] as? Int == MLXFastConstants.hiddenSize)
    #expect(text["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
    #expect(text["mtp_num_hidden_layers"] as? Int == 0)
    // Every source key survives except the duplicate quantization block, which
    // this pack does not publish.
    #expect(
        Set(config.keys)
            == Set(source.keys).subtracting(["quantization_config"])
    )
    let modules = try #require(config["modules"] as? [[String: Any]])
    #expect(modules.count == 402)

    let quantization = try #require(config["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == 128)
    #expect(quantization["bits"] as? Int == 2)
    #expect(quantization["mode"] as? String == "affine")
    #expect(Set(quantization.keys) == ["group_size", "bits", "mode"])
}

/// The untied head is the FIRST structural requirement the validator states,
/// so a pack that ships no `language_model.lm_head.*` is refused before the
/// copy.
@Test
func transformRejectsABonsaiPackWithoutTheUntiedHead() throws {
    let fixture = try writeBonsaiCheckpointFixture(
        tensors: [
            TensorFixture(
                name: "language_model.model.norm.weight",
                dtype: "F32",
                shape: [1],
                data: Data([0, 0, 0, 0])
            )
        ]
    )
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let rejection = transformRejection(fixture)
    #expect(rejection?.description.contains("untied output head") == true)
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

/// THE PACK CARRIES NO HEAD. This track's head is a separate pinned export, so
/// a pack that ships an embedded one is a different artifact and is refused
/// rather than absorbed.
@Test
func transformRejectsABonsaiPackThatCarriesAnEmbeddedMTPHead() throws {
    let tensors = packedHadamardTensorQuad(
        stem: "language_model.lm_head", outFeatures: 2, inFeatures: 1024)
        + [
            TensorFixture(
                name: "language_model.mtp.layers.0.fc.weight",
                dtype: "F32",
                shape: [1],
                data: Data([0, 0, 0, 0])
            )
        ]
    let fixture = try writeBonsaiCheckpointFixture(tensors: tensors)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let rejection = transformRejection(fixture)
    #expect(
        rejection?.description.contains(
            "declares no embedded multi-token-prediction head") == true
    )
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

/// Past the structural gates, the set must BE the pinned inventory.
@Test
func transformRejectsPartialBonsaiInventoryBeforePublishing() throws {
    let tensors = packedHadamardTensorQuad(
        stem: "language_model.lm_head", outFeatures: 2, inFeatures: 1024)
    let fixture = try writeBonsaiCheckpointFixture(tensors: tensors)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let rejection = transformRejection(fixture)
    #expect(rejection?.description.contains("exact public 2057-tensor contract") == true)
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

/// A packed module needs its scales, its biases AND its signs. The signs are
/// what make the folded weights readable, so a module that ships only two of
/// the three is refused by name, before the copy.
@Test
func transformRejectsAPackedModuleMissingItsSignVector() throws {
    let stem = "language_model.model.layers.1.mlp.gate_proj"
    var tensors = packedHadamardTensorQuad(
        stem: stem, outFeatures: 4, inFeatures: 1024)
    tensors.removeAll { $0.name == "\(stem).signs" }
    tensors += packedHadamardTensorQuad(
        stem: "language_model.lm_head", outFeatures: 2, inFeatures: 1024)
    let fixture = try writeBonsaiCheckpointFixture(tensors: tensors)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let rejection = transformRejection(fixture)
    #expect(
        rejection?.description.contains("must ship .scales, .biases and .signs") == true
    )
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

/// THE SIGNS COVER THE WHOLE CONTRACTED AXIS. A vector the width of one
/// Hadamard block would transform the first block and leave the rest
/// unrotated, which reads as a subtly wrong model rather than a failure.
@Test
func transformRejectsASignVectorNarrowerThanTheContractedAxis() throws {
    let stem = "language_model.model.layers.1.mlp.gate_proj"
    var tensors = packedHadamardTensorQuad(
        stem: stem, outFeatures: 4, inFeatures: 2048)
    tensors.removeAll { $0.name == "\(stem).signs" }
    tensors.append(
        TensorFixture(
            name: "\(stem).signs",
            dtype: "F32",
            shape: [1024],
            data: Data(count: 1024 * 4)
        )
    )
    tensors += packedHadamardTensorQuad(
        stem: "language_model.lm_head", outFeatures: 2, inFeatures: 1024)
    let fixture = try writeBonsaiCheckpointFixture(tensors: tensors)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let rejection = transformRejection(fixture)
    #expect(
        rejection?.description.contains(
            "covers the whole contracted axis") == true
    )
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

/// This pack is UNIFORM, so a per-tensor override in the quantization block is
/// a width the runtime never reads. Refuse rather than carry it.
@Test
func transformRejectsBonsaiQuantizationOverrides() throws {
    var config = try bonsai2ConfigObject()
    var quantization = try #require(config["quantization"] as? [String: Any])
    quantization["language_model.model.layers.1.mlp.gate_proj"] = [
        "group_size": 128,
        "bits": 4,
    ]
    config["quantization"] = quantization
    let configData = try JSONSerialization.data(
        withJSONObject: config,
        options: [.sortedKeys]
    )
    let fixture = try writeBonsaiCheckpointFixture(
        tensors: [
            TensorFixture(
                name: "language_model.model.norm.weight",
                dtype: "F32",
                shape: [1],
                data: Data([0, 0, 0, 0])
            )
        ],
        configJSON: String(decoding: configData, as: UTF8.self)
    )
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let rejection = transformRejection(fixture)
    #expect(rejection?.description.contains("this pack is uniform") == true)
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
}

@Test
func transformRejectsNonMLXCompressedAndGlobalScaleSchemas() throws {
    // `.biases` is NOT on this list: affine quantization legitimately ships
    // one beside every packed weight, which is exactly the difference from an
    // NVFP4 contract.
    for forbiddenName in [
        "language_model.model.layers.1.mlp.gate_proj.weight_packed",
        "language_model.model.layers.1.mlp.gate_proj.input_global_scale",
        "language_model.model.layers.1.mlp.gate_proj.weight_global_scale",
        "language_model.model.layers.3.self_attn.k_scale",
        "language_model.model.layers.3.self_attn.v_scale",
    ] {
        let fixture = try writeBonsaiCheckpointFixture(
            tensors: [
                TensorFixture(
                    name: forbiddenName,
                    dtype: "U8",
                    shape: [1],
                    data: Data([0])
                )
            ]
        )
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let rejection = transformRejection(fixture)
        #expect(
            rejection?.description.contains("rejects compressed-tensors") == true,
            Comment(rawValue: forbiddenName)
        )
        #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
    }
}

/// The pack publishes its affine spec ONCE. A duplicate is accepted and must
/// agree: emitting one of two conflicting specs would pick a quantization the
/// shards were not written with.
@Test
func transformRequiresTheDuplicateBonsaiQuantizationBlocksToAgree() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let configPath = root.appendingPathComponent("config.json")

    func emit(_ config: [String: Any]) throws -> Data {
        let data = try JSONSerialization.data(
            withJSONObject: config,
            options: [.sortedKeys]
        )
        try data.write(to: configPath)
        return try SwiftTransform.makeRuntimeConfigData(sourceConfigPath: configPath)
    }

    // One block alone -- what the pack publishes -- is enough.
    var config = try bonsai2ConfigObject()
    #expect(config["quantization_config"] == nil)
    #expect(throws: Never.self) { _ = try emit(config) }

    // An agreeing duplicate is accepted.
    config = try bonsai2ConfigObject()
    config["quantization_config"] = try #require(config["quantization"] as? [String: Any])
    #expect(throws: Never.self) { _ = try emit(config) }

    // Two blocks that disagree are refused.
    config = try bonsai2ConfigObject()
    var quantizationConfig = try #require(config["quantization"] as? [String: Any])
    quantizationConfig["bits"] = 4
    config["quantization_config"] = quantizationConfig
    #expect(throws: MLXFastError.self) { _ = try emit(config) }

    // A block missing a required scalar is refused rather than defaulted.
    config = try bonsai2ConfigObject()
    var quantization = try #require(config["quantization"] as? [String: Any])
    quantization.removeValue(forKey: "mode")
    config["quantization"] = quantization
    #expect(throws: MLXFastError.self) { _ = try emit(config) }

    // And a width this pack does not use is refused.
    config = try bonsai2ConfigObject()
    quantization = try #require(config["quantization"] as? [String: Any])
    quantization["group_size"] = 64
    config["quantization"] = quantization
    #expect(throws: MLXFastError.self) { _ = try emit(config) }
}

@Test
func transformRuntimeConfigCaptureDoesNotRereadChangedSource() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let configPath = root.appendingPathComponent("config.json")
    try bonsai2ReferenceConfigJSON().write(
        to: configPath,
        atomically: true,
        encoding: .utf8
    )

    let captured = try SwiftTransform.makeRuntimeConfigData(sourceConfigPath: configPath)
    try #"{"text_config":{"num_hidden_layers":1}}"#.write(
        to: configPath,
        atomically: true,
        encoding: .utf8
    )

    // The Bonsai runtime config is the source passed through, so the layer
    // count stays inside `text_config`.
    let object = try JSONSerialization.jsonObject(with: captured) as? [String: Any]
    let text = try #require(object?["text_config"] as? [String: Any])
    #expect(text["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
}

@Test
func transformRejectsSourceMetadataMutationsBeforePublishingOutput() throws {
    enum Mutation: CaseIterable {
        case config
        case index
        case tokenizer
    }

    for mutation in Mutation.allCases {
        let expectedError: String
        switch mutation {
        case .config:
            expectedError = "reference config changed while transform was running"
        case .index:
            expectedError = "checkpoint index changed while transform was running"
        case .tokenizer:
            expectedError = "reference tokenizer metadata changed while transform was running"
        }
        let fixture = try writeTransformFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try FileManager.default.createDirectory(
            at: fixture.output,
            withIntermediateDirectories: true
        )
        let sentinel = fixture.output.appendingPathComponent("sentinel.txt")
        try "preserve \(mutation)".write(to: sentinel, atomically: true, encoding: .utf8)

        var rejection: MLXFastError?
        do {
            _ = try SwiftTransform.run(
                TransformOptions(
                    referencePath: fixture.reference.path,
                    outputPath: fixture.output.path
                ),
                beforeSourceRevalidation: {
                    switch mutation {
                    case .config:
                        try #"{"text_config":{"num_hidden_layers":1}}"#.write(
                            to: fixture.reference.appendingPathComponent("config.json"),
                            atomically: true,
                            encoding: .utf8
                        )
                    case .index:
                        try writeCheckpointIndex(
                            fixture.reference.appendingPathComponent(
                                "model.safetensors.index.json"
                            ),
                            weightMap: [
                                "backbone.layers.0.mixer.q_proj.weight":
                                    "model-00001-of-00001.safetensors",
                            ],
                            metadata: ["generation": 2]
                        )
                    case .tokenizer:
                        try #"{"tokenizer":"changed"}"#.write(
                            to: fixture.reference.appendingPathComponent("tokenizer.json"),
                            atomically: true,
                            encoding: .utf8
                        )
                    }
                }
            )
        } catch let error as MLXFastError {
            rejection = error
        } catch {
            throw error
        }
        #expect(rejection?.description == expectedError, "mutation: \(mutation)")

        #expect(
            try String(contentsOf: sentinel, encoding: .utf8) == "preserve \(mutation)",
            "mutation: \(mutation)"
        )
        #expect(
            try transformStagingDirectories(nextTo: fixture.output).isEmpty,
            "mutation: \(mutation)"
        )
    }
}

@Test
func transformRejectsSymlinkedTokenizerMetadataAndPreservesOutput() throws {
    let fixture = try writeTransformFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let tokenizer = fixture.reference.appendingPathComponent("tokenizer.json")
    let tokenizerTarget = fixture.root.appendingPathComponent("tokenizer-target.json")
    try #"{"tokenizer":"external"}"#.write(
        to: tokenizerTarget,
        atomically: true,
        encoding: .utf8
    )
    try FileManager.default.removeItem(at: tokenizer)
    try FileManager.default.createSymbolicLink(at: tokenizer, withDestinationURL: tokenizerTarget)

    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let sentinel = fixture.output.appendingPathComponent("sentinel.txt")
    try "preserve".write(to: sentinel, atomically: true, encoding: .utf8)

    var rejection: MLXFastError?
    do {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
        )
    } catch let error as MLXFastError {
        rejection = error
    } catch {
        throw error
    }

    #expect(rejection?.description.contains("reference metadata is not a regular file") == true)
    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "preserve")
    #expect(try transformStagingDirectories(nextTo: fixture.output).isEmpty)
}

@Test
func transformRejectsReferenceAsOutputWithoutMutatingCheckpoint() throws {
    let fixture = try writeTransformFixture()
    let originalConfig = try Data(
        contentsOf: fixture.reference.appendingPathComponent("config.json")
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: fixture.reference.path
            )
        )
    }

    #expect(
        try Data(contentsOf: fixture.reference.appendingPathComponent("config.json"))
            == originalConfig
    )
    #expect(
        FileManager.default.fileExists(
            atPath: fixture.reference.appendingPathComponent("model-00001-of-00001.safetensors").path
        )
    )
}

@Test
func transformRejectsSymlinkAliasOfReferenceAsOutput() throws {
    let fixture = try writeTransformFixture()
    let outputAlias = fixture.root.appendingPathComponent("reference-alias", isDirectory: true)
    try FileManager.default.createSymbolicLink(
        at: outputAlias,
        withDestinationURL: fixture.reference
    )
    let originalConfig = try Data(
        contentsOf: fixture.reference.appendingPathComponent("config.json")
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: outputAlias.path
            )
        )
    }

    #expect(
        try Data(contentsOf: fixture.reference.appendingPathComponent("config.json"))
            == originalConfig
    )
}

@Test
func transformRejectsOutputThatContainsWorkingDirectory() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference")
    let workingDirectory = root.appendingPathComponent("workspace/project")
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)

    #expect(throws: MLXFastError.self) {
        try SwiftTransform.validateDistinctDirectories(
            referenceDirectory: reference,
            outputDirectory: root,
            workingDirectory: workingDirectory
        )
    }
}

@Test
func transformRejectsOutputInsideReferenceDirectory() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference")
    let output = reference.appendingPathComponent("weights")
    let workingDirectory = root.appendingPathComponent("workspace")
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)

    #expect(throws: MLXFastError.self) {
        try SwiftTransform.validateDistinctDirectories(
            referenceDirectory: reference,
            outputDirectory: output,
            workingDirectory: workingDirectory
        )
    }
}

@Test
func transformRejectsExistingRegularFileOutputWithoutReplacingIt() throws {
    let fixture = try writeTransformFixture()
    let outputFile = fixture.root.appendingPathComponent("existing-output")
    try "preserve me".write(to: outputFile, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: outputFile.path
            )
        )
    }

    #expect(try String(contentsOf: outputFile, encoding: .utf8) == "preserve me")
}

@Test
func transformAtomicallyReplacesExistingOutputAndRemovesStaleFiles() throws {
    let fixture = try writeTransformFixture()
    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let staleFile = fixture.output.appendingPathComponent("stale.txt")
    try "stale".write(to: staleFile, atomically: true, encoding: .utf8)

    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )

    #expect(!FileManager.default.fileExists(atPath: staleFile.path))
    #expect(
        FileManager.default.fileExists(
            atPath: fixture.output.appendingPathComponent("model.safetensors.index.json").path
        )
    )
}

@Test
func transformFailurePreservesExistingOutputAndRemovesStagingDirectory() throws {
    let fixture = try writeTransformFixture()
    try #"{"vision_config":{}}"#.write(
        to: fixture.reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try FileManager.default.createDirectory(at: fixture.output, withIntermediateDirectories: true)
    let sentinel = fixture.output.appendingPathComponent("sentinel.txt")
    try "original".write(to: sentinel, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
        )
    }

    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "original")
    let siblings = try FileManager.default.contentsOfDirectory(
        at: fixture.output.deletingLastPathComponent(),
        includingPropertiesForKeys: nil
    )
    #expect(
        !siblings.contains {
            $0.lastPathComponent.hasPrefix(".weights.mlxfast-transform-")
        }
    )
}

@Test
func checkpointIndexBuildRejectsDuplicateTensorNamesAcrossShards() throws {
    let root = try temporaryDirectory()
    let tensorName = "language_model.model.embed_tokens.weight"
    try writeSafetensors(
        root.appendingPathComponent("model-00001-of-00002.safetensors"),
        tensors: [TensorFixture(name: tensorName, dtype: "U8", shape: [1], data: Data([1]))]
    )
    try writeSafetensors(
        root.appendingPathComponent("model-00002-of-00002.safetensors"),
        tensors: [TensorFixture(name: tensorName, dtype: "U8", shape: [1], data: Data([2]))]
    )

    #expect(throws: MLXFastError.self) {
        _ = try CheckpointIndex.buildFromSafetensors(in: root)
    }
}

@Test
func transformRejectsCheckpointWithoutTextTowerTensorsBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try bonsai2ReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let foreignName = "language_model.model.layers.0.self_attn.q_proj.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [TensorFixture(name: foreignName, dtype: "U8", shape: [2], data: Data([1, 2]))]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [foreignName: shardName]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformVerifierAcceptsFreshSubmittedTransformOutputAndIgnoresLocalCacheMarkers() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "cache\n".write(
        to: fixture.output.appendingPathComponent(".benchmark-source.sha256"),
        atomically: true,
        encoding: .utf8
    )
    FileManager.default.createFile(
        atPath: fixture.output.appendingPathComponent(".gitkeep").path,
        contents: Data()
    )

    let report = try TransformVerifier.verify(
        TransformVerificationOptions(
            referencePath: fixture.reference.path,
            weightsPath: fixture.output.path,
            temporaryParentPath: fixture.root.path
        )
    )

    #expect(report.referencePath == fixture.reference.path)
    #expect(report.weightsPath == fixture.output.path)
    #expect(report.fileCount > 0)
    #expect(report.byteCount > 0)
    #expect(report.maxByteCount == MLXFastConstants.defaultMaxTransformedWeightsBytes)
    #expect(report.sha256.count == 64)
    #expect(report.deterministic)
}

@Test
func transformVerifierRejectsOutputThatDiffersFromFreshTransformRun() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "changed".write(
        to: fixture.output.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path
            )
        )
    }
}

@Test
func transformVerifierRejectsStaleExtraGeneratedFile() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )
    try "extra".write(
        to: fixture.output.appendingPathComponent("extra.txt"),
        atomically: true,
        encoding: .utf8
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path
            )
        )
    }
}

@Test
func transformVerifierRejectsOutputAboveConfiguredByteLimit() throws {
    let fixture = try writeTransformFixture()
    _ = try SwiftTransform.run(
        TransformOptions(referencePath: fixture.reference.path, outputPath: fixture.output.path)
    )

    #expect(throws: MLXFastError.self) {
        _ = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: fixture.reference.path,
                weightsPath: fixture.output.path,
                temporaryParentPath: fixture.root.path,
                maxByteCount: 1
            )
        )
    }
}

@Test
func transformRejectsUnsupportedIndexShardBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try bonsai2ReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "backbone.layers.0.mixer.q_proj.weight": "pytorch_model.bin",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformRejectsUnsafeIndexShardBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try bonsai2ReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "backbone.layers.0.mixer.q_proj.weight": "../model-00001.safetensors",
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformRejectsIndexTensorMissingFromShardHeaderBeforeCreatingOutput() throws {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try bonsai2ReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: "backbone.layers.0.mixer.k_proj.weight", dtype: "U8", shape: [2], data: Data([1, 2])),
        ]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            "backbone.layers.0.mixer.q_proj.weight": shardName,
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.run(
            TransformOptions(referencePath: reference.path, outputPath: output.path)
        )
    }
    #expect(!FileManager.default.fileExists(atPath: output.path))
}

@Test
func transformAcceptsSparseShardLargerThanInt32() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try gemmaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.layers.0.self_attn.q_proj.weight"
    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let shardName = "model-00001-of-00001.safetensors"
    let shard = reference.appendingPathComponent(shardName)
    try writeSafetensors(
        shard,
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [1], data: Data([4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [1], data: Data([8])),
        ]
    )
    try truncateFile(shard, toByteCount: Int64(Int32.max) + 1024)
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            textName: shardName,
            visionName: shardName,
        ]
    )

    let report = try SwiftTransform.run(
        TransformOptions(referencePath: reference.path, outputPath: output.path)
    )

    #expect(report.denseTensorCount == 1)
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private struct TransformFixturePaths {
    let root: URL
    let reference: URL
    let output: URL
}

private func writeTransformFixture() throws -> TransformFixturePaths {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try gemmaReferenceConfigJSON().write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try #"{"tokenizer":"fixture"}"#.write(
        to: reference.appendingPathComponent("tokenizer.json"),
        atomically: true,
        encoding: .utf8
    )

    let textName = "language_model.model.layers.0.self_attn.q_proj.weight"
    let visionName = "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(
        reference.appendingPathComponent(shardName),
        tensors: [
            TensorFixture(name: textName, dtype: "U8", shape: [4], data: Data([1, 2, 3, 4])),
            TensorFixture(name: visionName, dtype: "U8", shape: [3], data: Data([9, 8, 7])),
        ]
    )
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: [
            textName: shardName,
            visionName: shardName,
        ]
    )

    return TransformFixturePaths(root: root, reference: reference, output: output)
}

/// Legacy Gemma 4 multimodal source-config layout (nested `text_config`),
/// kept to cover the transform's `.gemma4` family path.
private func gemmaReferenceConfigJSON() -> String {
    """
    {
      "text_config": {
        "num_hidden_layers": \(MLXFastConstants.numHiddenLayers),
        "vocab_size": \(MLXFastConstants.vocabSize),
        "hidden_size": \(MLXFastConstants.hiddenSize)
      },
      "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
    }
    """
}

/// The pinned Ternary Bonsai 2 source config, loaded from the immutable
/// artifact contract fixture rather than reconstructed synthetically here.
private func bonsai2ReferenceConfigJSON() throws -> String {
    let data = try JSONSerialization.data(
        withJSONObject: try bonsai2ConfigObject(),
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )
    return String(decoding: data, as: UTF8.self)
}

/// One packed Hadamard module at the pinned widths: U32 packed codes, the FP16
/// scale and bias companions, and the FP32 sign vector. `inFeatures` must be a
/// multiple of the 1024-wide Hadamard block and of the 128 group size, so
/// `in / 16` U32 columns pair with `in / 128` group columns and `in` signs.
private func packedHadamardTensorQuad(
    stem: String,
    outFeatures: Int,
    inFeatures: Int
) -> [TensorFixture] {
    let packedWidth = inFeatures * 2 / 32
    let groupWidth = inFeatures / 128
    let weightShape = [outFeatures, packedWidth]
    let companionShape = [outFeatures, groupWidth]
    return [
        TensorFixture(
            name: "\(stem).weight",
            dtype: "U32",
            shape: weightShape,
            data: Data(count: weightShape.reduce(1, *) * 4)
        ),
        TensorFixture(
            name: "\(stem).scales",
            dtype: "F16",
            shape: companionShape,
            data: Data(count: companionShape.reduce(1, *) * 2)
        ),
        TensorFixture(
            name: "\(stem).biases",
            dtype: "F16",
            shape: companionShape,
            data: Data(count: companionShape.reduce(1, *) * 2)
        ),
        TensorFixture(
            name: "\(stem).signs",
            dtype: "F32",
            shape: [inFeatures],
            data: Data(count: inFeatures * 4)
        ),
    ]
}

private struct BonsaiFixturePaths {
    let root: URL
    let reference: URL
    let output: URL
    let shardName: String
}

private func writeBonsaiCheckpointFixture(
    tensors: [TensorFixture],
    configJSON: String? = nil
) throws -> BonsaiFixturePaths {
    let root = try temporaryDirectory()
    let reference = root.appendingPathComponent("reference", isDirectory: true)
    let output = root.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    let effectiveConfigJSON = try configJSON ?? bonsai2ReferenceConfigJSON()
    try effectiveConfigJSON.write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    let shardName = "model-00001-of-00001.safetensors"
    try writeSafetensors(reference.appendingPathComponent(shardName), tensors: tensors)
    try writeCheckpointIndex(
        reference.appendingPathComponent("model.safetensors.index.json"),
        weightMap: Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, shardName) })
    )
    return BonsaiFixturePaths(
        root: root,
        reference: reference,
        output: output,
        shardName: shardName
    )
}

/// Run the transform on a fixture and return the refusal it must produce.
private func transformRejection(_ fixture: BonsaiFixturePaths) -> MLXFastError? {
    do {
        _ = try SwiftTransform.run(
            TransformOptions(
                referencePath: fixture.reference.path,
                outputPath: fixture.output.path
            )
        )
        return nil
    } catch let error as MLXFastError {
        return error
    } catch {
        return nil
    }
}

private func writeCheckpointIndex(
    _ path: URL,
    weightMap: [String: String],
    metadata: [String: Any]? = nil
) throws {
    var object: [String: Any] = ["weight_map": weightMap]
    object["metadata"] = metadata
    let data = try JSONSerialization.data(
        withJSONObject: object,
        options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )
    try data.write(to: path)
}

private func transformStagingDirectories(nextTo output: URL) throws -> [URL] {
    try FileManager.default.contentsOfDirectory(
        at: output.deletingLastPathComponent(),
        includingPropertiesForKeys: nil
    ).filter {
        $0.lastPathComponent.hasPrefix(".\(output.lastPathComponent).mlxfast-transform-")
    }
}

private func writeSafetensors(_ path: URL, tensors: [TensorFixture]) throws {
    var object: [String: Any] = [:]
    var cursor = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        object[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + tensor.data.count],
        ]
        cursor += tensor.data.count
    }

    var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }

    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        output.append(tensor.data)
    }
    try output.write(to: path)
}

private func truncateFile(_ path: URL, toByteCount byteCount: Int64) throws {
    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(byteCount))
}

private func tensorBytes(_ path: URL, header: SafetensorsHeader, name: String) throws -> Data {
    let info = try #require(header.tensors[name])
    let data = try Data(contentsOf: path)
    let start = Int(header.dataBaseOffset) + info.dataStart
    return data.subdata(in: start..<(start + info.byteCount))
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
