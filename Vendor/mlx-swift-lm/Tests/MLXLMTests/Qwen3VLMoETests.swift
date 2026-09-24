import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXVLM
@testable import MLXLMCommon


final class Qwen3VLMoETests: XCTestCase {
    func testPublishedMoEGeometryDecodes() throws {
        let config = try decodeConfiguration(realShapeConfigurationJSON)

        XCTAssertEqual(config.modelType, "qwen3_vl_moe")
        XCTAssertEqual(config.textConfiguration.modelType, "qwen3_vl_moe_text")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 2_048)
        XCTAssertEqual(config.textConfiguration.numHiddenLayers, 48)
        XCTAssertEqual(config.textConfiguration.numExperts, 128)
        XCTAssertEqual(config.textConfiguration.numExpertsPerTok, 8)
        XCTAssertEqual(config.textConfiguration.moeIntermediateSize, 768)
        XCTAssertEqual(config.textConfiguration.decoderSparseStep, 1)
        XCTAssertEqual(config.textConfiguration.mlpOnlyLayers, [])
        XCTAssertTrue(config.textConfiguration.normTopKProb)
    }

    func testDenseConfigurationRetainsDenseMLP() throws {
        let config = try decodeConfiguration(tinyDenseConfigurationJSON)
        let model = Qwen3VLLanguage.Model(config.textConfiguration)

        XCTAssertEqual(config.textConfiguration.numExperts, 0)
        XCTAssertTrue(model.layers[0].mlp is Qwen3VLLanguage.MLP)
        XCTAssertFalse(model.layers[0].mlp is Qwen3VLLanguage.SparseMoeBlock)
    }

    func testSparseStepAndMLPOnlyLayersInstallImmutableRoutes() throws {
        let config = try decodeConfiguration(tinyMoEConfigurationJSON)
        let model = Qwen3VLLanguage.Model(config.textConfiguration)

        XCTAssertFalse(model.layers[0].mlp is Qwen3VLLanguage.SparseMoeBlock)
        XCTAssertTrue(model.layers[1].mlp is Qwen3VLLanguage.SparseMoeBlock)
        XCTAssertFalse(model.layers[2].mlp is Qwen3VLLanguage.SparseMoeBlock)
        XCTAssertFalse(model.layers[3].mlp is Qwen3VLLanguage.SparseMoeBlock)
    }

    func testPublishedMoEVisionUsesReferenceGELUVariants() throws {
        let config = try decodeConfiguration(realShapeConfigurationJSON)
        let model = Qwen3VLVision.VisionModel(config.visionConfiguration)

        assertApproximation(model.merger.activation, is: .none)
        for merger in model.deepstackMergers {
            assertApproximation(merger.activation, is: .none)
        }
        for block in model.blocks {
            assertApproximation(block.mlp.activation, is: .tanh)
        }
    }

    func testDenseVisionRetainsExistingGELUVariants() throws {
        let config = try decodeConfiguration(tinyDenseConfigurationJSON)
        let model = Qwen3VLVision.VisionModel(config.visionConfiguration)

        assertApproximation(model.merger.activation, is: .none)
        assertApproximation(model.blocks[0].mlp.activation, is: .fast)
    }

    func testSparseParameterTreeUsesFusedGateUpProjection() throws {
        let config = try decodeConfiguration(tinyAllMoEConfigurationJSON)
        let model = Qwen3VLLanguage.Model(config.textConfiguration)
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("layers.0.mlp.gate.weight"))
        XCTAssertTrue(keys.contains("layers.0.mlp.switch_mlp.gate_up_proj.weight"))
        XCTAssertTrue(keys.contains("layers.0.mlp.switch_mlp.down_proj.weight"))
        XCTAssertFalse(keys.contains { $0.contains("shared_expert") })
        XCTAssertFalse(keys.contains("layers.0.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertFalse(keys.contains("layers.0.mlp.switch_mlp.up_proj.weight"))
    }


    func testSplitCheckpointMigratesToFusedKeysAndStrictLoads() throws {
        let model = Qwen3VL(try decodeConfiguration(tinyAllMoEConfigurationJSON))
        let checkpoint = splitExpertCheckpoint(from: model, rawLanguagePrefix: true)
        let rawBase = "model.language_model.layers.0.mlp.switch_mlp"
        let w4 = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let policy = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(rawBase).gate_proj": .quantize(w4),
                "\(rawBase).up_proj": .quantize(w4),
            ])
        model.checkpointPerLayerQuantization = policy
        let sanitized = model.sanitize(weights: checkpoint)
        let base = "language_model.model.layers.0.mlp.switch_mlp"

        XCTAssertNotNil(sanitized["\(base).gate_up_proj.weight"])
        XCTAssertNil(sanitized["\(base).gate_proj.weight"])
        XCTAssertNil(sanitized["\(base).up_proj.weight"])
        let block = model.languageModel.model.layers[0].mlp
            as? Qwen3VLLanguage.SparseMoeBlock
        XCTAssertTrue(block?.switchMLP.hasFusedGateUp == true)
        XCTAssertEqual(
            resolveQuantization(
                path: "\(base).gate_up_proj",
                perLayerQuantization: policy,
                aliasing: model),
            w4)
        XCTAssertNoThrow(
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.all]))
    }

    func testIncompatibleQuantizationStaysSplitStrictLoadsAndMatchesLogits() throws {
        let config = try decodeConfiguration(tinyAllMoEConfigurationJSON)
        let fusedModel = Qwen3VL(config)
        let model = Qwen3VL(config)
        let checkpoint = splitExpertCheckpoint(from: fusedModel, rawLanguagePrefix: true)
        let rawBase = "model.language_model.layers.0.mlp.switch_mlp"
        model.checkpointPerLayerQuantization = BaseConfiguration.PerLayerQuantization(
            quantization: nil,
            perLayerQuantization: [
                "\(rawBase).gate_proj": .quantize(.init(groupSize: 64, bits: 4)),
                "\(rawBase).up_proj": .quantize(.init(groupSize: 64, bits: 8)),
            ])

        let sanitized = model.sanitize(weights: checkpoint)
        let base = "language_model.model.layers.0.mlp.switch_mlp"
        XCTAssertNotNil(sanitized["\(base).gate_proj.weight"])
        XCTAssertNotNil(sanitized["\(base).up_proj.weight"])
        XCTAssertNil(sanitized["\(base).gate_up_proj.weight"])
        let block = model.languageModel.model.layers[0].mlp
            as? Qwen3VLLanguage.SparseMoeBlock
        XCTAssertTrue(block?.switchMLP.hasFusedGateUp == false)
        XCTAssertNoThrow(
            try model.update(
                parameters: ModuleParameters.unflattened(sanitized), verify: [.all]))
        let tokens = MLXArray([Int32(1), Int32(2)]).reshaped([1, 2])
        let fusedLogits = fusedModel(tokens, cache: nil)
        let splitLogits = model(tokens, cache: nil)
        eval(fusedLogits, splitLogits)
        XCTAssertEqual(splitLogits.shape, fusedLogits.shape)
        XCTAssertTrue(
            splitLogits.allClose(fusedLogits, rtol: 1e-5, atol: 1e-6).item(Bool.self))
    }

    func testMoEModelTypeResolvesThroughPublicRegistry() async throws {
        let data = Data(tinyAllMoEConfigurationJSON.utf8)
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "qwen3_vl_moe")

        XCTAssertTrue(model is Qwen3VL)
    }

    func testPublishedCheckpointHasRegistryPreset() {
        XCTAssertEqual(
            VLMRegistry.qwen3VL30BA3BInstruct4Bit.name,
            "lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit")
        XCTAssertTrue(
            VLMRegistry.all().contains(VLMRegistry.qwen3VL30BA3BInstruct4Bit))
    }

    func testSparseRoutingAndOutputMatchIndependentReference() throws {
        let config = try decodeConfiguration(tinyAllMoEConfigurationJSON)
        let block = Qwen3VLLanguage.SparseMoeBlock(config.textConfiguration)
        let dimensions = config.textConfiguration.hiddenSize
        let hiddenDimensions = config.textConfiguration.moeIntermediateSize
        let experts = config.textConfiguration.numExperts
        let topK = config.textConfiguration.numExpertsPerTok
        let rows = 2

        let gateWeights = deterministicValues(experts * dimensions, scale: 0.071, offset: 1)
        let expertGateWeights = deterministicValues(
            experts * hiddenDimensions * dimensions, scale: 0.037, offset: 3)
        let expertUpWeights = deterministicValues(
            experts * hiddenDimensions * dimensions, scale: 0.029, offset: 7)
        let expertDownWeights = deterministicValues(
            experts * dimensions * hiddenDimensions, scale: 0.043, offset: 11)
        let inputValues = deterministicValues(rows * dimensions, scale: 0.113, offset: 5)

        let expertGate = MLXArray(
            expertGateWeights, [experts, hiddenDimensions, dimensions])
        let expertUp = MLXArray(
            expertUpWeights, [experts, hiddenDimensions, dimensions])
        block.update(parameters: ModuleParameters.unflattened([
            "gate.weight": MLXArray(gateWeights, [experts, dimensions]),
            "switch_mlp.gate_up_proj.weight": concatenated(
                [expertGate, expertUp], axis: -2),
            "switch_mlp.down_proj.weight": MLXArray(
                expertDownWeights, [experts, dimensions, hiddenDimensions]),
        ]))

        let input = MLXArray(inputValues, [1, rows, dimensions])
        let (indices, scores) = block.route(input)
        let output = block(input)
        eval(indices, scores, output)

        let reference = independentSparseReference(
            input: inputValues, rows: rows, dimensions: dimensions,
            gateWeights: gateWeights, expertGateWeights: expertGateWeights,
            expertUpWeights: expertUpWeights, expertDownWeights: expertDownWeights,
            experts: experts, topK: topK, hiddenDimensions: hiddenDimensions)

        let actualIndices = indices.asType(.int32).asArray(Int32.self).map(Int.init)
        let actualScores = scores.asArray(Float.self)
        for row in 0 ..< rows {
            let range = row * topK ..< (row + 1) * topK
            XCTAssertEqual(
                Set(actualIndices[range]), Set(reference.indices[row]),
                "row \(row) selected the wrong experts")
            for position in range {
                let expert = actualIndices[position]
                let expectedScore = reference.scores[row][expert]!
                XCTAssertEqual(actualScores[position], expectedScore, accuracy: 2e-4)
            }
        }

        let actualOutput = output.asArray(Float.self)
        XCTAssertEqual(actualOutput.count, reference.output.count)
        for (actual, expected) in zip(actualOutput, reference.output) {
            XCTAssertEqual(actual, expected, accuracy: 2e-5)
        }

        let splitBlock = Qwen3VLLanguage.SparseMoeBlock(config.textConfiguration)
        setSwitchGLUGateUpFused(false, at: "switch_mlp", in: splitBlock)
        splitBlock.update(parameters: ModuleParameters.unflattened([
            "gate.weight": MLXArray(gateWeights, [experts, dimensions]),
            "switch_mlp.gate_proj.weight": expertGate,
            "switch_mlp.up_proj.weight": expertUp,
            "switch_mlp.down_proj.weight": MLXArray(
                expertDownWeights, [experts, dimensions, hiddenDimensions]),
        ]))
        let (splitIndices, splitScores) = splitBlock.route(input)
        let splitOutput = splitBlock(input)
        eval(splitIndices, splitScores, splitOutput)

        XCTAssertEqual(splitIndices.dtype, indices.dtype)
        XCTAssertEqual(splitScores.dtype, scores.dtype)
        XCTAssertEqual(
            splitIndices.asType(.int32).asArray(Int32.self),
            indices.asType(.int32).asArray(Int32.self))
        XCTAssertTrue(splitScores.allClose(scores, rtol: 0, atol: 0).item(Bool.self))
        XCTAssertTrue(splitOutput.allClose(output, rtol: 1e-5, atol: 1e-6).item(Bool.self))
    }

    func testCBv2LayerKindsAndCapabilitiesAreConservative() throws {
        let model = Qwen3VL(try decodeConfiguration(tinyDenseConfigurationJSON))

        XCTAssertEqual(
            model.cbv2LayerKinds,
            [
                CBv2LayerKind(
                    attention: .full, headDim: 8, kvHeads: 1, queryHeads: 1,
                    modelLayerIndex: 0)
            ])
        XCTAssertEqual(
            model.cbv2Capabilities,
            CBv2ModelCapabilities(
                supportsPrefixReuse: false,
                supportsPagedKV: false,
                supportsCompiledDecode: false,
                supportsPackedPrefill: false,
                supportsMTP: false))
        XCTAssertEqual(model.cbv2PositionAxisCount, 3)
        XCTAssertEqual(model.imagePlaceholderTokenId, model.config.imageTokenIndex)
        XCTAssertEqual(model.videoPlaceholderTokenId, model.config.videoTokenIndex)
    }

    func testPositionResultRejectsImageGridWithoutMatchingPlaceholderRun() throws {
        let model = Qwen3VL(try decodeConfiguration(tinyDenseConfigurationJSON))

        XCTAssertThrowsError(
            try model.positionResult(
                tokens: MLXArray([Int32(1), 2, 3]),
                imageGrids: [THW(1, 2, 2)])
        ) { error in
            XCTAssertEqual(
                error as? Qwen35PositionSeamError,
                .visualTokenRunMismatch(
                    kind: "image", gridIndex: 0, expected: 4, actual: 0))
        }
    }

    func testImageFeatureSeamRejectsWrongFlattenedPatchShape() throws {
        let model = Qwen3VL(try decodeConfiguration(tinyDenseConfigurationJSON))

        XCTAssertThrowsError(
            try model.cbv2VisionFeatures(
                imagePixels: MLXArray.zeros([4, 11]),
                imageGrids: [THW(1, 2, 2)]))
    }

    func testQuantizedImageFeaturesUseEmbeddingActivationDType() throws {
        let quantizableJSON = tinyDenseConfigurationJSON
            .replacingOccurrences(of: "\"hidden_size\": 8", with: "\"hidden_size\": 32")
            .replacingOccurrences(of: "\"out_hidden_size\": 8", with: "\"out_hidden_size\": 32")
            .replacingOccurrences(of: "\"head_dim\": 8", with: "\"head_dim\": 32")
        let model = Qwen3VL(try decodeConfiguration(quantizableJSON))
        let embedding = model.languageModel.model.embedTokens
        quantize(
            model: model, groupSize: 32, bits: 4,
            filter: { _, layer in layer === embedding })
        XCTAssertTrue(model.languageModel.model.embedTokens is QuantizedEmbedding)

        let activationDType = model.scaledInputEmbeddings(
            MLXArray([Int32(0)]).reshaped([1, 1])
        ).dtype
        let vision = try model.cbv2VisionFeatures(
            imagePixels: MLXArray.zeros([4, 12]),
            imageGrids: [THW(1, 2, 2)])

        XCTAssertEqual(vision.features.map(\.dtype), [activationDType])
        XCTAssertNotEqual(model.languageModel.model.embedTokens.weight.dtype, activationDType)
    }

    func testCBv2DecodePositionsUseEveryRowsOffsetAndDelta() {
        let first = CBv2PositionState(
            promptPositionIds: MLXArray.zeros([3, 1, 1], dtype: .int32),
            decodeDeltas: [-2])
        let second = CBv2PositionState(
            promptPositionIds: MLXArray.zeros([3, 1, 1], dtype: .int32),
            decodeDeltas: [5])

        let positions = CBv2PositionState.decodePositionIds(
            states: [first, second], cacheOffsets: [10, 20], length: 2)!
        eval(positions)

        XCTAssertEqual(positions.shape, [3, 2, 2])
        XCTAssertEqual(
            positions.asArray(Int32.self),
            [8, 9, 25, 26, 8, 9, 25, 26, 8, 9, 25, 26])
    }

    func testDenseOrdinaryForwardRetainsLegacyPath() throws {
        let model = Qwen3VL(try decodeConfiguration(tinyDenseConfigurationJSON))
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])

        let actual = model(tokens, cache: nil)
        let expected = model.languageModel(
            tokens,
            cache: nil,
            inputEmbeddings: nil,
            mask: nil,
            positionIds: nil,
            visualMask: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil).logits
        eval(actual, expected)

        XCTAssertEqual(actual.shape, expected.shape)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    private func splitExpertCheckpoint(
        from model: Qwen3VL,
        rawLanguagePrefix: Bool
    ) -> [String: MLXArray] {
        let reference = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var checkpoint: [String: MLXArray] = [:]
        for (key, value) in reference {
            guard key.hasSuffix(".switch_mlp.gate_up_proj.weight") else {
                checkpoint[key] = value
                continue
            }
            var base = String(key.dropLast("gate_up_proj.weight".count))
            if rawLanguagePrefix {
                base = base.replacingOccurrences(
                    of: "language_model.model.", with: "model.language_model.")
            }
            let halves = split(value, parts: 2, axis: -2)
            checkpoint["\(base)gate_proj.weight"] = halves[0]
            checkpoint["\(base)up_proj.weight"] = halves[1]
        }
        return checkpoint
    }

    private func decodeConfiguration(_ json: String) throws -> Qwen3VLConfiguration {
        try JSONDecoder().decode(Qwen3VLConfiguration.self, from: Data(json.utf8))
    }

    private func assertApproximation(
        _ gelu: GELU, is expected: GELU.Approximation,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let matches: Bool
        switch (gelu.approximation, expected) {
        case (.none, .none), (.precise, .precise), (.tanh, .tanh), (.fast, .fast):
            matches = true
        default:
            matches = false
        }
        XCTAssertTrue(matches, file: file, line: line)
    }
}

private func deterministicValues(_ count: Int, scale: Float, offset: Int) -> [Float] {
    (0 ..< count).map { index in
        Float(((index * 7 + offset) % 17) - 8) * scale
    }
}

private struct SparseReference {
    let indices: [[Int]]
    let scores: [[Int: Float]]
    let output: [Float]
}

private func independentSparseReference(
    input: [Float], rows: Int, dimensions: Int,
    gateWeights: [Float], expertGateWeights: [Float],
    expertUpWeights: [Float], expertDownWeights: [Float],
    experts: Int, topK: Int, hiddenDimensions: Int
) -> SparseReference {
    var selectedByRow: [[Int]] = []
    var scoresByRow: [[Int: Float]] = []
    var output = [Float](repeating: 0, count: rows * dimensions)

    func dot(_ lhs: [Float], _ lhsOffset: Int, _ rhs: [Float], _ rhsOffset: Int, _ n: Int)
        -> Float
    {
        var result: Float = 0
        for index in 0 ..< n {
            result += lhs[lhsOffset + index] * rhs[rhsOffset + index]
        }
        return result
    }

    for row in 0 ..< rows {
        let inputOffset = row * dimensions
        let logits = (0 ..< experts).map { expert in
            dot(input, inputOffset, gateWeights, expert * dimensions, dimensions)
        }
        let maximum = logits.max()!
        let exponentials = logits.map { Foundation.exp($0 - maximum) }
        let denominator = exponentials.reduce(0, +)
        let probabilities = exponentials.map { $0 / denominator }
        let selected = probabilities.indices.sorted {
            probabilities[$0] > probabilities[$1]
        }.prefix(topK).map { $0 }
        let selectedDenominator = selected.reduce(Float(0)) {
            $0 + probabilities[$1]
        }
        let scoreMap = Dictionary(uniqueKeysWithValues: selected.map {
            ($0, probabilities[$0] / selectedDenominator)
        })

        selectedByRow.append(selected)
        scoresByRow.append(scoreMap)

        for expert in selected {
            var activated = [Float](repeating: 0, count: hiddenDimensions)
            for hidden in 0 ..< hiddenDimensions {
                let expertOffset = (expert * hiddenDimensions + hidden) * dimensions
                let gateValue = dot(
                    input, inputOffset, expertGateWeights, expertOffset, dimensions)
                let upValue = dot(
                    input, inputOffset, expertUpWeights, expertOffset, dimensions)
                activated[hidden] = gateValue / (1 + Foundation.exp(-gateValue)) * upValue
            }

            let score = scoreMap[expert]!
            for dimension in 0 ..< dimensions {
                let downOffset = (expert * dimensions + dimension) * hiddenDimensions
                let expertValue = dot(
                    activated, 0, expertDownWeights, downOffset, hiddenDimensions)
                output[inputOffset + dimension] += score * expertValue
            }
        }
    }

    return SparseReference(indices: selectedByRow, scores: scoresByRow, output: output)
}

private let tinyVisionConfigurationJSON = """
    "vision_config": {
        "model_type": "qwen3_vl_moe",
        "depth": 1,
        "hidden_size": 8,
        "hidden_act": "gelu_pytorch_tanh",
        "intermediate_size": 16,
        "out_hidden_size": 8,
        "num_heads": 1,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "num_position_embeddings": 8,
        "deepstack_visual_indexes": []
    }
    """

private let tinyDenseVisionConfigurationJSON = tinyVisionConfigurationJSON.replacingOccurrences(
    of: "qwen3_vl_moe", with: "qwen3_vl").replacingOccurrences(
        of: "gelu_pytorch_tanh", with: "gelu")

private let realShapeConfigurationJSON = """
    {
        "model_type": "qwen3_vl_moe",
        "text_config": {
            "model_type": "qwen3_vl_moe_text",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "rope_theta": 5000000,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 0.000001,
            "vocab_size": 151936,
            "tie_word_embeddings": false,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "moe_intermediate_size": 768,
            "norm_topk_prob": true
        },
        \(tinyVisionConfigurationJSON)
    }
    """

private let tinyDenseConfigurationJSON = """
    {
        "model_type": "qwen3_vl",
        "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "vocab_size": 32
        },
        \(tinyDenseVisionConfigurationJSON)
    }
    """

private let tinyMoEConfigurationJSON = """
    {
        "model_type": "qwen3_vl_moe",
        "text_config": {
            "model_type": "qwen3_vl_moe_text",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "vocab_size": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 2,
            "mlp_only_layers": [3],
            "moe_intermediate_size": 4,
            "norm_topk_prob": true
        },
        \(tinyVisionConfigurationJSON)
    }
    """

private let tinyAllMoEConfigurationJSON = """
    {
        "model_type": "qwen3_vl_moe",
        "text_config": {
            "model_type": "qwen3_vl_moe_text",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "vocab_size": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "moe_intermediate_size": 4,
            "norm_topk_prob": true
        },
        \(tinyVisionConfigurationJSON)
    }
    """
