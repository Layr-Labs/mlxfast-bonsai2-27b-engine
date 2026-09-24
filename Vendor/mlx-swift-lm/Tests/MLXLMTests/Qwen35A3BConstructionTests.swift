import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen35A3BConstructionTests: XCTestCase {
    func testConfigurationInspectionConsumesEveryRouterEntry() throws {
        var quantization: [String: Any] = [
            "mode": "affine", "bits": 4, "group_size": 64,
        ]
        for layer in 0 ..< 40 {
            quantization["language_model.model.layers.\(layer).mlp.gate"] = ["bits": 8]
            quantization["language_model.model.layers.\(layer).mlp.shared_expert_gate"] = [
                "bits": 8
            ]
        }
        let root: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "text_config": [
                "model_type": "qwen3_5_moe_text", "hidden_size": 2_048,
                "num_hidden_layers": 40, "num_experts": 256,
                "num_experts_per_tok": 8, "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512, "mtp_num_hidden_layers": 1,
                "layer_types": Array(repeating: [
                    "linear_attention", "linear_attention", "linear_attention",
                    "full_attention",
                ], count: 10).flatMap { $0 },
            ],
            "quantization": quantization,
            "mtplx_mtp": ["included": true],
            "mtplx_mtp_quantization": [
                "mode": "mxfp8", "bits": 8, "group_size": 32,
            ],
        ]
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".json")
        try JSONSerialization.data(withJSONObject: root).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let contract = try Qwen35A3BArtifactContract.inspect(configurationURL: url)
        XCTAssertEqual(contract.geometry.recurrentLayers, 30)
        XCTAssertEqual(contract.geometry.fullAttentionLayers, 10)

        quantization["language_model.model.layers.0.linear_attn.in_proj_qkv"] = [
            "bits": 8, "group_size": 64,
        ]
        var wrongProjectionRoot = root
        wrongProjectionRoot["quantization"] = quantization
        try JSONSerialization.data(withJSONObject: wrongProjectionRoot).write(to: url)
        XCTAssertThrowsError(try Qwen35A3BArtifactContract.inspect(configurationURL: url)) {
            XCTAssertEqual(
                $0 as? Qwen35A3BArtifactError,
                .mismatch(
                    field:
                        "quantization.language_model.model.layers.0.linear_attn.in_proj_qkv.bits",
                    expected: "4", actual: "8"))
        }
        quantization.removeValue(
            forKey: "language_model.model.layers.0.linear_attn.in_proj_qkv")

        quantization.removeValue(
            forKey: "language_model.model.layers.39.mlp.shared_expert_gate")
        var badRoot = root
        badRoot["quantization"] = quantization
        try JSONSerialization.data(withJSONObject: badRoot).write(to: url)
        XCTAssertThrowsError(try Qwen35A3BArtifactContract.inspect(configurationURL: url))
    }

    func testExactEigenLabsContractSelectsSeparateTargetAndMTPPacking() throws {
        let contract = try Qwen35A3BArtifactContract.inspect(
            fixture: .eigenLabsRouter8)

        XCTAssertEqual(
            contract.target,
            .affine(bits: 4, groupSize: 64, routerBits: 8, routerGroupSize: 64))
        XCTAssertEqual(contract.mtp, .mxfp8(bits: 8, groupSize: 32))
        XCTAssertEqual(
            contract.geometry,
            .init(
                hidden: 2_048, experts: 256, topK: 8, expertIntermediate: 512,
                sharedIntermediate: 512, layers: 40, recurrentLayers: 30,
                fullAttentionLayers: 10))
        XCTAssertEqual(contract.targetReaderIdentity, "affine-w4-g64")
        XCTAssertEqual(contract.mtpReaderIdentity, "mxfp8-g32")
        XCTAssertNotEqual(contract.targetReaderIdentity, contract.mtpReaderIdentity)
    }

    func testWrongMTPPackingFailsBeforeInstallation() throws {
        var fixture = Qwen35A3BArtifactFixture.eigenLabsRouter8
        fixture.mtpGroupSize = 64

        XCTAssertThrowsError(try Qwen35A3BArtifactContract.inspect(fixture: fixture)) {
            XCTAssertEqual(
                $0 as? Qwen35A3BArtifactError,
                .mismatch(field: "mtp.group_size", expected: "32", actual: "64"))
        }
    }

    func testExactProjectionRejectsUnquantizedModulesBeforeExecution() {
        let linear = Linear(64, 64, bias: false)
        XCTAssertThrowsError(
            try qwen35A3BValidateExactProjection(linear, path: "model.layers.0.q_proj")
        ) {
            XCTAssertEqual(
                $0 as? Qwen35A3BArtifactError,
                .mismatch(
                    field: "loaded.model.layers.0.q_proj.type",
                    expected: "QuantizedLinear", actual: "Linear"))
        }
    }

    func testWrongLayerOwnershipFailsBeforeInstallation() throws {
        var fixture = Qwen35A3BArtifactFixture.eigenLabsRouter8
        fixture.recurrentLayers = 29
        fixture.fullAttentionLayers = 11

        XCTAssertThrowsError(try Qwen35A3BArtifactContract.inspect(fixture: fixture))
    }

    func testProfilePublishesOnlyACompleteImmutableRoute() throws {
        let contract = try Qwen35A3BArtifactContract.inspect(
            fixture: .eigenLabsRouter8)
        let stock = try Qwen35A3BRouteTable.install(contract: contract, profile: .stock)
        let prefill = try Qwen35A3BRouteTable.install(contract: contract, profile: .prefill)

        XCTAssertEqual(stock.profile, .stock)
        XCTAssertEqual(stock.prefill, .stock)
        XCTAssertEqual(stock.targetDecode, .stock)
        XCTAssertEqual(stock.mtpDecode, .disabled)
        XCTAssertEqual(prefill.profile, .prefill)
        XCTAssertEqual(prefill.prefill, .rightShaped)
        XCTAssertEqual(prefill.targetDecode, .stock)
        XCTAssertEqual(prefill.mtpDecode, .disabled)
    }

    func testRowOwnedRouterPreservesExactTop8Contract() throws {
        for rows in [1, 2, 3, 16] {
            let logits = MLXArray(0 ..< rows * 256)
                .asType(.float32).reshaped([rows, 256])
            let probabilities = softmax(
                sin(logits * 0.173) + cos(logits * 0.071),
                axis: -1, precise: true).asType(.bfloat16)
            let kth = 256 - 8
            let expectedIDs = argPartition(
                probabilities, kth: kth, axis: -1)[0..., kth...]
            var expectedScores = takeAlong(
                probabilities, expectedIDs, axis: -1)
            expectedScores = expectedScores
                / expectedScores.sum(axis: -1, keepDims: true)

            let (actualIDs, actualScores) = qwen35A3BRowOwnedRoute(
                probabilities, rows: rows)
            let idDifference = max(abs(
                expectedIDs.asType(.int32) - actualIDs.asType(.int32)))
            let scoreDifference = max(abs(
                expectedScores.asType(.float32)
                    - actualScores.asType(.float32)))
            eval(idDifference, scoreDifference)
            XCTAssertEqual(idDifference.item(Int32.self), 0, "rows=\(rows)")
            XCTAssertEqual(scoreDifference.item(Float.self), 0, "rows=\(rows)")
        }
    }

    func testOutputOwnedCombinePreservesBF16ReductionOrder() throws {
        let contract = try Qwen35A3BArtifactContract.inspect(fixture: .eigenLabsRouter8)
        let installation = try Qwen35A3BConstructionInstallation.install(
            contract: contract, profile: .decode)
        let combine = Qwen35A3BConstructionContext.withInstallation(installation) {
            qwen35A3BExpertCombiner(hidden: 2_048, topK: 8)
        }

        for rows in [1, 2] {
            let routed = sin(MLXArray(0 ..< rows * 8 * 2_048)
                .asType(.float32) * 0.013)
                .asType(.bfloat16).reshaped([rows, 8, 2_048])
            let scores = softmax(MLXArray(0 ..< rows * 8)
                .asType(.float32).reshaped([rows, 8]), axis: -1)
                .asType(.bfloat16)
            let expected = (routed * expandedDimensions(scores, axis: -1))
                .sum(axis: -2)
            let actual = combine(routed, scores)
            let difference = max(abs(
                expected.asType(.float32) - actual.asType(.float32)))
            eval(difference)
            XCTAssertEqual(difference.item(Float.self), 0, "rows=\(rows)")
        }
    }

    func testExactTargetVerifyBuildsOnlyM1ProjectionCalls() {
        var observedShapes: [[Int]] = []
        let input = MLXArray(0 ..< 12).asType(.bfloat16).reshaped([1, 3, 4])
        let output = qwen35A3BTimewiseProjection(input) { row in
            observedShapes.append(row.shape)
            return row + 1
        }
        eval(output)

        XCTAssertEqual(observedShapes, [[1, 1, 4], [1, 1, 4], [1, 1, 4]])
        XCTAssertEqual(output.shape, [1, 3, 4])
        XCTAssertEqual(output.asArray(Int.self), Array(1 ... 12))
    }

    func testExactW4G64VerifyMatchesIndependentM1QMVBitwise() {
        let inputSize = 512
        let outputSize = 16
        let positions = 3
        let weights = sin(
            MLXArray(0 ..< outputSize * inputSize).asType(.float32) * 0.013
        ).asType(.bfloat16).reshaped([outputSize, inputSize])
        let linear = QuantizedLinear(
            weight: weights, bias: nil, groupSize: 64, bits: 4, mode: .affine)
        let input = cos(
            MLXArray(0 ..< positions * inputSize).asType(.float32) * 0.017
        ).asType(.bfloat16).reshaped([1, positions, inputSize])

        let expected = qwen35A3BTimewiseProjection(input) { linear($0) }
        let actual = qwen35A3BExactW4G64Projection(linear, input)
        let difference = max(abs(
            actual.asType(.float32) - expected.asType(.float32)))
        eval(expected, actual, difference)

        XCTAssertEqual(actual.shape, [1, positions, outputSize])
        XCTAssertEqual(difference.item(Float.self), 0)
    }

    func testExactW4G64ProjectionPairMatchesIndependentM1QMVBitwise() {
        let inputSize = 512
        let outputSize = 16
        let positions = 3
        let indices = MLXArray(0 ..< outputSize * inputSize).asType(.float32)
        let first = QuantizedLinear(
            weight: sin(indices * 0.013).asType(.bfloat16)
                .reshaped([outputSize, inputSize]),
            bias: nil, groupSize: 64, bits: 4, mode: .affine)
        let second = QuantizedLinear(
            weight: cos(indices * 0.019).asType(.bfloat16)
                .reshaped([outputSize, inputSize]),
            bias: nil, groupSize: 64, bits: 4, mode: .affine)
        let input = cos(
            MLXArray(0 ..< positions * inputSize).asType(.float32) * 0.017
        ).asType(.bfloat16).reshaped([1, positions, inputSize])

        let expectedFirst = qwen35A3BTimewiseProjection(input) { first($0) }
        let expectedSecond = qwen35A3BTimewiseProjection(input) { second($0) }
        let (actualFirst, actualSecond) = qwen35A3BExactW4G64ProjectionPair(
            first, second, input)
        let firstDifference = max(abs(
            actualFirst.asType(.float32) - expectedFirst.asType(.float32)))
        let secondDifference = max(abs(
            actualSecond.asType(.float32) - expectedSecond.asType(.float32)))
        eval(expectedFirst, expectedSecond, actualFirst, actualSecond,
             firstDifference, secondDifference)

        XCTAssertEqual(actualFirst.shape, [1, positions, outputSize])
        XCTAssertEqual(actualSecond.shape, [1, positions, outputSize])
        XCTAssertEqual(firstDifference.item(Float.self), 0)
        XCTAssertEqual(secondDifference.item(Float.self), 0)
    }

    func testExactW4G64ProjectionQuadMatchesIndependentM1QMVBitwise() {
        let inputSize = 512
        let positions = 3
        func makeLinear(_ outputSize: Int, _ scale: Float) -> QuantizedLinear {
            let indices = MLXArray(0 ..< outputSize * inputSize).asType(.float32)
            return QuantizedLinear(
                weight: sin(indices * scale).asType(.bfloat16)
                    .reshaped([outputSize, inputSize]),
                bias: nil, groupSize: 64, bits: 4, mode: .affine)
        }
        let first = makeLinear(16, 0.011)
        let second = makeLinear(20, 0.013)
        let third = makeLinear(24, 0.017)
        let fourth = makeLinear(28, 0.019)
        let input = cos(
            MLXArray(0 ..< positions * inputSize).asType(.float32) * 0.023
        ).asType(.bfloat16).reshaped([1, positions, inputSize])

        let expectedFirst = qwen35A3BTimewiseProjection(input) { first($0) }
        let expectedSecond = qwen35A3BTimewiseProjection(input) { second($0) }
        let expectedThird = qwen35A3BTimewiseProjection(input) { third($0) }
        let expectedFourth = qwen35A3BTimewiseProjection(input) { fourth($0) }
        let (actualFirst, actualSecond, actualThird, actualFourth) =
            qwen35A3BExactW4G64ProjectionQuad(
                first, second, third, fourth, input)
        let differences = [
            max(abs(actualFirst.asType(.float32) - expectedFirst.asType(.float32))),
            max(abs(actualSecond.asType(.float32) - expectedSecond.asType(.float32))),
            max(abs(actualThird.asType(.float32) - expectedThird.asType(.float32))),
            max(abs(actualFourth.asType(.float32) - expectedFourth.asType(.float32))),
        ]
        eval(actualFirst, actualSecond, actualThird, actualFourth)
        eval(differences)

        XCTAssertEqual(actualFirst.shape, [1, positions, 16])
        XCTAssertEqual(actualSecond.shape, [1, positions, 20])
        XCTAssertEqual(actualThird.shape, [1, positions, 24])
        XCTAssertEqual(actualFourth.shape, [1, positions, 28])
        for difference in differences {
            XCTAssertEqual(difference.item(Float.self), 0)
        }
    }

    func testStockRouterKeepsTheLastAxisForWidePrefill() throws {
        let finalize = qwen35A3BRouterFinalizer(
            hidden: 2_048, experts: 256, topK: 8, normalize: true)
        let probabilities = softmax(
            MLXArray(0 ..< 129 * 256).asType(.float32)
                .reshaped([1, 129, 256]), axis: -1, precise: true)
                .asType(.bfloat16)
        let (ids, scores) = finalize(probabilities)
        XCTAssertEqual(ids.shape, [1, 129, 8])
        XCTAssertEqual(scores.shape, [1, 129, 8])
    }

    func testInstalledHotRoutesDoNotRevalidateConstructionInvariants() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repositoryRoot.appendingPathComponent(
            "Libraries/MLXLLM/Models/Qwen35A3BOptimization.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let hotRoutes = try XCTUnwrap(
            source.components(separatedBy: "// MARK: - Exact E256/K8").last)

        XCTAssertFalse(hotRoutes.contains("precondition("))
        XCTAssertFalse(hotRoutes.contains("probabilities.dtype"))
        XCTAssertFalse(hotRoutes.contains("routed.dtype =="))
        XCTAssertFalse(hotRoutes.contains("scores.dtype =="))

        let exactSourceURL = repositoryRoot.appendingPathComponent(
            "Libraries/MLXLLM/Models/Qwen35A3BTargetVerify.swift")
        let exactSource = try String(contentsOf: exactSourceURL, encoding: .utf8)
        let installedExactRoute = try XCTUnwrap(
            exactSource.components(
                separatedBy: "func qwen35A3BExactW4G64Projection").last)
        XCTAssertFalse(installedExactRoute.contains(".bits"))
        XCTAssertFalse(installedExactRoute.contains(".groupSize"))
        XCTAssertFalse(installedExactRoute.contains(".mode"))
        XCTAssertFalse(installedExactRoute.contains("guard "))
    }
}
