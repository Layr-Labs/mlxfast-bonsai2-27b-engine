// Copyright © 2026 Eigen Labs.

import Foundation
import XCTest
@testable import MLX
@testable import MLXNN
@testable import MLXFast
@testable import MLXLLM
import MLXLMCommon

final class GatedDeltaFusedInputProjectionTests: XCTestCase {

    func testFusedInputProjectionParity() {
        let hiddenSize = 2048
        let qkvDim = 8192
        let zDim = 4096
        let bDim = 32
        let aDim = 32
        let totalOut = qkvDim + zDim + bDim + aDim // 12352

        let linQKV = Linear(hiddenSize, qkvDim, bias: false)
        let linZ = Linear(hiddenSize, zDim, bias: false)
        let linB = Linear(hiddenSize, bDim, bias: false)
        let linA = Linear(hiddenSize, aDim, bias: false)

        // Quantize all to QuantizedLinear (4-bit, group 64)
        let qQKV = QuantizedLinear(linQKV, groupSize: 64, bits: 4)
        let qZ = QuantizedLinear(linZ, groupSize: 64, bits: 4)
        let qB = QuantizedLinear(linB, groupSize: 64, bits: 4)
        let qA = QuantizedLinear(linA, groupSize: 64, bits: 4)

        // Concatenate weights along axis 0
        let fusedWeight = concatenated([qQKV.weight, qZ.weight, qB.weight, qA.weight], axis: 0)
        let fusedScales = concatenated([qQKV.scales, qZ.scales, qB.scales, qA.scales], axis: 0)
        let fusedBiases = concatenated([qQKV.biases!, qZ.biases!, qB.biases!, qA.biases!], axis: 0)

        let fusedLayer = QuantizedLinear(
            weight: fusedWeight,
            bias: nil,
            scales: fusedScales,
            biases: fusedBiases,
            groupSize: 64,
            bits: 4
        )

        let S = 512
        let inputs = MLXRandom.normal([1, S, hiddenSize], type: Float.self).asType(Float.self)

        // 1. Evaluate separate
        let outQKV = qQKV(inputs)
        let outZ = qZ(inputs)
        let outB = qB(inputs)
        let outA = qA(inputs)
        eval(outQKV, outZ, outB, outA)

        // 2. Evaluate fused
        let outFused = fusedLayer(inputs)
        let fused_qkv = outFused[0..., 0..., 0 ..< qkvDim]
        let fused_z = outFused[0..., 0..., qkvDim ..< (qkvDim + zDim)]
        let fused_b = outFused[0..., 0..., (qkvDim + zDim) ..< (qkvDim + zDim + bDim)]
        let fused_a = outFused[0..., 0..., (qkvDim + zDim + bDim) ..< totalOut]
        eval(fused_qkv, fused_z, fused_b, fused_a)

        // Assert exact bitwise/numerical identity
        let diffQKV = abs(outQKV - fused_qkv).max().item(Float.self)
        let diffZ = abs(outZ - fused_z).max().item(Float.self)
        let diffB = abs(outB - fused_b).max().item(Float.self)
        let diffA = abs(outA - fused_a).max().item(Float.self)

        print("Max diff QKV:", diffQKV)
        print("Max diff Z:", diffZ)
        print("Max diff B:", diffB)
        print("Max diff A:", diffA)

        XCTAssertEqual(diffQKV, 0.0, accuracy: 1e-5)
        XCTAssertEqual(diffZ, 0.0, accuracy: 1e-5)
        XCTAssertEqual(diffB, 0.0, accuracy: 1e-5)
        XCTAssertEqual(diffA, 0.0, accuracy: 1e-5)
    }
}


private func smallQwenGDNConfiguration() throws -> Qwen35TextConfiguration {
    let json = """
    {
      "hidden_size": 64,
      "num_hidden_layers": 1,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "linear_num_value_heads": 2,
      "linear_num_key_heads": 1,
      "linear_key_head_dim": 32,
      "linear_value_head_dim": 32,
      "linear_conv_kernel_dim": 4,
      "vocab_size": 128,
      "full_attention_interval": 4
    }
    """
    return try JSONDecoder().decode(
        Qwen35TextConfiguration.self, from: Data(json.utf8))
}

extension GatedDeltaFusedInputProjectionTests {
    func testFusionKeepsCheckpointTopologyAndFrozenCache() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])

        XCTAssertTrue(layer.prepareFusedInputProjection())
        XCTAssertTrue(layer.hasFusedInputProjection)
        XCTAssertNotNil(layer.inProjQKV)
        XCTAssertNotNil(layer.inProjZ)
        XCTAssertNotNil(layer.inProjB)
        XCTAssertNotNil(layer.inProjA)
        let keys = Set(layer.parameters().flattened().map(\.0))
        XCTAssertFalse(keys.contains("in_proj_fused.weight"))
        XCTAssertTrue(keys.contains("in_proj_qkv.weight"))
        XCTAssertFalse(layer.trainableParameters().flattened().contains {
            $0.0.hasPrefix("in_proj_fused")
        })
    }

    func testHeterogeneousQuantizationRetainsSeparateProjections() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 8)),
                "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])

        XCTAssertFalse(layer.prepareFusedInputProjection())
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertNotNil(layer.inProjQKV)
        XCTAssertNotNil(layer.inProjB)
    }

    func testAdapterBackedProjectionRetainsSeparateCalls() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        let adapted = LoRALinear.from(
            linear: layer.inProjQKV, rank: 4, scale: 1) as! Linear
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(adapted),
                "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])

        XCTAssertFalse(layer.prepareFusedInputProjection())
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertTrue(layer.inProjQKV is LoRALinear)
    }
    func testBatchedMoEInputsFlattenTokensAndPreserveTopK() {
        let x = MLXArray.zeros([2, 7, 2048], dtype: .bfloat16)
        let indices = MLXArray.zeros([2, 7, 8], dtype: .uint32)
        let scores = MLXArray.zeros([2, 7, 8], dtype: .bfloat16)
        let flattened = qwen35FlattenMoEInputs(
            x: x, indices: indices, scores: scores)
        XCTAssertEqual(flattened.x.shape, [14, 2048])
        XCTAssertEqual(flattened.indices.shape, [14, 8])
        XCTAssertEqual(flattened.scores.shape, [14, 8])
    }

    func testUnfrozenQuantizedProjectionsRetainDifferentiablePath() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        let modules = ModuleChildren(values: [
            "in_proj_qkv": .value(QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
            "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
            "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
            "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
        ])
        try layer.update(modules: modules, verify: [])
        layer.unfreeze(recursive: true)
        XCTAssertFalse(layer.prepareFusedInputProjection())
        XCTAssertFalse(layer.hasFusedInputProjection)
    }

    func testReplacingSourceProjectionInvalidatesInferenceCache() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])
        XCTAssertTrue(layer.prepareFusedInputProjection())
        XCTAssertTrue(layer.hasFusedInputProjection)

        let adapted = LoRALinear.from(
            linear: layer.inProjQKV, rank: 4, scale: 1) as! Linear
        try layer.update(
            modules: ModuleChildren(values: ["in_proj_qkv": .value(adapted)]),
            verify: [])
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertFalse(layer.prepareFusedInputProjection())
        XCTAssertFalse(
            ObjectIdentifier(type(of: layer.inProjQKV))
                == ObjectIdentifier(QuantizedLinear.self))
    }

    func testReplacingSourceParametersInvalidatesInferenceCache() throws {
        let layer = Qwen35GatedDeltaNet(try smallQwenGDNConfiguration())
        try layer.update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(QuantizedLinear(layer.inProjQKV, groupSize: 32, bits: 4)),
                "in_proj_z": .value(QuantizedLinear(layer.inProjZ, groupSize: 32, bits: 4)),
                "in_proj_b": .value(QuantizedLinear(layer.inProjB, groupSize: 32, bits: 4)),
                "in_proj_a": .value(QuantizedLinear(layer.inProjA, groupSize: 32, bits: 4)),
            ]), verify: [])
        XCTAssertTrue(layer.prepareFusedInputProjection())
        let replacement = layer.inProjQKV.weight + MLXArray.zeros(
            layer.inProjQKV.weight.shape, dtype: layer.inProjQKV.weight.dtype)
        try layer.update(
            parameters: ModuleParameters.unflattened([
                "in_proj_qkv.weight": replacement,
            ]), verify: [])
        XCTAssertFalse(layer.hasFusedInputProjection)
        XCTAssertTrue(layer.prepareFusedInputProjection())
        XCTAssertTrue(layer.hasFusedInputProjection)
    }

}
