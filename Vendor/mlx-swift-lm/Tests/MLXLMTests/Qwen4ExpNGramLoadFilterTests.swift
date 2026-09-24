// Qwen4ExpNGramLoadFilterTests.swift
//
// The n-gram shards must not enter memory at load.
//
// The table is disk resident: rows reach the model through a
// `Qwen4ExpNGramRowSource`. `Qwen4ExpModel` therefore conforms to
// `WeightNameFiltering` and answers `false` for a shard name, so the loader
// drops the name before it materializes the array. `sanitize` still drops the
// same names, for the callers that read weights themselves.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpNGramLoadFilterTests: XCTestCase {

    private let shardPrefix =
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"

    private func shardTensors() -> [String: MLXArray] {
        [
            "\(shardPrefix).weight": MLXArray(Array(repeating: Float(1), count: 8 * 1024)),
            "\(shardPrefix).scales": MLXArray(Array(repeating: Float(2), count: 8)),
            "\(shardPrefix).biases": MLXArray(Array(repeating: Float(3), count: 8)),
        ]
    }

    private func withCheckpoint(_ body: (URL, [String: MLXArray]) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4exp-ngram-filter-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let source = try Qwen4ExpFixture.model()
        let parameters = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        eval(Array(parameters.values))
        try save(
            arrays: parameters, url: directory.appendingPathComponent("model-00001.safetensors"))
        try save(
            arrays: shardTensors(),
            url: directory.appendingPathComponent("model-00002.safetensors"))
        try body(directory, parameters)
    }

    /// End to end: after `loadWeights` the model holds no n-gram tensor, and
    /// every real parameter arrived unchanged.
    func testLoadedModelHoldsNoNGramTensor() throws {
        try withCheckpoint { directory, parameters in
            let model = Qwen4ExpModel(text: try Qwen4ExpFixture.configuration(), withMTP: true)
            try loadWeights(modelDirectory: directory, model: model)

            let loaded = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            XCTAssertTrue(loaded.keys.allSatisfy { !$0.contains(".ngram_embedding.shard_") })
            XCTAssertEqual(loaded.keys.sorted(), parameters.keys.sorted())
            for (key, expected) in parameters {
                let actual = try XCTUnwrap(loaded[key])
                XCTAssertTrue(
                    allClose(actual, expected).item(Bool.self), "\(key) changed during the load")
            }
        }
    }

    /// The predicate the loader consults.
    func testShardNamesAreExcludedAndOtherNamesAreKept() throws {
        let model = try Qwen4ExpFixture.model()
        XCTAssertFalse(model.shouldLoadWeight(named: "\(shardPrefix).weight"))
        XCTAssertFalse(model.shouldLoadWeight(named: "\(shardPrefix).scales"))
        XCTAssertTrue(
            model.shouldLoadWeight(named: "model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(model.shouldLoadWeight(named: "lm_head.weight"))
    }

    /// The materialization count: zero shard bytes.
    ///
    /// This runs the loader's own sequence on the shard file — read the file,
    /// drop the excluded names with the model's predicate, evaluate the rest —
    /// and reads `_mlx_array_is_available`, which is `true` only for an array
    /// that holds its data. The shard arrays stay unmaterialized, so none of
    /// their bytes were read.
    func testShardBytesAreNeverRead() throws {
        try withCheckpoint { directory, _ in
            let model = try Qwen4ExpFixture.model()
            let (arrays, _) = try loadArraysAndMetadata(
                url: directory.appendingPathComponent("model-00002.safetensors"))
            XCTAssertEqual(arrays.count, 3)
            XCTAssertTrue(arrays.values.allSatisfy { !isMaterialized($0) })

            let kept = arrays.filter { model.shouldLoadWeight(named: $0.key) }
            XCTAssertTrue(kept.isEmpty)
            eval(Array(kept.values))

            for (key, value) in arrays {
                XCTAssertFalse(isMaterialized(value), "\(key) was materialized")
            }
        }
    }
}
