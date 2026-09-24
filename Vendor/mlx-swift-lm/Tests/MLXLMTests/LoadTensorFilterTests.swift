// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

/// True when `array` holds its data, i.e. it was materialized.
///
/// Shared with the model-side filter tests.
///
/// mlx builds one unevaluated array per safetensors tensor; the bytes are read
/// when the array is evaluated. `_mlx_array_is_available` therefore counts
/// materialization exactly: it is `false` for a tensor whose bytes were never
/// read and `true` after `eval`.
func isMaterialized(_ array: MLXArray) -> Bool {
    var available = false
    _ = _mlx_array_is_available(&available, array.ctx)
    return available
}

/// Records the keys `sanitize` receives. Not an `MLXArray` holder, so the
/// module parameter walk ignores it.
private final class SanitizeRecorder: @unchecked Sendable {
    var keys: [String] = []
}

private class PlainStubModel: Module, BaseLanguageModel {
    let recorder = SanitizeRecorder()
    var alpha: MLXArray = MLXArray.zeros([4])
    var beta: MLXArray = MLXArray.zeros([4])

    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        recorder.keys = weights.keys.sorted()
        return weights
    }
}

private class FilteringStubModel: Module, BaseLanguageModel, WeightNameFiltering {
    let recorder = SanitizeRecorder()
    var alpha: MLXArray = MLXArray.zeros([4])

    func shouldLoadWeight(named name: String) -> Bool {
        !name.contains("beta")
    }

    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        recorder.keys = weights.keys.sorted()
        return weights
    }
}

private func withCheckpoint(
    _ weights: [String: MLXArray], _ body: (URL) throws -> Void
) throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("load-tensor-filter-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
    try body(directory)
}

@Suite("Load tensor filter")
struct LoadTensorFilterTests {

    /// The mechanism the seam stands on: the safetensors reader returns
    /// unevaluated arrays, so a name dropped before `eval` never reads bytes.
    @Test func safetensorsArraysArriveUnmaterialized() throws {
        let source: [String: MLXArray] = [
            "alpha": MLXArray(Array(repeating: Float(1), count: 1024)),
            "beta": MLXArray(Array(repeating: Float(2), count: 1024)),
        ]
        try withCheckpoint(source) { directory in
            let (loaded, _) = try loadArraysAndMetadata(
                url: directory.appendingPathComponent("model.safetensors"))

            // Nothing is read by the load itself.
            for (key, value) in loaded {
                #expect(!isMaterialized(value), "\(key) was materialized by the load")
            }

            // Evaluating one tensor materializes that tensor only. This is
            // what makes dropping a key sufficient.
            let kept = loaded.filter { $0.key != "beta" }
            eval(Array(kept.values))
            #expect(isMaterialized(try #require(loaded["alpha"])))
            #expect(!isMaterialized(try #require(loaded["beta"])))
        }
    }

    @Test func excludedTensorsNeverReachTheModel() throws {
        let source: [String: MLXArray] = [
            "alpha": MLXArray([Float(1), 2, 3, 4]),
            "beta": MLXArray([Float(5), 6, 7, 8]),
        ]
        try withCheckpoint(source) { directory in
            let model = FilteringStubModel()
            try loadWeights(modelDirectory: directory, model: model)

            #expect(model.recorder.keys == ["alpha"])
            #expect(model.parameters().flattened().map(\.0).sorted() == ["alpha"])
            #expect(model.alpha.asArray(Float.self) == [1, 2, 3, 4])
        }
    }

    @Test func modelWithoutTheConformanceLoadsEveryTensor() throws {
        let source: [String: MLXArray] = [
            "alpha": MLXArray([Float(1), 2, 3, 4]),
            "beta": MLXArray([Float(5), 6, 7, 8]),
        ]
        try withCheckpoint(source) { directory in
            let model = PlainStubModel()
            try loadWeights(modelDirectory: directory, model: model)

            #expect(model.recorder.keys == ["alpha", "beta"])
            #expect(model.parameters().flattened().map(\.0).sorted() == ["alpha", "beta"])
            #expect(model.alpha.asArray(Float.self) == [1, 2, 3, 4])
            #expect(model.beta.asArray(Float.self) == [5, 6, 7, 8])
        }
    }
}
