import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLMCommon

@Suite("incremental checkpoint materialization", .serialized)
struct IncrementalCheckpointMaterializationTests {
    private final class Observation {
        weak var original: MLXArray?
        var calls = 0
        var stagingReleasedBeforeHook = false
    }
    private final class JoinedModel: Module, BaseLanguageModel, IncrementalCheckpointMaterializing {
        @ParameterInfo(key: "joined") var joined: MLXArray
        let observation: Observation
        let needsIncrementalCheckpointMaterialization: Bool

        init(observation: Observation, enabled: Bool) {
            self.observation = observation
            self.needsIncrementalCheckpointMaterialization = enabled
            self._joined.wrappedValue = MLXArray.zeros([4])
            super.init()
        }
        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            observation.original = weights["left"]
            return ["joined": concatenated([weights["left"]!, weights["right"]!])]
        }
        func materializeCheckpointWeightsIncrementally() throws {
            observation.calls += 1
            observation.stagingReleasedBeforeHook = observation.original == nil
            try MLX.checkedEval(joined)
        }
    }

    @Test("opt-in hook runs after old Swift shard/staging owners are gone")
    func releasedBeforeMaterialization() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(arrays: ["left": MLXArray([Float(1), 2]), "right": MLXArray([Float(3), 4])],
                 url: directory.appendingPathComponent("model.safetensors"))
        let observation = Observation()
        let model = JoinedModel(observation: observation, enabled: true)
        try loadWeights(modelDirectory: directory, model: model)
        #expect(observation.calls == 1)
        #expect(observation.stagingReleasedBeforeHook)
        #expect(model.joined.asArray(Float.self) == [1, 2, 3, 4])
    }

    @Test("inactive policy retains normal loader behavior")
    func inactivePolicy() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(arrays: ["left": MLXArray([Float(1), 2]), "right": MLXArray([Float(3), 4])],
                 url: directory.appendingPathComponent("model.safetensors"))
        let observation = Observation()
        let model = JoinedModel(observation: observation, enabled: false)
        try loadWeights(modelDirectory: directory, model: model)
        #expect(observation.calls == 0)
        #expect(model.joined.asArray(Float.self) == [1, 2, 3, 4])
    }
}
