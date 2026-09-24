import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum NemotronH35MTPError: Error, LocalizedError {
    case invalidArtifact(String)
    public var errorDescription: String? {
        switch self { case .invalidArtifact(let detail): "Invalid Nemotron Lightning MTP artifact: \(detail)" }
    }
}

extension NemotronH35MTPAssistant {
    public static func load(from directory: URL, target: NemotronH35Model) throws -> NemotronH35MTPAssistant {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard configData.count <= 4 * 1024 * 1024 else { throw NemotronH35MTPError.invalidArtifact("config too large") }
        let args = try JSONDecoder().decode(NemotronH35Configuration.self, from: configData).target
        guard args.numNextnPredictLayers == 1, args.mtpLayersBlockType == ["attention", "moe"] else {
            throw NemotronH35MTPError.invalidArtifact("expected one attention/MoE prediction layer")
        }
        // A same-directory declaration is insufficient: all decoded architecture
        // fields must match the exact loaded target before sharing embeddings/head.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let decoded = try JSONDecoder().decode(NemotronH35Configuration.self, from: configData)
        guard try encoder.encode(decoded) == encoder.encode(target.lightningConfiguration) else {
            throw NemotronH35MTPError.invalidArtifact("configuration differs from target")
        }
        struct Index: Decodable { let weight_map: [String: String] }
        let indexData = try Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        guard indexData.count <= 16 * 1024 * 1024 else { throw NemotronH35MTPError.invalidArtifact("index too large") }
        let selected = try JSONDecoder().decode(Index.self, from: indexData).weight_map.filter { $0.key.hasPrefix("mtp.") }
        guard !selected.isEmpty, selected.count <= 512 else { throw NemotronH35MTPError.invalidArtifact("missing or oversized MTP inventory") }
        var weights: [String: MLXArray] = [:]
        for shard in Set(selected.values).sorted() {
            guard shard == URL(fileURLWithPath: shard).lastPathComponent,
                !shard.contains(".."), shard.hasSuffix(".safetensors") else {
                throw NemotronH35MTPError.invalidArtifact("invalid shard path")
            }
            let (arrays, _) = try loadArraysAndMetadata(url: directory.appendingPathComponent(shard))
            for (key, expected) in selected where expected == shard {
                guard let value = arrays[key] else { throw NemotronH35MTPError.invalidArtifact("indexed tensor missing: \(key)") }
                weights[String(key.dropFirst(4))] = value
            }
        }
        // Official BF16 checkpoints store individual experts. Converted MLX
        // checkpoints already carry the stacked SwitchLinear parameters.
        for (source, dest) in [("up_proj", "fc1"), ("down_proj", "fc2")] {
            let prefix = "layers.1.mixer"
            if weights["\(prefix).experts.0.\(source).weight"] != nil {
                var experts: [MLXArray] = []
                for e in 0..<args.nRoutedExperts {
                    guard let value = weights.removeValue(forKey: "\(prefix).experts.\(e).\(source).weight") else {
                        throw NemotronH35MTPError.invalidArtifact("incomplete expert bank")
                    }
                    experts.append(value)
                }
                weights["\(prefix).switch_mlp.\(dest).weight"] = stacked(experts)
            }
        }
        let assistant = NemotronH35MTPAssistant(target: target)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        let scaled = Set(weights.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(7)) })
        for path in scaled {
            guard base.perLayerQuantization?.quantization(layer: "mtp." + path) != nil else {
                throw NemotronH35MTPError.invalidArtifact("quantization missing for \(path)")
            }
        }
        quantize(model: assistant.module) { path, _ in
            scaled.contains(path) ? base.perLayerQuantization?.quantization(layer: "mtp." + path)?.asTuple : nil
        }
        try assistant.module.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(assistant.module)
        return assistant
    }
}
