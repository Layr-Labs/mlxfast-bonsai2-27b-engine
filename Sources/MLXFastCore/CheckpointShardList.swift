import Foundation

/// The shard list of a safetensors checkpoint index, read by trusted code only.
///
/// `mlxfast-swift checkpoint-shards` calls this during `./setup.sh`, before
/// any sandbox is in place. It lives in this trusted module so that the setup
/// step runs no code from an editable path. The output is the same as the
/// earlier transform-module helper: the unique shard names of `weight_map`,
/// sorted, each one checked by `validateSafetensorsShardName`.
public enum CheckpointShardList {
    public static func safetensorShardNames(fromIndexAt indexPath: String) throws -> [String] {
        let indexURL = URL(fileURLWithPath: indexPath)
        let data = try Data(contentsOf: indexURL)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let raw = object as? [String: Any] else {
            throw MLXFastError.invalidInput("checkpoint index must be a JSON object: \(indexURL.path)")
        }
        guard let weightMap = raw["weight_map"] as? [String: String] else {
            throw MLXFastError.invalidInput("checkpoint index missing weight_map: \(indexURL.path)")
        }
        let shards = Set(weightMap.values)
        guard !shards.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no shard names: \(indexPath)")
        }
        let sorted = shards.sorted()
        for shard in sorted {
            try validateSafetensorsShardName(shard, context: "checkpoint index")
        }
        return sorted
    }
}
