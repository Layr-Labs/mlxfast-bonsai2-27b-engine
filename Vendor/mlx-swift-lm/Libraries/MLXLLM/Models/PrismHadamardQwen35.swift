// Copyright © 2026 Eigen Labs.
import Foundation
import MLXLMCommon

/// Text factory for Prism's folded Qwen3.8-27B pack.
///
/// The pack declares a vision tower. This track never serves an image, so the
/// load filter drops the 333 `vision_tower.*` tensors and the tower is never
/// built. `Qwen35Runner` declares `multimodal: false` for the same reason, so
/// the LLM factory is the only factory that resolves this model type.
///
/// The pack carries no MTP head (`mtp_num_hidden_layers: 0`). The track
/// attaches a separate, published head instead, so the speculative capability
/// stays as `Qwen35Model` declares it and the runner decides whether a head is
/// present. See `docs/bonsai2.md`.
public final class PrismHadamardQwen35TextModel: Qwen35Model, PrismHadamardLoading,
    WeightNameFiltering
{
    public let prismCheckpoint: PrismHadamardCheckpointConfiguration

    public init(configurationData: Data) throws {
        prismCheckpoint = try JSONDecoder().decode(PrismHadamardCheckpointConfiguration.self, from: configurationData)
        super.init(try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configurationData))
    }
    public func shouldLoadWeight(named name: String) -> Bool {
        !name.hasPrefix("vision_tower.")
    }
}
