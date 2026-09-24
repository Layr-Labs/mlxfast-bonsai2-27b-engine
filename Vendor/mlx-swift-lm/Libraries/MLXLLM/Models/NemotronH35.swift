//
//  NemotronH35.swift
//  mlx-swift-lm
//
//  Additive configuration boundary for Nemotron 3.5 Lightning checkpoints.
//  The target equations remain in NemotronHModel; this type keeps the newer
//  checkpoint contract distinct from legacy Nemotron Nano selection.

import Foundation
import MLX
import MLXLMCommon

public struct NemotronH35Configuration: Codable, Sendable {
    public let target: NemotronHConfiguration
    let hasExplicitExecutionLimits: Bool

    private enum CodingKeys: String, CodingKey {
        case layersBlockType = "layers_block_type"
        case timeStepLimit = "time_step_limit"
        case timeStepLimitMin = "time_step_limit_min"
        case timeStepLimitMax = "time_step_limit_max"
        case timeStepMin = "time_step_min"
        case timeStepMax = "time_step_max"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guard container.contains(.layersBlockType) else {
            throw DecodingError.keyNotFound(
                CodingKeys.layersBlockType,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "Nemotron 3.5 requires explicit layers_block_type"))
        }

        let target = try NemotronHConfiguration(from: decoder)
        guard target.mambaSSMCacheDType.lowercased() == "float32" else {
            throw DecodingError.dataCorruptedError(
                forKey: .layersBlockType, in: container,
                debugDescription:
                    "unsupported mamba_ssm_cache_dtype "
                    + target.mambaSSMCacheDType)
        }
        self.target = target
        hasExplicitExecutionLimits = try [CodingKeys.timeStepLimit, .timeStepLimitMin, .timeStepLimitMax]
            .contains { key in
                guard container.contains(key) else { return false }
                return !(try container.decodeNil(forKey: key))
            }
    }

    public func encode(to encoder: Encoder) throws {
        // JSON has no infinity. Encode native defaults as null, and retain
        // the distinction between training metadata and execution limits.
        var encoded = target
        if !encoded.timeStepLimitMax.isFinite { encoded.timeStepLimitMax = Float.greatestFiniteMagnitude }
        try encoded.encode(to: encoder)
        var container = encoder.container(keyedBy: CodingKeys.self)
        if hasExplicitExecutionLimits {
            try container.encode(target.timeStepLimitMin, forKey: .timeStepLimitMin)
            if target.timeStepLimitMax.isFinite {
                try container.encode(target.timeStepLimitMax, forKey: .timeStepLimitMax)
            } else { try container.encodeNil(forKey: .timeStepLimitMax) }
        } else {
            try container.encodeNil(forKey: .timeStepLimitMin)
            try container.encodeNil(forKey: .timeStepLimitMax)
            try container.encode(target.timeStepLimitMin, forKey: .timeStepMin)
            if target.timeStepLimitMax.isFinite {
                try container.encode(target.timeStepLimitMax, forKey: .timeStepMax)
            } else { try container.encodeNil(forKey: .timeStepMax) }
        }
        let blockTypes = target.hybridOverridePattern.map { blockType in
            switch blockType {
            case "M": return "mamba"
            case "*": return "attention"
            case "-": return "mlp"
            case "E": return "moe"
            default: preconditionFailure("validated Nemotron block type changed")
            }
        }
        try container.encode(blockTypes, forKey: .layersBlockType)
    }
}

public final class NemotronH35Model: NemotronHModel {
    public let lightningConfiguration: NemotronH35Configuration

    public override var cbv2Capabilities: CBv2ModelCapabilities {
        var capabilities = super.cbv2Capabilities
        capabilities.supportsRecurrentCheckpointReuse = true
        capabilities.supportsPagedKV = true
        capabilities.requiresNativePagedKV = true
        capabilities.supportsMTP = configuration.numNextnPredictLayers == 1
            && configuration.mtpLayersBlockType == ["attention", "moe"]
        return capabilities
    }

    public init(_ args: NemotronH35Configuration) {
        self.lightningConfiguration = args
        // mlx-lm 0.31.3 parses time_step_min/time_step_max as metadata but
        // initializes the NemotronH mixer from the absent time_step_limit
        // tuple, whose reference default is (0, +infinity). Applying the
        // copied HF min/max here changes greedy output ("NEMOTRON..." becomes
        // "The count: 1" on the audited checkpoint). Preserve the decoded
        // values above for provenance, but execute the selected MLX artifact
        // with its independent serial oracle's actual contract.
        var executionTarget = args.target
        if !args.hasExplicitExecutionLimits {
            executionTarget.timeStepLimitMin = 0
            executionTarget.timeStepLimitMax = .infinity
        }
        super.init(executionTarget)
    }
}

extension NemotronH35Model: CBv2CompleteCheckpointKVTypeProviding {
    public var cbv2CompleteCheckpointKVDTypes: [DType]? {
        observedAttentionKVDTypes()
    }
}
