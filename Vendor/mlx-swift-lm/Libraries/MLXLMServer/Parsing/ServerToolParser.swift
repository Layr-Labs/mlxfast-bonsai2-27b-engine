// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

public enum ServerToolParserError: Error, LocalizedError, Equatable {
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let name):
            return "Unsupported tool_call_parser '\(name)'"
        }
    }
}

public enum ServerToolParser {
    public static func resolve(requested: String?, modelType: String?) throws -> ToolCallFormat {
        let normalized = requested?.lowercased().replacingOccurrences(of: "-", with: "_")

        if normalized == nil || normalized == "auto" {
            if let modelType, let inferred = ToolCallFormat.infer(from: modelType) {
                return inferred
            }
            return .json
        }

        switch normalized {
        case "json", "default", "qwen3":
            return .json
        case "lfm2", "lfm2_5", "lfm25":
            return .lfm2
        case "xml", "xml_function", "qwen_xml", "qwen3_coder",
            "hermes":
            return .xmlFunction
        case "nemotron":
            return .nemotron
        // Qwen 3.5 gets its dual-dialect parser (XML first, framed
        // Hermes-JSON fallback), not the pure XML one: the model sporadically
        // emits its older JSON dialect inside the same <tool_call> frame.
        case "qwen3_5", "qwen35":
            return .qwen35
        case "glm4", "glm_4":
            return .glm4
        case "gemma", "gemma4", "gemma_4":
            return .gemma
        case "kimi_k2", "kimi":
            return .kimiK2
        case "minimax_m2", "minimax":
            return .minimaxM2
        case "mistral", "mistral_v11":
            return .mistral
        case "llama3", "llama3_json", "llama_3":
            return .llama3
        case "harmony", "gpt_oss", "openai_harmony":
            return .harmony
        case .some(let name):
            throw ServerToolParserError.unsupported(name)
        case .none:
            return .json
        }
    }
}
