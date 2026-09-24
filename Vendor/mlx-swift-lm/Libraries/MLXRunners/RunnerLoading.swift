// Copyright © 2026 Eigen Labs.
//
// MLXRunners — the default `Runner.load`, in terms of `adopt`.
//
// ONE construction path. `load` brings the module into memory and then
// adopts it; Darkbloom, whose slot lifecycle already owns a resident
// `ModelContainer`, calls `adopt` directly. Anything a runner derives at
// construction is therefore derived in exactly one place, and the in-process
// consumer cannot drift from the spawned one.
//
// Kept out of `Runner.swift` so the protocol file needs no factory imports.

import Foundation
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// The HF tokenizer loader, hoisted out of the generic extension below.
///
/// `#huggingFaceTokenizerLoader()` expands to a closure declaring a nested
/// struct, and a nested type is not allowed inside a generic context — which
/// a protocol extension is. Naming it once here also means one loader
/// instance for every runner.
private enum RunnerTokenizerLoader {
    static let shared: any TokenizerLoader = #huggingFaceTokenizerLoader()
}

extension Runner {

    /// Families with no drafter, and callers that already hold one.
    public static func loadDrafter(
        options: RunnerLoadOptions,
        directory: URL,
        target: any LanguageModel
    ) async throws -> (any CBv2MTPDrafter)? {
        options.preloadedDrafter
    }

    /// Load the container, then adopt it.
    ///
    /// The factory follows the manifest: a runner that declares
    /// `multimodal` serves a checkpoint whose `config.json` only the VLM
    /// factory understands. That is a reading of the runner's own
    /// declaration rather than a second family switch.
    public static func load(
        _ directory: URL, options: RunnerLoadOptions
    ) async throws -> Self {
        let context: ModelContext
        if manifest.multimodal {
            context = try await VLMModelFactory.shared.load(
                from: directory, using: RunnerTokenizerLoader.shared)
        } else {
            context = try await LLMModelFactory.shared.load(
                from: directory, using: RunnerTokenizerLoader.shared)
        }

        // The drafter is loaded HERE, not in `adopt`, because it reads
        // tensors. Once loaded it enters `adopt` the same way a caller's
        // resident drafter does, so both paths bind it identically.
        var options = options
        options.preloadedDrafter = try await loadDrafter(
            options: options, directory: directory, target: context.model)

        return try adopt(
            model: context.model,
            tokenizer: context.tokenizer,
            configuration: context.configuration,
            directory: directory,
            options: options)
    }
}
