// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Gemma 4 text runner.
//
// Serves the `Gemma4TextModel` tower: the LLM checkpoint directly, or the
// `Gemma4Model` wrapper's own text model (the wrapper OWNS that instance, so
// nothing here constructs a second language module or duplicates weights).
//
// The family's layer-kind derivation moved out of the engine directory and
// into `Gemma4Text.swift` next to the constructors it mirrors; this runner
// reads it off the model, never re-derives it.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

public final class Gemma4TextRunner: Runner, @unchecked Sendable {

    public static let manifest = RunnerManifest(
        runnerID: "layr/gemma4-text",
        modelTypes: ["gemma4", "gemma4_text"],
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: true,
            supportsPagedKV: true,
            supportsCompiledDecode: true,
            supportsPackedPrefill: true,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: false),
        kvBackends: [.contiguous, .paged],
        decoders: [
            DecoderDeclaration(
                mode: DecoderID.serial.rawValue, drafter: .none, state: .stateless,
                depth: nil),
            // The Gemma 4 drafter is a separate assistant checkpoint bound to
            // the loaded target (`Gemma4CBv2MTPDrafter`), stateless across
            // rounds. Depths are the engine's production-tested envelope.
            DecoderDeclaration(
                mode: DecoderID.mtp.rawValue, drafter: .assistantCheckpoint,
                state: .stateless,
                depth: 1 ... CBv2MTPConfig.testedMaxDraftTokens),
        ],
        regimes: [
            RegimeDeclaration(batch: .single, timing: .freeRun, perStreamTiming: false),
            RegimeDeclaration(
                batch: .upTo(CBv2MTPConfig.testedMaxSpeculativeBatch), timing: .freeRun,
                perStreamTiming: false),
            RegimeDeclaration(batch: .single, timing: .teacherForced, perStreamTiming: false),
        ],
        multimodal: false,
        recurrentLayers: false,
        requiresKeepMask: false)

    public let servingModel: any LanguageModel
    public let tokenizer: any MLXLMCommon.Tokenizer
    public let eosTokenIDs: Set<Int>
    public let layerKinds: [CBv2LayerKind]
    public let loadedDecoders: [DecoderID]
    public let headProvenance: HeadProvenance?
    public let loadedModelType: String

    private let textModel: Gemma4TextModel
    private let drafter: (any CBv2MTPDrafter)?
    private let kvBytesCapacity: Int
    private let maxSequenceLength: Int

    private init(
        textModel: Gemma4TextModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>,
        loadedModelType: String,
        drafter: (any CBv2MTPDrafter)?,
        headProvenance: HeadProvenance?,
        kvBytesCapacity: Int,
        maxSequenceLength: Int
    ) {
        self.textModel = textModel
        self.servingModel = textModel
        self.tokenizer = tokenizer
        self.eosTokenIDs = eosTokenIDs
        self.layerKinds = textModel.cbv2LayerKinds
        self.loadedModelType = loadedModelType
        self.drafter = drafter
        self.headProvenance = headProvenance
        self.kvBytesCapacity = kvBytesCapacity
        self.maxSequenceLength = maxSequenceLength
        self.loadedDecoders = drafter == nil ? [.serial] : [.serial, .mtp]
    }

    /// The tower CBv2 serves.
    ///
    /// Every shape the factories produce for this family resolves here, the
    /// MULTIMODAL wrapper included: Darkbloom's container loads a Gemma 4
    /// VLM checkpoint as `MLXVLM.Gemma4`, and refusing that would make the
    /// runner unusable for exactly the deployment the contract's adopt seam
    /// exists for. No extraction is needed for this family — each wrapper
    /// directly OWNS the `Gemma4TextModel` instance, so this is a property
    /// read and nothing here constructs a second language module or
    /// duplicates the checkpoint's resident weights.
    static func textTower(of model: any LanguageModel) throws -> Gemma4TextModel {
        switch model {
        case let text as Gemma4TextModel: return text
        case let wrapper as MLXLLM.Gemma4Model: return wrapper.textModel
        case let wrapper as MLXVLM.Gemma4: return wrapper.textModel
        default:
            throw RunnerError.unexpectedModel(String(describing: type(of: model)))
        }
    }

    public static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> Gemma4TextRunner {
        // Checkpoint facts FIRST, module second: these two reads are the
        // whole of this method's filesystem access, and taking them before
        // the type check is what lets a test prove it by handing `adopt` a
        // directory holding nothing but `config.json` and the index.
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let eosTokenIDs = RunnerCheckpoint.eosTokenIDs(
            at: directory, tokenizer: tokenizer)
        let textModel = try Self.textTower(of: model)

        // Provenance describes the artifact on DISK, so it is read only when
        // the caller named one. A resident drafter handed in by Darkbloom
        // carries no directory to hash.
        var provenance: HeadProvenance?
        if options.preloadedDrafter != nil, let drafterDirectory = options.drafterDirectory {
            provenance = try? RunnerCheckpoint.provenance(ofHeadAt: drafterDirectory)
        }

        return Gemma4TextRunner(
            textModel: textModel,
            tokenizer: tokenizer,
            eosTokenIDs: eosTokenIDs,
            loadedModelType: modelType,
            drafter: options.preloadedDrafter,
            headProvenance: provenance,
            kvBytesCapacity: options.kvBytesCapacity,
            maxSequenceLength: options.maxSequenceLength)
    }

    /// The Gemma 4 drafter is a separate assistant checkpoint. Reading it is
    /// why this is not part of `adopt`.
    public static func loadDrafter(
        options: RunnerLoadOptions,
        directory: URL,
        target: any LanguageModel
    ) async throws -> (any CBv2MTPDrafter)? {
        if let preloaded = options.preloadedDrafter { return preloaded }
        guard let drafterDirectory = options.drafterDirectory else { return nil }
        do {
            let assistant = try await Gemma4AssistantDraftModel.load(
                from: drafterDirectory)
            return try Gemma4CBv2MTPDrafter(
                drafter: assistant, target: try Self.textTower(of: target))
        } catch {
            // A drafter directory that was ASKED for and cannot load is a
            // refusal, not a silent demotion to serial: an mtp leg that
            // quietly measured serial would look like a very fast serial
            // engine.
            throw RunnerError.drafterUnavailable("\(error)")
        }
    }

    public func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        let model = textModel
        return try RunnerEngineAssembly.makeEngine(
            manifest: Self.manifest,
            loadedDecoders: loadedDecoders,
            model: model,
            tokenizer: tokenizer,
            layerKinds: layerKinds,
            newCaches: { make in try model.newCacheV2(makeLayerCache: make) },
            mtpDrafter: drafter,
            build: build)
    }

    public func makeStepper() throws -> any TeacherForcedStepper {
        let model = textModel
        return CBv2SingleRowStepper(
            model: model,
            layerKinds: layerKinds,
            newCaches: { make in try model.newCacheV2(makeLayerCache: make) },
            kvBytesCapacity: kvBytesCapacity,
            maxLength: maxSequenceLength)
    }
}
