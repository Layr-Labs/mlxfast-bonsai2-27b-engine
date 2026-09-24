// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Nemotron 3.5 Lightning (`nemotron_h` with `layers_block_type`).
//
// A HYBRID trunk: the checkpoint's block pattern mixes Mamba2 recurrence,
// full attention, MLP and MoE layers, and only the ATTENTION blocks own a
// key-value tape. `cbv2LayerKinds` is therefore the compact attention storage
// layout with `modelLayerIndex` mapping each stored row back to its decoder
// layer; the Mamba blocks are request-owned recurrent state.
//
// `model_type` alone does not name this family. The legacy Nemotron Nano
// checkpoints share the string and are served by `NemotronHModel`; the
// Lightning contract is the one whose `config.json` carries
// `layers_block_type`, which is exactly what `NemotronH35Configuration`
// requires and what `LLMTypeRegistry` keys on when it builds
// `NemotronH35Model`. Adoption refuses anything else by name.
//
// Speculation is the checkpoint's own `mtp.*` block, loaded by
// `NemotronH35MTPAssistant.load(from:target:)` and request-stateful across
// rounds. The assistant carries its own NATIVE paged row for the draft chain,
// proposes at most `CBv2MTPConfig.testedMaxDraftTokens` tokens (its own
// default `draftLimit`), and declares `maximumSpeculativeBatch == 1`, so only
// single-stream regimes are declared here.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

public final class NemotronH35Runner: Runner, @unchecked Sendable {

    public static let manifest = RunnerManifest(
        runnerID: "layr/nemotron35-lightning",
        modelTypes: ["nemotron_h"],
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: false,
            supportsPagedKV: true,
            supportsCompiledDecode: false,
            supportsPackedPrefill: false,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: false),
        // CONTIGUOUS ONLY, although the model supports paged KV. The model
        // also sets `requiresNativePagedKV`, and `EngineV2` then demands a
        // `PagedKVBackend` whose pool declares both `segmentSizeBytes` and
        // the observed native `layerDTypes`. `RunnerEngineAssembly`'s paged
        // branch builds neither, so declaring `.paged` here would reach
        // `EngineV2.backendCapabilityViolation` and `preconditionFailure`
        // the process rather than refuse a build.
        kvBackends: [.contiguous],
        decoders: [
            DecoderDeclaration(
                mode: DecoderID.serial.rawValue, drafter: .none, state: .stateless,
                depth: nil),
            DecoderDeclaration(
                mode: DecoderID.mtp.rawValue, drafter: .embeddedHead,
                state: .requestStateful,
                depth: 1 ... CBv2MTPConfig.testedMaxDraftTokens),
        ],
        regimes: [
            RegimeDeclaration(batch: .single, timing: .freeRun, perStreamTiming: false),
            RegimeDeclaration(batch: .single, timing: .teacherForced, perStreamTiming: false),
        ],
        multimodal: false,
        recurrentLayers: true,
        requiresKeepMask: false)

    public let servingModel: any LanguageModel
    public let tokenizer: any MLXLMCommon.Tokenizer
    public let eosTokenIDs: Set<Int>
    public let layerKinds: [CBv2LayerKind]
    public let loadedDecoders: [DecoderID]
    public let headProvenance: HeadProvenance?
    public let loadedModelType: String

    private let model: NemotronH35Model
    private let drafter: (any CBv2MTPDrafter)?
    private let kvBytesCapacity: Int
    private let maxSequenceLength: Int

    private init(
        model: NemotronH35Model,
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>,
        loadedModelType: String,
        drafter: (any CBv2MTPDrafter)?,
        headProvenance: HeadProvenance?,
        kvBytesCapacity: Int,
        maxSequenceLength: Int
    ) {
        self.model = model
        self.servingModel = model
        self.layerKinds = model.cbv2LayerKinds
        self.tokenizer = tokenizer
        self.eosTokenIDs = eosTokenIDs
        self.loadedModelType = loadedModelType
        self.drafter = drafter
        self.headProvenance = headProvenance
        self.kvBytesCapacity = kvBytesCapacity
        self.maxSequenceLength = maxSequenceLength
        self.loadedDecoders = drafter == nil ? [.serial] : [.serial, .mtp]
    }

    public static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> NemotronH35Runner {
        // Checkpoint facts FIRST, module second: these two reads are the
        // whole of this method's filesystem access, bar the head's own
        // shards when there is a head to attribute.
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let eosTokenIDs = RunnerCheckpoint.eosTokenIDs(
            at: directory, tokenizer: tokenizer)

        guard let model = model as? NemotronH35Model else {
            throw RunnerError.unexpectedModel(
                "\(type(of: model)) — a legacy nemotron_h checkpoint without "
                    + "layers_block_type is served by NemotronHModel, not by "
                    + "this runner")
        }

        // The head ships INSIDE the checkpoint, so the provenance of a head
        // loaded from there is the checkpoint's own directory. §12c: the one
        // embedded-head rule, in the one shared helper.
        //
        // FAIL CLOSED. The provenance is sealed into the hello, so a head
        // that cannot be hashed must stop the adoption; swallowing that error
        // would serve an UNATTRIBUTED head. Only a checkpoint with no head at
        // all gives nil, which the helper returns without throwing.
        var provenance: HeadProvenance?
        if options.preloadedDrafter != nil {
            provenance = try RunnerCheckpoint.provenance(
                ofEmbeddedHeadAt: options.drafterDirectory ?? directory)
        }

        return NemotronH35Runner(
            model: model,
            tokenizer: tokenizer,
            eosTokenIDs: eosTokenIDs,
            loadedModelType: modelType,
            drafter: options.preloadedDrafter,
            headProvenance: provenance,
            kvBytesCapacity: options.kvBytesCapacity,
            maxSequenceLength: options.maxSequenceLength)
    }

    /// The head ships inside the checkpoint, so the drafter directory
    /// DEFAULTS to the model directory. Reading it is why this is not part
    /// of `adopt`.
    public static func loadDrafter(
        options: RunnerLoadOptions,
        directory: URL,
        target: any LanguageModel
    ) async throws -> (any CBv2MTPDrafter)? {
        if let preloaded = options.preloadedDrafter { return preloaded }
        guard let target = target as? NemotronH35Model else {
            throw RunnerError.unexpectedModel(
                "\(type(of: target)) — the Nemotron 3.5 Lightning mtp head "
                    + "binds to NemotronH35Model")
        }
        do {
            return try NemotronH35MTPAssistant.load(
                from: options.drafterDirectory ?? directory, target: target)
        } catch {
            // An EXPLICIT drafter directory that fails is a refusal; the
            // implicit in-checkpoint head simply may not be there, and a
            // checkpoint without an `mtp.*` block is a serial-only model,
            // not a broken one.
            if options.drafterDirectory != nil {
                throw RunnerError.drafterUnavailable("\(error)")
            }
            return nil
        }
    }

    /// The family half of the `--verbose` load summary.
    ///
    /// The block pattern leads: this family's `model_type` is shared with the
    /// legacy Nemotron Nano contract, and the counts are what say WHICH
    /// checkpoint was bound and how many of its layers own a key-value tape.
    /// The MTP block's own declaration follows, because a head that loaded
    /// narrows both the depth and the verification mode the engine may use.
    ///
    /// Reads the resident configuration and resident tensors only; no file is
    /// reopened.
    public func loadSummary(
        weights: URL, options: RunnerLoadOptions
    ) -> RunnerLoadSummary {
        var summary = genericLoadSummary(weights: weights, options: options)
        // The base `configuration` is internal to MLXLLM; the Lightning
        // boundary exposes the same decoded values.
        let text = model.lightningConfiguration.target

        var counts: [Character: Int] = [:]
        for block in text.hybridOverridePattern { counts[block, default: 0] += 1 }
        summary.add("layers_block_type_attention", counts["*"] ?? 0)
        summary.add("layers_block_type_mamba", counts["M"] ?? 0)
        summary.add("layers_block_type_mlp", counts["-"] ?? 0)
        summary.add("layers_block_type_moe", counts["E"] ?? 0)

        summary.add("num_nextn_predict_layers", text.numNextnPredictLayers)
        summary.add(
            "mtp_layers_block_type",
            text.mtpLayersBlockType.isEmpty
                ? "none" : text.mtpLayersBlockType.joined(separator: ","))

        // What the loaded projections' attention tapes READ AS, in attention
        // storage order — the table a native paged pool would have to be
        // built over. "none" means the probe did not agree with the storage
        // layout.
        summary.add(
            "checkpoint_kv_dtypes",
            model.cbv2CompleteCheckpointKVDTypes
                .map { dtypes in
                    dtypes.map { "\($0)" }.joined(separator: ",")
                } ?? "none")

        if let drafter {
            summary.add(
                "mtp_max_draft_tokens",
                drafter.maximumDraftTokens.map { String($0) } ?? "unbounded")
            summary.add(
                "mtp_verification_mode",
                drafter.requiredVerificationMode?.rawValue ?? "engine_default")
        }
        return summary
    }

    /// The model's own cache vending. `newCacheV2` hands the closure the
    /// MODEL layer index, which on this hybrid trunk is not the storage
    /// position.
    private func newCaches(
        _ make: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) throws -> [any CBv2AttendingLayerCache] {
        try model.newCacheV2(makeLayerCache: make)
    }

    public func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        try RunnerEngineAssembly.makeEngine(
            manifest: Self.manifest,
            loadedDecoders: loadedDecoders,
            model: servingModel,
            tokenizer: tokenizer,
            layerKinds: layerKinds,
            newCaches: newCaches,
            mtpDrafter: drafter,
            build: build)
    }

    public func makeStepper() throws -> any TeacherForcedStepper {
        CBv2SingleRowStepper(
            model: servingModel,
            layerKinds: layerKinds,
            newCaches: newCaches,
            kvBytesCapacity: kvBytesCapacity,
            maxLength: maxSequenceLength)
    }
}
