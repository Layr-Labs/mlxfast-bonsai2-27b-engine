// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Qwen 3.5 runner, dense and MoE.
//
// A HYBRID trunk: only every `full_attention_interval`-th layer owns KV, and
// the rest are gated-delta-net recurrent layers carried as request-owned
// recurrent state. `cbv2LayerKinds` is therefore the COMPACT attention
// storage layout with `modelLayerIndex` mapping each stored row back to its
// transformer layer — which is why paged is not declared: the pool's dense
// storage subscript does not survive a hybrid trunk (`supportsPagedKV` is
// false on the model too).
//
// TWO SPECULATIVE DECODERS, one `--drafter` directory. `mtp` is a
// `Qwen35InlineMTPAssistant`, which drafts a CHAIN one token at a time.
// `dflash` is a `Qwen35DFlash2Assistant`, which proposes a whole BLOCK in one
// forward. Both are request-stateful across rounds and both borrow the trunk's
// `embed_tokens` and `lm_head`; in this pack both of those are packed Hadamard
// modules, so an assistant must call them as MODULES and never read their raw
// `.weight`.
//
// WHICH ONE LOADS IS READ FROM THE DIRECTORY'S OWN `config.json`, never from
// its name: a DFlash 2 export declares `DFlash2DraftModel` in `architectures`
// and carries a `dflash_config`, and an MTP head declares
// `model_type: qwen3_5_mtp`. The runner then advertises ONLY the mode whose
// drafter actually loaded (§6.2 rule 1), so a worker can never resolve a mode
// it did not advertise.
//
// THIS TRACK USES A SEPARATE HEAD. The Ternary Bonsai 2 27B pack declares
// `mtp_num_hidden_layers: 0` and ships no `mtp.*` tensors, so the MTP head is
// a published standalone export and the caller names its directory with
// `--drafter`. A checkpoint that carries its own `mtp.*` block still works:
// the drafter directory defaults to the model directory.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

public final class Qwen35Runner: Runner, @unchecked Sendable {

    public static let manifest = RunnerManifest(
        runnerID: "layr/qwen35",
        modelTypes: [
            "prism_hadamard_qwen35", "qwen3_5", "qwen3_5_moe", "qwen3_5_text",
        ],
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: false,
            supportsPagedKV: false,
            supportsCompiledDecode: false,
            supportsPackedPrefill: true,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: true),
        kvBackends: [.contiguous],
        decoders: [
            DecoderDeclaration(
                mode: DecoderID.serial.rawValue, drafter: .none, state: .stateless,
                depth: nil),
            // This track's head is a separate published export, not a block
            // of the target checkpoint.
            DecoderDeclaration(
                mode: DecoderID.mtp.rawValue, drafter: .assistantCheckpoint,
                state: .requestStateful,
                depth: 1 ... CBv2MTPConfig.testedMaxDraftTokens),
            // DFlash 2. A BLOCK drafter reaches deeper than a chain for the
            // same one drafter forward, so its depth range is its own.
            DecoderDeclaration(
                mode: DecoderID.dflash.rawValue, drafter: .assistantCheckpoint,
                state: .requestStateful,
                depth: 1 ... CBv2MTPConfig.testedMaxBlockDraftTokens),
        ],
        regimes: [
            RegimeDeclaration(batch: .single, timing: .freeRun, perStreamTiming: false),
            RegimeDeclaration(
                batch: .upTo(CBv2MTPConfig.testedMaxSpeculativeBatch), timing: .freeRun,
                perStreamTiming: false),
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

    /// Captured at load so `makeEngine`/`makeStepper` never re-switch on the
    /// concrete model type: the dense/MoE `Qwen35Model` and the text-only
    /// `Qwen35TextModel` expose the same hooks under different types.
    private let newCaches:
        (
            (_ layerIndex: Int, _ kind: CBv2LayerKind) throws -> any CBv2AttendingLayerCache
        ) throws -> [any CBv2AttendingLayerCache]
    private let drafter: (any CBv2MTPDrafter)?
    private let kvBytesCapacity: Int
    private let maxSequenceLength: Int

    private init(
        servingModel: any LanguageModel,
        layerKinds: [CBv2LayerKind],
        newCaches: @escaping (
            (_ layerIndex: Int, _ kind: CBv2LayerKind) throws -> any CBv2AttendingLayerCache
        ) throws -> [any CBv2AttendingLayerCache],
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>,
        loadedModelType: String,
        drafter: (any CBv2MTPDrafter)?,
        headProvenance: HeadProvenance?,
        kvBytesCapacity: Int,
        maxSequenceLength: Int
    ) {
        self.servingModel = servingModel
        self.layerKinds = layerKinds
        self.newCaches = newCaches
        self.tokenizer = tokenizer
        self.eosTokenIDs = eosTokenIDs
        self.loadedModelType = loadedModelType
        self.drafter = drafter
        self.headProvenance = headProvenance
        self.kvBytesCapacity = kvBytesCapacity
        self.maxSequenceLength = maxSequenceLength
        self.loadedDecoders =
            [.serial] + (Self.decoder(of: drafter).map { [$0] } ?? [])
    }

    /// The decoder a loaded drafter resolves, or nil for no drafter. A block
    /// drafter is `dflash` and everything else this runner loads is `mtp`.
    static func decoder(of drafter: (any CBv2MTPDrafter)?) -> DecoderID? {
        guard let drafter else { return nil }
        return drafter is any CBv2MTPBlockDrafter ? .dflash : .mtp
    }

    /// The dense/MoE `Qwen35Model`, the text-only `Qwen35TextModel` and the
    /// MULTIMODAL `MLXVLM.Qwen35` wrapper all resolve here. Resolved ONCE so
    /// `makeEngine` and `makeStepper` never re-switch.
    ///
    /// Unlike Gemma 4, the VLM wrapper does not own an MLXLLM target: it
    /// carries its own inline text model, so CBv2 needs a separate
    /// `Qwen35Model` built over the SAME immutable weight arrays. That is
    /// `QwenVLMTextExtraction`, ported here from the provider — it shares
    /// arrays, copies nothing, and reads no tensor from disk.
    static func hooks(
        of model: any LanguageModel,
        directory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> (
        serving: any LanguageModel,
        layerKinds: [CBv2LayerKind],
        newCaches: (
            (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
                any CBv2AttendingLayerCache
        ) throws -> [any CBv2AttendingLayerCache]
    ) {
        switch model {
        case let model as Qwen35Model:
            // Covers `Qwen35MoEModel`, which subclasses it.
            return (
                model, model.cbv2LayerKinds,
                { make in try model.newCacheV2(makeLayerCache: make) }
            )
        case let model as Qwen35TextModel:
            return (
                model, model.cbv2LayerKinds,
                { make in try model.newCacheV2(makeLayerCache: make) }
            )
        case is MLXVLM.Qwen35:
            // Extraction is memoized per WRAPPER INSTANCE, so the target the
            // engine steps and the target a drafter binds to are the same
            // object — MTP gates on that identity.
            let target = try QwenVLMTextExtraction.target(
                for: model, directory: directory, environment: environment)
            return (
                target, target.cbv2LayerKinds,
                { make in try target.newCacheV2(makeLayerCache: make) }
            )
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
    ) throws -> Qwen35Runner {
        // Checkpoint facts FIRST, module second: these two reads are the
        // whole of this method's filesystem access.
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let eosTokenIDs = RunnerCheckpoint.eosTokenIDs(
            at: directory, tokenizer: tokenizer)
        let hooks = try Self.hooks(
            of: model, directory: directory, environment: options.environment)

        // FAIL CLOSED. The provenance is sealed into the hello, so a head
        // that cannot be hashed must stop the adoption.
        //
        // A SEPARATE export is hashed whole: every safetensors shard in the
        // drafter directory. That is this track's shape, and the helper
        // throws rather than return nil when the directory holds no shard.
        //
        // An EMBEDDED head is the checkpoint's own `mtp.*` block (§12c: the
        // one rule, in the one shared helper). A checkpoint that declares
        // `mtp.*` tensors but whose index cannot be read is broken, and
        // swallowing that error would serve an UNATTRIBUTED head. Only a
        // checkpoint with no head at all gives nil.
        var provenance: HeadProvenance?
        if options.preloadedDrafter != nil {
            if let drafterDirectory = options.drafterDirectory,
                drafterDirectory.standardizedFileURL != directory.standardizedFileURL
            {
                provenance = try RunnerCheckpoint.provenance(ofHeadAt: drafterDirectory)
            } else {
                provenance = try RunnerCheckpoint.provenance(ofEmbeddedHeadAt: directory)
            }
        }

        return Qwen35Runner(
            servingModel: hooks.serving,
            layerKinds: hooks.layerKinds,
            newCaches: hooks.newCaches,
            tokenizer: tokenizer,
            eosTokenIDs: eosTokenIDs,
            loadedModelType: modelType,
            drafter: options.preloadedDrafter,
            headProvenance: provenance,
            kvBytesCapacity: options.kvBytesCapacity,
            maxSequenceLength: options.maxSequenceLength)
    }

    /// This track names its drafter with `--drafter`. The directory still
    /// DEFAULTS to the model directory, which is where a checkpoint that
    /// carries its own head keeps it. Reading it is why this is not part of
    /// `adopt`.
    ///
    /// The DIRECTORY DECIDES which decoder loads, and it decides by what its
    /// `config.json` declares. A name is not evidence: two exports of the same
    /// pair differ only in their contents, and a runner that guessed from a
    /// path would load a chain head for a block request and advertise the
    /// wrong mode.
    public static func loadDrafter(
        options: RunnerLoadOptions,
        directory: URL,
        target: any LanguageModel
    ) async throws -> (any CBv2MTPDrafter)? {
        if let preloaded = options.preloadedDrafter { return preloaded }
        let drafterDirectory = options.drafterDirectory ?? directory
        let serving = try Self.hooks(
            of: target, directory: directory, environment: options.environment
        ).serving
        if Qwen35DFlash2Assistant.isDrafterDirectory(drafterDirectory) {
            // A directory that DECLARES itself a DFlash 2 drafter and then
            // fails to load is a refusal whichever way it was named: nothing
            // else in the tree can serve that declaration.
            do {
                return try Qwen35DFlash2Assistant.load(
                    from: drafterDirectory, target: serving)
            } catch {
                throw RunnerError.drafterUnavailable("\(error)")
            }
        }
        do {
            return try Qwen35InlineMTPAssistant.load(
                from: drafterDirectory, target: serving)
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
