// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Qwen3-VL runner, dense and MoE.
//
// The loaded WRAPPER is the serving model: unlike the Qwen 3.5 VLM, this one
// is CBv2-adapted directly, so there is no text-tower extraction and no
// second module over the same weights. It is a full-attention decoder with
// M-RoPE positions, and its declared capabilities stay conservative until
// paged M-RoPE, prefix identity, compiled decode, packed prefill and MTP
// each have their own evidence — the model file says so and the manifest
// repeats it rather than assuming a default.

import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers

public final class Qwen3VLRunner: Runner, @unchecked Sendable {

    public static let manifest = RunnerManifest(
        runnerID: "layr/qwen3vl",
        modelTypes: ["qwen3_vl", "qwen3_vl_moe"],
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: false,
            supportsPagedKV: false,
            supportsCompiledDecode: false,
            supportsPackedPrefill: false,
            supportsMTP: false,
            supportsCompactRecurrentMTPReplay: false),
        kvBackends: [.contiguous],
        decoders: [
            DecoderDeclaration(
                mode: DecoderID.serial.rawValue, drafter: .none, state: .stateless,
                depth: nil)
        ],
        regimes: [
            RegimeDeclaration(batch: .single, timing: .freeRun, perStreamTiming: false),
            RegimeDeclaration(batch: .single, timing: .teacherForced, perStreamTiming: false),
        ],
        multimodal: true,
        recurrentLayers: false,
        requiresKeepMask: false)

    public let servingModel: any LanguageModel
    public let tokenizer: any MLXLMCommon.Tokenizer
    public let eosTokenIDs: Set<Int>
    public let layerKinds: [CBv2LayerKind]
    public let loadedDecoders: [DecoderID] = [.serial]
    public let headProvenance: HeadProvenance? = nil
    public let loadedModelType: String

    private let model: Qwen3VL
    private let kvBytesCapacity: Int
    private let maxSequenceLength: Int

    private init(
        model: Qwen3VL,
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>,
        loadedModelType: String,
        kvBytesCapacity: Int,
        maxSequenceLength: Int
    ) {
        self.model = model
        self.servingModel = model
        self.tokenizer = tokenizer
        self.eosTokenIDs = eosTokenIDs
        self.layerKinds = model.cbv2LayerKinds
        self.loadedModelType = loadedModelType
        self.kvBytesCapacity = kvBytesCapacity
        self.maxSequenceLength = maxSequenceLength
    }

    public static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> Qwen3VLRunner {
        // Checkpoint facts FIRST, module second: these two reads are the
        // whole of this method's filesystem access.
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let eosTokenIDs = RunnerCheckpoint.eosTokenIDs(
            at: directory, tokenizer: tokenizer)
        // No tower extraction: this family is CBv2-adapted directly, so the
        // loaded wrapper IS the serving model.
        // A drafter this family cannot use is refused BEFORE the module is
        // examined: accepting one and ignoring it would advertise serial
        // while the caller believes it handed over speculation.
        guard options.drafterDirectory == nil, options.preloadedDrafter == nil else {
            throw RunnerError.drafterUnavailable(
                "qwen3_vl declares no speculative decoder")
        }
        guard let model = model as? Qwen3VL else {
            throw RunnerError.unexpectedModel(String(describing: type(of: model)))
        }
        return Qwen3VLRunner(
            model: model,
            tokenizer: tokenizer,
            eosTokenIDs: eosTokenIDs,
            loadedModelType: modelType,
            kvBytesCapacity: options.kvBytesCapacity,
            maxSequenceLength: options.maxSequenceLength)
    }

    public func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        let model = self.model
        return try RunnerEngineAssembly.makeEngine(
            manifest: Self.manifest,
            loadedDecoders: loadedDecoders,
            model: model,
            tokenizer: tokenizer,
            layerKinds: layerKinds,
            newCaches: { make in try model.newCacheV2(makeLayerCache: make) },
            mtpDrafter: nil,
            build: build)
    }

    public func makeStepper() throws -> any TeacherForcedStepper {
        let model = self.model
        return CBv2SingleRowStepper(
            model: model,
            layerKinds: layerKinds,
            newCaches: { make in try model.newCacheV2(makeLayerCache: make) },
            kvBytesCapacity: kvBytesCapacity,
            maxLength: maxSequenceLength)
    }
}
