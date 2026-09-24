// Qwen4ExpRunnerTests.swift
//
// The Qwen 3.8 Flash-Next runner's declaration, as far as it can be proved
// without weights: the registry claim, and the hello table the worker
// derives from the manifest (contract §6.1). The canonical manifest bytes and
// the cross-repo digest are pinned next to the other runners in
// `RunnerManifestTests`.
//
// Model-free by construction. The behaviour that needs a forward pass —
// stepper against engine, and the paged refusal — lives in
// `Tests/MLXLMTests/Qwen4ExpRunnerEngineTests.swift`, beside the tiny model
// fixture it drives.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXRunners

/// A runner that carries the real Qwen 3.8 Flash-Next manifest and a
/// scripted loaded state, so the hello derivation can be read off the real
/// declaration with no checkpoint present. It holds no model: `hello()`
/// never asks for one.
private final class Qwen4ExpManifestRunner: Runner, @unchecked Sendable {
    static let manifest = Qwen4ExpRunner.manifest

    let loadedDecoders: [DecoderID]

    init(loadedDecoders: [DecoderID]) {
        self.loadedDecoders = loadedDecoders
    }

    var servingModel: any LanguageModel {
        preconditionFailure("manifest-only runner holds no model")
    }
    var tokenizer: any MLXLMCommon.Tokenizer {
        preconditionFailure("manifest-only runner holds no tokenizer")
    }
    let eosTokenIDs: Set<Int> = []
    let layerKinds: [CBv2LayerKind] = []
    let headProvenance: HeadProvenance? = nil
    let loadedModelType = "qwen4_exp"

    static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> Qwen4ExpManifestRunner {
        Qwen4ExpManifestRunner(loadedDecoders: [.serial, .mtp])
    }

    func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        preconditionFailure("manifest-only runner builds no engine")
    }
    func makeStepper() throws -> any TeacherForcedStepper {
        preconditionFailure("manifest-only runner builds no stepper")
    }
}

@Suite("Qwen 3.8 Flash-Next runner")
struct Qwen4ExpRunnerTests {

    private func hello(loadedDecoders: [DecoderID]) -> WorkerResponse {
        BenchWorkerServer(
            runner: Qwen4ExpManifestRunner(loadedDecoders: loadedDecoders),
            transport: ScriptedTransport(lines: []),
            trusted: false,
            build: "fixture",
            device: "fixture",
            kvBytesCapacity: 0,
            memory: FixtureMemoryReporter(),
            nonce: "fixturenonce"
        ).hello()
    }

    @Test("Both model types resolve to this runner")
    func registryClaims() throws {
        for modelType in ["qwen4_exp", "qwen4_exp_text"] {
            #expect(RunnerRegistry.shared.contains(modelType: modelType))
            let runner = try RunnerRegistry.shared.resolve(modelType: modelType)
            #expect(runner.manifest.runnerID == "layr/qwen4exp-125b-a6b")
        }
    }

    @Test("The declaration is the contract's section 11 declaration")
    func declaration() {
        let manifest = Qwen4ExpRunner.manifest
        #expect(manifest.backend == "mlx")
        #expect(manifest.kvBackends == [.contiguous])
        #expect(manifest.multimodal == false)
        #expect(manifest.recurrentLayers)
        #expect(manifest.requiresKeepMask)
        #expect(manifest.decoders.map(\.mode) == ["serial", "mtp"])
        #expect(manifest.decoders[1].drafter == .embeddedHead)
        #expect(manifest.decoders[1].state == .requestStateful)
        #expect(manifest.decoders[1].depth == 1 ... 6)
        #expect(manifest.regimes.allSatisfy { $0.batch.maxWidth == 1 })
        #expect(manifest.regimes.map(\.timing) == [.freeRun, .teacherForced])
    }

    @Test("Hello derives one free-run capability, no batching, and both modes")
    func helloDerivation() {
        let hello = hello(loadedDecoders: [.serial, .mtp])
        #expect(hello.backend == "mlx")
        // Single-stream only: one capability, and no max_batch_size at all.
        #expect(hello.capabilities == ["free_run_decode"])
        #expect(hello.maxBatchSize == nil)
        #expect(hello.specModes == ["serial", "mtp"])
        #expect(hello.runner?.id == "layr/qwen4exp-125b-a6b")
        #expect(hello.runner?.modelType == "qwen4_exp")
        #expect(
            hello.runner?.manifestSHA256
                == "0430b22f8325c9c9371910d1e14eb3c78b235932bf35fa6623a4c511dd68e180")
    }

    @Test("A checkpoint without the embedded head advertises serial only")
    func helloWithoutTheHead() {
        // §6.2 rule 1: a mode is advertised only if its drafter loaded.
        #expect(hello(loadedDecoders: [.serial]).specModes == ["serial"])
    }
}
