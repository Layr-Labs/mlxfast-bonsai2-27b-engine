// Qwen4ExpRunnerEngineTests.swift
//
// The Qwen 3.8 Flash-Next runner boundary over the tiny fixture model.
//
// Two claims:
//
//   * the runner's engine build REFUSES a paged `EngineBuild`, by name,
//     because the manifest declares contiguous storage only;
//   * the one-row teacher-forced stepper and the CBv2 engine, over the SAME
//     model instance, choose the same greedy tokens. Contract §5 rule 1: the
//     stepper is not a second implementation, and §7's tie-break (lowest
//     token id) is the same rule on both sides.
//
// The agreement test drives single-token decodes, so it needs the complete
// ahead-of-time metallib for the same reason `Qwen4ExpForwardParityTests`
// does, and it skips without it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import MLXRunners
import XCTest

@testable import MLXLLM

/// Deterministic id → text mapping. The engine needs a detokenizer; this test
/// reads tokens, not text.
private struct Qwen4ExpStubTokenizer: MLXLMCommon.Tokenizer {
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { "<\($0)>" }.joined()
    }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
}

final class Qwen4ExpRunnerEngineTests: XCTestCase {

    private static let kvBytesCapacity = 1 << 28

    private func newCaches(
        _ model: Qwen4ExpModel
    ) -> (
        (_ layerIndex: Int, _ kind: CBv2LayerKind) throws -> any CBv2AttendingLayerCache
    ) throws -> [any CBv2AttendingLayerCache] {
        { make in try model.newCacheV2(makeLayerCache: make) }
    }

    private func makeEngine(
        _ model: Qwen4ExpModel, kvBackend: KVBackendKind, mtpDepth: Int = 0
    ) throws -> any CBv2Engine {
        let drafter = mtpDepth > 0 ? Qwen4ExpInlineMTPAssistant(target: model) : nil
        if mtpDepth > 0 { XCTAssertNotNil(drafter, "the fixture carries an mtp head") }
        // The worker's own speculative build: automatic verification with
        // the envelope set to exactly this leg's shape.
        let mtpConfig =
            mtpDepth > 0
            ? CBv2MTPConfig(
                enabled: true, maxDraftTokens: mtpDepth, maxSpeculativeBatch: 1,
                fixedDraftTokens: mtpDepth, verificationMode: .automatic,
                maxAutomaticRectangularTokens: 1 + mtpDepth)
            : CBv2MTPConfig(enabled: false)
        return try RunnerEngineAssembly.makeEngine(
            manifest: Qwen4ExpRunner.manifest,
            loadedDecoders: mtpDepth > 0 ? [.serial, .mtp] : [.serial],
            model: model,
            tokenizer: Qwen4ExpStubTokenizer(),
            layerKinds: model.cbv2LayerKinds,
            newCaches: newCaches(model),
            mtpDrafter: drafter,
            build: EngineBuild(
                kvBackend: kvBackend,
                kvBytesCapacity: Self.kvBytesCapacity,
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: 1, prefillChunkSize: 64),
                decoder: mtpDepth > 0 ? .mtp : .serial,
                mtpConfig: mtpConfig))
    }

    private func greedyTokens(_ engine: any CBv2Engine, prompt: [Int], steps: Int) async throws -> [Int] {
        var request = CBv2Request(
            id: CBv2RequestID(1), promptTokens: prompt, maxTokens: steps + 1)
        request.sampling = CBv2SamplingParams(temperature: 0, topP: 1, topK: 0)
        request.stopTokens = []
        var free: [Int] = []
        for await event in try engine.submit(request) {
            if case .delta(_, let tokens, _) = event { free.append(contentsOf: tokens) }
        }
        return free
    }

    // MARK: - MTP rounds verify the window in ONE target forward

    /// A depth-2 free run through the engine must produce the serial
    /// engine's tokens AND verify every round rectangularly: one target
    /// forward over the `1 + k` window, never one forward per column.
    func testMTPRoundsVerifyRectangularly() async throws {
        try requireCompleteMetallib()
        let prompt: [Int] = (0 ..< 40).map { ($0 * 37 + 11) % 64 }
        let steps = 12
        let model = try Qwen4ExpFixture.model()
        Self.randomizeParameters(model, seed: 11)

        let serialEngine = try makeEngine(model, kvBackend: .contiguous)
        let serial = try await greedyTokens(serialEngine, prompt: prompt, steps: steps)
        await serialEngine.shutdown()

        let mtpEngine = try makeEngine(model, kvBackend: .contiguous, mtpDepth: 2)
        let speculative = try await greedyTokens(mtpEngine, prompt: prompt, steps: steps)
        let metrics = try XCTUnwrap(
            (mtpEngine as? EngineV2)?.mtpMetricsSnapshot(), "the mtp driver must be active")
        await mtpEngine.shutdown()

        XCTAssertEqual(speculative, serial)
        XCTAssertGreaterThan(metrics.rounds, 0, "the run must have drafted")
        XCTAssertGreaterThan(metrics.rectangularVerificationRounds, 0)
        XCTAssertEqual(metrics.serialVerificationRounds, 0, "no round may fall back to the serial oracle")
        XCTAssertEqual(metrics.controllerFallbacks["captured_verify_unsupported"], nil)
        XCTAssertEqual(metrics.controllerFallbacks["rectangular_cache_unsupported"], nil)
    }

    /// Replace every floating-point parameter with seeded normal noise.
    private static func randomizeParameters(_ model: Qwen4ExpModel, seed: UInt64) {
        MLXRandom.seed(seed)
        let replaced = model.parameters().flattened().compactMap {
            key, value -> (String, MLXArray)? in
            guard value.dtype == .float32 || value.dtype == .float16 || value.dtype == .bfloat16
            else { return nil }
            return (key, (MLXRandom.normal(value.shape) * 0.2).asType(value.dtype))
        }
        model.update(parameters: ModuleParameters.unflattened(replaced))
        eval(model)
    }

    // MARK: - Paged is refused, by name

    func testPagedEngineBuildIsRefused() throws {
        let model = try Qwen4ExpFixture.model(withMTP: false)
        XCTAssertThrowsError(try makeEngine(model, kvBackend: .paged)) { error in
            guard case RunnerError.kvBackendRefused(let requested, let declared) = error else {
                return XCTFail("expected a kv backend refusal, got \(error)")
            }
            XCTAssertEqual(requested, "paged")
            XCTAssertEqual(declared, ["contiguous"])
        }
    }

    // MARK: - Stepper and engine agree

    func testStepperAndEngineChooseTheSameTokens() async throws {
        try requireCompleteMetallib()
        // A prompt longer than the engine's prefill chunk, so the engine
        // prefills in several chunks while the stepper prefills in one
        // forward; and longer than the fixture's indexer budget, so the keep
        // mask is live. Deterministic content, no RNG.
        let prompt: [Int] = (0 ..< 200).map { ($0 * 37 + 11) % 64 }
        let steps = 12
        let model = try Qwen4ExpFixture.model(withMTP: false)
        // The fixture's default parameters leave the hyper-connection mixer
        // at its zero init, so every logit is 0.0 and the two drivers'
        // argmax differ only by tie-break (argPartition candidates vs the
        // first index), which proves nothing about the forward. Seeded
        // random parameters give the comparison teeth.
        Self.randomizeParameters(model, seed: 11)

        let stepper = CBv2SingleRowStepper(
            model: model,
            layerKinds: model.cbv2LayerKinds,
            newCaches: newCaches(model),
            kvBytesCapacity: Self.kvBytesCapacity,
            maxLength: 1024)
        try stepper.begin()
        var forced: [Int] = [try stepper.forward(prompt).argmax]
        for _ in 0 ..< steps {
            forced.append(try stepper.forward([forced[forced.count - 1]]).argmax)
        }
        XCTAssertEqual(stepper.forwards, steps + 1)

        let engine = try makeEngine(model, kvBackend: .contiguous)
        // The same greedy request bench-worker submits: no stop tokens, and
        // the sampler's tie-break is the stepper's.
        var request = CBv2Request(
            id: CBv2RequestID(1), promptTokens: prompt, maxTokens: steps + 1)
        request.sampling = CBv2SamplingParams(temperature: 0, topP: 1, topK: 0)
        request.stopTokens = []
        var free: [Int] = []
        for await event in try engine.submit(request) {
            if case .delta(_, let tokens, _) = event { free.append(contentsOf: tokens) }
        }
        await engine.shutdown()

        XCTAssertEqual(free, forced)
    }

    /// Same gate, and the same reason, as `Qwen4ExpForwardParityTests`: a
    /// single-token decode through the mixture-of-experts shared expert gate
    /// needs the `dot_product` kernel, which the metallib of an ordinary
    /// build does not carry, and a missing kernel aborts the process.
    private func requireCompleteMetallib() throws {
        let raw = ProcessInfo.processInfo.environment[
            Qwen4ExpForwardParityTests.optInVariable]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard ["1", "true", "yes", "on"].contains(raw ?? "") else {
            throw XCTSkip(
                "Set \(Qwen4ExpForwardParityTests.optInVariable)=1 to run this test. "
                    + "It needs a full ahead-of-time mlx.metallib carrying the "
                    + "dot_product kernels; see Qwen4ExpForwardParityTests.")
        }
    }
}
