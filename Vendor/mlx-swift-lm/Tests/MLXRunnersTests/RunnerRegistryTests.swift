// RunnerRegistryTests.swift
//
// The registry is Darkbloom's advertise gate and bench-worker's resolution
// path, so an unclaimed `model_type` must REFUSE rather than fall through to
// some default family.

import Foundation
import Testing

@testable import MLXRunners

@Suite("Runner registry")
struct RunnerRegistryTests {

    @Test("Every first-party model_type resolves to its own runner")
    func resolvesFirstParty() throws {
        let expected: [String: String] = [
            "gemma4": "layr/gemma4-text",
            "gemma4_text": "layr/gemma4-text",
            "gpt_oss": "layr/gptoss",
            "nemotron_h": "layr/nemotron35-lightning",
            "prism_hadamard_qwen35": "layr/qwen35",
            "qwen3_5": "layr/qwen35",
            "qwen3_5_moe": "layr/qwen35",
            "qwen3_5_text": "layr/qwen35",
            "qwen3_vl": "layr/qwen3vl",
            "qwen3_vl_moe": "layr/qwen3vl",
            "qwen4_exp": "layr/qwen4exp-125b-a6b",
            "qwen4_exp_text": "layr/qwen4exp-125b-a6b",
        ]
        for (modelType, runnerID) in expected {
            #expect(RunnerRegistry.shared.contains(modelType: modelType))
            let runner = try RunnerRegistry.shared.resolve(modelType: modelType)
            #expect(runner.manifest.runnerID == runnerID)
        }
    }

    @Test("An unclaimed model_type is refused, not defaulted")
    func refusesUnknown() {
        #expect(!RunnerRegistry.shared.contains(modelType: "not_a_model"))
        #expect(throws: RunnerRegistry.RegistryError.unknownModelType("not_a_model")) {
            _ = try RunnerRegistry.shared.resolve(modelType: "not_a_model")
        }
    }

    @Test("manifests() lists one manifest per runner, stably ordered")
    func manifestListing() {
        let ids = RunnerRegistry.shared.manifests().map(\.runnerID)
        #expect(ids == ids.sorted())
        #expect(
            Set(ids).isSuperset(
                of: [
                    "layr/gemma4-text", "layr/gptoss", "layr/nemotron35-lightning",
                    "layr/qwen35", "layr/qwen3vl", "layr/qwen4exp-125b-a6b",
                ]))
        #expect(ids.count == Set(ids).count)
    }

    /// A track repo registers its own runner before resolving; this proves
    /// the claim lands on every `model_type` the manifest names. The mock
    /// used here is the teacher-forced one, so nothing in the test suite ever
    /// registers a model_type a real runner will claim later.
    @Test("A registered runner claims every model_type in its manifest")
    func registrationClaimsAll() throws {
        RunnerRegistry.shared.register(TeacherForcedOnlyMockRunner.self)
        #expect(RunnerRegistry.shared.contains(modelType: "mock-teacher-forced"))
        let resolved = try RunnerRegistry.shared.resolve(modelType: "mock-teacher-forced")
        #expect(resolved.manifest.runnerID == "layr/mock-teacher-forced")
    }
}
