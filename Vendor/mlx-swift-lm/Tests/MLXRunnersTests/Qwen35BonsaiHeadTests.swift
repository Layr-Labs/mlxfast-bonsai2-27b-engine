// Qwen35BonsaiHeadTests.swift
//
// The Ternary Bonsai 2 27B pack declares `mtp_num_hidden_layers: 0` and ships
// no `mtp.*` tensors, so this track's drafters are SEPARATE published exports
// and the caller names one with `--drafter`. Three things follow, and all
// three are here: the manifest must say `assistantCheckpoint`, the provenance
// sealed into the hello must be the hash of that directory's shards rather
// than the embedded-head scan (which finds nothing there and returns nil), and
// the runner must advertise ONLY the mode whose drafter actually loaded.
//
// Model-free by construction: nothing here loads weights or touches Metal.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("Qwen 3.5 runner, separate head")
struct Qwen35BonsaiHeadTests {

    @Test("The runner claims the packed Bonsai model type")
    func claimsPackedModelType() throws {
        #expect(Qwen35Runner.manifest.modelTypes.contains("prism_hadamard_qwen35"))
        let runner = try RunnerRegistry.shared.resolve(modelType: "prism_hadamard_qwen35")
        #expect(runner.manifest.runnerID == "layr/qwen35")
    }

    @Test("Both speculative decoders declare a separate assistant checkpoint")
    func declaresAssistantCheckpoint() throws {
        let decoders = Qwen35Runner.manifest.decoders
        #expect(decoders.map(\.mode) == ["serial", "mtp", "dflash"])
        #expect(decoders[0].drafter == .none)
        #expect(decoders[0].depth == nil)
        for speculative in decoders.dropFirst() {
            #expect(speculative.drafter == .assistantCheckpoint)
            #expect(speculative.state == .requestStateful)
            #expect(speculative.depth?.lowerBound == 1)
        }
        // The chain's depth range is the tested chain ceiling; the block's is
        // its own and larger. One must never be read off the other.
        #expect(decoders[1].depth?.upperBound == CBv2MTPConfig.testedMaxDraftTokens)
        #expect(
            decoders[2].depth?.upperBound == CBv2MTPConfig.testedMaxBlockDraftTokens)
        #expect(decoders[1].depth?.upperBound != decoders[2].depth?.upperBound)
    }

    /// §6.2 rule 1: a worker may resolve only a mode it advertised, so the
    /// advertised mode is decided by WHICH drafter loaded, not by what the
    /// manifest could declare.
    @Test("Only the loaded drafter's own mode is advertised")
    func advertisesOnlyTheLoadedMode() {
        #expect(Qwen35Runner.decoder(of: nil) == nil)
        #expect(Qwen35Runner.decoder(of: BonsaiChainDrafterStub()) == .mtp)
        #expect(Qwen35Runner.decoder(of: BonsaiBlockDrafterStub()) == .dflash)
    }

    /// A separate export holds no `mtp.*` index entry, so the embedded-head
    /// scan reports "no head" over exactly the directory whose head the runner
    /// is about to attach. The whole-directory hash is the one that answers.
    @Test("A separate export hashes whole; the embedded scan finds nothing")
    func separateExportProvenance() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("bonsai-head-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let shard = directory.appendingPathComponent("model.safetensors")
        let bytes = Data((0 ..< 4096).map { UInt8(truncatingIfNeeded: $0) })
        try bytes.write(to: shard)
        let index = ["weight_map": ["fc.weight": "model.safetensors"]]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))

        #expect(try RunnerCheckpoint.provenance(ofEmbeddedHeadAt: directory) == nil)

        let provenance = try RunnerCheckpoint.provenance(ofHeadAt: directory)
        #expect(provenance.fileCount == 1)
        #expect(provenance.bytes == bytes.count)
        #expect(provenance.sha256.count == 64)
    }
}

// MARK: - Fixtures

private final class BonsaiPreparedStub: CBv2MTPPreparedCapture {}

/// A chain drafter: the engine asks it for one token at a time.
private final class BonsaiChainDrafterStub: CBv2MTPDrafter {
    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        BonsaiPreparedStub()
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        (tokens, hidden)
    }
}

/// A block drafter: one propose per round. The chain verbs come from the
/// protocol's own refusing defaults, which is part of what this pins.
private final class BonsaiBlockDrafterStub: CBv2MTPBlockDrafter {
    private final class State: CBv2MTPRequestState {
        var committedInputCount = 0
        var stagedInputCount = 0
    }

    func setBlockContextArmed(_ armed: Bool) throws {}
    func blockContextHidden() -> MLXArray? { nil }
    func makeRequestState() -> any CBv2MTPRequestState { State() }
    func releaseRequestState(_ requestState: any CBv2MTPRequestState) {}
    func discardRound(requestState: any CBv2MTPRequestState) {}
    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] { [] }

    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState
    ) {}

    func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray
    ) {}

    func proposeBlock(
        anchor: Int, depth: Int, requestState: any CBv2MTPRequestState
    ) throws -> MLXArray {
        MLXArray.zeros([1, depth], dtype: .int32)
    }

    func trimBlockState(
        _ requestState: any CBv2MTPRequestState, toCommittedLength committed: Int
    ) {}
}
