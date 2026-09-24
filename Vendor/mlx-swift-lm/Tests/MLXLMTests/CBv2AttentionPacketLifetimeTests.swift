import MLX
import Testing
@_spi(Diagnostics) @testable import MLXLMCommon

@Suite("Attention packet fence and lifetime", .serialized)
struct CBv2AttentionPacketLifetimeTests {
    @Test func pendingRawGraphIsRefusedBeforeAsDataCanEvaluateIt() throws {
        let fixture = try AttentionPacketFixture("segmented", dtype: .bfloat16)
        let (state, forward, _, _, _, _) = try fixture.capture(qType: .bfloat16)
        state.metadata.confirm(requestID: 2, outputIndex: 62, seed: 11346, target: 1928)
        forward.materialize(discarded: false)
        let packet = state.takeSnapshot()
        #expect(packet.evaluationStatus == "refused")
        #expect(packet.metadata.refusals["packet_tensor_not_available_after_step_fence"] == 1)
        #expect(packet.tensors.isEmpty && forward.arrays.isEmpty)
    }

    @Test(arguments: ["fixed", "segmented"])
    func fencedReadSurvivesLaterOverwriteAndRecycledPages(backend: String) throws {
        let fixture = try AttentionPacketFixture(backend, dtype: .bfloat16, history: 16)
        let (state, forward, _, k, v, output) = try fixture.capture(qType: .float32)
        let row = try #require(fixture.rows[0] as? PagedSequenceKV)
        let originalPages = Set(row.table)
        let step = CBv2InFlightStep(assignments: [], participants: [], sampledRows: [],
                                    sampledTokens: nil, evalTargets: [], wallStartedNanos: 0)
        #expect(step.permitsChainedSuccessor)
        step.chained = true
        step.attentionPacket = forward
        #expect(!step.permitsChainedSuccessor,
                "a captured chained successor must itself block its successor before retirement")
        // Adversarially bypass the scheduler barrier and overwrite the same
        // final page slot. Ordinary gather must also publish its read back-edge.
        row.rollback(1)
        let changed = MLXArray.full([1, 2, 1, 256], values: MLXArray(Float(7)), dtype: .bfloat16)
        let overwrite = row.update(keys: changed, values: changed)
        eval([output, overwrite.0, overwrite.1] + forward.evaluationTargets)
        state.metadata.confirm(requestID: 2, outputIndex: 62, seed: 11346, target: 1928)
        forward.materialize(discarded: false)
        step.attentionPacket = nil
        #expect(step.permitsChainedSuccessor && forward.arrays.isEmpty)
        let packet = state.takeSnapshot()
        #expect(packet.evaluationStatus == "completed")
        let keys = try #require(packet.tensors["storedKeys"])
        let values = try #require(packet.tensors["storedValues"])
        #expect(keys.data == concatenated([fixture.historyKeys, k], axis: 2).asData(access: .copy).data)
        #expect(values.data == concatenated([fixture.historyValues, v], axis: 2).asData(access: .copy).data)
        fixture.release()
        let reused = try fixture.backend.makeSequenceState(
            layerKinds: [fixture.kind], promptLength: 16, maxLength: 80)
        defer { fixture.backend.release(reused) }
        let reusedRow = try #require(reused[0] as? PagedSequenceKV)
        let overwriteAll = MLXArray.full([1, 2, 17, 256], values: MLXArray(Float(-9)), dtype: .bfloat16)
        let recycled = reusedRow.update(keys: overwriteAll, values: overwriteAll)
        eval(recycled.0, recycled.1)
        #expect(!originalPages.isDisjoint(with: Set(reusedRow.table)), "test must actually recycle pages")
        #expect(keys.data == packet.tensors["storedKeys"]?.data,
                "the packet contains owned host bytes, not a live slab alias")
        #expect(keys.data != recycled.0.asData(access: .copy).data)
    }

    @Test func failureCancellationAndDrainReleaseEveryTensorHandle() throws {
        for failure in [false, true] {
            let fixture = try AttentionPacketFixture("contiguous", dtype: .float32)
            let capture = try fixture.capture(qType: .float32)
            let state = capture.0
            weak var handle = capture.1.arrays["storedKeys"]
            #expect(handle != nil)
            if failure { capture.1.finish(succeeded: false) }
            else {
                eval([capture.5] + capture.1.evaluationTargets)
                capture.1.materialize(discarded: true)
            }
            #expect(capture.1.arrays.isEmpty)
            // A tuple still owns the original inputs/output, but the gathered
            // stored-key view must no longer be retained by packet state.
            #expect(handle == nil)
            let snapshot = state.takeSnapshot()
            #expect(snapshot.tensors.isEmpty)
            #expect(snapshot.evaluationStatus != "completed")
            #expect(snapshot.metadata.targetToken == nil)
            state.discardPendingForward()
        }
    }

    @Test func unsupportedGeometryAndUnconfirmedSamplesCannotExportBytes() throws {
        let fixture = try AttentionPacketFixture("contiguous", dtype: .float32)
        let (state, forward, _, _, _, output) = try fixture.capture(qType: .float32)
        eval([output] + forward.evaluationTargets)
        forward.materialize(discarded: false)
        #expect(state.takeSnapshot().tensors.isEmpty, "an evaluated but unconfirmed step is not a decision")
        #expect(forward.arrays.isEmpty)
        for mode in ["wide", "softcap", "spans", "sinks"] {
            let rejected = try CBv2AttentionPacketState(.init(requestID: 2, outputIndex: 62, storageLayerIndex: 0))
            let selected = try #require(rejected.select(requestID: 2, outputIndex: 62, batchSize: 1, phase: "decode"))
            #expect(selected.metadata.bindOwner(cache: fixture.cache, storageLayerIndex: 0,
                                                kind: fixture.kind))
            let width = mode == "wide" ? 2 : 1
            let q = MLXArray.zeros([1, 16, width, 256])
            let kv = MLXArray.zeros([1, 2, width, 256])
            #expect(selected.begin(cache: fixture.cache, queries: q, keys: kv, values: kv, scale: 0.0625,
                sinks: mode == "sinks" ? MLXArray.zeros([16]) : nil,
                softcap: mode == "softcap" ? 1 : nil, spans: mode == "spans") == nil)
            #expect(rejected.reservedBytes == 0 && selected.evaluationTargets.isEmpty)
            selected.finish(succeeded: false)
        }
    }
}
