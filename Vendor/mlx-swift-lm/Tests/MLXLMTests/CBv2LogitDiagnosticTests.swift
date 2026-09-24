import Foundation
import MLX
import Testing
@_spi(Diagnostics) @testable import MLXLMCommon
@testable import MLXLLM

@Suite("CBv2 bounded logit diagnostic", .serialized)
struct CBv2LogitDiagnosticTests {
    @Test func invalidConfigurationsAreRejected() {
        for (index, ids, maximum) in [(-1, [1, 2], 1), (3, [], 1),
                                     (3, [1, 1], 1), (3, [1, 2, 3], 1),
                                     (3, [-1], 1), (3, [1], 0), (3, [1], 17)] {
            #expect(throws: CBv2LogitDiagnosticError.self) {
                try CBv2LogitDiagnosticConfig(
                    requestID: 7, outputIndex: index, candidateIDs: ids, maximumRecords: maximum)
            }
        }
    }

    @Test func selectionAndVocabularyChecksDoNotConsumeBudget() throws {
        let state = CBv2LogitDiagnosticState(try .init(
            requestID: 7, outputIndex: 53, candidateIDs: [2, 4], maximumRecords: 1))
        #expect(!state.reserve(requestID: .init(8), outputIndex: 53, vocabularySize: 8))
        #expect(!state.reserve(requestID: .init(7), outputIndex: 52, vocabularySize: 8))
        #expect(!state.reserve(requestID: .init(7), outputIndex: 53, vocabularySize: 4))
        #expect(state.reservedRecords == 0)
        #expect(state.invalidVocabularyRecords == 1)
        #expect(state.reserve(requestID: .init(7), outputIndex: 53, vocabularySize: 8))
        #expect(!state.reserve(requestID: .init(7), outputIndex: 53, vocabularySize: 8))
        let snapshot = state.takeSnapshot()
        #expect(snapshot.reservedRecords == 1)
        #expect(snapshot.omittedRecords == 1)
        #expect(!state.mayCapture(requestID: .init(7), outputIndex: 53))
        #expect(state.takeSnapshot().reservedRecords == 1, "drain must not refill the budget")
    }

    @Test func actualQwenReductionPreservesTiesNaNsAndStridedRows() {
        let logits = MLXArray([Float.nan, 5, 4, 3, 4, 2, -2, .infinity])
            .reshaped([1, 4, 2]).transposed(0, 2, 1)
        let topTwo = qwen35MTPTopTwoRows(logits)
        for row in 0..<2 {
            let reduced = CBv2LogitDiagnosticPacket.reduce(
                logits: logits[0, row],
                topTwo: (ids: topTwo.ids[row], values: topTwo.values[row]),
                candidateIDs: [1, 3])
            eval(reduced.ids, reduced.values)
            let ids = reduced.ids.asArray(Int32.self)
            let values = reduced.values.asArray(Float.self)
            #expect(Array(ids[1...2]) == (row == 0 ? [1, 2] : [3, 0]))
            #expect(ids[3] == (row == 0 ? 1 : 0))
            #expect(ids[4] == (row == 0 ? 0 : 1))
            #expect(values[3] == (row == 0 ? 4 : 3))
            #expect(values[4] == (row == 0 ? -2 : .infinity))
            // Independent argMax is intentionally not replaced by the fused
            // ordering when the input contains NaNs.
            #expect(ids[0] == argMax(logits[0, row]).item(Int32.self))
        }
    }

    private func packet(column: Int) -> CBv2LogitDiagnosticPacket {
        CBv2LogitDiagnosticPacket(
            requestID: .init(7), outputIndex: 50 + column, phase: "rectangular_verify",
            batchIndex: 0, batchSize: 1, column: column, verificationWidth: 4,
            draftDepth: 3, seedToken: 20, cacheOffset: 5550, backend: "test",
            logitDType: "float32", vocabularySize: 8, candidateIDs: [1, 2],
            ids: MLXArray([Int32(1), 1, 2, 0, 0]),
            values: MLXArray([Float(9), 9, 8, 9, 8]))
    }

    @Test func rejectionAndTruncationKeepDistinctContexts() {
        let drafts = [1, 4, 5]
        let targets = [1, 2, 3, 6]
        for column in 0..<4 {
            let p = packet(column: column)
            p.reconcile(accepted: 1, confirmed: 2, drafts: drafts, targets: targets)
            eval(p.evaluationTargets)
            let record = p.materialize()
            #expect(record.outputBase == 50)
            #expect(record.outputIndex == 50 + column)
            #expect(record.draftPrefix == Array(drafts.prefix(column)))
            #expect(record.outcome == (column < 2 ? "confirmed" : "speculative_suffix"))
            #expect(record.targetToken == targets[column])
        }
        let truncated = packet(column: 2)
        truncated.reconcile(accepted: 3, confirmed: 1, drafts: drafts, targets: targets)
        #expect(truncated.outcome == "truncated")
        let discarded = packet(column: 1)
        #expect(discarded.outcome == "discarded")
        #expect(discarded.targetToken == nil)
    }

    @Test func discardedPacketRetiresWithoutRetainingDeviceArrays() async throws {
        let model = TinyTestModel.make(seed: 77, fullAttentionOnly: true)
        let engine = EngineV2(
            model: model, layerKinds: model.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 24)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds))
        let loop = engine.loopForTesting
        let config = try CBv2LogitDiagnosticConfig(
            requestID: 7, outputIndex: 51, candidateIDs: [1, 2], maximumRecords: 1)
        // TinyTestModel has no policy top-two capability; diagnostics remain available.
        try engine.configureLogitDiagnostic(config)
        try engine.configureLogitDiagnostic(nil)
        weak var retired: CBv2LogitDiagnosticPacket?
        loop.onEngineQueueSync {
            var constructed = false
            func forbiddenLogits() -> MLXArray {
                constructed = true
                return MLXArray([Float(1), 2, 3])
            }
            #expect(loop.makeLogitDiagnostic(
                logits: forbiddenLogits(), requestID: .init(7), outputIndex: 51,
                phase: "plain", batchIndex: 0, batchSize: 1, seedToken: 1, cacheOffset: 3) == nil)
            #expect(!constructed, "nil configuration must not even construct a logit view")
            loop.logitDiagnostic = CBv2LogitDiagnosticState(config)
            #expect(loop.logitDiagnostic!.reserve(requestID: .init(7), outputIndex: 51, vocabularySize: 8))
            let step = CBv2InFlightStep(
                assignments: [], participants: [], sampledRows: [], sampledTokens: nil, evalTargets: [], wallStartedNanos: 0)
            do {
                let p = packet(column: 1)
                retired = p
                step.logitDiagnostics = [p]
                step.discard.insert(.init(7))
                eval(p.evaluationTargets)
            }
            // Same retirement function used after normal step-cost observation.
            loop.materializeLogitDiagnostics(step)
            #expect(step.logitDiagnostics.isEmpty)
            #expect(retired == nil)
            let snapshot = loop.logitDiagnostic!.takeSnapshot()
            #expect(snapshot.records.count == 1)
            #expect(snapshot.records[0].outcome == "discarded")
            #expect(loop.logitDiagnostic!.records.isEmpty)
        }
        await engine.shutdown()
        #expect(try engine.takeLogitDiagnosticSnapshot() == nil)
    }
}
