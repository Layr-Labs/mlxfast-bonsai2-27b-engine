import Foundation
import MLX
import XCTest
@_spi(Diagnostics) @testable import MLXLMCommon

final class CBv2OrdinaryAttentionMetadataTests: XCTestCase {
    private final class AllTokens: CBv2TokenConstraint, @unchecked Sendable {
        let mode: CBv2TokenConstraintMode = .required
        let maxTokens = 8
        let fallbackTokenID = 0
        let initialState = 0
        func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] { Array(0..<128) }
        func nextState(state: Int, tokenID: Int) -> Int? { (0..<128).contains(tokenID) ? 0 : nil }
    }

    /// The ordinary model forward must continue binding and staging its real
    /// recurrent transaction once. Metadata never calls the non-recurrent seam.
    private final class RecordingModel: CBv2RecurrentSteppableModel {
        let inner = TinyTestModel.make(seed: 0xA77E, fullAttentionOnly: true)
        let cbv2Capabilities = CBv2ModelCapabilities.initialRecurrentTarget
        let recurrentStateSpec: CBv2RecurrentStateSpec? = .init(layers: [
            .init(modelLayerIndex: 11, convShape: [1, 1, 1], convDType: .float32,
                  ssmShape: [1, 1, 1, 1], ssmDType: .float32)
        ])
        let lock = NSLock()
        var shapes: [[Int]] = []
        var staged = 0
        var boundForwards = 0
        var lastCaches: [any CBv2AttendingLayerCache] = []
        var cancelSelected: (() -> Void)?

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            preconditionFailure("ordinary recurrent transaction must be preserved")
        }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
                     recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray {
            let selected = caches.contains {
                ($0 as? any CBv2AttentionMetadataBinding)?.attentionMetadata != nil
            }
            lock.lock()
            shapes.append(tokens.shape)
            lastCaches = caches
            if selected { boundForwards += 1 }
            lock.unlock()
            let output = inner.forward(tokens: tokens, caches: caches)
            for evaluation in recurrentState {
                let previous = evaluation.inputState(modelLayerIndex: 11)
                try! evaluation.stage(
                    modelLayerIndex: 11,
                    conv: (previous?.conv ?? MLXArray.zeros([1, 1, 1])) + 1,
                    ssm: (previous?.ssm ?? MLXArray.zeros([1, 1, 1, 1])) + 1)
                lock.lock(); staged += 1; lock.unlock()
            }
            if selected { cancelSelected?() }
            return output
        }
    }

    private struct Run {
        let tokens: [Int]
        let shapes: [[Int]]
        let staged: Int
        let boundForwards: Int
        let snapshot: CBv2AttentionMetadataSnapshot?
    }

    private func run(index: Int?, direct: Bool, cancel: Bool = false) async throws -> Run {
        let model = RecordingModel()
        let engine = EngineV2(
            model: model, layerKinds: model.inner.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 26)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.inner.layerKinds),
            sampler: CBv2GreedySampler(), schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 8, maxWaiting: 4, enablePrefixCache: false))
        let requestID = CBv2RequestID(2)
        if cancel { model.cancelSelected = { [weak engine] in engine?.cancel(requestID) } }
        do {
            if let index {
                try engine.configureAttentionMetadata(.init(requestID: 2, outputIndex: index))
            }
            let result = await cbv2SchedCollect(try engine.submit(CBv2Request(
                id: requestID, promptTokens: makePromptTokens(length: 17, seed: 71),
                sampling: .init(temperature: 0), maxTokens: 8, prefixCacheEnabled: false,
                tokenConstraint: direct ? AllTokens() : nil)))
            var snapshot: CBv2AttentionMetadataSnapshot?
            let idle = await cbv2SchedWait {
                do {
                    snapshot = try engine.takeAttentionMetadataSnapshot()
                    return true
                } catch { return false }
            }
            XCTAssertTrue(idle)
            XCTAssertTrue(model.lastCaches.allSatisfy {
                ($0 as? any CBv2AttentionMetadataBinding)?.attentionMetadata == nil
            }, "defer must clear every concrete cache binding")
            try engine.configureAttentionMetadata(nil)
            XCTAssertNil(try engine.takeAttentionMetadataSnapshot())
            await engine.shutdown()
            XCTAssertNil(engine.loopForTesting.attentionMetadata)
            return Run(tokens: result.tokens, shapes: model.shapes, staged: model.staged,
                       boundForwards: model.boundForwards, snapshot: snapshot)
        } catch {
            await engine.shutdown()
            throw error
        }
    }

    func testDirectAndChainedMappingKeepOutputsAndRecurrentForwards() async throws {
        for direct in [false, true] {
            let control = try await run(index: nil, direct: direct)
            let captured = try await run(index: 3, direct: direct)
            XCTAssertEqual(control.tokens, captured.tokens)
            XCTAssertEqual(control.shapes, captured.shapes)
            XCTAssertEqual(control.staged, captured.staged)
            XCTAssertEqual(captured.staged, captured.shapes.count)
            XCTAssertEqual(control.boundForwards, 0)
            XCTAssertEqual(captured.boundForwards, 1)
            let snapshot = try XCTUnwrap(captured.snapshot)
            XCTAssertEqual(snapshot.selectedForwards, 1)
            XCTAssertEqual(snapshot.records.count, snapshot.expectedOwnerCount)
            XCTAssertTrue(snapshot.refusals.isEmpty)
            XCTAssertTrue(snapshot.forwardSucceeded)
            XCTAssertEqual(snapshot.sampleOutcome, "confirmed")
            XCTAssertEqual(snapshot.seedToken, captured.tokens[2])
            XCTAssertEqual(snapshot.targetToken, captured.tokens[3])
            for record in snapshot.records {
                XCTAssertEqual(record.phase, direct ? "decode" : "chained_decode")
                XCTAssertEqual(record.outputIndex, 3)
                XCTAssertEqual(record.offsetBefore, 19)
                XCTAssertEqual(record.offsetAfter, 20)
            }
        }
    }

    func testCancellationRetiresConstructedMetadataWithoutConfirmingIt() async throws {
        let captured = try await run(index: 3, direct: false, cancel: true)
        let snapshot = try XCTUnwrap(captured.snapshot)
        XCTAssertEqual(captured.boundForwards, 1)
        XCTAssertEqual(snapshot.sampleOutcome, "discarded")
        XCTAssertNil(snapshot.targetToken)
        XCTAssertLessThanOrEqual(captured.tokens.count, 3)
        XCTAssertEqual(snapshot.records.count, snapshot.expectedOwnerCount)
    }
}
