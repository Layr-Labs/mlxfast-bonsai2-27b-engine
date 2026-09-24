import Foundation
import MLX
import XCTest

@_spi(Diagnostics) @testable import MLXLMCommon

/// Exercise the observation through real engine requests. A second model
/// forward, a mistaken pending-token index, or a prompt/decode phase mix-up
/// must not produce a plausible-looking diagnostic record.
final class CBv2OrdinaryLogitDiagnosticTests: XCTestCase {
    private final class RecordingModel: CBv2SteppableModel, CBv2MTPPolicyTopTwoProviding {
        let inner = TinyTestModel.make(seed: 0x10_617, fullAttentionOnly: true)
        private let lock = NSLock()
        private var shapes: [[Int]] = []
        private var topTwoCalls = 0

        var observation: (shapes: [[Int]], topTwoCalls: Int) {
            lock.lock()
            defer { lock.unlock() }
            return (shapes, topTwoCalls)
        }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            lock.lock()
            shapes.append(tokens.shape)
            lock.unlock()
            return inner.forward(tokens: tokens, caches: caches)
        }

        func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
            lock.lock()
            topTwoCalls += 1
            lock.unlock()
            let vocabulary = logits.dim(-1)
            let first = argMax(logits, axis: -1).asType(.int32)
            let vocabularyIDs = MLXArray(Int32(0) ..< Int32(vocabulary))
                .reshaped([1, 1, vocabulary])
            let masked = MLX.where(
                vocabularyIDs .== first[0..., 0..., .newAxis],
                MLXArray(-Float.infinity), logits)
            let second = argMax(masked, axis: -1).asType(.int32)
            let ids = stacked([first, second], axis: -1)
            return (ids, takeAlong(logits, ids, axis: -1))
        }
    }

    private struct Run {
        let tokens: [Int]
        let shapes: [[Int]]
        let topTwoCalls: Int
        let chainedSteps: Int
        let snapshot: CBv2LogitDiagnosticSnapshot?
    }

    private func run(prompt: [Int], captureIndex: Int?) async throws -> Run {
        let model = RecordingModel()
        let engine = EngineV2(
            model: model, layerKinds: model.inner.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 26)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.inner.layerKinds),
            sampler: CBv2GreedySampler(),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 8, maxWaiting: 4, enablePrefixCache: false))
        do {
            if let captureIndex {
                try engine.configureLogitDiagnostic(.init(
                    requestID: 501, outputIndex: captureIndex,
                    candidateIDs: [1, 2], maximumRecords: 1))
            }
            let output = await cbv2SchedCollect(try engine.submit(CBv2Request(
                id: CBv2RequestID(501), promptTokens: prompt,
                sampling: CBv2SamplingParams(temperature: 0), maxTokens: 8,
                prefixCacheEnabled: false)))
            XCTAssertEqual(output.tokens.count, 8)
            var snapshot: CBv2LogitDiagnosticSnapshot?
            if captureIndex != nil {
                let retired = await cbv2SchedWait {
                    do {
                        snapshot = try engine.takeLogitDiagnosticSnapshot()
                        return snapshot != nil
                    } catch { return false }
                }
                XCTAssertTrue(retired, "diagnostic is readable only after the actual step retires")
            }
            let chainedSteps = engine.chainedStepCount
            await engine.shutdown()
            let observation = model.observation
            return Run(
                tokens: output.tokens, shapes: observation.shapes,
                topTwoCalls: observation.topTwoCalls, chainedSteps: chainedSteps,
                snapshot: snapshot)
        } catch {
            await engine.shutdown()
            throw error
        }
    }

    func testChainedCaptureMapsPendingTokenWithoutAnotherForward() async throws {
        let prompt = makePromptTokens(length: 17, seed: 41)
        let control = try await run(prompt: prompt, captureIndex: nil)
        let observed = try await run(prompt: prompt, captureIndex: 3)
        XCTAssertEqual(control.tokens, observed.tokens)
        XCTAssertEqual(control.shapes, observed.shapes, "capture must not recompute model logits")
        XCTAssertEqual(control.topTwoCalls, 0, "default nil must not build diagnostic reductions")
        XCTAssertEqual(observed.topTwoCalls, 0, "diagnostic reductions must not invoke model policy capabilities")
        XCTAssertGreaterThan(observed.chainedSteps, 0)
        let snapshot = try XCTUnwrap(observed.snapshot)
        XCTAssertEqual(snapshot.records.count, 1)
        let record = try XCTUnwrap(snapshot.records.first)
        XCTAssertEqual(record.outputIndex, 3)
        XCTAssertEqual(record.phase, "chained_decode")
        XCTAssertEqual(record.cacheOffset, prompt.count + 2)
        XCTAssertEqual(record.seedToken, observed.tokens[2])
        XCTAssertEqual(record.argMaxID, observed.tokens[3])
        XCTAssertEqual(record.topTwoIDs.first, observed.tokens[3])
        XCTAssertEqual(record.targetToken, observed.tokens[3])
        XCTAssertEqual(record.outcome, "confirmed")
        XCTAssertEqual(record.nanCount, 0)
        XCTAssertEqual(record.infiniteCount, 0)
    }

    func testPromptFrontierIncludesDecodeShapedAndChunkedPrefill() async throws {
        for length in [1, 18] {
            let prompt = makePromptTokens(length: length, seed: 91)
            let control = try await run(prompt: prompt, captureIndex: nil)
            let observed = try await run(prompt: prompt, captureIndex: 0)
            XCTAssertEqual(control.tokens, observed.tokens)
            XCTAssertEqual(control.shapes, observed.shapes)
            let snapshot = try XCTUnwrap(observed.snapshot)
            XCTAssertEqual(snapshot.records.count, 1)
            let record = try XCTUnwrap(snapshot.records.first)
            XCTAssertEqual(record.outputIndex, 0)
            XCTAssertEqual(record.phase, "prefill")
            XCTAssertEqual(record.cacheOffset, length == 1 ? 0 : 16)
            XCTAssertEqual(record.seedToken, prompt.last)
            XCTAssertEqual(record.argMaxID, observed.tokens[0])
            XCTAssertEqual(record.targetToken, observed.tokens[0])
            XCTAssertEqual(record.outcome, "confirmed")
        }
    }
}
