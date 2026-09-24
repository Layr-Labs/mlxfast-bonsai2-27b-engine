import Foundation
import MLX
import Testing
@_spi(Benchmarking) @testable import MLXLMCommon

@Suite("Actual target dispatch width", .serialized)
struct CBv2ForwardShapeEngineTests {
    private final class Model: CBv2PackedPrefillSteppableModel {
        let split: Bool
        var calls = 0
        var supportsPackedPrefill: Bool { true }
        init(split: Bool) { self.split = split }

        private func leaf(_ tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            let observation = CBv2ForwardShapeObservation.beginTarget(
                liveBatchRows: tokens.dim(0), sequenceWidth: tokens.dim(1))
            defer { observation?.end() }
            calls += 1
            var logits = broadcast(MLXArray([Float(0), Float(1)]).reshaped([1, 1, 2]),
                to: [tokens.dim(0), tokens.dim(1), 2])
            for cache in caches {
                let offsets = cache.rows.map(\.absoluteOffset)
                let qkv = broadcast(tokens.asType(.float32).reshaped([tokens.dim(0), 1, tokens.dim(1), 1]),
                    to: [tokens.dim(0), 1, tokens.dim(1), cache.kind.headDim])
                let attended = cache.updateAndAttend(queries: qkv, keys: qkv, values: qkv,
                    scale: 1 / Float(cache.kind.headDim).squareRoot(), sinks: nil)
                // Both logits receive the same attention contribution, preserving
                // token 1 while making the real KV/attention work part of readback.
                logits = logits + sum(attended, axes: [1, 3]).expandedDimensions(axis: 2)
                for (offset, row) in zip(offsets, cache.rows) {
                    #expect(row.absoluteOffset == offset + tokens.dim(1))
                }
            }
            return logits
        }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            if split {
                let rows = caches.map(\.rows)
                defer {
                    for (cache, originalRows) in zip(caches, rows) { cache.setRows(originalRows) }
                }
                return concatenated((0..<tokens.dim(0)).map { index in
                    for (cache, originalRows) in zip(caches, rows) { cache.setRows([originalRows[index]]) }
                    return leaf(tokens[index..<(index + 1)], caches: caches)
                }, axis: 0)
            }
            return leaf(tokens, caches: caches)
        }

        func prefill(tokens: MLXArray, inputEmbeddings: MLXArray?,
            caches: [CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement) -> MLXArray
        {
            let value = forward(tokens: tokens, caches: caches)
            switch requirement {
            case .evaluationOnly: return value[0..., -1, 0..<1]
            case .lastPositionLogits: return value[0..., -1, 0...]
            }
        }
    }

    @Test(arguments: [false, true])
    func fourRequestsObserveActualPackedOrSplitLeafCalls(split: Bool) async throws {
        let model = Model(split: split)
        let kinds = [CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 1, queryHeads: 1)]
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20))
        let engine = EngineV2(model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(maxConcurrentRequests: 4, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 16, maxWaiting: 8, enablePrefixCache: false))
        let before = try engine.beginForwardShapeObservation()
        let streams = try engine.loopForTesting.onEngineQueueSync {
            // All enqueue operations precede the first scheduled engine step.
            try (0..<4).map { index in
                try engine.submit(.init(id: .init(UInt64(index + 1)), promptTokens: [1, 1],
                    sampling: .init(temperature: 0), maxTokens: 3, prefixCacheEnabled: false))
            }
        }
        for stream in streams {
            var finish: CBv2FinishReason?
            for await event in stream {
                if case .finished(let reason, _) = event { finish = reason }
            }
            #expect(finish == .length)
        }
        await engine.shutdown()
        let after = engine.forwardShapeSnapshot()
        let delta = after.delta(since: before)
        #expect(delta.complete)
        #expect(after.pendingSteps == 0 && after.unobservedDispatches == 0)
        let decode = delta.entries.filter { $0.axes.kind == .target && $0.axes.phase == .decode }
        #expect(!decode.isEmpty)
        if split {
            #expect(decode.allSatisfy { $0.axes.liveBatchRows == 1 })
        } else {
            #expect(decode.contains { $0.axes.liveBatchRows == 4 && $0.completedCalls > 0 })
        }
    }

    @Test func firstScopeRefusesDiscardedChainedWorkBeforeItsReadback() async throws {
        let model = Model(split: false)
        let kinds = [CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 1, queryHeads: 1)]
        let engine = EngineV2(model: model, layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 16, maxWaiting: 8, enablePrefixCache: false))
        let loop = engine.loopForTesting
        loop.onEngineQueueSync { loop.suspendStepExecutionAtCountForTesting = 2 }
        // The first sampled token stops the row only after its chained
        // successor has already entered the target trunk. No scope exists yet.
        let stream = try engine.submit(.init(id: .init(21), promptTokens: [1],
            sampling: .init(temperature: 0), maxTokens: 16, stopTokens: [1], prefixCacheEnabled: false))
        let tailHeld = await cbv2SchedWait {
            loop.onEngineQueueSync { model.calls == 2 && !loop.scheduler.hasWork }
        }
        #expect(tailHeld)
        #expect(!engine.forwardShapeSnapshot().enabled)
        #expect(throws: CBv2ForwardShapeError.self) { try engine.beginForwardShapeObservation() }
        #expect(!engine.forwardShapeSnapshot().enabled, "refusal must not create/reset a scope")
        loop.onEngineQueueSync { loop.suspendStepExecutionAtCountForTesting = nil }
        let result = await cbv2SchedCollect(stream)
        #expect(result.finishReason == .stop)
        await engine.shutdown()
        let scope = try engine.beginForwardShapeObservation()
        #expect(scope.enabled && scope.scope == 1 && scope.pendingSteps == 0 && scope.entries.isEmpty)
    }

    @Test(arguments: [false, true])
    func pagedRefusalRecordsOnlyCallsActuallyEntered(faultInside: Bool) async throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 1 << 20, segmentSizeBytes: 32768, layerDTypes: [.bfloat16]))
        let model = Model(split: false)
        let engine = EngineV2(model: model, layerKinds: [kind], backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()), sampler: CBv2GreedySampler())
        _ = try engine.beginForwardShapeObservation()
        engine.loopForTesting.onEngineQueueSync {
            let loop = engine.loopForTesting
            let step = loop.beginForwardShapeStep()
            defer { loop.endForwardShapeStep(step) }
            func refuse() {
                backend.pool.writeValidation.record(.init(layerIndex: 0, expected: .bfloat16,
                    keys: .float32, values: .bfloat16))
            }
            if !faultInside { refuse() }
            #expect(throws: CBv2PagedKVWriteError.self) {
                try loop.checkedModelForward {
                    let output = model.forward(tokens: MLXArray([Int32(1)]).reshaped([1, 1]), caches: [])
                    if faultInside { refuse() }
                    return output
                }
            }
            backend.pool.writeValidation.clearAfterRetirement()
        }
        #expect(model.calls == (faultInside ? 1 : 0))
        let snapshot = engine.forwardShapeSnapshot()
        #expect(snapshot.pendingSteps == 0 && snapshot.unobservedDispatches == 0 && snapshot.droppedCalls == 0)
        if faultInside {
            #expect(snapshot.entries.count == 1 && snapshot.abandonedSteps == 1)
            #expect(snapshot.entries.first?.submittedCalls == 1)
            #expect(snapshot.entries.first?.completedCalls == 0)
        } else {
            #expect(snapshot.entries.isEmpty && snapshot.abandonedSteps == 0)
        }
        await engine.shutdown()
    }
}
