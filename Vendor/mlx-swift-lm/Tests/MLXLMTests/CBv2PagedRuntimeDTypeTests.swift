import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Paged runtime native KV dtype contract", .serialized)
struct CBv2PagedRuntimeDTypeTests {
    private func kinds() -> [CBv2LayerKind] {
        [nil, nil, 0].map { source in
            CBv2LayerKind(attention: .full, sharesKVWithLayer: source,
                          headDim: 64, kvHeads: 1, queryHeads: 2)
        }
    }

    private func backend() throws -> PagedKVBackend {
        try PagedKVBackend(layerKinds: kinds(), config: .init(
            capacityBytes: 4 << 20, maxPrefillChunk: 16, maxBufferLength: 4 << 20,
            segmentSizeBytes: 32768, layerDTypes: [.bfloat16, .bfloat16, .bfloat16]))
    }

    @Test("K and V are checked independently before decode, prefill, and rectangular writes",
          arguments: [0, 1, 2], [0, 1, 2])
    func firstFaultSuppressesLaterOwningAndBorrowedLayers(phase: Int, component: Int) throws {
        let backend = try backend()
        let states = try (0 ..< 2).map { _ in
            try backend.makeSequenceState(layerKinds: kinds(), promptLength: 3, maxLength: 32)
        }
        defer { for state in states { backend.release(state) } }
        let caches = backend.makeLayerCaches()
        let bank = CBv2LayerCacheBank(caches: caches)
        _ = bank.layerCaches(rowStates: states)
        defer { bank.releaseBoundRows() }
        if phase == 2 { for cache in caches { cache.mtpSerializesRectangularAttention = true } }
        let length = phase == 0 ? 1 : 3
        let q = MLXArray.ones([2, 2, length, 64], dtype: .float32)
        let native = MLXArray.ones([2, 1, length, 64], dtype: .bfloat16)
        let wrong = native.asType(.float32)
        let keys = component == 1 ? native : wrong
        let values = component == 0 ? native : wrong
        let fences = backend.pool.groups.mapValues { ObjectIdentifier($0.writeFence) }
        let first = caches[0].updateAndAttend(queries: q, keys: keys, values: values, scale: 0.125, sinks: nil)
        #expect(first === q, "the model can complete shape-valid graph construction without evaluating a placeholder")
        let later = caches[1].updateAndAttend(queries: q, keys: native, values: native, scale: 0.125, sinks: nil)
        let borrowed = caches[2].attendBorrowing(source: caches[0], queries: q, scale: 0.125, sinks: nil)
        #expect(later === q && borrowed === q)
        let fault = try #require(backend.pool.writeValidation.fault)
        #expect(fault.layerIndex == 0 && fault.expected == .bfloat16)
        #expect(fault.keys == keys.dtype && fault.values == values.dtype)
        #expect(throws: CBv2PagedKVWriteError.self) { try backend.pool.writeValidation.check() }
        #expect(states.flatMap { $0.compactMap { $0 } }.allSatisfy { $0.absoluteOffset == 0 })
        #expect(backend.bytesInUse == 0)
        #expect(backend.pool.groups.mapValues { ObjectIdentifier($0.writeFence) } == fences)
        #expect(caches.allSatisfy { $0.innerState().isEmpty })
    }

    @Test func directRowAndSegmentEntrypointsCannotSilentlyNarrowValues() throws {
        let backend = try backend()
        let states = try backend.makeSequenceState(layerKinds: kinds(), promptLength: 1, maxLength: 32)
        defer { backend.release(states) }
        let row = try #require(states[0] as? PagedSequenceKV)
        let native = MLXArray.ones([1, 1, 64], dtype: .bfloat16)
        let wrong = native.asType(.float32)
        row.write(keys: native, values: wrong)
        #expect(row.absoluteOffset == 0 && row.table.isEmpty)
        let group = backend.pool.group(row.groupKey)
        let fence = group.writeFence
        // Even deliberately unusable addressing cannot reach the kernel once
        // the pool is faulted; the first dtype fault remains authoritative.
        PagedSegmentTransfers.write(group: group, slots: [Int32.max], keys: native, values: native)
        let query = MLXArray.ones([1, 2, 64], dtype: .float32)
        let output = PagedSegmentAttention.decode(
            queries: query, newKeys: native, newValues: native, group: group,
            rows: [], sinks: nil, params: MLXArray.zeros([8]), softcap: false,
            source: backend.pool.kernelSource)
        #expect(output === query && group.writeFence === fence)
        #expect(backend.pool.writeValidation.fault?.values == .float32)
    }

    @Test func snapshotImportRejectsBadValueDTypeBeforeReservationWithoutPoisoningLivePool() throws {
        let backend = try backend()
        let kinds = kinds()
        let capability = CBv2PrefixReuseCapability.derive(layerKinds: kinds, backend: .pagedFP16)
        let plan = try #require(capability.plan(matchedBoundary: 3, maximumSequenceLength: 32))
        let native = MLXArray.ones([1, 1, 3, 64], dtype: .bfloat16)
        let prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?] = [
            (native, native, 3), (native, native.asType(.float32), 3), nil
        ]
        #expect(throws: CBv2PagedKVWriteError.self) {
            try backend.makeSequenceState(adopting: prefix, plan: plan, layerKinds: kinds, maxLength: 32)
        }
        #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)
        #expect(!backend.pool.writeValidation.isFaulted)
    }

    @Test func publicFixedKernelRejectsMismatchedWritesWithoutAContextLatch() throws {
        let storage = MLXArray.zeros([2, 1, 16, 64], dtype: .bfloat16)
        let keys = MLXArray.ones([1, 1, 64], dtype: .bfloat16)
        let values = keys.asType(.float32)
        let fence = MLXArray.zeros([1], dtype: .int32)
        let source = try PagedAttentionResources.loadSourceForCurrentProcess()
        let (seqinfo, length) = PagedAttentionKernel.seqinfo([
            .init(attendStart: 0, attendLength: 1, tableLength: 1, writePage: 1, writeSlot: 0)
        ])
        #expect(throws: CBv2PagedKVWriteError.self) {
            try PagedAttentionKernel.decode(
                queries: MLXArray.ones([1, 2, 64], dtype: .float32), newKeys: keys, newValues: values,
                kSlab: storage, vSlab: storage, tables: MLXArray.zeros([1, 8], dtype: .int32),
                seqinfo: seqinfo, maxAttendLength: length, sinks: nil, params: MLXArray.zeros([8]),
                softcap: false, pageSize: 16, writeFence: fence, kernelSource: source)
        }
        #expect(throws: CBv2PagedKVWriteError.self) {
            try PagedAttentionKernel.bulkWrite(
                kSlab: storage, vSlab: storage, keys: keys, values: values,
                slots: MLXArray.zeros([8], dtype: .int32), prevFence: fence,
                pageSize: 16, kernelSource: source)
        }
    }
}

private final class RuntimeDTypeModel: CBv2RecurrentSteppableModel, @unchecked Sendable {
    let recurrentStateSpec: CBv2RecurrentStateSpec?
    var cbv2Capabilities: CBv2ModelCapabilities {
        var capabilities: CBv2ModelCapabilities =
            recurrentStateSpec == nil ? .attentionOnly : .initialRecurrentTarget
        capabilities.supportsPagedKV = true
        capabilities.requiresNativePagedKV = true
        return capabilities
    }
    private let lock = NSLock()
    private var failOnCall: Int?
    private var calls = 0
    private var recurrentCalls = 0
    var recurrentForwardCount: Int { lock.withLock { recurrentCalls } }

    init(recurrent: Bool, failOnCall: Int?) {
        self.failOnCall = failOnCall
        recurrentStateSpec = recurrent ? .init(layers: [.init(
            modelLayerIndex: 0, convShape: [1, 1, 1], convDType: .float32,
            ssmShape: [1, 1, 1, 1], ssmDType: .float32)]) : nil
    }

    func allowValidCalls() { lock.withLock { failOnCall = nil } }

    func forward(tokens: MLXArray, caches: [any CBv2AttendingLayerCache]) -> MLXArray {
        build(tokens: tokens, caches: caches, recurrentState: [])
    }

    func forward(tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
                 recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray {
        lock.withLock { recurrentCalls += 1 }
        return build(tokens: tokens, caches: caches, recurrentState: recurrentState)
    }

    private func build(tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
                       recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray {
        let fail = lock.withLock {
            calls += 1
            return calls == failOnCall
        }
        let batch = tokens.dim(0), length = tokens.dim(1)
        let q = MLXArray.ones([batch, 2, length, 64], dtype: .float32)
        let k = MLXArray.ones([batch, 1, length, 64], dtype: .bfloat16)
        var last = q
        for (index, cache) in caches.enumerated() {
            if let source = cache.kind.sharesKVWithLayer {
                last = cache.attendBorrowing(source: caches[source], queries: q, scale: 0.125, sinks: nil)
            } else {
                let v = fail && index == 1 ? k.asType(.float32) : k
                last = cache.updateAndAttend(queries: q, keys: k, values: v, scale: 0.125, sinks: nil)
            }
        }
        for evaluation in recurrentState {
            let old = evaluation.inputState(modelLayerIndex: 0)
            try! evaluation.stage(modelLayerIndex: 0,
                conv: (old?.conv ?? MLXArray.zeros([1, 1, 1])) + 1,
                ssm: (old?.ssm ?? MLXArray.zeros([1, 1, 1, 1])) + 1)
        }
        return broadcast(MLXArray([Float(0), Float(1)]).reshaped([1, 1, 2]),
                         to: [batch, length, 2]) + sum(last) * Float(0)
    }
}

private final class RuntimeDTypeSampler: CBv2StepSampler, @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    private let inner = CBv2GreedySampler()
    var calls: Int { lock.withLock { count } }
    func sample(logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
                stepIndex: Int, pendingSampledTokens: MLXArray?,
                rowContext: () -> [CBv2SamplerRow]) -> MLXArray {
        lock.withLock { count += 1 }
        return inner.sample(logits: logits, params: params, requestIDs: requestIDs,
                            stepIndex: stepIndex, pendingSampledTokens: pendingSampledTokens,
                            rowContext: rowContext)
    }
}

@Suite("Paged runtime dtype failure retirement", .serialized)
struct CBv2PagedRuntimeDTypeEngineTests {
    private var kinds: [CBv2LayerKind] {
        [nil, nil, 0].map { source in
            .init(attention: .full, sharesKVWithLayer: source,
                  headDim: 64, kvHeads: 1, queryHeads: 2)
        }
    }

    private func makeBackend() throws -> PagedKVBackend {
        try PagedKVBackend(layerKinds: kinds, config: .init(
            capacityBytes: 8 << 20, maxPrefillChunk: 16, maxBufferLength: 8 << 20,
            segmentSizeBytes: 32768, layerDTypes: [.bfloat16, .bfloat16, .bfloat16]))
    }

    @Test("faulted prefill/chained decode never samples and the same request ID recovers",
          arguments: [false, true], [1, 3])
    func modelFailureAndRecovery(recurrent: Bool, failOnCall: Int) async throws {
        let backend = try makeBackend()
        let model = RuntimeDTypeModel(recurrent: recurrent, failOnCall: failOnCall)
        let sampler = RuntimeDTypeSampler()
        let engine = EngineV2(model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()), sampler: sampler,
            schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: 16,
                                   prefillChunkSize: 16, maxConcurrentPartialPrefills: 1),
            admissionConfig: .init(watermarkFraction: 0))
        let id = CBv2RequestID(77)
        let request = CBv2Request(id: id, promptTokens: [1, 2, 3, 4, 5],
                                  sampling: .init(temperature: 0), maxTokens: 8)
        let failed = await cbv2SchedCollect(try engine.submit(request))
        guard case .error(let message) = failed.finishReason else {
            Issue.record("dtype-changing model did not finish with a request-local error")
            await engine.shutdown()
            return
        }
        #expect(message.contains("paged KV dtype mismatch") && message.contains("values float32"))
        #expect(sampler.calls == failOnCall - 1, "faulted forward cannot reach sampler")
        engine.loopForTesting.onEngineQueueSync {
            #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)
            #expect(engine.admissionForTesting.bytesReserved == 0)
            #expect(!backend.pool.writeValidation.isFaulted)
            if failOnCall == 3 { #expect(engine.loopForTesting.chainedStepCount > 0) }
        }
        model.allowValidCalls()
        let recovered = await cbv2SchedCollect(try engine.submit(CBv2Request(
            id: id, promptTokens: [2, 3], sampling: .init(temperature: 0), maxTokens: 2)))
        #expect(recovered.finishReason == .length && recovered.tokens.count == 2)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)
        #expect(engine.admissionForTesting.bytesReserved == 0)
    }

    @Test(arguments: [false, true], [1, 2])
    func teacherForcedFailureThrowsBeforeScoringAndReleasesPrivateRows(
        recurrent: Bool, failOnCall: Int
    ) async throws {
        let backend = try makeBackend()
        let model = RuntimeDTypeModel(recurrent: recurrent, failOnCall: failOnCall)
        let engine = EngineV2(model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
            sampler: CBv2GreedySampler(), admissionConfig: .init(watermarkFraction: 0))
        #expect(throws: CBv2PagedKVWriteError.self) {
            try engine.loopForTesting.teacherForcedTop1(promptTokens: [1, 2, 3], continuation: [1, 1])
        }
        #expect(model.recurrentForwardCount == (recurrent ? failOnCall : 0))
        engine.loopForTesting.onEngineQueueSync {
            #expect(backend.bytesReserved == 0 && backend.bytesWired == 0)
            #expect(!backend.pool.writeValidation.isFaulted)
            #expect(engine.admissionForTesting.bytesReserved == 0)
            #expect(engine.admissionForTesting.transientBytesReserved == 0)
        }
        model.allowValidCalls()
        let recovered = try engine.loopForTesting.teacherForcedTop1(
            promptTokens: [1, 2], continuation: [1, 1])
        #expect(recovered == [1, 1])
        #expect(engine.admissionForTesting.bytesReserved == 0)
        #expect(engine.admissionForTesting.transientBytesReserved == 0)
        await engine.shutdown()
    }
}
