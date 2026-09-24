import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Paged complete checkpoint codec", .serialized)
struct PagedCompleteCheckpointCodecTests {
    private struct Fixture {
        let backend: PagedKVBackend
        let admission: AdmissionV2
        let codec: CBv2CompleteCheckpointCodec
        let manifest: CBv2CompleteCheckpointManifest
        let request: CBv2Request
        let bytes: [Data]
    }

    private func fixture(dtype: DType = .bfloat16) throws -> Fixture {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2,
                                  queryHeads: 4, modelLayerIndex: 1)]
        let config = PagedKVPoolConfig(capacityBytes: 64 << 20, maxPrefillChunk: 64,
            segmentSizeBytes: 64 << 10, layerDTypes: [dtype])
        let backend = try PagedKVBackend(layerKinds: kinds, config: config)
        let spec = CBv2RecurrentStateSpec(layers: [.init(modelLayerIndex: 0,
            convShape: [1, 2, 2], convDType: dtype, ssmShape: [1, 1, 2, 2], ssmDType: .float32)])
        let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: 64 << 20,
            config: .init(watermarkFraction: 0, elementBytes: dtype.size,
                          fixedBytesPerRequest: try spec.fixedBytesPerRequest()),
            residency: CBv2PagedKVResidency(config: config))
        backend.pool.bindAdmission(admission)
        let identity = CBv2CompleteCheckpointIdentity(modelAggregateHash: "native-model",
            promptContractID: "template", buildID: "test-build", numericsFingerprint: "native-\(dtype)")
        let codec = CBv2CompleteCheckpointCodec(identity: identity, layerKinds: kinds,
            recurrentSpec: spec, kvDTypes: [dtype], assistant: nil, admission: admission, pagedConfig: config)
        let chunk = max(32, CBv2AttentionV1.queryBlockSize)
        let request = CBv2Request(id: .init(7), promptTokens: Array(repeating: 1, count: chunk + 3),
            maxTokens: 512, cacheSalt: "tenant", prefixCacheReceiptID: .init(1007))
        let manifest = CBv2CompleteCheckpointManifest(identity: identity, position: chunk, chunkSize: chunk,
            prefixTokens: Array(request.promptTokens.prefix(chunk)), cacheSalt: request.cacheSalt,
            assistantCodecID: nil, tensors: try codec.tensorDescriptors(position: chunk),
            backendLayout: CBv2CompleteCheckpointManifest.pagedLayout)
        let bytes = manifest.tensors.enumerated().map { index, descriptor in
            Data((0 ..< descriptor.byteCount).map { UInt8(truncatingIfNeeded: ($0 * 73) ^ ($0 >> 3) ^ (index * 19)) })
        }
        return Fixture(backend: backend, admission: admission, codec: codec,
                       manifest: manifest, request: request, bytes: bytes)
    }

    private func plan(_ fixture: Fixture) throws -> CBv2CompleteCheckpointImportPlan {
        try fixture.codec.plan(manifest: fixture.manifest, request: fixture.request,
            minimumChunkSize: fixture.manifest.chunkSize, maximumChunkSize: fixture.manifest.chunkSize)
    }

    private func fill(_ sink: CBv2CompleteCheckpointImport, fixture: Fixture) throws {
        for (index, bytes) in fixture.bytes.enumerated() {
            var offset = 0
            while offset < bytes.count {
                let count = min(1024, bytes.count - offset)
                try sink.appendSegment(tensorIndex: index, byteOffset: offset,
                                       data: bytes.subdata(in: offset ..< offset + count))
                offset += count
            }
        }
    }

    @Test("Actual-M destinations move into full-N request ownership without byte conversion",
          arguments: [DType.bfloat16, .float16, .float32])
    func roundTrip(dtype: DType) throws {
        let fixture = try fixture(dtype: dtype)
        func exercise() throws {
            let plan = try plan(fixture)
            let metadataBytes = try #require(plan.manifest.metadata.permit).bytes
            let nativePlan = try #require(plan.pagedStoragePlan)
            let auxiliaryBytes = try fixture.manifest.tensors.dropFirst(2).reduce(0) {
                try $0 + Memory.allocationFootprintUpperBound(byteCount: $1.byteCount)
            }
            #expect(plan.nativeDestinationBytes == nativePlan.nativeBytes + auxiliaryBytes)
            #expect(plan.nativeDestinationBytes < fixture.admission.allocatedBytes(forTokens: plan.maximumSequenceLength))
            #expect(!plan.usesProcessMemoryOwner)
            let released = PagedCodecCounter()
            let sink = try plan.allocate { released.increment() }
            #expect(fixture.admission.bytesReserved == metadataBytes + plan.nativeDestinationBytes + plan.scratchBytes
                    + CBv2CompleteCheckpointManifest.maximumProviderScratchBytes)
            try fill(sink, fixture: fixture)
            let staged = try sink.finish()
            #expect(staged.nativeDestinationBytes <= plan.nativeDestinationBytes)
            #expect(staged.nativeDestinationBytes >= fixture.manifest.tensors.dropFirst(2).reduce(0) { $0 + $1.byteCount })
            #expect(fixture.admission.bytesReserved == metadataBytes + staged.nativeDestinationBytes + plan.scratchBytes
                    + CBv2CompleteCheckpointManifest.maximumProviderScratchBytes)
            sink.close()
            var recurrent: CBv2RecurrentCheckpoint?
            var active = try staged.consumePreparedState { prepared in
                let frame = try #require(prepared.pagedFrame)
                prepared.pagedFrame = nil
                let adoption = try fixture.backend.pool.importCheckpoint(frame, admission: fixture.admission,
                    requestID: fixture.request.id, layerKinds: fixture.codec.layerKinds,
                    maximumTokens: plan.maximumSequenceLength)
                let state = try adoption.moveToActiveRequest { auxiliary in
                    recurrent = try fixture.codec.recurrentCheckpoint(manifest: fixture.manifest, auxiliary: auxiliary)
                }
                adoption.release() // Success disarmed this temporary owner's refund.
                #expect(fixture.admission.bytesReserved > 0 && released.value == 0)
                #expect(throws: CBv2CompleteCheckpointError.closed) {
                    try adoption.moveToActiveRequest { _ in }
                }
                return state
            }
            #expect(released.value == 1)
            #expect(fixture.admission.transientBytesReserved == metadataBytes)
            staged.close()
            #expect(released.value == 1)
            let source = try fixture.codec.export(checkpoint: #require(recurrent), state: active,
                tokens: fixture.request.promptTokens, cacheSalt: fixture.request.cacheSalt)
            #expect(source.manifest == fixture.manifest)
            for (index, expected) in fixture.bytes.enumerated() {
                var actual = Data(), offset = 0
                while offset < expected.count {
                    let part = try source.readSegment(tensorIndex: index, byteOffset: offset, maximumBytes: 1024)
                    actual.append(part)
                    offset += part.count
                }
                #expect(actual == expected, "NaNs, signed zeros and every native bit survive unchanged")
            }
            source.close()
            #expect(throws: CBv2CompleteCheckpointError.closed) {
                try source.readSegment(tensorIndex: 0, byteOffset: 0, maximumBytes: 4)
            }
            recurrent = nil
            fixture.backend.release(active)
            active.removeAll()
            fixture.admission.releaseAll(id: fixture.request.id)
            #expect(fixture.admission.bytesReserved == metadataBytes + (source.manifest.metadata.permit?.bytes ?? 0)
                    && fixture.backend.bytesWired == 0)
        }
        try exercise()
        #expect(fixture.admission.bytesReserved == 0, "last plan/staged/export manifest owner retired")
    }

    @Test func destinationBoundRefusesBeforeAnyNativeAllocation() throws {
        let fixture = try fixture()
        let plan = try plan(fixture)
        let before = fixture.admission.bytesReserved
        let total = plan.nativeDestinationBytes + plan.scratchBytes
            + CBv2CompleteCheckpointManifest.maximumProviderScratchBytes
        fixture.admission.updateBytesCapacity(before + total - 1)
        var evaluations = 0
        plan.evaluateDestinations = { _ in evaluations += 1 }
        let released = PagedCodecCounter()
        #expect(throws: CBv2KVError.self) { try plan.allocate { released.increment() } }
        #expect(evaluations == 0 && released.value == 1)
        #expect(fixture.admission.bytesReserved == before && fixture.backend.bytesWired == 0)
        #expect(plan.scratchBytes == (try Memory.allocationFootprintUpperBound(byteCount: 2))
            + (try Memory.allocationFootprintUpperBound(byteCount: 4)))
    }

    @Test("Backend identity and dtype mismatches reject before allocations")
    func incompatibleManifest() throws {
        let fixture = try fixture()
        let contiguous = CBv2CompleteCheckpointCodec(identity: fixture.codec.identity,
            layerKinds: fixture.codec.layerKinds, recurrentSpec: fixture.codec.recurrentSpec,
            kvDTypes: fixture.codec.kvDTypes, assistant: nil, admission: fixture.admission)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try contiguous.plan(manifest: fixture.manifest, request: fixture.request,
                minimumChunkSize: fixture.manifest.chunkSize, maximumChunkSize: fixture.manifest.chunkSize)
        }
        let other = try self.fixture(dtype: .float32)
        let wrongType = CBv2CompleteCheckpointManifest(identity: fixture.manifest.identity,
            position: fixture.manifest.position, chunkSize: fixture.manifest.chunkSize,
            prefixTokens: fixture.manifest.prefixTokens, cacheSalt: fixture.manifest.cacheSalt,
            assistantCodecID: nil, tensors: other.manifest.tensors,
            backendLayout: CBv2CompleteCheckpointManifest.pagedLayout)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try fixture.codec.plan(manifest: wrongType, request: fixture.request,
                minimumChunkSize: fixture.manifest.chunkSize, maximumChunkSize: fixture.manifest.chunkSize)
        }
        let legacy = CBv2CompleteCheckpointManifest(identity: fixture.manifest.identity,
            position: fixture.manifest.position, chunkSize: fixture.manifest.chunkSize,
            prefixTokens: fixture.manifest.prefixTokens, cacheSalt: fixture.manifest.cacheSalt,
            assistantCodecID: nil, tensors: fixture.manifest.tensors)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try fixture.codec.plan(manifest: legacy, request: fixture.request,
                minimumChunkSize: fixture.manifest.chunkSize, maximumChunkSize: fixture.manifest.chunkSize)
        }
        #expect(fixture.admission.bytesReserved == 0 && fixture.backend.bytesWired == 0)
    }

    @Test("Window and borrowed layers remain outside the complete codec contract", arguments: [false, true])
    func unsupportedKinds(borrowed: Bool) throws {
        let fixture = try fixture()
        var kinds = fixture.codec.layerKinds
        if borrowed { kinds[0].sharesKVWithLayer = 0 }
        else { kinds[0].attention = .slidingWindow(64) }
        let codec = CBv2CompleteCheckpointCodec(identity: fixture.codec.identity, layerKinds: kinds,
            recurrentSpec: fixture.codec.recurrentSpec, kvDTypes: fixture.codec.kvDTypes,
            assistant: nil, admission: fixture.admission, pagedConfig: fixture.backend.pool.config)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try codec.tensorDescriptors(position: fixture.manifest.position)
        }
        #expect(fixture.admission.bytesReserved == 0 && fixture.backend.bytesWired == 0)
    }

    @Test("Native and auxiliary allocation faults drop arrays before returning stage capacity",
          arguments: [1, 2])
    func allocationFailure(failAt: Int) throws {
        let fixture = try fixture()
        func exercise() throws {
            let plan = try plan(fixture)
            let metadataBytes = try #require(plan.manifest.metadata.permit).bytes
            let weakOwners = PagedCodecWeakOwners()
            var calls = 0
            plan.evaluateDestinations = { arrays in
                try withError { eval(arrays) }
                weakOwners.append(arrays)
                calls += 1
                if calls == failAt { throw CBv2CompleteCheckpointError.allocationFailed }
            }
            let released = PagedCodecCounter()
            #expect(throws: (any Error).self) {
                try plan.allocate {
                    #expect(weakOwners.isEmpty)
                    released.increment()
                }
            }
            #expect(calls == failAt && released.value == 1)
            #expect(fixture.admission.bytesReserved == metadataBytes && weakOwners.isEmpty)
        }
        try exercise()
        #expect(fixture.admission.bytesReserved == 0, "last plan/staged/export manifest owner retired")
    }

    @Test("Auxiliary eval-then-throw drains before refund and preserves native error priority",
          arguments: [0, 1, 2])
    func auxiliaryEvaluationFailure(failureKind: Int) throws {
        enum Injected: Error { case afterEvaluation }
        let fixture = try fixture()
        let plan = try plan(fixture)
        let metadataBytes = try #require(plan.manifest.metadata.permit).bytes
        let weakOwners = PagedCodecWeakOwners()
        var reachedAuxiliary = false
        plan.evaluateDestinations = { arrays in
            eval(arrays)
            weakOwners.append(arrays)
            // Page buffers are evaluated individually; this fixture's two
            // recurrent tensors are the one auxiliary destination batch.
            if arrays.count == 2 {
                reachedAuxiliary = true
                if failureKind > 0 {
                    // The same deterministic broadcast error used by MLX's
                    // own handler tests records a callback fault in this scope.
                    let a = MLXArray(0 ..< 10, [2, 5])
                    let b = MLXArray(0 ..< 15, [3, 5])
                    _ = a + b
                }
                if failureKind == 2 { throw MLXError.caught("primary injected native error") }
                throw Injected.afterEvaluation
            }
        }
        let released = PagedCodecCounter()
        do {
            _ = try plan.allocate { #expect(weakOwners.isEmpty); released.increment() }
            Issue.record("injected auxiliary failure should refuse the import")
        } catch {
            if failureKind == 2 {
                if case MLXError.caught(let message) = error {
                    #expect(message == "primary injected native error")
                } else { Issue.record("already-thrown native error must retain priority") }
            } else if failureKind == 1 { #expect(error is MLXError) }
            else { #expect(error is Injected) }
        }
        #expect(reachedAuxiliary && released.value == 1 && weakOwners.isEmpty)
        #expect(fixture.admission.bytesReserved == metadataBytes && fixture.backend.bytesWired == 0)
    }

    @Test("Close racing consumption cannot refund an in-flight adoption payload")
    func closeDuringConsumption() throws {
        let fixture = try fixture()
        func exercise() throws {
            let plan = try plan(fixture)
            let metadataBytes = try #require(plan.manifest.metadata.permit).bytes
            let weakOwners = PagedCodecWeakOwners()
            plan.evaluateDestinations = { arrays in try withError { eval(arrays) }; weakOwners.append(arrays) }
            let released = PagedCodecCounter()
            let sink = try plan.allocate { #expect(weakOwners.isEmpty); released.increment() }
            try fill(sink, fixture: fixture)
            let staged = try sink.finish()
            sink.close()
            try staged.consumePreparedState { _ in
                DispatchQueue.global().sync { staged.close() }
                #expect(released.value == 0 && !weakOwners.isEmpty && fixture.admission.bytesReserved > 0)
            }
            #expect(released.value == 1 && weakOwners.isEmpty && fixture.admission.bytesReserved == metadataBytes)
            #expect(throws: CBv2CompleteCheckpointError.closed) { try staged.consumePreparedState { _ in } }
        }
        try exercise()
        #expect(fixture.admission.bytesReserved == 0, "last plan/staged/export manifest owner retired")
    }

    @Test("Failed recurrent restoration refunds pages only after candidate aliases drain")
    func failedRestore() throws {
        let fixture = try fixture()
        func exercise() throws {
            let plan = try plan(fixture)
            let metadataBytes = try #require(plan.manifest.metadata.permit).bytes
            let weakOwners = PagedCodecWeakOwners()
            plan.evaluateDestinations = { arrays in try withError { eval(arrays) }; weakOwners.append(arrays) }
            let released = PagedCodecCounter()
            let sink = try plan.allocate { #expect(weakOwners.isEmpty); released.increment() }
            try fill(sink, fixture: fixture)
            let staged = try sink.finish()
            sink.close()
            #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
                try staged.consumePreparedState { prepared in
                    let frame = try #require(prepared.pagedFrame)
                    prepared.pagedFrame = nil
                    let adoption = try fixture.backend.pool.importCheckpoint(frame, admission: fixture.admission,
                        requestID: fixture.request.id, layerKinds: fixture.codec.layerKinds,
                        maximumTokens: plan.maximumSequenceLength)
                    return try adoption.moveToActiveRequest { auxiliary in
                        var candidate: CBv2RecurrentCheckpoint? = try fixture.codec.recurrentCheckpoint(
                            manifest: fixture.manifest, auxiliary: auxiliary)
                        #expect(candidate != nil && fixture.admission.bytesReserved > 0)
                        candidate = nil
                        throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                    }
                }
            }
            #expect(released.value == 1 && weakOwners.isEmpty)
            #expect(fixture.admission.bytesReserved == metadataBytes && fixture.backend.bytesWired == 0)
            staged.close()
        }
        try exercise()
        #expect(fixture.admission.bytesReserved == 0, "last plan/staged/export manifest owner retired")
    }

}

private final class PagedCodecCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var value: Int { lock.withLock { count } }
    func increment() { lock.withLock { count += 1 } }
}

private final class PagedCodecWeakOwners: @unchecked Sendable {
    private final class WeakOwner {
        weak var value: MLXArray?
        init(_ value: MLXArray) { self.value = value }
    }
    private var owners: [WeakOwner] = []
    func append(_ arrays: [MLXArray]) { owners.append(contentsOf: arrays.map(WeakOwner.init)) }
    var isEmpty: Bool { owners.allSatisfy { $0.value == nil } }
}
