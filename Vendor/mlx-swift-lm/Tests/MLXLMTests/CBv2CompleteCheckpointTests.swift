import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

private final class CheckpointCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
    func increment() { lock.lock(); count += 1; lock.unlock() }
}

final class CBv2CompleteCheckpointTests: XCTestCase {
    private let identity = CBv2CompleteCheckpointIdentity(
        modelAggregateHash: "verified-model", promptContractID: "template",
        buildID: "binary-native-codec", numericsFingerprint: "test-fp32")
    private var chunk: Int { max(32, CBv2AttentionV1.queryBlockSize) }
    private var kinds: [CBv2LayerKind] {
        [.init(attention: .full, headDim: 3, kvHeads: 2, queryHeads: 2, modelLayerIndex: 3)]
    }
    private var spec: CBv2RecurrentStateSpec {
        .init(layers: [.init(
            modelLayerIndex: 0, convShape: [1, 2, 2], convDType: .float32,
            ssmShape: [1, 1, 2, 2], ssmDType: .float32)])
    }

    private func fixture() throws -> (
        CBv2CompleteCheckpointCodec, CBv2CompleteCheckpointExport, CBv2Request, [UInt32]
    ) {
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 64 << 20,
            config: .init(watermarkFraction: 0, elementBytes: 4))
        let codec = CBv2CompleteCheckpointCodec(
            identity: identity, layerKinds: kinds, recurrentSpec: spec,
            kvDTypes: [.float32], assistant: nil, admission: admission)
        let pattern: [Float] = [0, -0.0, .infinity, -.infinity, Float(bitPattern: 0x7fc01234), 0.125]
        let data = (0 ..< 2 * (chunk + 7) * 3).map { pattern[$0 % pattern.count] }
        let backing = MLXArray(data).reshaped([1, 2, chunk + 7, 3])
        let conv = MLXArray([Float(1), 2, 3, 4]).reshaped([1, 2, 2])
        let ssm = MLXArray([Float(5), 6, 7, 8]).reshaped([1, 1, 2, 2])
        eval(backing, conv, ssm)
        let prefix = backing[.ellipsis, ..<chunk, 0...]
        let checkpoint = CBv2RecurrentCheckpoint(
            position: chunk, chunkSize: chunk,
            layers: [0: .init(conv: conv, ssm: ssm)], byteCount: 32)
        let request = CBv2Request(
            id: .init(7), promptTokens: Array(repeating: 1, count: chunk + 3), maxTokens: 4,
            cacheSalt: "tenant-a", prefixCacheReceiptID: .init(1007))
        let source = try codec.export(
            checkpoint: checkpoint, kv: [(keys: backing, values: backing, offset: chunk + 7)],
            tokens: request.promptTokens, cacheSalt: request.cacheSalt)
        return (codec, source, request, prefix.asArray(Float.self).map(\.bitPattern))
    }

    private func copy(_ source: CBv2CompleteCheckpointExport, into sink: CBv2CompleteCheckpointImport) throws {
        for (index, tensor) in source.manifest.tensors.enumerated() {
            var offset = 0
            while offset < tensor.byteCount {
                let data = try source.readSegment(tensorIndex: index, byteOffset: offset, maximumBytes: 20)
                try sink.appendSegment(tensorIndex: index, byteOffset: offset, data: data)
                offset += data.count
            }
        }
    }

    func testStridedNativeBitsTransferIntoFinalRowsWithoutSecondPrefixCopy() throws {
        func exercise() throws -> AdmissionV2 {
            let (codec, source, request, expected) = try fixture()
            let plan = try codec.plan(
                manifest: source.manifest, request: request, minimumChunkSize: chunk, maximumChunkSize: chunk)
            let kvBound = try Memory.allocationFootprintUpperBound(byteCount: (chunk + 7) * 2 * 3 * 4)
            let recurrentBound = try Memory.allocationFootprintUpperBound(byteCount: 16)
            XCTAssertEqual(plan.nativeDestinationBytes, 2 * kvBound + 2 * recurrentBound)
            let counter = CheckpointCounter()
            let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 64 << 20, kvDType: .float32))
            let sink = try plan.allocate { counter.increment() }
            XCTAssertGreaterThan(codec.admission.bytesReserved, plan.nativeDestinationBytes)
            try copy(source, into: sink)
            let staged = try sink.finish()
            sink.close()
            XCTAssertEqual(counter.value, 0)
            try codec.admission.reserve(id: request.id, additionalTokens: chunk + 7)
            var active = try staged.consumePreparedState { prepared in
                try backend.adoptPreparedCheckpoint(prepared.state)
                let row = try XCTUnwrap(prepared.state.first!)
                XCTAssertEqual(row.byteCount, (chunk + 7) * 2 * 3 * 4 * 2)
                XCTAssertEqual(row.snapshot().keys.asArray(Float.self).map(\.bitPattern), expected)
                XCTAssertEqual(row.snapshot().values.asArray(Float.self).map(\.bitPattern), expected)
                XCTAssertEqual(prepared.checkpoint?.layers[0]?.ssm?.asArray(Float.self), [5, 6, 7, 8])
                XCTAssertEqual(counter.value, 0, "stage remains charged through backend registration")
                return prepared.state
            }
            XCTAssertEqual(counter.value, 1)
            XCTAssertGreaterThan(backend.bytesReserved, 0)
            staged.close()
            XCTAssertEqual(counter.value, 1)
            XCTAssertThrowsError(try staged.consumePreparedState { _ in () })
            backend.release(active)
            active.removeAll()
            codec.admission.releaseAll(id: request.id)
            XCTAssertEqual(codec.admission.bytesReserved, try XCTUnwrap(source.manifest.metadata.permit).bytes)
            source.close()
            XCTAssertThrowsError(try source.readSegment(tensorIndex: 0, byteOffset: 0, maximumBytes: 4))
            return codec.admission
        }
        let admission = try exercise()
        XCTAssertEqual(admission.bytesReserved, 0, "final manifest owner returned its metadata permit")
    }

    func testTruncatedAndOutOfOrderTransferClosesExactlyOnce() throws {
        func exercise() throws -> AdmissionV2 {
            let (codec, source, request, _) = try fixture()
            let plan = try codec.plan(
                manifest: source.manifest, request: request, minimumChunkSize: chunk, maximumChunkSize: chunk)
            let counter = CheckpointCounter()
            let sink = try plan.allocate { counter.increment() }
            let bytes = try source.readSegment(tensorIndex: 0, byteOffset: 0, maximumBytes: 4)
            XCTAssertThrowsError(try sink.appendSegment(tensorIndex: 1, byteOffset: 0, data: bytes))
            XCTAssertThrowsError(try sink.appendSegment(tensorIndex: 0, byteOffset: 4, data: bytes))
            try sink.appendSegment(tensorIndex: 0, byteOffset: 0, data: bytes)
            XCTAssertThrowsError(try sink.appendSegment(tensorIndex: 0, byteOffset: 0, data: bytes))
            XCTAssertThrowsError(try sink.finish())
            sink.close()
            sink.close()
            XCTAssertEqual(counter.value, 1)
            XCTAssertEqual(codec.admission.bytesReserved, try XCTUnwrap(source.manifest.metadata.permit).bytes)
            XCTAssertThrowsError(try sink.finish())
            source.close()
            return codec.admission
        }
        let admission = try exercise()
        XCTAssertEqual(admission.bytesReserved, 0, "final manifest owner returned its metadata permit")
    }

    func testAllocationFaultReleasesEngineAndProviderReservations() throws {
        func exercise() throws -> AdmissionV2 {
            let (codec, source, request, _) = try fixture()
            let plan = try codec.plan(
                manifest: source.manifest, request: request, minimumChunkSize: chunk, maximumChunkSize: chunk)
            plan.evaluateDestinations = { _ in throw MLXError.caught("injected destination allocation failure") }
            let counter = CheckpointCounter()
            XCTAssertThrowsError(try plan.allocate { counter.increment() })
            XCTAssertEqual(counter.value, 1)
            XCTAssertEqual(codec.admission.bytesReserved, try XCTUnwrap(source.manifest.metadata.permit).bytes)
            source.close()
            return codec.admission
        }
        let admission = try exercise()
        XCTAssertEqual(admission.bytesReserved, 0, "final manifest owner returned its metadata permit")
    }

    func testManifestMismatchIsRejectedBeforeAnyAllocation() throws {
        func exercise() throws -> AdmissionV2 {
            let (codec, source, request, _) = try fixture()
            let encoded = try JSONEncoder().encode(source.manifest)
            func mutated(_ mutate: (inout [String: Any]) -> Void) throws -> CBv2CompleteCheckpointManifest {
                var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
                mutate(&object)
                return try JSONDecoder().decode(
                    CBv2CompleteCheckpointManifest.self, from: JSONSerialization.data(withJSONObject: object))
            }
            let cases = try [
                mutated { $0["schemaVersion"] = 999 },
                mutated { $0["cacheSalt"] = "tenant-b" },
                mutated { $0["assistantCodecID"] = "unrecognized" },
                mutated { $0["prefixTokens"] = Array(repeating: 2, count: chunk) },
                mutated { object in
                    var tensors = object["tensors"] as! [[String: Any]]
                    tensors[0]["shape"] = [1, 2, Int.max, 3]
                    object["tensors"] = tensors
                },
                mutated { object in
                    var tensors = object["tensors"] as! [[String: Any]]
                    tensors[0]["byteCount"] = 1
                    object["tensors"] = tensors
                },
                mutated { object in
                    let tensors = object["tensors"] as! [[String: Any]]
                    object["tensors"] = tensors + [tensors[0]]
                },
            ]
            for manifest in cases {
                XCTAssertThrowsError(try codec.plan(
                    manifest: manifest, request: request, minimumChunkSize: chunk, maximumChunkSize: chunk))
            }
            XCTAssertEqual(codec.admission.bytesReserved, try XCTUnwrap(source.manifest.metadata.permit).bytes)
            source.close()
            return codec.admission
        }
        let admission = try exercise()
        XCTAssertEqual(admission.bytesReserved, 0, "final manifest owner returned its metadata permit")
    }

    func testTransientAndDetachedLeasesEnforceCeilingWithoutOwningRequestIDs() throws {
        let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: 64, config: .init(watermarkFraction: 0))
        let lease = try admission.reserveTransient(bytes: 64)
        XCTAssertEqual(admission.bytesReserved, 64)
        XCTAssertEqual(admission.nonBackendBytesReserved, 64)
        XCTAssertEqual(admission.targetBytesReserved(partitionedBy: []).unmaterialized, 0)
        XCTAssertThrowsError(try admission.reserveTransient(bytes: 1))
        lease.release()
        lease.release()
        XCTAssertEqual(admission.bytesReserved, 0)
        try admission.reserve(id: .init(7), additionalTokens: 1)
        let oldBytes = admission.bytesReserved
        let retired = admission.detachReservation(id: .init(7))
        XCTAssertEqual(admission.bytesReserved, oldBytes)
        XCTAssertEqual(admission.targetBytesReserved(partitionedBy: []).unmaterialized, 0)
        try admission.reserve(id: .init(7), additionalTokens: 1)
        XCTAssertEqual(admission.bytesReserved, oldBytes * 2)
        retired.release()
        retired.release()
        XCTAssertEqual(admission.bytesReserved, oldBytes, "old completion preserves the reused ID's owner")
        admission.releaseAll(id: .init(7))
        XCTAssertEqual(admission.bytesReserved, 0)
    }
}
