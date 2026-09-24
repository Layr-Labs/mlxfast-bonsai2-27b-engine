import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("Complete checkpoint host metadata ownership", .serialized)
struct CompleteCheckpointMetadataOwnershipTests {
    private var kind: CBv2LayerKind {
        .init(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
    }

    private func admission(bytes: Int = 64 << 20) -> AdmissionV2 {
        AdmissionV2(layerKinds: [kind], bytesCapacity: bytes, config: .init(watermarkFraction: 0))
    }

    private func manifest(position: Int = 32) throws -> CBv2CompleteCheckpointManifest {
        .init(identity: .init(modelAggregateHash: "model", promptContractID: "template",
                             buildID: "build", numericsFingerprint: "native-fp16"),
              position: position, chunkSize: 32, prefixTokens: Array(repeating: 7, count: position),
              cacheSalt: "tenant", assistantCodecID: nil,
              tensors: [try .init(role: .keys, layer: 0, shape: [1, 2, position, 64], dtype: .float16)],
              backendLayout: CBv2CompleteCheckpointManifest.pagedLayout)
    }

    @Test("manifest value copies keep one permit after export close and last source release")
    func manifestAliases() throws {
        let admission = admission()
        let original = try manifest()
        let bytes = try CBv2CheckpointManifestMemory.reservationBytes(position: original.position)
        var retained: CBv2CompleteCheckpointManifest?
        var source: CBv2CompleteCheckpointExport?
        func construct() throws {
            let owned = try original.owningMetadata(admission: admission)
            let replanned = try owned.owningMetadata(admission: admission)
            #expect(owned.metadata === replanned.metadata)
            #expect(admission.bytesReserved == bytes)
            source = .init(manifest: owned, arrays: [])
            retained = replanned
        }
        try construct()
        source?.close()
        #expect(admission.bytesReserved == bytes)
        source = nil
        #expect(admission.bytesReserved == bytes)
        #expect(retained?.prefixTokens == original.prefixTokens)
        retained = nil
        #expect(admission.bytesReserved == 0)
    }

    @Test("wire encoding and equality ignore native manifest ownership")
    func ownershipNeutralWireFormat() throws {
        let admission = admission()
        let original = try manifest()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let unownedWire = try encoder.encode(original)
        var owned: CBv2CompleteCheckpointManifest? = try original.owningMetadata(admission: admission)
        #expect(owned == original)
        let ownedWire = try encoder.encode(try #require(owned))
        #expect(ownedWire == unownedWire)
        let decoded = try JSONDecoder().decode(CBv2CompleteCheckpointManifest.self, from: ownedWire)
        #expect(decoded == original && decoded.metadata.permit == nil)
        owned = nil
        #expect(admission.bytesReserved == 0)
    }

    @Test("two historical manifests require simultaneous permits and exact fit refuses a third")
    func simultaneousManifests() throws {
        let first = try manifest(), second = try manifest(position: 64)
        let firstBytes = try CBv2CheckpointManifestMemory.reservationBytes(position: first.position)
        let secondBytes = try CBv2CheckpointManifestMemory.reservationBytes(position: second.position)
        let admission = admission(bytes: firstBytes + secondBytes)
        var a: CBv2CompleteCheckpointManifest? = try first.owningMetadata(admission: admission)
        var b: CBv2CompleteCheckpointManifest? = try second.owningMetadata(admission: admission)
        #expect(a != nil && b != nil && admission.bytesReserved == firstBytes + secondBytes)
        #expect(throws: (any Error).self) { try first.owningMetadata(admission: admission) }
        a = nil
        #expect(admission.bytesReserved == secondBytes)
        b = nil
        #expect(admission.bytesReserved == 0)
    }

    @Test("export refuses metadata before constructing descriptors or page maps")
    func exportRefusesBeforeConstruction() throws {
        let admission = admission(bytes: 0)
        let codec = CBv2CompleteCheckpointCodec(identity: try manifest().identity, layerKinds: [kind],
            recurrentSpec: .init(layers: []), kvDTypes: [.float16], assistant: nil, admission: admission,
            pagedConfig: .init(capacityBytes: 1 << 20, segmentSizeBytes: 64 << 10, layerDTypes: [.float16]))
        let checkpoint = CBv2RecurrentCheckpoint(position: 32, chunkSize: 32, layers: [:], byteCount: 0)
        // Empty recurrent geometry and nil row would fail inside construction.
        // Capacity must win, before either descriptor or page-map construction.
        do {
            _ = try codec.exportPaged(checkpoint: checkpoint, state: [nil],
                                      tokens: Array(repeating: 7, count: 33), cacheSalt: nil)
            Issue.record("metadata permit unexpectedly accepted")
        } catch let error as CBv2KVError {
            guard case .capacityExhausted = error else { throw error }
        }
        #expect(admission.bytesReserved == 0)
        let accepted = self.admission()
        let failing = CBv2CompleteCheckpointCodec(identity: try manifest().identity, layerKinds: [kind],
            recurrentSpec: .init(layers: []), kvDTypes: [.float16], assistant: nil, admission: accepted,
            pagedConfig: codec.pagedConfig)
        #expect(throws: CBv2CompleteCheckpointError.incompatibleCheckpoint) {
            try failing.exportPaged(checkpoint: checkpoint, state: [nil],
                                    tokens: Array(repeating: 7, count: 33), cacheSalt: nil)
        }
        #expect(accepted.bytesReserved == 0, "throwing builder unwinds before its permit")
    }

    @Test("mapped records start only after admission and failed construction returns its permit")
    func pageMapRefusalAndFailure() throws {
        let key = PagedKVGroupKey(kind, dtype: .float16, separateWindow: true)
        let layout = try PagedKVSegmentLayout(pageBytes: 8192, targetBytes: 64 << 10,
                                             maximumBufferBytes: 1 << 20)
        let previous = MLXArray(Int32(0))
        let bytes = try CBv2PagedCheckpointPageMap.allocationBytes(pageCount: 2)
        var began = false
        let refused = admission(bytes: bytes - 1)
        #expect(throws: (any Error).self) {
            try CBv2PagedCheckpointPageMap(key: key, pageSize: 16, position: 32, table: [Int32(0), 0],
                layout: layout, segments: [:], previous: previous, admission: refused,
                beforeAllocation: { began = true })
        }
        #expect(!began && refused.bytesReserved == 0)
        let accepted = admission(bytes: bytes)
        #expect(throws: CBv2CompleteCheckpointError.incompleteTransfer) {
            try CBv2PagedCheckpointPageMap(key: key, pageSize: 16, position: 32, table: [Int32(0), 0],
                layout: layout, segments: [:], previous: previous, admission: accepted,
                beforeAllocation: { began = true; #expect(accepted.bytesReserved == bytes) })
        }
        #expect(began && accepted.bytesReserved == 0)
    }

    @Test("K and V share one exact mapped page permit until both sources close")
    func sharedPageMapLifetime() throws {
        let admission = admission()
        let config = PagedKVPoolConfig(capacityBytes: 16 << 20, segmentSizeBytes: 64 << 10,
                                        layerDTypes: [.float16])
        let backend = try PagedKVBackend(layerKinds: [kind], config: config)
        backend.pool.bindAdmission(admission)
        try admission.reserve(id: .init(1), additionalTokens: 32)
        let rows = try backend.makeSequenceState(layerKinds: [kind], promptLength: 32, maxLength: 32)
        let row = try #require(rows[0] as? PagedSequenceKV)
        row.write(keys: MLXArray.zeros([2, 32, 64], dtype: .float16),
                  values: MLXArray.zeros([2, 32, 64], dtype: .float16))
        try withError { eval(backend.pool.group(row.groupKey).writeFence) }
        let baseline = admission.bytesReserved
        let bytes = try CBv2PagedCheckpointPageMap.allocationBytes(pageCount: 2)
        var keys: CBv2PagedCheckpointTensorSource?, values: CBv2PagedCheckpointTensorSource?
        func construct() throws {
            let map = try CBv2PagedCheckpointPageMap(row: row, position: 32, admission: admission)
            keys = try .init(pageMap: map, values: false)
            values = try .init(pageMap: map, values: true)
        }
        try construct()
        #expect(admission.bytesReserved == baseline + bytes)
        keys?.close()
        #expect(admission.bytesReserved == baseline + bytes)
        values?.close()
        #expect(admission.bytesReserved == baseline)
        keys = nil
        values = nil
        backend.release(rows)
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 0 && backend.bytesWired == 0)
    }
}
