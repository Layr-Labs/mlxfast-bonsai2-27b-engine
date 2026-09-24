import CryptoKit
import Foundation
import MLX
import MLXRandom

@_spi(Diagnostics) extension PagedDecodeProfiler {
    /// A bounded, synthetic attention-only experiment. This does not load a
    /// model or measure end-to-end decode or pure Metal kernel duration.
    public struct SegmentMetadataConfiguration: Codable, Sendable {
        public var owners: Int
        public var initialOffset: Int
        public var warmup: Int
        public var steps: Int
        public var repetitions: Int

        public init(owners: Int = 10, initialOffset: Int = 5584, warmup: Int = 8,
                    steps: Int = 64, repetitions: Int = 3) {
            self.owners = owners
            self.initialOffset = initialOffset
            self.warmup = warmup
            self.steps = steps
            self.repetitions = repetitions
        }

        public func validate() throws {
            guard (1...10).contains(owners), (1...8192).contains(initialOffset),
                  (1...16).contains(warmup), warmup < initialOffset,
                  (1...128).contains(steps), (1...3).contains(repetitions) else {
                throw SegmentMetadataError.invalidConfiguration
            }
        }
    }

    public enum SegmentMetadataError: Error {
        case invalidConfiguration
        case unexpectedGeometry
        case outputOrHistoryMismatch(repetition: Int)
    }

    public struct SegmentMetadataReport: Codable, Sendable {
        public let scope: String
        public let timingDefinition: String
        public let configuration: SegmentMetadataConfiguration
        public let dtype: String
        public let batch: Int
        public let queryHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let scale: Float
        public let pageSize: Int
        public let segmentTargetBytes: Int
        public let capacityBytes: Int
        public let arms: [SegmentMetadataArm]
        public let allOutputAndFullHistoryDigestsEqual: Bool
    }

    public struct SegmentMetadataArm: Codable, Sendable {
        public struct Step: Codable, Sendable {
            public let offsetAfter: Int
            public let hostConstructionMs: Double
            public let fencedEvaluationMs: Double
            public let wholeStepMs: Double
        }
        public struct Counts: Codable, Sendable {
            public let owner: Int
            public let hits: Int
            public let rebuilds: Int
            public let bypasses: Int
            public let keyNanoseconds: UInt64
            public let preparationNanoseconds: UInt64
        }
        public struct Segment: Codable, Sendable {
            public let id: Int
            public let pageStart: Int
            public let pageEnd: Int
            public let valueOffset: Int
            public let logicalBytes: Int
            public let allocatedBytes: Int
        }
        public struct Geometry: Codable, Sendable, Equatable {
            public struct Bucket: Codable, Sendable, Equatable {
                public let segmentIDs: [Int]
                public let bindingClass: Int
                public let workCount: Int
            }
            public let partitionTokens: Int
            public let maxPartitions: Int
            public let buckets: [Bucket]
        }
        public struct GeometryEvent: Codable, Sendable {
            public let owner: Int
            public let offsetAfter: Int
            public let geometry: Geometry
        }
        public let repetition: Int
        public let mode: String
        public let residentOwners: Int
        public let segments: [Segment]
        public let geometryEvents: [GeometryEvent]
        public let steps: [Step]
        public let counts: [Counts]
        /// Native BF16 bytes after the normal eval/fence; one digest per owner
        /// per measured step. Hashing and all snapshots are outside timing.
        public let outputSHA256: [[String]]
        public let fullHistoryKeySHA256: [String]
        public let fullHistoryValueSHA256: [String]
    }

    public static func measureSegmentMetadata(
        _ configuration: SegmentMetadataConfiguration = .init()
    ) throws -> SegmentMetadataReport {
        try configuration.validate()
        let inputs = SegmentMetadataInputs(configuration)
        var arms: [SegmentMetadataArm] = []
        for repetition in 0..<configuration.repetitions {
            // Reverse the middle pair to expose ordering rather than hide it.
            let order = repetition % 2 == 0 ? [false, true] : [true, false]
            var pair: [SegmentMetadataArm] = []
            for fresh in order {
                pair.append(try segmentMetadataArm(configuration, inputs: inputs,
                    repetition: repetition, fresh: fresh))
            }
            guard pair[0].outputSHA256 == pair[1].outputSHA256,
                  pair[0].fullHistoryKeySHA256 == pair[1].fullHistoryKeySHA256,
                  pair[0].fullHistoryValueSHA256 == pair[1].fullHistoryValueSHA256 else {
                throw SegmentMetadataError.outputOrHistoryMismatch(repetition: repetition)
            }
            arms.append(contentsOf: pair)
        }
        return SegmentMetadataReport(
            scope: "Synthetic attention-only segmented production dispatch; not full-model decode or pure kernel time.",
            timingDefinition: "Host construction includes same-row binding, metadata clearing in the fresh arm and opt-in statistics. Fenced evaluation includes MLX encoding, submission and stream completion. Whole step is their sum. Setup, warmup, geometry inspection and native-byte hashing are excluded.",
            configuration: configuration, dtype: "bfloat16", batch: 1, queryHeads: 16,
            kvHeads: 2, headDim: 256, scale: 0.0625, pageSize: 16,
            segmentTargetBytes: 64 << 20, capacityBytes: 256 << 20,
            arms: arms, allOutputAndFullHistoryDigestsEqual: true)
    }
}

private struct SegmentMetadataInputs {
    let prefixKeys: [MLXArray]
    let prefixValues: [MLXArray]
    let queries: [[MLXArray]]
    let keys: [[MLXArray]]
    let values: [[MLXArray]]
    let zero = MLXArray(Float(0)).asType(.bfloat16)

    init(_ c: PagedDecodeProfiler.SegmentMetadataConfiguration) {
        func tile(_ shape: [Int], seed: UInt64) -> MLXArray {
            MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(.bfloat16)
        }
        prefixKeys = (0..<c.owners).map { tile([2, c.initialOffset - c.warmup, 256], seed: UInt64(100 + $0)) }
        prefixValues = (0..<c.owners).map { tile([2, c.initialOffset - c.warmup, 256], seed: UInt64(200 + $0)) }
        func tiles(heads: Int, base: Int) -> [[MLXArray]] {
            (0..<(c.warmup + c.steps)).map { step in
                (0..<c.owners).map { owner in
                    tile([1, heads, 1, 256], seed: UInt64(base + step * c.owners + owner))
                }
            }
        }
        queries = tiles(heads: 16, base: 1000)
        keys = tiles(heads: 2, base: 10000)
        values = tiles(heads: 2, base: 20000)
        eval(prefixKeys + prefixValues + queries.flatMap { $0 }
            + keys.flatMap { $0 } + values.flatMap { $0 } + [zero])
        StreamOrDevice.default.stream.synchronize()
    }
}

private extension PagedDecodeProfiler {
    static func segmentMetadataArm(
        _ c: SegmentMetadataConfiguration, inputs: SegmentMetadataInputs,
        repetition: Int, fresh: Bool
    ) throws -> SegmentMetadataArm {
        let kind = CBv2LayerKind(attention: .full, headDim: 256, kvHeads: 2, queryHeads: 16)
        let kinds = Array(repeating: kind, count: c.owners)
        let backend = try PagedKVBackend(layerKinds: kinds, config: .init(
            capacityBytes: 256 << 20, dtype: .bfloat16, maxPrefillChunk: 512,
            nominalMaxSequenceLength: c.initialOffset + c.steps, segmentSizeBytes: 64 << 20))
        let state = try backend.makeSequenceState(layerKinds: kinds, promptLength: 0,
                                                  maxLength: c.initialOffset + c.steps)
        defer { backend.release(state) }
        let rows = state.compactMap { $0 as? PagedSequenceKV }
        let layers = backend.makeLayerCaches()
        guard rows.count == c.owners, layers.count == c.owners else {
            throw SegmentMetadataError.unexpectedGeometry
        }
        defer { layers.forEach { $0.setRows([]) } }
        let group = backend.pool.group(rows[0].groupKey)
        for owner in 0..<c.owners {
            rows[owner].write(keys: inputs.prefixKeys[owner], values: inputs.prefixValues[owner])
            layers[owner].setRows([rows[owner]])
        }
        eval(group.writeFence)
        StreamOrDevice.default.stream.synchronize()
        func construct(_ step: Int) -> [MLXArray] {
            var carry = inputs.zero
            var outputs: [MLXArray] = []
            outputs.reserveCapacity(c.owners)
            for owner in 0..<c.owners {
                layers[owner].setRows([rows[owner]])
                if fresh { layers[owner].segmentDispatchCache.clear() }
                let output = layers[owner].updateAndAttend(
                    queries: inputs.queries[step][owner] + carry,
                    keys: inputs.keys[step][owner], values: inputs.values[step][owner],
                    scale: 0.0625, sinks: nil)
                outputs.append(output)
                carry = output[0, 0, 0, 0] * inputs.zero
            }
            return outputs
        }
        for step in 0..<c.warmup {
            eval(construct(step))
            StreamOrDevice.default.stream.synchronize()
        }
        for layer in layers { layer.segmentDispatchCache.statistics = .init() }
        var timings: [SegmentMetadataArm.Step] = []
        var outputHashes: [[String]] = []
        var geometryEvents: [SegmentMetadataArm.GeometryEvent] = []
        var lastGeometry: [SegmentMetadataArm.Geometry?] = Array(repeating: nil, count: c.owners)
        for step in c.warmup..<(c.warmup + c.steps) {
            let start = DispatchTime.now().uptimeNanoseconds
            let outputs = construct(step)
            let constructed = DispatchTime.now().uptimeNanoseconds
            eval(outputs)
            StreamOrDevice.default.stream.synchronize()
            let fenced = DispatchTime.now().uptimeNanoseconds
            timings.append(.init(offsetAfter: rows[0].absoluteOffset,
                hostConstructionMs: Double(constructed - start) / 1e6,
                fencedEvaluationMs: Double(fenced - constructed) / 1e6,
                wholeStepMs: Double(fenced - start) / 1e6))
            // Diagnostic host readback follows the timed normal eval/fence.
            outputHashes.append(outputs.map(segmentMetadataDigest))
            for owner in 0..<c.owners {
                guard let plan = layers[owner].segmentDispatchCache.prepared?.plan else {
                    throw SegmentMetadataError.unexpectedGeometry
                }
                let geometry = SegmentMetadataArm.Geometry(
                    partitionTokens: plan.partitionTokens, maxPartitions: plan.maxPartitions,
                    buckets: plan.buckets.map { .init(segmentIDs: $0.segmentIDs,
                        bindingClass: $0.bindingClass, workCount: $0.workCount) })
                if geometry != lastGeometry[owner] {
                    geometryEvents.append(.init(owner: owner, offsetAfter: rows[owner].absoluteOffset,
                                                geometry: geometry))
                    lastGeometry[owner] = geometry
                }
            }
        }
        var keyHashes: [String] = [], valueHashes: [String] = []
        for row in rows {
            let snapshot = row.snapshot()
            guard snapshot.offset == c.initialOffset + c.steps,
                  snapshot.keys.shape == [1, 2, snapshot.offset, 256],
                  snapshot.values.shape == snapshot.keys.shape,
                  snapshot.keys.dtype == .bfloat16, snapshot.values.dtype == .bfloat16 else {
                throw SegmentMetadataError.unexpectedGeometry
            }
            eval(snapshot.keys, snapshot.values)
            StreamOrDevice.default.stream.synchronize()
            keyHashes.append(segmentMetadataDigest(snapshot.keys))
            valueHashes.append(segmentMetadataDigest(snapshot.values))
        }
        return SegmentMetadataArm(
            repetition: repetition, mode: fresh ? "fresh-each-step" : "cached",
            residentOwners: rows.count,
            segments: group.segments.keys.sorted().map { id in
                let segment = group.segments[id]!
                return .init(id: id, pageStart: segment.pages.lowerBound,
                    pageEnd: segment.pages.upperBound, valueOffset: segment.valueOffset,
                    logicalBytes: segment.byteCount, allocatedBytes: segment.allocatedBytes)
            }, geometryEvents: geometryEvents, steps: timings,
            counts: layers.enumerated().map { owner, layer in
                let stats = layer.segmentDispatchCache.statistics!
                return .init(owner: owner, hits: stats.hits, rebuilds: stats.rebuilds,
                    bypasses: stats.bypasses, keyNanoseconds: stats.keyNanoseconds,
                    preparationNanoseconds: stats.preparationNanoseconds)
            }, outputSHA256: outputHashes, fullHistoryKeySHA256: keyHashes,
            fullHistoryValueSHA256: valueHashes)
    }

    static func segmentMetadataDigest(_ array: MLXArray) -> String {
        SHA256.hash(data: array.asData(access: .copy).data)
            .map { String(format: "%02x", $0) }.joined()
    }
}
