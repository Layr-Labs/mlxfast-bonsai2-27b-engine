import MLX
import Testing
@_spi(Diagnostics) @testable import MLXLMCommon

@Suite("CBv2 segmented metadata profiler", .serialized)
struct CBv2SegmentMetadataProfilerTests {
    @Test func refusesUnboundedInputsBeforeAllocation() {
        for c in [
            PagedDecodeProfiler.SegmentMetadataConfiguration(owners: 11),
            .init(initialOffset: 8193), .init(initialOffset: 1), .init(warmup: 0),
            .init(steps: 129), .init(repetitions: 4)
        ] {
            #expect(throws: PagedDecodeProfiler.SegmentMetadataError.self) {
                try PagedDecodeProfiler.measureSegmentMetadata(c)
            }
        }
    }

    @Test(arguments: [31, 255])
    func sameBinaryArmsMatchEveryOutputAndHistoryAcrossBoundary(offset: Int) throws {
        let report = try PagedDecodeProfiler.measureSegmentMetadata(.init(
            owners: 2, initialOffset: offset, warmup: 1, steps: 6, repetitions: 1))
        #expect(report.allOutputAndFullHistoryDigestsEqual)
        #expect(report.dtype == "bfloat16" && report.queryHeads == 16 && report.headDim == 256)
        let cached = try #require(report.arms.first { $0.mode == "cached" })
        let fresh = try #require(report.arms.first { $0.mode == "fresh-each-step" })
        #expect(cached.outputSHA256 == fresh.outputSHA256)
        #expect(cached.outputSHA256.count == 6 && cached.outputSHA256.allSatisfy { $0.count == 2 })
        #expect(cached.fullHistoryKeySHA256 == fresh.fullHistoryKeySHA256)
        #expect(cached.fullHistoryValueSHA256 == fresh.fullHistoryValueSHA256)
        #expect(cached.fullHistoryKeySHA256.count == 2)
        #expect(cached.steps.map(\.offsetAfter) == Array((offset + 1)...(offset + 6)))
        #expect(cached.counts.allSatisfy { $0.hits == 5 && $0.rebuilds == 1 && $0.bypasses == 0 })
        #expect(fresh.counts.allSatisfy { $0.hits == 0 && $0.rebuilds == 6 && $0.bypasses == 0 })
        #expect(cached.residentOwners == 2 && !cached.segments.isEmpty)
        #expect(cached.segments.map(\.allocatedBytes).allSatisfy { $0 > 0 })
        #expect(cached.geometryEvents.allSatisfy { $0.geometry.partitionTokens == 256 })
        if offset == 255 {
            #expect(Set(cached.geometryEvents.map { $0.geometry.maxPartitions }) == [1, 2])
        }
        // No assertion compares wall-clock timings or claims a speedup.
    }
}
