import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

/// Isolated control with no model/quantization/sampler: segmented readback around
/// the2401-token boundary where a loaded Qwen trace returned zeroed history.
final class CBv2PagedSegmentReadbackBoundaryTests: XCTestCase {
    func testReadbackRetainsEveryOldTokenAcrossSegmentBoundaries() throws {
        let kinds = (0..<2).map { CBv2LayerKind(attention: .full, headDim: 256,
            kvHeads: 2, queryHeads: 24, modelLayerIndex: $0) }
        func backend() throws -> PagedKVBackend {
            try PagedKVBackend(layerKinds: kinds, config: .init(
                capacityBytes: 512 << 20, maxPrefillChunk: 2048,
                nominalMaxSequenceLength: 2600, segmentSizeBytes: 64 << 10,
                layerDTypes: [.bfloat16, .bfloat16]))
        }
        let backends = try [backend(), backend()]
        var rows: [[[CBv2SequenceKV?]]] = []
        for backend in backends {
            rows.append(try (0..<2).map { _ in
                try backend.makeSequenceState(layerKinds: kinds, promptLength: 2398, maxLength: 2600)
            })
        }
        defer {
            for i in backends.indices { rows[i].forEach { backends[i].release($0) } }
        }
        func coded(_ range: Range<Int>, tag: Int) -> MLXArray {
            let data = (0..<2).flatMap { head in
                range.flatMap { position in
                    (0..<256).map { column in
                        Float(1 + (position * 7 + column * 3 + head * 11 + tag * 19) % 251)
                    }
                }
            }
            return MLXArray(data, [2, range.count, 256]).asType(.bfloat16)
        }
        func write(_ range: Range<Int>) throws {
            for backend in backends.indices {
                for row in 0..<2 {
                    for layer in kinds.indices {
                        let kv = try XCTUnwrap(rows[backend][row][layer] as? PagedSequenceKV)
                        let k = coded(range, tag: layer * 2 + row)
                        kv.write(keys: k, values: k + 1)
                    }
                }
            }
            eval(backends.flatMap { backend in
                backend.pool.groupKeys.map { backend.pool.group($0).writeFence }
            })
        }
        try write(0..<2398)
        for end in 2399...2405 {
            try write((end - 1)..<end)
            for repeatIndex in 0..<4 {
                for backend in backends.indices {
                    for row in 0..<2 {
                        for layer in kinds.indices {
                            let kv = try XCTUnwrap(rows[backend][row][layer])
                            let view = try XCTUnwrap(kv.snapshot())
                            let k = coded(0..<end, tag: layer * 2 + row).expandedDimensions(axis: 0)
                            let expected = [k, k + 1], actual = [view.keys, view.values]
                            for component in 0..<2 {
                                if actual[component].asData(access: .copy).data != expected[component].asData(access: .copy).data {
                                    let fresh = try XCTUnwrap(kv.snapshot())
                                    let retry = component == 0 ? fresh.keys : fresh.values
                                    let retryExact = retry.asData(access: .copy).data == expected[component].asData(access: .copy).data
                                    print("[paged-readback-boundary] end=\(end) pass=\(repeatIndex) backend=\(backend) row=\(row) layer=\(layer) component=\(component) retry_exact=\(retryExact)")
                                    XCTFail("Native segmented readback changed committed values")
                                    return
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
