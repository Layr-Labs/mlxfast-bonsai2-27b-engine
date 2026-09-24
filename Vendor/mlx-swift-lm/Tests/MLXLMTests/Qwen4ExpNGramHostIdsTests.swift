// Qwen4ExpNGramHostIdsTests.swift
//
// The host-side n-gram row ids are bit-for-bit the device's. This is the
// exactness claim behind moving the hash off the graph: a differing id would
// gather a different row and change the model's output silently.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpNGramHostIdsTests: XCTestCase {

    private func embedding() throws -> Qwen4ExpNGramEmbedding {
        let config = try Qwen4ExpFixture.configuration()
        return Qwen4ExpNGramEmbedding(config, embedDimensions: 8, pleLayerIndex: 0)
    }

    /// Histories with EOS inside the window (segment resets), at the edges,
    /// and ids large enough that the wrapping multiply goes negative.
    private func histories(eos: Int) -> [[Int64]] {
        [
            [3, 9, 14, 2, 21, 6],
            [Int64(eos), 9, 14, Int64(eos), 21, 6],
            [3, Int64(eos), Int64(eos), 2, 21, Int64(eos)],
            [248_319, 248_000, 200_000, 7, 248_319, 1],
            [0, 0, 0, 0, 0, 0],
        ]
    }

    func testHostRowIdsMatchTheDeviceHash() throws {
        let embedding = try embedding()
        let eos = embedding.eosTokenId
        for history in histories(eos: eos) {
            for newCount in [1, 3, history.count] {
                let context = Array(history.prefix(history.count - newCount))
                let ids = Array(history.suffix(newCount))
                let device = embedding.rowIds(
                    MLXArray(ids.map { Int32($0) }).reshaped([1, newCount]),
                    previousContext: MLXArray(context.map { Int32($0) }).reshaped([
                        1, context.count,
                    ]))
                eval(device)
                let expected = device.asType(.int64).asArray(Int64.self).map(Int.init)
                let host = embedding.hostRowIds(history: [history], newCount: newCount)
                XCTAssertEqual(host, expected, "history \(history) newCount \(newCount)")
            }
        }
    }

    func testHostRowIdsBatchIsRowMajor() throws {
        let embedding = try embedding()
        let rows = histories(eos: embedding.eosTokenId)
        let batched = embedding.hostRowIds(history: rows, newCount: 2)
        var separate: [Int] = []
        for row in rows { separate += embedding.hostRowIds(history: [row], newCount: 2) }
        XCTAssertEqual(batched, separate)
        XCTAssertEqual(batched.count, rows.count * 2 * embedding.ngramHeads)
    }

    /// End to end: the embedding through a host-id source equals the same
    /// embedding through a device-id source, on the same ids and context.
    func testEmbeddingIsIdenticalOnBothPaths() throws {
        let config = try Qwen4ExpFixture.configuration()
        let host = Qwen4ExpNGramEmbedding(config, embedDimensions: 8, pleLayerIndex: 0)
        host.install(rowSource: DeterministicNGramRowSource(rowDimensions: 2))
        let device = Qwen4ExpNGramEmbedding(config, embedDimensions: 8, pleLayerIndex: 0)
        device.install(rowSource: DeviceOnlyNGramRowSource(rowDimensions: 2))
        let ids = MLXArray([Int32(14), 2, 21]).reshaped([1, 3])
        let context = MLXArray([Int32(3), Int32(config.eosTokenId)]).reshaped([1, 2])
        let a = host(ids, previousContext: context)
        let b = device(ids, previousContext: context)
        eval(a, b)
        XCTAssertEqual(a.shape, b.shape)
        XCTAssertTrue(allClose(a, b, rtol: 0, atol: 0).item(Bool.self))
    }
}
