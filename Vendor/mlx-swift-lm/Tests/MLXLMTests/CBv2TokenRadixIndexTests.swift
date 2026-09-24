import XCTest
@testable import MLXLMCommon

final class CBv2TokenRadixIndexTests: XCTestCase {
    func testSplitsDoNotInventCheckpointAndRemovalCompactsBranches() {
        var index = CBv2TokenRadixIndex<Int>()
        index.insert(tokens: [1, 2, 3, 4][...], value: 10)
        index.insert(tokens: [1, 2, 5, 6][...], value: 20)
        XCTAssertTrue(index.matches(tokens: [1, 2][...]).isEmpty)
        XCTAssertTrue(index.matches(tokens: [1, 2, 3][...]).isEmpty)
        XCTAssertEqual(index.matches(tokens: [1, 2, 3, 4, 9][...]).first?.values, [10])
        index.insert(tokens: [1, 2][...], value: 30)
        XCTAssertEqual(index.matches(tokens: [1, 2, 3, 4][...]).map(\.position), [2, 4])
        index.remove(tokens: [1, 2, 3, 4][...], value: 10)
        index.remove(tokens: [1, 2][...], value: 30)
        XCTAssertEqual(index.matches(tokens: [1, 2, 5, 6][...]).first?.values, [20])
        index.remove(tokens: [1, 2, 5, 6][...], value: 20)
        XCTAssertTrue(index.isEmpty)
    }

    func testRandomizedOperationsMatchExactTokenOracle() {
        var seed: UInt64 = 0x7261646978
        func random(_ upper: Int) -> Int {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1
            return Int(seed >> 32) % upper
        }
        var index = CBv2TokenRadixIndex<Int>()
        var oracle: [[Int]: Set<Int>] = [:]
        for iteration in 0 ..< 5_000 {
            let key = (0 ..< 1 + random(12)).map { _ in random(4) }
            let value = random(16)
            if random(3) > 0 {
                index.insert(tokens: key[...], value: value)
                oracle[key, default: []].insert(value)
            } else if let existing = oracle.keys.sorted(by: { $0.lexicographicallyPrecedes($1) }).first,
                let removal = oracle[existing]?.first
            {
                index.remove(tokens: existing[...], value: removal)
                oracle[existing]?.remove(removal)
                if oracle[existing]?.isEmpty == true { oracle.removeValue(forKey: existing) }
            }
            let query = key + [random(4), random(4)]
            let actual = Dictionary(uniqueKeysWithValues: index.matches(tokens: query[...]))
            var expected: [Int: Set<Int>] = [:]
            for (prefix, values) in oracle where query.starts(with: prefix) {
                expected[prefix.count] = values
            }
            XCTAssertEqual(actual, expected, "operation \(iteration)")
        }
    }

    func testRecurrentGeometryRefusesPackedRaggedAndMissingChunks() {
        var cold = CBv2RecurrentCheckpointGeometry()
        XCTAssertTrue(cold.record(range: 0 ..< 4, cap: 4, promptLength: 10, packed: false))
        XCTAssertTrue(cold.record(range: 4 ..< 8, cap: 4, promptLength: 10, packed: false))
        XCTAssertFalse(cold.record(range: 8 ..< 10, cap: 4, promptLength: 10, packed: false))
        XCTAssertFalse(cold.isArmed)
        var packed = CBv2RecurrentCheckpointGeometry()
        XCTAssertFalse(packed.record(range: 0 ..< 4, cap: 4, promptLength: 12, packed: true))
        var skipped = CBv2RecurrentCheckpointGeometry()
        XCTAssertFalse(skipped.record(range: 4 ..< 8, cap: 4, promptLength: 12, packed: false))
        var warm = CBv2RecurrentCheckpointGeometry(position: 8, chunkSize: 4)
        XCTAssertTrue(warm.record(range: 8 ..< 12, cap: 4, promptLength: 16, packed: false))
        XCTAssertFalse(warm.record(range: 12 ..< 14, cap: 2, promptLength: 16, packed: false))
    }
}
