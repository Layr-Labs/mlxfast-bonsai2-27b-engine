// Qwen4ExpNGramTableTests.swift
//
// The n-gram table's cache must never change a value.
//
// The whole point of the disk-resident table is that the ceiling is a MEMORY
// decision and nothing else. A run with the cache off, a run with a cache
// that evicts on almost every row, and a run that holds the whole table must
// return the same rows in the same order.
//
// These also pin the mtplx-shape reader (ruled 2026-09-05): the hot-row LRU
// serves decode-sized gathers only, a prefill-sized gather bypasses it to the
// maps, and the load-time pre-read follows the hotness file when there is one.
//
// Ported from
// `Layr-Labs/mlxfast-qwen38-125b-a6b-engine-dev@b0b8d28:Tests/MLXFastTests/Qwen4ExpNGramCacheExactnessTests.swift`.
// Byte level only, so nothing here needs the MLX runtime or a GPU: the rows
// are compared as the raw checkpoint bytes the cache actually holds, which is
// where the invariant lives, and the gather itself is exercised through
// `gatherRawRows`, the step before MLX.

import Foundation
import XCTest

@testable import MLXLLM

final class Qwen4ExpNGramTableTests: XCTestCase {

    // A small table with the checkpoint's quantization: 4-bit affine, group 32.
    private let layout = Qwen4ExpNGramTableLayout(
        shardCount: 4, rowsPerShard: 8, rowDimensions: 64)
    private let tensorPrefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"

    /// Ids chosen to cross every shard, repeat rows, and revisit an id after
    /// enough traffic to have evicted it from a small cache.
    private let requestedIds: [Int] = [
        0, 1, 8, 9, 17, 31, 0, 24, 25, 7, 16, 1, 30, 2, 8, 31, 0, 15, 23, 9,
    ]

    // MARK: Synthetic table directory

    /// Write one safetensors file holding every shard of the small table.
    ///
    /// The values are deterministic and distinct per (shard, row, byte), so a
    /// row served from the wrong place cannot pass by accident.
    private func writeShardFile(at url: URL, layout: Qwen4ExpNGramTableLayout? = nil) throws {
        let layout = layout ?? self.layout
        var tensors: [String: [String: Any]] = [:]
        var payload = Data()
        var offset = 0

        func append(_ name: String, dtype: String, shape: [Int], bytes: Data) {
            tensors[name] = [
                "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.count],
            ]
            payload.append(bytes)
            offset += bytes.count
        }

        for shard in 0 ..< layout.shardCount {
            var weight = Data(count: layout.rowsPerShard * layout.weightBytesPerRow)
            for index in 0 ..< weight.count {
                weight[index] = UInt8((shard &* 37 &+ index &* 11 &+ 3) % 251)
            }
            var scales = Data(count: layout.rowsPerShard * layout.groupBytesPerRow)
            var biases = Data(count: layout.rowsPerShard * layout.groupBytesPerRow)
            for index in 0 ..< scales.count {
                // bfloat16 bit patterns that stay small and finite.
                scales[index] = UInt8((shard &* 5 &+ index &* 3 &+ 60) % 64)
                biases[index] = UInt8((shard &* 7 &+ index &* 13 &+ 32) % 64)
            }
            let base = "\(tensorPrefix).shard_\(shard)"
            append(
                "\(base).weight", dtype: "U32",
                shape: [layout.rowsPerShard, layout.weightWordsPerRow], bytes: weight)
            append(
                "\(base).scales", dtype: "BF16",
                shape: [layout.rowsPerShard, layout.groupsPerRow], bytes: scales)
            append(
                "\(base).biases", dtype: "BF16",
                shape: [layout.rowsPerShard, layout.groupsPerRow], bytes: biases)
        }

        let header = try JSONSerialization.data(withJSONObject: tensors, options: [.sortedKeys])
        var file = Data()
        var length = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &length) { file.append(contentsOf: $0) }
        file.append(header)
        file.append(payload)
        try file.write(to: url)
    }

    private func makeTable(
        ceilingBytes: Int,
        directory: URL,
        layout: Qwen4ExpNGramTableLayout? = nil,
        prewarmMode: Qwen4ExpNGramPrewarmMode? = .off,
        hotnessURL: URL? = nil,
        prewarmBudgetOverride: Int? = nil,
        prewarmReader: Qwen4ExpNGramTable.PrewarmReader? = nil
    ) throws -> Qwen4ExpNGramTable {
        try Qwen4ExpNGramTable(
            shardFiles: try Qwen4ExpNGramTable.shardFiles(in: directory),
            tensorPrefix: tensorPrefix,
            layout: layout ?? self.layout,
            ceilingBytes: ceilingBytes,
            prewarmMode: prewarmMode,
            hotnessURL: hotnessURL,
            prewarmBudgetOverride: prewarmBudgetOverride,
            prewarmReader: prewarmReader
        )
    }

    private func makeDirectory(layout: Qwen4ExpNGramTableLayout? = nil) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4exp-ngram-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try writeShardFile(
            at: directory.appendingPathComponent("shards.safetensors"), layout: layout)
        return directory
    }

    // MARK: The wide fixture, for the decode/prefill split

    /// A fixture with more rows than the hot path serves, so a gather can
    /// cross `hotPathMaximumRows` without a real checkpoint.
    private let wideLayout = Qwen4ExpNGramTableLayout(
        shardCount: 4, rowsPerShard: 2048, rowDimensions: 64)

    /// Rows the raw file holds for one id, read straight out of the file with
    /// no table involved: the third opinion the two gather paths are compared
    /// against.
    private func directRowBytes(
        _ id: Int, layout: Qwen4ExpNGramTableLayout, directory: URL
    ) throws -> [UInt8] {
        let url = directory.appendingPathComponent("shards.safetensors")
        let header = try SafetensorsHeader.read(url)
        let file = try Data(contentsOf: url)
        let shard = id / layout.rowsPerShard
        let row = id % layout.rowsPerShard
        let base = "\(tensorPrefix).shard_\(shard)"
        let dataBase = header.dataBaseOffset

        var bytes = Data()
        let weightStart =
            dataBase + header.tensors["\(base).weight"]!.dataStart + row * layout.weightBytesPerRow
        bytes.append(file[weightStart ..< (weightStart + layout.weightBytesPerRow)])
        let scaleStart =
            dataBase + header.tensors["\(base).scales"]!.dataStart + row * layout.groupBytesPerRow
        bytes.append(file[scaleStart ..< (scaleStart + layout.groupBytesPerRow)])
        let biasStart =
            dataBase + header.tensors["\(base).biases"]!.dataStart + row * layout.groupBytesPerRow
        bytes.append(file[biasStart ..< (biasStart + layout.groupBytesPerRow)])
        return [UInt8](bytes)
    }

    /// Write a minimal version-1 `.npy` of int64 row ids, the shape
    /// `tests/ngram_row_hotness.py` writes.
    private func writeHotnessFile(_ rows: [Int], at url: URL) throws {
        var header = "{'descr': '<i8', 'fortran_order': False, 'shape': (\(rows.count),), }"
        // The header is padded with spaces to a 64-byte boundary and ends in a
        // newline, exactly as numpy writes it.
        while (10 + header.count + 1) % 64 != 0 { header += " " }
        header += "\n"

        var file = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 1, 0])
        var length = UInt16(header.count).littleEndian
        withUnsafeBytes(of: &length) { file.append(contentsOf: $0) }
        file.append(header.data(using: .ascii)!)
        for row in rows {
            var value = Int64(row).littleEndian
            withUnsafeBytes(of: &value) { file.append(contentsOf: $0) }
        }
        try file.write(to: url)
    }

    /// A counting stand-in for the pre-read's `pread(2)`, so the pre-read is
    /// observable on a fixture that fits in a test.
    private final class PrewarmCounter {
        private(set) var calls = 0
        private(set) var bytes = 0
        private(set) var runs: [(file: Int, offset: Int, length: Int)] = []

        func reader() -> Qwen4ExpNGramTable.PrewarmReader {
            { file, offset, length in
                self.calls += 1
                self.bytes += length
                self.runs.append((file, offset, length))
                return length
            }
        }
    }

    // MARK: Geometry

    func testLayoutMatchesTheCheckpointRowCost() {
        // The pinned checkpoint: 128 shards, 2,500,012 rows, 160 values, 4-bit
        // affine group 32. That is 80 weight bytes plus two 10-byte group
        // vectors, so 100 bytes a row and 29.80 GiB in total.
        let pinned = Qwen4ExpNGramTableLayout(
            shardCount: 128, rowsPerShard: 2_500_012, rowDimensions: 160)
        XCTAssertEqual(pinned.weightWordsPerRow, 20)
        XCTAssertEqual(pinned.weightBytesPerRow, 80)
        XCTAssertEqual(pinned.groupsPerRow, 5)
        XCTAssertEqual(pinned.groupBytesPerRow, 10)
        XCTAssertEqual(pinned.bytesPerRow, 100)
        XCTAssertEqual(pinned.rowCount, 320_001_536)
        XCTAssertEqual(pinned.rowCount * pinned.bytesPerRow, 32_000_153_600)
    }

    // MARK: Ceiling resolution

    func testCeilingDefaultsToOneGibibyte() throws {
        XCTAssertEqual(try Qwen4ExpNGramCacheLimit.resolve(environment: [:]), 1 << 30)
    }

    func testCeilingReadsTheEnvironmentAndTheFlag() throws {
        XCTAssertEqual(
            try Qwen4ExpNGramCacheLimit.resolve(
                environment: [Qwen4ExpNGramCacheLimit.environmentName: "65536"]),
            65536)
        XCTAssertEqual(
            try Qwen4ExpNGramCacheLimit.resolve(
                flagValue: "0",
                environment: [Qwen4ExpNGramCacheLimit.environmentName: "65536"]),
            0)
    }

    func testCeilingRefusesANegativeOrUnparseableValue() {
        XCTAssertThrowsError(try Qwen4ExpNGramCacheLimit.parse("-1"))
        XCTAssertThrowsError(try Qwen4ExpNGramCacheLimit.parse("1GiB"))
    }

    // MARK: Exactness

    func testRowBytesAreIdenticalAtEveryCeiling() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // Off, room for two rows, and room for the whole table.
        let ceilings = [0, 2 * layout.bytesPerRow, layout.rowCount * layout.bytesPerRow]
        var results: [[[UInt8]]] = []
        for ceiling in ceilings {
            let table = try makeTable(ceilingBytes: ceiling, directory: directory)
            results.append(requestedIds.map { table.readRow($0) })
        }

        for index in 1 ..< results.count {
            XCTAssertEqual(
                results[0], results[index],
                "ceiling \(ceilings[index]) returned different bytes than a cache-free read")
        }
        // The synthetic data must actually distinguish rows, or the assertion
        // above would pass on an all-zero table.
        XCTAssertEqual(Set(results[0].map { Data($0) }).count, Set(requestedIds).count)
    }

    func testTheCacheIsUsedAndStillEvicts() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let off = try makeTable(ceilingBytes: 0, directory: directory)
        for id in requestedIds { _ = off.readRow(id) }
        XCTAssertEqual(off.hitCount, 0, "a zero ceiling must never serve a cached row")
        XCTAssertEqual(off.missCount, requestedIds.count)

        let whole = try makeTable(
            ceilingBytes: layout.rowCount * layout.bytesPerRow, directory: directory)
        for id in requestedIds { _ = whole.readRow(id) }
        XCTAssertEqual(whole.missCount, Set(requestedIds).count, "each row is read once")
        XCTAssertEqual(whole.hitCount, requestedIds.count - Set(requestedIds).count)

        let tiny = try makeTable(ceilingBytes: 2 * layout.bytesPerRow, directory: directory)
        for id in requestedIds { _ = tiny.readRow(id) }
        XCTAssertGreaterThan(tiny.missCount, whole.missCount, "a tiny ceiling must evict")
        XCTAssertLessThanOrEqual(tiny.missCount, requestedIds.count)
    }

    func testRowBytesMatchTheFileDirectly() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("shards.safetensors")
        let header = try SafetensorsHeader.read(url)
        let file = try Data(contentsOf: url)
        let table = try makeTable(ceilingBytes: 1 << 20, directory: directory)

        for id in requestedIds {
            let shard = id / layout.rowsPerShard
            let row = id % layout.rowsPerShard
            let base = "\(tensorPrefix).shard_\(shard)"
            let dataBase = header.dataBaseOffset
            let weight = header.tensors["\(base).weight"]!
            let scales = header.tensors["\(base).scales"]!
            let biases = header.tensors["\(base).biases"]!

            var expected = Data()
            let weightStart = dataBase + weight.dataStart + row * layout.weightBytesPerRow
            expected.append(file[weightStart ..< (weightStart + layout.weightBytesPerRow)])
            let scaleStart = dataBase + scales.dataStart + row * layout.groupBytesPerRow
            expected.append(file[scaleStart ..< (scaleStart + layout.groupBytesPerRow)])
            let biasStart = dataBase + biases.dataStart + row * layout.groupBytesPerRow
            expected.append(file[biasStart ..< (biasStart + layout.groupBytesPerRow)])

            XCTAssertEqual(Data(table.readRow(id)), expected, "row \(id) does not match the file")
        }
    }

    // MARK: Refusals

    func testAMissingShardIsRefusedByName() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertThrowsError(
            try Qwen4ExpNGramTable(
                shardFiles: try Qwen4ExpNGramTable.shardFiles(in: directory),
                tensorPrefix: tensorPrefix,
                layout: Qwen4ExpNGramTableLayout(
                    shardCount: 8, rowsPerShard: 8, rowDimensions: 64),
                ceilingBytes: 0
            )
        ) { error in
            XCTAssertTrue(
                "\(error)".contains("n-gram shards"), "unexpected refusal: \(error)")
        }
    }

    func testASingleFileIsRefusedByName() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("shards.safetensors")
        XCTAssertThrowsError(try Qwen4ExpNGramTable.shardFiles(in: file)) { error in
            XCTAssertTrue("\(error)".contains("is a file"), "unexpected refusal: \(error)")
            XCTAssertTrue("\(error)".contains("DIRECTORY"), "unexpected refusal: \(error)")
        }
    }

    // MARK: The decode / prefill split

    func testADecodeSizedGatherTakesTheHotRowCacheAndHitsOnRepeat() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let table = try makeTable(ceilingBytes: 1 << 20, directory: directory)

        let first = table.gatherRawRows(requestedIds)
        XCTAssertEqual(table.hotPathGathers, 1, "a decode-sized gather must take the LRU")
        XCTAssertEqual(table.bypassGathers, 0)
        XCTAssertEqual(table.cachedRowCount, Set(requestedIds).count)
        XCTAssertEqual(table.missCount, Set(requestedIds).count, "each row is read once")

        let missesAfterFirst = table.missCount
        let second = table.gatherRawRows(requestedIds)
        XCTAssertEqual(table.missCount, missesAfterFirst, "a repeated gather must hit")
        XCTAssertEqual(table.hitCount, requestedIds.count * 2 - missesAfterFirst)
        XCTAssertEqual(first, second, "a hit and a miss must return the same bytes")
    }

    func testAGatherWiderThanTheHotPathBypassesTheCache() throws {
        let directory = try makeDirectory(layout: wideLayout)
        defer { try? FileManager.default.removeItem(at: directory) }
        let table = try makeTable(
            ceilingBytes: 1 << 20, directory: directory, layout: wideLayout)

        // A prefill-sized gather: more distinct rows than the hot path serves.
        let wide = Array(0 ... Qwen4ExpNGramTable.hotPathMaximumRows)
        XCTAssertGreaterThan(wide.count, Qwen4ExpNGramTable.hotPathMaximumRows)
        _ = table.gatherRawRows(wide)

        XCTAssertEqual(table.bypassGathers, 1, "a prefill-sized gather must bypass the LRU")
        XCTAssertEqual(table.hotPathGathers, 0)
        XCTAssertEqual(table.cachedRowCount, 0, "a bypassing gather must not fill the LRU")
        XCTAssertEqual(table.hitCount, 0)
        XCTAssertEqual(table.missCount, 0)
        XCTAssertEqual(
            table.vectorizedGathers + table.preadGathers, 1,
            "the residency probe must decide every bypassing gather")

        // Exactly at the threshold the gather is still a decode gather.
        let decode = Array(0 ..< Qwen4ExpNGramTable.hotPathMaximumRows)
        _ = table.gatherRawRows(decode)
        XCTAssertEqual(table.hotPathGathers, 1)
        XCTAssertEqual(table.cachedRowCount, decode.count)
    }

    func testTheHotRowCacheStaysUnderItsByteCeiling() throws {
        let directory = try makeDirectory(layout: wideLayout)
        defer { try? FileManager.default.removeItem(at: directory) }
        // Room for eight rows, and 2,000 distinct rows asked for.
        let ceiling = 8 * wideLayout.bytesPerRow
        let table = try makeTable(ceilingBytes: ceiling, directory: directory, layout: wideLayout)

        for start in stride(from: 0, to: 2000, by: 100) {
            _ = table.gatherRawRows(Array(start ..< (start + 100)))
            XCTAssertLessThanOrEqual(table.cachedByteCount, ceiling, "the ceiling must hold")
        }
        XCTAssertEqual(table.cachedRowCount, 8)
        XCTAssertGreaterThan(table.missCount, 8, "a small ceiling must evict")
    }

    func testBothGatherPathsAndADirectFileReadAgree() throws {
        let directory = try makeDirectory(layout: wideLayout)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Ids that repeat, cross every shard, and revisit an evicted row.
        let ids = [0, 2047, 2048, 4095, 0, 6144, 8191, 17, 2048, 4096, 8191, 3]
        let expected = try ids.map {
            try directRowBytes($0, layout: wideLayout, directory: directory)
        }

        let hot = try makeTable(ceilingBytes: 1 << 20, directory: directory, layout: wideLayout)
        let throughLRU = hot.gatherRawRows(ids)
        XCTAssertEqual(hot.hotPathGathers, 1)

        // Ceiling zero: the same ids, off the maps, with no cache at all.
        let cold = try makeTable(ceilingBytes: 0, directory: directory, layout: wideLayout)
        let throughMaps = cold.gatherRawRows(ids)
        XCTAssertEqual(cold.bypassGathers, 1)

        XCTAssertEqual(throughLRU, expected, "the LRU path must return the file's bytes")
        XCTAssertEqual(throughMaps, expected, "the map path must return the file's bytes")

        // And once more across the hot-path threshold, where the bypass is
        // chosen by size rather than by ceiling.
        let wide = Array(0 ... Qwen4ExpNGramTable.hotPathMaximumRows)
        let wideRows = hot.gatherRawRows(wide)
        XCTAssertEqual(hot.bypassGathers, 1)
        for position in stride(from: 0, to: wide.count, by: 512) {
            XCTAssertEqual(
                wideRows[position],
                try directRowBytes(wide[position], layout: wideLayout, directory: directory),
                "prefill row \(wide[position]) differs from a direct read")
        }
    }

    // MARK: Prewarm

    func testPrewarmFollowsTheHotnessFileOrder() throws {
        let directory = try makeDirectory(layout: wideLayout)
        defer { try? FileManager.default.removeItem(at: directory) }
        let hotnessURL = directory.appendingPathComponent(Qwen4ExpNGramHotness.fileName)
        let hottest = [7000, 12, 4100, 2049, 300, 6001, 1024, 8000]
        try writeHotnessFile(hottest, at: hotnessURL)

        // A budget under the table's size, so the order actually decides
        // which pages are read.
        let counter = PrewarmCounter()
        let table = try makeTable(
            ceilingBytes: 0, directory: directory, layout: wideLayout, prewarmMode: .auto,
            hotnessURL: hotnessURL, prewarmBudgetOverride: 120_000,
            prewarmReader: counter.reader())
        let receipt = try XCTUnwrap(table.prewarmReceipt)

        XCTAssertEqual(receipt.order, "hotness")
        XCTAssertNil(receipt.skippedReason)
        XCTAssertGreaterThanOrEqual(receipt.rowsTaken, 1)
        XCTAssertLessThanOrEqual(receipt.rowsTaken, hottest.count)
        XCTAssertEqual(
            receipt.hotRows, Array(hottest.prefix(receipt.rowsTaken)),
            "the pre-read must take the hottest rows, in the file's order")
        XCTAssertEqual(counter.calls, receipt.runCount, "every planned run must be read")
        XCTAssertEqual(counter.bytes, receipt.bytesRead)
        XCTAssertGreaterThan(receipt.bytesRead, 0)
        XCTAssertLessThanOrEqual(receipt.bytesRead, 120_000, "the budget must hold")

        // The control: a different order warms different rows.
        try writeHotnessFile(hottest.reversed(), at: hotnessURL)
        let reversedCounter = PrewarmCounter()
        let reversedTable = try makeTable(
            ceilingBytes: 0, directory: directory, layout: wideLayout, prewarmMode: .auto,
            hotnessURL: hotnessURL, prewarmBudgetOverride: 120_000,
            prewarmReader: reversedCounter.reader())
        let reversedReceipt = try XCTUnwrap(reversedTable.prewarmReceipt)
        XCTAssertEqual(reversedReceipt.order, "hotness")
        XCTAssertNotEqual(
            reversedReceipt.hotRows, receipt.hotRows,
            "the pre-read must follow the file, not the row ids' own order")
    }

    func testAMissingOrUnreadableHotnessFileIsIgnored() throws {
        let directory = try makeDirectory(layout: wideLayout)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Missing: the pre-read falls back to the file prefix and loads.
        let counter = PrewarmCounter()
        let table = try makeTable(
            ceilingBytes: 0, directory: directory, layout: wideLayout, prewarmMode: .auto,
            hotnessURL: nil, prewarmBudgetOverride: 120_000, prewarmReader: counter.reader())
        let receipt = try XCTUnwrap(table.prewarmReceipt)
        XCTAssertEqual(receipt.order, "prefix")
        XCTAssertNil(receipt.skippedReason)
        XCTAssertGreaterThan(counter.bytes, 0)

        // Unreadable: the same, and never a refusal.
        let broken = directory.appendingPathComponent(Qwen4ExpNGramHotness.fileName)
        try Data("not a numpy file".utf8).write(to: broken)
        let brokenTable = try makeTable(
            ceilingBytes: 0, directory: directory, layout: wideLayout, prewarmMode: .auto,
            hotnessURL: broken, prewarmBudgetOverride: 120_000,
            prewarmReader: PrewarmCounter().reader())
        XCTAssertEqual(try XCTUnwrap(brokenTable.prewarmReceipt).order, "prefix")
    }

    func testPrewarmOffReadsNothing() throws {
        let directory = try makeDirectory(layout: wideLayout)
        defer { try? FileManager.default.removeItem(at: directory) }
        let hotnessURL = directory.appendingPathComponent(Qwen4ExpNGramHotness.fileName)
        try writeHotnessFile([1, 2, 3], at: hotnessURL)

        let counter = PrewarmCounter()
        let table = try makeTable(
            ceilingBytes: 0, directory: directory, layout: wideLayout, prewarmMode: .off,
            hotnessURL: hotnessURL, prewarmBudgetOverride: 120_000,
            prewarmReader: counter.reader())
        let receipt = try XCTUnwrap(table.prewarmReceipt)

        XCTAssertEqual(receipt.order, "none")
        XCTAssertEqual(receipt.skippedReason, "disabled")
        XCTAssertEqual(receipt.plan.budgetBytes, 0)
        XCTAssertEqual(counter.calls, 0, "off must read nothing at all")
        XCTAssertEqual(receipt.bytesRead, 0)
    }

    func testPrewarmKnobVocabulary() throws {
        XCTAssertEqual(try Qwen4ExpNGramPrewarm.resolve(environment: [:]), .auto)
        XCTAssertEqual(
            try Qwen4ExpNGramPrewarm.resolve(environment: [
                Qwen4ExpNGramPrewarm.environmentName: "off"
            ]), .off)
        XCTAssertEqual(
            try Qwen4ExpNGramPrewarm.resolve(environment: [
                Qwen4ExpNGramPrewarm.environmentName: "ON"
            ]), .on)
        XCTAssertThrowsError(try Qwen4ExpNGramPrewarm.parse("sometimes")) { error in
            XCTAssertTrue("\(error)".contains("MLXFAST_NGRAM_PREWARM"), "unexpected refusal: \(error)")
        }
    }

    func testPrewarmBudgetPolicy() {
        let table = 32 * (1 << 30)
        let off = Qwen4ExpNGramPrewarm.plan(
            mode: .off, tableBytes: table, freeBytes: table, physicalBytes: 256 * (1 << 30))
        XCTAssertEqual(off.budgetBytes, 0)

        // auto: free memory less the margin, never more than the table.
        let auto = Qwen4ExpNGramPrewarm.plan(
            mode: .auto, tableBytes: table, freeBytes: 20 * (1 << 30),
            physicalBytes: 128 * (1 << 30))
        XCTAssertEqual(auto.budgetBytes, 20 * (1 << 30) - Qwen4ExpNGramPrewarm.marginBytes)
        let starved = Qwen4ExpNGramPrewarm.plan(
            mode: .auto, tableBytes: table, freeBytes: 1 << 30, physicalBytes: 128 * (1 << 30))
        XCTAssertEqual(starved.budgetBytes, 0, "a machine with no headroom pre-reads nothing")

        // on: full residency above the floor only. A 128 GiB machine kernel
        // panicked on it, so the floor is the whole point of this case.
        let large = Qwen4ExpNGramPrewarm.plan(
            mode: .on, tableBytes: table, freeBytes: 20 * (1 << 30),
            physicalBytes: 192 * (1 << 30))
        XCTAssertEqual(
            large.budgetBytes, table, "above the floor, on reads the table whatever is free")
        let small = Qwen4ExpNGramPrewarm.plan(
            mode: .on, tableBytes: table, freeBytes: 20 * (1 << 30),
            physicalBytes: 128 * (1 << 30))
        XCTAssertEqual(small.budgetBytes, 20 * (1 << 30) - Qwen4ExpNGramPrewarm.marginBytes)
        XCTAssertLessThan(small.budgetBytes, table, "below the floor, on must not go resident")
        XCTAssertTrue(small.note.contains("floor"), "the cap must say why: \(small.note)")
    }

    func testHotnessParserReadsOnlyWhatItPromises() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4exp-hotness-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent(Qwen4ExpNGramHotness.fileName)
        try writeHotnessFile([9, 4, 0, 1_000_000], at: url)
        XCTAssertEqual(Qwen4ExpNGramHotness.load(url), [9, 4, 0, 1_000_000])
        XCTAssertEqual(Qwen4ExpNGramHotness.url(besideShardsIn: directory), url)

        XCTAssertNil(Qwen4ExpNGramHotness.load(nil))
        XCTAssertNil(Qwen4ExpNGramHotness.parse(Data("not a numpy file at all".utf8)))
        // A float table is not a row-id table, and is ignored rather than cast.
        var float = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 1, 0])
        var header = "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }"
        while (10 + header.count + 1) % 64 != 0 { header += " " }
        header += "\n"
        var length = UInt16(header.count).littleEndian
        withUnsafeBytes(of: &length) { float.append(contentsOf: $0) }
        float.append(header.data(using: .ascii)!)
        float.append(Data(repeating: 0, count: 8))
        XCTAssertNil(Qwen4ExpNGramHotness.parse(float))
    }

    func testAnEmptyDirectoryIsRefusedByName() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4exp-ngram-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertThrowsError(try Qwen4ExpNGramTable.shardFiles(in: directory)) { error in
            XCTAssertTrue(
                "\(error)".contains("no .safetensors file"), "unexpected refusal: \(error)")
        }
    }
}
