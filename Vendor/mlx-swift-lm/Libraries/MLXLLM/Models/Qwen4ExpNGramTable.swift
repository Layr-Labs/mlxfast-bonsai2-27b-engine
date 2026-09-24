//
//  Qwen4ExpNGramTable.swift
//  mlx-swift-lm
//
//  The disk-resident n-gram row source of Qwen 3.8 Flash-Next.
//
//  ORIGIN. Moved here from
//  `Layr-Labs/mlxfast-qwen38-125b-a6b-engine-dev@b0b8d28:Sources/MLXFastModel/Qwen4ExpNGramTable.swift`.
//  The file carries no dependency on that track: the geometry comes from
//  `Qwen4ExpTextConfiguration` through the model's PLE layers, the refusals
//  use this file's own error, and the safetensors header reader below is
//  private to this file.
//
//  ONE SEAM, MANY SOURCES. `Qwen4ExpNGram.swift` owns the
//  `Qwen4ExpNGramRowSource` protocol. This file holds ONE conformer, the
//  disk reader. A later conformer -- a computed function instead of a table,
//  for example -- is added beside it and reached through
//  `Qwen4ExpNGramRowSourceLoader`, so a runner never names a conformer.
//
//  SHAPE (mtplx parity, ruled 2026-09-05). The reader is the Swift form of
//  the mtplx sidecar gather (`mtplx/models/qwen4_exp.py` `_SidecarGather`,
//  `mtplx/ple_row_gather.py`). Five behaviours, in the order a run meets
//  them:
//
//  1. THE TABLE IS NEVER MATERIALIZED. Every shard file stays on disk,
//     memory mapped and advised `MADV_RANDOM`. The operating system page
//     cache is the large cache; this file adds only a small hot-row cache
//     above it.
//  2. HOT-ROW LRU. A bounded cache of RAW row bytes, keyed by global row id.
//     The default budget is one gibibyte. Row POPULARITY is Zipf (common
//     n-grams recur every step) even though row PLACEMENT is hash-uniform,
//     so a small resident hot set serves most decode gathers with no page
//     touch at all -- and keeps serving them when the operating system
//     reclaims the file's page cache under memory pressure.
//  3. DECODE ONLY. The LRU serves a gather of at most
//     `hotPathMaximumRows` (4,096) DISTINCT rows. A larger gather is a
//     prefill gather -- millions of ids -- where per-row cache bookkeeping
//     costs more than the reads it saves. Those go straight off the maps: a
//     residency probe (`mincore(2)` over a fixed sample of the gather's
//     rows) decides cold from warm; cold takes a pooled `pread(2)` warm pass
//     over the rows the gather needs and then reads them, warm reads them
//     from the maps directly. A prefill gather never enters or evicts the
//     LRU.
//  4. PREWARM AT LOAD. As much of the table as free memory allows is
//     pre-read once, at construction, in HOTNESS order when the model
//     directory carries `ngram-hotness.npy` beside the shards and in file
//     order otherwise. A pre-read is an optimisation, so it never raises:
//     a missing, unreadable or wrong-shaped hotness file is ignored.
//  5. EXACTNESS. Every path returns the same bytes. See the invariant below.
//
//  CONTRACT (runner-neutral; ngram-cache-design.md section 2).
//
//  * Row identity. A row is one integer, the global row id. The table holds
//    `shard_count * rows_per_shard` rows. Global row id `g` selects shard
//    `g / rows_per_shard` and row `g % rows_per_shard` inside it. Row ids are
//    a property of the checkpoint, not of a run.
//  * Ceiling semantics. `MLXFAST_NGRAM_CACHE_LIMIT` is a byte count and it
//    bounds the HOT-ROW LRU only -- it is this reader's spelling of mtplx's
//    `MTPLX_NGRAM_HOT_MB`, in bytes instead of mebibytes, and there is no
//    second knob for the same quantity. The default is one gibibyte, the
//    same budget mtplx defaults to. A flag value wins over the environment
//    value. Zero turns the cache off, and every gather then reads off the
//    maps. A ceiling below the cost of one row behaves as zero. The resolved
//    ceiling is never raised silently.
//  * Prewarm semantics. `MLXFAST_NGRAM_PREWARM` is `auto` (the default),
//    `off`, or `on`, mirroring mtplx's `MTPLX_NGRAM_PREWARM`. `auto` reads
//    what free memory allows at load, less a fixed margin. `on` reads the
//    whole table, but only on a machine with at least
//    `Qwen4ExpNGramPrewarm.fullResidencyFloorBytes` of physical memory: a
//    128 GB machine kernel-panicked twice under full n-gram residency in
//    August 2026, so below the floor `on` falls back to the `auto` budget.
//    `off` reads nothing. The mode changes WHICH PAGES ARE CACHED and
//    nothing else.
//  * Eviction. Least recently used, per row. A row a decode step reads moves
//    to the front. When the cache is full the row at the back is dropped and
//    its space is reused.
//  * EXACTNESS INVARIANT. Neither the cache, the ceiling, the eviction
//    order, the gather size, the residency probe nor the prewarm may change
//    a value the model computes. `copyRow(_:weights:scales:biases:)` is the
//    ONE place a row is produced; the LRU stores exactly its output and the
//    map path calls it directly, so an LRU-path row, a map-path row and a
//    direct read of the file are the same bytes and the dequantization that
//    follows is the same arithmetic on the same input. A source must not
//    store rows in a lower precision than the forward pass uses, must not
//    approximate or fall back to a nearby row on a miss, and must not change
//    the gather order. `Qwen4ExpNGramTableTests` pins this.
//

import Darwin
import Dispatch
import Foundation
import MLX

// MARK: - Refusals

/// Refusals of the disk-resident n-gram row source.
///
/// Every case names the file, the shard, or the path that caused it, because
/// the alternative -- a silent fallback -- allocates 29.8 GiB and fails far
/// from the cause.
public enum Qwen4ExpNGramTableError: Error, CustomStringConvertible, Equatable {
    case refused(String)

    public var description: String {
        switch self {
        case .refused(let detail): return detail
        }
    }
}

// MARK: - Geometry

/// Geometry of the sharded n-gram table of a Qwen 3.8 Flash-Next checkpoint.
///
/// Every value here is READ OFF the checkpoint, never guessed: the shard count
/// and the row count come from the model configuration, and the row width and
/// the quantization come from the shard tensors themselves.
public struct Qwen4ExpNGramTableLayout: Equatable, Sendable {
    public let shardCount: Int
    public let rowsPerShard: Int
    public let rowDimensions: Int
    public let bits: Int
    public let groupSize: Int

    public init(
        shardCount: Int, rowsPerShard: Int, rowDimensions: Int, bits: Int = 4, groupSize: Int = 32
    ) {
        self.shardCount = shardCount
        self.rowsPerShard = rowsPerShard
        self.rowDimensions = rowDimensions
        self.bits = bits
        self.groupSize = groupSize
    }

    /// Packed weight words per row. Eight 4-bit values share one 32-bit word.
    public var weightWordsPerRow: Int { rowDimensions * bits / 32 }
    public var weightBytesPerRow: Int { weightWordsPerRow * 4 }
    /// One scale and one bias per quantization group.
    public var groupsPerRow: Int { rowDimensions / groupSize }
    /// Scales and biases are bfloat16.
    public var groupBytesPerRow: Int { groupsPerRow * 2 }
    /// Bytes one row costs in the cache: weights, scales and biases together.
    public var bytesPerRow: Int { weightBytesPerRow + 2 * groupBytesPerRow }
    public var rowCount: Int { shardCount * rowsPerShard }
}

/// Hot-row cache ceiling, in bytes.
///
/// Resolution order: the explicit flag value, then the environment variable,
/// then the default of one gibibyte. A value of zero turns the cache off; the
/// table then reads every row off the maps. This is the whole budget knob --
/// mtplx's `MTPLX_NGRAM_HOT_MB` written in bytes -- and there is deliberately
/// no second spelling of it. The MODEL OUTPUT IS THE SAME at every ceiling --
/// see `Qwen4ExpNGramTable`.
public enum Qwen4ExpNGramCacheLimit {
    public static let environmentName = "MLXFAST_NGRAM_CACHE_LIMIT"
    public static let defaultBytes = 1 << 30

    public static func parse(_ raw: String, optionLabel: String = environmentName) throws -> Int {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return defaultBytes }
        guard let value = Int(trimmed), value >= 0 else {
            throw Qwen4ExpNGramTableError.refused(
                "\(optionLabel) must be a byte count of zero or more")
        }
        return value
    }

    public static func resolve(
        flagValue: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Int {
        if let flagValue {
            return try parse(flagValue, optionLabel: "--ngram-cache-limit")
        }
        return try parse(environment[environmentName] ?? "")
    }
}

// MARK: - Prewarm

/// What the load-time pre-read should do.
///
/// The Swift spelling of mtplx's `MTPLX_NGRAM_PREWARM`. The numeric budget
/// spelling mtplx also accepts is deliberately not carried: the byte budget
/// this reader takes from the environment is the cache ceiling, and one
/// quantity with two knobs is how the two disagree.
public enum Qwen4ExpNGramPrewarmMode: String, Sendable, CaseIterable {
    case auto
    case off
    case on
}

/// One byte range of one mapped shard file.
struct Qwen4ExpNGramByteRun: Equatable {
    let fileIndex: Int
    let offset: Int
    let length: Int
}

/// The pre-read decision, with every input that made it.
public struct Qwen4ExpNGramPrewarmPlan: Equatable, Sendable {
    public let mode: Qwen4ExpNGramPrewarmMode
    public let tableBytes: Int
    public let freeBytes: Int
    public let physicalBytes: Int
    public let marginBytes: Int
    public let budgetBytes: Int
    /// How the budget was reached, for the load log.
    public let note: String
}

/// What the pre-read actually did.
public struct Qwen4ExpNGramPrewarmReceipt: Sendable {
    public let plan: Qwen4ExpNGramPrewarmPlan
    /// `"hotness"`, `"prefix"` or `"none"`.
    public let order: String
    public let rowsTaken: Int
    public let runCount: Int
    public let bytesRead: Int
    public let seconds: Double
    public let skippedReason: String?
    /// The hottest rows the budget covered, in the hotness file's own order.
    /// Bounded so a full-table prewarm does not carry millions of ids in a
    /// receipt; it exists so a run can PROVE which order it warmed.
    public let hotRows: [Int]

    public static let hotRowsReceiptLimit = 1024
}

/// Budget arithmetic and knob parsing for the load-time pre-read.
///
/// Pure functions, separated from the table so the policy can be tested
/// without a file, a map or a machine of a given size.
public enum Qwen4ExpNGramPrewarm {
    public static let environmentName = "MLXFAST_NGRAM_PREWARM"

    /// Headroom an `auto` budget leaves untouched. The pre-read competes with
    /// the KV cache and the operating system for the same unified pool, and
    /// being wrong here is a swap storm rather than a slow first token, so the
    /// margin is a documented constant rather than a fraction. Six gibibytes
    /// is the mtplx value.
    public static let marginBytes = 6 * (1 << 30)

    /// Physical memory below which `on` never arms full residency. A 128 GB
    /// machine wired ~99 GB under full n-gram residency and kernel-panicked
    /// twice (2026-08-26), so the floor is a hard refusal, not a warning.
    public static let fullResidencyFloorBytes = 160 * (1 << 30)

    public static func parse(_ raw: String) throws -> Qwen4ExpNGramPrewarmMode {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !trimmed.isEmpty else { return .auto }
        guard let mode = Qwen4ExpNGramPrewarmMode(rawValue: trimmed) else {
            throw Qwen4ExpNGramTableError.refused(
                """
                \(environmentName) must be one of \
                \(Qwen4ExpNGramPrewarmMode.allCases.map(\.rawValue).joined(separator: ", ")), \
                got "\(raw)"
                """)
        }
        return mode
    }

    public static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Qwen4ExpNGramPrewarmMode {
        try parse(environment[environmentName] ?? "")
    }

    /// Bytes of the table to pre-read.
    public static func plan(
        mode: Qwen4ExpNGramPrewarmMode,
        tableBytes: Int,
        freeBytes: Int,
        physicalBytes: Int,
        marginBytes: Int = Qwen4ExpNGramPrewarm.marginBytes
    ) -> Qwen4ExpNGramPrewarmPlan {
        let table = max(0, tableBytes)
        func made(_ budget: Int, _ note: String) -> Qwen4ExpNGramPrewarmPlan {
            Qwen4ExpNGramPrewarmPlan(
                mode: mode, tableBytes: table, freeBytes: freeBytes,
                physicalBytes: physicalBytes, marginBytes: marginBytes,
                budgetBytes: max(0, min(table, budget)), note: note)
        }
        let headroom = freeBytes - marginBytes
        switch mode {
        case .off:
            return made(0, "off")
        case .on where physicalBytes >= fullResidencyFloorBytes:
            return made(table, "on(full residency)")
        case .on:
            return made(
                headroom,
                "on(capped: \(physicalBytes >> 30) GiB physical is below the "
                    + "\(fullResidencyFloorBytes >> 30) GiB full-residency floor)")
        case .auto:
            return made(headroom, "auto(free-margin)")
        }
    }

    /// Reclaimable memory right now, and how it was measured.
    ///
    /// Free plus inactive plus purgeable, the same definition mtplx reads out
    /// of `vm_stat`, taken here straight from the kernel. Zero on failure,
    /// which yields a zero `auto` budget -- the safe direction.
    public static func freeMemoryBytes() -> (bytes: Int, source: String) {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let status = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(mach_host_self(), host_flavor_t(HOST_VM_INFO64), rebound, &count)
            }
        }
        guard status == KERN_SUCCESS else {
            return (0, "unavailable: host_statistics64 returned \(status)")
        }
        let page = Int(sysconf(_SC_PAGESIZE))
        let pages = Int(stats.free_count) + Int(stats.inactive_count) + Int(stats.purgeable_count)
        return (pages * page, "host_statistics64(free+inactive+purgeable)")
    }

    /// Coalesce byte ranges into sorted, page-aligned runs, per file.
    static func pageRuns(_ ranges: [Qwen4ExpNGramByteRun], page: Int) -> [Qwen4ExpNGramByteRun] {
        guard page > 0, !ranges.isEmpty else { return [] }
        var aligned: [Qwen4ExpNGramByteRun] = []
        aligned.reserveCapacity(ranges.count)
        for run in ranges {
            let start: Int = (run.offset / page) * page
            let last: Int = run.offset + run.length + page - 1
            let end: Int = (last / page) * page
            aligned.append(
                Qwen4ExpNGramByteRun(
                    fileIndex: run.fileIndex, offset: start, length: end - start))
        }
        aligned.sort {
            $0.fileIndex == $1.fileIndex ? $0.offset < $1.offset : $0.fileIndex < $1.fileIndex
        }

        var runs: [Qwen4ExpNGramByteRun] = []
        var current = aligned[0]
        for run in aligned.dropFirst() {
            if run.fileIndex == current.fileIndex, run.offset <= current.offset + current.length {
                let end = max(current.offset + current.length, run.offset + run.length)
                current = Qwen4ExpNGramByteRun(
                    fileIndex: current.fileIndex, offset: current.offset,
                    length: end - current.offset)
            } else {
                runs.append(current)
                current = run
            }
        }
        runs.append(current)
        return runs
    }

    /// Page-aligned read runs for the hottest `rows` that fit `budgetBytes`.
    ///
    /// Returns the runs and how many rows of `rows` they cover.
    ///
    /// At a given budget every order warms the SAME number of pages -- a row
    /// is ~100 bytes and the rows are hash-scattered, so a warmed row costs a
    /// whole page whatever the order. What the hotness order changes is WHICH
    /// pages: the ones the model will gather, instead of whichever ones sit at
    /// the front of the file.
    ///
    /// The row count that fills the budget is SEARCHED, not estimated:
    /// coalescing makes bytes-per-row wildly non-linear (adjacent hot rows
    /// share a page, scattered ones do not), and mtplx measured a single
    /// estimate leaving 97% of the budget unspent.
    static func planHotRuns(
        rows: [Int],
        budgetBytes: Int,
        page: Int,
        rangesForRow: (Int) -> [Qwen4ExpNGramByteRun]
    ) -> (runs: [Qwen4ExpNGramByteRun], rowsTaken: Int) {
        guard !rows.isEmpty, budgetBytes > 0, page > 0 else { return ([], 0) }

        func cost(_ take: Int) -> (runs: [Qwen4ExpNGramByteRun], bytes: Int) {
            var ranges: [Qwen4ExpNGramByteRun] = []
            ranges.reserveCapacity(take * 3)
            for row in rows.prefix(take) { ranges.append(contentsOf: rangesForRow(row)) }
            let runs = pageRuns(ranges, page: page)
            return (runs, runs.reduce(0) { $0 + $1.length })
        }

        // One row must fit, or there is nothing to plan.
        var measured = cost(1)
        if measured.bytes > budgetBytes { return ([], 0) }
        var low = 1
        var best = (runs: measured.runs, rowsTaken: 1)
        var high: Int? = nil
        var take = 2
        while take <= rows.count {
            measured = cost(take)
            if measured.bytes > budgetBytes {
                high = take
                break
            }
            low = take
            best = (measured.runs, take)
            take *= 2
        }
        guard var upper = high else { return best }
        while upper - low > 1 {
            let middle = (low + upper) / 2
            measured = cost(middle)
            if measured.bytes > budgetBytes {
                upper = middle
            } else {
                low = middle
                best = (measured.runs, middle)
            }
        }
        return best
    }
}

// MARK: - The hotness file

/// The optional `ngram-hotness.npy` beside the shard files: the table's row
/// ids in descending gather frequency, most-gathered first, as written by
/// mtplx's `tests/ngram_row_hotness.py`.
///
/// The parser is minimal on purpose -- one-dimensional, little-endian,
/// C-order `int64` and nothing else -- because anything it cannot read is
/// ignored rather than repaired. A hotness file is an optimisation: it must
/// never be the reason a model fails to load.
public enum Qwen4ExpNGramHotness {
    public static let fileName = "ngram-hotness.npy"

    /// Row ids from a `.npy` file, or `nil` for anything unreadable.
    public static func load(_ url: URL?) -> [Int]? {
        guard let url, let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
            return nil
        }
        return parse(data)
    }

    /// Row ids from `.npy` BYTES, or `nil`. Never throws.
    ///
    /// One shape is read and every other is declined: version 1, 2 or 3 of the
    /// format, `<i8` (little-endian signed 64-bit), C order, one dimension.
    public static func parse(_ data: Data) -> [Int]? {
        let magic: [UInt8] = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]  // \x93NUMPY
        let start = data.startIndex
        let head = [UInt8](data.prefix(12))
        guard head.count >= 10, Array(head[0 ..< 6]) == magic else { return nil }

        let headerLength: Int
        let headerStart: Int
        switch head[6] {
        case 1:
            headerLength = Int(head[8]) | Int(head[9]) << 8
            headerStart = 10
        case 2, 3:
            guard head.count == 12 else { return nil }
            headerLength =
                Int(head[8]) | Int(head[9]) << 8 | Int(head[10]) << 16 | Int(head[11]) << 24
            headerStart = 12
        default:
            return nil
        }
        guard headerLength > 0, headerStart + headerLength <= data.count else { return nil }
        let headerBytes = data[
            (start + headerStart) ..< (start + headerStart + headerLength)]
        guard let header = String(data: headerBytes, encoding: .utf8) else { return nil }

        // The header is a Python dict literal. Only three of its fields
        // matter, and each is checked rather than assumed.
        let compact = header.replacingOccurrences(of: " ", with: "")
        guard compact.contains("'descr':'<i8'") || compact.contains("\"descr\":\"<i8\"") else {
            return nil
        }
        guard
            compact.contains("'fortran_order':False")
                || compact.contains("\"fortran_order\":false")
        else { return nil }
        guard let shapeRange = compact.range(of: "'shape':(") ?? compact.range(of: "\"shape\":(")
        else { return nil }
        guard let close = compact.range(of: ")", range: shapeRange.upperBound ..< compact.endIndex)
        else { return nil }
        let shapeText = String(compact[shapeRange.upperBound ..< close.lowerBound])
        let dimensions = shapeText.split(separator: ",").compactMap { Int($0) }
        guard dimensions.count == 1, shapeText.hasSuffix(","), let count = dimensions.first,
            count > 0
        else { return nil }

        let dataStart = headerStart + headerLength
        guard data.count - dataStart >= count * 8 else { return nil }
        var rows = [Int]()
        rows.reserveCapacity(count)
        data.withUnsafeBytes { raw in
            for index in 0 ..< count {
                let value = raw.loadUnaligned(
                    fromByteOffset: dataStart + index * 8, as: Int64.self
                ).littleEndian
                rows.append(Int(value))
            }
        }
        return rows
    }

    /// Where a table's hotness file lives, or `nil` when it is not there.
    public static func url(besideShardsIn directory: URL) -> URL? {
        let candidate = directory.appendingPathComponent(fileName)
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
    }
}

// MARK: - The table

/// Solid-state-disk resident n-gram table with a bounded hot-row cache.
///
/// The table is 29.8 GiB for the pinned checkpoint, which is more than a
/// third of the model. It stays in its safetensors files, memory mapped. A
/// forward pass asks for the rows of the current tokens -- sixteen rows per
/// token -- and the table returns them.
///
/// THE TWO GATHER PATHS. A gather of at most `hotPathMaximumRows` distinct
/// rows is a decode gather and takes the hot-row LRU. A larger gather is a
/// prefill gather and goes straight off the maps, after a `mincore(2)`
/// residency probe decides whether a pooled `pread(2)` warm pass is worth its
/// cost. A prefill gather never inserts into, reads from, or evicts the LRU.
///
/// EXACTNESS INVARIANT. Both paths produce their bytes with the same
/// `copyRow` call, and the LRU holds that output verbatim, so an LRU hit, an
/// LRU miss, a warm map read and a cold map read return the SAME bytes and the
/// dequantization that follows is the same arithmetic on the same input. The
/// ceiling, the eviction order, the gather size, the probe and the prewarm can
/// never change a value this table returns. `Qwen4ExpNGramTableTests` pins
/// this.
///
/// EVICTION. Least recently used, by row. A row that a decode step touches is
/// moved to the front. When the arena is full the row at the back is dropped
/// and its slot is reused.
public final class Qwen4ExpNGramTable: Qwen4ExpNGramHostRowSource {

    /// The most distinct rows a gather may have and still take the LRU.
    /// mtplx's `_HOT_PATH_MAX_ROWS`: a decode step gathers a few dozen rows,
    /// a prefill chunk gathers millions.
    public static let hotPathMaximumRows = 4096

    /// Rows of a gather to probe for residency. ~1.9 us a probe, so 256
    /// probes cost ~0.5 ms against the ~165 ms warm pass they decide; the two
    /// regimes this must tell apart probe at 1.00 and at 0.00, so 256 draws
    /// leave the sampling error far below the margin.
    public static let residencySampleRows = 256

    /// Take the map path without a warm pass only at essentially full
    /// residency: at 1% cold the read pays hundreds of serial faults, and the
    /// pooled warm pass is then strictly better.
    public static let residentFractionThreshold = 0.99

    /// One shard's byte offsets inside its mapped file.
    struct ShardLocation {
        let fileIndex: Int
        let weightOffset: Int
        let scalesOffset: Int
        let biasesOffset: Int
    }

    public let layout: Qwen4ExpNGramTableLayout
    public let ceilingBytes: Int

    private let files: [Data]
    private let descriptors: [Int32]
    private let shards: [ShardLocation]
    private let cache: RowCache
    /// Serializes every gather. The row cache is a plain LRU over one arena
    /// and two index tables; two forwards gathering at once — the engine
    /// loop's next step built while the previous one's host readback is
    /// still inside `rows(globalIds:)`, or a draft round beside a verify —
    /// mutate it under each other. A box ran 62 minutes of free-decode
    /// oracle windows and died in `RowCache.value(for:)` on a freed pointer
    /// shape (0x8000000000000010). A gather is host work of a few hundred
    /// rows; holding one lock for its duration costs nothing measurable and
    /// makes the table's exactness hold under concurrency, not only in a
    /// single-threaded test.
    private let gatherLock = NSLock()

    public var rowDimensions: Int { layout.rowDimensions }

    /// Rows served from the hot-row cache and read on a miss.
    public private(set) var hitCount: Int = 0
    public private(set) var missCount: Int = 0
    /// Gathers that took the LRU, and gathers that bypassed it to the maps.
    public private(set) var hotPathGathers: Int = 0
    public private(set) var bypassGathers: Int = 0
    /// Of the bypassing gathers: those the probe found resident, and those
    /// that paid a pooled warm pass first.
    public private(set) var vectorizedGathers: Int = 0
    public private(set) var preadGathers: Int = 0
    /// What the load-time pre-read decided and did. `nil` only before the
    /// initializer finishes.
    public private(set) var prewarmReceipt: Qwen4ExpNGramPrewarmReceipt?

    /// Rows the hot-row cache currently holds.
    public var cachedRowCount: Int { cache.count }
    /// Bytes the hot-row cache currently holds.
    public var cachedByteCount: Int { cache.count * layout.bytesPerRow }

    /// Counting seam for the pre-read's reads. Production leaves it `nil` and
    /// the runs go through `pread(2)`; a test installs a counter so the
    /// pre-read is observable without a 29.8 GiB table.
    public typealias PrewarmReader = (_ fileIndex: Int, _ offset: Int, _ length: Int) -> Int

    /// - Parameters:
    ///   - shardFiles: safetensors files that hold the n-gram shard tensors.
    ///     Files without one are ignored, so the whole shard directory can be
    ///     passed.
    ///   - tensorPrefix: prefix of the shard tensors, for example
    ///     `language_model.model.layers.1.ple.ple_embedding.ngram_embedding`.
    ///   - layout: the table geometry.
    ///   - ceilingBytes: hot-row cache ceiling. Zero turns the cache off.
    ///   - prewarmMode: the pre-read mode. `nil` reads
    ///     `MLXFAST_NGRAM_PREWARM`.
    ///   - hotnessURL: the optional `ngram-hotness.npy` beside the shards.
    ///   - prewarmBudgetOverride: a fixed pre-read budget in bytes, for tests
    ///     that must not depend on the machine's free memory.
    ///   - prewarmReader: counting seam; see `PrewarmReader`.
    public init(
        shardFiles: [URL],
        tensorPrefix: String,
        layout: Qwen4ExpNGramTableLayout,
        ceilingBytes: Int,
        prewarmMode: Qwen4ExpNGramPrewarmMode? = nil,
        hotnessURL: URL? = nil,
        prewarmBudgetOverride: Int? = nil,
        prewarmReader: PrewarmReader? = nil
    ) throws {
        self.layout = layout
        self.ceilingBytes = ceilingBytes

        var mapped: [Data] = []
        var fds: [Int32] = []
        var located: [Int: ShardLocation] = [:]

        for url in shardFiles {
            let header = try SafetensorsHeader.read(url)
            var used = false
            for shard in 0 ..< layout.shardCount {
                let base = "\(tensorPrefix).shard_\(shard)"
                guard let weight = header.tensors["\(base).weight"],
                    let scales = header.tensors["\(base).scales"],
                    let biases = header.tensors["\(base).biases"]
                else { continue }

                try Qwen4ExpNGramTable.validate(
                    shard: shard, weight: weight, scales: scales, biases: biases, layout: layout,
                    file: url)

                if !used {
                    // Mapped, not read: the point of this class is that the
                    // table never enters resident memory as a whole. The map is
                    // advised MADV_RANDOM so a per-token row fault does not drag
                    // in a readahead cluster -- see `mapShardFile(at:)`. The
                    // descriptor is kept: the prefill warm pass and the
                    // load-time pre-read both read through pread(2).
                    let opened = try Qwen4ExpNGramTable.mapShardFile(at: url)
                    mapped.append(opened.map)
                    fds.append(opened.descriptor)
                    used = true
                }
                let dataBase = header.dataBaseOffset
                located[shard] = ShardLocation(
                    fileIndex: mapped.count - 1,
                    weightOffset: dataBase + weight.dataStart,
                    scalesOffset: dataBase + scales.dataStart,
                    biasesOffset: dataBase + biases.dataStart
                )
            }
        }

        guard located.count == layout.shardCount else {
            for fd in fds { close(fd) }
            let missing = (0 ..< layout.shardCount).filter { located[$0] == nil }
            throw Qwen4ExpNGramTableError.refused(
                """
                Qwen4ExpNGramTable: \(missing.count) of \(layout.shardCount) n-gram shards \
                were not found under prefix "\(tensorPrefix)". First missing: \
                shard_\(missing.first ?? -1).
                """)
        }

        self.files = mapped
        self.descriptors = fds
        self.shards = (0 ..< layout.shardCount).map { located[$0]! }
        self.cache = RowCache(
            bytesPerRow: layout.bytesPerRow, ceilingBytes: ceilingBytes)

        // Last, on purpose: the budget is measured against free memory at this
        // instant, which is only meaningful once the maps exist, and the
        // hotness plan needs the row geometry above.
        let resolvedMode = try prewarmMode ?? Qwen4ExpNGramPrewarm.resolve()
        self.prewarmReceipt = prewarm(
            mode: resolvedMode,
            hotnessURL: hotnessURL,
            budgetOverride: prewarmBudgetOverride,
            reader: prewarmReader)
    }

    deinit {
        for fd in descriptors { close(fd) }
    }

    /// Memory-map one shard file with kernel readahead suppressed.
    ///
    /// A decode token faults in sixteen small, scattered rows through the mmap
    /// (`copyRow`). Under the kernel's default readahead every random
    /// first-touch fault pulls a whole page cluster the row read never uses --
    /// hundreds of disk page-ins a token on the decode critical path, so the
    /// GPU idles waiting on I/O. `MADV_RANDOM` tells the kernel not to read
    /// ahead, so each fault brings in only the page the row needs.
    ///
    /// This is an access-pattern hint only. It changes which pages the kernel
    /// prefetches, never a byte a row read copies. The mapping stays cached
    /// (no `F_NOCACHE`, no `MADV_DONTNEED`), so the page cache and the RowCache
    /// still serve re-touched rows. The map is created here, so the advice is
    /// certain to land on the real region; the `.custom` deallocator munmaps it
    /// when the table is released.
    ///
    /// The descriptor is returned rather than closed: the prefill warm pass
    /// and the load-time pre-read fault pages through `pread(2)`, which
    /// releases the page cache work to the kernel instead of paying serial
    /// mapping faults on the generation thread.
    private static func mapShardFile(at url: URL) throws -> (map: Data, descriptor: Int32) {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw Qwen4ExpNGramTableError.refused(
                "Qwen4ExpNGramTable: cannot open shard file \(url.lastPathComponent)")
        }
        // Suppress readahead on the descriptor too; MADV_RANDOM on the mapping
        // below is the load-bearing knob.
        _ = Darwin.fcntl(fd, F_RDAHEAD, 0)

        var status = stat()
        guard fstat(fd, &status) == 0, status.st_size > 0 else {
            close(fd)
            throw Qwen4ExpNGramTableError.refused(
                "Qwen4ExpNGramTable: cannot size shard file \(url.lastPathComponent)")
        }
        let length = Int(status.st_size)

        let mapping = mmap(nil, length, PROT_READ, MAP_PRIVATE | MAP_FILE, fd, 0)
        guard let base = mapping, base != MAP_FAILED else {
            close(fd)
            throw Qwen4ExpNGramTableError.refused(
                "Qwen4ExpNGramTable: cannot map shard file \(url.lastPathComponent)")
        }
        _ = madvise(base, length, MADV_RANDOM)

        let map = Data(
            bytesNoCopy: base, count: length,
            deallocator: .custom { pointer, byteCount in munmap(pointer, byteCount) })
        return (map, fd)
    }

    private static func validate(
        shard: Int,
        weight: SafetensorInfo,
        scales: SafetensorInfo,
        biases: SafetensorInfo,
        layout: Qwen4ExpNGramTableLayout,
        file: URL
    ) throws {
        func check(_ condition: Bool, _ message: String) throws {
            guard condition else {
                throw Qwen4ExpNGramTableError.refused(
                    "Qwen4ExpNGramTable: \(file.lastPathComponent) shard_\(shard): \(message)")
            }
        }
        try check(
            weight.shape == [layout.rowsPerShard, layout.weightWordsPerRow],
            "weight shape \(weight.shape) is not [\(layout.rowsPerShard), \(layout.weightWordsPerRow)]"
        )
        try check(weight.dtype == "U32", "weight dtype \(weight.dtype) is not U32")
        for (name, info) in [("scales", scales), ("biases", biases)] {
            try check(
                info.shape == [layout.rowsPerShard, layout.groupsPerRow],
                "\(name) shape \(info.shape) is not [\(layout.rowsPerShard), \(layout.groupsPerRow)]"
            )
            try check(info.dtype == "BF16", "\(name) dtype \(info.dtype) is not BF16")
        }
    }

    // MARK: Row gathering

    public func rows(globalIds: MLXArray) -> MLXArray {
        rows(
            globalIds: globalIds.asType(.int32).asArray(Int32.self).map(Int.init),
            shape: globalIds.shape)
    }

    /// The host-id form (`Qwen4ExpNGramHostRowSource`): no device readback.
    public func rows(globalIds ids: [Int], shape: [Int]) -> MLXArray {
        let count = ids.count

        let weightBytes = layout.weightBytesPerRow
        let groupBytes = layout.groupBytesPerRow
        var weights = Data(count: count * weightBytes)
        var scales = Data(count: count * groupBytes)
        var biases = Data(count: count * groupBytes)

        weights.withUnsafeMutableBytes { weightOut in
            scales.withUnsafeMutableBytes { scaleOut in
                biases.withUnsafeMutableBytes { biasOut in
                    gather(
                        ids: ids, weights: weightOut.baseAddress!, scales: scaleOut.baseAddress!,
                        biases: biasOut.baseAddress!)
                }
            }
        }

        let packed = MLXArray(weights, [count, layout.weightWordsPerRow], dtype: .uint32)
        let scaleArray = MLXArray(scales, [count, layout.groupsPerRow], dtype: .bfloat16)
        let biasArray = MLXArray(biases, [count, layout.groupsPerRow], dtype: .bfloat16)

        let dequantizedRows = dequantized(
            packed,
            scales: scaleArray,
            biases: biasArray,
            groupSize: layout.groupSize,
            bits: layout.bits
        )
        return dequantizedRows.reshaped(shape + [layout.rowDimensions])
    }

    /// THE gather: choose the path, then write the rows of `ids` into three
    /// destination buffers, one row of each per id, in the caller's order.
    ///
    /// This carries the whole decode/prefill split, and it is deliberately
    /// free of MLX: `rows(globalIds:)` is a thin wrapper that turns the ids
    /// into integers and the filled buffers into arrays, so the path choice
    /// can be exercised -- and its exactness pinned -- without the runtime.
    func gather(
        ids: [Int],
        weights: UnsafeMutableRawPointer,
        scales: UnsafeMutableRawPointer,
        biases: UnsafeMutableRawPointer
    ) {
        gatherLock.lock()
        defer { gatherLock.unlock() }
        // Distinct rows decide the path, exactly as mtplx's np.unique does.
        var seen = Set<Int>()
        var unique: [Int] = []
        unique.reserveCapacity(min(ids.count, Qwen4ExpNGramTable.hotPathMaximumRows + 1))
        for id in ids where seen.insert(id).inserted { unique.append(id) }

        let takesHotPath =
            !unique.isEmpty && unique.count <= Qwen4ExpNGramTable.hotPathMaximumRows
            && cache.slotCapacity > 0
        if takesHotPath {
            hotPathGathers += 1
        } else {
            bypassGathers += 1
            prepareMapPath(unique)
        }

        let weightBytes = layout.weightBytesPerRow
        let groupBytes = layout.groupBytesPerRow
        for (position, id) in ids.enumerated() {
            let weightSlot = weights + position * weightBytes
            let scaleSlot = scales + position * groupBytes
            let biasSlot = biases + position * groupBytes
            if takesHotPath {
                // The cache holds exactly what `copyRow` writes, so splitting
                // it back out here cannot change a byte.
                readRow(id).withUnsafeBytes { source in
                    let base = source.baseAddress!
                    memcpy(weightSlot, base, weightBytes)
                    memcpy(scaleSlot, base + weightBytes, groupBytes)
                    memcpy(biasSlot, base + weightBytes + groupBytes, groupBytes)
                }
            } else {
                copyRow(id, weights: weightSlot, scales: scaleSlot, biases: biasSlot)
            }
        }
    }

    /// The gather's raw bytes, one `[weights, scales, biases]` row per id.
    ///
    /// The same call `rows(globalIds:)` makes, stopping one step short of MLX.
    /// It exists so the exactness invariant can be checked on the bytes the
    /// two paths actually produce.
    func gatherRawRows(_ ids: [Int]) -> [[UInt8]] {
        let weightBytes = layout.weightBytesPerRow
        let groupBytes = layout.groupBytesPerRow
        var weights = [UInt8](repeating: 0, count: ids.count * weightBytes)
        var scales = [UInt8](repeating: 0, count: ids.count * groupBytes)
        var biases = [UInt8](repeating: 0, count: ids.count * groupBytes)
        weights.withUnsafeMutableBytes { weightOut in
            scales.withUnsafeMutableBytes { scaleOut in
                biases.withUnsafeMutableBytes { biasOut in
                    gather(
                        ids: ids, weights: weightOut.baseAddress!, scales: scaleOut.baseAddress!,
                        biases: biasOut.baseAddress!)
                }
            }
        }
        return (0 ..< ids.count).map { position in
            var row = [UInt8]()
            row.reserveCapacity(layout.bytesPerRow)
            row.append(
                contentsOf: weights[(position * weightBytes) ..< ((position + 1) * weightBytes)])
            row.append(
                contentsOf: scales[(position * groupBytes) ..< ((position + 1) * groupBytes)])
            row.append(
                contentsOf: biases[(position * groupBytes) ..< ((position + 1) * groupBytes)])
            return row
        }
    }

    /// The raw checkpoint bytes of one row: weights, then scales, then biases.
    ///
    /// The hot-row path. `Qwen4ExpNGramTableTests` reads rows through this.
    func readRow(_ globalId: Int) -> [UInt8] {
        if let cached = cache.value(for: globalId) {
            hitCount += 1
            return cached
        }
        missCount += 1
        let row = readRowFromFile(globalId)
        cache.insert(row, for: globalId)
        return row
    }

    func readRowFromFile(_ globalId: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: layout.bytesPerRow)
        let weightBytes = layout.weightBytesPerRow
        let groupBytes = layout.groupBytesPerRow
        out.withUnsafeMutableBytes { destination in
            let base = destination.baseAddress!
            copyRow(
                globalId, weights: base, scales: base + weightBytes,
                biases: base + weightBytes + groupBytes)
        }
        return out
    }

    /// THE one place a row's bytes are produced.
    ///
    /// Both gather paths call this, so "the LRU returns what the map returns"
    /// is true by construction rather than by comparison.
    private func copyRow(
        _ globalId: Int,
        weights: UnsafeMutableRawPointer,
        scales: UnsafeMutableRawPointer,
        biases: UnsafeMutableRawPointer
    ) {
        precondition(
            globalId >= 0 && globalId < layout.rowCount,
            "Qwen4ExpNGramTable: row \(globalId) is outside the table")
        let shard = shards[globalId / layout.rowsPerShard]
        let row = globalId % layout.rowsPerShard
        let file = files[shard.fileIndex]
        let weightBytes = layout.weightBytesPerRow
        let groupBytes = layout.groupBytesPerRow

        file.withUnsafeBytes { source in
            let base = source.baseAddress!
            memcpy(weights, base + shard.weightOffset + row * weightBytes, weightBytes)
            memcpy(scales, base + shard.scalesOffset + row * groupBytes, groupBytes)
            memcpy(biases, base + shard.biasesOffset + row * groupBytes, groupBytes)
        }
    }

    /// The three byte ranges one row occupies in its mapped file.
    func byteRanges(forRow globalId: Int) -> [Qwen4ExpNGramByteRun] {
        guard globalId >= 0, globalId < layout.rowCount else { return [] }
        let shard = shards[globalId / layout.rowsPerShard]
        let row = globalId % layout.rowsPerShard
        let weightBytes = layout.weightBytesPerRow
        let groupBytes = layout.groupBytesPerRow
        return [
            Qwen4ExpNGramByteRun(
                fileIndex: shard.fileIndex, offset: shard.weightOffset + row * weightBytes,
                length: weightBytes),
            Qwen4ExpNGramByteRun(
                fileIndex: shard.fileIndex, offset: shard.scalesOffset + row * groupBytes,
                length: groupBytes),
            Qwen4ExpNGramByteRun(
                fileIndex: shard.fileIndex, offset: shard.biasesOffset + row * groupBytes,
                length: groupBytes),
        ]
    }

    // MARK: The map path

    /// Decide, for a bypassing (prefill-sized) gather, whether to pay a warm
    /// pass before reading the rows off the maps.
    ///
    /// The probe costs ~0.5 ms; the warm pass it decides costs ~165 ms per
    /// 32,768 rows, and a demand-faulted map read on a cold table is an order
    /// of magnitude slower than a pooled `pread(2)` of the same rows. So the
    /// question is answered by measurement (`mincore(2)`), never by guessing:
    /// an unavailable answer means "cold", because taking the warm pass is
    /// never wrong, only sometimes wasteful.
    private func prepareMapPath(_ unique: [Int]) {
        guard !unique.isEmpty else { return }
        let fraction = residentFraction(unique)
        if let fraction, fraction >= Qwen4ExpNGramTable.residentFractionThreshold {
            vectorizedGathers += 1
            return
        }
        preadGathers += 1
        warmRows(unique)
    }

    /// The share of the sampled rows' pages already in core, or `nil` when
    /// `mincore(2)` cannot answer.
    ///
    /// The sample is a fixed stride through the rows rather than a random
    /// draw: the answer must not move run to run for the same gather, or an
    /// A/B could not attribute a delta to the code instead of to the sampler.
    func residentFraction(_ rows: [Int], sample: Int = Qwen4ExpNGramTable.residencySampleRows)
        -> Double?
    {
        guard !rows.isEmpty, sample > 0 else { return nil }
        let page = Int(sysconf(_SC_PAGESIZE))
        guard page > 0 else { return nil }
        let step = max(1, rows.count / sample)

        var pages = 0
        var resident = 0
        var vector = [Int8](repeating: 0, count: 64)
        var failed = false

        for index in stride(from: 0, to: rows.count, by: step) {
            for range in byteRanges(forRow: rows[index]) {
                let start = (range.offset / page) * page
                let end = ((range.offset + range.length + page - 1) / page) * page
                let span = end - start
                let pageCount = span / page
                if pageCount > vector.count { vector = [Int8](repeating: 0, count: pageCount) }
                let file = files[range.fileIndex]
                let answered = file.withUnsafeBytes { source -> Bool in
                    guard let base = source.baseAddress else { return false }
                    let address = UnsafeMutableRawPointer(mutating: base) + start
                    return vector.withUnsafeMutableBufferPointer { buffer in
                        mincore(address, span, buffer.baseAddress) == 0
                    }
                }
                guard answered else {
                    failed = true
                    break
                }
                pages += pageCount
                for slot in 0 ..< pageCount where vector[slot] & 0x1 == 1 { resident += 1 }
            }
            if failed { break }
        }
        guard !failed, pages > 0 else { return nil }
        return Double(resident) / Double(pages)
    }

    /// Fault the rows a gather needs into the page cache with pooled
    /// `pread(2)`.
    ///
    /// A cold mapping fault is serial and blocks the calling thread; `pread`
    /// does not, so the warm pass is worth a pool. The reads are discarded:
    /// the point is the page, not the value, and the gather that follows
    /// still copies its bytes out of the map.
    private func warmRows(_ rows: [Int]) {
        guard !rows.isEmpty, !descriptors.isEmpty else { return }
        let ranges = rows.flatMap { byteRanges(forRow: $0) }
        guard !ranges.isEmpty else { return }
        let workers = min(16, max(1, ProcessInfo.processInfo.activeProcessorCount))
        let chunk = (ranges.count + workers - 1) / workers
        let descriptors = self.descriptors
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let start = worker * chunk
            let end = min(ranges.count, start + chunk)
            guard start < end else { return }
            var scratch = [UInt8](repeating: 0, count: 4096)
            for index in start ..< end {
                let range = ranges[index]
                if range.length > scratch.count {
                    scratch = [UInt8](repeating: 0, count: range.length)
                }
                scratch.withUnsafeMutableBytes { buffer in
                    _ = pread(
                        descriptors[range.fileIndex], buffer.baseAddress, range.length,
                        off_t(range.offset))
                }
            }
        }
    }

    // MARK: The load-time pre-read

    /// Pre-read as much of the table as the budget allows.
    ///
    /// Never raises: the pre-read is an optimisation, and it must not be the
    /// reason a model fails to load. The order is the hotness file when the
    /// model directory carries one, and the file prefix otherwise -- at the
    /// same budget both warm the same number of pages, so the only question
    /// is which ones.
    private func prewarm(
        mode: Qwen4ExpNGramPrewarmMode,
        hotnessURL: URL?,
        budgetOverride: Int?,
        reader: PrewarmReader?
    ) -> Qwen4ExpNGramPrewarmReceipt {
        let tableBytes = files.reduce(0) { $0 + $1.count }
        let free = Qwen4ExpNGramPrewarm.freeMemoryBytes()
        var plan = Qwen4ExpNGramPrewarm.plan(
            mode: mode,
            tableBytes: tableBytes,
            freeBytes: free.bytes,
            physicalBytes: Int(ProcessInfo.processInfo.physicalMemory))
        if let budgetOverride, mode != .off {
            plan = Qwen4ExpNGramPrewarmPlan(
                mode: plan.mode, tableBytes: plan.tableBytes, freeBytes: plan.freeBytes,
                physicalBytes: plan.physicalBytes, marginBytes: plan.marginBytes,
                budgetBytes: max(0, min(tableBytes, budgetOverride)), note: "override")
        }

        func receipt(
            order: String, rowsTaken: Int, runCount: Int, bytesRead: Int, seconds: Double,
            skipped: String?, hotRows: [Int]
        ) -> Qwen4ExpNGramPrewarmReceipt {
            Qwen4ExpNGramPrewarmReceipt(
                plan: plan, order: order, rowsTaken: rowsTaken, runCount: runCount,
                bytesRead: bytesRead, seconds: seconds, skippedReason: skipped, hotRows: hotRows)
        }

        guard plan.budgetBytes > 0 else {
            return receipt(
                order: "none", rowsTaken: 0, runCount: 0, bytesRead: 0, seconds: 0,
                skipped: mode == .off ? "disabled" : "no_headroom", hotRows: [])
        }

        let page = Int(sysconf(_SC_PAGESIZE))
        let started = Date()
        // A budget that covers the table has nothing to prioritise, and a
        // sequential read is faster than the same pages fetched at random, so
        // full coverage always takes the prefix path.
        let hotness = plan.budgetBytes >= tableBytes ? nil : Qwen4ExpNGramHotness.load(hotnessURL)

        if let hotness, !hotness.isEmpty {
            let planned = Qwen4ExpNGramPrewarm.planHotRuns(
                rows: hotness, budgetBytes: plan.budgetBytes, page: page,
                rangesForRow: { [weak self] row in self?.byteRanges(forRow: row) ?? [] })
            if !planned.runs.isEmpty {
                let bytes = readPrewarmRuns(planned.runs, reader: reader)
                let taken = Array(
                    hotness.prefix(
                        min(planned.rowsTaken, Qwen4ExpNGramPrewarmReceipt.hotRowsReceiptLimit)))
                return receipt(
                    order: "hotness", rowsTaken: planned.rowsTaken, runCount: planned.runs.count,
                    bytesRead: bytes, seconds: -started.timeIntervalSinceNow, skipped: nil,
                    hotRows: taken)
            }
        }

        var runs: [Qwen4ExpNGramByteRun] = []
        var remaining = plan.budgetBytes
        for (index, file) in files.enumerated() where remaining > 0 {
            let length = min(remaining, file.count)
            runs.append(Qwen4ExpNGramByteRun(fileIndex: index, offset: 0, length: length))
            remaining -= length
        }
        let bytes = readPrewarmRuns(runs, reader: reader)
        return receipt(
            order: "prefix", rowsTaken: 0, runCount: runs.count, bytesRead: bytes,
            seconds: -started.timeIntervalSinceNow, skipped: nil, hotRows: [])
    }

    /// Read the planned runs into the page cache; return the bytes read.
    private func readPrewarmRuns(_ runs: [Qwen4ExpNGramByteRun], reader: PrewarmReader?) -> Int {
        let block = 8 * 1024 * 1024
        var total = 0
        var scratch = [UInt8](repeating: 0, count: block)
        for run in runs {
            if let reader {
                total += reader(run.fileIndex, run.offset, run.length)
                continue
            }
            var read = 0
            while read < run.length {
                let want = min(block, run.length - read)
                let got = scratch.withUnsafeMutableBytes { buffer in
                    pread(
                        descriptors[run.fileIndex], buffer.baseAddress, want,
                        off_t(run.offset + read))
                }
                if got <= 0 { break }
                read += got
            }
            total += read
        }
        return total
    }

    // MARK: Least-recently-used hot-row cache

    /// A fixed arena of row slots with a least-recently-used order.
    ///
    /// The arena is sized once from the ceiling, so there is no growth path and
    /// no allocation on the hot path. A ceiling below one row's cost turns the
    /// cache off. Only decode-sized gathers reach it; see
    /// `Qwen4ExpNGramTable.hotPathMaximumRows`.
    final class RowCache {
        private let bytesPerRow: Int
        private let capacity: Int
        private var arena: [UInt8]
        private var slotOfRow: [Int: Int] = [:]
        private var rowOfSlot: [Int]
        private var previous: [Int]
        private var next: [Int]
        private var head = -1
        private var tail = -1
        private var used = 0

        init(bytesPerRow: Int, ceilingBytes: Int) {
            self.bytesPerRow = bytesPerRow
            self.capacity = bytesPerRow > 0 ? ceilingBytes / bytesPerRow : 0
            self.arena = [UInt8](repeating: 0, count: capacity * bytesPerRow)
            self.rowOfSlot = [Int](repeating: -1, count: capacity)
            self.previous = [Int](repeating: -1, count: capacity)
            self.next = [Int](repeating: -1, count: capacity)
            self.slotOfRow.reserveCapacity(capacity)
        }

        var slotCapacity: Int { capacity }
        /// Rows held right now. Never above `slotCapacity`, which is what
        /// keeps the cache under its byte ceiling.
        var count: Int { slotOfRow.count }

        func value(for row: Int) -> [UInt8]? {
            guard let slot = slotOfRow[row] else { return nil }
            moveToFront(slot)
            let start = slot * bytesPerRow
            return Array(arena[start ..< (start + bytesPerRow)])
        }

        func insert(_ bytes: [UInt8], for row: Int) {
            guard capacity > 0 else { return }
            let slot: Int
            if used < capacity {
                slot = used
                used += 1
            } else {
                slot = tail
                slotOfRow.removeValue(forKey: rowOfSlot[slot])
                detach(slot)
            }
            rowOfSlot[slot] = row
            slotOfRow[row] = slot
            let start = slot * bytesPerRow
            arena.replaceSubrange(start ..< (start + bytesPerRow), with: bytes)
            pushFront(slot)
        }

        private func moveToFront(_ slot: Int) {
            guard head != slot else { return }
            detach(slot)
            pushFront(slot)
        }

        private func detach(_ slot: Int) {
            let p = previous[slot]
            let n = next[slot]
            if p != -1 { next[p] = n }
            if n != -1 { previous[n] = p }
            if head == slot { head = n }
            if tail == slot { tail = p }
            previous[slot] = -1
            next[slot] = -1
        }

        private func pushFront(_ slot: Int) {
            previous[slot] = -1
            next[slot] = head
            if head != -1 { previous[head] = slot }
            head = slot
            if tail == -1 { tail = slot }
        }
    }
}

// MARK: - Building the table from a checkpoint

extension Qwen4ExpNGramTable {

    /// Tensor-name prefixes to try for the n-gram shards of one PLE layer.
    ///
    /// The published checkpoint nests the tower under `language_model.`; a
    /// checkpoint that has already been through `sanitize(weights:)` does not.
    /// Both spellings are accepted so the table works either way.
    static func candidatePrefixes(pleLayerIndex: Int) -> [String] {
        let tail = "model.layers.\(pleLayerIndex).ple.ple_embedding.ngram_embedding"
        return ["language_model.\(tail)", tail]
    }

    /// The safetensors files of one shard directory, in name order.
    static func shardFiles(in directory: URL) throws -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory)
        else {
            throw Qwen4ExpNGramTableError.refused(
                """
                Qwen4ExpNGramTable: \(directory.path) does not exist. This source \
                takes the DIRECTORY of n-gram shard files that the offline \
                transform writes.
                """)
        }
        guard isDirectory.boolValue else {
            throw Qwen4ExpNGramTableError.refused(
                """
                Qwen4ExpNGramTable: \(directory.path) is a file. This source takes \
                the DIRECTORY of n-gram shard files that the offline transform \
                writes, not one file inside it.
                """)
        }
        let entries = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        let files = entries.filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw Qwen4ExpNGramTableError.refused(
                "Qwen4ExpNGramTable: \(directory.path) holds no .safetensors file")
        }
        return files
    }

    /// Build the table for a model's one PLE layer from a shard directory.
    ///
    /// The geometry comes from the model, so nothing here is guessed: the
    /// shard count, the rows per shard and the row width are the values
    /// `Qwen4ExpTextConfiguration` gave the PLE layer. The optional
    /// `ngram-hotness.npy` beside the shards orders the load-time pre-read.
    ///
    /// - Parameters:
    ///   - directory: the shard directory the offline transform writes. Files
    ///     that hold no n-gram shard are skipped.
    ///   - model: the loaded target.
    ///   - ceilingBytes: hot-row cache ceiling. Defaults to the resolved flag
    ///     or environment value, then to one gibibyte.
    public static func make(
        shardDirectory directory: URL,
        for model: Qwen4ExpModel,
        ceilingBytes: Int? = nil
    ) throws -> Qwen4ExpNGramTable {
        let embeddings = model.pleEmbeddings
        guard let embedding = embeddings.first else {
            throw Qwen4ExpNGramTableError.refused(
                "Qwen4ExpNGramTable: this model declares no PLE layer, so it has no n-gram table")
        }
        guard embeddings.count == 1 else {
            // The pinned checkpoint has exactly one PLE layer. More than one
            // would need one table each, which is a change, not a detail.
            throw Qwen4ExpNGramTableError.refused(
                """
                Qwen4ExpNGramTable: this model declares \(embeddings.count) PLE layers. \
                The builder serves one; extend it deliberately rather than sharing a table.
                """)
        }

        let layout = Qwen4ExpNGramTableLayout(
            shardCount: embedding.shardCount,
            rowsPerShard: embedding.rowsPerShard,
            rowDimensions: embedding.rowDimensions
        )
        let ceiling = try ceilingBytes ?? Qwen4ExpNGramCacheLimit.resolve()
        let files = try shardFiles(in: directory)
        let hotness = Qwen4ExpNGramHotness.url(besideShardsIn: directory)
        let layerIndex = model.model.pleLayerIndices[0]

        var lastError: Error?
        for prefix in candidatePrefixes(pleLayerIndex: layerIndex) {
            do {
                return try Qwen4ExpNGramTable(
                    shardFiles: files,
                    tensorPrefix: prefix,
                    layout: layout,
                    ceilingBytes: ceiling,
                    hotnessURL: hotness
                )
            } catch {
                lastError = error
            }
        }
        throw lastError
            ?? Qwen4ExpNGramTableError.refused("Qwen4ExpNGramTable: no n-gram shards were found")
    }
}

// MARK: - The one construction entry point

/// Builds a `Qwen4ExpNGramRowSource` from a path resource.
///
/// A runner names THIS type and the protocol, never a conformer. The disk
/// reader is one conformer today. A later conformer -- a computed function,
/// for example -- is chosen here, so no runner changes.
///
/// RESOURCE SHAPE. One shape only: the DIRECTORY of n-gram shard files that
/// the offline transform writes. A path to a single file is refused by name.
public enum Qwen4ExpNGramRowSourceLoader {

    public static func rowSource(
        at path: URL,
        for model: Qwen4ExpModel,
        ceilingBytes: Int? = nil
    ) throws -> any Qwen4ExpNGramRowSource {
        try Qwen4ExpNGramTable.make(
            shardDirectory: path, for: model, ceilingBytes: ceilingBytes)
    }
}

// MARK: - Minimal safetensors header reader

/// One tensor's placement inside a safetensors file.
///
/// Private to this file on purpose: the table needs offsets, not arrays, and
/// the shared loader reads arrays. Keeping this here is what lets the fork
/// hold the reader without a dependency on the track.
struct SafetensorInfo: Equatable {
    let dtype: String
    let shape: [Int]
    let dataStart: Int
    let dataEnd: Int
}

struct SafetensorsHeader: Equatable {
    let headerLength: Int
    let tensors: [String: SafetensorInfo]

    /// First byte of the tensor data block: the 8-byte length word plus the
    /// header itself.
    var dataBaseOffset: Int { headerLength + 8 }

    /// Matches the safetensors reference implementation's header limit.
    static let maximumHeaderByteCount = 100_000_000

    static func read(_ path: URL) throws -> SafetensorsHeader {
        func refuse(_ message: String) -> Qwen4ExpNGramTableError {
            .refused("Qwen4ExpNGramTable: \(message): \(path.path)")
        }

        let handle = try FileHandle(forReadingFrom: path)
        defer { try? handle.close() }

        var status = stat()
        guard fstat(handle.fileDescriptor, &status) == 0 else {
            throw refuse("cannot size safetensors file")
        }
        let fileByteCount = Int(status.st_size)
        guard fileByteCount >= 8 else { throw refuse("safetensors file is too small") }

        let prefix = handle.readData(ofLength: 8)
        guard prefix.count == 8 else { throw refuse("safetensors file is too small") }
        let rawHeaderLength = prefix.withUnsafeBytes { raw -> UInt64 in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard rawHeaderLength > 0 else { throw refuse("safetensors header is empty") }
        guard rawHeaderLength <= UInt64(maximumHeaderByteCount),
            let headerLength = Int(exactly: rawHeaderLength)
        else {
            throw refuse("safetensors header exceeds \(maximumHeaderByteCount) bytes")
        }
        guard headerLength <= fileByteCount - 8 else {
            throw refuse("truncated safetensors header")
        }

        let headerData = handle.readData(ofLength: headerLength)
        guard headerData.count == headerLength else { throw refuse("truncated safetensors header") }

        guard let object = try? JSONSerialization.jsonObject(with: headerData),
            let dictionary = object as? [String: Any]
        else {
            throw refuse("safetensors header is not a JSON object")
        }

        let dataByteCount = fileByteCount - 8 - headerLength
        var tensors: [String: SafetensorInfo] = [:]
        for (name, value) in dictionary where name != "__metadata__" {
            guard let tensor = value as? [String: Any] else { continue }
            guard let dtype = tensor["dtype"] as? String,
                let shape = (tensor["shape"] as? [NSNumber])?.map({ $0.intValue }),
                let offsets = (tensor["data_offsets"] as? [NSNumber])?.map({ $0.intValue }),
                offsets.count == 2
            else {
                throw refuse("invalid tensor header for \(name)")
            }
            guard offsets[0] >= 0, offsets[1] >= offsets[0], offsets[1] <= dataByteCount else {
                throw refuse("invalid data_offsets for \(name)")
            }
            tensors[name] = SafetensorInfo(
                dtype: dtype, shape: shape, dataStart: offsets[0], dataEnd: offsets[1])
        }

        return SafetensorsHeader(headerLength: headerLength, tensors: tensors)
    }
}
