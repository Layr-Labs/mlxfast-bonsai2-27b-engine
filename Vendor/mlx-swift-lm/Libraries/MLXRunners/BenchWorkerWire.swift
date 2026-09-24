// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Engine Protocol v1 wire types.
//
// The envelope is CLOSED: an unknown request key is a decode failure, not a
// silently ignored field. Optional response fields are OMITTED (never null)
// when unused, and the key ORDER is the schema's appended-last convention
// with `runner` last of all (contract §6.0).

import Foundation

// MARK: - Wire writer

/// Minimal JSON writer for responses.
///
/// `JSONEncoder` cannot produce this wire: it renders `Double` 0.0 as `0`
/// and 10.0 as `10`, while the pinned conformance fixture (and every other
/// producer of this protocol) writes `0.0` and `10.0`; and its key order for
/// a synthesized encoder is not the schema's. Both are load bearing — the
/// fixture is compared byte for byte on both sides — so responses are
/// serialized here and only PARSED with `JSONDecoder`.
enum WireValue {
    case int(Int)
    case double(Double)
    case bool(Bool)
    case string(String)
    case array([WireValue])
    case object([(String, WireValue)])

    var json: String {
        switch self {
        case .int(let value): return String(value)
        // `String(Double)` renders 0.0 / 10.0 / 18.5 exactly as the wire
        // spells them.
        case .double(let value): return String(value)
        case .bool(let value): return value ? "true" : "false"
        case .string(let value): return CanonicalJSON.string(value)
        case .array(let values): return "[" + values.map(\.json).joined(separator: ",") + "]"
        case .object(let fields):
            let body = fields.map { CanonicalJSON.string($0.0) + ":" + $0.1.json }
            return "{" + body.joined(separator: ",") + "}"
        }
    }
}

// MARK: - Requests

/// Benchmarker → engine.
public struct WorkerRequest: Codable, Sendable {

    public enum Kind: String, Codable, Sendable {
        case prefill
        case decodeBegin = "decode_begin"
        case decodeStep = "decode_step"
        case correctness
        case correctnessBegin = "correctness_begin"
        case correctnessStep = "correctness_step"
        case phaseDiagnostics = "phase_diagnostics"
        case freeDecodeBegin = "free_decode_begin"
        case freeDecodeRun = "free_decode_run"
        /// Trusted-build reference oracle. The v1 schema's `kind` enum omits
        /// this value at bench-protocol's current pin — a known defect being
        /// fixed on the benchd side — but the contract's §8 verb table names
        /// it, so it is served here.
        case cohortReferenceReplay = "cohort_reference_replay"
    }

    public var id: Int
    public var kind: Kind
    public var promptTokens: [Int]?
    public var token: Int?
    public var seedTokens: [Int]?
    public var steps: Int?
    public var count: Int?
    public var spec: SpecConfig?
    public var seedTokensByStream: [[Int]]?
    public var batchSize: Int?
    public var replaySeedsByStream: [[Int]]?
    public var committedByStream: [[Int]]?
    public var logitTopK: Int?
    public var relEnvelope: Double?
    public var replayWidth: String?

    public init(id: Int, kind: Kind) {
        self.id = id
        self.kind = kind
    }

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case promptTokens = "prompt_tokens"
        case token
        case seedTokens = "seed_tokens"
        case steps
        case count
        case spec
        case seedTokensByStream = "seed_tokens_by_stream"
        case batchSize = "batch_size"
        case replaySeedsByStream = "replay_seeds_by_stream"
        case committedByStream = "committed_by_stream"
        case logitTopK = "logit_top_k"
        case relEnvelope = "rel_envelope"
        case replayWidth = "replay_width"
    }
}

/// Per-module speculative configuration. Exactly one mode; the per-mode
/// block is nested under the mode key.
public struct SpecConfig: Codable, Sendable, Equatable {
    public var mode: String
    public var mtp: MtpSpec?
    /// The `dflash` module block. Each decoder owns its own block, so a
    /// DFlash depth never arrives or echoes under the `mtp` key.
    public var dflash: DFlashSpec?

    public init(mode: String, mtp: MtpSpec? = nil, dflash: DFlashSpec? = nil) {
        self.mode = mode
        self.mtp = mtp
        self.dflash = dflash
    }

    /// Draft depth for the selected mode; 0 for a non-drafting mode.
    public var depth: Int { mtp?.depth ?? dflash?.depth ?? 0 }

    /// The wire spelling of the non-drafting mode.
    public static let serialMode = "serial"

    var wire: WireValue {
        var fields: [(String, WireValue)] = [("mode", .string(mode))]
        if let mtp {
            fields.append(("mtp", .object([("depth", .int(mtp.depth))])))
        }
        if let dflash {
            fields.append(("dflash", .object([("depth", .int(dflash.depth))])))
        }
        return .object(fields)
    }
}

public struct MtpSpec: Codable, Sendable, Equatable {
    public var depth: Int
    public init(depth: Int) { self.depth = depth }
}

public struct DFlashSpec: Codable, Sendable, Equatable {
    public var depth: Int
    public init(depth: Int) { self.depth = depth }
}

public struct CorrectnessTraceLogit: Codable, Sendable, Equatable {
    public var token: Int
    public var logit: Double
    public init(token: Int, logit: Double) {
        self.token = token
        self.logit = logit
    }

    var wire: WireValue {
        .object([("token", .int(token)), ("logit", .double(logit))])
    }
}

/// The dense RAM-resident runtime reports the zero struct; the schema
/// requires every field.
public struct ExpertStreamingStats: Codable, Sendable, Equatable {
    public var expertCacheHits: Int = 0
    public var expertCacheMisses: Int = 0
    public var expertCacheEvictions: Int = 0
    public var expertBytesRead: Int = 0
    public var expertReadSeconds: Double = 0
    public var expertPeakCachedTensors: Int = 0

    public init() {}

    enum CodingKeys: String, CodingKey {
        case expertCacheHits = "expert_cache_hits"
        case expertCacheMisses = "expert_cache_misses"
        case expertCacheEvictions = "expert_cache_evictions"
        case expertBytesRead = "expert_bytes_read"
        case expertReadSeconds = "expert_read_seconds"
        case expertPeakCachedTensors = "expert_peak_cached_tensors"
    }

    var wire: WireValue {
        .object([
            ("expert_cache_hits", .int(expertCacheHits)),
            ("expert_cache_misses", .int(expertCacheMisses)),
            ("expert_cache_evictions", .int(expertCacheEvictions)),
            ("expert_bytes_read", .int(expertBytesRead)),
            ("expert_read_seconds", .double(expertReadSeconds)),
            ("expert_peak_cached_tensors", .int(expertPeakCachedTensors)),
        ])
    }
}

public struct WireHeadProvenance: Codable, Sendable, Equatable {
    public var sha256: String
    public var bytes: Int
    public var fileCount: Int

    public init(_ provenance: HeadProvenance) {
        self.sha256 = provenance.sha256
        self.bytes = provenance.bytes
        self.fileCount = provenance.fileCount
    }

    enum CodingKeys: String, CodingKey {
        case sha256
        case bytes
        case fileCount = "file_count"
    }

    var wire: WireValue {
        .object([
            ("sha256", .string(sha256)),
            ("bytes", .int(bytes)),
            ("file_count", .int(fileCount)),
        ])
    }
}

/// `hello.runner`, additive and LAST on the wire (contract §6.0).
public struct RunnerIdentity: Codable, Sendable, Equatable {
    public var id: String
    public var modelType: String
    public var manifestSHA256: String
    public var build: String

    public init(id: String, modelType: String, manifestSHA256: String, build: String) {
        self.id = id
        self.modelType = modelType
        self.manifestSHA256 = manifestSHA256
        self.build = build
    }

    enum CodingKeys: String, CodingKey {
        case id
        case modelType = "model_type"
        case manifestSHA256 = "manifest_sha256"
        case build
    }

    var wire: WireValue {
        .object([
            ("id", .string(id)),
            ("model_type", .string(modelType)),
            ("manifest_sha256", .string(manifestSHA256)),
            ("build", .string(build)),
        ])
    }
}

/// `hello.resident`, additive and LAST on the wire — after `runner`.
///
/// Identifies the RESIDENT process that actually holds the weights, so every
/// phase of one measurement window can be shown to have run against one
/// load. Residency lives HERE and not in `backend`: that field names the
/// compute backend and the conformance kit checks it against the manifest's. `load_epoch` is constant for a resident's whole life and changes
/// only when a new resident loads, which is exactly the claim an artifact
/// needs: the weights did not reload between phases.
///
/// GATED. benchd's envelope is closed (`deny_unknown_fields`), so a benchd
/// that does not yet know this field rejects the whole hello line and the
/// phase fails. It therefore rides the same `--speculative-protocol v1.1`
/// gate as the other v1.1 additions AND an explicit
/// `BENCH_WORKER_RESIDENT_HELLO=1`, until bench-dev admits it.
public struct ResidentIdentity: Codable, Sendable, Equatable {
    /// The resident process holding the weights.
    public var pid: Int32
    /// Monotonic stamp taken when this resident finished its ONE load.
    public var loadEpoch: UInt64

    public init(pid: Int32, loadEpoch: UInt64) {
        self.pid = pid
        self.loadEpoch = loadEpoch
    }

    enum CodingKeys: String, CodingKey {
        case pid
        case loadEpoch = "load_epoch"
    }

    var wire: WireValue {
        .object([
            ("pid", .int(Int(pid))),
            ("load_epoch", .int(Int(bitPattern: UInt(loadEpoch)))),
        ])
    }
}

/// The trusted oracle's per-stream reference report.
public struct CohortReferenceReplayReport: Sendable, Equatable {
    public struct Position: Sendable, Equatable {
        public var committedToken: Int
        public var sequentialArgmax: Int
        public init(committedToken: Int, sequentialArgmax: Int) {
            self.committedToken = committedToken
            self.sequentialArgmax = sequentialArgmax
        }
    }

    public struct Stream: Sendable, Equatable {
        public var slot: Int
        public var positions: [Position]
        public init(slot: Int, positions: [Position]) {
            self.slot = slot
            self.positions = positions
        }
    }

    public var replayWidth: String
    public var streams: [Stream]

    public init(replayWidth: String, streams: [Stream]) {
        self.replayWidth = replayWidth
        self.streams = streams
    }

    var wire: WireValue {
        .object([
            ("replay_width", .string(replayWidth)),
            (
                "streams",
                .array(
                    streams.map { stream in
                        .object([
                            ("slot", .int(stream.slot)),
                            (
                                "positions",
                                .array(
                                    stream.positions.map { position in
                                        .object([
                                            ("committed_token", .int(position.committedToken)),
                                            (
                                                "sequential_argmax",
                                                .int(position.sequentialArgmax)
                                            ),
                                        ])
                                    })
                            ),
                        ])
                    })
            ),
        ])
    }
}

// MARK: - Responses

/// Engine → benchmarker.
///
/// `Decodable` only: the wire is written by `jsonLine()` for the reasons in
/// `WireValue`, and one writer is better than two that can drift.
public struct WorkerResponse: Decodable, Sendable {
    public var id: Int
    public var nonce: String?
    public var ok: Bool
    public var error: String?
    public var token: Int?
    public var topLogits: [CorrectnessTraceLogit]?
    public var seedToken: Int?
    public var tokens: [Int]?
    public var expertStats: ExpertStreamingStats?
    public var peakRAMGB: Double?
    public var protocolVersion: Int?
    public var backend: String?
    public var device: String?
    public var completedWork: Int?
    public var cacheMemory: Int?
    public var capabilities: [String]?
    public var acceptanceLengths: [Int]?
    public var draftedTotal: Int?
    public var acceptedTotal: Int?
    public var committedTotal: Int?
    public var effectiveSpec: SpecConfig?
    public var specModes: [String]?
    public var headProvenance: WireHeadProvenance?
    public var mlxActiveMemoryBytes: Int?
    public var mlxCacheMemoryBytes: Int?
    public var mlxPeakMemoryBytes: Int?
    public var topLogitMargin: Double?
    public var expectedTokenLogit: Double?
    public var expectedTokenRank: Int?
    public var specDecoder: String?
    public var maxBatchSize: Int?
    public var seedTokenByStream: [Int]?
    public var effectiveBatchSize: Int?
    public var tokensByStream: [[Int]]?
    public var naturalAcceptedByStream: [[Int]]?
    public var rounds: Int?
    public var activeStreamsByRound: [Int]?
    public var depthClampReasons: [String: Int]?
    /// MTP target verification: the configured strategy and how many verify
    /// rounds ran rectangular (one forward per window) vs serial (one forward
    /// per column). Present on speculative windows only.
    public var verificationMode: String?
    public var rectangularVerificationRounds: Int?
    public var serialVerificationRounds: Int?
    public var prefillNsByStream: [Int]?
    public var decodeNsByStream: [Int]?
    public var verifyReplayDisagreements: Int?
    public var cohortReferenceReplay: CohortReferenceReplayReport?
    public var runner: RunnerIdentity?
    /// See ``ResidentIdentity``. Appended after `runner`.
    public var resident: ResidentIdentity?
    /// The resident's own checkpoint path, exchanged ONLY on the socket
    /// between a resident and the worker attaching to it so the worker can
    /// refuse a resident holding different weights. Never forwarded to
    /// benchd.
    public var residentWeights: String?

    public init(id: Int, ok: Bool, nonce: String? = nil) {
        self.id = id
        self.ok = ok
        self.nonce = nonce
    }

    enum CodingKeys: String, CodingKey {
        case id
        case nonce
        case ok
        case error
        case token
        case topLogits = "top_logits"
        case seedToken = "seed_token"
        case tokens
        case expertStats = "expert_stats"
        case peakRAMGB = "peak_ram_gb"
        case protocolVersion = "protocol_version"
        case backend
        case device
        case completedWork = "completed_work"
        case cacheMemory = "cache_memory"
        case capabilities
        case acceptanceLengths = "acceptance_lengths"
        case draftedTotal = "drafted_total"
        case acceptedTotal = "accepted_total"
        case committedTotal = "committed_total"
        case effectiveSpec = "effective_spec"
        case specModes = "spec_modes"
        case headProvenance = "head_provenance"
        case mlxActiveMemoryBytes = "mlx_active_memory_bytes"
        case mlxCacheMemoryBytes = "mlx_cache_memory_bytes"
        case mlxPeakMemoryBytes = "mlx_peak_memory_bytes"
        case topLogitMargin = "top_logit_margin"
        case expectedTokenLogit = "expected_token_logit"
        case expectedTokenRank = "expected_token_rank"
        case specDecoder = "spec_decoder"
        case maxBatchSize = "max_batch_size"
        case seedTokenByStream = "seed_token_by_stream"
        case effectiveBatchSize = "effective_batch_size"
        case tokensByStream = "tokens_by_stream"
        case naturalAcceptedByStream = "natural_accepted_by_stream"
        case rounds
        case activeStreamsByRound = "active_streams_by_round"
        case depthClampReasons = "depth_clamp_reasons"
        case verificationMode = "verification_mode"
        case rectangularVerificationRounds = "rectangular_verification_rounds"
        case serialVerificationRounds = "serial_verification_rounds"
        case prefillNsByStream = "prefill_ns_by_stream"
        case decodeNsByStream = "decode_ns_by_stream"
        case verifyReplayDisagreements = "verify_replay_disagreements"
        case runner
        case resident
        case residentWeights = "resident_weights"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(Int.self, forKey: .id)
        ok = try c.decode(Bool.self, forKey: .ok)
        nonce = try c.decodeIfPresent(String.self, forKey: .nonce)
        error = try c.decodeIfPresent(String.self, forKey: .error)
        token = try c.decodeIfPresent(Int.self, forKey: .token)
        topLogits = try c.decodeIfPresent([CorrectnessTraceLogit].self, forKey: .topLogits)
        seedToken = try c.decodeIfPresent(Int.self, forKey: .seedToken)
        tokens = try c.decodeIfPresent([Int].self, forKey: .tokens)
        expertStats = try c.decodeIfPresent(ExpertStreamingStats.self, forKey: .expertStats)
        peakRAMGB = try c.decodeIfPresent(Double.self, forKey: .peakRAMGB)
        protocolVersion = try c.decodeIfPresent(Int.self, forKey: .protocolVersion)
        backend = try c.decodeIfPresent(String.self, forKey: .backend)
        device = try c.decodeIfPresent(String.self, forKey: .device)
        completedWork = try c.decodeIfPresent(Int.self, forKey: .completedWork)
        cacheMemory = try c.decodeIfPresent(Int.self, forKey: .cacheMemory)
        capabilities = try c.decodeIfPresent([String].self, forKey: .capabilities)
        acceptanceLengths = try c.decodeIfPresent([Int].self, forKey: .acceptanceLengths)
        draftedTotal = try c.decodeIfPresent(Int.self, forKey: .draftedTotal)
        acceptedTotal = try c.decodeIfPresent(Int.self, forKey: .acceptedTotal)
        committedTotal = try c.decodeIfPresent(Int.self, forKey: .committedTotal)
        effectiveSpec = try c.decodeIfPresent(SpecConfig.self, forKey: .effectiveSpec)
        specModes = try c.decodeIfPresent([String].self, forKey: .specModes)
        headProvenance = try c.decodeIfPresent(WireHeadProvenance.self, forKey: .headProvenance)
        mlxActiveMemoryBytes = try c.decodeIfPresent(Int.self, forKey: .mlxActiveMemoryBytes)
        mlxCacheMemoryBytes = try c.decodeIfPresent(Int.self, forKey: .mlxCacheMemoryBytes)
        mlxPeakMemoryBytes = try c.decodeIfPresent(Int.self, forKey: .mlxPeakMemoryBytes)
        topLogitMargin = try c.decodeIfPresent(Double.self, forKey: .topLogitMargin)
        expectedTokenLogit = try c.decodeIfPresent(Double.self, forKey: .expectedTokenLogit)
        expectedTokenRank = try c.decodeIfPresent(Int.self, forKey: .expectedTokenRank)
        specDecoder = try c.decodeIfPresent(String.self, forKey: .specDecoder)
        maxBatchSize = try c.decodeIfPresent(Int.self, forKey: .maxBatchSize)
        seedTokenByStream = try c.decodeIfPresent([Int].self, forKey: .seedTokenByStream)
        effectiveBatchSize = try c.decodeIfPresent(Int.self, forKey: .effectiveBatchSize)
        tokensByStream = try c.decodeIfPresent([[Int]].self, forKey: .tokensByStream)
        naturalAcceptedByStream = try c.decodeIfPresent(
            [[Int]].self, forKey: .naturalAcceptedByStream)
        rounds = try c.decodeIfPresent(Int.self, forKey: .rounds)
        activeStreamsByRound = try c.decodeIfPresent([Int].self, forKey: .activeStreamsByRound)
        depthClampReasons = try c.decodeIfPresent([String: Int].self, forKey: .depthClampReasons)
        verificationMode = try c.decodeIfPresent(String.self, forKey: .verificationMode)
        rectangularVerificationRounds = try c.decodeIfPresent(Int.self, forKey: .rectangularVerificationRounds)
        serialVerificationRounds = try c.decodeIfPresent(Int.self, forKey: .serialVerificationRounds)
        prefillNsByStream = try c.decodeIfPresent([Int].self, forKey: .prefillNsByStream)
        decodeNsByStream = try c.decodeIfPresent([Int].self, forKey: .decodeNsByStream)
        verifyReplayDisagreements = try c.decodeIfPresent(
            Int.self, forKey: .verifyReplayDisagreements)
        runner = try c.decodeIfPresent(RunnerIdentity.self, forKey: .runner)
        resident = try c.decodeIfPresent(ResidentIdentity.self, forKey: .resident)
        residentWeights = try c.decodeIfPresent(String.self, forKey: .residentWeights)
        cohortReferenceReplay = nil
    }

    /// The response as one NDJSON line, keys in schema order.
    public func jsonLine() -> String {
        var fields: [(String, WireValue)] = []
        func put(_ name: String, _ value: WireValue?) {
            guard let value else { return }
            fields.append((name, value))
        }
        put("id", .int(id))
        put("nonce", nonce.map(WireValue.string))
        put("ok", .bool(ok))
        put("error", error.map(WireValue.string))
        put("token", token.map(WireValue.int))
        put("top_logits", topLogits.map { .array($0.map(\.wire)) })
        put("seed_token", seedToken.map(WireValue.int))
        put("tokens", tokens.map { .array($0.map(WireValue.int)) })
        put("expert_stats", expertStats?.wire)
        put("peak_ram_gb", peakRAMGB.map(WireValue.double))
        put("protocol_version", protocolVersion.map(WireValue.int))
        put("backend", backend.map(WireValue.string))
        put("device", device.map(WireValue.string))
        put("completed_work", completedWork.map(WireValue.int))
        put("cache_memory", cacheMemory.map(WireValue.int))
        put("capabilities", capabilities.map { .array($0.map(WireValue.string)) })
        put("acceptance_lengths", acceptanceLengths.map { .array($0.map(WireValue.int)) })
        put("drafted_total", draftedTotal.map(WireValue.int))
        put("accepted_total", acceptedTotal.map(WireValue.int))
        put("committed_total", committedTotal.map(WireValue.int))
        put("effective_spec", effectiveSpec?.wire)
        put("spec_modes", specModes.map { .array($0.map(WireValue.string)) })
        put("head_provenance", headProvenance?.wire)
        put("mlx_active_memory_bytes", mlxActiveMemoryBytes.map(WireValue.int))
        put("mlx_cache_memory_bytes", mlxCacheMemoryBytes.map(WireValue.int))
        put("mlx_peak_memory_bytes", mlxPeakMemoryBytes.map(WireValue.int))
        put("top_logit_margin", topLogitMargin.map(WireValue.double))
        put("expected_token_logit", expectedTokenLogit.map(WireValue.double))
        put("expected_token_rank", expectedTokenRank.map(WireValue.int))
        put("spec_decoder", specDecoder.map(WireValue.string))
        put("max_batch_size", maxBatchSize.map(WireValue.int))
        put("seed_token_by_stream", seedTokenByStream.map { .array($0.map(WireValue.int)) })
        put("effective_batch_size", effectiveBatchSize.map(WireValue.int))
        put(
            "tokens_by_stream",
            tokensByStream.map { .array($0.map { .array($0.map(WireValue.int)) }) })
        put(
            "natural_accepted_by_stream",
            naturalAcceptedByStream.map { .array($0.map { .array($0.map(WireValue.int)) }) })
        put("rounds", rounds.map(WireValue.int))
        put(
            "active_streams_by_round",
            activeStreamsByRound.map { .array($0.map(WireValue.int)) })
        put(
            "depth_clamp_reasons",
            depthClampReasons.map { reasons in
                .object(reasons.keys.sorted().map { ($0, .int(reasons[$0]!)) })
            })
        put("verification_mode", verificationMode.map(WireValue.string))
        put("rectangular_verification_rounds", rectangularVerificationRounds.map(WireValue.int))
        put("serial_verification_rounds", serialVerificationRounds.map(WireValue.int))
        put("prefill_ns_by_stream", prefillNsByStream.map { .array($0.map(WireValue.int)) })
        put("decode_ns_by_stream", decodeNsByStream.map { .array($0.map(WireValue.int)) })
        put("verify_replay_disagreements", verifyReplayDisagreements.map(WireValue.int))
        put("cohort_reference_replay", cohortReferenceReplay?.wire)
        put("runner", runner?.wire)
        put("resident", resident?.wire)
        put("resident_weights", residentWeights.map(WireValue.string))
        return WireValue.object(fields).json
    }
}
