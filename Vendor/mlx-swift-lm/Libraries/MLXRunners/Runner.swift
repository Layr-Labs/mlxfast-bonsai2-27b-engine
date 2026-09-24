// Copyright © 2026 Eigen Labs.
//
// MLXRunners — the runner boundary (Darkbloom runner contract v0.2, §5/§6/§9).
//
// A runner is ONE model family behind the CBv2 engine, plus a static
// manifest describing what that family can do. Two consumers drive it and
// neither carries family-specific construction code:
//
//   * Darkbloom, in process, for serving;
//   * benchd, through the generic `bench-worker` executable, over Engine
//     Protocol v1.
//
// Policy stays with the CALLER. `EngineBuild` carries KV sizing, backend
// selection, scheduler/loop configuration, and the prefix cache; the runner
// supplies the model, its layer kinds, its per-layer caches, its
// capabilities, and its drafter. That split is what lets one runner serve a
// provider slot (SSD prefix cache, paged preflight, slot vetoes) and a
// benchmark worker (fixed contiguous, no cache, no queue) without either
// consumer knowing the family.

import CryptoKit
import Foundation
import MLX
import MLXLMCommon

// MARK: - Declaration vocabulary

/// One decoder mode a runner can resolve. The raw value is the wire spelling
/// (`SpecConfig.mode` in Engine Protocol v1) and the `hello.spec_modes`
/// element.
public struct DecoderID: Hashable, Sendable, Codable, RawRepresentable,
    CustomStringConvertible
{
    public let rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }
    public init(_ rawValue: String) { self.rawValue = rawValue }
    public var description: String { rawValue }

    /// No speculation: one target forward per committed token.
    public static let serial = DecoderID("serial")
    /// Multi-token prediction, embedded head or assistant checkpoint.
    public static let mtp = DecoderID("mtp")
    /// DFlash speculative decoding.
    public static let dflash = DecoderID("dflash")
    /// DSpark speculative decoding (reserved).
    public static let dspark = DecoderID("dspark")
}

/// Where a decoder's draft tokens come from.
public enum DrafterKind: String, Sendable, Codable, Equatable {
    case none
    case embeddedHead
    case assistantCheckpoint
    case ngram
}

/// Whether a drafter carries per-request state across rounds.
public enum DrafterState: String, Sendable, Codable, Equatable {
    case stateless
    case requestStateful
}

/// The KV storage backends a runner will serve on.
public enum KVBackendKind: String, Sendable, Codable, Equatable {
    case contiguous
    case paged
}

/// Free-run (the engine picks the next token) versus teacher-forced (the
/// caller forces it).
public enum TimingKind: String, Sendable, Codable, Equatable {
    case freeRun
    case teacherForced
}

/// Cohort width a regime serves.
///
/// Wire encoding pinned by contract §6.0: `.single` is the STRING
/// `"single"`, `.upTo(n)` is the OBJECT `{"upTo": n}`. Both sides
/// (bench-dev and this fork) hand-write the conformance — Swift's
/// synthesized enum form (`{"upTo":{"_0":n}}`) is wrong on the wire and
/// would change every digest.
public enum BatchDeclaration: Sendable, Codable, Equatable {
    case single
    case upTo(Int)

    /// Largest width this declaration serves.
    public var maxWidth: Int {
        switch self {
        case .single: return 1
        case .upTo(let n): return n
        }
    }

    private enum ObjectKeys: String, CodingKey {
        case upTo
    }

    public init(from decoder: Swift.Decoder) throws {
        if let single = try? decoder.singleValueContainer(),
            let raw = try? single.decode(String.self)
        {
            guard raw == "single" else {
                throw DecodingError.dataCorruptedError(
                    in: single, debugDescription: "unknown batch declaration \(raw)")
            }
            self = .single
            return
        }
        let container = try decoder.container(keyedBy: ObjectKeys.self)
        self = .upTo(try container.decode(Int.self, forKey: .upTo))
    }

    public func encode(to encoder: Swift.Encoder) throws {
        switch self {
        case .single:
            var container = encoder.singleValueContainer()
            try container.encode("single")
        case .upTo(let width):
            var container = encoder.container(keyedBy: ObjectKeys.self)
            try container.encode(width, forKey: .upTo)
        }
    }
}

/// Provenance of the drafter artifact a runner actually loaded. Sealed on
/// the hello (`hello.head_provenance`); audit only, never scored.
public struct HeadProvenance: Sendable, Codable, Equatable {
    public let sha256: String
    public let bytes: Int
    public let fileCount: Int

    public init(sha256: String, bytes: Int, fileCount: Int) {
        self.sha256 = sha256
        self.bytes = bytes
        self.fileCount = fileCount
    }

    enum CodingKeys: String, CodingKey {
        case sha256
        case bytes
        case fileCount = "file_count"
    }
}

// MARK: - Manifest

/// One decoder mode the runner declares it can resolve.
public struct DecoderDeclaration: Sendable, Codable, Equatable {
    /// Wire spelling of the mode (`DecoderID.rawValue`).
    public let mode: String
    public let drafter: DrafterKind
    public let state: DrafterState
    /// Draft depths the mode covers; nil for `serial`.
    public let depth: ClosedRange<Int>?

    public init(
        mode: String, drafter: DrafterKind, state: DrafterState,
        depth: ClosedRange<Int>?
    ) {
        self.mode = mode
        self.drafter = drafter
        self.state = state
        self.depth = depth
    }

    public var id: DecoderID { DecoderID(mode) }

    /// Frozen field order. The digest is taken over this order.
    enum CodingKeys: String, CodingKey {
        case mode
        case drafter
        case state
        case depth
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        mode = try container.decode(String.self, forKey: .mode)
        drafter = try container.decode(DrafterKind.self, forKey: .drafter)
        state = try container.decode(DrafterState.self, forKey: .state)
        if let bounds = try container.decodeIfPresent([Int].self, forKey: .depth) {
            guard bounds.count == 2, bounds[0] <= bounds[1] else {
                throw DecodingError.dataCorruptedError(
                    forKey: .depth, in: container,
                    debugDescription: "depth must be [low, high] with low <= high")
            }
            depth = bounds[0] ... bounds[1]
        } else {
            depth = nil
        }
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(mode, forKey: .mode)
        try container.encode(drafter, forKey: .drafter)
        try container.encode(state, forKey: .state)
        try container.encode(
            depth.map { [$0.lowerBound, $0.upperBound] }, forKey: .depth)
    }
}

/// One measurement regime the runner declares it can serve.
public struct RegimeDeclaration: Sendable, Codable, Equatable {
    public let batch: BatchDeclaration
    public let timing: TimingKind
    public let perStreamTiming: Bool

    public init(batch: BatchDeclaration, timing: TimingKind, perStreamTiming: Bool) {
        self.batch = batch
        self.timing = timing
        self.perStreamTiming = perStreamTiming
    }

    /// Frozen field order.
    enum CodingKeys: String, CodingKey {
        case batch
        case timing
        case perStreamTiming
    }
}

/// Static, digest-stable declaration of one runner.
///
/// Field ORDER is load bearing: `canonicalJSON()` emits the fields in the
/// order declared by `CodingKeys` below, and `sha256Digest()` hashes those
/// exact bytes. Sorted keys would NOT be enough — a re-ordering of the
/// declaration would then be invisible, and the whole point of pinning a
/// digest is that a manifest change is visible to whoever pinned it.
public struct RunnerManifest: Sendable, Codable, Equatable {
    public let schemaVersion: Int
    public let runnerID: String
    public let modelTypes: [String]
    public let backend: String
    /// MANDATORY explicit declaration. `.attentionOnly` by default is not
    /// accepted (§6.2 rule 2) — every field is spelled out by the runner.
    public let engine: CBv2ModelCapabilities
    public let kvBackends: [KVBackendKind]
    public let decoders: [DecoderDeclaration]
    public let regimes: [RegimeDeclaration]
    public let multimodal: Bool
    public let recurrentLayers: Bool
    /// §10 seam. DATA ONLY in the scaffold: the `keepMask` seam itself lands
    /// with the Qwen 3.8 Flash runner. Nothing here reads this field.
    public let requiresKeepMask: Bool

    public init(
        schemaVersion: Int = 1,
        runnerID: String,
        modelTypes: [String],
        backend: String = "mlx",
        engine: CBv2ModelCapabilities,
        kvBackends: [KVBackendKind],
        decoders: [DecoderDeclaration],
        regimes: [RegimeDeclaration],
        multimodal: Bool,
        recurrentLayers: Bool,
        requiresKeepMask: Bool
    ) {
        self.schemaVersion = schemaVersion
        self.runnerID = runnerID
        self.modelTypes = modelTypes
        self.backend = backend
        self.engine = engine
        self.kvBackends = kvBackends
        self.decoders = decoders
        self.regimes = regimes
        self.multimodal = multimodal
        self.recurrentLayers = recurrentLayers
        self.requiresKeepMask = requiresKeepMask
    }

    /// FROZEN field order. Changing it changes every digest.
    enum CodingKeys: String, CodingKey {
        case schemaVersion
        case runnerID
        case modelTypes
        case backend
        case engine
        case kvBackends
        case decoders
        case regimes
        case multimodal
        case recurrentLayers
        case requiresKeepMask
    }

    // MARK: Canonical bytes

    /// Deterministic UTF-8 bytes of this manifest.
    ///
    /// Hand-written rather than `JSONEncoder(.sortedKeys)`: sorted keys are
    /// a DIFFERENT order from the declared one and would hide a field
    /// re-ordering behind a stable digest. No whitespace, no escaping beyond
    /// the JSON minimum, integers only (there are no floating-point fields),
    /// so the bytes are reproducible across platforms and Foundation
    /// versions.
    public func canonicalJSON() -> Data {
        var out = "{"
        out += CanonicalJSON.field("schemaVersion", CanonicalJSON.number(schemaVersion))
        out += "," + CanonicalJSON.field("runnerID", CanonicalJSON.string(runnerID))
        out += ","
            + CanonicalJSON.field(
                "modelTypes", CanonicalJSON.array(modelTypes.map(CanonicalJSON.string)))
        out += "," + CanonicalJSON.field("backend", CanonicalJSON.string(backend))
        out += "," + CanonicalJSON.field("engine", Self.canonicalEngine(engine))
        out += ","
            + CanonicalJSON.field(
                "kvBackends",
                CanonicalJSON.array(kvBackends.map { CanonicalJSON.string($0.rawValue) }))
        out += ","
            + CanonicalJSON.field(
                "decoders", CanonicalJSON.array(decoders.map(Self.canonicalDecoder)))
        out += ","
            + CanonicalJSON.field(
                "regimes", CanonicalJSON.array(regimes.map(Self.canonicalRegime)))
        out += "," + CanonicalJSON.field("multimodal", CanonicalJSON.bool(multimodal))
        out += ","
            + CanonicalJSON.field("recurrentLayers", CanonicalJSON.bool(recurrentLayers))
        out += ","
            + CanonicalJSON.field("requiresKeepMask", CanonicalJSON.bool(requiresKeepMask))
        out += "}"
        return Data(out.utf8)
    }

    /// Lowercase hex sha256 of `canonicalJSON()`. This is what the hello's
    /// `runner.manifest_sha256` carries and what the fork's tests pin.
    public func sha256Digest() -> String {
        SHA256.hash(data: canonicalJSON())
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func canonicalEngine(_ engine: CBv2ModelCapabilities) -> String {
        var out = "{"
        out += CanonicalJSON.field(
            "supportsPrefixReuse", CanonicalJSON.bool(engine.supportsPrefixReuse))
        out += ","
            + CanonicalJSON.field(
                "supportsPagedKV", CanonicalJSON.bool(engine.supportsPagedKV))
        out += ","
            + CanonicalJSON.field(
                "supportsCompiledDecode", CanonicalJSON.bool(engine.supportsCompiledDecode))
        out += ","
            + CanonicalJSON.field(
                "supportsPackedPrefill", CanonicalJSON.bool(engine.supportsPackedPrefill))
        out += "," + CanonicalJSON.field("supportsMTP", CanonicalJSON.bool(engine.supportsMTP))
        out += ","
            + CanonicalJSON.field(
                "supportsCompactRecurrentMTPReplay",
                CanonicalJSON.bool(engine.supportsCompactRecurrentMTPReplay))
        out += "}"
        return out
    }

    private static func canonicalDecoder(_ decoder: DecoderDeclaration) -> String {
        var out = "{"
        out += CanonicalJSON.field("mode", CanonicalJSON.string(decoder.mode))
        out += "," + CanonicalJSON.field("drafter", CanonicalJSON.string(decoder.drafter.rawValue))
        out += "," + CanonicalJSON.field("state", CanonicalJSON.string(decoder.state.rawValue))
        let depth =
            decoder.depth.map {
                CanonicalJSON.array([
                    CanonicalJSON.number($0.lowerBound), CanonicalJSON.number($0.upperBound),
                ])
            } ?? "null"
        out += "," + CanonicalJSON.field("depth", depth)
        out += "}"
        return out
    }

    private static func canonicalRegime(_ regime: RegimeDeclaration) -> String {
        let batch: String
        switch regime.batch {
        case .single:
            batch = CanonicalJSON.string("single")
        case .upTo(let width):
            batch = "{" + CanonicalJSON.field("upTo", CanonicalJSON.number(width)) + "}"
        }
        var out = "{"
        out += CanonicalJSON.field("batch", batch)
        out += "," + CanonicalJSON.field("timing", CanonicalJSON.string(regime.timing.rawValue))
        out += ","
            + CanonicalJSON.field(
                "perStreamTiming", CanonicalJSON.bool(regime.perStreamTiming))
        out += "}"
        return out
    }
}

/// Minimal deterministic JSON writer for the manifest digest.
enum CanonicalJSON {
    static func field(_ name: String, _ value: String) -> String {
        "\(string(name)):\(value)"
    }

    static func string(_ value: String) -> String {
        var out = "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }

    static func number(_ value: Int) -> String { String(value) }
    static func bool(_ value: Bool) -> String { value ? "true" : "false" }
    static func array(_ elements: [String]) -> String { "[" + elements.joined(separator: ",") + "]" }
}

// MARK: - Load options and engine build

/// Objects a runner needs at load time and cannot build itself, keyed by a
/// name the runner declares.
///
/// One family needs this today: the Qwen 3.8 Flash-Next n-gram table is 29.8
/// GiB, is never held as model parameters, and its disk-resident
/// implementation lives in the track repo rather than the fork. The seam is
/// deliberately opaque — the runner casts to the protocol it wants — so a
/// track repo can hand a fork runner an implementation the fork does not
/// know about.
public struct RunnerResources: @unchecked Sendable {
    private var storage: [String: AnyObject] = [:]

    public init() {}

    public subscript(name: String) -> AnyObject? {
        get { storage[name] }
        set { storage[name] = newValue }
    }
}

/// What the CALLER wants honored at load time. The runner never downloads
/// and never reads the network (§5 rule 4).
///
/// `@unchecked Sendable` because `preloadedDrafter` is a live module the
/// caller already holds; the options struct is handed across the load
/// boundary and never mutated after it.
public struct RunnerLoadOptions: @unchecked Sendable {
    /// Directory holding the drafter artifact (assistant checkpoint or a
    /// standalone embedded-head export). nil means "serial only".
    public var drafterDirectory: URL?
    /// KV byte capacity the caller intends to grant the engine. Runners that
    /// size anything at load time read it here; the ENGINE's ceiling still
    /// arrives on `EngineBuild`.
    public var kvBytesCapacity: Int
    /// Longest single sequence the runner's own one-row stepper reserves
    /// for. The ENGINE's context bound is the caller's scheduler config;
    /// this is the stepper's row ceiling only.
    public var maxSequenceLength: Int
    /// Environment the caller wants honored (kill switches, dtype knobs).
    /// Passed explicitly rather than read from the process so a test can
    /// drive a runner hermetically.
    public var environment: [String: String]
    /// Family resources the caller supplies. See ``RunnerResources``.
    public var resources: RunnerResources
    /// A drafter the caller ALREADY holds resident.
    ///
    /// Darkbloom's slot lifecycle keeps the Gemma 4 assistant loaded across
    /// engine rebuilds, so handing it in is the difference between binding a
    /// module and reading its tensors a second time. When set it is used as
    /// is and `drafterDirectory` is not opened.
    public var preloadedDrafter: (any CBv2MTPDrafter)?

    public init(
        drafterDirectory: URL? = nil,
        kvBytesCapacity: Int = 0,
        maxSequenceLength: Int = 32768,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        resources: RunnerResources = RunnerResources(),
        preloadedDrafter: (any CBv2MTPDrafter)? = nil
    ) {
        self.drafterDirectory = drafterDirectory
        self.kvBytesCapacity = kvBytesCapacity
        self.maxSequenceLength = maxSequenceLength
        self.environment = environment
        self.resources = resources
        self.preloadedDrafter = preloadedDrafter
    }
}

/// Arithmetic the paged pool's pages are stored IN.
///
/// This is a MEASUREMENT knob, not a tuning one. An operator who asks for
/// fp32 and is served fp16 has run a different experiment than they believe,
/// and a control arm that is secretly a second copy of the baseline looks
/// exactly like agreement — so a request here is honoured exactly or refused
/// by name, never quietly downgraded.
///
/// Capacity is a real wall at fp32: a page costs
/// `2 * kvHeads * pageSize * headDim * dtype.size` bytes, so fp32 pages cost
/// exactly twice as much and the same byte grant buys HALF the pages, while
/// page DEMAND is unchanged. A configuration that fits at fp16 and refuses
/// at fp32 is CORRECT — refused rather than quietly served at half the
/// context.
public enum PagedPoolDType: String, Sendable, Equatable, Codable {
    case float16
    case float32

    public var dtype: DType {
        switch self {
        case .float16: return .float16
        case .float32: return .float32
        }
    }
}

/// The paged pool a caller asks for. Ignored entirely by a contiguous build.
public struct PagedPoolPlan: Sendable, Equatable {
    /// Page arithmetic. See ``PagedPoolDType``.
    public var dtype: PagedPoolDType
    /// When the slabs are wired: at construction (allocation out of the
    /// timed region, an idle pool holding unified memory) or at the pool's
    /// first admission (the production posture, so an unused pool costs a
    /// co-resident model's headroom measurement nothing).
    public var slabCommitment: PagedKVSlabCommitment
    /// Sequence length the pool sizes its groups against.
    public var nominalMaxSequenceLength: Int
    /// Byte capacity for the POOL, when it differs from the engine's
    /// admission ceiling. nil means `EngineBuild.kvBytesCapacity`.
    public var capacityBytes: Int?
    /// Device buffer ceiling; nil reads the device's own.
    public var maxBufferLength: Int?

    public init(
        dtype: PagedPoolDType = .float16,
        slabCommitment: PagedKVSlabCommitment = .atFirstAdmission,
        nominalMaxSequenceLength: Int = RunnerEngineAssembly.nominalMaxSequenceLength,
        capacityBytes: Int? = nil,
        maxBufferLength: Int? = nil
    ) {
        self.dtype = dtype
        self.slabCommitment = slabCommitment
        self.nominalMaxSequenceLength = nominalMaxSequenceLength
        self.capacityBytes = capacityBytes
        self.maxBufferLength = maxBufferLength
    }
}

/// Policy the CALLER owns (§9). Darkbloom fills this from its slot factory;
/// bench-worker fills it from the fixed contiguous benchmark build.
public struct EngineBuild: Sendable {
    /// Caller's backend choice. The runner REFUSES a kind absent from
    /// `manifest.kvBackends` rather than quietly serving the other one.
    public var kvBackend: KVBackendKind
    public var kvBytesCapacity: Int
    public var schedulerConfig: CBv2SchedulerConfig
    public var loopConfig: CBv2EngineLoopConfig
    public var prefixCache: (any CBv2PrefixCache)?
    /// Must be in `runner.loadedDecoders`.
    public var decoder: DecoderID
    public var mtpConfig: CBv2MTPConfig
    /// The paged pool this build asks for. Read only when `kvBackend` is
    /// `.paged`; a contiguous build has no pages to describe.
    public var pagedPool: PagedPoolPlan
    public var environment: [String: String]

    public init(
        kvBackend: KVBackendKind,
        kvBytesCapacity: Int,
        schedulerConfig: CBv2SchedulerConfig = CBv2SchedulerConfig(),
        loopConfig: CBv2EngineLoopConfig = CBv2EngineLoopConfig(),
        prefixCache: (any CBv2PrefixCache)? = nil,
        decoder: DecoderID = .serial,
        mtpConfig: CBv2MTPConfig = CBv2MTPConfig(),
        pagedPool: PagedPoolPlan = PagedPoolPlan(),
        environment: [String: String] = [:]
    ) {
        self.kvBackend = kvBackend
        self.kvBytesCapacity = kvBytesCapacity
        self.schedulerConfig = schedulerConfig
        self.loopConfig = loopConfig
        self.prefixCache = prefixCache
        self.decoder = decoder
        self.mtpConfig = mtpConfig
        self.pagedPool = pagedPool
        self.environment = environment
    }
}

// MARK: - Runner

/// Refusals every runner shares. Each one is a REFUSAL to build something
/// the manifest does not cover — never a quiet substitution.
public enum RunnerError: Error, CustomStringConvertible, Equatable {
    /// `config.json` was missing, unreadable, or carried no `model_type`.
    case invalidCheckpoint(String)
    /// The loaded module is not the family this runner claims.
    case unexpectedModel(String)
    /// `EngineBuild.kvBackend` is not in `manifest.kvBackends`.
    case kvBackendRefused(requested: String, declared: [String])
    /// `EngineBuild.decoder` is not in `runner.loadedDecoders`. A worker
    /// that resolves a mode it did not advertise is a bug, not a fallback
    /// (§6.2 rule 1).
    case decoderNotLoaded(requested: String, loaded: [String])
    /// The drafter artifact could not be loaded from the given directory.
    case drafterUnavailable(String)
    /// A resource the runner cannot build itself was absent from
    /// `RunnerLoadOptions.resources`.
    case resourceMissing(String)
    /// The loaded norm weights contradict the configured RMSNorm weight
    /// offset. Applying the wrong convention scales every non-gated norm in
    /// the tower by about one unit and produces incoherent output while
    /// every shape and digest still checks out, so this REFUSES rather than
    /// runs.
    case normConventionMismatch(expectedOffset: Float, observed: String)
    /// The paged pool was built with page arithmetic other than the one the
    /// caller asked for. An explicit paged request is honoured exactly or
    /// refused BY NAME: serving fp16 under an fp32 label would make a
    /// control arm a second copy of the baseline.
    case pagedPoolDTypeUnsupported(requested: String, served: String)

    public var description: String {
        switch self {
        case .invalidCheckpoint(let detail):
            return "runner: checkpoint unusable (\(detail))"
        case .unexpectedModel(let type):
            return "runner: loaded module \(type) is not the declared family"
        case .kvBackendRefused(let requested, let declared):
            return "runner: kv backend \(requested) not declared "
                + "(manifest declares \(declared.joined(separator: ", ")))"
        case .decoderNotLoaded(let requested, let loaded):
            return "runner: decoder \(requested) is not loaded "
                + "(loaded: \(loaded.joined(separator: ", ")))"
        case .drafterUnavailable(let detail):
            return "runner: drafter unavailable (\(detail))"
        case .resourceMissing(let detail):
            return "runner: required resource absent (\(detail))"
        case .normConventionMismatch(let expectedOffset, let observed):
            return "runner: the checkpoint's norm weights contradict the configured "
                + "rms_norm_weight_offset \(expectedOffset) — \(observed). Applying the "
                + "wrong convention produces incoherent output while every shape and "
                + "digest still checks out"
        case .pagedPoolDTypeUnsupported(let requested, let served):
            return "runner: paged pool requested \(requested) but was built \(served)"
        }
    }
}

/// One model family behind the CBv2 engine.
///
/// `static func load` returns `Self`, as the contract writes it. Swift's
/// covariant `Self` erasure makes that callable through the existential
/// metatype the registry stores (`any Runner.Type` → `any Runner`), so no
/// signature change was needed.
public protocol Runner: AnyObject, Sendable {
    /// Static declaration. Every hello field derives from this plus the
    /// loaded state (§6.1).
    static var manifest: RunnerManifest { get }

    /// Adopt a module the caller ALREADY has resident.
    ///
    /// Reads NO tensors. Everything it derives — the serving model after
    /// tower extraction, the layer kinds, the loaded decoders, the head
    /// provenance — comes from the module it was handed plus `config.json`
    /// and the safetensors index under `directory`.
    ///
    /// This exists because Darkbloom's slot lifecycle owns a resident
    /// `ModelContainer`: a runner that could only `load` would read a
    /// checkpoint the process already holds, and on a 113 GB model that is
    /// not a slow path, it is a second copy in unified memory.
    ///
    /// `configuration` is carried for runners that key on the resolved
    /// model identity; the authoritative `model_type` is read from
    /// `config.json`, which is what the hello reports.
    static func adopt(
        model: any LanguageModel,
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> Self

    /// Load the family's drafter from disk and bind it to `target` (the RAW
    /// loaded module — the family extracts its own tower).
    ///
    /// Separate from `adopt` precisely because it READS TENSORS. Default:
    /// whatever the caller already handed in, which is nil for a family with
    /// no drafter.
    static func loadDrafter(
        options: RunnerLoadOptions,
        directory: URL,
        target: any LanguageModel
    ) async throws -> (any CBv2MTPDrafter)?

    /// Load the checkpoint. Loads weights ONCE. No download. No network.
    ///
    /// Defaulted in terms of `adopt`, so construction has ONE path: bring
    /// the module into memory, then adopt it.
    static func load(_ directory: URL, options: RunnerLoadOptions) async throws -> Self

    /// The serving model after tower extraction (VLM text tower, MoE target).
    var servingModel: any LanguageModel { get }
    var tokenizer: any Tokenizer { get }
    var eosTokenIDs: Set<Int> { get }

    /// Per-layer attention structure. Model-owned. One entry per attending
    /// layer the engine stores KV for.
    var layerKinds: [CBv2LayerKind] { get }

    /// Which decoders actually loaded. Subset of `manifest.decoders`. A
    /// decoder is present only if its drafter is resident.
    var loadedDecoders: [DecoderID] { get }

    /// Provenance of the loaded drafter artifact, if any. Sealed on hello.
    var headProvenance: HeadProvenance? { get }

    /// The `--verbose` load summary. A protocol REQUIREMENT, not only an
    /// extension default: the worker holds an `any Runner`, and a method that
    /// lives only in the extension dispatches statically to the generic half
    /// — a box ran with the family's lines silently missing. The default
    /// implementation is the generic summary; a family overrides to add its
    /// own lines.
    func loadSummary(weights: URL, options: RunnerLoadOptions) -> RunnerLoadSummary

    /// The `model_type` string read out of the loaded checkpoint's
    /// `config.json`. Rides the hello's `runner.model_type`.
    var loadedModelType: String { get }

    /// Build the engine. Free-run, cohort, reference replay, serving.
    func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine

    /// Build a single-row teacher-forced stepper over the SAME forward.
    func makeStepper() throws -> any TeacherForcedStepper
}

extension Runner {
    /// Instance-side access to the type's static declaration.
    public var manifest: RunnerManifest { Self.manifest }
}

// MARK: - Checkpoint helpers

/// Reading `config.json` without loading weights. Every runner needs the
/// `model_type` for the hello, and the registry needs it before it can pick
/// a runner at all.
public enum RunnerCheckpoint {
    /// The `model_type` declared by `<directory>/config.json`.
    public static func modelType(at directory: URL) throws -> String {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url) else {
            throw RunnerError.invalidCheckpoint("cannot read \(url.path)")
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modelType = object["model_type"] as? String
        else {
            throw RunnerError.invalidCheckpoint("no model_type in \(url.path)")
        }
        return modelType
    }

    /// Resolved stop-token ids: the tokenizer's own EOS plus every id in
    /// `config.json`'s `eos_token_id` (a scalar or an array, both shapes
    /// occur in the wild). Resolved ONCE at load, exactly as the engine
    /// requires (stop tokens are already-resolved ids on `CBv2Request`).
    public static func eosTokenIDs(at directory: URL, tokenizer: any Tokenizer) -> Set<Int> {
        var ids = Set<Int>()
        if let id = tokenizer.eosTokenId { ids.insert(id) }
        let url = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: url),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            if let single = object["eos_token_id"] as? Int {
                ids.insert(single)
            } else if let many = object["eos_token_id"] as? [Int] {
                ids.formUnion(many)
            }
        }
        return ids
    }

    /// Provenance of a drafter directory: sha256 over the concatenated
    /// safetensors bytes in NAME ORDER, the total byte size, and the file
    /// count. Name order is what makes the digest reproducible — directory
    /// enumeration order is not.
    public static func provenance(ofHeadAt directory: URL) throws -> HeadProvenance {
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
        let shards =
            contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !shards.isEmpty else {
            throw RunnerError.drafterUnavailable(
                "no safetensors shards in \(directory.path)")
        }
        var hasher = SHA256()
        var bytes = 0
        for shard in shards {
            let handle = try FileHandle(forReadingFrom: shard)
            defer { try? handle.close() }
            while let chunk = try handle.read(upToCount: 8 * 1024 * 1024), !chunk.isEmpty {
                hasher.update(data: chunk)
                bytes += chunk.count
            }
        }
        return HeadProvenance(
            sha256: hasher.finalize().map { String(format: "%02x", $0) }.joined(),
            bytes: bytes,
            fileCount: shards.count)
    }

    /// Provenance of an EMBEDDED head — the `mtp.*` block of a checkpoint
    /// that also carries the target weights (contract §12c).
    ///
    /// THE RULE, IN ONE PLACE. Every runner with an embedded head calls this
    /// and none writes its own:
    ///
    /// 1. Find every head tensor and the shard file that holds it. A head
    ///    tensor is one whose name is `mtp.<...>` once the checkpoint's
    ///    wrapper prefixes are stripped: the flattened text tree writes
    ///    `language_model.mtp.*`, an upstream torch checkpoint
    ///    `model.language_model.mtp.*` or `model.mtp.*`, and the module key
    ///    is bare `mtp.*`. The model's own key remap accepts exactly these
    ///    spellings; the scan must accept the same set or it reports "none"
    ///    on a checkpoint whose head the module has bound — which is what a
    ///    box saw (`language_model.mtp.*`, 76 tensors, one shard). The shard map is the checkpoint's
    ///    `*.safetensors.index.json` `weight_map`; a checkpoint with no index
    ///    is read from each shard's own safetensors header.
    /// 2. If there is no such tensor the checkpoint has no head: the result
    ///    is nil, not a digest of the target weights.
    /// 3. sha256 over the WHOLE bytes of the shards that carry those tensors,
    ///    concatenated in index order (shard file name ascending, which is
    ///    the shard numbering), followed by the sorted index entries, one per
    ///    line as `<tensor name>\t<shard file name>\n`, ascending by tensor
    ///    name.
    /// 4. `bytes` is the sum of those shard sizes and `file_count` is how
    ///    many shards they are. Neither counts a shard with no `mtp.` tensor.
    ///
    /// The entry lines are hashed as well as the bytes because two
    /// checkpoints can share a shard file and place the head differently in
    /// it; the digest names WHAT was found, not only where.
    public static func provenance(ofEmbeddedHeadAt directory: URL) throws -> HeadProvenance? {
        let entries = try embeddedHeadIndexEntries(at: directory)
        guard !entries.isEmpty else { return nil }

        let shardNames = Set(entries.map(\.shard)).sorted()
        var hasher = SHA256()
        var bytes = 0
        for name in shardNames {
            let url = directory.appendingPathComponent(name)
            guard let handle = try? FileHandle(forReadingFrom: url) else {
                throw RunnerError.drafterUnavailable("cannot read \(url.path)")
            }
            defer { try? handle.close() }
            while let chunk = try handle.read(upToCount: 8 * 1024 * 1024), !chunk.isEmpty {
                hasher.update(data: chunk)
                bytes += chunk.count
            }
        }
        for entry in entries.sorted(by: { $0.tensor < $1.tensor }) {
            hasher.update(data: Data("\(entry.tensor)\t\(entry.shard)\n".utf8))
        }
        return HeadProvenance(
            sha256: hasher.finalize().map { String(format: "%02x", $0) }.joined(),
            bytes: bytes,
            fileCount: shardNames.count)
    }

    /// Whether an ON-DISK tensor name is part of the embedded head. The
    /// wrapper components the checkpoint may nest the head under are
    /// dropped first; what remains must start at the `mtp` component. This
    /// is the same acceptance set as the model's key remap — keep the two
    /// in step.
    static func isEmbeddedHeadTensor(_ name: String) -> Bool {
        var components = name.split(separator: ".", omittingEmptySubsequences: false)[...]
        while let first = components.first, first == "model" || first == "language_model" {
            components = components.dropFirst()
        }
        return components.count >= 2 && components.first == "mtp"
    }

    /// Head tensor names and the shard file each one lives in.
    private static func embeddedHeadIndexEntries(
        at directory: URL
    ) throws -> [(tensor: String, shard: String)] {
        let contents =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []

        if let index = contents.first(where: {
            $0.lastPathComponent.hasSuffix(".safetensors.index.json")
        }) {
            guard let data = try? Data(contentsOf: index),
                let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let map = object["weight_map"] as? [String: String]
            else {
                throw RunnerError.invalidCheckpoint("no weight_map in \(index.path)")
            }
            return map
                .filter { isEmbeddedHeadTensor($0.key) }
                .map { (tensor: $0.key, shard: $0.value) }
        }

        // No index: one shard per file, and each file names its own tensors.
        var entries: [(tensor: String, shard: String)] = []
        for shard in contents.filter({ $0.pathExtension == "safetensors" }) {
            for tensor in try safetensorsTensorNames(at: shard)
            where isEmbeddedHeadTensor(tensor) {
                entries.append((tensor: tensor, shard: shard.lastPathComponent))
            }
        }
        return entries
    }

    /// Tensor names of one safetensors file, from its header alone: eight
    /// little-endian bytes of header length, then that many bytes of JSON
    /// whose keys are the tensor names plus `__metadata__`.
    private static func safetensorsTensorNames(at url: URL) throws -> [String] {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            throw RunnerError.drafterUnavailable("cannot read \(url.path)")
        }
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw RunnerError.invalidCheckpoint("\(url.path) is not a safetensors file")
        }
        let length = prefix.reversed().reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
        guard length > 0, length <= 512 * 1024 * 1024,
            let header = try handle.read(upToCount: Int(length)),
            header.count == Int(length),
            let object = try? JSONSerialization.jsonObject(with: header) as? [String: Any]
        else {
            throw RunnerError.invalidCheckpoint("\(url.path) has no safetensors header")
        }
        return object.keys.filter { $0 != "__metadata__" }
    }
}

// MARK: - Shared engine assembly

/// The one place `EngineV2` is assembled for a fork runner.
///
/// Lifted from d-inference's `EngineV2Factory+Production` (the code this
/// contract replaces), minus everything that was POLICY there: KV sizing,
/// paged physical-capacity planning, slot vetoes, kill switches, the SSD
/// prefix cache, and the degrade-or-refuse ladder. Those decisions belong to
/// the caller and arrive on `EngineBuild`; what stays here is the wiring
/// that is identical for every family — the steppable adapter, the cache
/// bank over the model's own `newCacheV2`, the default sampler, the text
/// detokenizer factory, and the MTP drafter the runner loaded.
public enum RunnerEngineAssembly {

    /// Nominal maximum sequence length used to size a paged pool when the
    /// caller states none. Same constant the production factory falls back
    /// to; paged SIZING policy is otherwise the caller's.
    public static let nominalMaxSequenceLength = 8192

    public static func makeEngine(
        manifest: RunnerManifest,
        loadedDecoders: [DecoderID],
        model: any LanguageModel,
        tokenizer: any Tokenizer,
        layerKinds: [CBv2LayerKind],
        newCaches: (
            (_ layerIndex: Int, _ kind: CBv2LayerKind) throws -> any CBv2AttendingLayerCache
        ) throws -> [any CBv2AttendingLayerCache],
        mtpDrafter: (any CBv2MTPDrafter)?,
        build: EngineBuild
    ) throws -> any CBv2Engine {
        guard manifest.kvBackends.contains(build.kvBackend) else {
            throw RunnerError.kvBackendRefused(
                requested: build.kvBackend.rawValue,
                declared: manifest.kvBackends.map(\.rawValue))
        }
        guard loadedDecoders.contains(build.decoder) else {
            throw RunnerError.decoderNotLoaded(
                requested: build.decoder.rawValue,
                loaded: loadedDecoders.map(\.rawValue))
        }

        let backend: CBv2KVBackend
        let caches: [any CBv2AttendingLayerCache]
        switch build.kvBackend {
        case .contiguous:
            backend = CBv2ContiguousKVBackend(
                config: CBv2ContiguousBackendConfig(bytesCapacity: build.kvBytesCapacity))
            caches = try newCaches { index, kind in
                CBv2LayerCache(layerIndex: index, kind: kind)
            }
        case .paged:
            let plan = build.pagedPool
            let paged = try PagedKVBackend(
                layerKinds: layerKinds,
                config: PagedKVPoolConfig(
                    capacityBytes: plan.capacityBytes ?? build.kvBytesCapacity,
                    dtype: plan.dtype.dtype,
                    // LOCKSTEP: a windowed-layer update larger than the
                    // ring's provision traps the process, so the pool is
                    // sized from the chunk the engine can actually schedule
                    // — a solo stripe included.
                    maxPrefillChunk: max(
                        build.schedulerConfig.prefillChunkSize,
                        build.schedulerConfig.soloPrefillStripeTokens ?? 0),
                    nominalMaxSequenceLength: plan.nominalMaxSequenceLength,
                    maxBufferLength: plan.maxBufferLength
                        ?? MLX.GPU.deviceInfo().maxBufferSize),
                slabCommitment: plan.slabCommitment)
            // Read off the CONSTRUCTED pool, never off the request: this is
            // the built artifact reporting itself, and it is what separates
            // "fp32 served" from "fp32 asked for and ignored".
            guard paged.pool.config.dtype == plan.dtype.dtype else {
                throw RunnerError.pagedPoolDTypeUnsupported(
                    requested: plan.dtype.rawValue,
                    served: "\(paged.pool.config.dtype)")
            }
            let pagedCaches = paged.makeLayerCaches()
            // `newCacheV2` hands the closure the MODEL layer index
            // (`kind.modelLayerIndex ?? storagePosition`) while
            // `makeLayerCaches()` is dense per STORAGE slot. On a hybrid
            // trunk the two differ, so map rather than subscript.
            var storageForModelIndex: [Int: Int] = [:]
            for (storage, kind) in layerKinds.enumerated() {
                storageForModelIndex[kind.modelLayerIndex ?? storage] = storage
            }
            backend = paged
            caches = try newCaches { index, _ in
                guard let storage = storageForModelIndex[index],
                    storage < pagedCaches.count
                else {
                    preconditionFailure(
                        "paged cache storage mapping missing model layer \(index) "
                            + "(\(pagedCaches.count) storage slots)")
                }
                return pagedCaches[storage]
            }
        }

        var schedulerConfig = build.schedulerConfig
        schedulerConfig.enablePrefixCache = build.prefixCache != nil

        // A BLOCK drafter reads the TARGET's hidden state at named layers, so
        // the target has to keep those layers on every forward. Arm that tap
        // HERE, where both directions are visible: a build that will not
        // speculate disarms it, so a serial leg in a process that also served
        // a block leg pays nothing for a drafter that is merely resident.
        if let block = mtpDrafter as? any CBv2MTPBlockDrafter {
            try block.setBlockContextArmed(build.decoder != .serial)
        }

        return EngineV2(
            model: CBv2SteppableLanguageModelAdapter(model),
            layerKinds: layerKinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches),
            sampler: CBv2DefaultSampler(),
            detokenizerFactory: CBv2TextDetokenizerFactory(tokenizer: tokenizer),
            schedulerConfig: schedulerConfig,
            loopConfig: build.loopConfig,
            prefixCache: build.prefixCache,
            mtpDrafter: build.decoder == .serial ? nil : mtpDrafter,
            mtpConfig: build.decoder == .serial
                ? CBv2MTPConfig(enabled: false) : build.mtpConfig)
    }
}
