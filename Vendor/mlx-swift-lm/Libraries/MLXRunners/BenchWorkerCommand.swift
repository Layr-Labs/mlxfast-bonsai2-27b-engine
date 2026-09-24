// Copyright © 2026 Eigen Labs.
//
// MLXRunners — the top of bench-worker's argv: which subcommand, and the two
// that never touch a checkpoint.
//
// `manifest` exists because the conformance kit takes `--manifest <path>`
// and, until now, nothing could produce that file. A manifest is static data
// on a runner TYPE, so printing it needs no weights, no GPU and no
// checkpoint beyond the `config.json` that says which family a directory
// holds. The bytes printed are the section 6.0 CANONICAL serialization —
// exactly the bytes the digest is computed over — so a kit that hashes what
// it reads gets the same digest the hello reports, and a manifest change is
// a diff of readable bytes rather than an opaque hash move.
//
// `--help` is here for the same reason the launch parser is: it is argv
// only, and it must answer without reading anything.

import Foundation

/// What this invocation is, once the leading token has been read.
public enum BenchWorkerCommand: Sendable {
    /// `runtime-worker` or `resident`: everything that serves.
    case serve(BenchWorkerLaunchOptions)
    /// `manifest`: print a runner's declaration and exit.
    case manifest(BenchWorkerManifestOptions)
    /// `--help`, or `<subcommand> --help`.
    case help(BenchWorkerUsage.Topic)

    /// Parse argv (minus the executable name).
    ///
    /// Reads NOTHING under `--weights`. The serve subcommands hand off to
    /// `BenchWorkerLaunchOptions.parse`, which keeps that guarantee; the
    /// other two never look at a checkpoint at all.
    public static func parse(
        _ arguments: [String],
        fileManager: FileManager = .default
    ) throws -> BenchWorkerCommand {
        // `--help` anywhere wins: someone asking how to run this must not
        // first have to get the rest of the line right.
        let topic = BenchWorkerUsage.Topic(subcommand: arguments.first)
        if arguments.contains("--help") || arguments.contains("-h") {
            return .help(topic)
        }
        guard let first = arguments.first else {
            // No subcommand at all is the per-phase worker benchd spawns;
            // the serve parser refuses it for the missing `--weights`.
            return .serve(try BenchWorkerLaunchOptions.parse(arguments, fileManager: fileManager))
        }
        if first == "manifest" {
            return .manifest(try BenchWorkerManifestOptions.parse(Array(arguments.dropFirst())))
        }
        // A leading token that is neither a flag nor a known subcommand is a
        // MISTYPED subcommand, and saying so beats "unknown argument": the
        // caller gets the usage screen rather than a note about one word.
        if !first.hasPrefix("-"),
            BenchWorkerSubcommand(rawValue: first) == nil
        {
            throw WorkerLaunchError.unknownSubcommand(first)
        }
        return .serve(try BenchWorkerLaunchOptions.parse(arguments, fileManager: fileManager))
    }
}

// MARK: - manifest

/// `bench-worker manifest (--runner <id> | --weights <dir>) [--digest]`.
public struct BenchWorkerManifestOptions: Sendable {
    /// Name the runner directly.
    public var runnerID: String?
    /// Or name a checkpoint, and let its `config.json` `model_type` pick the
    /// runner through the registry. ONLY `config.json` is read.
    public var weights: URL?
    /// Print the sha256 hex instead of the bytes.
    public var digestOnly: Bool

    public init(runnerID: String? = nil, weights: URL? = nil, digestOnly: Bool = false) {
        self.runnerID = runnerID
        self.weights = weights
        self.digestOnly = digestOnly
    }

    public static func parse(_ arguments: [String]) throws -> BenchWorkerManifestOptions {
        var options = BenchWorkerManifestOptions()
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            index += 1
            func next(_ name: String) throws -> String {
                guard index < arguments.count else {
                    throw WorkerLaunchError.missingValue(name)
                }
                defer { index += 1 }
                return arguments[index]
            }
            switch argument {
            case "--runner": options.runnerID = try next("--runner")
            case "--weights": options.weights = URL(fileURLWithPath: try next("--weights"))
            case "--digest": options.digestOnly = true
            default: throw WorkerLaunchError.unknownArgument(argument)
            }
        }

        // Exactly one target. Both would make the output depend on which one
        // this code happened to prefer, and neither has nothing to print.
        switch (options.runnerID, options.weights) {
        case (nil, nil): throw WorkerLaunchError.manifestTargetMissing
        case (.some, .some): throw WorkerLaunchError.manifestTargetAmbiguous
        default: break
        }
        if let runnerID = options.runnerID,
            runnerID.trimmingCharacters(in: .whitespaces).isEmpty
        {
            throw WorkerLaunchError.emptyRunnerID
        }
        return options
    }

    /// The runner this invocation names.
    ///
    /// With `--weights`, the ONLY thing read is `config.json`: a manifest is
    /// static data on the type, so printing one must never cost a checkpoint
    /// load.
    public func resolveRunner(
        registry: RunnerRegistry = .shared
    ) throws -> any Runner.Type {
        if let runnerID {
            guard
                let match = registry.manifests().first(where: { $0.runnerID == runnerID })
            else {
                throw WorkerLaunchError.unknownRunner(runnerID)
            }
            guard let modelType = match.modelTypes.first else {
                throw WorkerLaunchError.unknownRunner(runnerID)
            }
            return try registry.resolve(modelType: modelType)
        }
        guard let weights else { throw WorkerLaunchError.manifestTargetMissing }
        return try registry.resolve(checkpoint: weights)
    }

    /// What to print, newline included.
    public func render(registry: RunnerRegistry = .shared) throws -> String {
        let manifest = try resolveRunner(registry: registry).manifest
        if digestOnly { return manifest.sha256Digest() + "\n" }
        // The CANONICAL bytes, not a re-encoding: a kit that hashes what it
        // reads must get the digest the hello reports.
        return String(decoding: manifest.canonicalJSON(), as: UTF8.self) + "\n"
    }
}

// MARK: - Usage

/// One screen per topic. Deliberately plain text built here rather than a
/// dependency: the worker's argv surface is small enough to state, and an
/// operator on a box reads this before they read a document.
public enum BenchWorkerUsage {

    public enum Topic: Sendable, Equatable {
        case general
        case runtimeWorker
        case resident
        case manifest

        public init(subcommand: String?) {
            switch subcommand {
            case "runtime-worker": self = .runtimeWorker
            case "resident": self = .resident
            case "manifest": self = .manifest
            default: self = .general
            }
        }
    }

    public static func text(for topic: Topic) -> String {
        switch topic {
        case .general: return general
        case .runtimeWorker: return runtimeWorker
        case .resident: return resident
        case .manifest: return manifest
        }
    }

    public static let general = """
        bench-worker — Engine Protocol v1 over the MLXRunners runner boundary.

        USAGE
          bench-worker runtime-worker --weights <dir> [options]
          bench-worker resident       --weights <dir> --socket <path> [options]
          bench-worker manifest       (--runner <id> | --weights <dir>) [--digest]
          bench-worker diag-parity    --weights <dir> --golden <json> [--steps <n>] [options]
          bench-worker <subcommand> --help

        SUBCOMMANDS
          runtime-worker   Serve ONE benchd phase over stdin/stdout. Attaches to a
                           resident when BENCH_WORKER_RESIDENT_SOCKET names one, and
                           then loads nothing; otherwise loads and serves in process.
          resident         Serve ONE measurement window: load once, listen on a Unix
                           socket, serve one connection at a time.
          manifest         Print a runner's canonical manifest, or its sha256. Loads
                           no weights.
          diag-parity      DIAGNOSTIC. Load once, then teacher-force a golden through
                           the fork's legacy forward and the CBv2 stepper side by side
                           and print the first step where they part. Serves nothing.
        """

    public static let runtimeWorker = """
        bench-worker runtime-worker — serve one benchd phase.

        USAGE
          bench-worker runtime-worker --weights <dir> [options]

        OPTIONS
          --weights <dir>              Checkpoint directory. Required.
          --runner <id>                Use this runner instead of resolving the
                                       checkpoint's config.json model_type.
          --drafter <dir>              Speculative drafter artifact.
          --trusted                    Advertise cohort_reference_replay.
          --speculative-protocol v1.1  Enable the speculative surface: spec_modes,
                                       the free-run capabilities and effective_spec.
                                       Absent, the worker speaks plain v1 and refuses
                                       a spec by name.
          --resource <name>=<path>     Repeatable. An out-of-checkpoint input the
                                       runner needs; passed through opaque.
          --kv-bytes <n>               KV byte grant.
          --verbose                    Write a load summary to stderr BEFORE the
                                       hello: what the load decided, one key=value
                                       per line. stdout is untouched.

        ENVIRONMENT
          BENCH_WORKER_RESIDENT_SOCKET  Attach to the resident at this socket and
                                        load nothing. The hello then carries a
                                        `resident` block; `backend` is unchanged.
                                        Unreachable, refusing or wrongly-loaded is
                                        a REFUSAL, never a fall-through to loading
                                        in process.
          BENCH_WORKER_RESIDENT_HELLO   Set to 1 to emit the additive `resident`
                                        hello object. Off by default: benchd's
                                        envelope must admit the field first.
          BENCH_WORKER_VERBOSE          Set to 1 for --verbose.
        """

    public static let resident = """
        bench-worker resident — one weight load for one measurement window.

        USAGE
          bench-worker resident --weights <dir> --socket <path> [options]

        Loads ONCE, then listens. Each phase's runtime-worker attaches over the
        socket and loads nothing. Exactly one connection is served at a time; a
        second is refused with one line and closed. The allocator is drained when a
        connection is accepted, so a phase never inherits the one before it.

        OPTIONS
          --socket <path>              Unix socket to listen on. Required.
          --weights <dir>              Checkpoint directory. Required.
          --runner, --drafter, --trusted, --speculative-protocol, --resource,
          --kv-bytes, --verbose        As for runtime-worker; this process performs
                                       the load the phases then share.
          --pidfile <path>             Written when the resident starts listening
                                       and removed at teardown.
        """

    public static let manifest = """
        bench-worker manifest — print a runner's static declaration.

        USAGE
          bench-worker manifest --runner <id> [--digest]
          bench-worker manifest --weights <dir> [--digest]

        Prints the CANONICAL manifest bytes — the section 6.0 serialization the
        sha256 is computed over — followed by a newline. This is the file benchd's
        conformance kit reads as --manifest.

        Loads NO weights. With --weights, the only file read is config.json, whose
        model_type selects the runner through the registry.

        OPTIONS
          --runner <id>    Runner id, e.g. layr/qwen4exp-125b-a6b.
          --weights <dir>  Checkpoint directory; its model_type picks the runner.
          --digest         Print only the lowercase sha256 hex.
        """
}
