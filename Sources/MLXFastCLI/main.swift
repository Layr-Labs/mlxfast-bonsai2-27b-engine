import Darwin
import Foundation
import MLXFastCore
import MLXFastHarness
import MLXFastTransform
import Tokenizers

let exitCode = MLXFastCLI.run(arguments: Array(CommandLine.arguments.dropFirst()))
exit(Int32(exitCode))

private enum MLXFastCLI {
    static func run(arguments: [String]) -> Int {
        guard let command = arguments.first, command != "help", command != "--help", command != "-h" else {
            printUsage()
            return 0
        }

        let options = ParsedOptions(Array(arguments.dropFirst()))

        do {
            switch command {
            case "transform":
                try runTransform(options)
                return 0
            case "verify-transform":
                try runVerifyTransform(options)
                return 0
            case "attach-benchmark-oracle":
                try runAttachBenchmarkOracle(options)
                return 0
            case "checkpoint-shards":
                try runCheckpointShards(options)
                return 0
            case "mtp-verify":
                // MTP ARM DEFERRED (2026-08-22, Gemma 4 26B A4B harness port).
                // The verb is RETAINED and REFUSES rather than being deleted:
                // the arm is deferred, not abandoned, so keeping the name
                // reserved means the follow-up increment restores behaviour
                // behind an entry point that already exists, and any caller
                // still invoking it is told what happened instead of
                // "unknown command".
                //
                // What went: the whole MTP surface was written against a
                // model the harness no longer carries, and the trusted CLI
                // loads no model at all. Renaming the types would have left
                // code that compiles and cannot run, which is worse than a
                // refusal. This track's speculative head is embedded in the
                // pinned checkpoint and is driven by the engine, not by a
                // trusted CLI verb.
                throw MLXFastError.invalidInput(
                    "mtp-verify is not runnable on this engine: the MTP arm "
                        + "is driven by the engine, not by this binary."
                )
            default:
                fputs("mlxfast-swift: unknown command '\(command)'\n\n", stderr)
                printUsage()
                return 2
            }
        } catch {
            fputs("mlxfast-swift: \(error)\n", stderr)
            return 1
        }
    }

    private static func runTransform(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--reference", "--output"])
        let referencePath = options.value(
            for: "--reference",
            default: environmentValue(
                "MLXFAST_REFERENCE_DIR",
                fallback: MLXFastConstants.defaultReferencePath
            )
        )
        let outputPath = options.value(
            for: "--output",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        // The transform module is an editable path. Confine it before it runs:
        // it may write only its output tree, the hidden `.<output name>.*`
        // staging siblings it builds that tree in, and its own temporary
        // directory.
        let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
        try reexecUnderParentToolSandboxIfRequested(
            subcommand: "transform",
            writableSubpaths: [outputPath],
            writablePrefixes: [
                outputURL.deletingLastPathComponent().path + "/.\(outputURL.lastPathComponent)."
            ]
        )
        let report = try SwiftTransform.run(
            TransformOptions(referencePath: referencePath, outputPath: outputPath)
        )
        print("reference: \(report.referencePath)")
        print("output: \(report.outputPath)")
        print("dense tensors: \(report.denseTensorCount) across \(report.denseShardCount) shard(s)")
        print("config: \(report.configPath)")
        print("index: \(report.indexPath)")
    }

    private static func runVerifyTransform(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--reference", "--weights", "--tmp-parent", "--max-bytes"])
        let referencePath = options.value(
            for: "--reference",
            default: environmentValue(
                "MLXFAST_REFERENCE_DIR",
                fallback: MLXFastConstants.defaultReferencePath
            )
        )
        let weightsPath = options.value(
            for: "--weights",
            default: environmentValue(
                "MLXFAST_WEIGHTS_PATH",
                fallback: MLXFastConstants.defaultWeightsPath
            )
        )
        let temporaryParentPath = options.value(for: "--tmp-parent", default: "")
        let maxBytesRaw = options.value(
            for: "--max-bytes",
            default: environmentValue(
                "MLXFAST_MAX_WEIGHTS_BYTES",
                fallback: "\(MLXFastConstants.defaultMaxTransformedWeightsBytes)"
            )
        )
        let maxByteCount = try parseTransformedWeightsByteLimit(
            raw: maxBytesRaw,
            defaultByteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes,
            optionLabel: "--max-bytes"
        )
        // verify-transform runs the editable transform again (SwiftTransform.run
        // inside TransformVerifier), so it gets the same confinement. It may
        // write only the scratch trees TransformVerifier creates: under an
        // explicit --tmp-parent, or as `.mlxfast-transform-verify-*` (and the
        // transform's `..mlxfast-transform-verify-*` staging siblings) beside
        // the weights directory.
        if temporaryParentPath.isEmpty {
            let weightsParent = URL(fileURLWithPath: weightsPath).standardizedFileURL
                .deletingLastPathComponent().path
            let verifyParent = weightsParent.isEmpty
                ? FileManager.default.currentDirectoryPath
                : weightsParent
            try reexecUnderParentToolSandboxIfRequested(
                subcommand: "verify-transform",
                writableSubpaths: [],
                writablePrefixes: [
                    verifyParent + "/.mlxfast-transform-verify-",
                    verifyParent + "/..mlxfast-transform-verify-",
                ]
            )
        } else {
            try reexecUnderParentToolSandboxIfRequested(
                subcommand: "verify-transform",
                writableSubpaths: [temporaryParentPath]
            )
        }
        let report = try TransformVerifier.verify(
            TransformVerificationOptions(
                referencePath: referencePath,
                weightsPath: weightsPath,
                temporaryParentPath: temporaryParentPath.isEmpty ? nil : temporaryParentPath,
                maxByteCount: maxByteCount
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        print("")
    }

    private static func runAttachBenchmarkOracle(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--golden", "--output"])
        let goldenPath = options.value(
            for: "--golden",
            default: environmentValue(
                "MLXFAST_CORRECTNESS_GOLDEN_PATH",
                fallback: MLXFastConstants.defaultGoldenPath
            )
        )
        let outputPath = options.value(for: "--output", default: goldenPath)

        try requireFile(goldenPath, description: "correctness golden file")
        // Strict-validate the INPUT before any write. --output defaults to the
        // input path, so a malformed input must fail here -- never after the
        // original has been replaced on disk. Through the Qwen loader: the
        // oracle it derives is what a ranked Qwen run is scored against.
        _ = try loadQwenGoldenFixture(from: goldenPath)
        let goldenData = try Data(contentsOf: URL(fileURLWithPath: goldenPath))
        let golden = try JSONDecoder().decode(GoldenDocument.self, from: goldenData)

        let merged = try goldenDocumentAttachingDerivedBenchmarkOracle(golden)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try writeValidatedGoldenDocument(encoder.encode(merged), to: outputPath)
        guard let oracle = merged.benchmark else {
            throw MLXFastError.invalidInput("attach-benchmark-oracle produced no benchmark oracle")
        }
        print(
            "attached benchmark oracle prefill_tokens=\(oracle.prefillPromptTokens.count) "
                + "decode_seed_tokens=\(oracle.decodeSeedTokens.count) "
                + "expected_decode_tokens=\(oracle.expectedDecodeTokens.count) "
                + "baselines=none "
                + "output=\(outputPath)"
        )
    }

    // Writes a merged golden by staging to a temp sibling and proving the
    // result loads through the strict fixture loader BEFORE it can touch the
    // destination. The attach commands default --output to the input golden,
    // so an in-place write followed by a failed validation would destroy the
    // original (typically the private golden) with nothing to roll back to.
    private static func writeValidatedGoldenDocument(_ outputData: Data, to outputPath: String) throws {
        let outputURL = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let temporaryURL = outputURL.deletingLastPathComponent()
            .appendingPathComponent(".\(outputURL.lastPathComponent).attach-\(UUID().uuidString).tmp")
        defer {
            try? FileManager.default.removeItem(at: temporaryURL)
        }
        try outputData.write(to: temporaryURL, options: [.atomic])
        // Qwen loader on the staged bytes: every caller of this helper is an
        // attach verb writing a Qwen golden, so an attach can never land a
        // document that has lost (or never carried) the model identity.
        _ = try loadQwenGoldenFixture(from: temporaryURL.path)
        if FileManager.default.fileExists(atPath: outputURL.path) {
            _ = try FileManager.default.replaceItemAt(outputURL, withItemAt: temporaryURL)
        } else {
            try FileManager.default.moveItem(at: temporaryURL, to: outputURL)
        }
    }

    private static func requirePrivateOutputPath(_ path: String, description: String) throws {
        let privateDir = environmentValue("MLXFAST_PRIVATE_DIR", fallback: "")
        guard !privateDir.isEmpty else {
            return
        }
        let outputPath = absolutePath(path)
        let privatePath = absolutePath(privateDir)
        guard outputPath.hasPrefix(privatePath + "/") else {
            throw MLXFastError.invalidInput("\(description) must be under MLXFAST_PRIVATE_DIR")
        }
    }

    private static func parsePositiveInt(_ rawValue: String, optionName: String) throws -> Int {
        guard let value = Int(rawValue), value > 0 else {
            throw MLXFastError.invalidInput("\(optionName) must be a positive integer")
        }
        return value
    }

    private static func parseNonNegativeInt(_ rawValue: String, optionName: String) throws -> Int {
        guard let value = Int(rawValue), value >= 0 else {
            throw MLXFastError.invalidInput("\(optionName) must be a non-negative integer")
        }
        return value
    }

    private static func positiveInteger(
        _ text: String,
        name: String
    ) throws -> Int {
        guard let value = Int(text), value > 0 else {
            throw MLXFastError.invalidInput("\(name) requires a positive integer")
        }
        return value
    }

    /// A non-negative count that is absent when the flag was not passed.
    private static func optionalCount(
        _ text: String,
        name: String
    ) throws -> Int? {
        guard !text.isEmpty else { return nil }
        guard let value = Int(text), value >= 0 else {
            throw MLXFastError.invalidInput(
                "\(name) requires a non-negative integer"
            )
        }
        return value
    }

    private static func currentExecutablePath() throws -> String {
        if let executableURL = Bundle.main.executableURL {
            let path = executableURL.standardizedFileURL
                .resolvingSymlinksInPath().path
            if FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }

        var requiredSize: UInt32 = 0
        _ = _NSGetExecutablePath(nil, &requiredSize)
        if requiredSize > 0 {
            var buffer = [CChar](
                repeating: 0,
                count: Int(requiredSize)
            )
            if _NSGetExecutablePath(&buffer, &requiredSize) == 0 {
                let executableBytes = buffer
                    .prefix { $0 != 0 }
                    .map { UInt8(bitPattern: $0) }
                let path = URL(
                    fileURLWithPath: String(
                        decoding: executableBytes,
                        as: UTF8.self
                    )
                ).standardizedFileURL.resolvingSymlinksInPath().path
                if FileManager.default.isExecutableFile(atPath: path) {
                    return path
                }
            }
        }

        if let rawExecutable = CommandLine.arguments.first,
           !rawExecutable.isEmpty
        {
            if rawExecutable.contains("/") {
                let path = absolutePath(rawExecutable)
                if FileManager.default.isExecutableFile(atPath: path) {
                    return path
                }
            } else {
                let searchPath = ProcessInfo.processInfo.environment[
                    "PATH"
                ] ?? ""
                for directory in searchPath.split(
                    separator: ":",
                    omittingEmptySubsequences: false
                ) {
                    let root = directory.isEmpty
                        ? FileManager.default.currentDirectoryPath
                        : String(directory)
                    let path = URL(fileURLWithPath: root)
                        .appendingPathComponent(rawExecutable).path
                    if FileManager.default.isExecutableFile(atPath: path) {
                        return URL(fileURLWithPath: path)
                            .standardizedFileURL
                            .resolvingSymlinksInPath().path
                    }
                }
            }
        }

        throw MLXFastError.invalidInput(
            "mlxfast-swift could not resolve its actual executable path "
                + "from Bundle.main, _NSGetExecutablePath, argv[0], or PATH"
        )
    }

    // Confine the `transform` and `verify-transform` command paths behind a
    // Seatbelt profile before they run any code from the editable transform
    // module. These subcommands run the submission-built transform in THIS
    // process (they do not spawn the separately sandboxed runtime worker), so
    // on the ranked box they would otherwise run participant code with every
    // right of the runner account. This re-executes the current process under
    // `/usr/bin/sandbox-exec` with a profile that:
    //
    //   * denies network, process-fork, process-exec (of anything but this
    //     binary) and the DNS resolver mach-lookup;
    //   * denies every file write, then allows writes only to the subcommand's
    //     own output tree and the hidden staging siblings the transform builds
    //     it in, to a private temporary directory made for this run (TMPDIR
    //     points at it), and to the harmless device nodes;
    //   * denies every read and write of the evaluator-only material the
    //     runner service environment names (hidden goldens, the reference
    //     workspace of the paired control leg, the box calibration file, the
    //     benchd binary directory, the private directory, the build cache
    //     root and the runner registration files).
    //
    // Reads outside that list stay allowed: the transform reads the reference
    // checkpoint, and dyld and Foundation read system paths.
    //
    // Trigger + fail-closed policy: the ranked workflow sets
    // MLXFAST_SANDBOX_PARENT_TOOLS=1 and MLXFAST_OFFICIAL_BENCHMARK_RUN=1 on the
    // transform step; either one arms the sandbox. When armed, a missing
    // sandbox-exec or MLXFAST_NO_SANDBOX=1 aborts the run. An official run
    // also refuses when the runner environment does not name the evaluator
    // paths, because the deny list would then be empty. The re-executed child
    // proves that the profile is in force before it continues (a write probe
    // outside the allowed paths must fail). Local invocations set neither
    // flag, so participant workflows are unchanged. MLXFAST_PARENT_SANDBOX_ACTIVE=1
    // is set on the re-exec so the sandboxed child does not recurse.
    private static func reexecUnderParentToolSandboxIfRequested(
        subcommand: String,
        writableSubpaths: [String],
        writablePrefixes: [String] = []
    ) throws {
        if environmentValue("MLXFAST_PARENT_SANDBOX_ACTIVE", fallback: "0") == "1" {
            try requireParentToolSandboxInForce(subcommand: subcommand)
            return
        }
        let officialRun = environmentValue("MLXFAST_OFFICIAL_BENCHMARK_RUN", fallback: "0") == "1"
        let requested = officialRun
            || environmentValue("MLXFAST_SANDBOX_PARENT_TOOLS", fallback: "0") == "1"
        guard requested else {
            return
        }
        if environmentValue("MLXFAST_NO_SANDBOX", fallback: "0") == "1" {
            throw MLXFastError.invalidInput(
                "\(subcommand) in a benchmark context requires the parent-tool sandbox; unset MLXFAST_NO_SANDBOX"
            )
        }
        let sandboxExecutable = "/usr/bin/sandbox-exec"
        guard FileManager.default.isExecutableFile(atPath: sandboxExecutable) else {
            throw MLXFastError.invalidInput(
                "\(subcommand) in a benchmark context requires sandbox-exec for the parent-tool sandbox"
            )
        }
        if officialRun {
            let missing = requiredEvaluatorPathVariables.filter {
                environmentValue($0, fallback: "").isEmpty
            }
            guard missing.isEmpty else {
                throw MLXFastError.invalidInput(
                    "\(subcommand) in an official run requires the runner environment to name "
                        + "the evaluator-only paths the sandbox denies; unset: "
                        + missing.joined(separator: ", ")
                )
            }
        }
        let executablePath = try currentExecutablePath()
        guard FileManager.default.isExecutableFile(atPath: executablePath) else {
            throw MLXFastError.invalidInput(
                "\(subcommand) parent-tool sandbox resolved a non-executable self path: \(executablePath)"
            )
        }
        let privateTemporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxfast-parent-tool-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: privateTemporaryDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        let privateTemporaryPath = sandboxPath(privateTemporaryDirectory.path)
        let profilePath = try writeParentToolSandboxProfile(
            allowedExecutablePath: executablePath,
            writableSubpaths: [privateTemporaryPath] + writableSubpaths.map(sandboxPath),
            writablePrefixes: writablePrefixes.map(sandboxPrefix),
            deniedSubpaths: evaluatorOnlyPaths(),
            profileDirectory: privateTemporaryPath
        )
        let argv = [sandboxExecutable, "-f", profilePath, executablePath]
            + Array(CommandLine.arguments.dropFirst())
        setenv("MLXFAST_PARENT_SANDBOX_ACTIVE", "1", 1)
        setenv("TMPDIR", privateTemporaryPath + "/", 1)
        var cArgs: [UnsafeMutablePointer<CChar>?] = argv.map { strdup($0) }
        cArgs.append(nil)
        defer {
            for pointer in cArgs {
                if let pointer {
                    free(pointer)
                }
            }
        }
        _ = sandboxExecutable.withCString { pathPointer in
            execv(pathPointer, cArgs)
        }
        // execv only returns on failure.
        throw MLXFastError.invalidInput(
            "\(subcommand) failed to re-exec under sandbox-exec (errno=\(errno))"
        )
    }

    // The runner service exports these on the ranked box, and the workflow's
    // first step refuses a box that does not. An official run needs all four
    // before it arms the sandbox.
    private static let requiredEvaluatorPathVariables = [
        "MLXFAST_QWEN38_GOLDEN_DIR",
        "MLXFAST_BASELINE_WORKSPACE",
        "MLXFAST_BASELINE_CALIBRATION",
        "BENCHD_BIN_DIR",
    ]

    // Every evaluator-only path the environment names. A variable that is not
    // set adds nothing. The build cache root has the same default that
    // tools/build-cache.sh uses. The runner registration files sit two levels
    // above RUNNER_WORKSPACE (<runner>/_work/<repository>) in a standard
    // actions-runner layout; a deny of a path that does not exist is harmless.
    private static func evaluatorOnlyPaths() -> [String] {
        var paths: [String] = []
        for name in requiredEvaluatorPathVariables + [
            "MLXFAST_CORRECTNESS_GOLDEN_PATH",
            "MLXFAST_PRIVATE_DIR",
        ] {
            let value = environmentValue(name, fallback: "")
            if !value.isEmpty {
                paths.append(value)
            }
        }
        let home = environmentValue("HOME", fallback: NSHomeDirectory())
        paths.append(
            environmentValue(
                "MLXFAST_BUILD_CACHE_DIR",
                fallback: home + "/.cache/mlxfast-engine-build"
            )
        )
        let runnerWorkspace = environmentValue("RUNNER_WORKSPACE", fallback: "")
        if !runnerWorkspace.isEmpty {
            let runnerRoot = URL(fileURLWithPath: runnerWorkspace).standardizedFileURL
                .deletingLastPathComponent()
                .deletingLastPathComponent()
            for name in [".credentials", ".credentials_rsaparams", ".runner"] {
                paths.append(runnerRoot.appendingPathComponent(name).path)
            }
        }
        return paths.map(sandboxPath)
    }

    private static func writeParentToolSandboxProfile(
        allowedExecutablePath: String,
        writableSubpaths: [String],
        writablePrefixes: [String],
        deniedSubpaths: [String],
        profileDirectory: String
    ) throws -> String {
        let profileURL = URL(fileURLWithPath: profileDirectory)
            .appendingPathComponent("parent-tool.sb")
        let absoluteExecutablePath = absolutePath(allowedExecutablePath)
        var lines = [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            "(deny process-fork)",
            "(deny process-exec*)",
            "(allow process-exec (literal \"\(seatbeltEscaped(absoluteExecutablePath))\"))",
            "(deny mach-lookup (global-name \"com.apple.mDNSResponder\"))",
            "(deny mach-lookup (global-name \"com.apple.system.mDNSResponder\"))",
            "(deny mach-lookup (global-name-prefix \"com.apple.mDNSResponder\"))",
            "(deny file-write*)",
            "(allow file-write* (literal \"/dev/null\") (literal \"/dev/zero\") (literal \"/dev/dtracehelper\"))",
        ]
        for path in writableSubpaths {
            lines.append("(allow file-write* (subpath \"\(seatbeltEscaped(path))\"))")
        }
        for prefix in writablePrefixes {
            lines.append("(allow file-write* (regex #\"^\(seatbeltRegexEscaped(prefix))\"))")
        }
        // Last, so that these rules win over every allow above.
        for path in deniedSubpaths {
            lines.append("(deny file-read* file-write* (subpath \"\(seatbeltEscaped(path))\"))")
        }
        try (lines.joined(separator: "\n") + "\n")
            .write(to: profileURL, atomically: true, encoding: .utf8)
        return profileURL.path
    }

    // The re-executed child checks that the profile is in force before any
    // editable code runs. Creating a file beside this binary is outside every
    // allowed write path, so the sandbox must refuse it. When the create
    // succeeds, no sandbox is in force: the probe is removed and the run stops.
    private static func requireParentToolSandboxInForce(subcommand: String) throws {
        let executableDirectory = URL(fileURLWithPath: try currentExecutablePath())
            .deletingLastPathComponent()
        let probe = executableDirectory
            .appendingPathComponent(".mlxfast-parent-sandbox-probe-\(UUID().uuidString)").path
        let descriptor = open(probe, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        guard descriptor >= 0 else {
            return
        }
        close(descriptor)
        unlink(probe)
        throw MLXFastError.invalidInput(
            "\(subcommand) expected the parent-tool sandbox to be in force, but a write outside "
                + "the allowed paths succeeded; refusing to run the transform unconfined"
        )
    }

    // Seatbelt matches the resolved path. Resolve the deepest existing ancestor
    // (so /tmp becomes /private/tmp and /var becomes /private/var) and keep the
    // rest, because an output tree may not exist yet.
    private static func sandboxPath(_ path: String) -> String {
        var existing = URL(fileURLWithPath: absolutePath(path))
        var rest: [String] = []
        while !FileManager.default.fileExists(atPath: existing.path), existing.path != "/" {
            rest.insert(existing.lastPathComponent, at: 0)
            existing.deleteLastPathComponent()
        }
        guard let resolved = realpath(existing.path, nil) else {
            return existing.path
        }
        defer { free(resolved) }
        var url = URL(fileURLWithPath: String(cString: resolved))
        for component in rest {
            url.appendPathComponent(component)
        }
        return url.path
    }

    // A prefix ends in a partial file name, so only its directory is resolved.
    private static func sandboxPrefix(_ prefix: String) -> String {
        let url = URL(fileURLWithPath: prefix)
        return sandboxPath(url.deletingLastPathComponent().path) + "/" + url.lastPathComponent
    }

    private static func seatbeltRegexEscaped(_ value: String) -> String {
        var escaped = ""
        for character in value {
            if "\\^$.|?*+()[]{}\"".contains(character) {
                escaped.append("\\")
            }
            escaped.append(character)
        }
        return escaped
    }

    private static func absolutePath(_ path: String) -> String {
        let url: URL
        if path.hasPrefix("/") {
            url = URL(fileURLWithPath: path)
        } else {
            url = URL(
                fileURLWithPath: path,
                relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            )
        }
        return url.standardizedFileURL.resolvingSymlinksInPath().path
    }

    private static func seatbeltEscaped(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }

    private static func runCheckpointShards(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--index"])
        let indexPath = options.value(for: "--index", default: "")
        guard !indexPath.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint-shards requires --index PATH")
        }
        // Trusted code only: setup.sh runs this verb before any sandbox is in
        // place, so it must not call into the editable transform module.
        for shard in try CheckpointShardList.safetensorShardNames(fromIndexAt: indexPath) {
            print(shard)
        }
    }

    private static func printUsage() {
        print(
            """
            Usage:
              mlxfast-swift transform [--reference PATH] [--output PATH]
              mlxfast-swift verify-transform [--reference PATH] [--weights PATH] [--tmp-parent PATH] [--max-bytes N]
              mlxfast-swift attach-benchmark-oracle [--golden PATH] [--output PATH]
              mlxfast-swift checkpoint-shards --index PATH
              mlxfast-swift mtp-verify  (deferred: refuses; returns with the follow-up increment)

            Swift-only Ternary Bonsai 2 27B 2-bit harness entrypoint.
            """
        )
    }

    private static func environmentValue(_ name: String, fallback: String) -> String {
        let value = ProcessInfo.processInfo.environment[name] ?? ""
        return value.isEmpty ? fallback : value
    }

    private static func trimmedNonEmpty(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

}





private struct ParsedOptions {
    private var values: [String: String] = [:]
    private var flags: Set<String> = []
    private var positionals: [String] = []
    private var duplicates: Set<String> = []

    init(_ arguments: [String]) {
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            if argument.hasPrefix("--") {
                if let separator = argument.firstIndex(of: "=") {
                    let key = String(argument[..<separator])
                    let value = String(argument[argument.index(after: separator)...])
                    recordOption(key)
                    values[key] = value
                    index += 1
                } else if index + 1 < arguments.count && !arguments[index + 1].hasPrefix("--") {
                    recordOption(argument)
                    values[argument] = arguments[index + 1]
                    index += 2
                } else {
                    recordOption(argument)
                    flags.insert(argument)
                    index += 1
                }
            } else {
                positionals.append(argument)
                index += 1
            }
        }
    }

    private mutating func recordOption(_ name: String) {
        if values[name] != nil || flags.contains(name) {
            duplicates.insert(name)
        }
    }

    func value(for name: String, default defaultValue: String) -> String {
        values[name] ?? defaultValue
    }

    func hasFlag(_ name: String) -> Bool {
        flags.contains(name)
    }

    func validate(
        valueOptions: Set<String>,
        flagOptions: Set<String> = [],
        allowPositionals: Bool = false
    ) throws {
        if let duplicate = duplicates.first {
            throw MLXFastError.invalidInput("duplicate option \(duplicate)")
        }
        for name in values.keys where !valueOptions.contains(name) {
            throw MLXFastError.invalidInput("unknown option \(name)")
        }
        for (name, value) in values where value.isEmpty {
            throw MLXFastError.invalidInput("\(name) requires a non-empty value")
        }
        for flag in flags {
            if valueOptions.contains(flag) {
                throw MLXFastError.invalidInput("\(flag) requires a value")
            }
            if !flagOptions.contains(flag) {
                throw MLXFastError.invalidInput("unknown option \(flag)")
            }
        }
        if !allowPositionals, let positional = positionals.first {
            throw MLXFastError.invalidInput("unexpected argument \(positional)")
        }
    }
}
