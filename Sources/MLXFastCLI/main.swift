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
        // The transform module is an editable path. On an official run it
        // must already be confined by tools/sandboxed-cli.sh, which allows
        // writes only to the output tree, the hidden `.<output name>.*`
        // staging siblings the transform builds it in, and a private TMPDIR.
        try requireConfinementOnOfficialRun(subcommand: "transform")
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
        // inside TransformVerifier), so on an official run it must already be
        // confined the same way. tools/sandboxed-cli.sh allows writes only to
        // the scratch trees TransformVerifier creates: under an explicit
        // --tmp-parent, or as `.mlxfast-transform-verify-*` (and the
        // transform's `..mlxfast-transform-verify-*` staging siblings) beside
        // the weights directory.
        try requireConfinementOnOfficialRun(subcommand: "verify-transform")
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

    // The `transform` and `verify-transform` verbs run code from the editable
    // transform module in THIS process. This binary also links that module,
    // so it cannot confine itself: code in a linked module can run before
    // main(). The confinement therefore starts outside this binary. On the
    // ranked box, tools/sandboxed-cli.sh writes the Seatbelt profile
    // (tools/seatbelt-profile.py) and starts this binary under
    // /usr/bin/sandbox-exec.
    //
    // This check is the second line. On an official run
    // (MLXFAST_OFFICIAL_BENCHMARK_RUN=1, or RUNNER_ENVIRONMENT=self-hosted)
    // the verb refuses unless the process is already confined:
    //
    //   * MLXFAST_NO_SANDBOX=1 is refused;
    //   * the runner environment must name the evaluator-only paths that the
    //     profile denies;
    //   * positive control: a new file in TMPDIR (the private temporary
    //     directory the wrapper makes) must be created, so the uid can write
    //     and the probe below means something;
    //   * denied write: a new file beside this binary, which is outside every
    //     path the profile allows, must fail with EPERM. EPERM is the Seatbelt
    //     deny. Any other result (success, EACCES, ENOENT, ENOSPC and so on)
    //     is a refusal that names the errno.
    //
    // Local invocations are not official, so participant workflows do not
    // change.
    private static func requireConfinementOnOfficialRun(subcommand: String) throws {
        let officialRun = environmentValue("MLXFAST_OFFICIAL_BENCHMARK_RUN", fallback: "0") == "1"
            || environmentValue("RUNNER_ENVIRONMENT", fallback: "") == "self-hosted"
        guard officialRun else {
            return
        }
        if environmentValue("MLXFAST_NO_SANDBOX", fallback: "0") == "1" {
            throw MLXFastError.invalidInput(
                "\(subcommand) in an official run must be confined; unset MLXFAST_NO_SANDBOX"
            )
        }
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
        let temporaryDirectory = environmentValue("TMPDIR", fallback: "")
        guard !temporaryDirectory.isEmpty else {
            throw MLXFastError.invalidInput(
                "\(subcommand) in an official run must be started by tools/sandboxed-cli.sh, "
                    + "which sets TMPDIR to the private directory the profile allows; TMPDIR is unset"
            )
        }
        let control = URL(fileURLWithPath: temporaryDirectory)
            .appendingPathComponent(".mlxfast-confinement-control-\(UUID().uuidString)").path
        let controlDescriptor = open(control, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        guard controlDescriptor >= 0 else {
            throw MLXFastError.invalidInput(
                "\(subcommand) confinement probe: the positive control could not create \(control) "
                    + "(\(errnoDescription(errno))); without it a denied write proves nothing"
            )
        }
        close(controlDescriptor)
        unlink(control)

        let executableDirectory = URL(fileURLWithPath: try currentExecutablePath())
            .deletingLastPathComponent()
        let probe = executableDirectory
            .appendingPathComponent(".mlxfast-confinement-probe-\(UUID().uuidString)").path
        let probeDescriptor = open(probe, O_WRONLY | O_CREAT | O_EXCL, 0o600)
        let probeErrno = errno
        if probeDescriptor >= 0 {
            close(probeDescriptor)
            unlink(probe)
            throw MLXFastError.invalidInput(
                "\(subcommand) is not confined: a write outside the allowed paths (\(probe)) "
                    + "succeeded. Start it with tools/sandboxed-cli.sh."
            )
        }
        guard probeErrno == EPERM else {
            throw MLXFastError.invalidInput(
                "\(subcommand) confinement probe: the write outside the allowed paths (\(probe)) "
                    + "failed with \(errnoDescription(probeErrno)), not EPERM, so the failure "
                    + "does not prove a Seatbelt deny. Start it with tools/sandboxed-cli.sh."
            )
        }
    }

    private static func errnoDescription(_ code: Int32) -> String {
        let names: [Int32: String] = [
            EPERM: "EPERM", ENOENT: "ENOENT", EACCES: "EACCES", EEXIST: "EEXIST",
            ENOTDIR: "ENOTDIR", EISDIR: "EISDIR", ENOSPC: "ENOSPC", EROFS: "EROFS",
            EDQUOT: "EDQUOT", ELOOP: "ELOOP", ENAMETOOLONG: "ENAMETOOLONG",
        ]
        let name = names[code] ?? "errno"
        return "\(name) \(code): \(String(cString: strerror(code)))"
    }

    // The runner service exports these on the ranked box, and the workflow's
    // runner-environment step refuses a box that does not. The profile denies
    // them, so an official run needs all four.
    private static let requiredEvaluatorPathVariables = [
        "MLXFAST_QWEN38_GOLDEN_DIR",
        "MLXFAST_BASELINE_WORKSPACE",
        "MLXFAST_BASELINE_CALIBRATION",
        "BENCHD_BIN_DIR",
    ]

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

    private static func runCheckpointShards(_ options: ParsedOptions) throws {
        try options.validate(valueOptions: ["--index"])
        let indexPath = options.value(for: "--index", default: "")
        guard !indexPath.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint-shards requires --index PATH")
        }
        // Trusted code only: this verb must not call into the editable
        // transform module. On an official run setup.sh starts it through
        // tools/sandboxed-cli.sh.
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
