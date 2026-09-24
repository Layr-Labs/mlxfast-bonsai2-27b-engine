// BenchWorkerManifestCommandTests.swift
//
// `bench-worker manifest` and the usage screens.
//
// The manifest command exists so benchd's conformance kit has a file to read
// for `--manifest`, and what it prints must be the CANONICAL bytes — the
// ones the digest is computed over — or a kit that hashes what it reads gets
// a different answer from the hello it is checking. That equality is what
// these pin, in both directions: the bytes, and the digest of those bytes.
//
// Nothing here loads weights. The `--weights` case runs against a directory
// holding only `config.json` and an index, so a command that read anything
// else would fail on a missing file rather than print.

import CryptoKit
import Foundation
import Testing

@testable import MLXRunners

@Suite("bench-worker manifest command")
struct BenchWorkerManifestCommandTests {

    /// A checkpoint with exactly the file the resolution may read, plus the
    /// index a real one carries.
    private func checkpoint(modelType: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("manifest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data("{\"model_type\":\"\(modelType)\"}".utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        try Data("{\"metadata\":{\"total_size\":0},\"weight_map\":{}}".utf8)
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    private func render(_ arguments: [String]) throws -> String {
        guard case .manifest(let options) = try BenchWorkerCommand.parse(arguments) else {
            Issue.record("\(arguments) did not parse as a manifest command")
            return ""
        }
        return try options.render()
    }

    // MARK: - The bytes

    /// The cross-repo vector, printed by the command: the digest of what it
    /// prints IS the digest the hello reports.
    @Test("--runner prints the canonical bytes, and they digest to the vector")
    func runnerPrintsCanonicalBytes() throws {
        let printed = try render(["manifest", "--runner", "layr/qwen4exp-125b-a6b"])
        #expect(printed.hasSuffix("\n"))

        let bytes = String(printed.dropLast())
        let canonical = String(
            decoding: ContractManifests.sectionEleven.canonicalJSON(), as: UTF8.self)
        #expect(bytes == canonical)

        let digest = SHA256Hex.of(Data(bytes.utf8))
        #expect(digest == "0430b22f8325c9c9371910d1e14eb3c78b235932bf35fa6623a4c511dd68e180")
        #expect(digest == ContractManifests.sectionEleven.sha256Digest())
    }

    @Test("--digest prints only the hex")
    func digestOnly() throws {
        let printed = try render([
            "manifest", "--runner", "layr/qwen4exp-125b-a6b", "--digest",
        ])
        #expect(
            printed
                == "0430b22f8325c9c9371910d1e14eb3c78b235932bf35fa6623a4c511dd68e180\n")
    }

    /// `--weights` reaches the same runner through the registry, reading
    /// only `config.json`.
    @Test("--weights resolves the same manifest from config.json alone")
    func weightsResolvesTheSameManifest() throws {
        let directory = try checkpoint(modelType: "qwen4_exp_text")
        defer { try? FileManager.default.removeItem(at: directory) }

        let fromWeights = try render(["manifest", "--weights", directory.path])
        let fromID = try render(["manifest", "--runner", "layr/qwen4exp-125b-a6b"])
        #expect(fromWeights == fromID)
    }

    @Test("--weights honours --digest too")
    func weightsDigest() throws {
        let directory = try checkpoint(modelType: "qwen4_exp_text")
        defer { try? FileManager.default.removeItem(at: directory) }
        let printed = try render(["manifest", "--weights", directory.path, "--digest"])
        #expect(
            printed
                == "0430b22f8325c9c9371910d1e14eb3c78b235932bf35fa6623a4c511dd68e180\n")
    }

    @Test("Every first-party runner prints its own manifest")
    func everyFirstPartyRunner() throws {
        for manifest in RunnerRegistry.shared.manifests() {
            let printed = try render(["manifest", "--runner", manifest.runnerID])
            #expect(
                String(printed.dropLast())
                    == String(decoding: manifest.canonicalJSON(), as: UTF8.self))
        }
    }

    // MARK: - Refusals

    @Test("An unknown runner id is refused")
    func unknownRunnerRefused() {
        #expect(throws: WorkerLaunchError.unknownRunner("layr/nope")) {
            _ = try render(["manifest", "--runner", "layr/nope"])
        }
    }

    @Test("A checkpoint no runner claims is refused")
    func unclaimedModelTypeRefused() throws {
        let directory = try checkpoint(modelType: "not_a_family")
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(throws: RunnerRegistry.RegistryError.unknownModelType("not_a_family")) {
            _ = try render(["manifest", "--weights", directory.path])
        }
    }

    @Test("A checkpoint with no config.json is refused")
    func unreadableCheckpointRefused() throws {
        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent("manifest-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        #expect(throws: RunnerError.self) {
            _ = try render(["manifest", "--weights", empty.path])
        }
    }

    @Test("Both targets, or neither, are refused")
    func targetRefusals() {
        #expect(throws: WorkerLaunchError.manifestTargetAmbiguous) {
            _ = try BenchWorkerCommand.parse([
                "manifest", "--runner", "layr/gptoss", "--weights", "/nonexistent",
            ])
        }
        #expect(throws: WorkerLaunchError.manifestTargetMissing) {
            _ = try BenchWorkerCommand.parse(["manifest"])
        }
        #expect(throws: WorkerLaunchError.manifestTargetMissing) {
            _ = try BenchWorkerCommand.parse(["manifest", "--digest"])
        }
    }

    @Test("A flag the manifest command does not take is refused")
    func unknownFlagRefused() {
        #expect(throws: WorkerLaunchError.unknownArgument("--socket")) {
            _ = try BenchWorkerCommand.parse([
                "manifest", "--runner", "layr/gptoss", "--socket", "/tmp/s",
            ])
        }
        #expect(throws: WorkerLaunchError.missingValue("--runner")) {
            _ = try BenchWorkerCommand.parse(["manifest", "--runner"])
        }
    }

    // MARK: - Usage

    @Test("--help names every subcommand")
    func generalUsage() throws {
        guard case .help(let topic) = try BenchWorkerCommand.parse(["--help"]) else {
            Issue.record("--help did not parse as help")
            return
        }
        #expect(topic == .general)
        let text = BenchWorkerUsage.text(for: .general)
        for subcommand in ["runtime-worker", "resident", "manifest"] {
            #expect(text.contains(subcommand))
        }
    }

    @Test("Each subcommand has its own screen")
    func subcommandUsage() throws {
        let topics: [(String, BenchWorkerUsage.Topic, String)] = [
            ("runtime-worker", .runtimeWorker, "BENCH_WORKER_RESIDENT_SOCKET"),
            ("resident", .resident, "--socket"),
            ("manifest", .manifest, "--digest"),
        ]
        for (subcommand, expected, mustMention) in topics {
            guard
                case .help(let topic) = try BenchWorkerCommand.parse([
                    subcommand, "--help",
                ])
            else {
                Issue.record("\(subcommand) --help did not parse as help")
                continue
            }
            #expect(topic == expected)
            #expect(BenchWorkerUsage.text(for: expected).contains(mustMention))
        }
    }

    /// `--help` wins over the rest of the line: someone asking how to run
    /// this must not first have to get the line right.
    @Test("--help answers even when the rest of the line is wrong")
    func helpBeatsOtherRefusals() throws {
        guard
            case .help(let topic) = try BenchWorkerCommand.parse([
                "resident", "--help", "--socket",
            ])
        else {
            Issue.record("--help lost to another refusal")
            return
        }
        #expect(topic == .resident)
    }

    @Test("A mistyped subcommand is named as one")
    func unknownSubcommand() {
        #expect(throws: WorkerLaunchError.unknownSubcommand("residnet")) {
            _ = try BenchWorkerCommand.parse([
                "residnet", "--weights", "/nonexistent", "--socket", "/tmp/s",
            ])
        }
    }

    /// The serving subcommands still reach the launch parser unchanged.
    @Test("The serve subcommands still parse as before")
    func serveStillParses() throws {
        guard
            case .serve(let launch) = try BenchWorkerCommand.parse([
                "resident", "--weights", "/nonexistent", "--socket", "/tmp/s",
            ])
        else {
            Issue.record("resident did not parse as a serve command")
            return
        }
        #expect(launch.subcommand == .resident)
        #expect(launch.socket == "/tmp/s")
    }
}

/// sha256 hex, so the test hashes the printed BYTES rather than trusting the
/// manifest to hash itself.
enum SHA256Hex {
    static func of(_ data: Data) -> String {
        var hasher = SHA256()
        hasher.update(data: data)
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
