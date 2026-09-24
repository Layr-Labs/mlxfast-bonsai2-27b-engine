// BenchWorkerLaunchTests.swift
//
// bench-worker's argv ORDER. Every argument refusal must be answerable from
// argv alone: benchd points the worker at a checkpoint that can be 113 GB,
// so a wrong flag reported as "cannot read <weights>/config.json" both names
// the wrong thing and, on a real box, only after the operator has waited.
//
// The `--weights` path used throughout is deliberately one that does not
// exist: if any check reached the checkpoint, these tests would report a
// filesystem error instead of the flag.

import Foundation
import Testing

@testable import MLXRunners

@Suite("bench-worker launch arguments")
struct BenchWorkerLaunchTests {

    private let absentCheckpoint = "/nonexistent"

    private func parse(_ arguments: [String]) throws -> BenchWorkerLaunchOptions {
        try BenchWorkerLaunchOptions.parse(arguments)
    }

    @Test("The official spawn parses")
    func officialSpawn() throws {
        let launch = try parse([
            "runtime-worker", "--weights", absentCheckpoint,
            "--speculative-protocol", "v1.1",
        ])
        #expect(launch.weights.path == absentCheckpoint)
        #expect(launch.speculative)
        #expect(launch.trusted == false)
        #expect(launch.runnerID == nil)
    }

    @Test("Without the flag the launch is plain v1")
    func plainV1Spawn() throws {
        let launch = try parse(["runtime-worker", "--weights", absentCheckpoint])
        #expect(launch.speculative == false)
    }

    /// The reported defect: a wrong `--speculative-protocol` answered with a
    /// checkpoint read.
    @Test("A wrong --speculative-protocol refuses before the checkpoint is read")
    func speculativeProtocolRefusedFirst() {
        #expect(throws: SpeculativeProtocol.ArgumentError.unsupportedVersion("v9")) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint,
                "--speculative-protocol", "v9",
            ])
        }
    }

    @Test("A duplicate --resource refuses before the checkpoint is read")
    func duplicateResourceRefusedFirst() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("launch-\(UUID().uuidString).bin")
        try Data([0x00]).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }

        #expect(throws: RunnerResourceArgumentError.duplicateName("ngram")) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint,
                "--resource", "ngram=\(file.path)",
                "--resource", "ngram=\(file.path)",
            ])
        }
    }

    @Test("A missing --resource path refuses before the checkpoint is read")
    func missingResourceRefusedFirst() {
        #expect(
            throws: RunnerResourceArgumentError.pathMissing(
                name: "ngram", path: "/nonexistent/table.bin")
        ) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint,
                "--resource", "ngram=/nonexistent/table.bin",
            ])
        }
    }

    @Test("An empty --runner id refuses before the checkpoint is read")
    func emptyRunnerIDRefusedFirst() {
        #expect(throws: WorkerLaunchError.emptyRunnerID) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint, "--runner", "   ",
            ])
        }
    }

    @Test("An unknown argument, a missing value and a bad --kv-bytes all refuse")
    func argumentShapeRefusals() {
        #expect(throws: WorkerLaunchError.unknownArgument("--nope")) {
            _ = try parse(["runtime-worker", "--weights", absentCheckpoint, "--nope"])
        }
        #expect(throws: WorkerLaunchError.missingValue("--weights")) {
            _ = try parse(["runtime-worker", "--weights"])
        }
        #expect(throws: WorkerLaunchError.badValue("--kv-bytes")) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint, "--kv-bytes", "lots",
            ])
        }
        #expect(throws: WorkerLaunchError.missingWeights) {
            _ = try parse(["runtime-worker", "--speculative-protocol", "v1.1"])
        }
    }

    /// The ordering stated as an invariant rather than inferred from one
    /// case: parsing a launch whose checkpoint does not exist SUCCEEDS, so
    /// nothing in it can have read the checkpoint. Resolution is the first
    /// thing that does, and it is a separate call.
    @Test("Parsing never reads the checkpoint; resolveRunner is what does")
    func parsingNeverReadsTheCheckpoint() throws {
        let launch = try parse([
            "runtime-worker", "--weights", absentCheckpoint,
            "--speculative-protocol", "v1.1", "--trusted",
        ])
        #expect(launch.trusted)

        #expect(throws: RunnerError.self) {
            _ = try launch.resolveRunner()
        }
    }

    @Test("An unknown --runner id refuses without reading the checkpoint")
    func unknownRunnerID() throws {
        let launch = try parse([
            "runtime-worker", "--weights", absentCheckpoint, "--runner", "layr/nope",
        ])
        #expect(throws: WorkerLaunchError.unknownRunner("layr/nope")) {
            _ = try launch.resolveRunner()
        }
    }
}

// MARK: - Resident subcommand

@Suite("bench-worker resident arguments")
struct BenchWorkerResidentLaunchTests {

    private let absentCheckpoint = "/nonexistent"

    private func parse(_ arguments: [String]) throws -> BenchWorkerLaunchOptions {
        try BenchWorkerLaunchOptions.parse(arguments)
    }

    @Test("The resident spawn parses, socket carried")
    func residentSpawn() throws {
        let launch = try parse([
            "resident", "--weights", absentCheckpoint,
            "--socket", "/tmp/window.sock", "--speculative-protocol", "v1.1",
        ])
        #expect(launch.subcommand == .resident)
        #expect(launch.socket == "/tmp/window.sock")
        #expect(launch.speculative)
    }

    @Test("No subcommand is the per-phase worker benchd spawns")
    func defaultSubcommand() throws {
        #expect(try parse(["--weights", absentCheckpoint]).subcommand == .runtimeWorker)
        #expect(
            try parse(["runtime-worker", "--weights", absentCheckpoint]).subcommand
                == .runtimeWorker)
    }

    /// Every one of these fires from argv alone: `--weights` points at a
    /// path that does not exist throughout, so any check that reached the
    /// checkpoint would report a filesystem error instead.
    @Test("resident without --socket is refused before the checkpoint is read")
    func residentRequiresSocket() {
        #expect(throws: WorkerLaunchError.missingSocket) {
            _ = try parse(["resident", "--weights", absentCheckpoint])
        }
        #expect(throws: WorkerLaunchError.missingSocket) {
            _ = try parse(["resident", "--weights", absentCheckpoint, "--socket", "  "])
        }
    }

    /// The attaching worker is told where the resident is by the
    /// ENVIRONMENT, so a `--socket` on `runtime-worker` means the caller
    /// believes something untrue about this process.
    @Test("--socket on runtime-worker is refused before the checkpoint is read")
    func runtimeWorkerRejectsSocket() {
        #expect(throws: WorkerLaunchError.socketNotAccepted("runtime-worker")) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint,
                "--socket", "/tmp/window.sock",
            ])
        }
        // Including the implicit form.
        #expect(throws: WorkerLaunchError.socketNotAccepted("runtime-worker")) {
            _ = try parse(["--weights", absentCheckpoint, "--socket", "/tmp/w.sock"])
        }
    }

    @Test("Two subcommands are refused")
    func ambiguousSubcommand() {
        #expect(
            throws: WorkerLaunchError.ambiguousSubcommand("runtime-worker then resident")
        ) {
            _ = try parse([
                "runtime-worker", "resident", "--weights", absentCheckpoint,
                "--socket", "/tmp/w.sock",
            ])
        }
    }

    @Test("A missing --socket value is refused")
    func socketNeedsAValue() {
        #expect(throws: WorkerLaunchError.missingValue("--socket")) {
            _ = try parse(["resident", "--weights", absentCheckpoint, "--socket"])
        }
    }

    /// The resident takes the same load-shaping flags a per-phase worker
    /// does, because it performs the load the phases then share.
    @Test("The resident carries the load flags")
    func residentCarriesLoadFlags() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("resident-launch-\(UUID().uuidString).bin")
        try Data([0x00]).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }

        let launch = try parse([
            "resident", "--weights", absentCheckpoint, "--socket", "/tmp/w.sock",
            "--drafter", "/nonexistent/assistant", "--trusted",
            "--resource", "ngram=\(file.path)",
        ])
        #expect(launch.trusted)
        #expect(launch.drafter?.path == "/nonexistent/assistant")
        #expect((launch.resources["ngram"] as? URL)?.path == file.path)
    }

    /// Ordering holds for the new subcommand too: parsing a resident launch
    /// whose checkpoint is absent SUCCEEDS, so nothing in it read the
    /// checkpoint.
    @Test("Resident parsing never reads the checkpoint")
    func residentParsingNeverReadsCheckpoint() throws {
        let launch = try parse([
            "resident", "--weights", absentCheckpoint, "--socket", "/tmp/w.sock",
        ])
        #expect(launch.weights.path == absentCheckpoint)
        #expect(throws: RunnerError.self) { _ = try launch.resolveRunner() }
    }
}
