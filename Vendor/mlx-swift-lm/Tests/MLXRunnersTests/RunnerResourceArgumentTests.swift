// RunnerResourceArgumentTests.swift
//
// `--resource <name>=<path>` parsing. Every refusal must fire BEFORE the
// load, so these are pure argv/filesystem checks with no runner in sight.

import Foundation
import Testing

@testable import MLXRunners

@Suite("Runner resource arguments")
struct RunnerResourceArgumentTests {

    /// A temporary directory holding one regular file, one subdirectory and
    /// one fifo, so the accept and refuse paths are exercised against real
    /// filesystem objects rather than a stubbed manager.
    struct Sandbox {
        let root: URL
        let file: URL
        let directory: URL
        let fifo: URL

        init() throws {
            root = FileManager.default.temporaryDirectory
                .appendingPathComponent("runner-resources-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            file = root.appendingPathComponent("table.bin")
            try Data([0x01, 0x02]).write(to: file)
            directory = root.appendingPathComponent("shards")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            fifo = root.appendingPathComponent("pipe")
            #expect(mkfifo(fifo.path, 0o600) == 0)
        }

        func cleanUp() {
            try? FileManager.default.removeItem(at: root)
        }
    }

    @Test("A file and a directory both parse, and the values pass through opaque")
    func parsesFilesAndDirectories() throws {
        let sandbox = try Sandbox()
        defer { sandbox.cleanUp() }

        let resources = try RunnerResourceArguments.parse([
            "ngram=\(sandbox.file.path)",
            "shards=\(sandbox.directory.path)",
        ])
        // Boxed as NSURL, read back as URL: no new type crosses the seam and
        // the worker never opened either location.
        #expect((resources["ngram"] as? URL)?.path == sandbox.file.path)
        #expect((resources["shards"] as? URL)?.path == sandbox.directory.path)
        #expect(resources["absent"] == nil)
    }

    @Test("A path containing '=' keeps everything after the first separator")
    func splitsOnTheFirstSeparator() throws {
        let sandbox = try Sandbox()
        defer { sandbox.cleanUp() }
        let odd = sandbox.root.appendingPathComponent("a=b.bin")
        try Data([0x00]).write(to: odd)

        let resources = try RunnerResourceArguments.parse(["ngram=\(odd.path)"])
        #expect((resources["ngram"] as? URL)?.path == odd.path)
    }

    @Test("An argument that is not name=path is refused")
    func refusesMalformed() throws {
        let sandbox = try Sandbox()
        defer { sandbox.cleanUp() }
        for argument in ["ngram", "=\(sandbox.file.path)", "ngram="] {
            #expect(throws: RunnerResourceArgumentError.malformed(argument)) {
                _ = try RunnerResourceArguments.parse([argument])
            }
        }
    }

    /// REFUSAL 1: a repeated name. Last-one-wins would make the meaning of a
    /// command depend on argument order.
    @Test("A duplicate name is refused before the load")
    func refusesDuplicateName() throws {
        let sandbox = try Sandbox()
        defer { sandbox.cleanUp() }

        #expect(throws: RunnerResourceArgumentError.duplicateName("ngram")) {
            _ = try RunnerResourceArguments.parse([
                "ngram=\(sandbox.file.path)",
                "ngram=\(sandbox.directory.path)",
            ])
        }
    }

    /// REFUSAL 2: a path that is not there, or is there but is not something
    /// a runner could open.
    @Test("A missing path is refused by name")
    func refusesMissingPath() throws {
        let sandbox = try Sandbox()
        defer { sandbox.cleanUp() }
        let absent = sandbox.root.appendingPathComponent("not-here.bin").path

        #expect(
            throws: RunnerResourceArgumentError.pathMissing(name: "ngram", path: absent)
        ) {
            _ = try RunnerResourceArguments.parse(["ngram=\(absent)"])
        }
    }

    @Test("A path that is neither a file nor a directory is refused by name")
    func refusesNonFilePath() throws {
        let sandbox = try Sandbox()
        defer { sandbox.cleanUp() }

        #expect(
            throws: RunnerResourceArgumentError.pathNotAFileOrDirectory(
                name: "pipe", path: sandbox.fifo.path)
        ) {
            _ = try RunnerResourceArguments.parse(["pipe=\(sandbox.fifo.path)"])
        }
    }

    @Test("No resources at all is an empty bag, not a refusal")
    func emptyIsFine() throws {
        let resources = try RunnerResourceArguments.parse([])
        #expect(resources["anything"] == nil)
    }
}
