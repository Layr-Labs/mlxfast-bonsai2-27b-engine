// BenchWorkerResidentTests.swift
//
// The resident worker (contract §12f) and the attach path, over the mock
// runner on a temporary socket. No weights, no GPU.
//
// What these pin is the part residency can silently get wrong: the wire a
// phase sees must be the SAME bytes whether it was served in process or
// relayed through a resident — the hello excepted, which is the one line
// that makes a claim about where the work ran. Everything else here is a
// refusal, because every failure mode of a resident (unreachable, busy,
// holding the wrong checkpoint) has a plausible-looking wrong answer:
// loading the weights again inside a timed phase.

import Foundation
import Testing

@testable import MLXRunners

@Suite("bench-worker resident", .serialized)
struct BenchWorkerResidentTests {

    // MARK: Harness

    /// A resident on a private socket, torn down with the test.
    final class Harness {
        let resident: BenchWorkerResident
        let socketPath: String
        private let thread: Thread

        init(
            loadEpoch: UInt64 = 4242,
            script: MockRoundScript? = .fixtureWindow,
            nonce: String? = "fixturenonce"
        ) throws {
            socketPath = FileManager.default.temporaryDirectory
                .appendingPathComponent("resident-\(UUID().uuidString).sock").path
            let listener = try BenchWorkerSocketListener(path: socketPath)
            resident = BenchWorkerResident(
                runner: MockRunner(script: script),
                listener: listener,
                weightsPath: "/models/mock",
                trusted: false,
                speculative: true,
                build: "fixture",
                device: "mock",
                kvBytesCapacity: 1 << 20,
                maxDecodeTokens: 64,
                memory: FixtureMemoryReporter(),
                loadEpoch: loadEpoch,
                // Pinned so a relayed line can be compared to the fixture
                // byte for byte; nil takes the real per-connection UUID.
                nonceFactory: Harness.nonceFactory(nonce))
            let resident = self.resident
            thread = Thread { resident.serve() }
            thread.start()
        }

        static func nonceFactory(_ pinned: String?) -> @Sendable () -> String {
            guard let pinned else { return { UUID().uuidString } }
            return { pinned }
        }

        func cleanUp() {
            resident.shutDown()
        }
    }

    /// Collects what the relay wrote.
    final class CollectingTransport: BenchWorkerTransport {
        private var pending: [String]
        private(set) var written: [String] = []

        init(lines: [String] = []) { pending = lines }
        func readLine() -> String? { pending.isEmpty ? nil : pending.removeFirst() }
        func write(line: String) { written.append(line) }
    }

    private func attach(
        _ harness: Harness,
        weightsPath: String = "/models/mock",
        emitResidentIdentity: Bool = false
    ) -> BenchWorkerAttach {
        BenchWorkerAttach(
            socketPath: harness.socketPath,
            weightsPath: weightsPath,
            speculative: true,
            emitResidentIdentity: emitResidentIdentity)
    }

    private func fixture() throws -> BenchWorkerProtocolTests.Fixture {
        try BenchWorkerProtocolTests.Fixture()
    }

    // MARK: - The relayed session

    /// The whole shared fixture, through the socket. Every response line is
    /// byte-identical to the in-process worker's — the hello excepted, and
    /// that difference is asserted exactly rather than skipped.
    @Test("A relayed phase reproduces the fixture byte for byte")
    func relayedFixtureReplay() throws {
        let harness = try Harness()
        defer { harness.cleanUp() }
        let fixture = try fixture()

        let output = CollectingTransport()
        try attach(harness).relay(
            input: CollectingTransport(lines: fixture.requests), output: output)

        #expect(output.written.count == fixture.responses.count + 1)
        for (index, expected) in fixture.responses.enumerated() {
            #expect(
                output.written[index + 1] == expected,
                "relayed response \(index + 1) diverged from the fixture")
        }

        // The hello too, byte for byte. `backend` is NOT rewritten — it
        // names the compute backend and the conformance kit checks it
        // against the manifest's, which refused an `mlx-resident` there —
        // and the resident's private weights path never reaches benchd, so
        // with the identity gated off the relayed hello IS the fixture's.
        #expect(output.written[0] == fixture.hello)
        #expect(!output.written[0].contains("resident_weights"))
        #expect(!output.written[0].contains("\"resident\""))
    }

    /// The identity object rides only when BOTH gates are open, and it is
    /// LAST — after `runner`.
    @Test("The resident identity is gated and appended last")
    func residentIdentityGating() throws {
        let harness = try Harness(loadEpoch: 4242)
        defer { harness.cleanUp() }
        let fixture = try fixture()

        let output = CollectingTransport()
        try attach(harness, emitResidentIdentity: true).relay(
            input: CollectingTransport(lines: []), output: output)

        let hello = output.written[0]
        // The manifest's backend, unchanged by the relay.
        #expect(hello.contains("\"backend\":\"mock\""))
        #expect(!hello.contains("mlx-resident"))
        #expect(
            hello.hasSuffix(
                ",\"resident\":{\"pid\":\(harness.resident.identity.pid),"
                    + "\"load_epoch\":4242}}"))
        // Still after `runner`, which the fixture pins as the previous last
        // field.
        let runnerIndex = try #require(hello.range(of: "\"runner\":"))
        let residentIndex = try #require(hello.range(of: "\"resident\":"))
        #expect(runnerIndex.lowerBound < residentIndex.lowerBound)
        _ = fixture
    }

    @Test("Both gates are required for the identity object")
    func identityGateRequiresBoth() {
        let on = [BenchWorkerResidentEnvironment.helloIdentity: "1"]
        #expect(BenchWorkerAttach.emitsResidentIdentity(speculative: true, environment: on))
        // v1.1 absent: no additive hello fields at all.
        #expect(!BenchWorkerAttach.emitsResidentIdentity(speculative: false, environment: on))
        // Opt-in absent: benchd's envelope is closed and would reject the line.
        #expect(!BenchWorkerAttach.emitsResidentIdentity(speculative: true, environment: [:]))
        #expect(
            !BenchWorkerAttach.emitsResidentIdentity(
                speculative: true,
                environment: [BenchWorkerResidentEnvironment.helloIdentity: "0"]))
    }

    // MARK: - One connection at a time

    /// The ds4 lesson: a resident serving two attached workers stalls the
    /// window. The second attach is refused with one line and closed, so the
    /// failure names itself instead of looking like a slow engine.
    @Test("A second concurrent connection is refused")
    func secondConnectionRefused() throws {
        let harness = try Harness()
        defer { harness.cleanUp() }

        // First client: connect and read the hello, which proves the session
        // is established before the second attach is attempted.
        let first = try BenchWorkerSocketClient.connect(path: harness.socketPath)
        defer { first.close() }
        let firstHello = try #require(first.readLine())
        #expect(firstHello.contains("\"ok\":true"))

        let second = try BenchWorkerSocketClient.connect(path: harness.socketPath)
        defer { second.close() }
        let refusal = try #require(second.readLine())
        let decoded = try JSONDecoder().decode(
            WorkerResponse.self, from: Data(refusal.utf8))
        #expect(decoded.ok == false)
        #expect(
            decoded.error
                == "resident is already serving a phase; one connection at a time")
        // And it is CLOSED, not queued.
        #expect(second.readLine() == nil)
    }

    /// One load per window: the epoch a phase seals is the same one the next
    /// phase seals, which is the whole claim residency makes.
    @Test("load_epoch is constant across sequential connections")
    func loadEpochConstantAcrossPhases() throws {
        let harness = try Harness(loadEpoch: 99)
        defer { harness.cleanUp() }

        var epochs: [UInt64] = []
        var pids: [Int32] = []
        for _ in 0 ..< 2 {
            let output = CollectingTransport()
            try attach(harness, emitResidentIdentity: true).relay(
                input: CollectingTransport(lines: []), output: output)
            let hello = try JSONDecoder().decode(
                WorkerResponse.self, from: Data(output.written[0].utf8))
            epochs.append(try #require(hello.resident?.loadEpoch))
            pids.append(try #require(hello.resident?.pid))
        }
        #expect(epochs == [99, 99])
        #expect(pids[0] == pids[1])
    }

    /// A sequential second phase is SERVED — the one-at-a-time rule is about
    /// concurrency, not a one-shot resident.
    @Test("A sequential second phase is served, with a fresh nonce")
    func sequentialPhasesAreServed() throws {
        let harness = try Harness(nonce: nil)
        defer { harness.cleanUp() }
        let prefill =
            "{\"id\":1,\"kind\":\"prefill\",\"prompt_tokens\":[11,12,13]}"

        var nonces: [String] = []
        for _ in 0 ..< 2 {
            let output = CollectingTransport()
            try attach(harness).relay(
                input: CollectingTransport(lines: [prefill]), output: output)
            let answer = try JSONDecoder().decode(
                WorkerResponse.self, from: Data(output.written[1].utf8))
            #expect(answer.ok)
            #expect(answer.token == 100_003)
            nonces.append(try #require(answer.nonce))
        }
        // A session per connection: the nonce cannot be reused across
        // phases, or a stale line from one phase would validate in the next.
        #expect(nonces[0] != nonces[1])
    }

    // MARK: - Refusals

    /// A resident holding a different checkpoint would produce numbers for a
    /// model nobody asked about, with every field on the wire looking right.
    @Test("A resident holding different weights is refused")
    func weightsMismatchRefused() throws {
        let harness = try Harness()
        defer { harness.cleanUp() }

        #expect(
            throws: BenchWorkerResidentError.weightsMismatch(
                resident: "/models/mock", requested: "/models/other")
        ) {
            try attach(harness, weightsPath: "/models/other").relay(
                input: CollectingTransport(lines: []), output: CollectingTransport())
        }
    }

    /// No resident there: a refusal, never a fall-through to loading the
    /// checkpoint in process.
    @Test("An absent socket refuses rather than falling through")
    func absentSocketRefuses() {
        let absent = FileManager.default.temporaryDirectory
            .appendingPathComponent("no-resident-\(UUID().uuidString).sock").path
        let attach = BenchWorkerAttach(
            socketPath: absent, weightsPath: "/models/mock",
            speculative: true, emitResidentIdentity: false)

        do {
            try attach.relay(
                input: CollectingTransport(lines: []), output: CollectingTransport())
            Issue.record("the attach succeeded with no resident listening")
        } catch let error as BenchWorkerResidentError {
            guard case .unreachable = error else {
                Issue.record("refused with \(error), not unreachability")
                return
            }
        } catch {
            Issue.record("refused with \(error)")
        }
    }

    /// A resident that answers `ok:false` — the busy refusal, for instance —
    /// is reported verbatim rather than guessed at.
    @Test("A refusing resident is reported with its own reason")
    func refusingResidentIsReported() throws {
        let attach = BenchWorkerAttach(
            socketPath: "/unused", weightsPath: "/models/mock",
            speculative: true, emitResidentIdentity: false)
        var refusal = WorkerResponse(id: -1, ok: false)
        refusal.error = "resident is already serving a phase; one connection at a time"

        #expect(
            throws: BenchWorkerResidentError.refused(
                "resident is already serving a phase; one connection at a time")
        ) {
            _ = try attach.rewriteHello(refusal.jsonLine())
        }
    }

    @Test("An unreadable hello is refused, not forwarded")
    func unreadableHelloRefused() {
        let attach = BenchWorkerAttach(
            socketPath: "/unused", weightsPath: "/models/mock",
            speculative: true, emitResidentIdentity: false)
        #expect(throws: BenchWorkerResidentError.self) {
            _ = try attach.rewriteHello("{ not json")
        }
    }
}

// MARK: - Termination

/// The SIGTERM teardown, over the mock runner.
///
/// The defect this pins is a real one from M5 #4: the signal handler was a
/// closure written in `main.swift`, whose top-level code is
/// `@MainActor`-isolated, so dispatch ran a MainActor-isolated thunk on the
/// signal source's own thread, Swift concurrency checked the executor, and
/// the process trapped — killing a resident holding 113 GB before it could
/// unlink its socket. The handler now only ASKS the resident to stop; the
/// accept loop returns and its owner tears down.
///
/// `.serialized` because these install process-wide signal dispositions.
@Suite("bench-worker resident termination", .serialized)
struct BenchWorkerResidentTerminationTests {

    private func makeResident(
        socketPath: String, pidfilePath: String?
    ) throws -> BenchWorkerResident {
        BenchWorkerResident(
            runner: MockRunner(),
            listener: try BenchWorkerSocketListener(path: socketPath),
            weightsPath: "/models/mock",
            trusted: false,
            speculative: true,
            build: "fixture",
            device: "mock",
            kvBytesCapacity: 1 << 20,
            pidfilePath: pidfilePath,
            memory: FixtureMemoryReporter(),
            loadEpoch: 7)
    }

    /// Short, and under `/tmp`: a `sockaddr_un` path is capped near 104
    /// bytes and the per-user temporary directory already spends most of
    /// that, which is exactly what a box hits when it puts a window's socket
    /// somewhere tidy-looking.
    private func temporaryPath(_ suffix: String) -> String {
        "/tmp/bw-\(UUID().uuidString.prefix(8))\(suffix)"
    }

    /// The whole sequence a window's tooling depends on: a real SIGTERM
    /// delivered to this process, the accept loop returning rather than the
    /// process trapping, and the socket and pidfile gone afterwards.
    @Test("SIGTERM stops the accept loop and the owner tears down cleanly")
    func sigtermTearsDownCleanly() throws {
        let socketPath = temporaryPath(".sock")
        let pidfilePath = temporaryPath(".pid")
        let resident = try makeResident(socketPath: socketPath, pidfilePath: pidfilePath)

        // The same installation path main.swift uses, on SIGUSR2 rather than
        // SIGTERM. The mechanism under test — dispatch source, handler,
        // accept loop — is signal-number agnostic, while a real SIGTERM is
        // delivered to the whole TEST PROCESS and disturbs residents other
        // suites are running in parallel. The production list is asserted
        // separately below. Retained for the test's lifetime: a source that
        // goes out of scope stops firing.
        let sources = BenchWorkerResidentTermination.install(
            for: resident, signals: [SIGUSR2])
        defer {
            sources.forEach { $0.cancel() }
            signal(SIGUSR2, SIG_DFL)
        }

        let returned = DispatchSemaphore(value: 0)
        let thread = Thread {
            resident.serve()
            returned.signal()
        }
        thread.start()

        // Wait for the pidfile, which `serve()` writes before it accepts —
        // so the resident is listening and the signal cannot arrive early.
        var listening = false
        for _ in 0 ..< 200 where !listening {
            listening = FileManager.default.fileExists(atPath: pidfilePath)
            if !listening { usleep(10_000) }
        }
        #expect(listening)

        raise(SIGUSR2)

        // The accept loop RETURNS. Before the fix the process trapped here
        // instead, so a timeout is the failure this test exists to catch.
        #expect(returned.wait(timeout: .now() + 5) == .success)

        // The owner's teardown, exactly as main.swift runs it before exit 0.
        resident.shutDown()
        #expect(!FileManager.default.fileExists(atPath: socketPath))
        #expect(!FileManager.default.fileExists(atPath: pidfilePath))

        // And nothing can attach to a torn-down resident.
        #expect(throws: BenchWorkerSocketError.self) {
            _ = try BenchWorkerSocketClient.connect(path: socketPath)
        }
    }

    /// The signals a window's tooling actually sends.
    @Test("The production handlers cover SIGINT and SIGTERM")
    func productionSignals() {
        #expect(BenchWorkerResidentTermination.signals == [SIGINT, SIGTERM])
    }

    /// Teardown runs from the accept loop AND from a signal handler, so
    /// calling it twice must not close a descriptor twice or unlink a path a
    /// later resident has already rebound.
    @Test("Teardown is idempotent")
    func teardownIsIdempotent() throws {
        let socketPath = temporaryPath(".sock")
        let pidfilePath = temporaryPath(".pid")
        let resident = try makeResident(socketPath: socketPath, pidfilePath: pidfilePath)
        try Data("1\n".utf8).write(to: URL(fileURLWithPath: pidfilePath))

        resident.requestShutdown()
        resident.shutDown()
        resident.shutDown()
        resident.requestShutdown()

        #expect(!FileManager.default.fileExists(atPath: socketPath))
        #expect(!FileManager.default.fileExists(atPath: pidfilePath))
    }

    /// A resident asked to stop before it ever accepted still returns, and
    /// still leaves nothing behind.
    @Test("A shutdown before the first connection still returns")
    func shutdownBeforeFirstConnection() throws {
        let socketPath = temporaryPath(".sock")
        let resident = try makeResident(socketPath: socketPath, pidfilePath: nil)

        let returned = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            resident.serve()
            returned.signal()
        }
        usleep(50_000)
        resident.requestShutdown()
        #expect(returned.wait(timeout: .now() + 5) == .success)
        #expect(!FileManager.default.fileExists(atPath: socketPath))
    }

    /// The resident writes its pid, so tooling that did not start it can
    /// still reap it.
    @Test("The pidfile carries this process's pid")
    func pidfileCarriesThePid() throws {
        let socketPath = temporaryPath(".sock")
        let pidfilePath = temporaryPath(".pid")
        let resident = try makeResident(socketPath: socketPath, pidfilePath: pidfilePath)

        let returned = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            resident.serve()
            returned.signal()
        }
        var contents = ""
        for _ in 0 ..< 200 where contents.isEmpty {
            contents = (try? String(contentsOfFile: pidfilePath, encoding: .utf8)) ?? ""
            if contents.isEmpty { usleep(10_000) }
        }
        #expect(
            contents.trimmingCharacters(in: .whitespacesAndNewlines)
                == "\(ProcessInfo.processInfo.processIdentifier)")

        resident.shutDown()
        #expect(returned.wait(timeout: .now() + 5) == .success)
    }
}
