// Copyright © 2026 Eigen Labs.
//
// MLXRunners — the RESIDENT worker (contract §12f) and the attach path.
//
// The rule is David's, 2026-08-30: one weight load per measurement window.
// benchd starts a worker per PHASE — warmup, timed prefill, timed decode,
// correctness, twice per leg, once per cohort pair — so a worker that loads
// on startup loads a 113 GB checkpoint a dozen times for one window. The
// ranked pipeline has twenty minutes. It does not fit.
//
// So the weights get an owner. `bench-worker resident` is one process for
// one window: it validates argv, loads ONCE, and then listens on a Unix
// socket. Each phase's `bench-worker runtime-worker` attaches, speaks the
// whole Engine Protocol v1 session over that socket, and loads nothing.
//
// Modelled on the CUDA track's `ds4-resident`
// (cudafast-qwen38-125b-a6b-engine-dev: `harness/protocol-adapter/src/
// resident.rs`, `ds4_shim/ds4_resident.c`, `docs/ds4-resident.md`), with one
// deliberate difference. ds4 lets a second connection wait in the listen
// backlog; here it is REFUSED. A resident serving two attached workers
// stalls benchd's window, and a queued second phase looks like a slow engine
// rather than a double attach — the failure names itself this way.
//
// What is NOT changed by residency:
//
//   * phase isolation — the allocator is drained when a connection is
//     accepted, before the client's first byte, so a phase can never inherit
//     the one before it, and the per-phase drain still runs as today;
//   * the clock — the parent times the round trip, the worker still emits no
//     timing;
//   * the session — a fresh nonce per connection, hello first, and any error
//     discards that session exactly as in process.
//
// Identity on the wire: the attaching worker appends a `resident` object
// carrying the resident's pid and `load_epoch`, so an artifact can SHOW that
// every phase of a window ran against one load rather than assuming it.
//
// `backend` is NOT touched. It names the COMPUTE backend, and the
// conformance kit compares it to the manifest's own `backend` — an
// `mlx-resident` there is refused as "not the manifest backend mlx", which
// is correct: residency is a topology fact about WHERE the weights live, not
// a different way of computing. The ds4 precedent did label the backend,
// because it had no `resident` block to put the fact in; this one does.
//
// Environment:
//
//   BENCH_WORKER_RESIDENT_SOCKET   names the resident's socket; when set,
//                                  `runtime-worker` attaches instead of
//                                  serving in process. Absent, behaviour is
//                                  exactly as today.
//   BENCH_WORKER_RESIDENT_HELLO=1  emit the `resident` hello object. OFF by
//                                  default: benchd's envelope is closed, so
//                                  a benchd that does not yet admit the
//                                  field rejects the whole hello line. See
//                                  ``ResidentIdentity``.
//
// benchd side (bench-dev): `BENCH_WORKER_` joins the engine child
// environment allowlist beside `DS4_`, so the socket name reaches the
// spawned worker, and the closed envelope must admit `resident` before this
// ships.

import Foundation
import MLX

#if canImport(Darwin)
    import Darwin
#endif

/// Environment names the resident pair reads.
public enum BenchWorkerResidentEnvironment {
    /// Names the resident's socket. Set by the window's tooling.
    public static let socket = "BENCH_WORKER_RESIDENT_SOCKET"
    /// Opt in to the additive `resident` hello object.
    public static let helloIdentity = "BENCH_WORKER_RESIDENT_HELLO"
}

/// Refusals from the resident pair. Every one is a REFUSAL: a resident that
/// is unreachable, busy, or holding different weights must fail the phase,
/// never fall through to loading the checkpoint in process. A silent
/// fall-through would load 113 GB inside a timed window and report the
/// result as if it had not.
public enum BenchWorkerResidentError: Error, CustomStringConvertible, Equatable {
    case unreachable(path: String, detail: String)
    case refused(String)
    case noHello(path: String)
    case weightsMismatch(resident: String, requested: String)

    public var description: String {
        switch self {
        case .unreachable(let path, let detail):
            return "resident at \(path) is unreachable (\(detail))"
        case .refused(let detail):
            return "resident refused the attach (\(detail))"
        case .noHello(let path):
            return "resident at \(path) sent no hello"
        case .weightsMismatch(let resident, let requested):
            return "resident holds \(resident) but this phase asked for \(requested)"
        }
    }
}

// MARK: - Resident

/// One process, one load, one window.
///
/// `serve()` blocks on the accept loop. A connection is one benchd phase: on
/// accept the allocator is drained and a fresh `BenchWorkerServer` — new
/// nonce, new session — speaks the whole protocol over it. Exactly ONE
/// connection is served at a time; anything that arrives meanwhile gets one
/// `ok:false` line and is closed, because a resident that quietly serves two
/// attached workers stalls the window it exists to make fast.
public final class BenchWorkerResident: @unchecked Sendable {

    /// The one loaded runner, shared by every phase of this window.
    private let runner: any Runner
    private let listener: BenchWorkerSocketListener
    private let trusted: Bool
    private let speculative: Bool
    private let build: String
    private let device: String
    private let kvBytesCapacity: Int
    private let maxDecodeTokens: Int
    private let memory: any WorkerMemoryReporter
    /// One nonce per CONNECTION. Injectable so a conformance test can pin
    /// the fixture's, which is the only reason a relayed line could differ
    /// from an in-process one.
    private let nonceFactory: @Sendable () -> String
    /// The checkpoint this resident holds, so an attaching worker can refuse
    /// a resident loaded from somewhere else.
    private let weightsPath: String
    /// Written when the resident starts listening and removed at teardown,
    /// when the caller asked for one. The window's tooling reads it to reap
    /// a resident it did not start.
    private let pidfilePath: String?

    /// This resident's identity. `loadEpoch` is stamped once, when the load
    /// finished, and never moves again — so every phase of one window seals
    /// the same value, and a value that CHANGED is a reload that should not
    /// have happened.
    public let identity: ResidentIdentity

    /// `NSCondition` rather than a plain lock: a phase that has just
    /// disconnected leaves its session thread briefly finishing, and a NEW
    /// phase arriving in that window is a sequential attach, not a
    /// concurrent one. The doorman waits out that handover before it
    /// refuses, so the one-connection rule catches a real double attach
    /// without failing benchd's next phase for a scheduling gap.
    private let condition = NSCondition()
    private var sessionActive = false
    private var stopped = false

    /// How long a new attach waits for a FINISHING session before it is
    /// refused. Teardown is microseconds — the client has already closed and
    /// the session thread only has to notice EOF — so this is three orders
    /// of magnitude of slack, while a genuine double attach still hears back
    /// in half a second.
    public static let sessionHandoverGrace: TimeInterval = 0.5

    public init(
        runner: any Runner,
        listener: BenchWorkerSocketListener,
        weightsPath: String,
        trusted: Bool,
        speculative: Bool,
        build: String,
        device: String,
        kvBytesCapacity: Int,
        pidfilePath: String? = nil,
        maxDecodeTokens: Int = 4096,
        memory: any WorkerMemoryReporter = MLXMemoryReporter(),
        loadEpoch: UInt64 = DispatchTime.now().uptimeNanoseconds,
        nonceFactory: @escaping @Sendable () -> String = { UUID().uuidString }
    ) {
        self.runner = runner
        self.listener = listener
        self.weightsPath = weightsPath
        self.trusted = trusted
        self.speculative = speculative
        self.build = build
        self.device = device
        self.kvBytesCapacity = kvBytesCapacity
        self.pidfilePath = pidfilePath
        self.maxDecodeTokens = maxDecodeTokens
        self.memory = memory
        self.nonceFactory = nonceFactory
        self.identity = ResidentIdentity(
            pid: ProcessInfo.processInfo.processIdentifier, loadEpoch: loadEpoch)
    }

    /// Write the pidfile, then accept until shutdown. Blocks the calling
    /// thread; returns when the listener is closed.
    public func serve() {
        if let pidfilePath {
            try? Data("\(identity.pid)\n".utf8).write(
                to: URL(fileURLWithPath: pidfilePath))
        }
        acceptLoop()
    }

    /// Accept until `shutDown()`.
    private func acceptLoop() {
        while true {
            condition.lock()
            let done = stopped
            condition.unlock()
            if done { return }

            guard let connection = listener.accept() else { return }

            condition.lock()
            let deadline = Date().addingTimeInterval(Self.sessionHandoverGrace)
            while sessionActive, Date() < deadline {
                _ = condition.wait(until: deadline)
            }
            let busy = sessionActive
            if !busy { sessionActive = true }
            condition.unlock()

            guard !busy else {
                // ONE line, then closed. The refusal is a protocol response
                // so the attaching worker can report it verbatim instead of
                // guessing at a dropped connection.
                var response = WorkerResponse(id: -1, ok: false)
                response.error =
                    "resident is already serving a phase; one connection at a time"
                connection.write(line: response.jsonLine())
                connection.close()
                continue
            }

            // Serve OFF the accept loop, so the loop is back at `accept()`
            // and can refuse a second attach immediately rather than leaving
            // it in the backlog looking like a slow engine.
            let session = connection
            Thread.detachNewThread { [weak self] in
                guard let self else { return }
                self.runSession(over: session)
                self.condition.lock()
                self.sessionActive = false
                self.condition.broadcast()
                self.condition.unlock()
            }
        }
    }

    /// Ask the resident to stop, from ANY thread — a signal source
    /// included.
    ///
    /// This is the seam the SIGTERM handler uses, and it is deliberately
    /// `nonisolated` and free of Swift concurrency: it takes a lock, sets a
    /// flag, and closes the listener, which is what wakes the blocked
    /// `accept()`. A handler that instead called an actor-isolated or
    /// MainActor-isolated method would run the executor-isolation check on
    /// the dispatch-source thread and TRAP — the crash this replaces, where
    /// the resident died on SIGTERM without ever unlinking its socket.
    ///
    /// Idempotent, so the handler and the accept loop can both call it.
    public nonisolated func requestShutdown() {
        condition.lock()
        stopped = true
        condition.broadcast()
        condition.unlock()
        listener.shutDown()
    }

    /// Full teardown: stop accepting, close the socket, remove the pidfile.
    /// Idempotent, and safe after `requestShutdown()`.
    public nonisolated func shutDown() {
        requestShutdown()
        guard let pidfilePath else { return }
        try? FileManager.default.removeItem(atPath: pidfilePath)
    }

    /// One phase.
    func runSession(over connection: BenchWorkerSocketConnection) {
        // PHASE START: drain before the client's first byte, so a phase
        // cannot inherit the allocator state of the phase before it. The
        // worker's own `phase_diagnostics` drain runs too; that is
        // idempotent and deliberate.
        // The reporter's drain settles the stream first; see
        // `MLXMemoryReporter.drain`.
        memory.drain()

        let transport = BenchWorkerSocketTransport(connection: connection)
        let server = BenchWorkerServer(
            runner: runner,
            transport: transport,
            trusted: trusted,
            speculative: speculative,
            build: build,
            device: device,
            kvBytesCapacity: kvBytesCapacity,
            maxDecodeTokens: maxDecodeTokens,
            memory: memory,
            residentIdentity: identity,
            residentWeights: weightsPath,
            nonce: nonceFactory())

        let finished = DispatchSemaphore(value: 0)
        Task {
            await server.run()
            finished.signal()
        }
        finished.wait()
        connection.close()
        // PHASE END: settle and drain before the next accept, so the
        // handover window never leaves cached bytes for the next phase.
        memory.drain()
    }
}

/// The last thing a serving process does before `exit`.
///
/// WHY. A phase that ends in a protocol fault leaves the engine's MLX graph
/// half-evaluated: a Metal command encoder is open with work not yet
/// committed. `exit` then runs the C++ static destructors, MLX's
/// `CommandEncoder` destructor commits the buffer, and Metal aborts the
/// process on `commit command buffer with uncommitted encoder` (SIGABRT, the
/// box's `.ips` after every failed official leg). Synchronizing the default
/// stream first commits and waits for whatever is pending, so the destructors
/// find nothing open. A clean window is unaffected: there is nothing to wait
/// for.
public enum BenchWorkerTeardown {
    public static func synchronizeMLX() {
        MLX.Stream().synchronize()
    }
}

/// Termination handling for a resident, installed from a NON-isolated
/// context.
///
/// This lives here rather than in `main.swift` for one reason, and it is the
/// bug this fixes: top-level code in `main.swift` is `@MainActor`-isolated,
/// so a `DispatchSource` event handler written there is a MainActor-isolated
/// `@Sendable` thunk. Dispatch runs it on the signal source's own thread,
/// Swift concurrency checks the executor, and the process traps
/// (`_dispatch_assert_queue_fail` under `swift_task_isCurrentExecutor…`).
/// On the box that killed a resident holding 113 GB before it could unlink
/// its socket.
///
/// The handler therefore does the least possible: it asks the resident to
/// stop. The accept loop observes that, returns, and the OWNER performs the
/// teardown and exits — no isolation is touched from the signal thread, and
/// nothing about the isolation checks is relaxed to make the trap go away.
public enum BenchWorkerResidentTermination {

    /// Signals a window's tooling uses to reap a resident.
    public static let signals: [Int32] = [SIGINT, SIGTERM]

    /// Install handlers that ask `resident` to stop.
    ///
    /// The returned sources MUST be retained: a `DispatchSourceSignal` that
    /// goes out of scope stops firing.
    public static func install(
        for resident: BenchWorkerResident,
        signals: [Int32] = BenchWorkerResidentTermination.signals,
        queue: DispatchQueue = DispatchQueue(label: "bench-worker.resident.signals")
    ) -> [DispatchSourceSignal] {
        signals.map { number in
            // The default disposition would kill the process before the
            // source ever runs.
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: queue)
            source.setEventHandler { resident.requestShutdown() }
            source.resume()
            return source
        }
    }
}

/// The protocol loop's transport over one accepted connection.
final class BenchWorkerSocketTransport: BenchWorkerTransport {
    private let connection: BenchWorkerSocketConnection

    init(connection: BenchWorkerSocketConnection) {
        self.connection = connection
    }

    func readLine() -> String? { connection.readLine() }
    func write(line: String) { connection.write(line: line) }
}

// MARK: - Attach

/// `runtime-worker` attached to a resident: a verb-by-verb relay.
///
/// Every response is forwarded to stdout UNCHANGED except the hello, where
/// the `resident` identity is appended last and the resident's private
/// `resident_weights` is dropped. `backend` is left alone: it names the
/// COMPUTE backend and the conformance kit checks it against the manifest's,
/// so residency belongs in its own block and not in that field. Nothing else
/// is rewritten — a relay that reformatted responses would make the socket
/// path and the in-process path differ in bytes for no reason, and benchd
/// compares bytes.
public struct BenchWorkerAttach {

    private let socketPath: String
    private let weightsPath: String
    private let speculative: Bool
    private let emitResidentIdentity: Bool

    public init(
        socketPath: String,
        weightsPath: String,
        speculative: Bool,
        emitResidentIdentity: Bool
    ) {
        self.socketPath = socketPath
        self.weightsPath = weightsPath
        self.speculative = speculative
        self.emitResidentIdentity = emitResidentIdentity
    }

    /// Whether the `resident` hello object may ride.
    ///
    /// BOTH gates: the v1.1 speculative protocol, like every other additive
    /// hello field, and the explicit opt-in, because benchd's envelope is
    /// closed and rejects a field it does not know. Drop the second gate
    /// once bench-dev admits `resident`.
    public static func emitsResidentIdentity(
        speculative: Bool, environment: [String: String]
    ) -> Bool {
        guard speculative else { return false }
        return environment[BenchWorkerResidentEnvironment.helloIdentity] == "1"
    }

    /// Relay `input` to the resident and its answers to `output`.
    ///
    /// Throws — never falls back to in-process serving — when the resident is
    /// unreachable, sends no hello, refuses the attach, or holds a different
    /// checkpoint than this phase asked for.
    public func relay(
        input: any BenchWorkerTransport,
        output: any BenchWorkerTransport
    ) throws {
        let connection: BenchWorkerSocketConnection
        do {
            connection = try BenchWorkerSocketClient.connect(path: socketPath)
        } catch {
            throw BenchWorkerResidentError.unreachable(
                path: socketPath,
                detail: (error as? CustomStringConvertible)?.description ?? "\(error)")
        }
        defer { connection.close() }

        guard let helloLine = connection.readLine() else {
            throw BenchWorkerResidentError.noHello(path: socketPath)
        }
        output.write(line: try rewriteHello(helloLine))

        // Verb by verb, in order. One request line, one response line; a
        // resident that closed mid-phase ends the relay rather than letting
        // the parent wait on a response that will never come.
        while let request = input.readLine() {
            if request.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            connection.write(line: request)
            guard let response = connection.readLine() else {
                throw BenchWorkerResidentError.unreachable(
                    path: socketPath, detail: "the resident closed mid-phase")
            }
            output.write(line: response)
        }
    }

    /// The one rewritten line.
    func rewriteHello(_ line: String) throws -> String {
        var hello: WorkerResponse
        do {
            hello = try JSONDecoder().decode(WorkerResponse.self, from: Data(line.utf8))
        } catch {
            throw BenchWorkerResidentError.refused(
                "unreadable hello from the resident: \(line)")
        }
        guard hello.ok else {
            throw BenchWorkerResidentError.refused(
                hello.error ?? "the resident refused without a reason")
        }
        // The weights check is the whole reason the resident states its path:
        // a window whose phases attach to a resident holding a DIFFERENT
        // checkpoint would produce numbers for a model nobody asked about,
        // and every field on the wire would look right.
        if let residentWeights = hello.residentWeights {
            let held = URL(fileURLWithPath: residentWeights).standardizedFileURL.path
            let asked = URL(fileURLWithPath: weightsPath).standardizedFileURL.path
            guard held == asked else {
                throw BenchWorkerResidentError.weightsMismatch(
                    resident: held, requested: asked)
            }
        }

        // `backend` untouched: see the type's doc comment.
        hello.residentWeights = nil
        if !emitResidentIdentity { hello.resident = nil }
        return hello.jsonLine()
    }
}
