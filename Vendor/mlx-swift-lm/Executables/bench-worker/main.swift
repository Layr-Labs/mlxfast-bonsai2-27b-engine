// Copyright © 2026 Eigen Labs.
//
// bench-worker — Engine Protocol v1 server over `Runner`.
//
// benchd spawns `<engine> runtime-worker --weights <DIR> --speculative-protocol v1.1`.
//
// SUBCOMMANDS: `runtime-worker` (one phase), `resident` (one window), and
// `manifest`, which prints a runner's canonical declaration and loads no
// weights — the file benchd's conformance kit reads as `--manifest`.
// `--help`, alone or after a subcommand, prints one screen and exits 0.
//
// RESIDENT MODE (contract §12f). `bench-worker resident --weights <DIR>
// --socket <PATH>` is one process for one measurement window: it validates
// argv, loads ONCE, and listens. Each phase's `runtime-worker` attaches when
// `BENCH_WORKER_RESIDENT_SOCKET` names that socket and loads nothing, so a
// window costs one weight load instead of one per phase. Exactly one
// connection is served at a time; a second is refused with one line and
// closed. The attached hello reports `backend` = `mlx-resident` plus the
// resident's pid and `load_epoch` (gated — see `BenchWorkerResident.swift`
// for the environment names, the identity fields and the ds4 precedent).
// One binary serves every runner: the checkpoint's `config.json` `model_type`
// picks it out of `RunnerRegistry`, or `--runner <id>` names it.
//
// A SHIM. Argument parsing and validation live in
// `MLXRunners/BenchWorkerLaunch.swift` and the protocol loop in
// `MLXRunners/BenchWorker.swift`, so both are testable in process; a test
// target that depended on this executable would be run by SwiftPM as the
// swift-testing host and would abort the package's whole `@Test` pass (see
// Package.swift's BenchCBv2Core/BenchCBv2 note).
//
// The ORDER below is load bearing and reads top to bottom: validate argv,
// then resolve the runner (the first read under `--weights`), then load the
// weights ONCE, then serve. A checkpoint here can be 113 GB, so a bad flag
// must be answered from argv alone.

import Foundation
import MLXLMCommon
import MLXRunners

#if canImport(Darwin)
    import Darwin
#endif

/// Device identity for the hello.
///
/// The Metal device name is not exposed by mlx-swift at this pin, so the
/// probe reads the machine's CPU brand string, which names the same chip
/// ("Apple M5 Max" → "m5"). A machine that answers neither reports
/// "unknown" rather than a guess.
func probeDevice() -> String {
    #if canImport(Darwin)
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else {
            return "unknown"
        }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 else {
            return "unknown"
        }
        let brand = String(cString: buffer).lowercased()
        for token in brand.split(separator: " ") where token.count >= 2 && token.hasPrefix("m") {
            if token.dropFirst().allSatisfy(\.isNumber) { return String(token) }
        }
        return brand.isEmpty ? "unknown" : brand
    #else
        return "unknown"
    #endif
}

/// Usage on stderr, then a non-zero status: a mistyped subcommand is
/// answered with the whole screen, because the caller does not yet know
/// what the right words are.
func failWithUsage(_ error: some Error, topic: BenchWorkerUsage.Topic) -> Never {
    let message: String =
        (error as? CustomStringConvertible)?.description ?? "\(error)"
    FileHandle.standardError.write(
        Data(
            ("bench-worker: " + message + "\n\n" + BenchWorkerUsage.text(for: topic) + "\n")
                .utf8))
    exit(2)
}

/// One line on stderr and a non-zero status — never a Swift runtime trap.
/// benchd reads this line to tell an operator what to fix, and a crash
/// report buries it under a backtrace.
func fail(_ error: some Error) -> Never {
    let message: String =
        (error as? CustomStringConvertible)?.description ?? "\(error)"
    FileHandle.standardError.write(Data(("bench-worker: " + message + "\n").utf8))
    exit(2)
}

/// Retained for the resident's lifetime; a `DispatchSourceSignal` that goes
/// out of scope stops firing.
var residentSignalSources: [DispatchSourceSignal] = []

// 1. ARGV. Every argument refusal happens here, with nothing under
//    `--weights` read yet.
let rawArguments = Array(CommandLine.arguments.dropFirst())
let command: BenchWorkerCommand
do {
    command = try BenchWorkerCommand.parse(rawArguments)
} catch let error as WorkerLaunchError {
    if case .unknownSubcommand = error {
        failWithUsage(error, topic: .general)
    }
    failWithUsage(error, topic: BenchWorkerUsage.Topic(subcommand: rawArguments.first))
} catch {
    fail(error)
}

// 1a. The two commands that never touch a checkpoint.
switch command {
case .help(let topic):
    print(BenchWorkerUsage.text(for: topic))
    exit(0)
case .manifest(let options):
    // A manifest is static data on the runner TYPE. With `--weights` the
    // only file read is `config.json`; no weights are loaded, here or
    // anywhere below.
    do {
        FileHandle.standardOutput.write(Data(try options.render().utf8))
    } catch {
        fail(error)
    }
    exit(0)
case .serve:
    break
}

guard case .serve(let launch) = command else {
    fatalError("unreachable: the non-serving commands exited above")
}

// 2. ATTACH, before anything reads the checkpoint. A `runtime-worker` whose
//    environment names a resident loads NOTHING: the resident holds the
//    weights for the whole window. An unreachable, busy or wrongly-loaded
//    resident is a refusal — never a fall-through to loading 113 GB inside a
//    timed phase and reporting the result as if it had not.
let environment = ProcessInfo.processInfo.environment
if launch.subcommand == .runtimeWorker,
    let socket = environment[BenchWorkerResidentEnvironment.socket],
    !socket.trimmingCharacters(in: .whitespaces).isEmpty
{
    let attach = BenchWorkerAttach(
        socketPath: socket,
        weightsPath: launch.weights.path,
        speculative: launch.speculative,
        emitResidentIdentity: BenchWorkerAttach.emitsResidentIdentity(
            speculative: launch.speculative, environment: environment))
    do {
        try attach.relay(input: StdioTransport(), output: StdioTransport())
    } catch {
        fail(error)
    }
    exit(0)
}

// 3. The checkpoint. This is the first read under `--weights`.
let runnerType: any Runner.Type
do {
    runnerType = try launch.resolveRunner()
} catch {
    fail(error)
}

// 4. ONE load. For `resident` this is the window's ONLY load; for an
//    in-process `runtime-worker` it is this phase's.
let loadStarted = Date()
let runner: any Runner
do {
    runner = try await runnerType.load(
        launch.weights,
        options: RunnerLoadOptions(
            drafterDirectory: launch.drafter,
            kvBytesCapacity: launch.kvBytesCapacity,
            resources: launch.resources))
} catch {
    fail(error)
}
let loadSeconds = Date().timeIntervalSince(loadStarted)

// 4a. The load summary, BEFORE the hello and on stderr only — stdout is the
//     wire and stays byte-identical whether or not this is on. Emitted here
//     so it is present even when the first verb is what fails, which is the
//     case that left a box with nothing to read.
if RunnerLoadSummary.isEnabled(flag: launch.verbose, environment: environment) {
    var summary = runner.loadSummary(
        weights: launch.weights,
        options: RunnerLoadOptions(
            drafterDirectory: launch.drafter,
            kvBytesCapacity: launch.kvBytesCapacity,
            resources: launch.resources))
    summary.add("mode", launch.subcommand.rawValue)
    summary.add("speculative_protocol", launch.speculative ? "v1.1" : "v1")
    summary.add("load", seconds: loadSeconds)
    summary.writeToStandardError()
}

// 5. Serve — or, for the diagnostic, compare and exit.
switch launch.subcommand {
case .diagParity:
    do {
        try BenchWorkerLegacyParity.run(
            runner: runner, golden: launch.golden!, steps: launch.steps,
            output: FileHandle.standardOutput)
    } catch {
        fail(error)
    }
    BenchWorkerTeardown.synchronizeMLX()
    exit(0)
case .resident:
    // One process, one window. The socket is bound AFTER the load, so a
    // worker that connects successfully knows the weights are already up —
    // there is no window where an attach races the load.
    guard let socketPath = launch.socket else { fail(WorkerLaunchError.missingSocket) }
    let listener: BenchWorkerSocketListener
    do {
        listener = try BenchWorkerSocketListener(path: socketPath)
    } catch {
        fail(error)
    }
    let resident = BenchWorkerResident(
        runner: runner,
        listener: listener,
        weightsPath: launch.weights.path,
        trusted: launch.trusted,
        speculative: launch.speculative,
        build: BenchBuildRevision.value,
        device: probeDevice(),
        kvBytesCapacity: launch.kvBytesCapacity,
        pidfilePath: launch.pidfile)

    // The resident must not outlive its window: a stale one holds the
    // artifact's memory and blocks the box. The handlers are installed from
    // MLXRunners, NOT from a closure written here — top-level code in
    // main.swift is @MainActor-isolated, so a handler written inline runs
    // the executor-isolation check on the dispatch-source thread and traps
    // before any teardown happens. See `BenchWorkerResidentTermination`.
    residentSignalSources = BenchWorkerResidentTermination.install(for: resident)

    // The signal handler only ASKS. The accept loop returns, and the owner —
    // this thread — performs the teardown, so the socket and the pidfile are
    // gone before the process exits.
    //
    // Warm the resident before it serves: the first phase's prefill must not
    // be a cold or mid-ramp pass (see BenchWorkerResidentWarm).
    if BenchWorkerResidentWarm.isEnabled() {
        do {
            let warm = try BenchWorkerResidentWarm.run(runner: runner, memory: MLXMemoryReporter())
            FileHandle.standardError.write(
                Data(
                    String(
                        format: "bench-worker: resident warm pass: prefill %d tokens in %.3f s, %d decode steps in %.3f s (%d forwards); allocator drained\n",
                        warm.promptLength, warm.prefillSeconds, warm.decodeSteps, warm.decodeSeconds, warm.forwards
                    ).utf8))
        } catch {
            fail(error)
        }
    }
    resident.serve()
    resident.shutDown()
    // Commit and drain any Metal work a faulted phase left open, BEFORE the
    // static destructors run. See `BenchWorkerTeardown`.
    BenchWorkerTeardown.synchronizeMLX()
    exit(0)

case .runtimeWorker:
    let server = BenchWorkerServer(
        runner: runner,
        transport: StdioTransport(),
        trusted: launch.trusted,
        speculative: launch.speculative,
        // Stamped by the BenchRevisionStamp prebuild plugin: the revision
        // that PRODUCED this binary, not whatever the checkout moved to
        // afterwards.
        build: BenchBuildRevision.value,
        device: probeDevice(),
        kvBytesCapacity: launch.kvBytesCapacity)
    await server.run()
    BenchWorkerTeardown.synchronizeMLX()
}
