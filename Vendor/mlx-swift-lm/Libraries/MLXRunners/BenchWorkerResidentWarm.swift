// Copyright © 2026 Eigen Labs.
//
// MLXRunners — the resident's boot warm pass.
//
// A fresh process pays its first prefill cold (Metal pipeline state, lazily
// compiled kernels, page-ins) and its second prefill still on the ramp: on
// the Qwen 3.8 125B ranked box the first three prefills of a process read
// 3.56, 0.87 and 0.635 ms/token, and every later one 0.635 ±0.2 %. benchd's
// official path warms ONE leg and then times the next, so the timed prefill
// landed mid-ramp (0.66–0.76 ms/token across boots) and the ±5 % prefill
// band refused runs at random. ds4 warms its serve at boot; this does the
// same for the resident: one fixed pass — a deterministic prompt through the
// stepper, then a few single-token forwards — before the socket serves its
// first connection. No loop, no settling criterion, nothing on the wire.

import Foundation

public enum BenchWorkerResidentWarm {
    /// Opt out (diagnostics only): `BENCH_WORKER_RESIDENT_WARM=0`.
    public static let environmentSwitch = "BENCH_WORKER_RESIDENT_WARM"
    public static let defaultPromptLength = 1024
    public static let defaultDecodeSteps = 8

    public struct Report: Sendable, Equatable {
        public let promptLength: Int
        public let decodeSteps: Int
        public let prefillSeconds: Double
        public let decodeSeconds: Double
        public let forwards: Int
    }

    public static func isEnabled(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let raw = environment[environmentSwitch]?.trimmingCharacters(in: .whitespaces).lowercased()
        return !["0", "false", "no", "off"].contains(raw ?? "")
    }

    /// A deterministic prompt of ordinary token ids (no specials): the warm
    /// pass exercises the kernels, not the model's opinion of the text.
    public static func prompt(length: Int) -> [Int] {
        (0 ..< length).map { 100 + ($0 &* 7919) % 20_000 }
    }

    /// One warm pass through the runner's own stepper, then the allocator is
    /// drained so the first served phase starts from the same memory state a
    /// cold resident would have given it.
    public static func run(
        runner: any Runner,
        memory: any WorkerMemoryReporter,
        promptLength: Int = defaultPromptLength,
        decodeSteps: Int = defaultDecodeSteps
    ) throws -> Report {
        let stepper = try runner.makeStepper()
        try stepper.begin()
        let start = DispatchTime.now().uptimeNanoseconds
        var output = try stepper.forward(prompt(length: promptLength))
        let afterPrefill = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< decodeSteps {
            output = try stepper.forward([output.argmax])
        }
        let end = DispatchTime.now().uptimeNanoseconds
        let forwards = stepper.forwards
        memory.drain()
        return Report(
            promptLength: promptLength,
            decodeSteps: decodeSteps,
            prefillSeconds: Double(afterPrefill - start) / 1e9,
            decodeSeconds: Double(end - afterPrefill) / 1e9,
            forwards: forwards)
    }
}
