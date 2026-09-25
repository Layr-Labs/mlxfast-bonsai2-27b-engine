// The phase-close drain SETTLES before it reports.
//
// `phase_diagnostics` drains the MLX free-buffer cache and reports
// `cache_memory`; the parent (benchd) fails the run closed unless that is
// exactly 0. A free that lands after `Memory.clearCache()` — a step's last
// `asyncEval` temporaries, the engine loop's teardown on its own queue, a
// Metal command buffer's completion handler — puts bytes back in the cache
// milliseconds later. Four drains issued back to back all ran before such a
// straggler and read the same residual. On 2026-09-25 six ranked runs died
// this way with 240 KiB to 3.7 MiB left, every one on the candidate's
// warm-up leg.
//
// These tests drive the real server through the wire with a memory reporter
// that models a straggler: non-zero on the first N reads after a drain, zero
// after. They pin three things: a straggler that outlives the old budget now
// settles and the wire says 0; a cache that never drains is still reported
// non-zero (the parent's fail-closed rule is untouched); a clean drain costs
// one read and no pause.

import Foundation
import Testing

@testable import MLXRunners

@Suite("phase_diagnostics drain settles before it reports")
struct BenchWorkerPhaseDrainTests {
    /// A straggler: the first `stragglerReads` reads after a drain see
    /// bytes, every read after that sees 0.
    private final class StragglerMemory: WorkerMemoryReporter, @unchecked Sendable {
        static let residual = 327_680

        let stragglerReads: Int
        private(set) var drains = 0
        private(set) var reads = 0

        init(stragglerReads: Int) { self.stragglerReads = stragglerReads }

        func preDrainSnapshot() -> (active: Int, cache: Int, peak: Int)? { (1, 2, 3) }
        func drain() { drains += 1 }
        func cacheMemoryAfterDrain() -> Int? {
            reads += 1
            return reads <= stragglerReads ? Self.residual : 0
        }
        func peakRAMGB() -> Double? { 1 }
    }

    private func phaseDiagnostics(memory: StragglerMemory) async throws -> WorkerResponse {
        let transport = ScriptedTransport(lines: ["{\"id\":7,\"kind\":\"phase_diagnostics\"}"])
        let server = BenchWorkerServer(
            runner: MockRunner(),
            transport: transport,
            trusted: false,
            speculative: true,
            build: "fixture",
            device: "mock",
            kvBytesCapacity: 1 << 20,
            maxDecodeTokens: 64,
            memory: memory,
            nonce: "fixturenonce")
        await server.run()
        // written[0] is the hello.
        return try JSONDecoder().decode(WorkerResponse.self, from: Data(transport.written[1].utf8))
    }

    @Test("a straggler that outlives four drains settles, and the wire reports 0")
    func stragglerSettles() async throws {
        let memory = StragglerMemory(stragglerReads: 6)
        let response = try await phaseDiagnostics(memory: memory)
        #expect(response.ok)
        #expect(response.cacheMemory == 0)
        #expect(memory.drains == 7, "one drain per read until the first zero")
        #expect(memory.reads == 7, "the wire carries the settle loop's last read, not a fresh one")
        #expect(response.mlxCacheMemoryBytes == 2, "the pre-drain snapshot is untouched")
    }

    @Test("a cache that never drains is still reported non-zero")
    func leakStaysVisible() async throws {
        let memory = StragglerMemory(stragglerReads: .max)
        let response = try await phaseDiagnostics(memory: memory)
        #expect(response.ok)
        #expect(response.cacheMemory == StragglerMemory.residual)
        #expect(memory.drains == BenchWorkerServer.drainSettleAttempts, "the budget is bounded")
    }

    @Test("a clean drain settles on the first read")
    func cleanDrainIsOneRead() async throws {
        let memory = StragglerMemory(stragglerReads: 0)
        let response = try await phaseDiagnostics(memory: memory)
        #expect(response.cacheMemory == 0)
        #expect(memory.drains == 1)
        #expect(memory.reads == 1)
    }
}
