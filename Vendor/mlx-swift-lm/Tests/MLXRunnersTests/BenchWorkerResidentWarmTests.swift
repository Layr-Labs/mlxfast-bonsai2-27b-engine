import XCTest

@testable import MLXRunners

final class BenchWorkerResidentWarmTests: XCTestCase {
    private final class CountingMemory: WorkerMemoryReporter, @unchecked Sendable {
        var drains = 0
        func preDrainSnapshot() -> (active: Int, cache: Int, peak: Int)? { nil }
        func drain() { drains += 1 }
        func cacheMemoryAfterDrain() -> Int? { 0 }
        func peakRAMGB() -> Double? { nil }
    }

    func testWarmPassRunsOnePrefillThenTheDecodeStepsAndDrains() throws {
        let runner = MockRunner()
        let memory = CountingMemory()
        let report = try BenchWorkerResidentWarm.run(
            runner: runner, memory: memory, promptLength: 16, decodeSteps: 3)
        XCTAssertEqual(report.promptLength, 16)
        XCTAssertEqual(report.decodeSteps, 3)
        XCTAssertEqual(report.forwards, 4, "one prefill forward plus one forward per decode step")
        XCTAssertEqual(memory.drains, 1, "the allocator is drained once, after the pass")
        XCTAssertGreaterThanOrEqual(report.prefillSeconds, 0)
        XCTAssertGreaterThanOrEqual(report.decodeSeconds, 0)
    }

    func testPromptIsDeterministicAndAvoidsSpecialIds() {
        let a = BenchWorkerResidentWarm.prompt(length: 1024)
        let b = BenchWorkerResidentWarm.prompt(length: 1024)
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.count, 1024)
        XCTAssertTrue(a.allSatisfy { $0 >= 100 && $0 < 20_100 })
    }

    func testEnvironmentSwitchDefaultsOnAndHonorsOptOut() {
        XCTAssertTrue(BenchWorkerResidentWarm.isEnabled([:]))
        XCTAssertTrue(BenchWorkerResidentWarm.isEnabled(["BENCH_WORKER_RESIDENT_WARM": "1"]))
        XCTAssertFalse(BenchWorkerResidentWarm.isEnabled(["BENCH_WORKER_RESIDENT_WARM": "0"]))
        XCTAssertFalse(BenchWorkerResidentWarm.isEnabled(["BENCH_WORKER_RESIDENT_WARM": "off"]))
    }
}
