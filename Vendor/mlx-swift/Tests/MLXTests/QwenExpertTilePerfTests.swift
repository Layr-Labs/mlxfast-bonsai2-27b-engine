// Copyright © 2026 Eigen Labs.
//
// Routed-prefill unit microbenchmark for the Qwen 3.5/3.6 E=256 expert-tile
// route (notes/qwen36-m4-benchmarks.md methodology: warmup, then median /
// p10-p90 over >= 15 timed iterations at T=512 top-8, i.e. 4096 sorted
// assignments).
//
// The expert-slice route is chosen once per process at Device construction
// (MLX_GATHER_QMM_EXPERT_SLICES), so the A/B is two invocations:
//
//   MLX_EXPERT_TILES_PERF=1 MLX_GATHER_QMM_EXPERT_SLICES=0 \
//     swift test --filter QwenExpertTilePerfTests   # legacy route
//   MLX_EXPERT_TILES_PERF=1 MLX_GATHER_QMM_EXPERT_SLICES=1 \
//     swift test --filter QwenExpertTilePerfTests   # tile route
//
// Requires a source-matched mlx.metallib colocated with the test binary
// (scripts/fetch-metallib.sh <path>). Skips unless MLX_EXPERT_TILES_PERF=1.

import Foundation
import MLX
import XCTest

#if canImport(Metal)

final class QwenExpertTilePerfTests: XCTestCase {
    override class func setUp() {
        setDefaultDevice()
    }

    func testQwenRoutedPrefillUnitTimings() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLX_EXPERT_TILES_PERF"] == "1",
            "perf run is explicit opt-in")

        let expertCount = 256
        let groupSize = 64
        let bits = 4
        let iterations = 25
        let warmup = 5

        // (name, K, N, assignments): fused gate_up and down at T=512 and
        // T=1024 chunked prefill; split gate/up at T=512 for the unfused
        // comparison point.
        let cases: [(name: String, k: Int, n: Int, m: Int)] = [
            ("gate_up_fused T512", 2048, 1024, 4096),
            ("gate_split T512", 2048, 512, 4096),
            ("down T512", 512, 2048, 4096),
            ("gate_up_fused T1024", 2048, 1024, 8192),
            ("down T1024", 512, 2048, 8192),
        ]

        // The device owns the route decision (0/1/trust all parse there), so
        // derive the A/B label from what it actually enabled rather than
        // re-parsing MLX_GATHER_QMM_EXPERT_SLICES here.
        let diagnostics = GPU.gemma4ExpertQMMDiagnostics()
        var header =
            "[qwen-expert-tile-perf] route=" + (diagnostics.requested ? "tile" : "legacy")
        header += " aot=\(diagnostics.aotAvailable) nax=\(diagnostics.naxAvailable)\n"
        FileHandle.standardError.write(Data(header.utf8))

        for benchCase in cases {
            let weights = MLXRandom.normal(
                [expertCount, benchCase.n, benchCase.k]
            ).asType(.bfloat16)
            let (quantizedWeights, scales, biases) = quantized(
                weights, groupSize: groupSize, bits: bits, mode: .affine)
            // Top-8 routing over 256 experts: near-uniform sorted histogram.
            let perExpert = benchCase.m / expertCount
            let expertIndices: [Int32] = (0 ..< expertCount).flatMap {
                Array(repeating: Int32($0), count: perExpert)
            }
            let rhsIndices = MLXArray(expertIndices).asType(.uint32)
            let x = MLXRandom.normal([benchCase.m, 1, benchCase.k]).asType(.bfloat16)
            eval(quantizedWeights, scales, biases, rhsIndices, x)

            func run() -> MLXArray {
                gatherQuantizedMM(
                    x,
                    quantizedWeights,
                    scales: scales,
                    biases: biases,
                    rhsIndices: rhsIndices,
                    transpose: true,
                    groupSize: groupSize,
                    bits: bits,
                    mode: .affine,
                    sortedIndices: true
                )
            }
            for _ in 0 ..< warmup { eval(run()) }

            var samples: [Double] = []
            samples.reserveCapacity(iterations)
            GPU.clearAndArmGemma4ExpertQMMDiagnostics()
            for _ in 0 ..< iterations {
                let start = DispatchTime.now().uptimeNanoseconds
                eval(run())
                let end = DispatchTime.now().uptimeNanoseconds
                samples.append(Double(end - start) / 1e6)
            }
            let routeCounters = GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics()
            samples.sort()
            let median = samples[samples.count / 2]
            let p10 = samples[samples.count / 10]
            let p90 = samples[(samples.count * 9) / 10]
            // Weight bytes actually streamed per call: M/E-weighted expert
            // rows are all touched (uniform histogram touches all E experts).
            // Per output row: k/2 packed W4 bytes plus one BF16 scale and one
            // BF16 bias per group of 64, i.e. (k / groupSize) * 4 bytes.
            let weightBytes =
                expertCount * benchCase.n
                * (benchCase.k / 2 + benchCase.k / groupSize * 4)
            let gbps = Double(weightBytes) / (median / 1e3) / 1e9
            var line = "[qwen-expert-tile-perf] \(benchCase.name): "
            line += "median " + String(format: "%.4f", median) + " ms "
            line += "p10 " + String(format: "%.4f", p10) + " "
            line += "p90 " + String(format: "%.4f", p90) + " "
            line += "(~" + String(format: "%.0f", gbps) + " GB/s wt) "
            line += "hits=\(routeCounters.hits) fallbacks=\(routeCounters.fallbacks) "
            line += "n=\(iterations)\n"
            FileHandle.standardError.write(Data(line.utf8))
            Memory.clearCache()
        }
    }
}
#endif
