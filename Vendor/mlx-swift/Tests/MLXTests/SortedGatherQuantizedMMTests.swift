// Copyright © 2026 Apple Inc.
//
// Run the HIT-path tests with: MLX_GATHER_QMM_EXPERT_SLICES=1 swift test --filter SortedGatherQuantizedMMTests
// They skip when the flag or AOT symbols are absent; every other test runs
// under plain `swift test --filter SortedGatherQuantizedMMTests`.

import Foundation
import MLX
import XCTest

#if canImport(Metal)

final class SortedGatherQuantizedMMTests: XCTestCase {
    private let bits = 4
    private let expertCount = 8
    private let groupSize = 64
    private let inputDimensions = 64
    private let outputDimensions = 64

    override class func setUp() {
        setDefaultDevice()
    }

    func testGemma4ExpertQMMDiagnosticsReflectProcessConfigurationAndArmState() {
        let enabledValues = ["1", "true", "on", "yes", "trust"]
        let requested = ProcessInfo.processInfo.environment["MLX_GATHER_QMM_EXPERT_SLICES"]?
            .lowercased()

        _ = GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics()
        GPU.resetGemma4ExpertQMMDiagnostics()
        let unarmed = GPU.gemma4ExpertQMMDiagnostics()
        XCTAssertEqual(unarmed.requested, enabledValues.contains(requested ?? ""))
        XCTAssertFalse(unarmed.armed)
        assertCounterInvariant(unarmed)
        XCTAssertEqual(unarmed.attempts, 0)

        GPU.clearAndArmGemma4ExpertQMMDiagnostics()
        let armed = GPU.gemma4ExpertQMMDiagnostics()
        XCTAssertTrue(armed.armed)
        XCTAssertEqual(armed.requested, unarmed.requested)
        XCTAssertEqual(armed.aotAvailable, unarmed.aotAvailable)
        XCTAssertEqual(armed.naxAvailable, unarmed.naxAvailable)
        XCTAssertEqual(armed.attempts, 0)

        let measured = GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics()
        XCTAssertTrue(measured.armed)
        XCTAssertEqual(measured.attempts, 0)
        XCTAssertFalse(GPU.gemma4ExpertQMMDiagnostics().armed)
    }

    /// Activates the down-projection specialization exactly: E=128, K=704,
    /// N=2816, affine 4-bit/group-64, bfloat16, and each allowlisted prefill
    /// assignment count. The patterns cover aligned and unaligned boundaries,
    /// empty experts, 16/17-row tails, 127 one-row experts, and heavy skew.
    func testGemmaSortedAffineGatherQuantizedMMExpertSlices() throws {
        try requireExpertSlicesEnabled()
        let gemmaExpertCount = 128
        let gemmaInputDimensions = 704
        let gemmaOutputDimensions = 2816
        let fixture = gemmaAffineFixture(
            expertCount: gemmaExpertCount,
            inputDimensions: gemmaInputDimensions,
            outputDimensions: gemmaOutputDimensions
        )

        var unaligned4096 = Array(repeating: 32, count: gemmaExpertCount)
        unaligned4096.replaceSubrange(0 ..< 4, with: [3, 4, 5, 116])

        var explicitTails4096 = Array(repeating: 32, count: gemmaExpertCount)
        explicitTails4096.replaceSubrange(0 ..< 4, with: [1, 16, 17, 94])

        var emptyExperts8192 = Array(repeating: 0, count: gemmaExpertCount)
        emptyExperts8192[0] = 5
        emptyExperts8192[2] = 2
        emptyExperts8192[4] = 17
        emptyExperts8192[5] = 1
        emptyExperts8192[gemmaExpertCount - 1] = 8167

        let maximallyFragmented16384 =
            Array(repeating: 1, count: gemmaExpertCount - 1) + [16257]
        let cases: [(name: String, expertCounts: [Int])] = [
            ("M4096, BM32-aligned boundaries", Array(repeating: 32, count: gemmaExpertCount)),
            ("M4096, BM32-unaligned multi-boundary tiles", unaligned4096),
            ("M4096, explicit 1-row and BM16/BM32 tail boundary", explicitTails4096),
            ("M8192, empty experts and heavy skew", emptyExperts8192),
            ("M16384, maximally fragmented expert boundaries", maximallyFragmented16384),
        ]

        for testCase in cases {
            let expertIndices = sortedExpertIndices(testCase.expertCounts)
            let rowCount = expertIndices.count
            XCTAssertTrue([4096, 8192, 16384].contains(rowCount))
            let rhsIndices = MLXArray(expertIndices).asType(.uint32)
            let x = ones([rowCount, 1, gemmaInputDimensions], dtype: .bfloat16)

            GPU.clearAndArmGemma4ExpertQMMDiagnostics()
            let actualSorted = gatherQuantizedMM(
                x,
                fixture.weights,
                scales: fixture.scales,
                biases: fixture.biases,
                rhsIndices: rhsIndices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .affine,
                sortedIndices: true
            )
            let expected = broadcast(
                take(fixture.expertOutputs, rhsIndices).reshaped([rowCount, 1, 1]),
                to: [rowCount, 1, gemmaOutputDimensions]
            )

            eval(actualSorted, expected)
            XCTAssertEqual(actualSorted.shape, expected.shape, testCase.name)
            XCTAssertTrue(
                actualSorted.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
                "sorted gather-QMM selected incorrect expert rows for \(testCase.name); "
                    + "max absolute error "
                    + "\((actualSorted - expected).abs().max().item(Float.self))"
            )
            assertExactGemmaRoute(
                GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics(), testCase.name)
            Memory.clearCache()
        }
    }

    /// Covers the fused gate/up projection and the BM=16 and BM=32
    /// expert-only bodies.
    func testGemmaGateUpSortedAffineGatherQuantizedMMExpertSlices() throws {
        try requireExpertSlicesEnabled()
        let gemmaExpertCount = 128
        let gemmaInputDimensions = 2816
        let gemmaOutputDimensions = 1408
        let assignmentCount = 4096
        let fixture = gemmaAffineFixture(
            expertCount: gemmaExpertCount,
            inputDimensions: gemmaInputDimensions,
            outputDimensions: gemmaOutputDimensions
        )

        var expertCounts = Array(repeating: 32, count: gemmaExpertCount)
        expertCounts.replaceSubrange(0 ..< 4, with: [3, 4, 5, 116])
        let expertIndices = sortedExpertIndices(expertCounts)
        XCTAssertEqual(expertIndices.count, assignmentCount)

        let rhsIndices = MLXArray(expertIndices).asType(.uint32)
        let x = ones([assignmentCount, 1, gemmaInputDimensions], dtype: .bfloat16)
        GPU.clearAndArmGemma4ExpertQMMDiagnostics()
        let actualSorted = gatherQuantizedMM(
            x,
            fixture.weights,
            scales: fixture.scales,
            biases: fixture.biases,
            rhsIndices: rhsIndices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: true
        )
        let expected = broadcast(
            take(fixture.expertOutputs, rhsIndices).reshaped([assignmentCount, 1, 1]),
            to: [assignmentCount, 1, gemmaOutputDimensions]
        )

        eval(actualSorted, expected)
        XCTAssertEqual(actualSorted.shape, expected.shape)
        XCTAssertTrue(
            actualSorted.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            "gate/up sorted gather-QMM selected incorrect expert rows; max absolute error "
                + "\((actualSorted - expected).abs().max().item(Float.self))"
        )
        assertExactGemmaRoute(
            GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics(),
            "gate/up M4096, BM16/BM32 boundaries")
        Memory.clearCache()
    }

    /// A full Gemma topology with a prefill-like outer route but a disallowed
    /// assignment count must stay on the legacy implementation.
    func testGemmaSelectorRejectsNonAllowlistedAssignmentCount() {
        let gemmaExpertCount = 128
        let rowCount = 512
        let gemmaInputDimensions = 704
        let gemmaOutputDimensions = 2816
        let fixture = gemmaAffineFixture(
            expertCount: gemmaExpertCount,
            inputDimensions: gemmaInputDimensions,
            outputDimensions: gemmaOutputDimensions
        )
        let expertCounts = Array(repeating: 4, count: gemmaExpertCount)
        let rhsIndices = MLXArray(sortedExpertIndices(expertCounts)).asType(.uint32)
        let x = ones([rowCount, 1, gemmaInputDimensions], dtype: .bfloat16)

        GPU.clearAndArmGemma4ExpertQMMDiagnostics()
        let actual = gatherQuantizedMM(
            x,
            fixture.weights,
            scales: fixture.scales,
            biases: fixture.biases,
            rhsIndices: rhsIndices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: true
        )
        let expected = broadcast(
            take(fixture.expertOutputs, rhsIndices).reshaped([rowCount, 1, 1]),
            to: [rowCount, 1, gemmaOutputDimensions]
        )

        eval(actual, expected)
        XCTAssertTrue(actual.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        let diagnostics = GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics()
        XCTAssertTrue(diagnostics.armed)
        assertCounterInvariant(diagnostics)
        if !diagnostics.requested {
            XCTAssertEqual(diagnostics.attempts, 0)
        } else if diagnostics.naxAvailable {
            XCTAssertEqual(diagnostics.fallbackNAX, 1)
            XCTAssertEqual(diagnostics.hits, 0)
        } else {
            XCTAssertEqual(diagnostics.fallbackAssignmentCount, 1)
            XCTAssertEqual(diagnostics.hits, 0)
        }
        Memory.clearCache()
    }

    /// Activates the Qwen 3.5/3.6 E=256 specializations exactly: fused
    /// gate_up [256,1024,2048], split gate/up [256,512,2048], and down
    /// [256,2048,512] at affine 4-bit/group-64 bfloat16 across the
    /// allowlisted prefill assignment counts (T x top-8, T in {512, 1024,
    /// 2048}). Patterns cover aligned tiles, multi-boundary unaligned tiles,
    /// empty experts with heavy skew, and maximal fragmentation (255 one-row
    /// experts).
    func testQwenSortedAffineGatherQuantizedMMExpertSlices() throws {
        try requireExpertSlicesEnabled()
        let qwenExpertCount = 256
        let geometries: [(name: String, k: Int, n: Int)] = [
            ("fused gate_up K2048 N1024", 2048, 1024),
            ("split gate/up K2048 N512", 2048, 512),
            ("down K512 N2048", 512, 2048),
        ]

        var unaligned4096 = Array(repeating: 16, count: qwenExpertCount)
        unaligned4096.replaceSubrange(0 ..< 4, with: [3, 4, 5, 52])

        var emptyExperts8192 = Array(repeating: 0, count: qwenExpertCount)
        emptyExperts8192[0] = 5
        emptyExperts8192[2] = 2
        emptyExperts8192[4] = 17
        emptyExperts8192[5] = 1
        emptyExperts8192[qwenExpertCount - 1] = 8167

        let maximallyFragmented16384 =
            Array(repeating: 1, count: qwenExpertCount - 1) + [16129]
        let cases: [(name: String, expertCounts: [Int])] = [
            ("M4096, BM16 top-8 uniform", Array(repeating: 16, count: qwenExpertCount)),
            ("M4096, multi-boundary unaligned tiles", unaligned4096),
            ("M8192, empty experts and heavy skew", emptyExperts8192),
            ("M16384, maximally fragmented expert boundaries", maximallyFragmented16384),
        ]

        for geometry in geometries {
            let fixture = gemmaAffineFixture(
                expertCount: qwenExpertCount,
                inputDimensions: geometry.k,
                outputDimensions: geometry.n
            )
            for testCase in cases {
                let expertIndices = sortedExpertIndices(testCase.expertCounts)
                let rowCount = expertIndices.count
                XCTAssertTrue([4096, 8192, 16384].contains(rowCount))
                let rhsIndices = MLXArray(expertIndices).asType(.uint32)
                let x = ones([rowCount, 1, geometry.k], dtype: .bfloat16)

                GPU.clearAndArmGemma4ExpertQMMDiagnostics()
                let actualSorted = gatherQuantizedMM(
                    x,
                    fixture.weights,
                    scales: fixture.scales,
                    biases: fixture.biases,
                    rhsIndices: rhsIndices,
                    transpose: true,
                    groupSize: groupSize,
                    bits: bits,
                    mode: .affine,
                    sortedIndices: true
                )
                let expected = broadcast(
                    take(fixture.expertOutputs, rhsIndices).reshaped([rowCount, 1, 1]),
                    to: [rowCount, 1, geometry.n]
                )

                eval(actualSorted, expected)
                let context = "\(geometry.name): \(testCase.name)"
                XCTAssertEqual(actualSorted.shape, expected.shape, context)
                XCTAssertTrue(
                    actualSorted.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
                    "sorted gather-QMM selected incorrect expert rows for \(context); "
                        + "max absolute error "
                        + "\((actualSorted - expected).abs().max().item(Float.self))"
                )
                assertExactGemmaRoute(
                    GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics(), context)
                Memory.clearCache()
            }
        }
    }

    /// Random-tensor equivalence at the exact Qwen shapes: the tile route's
    /// output must match both the legacy unsorted gather-QMM and the
    /// dequantized gatherMM reference within FP32-accumulation tolerances.
    func testQwenSortedGatherQuantizedMMMatchesLegacyOnRandomTensors() throws {
        try requireExpertSlicesEnabled()
        let qwenExpertCount = 256
        let geometries: [(name: String, k: Int, n: Int)] = [
            ("fused gate_up K2048 N1024", 2048, 1024),
            ("down K512 N2048", 512, 2048),
        ]
        for geometry in geometries {
            let weights = deterministicValues(
                count: qwenExpertCount * geometry.n * geometry.k,
                multiplier: 17,
                modulus: 127
            ).reshaped([qwenExpertCount, geometry.n, geometry.k]).asType(.bfloat16)
            let (quantizedWeights, scales, biases) = quantized(
                weights, groupSize: groupSize, bits: bits, mode: .affine)
            let dequantizedWeights = dequantized(
                quantizedWeights,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: .affine,
                dtype: .bfloat16
            ).swappedAxes(-1, -2)

            var expertCounts = Array(repeating: 16, count: qwenExpertCount)
            expertCounts.replaceSubrange(0 ..< 4, with: [1, 15, 33, 15])
            let expertIndices = sortedExpertIndices(expertCounts)
            let rowCount = expertIndices.count
            XCTAssertEqual(rowCount, 4096)
            let rhsIndices = MLXArray(expertIndices).asType(.uint32)
            let x = deterministicValues(
                count: rowCount * geometry.k,
                multiplier: 29,
                modulus: 113
            ).reshaped([rowCount, 1, geometry.k]).asType(.bfloat16)

            GPU.clearAndArmGemma4ExpertQMMDiagnostics()
            let actualSorted = gatherQuantizedMM(
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
            // Force the routed gather to actually execute while diagnostics
            // are armed: MLX is lazy, so the snapshot must come after eval.
            eval(actualSorted)
            let route = GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics()
            let expectedQuantized = gatherQuantizedMM(
                x,
                quantizedWeights,
                scales: scales,
                biases: biases,
                rhsIndices: rhsIndices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .affine,
                sortedIndices: false
            )
            let expectedDequantized = gatherMM(
                x, dequantizedWeights, rhsIndices: rhsIndices, sortedIndices: false)

            eval(actualSorted, expectedQuantized, expectedDequantized)
            assertExactGemmaRoute(route, geometry.name)
            // Outputs are bf16 with |y| up to ~15 for K=2048 dots of these
            // deterministic inputs; one bf16 ulp at that magnitude is
            // 2^-8 * 16 = 0.0625. Different (but both FP32-accumulated)
            // tile orders legitimately differ by a few output ulps, so the
            // absolute tolerance is ulp-scaled from the reference itself;
            // a wrong expert row would err at the full |y| scale (~10+).
            let referenceScale = expectedQuantized.abs().max().item(Float.self)
            let ulpTolerance = max(1e-2, 4 * referenceScale / 256)
            XCTAssertTrue(
                actualSorted.allClose(
                    expectedQuantized, rtol: 2e-2, atol: Double(ulpTolerance)
                ).item(Bool.self),
                "tile route and legacy gather-QMM differ for \(geometry.name); "
                    + "max absolute error "
                    + "\((actualSorted - expectedQuantized).abs().max().item(Float.self))"
            )
            XCTAssertTrue(
                actualSorted.allClose(
                    expectedDequantized, rtol: 2e-2, atol: Double(ulpTolerance)
                ).item(Bool.self),
                "tile route drifted from dequantized reference for \(geometry.name); "
                    + "max absolute error "
                    + "\((actualSorted - expectedDequantized).abs().max().item(Float.self))"
            )
            Memory.clearCache()
        }
    }

    /// Mis-sorted indices under the sorted contract must retract on-device
    /// and re-route to the order-agnostic legacy kernel with a correct
    /// result.
    func testQwenSortednessViolationRetractsToLegacyRoute() throws {
        try requireExpertSlicesEnabled()
        let diagnosticsProbe = GPU.gemma4ExpertQMMDiagnostics()
        try XCTSkipUnless(
            diagnosticsProbe.aotAvailable && !diagnosticsProbe.naxAvailable,
            "retract coverage requires the AOT tile kernels on a non-NAX device")
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["MLX_GATHER_QMM_EXPERT_SLICES"]?
                .lowercased() == "trust",
            "trust mode skips the sortedness-retract readback by design")
        let qwenExpertCount = 256
        let k = 512
        let n = 2048
        let fixture = gemmaAffineFixture(
            expertCount: qwenExpertCount,
            inputDimensions: k,
            outputDimensions: n
        )
        var expertIndices = sortedExpertIndices(
            Array(repeating: 16, count: qwenExpertCount))
        // One inversion in the middle breaks global monotonicity.
        expertIndices.swapAt(2048, 2049 + 16)
        let rowCount = expertIndices.count
        let rhsIndices = MLXArray(expertIndices).asType(.uint32)
        let x = ones([rowCount, 1, k], dtype: .bfloat16)

        GPU.clearAndArmGemma4ExpertQMMDiagnostics()
        let actual = gatherQuantizedMM(
            x,
            fixture.weights,
            scales: fixture.scales,
            biases: fixture.biases,
            rhsIndices: rhsIndices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: true
        )
        let expected = broadcast(
            take(fixture.expertOutputs, rhsIndices).reshaped([rowCount, 1, 1]),
            to: [rowCount, 1, n]
        )
        eval(actual, expected)
        XCTAssertTrue(
            actual.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            "retracted call must still produce the legacy-path result; "
                + "max absolute error "
                + "\((actual - expected).abs().max().item(Float.self))"
        )
        let diagnostics = GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics()
        XCTAssertEqual(diagnostics.attempts, 1)
        XCTAssertEqual(diagnostics.hits, 0)
        XCTAssertEqual(diagnostics.fallbackSortednessRetracted, 1)
        Memory.clearCache()
    }

    /// T=128 chunks (1024 assignments) intentionally stay on the legacy
    /// implementation for Qwen shapes.
    func testQwenSelectorRejectsNonAllowlistedAssignmentCount() {
        let qwenExpertCount = 256
        let rowCount = 1024
        let k = 512
        let n = 2048
        let fixture = gemmaAffineFixture(
            expertCount: qwenExpertCount,
            inputDimensions: k,
            outputDimensions: n
        )
        let expertCounts = Array(repeating: 4, count: qwenExpertCount)
        let rhsIndices = MLXArray(sortedExpertIndices(expertCounts)).asType(.uint32)
        let x = ones([rowCount, 1, k], dtype: .bfloat16)

        GPU.clearAndArmGemma4ExpertQMMDiagnostics()
        let actual = gatherQuantizedMM(
            x,
            fixture.weights,
            scales: fixture.scales,
            biases: fixture.biases,
            rhsIndices: rhsIndices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: true
        )
        let expected = broadcast(
            take(fixture.expertOutputs, rhsIndices).reshaped([rowCount, 1, 1]),
            to: [rowCount, 1, n]
        )

        eval(actual, expected)
        XCTAssertTrue(actual.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        let diagnostics = GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics()
        XCTAssertTrue(diagnostics.armed)
        assertCounterInvariant(diagnostics)
        if !diagnostics.requested {
            XCTAssertEqual(diagnostics.attempts, 0)
        } else if diagnostics.naxAvailable {
            XCTAssertEqual(diagnostics.fallbackNAX, 1)
            XCTAssertEqual(diagnostics.hits, 0)
        } else {
            XCTAssertEqual(diagnostics.fallbackAssignmentCount, 1)
            XCTAssertEqual(diagnostics.hits, 0)
        }
        Memory.clearCache()
    }

    /// Protects the established fallback BM=16 schedule on compact shapes.
    func testSortedAffineGatherQuantizedMMFallbackExpertBoundaries() {
        let weights = deterministicValues(
            count: expertCount * outputDimensions * inputDimensions,
            multiplier: 17,
            modulus: 127
        ).reshaped([expertCount, outputDimensions, inputDimensions]).asType(.bfloat16)
        let (quantizedWeights, scales, biases) = quantized(
            weights,
            groupSize: groupSize,
            bits: bits,
            mode: .affine
        )
        let dequantizedWeights = dequantized(
            quantizedWeights,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            dtype: .bfloat16
        ).swappedAxes(-1, -2)

        let cases: [(name: String, expertCounts: [Int])] = [
            ("all experts, aligned, divisible M", Array(repeating: 16, count: expertCount)),
            ("all experts, unaligned, multiple boundaries, divisible M", [3, 4, 5, 4, 20, 11, 9, 8]),
            ("empty experts, unaligned, multiple boundaries, partial M", [5, 0, 2, 0, 17, 1, 0, 12]),
        ]

        for testCase in cases {
            let expertIndices = sortedExpertIndices(testCase.expertCounts)
            let rowCount = expertIndices.count
            let x = deterministicValues(
                count: rowCount * inputDimensions,
                multiplier: 29,
                modulus: 113
            ).reshaped([rowCount, 1, inputDimensions]).asType(.bfloat16)
            let rhsIndices = MLXArray(expertIndices).asType(.uint32)

            _ = GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics()
            GPU.resetGemma4ExpertQMMDiagnostics()
            let expectedQuantized = gatherQuantizedMM(
                x,
                quantizedWeights,
                scales: scales,
                biases: biases,
                rhsIndices: rhsIndices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .affine,
                sortedIndices: false
            )
            eval(expectedQuantized)
            let untracked = GPU.gemma4ExpertQMMDiagnostics()
            XCTAssertFalse(untracked.armed)
            XCTAssertEqual(
                untracked.attempts, 0,
                "unarmed fallback work must not synchronize route counters")
            GPU.clearAndArmGemma4ExpertQMMDiagnostics()
            let actualSorted = gatherQuantizedMM(
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
            let expectedDequantized = gatherMM(
                x,
                dequantizedWeights,
                rhsIndices: rhsIndices,
                sortedIndices: false
            )

            eval(actualSorted, expectedDequantized)
            XCTAssertEqual(actualSorted.shape, [rowCount, 1, outputDimensions], testCase.name)
            XCTAssertTrue(
                actualSorted.allClose(expectedQuantized, rtol: 1e-2, atol: 1e-2).item(Bool.self),
                "sorted and unsorted gather-QMM differ for \(testCase.name); max absolute error "
                    + "\((actualSorted - expectedQuantized).abs().max().item(Float.self))"
            )
            XCTAssertTrue(
                actualSorted.allClose(expectedDequantized, rtol: 2e-2, atol: 2e-2).item(Bool.self),
                "sorted gather-QMM selected incorrect expert rows for \(testCase.name); "
                    + "max absolute error "
                    + "\((actualSorted - expectedDequantized).abs().max().item(Float.self))"
            )

            let diagnostics = GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics()
            XCTAssertTrue(diagnostics.armed)
            assertCounterInvariant(diagnostics)
            if !diagnostics.requested {
                XCTAssertEqual(diagnostics.attempts, 0)
            } else if diagnostics.naxAvailable {
                XCTAssertEqual(diagnostics.fallbackNAX, 1)
                XCTAssertEqual(diagnostics.hits, 0)
            } else {
                XCTAssertEqual(diagnostics.fallbackTopology, 1)
                XCTAssertEqual(diagnostics.hits, 0)
            }
        }
    }

    /// Ordinary, non-gather QMM must retain its established global kernel for
    /// aligned and tail row counts and must not touch expert-route counters.
    func testOrdinaryQuantizedMMPreservesGlobalQMM() {
        let fixture = gemmaAffineFixture(
            expertCount: 1,
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions
        )
        let packedInputDimensions = inputDimensions * bits / 32
        let weights = fixture.weights.reshaped([outputDimensions, packedInputDimensions])
        let scales = fixture.scales.reshaped([outputDimensions, inputDimensions / groupSize])
        let biases = fixture.biases.reshaped([outputDimensions, inputDimensions / groupSize])

        for rowCount in [16, 17, 32, 33] {
            let x = ones([rowCount, inputDimensions], dtype: .bfloat16)
            let expected = broadcast(
                fixture.expertOutputs.reshaped([1, 1]),
                to: [rowCount, outputDimensions]
            )

            GPU.clearAndArmGemma4ExpertQMMDiagnostics()
            let actual = quantizedMM(
                x,
                weights,
                scales: scales,
                biases: biases,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
            eval(actual, expected)
            XCTAssertEqual(actual.shape, expected.shape)
            XCTAssertTrue(
                actual.allClose(expected, rtol: 1e-3, atol: 1e-3).item(Bool.self),
                "ordinary QMM changed for row count \(rowCount); max absolute error "
                    + "\((actual - expected).abs().max().item(Float.self))"
            )

            let diagnostics = GPU.snapshotAndDisarmGemma4ExpertQMMDiagnostics()
            XCTAssertTrue(diagnostics.armed)
            XCTAssertEqual(diagnostics.attempts, 0)
            XCTAssertEqual(diagnostics.hits, 0)
            XCTAssertEqual(diagnostics.fallbacks, 0)
        }
    }

    /// Builds a compact lazy fixture whose eventual contiguous arrays retain
    /// the requested shapes. Every expert and quantization group is distinct,
    /// giving a closed-form reference independent of the legacy gather path.
    private func gemmaAffineFixture(
        expertCount: Int,
        inputDimensions: Int,
        outputDimensions: Int
    ) -> (weights: MLXArray, scales: MLXArray, biases: MLXArray, expertOutputs: MLXArray) {
        let packedInputDimensions = inputDimensions * bits / 32
        let scaleGroups = inputDimensions / groupSize
        let codes = (0 ..< expertCount).map { Float($0 % 15 + 1) }
        let packedCodes = (0 ..< expertCount).map {
            UInt32($0 % 15 + 1) * UInt32(0x1111_1111)
        }
        let expertScaleFactors = (0 ..< expertCount).map { Float($0 % 4 + 1) / 4 }
        let groupScaleFactors = (0 ..< scaleGroups).map { Float(1 << ($0 % 4)) / 8 }
        let expertBiases = (0 ..< expertCount).map { Float($0 % 8) / 16 - 0.25 }

        let weights = broadcast(
            MLXArray(packedCodes).reshaped([expertCount, 1, 1]),
            to: [expertCount, outputDimensions, packedInputDimensions]
        )
        let scales = broadcast(
            MLXArray(expertScaleFactors).reshaped([expertCount, 1, 1])
                * MLXArray(groupScaleFactors).reshaped([1, 1, scaleGroups]),
            to: [expertCount, outputDimensions, scaleGroups]
        ).asType(.bfloat16)
        let biases = broadcast(
            MLXArray(expertBiases).reshaped([expertCount, 1, 1]),
            to: [expertCount, outputDimensions, scaleGroups]
        ).asType(.bfloat16)
        let groupScaleSum = groupScaleFactors.reduce(0, +)
        let expertOutputs = MLXArray(
            (0 ..< expertCount).map { expert in
                Float(groupSize) * codes[expert] * expertScaleFactors[expert]
                    * groupScaleSum
                    + Float(inputDimensions) * expertBiases[expert]
            }
        ).asType(.bfloat16)
        return (weights, scales, biases, expertOutputs)
    }

    private func assertExactGemmaRoute(
        _ diagnostics: GPU.Gemma4ExpertQMMDiagnostics,
        _ context: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        assertCounterInvariant(diagnostics, file: file, line: line)
        XCTAssertTrue(diagnostics.requested, context, file: file, line: line)
        XCTAssertTrue(diagnostics.armed, context, file: file, line: line)
        XCTAssertEqual(diagnostics.attempts, 1, context, file: file, line: line)
        if diagnostics.naxAvailable {
            XCTAssertEqual(diagnostics.fallbackNAX, 1, context, file: file, line: line)
            XCTAssertEqual(diagnostics.hits, 0, context, file: file, line: line)
        } else {
            // requireExpertSlicesEnabled() skips hit-path suites when the AOT
            // symbols are absent, so anything short of a hit here is a bug —
            // never accept the metallib-unavailable fallback as success.
            XCTAssertTrue(diagnostics.aotAvailable, context, file: file, line: line)
            XCTAssertEqual(diagnostics.hits, 1, context, file: file, line: line)
            XCTAssertEqual(diagnostics.fallbacks, 0, context, file: file, line: line)
        }
    }

    private func assertCounterInvariant(
        _ diagnostics: GPU.Gemma4ExpertQMMDiagnostics,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        // The attempts == hits + fallbacks invariant is constructional: the
        // C++ snapshot derives `attempts` as exactly that sum
        // (Gemma4ExpertQMMCounterSnapshot.attempts()), and the C facade maps
        // it through verbatim, so there is no independent quantity left to
        // assert. The per-call-site attempts == 0/1 and per-counter
        // assertions carry the real signal.
    }

    private func requireExpertSlicesEnabled() throws {
        let value = ProcessInfo.processInfo.environment["MLX_GATHER_QMM_EXPERT_SLICES"]?
            .lowercased()
        try XCTSkipUnless(
            ["1", "true", "on", "yes", "trust"].contains(value ?? ""),
            "exact-shape coverage requires R1 enabled"
        )
        // The hit-path suites certify the AOT tile kernels; without a staged
        // source-matched mlx.metallib they would silently exercise only the
        // legacy implementation, so skip instead of accepting the fallback.
        try XCTSkipUnless(
            GPU.gemma4ExpertQMMDiagnostics().aotAvailable,
            "hit-path coverage requires the AOT tile kernels; stage a "
                + "source-matched mlx.metallib (scripts/fetch-metallib.sh)"
        )
    }

    private func deterministicValues(count: Int, multiplier: Int, modulus: Int) -> MLXArray {
        MLXArray(
            (0 ..< count).map { index in
                Float((index * multiplier) % modulus - modulus / 2) / Float(modulus)
            }
        )
    }

    private func sortedExpertIndices(_ expertCounts: [Int]) -> [Int32] {
        expertCounts.enumerated().flatMap { expert, count in
            Array(repeating: Int32(expert), count: count)
        }
    }
}
#endif
