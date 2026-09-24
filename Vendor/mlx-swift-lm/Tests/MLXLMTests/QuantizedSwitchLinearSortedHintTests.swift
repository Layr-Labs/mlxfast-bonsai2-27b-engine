import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

// THE SORTED-INDICES HINT IS SAFE ON INDEX-ALIGNED ROWS AND UNSAFE ON
// BROADCAST ONES, AND THE DIFFERENCE IS A STALE ROW COUNT.
//
// In the MLX this package pins, `GatherQMM::eval_gpu` computes `M = x.size()
// / K` from the array it was handed and passes that `M` into
// `gather_qmm_rhs`. `gather_qmm_rhs` broadcasts `x` up to one row per index
// when it is not already that shape, and never recomputes `M`; the dispatch
// grid and the kernel's row bound both keep the stale value. Only the first
// `x.size() / K` rows of the output are written. The rest keeps whatever the
// pool held, which is wrong from the first call, carries no NaN, and repeats
// exactly on reuse -- a stable wrong answer rather than noise.
//
// So the fault needs a broadcast. Measured here, on a 128-expert [704, 2816]
// stack at 4 bits and group 64 with 8192 sorted assignments, against the same
// gather computed densely on the dequantized weights:
//
//     nested x [1, 1024, 1, 1, 2816], indices [1, 1024, 8]   -> broadcast
//         hint off   max|d| = 2.0    hint on   max|d| = 251.0
//
//     flat x [8192, 1, 2816], indices [8192]                 -> aligned
//         hint off   max|d| = 2.5    hint on   max|d| = 1.0
//
// On aligned rows the hinted path is not merely safe, it is CLOSER to the
// truth, and it is the fast one: the hint is what selects the batched
// sorted-rhs kernel, measured here at 0.029 to 0.040 seconds per call
// against 0.104 to 0.120 unhinted, a factor of 2.6 to 4.1 across runs.
//
// THE NON-QUANTIZED GATHER IS NOT EXPOSED, and the reason is structural
// rather than lucky. `GatherMM::eval_gpu` derives its own shapes inside
// `gather_mm_rhs`; only `GatherQMM::eval_gpu` hands a precomputed row count
// across the broadcast. The control in the reproducer measures 1.0 between
// the hinted and unhinted non-quantized calls on the same broadcast rows,
// which is accumulation order on outputs of scale 50, not the 251 break.
//
// TWO THRESHOLDS, WHICH ARE NOT THE SAME THRESHOLD. `SwitchGLU` sorts its
// expert assignments once they number 64 or more and only then hints. The
// KERNEL only takes a different route when `M == 1 && B >= 16 && B / E >= 4`,
// where B is the assignment count and E the expert count. A test below 4 * E
// assignments never reaches the route at all, whatever the caller did; only
// the legs that cross it are load-bearing.
//
// WHY PRODUCTION IS NOT EXPOSED. Every caller in this package hints only
// after `gatherSort`, which returns exactly one row per index. Those rows are
// aligned, so no broadcast happens and the stale `M` is the true row count.
// `gatherSortFeedsTheGatherOneRowPerIndex` pins that property.
@Suite("QuantizedSwitchLinear sorted-hint", .serialized)
struct QuantizedSwitchLinearSortedHintTests {

    /// How far a quantized gather may sit from the same gather computed on the
    /// unquantized weights.
    ///
    /// This is a quantization-error budget, not a fudge. The two sides
    /// multiply the same numbers in a different order and one of them rounds
    /// to 4 bits per group, so they cannot agree bit for bit. What the budget
    /// has to exclude is a WRONG ANSWER, and the separation is two orders of
    /// magnitude: the failure this file exists for reads 251.
    ///
    /// This budget covers the small stacks the always-on legs build. The
    /// reproducer's stack is far deeper (K = 2816 against 512) and normally
    /// distributed rather than uniform, so its error is larger and it carries
    /// its own budget.
    static let quantizationTolerance: Float = 1.0

    /// The reproducer's geometry, at which the fault above was measured.
    ///
    /// DO NOT RETUNE THESE. The fault is not a simple function of the
    /// assignment count -- smaller counts and other call shapes read clean in
    /// the same probe -- so a geometry that no longer reproduces looks exactly
    /// like a repaired MLX. `theBroadcastGatherIsWrongWithTheSortedHint` pins
    /// them so that a change to them is a deliberate edit rather than a drift.
    enum Reproducer {
        static let experts = 128
        static let outputDims = 704
        static let inputDims = 2816
        static let groupSize = 64
        static let bits = 4
        static let tokens = 1024
        static let topK = 8
        static var assignments: Int { tokens * topK }

        /// Measured on this geometry: the unhinted gather reads 2.0 from dense
        /// truth on broadcast rows and 2.5 on aligned ones, and the hinted
        /// gather reads 1.0 on aligned rows. The break it must exclude is 251.
        static let tolerance: Float = 4.0
    }

    private static var reproEnabled: Bool {
        ProcessInfo.processInfo.environment["MLX_RUN_GATHER_QMM_REPRO"] == "1"
    }

    private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func nanCount(_ x: MLXArray) -> Int {
        let f = x.asType(.float32)
        return Int(
            MLX.which(f .!= f, MLXArray(Int32(1)), MLXArray(Int32(0))).sum().item(Int32.self))
    }

    /// Deterministic expert ids in `0 ..< experts`, independent of MLX's
    /// generator so that the host can sort them.
    private func expertIds(_ count: Int, experts: Int, seed: UInt64) -> [Int32] {
        var state = seed
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int32((state >> 33) % UInt64(experts))
        }
    }

    /// Run `body` `runs` times and report whether every call agreed bit for
    /// bit with the first. `Float.infinity` means a NaN appeared.
    private func repeatRun(
        runs: Int = 6, _ body: () -> MLXArray
    ) -> (identical: Bool, firstDivergence: Int?, firstNaN: Int?, worst: Float) {
        var reference: MLXArray?
        var firstDivergence: Int?
        var firstNaN: Int?
        var worst: Float = 0
        for run in 0 ..< runs {
            let out = body()
            eval(out)
            if nanCount(out) > 0, firstNaN == nil { firstNaN = run }
            if let reference {
                let delta = maxAbs(reference, out)
                if delta.isNaN || delta > 0 {
                    if firstDivergence == nil { firstDivergence = run }
                    worst = delta.isNaN ? .infinity : Swift.max(worst, delta)
                }
            } else {
                reference = out
            }
        }
        return (firstDivergence == nil, firstDivergence, firstNaN, worst)
    }

    /// A dense expert stack and its 4-bit twin, sharing the same weights, so
    /// that the unquantized gather can stand as ground truth.
    private func expertStack(
        inputDims: Int, outputDims: Int, numExperts: Int, seed: UInt64
    ) -> (dense: SwitchLinear, quantized: QuantizedSwitchLinear) {
        MLXRandom.seed(seed)
        let dense = SwitchLinear(
            inputDims: inputDims, outputDims: outputDims, numExperts: numExperts, bias: false)
        dense.update(parameters: dense.parameters().mapValues { $0.asType(.bfloat16) })
        eval(dense)
        return (dense, QuantizedSwitchLinear(dense, groupSize: 64, bits: 4, mode: .affine))
    }

    /// The layer's own quantized parameters, called directly, so that a test
    /// can name the hinted and the unhinted kernel rather than infer them.
    private func directGather(
        _ layer: QuantizedSwitchLinear
    ) -> (MLXArray, MLXArray, Bool) -> MLXArray {
        let parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
        return { x, indices, hint in
            MLX.gatherQuantizedMM(
                x, parameters["weight"]!, scales: parameters["scales"]!,
                biases: parameters["biases"], rhsIndices: indices, transpose: true,
                groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode,
                sortedIndices: hint)
        }
    }

    // A shape that crosses the kernel's route gate: 512 assignments over 64
    // experts is B / E = 8 and B = 512, both over the bar, at M == 1.
    private static let experts = 64
    private static let inputDims = 512
    private static let outputDims = 256
    private static let assignments = 512
    private static let topK = 8

    /// ALIGNED ROWS KEEP THE HINT. This is the shape `gatherSort` produces and
    /// the shape production runs on, and the layer must pass the hint through
    /// to it: the sorted-rhs kernel is worth roughly 4.7x on the gather.
    ///
    /// Forwarding is identified rather than inferred -- the layer's hinted
    /// output is required to be bit-identical to the hinted kernel called
    /// directly. The middle assertion is what gives the first one its meaning:
    /// at this shape the hint really does change the answer, so equality with
    /// the hinted call is evidence and not a coincidence. If that middle
    /// assertion ever fails, the hint has stopped selecting a different kernel
    /// in MLX; read it as news, not as a fault in this layer.
    @Test
    func alignedRowsKeepTheSortedHint() {
        let (dense, quantized) = expertStack(
            inputDims: Self.inputDims, outputDims: Self.outputDims,
            numExperts: Self.experts, seed: 0x51_0A)
        let gather = directGather(quantized)

        MLXRandom.seed(0xA1_16)
        // One row per index: [assignments, 1, K] against flat sorted ids.
        let x = MLXRandom.normal([Self.assignments, 1, Self.inputDims]).asType(.bfloat16)
        let ids = expertIds(Self.assignments, experts: Self.experts, seed: 0xA1_17)
        let indices = MLXArray(ids.sorted())
        eval(x, indices)
        #expect(x.size == indices.size * x.dim(-2) * x.dim(-1))

        let hintedKernel = gather(x, indices, true)
        let unhintedKernel = gather(x, indices, false)
        let fromLayer = quantized(x, indices, sortedIndices: true)
        let truth = dense(x, indices, sortedIndices: false)
        eval(hintedKernel, unhintedKernel, fromLayer, truth)

        let hintIsLoadBearing = maxAbs(hintedKernel, unhintedKernel)
        #expect(
            hintIsLoadBearing > 0,
            Comment(
                rawValue: "the sorted hint no longer changes the kernel at "
                    + "\(Self.assignments) assignments over \(Self.experts) experts"))

        #expect(
            maxAbs(fromLayer, hintedKernel) == 0,
            "QuantizedSwitchLinear withheld the sorted hint from index-aligned rows")

        let error = maxAbs(fromLayer, truth)
        #expect(
            error <= Self.quantizationTolerance,
            Comment(
                rawValue: "the hinted gather on aligned rows is \(error) from the "
                    + "unquantized gather, which is more than 4-bit error"))
        #expect(nanCount(fromLayer) == 0)
    }

    /// BROADCAST ROWS LOSE THE HINT. A nested `x` carries one row per TOKEN
    /// and the indices ask for one row per ASSIGNMENT, so MLX broadcasts and
    /// the stale row count applies. The layer must withhold.
    ///
    /// This is the leg the negative control turns red: forward the hint
    /// unconditionally and the layer stops matching the unhinted kernel here.
    @Test
    func broadcastRowsWithholdTheSortedHint() {
        let tokens = Self.assignments / Self.topK
        let (dense, quantized) = expertStack(
            inputDims: Self.inputDims, outputDims: Self.outputDims,
            numExperts: Self.experts, seed: 0x51_0B)
        let gather = directGather(quantized)

        MLXRandom.seed(0xB1_16)
        // One row per token, indices asking for eight times as many.
        let x = MLXRandom.normal([1, tokens, 1, 1, Self.inputDims]).asType(.bfloat16)
        let ids = expertIds(Self.assignments, experts: Self.experts, seed: 0xB1_17)
        let indices = MLXArray(ids.sorted(), [1, tokens, Self.topK])
        eval(x, indices)
        #expect(x.size != indices.size * x.dim(-2) * x.dim(-1))

        let unhintedKernel = gather(x, indices, false)
        let fromLayer = quantized(x, indices, sortedIndices: true)
        let truth = dense(x, indices, sortedIndices: false)
        eval(unhintedKernel, fromLayer, truth)

        #expect(
            maxAbs(fromLayer, unhintedKernel) == 0,
            "QuantizedSwitchLinear forwarded the sorted hint to broadcast rows")

        let error = maxAbs(fromLayer, truth)
        #expect(
            error <= Self.quantizationTolerance,
            Comment(
                rawValue: "the withheld-hint gather on broadcast rows is \(error) from "
                    + "the unquantized gather, which is more than 4-bit error"))
        #expect(nanCount(fromLayer) == 0)
    }

    /// WHY PRODUCTION NEVER MEETS THE FAULT. `gatherSort` is what every hinting
    /// caller in this package runs first, and it returns one row per index.
    /// That is the layer's alignment condition exactly, so the hinted branch is
    /// the branch production takes -- above the caller's sort threshold and
    /// below it, and either side of the kernel's route gate.
    @Test
    func gatherSortFeedsTheGatherOneRowPerIndex() {
        // 128 assignments is over the caller's sort threshold but under the
        // kernel's 4 * E route gate; 512 is over both.
        for assignments in [128, Self.assignments] {
            let tokens = assignments / Self.topK
            MLXRandom.seed(UInt64(assignments))
            let hidden = MLXRandom.normal([tokens, Self.inputDims]).asType(.bfloat16)
            let x = MLX.expandedDimensions(hidden, axes: [-2, -3])
            let ids = expertIds(assignments, experts: Self.experts, seed: 0xC1_17)
            let indices = MLXArray(ids, [tokens, Self.topK])
            eval(x, indices)

            let (sortedX, sortedIndices, _) = gatherSort(x: x, indices: indices)
            eval(sortedX, sortedIndices)

            #expect(
                sortedX.size == sortedIndices.size * sortedX.dim(-2) * sortedX.dim(-1),
                Comment(
                    rawValue: "gatherSort returned \(sortedX.size / sortedX.dim(-1)) rows "
                        + "for \(sortedIndices.size) indices at \(assignments) assignments"))
        }
    }

    /// THE REPRODUCER, with no layer in it: one quantized expert stack, one
    /// activation, one index vector, gathered on broadcast rows and on aligned
    /// ones.
    ///
    /// WHAT IS ASSERTED is what the fix rests on. The unhinted gather is
    /// correct on both shapes, and the HINTED gather is correct on aligned
    /// rows, which is the branch the fix keeps.
    ///
    /// THE HINTED CALL ON BROADCAST ROWS IS MEASURED AND PRINTED, NOT
    /// ASSERTED. It is wrong on the MLX pinned today, and asserting that would
    /// turn a repaired MLX into a red build. Its printed line is what to read
    /// before deciding the alignment condition can be dropped.
    ///
    /// Opt in with MLX_RUN_GATHER_QMM_REPRO=1. It allocates on the order of a
    /// gigabyte and needs an MLX GPU.
    @Test(.enabled(if: QuantizedSwitchLinearSortedHintTests.reproEnabled))
    func theBroadcastGatherIsWrongWithTheSortedHint() {
        // The geometry is load-bearing. See the note on `Reproducer`.
        #expect(Reproducer.experts == 128)
        #expect(Reproducer.outputDims == 704)
        #expect(Reproducer.inputDims == 2816)
        #expect(Reproducer.groupSize == 64)
        #expect(Reproducer.bits == 4)
        #expect(Reproducer.tokens == 1024)
        #expect(Reproducer.topK == 8)
        #expect(Reproducer.assignments == 8192)
        // Crosses the kernel's route gate: B / E = 64, B = 8192, M = 1.
        #expect(Reproducer.assignments / Reproducer.experts >= 4)

        Device.withDefaultDevice(.gpu) {
            let groupSize = Reproducer.groupSize
            let bits = Reproducer.bits

            MLXRandom.seed(0x51_0D)
            let dense = MLXRandom.normal([
                Reproducer.experts, Reproducer.outputDims, Reproducer.inputDims,
            ]).asType(.bfloat16)
            eval(dense)
            // Affine quantization always produces biases; the mxfp modes do not.
            let (packed, scales, optionalBiases) = MLX.quantized(
                dense, groupSize: groupSize, bits: bits)
            guard let biases = optionalBiases else {
                Issue.record("affine quantization must produce biases")
                return
            }
            eval(packed, scales, biases)

            let raw = expertIds(
                Reproducer.assignments, experts: Reproducer.experts, seed: 0x3D_7)
            let order = raw.indices.sorted { raw[$0] < raw[$1] || (raw[$0] == raw[$1] && $0 < $1) }
            let sortedIds = order.map { raw[$0] }

            MLXRandom.seed(0x51_0E)
            let tokenRows = MLXRandom.normal([Reproducer.tokens, Reproducer.inputDims])
                .asType(.bfloat16)
            // Broadcast shape: one row per token, indices per assignment.
            let nestedX = tokenRows.reshaped([
                1, Reproducer.tokens, 1, 1, Reproducer.inputDims,
            ])
            let nestedIndices = MLXArray(
                sortedIds, [1, Reproducer.tokens, Reproducer.topK])
            // Aligned shape: the rows `gatherSort` would hand over.
            let alignedX = tokenRows[MLXArray(order.map { Int32($0 / Reproducer.topK) })]
                .reshaped([Reproducer.assignments, 1, Reproducer.inputDims])
            let alignedIndices = MLXArray(sortedIds)
            eval(nestedX, nestedIndices, alignedX, alignedIndices)

            func gather(_ x: MLXArray, _ indices: MLXArray, hint: Bool) -> MLXArray {
                MLX.gatherQuantizedMM(
                    x, packed, scales: scales, biases: biases, rhsIndices: indices,
                    transpose: true, groupSize: groupSize, bits: bits, sortedIndices: hint)
            }

            // GROUND TRUTH. Stability is not correctness: a kernel can be
            // perfectly repeatable and perfectly wrong. Dequantize the packed
            // weights and run the ordinary gather on them; what is left is
            // 4-bit error.
            let dequantized = MLX.dequantized(
                packed, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
            let dequantizedTransposed = dequantized.swappedAxes(-1, -2)
            eval(dequantized, dequantizedTransposed)
            let nestedTruth = MLX.gatherMM(
                nestedX, dequantizedTransposed, rhsIndices: nestedIndices, sortedIndices: false)
            let alignedTruth = MLX.gatherMM(
                alignedX, dequantizedTransposed, rhsIndices: alignedIndices, sortedIndices: false)
            eval(nestedTruth, alignedTruth)

            #expect(repeatRun { gather(nestedX, nestedIndices, hint: false) }.identical)

            let nestedHintOff = gather(nestedX, nestedIndices, hint: false)
            let alignedHintOff = gather(alignedX, alignedIndices, hint: false)
            let alignedHintOn = gather(alignedX, alignedIndices, hint: true)
            eval(nestedHintOff, alignedHintOff, alignedHintOn)

            let nestedOffError = maxAbs(nestedHintOff, nestedTruth)
            let alignedOffError = maxAbs(alignedHintOff, alignedTruth)
            let alignedOnError = maxAbs(alignedHintOn, alignedTruth)
            print("gather_qmm broadcast rows, hint off, vs truth: max|d| = \(nestedOffError)")
            print("gather_qmm aligned   rows, hint off, vs truth: max|d| = \(alignedOffError)")
            print("gather_qmm aligned   rows, hint ON,  vs truth: max|d| = \(alignedOnError)")

            #expect(nestedOffError <= Reproducer.tolerance)
            #expect(alignedOffError <= Reproducer.tolerance)
            #expect(
                alignedOnError <= Reproducer.tolerance,
                Comment(
                    rawValue: "the hinted gather on aligned rows is \(alignedOnError) from "
                        + "dense truth: the branch the fix keeps is not safe"))

            // REPORTED, NOT ASSERTED: the broadcast rows with the hint on.
            let nestedHintOn = gather(nestedX, nestedIndices, hint: true)
            eval(nestedHintOn)
            print(
                "gather_qmm broadcast rows, hint ON,  vs truth: max|d| = "
                    + "\(maxAbs(nestedHintOn, nestedTruth)) nan=\(nanCount(nestedHintOn))")
            let hinted = repeatRun { gather(nestedX, nestedIndices, hint: true) }
            print(
                "gather_qmm broadcast rows, hint ON, repeated: identical=\(hinted.identical) "
                    + "firstDivergence=\(hinted.firstDivergence.map(String.init) ?? "none") "
                    + "firstNaN=\(hinted.firstNaN.map(String.init) ?? "none") "
                    + "worst=\(hinted.worst)")

            // THE NON-QUANTIZED GATHER IS THE CONTROL that keeps the fault
            // where it belongs. Repeat-stability would not say this, so the
            // hinted and unhinted calls are compared against each other on the
            // SAME broadcast rows.
            //
            // Exact equality is not the expectation and never was: `gather_mm`
            // takes a genuinely different kernel with the hint, and bf16
            // accumulation order differs between them. What is asserted is the
            // absence of the break -- these outputs run to scale 50, so a
            // stale row count would read in the hundreds, as it does one panel
            // up.
            let denseTransposed = dense.swappedAxes(-1, -2)
            eval(denseTransposed)
            let gatherMMHintOn = MLX.gatherMM(
                nestedX, denseTransposed, rhsIndices: nestedIndices, sortedIndices: true)
            let gatherMMHintOff = MLX.gatherMM(
                nestedX, denseTransposed, rhsIndices: nestedIndices, sortedIndices: false)
            eval(gatherMMHintOn, gatherMMHintOff)
            let gatherMMDelta = maxAbs(gatherMMHintOn, gatherMMHintOff)
            print("gather_mm broadcast rows, hint on vs hint off: max|d| = \(gatherMMDelta)")
            #expect(
                gatherMMDelta <= Reproducer.tolerance,
                Comment(
                    rawValue: "the NON-quantized gather broke with the hint too "
                        + "(\(gatherMMDelta)): the fault is not confined to the kernel "
                        + "that is handed a precomputed row count"))

            // WHAT THE HINT BUYS on the aligned rows production runs, which is
            // why the fix keeps it rather than withholding everywhere.
            // Reported only; a timing is not a contract.
            func elapsed(_ body: () -> MLXArray) -> Double {
                for _ in 0 ..< 2 { eval(body()) }
                let start = Date()
                for _ in 0 ..< 10 { eval(body()) }
                return Date().timeIntervalSince(start) / 10
            }
            let onSeconds = elapsed { gather(alignedX, alignedIndices, hint: true) }
            let offSeconds = elapsed { gather(alignedX, alignedIndices, hint: false) }
            print(
                "gather_qmm aligned rows, seconds per call: hint on \(onSeconds), "
                    + "hint off \(offSeconds), speedup \(offSeconds / onSeconds)")
        }
    }
}
