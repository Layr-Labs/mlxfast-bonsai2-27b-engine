import Foundation
import MLX
import MLXLMCommon
import Testing


@Suite(.serialized)
struct QwenDirectExpertReductionPerfTests {
    @Test func pairedDirectReductionVersusLegacy() throws {
        guard ProcessInfo.processInfo.environment["MLX_QWEN_DIRECT_REDUCTION_PERF"] == "1"
        else { return }
        let tokens = 2048
        let topK = 8
        let hidden = 2048
        let assignments = tokens * topK
        MLXRandom.seed(9127)
        let original = MLXRandom.uniform(
            low: -0.25, high: 0.25, [tokens, topK, hidden]
        ).asType(.bfloat16)
        let weights = softmax(
            MLXRandom.normal([tokens, topK]).asType(.bfloat16),
            axis: -1, precise: true)
        let expertIndices = MLXArray(
            (0 ..< assignments).map { UInt32(($0 * 37 + $0 / topK * 11) % 256) })
        let order = argSort(expertIndices)
        let inverse = argSort(order)
        let sorted = original.reshaped(assignments, hidden)[order]
        eval(original, weights, inverse, sorted)

        func legacy() -> MLXArray { weightedExpertSum(original, weights) }
        func direct() -> MLXArray {
            weightedExpertUnsort(
                sortedOutputs: sorted, inverseOrder: inverse, weights: weights)
        }
        for _ in 0 ..< 5 { eval(legacy()); eval(direct()) }
        var legacyTimes: [Double] = []
        var directTimes: [Double] = []
        for i in 0 ..< 25 {
            if i % 2 == 0 {
                var t = DispatchTime.now().uptimeNanoseconds
                eval(legacy())
                legacyTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
                t = DispatchTime.now().uptimeNanoseconds
                eval(direct())
                directTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
            } else {
                var t = DispatchTime.now().uptimeNanoseconds
                eval(direct())
                directTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
                t = DispatchTime.now().uptimeNanoseconds
                eval(legacy())
                legacyTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6)
            }
        }
        legacyTimes.sort(); directTimes.sort()
        print(String(format:
            "[qwen-direct-reduction-perf] legacy=%.4fms direct=%.4fms speedup=%.3fx",
            legacyTimes[12], directTimes[12], legacyTimes[12] / directTimes[12]))
    }
}
