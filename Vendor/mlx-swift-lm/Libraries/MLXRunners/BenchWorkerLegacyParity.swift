// Copyright © 2026 Eigen Labs.
//
// MLXRunners — `bench-worker diag-parity`.
//
// WHY THIS EXISTS. On the 125B the free run and the teacher-forced oracle
// agree for nine tokens after a one-forward prefill and part at the tenth,
// on both drivers, while the old engine — which vendors the SAME legacy
// forward and cache code this fork carries — reproduces the golden past
// that step. The fixture model does not show it. So the question has to be
// put to the real weights: does the fork's LEGACY path (`model(ids, cache:)`
// over `makeCache()`, the code the old engine ran) still match the golden,
// and does the CBv2 stepper match the legacy path? Whichever pair parts
// first names the layer the defect lives in: the CBv2 recurrent-state
// bookkeeping, or something outside it (transform, n-gram reader, config).
//
// DIAGNOSTIC ONLY. Loads once, serves nothing, writes plain text to stdout.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

public enum BenchWorkerLegacyParity {

    /// The subset of a correctness golden this needs, from any of the three
    /// shapes the track carries: the free-run window under `benchmark`
    /// (pool goldens), the same keys at the top level, or a teacher-forced
    /// case (`cases[0].prompt_tokens` + `expected_tokens`, the public
    /// gate golden).
    struct Golden {
        var decodeSeedTokens: [Int]
        var expectedDecodeSeedToken: Int?
        var expectedDecodeTokens: [Int]

        struct Window: Decodable {
            var decodeSeedTokens: [Int]
            var expectedDecodeSeedToken: Int?
            var expectedDecodeTokens: [Int]
            enum CodingKeys: String, CodingKey {
                case decodeSeedTokens = "decode_seed_tokens"
                case expectedDecodeSeedToken = "expected_decode_seed_token"
                case expectedDecodeTokens = "expected_decode_tokens"
            }
        }
        struct Case: Decodable {
            var promptTokens: [Int]
            var expectedTokens: [Int]
            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case expectedTokens = "expected_tokens"
            }
        }
        struct File: Decodable {
            var benchmark: Window?
            var cases: [Case]?
        }

        static func load(_ url: URL) throws -> Golden {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            if let window = try? decoder.decode(Window.self, from: data) {
                return Golden(
                    decodeSeedTokens: window.decodeSeedTokens,
                    expectedDecodeSeedToken: window.expectedDecodeSeedToken,
                    expectedDecodeTokens: window.expectedDecodeTokens)
            }
            let file = try decoder.decode(File.self, from: data)
            if let window = file.benchmark {
                return Golden(
                    decodeSeedTokens: window.decodeSeedTokens,
                    expectedDecodeSeedToken: window.expectedDecodeSeedToken,
                    expectedDecodeTokens: window.expectedDecodeTokens)
            }
            guard let first = file.cases?.first, !first.expectedTokens.isEmpty else {
                throw Failure.noChain(url.path)
            }
            return Golden(
                decodeSeedTokens: first.promptTokens,
                expectedDecodeSeedToken: first.expectedTokens[0],
                expectedDecodeTokens: Array(first.expectedTokens.dropFirst()))
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case notQwen4Exp(String)
        case noChain(String)
        case streamPromoted(String)
        public var description: String {
            switch self {
            case .notQwen4Exp(let type):
                return "diag-parity compares the Qwen4Exp legacy path; the loaded model is \(type)"
            case .noChain(let path):
                return "diag-parity: \(path) carries no benchmark window and no cases[0] chain"
            case .streamPromoted(let detail):
                return "diag-parity: \(detail)"
            }
        }
    }

    public static func run(
        runner: any Runner, golden goldenURL: URL, steps: Int, output: FileHandle
    ) throws {
        let golden = try Golden.load(goldenURL)
        guard let model = runner.servingModel as? Qwen4ExpModel else {
            throw Failure.notQwen4Exp(String(describing: type(of: runner.servingModel)))
        }
        func emit(_ line: String) { output.write(Data((line + "\n").utf8)) }

        // The forced chain: the seed, then the golden's expected tokens.
        // Step 0 is the prefill; step k >= 1 feeds expected[k - 1].
        let count = min(steps, golden.expectedDecodeTokens.count)
        var expectedNext: [Int?] = [golden.expectedDecodeSeedToken]
        expectedNext.append(
            contentsOf: golden.expectedDecodeTokens.prefix(count).map { Optional($0) })
        var fed: [[Int]] = [golden.decodeSeedTokens]
        if let seedToken = golden.expectedDecodeSeedToken { fed.append([seedToken]) }
        fed.append(
            contentsOf: golden.expectedDecodeTokens.prefix(
                count - (golden.expectedDecodeSeedToken == nil ? 0 : 1)
            ).map { [$0] })

        emit(
            "diag-parity: seed \(golden.decodeSeedTokens.count) tokens, \(fed.count) forwards, "
                + "legacy = model(ids, cache: makeCache()) in one forward, cbv2 = runner.makeStepper()"
        )

        // The model dtype: the first floating tensor of the tower (a norm
        // weight or a quantized scale; packed weights are integer).
        guard
            let modelDType = model.model.parameters().flattened()
                .first(where: { [.bfloat16, .float16, .float32].contains($0.1.dtype) })?.1.dtype
        else { throw Failure.streamPromoted("the tower has no floating tensor") }
        let legacyCaches = model.makeCache()
        let stepper = try runner.makeStepper()
        try stepper.begin()
        guard let raw = stepper as? CBv2SingleRowStepper else {
            emit("diag-parity: stepper is not CBv2SingleRowStepper; top-k only")
            return
        }

        var firstLegacyMiss: Int?
        var firstPathSplit: Int?
        for (step, tokens) in fed.enumerated() {
            let ids = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
            // Wall time per path, forward to readback, so the same run also
            // says what one step costs on each driver.
            let legacyStart = DispatchTime.now().uptimeNanoseconds
            let legacyRaw = model(ids, cache: legacyCaches)[0..., -1, 0...]
            let legacy = legacyRaw.asType(.float32)
            eval(legacy)
            // The head reads the hyper stream, so the raw logits carry the
            // stream's dtype at the last layer. A stray float32 scalar
            // anywhere in the tower promotes it, and everything after runs
            // in float32 (the PLE gate floor did exactly that).
            if step == 0, legacyRaw.dtype != modelDType {
                throw Failure.streamPromoted(
                    "stream dtype at the last layer is \(legacyRaw.dtype), model dtype is \(modelDType)")
            }
            let legacyMs = Double(DispatchTime.now().uptimeNanoseconds - legacyStart) / 1e6
            let cbv2Start = DispatchTime.now().uptimeNanoseconds
            let cbv2 = try raw.forwardLogits(tokens).asType(.float32)
            eval(cbv2)
            let cbv2Ms = Double(DispatchTime.now().uptimeNanoseconds - cbv2Start) / 1e6
            let lTop = top(legacy, 4)
            let cTop = top(cbv2, 4)
            let linf = MLX.abs(legacy - cbv2).max().item(Float.self)
            let expected = step < expectedNext.count ? expectedNext[step] : nil
            let legacyOK = expected.map { $0 == lTop[0].token }
            let cbv2OK = expected.map { $0 == cTop[0].token }
            if firstLegacyMiss == nil, legacyOK == false { firstLegacyMiss = step }
            if firstPathSplit == nil, lTop[0].token != cTop[0].token { firstPathSplit = step }
            emit(
                "step \(step) fed \(tokens.count == 1 ? String(tokens[0]) : "seed") expected \(expected.map(String.init) ?? "-") "
                    + "| legacy \(render(lTop)) \(legacyOK.map { $0 ? "OK" : "MISS" } ?? "") "
                    + "| cbv2 \(render(cTop)) \(cbv2OK.map { $0 ? "OK" : "MISS" } ?? "") "
                    + "| linf \(String(format: "%.4f", linf)) "
                    + "| ms legacy \(String(format: "%.1f", legacyMs)) cbv2 \(String(format: "%.1f", cbv2Ms))"
            )
        }
        emit(
            "diag-parity: first legacy miss vs golden: \(firstLegacyMiss.map(String.init) ?? "none"); "
                + "first legacy/cbv2 argmax split: \(firstPathSplit.map(String.init) ?? "none")")
    }

    private static func top(_ row: MLXArray, _ k: Int) -> [(token: Int, logit: Float)] {
        let flat = row.reshaped([-1])
        let n = min(k, flat.dim(0))
        let ids = argPartition(-flat, kth: n - 1)[..<n]
        let values = flat[ids]
        eval(ids, values)
        return zip(ids.asArray(Int32.self), values.asArray(Float.self))
            .map { (token: Int($0), logit: $1) }
            .sorted { $0.logit > $1.logit || ($0.logit == $1.logit && $0.token < $1.token) }
    }

    private static func render(_ top: [(token: Int, logit: Float)]) -> String {
        "[" + top.map { "\($0.token) \(String(format: "%.3f", $0.logit))" }.joined(separator: ", ")
            + "]"
    }
}
