import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXLLM
import MLXLMCommon

/// Optional diagnostic with real quantized weights. No tokenizer/downloads,
/// scheduler, prefix cache, or MTP. Each prompt row is prefetched separately,
/// then ordinary decode uses the requested rectangular batch.
@Suite("GPTOSS real checkpoint optimization parity", .serialized)
struct GPTOSSRealCheckpointParityTests {
    struct Top2: Codable {
        let ids: [Int]
        let values: [Float]
        var margin: Float { values[0] - values[1] }
    }
    struct Position: Codable {
        let index: Int
        let row: Int
        let reference: Top2
        let candidate: Top2
        let referenceMargin: Float
        let candidateMargin: Float
        let maxAbsDifference: Float
        let top1Agrees: Bool
    }
    struct CacheReceipt: Codable {
        let layer: Int
        let row: Int
        let kind: String
        let offset: Int
        let retainedCount: Int
        let keyShape: [Int]
        let valueShape: [Int]
        let keyDType: String
        let valueDType: String
        let keyFP32Digest: String
        let valueFP32Digest: String
    }
    struct Arm: Codable {
        let policy: String
        let fusedGateUp: Bool
        let batch: Int
        let promptTokens: Int
        let chunks: [Int]
        let promptIDs: [[Int]]
        let referenceGreedyIDs: [[Int]]
        let candidateGreedyIDs: [[Int]]
        let teacherForced: [Position]
        let referencePrefillCache: [CacheReceipt]
        let candidatePrefillCache: [CacheReceipt]
        let candidateFinalCache: [CacheReceipt]
    }
    struct Report: Codable {
        let schemaVersion: Int
        let modelPath: String
        let configSHA256: String
        let scope: String
        let arms: [Arm]
    }
    struct State {
        var frontier: MLXArray
        let caches: [CBv2LayerCache]
    }
    struct Trajectory {
        let rows: [[Int]]
        let logits: [[[Float]]]
        let prefillCache: [CacheReceipt]
        let finalCache: [CacheReceipt]
    }

    @Test("full/intermediate/last and split/fused, equal prefixes and greedy output", .enabled(if:
        ProcessInfo.processInfo.environment["DARKBLOOM_GPTOSS_REAL_PARITY_MODEL"] != nil))
    func realCheckpoint() throws {
        let env = ProcessInfo.processInfo.environment
        let path = try #require(env["DARKBLOOM_GPTOSS_REAL_PARITY_MODEL"])
        let output = try #require(env["DARKBLOOM_GPTOSS_REAL_PARITY_OUTPUT"])
        try #require(!GPTOSSModel.fusedGateUpEnabled, "run with DARKBLOOM_GPTOSS_FUSED_GATE_UP=0")
        let batches = list(env["DARKBLOOM_GPTOSS_REAL_PARITY_BATCHES"], fallback: [1, 2, 4])
        let lengths = list(env["DARKBLOOM_GPTOSS_REAL_PARITY_LENGTHS"], fallback: [512, 2065])
        try #require(!batches.isEmpty && batches.allSatisfy { [1, 2, 4].contains($0) })
        try #require(!lengths.isEmpty && lengths.allSatisfy { $0 > 0 && $0 <= 4096 })
        let directory = URL(fileURLWithPath: path)
        let bytes = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(GPTOSSConfiguration.self, from: bytes)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: bytes)
        let split = GPTOSSModel(configuration)
        try loadWeights(modelDirectory: directory, model: split,
                        perLayerQuantization: base.perLayerQuantization)
        eval(split)
        // Fresh model from exactly the same packed checkpoint parameters.
        // Only new gate/up modules need quantization; no requantization of
        // checkpoint values occurs before the strict parameter update.
        let fused = GPTOSSModel(configuration)
        fused.checkpointPerLayerQuantization = base.perLayerQuantization
        let original = Dictionary(uniqueKeysWithValues: split.parameters().flattened())
        let adjusted = fused.fuseGateUpWeights(original, enabled: true)
        quantize(model: fused) { modulePath, _ in
            guard adjusted[modulePath + ".scales"] != nil,
                  let table = base.perLayerQuantization else { return nil }
            return resolveQuantization(path: modulePath, perLayerQuantization: table,
                                       aliasing: fused)?.asTuple
        }
        try fused.update(parameters: ModuleParameters.unflattened(adjusted), verify: [.all])
        eval(fused)
        let expertModules = fused.namedModules().compactMap { $0.1 as? SwiGLUSwitchGLU }
        let allFused = expertModules.allSatisfy { $0.hasFusedGateUp }
        try #require(allFused)
        var arms: [Arm] = []
        for length in lengths {
            let chunks = length > 2048 ? [2048, length - 2048] : [length]
            for batch in batches {
                let prompts = (0..<batch).map { prompt(length: length, row: $0) }
                let reference = trajectory(split, prompts: prompts, chunks: chunks,
                                           policy: .full, forced: nil)
                for (model, isFused) in [(split, false), (fused, true)] {
                    for policy in GPTOSSPrefillOutputPolicy.allCases {
                        if !isFused && policy == .full { continue }
                        let forced = trajectory(model, prompts: prompts, chunks: chunks,
                                                policy: policy, forced: reference.rows)
                        let greedy = trajectory(model, prompts: prompts, chunks: chunks,
                                                policy: policy, forced: nil)
                        var positions: [Position] = []
                        for index in 0..<16 {
                            for row in 0..<batch {
                                let a = reference.logits[index][row], b = forced.logits[index][row]
                                let at = top2(a), bt = top2(b)
                                let error = zip(a, b).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
                                positions.append(Position(index: index, row: row,
                                    reference: at, candidate: bt,
                                    referenceMargin: at.margin, candidateMargin: bt.margin,
                                    maxAbsDifference: error, top1Agrees: at.ids[0] == bt.ids[0]))
                            }
                        }
                        arms.append(Arm(policy: policy.rawValue, fusedGateUp: isFused,
                            batch: batch, promptTokens: length, chunks: chunks, promptIDs: prompts,
                            referenceGreedyIDs: reference.rows, candidateGreedyIDs: greedy.rows,
                            teacherForced: positions,
                            referencePrefillCache: reference.prefillCache,
                            candidatePrefillCache: forced.prefillCache, candidateFinalCache: forced.finalCache))
                        let report = Report(schemaVersion: 1, modelPath: path,
                            configSHA256: sha(bytes),
                            scope: "Direct production owning contiguous/windowed caches; sequential B1 prompt rows then rectangular decode; 16 forced and greedy positions; no scheduler/cache reuse/speculation. Diagnostic readbacks are not performance measurements.", arms: arms)
                        let encoder = JSONEncoder()
                        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                        try encoder.encode(report).write(to: URL(fileURLWithPath: output), options: .atomic)
                        print("[gptoss-real-parity] L=\(length) B=\(batch) policy=\(policy.rawValue) fused=\(isFused) greedyExact=\(reference.rows == greedy.rows) maxDiff=\(positions.map(\.maxAbsDifference).max() ?? 0)")
                        if !isFused && policy == .intermediate {
                            #expect(reference.rows == greedy.rows)
                            #expect(positions.allSatisfy { $0.maxAbsDifference == 0 })
                        }
                    }
                }
            }
        }
    }

    private func trajectory(_ model: GPTOSSModel, prompts: [[Int]], chunks: [Int],
                            policy: GPTOSSPrefillOutputPolicy, forced: [[Int]]?) -> Trajectory {
        var state = prefill(model, prompts: prompts, chunks: chunks, policy: policy)
        let prefillCache = receipts(state.caches)
        var tokens = Array(repeating: [Int](), count: prompts.count)
        var logits: [[[Float]]] = []
        for position in 0..<16 {
            let rows = (0..<prompts.count).map {
                state.frontier[$0].asType(.float32).asArray(Float.self)
            }
            #expect(rows.allSatisfy { $0.allSatisfy(\.isFinite) })
            logits.append(rows)
            let chosen = rows.map { top2($0).ids[0] }
            for row in prompts.indices { tokens[row].append(chosen[row]) }
            if position < 15 {
                let next = prompts.indices.map { Int32(forced?[$0][position] ?? chosen[$0]) }
                state.frontier = model(MLXArray(next).reshaped(prompts.count, 1),
                                       cache: state.caches)[0..., -1, 0...]
                eval(state.frontier)
                eval(state.caches.flatMap { $0.innerState() })
                #expect(state.caches.allSatisfy { cache in
                    cache.rows.allSatisfy { $0.absoluteOffset == prompts[0].count + position + 1 }
                })
            }
        }
        return Trajectory(rows: tokens, logits: logits, prefillCache: prefillCache,
                          finalCache: receipts(state.caches))
    }

    private func prefill(_ model: GPTOSSModel, prompts: [[Int]], chunks: [Int],
                         policy: GPTOSSPrefillOutputPolicy) -> State {
        var allCaches: [[CBv2LayerCache]] = []
        var logits: [MLXArray] = []
        for prompt in prompts {
            var caches: [CBv2LayerCache] = []
            _ = model.newCacheV2 { index, kind in
                let row: CBv2SequenceKV
                switch kind.attention {
                case .full:
                    row = CBv2FullSequenceKV(promptLength: prompt.count, maxLength: prompt.count + 32,
                                            kvHeads: kind.kvHeads, headDim: kind.headDim)
                case .slidingWindow(let window):
                    row = CBv2WindowedSequenceKV(window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
                }
                let cache = CBv2LayerCache(layerIndex: index, kind: kind, rows: [row])
                caches.append(cache)
                return cache
            }
            var cursor = 0
            var output = MLXArray()
            for (index, count) in chunks.enumerated() {
                let input = MLXArray(prompt[cursor..<(cursor + count)].map(Int32.init)).reshaped(1, count)
                output = model.prefillOutput(input, inputEmbedding: nil, cache: caches,
                    requirement: index == chunks.count - 1 ? .lastPositionLogits : .evaluationOnly,
                    policy: policy)
                eval(output)
                eval(caches.flatMap { $0.innerState() })
                cursor += count
            }
            #expect(cursor == prompt.count)
            allCaches.append(caches)
            logits.append(output)
        }
        var combined: [CBv2LayerCache] = []
        for layer in allCaches[0].indices {
            var rows: [CBv2SequenceKV] = []
            for caches in allCaches { rows.append(contentsOf: caches[layer].rows) }
            combined.append(CBv2LayerCache(layerIndex: layer,
                            kind: allCaches[0][layer].kind, rows: rows))
        }
        return State(frontier: concatenated(logits, axis: 0), caches: combined)
    }

    private func receipts(_ caches: [CBv2LayerCache]) -> [CacheReceipt] {
        caches.enumerated().flatMap { layer, cache in
            cache.rows.enumerated().map { row, state in
                let snapshot = state.snapshot()
                return CacheReceipt(layer: layer, row: row, kind: String(describing: cache.kind.attention),
                    offset: state.absoluteOffset, retainedCount: state.retainedCount,
                    keyShape: snapshot.keys.shape, valueShape: snapshot.values.shape,
                    keyDType: String(describing: snapshot.keys.dtype), valueDType: String(describing: snapshot.values.dtype),
                    keyFP32Digest: digest(snapshot.keys), valueFP32Digest: digest(snapshot.values))
            }
        }
    }
    private func digest(_ array: MLXArray) -> String {
        var values = array.asType(.float32).asArray(Float.self).map { $0.bitPattern.littleEndian }
        return values.withUnsafeMutableBytes { sha(Data($0)) }
    }
    private func top2(_ values: [Float]) -> Top2 {
        var a = 0, b = 1
        if values[b] > values[a] { swap(&a, &b) }
        for index in 2..<values.count {
            if values[index] > values[a] { b = a; a = index }
            else if values[index] > values[b] { b = index }
        }
        return Top2(ids: [a, b], values: [values[a], values[b]])
    }
    private func prompt(length: Int, row: Int) -> [Int] {
        // Fixed common GPTOSS token IDs; repeated and rotated, with the exact
        // prompt IDs saved in the report. No tokenizer or template drift.
        let seed = [813, 581, 6855, 28460, 326, 279, 2167, 1309, 316, 8420, 290, 4792, 25, 1416, 679, 261]
        var tokens: [Int] = []
        tokens.reserveCapacity(length)
        for position in 0..<length {
            tokens.append(seed[(position + row * 7 + 1) % seed.count])
        }
        return tokens
    }
    private func list(_ raw: String?, fallback: [Int]) -> [Int] {
        raw.map { $0.split(separator: ",").compactMap { Int($0) } } ?? fallback
    }
    private func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
