// GPTOSS prompt-output policy: conservative exactness and final-row parity.
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXLLM
@testable import MLXLMCommon

@Suite("GPTOSS output-tail prefill parity", .serialized)
struct GPTOSSPrefillOutputTests {
    private struct Run {
        let logits: MLXArray
        let caches: [CBv2LayerCache]
    }

    private func model(sinks: Float) throws -> GPTOSSModel {
        let config = try JSONDecoder().decode(GPTOSSConfiguration.self, from: Data("""
        {"model_type":"gpt_oss","num_hidden_layers":4,"num_local_experts":4,
         "num_experts_per_tok":2,"vocab_size":64,"rms_norm_eps":0.00001,
         "hidden_size":32,"intermediate_size":32,"head_dim":8,
         "num_attention_heads":4,"num_key_value_heads":2,"sliding_window":8}
        """.utf8))
        MLXRandom.seed(0x4750544F5353)
        let model = GPTOSSModel(config)
        for layer in model.loraLayers {
            let block = try #require(layer as? GPTOSSTransformerBlock)
            block.selfAttn.update(parameters: ModuleParameters.unflattened([
                "sinks": MLXArray(Array(repeating: sinks, count: 4))
            ]))
        }
        eval(model)
        return model
    }

    private func caches(_ model: GPTOSSModel, batch: Int, length: Int) -> [CBv2LayerCache] {
        var result: [CBv2LayerCache] = []
        _ = model.newCacheV2 { index, kind in
            let rows: [CBv2SequenceKV] = (0..<batch).map { _ in
                switch kind.attention {
                case .full:
                    return CBv2FullSequenceKV(
                        promptLength: length, maxLength: length + 16,
                        kvHeads: kind.kvHeads, headDim: kind.headDim)
                case .slidingWindow(let window):
                    return CBv2WindowedSequenceKV(
                        window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
                }
            }
            let cache = CBv2LayerCache(layerIndex: index, kind: kind, rows: rows)
            result.append(cache)
            return cache
        }
        return result
    }

    private func prompt(batch: Int, length: Int) -> MLXArray {
        var values: [Int32] = []
        values.reserveCapacity(batch * length)
        for row in 0..<batch {
            for position in 0..<length {
                let token = (position * 7 + row * 11 + 3) % 64
                values.append(Int32(token))
            }
        }
        return MLXArray(values).reshaped(batch, length)
    }

    private func prefill(
        _ model: GPTOSSModel, tokens: MLXArray, chunks: [Int], optimized: Bool,
        embeddings: MLXArray? = nil, policy: GPTOSSPrefillOutputPolicy = .last
    ) throws -> Run {
        let caches = caches(model, batch: tokens.dim(0), length: tokens.dim(1))
        let kvCaches: [KVCache] = caches
        var offset = 0
        var frontier: MLXArray?
        for (index, count) in chunks.enumerated() {
            let chunk = tokens[0..., offset..<(offset + count)]
            let embedded = embeddings.map { $0[0..., offset..<(offset + count), 0...] }
            let final = index == chunks.count - 1
            let output: MLXArray
            if optimized {
                output = model.prefillOutput(
                    chunk, inputEmbedding: embedded, cache: kvCaches,
                    requirement: final ? .lastPositionLogits : .evaluationOnly, policy: policy)
                #expect(output.shape == [tokens.dim(0), final ? 64 : 1])
            } else {
                let logits: MLXArray
                if let embedded {
                    logits = model.lmHead(model.model(
                        chunk, cache: kvCaches, inputEmbeddings: embedded))
                } else {
                    logits = model(chunk, cache: kvCaches)
                }
                #expect(logits.shape == [tokens.dim(0), count, 64])
                // Match the engine's legacy consumer, including the scalar
                // discarded intermediate logit. Do not compare that scalar
                // against the candidate's intentionally different hidden handle.
                output = final ? logits[0..., -1, 0...] : logits[0..., -1, 0..<1]
            }
            eval(output)
            // The production engine evaluates cache inner state along with
            // output; copying this contract avoids an artificial eager driver.
            eval(caches.flatMap { $0.innerState() })
            offset += count
            #expect(caches.allSatisfy { $0.rows.allSatisfy { $0.absoluteOffset == offset } })
            if final { frontier = output }
        }
        #expect(offset == tokens.dim(1))
        return Run(logits: try #require(frontier), caches: caches)
    }

    private func assertClose(_ lhs: MLXArray, _ rhs: MLXArray, tolerance: Float) {
        #expect(lhs.shape == rhs.shape)
        guard lhs.shape == rhs.shape else { return }
        let error = abs(lhs.asType(.float32) - rhs.asType(.float32)).max().item(Float.self)
        #expect(error.isFinite && error <= tolerance)
    }

    private func assertCaches(_ lhs: Run, _ rhs: Run, exact: Bool) {
        #expect(lhs.caches.count == rhs.caches.count)
        for (a, b) in zip(lhs.caches, rhs.caches) {
            #expect(a.rows.count == b.rows.count)
            for (ar, br) in zip(a.rows, b.rows) {
                #expect(ar.absoluteOffset == br.absoluteOffset)
                #expect(ar.retainedCount == br.retainedCount)
                let av = ar.snapshot(), bv = br.snapshot()
                #expect(av.offset == bv.offset)
                eval(av.keys, av.values, bv.keys, bv.values)
                #expect(av.keys.shape == bv.keys.shape)
                #expect(av.values.shape == bv.values.shape)
                guard av.keys.shape == bv.keys.shape, av.values.shape == bv.values.shape else { continue }
                if exact {
                    #expect((av.keys .== bv.keys).all().item(Bool.self))
                    #expect((av.values .== bv.values).all().item(Bool.self))
                } else {
                    assertClose(av.keys, bv.keys, tolerance: 5e-5)
                    assertClose(av.values, bv.values, tolerance: 5e-5)
                }
            }
        }
    }

    private func assertContinuation(
        _ model: GPTOSSModel, reference: Run, candidate: Run, exactCaches: Bool
    ) {
        var left = reference.logits, right = candidate.logits
        for _ in 0..<6 {
            assertClose(left, right, tolerance: 5e-5)
            let a = argMax(left, axis: -1).asArray(Int32.self)
            let b = argMax(right, axis: -1).asArray(Int32.self)
            #expect(a == b)
            // Feed each path its own greedy token: a divergent output cannot
            // be hidden by teacher-forcing both paths with the reference.
            left = model(MLXArray(a).reshaped(a.count, 1), cache: reference.caches)[0..., -1, 0...]
            right = model(MLXArray(b).reshaped(b.count, 1), cache: candidate.caches)[0..., -1, 0...]
            eval(left, right)
            eval(reference.caches.flatMap { $0.innerState() })
            eval(candidate.caches.flatMap { $0.innerState() })
            assertCaches(reference, candidate, exact: exactCaches)
        }
        assertClose(left, right, tolerance: 5e-5)
    }

    @Test("prompt optimization does not claim packed prefill support")
    func packedPolicy() throws {
        let model = try model(sinks: 0)
        #expect(model.cbv2SupportsPackedPrefill == false)
        #expect(model.cbv2SupportsPackedMultimodalPrefill == false)
    }

    @Test("conservative intermediate elision preserves exact frontier logits and decode")
    func conservativeParity() throws {
        let model = try model(sinks: 0.5)
        for batch in [1, 2, 4] {
            let tokens = prompt(batch: batch, length: 33)
            let reference = try prefill(model, tokens: tokens, chunks: [7, 1, 1, 8, 16], optimized: false)
            for policy in [GPTOSSPrefillOutputPolicy.full, .intermediate] {
                let candidate = try prefill(model, tokens: tokens, chunks: [7, 1, 1, 8, 16],
                                            optimized: true, policy: policy)
                #expect((reference.logits .== candidate.logits).all().item(Bool.self))
                assertCaches(reference, candidate, exact: true)
                // Each comparison gets its own fresh reference caches.
                let continuationReference = try prefill(
                    model, tokens: tokens, chunks: [7, 1, 1, 8, 16], optimized: false)
                assertContinuation(model, reference: continuationReference,
                                   candidate: candidate, exactCaches: true)
            }
        }
    }

    @Test("policy defaults to final-position output and retains a full rollback")
    func policySelection() {
        #expect(GPTOSSPrefillOutputPolicy.resolve(nil) == .last)
        #expect(GPTOSSPrefillOutputPolicy.resolve("invalid") == .last)
        #expect(GPTOSSPrefillOutputPolicy.resolve("full") == .full)
        #expect(GPTOSSPrefillOutputPolicy.resolve("intermediate") == .intermediate)
        #expect(GPTOSSPrefillOutputPolicy.resolve("last") == .last)
        #expect(GPTOSSPrefillOutputPolicy.resolve("last-layer") == .lastLayer)
    }

    @Test("final-layer last-query pruning keeps full KV and continuing decode")
    func finalLayerParity() throws {
        for sinks in [Float(0), Float(0.5)] {
            let model = try model(sinks: sinks)
            for batch in [1, 2, 4] {
                let tokens = prompt(batch: batch, length: 33)
                let reference = try prefill(model, tokens: tokens, chunks: [7, 1, 1, 8, 16], optimized: false)
                let candidate = try prefill(model, tokens: tokens, chunks: [7, 1, 1, 8, 16],
                                            optimized: true, policy: .lastLayer)
                assertClose(reference.logits, candidate.logits, tolerance: 5e-5)
                assertCaches(reference, candidate, exact: true)
                assertContinuation(model, reference: reference, candidate: candidate, exactCaches: true)
                let tailCaches = caches(model, batch: batch, length: 33)
                let hidden = model.model(tokens, cache: tailCaches, lastLayerLastQuery: true)
                eval(hidden)
                #expect(hidden.shape == [batch, 1, 32])
            }
        }
    }

    @Test("last-query gate rejects short, sliding, shared, and incapable caches")
    func lastQueryGate() {
        let full = CBv2LayerKind(attention: .full, sharesKVWithLayer: nil,
                                hasSinks: true, headDim: 8, kvHeads: 2, queryHeads: 4)
        let sliding = CBv2LayerKind(attention: .slidingWindow(8), sharesKVWithLayer: nil,
                                   hasSinks: true, headDim: 8, kvHeads: 2, queryHeads: 4)
        let shared = CBv2LayerKind(attention: .full, sharesKVWithLayer: 0,
                                  hasSinks: true, headDim: 8, kvHeads: 2, queryHeads: 4)
        let capable = CBv2LayerCache(layerIndex: 1, kind: full)
        #expect(gptossLastQueryPrefillEligible(sequenceLength: 16, isFinalFullLayer: true, cache: capable))
        #expect(!gptossLastQueryPrefillEligible(sequenceLength: 1, isFinalFullLayer: true, cache: capable))
        #expect(!gptossLastQueryPrefillEligible(sequenceLength: 16, isFinalFullLayer: false, cache: capable))
        #expect(!gptossLastQueryPrefillEligible(sequenceLength: 16, isFinalFullLayer: true, cache: nil))
        #expect(!gptossLastQueryPrefillEligible(sequenceLength: 16, isFinalFullLayer: true, cache: StandardKVCache()))
        #expect(!gptossLastQueryPrefillEligible(sequenceLength: 16, isFinalFullLayer: true,
                                              cache: CBv2LayerCache(layerIndex: 1, kind: sliding)))
        #expect(!gptossLastQueryPrefillEligible(sequenceLength: 16, isFinalFullLayer: true,
                                              cache: CBv2LayerCache(layerIndex: 1, kind: shared)))
    }

    @Test("last-layer policy falls back to last-head policy with legacy caches")
    func lastLayerLegacyFallback() throws {
        let model = try model(sinks: 0.5)
        let tokens = prompt(batch: 1, length: 9)
        let reference = model.prefillOutput(tokens, inputEmbedding: nil,
            cache: model.newCache(parameters: nil), requirement: .lastPositionLogits, policy: .last)
        let candidate = model.prefillOutput(tokens, inputEmbedding: nil,
            cache: model.newCache(parameters: nil), requirement: .lastPositionLogits, policy: .lastLayer)
        eval(reference, candidate)
        #expect((reference .== candidate).all().item(Bool.self))
    }

    @Test("same chunks preserve all-layer KV exactly with inactive and active sinks")
    func chunkParity() throws {
        for sinks in [Float(0), Float(0.5)] {
            let model = try model(sinks: sinks)
            for batch in [1, 2, 4] {
                // W-1/W/W+1 and multi-wrap boundary; 33 tokens also crosses
                // GPTOSS's MoE sort threshold of 64 routed expert entries.
                for chunks in [[1], [3], [7], [8], [9], [33], [7, 1, 1, 8, 16]] {
                    let tokens = prompt(batch: batch, length: chunks.reduce(0, +))
                    let reference = try prefill(model, tokens: tokens, chunks: chunks, optimized: false)
                    let candidate = try prefill(model, tokens: tokens, chunks: chunks, optimized: true)
                    assertClose(reference.logits, candidate.logits, tolerance: 1e-5)
                    assertCaches(reference, candidate, exact: true)
                    assertContinuation(model, reference: reference, candidate: candidate, exactCaches: true)
                }
            }
        }
    }

    @Test("chunking remains equivalent across repeated sliding-window eviction")
    func chunkedVersusUnchunked() throws {
        let model = try model(sinks: 0.5)
        let tokens = prompt(batch: 2, length: 33)
        let reference = try prefill(model, tokens: tokens, chunks: [33], optimized: false)
        let candidate = try prefill(model, tokens: tokens, chunks: [7, 1, 1, 8, 16], optimized: true)
        // Different attention/GEMM rectangles can reorder float32 reductions;
        // use a bounded logit/KV tolerance across those shapes, plus exact
        // greedy IDs. Same-shape candidate checks above keep KV byte-exact.
        assertClose(reference.logits, candidate.logits, tolerance: 5e-5)
        assertCaches(reference, candidate, exact: false)
        assertContinuation(model, reference: reference, candidate: candidate, exactCaches: false)
    }

    @Test("explicit embeddings preserve the public prompt seam")
    func embeddingParity() throws {
        let model = try model(sinks: 0.5)
        let tokens = prompt(batch: 2, length: 9)
        let embeddings = model.model.embedTokens(tokens) * 0.75
        eval(embeddings)
        let reference = try prefill(
            model, tokens: tokens, chunks: [4, 5], optimized: false, embeddings: embeddings)
        let candidate = try prefill(
            model, tokens: tokens, chunks: [4, 5], optimized: true, embeddings: embeddings)
        assertClose(reference.logits, candidate.logits, tolerance: 1e-5)
        assertCaches(reference, candidate, exact: true)
        assertContinuation(model, reference: reference, candidate: candidate, exactCaches: true)
    }
}
