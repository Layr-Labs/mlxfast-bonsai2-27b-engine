import MLX
import MLXLMCommon
import MLXNN
import Testing

private class GemmaLikePositionlessModel: Module, LanguageModel, CBv2EmbeddingForwardable {
    private(set) var forwardCount = 0

    var supportsVisionSpanPrefill: Bool { true }
    var supportsCausalVisionPrefill: Bool { false }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .logits(LMOutput(logits: MLXArray.zeros([1, 1, 2])))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fixtureForward(inputs, cache: cache)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

    func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        inputs.asType(.float32)[0..., 0..., .newAxis]
    }

    func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        fixtureForward(inputs, cache: cache)
    }

    fileprivate func fixtureForward(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardCount += 1
        let batch = inputs.dim(0)
        let length = inputs.dim(1)
        let qkv = MLXArray.zeros([batch, 1, length, 1])
        for cache in cache ?? [] {
            guard let layer = cache as? CBv2AttendingLayerCache else {
                preconditionFailure("positioned adapter fixture requires CBv2 caches")
            }
            _ = layer.updateAndAttend(
                queries: qkv, keys: qkv, values: qkv, scale: 1, sinks: nil)
        }
        return broadcast(
            MLXArray([Float(0), Float(1)]).reshaped(1, 1, 2),
            to: [batch, length, 2])
    }
}

private final class QwenLikePositionedModel: GemmaLikePositionlessModel,
    CBv2PositionedLanguageModelForwardable, CBv2PositionedEmbeddingForwardable,
    CBv2PositionAxisProviding
{
    private(set) var positionedShapes: [[Int]] = []

    override var supportsVisionSpanPrefill: Bool { false }
    override var supportsCausalVisionPrefill: Bool { true }
    var cbv2PositionAxisCount: Int? { 3 }

    func cbv2Forward(
        _ inputs: MLXArray, cache: [KVCache]?, positionIds: MLXArray?
    ) -> MLXArray {
        record(positionIds)
        return fixtureForward(inputs, cache: cache)
    }

    func embeddingForward(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray,
        cache: [KVCache]?,
        positionIds: MLXArray?
    ) -> MLXArray {
        record(positionIds)
        return fixtureForward(inputs, cache: cache)
    }

    private func record(_ positionIds: MLXArray?) {
        guard let positionIds else {
            Issue.record("Qwen-like positioned forward silently dropped position ids")
            return
        }
        positionedShapes.append(positionIds.shape)
    }
}

@Suite("CBv2 positioned language-model adapter")
struct CBv2PositionedAdapterTests {
    private let kinds = [
        CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
    ]

    private func engine(_ model: any LanguageModel) -> EngineV2 {
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        return EngineV2(
            model: adapter,
            layerKinds: kinds,
            backend: CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 1 << 20, kvDType: .float32)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 8,
                prefillChunkSize: 8,
                maxWaiting: 2),
            admissionConfig: .init(watermarkFraction: 0))
    }

    @Test("position state is typed-rejected when the wrapped model lacks positioned forwarding")
    func rejectsPositionStateBeforeGemmaLikeForward() async {
        let model = GemmaLikePositionlessModel()
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        #expect(!adapter.supportsPositionedForwarding)
        let engine = engine(model)
        let positions = CBv2PositionState(
            promptPositionIds: MLXArray.zeros([3, 1, 3], dtype: .int32),
            decodeDeltas: [0])
        let media = CBv2MultimodalInput(
            spans: [.init(tokenOffset: 1, length: 1)],
            attention: .bidirectionalSpans,
            positionState: positions
        ) { [MLXArray.ones([1, 1, 1])] }

        do {
            _ = try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(9101), promptTokens: [1, 7, 2],
                    maxTokens: 1, multimodal: media, positionState: positions))
            Issue.record("unsupported positioned request was admitted")
        } catch let error as CBv2MultimodalError {
            guard case .unsupportedModel(let detail) = error else {
                Issue.record("unexpected typed rejection: \(error)")
                await engine.shutdown()
                return
            }
            #expect(detail.contains("positioned model forwarding"))
        } catch {
            Issue.record("unexpected rejection type: \(error)")
        }

        #expect(model.forwardCount == 0)
        await engine.shutdown()
    }

    @Test("Qwen-like wrapped model executes positioned prefill and decode")
    func executesQwenLikePositionedRequest() async throws {
        let model = QwenLikePositionedModel()
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        #expect(adapter.supportsPositionedForwarding)
        let engine = engine(model)
        let positions = CBv2PositionState(
            promptPositionIds: MLXArray.zeros([3, 1, 3], dtype: .int32),
            decodeDeltas: [0])
        let media = CBv2MultimodalInput(
            spans: [.init(tokenOffset: 1, length: 1)],
            attention: .causal,
            positionState: positions
        ) { [MLXArray.ones([1, 1, 1])] }

        let stream = try engine.submit(
            CBv2Request(
                id: CBv2RequestID(9102), promptTokens: [1, 7, 2],
                maxTokens: 2, multimodal: media, positionState: positions))
        for await _ in stream {}
        await engine.shutdown()

        #expect(model.positionedShapes.contains([3, 1, 3]))
        #expect(model.positionedShapes.contains([3, 1, 1]))
        #expect(model.forwardCount >= 2)
    }
}
