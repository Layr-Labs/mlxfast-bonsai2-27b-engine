import Foundation
import MLX
import MLXNN
import Testing
@_spi(Diagnostics) @testable import MLXLMCommon

@Suite("CBv2 bounded teacher-forced numerical scores", .serialized)
struct CBv2TeacherForcedScoreDiagnosticTests {
    @Test(arguments: [DType.float16, .bfloat16, .float32])
    func independentNativeValueReferenceIncludesStridesAndTies(dtype: DType) throws {
        let values: [Float] = [0.1, -0.3, 1.2, 1.2, 2.7, -4]
        let source = MLXArray(values.flatMap { [$0, Float(999)] })
            .asType(dtype)
        let logits = asStrided(source, [values.count], strides: [2])
        eval(logits)
        #expect(logits.strides[0] == 2)
        let observed = logits.asArray(Float.self)
        let packet = try CBv2TeacherForcedScoreCollector.Packet(
            logits: logits, index: 2, contextLength: 39, forcedToken: 1,
            vocabularySize: values.count)
        eval(packet.evaluationTargets)
        let score = packet.materialize()
        let maximum = Double(observed.max()!)
        let normalizer = maximum + log(observed.reduce(0.0) { $0 + exp(Double($1) - maximum) })
        #expect(abs(Double(Float(bitPattern: score.logSumExpBits)) - normalizer) < 0.00001)
        #expect(abs(Double(Float(bitPattern: score.nllBits)) - (normalizer - Double(observed[1]))) < 0.00001)
        #expect(abs(Double(Float(bitPattern: score.forcedLogProbabilityBits)) + normalizer - Double(observed[1])) < 0.00001)
        #expect(score.topTwoIDs == [4, 2])
        #expect(score.argMaxID == 4)
        #expect(score.topTwoValueBits == [observed[4].bitPattern, observed[2].bitPattern])
        #expect(score.forcedTokenValueBits == observed[1].bitPattern)
        #expect(score.topTwoMarginBits == (observed[4] - observed[2]).bitPattern)
        #expect(score.logitDType == String(describing: dtype))
        #expect(score.index == 2 && score.contextLength == 39 && score.forcedToken == 1)
        #expect(score.isFinite)
    }

    @Test func nonfiniteEvidenceRemainsJSONSafeAndInconclusive() throws {
        let vectors: [[Float]] = [[.nan, 1, 2], [.infinity, 1, 2], [-.infinity, 1, 2]]
        for values in vectors {
            let packet = try CBv2TeacherForcedScoreCollector.Packet(
                logits: MLXArray(values), index: 0, contextLength: 2,
                forcedToken: 1, vocabularySize: values.count)
            eval(packet.evaluationTargets)
            let score = packet.materialize()
            #expect(!score.isFinite)
            #expect(score.nanCount + score.infiniteCount == 1)
            let encoded = try JSONEncoder().encode(score)
            #expect(try JSONDecoder().decode(CBv2TeacherForcedScore.self, from: encoded) == score)
        }
    }

    @Test func stableNormalizerHandlesLargeFiniteLogitsAndInvalidForcedIndex() throws {
        let logits = MLXArray([Float(1000), 1001, -1000])
        let packet = try CBv2TeacherForcedScoreCollector.Packet(logits: logits,
            index: 0, contextLength: 1, forcedToken: 0, vocabularySize: 3)
        eval(packet.evaluationTargets)
        let score = packet.materialize()
        let expectedNLL = 1 + log(1 + exp(-1.0))
        #expect(score.isFinite)
        #expect(abs(Double(Float(bitPattern: score.nllBits)) - expectedNLL) < 0.0001)
        #expect(score.topTwoIDs == [1, 0])
        for forced in [-1, 3] {
            #expect(throws: CBv2TeacherForcedScoreError.self) {
                try CBv2TeacherForcedScoreCollector.Packet(logits: logits,
                    index: 0, contextLength: 1, forcedToken: forced, vocabularySize: 3)
            }
        }
    }

    @Test func completedCollectorRetainsDisagreementForCallerClassification() throws {
        let request = try CBv2TeacherForcedScoreRequest(
            promptTokens: [1], continuation: [1], vocabularySize: 3)
        let collector = CBv2TeacherForcedScoreCollector(request)
        eval(try collector.capture(logits: MLXArray([Float(0), 1, 2]), index: 0))
        // Deliberately inconsistent normal result: retain both observations.
        try collector.finish(top1: [0])
        let snapshot = try #require(collector.snapshot)
        #expect(snapshot.top1 == [0] && snapshot.records.map(\.argMaxID) == [2])
        #expect(snapshot.records.count == 1 && snapshot.allFinite)
    }

    @Test func boundsAndTokenIDsRefuseBeforeCollectorOrStateAllocation() {
        for (prompt, continuation, vocabulary) in [
            ([], [1], 8), ([1], [], 8), ([1], [8], 8), ([-1], [1], 8),
            ([1], [-1], 8), ([1], [1], 1), ([1], [1], 1_048_577),
            ([1], Array(repeating: 1, count: 257), 8),
            (Array(repeating: 1, count: 32769), [1], 8)
        ] {
            #expect(throws: CBv2TeacherForcedScoreError.self) {
                try CBv2TeacherForcedScoreRequest(promptTokens: prompt,
                    continuation: continuation, vocabularySize: vocabulary)
            }
        }
    }

    @Test(arguments: ["contiguous", "paged"])
    func plainInstrumentedRepeatAndFailureRetirementUseSameEngineForwards(backendName: String) async throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE, headDim: 64)
        model.update(parameters: ModuleParameters.unflattened(
            model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        eval(model)
        let backend: any CBv2KVBackend
        let bank: CBv2LayerCacheBank
        if backendName == "paged" {
            let paged = try PagedKVBackend(layerKinds: model.layerKinds, config: .init(
                capacityBytes: 16 << 20, dtype: .bfloat16, maxPrefillChunk: 16,
                nominalMaxSequenceLength: 128, segmentSizeBytes: 1 << 18))
            backend = paged
            bank = CBv2LayerCacheBank(caches: paged.makeLayerCaches())
        } else {
            backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 16 << 20))
            bank = CBv2LayerCacheBank(layerKinds: model.layerKinds)
        }
        let engine = EngineV2(model: model, layerKinds: model.layerKinds, backend: backend,
            cacheProvider: bank, sampler: CBv2GreedySampler(), schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 16, maxWaiting: 4, enablePrefixCache: false))
        do {
            let prompt = makePromptTokens(length: 37, seed: 987)
            let continuation = (0..<12).map { ($0 * 7 + 3) % TinyTestModelConfig().vocabSize }
            let request = try CBv2TeacherForcedScoreRequest(promptTokens: prompt,
                continuation: continuation, vocabularySize: TinyTestModelConfig().vocabSize)
            let before = engine.teacherForcedScoringActivity()
            let plain = try engine.teacherForcedTop1(promptTokens: prompt, continuation: continuation)
            let a = engine.teacherForcedScoringActivity()
            let first = try engine.teacherForcedScores(request)
            let b = engine.teacherForcedScoringActivity()
            let repeated = try engine.teacherForcedScores(request)
            let c = engine.teacherForcedScoringActivity()
            #expect(first.top1 == plain && repeated == first && first.allFinite)
            #expect(first.records.map(\.argMaxID) == plain)
            #expect(first.records.map(\.contextLength) == Array(37..<49))
            #expect(first.records.map(\.forcedToken) == continuation)
            #expect(first.records.allSatisfy { $0.logitDType == "bfloat16" })
            for (start, end) in [(before, a), (a, b), (b, c)] {
                #expect(end.prefillChunksExecuted - start.prefillChunksExecuted == 3)
                #expect(end.decodeForwardsExecuted - start.decodeForwardsExecuted == 11)
            }
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0)
            let wrongVocabulary = try CBv2TeacherForcedScoreRequest(promptTokens: prompt,
                continuation: continuation, vocabularySize: TinyTestModelConfig().vocabSize + 1)
            #expect(throws: CBv2TeacherForcedScoreError.self) {
                try engine.teacherForcedScores(wrongVocabulary)
            }
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0)
            let recovered = try engine.teacherForcedTop1(promptTokens: prompt, continuation: continuation)
            #expect(recovered == plain)
            await engine.shutdown()
            #expect(throws: CBv2TeacherForcingError.self) { try engine.teacherForcedScores(request) }
        } catch {
            await engine.shutdown()
            throw error
        }
    }
}
