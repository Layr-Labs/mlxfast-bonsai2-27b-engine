import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXLLM
@_spi(Diagnostics) @testable import MLXLMCommon

@Suite("Teacher-forced real Qwen hybrid state", .serialized)
struct CBv2TeacherForcedRecurrentTests {
    private final class ObservedModel: CBv2RecurrentPrefillSteppableModel,
        CBv2PositionedRecurrentSteppableModel, CBv2PositionAxisProviding
    {
        struct Call {
            let prefill: Bool
            let length: Int
            let initialState: Bool
            let tokens: [Int]
        }
        let base: CBv2SteppableLanguageModelAdapter
        private let lock = NSLock()
        private var recorded: [Call] = []
        private var plainCalls = 0

        init(_ base: CBv2SteppableLanguageModelAdapter) { self.base = base }
        var cbv2Capabilities: CBv2ModelCapabilities { base.cbv2Capabilities }
        var recurrentStateSpec: CBv2RecurrentStateSpec? { base.recurrentStateSpec }
        var cbv2PositionAxisCount: Int? { base.cbv2PositionAxisCount }
        var supportsPositionedForwarding: Bool { base.supportsPositionedForwarding }

        func takeCalls() -> (calls: [Call], plain: Int) {
            lock.withLock {
                let result = (recorded, plainCalls)
                recorded.removeAll()
                plainCalls = 0
                return result
            }
        }

        private func record(_ tokens: MLXArray, _ states: [CBv2RecurrentStateEvaluation], prefill: Bool) {
            let initial = states.allSatisfy { state in
                recurrentStateSpec!.modelLayerIndices.allSatisfy { state.inputState(modelLayerIndex: $0) == nil }
            }
            let tokenIDs = tokens.asArray(Int32.self).map(Int.init)
            lock.withLock { recorded.append(Call(prefill: prefill, length: tokens.dim(1), initialState: initial, tokens: tokenIDs)) }
        }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            lock.withLock { plainCalls += 1 }
            return MLXArray.zeros([tokens.dim(0), tokens.dim(1), 64])
        }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray {
            record(tokens, recurrentState, prefill: false)
            return base.forward(tokens: tokens, caches: caches, recurrentState: recurrentState)
        }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?) -> MLXArray {
            record(tokens, recurrentState, prefill: false)
            return base.forward(tokens: tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
        }

        func recurrentPrefill(tokens: MLXArray, inputEmbeddings: MLXArray?, caches: [CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
            requirement: CBv2PrefillRequirement) -> MLXArray {
            record(tokens, recurrentState, prefill: true)
            return base.recurrentPrefill(tokens: tokens, inputEmbeddings: inputEmbeddings, caches: caches,
                recurrentState: recurrentState, positionIds: positionIds, requirement: requirement)
        }
    }

    private func target(experts: Int) throws -> Qwen35TextModel {
        let json = """
        {"model_type":"qwen3_5_moe_text","hidden_size":64,"num_hidden_layers":4,
         "intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,
         "linear_num_value_heads":1,"linear_num_key_heads":1,"linear_key_head_dim":64,
         "linear_value_head_dim":64,"linear_conv_kernel_dim":4,"full_attention_interval":2,
         "vocab_size":64,"num_experts":\(experts),"num_experts_per_tok":\(experts > 0 ? 2 : 0),
         "moe_intermediate_size":32,"shared_expert_intermediate_size":32,"norm_topk_prob":true}
        """
        MLXRandom.seed(7029)
        let model = Qwen35TextModel(try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8)))
        model.update(parameters: ModuleParameters.unflattened(
            model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        eval(model)
        return model
    }

    private func engine(_ target: Qwen35TextModel, backendName: String, observed: ObservedModel? = nil)
        throws -> (EngineV2, any CBv2KVBackend)
    {
        let kinds = target.cbv2LayerKinds
        let backend: any CBv2KVBackend
        let caches: [any CBv2AttendingLayerCache]
        if backendName == "paged" {
            let nativeTypes = try CBv2NativeKVTypeProbe.run(
                model: CBv2SteppableLanguageModelAdapter(target), layerKinds: kinds,
                caches: target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }).layerDTypes
            let paged = try PagedKVBackend(layerKinds: kinds, config: .init(capacityBytes: 64 << 20,
                dtype: .bfloat16, maxPrefillChunk: 16, nominalMaxSequenceLength: 128,
                segmentSizeBytes: 1 << 18, layerDTypes: nativeTypes))
            backend = paged
            let storage = paged.makeLayerCaches()
            let indices = Dictionary(uniqueKeysWithValues: kinds.enumerated().map {
                ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
            })
            caches = target.newCacheV2 { index, _ in storage[indices[index]!] }
        } else {
            backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 64 << 20, kvDType: .bfloat16))
            caches = target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }
        }
        let model: any CBv2SteppableModel
        if let observed { model = observed }
        else { model = CBv2SteppableLanguageModelAdapter(target) }
        return (EngineV2(model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: 64,
                prefillChunkSize: 16, maxWaiting: 4, enablePrefixCache: false)), backend)
    }

    private func requireStatefulCalls(_ observed: ObservedModel, prompt: [Int], decodedTokens: [Int]) {
        let result = observed.takeCalls()
        #expect(result.plain == 0)
        #expect(result.calls.count == 3 + decodedTokens.count)
        #expect(result.calls.prefix(3).map(\.length) == [16, 16, 5])
        #expect(result.calls.prefix(3).allSatisfy { $0.prefill })
        #expect(result.calls.dropFirst(3).allSatisfy { !$0.prefill && $0.length == 1 })
        #expect(result.calls.prefix(3).flatMap(\.tokens) == prompt)
        #expect(result.calls.dropFirst(3).flatMap(\.tokens) == decodedTokens)
        #expect(result.calls.first?.initialState == true)
        #expect(result.calls.dropFirst().allSatisfy { !$0.initialState })
    }

    @Test(arguments: ["contiguous", "paged"], [0, 4])
    func realHybridPlainDiagnosticRepeatAndFailureRetirement(backendName: String, experts: Int) async throws {
        let model = try target(experts: experts)
        let prompt = (0..<37).map { ($0 * 7 + 3) % 64 }
        let (ordinary, _) = try engine(model, backendName: backendName)
        var greedy: [Int] = []
        do {
            var finish: CBv2FinishReason?
            for await event in try ordinary.submit(.init(id: .init(901), promptTokens: prompt,
                sampling: .init(temperature: 0), maxTokens: 8, prefixCacheEnabled: false)) {
                switch event {
                case .delta(_, let tokens, _): greedy += tokens
                case .finished(let reason, _): finish = reason
                }
            }
            #expect(finish == .length && greedy.count == 8)
            await ordinary.shutdown()
        } catch {
            await ordinary.shutdown()
            throw error
        }

        let observed = ObservedModel(CBv2SteppableLanguageModelAdapter(model))
        let (scorer, backend) = try engine(model, backendName: backendName, observed: observed)
        do {
            #expect(try scorer.teacherForcedTop1(promptTokens: prompt, continuation: greedy) == greedy)
            requireStatefulCalls(observed, prompt: prompt, decodedTokens: Array(greedy.dropLast()))
            let forced = (0..<12).map { ($0 * 11 + 5) % 64 }
            let request = try CBv2TeacherForcedScoreRequest(promptTokens: prompt, continuation: forced, vocabularySize: 64)
            let before = scorer.teacherForcedScoringActivity()
            let plain = try scorer.teacherForcedTop1(promptTokens: prompt, continuation: forced)
            requireStatefulCalls(observed, prompt: prompt, decodedTokens: Array(forced.dropLast()))
            let first = try scorer.teacherForcedScores(request)
            requireStatefulCalls(observed, prompt: prompt, decodedTokens: Array(forced.dropLast()))
            let repeated = try scorer.teacherForcedScores(request)
            requireStatefulCalls(observed, prompt: prompt, decodedTokens: Array(forced.dropLast()))
            #expect(first.top1 == plain && repeated == first && first.allFinite)
            #expect(first.records.map(\.forcedToken) == forced)
            #expect(first.records.map(\.contextLength) == Array(37..<49))
            let after = scorer.teacherForcedScoringActivity()
            #expect(after.prefillChunksExecuted - before.prefillChunksExecuted == 9)
            #expect(after.decodeForwardsExecuted - before.decodeForwardsExecuted == 33)
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0)
            let invalid = try CBv2TeacherForcedScoreRequest(promptTokens: prompt, continuation: forced, vocabularySize: 65)
            #expect(throws: CBv2TeacherForcedScoreError.self) { try scorer.teacherForcedScores(invalid) }
            requireStatefulCalls(observed, prompt: prompt, decodedTokens: [])
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0)
            #expect(try scorer.teacherForcedTop1(promptTokens: prompt, continuation: forced) == plain)
            requireStatefulCalls(observed, prompt: prompt, decodedTokens: Array(forced.dropLast()))
            await scorer.shutdown()
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0)
        } catch {
            await scorer.shutdown()
            throw error
        }
    }
}
