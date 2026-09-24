import Foundation
import MLX
import Testing

@_spi(Diagnostics) @testable import MLXLMCommon

@Suite("Teacher-forced recurrent peak admission", .serialized)
struct CBv2TeacherForcedCapacityTests {
    private final class RecurrentModel: CBv2RecurrentSteppableModel {
        let recurrentStateSpec: CBv2RecurrentStateSpec? = .init(layers: [.init(
            modelLayerIndex: 1, convShape: [1, 1, 8192], convDType: .float32,
            ssmShape: [1, 1, 1, 8192], ssmDType: .float32)])
        var cbv2Capabilities: CBv2ModelCapabilities {
            var value = CBv2ModelCapabilities.initialRecurrentTarget
            value.supportsPagedKV = true
            value.requiresNativePagedKV = true
            return value
        }
        private(set) var initialStates: [Bool] = []
        var observeReservation: (() -> Void)?

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            preconditionFailure("fixture requires request-owned recurrent state")
        }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray
        {
            observeReservation?()
            initialStates.append(recurrentState.allSatisfy {
                $0.inputState(modelLayerIndex: 1) == nil
            })
            let batch = tokens.dim(0), length = tokens.dim(1)
            let qkv = MLXArray.ones([batch, 1, length, 64], dtype: .bfloat16)
            var attended = qkv
            for cache in caches {
                attended = cache.updateAndAttend(
                    queries: qkv, keys: qkv, values: qkv, scale: 0.125, sinks: nil)
            }
            for evaluation in recurrentState {
                let old = evaluation.inputState(modelLayerIndex: 1)
                try! evaluation.stage(modelLayerIndex: 1,
                    conv: (old?.conv ?? MLXArray.zeros([1, 1, 8192])) + 1,
                    ssm: (old?.ssm ?? MLXArray.zeros([1, 1, 1, 8192])) + 1)
            }
            return broadcast(MLXArray([Float(0), Float(1)]).reshaped([1, 1, 2]),
                to: [batch, length, 2]) + sum(attended).asType(.float32) * Float(0)
        }
    }

    @Test(arguments: ["contiguous", "paged"])
    func tightBudgetRefusesBeforeForwardAndEveryExitRefunds(backendName: String) async throws {
        let kinds: [CBv2LayerKind] = [.init(
            attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)]
        let backend: any CBv2KVBackend
        let bank: CBv2LayerCacheBank
        if backendName == "paged" {
            let paged = try PagedKVBackend(layerKinds: kinds, config: .init(
                capacityBytes: 8 << 20, dtype: .bfloat16, maxPrefillChunk: 16,
                nominalMaxSequenceLength: 32, segmentSizeBytes: 32768,
                layerDTypes: [.bfloat16]))
            backend = paged
            bank = CBv2LayerCacheBank(caches: paged.makeLayerCaches())
        } else {
            backend = CBv2ContiguousKVBackend(config: .init(
                bytesCapacity: 8 << 20, kvDType: .bfloat16))
            bank = CBv2LayerCacheBank(layerKinds: kinds)
        }
        let model = RecurrentModel()
        let engine = EngineV2(model: model, layerKinds: kinds, backend: backend,
            cacheProvider: bank, sampler: CBv2GreedySampler(), schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 16,
                prefillChunkSize: 2, maxWaiting: 4, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0, elementBytes: 2))
        let admission = engine.admissionForTesting
        func requireRetired() {
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0)
            #expect(admission.bytesReserved == 0 && admission.transientBytesReserved == 0)
            #expect(admission.targetBytesReserved(partitionedBy: []).materialized == 0)
        }
        do {
            let generation = try model.recurrentStateSpec!.fixedBytesPerRequest()
            // Measure this backend's KV promise/backing without running a
            // model. The old one-generation helper accepts the budget below.
            let target = try engine.loopForTesting.onEngineQueueSync {
                var row = try backend.makeSequenceState(
                    layerKinds: kinds, promptLength: 3, maxLength: 5)
                let bytes = max(backend.bytesReserved, (backend as? PagedKVBackend)?.bytesWired ?? 0)
                backend.release(row)
                row.removeAll()
                return bytes
            }
            requireRetired()
            let tightBudget = target + generation + generation / 2
            #expect(target + generation <= tightBudget)
            #expect(target + 2 * generation > tightBudget)
            #expect(admission.fixedBytesPerRequest >= 3 * generation)
            engine.updateKVBytesCapacity(tightBudget)
            #expect(throws: CBv2KVError.self) {
                try engine.teacherForcedTop1(promptTokens: [0, 1, 0], continuation: [1, 1])
            }
            #expect(model.initialStates.isEmpty)
            requireRetired()

            engine.updateKVBytesCapacity(8 << 20)
            model.observeReservation = {
                #expect(admission.transientBytesReserved >= admission.fixedBytesPerRequest)
                #expect(admission.nonBackendBytesReserved >= admission.fixedBytesPerRequest)
                #expect(admission.targetBytesReserved(partitionedBy: []).materialized > 0)
            }
            #expect(try engine.teacherForcedTop1(
                promptTokens: [0, 1, 0], continuation: [1, 1]) == [1, 1])
            #expect(model.initialStates == [true, false, false])
            requireRetired()
            let invalid = try CBv2TeacherForcedScoreRequest(
                promptTokens: [0, 1, 0], continuation: [1, 1], vocabularySize: 3)
            #expect(throws: CBv2TeacherForcedScoreError.self) { try engine.teacherForcedScores(invalid) }
            requireRetired()
            #expect(try engine.teacherForcedTop1(
                promptTokens: [0, 1, 0], continuation: [1, 1]) == [1, 1])
            #expect(model.initialStates == [true, false, false, true, false, true, false, false])
            requireRetired()
            await engine.shutdown()
        } catch {
            await engine.shutdown()
            throw error
        }
    }

    @Test func privateReservationHonorsPhysicalFloorWatermarkAndExternalOverhead() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 512,
            config: .init(watermarkFraction: 0.125, elementBytes: 2, fixedBytesPerRequest: 300),
            externalReserveBytes: 64)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 80)
        defer { physical.close() }
        // The ordinary projection is 16 target + 300 fixed. The 80-byte
        // physical floor replaces target only: 380 charged of a 384 ceiling.
        let reservation = try admission.reserveUnscheduledRequest(maximumTokens: 4, minimumTargetBytes: 16)
        #expect(admission.bytesReserved == 380)
        #expect(admission.transientBytesReserved == 316)
        #expect(admission.targetBytesReserved(partitionedBy: []).materialized == 16)
        #expect(admission.nonBackendBytesReserved == 428) // 300 fixed + 64 external + 64 slack
        #expect(throws: CBv2KVError.self) { try admission.reserveTransient(bytes: 5) }
        reservation.release()
        reservation.release()
        #expect(admission.bytesReserved == 80 && admission.transientBytesReserved == 0)
        #expect(admission.targetBytesReserved(partitionedBy: []).materialized == 0)
        #expect(admission.nonBackendBytesReserved == 144)
    }
}
