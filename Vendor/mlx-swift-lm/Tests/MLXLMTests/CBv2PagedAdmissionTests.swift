import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("CBv2 paged physical admission", .serialized)
struct CBv2PagedAdmissionTests {
    private var tinyKind: CBv2LayerKind {
        .init(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
    }
    private func ledger(bytes: Int, auxiliary: Bool = true) -> AdmissionV2 {
        AdmissionV2(layerKinds: [tinyKind], bytesCapacity: bytes, config: .init(
            watermarkFraction: 0, elementBytes: 2, layerElementBytes: nil,
            fixedBytesPerRequest: auxiliary ? 8 : 0,
            auxiliaryBytesPerToken: auxiliary ? 2 : 0,
            auxiliaryTokenGranularity: 16, auxiliaryTokenAllocationPadding: 3),
            residency: CBv2PagedKVResidency(config: .init(pageSize: 16, capacityBytes: bytes, maxBufferLength: 1 << 20)))
    }

    @Test func physicalSlackTransfersToRequestsWithoutDiscountingAuxiliaryState() throws {
        let admission = ledger(bytes: 400)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 256)
        defer { physical.close() }
        // Per row: 64 target bytes + 8 fixed + 32*2 rounded assistant bytes.
        #expect(admission.allocatedBytes(forTokens: 16) == 136)
        try admission.reserve(id: .init(1), additionalTokens: 16)
        #expect(admission.bytesReserved == 328)
        try admission.reserve(id: .init(2), additionalTokens: 16)
        #expect(admission.bytesReserved == 400)
        #expect(admission.snapshot(activeRequests: 2, waitingRequests: 0, activeTokens: 32).kvBytesReserved == 400)
        #expect(throws: CBv2KVError.self) { try admission.reserveTransient(bytes: 1) }
        #expect(throws: CBv2KVError.self) { try physical.resize(to: 257) }
        #expect(physical.bytes == 256 && admission.bytesReserved == 400)
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 328)
        admission.releaseAll(id: .init(2))
        #expect(admission.bytesReserved == 256)
        physical.release(to: 0)
        #expect(admission.bytesReserved == 0)
    }

    @Test func shrinkDebtCanBeReusedButNotIncreasedOrCrossOffsetBetweenSlots() throws {
        let first = ledger(bytes: 256, auxiliary: false)
        let second = ledger(bytes: 256, auxiliary: false)
        let firstPhysical = first.bindBackendPhysicalFloor(initialBytes: 256)
        let secondPhysical = second.bindBackendPhysicalFloor(initialBytes: 256)
        defer { firstPhysical.close(); secondPhysical.close() }
        try first.reserve(id: .init(1), additionalTokens: 16)
        first.updateBytesCapacity(128)
        second.updateBytesCapacity(0)
        for raw in 2 ... 4 { try first.reserve(id: .init(UInt64(raw)), additionalTokens: 16) }
        #expect(first.bytesReserved == 256 && first.bytesCapacity == 128)
        #expect(second.bytesReserved == 256 && second.bytesCapacity == 0)
        #expect(throws: CBv2KVError.self) { try first.reserve(id: .init(5), additionalTokens: 1) }
        #expect(throws: CBv2KVError.self) { try first.reserveTransient(bytes: 1) }
        #expect(throws: CBv2KVError.self) { try second.reserveTransient(bytes: 1) }
        for raw in 1 ... 4 { first.releaseAll(id: .init(UInt64(raw))) }
        #expect(first.bytesReserved == 256) // ownership remains in its own pool
        firstPhysical.release(to: 0)
        #expect(first.bytesReserved == 0 && second.bytesReserved == 256)
        first.updateBytesCapacity(64)
        try first.reserve(id: .init(5), additionalTokens: 16)
        #expect(first.bytesReserved == 64)
        first.releaseAll(id: .init(5))
    }

    @Test func detachedOldRequestCannotReleaseReusedIDOrItsPhysicalOffset() throws {
        let admission = ledger(bytes: 400)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 256)
        defer { physical.close() }
        let id = CBv2RequestID(7)
        try admission.reserve(id: id, additionalTokens: 16)
        let old = admission.detachReservation(id: id)
        #expect(admission.bytesReserved == 328)
        try admission.reserve(id: id, additionalTokens: 16)
        #expect(admission.bytesReserved == 400)
        old.release()
        old.release()
        #expect(admission.bytesReserved == 328)
        admission.releaseAll(id: id)
        #expect(admission.bytesReserved == 256)
    }

    @Test func deadlineSimulationMatchesFloorTransfersAndRetainsPhysicalAfterRelease() throws {
        let admission = ledger(bytes: 400)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 256)
        defer { physical.close() }
        try admission.reserve(id: .init(1), additionalTokens: 16)
        let next = CBv2ProjectedCapacityReservation(id: .init(2), additionalTokens: 16, additionalBytes: 0)
        let scratch = CBv2ProjectedCapacityReservation(id: .init(3), additionalTokens: 0, additionalBytes: 1)
        #expect(admission.canGuarantee(projectedOperations: [.reserve(next)]))
        #expect(!admission.canGuarantee(projectedOperations: [.reserve(next), .reserve(scratch)]))
        #expect(admission.canGuarantee(projectedOperations: [.reserve(next), .reserveIfAvailable(scratch)]))
        #expect(admission.canGuarantee(projectedOperations: [.reserve(next), .unreserve(next), .reserve(scratch)]))
        #expect(admission.bytesReserved == 328)
        admission.updateBytesCapacity(300)
        #expect(!admission.canGuarantee(projectedOperations: [.release(.init(1)), .reserve(next)]))
        #expect(admission.canGuarantee(projectedOperations: [.release(.init(1))]))
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 256)
    }

    @Test func coldFullPromiseSurvivesChunkingAndSpeculationButRollsBackInitialWork() throws {
        let admission = ledger(bytes: 1 << 20)
        let physical = admission.bindBackendPhysicalFloor(initialBytes: 0)
        defer { physical.close() }
        let scheduler = SchedulerV2(config: .init(
            maxConcurrentRequests: 2, maxBatchedTokensPerStep: 16,
            prefillChunkSize: 16, maxConcurrentPartialPrefills: 1, maxWaiting: 4), capacity: admission)
        scheduler.reserveFullSequenceTokens = true
        let request = CBv2SchedFixtures.request(prompt: Array(repeating: 1, count: 33), maxTokens: 31)
        let rec = try scheduler.enqueue(request)
        guard case .bounded(let work, let operations) = scheduler.firstTokenWorkProjection(for: request.id) else {
            Issue.record("segmented cold request had no bounded deadline projection")
            return
        }
        #expect(work.prefillTokens == 33 && work.scheduledSteps == 3)
        let reservations = operations.compactMap { operation -> Int? in
            if case .reserve(let value) = operation { return value.additionalTokens }
            return nil
        }
        #expect(reservations == [64])
        #expect(admission.canGuarantee(projectedOperations: operations))
        let plan = scheduler.plan()
        #expect(plan.assignments.first?.numTokens == 16)
        #expect(rec.numComputedTokens == 16 && rec.fullSequenceCapacityTokens == 64)
        #expect(admission.bytesReserved == 424) // 256 KV + 8 fixed + 80*2 assistant
        scheduler.rollback(plan)
        #expect(rec.numComputedTokens == 0 && admission.bytesReserved == 0)
        _ = scheduler.plan()
        _ = scheduler.plan()
        #expect(rec.numComputedTokens == 32 && admission.bytesReserved == 424)
        scheduler.rollbackComputed(id: request.id, tokens: 3)
        #expect(rec.numComputedTokens == 29 && admission.bytesReserved == 424)
        scheduler.requestCancel(request.id)
        scheduler.finish(id: request.id, reason: .cancelled)
        admission.releaseAll(id: request.id)
        #expect(scheduler.runningCount == 0 && admission.bytesReserved == 0)
    }

    @Test func adoptedAndColdRowsPrepayIdenticalTargetAndAuxiliaryCapacity() throws {
        let admission = ledger(bytes: 1 << 20)
        let capability = CBv2PrefixReuseCapability.derive(layerKinds: [tinyKind], backend: .pagedFP16)
        let plan = try #require(capability.plan(
            matchedBoundary: 16, maximumSequenceLength: 64,
            nominalFullKVBytesPerToken: admission.fullKVBytesPerToken,
            reserveFullSequenceTokens: true))
        #expect(plan.capacityReservationTokens == 64 && plan.initialAdditionalCapacityBytes == 0)
        #expect(plan.matchedBoundary == 16 && plan.prefillTokensSaved == 16)
        try admission.reserve(id: .init(1), additionalTokens: 64)
        let cold = admission.bytesReserved
        admission.releaseAll(id: .init(1))
        try admission.reserve(id: .init(2), additionalTokens: plan.capacityReservationTokens,
                              additionalBytes: plan.initialAdditionalCapacityBytes)
        #expect(admission.bytesReserved == cold && cold == 424)
        #expect(CBv2ScheduledRequest.capacityTokensForChunk(
            start: 16, count: 16, plan: plan, fullSequenceTokens: 64) == 0)
        admission.releaseAll(id: .init(2))
    }

    @Test func qwenLedgerCapacityAcrossCoResidentDeviceEnvelopesHasNoEightGiBCap() throws {
        // Allocation-free accounting scenarios, not model execution or a
        // provider sizing recommendation. 36/64/128 GiB devices assign this
        // slot 8/16/32 GiB respectively; the rest belongs to other owners.
        let kinds = (0 ..< 16).map { _ in CBv2LayerKind(
            attention: .full, headDim: 256, kvHeads: 4, queryHeads: 24) }
        let convElements = 3 * (2 * 16 * 128 + 48 * 128)
        let fixedBytes = 4 * 48 * (convElements * 2 + 48 * 128 * 128 * 4)
        let auxiliaryBytes = 2 * 4 * 256 * 2 + 5_120 * 2 + 4
        let gib = 1 << 30, tokens = 32_768, rowPages = 16 * (32_768 / 16)
        let physicalPageBytes = 2 * 4 * 16 * 256 * 2
        let layout = try PagedKVSegmentLayout(
            pageBytes: physicalPageBytes, targetBytes: 64 << 20, maximumBufferBytes: 1 << 30)
        for (deviceGiB, slotGiB, expected) in [(36, 8, 2), (64, 16, 5), (128, 32, 10)] {
            #expect(slotGiB < deviceGiB)
            let admission = AdmissionV2(layerKinds: kinds, bytesCapacity: slotGiB * gib,
                config: .init(watermarkFraction: 0.05, elementBytes: 2, layerElementBytes: nil,
                              fixedBytesPerRequest: fixedBytes, auxiliaryBytesPerToken: auxiliaryBytes,
                              auxiliaryTokenGranularity: 256, auxiliaryTokenAllocationPadding: 4),
                residency: CBv2PagedKVResidency(config: .init(
                    capacityBytes: slotGiB * gib, maxBufferLength: 1 << 30)))
            let physical = admission.bindBackendPhysicalFloor(initialBytes: 0)
            var accepted = 0
            for raw in 1 ... 16 {
                let id = CBv2RequestID(UInt64(raw))
                do {
                    try admission.reserve(id: id, additionalTokens: tokens)
                    try physical.resize(to: #require(layout.allocationBytes(addingUsablePages: (accepted + 1) * rowPages)))
                    accepted += 1
                    if [1, 2, 4].contains(accepted) {
                        #expect(admission.bytesReserved <= admission.admissibleBytesCapacity)
                        #expect(physical.bytes >= accepted * tokens * 65_536)
                    }
                } catch let error as CBv2KVError {
                    guard case .capacityExhausted = error else { throw error }
                    admission.releaseAll(id: id)
                    break
                }
            }
            #expect(accepted == expected)
            if slotGiB >= 16 { #expect(physical.bytes > 8 * gib) }
            let before = admission.bytesReserved
            admission.releaseAll(id: .init(1))
            // Standing backing remains charged, but a replacement consumes
            // that same pool ownership and succeeds without double charging.
            try admission.reserve(id: .init(100), additionalTokens: tokens)
            #expect(admission.bytesReserved == before)
            for raw in 2 ... accepted { admission.releaseAll(id: .init(UInt64(raw))) }
            admission.releaseAll(id: .init(100))
            #expect(admission.bytesReserved == physical.bytes)
            physical.release(to: 0)
            physical.close()
            #expect(admission.bytesReserved == 0)
        }
    }

    private let pageBytes = 2 * 16 * 64 * 2
    private var kind: CBv2LayerKind {
        .init(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
    }
    private func nativeFixture() throws -> (PagedKVBackend, AdmissionV2) {
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 100 * pageBytes, dtype: .bfloat16, maxPrefillChunk: 64,
            nominalMaxSequenceLength: 2048, maxBufferLength: 1 << 20,
            segmentSizeBytes: 9 * pageBytes))
        let admission = AdmissionV2(layerKinds: [kind], bytesCapacity: 2 * (try Memory.allocationFootprintUpperBound(byteCount: 5 * pageBytes)),
            config: .init(watermarkFraction: 0, elementBytes: 2), residency: backend.kvResidency)
        backend.pool.bindAdmission(admission)
        return (backend, admission)
    }

    private final class NativeOwnerProbe: @unchecked Sendable {
        weak var array: MLXArray?
        var observedRefund = false
        var activeBytesBeforeRelease = 0
        var expectedPhysical = 0
    }

    @Test func prefillOnlyTerminalDropsLastNativeOwnerBeforeFloorRefund() throws {
        let (backend, admission) = try nativeFixture()
        let original = try #require(backend.pool.physicalLease)
        let probe = NativeOwnerProbe()
        // Observe the real ledger transfer without a production-only hook.
        // The wrapper keeps the original lease, and forwards every operation.
        backend.pool.physicalLease = CBv2BackendPhysicalLease(bytes: original.bytes, resize: { bytes in
            if bytes == 0 {
                #expect(probe.array == nil, "the last MLX owner must die before the physical refund")
                // This suite runs alone on the exclusive native GPU lane.
                // C++ graphs can retain buffers after a Swift wrapper dies;
                // verify actual active allocation release, not cache/RSS.
                #expect(Memory.activeMemory <= probe.activeBytesBeforeRelease - probe.expectedPhysical)
                #expect(admission.bytesReserved == probe.expectedPhysical)
                probe.observedRefund = true
            }
            try original.resize(to: bytes)
        }, onClose: { original.close() })
        try admission.reserve(id: .init(1), additionalTokens: 64)
        let state = try backend.makeSequenceState(layerKinds: [kind], promptLength: 16, maxLength: 64)
        probe.expectedPhysical = backend.bytesWired
        let row = try #require(state[0] as? PagedSequenceKV)
        let cache = try #require(backend.makeLayerCaches().first)
        cache.setRows([row])
        let query = MLXArray.ones([1, 2, 16, 64], dtype: .bfloat16)
        let keys = MLXArray.ones([1, 1, 16, 64], dtype: .bfloat16)
        let output = cache.updateAndAttend(
            queries: query, keys: keys, values: keys, scale: 0.125, sinks: nil)
        // Exactly the normal step roots: no row.snapshot, extra gather, or
        // test-only direct fence eval that could hide a missing innerState.
        eval([output] + cache.innerState())
        probe.array = backend.pool.group(row.groupKey).segments.values.first?.storage
        #expect(probe.array != nil)
        cache.setRows([])
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == probe.expectedPhysical)
        probe.activeBytesBeforeRelease = Memory.activeMemory
        backend.release(state)
        #expect(probe.observedRefund && probe.array == nil)
        #expect(backend.bytesWired == 0 && admission.bytesReserved == 0)
    }

    @Test func deadlineProbePricesPoisonBeforeAnyNativeAllocation() throws {
        let (backend, admission) = try nativeFixture()
        admission.updateBytesCapacity(4 * pageBytes)
        let request = CBv2ProjectedCapacityReservation(id: .init(1), additionalTokens: 64, additionalBytes: 0)
        #expect(admission.canEverFit(promptTokens: 32, maxTokens: 32))
        let overhead = try #require(backend.pool.minimumSegmentedOverhead(tokens: 64, layerKinds: [kind]))
        #expect(overhead == (try Memory.allocationFootprintUpperBound(byteCount: 5 * pageBytes)) - 4 * pageBytes)
        #expect(!admission.canEverFit(promptTokens: 32, maxTokens: 32, additionalBackendBytes: overhead))
        #expect(!admission.canGuarantee(projectedOperations: [.reserve(request)], projectedPhysicalBytes: {
            backend.pool.projectedPhysicalBytes(reservedTokens: $0, layerKinds: [self.kind])
        }))
        #expect(backend.bytesWired == 0 && admission.bytesReserved == 0)
        admission.updateBytesCapacity(4 * pageBytes + overhead)
        #expect(admission.canGuarantee(projectedOperations: [.reserve(request)], projectedPhysicalBytes: {
            backend.pool.projectedPhysicalBytes(reservedTokens: $0, layerKinds: [self.kind])
        }))
        #expect(backend.bytesWired == 0 && admission.bytesReserved == 0)
    }

    @Test func nativeGrowthReservesPeakBeforeAllocationAndRefusesWithoutSideEffects() throws {
        let (backend, admission) = try nativeFixture()
        defer { backend.pool.slabEval = { eval($0) } }
        try admission.reserve(id: .init(1), additionalTokens: 64)
        let first = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        let firstPhysical = backend.bytesWired
        let bound = try Memory.allocationFootprintUpperBound(byteCount: 5 * pageBytes)
        admission.updateBytesCapacity(firstPhysical + bound)
        #expect(admission.bytesReserved == firstPhysical)
        let scratch = try admission.reserveTransient(bytes: 1)
        try admission.reserve(id: .init(2), additionalTokens: 64)
        let heldBeforeFailure = admission.bytesReserved
        var allocations = 0
        backend.pool.slabEval = { _ in allocations += 1 }
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        }
        #expect(allocations == 0 && backend.bytesReserved == 4 * pageBytes)
        #expect(backend.bytesWired == firstPhysical && admission.bytesReserved == heldBeforeFailure)
        admission.releaseAll(id: .init(2))
        scratch.release()
        #expect(admission.bytesReserved == firstPhysical)
        try admission.reserve(id: .init(2), additionalTokens: 64)
        backend.pool.slabEval = { array in
            #expect(admission.bytesReserved == firstPhysical + bound)
            allocations += 1
            eval(array)
        }
        let retry = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        let secondPhysical = backend.bytesWired - firstPhysical
        #expect(allocations == 1 && admission.bytesReserved == backend.bytesWired)
        backend.release(first)
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == secondPhysical)
        backend.release(retry)
        admission.releaseAll(id: .init(2))
        #expect(backend.bytesWired == 0 && admission.bytesReserved == 0)
    }

    @Test(arguments: [false, true]) func failedPrivateGrowthDropsArraysBeforeFloorRefund(epochRace: Bool) throws {
        let (backend, admission) = try nativeFixture()
        defer { backend.pool.slabEval = { eval($0) } }
        try admission.reserve(id: .init(1), additionalTokens: 64)
        let first = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        let firstPhysical = backend.bytesWired
        let bound = try Memory.allocationFootprintUpperBound(byteCount: 5 * pageBytes)
        try admission.reserve(id: .init(2), additionalTokens: 64)
        weak var candidate: MLXArray?
        backend.pool.slabEval = { array in
            #expect(admission.bytesReserved == firstPhysical + bound)
            candidate = array
            eval(array)
            if epochRace { backend.updateBytesCapacity(99 * self.pageBytes) }
            else { throw CBv2KVError.capacityExhausted(needed: 1, available: 0) }
        }
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        }
        #expect(candidate == nil)
        #expect(backend.bytesWired == firstPhysical && backend.bytesReserved == 4 * pageBytes)
        // The request ledger still owns its uncommitted logical promise until
        // the caller rolls back; failed native growth released only its floor.
        #expect(admission.bytesReserved == max(firstPhysical, 8 * pageBytes))
        admission.releaseAll(id: .init(2))
        #expect(admission.bytesReserved == firstPhysical)
        backend.release(first)
        admission.releaseAll(id: .init(1))
        #expect(admission.bytesReserved == 0)
    }
}
