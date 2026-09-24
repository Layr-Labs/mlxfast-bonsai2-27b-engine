import Foundation
import MLX

extension PagedKVPool {
    func bindAdmission(_ admission: AdmissionV2) {
        precondition(segmentGrant != nil && physicalLease == nil)
        memoryAdmission = admission
        physicalLease = admission.bindBackendPhysicalFloor(initialBytes: bytesMaterialized)
    }

    struct SegmentGrowth {
        let group: PagedKVGroup
        let plan: PagedKVGroup.GrowthPlan
    }

    /// Build complete group plans using one aggregate grant. This stage may
    /// allocate host metadata, but creates no GPU buffers or visible pages.
    private func planSegments(
        additional needs: [PagedKVGroupKey: Int] = [:], eager: Bool = false,
        grant: PagedKVGrant.Snapshot
    ) throws -> [SegmentGrowth] {
        var logical = bytesReserved
        let reuseCeiling = max(grant.bytes, bytesMaterialized)
        for (key, pages) in needs {
            guard pages >= 0 else {
                throw CBv2KVError.backendIneligible(reason: "negative paged reservation")
            }
            let (extra, multiplyOverflow) = pages.multipliedReportingOverflow(by: group(key).pageBytes)
            let (total, addOverflow) = logical.addingReportingOverflow(extra)
            guard !multiplyOverflow, !addOverflow, total <= reuseCeiling else {
                throw CBv2KVError.capacityExhausted(needed: multiplyOverflow || addOverflow ? Int.max : extra, available: max(0, reuseCeiling - logical))
            }
            logical = total
        }
        var result: [SegmentGrowth] = []
        for key in groupKeys {
            let g = group(key)
            var target = g.pagesReserved + needs[key, default: 0]
            if eager {
                let product = UInt64(grant.bytes).multipliedFullWidth(by: UInt64(groupDemandBytes[key]!))
                let bytes = UInt64(totalDemandBytes).dividingFullWidth(product).quotient
                let pages = Int(bytes) / g.pageBytes
                target = max(target, g.segmentLayout!.usablePages(fittingPhysicalPages: pages))
            }
            result.append(SegmentGrowth(group: g, plan: try g.planGrowth(usablePages: target)))
        }
        return result
    }

    private func physicalBytes(_ plans: [SegmentGrowth]) throws -> Int {
        var sum = 0
        for item in plans {
            let (next, overflow) = sum.addingReportingOverflow(item.plan.physicalBytes)
            guard !overflow else {
                throw CBv2KVError.backendIneligible(reason: "paged physical byte overflow")
            }
            sum = next
        }
        return sum
    }

    private func requestedPages(
        tokens: some Sequence<Int>, layerKinds: [CBv2LayerKind]
    ) -> [PagedKVGroupKey: Int]? {
        let residency = CBv2PagedKVResidency(config: config)
        var requested: [PagedKVGroupKey: Int] = [:]
        for count in tokens where count > 0 {
            for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
                guard let rows = residency.residentRows(layer: kind, tokens: count) else { return nil }
                let key = groupKey(forLayer: index)
                let (total, overflow) = requested[key, default: 0].addingReportingOverflow(rows / config.pageSize)
                guard !overflow else { return nil }
                requested[key] = total
            }
        }
        return requested
    }

    /// Any-thread submit probe over immutable configuration. A request that
    /// fits nominal pages but cannot fit its minimum poison overhead must not
    /// enter an endless allocate/preempt loop. This allocates no page map.
    func minimumSegmentedOverhead(tokens: Int, layerKinds: [CBv2LayerKind]) -> Int? {
        guard segmentGrant != nil, let pages = requestedPages(tokens: [tokens], layerKinds: layerKinds) else { return nil }
        var overhead = 0
        for (key, count) in pages {
            // Geometry was checked at construction; the token-dependent
            // products below still use checked arithmetic.
            let pageBytes = 2 * key.kvHeads * config.pageSize * key.headDim * key.dtype.size
            guard let layout = try? PagedKVSegmentLayout(
                pageBytes: pageBytes,
                targetBytes: config.segmentSizeBytes ?? PagedKVSegmentLayout.defaultTargetBytes,
                maximumBufferBytes: config.maxBufferLength,
                maximumAddressPages: Int(Int32.max) / config.pageSize),
                let physical = layout.allocationBytes(addingUsablePages: count)
            else { return nil }
            let (nominal, multiplyOverflow) = count.multipliedReportingOverflow(by: pageBytes)
            guard !multiplyOverflow, physical >= nominal else { return nil }
            let (next, addOverflow) = overhead.addingReportingOverflow(physical - nominal)
            guard !addOverflow else { return nil }
            overhead = next
        }
        return overhead
    }

    /// Engine-queue-only deadline probe. Charge projected row promises and
    /// retain existing backing; a projected release does not promise that free
    /// segments have retired. No GPU work or grant-sized metadata allocation.
    func projectedPhysicalBytes(
        reservedTokens: [CBv2RequestID: Int], layerKinds: [CBv2LayerKind]
    ) -> Int? {
        guard segmentGrant != nil,
            let requested = requestedPages(tokens: reservedTokens.values, layerKinds: layerKinds)
        else { return nil }
        var physical = bytesMaterialized
        for (key, pages) in requested {
            let group = group(key)
            let extra = max(0, pages - group.committedUsablePages)
            guard let additional = group.segmentLayout?.allocationBytes(addingUsablePages: extra) else { return nil }
            let (next, overflow) = physical.addingReportingOverflow(additional)
            guard !overflow else { return nil }
            physical = next
        }
        // Existing debt can be reused; new physical growth must fit the grant.
        return physical <= max(segmentGrant!.snapshot().bytes, bytesMaterialized) ? physical : nil
    }

    /// Reserve logical pages against the exact physical growth plan. A later
    /// commitment rechecks the grant before publishing any native buffers.
    func reserveSegments(_ needs: [PagedKVGroupKey: Int]) throws {
        let grant = segmentGrant!.snapshot()
        let plans: [SegmentGrowth]
        do { plans = try planSegments(additional: needs, grant: grant) }
        catch {
            if let error = error as? CBv2KVError, case .capacityExhausted = error {
                PagedKVStorageTelemetry.increment(&storageTelemetry.grantRefusals)
            }
            throw error
        }
        let physical = try physicalBytes(plans)
        let accepted = segmentGrant!.publish(
            expected: grant, physicalBytes: physical, existingPhysicalBytes: bytesMaterialized) {
            for (key, pages) in needs { group(key).pagesReserved += pages }
        }
        storageTelemetry.record(accepted)
        guard accepted == .installed else {
            throw CBv2KVError.capacityExhausted(
                needed: physical, available: segmentGrant!.snapshot().bytes)
        }
    }

    func materializeSegments(all: Bool) throws {
        let grant = segmentGrant!.snapshot()
        let plans = try planSegments(eager: all, grant: grant)
        // Existing promises survive a shrink. With no allocation to publish,
        // their committed backing remains valid even above the new grant.
        guard plans.contains(where: { !$0.plan.segmentIDs.isEmpty }) else { return }
        let physical = try physicalBytes(plans)
        guard physical <= grant.bytes else {
            PagedKVStorageTelemetry.increment(&storageTelemetry.grantRefusals)
            throw CBv2KVError.capacityExhausted(needed: physical, available: grant.bytes)
        }
        let previousPhysicalBytes = bytesMaterialized
        // The plan contains all old backing plus every private new segment.
        // Admission charges the peak before the first native allocation.
        do { try physicalLease?.resize(to: physical) }
        catch {
            PagedKVStorageTelemetry.increment(&storageTelemetry.admissionRefusals)
            throw error
        }
        var prepared: [(PagedKVGroup, PagedKVGroup.PreparedGrowth)] = []
        var preparing = true
        do {
            for item in plans where !item.plan.segmentIDs.isEmpty {
                prepared.append((item.group, try item.group.prepareGrowth(
                    item.plan, evaluate: slabEval, admission: memoryAdmission)))
            }
            preparing = false
            let actual = prepared.reduce(bytesMaterialized) { total, entry in
                total - entry.0.committedSegmentBytes
                    + entry.1.segments.values.reduce(0) { $0 + $1.allocatedBytes }
            }
            let accepted = segmentGrant!.publish(expected: grant, physicalBytes: actual) {
                for (group, replacement) in prepared { group.installGrowth(replacement) }
            }
            storageTelemetry.record(accepted)
            guard accepted == .installed else {
                throw CBv2KVError.capacityExhausted(
                    needed: actual, available: segmentGrant!.snapshot().bytes)
            }
            physicalLease?.release(to: actual)
            storageTelemetry.recordSettlement(bound: physical, actual: actual)
        } catch {
            if preparing { PagedKVStorageTelemetry.increment(&storageTelemetry.allocationFailures) }
            // Failed preparation has already destroyed its local arrays. Drop
            // all earlier groups before refunding the private-allocation peak.
            prepared.removeAll()
            physicalLease?.release(to: previousPhysicalBytes)
            throw error
        }
    }

    /// Capture on the engine queue, then hand the immutable value to gauges.
    var segmentStorageSnapshot: PagedKVStorageSnapshot? {
        guard let segmentGrant else { return nil }
        let capturedAt = storageTelemetry.capture()
        let accounting = physicalLease?.accountingSnapshot
        let committed = groups.values.reduce(0) { $0 + $1.committedSegmentBytes }
        let poison = groups.values.reduce(0) { $0 + $1.segments.count * $1.pageBytes }
        let logical = groups.values.reduce(0) { $0 + $1.committedLogicalBytes }
        return PagedKVStorageSnapshot(
            generation: storageTelemetry.generation,
            captureSequence: storageTelemetry.captureSequence,
            capturedUptimeNanoseconds: capturedAt,
            grantBytes: segmentGrant.snapshot().bytes, committedBytes: committed,
            reservedPageBytes: bytesReserved, livePageBytes: bytesInUse,
            poisonBytes: poison, slackBytes: max(0, logical - poison - bytesReserved),
            allocatorPaddingBytes: max(0, committed - logical),
            lastAllocationAllowanceBytes: storageTelemetry.lastAllocationAllowanceBytes,
            segmentCount: groups.values.reduce(0) { $0 + $1.segments.count },
            addressPages: groups.values.reduce(0) { $0 + $1.pageCount },
            nominalKVBytes: accounting?.nominalKVBytes,
            physicalFloorOverheadBytes: accounting?.physicalFloorOverheadBytes,
            allocationFailures: storageTelemetry.allocationFailures,
            admissionRefusals: storageTelemetry.admissionRefusals,
            grantRefusals: storageTelemetry.grantRefusals,
            grantEpochRetries: storageTelemetry.grantEpochRetries)
    }
}
