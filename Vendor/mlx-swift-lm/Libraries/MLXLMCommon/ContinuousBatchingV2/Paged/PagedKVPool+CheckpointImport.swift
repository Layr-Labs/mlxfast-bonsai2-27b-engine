import MLX

extension PagedKVPool {
    private struct CheckpointGroupImport {
        let group: PagedKVGroup
        let plan: PagedKVGroup.ImportPlan
    }

    /// Queue-only attachment of evaluated M-page backing. This consumes the
    /// frame on every path; failed imports stay cold and require fresh staging.
    /// Full and historical-window owners attach once; canonical borrowers
    /// acquire no backing. Loaded codec validation precedes this transaction.
    func importCheckpoint(
        _ frame: CBv2PagedCheckpointFrame, admission: AdmissionV2,
        requestID: CBv2RequestID, layerKinds: [CBv2LayerKind], maximumTokens: Int
    ) throws -> CBv2PagedCheckpointAdoption {
        let owner = try frame.consume()
        defer { owner.close() }
        let storage = owner.storage
        let checkpoint = storage.plan
        guard let segmentGrant, let physicalLease,
            checkpoint.pageSize == config.pageSize,
            layerKinds == self.layerKinds, layerKinds.count == layerDTypes.count,
            checkpoint.position <= maximumTokens,
            checkpoint.ownerMap.count == layerKinds.count,
            checkpoint.layers.map(\.modelIndex) == checkpoint.ownerMap.indices.filter({ checkpoint.ownerMap[$0] == $0 }),
            checkpoint.groups.count == groups.count
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        var needs: [PagedKVGroupKey: Int] = [:]
        var rowNeeds: [Int] = []
        for layer in checkpoint.layers {
            let index = layer.modelIndex
            let kind = layerKinds[index]
            guard kind.sharesKVWithLayer == nil,
                layer.key == groupKey(forLayer: index),
                layer.key == PagedKVGroupKey(kind, dtype: layerDTypes[index], separateWindow: true)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            let pages = Self.pageDemand(kind: kind, maxLength: maximumTokens, config: config)
            guard layer.pageCount <= pages else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            let key = groupKey(forLayer: index)
            let (sum, overflow) = needs[key, default: 0].addingReportingOverflow(pages)
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            needs[key] = sum
            rowNeeds.append(pages)
        }
        // Borrowers never acquire a row or pages. This is the same canonical
        // mapping authenticated by the loaded codec before staging.
        for (index, owner) in checkpoint.ownerMap.enumerated() where owner != index {
            guard owner >= 0, owner < index, checkpoint.ownerMap[owner] == owner,
                  layerKinds[index].sharesKVWithLayer == owner,
                  groupKey(forLayer: index) == groupKey(forLayer: owner)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        }
        let grant = segmentGrant.snapshot()
        var plans: [CheckpointGroupImport] = []
        var physical = 0
        for key in groupKeys {
            guard let source = storage.groups[key] else { throw CBv2CompleteCheckpointError.closed }
            let group = group(key)
            let plan = try group.planImport(source, additionalReservedPages: needs[key, default: 0])
            let (sum, overflow) = physical.addingReportingOverflow(plan.physicalBytes)
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            physical = sum
            plans.append(CheckpointGroupImport(group: group, plan: plan))
        }
        guard physical <= grant.bytes else {
            PagedKVStorageTelemetry.increment(&storageTelemetry.grantRefusals)
            throw CBv2KVError.capacityExhausted(needed: physical, available: grant.bytes)
        }
        guard storage.groups.values.allSatisfy({ group in
            group.segments.values.allSatisfy { $0.backing.belongs(to: admission) }
        }) else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let reservation: CBv2CheckpointAdoptionReservation
        do {
            reservation = try physicalLease.transferCheckpoint(to: physical, admission: admission) { previous in
                try admission.transferCheckpointStage(
                    owner.lease, requestID: requestID, maximumTokens: maximumTokens,
                    previousPhysicalBytes: previous, physicalBytes: physical)
            }
        } catch {
            PagedKVStorageTelemetry.increment(&storageTelemetry.admissionRefusals)
            throw error
        }
        var prepared: [(PagedKVGroup, PagedKVGroup.PreparedImport)] = []
        var rows: [PagedSequenceKV] = []
        var tables: [[Int32]] = []
        var allocating = true
        do {
            for source in storage.groups.values {
                for segment in source.segments.values {
                    try segment.backing.cover(using: admission, bytes: segment.byteCount)
                }
            }
            for item in plans {
                prepared.append((item.group, try item.group.prepareImport(
                    item.plan, source: storage.groups[item.group.key]!, evaluate: slabEval,
                    admission: memoryAdmission)))
            }
            let pageMaps = Dictionary(uniqueKeysWithValues: plans.map { ($0.group.key, $0.plan.pages) })
            for (index, layer) in checkpoint.layers.enumerated() {
                let pages = pageMaps[layer.key]!
                tables.append(Array(pages[layer.firstPage ..< layer.firstPage + layer.pageCount]))
                rows.append(PagedSequenceKV(
                    pool: self, kind: layerKinds[layer.modelIndex], groupKey: layer.key,
                    maxLength: maximumTokens, reservedPages: rowNeeds[index]))
            }
            allocating = false
            let actual = prepared.reduce(0) { total, entry in
                total + entry.1.growth.segments.values.reduce(0) { $0 + $1.allocatedBytes }
            }
            let result = segmentGrant.publish(expected: grant, physicalBytes: actual) {
                for (group, value) in prepared { group.installImport(value) }
                for index in rows.indices {
                    let layer = checkpoint.layers[index]
                    if layer.ringPages != nil {
                        rows[index].adoptHistoricalWindowPages(
                            tables[index], retainedStart: layer.tokenStart, storedThrough: checkpoint.position)
                    } else {
                        rows[index].adoptExclusiveCheckpointPages(tables[index], storedThrough: checkpoint.position)
                    }
                }
            }
            storageTelemetry.record(result)
            guard result == .installed else {
                throw CBv2KVError.capacityExhausted(needed: physical, available: segmentGrant.snapshot().bytes)
            }
        } catch {
            if allocating { PagedKVStorageTelemetry.increment(&storageTelemetry.allocationFailures) }
            for row in rows { row.discardUninstalledCheckpointRow() }
            rows.removeAll()
            prepared.removeAll()
            // Discard private imported sources AND auxiliary arrays before the
            // transferred request/floor reservation is rolled back.
            owner.close()
            checkpointImportBeforeRollback?()
            reservation.rollbackAfterDroppingOwners()
            throw error
        }
        // Drop duplicate segment wrappers before handing active rows to code
        // that may fail recurrent/MTP restoration and retire the pool.
        prepared.removeAll()
        storage.close()
        reservation.commit()
        let actual = bytesMaterialized
        physicalLease.release(to: actual)
        storageTelemetry.recordSettlement(bound: physical, actual: actual)
        let identity = owner.lease.identity
        let result = CBv2PagedCheckpointAdoption(
            rows: rows, auxiliary: owner.auxiliary,
            modelIndices: checkpoint.layers.map(\.modelIndex), stateCount: layerKinds.count,
            releaseAdmission: { admission.releaseCheckpointRequest(id: requestID, ownerIdentity: identity) })
        owner.auxiliary.removeAll()
        return result
    }
}
