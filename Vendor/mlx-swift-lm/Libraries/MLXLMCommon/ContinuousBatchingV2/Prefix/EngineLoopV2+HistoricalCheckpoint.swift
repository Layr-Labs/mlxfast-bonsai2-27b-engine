import MLX

extension CBv2CompleteCheckpointCapture {
    /// Engine queue, before asyncEval and before constructing any successor.
    /// A candidate never enters the durable publication set before commit.
    func prepareHistorical(
        position: Int, chunkSize: Int, state: [CBv2SequenceKV?]
    ) throws -> CBv2CapturedCompleteCheckpoint? {
        guard !isClosed, let layout = codec.historicalLayout,
              state.count == layout.layers.count else { return nil }
        do {
            // Scalar policy projection first; no descriptor/page/token table
            // is built merely to ask the store whether this boundary fits.
            var packedBytes = 0
            for layer in layout.layers.enumerated() where layer.element.owner == layer.offset {
                let item = layer.element
                let bytes = try CBv2CheckpointTensorDescriptor.checkedByteCount(
                    shape: [2, item.kvHeads, position - item.tokenStart(at: position), item.headDim],
                    dtype: item.dtype.mlxDType)
                let (next, overflow) = packedBytes.addingReportingOverflow(bytes)
                guard !overflow else { return nil }
                packedBytes = next
            }
            guard store.acceptsCheckpoint(position: position, packedBytes: packedBytes) else { return nil }
            var windows: [Int: CBv2HistoricalWindow] = [:]
            for (index, layer) in layout.layers.enumerated() {
                if layer.owner != index {
                    guard state[index] == nil else { return nil }
                    continue
                }
                guard let row = state[index] as? PagedSequenceKV,
                      row.absoluteOffset == position, row.pool.layerKinds == codec.layerKinds,
                      row.groupKey.dtype == layer.dtype.mlxDType
                else { return nil }
                if layer.window != nil {
                    windows[index] = try makeHistoricalWindow(row, position, codec.admission)
                }
            }
            return .init(historical: .init(position: position, chunkSize: chunkSize, windows: windows))
        } catch let error as MLXError { throw error }
        catch { return nil }
    }

    func commitHistorical(_ candidate: CBv2CapturedCompleteCheckpoint, requestID: CBv2RequestID) {
        guard !isClosed,
              !(staged[requestID]?.contains { $0.position == candidate.position } ?? false)
        else { candidate.finishEvaluationAndClose(); return }
        if staged[requestID, default: []].count == 2 {
            let previous = staged[requestID]!.removeLast()
            queue.async { previous.finishEvaluationAndClose() }
        }
        staged[requestID, default: []].append(candidate)
    }
}

extension EngineLoopV2 {
    /// Exact scalar geometry advances on launch. It belongs to that immutable
    /// step; scheduler cursors and mutable windows may already be ahead when
    /// finalize runs. Preemption/ragged/packed history disarms this generation.
    func prepareHistoricalCheckpoints(_ step: CBv2InFlightStep) throws -> [MLXArray] {
        guard let capture = completeCheckpointCapture, capture.codec.historicalLayout != nil else { return [] }
        for (id, range) in step.computedRanges {
            guard let rec = scheduler.record(for: id), rec.request.prefixCacheEnabled,
                  rec.request.multimodal == nil, rec.request.positionState == nil, rec.preemptionCount == 0,
                  let cap = step.recurrentCheckpointChunkSizes[id], cap >= scheduler.config.prefillChunkSize,
                  CBv2AttentionV1.queryBlockSize <= 0 || cap % CBv2AttentionV1.queryBlockSize == 0,
                  let state = kvStates[id]
            else { continue }
            var geometry = recurrentCheckpointGeometry[id] ?? .init()
            let eligible = geometry.record(range: range, cap: cap,
                promptLength: rec.request.promptTokens.count, packed: step.packedPrefixRows.contains(id))
            recurrentCheckpointGeometry[id] = geometry
            guard eligible, range.upperBound < rec.request.promptTokens.count,
                  let candidate = try capture.prepareHistorical(position: range.upperBound, chunkSize: cap, state: state)
            else { continue }
            step.historicalCheckpoints[id] = candidate
        }
        let roots = step.historicalCheckpoints.values.flatMap(\.evaluationRoots)
        for candidate in step.historicalCheckpoints.values { candidate.historical?.markSubmitted() }
        return roots
    }

    func commitHistoricalCheckpoints(_ step: CBv2InFlightStep) -> MLXError? {
        guard let capture = completeCheckpointCapture else { return nil }
        var nativeFailure: MLXError?
        for (id, candidate) in step.historicalCheckpoints {
            // This step's sample can finish before its final gather witness.
            // Wait for the already submitted roots before promotion/retirement.
            do {
                try candidate.finishEvaluation()
                if step.discard.contains(id) || scheduler.record(for: id)?.preemptionCount != 0 {
                    candidate.finishEvaluationAndClose()
                } else {
                    capture.commitHistorical(candidate, requestID: id)
                }
            } catch {
                if let error = error as? MLXError { nativeFailure = nativeFailure ?? error }
                candidate.finishEvaluationAndClose()
            }
        }
        step.historicalCheckpoints.removeAll()
        if nativeFailure != nil { step.discard.formUnion(step.participants) }
        return nativeFailure
    }
}
