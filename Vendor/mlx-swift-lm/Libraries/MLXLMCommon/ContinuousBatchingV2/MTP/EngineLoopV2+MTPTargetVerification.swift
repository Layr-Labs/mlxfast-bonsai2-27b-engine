// EngineLoopV2+MTPTargetVerification.swift
//
// Target-authoritative scoring strategies for one known MTP draft chain.

import MLX

extension EngineLoopV2 {

    /// Serial mode is the chip-independent authority path: every column
    /// executes the same `[B, 1]` eager forward used by ordinary decode,
    /// while one surrounding speculative KV transaction defers commit until
    /// the accept walk. Rectangular mode is an explicit optimized strategy.
    ///
    /// Attention-only production targets select rectangular verification:
    /// `CBv2MTPRoundDriver.maximumAutomaticDepth` pre-clamps depth so
    /// `(1 + k) * B <= maxAutomaticRectangularTokens`. Recurrent targets take
    /// the rectangular path only through CAPTURE-VERIFY (the MTPLX GDN
    /// pattern): one `[B, 1+k]` forward whose recurrent layers stage
    /// per-position captured conv/SSM stacks, so finalize can commit the
    /// state at the accepted position on device and rollback stays a
    /// snapshot restore. Serial remains the safety oracle everywhere else.
    ///
    /// Scoring: `scores` is the per-position token the accept walk compares
    /// drafts against AND emits. Historically that was the target argmax
    /// (greedy-exact). When the drafter opted into target-prefix acceptance
    /// and the sampler supports verify pre-sampling, non-greedy rows score
    /// with genuine target samples drawn with the request's real sampler
    /// and per-request RNG stream — exact for the output distribution at
    /// any temperature. All-greedy batches keep the bit-identical argmax.
    func mtpBuildTargetVerification(
        columns: [MLXArray], rows: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver
    ) throws -> (
        scores: MLXArray, hidden: MLXArray,
        shortlist: (ids: MLXArray, massScaled: MLXArray)?,
        policyTopTwo: (ids: MLXArray, values: MLXArray)?,
        cacheInnerState: [MLXArray],
        diagnostics: [CBv2LogitDiagnosticPacket],
        recurrent: [CBv2RequestID: [CBv2RecurrentStateEvaluation]],
        blockContext: MLXArray?
    ) {
        precondition(!columns.isEmpty, "CBv2 MTP: target verification requires a seed column")
        let caches = eagerCaches(rowStates: rows.map { kvStates[$0.rec.id]! })
        let scores: MLXArray
        let hidden: MLXArray
        var shortlist: (ids: MLXArray, massScaled: MLXArray)?
        var policyTopTwo: (ids: MLXArray, values: MLXArray)?
        var recurrent: [CBv2RequestID: [CBv2RecurrentStateEvaluation]] = [:]
        // The fused tapped context a BLOCK drafter reads. The tap holds ONE
        // forward's rows, so it is read immediately after each target
        // forward, before the next one overwrites it.
        var blockContextColumns: [MLXArray] = []
        var capturedInnerState: [MLXArray] = []
        var diagnostics: [CBv2LogitDiagnosticPacket] = []
        let diagnosticOffsets = logitDiagnostic == nil ? nil : rows.map {
            Self.positionOffset(kvStates[$0.rec.id]!)
        }

        let recurrentModel =
            (mtp.model as? any CBv2RecurrentMTPSteppableModel).flatMap { model in
                model.recurrentStateSpec != nil ? model : nil
            }

        // Target-prefix pre-sampling activates only for batches containing a
        // stochastic row; all-greedy batches keep the historical argmax
        // packet bit for bit.
        let useTargetPrefix =
            mtp.targetPrefixAcceptance && sampler.supportsMTPTargetPrefix
            && rows.contains {
                $0.rec.request.sampling.temperature >= LogitsPipelineV2.greedyEpsilon
            }
        let verifyParams = rows.map(\.rec.request.sampling)
        let verifyIDs = rows.map(\.rec.id)
        // Per-request output-step index of window position 0: the seed token
        // (already confirmed output) was drawn at index base-1, so position j
        // of the window is output index base + j.
        let verifyStepBases = rows.map(\.rec.generatedTokenCount)

        func scoreColumns(_ logits: MLXArray, columnOffset: Int) -> MLXArray {
            guard useTargetPrefix else {
                return argMax(logits, axis: -1).asType(.int32)
            }
            guard
                let sampled = sampler.mtpVerifySample(
                    logits: logits, params: verifyParams, requestIDs: verifyIDs,
                    stepBases: verifyStepBases.map { $0 + columnOffset })
            else {
                preconditionFailure(
                    "CBv2 MTP: sampler advertised target-prefix support but returned nil")
            }
            return sampled
        }

        func captureDiagnostics(
            _ logits: MLXArray, columnOffset: Int, phase: String,
            topTwo: (ids: MLXArray, values: MLXArray)? = nil
        ) {
            guard let diagnosticOffsets, let diagnostic = logitDiagnostic else { return }
            for (batchIndex, row) in rows.enumerated()
            where row.rec.id.raw == diagnostic.configuration.requestID {
                let column = diagnostic.configuration.outputIndex - verifyStepBases[batchIndex]
                let localColumn = column - columnOffset
                guard localColumn >= 0, localColumn < logits.dim(1) else { continue }
                let retainedTopTwo = topTwo.map {
                    (ids: $0.ids[batchIndex, localColumn], values: $0.values[batchIndex, localColumn])
                }
                if let packet = makeLogitDiagnostic(
                    logits: logits[batchIndex, localColumn], requestID: row.rec.id,
                    outputIndex: verifyStepBases[batchIndex] + column, phase: phase,
                    batchIndex: batchIndex, batchSize: rows.count, column: column,
                    verificationWidth: columns.count, draftDepth: columns.count - 1,
                    seedToken: row.carry?.token, cacheOffset: diagnosticOffsets[batchIndex],
                    policyTopTwo: retainedTopTwo)
                { diagnostics.append(packet) }
            }
        }

        var useRectangular = switch mtp.config.verificationMode {
        case .serialTarget: false
        case .rectangular, .rectangularExact: true
        case .automatic:
            columns.count * columns[0].dim(0) <= mtp.config.maxAutomaticRectangularTokens
        }

        // A recurrent target may only verify rectangularly through the
        // captured-window seam. Stateful production never falls back to
        // serial after draft construction: that would add target forwards.
        if useRectangular, recurrentModel != nil,
            recurrentModel?.supportsCapturedVerifyWindow != true
        {
            if mtp.usesRequestStatefulDrafter {
                preconditionFailure(
                    "CBv2 production request-stateful MTP requires captured rectangular verification")
            }
            mtp.recordControllerFallback("captured_verify_unsupported")
            useRectangular = false
        }

        // Rectangular verification obliges every layer cache in the bank to
        // serialise its attention one query position at a time for the
        // duration of the round. That capability is the opt-in marker
        // `CBv2MTPRectangularSerializing` (Paged/PagedSeamContract.swift),
        // NOT a concrete type: `CBv2LayerCache` conforms by extension, and a
        // paged bank conforms only once `PagedLayerCache.updateAndAttend`
        // grows the per-column loop (WS-3.4).
        //
        // This was `as? CBv2LayerCache` behind a `preconditionFailure`.
        // `CBv2LayerCache` is `final` and `PagedLayerCache` is a SIBLING
        // conformer of `CBv2AttendingLayerCache`, never a subclass, so that
        // cast could not succeed for a paged bank — and `preconditionFailure`
        // is a `fatalError`: daemon death, every co-resident model's
        // in-flight requests lost, and not one line of telemetry. A bank that
        // cannot serialise MUST degrade to the serial oracle above and MUST
        // NOT trap (PagedSeamContract: "Callers MUST degrade to serial
        // verification for a cache that does not conform, and MUST NOT trap").
        var serializingCaches: [CBv2MTPRectangularSerializing] = []
        if useRectangular {
            serializingCaches = caches.compactMap { $0 as? CBv2MTPRectangularSerializing }
            if serializingCaches.count != caches.count {
                if mtp.usesRequestStatefulDrafter, recurrentModel != nil {
                    preconditionFailure(
                        "CBv2 production request-stateful MTP cache lacks rectangular serialization")
                }
                mtp.recordControllerFallback("rectangular_cache_unsupported")
                useRectangular = false
            }
        }
        mtp.recordVerificationStrategy(rectangular: useRectangular)

        if !useRectangular {
            var scoreColumnsAccum: [MLXArray] = []
            var hiddenColumns: [MLXArray] = []
            scoreColumnsAccum.reserveCapacity(columns.count)
            hiddenColumns.reserveCapacity(columns.count)
            for (columnIndex, column) in columns.enumerated() {
                precondition(column.dim(1) == 1, "CBv2 MTP: serial target column must have L=1")
                let output: (logits: MLXArray, lastHidden: MLXArray)
                var recurrentArrays: [MLXArray] = []
                if let recurrentModel {
                    let evaluations = rows.map { row -> CBv2RecurrentStateEvaluation in
                        guard let state = recurrentStates[row.rec.id] else {
                            preconditionFailure(
                                "CBv2 recurrent MTP state missing for \(row.rec.id)")
                        }
                        do { return try state.bind() } catch {
                            preconditionFailure(
                                "CBv2 recurrent MTP bind failed for \(row.rec.id): \(error)")
                        }
                    }
                    let positionIds = CBv2PositionState.decodePositionIds(
                        states: rows.map(\.rec.request.positionState),
                        cacheOffsets: rows.map { Self.positionOffset(kvStates[$0.rec.id]!) })
                    output = try checkedModelForward(phase: .mtpVerification) { recurrentModel.forwardWithHidden(
                        tokens: column, caches: caches, recurrentState: evaluations,
                        positionIds: positionIds) }
                    for (row, evaluation) in zip(rows, evaluations) {
                        do { recurrentArrays.append(contentsOf: try evaluation.evaluate()) } catch {
                            preconditionFailure(
                                "CBv2 recurrent MTP evaluation failed for \(row.rec.id): \(error)")
                        }
                        recurrent[row.rec.id, default: []].append(evaluation)
                    }
                } else {
                    output = try checkedModelForward(phase: .mtpVerification) { mtp.model.forwardWithHidden(tokens: column, caches: caches) }
                }
                if let block = mtp.blockDrafter {
                    guard let column = block.blockContextHidden() else {
                        preconditionFailure(
                            "CBv2 block MTP: the target's context tap is not armed")
                    }
                    blockContextColumns.append(column)
                }
                captureDiagnostics(
                    output.logits, columnOffset: columnIndex, phase: "serial_verify")
                let columnScores = scoreColumns(output.logits, columnOffset: columnIndex)
                // Building several eager decode calls in one lazy graph can
                // let mutable KV buffers observe a later version. Complete
                // each canonical target step before constructing the next.
                var evaluationTargets =
                    [columnScores, output.lastHidden] + eagerCacheInnerState(caches) + recurrentArrays
                for packet in diagnostics where packet.column == columnIndex {
                    evaluationTargets.append(contentsOf: packet.evaluationTargets)
                }
                eval(evaluationTargets)
                // One blocking evaluation per serial verify column, counted.
                CBv2CoreInstrumentation.recordHostSync()
                scoreColumnsAccum.append(columnScores)
                hiddenColumns.append(output.lastHidden)
            }
            scores = concatenated(scoreColumnsAccum, axis: 1)
            hidden = concatenated(hiddenColumns, axis: 1)

        } else {
            for cache in serializingCaches {
                cache.mtpSerializesRectangularAttention = true
                cache.mtpBatchesRectangularAttention = mtp.drafter.prefersBatchedRectangularAttention
            }
            defer {
                for cache in serializingCaches {
                    cache.mtpSerializesRectangularAttention = false
                    cache.mtpBatchesRectangularAttention = false
                }
            }
            let tokens = concatenated(columns, axis: 1)
            let output: (logits: MLXArray, lastHidden: MLXArray)
            if let recurrentModel {
                // Capture-verify: ONE transaction per row spans the whole
                // window; the model stages [1+k, ...] captured stacks per
                // recurrent layer. Finalize commits the accepted position
                // (device-side slice) or rolls the transaction back — no
                // repair forward on either path.
                let evaluations = rows.map { row -> CBv2RecurrentStateEvaluation in
                    guard let state = recurrentStates[row.rec.id] else {
                        preconditionFailure(
                            "CBv2 recurrent MTP state missing for \(row.rec.id)")
                    }
                    do { return try state.bind() } catch {
                        preconditionFailure(
                            "CBv2 recurrent MTP bind failed for \(row.rec.id): \(error)")
                    }
                }
                let positionIds = CBv2PositionState.decodePositionIds(
                    states: rows.map(\.rec.request.positionState),
                    cacheOffsets: rows.map { Self.positionOffset(kvStates[$0.rec.id]!) },
                    length: tokens.dim(1))
                output = try checkedModelForward(phase: .mtpVerification) { recurrentModel.forwardWithHiddenCaptured(
                    tokens: tokens, caches: caches, recurrentState: evaluations,
                    positionIds: positionIds) }
                for (row, evaluation) in zip(rows, evaluations) {
                    precondition(
                        evaluation.isCaptured,
                        "CBv2 capture-verify forward did not stage captured stacks")
                    do {
                        capturedInnerState.append(contentsOf: try evaluation.evaluate())
                    } catch {
                        preconditionFailure(
                            "CBv2 recurrent MTP evaluation failed for \(row.rec.id): \(error)")
                    }
                    recurrent[row.rec.id] = [evaluation]
                }
            } else {
                output = try checkedModelForward(phase: .mtpVerification) { mtp.model.forwardWithHidden(tokens: tokens, caches: caches) }
            }
            if let block = mtp.blockDrafter {
                guard let context = block.blockContextHidden() else {
                    preconditionFailure(
                        "CBv2 block MTP: the target's context tap is not armed")
                }
                blockContextColumns.append(context)
            }
            if mtp.usesRequestStatefulDrafter {
                guard let provider = mtp.model as? any CBv2MTPPolicyTopTwoProviding else {
                    preconditionFailure(
                        "CBv2 request-stateful rectangular MTP target lacks top-two provider")
                }
                let batch = output.logits.dim(0)
                let width = output.logits.dim(1)
                let vocabulary = output.logits.dim(2)
                let flat = output.logits.reshaped([1, batch * width, vocabulary])
                let topTwo = provider.cbv2MTPTopTwo(flat)
                policyTopTwo = (
                    topTwo.ids.reshaped([batch, width, 2]).asType(.int32),
                    topTwo.values.reshaped([batch, width, 2]).asType(.float32))
            }
            if useTargetPrefix {
                scores = scoreColumns(output.logits, columnOffset: 0)
            } else if let policyTopTwo {
                scores = policyTopTwo.ids[0..., 0..., 0]
            } else {
                scores = argMax(output.logits, axis: -1).asType(.int32)
            }
            captureDiagnostics(
                output.logits, columnOffset: 0, phase: "rectangular_verify", topTwo: policyTopTwo)
            hidden = output.lastHidden
            // Draft-head shortlist (rectangular only; the serial oracle stays
            // byte-identical to the shipped path): each verify position's
            // target top-K ids feed the NEXT round's shortlisted draft, and
            // the captured probability mass rides the acceptance packet so
            // finalize can gate coverage without an extra host sync.
            if let size = (mtp.drafter as? any CBv2MTPRequestStatefulDrafter)?
                .draftShortlistSize
            {
                shortlist = Self.mtpDraftShortlist(logits: output.logits, size: size)
            }
        }

        let blockContext: MLXArray? =
            blockContextColumns.isEmpty
            ? nil
            : (blockContextColumns.count == 1
                ? blockContextColumns[0]
                : concatenated(blockContextColumns, axis: 1))
        return (
            scores, hidden, shortlist, policyTopTwo,
            eagerCacheInnerState(caches) + capturedInnerState, diagnostics, recurrent,
            blockContext)
    }

    /// Top-`size` token ids per verify position plus their probability mass
    /// scaled to int32 parts-per-million. nil when the shortlist would not
    /// actually narrow the head.
    static func mtpDraftShortlist(
        logits: MLXArray, size: Int
    ) -> (ids: MLXArray, massScaled: MLXArray)? {
        let vocabulary = logits.dim(-1)
        guard size > 0, size < vocabulary else { return nil }
        // argPartition guarantees positions kth... hold the largest values
        // (unsorted — argmax over the gathered rows doesn't need order).
        let ids = argPartition(logits, kth: vocabulary - size, axis: -1)[
            .ellipsis, (vocabulary - size)...]
        let values = takeAlong(logits, ids, axis: -1).asType(.float32)
        let mass = sum(
            exp(values - logSumExp(logits.asType(.float32), axis: -1, keepDims: true)),
            axis: -1)
        return (ids.asType(.int32), (mass * 1_000_000).asType(.int32))
    }
}
