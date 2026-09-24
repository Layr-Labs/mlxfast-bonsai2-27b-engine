// EngineLoopV2+MTPPlanning.swift
//
// MTP row eligibility and step-global scheduler planning.

extension EngineLoopV2 {

    /// Per-row hard gates. Ineligible rows remain ordinary target rows and do
    /// not contribute MTP skip metrics.
    ///
    /// The historical `temperature == 0` gate is lifted when the drafter
    /// opted into target-prefix acceptance AND the installed sampler can
    /// pre-sample verify positions with the request's real sampler semantics
    /// (`CBv2StepSampler.mtpVerifySample`): every committed token is then a
    /// genuine target sample, which is exact for the output distribution at
    /// any temperature/top-p/top-k/min-p. The remaining gates stay: token
    /// constraints, logprob capture, logit bias, and penalties are stateful
    /// per-position transforms the verify pre-sampler does not reproduce,
    /// and stop strings need the serial detokenizer walk.
    func mtpBasicEligible(_ rec: CBv2ScheduledRequest) -> Bool {
        let sampling = rec.request.sampling
        let samplingEligible =
            sampling.temperature == 0
            || (mtp?.targetPrefixAcceptance == true && sampler.supportsMTPTargetPrefix)
        guard rec.request.tokenConstraint == nil,
            samplingEligible,
            sampling.topLogprobs == 0,
            sampling.logitBias.isEmpty,
            sampling.repetitionPenalty == 1,
            sampling.frequencyPenalty == 0,
            sampling.presencePenalty == 0,
            rec.request.stopStrings.isEmpty
        else { return false }
        // The current multimodal seam does not return trusted pre-norm
        // hidden rows from embedding-spliced forwards. Only persistent-history
        // drafters require those rows; stateless Gemma keeps its established
        // frozen-KV multimodal path.
        let hasMultimodalSpans =
            multimodalByID[rec.id]?.spans.isEmpty == false
        if !Self.mtpMultimodalHistoryEligible(
            hasSpans: hasMultimodalSpans,
            tracksPersistentHistory: mtp?.tracksPersistentHistory == true)
        {
            return false
        }
        return true
    }

    static func mtpMultimodalHistoryEligible(
        hasSpans: Bool, tracksPersistentHistory: Bool
    ) -> Bool {
        !hasSpans || !tracksPersistentHistory
    }

    /// Every storage-owning row must support value-exact multi-token writes
    /// and rollback. Kept internal for backend contract tests.
    static func mtpStorageEligible(_ state: [CBv2SequenceKV?]) -> Bool {
        state.allSatisfy { $0?.supportsSpeculativeWrites ?? true }
    }

    private enum CBv2MTPPlanAction {
        case round(k: Int)
        case seed
        case none
    }

    private func mtpPlanAction(
        for rec: CBv2ScheduledRequest, recordSkips: Bool
    ) -> CBv2MTPPlanAction {
        guard let mtp else { return .none }
        guard mtpBasicEligible(rec) else { return .none }
        let depth = mtp.planDepth
        guard let state = kvStates[rec.id] else { return .none }
        guard Self.mtpStorageEligible(state) else {
            if recordSkips { mtp.recordSkip("kv_unsupported") }
            return .none
        }

        let remainingToLength = rec.request.maxTokens - rec.generatedTokenCount
        if mtp.forceSeedPlan {
            return remainingToLength > 0 ? .seed : .none
        }
        if depth == 0 {
            return mtp.tracksPersistentHistory && remainingToLength > 0 ? .seed : .none
        }

        // Verify writes 1+k entries at [C-1, C-1+k]. A seed only pays off
        // when a full round can follow after its one generated token.
        switch mtp.validatedCarry(for: rec) {
        case .valid:
            guard remainingToLength >= depth else {
                if recordSkips { mtp.recordSkip("max_tokens") }
                return .none
            }
            return .round(k: depth)
        case .stale:
            if recordSkips { mtp.recordSkip("carry_invalid") }
            fallthrough
        case .none:
            guard remainingToLength >= depth + 1 else { return .none }
            return .seed
        }
    }

    /// Scheduler speculation hook for one decode-ready running row.
    func mtpPlanSpeculation(for rec: CBv2ScheduledRequest) -> Int {
        guard let mtp else { return 0 }
        switch mtpPlanAction(for: rec, recordSkips: true) {
        case .round(let k):
            mtp.markRound(rec.id, k: k)
            return k
        case .seed:
            mtp.markSeed(rec.id)
            return 0
        case .none:
            return 0
        }
    }

    /// Break the chained fast path when the next step must seed or verify.
    func mtpWantsStep(ids: [CBv2RequestID]) -> Bool {
        guard let mtp else { return false }
        let rows = ids.compactMap { scheduler.record(for: $0) }
        if mtp.config.fixedDraftTokens == 0, mtp.usesRequestStatefulDrafter {
            return false
        }
        let withinBatchGate = ids.count <= mtp.config.maxSpeculativeBatch
        let canSpeculate = withinBatchGate && rows.count == ids.count
            && mtpRowsCanSpeculate(rows)
        let decision = mtp.previewDecision(
            plannedDecodeRows: ids.count, canSpeculate: canSpeculate)
        let eligible = rows.filter { rec in
            guard mtpBasicEligible(rec), let state = kvStates[rec.id] else { return false }
            return Self.mtpStorageEligible(state)
        }
        guard !eligible.isEmpty else { return false }
        // Every positive-cap stateful Qwen request owns trusted target
        // history, including fixed-depth requests temporarily forced to k=0
        // by batch/headroom pressure. Route those target-only steps through
        // the hidden-returning MTP path. Explicit fixed k=0 has no history
        // ownership and remains on ordinary chained decode.
        if mtp.tracksPersistentHistory { return true }
        if decision.depth == 0 {
            if mtp.config.verificationMode == .automatic,
                mtp.maximumAutomaticDepth(plannedDecodeRows: ids.count) == 0
            {
                return false
            }
            return canSpeculate && mtp.requiresNonChainedDepthZeroProbe(decision)
        }
        guard canSpeculate else { return false }
        return eligible.allSatisfy { rec in
            let remaining = rec.request.maxTokens - rec.generatedTokenCount
            return mtp.hasValidCarry(for: rec)
                ? remaining >= 1
                : remaining >= 2
        }
    }

    /// A depth-k round verifies and can emit k+1 target tokens. Reserve one
    /// output slot for the mandatory target token, even when a carry already
    /// exists. A draft with only one output slot left cannot avoid any target
    /// work. Apply this bound to every offer, including fixed and exploration
    /// depths, using the shortest row's budget for the common rectangle.
    static func mtpDepthWithinOutputBudget(
        offeredDepth: Int, remainingTokens: [Int]
    ) -> Int {
        guard offeredDepth > 0, !remainingTokens.isEmpty else { return 0 }
        return remainingTokens.reduce(offeredDepth) { depth, remaining in
            min(depth, remaining > 1 ? remaining - 1 : 0)
        }
    }

    /// Select one controller depth for all decode rows in the scheduler plan.
    /// Chunked-prefill neighbors do not change the controller batch bucket.
    func beginMTPPlan() {
        guard let mtp else { return }
        let rows = scheduler.running.filter {
            !$0.isPaused && !$0.cancelRequested && $0.isDecodeReady
        }
        let withinBatchGate = rows.count <= mtp.config.maxSpeculativeBatch
        let canSpeculate = withinBatchGate && mtpRowsCanSpeculate(rows)
        mtp.beginPlan(
            plannedDecodeRows: rows.count, canSpeculate: canSpeculate,
            rowIDs: rows.map(\.id))
        let eligibleRows = rows.filter { rec in
            guard mtpBasicEligible(rec), let state = kvStates[rec.id] else { return false }
            return Self.mtpStorageEligible(state)
        }


        if mtp.shouldApplyMarginalPolicyToPlan, mtp.planDepth > 0,
            !eligibleRows.isEmpty
        {
            let verificationLimit = mtp.maximumAutomaticDepth(
                plannedDecodeRows: rows.count)
            let marginalDepth =
                eligibleRows.map { rec in
                    mtp.marginalDepth(
                        for: rec.id,
                        offeredDepth: mtp.planDepth,
                        remainingTokens: rec.request.maxTokens - rec.generatedTokenCount,
                        verificationLimit: verificationLimit,
                        decodeRowBucket: mtp.planDecodeRowBucket)
                }.min() ?? 0
            if marginalDepth < mtp.planDepth {
                mtp.clampPlanDepth(to: marginalDepth, reason: "marginal_policy")
            }
        }
        if mtp.planDepth > 0 {
            let tailDepth = Self.mtpDepthWithinOutputBudget(
                offeredDepth: mtp.planDepth,
                remainingTokens: eligibleRows.map {
                    $0.request.maxTokens - $0.generatedTokenCount
                })
            if tailDepth < mtp.planDepth {
                mtp.clampPlanDepth(to: tailDepth, reason: "tail_depth")
            }
        }
        if mtp.planDepth > 0 {
            let eligibleIDs = Set(eligibleRows.map(\.id))
            let stepTokens = rows.count + eligibleRows.count * mtp.planDepth
            let capacityTokens = rows.reduce(0) { total, rec in
                let count = 1 + (eligibleIDs.contains(rec.id) ? mtp.planDepth : 0)
                return total
                    + rec.capacityTokensForChunk(start: rec.numComputedTokens, count: count)
            }
            if stepTokens > scheduler.config.maxBatchedTokensPerStep {
                mtp.clampPlanDepth(to: 0, reason: "step_token_budget")
            } else if !(capacity?.hasHeadroom(additionalTokens: capacityTokens) ?? true) {
                mtp.clampPlanDepth(to: 0, reason: "step_kv_headroom")
            }
        }
        if mtp.planDepth > 0,
            eligibleRows.contains(where: { !mtp.hasValidCarry(for: $0) })
        {
            // Seeding is step-global. Keep existing persistent histories
            // resident and perform one ordinary hidden-returning target step
            // for every eligible row rather than destroying their carries.
            mtp.forceSynchronizedSeed()
            mtp.recordControllerFallback("synchronized_seed")
        }

        for rec in rows {
            let storageEligible = kvStates[rec.id].map(Self.mtpStorageEligible) ?? false
            if mtpBasicEligible(rec), kvStates[rec.id] != nil, !storageEligible {
                mtp.recordSkip("kv_unsupported")
            }
            if !mtpBasicEligible(rec) || !storageEligible
                || (mtp.planDepth == 0 && !mtp.tracksPersistentHistory)
            {
                mtp.invalidateCarry(rec.id)
            }
        }
        if !rows.isEmpty, !withinBatchGate {
            for _ in rows { mtp.recordSkip("batch_gate") }
        }
    }

    private func mtpRowsCanSpeculate(_ rows: [CBv2ScheduledRequest]) -> Bool {
        !rows.isEmpty && rows.allSatisfy { rec in
            guard mtpBasicEligible(rec), let state = kvStates[rec.id] else { return false }
            return Self.mtpStorageEligible(state)
        }
    }

    /// True when this scheduler plan carries seed or verify work.
    func mtpRoundNeeded(_ plan: CBv2StepPlan) -> Bool {
        guard let mtp else { return false }
        if mtp.planHasMTPWork {
            return plan.assignments.contains {
                mtp.roundMark(for: $0.id) != nil || mtp.isSeedMarked($0.id)
            }
        }
        guard mtp.tracksPersistentHistory else { return false }
        return plan.assignments.contains { assignment in
            guard let rec = scheduler.record(for: assignment.id),
                mtpBasicEligible(rec)
            else { return false }
            // Prompt rows acquire their KV state inside execution. Absence
            // here is not ineligibility; route them through the hidden-returning
            // MTP prefill path so persistent history starts at prompt position 0.
            guard let state = kvStates[assignment.id] else { return true }
            return Self.mtpStorageEligible(state)
        }
    }
}
