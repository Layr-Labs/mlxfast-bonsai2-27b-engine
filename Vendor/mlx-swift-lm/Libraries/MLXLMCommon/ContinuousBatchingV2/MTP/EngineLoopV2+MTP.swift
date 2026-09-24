// EngineLoopV2+MTP.swift
//
// Orchestrates one MTP engine step. Eligibility and planning, graph
// construction, finalize-time acceptance, and controller measurement live in
// focused sibling extensions. MTP steps never use compiled decode or chain.

import Foundation
import MLX

extension EngineLoopV2 {

    /// Execute a plan containing MTP work. Graph construction includes seed
    /// decodes, frozen-KV drafting, target-authoritative verification, ordinary
    /// decode neighbors, and per-request prefill chunks.
    func executeMTPRound(_ plan: CBv2StepPlan) throws -> CBv2InFlightStep? {
        guard let mtp else { return try executeMixed(plan) }
        let shapes = beginForwardShapeStep()
        defer { endForwardShapeStep(shapes) }
        let wallStartedNanos = DispatchTime.now().uptimeNanoseconds
        launchClockNanos = wallStartedNanos
        defer { launchClockNanos = 0 }

        // A per-row reserve retry can demote a round after the step-global
        // preflight. Demote all round rows so the target still sees one L=1
        // batch, and refund every speculative suffix before graph build.
        let demoteAllRounds = plan.assignments.contains { assignment in
            guard let k = mtp.roundMark(for: assignment.id) else { return false }
            return assignment.numTokens != 1 + k
        }
        if demoteAllRounds {
            mtpRecordSchedulerDemotions(plan)
        }

        let buildStart = CBv2StepProfiler.enabled ? CFAbsoluteTimeGetCurrent() : 0
        let work = mtpPrepareRoundWork(
            plan, driver: mtp, demoteAllRounds: demoteAllRounds,
            launchNanos: wallStartedNanos)
        guard !work.isEmpty else {
            // Undo optimistic scheduler advances before pending samples can
            // block waiting admission.
            scheduler.rollback(plan)
            return nil
        }
        // Timing stamps mirror `executeMixed`: admission was stamped in
        // `mtpPrepareRoundWork` (before `ensureKVState`); solo prefill
        // chunks are stamped inside `mtpBuildRoundGraph`, where each row's
        // multimodal input is already bound (MTP rounds never pack). Only
        // the decode-shaped one-token prompt chunk is stamped here: one
        // compare per decode row, no lookup, no allocation.
        for row in work where row.isDecode && row.start < row.rec.request.promptTokens.count {
            row.rec.stampPrefillChunkLaunch(
                tokens: 1, packed: false, vision: false, stripe: false,
                launchNanos: wallStartedNanos)
        }

        let graph = try mtpBuildRoundGraph(work, driver: mtp, launchNanos: wallStartedNanos)
        scheduler.markPendingSamples(ids: graph.sampledRows)
        if let verify = graph.verify {
            scheduler.markPendingSamples(
                counts: verify.rows.map { (id: $0.id, count: 1 + verify.k) })
        }
        mtp.recordSeedSteps(graph.seedRows.count)

        asyncEval(graph.asyncEvalTargets)
        if CBv2StepProfiler.enabled {
            CBv2StepProfiler.record(
                "v2.mtp.launch.total", seconds: CFAbsoluteTimeGetCurrent() - buildStart)
        }

        let step = CBv2InFlightStep(
            assignments: work.map { (id: $0.rec.id, numTokens: $0.count) },
            participants: Set(work.map(\.rec.id)),
            sampledRows: graph.sampledRows,
            sampledTokens: graph.sampledTokens,
            evalTargets: graph.prefillEvalTargets,
            computedRanges: Dictionary(
                uniqueKeysWithValues: work.map {
                    ($0.rec.id, $0.start ..< ($0.start + $0.count))
                }),
            wallStartedNanos: wallStartedNanos)
        step.logprobSegments = graph.logprobSegments
        step.logitDiagnostics = graph.diagnostics
        step.recurrentEvaluations = graph.recurrentEvaluations
        if hybridPrefixCache != nil || completeCheckpointCapture != nil {
            step.recurrentCheckpointChunkSizes = Dictionary(
                uniqueKeysWithValues: work.map { ($0.rec.id, $0.rec.plannedPrefillChunkSize) })
        }
        if graph.verify != nil || !graph.seedRows.isEmpty
            || !graph.committedObservationRows.isEmpty
        {
            step.mtpRound = CBv2MTPRoundInFlight(
                verify: graph.verify,
                seedRows: graph.seedRows,
                seedHidden: graph.seedHidden,
                seedPolicyTopTwoValues: graph.seedPolicyTopTwoValues,
                committedObservationRows: graph.committedObservationRows)
        }
        step.forwardShapes = shapes
        shapes?.attach()
        return step
    }
}
