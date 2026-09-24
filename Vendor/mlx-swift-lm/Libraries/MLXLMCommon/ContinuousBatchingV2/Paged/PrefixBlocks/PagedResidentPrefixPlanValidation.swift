// PagedResidentPrefixPlanValidation.swift
//
// Fail-closed validation for the M/C/R contract before resident pages are
// retained. A malformed hybrid replay plan can otherwise produce plausible
// but incomplete sliding-window KV and silently change logits.

import Foundation

func cbv2ValidateResidentPagedPrefixPlan(
    _ plan: CBv2PrefixReusePlan,
    layerKinds: [CBv2LayerKind],
    maxLength: Int,
    reserveFullSequenceTokens: Bool = false
) throws {
    guard plan.backend == .pagedFP16,
        plan.matchedBoundary > 0,
        plan.matchedBoundary <= maxLength,
        plan.replayStart > 0,
        plan.replayStart <= plan.matchedBoundary,
        plan.replayTokens == plan.matchedBoundary - plan.replayStart,
        plan.prefillTokensSaved == plan.replayStart,
        plan.capacityReservationTokens == (reserveFullSequenceTokens
            ? plan.fullCapacityTokensReserved : plan.restoredFullTokens),
        plan.fullCapacityTokensReserved >= maxLength,
        plan.nominalFullKVBytesPerToken >= 0,
        plan.fullKVBytesPerToken >= 0,
        plan.additionalFullKVBytesPerToken >= 0,
        plan.initialAdditionalCapacityBytes >= 0,
        plan.stagedFullKVBytes >= 0,
        plan.residentFullKVBytes >= 0
    else {
        throw CBv2KVError.backendIneligible(
            reason: "invalid resident prefix replay plan")
    }

    switch plan.strategy {
    case .direct, .tailReplay:
        guard plan.restoredFullTokens == plan.replayStart else {
            throw CBv2KVError.backendIneligible(
                reason: "resident ordinary plan must restore full rows through replay start")
        }
    case .frozenFullReplay:
        guard plan.restoredFullTokens == plan.matchedBoundary else {
            throw CBv2KVError.backendIneligible(
                reason: "resident frozen plan must restore full rows through matched boundary")
        }
    }

    if !plan.requiresExactWindowRestore {
        let required = cbv2RequiredRecompute(
            layerKinds: layerKinds, matched: plan.matchedBoundary)
        guard plan.replayTokens >= required else {
            throw CBv2KVError.backendIneligible(
                reason: "resident replay of \(plan.replayTokens) tokens is shorter than the "
                    + "\(required) this layout needs")
        }
    }
}
