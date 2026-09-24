import Foundation

public struct CBv2ForwardShapeDelta: Codable, Sendable {
    public let schema: Int
    public let scope: UInt64
    public let entries: [CBv2ForwardShapeCount]
    public let complete: Bool
    public let reasons: [String]
    public let pendingStepsBefore: Int
    public let pendingStepsAfter: Int
}

extension CBv2ForwardShapeSnapshot {
    /// Read-only subtraction at idle cohort boundaries. A scope reset,
    /// unknown dispatch, dropped bucket or unfinished step fails closed.
    public func delta(since before: CBv2ForwardShapeSnapshot) -> CBv2ForwardShapeDelta {
        var reasons: [String] = []
        var values: [CBv2ForwardShapeCount] = []
        if !enabled || !before.enabled { reasons.append("disabled") }
        if schema != 1 || before.schema != 1 || scope != before.scope { reasons.append("scope_mismatch") }
        if pendingSteps != 0 || before.pendingSteps != 0 { reasons.append("pending_steps") }
        if abandonedSteps != before.abandonedSteps { reasons.append("abandoned_step") }
        if unobservedDispatches != before.unobservedDispatches { reasons.append("unobserved_dispatch") }
        if droppedCalls != before.droppedCalls { reasons.append("dropped_calls") }
        if !reasons.contains("scope_mismatch") {
            var older: [CBv2ForwardAxes: CBv2ForwardShapeCount] = [:]
            for entry in before.entries {
                if older.updateValue(entry, forKey: entry.axes) != nil { reasons.append("duplicate_axes") }
            }
            var seen = Set<CBv2ForwardAxes>()
            for entry in entries {
                if !seen.insert(entry.axes).inserted { reasons.append("duplicate_axes") }
                let old = older.removeValue(forKey: entry.axes)
                let submitted = old?.submittedCalls ?? 0
                let completed = old?.completedCalls ?? 0
                guard entry.submittedCalls >= submitted, entry.completedCalls >= completed else {
                    reasons.append("counter_regression"); continue
                }
                let a = entry.submittedCalls - submitted, b = entry.completedCalls - completed
                if a != b { reasons.append("unconfirmed_calls") }
                if a != 0 || b != 0 {
                    values.append(.init(axes: entry.axes, submittedCalls: a, completedCalls: b))
                }
            }
            if !older.isEmpty { reasons.append("counter_regression") }
        }
        reasons = Array(Set(reasons)).sorted()
        return .init(schema: 1, scope: scope, entries: values,
            complete: reasons.isEmpty, reasons: reasons,
            pendingStepsBefore: before.pendingSteps, pendingStepsAfter: pendingSteps)
    }
}
