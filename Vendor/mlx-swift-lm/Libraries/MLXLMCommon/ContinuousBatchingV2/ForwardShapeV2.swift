import Foundation

public enum CBv2ForwardPhase: String, Codable, Sendable {
    case prefill, decode
    case mtpVerification = "mtp_verification"
    case mixedFrontier = "mixed_prefill_decode"
}

public enum CBv2ForwardKind: String, Codable, Sendable {
    case target
    case compiledComponent = "compiled_component"
}

public enum CBv2CompiledComponent: String, Codable, Sendable {
    case siluProduct, weightedExpertSum, gelu, swiGLU, geGLU
    case gemmaGelu, gemmaSoftcap, gptossExperts
}

/// Scalars only. Component leading rows may flatten tokens/experts and are
/// never interpreted as live request batch rows.
public struct CBv2ForwardAxes: Hashable, Codable, Sendable {
    public let phase: CBv2ForwardPhase
    public let kind: CBv2ForwardKind
    public let liveBatchRows: Int
    public let sequenceWidth: Int
    public let physicalBatchRows: Int
    public let physicalComponentRows: Int?
    public let component: CBv2CompiledComponent?
}

public struct CBv2ForwardShapeCount: Codable, Sendable {
    public let axes: CBv2ForwardAxes
    /// Actual dispatches entered, not GPU submissions or kernel launches.
    public var submittedCalls: UInt64
    /// Calls whose owning step reached its existing successful readback.
    /// Acceptance/rollback of speculative tokens does not change this count.
    public var completedCalls: UInt64
}

public struct CBv2ForwardShapeSnapshot: Codable, Sendable {
    public let schema: Int
    public let scope: UInt64
    public let enabled: Bool
    public let entries: [CBv2ForwardShapeCount]
    public let pendingSteps: Int
    public let abandonedSteps: UInt64
    public let unobservedDispatches: UInt64
    public let droppedCalls: UInt64

    public static let disabled = CBv2ForwardShapeSnapshot(schema: 1, scope: 0, enabled: false,
        entries: [], pendingSteps: 0, abandonedSteps: 0, unobservedDispatches: 0, droppedCalls: 0)
}

public enum CBv2ForwardShapeError: Error {
    case engineBusy, scopeExhausted
}

/// Engine-queue confined; no tensor, request ID, token, text or clock storage.
final class CBv2ForwardShapeRecorder {
    static let maximumBuckets = 256
    private(set) var scope: UInt64 = 0
    private var entries: [CBv2ForwardAxes: CBv2ForwardShapeCount] = [:]
    private var pending = 0
    private var abandoned: UInt64 = 0
    private var unobserved: UInt64 = 0
    private var dropped: UInt64 = 0

    func reset() throws {
        guard pending == 0 else { throw CBv2ForwardShapeError.engineBusy }
        guard scope < UInt64.max else { throw CBv2ForwardShapeError.scopeExhausted }
        scope += 1
        entries.removeAll(keepingCapacity: true)
        abandoned = 0; unobserved = 0; dropped = 0
    }

    func beginStep() -> CBv2ForwardShapeStep {
        pending += 1
        return CBv2ForwardShapeStep(owner: self)
    }

    fileprivate func submit(_ axes: CBv2ForwardAxes) -> Bool {
        guard (1...256).contains(axes.liveBatchRows),
            (1...1_048_576).contains(axes.sequenceWidth),
            (axes.liveBatchRows...1_048_576).contains(axes.physicalBatchRows),
            axes.physicalComponentRows.map({ (1...16_777_216).contains($0) }) ?? true,
            entries[axes] != nil || entries.count < Self.maximumBuckets
        else { dropped = Self.add(dropped, 1); return false }
        var value = entries[axes] ?? .init(axes: axes, submittedCalls: 0, completedCalls: 0)
        value.submittedCalls = Self.add(value.submittedCalls, 1)
        entries[axes] = value
        return true
    }

    fileprivate func missingDispatch() { unobserved = Self.add(unobserved, 1) }

    fileprivate func retire(_ counts: [CBv2ForwardAxes: UInt64], completed: Bool) {
        precondition(pending > 0)
        pending -= 1
        if completed {
            for (axes, count) in counts {
                guard var value = entries[axes] else { continue }
                value.completedCalls = Self.add(value.completedCalls, count)
                entries[axes] = value
            }
        } else if !counts.isEmpty {
            abandoned = Self.add(abandoned, 1)
        }
    }

    func snapshot() -> CBv2ForwardShapeSnapshot {
        let ordered = entries.values.sorted {
            let a = $0.axes, b = $1.axes
            return [a.phase.rawValue, a.kind.rawValue, String(a.liveBatchRows), String(a.sequenceWidth),
                    String(a.physicalBatchRows), String(a.physicalComponentRows ?? 0), a.component?.rawValue ?? ""]
                .lexicographicallyPrecedes([b.phase.rawValue, b.kind.rawValue, String(b.liveBatchRows),
                    String(b.sequenceWidth), String(b.physicalBatchRows), String(b.physicalComponentRows ?? 0), b.component?.rawValue ?? ""])
        }
        return .init(schema: 1, scope: scope, enabled: true, entries: ordered,
            pendingSteps: pending, abandonedSteps: abandoned,
            unobservedDispatches: unobserved, droppedCalls: dropped)
    }

    fileprivate static func add(_ a: UInt64, _ b: UInt64) -> UInt64 {
        let (sum, overflow) = a.addingReportingOverflow(b)
        return overflow ? .max : sum
    }
}

/// One engine step's scalar dispatch receipts. Abandonment does not assert
/// that no GPU work ran: model implementations may evaluate internally.
final class CBv2ForwardShapeStep {
    private let owner: CBv2ForwardShapeRecorder
    private var counts: [CBv2ForwardAxes: UInt64] = [:]
    private var attached = false
    private var retired = false

    init(owner: CBv2ForwardShapeRecorder) { self.owner = owner }
    deinit { abandon() }
    func attach() { attached = true }
    func finishBuilding() { if !attached { abandon() } }
    func complete() {
        guard !retired else { return }
        retired = true
        owner.retire(counts, completed: true)
    }
    private func abandon() {
        guard !retired else { return }
        retired = true
        owner.retire(counts, completed: false)
    }
    func submit(_ axes: CBv2ForwardAxes) {
        guard !retired, owner.submit(axes) else { return }
        counts[axes] = CBv2ForwardShapeRecorder.add(counts[axes, default: 0], 1)
    }
    func missingDispatch() { owner.missingDispatch() }
}
