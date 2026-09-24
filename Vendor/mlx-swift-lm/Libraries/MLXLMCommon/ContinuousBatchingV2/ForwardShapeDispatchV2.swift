import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

/// Synchronous dispatch context. pthread TLS prevents one engine's observer
/// from reaching another engine or an unrelated model on a different thread.
public enum CBv2ForwardShapeObservation {
    fileprivate final class Frame {
        let step: CBv2ForwardShapeStep
        let phase: CBv2ForwardPhase
        var target: CBv2ForwardAxes?
        var lastTarget: CBv2ForwardAxes?
        var targets = 0
        init(step: CBv2ForwardShapeStep, phase: CBv2ForwardPhase) {
            self.step = step; self.phase = phase
        }
    }

    private static let key: pthread_key_t = {
        var value = pthread_key_t()
        precondition(pthread_key_create(&value, nil) == 0)
        return value
    }()
    private static var frame: Frame? {
        guard let pointer = pthread_getspecific(key) else { return nil }
        return Unmanaged<Frame>.fromOpaque(pointer).takeUnretainedValue()
    }
    public static var isActive: Bool { pthread_getspecific(key) != nil }

    static func dispatch<T>(step: CBv2ForwardShapeStep?, phase: CBv2ForwardPhase, _ body: () throws -> T) rethrows -> T {
        guard let step else { return try body() }
        let current = Frame(step: step, phase: phase)
        let previous = pthread_getspecific(key)
        pthread_setspecific(key, Unmanaged.passUnretained(current).toOpaque())
        defer {
            pthread_setspecific(key, previous)
            if current.targets == 0 { step.missingDispatch() }
            withExtendedLifetime(current) {}
        }
        return try body()
    }

    /// Install only at the actual target trunk entry, after any adapter row
    /// splitting. The returned lifetime owns scalars, never the input tensor.
    public static func beginTarget(liveBatchRows: Int, sequenceWidth: Int,
        physicalBatchRows: Int? = nil) -> TargetCall?
    {
        guard let current = frame else { return nil }
        let axes = CBv2ForwardAxes(phase: current.phase, kind: .target,
            liveBatchRows: liveBatchRows, sequenceWidth: sequenceWidth,
            physicalBatchRows: physicalBatchRows ?? liveBatchRows,
            physicalComponentRows: nil, component: nil)
        let call = TargetCall(frame: current, previous: current.target)
        current.targets += 1
        current.target = axes
        current.lastTarget = axes
        current.step.submit(axes)
        return call
    }

    public final class TargetCall {
        private let frame: Frame
        private let previous: CBv2ForwardAxes?
        private var ended = false
        fileprivate init(frame: Frame, previous: CBv2ForwardAxes?) {
            self.frame = frame; self.previous = previous
        }
        public func end() {
            guard !ended else { return }
            ended = true
            frame.target = previous
        }
        deinit { end() }
    }

    /// Count outside the compiled closure, once per invocation. Suppression
    /// prevents trace callbacks/nested compiled bodies from becoming fake calls.
    public static func compiledComponent<T>(_ component: CBv2CompiledComponent,
        physicalRows: Int, _ body: () throws -> T) rethrows -> T
    {
        // A model's final softcap can follow the trunk's return. Its component
        // remains associated with that dispatch's last actual target axes.
        guard let current = frame, let target = current.target ?? current.lastTarget else { return try body() }
        current.step.submit(.init(phase: target.phase, kind: .compiledComponent,
            liveBatchRows: target.liveBatchRows, sequenceWidth: target.sequenceWidth,
            physicalBatchRows: target.physicalBatchRows,
            physicalComponentRows: physicalRows, component: component))
        let previous = pthread_getspecific(key)
        pthread_setspecific(key, nil)
        defer { pthread_setspecific(key, previous) }
        return try body()
    }
}
