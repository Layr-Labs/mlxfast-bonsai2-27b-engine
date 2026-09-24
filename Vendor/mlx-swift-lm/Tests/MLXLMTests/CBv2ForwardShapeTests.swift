import Foundation
import XCTest
#if canImport(MLXLMCommon)
@testable import MLXLMCommon
#endif

final class CBv2ForwardShapeTests: XCTestCase {
    private final class LeafSpy {
        var calls = 0
        func forward(rows: Int, columns: Int, body: () -> Void = {}) {
            let call = CBv2ForwardShapeObservation.beginTarget(liveBatchRows: rows, sequenceWidth: columns)
            defer { call?.end() }
            calls += 1
            body()
        }
    }

    private func counts(_ snapshot: CBv2ForwardShapeSnapshot, kind: CBv2ForwardKind = .target) -> [CBv2ForwardShapeCount] {
        snapshot.entries.filter { $0.axes.kind == kind }
    }

    func testRowSplittingRecordsFourB1CallsAndPackedRecordsOneB4Call() throws {
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let leaf = LeafSpy()
        let split = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: split, phase: .decode) {
            // The outer request cohort is four, but the actual leaf is B1.
            for _ in 0..<4 { leaf.forward(rows: 1, columns: 1) }
        }
        split.attach()
        XCTAssertEqual(counts(recorder.snapshot()).map(\.submittedCalls), [4])
        XCTAssertEqual(counts(recorder.snapshot()).map(\.completedCalls), [0])
        XCTAssertEqual(recorder.snapshot().pendingSteps, 1)
        split.complete()
        XCTAssertEqual(counts(recorder.snapshot()).map { $0.axes.liveBatchRows }, [1])
        XCTAssertEqual(counts(recorder.snapshot()).map(\.completedCalls), [4])
        try recorder.reset()
        let packed = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: packed, phase: .decode) { leaf.forward(rows: 4, columns: 1) }
        packed.attach(); packed.complete()
        XCTAssertEqual(counts(recorder.snapshot()).map { $0.axes.liveBatchRows }, [4])
        XCTAssertEqual(counts(recorder.snapshot()).map(\.completedCalls), [1])
    }

    func testLookaheadAndCompiledPhysicalRowsNeverBecomeLiveBatchRows() throws {
        let recorder = CBv2ForwardShapeRecorder(); try recorder.reset()
        let step = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .mtpVerification) {
            LeafSpy().forward(rows: 1, columns: 4) {
                CBv2ForwardShapeObservation.compiledComponent(.gptossExperts, physicalRows: 8) {}
            }
        }
        step.attach(); step.complete()
        let target = try XCTUnwrap(counts(recorder.snapshot()).first)
        XCTAssertEqual(target.axes.phase, .mtpVerification)
        XCTAssertEqual(target.axes.liveBatchRows, 1)
        XCTAssertEqual(target.axes.sequenceWidth, 4)
        XCTAssertEqual(target.axes.physicalBatchRows, 1)
        let compiled = try XCTUnwrap(counts(recorder.snapshot(), kind: .compiledComponent).first)
        XCTAssertEqual(compiled.axes.liveBatchRows, 1)
        XCTAssertEqual(compiled.axes.physicalComponentRows, 8)
    }

    func testCompiledInvocationsCountAfterTraceAndExcludeTraceCallbacks() throws {
        let recorder = CBv2ForwardShapeRecorder(); try recorder.reset()
        let step = recorder.beginStep(), leaf = LeafSpy()
        var traces = 0, executions = 0
        func compiled() {
            CBv2ForwardShapeObservation.compiledComponent(.gptossExperts, physicalRows: 4) {
                if traces == 0 {
                    traces += 1
                    // A tracing callback must not fabricate a real B4 target.
                    LeafSpy().forward(rows: 4, columns: 1)
                }
                executions += 1
            }
        }
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .decode) {
            for _ in 0..<2 { leaf.forward(rows: 2, columns: 1, body: compiled) }
        }
        step.attach(); step.complete()
        XCTAssertEqual(traces, 1); XCTAssertEqual(executions, 2)
        XCTAssertEqual(counts(recorder.snapshot()).map { $0.axes.liveBatchRows }, [2])
        XCTAssertEqual(counts(recorder.snapshot()).map(\.completedCalls), [2])
        XCTAssertEqual(counts(recorder.snapshot(), kind: .compiledComponent).map(\.completedCalls), [2])
    }

    func testSerialVerificationColumnsStaySeparateB1Calls() throws {
        let recorder = CBv2ForwardShapeRecorder(); try recorder.reset()
        let step = recorder.beginStep()
        for _ in 0..<4 {
            CBv2ForwardShapeObservation.dispatch(step: step, phase: .mtpVerification) {
                LeafSpy().forward(rows: 1, columns: 1)
            }
        }
        step.attach(); step.complete()
        let value = try XCTUnwrap(counts(recorder.snapshot()).first)
        XCTAssertEqual(value.axes.liveBatchRows, 1)
        XCTAssertEqual(value.axes.sequenceWidth, 1)
        XCTAssertEqual(value.axes.phase, .mtpVerification)
        XCTAssertEqual(value.completedCalls, 4)
    }

    func testWarmupAndBeforeAfterDeltasExcludeEarlierCalls() throws {
        LeafSpy().forward(rows: 8, columns: 32) // No dispatch scope: warmup is invisible.
        let recorder = CBv2ForwardShapeRecorder(); try recorder.reset()
        let before = recorder.snapshot()
        let step = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .prefill) { LeafSpy().forward(rows: 2, columns: 16) }
        step.attach(); step.complete()
        let after = recorder.snapshot(), delta = after.delta(since: before)
        XCTAssertTrue(delta.complete)
        XCTAssertEqual(delta.entries.count, 1)
        XCTAssertEqual(delta.entries.first?.submittedCalls, 1)
        XCTAssertTrue(after.delta(since: after).entries.isEmpty)
        try recorder.reset()
        XCTAssertFalse(recorder.snapshot().delta(since: after).complete)
    }

    func testAbandonmentIsNotCompletionAndResetRefusesPendingWork() throws {
        let recorder = CBv2ForwardShapeRecorder(); try recorder.reset()
        let before = recorder.snapshot(), step = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .decode) { LeafSpy().forward(rows: 2, columns: 1) }
        XCTAssertThrowsError(try recorder.reset())
        step.finishBuilding() // Rejected construction: no successful step completion.
        let after = recorder.snapshot()
        XCTAssertEqual(after.pendingSteps, 0)
        XCTAssertEqual(after.abandonedSteps, 1)
        XCTAssertEqual(counts(after).first?.submittedCalls, 1)
        XCTAssertEqual(counts(after).first?.completedCalls, 0)
        XCTAssertFalse(after.delta(since: before).complete)
    }

    func testThrownDispatchRestoresContextAndRetiresEnteredCallsWithoutCompletion() throws {
        enum Refusal: Error { case rejected }
        let recorder = CBv2ForwardShapeRecorder(); try recorder.reset()
        let before = recorder.snapshot(), step = recorder.beginStep()
        XCTAssertThrowsError(try CBv2ForwardShapeObservation.dispatch(step: step, phase: .mtpVerification) {
            LeafSpy().forward(rows: 2, columns: 3)
            throw Refusal.rejected
        })
        XCTAssertFalse(CBv2ForwardShapeObservation.isActive)
        step.finishBuilding()
        let after = recorder.snapshot(), delta = after.delta(since: before)
        XCTAssertEqual(after.pendingSteps, 0)
        XCTAssertEqual(after.unobservedDispatches, 0)
        XCTAssertEqual(after.droppedCalls, 0)
        XCTAssertEqual(counts(after).first?.submittedCalls, 1)
        XCTAssertEqual(counts(after).first?.completedCalls, 0)
        XCTAssertEqual(Set(delta.reasons), Set(["abandoned_step", "unconfirmed_calls"]))
    }

    func testMissingLeafAndInvalidOrExcessAxesFailClosedWithoutPrivatePayload() throws {
        let recorder = CBv2ForwardShapeRecorder(); try recorder.reset()
        let step = recorder.beginStep()
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .decode) {}
        CBv2ForwardShapeObservation.dispatch(step: step, phase: .prefill) {
            let leaf = LeafSpy()
            for width in 1...257 { leaf.forward(rows: 1, columns: width) }
            leaf.forward(rows: Int.max, columns: Int.max)
        }
        step.attach(); step.complete()
        let value = recorder.snapshot()
        XCTAssertEqual(value.entries.count, 256)
        XCTAssertEqual(value.droppedCalls, 2)
        XCTAssertEqual(value.unobservedDispatches, 1)
        let encoded = String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
        XCTAssertLessThan(encoded.utf8.count, 100_000)
        for forbidden in ["token_ids", "request_id", "prompt", "text", "tensor", "\(Int.max)"] {
            XCTAssertFalse(encoded.contains(forbidden))
        }
    }
}
