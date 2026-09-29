// CBv2VerifyLeadingSpeculation.swift
//
// The next verify's leading layers, built before the readback of a
// prompt-lookup round.
//
// While the output quotes the prompt, a round's 15 ids come from the prompt
// (`CBv2PromptLookupDraft`) and the drafter is skipped. When such a round
// accepts everything, the next round's verify input is known before its
// readback: row 0 is the target's bonus token, which a quoting round expects
// to be the next prompt token, and rows 1-15 are the lookup of the history
// that bonus completes. So before the readback the engine builds the next
// verify's first recurrent layers from those ids and from the state the full
// acceptance will commit (`speculativeFullAcceptanceEvaluation`: the same
// builders the commit runs), and submits them behind the verify. The GPU goes
// from the verify straight into them while the host reads back, finalizes,
// looks the continuation up and builds the rest of the verify.
//
// The build itself waits for part of the verify in flight: a layer whose
// input is the verify's deferred full-acceptance replay runs the fused scan
// (`Qwen35GDNReplayFused.launch`), which evaluates the verify's tape of that
// layer (`eval(previous)`) before it encodes. That wait is host time before
// the readback, which waits for the whole verify anyway; it changes no value,
// and the measured round times (trial and timed window) include it.
//
// Adoption is strict: every id accepted, the bonus equal to the one assumed,
// the round's own lookup equal to the ids assumed, the next verify built from
// exactly that carry and proposal (same objects), and in the model the input
// state the commit left of the same generation (same deferred replay inputs
// and keep, or the same final state). Anything else drops the speculation
// (its layers only cost GPU time) and the verify builds every layer itself.
// Only recurrent layers are speculated (no KV write, no position), and the
// model checks that none of them is a tap layer.
//
// Load-time bitwise self-test and in-worker trial (`Qwen35VerifyLeading` in
// the model, run right after the drafter's load-time engine-round warm); the
// speculation stays off unless both pass, and with no layer count chosen and
// no trial running a round takes the record's host path unchanged.
// `DARKBLOOM_BONSAI_VERIFY_LEADING=0` turns it off, `=1|2|3` forces that many
// layers once the self-test passes (no trial).
//
// Every mutable field below is process-wide and guarded by one lock; the
// speculation of a round is also keyed by the engine that made it.

import Foundation
import MLX

/// A target whose verify window's leading layers can be built ahead.
public protocol CBv2VerifyLeadingSpeculating {
    /// Layers `0 ..< layers` of a verify window of `tokens` over the detached
    /// transaction `recurrentState`, built and submitted; nil when that does
    /// not apply. The result goes back to the model through `handoff`.
    func speculateVerifyLeading(
        tokens: [Int32], recurrentState: CBv2RecurrentStateEvaluation, layers: Int
    ) -> AnyObject?
}

public enum CBv2VerifyLeading {
    /// nil: trial-decided; 0: off; 1...3: forced layer count.
    public static let forced: Int? = {
        guard
            let raw = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_VERIFY_LEADING"]?
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !raw.isEmpty
        else { return nil }
        if ["0", "off", "false", "no"].contains(raw) { return 0 }
        return Int(raw).map { min(max($0, 0), 3) }
    }()

    private static let lock = NSLock()
    nonisolated(unsafe) private static var chosenLayers = 0
    nonisolated(unsafe) private static var handoffSlot: AnyObject?
    nonisolated(unsafe) private static var adopted = false
    nonisolated(unsafe) private static var pendingSlot: Pending?

    /// Layers the served rounds speculate: 0 until the self-test and the
    /// trial pick a count.
    public static var layers: Int {
        get { lock.withLock { chosenLayers } }
        set { lock.withLock { chosenLayers = newValue } }
    }

    /// Whether a round may speculate at all: a count chosen or a trial
    /// running. False is the record's host path.
    static var speculating: Bool { lock.withLock { chosenLayers > 0 || !trialModes.isEmpty } }

    /// The speculation the verify being built may adopt; taken by the model.
    public static var handoff: AnyObject? {
        get { lock.withLock { handoffSlot } }
        set { lock.withLock { handoffSlot = newValue } }
    }

    public static func takeHandoff() -> AnyObject? {
        lock.withLock {
            defer { handoffSlot = nil }
            return handoffSlot
        }
    }

    /// Set by the model when a verify adopted the handoff.
    public static func noteAdopted() { lock.withLock { adopted = true } }

    /// Whether a verify adopted the handoff since the last call (self-tests).
    public static func takeAdopted() -> Bool {
        lock.withLock {
            defer { adopted = false }
            return adopted
        }
    }

    struct Pending {
        let engine: ObjectIdentifier
        let id: CBv2RequestID
        let handle: AnyObject?
        let mode: Int
        let bonus: Int
        let next: [Int]
        let tokensCount: Int
        var proposal: MLXArray? = nil
    }

    /// The speculation of the round `engine` is finalizing, then (once
    /// confirmed) of the round its next verify build makes.
    static func setPending(_ pending: Pending?) { lock.withLock { pendingSlot = pending } }

    /// `engine`'s speculation, removed; nil when there is none.
    static func takePending(engine: ObjectIdentifier) -> Pending? {
        lock.withLock {
            guard let pending = pendingSlot, pending.engine == engine else { return nil }
            pendingSlot = nil
            return pending
        }
    }

    // MARK: Trial

    /// Modes (layer counts, 0 = the record's path) the running trial
    /// interleaves; empty when no trial runs.
    nonisolated(unsafe) private static var trialModes: [Int] = []
    nonisolated(unsafe) private static var trialCounter = 0
    nonisolated(unsafe) private static var trialTimes: [Int: [UInt64]] = [:]
    /// The previous confirmed round's mode and readback instant.
    nonisolated(unsafe) private static var previous: (mode: Int, nanos: UInt64)?
    /// Whether the verify now in flight was the one the previous round
    /// expected (and, for a speculating mode, adopted its layers).
    nonisolated(unsafe) private static var builtAsExpected = false

    public static let trialSamplesPerMode = 6
    public static let trialModesDefault = [0, 1, 2, 3]
    /// Rounds the trial's warm request needs at full acceptance.
    public static var trialRoundsNeeded: Int {
        4 + trialModesDefault.count * (trialSamplesPerMode + 1)
    }

    public static func beginTrial(modes: [Int] = trialModesDefault) {
        lock.withLock {
            trialModes = modes
            trialCounter = 0
            trialTimes = [:]
            previous = nil
            builtAsExpected = false
        }
    }

    /// Ends the trial: the chosen layer count and one line of medians.
    public static func finishTrial() -> (layers: Int, detail: String) {
        let (modes, times) = lock.withLock {
            defer {
                trialModes = []
                previous = nil
                pendingSlot = nil
                handoffSlot = nil
            }
            return (trialModes, trialTimes)
        }
        func median(_ values: [UInt64]) -> Double? {
            guard values.count >= 4 else { return nil }
            let sorted = values.sorted()
            return Double(sorted[sorted.count / 2]) / 1000
        }
        var parts: [String] = []
        let base = median(times[0] ?? [])
        var best: (layers: Int, us: Double)?
        for mode in modes {
            let values = times[mode] ?? []
            guard let m = median(values) else {
                parts.append("\(mode): \(values.count) rounds")
                continue
            }
            parts.append(String(format: "%d: %.1f us (%d)", mode, m, values.count))
            if mode > 0, best == nil || m < best!.us { best = (mode, m) }
        }
        var chosen = 0
        if let base, let best, best.us <= 0.99 * base { chosen = best.layers }
        let ratio = base.flatMap { b in best.map { String(format: "%.4f", $0.us / b) } } ?? "n/a"
        return (chosen, "round medians [" + parts.joined(separator: ", ") + "], best/record \(ratio)")
    }

    /// The layer count for the speculation made now.
    static func modeForSpeculation() -> Int {
        lock.withLock {
            guard !trialModes.isEmpty else { return chosenLayers }
            defer { trialCounter += 1 }
            return trialModes[trialCounter % trialModes.count]
        }
    }

    /// At each MTP readback: one timed round for the trial, and the instant
    /// (0 when no trial runs; nothing else reads it).
    static func noteReadback() -> UInt64 {
        lock.withLock {
            guard !trialModes.isEmpty else { return 0 }
            let nanos = DispatchTime.now().uptimeNanoseconds
            if let previous, builtAsExpected {
                trialTimes[previous.mode, default: []].append(nanos - previous.nanos)
            }
            previous = nil
            builtAsExpected = false
            return nanos
        }
    }

    /// A round confirmed at full acceptance: the next round is timed.
    static func noteConfirmed(mode: Int, readbackNanos: UInt64) {
        lock.withLock {
            guard !trialModes.isEmpty else { return }
            previous = (mode, readbackNanos)
        }
    }

    /// At the verify build: hands the model `handle` (nil for the record's
    /// path, which then counts as built as expected).
    static func arm(_ handle: AnyObject?) {
        lock.withLock {
            handoffSlot = handle
            adopted = handle == nil
        }
    }

    /// After the verify build: drops an unadopted handoff.
    static func close() {
        lock.withLock {
            handoffSlot = nil
            builtAsExpected = adopted
            adopted = false
        }
    }
}

extension EngineLoopV2 {
    /// Before the readback of a round whose ids came from a host lookup:
    /// the next verify's leading layers, for full acceptance. With no count
    /// chosen and no trial running it returns at once (no lookup, no
    /// preview, nothing recorded): the record's host path.
    func speculateVerifyLeadingBeforeReadback(_ verify: CBv2MTPRoundInFlight.Verify, step: CBv2InFlightStep) {
        guard CBv2VerifyLeading.speculating else { return }
        let engine = ObjectIdentifier(self)
        _ = CBv2VerifyLeading.takePending(engine: engine)
        guard CBv2VerifyLeading.forced != 0, let mtp, verify.rows.count == 1,
            let metadata = verify.rows.first, !step.discard.contains(metadata.id),
            let drafts = CBv2PromptLookupDraft.hostProposal(metadata.id),
            drafts.count == verify.k, mtp.config.fixedDraftTokens == verify.k,
            let rec = scheduler.record(for: metadata.id),
            rec.request.maxTokens - rec.generatedTokenCount > 2 * verify.k + 1,
            let evaluations = verify.recurrentEvaluations[metadata.id],
            evaluations.count == 1, evaluations[0].isCaptured,
            let state = recurrentStates[metadata.id],
            let target = mtp.model as? any CBv2VerifyLeadingSpeculating
        else { return }
        let promptLength = rec.request.promptTokens.count
        var history = rec.tokens + drafts
        guard
            let bonus = CBv2PromptLookupDraft.continuation(
                history: history, promptLength: promptLength, depth: verify.k + 1)?.ids.first
        else { return }
        history.append(bonus)
        guard
            let next = CBv2PromptLookupDraft.continuation(
                history: history, promptLength: promptLength, depth: verify.k)?.ids
        else { return }
        let mode = CBv2VerifyLeading.modeForSpeculation()
        var handle: AnyObject?
        if mode > 0 {
            guard
                let preview = state.speculativeFullAcceptanceEvaluation(layers: Array(0 ..< mode)),
                let built = target.speculateVerifyLeading(
                    tokens: [Int32(bonus)] + next.map { Int32($0) }, recurrentState: preview,
                    layers: mode)
            else { return }
            handle = built
        }
        CBv2VerifyLeading.setPending(
            CBv2VerifyLeading.Pending(
                engine: engine, id: metadata.id, handle: handle, mode: mode, bonus: bonus,
                next: next, tokensCount: rec.tokens.count + drafts.count + 1))
    }

    /// After the readback, once the round's own lookup ran: keep the
    /// speculation for the next verify build only if everything it assumed
    /// holds.
    func settleVerifyLeading(
        id: CBv2RequestID, accepted: Int, confirmed: Int, k: Int, targets: [Int],
        finished: Bool, lookup: MLXArray?, tokensCount: Int, readbackNanos: UInt64
    ) {
        guard var pending = CBv2VerifyLeading.takePending(engine: ObjectIdentifier(self)) else {
            return
        }
        guard pending.id == id else {
            CBv2VerifyLeading.setPending(pending)
            return
        }
        guard !finished, accepted == k, confirmed == k + 1,
            targets.count == k + 1, targets[k] == pending.bonus, let lookup,
            CBv2PromptLookupDraft.lastHitIDs == pending.next,
            tokensCount == pending.tokensCount
        else { return }
        pending.proposal = lookup
        CBv2VerifyLeading.setPending(pending)
        CBv2VerifyLeading.noteConfirmed(mode: pending.mode, readbackNanos: readbackNanos)
    }

    /// At the verify build: hands the model the speculation when the verify
    /// is exactly the one it was built for. True when something was armed
    /// (then `closeVerifyLeadingHandoff` follows the build).
    func armVerifyLeadingHandoff(_ rows: [CBv2MTPRowWork], k: Int) -> Bool {
        guard let pending = CBv2VerifyLeading.takePending(engine: ObjectIdentifier(self)),
            pending.proposal != nil, rows.count == 1, let row = rows.first,
            row.rec.id == pending.id, let carry = row.carry, carry.token == pending.bonus,
            let early = carry.earlyBlock, early.depth == k, k == pending.next.count,
            early.tokens === pending.proposal
        else { return false }
        CBv2VerifyLeading.arm(pending.handle)
        return true
    }

    /// After the verify build: drops an unadopted handoff.
    func closeVerifyLeadingHandoff() {
        CBv2VerifyLeading.close()
    }
}
