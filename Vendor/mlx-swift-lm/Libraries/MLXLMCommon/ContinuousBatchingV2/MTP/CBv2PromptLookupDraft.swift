// A long exact copy of the prompt, proposed in place of the drafter's block.
//
// The drafter never reads token ids. When the committed text repeats a span
// of the prompt, the tokens that followed that span are already known, and a
// block drafter still spends a round re-deriving them. This replaces the
// draft ids — and only the draft ids — when a unique prompt span of at least
// `minimumMatch` tokens matches the committed suffix and the following
// `depth` tokens sit in the prompt. The target still verifies every id and
// still emits its own greedy tokens. A miss returns the same proposal object,
// so the graph, the kernels and the values are unchanged.
//
// Short matches are not used. A rank-merge at n = 2...4 replaced correct
// drafter tokens and added rounds (9ac82eeb, 12 rounds -> 15). A unique
// 16-token prompt span does not.

import Foundation
import MLX

enum CBv2PromptLookupDraft {
    /// `MLXFAST_DFLASH_LOOKUP=0` keeps the drafter's block.
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_LOOKUP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `MLXFAST_DFLASH_LOOKUP_MIN` sets the shortest suffix that may replace a
    /// block. 16 is the floor: long enough that an accidental repeat is rare,
    /// short enough to catch a prompt that the output is quoting.
    static let minimumMatch: Int = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_LOOKUP_MIN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return max(8, raw.flatMap(Int.init) ?? 16)
    }()

    /// `MLXFAST_DFLASH_LOOKUP_SKIP=0` runs the drafter's block in every round.
    ///
    /// While the output quotes the prompt, the next round's ids come from the
    /// prompt, and the drafter's block forward (its heaviest work: every
    /// drafter weight read for sixteen rows) produces a proposal nobody uses.
    /// So after a round whose proposal came from the prompt, the engine does
    /// not build the next block before the readback; after the readback it
    /// looks the continuation up first and runs the drafter only on a miss.
    /// The committed context rows of a skipped round stay pending in the
    /// drafter's state (a layer's context keys and values are a function of
    /// those rows alone), and the next block that runs absorbs them all.
    /// Only the draft changes; the target verifies every id.
    static let skipEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_LOOKUP_SKIP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let lock = NSLock()
    /// Requests whose newest proposal came from the prompt.
    nonisolated(unsafe) private static var fromPrompt: Set<CBv2RequestID> = []

    /// Records where `id`'s newest proposal came from.
    static func noteProposal(_ id: CBv2RequestID, fromPrompt prompt: Bool) {
        guard enabled, skipEnabled else { return }
        lock.withLock {
            if prompt { fromPrompt.insert(id) } else { fromPrompt.remove(id) }
        }
    }

    /// True when `id`'s newest proposal came from the prompt, so its next
    /// round looks the continuation up before running the drafter.
    static func expectsPromptProposal(_ id: CBv2RequestID) -> Bool {
        guard enabled, skipEnabled else { return false }
        return lock.withLock { fromPrompt.contains(id) }
    }

    /// The next round's ids from the prompt, or nil (then the drafter runs).
    static func lookup(history: [Int], promptLength: Int, depth: Int) -> MLXArray? {
        guard enabled, depth > 0,
            let hit = continuation(history: history, promptLength: promptLength, depth: depth)
        else { return nil }
        FileHandle.standardError.write(
            Data("dflash2 prompt lookup: match=\(hit.match) depth=\(depth), drafter skipped\n".utf8))
        return MLXArray(hit.ids, [1, depth])
    }

    /// `MLXFAST_DFLASH_SPLICE=0` keeps the host lookup alone (the block is
    /// then the drafter's whenever no unique 16-token prompt span matches).
    static let spliceEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// `MLXFAST_DFLASH_SPLICE_MIN` sets the shortest alignment the splice
    /// accepts: the drafter's own tokens that equal a prompt span, plus (when
    /// the drafter's block agrees from its first token) the committed suffix
    /// that already runs along that span. 8 by default, 6 at the least.
    static let spliceMinimum: Int = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_MIN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return max(6, raw.flatMap(Int.init) ?? 8)
    }()

    /// `MLXFAST_DFLASH_SPLICE_TRACE=1` reads the splice's choice back and
    /// prints it. Diagnostic only: the readback waits for the drafter.
    static let spliceTrace: Bool =
        ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_TRACE"] == "1"

    /// The proposal, or the same object when lookup does not apply.
    static func override(
        _ proposal: MLXArray, history: [Int], promptLength: Int, depth: Int
    ) -> MLXArray {
        guard enabled, depth > 0, proposal.ndim == 2, proposal.dim(0) == 1,
            proposal.dim(1) == depth
        else { return proposal }
        if let hit = continuation(history: history, promptLength: promptLength, depth: depth) {
            FileHandle.standardError.write(
                Data("dflash2 prompt lookup: match=\(hit.match) depth=\(depth)\n".utf8))
            return MLXArray(hit.ids, [1, depth])
        }
        guard spliceEnabled else { return proposal }
        return splice(proposal, history: history, promptLength: promptLength, depth: depth)
            ?? proposal
    }

    /// The drafter's block, continued along the prompt span it is quoting.
    ///
    /// The host lookup above needs 16 committed tokens that run along one
    /// prompt span, so the round in which the output starts quoting its prompt
    /// still gets the drafter's block, and the drafter's extension rows lose
    /// the quote after a few tokens. Those rows' leading tokens are themselves
    /// evidence: when the drafter's block, from position `j` on, equals a
    /// prompt span for `a` tokens, the span's next tokens are proposed after
    /// them. Position `j` = 0 also counts the committed suffix that already
    /// runs along the same span. The best alignment (longest evidence, then
    /// the smallest `j`, then the earliest span) is taken when its evidence
    /// reaches `spliceMinimum`; otherwise the drafter's block is kept.
    ///
    /// The proposal equals the drafter's block on its first `j + a`
    /// positions, so a round accepts less than the drafter's block would only
    /// when the drafter was right past the point where it and the prompt part.
    ///
    /// Everything is lazy device work beside the drafter's own graph: the
    /// drafter's ids are never read on the host, so the early block still
    /// overlaps the host's finalize. The candidate table is the prompt's own
    /// tokens. `CBv2PromptSpliceReuse` keeps that table, the depth-only index
    /// arrays, and the scalar int32s; the committed-suffix runs are rebuilt
    /// because they follow the generated text.
    static func splice(
        _ drafted: MLXArray, history: [Int], promptLength: Int, depth: Int
    ) -> MLXArray? {
        let minimum = spliceMinimum
        let count = history.count
        let prompt = min(max(promptLength, 0), count)
        // Alignment c: the drafter's token at position j + t is compared with
        // prompt token c + 1 + t. Every c keeps its whole continuation inside
        // the prompt (c + depth <= prompt - 1).
        let candidates = prompt - depth
        guard depth >= 1, count >= 1, candidates >= 1 else { return nil }

        // The committed suffix's run along each alignment, on the host: the
        // largest r with history[c - r + 1 ... c] equal to the last r tokens.
        let anchor = history[count - 1]
        var runs = [Int32](repeating: 0, count: candidates)
        for c in 0 ..< candidates where history[c] == anchor {
            var length = 1
            while length < 64, c - length >= 0,
                history[c - length] == history[count - 1 - length]
            {
                length += 1
            }
            runs[c] = Int32(length)
        }

        let geom = CBv2PromptSpliceReuse.geometry(depth: depth)
        let scalars = CBv2PromptSpliceReuse.scalars(minimum: minimum)
        let kept = CBv2PromptSpliceReuse.table(
            history: history, prompt: prompt, depth: depth, candidates: candidates)
        let block = drafted.reshaped([depth]).asType(.int32)
        let shifted = which(
            geom.inside,
            take(block, geom.shift, axis: 0),
            scalars.negOne)
        // Leading agreement of each (j, c): [depth, candidates].
        let agree = cumprod((shifted .== kept.continuation).asType(.int32), axis: 2).sum(axis: 2)
        let score =
            agree + geom.firstRow * MLXArray(runs, [1, candidates])
        let eligible = (agree .>= scalars.one) .&& (score .>= scalars.minimum)
        let ranked = which(eligible, score, scalars.zero).reshaped([depth * candidates])
        let best = argMax(ranked, axis: 0).asType(.int32)
        let fire = take(ranked, best, axis: 0) .>= scalars.minimum
        let j = floorDivide(best, kept.candidates)
        let c = best - j * kept.candidates
        let source = maximum(c + scalars.one + geom.steps - j, scalars.zero)
        let spliced = which(geom.steps .< j, block, take(kept.promptIDs, source, axis: 0))
        let proposal = which(fire, spliced, block).reshaped([1, depth]).asType(drafted.dtype)
        if spliceTrace {
            eval(best, fire)
            let flat = best.item(Int32.self)
            FileHandle.standardError.write(
                Data(
                    ("dflash2 prompt splice: fire=\(fire.item(Bool.self)) "
                        + "j=\(Int(flat) / candidates) c=\(Int(flat) % candidates) "
                        + "depth=\(depth)\n").utf8))
        }
        return proposal
    }


/// Depth-only indexes, scalar int32s, and the prompt continuation table the
/// splice reads. Each is the same values the splice used to build every
/// round. The runs along the committed suffix are not here: they change
/// when a token is accepted.
enum CBv2PromptSpliceReuse {
    private static func flag(_ name: String) -> Bool {
        let value = ProcessInfo.processInfo.environment[name]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }

    /// `DARKBLOOM_DFLASH_SPLICE_GEOM=0` rebuilds the depth-only indexes.
    private static let reuseGeometry = flag("DARKBLOOM_DFLASH_SPLICE_GEOM")
    /// `DARKBLOOM_DFLASH_SPLICE_SCALARS=0` rebuilds the scalar int32s.
    private static let reuseScalars = flag("DARKBLOOM_DFLASH_SPLICE_SCALARS")
    /// `DARKBLOOM_DFLASH_SPLICE_TABLE=0` rebuilds the prompt continuation.
    private static let reuseTable = flag("DARKBLOOM_DFLASH_SPLICE_TABLE")

    private static let lock = NSLock()
    private struct Geometry {
        let depth: Int
        let shift: MLXArray
        let inside: MLXArray
        let firstRow: MLXArray
        let steps: MLXArray
    }
    private struct Scalars {
        let minimum: Int
        let negOne: MLXArray
        let zero: MLXArray
        let one: MLXArray
        let minimumArray: MLXArray
    }
    private struct Table {
        let prompt: Int
        let depth: Int
        let candidates: Int
        let tokens: [Int]
        let continuation: MLXArray
        let promptIDs: MLXArray
        let candidatesArray: MLXArray
    }
    nonisolated(unsafe) private static var geometryCache: Geometry?
    nonisolated(unsafe) private static var scalarCache: Scalars?
    nonisolated(unsafe) private static var tableCache: Table?

    struct Indexes {
        let shift: MLXArray
        let inside: MLXArray
        let firstRow: MLXArray
        let steps: MLXArray
    }

    struct Constants {
        let negOne: MLXArray
        let zero: MLXArray
        let one: MLXArray
        let minimum: MLXArray
    }

    struct Kept {
        let continuation: MLXArray
        let promptIDs: MLXArray
        let candidates: MLXArray
    }

    /// `shift`, `inside`, `firstRow`, and `steps` for this depth.
    static func geometry(depth: Int) -> Indexes {
        if reuseGeometry, let hit = lock.withLock({ geometryCache }), hit.depth == depth {
            return Indexes(shift: hit.shift, inside: hit.inside, firstRow: hit.firstRow, steps: hit.steps)
        }
        var shift = [Int32]()
        var inside = [Bool]()
        shift.reserveCapacity(depth * depth)
        inside.reserveCapacity(depth * depth)
        for j in 0 ..< depth {
            for t in 0 ..< depth {
                shift.append(Int32(min(j + t, depth - 1)))
                inside.append(j + t < depth)
            }
        }
        var firstRow = [Int32](repeating: 0, count: depth)
        firstRow[0] = 1
        let made = Geometry(
            depth: depth,
            shift: MLXArray(shift, [depth, 1, depth]),
            inside: MLXArray(inside, [depth, 1, depth]),
            firstRow: MLXArray(firstRow, [depth, 1]),
            steps: MLXArray((0 ..< depth).map { Int32($0) }))
        if reuseGeometry {
            lock.withLock { geometryCache = made }
        }
        return Indexes(
            shift: made.shift, inside: made.inside, firstRow: made.firstRow, steps: made.steps)
    }

    /// The int32 scalars `-1`, `0`, `1`, and `minimum`.
    static func scalars(minimum: Int) -> Constants {
        if reuseScalars, let hit = lock.withLock({ scalarCache }), hit.minimum == minimum {
            return Constants(
                negOne: hit.negOne, zero: hit.zero, one: hit.one, minimum: hit.minimumArray)
        }
        let made = Scalars(
            minimum: minimum,
            negOne: MLXArray(Int32(-1)),
            zero: MLXArray(Int32(0)),
            one: MLXArray(Int32(1)),
            minimumArray: MLXArray(Int32(minimum)))
        if reuseScalars {
            lock.withLock { scalarCache = made }
        }
        return Constants(
            negOne: made.negOne, zero: made.zero, one: made.one, minimum: made.minimumArray)
    }

    /// The prompt continuation `[1, candidates, depth]`, the prompt ids, and
    /// the candidate count. A miss rebuilds them from `history[0..<prompt]`.
    static func table(history: [Int], prompt: Int, depth: Int, candidates: Int) -> Kept {
        if reuseTable, let hit = lock.withLock({ tableCache }),
            hit.prompt == prompt, hit.depth == depth, hit.candidates == candidates,
            hit.tokens.count == prompt, hit.tokens.elementsEqual(history.prefix(prompt))
        {
            return Kept(
                continuation: hit.continuation, promptIDs: hit.promptIDs,
                candidates: hit.candidatesArray)
        }
        var rows = [Int32]()
        rows.reserveCapacity(candidates * depth)
        for c in 0 ..< candidates {
            for t in 0 ..< depth { rows.append(Int32(history[c + 1 + t])) }
        }
        let made = Table(
            prompt: prompt,
            depth: depth,
            candidates: candidates,
            tokens: Array(history.prefix(prompt)),
            continuation: MLXArray(rows, [1, candidates, depth]),
            promptIDs: MLXArray(history[0 ..< prompt].map { Int32($0) }),
            candidatesArray: MLXArray(Int32(candidates)))
        if reuseTable {
            lock.withLock { tableCache = made }
        }
        return Kept(
            continuation: made.continuation, promptIDs: made.promptIDs,
            candidates: made.candidatesArray)
    }
}

    struct Hit {
        let match: Int
        let ids: [Int]
    }

    /// Longest unique prompt continuation of `history`'s suffix, or nil.
    ///
    /// The continuation has to lie entirely inside the prompt. Two prompt
    /// spans of the same length with different continuations are ambiguous,
    /// and this length is skipped rather than guessed.
    static func continuation(history: [Int], promptLength: Int, depth: Int) -> Hit? {
        let minimum = minimumMatch
        let count = history.count
        let prompt = min(max(promptLength, 0), count)
        guard depth >= 1, prompt >= minimum + depth, count >= minimum else { return nil }
        let longest = min(64, count - depth, prompt - depth)
        guard longest >= minimum else { return nil }
        for length in stride(from: longest, through: minimum, by: -1) {
            let suffix = count - length
            let lastStart = prompt - length - depth
            if lastStart < 0 { continue }
            var chosen: [Int]?
            var ambiguous = false
            var start = lastStart
            while start >= 0 {
                if history[start] == history[suffix],
                    history[start + length - 1] == history[count - 1]
                {
                    var same = true
                    var offset = 1
                    while offset < length - 1 {
                        if history[start + offset] != history[suffix + offset] {
                            same = false
                            break
                        }
                        offset += 1
                    }
                    if same {
                        let from = start + length
                        let ids = Array(history[from ..< (from + depth)])
                        if let chosen, chosen != ids {
                            ambiguous = true
                            break
                        }
                        chosen = ids
                    }
                }
                start -= 1
            }
            if let chosen, !ambiguous {
                return Hit(match: length, ids: chosen)
            }
        }
        return nil
    }
}

/// Whether the next verify forward has no drafter block queued ahead of it on
/// the GPU (a round whose ids came from the prompt with the drafter skipped).
/// The verify's early-submission plan assumes a ~6 ms block ahead to hide the
/// host's first layers; without it the GPU waits for them, so that verify
/// submits sooner (`Qwen35TrunkSubmission.verifyUnqueued`). Set and taken on
/// the engine thread, which builds the verify right after the finalize that
/// sets it.
public enum CBv2VerifyQueueHint {
    nonisolated(unsafe) private static var nothingAhead = false

    public static func markNothingAhead() { nothingAhead = true }

    /// The hint for the verify being built now, cleared as it is read.
    public static func takeNothingAhead() -> Bool {
        defer { nothingAhead = false }
        return nothingAhead
    }
}
