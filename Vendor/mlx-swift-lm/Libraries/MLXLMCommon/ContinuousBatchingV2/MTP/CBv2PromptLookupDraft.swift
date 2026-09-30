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

/// What the model may know about the verify being built: whether its
/// proposal came from the prompt (the lookup or the splice). Set by the
/// engine around a single-row block verify's graph build, false otherwise.
public enum CBv2VerifyRoundHint {
    nonisolated(unsafe) public private(set) static var proposalFromPrompt = false

    /// Runs `body` (the verify build) with the hint set to `fromPrompt`.
    static func building<T>(fromPrompt: Bool, _ body: () throws -> T) rethrows -> T {
        proposalFromPrompt = fromPrompt
        defer { proposalFromPrompt = false }
        return try body()
    }
}

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

    /// Requests whose newest proposal is a host lookup's continuation (the
    /// ids read from the prompt on the host, not a device-side splice).
    nonisolated(unsafe) private static var hostLookup: Set<CBv2RequestID> = []

    /// Records where `id`'s newest proposal came from; `host` when it is a host
    /// lookup's continuation (`lookup`, or `override`'s own match).
    static func noteProposal(_ id: CBv2RequestID, fromPrompt prompt: Bool, host: Bool = false) {
        guard enabled else { return }
        lock.withLock {
            if host { hostLookup.insert(id) } else { hostLookup.remove(id) }
            guard skipEnabled else { return }
            if prompt { fromPrompt.insert(id) } else { fromPrompt.remove(id) }
        }
    }

    /// True when `id`'s newest proposal is a host lookup's continuation.
    static func proposalIsHostLookup(_ id: CBv2RequestID) -> Bool {
        guard enabled else { return false }
        return lock.withLock { hostLookup.contains(id) }
    }

    /// Whether the last `override` returned its host lookup's continuation.
    /// Read by the engine thread right after the call that set it.
    nonisolated(unsafe) static var lastOverrideWasHostLookup = false

    /// True when `id`'s newest proposal came from the prompt, so its next
    /// round looks the continuation up before running the drafter.
    static func expectsPromptProposal(_ id: CBv2RequestID) -> Bool {
        guard enabled, skipEnabled else { return false }
        return lock.withLock { fromPrompt.contains(id) }
    }

    /// `MLXFAST_DFLASH_SPLICE_SPECULATION=0` restores the old gate: no next
    /// block before the readback after any proposal that went through the
    /// prompt path.
    ///
    /// The splice decides on the device, so it returns a new array in every
    /// round it runs, and `noteProposal` marked every drafter block it saw as
    /// a proposal from the prompt: for any prompt longer than the depth the
    /// engine then never built the next block before the readback
    /// (`CBv2MTPDraftBeforeReadback`), even while nothing is quoted. With this
    /// on, only a host lookup's continuation (the output is quoting the
    /// prompt) holds the next block back (`holdsSpeculation`). After a splice
    /// the block is built before the readback as after any drafter block, and
    /// it is dropped, unadopted, whenever the host lookup then hits
    /// (`lookupPreempts`), so the lookup path runs exactly as it did: the
    /// drafter skipped and the round's context rows left pending. A round
    /// the lookup misses adopts the block, which the load-time self-test
    /// proves equal, bit for bit, to `finalizeRound` + `proposeBlock` (ids,
    /// every cached row and cursor); its ids then go through `override` as
    /// before. Drafts, acceptance and tokens are unchanged.
    static let spliceSpeculationEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_SPECULATION"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// True when `id`'s next block may not be built before the readback.
    static func holdsSpeculation(_ id: CBv2RequestID) -> Bool {
        guard expectsPromptProposal(id) else { return false }
        return !spliceSpeculationEnabled || proposalIsHostLookup(id)
    }

    /// True when a block built before the readback must be dropped because
    /// the lookup that runs first (`expectsPromptProposal`) hits.
    static func lookupPreempts(
        _ id: CBv2RequestID, history: [Int], promptLength: Int, depth: Int
    ) -> Bool {
        guard spliceSpeculationEnabled, expectsPromptProposal(id) else { return false }
        return continuation(history: history, promptLength: promptLength, depth: depth) != nil
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

    /// `MLXFAST_DFLASH_SPLICE_FIRST_MIN` sets the shortest alignment at
    /// position 0 in the first round after the seed, when the committed text
    /// is the prompt and the seed token alone: the drafter's block from its
    /// first token plus the seed token running along the same prompt span.
    /// 6 by default, 3 at the least, never above `spliceMinimum` (which it
    /// equals, the record's bar, when set to 8); later positions and later
    /// rounds keep `spliceMinimum`.
    static let spliceFirstMinimum: Int = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_FIRST_MIN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return min(spliceMinimum, max(3, raw.flatMap(Int.init) ?? 6))
    }()

    /// `MLXFAST_DFLASH_SPLICE_TRACE=1` reads the splice's choice back and
    /// prints it. Diagnostic only: the readback waits for the drafter.
    static let spliceTrace: Bool =
        ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_TRACE"] == "1"

    /// The proposal, or the same object when lookup does not apply.
    static func override(
        _ proposal: MLXArray, history: [Int], promptLength: Int, depth: Int
    ) -> MLXArray {
        lastOverrideWasHostLookup = false
        guard enabled, depth > 0, proposal.ndim == 2, proposal.dim(0) == 1,
            proposal.dim(1) == depth
        else { return proposal }
        if let hit = continuation(history: history, promptLength: promptLength, depth: depth) {
            FileHandle.standardError.write(
                Data("dflash2 prompt lookup: match=\(hit.match) depth=\(depth)\n".utf8))
            lastOverrideWasHostLookup = true
            return MLXArray(hit.ids, [1, depth])
        }
        guard spliceEnabled else { return proposal }
        return splice(proposal, history: history, promptLength: promptLength, depth: depth)
            ?? proposal
    }

    /// Integer-only equivalent of the array splice below. Keep the array path
    /// for diagnostics and a runtime comparison/disable switch.
    static let fusedSpliceEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_FUSED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let spliceScore = MLXFast.metalKernel(
        name: "cbv2_prompt_splice_score",
        inputNames: ["block", "prompt", "runs", "dims"], outputNames: ["ranked"],
        source: """

        uint x = thread_position_in_grid.x;
        int n = dims[0], d = dims[1];
        if (x >= n*d) return;
        int j = int(x)/n, c = int(x)%n;
        int minimum = j == 0 ? dims[3] : dims[2];
        int a = 0;
        while (j+a < d && block[j+a] == prompt[c+1+a]) ++a;
        int s = a + (j == 0 ? runs[c] : 0);
        ranked[x] = a > 0 && s >= minimum ? s : 0;
        """, ensureRowContiguous: true)

    private static let splicePick = MLXFast.metalKernel(
        name: "cbv2_prompt_splice_pick",
        inputNames: ["ranked", "block", "prompt", "dims"], outputNames: ["out"],
        source: """

        uint tid = thread_position_in_threadgroup.x;
        int n = dims[0], d = dims[1];
        int bs = 0, bi = 0;
        for (int i = int(tid); i < n*d; i += 256) {
         int s = ranked[i];
         if (s > bs || (s == bs && i < bi)) { bs=s; bi=i; }
        }
        threadgroup int scores[256];
        threadgroup int indices[256];
        scores[tid]=bs; indices[tid]=bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride=128; stride>0; stride>>=1) {
         if (tid < stride) {
          int s=scores[tid+stride], i=indices[tid+stride];
          if (s > scores[tid] || (s == scores[tid] && i < indices[tid])) {
           scores[tid]=s; indices[tid]=i;
          }
         }
         threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid < uint(d)) {
         int j=indices[0]/n, c=indices[0]%n;
         out[tid] = scores[0] > 0 && int(tid) >= j ? prompt[c+1+int(tid)-j] : block[tid];
        }
        """, ensureRowContiguous: true)

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
    /// overlaps the host's finalize. The candidate table is built from the
    /// request's own prompt, every round, and is not kept.
    static func splice(
        _ drafted: MLXArray, history: [Int], promptLength: Int, depth: Int
    ) -> MLXArray? {
        let minimum = spliceMinimum
        let count = history.count
        let prompt = min(max(promptLength, 0), count)
        // The first round after the seed has only the seed token to run
        // along a span, so its position-0 alignments take a lower bar.
        let firstMinimum = count == prompt + 1 ? spliceFirstMinimum : minimum
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

        if fusedSpliceEnabled && !spliceTrace && depth <= 256 {
            let block = drafted.reshaped([depth]).asType(.int32)
            let promptIDs = MLXArray(history[0 ..< prompt].map { Int32($0) })
            let dims = MLXArray([
                Int32(candidates), Int32(depth), Int32(minimum), Int32(firstMinimum),
            ])
            let ranked = spliceScore(
                [block, promptIDs, MLXArray(runs), dims],
                grid: (candidates * depth, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[candidates * depth]], outputDTypes: [.int32])[0]
            return splicePick(
                [ranked, block, promptIDs, dims],
                grid: (256, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[1, depth]], outputDTypes: [.int32])[0].asType(drafted.dtype)
        }

        // The prompt continuation of every alignment, [candidates, depth].
        var table = [Int32]()
        table.reserveCapacity(candidates * depth)
        for c in 0 ..< candidates {
            for t in 0 ..< depth { table.append(Int32(history[c + 1 + t])) }
        }
        // The drafter's block read from position j, [depth, depth], with -1
        // (never a token) past its end, and the first row's run bonus.
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

        let block = drafted.reshaped([depth]).asType(.int32)
        let continuation = MLXArray(table, [1, candidates, depth])
        let shifted = which(
            MLXArray(inside, [depth, 1, depth]),
            take(block, MLXArray(shift, [depth, 1, depth]), axis: 0),
            MLXArray(Int32(-1)))
        // Leading agreement of each (j, c): [depth, candidates].
        let agree = cumprod((shifted .== continuation).asType(.int32), axis: 2).sum(axis: 2)
        let score =
            agree + MLXArray(firstRow, [depth, 1]) * MLXArray(runs, [1, candidates])
        var minima = [Int32](repeating: Int32(minimum), count: depth)
        minima[0] = Int32(firstMinimum)
        let eligible =
            (agree .>= MLXArray(Int32(1))) .&& (score .>= MLXArray(minima, [depth, 1]))
        let ranked = which(eligible, score, MLXArray(Int32(0))).reshaped([depth * candidates])
        let best = argMax(ranked, axis: 0).asType(.int32)
        let fire = take(ranked, best, axis: 0) .> MLXArray(Int32(0))
        let j = floorDivide(best, MLXArray(Int32(candidates)))
        let c = best - j * MLXArray(Int32(candidates))
        let steps = MLXArray((0 ..< depth).map { Int32($0) })
        let source = maximum(c + MLXArray(Int32(1)) + steps - j, MLXArray(Int32(0)))
        let promptIDs = MLXArray(history[0 ..< prompt].map { Int32($0) })
        let spliced = which(steps .< j, block, take(promptIDs, source, axis: 0))
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

    struct Hit {
        let match: Int
        let ids: [Int]
    }

    /// `MLXFAST_DFLASH_SEED_LOOKUP=0` keeps the drafter in the first round.
    ///
    /// In the first round after the seed the committed text is the prompt
    /// and the seed token. When the seed token and the committed tokens
    /// before it run along exactly one prompt position for at least
    /// `MLXFAST_DFLASH_SEED_LOOKUP_MIN` tokens (2 by default), longer than
    /// along any other, the first round's ids are that position's prompt
    /// continuation and the drafter's block is not run (its context rows stay
    /// pending, as after any lookup round).
    static let seedLookupEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SEED_LOOKUP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static let seedLookupMinimum: Int = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SEED_LOOKUP_MIN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return max(2, raw.flatMap(Int.init) ?? 2)
    }()

    /// The first round's ids from the prompt, or nil (then the drafter runs).
    static func seedLookup(history: [Int], promptLength: Int, depth: Int) -> MLXArray? {
        guard enabled, seedLookupEnabled, depth > 0,
            let hit = seedContinuation(history: history, promptLength: promptLength, depth: depth)
        else { return nil }
        FileHandle.standardError.write(
            Data("dflash2 prompt lookup: seed run=\(hit.match) depth=\(depth), drafter skipped\n".utf8))
        return MLXArray(hit.ids, [1, depth])
    }

    /// The prompt continuation after the one position where the seed token
    /// and the tokens before it run longest, or nil when the history is not
    /// the prompt plus the seed token, the longest run is shorter than the
    /// minimum, or two positions tie with different continuations.
    static func seedContinuation(history: [Int], promptLength: Int, depth: Int) -> Hit? {
        let count = history.count
        let prompt = min(max(promptLength, 0), count)
        guard depth >= 1, count == prompt + 1, prompt > depth else { return nil }
        let anchor = history[count - 1]
        var best = 0
        var chosen: [Int]?
        var ambiguous = false
        for c in 0 ..< (prompt - depth) where history[c] == anchor {
            var run = 1
            while run < 64, c - run >= 0, history[c - run] == history[count - 1 - run] {
                run += 1
            }
            if run < best { continue }
            let ids = Array(history[(c + 1) ..< (c + 1 + depth)])
            if run > best {
                (best, chosen, ambiguous) = (run, ids, false)
            } else if chosen != ids {
                ambiguous = true
            }
        }
        guard best >= seedLookupMinimum, !ambiguous, let chosen else { return nil }
        return Hit(match: best, ids: chosen)
    }

    /// `MLXFAST_DFLASH_PREFETCH_ON_MISS=0` absorbs a prompt's drafter context
    /// behind the prompt forward whatever the first round does.
    ///
    /// The prompt row's context absorb (~19 ms of BF16 GEMM on an M4 Max for
    /// 512 rows) is read only by a drafter block. A prompt row that can hit
    /// the seed lookup defers it to its finalize, which submits it only on a
    /// miss; on a hit the rows stay pending for the next block that runs, as
    /// after any lookup round (the absorbed rows are the same either way).
    static let prefetchOnMissEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_PREFETCH_ON_MISS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// True when a prompt row's context absorb waits for its seed token: the
    /// first round has a known depth and could take the seed lookup (whose run
    /// needs the prompt's last token at an earlier prompt position).
    static func defersPromptAbsorb(prompt: [Int], depth: Int?) -> Bool {
        guard prefetchOnMissEnabled, enabled, seedLookupEnabled, let depth, depth > 0,
            prompt.count > depth + 1, let last = prompt.last
        else { return false }
        return prompt[0 ..< (prompt.count - depth - 1)].contains(last)
    }

    /// True when the first round will take its ids from the prompt (the same
    /// test `seedLookup` makes), from the prompt and the seed token.
    static func seedLookupHits(history: [Int], promptLength: Int, depth: Int) -> Bool {
        guard enabled, seedLookupEnabled, depth > 0 else { return false }
        return seedContinuation(history: history, promptLength: promptLength, depth: depth) != nil
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
        // Every longer eligible match contains this suffix and leaves the
        // same continuation inside the prompt. Prove a miss in one scan
        // before scanning all longer lengths; ambiguous hits still use the
        // original longest-match selection below.
        let minimumSuffix = history[(count - minimum) ..< count]
        let lastMinimumStart = prompt - minimum - depth
        guard (0 ... lastMinimumStart).contains(where: { start in
            history[start ..< (start + minimum)].elementsEqual(minimumSuffix)
        }) else { return nil }
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
