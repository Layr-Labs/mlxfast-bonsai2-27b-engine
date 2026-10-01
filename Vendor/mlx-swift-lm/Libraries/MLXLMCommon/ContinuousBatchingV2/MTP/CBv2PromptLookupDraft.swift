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

    /// Requests whose newest proposal went through the splice, with the
    /// splice's device flag: whether it found a prompt span to continue.
    nonisolated(unsafe) private static var spliceSpan: [CBv2RequestID: MLXArray] = [:]

    /// Records where `id`'s newest proposal came from; `host` when it is a host
    /// lookup's continuation (`lookup`, or `override`'s own match); `span` the
    /// splice's flag when the splice ran (`lastSpliceFound`).
    static func noteProposal(
        _ id: CBv2RequestID, fromPrompt prompt: Bool, host: Bool = false, span: MLXArray? = nil
    ) {
        guard enabled else { return }
        lock.withLock {
            if host { hostLookup.insert(id) } else { hostLookup.remove(id) }
            spliceSpan[id] = host ? nil : span
            guard skipEnabled else { return }
            if prompt { fromPrompt.insert(id) } else { fromPrompt.remove(id) }
        }
    }

    /// `MLXFAST_DFLASH_SPLICE_SPAN_HOLD=0` builds the next block before the
    /// readback after every splice, as `spliceSpeculationEnabled` alone does.
    ///
    /// A splice that found a prompt span means the output has started
    /// quoting the prompt, so the next round's host lookup is likely to hit.
    /// A block built before the readback is then dropped unadopted, but its
    /// leading layers were already submitted and run on the GPU ahead of
    /// that round's verify. With this on, a splice that found a span holds
    /// the next block back like a host lookup does (the lookup runs first,
    /// the drafter only on a miss); a splice that found nothing lets it be
    /// built before the readback. The flag is evaluated with the proposal,
    /// ahead of the verify that reads the proposal, so reading it waits for
    /// the drafter's tail at most, never for the verify.
    static let spliceSpanHoldEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_SPAN_HOLD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The last `override`'s splice flag (a boolean scalar, to evaluate with
    /// the proposal), or nil when the splice did not run. Read by the engine
    nonisolated(unsafe) static var lastSpliceFound: MLXArray?

    /// The last prompt's ids as a device int32 row, reused across rounds of
    /// one request while the prompt prefix is unchanged (`history`'s first
    /// `prompt` tokens are append-only; a different prompt or a shrunken
    /// array rebuilds). `splice` rebuilds this copy every round otherwise.
    /// `MLXFAST_DFLASH_SPLICE_PROMPT_CACHE=0` restores the per-round build.
    private static let promptRowCacheEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_PROMPT_CACHE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
    nonisolated(unsafe) private static var promptRowCache: (key: [Int], ids: MLXArray)?

    /// `history[0 ..< prompt]` as a device int32 row, from the cache when the
    /// prefix matches the last request's.
    private static func promptRow(_ history: [Int], _ prompt: Int) -> MLXArray {
        if promptRowCacheEnabled, let held = promptRowCache {
            let key = held.key
            if key.count == prompt, key.withUnsafeBytes({ kb in
                history.withUnsafeBytes { hb in
                    hb.count >= key.count * MemoryLayout<Int>.size
                        && memcmp(kb.baseAddress!, hb.baseAddress!, key.count * MemoryLayout<Int>.size) == 0
                }
            }) {
                return held.ids
            }
        }
        let ids = MLXArray(history[0 ..< prompt].map { Int32($0) })
        if promptRowCacheEnabled {
            promptRowCache = (Array(history[0 ..< prompt]), ids)
        }
        return ids
    }

    /// True when `id`'s newest proposal went through a splice that found a
    /// prompt span (the flag is read once, then kept as a host value).
    private static func spliceFoundSpan(_ id: CBv2RequestID) -> Bool {
        guard spliceSpanHoldEnabled,
            let flag = lock.withLock({ spliceSpan[id] })
        else { return false }
        let found = flag.item(Bool.self)
        lock.withLock {
            if spliceSpan[id] === flag { spliceSpan[id] = found ? flag : nil }
        }
        return found
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

    /// `DARKBLOOM_DFLASH_FIRST_LOOKUP=0` keeps the drafter's block in the
    /// first round after the prompt.
    static let firstRoundEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH_FIRST_LOOKUP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// True when `id`'s next block may not be built before the readback.
    static func holdsSpeculation(_ id: CBv2RequestID) -> Bool {
        guard expectsPromptProposal(id) else { return false }
        return !spliceSpeculationEnabled || proposalIsHostLookup(id) || spliceFoundSpan(id)
    }

    /// True when a block built before the readback must be dropped because
    /// the lookup that runs first (`expectsPromptProposal`) hits.
    static func lookupPreempts(
        _ id: CBv2RequestID, history: [Int], promptLength: Int, depth: Int
    ) -> Bool {
        guard spliceSpeculationEnabled, expectsPromptProposal(id) else { return false }
        return continuation(history: history, promptLength: promptLength, depth: depth) != nil
    }

    /// The first round after the prompt, before any drafter block: its ids
    /// from the prompt, or nil (then the drafter's block and the splice run).
    ///
    /// The committed output is one token here. When that token and the
    /// prompt's own last token run along exactly one prompt position `c`
    /// (`history[c - 1]` and `history[c]` equal the last two committed
    /// tokens) whose next `depth` tokens lie inside the prompt, those tokens
    /// are the block. Two or more such positions are ambiguous and fall back
    /// to the drafter. No block is built: the prompt's context rows stay in
    /// the drafter's cache (absorbed after the prompt forward) or pending,
    /// and the next block that runs takes every committed row, as after any
    /// lookup round. Only the draft changes; the target verifies every id.
    static func firstRoundLookup(history: [Int], promptLength: Int, depth: Int) -> MLXArray? {
        guard enabled, firstRoundEnabled, depth > 0 else { return nil }
        let count = history.count
        let prompt = min(max(promptLength, 0), count)
        guard count == prompt + 1, prompt >= depth + 2 else { return nil }
        let anchor = history[count - 1]
        let previous = history[count - 2]
        var chosen = -1
        for c in 1 ... (prompt - depth - 1)
        where history[c] == anchor && history[c - 1] == previous {
            if chosen >= 0 { return nil }
            chosen = c
        }
        guard chosen >= 0 else { return nil }
        let ids = Array(history[(chosen + 1) ... (chosen + depth)])
        FileHandle.standardError.write(
            Data("dflash2 first-round lookup: position=\(chosen) depth=\(depth), drafter skipped\n".utf8))
        return MLXArray(ids, [1, depth])
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
    /// that already runs along that span. 7 by default, 6 at the least.
    static let spliceMinimum: Int = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_MIN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return max(6, raw.flatMap(Int.init) ?? 7)
    }()

    /// `DARKBLOOM_DFLASH_SPLICE_ANCHOR_MIN` sets the evidence an ANCHORED
    /// alignment needs: block position `j` = 0 at a prompt position whose
    /// committed suffix already runs along the span (run >= 1). 5 by default;
    /// `0` or `off` holds every alignment to `spliceMinimum`.
    ///
    /// The committed run is capped by how much output there is. In the first
    /// round after the seed the output is ONE token, so an anchored alignment
    /// can show at most that token plus the prompt-template tokens that also
    /// precede the quoted span (`\n\n` before the opening fence on the public
    /// captures): a run of 2. Under the general bar of 8 that round's block
    /// is continued only when the drafter is itself right for 6 or more
    /// tokens. The spliced block equals the drafter's on its first `a`
    /// positions, so it can accept fewer tokens than the drafter's block only
    /// when the drafter was right past the point where it and the prompt
    /// part; an anchored alignment that the drafter follows for `a` tokens
    /// risks nothing the drafter got right before `a`. Every other alignment
    /// (`j` > 0, or no committed run) keeps the general bar, and the best
    /// alignment is still the one with the most evidence, so a round the
    /// general bar already fired picks the same continuation.
    static let spliceAnchorMinimum: Int = {
        let raw = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH_SPLICE_ANCHOR_MIN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let raw, ["0", "false", "no", "off"].contains(raw) { return 0 }
        return max(2, raw.flatMap(Int.init) ?? 5)
    }()

    /// `MLXFAST_DFLASH_SPLICE_TRACE=1` (or `DARKBLOOM_DFLASH_SPLICE_TRACE=1`,
    /// which the benchmarker forwards to its worker) reads the splice's choice
    /// back and prints it. Diagnostic only: the readback waits for the drafter.
    static let spliceTrace: Bool =
        ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_TRACE"] == "1"
        || ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH_SPLICE_TRACE"] == "1"

    /// The proposal, or the same object when lookup does not apply.
    static func override(
        _ proposal: MLXArray, history: [Int], promptLength: Int, depth: Int
    ) -> MLXArray {
        lastOverrideWasHostLookup = false
        lastSpliceFound = nil
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
        int n = dims[0], d = dims[1], minimum = dims[2], anchored = dims[3];
        if (x >= n*d) return;
        int j = int(x)/n, c = int(x)%n;
        int a = 0;
        while (j+a < d && block[j+a] == prompt[c+1+a]) ++a;
        int s = a + (j == 0 ? runs[c] : 0);
        // anchored > 0: j = 0 on a committed run takes the lower bar.
        bool eligible = s >= minimum || (anchored > 0 && j == 0 && runs[c] >= 1 && s >= anchored);
        ranked[x] = a > 0 && eligible ? s : 0;
        """, ensureRowContiguous: true)

    private static let splicePick = MLXFast.metalKernel(
        name: "cbv2_prompt_splice_pick",
        inputNames: ["ranked", "block", "prompt", "dims"], outputNames: ["out"],
        source: """

        uint tid = thread_position_in_threadgroup.x;
        int n = dims[0], d = dims[1], minimum = dims[2], anchored = dims[3];
        int floor_ = anchored > 0 && anchored < minimum ? anchored : minimum;
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
         out[tid] = scores[0] >= floor_ && int(tid) >= j ? prompt[c+1+int(tid)-j] : block[tid];
        }
        """, ensureRowContiguous: true)

    /// `MLXFAST_DFLASH_SPLICE_ONELAUNCH=0` keeps the two-launch splice
    /// (score buffer, then pick) above.
    static let spliceOneLaunchEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_DFLASH_SPLICE_ONELAUNCH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The score and pick launches in one threadgroup of 256: each thread
    /// folds its share of the (j, c) grid's leading-agreement scores the same
    /// way the two-launch path does (same `a`, `s`, eligibility and the same
    /// "larger score, smaller index" order, which is associative), then the
    /// same 256-way tree picks the best. `found` is 1 exactly when the two
    /// launches' `ranked.max() >= minimum` fires (the best score reaches the
    /// floor iff some score does). Saves one launch and the ranked buffer's
    /// round trip every round the splice runs.
    private static let spliceFused = MLXFast.metalKernel(
        name: "cbv2_prompt_splice_fused",
        inputNames: ["block", "prompt", "runs", "dims"], outputNames: ["out", "found"],
        source: """

        uint tid = thread_position_in_threadgroup.x;
        int n = dims[0], d = dims[1], minimum = dims[2], anchored = dims[3];
        int floor_ = anchored > 0 && anchored < minimum ? anchored : minimum;
        int bs = 0, bi = 0;
        for (int i = int(tid); i < n*d; i += 256) {
         int j = i / n, c = i - j * n;
         int a = 0;
         while (j+a < d && block[j+a] == prompt[c+1+a]) ++a;
         int s = a + (j == 0 ? runs[c] : 0);
         bool eligible = s >= minimum || (anchored > 0 && j == 0 && runs[c] >= 1 && s >= anchored);
         int r = a > 0 && eligible ? s : 0;
         if (r > bs || (r == bs && i < bi)) { bs=r; bi=i; }
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
        if (tid == 0) found[0] = scores[0] >= floor_ ? 1 : 0;
        if (tid < uint(d)) {
         int j=indices[0]/n, c=indices[0]%n;
         out[tid] = scores[0] >= floor_ && int(tid) >= j ? prompt[c+1+int(tid)-j] : block[tid];
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
            let promptIDs = promptRow(history, prompt)
            let anchored = spliceAnchorMinimum > 0 && spliceAnchorMinimum < minimum ? spliceAnchorMinimum : 0
            let dims = MLXArray([Int32(candidates), Int32(depth), Int32(minimum), Int32(anchored)])
            if spliceOneLaunchEnabled {
                let pair = spliceFused(
                    [block, promptIDs, MLXArray(runs), dims],
                    grid: (256, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: [[1, depth], [1]], outputDTypes: [.int32, .int32])
                lastSpliceFound = pair[1] .!= MLXArray(Int32(0))
                return pair[0].asType(drafted.dtype)
            }
            let ranked = spliceScore(
                [block, promptIDs, MLXArray(runs), dims],
                grid: (candidates * depth, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[candidates * depth]], outputDTypes: [.int32])[0]
            // `splicePick` fires on the best score; a score is 0 or at least
            // `minimum`.
            lastSpliceFound = ranked.max() .>= MLXArray(Int32(minimum))
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
        let firstRows = MLXArray(firstRow, [depth, 1])
        let runColumns = MLXArray(runs, [1, candidates])
        let score = agree + firstRows * runColumns
        let agreeing = agree .>= MLXArray(Int32(1))
        var eligible = agreeing .&& (score .>= MLXArray(Int32(minimum)))
        // Anchored alignments (j = 0 on a committed run) take the lower bar.
        let anchorMinimum = spliceAnchorMinimum
        var fireFloor = minimum
        if anchorMinimum > 0, anchorMinimum < minimum {
            let anchored =
                (firstRows .== MLXArray(Int32(1))) .&& (runColumns .>= MLXArray(Int32(1)))
            eligible =
                eligible
                .|| (anchored .&& agreeing .&& (score .>= MLXArray(Int32(anchorMinimum))))
            fireFloor = anchorMinimum
        }
        let ranked = which(eligible, score, MLXArray(Int32(0))).reshaped([depth * candidates])
        let best = argMax(ranked, axis: 0).asType(.int32)
        let fire = take(ranked, best, axis: 0) .>= MLXArray(Int32(fireFloor))
        let j = floorDivide(best, MLXArray(Int32(candidates)))
        let c = best - j * MLXArray(Int32(candidates))
        let steps = MLXArray((0 ..< depth).map { Int32($0) })
        let source = maximum(c + MLXArray(Int32(1)) + steps - j, MLXArray(Int32(0)))
        let promptIDs = promptRow(history, prompt)
        let spliced = which(steps .< j, block, take(promptIDs, source, axis: 0))
        let proposal = which(fire, spliced, block).reshaped([1, depth]).asType(drafted.dtype)
        if spliceTrace {
            let open = which(agreeing, score, MLXArray(Int32(0))).reshaped([depth * candidates])
            let openBest = argMax(open, axis: 0).asType(.int32)
            let openScore = take(open, openBest, axis: 0)
            let bestScore = take(ranked, best, axis: 0)
            eval(best, fire, bestScore, openBest, openScore)
            let flat = Int(best.item(Int32.self))
            let openFlat = Int(openBest.item(Int32.self))
            let fired: Bool = fire.item(Bool.self)
            let chosenScore = Int(bestScore.item(Int32.self))
            let anyScore = Int(openScore.item(Int32.self))
            let anyRun = Int(runs[openFlat % candidates])
            var line = "dflash2 prompt splice: fire=\(fired) j=\(flat / candidates)"
            line += " c=\(flat % candidates) score=\(chosenScore)"
            line += " best-any=\(anyScore)@j\(openFlat / candidates)c\(openFlat % candidates)"
            line += " run=\(anyRun) committed=\(count - prompt) depth=\(depth)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
        lastSpliceFound = fire
        return proposal
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
    // Each continuation endpoint contributes its longest matching suffix.
    // Shorter suffixes add endpoints but cannot remove existing ambiguity:
    // if two maximal matches have different continuations, every shorter
    // length retains both. Thus only the global longest match needs selection.
    static func continuation(history: [Int], promptLength: Int, depth: Int) -> Hit? {
        let count = history.count
        let prompt = min(max(promptLength, 0), count)
        guard depth >= 1, prompt >= minimumMatch + depth, count >= minimumMatch else { return nil }
        let longest = min(64, count - depth, prompt - depth)
        guard longest >= minimumMatch else { return nil }
        var best = minimumMatch - 1
        var chosenEnd: Int? = nil
        var ambiguous = false
        for end in minimumMatch ... (prompt - depth) {
            guard history[end - 1] == history[count - 1] else { continue }
            var length = 1
            let limit = min(longest, end)
            while length < limit && history[end - length - 1] == history[count - length - 1] {
                length += 1
            }
            guard length >= minimumMatch, length >= best else { continue }
            if length > best {
                best = length
                chosenEnd = end
                ambiguous = false
            } else if let chosenEnd {
                for offset in 0 ..< depth where history[chosenEnd + offset] != history[end + offset] {
                    ambiguous = true
                    break
                }
            }
        }
        guard let chosenEnd, !ambiguous else { return nil }
        return Hit(match: best, ids: Array(history[chosenEnd ..< chosenEnd + depth]))
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
