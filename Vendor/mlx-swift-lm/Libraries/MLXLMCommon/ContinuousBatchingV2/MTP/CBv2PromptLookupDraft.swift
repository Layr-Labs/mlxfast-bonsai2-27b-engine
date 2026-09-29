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
        int n = dims[0], d = dims[1], minimum = dims[2];
        if (x >= n*d) return;
        int j = int(x)/n, c = int(x)%n;
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
        int n = dims[0], d = dims[1], minimum = dims[2];
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
         out[tid] = scores[0] >= minimum && int(tid) >= j ? prompt[c+1+int(tid)-j] : block[tid];
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
            let promptIDs = MLXArray(history[0 ..< prompt].map { Int32($0) })
            let dims = MLXArray([Int32(candidates), Int32(depth), Int32(minimum)])
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
        let eligible = (agree .>= MLXArray(Int32(1))) .&& (score .>= MLXArray(Int32(minimum)))
        let ranked = which(eligible, score, MLXArray(Int32(0))).reshaped([depth * candidates])
        let best = argMax(ranked, axis: 0).asType(.int32)
        let fire = take(ranked, best, axis: 0) .>= MLXArray(Int32(minimum))
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


// MARK: - The next verify built ahead of the readback

// In a lookup streak the next round's verify is built ahead of this round's
// readback. A row whose draft ids came from the prompt has its NEXT ids known
// before the target answers, provided the target accepts the whole block and
// its bonus token is the prompt's next token: the next block is the prompt's
// continuation, looked up on the history a full acceptance leaves. Then
// everything the next verify reads is known as well: its ids, the KV rows
// after the 1+k accepted columns (already written, the offsets already
// advanced), and the recurrent state the commit at 1+k installs (prepared
// now, installed at the readback). So the first `layers` trunk layers of that
// verify are built while the GPU still runs this round, their boundary
// submissions recorded instead of issued. At the readback a hit issues them at
// once, before the rest of finalize, and the round's build resumes the window
// from layer `layers`: the GPU goes from this verify into the next without the
// host's finalize and first layers in between. A miss drops the prefix:
// nothing of it was submitted, so the rows' lazy writes are put back, the
// speculative recurrent binding is abandoned, and the stock path runs
// untouched. The proposals, the emitted tokens and the acceptance are the
// stock path's; only when the first layers are built and submitted moves.
//
// `MLXFAST_LOOKUP_PIPELINE=0` turns it off; `MLXFAST_LOOKUP_PIPELINE_LAYERS`
// sets the prefix (default 8). `MLXFAST_LOOKUP_PIPELINE_FORCE_MISS=1` builds
// the prefix and always drops it (a test of the miss path).

enum CBv2MTPLookupPipeline {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_LOOKUP_PIPELINE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static let layers: Int = {
        let raw = ProcessInfo.processInfo.environment["MLXFAST_LOOKUP_PIPELINE_LAYERS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return max(1, raw.flatMap(Int.init) ?? 8)
    }()

    static let forcesMiss: Bool =
        ProcessInfo.processInfo.environment["MLXFAST_LOOKUP_PIPELINE_FORCE_MISS"] == "1"
}

/// One row's verify prefix, built ahead of its round. Exactly one of
/// `adopt` (the round's build resumed it) or `discard` ends it.
final class CBv2MTPVerifyPrebuild {
    let id: CBv2RequestID
    let k: Int
    /// The columns this round confirms on a hit: its k draft ids, then the
    /// predicted bonus token, which anchors the next window.
    let confirmedIfHit: [Int]
    let match: Int
    /// `[1, k]` draft ids of the next window, the early block a hit stores.
    let earlyTokens: MLXArray
    /// `[1, 1 + k]`, concatenated as the round's build concatenates them.
    let tokens: MLXArray
    let positionIds: MLXArray?
    let evaluation: CBv2RecurrentStateEvaluation
    let prepared: CBv2RecurrentRequestState.PreparedCommit
    let cursor: AnyObject
    /// The KV offset the next round anchors at (its carry's `kvOffset`).
    let anchorOffset: Int
    private let deferredSubmissions: [[MLXArray]]
    /// The rows' storage and offsets before the prefix, for a miss; released
    /// at a hit (an alias alive at evaluation would forbid the donation the
    /// in-place write relies on), after which only the offsets remain.
    private var restores: [(row: CBv2FullSequenceKV, checkpoint: CBv2FullSequenceKV.Checkpoint)]?
    private let offsets: [(row: CBv2FullSequenceKV, offset: Int)]
    private weak var engine: EngineLoopV2?
    private(set) var submitted = false
    private var resolved = false

    init(
        id: CBv2RequestID, k: Int, confirmedIfHit: [Int], match: Int,
        earlyTokens: MLXArray, tokens: MLXArray, positionIds: MLXArray?,
        evaluation: CBv2RecurrentStateEvaluation,
        prepared: CBv2RecurrentRequestState.PreparedCommit,
        prefix: CBv2VerifyPrefix, anchorOffset: Int,
        checkpoints: [(row: CBv2FullSequenceKV, checkpoint: CBv2FullSequenceKV.Checkpoint)],
        engine: EngineLoopV2
    ) {
        self.id = id
        self.k = k
        self.confirmedIfHit = confirmedIfHit
        self.match = match
        self.earlyTokens = earlyTokens
        self.tokens = tokens
        self.positionIds = positionIds
        self.evaluation = evaluation
        self.prepared = prepared
        self.cursor = prefix.cursor
        self.deferredSubmissions = prefix.deferredSubmissions
        self.anchorOffset = anchorOffset
        self.restores = checkpoints
        self.offsets = checkpoints.map { ($0.row, $0.checkpoint.offset) }
        self.engine = engine
    }

    deinit { discard() }

    /// The hit: the prefix's recorded submissions, in order.
    func submit() {
        precondition(!submitted && !resolved, "CBv2 verify prebuild submitted twice")
        restores = nil
        for targets in deferredSubmissions { asyncEval(targets) }
        submitted = true
    }

    /// The round's build resumed the window.
    func adopt() {
        precondition(submitted && !resolved, "CBv2 verify prebuild adopted without a hit")
        resolved = true
    }

    /// Drop the prefix. Before a hit nothing of it reached the GPU, so the
    /// rows' lazy writes are put back; after one, its K/V rows are written
    /// past the row's offset (a rejected round leaves the same), so only the
    /// offsets come back. Either way the row is rebound before its next use.
    func discard() {
        guard !resolved else { return }
        resolved = true
        if let restores {
            for (row, checkpoint) in restores { row.restore(checkpoint) }
            self.restores = nil
        } else {
            for (row, offset) in offsets { row.rollback(row.absoluteOffset - offset) }
        }
        do { try evaluation.abandon() } catch {
            preconditionFailure("CBv2 verify prebuild binding could not be abandoned: \(error)")
        }
        engine?.eagerCompositionStale = true
    }
}

extension EngineLoopV2 {

    /// Build the next round's verify prefix for the single verify row of a
    /// lookup streak, or nil where the next window is not knowable, not
    /// splittable, or not exactly the stock build. Called before the readback.
    func mtpPrebuildVerify(
        verify: CBv2MTPRoundInFlight.Verify, step: CBv2InFlightStep,
        driver mtp: CBv2MTPRoundDriver
    ) -> CBv2MTPVerifyPrebuild? {
        guard CBv2MTPLookupPipeline.enabled, verify.rows.count == 1,
            let metadata = verify.rows.first, mtp.blockDrafter != nil,
            mtp.usesRequestStatefulDrafter, mtp.config.fixedDraftTokens == verify.k,
            CBv2PromptLookupDraft.expectsPromptProposal(metadata.id),
            !step.discard.contains(metadata.id),
            let rec = scheduler.record(for: metadata.id),
            logitDiagnostic == nil, rec.request.stopStrings.isEmpty,
            let evaluations = verify.recurrentEvaluations[metadata.id],
            evaluations.count == 1, evaluations[0].isCaptured,
            let recurrentModel = mtp.model as? any CBv2RecurrentMTPSteppableModel,
            recurrentModel.recurrentStateSpec != nil,
            recurrentModel.supportsCapturedVerifyWindow,
            !backend.requiresMaterializedSnapshots,
            let recurrentState = recurrentStates[metadata.id],
            let state = kvStates[metadata.id]
        else { return nil }
        let k = verify.k
        let width = 1 + k
        switch mtp.config.verificationMode {
        case .serialTarget: return nil
        case .rectangular, .rectangularExact: break
        case .automatic:
            guard width <= mtp.config.maxAutomaticRectangularTokens else { return nil }
        }
        // A hit confirms 1+k tokens and the row then takes a full next round:
        // the stock early block's own length condition, one round early.
        guard rec.request.maxTokens - rec.generatedTokenCount > 2 * k + 1 else { return nil }
        let promptLength = rec.request.promptTokens.count
        // The ids this round verifies came from the prompt (`override` and
        // `lookup` both take this continuation first, on this history), so
        // the history a full acceptance leaves is known, and the next window
        // with it. The stock lookup on that history must return the same
        // block, or the next round is not this one.
        guard let current = CBv2PromptLookupDraft.continuation(
            history: rec.tokens, promptLength: promptLength, depth: k)
        else { return nil }
        let accepted = rec.tokens + current.ids
        guard let predicted = CBv2PromptLookupDraft.continuation(
            history: accepted, promptLength: promptLength, depth: width),
            let next = CBv2PromptLookupDraft.continuation(
                history: accepted + [predicted.ids[0]], promptLength: promptLength, depth: k),
            next.ids == Array(predicted.ids[1...])
        else { return nil }
        let confirmedIfHit = current.ids + [predicted.ids[0]]
        guard !confirmedIfHit.contains(where: { rec.request.stopTokens.contains($0) }) else {
            return nil
        }
        // Contiguous rows, all at the offset the next round anchors at (this
        // round's columns already written).
        let rows = state.compactMap { $0 }
        let anchorOffset = rec.numComputedTokens
        guard !rows.isEmpty, rows.allSatisfy({ $0.absoluteOffset == anchorOffset }) else {
            return nil
        }
        var checkpoints: [(row: CBv2FullSequenceKV, checkpoint: CBv2FullSequenceKV.Checkpoint)] = []
        for row in rows {
            guard let full = row as? CBv2FullSequenceKV else { return nil }
            checkpoints.append((full, full.checkpoint()))
        }
        // The recurrent commit a hit installs, and the binding over it. Every
        // layer's replay must stay unbuilt, as the stock commit leaves it.
        let prepared: CBv2RecurrentRequestState.PreparedCommit
        do { prepared = try evaluations[0].prepareCommit(keepPositions: width) } catch { return nil }
        guard prepared.allReplaysPending,
            let evaluation = try? recurrentState.bindSpeculative(over: prepared)
        else { return nil }
        func abandon() {
            do { try evaluation.abandon() } catch {
                preconditionFailure("CBv2 verify prebuild binding could not be abandoned: \(error)")
            }
        }
        // The window's ids, built as the round's build builds them from its
        // early block (`mtpBuildVerifyGraph`, `mtpBuildTargetVerification`).
        let earlyTokens = MLXArray(next.ids, [1, k])
        let seedColumn = MLXArray([Int32(predicted.ids[0])]).reshaped([1, 1])
        let draftSteps = (0 ..< k).map { earlyTokens[0..., $0] }
        let columns = [seedColumn] + draftSteps.map { $0.reshaped([1, 1]) }
        let tokens =
            CBv2VerifyTokenStack.tokens(seed: seedColumn, block: earlyTokens, columns: columns)
            ?? concatenated(columns, axis: 1)
        let caches = eagerCaches(rowStates: [state])
        let serializing = caches.compactMap { $0 as? CBv2MTPRectangularSerializing }
        // Plain layer caches with no attention observation bound: a forward
        // built here is not one an observer expects.
        guard serializing.count == caches.count,
            caches.allSatisfy({
                ($0 as? CBv2LayerCache).map {
                    $0.attentionMetadata == nil && $0.attentionPacket == nil
                } ?? false
            })
        else {
            abandon()
            return nil
        }
        for cache in serializing {
            cache.mtpSerializesRectangularAttention = true
            cache.mtpBatchesRectangularAttention = mtp.drafter.prefersBatchedRectangularAttention
        }
        defer {
            for cache in serializing {
                cache.mtpSerializesRectangularAttention = false
                cache.mtpBatchesRectangularAttention = false
            }
        }
        let positionIds = CBv2PositionState.decodePositionIds(
            states: [rec.request.positionState], cacheOffsets: [anchorOffset], length: width)
        guard let prefix = recurrentModel.forwardVerifyPrefix(
            tokens: tokens, caches: caches, recurrentState: [evaluation],
            positionIds: positionIds, layers: CBv2MTPLookupPipeline.layers)
        else {
            for (row, checkpoint) in checkpoints { row.restore(checkpoint) }
            abandon()
            eagerCompositionStale = true
            return nil
        }
        return CBv2MTPVerifyPrebuild(
            id: metadata.id, k: k, confirmedIfHit: confirmedIfHit, match: next.match,
            earlyTokens: earlyTokens, tokens: tokens, positionIds: positionIds,
            evaluation: evaluation, prepared: prepared, prefix: prefix,
            anchorOffset: anchorOffset, checkpoints: checkpoints, engine: self)
    }
}
