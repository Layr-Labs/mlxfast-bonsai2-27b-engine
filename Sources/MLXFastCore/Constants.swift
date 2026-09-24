public enum MLXFastConstants {
    // Ternary Bonsai 2 27B track identity (`bonsai2-27b-mlx-v1`).
    //
    // The pinned target is an MLX 2-bit affine pack published as
    // `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`. It declares `model_type`
    // "prism_hadamard_qwen35" at the top level and carries a nested
    // `text_config` whose own `model_type` is "qwen3_5_text". Every packed
    // linear folds a signed block Walsh-Hadamard transform into its weights,
    // and `hadamard.json` is the manifest for that transform.
    //
    // THE PACK HAS NO SPECULATIVE HEAD (`mtp_num_hidden_layers: 0`). This
    // track's head is a SEPARATE pinned export,
    // `EigenLabs/Qwen3.8-27B-MTP-4bit`, staged beside the target and named to
    // the engine with `--drafter`. It is used unmodified.
    //
    // RETIRED-NAME CHECK (AGENTS.md). The retired names are `gemma4-31b-it`,
    // `MLXFAST_MTP_`, `mtp-ranked`, `measure-mtp-job`, `mtp-weights` and
    // `laguna-xs-2.1-mtp`. `bonsai2-27b-mlx-v1` is substring-clean
    // against every one of them, and this track introduces no
    // `MLXFAST_MTP_`-prefixed environment name.
    //
    // WHAT THE PIN COSTS, stated instead of discovered. `Golden.swift`
    // validates the provenance block of every golden against
    // repository+revision, so a golden recorded against any other checkpoint
    // is REJECTED against these constants. That is the fail-closed direction:
    // the goldens are hardware-generated and are recorded on the ranked box.
    // The public captures for local runs are R2 objects that the contract
    // fixture pins in `public_captures`; `tools/fetch-goldens.sh --public`
    // fetches them. They are never in git.
    public static let referenceModelRepository =
        "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
    public static let referenceModelRevision =
        "3f926b415992eaa2ae9dd7b573706494d6bbf787"
    public static let referenceModelName =
        "Ternary-Bonsai-2-27B-mlx-2bit"
    public static let defaultReferencePath =
        "reference_weights/Ternary-Bonsai-2-27B-mlx-2bit"
    public static let defaultReferenceCachePath =
        ".cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/3f926b415992eaa2ae9dd7b573706494d6bbf787"
    public static let defaultWeightsPath = "weights"
    public static let defaultGoldenPath = "correctness_golden.json"
    public static let defaultScorePath = "score.json"
    public static let defaultLocalIterateScorePath = "score.local-iterate.json"

    // The model identity every golden must declare in its `model_type` key.
    // SINGLE SOURCE for this fork, the way benchd single-sources it as
    // `bench_core::constants::REQUIRED_GOLDEN_MODEL_TYPE`: the loader wrapper
    // in Golden.swift is the only consumer, so no call site gets to spell the
    // literal again and drift from it.
    //
    // This is the pinned target's `model_type`: the pack declares
    // "prism_hadamard_qwen35" at its top level. Its nested `text_config`
    // declares "qwen3_5_text", which names the BACKBONE and not the artifact;
    // a golden recorded against this pack must name the artifact, because the
    // packed weights are what produced it.
    //
    // Deliberately NOT unified with the frozen-invariant `model_type` check in
    // Sources/MLXFastTransform. That one reads the transformed WEIGHTS
    // config.json; this one reads a GOLDEN document. They are separate
    // contracts and the reference keeps them apart for the same reason.
    public static let requiredGoldenModelType = "prism_hadamard_qwen35"

    // Frozen geometry of
    // prism-ml/Ternary-Bonsai-2-27B-mlx-2bit @ 3f926b41, READ OFF the pinned
    // revision's own config.json.
    //
    // This block, the checkpoint validator in Sources/MLXFastTransform and the
    // contract fixture's `target.*` geometry move as ONE SET. A gate holding
    // some fields of one model and some of another rejects every checkpoint
    // and explains none of them. MLXFastCore is trusted and cannot import the
    // editable model target, so this block and
    // `PrismHadamardCheckpointValidation.PinnedGeometry` are deliberately
    // duplicated and move in lockstep.
    //
    // ONLY WHAT THE TRUSTED PATHS READ LIVES HERE. The tower has a great deal
    // more geometry -- the gated-delta-net widths, the packed module list, the
    // attention schedule -- and all of it is pinned ONCE, in the transform
    // validator, which is the code that checks it. Mirroring the rest here
    // would give a repin two places to move and one of them no test.
    //
    // THE TOWER IS HYBRID. 64 layers on a fixed schedule: every fourth layer
    // is full attention and the other three are gated-delta-net linear
    // attention. So 16 layers carry a key-value cache and 48 carry a
    // constant-size recurrent state.
    public static let vocabSize = 248_320
    public static let hiddenSize = 5_120
    public static let numHiddenLayers = 64

    // 512: David's 2026-09-18 seed-length ruling for this track. The source
    // tracks used 1024 (2026-08-24 ruling) and 512 before that. The decode
    // window is unchanged (`benchmarkDecodeSteps` stays 128); golden shape is
    // 512 prompt_tokens + 129 expected_tokens (seed next-token + 128 checked
    // steps). The contract fixture's `seed_tokens` carries the same value and
    // the benchmarker reads it from there.
    public static let correctnessPromptTokens = 512
    // Keep the public gate long enough to catch broad decode regressions while
    // leaving budget for the hidden GPQA behavior checks in the official job.
    public static let correctnessSteps = 64
    public static let correctnessTopLogits = 8
    public static let correctnessLogitTieTolerance = 1e-6
    public static let correctnessMaxAnchorContextTokens = 1_024
    public static let correctnessMaxFreeRunSteps = 256
    public static let correctnessMaxBehaviorPromptTokens = 2_048
    public static let correctnessMaxBehaviorSteps = 128
    public static let correctnessGPQACaseCount = 9
    // Cross-machine greedy decode can drift on hidden GPQA even with pinned
    // Swift/MLX. Semantic GPQA behavior captures a short continuation for the
    // private judge; exact token enforcement stays on the long copy gate and
    // non-semantic behavior fixtures.
    // 128 (was 64; before that 10, DeepSeek-era): the 10->64 history and its
    // calibration runs predate the GPQA prompt-encoding (BOS) fix and
    // measured degenerate no-BOS completions, so they no longer bind. With
    // BOS the reference answers letter-first and then explains; 128 lets the
    // explanation finish for the judge instead of cutting mid-sentence.
    // Generation happens in the untimed gates phase (never the frozen timed
    // window), so the cost is ~10-15s of job wall-clock, not score.
    public static let correctnessGPQAMaxNewTokens = 128
    // Semantic judging uses short hidden GPQA answers as a baseline-calibrated
    // gate for optimizations that preserve the exact prefix but damage answer
    // sense. 9 (was 5): raised to the full fixture together with the GPQA
    // prompt-encoding (BOS) fix -- selection takes the first N budget-valid
    // cases in file order, and the old window of 5 contained only two of the
    // five cases the correctly-prompted reference answers right. Per-case
    // cost is one short untimed generation plus one judge call; the 4 extra
    // cases add roughly a minute to the job.
    // The captured answer is a prefix of the behavior-gate generation, so
    // semanticGPQAMaxNewTokens is only effective up to
    // correctnessGPQAMaxNewTokens (and must stay <= correctnessMaxBehaviorSteps).
    public static let semanticGPQACaseCount = 9
    public static let semanticGPQAMaxNewTokens = 128
    // 7 of 9, set from measurement rather than prediction. The gate compares
    // the candidate against the pinned reference model's own recorded answers
    // (accepted_responses in the hidden fixture), so it is a regression check:
    // an unmodified candidate reproduces the reference on every case by
    // construction, independent of whether those answers are factually right.
    // That is the point of the design -- the reference model is at chance on
    // these questions, so a correctness-based gate could only ever sit on the
    // noise floor (see the 2026-07-27 measurements: 1-4 of 9 correct depending
    // on option order, with a single case carrying the entire margin).
    // Calibration, 2026-07-27, offline against the real gate script:
    //   self-match (unmodified candidate), 27 runs / 243 judgements: 9/9 every
    //     run, zero variance, including the one case whose reference output is
    //     degenerate.
    //   three answers changed to a different option, 8 runs: exactly 6/9 every
    //     run, failing only the changed cases.
    //   answer content preserved but label flipped or tail truncated, 8 runs:
    //     9/9 -- cosmetic near-tie drift is tolerated.
    // So judge nondeterminism costs nothing, each damaged answer costs exactly
    // one case, and a floor of 7 absorbs two independent damaged answers. It
    // also clears the >= 6 needed to reject a submission that hardcodes one
    // fixed letter: the reference selects a spread of letters, so a constant
    // answer matches at most 5 of 9.
    // Earlier floors of 1 were calibrated against pre-BOS-fix runs whose
    // reference never answered at all, so they justified nothing.
    // Keep in sync with the workflow env MLXFAST_SEMANTIC_GPQA_MIN_PASS and
    // run-semantic-gpqa-gate.sh. Regenerating the fixture's accepted_responses
    // (new prompts, token budget, or reference checkpoint) invalidates this
    // calibration -- re-run it.
    public static let semanticGPQAMinPassCount = 7
    // 512: moves with `correctnessPromptTokens`. A baseline or calibration
    // value derived at another prefill window does not apply.
    public static let benchmarkPrefillPromptTokens = 512
    // Scored decode is parent-measured wall time for decode setup plus this
    // many checked token steps. Charging setup prevents submitted model code
    // from precomputing future decode tokens in an unscored seed-prefill phase.
    public static let benchmarkDecodeSteps = 128
    // Local iterate charges the same 512-token seed prefill as the official
    // decode window, so it must use the same denominator to produce a
    // comparable decode seconds-per-token estimate.
    public static let localIterateBenchmarkDecodeSteps = benchmarkDecodeSteps
    // Local submit uses a longer public fixture so the Yukon pre-submit hook
    // exercises one continuous decode trajectory for about ten minutes instead
    // of repeating the short local-iterate correctness window.
    public static let localSubmitBenchmarkDecodeSteps = 1023
    public static let localSubmitBenchmarkRepeats = 1
    // Seed measured decode with the full prompt. A short instruction-prefix
    // seed can free-run differently across Apple Silicon/MLX versions even
    // when teacher-forced correctness agrees, which makes the timed oracle
    // fragile for reasons unrelated to kernel performance.
    // 512: moves with `correctnessPromptTokens` /
    // `benchmarkPrefillPromptTokens`.
    public static let benchmarkDecodeSeedTokens = 512
    // Official paired timing runs LAST at workflow level, after correctness,
    // GPQA, and the hidden-material scrub. The on-box wrapper launches the
    // baseline and candidate in fresh worker processes, and each timed prefill
    // starts without an in-process warmup. The calibration below used that
    // same shape, so keep zero warmup and one measured run.
    public static let benchmarkPrefillWarmupRuns = 0
    public static let benchmarkPrefillTimedRuns = 1
    // Acceptance bands (see AcceptanceBand + docs/benchmark-window-freeze.md).
    // Prefill and decode are noisy single measurements, gated against the same-VM
    // paired baseline B (which cancels host-speed differences). Each run's value must
    // land within [B*(1-down), B*(1+up)]; > +up = slowdown/regression (fail),
    // < -down = improvement too large for one submission / lucky reading (fail).
    //
    // Prefill: +/-3% symmetric -- prefill is not a real optimization axis, so it is a
    // health gate (regression and lucky-fast both fail past 3%).
    //
    // Decode: +1% regression / -2.5% gain -- tight on regressions (decode is the primary
    // scored axis), and a single submission's decode gain is capped at 2.5%; larger wins
    // must be CHUNKED across submissions (bounds lucky-measurement inflation and forces
    // incremental, verifiable progress). Decode is the axis the score rewards, but the
    // per-submission step is capped, not the cumulative total across submissions.
    //
    // BAND DERIVATION (gemma4 calibration session 2026-08-25, calibration-20260825T102741Z;
    // flagged for reviewer attention because the methodology, not just the numbers, is new):
    // the band SHAPE is preserved from the qwen-era precedent (symmetric prefill health
    // gate; asymmetric decode with the tight side on regressions), and the magnitudes are
    // re-derived from this session's measured variability. The qwen precedent pinned its
    // tight sides at ~7.7x the session CV (prefill 5% over CV 0.65%; decode +2% over CV
    // 0.26%) and its decode gain cap at ~19x; i.e. bands sat ~8-25x over measured CV.
    // Rule applied here: tight-side band = ceil-to-half-percent of 10x the per-axis
    // session CV (10x sits at the conservative end of that precedent envelope):
    //   prefill: 10 x 0.2712% = 2.712% -> 3.0% both sides (symmetric).
    //            Bounds: < qwen's 5% (never loosened); >= 4x session spread
    //            (4 x 0.581% = 2.324%) so ordinary run-to-run scatter cannot trip it.
    //   decode up: 10 x 0.0937% = 0.937% -> 1.0%. Bounds: < qwen's +2%;
    //            >= 4 x 0.2084% = 0.834%.
    //   decode down: qwen's down:up asymmetry ratio (5/2 = 2.5) preserved:
    //            1.0% x 2.5 = 2.5%. The alternative candidate was keeping the 5%
    //            per-submission chunking cap unchanged (it is partly policy, not pure
    //            variability); the tighter candidate is pinned per operator instruction,
    //            with the alternative recorded in the calibration PR for review.
    public static let prefillBandUpTolerance = 0.03
    public static let prefillBandDownTolerance = 0.03
    public static let decodeBandUpTolerance = 0.01
    public static let decodeBandDownTolerance = 0.025
    // Gemma 4 26B A4B cached calibration as of 2026-08-25: the mean of the
    // prefill/decode seconds-per-token published by four consecutive cool-gated
    // (40C, read via macmon) runs of `./benchmark.sh --local-iterate` on the
    // stock committed tree, on box 3 -- the RANKED box for this track per
    // David's 2026-08-25 ruling -- with a fresh worker per phase and zero
    // warmup (one timed run per phase). Identity: engine
    // dec515a5748619954a1ea2d500f0c4a8e19c33fe, benchd
    // c2327d156cedd98593b552b3d5323416c172fcfc, benchctl built from that same
    // benchd commit, golden = the 1024_1024 public local-submit golden
    // (29576 bytes). Artifact sha256s:
    //   benchctl 395764ce38f1000372c6bfc45e1c4c95ea89a6b5ef6740a83d9191990d7f9d52
    //   worker   48692dfa1268d0601f883f606f3acc07446271bca1628a9413a1178cfec0886b
    //   golden   36290b93b1445f354b9b8e3d5ba592976830b40dd924324f822ec55a87140be4
    // Evidence directory: calibration-20260825T102741Z.
    // Prefill samples: 0.0003277473955078125, 0.00032697843359375,
    // 0.00032883076953125, 0.000326931234375 (CV 0.2712%, spread 0.581%).
    // Decode samples: 0.0123674462890625, 0.0123643369140625,
    // 0.012390104164062499, 0.0123770774765625 (CV 0.0937%, spread 0.2084%).
    //
    // These constants are NOT the ranked scoring denominator. The ranked
    // runner times the candidate and the pinned reference tree back to back in
    // the same session behind the same 40C thermal gate; the paired ratio
    // against that live same-session baseline is what the ranked pipeline
    // folds into the final score. These constants keep two roles: local-mode
    // score estimates (--local-iterate / --local-submit) and the gates-only
    // pass's placeholder timing fields, which the paired-timing overlay
    // replaces. See the paired-baseline section of
    // docs/benchmark-window-freeze.md.
    public static let officialBaselinePrefillSecondsPerToken = 0.0003276219582519531
    public static let officialBaselineDecodeSecondsPerToken = 0.012374741210937498
    public static let scorePrefillWeight = 0.25
    public static let scoreDecodeWeight = 0.75
    public static let scorePrefillSpeedupFloor = 0.95
    public static let scoreDecodeSpeedupFloor = 0.95
    // The Ternary Bonsai 2 27B pack is ~8.6 GB; 25 GiB keeps
    // ample headroom for shard alignment/padding without approving a second
    // full copy of the model.
    public static let defaultMaxTransformedWeightsBytes = 25 * 1024 * 1024 * 1024
    public static let defaultMaxSubmissionSourceBytes = 256 * 1024 * 1024
    // Diagnostic (non-ranking) real-valued score fields are published rounded to
    // this many significant figures. Submitted model code controls its own
    // latency/memory, so every full-precision analog field it can influence
    // (RAM, bandwidth, wall/preflight/correctness/TTFT seconds, hit rate) is a
    // covert channel for exfiltrating the hidden prompt/golden it sees. Coarsening
    // these -- which carry no ranking weight -- collapses each from ~30 bits to a
    // few. The ranking fields (decode/prefill seconds-per-token and speedups) are
    // left precise here on purpose; bounding that residual channel is a publishing/
    // rate-limit decision on the scoring backend, not a repo-side change.
    public static let publicDiagnosticSignificantFigures = 2

    // FREE-RUN / COHORT TOKEN CEILING. The largest `total_tokens` a configured
    // free-run or cohort request may ask a worker for, enforced by
    // `RuntimeWorkerRequestValidation` and `RuntimeWorkerCohortSupport` and used
    // as the per-stream `maxTokensPerStream` (+1 for the seed) the free-run
    // sessions open their engines with. 1,536 is three wraps of a 512-position
    // sliding-window cache, which is what keeps a wrap-seam tail boundary
    // reachable from a configured diagnostic run.
    //
    // Renamed from the old `experimentalDFlash`-prefixed spelling with the
    // DFlash excision: the VALUE is unchanged, and every consumer of it was,
    // and still is, a non-DFlash free-run/cohort path.
    public static let freeRunMaxConfiguredTotalTokens = 1_536
}
