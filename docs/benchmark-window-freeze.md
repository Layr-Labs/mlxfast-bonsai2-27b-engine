# Benchmark window freeze — the surviving record

> **HISTORICAL IN PART.** This document froze the timed window for the retired
> serial track `laguna-xs-2.1-serial-v2`. That track is gone, and so are its
> files, tests, scripts and constants.
>
> What survives here is the part that still describes live code: the acceptance
> bands, the cached local-mode calibration, and the rule that no file stores a
> ranked baseline pair. `Sources/MLXFastCore/Constants.swift` and
> `Sources/MLXFastCore/Golden.swift` cite this document for those three things.
>
> The LIVE ranked track is `bonsai2-27b-mlx-v1`. Its window is a single
> stream over a 512-token seed and a 128-step decode window, scored as
> `composite = prefill_gain^0.25 * decode_gain^0.75`. For the live window and
> the live scoring, read `docs/participant-contract.md` section 5 and
> `benchmark.json`. This document does not define them.

## Acceptance bands

`AcceptanceBand` gates prefill and decode against a pinned calibration
reference `R`. Each axis is a single noisy measurement. After the speedup
floors, a measured value must land inside
`[R * (1 - downTolerance), R * (1 + upTolerance)]`. It fails above
`R * (1 + upTolerance)`, which is a regression, and it fails below
`R * (1 - downTolerance)`, which is either an improvement too large to trust in
one submission or a suspiciously lucky reading.

| Constant | Value |
|---|---|
| `prefillBandUpTolerance` | 0.03 |
| `prefillBandDownTolerance` | 0.03 |
| `decodeBandUpTolerance` | 0.01 |
| `decodeBandDownTolerance` | 0.025 |

- **Prefill: 3% symmetric.** Prefill is not a real optimization axis here, so it
  is a health gate. A regression and a lucky-fast reading past 3% both fail.
- **Decode: +1% regression, -2.5% gain.** Decode is the axis the score rewards,
  so the regression side is tight. The gain side caps a SINGLE submission's
  decode improvement at 2.5%. Larger wins are welcome, and they are chunked
  across submissions so each step stays inside the band and stays independently
  verifiable. The cap is per submission, not cumulative.

**Where the magnitudes come from.** The band SHAPE is preserved from the
qwen-era precedent: a symmetric prefill health gate, and an asymmetric decode
band with the tight side on regressions. The magnitudes were re-derived from
the gemma4 2026-08-25 box-3 calibration session. The rule applied was
tight-side band = ceil-to-half-percent of 10x the per-axis session coefficient
of variation, bounded never looser than the qwen absolutes and never tighter
than 4x the measured session spread. The full arithmetic is the BAND DERIVATION
comment block in `Sources/MLXFastCore/Constants.swift`, which this document
does not duplicate.

`R` and the per-axis tolerances are ranking-contract decisions. They are
operator-owned.

## Cached local-mode constants

The `officialBaseline*` constants in `Sources/MLXFastCore/Constants.swift` are a
cached calibration. They are NOT the ranked score denominator.

They keep two roles: local-mode estimates, and the gates-only pass's
placeholder timing fields.

- `officialBaselineDecodeSecondsPerToken = 0.012374741210937498`
- `officialBaselinePrefillSecondsPerToken = 0.0003276219582519531`

The values are the Gemma 4 26B A4B 2026-08-25 calibration: the mean
prefill and decode seconds per token from four consecutive cool-gated runs of
`./benchmark.sh --local-iterate` on the stock committed tree, on box 3, with a
fresh worker per phase and zero warmup. Decode carried a coefficient of
variation of 0.0937% and prefill 0.2712%. The full identity of that session --
the engine commit, the benchd commit, the golden and the per-sample values --
is in the constants' own comment.

If either number here disagrees with `Sources/MLXFastCore/Constants.swift`, one
of the two is stale. The code is the authority.

## The ranked pair is measured, never stored

The ranked score denominator is the live serial-control leg, measured on the
same box in the same job as the candidate leg. Nothing caches it.

**NO FILE STORES A BASELINE PAIR.** Not the constants above, not the contract
fixture, not a golden. `tools/lint-benchmark-manifest.py` check 5b keeps
`benchmark.baseline_prefill_seconds_per_token` and
`benchmark.baseline_decode_seconds_per_token` out of this repository, and the
ranked path refuses a golden that carries either.

Each ranked box additionally carries its own calibration file. That file is a
HEALTH BAND for the control leg and never a denominator: a stale one can stop a
run, and it can never move a score. `docs/participant-contract.md` section
5.1.0.1 is the authority on it.

## Per-prompt baselines in the golden oracle

`BenchmarkGolden` in `Sources/MLXFastCore/Golden.swift` can decode two optional
fields:

```json
"benchmark": {
  "baseline_prefill_seconds_per_token": 0.0101,
  "baseline_decode_seconds_per_token": 0.1317
}
```

The schema rules, enforced at golden load:

- Both fields must be present together or absent together, finite and positive.
  A half-calibrated oracle would silently mix two calibration regimes.
- When present, the scored speedups, the floors and the published `baseline_*`
  metrics resolve from the golden. When absent, they resolve from the constants
  above. All public fixtures carry neither field.

The fields exist because a pre-pool design scored each rotated prompt against
its own measured calibration. On the ranked path they are now REFUSED, as the
section above states: the baseline is measured live. The schema keeps the
fields so that a golden carrying them is rejected by name rather than
misread.
