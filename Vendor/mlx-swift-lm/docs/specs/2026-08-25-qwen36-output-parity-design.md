# Qwen3.6 A3B output-parity mode design

Date: 2026-08-25

## Scope

The benchmark exposes `--output-parity fast|byte-exact` only for the
one-prompt `decode` and `full` campaign routes with MTP depth greater than zero.
`fast` is the default.

`fast` installs `.rectangular` verification and records
`rectangular-target-authoritative`. `byte-exact` installs
`.rectangularExact`, exact M1 target projection arithmetic, and records
`rectangular-timewise-byte-exact`. Exact means byte-identical generated token
IDs to the optimized autoregressive target route. It does not claim identity
to a separately constructed stock model.

## Construction contract

The configuration is inspected before model construction. The resulting
artifact contract proves model topology, layer ownership, target affine
W4/g64 packing, W8/g64 router/shared-gate packing, and MXFP8/g32 MTP packing.
Every per-layer override reached by an exact projection is validated.

The contract, profile, and target arithmetic form one immutable task-scoped
installation. A model load cannot publish a profile or exact arithmetic
without the inspected contract, and concurrent loads cannot mix settings.
After weights load, exact mode verifies the concrete projection types, affine
bias buffers, BF16 scale/bias dtype, output tiling, and expected module count.
Failure occurs before the vocabulary probe, warmup, or measured generation.

The assistant refuses `.rectangularExact` unless its target captured exact
arithmetic. `DARKBLOOM_QWEN_MTP_SERIAL` remains an unconditional diagnostic
oracle override. A requested fixed depth greater than the installed drafter
maximum is rejected before warmup.

## Data flow

1. Parse the typed output-parity enum and reject campaign-only flags outside a
   prompt-file campaign.
2. Inspect `config.json` and create a validated construction installation.
3. Load the model inside that task-scoped installation.
4. For exact mode, validate all loaded modules and buffers used by unchecked
   kernels.
5. Load the assistant and verify target arithmetic, verifier mode, and fixed
   depth.
6. Execute the installed route directly. Only logical M and execution phase
   remain runtime routing inputs.
7. Record typed parity metadata and the exact comma-delimited installed
   verifier field in the receipt.

Chat-template failure is fatal for a campaign prompt; the campaign never falls
back to raw encoding and silently changes the prompt contract.

## Receipt contract

Decode-capable receipts record:

- `outputParity`: `fast` or `byte-exact`;
- `verificationRoute`: `rectangular-target-authoritative` or
  `rectangular-timewise-byte-exact`;
- an `mtp` route whose `verify` field must exactly equal the installed typed
  verifier, rather than merely sharing a string prefix;
- the requested depth only after it has been checked against the installed
  drafter maximum.

## Verification

Focused tests cover parser rejection, exact route-field matching, fixed-depth
drift, environment-override precedence, target-arithmetic binding, every
router entry, malformed exact-projection overrides, unquantized loaded
projections, and the no-hot-path-validation invariant.

The authentic gate is a locked, rebased-source comparison:

- optimized AR versus rectangular-exact for all 1,024 generated tokens;
- matched optimized AR, fast K2, and exact K2 timing;
- separate prefill, decode, wall-time, and peak-memory reporting;
- source and dependency revisions in every receipt.

## Non-goals

- Making exact mode as fast as fast mode.
- Claiming that fast mode has the same generated stream as optimized AR.
- Applying parity selection to ordinary synthetic matrices, stock-only runs,
  or prefill-only runs.
- Applying target affine kernels to the MXFP8 assistant.
