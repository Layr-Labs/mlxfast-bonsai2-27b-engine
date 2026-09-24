#!/usr/bin/env python3
"""Write the arming patch for fixtures/bonsai2_27b_mlx_v1_track.json.

The organizer records the track's goldens on the track's box. This tool reads
a directory of those recorded files and writes the fixture values that arm the
track. It computes every {sha256, bytes} pin itself. It reads the files; it
never prints their contents.

The directory holds these files, and nothing else:

  <name>.golden.json              the 8 timed-pool tapes, recorded serial.
                                  --live names one of them.
  <live>.mtpN.golden.json         one per-depth oracle for every depth in
                                  mtp_head.permitted_draft_depths.
  <live>.dflashN.golden.json      one per-depth oracle for every depth in
                                  dflash_drafter.permitted_draft_depths.
  <basename of each public_captures r2_path>
                                  the public captures for the local modes.

THE PER-DEPTH SHAPE IS DECIDED BY THE BYTES. A per-depth oracle can differ from
the serial tape (a speculative round verifies at M > 1, and MLX dispatches a
different kernel there), and it can equal it. Every live_golden_speculative key
gets its own pin, but keys whose bytes are identical share ONE R2 object: a key
whose oracle equals the serial live golden points at the live golden's pool
key, and a key whose oracle equals an earlier key's points at that key's
object. The organizer uploads each distinct object once. The upload list goes
to stderr.

Usage:
  python3 tools/golden-arming-patch.py --dir DIR --live NAME
  python3 tools/golden-arming-patch.py --dir DIR --live NAME --apply

Without --apply the tool prints the patch (JSON, values only) on stdout. With
--apply it writes the values into the fixture in place.
Exit: 0 on success, 1 on any refusal (printed with a REFUSE prefix).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = os.path.join(ROOT, "fixtures", "bonsai2_27b_mlx_v1_track.json")
POOL_SIZE = 8
FORBIDDEN_BENCHMARK_KEYS = (
    "baseline_prefill_seconds_per_token",
    "baseline_decode_seconds_per_token",
)


class Refusal(Exception):
    pass


def pin(path: str) -> dict:
    """The {sha256, bytes} pin of one recorded file, after a shape check."""
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Refusal(f"{os.path.basename(path)} is not JSON: {exc.__class__.__name__}")
    benchmark = document.get("benchmark") if isinstance(document, dict) else None
    if isinstance(benchmark, dict):
        stored = [key for key in FORBIDDEN_BENCHMARK_KEYS if key in benchmark]
        if stored:
            raise Refusal(
                f"{os.path.basename(path)} stores a baseline pair ({', '.join(stored)}); "
                "the pair is measured live on the box and never stored"
            )
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def build_patch(contract: dict, directory: str, live: str) -> tuple[dict, list[str]]:
    track = contract["track_id"]
    prefix = f"correctness_prompts/{track}/"
    names = sorted(n for n in os.listdir(directory) if n.endswith(".golden.json"))
    others = sorted(n for n in os.listdir(directory) if not n.endswith(".golden.json"))
    if others:
        raise Refusal(f"the directory holds files that are not *.golden.json: {', '.join(others)}")

    public_captures = contract.get("public_captures") or {}
    public_names = {
        os.path.basename(entry["r2_path"]): role for role, entry in public_captures.items()
    }
    depth_keys = [f"mtp{d}" for d in contract["mtp_head"]["permitted_draft_depths"]] + [
        f"dflash{d}" for d in contract["dflash_drafter"]["permitted_draft_depths"]
    ]
    depth_re = re.compile(r"^(?P<name>.+)\.(?P<key>(mtp|dflash)[0-9]+)\.golden\.json$")

    pool_names, depth_files, public_files = [], {}, {}
    for name in names:
        match = depth_re.match(name)
        if name in public_names:
            public_files[public_names[name]] = name
        elif match:
            if match["name"] != live:
                raise Refusal(f"{name} is a per-depth oracle for {match['name']}, not for the live golden {live}")
            if match["key"] not in depth_keys:
                raise Refusal(f"{name} is for {match['key']}, which no decoder declares")
            depth_files[match["key"]] = name
        else:
            pool_names.append(name[: -len(".golden.json")])

    if len(pool_names) != POOL_SIZE:
        raise Refusal(f"found {len(pool_names)} timed-pool tapes, expected {POOL_SIZE}: {', '.join(pool_names)}")
    if live not in pool_names:
        raise Refusal(f"--live {live} is not one of the timed-pool tapes")
    missing = [key for key in depth_keys if key not in depth_files]
    if missing:
        raise Refusal(f"no per-depth oracle for: {', '.join(missing)} (the measure script refuses a declared depth without one)")
    missing = [role for role in public_captures if role not in public_files]
    if missing:
        raise Refusal(f"no public capture for: {', '.join(missing)}")

    pool = []
    live_entry = None
    for name in pool_names:
        entry = {"r2_path": f"{prefix}{name}.golden.json", **pin(os.path.join(directory, f"{name}.golden.json"))}
        pool.append(entry)
        if name == live:
            live_entry = entry
    assert live_entry is not None

    # Content addressing: identical bytes share one R2 object.
    object_by_sha = {live_entry["sha256"]: live_entry["r2_path"]}
    speculative = {}
    for key in depth_keys:
        own = pin(os.path.join(directory, depth_files[key]))
        r2_path = object_by_sha.setdefault(own["sha256"], f"{prefix}{depth_files[key]}")
        speculative[key] = {"r2_path": r2_path, **own}

    hidden_shas = {entry["sha256"] for entry in pool} | {entry["sha256"] for entry in speculative.values()}
    public = {}
    for role, entry in public_captures.items():
        own = pin(os.path.join(directory, public_files[role]))
        if own["sha256"] in hidden_shas:
            raise Refusal(f"the public capture {role} has the same bytes as a hidden tape or oracle; publishing it would publish that tape")
        public[role] = {"r2_path": entry["r2_path"], **own}

    patch = {
        "official_scoring_enabled": True,
        "live_golden": live,
        "timed_prompt_pool": pool,
        "hidden_correctness_golden": {"sha256": live_entry["sha256"], "bytes": live_entry["bytes"]},
        "live_golden_speculative": speculative,
        "public_captures": public,
    }

    uploads = []
    local_by_r2 = {entry["r2_path"]: f"{os.path.basename(entry['r2_path'])}" for entry in pool}
    for key in depth_keys:
        local_by_r2.setdefault(speculative[key]["r2_path"], depth_files[key])
    for role in public:
        local_by_r2[public[role]["r2_path"]] = public_files[role]
    pins_by_r2 = {e["r2_path"]: e for e in pool + list(speculative.values()) + list(public.values())}
    for r2_path in sorted(local_by_r2):
        entry = pins_by_r2[r2_path]
        uploads.append(f"{local_by_r2[r2_path]} -> {r2_path} ({entry['sha256'][:12]}..., {entry['bytes']} bytes)")
    shared = sum(1 for key in depth_keys if speculative[key]["r2_path"] != f"{prefix}{depth_files[key]}")
    uploads.append(f"{len(local_by_r2)} distinct object(s); {shared} of {len(depth_keys)} per-depth key(s) share an earlier object")
    return patch, uploads


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="the directory of recorded golden files")
    ap.add_argument("--live", required=True, help="the pool tape the track scores over (its name without .golden.json)")
    ap.add_argument("--apply", action="store_true", help="write the values into the fixture in place")
    ap.add_argument("--contract", default=CONTRACT, help=argparse.SUPPRESS)
    args = ap.parse_args()

    with open(args.contract, encoding="utf-8") as fh:
        contract = json.load(fh)
    try:
        patch, uploads = build_patch(contract, args.dir, args.live)
    except Refusal as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1

    for line in uploads:
        print(f"upload: {line}", file=sys.stderr)
    if args.apply:
        contract.update(patch)
        with open(args.contract, "w", encoding="utf-8") as fh:
            json.dump(contract, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"applied: {os.path.relpath(args.contract, ROOT)}", file=sys.stderr)
    else:
        json.dump(patch, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
