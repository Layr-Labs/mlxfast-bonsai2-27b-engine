import Foundation
import Testing

// Structural checks on the Yukon track manifest (benchmark.json) for track
// bonsai2-27b-mlx-v1. This is NOT a re-implementation of Yukon's zod
// schema (src/benchmark/manifest.ts, RawBenchmarkConfigSchema /
// parseRepositoryBenchmarkManifest) -- that schema was validated directly by
// running `bun run` against the real yukon checkout
// ($HOME/projects/layr-labs/.worktrees/yukon-gemma-onboarding),
// importing parseRepositoryBenchmarkManifest and feeding it this file's own
// benchmark.json at authoring time, which returned PARSE OK with
// schemaVersion 1 and the expected editablePathsCount. That is a one-time,
// external verification method (recorded here so it can be re-run, not
// because this suite re-implements it) and is out of this repository's swift
// test loop because it needs the yukon checkout and its own bun/npm
// dependency tree, neither of which this repository depends on or vendors.
//
// What THIS suite pins, so a manifest edit cannot silently drift from the
// tree it describes or from tools/lint-benchmark-manifest.py's own rules
// without a swift test failure calling it out:
//   - benchmark.json parses as JSON and required top-level keys are present
//     with the right JSON type (a lightweight mirror of the zod schema's
//     shape requirements, not a full re-implementation).
//   - every editablePaths / optionalEditablePaths entry exists in this repo's
//     tree (repo-root relative, same as the Yukon importer resolves them).
//   - optionalEditablePaths is a subset of editablePaths.
//   - no two editablePaths entries are duplicates or nest inside each other.
//   - scorePath is not inside any editablePaths entry.
//   - runner.workflow names a real file under .github/workflows/.
//   - contractPath exists, parses as JSON, and its track_id equals trackId.
//   - editableSurfaceByteBudget's four numeric caps equal the compiled-in
//     EditableSurfaceByteBudget defaults (manifest/enforcer drift guard,
//     mirroring tools/lint-benchmark-manifest.py check 3b).
//
// Scope note: no MLX device work, no weights, no GPU, no network, and the pinned
// benchd is NOT required to be resolved (benchd-bin/ need not exist) -- every
// assertion below is static content / filesystem existence, exactly like
// HarnessHashRootSetTests's own scope note.

private func repoRootRelative(_ path: String) -> String { path }

private func loadJSONObject(atRepoPath path: String) throws -> [String: Any] {
    let data = try Data(contentsOf: URL(fileURLWithPath: repoRootRelative(path)))
    let object = try JSONSerialization.jsonObject(with: data)
    guard let dict = object as? [String: Any] else {
        Issue.record("\(path) did not decode to a JSON object")
        return [:]
    }
    return dict
}

@Suite("Gemma4 benchmark.json manifest")
struct Gemma4BenchmarkManifestTests {

    @Test
    func manifestParsesAndCarriesRequiredKeys() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")

        // Mirrors RawBenchmarkManifestV1Schema's required keys (yukon
        // src/benchmark/manifest.ts) plus this repository's own convention
        // of always carrying a "scoring" object (tools/lint-benchmark-
        // manifest.py SCHEMA, line ~113: "scoring": dict).
        let requiredStringKeys = [
            "name", "description", "category", "direction", "scorePath", "contractPath",
            "staticReviewTrackId", "trackId",
        ]
        for key in requiredStringKeys {
            #expect(manifest[key] is String, "benchmark.json[\(key)] must be a string")
        }

        #expect(manifest["schemaVersion"] as? Int == 1)
        #expect(manifest["direction"] as? String == "+")
        #expect(manifest["category"] as? String == "swift")
        #expect(manifest["trackId"] as? String == "bonsai2-27b-mlx-v1")

        #expect(manifest["editablePaths"] is [String])
        #expect((manifest["editablePaths"] as? [String])?.isEmpty == false)
        #expect(manifest["setupCommand"] is [String])
        #expect(manifest["benchmarkCommand"] is [String])
        #expect(manifest["scoring"] is [String: Any])

        let runner = try #require(manifest["runner"] as? [String: Any])
        #expect(runner["provider"] as? String == "github-actions")
        #expect(runner["workflow"] as? String == "benchmark.yml")
    }

    @Test
    func everyEditablePathExistsInTree() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])
        // 92 -> 89 on the Qwen 3.8 125B A6B port. The DFlash arm is removed
        // (RULED, David 2026-08-27 relayed by orchestrator: "Remove it,
        // MTP-only"), so its declaration file and its drafter MODEL file leave
        // with it, and the five Gemma tower files are replaced by the four
        // `Qwen4Exp*.swift` model files of this target. `mtp-head.manifest.json`
        // STAYS editable: it is how a participant declares a re-quantization of
        // the head, which on this track is the head EMBEDDED in the pinned
        // checkpoint. The head WEIGHTS directories are gone entirely -- this
        // track stages no head file at all.
        //
        // 70 -> 80 on the Nemotron 3.5 Lightning bring-up. `Vendor/mlx-swift-lm`
        // stopped being a submodule and became a VENDORED TREE, because an
        // editablePaths entry names bytes and a gitlink names a commit. The ten
        // `NemotronH*.swift` model files in it were the model surface then.
        //
        // 80 -> 94 on David's ruling of 2026-09-16 ("the mtp and everything
        // else should be editable like gemma"): the whole ContinuousBatchingV2
        // directory, twelve MLXLMCommon cache/attention/RoPE/config files and
        // MLXLMServer/Runtime/ToolStreamHandler.swift joined the surface. The
        // runner, its registry, bench-worker and the server stay trusted:
        // nothing names them.
        //
        // 94 -> 95 on the Ternary Bonsai 2 27B port. The ten `NemotronH*.swift`
        // files leave and eleven take their place: the packed checkpoint
        // contract and its text model, the nine `Qwen35*.swift` tower and MTP
        // files this pack actually runs through, and
        // `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift`, which holds the
        // signed-transform layers every packed projection is. That last one is
        // in the OTHER vendored tree, and it belongs on the surface for the
        // same reason the Metal kernels beside it do: it is arithmetic this
        // track runs on every projection of every layer.
        //
        // 95 -> 96 with the DFlash 2 block drafter. `DFlash2Draft.swift` holds
        // the drafter that proposes a whole block in one forward, and the
        // depth, the block geometry and the candidate path are competitive
        // surface for exactly the reason the MTP head's file already is. Its
        // CBv2 adapter (`Qwen35DFlash2Assistant.swift`) stays TRUSTED: it is
        // the seam that hands the engine committed target rows, not
        // arithmetic a participant tunes.
        #expect(editablePaths.count == 97, "editablePaths count drifted; update this pin deliberately if the surface changed")

        let fm = FileManager.default
        for path in editablePaths {
            #expect(fm.fileExists(atPath: path), "editablePaths entry does not exist in tree: \(path)")
        }
    }

    @Test
    func optionalEditablePathsIsSubsetOfEditablePaths() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = Set(try #require(manifest["editablePaths"] as? [String]))
        let optionalPaths = try #require(manifest["optionalEditablePaths"] as? [String])
        #expect(!optionalPaths.isEmpty)
        for path in optionalPaths {
            #expect(editablePaths.contains(path), "optionalEditablePaths entry \(path) is not also an editablePaths entry")
        }
    }

    @Test
    func editablePathsHaveNoDuplicatesOrNesting() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])

        #expect(Set(editablePaths).count == editablePaths.count, "editablePaths contains a duplicate entry")

        func isEqualOrDescendant(_ candidate: String, of other: String) -> Bool {
            candidate == other || candidate.hasPrefix(other + "/")
        }

        for i in editablePaths.indices {
            for j in editablePaths.indices where i != j {
                let a = editablePaths[i]
                let b = editablePaths[j]
                #expect(
                    !isEqualOrDescendant(a, of: b),
                    "editablePaths entries overlap/nest: \(a) is inside \(b)"
                )
            }
        }
    }

    @Test
    func scorePathIsNotInsideAnyEditablePath() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])
        let scorePath = try #require(manifest["scorePath"] as? String)

        for editable in editablePaths {
            let insideEditable = scorePath == editable || scorePath.hasPrefix(editable + "/")
            #expect(!insideEditable, "scorePath \(scorePath) is inside editablePaths entry \(editable)")
        }
    }

    @Test
    func runnerWorkflowFileExists() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let runner = try #require(manifest["runner"] as? [String: Any])
        let workflow = try #require(runner["workflow"] as? String)
        let path = ".github/workflows/\(workflow)"
        #expect(FileManager.default.fileExists(atPath: path), "runner.workflow does not resolve to a real file: \(path)")
    }

    @Test
    func contractPathExistsAndTrackIdMatches() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let contractPath = try #require(manifest["contractPath"] as? String)
        #expect(FileManager.default.fileExists(atPath: contractPath))

        let contract = try loadJSONObject(atRepoPath: contractPath)
        let contractTrackId = try #require(contract["track_id"] as? String)
        let manifestTrackId = try #require(manifest["trackId"] as? String)
        #expect(contractTrackId == manifestTrackId)
    }

    @Test
    func byteBudgetMatchesTrustedEnforcerDefaults() throws {
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let budget = try #require(manifest["editableSurfaceByteBudget"] as? [String: Any])

        // Mirrors Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift's
        // defaultMaxTotalBytes / defaultMaxFileBytes / defaultMaxGrowthBytes /
        // defaultExemptPathMaxBytes -- read directly off that file at authoring
        // time (values verified there, not re-derived here). tools/lint-
        // benchmark-manifest.py check 3b enforces the same equality; this test
        // gives the same guard a swift-test-visible failure.
        //
        // maxTotalBytes RE-DERIVED 2026-08-27 for this track's editable
        // surface (4,404,587 -> 4,198,878), by the same method the previous
        // value used: the ENFORCED at-rest surface plus a stated margin of
        // exactly 1 MiB (1,048,576 B = 4x maxGrowthBytes).
        //
        //     3,150,302 (at rest) + 1,048,576 (margin) = 4,198,878
        //
        // RE-DERIVED AGAIN 2026-08-28, to 4,199,757, by the same method:
        //
        //     3,151,181 (at rest) + 1,048,576 (margin) = 4,199,757
        //
        // AND AGAIN 2026-08-28, to 4,202,851, when the norm-convention fix
        // added the weight offset to the vendored norm and its ten call sites:
        //
        //     3,154,275 (at rest) + 1,048,576 (margin) = 4,202,851
        //
        // AND AGAIN 2026-08-28, to 4,204,273, when the quantized-gather
        // determinism fix landed in SwitchLayers.swift:
        //
        //     3,155,697 (at rest) + 1,048,576 (margin) = 4,204,273
        //
        // The QSA causality fix added its reasoning to an EDITABLE file
        // (Qwen4ExpText.swift), the margin fell to 1,047,697 -- 879 B under the
        // stated minimum -- and `enforcedSurfaceStaysUnderTotalCap` red. That
        // is exactly the condition the note below says makes a re-derivation
        // due, so the cap moved rather than the comment.
        //
        // NOTHING IS EXEMPT on this track, so the at-rest figure and the
        // enforced figure are the same number and there is no exempt-byte
        // double-count to get wrong. The old value would still have PASSED
        // the margin check, which is exactly why it is re-derived rather than
        // left: carrying it leaves 1,254,286 B of unexplained slack.
        //
        // THE CAP IS RE-DERIVED WHEN THE MARGIN FALLS BELOW 1 MiB, NOT ON
        // EVERY SURFACE CHANGE. Later edits shrank the surface slightly (the
        // review round of 2026-08-27 took 173 B out of Qwen4Exp.swift), which
        // WIDENS the margin. Chasing a shrink downward would churn all five
        // declaration sites for no gain and would tighten a cap participants
        // work against. `enforcedSurfaceStaysUnderTotalCap` below is the
        // instrument: it re-derives the margin from the tree on every run and
        // reds when the margin drops under 1 MiB, which is the condition that
        // makes a re-derivation due.
        // `enforcedSurfaceStaysUnderTotalCap` below re-derives the margin from
        // the tree rather than restating it.
        //
        // Raised again 2026-08-30 (4,204,273 -> 9,447,153, +5 MiB, David
        // ruling) after the enforced at-rest surface on main reached 3,156,707 B
        // and the margin fell to 1,047,566 B, under the 1 MiB minimum. Dev-repo
        // companion to public PR #53, which applied the same +5 MiB raise on the
        // same tripwire. This literal MIRRORS the enforcer default, so it moves
        // with it; the 1 MiB minimum-margin assertion below is unchanged.
        //
        // Raised again 2026-09-15 (2,706,783 -> 7,949,663, +5 MiB) on the same
        // tripwire: the at-rest surface reached 1,730,491 B and the margin fell
        // to 976,292 B. Same mirror, same unchanged minimum.
        #expect(budget["maxTotalBytes"] as? Int == 7_949_663)
        #expect(budget["maxFileBytes"] as? Int == 524_288)
        #expect(budget["maxGrowthBytes"] as? Int == 262_144)

        // THE TWO EXEMPT CAPS ARE KEPT, DELIBERATELY, WITH NOTHING EXEMPT.
        // `exemptPaths` is GONE (asserted below), so neither cap can bind
        // today. They stay DECLARED because both enforcers still carry them as
        // compiled-in fallbacks --
        // EditableSurfaceByteBudget.defaultExemptPathMaxBytes /
        // .defaultExemptPathMaxFileBytes, and benchd's
        // DEFAULT_EXEMPT_PATH_MAX_BYTES / DEFAULT_EXEMPT_PATH_MAX_FILE_BYTES --
        // and the manifest is what pins those constants to a reviewed number
        // (tools/lint-benchmark-manifest.py check 3b, BUDGET_CAP_KEYS).
        // Deleting the declaration would delete that drift check and leave the
        // constants as the cap with nothing checking them, which is the
        // single-source design inverted. Keeping an inert-but-checked cap costs
        // nothing and refuses nothing.
        #expect(budget["exemptPathMaxBytes"] as? Int == 512_000_000)
        #expect(budget["exemptPathMaxFileBytes"] as? Int == 100_000_000)

        // NOTHING IS EXEMPT (David requant-only ruling, 2026-08-26). An exempt
        // path only ever existed to let head WEIGHTS ride in a submission
        // outside the 4.4 MB source budget. A submission no longer carries head
        // weights at all, so an exemption would be an exemption from nothing --
        // and a stale `exemptPaths` naming a path that is no longer editable is
        // a hard failure in tools/lint-benchmark-manifest.py (exemptPaths must
        // be a subset of editablePaths), not a harmless leftover.
        #expect(budget["exemptPaths"] == nil, "editableSurfaceByteBudget.exemptPaths must be absent: nothing rides in a submission that is exempt from the code budget")
    }

    @Test
    func headWeightDirectoriesAreNotEditable() throws {
        // THE ENFORCEMENT HALF OF THE REQUANT-ONLY RULING (David, 2026-08-26).
        // The two speculative heads are the organizer's pinned weights. A
        // participant may declare a re-quantization of them; a participant may
        // not upload head weights. What makes that real is this list: with the
        // directories out of editablePaths,
        // .github/scripts/enforce-modifiable-surface.sh refuses any changed file
        // under them, and .github/scripts/overlay-editable-paths.sh never copies
        // one into the measured tree.
        //
        // The DECLARATION files stay editable. They are the recipe surface, and
        // they carry no weights.
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])
        let optionalPaths = try #require(manifest["optionalEditablePaths"] as? [String])
        let budget = try #require(manifest["editableSurfaceByteBudget"] as? [String: Any])
        let exemptPaths = (budget["exemptPaths"] as? [String]) ?? []

        let headWeightDirectories = ["mtp-head", "dflash-head"]
        for bucket in [
            ("editablePaths", editablePaths),
            ("optionalEditablePaths", optionalPaths),
            ("editableSurfaceByteBudget.exemptPaths", exemptPaths),
        ] {
            for entry in bucket.1 {
                for directory in headWeightDirectories {
                    #expect(
                        entry != directory && !entry.hasPrefix(directory + "/"),
                        "\(bucket.0) entry \(entry) reaches head weights directory \(directory); a submission must not be able to carry head weights"
                    )
                }
            }
        }

        // POSITIVE DISCRIMINATOR. The rule is "no head WEIGHTS", not "no head
        // declaration": a requant submission has to have something to edit.
        for declaration in ["mtp-head.manifest.json"] {
            #expect(editablePaths.contains(declaration), "\(declaration) must stay editable: it is how a re-quantization is declared")
            #expect(optionalPaths.contains(declaration), "\(declaration) must stay optional: an absent declaration selects the organizer-pinned head")
        }
    }

    @Test
    func enforcedSurfaceStaysUnderTotalCap() throws {
        // Re-derives the maxTotalBytes margin from the tree instead of trusting
        // the comment above it. With `exemptPaths` gone, every editable path is
        // now ENFORCED, so this is the whole walk -- and it must still clear the
        // declared cap with the stated 1 MiB margin. If the head directories had
        // been left editable-but-no-longer-exempt, this is the assertion that
        // would have caught it the moment a staged head landed in the tree.
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])
        let budget = try #require(manifest["editableSurfaceByteBudget"] as? [String: Any])
        let maxTotalBytes = try #require(budget["maxTotalBytes"] as? Int)

        let fm = FileManager.default
        var totalBytes = 0
        for path in editablePaths {
            var isDirectory = ObjCBool(false)
            guard fm.fileExists(atPath: path, isDirectory: &isDirectory) else { continue }
            if !isDirectory.boolValue {
                totalBytes += (try fm.attributesOfItem(atPath: path)[.size] as? Int) ?? 0
                continue
            }
            guard let enumerator = fm.enumerator(atPath: path) else { continue }
            // The entry NAME is not used -- `fileAttributes` describes whatever
            // the enumerator last returned -- so the call advances the walk and
            // its result is deliberately discarded rather than bound.
            while enumerator.nextObject() != nil {
                guard enumerator.fileAttributes?[.type] as? FileAttributeType == .typeRegular else {
                    continue
                }
                totalBytes += (enumerator.fileAttributes?[.size] as? NSNumber)?.intValue ?? 0
            }
        }

        #expect(totalBytes < maxTotalBytes, "enforced at-rest editable surface (\(totalBytes) B) is above maxTotalBytes (\(maxTotalBytes) B)")
        #expect(
            maxTotalBytes - totalBytes >= 1_048_576,
            "maxTotalBytes leaves \(maxTotalBytes - totalBytes) B of margin over the \(totalBytes) B at-rest surface, below the stated 1 MiB minimum"
        )
    }

    @Test
    func benchdGitlinkPathIsNeverEditable() throws {
        // Load-bearing exclusion (tools/lint-benchmark-manifest.py check 3):
        // no editablePaths entry may cover the pinned measurement harness at any
        // prefix depth -- an editable entry there would let a submission repoint
        // the thing that scores it at its own code.
        //
        // benchd used to be a SOURCE submodule (`benchd` + `.gitmodules`), then
        // a file pin (`benchd.pin`). It is now a PREBUILT binary resolved from
        // the bench repository's dist channel into `benchd-bin/` by
        // tools/fetch-benchd.sh, so the resolved-binary directory is the LIVE
        // entry. The submodule and pin spellings are kept so a reintroduced
        // gitlink or a planted pin file is covered on arrival rather than after
        // someone remembers to re-add the check.
        let manifest = try loadJSONObject(atRepoPath: "benchmark.json")
        let editablePaths = try #require(manifest["editablePaths"] as? [String])
        for path in editablePaths {
            #expect(path != "benchd" && !path.hasPrefix("benchd/"), "editablePaths entry reaches the benchd gitlink: \(path)")
            #expect(path != ".gitmodules", "editablePaths entry reaches .gitmodules: \(path)")
            #expect(path != "benchd.pin", "editablePaths entry reaches the benchd pin: \(path)")
            #expect(path != "benchd-bin" && !path.hasPrefix("benchd-bin/"), "editablePaths entry reaches the resolved benchd: \(path)")
        }
    }
}
