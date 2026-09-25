import Foundation
import CryptoKit
import Testing
@testable import MLXFastCore

/// THE NEGATIVE CONTROL, kept from the era when the public goldens WERE
/// Gemma files and this was the only thing standing between a Gemma golden and
/// a Qwen run.
///
/// A golden whose `model_provenance` names the Gemma checkpoint must refuse,
/// and must refuse on the PROVENANCE -- not merely on the model type -- so
/// that the model-agnostic loader refuses it too. Losing this would let a
/// checkpoint swap go unnoticed the next time these files are regenerated.
@Test
func aGemmaGoldenIsRefusedAgainstTheQwenTarget() throws {
    let gemmaDocument = Data(
        """
        {
          "version": 1,
          "model_type": "gemma4_text",
          "model_provenance": {
            "repository": "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
            "revision": "0e3cbab38ce568cf6e23543010d08d03b731910c"
          },
          "cases": [{
            "name": "longcopy-gate-english-1024",
            "prompt_tokens": \(correctnessPromptJSON()),
            "expected_tokens": \(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
          }]
        }
        """.utf8
    )
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let gemmaPath = directory.appendingPathComponent("gemma-golden.json")
    try gemmaDocument.write(to: gemmaPath)

    // BOTH loaders refuse, and they refuse for DIFFERENT reasons, which is the
    // point of checking both.
    func refusal(_ load: (String) throws -> GoldenFixture) -> String? {
        do {
            _ = try load(gemmaPath.path)
            return nil
        } catch {
            return "\(error)"
        }
    }

    // The MODEL-AGNOSTIC loader is the one that matters here: it does not look
    // at `model_type` at all, so its refusal can only come from the provenance
    // block -- which means no `model_type` edit talks a Gemma golden past it.
    let agnostic = try #require(
        refusal { try loadGoldenFixture(from: $0) },
        "the model-agnostic loader accepted a Gemma golden")
    #expect(
        agnostic.contains("model_provenance does not match"),
        "expected a provenance refusal, got: \(agnostic)")

    // The QWEN loader refuses earlier and more cheaply, on the model identity.
    // Asserting the provenance wording here would be asserting the ORDER of
    // two checks rather than the refusal, so this asserts what it actually
    // guarantees: it does not accept the file.
    let strict = try #require(
        refusal { try loadQwenGoldenFixture(from: $0) },
        "the Qwen loader accepted a Gemma golden")
    #expect(strict.contains("gemma4_text") || strict.contains("model_provenance does not match"))
}

/// A golden that carries NO `model_provenance` must be REFUSED on the
/// public-golden loader path, and the refusal must NAME the missing field.
///
/// Absence is not a pass. A golden with no provenance block names no
/// checkpoint, so nothing binds its expected tokens to the pinned reference
/// model, and a loader that skips the pin check when the block is missing
/// accepts a capture from any model.
///
/// The requirement is opt-in and lives in ONE place, the same way
/// `requiredModelType` does: `loadQwenGoldenFixture` demands it, the
/// model-agnostic loader keeps the REFERENCE schema, where the block does not
/// exist at all and a corpus golden carries only `model_type`. Every golden
/// consumer in `Sources/` goes through the wrapper, so nothing loads a golden
/// without the check.
///
/// The pair under test is one document and the same document with that one
/// block removed, so the only difference between the file that loads and the
/// file that refuses is the field under test. It is built here rather than
/// read from `correctness_prompts/`, so the control that must LOAD is a
/// document this test owns, not a shipped capture.
@Test
func aGoldenWithoutModelProvenanceIsRefusedByName() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    var root: [String: Any] = [
        "version": 1,
        "model_type": MLXFastConstants.requiredGoldenModelType,
        "model_provenance": [
            "repository": MLXFastConstants.referenceModelRepository,
            "revision": MLXFastConstants.referenceModelRevision,
        ],
        "cases": [
            [
                "name": "provenance-control",
                "prompt_tokens": correctnessPrompt(),
                "expected_tokens": Array(repeating: 7, count: 256),
            ]
        ],
    ]
    let checkedInPath = directory.appendingPathComponent("with-provenance-golden.json")
    try JSONSerialization.data(withJSONObject: root).write(to: checkedInPath)

    #expect(
        root.removeValue(forKey: "model_provenance") != nil,
        "the control golden carries no model_provenance to remove")
    let stripped = try JSONSerialization.data(withJSONObject: root)
    let strippedPath = directory.appendingPathComponent("no-provenance-golden.json")
    try stripped.write(to: strippedPath)

    // The source file LOADS, so a refusal below is the removed block and
    // nothing else.
    _ = try loadQwenGoldenFixture(from: checkedInPath.path, requiredSteps: 256)

    var refusal: String?
    do {
        _ = try loadQwenGoldenFixture(from: strippedPath.path, requiredSteps: 256)
    } catch {
        refusal = "\(error)"
    }
    let message = try #require(
        refusal,
        "the public-golden loader accepted a golden with no model_provenance")
    #expect(
        message.contains("model_provenance"),
        "expected a refusal naming model_provenance, got: \(message)")

    // THE PLACEMENT CONTROL. The model-agnostic loader is the reference
    // schema and still accepts the file, which is what proves the requirement
    // is held in exactly one place rather than copied into both entry points.
    _ = try loadGoldenFixture(from: strippedPath.path, requiredSteps: 256)
}

@Test
func goldenModelProvenanceIsStrictAndPinned() throws {
    func documentJSON(repository: String, revision: String, extra: String = "") -> Data {
        Data(
            """
            {
              "version": 1,
              "model_provenance": {
                "repository": "\(repository)",
                "revision": "\(revision)"\(extra)
              },
              "cases": [{
                "name": "provenance-contract",
                "prompt_tokens": \(correctnessPromptJSON()),
                "expected_tokens": \(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
              }]
            }
            """.utf8
        )
    }

    func load(_ data: Data) throws -> GoldenFixture {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("golden.json")
        try data.write(to: path)
        return try loadGoldenFixture(from: path.path)
    }

    let valid = try load(
        documentJSON(
            repository: MLXFastConstants.referenceModelRepository,
            revision: MLXFastConstants.referenceModelRevision
        )
    )
    #expect(valid.cases.count == 1)

    #expect(throws: MLXFastError.self) {
        _ = try load(
            documentJSON(
                repository: "mlx-community/Laguna-XS-2.1-4bit",
                revision: "c42e0a8f8d504ceacde015a535dcb286d65c8799"
            )
        )
    }
    #expect(throws: MLXFastError.self) {
        _ = try load(
            documentJSON(
                repository: MLXFastConstants.referenceModelRepository,
                revision: MLXFastConstants.referenceModelRevision,
                extra: ", \"unexpected\": true"
            )
        )
    }
}

@Test
func loadGoldenFixtureAcceptsLayeredCorrectnessGates() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "anchors": [
          {
            "name": "anchor-0",
            "context_tokens": [1, 2, 3, 4],
            "expected_token": 5,
            "accepted_tokens": [6],
            "max_expected_rank": 2,
            "max_top_logit_delta": 0.001
          }
        ],
        "free_run": [
          {
            "name": "free-run-0",
            "prompt_tokens": \(correctnessPromptJSON(2)),
            "expected_tokens": [8, 9, 10],
            "exact_prefix_tokens": 2
          }
        ],
        "behavior": [
          {
            "name": "behavior-0",
            "prompt_tokens": [3, 4, 5],
            "accepted_token_sequences": [[11, 12], [12, 13]],
            "max_new_tokens": 2
          }
        ]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    let fixture = try loadGoldenFixture(from: path.path)

    #expect(fixture.cases.count == 1)
    #expect(fixture.correctnessGates?.anchorCases.count == 1)
    #expect(fixture.correctnessGates?.freeRunCases.count == 1)
    #expect(fixture.correctnessGates?.behaviorCases.count == 1)
    #expect(fixture.totalCorrectnessCaseCount == 4)
}

@Test
func loadGoldenFixtureRejectsMalformedLayeredCorrectnessGate() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "behavior": [
          {
            "name": "behavior-0",
            "prompt_tokens": \(correctnessPromptJSON(3)),
            "accepted_token_sequences": [[11, 12]],
            "max_new_tokens": 1
          }
        ]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenFixtureRejectsDuplicateLayeredCorrectnessNames() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "duplicate",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "anchors": [
          {
            "name": "duplicate",
            "context_tokens": [1, 2, 3],
            "expected_token": 4
          }
        ]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenFixtureRejectsNoopAnchorDelta() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "anchors": [
          {
            "name": "anchor-0",
            "context_tokens": [1, 2, 3],
            "expected_token": 4,
            "max_top_logit_delta": 0.001
          }
        ]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenFixtureRejectsUnknownCorrectnessGateKey() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "free_runs": [
          {
            "name": "typo",
            "prompt_tokens": \(correctnessPromptJSON(2)),
            "expected_tokens": [8]
          }
        ]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenFixtureRejectsEmptyCorrectnessGateSection() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "correctness_gates": {
        "anchors": []
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func goldenSequenceMatcherChecksExactPrefixes() {
    let pass = GoldenSequenceMatcher.firstPrefixMismatch(
        expected: [1, 2, 3],
        actual: [1, 2, 9],
        prefixTokens: 2
    )
    #expect(pass.passed)

    let fail = GoldenSequenceMatcher.firstPrefixMismatch(
        expected: [1, 2, 3],
        actual: [1, 8, 3],
        prefixTokens: 3
    )
    #expect(!fail.passed)
    #expect(fail.step == 1)
    #expect(fail.expectedToken == 2)
    #expect(fail.actualToken == 8)
}

@Test
func goldenSequenceMatcherAcceptsShortBehaviorPrefixes() {
    let pass = GoldenSequenceMatcher.matchesAnyAcceptedPrefix(
        acceptedSequences: [[101], [202, 203]],
        actual: [101, 999]
    )
    #expect(pass.passed)

    let fail = GoldenSequenceMatcher.matchesAnyAcceptedPrefix(
        acceptedSequences: [[101], [202, 203]],
        actual: [202, 999]
    )
    #expect(!fail.passed)
    #expect(fail.step == 1)
    #expect(fail.expectedToken == 203)
    #expect(fail.actualToken == 999)
}

@Test
func goldenSequenceMatcherAcceptsExactAnswerSequences() {
    let pass = GoldenSequenceMatcher.matchesAnyExactSequence(
        acceptedSequences: [[10, 11], [20, 21]],
        actual: [20, 21]
    )
    #expect(pass.passed)

    let fail = GoldenSequenceMatcher.matchesAnyExactSequence(
        acceptedSequences: [[10, 11], [20, 21]],
        actual: [20, 22, 99]
    )
    #expect(!fail.passed)
    #expect(fail.step == 1 || fail.step == 2)
}

@Test
func loadGoldenCasesAcceptsValidFixture() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    let cases = try loadGoldenCases(from: path.path)

    #expect(cases.count == 1)
    #expect(cases[0].name == "hidden-0")
    #expect(cases[0].promptTokens == correctnessPrompt())
    #expect(cases[0].expectedTokens.count == MLXFastConstants.correctnessSteps)

    let fixture = try loadGoldenFixture(from: path.path)
    let digest = SHA256.hash(data: try Data(contentsOf: path))
    let expectedHash = digest.map { String(format: "%02x", $0) }.joined()
    #expect(fixture.cases == cases)
    #expect(fixture.sha256 == expectedHash)
}

@Test
func loadGoldenFixtureAcceptsBenchmarkOracle() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let prefill = Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens)
    let seed = Array(repeating: 2, count: MLXFastConstants.benchmarkDecodeSeedTokens)
    let decode = Array(repeating: 3, count: MLXFastConstants.benchmarkDecodeSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(prefill),
        "expected_prefill_token": 4,
        "decode_seed_tokens": \(seed),
        "expected_decode_seed_token": 5,
        "expected_decode_tokens": \(decode)
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    let fixture = try loadGoldenFixture(from: path.path)

    #expect(fixture.benchmark?.prefillPromptTokens == prefill)
    #expect(fixture.benchmark?.expectedPrefillToken == 4)
    #expect(fixture.benchmark?.decodeSeedTokens == seed)
    #expect(fixture.benchmark?.expectedDecodeSeedToken == 5)
    #expect(fixture.benchmark?.expectedDecodeTokens == decode)
    // No per-prompt baselines carried: scoring resolves to the calibrated constants.
    #expect(fixture.benchmark?.baselinePrefillSecondsPerToken == nil)
    #expect(fixture.benchmark?.baselineDecodeSecondsPerToken == nil)
    #expect(
        fixture.benchmark?.resolvedBaselinePrefillSecondsPerToken
            == MLXFastConstants.officialBaselinePrefillSecondsPerToken
    )
    #expect(
        fixture.benchmark?.resolvedBaselineDecodeSecondsPerToken
            == MLXFastConstants.officialBaselineDecodeSecondsPerToken
    )
}

private func benchmarkOracleGoldenJSON(baselineFieldsJSON: String) -> String {
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let prefill = Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens)
    let seed = Array(repeating: 2, count: MLXFastConstants.benchmarkDecodeSeedTokens)
    let decode = Array(repeating: 3, count: MLXFastConstants.benchmarkDecodeSteps)
    return """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(prefill),
        "expected_prefill_token": 4,
        "decode_seed_tokens": \(seed),
        "expected_decode_seed_token": 5,
        "expected_decode_tokens": \(decode)\(baselineFieldsJSON)
      }
    }
    """
}

@Test
func loadGoldenFixtureAcceptsPerPromptBenchmarkBaselines() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let json = benchmarkOracleGoldenJSON(baselineFieldsJSON: """
    ,
        "baseline_prefill_seconds_per_token": 0.25,
        "baseline_decode_seconds_per_token": 4.5
    """)
    try json.write(to: path, atomically: true, encoding: .utf8)

    let fixture = try loadGoldenFixture(from: path.path)

    #expect(fixture.benchmark?.baselinePrefillSecondsPerToken == 0.25)
    #expect(fixture.benchmark?.baselineDecodeSecondsPerToken == 4.5)
    // Carried baselines win over the calibrated constants.
    #expect(fixture.benchmark?.resolvedBaselinePrefillSecondsPerToken == 0.25)
    #expect(fixture.benchmark?.resolvedBaselineDecodeSecondsPerToken == 4.5)
}

@Test
func loadGoldenFixtureRejectsUnknownBenchmarkKeys() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    // A typo'd scoring-critical key must fail loudly. Without strict nested
    // key validation, JSONDecoder drops the unknown key, both baselines decode
    // as nil, and the run silently scores against the calibrated constants
    // instead of the intended per-prompt baseline.
    let json = benchmarkOracleGoldenJSON(baselineFieldsJSON: """
    ,
        "baseline_decode_second_per_token": 4.5
    """)
    try json.write(to: path, atomically: true, encoding: .utf8)

    do {
        _ = try loadGoldenFixture(from: path.path)
        Issue.record("expected unknown benchmark key to be rejected")
    } catch let MLXFastError.invalidInput(message) {
        #expect(message.contains("unknown key"))
        #expect(message.contains("baseline_decode_second_per_token"))
    } catch {
        Issue.record("expected MLXFastError.invalidInput, got \(error)")
    }
}

@Test
func loadGoldenFixtureRejectsNullBenchmarkObject() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": null
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    do {
        _ = try loadGoldenFixture(from: path.path)
        Issue.record("expected null benchmark object to be rejected")
    } catch let MLXFastError.invalidInput(message) {
        #expect(message.contains("benchmark must not be null"))
    } catch {
        Issue.record("expected MLXFastError.invalidInput, got \(error)")
    }
}

@Test
func loadGoldenFixtureRejectsHalfCalibratedBenchmarkBaselines() throws {
    let directory = try temporaryDirectory()
    for lonelyField in [
        "\"baseline_prefill_seconds_per_token\": 0.25",
        "\"baseline_decode_seconds_per_token\": 4.5",
    ] {
        let path = directory.appendingPathComponent("golden-\(UUID().uuidString).json")
        let json = benchmarkOracleGoldenJSON(baselineFieldsJSON: ",\n    \(lonelyField)")
        try json.write(to: path, atomically: true, encoding: .utf8)

        do {
            _ = try loadGoldenFixture(from: path.path)
            Issue.record("expected half-calibrated baseline pair to be rejected")
        } catch let MLXFastError.invalidInput(message) {
            #expect(message.contains("must be provided together"))
        } catch {
            Issue.record("expected MLXFastError.invalidInput, got \(error)")
        }
    }
}

@Test
func loadGoldenFixtureRejectsNonPositiveBenchmarkBaselines() throws {
    let directory = try temporaryDirectory()
    for badPair in [
        "\"baseline_prefill_seconds_per_token\": 0, \"baseline_decode_seconds_per_token\": 4.5",
        "\"baseline_prefill_seconds_per_token\": 0.25, \"baseline_decode_seconds_per_token\": -1.0",
    ] {
        let path = directory.appendingPathComponent("golden-\(UUID().uuidString).json")
        let json = benchmarkOracleGoldenJSON(baselineFieldsJSON: ",\n    \(badPair)")
        try json.write(to: path, atomically: true, encoding: .utf8)

        do {
            _ = try loadGoldenFixture(from: path.path)
            Issue.record("expected non-positive baseline to be rejected")
        } catch let MLXFastError.invalidInput(message) {
            #expect(message.contains("must be finite and positive"))
        } catch {
            Issue.record("expected MLXFastError.invalidInput, got \(error)")
        }
    }
}

@Test
func loadGoldenFixtureRejectsMalformedBenchmarkOracle() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "hidden-0",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": [1],
        "expected_prefill_token": 4,
        "decode_seed_tokens": [2],
        "expected_decode_seed_token": 5,
        "expected_decode_tokens": [3]
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenFixture(from: path.path)
    }
}

@Test
func loadGoldenFixtureStaleBenchmarkOracleErrorMentionsPrecomputedFixture() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = arrayJSON(Array(repeating: 9, count: MLXFastConstants.correctnessSteps))
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "case-a",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens))),
        "expected_prefill_token": 2,
        "decode_seed_tokens": \(arrayJSON(Array(repeating: 3, count: 32))),
        "expected_decode_seed_token": 4,
        "expected_decode_tokens": \(arrayJSON(Array(repeating: 5, count: MLXFastConstants.benchmarkDecodeSteps)))
      }
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    do {
        _ = try loadGoldenFixture(from: path.path)
        Issue.record("expected stale benchmark oracle error")
    } catch let MLXFastError.invalidInput(message) {
        #expect(message.contains("Replace stale local goldens with an updated precomputed golden fixture"))
    } catch {
        Issue.record("expected MLXFastError.invalidInput, got \(error)")
    }
}

@Test
func benchmarkOutputValidatorReportsTokenMismatches() {
    let oracle = BenchmarkGolden(
        prefillPromptTokens: Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens),
        expectedPrefillToken: 10,
        decodeSeedTokens: Array(repeating: 2, count: MLXFastConstants.benchmarkDecodeSeedTokens),
        expectedDecodeSeedToken: 20,
        expectedDecodeTokens: [30, 31, 32]
    )

    let prefill = BenchmarkOutputValidator.comparePrefillToken(
        expected: oracle,
        actualToken: 11
    )
    #expect(!prefill.passed)
    #expect(prefill.expectedToken == 10)
    #expect(prefill.actualToken == 11)

    let seed = BenchmarkOutputValidator.compareDecodeSeedToken(
        expected: oracle,
        actualToken: 21
    )
    #expect(!seed.passed)
    #expect(seed.expectedToken == 20)
    #expect(seed.actualToken == 21)

    let decode = BenchmarkOutputValidator.compareDecodeTokens(
        expected: oracle,
        actualTokens: [30, 99, 32]
    )
    #expect(!decode.passed)
    #expect(decode.step == 1)
    #expect(decode.expectedToken == 31)
    #expect(decode.actualToken == 99)
}

@Test
func loadGoldenCasesRejectsOutOfRangeToken() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    var prompt = correctnessPrompt()
    prompt[0] = MLXFastConstants.vocabSize
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad",
          "prompt_tokens": \(arrayJSON(prompt)),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsMissingVersion() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "cases": [
        {
          "name": "missing-version",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsDuplicateCaseNames() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "duplicate",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        },
        {
          "name": "duplicate",
          "prompt_tokens": \(correctnessPromptJSON(2)),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsNamesWithSurroundingWhitespace() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": " ambiguous ",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsNamesWithControlCharacters() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad\\nname",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsWrongExpectedTokenCount() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps - 1)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "wrong-count",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsWrongPromptTokenCount() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "wrong-prompt-count",
          "prompt_tokens": [1],
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path)
    }
}

@Test
func loadGoldenCasesRejectsNonPositiveRequiredSteps() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = [7]
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad-steps",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path, requiredSteps: 0)
    }
}

@Test
func loadGoldenCasesRejectsNonPositiveRequiredPromptTokens() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let expected = Array(repeating: 7, count: MLXFastConstants.correctnessSteps)
    let json = """
    {
      "version": 1,
      "cases": [
        {
          "name": "bad-prompt-steps",
          "prompt_tokens": \(correctnessPromptJSON()),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
    try json.write(to: path, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try loadGoldenCases(from: path.path, requiredPromptTokens: 0)
    }
}

// MARK: - attach-benchmark-oracle (goldenDocumentAttachingDerivedBenchmarkOracle)

// A base case long enough for the derived oracle to cover the timed decode
// window: expected_tokens must be >= benchmarkDecodeSteps + 1 (the decode seed
// next-token plus the 128 checked decode tokens). The real hidden goldens
// carry 256.
private func oracleSourceExpectedTokens(
    count: Int = MLXFastConstants.benchmarkDecodeSteps * 2
) -> [Int] {
    (0..<count).map { 900 + $0 }
}

private func oracleSourceDocument(
    expectedTokens: [Int]? = nil,
    cases: [GoldenCase]? = nil,
    gates: GoldenCorrectnessGates? = nil,
    benchmark: BenchmarkGolden? = nil
) -> GoldenDocument {
    GoldenDocument(
        version: 1,
        modelProvenance: GoldenModelProvenance(
            repository: MLXFastConstants.referenceModelRepository,
            revision: MLXFastConstants.referenceModelRevision
        ),
        cases: cases
            ?? [
                GoldenCase(
                    name: "hidden-base",
                    promptTokens: correctnessPrompt(11),
                    expectedTokens: expectedTokens ?? oracleSourceExpectedTokens()
                )
            ],
        correctnessGates: gates
            ?? GoldenCorrectnessGates(
                freeRun: [
                    GoldenFreeRunCase(
                        name: "free-run-decode-offset-coverage",
                        promptTokens: correctnessPrompt(11),
                        expectedTokens: Array(repeating: 5, count: MLXFastConstants.benchmarkDecodeSteps)
                    )
                ]
            ),
        benchmark: benchmark
    )
}

@Test
func attachDerivedBenchmarkOracleReproducesTheSerialEraPrecedentRule() throws {
    let source = oracleSourceDocument()
    let baseCase = source.cases[0]

    let merged = try goldenDocumentAttachingDerivedBenchmarkOracle(source)

    let oracle = try #require(merged.benchmark)
    // The five identities read off the DFlash ranked golden (94239d59): the
    // oracle restates the golden's own base case, introducing no new token.
    #expect(oracle.prefillPromptTokens == baseCase.promptTokens)
    #expect(oracle.decodeSeedTokens == baseCase.promptTokens)
    #expect(oracle.expectedPrefillToken == baseCase.expectedTokens[0])
    #expect(oracle.expectedDecodeSeedToken == baseCase.expectedTokens[0])
    #expect(oracle.expectedDecodeTokens == Array(baseCase.expectedTokens.dropFirst()))
    // Precedent shape: 1024 / 1024 / (expected - 1).
    #expect(oracle.prefillPromptTokens.count == MLXFastConstants.benchmarkPrefillPromptTokens)
    #expect(oracle.decodeSeedTokens.count == MLXFastConstants.benchmarkDecodeSeedTokens)
    #expect(oracle.expectedDecodeTokens.count == baseCase.expectedTokens.count - 1)
}

@Test
func attachDerivedBenchmarkOracleCarriesNoPerPromptBaselines() throws {
    let merged = try goldenDocumentAttachingDerivedBenchmarkOracle(oracleSourceDocument())

    let oracle = try #require(merged.benchmark)
    // A hidden correctness golden is not a prompt-pool golden, so it must not
    // carry pool-rotation baselines; scoring resolves to the calibrated
    // constants exactly as the DFlash precedent does.
    #expect(oracle.baselinePrefillSecondsPerToken == nil)
    #expect(oracle.baselineDecodeSecondsPerToken == nil)
    #expect(
        oracle.resolvedBaselinePrefillSecondsPerToken
            == MLXFastConstants.officialBaselinePrefillSecondsPerToken
    )
    #expect(
        oracle.resolvedBaselineDecodeSecondsPerToken
            == MLXFastConstants.officialBaselineDecodeSecondsPerToken
    )
}

@Test
func attachDerivedBenchmarkOracleLeavesEveryOtherSectionUntouched() throws {
    let anchors = [
        GoldenAnchorCase(name: "anchor-0", contextTokens: correctnessPrompt(3), expectedToken: 42)
    ]
    let behavior = [
        GoldenBehaviorCase(
            name: "gpqa-0",
            promptTokens: correctnessPrompt(4),
            acceptedTokenSequences: [[7, 8]],
            maxNewTokens: 16
        )
    ]
    let freeRun = [
        GoldenFreeRunCase(
            name: "free-run-decode-offset-coverage",
            promptTokens: correctnessPrompt(11),
            expectedTokens: Array(repeating: 5, count: MLXFastConstants.benchmarkDecodeSteps),
            exactPrefixTokens: 8
        )
    ]
    let source = oracleSourceDocument(
        gates: GoldenCorrectnessGates(anchors: anchors, freeRun: freeRun, behavior: behavior)
    )

    let merged = try goldenDocumentAttachingDerivedBenchmarkOracle(source)

    // ADDITIVE ONLY: `.benchmark` appears, nothing else moves.
    #expect(merged.cases == source.cases)
    #expect(merged.correctnessGates == source.correctnessGates)
    #expect(merged.correctnessGates?.anchors == anchors)
    #expect(merged.correctnessGates?.freeRun == freeRun)
    #expect(merged.correctnessGates?.behavior == behavior)
    #expect(merged.modelProvenance == source.modelProvenance)
    #expect(merged.version == source.version)
    #expect(source.benchmark == nil)
    #expect(merged.benchmark != nil)
}

@Test
func attachDerivedBenchmarkOracleRefusesToOverwriteAnExistingOracle() throws {
    // A golden that already carries an oracle may have had it MEASURED rather
    // than derived; silently replacing it would discard that provenance.
    let existing = BenchmarkGolden(
        prefillPromptTokens: Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens),
        expectedPrefillToken: 4,
        decodeSeedTokens: Array(repeating: 2, count: MLXFastConstants.benchmarkDecodeSeedTokens),
        expectedDecodeSeedToken: 5,
        expectedDecodeTokens: Array(repeating: 3, count: MLXFastConstants.benchmarkDecodeSteps)
    )
    let source = oracleSourceDocument(benchmark: existing)

    do {
        _ = try goldenDocumentAttachingDerivedBenchmarkOracle(source)
        Issue.record("expected an existing benchmark oracle to be refused")
    } catch let MLXFastError.invalidInput(message) {
        #expect(message.contains("already contains a benchmark oracle"))
    } catch {
        Issue.record("expected MLXFastError.invalidInput, got \(error)")
    }
}

@Test
func attachDerivedBenchmarkOracleRejectsABaseCaseShorterThanTheTimedDecodeWindow() throws {
    // benchmarkDecodeSteps expected tokens derive only benchmarkDecodeSteps-1
    // decode tokens, one short of covering the timed window: the validator
    // must reject it rather than write a golden that fails later on the box.
    let source = oracleSourceDocument(
        expectedTokens: oracleSourceExpectedTokens(count: MLXFastConstants.benchmarkDecodeSteps)
    )

    do {
        _ = try goldenDocumentAttachingDerivedBenchmarkOracle(source)
        Issue.record("expected a short base case to be rejected")
    } catch let MLXFastError.invalidInput(message) {
        #expect(message.contains("expected_decode_tokens"))
    } catch {
        Issue.record("expected MLXFastError.invalidInput, got \(error)")
    }
}

@Test
func attachDerivedBenchmarkOracleAcceptsTheExactTimedWindowBoundary() throws {
    // One more expected token than the case above is exactly enough.
    let source = oracleSourceDocument(
        expectedTokens: oracleSourceExpectedTokens(count: MLXFastConstants.benchmarkDecodeSteps + 1)
    )

    let merged = try goldenDocumentAttachingDerivedBenchmarkOracle(source)

    #expect(merged.benchmark?.expectedDecodeTokens.count == MLXFastConstants.benchmarkDecodeSteps)
}

@Test
func attachDerivedBenchmarkOracleRejectsAGoldenWithNoBaseCases() throws {
    let source = oracleSourceDocument(cases: [])

    do {
        _ = try goldenDocumentAttachingDerivedBenchmarkOracle(source)
        Issue.record("expected a golden with no base cases to be rejected")
    } catch let MLXFastError.invalidInput(message) {
        #expect(message.contains("no base cases"))
    } catch {
        Issue.record("expected MLXFastError.invalidInput, got \(error)")
    }
}

@Test
func attachDerivedBenchmarkOracleOutputLoadsThroughTheStrictFixtureLoader() throws {
    // End-to-end shape check: what the verb writes must be exactly what the
    // ranked gates phase loads, oracle present, so the run gets past the
    // "benchmark golden file must contain a benchmark oracle" guard.
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("golden.json")
    let merged = try goldenDocumentAttachingDerivedBenchmarkOracle(oracleSourceDocument())

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(merged).write(to: path)

    let fixture = try loadGoldenFixture(from: path.path)

    #expect(fixture.benchmark != nil)
    #expect(fixture.cases == merged.cases)
    #expect(fixture.correctnessGates == merged.correctnessGates)
    #expect(fixture.benchmark?.prefillPromptTokens == fixture.cases[0].promptTokens)
    #expect(fixture.benchmark?.decodeSeedTokens == fixture.cases[0].promptTokens)
}

// The reference schema names the pinned model with `model_type`, and the whole
// existing golden corpus carries it. This fork's loader had swapped that key
// for `model_provenance`, so a corpus-shaped golden failed to load at all
// (score=null -> E3 PARITY: FAIL). `model_type` is the schema key; provenance
// stays as an ADDITIVE optional key, never as a rename.
@Test
func goldenLoaderAcceptsReferenceSchemaModelType() throws {
    func documentJSON(_ identityKeys: String) -> Data {
        Data(
            """
            {
              "version": 1,
              \(identityKeys)
              "cases": [{
                "name": "model-type-contract",
                "prompt_tokens": \(correctnessPromptJSON()),
                "expected_tokens": \(arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps)))
              }]
            }
            """.utf8
        )
    }

    func load(_ data: Data) throws -> GoldenFixture {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("golden.json")
        try data.write(to: path)
        return try loadGoldenFixture(from: path.path)
    }

    let provenanceJSON = """
        "model_provenance": {
            "repository": "\(MLXFastConstants.referenceModelRepository)",
            "revision": "\(MLXFastConstants.referenceModelRevision)"
        },
        """

    // 1. A corpus-shaped golden (model_type, no provenance) loads.
    let corpusShaped = try load(documentJSON("\"model_type\": \"qwen3_5_text\","))
    #expect(corpusShaped.modelType == "qwen3_5_text")
    #expect(corpusShaped.cases.count == 1)

    // 2. A golden carrying BOTH keys loads, and both survive the load.
    let both = try load(documentJSON("\"model_type\": \"qwen3_5_text\",\n\(provenanceJSON)"))
    #expect(both.modelType == "qwen3_5_text")
    #expect(both.cases.count == 1)

    // 3. Provenance-only (what this fork generates) still loads -- additive,
    //    so the existing in-repo goldens are not broken by the restore.
    let provenanceOnly = try load(documentJSON(provenanceJSON))
    #expect(provenanceOnly.modelType == nil)

    // 4. Unknown top-level keys are still rejected: the restore widens the
    //    schema by exactly one known key, it does not loosen the loader.
    #expect(throws: MLXFastError.self) {
        _ = try load(documentJSON("\"model_kind\": \"qwen3_5_text\","))
    }

    // model_type keeps the reference loader's value semantics.
    #expect(throws: MLXFastError.self) {
        _ = try load(documentJSON("\"model_type\": \" qwen3_5_text \","))
    }
    #expect(throws: MLXFastError.self) {
        _ = try load(documentJSON("\"model_type\": \"\","))
    }
    #expect(throws: MLXFastError.self) {
        _ = try load(documentJSON("\"model_type\": null,"))
    }
    #expect(throws: MLXFastError.self) {
        _ = try load(documentJSON("\"model_type\": 7,"))
    }
}

// Restoring the KEY was only half the reference's contract; this is the VALUE
// half. Measured on box at the previous tip: benchd and the reference both
// REJECTED `missing_model_type.json` and `wrong_model_type.json` while this
// fork ACCEPTED both -- the fork was the odd one out, and E3 loader-parity
// stayed FAIL with those two rows as the only MISMATCHes. The diagnostics are
// asserted as STRINGS, not just as "some error", because the fork's job here is
// to agree with a specific reference wording, and an equivalent-but-different
// message is still a divergence a reader has to reconcile by hand.
@Test
func goldenModelIdentityFailsClosedWhenRequired() throws {
    func documentJSON(_ identityKeys: String) -> Data {
        Data(
            """
            {
              "version": 1,
              \(identityKeys)
              "cases": [{
                "name": "model-identity-contract",
                "prompt_tokens": \(correctnessPromptJSON()),
                "expected_tokens": \(arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps)))
              }]
            }
            """.utf8
        )
    }

    // Returns the loader's message on reject and nil on accept, so a test can
    // assert the exact diagnostic rather than merely that something threw.
    func rejection(
        _ data: Data,
        _ load: (String) throws -> GoldenFixture
    ) throws -> String? {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("golden.json")
        try data.write(to: path)
        do {
            _ = try load(path.path)
            return nil
        } catch {
            return String(describing: error)
        }
    }

    let qwen: (String) throws -> GoldenFixture = { try loadQwenGoldenFixture(from: $0) }
    let agnostic: (String) throws -> GoldenFixture = { try loadGoldenFixture(from: $0) }

    let provenanceJSON = """
        "model_provenance": {
            "repository": "\(MLXFastConstants.referenceModelRepository)",
            "revision": "\(MLXFastConstants.referenceModelRevision)"
        },
        """

    // 1. ABSENT model_type is a REJECT once a caller names the model it wants.
    //    "This golden names no model" is not "this golden names mine".
    #expect(
        try rejection(documentJSON(""), qwen)
            == "correctness golden file model_type=nil expected prism_hadamard_qwen35"
    )

    // 2. A golden naming the WRONG model is rejected, Optional-wrapped exactly
    //    as the reference prints it (benchd's `{:?}` renders the same shape).
    #expect(
        try rejection(documentJSON("\"model_type\": \"gemma_text\","), qwen)
            == "correctness golden file model_type=Optional(\"gemma_text\") "
                + "expected prism_hadamard_qwen35"
    )

    // 3. The pinned identity is accepted, and it is the constant that decides
    //    -- no call site respells the literal. The provenance block rides
    //    along because the Qwen loader demands one. This leg is about the
    //    model identity; absence of the block has its own test.
    let accepted = try rejection(
        documentJSON(
            "\"model_type\": \"\(MLXFastConstants.requiredGoldenModelType)\",\n"
                + provenanceJSON
        ),
        qwen
    )
    #expect(accepted == nil)
    #expect(MLXFastConstants.requiredGoldenModelType == "prism_hadamard_qwen35")

    // 4. `requiredModelType` is opt-in: the model-agnostic loader is unchanged,
    //    so every non-Qwen caller and the whole existing corpus keep loading.
    //    `requireModelProvenance` is opt-in the same way, which is why neither
    //    document below is asked for a provenance block.
    #expect(try rejection(documentJSON(""), agnostic) == nil)
    #expect(try rejection(documentJSON("\"model_type\": \"gemma_text\","), agnostic) == nil)

    // 5. model_provenance under the identity check: present alongside
    //    model_type it loads, and a mismatched block rejects on ITS own
    //    message, so the identity check did not swallow the pin check.
    #expect(
        try rejection(
            documentJSON("\"model_type\": \"prism_hadamard_qwen35\",\n\(provenanceJSON)"),
            qwen
        ) == nil
    )
    // ...and a mismatched provenance still rejects on ITS own message, so the
    // identity check did not swallow the pin check.
    #expect(
        try rejection(
            documentJSON(
                """
                "model_type": "prism_hadamard_qwen35",
                "model_provenance": {
                    "repository": "mlx-community/Laguna-XS-2.1-4bit",
                    "revision": "c42e0a8f8d504ceacde015a535dcb286d65c8799"
                },
                """
            ),
            qwen
        ) == "correctness golden model_provenance does not match the pinned reference model"
    )
}

// A golden's model_type must survive a rewrite: the attach tools decode and
// re-encode the document, and a dropped key would silently strip the corpus's
// identity key back off on the first oracle attach.
@Test
func attachingBenchmarkOraclePreservesModelType() throws {
    let document = GoldenDocument(
        version: 1,
        modelType: "qwen3_5_text",
        cases: [
            GoldenCase(
                name: "preserve-model-type",
                promptTokens: correctnessPrompt(),
                // The derived oracle needs benchmarkDecodeSteps + 1 expected
                // tokens (the seed next-token plus the checked decode window).
                expectedTokens: Array(
                    repeating: 7,
                    count: MLXFastConstants.benchmarkDecodeSteps + 1
                )
            )
        ],
        benchmark: nil
    )

    let merged = try goldenDocumentAttachingDerivedBenchmarkOracle(document)
    #expect(merged.modelType == "qwen3_5_text")

    let encoded = try JSONEncoder().encode(merged)
    let root = try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
    #expect(root?["model_type"] as? String == "qwen3_5_text")
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func correctnessPrompt(_ token: Int = 1) -> [Int] {
    Array(repeating: token, count: MLXFastConstants.correctnessPromptTokens)
}

private func correctnessPromptJSON(_ token: Int = 1) -> String {
    arrayJSON(correctnessPrompt(token))
}

private func arrayJSON(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}

/// THE SHIPPED PUBLIC CAPTURES LOAD. `--local-iterate` and `--local-submit`
/// check correctness against the two captures under
/// `correctness_prompts/bonsai2-27b-mlx-v1/`. The organizer recorded them on
/// the track's box against the pinned target, so the strict loader must accept
/// each one at the step count its mode demands, and the short capture must be
/// the long capture truncated: they are one recording. The digests are
/// measured from the tree (`shasum -a 256`, `wc -c`), so a re-recording
/// replaces them here in the same commit as the files.
@Test
func theShippedPublicCapturesLoadThroughTheStrictLoader() throws {
    let directory = "correctness_prompts/bonsai2-27b-mlx-v1/"
    let localIteratePath = directory + "public-local-iterate.golden.json"
    let localSubmitPath = directory + "public-local-submit.golden.json"

    func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    let localIterateData = try Data(contentsOf: URL(fileURLWithPath: localIteratePath))
    #expect(digest(localIterateData) == "cc5952f782f6ddba1ff46abf173c3372704579d82d4d6a5a2c45d90cde42be9e")
    #expect(localIterateData.count == 10_663)
    let localSubmitData = try Data(contentsOf: URL(fileURLWithPath: localSubmitPath))
    #expect(digest(localSubmitData) == "c1c42007bfd0989543c4b4d664483749bd3c278ec998915b80a294bd505122fd")
    #expect(localSubmitData.count == 21_007)

    // Each mode's golden must hold one more expected token than the mode
    // decodes: 128 steps for --local-iterate, 1023 for --local-submit.
    let short = try loadQwenGoldenFixture(
        from: localIteratePath,
        requiredSteps: MLXFastConstants.localIterateBenchmarkDecodeSteps + 1
    )
    let long = try loadQwenGoldenFixture(from: localSubmitPath, requiredSteps: 1_024)

    #expect(short.cases.count == 1)
    #expect(long.cases.count == 1)
    let shortCase = try #require(short.cases.first)
    let longCase = try #require(long.cases.first)
    #expect(shortCase.name == longCase.name)
    #expect(shortCase.promptTokens.count == MLXFastConstants.correctnessPromptTokens)
    #expect(shortCase.promptTokens == longCase.promptTokens)
    #expect(shortCase.expectedTokens.count == 256)
    #expect(longCase.expectedTokens.count == 1_024)
    #expect(Array(longCase.expectedTokens.prefix(256)) == shortCase.expectedTokens)
}
