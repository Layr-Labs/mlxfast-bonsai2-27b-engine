// RunnerManifestTests.swift
//
// Manifest digest gates (Darkbloom runner contract §6.0).
//
// The canonical JSON bytes are PINNED here, spelled out in full, next to
// their sha256. A manifest edit therefore shows up as a diff of the exact
// bytes benchd will hash, not as an opaque hash change nobody can read — and
// a re-ordering of the declared fields cannot hide behind a stable digest,
// which is the reason the encoder walks the declared order instead of
// sorting keys.
//
// Model-free by construction: nothing here loads weights or touches Metal.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXRunners

@Suite("Runner manifest digests")
struct RunnerManifestTests {

    private func canonical(_ manifest: RunnerManifest) -> String {
        String(decoding: manifest.canonicalJSON(), as: UTF8.self)
    }

    /// The contract's §11 manifest for Qwen 3.8 Flash-Next. The DIGEST is the
    /// cross-repo test vector both sides pin, so these bytes are the
    /// interface, not an internal detail.
    @Test("Section 11 manifest reproduces the cross-repo test vector")
    func sectionElevenVector() {
        // `ContractManifests.sectionEleven` is `Qwen4ExpRunner.manifest`, and
        // it is also what the shared conformance fixture's hello is driven
        // over, so the digest pinned here and the digest that reaches the wire
        // cannot drift apart.
        let manifest = ContractManifests.sectionEleven

        #expect(
            canonical(manifest) == """
                {"schemaVersion":1,"runnerID":"layr/qwen4exp-125b-a6b","modelTypes":\
                ["qwen4_exp","qwen4_exp_text"],"backend":"mlx","engine":\
                {"supportsPrefixReuse":false,"supportsPagedKV":false,\
                "supportsCompiledDecode":false,"supportsPackedPrefill":false,\
                "supportsMTP":true,"supportsCompactRecurrentMTPReplay":false},\
                "kvBackends":["contiguous"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null},{"mode":"mtp",\
                "drafter":"embeddedHead","state":"requestStateful","depth":[1,6]}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":false,"recurrentLayers":true,"requiresKeepMask":true}
                """)
        #expect(
            manifest.sha256Digest()
                == "0430b22f8325c9c9371910d1e14eb3c78b235932bf35fa6623a4c511dd68e180")
    }

    /// The mock-adapter manifest benchd checked in beside the fixture,
    /// DECODED here and digested by this repo's own encoder. Both sides load
    /// the same bytes, so this is a cross-repo agreement on the §6.0
    /// encodings — the object form of `batch`, the array form of `depth`, the
    /// declared field order — not two Swift/Rust declarations that happen to
    /// agree today.
    @Test("The shared mock-adapter manifest decodes and digests as pinned")
    func mockAdapterManifestDigest() {
        let manifest = FixtureManifest.mockAdapter
        #expect(manifest.runnerID == "layr/mock-adapter")
        #expect(manifest.backend == "mock")
        #expect(
            canonical(manifest) == """
                {"schemaVersion":1,"runnerID":"layr/mock-adapter","modelTypes":\
                ["qwen4_exp","qwen4_exp_text"],"backend":"mock","engine":\
                {"supportsPrefixReuse":false,"supportsPagedKV":false,\
                "supportsCompiledDecode":false,"supportsPackedPrefill":false,\
                "supportsMTP":true,"supportsCompactRecurrentMTPReplay":false},\
                "kvBackends":["contiguous"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null},{"mode":"mtp",\
                "drafter":"embeddedHead","state":"requestStateful","depth":[1,3]}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":false,"recurrentLayers":true,"requiresKeepMask":true}
                """)
        #expect(
            manifest.sha256Digest()
                == "850ef7df262d03f851a355d1572257b04ade4834d2563c18f01b9cf344a6fffd")

        // It differs from the §11 manifest in exactly two fields, so the two
        // digests must differ — a decoder that dropped either would collide.
        #expect(manifest.sha256Digest() != ContractManifests.sectionEleven.sha256Digest())
    }

    @Test("Gemma 4 text manifest")
    func gemma4TextManifest() {
        #expect(
            canonical(Gemma4TextRunner.manifest) == """
                {"schemaVersion":1,"runnerID":"layr/gemma4-text","modelTypes":\
                ["gemma4","gemma4_text"],"backend":"mlx","engine":\
                {"supportsPrefixReuse":true,"supportsPagedKV":true,\
                "supportsCompiledDecode":true,"supportsPackedPrefill":true,\
                "supportsMTP":true,"supportsCompactRecurrentMTPReplay":false},\
                "kvBackends":["contiguous","paged"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null},{"mode":"mtp",\
                "drafter":"assistantCheckpoint","state":"stateless","depth":[1,7]}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":{"upTo":8},"timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":false,"recurrentLayers":false,"requiresKeepMask":false}
                """)
        #expect(
            Gemma4TextRunner.manifest.sha256Digest()
                == "e1890e74e9cefc901393f3826b26efeb9165d5a21c3505daea566487bf40b63d")
    }

    @Test("GPT-OSS manifest")
    func gptossManifest() {
        #expect(
            canonical(GPTOSSRunner.manifest) == """
                {"schemaVersion":1,"runnerID":"layr/gptoss","modelTypes":["gpt_oss"],\
                "backend":"mlx","engine":{"supportsPrefixReuse":true,\
                "supportsPagedKV":true,"supportsCompiledDecode":true,\
                "supportsPackedPrefill":true,"supportsMTP":false,\
                "supportsCompactRecurrentMTPReplay":false},\
                "kvBackends":["contiguous","paged"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":{"upTo":8},"timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":false,"recurrentLayers":false,"requiresKeepMask":false}
                """)
        #expect(
            GPTOSSRunner.manifest.sha256Digest()
                == "2afbe23fc67ea9f2e9fd017098bb8b2877166190c9d56966dbeaa05656fcf8a2")
    }

    /// Nemotron 3.5 Lightning. `kvBackends` is contiguous-only although the
    /// engine capability declares paged support: the model also requires the
    /// NATIVE segmented paged pool, which the shared assembly does not build.
    @Test("Nemotron 3.5 Lightning manifest")
    func nemotronH35Manifest() {
        #expect(
            canonical(NemotronH35Runner.manifest) == """
                {"schemaVersion":1,"runnerID":"layr/nemotron35-lightning","modelTypes":\
                ["nemotron_h"],"backend":"mlx","engine":\
                {"supportsPrefixReuse":false,"supportsPagedKV":true,\
                "supportsCompiledDecode":false,"supportsPackedPrefill":false,\
                "supportsMTP":true,"supportsCompactRecurrentMTPReplay":false},\
                "kvBackends":["contiguous"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null},{"mode":"mtp",\
                "drafter":"embeddedHead","state":"requestStateful","depth":[1,7]}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":false,"recurrentLayers":true,"requiresKeepMask":false}
                """)
        #expect(
            NemotronH35Runner.manifest.sha256Digest()
                == "10b8718181feb354dd6b0bd75e6a0df554f23313d3dfe3d018ccf91f77b74c73")
    }

    @Test("Qwen 3.5 manifest")
    func qwen35Manifest() {
        #expect(
            canonical(Qwen35Runner.manifest) == """
                {"schemaVersion":1,"runnerID":"layr/qwen35","modelTypes":\
                ["prism_hadamard_qwen35","qwen3_5","qwen3_5_moe","qwen3_5_text"],\
                "backend":"mlx","engine":\
                {"supportsPrefixReuse":false,"supportsPagedKV":false,\
                "supportsCompiledDecode":false,"supportsPackedPrefill":true,\
                "supportsMTP":true,"supportsCompactRecurrentMTPReplay":true},\
                "kvBackends":["contiguous"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null},{"mode":"mtp",\
                "drafter":"assistantCheckpoint","state":"requestStateful","depth":[1,7]},\
                {"mode":"dflash","drafter":"assistantCheckpoint",\
                "state":"requestStateful","depth":[1,16]}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":{"upTo":8},"timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":false,"recurrentLayers":true,"requiresKeepMask":false}
                """)
        #expect(
            Qwen35Runner.manifest.sha256Digest()
                == "0353932f2fc83a43ab6bb0324e05f7b52f7cb7805fb89ef1f404f624472fa299")
    }

    @Test("Qwen3-VL manifest")
    func qwen3vlManifest() {
        #expect(
            canonical(Qwen3VLRunner.manifest) == """
                {"schemaVersion":1,"runnerID":"layr/qwen3vl","modelTypes":\
                ["qwen3_vl","qwen3_vl_moe"],"backend":"mlx","engine":\
                {"supportsPrefixReuse":false,"supportsPagedKV":false,\
                "supportsCompiledDecode":false,"supportsPackedPrefill":false,\
                "supportsMTP":false,"supportsCompactRecurrentMTPReplay":false},\
                "kvBackends":["contiguous"],"decoders":[{"mode":"serial",\
                "drafter":"none","state":"stateless","depth":null}],\
                "regimes":[{"batch":"single","timing":"freeRun","perStreamTiming":false},\
                {"batch":"single","timing":"teacherForced","perStreamTiming":false}],\
                "multimodal":true,"recurrentLayers":false,"requiresKeepMask":false}
                """)
        #expect(
            Qwen3VLRunner.manifest.sha256Digest()
                == "a9145332bb9a78961831bc365af055348a8cedc45f8556ad58573ec32d821d3c")
    }

    /// Round trip through `Codable` in the §6.0 encodings: `batch` as the
    /// string `"single"` or the object `{"upTo": n}`, `depth` as `[lo, hi]`
    /// or `null`. The synthesized enum/range forms would decode to the same
    /// Swift values and to DIFFERENT bytes, so the assertion is on the JSON.
    @Test("Manifest Codable uses the pinned wire encodings")
    func codableEncodings() throws {
        let manifest = Gemma4TextRunner.manifest
        let encoded = try JSONEncoder().encode(manifest)
        let text = String(decoding: encoded, as: UTF8.self)
        #expect(text.contains("\"batch\":{\"upTo\":8}"))
        #expect(text.contains("\"batch\":\"single\""))
        #expect(text.contains("\"depth\":[1,7]"))
        #expect(text.contains("\"depth\":null"))

        let decoded = try JSONDecoder().decode(RunnerManifest.self, from: encoded)
        #expect(decoded == manifest)
        #expect(decoded.sha256Digest() == manifest.sha256Digest())
    }
}
