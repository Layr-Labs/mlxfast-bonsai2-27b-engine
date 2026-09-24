import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Paired backends, never a paged implementation serving as its own oracle.
/// Tiny fixture precision is explicit; the real-artifact suite never casts or
/// requantizes loaded weights. Failures describe the boundary, not tensor data.
final class NemotronH35BackendParityTests: XCTestCase {
    func testCapturedFloat32WindowsMatchEverySerialPrefix() throws {
        let model = try NemotronH35BackendOracle.tinyModel(dtype: .float32)
        for width in 2...8 {
            try NemotronH35BackendOracle.capturedWindow(model: model,
                prompt: Array(NemotronH35BackendOracle.tinyPrompt.prefix(40)), width: width)
        }
    }
    func testCapturedTwoTokenVerificationMatchesSerialLogitsAndEveryCommittedPrefix() throws {
        let model = try NemotronH35BackendOracle.tinyModel(dtype: .bfloat16)
        for width in 2...8 {
            try NemotronH35BackendOracle.capturedWindow(model: model,
                prompt: Array(NemotronH35BackendOracle.tinyPrompt.prefix(40)), width: width)
        }
    }
    func testTinyFloat32CanonicalBackendLogitsAndStateRepeatExactly() throws {
        let model = try NemotronH35BackendOracle.tinyModel(dtype: .float32)
        for pages in [false, true] {
            try NemotronH35BackendOracle.steps(model: model, prompt: NemotronH35BackendOracle.tinyPrompt,
                referencePaged: pages, candidatePaged: pages, requireLogitBytes: true)
        }
    }
    func testTinyBFloat16CanonicalBackendLogitsAndStateRepeatExactly() throws {
        let model = try NemotronH35BackendOracle.tinyModel(dtype: .bfloat16)
        for pages in [false, true] {
            try NemotronH35BackendOracle.steps(model: model, prompt: NemotronH35BackendOracle.tinyPrompt,
                referencePaged: pages, candidatePaged: pages, requireLogitBytes: true)
        }
    }
    func testTinyFloat32EarlierAndLatestCheckpointSuffixesMatchContiguous() async throws {
        let model = try NemotronH35BackendOracle.tinyModel(dtype: .float32)
        try await NemotronH35BackendOracle.checkpoints(model: model, prompt: NemotronH35BackendOracle.tinyPrompt)
    }
    func testTinyBFloat16EarlierAndLatestCheckpointSuffixesMatchContiguous() async throws {
        let model = try NemotronH35BackendOracle.tinyModel(dtype: .bfloat16)
        try await NemotronH35BackendOracle.checkpoints(model: model, prompt: NemotronH35BackendOracle.tinyPrompt)
    }
}

/// Cross-backend storage and committed-token invariants, alongside the exact
/// same-backend logits/state repeat controls above. Contiguous and paged use
/// different existing attention reductions; do not claim bit-identical logits
/// between them. The investigation's strict cross-backend hypothesis and its
/// failed results are retained in the private qualification evidence archive.
final class NemotronH35StorageParityTests: XCTestCase {
    func testFloat32NativeStorageTokensAndRollback() throws {
        let model = try NemotronH35BackendOracle.tinyModel(dtype: .float32)
        try NemotronH35BackendOracle.steps(model: model,
            prompt: NemotronH35BackendOracle.tinyPrompt, requireLogitBytes: false)
    }
    func testBFloat16NativeStorageTokensAndRollback() throws {
        let model = try NemotronH35BackendOracle.tinyModel(dtype: .bfloat16)
        try NemotronH35BackendOracle.steps(model: model,
            prompt: NemotronH35BackendOracle.tinyPrompt, requireLogitBytes: false)
    }
}

/// Run explicitly and reject skips when claiming real-checkpoint qualification.
/// Kept out of the mandatory tiny suite's filter so absent weights cannot turn
/// a supposedly executed artifact gate into a tiny-only success.
final class NemotronH35RealBackendParityTests: XCTestCase {
    func testLoadedCapturedTwoTokenWindowMatchesSerialNativeState() throws {
        guard let path = ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_MTP_MODEL"] else {
            throw XCTSkip("Requires the converted Nemotron MTP artifact")
        }
        let directory = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let model = NemotronH35Model(try JSONDecoder().decode(NemotronH35Configuration.self, from: data))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        try loadWeights(modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization)
        eval(model)
        for width in 2...8 {
            try NemotronH35BackendOracle.capturedWindow(model: model,
                prompt: (0..<256).map { 1000 + $0 % 200 }, width: width)
        }
    }
    func testLoadedArtifactPairsNativeBackendsAndRestoredState() async throws {
        guard let path = ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_REAL_MODEL"],
              !path.isEmpty else {
            throw XCTSkip("Set DARKBLOOM_NEMOTRON35_REAL_MODEL to the verified Lightning artifact")
        }
        let directory = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(NemotronH35Configuration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = NemotronH35Model(configuration)
        try loadWeights(modelDirectory: directory, model: model,
                        perLayerQuantization: base.perLayerQuantization)
        eval(model.parameters().flattened().map(\.1))
        let spec = model.cbv2RecurrentStateSpec
        XCTAssertEqual(spec.layers.count, 23)
        XCTAssertTrue(spec.layers.allSatisfy { $0.convDType == .bfloat16 })
        XCTAssertTrue(spec.layers.allSatisfy { $0.ssmDType == .float32 })
        XCTAssertEqual(try spec.fixedBytesPerRequest(), 49_082_368)

        // The retained real-checkpoint serial fixture crosses both 256-token
        // SSM scan boundaries. The separate serial suite pins its MLX-LM IDs.
        let prompt = [10, 25708, 1010, 11, 1010, 10, 3263, 1010]
            + (0..<600).map { 1000 + ($0 % 200) }
            + [11, 1010, 10, 1503, 19464, 1010, 12, 1010]
        for pages in [false, true] {
            try NemotronH35BackendOracle.steps(model: model, prompt: prompt,
                referencePaged: pages, candidatePaged: pages, requireLogitBytes: true)
        }
        // Different existing attention reductions can propagate roundoff into
        // later layers of the full model. Cross-backend committed IDs remain
        // checked; exact state/rollback and logits are checked within EACH
        // canonical backend above, not asserted interchangeable between them.
        try NemotronH35BackendOracle.steps(model: model, prompt: prompt,
            requireLogitBytes: false, requireCrossBackendStateBytes: false)
        try await NemotronH35BackendOracle.checkpoints(model: model, prompt: prompt,
            canonicalBackendReferences: true)
    }
}

private enum NemotronH35BackendOracle {
    static var chunk: Int { max(256, CBv2AttentionV1.queryBlockSize) }
    static var tinyPrompt: [Int] { (0..<2 * chunk + 7).map { 1 + ($0 * 7) % 97 } }
    private static let capacity = 512 << 20
    private enum Failure: Error { case mismatch(String) }

    private static func require(_ condition: Bool, _ message: String) throws {
        guard condition else {
            XCTFail(message)
            throw Failure.mismatch(message)
        }
    }

    static func tinyModel(dtype: DType) throws -> NemotronH35Model {
        // Same miniature Lightning topology as NemotronHTests, with two Mamba
        // layers separated by MoE and one native attention layer.
        let data = Data("""
            {"model_type":"nemotron_h", "vocab_size":100, "hidden_size":64,
             "num_hidden_layers":4, "num_attention_heads":4, "num_key_value_heads":2,
             "head_dim":64, "mamba_num_heads":4, "mamba_head_dim":16,
             "ssm_state_size":16, "conv_kernel":4, "n_groups":2,
             "intermediate_size":128, "moe_intermediate_size":64,
             "moe_shared_expert_intermediate_size":64, "n_routed_experts":4,
             "num_experts_per_tok":2, "layers_block_type":["mamba","moe","mamba","attention"],
             "norm_eps":0.00001, "mamba_ssm_cache_dtype":"float32"}
            """.utf8)
        MLXRandom.seed(350035)
        let model = NemotronH35Model(try JSONDecoder().decode(NemotronH35Configuration.self, from: data))
        let cast = model.castPredicate
        model.update(parameters: ModuleParameters.unflattened(model.parameters().flattened().map {
            ($0.0, cast?($0.0) == false ? $0.1 : $0.1.asType(dtype))
        }))
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        return model
    }

    private struct TensorBytes: Equatable {
        let label: String
        let shape: [Int]
        let dtype: DType
        let data: Data

        init(_ label: String, _ value: MLXArray) {
            self.label = label
            shape = value.shape
            dtype = value.dtype
            data = value.asData(access: .copy).data
        }
    }

    private struct StateBytes: Equatable {
        let offsets: [Int]
        let tensors: [TensorBytes]

        func mismatchSummary(_ other: Self) -> String {
            if offsets != other.offsets { return "offsets \(offsets) versus \(other.offsets)" }
            if tensors.count != other.tensors.count { return "tensor counts differ" }
            for (actual, expected) in zip(tensors, other.tensors) where actual != expected {
                let firstByte = zip(actual.data, expected.data).enumerated()
                    .first(where: { $0.element.0 != $0.element.1 })?.offset
                return "tensor=\(actual.label), actual_shape=\(actual.shape), expected_shape=\(expected.shape), dtype=\(actual.dtype), bytes=\(actual.data.count)/\(expected.data.count), first_different_byte=\(firstByte ?? -1)"
            }
            return "no differing stored tensor"
        }
    }

    private final class PublicationWitness: @unchecked Sendable {
        private let lock = NSLock()
        private var observed: Set<Int> = []
        private var completed = false
        let required: Set<Int>
        let ready = XCTestExpectation(description: "complete encoded checkpoint publication")

        init(required: Set<Int>) { self.required = required }
        var snapshot: [Int] { lock.withLock { observed.sorted() } }
        func record(_ positions: [Int]) {
            lock.lock()
            observed.formUnion(positions)
            let signal = !completed && required.isSubset(of: observed)
            if signal { completed = true }
            lock.unlock()
            if signal { ready.fulfill() }
        }
    }

    private final class Storage {
        let backend: any CBv2KVBackend
        let bank: CBv2LayerCacheBank
        let types: [DType]

        init(model: NemotronH35Model, paged: Bool, maximumLength: Int) throws {
            let kinds = model.cbv2LayerKinds
            types = try XCTUnwrap(model.cbv2CompleteCheckpointKVDTypes)
            try require(types.count == kinds.count && !types.isEmpty, "missing native KV dtype contract")
            let caches: [any CBv2AttendingLayerCache]
            if paged {
                let pages = try PagedKVBackend(layerKinds: kinds, config: .init(
                    capacityBytes: capacity, maxPrefillChunk: chunk,
                    nominalMaxSequenceLength: maximumLength, segmentSizeBytes: 64 << 10,
                    layerDTypes: types))
                backend = pages
                let storage = pages.makeLayerCaches()
                let indices = Dictionary(uniqueKeysWithValues: kinds.enumerated().map {
                    ($0.element.modelLayerIndex ?? $0.offset, $0.offset)
                })
                caches = try model.newCacheV2 { index, _ in
                    storage[try XCTUnwrap(indices[index])]
                }
            } else {
                backend = CBv2ContiguousKVBackend(config: .init(
                    bytesCapacity: capacity, kvDType: types.max { $0.size < $1.size }!))
                caches = model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }
            }
            bank = CBv2LayerCacheBank(caches: caches)
        }
    }

    private final class Row {
        let storage: Storage
        let recurrent: CBv2RecurrentRequestState
        let adapter: CBv2SteppableLanguageModelAdapter
        var rows: [CBv2SequenceKV?]

        init(model: NemotronH35Model, paged: Bool, promptLength: Int) throws {
            let maximumLength = promptLength + 32
            storage = try Storage(model: model, paged: paged, maximumLength: maximumLength)
            recurrent = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
            adapter = CBv2SteppableLanguageModelAdapter(model)
            rows = try storage.backend.makeSequenceState(layerKinds: model.cbv2LayerKinds,
                promptLength: promptLength, maxLength: maximumLength)
        }

        func forward(_ tokens: [Int], commit: Bool = true) throws
            -> (logits: TensorBytes, token: Int, pending: CBv2RecurrentStateEvaluation)
        {
            let caches = storage.bank.layerCaches(rowStates: [rows])
            let binding = try recurrent.bind()
            let output = adapter.forward(tokens: MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count),
                caches: caches, recurrentState: [binding])
            try (storage.backend as? PagedKVBackend)?.pool.writeValidation.check()
            let roots = try binding.evaluate()
            let cacheRoots = try caches.flatMap { try XCTUnwrap($0 as? KVCache).innerState() }
            eval([output] + roots + cacheRoots)
            if commit { try binding.commit() }
            return (TensorBytes("target logits", output), output[0, -1].argMax().item(Int.self), binding)
        }

        func snapshot() throws -> StateBytes {
            var tensors: [TensorBytes] = []
            var offsets: [Int] = []
            for (index, optional) in rows.enumerated() {
                let row = try XCTUnwrap(optional)
                let state = try XCTUnwrap(row.snapshot())
                try require(state.keys.dtype == storage.types[index]
                    && state.values.dtype == storage.types[index], "KV dtype changed at layer \(index)")
                offsets.append(row.absoluteOffset)
                tensors.append(TensorBytes("keys \(index)", state.keys))
                tensors.append(TensorBytes("values \(index)", state.values))
            }
            for spec in recurrent.spec.layers {
                let layer = try XCTUnwrap(recurrent.state(modelLayerIndex: spec.modelLayerIndex))
                let conv = try XCTUnwrap(layer.conv), ssm = try XCTUnwrap(layer.ssm)
                try require(conv.dtype == spec.convDType && ssm.dtype == spec.ssmDType,
                            "recurrent dtype changed at layer \(spec.modelLayerIndex)")
                tensors.append(TensorBytes("conv \(spec.modelLayerIndex)", conv))
                tensors.append(TensorBytes("ssm \(spec.modelLayerIndex)", ssm))
            }
            return StateBytes(offsets: offsets, tensors: tensors)
        }

        func captured(_ tokens: [Int], keep: Int) throws -> [TensorBytes] {
            let caches = storage.bank.layerCaches(rowStates: [rows])
            let serialization = try caches.map { try XCTUnwrap($0 as? any CBv2MTPRectangularSerializing) }
            for c in serialization { c.mtpSerializesRectangularAttention = true }
            defer { for c in serialization { c.mtpSerializesRectangularAttention = false } }
            for row in rows.compactMap({ $0 }) { row.beginSpeculativeWrite() }
            let binding = try recurrent.bind()
            let output = adapter.forwardWithHiddenCaptured(
                tokens: MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count),
                caches: caches, recurrentState: [binding], positionIds: nil)
            let roots = try binding.evaluate()
            let cacheRoots = try caches.flatMap { try XCTUnwrap($0 as? KVCache).innerState() }
            eval([output.logits, output.lastHidden] + roots + cacheRoots)
            let columns = tokens.indices.map { TensorBytes("target logits", output.logits[0..., $0..<($0 + 1), 0...]) }
            if keep > 0 { try binding.commit(keepPositions: keep) } else { try binding.rollback() }
            for row in rows.compactMap({ $0 }) {
                if tokens.count > keep { row.rollback(tokens.count - keep) }
                row.commitSpeculativeWrite()
            }
            storage.bank.invalidateBoundComposition()
            return columns
        }

        func close() {
            storage.bank.releaseBoundRows()
            storage.backend.release(rows)
            rows.removeAll()
            recurrent.discardPendingAfterSynchronization()
            try? recurrent.release()
            XCTAssertEqual(storage.backend.bytesReserved, 0)
            if let paged = storage.backend as? PagedKVBackend { XCTAssertEqual(paged.bytesWired, 0) }
        }
    }

    static func steps(model: NemotronH35Model, prompt: [Int], referencePaged: Bool = false,
                      candidatePaged: Bool = true, requireLogitBytes: Bool = true,
                      requireCrossBackendStateBytes: Bool = true) throws {
        let reference = try Row(model: model, paged: referencePaged, promptLength: prompt.count)
        defer { reference.close() }
        let candidate = try Row(model: model, paged: candidatePaged, promptLength: prompt.count)
        defer { candidate.close() }
        var next = 0
        func paired(_ tokens: [Int], label: String) throws {
            let expected = try reference.forward(tokens), actual = try candidate.forward(tokens)
            try require(actual.token == expected.token, "\(label): committed token IDs differ")
            if requireLogitBytes {
                try require(actual.logits == expected.logits, "\(label): target logits differ in native bytes")
            }
            let actualState = try candidate.snapshot(), expectedState = try reference.snapshot()
            try require(actualState.offsets == expectedState.offsets, "\(label): native offsets differ")
            if requireCrossBackendStateBytes {
                try require(actualState == expectedState,
                            "\(label): native KV/recurrent state differs: \(actualState.mismatchSummary(expectedState))")
            }
            next = expected.token
        }
        for start in stride(from: 0, to: prompt.count, by: chunk) {
            let end = min(start + chunk, prompt.count)
            try paired(Array(prompt[start..<end]), label: "prefill through \(end)")
        }
        for step in 0..<4 { try paired([next], label: "serial decode \(step)") }
        let before = try reference.snapshot()
        let candidateBefore = try candidate.snapshot()
        if requireCrossBackendStateBytes {
            try require(candidateBefore == before, "pre-rollback states differ")
        }
        for row in (reference.rows + candidate.rows).compactMap({ $0 }) {
            try require(row.supportsSpeculativeWrites, "rollback fixture lacks native transaction support")
            row.beginSpeculativeWrite()
        }
        let expected = try reference.forward([next], commit: false)
        let actual = try candidate.forward([next], commit: false)
        try require(actual.token == expected.token && (!requireLogitBytes || actual.logits == expected.logits),
                    "pending serial step differs before rollback")
        if requireCrossBackendStateBytes {
            try require(candidate.snapshot() == reference.snapshot(), "pending KV/recurrent state differs")
        }
        try expected.pending.rollback()
        try actual.pending.rollback()
        for owner in [reference, candidate] {
            for row in owner.rows.compactMap({ $0 }) {
                row.rollback(1)
                row.commitSpeculativeWrite()
            }
            owner.storage.bank.invalidateBoundComposition()
            let canonicalBefore = owner === reference ? before : candidateBefore
            try require(owner.snapshot() == canonicalBefore, "rollback did not restore exact prior native bytes")
        }
        try paired([next], label: "readmit after rollback")
    }

    static func capturedWindow(model: NemotronH35Model, prompt: [Int], width: Int = 2) throws {
        for keep in 0...width {
            let serial = try Row(model: model, paged: true, promptLength: prompt.count)
            let candidate = try Row(model: model, paged: true, promptLength: prompt.count)
            let continuation = try Row(model: model, paged: true, promptLength: prompt.count)
            defer { serial.close(); candidate.close(); continuation.close() }
            let first = try serial.forward(prompt).token
            _ = try candidate.forward(prompt)
            _ = try continuation.forward(prompt)
            var expectedStates = [try serial.snapshot()]
            var expectedLogits: [TensorBytes] = []
            var window: [Int] = []
            var token = first
            for _ in 0..<width {
                window.append(token)
                let step = try serial.forward([token])
                token = step.token
                expectedLogits.append(step.logits)
                expectedStates.append(try serial.snapshot())
            }
            let actual = try candidate.captured(window, keep: keep)
            for column in 0..<width {
                try require(actual[column] == expectedLogits[column], "captured width=\(width) column=\(column) differs from serial native logits")
            }
            let expected = expectedStates[keep]
            let current = try candidate.snapshot()
            try require(current == expected, "captured keep=\(keep) state differs: \(current.mismatchSummary(expected))")
            for token in window.prefix(keep) { _ = try continuation.forward([token]) }
            let after = try candidate.forward([17]), reference = try continuation.forward([17])
            try require(after.logits == reference.logits, "continued target differs after captured prefix \(keep)")
            try require(candidate.snapshot() == continuation.snapshot(), "continued state differs after captured prefix \(keep)")
        }
    }

    private struct Run {
        let result: CBv2SchedCollected
        let archives: [CompleteCheckpointFixtureStore.Archive]
    }

    private static func run(model: NemotronH35Model, paged: Bool, prompt: [Int],
                            archives: [CompleteCheckpointFixtureStore.Archive] = [],
                            expectedSaved: Int = 0) async throws -> Run {
        let store = CompleteCheckpointFixtureStore(archives: archives, segmentBytes: 64 << 10)
        let storage = try Storage(model: model, paged: paged, maximumLength: prompt.count + 8)
        let engine = EngineV2(model: CBv2SteppableLanguageModelAdapter(model),
            layerKinds: model.cbv2LayerKinds, backend: storage.backend, cacheProvider: storage.bank,
            sampler: CBv2GreedySampler(), schedulerConfig: .init(maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: chunk, prefillChunkSize: chunk, maxWaiting: 2,
                enablePrefixCache: true), admissionConfig: .init(watermarkFraction: 0),
            completePrefixCache: store)
        let eligiblePositions = Array(stride(from: chunk, to: prompt.count, by: chunk)
            .filter { $0 > expectedSaved })
        // CompleteCheckpointCapture deliberately retains the first and latest
        // candidate, retiring intervening captures to bound memory. This is
        // not an every-chunk persistence contract, including after a restore.
        let expectedPositions = Set([eligiblePositions.first, eligiblePositions.last].compactMap { $0 })
        let publication = PublicationWitness(required: expectedPositions)
        engine.setCompletePrefixPublicationHandler { _, positions in publication.record(positions) }
        do {
            let request = CBv2Request(id: .init(3501), promptTokens: prompt,
                sampling: .init(temperature: 0), maxTokens: 6, cacheSalt: "backend-oracle",
                prefixCacheReceiptID: .init(4501))
            if expectedSaved > 0 {
                try require(store.stage(engine: engine, request: request), "checkpoint was not staged")
            }
            let result = await cbv2SchedCollect(try engine.submit(request), timeoutSeconds: 180)
            try require(result.finishReason == .length && result.tokens.count == 6,
                        "paired engine did not finish exactly six committed tokens")
            try require(result.usage?.prefixCachePrefillTokensSaved == expectedSaved,
                        "checkpoint saved-token boundary differs")
            try require(result.usage?.prefixCacheReplayTokens == 0, "unexpected checkpoint replay")
            // Successful generation can finish before the store's asynchronous
            // encoded writes/publication. Observe the actual readiness event;
            // taking saved immediately raced the two backends' different timing.
            if !expectedPositions.isEmpty {
                let publicationResult = await XCTWaiter.fulfillment(of: [publication.ready], timeout: 10)
                try require(publicationResult == .completed,
                    "complete publication did not settle: paged=\(paged), codec=\(engine.completeCheckpointCodec != nil), prompt=\(prompt.count), restored=\(expectedSaved), required=\(expectedPositions.sorted()), observed=\(publication.snapshot), stored=\(store.saved.map(\.manifest.position))")
            }
            let saved = Array(store.saved.dropFirst(archives.count))
            await engine.shutdown()
            XCTAssertEqual(storage.backend.bytesReserved, 0)
            XCTAssertEqual(engine.admissionForTesting.bytesReserved, 0)
            XCTAssertTrue(engine.loopForTesting.recurrentStates.isEmpty)
            if let pages = storage.backend as? PagedKVBackend { XCTAssertEqual(pages.bytesWired, 0) }
            return Run(result: result, archives: saved)
        } catch {
            await engine.shutdown()
            throw error
        }
    }

    private static func sameArchives(_ actual: [CompleteCheckpointFixtureStore.Archive],
                                     _ expected: [CompleteCheckpointFixtureStore.Archive]) throws {
        let actualPositions = actual.map(\.manifest.position)
        let expectedPositions = expected.map(\.manifest.position)
        try require(actualPositions == expectedPositions,
                    "checkpoint publication boundaries differ: actual=\(actualPositions), expected=\(expectedPositions)")
        for (a, b) in zip(actual, expected) {
            try require(a.manifest.tensors == b.manifest.tensors,
                        "checkpoint native tensor descriptors differ at \(a.manifest.position)")
            // Page boundaries can split native reads differently. Compare
            // complete logical tensor bytes, while rejecting holes/overlap.
            for (index, descriptor) in a.manifest.tensors.enumerated() {
                func bytes(_ archive: CompleteCheckpointFixtureStore.Archive) throws -> Data {
                    var result = Data()
                    for part in archive.chunks.filter({ $0.tensor == index }).sorted(by: { $0.offset < $1.offset }) {
                        try require(part.offset == result.count && !part.bytes.isEmpty,
                                    "checkpoint tensor segments contain a hole or overlap")
                        result.append(part.bytes)
                    }
                    try require(result.count == descriptor.byteCount, "checkpoint tensor byte count differs")
                    return result
                }
                try require(bytes(a) == bytes(b),
                            "encoded native state differs at checkpoint \(a.manifest.position), tensor \(index)")
            }
        }
    }

    static func checkpoints(model: NemotronH35Model, prompt: [Int],
                            canonicalBackendReferences: Bool = false) async throws {
        try require(prompt.count > 2 * chunk && prompt.count < 3 * chunk,
                    "fixture must cross two configured prefill chunks and retain a partial tail")
        let contiguous = try await run(model: model, paged: false, prompt: prompt)
        let paged = try await run(model: model, paged: true, prompt: prompt)
        try require(paged.result.tokens == contiguous.result.tokens, "cold paged token IDs differ from contiguous")
        if !canonicalBackendReferences { try sameArchives(paged.archives, contiguous.archives) }
        try require(contiguous.archives.map(\.manifest.position) == [chunk, 2 * chunk],
                    "both complete native boundaries must be exported")
        try require(paged.archives.allSatisfy { $0.manifest.backendLayout == CBv2CompleteCheckpointManifest.pagedLayout },
                    "candidate did not export native paged checkpoints")
        for earlier in [false, true] {
            var suffix = prompt + (0..<chunk).map { prompt[$0] }
            if earlier { suffix[chunk + 3] = (suffix[chunk + 3] + 1) % model.vocabularySize }
            let cold = try await run(model: model, paged: false, prompt: suffix)
            let boundary = earlier ? chunk : 2 * chunk
            for (usePages, donor) in [(false, contiguous), (true, paged)] {
                let canonicalCold: Run
                if canonicalBackendReferences && usePages {
                    canonicalCold = try await run(model: model, paged: true, prompt: suffix)
                } else {
                    canonicalCold = cold
                }
                let restored = try await run(model: model, paged: usePages, prompt: suffix,
                    archives: donor.archives, expectedSaved: boundary)
                try require(restored.result.tokens == canonicalCold.result.tokens,
                            "restored suffix committed IDs differ from canonical cold at \(boundary)")
                let eligible = Array(stride(from: chunk, to: suffix.count, by: chunk).filter { $0 > boundary })
                let retained = Set([eligible.first, eligible.last].compactMap { $0 }).sorted()
                try require(!retained.isEmpty, "restored suffix must cross a fresh state checkpoint")
                try require(restored.archives.map(\.manifest.position) == retained,
                            "restored checkpoint retention differs from first/latest policy")
                // A warm donor's first *new* checkpoint may be an intermediate
                // point retired by the longer cold donor. Recompute that exact
                // causal boundary rather than requiring equal retention sets.
                for archive in restored.archives {
                    let position = archive.manifest.position
                    let expected: CompleteCheckpointFixtureStore.Archive
                    if let existing = canonicalCold.archives.first(where: { $0.manifest.position == position }) {
                        expected = existing
                    } else {
                        let prefixOracle = try await run(model: model,
                            paged: canonicalBackendReferences && usePages,
                            prompt: Array(suffix.prefix(position + 1)))
                        expected = try XCTUnwrap(prefixOracle.archives.first(where: {
                            $0.manifest.position == position
                        }))
                    }
                    try sameArchives([archive], [expected])
                }
            }
        }
    }
}
