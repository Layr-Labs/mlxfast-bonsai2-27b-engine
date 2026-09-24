// Qwen4ExpCBv2Tests.swift
//
// ContinuousBatchingV2 conformance tests for Qwen 3.8 Flash-Next.
//
// Every test runs on the tiny seeded-random fixture in
// `Qwen4ExpTestSupport.swift`. Nothing is downloaded and the real checkpoint
// is never read.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpCBv2Tests: XCTestCase {

    // MARK: - (a) Layer kinds, recurrent spec, auxiliary-index convention

    func testLayerKindsCoverOnlyFullAttentionLayers() throws {
        let configuration = try Qwen4ExpFixture.configuration()
        XCTAssertEqual(configuration.layerTypes.count, 4)
        XCTAssertEqual(configuration.fullAttentionLayerIndices, [1, 3])
        XCTAssertEqual(configuration.recurrentLayerIndices, [0, 2])

        let kinds = configuration.cbv2LayerKinds
        XCTAssertEqual(kinds.count, 2)
        XCTAssertEqual(kinds.map(\.modelLayerIndex), [1, 3])
        for kind in kinds {
            XCTAssertEqual(kind.attention, .full)
            XCTAssertNil(kind.sharesKVWithLayer)
            XCTAssertFalse(kind.hasSinks)
            XCTAssertEqual(kind.headDim, configuration.headDim)
            XCTAssertEqual(kind.kvHeads, configuration.kvHeads)
            XCTAssertEqual(kind.queryHeads, configuration.attentionHeads)
        }
    }

    func testRecurrentSpecCoversDeltanetLayersAndPLEState() throws {
        let configuration = try Qwen4ExpFixture.configuration()
        let spec = configuration.cbv2RecurrentStateSpec(activationDType: .float32)

        // Two gated-deltanet layers plus one auxiliary PLE entry.
        XCTAssertEqual(spec.layers.count, 3)
        XCTAssertEqual(spec.modelLayerIndices, [0, 2, 4])

        let deltanet = try XCTUnwrap(spec.layers.first { $0.modelLayerIndex == 0 })
        let keyDim = configuration.linearNumKeyHeads * configuration.linearKeyHeadDim
        let valueDim = configuration.linearNumValueHeads * configuration.linearValueHeadDim
        XCTAssertEqual(
            deltanet.convShape, [1, configuration.linearConvKernelDim - 1, 2 * keyDim + valueDim])
        XCTAssertEqual(
            deltanet.ssmShape,
            [
                1, configuration.linearNumValueHeads, configuration.linearValueHeadDim,
                configuration.linearKeyHeadDim,
            ])
        XCTAssertEqual(deltanet.ssmDType, .float32)

        // The byte ledger must accept the spec: duplicate keys would throw.
        XCTAssertNoThrow(try spec.fixedBytesPerRequest())
    }

    /// Pins the auxiliary-state convention the runner lane reads.
    func testAuxiliaryStateLayerIndicesArePastTheLastRealLayer() throws {
        let configuration = try Qwen4ExpFixture.configuration()
        XCTAssertEqual(configuration.cbv2AuxiliaryStateLayerIndices, [4])
        XCTAssertEqual(configuration.pleStateLayerIndex(ordinal: 0), configuration.hiddenLayers)

        // No auxiliary key may collide with a decoder layer's own key.
        let decoderKeys = Set(0 ..< configuration.hiddenLayers)
        for key in configuration.cbv2AuxiliaryStateLayerIndices {
            XCTAssertFalse(decoderKeys.contains(key))
        }

        let spec = configuration.cbv2RecurrentStateSpec(activationDType: .float32)
        let auxiliary = try XCTUnwrap(spec.layers.first { $0.modelLayerIndex == 4 })
        // conv holds the short convolution, ssm holds the token history.
        XCTAssertEqual(
            auxiliary.convShape,
            [
                1, (configuration.pleConvKernelSize - 1) * configuration.ngramSize,
                configuration.hcCount * configuration.hiddenSize,
            ])
        XCTAssertEqual(auxiliary.ssmShape, [1, configuration.ngramSize - 1])
        XCTAssertEqual(auxiliary.ssmDType, .int32)

        let model = try Qwen4ExpFixture.model()
        XCTAssertEqual(model.cbv2AuxiliaryStateLayerIndices, [4])
    }

    func testCapabilitiesAreExplicit() throws {
        let capabilities = try Qwen4ExpFixture.configuration().cbv2Capabilities
        XCTAssertFalse(capabilities.supportsPagedKV)
        XCTAssertFalse(capabilities.supportsPrefixReuse)
        XCTAssertFalse(capabilities.supportsPackedPrefill)
        XCTAssertFalse(capabilities.supportsCompiledDecode)
        XCTAssertFalse(capabilities.supportsCompactRecurrentMTPReplay)
        XCTAssertTrue(capabilities.supportsMTP)
    }

    // MARK: - (e) Rollback keeps the indexer tape in step with the KV tape

    func testIndexerTapeFollowsRowRollback() throws {
        let model = try Qwen4ExpFixture.model(withMTP: false)
        let kind = model.cbv2LayerKinds[0]
        let cache = Qwen4ExpCBv2LayerCache(layerIndex: kind.modelLayerIndex ?? 0, kind: kind)
        let row = CBv2FullSequenceKV(
            promptLength: 0, maxLength: 64, kvHeads: kind.kvHeads, headDim: kind.headDim)
        cache.setRows([row])

        let headDim = try Qwen4ExpFixture.configuration().indexerHeadDim
        func step(_ count: Int) {
            _ = cache.updateIndexerTape(keys: MLXRandom.normal([1, count, headDim]))
            // The key-value tape advances with it, exactly as the model does.
            _ = row.update(
                keys: MLXRandom.normal([1, kind.kvHeads, count, kind.headDim]),
                values: MLXRandom.normal([1, kind.kvHeads, count, kind.headDim]))
        }

        step(4)
        XCTAssertEqual(row.absoluteOffset, 4)
        XCTAssertEqual(cache.indexerTapeLength, 4)

        step(2)
        XCTAssertEqual(row.absoluteOffset, 6)
        XCTAssertEqual(cache.indexerTapeLength, 6)

        // A rejected speculative tail: the engine rolls the ROW back and the
        // tape must follow on the next append, not keep stale rows.
        row.rollback(3)
        XCTAssertEqual(row.absoluteOffset, 3)
        step(1)
        XCTAssertEqual(row.absoluteOffset, 4)
        XCTAssertEqual(
            cache.indexerTapeLength, 4,
            "the indexer tape kept rows the key-value tape rolled back")

        // A row that leaves the batch takes its tape with it.
        cache.setRows([])
        cache.setRows([row])
        XCTAssertEqual(cache.indexerTapeLength, 0)
    }

}
