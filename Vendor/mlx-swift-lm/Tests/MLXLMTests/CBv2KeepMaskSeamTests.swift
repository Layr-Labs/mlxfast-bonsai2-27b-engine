// CBv2KeepMaskSeamTests.swift
//
// The optional attention keep mask: the contiguous backend honors it exactly,
// a backend that makes no claim refuses it by name, and the engine refuses at
// CONSTRUCTION when a model that needs it meets a provider that cannot serve
// it.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class CBv2KeepMaskSeamTests: XCTestCase {

    private let kind = CBv2LayerKind(
        attention: .full, headDim: 8, kvHeads: 2, queryHeads: 4, modelLayerIndex: 0)

    private func makeRow(maxLength: Int = 64) -> CBv2FullSequenceKV {
        CBv2FullSequenceKV(
            promptLength: 0, maxLength: maxLength, kvHeads: kind.kvHeads, headDim: kind.headDim)
    }

    // MARK: - Contiguous backend: exact

    /// A keep mask that admits everything the causal mask admits must change
    /// nothing at all.
    func testAllTrueKeepMaskIsTheDenseResult() {
        MLXRandom.seed(11)
        let length = 6
        let queries = MLXRandom.normal([1, kind.queryHeads, length, kind.headDim])
        let keys = MLXRandom.normal([1, kind.kvHeads, length, kind.headDim])
        let values = MLXRandom.normal([1, kind.kvHeads, length, kind.headDim])
        let scale = 1.0 / Float(kind.headDim).squareRoot()

        let dense = CBv2LayerCache(layerIndex: 0, kind: kind)
        dense.setRows([makeRow()])
        let denseOut = dense.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)

        let masked = CBv2LayerCache(layerIndex: 0, kind: kind)
        masked.setRows([makeRow()])
        let keepMask = MLXArray.full([1, 1, length, length], values: MLXArray(true), dtype: .bool)
        let maskedOut = masked.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil,
            keepMask: keepMask)

        eval(denseOut, maskedOut)
        XCTAssertEqual(denseOut.shape, maskedOut.shape)
        XCTAssertLessThan(
            (denseOut - maskedOut).abs().max().item(Float.self), 1e-5,
            "an all-true keep mask changed the attention result")
    }

    /// A decode query restricted to a subset of keys must equal attention over
    /// exactly that subset, gathered. This is an INDEPENDENT reference: it
    /// never builds a mask, it removes the columns.
    func testKeepMaskSelectsExactlyTheKeptColumns() {
        MLXRandom.seed(12)
        let history = 7
        let keptColumns = [0, 2, 5]
        let scale = 1.0 / Float(kind.headDim).squareRoot()

        let cache = CBv2LayerCache(layerIndex: 0, kind: kind)
        let row = makeRow()
        cache.setRows([row])
        let historyKeys = MLXRandom.normal([1, kind.kvHeads, history, kind.headDim])
        let historyValues = MLXRandom.normal([1, kind.kvHeads, history, kind.headDim])
        _ = cache.updateAndAttend(
            queries: MLXRandom.normal([1, kind.queryHeads, history, kind.headDim]),
            keys: historyKeys, values: historyValues, scale: scale, sinks: nil)

        let queries = MLXRandom.normal([1, kind.queryHeads, 1, kind.headDim])
        let stepKeys = MLXRandom.normal([1, kind.kvHeads, 1, kind.headDim])
        let stepValues = MLXRandom.normal([1, kind.kvHeads, 1, kind.headDim])

        var keep = [Bool](repeating: false, count: history + 1)
        for column in keptColumns { keep[column] = true }
        let keepMask = MLXArray(keep).reshaped([1, 1, 1, history + 1])
        let masked = cache.updateAndAttend(
            queries: queries, keys: stepKeys, values: stepValues, scale: scale, sinks: nil,
            keepMask: keepMask)

        // Reference: attend the gathered columns with no mask at all.
        let allKeys = concatenated([historyKeys, stepKeys], axis: 2)
        let allValues = concatenated([historyValues, stepValues], axis: 2)
        let selection = MLXArray(keptColumns.map { Int32($0) })
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: takeAlong(
                allKeys,
                MLX.broadcast(
                    selection.reshaped([1, 1, keptColumns.count, 1]),
                    to: [1, kind.kvHeads, keptColumns.count, kind.headDim]),
                axis: 2),
            values: takeAlong(
                allValues,
                MLX.broadcast(
                    selection.reshaped([1, 1, keptColumns.count, 1]),
                    to: [1, kind.kvHeads, keptColumns.count, kind.headDim]),
                axis: 2),
            scale: scale,
            mask: .none)

        eval(masked, reference)
        XCTAssertLessThan(
            (masked - reference).abs().max().item(Float.self), 1e-5,
            "the keep mask did not select exactly the kept columns")
    }

    /// The mask must still be intersected with causality, not replace it.
    func testKeepMaskCannotAdmitFutureKeys() {
        MLXRandom.seed(13)
        let length = 5
        let scale = 1.0 / Float(kind.headDim).squareRoot()
        let queries = MLXRandom.normal([1, kind.queryHeads, length, kind.headDim])
        let keys = MLXRandom.normal([1, kind.kvHeads, length, kind.headDim])
        let values = MLXRandom.normal([1, kind.kvHeads, length, kind.headDim])

        let causal = CBv2LayerCache(layerIndex: 0, kind: kind)
        causal.setRows([makeRow()])
        let causalOut = causal.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil)

        let permissive = CBv2LayerCache(layerIndex: 0, kind: kind)
        permissive.setRows([makeRow()])
        let allTrue = MLXArray.full([1, 1, length, length], values: MLXArray(true), dtype: .bool)
        let permissiveOut = permissive.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: scale, sinks: nil,
            keepMask: allTrue)

        eval(causalOut, permissiveOut)
        XCTAssertLessThan(
            (causalOut - permissiveOut).abs().max().item(Float.self), 1e-5,
            "a permissive keep mask let a query see future keys")
    }

    // MARK: - Capability plumbing

    /// A cache that makes no claim keeps the refusing protocol default, so the
    /// bank must report the provider as unable.
    private final class UnclaimingLayerCache: CBv2AttendingLayerCache {
        let layerIndex: Int
        let kind: CBv2LayerKind
        private(set) var rows: [CBv2SequenceKV] = []

        init(layerIndex: Int, kind: CBv2LayerKind) {
            self.layerIndex = layerIndex
            self.kind = kind
        }

        var positionOffsets: MLXArray { MLXArray([Int32]()) }
        func setRows(_ rows: [CBv2SequenceKV]) { self.rows = rows }
        func updateAndAttend(
            queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, sinks: MLXArray?
        ) -> MLXArray { queries }
        func attendBorrowing(
            source: CBv2AttendingLayerCache, queries: MLXArray, scale: Float, sinks: MLXArray?
        ) -> MLXArray { queries }
    }

    func testContiguousCachesClaimKeepMaskSupport() {
        let cache = CBv2LayerCache(layerIndex: 0, kind: kind)
        XCTAssertTrue((cache as? CBv2KeepMaskCapableCache)?.honorsKeepMask ?? false)
        XCTAssertTrue(CBv2LayerCacheBank(caches: [cache]).supportsKeepMask)
    }

    func testQwen4ExpCacheClaimsKeepMaskSupport() {
        let cache = Qwen4ExpCBv2LayerCache(layerIndex: 0, kind: kind)
        XCTAssertTrue((cache as? CBv2KeepMaskCapableCache)?.honorsKeepMask ?? false)
        XCTAssertTrue(CBv2LayerCacheBank(caches: [cache]).supportsKeepMask)
    }

    func testUnclaimingCacheMakesTheProviderRefuse() {
        let cache = UnclaimingLayerCache(layerIndex: 0, kind: kind)
        XCTAssertNil(cache as? CBv2KeepMaskCapableCache)
        XCTAssertFalse(CBv2LayerCacheBank(caches: [cache]).supportsKeepMask)

        // One unclaiming layer is enough to disqualify the whole provider.
        let mixed = CBv2LayerCacheBank(caches: [
            CBv2LayerCache(layerIndex: 0, kind: kind), cache,
        ])
        XCTAssertFalse(mixed.supportsKeepMask)
    }

    /// The paged backend has no keep-mask semantics and must not claim any.
    func testPagedLayerCacheDoesNotClaimKeepMaskSupport() {
        XCTAssertFalse(
            (PagedLayerCache.self as Any.Type) is CBv2KeepMaskCapableCache.Type,
            "the paged layer cache claimed keep-mask support it does not implement")
    }

    // MARK: - Construction-time refusal

    func testEngineRefusesAKeepMaskModelOnAnUnableProvider() throws {
        let model = try Qwen4ExpFixture.model(withMTP: false)
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        XCTAssertTrue(adapter.cbv2RequiresKeepMask)

        let unable = CBv2LayerCacheBank(caches: [
            UnclaimingLayerCache(layerIndex: 0, kind: kind)
        ])
        let violation = EngineV2.keepMaskViolation(model: adapter, cacheProvider: unable)
        let message = try XCTUnwrap(violation, "the engine accepted a provider that drops the mask")
        XCTAssertTrue(message.contains("supportsKeepMask == false"))
        XCTAssertTrue(message.contains("requires the attention keep mask"))

        let able = CBv2LayerCacheBank(caches: [CBv2LayerCache(layerIndex: 0, kind: kind)])
        XCTAssertNil(EngineV2.keepMaskViolation(model: adapter, cacheProvider: able))
    }

    /// A model that needs nothing must not be blocked by the new gate.
    func testEngineAcceptsAModelThatDoesNotNeedTheMask() throws {
        let model = try Qwen4ExpFixture.model(withMTP: false)
        let adapter = CBv2SteppableLanguageModelAdapter(model)
        let unable = CBv2LayerCacheBank(caches: [
            UnclaimingLayerCache(layerIndex: 0, kind: kind)
        ])
        // Same provider, but a model that makes no claim: no violation.
        XCTAssertNil(
            EngineV2.keepMaskViolation(
                model: NonRequiringSteppableModel(), cacheProvider: unable))
        XCTAssertNotNil(EngineV2.keepMaskViolation(model: adapter, cacheProvider: unable))
    }

    private final class NonRequiringSteppableModel: CBv2SteppableModel {
        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray { tokens }
    }
}
