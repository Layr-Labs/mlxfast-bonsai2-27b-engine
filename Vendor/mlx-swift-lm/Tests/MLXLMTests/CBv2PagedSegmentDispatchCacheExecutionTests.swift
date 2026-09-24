import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
import XCTest
@_spi(Diagnostics) @testable import MLXLMCommon

@Suite("CBv2 cached segmented execution", .serialized)
struct CBv2PagedSegmentDispatchCacheExecutionTests {
    @Test(arguments: [DType.bfloat16, .float32])
    func lazySuccessorsKeepRecordsAndLiveWriteFenceAcrossReplacement(dtype: DType) throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
        let q = MLXRandom.normal([1, 2, 1, 64], key: MLXRandom.key(91)).asType(dtype)
        let k = MLXRandom.normal([1, 1, 1, 64], key: MLXRandom.key(92)).asType(dtype)
        let v = MLXRandom.normal([1, 1, 1, 64], key: MLXRandom.key(93)).asType(dtype)
        let prefixK = MLXRandom.normal([1, 30, 64], key: MLXRandom.key(94)).asType(dtype)
        let prefixV = MLXRandom.normal([1, 30, 64], key: MLXRandom.key(95)).asType(dtype)
        func run(clearEachStep: Bool) throws -> (outputs: [Data], keys: Data, values: Data, hits: Int) {
            let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
                capacityBytes: 1 << 20, dtype: dtype, maxPrefillChunk: 64,
                segmentSizeBytes: 2 * 16 * 64 * dtype.size * 3))
            let state = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
            defer { backend.release(state) }
            let row = try #require(state[0] as? PagedSequenceKV)
            row.write(keys: prefixK, values: prefixV)
            eval(backend.pool.group(row.groupKey).writeFence)
            let layer = try #require(backend.makeLayerCaches()[0] as? PagedLayerCache)
            layer.segmentDispatchCache.statistics = .init()
            var outputs: [MLXArray] = []
            var carry = MLXArray(Float(0)).asType(dtype)
            for _ in 0..<6 {
                layer.setRows([row])
                if clearEachStep { layer.segmentDispatchCache.clear() }
                let output = layer.updateAndAttend(queries: q + carry, keys: k, values: v,
                                                   scale: 0.125, sinks: nil)
                outputs.append(output)
                carry = output[0, 0, 0, 0] * MLXArray(Float(0)).asType(dtype)
            }
            // The old metadata is replaced at offset33 while its graph is still
            // pending. One final fence must safely execute all six successors.
            let snapshot = row.snapshot()
            eval(outputs + [snapshot.keys, snapshot.values])
            let data = outputs.map { $0.asData(access: .copy).data }
            let result = (data, snapshot.keys.asData(access: .copy).data,
                          snapshot.values.asData(access: .copy).data,
                          layer.segmentDispatchCache.statistics!.hits)
            layer.setRows([])
            #expect(layer.segmentDispatchCache.prepared == nil)
            return result
        }
        let fresh = try run(clearEachStep: true), cached = try run(clearEachStep: false)
        #expect(fresh.outputs == cached.outputs)
        #expect(fresh.keys == cached.keys && fresh.values == cached.values)
        #expect(fresh.hits == 0 && cached.hits == 4)
    }

    @Test func adoptedSharedPrefixGetsPrivateTailAndFreshRowIdentity() throws {
        let kind = CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 2)
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 2 << 20, dtype: .bfloat16, maxPrefillChunk: 64,
            segmentSizeBytes: 2 * 16 * 64 * 2 * 3))
        let donor = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        let adopter = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 64)
        defer { backend.release(donor); backend.release(adopter) }
        let a = try #require(donor[0] as? PagedSequenceKV), b = try #require(adopter[0] as? PagedSequenceKV)
        a.write(keys: MLXArray.ones([1, 32, 64], dtype: .bfloat16),
                values: MLXArray.full([1, 32, 64], values: MLXArray(Float(2)), dtype: .bfloat16))
        let handles = try #require(a.prefixPageHandles(tokens: 0..<32))
        #expect(backend.pool.retainPages(handles))
        b.adoptSharedPages(handles, storedThrough: 32, cursor: 32, frozenThrough: 0)
        let group = backend.pool.group(a.groupKey)
        eval(group.writeFence)
        let layer = try #require(backend.makeLayerCaches()[0] as? PagedLayerCache)
        func decode(_ row: PagedSequenceKV, value: Float) -> MLXArray {
            layer.setRows([row])
            return layer.updateAndAttend(
                queries: MLXArray.ones([1, 2, 1, 64], dtype: .bfloat16),
                keys: MLXArray.ones([1, 1, 1, 64], dtype: .bfloat16),
                values: MLXArray.full([1, 1, 1, 64], values: MLXArray(value), dtype: .bfloat16),
                scale: 0.125, sinks: nil)
        }
        let donorOutput = decode(a, value: 3)
        let donorPlan = try #require(layer.segmentDispatchCache.prepared)
        eval(donorOutput)
        let donorBytes = a.snapshot().values.asData(access: .copy).data
        let adopterOutput = decode(b, value: 7)
        eval(adopterOutput)
        #expect(layer.segmentDispatchCache.prepared !== donorPlan)
        #expect(a.table.prefix(2) == b.table.prefix(2) && a.table.last != b.table.last)
        #expect(a.snapshot().values.asData(access: .copy).data == donorBytes)
        #expect(b.snapshot().values[0, 0, 32, 0].item(Float.self) == 7)
        layer.setRows([])
    }
}

private final class SegmentMetadataEngineModel: CBv2SteppableModel {
    let inner = TinyTestModel.make(seed: 0xCA55, headDim: 64, fullAttentionOnly: true)
    let clearEachForward: Bool
    var shapes: [[Int]] = []
    init(clearEachForward: Bool) {
        self.clearEachForward = clearEachForward
        inner.update(parameters: ModuleParameters.unflattened(
            inner.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        eval(inner)
    }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        shapes.append(tokens.shape)
        if clearEachForward {
            for cache in caches { (cache as? PagedLayerCache)?.segmentDispatchCache.clear() }
        }
        return inner.forward(tokens: tokens, caches: caches)
    }
}

final class CBv2PagedSegmentDispatchCacheEngineTests: XCTestCase {
    private func run(clear: Bool, direct: Bool) async throws -> (tokens: [Int], shapes: [[Int]], hits: Int) {
        let model = SegmentMetadataEngineModel(clearEachForward: clear)
        let kinds = model.inner.layerKinds
        let backend = try PagedKVBackend(layerKinds: kinds, config: .init(
            capacityBytes: 1 << 24, dtype: .bfloat16, maxPrefillChunk: 8, segmentSizeBytes: 1 << 18))
        let caches = backend.makeLayerCaches()
        for cache in caches { (cache as? PagedLayerCache)?.segmentDispatchCache.statistics = .init() }
        let engine = EngineV2(model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: caches), sampler: CBv2GreedySampler(),
            schedulerConfig: .init(maxConcurrentRequests: 1, maxBatchedTokensPerStep: 64,
                                  prefillChunkSize: 8, maxWaiting: 4, enablePrefixCache: false))
        do {
            let result = await cbv2SchedCollect(try engine.submit(.init(
                id: .init(2), promptTokens: makePromptTokens(length: 29, seed: 71),
                sampling: .init(temperature: 0), maxTokens: 8, prefixCacheEnabled: false,
                tokenConstraint: direct ? AttentionPacketAllTokens() : nil)))
            let idle = await cbv2SchedWait {
                do { _ = try engine.takeAttentionMetadataSnapshot(); return true }
                catch { return false }
            }
            XCTAssertTrue(idle)
            await engine.shutdown()
            XCTAssertEqual(backend.bytesWired, 0)
            XCTAssertTrue(caches.allSatisfy { ($0 as? PagedLayerCache)?.segmentDispatchCache.prepared == nil })
            return (result.tokens, model.shapes, caches.reduce(0) {
                $0 + (($1 as? PagedLayerCache)?.segmentDispatchCache.statistics?.hits ?? 0)
            })
        } catch {
            await engine.shutdown()
            throw error
        }
    }

    func testActualDirectAndChainedEngineMatchesFreshPlanTrajectory() async throws {
        for direct in [true, false] {
            let fresh = try await run(clear: true, direct: direct)
            let cached = try await run(clear: false, direct: direct)
            XCTAssertEqual(cached.tokens.count, 8)
            XCTAssertEqual(cached.tokens, fresh.tokens)
            XCTAssertEqual(cached.shapes, fresh.shapes)
            XCTAssertEqual(fresh.hits, 0)
            XCTAssertGreaterThan(cached.hits, 0)
        }
    }
}
