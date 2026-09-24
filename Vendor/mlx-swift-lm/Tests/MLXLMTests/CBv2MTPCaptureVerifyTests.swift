// CBv2MTPCaptureVerifyTests.swift
//
// MTP capture-verify (MTPLX GDN capture-commit pattern) — targeted tests:
//  1. Captured recurrent transaction lifecycle (stageCaptured /
//     commit(keepPositions:) / rollback / bind fencing).
//  2. Qwen35GatedDeltaNet captured-window forward equivalence against the
//     serial per-column oracle, for both the rejected (keep 1) and accepted
//     (keep full window) commit paths.
//  3. Compact legacy-cache prefix replay across all accepted boundaries,
//     including fallback and terminal tape cleanup.
//  4. Default-sampler MTP verify pre-sampling reproduces the ordinary
//     per-token draw (greedy bitwise; stochastic same keyed RNG stream).

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class CBv2MTPCaptureVerifyTests: XCTestCase {

    // MARK: - 1. Captured transaction lifecycle

    private var scalarSpec: CBv2RecurrentStateSpec {
        CBv2RecurrentStateSpec(layers: [
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 0,
                convShape: [1, 2, 3], convDType: .float32,
                ssmShape: [1, 1, 2, 2], ssmDType: .float32)
        ])
    }

    private func capturedStacks(positions: Int, base: Float) -> (conv: MLXArray, ssm: MLXArray) {
        let conv = concatenated(
            (0 ..< positions).map {
                MLXArray.full([1, 2, 3], values: MLXArray(base + Float($0 + 1)))
            }, axis: 0)
        let ssm = concatenated(
            (0 ..< positions).map {
                MLXArray.full([1, 1, 2, 2], values: MLXArray(base + Float($0 + 1)))
            }, axis: 0)
        return (conv, ssm)
    }

    func testCapturedCommitSelectsPositionAndRollbackRestores() throws {
        let state = try CBv2RecurrentRequestState(spec: scalarSpec)

        // Commit a plain baseline generation with value 10.
        let baseline = try state.bind()
        try baseline.stage(
            modelLayerIndex: 0,
            conv: MLXArray.full([1, 2, 3], values: MLXArray(Float(10))),
            ssm: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(10))))
        _ = try baseline.evaluate()
        try baseline.commit()

        // Captured window of 3 positions (values 11, 12, 13); commit keep=2.
        let captured = try state.bind()
        let stacks = capturedStacks(positions: 3, base: 10)
        try captured.stageCaptured(
            modelLayerIndex: 0, conv: stacks.conv, ssm: stacks.ssm, positions: 3)
        XCTAssertTrue(captured.isCaptured)
        _ = try captured.evaluate()
        try captured.commit(keepPositions: 2)

        let afterCommit = state.state(modelLayerIndex: 0)!
        XCTAssertEqual(afterCommit.conv!.shape, [1, 2, 3])
        XCTAssertEqual(afterCommit.ssm!.shape, [1, 1, 2, 2])
        XCTAssertEqual(afterCommit.conv![0, 0, 0].item(Float.self), 12)
        XCTAssertEqual(afterCommit.ssm![0, 0, 0, 0].item(Float.self), 12)

        // Rollback of a captured window restores the committed state.
        let rejected = try state.bind()
        let rejectedStacks = capturedStacks(positions: 2, base: 12)
        try rejected.stageCaptured(
            modelLayerIndex: 0, conv: rejectedStacks.conv, ssm: rejectedStacks.ssm,
            positions: 2)
        _ = try rejected.evaluate()
        try rejected.rollback()
        let afterRollback = state.state(modelLayerIndex: 0)!
        XCTAssertEqual(afterRollback.conv![0, 0, 0].item(Float.self), 12)
        XCTAssertEqual(afterRollback.ssm![0, 0, 0, 0].item(Float.self), 12)
    }

    func testCapturedLifecycleFencing() throws {
        let state = try CBv2RecurrentRequestState(spec: scalarSpec)
        let captured = try state.bind()
        let stacks = capturedStacks(positions: 2, base: 0)

        // Plain stage after a captured stage is a violation.
        try captured.stageCaptured(
            modelLayerIndex: 0, conv: stacks.conv, ssm: stacks.ssm, positions: 2)
        XCTAssertThrowsError(
            try captured.stage(
                modelLayerIndex: 0,
                conv: MLXArray.zeros([1, 2, 3]),
                ssm: MLXArray.zeros([1, 1, 2, 2])))

        _ = try captured.evaluate()

        // A chained bind over an unresolved captured window must fail loud:
        // its per-position stacks are not a readable input state.
        XCTAssertThrowsError(try state.bind())

        // Plain commit on a captured generation is a violation; out-of-range
        // keeps are violations; a valid keep succeeds.
        XCTAssertThrowsError(try captured.commit())
        XCTAssertThrowsError(try captured.commit(keepPositions: 3))
        try captured.commit(keepPositions: 1)
        XCTAssertEqual(state.state(modelLayerIndex: 0)!.conv![0, 0, 0].item(Float.self), 1)
    }

    func testCapturedWindowResidencyChargesPerPosition() throws {
        // The bare-loop capacity gauge derives from materializedByteCount:
        // a captured verify window of P positions holds P full conv/SSM
        // copies, so it must be charged P× — not as one generation.
        let state = try CBv2RecurrentRequestState(spec: scalarSpec)
        XCTAssertEqual(state.materializedByteCount, 0)

        let baseline = try state.bind()
        try baseline.stage(
            modelLayerIndex: 0,
            conv: MLXArray.full([1, 2, 3], values: MLXArray(Float(10))),
            ssm: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(10))))
        _ = try baseline.evaluate()
        try baseline.commit()
        XCTAssertEqual(state.materializedByteCount, state.byteCount)

        // Committed state + captured window of 2 positions = 3 copies.
        let captured = try state.bind()
        let stacks = capturedStacks(positions: 2, base: 10)
        try captured.stageCaptured(
            modelLayerIndex: 0, conv: stacks.conv, ssm: stacks.ssm, positions: 2)
        _ = try captured.evaluate()
        XCTAssertEqual(state.materializedByteCount, 3 * state.byteCount)

        // The committed prefix is a view retaining both captured positions.
        try captured.commit(keepPositions: 1)
        XCTAssertEqual(state.materializedByteCount, 2 * state.byteCount)
    }

    func testCompactReplayStrictPrefixFullCommitAndRollback() throws {
        func makeState() throws -> CBv2RecurrentRequestState {
            let state = try CBv2RecurrentRequestState(spec: scalarSpec)
            let baseline = try state.bind()
            try baseline.stage(
                modelLayerIndex: 0,
                conv: MLXArray.full([1, 2, 3], values: MLXArray(Float(10))),
                ssm: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(10))))
            _ = try baseline.evaluate()
            try baseline.commit()
            return state
        }

        let strictState = try makeState()
        let strict = try strictState.bind()
        var strictReplayKeeps: [Int] = []
        let root = MLXArray(Float(123))
        try strict.stagePrefixReplay(
            modelLayerIndex: 0, positions: 4,
            finalConv: MLXArray.full([1, 2, 3], values: MLXArray(Float(14))),
            finalSSM: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(14))),
            materializedByteCount: 37, evaluationRoots: [root],
            strictReplayRetainedByteCount: 17,
            strictReplayRetainedRoots: [root]
        ) { keep in
            strictReplayKeeps.append(keep)
            return CBv2RecurrentLayerState(
                conv: MLXArray.full([1, 2, 3], values: MLXArray(Float(10 + keep))),
                ssm: MLXArray.full(
                    [1, 1, 2, 2], values: MLXArray(Float(10 + keep))))
        }
        XCTAssertTrue(strict.isCaptured)
        let roots = try strict.evaluate()
        XCTAssertEqual(roots.count, 3)
        XCTAssertEqual(roots.last!.item(Float.self), 123)
        XCTAssertEqual(strictState.materializedByteCount, strictState.byteCount + 37)
        XCTAssertThrowsError(try strictState.bind())
        try strict.commit(keepPositions: 2)
        XCTAssertEqual(strictReplayKeeps, [2])
        XCTAssertEqual(
            strictState.state(modelLayerIndex: 0)!.ssm![0, 0, 0, 0].item(Float.self), 12)
        XCTAssertEqual(
            strictState.materializedByteCount, strictState.byteCount + 17)
        let settled = try strictState.bind()
        let strictCommitted = strictState.state(modelLayerIndex: 0)!
        try settled.stage(
            modelLayerIndex: 0,
            conv: try XCTUnwrap(strictCommitted.conv),
            ssm: try XCTUnwrap(strictCommitted.ssm))
        eval(try settled.evaluate())
        try settled.commit()
        XCTAssertEqual(strictState.materializedByteCount, strictState.byteCount)

        let fullState = try makeState()
        let full = try fullState.bind()
        let fullRetentionRoot = MLXArray(Float(456))
        var fullMaterializationCount = 0
        var fullReplayCount = 0
        try full.stagePrefixReplay(
            modelLayerIndex: 0, positions: 4,
            finalConv: MLXArray.full([1, 2, 3], values: MLXArray(Float(99))),
            finalSSM: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(99))),
            materializedByteCount: 41, evaluationRoots: [],
            strictReplayRetainedByteCount: 0,
            strictReplayRetainedRoots: [],
            fullAcceptanceRetainedByteCount: 19,
            fullAcceptanceRetainedRoots: [fullRetentionRoot],
            fullAcceptance: {
                fullMaterializationCount += 1
                return CBv2RecurrentLayerState(
                    conv: MLXArray.full([1, 2, 3], values: MLXArray(Float(14))),
                    ssm: MLXArray.full(
                        [1, 1, 2, 2], values: MLXArray(Float(14))))
            }
        ) { _ in
            fullReplayCount += 1
            return CBv2RecurrentLayerState(conv: nil, ssm: nil)
        }
        _ = try full.evaluate()
        XCTAssertEqual(fullMaterializationCount, 0)
        try full.commit(keepPositions: 4)
        XCTAssertEqual(fullMaterializationCount, 1)
        XCTAssertEqual(fullReplayCount, 0)
        XCTAssertEqual(
            fullState.state(modelLayerIndex: 0)!.conv![0, 0, 0].item(Float.self), 14)
        XCTAssertEqual(fullState.materializedByteCount, fullState.byteCount + 19)
        let fullSettled = try fullState.bind()
        let fullCommitted = fullState.state(modelLayerIndex: 0)!
        try fullSettled.stage(
            modelLayerIndex: 0,
            conv: try XCTUnwrap(fullCommitted.conv),
            ssm: try XCTUnwrap(fullCommitted.ssm))
        eval(try fullSettled.evaluate())
        try fullSettled.commit()
        XCTAssertEqual(fullState.materializedByteCount, fullState.byteCount)

        let rolledBackState = try makeState()
        let rolledBack = try rolledBackState.bind()
        var rollbackFullAcceptanceCount = 0
        var rollbackReplayCount = 0
        try rolledBack.stagePrefixReplay(
            modelLayerIndex: 0, positions: 3,
            finalConv: MLXArray.full([1, 2, 3], values: MLXArray(Float(13))),
            finalSSM: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(13))),
            materializedByteCount: 29, evaluationRoots: [],
            strictReplayRetainedByteCount: 0,
            strictReplayRetainedRoots: [],
            fullAcceptance: {
                rollbackFullAcceptanceCount += 1
                return CBv2RecurrentLayerState(conv: nil, ssm: nil)
            }
        ) { _ in
            rollbackReplayCount += 1
            return CBv2RecurrentLayerState(conv: nil, ssm: nil)
        }
        _ = try rolledBack.evaluate()
        try rolledBack.rollback()
        XCTAssertEqual(rollbackFullAcceptanceCount, 0)
        XCTAssertEqual(rollbackReplayCount, 0)
        XCTAssertEqual(
            rolledBackState.state(modelLayerIndex: 0)!.conv![0, 0, 0].item(Float.self), 10)
    }

    func testCompactReplayRejectsMixedModesLayersAndWidths() throws {
        let twoLayerSpec = CBv2RecurrentStateSpec(layers: [
            scalarSpec.layers[0],
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 1,
                convShape: [1, 2, 3], convDType: .float32,
                ssmShape: [1, 1, 2, 2], ssmDType: .float32),
        ])
        let state = try CBv2RecurrentRequestState(spec: twoLayerSpec)
        let transaction = try state.bind()
        func stageReplay(layer: Int, positions: Int) throws {
            try transaction.stagePrefixReplay(
                modelLayerIndex: layer, positions: positions,
                finalConv: MLXArray.zeros([1, 2, 3]),
                finalSSM: MLXArray.zeros([1, 1, 2, 2]),
                materializedByteCount: 1, evaluationRoots: [],
                strictReplayRetainedByteCount: 0,
                strictReplayRetainedRoots: []
            ) { _ in
                CBv2RecurrentLayerState(
                    conv: MLXArray.zeros([1, 2, 3]),
                    ssm: MLXArray.zeros([1, 1, 2, 2]))
            }
        }
        try stageReplay(layer: 0, positions: 4)
        XCTAssertThrowsError(
            try transaction.stage(
                modelLayerIndex: 1, conv: MLXArray.zeros([1, 2, 3]),
                ssm: MLXArray.zeros([1, 1, 2, 2])))
        XCTAssertThrowsError(
            try transaction.stageCaptured(
                modelLayerIndex: 1, conv: MLXArray.zeros([4, 2, 3]),
                ssm: MLXArray.zeros([4, 1, 2, 2]), positions: 4))
        XCTAssertThrowsError(try stageReplay(layer: 1, positions: 3))
        try stageReplay(layer: 1, positions: 4)
        _ = try transaction.evaluate()
        XCTAssertThrowsError(try transaction.commit())
        try transaction.commit(keepPositions: 4)
    }

    // MARK: - 2. GDN captured-window equivalence

    private func smallGDNConfiguration() throws -> Qwen35TextConfiguration {
        // linear_key_head_dim must be >= 32: the gated-delta Metal kernel
        // tiles the state as Dk/32 floats per thread.
        let json = Data(
            """
            {
              "model_type": "qwen3_5_moe",
              "hidden_size": 16,
              "num_hidden_layers": 4,
              "intermediate_size": 32,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "linear_num_value_heads": 4,
              "linear_num_key_heads": 2,
              "linear_key_head_dim": 32,
              "linear_value_head_dim": 16,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 32,
              "head_dim": 8,
              "full_attention_interval": 4,
              "num_experts": 0,
              "num_experts_per_tok": 0
            }
            """.utf8)
        return try JSONDecoder.json5().decode(Qwen35TextConfiguration.self, from: json)
    }

    private func exactW4G64GDNConfiguration() throws -> Qwen35TextConfiguration {
        let json = Data(
            """
            {
              "model_type": "qwen3_5_moe",
              "hidden_size": 512,
              "num_hidden_layers": 4,
              "intermediate_size": 512,
              "num_attention_heads": 8,
              "num_key_value_heads": 1,
              "linear_num_value_heads": 16,
              "linear_num_key_heads": 2,
              "linear_key_head_dim": 32,
              "linear_value_head_dim": 32,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 512,
              "head_dim": 64,
              "full_attention_interval": 4,
              "num_experts": 0,
              "num_experts_per_tok": 0
            }
            """.utf8)
        return try JSONDecoder.json5().decode(Qwen35TextConfiguration.self, from: json)
    }

    private func gdnSpec(_ config: Qwen35TextConfiguration) -> CBv2RecurrentStateSpec {
        let keyDim = config.linearNumKeyHeads * config.linearKeyHeadDim
        let valueDim = config.linearNumValueHeads * config.linearValueHeadDim
        return CBv2RecurrentStateSpec(layers: [
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 0,
                convShape: [1, config.linearConvKernelDim - 1, 2 * keyDim + valueDim],
                convDType: .float32,
                ssmShape: [
                    1, config.linearNumValueHeads, config.linearValueHeadDim,
                    config.linearKeyHeadDim,
                ],
                ssmDType: .float32)
        ])
    }

    /// After a REJECTED draft the captured commit at keep=1 must reproduce
    /// the state of a control that never speculated (only consumed the seed
    /// column); partial and full acceptance must match controls that decoded
    /// exactly that many tokens serially. Outputs
    /// use the installed W4/g64 exact projection geometry.
    func testExactGDNCapturedWindowMatchesSerialOracle() throws {
        func assertExact(_ actual: MLXArray, _ expected: MLXArray) {
            let difference = max(abs(
                actual.asType(.float32) - expected.asType(.float32)))
            eval(difference)
            XCTAssertEqual(difference.item(Float.self), 0)
        }

        MLXRandom.seed(4711)
        let config = try exactW4G64GDNConfiguration()
        let layer = Qwen35GatedDeltaNet(config)
        quantize(model: layer, groupSize: 64, bits: 4)
        eval(layer)

        let hidden = config.hiddenSize
        MLXRandom.seed(90210)
        let x0 = MLXRandom.normal([1, 1, hidden])
        let x1 = MLXRandom.normal([1, 1, hidden])
        let x2 = MLXRandom.normal([1, 1, hidden])
        let window = concatenated([x0, x1, x2], axis: 1)

        // Warm history so the states are non-trivial before the window.
        let history = MLXRandom.normal([1, 3, hidden])

        func freshState() throws -> CBv2RecurrentRequestState {
            let state = try CBv2RecurrentRequestState(spec: gdnSpec(config))
            let warm = try state.bind()
            _ = layer.cbv2Forward(history, modelLayerIndex: 0, recurrentState: [warm])
            _ = try warm.evaluate()
            try warm.commit()
            return state
        }

        // Serial oracle: three [1, 1] columns through plain transactions.
        let serialState = try freshState()
        let serial0 = try serialState.bind()
        let y0 = layer.cbv2Forward(x0, modelLayerIndex: 0, recurrentState: [serial0])
        _ = try serial0.evaluate()
        try serial0.commit()
        let afterSeed = serialState.state(modelLayerIndex: 0)!
        let afterSeedConv = afterSeed.conv! + 0
        let afterSeedSSM = afterSeed.ssm! + 0
        let serial1 = try serialState.bind()
        let y1 = layer.cbv2Forward(x1, modelLayerIndex: 0, recurrentState: [serial1])
        _ = try serial1.evaluate()
        try serial1.commit()
        let afterFirstDraft = serialState.state(modelLayerIndex: 0)!
        let afterFirstDraftConv = afterFirstDraft.conv! + 0
        let afterFirstDraftSSM = afterFirstDraft.ssm! + 0
        let serial2 = try serialState.bind()
        let y2 = layer.cbv2Forward(x2, modelLayerIndex: 0, recurrentState: [serial2])
        _ = try serial2.evaluate()
        try serial2.commit()
        let afterDraft = serialState.state(modelLayerIndex: 0)!
        eval(y0, y1, y2, afterSeedConv, afterSeedSSM, afterDraft.conv!, afterDraft.ssm!)

        // Rejected draft: captured window, keep 1 → state == control that
        // never consumed the draft.
        let rejectedState = try freshState()
        let rejected = try rejectedState.bind()
        let rejectedOut = layer.cbv2ForwardCaptured(
            window, modelLayerIndex: 0, recurrentState: [rejected],
            exactTargetVerify: true)
        XCTAssertTrue(rejected.isCaptured)
        _ = try rejected.evaluate()
        eval(rejectedOut)
        try rejected.commit(keepPositions: 1)
        let rejectedFinal = rejectedState.state(modelLayerIndex: 0)!
        XCTAssertEqual(rejectedFinal.conv!.shape, afterSeedConv.shape)
        XCTAssertEqual(rejectedFinal.ssm!.shape, afterSeedSSM.shape)
        assertExact(rejectedFinal.conv!, afterSeedConv)
        assertExact(rejectedFinal.ssm!, afterSeedSSM)

        // Partial acceptance exercises the strict-prefix replay boundary
        // between total rejection and full acceptance with the exact oracle.
        let partialState = try freshState()
        let partial = try partialState.bind()
        let partialOut = layer.cbv2ForwardCaptured(
            window, modelLayerIndex: 0, recurrentState: [partial],
            exactTargetVerify: true)
        eval(try partial.evaluate())
        try partial.commit(keepPositions: 2)
        let partialFinal = partialState.state(modelLayerIndex: 0)!
        assertExact(partialFinal.conv!, afterFirstDraftConv)
        assertExact(partialFinal.ssm!, afterFirstDraftSSM)

        // Accepted draft: captured window, keep 3 → state == control that
        // decoded all three tokens serially; window outputs match the serial
        // column outputs.
        let acceptedState = try freshState()
        let accepted = try acceptedState.bind()
        let acceptedOut = layer.cbv2ForwardCaptured(
            window, modelLayerIndex: 0, recurrentState: [accepted],
            exactTargetVerify: true)
        _ = try accepted.evaluate()
        eval(acceptedOut)
        try accepted.commit(keepPositions: 3)
        let acceptedFinal = acceptedState.state(modelLayerIndex: 0)!
        assertExact(acceptedFinal.conv!, afterDraft.conv!)
        assertExact(acceptedFinal.ssm!, afterDraft.ssm!)
        XCTAssertTrue(acceptedOut.shape == [1, 3, hidden])
        assertExact(acceptedOut, concatenated([y0, y1, y2], axis: 1))
        for value in [rejectedOut, partialOut, acceptedOut] {
            XCTAssertTrue(isFinite(value).all().item(Bool.self))
        }
    }

    /// Independent recurrence oracle: consumes each already-projected verify
    /// row once, preserving a snapshot at every prefix. It deliberately does
    /// not use the production prefix-replay helper or its captured closure.
    private func tokenwiseRecurrenceOracle(
        tape: ArraysCache.PrefixReplayTape, layer: Qwen35GatedDeltaNet
    ) throws -> [(conv: MLXArray, ssm: MLXArray)] {
        var conv = tape.convInput[0..., 0..<tape.convStateRows, 0...]
        var ssm = try XCTUnwrap(tape.ssmPre)
        var snapshots: [(conv: MLXArray, ssm: MLXArray)] = []
        for row in 0..<tape.rowCount {
            let next = row..<(row + 1)
            ssm = gatedDeltaUpdate(
                q: tape.q[0..., next], k: tape.k[0..., next], v: tape.v[0..., next],
                a: tape.a[0..., next], b: tape.b[0..., next],
                aLog: layer.aLog, dtBias: layer.dtBias, state: ssm,
                mask: tape.mask.map { $0[0..., next] }).1
            let incoming = tape.convStateRows + row
            conv = concatenated([
                conv[0..., 1..., 0...],
                tape.convInput[0..., incoming..<(incoming + 1), 0...],
            ], axis: 1)
            let snapshot = (conv: conv + 0, ssm: ssm + 0)
            eval(snapshot.conv, snapshot.ssm)
            snapshots.append(snapshot)
        }
        return snapshots
    }

    func testGDNWideWindowUsesCompactReplayAndMatchesEveryRecurrencePrefix() throws {
        MLXRandom.seed(0xC0FFEE)
        let config = try smallGDNConfiguration()
        let layer = Qwen35GatedDeltaNet(config)
        eval(layer)
        let hidden = config.hiddenSize
        let columns = (0 ..< 4).map { _ in MLXRandom.normal([1, 1, hidden]) }
        let window = concatenated(columns, axis: 1)
        let history = MLXRandom.normal([1, 3, hidden])

        func freshState() throws -> CBv2RecurrentRequestState {
            let state = try CBv2RecurrentRequestState(spec: gdnSpec(config))
            let warm = try state.bind()
            _ = layer.cbv2Forward(history, modelLayerIndex: 0, recurrentState: [warm])
            eval(try warm.evaluate())
            try warm.commit()
            return state
        }

        // Default rectangular arithmetic need not reproduce width-one dense
        // projection rounding. Obtain a separate same-width tape, then compute
        // each expected boundary independently one recurrence step at a time.
        let initial = try freshState().state(modelLayerIndex: 0)!
        let legacy = MambaCache()
        legacy[0] = initial.conv
        legacy[1] = initial.ssm
        let legacyOutput = layer(window, cache: legacy, nConfirmed: 1)
        eval([legacyOutput] + legacy.innerState())
        let tape = try XCTUnwrap(legacy.prefixReplayTape)
        let oracle = try tokenwiseRecurrenceOracle(tape: tape, layer: layer)

        for keep in 1 ... columns.count {
            let compactState = try freshState()
            let compact = try compactState.bind()
            _ = layer.cbv2ForwardCaptured(
                window, modelLayerIndex: 0, recurrentState: [compact])
            eval(try compact.evaluate())
            XCTAssertTrue(compact.isCaptured)
            XCTAssertLessThan(
                compactState.materializedByteCount,
                (columns.count + 1) * compactState.byteCount,
                "wide production verify must not materialize a full state stack")
            try compact.commit(keepPositions: keep)
            let actual = compactState.state(modelLayerIndex: 0)!
            XCTAssertTrue(
                allClose(actual.conv!, oracle[keep - 1].conv, rtol: 1e-4, atol: 1e-5)
                    .item(Bool.self))
            XCTAssertTrue(
                allClose(actual.ssm!, oracle[keep - 1].ssm, rtol: 1e-4, atol: 1e-5)
                    .item(Bool.self))
        }
    }

    /// Same-width control isolates captured bookkeeping from recomputing
    /// projections and convolution with a different forward width.
    func testGDNWideCapturedMatchesOrdinaryWideFromMaterializedState() throws {
        MLXRandom.seed(0xC0FFEE)
        let config = try smallGDNConfiguration()
        let layer = Qwen35GatedDeltaNet(config)
        eval(layer)
        let columns = (0 ..< 4).map { _ in MLXRandom.normal([1, 1, config.hiddenSize]) }
        let window = concatenated(columns, axis: 1)
        let history = MLXRandom.normal([1, 3, config.hiddenSize])
        eval(window, history)

        func freshState() throws -> CBv2RecurrentRequestState {
            let state = try CBv2RecurrentRequestState(spec: gdnSpec(config))
            let warm = try state.bind()
            _ = layer.cbv2Forward(history, modelLayerIndex: 0, recurrentState: [warm])
            eval(try warm.evaluate())
            try warm.commit()
            return state
        }
        let ordinaryState = try freshState()
        let ordinary = try ordinaryState.bind()
        _ = layer.cbv2Forward(window, modelLayerIndex: 0, recurrentState: [ordinary])
        eval(try ordinary.evaluate())
        try ordinary.commit()

        let capturedState = try freshState()
        let captured = try capturedState.bind()
        _ = layer.cbv2ForwardCaptured(window, modelLayerIndex: 0, recurrentState: [captured])
        eval(try captured.evaluate())
        try captured.commit(keepPositions: 4)
        let actual = capturedState.state(modelLayerIndex: 0)!
        let expected = ordinaryState.state(modelLayerIndex: 0)!
        for (name, left, right) in [("conv", actual.conv!, expected.conv!),
                                  ("ssm", actual.ssm!, expected.ssm!)] {
            XCTAssertTrue(allClose(left, right, rtol: 1e-4, atol: 1e-5).item(Bool.self),
                "same-width captured \(name) must match ordinary forward")
        }
    }

    func testProductionWidthCompactReplayFitsFourGenerationContinuationCharge() throws {
        let json = Data(
            """
            {
              "model_type": "qwen3_5_moe",
              "hidden_size": 16,
              "num_hidden_layers": 4,
              "intermediate_size": 32,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "linear_num_value_heads": 32,
              "linear_num_key_heads": 16,
              "linear_key_head_dim": 128,
              "linear_value_head_dim": 128,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 32,
              "head_dim": 8,
              "full_attention_interval": 4,
              "num_experts": 0,
              "num_experts_per_tok": 0
            }
            """.utf8)
        let config = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: json)
        let layer = Qwen35GatedDeltaNet(config)
        eval(layer)
        let spec = gdnSpec(config)
        let state = try CBv2RecurrentRequestState(spec: spec)

        let warm = try state.bind()
        _ = layer.cbv2Forward(
            MLXRandom.normal([1, 3, config.hiddenSize]),
            modelLayerIndex: 0, recurrentState: [warm])
        _ = try warm.evaluate()
        try warm.commit()

        let verify = try state.bind()
        _ = layer.cbv2ForwardCaptured(
            MLXRandom.normal([1, 5, config.hiddenSize]),
            modelLayerIndex: 0, recurrentState: [verify])
        _ = try verify.evaluate()

        let fixedThreeGenerations = try spec.peakBytesPerRequest()
        XCTAssertEqual(fixedThreeGenerations, 3 * state.byteCount)
        XCTAssertLessThanOrEqual(
            state.materializedByteCount, fixedThreeGenerations,
            "the compact verify itself must fit the base recurrent charge")

        // A strict-prefix commit retains its replay tape until a successor
        // generation commits. That reachable transition exceeds 3F but must
        // fit the compact driver's resolved 4F admission reservation.
        try verify.commit(keepPositions: 3)
        let successor = try state.bind()
        _ = layer.cbv2Forward(
            MLXRandom.normal([1, 1, config.hiddenSize]),
            modelLayerIndex: 0, recurrentState: [successor])
        eval(try successor.evaluate())
        let fixedFourGenerations = 4 * state.byteCount
        XCTAssertGreaterThan(state.materializedByteCount, fixedThreeGenerations)
        XCTAssertLessThanOrEqual(
            state.materializedByteCount, fixedFourGenerations,
            "strict-prefix continuation must fit resolved compact 4F headroom")
        try successor.commit()
        XCTAssertEqual(state.materializedByteCount, state.byteCount)

        // Full acceptance defers the exact conv-tail copy until commit. Its
        // lazy graph keeps the verify-window backing charged until the next
        // evaluated generation provides the release fence.
        let fullVerify = try state.bind()
        _ = layer.cbv2ForwardCaptured(
            MLXRandom.normal([1, 5, config.hiddenSize]),
            modelLayerIndex: 0, recurrentState: [fullVerify])
        eval(try fullVerify.evaluate())
        try fullVerify.commit(keepPositions: 5)
        XCTAssertGreaterThan(state.materializedByteCount, state.byteCount)
        XCTAssertLessThanOrEqual(state.materializedByteCount, fixedThreeGenerations)

        let fullSuccessor = try state.bind()
        _ = layer.cbv2Forward(
            MLXRandom.normal([1, 1, config.hiddenSize]),
            modelLayerIndex: 0, recurrentState: [fullSuccessor])
        eval(try fullSuccessor.evaluate())
        XCTAssertLessThanOrEqual(
            state.materializedByteCount, fixedFourGenerations)
        try fullSuccessor.commit()
        XCTAssertEqual(state.materializedByteCount, state.byteCount)
    }

    // MARK: - 3. Compact recurrent prefix replay

    private func legacyCaches(_ model: Qwen35TextModel) -> [any KVCache] {
        model.newCache(parameters: nil).map { $0 as any KVCache }
    }

    private func copiedCaches(_ caches: [any KVCache]) -> [any KVCache] {
        caches.map { $0.copy() }
    }

    @discardableResult
    private func legacyForward(
        _ model: Qwen35TextModel,
        tokens: [Int32],
        caches: [any KVCache],
        nConfirmed: Int
    ) -> MLXArray {
        let input = MLXArray(tokens).reshaped(1, tokens.count)
        let output = model.model(
            input, cache: caches.map { Optional($0) }, nConfirmed: nConfirmed)
        eval([output] + caches.flatMap { $0.innerState() })
        return output
    }

    private func assertRecurrentStateEqual(
        _ lhs: [any KVCache],
        _ rhs: [any KVCache],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.count, rhs.count, file: file, line: line)
        for index in lhs.indices {
            guard let left = lhs[index] as? MambaCache,
                  let right = rhs[index] as? MambaCache
            else { continue }
            XCTAssertEqual(left.state.count, right.state.count, file: file, line: line)
            for (leftArray, rightArray) in zip(left.state, right.state) {
                XCTAssertEqual(leftArray.shape, rightArray.shape, file: file, line: line)
                XCTAssertTrue(
                    allClose(leftArray, rightArray, rtol: 1e-5, atol: 1e-6)
                        .item(Bool.self),
                    file: file,
                    line: line)
            }
        }
    }

    /// A partially accepted wide verify rebuilds each recurrent layer from its
    /// compact tape, while attention remains at the speculative offset until
    /// the caller trims exactly the rejected suffix.
    func testRecurrentPrefixReplayMatchesRecurrencePrefixAndAttentionTrim() throws {
        MLXRandom.seed(61_337)
        let model = Qwen35TextModel(try smallGDNConfiguration())
        eval(model)

        let base = legacyCaches(model)
        _ = legacyForward(
            model, tokens: [1, 2, 3], caches: base, nConfirmed: 0)
        let baseAttentionOffsets = base.map(\.offset)
        let verifyTokens: [Int32] = [4, 5, 6, 7]

        // One accepted verify row is a total draft rejection; larger proper
        // prefixes exercise partial acceptance at every available boundary.
        for committedRows in 1 ..< verifyTokens.count {
            let verify = copiedCaches(base)
            let oracle = copiedCaches(base)
            _ = legacyForward(
                model, tokens: verifyTokens, caches: verify, nConfirmed: 1)
            for case let recurrent as MambaCache in verify {
                let tape = try XCTUnwrap(recurrent.prefixReplayTape)
                XCTAssertEqual(tape.rowCount, verifyTokens.count)
                XCTAssertNil(recurrent.rollbackState)
            }
            for index in verify.indices where !(verify[index] is ArraysCache) {
                XCTAssertEqual(
                    verify[index].offset,
                    baseAttentionOffsets[index] + verifyTokens.count)
            }

            var recurrentOracle: [Int: (conv: MLXArray, ssm: MLXArray)] = [:]
            for (index, cache) in verify.enumerated() {
                guard let recurrent = cache as? MambaCache else { continue }
                let tape = try XCTUnwrap(recurrent.prefixReplayTape)
                let layer = try XCTUnwrap(model.model.namedModules().first {
                    $0.0 == "layers.\(index).linear_attn"
                }?.1 as? Qwen35GatedDeltaNet)
                let prefixes = try tokenwiseRecurrenceOracle(tape: tape, layer: layer)
                recurrentOracle[index] = prefixes[committedRows - 1]
            }
            XCTAssertTrue(
                model.replayRecurrentPrefix(
                    cache: verify, committedRows: committedRows))
            for case let recurrent as MambaCache in verify {
                XCTAssertNil(recurrent.prefixReplayTape)
                XCTAssertNil(recurrent.rollbackState)
            }
            // Recurrent replay must not mutate attention bookkeeping.
            let rejectedRows = verifyTokens.count - committedRows
            for index in verify.indices where !(verify[index] is ArraysCache) {
                XCTAssertEqual(
                    verify[index].offset,
                    baseAttentionOffsets[index] + verifyTokens.count)
                XCTAssertEqual(verify[index].trim(rejectedRows), rejectedRows)
                XCTAssertEqual(
                    verify[index].offset,
                    baseAttentionOffsets[index] + committedRows)
            }

            _ = legacyForward(
                model,
                tokens: Array(verifyTokens.prefix(committedRows)),
                caches: oracle,
                nConfirmed: 0)
            // Ordinary prefix forwarding remains the attention-offset
            // control. Recurrent numerics use the same wide transformed inputs
            // as verification, avoiding an unsupported cross-width identity.
            for (index, expected) in recurrentOracle {
                let recurrent = try XCTUnwrap(oracle[index] as? MambaCache)
                recurrent[0] = expected.conv
                recurrent[1] = expected.ssm
            }
            assertRecurrentStateEqual(verify, oracle)
            for index in oracle.indices where !(oracle[index] is ArraysCache) {
                XCTAssertEqual(verify[index].offset, oracle[index].offset)
            }
        }
    }

    /// Shape-incompatible tapes fail closed before recurrent state mutation and
    /// release every layer's verify-only arrays for the fallback path.
    func testIneligibleRecurrentReplayCleansAllTapesWithoutStateMutation() throws {
        MLXRandom.seed(61_338)
        let model = Qwen35TextModel(try smallGDNConfiguration())
        eval(model)

        let caches = legacyCaches(model)
        _ = legacyForward(
            model, tokens: [1, 2, 3, 4], caches: caches, nConfirmed: 1)
        let recurrent = caches.compactMap { $0 as? MambaCache }
        let before = recurrent.map { $0.state.map { $0 + 0 } }
        eval(before.flatMap { $0 })

        // Corrupt the final recurrent layer so a mutate-as-it-walks replay
        // would damage earlier layers before discovering ineligibility.
        let invalid = try XCTUnwrap(recurrent.last)
        let tape = try XCTUnwrap(invalid.prefixReplayTape)
        invalid.prefixReplayTape = ArraysCache.PrefixReplayTape(
            convInput: tape.convInput,
            q: MLXArray.zeros([1]),
            k: tape.k,
            v: tape.v,
            a: tape.a,
            b: tape.b,
            ssmPre: tape.ssmPre,
            mask: tape.mask,
            rowCount: tape.rowCount,
            convStateRows: tape.convStateRows)

        XCTAssertFalse(
            model.replayRecurrentPrefix(cache: caches, committedRows: 2))
        for (index, cache) in recurrent.enumerated() {
            XCTAssertNil(cache.prefixReplayTape)
            XCTAssertNil(cache.rollbackState)
            for (actual, expected) in zip(cache.state, before[index]) {
                XCTAssertTrue(
                    allClose(actual, expected, rtol: 0, atol: 0)
                        .item(Bool.self))
            }
        }
    }

    /// Full acceptance explicitly discards the tape without touching state;
    /// an ordinary next forward and a direct cache reset also cannot retain it.
    func testRecurrentReplayTapeCleanupOnTerminalPaths() throws {
        MLXRandom.seed(61_339)
        let model = Qwen35TextModel(try smallGDNConfiguration())
        eval(model)

        let base = legacyCaches(model)
        let accepted = copiedCaches(base)
        _ = legacyForward(
            model, tokens: [1, 2, 3], caches: accepted, nConfirmed: 1)
        let acceptedState = accepted.compactMap { ($0 as? MambaCache)?.state.map { $0 + 0 } }
        eval(acceptedState.flatMap { $0 })
        model.clearRecurrentPrefixReplay(cache: accepted)
        for (index, cache) in accepted.compactMap({ $0 as? MambaCache }).enumerated() {
            XCTAssertNil(cache.prefixReplayTape)
            XCTAssertNil(cache.rollbackState)
            for (actual, expected) in zip(cache.state, acceptedState[index]) {
                XCTAssertTrue(
                    allClose(actual, expected, rtol: 0, atol: 0)
                        .item(Bool.self))
            }
        }

        let continued = copiedCaches(base)
        _ = legacyForward(
            model, tokens: [1, 2, 3], caches: continued, nConfirmed: 1)
        _ = legacyForward(
            model, tokens: [4], caches: continued, nConfirmed: 0)
        for case let cache as MambaCache in continued {
            XCTAssertNil(cache.prefixReplayTape)
            XCTAssertNil(cache.rollbackState)
        }

        let reset = copiedCaches(base)
        _ = legacyForward(
            model, tokens: [1, 2, 3], caches: reset, nConfirmed: 1)
        let resetFirst = try XCTUnwrap(reset.first)
        let resetCache = try XCTUnwrap(resetFirst as? MambaCache)
        XCTAssertNotNil(resetCache.prefixReplayTape)
        resetCache.rollbackState = (
            try XCTUnwrap(resetCache[0]),
            try XCTUnwrap(resetCache[1]))
        resetCache.state = []
        XCTAssertNil(resetCache.prefixReplayTape)
        XCTAssertNil(resetCache.rollbackState)
    }

    // MARK: - 4. Verify pre-sampling reproduces the ordinary draw

    func testVerifySampleMatchesOrdinaryDraws() throws {
        let vocab = 64
        let id = CBv2RequestID(7)
        let params = CBv2SamplingParams(
            temperature: 0.7, topP: 0.9, topK: 12, seed: 1234)
        let row = CBv2SamplerRow(
            id: id, params: params, promptTokens: [1, 2, 3], outputTokens: [])

        MLXRandom.seed(5)
        let logits0 = MLXRandom.normal([1, vocab])
        let logits1 = MLXRandom.normal([1, vocab])
        eval(logits0, logits1)

        // Ordinary path: two consecutive draws through the stateful sampler.
        let ordinary = CBv2DefaultSampler(fallbackSeed: 99)
        let draw0 = ordinary.sample(
            logits: logits0, params: [params], requestIDs: [id], stepIndex: 0,
            pendingSampledTokens: nil, rowContext: { [row] })
        let draw1 = ordinary.sample(
            logits: logits1, params: [params], requestIDs: [id], stepIndex: 1,
            pendingSampledTokens: nil, rowContext: { [row] })
        eval(draw0, draw1)

        // Verify pre-sampling: both positions in one [1, 2, vocab] window,
        // stepBase 0 (no tokens generated yet).
        let verifier = CBv2DefaultSampler(fallbackSeed: 99)
        XCTAssertTrue(verifier.supportsMTPTargetPrefix)
        let window = concatenated(
            [logits0.reshaped([1, 1, vocab]), logits1.reshaped([1, 1, vocab])], axis: 1)
        let sampled = try XCTUnwrap(
            verifier.mtpVerifySample(
                logits: window, params: [params], requestIDs: [id], stepBases: [0]))
        eval(sampled)
        XCTAssertEqual(sampled.shape, [1, 2])
        XCTAssertEqual(sampled[0, 0].item(Int32.self), draw0[0].item(Int32.self))
        XCTAssertEqual(sampled[0, 1].item(Int32.self), draw1[0].item(Int32.self))

        // Greedy rows go through the bit-identical argmax path.
        let greedy = CBv2SamplingParams(temperature: 0)
        let greedySampled = try XCTUnwrap(
            verifier.mtpVerifySample(
                logits: window, params: [greedy], requestIDs: [id], stepBases: [0]))
        let expected = argMax(window.reshaped([2, vocab]), axis: -1).asType(.int32)
        eval(greedySampled, expected)
        XCTAssertEqual(greedySampled[0, 0].item(Int32.self), expected[0].item(Int32.self))
        XCTAssertEqual(greedySampled[0, 1].item(Int32.self), expected[1].item(Int32.self))
    }
}
