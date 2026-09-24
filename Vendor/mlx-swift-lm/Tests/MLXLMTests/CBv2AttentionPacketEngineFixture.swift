import Foundation
import MLX
import MLXNN
import XCTest
@_spi(Diagnostics) @testable import MLXLMCommon

final class AttentionPacketEngineModel: CBv2RecurrentSteppableModel, CBv2MTPPolicyTopTwoProviding {
    let inner = TinyTestModel.make(seed: 0xA77E, headDim: 64, fullAttentionOnly: true)
    init(dtype: DType) {
        if dtype != .float32 {
            inner.update(parameters: ModuleParameters.unflattened(
                inner.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
            eval(inner)
        }
    }
    var cbv2Capabilities: CBv2ModelCapabilities {
        var result = CBv2ModelCapabilities.initialRecurrentTarget
        result.supportsPagedKV = true
        return result
    }
    let recurrentStateSpec: CBv2RecurrentStateSpec? = .init(layers: [
        .init(modelLayerIndex: 11, convShape: [1, 1, 1], convDType: .float32,
              ssmShape: [1, 1, 1, 1], ssmDType: .float32)
    ])
    var kinds: [CBv2LayerKind] {
        inner.layerKinds.enumerated().map { index, original in
            var kind = original
            kind.modelLayerIndex = 3 + index * 4
            return kind
        }
    }
    // Read only after the engine's idle queue barrier.
    var shapes: [[Int]] = []
    var staged = 0
    var packetForwards = 0
    var metadataForwards = 0
    var topTwoCalls = 0
    var forwardsAfterPacket = 0
    var successorObservedPendingPacket = false
    var lastCaches: [any CBv2AttendingLayerCache] = []
    weak var observedPacket: CBv2AttentionPacketForward?
    var selectedAction: (() -> Void)?

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        preconditionFailure("ordinary recurrent transaction must be preserved")
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache],
                 recurrentState: [CBv2RecurrentStateEvaluation]) -> MLXArray {
        let packet = caches.compactMap { ($0 as? any CBv2AttentionPacketBinding)?.attentionPacket }.first
        let metadata = caches.contains { ($0 as? any CBv2AttentionMetadataBinding)?.attentionMetadata != nil }
        if packetForwards > 0 && packet == nil {
            forwardsAfterPacket += 1
            if let previous = observedPacket, !previous.arrays.isEmpty {
                successorObservedPendingPacket = true
            }
        }
        shapes.append(tokens.shape)
        lastCaches = caches
        if let packet { packetForwards += 1; observedPacket = packet }
        if metadata { metadataForwards += 1 }
        let output = inner.forward(tokens: tokens, caches: caches)
        for evaluation in recurrentState {
            let previous = evaluation.inputState(modelLayerIndex: 11)
            try! evaluation.stage(modelLayerIndex: 11,
                conv: (previous?.conv ?? MLXArray.zeros([1, 1, 1])) + 1,
                ssm: (previous?.ssm ?? MLXArray.zeros([1, 1, 1, 1])) + 1)
            staged += 1
        }
        if packet != nil { selectedAction?() }
        return output
    }

    func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        topTwoCalls += 1
        let vocabulary = logits.dim(-1)
        let first = argMax(logits, axis: -1).asType(.int32)
        let vocabularyIDs = MLXArray(Int32(0)..<Int32(vocabulary)).reshaped([1, 1, vocabulary])
        let masked = MLX.where(vocabularyIDs .== first[0..., 0..., .newAxis],
                               MLXArray(-Float.infinity), logits)
        let second = argMax(masked, axis: -1).asType(.int32)
        let ids = stacked([first, second], axis: -1)
        return (ids, takeAlong(logits, ids, axis: -1))
    }
}

final class AttentionPacketAllTokens: CBv2TokenConstraint, @unchecked Sendable {
    let mode: CBv2TokenConstraintMode = .required
    let maxTokens = 8
    let fallbackTokenID = 0
    let initialState = 0
    func allowedTokenIDs(state: Int, remainingTokens: Int) -> [Int] { Array(0..<128) }
    func nextState(state: Int, tokenID: Int) -> Int? { (0..<128).contains(tokenID) ? 0 : nil }
}

struct AttentionPacketEngineRun {
    let tokens: [Int]
    let shapes: [[Int]]
    let staged: Int
    let packetForwards: Int
    let metadataForwards: Int
    let topTwoCalls: Int
    let forwardsAfterPacket: Int
    let successorObservedPendingPacket: Bool
    let packet: CBv2AttentionPacketSnapshot?
    let metadata: CBv2AttentionMetadataSnapshot?
    let logits: CBv2LogitDiagnosticSnapshot?
}
