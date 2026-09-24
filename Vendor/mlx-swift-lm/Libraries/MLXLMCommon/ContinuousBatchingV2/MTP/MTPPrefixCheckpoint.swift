import MLX

/// Immutable trusted assistant history at an exact target input boundary.
/// It never owns mutable draft KV or an unconfirmed speculative round.
public protocol CBv2MTPPrefixCheckpoint: AnyObject {
    var targetInputCount: Int { get }
    var materializedBytes: Int { get }
    var evaluationTargets: [MLXArray] { get }
}

/// Optional capability for assistants that can restore the same prompt history
/// and first-draft geometry as cold execution. Checkpoints stay model-owned.
public protocol CBv2MTPPrefixCheckpointDrafter: CBv2MTPRequestStatefulDrafter {
    func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount: Int
    ) -> (any CBv2MTPPrefixCheckpoint)?
    func restorePrefixCheckpoint(
        _ checkpoint: any CBv2MTPPrefixCheckpoint
    ) -> (any CBv2MTPRequestState)?
}
