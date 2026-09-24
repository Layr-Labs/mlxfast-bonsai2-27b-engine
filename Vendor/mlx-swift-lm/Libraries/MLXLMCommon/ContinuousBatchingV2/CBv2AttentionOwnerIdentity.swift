import Foundation

/// A bound cache may expose its original model index (contiguous production
/// adapters) or a dense storage index (paged). Neither convention is changed.
struct CBv2AttentionOwnerIdentity {
    let storageLayerIndex: Int
    let cacheLayerIndex: Int
    let kind: CBv2LayerKind

    func matches(_ cache: CBv2AttendingLayerCache) -> Bool {
        cache.layerIndex == cacheLayerIndex && cache.kind == kind
    }
}

extension CBv2AttentionMetadataForward {
    /// Register the actual object at its position in the engine's ordered
    /// layerKinds/cache arrays. Lookup never infers storage from layerIndex.
    func bindOwner(cache: CBv2AttendingLayerCache, storageLayerIndex: Int,
                   kind: CBv2LayerKind) -> Bool {
        let object = ObjectIdentifier(cache)
        let modelIndex = kind.modelLayerIndex ?? storageLayerIndex
        guard state.expectedOwners.contains(storageLayerIndex), cache.kind == kind,
            cache.layerIndex == storageLayerIndex || cache.layerIndex == modelIndex,
            boundOwners[object] == nil,
            !boundOwners.values.contains(where: { $0.storageLayerIndex == storageLayerIndex }) else {
            state.refuse("invalid_or_repeated_attention_owner_binding")
            return false
        }
        boundOwners[object] = .init(storageLayerIndex: storageLayerIndex,
                                    cacheLayerIndex: cache.layerIndex, kind: kind)
        return true
    }
}
