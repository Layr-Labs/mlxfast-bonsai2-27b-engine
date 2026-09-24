import Foundation

extension CBv2CompleteCheckpointManifest {
    private enum CodingKeys: String, CodingKey {
        case schemaVersion, identity, backendLayout, position, chunkSize
        case prefixTokens, cacheSalt, assistantCodecID, tensors, attentionLayers
    }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            schemaVersion: try values.decode(Int.self, forKey: .schemaVersion),
            identity: try values.decode(CBv2CompleteCheckpointIdentity.self, forKey: .identity),
            backendLayout: try values.decode(String.self, forKey: .backendLayout),
            position: try values.decode(Int.self, forKey: .position),
            chunkSize: try values.decode(Int.self, forKey: .chunkSize),
            cacheSalt: try values.decodeIfPresent(String.self, forKey: .cacheSalt),
            assistantCodecID: try values.decodeIfPresent(String.self, forKey: .assistantCodecID),
            metadata: .init(
                tokens: try values.decode([Int].self, forKey: .prefixTokens),
                tensors: try values.decode([CBv2CheckpointTensorDescriptor].self, forKey: .tensors),
                attentionLayers: try values.decodeIfPresent([CBv2CheckpointAttentionLayer].self, forKey: .attentionLayers)))
    }

    public func encode(to encoder: any Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(schemaVersion, forKey: .schemaVersion)
        try values.encode(identity, forKey: .identity)
        try values.encode(backendLayout, forKey: .backendLayout)
        try values.encode(position, forKey: .position)
        try values.encode(chunkSize, forKey: .chunkSize)
        try values.encode(prefixTokens, forKey: .prefixTokens)
        try values.encodeIfPresent(cacheSalt, forKey: .cacheSalt)
        try values.encodeIfPresent(assistantCodecID, forKey: .assistantCodecID)
        try values.encode(tensors, forKey: .tensors)
        try values.encodeIfPresent(attentionLayers, forKey: .attentionLayers)
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.schemaVersion == rhs.schemaVersion && lhs.identity == rhs.identity
            && lhs.backendLayout == rhs.backendLayout && lhs.position == rhs.position
            && lhs.chunkSize == rhs.chunkSize && lhs.prefixTokens == rhs.prefixTokens
            && lhs.cacheSalt == rhs.cacheSalt && lhs.assistantCodecID == rhs.assistantCodecID
            && lhs.tensors == rhs.tensors && lhs.attentionLayers == rhs.attentionLayers
    }
}
