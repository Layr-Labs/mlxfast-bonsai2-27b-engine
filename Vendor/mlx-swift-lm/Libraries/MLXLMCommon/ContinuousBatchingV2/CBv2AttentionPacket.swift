import Foundation

/// One native-dtype attention packet for offline analysis; never serving telemetry.
@_spi(Diagnostics)
public struct CBv2AttentionPacketConfig: Sendable, Encodable {
    public static let byteLimit = 32 * 1_024 * 1_024
    public let requestID: UInt64
    public let outputIndex: Int
    public let storageLayerIndex: Int
    public let maximumBytes: Int

    public init(requestID: UInt64, outputIndex: Int, storageLayerIndex: Int,
                maximumBytes: Int = Self.byteLimit) throws {
        guard requestID > 0, (1...1_000_000).contains(outputIndex),
            (0..<1_024).contains(storageLayerIndex), (1...Self.byteLimit).contains(maximumBytes)
        else { throw CBv2AttentionPacketError.invalidConfiguration }
        self.requestID = requestID
        self.outputIndex = outputIndex
        self.storageLayerIndex = storageLayerIndex
        self.maximumBytes = maximumBytes
    }
}

@_spi(Diagnostics)
public enum CBv2AttentionPacketError: Error {
    case invalidConfiguration, engineBusy, unsupportedModelOrMTP
}

/// Host-owned packed bytes. No MLX handle escapes the finalized step.
@_spi(Diagnostics)
public struct CBv2AttentionPacketTensor: Sendable {
    public let dtype: String
    public let shape: [Int]
    public let packedStrides: [Int]
    public let data: Data
}

@_spi(Diagnostics)
public struct CBv2AttentionPacketSnapshot: Sendable {
    public let configuration: CBv2AttentionPacketConfig
    public let metadata: CBv2AttentionMetadataSnapshot
    public let evaluationStatus: String
    public let reservedBytes: Int
    public let tensors: [String: CBv2AttentionPacketTensor]
}
