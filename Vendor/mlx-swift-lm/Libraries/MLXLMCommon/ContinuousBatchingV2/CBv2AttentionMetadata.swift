import Foundation
import MLX

/// One ordinary decode forward, observed without evaluating or retaining tensors.
@_spi(Diagnostics)
public struct CBv2AttentionMetadataConfig: Sendable, Encodable {
    public let requestID: UInt64
    public let outputIndex: Int
    public let maximumRecords: Int

    public init(requestID: UInt64, outputIndex: Int, maximumRecords: Int = 64) throws {
        guard requestID > 0, (1...1_000_000).contains(outputIndex),
            (1...128).contains(maximumRecords)
        else { throw CBv2AttentionMetadataError.invalidConfiguration }
        self.requestID = requestID
        self.outputIndex = outputIndex
        self.maximumRecords = maximumRecords
    }
}

@_spi(Diagnostics)
public enum CBv2AttentionMetadataError: Error {
    case invalidConfiguration, engineBusy, unsupportedModelOrMTP
}

@_spi(Diagnostics)
public struct CBv2AttentionTensorMetadata: Sendable, Encodable {
    public let dtype: String
    public let shape: [Int]
    /// MLX may change strides during evaluation. These describe graph
    /// construction only, never an evaluated physical layout.
    public let graphConstructionStrides: [Int]

    init(_ array: MLXArray) {
        dtype = String(describing: array.dtype)
        shape = array.shape
        graphConstructionStrides = array.strides
    }
}

@_spi(Diagnostics)
public struct CBv2AttentionMetadataRecord: Sendable, Encodable {
    public let requestID: UInt64
    public let outputIndex: Int
    public let phase: String
    public let batchIndex: Int
    public let batchSize: Int
    public let inputWidth: Int
    public let storageLayerIndex: Int
    public let modelLayerIndex: Int
    public let offsetBefore: Int
    public let offsetAfter: Int
    public let scaleBits: UInt32
    public let queries: CBv2AttentionTensorMetadata
    public let incomingKeys: CBv2AttentionTensorMetadata
    public let incomingValues: CBv2AttentionTensorMetadata
    public let storage: [String: CBv2AttentionTensorMetadata]
    public let kernelOutputDType: String
    public let output: CBv2AttentionTensorMetadata
    public let dispatch: String
    public let sinksPresent: Bool
    public let softcapPresent: Bool
    public let spansPresent: Bool
}

@_spi(Diagnostics)
public struct CBv2AttentionMetadataSnapshot: Sendable, Encodable {
    public let configuration: CBv2AttentionMetadataConfig
    public let records: [CBv2AttentionMetadataRecord]
    public let selectedForwards: Int
    public let expectedOwnerCount: Int
    /// Graph construction completed; this does not assert sample confirmation.
    public let forwardSucceeded: Bool
    public let sampleOutcome: String
    public let seedToken: Int?
    public let targetToken: Int?
    public let refusals: [String: Int]
}
