import Foundation
import MLX

/// Disk compatibility, distinct from a process-local assistant identity.
/// The provider supplies verified artifact/template identities and a fingerprint
/// of the binary, native implementation and numerical execution settings.
public struct CBv2CompleteCheckpointIdentity: Codable, Sendable, Equatable {
    public let modelAggregateHash: String
    public let promptContractID: String
    public let buildID: String
    public let numericsFingerprint: String

    public init(
        modelAggregateHash: String, promptContractID: String,
        buildID: String, numericsFingerprint: String
    ) {
        self.modelAggregateHash = modelAggregateHash
        self.promptContractID = promptContractID
        self.buildID = buildID
        self.numericsFingerprint = numericsFingerprint
    }

    var isValid: Bool {
        [modelAggregateHash, promptContractID, buildID, numericsFingerprint]
            .allSatisfy { !$0.isEmpty && $0.utf8.count <= 512 }
    }
}

public enum CBv2CheckpointTensorRole: String, Codable, Sendable {
    case keys, values, convolution, recurrent
    case assistantHidden, assistantTokens, assistantFrontier
}

public enum CBv2CheckpointDType: String, Codable, Sendable {
    case float16, bfloat16, float32, int32

    public var isFloatingPoint: Bool {
        self == .float16 || self == .bfloat16 || self == .float32
    }

    public var mlxDType: DType {
        switch self {
        case .float16: .float16
        case .bfloat16: .bfloat16
        case .float32: .float32
        case .int32: .int32
        }
    }

    public init?(_ dtype: DType) {
        switch dtype {
        case .float16: self = .float16
        case .bfloat16: self = .bfloat16
        case .float32: self = .float32
        case .int32: self = .int32
        default: return nil
        }
    }
}

/// Describes logical packed bytes, without the donor's allocation slack.
public struct CBv2CheckpointTensorDescriptor: Codable, Sendable, Equatable {
    public let role: CBv2CheckpointTensorRole
    public let layer: Int?
    public let shape: [Int]
    public let dtype: CBv2CheckpointDType
    public let byteCount: Int

    public init(
        role: CBv2CheckpointTensorRole, layer: Int? = nil,
        shape: [Int], dtype: CBv2CheckpointDType
    ) throws {
        self.role = role
        self.layer = layer
        self.shape = shape
        self.dtype = dtype
        self.byteCount = try Self.checkedByteCount(shape: shape, dtype: dtype.mlxDType)
    }

    func validate() throws {
        guard byteCount == (try Self.checkedByteCount(shape: shape, dtype: dtype.mlxDType)),
            layer == nil || layer! >= 0
        else { throw CBv2CompleteCheckpointError.invalidManifest }
    }

    static func checkedByteCount(shape: [Int], dtype: DType) throws -> Int {
        guard !shape.isEmpty, shape.count <= 8 else {
            throw CBv2CompleteCheckpointError.invalidManifest
        }
        var bytes = dtype.size
        for dimension in shape {
            guard dimension > 0, dimension <= Int(Int32.max) else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            let (next, overflow) = bytes.multipliedReportingOverflow(by: dimension)
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            bytes = next
        }
        return bytes
    }
}

/// This entire manifest, including token IDs and tenant scope, is encrypted.
/// Public disk metadata must contain only opaque lookup keys and envelope data.
public struct CBv2CompleteCheckpointManifest: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1
    public static let maximumEncodedBytes = 1 << 20
    public static let maximumSegmentBytes = 4 << 20
    public static let maximumProviderScratchBytes = 20 << 20
    public static let layout = "native-contiguous-full-recurrent-v1"
    public static let pagedLayout = "native-paged-full-recurrent-v1"
    public static let historicalAttentionLayout = "native-paged-historical-attention-v2"

    public let schemaVersion: Int
    public let identity: CBv2CompleteCheckpointIdentity
    public let backendLayout: String
    public let position: Int
    public let chunkSize: Int
    /// Read-only array aliases borrow the manifest's host ownership. Serving
    /// callers retain this manifest while using an alias; independently retained
    /// copies belong to the caller's own host-memory budget.
    public var prefixTokens: [Int] { metadata.tokens }
    public let cacheSalt: String?
    public let assistantCodecID: String?
    /// Shares the same ownership boundary as prefixTokens, including shape arrays.
    public var tensors: [CBv2CheckpointTensorDescriptor] { metadata.tensors }
    public var attentionLayers: [CBv2CheckpointAttentionLayer]? { metadata.attentionLayers }
    private(set) var metadata: CBv2CheckpointManifestMemory

    public init(
        identity: CBv2CompleteCheckpointIdentity, position: Int, chunkSize: Int,
        prefixTokens: [Int], cacheSalt: String?, assistantCodecID: String?,
        tensors: [CBv2CheckpointTensorDescriptor], backendLayout: String = Self.layout,
        attentionLayers: [CBv2CheckpointAttentionLayer]? = nil
    ) {
        self.init(schemaVersion: Self.currentSchemaVersion, identity: identity,
                  backendLayout: backendLayout, position: position, chunkSize: chunkSize,
                  cacheSalt: cacheSalt, assistantCodecID: assistantCodecID,
                  metadata: .init(tokens: prefixTokens, tensors: tensors, attentionLayers: attentionLayers))
    }

    init(schemaVersion: Int, identity: CBv2CompleteCheckpointIdentity, backendLayout: String,
         position: Int, chunkSize: Int, cacheSalt: String?, assistantCodecID: String?,
         metadata: CBv2CheckpointManifestMemory) {
        self.schemaVersion = schemaVersion
        self.identity = identity
        self.backendLayout = backendLayout
        self.position = position
        self.chunkSize = chunkSize
        self.cacheSalt = cacheSalt
        self.assistantCodecID = assistantCodecID
        self.metadata = metadata
    }

    func replacingMetadata(_ metadata: CBv2CheckpointManifestMemory) -> Self {
        var copy = self
        copy.metadata = metadata
        return copy
    }

    public func validateStructure() throws -> Int {
        defer { withExtendedLifetime(metadata) {} }
        guard schemaVersion == Self.currentSchemaVersion, identity.isValid,
            (backendLayout == Self.layout || backendLayout == Self.pagedLayout
                || backendLayout == Self.historicalAttentionLayout),
            (backendLayout == Self.historicalAttentionLayout
                ? attentionLayers?.isEmpty == false && attentionLayers!.count <= 2048
                : attentionLayers == nil),
            position > 1, chunkSize > 1,
            position % chunkSize == 0, prefixTokens.count == position,
            position <= Self.maximumEncodedBytes / 2,
            prefixTokens.allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) }),
            !tensors.isEmpty, tensors.count <= 4096,
            (cacheSalt?.utf8.count ?? 0) <= 4096,
            (assistantCodecID?.utf8.count ?? 0) <= 512
        else { throw CBv2CompleteCheckpointError.invalidManifest }
        var total = 0
        var roles = Set<String>()
        for tensor in tensors {
            try tensor.validate()
            guard roles.insert("\(tensor.role.rawValue):\(tensor.layer ?? -1)").inserted else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            let (next, overflow) = total.addingReportingOverflow(tensor.byteCount)
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            total = next
        }
        return total
    }
}

public enum CBv2CompleteCheckpointError: Error, Sendable, Equatable {
    case invalidManifest
    case incompatibleCheckpoint
    case invalidSegment
    case incompleteTransfer
    case closed
    case allocationFailed
}

/// Explicit loaded-model contract. A recurrent dtype alone does not establish
/// the dtype of attention K/V. The list follows the compact CBv2 layer layout.
public protocol CBv2CompleteCheckpointKVTypeProviding {
    var cbv2CompleteCheckpointKVDTypes: [DType]? { get }
}

/// Disk codecs reconstruct a checkpoint belonging to this loaded assistant;
/// they never deserialize mutable assistant KV or speculative state.
public protocol CBv2MTPPrefixCheckpointCoding: CBv2MTPPrefixCheckpointDrafter {
    var prefixCheckpointCodecID: String { get }
    func prefixCheckpointTensorDescriptors(targetInputCount: Int)
        -> [CBv2CheckpointTensorDescriptor]?
    func encodePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint) -> [MLXArray]?
    func decodePrefixCheckpoint(tensors: [MLXArray], prefixTokens: [Int])
        -> (any CBv2MTPPrefixCheckpoint)?
}

/// Provider-owned durable I/O. Neither lookup nor engine adoption performs I/O.
/// All methods must be thread-safe; each transfer object serializes its owner.
public protocol CBv2CompletePrefixCache: AnyObject, Sendable {
    var identity: CBv2CompleteCheckpointIdentity { get }

    /// Allocation-free policy probe. Refusing a later endpoint must preserve
    /// the donor's earlier reusable checkpoint. packedBytes excludes envelope
    /// and the bounded encrypted manifest, which the store accounts separately.
    func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool

    /// Consume one fully authenticated stage ticket, identified by submission
    /// receipt ID, not the potentially reused sampling request ID.
    func takeStaged(
        requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?, maximumSequenceLength: Int
    ) -> CBv2StagedCompleteCheckpoint?

    /// Always calls completion, including refusal/cancel. Nonempty positions
    /// mean durable commit. Close source before completion to release its
    /// tensor aliases; the engine keeps donor ownership until this callback.
    func donate(
        _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
        tokens: [Int], cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void
    )

    func close()
}

extension CBv2CompletePrefixCache {
    public func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool { true }
}
