import CryptoKit
import Foundation
import MLXFastCore

/// Fixture locations and pins for the track target,
/// `bonsai2-27b-mlx-v1`.
let bonsai2Repository = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
let bonsai2Revision = "3f926b415992eaa2ae9dd7b573706494d6bbf787"

/// The track's MTP head is a SEPARATE published export. The pack declares
/// `mtp_num_hidden_layers: 0` and ships no `mtp.*` tensors.
let bonsai2HeadRepository = "EigenLabs/Qwen3.8-27B-MTP-4bit"
let bonsai2HeadRevision = "329261c5e0b3f9c233485e682cb3b67b88c20a55"

/// SHA256 and byte count of the pack's own `config.json` exactly as published
/// at the pinned revision.
///
/// The checked-in `fixtures/bonsai2_27b_config.json` is that file VERBATIM,
/// not a normalized re-render, so the digest below is the fixture's own digest
/// and the tests assert both at once. The same pair appears in the reference
/// manifest and in the tensor inventory's `source` block; a test holds all
/// three together, because a fixture that agrees with only one of them is a
/// fixture nobody can trust.
let bonsai2ConfigSHA256 =
    "238de7c512cc56a733421e3fd011d88f8260739e3d00e32c5d65b7943cc9f837"
let bonsai2ConfigByteCount = 58_145

// Tests/MLXFastTests/Model/<this file> -> repository root is four levels up.
private let bonsai2RepositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

let bonsai2ConfigFixtureURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/bonsai2_27b_config.json")

let bonsai2TrackContractURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/bonsai2_27b_mlx_v1_track.json")

let bonsai2TensorInventoryURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/bonsai2_27b_tensor_inventory.json")

let bonsai2ReferenceManifestURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/reference_bonsai2_27b_2bit.sha256")

let bonsai2HeadManifestURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256")

let bonsai2HeadDeclarationURL = bonsai2RepositoryRoot
    .appendingPathComponent("mtp-head.manifest.json")

func bonsai2TrackContractObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: bonsai2TrackContractURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Bonsai 2 track contract fixture must be a JSON object"
        )
    }
    return object
}

func bonsai2ConfigObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: bonsai2ConfigFixtureURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Bonsai 2 config fixture must be a JSON object"
        )
    }
    return object
}

/// One record of `fixtures/bonsai2_27b_tensor_inventory.json`.
struct Bonsai2TensorRecord: Decodable, Equatable {
    let dtype: String
    let shape: [Int]
    let shardIndex: Int

    enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case shardIndex = "shard_index"
    }
}

struct Bonsai2InventoryFixture: Decodable {
    struct Source: Decodable {
        let upstreamModelID: String
        let upstreamRevision: String
        let configBytes: Int
        let configSHA256: String
        let hadamardManifestBytes: Int
        let hadamardManifestSHA256: String

        enum CodingKeys: String, CodingKey {
            case upstreamModelID = "upstream_model_id"
            case upstreamRevision = "upstream_revision"
            case configBytes = "config_bytes"
            case configSHA256 = "config_sha256"
            case hadamardManifestBytes = "hadamard_manifest_bytes"
            case hadamardManifestSHA256 = "hadamard_manifest_sha256"
        }
    }

    struct Shard: Decodable {
        let index: Int
        let name: String
        let tensorCount: Int

        enum CodingKeys: String, CodingKey {
            case index
            case name
            case tensorCount = "tensor_count"
        }
    }

    struct Summary: Decodable {
        let tensorCount: Int
        let shardCount: Int
        let dtypeCounts: [String: Int]
        let mtpTensorCount: Int
        let languageModelTensorCount: Int
        let visionTowerTensorCount: Int
        let packedModuleCount: Int
        let signVectorCount: Int
        let lmHeadTensorCount: Int
        let totalSizeBytes: Int

        enum CodingKeys: String, CodingKey {
            case tensorCount = "tensor_count"
            case shardCount = "shard_count"
            case dtypeCounts = "dtype_counts"
            case mtpTensorCount = "mtp_tensor_count"
            case languageModelTensorCount = "language_model_tensor_count"
            case visionTowerTensorCount = "vision_tower_tensor_count"
            case packedModuleCount = "packed_module_count"
            case signVectorCount = "sign_vector_count"
            case lmHeadTensorCount = "lm_head_tensor_count"
            case totalSizeBytes = "total_size_bytes"
        }
    }

    let schemaVersion: Int
    let source: Source
    let shards: [Shard]
    let tensors: [String: Bonsai2TensorRecord]
    let summary: Summary

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case source
        case shards
        case tensors
        case summary
    }
}

func bonsai2InventoryFixture() throws -> Bonsai2InventoryFixture {
    try JSONDecoder().decode(
        Bonsai2InventoryFixture.self,
        from: Data(contentsOf: bonsai2TensorInventoryURL)
    )
}

/// One `<sha256> <byte_count> <relative_path>` record of a pinned manifest.
/// Comment and blank lines are skipped.
struct Bonsai2ManifestRecord: Equatable {
    let sha256: String
    let byteCount: Int
    let path: String
}

func bonsai2ManifestRecords(at url: URL) throws -> [Bonsai2ManifestRecord] {
    let text = try String(contentsOf: url, encoding: .utf8)
    return try text.split(separator: "\n").compactMap { rawLine -> Bonsai2ManifestRecord? in
        let line = rawLine.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty, !line.hasPrefix("#") else { return nil }
        let fields = line.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count == 3, let byteCount = Int(fields[1]) else {
            throw MLXFastError.invalidInput(
                "reference manifest record is not <sha256> <bytes> <path>: \(line)"
            )
        }
        return Bonsai2ManifestRecord(
            sha256: String(fields[0]),
            byteCount: byteCount,
            path: String(fields[2])
        )
    }
}

func bonsai2ReferenceManifestRecords() throws -> [Bonsai2ManifestRecord] {
    try bonsai2ManifestRecords(at: bonsai2ReferenceManifestURL)
}

func bonsai2HeadManifestRecords() throws -> [Bonsai2ManifestRecord] {
    try bonsai2ManifestRecords(at: bonsai2HeadManifestURL)
}

/// The two shape pins a manifest header declares, read back out of it. They
/// are asserted against the records they must be summed from: a count that is
/// not summed from the digests it must agree with is not a pin.
func bonsai2ManifestHeaderPins(at url: URL) throws -> (records: Int, bytes: Int) {
    let text = try String(contentsOf: url, encoding: .utf8)
    func pin(_ key: String) throws -> Int {
        guard let line = text.split(separator: "\n").first(where: { $0.contains(key) }),
              let value = line.split(separator: ":").last
                  .map({ $0.trimmingCharacters(in: .whitespaces) }),
              let number = Int(value)
        else {
            throw MLXFastError.invalidInput(
                "reference manifest header is missing \(key)"
            )
        }
        return number
    }
    return (
        try pin("MLXFAST_REFERENCE_MANIFEST_RECORDS"),
        try pin("MLXFAST_REFERENCE_MANIFEST_BYTES")
    )
}

/// Lowercase hex SHA256 of some bytes, the spelling every pin in this
/// repository is written in.
func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
