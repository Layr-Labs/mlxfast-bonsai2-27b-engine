import CoreFoundation
import Foundation
import MLXFastCore
#if canImport(Darwin)
import Darwin
#endif

public struct TransformOptions: Equatable {
    public let referencePath: String
    public let outputPath: String

    public init(referencePath: String, outputPath: String) {
        self.referencePath = referencePath
        self.outputPath = outputPath
    }
}

public struct TransformReport: Equatable {
    public let referencePath: String
    public let outputPath: String
    public let denseTensorCount: Int
    public let denseShardCount: Int
    public let configPath: String
    public let indexPath: String

    public init(
        referencePath: String,
        outputPath: String,
        denseTensorCount: Int,
        denseShardCount: Int,
        configPath: String,
        indexPath: String
    ) {
        self.referencePath = referencePath
        self.outputPath = outputPath
        self.denseTensorCount = denseTensorCount
        self.denseShardCount = denseShardCount
        self.configPath = configPath
        self.indexPath = indexPath
    }
}

/// Model family of the source reference checkpoint, detected from its
/// config.json. The transform supports this track's pinned Ternary Bonsai 2
/// 27B target (`model_type` "prism_hadamard_qwen35", packed Hadamard modules,
/// untied head, no embedded speculative head) and the legacy Gemma 4
/// multimodal layout (nested `text_config`, tied head).
///
/// The legacy `.gemma4` family no longer has a runtime consumer (the Gemma 4
/// runtime and its MTP track were removed); it is retained here deliberately
/// as the only source-config family whose full transform pipeline (staging,
/// atomic install, revalidation, verifier) can be exercised end to end with
/// small synthetic fixtures -- the Bonsai 2 family requires the exact pinned
/// 2,057-tensor inventory with real shapes.
enum TransformModelFamily: Equatable {
    case gemma4
    case prismHadamardQwen35
}

/// Offline transform for the pinned reference checkpoint: selects the tensors
/// the runtime loads, drops every other prefix, and rewrites the selected
/// tensors into dense safetensors shard(s) plus a
/// `model.safetensors.index.json` and a runtime-authored `config.json`.
///
/// Two source checkpoint families are supported, detected from the source
/// config.json:
///
/// - Ternary Bonsai 2 27B 2-bit (`model_type` "prism_hadamard_qwen35"): this
///   track's target. The source is already MLX affine-quantized, so the
///   transform validates and passes through -- byte-for-byte, source tensor
///   names unchanged -- the 2,057-tensor `language_model.` set described by
///   `PrismHadamardCheckpointValidation`: the packed embedding, the 64 hybrid
///   linear-attention / full-attention layers, the final norm and the untied
///   `lm_head` quadruple. The pack also ships a 333-tensor vision tower; this
///   track serves text only, so that namespace is DROPPED here the same way
///   the model's own weight filter drops it at load. No metadata sidecars are
///   emitted: the runtime reads the pack's own packed tensors directly. The
///   runtime config.json is the source config whole, because the loader reads
///   its `modules`, `hadamard_config` and `components` blocks as the artifact
///   contract.
/// - Legacy Gemma 4 31B 4-bit (nested `text_config`): the archived dense
///   path, unchanged: flattened `text_config` runtime config plus the
///   projection/tied-head metadata sidecars.
///
/// There is no expert streaming manifest -- the whole selected tree is one
/// flat set of dense tensors, matching how the model (including every routed
/// expert) is loaded fully into RAM at runtime init.
public enum SwiftTransform {
    /// Tensor name prefix that marks a checkpoint tensor as part of the text
    /// tower. Every other prefix (`vision_tower.`, `embed_vision.`,
    /// `audio_tower.`, `multi_modal_projector.`, ...) is vision/audio/
    /// multimodal-glue and is out of scope for this text-only challenge.
    static let textTowerPrefix = "language_model."

    public static func run(_ options: TransformOptions) throws -> TransformReport {
        try run(
            options,
            beforeSidecarGeneration: nil,
            beforeSourceRevalidation: nil
        )
    }

    static func run(
        _ options: TransformOptions,
        beforeSidecarGeneration: (() throws -> Void)? = nil,
        beforeSourceRevalidation: (() throws -> Void)?
    ) throws -> TransformReport {
        let referenceDirectory = canonicalURL(
            try findReferenceDirectory(URL(fileURLWithPath: options.referencePath))
        )
        let outputDirectory = canonicalURL(URL(fileURLWithPath: options.outputPath))
        try validateDistinctDirectories(
            referenceDirectory: referenceDirectory,
            outputDirectory: outputDirectory
        )
        var outputIsDirectory = ObjCBool(false)
        if FileManager.default.fileExists(
            atPath: outputDirectory.path,
            isDirectory: &outputIsDirectory
        ), !outputIsDirectory.boolValue {
            throw MLXFastError.invalidInput(
                "transform output exists and is not a directory: \(outputDirectory.path)"
            )
        }

        let referenceConfigPath = referenceDirectory.appendingPathComponent("config.json")
        try requireFile(
            referenceConfigPath.path,
            description: "reference checkpoint config"
        )
        let sourceConfigRoot = try loadReferenceConfigRoot(referenceConfigPath)
        let modelFamily = try detectModelFamily(sourceConfigRoot: sourceConfigRoot)
        let runtimeConfigData = try makeRuntimeConfigData(
            sourceConfigRoot: sourceConfigRoot,
            family: modelFamily
        )
        let metadataSnapshot = try captureMetadataFiles(from: referenceDirectory)

        let index = try loadIndex(referenceDirectory)
        let indexSnapshot = try index.canonicalData()
        let validatedHeaders = try validateCheckpointIndex(
            index,
            referenceDirectory: referenceDirectory
        )
        let textKeys = Set(
            index.weightMap.keys.filter { isSelectedTextTowerKey($0, family: modelFamily) }
        )
        guard !textKeys.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no text-tower tensors")
        }

        let textKeysByShard = Dictionary(grouping: textKeys) { key in
            index.weightMap[key] ?? ""
        }
        var totalTensorByteCount = 0
        for key in textKeys.sorted() {
            guard let shardName = index.weightMap[key],
                  let info = validatedHeaders[shardName]?.tensors[key]
            else {
                throw MLXFastError.invalidInput(
                    "missing validated tensor metadata for \(key)"
                )
            }
            let (nextTotal, overflow) = totalTensorByteCount.addingReportingOverflow(
                info.byteCount
            )
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "transformed tensor byte count overflows Int"
                )
            }
            totalTensorByteCount = nextTotal
        }

        // Fail before the multi-GB copy if the selected tensor set is
        // structurally inconsistent with the quantization spec the emitted
        // config.json declares (the runtime re-validates the full geometry
        // against its own config type at load). Gemma 4 is the legacy
        // synthetic-fixture family and has no hardcoded public inventory.
        switch modelFamily {
        case .prismHadamardQwen35:
            try PrismHadamardCheckpointValidation.validateSelectedTensors(
                selectedKeys: textKeys,
                index: index,
                headers: validatedHeaders,
                quantization: PrismHadamardCheckpointValidation.quantizationSpec(
                    fromConfigRoot: sourceConfigRoot
                )
            )
        case .gemma4:
            break
        }

        let fileManager = FileManager.default
        let stagingDirectory = outputDirectory.deletingLastPathComponent().appendingPathComponent(
            ".\(outputDirectory.lastPathComponent).mlxfast-transform-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: stagingDirectory.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: false)
        var installed = false
        defer {
            if !installed {
                try? fileManager.removeItem(at: stagingDirectory)
            }
        }

        var copiedTensors = 0
        var stagedHeaders: [String: SafetensorsHeader] = [:]
        for shardName in textKeysByShard.keys.sorted() {
            let source = referenceDirectory.appendingPathComponent(shardName)
            let destination = stagingDirectory.appendingPathComponent(shardName)
            guard let header = validatedHeaders[shardName] else {
                throw MLXFastError.invalidInput("missing validated header for checkpoint shard \(shardName)")
            }
            let selectedNames = textKeysByShard[shardName, default: []].sorted()
            if Set(selectedNames) == Set(header.tensors.keys) {
                // Poolside's reference is already text-only. Preserve the
                // byte-identical shard and use an APFS copy-on-write clone
                // when source/output share a volume; fall back to a normal
                // independent file copy elsewhere. Never publish a symlink.
                try cloneOrCopyShard(from: source, to: destination)
                copiedTensors += selectedNames.count
            } else {
                copiedTensors += try Safetensors.copySubset(
                    from: source,
                    to: destination,
                    tensorNames: selectedNames,
                    validatedHeader: header
                )
            }
            stagedHeaders[shardName] = try Safetensors.readHeader(destination)
        }

        try beforeSidecarGeneration?()
        let generatedProjectionMetadata: GeneratedAffineMetadataReport
        let generatedTiedHeadMetadata: GeneratedAffineMetadataReport
        switch modelFamily {
        case .gemma4:
            generatedProjectionMetadata = try AffineMetadataCoding.writeProjectionSidecar(
                sourceDirectory: stagingDirectory,
                index: index,
                sourceHeaders: stagedHeaders,
                selectedKeys: textKeys,
                destinationDirectory: stagingDirectory
            )
            generatedTiedHeadMetadata = try TiedHeadMetadataCoding.writeSidecar(
                sourceDirectory: stagingDirectory,
                index: index,
                sourceHeaders: stagedHeaders,
                selectedKeys: textKeys,
                destinationDirectory: stagingDirectory
            )
        case .prismHadamardQwen35:
            // Bonsai 2 has an untied `lm_head` and the runtime reads the
            // pack's own packed tensors directly, so neither the Gemma
            // projection sidecar nor the tied-head packed13 sidecar means
            // anything on this family. Emitting one would publish a derived
            // artifact nothing loads, so emit nothing beyond the pass-through
            // tensor set.
            generatedProjectionMetadata = GeneratedAffineMetadataReport(
                weightMap: [:],
                tensorByteCount: 0
            )
            generatedTiedHeadMetadata = GeneratedAffineMetadataReport(
                weightMap: [:],
                tensorByteCount: 0
            )
        }
        let (projectionOutputByteCount, projectionSizeOverflow) =
            totalTensorByteCount.addingReportingOverflow(
                generatedProjectionMetadata.tensorByteCount
            )
        let (outputTensorByteCount, tiedHeadSizeOverflow) =
            projectionOutputByteCount.addingReportingOverflow(
                generatedTiedHeadMetadata.tensorByteCount
            )
        guard !projectionSizeOverflow, !tiedHeadSizeOverflow else {
            throw MLXFastError.invalidInput("transformed tensor byte count overflows Int")
        }
        let generatedWeightMap = generatedProjectionMetadata.weightMap.merging(
            generatedTiedHeadMetadata.weightMap
        ) { _, _ in
            preconditionFailure("generated metadata tensor names collide")
        }

        try writeMetadataFiles(metadataSnapshot, to: stagingDirectory)
        try index.writeStripped(
            to: stagingDirectory.appendingPathComponent("model.safetensors.index.json"),
            keeping: textKeys,
            totalTensorByteCount: outputTensorByteCount,
            additionalWeightMap: generatedWeightMap
        )

        try runtimeConfigData.write(
            to: stagingDirectory.appendingPathComponent("config.json")
        )
        try beforeSourceRevalidation?()
        try validateConfigAndIndexSnapshot(
            referenceDirectory: referenceDirectory,
            referenceConfigPath: referenceConfigPath,
            runtimeConfigData: runtimeConfigData,
            indexSnapshot: indexSnapshot,
            metadataSnapshot: metadataSnapshot
        )
        for shardName in validatedHeaders.keys.sorted() {
            guard let header = validatedHeaders[shardName] else {
                throw MLXFastError.invalidInput(
                    "missing validated header for checkpoint shard \(shardName)"
                )
            }
            try Safetensors.validateSourceIdentity(
                referenceDirectory.appendingPathComponent(shardName),
                against: header
            )
        }
        try validateConfigAndIndexSnapshot(
            referenceDirectory: referenceDirectory,
            referenceConfigPath: referenceConfigPath,
            runtimeConfigData: runtimeConfigData,
            indexSnapshot: indexSnapshot,
            metadataSnapshot: metadataSnapshot
        )
        try installTransformedDirectory(
            stagingDirectory,
            at: outputDirectory,
            fileManager: fileManager
        )
        installed = true

        let indexPath = outputDirectory.appendingPathComponent("model.safetensors.index.json")
        let configPath = outputDirectory.appendingPathComponent("config.json")

        return TransformReport(
            referencePath: referenceDirectory.path,
            outputPath: outputDirectory.path,
            denseTensorCount: copiedTensors
                + generatedProjectionMetadata.tensorCount
                + generatedTiedHeadMetadata.tensorCount,
            denseShardCount: textKeysByShard.count
                + generatedProjectionMetadata.shardCount
                + generatedTiedHeadMetadata.shardCount,
            configPath: configPath.path,
            indexPath: indexPath.path
        )
    }

    private static func loadIndex(_ referenceDirectory: URL) throws -> CheckpointIndex {
        let indexPath = referenceDirectory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexPath.path) {
            return try CheckpointIndex.load(from: indexPath)
        }
        return try CheckpointIndex.buildFromSafetensors(in: referenceDirectory)
    }

    private static func validateCheckpointIndex(
        _ index: CheckpointIndex,
        referenceDirectory: URL
    ) throws -> [String: SafetensorsHeader] {
        guard !index.weightMap.isEmpty else {
            throw MLXFastError.invalidInput("checkpoint index contains no tensors")
        }

        let keysByShard = Dictionary(grouping: index.weightMap.keys.sorted()) { key in
            index.weightMap[key] ?? ""
        }
        var headersByShard: [String: SafetensorsHeader] = [:]
        for shardName in keysByShard.keys.sorted() {
            try validateSafetensorsShardName(shardName, context: "checkpoint index")

            let shardURL = referenceDirectory.appendingPathComponent(shardName)
            try requireFile(shardURL.path, description: "checkpoint shard \(shardName)")
            let header = try Safetensors.readHeader(shardURL)
            headersByShard[shardName] = header

            for key in keysByShard[shardName, default: []].sorted() {
                guard let info = header.tensors[key] else {
                    throw MLXFastError.invalidInput(
                        "checkpoint index lists tensor \(key) in \(shardName), but the shard header does not contain it"
                    )
                }
                let dtype = try TensorDType.parse(info.dtype)
                let expectedByteLength = try expectedTensorByteCount(
                    name: key,
                    dtype: dtype,
                    shape: info.shape
                )
                guard info.byteCount == expectedByteLength else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) byte length \(info.byteCount) does not match dtype \(info.dtype) and shape \(info.shape) expected \(expectedByteLength)"
                    )
                }
                // readHeader validated every data range against the opened
                // target descriptor. Do not recheck via the shard pathname:
                // FileManager reports a symlink's own size, not its target's.
                guard info.dataStart >= 0, info.byteCount > 0 else {
                    throw MLXFastError.invalidInput(
                        "checkpoint tensor \(key) has an empty or invalid byte range"
                    )
                }
            }
        }
        return headersByShard
    }

    private static func canonicalURL(_ url: URL) -> URL {
        url.standardizedFileURL.resolvingSymlinksInPath()
    }

    static func validateDistinctDirectories(
        referenceDirectory: URL,
        outputDirectory: URL,
        workingDirectory: URL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true
        )
    ) throws {
        guard referenceDirectory.path != outputDirectory.path else {
            throw MLXFastError.invalidInput(
                "transform reference and output directories must be different: \(referenceDirectory.path)"
            )
        }
        let outputPrefix = outputDirectory.path == "/" ? "/" : outputDirectory.path + "/"
        guard !referenceDirectory.path.hasPrefix(outputPrefix) else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot contain the reference directory: \(outputDirectory.path)"
            )
        }
        let referencePrefix = referenceDirectory.path == "/"
            ? "/"
            : referenceDirectory.path + "/"
        guard !outputDirectory.path.hasPrefix(referencePrefix) else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot be inside the reference directory: \(outputDirectory.path)"
            )
        }

        let canonicalWorkingDirectory = canonicalURL(workingDirectory)
        guard canonicalWorkingDirectory.path != outputDirectory.path,
              !canonicalWorkingDirectory.path.hasPrefix(outputPrefix)
        else {
            throw MLXFastError.invalidInput(
                "transform output directory cannot contain the current working directory: \(outputDirectory.path)"
            )
        }
    }

    private static func installTransformedDirectory(
        _ stagedDirectory: URL,
        at outputDirectory: URL,
        fileManager: FileManager
    ) throws {
        if fileManager.fileExists(atPath: outputDirectory.path) {
            _ = try fileManager.replaceItemAt(outputDirectory, withItemAt: stagedDirectory)
        } else {
            try fileManager.moveItem(at: stagedDirectory, to: outputDirectory)
        }
    }

    private static func findReferenceDirectory(_ base: URL) throws -> URL {
        if FileManager.default.fileExists(
            atPath: base.appendingPathComponent("config.json").path
        ) {
            return base
        }

        guard let enumerator = FileManager.default.enumerator(
            at: base,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw MLXFastError.missingFile("reference path not found at \(base.path)")
        }

        for case let url as URL in enumerator {
            if url.lastPathComponent == "config.json" {
                return url.deletingLastPathComponent()
            }
        }

        throw MLXFastError.missingFile(
            "no config.json found under \(base.path); place the pinned reference checkpoint there"
        )
    }

    static func isTextTowerKey(_ key: String) -> Bool {
        key.hasPrefix(textTowerPrefix)
    }

    /// Both families select the `language_model.*` namespace. Bonsai 2 drops
    /// the pack's `vision_tower.*` tensors with the same test.
    static func isSelectedTextTowerKey(_ key: String, family: TransformModelFamily) -> Bool {
        switch family {
        case .gemma4:
            return isTextTowerKey(key)
        case .prismHadamardQwen35:
            // The pinned pack holds 2,390 tensors: 2,057 under
            // `language_model.` and a 333-tensor vision tower. This track
            // serves text only, so the tower is dropped here and the
            // exact-inventory pass below refuses any third namespace.
            return PrismHadamardCheckpointValidation.isSelectedKey(key)
        }
    }

    private static func captureMetadataFiles(from source: URL) throws -> [String: Data] {
        let files = try FileManager.default.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )
        var snapshot: [String: Data] = [:]
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if file.lastPathComponent == "model.safetensors.index.json" || file.lastPathComponent == "config.json" {
                continue
            }
            if shouldCopyMetadataFile(file) {
                let values = try file.resourceValues(
                    forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
                )
                guard values.isRegularFile == true, values.isSymbolicLink != true else {
                    throw MLXFastError.invalidInput(
                        "reference metadata is not a regular file: \(file.path)"
                    )
                }
                snapshot[file.lastPathComponent] = try Data(contentsOf: file)
            }
        }
        return snapshot
    }

    private static func writeMetadataFiles(
        _ snapshot: [String: Data],
        to destination: URL
    ) throws {
        for name in snapshot.keys.sorted() {
            try snapshot[name]?.write(to: destination.appendingPathComponent(name))
        }
    }

    private static func shouldCopyMetadataFile(_ url: URL) -> Bool {
        let name = url.lastPathComponent
        if name.hasSuffix(".safetensors") {
            return false
        }
        switch url.pathExtension {
        // `jinja`: the pinned checkpoint ships its chat template as
        // `chat_template.jinja` rather than inside `tokenizer_config.json`.
        case "jinja", "json", "model", "tiktoken", "txt":
            return true
        default:
            return name == "tokenizer" || name == "vocab"
        }
    }

    private static func cloneOrCopyShard(from source: URL, to destination: URL) throws {
        #if canImport(Darwin)
        let cloned = source.path.withCString { sourcePath in
            destination.path.withCString { destinationPath in
                clonefile(sourcePath, destinationPath, 0) == 0
            }
        }
        if !cloned {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.copyItem(at: source, to: destination)
        }
        #else
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: source, to: destination)
        #endif
        let values = try destination.resourceValues(
            forKeys: [.isRegularFileKey, .isSymbolicLinkKey]
        )
        guard values.isRegularFile == true, values.isSymbolicLink != true else {
            throw MLXFastError.invalidInput(
                "transformed shard copy is not an independent regular file: \(destination.path)"
            )
        }
    }

    private static func loadReferenceConfigRoot(_ sourceConfigPath: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: sourceConfigPath)
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any] else {
            throw MLXFastError.invalidInput("reference config.json must be a JSON object")
        }
        return root
    }

    /// Model-type prefix of this track's target. The pinned pack declares
    /// `prism_hadamard_qwen35`; the prefix is matched (rather than the exact
    /// string) so a point revision of the same architecture family is still
    /// routed to the packed path instead of being refused outright.
    static let prismHadamardModelTypePrefix = "prism_hadamard_qwen35"

    static func detectModelFamily(
        sourceConfigRoot root: [String: Any]
    ) throws -> TransformModelFamily {
        // THIS TRACK'S TARGET declares `prism_hadamard_qwen35` at the top
        // level AND carries a `text_config`, so it is checked FIRST: the
        // `text_config`-means-legacy-Gemma fallthrough below is unconditional
        // and would otherwise swallow it.
        if let modelType = root["model_type"] as? String,
           modelType.hasPrefix(prismHadamardModelTypePrefix)
        {
            return .prismHadamardQwen35
        }
        if root["text_config"] is [String: Any] {
            return .gemma4
        }
        throw MLXFastError.invalidInput(
            "reference config.json is missing text_config and does not declare "
                + "model_type prism_hadamard_qwen35"
        )
    }

    static func makeRuntimeConfigData(sourceConfigPath: URL) throws -> Data {
        let root = try loadReferenceConfigRoot(sourceConfigPath)
        return try makeRuntimeConfigData(
            sourceConfigRoot: root,
            family: detectModelFamily(sourceConfigRoot: root)
        )
    }

    /// Writes the runtime's `config.json`.
    ///
    /// Gemma 4 (legacy): the source checkpoint's `text_config` fields
    /// flattened to the top level (the schema the archived Gemma 4 runtime
    /// read), plus the
    /// checkpoint-wide `quantization` block -- no vision or audio config, no
    /// architecture/tokenizer metadata duplicated from
    /// `tokenizer_config.json`.
    ///
    /// Bonsai 2: the source config is the artifact contract the packed loader
    /// reads, so it is passed through whole. The one edit is the duplicate
    /// quantization block, below.
    static func makeRuntimeConfigData(
        sourceConfigRoot root: [String: Any],
        family: TransformModelFamily
    ) throws -> Data {
        var runtimeConfig: [String: Any]
        switch family {
        case .gemma4:
            guard let textConfig = root["text_config"] as? [String: Any] else {
                throw MLXFastError.invalidInput("reference config.json is missing text_config")
            }
            runtimeConfig = textConfig
            if let quantization = root["quantization"] {
                runtimeConfig["quantization"] = quantization
            } else if let quantizationConfig = root["quantization_config"] {
                runtimeConfig["quantization"] = quantizationConfig
            }
        case .prismHadamardQwen35:
            // The source config is passed through WHOLE. It is the artifact
            // contract, not just a geometry: `PrismHadamardCheckpoint` reads
            // its `modules`, `components`, `hadamard_config`,
            // `gdn_activation_layout` and `tensor_namespace` before it will
            // install a single packed module, so a dropped field refuses the
            // load.
            //
            // The pack publishes the affine spec ONCE, as `quantization`. A
            // `quantization_config` duplicate is accepted and must agree;
            // emitting one of two conflicting specs would silently pick a
            // quantization the shards were not written with. The duplicate is
            // then removed so a later hand edit cannot desync the pair inside
            // a transformed tree.
            let spec = try PrismHadamardCheckpointValidation.quantizationSpec(
                fromConfigRoot: root
            )
            runtimeConfig = root
            runtimeConfig.removeValue(forKey: "quantization_config")
            runtimeConfig["quantization"] = [
                "group_size": spec.groupSize,
                "bits": spec.bits,
                "mode": spec.mode,
            ]
        }

        return try JSONSerialization.data(
            withJSONObject: runtimeConfig,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
    }

    private static func validateConfigAndIndexSnapshot(
        referenceDirectory: URL,
        referenceConfigPath: URL,
        runtimeConfigData: Data,
        indexSnapshot: Data,
        metadataSnapshot: [String: Data]
    ) throws {
        guard try makeRuntimeConfigData(sourceConfigPath: referenceConfigPath)
            == runtimeConfigData
        else {
            throw MLXFastError.invalidInput(
                "reference config changed while transform was running"
            )
        }
        guard try loadIndex(referenceDirectory).canonicalData() == indexSnapshot else {
            throw MLXFastError.invalidInput(
                "checkpoint index changed while transform was running"
            )
        }
        guard try captureMetadataFiles(from: referenceDirectory) == metadataSnapshot else {
            throw MLXFastError.invalidInput(
                "reference tokenizer metadata changed while transform was running"
            )
        }
    }
}
