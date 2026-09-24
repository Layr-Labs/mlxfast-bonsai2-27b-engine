import CoreFoundation
import Foundation
import MLXFastCore

/// Quantization expectations parsed from the pinned Ternary Bonsai 2 27B
/// pack's `quantization` block.
///
/// The pack is UNIFORM: MLX affine, 2 bits, group size 128, and the contract
/// fixture records `mixed_precision: false`. There is no per-tensor override
/// table, so the three scalars ARE the whole block and a fourth key is
/// refused rather than half-read.
struct PrismHadamardQuantizationSpec: Equatable {
    let groupSize: Int
    let bits: Int
    let mode: String
}

/// Transform-side structural validation of the Ternary Bonsai 2 27B tensor
/// set.
///
/// The source pack (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`) is already MLX
/// affine-quantized, so the transform passes tensors through unchanged. This
/// pass fails fast -- before the multi-gigabyte copy -- when the set it would
/// copy cannot satisfy the runtime loader:
///
/// - the selected namespace must be EXACTLY the published 2,057-tensor
///   `language_model.` inventory, tensor for tensor, at the exact dtype and
///   shape;
/// - every packed projection must ship FP16 `.scales`, FP16 `.biases` AND an
///   FP32 `.signs` vector. The signs are what make the weights readable: the
///   pack folds a signed block Walsh-Hadamard transform into every packed
///   linear, and a triple without its signs is unusable;
/// - each packed width must match the group size and bit width the config
///   declares: `in / 16` U32 columns and `in / 128` scale and bias columns at
///   2 bits and group 128, with `in` sign entries;
/// - compressed-tensors aliases, global-scale tensors and FP8 KV scales are
///   rejected outright;
/// - the `lm_head.*` quadruple MUST appear. `tie_word_embeddings` is false,
///   so a pack without its own output projection is a different artifact;
/// - NO `mtp.*` tensor may appear. This pack declares
///   `mtp_num_hidden_layers: 0` and ships no head. The track's head is a
///   separate pinned export (`fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256`),
///   and a pack carrying an embedded head is a different artifact.
///
/// THE TOWER IS HYBRID, and that is the fact this file is built around. 64
/// layers on a fixed schedule: every fourth layer is full attention and the
/// other three are gated-delta-net linear attention. Both kinds carry the
/// same two layer norms and the same dense MLP; they differ in the mixer.
///
/// THE VISION TOWER IS NOT SELECTED. The pack declares one and ships its 333
/// tensors. This track serves text only, the runner declares
/// `multimodal: false`, and the model's own weight filter drops the same
/// prefix at load, so the transform drops it here and the tower never reaches
/// `weights/`.
///
/// Deliberately independent of shard placement: this pass pins the complete
/// tensor namespace, dtype and shape, while the public inventory fixture
/// `fixtures/bonsai2_27b_tensor_inventory.json` pins placement.
enum PrismHadamardCheckpointValidation {
    struct ExpectedTensorMetadata: Equatable {
        let dtype: String
        let shape: [Int]
    }

    /// One entry of the fixed `layer_types` schedule.
    enum LayerKind: String, Equatable {
        case linearAttention = "linear_attention"
        case fullAttention = "full_attention"
    }

    /// Frozen geometry of the pinned pack. Kept as literals rather than read
    /// from the config under validation: a validator that derives its
    /// expectations from the artifact it is checking cannot detect a changed
    /// artifact.
    enum PinnedGeometry {
        static let vocabSize = 248_320
        static let hiddenSize = 5_120
        static let intermediateSize = 17_408
        static let layerCount = 64
        /// Every `fullAttentionInterval`-th layer owns KV; the rest are
        /// recurrent.
        static let fullAttentionInterval = 4

        static let attentionHeads = 24
        static let keyValueHeads = 4
        static let headDim = 256
        /// `attn_output_gate` is true, so the query projection emits the
        /// query AND its gate.
        static let attentionOutputGate = true

        static let linearKeyHeads = 16
        static let linearValueHeads = 48
        static let linearKeyHeadDim = 128
        static let linearValueHeadDim = 128
        static let linearConvKernel = 4

        static let quantizationGroupSize = 128
        static let quantizationBits = 2
        static let quantizationMode = "affine"

        /// The signed block Walsh-Hadamard block width every packed module
        /// declares in `hadamard.json`.
        static let hadamardBlockSize = 1_024

        /// The fixed schedule, derived from the interval so a geometry the
        /// tower does not have cannot be spelled out by hand.
        static var layerTypes: [LayerKind] {
            (0 ..< layerCount).map {
                ($0 + 1).isMultiple(of: fullAttentionInterval)
                    ? .fullAttention : .linearAttention
            }
        }

        static var attentionDim: Int { attentionHeads * headDim }
        static var queryProjectionDim: Int {
            attentionOutputGate ? 2 * attentionDim : attentionDim
        }
        static var keyValueDim: Int { keyValueHeads * headDim }

        static var linearKeyDim: Int { linearKeyHeads * linearKeyHeadDim }
        static var linearValueDim: Int { linearValueHeads * linearValueHeadDim }
        /// The depthwise convolution runs over the query, the key and the
        /// value streams.
        static var linearConvDim: Int { 2 * linearKeyDim + linearValueDim }
    }

    /// 2,057 tensors under `language_model.`.
    static let expectedTensorCount = 2_057
    /// `model.embed_tokens` (4), `model.norm` (1) and `lm_head` (4).
    static let expectedTopLevelTensorCount = 9
    /// A linear-attention layer: two layer norms, the 16 mixer tensors and
    /// the 12 MLP tensors.
    static let expectedLinearAttentionLayerTensorCount = 32
    /// A full-attention layer: two layer norms, the 18 mixer tensors and the
    /// 12 MLP tensors.
    static let expectedFullAttentionLayerTensorCount = 32
    /// Every packed module ships one sign vector.
    static let expectedPackedModuleCount = 402

    /// The one namespace the transform selects. `vision_tower.` is the only
    /// other namespace the pack ships and it is deliberately dropped.
    static let selectedPrefix = "language_model."
    static let visionPrefix = "vision_tower."

    private static let layerPrefix = "language_model.model.layers."

    static func isSelectedKey(_ key: String) -> Bool {
        key.hasPrefix(selectedPrefix)
    }

    /// Parses the pack's quantization block.
    ///
    /// The pack publishes the spec ONCE, as `quantization`. A
    /// `quantization_config` duplicate is accepted when present and must
    /// agree, so this validator applies exactly the policy
    /// `SwiftTransform.makeRuntimeConfigData` applies rather than a stricter
    /// one.
    static func quantizationSpec(
        fromConfigRoot root: [String: Any]
    ) throws -> PrismHadamardQuantizationSpec {
        func parseBlock(_ key: String) throws -> PrismHadamardQuantizationSpec? {
            guard let value = root[key], !(value is NSNull) else { return nil }
            guard let block = value as? [String: Any] else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 config \(key) must be an object")
            }
            let allowedKeys: Set<String> = ["group_size", "bits", "mode"]
            // UNIFORM. Every extra key is a per-tensor override this transform
            // would carry into a config the runtime then resolves one width
            // from. Refuse rather than emit a block whose extra half nothing
            // reads.
            let unexpectedKeys = Set(block.keys).subtracting(allowedKeys).sorted()
            guard unexpectedKeys.isEmpty else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 config \(key) carries \(unexpectedKeys.count) "
                        + "unsupported entr(ies) and this pack is uniform, "
                        + "first: \(unexpectedKeys[0])")
            }
            guard block["group_size"] != nil, block["bits"] != nil,
                block["mode"] != nil
            else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 config \(key) must explicitly define group_size, "
                        + "bits, and mode")
            }
            let groupSize = try intField("group_size", in: block)
            let bits = try intField("bits", in: block)
            let mode = try stringField("mode", in: block)
            guard mode == PinnedGeometry.quantizationMode,
                groupSize == PinnedGeometry.quantizationGroupSize,
                bits == PinnedGeometry.quantizationBits
            else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 quantization must be affine 2-bit group_size 128")
            }
            return PrismHadamardQuantizationSpec(
                groupSize: groupSize, bits: bits, mode: mode)
        }

        let quantization = try parseBlock("quantization")
        let quantizationConfig = try parseBlock("quantization_config")
        if let quantization, let quantizationConfig {
            guard quantization == quantizationConfig else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 config quantization and quantization_config must "
                        + "match exactly")
            }
            return quantization
        }
        guard let spec = quantization ?? quantizationConfig else {
            throw MLXFastError.invalidInput(
                "Bonsai 2 config is missing both quantization and "
                    + "quantization_config")
        }
        return spec
    }

    /// Exact metadata contract of the transformed checkpoint.
    ///
    /// DTYPES are read off the pinned artifact: packed codes `U32`, their
    /// scale and bias companions `F16`, and every unpacked parameter -- the
    /// norms, the gate state, the convolution kernel, the small `a`/`b`
    /// projections and the sign vectors -- `F32`. SHAPES are DERIVED from the
    /// pinned geometry above, so a geometry the tower does not have fails here
    /// rather than silently reproducing whatever the artifact holds.
    static func expectedTensorInventory() -> [String: ExpectedTensorMetadata] {
        let hidden = PinnedGeometry.hiddenSize
        let vocab = PinnedGeometry.vocabSize
        let groupSize = PinnedGeometry.quantizationGroupSize
        let bits = PinnedGeometry.quantizationBits

        var inventory: [String: ExpectedTensorMetadata] = [:]
        func add(_ name: String, _ dtype: TensorDType, _ shape: [Int]) {
            precondition(
                inventory[name] == nil,
                "duplicate expected Bonsai 2 tensor \(name)")
            inventory[name] = ExpectedTensorMetadata(
                dtype: dtype.rawValue, shape: shape)
        }
        /// One packed Hadamard module: the U32 codes, the FP16 scale and bias
        /// companions the affine scheme requires, and the FP32 sign vector the
        /// transform is defined by. The sign vector covers the whole
        /// contracted axis, not one block of it.
        func addPacked(_ stem: String, outFeatures: Int, inFeatures: Int) {
            precondition(
                inFeatures.isMultiple(of: groupSize)
                    && inFeatures.isMultiple(of: PinnedGeometry.hadamardBlockSize)
                    && (inFeatures * bits).isMultiple(of: 32),
                "Bonsai 2 tensor \(stem) is not packable at \(bits) bits")
            add("\(stem).weight", .u32, [outFeatures, inFeatures * bits / 32])
            add("\(stem).scales", .f16, [outFeatures, inFeatures / groupSize])
            add("\(stem).biases", .f16, [outFeatures, inFeatures / groupSize])
            add("\(stem).signs", .f32, [inFeatures])
        }
        /// The dense MLP. `hidden_act` is silu, so it is a gate/up pair into a
        /// down projection.
        func addMLP(_ prefix: String) {
            for projection in ["gate_proj", "up_proj"] {
                addPacked(
                    "\(prefix).\(projection)",
                    outFeatures: PinnedGeometry.intermediateSize,
                    inFeatures: hidden)
            }
            addPacked(
                "\(prefix).down_proj", outFeatures: hidden,
                inFeatures: PinnedGeometry.intermediateSize)
        }
        /// The gated-delta-net mixer. `A_log` and `dt_bias` are per value
        /// head, the small `a`/`b` projections stay in full precision, and the
        /// depthwise kernel is in MLX layout: (channels, kernel, 1).
        func addLinearAttentionMixer(_ prefix: String) {
            for parameter in ["A_log", "dt_bias"] {
                add(
                    "\(prefix).\(parameter)", .f32,
                    [PinnedGeometry.linearValueHeads])
            }
            add(
                "\(prefix).conv1d.weight", .f32,
                [PinnedGeometry.linearConvDim, PinnedGeometry.linearConvKernel, 1])
            add("\(prefix).norm.weight", .f32, [PinnedGeometry.linearValueHeadDim])
            for projection in ["in_proj_a", "in_proj_b"] {
                add(
                    "\(prefix).\(projection).weight", .f32,
                    [PinnedGeometry.linearValueHeads, hidden])
            }
            addPacked(
                "\(prefix).in_proj_qkv",
                outFeatures: PinnedGeometry.linearConvDim, inFeatures: hidden)
            addPacked(
                "\(prefix).in_proj_z",
                outFeatures: PinnedGeometry.linearValueDim, inFeatures: hidden)
            addPacked(
                "\(prefix).out_proj", outFeatures: hidden,
                inFeatures: PinnedGeometry.linearValueDim)
        }
        /// The full-attention mixer. `attention_bias` is false, the query and
        /// key norms are per head dimension, and the query projection carries
        /// its output gate.
        func addFullAttentionMixer(_ prefix: String) {
            addPacked(
                "\(prefix).q_proj",
                outFeatures: PinnedGeometry.queryProjectionDim, inFeatures: hidden)
            for projection in ["k_proj", "v_proj"] {
                addPacked(
                    "\(prefix).\(projection)",
                    outFeatures: PinnedGeometry.keyValueDim, inFeatures: hidden)
            }
            addPacked(
                "\(prefix).o_proj", outFeatures: hidden,
                inFeatures: PinnedGeometry.attentionDim)
            for norm in ["q_norm", "k_norm"] {
                add("\(prefix).\(norm).weight", .f32, [PinnedGeometry.headDim])
            }
        }

        addPacked(
            "language_model.model.embed_tokens", outFeatures: vocab,
            inFeatures: hidden)
        add("language_model.model.norm.weight", .f32, [hidden])
        // The head is UNTIED on this pack.
        addPacked("language_model.lm_head", outFeatures: vocab, inFeatures: hidden)

        for (index, kind) in PinnedGeometry.layerTypes.enumerated() {
            let prefix = "\(layerPrefix)\(index)"
            add("\(prefix).input_layernorm.weight", .f32, [hidden])
            add("\(prefix).post_attention_layernorm.weight", .f32, [hidden])
            addMLP("\(prefix).mlp")
            switch kind {
            case .linearAttention:
                addLinearAttentionMixer("\(prefix).linear_attn")
            case .fullAttention:
                addFullAttentionMixer("\(prefix).self_attn")
            }
        }
        return inventory
    }

    static func validateSelectedTensors(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        quantization: PrismHadamardQuantizationSpec
    ) throws {
        // Affine quantization legitimately ships `.biases`, so that suffix is
        // NOT forbidden here.
        let forbiddenSuffixes = [
            ".weight_packed",
            ".input_global_scale",
            ".weight_global_scale",
            ".k_scale",
            ".v_scale",
        ]
        if let forbiddenName = selectedKeys.sorted().first(where: { name in
            forbiddenSuffixes.contains { suffix in name.hasSuffix(suffix) }
        }) {
            throw MLXFastError.invalidInput(
                "Bonsai 2 MLX transform rejects compressed-tensors/global-scale "
                    + "and FP8 KV-scale tensor \(forbiddenName)")
        }
        // The head is UNTIED on this pack, so `lm_head.*` is a real quadruple
        // the transform MUST select.
        for component in ["weight", "scales", "biases", "signs"] {
            let name = "language_model.lm_head.\(component)"
            guard selectedKeys.contains(name) else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 has an untied output head and must select \(name)")
            }
        }
        // The pack declares `mtp_num_hidden_layers: 0` and ships no head. The
        // track's head is a SEPARATE pinned export, so a pack that carries one
        // is a different artifact and is refused rather than absorbed.
        if let embedded = selectedKeys.sorted().first(where: {
            $0.contains(".mtp.") || $0.hasPrefix("mtp.")
        }) {
            throw MLXFastError.invalidInput(
                "Bonsai 2 declares no embedded multi-token-prediction head, but "
                    + "the checkpoint carries \(embedded); this track's head is a "
                    + "separate pinned export")
        }

        for name in selectedKeys.sorted() where name.hasSuffix(".weight") {
            let stem = String(name.dropLast(".weight".count))
            let scalesName = "\(stem).scales"
            let biasesName = "\(stem).biases"
            let signsName = "\(stem).signs"
            guard selectedKeys.contains(scalesName)
                || selectedKeys.contains(biasesName)
                || selectedKeys.contains(signsName)
            else {
                continue
            }
            guard selectedKeys.contains(scalesName),
                selectedKeys.contains(biasesName),
                selectedKeys.contains(signsName)
            else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 packed module \(stem) must ship .scales, .biases "
                        + "and .signs")
            }
            let weightInfo = try tensorInfo(named: name, index: index, headers: headers)
            let scalesInfo = try tensorInfo(named: scalesName, index: index, headers: headers)
            let biasesInfo = try tensorInfo(named: biasesName, index: index, headers: headers)
            let signsInfo = try tensorInfo(named: signsName, index: index, headers: headers)
            guard weightInfo.dtype == TensorDType.u32.rawValue,
                scalesInfo.dtype == TensorDType.f16.rawValue,
                biasesInfo.dtype == TensorDType.f16.rawValue,
                signsInfo.dtype == TensorDType.f32.rawValue,
                weightInfo.shape.count == 2,
                scalesInfo.shape == biasesInfo.shape,
                scalesInfo.shape.count == 2,
                signsInfo.shape.count == 1,
                weightInfo.shape[0] == scalesInfo.shape[0],
                weightInfo.shape.allSatisfy({ $0 > 0 }),
                scalesInfo.shape.allSatisfy({ $0 > 0 }),
                signsInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 packed module \(stem) has incompatible weight, "
                        + "scale, bias or sign metadata")
            }

            let packedWidth = weightInfo.shape[1]
            let groupCount = scalesInfo.shape[1]
            let (inputFeatures, inputOverflow) =
                groupCount.multipliedReportingOverflow(by: quantization.groupSize)
            let (packedBits, packedOverflow) =
                packedWidth.multipliedReportingOverflow(by: 32)
            guard !inputOverflow, !packedOverflow else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 module \(stem) packed width overflows Int")
            }
            let (expectedPackedBits, expectedOverflow) =
                inputFeatures.multipliedReportingOverflow(by: quantization.bits)
            guard !expectedOverflow, packedBits == expectedPackedBits else {
                throw MLXFastError.invalidInput(
                    "packed Bonsai 2 module \(stem) stored width \(packedWidth) "
                        + "does not match config quantization group_size "
                        + "\(quantization.groupSize) bits \(quantization.bits) for "
                        + "input dimension \(inputFeatures)")
            }
            // THE SIGNS COVER THE WHOLE CONTRACTED AXIS. A vector the width of
            // one Hadamard block would transform the first block and leave the
            // rest unrotated, which reads as a subtly wrong model rather than a
            // failure.
            guard signsInfo.shape[0] == inputFeatures,
                inputFeatures.isMultiple(of: PinnedGeometry.hadamardBlockSize)
            else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 module \(stem) declares \(signsInfo.shape[0]) signs "
                        + "for input dimension \(inputFeatures); the sign vector "
                        + "covers the whole contracted axis, in whole "
                        + "\(PinnedGeometry.hadamardBlockSize)-wide blocks")
            }
        }

        try validateExactPublicInventory(
            selectedKeys: selectedKeys, index: index, headers: headers)
    }

    static func validateExactPublicInventory(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws {
        let expected = expectedTensorInventory()
        let expectedNames = Set(expected.keys)
        // The pack ships a vision tower the transform drops, so the selected
        // set is a SUBSET of the index rather than equal to it. Every selected
        // name must still be indexed exactly once and present in exactly one
        // header.
        let headerNameList = headers.values.flatMap { $0.tensors.keys }
        let headerNames = Set(headerNameList)

        guard selectedKeys == expectedNames,
            expectedNames.isSubset(of: Set(index.weightMap.keys)),
            expectedNames.isSubset(of: headerNames),
            headerNameList.count == headerNames.count
        else {
            let missing = expectedNames.subtracting(selectedKeys).sorted()
            let extra = selectedKeys.subtracting(expectedNames).sorted()
            let unindexed = expectedNames
                .subtracting(Set(index.weightMap.keys)).sorted()
            throw MLXFastError.invalidInput(
                "Bonsai 2 checkpoint tensor inventory must match the exact public "
                    + "\(expectedTensorCount)-tensor contract "
                    + "(missing: \(missing.prefix(8).joined(separator: ", ")); "
                    + "extra: \(extra.prefix(8).joined(separator: ", ")); "
                    + "unindexed/duplicate header tensors: "
                    + "\(unindexed.prefix(8).joined(separator: ", ")))")
        }

        for name in expected.keys.sorted() {
            guard let expectedMetadata = expected[name] else {
                preconditionFailure("missing expected Bonsai 2 metadata for \(name)")
            }
            let actual = try tensorInfo(named: name, index: index, headers: headers)
            guard actual.dtype == expectedMetadata.dtype,
                actual.shape == expectedMetadata.shape
            else {
                throw MLXFastError.invalidInput(
                    "Bonsai 2 tensor \(name) is \(actual.dtype) \(actual.shape); "
                        + "the pinned inventory declares \(expectedMetadata.dtype) "
                        + "\(expectedMetadata.shape)")
            }
        }
    }

    private static func tensorInfo(
        named name: String,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws -> SafetensorInfo {
        guard let shardName = index.weightMap[name],
            let info = headers[shardName]?.tensors[name]
        else {
            throw MLXFastError.invalidInput(
                "missing validated tensor metadata for \(name)")
        }
        return info
    }

    private static func intField(
        _ key: String, in object: [String: Any]
    ) throws -> Int {
        guard let number = object[key] as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID(),
            !CFNumberIsFloatType(number),
            let integer = Int(number.stringValue)
        else {
            throw MLXFastError.invalidInput(
                "Bonsai 2 quantization field \(key) must be a finite integer in "
                    + "Int range")
        }
        return integer
    }

    private static func stringField(
        _ key: String, in object: [String: Any]
    ) throws -> String {
        guard let string = object[key] as? String else {
            throw MLXFastError.invalidInput(
                "Bonsai 2 quantization field \(key) must be a string")
        }
        return string
    }
}
