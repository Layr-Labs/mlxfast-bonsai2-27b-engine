// Adapted from PrismML-Eng/mlx-swift 6d3a84de28225d1f5bc0a56f5c781596997242f9 (MIT).
// Preserve the published Bonsai pack's FP32 transform / original output dtype contract.
import Foundation
@_spi(QuantizedConstantCache) import MLX

/// Invalid transform metadata or incompatible packed weights.
public enum HadamardError: Error {
    case invalidBlockSize
    case invalidSigns
    case incompatibleShape
    case unsupportedContract
    case missingSignWidth(Int)
}

/// A normalized block Walsh-Hadamard transform with explicit input signs.
///
/// Forward evaluation computes `(x * signs) H`; inverse evaluation computes
/// `(x H) * signs`. Signs cover the entire final dimension, not just one block.
public struct SignedBlockHadamard {
    public let blockSize: Int
    public let width: Int
    private let signs: MLXArray
    private let signValues: [Float]

    public init(blockSize: Int, signs: [Float]) throws {
        guard blockSize > 0, blockSize <= 8192,
            blockSize & (blockSize - 1) == 0
        else { throw HadamardError.invalidBlockSize }
        guard !signs.isEmpty, signs.count % blockSize == 0,
            signs.allSatisfy({ $0 == -1 || $0 == 1 })
        else { throw HadamardError.invalidSigns }
        self.blockSize = blockSize
        self.width = signs.count
        self.signs = MLXArray(signs)
        self.signValues = signs
    }

    /// Check serialized signs against the independently decoded metadata.
    public func matches(signs values: [Float]) -> Bool { values == signValues }

    /// True when both transforms compute the same function. Transforms decoded
    /// for one width share one sign buffer, so the common case is O(1).
    public func isIdentical(to other: SignedBlockHadamard) -> Bool {
        blockSize == other.blockSize && width == other.width
            && (signs === other.signs || signValues == other.signValues)
    }

    /// Transform activations before multiplication by folded weights.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        validate(x)
        return hadamardTransform((x.asType(.float32) * signs).reshaped([-1, blockSize]))
            .reshaped(x.shape).asType(x.dtype)
    }

    /// The sign vector as an array, for a caller that folds the sign flip into
    /// an elementwise op it already runs on the activation. Read only.
    public var signVector: MLXArray { signs }

    /// The forward transform of an activation that already carries the signs
    /// (`x * signVector`, in FP32). Identical to `callAsFunction` on the
    /// unsigned activation; the multiply has simply been done by the caller.
    public func applyPreSigned(_ signed: MLXArray) -> MLXArray {
        validate(signed)
        return hadamardTransform(signed.asType(.float32).reshaped([-1, blockSize]))
            .reshaped(signed.shape).asType(signed.dtype)
    }

    /// Recover the original basis after looking up folded embedding rows.
    public func inverse(_ x: MLXArray) -> MLXArray {
        validate(x)
        return (hadamardTransform(x.asType(.float32).reshaped([-1, blockSize])).reshaped(x.shape)
            * signs).asType(x.dtype)
    }

    private func validate(_ x: MLXArray) {
        precondition(x.ndim > 0 && x.dim(-1) == width, "Hadamard input width mismatch")
        precondition(
            [DType.float32, .float16, .bfloat16].contains(x.dtype),
            "Hadamard input must have a real floating-point dtype")
    }
}

/// The version-1 `prism.hadamard.*` metadata exported as a JSON object.
///
/// This decodes extracted metadata, not GGUF bytes. Tensor names remain in the
/// source namespace. A model loader must map them to its modules and honor
/// `gdnVGrouped` when arranging GDN values before an output projection.
public struct PrismHadamardConfiguration: Decodable {
    public let blockSize: Int
    public let weightNames: [String]
    public let inverseWeightNames: [String]
    public let gdnVGrouped: Bool
    private let transforms: [Int: SignedBlockHadamard]

    private enum CodingKeys: String, CodingKey {
        case version = "prism.hadamard.version"
        case blockSize = "prism.hadamard.block_size"
        case transform = "prism.hadamard.transform"
        case axis = "prism.hadamard.axis"
        case signMode = "prism.hadamard.sign_mode"
        case weightNames = "prism.hadamard.weight_names"
        case inverseWeightNames = "prism.hadamard.inverse_weight_names"
        case signWidths = "prism.hadamard.sign_widths"
        case signValues = "prism.hadamard.sign_values"
        case gdnVGrouped = "prism.hadamard.gdn_v_grouped"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard try c.decode(Int.self, forKey: .version) == 1,
            try c.decode(String.self, forKey: .transform) == "normalized-sylvester-walsh-hadamard",
            try c.decode(String.self, forKey: .axis) == "input-last-dimension",
            try c.decode(String.self, forKey: .signMode) == "explicit"
        else { throw HadamardError.unsupportedContract }
        blockSize = try c.decode(Int.self, forKey: .blockSize)
        weightNames = try c.decode([String].self, forKey: .weightNames)
        inverseWeightNames = try c.decodeIfPresent([String].self, forKey: .inverseWeightNames) ?? []
        gdnVGrouped = try c.decodeIfPresent(Bool.self, forKey: .gdnVGrouped) ?? false
        guard Set(weightNames).count == weightNames.count,
            Set(inverseWeightNames).count == inverseWeightNames.count,
            Set(weightNames).isDisjoint(with: inverseWeightNames),
            (weightNames + inverseWeightNames).allSatisfy({ !$0.isEmpty })
        else { throw HadamardError.unsupportedContract }
        let widths = try c.decode([Int].self, forKey: .signWidths)
        let values = try c.decode([Float].self, forKey: .signValues)
        guard !widths.isEmpty, Set(widths).count == widths.count else {
            throw HadamardError.invalidSigns
        }
        var offset = 0
        var transforms = [Int: SignedBlockHadamard]()
        for width in widths {
            guard width > 0, width <= values.count - offset else {
                throw HadamardError.invalidSigns
            }
            transforms[width] = try SignedBlockHadamard(
                blockSize: blockSize, signs: Array(values[offset ..< offset + width]))
            offset += width
        }
        guard offset == values.count else { throw HadamardError.invalidSigns }
        self.transforms = transforms
    }

    public func transform(forWidth width: Int) throws -> SignedBlockHadamard {
        guard let transform = transforms[width] else {
            throw HadamardError.missingSignWidth(width)
        }
        return transform
    }
}

private func validateHadamardWeights(
    _ weight: MLXArray, scales: MLXArray, biases: MLXArray?,
    groupSize: Int, bits: Int, transform: SignedBlockHadamard
) throws {
    guard [2, 3, 4, 5, 6, 8].contains(bits), [32, 64, 128].contains(groupSize),
        weight.ndim == 2, weight.dtype == .uint32,
        weight.dim(0) > 0, weight.dim(1) == transform.width / 32 * bits,
        transform.width % 32 == 0, transform.width % groupSize == 0,
        scales.shape == [weight.dim(0), transform.width / groupSize],
        [DType.float32, .float16, .bfloat16].contains(scales.dtype),
        biases == nil || (biases!.shape == scales.shape && biases!.dtype == scales.dtype)
    else { throw HadamardError.incompatibleShape }
}

/// Reorders tiled GDN values into the grouped order of folded output weights.
///
/// The final axis changes from `[repeat, keyHead, headDimension]` to
/// `[keyHead, repeat, headDimension]` before signs and Hadamard are applied.
/// Use only when the upstream GDN produces tiled values; already grouped values
/// must not be permuted again.
public struct HadamardGDNLayout {
    public let width: Int
    public let keyHeads: Int
    public let valueHeads: Int

    public init(width: Int, keyHeads: Int, valueHeads: Int) throws {
        guard width > 0, keyHeads > 0, valueHeads > 0,
            valueHeads % keyHeads == 0, width % valueHeads == 0
        else { throw HadamardError.incompatibleShape }
        self.width = width
        self.keyHeads = keyHeads
        self.valueHeads = valueHeads
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim > 0 && x.dim(-1) == width, "GDN input width mismatch")
        let repeats = valueHeads / keyHeads
        if repeats == 1 { return x }
        return x.reshaped([-1, repeats, keyHeads, width / valueHeads])
            .transposed(0, 2, 1, 3).reshaped(x.shape)
    }
}


/// Input-independent operands for the matrix-regime route of a packed
/// projection: the same packed weight viewed with a leading expert axis of
/// one, its FP32-widened constants in that view, and the zero expert index.
/// A plain class, never a Module or an MLXArray, so reflecting the owning
/// layer cannot add any of these to the parameter tree. Nothing here depends
/// on a request; it is keyed on the layer's own frozen constants.
private final class HadamardMatrixRouteOperands {
    private let lock = NSLock()
    /// Sibling projections fused along their output axis; see
    /// `HadamardFusedSiblings`. Owned by the first sibling's operands.
    var fusedSiblings: HadamardFusedSiblings?
    private var weightSource: MLXArray?
    private var weight3: MLXArray?
    private var scalesSource: MLXArray?
    private var scales3: MLXArray?
    private var biasesSource: MLXArray?
    private var biases3: MLXArray?
    private var expertIndex: MLXArray?
    let scaleCache = ConstantArrayCastCache()
    let offsetCache = ConstantArrayCastCache()

    func clear() {
        lock.withLock {
            fusedSiblings = nil
            weightSource = nil
            weight3 = nil
            scalesSource = nil
            scales3 = nil
            biasesSource = nil
            biases3 = nil
        }
        scaleCache.clear()
        offsetCache.clear()
    }

    /// The stacked operand for exactly these siblings, built on first use and
    /// rebuilt only when a sibling or its weight object changes.
    func fusedSiblings(for siblings: [HadamardQuantizedLinear]) -> HadamardFusedSiblings {
        lock.withLock {
            if let fusedSiblings, fusedSiblings.matches(siblings) {
                return fusedSiblings
            }
            let built = HadamardFusedSiblings(siblings)
            fusedSiblings = built
            return built
        }
    }

    /// The gather operands: `weight` as `[1, N, packedK]`, `scales` and
    /// `biases` as `[1, N, groups]`, and a single `uint32` zero index. The
    /// reshapes are views of arrays that already exist; they are rebuilt only
    /// when the source array object changes.
    func gatherOperands(weight: MLXArray, scales: MLXArray, biases: MLXArray?)
        -> (weight: MLXArray, scales: MLXArray, biases: MLXArray?, index: MLXArray)
    {
        lock.withLock {
            if weightSource !== weight || weight3 == nil {
                weight3 = weight.reshaped([1] + weight.shape)
                weightSource = weight
            }
            if scalesSource !== scales || scales3 == nil {
                scales3 = scales.reshaped([1] + scales.shape)
                scalesSource = scales
            }
            if let biases {
                if biasesSource !== biases || biases3 == nil {
                    biases3 = biases.reshaped([1] + biases.shape)
                    biasesSource = biases
                }
            } else {
                biases3 = nil
                biasesSource = nil
            }
            if expertIndex == nil {
                expertIndex = MLXArray([UInt32(0)])
            }
            return (weight3!, scales3!, biases3, expertIndex!)
        }
    }
}


/// Several packed projections that read one rotated activation, stacked along
/// their output axis into one packed operand: the rows of `weight`, `scales`
/// and `biases` are the siblings' rows in order, byte for byte. One matmul
/// then replaces one per sibling, and the wide result is split back. The
/// stacking copies packed rows; it does not unpack, requantize or re-scale
/// anything. A plain class for the same reason as the route operands.
private final class HadamardFusedSiblings {
    let siblingIDs: [ObjectIdentifier]
    let weightSources: [MLXArray]
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    /// Cumulative output boundaries; the last entry is the total width.
    let boundaries: [Int]
    let operands = HadamardMatrixRouteOperands()

    init(_ siblings: [HadamardQuantizedLinear]) {
        siblingIDs = siblings.map { ObjectIdentifier($0) }
        weightSources = siblings.map(\.weight)
        weight = concatenated(siblings.map(\.weight), axis: 0)
        scales = concatenated(siblings.map(\.scales), axis: 0)
        if siblings.allSatisfy({ $0.biases != nil }) {
            biases = concatenated(siblings.map { $0.biases! }, axis: 0)
        } else {
            biases = nil
        }
        var edges = [Int]()
        var total = 0
        for sibling in siblings {
            total += sibling.weight.dim(0)
            edges.append(total)
        }
        boundaries = edges
    }

    /// True when this stack was built from exactly these layers holding
    /// exactly these weight objects.
    func matches(_ siblings: [HadamardQuantizedLinear]) -> Bool {
        guard siblings.count == siblingIDs.count else { return false }
        for (index, sibling) in siblings.enumerated() {
            guard ObjectIdentifier(sibling) == siblingIDs[index],
                sibling.weight === weightSources[index]
            else { return false }
        }
        return true
    }
}

/// Affine packed linear weights with a signed Hadamard input transform.
///
/// Pass weights already folded and packed in MLX format. The initializer does
/// not requantize them and does not accept raw GGUF quantization blocks.
public final class HadamardQuantizedLinear: QuantizedLinear {
    public let transform: SignedBlockHadamard
    public let gdnLayout: HadamardGDNLayout?
    /// On unless explicitly disabled. The native packed operator widens the
    /// FP16 scales and offsets to FP32 on every call when the rotated input
    /// is FP32; reuse keeps that exact widening instead of recomputing it.
    private static let reuseFloat16Constants: Bool = {
        let value = ProcessInfo.processInfo.environment[
            "DARKBLOOM_BONSAI_F16_CONSTANT_CACHE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Qualification witness; this selects reuse, never a different precision
    /// or packed-matmul kernel. The process-wide generic cache kill switch also
    /// remains effective. No cached arrays enter the module parameter tree.
    var permitsFloat16ConstantReuse: Bool {
        Self.reuseFloat16Constants && bits == 2 && groupSize == 128
            && transform.blockSize == 1024 && scales.dtype == .float16
    }

    public init(
        weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, transform: SignedBlockHadamard,
        gdnLayout: HadamardGDNLayout? = nil
    ) throws {
        try validateHadamardWeights(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, transform: transform)
        guard
            bias == nil
                || (bias!.shape == [weight.dim(0)]
                    && [DType.float32, .float16, .bfloat16].contains(bias!.dtype))
        else {
            throw HadamardError.incompatibleShape
        }
        guard gdnLayout == nil || gdnLayout!.width == transform.width else {
            throw HadamardError.incompatibleShape
        }
        self.gdnLayout = gdnLayout
        self.transform = transform
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits)
        freeze()
        FewRowPackedMatmul.warm(
            weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        applyRotated(rotate(x))
    }

    /// The input transform alone: GDN layout, signs, Hadamard, dtype restore.
    public func rotate(_ x: MLXArray) -> MLXArray {
        transform(gdnLayout.map { $0(x) } ?? x)
    }

    /// The packed matmul on an input already passed through `rotate`.
    public func applyRotated(_ rotated: MLXArray) -> MLXArray {
        if let routed = matrixRegimeForward(rotated) {
            return routed
        }
        if mode == .affine,
            let y = FewRowPackedMatmul.apply(
                rotated, weight, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits)
        {
            return bias.map { y + $0.asType(y.dtype) } ?? y
        }
        if permitsFloat16ConstantReuse && rotated.dtype == .float32 {
            return constantCachedForward(rotated, allowFloat16: true)
        }
        return super.callAsFunction(rotated)
    }

    // MARK: - Matrix-regime route

    /// On unless explicitly disabled. See `matrixRegimeForward`.
    private static let matrixRouteEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment[
            "DARKBLOOM_BONSAI_MATRIX_ROUTE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The dtype the packed matmul reads its rotated activation in on the
    /// matrix-regime route. FP16 is the published Prism runtime's own choice
    /// for this pack (its packed constants are FP16, so nothing is widened);
    /// `DARKBLOOM_BONSAI_PACKED_INPUT=float32` keeps the FP32 read and the
    /// FP32-widened constants instead.
    private static let matrixRouteInputDType: DType = {
        let value = ProcessInfo.processInfo.environment[
            "DARKBLOOM_BONSAI_PACKED_INPUT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch value {
        case "float32", "fp32", "f32": return .float32
        default: return .float16
        }
    }()

    /// The core's vector-versus-matrix threshold for this pack's shapes on the
    /// M5 generation: fewer rows than this take the scalar vector kernel.
    private static let matrixRegimeMinimumRows = 13
    /// A projection at least this wide is a vocabulary head. Its products are
    /// logits that an argmax reads directly, so it keeps the FP32 read (TF32
    /// tensor products, FP32 logits) and the cached widened constants; only
    /// the tower's projections take the FP16 read.
    private static let vocabularyHeadMinimumRows = 65536
    /// The core splits K whenever the 32x32 tile count is at most this.
    private static let splitKTileCeiling = 256

    private let matrixRoute = HadamardMatrixRouteOperands()

    @discardableResult
    public override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        matrixRoute.clear()
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    /// The packed matmul for a multi-row input, routed onto the M5 matrix
    /// kernels for every projection of the tower.
    ///
    /// The core dispatch (`QuantizedMatmul::eval_gpu`) sends fewer than 13
    /// rows to the scalar `qmv_wide` kernel, which pays a device load and an
    /// FMA per weight per row, and sends 13 or more rows either to the tensor
    /// `qmm_t_nax` kernel or, for a projection whose 32x32 tile count is 256 or
    /// fewer (k, v, o, z, out and down on this pack at a verify width), to the
    /// FP32 split-K `qmm_t_splitk` kernel plus a reduction. Both scalar paths
    /// are ALU-bound at a verify width; the weight read is not the cost.
    ///
    /// This route keeps the same weights and only changes which kernel sees
    /// them and in which activation dtype:
    ///
    /// - rows below the threshold are zero-padded up to it, so the core takes
    ///   the matrix path (the padded rows are dropped from the result);
    /// - a projection the core would split-K instead goes through the gather
    ///   form of the same operator with one expert and the zero index, whose
    ///   dispatch has no split-K branch and reaches `gather_qmm_t_nax` directly;
    /// - the rotated activation is read in `matrixRouteInputDType`. With FP16
    ///   the packed FP16 constants are used as stored and the FP16 result is
    ///   widened back to the incoming FP32, so every consumer sees the dtype
    ///   it saw before. With FP32 the existing widened constants are reused.
    ///
    /// The tensor kernels already round an FP32 input to a 10-bit mantissa,
    /// which is FP16's mantissa, so the FP16 read changes the range and the
    /// output rounding rather than the product precision. The token gate
    /// prices that. The vocabulary head is the exception: its products are
    /// the logits, so it keeps the FP32 read and FP32 logits. A BF16 input
    /// (the drafter reading the shared head) is widened to FP32 exactly, as
    /// the core would, and then reuses the cached widened constants instead
    /// of casting them on every call. Returns nil when the route does not
    /// apply.
    private func matrixRegimeForward(_ x: MLXArray, widenOutput: Bool = true) -> MLXArray? {
        guard Self.routeApplies(to: self), x.dtype == .float32 || x.dtype == .bfloat16,
            x.ndim >= 2
        else { return nil }
        let k = x.dim(-1)
        guard x.size / k >= 2, k % 64 == 0, k % groupSize == 0 else { return nil }
        return Self.matrixRoutedMatmul(
            x, weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode, operands: matrixRoute,
            widenOutput: widenOutput)
    }

    /// `callAsFunction` for a consumer that promotes dtypes itself, such as
    /// the residual add: when the route applies, the FP16 product is returned
    /// as is instead of being widened first. The consumer's promotion widens
    /// the same values exactly, so the arithmetic is unchanged and one cast
    /// dispatch per call is saved.
    public func forwardUnwidened(_ x: MLXArray) -> MLXArray {
        let rotated = rotate(x)
        return matrixRegimeForward(rotated, widenOutput: false) ?? applyRotated(rotated)
    }

    /// The projection of an activation that already carries the transform's
    /// signs (see `SignedBlockHadamard.applyPreSigned`), optionally leaving the
    /// FP16 product unwidened. Not for a layer with a GDN layout.
    public func forwardPreSigned(_ signed: MLXArray, widenOutput: Bool = true) -> MLXArray {
        precondition(gdnLayout == nil, "pre-signed forward needs an ungrouped layout")
        let rotated = transform.applyPreSigned(signed)
        if !widenOutput, let routed = matrixRegimeForward(rotated, widenOutput: false) {
            return routed
        }
        return applyRotated(rotated)
    }

    /// The representation the route handles: the pack's 2-bit affine layout
    /// with FP16 constants, no linear bias, and a 64-aligned output width.
    private static func routeApplies(to layer: HadamardQuantizedLinear) -> Bool {
        matrixRouteEnabled && layer.mode == .affine && layer.bits == 2 && layer.bias == nil
            && layer.scales.dtype == .float16 && layer.weight.dim(0) % 64 == 0
    }

    /// The routed matmul over `x` (any leading shape, `[..., K]`) with the
    /// given packed operand; returns `[..., N]` in `x.dtype`.
    private static func matrixRoutedMatmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode,
        operands: HadamardMatrixRouteOperands, widenOutput: Bool = true
    ) -> MLXArray {
        let k = x.dim(-1)
        let rows = x.size / k
        let n = weight.dim(0)

        // BF16 activations (the drafter's head input) widen to FP32 exactly,
        // as the core's own promotion would; a vocabulary head keeps FP32.
        let inputDType: DType =
            (n >= vocabularyHeadMinimumRows || x.dtype == .bfloat16)
            ? .float32 : matrixRouteInputDType
        let routeScales: MLXArray
        let routeBiases: MLXArray?
        if inputDType == .float32 {
            routeScales =
                operands.scaleCache.cachedCast(scales, to: .float32, allowFloat16: true)
                ?? scales
            routeBiases = biases.map { offsets in
                operands.offsetCache.cachedCast(offsets, to: .float32, allowFloat16: true)
                    ?? offsets
            }
        } else {
            routeScales = scales
            routeBiases = biases
        }

        let paddedRows = max(rows, matrixRegimeMinimumRows)
        var input = x.reshaped(rows, k)
        if inputDType != x.dtype {
            input = input.asType(inputDType)
        }
        if paddedRows > rows {
            input = concatenated(
                [input, MLXArray.zeros([paddedRows - rows, k], dtype: inputDType)], axis: 0)
        }

        let nTiles = (n + 31) / 32
        let mTiles = (paddedRows + 31) / 32
        var output: MLXArray
        if nTiles * mTiles > splitKTileCeiling {
            output = quantizedMM(
                input, weight, scales: routeScales, biases: routeBiases,
                transpose: true, groupSize: groupSize, bits: bits, mode: mode)
        } else if let fewRow = FewRowPackedMatmul.apply(
            x.reshaped(rows, k), weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits)
        {
            // A narrow output (o, out, down) has too few 64-column tiles to
            // fill the GPU on the gather tensor kernel; struffl's few-row
            // simdgroup kernel splits K instead and reads the FP16 constants
            // directly. It takes the unpadded rows in the incoming dtype and
            // returns that dtype.
            return fewRow.reshaped(Array(x.shape.dropLast()) + [n])
        } else {
            let gather = operands.gatherOperands(
                weight: weight, scales: routeScales, biases: routeBiases)
            output = gatherQuantizedMM(
                input.reshaped(1, paddedRows, k), gather.weight,
                scales: gather.scales, biases: gather.biases,
                rhsIndices: gather.index,
                transpose: true, groupSize: groupSize, bits: bits, mode: mode
            ).reshaped(paddedRows, n)
        }
        if paddedRows > rows {
            output = output[0 ..< rows]
        }
        // The core promotes a BF16 activation with FP16 constants to FP32 and
        // returns FP32; the widened result therefore matches what the plain
        // operator would have returned for either input dtype.
        let plainOutputDType: DType = x.dtype == .bfloat16 ? .float32 : x.dtype
        if widenOutput, output.dtype != plainOutputDType {
            output = output.asType(plainOutputDType)
        }
        return output.reshaped(Array(x.shape.dropLast()) + [n])
    }

    // MARK: - Fused siblings

    /// On unless explicitly disabled. See `fusedSiblingsForward`.
    private static let siblingFusionEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment[
            "DARKBLOOM_BONSAI_FUSE_SIBLINGS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// One routed matmul for several siblings that read the same rotated
    /// activation, over their packed rows stacked along the output axis, split
    /// back into one result per sibling.
    ///
    /// Two effects at a verify width. The stack has a wide output, so the core
    /// gives it the 32-row tensor kernel where a narrow sibling on its own
    /// (k, v, z) would have been split-K or, on this route, a 64-row gather
    /// tile. And one dispatch replaces one per sibling for the matmul and for
    /// each surrounding cast, with the rotated activation read once.
    ///
    /// Returns nil when the siblings do not all fit the route (a caller then
    /// applies each one to the rotated activation as before).
    fileprivate func fusedSiblingsForward(
        _ rotated: MLXArray, siblings: [HadamardQuantizedLinear], widenOutput: Bool = true
    ) -> [MLXArray]? {
        guard Self.siblingFusionEnabled, siblings.count >= 2,
            rotated.dtype == .float32, rotated.ndim >= 2
        else { return nil }
        let k = rotated.dim(-1)
        guard rotated.size / k >= 2, k % 64 == 0 else { return nil }
        for sibling in siblings {
            guard Self.routeApplies(to: sibling), sibling.groupSize == groupSize,
                sibling.weight.dim(1) == weight.dim(1), k % sibling.groupSize == 0
            else { return nil }
        }
        let fused = matrixRoute.fusedSiblings(for: siblings)
        let wide = Self.matrixRoutedMatmul(
            rotated, weight: fused.weight, scales: fused.scales, biases: fused.biases,
            groupSize: groupSize, bits: bits, mode: mode, operands: fused.operands,
            widenOutput: widenOutput)
        return MLX.split(wide, indices: Array(fused.boundaries.dropLast()), axis: -1)
    }

    /// True when `rotate` is the same function on both layers, so one rotated
    /// activation can feed both packed matmuls with bit-identical results.
    public func sharesInputTransform(with other: HadamardQuantizedLinear) -> Bool {
        gdnLayout == nil && other.gdnLayout == nil
            && transform.isIdentical(to: other.transform)
    }
}

/// Applies each packed Hadamard projection to the same activation, rotating it
/// once. Every projection reads the identical rotated array it would have
/// computed itself, so outputs are bit-identical to calling each one. Returns
/// nil when any projection is not packed or uses a different transform.
public func sharedHadamardProjections(
    _ x: MLXArray, _ projections: [Linear], widenOutput: Bool = true
) -> [MLXArray]? {
    guard let first = projections.first as? HadamardQuantizedLinear else { return nil }
    var packed = [HadamardQuantizedLinear]()
    packed.reserveCapacity(projections.count)
    for projection in projections {
        guard let layer = projection as? HadamardQuantizedLinear,
            layer.sharesInputTransform(with: first)
        else { return nil }
        packed.append(layer)
    }
    let rotated = first.rotate(x)
    if let fused = first.fusedSiblingsForward(
        rotated, siblings: packed, widenOutput: widenOutput)
    {
        return fused
    }
    return packed.map { $0.applyRotated(rotated) }
}

/// The packed projections that share one transform, when every one of them is
/// packed with that same transform and none has a GDN layout; nil otherwise.
public func sharedHadamardSiblings(_ projections: [Linear]) -> [HadamardQuantizedLinear]? {
    guard let first = projections.first as? HadamardQuantizedLinear else { return nil }
    var packed = [HadamardQuantizedLinear]()
    packed.reserveCapacity(projections.count)
    for projection in projections {
        guard let layer = projection as? HadamardQuantizedLinear,
            layer.sharesInputTransform(with: first)
        else { return nil }
        packed.append(layer)
    }
    return packed
}

/// Packed folded embeddings with an inverse transform after lookup.
///
/// `asLinear` applies the forward transform, allowing the same packed weights
/// to serve as a tied output projection without unfolding the full vocabulary.
public final class HadamardQuantizedEmbedding: Embedding, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode = .affine
    public let scales: MLXArray
    public let biases: MLXArray?
    public let transform: SignedBlockHadamard

    public override var shape: (Int, Int) { (weight.dim(0), transform.width) }

    public init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, transform: SignedBlockHadamard
    ) throws {
        try validateHadamardWeights(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, transform: transform)
        self.groupSize = groupSize
        self.bits = bits
        self.scales = scales
        self.biases = biases
        self.transform = transform
        super.init(weight: weight)
        freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let indices = x.flattened()
        let rows = dequantized(
            weight[indices], scales: scales[indices],
            biases: biases.map { $0[indices] }, groupSize: groupSize, bits: bits)
        return transform.inverse(rows).reshaped(x.shape + [transform.width])
    }

    public override func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            transform(x), weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits)
    }
}

/// A packed affine matmul for the few rows of a speculative verify. Each lane
/// turns two weight codes into halves straight inside a simdgroup matrix
/// fragment, and every fragment is multiplied against all of the rows, so
/// sixteen rows cost about what eight do. Scales and offsets apply once per
/// group to the partial sums, the offset through each group's activation sum.
enum FewRowPackedMatmul {
    static let rows = 4 ... 16
    static let rowBlocks = 4
    static let simdgroups = 2
    static let unroll = 2
    static let occupancy = 2048

    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_FEW_ROW_VERIFY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let warmed = NSLock()
    nonisolated(unsafe) private static var warmedShapes = Set<[Int]>()

    /// Compiles both row-block variants for one weight shape ahead of the
    /// first verify, so no speculative round pays a Metal library build.
    static func warm(
        _ w: MLXArray, scales: MLXArray, biases: MLXArray?, groupSize: Int, bits: Int
    ) {
        guard enabled, let biases else { return }
        let k = w.dim(1) * 32 / bits
        let key = [k, w.dim(0), bits, groupSize, scales.dtype.size]
        warmed.lock()
        let fresh = warmedShapes.insert(key).inserted
        warmed.unlock()
        guard fresh else { return }
        let weights = MLXArray.zeros(w.shape, dtype: w.dtype)
        let groups = MLXArray.zeros(scales.shape, dtype: scales.dtype)
        let offsets = MLXArray.zeros(biases.shape, dtype: biases.dtype)
        var outputs = [MLXArray]()
        for dtype in [DType.float32, .bfloat16, .float16] {
            for m in [8, 16] {
                if let y = apply(
                    MLXArray.zeros([m, k], dtype: dtype), weights, scales: groups,
                    biases: offsets, groupSize: groupSize, bits: bits)
                {
                    outputs.append(y)
                }
            }
        }
        eval(outputs)
    }

    static func apply(
        _ x: MLXArray, _ w: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int
    ) -> MLXArray? {
        guard enabled, let biases, x.ndim >= 2 else { return nil }
        let k = x.dim(-1)
        let m = x.size / k
        let n = w.dim(0)
        let rowsPerGroup = 8 * rowBlocks * simdgroups
        guard rows.contains(m), [2, 4, 8].contains(bits),
            groupSize % (64 * unroll) == 0, k % groupSize == 0,
            n % rowsPerGroup == 0, w.dim(1) * 32 == k * bits,
            [DType.float32, .float16, .bfloat16].contains(x.dtype)
        else { return nil }
        let groups = k / groupSize
        let blocks = (n / rowsPerGroup) * simdgroups
        let split = (1 ... groups).first { groups % $0 == 0 && blocks * $0 >= occupancy } ?? groups
        let y = kernel(
            [x.reshaped([m, k]), w, scales, biases, m],
            template: [
                ("K", k), ("N", n), ("BITS", bits), ("GS", groupSize),
                ("RB", rowBlocks), ("SGS", simdgroups), ("U", unroll), ("SPLIT", split),
                ("NB", (m + 7) / 8),
            ],
            grid: ((n / rowsPerGroup) * 32 * simdgroups, split, 1),
            threadGroup: (32 * simdgroups, 1, 1),
            outputShapes: [split > 1 ? [split, m, n] : [m, n]],
            outputDTypes: [.float32])[0]
        let out = split > 1 ? y.sum(axis: 0) : y
        return out.asType(x.dtype).reshaped(Array(x.shape.dropLast()) + [n])
    }

    private static let kernel = MLXFast.metalKernel(
        name: "mlxfast_few_row_packed_matmul",
        inputNames: ["x", "w", "scales", "biases", "M"],
        outputNames: ["y"],
        source: """
            constexpr uint CH = 16 * U;
            constexpr uint TK = 64 * U;
            constexpr uint row_bytes = K * BITS / 8;
            constexpr uint G = K / GS;
            constexpr uint C = 32 / BITS;
            constexpr uint D = 16 / BITS;
            constexpr uint RW = CH / C;
            constexpr uint pair = ((1u << BITS) - 1u) * 0x00010001u;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint qid = lane / 4;
            uint fm = (qid & 4) + ((lane / 2) % 4);
            uint fn = (qid & 2) * 2 + (lane % 2) * 2;
            uint row_base = (threadgroup_position_in_grid.x * SGS + sg) * 8 * RB;
            simdgroup_float8x8 acc[RB][NB];
            simdgroup_float8x8 part[RB][NB];
            UNROLL for (uint b = 0; b < RB; ++b) UNROLL for (uint c = 0; c < NB; ++c) {
                acc[b][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                part[b][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            device const uchar *wb = (device const uchar *)w;
            uint kl = (fn / 2) * CH;
            uint kb = (fm / 2) * CH + (fm % 2) * D;
            float sum0[NB], sum1[NB];
            UNROLL for (uint c = 0; c < NB; ++c) {
                sum0[c] = 0.0f; sum1[c] = 0.0f;
            }
            constexpr uint KS = K / SPLIT;
            uint split = threadgroup_position_in_grid.y;
            for (uint k0 = split * KS; k0 < (split + 1) * KS; k0 += TK) {
                uint words[RB][RW];
                UNROLL for (uint b = 0; b < RB; ++b) {
                    uint row = row_base + b * 8 + fm;
                    device const uint *p =
                        (device const uint *)(wb + row * row_bytes + ((k0 + kl) * BITS) / 8);
                    UNROLL for (uint j = 0; j < RW; ++j) words[b][j] = p[j];
                }
                UNROLL for (uint j = 0; j < CH / 2; ++j) {
                    uint k = k0 + kb + (j % D) + C * (j / D);
                    simdgroup_float8x8 bx[NB];
                    UNROLL for (uint c = 0; c < NB; ++c) {
                        float v0 = float(x[min(fn + 8 * c, uint(M) - 1) * K + k]);
                        float v1 = float(x[min(fn + 8 * c + 1, uint(M) - 1) * K + k]);
                        sum0[c] += v0; sum1[c] += v1;
                        bx[c].thread_elements()[0] = v0;
                        bx[c].thread_elements()[1] = v1;
                    }
                    UNROLL for (uint b = 0; b < RB; ++b) {
                        uint bitsv = (words[b][j / D] >> (BITS * (j % D))) & pair;
                        half2 q = as_type<half2>(bitsv | 0x64006400u) - half2(1024.0h);
                        simdgroup_half8x8 a;
                        a.thread_elements()[0] = q.x;
                        a.thread_elements()[1] = q.y;
                        UNROLL for (uint c = 0; c < NB; ++c)
                            simdgroup_multiply_accumulate(part[b][c], a, bx[c], part[b][c]);
                    }
                }
                if ((k0 + TK) % GS == 0) {
                    uint g = (k0 + TK) / GS - 1;
                    float s0[NB], s1[NB];
                    UNROLL for (uint c = 0; c < NB; ++c) {
                        s0[c] = sum0[c] + simd_shuffle_xor(sum0[c], 2);
                        s0[c] += simd_shuffle_xor(s0[c], 4);
                        s0[c] += simd_shuffle_xor(s0[c], 16);
                        s1[c] = sum1[c] + simd_shuffle_xor(sum1[c], 2);
                        s1[c] += simd_shuffle_xor(s1[c], 4);
                        s1[c] += simd_shuffle_xor(s1[c], 16);
                        sum0[c] = 0.0f; sum1[c] = 0.0f;
                    }
                    UNROLL for (uint b = 0; b < RB; ++b) {
                        uint gi = (row_base + b * 8 + fm) * G + g;
                        float sc = float(scales[gi]);
                        float bi = float(biases[gi]);
                        UNROLL for (uint c = 0; c < NB; ++c) {
                            acc[b][c].thread_elements()[0] += sc * part[b][c].thread_elements()[0] + bi * s0[c];
                            acc[b][c].thread_elements()[1] += sc * part[b][c].thread_elements()[1] + bi * s1[c];
                            part[b][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                        }
                    }
                }
            }
            UNROLL for (uint b = 0; b < RB; ++b) {
                uint row = row_base + b * 8 + fm;
                UNROLL for (uint c = 0; c < NB; ++c) {
                    if (fn + 8 * c < uint(M)) y[(split * uint(M) + fn + 8 * c) * N + row] = acc[b][c].thread_elements()[0];
                    if (fn + 8 * c + 1 < uint(M)) y[(split * uint(M) + fn + 8 * c + 1) * N + row] = acc[b][c].thread_elements()[1];
                }
            }
            """,
        header: """
            #include <metal_simdgroup_matrix>
            #define UNROLL _Pragma("clang loop unroll(full)")

            """)
}
