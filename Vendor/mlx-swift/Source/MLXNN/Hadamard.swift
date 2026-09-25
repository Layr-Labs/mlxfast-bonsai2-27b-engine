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

    /// The core's vector-versus-matrix threshold for this pack's shapes on the
    /// M5 generation: fewer rows than this take the scalar vector kernel.
    private static let matrixRegimeMinimumRows = 13
    /// Widest input this route handles. A seed prefill is far wider and keeps
    /// the core's own dispatch.
    private static let matrixRegimeMaximumRows = 64
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

    /// The packed matmul for a few-row input, routed onto the M5 matrix
    /// kernels for every projection of the tower.
    ///
    /// The core dispatch (`QuantizedMatmul::eval_gpu`) sends fewer than 13
    /// rows to the scalar `qmv_wide` kernel, which pays a device load and an
    /// FMA per weight per row, and sends 13 or more rows either to the tensor
    /// `qmm_t_nax` kernel or, for a projection whose 32x32 tile count is 256 or
    /// fewer (k, v, o, z, out and down on this pack), to the FP32 split-K
    /// `qmm_t_splitk` kernel plus a reduction. Both scalar paths are ALU-bound
    /// at a verify width; the weight read is not the cost.
    ///
    /// This route keeps the same weights, the same FP32 activations and the
    /// same widened FP32 constants, and only changes which kernel sees them:
    ///
    /// - rows below the threshold are zero-padded up to it, so the core takes
    ///   the matrix path (the padded rows are dropped from the result);
    /// - a projection the core would split-K instead goes through the gather
    ///   form of the same operator with one expert and the zero index, whose
    ///   dispatch has no split-K branch and reaches `gather_qmm_t_nax` directly.
    ///
    /// The tensor kernels round the FP32 input the way the core already does
    /// for every projection at the seed prefill and for the wide projections
    /// of a verify, so the change is priced by the token gate, not by a new
    /// precision. Returns nil when the route does not apply.
    private func matrixRegimeForward(_ x: MLXArray) -> MLXArray? {
        guard Self.matrixRouteEnabled, mode == .affine, bits == 2, bias == nil,
            x.dtype == .float32, x.ndim >= 2
        else { return nil }
        let k = x.dim(-1)
        let rows = x.size / k
        guard rows >= 2, rows <= Self.matrixRegimeMaximumRows else { return nil }
        let n = weight.dim(0)
        guard k % 64 == 0, n % 64 == 0, k % groupSize == 0 else { return nil }

        let widenedScales =
            matrixRoute.scaleCache.cachedCast(scales, to: .float32, allowFloat16: true)
            ?? scales
        let widenedBiases: MLXArray? = biases.map { offsets in
            matrixRoute.offsetCache.cachedCast(offsets, to: .float32, allowFloat16: true)
                ?? offsets
        }

        let paddedRows = max(rows, Self.matrixRegimeMinimumRows)
        var input = x.reshaped(rows, k)
        if paddedRows > rows {
            input = concatenated(
                [input, MLXArray.zeros([paddedRows - rows, k], dtype: x.dtype)], axis: 0)
        }

        let nTiles = (n + 31) / 32
        let mTiles = (paddedRows + 31) / 32
        var output: MLXArray
        if nTiles * mTiles > Self.splitKTileCeiling {
            output = quantizedMM(
                input, weight, scales: widenedScales, biases: widenedBiases,
                transpose: true, groupSize: groupSize, bits: bits, mode: mode)
        } else {
            let operands = matrixRoute.gatherOperands(
                weight: weight, scales: widenedScales, biases: widenedBiases)
            output = gatherQuantizedMM(
                input.reshaped(1, paddedRows, k), operands.weight,
                scales: operands.scales, biases: operands.biases,
                rhsIndices: operands.index,
                transpose: true, groupSize: groupSize, bits: bits, mode: mode
            ).reshaped(paddedRows, n)
        }
        if paddedRows > rows {
            output = output[0 ..< rows]
        }
        return output.reshaped(Array(x.shape.dropLast()) + [n])
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
public func sharedHadamardProjections(_ x: MLXArray, _ projections: [Linear]) -> [MLXArray]? {
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
    return packed.map { $0.applyRotated(rotated) }
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
