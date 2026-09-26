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

    /// A one-launch implementation of the forward transform, installed by a
    /// module that can build custom Metal kernels (the model file installs
    /// it at load). It receives the activation, the sign vector, the block
    /// size, whether the activation already carries the signs, an optional
    /// GDN layout to gather through, and the output dtype. It must compute
    /// exactly what the op chain computes (FP32 sign multiply, the FP32
    /// block transform with the stock kernel's butterfly order and scale,
    /// then one cast to the output dtype). Returns nil to decline.
    public typealias FusedTransform = (
        _ x: MLXArray, _ signs: MLXArray, _ blockSize: Int, _ preSigned: Bool,
        _ gdnLayout: HadamardGDNLayout?, _ outputDType: DType
    ) -> MLXArray?
    nonisolated(unsafe) public static var fusedTransform: FusedTransform?

    /// Transform activations before multiplication by folded weights.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        forward(x, gdnLayout: nil, outputDType: x.dtype)
    }

    /// The forward transform of `x` (optionally gathered through a GDN
    /// layout first), returned in `outputDType`. Same values as
    /// `callAsFunction(layout(x)).asType(outputDType)`.
    public func forward(_ x: MLXArray, gdnLayout: HadamardGDNLayout?, outputDType: DType)
        -> MLXArray
    {
        validate(x)
        if let fused = Self.fusedTransform,
            let y = fused(x, signs, blockSize, false, gdnLayout, outputDType)
        {
            return y
        }
        let laidOut = gdnLayout.map { $0(x) } ?? x
        let rotated = hadamardTransform(
            (laidOut.asType(.float32) * signs).reshaped([-1, blockSize])
        ).reshaped(x.shape).asType(x.dtype)
        return rotated.dtype == outputDType ? rotated : rotated.asType(outputDType)
    }

    /// `applyPreSigned((silu(gate) * up) * signVector)` with the SwiGLU product
    /// and the sign flip formed inside the fused rotation's read: the compiled
    /// `(silu(gate.asType(.float32)) * up.asType(.float32)) * signs` chain, each
    /// product rounded to FP32 in that order, MLX's `Sigmoid` verbatim. Gate
    /// and up may be FP32 or FP16 (widened exactly). Nil when it does not apply.
    public func rotatedSwiGLU(
        gate: MLXArray, up: MLXArray, outputDType: DType = .float32
    ) -> MLXArray? {
        guard FusedInputHadamardKernel.swigluEnabled, blockSize == 1024, width % 1024 == 0,
            gate.dtype == up.dtype, gate.dtype == .float32 || gate.dtype == .float16,
            gate.shape == up.shape, gate.ndim > 0, gate.dim(-1) == width
        else { return nil }
        return FusedInputHadamardKernel.gated(
            gate, up, signs: signs, width: width, mode: 1, outputDType: outputDType)
    }

    /// The layer boundary `h = x + r` (FP16 residual add), `rmsNorm(h)` over
    /// the FP32-promoted row with an FP32 gain, the signs and the transform,
    /// in ONE kernel that writes `h` and the rotation (in `outputDType`).
    ///
    /// The composed path is four dispatches: MLX's FP16 `Add`, the `AsType`
    /// that `fast::rms_norm` inserts to promote `h` to the gain's FP32, the
    /// `rms_looped` kernel (1024 threads, `N_READS = 4` for this width), and
    /// the rotation. The kernel keeps every operation's arithmetic: the add is
    /// a correctly rounded FP16 sum; the sum of squares follows `rms_looped`'s
    /// thread/element mapping and `acc += xi * xi` chain (both compiled without
    /// fast math, so contraction is decided alike), its `simd_sum` and 32-way threadgroup
    /// pass, then `precise::rsqrt(acc / width + eps)` and `w * (x * inv)`;
    /// the signs and MLX's `hadamard_n<float, 1024, 16, 4>` butterfly follow,
    /// and the result is rounded once at the store. With `keepNormed` the FP32
    /// norm output (unsigned, as `rmsNorm` returns it) is written as well, for
    /// consumers that read the activation outside the rotation. Nil when it
    /// does not apply.
    public func residualNormRotated(
        _ x: MLXArray, _ r: MLXArray, weight: MLXArray, eps: Float, outputDType: DType,
        keepNormed: Bool = false
    ) -> (h: MLXArray, normed: MLXArray?, rotated: MLXArray)? {
        guard FusedInputHadamardKernel.residualNormEnabled, blockSize == 1024, width == 5120,
            x.dtype == .float16, r.dtype == .float16, x.shape == r.shape,
            x.ndim >= 2, x.dim(-1) == width, weight.dtype == .float32, weight.ndim == 1,
            weight.dim(0) == width,
            outputDType == .float16 || outputDType == .float32
        else { return nil }
        let rows = x.size / width
        if keepNormed {
            let outputs = FusedInputHadamardKernel.residualNormKernelNormed(
                [x, r, weight, signs, MLXArray(eps)],
                template: [("OutT", outputDType)],
                grid: (1024 * rows, 1, 1),
                threadGroup: (1024, 1, 1),
                outputShapes: [x.shape, x.shape, x.shape],
                outputDTypes: [.float16, .float32, outputDType])
            return (outputs[0], outputs[1], outputs[2])
        }
        let outputs = FusedInputHadamardKernel.residualNormKernel(
            [x, r, weight, signs, MLXArray(eps)],
            template: [("OutT", outputDType)],
            grid: (1024 * rows, 1, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [x.shape, x.shape],
            outputDTypes: [.float16, outputDType])
        return (outputs[0], nil, outputs[1])
    }

    /// `rotatedSwiGLU` reading gate and up from the stacked gate|up product
    /// `wide` at `gateOffset` / `upOffset` (row-major, width `self.width` each).
    public func rotatedSwiGLUStacked(
        _ wide: MLXArray, gateOffset: Int, upOffset: Int, outputDType: DType
    ) -> MLXArray? {
        guard FusedInputHadamardKernel.stackedReads, blockSize == 1024, width % 1024 == 0,
            wide.dtype == .float32 || wide.dtype == .float16, wide.ndim >= 2,
            gateOffset >= 0, upOffset >= 0,
            wide.dim(-1) >= max(gateOffset, upOffset) + width
        else { return nil }
        return FusedInputHadamardKernel.gatedStacked(
            wide, width: width, aOffset: gateOffset, bOffset: upOffset, signs: signs, mode: 1,
            outputDType: outputDType)
    }

    /// `self(x * sigmoid(gate))`, the attention output gate, fused the same way.
    public func rotatedSigmoidGate(
        _ x: MLXArray, gate: MLXArray, outputDType: DType = .float32
    ) -> MLXArray? {
        guard FusedInputHadamardKernel.gateEnabled, blockSize == 1024, width % 1024 == 0,
            x.dtype == .float32, gate.dtype == .float32, x.shape == gate.shape,
            x.ndim > 0, x.dim(-1) == width
        else { return nil }
        return FusedInputHadamardKernel.gated(
            x, gate, signs: signs, width: width, mode: 2, outputDType: outputDType)
    }

    /// `rotatedSigmoidGate(x.reshaped(B, S, -1), gate.reshaped(B, S, -1))` for
    /// `[B, S, heads, headDim]` operands of any strides (see
    /// `HadamardQuantizedLinear.applyAfterSigmoidGateHeads`).
    public func rotatedSigmoidGateHeads(
        _ x: MLXArray, gate: MLXArray, outputDType: DType = .float32
    ) -> MLXArray? {
        guard FusedInputHadamardKernel.gateEnabled, FusedInputHadamardKernel.gateHeadsEnabled,
            blockSize == 1024, width % 1024 == 0,
            x.dtype == .float32, gate.dtype == .float32, x.shape == gate.shape,
            x.ndim == 4, x.dim(2) * x.dim(3) == width
        else { return nil }
        return FusedInputHadamardKernel.gatedHeads(
            x, gate, signs: signs, width: width, headDim: x.dim(3), outputDType: outputDType)
    }

    /// `self(layout((silu(z) * rmsNorm(x, weight, eps)).reshaped(width)))` for
    /// the GDN output: per-head RMSNorm exactly as MLX's `rms_single_row` over a
    /// 128-wide head (32 lanes x 4 reads, `simd_sum`, `precise::rsqrt`,
    /// `w * (x * inv)`), the compiled `silu(z) * normed` tail, the value-head
    /// permutation, the signs and the transform, in one kernel.
    public func rotatedGatedRMSNorm(
        _ x: MLXArray, gate z: MLXArray, weight: MLXArray, eps: Float,
        layout: HadamardGDNLayout?, outputDType: DType = .float32
    ) -> MLXArray? {
        guard x.ndim == 4 else { return nil }
        let headDim = x.dim(-1)
        let valueHeads = x.dim(-2)
        // No layout: heads stay in place (repeats 1). Grouped layout: the
        // value-head permutation is folded into the reads.
        let repeats = layout.map { $0.valueHeads / $0.keyHeads } ?? 1
        let keyHeads = layout?.keyHeads ?? valueHeads
        guard FusedInputHadamardKernel.gatedNormEnabled, blockSize == 1024,
            width == valueHeads * headDim, width % 1024 == 0,
            headDim == 128, 1024 % headDim == 0,
            layout == nil || (layout!.width == width && layout!.valueHeads == valueHeads),
            repeats * keyHeads == valueHeads, x.shape == z.shape, x.dtype == .float32,
            z.dtype == .float32 || z.dtype == .float16, weight.dtype == .float32,
            weight.ndim == 1,
            weight.dim(0) == headDim
        else { return nil }
        return FusedInputHadamardKernel.gatedRMSNorm(
            x, z, weight: weight, eps: eps, signs: signs,
            repeats: repeats, keyHeads: keyHeads, headDim: headDim,
            outputDType: outputDType)
    }

    /// The sign vector as an array, for a caller that folds the sign flip into
    /// an elementwise op it already runs on the activation. Read only.
    public var signVector: MLXArray { signs }

    /// The forward transform of an activation that already carries the signs
    /// (`x * signVector`, in FP32). Identical to `callAsFunction` on the
    /// unsigned activation; the multiply has simply been done by the caller.
    public func applyPreSigned(_ signed: MLXArray) -> MLXArray {
        applyPreSigned(signed, outputDType: signed.dtype)
    }

    /// `applyPreSigned` returned in `outputDType` (the cast folded into the
    /// transform when a fused implementation is installed).
    public func applyPreSigned(_ signed: MLXArray, outputDType: DType) -> MLXArray {
        validate(signed)
        if let fused = Self.fusedTransform,
            let y = fused(signed, signs, blockSize, true, nil, outputDType)
        {
            return y
        }
        let rotated = hadamardTransform(signed.asType(.float32).reshaped([-1, blockSize]))
            .reshaped(signed.shape).asType(signed.dtype)
        return rotated.dtype == outputDType ? rotated : rotated.asType(outputDType)
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
/// projection: its FP32-widened constants and, for the first of a group of
/// siblings, their stacked operand. A plain class, never a Module or an
/// MLXArray, so reflecting the owning layer cannot add any of these to the
/// parameter tree. Nothing here depends on a request; it is keyed on the
/// layer's own frozen constants.
private final class HadamardMatrixRouteOperands {
    private let lock = NSLock()
    /// Sibling projections fused along their output axis; see
    /// `HadamardFusedSiblings`. Owned by the first sibling's operands.
    var fusedSiblings: HadamardFusedSiblings?
    let scaleCache = ConstantArrayCastCache()
    let offsetCache = ConstantArrayCastCache()

    func clear() {
        lock.withLock { fusedSiblings = nil }
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
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        applyRotated(rotate(x))
    }

    /// The input transform alone: GDN layout, signs, Hadamard, dtype restore.
    public func rotate(_ x: MLXArray) -> MLXArray {
        transform.forward(x, gdnLayout: gdnLayout, outputDType: x.dtype)
    }

    /// `rotate` returned in `outputDType` (one cast folded into the transform
    /// when a fused implementation is installed).
    public func rotate(_ x: MLXArray, outputDType: DType) -> MLXArray {
        transform.forward(x, gdnLayout: gdnLayout, outputDType: outputDType)
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
    /// The split-K tensor body takes FP16 input for one 16-row half of its
    /// 32-row tile; a split-K projection over more rows keeps the FP32 read.
    private static let splitKHalfRowLimit = 16

    /// On unless explicitly disabled: a narrow (split-K) projection at a
    /// verify width reads its rotated activation in the route dtype too.
    private static let narrowHalfRead: Bool = {
        let value = ProcessInfo.processInfo.environment[
            "DARKBLOOM_BONSAI_NARROW_HALF"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

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
    /// FMA per weight per row, and 13 or more rows to the tensor `qmm_t_nax`
    /// kernel or, for a projection with at most 256 32x32 tiles, to the
    /// split-K kernel whose body is on the tensor unit for FP32 input. This
    /// route keeps the same weights and changes only what the kernels see:
    ///
    /// - rows below the threshold are zero-padded up to it, so the core takes
    ///   the matrix path (the padded rows are dropped from the result);
    /// - a tower projection reads its rotated activation in
    ///   `matrixRouteInputDType` (FP16: the packed FP16 constants are used as
    ///   stored, the FP16 result is widened back so every consumer sees the
    ///   dtype it saw before), a wide one on `qmm_t_nax` and a narrow one on
    ///   the split-K tensor body up to 16 rows; a narrow one over more rows and
    ///   the vocabulary head keep the FP32 read with the cached widened constants;
    /// - a BF16 input (the drafter reading the shared head) is widened to
    ///   FP32 exactly, as the core would, and reuses the cached widened
    ///   constants instead of casting them on every call.
    ///
    /// The tensor kernels already round an FP32 input to a 10-bit mantissa,
    /// which is FP16's mantissa, so the FP16 read changes the range and the
    /// output rounding rather than the product precision. The token gate
    /// prices that. Returns nil when the route does not apply.
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

    /// The dtype a fused-input rotation feeding this layer alone should store:
    /// the dtype the matrix route reads for `rows` rows of an FP32 activation
    /// (so the route never casts it again). Nil when the route does not apply.
    private func fusedInputStoreDType(rows: Int) -> DType? {
        let k = transform.width
        guard Self.routeApplies(to: self), rows >= 2, k % 64 == 0, k % groupSize == 0
        else { return nil }
        return Self.routeInputDType(rows: rows, n: weight.dim(0), sourceDType: .float32)
    }

    /// The routed matmul of a fused-input rotation stored in the route dtype
    /// on behalf of an FP32 activation (the FP32 contract is kept for the
    /// output widening).
    private func fusedInputForward(_ rotated: MLXArray, widenOutput: Bool) -> MLXArray {
        Self.matrixRoutedMatmul(
            rotated, weight: weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode, operands: matrixRoute,
            widenOutput: widenOutput, sourceDType: .float32)
    }

    /// `forwardPreSigned((silu(gate) * up) * signs)` with the SwiGLU product,
    /// the signs, the transform and the route dtype's rounding in one kernel
    /// (ercumentyildirim, `ade7529`). Nil when it does not apply.
    public func applyAfterSwiGLU(gate: MLXArray, up: MLXArray, widenOutput: Bool = true)
        -> MLXArray?
    {
        guard gdnLayout == nil,
            let store = fusedInputStoreDType(rows: gate.size / max(transform.width, 1)),
            let rotated = transform.rotatedSwiGLU(gate: gate, up: up, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// `applyAfterSwiGLU` on the stacked gate|up product itself (gate at
    /// column `gateOffset`, up at `upOffset`): no split, no copies.
    public func applyAfterSwiGLUStacked(
        _ wide: MLXArray, gateOffset: Int, upOffset: Int, widenOutput: Bool = true
    ) -> MLXArray? {
        guard gdnLayout == nil,
            let store = fusedInputStoreDType(rows: wide.size / max(wide.dim(-1), 1)),
            let rotated = transform.rotatedSwiGLUStacked(
                wide, gateOffset: gateOffset, upOffset: upOffset, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// `self(x * sigmoid(gate))` with the gate product fused into the rotation.
    public func applyAfterSigmoidGate(_ x: MLXArray, gate: MLXArray, widenOutput: Bool = true)
        -> MLXArray?
    {
        guard gdnLayout == nil,
            let store = fusedInputStoreDType(rows: x.size / max(transform.width, 1)),
            let rotated = transform.rotatedSigmoidGate(x, gate: gate, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// `applyAfterSigmoidGate` on `[B, S, heads, headDim]` views read through
    /// their strides: the attention output straight from its head transpose
    /// and the gate half of the q|gate projection, neither reshaped into a
    /// copy first. Nil when it does not apply.
    public func applyAfterSigmoidGateHeads(
        _ x: MLXArray, gate: MLXArray, widenOutput: Bool = true
    ) -> MLXArray? {
        guard gdnLayout == nil, x.ndim == 4,
            let store = fusedInputStoreDType(rows: x.size / max(transform.width, 1)),
            let rotated = transform.rotatedSigmoidGateHeads(x, gate: gate, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// The GDN output projection of `silu(z) * rmsNorm(x, weight, eps)` with the
    /// norm, the gate, the value layout and the rotation in one kernel.
    public func applyAfterGatedRMSNorm(
        _ x: MLXArray, gate z: MLXArray, weight: MLXArray, eps: Float, widenOutput: Bool = true
    ) -> MLXArray? {
        guard let store = fusedInputStoreDType(rows: x.size / max(transform.width, 1)),
            let rotated = transform.rotatedGatedRMSNorm(
                x, gate: z, weight: weight, eps: eps, layout: gdnLayout, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// The representation the route handles: the pack's 2-bit affine layout
    /// with FP16 constants, no linear bias, and a 64-aligned output width.
    private static func routeApplies(to layer: HadamardQuantizedLinear) -> Bool {
        matrixRouteEnabled && layer.mode == .affine && layer.bits == 2 && layer.bias == nil
            && layer.scales.dtype == .float16 && layer.weight.dim(0) % 64 == 0
    }

    /// The routed matmul over `x` (any leading shape, `[..., K]`) with the
    /// given packed operand; returns `[..., N]` in `x.dtype`.
    /// The activation dtype the route reads for a projection of `n` output
    /// rows at `rows` input rows whose activation was originally `sourceDType`.
    static func routeInputDType(rows: Int, n: Int, sourceDType: DType) -> DType {
        let paddedRows = max(rows, matrixRegimeMinimumRows)
        let nTiles = (n + 31) / 32
        let mTiles = (paddedRows + 31) / 32
        let narrow = nTiles * mTiles <= splitKTileCeiling
        // A narrow (split-K) projection reads FP16 up to one 16-row half of
        // its tile (fkiene `0fa9a35`); above that it keeps the FP32 read.
        let narrowFloat32 =
            narrow && !(narrowHalfRead && paddedRows <= splitKHalfRowLimit)
        return (narrowFloat32 || n >= vocabularyHeadMinimumRows || sourceDType == .bfloat16)
            ? .float32 : matrixRouteInputDType
    }

    private static func matrixRoutedMatmul(
        _ x: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode,
        operands: HadamardMatrixRouteOperands, widenOutput: Bool = true,
        sourceDType: DType? = nil
    ) -> MLXArray {
        let k = x.dim(-1)
        let rows = x.size / k
        let n = weight.dim(0)
        let paddedRows = max(rows, matrixRegimeMinimumRows)
        // The dtype the caller's activation had before the rotation; a fused
        // rotation may already have produced `x` in the route dtype.
        let sourceDType = sourceDType ?? x.dtype

        // The core splits K for a projection with at most 256 32x32 tiles and
        // runs the split-K body; that body is on the tensor unit for FP32 input
        // and, for one 16-row half, for FP16 input, so a narrow projection (o,
        // out, down on this pack) reads FP16 at a verify width and FP32 above
        // it. A wide one takes `qmm_t_nax` in the route dtype. BF16
        // activations (the drafter's head input) widen to FP32 exactly, as
        // the core's own promotion would, and a vocabulary head keeps FP32.
        let inputDType = Self.routeInputDType(
            rows: rows, n: n, sourceDType: sourceDType)
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

        var input = x.reshaped(rows, k)
        if inputDType != x.dtype {
            input = input.asType(inputDType)
        }
        if paddedRows > rows {
            input = concatenated(
                [input, MLXArray.zeros([paddedRows - rows, k], dtype: inputDType)], axis: 0)
        }

        var output = quantizedMM(
            input, weight, scales: routeScales, biases: routeBiases,
            transpose: true, groupSize: groupSize, bits: bits, mode: mode)
        if paddedRows > rows {
            output = output[0 ..< rows]
        }
        // The core promotes a BF16 activation with FP16 constants to FP32 and
        // returns FP32; the widened result therefore matches what the plain
        // operator would have returned for either input dtype.
        let plainOutputDType: DType = sourceDType == .bfloat16 ? .float32 : sourceDType
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
    /// True when `fusedSiblingsForward` will take these siblings for an
    /// activation of this shape (the route applies to every sibling).
    fileprivate func fusedSiblingsApply(
        _ siblings: [HadamardQuantizedLinear], rows: Int, k: Int
    ) -> Bool {
        guard Self.siblingFusionEnabled, siblings.count >= 2, rows >= 2, k % 64 == 0
        else { return false }
        for sibling in siblings {
            guard Self.routeApplies(to: sibling), sibling.groupSize == groupSize,
                sibling.weight.dim(1) == weight.dim(1), k % sibling.groupSize == 0
            else { return false }
        }
        return true
    }

    /// The dtype the stacked sibling matmul reads at `rows` input rows.
    fileprivate func fusedSiblingsInputDType(
        _ siblings: [HadamardQuantizedLinear], rows: Int, sourceDType: DType
    ) -> DType {
        let n = siblings.reduce(0) { $0 + $1.weight.dim(0) }
        return Self.routeInputDType(rows: rows, n: n, sourceDType: sourceDType)
    }

    fileprivate func fusedSiblingsForward(
        _ rotated: MLXArray, siblings: [HadamardQuantizedLinear], widenOutput: Bool = true,
        sourceDType: DType = .float32
    ) -> [MLXArray]? {
        guard
            let (wide, boundaries) = fusedSiblingsWide(
                rotated, siblings: siblings, widenOutput: widenOutput, sourceDType: sourceDType)
        else { return nil }
        return MLX.split(wide, indices: Array(boundaries.dropLast()), axis: -1)
    }

    /// `fusedSiblingsForward` before its split: the one stacked product and
    /// each sibling's end column in it.
    fileprivate func fusedSiblingsWide(
        _ rotated: MLXArray, siblings: [HadamardQuantizedLinear], widenOutput: Bool = true,
        sourceDType: DType = .float32
    ) -> (MLXArray, [Int])? {
        guard rotated.ndim >= 2, rotated.dtype == .float32 || rotated.dtype == .float16
        else { return nil }
        let k = rotated.dim(-1)
        guard fusedSiblingsApply(siblings, rows: rotated.size / k, k: k) else { return nil }
        let fused = matrixRoute.fusedSiblings(for: siblings)
        let wide = Self.matrixRoutedMatmul(
            rotated, weight: fused.weight, scales: fused.scales, biases: fused.biases,
            groupSize: groupSize, bits: bits, mode: mode, operands: fused.operands,
            widenOutput: widenOutput, sourceDType: sourceDType)
        return (wide, Array(fused.boundaries))
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
    // When the siblings will run as one routed stack, rotate straight into the
    // dtype that stack reads: the fused transform folds the cast into its
    // single launch, and the route then has nothing left to cast.
    let k = x.dim(-1)
    if x.dtype == .float32, first.fusedSiblingsApply(packed, rows: x.size / k, k: k) {
        let routeDType = first.fusedSiblingsInputDType(
            packed, rows: x.size / k, sourceDType: x.dtype)
        let rotated = first.rotate(x, outputDType: routeDType)
        if let fused = first.fusedSiblingsForward(
            rotated, siblings: packed, widenOutput: widenOutput, sourceDType: x.dtype)
        {
            return fused
        }
        let plain = routeDType == x.dtype ? rotated : rotated.asType(x.dtype)
        return packed.map { $0.applyRotated(plain) }
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

/// The stacked sibling matmul on an activation already rotated (and stored in
/// the stack's read dtype) on behalf of an FP32 activation: the unsplit product
/// and each sibling's end column. Nil when the stacked route does not apply or
/// reads a dtype other than `rotated.dtype`.
public func sharedHadamardStackOnRotated(
    _ rotated: MLXArray, _ siblings: [HadamardQuantizedLinear], widenOutput: Bool = true
) -> (wide: MLXArray, boundaries: [Int])? {
    guard let first = siblings.first,
        siblings.allSatisfy({ $0.sharesInputTransform(with: first) })
    else { return nil }
    let k = rotated.dim(-1)
    guard first.fusedSiblingsApply(siblings, rows: rotated.size / k, k: k),
        first.fusedSiblingsInputDType(siblings, rows: rotated.size / k, sourceDType: .float32)
            == rotated.dtype,
        let (wide, boundaries) = first.fusedSiblingsWide(
            rotated, siblings: siblings, widenOutput: widenOutput, sourceDType: .float32)
    else { return nil }
    return (wide, boundaries)
}

/// The dtype the stacked sibling matmul reads for `rows` rows of an FP32
/// activation (nil when the stacked route does not apply).
public func sharedHadamardStackReadDType(
    _ siblings: [HadamardQuantizedLinear], rows: Int
) -> DType? {
    guard let first = siblings.first,
        siblings.allSatisfy({ $0.sharesInputTransform(with: first) })
    else { return nil }
    let k = first.transform.width
    guard first.fusedSiblingsApply(siblings, rows: rows, k: k) else { return nil }
    return first.fusedSiblingsInputDType(siblings, rows: rows, sourceDType: .float32)
}

/// `sharedHadamardProjectionsPreSigned` when the siblings run as one routed
/// stack: the stacked product (unsplit) and each sibling's end column. Nil
/// whenever the stacked route does not apply (the caller takes the split path).
public func sharedHadamardProjectionsPreSignedWide(
    _ signed: MLXArray, _ siblings: [HadamardQuantizedLinear], widenOutput: Bool = true
) -> (wide: MLXArray, boundaries: [Int])? {
    guard let first = siblings.first,
        siblings.allSatisfy({ $0.sharesInputTransform(with: first) }), signed.dtype == .float32
    else { return nil }
    let k = signed.dim(-1)
    guard first.fusedSiblingsApply(siblings, rows: signed.size / k, k: k) else { return nil }
    let routeDType = first.fusedSiblingsInputDType(
        siblings, rows: signed.size / k, sourceDType: signed.dtype)
    let rotated = first.transform.applyPreSigned(signed, outputDType: routeDType)
    guard
        let (wide, boundaries) = first.fusedSiblingsWide(
            rotated, siblings: siblings, widenOutput: widenOutput, sourceDType: signed.dtype)
    else { return nil }
    return (wide, boundaries)
}

/// `sharedHadamardProjections` for an activation that already carries the
/// shared transform's signs (see `SignedBlockHadamard.applyPreSigned`): the
/// rotation skips its sign multiply, and every sibling reads the rotated
/// array `sharedHadamardProjections` would have formed from the unsigned
/// activation. Nil when the siblings do not share one ungrouped transform.
public func sharedHadamardProjectionsPreSigned(
    _ signed: MLXArray, _ siblings: [HadamardQuantizedLinear], widenOutput: Bool = true
) -> [MLXArray]? {
    guard let first = siblings.first,
        siblings.allSatisfy({ $0.sharesInputTransform(with: first) })
    else { return nil }
    // As in `sharedHadamardProjections`: when the siblings run as one routed
    // stack, rotate straight into the dtype the stack reads.
    let k = signed.dim(-1)
    if signed.dtype == .float32, first.fusedSiblingsApply(siblings, rows: signed.size / k, k: k) {
        let routeDType = first.fusedSiblingsInputDType(
            siblings, rows: signed.size / k, sourceDType: signed.dtype)
        let rotated = first.transform.applyPreSigned(signed, outputDType: routeDType)
        if let fused = first.fusedSiblingsForward(
            rotated, siblings: siblings, widenOutput: widenOutput, sourceDType: signed.dtype)
        {
            return fused
        }
        let plain = routeDType == signed.dtype ? rotated : rotated.asType(signed.dtype)
        return siblings.map { $0.applyRotated(plain) }
    }
    let rotated = first.transform.applyPreSigned(signed)
    if let fused = first.fusedSiblingsForward(
        rotated, siblings: siblings, widenOutput: widenOutput)
    {
        return fused
    }
    return siblings.map { $0.applyRotated(rotated) }
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

/// ercumentyildirim's (`ade7529`) fused-INPUT rotations: the SwiGLU product,
/// the attention output gate, or the GDN output's per-head RMSNorm and gated
/// tail, formed in the read of MLX's `hadamard_n<float, 1024, 16, 4>` with the
/// signs, and the result stored once in the dtype the packed matmul reads.
/// Every product is the composed op chain's, rounded to FP32 in the same
/// order, so the stored values equal the chain's FP32 rotation cast to that
/// dtype. (polymorf's plain rotation keeps the name
/// `bonsai_signed_hadamard_1024`; these kernels use their own names.)
enum FusedInputHadamardKernel {
    private static func flag(_ name: String) -> Bool {
        let value = ProcessInfo.processInfo.environment[name]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }
    /// Per-path switches (all default on; `BONSAI_FUSED_HADAMARD=0` turns all off).
    static let plainEnabled = enabled && flag("BONSAI_FUSED_PLAIN")
    static let swigluEnabled = enabled && flag("BONSAI_FUSED_SWIGLU")
    static let gateEnabled = enabled && flag("BONSAI_FUSED_GATE")
    static let gatedNormEnabled = enabled && flag("BONSAI_FUSED_GNORM")

    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_FUSED_HADAMARD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The fused residual add + RMSNorm + rotation (`BONSAI_FUSED_RESNORM=0` off).
    static let residualNormEnabled = enabled && flag("BONSAI_FUSED_RESNORM")

    /// One threadgroup of 1024 threads per 5120-wide row. See
    /// `SignedBlockHadamard.residualNormRotated`.
    static let residualNormKernel = MLXFast.metalKernel(
        name: "bonsai_residual_rmsnorm_hadamard_5120",
        inputNames: ["x", "r", "w", "signs", "eps"],
        outputNames: ["h", "out"],
        source: "#define BONSAI_STORE_NORMED(e, n)\n" + residualNormSource,
        header: residualNormHeader)

    /// `residualNormKernel` also writing the FP32 norm output `nrm`.
    static let residualNormKernelNormed = MLXFast.metalKernel(
        name: "bonsai_residual_rmsnorm_hadamard_5120_normed",
        inputNames: ["x", "r", "w", "signs", "eps"],
        outputNames: ["h", "nrm", "out"],
        source: "#define BONSAI_STORE_NORMED(e, n) nrm[base + (e)] = (n)\n" + residualNormSource,
        header: residualNormHeader)

    private static let residualNormSource = """
            constexpr uint W = 5120;
            constexpr short NT = 64;
            const uint lid = thread_position_in_threadgroup.x;
            const uint row = threadgroup_position_in_grid.x;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const size_t base = size_t(row) * W;

            threadgroup float buf[W];
            threadgroup float local_sums[32];
            threadgroup float local_inv[1];

            // The FP16 residual add, and the sum of squares of the promoted
            // row with rms_looped's mapping: pass 0 covers elements
            // 4*lid..4*lid+3, pass 1 elements 4096+4*lid.. for lid < 256.
            float hv[8];
            float acc = 0.0f;
            BONSAI_UNROLL for (int i = 0; i < 4; i++) {
              const uint e = 4 * lid + i;
              const half s = x[base + e] + r[base + e];
              h[base + e] = s;
              hv[i] = float(s);
              acc += hv[i] * hv[i];
            }
            if (lid < 256) {
              BONSAI_UNROLL for (int i = 0; i < 4; i++) {
                const uint e = 4096 + 4 * lid + i;
                const half s = x[base + e] + r[base + e];
                h[base + e] = s;
                hv[4 + i] = float(s);
                acc += hv[4 + i] * hv[4 + i];
              }
            }
            acc = simd_sum(acc);
            if (sg == 0) { local_sums[lane] = 0; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) { local_sums[sg] = acc; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
              float t = simd_sum(local_sums[lane]);
              if (lane == 0) { local_inv[0] = metal::precise::rsqrt(t / float(W) + eps); }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv = local_inv[0];
            BONSAI_UNROLL for (int i = 0; i < 4; i++) {
              const uint e = 4 * lid + i;
              const float n = w[e] * static_cast<float>(hv[i] * inv);
              BONSAI_STORE_NORMED(e, n);
              buf[e] = n * signs[e];
            }
            if (lid < 256) {
              BONSAI_UNROLL for (int i = 0; i < 4; i++) {
                const uint e = 4096 + 4 * lid + i;
                const float n = w[e] * static_cast<float>(hv[4 + i] * inv);
                BONSAI_STORE_NORMED(e, n);
                buf[e] = n * signs[e];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // hadamard_n<float, 1024, 16, 4> on each of the five blocks, 64
            // threads per block, the butterfly and operand order unchanged.
            const bool active = lid < 5 * 64;
            const short i = short(lid % 64);
            threadgroup float* blk = buf + (lid / 64) * 1024;
            float v[16];
            short hh = 1;
            BONSAI_UNROLL for (short st = 0; st < 2; st++) {
              if (active) {
                short k = i & (hh - 1);
                short j = ((i - k) << 4) + k;
                BONSAI_UNROLL for (short q = 0; q < 16; q++) { v[q] = blk[j + hh * q]; }
                bonsai_hadamard_radix<16>(v);
                BONSAI_UNROLL for (short q = 0; q < 16; q++) { blk[j + hh * q] = v[q]; }
              }
              hh <<= 4;
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (active) {
              BONSAI_UNROLL for (short t = 0; t < 4; t++) {
                short index = i + t * NT;
                short k = index & (hh - 1);
                short j = ((index - k) << 2) + k;
                BONSAI_UNROLL for (short q = 0; q < 4; q++) { v[q] = blk[j + hh * q]; }
                bonsai_hadamard_radix<4>(v);
                BONSAI_UNROLL for (short q = 0; q < 4; q++) { blk[j + hh * q] = v[q]; }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            BONSAI_UNROLL for (int q = 0; q < 5; q++) {
              const uint e = q * 1024 + lid;
              out[base + e] = static_cast<OutT>(buf[e] * 0.03125f);
            }
            """

    private static let residualNormHeader = """
            #define BONSAI_UNROLL _Pragma("clang loop unroll(full)")

            template <short R>
            METAL_FUNC void bonsai_hadamard_radix(thread float* x) {
              constexpr short logR = __builtin_ctz(R);
              short h = 1;
              BONSAI_UNROLL for (short s = 0; s < logR; s++) {
                BONSAI_UNROLL for (short i = 0; i < R / 2; i++) {
                  short k = i & (h - 1);
                  short j = ((i - k) << 1) + k;
                  float a = x[j];
                  float b = x[j + h];
                  x[j] = a + b;
                  x[j + h] = a - b;
                }
                h <<= 1;
              }
            }
            """

    /// Stacked reads: the SwiGLU rotation takes the gate|up product unsplit
    /// (`BONSAI_FUSED_STACKED=0` splits it into copied halves as before).
    static let stackedReads = enabled && swigluEnabled && flag("BONSAI_FUSED_STACKED")

    static func applies(blockSize: Int, width: Int, dtype: DType) -> Bool {
        plainEnabled && blockSize == 1024 && width % 1024 == 0 && dtype == .float32
    }

    /// `gated` with both operands read out of ONE row-major stacked product
    /// `wide` ([..., stride]) at column offsets `aOffset` and `bOffset`, so
    /// the two halves are never split into strided views and copied.
    static func gatedStacked(
        _ wide: MLXArray, width: Int, aOffset: Int, bOffset: Int, signs: MLXArray, mode: Int,
        outputDType: DType
    ) -> MLXArray {
        let stride = wide.dim(-1)
        let rows = wide.size / stride
        return gatedStackedKernel(
            [wide, signs],
            template: [
                ("WIDTH", width), ("MODE", mode), ("InT", wide.dtype), ("OutT", outputDType),
                ("STRIDE", stride), ("A_OFF", aOffset), ("B_OFF", bOffset),
            ],
            grid: (64, rows * (width / 1024), 1),
            threadGroup: (64, 1, 1),
            outputShapes: [Array(wide.shape.dropLast()) + [width]],
            outputDTypes: [outputDType])[0]
    }

    static func gated(
        _ a: MLXArray, _ b: MLXArray, signs: MLXArray, width: Int, mode: Int,
        outputDType: DType = .float32
    ) -> MLXArray {
        return gatedKernel(
            [a, b, signs],
            template: [
                ("WIDTH", width), ("MODE", mode), ("InT", a.dtype), ("OutT", outputDType),
            ],
            grid: (64, a.size / 1024, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [a.shape],
            outputDTypes: [outputDType])[0]
    }

    /// `a * sigmoid(b)` (the gated kernel's MODE 2) for `[B, S, heads, HEAD_DIM]`
    /// operands read through their strides.
    static func gatedHeads(
        _ a: MLXArray, _ b: MLXArray, signs: MLXArray, width: Int, headDim: Int,
        outputDType: DType = .float32
    ) -> MLXArray {
        return gatedHeadsKernel(
            [a, b, signs],
            template: [
                ("WIDTH", width), ("MODE", 2), ("HEAD_DIM", headDim), ("InT", a.dtype),
                ("OutT", outputDType),
            ],
            grid: (64, a.size / 1024, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[a.dim(0), a.dim(1), width]],
            outputDTypes: [outputDType])[0]
    }

    /// `BONSAI_FUSED_GATE_HEADS=0` keeps the reshaped (copied) operands.
    static let gateHeadsEnabled = enabled && flag("BONSAI_FUSED_GATE_HEADS")

    private static let gatedHeadsKernel = MLXFast.metalKernel(
        name: "bonsai_fused_input_gated_heads_hadamard_1024",
        inputNames: ["a", "b", "signs"],
        outputNames: ["out"],
        source: """
            #define BONSAI_ROW_SETUP \\
              const int64_t row_idx = int64_t(blk / BLOCKS); \\
              const int64_t s_len = int64_t(a_shape[1]); \\
              const int64_t arow = (row_idx / s_len) * a_strides[0] + (row_idx % s_len) * a_strides[1]; \\
              const int64_t brow = (row_idx / s_len) * b_strides[0] + (row_idx % s_len) * b_strides[1];
            #define BONSAI_LOAD_AB(p, av, bv) \\
              const int64_t hh = int64_t((p) / HEAD_DIM); \\
              const int64_t dd = int64_t((p) % HEAD_DIM); \\
              av = static_cast<float>(a[arow + hh * a_strides[2] + dd * a_strides[3]]); \\
              bv = static_cast<float>(b[brow + hh * b_strides[2] + dd * b_strides[3]])

            """ + gatedSource,
        header: gatedHeader,
        ensureRowContiguous: false)

    /// MODE 1: `(a * sigmoid(a)) * b` (SwiGLU, inputs FP32 or FP16 widened
    /// exactly). MODE 2: `a * sigmoid(b)`.
    private static let gatedKernel = MLXFast.metalKernel(
        name: "bonsai_fused_input_gated_hadamard_1024",
        inputNames: ["a", "b", "signs"],
        outputNames: ["out"],
        source: """
            #define BONSAI_ROW_SETUP
            #define BONSAI_LOAD_AB(p, av, bv) \\
              av = static_cast<float>(a[row_base + (p)]); \\
              bv = static_cast<float>(b[row_base + (p)])

            """ + gatedSource,
        header: gatedHeader)

    private static let gatedSource = """
            constexpr short NT = 64;
            constexpr uint BLOCKS = WIDTH / 1024;
            short i = short(thread_position_in_grid.x);
            uint blk = thread_position_in_grid.y;
            uint row_base = (blk / BLOCKS) * WIDTH;
            uint col0 = (blk % BLOCKS) * 1024;
            BONSAI_ROW_SETUP

            threadgroup float buf[1024];

            BONSAI_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                uint p = col0 + index + r;
                float av;
                float bv;
                { BONSAI_LOAD_AB(p, av, bv); }
                float v;
                if (MODE == 1) {
                  float t = av * bonsai_sigmoid(av);
                  v = t * bv;
                } else {
                  v = av * bonsai_sigmoid(bv);
                }
                buf[index + r] = v * signs[p];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float v[16];
            short h = 1;
            BONSAI_UNROLL for (short s = 0; s < 2; s++) {
              short k = i & (h - 1);
              short j = ((i - k) << 4) + k;
              BONSAI_UNROLL for (short r = 0; r < 16; r++) {
                v[r] = buf[j + h * r];
              }
              bonsai_hadamard_radix<16>(v);
              BONSAI_UNROLL for (short r = 0; r < 16; r++) {
                buf[j + h * r] = v[r];
              }
              h <<= 4;
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            BONSAI_UNROLL for (short t = 0; t < 4; t++) {
              short index = i + t * NT;
              short k = index & (h - 1);
              short j = ((index - k) << 2) + k;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                v[r] = buf[j + h * r];
              }
              bonsai_hadamard_radix<4>(v);
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                buf[j + h * r] = v[r];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            BONSAI_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                out[row_base + col0 + index + r] = static_cast<OutT>(buf[index + r] * 0.03125f);
              }
            }
            """

    private static let gatedHeader = """
            #define BONSAI_UNROLL _Pragma("clang loop unroll(full)")

            template <short R>
            METAL_FUNC void bonsai_hadamard_radix(thread float* x) {
              constexpr short logR = __builtin_ctz(R);
              short h = 1;
              BONSAI_UNROLL for (short s = 0; s < logR; s++) {
                BONSAI_UNROLL for (short i = 0; i < R / 2; i++) {
                  short k = i & (h - 1);
                  short j = ((i - k) << 1) + k;
                  float a = x[j];
                  float b = x[j + h];
                  x[j] = a + b;
                  x[j + h] = a - b;
                }
                h <<= 1;
              }
            }

            // MLX `Sigmoid` (unary_ops.h), verbatim.
            METAL_FUNC float bonsai_sigmoid(float x) {
              auto y = 1 / (1 + metal::exp(metal::abs(x)));
              return (x < 0) ? y : 1 - y;
            }

            """

    /// `gatedKernel` reading both operands from one stacked product `ab`.
    private static let gatedStackedKernel = MLXFast.metalKernel(
        name: "bonsai_fused_input_gated_hadamard_1024_stacked",
        inputNames: ["ab", "signs"],
        outputNames: ["out"],
        source: """
            constexpr short NT = 64;
            constexpr uint BLOCKS = WIDTH / 1024;
            short i = short(thread_position_in_grid.x);
            uint blk = thread_position_in_grid.y;
            uint row = blk / BLOCKS;
            uint row_base = row * WIDTH;
            uint col0 = (blk % BLOCKS) * 1024;

            threadgroup float buf[1024];

            BONSAI_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                uint p = col0 + index + r;
                float av = static_cast<float>(ab[row * STRIDE + A_OFF + p]);
                float bv = static_cast<float>(ab[row * STRIDE + B_OFF + p]);
                float v;
                if (MODE == 1) {
                  float t = av * bonsai_sigmoid(av);
                  v = t * bv;
                } else {
                  v = av * bonsai_sigmoid(bv);
                }
                buf[index + r] = v * signs[p];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float v[16];
            short h = 1;
            BONSAI_UNROLL for (short s = 0; s < 2; s++) {
              short k = i & (h - 1);
              short j = ((i - k) << 4) + k;
              BONSAI_UNROLL for (short r = 0; r < 16; r++) {
                v[r] = buf[j + h * r];
              }
              bonsai_hadamard_radix<16>(v);
              BONSAI_UNROLL for (short r = 0; r < 16; r++) {
                buf[j + h * r] = v[r];
              }
              h <<= 4;
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            BONSAI_UNROLL for (short t = 0; t < 4; t++) {
              short index = i + t * NT;
              short k = index & (h - 1);
              short j = ((index - k) << 2) + k;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                v[r] = buf[j + h * r];
              }
              bonsai_hadamard_radix<4>(v);
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                buf[j + h * r] = v[r];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            BONSAI_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                out[row_base + col0 + index + r] = static_cast<OutT>(buf[index + r] * 0.03125f);
              }
            }
            """,
        header: """
            #define BONSAI_UNROLL _Pragma("clang loop unroll(full)")

            template <short R>
            METAL_FUNC void bonsai_hadamard_radix(thread float* x) {
              constexpr short logR = __builtin_ctz(R);
              short h = 1;
              BONSAI_UNROLL for (short s = 0; s < logR; s++) {
                BONSAI_UNROLL for (short i = 0; i < R / 2; i++) {
                  short k = i & (h - 1);
                  short j = ((i - k) << 1) + k;
                  float a = x[j];
                  float b = x[j + h];
                  x[j] = a + b;
                  x[j + h] = a - b;
                }
                h <<= 1;
              }
            }

            // MLX `Sigmoid` (unary_ops.h), verbatim.
            METAL_FUNC float bonsai_sigmoid(float x) {
              auto y = 1 / (1 + metal::exp(metal::abs(x)));
              return (x < 0) ? y : 1 - y;
            }

            """)
}

extension FusedInputHadamardKernel {
    /// GDN output: per-head RMSNorm (MLX `rms_single_row`, 32 lanes x 4 reads),
    /// `silu(z) * normed`, value-head permutation, signs and the transform.
    static func gatedRMSNorm(
        _ x: MLXArray, _ z: MLXArray, weight: MLXArray, eps: Float, signs: MLXArray,
        repeats: Int, keyHeads: Int, headDim: Int, outputDType: DType = .float32
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        // Both are read through their strides as [B, S, heads, headDim]: z is
        // a column slice of the stacked qkv|z product (a view; the reshape
        // splits its last axis without a copy), so the launch no longer
        // copies it to a row-contiguous buffer first.
        let heads = repeats * keyHeads
        return gatedRMSNormKernel(
            [x.reshaped(B, S, heads, headDim), z.reshaped(B, S, heads, headDim), weight, signs,
             MLXArray(eps)],
            template: [
                ("REPEATS", repeats), ("KEY_HEADS", keyHeads), ("HEAD_DIM", headDim),
                ("OutT", outputDType),
            ],
            grid: (64, x.size / 1024, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[B, S, repeats * keyHeads * headDim]],
            outputDTypes: [outputDType])[0]
    }

    private static let gatedRMSNormKernel = MLXFast.metalKernel(
        name: "bonsai_fused_input_gated_rmsnorm_hadamard_1024",
        inputNames: ["x", "z", "w", "signs", "eps"],
        outputNames: ["out"],
        source: """
            constexpr short NT = 64;
            constexpr uint WIDTH = REPEATS * KEY_HEADS * HEAD_DIM;
            constexpr uint BLOCKS = WIDTH / 1024;
            constexpr uint HEADS_PER_BLOCK = 1024 / HEAD_DIM;
            short i = short(thread_position_in_grid.x);
            uint blk = thread_position_in_grid.y;
            uint row_base = (blk / BLOCKS) * WIDTH;
            uint col0 = (blk % BLOCKS) * 1024;
            const int64_t row_idx = int64_t(blk / BLOCKS);
            const int64_t s_len = int64_t(x_shape[1]);
            const int64_t xrow = (row_idx / s_len) * x_strides[0] + (row_idx % s_len) * x_strides[1];
            const int64_t zrow = (row_idx / s_len) * z_strides[0] + (row_idx % s_len) * z_strides[1];
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;

            threadgroup float buf[1024];
            threadgroup float inv_rms[HEADS_PER_BLOCK];

            // Per-head RMS as rms_single_row with 32 threads x 4 reads: lane l
            // sums elements 4l..4l+3 of the head in order, then simd_sum.
            BONSAI_UNROLL for (uint hh = sg; hh < HEADS_PER_BLOCK; hh += 2) {
              uint p0 = col0 + hh * HEAD_DIM;
              uint kh = p0 / (REPEATS * HEAD_DIM);
              uint rep = (p0 % (REPEATS * HEAD_DIM)) / HEAD_DIM;
              uint src_head = rep * KEY_HEADS + kh;
              const int64_t xh = xrow + int64_t(src_head) * x_strides[2];
              float acc = 0;
              float tx[4];
              BONSAI_UNROLL for (int r = 0; r < 4; r++) {
                tx[r] = x[xh + int64_t(lane * 4 + r) * x_strides[3]];
                acc += tx[r] * tx[r];
              }
              acc = simd_sum(acc);
              if (lane == 0) {
                inv_rms[hh] = metal::precise::rsqrt(acc / HEAD_DIM + eps);
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            BONSAI_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                uint p = col0 + index + r;
                uint kh = p / (REPEATS * HEAD_DIM);
                uint rem = p % (REPEATS * HEAD_DIM);
                uint src = ((rem / HEAD_DIM) * KEY_HEADS + kh) * HEAD_DIM + rem % HEAD_DIM;
                const int64_t sh = int64_t(src / HEAD_DIM);
                const int64_t sd = int64_t(src % HEAD_DIM);
                float xn = w[src % HEAD_DIM]
                    * (x[xrow + sh * x_strides[2] + sd * x_strides[3]] * inv_rms[(index + r) / HEAD_DIM]);
                float zv = z[zrow + sh * z_strides[2] + sd * z_strides[3]];
                float gz = zv * bonsai_sigmoid(zv);
                float v = gz * xn;
                buf[index + r] = v * signs[p];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float v[16];
            short h = 1;
            BONSAI_UNROLL for (short s = 0; s < 2; s++) {
              short k = i & (h - 1);
              short j = ((i - k) << 4) + k;
              BONSAI_UNROLL for (short r = 0; r < 16; r++) {
                v[r] = buf[j + h * r];
              }
              bonsai_hadamard_radix<16>(v);
              BONSAI_UNROLL for (short r = 0; r < 16; r++) {
                buf[j + h * r] = v[r];
              }
              h <<= 4;
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            BONSAI_UNROLL for (short t = 0; t < 4; t++) {
              short index = i + t * NT;
              short k = index & (h - 1);
              short j = ((index - k) << 2) + k;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                v[r] = buf[j + h * r];
              }
              bonsai_hadamard_radix<4>(v);
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                buf[j + h * r] = v[r];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            BONSAI_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                out[row_base + col0 + index + r] = static_cast<OutT>(buf[index + r] * 0.03125f);
              }
            }
            """,
        header: """
            #define BONSAI_UNROLL _Pragma("clang loop unroll(full)")

            template <short R>
            METAL_FUNC void bonsai_hadamard_radix(thread float* x) {
              constexpr short logR = __builtin_ctz(R);
              short h = 1;
              BONSAI_UNROLL for (short s = 0; s < logR; s++) {
                BONSAI_UNROLL for (short i = 0; i < R / 2; i++) {
                  short k = i & (h - 1);
                  short j = ((i - k) << 1) + k;
                  float a = x[j];
                  float b = x[j + h];
                  x[j] = a + b;
                  x[j + h] = a - b;
                }
                h <<= 1;
              }
            }

            // MLX `Sigmoid` (unary_ops.h), verbatim.
            METAL_FUNC float bonsai_sigmoid(float x) {
              auto y = 1 / (1 + metal::exp(metal::abs(x)));
              return (x < 0) ? y : 1 - y;
            }

            """,
        ensureRowContiguous: !gatedRMSNormStrided)

    /// `BONSAI_FUSED_GATED_STRIDED=0` copies x and z to row-contiguous buffers
    /// before the launch, as before.
    private static let gatedRMSNormStrided: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_FUSED_GATED_STRIDED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()
}
