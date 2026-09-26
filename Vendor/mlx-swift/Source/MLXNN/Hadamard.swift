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

    /// A rotated activation quantized for the tensor route: `codes` holds
    /// `round(x / scale) + 128` as UInt8 with one symmetric absmax `scale`
    /// (FP32) per row and `groupSize` consecutive elements, and `scaledSums`
    /// holds `scale * sum(round(x / scale))` per row and group (FP32), the
    /// term a packed matmul over raw codes needs for the affine offsets.
    public struct Int8Activation {
        public let codes: MLXArray
        public let scales: MLXArray
        public let scaledSums: MLXArray
        public init(codes: MLXArray, scales: MLXArray, scaledSums: MLXArray) {
            self.codes = codes
            self.scales = scales
            self.scaledSums = scaledSums
        }
    }

    /// The forward transform that quantizes its FP32 result per group into
    /// an `Int8Activation` (`[..., width]` codes, `[..., width / groupSize]`
    /// scales and scaled sums). Installed by the model file next to
    /// `fusedTransform`; nil declines.
    public typealias FusedTransformInt8 = (
        _ x: MLXArray, _ signs: MLXArray, _ blockSize: Int, _ preSigned: Bool,
        _ gdnLayout: HadamardGDNLayout?, _ groupSize: Int
    ) -> Int8Activation?
    nonisolated(unsafe) public static var fusedTransformInt8: FusedTransformInt8?

    /// `forward` (or `applyPreSigned` when `preSigned`) quantized per group;
    /// nil when no fused implementation provides it.
    public func forwardInt8(
        _ x: MLXArray, gdnLayout: HadamardGDNLayout?, preSigned: Bool, groupSize: Int
    ) -> Int8Activation? {
        validate(x)
        guard let fused = Self.fusedTransformInt8 else { return nil }
        if preSigned { precondition(gdnLayout == nil, "pre-signed transform needs an ungrouped layout") }
        return fused(x, signs, blockSize, preSigned, gdnLayout, groupSize)
    }

    /// The forward transform stored in `outputDType` together with the FP32
    /// sums of every `groupSize` consecutive rounded outputs (`[..., width /
    /// groupSize]`), for the verify-width tensor route. Installed by the
    /// model file; nil declines.
    public typealias FusedTransformWithGroupSums = (
        _ x: MLXArray, _ signs: MLXArray, _ blockSize: Int, _ preSigned: Bool,
        _ gdnLayout: HadamardGDNLayout?, _ outputDType: DType, _ groupSize: Int
    ) -> (MLXArray, MLXArray)?
    nonisolated(unsafe) public static var fusedTransformWithGroupSums: FusedTransformWithGroupSums?

    /// `forward` (or `applyPreSigned` when `preSigned`) with its per-group
    /// sums; nil when no fused implementation provides them.
    public func forwardWithGroupSums(
        _ x: MLXArray, gdnLayout: HadamardGDNLayout?, preSigned: Bool, outputDType: DType,
        groupSize: Int
    ) -> (MLXArray, MLXArray)? {
        validate(x)
        guard let fused = Self.fusedTransformWithGroupSums else { return nil }
        if preSigned { precondition(gdnLayout == nil, "pre-signed transform needs an ungrouped layout") }
        return fused(x, signs, blockSize, preSigned, gdnLayout, outputDType, groupSize)
    }

    /// A packed projection's input producer that the quantizing rotation can
    /// form in its read instead of reading a materialized FP32 array:
    /// `silu(gate) * up`, `x * sigmoid(gate)`, or the GDN output's per-head
    /// RMSNorm with its `silu(gate) * normed` tail (the same FP32 arithmetic
    /// as the model's compiled chains, then the signs, the transform and the
    /// quantization).
    public enum Int8Producer {
        case swiglu(gate: MLXArray, up: MLXArray)
        case sigmoidGate(x: MLXArray, gate: MLXArray)
        case gatedRMSNorm(x: MLXArray, gate: MLXArray, weight: MLXArray, eps: Float)

        /// The array whose shape and dtype the product follows.
        public var primary: MLXArray {
            switch self {
            case .swiglu(let gate, _): return gate
            case .sigmoidGate(let x, _): return x
            case .gatedRMSNorm(let x, _, _, _): return x
            }
        }
    }

    public typealias FusedTransformInt8Producer = (
        _ producer: Int8Producer, _ signs: MLXArray, _ blockSize: Int,
        _ gdnLayout: HadamardGDNLayout?, _ groupSize: Int
    ) -> Int8Activation?
    nonisolated(unsafe) public static var fusedTransformInt8Producer: FusedTransformInt8Producer?

    /// `forwardInt8` of a producer's output formed inside the rotation; nil
    /// when no fused implementation provides it.
    public func forwardInt8(
        producer: Int8Producer, gdnLayout: HadamardGDNLayout?, groupSize: Int
    ) -> Int8Activation? {
        let x = producer.primary
        precondition(x.ndim > 0 && x.size % width == 0, "Hadamard input width mismatch")
        guard let fused = Self.fusedTransformInt8Producer else { return nil }
        return fused(producer, signs, blockSize, gdnLayout, groupSize)
    }

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

    /// `self(x * sigmoid(gate))`, the attention output gate, fused the same way.
    /// With strided inputs on, `[B, S, heads, headDim]` operands are read in
    /// place (the result is `[B, S, width]`).
    public func rotatedSigmoidGate(
        _ x: MLXArray, gate: MLXArray, outputDType: DType = .float32
    ) -> MLXArray? {
        guard FusedInputHadamardKernel.gateEnabled, blockSize == 1024, width % 1024 == 0,
            x.dtype == .float32, gate.dtype == .float32, x.shape == gate.shape,
            x.ndim > 0,
            x.dim(-1) == width
                || (HadamardStridedInputs.enabled && x.ndim == 4 && x.dim(2) * x.dim(3) == width)
        else { return nil }
        return FusedInputHadamardKernel.gated(
            x, gate, signs: signs, width: width, mode: 2, outputDType: outputDType)
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
/// Constants derived from a packed projection's frozen scales and offsets
/// for the tensor route (their FP16 values transposed to `[groups, rows]`,
/// and the per-group code sums folded with the scales), built once per
/// constant array and reused while that array object is unchanged. Like the
/// cast cache, a plain class that is neither an MLXArray nor a Module.
public final class HadamardConstantLayoutCache {
    private struct Entry {
        let source: MLXArray
        let tag: Int
        let derived: MLXArray
    }
    private let lock = NSLock()
    private var entries: [Entry] = []

    public init() {}

    /// The array `build(source)` for this `source` object and `tag`, built on
    /// first use.
    public func derived(_ source: MLXArray, tag: Int, build: (MLXArray) -> MLXArray) -> MLXArray {
        lock.withLock {
            for entry in entries where entry.source === source && entry.tag == tag {
                return entry.derived
            }
            let built = build(source)
            if entries.count >= 6 {
                entries.removeFirst()
            }
            entries.append(Entry(source: source, tag: tag, derived: built))
            return built
        }
    }

    public func clear() {
        lock.withLock { entries.removeAll() }
    }
}

private final class HadamardMatrixRouteOperands {
    private let lock = NSLock()
    /// Sibling projections fused along their output axis; see
    /// `HadamardFusedSiblings`. Owned by the first sibling's operands.
    var fusedSiblings: HadamardFusedSiblings?
    let scaleCache = ConstantArrayCastCache()
    let offsetCache = ConstantArrayCastCache()
    let layoutCache = HadamardConstantLayoutCache()

    func clear() {
        lock.withLock { fusedSiblings = nil }
        scaleCache.clear()
        offsetCache.clear()
        layoutCache.clear()
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
        // The vocabulary head at verify width (16 rows) reaches the tensor
        // route here; the tower projections reach it through the shared and
        // pre-signed forwards below.
        if let routed = tensorRouteForward(x, siblings: [self], preSigned: false, widenOutput: true) {
            return routed[0]
        }
        return applyRotated(rotate(x))
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

    /// On unless explicitly disabled: a BF16 activation reading a vocabulary
    /// head (the DFlash 2 drafter's shared-head read) takes the FP16 read
    /// and, through `forwardUnwidened`, returns the FP16 logits as they are.
    /// `DARKBLOOM_DFLASH2_HEAD_F16=0` restores the FP32 widening.
    public static let drafterHeadFloat16: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_DFLASH2_HEAD_F16"]?
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

    // MARK: - Tensor route (raw codes on the tensor unit)

    /// A packed matmul that multiplies a quantized rotated activation by the
    /// raw 2-bit codes on the tensor unit and applies the FP16 scale and
    /// offset of every 128-group in FP32. Weights are read as stored; the
    /// constants are read through a `HadamardConstantLayoutCache`. Installed
    /// by the model file; nil declines. Arguments: the activation (`[rows,
    /// k]` codes), packed weight `[n, k / 16]`, scales and offsets `[n, k /
    /// groupSize]` FP16, the group size, the output dtype, the layout cache.
    public typealias TensorPackedMatmul = (
        _ activation: SignedBlockHadamard.Int8Activation, _ weight: MLXArray,
        _ scales: MLXArray, _ biases: MLXArray, _ groupSize: Int, _ outputDType: DType,
        _ layoutCache: HadamardConstantLayoutCache
    ) -> MLXArray?
    nonisolated(unsafe) public static var tensorPackedMatmul: TensorPackedMatmul?
    /// Whether the installed matmul takes a product of this shape.
    nonisolated(unsafe) public static var tensorPackedMatmulApplies:
        ((_ rows: Int, _ n: Int, _ k: Int) -> Bool)?

    /// The verify-width form of the tensor route: the FP16 rotated activation
    /// (`[16, k]`, rows beyond the real ones zero) with its FP32 group sums
    /// (`[16, k / groupSize]`); returns `[16, n]`. Installed by the model file.
    public typealias TensorPackedMatmulNarrow = (
        _ rotated: MLXArray, _ groupSums: MLXArray, _ weight: MLXArray, _ scales: MLXArray,
        _ biases: MLXArray, _ groupSize: Int, _ outputDType: DType,
        _ layoutCache: HadamardConstantLayoutCache
    ) -> MLXArray?
    nonisolated(unsafe) public static var tensorPackedMatmulNarrow: TensorPackedMatmulNarrow?
    nonisolated(unsafe) public static var tensorPackedMatmulNarrowApplies:
        ((_ rows: Int, _ n: Int, _ k: Int) -> Bool)?
    /// A verify window: at most this many rows take the narrow form.
    static let tensorRouteMaximumNarrowRows = 16

    /// On unless explicitly disabled: `DARKBLOOM_BONSAI_TENSOR_ROUTE=0` keeps
    /// the dequantizing matrix route for every width.
    private static let tensorRouteEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Prompt widths only: a verify window (<= 17 rows) keeps the matrix route.
    private static let tensorRouteMinimumRows = 128

    private func tensorRouteTakes(_ layer: HadamardQuantizedLinear) -> Bool {
        layer.mode == .affine && layer.bits == 2 && layer.groupSize == 128 && layer.bias == nil
            && layer.scales.dtype == .float16 && layer.biases != nil
            && layer.biases!.dtype == .float16 && layer.weight.dim(0) % 64 == 0
            && layer.weight.dim(1) == weight.dim(1) && layer.transform.blockSize == 1024
    }

    /// The tensor route for `siblings` (self first) over one FP32 activation:
    /// one fused rotation with group sums, one packed matmul over the stacked
    /// codes, split per sibling. Nil when the route does not apply.
    fileprivate func tensorRouteForward(
        _ x: MLXArray, siblings: [HadamardQuantizedLinear], preSigned: Bool, widenOutput: Bool
    ) -> [MLXArray]? {
        guard Self.tensorRouteEnabled, x.dtype == .float32, x.ndim >= 2,
            !siblings.isEmpty, siblings[0] === self
        else { return nil }
        let k = x.dim(-1)
        let rows = x.size / k
        let promptWidth = rows >= Self.tensorRouteMinimumRows && rows % 64 == 0
        let verifyWidth = rows <= Self.tensorRouteMaximumNarrowRows
        guard k % 128 == 0,
            (promptWidth && Self.tensorPackedMatmul != nil && Self.tensorPackedMatmulApplies != nil)
                || (verifyWidth && Self.tensorPackedMatmulNarrow != nil
                    && Self.tensorPackedMatmulNarrowApplies != nil)
        else { return nil }
        var n = 0
        for sibling in siblings {
            guard tensorRouteTakes(sibling) else { return nil }
            if sibling !== self {
                guard gdnLayout == nil, sibling.sharesInputTransform(with: self) else { return nil }
            }
            n += sibling.weight.dim(0)
        }
        let outputDType: DType = widenOutput ? .float32 : .float16
        let leading = Array(x.shape.dropLast())
        if !promptWidth {
            return tensorRouteForwardNarrow(
                x.reshaped(rows, k), rows: rows, k: k, n: n, siblings: siblings,
                preSigned: preSigned, outputDType: outputDType, leading: leading)
        }
        guard let applies = Self.tensorPackedMatmulApplies,
            applies(rows, n, k),
            let activation = transform.forwardInt8(
                x.reshaped(rows, k), gdnLayout: gdnLayout, preSigned: preSigned, groupSize: 128)
        else { return nil }
        return tensorRoutePromptMatmul(
            activation, siblings: siblings, n: n, outputDType: outputDType, leading: leading)
    }

    /// Whether `tensorRouteForward` may take an FP32 activation of `rows` rows
    /// for `siblings` (self first); false means the matrix route serves it.
    fileprivate func tensorRouteMayTake(rows: Int, siblings: [HadamardQuantizedLinear]) -> Bool {
        guard Self.tensorRouteEnabled, !siblings.isEmpty, siblings[0] === self else { return false }
        let promptWidth = rows >= Self.tensorRouteMinimumRows && rows % 64 == 0
        let verifyWidth = rows <= Self.tensorRouteMaximumNarrowRows
        guard transform.width % 128 == 0,
            (promptWidth && Self.tensorPackedMatmul != nil && Self.tensorPackedMatmulApplies != nil)
                || (verifyWidth && Self.tensorPackedMatmulNarrow != nil
                    && Self.tensorPackedMatmulNarrowApplies != nil)
        else { return false }
        return siblings.allSatisfy { tensorRouteTakes($0) }
    }

    /// The prompt-width packed matmul of `tensorRouteForward` over an
    /// activation already quantized for the route: one matmul for one
    /// projection, or one over the stacked siblings split back per sibling.
    private func tensorRoutePromptMatmul(
        _ activation: SignedBlockHadamard.Int8Activation, siblings: [HadamardQuantizedLinear],
        n: Int, outputDType: DType, leading: [Int]
    ) -> [MLXArray]? {
        guard let matmul = Self.tensorPackedMatmul else { return nil }
        if siblings.count == 1 {
            guard let y = matmul(
                activation, weight, scales, biases!, groupSize, outputDType,
                matrixRoute.layoutCache)
            else { return nil }
            return [y.reshaped(leading + [n])]
        }
        let fused = matrixRoute.fusedSiblings(for: siblings)
        guard let fusedBiases = fused.biases,
            let wide = matmul(
                activation, fused.weight, fused.scales, fusedBiases, groupSize, outputDType,
                fused.operands.layoutCache)
        else { return nil }
        return MLX.split(
            wide.reshaped(leading + [n]), indices: Array(fused.boundaries.dropLast()), axis: -1)
    }

    /// True when the prompt-width tensor route is installed and on and `rows`
    /// is a width its prompt branch takes (the per-projection guards of
    /// `sharedHadamardTensorRouteTakesPrompt` still apply).
    public static func tensorRouteTakesPromptRows(_ rows: Int) -> Bool {
        tensorRouteEnabled && tensorPackedMatmul != nil && tensorPackedMatmulApplies != nil
            && SignedBlockHadamard.fusedTransformInt8 != nil
            && rows >= tensorRouteMinimumRows && rows % 64 == 0
    }

    /// True when `tensorRouteForward` takes `siblings` (self first) at prompt
    /// width for an FP32 `[rows, transform.width]` activation: every guard of
    /// its prompt branch, including the installed quantizing rotation that
    /// `forwardInt8` needs. A caller that forms the quantized activation some
    /// other way asks this before it launches anything.
    fileprivate func tensorRouteTakesPrompt(
        rows: Int, siblings: [HadamardQuantizedLinear]
    ) -> Bool {
        guard Self.tensorRouteEnabled, !siblings.isEmpty, siblings[0] === self,
            Self.tensorPackedMatmul != nil, let applies = Self.tensorPackedMatmulApplies,
            SignedBlockHadamard.fusedTransformInt8 != nil
        else { return false }
        let k = transform.width
        guard rows >= Self.tensorRouteMinimumRows, rows % 64 == 0, k % 128 == 0 else {
            return false
        }
        var n = 0
        for sibling in siblings {
            guard tensorRouteTakes(sibling) else { return false }
            if sibling !== self {
                guard gdnLayout == nil, sibling.sharesInputTransform(with: self) else {
                    return false
                }
            }
            n += sibling.weight.dim(0)
        }
        return applies(rows, n, k)
    }

    /// `tensorRouteForward`'s prompt branch on an activation that is exactly
    /// what `forwardInt8` would have returned for the `[rows, k]` input: the
    /// same guards, the same matmul, the result shaped `leading + [n]` and
    /// split per sibling. Nil when the route does not take it.
    fileprivate func tensorRouteForwardQuantized(
        _ activation: SignedBlockHadamard.Int8Activation, rows: Int, leading: [Int],
        siblings: [HadamardQuantizedLinear], widenOutput: Bool
    ) -> [MLXArray]? {
        let k = transform.width
        guard tensorRouteTakesPrompt(rows: rows, siblings: siblings),
            leading.reduce(1, *) == rows,
            activation.codes.dtype == .uint8, activation.codes.shape == [rows, k],
            activation.scales.dtype == .float32, activation.scales.shape == [rows, k / 128],
            activation.scaledSums.dtype == .float32,
            activation.scaledSums.shape == [rows, k / 128]
        else { return nil }
        let n = siblings.reduce(0) { $0 + $1.weight.dim(0) }
        return tensorRoutePromptMatmul(
            activation, siblings: siblings, n: n,
            outputDType: widenOutput ? .float32 : .float16, leading: leading)
    }

    /// On unless explicitly disabled: `DARKBLOOM_BONSAI_TENSOR_ROUTE_PRODUCER=0`
    /// keeps the materialized producer (the compiled elementwise chain) in
    /// front of the quantizing rotation at prompt width.
    private static let producerRouteEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_TENSOR_ROUTE_PRODUCER"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// The prompt-width tensor route with this projection's input producer
    /// folded into the quantizing rotation. Nil when it does not apply.
    fileprivate func tensorRouteForwardProducer(
        _ producer: SignedBlockHadamard.Int8Producer, widenOutput: Bool
    ) -> MLXArray? {
        guard Self.tensorRouteEnabled, Self.producerRouteEnabled,
            let matmul = Self.tensorPackedMatmul, let applies = Self.tensorPackedMatmulApplies
        else { return nil }
        let x = producer.primary
        let k = transform.width
        guard x.ndim >= 2, x.size % k == 0 else { return nil }
        let rows = x.size / k
        let n = weight.dim(0)
        guard rows >= Self.tensorRouteMinimumRows, rows % 64 == 0, k % 128 == 0,
            tensorRouteTakes(self), applies(rows, n, k),
            let activation = transform.forwardInt8(
                producer: producer, gdnLayout: gdnLayout, groupSize: 128)
        else { return nil }
        let flat = SignedBlockHadamard.Int8Activation(
            codes: activation.codes.reshaped(rows, k),
            scales: activation.scales.reshaped(rows, k / 128),
            scaledSums: activation.scaledSums.reshaped(rows, k / 128))
        let outputDType: DType = widenOutput ? .float32 : .float16
        guard let y = matmul(
            flat, weight, scales, biases!, groupSize, outputDType, matrixRoute.layoutCache)
        else { return nil }
        let leading = x.ndim == 4 ? [x.dim(0), x.dim(1)] : Array(x.shape.dropLast())
        return y.reshaped(leading + [n])
    }

    /// The verify-width tensor route: one FP16 rotation with group sums, rows
    /// padded to 16, one packed matmul over the (stacked) codes.
    private func tensorRouteForwardNarrow(
        _ x: MLXArray, rows: Int, k: Int, n: Int, siblings: [HadamardQuantizedLinear],
        preSigned: Bool, outputDType: DType, leading: [Int]
    ) -> [MLXArray]? {
        guard let matmul = Self.tensorPackedMatmulNarrow,
            let applies = Self.tensorPackedMatmulNarrowApplies, applies(rows, n, k),
            let (rotated, sums) = transform.forwardWithGroupSums(
                x, gdnLayout: gdnLayout, preSigned: preSigned, outputDType: .float16,
                groupSize: 128)
        else { return nil }
        var a = rotated
        var g = sums
        let padded = Self.tensorRouteMaximumNarrowRows
        if rows < padded {
            a = concatenated([a, MLXArray.zeros([padded - rows, k], dtype: .float16)], axis: 0)
            g = concatenated([g, MLXArray.zeros([padded - rows, k / 128], dtype: .float32)], axis: 0)
        }
        if siblings.count == 1 {
            guard let y = matmul(
                a, g, weight, scales, biases!, groupSize, outputDType, matrixRoute.layoutCache)
            else { return nil }
            let rowsOut = rows < padded ? y[0 ..< rows] : y
            return [rowsOut.reshaped(leading + [n])]
        }
        let fused = matrixRoute.fusedSiblings(for: siblings)
        guard let fusedBiases = fused.biases,
            let wide = matmul(
                a, g, fused.weight, fused.scales, fusedBiases, groupSize, outputDType,
                fused.operands.layoutCache)
        else { return nil }
        let rowsOut = rows < padded ? wide[0 ..< rows] : wide
        return MLX.split(
            rowsOut.reshaped(leading + [n]), indices: Array(fused.boundaries.dropLast()), axis: -1)
    }

    /// `callAsFunction` for a consumer that promotes dtypes itself, such as
    /// the residual add: when the route applies, the FP16 product is returned
    /// as is instead of being widened first. The consumer's promotion widens
    /// the same values exactly, so the arithmetic is unchanged and one cast
    /// dispatch per call is saved.
    public func forwardUnwidened(_ x: MLXArray) -> MLXArray {
        if let routed = tensorRouteForward(x, siblings: [self], preSigned: false, widenOutput: false) {
            return routed[0]
        }
        let rotated = rotate(x)
        return matrixRegimeForward(rotated, widenOutput: false) ?? applyRotated(rotated)
    }

    /// The projection of an activation that already carries the transform's
    /// signs (see `SignedBlockHadamard.applyPreSigned`), optionally leaving the
    /// FP16 product unwidened. Not for a layer with a GDN layout.
    public func forwardPreSigned(_ signed: MLXArray, widenOutput: Bool = true) -> MLXArray {
        precondition(gdnLayout == nil, "pre-signed forward needs an ungrouped layout")
        if let routed = tensorRouteForward(
            signed, siblings: [self], preSigned: true, widenOutput: widenOutput)
        {
            return routed[0]
        }
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
        // At prompt width the tensor route (raw codes on the tensor unit, an
        // 8-bit activation) takes the projection instead; its callers fall
        // back to the pre-signed / gated paths that reach `tensorRouteForward`.
        if Self.tensorRouteEnabled, Self.tensorPackedMatmul != nil,
            rows >= Self.tensorRouteMinimumRows, rows % 64 == 0
        {
            return nil
        }
        if Self.tensorRouteEnabled, Self.tensorPackedMatmulNarrow != nil,
            rows <= Self.tensorRouteMaximumNarrowRows
        {
            return nil
        }
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
        if gdnLayout == nil,
            let y = tensorRouteForwardProducer(.swiglu(gate: gate, up: up), widenOutput: widenOutput)
        {
            return y
        }
        guard gdnLayout == nil,
            let store = fusedInputStoreDType(rows: gate.size / max(transform.width, 1)),
            let rotated = transform.rotatedSwiGLU(gate: gate, up: up, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// `self(x * sigmoid(gate))` with the gate product fused into the rotation.
    public func applyAfterSigmoidGate(_ x: MLXArray, gate: MLXArray, widenOutput: Bool = true)
        -> MLXArray?
    {
        if gdnLayout == nil,
            let y = tensorRouteForwardProducer(.sigmoidGate(x: x, gate: gate), widenOutput: widenOutput)
        {
            return y
        }
        guard gdnLayout == nil,
            let store = fusedInputStoreDType(rows: x.size / max(transform.width, 1)),
            let rotated = transform.rotatedSigmoidGate(x, gate: gate, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// `applyAfterSigmoidGate` for `[B, S, heads, headDim]` operands read
    /// through their strides, on the prompt-width tensor route only: its
    /// producer reads the head-transposed attention output and the gate half
    /// of each q|gate head in place (newjordan's `9024f66b`), so neither is
    /// reshaped into a copy first. Nil when the route does not take them (the
    /// caller then reshapes and calls `applyAfterSigmoidGate`).
    public func applyAfterSigmoidGateHeadsOnRoute(
        _ x: MLXArray, gate: MLXArray, widenOutput: Bool = true
    ) -> MLXArray? {
        guard gdnLayout == nil, x.ndim == 4, x.shape == gate.shape else { return nil }
        return tensorRouteForwardProducer(
            .sigmoidGate(x: x, gate: gate), widenOutput: widenOutput)
    }

    /// `applyAfterSigmoidGate` for `[B, S, heads, headDim]` operands read
    /// through their strides by the fused-input rotation (the verify width's
    /// matrix route): the head-transposed attention output and the gate half
    /// of each q|gate head are not reshaped into copies first. Same elements,
    /// same arithmetic. Nil when it does not apply (the caller then reshapes).
    public func applyAfterSigmoidGateHeads(
        _ x: MLXArray, gate: MLXArray, widenOutput: Bool = true
    ) -> MLXArray? {
        guard HadamardStridedInputs.enabled, gdnLayout == nil, x.ndim == 4,
            x.shape == gate.shape, x.dim(2) * x.dim(3) == transform.width,
            let store = fusedInputStoreDType(rows: x.dim(0) * x.dim(1)),
            let rotated = transform.rotatedSigmoidGate(x, gate: gate, outputDType: store)
        else { return nil }
        return fusedInputForward(rotated, widenOutput: widenOutput)
    }

    /// The GDN output projection of `silu(z) * rmsNorm(x, weight, eps)` with the
    /// norm, the gate, the value layout and the rotation in one kernel.
    public func applyAfterGatedRMSNorm(
        _ x: MLXArray, gate z: MLXArray, weight: MLXArray, eps: Float, widenOutput: Bool = true
    ) -> MLXArray? {
        if let y = tensorRouteForwardProducer(
            .gatedRMSNorm(x: x, gate: z, weight: weight, eps: eps), widenOutput: widenOutput)
        {
            return y
        }
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
        let nTiles = (n + 31) / 32
        let mTiles = (paddedRows + 31) / 32
        let narrow = nTiles * mTiles <= splitKTileCeiling
        // The drafter's shared-head read (a BF16 activation, vocabulary
        // width): the FP16 read halves the A-fragment and logit traffic of
        // the 318 MB head pass; its products feed the drafter's own top-k
        // only, never an emitted token.
        let float16Head =
            drafterHeadFloat16 && !narrow && sourceDType == .bfloat16
            && n >= vocabularyHeadMinimumRows
        let inputDType: DType =
            float16Head
            ? .float16
            : Self.routeInputDType(rows: rows, n: n, sourceDType: sourceDType)
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
        guard rotated.ndim >= 2, rotated.dtype == .float32 || rotated.dtype == .float16
        else { return nil }
        let k = rotated.dim(-1)
        guard fusedSiblingsApply(siblings, rows: rotated.size / k, k: k) else { return nil }
        let fused = matrixRoute.fusedSiblings(for: siblings)
        let wide = Self.matrixRoutedMatmul(
            rotated, weight: fused.weight, scales: fused.scales, biases: fused.biases,
            groupSize: groupSize, bits: bits, mode: mode, operands: fused.operands,
            widenOutput: widenOutput, sourceDType: sourceDType)
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
    if let routed = first.tensorRouteForward(
        x, siblings: packed, preSigned: false, widenOutput: widenOutput)
    {
        return routed
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

/// True when `sharedHadamardProjections` (or its pre-signed form) over an FP32
/// activation of `rows` rows would run these siblings on the prompt-width
/// tensor route, quantizing the activation with `forwardInt8`. Siblings must
/// share one ungrouped transform, as `sharedHadamardSiblings` returns them.
public func sharedHadamardTensorRouteTakesPrompt(
    _ siblings: [HadamardQuantizedLinear], rows: Int
) -> Bool {
    guard let first = siblings.first,
        siblings.allSatisfy({ $0.sharesInputTransform(with: first) })
    else { return false }
    return first.tensorRouteTakesPrompt(rows: rows, siblings: siblings)
}

/// `sharedHadamardProjections` for an input whose quantized rotation is
/// already formed: `activation` must be exactly the tuple `forwardInt8` would
/// return for the FP32 input reshaped to `[rows, k]` (codes `[rows, k]`,
/// scales and scaled sums `[rows, k / 128]`), and `leading` the input's shape
/// without its last axis. The siblings read it through the same prompt-width
/// matmul. Nil when the route does not take it (the caller then runs
/// `sharedHadamardProjections` on the input itself).
public func sharedHadamardProjectionsQuantized(
    _ activation: SignedBlockHadamard.Int8Activation, leading: [Int],
    _ siblings: [HadamardQuantizedLinear], widenOutput: Bool = true
) -> [MLXArray]? {
    guard let first = siblings.first,
        siblings.allSatisfy({ $0.sharesInputTransform(with: first) })
    else { return nil }
    return first.tensorRouteForwardQuantized(
        activation, rows: leading.reduce(1, *), leading: leading, siblings: siblings,
        widenOutput: widenOutput)
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
    if let routed = first.tensorRouteForward(
        signed, siblings: siblings, preSigned: true, widenOutput: widenOutput)
    {
        return routed
    }
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

/// The dtype `sharedHadamardProjections` (and its pre-signed form) rotates an
/// FP32 activation of `rows` rows into when these siblings run as one stacked
/// matrix-route matmul, or nil when that call takes another path (the tensor
/// route, or one matmul per sibling). Siblings as `sharedHadamardSiblings`
/// returns them.
public func sharedHadamardMatrixStackInputDType(
    _ siblings: [HadamardQuantizedLinear], rows: Int
) -> DType? {
    guard let first = siblings.first,
        siblings.allSatisfy({ $0.sharesInputTransform(with: first) }),
        !first.tensorRouteMayTake(rows: rows, siblings: siblings),
        first.fusedSiblingsApply(siblings, rows: rows, k: first.transform.width)
    else { return nil }
    return first.fusedSiblingsInputDType(siblings, rows: rows, sourceDType: .float32)
}

/// The stacked matrix-route projections of an FP32 activation whose rotation
/// (plain, or pre-signed) is already formed in the dtype
/// `sharedHadamardMatrixStackInputDType` returned: the same matmul
/// `sharedHadamardProjections` runs on the rotation it forms itself. Nil when
/// the stack does not take it.
public func sharedHadamardProjectionsRotated(
    _ rotated: MLXArray, _ siblings: [HadamardQuantizedLinear], widenOutput: Bool = true
) -> [MLXArray]? {
    guard let first = siblings.first,
        siblings.allSatisfy({ $0.sharesInputTransform(with: first) })
    else { return nil }
    return first.fusedSiblingsForward(
        rotated, siblings: siblings, widenOutput: widenOutput, sourceDType: .float32)
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

/// The row count from which a forward counts as prompt width: the timed
/// prefill and the seed prefill, never a verify window (at most 17 rows).
/// Paths measured at prompt width only gate on it (the composed causal
/// attention of a prompt's query blocks, the fresh recurrent state of a new
/// request's first chunk), so every verify-width path keeps its kernels.
/// `BONSAI_PROMPT_MIN_ROWS` overrides the default of 64.
public enum BonsaiPromptWidth {
    public static let minimumRows: Int = {
        let value = ProcessInfo.processInfo.environment["BONSAI_PROMPT_MIN_ROWS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return value.flatMap { Int($0) } ?? 64
    }()
}

/// Kernel inputs read in place through their strides. A fused-input rotation
/// that consumes a column slice of a stacked projection (the gate|up halves,
/// the GDN z of qkv|z) or a head-strided view (the attention output and its
/// gate) would otherwise be launched row-contiguous, and MLX copies every such
/// operand to a fresh buffer first (one copy launch each, at every verify
/// window). The kernels that opt in index their inputs through the strides
/// MLX passes instead: the same values, read where they already are.
/// `BONSAI_FUSED_INPUT_STRIDED=0` keeps the row-contiguous launches (and the
/// copies).
public enum HadamardStridedInputs {
    public static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_FUSED_INPUT_STRIDED"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Row addressing for an input whose leading `LEAD` dimensions (1 or 2)
    /// index the rows and whose trailing `TRAIL` dimensions (1 or 2) make up
    /// a row (the caller reshapes any other layout to `[rows, width]`, a view
    /// when MLX can collapse it). A kernel takes one uniform branch per
    /// threadgroup: packed rows (unit innermost stride, a two-dimensional row
    /// contiguous) keep the plain loads at `row base + column`; anything else
    /// forms each column's offset from the strides.
    public static let header = """
        template <int LEAD>
        METAL_FUNC int64_t bonsai_row_base(
            uint row, constant const int* shape, constant const int64_t* strides) {
          if (LEAD == 1) {
            return int64_t(row) * strides[0];
          }
          const uint n1 = uint(shape[1]);
          return (row < n1) ? int64_t(row) * strides[1]
                            : int64_t(row / n1) * strides[0] + int64_t(row % n1) * strides[1];
        }
        template <int LEAD, int TRAIL>
        METAL_FUNC int64_t bonsai_col_off(
            uint c, constant const int* shape, constant const int64_t* strides) {
          if (TRAIL == 1) {
            return int64_t(c) * strides[LEAD];
          }
          const uint n = uint(shape[LEAD + 1]);
          return int64_t(c / n) * strides[LEAD] + int64_t(c % n) * strides[LEAD + 1];
        }
        template <int LEAD, int TRAIL>
        METAL_FUNC bool bonsai_row_packed(constant const int* shape, constant const int64_t* strides) {
          if (TRAIL == 1) {
            return strides[LEAD] == 1;
          }
          return strides[LEAD + 1] == 1
              && (shape[LEAD] == 1 || strides[LEAD] == int64_t(shape[LEAD + 1]));
        }

        """
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

    static func applies(blockSize: Int, width: Int, dtype: DType) -> Bool {
        plainEnabled && blockSize == 1024 && width % 1024 == 0 && dtype == .float32
    }

    static func gated(
        _ a: MLXArray, _ b: MLXArray, signs: MLXArray, width: Int, mode: Int,
        outputDType: DType = .float32
    ) -> MLXArray {
        // Rows over the leading one or two dimensions, a row over the
        // trailing one (`[..., width]`) or two (`[B, S, heads, headDim]`: the
        // head-transposed attention output and the gate half of each q|gate
        // head); any other layout is reshaped to `[rows, width]`.
        var trail = a.dim(-1) == width ? 1 : 2
        var lead = a.ndim - trail
        var ra = a
        var rb = b
        if lead < 1 || lead > 2 || (trail == 2 && a.dim(-1) * a.dim(-2) != width)
            || b.shape != a.shape
        {
            ra = a.reshaped(a.size / width, width)
            rb = b.reshaped(a.size / width, width)
            trail = 1
            lead = 1
        }
        return gatedKernel(
            [ra, rb, signs],
            template: [
                ("WIDTH", width), ("MODE", mode), ("InT", a.dtype), ("OutT", outputDType),
                ("LEAD", lead), ("TRAIL", trail),
            ],
            grid: (64, a.size / 1024, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [Array(ra.shape.dropLast(trail)) + [width]],
            outputDTypes: [outputDType])[0]
    }

    /// MODE 1: `(a * sigmoid(a)) * b` (SwiGLU, inputs FP32 or FP16 widened
    /// exactly). MODE 2: `a * sigmoid(b)`. The inputs are read through their
    /// strides (`HadamardStridedInputs`: rows over the leading LEAD dims, a
    /// row over the trailing TRAIL dims).
    private static let gatedKernel = MLXFast.metalKernel(
        name: "bonsai_fused_input_gated_hadamard_1024",
        inputNames: ["a", "b", "signs"],
        outputNames: ["out"],
        source: """
            constexpr short NT = 64;
            constexpr uint BLOCKS = WIDTH / 1024;
            short i = short(thread_position_in_grid.x);
            uint blk = thread_position_in_grid.y;
            uint row_base = (blk / BLOCKS) * WIDTH;
            uint col0 = (blk % BLOCKS) * 1024;
            const uint row = blk / BLOCKS;
            const int64_t ra = bonsai_row_base<LEAD>(row, a_shape, a_strides);
            const int64_t rb = bonsai_row_base<LEAD>(row, b_shape, b_strides);

            threadgroup float buf[1024];

            // One uniform branch per threadgroup picks the loads (packed rows
            // at row base + column, anything else through the strides).
            auto fill = [&](auto load_a, auto load_b, auto load_s) {
            BONSAI_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                uint p = col0 + index + r;
                float av = load_a(p);
                float bv = load_b(p);
                float v;
                if (MODE == 1) {
                  float t = av * bonsai_sigmoid(av);
                  v = t * bv;
                } else {
                  v = av * bonsai_sigmoid(bv);
                }
                buf[index + r] = v * load_s(p);
              }
            }
            };
            if (bonsai_row_packed<LEAD, TRAIL>(a_shape, a_strides)
                && bonsai_row_packed<LEAD, TRAIL>(b_shape, b_strides) && signs_strides[0] == 1) {
              const device InT* ap = a + ra;
              const device InT* bp = b + rb;
              fill([&](uint p) { return static_cast<float>(ap[p]); },
                   [&](uint p) { return static_cast<float>(bp[p]); },
                   [&](uint p) { return signs[p]; });
            } else {
              fill([&](uint p) { return static_cast<float>(a[ra + bonsai_col_off<LEAD, TRAIL>(p, a_shape, a_strides)]); },
                   [&](uint p) { return static_cast<float>(b[rb + bonsai_col_off<LEAD, TRAIL>(p, b_shape, b_strides)]); },
                   [&](uint p) { return signs[int64_t(p) * signs_strides[0]]; });
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

            """ + HadamardStridedInputs.header,
        ensureRowContiguous: !HadamardStridedInputs.enabled)
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
        return gatedRMSNormKernel(
            [x, z, weight, signs, MLXArray(eps)],
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
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            // x and z [B, S, heads, HEAD_DIM], read through their strides.
            const uint row = blk / BLOCKS;
            const int64_t rx = bonsai_row_base<2>(row, x_shape, x_strides);
            const int64_t rz = bonsai_row_base<2>(row, z_shape, z_strides);

            threadgroup float buf[1024];
            threadgroup float inv_rms[HEADS_PER_BLOCK];

            // One uniform branch per threadgroup picks the loads (packed rows
            // at row base + column, anything else through the strides).
            auto fill = [&](auto load_x, auto load_z, auto load_w, auto load_s) {
            // Per-head RMS as rms_single_row with 32 threads x 4 reads: lane l
            // sums elements 4l..4l+3 of the head in order, then simd_sum.
            BONSAI_UNROLL for (uint hh = sg; hh < HEADS_PER_BLOCK; hh += 2) {
              uint p0 = col0 + hh * HEAD_DIM;
              uint kh = p0 / (REPEATS * HEAD_DIM);
              uint rep = (p0 % (REPEATS * HEAD_DIM)) / HEAD_DIM;
              uint src_head = rep * KEY_HEADS + kh;
              uint xh = src_head * HEAD_DIM + lane * 4;
              float acc = 0;
              float tx[4];
              BONSAI_UNROLL for (int r = 0; r < 4; r++) {
                tx[r] = load_x(xh + r);
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
                float xn = load_w(src % HEAD_DIM) * (load_x(src) * inv_rms[(index + r) / HEAD_DIM]);
                float zv = load_z(src);
                float gz = zv * bonsai_sigmoid(zv);
                float v = gz * xn;
                buf[index + r] = v * load_s(p);
              }
            }
            };
            if (bonsai_row_packed<2, 2>(x_shape, x_strides) && bonsai_row_packed<2, 2>(z_shape, z_strides)
                && w_strides[0] == 1 && signs_strides[0] == 1) {
              const auto xp = x + rx;
              const auto zp = z + rz;
              fill([&](uint c) { return float(xp[c]); }, [&](uint c) { return float(zp[c]); },
                   [&](uint c) { return w[c]; }, [&](uint c) { return signs[c]; });
            } else {
              fill([&](uint c) { return float(x[rx + bonsai_col_off<2, 2>(c, x_shape, x_strides)]); },
                   [&](uint c) { return float(z[rz + bonsai_col_off<2, 2>(c, z_shape, z_strides)]); },
                   [&](uint c) { return w[int64_t(c) * w_strides[0]]; },
                   [&](uint c) { return signs[int64_t(c) * signs_strides[0]]; });
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

            """ + HadamardStridedInputs.header,
        ensureRowContiguous: !HadamardStridedInputs.enabled)
}
