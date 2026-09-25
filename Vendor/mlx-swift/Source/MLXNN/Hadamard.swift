// Adapted from PrismML-Eng/mlx-swift 6d3a84de28225d1f5bc0a56f5c781596997242f9 (MIT).
// Preserve the published Bonsai pack's FP32 transform / original output dtype contract.
import Foundation
import MLX

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

    /// On unless explicitly disabled. Signs, the FP32 block transform, its
    /// scale and the dtype restore run as one dispatch instead of a sign
    /// multiply followed by the stock transform. The butterfly stages, their
    /// order and the scale match the stock kernel, so outputs are identical.
    private static let fusedSignedTransform: Bool = {
        let value = ProcessInfo.processInfo.environment[
            "DARKBLOOM_BONSAI_FUSED_SIGNED_HADAMARD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    private static let warmth = FusedTransformWarmth()

    /// Blocks of 4^k keep the stock 1/sqrt(N) scale an exact power of two.
    private var usesFusedTransform: Bool {
        Self.fusedSignedTransform && blockSize >= 16
            && blockSize.trailingZeroBitCount.isMultiple(of: 2)
            && Device.defaultDevice().deviceType == .gpu
    }

    /// Builds the fused kernel for every activation dtype once per block
    /// size, at load, so no timed forward pays a Metal library build for it.
    public func warmFusedTransform() {
        guard usesFusedTransform, Self.warmth.claim(blockSize) else { return }
        let outputs: [MLXArray] = [DType.float32, .bfloat16, .float16].map {
            self.callAsFunction(MLXArray.zeros([1, width], dtype: $0))
        }
        eval(outputs)
    }

    /// Transform activations before multiplication by folded weights.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        validate(x)
        if usesFusedTransform, x.size > 0 {
            let lanes = blockSize / 16
            return signedBlockHadamardKernel(
                [x, signs],
                template: [("OutT", x.dtype), ("N", blockSize)],
                grid: (lanes, width / blockSize, x.size / width),
                threadGroup: (lanes, 1, 1),
                outputShapes: [x.shape],
                outputDTypes: [x.dtype])[0]
        }
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

/// Block sizes whose fused rotation kernels were already built at load.
private final class FusedTransformWarmth: @unchecked Sendable {
    private let lock = NSLock()
    private var blockSizes = Set<Int>()

    func claim(_ blockSize: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return blockSizes.insert(blockSize).inserted
    }
}

/// `(x * signs) H` for one block per threadgroup, in one dispatch. The grid is
/// (lanes, blocks per row, rows): a threadgroup's y position is its block
/// within the row and selects the matching slice of the sign vector.
///
/// This is the stock `hadamard_n` schedule (radix 16, four-wide reads, the
/// same stage order and butterfly operands, FP32 threadgroup buffer), with the
/// sign multiply applied as each element is read and the output dtype
/// conversion applied as each element is written. The butterflies run with
/// reassociation and contraction off, so each rounds exactly as in the stock
/// kernel; the sign and power-of-two scale multiplies are exact.
private let signedBlockHadamardKernel = MLXFast.metalKernel(
    name: "prism_signed_block_hadamard",
    inputNames: ["x", "signs"],
    outputNames: ["out"],
    source: """
        constexpr int R = 16;
        constexpr int W = 4;
        constexpr int lanes = N / R;
        constexpr int logN = __builtin_ctz(N);
        constexpr int logR = 4;
        constexpr int steps = logN / logR;
        constexpr int logFinal = logN % logR;
        constexpr int finalRadix = 1 << logFinal;
        constexpr float scale = 1.0f / float(1 << (logN / 2));

        threadgroup float buf[N];

        const int i = int(thread_position_in_threadgroup.x);
        const int blockInRow = int(threadgroup_position_in_grid.y);
        const size_t block = size_t(threadgroup_position_in_grid.z) * threadgroups_per_grid.y
            + size_t(blockInRow);
        const size_t base = block * size_t(N);
        const int signBase = blockInRow * N;

        #pragma clang loop unroll(full)
        for (int j = 0; j < R / W; j++) {
          const int index = j * W * lanes + i * W;
          #pragma clang loop unroll(full)
          for (int r = 0; r < W; r++) {
            buf[index + r] =
                static_cast<float>(x[base + index + r]) * signs[signBase + index + r];
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float v[R];
        int h = 1;
        #pragma clang loop unroll(full)
        for (int s = 0; s < steps; s++) {
          const int k = i & (h - 1);
          const int j = ((i - k) << logR) + k;
          #pragma clang loop unroll(full)
          for (int r = 0; r < R; r++) {
            v[r] = buf[j + h * r];
          }
          prism_signed_hadamard_radix<R>(v);
          #pragma clang loop unroll(full)
          for (int r = 0; r < R; r++) {
            buf[j + h * r] = v[r];
          }
          h <<= logR;
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (finalRadix > 1) {
          #pragma clang loop unroll(full)
          for (int t = 0; t < R / finalRadix; t++) {
            const int index = i + t * lanes;
            const int k = index & (h - 1);
            const int j = ((index - k) << logFinal) + k;
            #pragma clang loop unroll(full)
            for (int r = 0; r < finalRadix; r++) {
              v[r] = buf[j + h * r];
            }
            prism_signed_hadamard_radix<finalRadix>(v);
            #pragma clang loop unroll(full)
            for (int r = 0; r < finalRadix; r++) {
              buf[j + h * r] = v[r];
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        #pragma clang loop unroll(full)
        for (int j = 0; j < R / W; j++) {
          const int index = j * W * lanes + i * W;
          #pragma clang loop unroll(full)
          for (int r = 0; r < W; r++) {
            out[base + index + r] = static_cast<OutT>(buf[index + r] * scale);
          }
        }
        """,
    header: """
        template <int R>
        inline void prism_signed_hadamard_radix(thread float* v) {
          #pragma clang fp reassociate(off)
          #pragma clang fp contract(off)
          constexpr int logR = __builtin_ctz(R);
          int h = 1;
          #pragma clang loop unroll(full)
          for (int s = 0; s < logR; s++) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < R / 2; i++) {
              const int k = i & (h - 1);
              const int j = ((i - k) << 1) + k;
              const float a = v[j];
              const float b = v[j + h];
              v[j] = a + b;
              v[j + h] = a - b;
            }
            h <<= 1;
          }
        }
        """)

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
        transform.warmFusedTransform()
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
        if permitsFloat16ConstantReuse && rotated.dtype == .float32 {
            return constantCachedForward(rotated, allowFloat16: true)
        }
        return super.callAsFunction(rotated)
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
