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

    /// Transform activations before multiplication by folded weights.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        validate(x)
        if SignedHadamardKernel.applies(blockSize: blockSize, width: width, dtype: x.dtype) {
            return SignedHadamardKernel.apply(x, signs: signs, width: width, layout: nil)
        }
        return hadamardTransform((x.asType(.float32) * signs).reshaped([-1, blockSize]))
            .reshaped(x.shape).asType(x.dtype)
    }

    /// `self(silu(gate) * up)` with the SwiGLU product formed inside the fused
    /// rotation's read. Bit-identical to the composed path: the product is the
    /// compiled `x * sigmoid(x)` followed by `* up`, each rounded to FP32 in the
    /// same order, with MLX's own `Sigmoid` formula. Nil when it does not apply.
    public func rotatedSwiGLU(gate: MLXArray, up: MLXArray) -> MLXArray? {
        guard SignedHadamardKernel.appliesGated(
            blockSize: blockSize, width: width, a: gate, b: up)
        else { return nil }
        return SignedHadamardKernel.applyGated(gate, up, signs: signs, width: width, mode: 1)
    }

    /// `self(x * sigmoid(gate))`, the attention output gate, fused the same way.
    public func rotatedSigmoidGate(_ x: MLXArray, gate: MLXArray) -> MLXArray? {
        guard SignedHadamardKernel.appliesGated(
            blockSize: blockSize, width: width, a: x, b: gate)
        else { return nil }
        return SignedHadamardKernel.applyGated(x, gate, signs: signs, width: width, mode: 2)
    }

    /// `callAsFunction(layout(x))`, with the GDN value permutation folded into
    /// the fused kernel's reads instead of materialized as its own copy.
    public func callAsFunction(_ x: MLXArray, layout: HadamardGDNLayout?) -> MLXArray {
        guard let layout else { return self(x) }
        validate(x)
        if SignedHadamardKernel.applies(blockSize: blockSize, width: width, dtype: x.dtype) {
            return SignedHadamardKernel.apply(x, signs: signs, width: width, layout: layout)
        }
        return self(layout(x))
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
        transform(x, layout: gdnLayout)
    }

    /// The packed matmul on an input already passed through `rotate`.
    public func applyRotated(_ rotated: MLXArray) -> MLXArray {
        if permitsFloat16ConstantReuse && rotated.dtype == .float32 {
            return constantCachedForward(rotated, allowFloat16: true)
        }
        return super.callAsFunction(rotated)
    }

    /// `self(silu(gate) * up)`, with the gating fused into the input rotation.
    public func applyAfterSwiGLU(gate: MLXArray, up: MLXArray) -> MLXArray? {
        guard gdnLayout == nil, let rotated = transform.rotatedSwiGLU(gate: gate, up: up)
        else { return nil }
        return applyRotated(rotated)
    }

    /// `self(x * sigmoid(gate))`, with the gating fused into the input rotation.
    public func applyAfterSigmoidGate(_ x: MLXArray, gate: MLXArray) -> MLXArray? {
        guard gdnLayout == nil, let rotated = transform.rotatedSigmoidGate(x, gate: gate)
        else { return nil }
        return applyRotated(rotated)
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

/// The signed block Walsh-Hadamard transform in ONE pass for FP32
/// activations. The composed path is an element-wise `x * signs` dispatch
/// followed by MLX's `hadamard_n` kernel, so every rotation reads and writes
/// the activation twice (and the GDN output layout adds a third copy). This
/// kernel is MLX's `hadamard_n<float, 1024, 16, 4>` with the sign multiply
/// (exact: the signs are +1 or -1) and the GDN value permutation moved into
/// its read. The butterfly network, its operand order, the threadgroup
/// staging, the full unrolling and the final `* 1/sqrt(1024)` are unchanged, and both kernels
/// compile in MLX's safe math mode, so the output is bit-identical.
enum SignedHadamardKernel {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["BONSAI_FUSED_HADAMARD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static func applies(blockSize: Int, width: Int, dtype: DType) -> Bool {
        enabled && blockSize == 1024 && width % 1024 == 0 && dtype == .float32
    }

    static func apply(
        _ x: MLXArray, signs: MLXArray, width: Int, layout: HadamardGDNLayout?
    ) -> MLXArray {
        let blocks = x.size / 1024
        let repeats = layout.map { $0.valueHeads / $0.keyHeads } ?? 1
        let permuted = repeats > 1
        return kernel(
            [x, signs],
            template: [
                ("WIDTH", width),
                ("PERMUTE", permuted),
                ("REPEATS", repeats),
                ("KEY_HEADS", layout?.keyHeads ?? 1),
                ("HEAD_DIM", layout.map { $0.width / $0.valueHeads } ?? 1),
            ],
            grid: (64, blocks, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [.float32])[0]
    }

    static func appliesGated(blockSize: Int, width: Int, a: MLXArray, b: MLXArray) -> Bool {
        applies(blockSize: blockSize, width: width, dtype: a.dtype)
            && b.dtype == .float32 && a.shape == b.shape && a.ndim > 0 && a.dim(-1) == width
    }

    static func applyGated(
        _ a: MLXArray, _ b: MLXArray, signs: MLXArray, width: Int, mode: Int
    ) -> MLXArray {
        gatedKernel(
            [a, b, signs],
            template: [("WIDTH", width), ("MODE", mode)],
            grid: (64, a.size / 1024, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [a.shape],
            outputDTypes: [.float32])[0]
    }

    private static let kernel = MLXFast.metalKernel(
        name: "bonsai_signed_hadamard_1024",
        inputNames: ["x", "signs"],
        outputNames: ["out"],
        source: """
            // hadamard_n<float, N = 1024, max_radix = 16, read_width = 4>
            constexpr short NT = 64;
            constexpr uint BLOCKS = WIDTH / 1024;
            short i = short(thread_position_in_grid.x);
            uint blk = thread_position_in_grid.y;
            uint row_base = (blk / BLOCKS) * WIDTH;
            uint col0 = (blk % BLOCKS) * 1024;

            threadgroup float buf[1024];

            BONSAI_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                uint p = col0 + index + r;
                uint src = p;
                if (PERMUTE) {
                  uint kh = p / (REPEATS * HEAD_DIM);
                  uint rem = p % (REPEATS * HEAD_DIM);
                  src = ((rem / HEAD_DIM) * KEY_HEADS + kh) * HEAD_DIM + rem % HEAD_DIM;
                }
                buf[index + r] = x[row_base + src] * signs[p];
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
                out[row_base + col0 + index + r] = buf[index + r] * 0.03125f;
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

            """)

    /// MODE 1: `(a * sigmoid(a)) * b` (SwiGLU). MODE 2: `a * sigmoid(b)`.
    private static let gatedKernel = MLXFast.metalKernel(
        name: "bonsai_gated_signed_hadamard_1024",
        inputNames: ["a", "b", "signs"],
        outputNames: ["out"],
        source: """
            constexpr short NT = 64;
            constexpr uint BLOCKS = WIDTH / 1024;
            short i = short(thread_position_in_grid.x);
            uint blk = thread_position_in_grid.y;
            uint row_base = (blk / BLOCKS) * WIDTH;
            uint col0 = (blk % BLOCKS) * 1024;

            threadgroup float buf[1024];

            BONSAI_UNROLL for (short j = 0; j < 4; j++) {
              short index = j * 4 * NT + i * 4;
              BONSAI_UNROLL for (short r = 0; r < 4; r++) {
                uint p = col0 + index + r;
                float av = a[row_base + p];
                float bv = b[row_base + p];
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
                out[row_base + col0 + index + r] = buf[index + r] * 0.03125f;
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
