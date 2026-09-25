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
    static let occupancy = 1024

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
