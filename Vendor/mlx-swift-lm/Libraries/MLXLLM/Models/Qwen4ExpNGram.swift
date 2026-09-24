//
//  Qwen4ExpNGram.swift
//  mlx-swift-lm
//
//  The sharded n-gram embedding and the PLE layer of Qwen 3.8 Flash-Next.
//
//  REFERENCE AND LICENSE. Swift port of the MIT-licensed mlx-lm reference
//  implementation: ml-explore/mlx-lm PR #1788 `mlx_lm/models/qwen4_exp.py` at
//  head c961f839 (`NGramEmbedding`, `_ShardedEmbedding`, `PLELayer`).
//  No AGPL-licensed source was read or ported.
//
//  WHY THE TABLE IS A SEAM. The n-gram table of this checkpoint is 128 shards
//  of 2,500,012 rows of 160 values, quantized 4-bit affine with group size 32.
//  That is 100 bytes per row and 29.8 GiB in total -- more than a third of the
//  whole model, and more than a 128 GB machine can hold beside the rest of the
//  tower. So this file does NOT declare the shard tensors as parameters. It
//  declares only the hash, and asks an injected `Qwen4ExpNGramRowSource` for
//  the rows it needs. The engine installs a source that keeps the rows on the
//  solid-state disk and caches a bounded number of them.
//  See docs/ngram-cache-design.md.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Row source seam

/// Supplies n-gram embedding rows by global row id.
///
/// CONTRACT. `rows(globalIds:)` must return the SAME values for the same ids
/// on every call, whatever it does internally to find them. A cache, a cache
/// size, or an eviction order must never change a returned row. The row values
/// are the dequantized checkpoint rows, in the checkpoint's own row order.
public protocol Qwen4ExpNGramRowSource: AnyObject {
    /// Width of one row.
    var rowDimensions: Int { get }

    /// Rows for `globalIds`, an integer array of any shape.
    /// - Returns: an array shaped `globalIds.shape + [rowDimensions]`.
    func rows(globalIds: MLXArray) -> MLXArray
}

/// Holder that keeps the installed row source OUT of the module parameter
/// tree. A bare property would be reported by `Module.items()` and, if the
/// source happened to be a `Module`, its parameters would be walked as if they
/// belonged to the model.
/// A row source that can gather from HOST ids.
///
/// WHY. The row ids are a pure function of the last `ngramSize` token ids,
/// which the host already holds. Hashing them on the device and reading the
/// result back put a GPU→CPU→GPU sync in the MIDDLE of every decode step's
/// graph (the layers before the PLE layer were evaluated alone, then the
/// rest), and the table's gather is host work anyway. A source that takes
/// host ids lets the embedding hash on the host, gather, and hand the
/// device ONE input array, so the step submits as one graph.
public protocol Qwen4ExpNGramHostRowSource: Qwen4ExpNGramRowSource {
    /// Rows for `globalIds` (row-major over `shape`), shaped
    /// `shape + [rowDimensions]`.
    func rows(globalIds: [Int], shape: [Int]) -> MLXArray
}

public final class Qwen4ExpNGramRowSourceHolder {
    public var source: Qwen4ExpNGramRowSource?
    public init() {}
}

// MARK: - Hash constants

/// Host-side constants of the n-gram hash.
///
/// These are also stored in the checkpoint (`layer_multipliers`,
/// `ngram_heads_vocab_sizes`, `ngram_heads_offsets`), and the module below
/// declares those tensors so a load can absorb them. The values USED are these
/// ones, rebuilt from the configuration, because the checkpoint copies sit in
/// the parameter tree where a dtype sweep would destroy them.
final class Qwen4ExpNGramConstants {
    let multipliers: MLXArray
    let sizes: MLXArray
    let offsets: MLXArray
    let headVocabSizes: [Int]
    let headOffsets: [Int]
    let rowsPerShard: Int
    let shardCount: Int

    init(_ args: Qwen4ExpTextConfiguration, pleLayerIndex: Int) {
        let ngramHeads = (args.ngramSize - 1) * args.headsPerNGram

        var sizes: [Int] = []
        var offsets: [Int] = []
        var total = 0
        for head in 0 ..< ngramHeads {
            let global = pleLayerIndex * ngramHeads + head
            let size = Qwen4ExpNGramConstants.nthPrimeAfter(
                args.ngramVocabSizeBase - 1, count: global + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        }
        self.headVocabSizes = sizes
        self.headOffsets = offsets

        let divisor = args.makeNGramVocabSizeDivisibleBy
        let padded = ((total + divisor - 1) / divisor) * divisor
        self.shardCount = args.splitNGramParts
        self.rowsPerShard = (padded + shardCount - 1) / shardCount

        // The reference builds the multipliers from a splitmix64 stream seeded
        // by the configuration seed and the PLE layer index.
        let gamma: UInt64 = 0x9E37_79B9_7F4A_7C15
        let maxLong = UInt64(Int64.max)
        let half = Swift.max(UInt64(1), (maxLong / UInt64(Swift.max(args.vocabularySize, 1))) / 2)
        let baseSeed = UInt64(bitPattern: Int64(args.seed)) &+ (10007 &* UInt64(pleLayerIndex))
        var multipliers: [Int64] = []
        for i in 0 ..< args.ngramSize {
            let mixed = Qwen4ExpNGramConstants.splitmix64(baseSeed &+ (gamma &* UInt64(i + 1)))
            multipliers.append(Int64(2 &* (mixed % half) &+ 1))
        }

        self.multipliers = MLXArray(multipliers)
        self.sizes = MLXArray(sizes.map { Int64($0) })
        self.offsets = MLXArray(offsets.map { Int64($0) })
    }

    static func splitmix64(_ value: UInt64) -> UInt64 {
        var v = value &+ 0x9E37_79B9_7F4A_7C15
        v = (v ^ (v >> 30)) &* 0xBF58_476D_1CE4_E5B9
        v = (v ^ (v >> 27)) &* 0x94D0_49BB_1331_11EB
        return v ^ (v >> 31)
    }

    static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var d = 3
        while d * d <= value {
            if value % d == 0 { return false }
            d += 2
        }
        return true
    }

    static func nthPrimeAfter(_ start: Int, count: Int) -> Int {
        var p = start
        for _ in 0 ..< count {
            p += 1
            while !isPrime(p) { p += 1 }
        }
        return p
    }
}

// MARK: - N-gram embedding

/// Hashes the recent 2- and 3-token history into row ids and looks the rows up.
///
/// One row id per head: `heads_per_ngram` heads read the 2-token history and
/// `heads_per_ngram` more read the 3-token history, so the lookup returns
/// `(ngram_size - 1) * heads_per_ngram` rows per token, concatenated into one
/// `ple_embed_dim` vector.
public final class Qwen4ExpNGramEmbedding: Module {
    let ngramSize: Int
    let headsPerNGram: Int
    let ngramHeads: Int
    let eosTokenId: Int
    let headDimensions: Int

    /// Checkpoint buffers. Declared so a strict load can place them; the values
    /// used are in `constants`.
    @ParameterInfo(key: "layer_multipliers") var layerMultipliers: MLXArray
    @ParameterInfo(key: "ngram_heads_vocab_sizes") var ngramHeadsVocabSizes: MLXArray
    @ParameterInfo(key: "ngram_heads_offsets") var ngramHeadsOffsets: MLXArray

    let constants: Qwen4ExpNGramConstants
    public let rowSourceHolder = Qwen4ExpNGramRowSourceHolder()

    public init(_ args: Qwen4ExpTextConfiguration, embedDimensions: Int, pleLayerIndex: Int) {
        self.ngramSize = args.ngramSize
        self.headsPerNGram = args.headsPerNGram
        self.ngramHeads = (args.ngramSize - 1) * args.headsPerNGram
        self.eosTokenId = args.eosTokenId
        self.headDimensions = embedDimensions / ngramHeads
        self.constants = Qwen4ExpNGramConstants(args, pleLayerIndex: pleLayerIndex)

        _layerMultipliers.wrappedValue = MLXArray.zeros([args.ngramSize], dtype: .int64)
        _ngramHeadsVocabSizes.wrappedValue = MLXArray.zeros([ngramHeads], dtype: .int64)
        _ngramHeadsOffsets.wrappedValue = MLXArray.zeros([ngramHeads], dtype: .int64)
        super.init()
    }

    /// Number of rows one shard holds. The engine needs it to split a global
    /// row id into a shard and a row.
    public var rowsPerShard: Int { constants.rowsPerShard }
    public var shardCount: Int { constants.shardCount }
    public var rowDimensions: Int { headDimensions }

    /// Install the row source. Without one the forward pass refuses.
    public func install(rowSource: Qwen4ExpNGramRowSource) {
        precondition(
            rowSource.rowDimensions == headDimensions,
            """
            Qwen4ExpNGramEmbedding: the row source serves rows of \
            \(rowSource.rowDimensions) values; this table has \(headDimensions).
            """)
        rowSourceHolder.source = rowSource
    }

    /// Shift right by `shift` without crossing an end-of-sequence boundary.
    func shiftRight(_ ids: MLXArray, by shift: Int) -> MLXArray {
        if shift == 0 { return ids }
        let B = ids.dim(0)
        let T = ids.dim(1)
        let eos = MLXArray(int64: eosTokenId)
        let positions = MLXArray(Int32(0) ..< Int32(T)).asType(.int64)

        let eosAt = MLX.where(ids .== eos, positions, MLXArray(int64: -1))
        let previousInclusive = cummax(eosAt, axis: 1)
        let previous = concatenated(
            [
                MLXArray.full([B, 1], values: MLXArray(int64: -1), dtype: previousInclusive.dtype),
                previousInclusive[0..., ..<(T - 1)],
            ], axis: 1)
        let inSegment = positions[.newAxis] - (previous + MLXArray(int64: 1))

        let source = positions - MLXArray(int64: shift)
        let clamped = maximum(source, MLXArray(int64: 0))
        let gathered = takeAlong(ids, MLX.broadcast(clamped[.newAxis], to: [B, T]), axis: 1)
        let usable =
            (inSegment .>= MLXArray(int64: shift)) & (source[.newAxis] .>= MLXArray(int64: 0))
        return MLX.where(usable, gathered, eos)
    }

    /// Global row ids for the `ids` tokens, shaped `[B, T, ngramHeads]`.
    public func rowIds(_ ids: MLXArray, previousContext: MLXArray) -> MLXArray {
        let newCount = ids.dim(1)
        let history = concatenated([previousContext, ids], axis: 1).asType(.int64)
        let shifted = (0 ..< ngramSize).map { shiftRight(history, by: $0) }

        var blocks: [MLXArray] = []
        for ngram in 2 ... ngramSize {
            let low = (ngram - 2) * headsPerNGram
            let high = low + headsPerNGram
            var mixed = shifted[0] * constants.multipliers[0]
            for p in 1 ..< ngram {
                mixed = bitwiseXOr(mixed, shifted[p] * constants.multipliers[p])
            }
            let sizes = constants.sizes[low ..< high].reshaped(1, 1, -1)
            let offsets = constants.offsets[low ..< high].reshaped(1, 1, -1)
            blocks.append(mixed[.ellipsis, .newAxis] % sizes + offsets)
        }
        return concatenated(blocks, axis: -1)[0..., (-newCount)..., 0...]
    }

    /// `rowIds` on the host: the same hash, over host integers, for a batch
    /// of histories (`previousContext ++ ids` per row). Returns
    /// `[B][newCount][ngramHeads]` flattened row-major. Bit-for-bit the
    /// device result: wrapping Int64 multiply, XOR, and Python-style
    /// remainder (sign of the divisor), which is what MLX's `%` computes.
    public func hostRowIds(history: [[Int64]], newCount: Int) -> [Int] {
        let eos = Int64(eosTokenId)
        let multipliers = constants.multipliers.asArray(Int64.self)
        let sizes = constants.headVocabSizes.map(Int64.init)
        let offsets = constants.headOffsets.map(Int64.init)
        var out: [Int] = []
        out.reserveCapacity(history.count * newCount * ngramHeads)
        for row in history {
            let T = row.count
            // previous[t] = position of the last EOS strictly before t, or -1.
            var previous = [Int](repeating: -1, count: T)
            var last = -1
            for t in 0 ..< T {
                previous[t] = last
                if row[t] == eos { last = t }
            }
            func shifted(_ s: Int, _ t: Int) -> Int64 {
                if s == 0 { return row[t] }
                let inSegment = t - (previous[t] + 1)
                let source = t - s
                return (inSegment >= s && source >= 0) ? row[source] : eos
            }
            for t in Swift.max(0, T - newCount) ..< T {
                for ngram in 2 ... ngramSize {
                    var mixed = shifted(0, t) &* multipliers[0]
                    for p in 1 ..< ngram { mixed ^= shifted(p, t) &* multipliers[p] }
                    let low = (ngram - 2) * headsPerNGram
                    for head in low ..< low + headsPerNGram {
                        var r = mixed % sizes[head]
                        if r != 0, (r < 0) != (sizes[head] < 0) { r += sizes[head] }
                        out.append(Int(r + offsets[head]))
                    }
                }
            }
        }
        return out
    }

    public func callAsFunction(_ ids: MLXArray, previousContext: MLXArray) -> MLXArray {
        if let host = rowSourceHolder.source as? Qwen4ExpNGramHostRowSource {
            // Host path: the ids are inputs (the token fed, and the staged
            // context), so reading them evaluates nothing of this step's
            // graph; the rows come back as ONE input array.
            let B = ids.dim(0)
            let newCount = ids.dim(1)
            let context = previousContext.asType(.int64).asArray(Int64.self)
            let tokens = ids.asType(.int64).asArray(Int64.self)
            let contextLength = previousContext.dim(1)
            var history: [[Int64]] = []
            history.reserveCapacity(B)
            for b in 0 ..< B {
                history.append(
                    Array(context[(b * contextLength) ..< ((b + 1) * contextLength)])
                        + Array(tokens[(b * newCount) ..< ((b + 1) * newCount)]))
            }
            let gid = hostRowIds(history: history, newCount: newCount)
            return host.rows(globalIds: gid, shape: [B, newCount, ngramHeads])
                .reshaped(B, newCount, -1)
        }
        guard let source = rowSourceHolder.source else {
            preconditionFailure(
                """
                Qwen4ExpNGramEmbedding: no row source is installed. The n-gram \
                table is 29.8 GiB and is never held as model parameters; the \
                runtime must call install(rowSource:) before the first forward \
                pass. See docs/ngram-cache-design.md.
                """)
        }
        let gid = rowIds(ids, previousContext: previousContext)
        let rows = source.rows(globalIds: gid)
        return rows.reshaped(gid.dim(0), gid.dim(1), -1)
    }
}

// MARK: - PLE layer

/// Per-layer embedding block. It reads the n-gram embedding of the recent
/// history, gates it against the current hidden state, and adds a dilated
/// short convolution over the result.
public final class Qwen4ExpPLELayer: Module {
    let hiddenSize: Int
    let hcCount: Int
    let dilation: Int
    let shortConvStateLength: Int

    @ModuleInfo(key: "ple_embedding") public var pleEmbedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpRMSNorm
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    public init(_ args: Qwen4ExpTextConfiguration, pleLayerIndex: Int) {
        self.hiddenSize = args.hiddenSize
        self.hcCount = args.hcCount
        self.dilation = args.ngramSize
        self.shortConvStateLength = (args.pleConvKernelSize - 1) * args.ngramSize
        let wide = args.hiddenSize * args.hcCount

        _pleEmbedding.wrappedValue = Qwen4ExpNGramEmbedding(
            args, embedDimensions: args.pleEmbedDim, pleLayerIndex: pleLayerIndex)
        _keyProj.wrappedValue = Linear(args.pleEmbedDim, wide, bias: false)
        _valueProj.wrappedValue = Linear(args.pleEmbedDim, args.hiddenSize, bias: false)
        _normKey.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: wide, groupSize: args.hiddenSize, eps: args.rmsNormEps,
            weightOffset: args.rmsNormWeightOffset)
        _normQuery.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: wide, groupSize: args.hiddenSize, eps: args.rmsNormEps,
            weightOffset: args.rmsNormWeightOffset)
        _normConv.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: wide, groupSize: args.hiddenSize, eps: args.rmsNormEps,
            weightOffset: args.rmsNormWeightOffset)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: wide,
            outputChannels: wide,
            kernelSize: args.pleConvKernelSize,
            stride: 1,
            padding: 0,
            dilation: args.ngramSize,
            groups: wide,
            bias: false
        )
        super.init()
    }

    private func shortConv(_ x: MLXArray, cache: Qwen4ExpLayerCache?) -> MLXArray {
        let S = x.dim(1)
        let n = shortConvStateLength
        let state =
            cache?[Qwen4ExpLayerCache.pleConvSlot]
            ?? MLXArray.zeros([x.dim(0), n, x.dim(-1)], dtype: x.dtype)
        let full = concatenated([state, x], axis: 1)
        if let cache {
            cache[Qwen4ExpLayerCache.pleConvSlot] = contiguous(full[0..., (-n)..., 0...])
        }
        return silu(conv1d(full[0..., (-(n + S))..., 0...]))
    }

    public func callAsFunction(
        _ hidden: MLXArray,
        ids: MLXArray,
        previousContext: MLXArray,
        cache: Qwen4ExpLayerCache?
    ) -> MLXArray {
        let embedded = pleEmbedding(ids, previousContext: previousContext).asType(hidden.dtype)
        let keyFlat = normKey(keyProj(embedded))
        let key = keyFlat.reshaped(keyFlat.shape.dropLast() + [hcCount, hiddenSize])
        let value = valueProj(embedded)
        let queryFlat = normQuery(hidden)
        let query = queryFlat.reshaped(queryFlat.shape.dropLast() + [hcCount, hiddenSize])

        var gate = (key * query).sum(axis: -1, keepDims: true) / Foundation.sqrt(Float(hiddenSize))
        // The floor is a scalar IN THE GATE'S DTYPE. `MLXArray(Float(1e-6))` is
        // a float32 array, not a weak scalar, and `maximum` promotes: the gate,
        // the gated value and the PLE output become float32, and `stream + ple`
        // then carries the whole hyper stream in float32 for every layer after
        // this one. The reference (`gate.abs().clamp_min(1e-6)`) keeps the
        // model dtype.
        let floor = MLXArray(Float(1e-6), dtype: gate.dtype)
        gate = MLX.sqrt(maximum(MLX.abs(gate), floor)) * MLX.sign(gate)

        var gated = sigmoid(gate) * value[.ellipsis, .newAxis, 0...]
        gated = gated.reshaped(gated.shape.dropLast(2) + [-1])
        return gated + shortConv(normConv(gated), cache: cache)
    }
}
