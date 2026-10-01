// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLMCommon

/// Text factory for Prism's folded Qwen3.8-27B pack.
///
/// The pack declares a vision tower. This track never serves an image, so the
/// load filter drops the 333 `vision_tower.*` tensors and the tower is never
/// built. `Qwen35Runner` declares `multimodal: false` for the same reason, so
/// the LLM factory is the only factory that resolves this model type.
///
/// The pack carries no MTP head (`mtp_num_hidden_layers: 0`). The track
/// attaches a separate, published head instead, so the speculative capability
/// stays as `Qwen35Model` declares it and the runner decides whether a head is
/// present. See `docs/bonsai2.md`.
public final class PrismHadamardQwen35TextModel: Qwen35Model, PrismHadamardLoading,
    WeightNameFiltering
{
    public let prismCheckpoint: PrismHadamardCheckpointConfiguration

    public init(configurationData: Data) throws {
        prismCheckpoint = try JSONDecoder().decode(PrismHadamardCheckpointConfiguration.self, from: configurationData)
        super.init(try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configurationData))
    }
    public func shouldLoadWeight(named name: String) -> Bool {
        !name.hasPrefix("vision_tower.")
    }
}

/// `x @ w.T` for a verify-width FP32 `x` (at most 16 rows) and a narrow FP32
/// `w` `[N, K]` (N a multiple of 32, K of 128): the GDN layers' stacked
/// `in_proj_b | in_proj_a` (N = 96, K = 5120).
///
/// MLX's GEMM gives this shape three threadgroups (N / 32 by one row tile)
/// that each walk all of K, so on the verify window it costs about as much
/// as a packed projection twenty times its size; it also runs FP32 through
/// the tensor unit's reduced-precision path (max error ~7e-4 of the output
/// range against a CPU FP32 product on random operands). Here K is split
/// into 128-wide chunks, one threadgroup per (32 columns, chunk), each thread
/// accumulating four rows of one column in FP32, and a second kernel adds the
/// chunks in chunk order. The result is the FP32 product (max error ~4e-4
/// absolute on outputs of magnitude ~260, i.e. rounding).
/// `DARKBLOOM_QWEN35_SPLITK_BA=0` keeps MLX's GEMM.
enum Qwen35SmallNMatmul {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN35_SPLITK_BA"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static let chunk = 128

    /// The reduce takes the qkv|z product as an unread input (`after`), so MLX
    /// encodes it after that product and the partial runs beside the product
    /// instead of alone. `DARKBLOOM_QWEN35_SPLITK_BA_OVERLAP=0` drops it.
    static let overlap: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN35_SPLITK_BA_OVERLAP"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    // grid (N / 32 * 128, K / 128, 1), threadgroup (128, 1, 1). The chunk's
    // x rows [M x 128] are staged in threadgroup memory; thread t takes column
    // nb + (t & 31) and k sub-range 32 * (t >> 5) .. + 31 for every row (eight
    // independent float4 weight loads), and the four sub-range sums of each
    // (row, column) are added in sub-range order.
    private static let partialSource = """
        const int K = dims[0]; const int M = dims[1]; const int N = dims[2];
        const int nb = int(threadgroup_position_in_grid.x) * 32;
        const int kc = int(threadgroup_position_in_grid.y);
        const int k0 = kc * 128;
        const uint t = thread_position_in_threadgroup.x;
        threadgroup float4 xs[16 * 32];
        threadgroup float red[4 * 16 * 33];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++) {
          const uint i = t + 128 * j;
          const int m = int(i >> 5); const int q = int(i & 31);
          xs[i] = m < M ? *(const device float4*)(x + (size_t)m * K + k0 + 4 * q) : float4(0.0f);
        }
        const int c = int(t & 31);
        const int s = int(t >> 5);
        const device float4* wp = (const device float4*)(w + (size_t)(nb + c) * K + k0 + 32 * s);
        float4 wv[8];
        #pragma clang loop unroll(full)
        for (int j = 0; j < 8; j++) { wv[j] = wp[j]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int m = 0; m < 16; m++) {
          float acc = 0.0f;
          #pragma clang loop unroll(full)
          for (int j = 0; j < 8; j++) { acc += dot(xs[m * 32 + 8 * s + j], wv[j]); }
          red[(s * 16 + m) * 33 + c] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; j++) {
          const uint o = t + 128 * j;
          const int m = int(o >> 5); const int cc = int(o & 31);
          if (m < M) {
            const float v = ((red[(0 * 16 + m) * 33 + cc] + red[(1 * 16 + m) * 33 + cc])
                + red[(2 * 16 + m) * 33 + cc]) + red[(3 * 16 + m) * 33 + cc];
            part[((size_t)kc * M + m) * N + nb + cc] = v;
          }
        }
        """

    // One thread per output: the KS chunk partials loaded together (unrolled),
    // then added in chunk order.
    private static let reduceSource = """
        const int M = dims[1]; const int N = dims[2];
        const uint i = thread_position_in_grid.x;
        if (i >= uint(M * N)) { return; }
        float v[KS];
        #pragma clang loop unroll(full)
        for (int s = 0; s < KS; s++) { v[s] = part[(size_t)s * M * N + i]; }
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (int s = 0; s < KS; s++) { sum += v[s]; }
        out[i] = sum;
        """

    private static let partialKernel = MLXFast.metalKernel(
        name: "qwen35_splitk_partial", inputNames: ["x", "w", "dims"], outputNames: ["part"],
        source: partialSource, ensureRowContiguous: true)
    private static let reduceKernel = MLXFast.metalKernel(
        name: "qwen35_splitk_reduce", inputNames: ["part", "dims", "dep"], outputNames: ["out"],
        source: reduceSource, ensureRowContiguous: false)

    static func apply(_ x: MLXArray, _ w: MLXArray, after: MLXArray? = nil) -> MLXArray? {
        guard enabled, x.dtype == .float32, w.dtype == .float32, w.ndim == 2 else { return nil }
        let k = x.dim(-1)
        let n = w.dim(0)
        let rows = x.size / k
        guard rows >= 1, w.dim(1) == k, n % 32 == 0, k % chunk == 0 else { return nil }
        // Prompt width: the 64-row simdgroup-matrix split-K (`Qwen35WideNMatmul`).
        if rows > 16 { return Qwen35WideNMatmul.apply(x, w, rows: rows, k: k, n: n) }
        guard let p = partials(x, w) else { return nil }
        return reduce(p, after: after)
    }

    /// The verify-width partial kernel's output before the reduce: chunk
    /// partials `[K / 128, rows, N]` (FP32, chunk-major), with the shape the
    /// reduced product takes. Nil where `apply` would not run this kernel
    /// (switch off, not FP32, rows > 16, or an unaligned shape).
    struct Partials {
        let part: MLXArray
        let rows: Int
        let n: Int
        let chunks: Int
        let leading: [Int]
        let dims: MLXArray
    }

    static func partials(_ x: MLXArray, _ w: MLXArray) -> Partials? {
        guard enabled, x.dtype == .float32, w.dtype == .float32, w.ndim == 2 else { return nil }
        let k = x.dim(-1)
        let n = w.dim(0)
        let rows = x.size / k
        guard rows >= 1, rows <= 16, w.dim(1) == k, n % 32 == 0, k % chunk == 0 else { return nil }
        let dims = MLXArray([Int32(k), Int32(rows), Int32(n)])
        let part = partialKernel(
            [x.reshaped(rows, k), w, dims],
            grid: (n / 32 * 128, k / chunk, 1), threadGroup: (128, 1, 1),
            outputShapes: [[k / chunk, rows, n]], outputDTypes: [.float32])[0]
        return Partials(
            part: part, rows: rows, n: n, chunks: k / chunk,
            leading: Array(x.shape.dropLast()), dims: dims)
    }

    /// The reduce launch: the chunk partials added in chunk order from 0.0f.
    /// `after` is the record's unread `dep` input (`overlap`): the reduce is
    /// encoded after the qkv|z product, so the partial runs beside it.
    static func reduce(_ p: Partials, after: MLXArray? = nil) -> MLXArray {
        let y = reduceKernel(
            [p.part, p.dims, (overlap ? after : nil) ?? p.dims], template: [("KS", p.chunks)],
            grid: ((p.rows * p.n + 31) / 32 * 32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[p.rows, p.n]], outputDTypes: [.float32])[0]
        return y.reshaped(p.leading + [p.n])
    }
}

/// Where a verify forward hands the b|a stack's chunk partials to the GDN
/// prework instead of the reduce launch (`Qwen35SplitKFold`): the stack
/// fills it when its split-K partial kernel ran; empty means the reduced
/// product is the only form.
final class Qwen35BAPartialsCapture {
    var partials: Qwen35SmallNMatmul.Partials?
    /// The column where `in_proj_a` starts in the stacked product (`b | a`).
    var boundary = 0
}

// MARK: - The b|a reduce folded into the GDN prework

/// The verify-width `b | a` split-K (`Qwen35SmallNMatmul`) is two launches
/// per GDN layer: the partial kernel (K in 128-wide chunks) and a reduce that
/// adds the 40 chunk partials of each of the 16 x 96 outputs in chunk order.
/// The only consumer of that product in a verify round is the GDN prework
/// launch (`Qwen35GDNPrework`), which reads one `a` and one `b` value per
/// (row, value head) to form the gates, and the replay tape, which keeps the
/// `a` and `b` rows. Here the prework's gate threads read the chunk partials
/// themselves and add them in the reduce kernel's order (from 0.0f, chunk 0
/// first), then write the sums out as the `a` and `b` arrays the tape keeps.
/// The reduce launch disappears from the round (48 launches, each a small
/// grid with its own ramp and drain, behind a dependency barrier), the
/// partial kernel is unchanged, and every value is the same bit: the same
/// FP32 additions in the same order, checked at load (`prepare`) on the
/// layer's geometry for every kernel variant the verify can take, with the
/// reduce launch kept on any mismatch. On terrapinelf's tip the verify
/// window's launch is the reads-first kernel
/// (`Qwen35GDNPrework.verifyLoadsFirstSource`, `Qwen35A3BTargetVerify.swift`)
/// whenever its own self-test passed: the fold derives a `_bafold` variant
/// from that source as well and mirrors `run`'s pick, so the folded launch is
/// always the kernel the verify path would have taken behind the reduce.
/// `DARKBLOOM_QWEN35_SPLITK_BA_FOLD=0` keeps the reduce launch.
enum Qwen35SplitKFold {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN35_SPLITK_BA_FOLD"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Chunk partial counts the fold compiles for (`KSP`); K = 5120 gives 40.
    static let maximumChunks = 64

    /// One compiled kernel per (qkv dtype, strided reads, conv-input output,
    /// reads-first source).
    struct Variant: Hashable {
        let dtype: String
        let strided: Bool
        let convInput: Bool
        let loadsFirst: Bool
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var verdicts: [Variant: Bool] = [:]
    nonisolated(unsafe) private static var prepared: Set<String> = []
    nonisolated(unsafe) static var probeFailed = false

    /// Whether the fold passed its bitwise self-test for this variant.
    static func verified(dtype: DType, strided: Bool, convInput: Bool, loadsFirst: Bool) -> Bool {
        let variant = Variant(
            dtype: "\(dtype)", strided: strided, convInput: convInput, loadsFirst: loadsFirst)
        return lock.withLock { verdicts[variant] ?? false }
    }

    /// Whether `Qwen35GDNPrework.run` takes the reads-first kernel for this
    /// variant (strided reads with the conv-input output, the switch on and
    /// `prepareVerify`'s verdict recorded for the geometry and dtype). The
    /// fold mirrors the pick: its folded launch is derived from the kernel
    /// the verify path takes.
    static func takesLoadsFirst(
        dtype: DType, strided: Bool, convInput: Bool, hk: Int, hv: Int, cd: Int, ks: Int
    ) -> Bool {
        strided && convInput
            && Qwen35GDNPrework.verifyLoadsFirstVerified(
                keyHeads: hk, valueHeads: hv, cd: cd, ks: ks, dtype: dtype)
    }

    /// The fold applies to a verify forward of `rows` rows when its switches
    /// are on and at least one variant passed.
    static func active(rows: Int) -> Bool {
        guard enabled, Qwen35SmallNMatmul.enabled, Qwen35GDNPrework.enabled, rows >= 1,
            rows <= 16
        else { return false }
        return lock.withLock { verdicts.values.contains(true) }
    }

    // The gate reads of the prework source, and what replaces them: the a and
    // b values become the ordered sums of the chunk partials `abp`
    // [KSP, rows, NAB] (read through its strides, so a strided partial works
    // too), written out as `ao` / `bo` for the replay tape. One ordered loop
    // (from 0.0f, chunk 0 first: the reduce kernel's sequence; MLX compiles
    // custom kernels with Metal's safe math mode, so the order stands) with
    // no per-chunk register arrays, so the gate threads do not raise the
    // launch's register footprint.
    private static let foldBlock = """
        float asum = 0.0f;
                  float bsum = 0.0f;
                  {
                    const int64_t prow = (int64_t(bb) * int64_t(Sn) + int64_t(t)) * abp_strides[1];
                    const int64_t pa = prow + int64_t(AOFF + int(hv)) * abp_strides[2];
                    const int64_t pb = prow + int64_t(BOFF + int(hv)) * abp_strides[2];
                    #pragma clang loop unroll_count(8)
                    for (int s = 0; s < KSP; s++) {
                      const int64_t pi = int64_t(s) * abp_strides[0];
                      asum += abp[pa + pi];
                      bsum += abp[pb + pi];
                    }
                  }
                  ao[grow] = asum;
                  bo[grow] = bsum;

        """

    /// Replace `target` in `text` where it occurs exactly once; false (text
    /// untouched) otherwise. No trap: a stock source that drifted away from
    /// these anchors leaves the fold's sources nil and the reduce launch in
    /// place (`prepare` reports it).
    private static func replacing(_ text: inout String, _ target: String, _ replacement: String)
        -> Bool
    {
        guard text.components(separatedBy: target).count == 2 else { return false }
        text = text.replacingOccurrences(of: target, with: replacement)
        return true
    }

    /// `Qwen35GDNPrework.source` with the gates formed from the partials; nil
    /// when the stock source no longer carries the two gate reads. (The
    /// post-check names the reads as statements: the `beta[grow]` store also
    /// contains the letters `a[grow]`.)
    static let foldedSource: String? = {
        var text = Qwen35GDNPrework.source
        guard
            replacing(
                &text, "const float av = a[grow] + dtb[hv];",
                foldBlock + "          const float av = asum + dtb[hv];"),
            replacing(&text, "const float bv = b[grow];", "const float bv = bsum;"),
            !text.contains("= a[grow]"), !text.contains("= b[grow]")
        else { return nil }
        return text
    }()

    /// `Qwen35GDNPrework.stridedSource` with the gates formed from the
    /// partials; the a/b row bases of the strided header go with their inputs.
    /// Nil when the strided source no longer carries them.
    static let foldedStridedSource: String? = {
        var text = Qwen35GDNPrework.stridedSource
        guard
            replacing(
                &text,
                "\n        const int64_t ab = int64_t(bb) * a_strides[0] + int64_t(t) * a_strides[1];",
                ""),
            replacing(
                &text,
                "\n        const int64_t bbase = int64_t(bb) * b_strides[0] + int64_t(t) * b_strides[1];",
                ""),
            replacing(
                &text,
                "const float av = a[ab + int64_t(hv) * a_strides[2]] + dtb[int64_t(hv) * dtb_strides[0]];",
                foldBlock + "          const float av = asum + dtb[int64_t(hv) * dtb_strides[0]];"),
            replacing(
                &text, "const float bv = b[bbase + int64_t(hv) * b_strides[2]];",
                "const float bv = bsum;"),
            !text.contains("a[ab"), !text.contains("b[bbase"), !text.contains("* a_strides"),
            !text.contains("* b_strides")
        else { return nil }
        return text
    }()

    // The reads-first kernel's gate reads and what replaces them: `av` and
    // `bv` are declared before the gate block and every store comes last, so
    // the sums are declared with them, formed in the gate block (the same
    // ordered loop as `foldBlock`) and stored beside `g` / `beta`: the launch
    // still reads first and stores last.
    private static let loadsFirstFoldBlock = """
        {
            const int64_t prow = (int64_t(bb) * int64_t(Sn) + int64_t(t)) * abp_strides[1];
            const int64_t pa = prow + int64_t(AOFF + int(hv)) * abp_strides[2];
            const int64_t pb = prow + int64_t(BOFF + int(hv)) * abp_strides[2];
            #pragma clang loop unroll_count(8)
            for (int s = 0; s < KSP; s++) {
              const int64_t pi = int64_t(s) * abp_strides[0];
              asum += abp[pa + pi];
              bsum += abp[pb + pi];
            }
          }
          av = asum + dtb[int64_t(hv) * dtb_strides[0]];
        """

    /// `Qwen35GDNPrework.verifyLoadsFirstSource` (terrapinelf's reads-first
    /// verify launch) with the gates formed from the partials and `ao` / `bo`
    /// stored with the gates; nil when that source no longer carries its
    /// anchors (the variant then stays on the record's path).
    static let foldedLoadsFirstSource: String? = {
        var text = Qwen35GDNPrework.verifyLoadsFirstSource
        guard
            replacing(
                &text, "float dcy = 0.0f;",
                "float dcy = 0.0f;\n        float asum = 0.0f;\n        float bsum = 0.0f;"),
            replacing(
                &text, "const int64_t ab = int64_t(bb) * a_strides[0] + int64_t(t) * a_strides[1];",
                ""),
            replacing(
                &text,
                "const int64_t bbase = int64_t(bb) * b_strides[0] + int64_t(t) * b_strides[1];",
                ""),
            replacing(
                &text, "av = a[ab + int64_t(hv) * a_strides[2]] + dtb[int64_t(hv) * dtb_strides[0]];",
                loadsFirstFoldBlock),
            replacing(&text, "bv = b[bbase + int64_t(hv) * b_strides[2]];", "bv = bsum;"),
            replacing(
                &text, "beta[grow] = betav;",
                "beta[grow] = betav;\n          ao[grow] = asum;\n          bo[grow] = bsum;"),
            !text.contains("a[ab"), !text.contains("b[bbase"), !text.contains("* a_strides"),
            !text.contains("* b_strides"), text.contains("ao[grow] = asum;"),
            text.contains("bo[grow] = bsum;")
        else { return nil }
        return text
    }()

    /// `Qwen35GDNPrework.withConvInput` without its trap: nil when the
    /// conv-tail anchor is not exactly once in `text`.
    private static func convInputSource(_ text: String?) -> String? {
        guard let text,
            text.components(separatedBy: Qwen35GDNPrework.convInputAnchor).count == 2
        else { return nil }
        return Qwen35GDNPrework.withConvInput(text)
    }

    /// Whether both derived sources were built (the stock sources still
    /// carry the anchors); false leaves every variant on the reduce launch.
    static var sourcesAvailable: Bool {
        foldedSource != nil && foldedStridedSource != nil
            && convInputSource(foldedSource) != nil && convInputSource(foldedStridedSource) != nil
    }

    private static let inputNames = ["qkv", "cs", "w", "abp", "decay", "dtb", "wq", "wk", "S"]

    private static let plainKernel: MLXFast.MLXFastKernel? = foldedSource.map {
        MLXFast.metalKernel(
            name: "qwen35_gdn_prework_bafold", inputNames: inputNames,
            outputNames: ["q", "k", "v", "g", "beta", "tail", "ao", "bo"],
            source: $0, ensureRowContiguous: true)
    }
    private static let plainConvInputKernel: MLXFast.MLXFastKernel? = convInputSource(foldedSource)
        .map {
            MLXFast.metalKernel(
                name: "qwen35_gdn_prework_ci_bafold", inputNames: inputNames,
                outputNames: ["q", "k", "v", "g", "beta", "tail", "ci", "ao", "bo"],
                source: $0, ensureRowContiguous: true)
        }
    private static let stridedKernel: MLXFast.MLXFastKernel? = foldedStridedSource.map {
        MLXFast.metalKernel(
            name: "qwen35_gdn_prework_strided_bafold", inputNames: inputNames,
            outputNames: ["q", "k", "v", "g", "beta", "tail", "ao", "bo"],
            source: $0, ensureRowContiguous: false)
    }
    private static let stridedConvInputKernel: MLXFast.MLXFastKernel? = convInputSource(
        foldedStridedSource
    ).map {
        MLXFast.metalKernel(
            name: "qwen35_gdn_prework_ci_strided_bafold", inputNames: inputNames,
            outputNames: ["q", "k", "v", "g", "beta", "tail", "ci", "ao", "bo"],
            source: $0, ensureRowContiguous: false)
    }

    private static let loadsFirstKernel: MLXFast.MLXFastKernel? = foldedLoadsFirstSource.map {
        MLXFast.metalKernel(
            name: "qwen35_gdn_prework_verify_lf_bafold", inputNames: inputNames,
            outputNames: ["q", "k", "v", "g", "beta", "ci", "ao", "bo"],
            source: $0, ensureRowContiguous: false)
    }

    /// The variant's kernel; nil when its source could not be derived (the
    /// reads-first variant exists only for strided reads with the conv input).
    static func kernel(strided: Bool, convInput: Bool, loadsFirst: Bool) -> MLXFast.MLXFastKernel? {
        if loadsFirst { return strided && convInput ? loadsFirstKernel : nil }
        if strided {
            return convInput ? stridedConvInputKernel : stridedKernel
        }
        return convInput ? plainConvInputKernel : plainKernel
    }

    /// Compile the fold's kernels and check them bit for bit against the
    /// prework launch fed by the reduce kernel, once per GDN geometry, at
    /// model construction (before any timed forward): for the qkv dtypes the
    /// verify window forms (FP16 and FP32), with and without strided reads,
    /// with and without the conv-input output, on 16-row and 3-row windows
    /// of the layer's own shapes (`hidden` -> 2 x `hv`, K in 128-wide
    /// chunks). A variant that fails, or whose kernel does not compile on
    /// this device, keeps the reduce launch; one stderr line reports the pick.
    static func prepare(hk: Int, dk: Int, hv: Int, dv: Int, ks: Int, hidden: Int) {
        guard enabled, Qwen35SmallNMatmul.enabled, Qwen35GDNPrework.enabled, dk == 128,
            dv == 128, hk > 0, hv % hk == 0, ks == 4, hidden % Qwen35SmallNMatmul.chunk == 0,
            hidden / Qwen35SmallNMatmul.chunk <= maximumChunks, (2 * hv) % 32 == 0
        else { return }
        let geometry = "\(hk)/\(dk)/\(hv)/\(dv)/\(ks)/\(hidden)"
        let first = lock.withLock { () -> Bool in
            guard !prepared.contains(geometry) else { return false }
            prepared.insert(geometry)
            return true
        }
        guard first else { return }
        guard sourcesAvailable || foldedLoadsFirstSource != nil else {
            // Every prework source drifted from the fold's anchors: no kernel
            // to test, every variant stays on the reduce launch.
            FileHandle.standardError.write(
                Data(
                    "qwen35 split-K b|a fold: no variant could be derived (the prework sources no longer match the fold's anchors); reduce launch kept\n"
                        .utf8))
            return
        }
        let started = Date()
        let cd = 2 * hk * dk + hv * dv
        var passed: [String] = []
        var failed: [String] = []
        var underived: [String] = []
        var compared = 0
        var mismatches = 0
        // The variants the verify forward takes (`cbv2ForwardCaptured`).
        var verifyPath: [String] = []
        for dtype in [DType.float16, .float32] {
            for strided in [true, false] {
                for convInput in [true, false] {
                    let loadsFirst = takesLoadsFirst(
                        dtype: dtype, strided: strided, convInput: convInput, hk: hk, hv: hv,
                        cd: cd, ks: ks)
                    let variant = Variant(
                        dtype: "\(dtype)", strided: strided, convInput: convInput,
                        loadsFirst: loadsFirst)
                    let label =
                        "\(dtype)" + (strided ? "/strided" : "/contiguous") + (convInput ? "/ci" : "")
                        + (loadsFirst ? "/lf" : "")
                    guard kernel(strided: strided, convInput: convInput, loadsFirst: loadsFirst) != nil
                    else {
                        // Its source could not be derived: the record's path.
                        lock.withLock { verdicts[variant] = false }
                        underived.append(label)
                        continue
                    }
                    let (ok, values, differ) = selfCheck(
                        hk: hk, dk: dk, hv: hv, dv: dv, ks: ks, hidden: hidden, dtype: dtype,
                        strided: strided, convInput: convInput)
                    lock.withLock { verdicts[variant] = ok }
                    compared += values
                    mismatches += differ
                    if ok {
                        passed.append(label)
                        if strided == Qwen35GDNPrework.verifyStridedReads,
                            convInput == Qwen35GatedDeltaNet.preworkWritesConvInput
                        {
                            verifyPath.append(label)
                        }
                    } else {
                        failed.append(label + (differ > 0 ? " (\(differ) mismatches)" : ""))
                    }
                }
            }
        }
        Memory.clearCache()
        let ms = Int((Date().timeIntervalSince(started) * 1000).rounded())
        var summary: String
        if failed.isEmpty, underived.isEmpty {
            summary =
                "self-test passed (\(passed.count) variants, \(compared) values compared bitwise, \(mismatches) mismatches)"
        } else {
            summary = "self-test"
            if !failed.isEmpty { summary += " FAILED for \(failed.joined(separator: ", "))" }
            if !underived.isEmpty {
                summary +=
                    (failed.isEmpty ? "" : ";")
                    + " not derived (source anchors drifted; the record's path) for \(underived.joined(separator: ", "))"
            }
            if !passed.isEmpty {
                summary +=
                    "; passed \(passed.joined(separator: ", ")) (\(compared) values compared bitwise, \(mismatches) mismatches)"
            }
        }
        let outcome =
            verifyPath.isEmpty
            ? (passed.isEmpty
                ? "; reduce launch kept"
                : "; reduce launch kept on the verify path")
            : "; the verify prework (\(verifyPath.joined(separator: ", "))) sums the b|a chunk partials (\(hidden / Qwen35SmallNMatmul.chunk) per output)"
                + (verifyPath.contains { $0.hasSuffix("/lf") } ? ", reading first" : "")
        FileHandle.standardError.write(
            Data(("qwen35 split-K b|a fold: " + summary + outcome + "; \(ms) ms\n").utf8))
    }

    /// One variant's check: whether every bit agreed, the values compared and
    /// the values that differed (0 on a pass; 0 with `false` when the check
    /// could not run at all).
    private static func selfCheck(
        hk: Int, dk: Int, hv: Int, dv: Int, ks: Int, hidden: Int, dtype: DType,
        strided: Bool, convInput: Bool
    ) -> (Bool, Int, Int) {
        let cd = 2 * hk * dk + hv * dv
        let n = 2 * hv
        let keys = MLXRandom.split(key: MLXRandom.key(0x6261_666f), into: 10)
        var same = true
        var compared = 0
        var mismatches = 0
        probeFailed = false
        withErrorHandler({ _ in Qwen35SplitKFold.probeFailed = true }) {
            for S in [16, 3] {
                // The normed FP32 input with a wide magnitude spread, and a
                // weight of the layer's scale: the chunk partials then differ
                // in order of magnitude and the sums see real rounding.
                let x = MLXRandom.normal([1, S, hidden], key: keys[0])
                    * exp(MLXRandom.normal([1, S, hidden], key: keys[1]))
                let w = MLXRandom.normal([n, hidden], key: keys[2]) * Float(0.02)
                    * exp(MLXRandom.normal([n, hidden], key: keys[3]) * Float(0.5))
                guard let p = Qwen35SmallNMatmul.partials(x, w) else {
                    same = false
                    return
                }
                let y = Qwen35SmallNMatmul.reduce(p)
                let b = y[.ellipsis, ..<hv]
                let a = y[.ellipsis, hv...]
                // qkv as a column slice of a wider stack, as the model's qkv|z
                // product at verify width.
                let width = cd + hv * dv
                let stack = (MLXRandom.normal([1, S, width], key: keys[4])
                    * exp(MLXRandom.normal([1, S, width], key: keys[5]))).asType(dtype)
                let qkv = stack[.ellipsis, ..<cd]
                let convState = MLXRandom.normal([1, ks - 1, cd], key: keys[6])
                let convWeight = MLXRandom.normal([cd, ks, 1], key: keys[7]) * Float(0.5)
                let aDecay = Qwen35GDNDerived().decay(MLXRandom.normal([hv], key: keys[8]) * Float(0.5))
                let dtBias = MLXRandom.normal([hv], key: keys[9])
                let normScales = (
                    q: MLXRandom.normal([dk], key: keys[6]) * Float(0.3) + Float(1),
                    k: MLXRandom.normal([dk], key: keys[7]) * Float(0.3) + Float(1)
                )
                guard
                    let stock = Qwen35GDNPrework.run(
                        qkv: qkv, convState: convState, convWeight: convWeight, a: a, b: b,
                        aDecay: aDecay, dtBias: dtBias, normScales: normScales,
                        keyHeads: hk, valueHeads: hv, headKDim: dk, headVDim: dv,
                        writeConvInput: convInput, stridedReads: strided),
                    let folded = Qwen35GDNPrework.runFoldedUnchecked(
                        qkv: qkv, convState: convState, convWeight: convWeight,
                        abPartials: p.part, aOffset: hv, bOffset: 0,
                        aDecay: aDecay, dtBias: dtBias, normScales: normScales,
                        keyHeads: hk, valueHeads: hv, headKDim: dk, headVDim: dv,
                        writeConvInput: convInput, stridedReads: strided),
                    let fa = folded.a, let fb = folded.b
                else {
                    same = false
                    return
                }
                var pairs: [(MLXArray, MLXArray)] = [
                    (stock.q, folded.q), (stock.k, folded.k), (stock.v, folded.v),
                    (stock.g, folded.g), (stock.beta, folded.beta), (stock.tail, folded.tail),
                    (a, fa), (b, fb),
                ]
                if convInput {
                    guard let sci = stock.convInput, let fci = folded.convInput else {
                        same = false
                        return
                    }
                    pairs.append((sci, fci))
                }
                var differ = MLXArray(Int32(0))
                for (lhs, rhs) in pairs {
                    guard lhs.shape == rhs.shape, lhs.dtype == rhs.dtype, lhs.dtype == .float32
                    else {
                        same = false
                        return
                    }
                    differ =
                        differ
                        + (lhs.view(dtype: .uint32) .!= rhs.view(dtype: .uint32)).asType(.int32).sum()
                    compared += lhs.size
                }
                eval(differ)
                if probeFailed {
                    same = false
                    return
                }
                let count = Int(differ.item(Int32.self))
                mismatches += count
                if count != 0 {
                    same = false
                    return
                }
            }
        }
        return (same && !probeFailed, compared, mismatches)
    }
}

extension Qwen35GDNPrework {
    /// `run` with the b|a product's chunk partials `abPartials` `[KSP, B * S,
    /// NAB]` in place of `a` and `b` (`a` at columns `aOffset ..< aOffset +
    /// HV`, `b` at `bOffset ...`), for a variant whose fold passed its
    /// self-test; nil exactly where the caller should run `run` on the
    /// reduced product instead. The outputs carry the summed `a` and `b`.
    static func runFolded(
        qkv: MLXArray, convState: MLXArray, convWeight: MLXArray,
        abPartials: MLXArray, aOffset: Int, bOffset: Int,
        aDecay: MLXArray, dtBias: MLXArray, normScales: (q: MLXArray, k: MLXArray),
        keyHeads: Int, valueHeads: Int, headKDim: Int, headVDim: Int,
        writeConvInput: Bool, stridedReads: Bool
    ) -> Outputs? {
        guard Qwen35SplitKFold.enabled, qkv.ndim == 3, convWeight.ndim == 3 else { return nil }
        let loadsFirst = Qwen35SplitKFold.takesLoadsFirst(
            dtype: qkv.dtype, strided: stridedReads, convInput: writeConvInput, hk: keyHeads,
            hv: valueHeads, cd: qkv.dim(2), ks: convWeight.dim(1))
        guard
            Qwen35SplitKFold.verified(
                dtype: qkv.dtype, strided: stridedReads, convInput: writeConvInput,
                loadsFirst: loadsFirst)
        else { return nil }
        return runFoldedUnchecked(
            qkv: qkv, convState: convState, convWeight: convWeight,
            abPartials: abPartials, aOffset: aOffset, bOffset: bOffset,
            aDecay: aDecay, dtBias: dtBias, normScales: normScales,
            keyHeads: keyHeads, valueHeads: valueHeads, headKDim: headKDim, headVDim: headVDim,
            writeConvInput: writeConvInput, stridedReads: stridedReads)
    }

    /// `runFolded` without the verdict (the self-test's entry).
    static func runFoldedUnchecked(
        qkv: MLXArray, convState: MLXArray, convWeight: MLXArray,
        abPartials: MLXArray, aOffset: Int, bOffset: Int,
        aDecay: MLXArray, dtBias: MLXArray, normScales: (q: MLXArray, k: MLXArray),
        keyHeads: Int, valueHeads: Int, headKDim: Int, headVDim: Int,
        writeConvInput: Bool, stridedReads: Bool
    ) -> Outputs? {
        guard enabled, qkv.ndim == 3, convState.ndim == 3, convWeight.ndim == 3,
            abPartials.ndim == 3
        else { return nil }
        let B = qkv.dim(0)
        let S = qkv.dim(1)
        let CD = qkv.dim(2)
        let KS = convWeight.dim(1)
        let ksp = abPartials.dim(0)
        let nab = abPartials.dim(2)
        guard headKDim == 128, headVDim == 128, valueHeads % keyHeads == 0,
            CD == 2 * keyHeads * headKDim + valueHeads * headVDim,
            convState.shape == [B, KS - 1, CD], convWeight.shape == [CD, KS, 1],
            [DType.float32, .float16, .bfloat16].contains(qkv.dtype),
            convState.dtype == .float32, convWeight.dtype == .float32,
            abPartials.dtype == .float32, abPartials.dim(1) == B * S,
            ksp >= 1, ksp <= Qwen35SplitKFold.maximumChunks,
            aOffset >= 0, bOffset >= 0, aOffset + valueHeads <= nab, bOffset + valueHeads <= nab,
            aDecay.shape == [valueHeads], aDecay.dtype == .float32,
            dtBias.shape == [valueHeads],
            normScales.q.dtype == .float32, normScales.k.dtype == .float32,
            normScales.q.shape == [headKDim], normScales.k.shape == [headKDim],
            S > 0, S < 65536
        else { return nil }
        let dtb = dtBias.dtype == .float32 ? dtBias : dtBias.asType(.float32)
        // The kernel `run` would take: the reads-first verify launch where
        // its verdict stands, else the stock (strided / conv-input) one.
        let loadsFirst = Qwen35SplitKFold.takesLoadsFirst(
            dtype: qkv.dtype, strided: stridedReads, convInput: writeConvInput, hk: keyHeads,
            hv: valueHeads, cd: CD, ks: KS)
        guard
            let launch = Qwen35SplitKFold.kernel(
                strided: stridedReads, convInput: writeConvInput, loadsFirst: loadsFirst)
        else { return nil }
        let inputs = [
            qkv, convState, convWeight, abPartials, aDecay, dtb, normScales.q, normScales.k,
            MLXArray(Int32(S)),
        ]
        let template: [(String, any KernelTemplateArg)] = [
            ("InT", qkv.dtype), ("HK", keyHeads), ("HV", valueHeads), ("DK", headKDim),
            ("DV", headVDim), ("CD", CD), ("KS", KS),
            ("KSP", ksp), ("AOFF", aOffset), ("BOFF", bOffset),
        ]
        if loadsFirst {
            // q, k, v, g, beta, ci, ao, bo: the tail is the conv input's last
            // NK rows, as the reads-first launch returns it.
            let outputs = launch(
                inputs, template: template,
                grid: (128 * keyHeads, S, B), threadGroup: (128, 1, 1),
                outputShapes: [
                    [B, S, keyHeads, headKDim], [B, S, keyHeads, headKDim],
                    [B, S, valueHeads, headVDim], [B, S, valueHeads], [B, S, valueHeads],
                    [B, KS - 1 + S, CD], [B, S, valueHeads], [B, S, valueHeads],
                ],
                outputDTypes: Array(repeating: DType.float32, count: 8))
            return Outputs(
                q: outputs[0], k: outputs[1], v: outputs[2], g: outputs[3], beta: outputs[4],
                tail: outputs[5][0..., S..., 0...], convInput: outputs[5],
                a: outputs[6], b: outputs[7])
        }
        var outputShapes: [[Int]] = [
            [B, S, keyHeads, headKDim], [B, S, keyHeads, headKDim],
            [B, S, valueHeads, headVDim], [B, S, valueHeads], [B, S, valueHeads],
            [B, KS - 1, CD],
        ]
        if writeConvInput { outputShapes.append([B, KS - 1 + S, CD]) }
        outputShapes.append([B, S, valueHeads])
        outputShapes.append([B, S, valueHeads])
        let outputDTypes: [DType] = Array(repeating: DType.float32, count: outputShapes.count)
        let outputs = launch(
            inputs, template: template,
            grid: (128 * keyHeads, S, B), threadGroup: (128, 1, 1),
            outputShapes: outputShapes,
            outputDTypes: outputDTypes)
        let last = outputs.count - 1
        return Outputs(
            q: outputs[0], k: outputs[1], v: outputs[2], g: outputs[3], beta: outputs[4],
            tail: outputs[5], convInput: writeConvInput ? outputs[6] : nil,
            a: outputs[last - 1], b: outputs[last])
    }
}
