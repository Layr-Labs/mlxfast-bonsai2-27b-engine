// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

// MARK: - Hierarchical top-2

// Metal entry-point names remain stable so extraction preserves pipeline identity.
/// Shared exact ordering for the two-stage candidate-only top-2 reduction:
/// value descending, token id ascending on exact ties, and NaNs last.
private let cbv2TopTwoHeader = """
    struct darkbloom_qwen35_mtp_top2_state {
        float first_value;
        float second_value;
        uint first_id;
        uint second_id;
        uint count;
    };

    inline darkbloom_qwen35_mtp_top2_state darkbloom_qwen35_mtp_top2_empty() {
        darkbloom_qwen35_mtp_top2_state state;
        state.first_value = 0.0f;
        state.second_value = 0.0f;
        state.first_id = 0;
        state.second_id = 0;
        state.count = 0;
        return state;
    }

    inline bool darkbloom_qwen35_mtp_top2_better(
        float candidate_value,
        uint candidate_id,
        float current_value,
        uint current_id
    ) {
        bool candidate_nan = isnan(candidate_value);
        bool current_nan = isnan(current_value);
        if (candidate_nan != current_nan) {
            return !candidate_nan;
        }
        if (candidate_value > current_value) {
            return true;
        }
        if (candidate_value < current_value) {
            return false;
        }
        return candidate_id < current_id;
    }

    inline void darkbloom_qwen35_mtp_top2_insert(
        thread darkbloom_qwen35_mtp_top2_state &state,
        float value,
        uint id
    ) {
        if (state.count > 0 && state.first_id == id) {
            return;
        }
        if (state.count > 1 && state.second_id == id) {
            return;
        }
        if (state.count == 0
            || darkbloom_qwen35_mtp_top2_better(
                value, id, state.first_value, state.first_id)) {
            if (state.count > 0) {
                state.second_value = state.first_value;
                state.second_id = state.first_id;
            }
            state.first_value = value;
            state.first_id = id;
            state.count = min(state.count + 1, 2u);
            return;
        }
        if (state.count == 1
            || darkbloom_qwen35_mtp_top2_better(
                value, id, state.second_value, state.second_id)) {
            state.second_value = value;
            state.second_id = id;
            state.count = 2;
        }
    }
"""

/// The tree loop of the branch-free stage one and of stage two with its
/// barrier taken only between strides (`BONSAI_LAST_BARRIER_SKIP=0` keeps the
/// stock texts). At stride 1 lane 0 alone writes `scratch[0]`, and after the
/// loop lane 0 alone reads it: its own store, in program order, so that
/// barrier orders nothing. Every other barrier, load, store and merge is the
/// stock text's; a text without exactly one such loop tail is kept as it is.
/// The stock stage one stays the fast one's bitwise reference.
private func cbv2TopTwoTreeTail(_ kernel: String, _ text: String) -> String {
    let value = ProcessInfo.processInfo.environment["BONSAI_LAST_BARRIER_SKIP"]?
        .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !["0", "false", "no", "off"].contains(value ?? "") else { return text }
    let tail = "threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n\n    if (lane == 0) {"
    guard text.components(separatedBy: tail).count == 2 else {
        FileHandle.standardError.write(
            "cbv2 top-2 \(kernel): tree loop tail not found, stock text kept\n".data(using: .utf8)!)
        return text
    }
    FileHandle.standardError.write(
        "cbv2 top-2 \(kernel): tree loop's last barrier skipped\n".data(using: .utf8)!)
    return text.replacingOccurrences(
        of: tail, with: "if (stride > 1u) { threadgroup_barrier(mem_flags::mem_threadgroup); }"
            + "\n    }\n\n    if (lane == 0) {")
}

/// Stage one: 32 threadgroups per row reduce disjoint vocabulary stripes.
private let cbv2TopTwoPartialKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_partial",
    inputNames: ["logits"],
    outputNames: ["partial_ids", "partial_values"],
    source: """
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint row = group_index / 32;
        uint group = group_index % 32;
        uint vocab = uint(logits_shape[2]);
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();

        for (uint index = group * 256 + lane;
             index < vocab;
             index += 32 * 256) {
            ulong offset = ulong(row) * ulong(logits_strides[1])
                + ulong(index) * ulong(logits_strides[2]);
            darkbloom_qwen35_mtp_top2_insert(local, float(logits[offset]), index);
        }

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[256];
        scratch[lane] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lane];
                darkbloom_qwen35_mtp_top2_state other = scratch[lane + stride];
                if (other.count > 0) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.first_value, other.first_id);
                }
                if (other.count > 1) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.second_value, other.second_id);
                }
                scratch[lane] = merged;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            uint base = (row * 32 + group) * 2;
            uint sentinel_id = vocab + group * 2;
            float sentinel_value = as_type<float>(0x7fc00000u);
            partial_ids[base] = scratch[0].count > 0
                ? int(scratch[0].first_id) : int(sentinel_id);
            partial_ids[base + 1] = scratch[0].count > 1
                ? int(scratch[0].second_id) : int(sentinel_id + 1);
            partial_values[base] = scratch[0].count > 0
                ? scratch[0].first_value : sentinel_value;
            partial_values[base + 1] = scratch[0].count > 1
                ? scratch[0].second_value : sentinel_value;
        }
    """,
    header: cbv2TopTwoHeader,
    ensureRowContiguous: false
)

/// Stage two: one 32-lane threadgroup per row merges the partial pairs.
private let cbv2TopTwoFinalizeKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_finalize",
    inputNames: ["partial_ids", "partial_values"],
    outputNames: ["top_ids", "top_values"],
    source: cbv2TopTwoTreeTail("stage two", """
        uint lane = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint base = (row * 32 + lane) * 2;
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base], uint(partial_ids[base]));
        darkbloom_qwen35_mtp_top2_insert(
            local, partial_values[base + 1], uint(partial_ids[base + 1]));

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[32];
        scratch[lane] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 16; stride > 0; stride >>= 1) {
            if (lane < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lane];
                darkbloom_qwen35_mtp_top2_state other = scratch[lane + stride];
                darkbloom_qwen35_mtp_top2_insert(
                    merged, other.first_value, other.first_id);
                darkbloom_qwen35_mtp_top2_insert(
                    merged, other.second_value, other.second_id);
                scratch[lane] = merged;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            uint output_base = row * 2;
            top_ids[output_base] = int(scratch[0].first_id);
            top_ids[output_base + 1] = int(scratch[0].second_id);
            top_values[output_base] = scratch[0].first_value;
            top_values[output_base + 1] = scratch[0].second_value;
        }
    """),
    header: cbv2TopTwoHeader,
    ensureRowContiguous: false
)

/// Stage one with a branch-free update (`MLXFAST_TOP2_FASTPATH=0` keeps the
/// stock stage one). Same threadgroups, stripes, scan order and reduction as
/// the stock kernel; only a thread's running update changes. Once the thread
/// holds two candidates and the second is a number (not NaN), a new value
/// can enter exactly when it is greater than a held value: the thread scans
/// its ids in increasing order, so an equal value has the larger id and the
/// stock order (value descending, id ascending on exact ties, NaN last)
/// keeps the held one, and a NaN never beats a number. That case becomes two
/// compares and selects; every other case (fewer than two held, a NaN held)
/// runs the stock insert. Every thread's pair, and so every partial and the
/// row's top two, are the stock kernel's bit for bit (self-tested at first
/// use against the stock stage one on random rows, exact ties, signed zeros,
/// NaN and infinities, all-NaN and constant rows, and a strided view; any
/// mismatch keeps the stock kernel).
private let cbv2TopTwoPartialFastKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_partial_fast",
    inputNames: ["logits"],
    outputNames: ["partial_ids", "partial_values"],
    source: cbv2TopTwoTreeTail("branch-free stage one", """
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint row = group_index / 32;
        uint group = group_index % 32;
        uint vocab = uint(logits_shape[2]);
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        bool numbers = false;

        for (uint index = group * 256 + lane;
             index < vocab;
             index += 32 * 256) {
            ulong offset = ulong(row) * ulong(logits_strides[1])
                + ulong(index) * ulong(logits_strides[2]);
            float value = float(logits[offset]);
            if (numbers) {
                bool above_first = value > local.first_value;
                bool above_second = value > local.second_value;
                float second_value = above_first
                    ? local.first_value : (above_second ? value : local.second_value);
                uint second_id = above_first
                    ? local.first_id : (above_second ? index : local.second_id);
                local.first_value = above_first ? value : local.first_value;
                local.first_id = above_first ? index : local.first_id;
                local.second_value = second_value;
                local.second_id = second_id;
            } else {
                darkbloom_qwen35_mtp_top2_insert(local, value, index);
                numbers = local.count == 2 && !isnan(local.second_value);
            }
        }

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[256];
        scratch[lane] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lane];
                darkbloom_qwen35_mtp_top2_state other = scratch[lane + stride];
                if (other.count > 0) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.first_value, other.first_id);
                }
                if (other.count > 1) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.second_value, other.second_id);
                }
                scratch[lane] = merged;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            uint base = (row * 32 + group) * 2;
            uint sentinel_id = vocab + group * 2;
            float sentinel_value = as_type<float>(0x7fc00000u);
            partial_ids[base] = scratch[0].count > 0
                ? int(scratch[0].first_id) : int(sentinel_id);
            partial_ids[base + 1] = scratch[0].count > 1
                ? int(scratch[0].second_id) : int(sentinel_id + 1);
            partial_values[base] = scratch[0].count > 0
                ? scratch[0].first_value : sentinel_value;
            partial_values[base + 1] = scratch[0].count > 1
                ? scratch[0].second_value : sentinel_value;
        }
    """),
    header: cbv2TopTwoHeader,
    ensureRowContiguous: false
)
/// Stage one with four-wide loads (`MLXFAST_TOP2_VEC=0` keeps the scalar
/// stage one). When the last axis is contiguous each thread reads one float4
/// per stripe step instead of four scalar loads, and the in-loop offsets are
/// 32-bit: `index < vocab` makes `index` a `uint` already and the row offset
/// is formed once in 64 bits outside the loop. A thread scans the indices
/// `group * 1024 + lane * 4 + {0..3}`, then strides of 32 * 1024 — every
/// element enters exactly once, still in ascending id order per thread, so
/// each partial, the merge and the row's top two are the scalar kernel's
/// bit for bit (self-tested with it at first use, including a strided view
/// where the scalar loop runs instead). A stripe tail shorter than four
/// elements falls back to scalar reads inside the same loop.
private let cbv2TopTwoPartialVecKernel = MLXFast.metalKernel(
    name: "darkbloom_qwen35_mtp_top2_partial_vec",
    inputNames: ["logits"],
    outputNames: ["partial_ids", "partial_values"],
    source: cbv2TopTwoTreeTail("vectorized stage one", """
        uint lane = thread_position_in_threadgroup.x;
        uint group_index = threadgroup_position_in_grid.x;
        uint row = group_index / 32;
        uint group = group_index % 32;
        uint vocab = uint(logits_shape[2]);
        darkbloom_qwen35_mtp_top2_state local = darkbloom_qwen35_mtp_top2_empty();
        bool numbers = false;

        const ulong rowoff = ulong(row) * ulong(logits_strides[1]);
        if (logits_strides[2] == 1) {
            const device float* rowp = (const device float*)(logits + rowoff);
            uint index = group * 1024 + lane * 4;
            for (; index + 3 < vocab; index += 32 * 1024) {
                float4 v = *(const device float4*)(rowp + index);
                for (uint e = 0; e < 4; e++) {
                    float value = v[e];
                    uint i = index + e;
                    if (numbers) {
                        bool above_first = value > local.first_value;
                        bool above_second = value > local.second_value;
                        float second_value = above_first
                            ? local.first_value
                            : (above_second ? value : local.second_value);
                        uint second_id = above_first
                            ? local.first_id
                            : (above_second ? i : local.second_id);
                        local.first_value = above_first ? value : local.first_value;
                        local.first_id = above_first ? i : local.first_id;
                        local.second_value = second_value;
                        local.second_id = second_id;
                    } else {
                        darkbloom_qwen35_mtp_top2_insert(local, value, i);
                        numbers = local.count == 2 && !isnan(local.second_value);
                    }
                }
            }
            // Tail: the same per-lane stripe, partial float4 runs scalar.
            for (; index < vocab; index += 32 * 1024) {
                const uint last = min(index + 3, vocab - 1);
                for (uint i = index; i <= last; i++) {
                    float value = rowp[i];
                    if (numbers) {
                        bool above_first = value > local.first_value;
                        bool above_second = value > local.second_value;
                        float second_value = above_first
                            ? local.first_value
                            : (above_second ? value : local.second_value);
                        uint second_id = above_first
                            ? local.first_id
                            : (above_second ? i : local.second_id);
                        local.first_value = above_first ? value : local.first_value;
                        local.first_id = above_first ? i : local.first_id;
                        local.second_value = second_value;
                        local.second_id = second_id;
                    } else {
                        darkbloom_qwen35_mtp_top2_insert(local, value, i);
                        numbers = local.count == 2 && !isnan(local.second_value);
                    }
                }
            }
        } else {
            for (uint index = group * 256 + lane;
                 index < vocab;
                 index += 32 * 256) {
                ulong offset = rowoff + ulong(index) * ulong(logits_strides[2]);
                float value = float(logits[offset]);
                if (numbers) {
                    bool above_first = value > local.first_value;
                    bool above_second = value > local.second_value;
                    float second_value = above_first
                        ? local.first_value : (above_second ? value : local.second_value);
                    uint second_id = above_first
                        ? local.first_id : (above_second ? index : local.second_id);
                    local.first_value = above_first ? value : local.first_value;
                    local.first_id = above_first ? index : local.first_id;
                    local.second_value = second_value;
                    local.second_id = second_id;
                } else {
                    darkbloom_qwen35_mtp_top2_insert(local, value, index);
                    numbers = local.count == 2 && !isnan(local.second_value);
                }
            }
        }

        threadgroup darkbloom_qwen35_mtp_top2_state scratch[256];
        scratch[lane] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) {
                darkbloom_qwen35_mtp_top2_state merged = scratch[lane];
                darkbloom_qwen35_mtp_top2_state other = scratch[lane + stride];
                if (other.count > 0) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.first_value, other.first_id);
                }
                if (other.count > 1) {
                    darkbloom_qwen35_mtp_top2_insert(
                        merged, other.second_value, other.second_id);
                }
                scratch[lane] = merged;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0) {
            uint base = (row * 32 + group) * 2;
            uint sentinel_id = vocab + group * 2;
            float sentinel_value = as_type<float>(0x7fc00000u);
            partial_ids[base] = scratch[0].count > 0
                ? int(scratch[0].first_id) : int(sentinel_id);
            partial_ids[base + 1] = scratch[0].count > 1
                ? int(scratch[0].second_id) : int(sentinel_id + 1);
            partial_values[base] = scratch[0].count > 0
                ? scratch[0].first_value : sentinel_value;
            partial_values[base + 1] = scratch[0].count > 1
                ? scratch[0].second_value : sentinel_value;
        }
    """),
    header: cbv2TopTwoHeader,
    ensureRowContiguous: false
)

/// `MLXFAST_TOP2_FASTPATH` / `MLXFAST_TOP2_VEC`, and the first-use verdicts.
private enum CBv2TopTwoFast {
    static let enabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_TOP2_FASTPATH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    static let lock = NSLock()
    nonisolated(unsafe) static var verdict: Bool?
    nonisolated(unsafe) static var vecVerdict: Bool?
    nonisolated(unsafe) static var testing = false

    static let vecEnabled: Bool = {
        let value = ProcessInfo.processInfo.environment["MLXFAST_TOP2_VEC"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return !["0", "false", "no", "off"].contains(value ?? "")
    }()

    /// Stage one: vectorized when its verdict holds, else the branch-free or
    /// stock twin per `fast`.
    static func partials(_ logits: MLXArray, vec: Bool, fast: Bool) -> [MLXArray] {
        let kernel = vec ? cbv2TopTwoPartialVecKernel
            : fast ? cbv2TopTwoPartialFastKernel : cbv2TopTwoPartialKernel
        let rows = logits.dim(1)
        return kernel(
            [logits],
            grid: (rows * 32 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, 32, 2], [rows, 32, 2]],
            outputDTypes: [.int32, .float32]
        )
    }

    /// Stock vs the branch-free and the four-wide stage one on synthetic
    /// rows: every partial id and value, bit for bit. The vectorized kernel's
    /// strided-view path runs its own scalar loop, covered by the strided
    /// case below.
    static func verified() -> Bool {
        if testing { return true }
        if let verdict = lock.withLock({ verdict }) { return verdict }
        testing = true
        defer { testing = false }
        var passed = true
        var values = 0
        var mismatches = 0
        var vecMismatches = 0
        let rows = 16
        let vocab = 248_320
        let keys = MLXRandom.split(key: MLXRandom.key(0x7432_6670), into: 4)
        var cases: [MLXArray] = []
        let base = MLXRandom.normal([1, rows, vocab], key: keys[0]) * 4
        cases.append(base)
        // Exact ties at near and distant columns.
        let columns = MLXArray(0 ..< vocab).reshaped(1, 1, vocab)
        let tieColumns = (columns .== MLXArray(Int32(11))) .|| (columns .== MLXArray(Int32(12)))
            .|| (columns .== MLXArray(Int32(200_001)))
        cases.append(which(tieColumns, MLXArray(Float(99)), base))
        // NaN, infinities, signed zeros and a repeated value mixed in.
        let draw = MLXRandom.uniform(Float(0) ..< Float(1), [1, rows, vocab], key: keys[1])
        var mixed = which(draw .< Float(0.02), MLXArray(Float.nan), base)
        mixed = which((draw .>= Float(0.02)) .&& (draw .< Float(0.03)), MLXArray(Float.infinity), mixed)
        mixed = which((draw .>= Float(0.03)) .&& (draw .< Float(0.04)), MLXArray(-Float.infinity), mixed)
        mixed = which((draw .>= Float(0.04)) .&& (draw .< Float(0.10)), MLXArray(Float(-0.0)), mixed)
        mixed = which((draw .>= Float(0.10)) .&& (draw .< Float(0.16)), MLXArray(Float(0.0)), mixed)
        mixed = which((draw .>= Float(0.16)) .&& (draw .< Float(0.30)), MLXArray(Float(7)), mixed)
        cases.append(mixed)
        cases.append(MLXArray.zeros([1, rows, vocab], dtype: .float32) + Float.nan)
        cases.append(MLXArray.zeros([1, rows, vocab], dtype: .float32) + Float(7))
        // A strided view (every other column of a wider row).
        let wide = MLXRandom.normal([1, rows, 2 * vocab], key: keys[2]) * 4
        cases.append(wide[0..., 0..., .stride(by: 2)])
        for logits in cases {
            eval(logits)
            let stock = partials(logits, vec: false, fast: false)
            let fast = partials(logits, vec: false, fast: true)
            let vec = partials(logits, vec: true, fast: false)
            let differ = (stock[0] .!= fast[0]).asType(.int32).sum()
                + (stock[1].view(dtype: .uint32) .!= fast[1].view(dtype: .uint32))
                .asType(.int32).sum()
            let vecDiffer = (stock[0] .!= vec[0]).asType(.int32).sum()
                + (stock[1].view(dtype: .uint32) .!= vec[1].view(dtype: .uint32))
                .asType(.int32).sum()
            eval(differ, vecDiffer)
            values += stock[0].size + stock[1].size
            let count = Int(differ.item(Int32.self))
            let vecCount = Int(vecDiffer.item(Int32.self))
            mismatches += count
            vecMismatches += vecCount
            if count != 0 { passed = false }
        }
        let vecPassed = vecMismatches == 0
        lock.withLock { verdict = passed; vecVerdict = vecPassed }
        FileHandle.standardError.write(
            ("cbv2 top-2 stage one: self-test " + (passed ? "passed" : "FAILED")
                + (vecPassed ? " (vec passed" : " (vec FAILED")
                + ", \(cases.count) cases of \(rows) x \(vocab), \(values) partial values, "
                + "\(mismatches) + \(vecMismatches) mismatches)"
                + (passed ? "\n" : "; stock stage one kept\n"))
                .data(using: .utf8)!)
        return passed
    }

}

/// Exact top-2 token ids and logit values for every row of `[1, rows, vocab]`.
///
/// Returns lazy device arrays shaped `[rows, 2]`: ids are `int32`, values are
/// `float32`. No evaluation or host read occurs here. Results are ordered by
/// value descending, then token id ascending on exact ties, with NaNs last.
package func cbv2TopTwoRows(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
    precondition(logits.ndim == 3 && logits.dim(0) == 1)
    let rows = logits.dim(1)
    let vocabularySize = logits.dim(2)
    precondition(rows > 0 && vocabularySize >= 2)

    // The branch-free stage one, or its four-wide-load twin: the same
    // partials bit for bit.
    let ok = CBv2TopTwoFast.enabled && CBv2TopTwoFast.verified()
    let vec = CBv2TopTwoFast.vecEnabled
        && CBv2TopTwoFast.lock.withLock { CBv2TopTwoFast.vecVerdict ?? false }
    let partials = CBv2TopTwoFast.partials(logits, vec: vec, fast: ok && !vec)
    let outputs = cbv2TopTwoFinalizeKernel(
        partials,
        grid: (rows * 32, 1, 1),
        threadGroup: (32, 1, 1),
        outputShapes: [[rows, 2], [rows, 2]],
        outputDTypes: [.int32, .float32]
    )
    return (outputs[0], outputs[1])
}
