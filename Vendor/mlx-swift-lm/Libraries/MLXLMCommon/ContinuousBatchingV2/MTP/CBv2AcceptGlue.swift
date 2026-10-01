// The acceptance packet in one launch and the accept walk without its cast.

import Foundation
import MLX

/// ACCGLUE: the acceptance packet in one launch, and the speculation's accept
/// walk without its bool-to-int32 cast.
///
/// Each decode round joins the block's draft ids and the verify's per-position
/// top-1 ids into the acceptance packet (`[k drafts | k + 1 targets]`, int32).
/// The host reads it back; the next block's speculation (`speculateBlock`)
/// slices it on the device. `concatenated` copies each part into the packet in
/// its own launch (the targets are a stride-2 column of the verify's top-two
/// ids). `packet` writes both parts in one launch that reads each part through
/// its own strides: the same int32 values in the same order.
///
/// The walk counts the leading draft == target matches as the sum of the
/// cumulative product of the comparison cast to int32. Scanning the bool
/// comparison directly (prefix AND) and summing it (bool sums to int32) gives
/// the same count without the cast's launch (`chainWalk`).
///
/// `MLXFAST_ACCEPT_GLUE_FUSED=0` keeps the stock ops. Otherwise `prepare()`
/// (at the DFlash2 assistant's load) compares both forms with the stock ones
/// on the production layouts and turns them on only if every value matches
/// without an MLX error.
///
/// WALK1: the walk itself in one launch (`walk`). The comparison, the scan,
/// the sum, the anchor's gather and the confirmed count's add were five
/// launches over 31 ids; one thread now counts the leading matches m and
/// writes the anchor (target m) and the confirmed count m + 1.
/// `MLXFAST_ACCEPT_WALK_FUSED=0` keeps the chain; `prepare()` compares the
/// one-launch walk with the record's chain for every accepted count first.
///
/// ACCWALK: the packet and the walk in one launch. The walk reads only the
/// packet, so the packet's launch also walks it: its threads write the packet
/// and the same ids to threadgroup memory, and after a barrier thread 0 runs
/// the walk's integer loop over them (the anchor `[1]`, the confirmed count
/// `[1]`). The walk of that packet object then takes these two outputs
/// instead of its own launch; any other packet runs the walk's launch.
/// `MLXFAST_ACCEPT_ONE_LAUNCH=0` keeps the two launches; `prepare()` compares
/// the one launch with them (and the known answer) for every accepted count.
public enum CBv2AcceptGlue {
    nonisolated(unsafe) public private(set) static var enabled = false
    nonisolated(unsafe) public private(set) static var walkEnabled = false
    nonisolated(unsafe) public private(set) static var oneLaunch = false
    nonisolated(unsafe) private static var prepared = false
    /// The walk's outputs made by the last packet's launch, for that packet
    /// (held, so no other array takes its identity) and its k.
    nonisolated(unsafe) private static var made:
        (packet: MLXArray, k: Int, anchor: MLXArray, confirmed: MLXArray)?
    private static let lock = NSLock()

    private static let packetKernel = MLXFast.metalKernel(
        name: "cbv2_accept_packet",
        inputNames: ["drafts", "targets"], outputNames: ["packet"],
        source: """
            int i = int(thread_position_in_grid.x);
            int k = drafts_shape[0];
            if (i < k) {
                packet[i] = drafts[int64_t(i) * drafts_strides[0]];
            } else if (i < k + targets_shape[0]) {
                packet[i] = targets[int64_t(i - k) * targets_strides[0]];
            }
            """,
        ensureRowContiguous: false)

    /// The one-row packet (`[drafts, targets]`, 1-D int32, k + 1 targets for
    /// k drafts, k ≤ 16) in one launch; nil keeps `concatenated`.
    public static func packet(_ parts: [MLXArray]) -> MLXArray? {
        lock.withLock { made = nil }
        guard enabled, parts.count == 2 else { return nil }
        if oneLaunch, let out = joinAndWalk(parts[0], parts[1]) {
            lock.withLock { made = (out[0], parts[0].dim(0), out[1], out[2]) }
            return out[0]
        }
        return join(parts[0], parts[1])
    }

    private static let packetWalkKernel = MLXFast.metalKernel(
        name: "cbv2_accept_packet_walk",
        inputNames: ["drafts", "targets"], outputNames: ["packet", "anchor", "confirmed"],
        // One threadgroup of n <= 33 threads; every thread reaches the barrier.
        source: """
            const int i = int(thread_position_in_grid.x);
            const int k = drafts_shape[0];
            const int n = k + targets_shape[0];
            threadgroup int32_t ids[33];
            if (i < n) {
                int32_t v;
                if (i < k) {
                    v = drafts[int64_t(i) * drafts_strides[0]];
                } else {
                    v = targets[int64_t(i - k) * targets_strides[0]];
                }
                ids[i] = v;
                packet[i] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (i == 0) {
                int m = 0;
                while (m < k && ids[m] == ids[k + m]) {
                    ++m;
                }
                anchor[0] = ids[k + m];
                confirmed[0] = m + 1;
            }
            """,
        ensureRowContiguous: false)

    /// `join`'s packet and the walk's anchor `[1]` and confirmed count `[1]`
    /// over it, in one launch (one threadgroup); nil where `join` is nil.
    private static func joinAndWalk(_ drafts: MLXArray, _ targets: MLXArray) -> [MLXArray]? {
        guard drafts.ndim == 1, targets.ndim == 1, drafts.dtype == .int32,
            targets.dtype == .int32, (1 ... 16).contains(drafts.dim(0)),
            targets.dim(0) == drafts.dim(0) + 1
        else { return nil }
        let n = drafts.dim(0) + targets.dim(0)
        return packetWalkKernel(
            [drafts, targets], grid: (n, 1, 1), threadGroup: (n, 1, 1),
            outputShapes: [[n], [1], [1]], outputDTypes: [.int32, .int32, .int32])
    }

    private static func join(_ drafts: MLXArray, _ targets: MLXArray) -> MLXArray? {
        guard drafts.ndim == 1, targets.ndim == 1, drafts.dtype == .int32,
            targets.dtype == .int32, (1 ... 16).contains(drafts.dim(0)),
            targets.dim(0) == drafts.dim(0) + 1
        else { return nil }
        let n = drafts.dim(0) + targets.dim(0)
        return packetKernel(
            [drafts, targets], grid: (n, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[n]], outputDTypes: [.int32])[0]
    }

    private static let walkKernel = MLXFast.metalKernel(
        name: "cbv2_accept_walk",
        inputNames: ["packet"], outputNames: ["anchor", "confirmed"],
        // The grid attribute keeps MLX's generated signature well formed: with
        // shape/stride buffers, two outputs and no attribute it closes the
        // parameter list after the first output.
        source: """
            if (thread_position_in_grid.x != 0) {
                return;
            }
            const int k = (int(packet_shape[0]) - 1) / 2;
            const int64_t s = int64_t(packet_strides[0]);
            int m = 0;
            while (m < k && packet[int64_t(m) * s] == packet[int64_t(k + m) * s]) {
                ++m;
            }
            anchor[0] = packet[int64_t(k + m) * s];
            confirmed[0] = m + 1;
            """,
        ensureRowContiguous: false)

    /// The walk over the packet (`[k drafts | k + 1 targets]`, 1-D int32,
    /// k ≤ 16) in one launch: the anchor `[1]` and the confirmed count `[]`,
    /// int32, as the chain gives them; nil keeps the chain (`chainWalk`).
    public static func walk(_ packet: MLXArray, depth k: Int)
        -> (anchor: MLXArray, confirmed: MLXArray)?
    {
        guard walkEnabled else { return nil }
        let done: (anchor: MLXArray, confirmed: MLXArray)? = lock.withLock {
            guard let m = made, m.packet === packet, m.k == k else { return nil }
            made = nil
            return (m.anchor, m.confirmed.reshaped([]))
        }
        return done ?? oneLaunchWalk(packet, depth: k)
    }

    private static func oneLaunchWalk(_ packet: MLXArray, depth k: Int)
        -> (anchor: MLXArray, confirmed: MLXArray)?
    {
        guard packet.ndim == 1, packet.dtype == .int32, (1 ... 16).contains(k),
            packet.dim(0) == 2 * k + 1
        else { return nil }
        let out = walkKernel(
            [packet], grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [[1], [1]], outputDTypes: [.int32, .int32])
        return (out[0], out[1].reshaped([]))
    }

    /// The walk as ops: `accepted` = the sum of the cumulative product of
    /// draft == target, anchor = target `accepted`, confirmed = `accepted + 1`.
    public static func chainWalk(_ packet: MLXArray, depth k: Int, cast: Bool? = nil)
        -> (anchor: MLXArray, confirmed: MLXArray)
    {
        let targets = packet[k ..< (2 * k + 1)]
        let match = packet[0 ..< k] .== targets[0 ..< k]
        let scanned = (cast ?? !enabled) ? match.asType(.int32) : match
        let accepted = cumprod(scanned, axis: 0).sum().asType(.int32)
        return (targets.take(accepted.reshaped([1]), axis: 0), accepted + MLXArray(Int32(1)))
    }

    /// Load-time proof: the packet against `concatenated` for k = 1...16 with
    /// the targets as a stride-2 column view (and contiguous, and the drafts
    /// strided too), and the walk's count against the stock chain for every
    /// accepted count 0...k, on random int32 ids. Any mismatch or MLX error
    /// keeps the stock glue.
    public static func prepare() {
        guard !prepared else { return }
        prepared = true
        let environment = ProcessInfo.processInfo.environment
        if environment["MLXFAST_ACCEPT_GLUE_FUSED"] != "0" { prepareGlue() }
        if environment["MLXFAST_ACCEPT_WALK_FUSED"] != "0" { prepareWalk() }
        if environment["MLXFAST_ACCEPT_ONE_LAUNCH"] != "0", enabled, walkEnabled {
            prepareOneLaunch()
        }
        CBv2PromptLookupDraft.prepareSpliceFlag()
    }

    /// The one launch against the two it replaces (`join`, then
    /// `oneLaunchWalk`) and the known answer, for k = 1...16 and every
    /// accepted count 0...k, with the drafts and the targets each contiguous
    /// and as a stride-2 column (the verify's top-two layout).
    private static func prepareOneLaunch() {
        var same = true
        var rounds = 0
        do {
            try withError { error in
                var state: UInt64 = 0x9E37_79B9_7F4A_7C15
                func next() -> Int32 {
                    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    return Int32(truncatingIfNeeded: state >> 29)
                }
                var cases: [(one: [MLXArray], two: [MLXArray], host: [Int32], accepted: Int)] = []
                for k in 1 ... 16 {
                    for accepted in 0 ... k {
                        var host = (0 ..< 2 * k + 1).map { _ in next() & 3 }
                        for i in 0 ..< accepted { host[k + i] = host[i] }
                        if accepted < k { host[k + accepted] = host[accepted] &+ 1 }
                        let drafts = Array(host[0 ..< k])
                        let targets = Array(host[k...])
                        let draftRow = MLXArray(drafts, [1, k])
                        let targetRow = MLXArray(targets, [1, k + 1])
                        let draftPairs = MLXArray(drafts.flatMap { [$0, next()] }, [1, k, 2])
                        let topTwo = MLXArray(targets.flatMap { [$0, next()] }, [1, k + 1, 2])
                        for (d, t) in [
                            (draftRow, topTwo[0..., 0..., 0]), (draftRow, targetRow),
                            (draftPairs[0..., 0..., 0], topTwo[0..., 0..., 0]),
                            (draftPairs[0..., 0..., 0], targetRow),
                        ] {
                            let (dr, tr) = (d.reshaped([-1]), t.reshaped([-1]))
                            guard let one = joinAndWalk(dr, tr), let packet = join(dr, tr),
                                let walk = oneLaunchWalk(packet, depth: k)
                            else {
                                same = false
                                continue
                            }
                            cases.append((
                                [one[0], one[1], one[2].reshaped([])],
                                [packet, walk.anchor, walk.confirmed], host, accepted))
                        }
                    }
                }
                eval(cases.flatMap { $0.one + $0.two })
                for c in cases {
                    for (one, two) in zip(c.one, c.two) {
                        same = same && one.dtype == two.dtype && one.shape == two.shape
                            && one.asArray(Int32.self) == two.asArray(Int32.self)
                    }
                    let k = c.host.count / 2
                    same = same && c.two[0].asArray(Int32.self) == c.host
                        && c.two[1].asArray(Int32.self) == [c.host[k + c.accepted]]
                        && c.two[2].asArray(Int32.self) == [Int32(c.accepted + 1)]
                    rounds += 1
                }
                try error.check()
            }
        } catch {
            same = false
        }
        oneLaunch = same
        FileHandle.standardError.write(
            ("accept packet and walk in one launch: "
                + (same
                    ? "self-test passed: \(rounds) rounds (k = 1...16, every accepted count, contiguous and stride-2 drafts and targets) compared with the two launches, 0 mismatches; on\n"
                    : "self-test failed; the two launches kept\n")).data(using: .utf8)!)
    }

    private static func prepareGlue() {
        var same = true
        var values = 0
        var walks = 0
        do {
            try withError { error in
                var state: UInt64 = 0x5DEE_CE66_D1CE_4E5B
                func next() -> Int32 {
                    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    return Int32(truncatingIfNeeded: state >> 29)
                }
                var packets: [(MLXArray, MLXArray)] = []
                var counts: [(MLXArray, MLXArray, Int)] = []
                for k in 1 ... 16 {
                    let draftRow = MLXArray((0 ..< k).map { _ in next() }, [1, k])
                    let draftPairs = MLXArray((0 ..< 2 * k).map { _ in next() }, [1, k, 2])
                    let topTwo = MLXArray((0 ..< 2 * (k + 1)).map { _ in next() }, [1, k + 1, 2])
                    let contiguous = MLXArray((0 ..< k + 1).map { _ in next() }, [1, k + 1])
                    for (drafts, targets) in [
                        (draftRow, topTwo[0..., 0..., 0]),
                        (draftPairs[0..., 0..., 1], topTwo[0..., 0..., 1]),
                        (draftPairs[0..., 0..., 0], contiguous),
                    ] {
                        let parts = [drafts.reshaped([-1]), targets.reshaped([-1])]
                        guard let fused = join(parts[0], parts[1]) else {
                            same = false
                            continue
                        }
                        packets.append((fused, concatenated(parts, axis: 0)))
                    }
                    for accepted in 0 ... k {
                        var host = (0 ..< 2 * k + 1).map { _ in next() & 3 }
                        for i in 0 ..< accepted { host[k + i] = host[i] }
                        if accepted < k { host[k + accepted] = host[accepted] &+ 1 }
                        let packet = MLXArray(host, [2 * k + 1])
                        let match = packet[0 ..< k] .== packet[k ..< (2 * k)]
                        counts.append((
                            cumprod(match, axis: 0).sum().asType(.int32),
                            cumprod(match.asType(.int32), axis: 0).sum().asType(.int32),
                            accepted))
                    }
                }
                eval(packets.flatMap { [$0.0, $0.1] } + counts.flatMap { [$0.0, $0.1] })
                for (fused, stock) in packets {
                    same = same && fused.dtype == .int32 && fused.shape == stock.shape
                        && fused.asArray(Int32.self) == stock.asArray(Int32.self)
                    values += stock.size
                }
                for (fused, stock, accepted) in counts {
                    same = same && fused.dtype == stock.dtype && fused.shape == stock.shape
                        && fused.item(Int32.self) == stock.item(Int32.self)
                        && stock.item(Int32.self) == Int32(accepted)
                    walks += 1
                }
                try error.check()
            }
        } catch {
            same = false
        }
        enabled = same
        FileHandle.standardError.write(
            ("accept glue (packet in one launch, walk without its cast): "
                + (same
                    ? "self-test passed: \(values) packet values and \(walks) accept walks compared, 0 mismatches; on\n"
                    : "self-test failed; stock concatenation and cast kept\n")).data(using: .utf8)!)
    }

    /// The one-launch walk against the record's chain (the int32 cast) and the
    /// known answer, for k = 1...16 and every accepted count 0...k, on a
    /// contiguous packet and on a stride-2 view of one.
    private static func prepareWalk() {
        var same = true
        var walks = 0
        do {
            try withError { error in
                var state: UInt64 = 0x2545_F491_4F6C_DD1D
                func next() -> Int32 {
                    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    return Int32(truncatingIfNeeded: state >> 29)
                }
                var cases: [(fused: (MLXArray, MLXArray), chain: (MLXArray, MLXArray), anchor: Int32, confirmed: Int32)] = []
                for k in 1 ... 16 {
                    for accepted in 0 ... k {
                        var host = (0 ..< 2 * k + 1).map { _ in next() & 3 }
                        for i in 0 ..< accepted { host[k + i] = host[i] }
                        if accepted < k { host[k + accepted] = host[accepted] &+ 1 }
                        let pairs = host.flatMap { [$0, next()] }
                        for packet in [
                            MLXArray(host, [2 * k + 1]), MLXArray(pairs, [2 * k + 1, 2])[0..., 0],
                        ] {
                            guard let fused = oneLaunchWalk(packet, depth: k) else {
                                same = false
                                continue
                            }
                            let chain = chainWalk(packet, depth: k, cast: true)
                            cases.append((
                                (fused.anchor, fused.confirmed), (chain.anchor, chain.confirmed),
                                host[k + accepted], Int32(accepted + 1)))
                        }
                    }
                }
                eval(cases.flatMap { [$0.fused.0, $0.fused.1, $0.chain.0, $0.chain.1] })
                for c in cases {
                    for (fused, chain, expected) in [
                        (c.fused.0, c.chain.0, c.anchor), (c.fused.1, c.chain.1, c.confirmed),
                    ] {
                        same = same && fused.dtype == chain.dtype && fused.shape == chain.shape
                            && fused.item(Int32.self) == chain.item(Int32.self)
                            && chain.item(Int32.self) == expected
                    }
                    walks += 1
                }
                try error.check()
            }
        } catch {
            same = false
        }
        walkEnabled = same
        FileHandle.standardError.write(
            ("accept walk in one launch: "
                + (same
                    ? "self-test passed: \(walks) walks (k = 1...16, every accepted count, contiguous and strided packets) compared with the chain, 0 mismatches; on\n"
                    : "self-test failed; the chain kept\n")).data(using: .utf8)!)
    }
}
