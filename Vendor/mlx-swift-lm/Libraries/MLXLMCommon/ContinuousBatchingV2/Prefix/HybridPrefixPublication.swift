import MLX

/// A single publication-queue owner. The engine keeps the retired source KV
/// reserved until completion; `kvBytes` separately charges any bank-owned
/// destination before its copy graph is built or evaluated.
final class CBv2HybridPrefixPublication {
    typealias Row = (keys: MLXArray, values: MLXArray, offset: Int)

    var checkpoints: [CBv2RecurrentCheckpoint]
    var kv: [Row?]
    var kvBytes: Int
    let mayCompact: Bool

    init(checkpoints: [CBv2RecurrentCheckpoint], kv: [Row?], kvBytes: Int, mayCompact: Bool) {
        self.checkpoints = checkpoints
        self.kv = kv
        self.kvBytes = kvBytes
        self.mayCompact = mayCompact
    }

    /// Logical bytes for an exact prefix, independent of donor allocation
    /// slack. Invalid shapes and overflow cannot authorize a copy reservation.
    func compactedBytes(through position: Int) -> Int? {
        guard position > 0, !kv.isEmpty else { return nil }
        var bytes = 0
        for row in kv {
            guard let row, row.offset >= position else { return nil }
            for array in [row.keys, row.values] {
                guard array.ndim == 4, array.dim(2) >= position else { return nil }
                var size = array.dtype.size
                for dimension in [array.dim(0), array.dim(1), position, array.dim(3)] {
                    let (product, overflow) = size.multipliedReportingOverflow(by: dimension)
                    guard dimension > 0, !overflow else { return nil }
                    size = product
                }
                let (sum, overflow) = bytes.addingReportingOverflow(size)
                guard !overflow else { return nil }
                bytes = sum
            }
        }
        return bytes
    }

    /// Called only after reservation and shape validation, under the bank lock
    /// so a prior close cannot start new copies. No GPU wait occurs here.
    func compactKV(through position: Int) {
        kv = kv.map { row in
            guard let row else { return nil }
            let keys = row.keys[.ellipsis, 0 ..< position, 0...]
            let values = row.values[.ellipsis, 0 ..< position, 0...]
            // Byte-preserving copies release the donor's unused tail/slack.
            return (
                keys: MLX.where(MLXArray(true), keys, keys),
                values: MLX.where(MLXArray(true), values, values), offset: position)
        }
    }
}
