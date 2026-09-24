import MLX

/// One declared family of request-owned buffers, separate from target KV.
public struct CBv2AuxiliaryAllocationSpec: Sendable {
    public let bytesPerToken: Int
    public let allocationCount: Int
    public let tokenGranularity: Int
    public let tokenPadding: Int
    /// History can consist of many immutable chunks with independent padding.
    public let partitioned: Bool

    public init(bytesPerToken: Int, allocationCount: Int = 1,
                tokenGranularity: Int = 1, tokenPadding: Int = 0, partitioned: Bool = false) {
        self.bytesPerToken = bytesPerToken
        self.allocationCount = allocationCount
        self.tokenGranularity = tokenGranularity
        self.tokenPadding = tokenPadding
        self.partitioned = partitioned
    }
}

/// Captured allocator geometry and model buffer counts. Queries under Admission
/// use checked arithmetic only; no callback, allocation, or N-sized lookup table.
public struct CBv2AuxiliaryAllocationProjection: Sendable {
    private struct Buffer: Sendable {
        let spec: CBv2AuxiliaryAllocationSpec
        let partitionedBytesPerToken: Int?
    }
    private let policy: AllocationFootprintPolicy
    private let buffers: [Buffer]
    let maximumGrowthBytes: Int

    public init?(policy: AllocationFootprintPolicy, buffers specs: [CBv2AuxiliaryAllocationSpec]) {
        guard let overhead = policy.maximumExtraBytes else { return nil }
        var buffers: [Buffer] = [], growth = 0
        for spec in specs {
            guard spec.bytesPerToken > 0, spec.allocationCount > 0,
                  spec.tokenGranularity > 0, spec.tokenPadding >= 0 else { return nil }
            let partitioned: Int?
            if spec.partitioned {
                guard let bytes = Self.maximumBytesPerUnit(spec.bytesPerToken, policy: policy) else { return nil }
                partitioned = bytes
            } else { partitioned = nil }
            // Include the initial 0 -> 1 transition as well as later block
            // crossings; padding is request-local and appears only once.
            guard let firstPadded = Self.add(1, spec.tokenPadding),
                  let firstRounding = Self.add(firstPadded, spec.tokenGranularity - 1),
                  let firstRows = Self.multiply(firstRounding / spec.tokenGranularity, spec.tokenGranularity),
                  let block = Self.multiply(partitioned ?? spec.bytesPerToken, firstRows),
                  let bound = Self.add(block, partitioned == nil ? overhead : 0),
                  let all = Self.multiply(bound, spec.allocationCount),
                  let next = Self.add(growth, all) else { return nil }
            growth = next
            buffers.append(.init(spec: spec, partitionedBytesPerToken: partitioned))
        }
        self.policy = policy
        self.buffers = buffers
        self.maximumGrowthBytes = growth
    }

    func bytes(forTokens tokens: Int) -> Int? {
        guard tokens >= 0 else { return nil }
        if tokens == 0 { return 0 }
        var total = 0
        for buffer in buffers {
            let spec = buffer.spec
            guard let padded = Self.add(tokens, spec.tokenPadding),
                  let rounding = Self.add(padded, spec.tokenGranularity - 1),
                  let rows = Self.multiply(rounding / spec.tokenGranularity, spec.tokenGranularity),
                  let logical = Self.multiply(rows, buffer.partitionedBytesPerToken ?? spec.bytesPerToken),
                  let physical = buffer.partitionedBytesPerToken == nil ? policy.upperBound(byteCount: logical) : logical,
                  let all = Self.multiply(physical, spec.allocationCount),
                  let next = Self.add(total, all) else { return nil }
            total = next
        }
        return total
    }

    /// Bound one unit even when several units share an allocation or split
    /// across independent buffers. B(n*u) <= n*u+extra lets this finite scan
    /// stop once every remaining ratio is no greater than the current bound.
    /// Initialization only; token admission never performs this scan.
    static func maximumBytesPerUnit(_ unit: Int, policy: AllocationFootprintPolicy) -> Int? {
        guard unit > 0, let extra = policy.maximumExtraBytes else { return nil }
        var count = 1, best = unit
        while best == unit || count <= extra / (best - unit) {
            guard let logical = multiply(count, unit),
                  let bytes = policy.upperBound(byteCount: logical),
                  let rounded = add(bytes / count, bytes % count == 0 ? 0 : 1) else { return nil }
            best = max(best, rounded)
            if extra == 0 { return best }
            guard let next = add(count, 1) else { return nil }
            count = next
        }
        return best
    }

    private static func add(_ a: Int, _ b: Int) -> Int? {
        let (value, overflow) = a.addingReportingOverflow(b)
        return a >= 0 && b >= 0 && !overflow ? value : nil
    }

    private static func multiply(_ a: Int, _ b: Int) -> Int? {
        let (value, overflow) = a.multipliedReportingOverflow(by: b)
        return a >= 0 && b >= 0 && !overflow ? value : nil
    }
}
