import Cmlx
import Foundation
import MLX

/// Immutable staging geometry. Computing this plan builds no page-address map
/// and allocates no native buffers. Only the actual checkpoint M is staged;
/// the active pool later supplies the unfilled part of the request's N promise.
struct CBv2PagedCheckpointStoragePlan: Sendable {
    struct Layer: Sendable {
        let modelIndex: Int
        let tokenStart: Int
        let tokenCount: Int
        let ringPages: Int?
        let key: PagedKVGroupKey
        let firstPage: Int
        let pageCount: Int
    }

    struct Group: Sendable {
        let key: PagedKVGroupKey
        let layout: PagedKVSegmentLayout
        let usablePages: Int
        let nativeBytes: Int
    }

    let position: Int
    let pageSize: Int
    let layers: [Layer]
    let groups: [Group]
    /// Allocation bound before buffers exist; evaluated storage has an exact footprint.
    let nativeBytes: Int
    let ownerMap: [Int]

    init(layerKinds: [CBv2LayerKind], config: PagedKVPoolConfig, position: Int,
         historicalLayout: CBv2HistoricalAttentionLayout? = nil, maximumSequenceLength: Int? = nil) throws {
        guard position > 1, position <= Int(Int32.max), config.pageSize > 0,
            let targetBytes = config.segmentSizeBytes, config.layerDTypes != nil
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        let types = try PagedKVStorageLayout.resolve(layerKinds: layerKinds, config: config)
        if let historicalLayout {
            guard historicalLayout.layers == (try CBv2HistoricalAttentionLayout(layerKinds: layerKinds, dtypes: types)).layers,
                  let maximumSequenceLength, maximumSequenceLength >= position,
                  maximumSequenceLength <= Int(Int32.max)
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        }
        let pages = (position - 1) / config.pageSize + 1
        var layers: [Layer] = []
        var demand: [PagedKVGroupKey: Int] = [:]
        let ownerMap = historicalLayout?.layers.map(\.owner) ?? Array(layerKinds.indices)
        guard ownerMap.count == layerKinds.count,
              historicalLayout == nil || maximumSequenceLength.map({ $0 >= position }) == true
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        for (index, kind) in layerKinds.enumerated() where ownerMap[index] == index {
            let start: Int, ring: Int?, count: Int
            if let historicalLayout {
                guard historicalLayout.layers[index].dtype.mlxDType == types[index] else {
                    throw CBv2CompleteCheckpointError.incompatibleCheckpoint
                }
                start = historicalLayout.layers[index].tokenStart(at: position)
                if let window = historicalLayout.layers[index].window {
                    ring = PagedKVPool.ringPageCount(window: window, config: config)
                    // Dense canonical ring slots, including speculative margin.
                    // Missing slots would change modulo addressing after resume.
                    count = PagedKVPool.pageDemand(kind: kind, maxLength: maximumSequenceLength!, config: config)
                } else { ring = nil; count = pages }
            } else {
                guard case .full = kind.attention, kind.sharesKVWithLayer == nil,
                      kind.kvHeads > 0, kind.headDim >= 64
                else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                start = 0; ring = nil; count = pages
            }
            let key = PagedKVGroupKey(kind, dtype: types[index], separateWindow: true)
            let first = demand[key, default: 0]
            let (next, overflow) = first.addingReportingOverflow(count)
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            layers.append(Layer(modelIndex: index, tokenStart: start, tokenCount: position - start,
                                ringPages: ring, key: key, firstPage: first, pageCount: count))
            demand[key] = next
        }
        guard !layers.isEmpty else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        var groups: [Group] = []
        var total = 0
        for key in demand.keys.sorted(by: { $0.sortKey < $1.sortKey }) {
            let pageBytes = try CBv2CheckpointTensorDescriptor.checkedByteCount(
                shape: [2, key.kvHeads, config.pageSize, key.headDim], dtype: key.dtype)
            let layout = try PagedKVSegmentLayout(
                pageBytes: pageBytes, targetBytes: targetBytes,
                maximumBufferBytes: config.maxBufferLength,
                maximumAddressPages: Int(Int32.max) / config.pageSize)
            let count = demand[key]!
            guard let bytes = layout.allocationBytes(addingUsablePages: count) else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            let (next, overflow) = total.addingReportingOverflow(bytes)
            guard !overflow else { throw CBv2CompleteCheckpointError.invalidManifest }
            total = next
            groups.append(Group(key: key, layout: layout, usablePages: count, nativeBytes: bytes))
        }
        self.ownerMap = ownerMap
        self.position = position
        self.pageSize = config.pageSize
        self.layers = layers
        self.groups = groups
        self.nativeBytes = total
    }
}

/// Private, evaluated segment buffers. No page handle from this owner is valid
/// in an engine pool until an all-group attachment transaction rebases it.
/// The caller must hold the plan's native byte reservation before construction
/// and keep it until these buffers are destroyed or transferred to that pool.
final class CBv2PagedCheckpointStorage {
    struct Group {
        let layout: PagedKVSegmentLayout
        let segments: [Int: PagedKVSegment]
        /// Logical ordinal across this group's owning layers -> private page.
        let pages: [Int32]
    }

    let plan: CBv2PagedCheckpointStoragePlan
    private(set) var groups: [PagedKVGroupKey: Group] = [:]
    var allocatedBytes: Int {
        groups.values.reduce(0) { total, group in
            total + group.segments.values.reduce(0) { $0 + $1.allocatedBytes }
        }
    }

    init(plan: CBv2PagedCheckpointStoragePlan, evaluate: (MLXArray) throws -> Void,
         admission: AdmissionV2? = nil) throws {
        self.plan = plan
        for group in plan.groups {
            let prepared = try group.layout.adding(usablePages: group.usablePages, excluding: [])
            var segments: [Int: PagedKVSegment] = [:]
            var pages: [Int32] = []
            pages.reserveCapacity(group.usablePages)
            for index in prepared.segmentIDs {
                let segment = try PagedKVSegment(
                    index: index, layout: prepared.layout, key: group.key,
                    pageSize: plan.pageSize, dtype: group.key.dtype, evaluate: evaluate,
                    admission: admission)
                guard mlx_array_data_uint8(segment.storage.ctx) != nil else {
                    throw CBv2CompleteCheckpointError.allocationFailed
                }
                segments[index] = segment
                pages.append(contentsOf: segment.pages.dropFirst().map(Int32.init))
            }
            groups[group.key] = Group(layout: prepared.layout, segments: segments, pages: pages)
        }
    }

    /// Ordered-transfer validation belongs to the enclosing import owner.
    /// This sink validates its native addressing independently before writing.
    func append(layerIndex: Int, values: Bool, byteOffset: Int, data: Data) throws {
        guard plan.layers.indices.contains(layerIndex) else {
            throw CBv2CompleteCheckpointError.invalidSegment
        }
        let layer = plan.layers[layerIndex]
        guard let group = groups[layer.key] else { throw CBv2CompleteCheckpointError.closed }
        let bytes = try CBv2CheckpointTensorDescriptor.checkedByteCount(
            shape: [1, layer.key.kvHeads, layer.tokenCount, layer.key.headDim], dtype: layer.key.dtype)
        let width = layer.key.dtype.size
        guard byteOffset >= 0, byteOffset < bytes, byteOffset % width == 0,
            !data.isEmpty, data.count <= CBv2CompleteCheckpointManifest.maximumSegmentBytes,
            data.count % width == 0, data.count <= bytes - byteOffset
        else { throw CBv2CompleteCheckpointError.invalidSegment }
        try data.withUnsafeBytes { source in
            try CBv2PagedCheckpointByteLayout.runs(
                headDim: layer.key.headDim, position: layer.tokenCount, pageSize: plan.pageSize,
                tokenStart: layer.tokenStart, ringPages: layer.ringPages, itemSize: width, byteOffset: byteOffset, count: data.count
            ) { logicalPage, head, slot, feature, packedOffset, count in
                let page = group.pages[layer.firstPage + logicalPage]
                let segment = group.segments[group.layout.segmentIndex(page: page)]!
                guard let pointer = mlx_array_data_uint8(segment.storage.ctx) else {
                    throw CBv2CompleteCheckpointError.allocationFailed
                }
                let localPage = group.layout.localPage(page)
                let element = ((localPage * layer.key.kvHeads + head) * plan.pageSize + slot)
                    * layer.key.headDim + feature + (values ? segment.valueOffset : 0)
                UnsafeMutableRawPointer(mutating: pointer).advanced(by: element * width).copyMemory(
                    from: source.baseAddress!.advanced(by: packedOffset), byteCount: count)
            }
        }
    }

    func close() { groups.removeAll() }
}

/// Packed [1,H,M,D] bytes split at physical page and head boundaries. Partial
/// feature vectors are legal; dtype widths, rather than numeric casts, govern
/// every copy. Descriptor validation checks all products before this helper.
enum CBv2PagedCheckpointByteLayout {
    static func runs(
        headDim: Int, position: Int, pageSize: Int, tokenStart: Int = 0,
        ringPages: Int? = nil, itemSize: Int,
        byteOffset: Int, count: Int,
        run: (Int, Int, Int, Int, Int, Int) throws -> Void
    ) rethrows {
        var element = byteOffset / itemSize
        var copied = 0
        while copied < count {
            let head = element / (position * headDim)
            let token = (element / headDim) % position
            let feature = element % headDim
            let absolute = tokenStart + token
            let slot = absolute % pageSize
            let availableTokens = min(pageSize - slot, position - token)
            let available = (availableTokens * headDim - feature) * itemSize
            let length = min(available, count - copied)
            let page = ringPages.map { (absolute / pageSize) % $0 } ?? (absolute / pageSize)
            try run(page, head, slot, feature, copied, length)
            element += length / itemSize
            copied += length
        }
    }
}
