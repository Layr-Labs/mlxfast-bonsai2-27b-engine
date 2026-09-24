// Copyright © 2026 Eigen Labs.

@preconcurrency import AVFoundation
import Foundation

/// Errors raised before or while AVFoundation reads an in-memory ISO BMFF video.
public enum MemoryBackedVideoAssetError: Error, Equatable, Sendable {
    case emptyData
    case invalidContainer
    case invalidRange
    case unexpectedResource
}

/// Owns an `AVURLAsset` whose bytes are served directly from memory.
///
/// `AVAssetResourceLoader` keeps its delegate weakly, so returning a bare
/// `AVURLAsset` is unsafe: the delegate (and therefore the bytes) can disappear
/// while AVFoundation is still probing or sampling the video. This owner keeps
/// the asset, loader, data, and delegate queue alive together. The custom URL is
/// never backed by a file or network resource.
public final class MemoryBackedVideoAsset: @unchecked Sendable {
    private let asset: AVURLAsset

    private let loader: MemoryAssetResourceLoader
    private let delegateQueue: DispatchQueue

    public var byteCount: Int { loader.byteCount }

    /// Creates an in-memory ISO BMFF MP4 or QuickTime asset.
    ///
    /// The top-level atom table must be structurally complete. Modern files are
    /// identified from `ftyp`; legacy QuickTime files without `ftyp` must contain
    /// both `moov` and `mdat`. This rejects arbitrary bytes before AVFoundation
    /// sees them rather than asking the framework to infer an unrelated format.
    public init(videoData data: Data) throws {
        let contentType = try Self.validatedContentType(data)

        let token = UUID().uuidString.lowercased()
        guard let resourceURL = URL(string: "darkbloom-memory-video://\(token)/asset.mp4") else {
            throw MemoryBackedVideoAssetError.invalidContainer
        }
        let loader = MemoryAssetResourceLoader(
            data: data,
            resourceURL: resourceURL,
            contentType: contentType)
        let queue = DispatchQueue(
            label: "ai.darkbloom.memory-video.\(token)",
            qos: .userInitiated)
        let asset = AVURLAsset(
            url: resourceURL,
            options: [
                AVURLAssetPreferPreciseDurationAndTimingKey: true,
                AVURLAssetReferenceRestrictionsKey:
                    AVAssetReferenceRestrictions.forbidAll.rawValue,
            ])
        asset.resourceLoader.setDelegate(loader, queue: queue)

        self.asset = asset
        self.loader = loader
        self.delegateQueue = queue
    }

    /// Runs an operation while keeping the asset's resource-loader owner alive.
    ///
    /// This is package-scoped so a bare memory-backed asset cannot escape through
    /// the public API. The explicit lifetime guarantee covers every suspension
    /// and every throwing exit from the operation.
    package func withAsset<Result>(
        _ operation: (AVURLAsset) async throws -> Result
    ) async rethrows -> Result {
        defer { withExtendedLifetime(self) {} }
        return try await operation(asset)
    }

    package var resourceRequestCount: Int { loader.resourceRequestCount }

    static func validatedContentType(_ data: Data) throws -> String {
        guard !data.isEmpty else { throw MemoryBackedVideoAssetError.emptyData }
        guard data.count >= 8 else { throw MemoryBackedVideoAssetError.invalidContainer }

        let ftyp = UInt64(0x66_74_79_70)
        let moov = UInt64(0x6d_6f_6f_76)
        let mdat = UInt64(0x6d_64_61_74)
        let quickTimeBrand = UInt64(0x71_74_20_20)

        var offset = 0
        var ftypContentType: String?
        var foundMoov = false
        var foundMdat = false

        while offset < data.count {
            let remaining = data.count - offset
            guard remaining >= 8,
                let size32 = unsignedInteger(in: data, at: offset, byteCount: 4),
                let atomType = unsignedInteger(in: data, at: offset + 4, byteCount: 4)
            else { throw MemoryBackedVideoAssetError.invalidContainer }

            let headerSize: Int
            let atomSize: Int
            switch size32 {
            case 0:
                headerSize = 8
                atomSize = remaining
            case 1:
                headerSize = 16
                guard remaining >= headerSize,
                    let extendedSize = unsignedInteger(
                        in: data, at: offset + 8, byteCount: 8),
                    extendedSize >= UInt64(headerSize),
                    extendedSize <= UInt64(remaining)
                else { throw MemoryBackedVideoAssetError.invalidContainer }
                atomSize = Int(extendedSize)
            default:
                headerSize = 8
                guard size32 >= UInt64(headerSize), size32 <= UInt64(remaining) else {
                    throw MemoryBackedVideoAssetError.invalidContainer
                }
                atomSize = Int(size32)
            }

            let (atomEnd, overflow) = offset.addingReportingOverflow(atomSize)
            guard !overflow, atomEnd <= data.count else {
                throw MemoryBackedVideoAssetError.invalidContainer
            }

            switch atomType {
            case ftyp:
                guard ftypContentType == nil, size32 != 0, atomSize >= headerSize + 8,
                    let majorBrand = unsignedInteger(
                        in: data, at: offset + headerSize, byteCount: 4)
                else { throw MemoryBackedVideoAssetError.invalidContainer }
                ftypContentType =
                    majorBrand == quickTimeBrand ? AVFileType.mov.rawValue : AVFileType.mp4.rawValue
            case moov:
                foundMoov = true
            case mdat:
                foundMdat = true
            default:
                break
            }

            offset = atomEnd
        }

        if let ftypContentType {
            return ftypContentType
        }
        guard foundMoov, foundMdat else {
            throw MemoryBackedVideoAssetError.invalidContainer
        }
        return AVFileType.mov.rawValue
    }

    private static func unsignedInteger(
        in data: Data, at offset: Int, byteCount: Int
    ) -> UInt64? {
        guard offset >= 0, byteCount == 4 || byteCount == 8 else { return nil }
        let (end, overflow) = offset.addingReportingOverflow(byteCount)
        guard !overflow, end <= data.count else { return nil }

        var value: UInt64 = 0
        var index = data.index(data.startIndex, offsetBy: offset)
        for _ in 0 ..< byteCount {
            value = (value << 8) | UInt64(data[index])
            data.formIndex(after: &index)
        }
        return value
    }

    /// Computes the safe slice for an AVFoundation data request.
    ///
    /// AVFoundation may ask past EOF; that request is clamped to EOF. Negative,
    /// overflowing, backward, and wholly out-of-resource offsets are rejected.
    /// Kept internal so range semantics can be tested without constructing the
    /// framework-owned `AVAssetResourceLoadingRequest` type.
    static func responseRange(
        resourceLength: Int,
        requestedOffset: Int64,
        requestedLength: Int,
        currentOffset: Int64,
        requestsAllDataToEnd: Bool
    ) throws -> Range<Int> {
        guard resourceLength >= 0, requestedOffset >= 0, requestedLength >= 0,
            currentOffset >= 0
        else { throw MemoryBackedVideoAssetError.invalidRange }

        let length = Int64(resourceLength)
        let start = max(requestedOffset, currentOffset)
        guard start <= length else { throw MemoryBackedVideoAssetError.invalidRange }

        let requestedEnd: Int64
        if requestsAllDataToEnd {
            requestedEnd = length
        } else {
            let (end, overflow) = requestedOffset.addingReportingOverflow(
                Int64(requestedLength))
            guard !overflow, end >= start else {
                throw MemoryBackedVideoAssetError.invalidRange
            }
            requestedEnd = min(end, length)
        }

        return Int(start) ..< Int(requestedEnd)
    }
}

private final class MemoryAssetResourceLoader: NSObject,
    AVAssetResourceLoaderDelegate, @unchecked Sendable
{
    let byteCount: Int

    var resourceRequestCount: Int {
        stateLock.withLock { _resourceRequestCount }
    }

    private let data: Data
    private let resourceURL: URL
    private let contentType: String
    private let stateLock = NSLock()
    private var _resourceRequestCount = 0

    init(data: Data, resourceURL: URL, contentType: String) {
        self.data = data
        self.byteCount = data.count
        self.resourceURL = resourceURL
        self.contentType = contentType
        super.init()
    }

    func resourceLoader(
        _ resourceLoader: AVAssetResourceLoader,
        shouldWaitForLoadingOfRequestedResource loadingRequest: AVAssetResourceLoadingRequest
    ) -> Bool {
        stateLock.withLock {
            _resourceRequestCount += 1
        }
        guard loadingRequest.request.url == resourceURL else {
            loadingRequest.finishLoading(with: MemoryBackedVideoAssetError.unexpectedResource)
            return true
        }

        if let information = loadingRequest.contentInformationRequest {
            information.contentType = contentType
            information.contentLength = Int64(data.count)
            information.isByteRangeAccessSupported = true
        }

        guard let request = loadingRequest.dataRequest else {
            loadingRequest.finishLoading()
            return true
        }

        do {
            let range = try MemoryBackedVideoAsset.responseRange(
                resourceLength: data.count,
                requestedOffset: request.requestedOffset,
                requestedLength: request.requestedLength,
                currentOffset: request.currentOffset,
                requestsAllDataToEnd: request.requestsAllDataToEndOfResource)
            if !range.isEmpty {
                let lowerBound = data.index(data.startIndex, offsetBy: range.lowerBound)
                let upperBound = data.index(data.startIndex, offsetBy: range.upperBound)
                request.respond(with: data[lowerBound ..< upperBound])
            }
            loadingRequest.finishLoading()
        } catch {
            loadingRequest.finishLoading(with: error)
        }
        return true
    }

    func resourceLoader(
        _ resourceLoader: AVAssetResourceLoader,
        didCancel loadingRequest: AVAssetResourceLoadingRequest
    ) {
        // Cancellation is terminal and owned by AVFoundation. In particular,
        // do not retry, fetch another URL, or materialize a fallback file.
    }
}
