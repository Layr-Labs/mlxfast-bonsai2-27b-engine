// Copyright © 2024 Apple Inc.

import AVFoundation
import CoreMedia
import Foundation
import MLX
import MLXVLM
import XCTest

@testable import MLXLMCommon

private enum SuspendedDurationLoaderError: Error {
    case cancelledByTest
    case pendingRequestStreamEnded
    case pendingRequestTimedOut
}

private final class SuspendedDurationResourceLoader: NSObject,
    AVAssetResourceLoaderDelegate, @unchecked Sendable
{
    private let delegateQueue = DispatchQueue(
        label: "MediaProcessingTests.SuspendedDurationResourceLoader")
    private let resourceData = Data([
        0, 0, 0, 20, 0x66, 0x74, 0x79, 0x70,
        0x71, 0x74, 0x20, 0x20, 0, 0, 0, 0,
        0x71, 0x74, 0x20, 0x20,
    ])
    private let stateLock = NSLock()
    private let pendingRequestEvents: AsyncStream<Void>
    private let pendingRequestContinuation: AsyncStream<Void>.Continuation

    var pendingRequestCount: Int {
        stateLock.withLock { pendingRequests.count }
    }
    private var pendingRequests: [AVAssetResourceLoadingRequest] = []
    private var hasSignalledPendingRequest = false

    override init() {
        let events = AsyncStream<Void>.makeStream(
            of: Void.self, bufferingPolicy: .bufferingNewest(1))
        pendingRequestEvents = events.stream
        pendingRequestContinuation = events.continuation
        super.init()
    }

    func makeAsset() -> AVURLAsset {
        let url = URL(string: "suspended-duration://asset/video.mov")!
        let asset = AVURLAsset(url: url)
        asset.resourceLoader.setDelegate(self, queue: delegateQueue)
        return asset
    }

    func waitForPendingRequest() async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { [pendingRequestEvents] in
                var iterator = pendingRequestEvents.makeAsyncIterator()
                guard await iterator.next() != nil else {
                    throw SuspendedDurationLoaderError.pendingRequestStreamEnded
                }
            }
            group.addTask {
                try await Task.sleep(for: .seconds(5))
                throw SuspendedDurationLoaderError.pendingRequestTimedOut
            }
            defer { group.cancelAll() }
            _ = try await group.next()
        }
    }

    func failPendingRequests() {
        let requests = stateLock.withLock {
            defer { pendingRequests.removeAll() }
            return pendingRequests
        }
        for request in requests {
            request.finishLoading(with: SuspendedDurationLoaderError.cancelledByTest)
        }
    }

    func resourceLoader(
        _ resourceLoader: AVAssetResourceLoader,
        shouldWaitForLoadingOfRequestedResource loadingRequest: AVAssetResourceLoadingRequest
    ) -> Bool {
        if let information = loadingRequest.contentInformationRequest {
            information.contentType = AVFileType.mov.rawValue
            information.contentLength = Int64(resourceData.count)
            information.isByteRangeAccessSupported = true
        }

        guard loadingRequest.dataRequest != nil else {
            loadingRequest.finishLoading()
            return true
        }

        let shouldSignal = stateLock.withLock {
            pendingRequests.append(loadingRequest)
            guard !hasSignalledPendingRequest else { return false }
            hasSignalledPendingRequest = true
            return true
        }
        if shouldSignal {
            pendingRequestContinuation.yield()
        }
        return true
    }

    func resourceLoader(
        _ resourceLoader: AVAssetResourceLoader,
        didCancel loadingRequest: AVAssetResourceLoadingRequest
    ) {
        stateLock.withLock {
            pendingRequests.removeAll { $0 === loadingRequest }
        }
    }
}

public class MediaProcesingTests: XCTestCase {

    func testResize() {
        // resampleBicubic should produce an image with the desired dimensions
        let inputFilter = CIFilter(name: "CIConstantColorGenerator")!
        inputFilter.setValue(CIColor.red, forKey: "inputColor")
        let input = inputFilter.outputImage!.cropped(
            to: CGRect(x: 0, y: 0, width: 1536, height: 1106))

        let target = CGSize(width: 1540, height: 1120)
        let output = MediaProcessing.resampleBicubic(input, to: target)

        XCTAssertEqual(output.extent.size, target)
    }

    func testVideoFileAsSimpleProcessedSequence() async throws {
        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }

        let video = UserInput.Video.url(fileURL)

        // We know video is exactly 5 seconds long, expect 5 samples
        let frames = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 1)

        XCTAssert(frames.frames.count == 5)
    }

    func testTemporaryMemoryBackedOwnerSurvivesFullSamplingWithoutTempFile() async throws {
        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }
        let bytes = try Data(contentsOf: fileURL)
        let before = try memoryVideoTempFiles()

        let frames = try await MediaProcessing.asProcessedSequence(
            .memoryBacked(try MemoryBackedVideoAsset(videoData: bytes)),
            samplesPerSecond: 1)

        XCTAssertEqual(frames.frames.count, 5)
        XCTAssertEqual(try memoryVideoTempFiles(), before)
    }

    func testMemoryBackedMetadataRetainsTemporaryOwnerAndReportsDecodedDimensions() async throws {
        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }
        let bytes = try Data(contentsOf: fileURL)

        let metadata = try await MediaProcessing.metadata(
            for: try MemoryBackedVideoAsset(videoData: bytes))

        XCTAssertEqual(metadata.duration.seconds, 5, accuracy: 0.01)
        XCTAssertFalse(metadata.tracks.isEmpty)
        XCTAssertTrue(
            metadata.tracks.contains { track in
                let sizes = track.decodedFrameDimensions + [track.naturalSize].compactMap { $0 }
                return sizes.contains {
                    Int($0.width) * Int($0.height) == 1920 * 1080
                }
            })
    }

    func testSlicedMemoryBackedDataSupportsRandomAccessAndFullSampling() async throws {
        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }
        let bytes = try Data(contentsOf: fileURL)
        let prefixCount = 37
        var prefixed = Data(repeating: 0xa5, count: prefixCount)
        prefixed.append(bytes)
        let sliced = prefixed.dropFirst(prefixCount)
        XCTAssertEqual(sliced.startIndex, prefixCount)

        let frames = try await MediaProcessing.asProcessedSequence(
            .memoryBacked(try MemoryBackedVideoAsset(videoData: sliced)),
            samplesPerSecond: 1)

        XCTAssertEqual(frames.frames.count, 5)
    }

    func testQuickTimeContentTypeUsesNormalAndExtendedFtypMajorBrandOffsets() throws {
        let normalFtyp = Data([
            0, 0, 0, 16, 0x66, 0x74, 0x79, 0x70,
            0x71, 0x74, 0x20, 0x20, 0, 0, 0, 0,
        ])
        let extendedFtyp = Data([
            0, 0, 0, 1, 0x66, 0x74, 0x79, 0x70,
            0, 0, 0, 0, 0, 0, 0, 24,
            0x71, 0x74, 0x20, 0x20, 0, 0, 0, 0,
        ])

        XCTAssertEqual(
            try MemoryBackedVideoAsset.validatedContentType(normalFtyp),
            AVFileType.mov.rawValue)
        XCTAssertEqual(
            try MemoryBackedVideoAsset.validatedContentType(extendedFtyp),
            AVFileType.mov.rawValue)
    }

    func testLegacyQuickTimeAtomLayoutsWithoutFtypAreRecognizedAsMovies() throws {
        let leadingMoov = Data([
            0, 0, 0, 16, 0x6d, 0x6f, 0x6f, 0x76,
            0, 0, 0, 8, 0x6d, 0x76, 0x68, 0x64,
            0, 0, 0, 8, 0x6d, 0x64, 0x61, 0x74,
        ])
        let mediaBeforeMetadata = Data([
            0, 0, 0, 8, 0x6d, 0x64, 0x61, 0x74,
            0, 0, 0, 16, 0x6d, 0x6f, 0x6f, 0x76,
            0, 0, 0, 8, 0x6d, 0x76, 0x68, 0x64,
        ])

        XCTAssertEqual(
            try MemoryBackedVideoAsset.validatedContentType(leadingMoov),
            AVFileType.mov.rawValue)
        XCTAssertEqual(
            try MemoryBackedVideoAsset.validatedContentType(mediaBeforeMetadata),
            AVFileType.mov.rawValue)
    }

    func testLegacyQuickTimeValidationRejectsIncompleteOrOverflowingAtoms() throws {
        let missingMediaData = Data([
            0, 0, 0, 16, 0x6d, 0x6f, 0x6f, 0x76,
            0, 0, 0, 8, 0x6d, 0x76, 0x68, 0x64,
        ])
        let truncatedAtom = Data([
            0, 0, 0, 24, 0x6d, 0x6f, 0x6f, 0x76,
            0, 0, 0, 8, 0x6d, 0x64, 0x61, 0x74,
        ])
        let overflowingExtendedAtom = Data([
            0, 0, 0, 1, 0x6d, 0x6f, 0x6f, 0x76,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0, 0, 0, 8, 0x6d, 0x64, 0x61, 0x74,
        ])

        for malformed in [missingMediaData, truncatedAtom, overflowingExtendedAtom] {
            XCTAssertThrowsError(try MemoryBackedVideoAsset.validatedContentType(malformed)) {
                error in
                XCTAssertEqual(error as? MemoryBackedVideoAssetError, .invalidContainer)
            }
        }
    }

    func testMemoryBackedMP4RejectsMalformedContainers() throws {
        XCTAssertThrowsError(try MemoryBackedVideoAsset(videoData: Data())) { error in
            XCTAssertEqual(error as? MemoryBackedVideoAssetError, .emptyData)
        }
        XCTAssertThrowsError(
            try MemoryBackedVideoAsset(videoData: Data("not an mp4".utf8))
        ) { error in
            XCTAssertEqual(error as? MemoryBackedVideoAssetError, .invalidContainer)
        }
    }

    func testMemoryBackedMP4RangeValidation() throws {
        XCTAssertEqual(
            try MemoryBackedVideoAsset.responseRange(
                resourceLength: 100, requestedOffset: 0, requestedLength: 20,
                currentOffset: 0, requestsAllDataToEnd: false),
            0 ..< 20)
        XCTAssertEqual(
            try MemoryBackedVideoAsset.responseRange(
                resourceLength: 100, requestedOffset: 0, requestedLength: 20,
                currentOffset: 5, requestsAllDataToEnd: false),
            5 ..< 20)
        XCTAssertEqual(
            try MemoryBackedVideoAsset.responseRange(
                resourceLength: 100, requestedOffset: 90, requestedLength: 40,
                currentOffset: 90, requestsAllDataToEnd: false),
            90 ..< 100)
        XCTAssertEqual(
            try MemoryBackedVideoAsset.responseRange(
                resourceLength: 100, requestedOffset: 25, requestedLength: 0,
                currentOffset: 25, requestsAllDataToEnd: true),
            25 ..< 100)

        XCTAssertThrowsError(
            try MemoryBackedVideoAsset.responseRange(
                resourceLength: 100, requestedOffset: -1, requestedLength: 1,
                currentOffset: 0, requestsAllDataToEnd: false))
        XCTAssertThrowsError(
            try MemoryBackedVideoAsset.responseRange(
                resourceLength: 100, requestedOffset: 101, requestedLength: 1,
                currentOffset: 101, requestsAllDataToEnd: false))
        XCTAssertThrowsError(
            try MemoryBackedVideoAsset.responseRange(
                resourceLength: 100, requestedOffset: Int64.max, requestedLength: 1,
                currentOffset: Int64.max, requestsAllDataToEnd: false))
    }

    func testCancelledMemoryBackedMP4StopsBeforeValidationOrResourceLoading() async throws {
        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }
        let bytes = try Data(contentsOf: fileURL)
        let owner = try MemoryBackedVideoAsset(videoData: bytes)
        let task = Task<Void, Error> {
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try await MediaProcessing.asProcessedSequence(
                .memoryBacked(owner), samplesPerSecond: 1)
        }
        do {
            try await task.value
            XCTFail("Expected CancellationError")
        } catch is CancellationError {
            XCTAssertEqual(owner.resourceRequestCount, 0)
        }
    }

    func testCancellationDuringSuspendedDurationLoadPropagatesCancellationError() async throws {
        let loader = SuspendedDurationResourceLoader()
        defer { withExtendedLifetime(loader) {} }
        let asset = loader.makeAsset()
        let processingTask = Task<Void, Error> {
            _ = try await MediaProcessing.asProcessedSequence(
                asset, maxFrames: 1, targetFPS: { _ in 1 })
        }
        defer {
            processingTask.cancel()
            loader.failPendingRequests()
        }

        // A custom-scheme range request can only be pending after MediaProcessing
        // has passed its entry cancellation check and suspended in asset.load(.duration).
        try await loader.waitForPendingRequest()
        XCTAssertFalse(processingTask.isCancelled)
        XCTAssertGreaterThan(loader.pendingRequestCount, 0)

        processingTask.cancel()
        loader.failPendingRequests()

        do {
            _ = try await processingTask.value
            XCTFail("Expected CancellationError")
        } catch is CancellationError {
            // The former `try?` path discarded this cancellation and replaced it
            // with MediaProcessing's generic duration-loading NSError.
        } catch {
            XCTFail("Expected CancellationError, got \(error)")
        }
    }

    func testFinalFrameCancellationIsObservedBeforeArrayConversion() async throws {
        let image = CIImage(color: .red).cropped(
            to: CGRect(x: 0, y: 0, width: 2, height: 2))
        let task = Task<Void, Error> {
            _ = try await MediaProcessing.asProcessedSequence(
                .frames([VideoFrame(frame: image, timeStamp: .zero)]),
                samplesPerSecond: 1
            ) { frame in
                withUnsafeCurrentTask { $0?.cancel() }
                return frame
            }
        }

        do {
            try await task.value
            XCTFail("Expected CancellationError")
        } catch is CancellationError {
            // The cancellation occurs in the final frame callback and must be
            // observed before the synchronous CIImage-to-MLX conversion begins.
        }
    }

    func testCancellationPropagatesWhenAnAssetProducesNoFrames() async throws {
        let emptyAsset = AVMutableComposition()
        let task = Task<Void, Error> {
            _ = try await MediaProcessing.asProcessedSequence(
                emptyAsset, maxFrames: 1,
                targetFPS: { _ in
                    withUnsafeCurrentTask { $0?.cancel() }
                    return 0
                })
        }

        do {
            try await task.value
            XCTFail("Expected CancellationError")
        } catch is CancellationError {
            // The empty asset has no image results, so cancellation must be
            // checked after the generator sequence reaches its terminal boundary.
        }
    }

    private func memoryVideoTempFiles() throws -> Set<String> {
        let fileManager = FileManager.default
        return Set(
            try fileManager.contentsOfDirectory(atPath: fileManager.temporaryDirectory.path)
                .filter {
                    $0.hasPrefix("vlm-") && $0.hasSuffix(".mp4")
                        || $0.hasPrefix("darkbloom-memory-video")
                })
    }

    func testVideoFileValidationThisShouldFail() async throws {
        guard let fileURL = Bundle.module.url(forResource: "audio_only", withExtension: "mov")
        else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }

        let video = UserInput.Video.url(fileURL)

        do {
            let _ = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 1)
        } catch {
            XCTAssertEqual(error as? VLMError, VLMError.noVideoTrackFound)
        }
    }

    func testVideoFileAsProcessedSequence() async throws {
        // Bogus preprocessing values
        func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
            image
                .toSRGB()
                .resampled(to: resizedSize, method: .bicubic)
                .normalized(mean: (0.1, 0.2, 0.3), std: (0.4, 0.5, 0.6))
        }

        guard let fileURL = Bundle.module.url(forResource: "1080p_30", withExtension: "mov") else {
            XCTFail("Missing file: 1080p_30.mov")
            return
        }

        let video = UserInput.Video.url(fileURL)

        // We know video is exactly 5 seconds long, expect 10 samples
        let frames = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 2) {
            frame in
            let image = preprocess(image: frame.frame, resizedSize: .init(width: 224, height: 224))

            return VideoFrame.init(frame: image, timeStamp: frame.timeStamp)
        }

        XCTAssert(frames.frames.count == 10)
        XCTAssert(frames.frames[0].shape == [1, 3, 224, 224])
    }

    func testVideoFramesAsProcessedSequence() async throws {
        // a function to make a set of frames from images
        func imageWithColor(_ color: CIColor) -> CIImage {
            let inputFilter = CIFilter(name: "CIConstantColorGenerator")!
            inputFilter.setValue(color, forKey: "inputColor")
            return inputFilter.outputImage!.cropped(
                to: CGRect(x: 0, y: 0, width: 1920, height: 1080))
        }

        let colors: [CIColor] = [
            .red, .green, .blue, .cyan, .magenta, .yellow, .white, .black, .gray, .clear,
        ]

        let seconds = 5
        let framerate = 30
        var rawFrames: [VideoFrame] = []

        for i in 0 ..< (seconds * framerate) {
            let image = imageWithColor(colors.randomElement()!)
            let timeStamp: CMTime = .init(value: Int64(i), timescale: Int32(framerate))
            rawFrames.append(VideoFrame(frame: image, timeStamp: timeStamp))
        }

        // Bogus preprocessing values
        func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
            image
                .toSRGB()
                .resampled(to: resizedSize, method: .bicubic)
                .normalized(mean: (0.1, 0.2, 0.3), std: (0.4, 0.5, 0.6))
        }

        let video = UserInput.Video.frames(rawFrames)

        // We know video is exactly 5 seconds long, expect 10 samples
        let frames = try await MediaProcessing.asProcessedSequence(video, samplesPerSecond: 2) {
            frame in
            let image = preprocess(image: frame.frame, resizedSize: .init(width: 224, height: 224))

            return VideoFrame.init(frame: image, timeStamp: frame.timeStamp)
        }

        XCTAssert(frames.frames.count == 10)
        XCTAssert(frames.frames[0].shape == [1, 3, 224, 224])
    }
}
