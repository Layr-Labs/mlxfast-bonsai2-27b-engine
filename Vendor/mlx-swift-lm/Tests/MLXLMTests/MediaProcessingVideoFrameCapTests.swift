import AVFoundation
import CoreImage
import MLXLMCommon
import MLXVLM
import Testing

@Test func predecodedVideoHonorsMaximumFrameCount() async throws {
    let image = CIImage(color: .red).cropped(
        to: CGRect(x: 0, y: 0, width: 2, height: 2))
    let frames = (0 ..< 20).map {
        UserInput.VideoFrame(
            frame: image,
            timeStamp: CMTime(value: CMTimeValue($0), timescale: 1))
    }

    let processed = try await MediaProcessing.asProcessedSequence(
        .frames(frames),
        targetFPS: { _ in 30 },
        maxFrames: 8)

    #expect(processed.frames.count == 8)
    #expect(processed.timestamps.count == 8)
    #expect(processed.timestamps.first == .zero)
    #expect(processed.timestamps.last == CMTime(value: 19, timescale: 1))
}
