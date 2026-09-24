import Foundation
@_spi(Diagnostics) import MLXLMCommon

// No model, key, provider or network operations. Output is one JSON report.
var configuration = PagedDecodeProfiler.SegmentMetadataConfiguration()
let arguments = Array(CommandLine.arguments.dropFirst())
do {
    if arguments == ["--help"] {
        print("BenchSegmentedDecode [--owners 1...10] [--offset 2...8192] [--warmup 1...16] [--steps 1...128] [--repetitions 1...3]")
    } else {
        guard arguments.count % 2 == 0 else {
            throw PagedDecodeProfiler.SegmentMetadataError.invalidConfiguration
        }
        var seen = Set<String>()
        for index in stride(from: 0, to: arguments.count, by: 2) {
            let option = arguments[index]
            guard seen.insert(option).inserted, let value = Int(arguments[index + 1]) else {
                throw PagedDecodeProfiler.SegmentMetadataError.invalidConfiguration
            }
            switch option {
            case "--owners": configuration.owners = value
            case "--offset": configuration.initialOffset = value
            case "--warmup": configuration.warmup = value
            case "--steps": configuration.steps = value
            case "--repetitions": configuration.repetitions = value
            default: throw PagedDecodeProfiler.SegmentMetadataError.invalidConfiguration
            }
        }
        let report = try PagedDecodeProfiler.measureSegmentMetadata(configuration)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        FileHandle.standardOutput.write(try encoder.encode(report))
        print("")
    }
} catch {
    FileHandle.standardError.write(Data("BenchSegmentedDecode refused: \(error)\n".utf8))
    exit(1)
}
