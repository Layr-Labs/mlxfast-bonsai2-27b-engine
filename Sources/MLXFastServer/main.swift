// Local OpenAI-compatible API for the transformed Bonsai 2 checkpoint.
// Reuse the vendored server's request handling and model loader; the transform
// copies the checkpoint's tokenizer and chat template into weights/.
import Foundation
import MLXLMServer

do {
    var environment = ProcessInfo.processInfo.environment
    // Unlike the generic mlx-server, this entry point serves this track's
    // transformed checkpoint by default. --model and MLX_SERVER_MODEL still
    // override it, as in the upstream CLI.
    if environment["MLX_SERVER_MODEL"] == nil {
        environment["MLX_SERVER_MODEL"] = "weights"
    }
    switch try MLXServerCLI.parse(environment: environment) {
    case .help:
        print(MLXServerCLI.help)
    case .listRoutes:
        let data = try JSONEncoder.openAIServer.encode(MLXServerRoute.manifest)
        print(String(decoding: data, as: UTF8.self))
    case .run(let configuration):
        // The generic server interprets a nonexistent path as a Hub ID. This
        // track's entry point is local-only: never download a model on startup.
        let modelPath = (configuration.model as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw NSError(
                domain: "MLXFastServer", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "Local model directory '\(configuration.model)' not found; run the checkpoint transform first"])
        }
        try await MLXServer.run(configuration: configuration)
    }
} catch {
    FileHandle.standardError.write(Data("mlxfast-server: \(error.localizedDescription)\n".utf8))
    Foundation.exit(1)
}
