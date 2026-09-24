import MLX

@_spi(Diagnostics)
public struct Gemma4Layer0ProjectionDiagnostic {
    public enum Failure: Error {
        case invalidTokenPair, unsupportedLayer, expectedQuantizedProjections
    }

    public struct Quantization {
        public let groupSize: Int
        public let bits: Int
        public let mode: String

        init(groupSize: Int, bits: Int, mode: String) {
            self.groupSize = groupSize
            self.bits = bits
            self.mode = mode
        }
    }

    public let tensors: [String: MLXArray]
    public let parameters: [String: MLXArray]
    public let quantization: [String: Quantization]

    init(tensors: [String: MLXArray], parameters: [String: MLXArray],
         quantization: [String: Quantization]) {
        self.tensors = tensors
        self.parameters = parameters
        self.quantization = quantization
    }
}
