import MLX
import MLXLMCommon
import MLXNN

extension Qwen35TextModel: CBv2CompleteCheckpointKVTypeProviding {
    /// The packed integer weight of a quantized embedding is not its output
    /// dtype. Affine dequantization promotes scales and biases; requiring the
    /// same floating dtype keeps this metadata-only capability exact without
    /// constructing an embedding graph or enabling unvalidated formats.
    var cbv2CheckpointActivationDType: DType? {
        let dtype: DType
        if let embedding = model.embedTokens as? HadamardQuantizedEmbedding {
            guard let biases = embedding.biases, embedding.scales.dtype == biases.dtype else { return nil }
            // The published Prism pack keeps FP32 normalizers. RMSNorm promotes
            // the FP16 embedding output before projections, so native KV and
            // recurrent convolution rows are FP32, not the packed scale dtype.
            guard model.cbv2UniformLayerNormDType == .float32 else { return nil }
            dtype = .float32
        } else if let embedding = model.embedTokens as? QuantizedEmbedding {
            guard embedding.mode == .affine, let biases = embedding.biases,
                embedding.scales.dtype == biases.dtype
            else { return nil }
            dtype = embedding.scales.dtype
        } else {
            dtype = model.embedTokens.weight.dtype
        }
        guard dtype == .float16 || dtype == .bfloat16 || dtype == .float32 else { return nil }
        return dtype
    }

    public var cbv2CompleteCheckpointKVDTypes: [DType]? {
        guard let dtype = cbv2CheckpointActivationDType else { return nil }
        // Native Qwen projections and rotary application preserve the loaded
        // embedding activation dtype for every compact attention-cache row.
        return Array(repeating: dtype, count: configuration.cbv2LayerKinds.count)
    }
}

extension Qwen35Model: CBv2CompleteCheckpointKVTypeProviding {
    public var cbv2CompleteCheckpointKVDTypes: [DType]? {
        languageModel.cbv2CompleteCheckpointKVDTypes
    }
}
