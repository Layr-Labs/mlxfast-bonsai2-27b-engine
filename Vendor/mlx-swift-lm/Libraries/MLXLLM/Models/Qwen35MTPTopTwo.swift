// Copyright © 2026 Eigen Labs.

import MLX
import MLXLMCommon

/// Exact top-2 token ids and logit values for every row of `[1, rows, vocab]`.
///
/// The Qwen policy entry point keeps its existing shape and lazy reduction and
/// forwards to the shared CBv2 reduction. It is `public` because the Qwen 3.8
/// Flash-Next model files call it from an editable copy outside this fork.
public func qwen35MTPTopTwoRows(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
    cbv2TopTwoRows(logits)
}
