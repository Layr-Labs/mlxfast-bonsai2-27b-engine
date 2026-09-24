# Qwen3-VL 30B-A3B MLX Port Design

## Objective

Add first-class `MLXVLM` support for
`lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit`, including strict
loading of the published four-shard MLX checkpoint and verified text and image
generation through the existing VLM interfaces. The existing Qwen3-VL video
processor path is unchanged by this language-MoE port but is not independently
exercised by the final parity gate.

## Evidence and premise

The port is necessary because `VLMTypeRegistry` currently recognizes
`qwen3_vl` but not the checkpoint's `qwen3_vl_moe` model type. The existing
Qwen3-VL implementation already owns the matching processor, vision topology,
DeepStack injection, multimodal RoPE, KV cache, and generation seam. One
numerical vision detail differs: this checkpoint declares
`hidden_act: gelu_pytorch_tanh`, while the existing dense Swift path installs
its historical fast GELU approximation in vision-block MLPs.

The published `config.json` describes 48 standard-attention language layers
with hidden size 2048, 128 routed experts, top-8 routing, expert intermediate
size 768, normalized selected probabilities, and no shared expert. Direct
inspection of the real safetensors headers is more authoritative than the
repository's stale `model.safetensors.index.json`: the four downloadable MLX
shards contain `language_model.model.layers.*.mlp.switch_mlp.{gate_proj,
up_proj,down_proj}.{weight,scales,biases}`. The installed Swift module tree
must match that split W4/g64 layout exactly.

Qwen3.6 35B-A3B is therefore a useful integration reference, but not a model
implementation template. Its hybrid recurrent/full-attention stack, 256
experts, expert width 512, shared expert, shared-expert gate, and optimized
router/reduction geometry do not match this checkpoint.

## Architecture

`Qwen3VLConfiguration.TextConfiguration` gains optional MoE fields matching
the upstream `qwen3_vl_moe` schema. Their dense defaults preserve the existing
4B/8B Qwen3-VL path. Construction validates the complete MoE geometry once;
no model/config checks are added to token execution.

`Qwen3VLLanguage.DecoderLayer` installs one of two immutable module shapes:

- Dense layers retain the current `MLP` with `gate_proj`, `up_proj`, and
  `down_proj`.
- Eligible MoE layers install `SparseMoeBlock`, containing `gate` and a split
  `SwitchGLU` named `switch_mlp`.

The MoE forward pass preserves the reference arithmetic:

1. Apply the router linear projection.
2. Compute precise softmax over all 128 experts.
3. Select the top eight with `argPartition`.
4. Gather selected probabilities and normalize them to sum to one.
5. Run the selected experts through split `SwitchGLU`.
6. Multiply expert outputs by the selected probabilities and sum on the
   expert axis.

The generic `SwitchGLU` profile is deliberate. This port does not install the
Qwen3.6-specific direct reduction, shared-expert path, fused gate/up layout, or
environment-controlled optimized lane.

`Qwen3VLVision.VisionModel` also resolves its activation profile once during
construction. `qwen3_vl_moe` blocks use tanh-approximate GELU, matching the
checkpoint config and the original Hugging Face implementation; patch and
DeepStack mergers retain exact GELU. Dense `qwen3_vl` keeps its existing Swift
profile. No model-type check is added to vision execution.

`VLMTypeRegistry` maps `qwen3_vl_moe` to the existing `Qwen3VL` wrapper. The
processor remains `Qwen3VLProcessor`, as declared by the checkpoint. A
`VLMRegistry` preset exposes the exact LM Studio model identifier.

## Validation

Unit coverage will prove:

- The real model config decodes every required MoE field.
- Dense Qwen3-VL configs still install dense layers.
- MoE layer selection follows `mlp_only_layers` and
  `decoder_sparse_step` at construction.
- The installed parameter paths match the real split `switch_mlp` shard
  layout and do not contain Qwen3.6 shared-expert or fused-gate paths.
- Routed output matches an independent implementation of the reference
  selection, normalization, expert execution, and weighted sum on small
  deterministic tensors.
- `qwen3_vl_moe` resolves through the public model registry.

Real-checkpoint verification will download the exact Hugging Face revision,
strict-load all four shards, and run deterministic greedy text and image
prompts. Swift token sequences will be compared under the same checkpoint,
prompt, media, and generation settings. The current `mlx-vlm 0.6.16`
Qwen3-VL-MoE vision implementation reverses the block and merger GELU variants;
the image reference must correct those installed activations to the checkpoint
and original Hugging Face contract before comparison.

## Failure-mode check

1. **Quantized expert paths fail strict load.** This is critical. Tests assert
   the installed split `switch_mlp` parameter names, and the final gate uses
   the real four-shard checkpoint with strict model update verification.
2. **Adding optional fields changes dense Qwen3-VL behavior.** This is
   critical. Dense defaults select the unchanged `MLP`; focused dense
   construction tests protect module selection, and the dense arithmetic is
   left unchanged.
3. **Text generation works but multimodal vision arithmetic diverges.** This is
   critical for a VLM port. Unit tests lock the checkpoint-specific GELU
   construction, and final verification includes a fixed image and corrected
   reference-token comparison, not only a text-only smoke test.

## Non-goals

- Adding MTP, speculative decoding, or continuous-batching support specific
  to this checkpoint.
- Transplanting Qwen3.6 performance kernels without matched measurements on
  Qwen3-VL's real 128-expert, width-768 shapes.
- Repairing the model repository's stale safetensors index; the normal Swift
  loader enumerates the actual shard files and does not consume that index.
