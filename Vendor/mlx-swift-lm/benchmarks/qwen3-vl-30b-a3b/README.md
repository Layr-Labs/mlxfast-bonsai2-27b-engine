# Qwen3-VL 30B-A3B compatibility receipts

These receipts validate the Swift port against the exact MLX checkpoint and a
Python `mlx-vlm` reference. They establish strict loading and greedy output
parity; they are not performance benchmarks.

## Artifact

- Model: `lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit`
- Revision: `61c11f42d7bc01e00f5ea7f2e667c0a216f48397`
- Quantization: affine 4-bit, group size 64
- Architecture: 48 attention layers, hidden size 2048, 128 experts, top 8,
  expert intermediate size 768

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `model-00001-of-00004.safetensors` | 5,345,270,475 | `6c5bf3700d411abeee819ec2398da2548726ef3eceda910b06a26d9af1d47415` |
| `model-00002-of-00004.safetensors` | 5,364,684,988 | `8fc999f5fdcada0a7c8b20064467fa3c7a92ff9cd5625a81d985e349de00e7b5` |
| `model-00003-of-00004.safetensors` | 5,274,796,907 | `70c39c70350f6e7f40c717c99fdc186743d2867975c00e42ccaf605a22cd1d83` |
| `model-00004-of-00004.safetensors` | 2,267,351,303 | `cde81617b0edc52eb2c24c02316840c4bc8fefe1239508c9457cd2c10d39d228` |

The repository's `model.safetensors.index.json` describes a stale 13-shard
export. The real snapshot contains the four shards above, and both loaders
enumerate those files when the stale shard names are absent.

## Cases

Text prompt:

```text
Reply with exactly: Swift MoE ready
```

Image prompt:

```text
What color is this image? Reply with just the color name.
```

The image input is a 100x100 RGBA red PNG with SHA-256
`2df3a4b2c1198a2e96087274ba526738fb3fbec3c3780dfbdcc535e2820428c4`.
It is generated locally and is not committed.

Both implementations use greedy decoding (`temperature = 0`) with an
eight-token limit. The temporary Swift validation executable loads the model
through `VLMModelFactory` with strict weight verification and is removed from
the PR after the receipts are captured.

```bash
DARKBLOOM_BF16_WEIGHTS=0 \
  Qwen3VLPortCheck <snapshot-path> both 8 /tmp/qwen3vl-red.png

uv run --with mlx-vlm --with jinja2 \
  python /tmp/qwen3vl_reference.py \
  <snapshot-path> /tmp/qwen3vl-red.png 8
```

For image parity, the Python check corrects `mlx-vlm 0.6.16`'s reversed
Qwen3-VL-MoE vision GELU installation: vision blocks use tanh-approximate GELU
as declared by the checkpoint and original Hugging Face implementation, while
the patch and DeepStack mergers use exact GELU.

Raw case results are in `text-generation.json` and `image-generation.json`.

## 1K / 16K Swift benchmark

This text-only benchmark exercises the language/MoE path, not the vision
tower. It uses the release build on an Apple M5 Max with 128 GB of unified
memory, greedy argmax sampling, exactly 1,024 prepared prompt tokens, and a
forced 16,384-token decode. Stop tokens are deliberately ignored so both runs
measure the same full decode length.

| Run | Input / output tokens | Prefill tok/s | Decode tok/s | Wall time (s) | Output tok/s, end to end | Peak MLX memory (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,024 / 16,384 | 3,991.13 | 60.48 | 271.16 | 60.42 | 18.73 |
| 2 | 1,024 / 16,384 | 3,661.49 | 61.55 | 266.46 | 61.49 | 18.73 |
| Mean | 1,024 / 16,384 | 3,826.31 | 61.02 | 268.81 | 60.95 | 18.73 |

The two long decode results differ by 1.76%. Both produced the same 16,384-token
FNV-1a digest, `e12a32d50e0cf35d`. The source fixture rendered to 15,183 tokens
with the Qwen3-VL chat template; the benchmark preserved the chat prefix and
final 32 tokens while removing the middle to construct the exact 1,024-token
input. Unrounded measurements, prompt provenance, revisions, and the command
are in `generation-1024x16384.json`.
