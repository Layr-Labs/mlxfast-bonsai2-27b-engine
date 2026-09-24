#!/usr/bin/env python3
"""Build exact-length, non-repeating Python coding prompts for the Qwen campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer


TARGETS = (1_024, 16_384, 32_768, 65_536)
HEADER = """You are reviewing a real excerpt from the CPython 3.12 standard library.
Use the code as realistic repository context for the coding task at the end.

<python_repository_context>
"""
TASK = """
</python_repository_context>

Implement a Python 3.12 function `merge_intervals(intervals)` that accepts an
iterable of integer `(start, end)` pairs and returns a new list of disjoint
intervals sorted by start. Reject pairs whose start exceeds end with
`ValueError`. Merge touching intervals as well as overlapping ones, do not
mutate the input, and handle an empty iterable. Include type hints, a concise
docstring, and `pytest` tests covering empty input, unsorted input, nested
intervals, touching boundaries, negative values, duplicate intervals, and
invalid pairs. Explain the time and space complexity after the code.
"""
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n"


def corpus(stdlib: Path) -> str:
    chunks: list[str] = []
    total = 0
    for path in sorted(stdlib.rglob("*.py")):
        relative = path.relative_to(stdlib)
        if "test" in relative.parts or "site-packages" in relative.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        chunk = f"\n# file: Lib/{relative.as_posix()}\n{source}\n"
        chunks.append(chunk)
        total += len(chunk)
        if total >= 1_200_000:
            break
    return "".join(chunks)


def rendered_tokens(tokenizer: Tokenizer, content: str) -> list[int]:
    return tokenizer.encode(CHAT_PREFIX + content.strip() + CHAT_SUFFIX).ids


def exact_prompt(tokenizer: Tokenizer, source: str, target: int) -> str:
    source_ids = tokenizer.encode(source).ids
    low, high = 0, len(source_ids)
    best: tuple[int, str] | None = None
    while low <= high:
        middle = (low + high) // 2
        excerpt = tokenizer.decode(source_ids[:middle])
        prompt = HEADER + excerpt + TASK
        count = len(rendered_tokens(tokenizer, prompt))
        if count <= target and (best is None or count > best[0]):
            best = (count, prompt)
        if count < target:
            low = middle + 1
        elif count > target:
            high = middle - 1
        else:
            return prompt

    # BPE boundary merges can skip a count. A short, ordinary Python comment
    # closes the final gap while leaving the repository excerpt non-repeating.
    assert best is not None
    count, prompt = best
    for padding in range(1, 512):
        candidate = prompt.removesuffix(TASK) + ("\n# context boundary" * padding) + TASK
        if len(rendered_tokens(tokenizer, candidate)) == target:
            return candidate
    raise RuntimeError(f"could not construct exact {target}-token prompt (nearest {count})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--stdlib", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    source = corpus(args.stdlib)
    if len(tokenizer.encode(source).ids) < max(TARGETS):
        raise RuntimeError("Python source corpus is too small for the 64K prompt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        prompt = exact_prompt(tokenizer, source, target)
        path = args.output_dir / f"python-coding-context-{target}.txt"
        path.write_text(prompt, encoding="utf-8")
        actual = len(rendered_tokens(tokenizer, prompt))
        print(f"{path}: {actual} tokens, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
