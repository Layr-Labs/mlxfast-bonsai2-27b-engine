#!/usr/bin/env python3
"""No benchmark-path code calls the Hugging Face Hub client or its cache.

swift-huggingface (GHSA-vvrq-6cfc-v43m) is linked into both benchmark
binaries: swift-transformers' Tokenizers target depends on its Hub target, and
Hub depends on HuggingFace. The pinned 0.11.0 still has path defects in its
cache code. The accepted risk is that no benchmark-path code calls that code.
This lint makes the claim a check.

The benchmark path is every target that is linked into the two benchmark
binaries: mlxfast-swift (target MLXFastCLI of Package.swift) and bench-worker
(target bench-worker of Vendor/mlx-swift-lm/Package.swift). The script reads
both manifests, follows the local target and product dependencies from those
two roots, and scans the source directory of every target it reaches, plus
all of Vendor/mlx-swift/Source (the MLX core, which depends on no Hub code).
A macro target is not linked; it runs in the compiler. Its expansions are
checked at the call sites: the Hub macros are forbidden patterns.

A line fails when its code (comments removed) matches a forbidden pattern.
Files outside the benchmark path that match are listed in ALLOWLIST. The
script fails when an allowlisted file is on the benchmark path, when it no
longer exists, or when it no longer matches, so that the list stays exact.

Usage: python3 tools/ci-hub-reachability-scan.py
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORBIDDEN = [
    re.compile(r"\bHubClient\b"),
    re.compile(r"\bHubCache\b"),
    re.compile(r"\bdownloadSnapshot\b"),
    re.compile(r"\bHubApi\b"),
    re.compile(r"^\s*(@\w+\s+)*import\s+((struct|class|enum|protocol|func|var|let|typealias)\s+)?(HuggingFace|Hub)(\.|\s|$)"),
    re.compile(r"#hubDownloader\b"),
    re.compile(r"#huggingFaceLoadModel"),
    re.compile(r"\bfrom\(\s*pretrained\s*:"),
]

# path -> why it is not on the benchmark path
ALLOWLIST = {
    "Vendor/mlx-swift-lm/Libraries/MLXLMServer/Runtime/MLXEmbeddingServerEngine.swift":
        "target MLXLMServer; only mlx-server and MLXLMServerTests depend on it",
    "Vendor/mlx-swift-lm/Libraries/MLXLMServer/Runtime/MLXModelContainerEngine.swift":
        "target MLXLMServer; only mlx-server and MLXLMServerTests depend on it",
    "Vendor/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/ToolCallIntegrationTests.swift":
        "Xcode project IntegrationTesting.xcodeproj; not a SwiftPM target",
    "Vendor/mlx-swift-lm/Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift":
        "macro target MLXHuggingFaceMacros; runs in the compiler and is not linked",
}

CODE_SUFFIXES = (".swift", ".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm")

ROOTS = [
    ("Package.swift", "MLXFastCLI"),
    ("Vendor/mlx-swift-lm/Package.swift", "bench-worker"),
]
EXTRA_SCAN_DIRS = ["Vendor/mlx-swift/Source"]
LOCAL_PACKAGES = {
    "mlx-swift-lm": "Vendor/mlx-swift-lm/Package.swift",
}


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return [re.sub(r"//.*$", "", line) for line in text.split("\n")]


def balanced(text, start):
    """Return the text from text[start] (an opening bracket) to its match."""
    pairs = {"(": ")", "[": "]"}
    stack = []
    in_string = False
    i = start
    while i < len(text):
        c = text[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c in pairs:
            stack.append(pairs[c])
        elif stack and c == stack[-1]:
            stack.pop()
            if not stack:
                return text[start:i + 1]
        i += 1
    raise SystemExit("ci-hub-reachability-scan: unbalanced brackets in a manifest")


def parse_manifest(relpath):
    """Targets and library products of one manifest."""
    text = "\n".join(strip_comments(open(os.path.join(REPO, relpath), encoding="utf-8").read()))
    package_dir = os.path.dirname(relpath)
    targets = {}
    for match in re.finditer(r"\.(target|executableTarget|macro|testTarget|plugin)\(", text):
        body = balanced(text, match.end() - 1)
        name = re.search(r'name:\s*"([^"]+)"', body)
        if not name:
            continue
        name = name.group(1)
        path = re.search(r'\bpath:\s*"([^"]+)"', body)
        if path:
            source = os.path.join(package_dir, path.group(1))
        else:
            source = os.path.join(package_dir, "Tests" if match.group(1) == "testTarget" else "Sources", name)
        deps_local, deps_products = [], []
        deps = re.search(r"\bdependencies:\s*\[", body)
        if deps:
            deps_text = balanced(body, deps.end() - 1)
            for product in re.finditer(r'\.product\(\s*name:\s*"([^"]+)"\s*,\s*package:\s*"([^"]+)"', deps_text):
                deps_products.append((product.group(1), product.group(2)))
            without_products = re.sub(r"\.product\([^)]*\)", "", deps_text)
            deps_local += re.findall(r'"([^"]+)"', without_products)
        targets[name] = {"kind": match.group(1), "source": os.path.normpath(source), "local": deps_local,
                         "products": deps_products}
    products = {}
    for match in re.finditer(r'\.library\(\s*name:\s*"([^"]+)"\s*,\s*targets:\s*\[([^\]]*)\]', text):
        products[match.group(1)] = re.findall(r'"([^"]+)"', match.group(2))
    return targets, products


def linked_closure():
    manifests = {path: parse_manifest(path) for path in {path for path, _ in ROOTS} | set(LOCAL_PACKAGES.values())}
    reached = {}
    external = set()
    todo = list(ROOTS)
    while todo:
        manifest, name = todo.pop()
        key = (manifest, name)
        if key in reached:
            continue
        targets, _ = manifests[manifest]
        if name not in targets:
            raise SystemExit("ci-hub-reachability-scan: %s has no target %s" % (manifest, name))
        target = targets[name]
        reached[key] = target
        if target["kind"] == "macro":
            # A macro runs in the compiler; its own dependencies are not linked.
            continue
        for dep in target["local"]:
            todo.append((manifest, dep))
        for product, package in target["products"]:
            local = LOCAL_PACKAGES.get(package)
            if package == "mlx-swift":
                continue  # scanned whole, as EXTRA_SCAN_DIRS
            if local:
                for dep in manifests[local][1].get(product, [product]):
                    todo.append((local, dep))
            else:
                external.add(package)
    return reached, external


def matches(path):
    hits = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(strip_comments(handle.read()), 1):
            if any(pattern.search(line) for pattern in FORBIDDEN):
                hits.append((number, line.strip()))
    return hits


def code_files(directory):
    for root, _, files in os.walk(os.path.join(REPO, directory)):
        for name in files:
            if name.endswith(CODE_SUFFIXES):
                yield os.path.relpath(os.path.join(root, name), REPO)


def inside(path, directory):
    return path == directory or path.startswith(directory.rstrip("/") + "/")


def main():
    reached, external = linked_closure()
    linked_dirs = sorted({t["source"] for t in reached.values() if t["kind"] != "macro"})
    macro_dirs = sorted({t["source"] for t in reached.values() if t["kind"] == "macro"})
    scan_dirs = linked_dirs + EXTRA_SCAN_DIRS
    failures = []

    scanned = 0
    for directory in scan_dirs:
        for path in code_files(directory):
            scanned += 1
            for number, line in matches(os.path.join(REPO, path)):
                failures.append("%s:%d: benchmark-path code references the Hub client or cache: %s" % (path, number, line))

    for path, reason in sorted(ALLOWLIST.items()):
        full = os.path.join(REPO, path)
        if not os.path.isfile(full):
            failures.append("allowlist entry %s does not exist; remove it" % path)
            continue
        if any(inside(path, directory) for directory in scan_dirs):
            failures.append("allowlist entry %s is on the benchmark path (%s); it cannot be allowlisted" % (path, reason))
        if not matches(full):
            failures.append("allowlist entry %s no longer matches; remove it" % path)

    # Every other code file in the repository that matches must be allowlisted.
    for directory in ("Sources", "Vendor", "Tests"):
        for path in code_files(directory):
            if path in ALLOWLIST or any(inside(path, d) for d in scan_dirs):
                continue
            if matches(os.path.join(REPO, path)):
                failures.append("%s references the Hub client or cache and is not in the allowlist; "
                                "add it with the reason it is not on the benchmark path" % path)

    if failures:
        for failure in failures:
            print("::error::%s" % failure)
        return 1
    names = sorted(name for (_, name), t in reached.items() if t["kind"] != "macro")
    print("ok: %d code files in %d linked targets (%s) and %s reference no Hub client, cache or download API"
          % (scanned, len(names), ", ".join(names), ", ".join(EXTRA_SCAN_DIRS)))
    print("ok: macro targets not linked: %s; third-party packages linked (not scanned): %s"
          % (", ".join(macro_dirs) or "none", ", ".join(sorted(external))))
    print("ok: %d allowlisted files are off the benchmark path" % len(ALLOWLIST))
    return 0


if __name__ == "__main__":
    sys.exit(main())
