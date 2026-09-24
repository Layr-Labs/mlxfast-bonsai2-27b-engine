// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "mlxfast-challenge-dev",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "mlxfast-swift", targets: ["MLXFastCLI"]),
        .library(name: "MLXFastCore", targets: ["MLXFastCore"]),
        .library(name: "MLXFastTransform", targets: ["MLXFastTransform"]),
        .library(name: "MLXFastHarness", targets: ["MLXFastHarness"]),
    ],
    dependencies: [
        // THE ENGINE IS THE FORK. Vendor/mlx-swift-lm is a VENDORED TREE
        // cut from Layr-Labs/mlx-swift-lm feat/qwen38-flash-next-runner
        // 859f4d97e0d1ae9cce0883b044f764283841edfb with fork main 0607011
        // merged in; the merge, the build fixes and the Bonsai 2 support
        // files taken from fork main fd0eaac (pull requests 154 and 155) are
        // this repository's commits, and the tree is edited
        // here, never upstream. It carries the
        // model ports, the MLXRunners scaffold and the generic
        // `bench-worker` Engine Protocol v1 server. benchd spawns that
        // binary; this package builds it as a dependency product.
        //
        // It is vendored the way Vendor/mlx-swift is, and for the same
        // reason: its Bonsai 2 and Qwen 3.5 model files are an editable
        // surface, and an editable path cannot be a submodule. An
        // editablePaths entry names bytes in this tree; a gitlink names a
        // commit. The fork was a submodule at 449f2d0 until that ruling.
        // The tree is edited HERE; nothing goes upstream. A fresh cut from a
        // new upstream base is a deliberate whole-directory replacement --
        // docs/new-track-repo-procedure.md section 6 holds the command.
        // See docs/bonsai2-27b-port-notes.md section 3.4.
        //
        // mlx-swift is UNCHANGED at Layr-Labs/mlx-swift 70052b2 (MLX
        // 0.32.2), with its nested submodules copied in as plain files:
        // Source/Cmlx/mlx at 3fa8f25e and Source/Cmlx/mlx-c at 02cf6f4d.
        // 70052b2 adds Source/MLXNN/Hadamard.swift, which holds the packed
        // signed-Hadamard layers the Bonsai 2 pack loads into.
        // It stays a VENDORED TREE, because its Metal kernel sources are
        // the track's optimization surface, and an editable path cannot be
        // a submodule. 70052b2 is the MLX core that Darkbloom
        // (Layr-Labs/d-inference) builds the same fork code against. The
        // fork's Package.swift path-depends on a sibling ../mlx-swift when
        // one exists, so one runner must have ONE core: a different
        // vendored commit here would compile the fork against a core its
        // own repository never tested.
        // The fork declares its own mlx-swift dependency as a floating
        // `branch: "main"` URL; SwiftPM resolves a ROOT path dependency of
        // the same package identity ahead of it, so every target in the
        // graph -- the fork's included -- builds against Vendor/mlx-swift.
        // SwiftPM reports that override as a "conflicting identity"
        // warning. See docs/bonsai2-27b-port-notes.md section 3.4.
        .package(path: "Vendor/mlx-swift"),
        .package(path: "Vendor/mlx-swift-lm"),
        // The resolved dependency graph is frozen. In the engine repository
        // the enforcement that survives is the one that lives here: setup.sh
        // refuses to build over a Package.swift/Package.resolved that differs
        // from the committed state, and every build and resolve passes
        // --force-resolved-versions so SwiftPM fails closed instead of
        // silently re-resolving. The byte-verification of this manifest
        // against a trusted reference runs from .github/scripts/, which THIS
        // tree carries: submission-static-review-checks.sh is the gate, and
        // the roster it verifies includes this file.
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .target(name: "MLXFastCore"),
        .target(
            name: "MLXFastTransform",
            dependencies: ["MLXFastCore"]
        ),
        // The trusted-harness source scope is this manifest,
        // Package.resolved, Sources/MLXFastCLI, Sources/MLXFastTrustedHarness
        // and Sources/MLXFastCore. It is declared here so a submission cannot
        // expand or repoint the targets feeding the trusted binary without
        // that showing up as a manifest diff. The byte-verification of this
        // scope against trusted git content ran from .github/scripts/, which
        // this engine repository no longer carries; enforcing it is the
        // grading pipeline's job.
        .target(
            name: "MLXFastHarness",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXFastTrustedHarness",
            swiftSettings: [
                .define("MLXFAST_TRUSTED_HARNESS")
            ]
        ),
        .executableTarget(
            name: "MLXFastCLI",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastHarness",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "MLXFastTests",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastHarness",
            ]
        ),
    ]
)
