import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum GPTOSSCompiledExpertsPolicy {
    static let enabled = resolve(
        ProcessInfo.processInfo.environment["DARKBLOOM_GPTOSS_COMPILED_EXPERTS"],
        globalCompile: ProcessInfo.processInfo.environment["MLX_COMPILED_DECODE"])

    static func resolve(_ raw: String?, globalCompile: String?) -> Bool {
        raw == "1" && !["0", "false", "no", "off"].contains(globalCompile?.lowercased() ?? "")
    }

    static func eligible(inputDims: Int, hiddenDims: Int, experts: Int,
                         shape: [Int], indicesShape: [Int], dtype: DType) -> Bool {
        guard inputDims == 2880, hiddenDims == 2880, experts == 32,
              shape.count == 3, [1, 2, 4].contains(shape[0]),
              shape[1] == 1, shape[2] == inputDims,
              indicesShape == [shape[0], 1, 4] else { return false }
        return dtype == .float32 || dtype == .bfloat16
    }
}

/// `compile` owns its Updatable inputs. A weak owner avoids the cycle
/// module -> compiled closure -> compiled inputs -> module.
private final class GPTOSSWeakExpertState: Updatable {
    weak var owner: SwiGLUSwitchGLU?
    init(_ owner: SwiGLUSwitchGLU) { self.owner = owner }
    func innerState() -> [MLXArray] {
        guard let owner else { return [] }
        return owner.innerState()
    }
}

/// Opaque non-Module holder: Module.items must not snapshot the cached closure.
final class GPTOSSCompiledExpertCache {
    private let lock = NSLock()
    private var cached: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    func callAsFunction(_ owner: SwiGLUSwitchGLU, _ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let forward = lock.withLock {
            if let cached { return cached }
            let state = GPTOSSWeakExpertState(owner)
            let created = compile(inputs: [state], shapeless: false) {
                [weak owner] (input: MLXArray, indices: MLXArray) -> MLXArray in
                guard let owner else {
                    preconditionFailure("Compiled GPTOSS experts outlived their owner")
                }
                return owner.uncompiledExpertForward(input, indices)
            }
            cached = created
            return created
        }
        // MLX acquires its eval/compile locks outside the holder lock.
        if CBv2ForwardShapeObservation.isActive {
            return CBv2ForwardShapeObservation.compiledComponent(.gptossExperts,
                physicalRows: x.dim(0)) { forward(x, indices) }
        }
        return forward(x, indices)
    }

    func clear() { lock.withLock { cached = nil } }
}

extension SwiGLUSwitchGLU {
    /// Internal entry also exercises small synthetic modules in unit tests.
    func compiledExpertForward(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        compiledExpertCache(self, x, indices)
    }
}
