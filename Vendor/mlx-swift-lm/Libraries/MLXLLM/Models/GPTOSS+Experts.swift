import Foundation
import MLX
import MLXLMCommon
import MLXNN

private func swiglu(_ xLinear: MLXArray, _ xGlu: MLXArray, alpha: Float = 1.702, limit: Float = 7.0)
    -> MLXArray
{
    var xLinear = xLinear
    var xGlu = xGlu
    xGlu = clip(xGlu, max: MLXArray(limit))
    xLinear = clip(xLinear, min: MLXArray(-limit), max: MLXArray(limit))

    let gluScaled = alpha * xGlu
    let sig = sigmoid(gluScaled)

    let outGlu = xGlu * sig
    return outGlu * (xLinear + 1)
}

private let compiledSwiglu: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { xLinear, xGlu in
    swiglu(xLinear, xGlu)
}

class SwiGLUSwitchGLU: Module {
    let compiledExpertCache = GPTOSSCompiledExpertCache()

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        defer { compiledExpertCache.clear() }
        return try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    @discardableResult
    override func update(
        modules: ModuleChildren, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        defer { compiledExpertCache.clear() }
        return try super.update(modules: modules, verify: verify, path: path, modulePath: modulePath)
    }

    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear?
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear?
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear?
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let hasBias: Bool
    var hasFusedGateUp: Bool { gateUpProj != nil }

    init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false,
        fusedGateUp: Bool = false,
        downProjection: SwitchLinear? = nil
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.hasBias = bias

        if fusedGateUp {
            _gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2, numExperts: numExperts, bias: bias)
        } else {
            _gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
            _upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        }
        _downProj.wrappedValue = downProjection ?? SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Structural replacement used only during checkpoint sanitization.
    /// Strict weight loading fills the new gate/up modules immediately after.
    func withGateUpLayout(fused: Bool) -> SwiGLUSwitchGLU {
        SwiGLUSwitchGLU(inputDims: inputDims, hiddenDims: hiddenDims,
                       numExperts: numExperts, bias: hasBias, fusedGateUp: fused,
                       downProjection: downProj)
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        if GPTOSSCompiledExpertsPolicy.enabled,
           GPTOSSCompiledExpertsPolicy.eligible(
               inputDims: inputDims, hiddenDims: hiddenDims, experts: numExperts,
               shape: x.shape, indicesShape: indices.shape, dtype: x.dtype) {
            return compiledExpertForward(x, indices)
        }
        return uncompiledExpertForward(x, indices)
    }

    func uncompiledExpertForward(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp: MLXArray
        let xGate: MLXArray
        if let gateUpProj {
            let projected = gateUpProj(x, idx, sortedIndices: doSort)
            xGate = projected[.ellipsis, ..<hiddenDims]
            xUp = projected[.ellipsis, hiddenDims...]
        } else {
            guard let gateProj, let upProj else {
                preconditionFailure("GPTOSS experts require split or fused gate/up projections")
            }
            xUp = upProj(x, idx, sortedIndices: doSort)
            xGate = gateProj(x, idx, sortedIndices: doSort)
        }
        x = downProj(compiledSwiglu(xUp, xGate), idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return x.squeezed(axis: -2)
    }
}
