import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXLLM
@testable import MLXLMCommon

@Suite("GPTOSS compiled expert graph", .serialized)
struct GPTOSSCompiledExpertsTests {
    @Test func shapeCountersObserveRuntimeInvocationsAfterTraceWarmup() throws {
        let expert = layer()
        let (input, indices) = inputs(batch: 2, dtype: .bfloat16)
        eval(expert.compiledExpertForward(input, indices))
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let before = recorder.snapshot()
        for _ in 0..<2 {
            let step = recorder.beginStep()
            let output = CBv2ForwardShapeObservation.dispatch(step: step, phase: .decode) {
                let call = CBv2ForwardShapeObservation.beginTarget(liveBatchRows: 2, sequenceWidth: 1)
                defer { call?.end() }
                return expert.compiledExpertForward(input, indices)
            }
            step.attach()
            eval(output)
            step.complete()
        }
        let delta = recorder.snapshot().delta(since: before)
        #expect(delta.complete)
        let components = delta.entries.filter { $0.axes.kind == .compiledComponent }
        #expect(components.count == 1)
        #expect(components.first?.axes.component == .gptossExperts)
        #expect(components.first?.axes.liveBatchRows == 2)
        #expect(components.first?.completedCalls == 2)
    }

    private func layer(fused: Bool = false) -> SwiGLUSwitchGLU {
        MLXRandom.seed(0xC0DEC0DE)
        let layer = SwiGLUSwitchGLU(inputDims: 64, hiddenDims: 64,
                                    numExperts: 4, bias: true, fusedGateUp: fused)
        quantize(model: layer, groupSize: 32, bits: 4, mode: .mxfp4)
        eval(layer)
        return layer
    }

    private func inputs(batch: Int, dtype: DType) -> (MLXArray, MLXArray) {
        let input = (MLXRandom.normal([batch, 1, 64]) * 0.25).asType(dtype)
        let ids: [UInt32] = (0..<(batch * 4)).map { UInt32(($0 * 3 + 1) % 4) }
        return (input, MLXArray(ids).reshaped(batch, 1, 4))
    }

    private func assertParity(_ layer: SwiGLUSwitchGLU, _ x: MLXArray, _ ids: MLXArray) {
        let expected = layer.uncompiledExpertForward(x, ids)
        let actual = layer.compiledExpertForward(x, ids)
        eval(expected, actual)
        #expect(actual.dtype == expected.dtype)
        #expect(actual.shape == expected.shape)
        let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        #expect(error.isFinite && error <= 1e-5)
        #expect(argMax(actual, axis: -1).asArray(Int32.self) == argMax(expected, axis: -1).asArray(Int32.self))
    }

    @Test("default off; multi-token prefill, B8, and other geometry stay eager")
    func policy() {
        #expect(!GPTOSSCompiledExpertsPolicy.resolve(nil, globalCompile: nil))
        #expect(GPTOSSCompiledExpertsPolicy.resolve("1", globalCompile: nil))
        #expect(!GPTOSSCompiledExpertsPolicy.resolve("1", globalCompile: "0"))
        for batch in [1, 2, 4] {
            #expect(GPTOSSCompiledExpertsPolicy.eligible(inputDims: 2880, hiddenDims: 2880, experts: 32,
                        shape: [batch, 1, 2880], indicesShape: [batch, 1, 4], dtype: .float32))
        }
        for shape in [[8, 1, 2880], [1, 2, 2880], [1, 1, 64]] {
            #expect(!GPTOSSCompiledExpertsPolicy.eligible(inputDims: 2880, hiddenDims: 2880, experts: 32,
                        shape: shape, indicesShape: [shape[0], shape[1], 4], dtype: .float32))
        }
    }

    @Test("split/fused graph preserves dtype, finite output parity, and parameter tree")
    func parity() {
        for fused in [false, true] {
            let module = layer(fused: fused)
            let names = module.parameters().flattened().map(\.0).sorted()
            for dtype in [DType.bfloat16, .float32] {
                for batch in [1, 2, 4] {
                    let (x, ids) = inputs(batch: batch, dtype: dtype)
                    assertParity(module, x, ids)
                    assertParity(module, x, ids)
                }
            }
            #expect(module.parameters().flattened().map(\.0).sorted() == names)
        }
    }

    @Test("parameter/context updates and supported module-object replacement remain live")
    func mutations() {
        let module = layer()
        let (x, ids) = inputs(batch: 2, dtype: .float32)
        assertParity(module, x, ids)
        let parameters = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        module.update(parameters: ModuleParameters.unflattened([("down_proj.bias", parameters["down_proj.bias"]! + 1)]))
        assertParity(module, x, ids)
        parameters["up_proj.bias"]!._updateInternal(parameters["up_proj.bias"]! + 2)
        assertParity(module, x, ids)
        let other = layer()
        module.update(modules: ModuleChildren.unflattened([("down_proj", other.downProj)]))
        assertParity(module, x, ids)
    }

    @Test("compiled cache does not keep the module alive")
    func lifetime() {
        weak var observed: SwiGLUSwitchGLU?
        do {
            let module = layer()
            observed = module
            let (x, ids) = inputs(batch: 1, dtype: .float32)
            eval(module.compiledExpertForward(x, ids))
        }
        #expect(observed == nil)
    }
}
