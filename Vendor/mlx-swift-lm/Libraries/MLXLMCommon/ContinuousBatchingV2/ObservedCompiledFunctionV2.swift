import MLX

/// Wrap the invocation, not the traced body. The observer stores scalar axes
/// only; the original compiled function keeps its existing ownership/lifetime.
public func cbv2ObservedCompiled(
    _ component: CBv2CompiledComponent,
    _ function: @escaping @Sendable (MLXArray) -> MLXArray
) -> @Sendable (MLXArray) -> MLXArray {
    { input in
        guard CBv2ForwardShapeObservation.isActive else { return function(input) }
        return CBv2ForwardShapeObservation.compiledComponent(component,
            physicalRows: input.ndim > 0 ? input.dim(0) : 1) { function(input) }
    }
}

public func cbv2ObservedCompiled(
    _ component: CBv2CompiledComponent,
    _ function: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray
) -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    { input, other in
        guard CBv2ForwardShapeObservation.isActive else { return function(input, other) }
        return CBv2ForwardShapeObservation.compiledComponent(component,
            physicalRows: input.ndim > 0 ? input.dim(0) : 1) { function(input, other) }
    }
}
