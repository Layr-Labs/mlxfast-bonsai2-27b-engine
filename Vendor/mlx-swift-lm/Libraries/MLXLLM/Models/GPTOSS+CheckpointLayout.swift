import MLX
import MLXLMCommon
import MLXNN

extension GPTOSSModel {
    /// Saved module parameters use contiguous gate/up halves, unlike the
    /// interleaved upstream `_blocks` format. Normalize this namespace first
    /// so reloading and the fusion rollback share the validated split path.
    func splitSavedGateUpWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        savedFusedExpertPaths.removeAll()
        var result = weights
        let marker = ".experts.gate_up_proj."
        var paths = Set<String>()
        for (key, value) in weights {
            guard let range = key.range(of: marker) else { continue }
            let suffix = String(key[range.upperBound...])
            guard ["weight", "scales", "biases", "bias"].contains(suffix) else { continue }
            let bias = suffix == "bias"
            guard value.ndim == (bias ? 2 : 3) else { continue }
            let axis = bias ? -1 : -2
            guard value.dim(axis) > 0, value.dim(axis) % 2 == 0 else { continue }
            let half = value.dim(axis) / 2
            let base = String(key[..<range.lowerBound]) + ".experts."
            let gate = bias ? value[.ellipsis, ..<half] : value[.ellipsis, ..<half, 0...]
            let up = bias ? value[.ellipsis, half...] : value[.ellipsis, half..., 0...]
            result[base + "gate_proj." + suffix] = contiguous(gate)
            result[base + "up_proj." + suffix] = contiguous(up)
            result.removeValue(forKey: key)
            paths.insert(String(base.dropLast()))
        }
        savedFusedExpertPaths = paths
        for path in paths { setExpertGateUpLayout(at: path, fused: false) }
        return result
    }

    func setExpertGateUpLayout(at path: String, fused: Bool) {
        let parentPath = String(path.dropLast(".experts".count))
        guard let owner = namedModules().first(where: { $0.0 == parentPath })?.1 as? MLPBlock,
              owner.experts.hasFusedGateUp != fused else { return }
        // A direct owner update avoids sparse layers[N] module arrays.
        owner.update(modules: ModuleChildren.unflattened([
            ("experts", owner.experts.withGateUpLayout(fused: fused))
        ]))
    }
}
