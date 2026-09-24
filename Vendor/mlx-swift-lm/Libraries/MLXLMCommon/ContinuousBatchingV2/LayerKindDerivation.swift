// LayerKindDerivation.swift
//
// ContinuousBatchingV2 — WS-F (model adaptation).
//
// Family-INDEPENDENT derivation helpers for `[CBv2LayerKind]` (model
// attention structure as data, invariant 11 in report 10). The functions
// here take raw config values so they can live in MLXLMCommon without
// importing the model modules.
//
// Per-family derivations live with the family that owns them — in the model
// file whose constructors they mirror, and from there in that family's
// runner (`Libraries/MLXRunners`). The engine directory keeps no family
// name: a derivation that must track one family's constructor field for
// field belongs next to that constructor, not here, where a change to it is
// invisible.

import Foundation

public enum CBv2LayerKindDerivation {

    /// Canonical `layer_types` strings (HF config vocabulary).
    public static let slidingAttentionType = "sliding_attention"
    public static let fullAttentionType = "full_attention"

    /// Derive `layer_types` from a repeating sliding-window pattern:
    /// `(slidingWindowPattern - 1)` sliding layers followed by one full
    /// layer, repeated and truncated to `numLayers`.
    public static func layerTypes(slidingWindowPattern: Int, numLayers: Int) -> [String] {
        precondition(slidingWindowPattern > 0, "slidingWindowPattern must be positive")
        var pattern = [String]()
        for i in 0 ..< slidingWindowPattern {
            pattern.append(
                i == slidingWindowPattern - 1 ? fullAttentionType : slidingAttentionType)
        }
        var types = [String]()
        while types.count < numLayers {
            types.append(contentsOf: pattern)
        }
        return Array(types.prefix(numLayers))
    }
}
