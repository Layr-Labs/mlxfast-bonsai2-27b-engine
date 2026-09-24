// Weight-free tests for the DFlash 2 block drafter.
//
// None of these loads the pinned 3.85 GB drafter. They pin the parts of the
// port a reader cannot check by eye: the config decode against the drafter's
// REAL config.json, the two-tap grouped dynamic convolution against a
// hand-computed case, the candidate selector's greedy path against a case where
// the edge term overrides the unary term at every position, the sliding-window
// mask arithmetic, and the loader's key set against the checkpoint's real
// tensor inventory.
//
// The expected numbers in the convolution and selector cases are hand-computed
// from the reference source, `dflash/model_mlx.py` of `z-lab/dflash` at
// `07ebd93`, because MLX Python does not import on the machine the port was
// written on. Each case states the arithmetic it stands on.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

// Tests/MLXLMTests/<this file> -> the repository root is six levels up
// (Vendor/mlx-swift-lm/Tests/MLXLMTests).
private let dflash2RepositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

private let dflash2ConfigFixtureURL = dflash2RepositoryRoot
    .appendingPathComponent("fixtures/bonsai2_27b_dflash2_drafter_config.json")

private let dflash2InventoryFixtureURL = dflash2RepositoryRoot
    .appendingPathComponent("fixtures/bonsai2_27b_dflash2_drafter_inventory.json")

private func dflash2Configuration() throws -> DFlash2Configuration {
    try JSONDecoder().decode(
        DFlash2Configuration.self, from: Data(contentsOf: dflash2ConfigFixtureURL))
}

// MARK: - The configuration decode

@Test("the drafter's own config.json decodes field for field")
func dflash2ConfigurationDecodesTheShippedFixture() throws {
    let config = try dflash2Configuration()

    #expect(config.architectures == ["DFlash2DraftModel"])
    #expect(config.modelType == "qwen3")
    #expect(config.hiddenSize == 5120)
    #expect(config.hiddenLayers == 5)
    #expect(config.intermediateSize == 17408)
    #expect(config.attentionHeads == 32)
    #expect(config.kvHeads == 8)
    #expect(config.headDim == 128)
    #expect(config.vocabularySize == 248_320)
    #expect(config.rmsNormEps == 1e-6)
    #expect(config.maxPositionEmbeddings == 262_144)
    #expect(config.numTargetLayers == 64)
    #expect(config.slidingWindow == 2048)
    #expect(config.layerTypes == Array(repeating: .slidingAttention, count: 5))

    // A DFlash 2 block is NOT causal inside itself: every block position sees
    // every other one. The reference reads this key and defaults it to the
    // layer's own sliding flag; this config states it.
    #expect(config.isCausal == false)

    // rope_theta is NOT at the top level of this config. It is inside
    // `rope_parameters`, and a decoder that only reads the top-level key would
    // silently run the drafter at base 10000 instead of 1e7.
    #expect(config.ropeTheta == 1e7)

    #expect(config.blockSize == 8)
    #expect(config.draftDepth == 7)
    #expect(config.maskTokenId == 248_070)
    #expect(config.targetLayerIds == [5, 19, 33, 47, 61])
    #expect(config.dflash.convKernelSize == 2)
    #expect(config.dflash.convGroupSize == 16)
    #expect(config.dflash.selectorRank == 256)
    #expect(config.dflash.selectorTopK == 16)

    // The fused target hidden state is the five tapped layers side by side.
    #expect(config.targetHiddenSize == 5 * 5120)

    // The reference carries all three logit knobs and this drafter sets none of
    // them, so the port must default them to no-ops rather than drop them.
    #expect(config.dflash.inputEmbeddingScale == 1)
    #expect(config.dflash.outputMultiplier == 1)
    #expect(config.dflash.finalLogitSoftcapping == nil)
}

@Test("a config whose layer_types do not match num_hidden_layers is refused")
func dflash2ConfigurationRefusesMismatchedLayerTypes() throws {
    let json = """
        {"hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,
         "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
         "vocab_size":32,"rms_norm_eps":1e-6,"max_position_embeddings":128,
         "sliding_window":4,"layer_types":["sliding_attention"],
         "dflash_config":{"block_size":4,"mask_token_id":1,"target_layer_ids":[0],
                          "conv_kernel_size":2,"conv_group_size":2,
                          "selector_rank":2,"selector_top_k":2}}
        """
    #expect(throws: (any Error).self) {
        _ = try JSONDecoder().decode(DFlash2Configuration.self, from: Data(json.utf8))
    }
}

@Test("a sliding config with no sliding_window is refused")
func dflash2ConfigurationRefusesSlidingWithoutWindow() throws {
    let json = """
        {"hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,
         "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
         "vocab_size":32,"rms_norm_eps":1e-6,"max_position_embeddings":128,
         "layer_types":["sliding_attention"],
         "dflash_config":{"block_size":4,"mask_token_id":1,"target_layer_ids":[0],
                          "conv_kernel_size":2,"conv_group_size":2,
                          "selector_rank":2,"selector_top_k":2}}
        """
    #expect(throws: (any Error).self) {
        _ = try JSONDecoder().decode(DFlash2Configuration.self, from: Data(json.utf8))
    }
}

@Test("a conv_group_size that does not divide hidden_size is refused")
func dflash2ConfigurationRefusesRaggedConvGroups() throws {
    let json = """
        {"hidden_size":9,"num_hidden_layers":1,"intermediate_size":16,
         "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
         "vocab_size":32,"rms_norm_eps":1e-6,"max_position_embeddings":128,
         "layer_types":["full_attention"],
         "dflash_config":{"block_size":4,"mask_token_id":1,"target_layer_ids":[0],
                          "conv_kernel_size":2,"conv_group_size":2,
                          "selector_rank":2,"selector_top_k":2}}
        """
    #expect(throws: (any Error).self) {
        _ = try JSONDecoder().decode(DFlash2Configuration.self, from: Data(json.utf8))
    }
}

// MARK: - The grouped dynamic causal convolution

@Test("the grouped dynamic convolution matches a hand-computed case")
func dflash2GroupedDynamicConvolutionMatchesHandComputedCase() {
    // Reference (`_grouped_dynamic_convolve`), written out:
    //
    //   out[l][g*G + c] = sum_o (base[o][g*G + c] + dyn[l][o][g])
    //                           * hidden[l - o][g*G + c]
    //
    // with hidden[l - o] = 0 when l - o < 0. hidden_size 4, group_size 2, so
    // two groups of two channels; kernel_size 2, so two taps; length 3.
    let hidden = MLXArray(
        [Float](
            [
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
            ]), [1, 3, 4])
    let base = MLXArray(
        [Float](
            [
                0.5, -1, 2, 0.25,
                1, 1, -0.5, 2,
            ]), [2, 4])
    // dynamic[l][o][g]
    let dynamic = MLXArray(
        [Float](
            [
                1, 2, 3, 4,
                0, 1, 2, 3,
                -1, 0, 1, -2,
            ]), [1, 3, 2, 2])

    let out = DFlash2GroupedDynamicCausalConv.convolve(
        hidden: hidden, dynamic: dynamic, base: base, groupSize: 2)
    eval(out)

    // l = 0 has no l-1 term:
    //   (0.5+1)*1 = 1.5   (-1+1)*2 = 0   (2+2)*3 = 12   (0.25+2)*4 = 9
    // l = 1:
    //   (0.5+0)*5 + (1+2)*1   = 5.5
    //   (-1+0)*6 + (1+2)*2    = 0
    //   (2+1)*7  + (-0.5+3)*3 = 28.5
    //   (0.25+1)*8 + (2+3)*4  = 30
    // l = 2:
    //   (0.5-1)*9  + (1+1)*5    = 5.5
    //   (-1-1)*10  + (1+1)*6    = -8
    //   (2+0)*11   + (-0.5-2)*7 = 4.5
    //   (0.25+0)*12 + (2-2)*8   = 3
    let expected: [Float] = [
        1.5, 0, 12, 9,
        5.5, 0, 28.5, 30,
        5.5, -8, 4.5, 3,
    ]
    #expect(out.shape == [1, 3, 4])
    let got = out.asArray(Float.self)
    for (index, want) in expected.enumerated() {
        #expect(abs(got[index] - want) < 1e-4, "position \(index)")
    }
}

@Test("the first convolution tap is causal: position zero sees no history")
func dflash2GroupedDynamicConvolutionIsCausal() {
    // A single non-zero position must not reach BACKWARDS. Put the signal at
    // l = 1 and check l = 0 stays zero.
    let hidden = MLXArray([Float]([0, 0, 1, 1]), [1, 2, 2])
    let base = MLXArray([Float]([1, 1, 1, 1]), [2, 2])
    let dynamic = MLXArray([Float]([0, 0, 0, 0]), [1, 2, 2, 1])
    let out = DFlash2GroupedDynamicCausalConv.convolve(
        hidden: hidden, dynamic: dynamic, base: base, groupSize: 2)
    eval(out)
    #expect(out.asArray(Float.self) == [0, 0, 1, 1])
}

// MARK: - The candidate selector

@Test("the selector's greedy path follows the edge term, not the unary term")
func dflash2CandidateSelectorGreedyPath() throws {
    // A five-token vocabulary, rank 2, top-2 candidates, two block positions.
    // At BOTH positions the unary logit prefers one candidate and the low-rank
    // edge score prefers the other, so a port that dropped the edge term would
    // return [1, 0] instead of [3, 2].
    let json = """
        {"hidden_size":2,"num_hidden_layers":1,"intermediate_size":4,
         "num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,
         "vocab_size":5,"rms_norm_eps":1e-6,"max_position_embeddings":16,
         "layer_types":["full_attention"],
         "dflash_config":{"block_size":3,"mask_token_id":1,"target_layer_ids":[0],
                          "conv_kernel_size":2,"conv_group_size":2,
                          "selector_rank":2,"selector_top_k":2}}
        """
    let config = try JSONDecoder().decode(
        DFlash2Configuration.self, from: Data(json.utf8))
    let selector = DFlash2CandidateSelector(config)
    try selector.update(
        parameters: ModuleParameters.unflattened([
            // hidden_projection is the identity, so the projected hidden state
            // IS the hidden state.
            "hidden_projection.weight": MLXArray([Float]([1, 0, 0, 1]), [2, 2]),
            "predecessor_codebook": MLXArray(
                [Float]([1, 0, 0, 1, 1, 1, -1, 0, 0, -1]), [5, 2]),
            "successor_codebook": MLXArray(
                [Float]([2, 0, 0, 2, 1, -1, 3, 1, 0, 0]), [5, 2]),
        ]), verify: [.all])

    let hidden = MLXArray([Float]([1, 0, 1, 1]), [1, 2, 2])
    let logits = MLXArray(
        [Float](
            [
                0, 3, 1, 2, -1,
                5, 0, 4.5, 1, 2,
            ]), [1, 2, 5])
    let anchor = MLXArray([Int32(2)])

    let path = selector.selectGreedy(hidden: hidden, logits: logits, anchor: anchor)
    eval(path)

    // Position 0. Candidates are {1, 3} (logits 3 and 2). The anchor is token 2,
    // whose predecessor row is [1, 1]; the hidden state is [1, 0], so the
    // element-wise product is [1, 0] and each edge score is the candidate's
    // successor row's FIRST component.
    //   token 1: 3 + 0 = 3      token 3: 2 + 3 = 5   -> token 3
    // Position 1. Candidates are {0, 2} (logits 5 and 4.5). The predecessor is
    // now token 3, row [-1, 0]; the hidden state is [1, 1], so the product is
    // [-1, 0] and each edge score is MINUS the successor row's first component.
    //   token 0: 5 - 2 = 3      token 2: 4.5 - 1 = 3.5 -> token 2
    #expect(path.shape == [1, 2])
    #expect(path.asArray(Int32.self) == [3, 2])
}

// MARK: - The sliding-window mask

@Test("the block mask windows the context and leaves the block non-causal")
func dflash2SlidingMaskIsNonCausalInsideTheBlock() {
    // Reference (`DFlashAttention.__call__`):
    //   query   = ctx_len + arange(L)
    //   context = (key < ctx_len) & (query - key < sliding_window)
    //   block   = key >= ctx_len
    //   mask    = context | block
    //
    // ctx_len 3, L 2, sliding_window 4. Block position i can reach context key
    // j only while 3 + i - j < 4, that is j >= i.
    let mask = DFlash2SlidingMask.make(
        contextLength: 3, blockLength: 2, slidingWindow: 4, isCausal: false)
    eval(mask)
    #expect(mask.shape == [2, 5])
    #expect(
        mask.asArray(Bool.self) == [
            true, true, true, true, true,
            false, true, true, true, true,
        ])
}

@Test("a causal block mask also lower-triangulates the block half")
func dflash2SlidingMaskHonoursIsCausal() {
    let mask = DFlash2SlidingMask.make(
        contextLength: 3, blockLength: 2, slidingWindow: 4, isCausal: true)
    eval(mask)
    #expect(
        mask.asArray(Bool.self) == [
            true, true, true, true, false,
            false, true, true, true, true,
        ])
}

@Test("the context skip keeps sliding_window minus one rows")
func dflash2ContextSkipKeepsTheWindow() {
    // The layer drops the oldest context rows and advances the cache offset by
    // the same count, so the rows it keeps stay at the positions they had.
    #expect(DFlash2SlidingMask.contextSkip(contextLength: 5, slidingWindow: 4) == 2)
    #expect(DFlash2SlidingMask.contextSkip(contextLength: 3, slidingWindow: 4) == 0)
    #expect(DFlash2SlidingMask.contextSkip(contextLength: 4, slidingWindow: 4) == 1)
    #expect(DFlash2SlidingMask.contextSkip(contextLength: 0, slidingWindow: 2048) == 0)
}

// MARK: - The strict loader key set

/// One record of `fixtures/bonsai2_27b_dflash2_drafter_inventory.json`.
private struct DFlash2InventoryFixture: Decodable {
    struct Tensor: Decodable {
        let dtype: String
        let shape: [Int]
    }

    struct Shard: Decodable {
        let tensorCount: Int

        enum CodingKeys: String, CodingKey {
            case tensorCount = "tensor_count"
        }
    }

    let shards: [Shard]
    let tensors: [String: Tensor]
}

@Test("the drafter's parameter key set is exactly the checkpoint's tensor set")
func dflash2LoaderKeySetMatchesTheTensorInventory() throws {
    // The loader runs `update(parameters:verify: [.all])`, so a key set that
    // does not match exactly is a refusal rather than a silent partial load.
    // This is the test that catches it WITHOUT the 3.85 GB shard: the module
    // tree is built lazily, and only its parameter NAMES are read.
    let inventory = try JSONDecoder().decode(
        DFlash2InventoryFixture.self, from: Data(contentsOf: dflash2InventoryFixtureURL))
    #expect(inventory.shards.first?.tensorCount == 81)
    #expect(inventory.tensors.count == 81)
    #expect(inventory.tensors.values.allSatisfy { $0.dtype == "BF16" })

    let drafter = DFlash2DraftModel(config: try dflash2Configuration())
    let parameters = Set(drafter.parameters().flattened().map(\.0))
    let tensors = Set(inventory.tensors.keys)

    #expect(parameters == tensors)
    // The drafter owns NO embedding table and NO output projection: it binds
    // the target's, which on this pack are packed Hadamard modules.
    #expect(!parameters.contains { $0.hasPrefix("embed_tokens.") })
    #expect(!parameters.contains { $0.hasPrefix("lm_head.") })
    // The two selector codebooks are RAW parameters in the checkpoint, not
    // `Embedding` sub-modules, so neither carries a `.weight` suffix.
    #expect(parameters.contains("candidate_selector.predecessor_codebook"))
    #expect(parameters.contains("candidate_selector.successor_codebook"))
    #expect(parameters.contains("fc.weight"))
    #expect(parameters.contains("layers.4.mlp_conv.base_kernel"))
}

@Test("the drafter's parameter shapes match the checkpoint's")
func dflash2ParameterShapesMatchTheTensorInventory() throws {
    let inventory = try JSONDecoder().decode(
        DFlash2InventoryFixture.self, from: Data(contentsOf: dflash2InventoryFixtureURL))
    let drafter = DFlash2DraftModel(config: try dflash2Configuration())
    for (name, array) in drafter.parameters().flattened() {
        let tensor = try #require(inventory.tensors[name], "unknown parameter \(name)")
        #expect(array.shape == tensor.shape, "shape of \(name)")
    }
}

@Test("a directory is recognised as DFlash 2 by its config, not its name")
func dflash2DirectoryDetectionReadsTheConfig() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("dflash2-detect-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let configURL = directory.appendingPathComponent("config.json")

    try Data(contentsOf: dflash2ConfigFixtureURL).write(to: configURL)
    #expect(DFlash2DraftModel.isDFlash2Directory(directory))

    // A qwen3_5_mtp head in a directory of any name is NOT a DFlash 2 drafter.
    try Data(#"{"model_type":"qwen3_5_mtp","architectures":["Qwen35MTP"]}"#.utf8)
        .write(to: configURL)
    #expect(!DFlash2DraftModel.isDFlash2Directory(directory))

    try FileManager.default.removeItem(at: configURL)
    #expect(!DFlash2DraftModel.isDFlash2Directory(directory))
}

// MARK: - The target tap

/// A stand-in target that records what the tap was armed with. It holds no
/// weights: the tap contract is about ids and shapes, not about a tower.
private final class DFlash2TapStub: DFlash2TapTarget {
    let dFlash2VocabularySize = 248_320
    let dFlash2HiddenSize = 5120
    let dFlash2LayerCount = 64
    var dFlash2TapLayerIds: [Int]?
    var dFlash2TappedHidden: MLXArray?

    func embedTokensForDFlash2(_ tokens: MLXArray) -> MLXArray { tokens }
    func logitsForDFlash2Hidden(_ hidden: MLXArray) -> MLXArray { hidden }
}

@Test("arming the tap checks the ids against the tower")
func dflash2TapArmingChecksTheLayerIds() throws {
    let target = DFlash2TapStub()
    #expect(target.dFlash2TapLayerIds == nil)

    try target.armDFlash2Tap(layerIds: [5, 19, 33, 47, 61])
    #expect(target.dFlash2TapLayerIds == [5, 19, 33, 47, 61])

    // A bad id is a refusal, not a clamp. A drafter reading the wrong layers
    // still produces tokens, and the only symptom would be a poor accept rate
    // that reads as a property of the pairing.
    #expect(throws: DFlash2Error.self) { try target.armDFlash2Tap(layerIds: []) }
    #expect(throws: DFlash2Error.self) { try target.armDFlash2Tap(layerIds: [5, 5]) }
    #expect(throws: DFlash2Error.self) { try target.armDFlash2Tap(layerIds: [64]) }
    #expect(throws: DFlash2Error.self) { try target.armDFlash2Tap(layerIds: [-1]) }
}

@Test("the drafter binds to a target of matching geometry and refuses another")
func dflash2BindChecksGeometry() throws {
    let drafter = DFlash2DraftModel(config: try dflash2Configuration())
    let target = DFlash2TapStub()
    try drafter.bind(target: target)

    // The pinned drafter's tapped ids are the ones the config names, and they
    // must be inside the Bonsai 2 tower's 64 layers.
    #expect(drafter.config.targetLayerIds.allSatisfy { $0 < target.dFlash2LayerCount })
    #expect(drafter.config.vocabularySize == target.dFlash2VocabularySize)
    #expect(drafter.config.hiddenSize == target.dFlash2HiddenSize)
}

@Test("the tap slot is not a module parameter")
func dflash2TapSlotStaysOffTheModuleTree() {
    // `Module` reflection classifies a stored property by its DYNAMIC TYPE, and
    // an `MLXArray` is classified as a parameter. A tapped hidden state stored
    // on the tower would therefore join `parameters()` the moment it is
    // non-nil, and with it strict key verification, the quantization walk and
    // `eval(model)`. This pins the box that keeps it out.
    final class Holder: Module {
        let slot = DFlash2TapSlot()
        @ParameterInfo(key: "w") var weight: MLXArray
        override init() {
            _weight.wrappedValue = MLXArray([Float]([1, 2]), [2])
            super.init()
        }
    }
    let holder = Holder()
    holder.slot.layerIds = [0]
    holder.slot.tappedHidden = MLXArray([Float]([3, 4]), [2])

    let names = Set(holder.parameters().flattened().map(\.0))
    #expect(names == ["w"])
}
