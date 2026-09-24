// Qwen4ExpNormConventionTests.swift
//
// The RMSNorm weight-offset convention: the configuration default, both
// placements of the key, and the validator that refuses a module whose
// loaded tensors contradict it.
//
// This is the defect that produced a step-0 teacher-forced parity failure on
// the box: the fork carried the knob and nothing ever set it, so the tower
// applied `y * (1 + w)` to a checkpoint that already stores `1 + w`. The
// top-8 came back flat, low and preferring a newline, while every shape,
// tensor count and digest checked out. Confirmed on the box: with the offset
// at 0 the step-0 argmax is 6184, matching the previous engine.
//
// The measurement cases build MLXArrays, so they need the metallib and are
// gated like the other GPU-bound suites; the configuration-decode cases are
// pure Codable and always run.

import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXRunners

@Suite("Qwen4Exp norm convention: configuration")
struct Qwen4ExpNormConventionConfigurationTests {

    private func decode(_ json: String) throws -> Qwen4ExpConfiguration {
        try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: Data(json.utf8))
    }

    /// ABSENT is the case that mattered: nothing in either tree's
    /// `config.json` names the convention, and this checkpoint bakes the
    /// offset. The previous engine pinned the same value as
    /// `MLXFastConstants.rmsNormConvention == .offsetBaked`.
    @Test("Absent anywhere, the family default is 0")
    func defaultsToBaked() throws {
        let flattened = try decode(#"{"model_type":"qwen4_exp","hidden_size":8}"#)
        #expect(flattened.textConfig.rmsNormWeightOffset == 0)
        #expect(flattened.textConfig.rmsNormWeightOffsetIsExplicit == false)

        let nested = try decode(
            #"{"model_type":"qwen4_exp","text_config":{"hidden_size":8}}"#)
        #expect(nested.textConfig.rmsNormWeightOffset == 0)
        #expect(nested.textConfig.rmsNormWeightOffsetIsExplicit == false)
    }

    /// The TRANSFORMED tree: the transform flattens `text_config` away, so
    /// the key sits at the root and the model read it there.
    @Test("At the top level, with no text_config, it is honoured")
    func honoursTopLevel() throws {
        let configuration = try decode(
            #"{"model_type":"qwen4_exp","rms_norm_weight_offset":1,"hidden_size":8}"#)
        #expect(configuration.textConfig.rmsNormWeightOffset == 1)
        #expect(configuration.textConfig.rmsNormWeightOffsetIsExplicit)
    }

    /// The raw HF checkpoint: the key lives inside `text_config`.
    @Test("Inside text_config, it is honoured")
    func honoursNested() throws {
        let configuration = try decode(
            #"{"model_type":"qwen4_exp","text_config":{"rms_norm_weight_offset":1}}"#)
        #expect(configuration.textConfig.rmsNormWeightOffset == 1)
        #expect(configuration.textConfig.rmsNormWeightOffsetIsExplicit)
    }

    /// A `text_config` that does not name it while the root does.
    @Test("A root key reaches a text_config that is silent about it")
    func rootReachesSilentTextConfig() throws {
        let configuration = try decode(
            #"""
            {"model_type":"qwen4_exp","rms_norm_weight_offset":1,
             "text_config":{"hidden_size":8}}
            """#)
        #expect(configuration.textConfig.rmsNormWeightOffset == 1)
    }

    /// And nested still wins when both name it, so nothing that works today
    /// changes meaning.
    @Test("A nested key beats the root")
    func nestedBeatsRoot() throws {
        let configuration = try decode(
            #"""
            {"model_type":"qwen4_exp","rms_norm_weight_offset":1,
             "text_config":{"rms_norm_weight_offset":0}}
            """#)
        #expect(configuration.textConfig.rmsNormWeightOffset == 0)
    }

    /// An explicit 0 is not the same fact as an absent key, even though the
    /// value matches: one is the checkpoint speaking, the other is this
    /// family's default.
    @Test("An explicit 0 is recorded as explicit")
    func explicitZeroIsExplicit() throws {
        let configuration = try decode(
            #"{"model_type":"qwen4_exp","rms_norm_weight_offset":0}"#)
        #expect(configuration.textConfig.rmsNormWeightOffset == 0)
        #expect(configuration.textConfig.rmsNormWeightOffsetIsExplicit)
    }
}

/// The validator itself.
///
/// GPU-GATED: every case here builds `MLXArray`s, so it needs MLX's default
/// metallib and is skipped on a machine without it — the same gate the
/// Qwen4Exp adopt/runner/n-gram suites carry. Run it on a box.
@Suite("Qwen4Exp norm convention: validator")
struct Qwen4ExpNormConventionValidatorTests {

    /// A module holding one norm, so the walk has something to read.
    final class OneNorm: Module {
        @ModuleInfo(key: "norm") var norm: Qwen4ExpRMSNorm

        init(weights: [Float], weightOffset: Float) {
            let norm = Qwen4ExpRMSNorm(
                dimensions: weights.count, eps: 1e-6, weightOffset: weightOffset)
            norm.weight = MLXArray(weights)
            _norm.wrappedValue = norm
            super.init()
        }
    }

    /// Centred at 1 and all positive: the checkpoint BAKES the offset, so
    /// the model must apply 0.
    private let baked: [Float] = [0.9, 1.0, 1.1, 1.05, 0.95, 1.2, 0.8, 1.0]
    /// Scattered around 0: the checkpoint is ZERO-CENTERED and the model
    /// must apply 1.
    private let zeroCentered: [Float] = [-0.2, 0.3, -0.1, 0.25, -0.3, 0.15, -0.05, 0.1]

    @Test("Baked weights with offset 0 are accepted")
    func bakedWithZeroOffset() {
        let model = OneNorm(weights: baked, weightOffset: 0)
        #expect(Qwen4ExpNormConvention.mismatch(model: model, expectedOffset: 0) == nil)
    }

    /// The defect, exactly: baked weights served as if they were
    /// zero-centered.
    @Test("Baked weights with offset 1 are refused")
    func bakedWithOneOffsetRefused() {
        let model = OneNorm(weights: baked, weightOffset: 1)
        let mismatch = Qwen4ExpNormConvention.mismatch(model: model, expectedOffset: 1)
        #expect(mismatch != nil)
        #expect(mismatch?.observed.contains("reads as offsetBaked") == true)
        #expect(mismatch?.impliedOffset == 0)
    }

    @Test("Zero-centered weights with offset 1 are accepted")
    func zeroCenteredWithOneOffset() {
        let model = OneNorm(weights: zeroCentered, weightOffset: 1)
        #expect(Qwen4ExpNormConvention.mismatch(model: model, expectedOffset: 1) == nil)
    }

    @Test("Zero-centered weights with offset 0 are refused")
    func zeroCenteredWithZeroOffsetRefused() {
        let model = OneNorm(weights: zeroCentered, weightOffset: 0)
        let mismatch = Qwen4ExpNormConvention.mismatch(model: model, expectedOffset: 0)
        #expect(mismatch != nil)
        #expect(mismatch?.observed.contains("reads as zeroCentered") == true)
        #expect(mismatch?.impliedOffset == 1)
    }

    /// A module configured differently from what the caller expects is
    /// caught BEFORE the measurement: the tensors say what the checkpoint
    /// holds, the modules say what the model will do with it.
    @Test("A module applying another offset is named first")
    func moduleOffsetCheckedFirst() {
        let model = OneNorm(weights: baked, weightOffset: 1)
        let mismatch = Qwen4ExpNormConvention.mismatch(model: model, expectedOffset: 0)
        #expect(mismatch?.observed.contains("applies offset 1.0") == true)
    }

    /// Neither pattern: all positive but centred at zero, which is what a
    /// mis-converted file looks like.
    @Test("An unreadable tensor is refused as neither")
    func unreadableRefused() {
        let model = OneNorm(
            weights: [0.001, 0.002, 0.0, 0.003, 0.001, 0.0, 0.002, 0.001],
            weightOffset: 0)
        let mismatch = Qwen4ExpNormConvention.mismatch(model: model, expectedOffset: 0)
        #expect(mismatch?.observed.contains("reads as neither") == true)
    }

    /// The thresholds are the previous engine's, unchanged.
    @Test("The thresholds match the pinned constants")
    func thresholds() {
        #expect(Qwen4ExpNormConvention.zeroCenteredMinNegativeFraction == 0.20)
        #expect(Qwen4ExpNormConvention.zeroCenteredMaxNegativeFraction == 0.80)
        #expect(Qwen4ExpNormConvention.bakedMinMean == 0.05)
        #expect(Qwen4ExpNormConvention.bakedWeightOffset == 0)
        #expect(Qwen4ExpNormConvention.zeroCenteredWeightOffset == 1)
    }
}
