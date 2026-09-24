// Copyright © 2026 Eigen Labs.
//
// Qwen 3.8 Flash-Next — the RMSNorm weight-offset convention, verified
// against the weights that were actually loaded.
//
// A checkpoint either stores `w` and wants `y * (1 + w)` (ZERO-CENTERED), or
// stores `1 + w` and wants `y * w` (OFFSET BAKED). Nothing in `config.json`
// says which. Applying the wrong one scales every non-gated norm in the
// tower by about one unit and produces incoherent output while every shape,
// every tensor count and every digest still checks out — the box saw exactly
// that as a step-0 teacher-forced parity failure whose top-8 was flat, low,
// and preferred a newline.
//
// So the configured offset is CHECKED against the tensors. Ported from the
// previous engine, which pinned the fact and enforced it:
// mlxfast-qwen38-125b-a6b-engine-dev
// `Sources/MLXFastHarness/Qwen4ExpNormConventionBind.swift:60-230`
// (`qwen4ExpMeasureNormWeight`, `validateLoadedNormConvention`) and
// `Sources/MLXFastCore/Constants.swift:139-144,176-230` (the convention and
// its thresholds). The thresholds below are that file's, unchanged.
//
// WHAT IS MEASURED, AND WHY IT IS THE SIGN PATTERN RATHER THAN THE MEAN. A
// zero-centered tensor is scattered around zero and has roughly half its
// entries negative; a baked tensor is `1 + w` and is negative only where some
// `w < -1`. The mean alone would misread `mtp.pre_fc_norm_embedding`, which
// averages 0.236 — near enough to zero to fool a mean threshold — while
// every one of its entries is positive.
//
// WHICH NORMS. Every `Qwen4ExpRMSNorm` in the loaded tree, which is exactly
// the non-gated set: the gated deltanet norm is a different type
// (`Qwen4ExpRMSNormGated`, 36 tensors) and always computes `y * w` in both
// conventions, so it carries no evidence and is not read.

import Foundation
import MLX
import MLXNN

/// What one norm tensor's stored values say about the convention.
public enum Qwen4ExpNormReading: Equatable, Sendable {
    case baked
    case zeroCentered
    /// Neither pattern. The payload NAMES THE RULE that fired, so a refusal
    /// says why rather than only what.
    case unreadable(rule: String)
}

/// What one tensor measured, kept beside its verdict so a refusal can quote
/// the numbers instead of asking the reader to take them again.
public struct Qwen4ExpNormMeasurement: Sendable {
    public let negativeFraction: Float
    public let mean: Float
    public let reading: Qwen4ExpNormReading
}

/// The convention check.
public enum Qwen4ExpNormConvention {

    /// The offset a BAKED checkpoint's model applies: it already holds
    /// `1 + w`, so the model multiplies by the stored weight as it is.
    public static let bakedWeightOffset: Float = 0
    /// The offset a ZERO-CENTERED checkpoint's model applies.
    public static let zeroCenteredWeightOffset: Float = 1

    /// A tensor reads ZERO-CENTERED when its negative fraction lands in
    /// `[min, max]`, and BAKED when it is below `min` and the mean clears
    /// `bakedMinMean`. The two rules meet at one number, so no tensor can
    /// fall between them.
    ///
    /// 0.20 is where the two populations separate. Over all 157 non-gated
    /// norm tensors of the pinned tree the LARGEST negative fraction was
    /// 0.0551; a genuinely zero-centered tensor sits near 0.5. That leaves
    /// about 3.6x of headroom on each side. An earlier 0.05 ceiling was
    /// wrong — the hyper-connection norms are not a unit-scale gain, and one
    /// tensor in 193 exceeded it, so the pinned checkpoint would have been
    /// refused at load.
    public static let zeroCenteredMinNegativeFraction: Float = 0.20
    public static let zeroCenteredMaxNegativeFraction: Float = 0.80
    /// A baked tensor's mean must also clear this. The lowest the box
    /// measured is 0.236, so the floor sits below that with room; it exists
    /// to catch an all-positive tensor of near-zeros, which is neither
    /// convention and is what a mis-converted file looks like.
    public static let bakedMinMean: Float = 0.05

    /// Classify one norm weight tensor.
    ///
    /// Reads a tensor that is ALREADY RESIDENT. No file is opened.
    public static func measure(_ weight: MLXArray) -> Qwen4ExpNormMeasurement {
        let values = weight.asType(.float32)
        let count = values.size
        guard count > 0 else {
            return Qwen4ExpNormMeasurement(
                negativeFraction: 0, mean: 0,
                reading: .unreadable(rule: "the tensor is empty"))
        }
        let negatives = (values .< MLXArray(Float(0))).asType(.float32).sum()
            .item(Float.self)
        let negativeFraction = negatives / Float(count)
        let mean = values.mean().item(Float.self)

        let reading: Qwen4ExpNormReading
        if negativeFraction >= zeroCenteredMinNegativeFraction {
            if negativeFraction <= zeroCenteredMaxNegativeFraction {
                reading = .zeroCentered
            } else {
                reading = .unreadable(
                    rule: "more than \(zeroCenteredMaxNegativeFraction) of the entries are "
                        + "negative, which neither convention produces")
            }
        } else if mean >= bakedMinMean {
            reading = .baked
        } else {
            reading = .unreadable(
                rule: "fewer than \(zeroCenteredMinNegativeFraction) of the entries are "
                    + "negative, so it is not zero-centered, but the mean is below "
                    + "\(bakedMinMean), so it is not baked either")
        }
        return Qwen4ExpNormMeasurement(
            negativeFraction: negativeFraction, mean: mean, reading: reading)
    }

    /// Every NON-GATED norm in the loaded tree, in sorted path order.
    ///
    /// Sorted, so the reading is the same on two runs of the same model and a
    /// refusal names the same tensors twice.
    public static func nonGatedNormWeights(
        _ model: Module
    ) -> [(path: String, weight: MLXArray)] {
        model.leafModules().flattened()
            .compactMap { path, module -> (String, MLXArray)? in
                guard let norm = module as? Qwen4ExpRMSNorm else { return nil }
                return (path, norm.weight)
            }
            .sorted { $0.0 < $1.0 }
    }

    /// What the loaded tensors say the offset should be, and why.
    public struct Verdict: Sendable {
        /// The offset the weights imply: 0 when they read baked, 1 when they
        /// read zero-centered.
        public let impliedOffset: Float
        /// Tensors that contradict `impliedOffset`, worst first, each with
        /// its numbers.
        public let disagreements: [String]
        /// How many non-gated norms were read.
        public let inspected: Int
    }

    /// Read the loaded model's non-gated norms against `expectedOffset`.
    ///
    /// Returns nil when the model carries no non-gated norms — a module that
    /// is not this family, which its own type check refuses first.
    public static func read(
        model: Module, expectedOffset: Float
    ) -> Verdict? {
        let norms = nonGatedNormWeights(model)
        guard !norms.isEmpty else { return nil }

        let expectBaked = expectedOffset == bakedWeightOffset
        var disagreements: [String] = []
        for (path, weight) in norms {
            let measured = measure(weight)
            let numbers =
                "negative fraction \(measured.negativeFraction), mean \(measured.mean)"
            switch measured.reading {
            case .baked where expectBaked: continue
            case .zeroCentered where !expectBaked: continue
            case .baked:
                disagreements.append(
                    "\(path) reads as offsetBaked (\(numbers)): fewer than "
                        + "\(zeroCenteredMinNegativeFraction) of its entries are negative "
                        + "and its mean clears \(bakedMinMean)")
            case .zeroCentered:
                disagreements.append(
                    "\(path) reads as zeroCentered (\(numbers)): its negative fraction is "
                        + "inside [\(zeroCenteredMinNegativeFraction), "
                        + "\(zeroCenteredMaxNegativeFraction)]")
            case .unreadable(let rule):
                disagreements.append("\(path) reads as neither (\(numbers)): \(rule)")
            }
        }
        // The reading of the MAJORITY is what the weights imply; a single
        // disagreement is still a refusal, and the caller reports both.
        let implied = disagreements.count * 2 > norms.count
            ? (expectBaked ? zeroCenteredWeightOffset : bakedWeightOffset)
            : expectedOffset
        return Verdict(
            impliedOffset: implied, disagreements: disagreements, inspected: norms.count)
    }

    /// A refusal message naming the first few offenders, or nil when the
    /// loaded weights agree with `expectedOffset`.
    ///
    /// The MODULES are checked first: the measurement says what the
    /// CHECKPOINT holds, this says what the model will actually DO with it,
    /// and configuration decode is what sets it.
    public static func mismatch(
        model: Module, expectedOffset: Float
    ) -> (observed: String, impliedOffset: Float)? {
        let wrongOffset = model.leafModules().flattened()
            .compactMap { path, module -> String? in
                guard let norm = module as? Qwen4ExpRMSNorm,
                    norm.weightOffset != expectedOffset
                else { return nil }
                return "\(path) applies offset \(norm.weightOffset)"
            }
            .sorted()
        if !wrongOffset.isEmpty {
            return (
                "\(wrongOffset.count) module(s) apply another offset, first: "
                    + wrongOffset.prefix(3).joined(separator: "; "),
                expectedOffset
            )
        }

        guard let verdict = read(model: model, expectedOffset: expectedOffset),
            !verdict.disagreements.isEmpty
        else { return nil }
        let summary = verdict.disagreements.prefix(3).joined(separator: "; ")
        let tail =
            verdict.disagreements.count > 3
            ? "; ... (\(verdict.disagreements.count) of \(verdict.inspected))" : ""
        return (summary + tail, verdict.impliedOffset)
    }
}
