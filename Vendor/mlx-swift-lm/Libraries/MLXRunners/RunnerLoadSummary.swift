// Copyright © 2026 Eigen Labs.
//
// MLXRunners — the `--verbose` load summary.
//
// WHY THIS EXISTS. A box hit a step-0 teacher-forced parity failure on the
// real checkpoint and the worker's stderr was EMPTY on every run, so nothing
// was triageable: the tower was applying the wrong RMSNorm weight-offset
// convention, and every shape, tensor count and digest still checked out.
// One line — the resolved offset and where it came from — would have named
// the defect on the first run. So the summary states what the load DECIDED,
// not that it succeeded.
//
// RULES.
//   * stderr only. stdout is the wire and stays byte-identical, verbose or
//     not; a summary on stdout would corrupt the protocol.
//   * Emitted BEFORE the hello, so it is present even when the first verb
//     fails.
//   * Never without `--verbose` or `BENCH_WORKER_VERBOSE=1`.
//   * One `key=value` per line, prefixed `bench-worker:`, so a line is
//     greppable and an operator can diff two runs.
//   * Values are read from state the load ALREADY holds. Nothing here opens
//     a weight file.

import Foundation

/// An ordered `key=value` collection, rendered one line per entry.
///
/// Ordered, not a dictionary: the reading order is the load's own order —
/// identity, then the checkpoint, then what was bound — and two runs of the
/// same load produce diffable output.
public struct RunnerLoadSummary: Sendable {
    public private(set) var entries: [(key: String, value: String)] = []

    public init() {}

    public mutating func add(_ key: String, _ value: String) {
        entries.append((key, value))
    }

    public mutating func add(_ key: String, _ value: Int) {
        add(key, String(value))
    }

    public mutating func add(_ key: String, _ value: Bool) {
        add(key, value ? "yes" : "no")
    }

    public mutating func add(_ key: String, _ value: Float) {
        add(key, "\(value)")
    }

    /// Append another summary's entries, prefixing nothing: a family's lines
    /// read at the same level as the generic ones.
    public mutating func add(contentsOf other: RunnerLoadSummary) {
        entries.append(contentsOf: other.entries)
    }

    /// A duration in milliseconds, to one decimal.
    public mutating func add(_ key: String, seconds: TimeInterval) {
        add(key, String(format: "%.1fms", seconds * 1000))
    }

    /// The rendered lines, newline-terminated.
    public func rendered(prefix: String = "bench-worker: ") -> String {
        entries.map { "\(prefix)\($0.key)=\($0.value)\n" }.joined()
    }

    /// Write to stderr. The ONLY destination: see the file header.
    public func writeToStandardError(prefix: String = "bench-worker: ") {
        FileHandle.standardError.write(Data(rendered(prefix: prefix).utf8))
    }

    /// Whether the caller asked for this.
    public static func isEnabled(
        flag: Bool, environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        flag || environment[verboseEnvironmentName] == "1"
    }

    public static let verboseEnvironmentName = "BENCH_WORKER_VERBOSE"
}

extension Runner {

    /// The generic half of the summary: everything derivable from the runner
    /// boundary itself. A family adds its own on top.
    public func genericLoadSummary(
        weights: URL, options: RunnerLoadOptions
    ) -> RunnerLoadSummary {
        var summary = RunnerLoadSummary()
        summary.add("runner_id", Self.manifest.runnerID)
        summary.add("manifest_sha256", Self.manifest.sha256Digest())
        summary.add("weights_dir", weights.path)
        summary.add("model_type", loadedModelType)
        summary.add("backend", Self.manifest.backend)

        summary.add("attending_layers", layerKinds.count)
        let attendingIndices = layerKinds.enumerated()
            .map { position, kind in kind.modelLayerIndex ?? position }
        summary.add(
            "full_attention_layer_indices",
            attendingIndices.map(String.init).joined(separator: ","))

        summary.add(
            "decoders_loaded", loadedDecoders.map(\.rawValue).joined(separator: ","))
        summary.add("mtp_head_bound", loadedDecoders.contains(.mtp))
        if let provenance = headProvenance {
            summary.add("head_provenance_sha256", provenance.sha256)
            summary.add("head_provenance_bytes", provenance.bytes)
            summary.add("head_provenance_files", provenance.fileCount)
        } else {
            summary.add("head_provenance", "none")
        }

        summary.add("requires_keep_mask", Self.manifest.requiresKeepMask)
        summary.add("kv_backends", Self.manifest.kvBackends.map(\.rawValue).joined(separator: ","))
        summary.add("eos_token_ids", eosTokenIDs.sorted().map(String.init).joined(separator: ","))
        // The rule, by name: two implementations that disagree here disagree
        // on every tie, and a tie is exactly where a parity run diverges.
        summary.add("argmax_tie_break", "lowest_token_id")
        return summary
    }

    /// The whole summary. A family overrides this to add its own lines.
    public func loadSummary(weights: URL, options: RunnerLoadOptions) -> RunnerLoadSummary {
        genericLoadSummary(weights: weights, options: options)
    }
}
