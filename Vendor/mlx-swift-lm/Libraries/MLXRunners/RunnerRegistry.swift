// Copyright © 2026 Eigen Labs.
//
// MLXRunners — `model_type` → runner type (Darkbloom runner contract §6.2
// rule 4).
//
// This is the ONLY place a family name appears on a consumer's path.
// Darkbloom's advertise gate is `RunnerRegistry.contains(modelType:)`;
// bench-worker resolves the runner for the checkpoint it was pointed at.
// Neither carries a family switch of its own.

import Foundation

/// Process-wide `model_type` → runner registry.
///
/// Registration is a lock-protected dictionary rather than a `let` table so
/// a track repo can register its own runner before resolving, and a test can
/// register a scripted one. The static table below is the fork's own set.
public final class RunnerRegistry: @unchecked Sendable {

    public static let shared = RunnerRegistry()

    private let lock = NSLock()
    private var runners: [String: any Runner.Type] = [:]

    /// Refusals from resolution.
    public enum RegistryError: Error, CustomStringConvertible, Equatable {
        case unknownModelType(String)

        public var description: String {
            switch self {
            case .unknownModelType(let type):
                return "runner registry: no runner claims model_type \(type)"
            }
        }
    }

    private init() {
        for runner in Self.firstPartyRunners {
            registerLocked(runner)
        }
    }

    /// The fork's own runners. One entry per family; every `model_type` in a
    /// runner's manifest is claimed.
    ///
    /// A later registration REPLACES an earlier claim on the same
    /// `model_type` — that is what lets a track repo shadow a fork runner
    /// while it is being promoted.
    private static var firstPartyRunners: [any Runner.Type] {
        [
            Gemma4TextRunner.self,
            GPTOSSRunner.self,
            NemotronH35Runner.self,
            Qwen35Runner.self,
            Qwen3VLRunner.self,
            Qwen4ExpRunner.self,
            // INSERTION POINT: register additional runners here
        ]
    }

    /// Claim every `model_type` in the runner's manifest.
    public func register(_ runner: any Runner.Type) {
        lock.lock()
        defer { lock.unlock() }
        registerLocked(runner)
    }

    private func registerLocked(_ runner: any Runner.Type) {
        for modelType in runner.manifest.modelTypes {
            runners[modelType] = runner
        }
    }

    /// The runner claiming `modelType`, or a refusal.
    public func resolve(modelType: String) throws -> any Runner.Type {
        lock.lock()
        defer { lock.unlock() }
        guard let runner = runners[modelType] else {
            throw RegistryError.unknownModelType(modelType)
        }
        return runner
    }

    /// Darkbloom's advertise gate.
    public func contains(modelType: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return runners[modelType] != nil
    }

    /// Every registered runner's manifest, de-duplicated by `runnerID` and
    /// sorted by it so the listing is stable across registration order.
    public func manifests() -> [RunnerManifest] {
        lock.lock()
        defer { lock.unlock() }
        var byID: [String: RunnerManifest] = [:]
        for runner in runners.values {
            byID[runner.manifest.runnerID] = runner.manifest
        }
        return byID.keys.sorted().compactMap { byID[$0] }
    }

    /// Resolve straight from a checkpoint directory's `config.json`.
    public func resolve(checkpoint directory: URL) throws -> any Runner.Type {
        try resolve(modelType: RunnerCheckpoint.modelType(at: directory))
    }
}
