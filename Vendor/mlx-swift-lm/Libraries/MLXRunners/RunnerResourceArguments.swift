// Copyright © 2026 Eigen Labs.
//
// MLXRunners — `--resource <name>=<path>` parsing for bench-worker.
//
// The worker never OPENS a resource. It validates that the caller named
// something that exists and could be opened, boxes the location, and hands
// the bag to the runner, which casts it to whatever protocol that family
// wants. A 29.8 GiB n-gram table is the case this exists for: the worker has
// no business knowing what is inside it.
//
// Every refusal happens BEFORE the load. A worker that discovers a bad
// resource path after paying for a multi-minute weight load has wasted the
// box's time to report something it could have known at argv.
//
// In `MLXRunners` rather than in the executable so the parse is testable;
// a test target that depended on the executable would be run by SwiftPM as
// the swift-testing host (see Package.swift).

import Foundation

/// Refusals from `--resource` parsing. Each names the offending resource, so
/// an operator reading one line of stderr knows which argument to fix.
public enum RunnerResourceArgumentError: Error, CustomStringConvertible, Equatable {
    /// The argument is not `<name>=<path>`, or either half is empty.
    case malformed(String)
    /// The same name was supplied twice. Last-one-wins would make the
    /// meaning of a command depend on argument order, and a repeated name is
    /// far more likely a mistake than an intent.
    case duplicateName(String)
    /// Nothing exists at that path.
    case pathMissing(name: String, path: String)
    /// Something exists there, but it is neither a regular file nor a
    /// directory — a socket, a fifo, a device. A runner asked to open it
    /// would block or fail deep inside a load.
    case pathNotAFileOrDirectory(name: String, path: String)

    public var description: String {
        switch self {
        case .malformed(let argument):
            return "--resource \(argument) is not <name>=<path>"
        case .duplicateName(let name):
            return "--resource \(name) was supplied more than once"
        case .pathMissing(let name, let path):
            return "--resource \(name): nothing exists at \(path)"
        case .pathNotAFileOrDirectory(let name, let path):
            return "--resource \(name): \(path) is neither a file nor a directory"
        }
    }
}

/// Parsing for the worker's repeatable `--resource <name>=<path>`.
public enum RunnerResourceArguments {

    /// Turn the raw `<name>=<path>` values into the runner's resource bag.
    ///
    /// Each location is boxed as `NSURL`, which bridges back to `URL` through
    /// `as?` on the runner's side — no new type crosses the seam, and the
    /// worker still never opens the thing.
    public static func parse(
        _ arguments: [String],
        fileManager: FileManager = .default
    ) throws -> RunnerResources {
        var resources = RunnerResources()
        var seen = Set<String>()
        for argument in arguments {
            guard let separator = argument.firstIndex(of: "=") else {
                throw RunnerResourceArgumentError.malformed(argument)
            }
            let name = String(argument[argument.startIndex ..< separator])
            let path = String(argument[argument.index(after: separator)...])
            guard !name.isEmpty, !path.isEmpty else {
                throw RunnerResourceArgumentError.malformed(argument)
            }
            guard seen.insert(name).inserted else {
                throw RunnerResourceArgumentError.duplicateName(name)
            }

            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: path, isDirectory: &isDirectory) else {
                throw RunnerResourceArgumentError.pathMissing(name: name, path: path)
            }
            if !isDirectory.boolValue {
                // `fileExists` is true for a fifo or a socket too, and the
                // ONLY thing that separates those from a regular file here is
                // the item's own type attribute.
                let attributes = try? fileManager.attributesOfItem(atPath: path)
                guard attributes?[.type] as? FileAttributeType == .typeRegular else {
                    throw RunnerResourceArgumentError.pathNotAFileOrDirectory(
                        name: name, path: path)
                }
            }
            resources[name] = URL(fileURLWithPath: path) as NSURL
        }
        return resources
    }
}
