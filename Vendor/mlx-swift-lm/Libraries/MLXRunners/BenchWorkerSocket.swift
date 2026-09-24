// Copyright © 2026 Eigen Labs.
//
// MLXRunners — the Unix-domain socket the resident worker listens on.
//
// NDJSON over `AF_UNIX`/`SOCK_STREAM`, one line per message, strict order —
// the same wire the resident's in-process sessions speak, so a phase that
// attaches over the socket and a phase served in process are byte-identical
// to benchd.
//
// Written directly on the POSIX calls rather than on a networking package:
// the package takes no dependency it does not already have, and the whole
// surface here is listen, accept, read a line, write a line, close.

import Foundation

#if canImport(Darwin)
import Darwin
#endif

/// Refusals from the socket layer. Each names the syscall that failed and
/// the path, because an operator reading one line needs to know whether the
/// resident is missing, busy, or unreachable.
public enum BenchWorkerSocketError: Error, CustomStringConvertible, Equatable {
    case pathTooLong(String)
    case cannotCreate(String)
    case cannotBind(path: String, errno: Int32)
    case cannotListen(path: String, errno: Int32)
    case cannotConnect(path: String, errno: Int32)
    case closed

    public var description: String {
        switch self {
        case .pathTooLong(let path):
            return "socket path is too long for a sockaddr_un: \(path)"
        case .cannotCreate(let detail):
            return "cannot create a unix socket (\(detail))"
        case .cannotBind(let path, let code):
            return "cannot bind \(path): \(String(cString: strerror(code)))"
        case .cannotListen(let path, let code):
            return "cannot listen on \(path): \(String(cString: strerror(code)))"
        case .cannotConnect(let path, let code):
            return "cannot reach the resident at \(path): "
                + "\(String(cString: strerror(code)))"
        case .closed:
            return "the connection closed"
        }
    }
}

/// One accepted or dialled connection: line in, line out.
///
/// `@unchecked Sendable` because the only mutable state is the read buffer,
/// and a connection is owned by exactly one session at a time — which is the
/// resident's whole concurrency model.
public final class BenchWorkerSocketConnection: @unchecked Sendable {
    /// -1 once closed. Guarded, and cleared BEFORE the `close(2)`, because a
    /// descriptor closed twice is not a harmless mistake: the number is
    /// returned to the process immediately and the kernel hands it to the
    /// next `accept`, so the second close shuts down an unrelated
    /// connection. That defect showed up here as an intermittent "resident
    /// sent no hello" — a phase whose freshly accepted socket had been
    /// closed out from under it by the PREVIOUS phase's `deinit`.
    private var descriptor: Int32
    private let descriptorLock = NSLock()
    private var buffer = Data()
    private var atEOF = false

    init(descriptor: Int32) {
        self.descriptor = descriptor
    }

    deinit { close() }

    /// The descriptor, or -1 when this connection no longer owns one.
    private var borrowedDescriptor: Int32 {
        descriptorLock.lock()
        defer { descriptorLock.unlock() }
        return descriptor
    }

    /// The next newline-terminated line, or nil at EOF.
    public func readLine() -> String? {
        while true {
            if let index = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                let line = buffer[buffer.startIndex ..< index]
                buffer.removeSubrange(buffer.startIndex ... index)
                return String(decoding: line, as: UTF8.self)
            }
            if atEOF {
                guard !buffer.isEmpty else { return nil }
                // A final line with no trailing newline is still a line.
                let line = String(decoding: buffer, as: UTF8.self)
                buffer.removeAll()
                return line
            }
            let fileDescriptor = borrowedDescriptor
            guard fileDescriptor >= 0 else {
                atEOF = true
                continue
            }
            var chunk = [UInt8](repeating: 0, count: 64 * 1024)
            let count = Darwin.read(fileDescriptor, &chunk, chunk.count)
            if count > 0 {
                buffer.append(contentsOf: chunk[0 ..< count])
            } else if count == 0 {
                atEOF = true
            } else if errno == EINTR {
                continue
            } else {
                atEOF = true
            }
        }
    }

    /// Write one line, newline included. Partial writes are retried: a short
    /// write that was ignored would truncate a response and desynchronize
    /// the whole session.
    public func write(line: String) {
        let fileDescriptor = borrowedDescriptor
        guard fileDescriptor >= 0 else { return }
        var bytes = Array((line + "\n").utf8)
        var offset = 0
        while offset < bytes.count {
            // `Darwin.write`, qualified: the unqualified name resolves to
            // this type's own `write(line:)` inside the class body.
            let written = bytes.withUnsafeBytes { raw -> Int in
                Darwin.write(
                    fileDescriptor, raw.baseAddress!.advanced(by: offset),
                    bytes.count - offset)
            }
            if written > 0 {
                offset += written
            } else if written < 0 && errno == EINTR {
                continue
            } else {
                return
            }
        }
    }

    /// Idempotent. The descriptor is surrendered under the lock before the
    /// `close(2)`, so a second call — `runSession` finishing and then
    /// `deinit`, say — cannot close a number the kernel has already reissued.
    public func close() {
        descriptorLock.lock()
        let owned = descriptor
        descriptor = -1
        descriptorLock.unlock()
        guard owned >= 0 else { return }
        _ = Darwin.close(owned)
    }
}

/// A bound, listening Unix socket.
public final class BenchWorkerSocketListener: @unchecked Sendable {
    private let descriptor: Int32
    public let path: String
    /// Teardown runs from the accept loop AND from a signal handler, so it
    /// must be safe to call twice: closing an already-closed descriptor is
    /// sloppy, and unlinking a path a LATER resident has already rebound
    /// would delete a live socket.
    private let lock = NSLock()
    private var isShutDown = false

    /// Bind and listen.
    ///
    /// A STALE socket file is removed first. A resident is one process per
    /// window and the window's tooling tears it down, so a leftover node
    /// means the previous resident died; refusing to bind over it would turn
    /// one crash into every later window failing to start.
    public init(path: String, backlog: Int32 = 16) throws {
        self.path = path
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        let capacity = MemoryLayout.size(ofValue: address.sun_path)
        let bytes = Array(path.utf8)
        guard bytes.count < capacity else {
            throw BenchWorkerSocketError.pathTooLong(path)
        }
        withUnsafeMutableBytes(of: &address.sun_path) { raw in
            raw.copyBytes(from: bytes)
        }
        address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)

        unlink(path)
        descriptor = socket(AF_UNIX, SOCK_STREAM, 0)
        guard descriptor >= 0 else {
            throw BenchWorkerSocketError.cannotCreate(String(cString: strerror(errno)))
        }
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bound == 0 else {
            let code = errno
            _ = Darwin.close(descriptor)
            throw BenchWorkerSocketError.cannotBind(path: path, errno: code)
        }
        guard listen(descriptor, backlog) == 0 else {
            let code = errno
            _ = Darwin.close(descriptor)
            unlink(path)
            throw BenchWorkerSocketError.cannotListen(path: path, errno: code)
        }
    }

    deinit { shutDown() }

    /// Block until a client connects. nil once the listener is shut down.
    public func accept() -> BenchWorkerSocketConnection? {
        while true {
            let client = Darwin.accept(descriptor, nil, nil)
            if client >= 0 { return BenchWorkerSocketConnection(descriptor: client) }
            if errno == EINTR { continue }
            return nil
        }
    }

    /// Close the listener and remove its socket file. Idempotent, and safe
    /// to call from a signal-source thread: closing the descriptor is what
    /// wakes a blocked `accept()`.
    public func shutDown() {
        lock.lock()
        let alreadyDown = isShutDown
        isShutDown = true
        lock.unlock()
        guard !alreadyDown, descriptor >= 0 else { return }
        _ = Darwin.close(descriptor)
        unlink(path)
    }
}

/// Dial a resident.
public enum BenchWorkerSocketClient {
    public static func connect(path: String) throws -> BenchWorkerSocketConnection {
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        let capacity = MemoryLayout.size(ofValue: address.sun_path)
        let bytes = Array(path.utf8)
        guard bytes.count < capacity else {
            throw BenchWorkerSocketError.pathTooLong(path)
        }
        withUnsafeMutableBytes(of: &address.sun_path) { raw in
            raw.copyBytes(from: bytes)
        }
        address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)

        let descriptor = socket(AF_UNIX, SOCK_STREAM, 0)
        guard descriptor >= 0 else {
            throw BenchWorkerSocketError.cannotCreate(String(cString: strerror(errno)))
        }
        let connected = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connected == 0 else {
            let code = errno
            _ = Darwin.close(descriptor)
            throw BenchWorkerSocketError.cannotConnect(path: path, errno: code)
        }
        return BenchWorkerSocketConnection(descriptor: descriptor)
    }
}
