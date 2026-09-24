// Copyright © 2024 Apple Inc.

import Cmlx
import Foundation
import Metal

/// API for controlling GPU related features.
///
/// Note: previously this also had properties that managed the buffer use -- those are now
/// found in ``Memory`` but remain here as deprecated.
///
/// ### See Also
/// - <doc:running-on-ios>
/// - ``Memory``
public enum GPU {

    public typealias Snapshot = Memory.Snapshot

    /// Get the actively used memory in bytes.
    ///
    /// Note, this will not always match memory use reported by the system because
    /// it does not include cached memory buffers.
    @available(*, deprecated, renamed: "Memory.activeMemory")
    public static var activeMemory: Int {
        Memory.activeMemory
    }

    /// Get the cache size in bytes.
    ///
    /// The cache includes memory not currently used that has not been returned
    /// to the system allocator. This represents buffers from previous
    /// computations that are kept in a buffer pool for potential reuse.
    ///
    /// During model inference, this can grow significantly as buffers of various
    /// sizes accumulate from intermediate computations.  See ``GPU`` for
    /// more information on cache size and tuning.
    ///
    /// The cache size is controlled by the cache limit (see ``Memory/cacheLimit``).
    /// When the limit is exceeded, older cached buffers are freed on the next allocatioån.
    @available(*, deprecated, renamed: "Memory.cacheMemory")
    public static var cacheMemory: Int {
        Memory.cacheMemory
    }

    /// Get the peak amount of active memory in bytes.
    ///
    /// The maximum memory used is recorded from the beginning of the program
    /// execution.
    @available(*, deprecated, renamed: "Memory.peakMemory")
    public static var peakMemory: Int {
        Memory.peakMemory
    }

    /// Return a snapshot of memory stats -- see ``Memory/Snapshot`` for more details.
    ///
    /// Get the current memory use.  This can be used to measure before/after and current memory use:
    ///
    /// ```swift
    /// let currentMemory = Memory.snapshot()
    /// print(currentMemory)
    /// ```
    @available(*, deprecated, renamed: "Memory.snapshot")
    public static func snapshot() -> Snapshot {
        Memory.snapshot()
    }

    /// Get the free cache limit.
    ///
    /// If using more than the given limit, free memory will be reclaimed
    /// from the cache on the next allocation.
    /// The cache limit defaults to the memory limit.
    ///
    /// ### See Also
    /// - ``Memory/cacheLimit``
    @available(*, deprecated, renamed: "Memory.cacheLimit")
    public static var cacheLimit: Int {
        Memory.cacheLimit
    }

    /// Set the free cache limit.
    ///
    /// If using more than the given limit, free memory will be reclaimed
    /// from the cache on the next allocation. To disable the cache,
    /// set the limit to 0.
    ///
    /// The cache limit defaults to the memory limit, which may allow very
    /// large cache sizes on systems with abundant RAM. For memory-constrained
    /// applications or to prevent excessive memory growth during long inference
    /// runs, consider setting a much lower cache limit.
    ///
    /// ## Performance Optimization
    ///
    /// The optimal cache size varies by workload. Many developers find that
    /// relatively small cache sizes (e.g., 2MB) perform just as well as
    /// unconstrained cache sizes for their specific use cases. Experiment
    /// with different values and measure performance to find the best setting
    /// for your workload.
    ///
    /// See ``Memory`` for more information on cache sizing and tuning.
    @available(*, deprecated, message: "use Memory.cacheLimit property")
    public static func set(cacheLimit: Int) {
        Memory.cacheLimit = cacheLimit
    }

    /// Get the memory limit.
    ///
    /// Calls to malloc will wait on scheduled tasks if the limit is exceeded. The
    /// memory limit defaults to 1.5 times the maximum recommended working set
    /// size reported by the device.
    ///
    /// ### See Also
    /// - ``Memory/memoryLimit``
    @available(*, deprecated, renamed: "Memory.memoryLimit")
    public static var memoryLimit: Int {
        Memory.memoryLimit
    }

    /// Set the memory limit.
    ///
    /// Calls to malloc will wait on scheduled tasks if the limit is exceeded.
    ///
    /// The memory limit defaults to 1.5 times the maximum recommended working set
    /// size reported by the device ([recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)).
    ///
    /// **Important**: This limit controls total MLX memory allocation. The cache limit
    /// (see ``Memory/cacheLimit``) defaults to this value, so systems with large memory
    /// limits may cache many GB of buffers. Consider setting a lower cache limit for
    /// memory-constrained applications.
    @available(*, deprecated, message: "use Memory.memoryLimit property")
    public static func set(memoryLimit: Int, relaxed: Bool = true) {
        Memory.memoryLimit = memoryLimit
    }

    /// Perform the block with a temporarily altered wired memory limit.
    ///
    /// Note: this manipulates a global value.  Nested calls will work as expected but
    /// concurrent calls cannot.
    ///
    /// See also ``DeviceInfo/maxRecommendedWorkingSetSize``.
    ///
    /// - Parameters:
    ///   - limit: new limit in bytes
    ///   - body: block to perform
    @available(*, deprecated, message: "Deprecated. Use WiredMemoryManager and tickets instead.")
    public static func withWiredLimit<R>(
        _ limit: Int, _ body: () throws -> R
    ) rethrows -> R {
        try Memory.withWiredLimit(limit, body)
    }

    /// Perform the block with a temporarily altered wired memory limit.
    ///
    /// Note: this manipulates a global value.  Nested calls will work as expected but
    /// concurrent calls cannot.
    ///
    /// See also ``DeviceInfo/maxRecommendedWorkingSetSize``.
    ///
    /// - Parameters:
    ///   - limit: new limit in bytes
    ///   - body: block to perform
    @available(*, deprecated, message: "Deprecated. Use WiredMemoryManager and tickets instead.")
    public static func withWiredLimit<R>(
        _ limit: Int, _ body: () async throws -> R
    ) async rethrows -> R {
        try await Memory.withWiredLimit(limit, body)
    }

    /// Cause all cached metal buffers to be deallocated.
    @available(*, deprecated, renamed: "Memory.clearCache")
    public static func clearCache() {
        Memory.clearCache()
    }

    /// Start capturing a metal trace into the given file.
    ///
    /// > There are several requirements for this to be used.
    ///
    /// - `mlx` must be built with `MLX_METAL_DEBUG`
    ///   - in Package.swift add `.define("MLX_METAL_DEBUG"),` to `Cmlx` `cxxSettings`
    /// - when running set the `MTL_CAPTURE_ENABLED=1` environment variable
    /// - make sure the file at the given path does not already exist
    ///
    /// See [the documentation](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)
    /// for more information.
    public static func startCapture(url: URL) {
        mlx_metal_start_capture(url.path().cString(using: .utf8))
    }

    /// Stop the metal capture.
    ///
    /// See ``startCapture(url:)``.
    public static func stopCapture(url: URL) {
        mlx_metal_stop_capture()
    }

    /// Reset the peak memory to zero.
    ///
    /// See ``Memory/Snapshot/peakMemory``.
    public static func resetPeakMemory() {
        mlx_reset_peak_memory()
    }

    public struct DeviceInfo: Sendable {
        public let architecture: String
        public let maxBufferSize: Int
        public let maxRecommendedWorkingSetSize: UInt64
        public let memorySize: Int
    }

    /// Returns Metal's recommended working set size in bytes as an `Int`.
    ///
    /// This value is derived from ``DeviceInfo/maxRecommendedWorkingSetSize`` and
    /// is clamped to `Int.max` when necessary. Returns `nil` when unavailable.
    public static func maxRecommendedWorkingSetBytes() -> Int? {
        let maxBytes = deviceInfo().maxRecommendedWorkingSetSize
        guard maxBytes > 0 else { return nil }
        if maxBytes > UInt64(Int.max) {
            return Int.max
        }
        return Int(maxBytes)
    }

    /// Get information about the GPU device and system settings
    public static func deviceInfo() -> DeviceInfo {
        var mib = [CTL_HW, HW_MEMSIZE]
        var memSize: size_t = 0
        var length: size_t = MemoryLayout.size(ofValue: memSize)
        sysctl(&mib, 2, &memSize, &length, nil, 0)

        if let device = MTLCreateSystemDefaultDevice() {
            let architecture: String
            if #available(macOS 14.0, iOS 17.0, tvOS 17.0, *) {
                architecture = device.architecture.name
            } else {
                architecture = device.name
            }

            return DeviceInfo(
                architecture: architecture, maxBufferSize: device.maxBufferLength,
                maxRecommendedWorkingSetSize: device.recommendedMaxWorkingSetSize,
                memorySize: memSize)
        } else {
            return DeviceInfo(
                architecture: "Unknown", maxBufferSize: 0, maxRecommendedWorkingSetSize: 0,
                memorySize: memSize)
        }
    }

    /// Runtime state for the opt-in Gemma 4 sorted expert-QMM specialization.
    ///
    /// Route counters change only while diagnostics are explicitly armed. The
    /// feature request and immutable runtime capabilities remain observable
    /// while diagnostics are unarmed.
    public struct Gemma4ExpertQMMDiagnostics: Equatable, Sendable {
        public let requested: Bool
        public let aotAvailable: Bool
        public let naxAvailable: Bool
        public let armed: Bool
        public let attempts: UInt64
        public let hits: UInt64
        public let fallbackNAX: UInt64
        public let fallbackOuterRoute: UInt64
        public let fallbackQuantization: UInt64
        public let fallbackTopology: UInt64
        public let fallbackAssignmentCount: UInt64
        public let fallbackGeometry: UInt64
        public let fallbackMetallibUnavailable: UInt64
        public let fallbackSortednessRetracted: UInt64

        /// Sum of all mutually exclusive fallback reasons.
        public var fallbacks: UInt64 {
            fallbackNAX + fallbackOuterRoute + fallbackQuantization
                + fallbackTopology + fallbackAssignmentCount + fallbackGeometry
                + fallbackMetallibUnavailable + fallbackSortednessRetracted
        }

        fileprivate init(_ value: mlx_metal_gemma4_expert_qmm_diagnostics) {
            requested = value.requested != 0
            aotAvailable = value.aot_available != 0
            naxAvailable = value.nax_available != 0
            armed = value.armed != 0
            attempts = value.attempts
            hits = value.hits
            fallbackNAX = value.fallback_nax
            fallbackOuterRoute = value.fallback_outer_route
            fallbackQuantization = value.fallback_quantization
            fallbackTopology = value.fallback_topology
            fallbackAssignmentCount = value.fallback_assignment_count
            fallbackGeometry = value.fallback_geometry
            fallbackMetallibUnavailable = value.fallback_metallib_unavailable
            fallbackSortednessRetracted = value.fallback_sortedness_retracted
        }
    }

    /// Snapshot Gemma 4 expert-QMM feature, AOT capability, and route counters
    /// without changing the armed state.
    public static func gemma4ExpertQMMDiagnostics() -> Gemma4ExpertQMMDiagnostics {
        var value = mlx_metal_gemma4_expert_qmm_diagnostics()
        mlx_metal_gemma4_expert_qmm_diagnostics_snapshot(&value)
        return Gemma4ExpertQMMDiagnostics(value)
    }

    /// Clear the route counters without changing the armed state.
    public static func resetGemma4ExpertQMMDiagnostics() {
        mlx_metal_gemma4_expert_qmm_diagnostics_reset()
    }

    /// Clear the route counters and arm accounting for a measured interval.
    /// Call only at an engine-idle boundary after warmup has completed.
    public static func clearAndArmGemma4ExpertQMMDiagnostics() {
        mlx_metal_gemma4_expert_qmm_diagnostics_clear_and_arm()
    }

    /// Snapshot the measured interval and disarm route accounting. Call only
    /// at an engine-idle boundary after measured work has completed.
    /// The returned `armed` value reports whether accounting was armed before
    /// this call.
    public static func snapshotAndDisarmGemma4ExpertQMMDiagnostics()
        -> Gemma4ExpertQMMDiagnostics
    {
        var value = mlx_metal_gemma4_expert_qmm_diagnostics()
        mlx_metal_gemma4_expert_qmm_diagnostics_snapshot_and_disarm(&value)
        return Gemma4ExpertQMMDiagnostics(value)
    }

}
