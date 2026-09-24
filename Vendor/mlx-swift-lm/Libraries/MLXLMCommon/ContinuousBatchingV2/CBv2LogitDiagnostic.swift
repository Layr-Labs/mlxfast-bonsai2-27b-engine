// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Explicit offline observation of one actual generated position. This does
/// not change sampling, verification strategy, or model inputs.
@_spi(Diagnostics)
public struct CBv2LogitDiagnosticConfig: Sendable, Encodable {
    public let requestID: UInt64
    public let outputIndex: Int
    public let candidateIDs: [Int]
    public let maximumRecords: Int

    public init(
        requestID: UInt64, outputIndex: Int, candidateIDs: [Int], maximumRecords: Int = 8
    ) throws {
        guard outputIndex >= 0, outputIndex <= 1_000_000,
            (1...2).contains(candidateIDs.count), Set(candidateIDs).count == candidateIDs.count,
            candidateIDs.allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) }),
            (1...16).contains(maximumRecords)
        else { throw CBv2LogitDiagnosticError.invalidConfiguration }
        self.requestID = requestID
        self.outputIndex = outputIndex
        self.candidateIDs = candidateIDs
        self.maximumRecords = maximumRecords
    }
}

@_spi(Diagnostics)
public enum CBv2LogitDiagnosticError: Error {
    case invalidConfiguration, engineBusy
}

/// Float32 bits preserve NaN/Inf and signed zero without invalid JSON numbers.
@_spi(Diagnostics)
public struct CBv2LogitDiagnosticRecord: Sendable, Encodable {
    public let requestID: UInt64
    public let outputIndex: Int
    public let outputBase: Int
    public let phase: String
    public let batchIndex: Int
    public let batchSize: Int
    public let column: Int
    public let verificationWidth: Int
    public let draftDepth: Int
    public let cacheOffset: Int
    public let backend: String
    public let logitDType: String
    public let vocabularySize: Int
    public let seedToken: Int?
    public let outcome: String
    public let acceptedDrafts: Int
    public let confirmedWidth: Int
    public let draftPrefix: [Int]
    public let targetToken: Int?
    public let argMaxID: Int
    public let topTwoIDs: [Int]
    public let candidateIDs: [Int]
    /// argMax value, two fused top-two values, then candidate values.
    public let valueBits: [UInt32]
    public let nanCount: Int
    public let infiniteCount: Int
}

@_spi(Diagnostics)
public struct CBv2LogitDiagnosticSnapshot: Sendable, Encodable {
    public let configuration: CBv2LogitDiagnosticConfig
    public let records: [CBv2LogitDiagnosticRecord]
    public let reservedRecords: Int
    public let omittedRecords: Int
    public let invalidVocabularyRecords: Int
    public let hostMaterializationNanoseconds: UInt64
}

/// Engine-queue-owned, finite host storage. Draining never refills its budget.
final class CBv2LogitDiagnosticState {
    let configuration: CBv2LogitDiagnosticConfig
    private(set) var reservedRecords = 0
    private(set) var omittedRecords = 0
    private(set) var invalidVocabularyRecords = 0
    private(set) var records: [CBv2LogitDiagnosticRecord] = []
    var hostMaterializationNanoseconds: UInt64 = 0

    init(_ configuration: CBv2LogitDiagnosticConfig) { self.configuration = configuration }

    func mayCapture(requestID: CBv2RequestID, outputIndex: Int) -> Bool {
        guard requestID.raw == configuration.requestID,
            outputIndex == configuration.outputIndex else { return false }
        guard reservedRecords < configuration.maximumRecords else {
            omittedRecords += 1
            return false
        }
        return true
    }

    func reserve(requestID: CBv2RequestID, outputIndex: Int, vocabularySize: Int) -> Bool {
        guard requestID.raw == configuration.requestID,
            outputIndex == configuration.outputIndex else { return false }
        guard configuration.candidateIDs.allSatisfy({ $0 < vocabularySize }) else {
            invalidVocabularyRecords += 1
            return false
        }
        guard reservedRecords < configuration.maximumRecords else {
            omittedRecords += 1
            return false
        }
        reservedRecords += 1
        return true
    }

    func append(_ record: CBv2LogitDiagnosticRecord) {
        precondition(records.count < configuration.maximumRecords)
        records.append(record)
    }

    func takeSnapshot() -> CBv2LogitDiagnosticSnapshot {
        let result = CBv2LogitDiagnosticSnapshot(
            configuration: configuration, records: records,
            reservedRecords: reservedRecords, omittedRecords: omittedRecords,
            invalidVocabularyRecords: invalidVocabularyRecords,
            hostMaterializationNanoseconds: hostMaterializationNanoseconds)
        records.removeAll(keepingCapacity: false)
        return result
    }
}

/// Only compact lazy reductions survive the forward. No full logit vector,
/// model cache, or recurrent state is retained by this packet after its fence.
final class CBv2LogitDiagnosticPacket {
    let requestID: CBv2RequestID
    let outputIndex: Int
    let phase: String
    let batchIndex: Int
    let batchSize: Int
    let column: Int
    let verificationWidth: Int
    let draftDepth: Int
    let cacheOffset: Int
    let backend: String
    let logitDType: String
    let vocabularySize: Int
    let candidateIDs: [Int]
    let ids: MLXArray
    let values: MLXArray
    var seedToken: Int?
    private(set) var outcome = "discarded"
    private(set) var acceptedDrafts = 0
    private(set) var confirmedWidth = 0
    private(set) var draftPrefix: [Int] = []
    private(set) var targetToken: Int?

    var evaluationTargets: [MLXArray] { [ids, values] }

    init(
        requestID: CBv2RequestID, outputIndex: Int, phase: String,
        batchIndex: Int, batchSize: Int, column: Int, verificationWidth: Int,
        draftDepth: Int, seedToken: Int?, cacheOffset: Int, backend: String,
        logitDType: String, vocabularySize: Int, candidateIDs: [Int],
        ids: MLXArray, values: MLXArray
    ) {
        self.requestID = requestID
        self.outputIndex = outputIndex
        self.phase = phase
        self.batchIndex = batchIndex
        self.batchSize = batchSize
        self.column = column
        self.verificationWidth = verificationWidth
        self.draftDepth = draftDepth
        self.seedToken = seedToken
        self.cacheOffset = cacheOffset
        self.backend = backend
        self.logitDType = logitDType
        self.vocabularySize = vocabularySize
        self.candidateIDs = candidateIDs
        self.ids = ids
        self.values = values
    }

    /// A column after the first rejection uses an uncommitted draft history.
    /// A truncated column may share history but was not emitted.
    func reconcile(accepted: Int, confirmed: Int, drafts: [Int], targets: [Int]) {
        acceptedDrafts = accepted
        confirmedWidth = confirmed
        draftPrefix = Array(drafts.prefix(column))
        targetToken = targets.indices.contains(column) ? targets[column] : nil
        outcome = column < confirmed ? "confirmed"
            : column > accepted ? "speculative_suffix" : "truncated"
    }

    static func reduce(
        logits: MLXArray, topTwo: (ids: MLXArray, values: MLXArray), candidateIDs: [Int]
    ) -> (ids: MLXArray, values: MLXArray) {
        let independent = argMax(logits, axis: -1).asType(.int32).reshaped([1])
        let candidates = MLXArray(candidateIDs.map(Int32.init))
        let ids = concatenated([
            independent, topTwo.ids.reshaped([2]).asType(.int32),
            sum(isNaN(logits).asType(.int32)).reshaped([1]),
            sum(isInf(logits).asType(.int32)).reshaped([1]),
        ])
        let values = concatenated([
            takeAlong(logits, independent, axis: 0).asType(.float32),
            topTwo.values.reshaped([2]).asType(.float32),
            takeAlong(logits, candidates, axis: 0).asType(.float32),
        ])
        return (ids, values)
    }

    /// Called only after this packet rode the step's existing evaluation fence
    /// and after the existing adaptive step-cost timestamp. No eval/asyncEval.
    func materialize() -> CBv2LogitDiagnosticRecord {
        let hostIDs = ids.asArray(Int32.self)
        let hostValues = values.asArray(Float.self)
        return CBv2LogitDiagnosticRecord(
            requestID: requestID.raw, outputIndex: outputIndex, outputBase: outputIndex - column,
            phase: phase, batchIndex: batchIndex, batchSize: batchSize, column: column,
            verificationWidth: verificationWidth, draftDepth: draftDepth, cacheOffset: cacheOffset,
            backend: backend, logitDType: logitDType, vocabularySize: vocabularySize,
            seedToken: seedToken, outcome: outcome, acceptedDrafts: acceptedDrafts,
            confirmedWidth: confirmedWidth, draftPrefix: draftPrefix, targetToken: targetToken,
            argMaxID: Int(hostIDs[0]), topTwoIDs: hostIDs[1...2].map(Int.init),
            candidateIDs: candidateIDs, valueBits: hostValues.map(\.bitPattern),
            nanCount: Int(hostIDs[3]), infiniteCount: Int(hostIDs[4]))
    }
}

extension EngineLoopV2 {
    func makeLogitDiagnostic(
        logits makeLogits: @autoclosure () -> MLXArray, requestID: CBv2RequestID, outputIndex: Int, phase: String,
        batchIndex: Int, batchSize: Int, column: Int = 0, verificationWidth: Int = 1,
        draftDepth: Int = 0, seedToken: Int?, cacheOffset: Int,
        policyTopTwo: (ids: MLXArray, values: MLXArray)? = nil
    ) -> CBv2LogitDiagnosticPacket? {
        guard let diagnostic = logitDiagnostic,
            diagnostic.mayCapture(requestID: requestID, outputIndex: outputIndex)
        else { return nil }
        let logits = makeLogits()
        guard diagnostic.reserve(
            requestID: requestID, outputIndex: outputIndex, vocabularySize: logits.size)
        else { return nil }
        precondition(logits.ndim == 1 && logits.size >= 2)
        let topTwo: (ids: MLXArray, values: MLXArray)
        if let policyTopTwo { topTwo = policyTopTwo }
        else {
            // Observation does not advertise or require an MTP policy capability.
            let captured = cbv2TopTwoRows(logits.reshaped([1, 1, logits.size]))
            topTwo = (captured.ids.reshaped([2]), captured.values.reshaped([2]))
        }
        let (ids, values) = CBv2LogitDiagnosticPacket.reduce(
            logits: logits, topTwo: topTwo, candidateIDs: diagnostic.configuration.candidateIDs)
        return CBv2LogitDiagnosticPacket(
            requestID: requestID, outputIndex: outputIndex, phase: phase,
            batchIndex: batchIndex, batchSize: batchSize, column: column,
            verificationWidth: verificationWidth, draftDepth: draftDepth, seedToken: seedToken,
            cacheOffset: cacheOffset, backend: String(describing: type(of: backend)),
            logitDType: String(describing: logits.dtype), vocabularySize: logits.size,
            candidateIDs: diagnostic.configuration.candidateIDs, ids: ids, values: values)
    }

    func materializeLogitDiagnostics(_ step: CBv2InFlightStep) {
        guard !step.logitDiagnostics.isEmpty else { return }
        defer { step.logitDiagnostics.removeAll(keepingCapacity: false) }
        guard let diagnostic = logitDiagnostic else { return }
        let started = DispatchTime.now().uptimeNanoseconds
        for packet in step.logitDiagnostics { diagnostic.append(packet.materialize()) }
        diagnostic.hostMaterializationNanoseconds += DispatchTime.now().uptimeNanoseconds - started
    }
}
