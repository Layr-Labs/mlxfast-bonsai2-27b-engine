import Foundation
import MLX

/// Ordinary target scoring only. This does not execute a sampler, scheduler
/// step, speculative verifier, or recurrent verification commit/rollback.
@_spi(Diagnostics)
public struct CBv2TeacherForcedScoreRequest: Sendable {
    public let promptTokens: [Int]
    public let continuation: [Int]
    /// Supplied from the verified model declaration and checked again against
    /// the first actual logit vector before any forced-token forward.
    public let vocabularySize: Int

    public init(promptTokens: [Int], continuation: [Int], vocabularySize: Int) throws {
        guard (1...32768).contains(promptTokens.count), (1...256).contains(continuation.count),
              (2...1_048_576).contains(vocabularySize) else {
            throw CBv2TeacherForcedScoreError.invalidBounds
        }
        guard promptTokens.allSatisfy({ $0 >= 0 && $0 < vocabularySize }),
              continuation.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw CBv2TeacherForcedScoreError.invalidToken
        }
        self.promptTokens = promptTokens
        self.continuation = continuation
        self.vocabularySize = vocabularySize
    }
}

@_spi(Diagnostics)
public enum CBv2TeacherForcedScoreError: Error {
    case invalidBounds, invalidToken, unexpectedLogitGeometry, incompleteObservation
}

@_spi(Diagnostics)
public struct CBv2TeacherForcedScore: Sendable, Codable, Equatable {
    public let index: Int
    public let contextLength: Int
    public let forcedToken: Int
    public let logitDType: String
    public let vocabularySize: Int
    public let argMaxID: Int
    public let topTwoIDs: [Int]
    public let argMaxValueBits: UInt32
    public let topTwoValueBits: [UInt32]
    public let forcedTokenValueBits: UInt32
    public let logSumExpBits: UInt32
    public let forcedLogProbabilityBits: UInt32
    public let nllBits: UInt32
    public let topTwoMarginBits: UInt32
    public let nanCount: Int
    public let infiniteCount: Int

    public var isFinite: Bool {
        nanCount == 0 && infiniteCount == 0 &&
            ([argMaxValueBits, forcedTokenValueBits, logSumExpBits,
              forcedLogProbabilityBits, nllBits, topTwoMarginBits] + topTwoValueBits)
                .allSatisfy { Float(bitPattern: $0).isFinite }
    }
}

@_spi(Diagnostics)
public struct CBv2TeacherForcedScoreSnapshot: Sendable, Codable, Equatable {
    public let top1: [Int]
    public let records: [CBv2TeacherForcedScore]
    public var allFinite: Bool { !records.isEmpty && records.allSatisfy(\.isFinite) }
}

/// Call-owned compact reductions. No full logit vector, cache or model state
/// is retained once these arrays finish evaluation; no production binding.
final class CBv2TeacherForcedScoreCollector {
    struct Packet {
        let index: Int
        let contextLength: Int
        let forcedToken: Int
        let logitDType: String
        let vocabularySize: Int
        let ids: MLXArray
        let values: MLXArray
        var evaluationTargets: [MLXArray] { [ids, values] }

        init(logits: MLXArray, index: Int, contextLength: Int, forcedToken: Int,
             vocabularySize: Int) throws {
            guard logits.ndim == 1, logits.size == vocabularySize, vocabularySize >= 2,
                  [.float16, .bfloat16, .float32].contains(logits.dtype),
                  forcedToken >= 0, forcedToken < vocabularySize else {
                throw CBv2TeacherForcedScoreError.unexpectedLogitGeometry
            }
            self.index = index
            self.contextLength = contextLength
            self.forcedToken = forcedToken
            self.logitDType = String(describing: logits.dtype)
            self.vocabularySize = vocabularySize
            let topTwo = cbv2TopTwoRows(logits.reshaped([1, 1, vocabularySize]))
            let reduced = CBv2LogitDiagnosticPacket.reduce(
                logits: logits, topTwo: topTwo, candidateIDs: [forcedToken])
            ids = reduced.ids
            // Observational FP32 reductions over unchanged native logits.
            let normalizer = logSumExp(logits.asType(.float32)).reshaped([1])
            let forced = reduced.values[3..<4]
            values = concatenated([reduced.values, normalizer, forced - normalizer,
                normalizer - forced, reduced.values[1..<2] - reduced.values[2..<3]])
        }

        func materialize() -> CBv2TeacherForcedScore {
            let i = ids.asArray(Int32.self).map(Int.init)
            let v = values.asArray(Float.self).map(\.bitPattern)
            return CBv2TeacherForcedScore(index: index, contextLength: contextLength,
                forcedToken: forcedToken, logitDType: logitDType, vocabularySize: vocabularySize,
                argMaxID: i[0], topTwoIDs: Array(i[1...2]), argMaxValueBits: v[0],
                topTwoValueBits: Array(v[1...2]), forcedTokenValueBits: v[3],
                logSumExpBits: v[4], forcedLogProbabilityBits: v[5], nllBits: v[6],
                topTwoMarginBits: v[7], nanCount: i[3], infiniteCount: i[4])
        }
    }

    let request: CBv2TeacherForcedScoreRequest
    private var packets: [Packet] = []
    private(set) var snapshot: CBv2TeacherForcedScoreSnapshot?
    init(_ request: CBv2TeacherForcedScoreRequest) { self.request = request }

    func capture(logits: MLXArray, index: Int) throws -> [MLXArray] {
        guard index == packets.count, index < request.continuation.count else {
            throw CBv2TeacherForcedScoreError.incompleteObservation
        }
        let packet = try Packet(logits: logits.reshaped([-1]), index: index,
            contextLength: request.promptTokens.count + index,
            forcedToken: request.continuation[index], vocabularySize: request.vocabularySize)
        packets.append(packet)
        return packet.evaluationTargets
    }

    func finish(top1: [Int]) throws {
        defer { packets.removeAll(keepingCapacity: false) }
        guard packets.count == request.continuation.count, top1.count == packets.count else {
            throw CBv2TeacherForcedScoreError.incompleteObservation
        }
        // Every reduction already rode its forward's asyncEval submission.
        // Finish that bounded observation before readback and row retirement.
        eval(packets.flatMap(\.evaluationTargets))
        let records = packets.map { $0.materialize() }
        // Preserve a discrepant independent reduction as evidence. The caller
        // must classify its IDs versus the normal top1 result as inconclusive.
        snapshot = CBv2TeacherForcedScoreSnapshot(top1: top1, records: records)
    }
}
