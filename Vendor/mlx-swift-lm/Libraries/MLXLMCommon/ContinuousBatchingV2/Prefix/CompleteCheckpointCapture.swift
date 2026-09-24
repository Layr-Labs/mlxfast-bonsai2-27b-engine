import Foundation
import MLX

/// Request-local capture and retirement, with no resident index or idle arrays.
/// The engine queue owns staged; the retirement queue owns detached payloads.
final class CBv2CompleteCheckpointCapture: @unchecked Sendable {
    let codec: CBv2CompleteCheckpointCodec
    let store: any CBv2CompletePrefixCache
    let queue = DispatchQueue(label: "cbv2.complete-checkpoint-retirement", qos: .utility)
    var staged: [CBv2RequestID: [CBv2CapturedCompleteCheckpoint]] = [:]
    // Deterministic native construction/evaluation fault seam, engine-queue
    // only. Production always uses the ordinary private historical owner.
    var makeHistoricalWindow: (PagedSequenceKV, Int, AdmissionV2) throws -> CBv2HistoricalWindow = {
        try .init(row: $0, position: $1, admission: $2)
    }
    private let handlerLock = NSLock()
    private var publicationHandler: (@Sendable (CBv2RequestID, [Int]) -> Void)?
    private var closed = false

    init(codec: CBv2CompleteCheckpointCodec, store: any CBv2CompletePrefixCache) {
        self.codec = codec
        self.store = store
    }

    func setPublicationHandler(_ handler: (@Sendable (CBv2RequestID, [Int]) -> Void)?) {
        handlerLock.lock()
        publicationHandler = closed ? nil : handler
        handlerLock.unlock()
    }

    func close() {
        handlerLock.lock()
        closed = true
        publicationHandler = nil
        handlerLock.unlock()
    }

    var isClosed: Bool {
        handlerLock.lock()
        defer { handlerLock.unlock() }
        return closed
    }

    func reportPublication(receiptID: CBv2RequestID?, positions: [Int]) {
        guard let receiptID, !positions.isEmpty else { return }
        handlerLock.lock()
        let handler = publicationHandler
        handlerLock.unlock()
        handler?(receiptID, positions)
    }

    func hasCheckpoints(requestID: CBv2RequestID) -> Bool {
        staged[requestID]?.isEmpty == false
    }

    /// Reserve before any checkpoint copy graph is constructed. The extra
    /// allowance covers allocator padding, concatenate/copy intermediates and
    /// entire retained SSM backing; it survives until every alias retires.
    func capture(
        requestID: CBv2RequestID, position: Int, chunkSize: Int,
        layers: [Int: CBv2RecurrentLayerState], assistantState: (any CBv2MTPRequestState)?
    ) -> [MLXArray] {
        guard !isClosed, position > 1, chunkSize > 1, position % chunkSize == 0,
            !(staged[requestID]?.contains { $0.checkpoint?.position == position } ?? false)
        else { return [] }
        do {
            let descriptors = try codec.tensorDescriptors(position: position)
            var packedBytes = 0
            for descriptor in descriptors {
                let (next, overflow) = packedBytes.addingReportingOverflow(descriptor.byteCount)
                guard !overflow else { return [] }
                packedBytes = next
            }
            guard store.acceptsCheckpoint(position: position, packedBytes: packedBytes) else { return [] }
            let stateDescriptors = descriptors.filter { $0.role != .keys && $0.role != .values }
            for spec in codec.recurrentSpec?.layers ?? [] {
                guard let state = layers[spec.modelLayerIndex],
                    let conv = state.conv, let ssm = state.ssm,
                    conv.shape == spec.convShape, conv.dtype == spec.convDType,
                    ssm.shape == spec.ssmShape, ssm.dtype == spec.ssmDType
                else { return [] }
            }
            let bytes = try CBv2CheckpointAllocationFootprint.captureBytes(stateDescriptors, layers: layers)
            let reservation = try codec.admission.reserveTransient(bytes: bytes)
            let checkpoint = try withError { error in
                var assistant: (any CBv2MTPPrefixCheckpoint)?
                if let drafter = codec.assistant {
                    guard let assistantState,
                        let captured = drafter.capturePrefixCheckpoint(requestState: assistantState, targetInputCount: position)
                    else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
                    assistant = captured
                }
                try error.check()
                var copies: [Int: CBv2RecurrentLayerState] = [:]
                for spec in codec.recurrentSpec?.layers ?? [] {
                    let layer = layers[spec.modelLayerIndex]!
                    copies[spec.modelLayerIndex] = .init(
                        conv: MLX.where(MLXArray(true), layer.conv!, layer.conv!), ssm: layer.ssm)
                }
                try error.check()
                return CBv2RecurrentCheckpoint(
                    position: position, chunkSize: chunkSize, layers: copies,
                    byteCount: stateDescriptors.reduce(0) { $0 + $1.byteCount }, assistant: assistant)
            }
            let captured = CBv2CapturedCompleteCheckpoint(checkpoint: checkpoint, reservation: reservation)
            if staged[requestID, default: []].count == 2 {
                let previous = staged[requestID]!.removeLast()
                queue.async { previous.finishEvaluationAndClose() }
            }
            staged[requestID, default: []].append(captured)
            return checkpoint.evaluationRoots
        } catch {
            return []
        }
    }

    /// A final queued drop follows that request's rolling retirement copies.
    /// The engine counts this callback in its existing shutdown drain barrier.
    func drop(requestID: CBv2RequestID, completion: @escaping @Sendable () -> Void) -> Bool {
        guard let captures = staged.removeValue(forKey: requestID) else { return false }
        queue.async {
            captures.forEach { $0.finishEvaluationAndClose() }
            completion()
        }
        return true
    }

    /// Always completes. Call on the engine queue with retired KV owners;
    /// their backend/request reservation remains live until completion.
    func publish(
        intent: CBv2DonationIntent,
        state: [CBv2SequenceKV?],
        completion: @escaping @Sendable ([Int]) -> Void
    ) {
        let captures = staged.removeValue(forKey: intent.requestID) ?? []
        var exports: [CBv2CompleteCheckpointExport] = []
        for capture in captures where intent.allowsCompletePublication {
            if let checkpoint = capture.historical {
                if let source = try? codec.exportHistorical(
                    checkpoint: checkpoint, state: state, tokens: intent.tokens, cacheSalt: intent.cacheSalt) {
                    exports.append(source)
                }
            } else if let checkpoint = capture.checkpoint {
                if let source = try? codec.export(
                    checkpoint: checkpoint, state: state, tokens: intent.tokens, cacheSalt: intent.cacheSalt) {
                    exports.append(source)
                }
            }
        }
        let batch = CBv2CompleteCheckpointPublication(
            captures: captures, exports: exports, receiptID: intent.receiptID,
            tokens: intent.tokens, cacheSalt: intent.cacheSalt, completion: completion)
        queue.async { [self] in
            guard !isClosed, batch.prepare(),
                let scratch = try? codec.admission.reserveTransient(
                    bytes: codec.exportScratchBytes + (codec.admission.hasProcessMemoryOwner
                        ? 0 : CBv2CompleteCheckpointManifest.maximumProviderScratchBytes))
            else { batch.close(); return }
            batch.scratch = scratch
            publishNext(batch)
        }
    }

    private func publishNext(_ batch: CBv2CompleteCheckpointPublication) {
        guard !isClosed, let source = batch.nextSource else { batch.close(); return }
        store.donate(
            source, requestID: batch.receiptID, tokens: batch.tokens, cacheSalt: batch.cacheSalt
        ) { [self, batch, source] positions in
            source.close()
            queue.async { [self, batch, source] in
                if batch.didPublish(source: source, positions: positions) { publishNext(batch) }
            }
        }
    }
}

final class CBv2CapturedCompleteCheckpoint: @unchecked Sendable {
    private(set) var checkpoint: CBv2RecurrentCheckpoint?
    private(set) var historical: CBv2HistoricalCompleteCheckpoint?
    var evaluationRoots: [MLXArray] { checkpoint?.evaluationRoots ?? historical?.evaluationRoots ?? [] }
    var position: Int? { checkpoint?.position ?? historical?.position }
    private var reservation: CBv2CheckpointReservation?

    init(checkpoint: CBv2RecurrentCheckpoint, reservation: CBv2CheckpointReservation) {
        self.checkpoint = checkpoint
        self.reservation = reservation
    }

    init(historical: CBv2HistoricalCompleteCheckpoint) { self.historical = historical }

    func finishEvaluation() throws {
        if let historical { try historical.finishEvaluation() }
        else { try withError { eval(evaluationRoots) } }
    }

    func finishEvaluationAndClose() {
        try? finishEvaluation()
        historical = nil
        checkpoint = nil
        reservation?.release()
        reservation = nil
    }
}

/// Serial-queue publication state; array aliases are cleared before completion.
private final class CBv2CompleteCheckpointPublication: @unchecked Sendable {
    private var captures: [CBv2CapturedCompleteCheckpoint]
    private var exports: [CBv2CompleteCheckpointExport]
    private var completedPositions: [Int] = []
    private var cursor = 0
    private var completion: (@Sendable ([Int]) -> Void)?
    let receiptID: CBv2RequestID?
    let tokens: [Int]
    let cacheSalt: String?
    var scratch: CBv2CheckpointReservation?

    init(
        captures: [CBv2CapturedCompleteCheckpoint], exports: [CBv2CompleteCheckpointExport],
        receiptID: CBv2RequestID?, tokens: [Int], cacheSalt: String?,
        completion: @escaping @Sendable ([Int]) -> Void
    ) {
        self.captures = captures
        self.exports = exports
        self.receiptID = receiptID
        self.tokens = tokens
        self.cacheSalt = cacheSalt
        self.completion = completion
    }

    func prepare() -> Bool {
        do {
            for capture in captures {
                try capture.finishEvaluation()
            }
            return !exports.isEmpty
        } catch { return false }
    }

    var nextSource: CBv2CompleteCheckpointExport? { cursor < exports.count ? exports[cursor] : nil }

    func didPublish(source: CBv2CompleteCheckpointExport, positions: [Int]) -> Bool {
        guard completion != nil, nextSource === source else { return false }
        // The provider cannot claim a boundary different from this file.
        if positions.contains(source.manifest.position) {
            completedPositions.append(source.manifest.position)
        }
        cursor += 1
        return true
    }

    func close() {
        exports.forEach { $0.close() }
        exports.removeAll()
        captures.forEach { $0.finishEvaluationAndClose() }
        captures.removeAll()
        scratch?.release()
        scratch = nil
        let callback = completion
        completion = nil
        callback?(completedPositions)
    }
}
