import Foundation

extension EngineLoopV2 {
    func beginForwardShapeStep() -> CBv2ForwardShapeStep? {
        precondition(buildingForwardShapes == nil)
        let step = forwardShapeRecorder?.beginStep()
        buildingForwardShapes = step
        return step
    }

    func endForwardShapeStep(_ step: CBv2ForwardShapeStep?) {
        buildingForwardShapes = nil
        step?.finishBuilding()
    }

    func beginForwardShapeObservation() throws -> CBv2ForwardShapeSnapshot {
        try onEngineQueueSync {
            try requireIdleForwardShapeBoundary()
            let recorder = forwardShapeRecorder ?? CBv2ForwardShapeRecorder()
            try recorder.reset()
            forwardShapeRecorder = recorder
            return recorder.snapshot()
        }
    }

    func forwardShapeSnapshot() -> CBv2ForwardShapeSnapshot {
        onEngineQueueSync { forwardShapeRecorder?.snapshot() ?? .disabled }
    }
}
