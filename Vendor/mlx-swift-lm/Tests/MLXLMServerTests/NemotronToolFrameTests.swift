import MLXLMCommon
import MLXLMServer
import Testing

@Suite("Nemotron native XML frames")
struct NemotronToolFrameTests {
    @Test func pairedWrappersAndBareFunctionsSurviveEveryChunkBoundary() {
        let text = "<tool_call><function=one><parameter=a>19</parameter></function></tool_call>"
            + "<function=two><parameter=b>23</parameter></function>"
        for stride in [1, 2, 7, text.count] {
            let handler = BatchedToolStreamHandler(format: .nemotron, tools: nil)
            let chars = Array(text)
            var visible = ""
            for start in Swift.stride(from: 0, to: chars.count, by: stride) {
                visible += handler.processChunk(String(chars[start..<min(chars.count, start + stride)])) ?? ""
            }
            let calls = handler.finish()
            #expect(visible.isEmpty)
            #expect(calls.map(\.function.name) == ["one", "two"])
            #expect(handler.parseFailureCount == 0)
        }
    }

    @Test func incompleteAndAmbiguousParametersNeverBecomeCalls() {
        for text in [
            "<function=add><parameter=a>19</parameter>",
            "<function=add><parameter=a>19</function>",
            "<function=add><parameter=a>19</parameter><parameter=a>23</parameter></function>",
        ] {
            let handler = BatchedToolStreamHandler(format: .nemotron, tools: nil)
            _ = handler.processChunk(text)
            #expect(handler.finish().isEmpty)
        }
    }

    @Test func strictLegacyXMLAndBareJSONBehaviorStayUnchanged() {
        let strict = BatchedToolStreamHandler(format: .xmlFunction, tools: nil)
        let text = "<function=add><parameter=a>19</parameter></function>"
        _ = strict.processChunk(text)
        #expect(strict.finish().isEmpty)
        let native = BatchedToolStreamHandler(format: .nemotron, tools: nil)
        _ = native.processChunk(#"{"name":"add","arguments":{"a":19}}"#)
        #expect(native.finish().isEmpty)
    }
}
