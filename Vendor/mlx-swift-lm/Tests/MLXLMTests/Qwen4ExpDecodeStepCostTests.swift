import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXRunners
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// The decode step's FIXED cost, measured with negligible weights.
///
/// A real-shaped Qwen4Exp (48 layers, 24/2 heads × 256, GDN 16/48 × 128,
/// hc 4, 512 experts top-10, PLE, QSA indexer) with tiny hidden and expert
/// widths decodes at ~71 ms/step on a laptop in release — the bytes are
/// nothing, so that is the per-layer kernel/graph count (~1.3 ms/layer),
/// the same overhead the 125B pays on the box before its ~6 GB/token of
/// weights. This test prints the number; it asserts nothing, and runs only
/// under MLXLM_STEP_COST=1 with a full metallib. Knobs: ZZ_LAYERS,
/// ZZ_EXPERTS, ZZ_TOPK, ZZ_INTERVAL, ZZ_QUANT, ZZ_HIDDEN.
final class Qwen4ExpDecodeStepCostTests: XCTestCase {
    func testDecodeStepFixedCost() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MLXLM_STEP_COST"] == "1" else {
            throw XCTSkip(
                "Set MLXLM_STEP_COST=1 (and MLXLM_FULL_AOT_METALLIB=1) to measure the decode step cost"
            )
        }
        var json = Qwen4ExpFixture.configurationJSON
        for (a, b) in [
            ("\"hidden_size\": 32", "\"hidden_size\": \(env["ZZ_HIDDEN"] ?? "128")"),
            ("\"num_attention_heads\": 4", "\"num_attention_heads\": 24"),
            ("\"head_dim\": 8", "\"head_dim\": 256"),
            ("\"indexer_budget\": \(Qwen4ExpFixture.indexerBudget)", "\"indexer_budget\": 2048"),
            ("\"indexer_compress_ratio\": 2", "\"indexer_compress_ratio\": 4"),
            ("\"indexer_head_dim\": 8", "\"indexer_head_dim\": 128"),
            ("\"ple_layer_ids\": [1]", "\"ple_layer_ids\": [2]"),
            (
                "\"full_attention_interval\": 2",
                "\"full_attention_interval\": \(env["ZZ_INTERVAL"] ?? "4")"
            ),
            ("\"num_hidden_layers\": 4", "\"num_hidden_layers\": \(env["ZZ_LAYERS"] ?? "48")"),
            ("\"num_experts\": 4", "\"num_experts\": \(env["ZZ_EXPERTS"] ?? "512")"),
            ("\"num_experts_per_tok\": 2", "\"num_experts_per_tok\": \(env["ZZ_TOPK"] ?? "10")"),
            ("\"moe_intermediate_size\": 16", "\"moe_intermediate_size\": 64"),
            ("\"shared_expert_intermediate_size\": 16", "\"shared_expert_intermediate_size\": 64"),
            ("\"linear_num_key_heads\": 2", "\"linear_num_key_heads\": 16"),
            ("\"linear_num_value_heads\": 4", "\"linear_num_value_heads\": 48"),
            ("\"linear_key_head_dim\": 32", "\"linear_key_head_dim\": 128"),
            ("\"linear_value_head_dim\": 32", "\"linear_value_head_dim\": 128"),
            ("\"hc_count\": 2", "\"hc_count\": 4"),
        ] {
            precondition(json.contains(a), a)
            json = json.replacingOccurrences(of: a, with: b)
        }
        let config = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(7)
        let model = Qwen4ExpModel(text: config, withMTP: false)
        eval(model)
        let cast = model.parameters().flattened().map {
            ($0.0, $0.1.dtype == .float32 ? $0.1.asType(.bfloat16) : $0.1)
        }
        model.update(parameters: ModuleParameters.unflattened(cast))
        eval(model)
        if env["ZZ_QUANT"] == "1" {
            quantize(model: model) { _, m in
                if let l = m as? Linear, l.weight.dim(-1) % 64 == 0 { return (64, 4, .affine) }
                if let e = m as? Embedding, e.weight.dim(-1) % 64 == 0 { return (64, 4, .affine) }
                if let s = m as? SwitchLinear, s.weight.dim(-1) % 64 == 0 {
                    return (64, 4, .affine)
                }
                return nil
            }
            eval(model)
        }
        model.install(ngramRowSource: DeterministicNGramRowSource(rowDimensions: 2))
        let prompt: [Int] = (0 ..< 1024).map { ($0 * 37 + 11) % 64 }
        let newCaches:
            ((_ layerIndex: Int, _ kind: CBv2LayerKind) throws -> any CBv2AttendingLayerCache)
                throws -> [any CBv2AttendingLayerCache] = { make in
                    try model.newCacheV2(makeLayerCache: make)
                }
        let stepper = CBv2SingleRowStepper(
            model: model, layerKinds: model.cbv2LayerKinds, newCaches: newCaches,
            kvBytesCapacity: 1 << 28, maxLength: 4096)
        try stepper.begin()
        let t0 = DispatchTime.now().uptimeNanoseconds
        var tok = try stepper.forward(prompt).argmax
        let prefillMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        var times: [Double] = []
        for _ in 0 ..< 12 {
            let s = DispatchTime.now().uptimeNanoseconds
            tok = try stepper.forward([tok]).argmax
            times.append(Double(DispatchTime.now().uptimeNanoseconds - s) / 1e6)
        }
        let steady = times.dropFirst(2).sorted()
        let tag =
            "interval=\(env["ZZ_INTERVAL"] ?? "4") topk=\(env["ZZ_TOPK"] ?? "10") layers=\(env["ZZ_LAYERS"] ?? "48") experts=\(env["ZZ_EXPERTS"] ?? "512") quant=\(env["ZZ_QUANT"] ?? "0")"
        print(
            "STEPCOST \(tag) prefill(1024)=\(String(format: "%.1f", prefillMs)) ms decode p50=\(String(format: "%.1f", steady[steady.count / 2])) min=\(String(format: "%.1f", steady.first!)) max=\(String(format: "%.1f", steady.last!))"
        )
    }
}
