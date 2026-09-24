import Foundation
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class NemotronH35ProductionDefaultsTests: XCTestCase {
    func testOptimizedLosslessPathIsDefaultWithoutOverrides() throws {
        let keys = [
            "DARKBLOOM_NEMOTRON35_MTP_CAPTURE_VERIFY",
            "DARKBLOOM_NEMOTRON35_MTP_KV_ONLY_HISTORY",
            "DARKBLOOM_NEMOTRON35_MTP_MAX_DRAFT_TOKENS",
            "DARKBLOOM_NEMOTRON35_MTP_BATCHED_M1",
            "DARKBLOOM_NEMOTRON35_MTP_BATCHED_NORM",
            "DARKBLOOM_NEMOTRON35_MTP_WINDOW_SSM",
            "DARKBLOOM_NEMOTRON35_MTP_COMPILED_MOE",
            "DARKBLOOM_NEMOTRON35_MTP_BATCHED_ROUTER",
            "DARKBLOOM_NEMOTRON35_MTP_BATCHED_ATTENTION",
            "DARKBLOOM_NEMOTRON35_MTP_EXACT_MULTIROW_HEAD",
        ]
        XCTAssertTrue(keys.allSatisfy { ProcessInfo.processInfo.environment[$0] == nil },
            "Default test requires a clean process environment")
        let data = Data("""
        {"model_type":"nemotron_h","vocab_size":128,"hidden_size":64,"num_hidden_layers":1,
         "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,
         "mamba_num_heads":4,"mamba_head_dim":16,"ssm_state_size":16,
         "conv_kernel":4,"n_groups":2,"intermediate_size":128,
         "moe_intermediate_size":64,"moe_shared_expert_intermediate_size":128,
         "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
         "layers_block_type":["mamba"],"mtp_layers_block_type":["attention","moe"],
         "num_nextn_predict_layers":1,"mamba_ssm_cache_dtype":"float32"}
        """.utf8)
        let model = NemotronH35Model(try JSONDecoder().decode(NemotronH35Configuration.self, from: data))
        let assistant = NemotronH35MTPAssistant(target: model)
        XCTAssertEqual(assistant.requiredVerificationMode, .rectangularExact)
        XCTAssertEqual(assistant.maximumDraftTokens, 7)
        XCTAssertTrue(assistant.prefersBatchedRectangularAttention)
        XCTAssertTrue(NemotronMTPExecution.batchedM1)
        XCTAssertTrue(NemotronMTPExecution.batchedNorm)
        XCTAssertTrue(NemotronMTPExecution.windowSSM)
        XCTAssertTrue(NemotronMTPExecution.compiledMoE)
        XCTAssertTrue(NemotronMTPExecution.batchedRouter)
        XCTAssertTrue(NemotronMTPExecution.exactMultirowHead)
        XCTAssertTrue(PagedMTPBatchedColumns.enabled)
    }
}
