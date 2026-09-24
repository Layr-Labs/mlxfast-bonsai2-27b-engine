// Qwen4ExpNGramResourceTests.swift
//
// The runner side of the n-gram row source: what `RunnerLoadOptions.resources`
// may hold under `Qwen4ExpRunner.ngramRowSourceResource`, and what each shape
// does.
//
// The resource has ONE path shape, the DIRECTORY of n-gram shard files that
// the offline transform writes, which is also what `RunnerResourceArguments`
// boxes for `--resource <name>=<path>`. A path to a single file is refused by
// name.
//
// No checkpoint and no GPU. The model is the tiny seeded configuration below,
// built but never evaluated, and the shard directory is written by the test.

import Foundation
import MLX
import MLXLLM
import Testing

@testable import MLXRunners

@Suite("Qwen 3.8 Flash-Next n-gram resource")
struct Qwen4ExpNGramResourceTests {

    // A tiny tower with one PLE layer. `ple_embed_dim` divided by the four
    // n-gram heads gives a row of 64 values, which is the smallest width that
    // 4-bit affine quantization with group size 32 can carry.
    private static let configurationJSON = """
        {
          "model_type": "qwen4_exp_text",
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "vocab_size": 64,
          "rms_norm_eps": 1e-6,
          "full_attention_interval": 2,
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 16,
          "shared_expert_intermediate_size": 16,
          "linear_num_key_heads": 2,
          "linear_num_value_heads": 4,
          "linear_key_head_dim": 32,
          "linear_value_head_dim": 32,
          "linear_conv_kernel_dim": 4,
          "hc_count": 2,
          "hc_lowrank": 8,
          "indexer_n_heads": 2,
          "indexer_kv_heads": 1,
          "indexer_head_dim": 8,
          "indexer_budget": 8,
          "indexer_compress_ratio": 2,
          "ngram_size": 3,
          "heads_per_ngram": 2,
          "ngram_vocab_size_base": 1024,
          "make_ngram_vocab_size_divisible_by": 8,
          "split_ngram_parts": 4,
          "ple_embed_dim": 256,
          "ple_layer_ids": [1],
          "ple_conv_kernel_size": 4,
          "eos_token_id": 5,
          "partial_rotary_factor": 0.5,
          "rope_theta": 10000,
          "max_position_embeddings": 4096,
          "tie_word_embeddings": false
        }
        """

    private func makeModel() throws -> Qwen4ExpModel {
        let configuration = try JSONDecoder().decode(
            Qwen4ExpTextConfiguration.self, from: Data(Self.configurationJSON.utf8))
        return Qwen4ExpModel(text: configuration, withMTP: false)
    }

    /// Write the shard directory the offline transform would produce for this
    /// model: one safetensors file holding every shard of the table.
    private func makeShardDirectory(for model: Qwen4ExpModel) throws -> URL {
        let embedding = model.pleEmbeddings[0]
        let layout = Qwen4ExpNGramTableLayout(
            shardCount: embedding.shardCount,
            rowsPerShard: embedding.rowsPerShard,
            rowDimensions: embedding.rowDimensions)
        let layerIndex = model.model.pleLayerIndices[0]
        let prefix =
            "language_model.model.layers.\(layerIndex).ple.ple_embedding.ngram_embedding"

        var tensors: [String: [String: Any]] = [:]
        var payload = Data()
        var offset = 0
        func append(_ name: String, dtype: String, shape: [Int], byteCount: Int) {
            tensors[name] = [
                "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + byteCount],
            ]
            payload.append(Data(count: byteCount))
            offset += byteCount
        }
        for shard in 0 ..< layout.shardCount {
            let base = "\(prefix).shard_\(shard)"
            append(
                "\(base).weight", dtype: "U32",
                shape: [layout.rowsPerShard, layout.weightWordsPerRow],
                byteCount: layout.rowsPerShard * layout.weightBytesPerRow)
            append(
                "\(base).scales", dtype: "BF16",
                shape: [layout.rowsPerShard, layout.groupsPerRow],
                byteCount: layout.rowsPerShard * layout.groupBytesPerRow)
            append(
                "\(base).biases", dtype: "BF16",
                shape: [layout.rowsPerShard, layout.groupsPerRow],
                byteCount: layout.rowsPerShard * layout.groupBytesPerRow)
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4exp-ngram-resource-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let header = try JSONSerialization.data(withJSONObject: tensors, options: [.sortedKeys])
        var file = Data()
        var length = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &length) { file.append(contentsOf: $0) }
        file.append(header)
        file.append(payload)
        try file.write(to: directory.appendingPathComponent("ngram-00001-of-00001.safetensors"))
        return directory
    }

    @Test("A directory resource builds a row source of the model's row width")
    func directoryBuildsASource() throws {
        let model = try makeModel()
        let directory = try makeShardDirectory(for: model)
        defer { try? FileManager.default.removeItem(at: directory) }

        // The shape `RunnerResourceArguments` boxes for --resource.
        let resources = try RunnerResourceArguments.parse([
            "\(Qwen4ExpRunner.ngramRowSourceResource)=\(directory.path)"
        ])
        let source = try Qwen4ExpRunner.resolveNGramRowSource(
            resources[Qwen4ExpRunner.ngramRowSourceResource], for: model)
        #expect(source.rowDimensions == model.pleEmbeddings[0].rowDimensions)
        #expect(source is Qwen4ExpNGramTable)
    }

    @Test("A path as a plain String is the same shape")
    func stringPathBuildsASource() throws {
        let model = try makeModel()
        let directory = try makeShardDirectory(for: model)
        defer { try? FileManager.default.removeItem(at: directory) }

        let source = try Qwen4ExpRunner.resolveNGramRowSource(
            directory.path as NSString, for: model)
        #expect(source.rowDimensions == model.pleEmbeddings[0].rowDimensions)
    }

    @Test("An already built row source is taken as it is")
    func objectResourceIsTakenAsItIs() throws {
        let model = try makeModel()
        let given = FixedNGramRowSource(rowDimensions: 64)
        let source = try Qwen4ExpRunner.resolveNGramRowSource(given, for: model)
        #expect(source === given)
    }

    @Test("A single file is refused by name")
    func singleFileIsRefusedByName() throws {
        let model = try makeModel()
        let directory = try makeShardDirectory(for: model)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("ngram-00001-of-00001.safetensors")

        #expect(throws: Qwen4ExpNGramTableError.self) {
            try Qwen4ExpRunner.resolveNGramRowSource(file as NSURL, for: model)
        }
        do {
            _ = try Qwen4ExpRunner.resolveNGramRowSource(file as NSURL, for: model)
        } catch {
            #expect("\(error)".contains("is a file"))
            #expect("\(error)".contains("DIRECTORY"))
        }
    }

    @Test("No resource on a PLE checkpoint is RunnerError.resourceMissing")
    func missingResourceIsRefused() throws {
        let model = try makeModel()
        #expect(!model.pleEmbeddings.isEmpty)
        do {
            _ = try Qwen4ExpRunner.resolveNGramRowSource(nil, for: model)
            Issue.record("a checkpoint with PLE layers and no resource must be refused")
        } catch let error as RunnerError {
            #expect(
                error
                    == .resourceMissing(
                        "\(Qwen4ExpRunner.ngramRowSourceResource): this checkpoint has "
                            + "1 PLE layers and the n-gram table "
                            + "is never model parameters; pass the n-gram shard directory"))
        }
    }
}

/// A row source that needs no file. It stands in for an in-process caller's
/// already built source.
private final class FixedNGramRowSource: Qwen4ExpNGramRowSource {
    let rowDimensions: Int
    init(rowDimensions: Int) { self.rowDimensions = rowDimensions }
    func rows(globalIds: MLXArray) -> MLXArray {
        MLXArray.zeros(globalIds.shape + [rowDimensions])
    }
}
