import Foundation
import MLXFastCore
import Testing

// Contract tests for `fixtures/bonsai2_27b_mlx_v1_track.json`.
//
// The track is STAMPED AND NOT YET ARMED. `official_scoring_enabled` is false,
// the timed pool is empty, `live_golden` is unset and the hidden oracle still
// carries its pending-organizer sentinel: the goldens are organizer material
// and are captured on the ranked box. This suite asserts that unarmed state as
// a COHERENT WHOLE -- an empty pool with scoring switched on, or an armed pin
// beside a sentinel, is the half-armed shape that must red -- and it asserts
// the facts that do not wait for goldens: the target block, the tensor
// inventory, the two reference manifests and the compiled constants all
// describing one pack and one head.
//
// This is a laptop-side JSON-shape test only. It does not exercise benchd's
// Rust contract parsers; those live in the bench repository.

@Suite("Ternary Bonsai 2 27B track contract fixture")
struct Bonsai2TrackFixtureTests {

    /// The sentinel every unarmed organizer slot in this fixture carries.
    /// EXACT-MATCH ONLY -- never a prefix check.
    static let pendingOrganizerSentinel = "BONSAI2-27B-MLX-V1-PENDING-ORGANIZER"

    @Test("fixture parses as a JSON object")
    func fixtureParses() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["schema_version"] as? Int == 1)
    }

    @Test("track_id is pinned to the leaderboard / R2-prefix identity")
    func trackIdIsPinned() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["track_id"] as? String == "bonsai2-27b-mlx-v1")
    }

    @Test("benchmark_name matches the Yukon manifest name")
    func benchmarkNameMatchesManifest() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["benchmark_name"] as? String == "mlxfast-bonsai2-27b")
    }

    @Test("track_id is substring-clean against every retired MTP name")
    func trackIdIsCleanAgainstRetiredNames() throws {
        let object = try bonsai2TrackContractObject()
        let trackId = try #require(object["track_id"] as? String)
        let retired = [
            "MLXFAST_MTP_",
            "mtp-ranked",
            "measure-mtp-job",
            "mtp-weights",
            "laguna-xs-2.1-mtp",
            "gemma4-31b-it",
        ]
        for name in retired {
            #expect(
                !trackId.contains(name),
                "track_id must not substring-collide with retired name \(name)")
            #expect(
                !name.contains(trackId),
                "retired name \(name) must not substring-collide with track_id")
        }
    }

    /// THE TRACK IS ARMED, AND COHERENTLY. Scoring on, eight pool pins, a
    /// live golden that names one of them, a hidden correctness golden equal to
    /// that row's pin, and one oracle pin for every declarable depth must
    /// travel together: any of them missing is a fixture that would let a
    /// ranked run start against material it does not have.
    @Test("the fixture is coherently armed: scoring, pool, live golden, oracles")
    func theFixtureIsCoherentlyArmed() throws {
        let object = try bonsai2TrackContractObject()
        let prefix = "correctness_prompts/bonsai2-27b-mlx-v1/"
        func isPin(_ pin: [String: Any]) -> Bool {
            guard let path = pin["r2_path"] as? String, let sha256 = pin["sha256"] as? String,
                let bytes = pin["bytes"] as? Int
            else { return false }
            return path.hasPrefix(prefix) && path.hasSuffix(".golden.json") && sha256.count == 64
                && sha256.allSatisfy { "0123456789abcdef".contains($0) } && bytes > 0
        }
        #expect(object["official_scoring_enabled"] as? Bool == true)
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        #expect(pool.count == 8)
        #expect(pool.allSatisfy(isPin))
        let live = try #require(object["live_golden"] as? String)
        let liveRow = try #require(pool.first { ($0["r2_path"] as? String) == "\(prefix)\(live).golden.json" })
        // ROOT level -- benchd's `hidden_correctness_golden_pin_from_contract`
        // reads this key directly off the contract root, never a nested
        // wrapper.
        let golden = try #require(object["hidden_correctness_golden"] as? [String: Any])
        #expect(golden["sha256"] as? String == liveRow["sha256"] as? String)
        #expect(golden["bytes"] as? Int == liveRow["bytes"] as? Int)
        #expect(object["hidden_material"] == nil)
        let oracles = try #require(object["live_golden_speculative"] as? [String: [String: Any]])
        let depthKeys = Set((1...7).map { "mtp\($0)" } + (1...16).map { "dflash\($0)" })
        #expect(Set(oracles.keys) == depthKeys)
        #expect(oracles.values.allSatisfy(isPin))
    }

    /// MODE FENCE. This track has TWO speculative arms, each a separate pinned
    /// export: the MTP head and the DFlash 2 block drafter. `serial` must be
    /// present because the baseline leg is pinned serial and is validated
    /// against this same list. The list is exact, so a mode nothing here can
    /// run cannot be armed by widening it: `dspark` in particular must stay
    /// absent, and benchd refuses it by name in any case.
    @Test("allowed_modes declares serial, mtp and dflash")
    func allowedModesDeclaresSerialMTPAndDFlash() throws {
        let object = try bonsai2TrackContractObject()
        let modes = try #require(object["allowed_modes"] as? [String])
        #expect(modes == ["serial", "mtp", "dflash"])
        #expect(!modes.contains("dspark"))
    }

    /// DEPTH FENCE. The two decoders have INDEPENDENT depth envelopes, and
    /// `tools/spec-declaration.sh` reads each from its own contract block. MTP
    /// stays 1 to 7. DFlash 2 is 1 to 16: its depth is the block size minus
    /// one, and David set the limit at 16.
    @Test("each decoder declares its own permitted draft depths")
    func eachDecoderDeclaresItsOwnPermittedDraftDepths() throws {
        let object = try bonsai2TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        #expect(head["permitted_draft_depths"] as? [Int] == Array(1 ... 7))
        let drafter = try #require(object["dflash_drafter"] as? [String: Any])
        #expect(drafter["permitted_draft_depths"] as? [Int] == Array(1 ... 16))
    }

    @Test("kv_backend is pinned explicitly to contiguous")
    func kvBackendIsContiguous() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["kv_backend"] as? String == "contiguous")
    }

    @Test("scored_batch_size is pinned to the ruled single-stream width")
    func scoredBatchSizeIsOne() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["scored_batch_size"] as? Int == 1)
    }

    @Test("scored_exponents equals the ruled certify pair, exact field names")
    func scoredExponentsMatchesRuledPair() throws {
        let object = try bonsai2TrackContractObject()
        let exponents = try #require(object["scored_exponents"] as? [String: Any])
        // Field names must match benchd's `DeclaredScoredExponents` struct
        // exactly -- NOT the shorthand "prefill"/"decode" spelling, which
        // `ScoredExponents::certify` would treat as absent.
        let prefill = try #require(exponents["prefill_gain_exponent"] as? Double)
        let decode = try #require(exponents["decode_gain_exponent"] as? Double)
        #expect(prefill == 0.25)
        #expect(decode == 0.75)
        #expect(
            exponents.count == 2,
            "scored_exponents must carry exactly the certify pair, no extra keys")
    }

    @Test("the contract and the compiled constants agree on the scored window")
    func contractAgreesWithCompiledConstants() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["vocab_size"] as? Int == MLXFastConstants.vocabSize)
        #expect(object["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
        #expect(
            object["golden_model_type"] as? String
                == MLXFastConstants.requiredGoldenModelType)
        #expect(object["seed_tokens"] as? Int == MLXFastConstants.benchmarkDecodeSeedTokens)
        #expect(object["correctness_steps"] as? Int == MLXFastConstants.correctnessSteps)
        #expect(
            object["benchmark_decode_steps"] as? Int == MLXFastConstants.benchmarkDecodeSteps)
        #expect(
            object["local_submit_benchmark_decode_steps"] as? Int
                == MLXFastConstants.localSubmitBenchmarkDecodeSteps)
        #expect(object["score_decode_weight"] as? Double == MLXFastConstants.scoreDecodeWeight)
        #expect(object["score_prefill_weight"] as? Double == MLXFastConstants.scorePrefillWeight)
        #expect(
            object["decode_speedup_floor"] as? Double
                == MLXFastConstants.scoreDecodeSpeedupFloor)
        #expect(
            object["prefill_speedup_floor"] as? Double
                == MLXFastConstants.scorePrefillSpeedupFloor)
    }

    @Test("target reference-model pin is a real 40-hex revision, matching the compiled constants")
    func targetPinIsFortyHexAndMatchesConstants() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let modelId = try #require(target["upstream_model_id"] as? String)
        let revision = try #require(target["upstream_revision"] as? String)
        #expect(modelId == bonsai2Repository)
        #expect(revision == bonsai2Revision)
        #expect(modelId == MLXFastConstants.referenceModelRepository)
        #expect(revision == MLXFastConstants.referenceModelRevision)
        #expect(isFortyLowercaseHex(revision))
    }

    /// The `target` block is READ OFF the pack's own config.json, so every
    /// geometry field is asserted against that file rather than against a
    /// second hand-written copy of the same numbers.
    @Test("target geometry matches the pinned pack's own config.json")
    func targetGeometryMatchesConfig() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let config = try bonsai2ConfigObject()
        let text = try #require(config["text_config"] as? [String: Any])
        let rope = try #require(text["rope_parameters"] as? [String: Any])

        #expect(target["model_type"] as? String == config["model_type"] as? String)
        #expect(target["model_type"] as? String == "prism_hadamard_qwen35")
        #expect(target["base_model_type"] as? String == config["base_model_type"] as? String)
        #expect(target["text_model_type"] as? String == text["model_type"] as? String)
        #expect(target["schema_version"] as? Int == config["schema_version"] as? Int)

        for key in [
            "num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size",
            "max_position_embeddings", "full_attention_interval",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "linear_num_key_heads", "linear_num_value_heads",
            "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
            "bos_token_id", "mtp_num_hidden_layers",
        ] {
            #expect(
                target[key] as? Int == text[key] as? Int,
                Comment(rawValue: "target.\(key) must be the config's own"))
        }
        #expect(target["attention_bias"] as? Bool == text["attention_bias"] as? Bool)
        #expect(target["attn_output_gate"] as? Bool == text["attn_output_gate"] as? Bool)
        #expect(target["tie_word_embeddings"] as? Bool == text["tie_word_embeddings"] as? Bool)
        #expect(target["rope_theta"] as? Int == rope["rope_theta"] as? Int)
        #expect(target["partial_rotary_factor"] as? Double == rope["partial_rotary_factor"] as? Double)
        #expect(target["gdn_activation_layout"] as? String == config["gdn_activation_layout"] as? String)
        #expect(target["tensor_namespace"] as? String == config["tensor_namespace"] as? String)

        // THE HEAD IS UNTIED AND THE PACK CARRIES NO HEAD OF ITS OWN. Both are
        // load-bearing: an untied head means `lm_head` is a real packed module,
        // and `mtp_num_hidden_layers: 0` is why this track stages a separate
        // head export at all.
        #expect(target["tie_word_embeddings"] as? Bool == false)
        #expect(target["mtp_num_hidden_layers"] as? Int == 0)
    }

    /// The layer schedule is FIXED by the interval rather than published as a
    /// list on this pack, so the summary counts are asserted against the
    /// config's own `layer_types` array.
    @Test("the declared layer schedule is the config's own, and its summaries are derived")
    func layerScheduleSummariesAreDerived() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let config = try bonsai2ConfigObject()
        let text = try #require(config["text_config"] as? [String: Any])
        let layerTypes = try #require(text["layer_types"] as? [String])
        let interval = try #require(text["full_attention_interval"] as? Int)

        #expect(layerTypes.count == target["num_hidden_layers"] as? Int)
        let counts = try #require(target["layer_type_counts"] as? [String: Int])
        for kind in ["full_attention", "linear_attention"] {
            #expect(
                counts[kind] == layerTypes.filter { $0 == kind }.count,
                Comment(rawValue: "layer_type_counts.\(kind)"))
        }
        #expect(counts.values.reduce(0, +) == layerTypes.count)

        let indices = try #require(target["full_attention_layer_indices"] as? [Int])
        #expect(indices == layerTypes.indices.filter { layerTypes[$0] == "full_attention" })
        // Every fourth layer, and nothing else.
        #expect(indices == (0 ..< layerTypes.count).filter { ($0 + 1) % interval == 0 })
    }

    @Test("target quantization agrees with the pack and declares no mixed precision")
    func targetQuantizationAgreesWithConfig() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let quantization = try #require(target["quantization"] as? [String: Any])
        let config = try bonsai2ConfigObject()
        let published = try #require(config["quantization"] as? [String: Any])

        #expect(quantization["mode"] as? String == published["mode"] as? String)
        #expect(quantization["bits"] as? Int == published["bits"] as? Int)
        #expect(quantization["group_size"] as? Int == published["group_size"] as? Int)
        #expect(quantization["mode"] as? String == "affine")
        #expect(quantization["bits"] as? Int == 2)
        #expect(quantization["group_size"] as? Int == 128)
        #expect(quantization["mixed_precision"] as? Bool == false)
    }

    /// The signed block Hadamard transform is what makes the packed weights
    /// readable, so the fixture records its shape and the module count it
    /// applies to. The counts must be the inventory's own.
    @Test("the hadamard block agrees with the pack and the inventory")
    func hadamardBlockAgreesWithInventory() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let hadamard = try #require(target["hadamard"] as? [String: Any])
        let config = try bonsai2ConfigObject()
        let inventory = try bonsai2InventoryFixture()

        #expect(
            hadamard["manifest_path_in_checkpoint"] as? String
                == config["hadamard_config"] as? String)
        #expect(hadamard["manifest_path_in_checkpoint"] as? String == "hadamard.json")
        #expect(hadamard["block_size"] as? Int == 1024)
        #expect(hadamard["sign_dtype"] as? String == "float32")
        #expect(hadamard["arithmetic_dtype"] as? String == "float32")

        let modules = try #require(config["modules"] as? [[String: Any]])
        #expect(hadamard["packed_module_count"] as? Int == modules.count)
        #expect(hadamard["sign_vector_count"] as? Int == modules.count)
        #expect(hadamard["packed_module_count"] as? Int == inventory.summary.packedModuleCount)
        #expect(hadamard["sign_vector_count"] as? Int == inventory.summary.signVectorCount)
        // Exactly one module is the embedding, and it is the one whose output
        // needs the INVERSE transform.
        let embeddings = modules.filter { $0["embedding"] as? Bool == true }
        #expect(embeddings.count == 1)
        #expect(hadamard["embedding_module"] as? String == embeddings.first?["path"] as? String)
        // Every module declares the same block width the fixture pins.
        #expect(modules.allSatisfy { $0["block"] as? Int == hadamard["block_size"] as? Int })
    }

    @Test("the config fixture is the published bytes, and every pin of it agrees")
    func configFixtureIsThePublishedBytes() throws {
        let data = try Data(contentsOf: bonsai2ConfigFixtureURL)
        #expect(sha256Hex(data) == bonsai2ConfigSHA256)
        #expect(data.count == bonsai2ConfigByteCount)

        let inventory = try bonsai2InventoryFixture()
        #expect(inventory.source.configSHA256 == bonsai2ConfigSHA256)
        #expect(inventory.source.configBytes == bonsai2ConfigByteCount)

        let records = try bonsai2ReferenceManifestRecords()
        let configRecord = try #require(records.first { $0.path == "config.json" })
        #expect(configRecord.sha256 == bonsai2ConfigSHA256)
        #expect(configRecord.byteCount == bonsai2ConfigByteCount)
    }

    @Test("target tensor counts agree with the pinned tensor inventory")
    func targetTensorCountsAgreeWithInventory() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let inventory = try bonsai2InventoryFixture()

        #expect(target["raw_checkpoint_tensor_count"] as? Int == inventory.summary.tensorCount)
        #expect(target["raw_checkpoint_shard_count"] as? Int == inventory.summary.shardCount)
        #expect(
            target["raw_checkpoint_total_size_bytes"] as? Int
                == inventory.summary.totalSizeBytes)
        #expect(
            target["language_model_tensor_count"] as? Int
                == inventory.summary.languageModelTensorCount)
        #expect(
            target["vision_tower_tensor_count"] as? Int
                == inventory.summary.visionTowerTensorCount)
        #expect(target["lm_head_tensor_count"] as? Int == inventory.summary.lmHeadTensorCount)

        // The inventory's own totals must be summed from its records, not
        // written beside them.
        #expect(inventory.tensors.count == inventory.summary.tensorCount)
        #expect(inventory.shards.map(\.tensorCount).reduce(0, +) == inventory.summary.tensorCount)
        #expect(
            inventory.tensors.keys.filter { $0.hasPrefix("language_model.") }.count
                == inventory.summary.languageModelTensorCount)
        #expect(
            inventory.tensors.keys.filter { $0.hasPrefix("vision_tower.") }.count
                == inventory.summary.visionTowerTensorCount)
        #expect(
            inventory.tensors.keys.filter { $0.hasSuffix(".signs") }.count
                == inventory.summary.signVectorCount)
        // NO EMBEDDED HEAD. The pack ships none, and a pack that did would be a
        // different artifact from the one this track pins.
        #expect(inventory.summary.mtpTensorCount == 0)
        #expect(
            inventory.tensors.keys.allSatisfy {
                !$0.contains(".mtp.") && !$0.hasPrefix("mtp.")
            })
        // This track loads TEXT ONLY. The tower is published and is not served.
        #expect(target["vision_tower_loaded"] as? Bool == false)
    }

    /// THE HEAD IS SEPARATE, and that is the fact this test is built around.
    /// The source token is what tells a reader the head is not in the target
    /// pack, and it must travel with a real pin file and a real revision.
    @Test("mtp_head declares a separate pinned export, not an embedded block")
    func mtpHeadDeclaresASeparateExport() throws {
        let object = try bonsai2TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        #expect(head["role"] as? String == "spec_decode_mtp_head")
        #expect(head["source"] as? String == "separate_pinned_export")
        #expect(head["upstream_model_id"] as? String == bonsai2HeadRepository)
        #expect(head["upstream_revision"] as? String == bonsai2HeadRevision)
        #expect(isFortyLowercaseHex(try #require(head["upstream_revision"] as? String)))
        #expect(head["model_type"] as? String == "qwen3_5_mtp")
        #expect(head["num_hidden_layers"] as? Int == 1)
        #expect(head["tensor_prefix"] as? String == "")
        // The head owns no embedding and no output projection: it borrows the
        // target's, which on this pack are packed Hadamard modules.
        #expect(head["use_dedicated_embeddings"] as? Bool == false)
        #expect(head["shares_target_embed_tokens"] as? Bool == true)
        #expect(head["shares_target_lm_head"] as? Bool == true)

        let quantization = try #require(head["quantization"] as? [String: Any])
        #expect(quantization["mode"] as? String == "affine")
        #expect(quantization["bits"] as? Int == 4)
        #expect(quantization["group_size"] as? Int == 64)

        // The head is a DIFFERENT artifact from the target, at a different
        // quantization. Pinning them to the same repository would mean the
        // fixture had lost track of which one it was describing.
        let target = try #require(object["target"] as? [String: Any])
        #expect(head["upstream_model_id"] as? String != target["upstream_model_id"] as? String)
        #expect(head["manifest_path"] as? String != target["manifest_path"] as? String)
    }

    @Test("permitted draft depths fit the declared MTP envelope")
    func permittedDraftDepthsFitTheEnvelope() throws {
        let object = try bonsai2TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        let depths = try #require(head["permitted_draft_depths"] as? [Int])
        let protocolBlock = try #require(object["protocol"] as? [String: Any])
        let envelope = try #require(
            protocolBlock["mtp_envelope_constants_from_darkbloom"] as? [String: Any])
        let maxDraft = try #require(envelope["max_draft_tokens"] as? Int)

        #expect(!depths.isEmpty)
        #expect(depths == Array(1 ... maxDraft))
        #expect(depths.allSatisfy { $0 >= 1 && $0 <= maxDraft })
        #expect(envelope["max_speculative_batch"] as? Int == object["scored_batch_size"] as? Int)
    }

    /// Both manifests carry header pins, and a pin that is not summed from the
    /// records beside it is not a pin.
    @Test("both reference manifests have header pins summed from their records")
    func referenceManifestHeaderPinsAreSummed() throws {
        for url in [bonsai2ReferenceManifestURL, bonsai2HeadManifestURL] {
            let records = try bonsai2ManifestRecords(at: url)
            let pins = try bonsai2ManifestHeaderPins(at: url)
            #expect(records.count == pins.records, Comment(rawValue: url.lastPathComponent))
            #expect(
                records.map(\.byteCount).reduce(0, +) == pins.bytes,
                Comment(rawValue: url.lastPathComponent))
            #expect(records.allSatisfy { isSixtyFourLowercaseHex($0.sha256) })
            #expect(records.allSatisfy { $0.byteCount > 0 })
            #expect(Set(records.map(\.path)).count == records.count)
        }
    }

    /// The target manifest must pin the files the loader actually opens. The
    /// pack has no `model.safetensors.index.json`, so the single shard, the
    /// config and the Hadamard manifest are the three that matter.
    @Test("the target manifest pins the files the loader opens")
    func targetManifestPinsWhatTheLoaderOpens() throws {
        let records = try bonsai2ReferenceManifestRecords()
        for path in ["config.json", "hadamard.json", "model.safetensors"] {
            #expect(records.contains { $0.path == path }, Comment(rawValue: path))
        }
        let inventory = try bonsai2InventoryFixture()
        let hadamardRecord = try #require(records.first { $0.path == "hadamard.json" })
        #expect(hadamardRecord.sha256 == inventory.source.hadamardManifestSHA256)
        #expect(hadamardRecord.byteCount == inventory.source.hadamardManifestBytes)

        let shard = try #require(records.first { $0.path == "model.safetensors" })
        #expect(shard.byteCount == inventory.summary.totalSizeBytes)
        #expect(inventory.shards.map(\.name) == ["model.safetensors"])
    }

    /// The head manifest must pin the three files
    /// `Qwen35InlineMTPAssistant.load` reads, and must pin the same revision
    /// the contract's `mtp_head` block names.
    @Test("the head manifest pins the files the drafter loader opens")
    func headManifestPinsWhatTheDrafterOpens() throws {
        let records = try bonsai2HeadManifestRecords()
        for path in ["config.json", "model.safetensors", "model.safetensors.index.json"] {
            #expect(records.contains { $0.path == path }, Comment(rawValue: path))
        }
        let object = try bonsai2TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        #expect(head["manifest_path"] as? String
            == "fixtures/reference_bonsai2_27b_mtp_head_4bit.sha256")

        let text = try String(contentsOf: bonsai2HeadManifestURL, encoding: .utf8)
        #expect(text.contains(bonsai2HeadRevision))
        #expect(text.contains(bonsai2HeadRepository))

        let shard = try #require(records.first { $0.path == "model.safetensors" })
        let declared = try #require(head["tensor_bytes"] as? Int)
        // The file is the tensor payload plus the safetensors header, so the
        // pinned file is LARGER than the declared tensor bytes and within a
        // header's worth of it.
        #expect(shard.byteCount > declared)
        #expect(shard.byteCount - declared < 64 * 1024)
        // The declaration surface bounds the head, and this head must fit it.
        let declaration = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: bonsai2HeadDeclarationURL)) as? [String: Any])
        #expect(declaration["source"] as? String == "pinned")
        let maxBytes = try #require(declaration["max_bytes"] as? Int)
        #expect(shard.byteCount <= maxBytes)
    }

    @Test("scoring_semantics records the ruled composite formula")
    func scoringSemanticsRecordsRuledFormula() throws {
        let object = try bonsai2TrackContractObject()
        let semantics = try #require(object["scoring_semantics"] as? [String: Any])
        let formula = try #require(semantics["formula"] as? String)
        #expect(formula.contains("0.25"))
        #expect(formula.contains("0.75"))
        #expect(formula.contains("prefill_gain"))
        #expect(formula.contains("decode_gain"))
    }

    /// The license block names the repositories the weights actually come
    /// from. A license that verifies a different repository verifies nothing,
    /// and this track loads TWO artifacts.
    @Test("the license block verifies both pinned repositories")
    func licenseVerifiesThePinnedRepositories() throws {
        let object = try bonsai2TrackContractObject()
        let license = try #require(object["license"] as? [String: Any])
        let verified = try #require(license["verified_repositories"] as? [String])
        #expect(Set(verified) == [bonsai2Repository, bonsai2HeadRepository])
        let url = try #require(license["url"] as? String)
        #expect(url.contains(bonsai2Repository))
        #expect(url.contains(bonsai2Revision))
    }
}

private func isFortyLowercaseHex(_ value: String) -> Bool {
    isLowercaseHex(value, count: 40)
}

private func isSixtyFourLowercaseHex(_ value: String) -> Bool {
    isLowercaseHex(value, count: 64)
}

private func isLowercaseHex(_ value: String, count: Int) -> Bool {
    value.utf8.count == count && value.utf8.allSatisfy { byte in
        (byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9"))
            || (byte >= UInt8(ascii: "a") && byte <= UInt8(ascii: "f"))
    }
}
