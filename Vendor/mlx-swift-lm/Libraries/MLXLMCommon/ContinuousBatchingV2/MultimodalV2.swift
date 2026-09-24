// MultimodalV2.swift
//
// ContinuousBatchingV2 — vision (multimodal) prefill support.
//
// A vision request carries precomputed image EMBEDDINGS (vision tower +
// projector output) plus the placeholder-token spans they replace
// (`CBv2MultimodalInput`, CBv2Contracts.swift). The engine:
//
//  1. VALIDATES + RESOLVES the input once at submit (this file's
//     `CBv2MultimodalPlan.resolve`) — spans checked against the prompt,
//     adjacent spans coalesced into mask BLOCKS, embeddings materialized
//     and shape-checked. All failures are submit-time throws.
//  2. SCHEDULES prefill chunks that never split a block (SchedulerV2 snaps
//     chunk boundaries to block edges; a block longer than
//     `maxBatchedTokensPerStep` is rejected at submit).
//  3. EXECUTES span-containing chunks through the embedding-forward path:
//     input embeddings = scaled text-token embeddings with each row's span
//     embeddings spliced verbatim (`spliceEmbeddings`), and a row-local
//     boolean mask implementing causal-plus-bidirectional-within-blocks —
//     the exact semantics of MLXVLM Gemma4's
//     `gemma4BidirectionalVisionMask` overlay. Tokens of the same image block
//     attend each other in BOTH directions, overriding causality and the
//     sliding window; everything else stays causal (∧ window). Long rows
//     retain q=128 query blocking, expanding only touched blocks' K/V slices
//     enough to include the complete image span. Explicit model + cache
//     capabilities allow independently spliced vision rows and nil-context
//     text neighbors to share a rectangular call. Decode after prefill is
//     unchanged (image tokens are history in KV by then). The path choice is
//     a pure function of has-spans-in-chunk; never data-dependent.
//  4. EXCLUDES vision requests from prefix-cache lookup AND donation
//     (v1 policy): token-id chain hashes cannot see image content, so a
//     hit would silently reuse the wrong KV. Image-digest extra keys in the
//     block hash (the vLLM-V1 `extra_keys` design, research report 08 §D)
//     are the documented follow-up.

import Foundation
import MLX

// MARK: - Span-chunk attention context

/// Per-chunk span context bound to the layer caches for the duration of ONE
/// span-containing prefill chunk's graph build. Absolute token coordinates.
public struct CBv2SpanChunkContext: Sendable, Equatable {
    /// Absolute end position of the chunk (chunk start + chunk length).
    /// Queries are the last `L` positions before it; the row's returned KV
    /// view is the last `kL` positions before it (both backends' chunk
    /// views end at the chunk's last token by construction).
    public var chunkEnd: Int
    /// Coalesced bidirectional blocks FULLY contained in the chunk
    /// (scheduler invariant: blocks are never split across chunks).
    public var blocks: [CBv2ImageSpan]

    public init(chunkEnd: Int, blocks: [CBv2ImageSpan]) {
        self.chunkEnd = chunkEnd
        self.blocks = blocks
    }
}

/// Layer caches that can honor a span-chunk mask context. The engine binds
/// the context on every layer cache (storage-owning AND KV-shared borrowing
/// caches — the MLXVLM reference applies the overlay to both the global and
/// sliding masks that shared layers reuse) immediately before building a
/// span-containing chunk's graph, and unbinds it immediately after. Outside
/// that window the bound context is ALWAYS nil, so every other computation
/// (decode, text chunks, other requests' chunks) is untouched.
public protocol CBv2SpanMaskBinding: AnyObject {
    func bindSpanContext(_ context: CBv2SpanChunkContext?)
}

/// Stronger binding contract for rectangular multimodal prefill. The array
/// is row-aligned with the cache's current batch: non-nil entries carry that
/// row's vision overlay and nil entries retain ordinary text attention.
/// Custom and paged providers must opt in explicitly; structural
/// `CBv2SpanMaskBinding` conformance alone is insufficient.
public protocol CBv2PackedSpanMaskBinding: CBv2SpanMaskBinding {
    func bindSpanContexts(_ contexts: [CBv2SpanChunkContext?]?)
}

// MARK: - Model surfaces

/// Steppable models that can prefill from spliced input embeddings.
/// Additive refinement of `CBv2SteppableModel`; models that do not conform
/// (or conform with `supportsMultimodalPrefill == false`) reject multimodal
/// requests at submit.
public protocol CBv2MultimodalSteppableModel: CBv2SteppableModel {
    /// True when the underlying model actually supports the embedding
    /// forward (adapters over arbitrary models answer at runtime).
    var supportsMultimodalPrefill: Bool { get }
    func supportsMultimodalPrefill(attention: CBv2MultimodalAttention) -> Bool
    /// The scaled text-token embeddings exactly as the model's trunk would
    /// compute before layer 0 (`embed(tokens) * embedScale` for Gemma-class
    /// models) — the tensor the engine splices span embeddings into.
    func embedPromptTokens(_ tokens: MLXArray) -> MLXArray
    /// Forward from spliced input embeddings. `tokens` is still supplied
    /// ([1, L] int32, the placeholder-bearing prompt slice) for models whose
    /// trunk consumes token ids besides the embedding (Gemma4 per-layer
    /// embeddings); `inputEmbeddings` ([1, L, hidden]) replaces the trunk's
    /// own embedding lookup. Mirrors the MLXVLM wrapper's
    /// `languageModel(tokens, inputEmbedding:cache:)` call exactly.
    func forward(
        tokens: MLXArray, inputEmbeddings: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> MLXArray
}
/// Attention-only models whose request positions cannot be recovered from a
/// batch-wide scalar cache offset (Qwen M-RoPE).
public protocol CBv2PositionedSteppableModel:
    CBv2SteppableModel, CBv2PositionedForwardingCapabilityProviding
{
    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        positionIds: MLXArray?
    ) -> MLXArray
}

/// Multimodal models that inject additional vision features after selected
/// language layers (Qwen3-VL DeepStack).
public protocol CBv2DeepstackMultimodalSteppableModel: CBv2MultimodalSteppableModel {
    /// Exact number of language-layer injection arrays expected from the
    /// submit-time DeepStack provider.
    var deepstackLayerCount: Int { get }
    func forward(
        tokens: MLXArray,
        inputEmbeddings: MLXArray,
        deepstackEmbeddings: [MLXArray],
        caches: [CBv2AttendingLayerCache],
        positionIds: MLXArray?
    ) -> MLXArray
}

/// Attention-only multimodal models that require request-owned positions but
/// do not carry recurrent state (Qwen M-RoPE without DeepStack).
public protocol CBv2PositionedEmbeddingSteppableModel: CBv2MultimodalSteppableModel {
    func forward(
        tokens: MLXArray,
        inputEmbeddings: MLXArray,
        caches: [CBv2AttendingLayerCache],
        positionIds: MLXArray?
    ) -> MLXArray
}


/// Optional request-owned refinement for models whose embedding forward also
/// needs recurrent transactions and/or explicit model positions.
public protocol CBv2PositionedMultimodalSteppableModel: CBv2MultimodalSteppableModel {
    func forward(
        tokens: MLXArray,
        inputEmbeddings: MLXArray,
        caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> MLXArray
}

extension CBv2MultimodalSteppableModel {
    public var supportsMultimodalPrefill: Bool { true }
    public func supportsMultimodalPrefill(attention: CBv2MultimodalAttention) -> Bool {
        supportsMultimodalPrefill
    }
}

/// Model-level surface for `LanguageModel` conformers reached through
/// `CBv2SteppableLanguageModelAdapter` (Gemma4TextModel conforms): the
/// KVCache-shaped twins of the two `CBv2MultimodalSteppableModel`
/// requirements.
public protocol CBv2EmbeddingForwardable {
    /// True only when the model's WEIGHTS were trained for the blockwise
    /// bidirectional image-span attention the CBv2 span masks implement
    /// (Gemma4: `use_bidirectional_attention == "vision"`). Structural
    /// conformance is NOT capability: a text-only Gemma4TextModel can
    /// execute the embedding forward, but serving vision spans through it
    /// would apply attention its weights never saw — the legacy text path
    /// only enables the overlay for `"vision"` configs, and the adapter's
    /// `supportsMultimodalPrefill` must match, so such configs reject
    /// multimodal requests at submit (PR#63 review).
    var supportsVisionSpanPrefill: Bool { get }
    /// True when visual tokens are ordinary causal tokens (Qwen).
    var supportsCausalVisionPrefill: Bool { get }
    /// `embed(tokens) * embedScale` — the trunk's pre-layer-0 hidden state.
    func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray
    /// Forward with the trunk's embedding lookup replaced by
    /// `inputEmbedding`; token ids still flow to any id-consuming side
    /// inputs (Gemma4 PLE). Must route through the SAME v2 attention branch
    /// as the token forward when `cache` holds CBv2 layer caches.
    func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray
}
/// Model-level twin of `CBv2PositionedSteppableModel` for the generic
/// `LanguageModel` adapter.
public protocol CBv2PositionedLanguageModelForwardable {
    func cbv2Forward(
        _ inputs: MLXArray, cache: [KVCache]?, positionIds: MLXArray?
    ) -> MLXArray
}

/// Model-level DeepStack embedding forward reached through the generic
/// `LanguageModel` adapter.
public protocol CBv2DeepstackEmbeddingForwardable: CBv2EmbeddingForwardable {
    var deepstackLayerCount: Int { get }
    func embeddingForward(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray,
        deepstackEmbeddings: [MLXArray],
        cache: [KVCache]?,
        positionIds: MLXArray?
    ) -> MLXArray
}
/// Model-level positioned embedding forward reached through the generic
/// `LanguageModel` adapter for attention-only models.
public protocol CBv2PositionedEmbeddingForwardable: CBv2EmbeddingForwardable {
    func embeddingForward(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray,
        cache: [KVCache]?,
        positionIds: MLXArray?
    ) -> MLXArray
}


extension CBv2EmbeddingForwardable {
    public var supportsCausalVisionPrefill: Bool { false }
}


/// Model-level embedding forward with request-owned recurrent state and
/// explicit positions. Recurrent hybrids use this; Gemma remains legacy.
public protocol CBv2PositionedRecurrentEmbeddingForwardable: CBv2EmbeddingForwardable {
    func embeddingForward(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray,
        cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> MLXArray
}

// MARK: - Resolved multimodal state (engine-internal)

/// The submit-time resolution of a `CBv2MultimodalInput`: validated spans,
/// coalesced mask blocks, and materialized per-span embeddings normalized to
/// `[1, length, hidden]`. Immutable after creation; handed off exactly once
/// from the submit thread to the engine queue and read only there
/// (`@unchecked Sendable` by the same ownership-transfer argument as
/// `CBv2PrefixAdoption`).
struct CBv2ResolvedMultimodal: @unchecked Sendable {
    let spans: [CBv2ImageSpan]
    let attention: CBv2MultimodalAttention
    /// Adjacent spans coalesced (a contiguous placeholder run attends as ONE
    /// bidirectional block — `gemma4VisionBlockIds` groups contiguous runs).
    let blocks: [CBv2ImageSpan]
    /// One `[1, length, hidden]` array per span, same order as `spans`.
    let embeddings: [MLXArray]
    /// DeepStack embeddings ordered by language-layer injection point, then
    /// by `spans`.
    let deepstackEmbeddings: [[MLXArray]]

    /// The span context + per-span embedding selection for a prefill chunk
    /// `[start, start + count)`, or nil when the chunk contains no span
    /// tokens (the chunk then takes the untouched text path).
    func chunkContext(start: Int, count: Int) -> CBv2SpanChunkContext? {
        guard attention == .bidirectionalSpans else { return nil }
        let end = start + count
        let contained = blocks.filter { $0.tokenOffset >= start && $0.end <= end }
        // Scheduler invariant: a block intersecting the chunk is fully
        // contained in it. Straddling would silently produce wrong masks.
        for block in blocks {
            let intersects = block.tokenOffset < end && block.end > start
            precondition(
                !intersects || (block.tokenOffset >= start && block.end <= end),
                "CBv2 multimodal: block \(block) straddles chunk [\(start), \(end)) — scheduler must snap chunk boundaries to block edges"
            )
        }
        guard !contained.isEmpty else { return nil }
        return CBv2SpanChunkContext(chunkEnd: end, blocks: contained)
    }

    /// Spans fully contained in `[start, start + count)`, paired with their
    /// embedding arrays (same containment invariant as `chunkContext`).
    func spansInChunk(
        start: Int, count: Int
    ) -> [(span: CBv2ImageSpan, embedding: MLXArray)] {
        spansInChunk(embeddings: embeddings, start: start, count: count)
    }

    private func spansInChunk(
        embeddings source: [MLXArray], start: Int, count: Int
    ) -> [(span: CBv2ImageSpan, embedding: MLXArray)] {
        let end = start + count
        return zip(spans, source).compactMap { span, embedding in
            let lo = max(start, span.tokenOffset)
            let hi = min(end, span.end)
            guard lo < hi else { return nil }
            if lo == span.tokenOffset && hi == span.end {
                return (span, embedding)
            }
            precondition(
                attention == .causal,
                "CBv2 multimodal: bidirectional span split across a chunk")
            let relativeLo = lo - span.tokenOffset
            let relativeHi = hi - span.tokenOffset
            return (
                CBv2ImageSpan(tokenOffset: lo, length: hi - lo),
                embedding[0..., relativeLo ..< relativeHi, 0...])
        }
    }

    func deepstackInChunk(
        start: Int, count: Int, hidden: Int, dtype: DType
    ) -> [MLXArray] {
        deepstackEmbeddings.map { layer in
            CBv2MultimodalPlan.spliceEmbeddings(
                textEmbeddings: MLXArray.zeros([1, count, hidden], dtype: dtype),
                chunkStart: start,
                spans: spansInChunk(embeddings: layer, start: start, count: count))
        }
    }

    /// True when any span intersects `[start, start + count)` — the same
    /// test `spansInChunk` applies, without building the array or slicing
    /// embeddings. For per-row-per-step classification on the engine thread.
    func hasSpans(start: Int, count: Int) -> Bool {
        let end = start + count
        // Exactly `spansInChunk`'s non-empty test (`lo < hi`), so degenerate
        // ranges agree: an empty chunk never intersects and a zero-length
        // span never counts.
        return spans.contains { max(start, $0.tokenOffset) < min(end, $0.end) }
    }

    /// True when `position` lies inside a coalesced image block. Used by the
    /// engine's decode-row classification: a length-1 span landing on the
    /// FINAL prompt token produces an assignment shaped exactly like a decode
    /// row (n == 1, samples, last token), but its placeholder token must
    /// still take the embedding-splice path (PR#63 review). Genuine decode
    /// rows never trip this — their positions are past every span (spans
    /// live inside the prompt).
    func containsSpan(at position: Int) -> Bool {
        spans.contains { $0.tokenOffset <= position && position < $0.end }
    }
}

// MARK: - Validation, coalescing, splicing

enum CBv2MultimodalPlan {

    /// Coalesce adjacent spans into maximal contiguous placeholder runs.
    /// Ground truth: MLXVLM Gemma4's `gemma4VisionBlockIds` assigns one
    /// block id per contiguous run of vision tokens, so adjacent images with
    /// no separator token attend bidirectionally as a single block. Chunk
    /// snapping and mask construction both operate on these blocks.
    /// Precondition: spans sorted, non-overlapping (validated first).
    static func coalescedBlocks(spans: [CBv2ImageSpan]) -> [CBv2ImageSpan] {
        var blocks: [CBv2ImageSpan] = []
        for span in spans {
            if var last = blocks.last, last.end == span.tokenOffset {
                last.length += span.length
                blocks[blocks.count - 1] = last
            } else {
                blocks.append(span)
            }
        }
        return blocks
    }

    /// CHEAP submit-time validation — no provider call, no MLX graph work:
    /// model + backend capability, span structure, block-vs-budget. Returns
    /// the coalesced blocks. Split from `materialize` so `EngineV2.submit`
    /// can run every cheap admission gate (`canEverFit`, `maxWaiting`,
    /// duplicate id) BEFORE the vision tower's embeddings are materialized —
    /// a rejected request must never pay for heavy image work (PR#63
    /// review).
    static func validate(
        _ input: CBv2MultimodalInput,
        promptTokenCount: Int,
        model: CBv2SteppableModel,
        cacheProvider: CBv2LayerCacheProvider,
        maxBatchedTokensPerStep: Int
    ) throws -> [CBv2ImageSpan] {
        guard let mmModel = model as? CBv2MultimodalSteppableModel,
            mmModel.supportsMultimodalPrefill(attention: input.attention)
        else {
            throw CBv2MultimodalError.unsupportedModel(
                "\(type(of: model)) does not support embedding-spliced prefill")
        }
        // All production VLMs currently use contiguous KV. Causal visual
        // tokens need no span-mask binding, but retaining the backend gate
        // keeps the media path's compact/position semantics off unproven
        // paged caches until an explicit paged VLM test exists.
        guard cacheProvider.supportsMultimodalSpans else {
            throw CBv2MultimodalError.unsupportedBackend(
                "\(type(of: cacheProvider)) cannot honor multimodal prefill")
        }

        let spans = input.spans
        guard !spans.isEmpty else {
            throw CBv2MultimodalError.invalidSpans("multimodal input carries no spans")
        }
        var previousEnd = 0
        for (i, span) in spans.enumerated() {
            guard span.length > 0 else {
                throw CBv2MultimodalError.invalidSpans("span \(i) has non-positive length")
            }
            guard span.tokenOffset >= previousEnd else {
                throw CBv2MultimodalError.invalidSpans(
                    "span \(i) at \(span.tokenOffset) overlaps or is out of order (previous end \(previousEnd))"
                )
            }
            guard span.end <= promptTokenCount else {
                throw CBv2MultimodalError.invalidSpans(
                    "span \(i) [\(span.tokenOffset), \(span.end)) exceeds prompt length \(promptTokenCount)"
                )
            }
            previousEnd = span.end
        }

        let blocks = input.attention == .bidirectionalSpans
            ? coalescedBlocks(spans: spans) : []
        if let oversized = blocks.first(where: { $0.length > maxBatchedTokensPerStep }) {
            throw CBv2MultimodalError.spanTooLong(
                blockTokens: oversized.length,
                maxBatchedTokensPerStep: maxBatchedTokensPerStep)
        }
        return blocks
    }

    /// Full submit-time validation + materialization. Throws
    /// `CBv2MultimodalError`; never partially succeeds.
    static func resolve(
        _ input: CBv2MultimodalInput,
        promptTokenCount: Int,
        model: CBv2SteppableModel,
        cacheProvider: CBv2LayerCacheProvider,
        maxBatchedTokensPerStep: Int
    ) throws -> CBv2ResolvedMultimodal {
        let blocks = try validate(
            input, promptTokenCount: promptTokenCount, model: model,
            cacheProvider: cacheProvider,
            maxBatchedTokensPerStep: maxBatchedTokensPerStep)
        return try materialize(input, blocks: blocks, model: model)
    }

    /// HEAVY half of `resolve`: run the embeddings provider and shape-check
    /// its output. `blocks` must come from `validate` on the same input.
    static func materialize(
        _ input: CBv2MultimodalInput,
        blocks: [CBv2ImageSpan],
        model: CBv2SteppableModel
    ) throws -> CBv2ResolvedMultimodal {
        guard let mmModel = model as? CBv2MultimodalSteppableModel else {
            // Unreachable after `validate`; kept as the type-safety backstop.
            throw CBv2MultimodalError.unsupportedModel(
                "\(type(of: model)) does not support embedding-spliced prefill")
        }
        let spans = input.spans

        // Materialize each provider exactly once on the submit thread.
        let provided = try input.embeddings()
        let providedDeepstack = try input.deepstackEmbeddings?() ?? []
        let hidden = mmModel.embedPromptTokens(
            MLXArray(Int32(0)).reshaped([1, 1])).dim(-1)

        func normalize(_ arrays: [MLXArray], label: String) throws -> [MLXArray] {
            guard arrays.count == spans.count else {
                throw CBv2MultimodalError.embeddingMismatch(
                    "\(label) returned \(arrays.count) arrays for \(spans.count) spans")
            }
            return try arrays.enumerated().map { i, array in
                let span = spans[i]
                let shaped: MLXArray
                switch array.ndim {
                case 2: shaped = array.expandedDimensions(axis: 0)
                case 3 where array.dim(0) == 1: shaped = array
                default:
                    throw CBv2MultimodalError.embeddingMismatch(
                        "\(label) span \(i) has shape \(array.shape); expected [length, hidden] or [1, length, hidden]")
                }
                guard shaped.dim(1) == span.length else {
                    throw CBv2MultimodalError.embeddingMismatch(
                        "\(label) span \(i) covers \(shaped.dim(1)) tokens; span length is \(span.length)")
                }
                guard shaped.dim(2) == hidden else {
                    throw CBv2MultimodalError.embeddingMismatch(
                        "\(label) span \(i) hidden dim \(shaped.dim(2)) != model hidden dim \(hidden)")
                }
                return shaped
            }
        }

        let normalized = try normalize(provided, label: "provider")
        let normalizedDeepstack = try providedDeepstack.enumerated().map {
            try normalize($0.element, label: "DeepStack layer \($0.offset)")
        }
        let expectedDeepstackLayers =
            (model as? CBv2DeepstackMultimodalSteppableModel)?.deepstackLayerCount ?? 0
        guard normalizedDeepstack.count == expectedDeepstackLayers else {
            throw CBv2MultimodalError.embeddingMismatch(
                "DeepStack returned \(normalizedDeepstack.count) layers; model expects \(expectedDeepstackLayers)")
        }
        if !normalizedDeepstack.isEmpty,
            !(model is CBv2DeepstackMultimodalSteppableModel)
        {
            throw CBv2MultimodalError.unsupportedModel(
                "\(type(of: model)) received DeepStack embeddings without DeepStack support")
        }

        return CBv2ResolvedMultimodal(
            spans: spans, attention: input.attention, blocks: blocks,
            embeddings: normalized, deepstackEmbeddings: normalizedDeepstack)
    }

    /// Splice span embeddings over the scaled text embeddings of one prefill
    /// chunk. `textEmbeddings` is `[1, L, hidden]` for prompt positions
    /// `[chunkStart, chunkStart + L)`; every span is fully contained
    /// (scheduler invariant). Pure concatenation of interleaved text/span
    /// segments — value-identical to the MLXVLM wrapper's `maskedScatter`
    /// (span rows replaced verbatim, cast to the text embedding dtype).
    static func spliceEmbeddings(
        textEmbeddings: MLXArray,
        chunkStart: Int,
        spans: [(span: CBv2ImageSpan, embedding: MLXArray)]
    ) -> MLXArray {
        guard !spans.isEmpty else { return textEmbeddings }
        let L = textEmbeddings.dim(1)
        var segments: [MLXArray] = []
        segments.reserveCapacity(spans.count * 2 + 1)
        var cursor = 0  // chunk-relative
        for (span, embedding) in spans {
            let lo = span.tokenOffset - chunkStart
            let hi = lo + span.length
            precondition(
                lo >= cursor && hi <= L,
                "CBv2 multimodal: span \(span) not contained in chunk [\(chunkStart), \(chunkStart + L))"
            )
            if lo > cursor {
                segments.append(textEmbeddings[0..., cursor ..< lo, 0...])
            }
            segments.append(embedding.asType(textEmbeddings.dtype))
            cursor = hi
        }
        if cursor < L {
            segments.append(textEmbeddings[0..., cursor ..< L, 0...])
        }
        return segments.count == 1 ? segments[0] : concatenated(segments, axis: 1)
    }
}
