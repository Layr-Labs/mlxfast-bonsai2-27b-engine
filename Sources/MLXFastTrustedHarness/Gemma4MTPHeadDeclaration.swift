import Foundation
import MLXFastCore

// Head delivery for the native-MTP track.
//
// OPERATOR-RATIFIED 2026-08-14. The MTP head is part of the competitive
// surface: a submission may bring its own. It declares one in
// `mtp-head.manifest.json`, which is an editable path, and the RUNNER resolves
// that declaration pre-sandbox the way it resolves a hidden golden -- refuse on
// oversize. A declared digest is parsed and carried when present but is NOT
// verified against the head bytes and is not a gate. Per DECIDE-2 (Q-B) a
// bring-your-own head is NOT required to declare one, and a digestless head and
// a wrong-digest head are treated identically: both are bounded by SIZE ONLY.
//
// THE SAFETY ARGUMENT, in one line: a head only PROPOSES tokens. The
// organizer-pinned target decides every emitted token and the trusted parent
// re-checks the whole stream against a hidden serial trajectory after the clock
// stops, so a substituted head moves the accept rate -- which is the game --
// and cannot move the output.
//
// THIS FILE IS TRUSTED CODE. It parses the declaration and applies the size
// gate; it never decides whether a run passes.
//
// TWO DECODERS, ONE MANIFEST (bonsai2-27b-mlx-v1). The track declares two
// speculative decoders and one declaration file. `spec.decoder` selects which
// one a submission arms: `"mtp"` (the pinned MTP head) or `"dflash"` (the
// pinned DFlash 2 drafter). The key is optional and defaults to `"mtp"`, so a
// declaration written before the DFlash 2 arm existed keeps its meaning.
//
// THE DECODER SELECTS THE SIZE CAP. The two drafters are three orders of
// magnitude apart -- 239 MB of 4-bit MTP head against 3.85 GB of BF16 DFlash 2
// drafter -- so one cap cannot bound both. Each decoder carries its own track
// cap, and `max_bytes` may lower the cap of the DECLARED decoder and may not
// raise it. The MTP head's 2 GiB cap is unchanged.
//
// THERE IS NO `arm` SELECTION KEY. `spec.decoder` is the selection, and it
// lives beside the depth it goes with. `arm` is an unread key like any other
// unknown key.
//
// NEITHER DRAFTER IS STAGED OR MEASURED HERE. Both are SEPARATE pinned exports
// staged beside the target by `./setup-mtp-head.sh` and
// `./setup-dflash-drafter.sh`, so `source: "pinned"` means "the export this
// track pins for the declared decoder". A submission carries no drafter
// weights, so this file has no directory to digest and no artifact to fetch.
// It parses the declaration and applies the size gate.

/// One parsed head declaration (`mtp-head.manifest.json`).
public struct Gemma4MTPHeadDeclaration: Equatable, Sendable {
    public enum Source: String, Equatable, CaseIterable, Sendable {
        /// The head embedded in the organizer's pinned target checkpoint. The
        /// default, and what an ABSENT manifest means.
        case pinned
        /// Fetched by the runner from `sourceURL` (`hf:<repo>@<rev>` or
        /// `r2:<key>`) into the run's private directory.
        case remote
        /// Shipped in the submission under the editable weights directory
        /// named by `path`.
        case inBranch = "in_branch"
    }

    /// Which speculative decoder the declaration arms.
    public enum Decoder: String, Equatable, CaseIterable, Sendable {
        /// The pinned MTP head, `EigenLabs/Qwen3.8-27B-MTP-4bit`.
        case mtp
        /// The pinned DFlash 2 block drafter, `z-lab/Qwen3.8-27B-DFlash2`.
        case dflash

        /// The track cap on what this decoder's declaration may bound. A
        /// declaration may lower its own decoder's cap and may not raise it.
        ///
        /// The two caps are independent by design: raising the MTP head's cap
        /// to hold the DFlash 2 drafter would widen what a re-quantized MTP
        /// head may declare, which is a separate ruling nobody made.
        public var trackMaxBytes: Int {
            switch self {
            case .mtp: return 2_147_483_648
            case .dflash: return 4_294_967_296
            }
        }

        /// How a refusal names this decoder's drafter.
        public var declarationNoun: String {
            switch self {
            case .mtp: return "MTP"
            case .dflash: return "DFlash 2"
            }
        }
    }

    public let decoder: Decoder
    public let source: Source
    public let sourceURL: String?
    public let path: String?
    public let sha256: String?
    public let bytes: Int
    public let maxBytes: Int

    public init(
        source: Source,
        decoder: Decoder = .mtp,
        sourceURL: String? = nil,
        path: String? = nil,
        sha256: String? = nil,
        bytes: Int = 0,
        maxBytes: Int? = nil
    ) {
        self.decoder = decoder
        self.source = source
        self.sourceURL = sourceURL
        self.path = path
        self.sha256 = sha256
        self.bytes = bytes
        self.maxBytes = maxBytes ?? decoder.trackMaxBytes
    }

    /// 2 GiB, the MTP head's cap. Mirrored in `mtp-head.manifest.json`
    /// (`max_bytes`) and in the contract manifest's
    /// `editableSurfaceByteBudget.exemptPathMaxBytes`; a declaration may lower
    /// it and may not raise it.
    public static let defaultMaxBytes = Decoder.mtp.trackMaxBytes

    /// 4 GiB, the DFlash 2 drafter's cap. The pinned drafter is 3,848,808,960
    /// tensor bytes of BF16, which no 2 GiB cap can hold.
    public static let dflashMaxBytes = Decoder.dflash.trackMaxBytes

    /// The default when no manifest exists at all.
    public static let pinnedDefault = Gemma4MTPHeadDeclaration(source: .pinned)

    public static let relativePath = "mtp-head.manifest.json"

    /// How a refusal names the head when no declaration has been parsed yet.
    public static let declarationNoun = Decoder.mtp.declarationNoun

    /// Parse a declaration, FAILING CLOSED on anything malformed.
    ///
    /// Only two things select the pinned head: the file being absent, and an
    /// explicit `source: "pinned"`. A manifest that is present but unreadable,
    /// unparseable, or internally inconsistent is a REFUSAL -- never a silent
    /// fall back -- because "your head declaration was broken so we quietly
    /// scored you on the pinned head" is exactly the failure mode that makes a
    /// leaderboard number unattributable.
    public static func parse(
        contentsOf url: URL
    ) throws -> Gemma4MTPHeadDeclaration {
        guard let data = FileManager.default.contents(atPath: url.path) else {
            throw MLXFastError.invalidInput(
                "the \(declarationNoun) head declaration at \(url.path) "
                    + "exists but could not be read; refusing to fall back to "
                    + "the pinned head")
        }
        return try parse(data: data, origin: url.path)
    }

    public static func parse(
        data: Data,
        origin: String
    ) throws -> Gemma4MTPHeadDeclaration {
        guard let root = (try? JSONSerialization.jsonObject(with: data))
            as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the \(declarationNoun) head declaration at \(origin) is not a "
                    + "JSON object")
        }
        // The decoder is read FIRST: it names the drafter every later refusal
        // talks about, and it selects the size cap the declaration is bounded
        // by. An absent key is `mtp`, so a declaration written before the
        // DFlash 2 arm existed keeps its meaning.
        let spec = root["spec"] as? [String: Any]
        let rawDecoder = (spec?["decoder"] as? String) ?? Decoder.mtp.rawValue
        guard let decoder = Decoder(rawValue: rawDecoder) else {
            throw MLXFastError.invalidInput(
                "the head declaration at \(origin) names an unknown decoder "
                    + "'\(rawDecoder)'; this track declares "
                    + Decoder.allCases.map(\.rawValue).joined(separator: ", "))
        }
        let noun = decoder.declarationNoun

        let rawSource = (root["source"] as? String) ?? Source.pinned.rawValue
        guard let source = Source(rawValue: rawSource) else {
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) names an unknown source "
                    + "'\(rawSource)'; expected one of "
                    + Source.allCases.map(\.rawValue).joined(separator: ", "))
        }
        let trackMaxBytes = decoder.trackMaxBytes
        let maxBytes = (root["max_bytes"] as? NSNumber)?.intValue
            ?? trackMaxBytes
        guard maxBytes > 0, maxBytes <= trackMaxBytes else {
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) sets max_bytes "
                    + "\(maxBytes); it must be positive and may not exceed the "
                    + "\(decoder.rawValue) track cap \(trackMaxBytes)")
        }
        let sourceURL = root["source_url"] as? String
        let path = root["path"] as? String
        let sha256 = (root["sha256"] as? String)?.lowercased()
        let bytes = (root["bytes"] as? NSNumber)?.intValue ?? 0

        // REQUANT-ONLY (David ruling, 2026-08-26). `pinned` is the ONLY source
        // this track accepts. The head is the organizer's own weights; a
        // participant may declare a re-quantization of them and may not
        // substitute weights of their own. `remote` and `in_branch` are the two
        // spellings of "load bytes the participant chose", so both are named
        // refusals rather than gated flows.
        //
        // WHY THE CASES STAY IN THE ENUM. Deleting them would make a manifest
        // that names one fail as "unknown source", which reads like a typo. The
        // participant did not typo; they used a mode this track retired, and the
        // refusal should say so and say what replaced it.
        switch source {
        case .pinned:
            break
        case .remote:
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) selects source "
                    + "'remote'; this track accepts source 'pinned' only. The "
                    + "\(noun) drafter is a separate pinned export "
                    + "(fixtures/bonsai2_27b_mlx_v1_track.json), and custom "
                    + "drafter weights are not accepted. Declare a "
                    + "re-quantization of the pinned head instead")
        case .inBranch:
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) selects source "
                    + "'in_branch'; this track accepts source 'pinned' only. A "
                    + "submission carries no drafter weights -- both drafters "
                    + "are staged from their own pins -- so there is nothing an "
                    + "in-branch path could name. Declare a re-quantization of "
                    + "the pinned head instead")
        }

        // THE SIZE GATE SURVIVES THE NARROWING. It used to be reachable only
        // for a non-pinned source, which after the ruling would make it dead
        // code -- and deleting it would silently drop the one numeric bound a
        // declaration still carries. A `pinned` declaration may state the
        // `bytes` of the head it expects (a re-quantized head is smaller than
        // the shipped one, and stating it is how a participant records what they
        // expect), so the bound is "if you state a size, it must fit the cap"
        // rather than "non-pinned sources must state one". `bytes: 0` means
        // "not stated", which is what the checked-in declaration says.
        if bytes != 0 {
            guard bytes > 0 else {
                throw MLXFastError.invalidInput(
                    "the \(noun) head declaration at \(origin) states bytes "
                        + "\(bytes); a stated byte count must be positive")
            }
            guard bytes <= maxBytes else {
                throw MLXFastError.invalidInput(
                    "the declared \(noun) head is \(bytes) bytes, above the "
                        + "\(maxBytes)-byte cap in \(origin)")
            }
        }
        return Gemma4MTPHeadDeclaration(
            source: source,
            decoder: decoder,
            sourceURL: sourceURL,
            path: path,
            sha256: sha256,
            bytes: bytes,
            maxBytes: maxBytes
        )
    }

    /// Read the declaration next to a contract root, treating ABSENCE as the
    /// pinned default and everything else as parse-or-refuse.
    public static func resolve(
        contractRoot: URL
    ) throws -> Gemma4MTPHeadDeclaration {
        let url = contractRoot.appendingPathComponent(relativePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return pinnedDefault
        }
        return try parse(contentsOf: url)
    }
}
