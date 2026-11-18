"""SourceLock — hallucination check for AI-generated summaries.

Verifies that each claim in a summary is actually supported by real text in
the source posts it claims to be drawn from. Similarity is TF-IDF cosine over
the row's source pool (rare shared terms count more than function words, which
are removed by ``common.tokenize``), with an exact-substring shortcut at
confidence 1.0.

v0.2.0 adds atomic subclaim decomposition, number/date evidence checks, and
three explicitly *lexical* groundedness signals. They are triage signals, not
proof of entailment: see ``summary["limitations"]`` and the README.
"""

from __future__ import annotations

import math
import re
from typing import Any

from sourcelock.common import (
    RESULT_SCHEMA_VERSION,
    ToolResult,
    as_str,
    find_id,
    get,
    opt_float,
    opt_int,
    tokenize,
)
from sourcelock.stats import TfIdf, finding_envelope

CLAIM_COLUMNS = ["claim", "summary", "claim_text", "statement", "claim_sentence", "assertion"]
SOURCE_COLUMNS = [
    "source_posts",
    "sources",
    "source_text",
    "source_content",
    "source",
    "posts",
    "evidence",
]

METHOD_V2 = "lexical-v2"
LIMITATIONS = ["implicit_contradiction", "multi_hop", "no_semantic_entailment"]

# Negation markers: lexical overlap cannot tell "we released" from
# "we did not release", so asymmetric negation always needs human review.
_NEGATION_RE = re.compile(
    r"\b(not|no|never|none|neither|nor|without|hardly|scarcely|barely|didn'?t|"
    r"don'?t|doesn'?t|isn'?t|aren'?t|wasn'?t|weren'?t|haven'?t|hasn'?t|hadn'?t|"
    r"won'?t|wouldn'?t|couldn'?t|shouldn'?t|can'?t|cannot|kein|keine|nicht|nie|"
    r"non|ne|pas|jamais|sans)\b",
    re.IGNORECASE,
)


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


# ---------------------------------------------------------------------------
# Sentence / subclaim splitting
# ---------------------------------------------------------------------------

# Dots inside decimals and abbreviations are not sentence boundaries.
_DOT_SENTINEL = "\u0001"
_COMMA_SENTINEL = "\u0002"
_ABBREV_RE = re.compile(
    r"\b(?:[A-Za-z]\.){2,}|\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|approx|Fig|No)\.",
    re.IGNORECASE,
)
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")

# ASCII plus fullwidth CJK sentence punctuation (\u3002, \uff01, \uff1f).
_SENTENCE_SPLIT = re.compile(r"[\u3002\uff01\uff1f!?.;\n]+")
_SUBCLAIM_CONNECTOR_RE = re.compile(r"\s+\b(?:and|but|which|who|while)\b\s+", re.IGNORECASE)
_SUBCLAIM_PUNCT_RE = re.compile(r"\s*[,;]\s*")


def _protect_dots(text: str) -> str:
    masked = _ABBREV_RE.sub(lambda match: match.group(0).replace(".", _DOT_SENTINEL), text)
    return _DECIMAL_DOT_RE.sub(_DOT_SENTINEL, masked)


def _restore_dots(text: str) -> str:
    return text.replace(_DOT_SENTINEL, ".").replace(_COMMA_SENTINEL, ",")


def _split_sources(value: str) -> list[str]:
    chunks: list[str] = []
    for part in re.split(r"[|\n;]+", _protect_dots(value)):
        for sentence in _SENTENCE_SPLIT.split(part):
            cleaned = _restore_dots(sentence).strip()
            if cleaned:
                chunks.append(cleaned)
    return chunks


def _split_claims(value: str) -> list[str]:
    return [
        _restore_dots(sentence).strip()
        for sentence in _SENTENCE_SPLIT.split(_protect_dots(value))
        if sentence.strip()
    ]


def _split_subclaims(sentence: str) -> list[str]:
    """Split a sentence into atomic subclaims, keeping the sentence itself.

    The original sentence is never replaced — it is still scored as a whole;
    these fragments are extra probes for compound hallucinations.
    """
    masked = _THOUSANDS_RE.sub(_COMMA_SENTINEL, _protect_dots(sentence))
    masked = _SUBCLAIM_PUNCT_RE.sub("\n", masked)
    masked = _SUBCLAIM_CONNECTOR_RE.sub("\n", masked)
    return [part for line in masked.split("\n") if (part := _restore_dots(line).strip())]


# ---------------------------------------------------------------------------
# Numbers / dates
# ---------------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_RE = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?:\s*(?:%|percent|units?|items?|pcs?|kg|km|cm|mm|gb|mb|tb|"
    r"ms|hours?|days?|weeks?|months?|years?|users?|customers?|tickets?|posts?|"
    r"euros?|dollars?))?(?!\w|\.\d)",
    re.IGNORECASE,
)


def _extract_numbers(text: str) -> list[str]:
    """Return normalized numeric literals: ``100``, ``3.5%``, ``10 units``, dates."""
    found: set[str] = set()

    def _collect_date(match: re.Match[str]) -> str:
        date = match.group(0)
        found.update((date, date[:4]))  # the bare year also counts as evidence
        return " "

    cleaned = _ISO_DATE_RE.sub(_collect_date, text)
    # Thousands separators are formatting, not a subclaim/number boundary.
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)
    found.update(" ".join(match.group(0).lower().split()) for match in _NUMBER_RE.finditer(cleaned))
    return sorted(found)


# ---------------------------------------------------------------------------
# Similarity / signals
# ---------------------------------------------------------------------------


def _chunk_scores(claim: str, sources: list[str]) -> list[tuple[float, str]]:
    """Best score per source chunk, sorted high to low (stable on ties)."""
    claim_tokens = tokenize(claim)
    usable = [source.strip() for source in sources if source.strip()]
    if not claim_tokens or not usable:
        return []
    lowered = claim.lower()
    space = TfIdf([tokenize(source) for source in usable])
    claim_vec = space.vector(claim_tokens)
    scores: list[tuple[float, str]] = []
    for source in usable:
        if lowered in source.lower():
            scores.append((1.0, source))
        else:
            scores.append((TfIdf.cosine(claim_vec, space.vector(tokenize(source))), source))
    return sorted(scores, key=lambda pair: pair[0], reverse=True)


def _best_overlap(claim: str, sources: list[str]) -> tuple[float, str]:
    scores = _chunk_scores(claim, sources)
    return scores[0] if scores else (0.0, "")


def _entailment_probability(cosine: float) -> float:
    """Monotone map of lexical cosine into [0, 1] — a proxy, not a probability."""
    return min(1.0, max(0.0, 1.0 - math.exp(-3.0 * max(0.0, cosine))))


def _signals(
    confidence: float,
    subclaim_confidences: list[float],
    subclaim_conflicts: list[bool],
    threshold: float,
) -> dict[str, float]:
    total = len(subclaim_confidences)
    return {
        "entailment_probability": round(_entailment_probability(confidence), 3),
        "contradiction_rate": round(sum(subclaim_conflicts) / total, 3) if total else 0.0,
        "missing_evidence_frequency": round(sum(c < threshold for c in subclaim_confidences) / total, 3)
        if total
        else 0.0,
    }


def _metadata(
    claim: str,
    confidence: float,
    threshold: float,
    method: str,
    snippet: str,
    subclaims: list[dict[str, Any]],
    signals: dict[str, float],
    **extra: Any,
) -> dict[str, Any]:
    """Shared finding metadata: legacy keys plus v0.2.0 honest-signal labels."""
    metadata: dict[str, Any] = {
        "claim": claim,
        "confidence": round(confidence, 3),
        "threshold": threshold,
        "method": method,
        "method_v2": METHOD_V2,
        "best_snippet": snippet[:200],
        "subclaims": subclaims,
        "signals": signals,
        "confidence_label": "lexical_proxy",
        "triage_only": True,
    }
    metadata.update(extra)
    return metadata


class SourceLockLogic:
    """Claim-by-claim source verification with lexical groundedness signals."""

    def run(self, rows: list[dict], options: dict[str, Any]) -> ToolResult:
        threshold = min(1.0, max(0.0, opt_float(options, "threshold", 0.30)))
        min_chars = max(2, opt_int(options, "min_claim_chars", 4))

        findings: list[dict] = []
        warnings: list[str] = []
        checked = 0
        claims_checked = 0
        supported = 0
        unsupported = 0
        confidence_sum = 0.0
        entailment_sum = 0.0
        contradiction_sum = 0.0
        missing_sum = 0.0
        skipped = 0
        claimless_rows = 0
        all_short_rows = 0

        for row_index, row in enumerate(rows, start=1):
            claim_raw = as_str(get(row, CLAIM_COLUMNS))
            source_raw = as_str(get(row, SOURCE_COLUMNS))
            record_id = find_id(row)
            row_label = record_id or f"row {row_index}"

            if not claim_raw:
                claimless_rows += 1
                warnings.append(f"Row {row_label}: no claim text found; skipping.")
                continue
            if not source_raw:
                skipped += 1
                warnings.append(
                    f"No source text found for claim '{claim_raw[:60]}…' "
                    "(add a source_posts / sources column)."
                )
                continue

            checked += 1
            claims = _split_claims(claim_raw)
            sources = _split_sources(source_raw)
            source_numbers = {number for source in sources for number in _extract_numbers(source)}
            row_claims_checked = 0

            for claim in claims:
                if len(claim) < min_chars:
                    continue
                row_claims_checked += 1
                claims_checked += 1

                chunk_scores = _chunk_scores(claim, sources)
                confidence, snippet = chunk_scores[0] if chunk_scores else (0.0, "")
                confidence_sum += confidence

                # Atomic subclaims: probe compound statements without losing
                # the whole-sentence score.
                subclaim_texts = [text for text in _split_subclaims(claim) if len(text) >= min_chars]
                if not subclaim_texts:
                    subclaim_texts = [claim]
                subclaim_infos: list[dict[str, Any]] = []
                subclaim_confidences: list[float] = []
                subclaim_conflicts: list[bool] = []
                for subclaim in subclaim_texts:
                    sub_confidence, _ = _best_overlap(subclaim, sources)
                    sub_numbers = set(_extract_numbers(subclaim))
                    numeric_conflict = any(number not in source_numbers for number in sub_numbers)
                    negation_conflict = any(
                        _has_negation(chunk) != _has_negation(subclaim) for _, chunk in chunk_scores
                    )
                    subclaim_infos.append(
                        {
                            "text": subclaim,
                            "confidence": round(sub_confidence, 3),
                            "supported": sub_confidence >= threshold,
                        }
                    )
                    subclaim_confidences.append(sub_confidence)
                    subclaim_conflicts.append(numeric_conflict or negation_conflict)

                signals = _signals(confidence, subclaim_confidences, subclaim_conflicts, threshold)
                entailment_sum += signals["entailment_probability"]
                contradiction_sum += signals["contradiction_rate"]
                missing_sum += signals["missing_evidence_frequency"]

                claim_numbers = _extract_numbers(claim)
                missing_numbers = [number for number in claim_numbers if number not in source_numbers]

                # Negation gate now scans *every* source chunk, not just the
                # best lexical match.
                negation_source = ""
                for _, chunk in chunk_scores:
                    if _has_negation(chunk) != _has_negation(claim):
                        negation_source = chunk
                        break

                if negation_source:
                    unsupported += 1
                    findings.append(
                        finding_envelope(
                            "high" if confidence < threshold * 0.5 else "medium",
                            "claim",
                            f"Claim '{claim[:90]}' differs in negation from source chunk "
                            f"'{negation_source[:90]}' — similarity scores can't tell "
                            "'we released' from 'we did not release'. Verify manually.",
                            record_id,
                            _metadata(
                                claim,
                                confidence,
                                threshold,
                                "tfidf-cosine+negation",
                                snippet,
                                subclaim_infos,
                                signals,
                                negation_mismatch=True,
                                negation_source=negation_source[:200],
                            ),
                        )
                    )
                elif confidence >= threshold:
                    supported += 1
                else:
                    unsupported += 1
                    findings.append(
                        finding_envelope(
                            "high" if confidence < threshold * 0.5 else "medium",
                            "claim",
                            f"Claim '{claim[:90]}' is not supported by any source post "
                            f"(similarity {confidence:.2f} < threshold {threshold:.2f}).",
                            record_id,
                            _metadata(
                                claim,
                                confidence,
                                threshold,
                                "tfidf-cosine",
                                snippet,
                                subclaim_infos,
                                signals,
                            ),
                        )
                    )

                if missing_numbers:
                    example = claim[:200]
                    findings.append(
                        finding_envelope(
                            "medium",
                            "claim",
                            f"Claim '{claim[:90]}' contains number(s)/date(s) not found in any "
                            f"source chunk: {', '.join(missing_numbers)}. Verify manually.",
                            record_id,
                            _metadata(
                                claim,
                                confidence,
                                threshold,
                                "lexical-v2+numbers",
                                snippet,
                                subclaim_infos,
                                signals,
                                unsupported_number=True,
                                claim_numbers=claim_numbers,
                                source_numbers=sorted(source_numbers),
                                missing_numbers=missing_numbers,
                                example=example,
                                example_text=example,
                            ),
                        )
                    )

                weak_subclaims = [
                    info["text"] for info in subclaim_infos if not info["supported"]
                ]
                if confidence >= threshold and weak_subclaims:
                    findings.append(
                        finding_envelope(
                            "high",
                            "claim",
                            f"Claim '{claim[:90]}' scores as supported overall, but atomic "
                            f"subclaim(s) are not: {'; '.join(weak_subclaims)[:180]}.",
                            record_id,
                            _metadata(
                                claim,
                                confidence,
                                threshold,
                                "lexical-v2+subclaims",
                                snippet,
                                subclaim_infos,
                                signals,
                                unsupported_subclaim=True,
                                unsupported_subclaims=weak_subclaims,
                            ),
                        )
                    )

            if not row_claims_checked:
                all_short_rows += 1
                warnings.append(
                    f"Row {row_label}: all claims are shorter than min_claim_chars={min_chars}; "
                    "nothing checked."
                )

        if checked == 0:
            warnings.append(
                "No rows carried both a claim and source text "
                "(expected a claim/summary column plus a source_posts/sources column)."
            )

        skipped_flag = claims_checked == 0
        if not skipped_flag:
            skip_reason = None
        elif not rows:
            skip_reason = "no_rows"
        elif claimless_rows == len(rows):
            skip_reason = "no_claim_text"
        elif skipped == len(rows):
            skip_reason = "missing_source_text"
        elif all_short_rows:
            skip_reason = "claims_below_min_length"
        else:
            skip_reason = "no_usable_rows"

        summary: dict[str, Any] = {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "checked": checked,
            "claims_checked": claims_checked,
            "supported": supported,
            "unsupported": unsupported,
            "skipped_missing_sources": skipped,
            "skipped": skipped_flag,
            "skip_reason": skip_reason,
            "avg_confidence": round(confidence_sum / claims_checked, 3) if claims_checked else None,
            "entailment_probability": round(entailment_sum / claims_checked, 3) if claims_checked else None,
            "contradiction_rate": round(contradiction_sum / claims_checked, 3) if claims_checked else None,
            "missing_evidence_frequency": round(missing_sum / claims_checked, 3) if claims_checked else None,
            "method": METHOD_V2,
            "limitations": list(LIMITATIONS),
        }
        return ToolResult(
            status="success",
            findings=findings,
            warnings=warnings,
            summary=summary,
        )


def run(rows: list[dict], options: dict[str, Any] | None = None) -> ToolResult:
    return SourceLockLogic().run(rows, options or {})
