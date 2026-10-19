"""SourceLock — hallucination check for AI-generated summaries.

Verifies that each claim in a summary is actually supported by real text in
the source posts it claims to be drawn from, returning a confidence score per
claim and the best-matching source snippet. Similarity is TF-IDF cosine over
the row's source set (rare shared terms count more than filler words), with
an exact-substring shortcut at confidence 1.0.
"""

from __future__ import annotations

import re
from typing import Any

from sourcelock.common import ToolResult
from sourcelock.common import as_str, find_id, get, tokenize
from sourcelock.stats import TfIdf

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


# ASCII plus fullwidth CJK sentence punctuation (\u3002, \uff01, \uff1f).
_SENTENCE_SPLIT = re.compile(r"[\u3002\uff01\uff1f!?.;\n]+")


def _split_sources(value: str) -> list[str]:
    parts = [p for p in re.split(r"[|\n;]+", value) if p.strip()]
    chunks: list[str] = []
    for part in parts:
        for sentence in _SENTENCE_SPLIT.split(part):
            cleaned = sentence.strip()
            if cleaned:
                chunks.append(cleaned)
    return chunks or (parts if parts else [])


def _split_claims(value: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(value) if s.strip()]


def _best_overlap(claim: str, sources: list[str]) -> tuple[float, str]:
    claim_tokens = tokenize(claim)
    if not claim_tokens:
        return 0.0, ""
    usable = [s for s in sources if s.strip()]
    if not usable:
        return 0.0, ""
    lowered = claim.lower()
    for source in usable:
        if lowered and lowered in source.lower():
            return 1.0, source.strip()
    space = TfIdf([tokenize(s) for s in usable])
    claim_vec = space.vector(claim_tokens)
    best = 0.0
    best_snippet = ""
    for source in usable:
        score = TfIdf.cosine(claim_vec, space.vector(tokenize(source)))
        if score > best:
            best = score
            best_snippet = source.strip()
    return best, best_snippet


class SourceLockLogic:
    """Claim-by-claim source verification."""

    def run(self, rows: list[dict], options: dict[str, Any]) -> ToolResult:
        threshold = max(0.0, min(1.0, float(options.get("threshold", 0.30))))
        min_chars = max(2, int(options.get("min_claim_chars", 4)))

        findings: list[dict] = []
        warnings: list[str] = []
        checked = 0
        supported = 0
        unsupported = 0
        confidence_sum = 0.0
        confidence_count = 0
        skipped = 0

        for row in rows:
            claim_raw = as_str(get(row, CLAIM_COLUMNS))
            source_raw = as_str(get(row, SOURCE_COLUMNS))
            record_id = find_id(row)

            if not claim_raw:
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

            for claim in claims:
                if len(claim) < min_chars:
                    continue
                confidence, snippet = _best_overlap(claim, sources)
                confidence_sum += confidence
                confidence_count += 1
                if _has_negation(claim) != _has_negation(snippet):
                    # Lexical similarity cannot adjudicate negation: the words
                    # match but the meaning may be inverted. Never bless it.
                    unsupported += 1
                    findings.append(
                        {
                            "severity": "medium",
                            "field": "claim",
                            "message": (
                                f"Claim '{claim[:90]}' differs from its best source match "
                                "in negation — similarity scores can't tell 'we released' "
                                "from 'we did not release'. Verify manually."
                            ),
                            "record_id": record_id,
                            "metadata": {
                                "claim": claim,
                                "confidence": round(confidence, 3),
                                "threshold": threshold,
                                "method": "tfidf-cosine+negation",
                                "negation_mismatch": True,
                                "best_snippet": snippet[:200],
                            },
                        }
                    )
                    continue
                if confidence >= threshold:
                    supported += 1
                    continue
                unsupported += 1
                findings.append(
                    {
                        "severity": "high" if confidence < threshold * 0.5 else "medium",
                        "field": "claim",
                        "message": (
                            f"Claim '{claim[:90]}' is not supported by any source post "
                            f"(similarity {confidence:.2f} < threshold {threshold:.2f})."
                        ),
                        "record_id": record_id,
                        "metadata": {
                            "claim": claim,
                            "confidence": round(confidence, 3),
                            "threshold": threshold,
                            "method": "tfidf-cosine",
                            "best_snippet": snippet[:200],
                        },
                    }
                )

        if checked == 0:
            warnings.append(
                "No rows carried both a claim and source text "
                "(expected a claim/summary column plus a source_posts/sources column)."
            )

        summary: dict[str, Any] = {
            "checked": checked,
            "supported": supported,
            "unsupported": unsupported,
            "skipped_missing_sources": skipped,
            "avg_confidence": round(confidence_sum / confidence_count, 3) if confidence_count else None,
        }
        return ToolResult(
            status="success",
            findings=findings,
            warnings=warnings,
            summary=summary,
        )


def run(rows: list[dict], options: dict[str, Any] | None = None) -> ToolResult:
    return SourceLockLogic().run(rows, options or {})

