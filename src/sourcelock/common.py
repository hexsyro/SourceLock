"""Shared helpers for SourceLock: row coercion, dataset loading, result shape.

The rules in ``logic.py`` receive the rows of a dataset (a list of dicts) plus
an options dict and return a :class:`ToolResult`. The helpers here normalize
the messy real-world columns pipelines actually export — mixed casing,
trailing spaces, ``-`` vs ``_`` — so the rules can be defensive about field
presence without repeating lookup code.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}
# Bumped when the shape of ToolResult.summary changes in a way consumers must notice.
RESULT_SCHEMA_VERSION = "1.1"


@dataclass
class ToolResult:
    status: str = "success"
    findings: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Column lookup
# ---------------------------------------------------------------------------

_SEP_RE = re.compile(r"[^0-9a-z]+")


def norm_key(key: Any) -> str:
    """Normalize a column name for lookup: lowercase, strip, dashes -> underscore."""
    return _SEP_RE.sub("_", str(key).strip().lower())


def index_row(row: dict) -> dict[str, Any]:
    """Return a row indexed by normalized column name (first occurrence wins)."""
    indexed: dict[str, Any] = {}
    for key, value in row.items():
        normalized = norm_key(key)
        if normalized and normalized not in indexed:
            indexed[normalized] = value
    return indexed


def get(row: dict, keys: list[str], default: Any = None) -> Any:
    """Case-insensitive lookup for the first present non-null column among ``keys``."""
    indexed = index_row(row)
    for key in keys:
        value = indexed.get(norm_key(key))
        if value is not None:
            return value
    return default


def as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value).strip()


def to_float(value: Any, default: float | None = None) -> float | None:
    """Parse a number from a messy cell (percent signs, commas, ``k``/``M`` suffixes).

    Rejects date-like or ambiguous strings instead of silently grabbing the
    first number group.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = as_str(value).replace(",", "").strip()
    if not text:
        return default
    multiplier = 1.0
    if text.endswith(("k", "K")):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith(("m", "M")):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith(("b", "B")):
        multiplier = 1_000_000_000.0
        text = text[:-1]
    groups = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(groups) != 1:
        return default
    try:
        return float(groups[0]) * multiplier
    except ValueError:
        return default


def find_id(row: dict) -> str | None:
    """Best-effort record id for a row."""
    value = get(row, ["id", "record_id", "post_id", "uuid", "row_id", "key"])
    if value is None:
        return None
    text = as_str(value)
    return text or None


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y/%m/%d",
    "%Y/%m/%d %H:%M:%S",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%b %d, %Y",
    "%d %b %Y",
    "%Y-%m-%d %H:%M",
)


def parse_date(value: Any) -> datetime | None:
    """Parse a date-ish cell into a datetime (tz-aware values are kept aware)."""
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    text = as_str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def as_naive(value: datetime | None) -> datetime | None:
    """Drop tzinfo so aware and naive timestamps compare safely."""
    if value is None:
        return None
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def opt_float(options: dict, key: str, default: float) -> float:
    """Coerce an option to a finite float, falling back on malformed values."""
    value = options.get(key, default)
    if isinstance(value, bool):
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    return coerced if math.isfinite(coerced) else default


def opt_int(options: dict, key: str, default: int) -> int:
    """Coerce an option to an int, falling back on malformed values."""
    value = options.get(key, default)
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_rows(path: str | Path) -> list[dict]:
    """Load rows from a CSV, JSON, or JSONL file.

    CSV needs a header row. JSON must be a list of objects (or an object with
    a ``rows`` or ``data`` list). Each JSONL line must be one object.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    suffix = p.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: invalid JSON line: {exc.msg}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                raise ValueError(f"{p}:{lineno}: each JSONL line must be a JSON object")
        return rows

    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}: invalid JSON: {exc.msg}") from exc
        if isinstance(data, dict):
            data = data.get("rows", data.get("data"))
        if not isinstance(data, list):
            raise ValueError(f"{p}: expected a JSON list of objects (or an object with 'rows'/'data')")
        return [row for row in data if isinstance(row, dict)]

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError(f"{p}: CSV file has no header row")
    return [dict(row) for row in reader]


# Conservative function-word stoplist for English, German and French. Only
# high-frequency grammatical words live here: content words stay. Negation is
# decided separately on the raw text, so its markers need no TF-IDF weight.
STOPWORDS = {
    # English
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down",
    "during", "each", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "me", "might", "more", "most", "must", "my", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours",
    # German (words shared with English above appear only once)
    "aber", "als", "andere", "anderen", "auch", "auf", "aus",
    "bei", "bin", "bis", "bist", "da", "dann", "das", "dass", "dazu", "dein", "deine",
    "dem", "den", "denn", "der", "des", "dich", "die", "dies", "diese", "diesem",
    "diesen", "dieser", "deshalb", "doch", "dort", "du", "durch", "ein", "eine",
    "einem", "einen", "einer", "eines", "er", "es", "etwas", "euer", "eure", "für",
    "gegen", "gewesen", "hab", "habe", "haben", "hat", "hatte", "hatten", "hier",
    "hin", "hinter", "ich", "ihm", "ihn", "ihnen", "ihr", "ihre", "im", "indem",
    "ins", "ist", "jede", "jedem", "jeden", "jeder", "jetzt", "kann", "kein", "keine",
    "keinen", "können", "könnte", "machen", "man", "manche", "mein", "meine", "mich",
    "mir", "mit", "muss", "musste", "nach", "nicht", "nichts", "noch", "nun", "nur",
    "ob", "oder", "ohne", "sehr", "sein", "seine", "selbst", "sich", "sie", "sind",
    "soll", "sollte", "sondern", "sonst", "über", "um", "und", "uns", "unser",
    "unter", "viel", "vom", "von", "vor", "war", "waren", "warum", "weil",
    "weiter", "welche", "wenn", "werde", "werden", "wie", "wieder", "wir", "wird",
    "wo", "wurde", "wurden", "zu", "zum", "zur", "zwar", "zwischen",
    # French (words shared with the lists above appear only once)
    "au", "aux", "avait", "avaient", "avec", "aura", "ont", "ou", "où", "par",
    "pas", "plus", "pour", "que", "qui", "quoi", "sans", "se", "sera", "ses", "son",
    "sur", "très", "tu", "un", "une", "vos", "votre", "vous", "dans", "de",
    "elle", "en", "est", "et", "ils", "je", "la", "le", "les", "leur", "leurs",
    "lui", "ma", "mais", "mes", "mon", "ne", "nos", "notre", "nous",
    "oui", "sa", "si", "te", "tes", "toi", "ton", "tous", "tout", "toute", "toutes",
    # Single-letter French elisions kept by the tokenizer
    "c", "d", "j", "l", "m", "n", "qu", "s", "t",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with function words removed (used by TF-IDF)."""
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOPWORDS]


def tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

