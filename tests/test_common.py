"""Tests for shared helpers: stopwords, schema constant, option coercion."""

from __future__ import annotations

from sourcelock import RESULT_SCHEMA_VERSION, ToolResult, __version__
from sourcelock.common import STOPWORDS, opt_float, opt_int, tokenize


def test_version_and_exports():
    assert __version__ == "0.2.0"
    assert RESULT_SCHEMA_VERSION == "1.1"
    assert ToolResult().status == "success"


def test_stopwords_populated_and_applied():
    assert {"the", "and", "der", "und", "le", "et"} <= STOPWORDS
    tokens = tokenize("The users and the posts are in der Stadt, et les autres.")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "der" not in tokens
    assert "et" not in tokens
    assert {"users", "posts", "stadt", "autres"} <= set(tokens)


def test_tokenize_keeps_content_words_and_numbers():
    tokens = set(tokenize("Latency 3.5 percent improved"))
    assert {"latency", "3", "5", "percent", "improved"} <= tokens
    assert "3.5" not in tokens  # the tokenizer intentionally splits on punctuation


def test_opt_helpers_fall_back_on_malformed_values():
    assert opt_float({"threshold": "oops"}, "threshold", 0.3) == 0.3
    assert opt_float({"threshold": float("nan")}, "threshold", 0.3) == 0.3
    assert opt_float({"threshold": True}, "threshold", 0.3) == 0.3
    assert opt_float({"threshold": 0.75}, "threshold", 0.3) == 0.75
    assert opt_int({"min_claim_chars": "many"}, "min_claim_chars", 4) == 4
    assert opt_int({"min_claim_chars": 2.9}, "min_claim_chars", 4) == 2
    assert opt_int({"min_claim_chars": 12}, "min_claim_chars", 4) == 12
