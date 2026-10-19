"""Tests for SourceLock claim verification."""

from __future__ import annotations

from sourcelock.logic import SourceLockLogic

logic = SourceLockLogic()


def test_direct_match_supported():
    rows = [
        {
            "id": "r1",
            "claim": "Users are complaining about latency.",
            "source_posts": "We see users complaining about latency in the last 24 hours.",
        }
    ]
    result = logic.run(rows, {"threshold": 0.3})
    assert result.findings == []
    assert result.summary["supported"] == 1
    assert result.summary["checked"] == 1


def test_unsupported_high_severity():
    rows = [
        {
            "id": "r1",
            "claim": "Users are complaining about X.",
            "source_posts": "We launched a patch last week.",
        }
    ]
    result = logic.run(rows, {"threshold": 0.3})
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f["severity"] == "high"
    assert f["metadata"]["confidence"] < 0.3


def test_negation_mismatch_escalates():
    rows = [
        {
            "id": "r1",
            "claim": "We released the fix.",
            "source_posts": "We did not release the fix.",
        }
    ]
    result = logic.run(rows, {"threshold": 0.9})
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f["severity"] == "medium"
    assert f["metadata"]["negation_mismatch"] is True


def test_paraphrase_supported_above_threshold():
    rows = [
        {
            "id": "r1",
            "claim": "Latency is down significantly after the patch.",
            "source_posts": "After deploying the patch, latency dropped sharply across all regions.",
        }
    ]
    result = logic.run(rows, {"threshold": 0.25})
    assert result.summary["supported"] == 1
    assert result.findings == []


def test_skips_when_missing_sources():
    rows = [{"id": "r1", "claim": "Something happened."}]
    result = logic.run(rows, {})
    assert result.summary["checked"] == 0
    assert result.warnings


def test_multiple_claims_split():
    rows = [
        {
            "id": "r1",
            "claim": "Users complain about latency. We fixed it.",
            "source_posts": "We see users complaining about latency. After patch, latency improved.",
        }
    ]
    result = logic.run(rows, {"threshold": 0.2})
    assert result.summary["supported"] >= 1


def test_empty_rows_quiet():
    result = logic.run([], {})
    assert result.summary["checked"] == 0
