"""Tests for v0.2.0 claim decomposition, evidence signals and schema."""

from __future__ import annotations

import pytest

from sourcelock import RESULT_SCHEMA_VERSION, run
from sourcelock.logic import (
    LIMITATIONS,
    METHOD_V2,
    _extract_numbers,
    _split_claims,
    _split_subclaims,
)

SIGNAL_KEYS = {"entailment_probability", "contradiction_rate", "missing_evidence_frequency"}


def test_summary_schema_and_signals():
    result = run(
        [{"id": "r1", "claim": "Users complain about latency.", "source_posts": "Users complain about latency."}],
        {"threshold": 0.3},
    )
    summary = result.summary
    assert summary["result_schema_version"] == RESULT_SCHEMA_VERSION == "1.1"
    assert summary["method"] == METHOD_V2 == "lexical-v2"
    assert summary["limitations"] == LIMITATIONS
    assert summary.keys() >= SIGNAL_KEYS
    assert summary["skipped"] is False
    assert summary["skip_reason"] is None
    assert result.status == "success"
    for signal in SIGNAL_KEYS:
        assert 0.0 <= summary[signal] <= 1.0


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ([], "no_rows"),
        ([{"id": "r1", "source_posts": "Some source text."}], "no_claim_text"),
        ([{"id": "r1", "claim": "Something happened."}], "missing_source_text"),
        ([{"id": "r1", "claim": "OK", "source_posts": "OK."}], "claims_below_min_length"),
    ],
)
def test_skipped_summary_reasons(rows, reason):
    result = run(rows, {})
    assert result.summary["skipped"] is True
    assert result.summary["skip_reason"] == reason
    assert result.warnings


def test_checked_counts_rows_and_claims_checked_counts_claims():
    result = run(
        [
            {
                "id": "r1",
                "claim": "Users complain about latency. We shipped a patch.",
                "source_posts": "Users complain about latency. We shipped a patch.",
            },
            {
                "id": "r2",
                "claim": "Errors are down. Uptime is up.",
                "source_posts": "Errors are down. Uptime is up.",
            },
        ],
        {"threshold": 0.2},
    )
    assert result.summary["checked"] == 2
    assert result.summary["claims_checked"] == 4
    assert result.summary["supported"] == 4


def test_warnings_for_short_and_claimless_rows():
    result = run(
        [
            {"id": "empty", "source_posts": "Source text."},
            {"id": "short", "claim": "AI", "source_posts": "AI is widely used."},
        ],
        {"min_claim_chars": 4},
    )
    assert result.summary["checked"] == 1  # rows with both a claim and a source
    assert result.summary["claims_checked"] == 0
    assert any("no claim text" in warning for warning in result.warnings)
    assert any("shorter than min_claim_chars" in warning for warning in result.warnings)


def test_min_claim_chars_option_filters_claims():
    rows = [{"id": "r1", "claim": "AI", "source_posts": "AI is widely used."}]
    assert run(rows, {"min_claim_chars": 2}).summary["claims_checked"] == 1
    assert run(rows, {"min_claim_chars": 8}).summary["claims_checked"] == 0


def test_malformed_options_never_raise():
    rows = [{"id": "r1", "claim": "Users complain about latency.", "source_posts": "Users complain about latency."}]
    result = run(rows, {"threshold": "high", "min_claim_chars": "many"})
    assert result.summary["supported"] == 1
    assert 0.0 <= result.summary["entailment_probability"] <= 1.0


def test_subclaims_attached_to_every_finding():
    result = run(
        [
            {
                "id": "r1",
                "claim": "Users complain about latency, and revenue grew 45%.",
                "source_posts": "We launched a patch last week.",
            }
        ],
        {"threshold": 0.3},
    )
    assert result.findings
    for finding in result.findings:
        metadata = finding["metadata"]
        assert metadata["subclaims"], metadata
        assert all({"text", "confidence", "supported"} <= entry.keys() for entry in metadata["subclaims"])
        assert metadata["signals"].keys() >= SIGNAL_KEYS
        assert metadata["triage_only"] is True
        assert metadata["confidence_label"] == "lexical_proxy"
        assert metadata["method_v2"] == "lexical-v2"


def test_unsupported_subclaim_only_when_whole_claim_supported():
    # Whole claim shares enough terms with the source, but "revenue grew 45%"
    # does not appear anywhere in it.
    rows = [
        {
            "id": "r1",
            "claim": "Users complain about latency, and revenue grew 45%.",
            "source_posts": "Users complain about latency in the last 24 hours.",
        }
    ]
    result = run(rows, {"threshold": 0.25})
    whole = [f for f in result.findings if f["metadata"].get("unsupported_subclaim")]
    assert len(whole) == 1
    assert whole[0]["severity"] == "high"
    assert any("revenue" in text for text in whole[0]["metadata"]["unsupported_subclaims"])
    assert result.summary["supported"] == 1


def test_no_unsupported_subclaim_when_whole_claim_is_unsupported():
    rows = [
        {
            "id": "r1",
            "claim": "Quantum flux capacitors power the new engine.",
            "source_posts": "We updated the documentation.",
        }
    ]
    result = run(rows, {"threshold": 0.5})
    assert not [f for f in result.findings if f["metadata"].get("unsupported_subclaim")]
    assert [f for f in result.findings if f["severity"] == "high"]


def test_unsupported_number_finding_and_metadata():
    rows = [
        {
            "id": "r1",
            "claim": "Revenue grew 3.5% in 2024 to 1,000 units.",
            "source_posts": "Revenue grew last year after the patch.",
        }
    ]
    result = run(rows, {"threshold": 0.2})
    findings = [f for f in result.findings if f["metadata"].get("unsupported_number")]
    assert len(findings) == 1
    finding = findings[0]
    assert finding["severity"] == "medium"
    assert {"3.5%", "2024", "1000 units"} <= set(finding["metadata"]["claim_numbers"])
    assert finding["metadata"]["source_numbers"] == []
    assert set(finding["metadata"]["missing_numbers"]) == {"3.5%", "2024", "1000 units"}
    assert finding["metadata"]["example"]
    assert finding["metadata"]["signals"].keys() >= SIGNAL_KEYS


def test_supported_numbers_do_not_fire():
    rows = [
        {
            "id": "r1",
            "claim": "Revenue grew 3.5% in 2024.",
            "source_posts": "In 2024 revenue grew 3.5% after the patch.",
        }
    ]
    result = run(rows, {"threshold": 0.2})
    assert not [f for f in result.findings if f["metadata"].get("unsupported_number")]


def test_iso_date_extraction():
    assert set(_extract_numbers("Launched on 2024-01-15 with 10 units.")) == {
        "2024",
        "2024-01-15",
        "10 units",
    }


def test_negation_gate_checks_all_source_chunks():
    rows = [
        {
            "id": "r1",
            "claim": "We released the fix.",
            "source_posts": "We released the fix. We did not release the fix.",
        }
    ]
    result = run(rows, {"threshold": 0.9})
    findings = [f for f in result.findings if f["metadata"].get("negation_mismatch")]
    assert len(findings) == 1
    assert "not" in findings[0]["metadata"]["negation_source"]
    assert findings[0]["metadata"]["negation_source"] != findings[0]["metadata"]["best_snippet"]
    assert findings[0]["severity"] == "medium"  # confidence is high


def test_negation_gate_high_severity_on_very_low_confidence():
    rows = [
        {
            "id": "r1",
            "claim": "Alpha beta gamma delta was rolled out.",
            "source_posts": "The rollout did not happen.",
        }
    ]
    result = run(rows, {"threshold": 0.3})
    findings = [f for f in result.findings if f["metadata"].get("negation_mismatch")]
    assert len(findings) == 1
    assert findings[0]["severity"] == "high"
    assert findings[0]["metadata"]["confidence"] < 0.3 * 0.5


def test_sentence_split_protects_decimals_and_abbreviations():
    text = "Version 3.5 shipped in the U.S., e.g. in Boston. Done."
    claims = _split_claims(text)
    assert len(claims) == 2
    assert "3.5" in claims[0]
    assert "U.S." in claims[0]
    assert "e.g." in claims[0]
    assert claims[1] == "Done"


def test_thousands_separator_not_split_into_subclaims():
    subclaims = _split_subclaims("Revenue hit 1,000 units and churn fell.")
    assert len(subclaims) == 2
    assert "1,000 units" in subclaims[0]
