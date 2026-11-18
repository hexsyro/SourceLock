# SourceLock

AI-generated summaries can claim things the source posts never said. SourceLock
catches those unsupported claims.

## Why this exists

In AI-assisted analytics, summaries often state "users are complaining about X"
without a single post saying that. When credibility matters, the question is:
show me where that came from. SourceLock checks each claim against the
posts it's supposed to be drawn from, flagging anything unsupported and
returning the best-matching source snippet to make human verification quick.

## What it does

- **Claim-by-claim verification.** Splits claims into sentences and checks
  each against a pool of source text.
- **Atomic subclaims.** Compound sentences are also decomposed on `,`, `;`,
  `and`, `but`, `which`, `who`, and `while`. The whole sentence keeps its score;
  a high-severity `unsupported_subclaim` finding fires only when the whole claim
  looks supported but a fragment does not.
- **Lexical similarity.** TF-IDF cosine similarity with function-word removal
  and an exact-substring shortcut that returns confidence 1.0.
- **Number and date evidence.** Numbers with optional units (`100`, `3.5%`,
  `10 units`) and ISO-like dates (`2024-01-15`) present in the claim but absent
  from every source chunk produce a medium `unsupported_number` finding.
- **Negation awareness across the whole source pool.** If *any* source chunk
  has the opposite negation polarity (not just the best lexical match), the
  finding is escalated for human review and names its `negation_source`.
- **Three groundedness signals.** `entailment_probability`, `contradiction_rate`
  and `missing_evidence_frequency` are reported per finding and per run.
- **Honest confidence labels.** Every finding carries `signals` and
  `triage_only: true`. These are triage hints, not verified probabilities.
- **Explainable.** Findings include confidence, threshold, subclaim scores,
  and the best-matching source snippet so you can verify in seconds.

## What it is not

**Lexical verification cannot prove entailment.** SourceLock measures overlap
between strings; it does not understand meaning. It cannot prove that a source
entails a claim, and it cannot detect implicit, multi-hop, or semantically
subtle contradictions. Treat its output as a triage queue for a human reviewer
(`triage_only: true`), never as an automatic truth judgment.

## Install

Python 3.10+, no runtime dependencies.

```bash
git clone https://github.com/hexsyro/SourceLock.git
cd SourceLock
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests
python -m pyright src
```

## Usage

```bash
sourcelock examples/sample.csv
```

Column names are matched loosely: claims come from `claim`, `summary`,
`claim_text`, `statement`, `claim_sentence`, or `assertion`; sources from
`source_posts`, `sources`, `source_text`, `source_content`, `source`, `posts`,
or `evidence`.

## Options

| Flag | What it does |
| --- | --- |
| `--threshold T` | Minimum similarity (0.0–1.0) to consider a claim supported (default 0.30) |
| `--min-claim-chars N` | Minimum characters for a claim/subclaim to be checked (default 4) |
| `--config FILE` | JSON config; explicit flags win. Values must be correctly typed: `threshold` numeric, `min_claim_chars` integer |
| `--format {table,json,csv,md}` | Output format (default `table`) |
| `--output FILE` | Write to file instead of stdout |
| `--fail-on {none,low,medium,high}` | Exit 1 if any finding is at or above this severity |
| `--version` | Print version and exit |

Malformed config values never crash the library: `run()` falls back to defaults;
the CLI reports them on stderr and exits 2.

## Summary fields

`checked` counts **rows** that had both a claim and source text, while
`supported`, `unsupported`, `claims_checked`, and the confidence/signal averages
count **claims** (sentences) inside those rows. `skipped_missing_sources` counts
rows with a claim but no source. `skipped`/`skip_reason` say whether the whole
run had nothing to check. `result_schema_version` is `1.1`; `method` is
`lexical-v2` and `limitations` lists `implicit_contradiction`, `multi_hop`, and
`no_semantic_entailment`.

Warnings are emitted for claimless rows, rows whose claims are all shorter than
`min_claim_chars`, and rows missing source text.

## Output (json)

```json
{
  "tool": "sourcelock",
  "version": "0.2.0",
  "status": "success",
  "findings": [
    {
      "severity": "medium",
      "field": "claim",
      "message": "Claim 'users are complaining about X' is not supported by any source post (similarity 0.12 < threshold 0.30).",
      "record_id": "r1",
      "metadata": {
        "claim": "users are complaining about X",
        "confidence": 0.12,
        "threshold": 0.3,
        "method": "tfidf-cosine",
        "method_v2": "lexical-v2",
        "best_snippet": "we launched a patch last week",
        "subclaims": [
          {"text": "users are complaining about X", "confidence": 0.12, "supported": false}
        ],
        "signals": {
          "entailment_probability": 0.302,
          "contradiction_rate": 0.0,
          "missing_evidence_frequency": 1.0
        },
        "confidence_label": "lexical_proxy",
        "triage_only": true
      }
    }
  ],
  "summary": {
    "result_schema_version": "1.1",
    "checked": 1,
    "claims_checked": 1,
    "supported": 0,
    "unsupported": 1,
    "skipped": false,
    "skip_reason": null,
    "avg_confidence": 0.12,
    "method": "lexical-v2",
    "limitations": ["implicit_contradiction", "multi_hop", "no_semantic_entailment"]
  }
}
```

## Using it in CI

```bash
sourcelock report.csv --fail-on medium || exit 1
```

Exit codes: `0` ran fine, `1` gate tripped, `2` input broken.

## Library use

```python
from sourcelock import RESULT_SCHEMA_VERSION, ToolResult, run

result = run(rows, {"threshold": 0.4, "min_claim_chars": 4})
assert result.summary["result_schema_version"] == RESULT_SCHEMA_VERSION
```

## Limitations, honestly

- **No semantic entailment.** `entailment_probability` is a monotone transform
  of lexical cosine similarity, not a calibrated model probability. Shared
  words can look like support without any entailment.
- **Implicit and multi-hop contradiction.** A source that implies the opposite
  through reasoning or several documents will not be caught.
- **Negation is lexical.** It catches obvious not/no/never flips across the
  source pool, but complex double-negatives are still tricky — hence the
  human-review finding.
- **Short sources or very generic claims** give low similarity. That's a
  feature, not a bug.

## [0.2.0] - 2026-11-19

### Added

- Atomic subclaim decomposition and `unsupported_subclaim` findings.
- Per-finding/per-run `entailment_probability`, `contradiction_rate`, and
  `missing_evidence_frequency`; `triage_only` honest confidence labels.
- Number/date evidence checks, `--min-claim-chars`, `claims_checked`,
  `result_schema_version`, and `skipped`/`skip_reason` in summaries.
- English/German/French function-word stopwords for TF-IDF tokenization.

### Fixed

- Negation gate now scans every source chunk, not just the best match.
- Sentence splitting no longer breaks decimals or abbreviations.
- Malformed option types no longer raise from `run()`; the CLI exits 2 with a
  clear message. Redundant in-package LICENSE copy removed.

### Inspiration

- RAGAS claim decomposition — https://www.youtube.com/watch?v=xsDNArrmyuo
- FActScore atomic facts with entailment/contradiction/missing evidence — https://aman.ai/primers/ai/factuality-in-LLMs
- 1–5 groundedness rubric, NAACL 2025 — https://aclanthology.org/2025.findings-naacl.433
- NLI limits on implicit/multi-hop contradiction — https://www.sciencedirect.com/science/article/pii/S2949719126000361
- AWS on low recall of similarity-only hallucination detectors — https://aws.amazon.com/blogs/machine-learning/detect-hallucinations-for-rag-based-systems

## License

MIT. See [LICENSE](LICENSE).
