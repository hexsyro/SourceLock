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
- **Lexical similarity.** TF-IDF cosine similarity with an exact-substring
  shortcut that returns confidence 1.0.
- **Negation awareness.** If the claim and its best match differ in negation
  (e.g. "released" vs "did not release"), it never automatically blesses it —
  the finding is escalated for human review.
- **Explainable.** For unsupported claims, the metadata includes confidence,
  the threshold, and the best-matching source snippet so you can verify in
  seconds.

## What it is not

It does not "judge" truth. It tests traceability: can a claim be anchored to
the source text you provided? If not, don't ship the claim.

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
| `--config FILE` | JSON config; explicit flags win |
| `--format {table,json,csv,md}` | Output format (default `table`) |
| `--output FILE` | Write to file instead of stdout |
| `--fail-on {none,low,medium,high}` | Exit 1 if any finding is at or above this severity |
| `--version` | Print version and exit |

## Output (json)

```json
{
  "tool": "sourcelock",
  "version": "0.1.0",
  "status": "success",
  "findings": [
    {
      "severity": "medium",
      "field": "claim",
      "message": "Claim '...users are complaining about X...' is not supported by any source post (similarity 0.12 < threshold 0.30).",
      "record_id": "r1",
      "metadata": {
        "claim": "users are complaining about X",
        "confidence": 0.12,
        "threshold": 0.3,
        "method": "tfidf-cosine",
        "best_snippet": "we launched a patch last week"
      }
    }
  ],
  "summary": {"checked": 1, "supported": 0, "unsupported": 1, "avg_confidence": 0.12}
}
```

## Using it in CI

```bash
sourcelock report.csv --fail-on medium || exit 1
```

Exit codes: `0` ran fine, `1` gate tripped, `2` input broken.

## Library use

```python
from sourcelock import run

result = run(rows, {"threshold": 0.4})
```

## Limitations, honestly

- Traceability, not truth. Paraphrases can slip through only if the text
  shares enough terms; semantically subtle claims may need higher thresholds.
- Negation is lexical: it catches obvious not/no/never flips, but complex
  double-negatives are still tricky — hence the human-review finding.
- Short sources or very generic claims give low similarity. That's a feature,
  not a bug.

## License

MIT. See [LICENSE](LICENSE).
