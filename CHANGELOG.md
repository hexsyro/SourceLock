# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-11-19

### Added

- Atomic claim decomposition: compound sentences are split on commas, semicolons, `and`, `but`, `which`, `who` and `while`; per-subclaim confidence is attached to every finding, and a high-severity `unsupported_subclaim` finding fires when the whole claim scores supported but a fragment does not.
- Three groundedness signals (`entailment_probability`, `contradiction_rate`, `missing_evidence_frequency`) in finding metadata and run summaries, plus `method = "lexical-v2"`, `limitations`, and per-finding `method_v2` / `signals` / `confidence_label` / `triage_only: true` labels.
- Missing number/date detection: numbers with optional units and ISO-like dates absent from all source chunks produce a medium `unsupported_number` finding with `claim_numbers`, `source_numbers`, and example text.
- `--min-claim-chars N` CLI flag; `summary["claims_checked"]` to distinguish rows from claims; warnings for claimless rows and rows whose claims are all shorter than the minimum.
- `RESULT_SCHEMA_VERSION = "1.1"` and `ToolResult` exported from `sourcelock`, with `result_schema_version`, `skipped`, and `skip_reason` in every run summary (existing `status` values unchanged).
- Conservative English/German/French function-word stopwords applied in tokenization for TF-IDF.

### Fixed

- Negation gate now checks every source chunk and reports `negation_source`; severity escalates to high when confidence is very low (below half the threshold).
- Sentence splitting protects decimals (`3.5`) and abbreviations (`U.S.`, `e.g.`) from being split on `.`.
- Option types are validated: malformed config values never raise from `run()`/`main()`; the CLI exits 2 with a clear stderr message. Bare `float()`/`int()` conversions replaced with `opt_float`/`opt_int` helpers.
- Removed the redundant `src/sourcelock/LICENSE` copy (root `LICENSE` remains).

### Inspiration

- RAGAS claim decomposition — https://www.youtube.com/watch?v=xsDNArrmyuo
- FActScore atomic facts with entailment/contradiction/missing evidence — https://aman.ai/primers/ai/factuality-in-LLMs
- 1–5 groundedness rubric, NAACL 2025 — https://aclanthology.org/2025.findings-naacl.433
- NLI limits on implicit/multi-hop contradiction — https://www.sciencedirect.com/science/article/pii/S2949719126000361
- AWS on low recall of similarity-only hallucination detectors — https://aws.amazon.com/blogs/machine-learning/detect-hallucinations-for-rag-based-systems

## [0.1.0] - 2026-10-19

### Added

- Initial release.
- Claim-by-claim verification against source text: TF-IDF cosine similarity with exact-substring shortcut.
- Negation-aware gating to prevent false positives (differences in not/no/never escalate to human review).
- Per-row confidence, source snippet returned in metadata.
- Split claims and sources into sentences for better matching.
- CLI with table/json/csv/md, --fail-on exit codes, JSON config.
- Importable library API: sourcelock.run(rows, options).
