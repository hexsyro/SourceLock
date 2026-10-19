# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-10-19

### Added

- Initial release.
- Claim-by-claim verification against source text: TF-IDF cosine similarity with exact-substring shortcut.
- Negation-aware gating to prevent false positives (differences in not/no/never escalate to human review).
- Per-row confidence, source snippet returned in metadata.
- Split claims and sources into sentences for better matching.
- CLI with table/json/csv/md, --fail-on exit codes, JSON config.
- Importable library API: sourcelock.run(rows, options).
