"""Command-line interface for SourceLock.

Exit codes:
  0 — ran successfully (findings may or may not be present)
  1 — findings at or above the --fail-on severity (CI gate tripped)
  2 — usage or input error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sourcelock import __version__
from sourcelock.common import SEVERITY_RANK, load_rows
from sourcelock.logic import SourceLockLogic
from sourcelock.output import render, worst_severity

TOOL = "sourcelock"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="Verify AI-generated claims are supported by source text.",
    )
    parser.add_argument("data", help="dataset file to scan (.csv, .json, or .jsonl)")
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "csv", "md", "markdown"],
        default="table",
        help="output format (default: table)",
    )
    parser.add_argument("--output", "-o", metavar="FILE", help="write the report to FILE instead of stdout")
    parser.add_argument(
        "--fail-on",
        choices=["none", "low", "medium", "high"],
        default="none",
        help="exit 1 if any finding is at or above this severity (default: none)",
    )
    parser.add_argument("--config", metavar="FILE", help="JSON file of options")
    parser.add_argument(
        "--threshold",
        type=float,
        metavar="T",
        help="minimum similarity to consider a claim supported (default: 0.30)",
    )
    parser.add_argument(
        "--min-claim-chars",
        type=int,
        metavar="N",
        help="minimum characters for a claim/subclaim to be checked (default: 4)",
    )
    parser.add_argument("--version", action="version", version=f"{TOOL} {__version__}")
    return parser


def _validate_option_types(options: dict[str, Any], source: str) -> None:
    """Reject wrong-typed option values early so the CLI can exit 2 clearly."""
    if "threshold" in options:
        value = options["threshold"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{source}: 'threshold' must be a number, got {type(value).__name__}: {value!r}")
    if "min_claim_chars" in options:
        value = options["min_claim_chars"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{source}: 'min_claim_chars' must be an integer, got {type(value).__name__}: {value!r}")


def _merge_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if args.config:
        source = str(args.config)
        parsed = json.loads(Path(source).read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"{source}: config must be a JSON object")
        _validate_option_types(parsed, source)
        options.update(parsed)
    if args.threshold is not None:
        options["threshold"] = args.threshold
    if args.min_claim_chars is not None:
        options["min_claim_chars"] = args.min_claim_chars
    _validate_option_types(options, "options")
    return options


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse already printed a clear stderr message
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        options = _merge_options(args)
        rows = load_rows(args.data)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{TOOL}: {exc}", file=sys.stderr)
        return 2
    result = SourceLockLogic().run(rows, options)
    text = render(result, args.format, tool=TOOL, version=__version__)
    if args.output:
        try:
            Path(args.output).write_text(text, encoding="utf-8")
        except OSError as exc:
            print(f"{TOOL}: {exc}", file=sys.stderr)
            return 2
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    if args.fail_on != "none":
        worst = worst_severity(result)
        if worst and SEVERITY_RANK.get(worst, 0) >= SEVERITY_RANK[args.fail_on]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
