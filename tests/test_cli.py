"""CLI tests for SourceLock."""

from __future__ import annotations

import csv
import json

from sourcelock.cli import main


def write_csv(tmp_path, content: str):
    p = tmp_path / "data.csv"
    p.write_text(content, encoding="utf-8")
    return p


def test_json_output(tmp_path, capsys):
    p = write_csv(
        tmp_path,
        "id,claim,source_posts\nr1,Users complain about X.,We launched a patch last week.\n",
    )
    assert main([str(p), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "sourcelock"
    assert payload["findings"]


def test_fail_on_medium(tmp_path):
    p = write_csv(
        tmp_path,
        "id,claim,source_posts\nr1,Users complain about X.,We launched a patch last week.\n",
    )
    assert main([str(p), "--fail-on", "medium"]) == 1
    assert main([str(p), "--fail-on", "high"]) == 1
    assert main([str(p), "--fail-on", "none"]) == 0


def test_csv_output(tmp_path, capsys):
    p = write_csv(
        tmp_path,
        "id,claim,source_posts\nr1,Users complain about X.,We launched a patch last week.\n",
    )
    assert main([str(p), "--format", "csv"]) == 0
    rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
    assert rows


def test_config_threshold(tmp_path, capsys):
    p = write_csv(
        tmp_path,
        "id,claim,source_posts\nr1,Latency is much worse now.,After the patch latency dropped sharply.\n",
    )
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"threshold": 0.95}), encoding="utf-8")
    assert main([str(p), "--config", str(cfg), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["supported"] == 0
