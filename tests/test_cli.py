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


def test_min_claim_chars_flag(tmp_path, capsys):
    p = write_csv(tmp_path, "id,claim,source_posts\nr1,AI,AI is widely used.\n")
    assert main([str(p), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["claims_checked"] == 0

    assert main([str(p), "--min-claim-chars", "2", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["claims_checked"] == 1
    assert payload["summary"]["result_schema_version"] == "1.1"


def test_invalid_config_value_types_exit_2(tmp_path, capsys):
    p = write_csv(tmp_path, "id,claim,source_posts\nr1,Some claim.,Some source.\n")
    for key, value in [("threshold", "high"), ("min_claim_chars", 2.5)]:
        cfg = tmp_path / f"{key}.json"
        cfg.write_text(json.dumps({key: value}), encoding="utf-8")
        assert main([str(p), "--config", str(cfg)]) == 2
        assert key in capsys.readouterr().err


def test_invalid_flag_value_exits_2(tmp_path, capsys):
    p = write_csv(tmp_path, "id,claim,source_posts\nr1,Some claim.,Some source.\n")
    assert main([str(p), "--min-claim-chars", "abc"]) == 2
    assert "min-claim-chars" in capsys.readouterr().err
