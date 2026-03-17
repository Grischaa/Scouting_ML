from __future__ import annotations

import json
from pathlib import Path

from scouting_ml.scripts import run_weekly_scout_ops as rwso
from scouting_ml.tests.summary_contract import assert_common_summary_contract


def test_run_weekly_scout_ops_writes_summary_contract(tmp_path: Path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("player_id\np1\n", encoding="utf-8")

    def _write(path: Path, payload: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return str(path)

    def _build_weekly_kpi_report(**kwargs):
        out_dir = Path(kwargs["out_dir"])
        return {
            "out_csv": _write(out_dir / "weekly_kpi.csv", "cohort\nEredivisie\n"),
            "out_json": _write(out_dir / "weekly_kpi.json", json.dumps({"items": [{"cohort": "Eredivisie"}]})),
            "row_count": 1,
        }

    def _build_onboarding_report(**kwargs):
        _write(Path(kwargs["out_json"]), json.dumps({"status_counts": {"watch": 1}}))
        _write(Path(kwargs["out_csv"]), "league_slug,status\neredivisie,watch\n")
        return {"status_counts": {"watch": 1}}

    def _run_workflow(**kwargs):
        out_dir = Path(kwargs["out_dir"])
        shortlist_csv = out_dir / "shortlist.csv"
        shortlist_json = out_dir / "shortlist.json"
        summary_json = out_dir / "workflow_summary.json"
        memo_dir = out_dir / "memos"
        memo_dir.mkdir(parents=True, exist_ok=True)
        _write(shortlist_csv, "player_id\np1\n")
        _write(shortlist_json, json.dumps({"items": [{"player_id": "p1"}]}))
        _write(summary_json, json.dumps({"shortlist_count": 1}))
        _write(tmp_path / "watchlist.jsonl", '{"player_id":"p1"}\n')
        return {
            "shortlist_count": 1,
            "memo_count": 0,
            "shortlist_csv": str(shortlist_csv),
            "shortlist_json": str(shortlist_json),
            "summary_json": str(summary_json),
            "memo_dir": str(memo_dir),
            "watchlist_added": 1,
        }

    monkeypatch.setattr(rwso, "build_weekly_kpi_report", _build_weekly_kpi_report)
    monkeypatch.setattr(rwso, "build_onboarding_report", _build_onboarding_report)
    monkeypatch.setattr(rwso, "run_workflow", _run_workflow)

    summary_path = tmp_path / "weekly_ops_summary.json"
    summary = rwso.run_weekly_scout_ops(
        predictions=str(predictions),
        split="test",
        reports_out_dir=str(tmp_path / "reports"),
        workflow_out_dir=str(tmp_path / "workflow"),
        workflow_watchlist_path=str(tmp_path / "watchlist.jsonl"),
        summary_json=str(summary_path),
    )

    assert_common_summary_contract(summary)
    assert summary_path.exists()
    disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert_common_summary_contract(disk_summary)
    assert summary["steps"]["workflow"]["shortlist_count"] == 1
