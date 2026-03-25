from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scouting_ml.scripts.weekly_scout_decision_feedback_report import (
    build_weekly_scout_decision_feedback_report,
)


def _write_decision_log(path: Path) -> None:
    records = [
        {
            "decision_id": "d1",
            "created_at_utc": "2026-03-24T09:00:00+00:00",
            "player_id": "p1",
            "split": "test",
            "season": "2024/25",
            "action": "shortlist",
            "reason_tags": ["system_fit", "price_gap"],
            "source_surface": "workbench",
            "player_snapshot": {
                "league": "Eredivisie",
                "position": "FW",
                "league_trust_tier": "trusted",
                "league_adjustment_bucket": "standard",
            },
            "ranking_context": {
                "rank": 4,
                "system_template": "high_press_433",
                "discovery_reliability_weight": 0.96,
            },
        },
        {
            "decision_id": "d2",
            "created_at_utc": "2026-03-24T10:00:00+00:00",
            "player_id": "p2",
            "split": "test",
            "season": "2024/25",
            "action": "watch_live",
            "reason_tags": ["availability"],
            "source_surface": "watchlist",
            "player_snapshot": {
                "league": "Belgian Pro League",
                "position": "DF",
                "league_trust_tier": "watch",
                "league_adjustment_bucket": "weak",
            },
            "ranking_context": {
                "rank": 18,
                "discovery_reliability_weight": 0.82,
            },
        },
        {
            "decision_id": "d3",
            "created_at_utc": "2026-03-24T11:00:00+00:00",
            "player_id": "p3",
            "split": "test",
            "season": "2024/25",
            "action": "pass",
            "reason_tags": ["league_risk", "data_too_thin"],
            "source_surface": "system_fit",
            "player_snapshot": {
                "league": "Estonian Meistriliiga",
                "position": "FW",
                "league_trust_tier": "unknown",
                "league_adjustment_bucket": "severe_failed",
            },
            "ranking_context": {
                "rank": 7,
                "system_template": "transition_4231",
                "discovery_reliability_weight": 0.53,
            },
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_weekly_scout_decision_feedback_report_writes_outputs(tmp_path: Path) -> None:
    decisions_path = tmp_path / "scout_decisions.jsonl"
    out_dir = tmp_path / "reports"
    _write_decision_log(decisions_path)

    payload = build_weekly_scout_decision_feedback_report(
        decisions_path=str(decisions_path),
        out_dir=str(out_dir),
        lookback_days=7,
        now_utc=datetime(2026, 3, 24, 12, 0, tzinfo=timezone.utc),
    )

    summary = payload["summary"]
    assert summary["total_decisions"] == 3
    assert abs(float(summary["positive_decision_rate"]) - (2 / 3)) < 1e-9
    assert summary["pass_reason_counts"]["league_risk"] == 1
    assert summary["positive_rate_by_bucket"]["standard"] == 1.0
    assert Path(payload["paths"]["json"]).exists()
    assert Path(payload["paths"]["markdown"]).exists()
    assert Path(payload["paths"]["breakdowns_csv"]).exists()
    assert Path(payload["paths"]["action_counts_csv"]).exists()
    assert Path(payload["paths"]["pass_reasons_csv"]).exists()
