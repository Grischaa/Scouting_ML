from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts.build_external_feature_coverage_audit import build_external_feature_coverage_audit


def test_build_external_feature_coverage_audit_writes_family_summaries(tmp_path: Path) -> None:
    dataset = tmp_path / "clean.parquet"
    pd.DataFrame(
        [
            {
                "season": "2023/24",
                "league": "Primeira Liga",
                "injury_days": 10,
                "injury_avg_days_per_case": 5.0,
                "availability_risk_score": 0.2,
                "avail_minutes_share": 0.8,
                "availability_selection_score": 0.7,
                "fixture_points_per_match": 1.8,
                "fixture_team_form_score": 0.6,
                "sb_progressive_passes_per90": 4.2,
                "odds_strength_score": 0.3,
            },
            {
                "season": "2024/25",
                "league": "Eredivisie",
                "injury_days": None,
                "injury_avg_days_per_case": None,
                "availability_risk_score": None,
                "avail_minutes_share": None,
                "availability_selection_score": None,
                "fixture_points_per_match": None,
                "fixture_team_form_score": None,
                "sb_progressive_passes_per90": None,
                "odds_strength_score": None,
            },
        ]
    ).to_parquet(dataset, index=False)

    out_json = tmp_path / "audit.json"
    out_csv = tmp_path / "audit.csv"
    out_md = tmp_path / "audit.md"
    payload = build_external_feature_coverage_audit(
        dataset_path=str(dataset),
        out_json=str(out_json),
        out_csv=str(out_csv),
        out_md=str(out_md),
    )

    assert out_json.exists()
    assert out_csv.exists()
    assert out_md.exists()
    assert payload["family_overall"]["injury"]["feature_count"] >= 3
    assert payload["family_overall"]["availability"]["row_coverage_share"] == 0.5
    assert payload["family_by_season"]["fixture"][0]["split"] == "2023/24"
    written = json.loads(out_json.read_text(encoding="utf-8"))
    assert written["family_overall"]["statsbomb"]["feature_count"] == 1
