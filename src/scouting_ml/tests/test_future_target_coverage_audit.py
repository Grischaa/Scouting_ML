from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts.build_future_target_coverage_audit import build_future_target_coverage_audit


def test_build_future_target_coverage_audit_writes_outputs(tmp_path: Path) -> None:
    dataset_path = tmp_path / "clean.parquet"
    future_targets_output = tmp_path / "future_targets.parquet"
    out_json = tmp_path / "audit.json"
    out_csv = tmp_path / "audit.csv"
    source_dir = tmp_path / "processed" / "by_season" / "2025-26"
    source_dir.mkdir(parents=True)
    (source_dir / "austrian_bundesliga_2025-26_with_sofa.csv").write_text("player_id\np1\n", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": "2023/24",
                "league": "Eredivisie",
                "market_value_eur": 2_000_000,
                "minutes": 1000,
            },
            {
                "player_id": "p1",
                "season": "2024/25",
                "league": "Eredivisie",
                "market_value_eur": 3_000_000,
                "minutes": 1100,
            },
            {
                "player_id": "p1",
                "season": "2025/26",
                "league": "Austrian Bundesliga",
                "market_value_eur": 3_500_000,
                "minutes": 1200,
            },
        ]
    ).to_parquet(dataset_path, index=False)

    payload = build_future_target_coverage_audit(
        dataset_path=str(dataset_path),
        future_targets_output=str(future_targets_output),
        out_json=str(out_json),
        out_csv=str(out_csv),
        min_next_minutes=450.0,
        future_source_glob=str(tmp_path / "processed" / "**" / "*2025-26*_with_sofa.csv"),
    )

    assert future_targets_output.exists()
    assert out_json.exists()
    assert out_csv.exists()
    assert payload["future_source_files"][0]["name"] == "austrian_bundesliga_2025-26_with_sofa.csv"
    season_rows = {row["season"]: row for row in payload["season_rows"]}
    assert season_rows["2023/24"]["labeled_rows"] == 1
    assert season_rows["2024/25"]["labeled_rows"] == 1

    disk_payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert disk_payload["total_rows"] == 3
