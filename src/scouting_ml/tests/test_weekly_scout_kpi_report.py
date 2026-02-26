from __future__ import annotations

import pandas as pd

from scouting_ml.scripts.weekly_scout_kpi_report import build_weekly_kpi_report


def test_weekly_scout_kpi_report_writes_outputs(tmp_path) -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "player_id": f"ered_{i}",
                "league": "Eredivisie",
                "model_position": "FW" if i % 2 == 0 else "MF",
                "season": "2024/25",
                "age": 21,
                "minutes": 1500 + i,
                "market_value_eur": 3_000_000.0 if i < 30 else 8_000_000.0,
                "value_segment": "under_5m" if i < 30 else "5m_to_20m",
                "scout_target_score": float(120 - i),
                "future_success": 1 if i < 25 else 0,
            }
        )
    for i in range(60):
        rows.append(
            {
                "player_id": f"lig1_{i}",
                "league": "Ligue 1",
                "model_position": "DF",
                "season": "2024/25",
                "age": 24,
                "minutes": 1400 + i,
                "market_value_eur": 12_000_000.0,
                "value_segment": "5m_to_20m",
                "scout_target_score": float(80 - i),
                "future_success": 1 if i < 8 else 0,
            }
        )

    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)

    payload = build_weekly_kpi_report(
        predictions_path=str(pred_path),
        out_dir=str(tmp_path / "reports"),
        split="test",
        k_values=(10, 25, 50),
        min_minutes=900,
        max_age=25,
        cohort_min_labeled=30,
    )

    assert payload["score_column"] == "scout_target_score"
    assert payload["label_column"] == "future_success"
    assert payload["row_count"] > 0

    out_items = payload["items"]
    assert any(row["cohort_type"] == "league" and row["cohort"] == "Eredivisie" for row in out_items)
    assert any(row["cohort_type"] == "position" and row["cohort"] == "FW" for row in out_items)
    assert any(row["cohort_type"] == "value_segment" and row["cohort"] == "under_5m" for row in out_items)

    csv_path = tmp_path / "reports"
    assert any(path.suffix == ".csv" for path in csv_path.iterdir())
    assert any(path.suffix == ".json" for path in csv_path.iterdir())
