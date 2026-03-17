from __future__ import annotations

from pathlib import Path

import pandas as pd

from scouting_ml.reporting.future_value_benchmarks import (
    build_future_value_benchmark_payload,
    write_future_value_benchmark_report,
)


def test_future_value_benchmark_payload_and_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "clean.parquet"
    val_predictions = tmp_path / "val_predictions.csv"
    test_predictions = tmp_path / "test_predictions.csv"

    dataset_rows = []
    player_specs = [
        ("p1", "Eredivisie", "FW", [2_000_000, 4_000_000, 5_000_000]),
        ("p2", "Eredivisie", "MF", [3_000_000, 2_400_000, 2_000_000]),
        ("p3", "Eredivisie", "DF", [1_800_000, 2_600_000, 3_600_000]),
        ("p4", "Primeira Liga", "FW", [2_500_000, 2_000_000, 1_800_000]),
        ("p5", "Primeira Liga", "MF", [4_000_000, 5_200_000, 6_600_000]),
        ("p6", "Primeira Liga", "DF", [1_200_000, 900_000, 800_000]),
    ]
    seasons = ["2023/24", "2024/25", "2025/26"]
    for player_id, league, position, values in player_specs:
        for season, value in zip(seasons, values):
            dataset_rows.append(
                {
                    "player_id": player_id,
                    "name": player_id.upper(),
                    "dob": "2002-01-01",
                    "league": league,
                    "season": season,
                    "model_position": position,
                    "market_value_eur": value,
                    "minutes": 1400,
                }
            )
    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)

    val_scores = {
        "p1": 0.98,
        "p3": 0.91,
        "p5": 0.88,
        "p2": 0.45,
        "p4": 0.41,
        "p6": 0.18,
    }
    test_scores = {
        "p5": 0.95,
        "p3": 0.90,
        "p1": 0.86,
        "p2": 0.50,
        "p4": 0.34,
        "p6": 0.12,
    }
    val_rows = []
    test_rows = []
    for player_id, league, position, values in player_specs:
        val_rows.append(
            {
                "player_id": player_id,
                "name": player_id.upper(),
                "dob": "2002-01-01",
                "league": league,
                "season": "2023/24",
                "model_position": position,
                "market_value_eur": values[0],
                "minutes": 1400,
                "undervaluation_score": val_scores[player_id],
            }
        )
        test_rows.append(
            {
                "player_id": player_id,
                "name": player_id.upper(),
                "dob": "2002-01-01",
                "league": league,
                "season": "2024/25",
                "model_position": position,
                "market_value_eur": values[1],
                "minutes": 1500,
                "undervaluation_score": test_scores[player_id],
            }
        )
    pd.DataFrame(val_rows).to_csv(val_predictions, index=False)
    pd.DataFrame(test_rows).to_csv(test_predictions, index=False)

    payload = build_future_value_benchmark_payload(
        test_predictions_path=str(test_predictions),
        val_predictions_path=str(val_predictions),
        dataset_path=str(dataset_path),
        k_values=(2, 3),
        cohort_min_labeled=2,
        min_next_minutes=450.0,
        min_minutes=900.0,
        top_realized_limit=3,
    )

    assert payload["target_source"]["source"] == "dataset_built_in_memory"
    assert payload["splits"]["val"]["score_column"] == "undervaluation_score"
    assert payload["splits"]["val"]["join"]["labeled_rows"] == 6
    assert payload["splits"]["test"]["join"]["labeled_rows"] == 6

    val_overall_rows = [
        row
        for row in payload["splits"]["val"]["precision_at_k"]["positive_growth"]
        if row["cohort_type"] == "overall" and row["k"] == 2
    ]
    assert len(val_overall_rows) == 1
    assert float(val_overall_rows[0]["precision_at_k"]) == 1.0
    assert float(val_overall_rows[0]["positive_rate"]) == 0.5

    out_paths = write_future_value_benchmark_report(
        payload,
        out_json=str(tmp_path / "future_benchmark.json"),
        out_md=str(tmp_path / "future_benchmark.md"),
    )
    assert Path(out_paths["json"]).exists()
    md_text = Path(out_paths["markdown"]).read_text(encoding="utf-8")
    assert "Future Value Benchmark" in md_text
    assert "VAL Split" in md_text
    assert "TEST Split" in md_text


def test_future_value_benchmark_can_rejoin_predictions_already_carrying_future_columns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "clean.parquet"
    predictions = tmp_path / "predictions.csv"

    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": "2023/24",
                "league": "Eredivisie",
                "model_position": "FW",
                "market_value_eur": 2_000_000,
                "minutes": 1200,
            },
            {
                "player_id": "p1",
                "season": "2024/25",
                "league": "Eredivisie",
                "model_position": "FW",
                "market_value_eur": 3_500_000,
                "minutes": 1300,
            },
        ]
    ).to_parquet(dataset_path, index=False)

    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": "2023/24",
                "league": "Eredivisie",
                "model_position": "FW",
                "market_value_eur": 2_000_000,
                "minutes": 1200,
                "undervaluation_score": 0.9,
                "has_next_season_target": 0,
                "value_growth_positive_flag": 0,
            }
        ]
    ).to_csv(predictions, index=False)

    payload = build_future_value_benchmark_payload(
        val_predictions_path=str(predictions),
        dataset_path=str(dataset_path),
        k_values=(1,),
        cohort_min_labeled=1,
        min_next_minutes=450.0,
        min_minutes=900.0,
    )

    assert payload["splits"]["val"]["join"]["labeled_rows"] == 1
