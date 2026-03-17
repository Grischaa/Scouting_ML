from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import scouting_ml.services.market_value_service as market_value_service
from scouting_ml.scripts.build_future_scout_score import build_future_scout_score


def test_build_future_scout_score_writes_enriched_predictions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "clean.parquet"
    val_predictions = tmp_path / "val.csv"
    test_predictions = tmp_path / "test.csv"
    out_val = tmp_path / "val_scored.csv"
    out_test = tmp_path / "test_scored.csv"
    diagnostics_out = tmp_path / "future_score.json"

    dataset_rows = []
    val_rows = []
    test_rows = []
    player_specs = [
        ("p1", "Eredivisie", "FW", 0.95, 0.90, [2_000_000, 5_000_000, 6_000_000]),
        ("p2", "Eredivisie", "FW", 0.85, 0.82, [2_500_000, 4_000_000, 4_600_000]),
        ("p3", "Primeira Liga", "MF", 0.78, 0.76, [3_000_000, 4_200_000, 4_900_000]),
        ("p4", "Primeira Liga", "MF", 0.35, 0.32, [5_000_000, 4_500_000, 4_000_000]),
        ("p5", "Belgian Pro League", "DF", 0.25, 0.24, [4_200_000, 3_600_000, 3_200_000]),
        ("p6", "Belgian Pro League", "DF", 0.15, 0.12, [3_500_000, 3_000_000, 2_400_000]),
    ]
    seasons = ["2023/24", "2024/25", "2025/26"]
    for player_id, league, position, val_score, test_score, values in player_specs:
        for season, value in zip(seasons, values):
            dataset_rows.append(
                {
                    "player_id": player_id,
                    "name": player_id.upper(),
                    "dob": "2003-01-01",
                    "season": season,
                    "league": league,
                    "model_position": position,
                    "market_value_eur": value,
                    "minutes": 1400,
                    "age": 21,
                }
            )
        val_rows.append(
            {
                "player_id": player_id,
                "name": player_id.upper(),
                "dob": "2003-01-01",
                "season": "2023/24",
                "league": league,
                "model_position": position,
                "market_value_eur": values[0],
                "minutes": 1400,
                "age": 21,
                "undervaluation_score": val_score,
                "undervaluation_confidence": val_score,
                "value_gap_conservative_eur": max(values[1] - values[0], 0),
            }
        )
        test_rows.append(
            {
                "player_id": player_id,
                "name": player_id.upper(),
                "dob": "2003-01-01",
                "season": "2024/25",
                "league": league,
                "model_position": position,
                "market_value_eur": values[1],
                "minutes": 1500,
                "age": 22,
                "undervaluation_score": test_score,
                "undervaluation_confidence": test_score,
                "value_gap_conservative_eur": max(values[2] - values[1], 0),
            }
        )

    pd.DataFrame(dataset_rows).to_parquet(dataset_path, index=False)
    pd.DataFrame(val_rows).to_csv(val_predictions, index=False)
    pd.DataFrame(test_rows).to_csv(test_predictions, index=False)

    diagnostics = build_future_scout_score(
        val_predictions_path=str(val_predictions),
        test_predictions_path=str(test_predictions),
        out_val_path=str(out_val),
        out_test_path=str(out_test),
        diagnostics_out=str(diagnostics_out),
        dataset_path=str(dataset_path),
        min_next_minutes=450.0,
        min_minutes=900.0,
        label_mode="positive_growth",
        k_eval=3,
    )

    assert out_val.exists()
    assert out_test.exists()
    assert diagnostics_out.exists()
    assert diagnostics["training_rows"] == 6
    assert diagnostics["val_metrics"]["precision_at_k_blend"] is not None

    val_scored = pd.read_csv(out_val)
    assert "future_growth_probability" in val_scored.columns
    assert "future_scout_blend_score" in val_scored.columns

    payload = json.loads(diagnostics_out.read_text(encoding="utf-8"))
    assert payload["features"]["base_rank_column"] == "undervaluation_score"
    assert payload["val_metrics"]["k_eval"] == 3


def test_query_scout_targets_prefers_future_scout_blend_score(tmp_path: Path, monkeypatch) -> None:
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "league": "Eredivisie",
                "market_value_eur": 2_000_000,
                "fair_value_eur": 4_000_000,
                "value_gap_conservative_eur": 2_000_000,
                "undervaluation_confidence": 0.8,
                "minutes": 1400,
                "age": 21,
                "scout_target_score": 9.0,
                "future_scout_blend_score": 0.10,
            },
            {
                "player_id": "p2",
                "league": "Eredivisie",
                "market_value_eur": 2_000_000,
                "fair_value_eur": 4_000_000,
                "value_gap_conservative_eur": 2_000_000,
                "undervaluation_confidence": 0.8,
                "minutes": 1400,
                "age": 21,
                "scout_target_score": 1.0,
                "future_scout_blend_score": 0.95,
            },
        ]
    ).to_csv(pred_path, index=False)

    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(pred_path))
    market_value_service._PRED_CACHE.clear()

    payload = market_value_service.query_scout_targets(
        split="test",
        top_n=1,
        min_minutes=900,
        max_age=23,
        min_confidence=0.5,
        min_value_gap_eur=1_000_000.0,
        non_big5_only=False,
    )

    assert payload["diagnostics"]["score_column"] == "future_scout_blend_score"
    assert payload["items"][0]["player_id"] == "p2"
