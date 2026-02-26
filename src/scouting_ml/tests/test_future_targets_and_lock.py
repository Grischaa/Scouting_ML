from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts.build_future_value_targets import build_future_value_targets
from scouting_ml.scripts.lock_market_value_artifacts import build_lock_bundle


def test_build_future_value_targets_creates_next_season_growth(tmp_path: Path) -> None:
    input_path = tmp_path / "clean.parquet"
    output_path = tmp_path / "future_targets.parquet"

    df = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": "2022/23",
                "season_end_year": 2023,
                "market_value_eur": 1_000_000,
                "minutes": 1000,
            },
            {
                "player_id": "p1",
                "season": "2023/24",
                "season_end_year": 2024,
                "market_value_eur": 1_500_000,
                "minutes": 1200,
            },
            {
                "player_id": "p1",
                "season": "2024/25",
                "season_end_year": 2025,
                "market_value_eur": 1_200_000,
                "minutes": 900,
            },
            {
                "player_id": "p2",
                "season": "2024/25",
                "season_end_year": 2025,
                "market_value_eur": 800_000,
                "minutes": 700,
            },
        ]
    )
    df.to_parquet(input_path, index=False)

    build_future_value_targets(
        input_path=str(input_path),
        output_path=str(output_path),
        min_next_minutes=450.0,
        drop_na_target=False,
    )

    out = pd.read_parquet(output_path)
    p1_2023 = out[(out["player_id"] == "p1") & (out["season"] == "2022/23")].iloc[0]
    p1_2024 = out[(out["player_id"] == "p1") & (out["season"] == "2023/24")].iloc[0]
    p1_2025 = out[(out["player_id"] == "p1") & (out["season"] == "2024/25")].iloc[0]

    assert int(p1_2023["has_next_season_target"]) == 1
    assert float(p1_2023["value_growth_next_season_eur"]) == 500_000
    assert int(p1_2024["has_next_season_target"]) == 1
    assert float(p1_2024["value_growth_next_season_eur"]) == -300_000
    assert int(p1_2025["has_next_season_target"]) == 0


def test_lock_bundle_writes_manifest_and_env(tmp_path: Path) -> None:
    test_predictions = tmp_path / "pred_test.csv"
    val_predictions = tmp_path / "pred_val.csv"
    metrics_path = tmp_path / "metrics.json"
    manifest_out = tmp_path / "model_manifest.json"
    env_out = tmp_path / "model_artifacts.env"

    pd.DataFrame([{"player_id": "p1", "fair_value_eur": 1_000_000, "market_value_eur": 900_000}]).to_csv(
        test_predictions,
        index=False,
    )
    pd.DataFrame([{"player_id": "p2", "fair_value_eur": 2_000_000, "market_value_eur": 1_800_000}]).to_csv(
        val_predictions,
        index=False,
    )
    metrics_path.write_text(
        json.dumps({"dataset": "x", "val_season": "2023/24", "test_season": "2024/25"}),
        encoding="utf-8",
    )

    build_lock_bundle(
        test_predictions=test_predictions,
        val_predictions=val_predictions,
        metrics_path=metrics_path,
        manifest_out=manifest_out,
        env_out=env_out,
        strict_artifacts=True,
        label="unit_test",
    )

    assert manifest_out.exists()
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest["label"] == "unit_test"
    assert manifest["artifacts"]["test_predictions"]["sha256"]

    env_text = env_out.read_text(encoding="utf-8")
    assert "SCOUTING_TEST_PREDICTIONS_PATH=" in env_text
    assert "SCOUTING_STRICT_ARTIFACTS=1" in env_text
