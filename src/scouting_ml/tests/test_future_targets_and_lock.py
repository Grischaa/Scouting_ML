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
    assert float(p1_2023["value_growth_positive_flag"]) == 1.0
    assert int(p1_2024["has_next_season_target"]) == 1
    assert float(p1_2024["value_growth_next_season_eur"]) == -300_000
    assert float(p1_2024["value_growth_positive_flag"]) == 0.0
    assert int(p1_2025["has_next_season_target"]) == 0
    assert pd.isna(p1_2025["value_growth_positive_flag"])
    assert pd.isna(p1_2025["value_growth_gt25pct_flag"])


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


def test_lock_bundle_writes_dual_champion_manifest(tmp_path: Path) -> None:
    future_test = tmp_path / "future_test.csv"
    future_val = tmp_path / "future_val.csv"
    future_metrics = tmp_path / "future.metrics.json"
    valuation_test = tmp_path / "valuation_test.csv"
    valuation_val = tmp_path / "valuation_val.csv"
    valuation_metrics = tmp_path / "valuation.metrics.json"
    manifest_out = tmp_path / "model_manifest.json"
    env_out = tmp_path / "model_artifacts.env"

    pd.DataFrame([{"player_id": "p1", "fair_value_eur": 900_000, "market_value_eur": 800_000}]).to_csv(
        future_test,
        index=False,
    )
    pd.DataFrame([{"player_id": "p2", "fair_value_eur": 1_900_000, "market_value_eur": 1_700_000}]).to_csv(
        future_val,
        index=False,
    )
    pd.DataFrame([{"player_id": "p1", "fair_value_eur": 1_000_000, "market_value_eur": 800_000}]).to_csv(
        valuation_test,
        index=False,
    )
    pd.DataFrame([{"player_id": "p2", "fair_value_eur": 2_000_000, "market_value_eur": 1_700_000}]).to_csv(
        valuation_val,
        index=False,
    )
    future_metrics.write_text(
        json.dumps({"dataset": "future", "val_season": "2023/24", "test_season": "2024/25", "trials_per_position": 1}),
        encoding="utf-8",
    )
    valuation_metrics.write_text(
        json.dumps({"dataset": "valuation", "val_season": "2023/24", "test_season": "2024/25", "trials_per_position": 60}),
        encoding="utf-8",
    )

    build_lock_bundle(
        test_predictions=future_test,
        val_predictions=future_val,
        metrics_path=future_metrics,
        manifest_out=manifest_out,
        env_out=env_out,
        strict_artifacts=True,
        label="future_bundle",
        primary_role="future_shortlist",
        valuation_test_predictions=valuation_test,
        valuation_val_predictions=valuation_val,
        valuation_metrics_path=valuation_metrics,
        valuation_label="prod60",
        future_shortlist_label="future_bundle",
    )

    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest["registry_version"] == 2
    assert manifest["legacy_default_role"] == "valuation"
    assert manifest["artifacts"]["metrics"]["path"] == str(valuation_metrics)
    assert manifest["valuation_champion"]["label"] == "prod60"
    assert manifest["future_shortlist_champion"]["label"] == "future_bundle"
    assert manifest["valuation_champion"]["lane_state"] == "stable"
    assert manifest["future_shortlist_champion"]["lane_state"] == "live"
    assert manifest["valuation_champion"]["promotion_state"] == "advisory_only"
    assert manifest["future_shortlist_champion"]["promotion_state"] == "advisory_only"
    assert manifest["valuation_champion"]["generated_at_utc"]
    assert manifest["future_shortlist_champion"]["generated_at_utc"]
    assert manifest["valuation_champion"]["artifacts"]["metrics"]["path"] == str(valuation_metrics)
    assert manifest["future_shortlist_champion"]["artifacts"]["metrics"]["path"] == str(future_metrics)

    env_text = env_out.read_text(encoding="utf-8")
    assert "SCOUTING_VALUATION_TEST_PREDICTIONS_PATH=" in env_text
    assert "SCOUTING_FUTURE_SHORTLIST_TEST_PREDICTIONS_PATH=" in env_text
    assert f"SCOUTING_TEST_PREDICTIONS_PATH={valuation_test}" in env_text


def test_lock_bundle_preserves_existing_valuation_when_future_shortlist_updates(tmp_path: Path) -> None:
    valuation_test = tmp_path / "valuation_test.csv"
    valuation_val = tmp_path / "valuation_val.csv"
    valuation_metrics = tmp_path / "valuation.metrics.json"
    future_test = tmp_path / "future_test.csv"
    future_val = tmp_path / "future_val.csv"
    future_metrics = tmp_path / "future.metrics.json"
    manifest_out = tmp_path / "model_manifest.json"
    env_out = tmp_path / "model_artifacts.env"

    pd.DataFrame([{"player_id": "p1"}]).to_csv(valuation_test, index=False)
    pd.DataFrame([{"player_id": "p2"}]).to_csv(valuation_val, index=False)
    valuation_metrics.write_text(
        json.dumps({"dataset": "valuation", "val_season": "2023/24", "test_season": "2024/25"}),
        encoding="utf-8",
    )
    build_lock_bundle(
        test_predictions=valuation_test,
        val_predictions=valuation_val,
        metrics_path=valuation_metrics,
        manifest_out=manifest_out,
        env_out=env_out,
        strict_artifacts=True,
        label="valuation_bundle",
        primary_role="valuation",
    )

    pd.DataFrame([{"player_id": "p3"}]).to_csv(future_test, index=False)
    pd.DataFrame([{"player_id": "p4"}]).to_csv(future_val, index=False)
    future_metrics.write_text(
        json.dumps({"dataset": "future", "val_season": "2024/25", "test_season": "2025/26"}),
        encoding="utf-8",
    )
    build_lock_bundle(
        test_predictions=future_test,
        val_predictions=future_val,
        metrics_path=future_metrics,
        manifest_out=manifest_out,
        env_out=env_out,
        strict_artifacts=True,
        label="future_bundle",
        primary_role="future_shortlist",
    )

    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest["legacy_default_role"] == "valuation"
    assert manifest["artifacts"]["metrics"]["path"] == str(valuation_metrics)
    assert manifest["valuation_champion"]["artifacts"]["metrics"]["path"] == str(valuation_metrics)
    assert manifest["future_shortlist_champion"]["artifacts"]["metrics"]["path"] == str(future_metrics)
    assert manifest["valuation_champion"]["lane_state"] == "stable"
    assert manifest["future_shortlist_champion"]["lane_state"] == "live"


def test_build_future_value_targets_dedupes_duplicate_player_seasons(tmp_path: Path) -> None:
    input_path = tmp_path / "clean_dupes.parquet"
    output_path = tmp_path / "future_targets_dupes.parquet"

    df = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "season": "2023/24",
                "season_end_year": 2024,
                "market_value_eur": 1_000_000,
                "minutes": 900,
            },
            {
                "player_id": "p1",
                "season": "2023/24",
                "season_end_year": 2024,
                "market_value_eur": 1_000_000,
                "minutes": 900,
            },
            {
                "player_id": "p1",
                "season": "2024/25",
                "season_end_year": 2025,
                "market_value_eur": 1_600_000,
                "minutes": 1200,
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
    assert len(out) == 2
    p1_2024 = out[(out["player_id"] == "p1") & (out["season"] == "2023/24")]
    assert len(p1_2024) == 1
    assert int(p1_2024.iloc[0]["has_next_season_target"]) == 1
    assert float(p1_2024.iloc[0]["value_growth_next_season_eur"]) == 600_000
