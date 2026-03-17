from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts.run_provider_promotion_pipeline import (
    _evaluate_promotion_gate,
    _prepare_effective_provider_config,
    _provider_coverage_from_clean_dataset,
)


def test_prepare_effective_provider_config_prunes_missing_sections(tmp_path: Path) -> None:
    statsbomb_root = tmp_path / "statsbomb"
    statsbomb_root.mkdir()
    config_path = tmp_path / "provider_config.json"
    stage_external_dir = tmp_path / "external"
    output_path = tmp_path / "provider_config.effective.json"

    config_path.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-03-11",
                "statsbomb": {
                    "open_data_root": str(statsbomb_root),
                },
                "fixture_context": {
                    "provider": "sportmonks",
                    "input_json": [str(tmp_path / "missing_fixture.json")],
                },
                "player_availability": {
                    "provider": "sportmonks",
                    "api_url": "fixtures/players/availability",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = _prepare_effective_provider_config(
        config_path=config_path,
        stage_external_dir=stage_external_dir,
        output_path=output_path,
    )
    effective = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert set(payload["active_sections"]) == {"statsbomb", "player_availability"}
    assert "fixture_context" not in effective
    assert effective["player_links"] == str(stage_external_dir / "player_provider_links.csv")
    assert effective["club_links"] == str(stage_external_dir / "club_provider_links.csv")
    assert effective["statsbomb"]["output"] == str(stage_external_dir / "statsbomb_player_season_features.csv")
    assert effective["player_availability"]["output"] == str(stage_external_dir / "player_availability.csv")


def test_provider_coverage_and_gate_use_union_coverage(tmp_path: Path) -> None:
    clean_path = tmp_path / "candidate_clean.csv"
    pd.DataFrame(
        [
            {"season": "2024/25", "sb_progressive_passes_per90": 1.0, "fixture_matches": None},
            {"season": "2024/25", "sb_progressive_passes_per90": None, "fixture_matches": 12.0},
            {"season": "2024/25", "sb_progressive_passes_per90": None, "fixture_matches": None},
            {"season": "2023/24", "sb_progressive_passes_per90": 2.0, "fixture_matches": None},
        ]
    ).to_csv(clean_path, index=False)

    coverage = _provider_coverage_from_clean_dataset(clean_dataset_path=clean_path)
    test_coverage = coverage["by_season"]["2024/25"]

    assert test_coverage["sb_"]["row_coverage_share"] == 1.0 / 3.0
    assert test_coverage["fixture_"]["row_coverage_share"] == 1.0 / 3.0
    assert test_coverage["any_provider_coverage_share"] == 2.0 / 3.0

    promotion = _evaluate_promotion_gate(
        baseline_test={
            "r2": 0.70,
            "wmape": 0.42,
            "lowmid_weighted_wmape": 0.50,
        },
        candidate_test={
            "r2": 0.72,
            "wmape": 0.40,
            "lowmid_weighted_wmape": 0.47,
        },
        baseline_backtest={
            "mean_test_r2": 0.71,
            "mean_test_wmape": 0.40,
        },
        candidate_backtest={
            "mean_test_r2": 0.73,
            "mean_test_wmape": 0.39,
        },
        provider_coverage=coverage,
        test_season="2024/25",
        min_test_provider_coverage=0.50,
        max_test_wmape_delta=0.0,
        min_test_r2_delta=0.0,
        max_test_lowmid_wmape_delta=0.0,
        max_backtest_test_wmape_delta=0.0,
        min_backtest_test_r2_delta=0.0,
    )

    assert promotion["passed"] is True
    assert promotion["reasons"] == []


def test_provider_coverage_ignores_metadata_flags_without_real_provider_rows(tmp_path: Path) -> None:
    clean_path = tmp_path / "candidate_clean_flags.csv"
    pd.DataFrame(
        [
            {
                "season": "2024/25",
                "sb_progressive_passes_per90": 1.2,
                "sb_has_data": True,
                "sb_non_null_share": 0.8,
            },
            {
                "season": "2024/25",
                "sb_progressive_passes_per90": None,
                "sb_has_data": False,
                "sb_non_null_share": 0.0,
            },
            {
                "season": "2024/25",
                "sb_progressive_passes_per90": None,
                "sb_has_data": None,
                "sb_non_null_share": None,
            },
        ]
    ).to_csv(clean_path, index=False)

    coverage = _provider_coverage_from_clean_dataset(clean_dataset_path=clean_path)
    test_coverage = coverage["by_season"]["2024/25"]

    assert test_coverage["sb_"]["cols"] == 1
    assert test_coverage["sb_"]["row_coverage_share"] == 1.0 / 3.0
    assert test_coverage["any_provider_coverage_share"] == 1.0 / 3.0


def test_promotion_gate_blocks_zero_coverage_and_regression() -> None:
    promotion = _evaluate_promotion_gate(
        baseline_test={
            "r2": 0.74,
            "wmape": 0.41,
            "lowmid_weighted_wmape": 0.49,
        },
        candidate_test={
            "r2": 0.70,
            "wmape": 0.43,
            "lowmid_weighted_wmape": 0.51,
        },
        baseline_backtest=None,
        candidate_backtest=None,
        provider_coverage={"by_season": {"2024/25": {"any_provider_coverage_share": 0.0}}},
        test_season="2024/25",
        min_test_provider_coverage=0.05,
        max_test_wmape_delta=0.0,
        min_test_r2_delta=0.0,
        max_test_lowmid_wmape_delta=0.0,
        max_backtest_test_wmape_delta=0.0,
        min_backtest_test_r2_delta=0.0,
    )

    assert promotion["passed"] is False
    assert len(promotion["reasons"]) == 4
