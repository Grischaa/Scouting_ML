from __future__ import annotations

import asyncio
import builtins
import json
import sys
import time
import types
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from scouting_ml.api import main as api_main
from scouting_ml.api.main import app
from scouting_ml.api import routes_market_value as market_value_routes
from scouting_ml.scripts.lock_market_value_artifacts import build_lock_bundle
from scouting_ml.services import market_value_service as mvs
from scouting_ml.services import proxy_estimate_service as proxy_estimate_service_module
from scouting_ml.services import similarity_service as similarity_service_module
from scouting_ml.services import trajectory_service as trajectory_service_module


class _ASGITestClient:
    def __init__(self, app) -> None:
        self._app = app

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def _send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(_send())

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


def _write_test_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = [
        {
            "player_id": "epl_fw_1",
            "name": "Premier Prospect",
            "league": "Premier League",
            "club": "Example FC",
            "season": "2024/25",
            "age": 21,
            "model_position": "FW",
            "minutes": 2000,
            "market_value_eur": 15_000_000,
            "fair_value_eur": 21_000_000,
            "value_gap_conservative_eur": 6_000_000,
            "undervaluation_confidence": 1.2,
        },
        {
            "player_id": "ered_fw_1",
            "name": "Eredivisie Talent",
            "league": "Eredivisie",
            "club": "Ajax-lite",
            "season": "2024/25",
            "age": 20,
            "model_position": "FW",
            "minutes": 1850,
            "market_value_eur": 6_000_000,
            "fair_value_eur": 10_500_000,
            "value_gap_conservative_eur": 4_500_000,
            "undervaluation_confidence": 1.0,
        },
        {
            "player_id": "liga_mf_1",
            "name": "Primeira Mid",
            "league": "Primeira Liga",
            "club": "Lisbon Test",
            "season": "2024/25",
            "age": 22,
            "model_position": "MF",
            "minutes": 1700,
            "market_value_eur": 8_500_000,
            "fair_value_eur": 11_000_000,
            "value_gap_conservative_eur": 2_500_000,
            "undervaluation_confidence": 0.7,
        },
        {
            "player_id": "low_min_1",
            "name": "Low Minutes Player",
            "league": "Eredivisie",
            "club": "Minutes FC",
            "season": "2024/25",
            "age": 21,
            "model_position": "FW",
            "minutes": 450,
            "market_value_eur": 3_000_000,
            "fair_value_eur": 4_000_000,
            "value_gap_conservative_eur": 1_000_000,
            "undervaluation_confidence": 0.9,
        },
    ]
    df = pd.DataFrame(rows)
    test_path = tmp_path / "pred_test.csv"
    val_path = tmp_path / "pred_val.csv"
    metrics_path = tmp_path / "metrics.json"
    df.to_csv(test_path, index=False)
    df.to_csv(val_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": "tmp_dataset",
                "val_season": "2023/24",
                "test_season": "2024/25",
                "trials_per_position": 5,
                "optimize_metric": "lowmid_wmape",
                "overall": {"test": {"r2": 0.71, "mae_eur": 5_000_000}},
                "segments": {"test": {"under_5m": {"wmape": 0.42}}},
            }
        ),
        encoding="utf-8",
    )
    return test_path, val_path, metrics_path


def _write_profile_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for i in range(45):
        position_main = "Forward"
        position_alt = "Attack, Centre"
        if i % 4 == 1:
            position_main = "Right Winger"
            position_alt = "Attack"
        elif i % 4 == 2:
            position_main = "Left Winger"
            position_alt = "Attack"
        elif i % 4 == 3:
            position_main = "Second Striker"
            position_alt = "Attack"
        rows.append(
            {
                "player_id": f"profile_fw_{i}",
                "name": f"Profile FW {i}",
                "league": "Eredivisie",
                "club": "Profile Club",
                "season": "2024/25",
                "position_main": position_main,
                "position_alt": position_alt,
                "age": int(rng.integers(19, 25)),
                "model_position": "FW",
                "minutes": int(rng.integers(1100, 2600)),
                "market_value_eur": float(rng.integers(2_000_000, 22_000_000)),
                "fair_value_eur": float(rng.integers(3_000_000, 24_000_000)),
                "expected_value_low_eur": float(rng.integers(1_000_000, 14_000_000)),
                "expected_value_high_eur": float(rng.integers(6_000_000, 30_000_000)),
                "value_gap_conservative_eur": float(rng.integers(500_000, 6_000_000)),
                "undervaluation_confidence": float(rng.uniform(0.2, 1.2)),
                "prior_mae_eur": float(rng.integers(800_000, 4_500_000)),
                "sofa_goals_per90": float(rng.uniform(0.05, 0.65)),
                "sofa_assists_per90": float(rng.uniform(0.02, 0.45)),
                "sofa_expectedGoals_per90": float(rng.uniform(0.08, 0.70)),
                "sofa_totalShots_per90": float(rng.uniform(0.8, 4.8)),
                "sofa_keyPasses_per90": float(rng.uniform(0.4, 2.4)),
                "sofa_successfulDribbles_per90": float(rng.uniform(0.5, 3.0)),
                "sofa_accuratePassesPercentage": float(rng.uniform(68.0, 91.0)),
                "sofa_totalDuelsWonPercentage": float(rng.uniform(42.0, 63.0)),
                "sb_progressive_passes_per90": float(rng.uniform(1.0, 8.0)),
                "sb_progressive_carries_per90": float(rng.uniform(0.5, 5.0)),
                "sb_passes_into_box_per90": float(rng.uniform(0.1, 2.2)),
                "sb_pressures_per90": float(rng.uniform(1.0, 12.0)),
                "sb_snapshot_date": "2026-03-12",
                "sb_retrieved_at": "2026-03-12T08:15:00Z",
                "sb_source_version": "sb-v1",
                "sb_minutes_in_433": float(rng.integers(0, 1400)),
                "sb_minutes_in_4231": float(rng.integers(0, 900)),
                "avail_reports": float(rng.integers(0, 12)),
                "avail_start_share": float(rng.uniform(0.2, 1.0)),
                "avail_injury_count": float(rng.integers(0, 4)),
                "avail_snapshot_date": "2026-03-11",
                "avail_retrieved_at": "2026-03-11T07:00:00Z",
                "avail_source_version": "avail-v1",
                "fixture_matches": float(rng.integers(6, 30)),
                "fixture_mean_rest_days": float(rng.uniform(3.0, 8.0)),
                "fixture_congestion_share": float(rng.uniform(0.0, 0.5)),
                "fixture_snapshot_date": "2026-03-10",
                "fixture_retrieved_at": "2026-03-10T09:30:00Z",
                "fixture_source_version": "fixture-v1",
                "odds_implied_team_strength": float(rng.uniform(0.2, 0.7)),
                "odds_upset_probability": float(rng.uniform(0.2, 0.8)),
                "odds_snapshot_date": "2026-03-09",
                "odds_retrieved_at": "2026-03-09T06:45:00Z",
                "odds_source_version": "odds-v1",
                "injury_days_per_1000_min": float(rng.uniform(0.0, 12.0)),
                "contract_years_left": float(rng.uniform(0.3, 4.2)),
            }
        )

    rows[1].update(
        {
            "player_id": "profile_fw_target",
            "name": "Profile Target",
            "position_main": "Right Winger",
            "position_alt": "Attack",
            "age": 21,
            "minutes": 2200,
            "market_value_eur": 8_000_000.0,
            "fair_value_eur": 13_500_000.0,
            "expected_value_low_eur": 10_000_000.0,
            "expected_value_high_eur": 17_500_000.0,
            "value_gap_conservative_eur": 4_000_000.0,
            "undervaluation_confidence": 1.1,
            "prior_mae_eur": 800_000.0,
            "sofa_goals_per90": 0.62,
            "sofa_assists_per90": 0.38,
            "sofa_expectedGoals_per90": 0.66,
            "sofa_totalShots_per90": 4.6,
            "sofa_keyPasses_per90": 1.9,
            "sofa_successfulDribbles_per90": 2.9,
            "sofa_accuratePassesPercentage": 71.0,
            "sofa_totalDuelsWonPercentage": 44.0,
            "sb_progressive_passes_per90": 6.2,
            "sb_progressive_carries_per90": 3.4,
            "sb_passes_into_box_per90": 1.7,
            "sb_pressures_per90": 8.6,
            "sb_minutes_in_433": 1280.0,
            "sb_minutes_in_4231": 440.0,
            "avail_reports": 10.0,
            "avail_start_share": 0.6,
            "avail_injury_count": 3.0,
            "fixture_matches": 24.0,
            "fixture_mean_rest_days": 4.2,
            "fixture_congestion_share": 0.46,
            "odds_implied_team_strength": 0.41,
            "odds_upset_probability": 0.59,
            "injury_days_per_1000_min": 10.8,
            "contract_years_left": 0.8,
            "leaguectx_league_strength_index": 0.34,
        }
    )

    sparse = dict(rows[1])
    sparse.update(
        {
            "player_id": "profile_fw_sparse",
            "name": "Profile Sparse",
            "club": "Sparse Club",
            "market_value_eur": 3_200_000.0,
            "fair_value_eur": 5_400_000.0,
            "expected_value_low_eur": 3_800_000.0,
            "expected_value_high_eur": 7_000_000.0,
            "value_gap_conservative_eur": 1_600_000.0,
            "sofa_assists_per90": np.nan,
            "sofa_expectedGoals_per90": np.nan,
            "sofa_keyPasses_per90": np.nan,
            "sofa_successfulDribbles_per90": np.nan,
            "sb_progressive_passes_per90": np.nan,
            "sb_progressive_carries_per90": np.nan,
            "sb_pressures_per90": np.nan,
            "sofa_accuratePassesPercentage": np.nan,
            "sofa_totalDuelsWonPercentage": np.nan,
            "leaguectx_league_strength_index": 0.33,
        }
    )
    rows.append(sparse)

    df = pd.DataFrame(rows)
    test_path = tmp_path / "profile_pred_test.csv"
    val_path = tmp_path / "profile_pred_val.csv"
    metrics_path = tmp_path / "profile_metrics.json"
    df.to_csv(test_path, index=False)
    df.to_csv(val_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": "tmp_profile_dataset",
                "val_season": "2023/24",
                "test_season": "2024/25",
                "overall": {"test": {"r2": 0.70, "mae_eur": 3_100_000}},
            }
        ),
        encoding="utf-8",
    )
    return test_path, val_path, metrics_path


def _write_clean_profile_dataset(tmp_path: Path, source_csv: Path) -> Path:
    frame = pd.read_csv(source_csv)
    frame = frame.copy()
    frame["goals"] = frame.get("goals", frame.get("sofa_goals_per90", 0)).fillna(0) * 30
    frame["assists"] = frame.get("assists", frame.get("sofa_assists_per90", 0)).fillna(0) * 20
    frame["xg"] = frame.get("xg", frame.get("sofa_expectedGoals_per90", 0)).fillna(0) * 18
    frame["progressive_passes"] = frame.get("progressive_passes", frame.get("sb_progressive_passes_per90", 0)).fillna(0) * 20
    frame["progressive_carries"] = frame.get("progressive_carries", frame.get("sb_progressive_carries_per90", 0)).fillna(0) * 20

    target = frame.loc[frame["player_id"] == "profile_fw_target"].iloc[0].copy()
    history_rows: list[dict[str, object]] = []
    for season, market_value, fair_value, minutes, goals, assists, xg in (
        ("2022/23", 3_200_000.0, 4_400_000.0, 1450, 9.0, 4.0, 7.2),
        ("2023/24", 5_000_000.0, 7_600_000.0, 1875, 11.0, 7.0, 9.6),
    ):
        item = dict(target)
        item["season"] = season
        item["market_value_eur"] = market_value
        item["fair_value_eur"] = fair_value
        item["minutes"] = minutes
        item["goals"] = goals
        item["assists"] = assists
        item["xg"] = xg
        item["progressive_passes"] = 84.0 if season == "2022/23" else 109.0
        item["progressive_carries"] = 52.0 if season == "2022/23" else 70.0
        history_rows.append(item)

    clean = pd.concat([frame, pd.DataFrame(history_rows)], ignore_index=True)
    clean_path = tmp_path / "champion_players_clean.parquet"
    clean.to_parquet(clean_path, index=False)
    return clean_path


def _write_recruitment_filter_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = [
        {
            "player_id": "wing_target",
            "name": "Wing Target",
            "league": "Eredivisie",
            "club": "Recruit FC",
            "season": "2024/25",
            "age": 21,
            "model_position": "FW",
            "position_main": "Right Winger",
            "position_alt": "Attack",
            "minutes": 2100,
            "market_value_eur": 6_000_000,
            "fair_value_eur": 10_000_000,
            "value_gap_conservative_eur": 4_000_000,
            "undervaluation_confidence": 1.1,
            "contract_years_left": 1.0,
        },
        {
            "player_id": "wing_expensive",
            "name": "Wing Expensive",
            "league": "Eredivisie",
            "club": "Recruit FC",
            "season": "2024/25",
            "age": 21,
            "model_position": "FW",
            "position_main": "Left Winger",
            "position_alt": "Attack",
            "minutes": 2200,
            "market_value_eur": 14_000_000,
            "fair_value_eur": 18_000_000,
            "value_gap_conservative_eur": 4_000_000,
            "undervaluation_confidence": 1.0,
            "contract_years_left": 1.0,
        },
        {
            "player_id": "wing_long_contract",
            "name": "Wing Long Contract",
            "league": "Primeira Liga",
            "club": "Recruit FC",
            "season": "2024/25",
            "age": 22,
            "model_position": "FW",
            "position_main": "Left Winger",
            "position_alt": "Attack",
            "minutes": 1900,
            "market_value_eur": 5_500_000,
            "fair_value_eur": 9_000_000,
            "value_gap_conservative_eur": 3_500_000,
            "undervaluation_confidence": 0.95,
            "contract_years_left": 3.5,
        },
        {
            "player_id": "cb_target",
            "name": "CB Target",
            "league": "Belgian Pro League",
            "club": "Backline FC",
            "season": "2024/25",
            "age": 20,
            "model_position": "DF",
            "position_main": "Centre-Back",
            "position_alt": "Defender, Centre",
            "minutes": 2300,
            "market_value_eur": 4_000_000,
            "fair_value_eur": 7_500_000,
            "value_gap_conservative_eur": 3_500_000,
            "undervaluation_confidence": 1.05,
            "contract_years_left": 0.8,
        },
        {
            "player_id": "big5_forward",
            "name": "Big5 Forward",
            "league": "Premier League",
            "club": "Big Money FC",
            "season": "2024/25",
            "age": 21,
            "model_position": "FW",
            "position_main": "Centre-Forward",
            "position_alt": "Attack, Centre",
            "minutes": 2050,
            "market_value_eur": 6_500_000,
            "fair_value_eur": 9_800_000,
            "value_gap_conservative_eur": 3_300_000,
            "undervaluation_confidence": 1.0,
            "contract_years_left": 1.2,
        },
    ]
    df = pd.DataFrame(rows)
    test_path = tmp_path / "recruitment_pred_test.csv"
    val_path = tmp_path / "recruitment_pred_val.csv"
    metrics_path = tmp_path / "recruitment_metrics.json"
    df.to_csv(test_path, index=False)
    df.to_csv(val_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": "tmp_recruitment_dataset",
                "val_season": "2023/24",
                "test_season": "2024/25",
                "trials_per_position": 7,
                "overall": {"test": {"r2": 0.72, "mae_eur": 3_000_000}},
            }
        ),
        encoding="utf-8",
    )
    return test_path, val_path, metrics_path


def _write_league_adjustment_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = {
        "season": "2024/25",
        "history_strength_score": 68.0,
        "history_strength_coverage": 0.80,
        "future_growth_probability": 0.58,
        "future_scout_blend_score": 0.64,
        "has_next_season_target": 1,
        "talent_impact_score": 72.0,
        "talent_impact_coverage": 85.0,
        "talent_technical_score": 70.0,
        "talent_technical_coverage": 82.0,
        "talent_tactical_score": 69.0,
        "talent_tactical_coverage": 83.0,
        "talent_physical_score": 67.0,
        "talent_physical_coverage": 80.0,
        "talent_context_score": 66.0,
        "talent_context_coverage": 79.0,
        "talent_trajectory_score": 73.0,
        "talent_trajectory_coverage": 84.0,
        "sb_progressive_carries_per90": 3.8,
        "sb_passes_into_box_per90": 1.2,
        "sb_shot_assists_per90": 0.7,
        "sb_pressures_per90": 6.5,
        "sofa_successfulDribbles_per90": 2.4,
        "sofa_expectedGoals_per90": 0.28,
        "sofa_accurateCrossesPercentage": 31.0,
    }
    rows = [
        {
            **base,
            "player_id": "est_winger",
            "name": "Estonia Winger",
            "league": "Estonian Meistriliiga",
            "club": "Tallinn Test",
            "age": 20,
            "model_position": "FW",
            "position_main": "Right Winger",
            "position_alt": "Attack",
            "minutes": 2100,
            "market_value_eur": 1_000_000,
            "fair_value_eur": 5_000_000,
            "expected_value_eur": 5_000_000,
            "value_gap_eur": 4_000_000,
            "value_gap_conservative_eur": 3_000_000,
            "undervaluation_confidence": 1.0,
            "contract_years_left": 1.0,
            "leaguectx_league_strength_index": 0.06,
            "uefa_coeff_5yr_total": 5.0,
            "talent_position_family": "W",
            "future_potential_score": 79.0,
            "future_potential_confidence": 71.0,
            "sb_progressive_carries_per90": 4.9,
            "sofa_successfulDribbles_per90": 3.2,
        },
        {
            **base,
            "player_id": "cze_cm",
            "name": "Czech Mid",
            "league": "Czech Fortuna Liga",
            "club": "Prague Test",
            "age": 22,
            "model_position": "MF",
            "position_main": "Central Midfielder",
            "position_alt": "Midfield",
            "minutes": 2200,
            "market_value_eur": 2_500_000,
            "fair_value_eur": 6_000_000,
            "expected_value_eur": 6_000_000,
            "value_gap_eur": 3_500_000,
            "value_gap_conservative_eur": 2_700_000,
            "undervaluation_confidence": 0.95,
            "contract_years_left": 1.8,
            "leaguectx_league_strength_index": 0.18,
            "uefa_coeff_5yr_total": 12.0,
            "talent_position_family": "CM",
            "future_potential_score": 74.0,
            "future_potential_confidence": 69.0,
        },
        {
            **base,
            "player_id": "ned_winger",
            "name": "Dutch Winger",
            "league": "Eredivisie",
            "club": "Amsterdam Test",
            "age": 21,
            "model_position": "FW",
            "position_main": "Left Winger",
            "position_alt": "Attack",
            "minutes": 2250,
            "market_value_eur": 4_000_000,
            "fair_value_eur": 6_500_000,
            "expected_value_eur": 6_500_000,
            "value_gap_eur": 2_500_000,
            "value_gap_conservative_eur": 1_800_000,
            "undervaluation_confidence": 0.9,
            "contract_years_left": 2.0,
            "leaguectx_league_strength_index": 0.55,
            "uefa_coeff_5yr_total": 50.0,
            "talent_position_family": "W",
            "future_potential_score": 77.0,
            "future_potential_confidence": 72.0,
            "sb_progressive_carries_per90": 4.4,
            "sofa_successfulDribbles_per90": 2.9,
        },
        {
            **base,
            "player_id": "fin_striker",
            "name": "Finland Striker",
            "league": "Veikkausliiga",
            "club": "Helsinki Test",
            "age": 23,
            "model_position": "FW",
            "position_main": "Centre-Forward",
            "position_alt": "Attack, Centre",
            "minutes": 2050,
            "market_value_eur": 3_000_000,
            "fair_value_eur": 5_000_000,
            "expected_value_eur": 5_000_000,
            "value_gap_eur": 2_000_000,
            "value_gap_conservative_eur": 1_200_000,
            "undervaluation_confidence": 0.85,
            "contract_years_left": 1.4,
            "leaguectx_league_strength_index": 0.14,
            "uefa_coeff_5yr_total": 7.0,
            "talent_position_family": "ST",
            "future_potential_score": 70.0,
            "future_potential_confidence": 66.0,
        },
    ]
    df = pd.DataFrame(rows)
    test_path = tmp_path / "league_adjustment_test.csv"
    val_path = tmp_path / "league_adjustment_val.csv"
    metrics_path = tmp_path / "league_adjustment.metrics.json"
    df.to_csv(test_path, index=False)
    df.to_csv(val_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": "tmp_league_adjustment_dataset",
                "val_season": "2023/24",
                "test_season": "2024/25",
                "overall": {"test": {"r2": 0.70, "wmape": 0.42}},
            }
        ),
        encoding="utf-8",
    )
    holdouts = {
        "estonian_meistriliiga": {
            "league": "Estonian Meistriliiga",
            "overall": {"r2": -796.77, "wmape": 14.21, "interval_coverage": 0.337},
            "domain_shift": {"mean_abs_shift_z": 1.8},
        },
        "czech_fortuna_liga": {
            "league": "Czech Fortuna Liga",
            "overall": {"r2": -1.27, "wmape": 1.96, "interval_coverage": 0.48},
            "domain_shift": {"mean_abs_shift_z": 1.2},
        },
        "eredivisie": {
            "league": "Eredivisie",
            "overall": {"r2": 0.72, "wmape": 0.34, "interval_coverage": 0.81},
            "domain_shift": {"mean_abs_shift_z": 0.42},
        },
    }
    for slug, payload in holdouts.items():
        (tmp_path / f"league_adjustment.holdout_{slug}.metrics.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    return test_path, val_path, metrics_path


def _write_system_fit_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = {
        "season": "2024/25",
        "expected_value_eur": 0.0,
        "value_gap_eur": 0.0,
        "history_strength_score": 72.0,
        "history_strength_coverage": 0.85,
        "future_growth_probability": 0.64,
        "future_scout_blend_score": 0.71,
        "has_next_season_target": 1,
        "sb_progressive_passes_per90": 4.5,
        "sb_progressive_carries_per90": 3.2,
        "sb_passes_into_box_per90": 1.2,
        "sb_shot_assists_per90": 0.65,
        "sb_pressures_per90": 7.0,
        "sb_duel_win_rate": 0.58,
        "sb_aerial_win_rate": 0.55,
        "sofa_tackles_per90": 1.8,
        "sofa_interceptions_per90": 1.7,
        "sofa_successfulDribbles_per90": 2.3,
        "sofa_expectedGoals_per90": 0.32,
        "sofa_accuratePassesPercentage": 83.0,
        "sofa_keyPasses_per90": 1.4,
        "sofa_totalShots_per90": 2.8,
        "sofa_clearances_per90": 4.3,
        "sofa_accurateCrossesPercentage": 31.0,
        "talent_impact_score": 68.0,
        "talent_impact_coverage": 85.0,
        "talent_technical_score": 67.0,
        "talent_technical_coverage": 84.0,
        "talent_tactical_score": 69.0,
        "talent_tactical_coverage": 86.0,
        "talent_physical_score": 66.0,
        "talent_physical_coverage": 82.0,
        "talent_context_score": 65.0,
        "talent_context_coverage": 80.0,
        "talent_trajectory_score": 71.0,
        "talent_trajectory_coverage": 84.0,
    }
    rows = [
        {
            **base,
            "player_id": "sf_gk_1",
            "name": "System GK",
            "league": "Eredivisie",
            "club": "System XI",
            "age": 24,
            "model_position": "GK",
            "position_main": "Goalkeeper",
            "minutes": 2500,
            "market_value_eur": 3_000_000,
            "fair_value_eur": 4_200_000,
            "value_gap_conservative_eur": 1_200_000,
            "contract_years_left": 1.5,
            "talent_position_family": "GK",
            "future_potential_score": 61.0,
            "future_potential_confidence": 70.0,
        },
        {
            **base,
            "player_id": "sf_cb_1",
            "name": "System CB",
            "league": "Eredivisie",
            "club": "System XI",
            "age": 23,
            "model_position": "DF",
            "position_main": "Centre-Back",
            "position_alt": "Defender, Centre",
            "minutes": 2300,
            "market_value_eur": 5_500_000,
            "fair_value_eur": 7_800_000,
            "value_gap_conservative_eur": 2_300_000,
            "contract_years_left": 2.0,
            "talent_position_family": "CB",
            "talent_tactical_score": 78.0,
            "future_potential_score": 69.0,
            "future_potential_confidence": 72.0,
        },
        {
            **base,
            "player_id": "sf_fb_1",
            "name": "System FB",
            "league": "Eredivisie",
            "club": "System XI",
            "age": 22,
            "model_position": "DF",
            "position_main": "Right Back",
            "position_alt": "Defender, Right",
            "minutes": 2150,
            "market_value_eur": 4_800_000,
            "fair_value_eur": 7_100_000,
            "value_gap_conservative_eur": 2_300_000,
            "contract_years_left": 1.2,
            "talent_position_family": "FB",
            "talent_technical_score": 74.0,
            "future_potential_score": 74.0,
            "future_potential_confidence": 73.0,
            "sofa_accurateCrossesPercentage": 36.0,
        },
        {
            **base,
            "player_id": "sf_dm_1",
            "name": "System DM",
            "league": "Eredivisie",
            "club": "System XI",
            "age": 24,
            "model_position": "MF",
            "position_main": "Defensive Midfielder",
            "minutes": 2400,
            "market_value_eur": 5_200_000,
            "fair_value_eur": 7_600_000,
            "value_gap_conservative_eur": 2_400_000,
            "contract_years_left": 1.1,
            "talent_position_family": "CM",
            "talent_tactical_score": 79.0,
            "talent_context_score": 70.0,
            "future_potential_score": 68.0,
            "future_potential_confidence": 71.0,
        },
        {
            **base,
            "player_id": "sf_cm_watch",
            "name": "Watch League Mid",
            "league": "Primeira Liga",
            "club": "Watch FC",
            "age": 21,
            "model_position": "MF",
            "position_main": "Central Midfielder",
            "minutes": 2100,
            "market_value_eur": 4_900_000,
            "fair_value_eur": 7_200_000,
            "value_gap_conservative_eur": 2_300_000,
            "contract_years_left": 1.7,
            "talent_position_family": "CM",
            "talent_technical_score": 76.0,
            "talent_trajectory_score": 79.0,
            "future_potential_score": 81.0,
            "future_potential_confidence": 69.0,
        },
        {
            **base,
            "player_id": "sf_am_1",
            "name": "System Creator",
            "league": "Eredivisie",
            "club": "System XI",
            "age": 22,
            "model_position": "MF",
            "position_main": "Attacking Midfielder",
            "minutes": 1950,
            "market_value_eur": 5_000_000,
            "fair_value_eur": 8_100_000,
            "value_gap_conservative_eur": 3_100_000,
            "contract_years_left": 1.0,
            "talent_position_family": "AM",
            "talent_impact_score": 79.0,
            "talent_technical_score": 81.0,
            "future_potential_score": 84.0,
            "future_potential_confidence": 75.0,
            "sb_shot_assists_per90": 1.1,
            "sb_passes_into_box_per90": 1.8,
        },
        {
            **base,
            "player_id": "sf_rw_good",
            "name": "Trusted Winger",
            "league": "Eredivisie",
            "club": "System XI",
            "age": 20,
            "model_position": "FW",
            "position_main": "Right Winger",
            "position_alt": "Attack",
            "minutes": 2200,
            "market_value_eur": 4_000_000,
            "fair_value_eur": 7_400_000,
            "value_gap_conservative_eur": 3_400_000,
            "contract_years_left": 1.3,
            "talent_position_family": "W",
            "talent_impact_score": 83.0,
            "talent_technical_score": 80.0,
            "talent_trajectory_score": 82.0,
            "future_potential_score": 91.0,
            "future_potential_confidence": 77.0,
            "sb_progressive_carries_per90": 5.8,
            "sofa_successfulDribbles_per90": 3.4,
            "sofa_expectedGoals_per90": 0.44,
        },
        {
            **base,
            "player_id": "sf_rw_blocked",
            "name": "Blocked Winger",
            "league": "Belgian Pro League",
            "club": "Blocked FC",
            "age": 20,
            "model_position": "FW",
            "position_main": "Right Winger",
            "position_alt": "Attack",
            "minutes": 2100,
            "market_value_eur": 8_000_000,
            "fair_value_eur": 11_200_000,
            "value_gap_conservative_eur": 3_200_000,
            "contract_years_left": 3.0,
            "talent_position_family": "W",
            "talent_impact_score": 80.0,
            "talent_technical_score": 78.0,
            "future_potential_score": 88.0,
            "future_potential_confidence": 74.0,
            "sb_progressive_carries_per90": 5.1,
            "sofa_successfulDribbles_per90": 3.0,
        },
        {
            **base,
            "player_id": "sf_st_1",
            "name": "System Nine",
            "league": "Eredivisie",
            "club": "System XI",
            "age": 23,
            "model_position": "FW",
            "position_main": "Centre-Forward",
            "position_alt": "Attack, Centre",
            "minutes": 2250,
            "market_value_eur": 6_500_000,
            "fair_value_eur": 9_000_000,
            "value_gap_conservative_eur": 2_500_000,
            "contract_years_left": 1.4,
            "talent_position_family": "ST",
            "talent_impact_score": 81.0,
            "talent_technical_score": 75.0,
            "future_potential_score": 78.0,
            "future_potential_confidence": 73.0,
            "sofa_expectedGoals_per90": 0.58,
            "sofa_totalShots_per90": 4.1,
            "sb_pressures_per90": 8.4,
        },
    ]
    df = pd.DataFrame(rows)
    df["expected_value_eur"] = df["fair_value_eur"]
    df["value_gap_eur"] = df["value_gap_conservative_eur"]
    test_path = tmp_path / "system_fit_test.csv"
    val_path = tmp_path / "system_fit_val.csv"
    metrics_path = tmp_path / "system_fit_metrics.json"
    df.to_csv(test_path, index=False)
    df.to_csv(val_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": "tmp_system_fit_dataset",
                "val_season": "2023/24",
                "test_season": "2024/25",
                "trials_per_position": 5,
                "overall": {"test": {"r2": 0.72, "mae_eur": 3_000_000}},
            }
        ),
        encoding="utf-8",
    )
    return test_path, val_path, metrics_path


def _reset_service_caches() -> None:
    mvs._PRED_CACHE.clear()  # type: ignore[attr-defined]
    mvs._METRICS_CACHE = {}  # type: ignore[attr-defined]
    mvs._RESIDUAL_CALIBRATION_CACHE = None  # type: ignore[attr-defined]
    mvs._LEAGUE_HOLDOUT_CACHE = None  # type: ignore[attr-defined]
    mvs._INGESTION_HEALTH_CACHE = None  # type: ignore[attr-defined]
    mvs._OPERATOR_HEALTH_CACHE = None  # type: ignore[attr-defined]
    mvs._UI_BOOTSTRAP_CACHE = {}  # type: ignore[attr-defined]
    proxy_estimate_service_module._PROXY_ESTIMATE_SERVICE = None  # type: ignore[attr-defined]
    similarity_service_module._SIMILARITY_SERVICE = None  # type: ignore[attr-defined]
    trajectory_service_module._TRAJECTORY_SERVICE = None  # type: ignore[attr-defined]
    api_main._STARTUP_CHECKS_DONE = False  # type: ignore[attr-defined]
    api_main._STARTUP_CHECKS_ERROR = None  # type: ignore[attr-defined]


def _write_dual_role_artifacts(tmp_path: Path) -> dict[str, Path]:
    base_rows = [
        {
            "player_id": "dual_fw_1",
            "name": "Dual Role Prospect",
            "league": "Eredivisie",
            "club": "Dual FC",
            "season": "2024/25",
            "age": 20,
            "model_position": "FW",
            "minutes": 2100,
            "market_value_eur": 8_000_000,
            "fair_value_eur": 13_500_000,
            "value_gap_conservative_eur": 5_500_000,
            "undervaluation_confidence": 1.1,
        },
        {
            "player_id": "dual_mf_1",
            "name": "Dual Role Mid",
            "league": "Primeira Liga",
            "club": "Dual Lisbon",
            "season": "2024/25",
            "age": 22,
            "model_position": "MF",
            "minutes": 1800,
            "market_value_eur": 7_000_000,
            "fair_value_eur": 10_000_000,
            "value_gap_conservative_eur": 3_000_000,
            "undervaluation_confidence": 0.8,
        },
    ]
    valuation_df = pd.DataFrame(base_rows)
    future_df = valuation_df.copy()
    future_df["fair_value_eur"] = [11_000_000, 9_200_000]
    future_df["future_scout_blend_score"] = [0.93, 0.61]
    future_df["future_growth_probability"] = [0.74, 0.48]
    future_df["future_scout_score"] = [0.89, 0.57]
    future_df["has_next_season_target"] = [1, 1]
    future_df["next_market_value_eur"] = [15_000_000, 8_500_000]
    future_df["next_minutes"] = [2200, 1900]
    future_df["next_season"] = ["2025/26", "2025/26"]
    future_df["value_growth_positive_flag"] = [1.0, 1.0]
    future_df["value_growth_gt25pct_flag"] = [1.0, 0.0]
    future_df["value_growth_next_season_eur"] = [7_000_000.0, 1_500_000.0]
    future_df["value_growth_next_season_pct"] = [0.875, 0.214]
    future_df["value_growth_next_season_log_delta"] = [0.62, 0.19]
    future_df["_player_key"] = future_df["player_id"]
    future_df["_season_key"] = future_df["season"]

    valuation_test = tmp_path / "valuation_test.csv"
    valuation_val = tmp_path / "valuation_val.csv"
    valuation_metrics = tmp_path / "valuation.metrics.json"
    future_test = tmp_path / "future_test.csv"
    future_val = tmp_path / "future_val.csv"
    future_metrics = tmp_path / "future.metrics.json"
    manifest_out = tmp_path / "model_manifest.json"
    env_out = tmp_path / "model_artifacts.env"

    valuation_df.to_csv(valuation_test, index=False)
    valuation_df.to_csv(valuation_val, index=False)
    future_df.to_csv(future_test, index=False)
    future_df.to_csv(future_val, index=False)
    valuation_metrics.write_text(
        json.dumps(
            {
                "dataset": "valuation_dataset",
                "val_season": "2023/24",
                "test_season": "2024/25",
                "trials_per_position": 60,
                "overall": {"test": {"r2": 0.742, "wmape": 0.401}},
            }
        ),
        encoding="utf-8",
    )
    future_metrics.write_text(
        json.dumps(
            {
                "dataset": "future_dataset",
                "val_season": "2023/24",
                "test_season": "2024/25",
                "trials_per_position": 1,
                "overall": {"test": {"r2": 0.731, "wmape": 0.411}},
            }
        ),
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
    return {
        "valuation_test": valuation_test,
        "valuation_val": valuation_val,
        "valuation_metrics": valuation_metrics,
        "future_test": future_test,
        "future_val": future_val,
        "future_metrics": future_metrics,
        "manifest_out": manifest_out,
        "env_out": env_out,
    }


def test_api_root_returns_index() -> None:
    client = _ASGITestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["service"] == "scouting_ml_api"
    assert payload["docs"] == "/docs"
    assert payload["market_value_base"] == "/market-value"


def test_startup_checks_do_not_warm_optional_detail_services(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "_strict_artifacts_enabled", lambda: False)
    monkeypatch.setattr(
        api_main,
        "get_resolved_artifact_paths",
        lambda: {
            "test_predictions_path": "test.csv",
            "val_predictions_path": "val.csv",
            "metrics_path": "metrics.json",
        },
    )

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name in {
            "scouting_ml.services.similarity_service",
            "scouting_ml.services.proxy_estimate_service",
            "scouting_ml.services.trajectory_service",
        }:
            raise AssertionError(f"startup attempted to import optional warmup service: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    api_main._startup_checks()


def test_ui_bootstrap_endpoint_returns_aggregates_and_invalidates(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.delenv("SCOUTING_MODEL_MANIFEST_PATH", raising=False)
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get("/market-value/ui-bootstrap", params={"split": "test"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["split"] == "test"
    assert payload["seasons"] == ["2024/25"]
    assert "Eredivisie" in payload["leagues"]
    coverage = {row["league"]: row for row in payload["coverage_rows"]}
    assert coverage["Eredivisie"]["rows"] == 2
    assert coverage["Eredivisie"]["undervalued_share"] == pytest.approx(1.0)
    generated_at = payload["generated_at_utc"]

    cached = client.get("/market-value/ui-bootstrap", params={"split": "test"})
    assert cached.status_code == 200
    assert cached.json()["generated_at_utc"] == generated_at

    df = pd.read_csv(test_path)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "player_id": "bel_df_1",
                        "name": "Belgian Defender",
                        "league": "Belgian Pro League",
                        "club": "Bruges Test",
                        "season": "2025/26",
                        "age": 22,
                        "model_position": "DF",
                        "minutes": 1500,
                        "market_value_eur": 4_500_000,
                        "fair_value_eur": 6_000_000,
                        "value_gap_conservative_eur": 1_500_000,
                        "undervaluation_confidence": 0.8,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    time.sleep(0.01)
    df.to_csv(test_path, index=False)

    refreshed = client.get("/market-value/ui-bootstrap", params={"split": "test"})
    assert refreshed.status_code == 200
    refreshed_payload = refreshed.json()
    assert refreshed_payload["generated_at_utc"] != generated_at
    assert refreshed_payload["seasons"][0] == "2025/26"
    assert "Belgian Pro League" in refreshed_payload["leagues"]


def test_operator_health_endpoint_returns_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        market_value_routes,
        "get_operator_health",
        lambda: {
            "active_lanes": {
                "valuation": {"lane_state": "stable", "promotion_state": "promotable"},
                "future_shortlist": {"lane_state": "live", "promotion_state": "advisory_only"},
            },
            "promotion_gate": {"promotable": True},
            "ingestion_health": {"summary": {"status_counts": {"healthy": 1, "watch": 0, "blocked": 0}}},
            "stale_provider_snapshots": {"stale_count": 0},
            "live_partial_footprint": {"live_rows": 5},
        },
    )

    client = _ASGITestClient(app)
    resp = client.get("/market-value/operator-health")
    assert resp.status_code == 200
    payload = resp.json()["payload"]
    assert payload["promotion_gate"]["promotable"] is True
    assert payload["active_lanes"]["valuation"]["lane_state"] == "stable"
    assert payload["live_partial_footprint"]["live_rows"] == 5


def test_operator_health_payload_is_cached_until_artifacts_change(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.delenv("SCOUTING_MODEL_MANIFEST_PATH", raising=False)
    _reset_service_caches()

    calls = {"ingestion": 0, "live": 0}

    def counted_ingestion() -> dict[str, object]:
        calls["ingestion"] += 1
        return {"rows": [], "summary": {}, "_meta": {"source": "test"}}

    def counted_live() -> dict[str, object]:
        calls["live"] += 1
        return {"live_rows": 5, "live_share": 0.1}

    monkeypatch.setattr(mvs, "_get_ingestion_health_payload", counted_ingestion)
    monkeypatch.setattr(mvs, "_live_partial_footprint", counted_live)
    monkeypatch.setattr(mvs, "build_valuation_promotion_gate", lambda **kwargs: {"promotable": True, "holdout_coverage": {}})
    monkeypatch.setattr(mvs, "_load_json_snapshot", lambda path: {})
    monkeypatch.setattr(mvs, "get_model_manifest", lambda: {})

    first = mvs.get_operator_health()
    second = mvs.get_operator_health()
    assert first["live_partial_footprint"]["live_rows"] == 5
    assert second["live_partial_footprint"]["live_rows"] == 5
    assert calls == {"ingestion": 1, "live": 1}

    time.sleep(0.01)
    metrics_path.write_text(json.dumps({"dataset": "changed"}), encoding="utf-8")
    third = mvs.get_operator_health()
    assert third["generated_at_utc"] != first["generated_at_utc"]
    assert calls == {"ingestion": 2, "live": 2}


def test_model_manifest_and_scout_targets_endpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.delenv("SCOUTING_MODEL_MANIFEST_PATH", raising=False)
    _reset_service_caches()

    client = _ASGITestClient(app)

    manifest_resp = client.get("/market-value/model-manifest")
    assert manifest_resp.status_code == 200
    manifest = manifest_resp.json()["payload"]
    assert manifest["source"] == "derived"
    assert manifest["artifacts"]["test_predictions"]["exists"] is True
    assert manifest["artifacts"]["test_predictions"]["sha256"]
    assert manifest["config"]["test_season"] == "2024/25"

    active_resp = client.get("/market-value/active-artifacts")
    assert active_resp.status_code == 200
    active = active_resp.json()["payload"]
    assert active["test_predictions_path"] == str(test_path)
    assert active["val_predictions_path"] == str(val_path)
    assert active["metrics_path"] == str(metrics_path)
    assert active["test_predictions_sha256"]
    assert active["val_predictions_sha256"]
    assert active["metrics_sha256"]

    scout_resp = client.get(
        "/market-value/scout-targets",
        params={
            "split": "test",
            "top_n": 10,
            "max_age": 23,
            "min_minutes": 900,
            "min_confidence": 0.5,
            "min_value_gap_eur": 1_500_000,
            "non_big5_only": "true",
        },
    )
    assert scout_resp.status_code == 200
    payload = scout_resp.json()
    assert payload["count"] == 2

    ids = {row["player_id"] for row in payload["items"]}
    assert "ered_fw_1" in ids
    assert "liga_mf_1" in ids
    assert "epl_fw_1" not in ids
    assert "low_min_1" not in ids


def test_dual_champion_manifest_uses_valuation_base_with_future_overlay(tmp_path: Path, monkeypatch) -> None:
    artifacts = _write_dual_role_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(artifacts["future_test"]))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(artifacts["future_val"]))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(artifacts["future_metrics"]))
    monkeypatch.setenv("SCOUTING_MODEL_MANIFEST_PATH", str(artifacts["manifest_out"]))
    _reset_service_caches()

    merged = mvs._prepare_predictions_frame(mvs.get_predictions("test"))  # type: ignore[attr-defined]
    row = merged[merged["player_id"] == "dual_fw_1"].iloc[0]
    assert float(row["fair_value_eur"]) == 13_500_000.0
    assert float(row["future_scout_blend_score"]) == 0.93
    assert float(row["future_potential_score"]) > 0.0
    assert float(row["current_level_score"]) > 0.0
    assert row["talent_position_family"] in {"ST", "W"}
    assert isinstance(row["score_families"], dict)
    assert isinstance(row["score_explanations"], dict)

    metrics = mvs.get_metrics()
    assert metrics["trials_per_position"] == 60

    client = _ASGITestClient(app)
    manifest_resp = client.get("/market-value/model-manifest")
    assert manifest_resp.status_code == 200
    manifest = manifest_resp.json()["payload"]
    assert manifest["valuation_champion"]["artifacts"]["metrics"]["path"] == str(artifacts["valuation_metrics"])
    assert manifest["future_shortlist_champion"]["artifacts"]["metrics"]["path"] == str(artifacts["future_metrics"])

    scout_resp = client.get(
        "/market-value/scout-targets",
        params={
            "split": "test",
            "top_n": 5,
            "min_minutes": 900,
            "min_confidence": 0.5,
            "min_value_gap_eur": 1_000_000,
            "max_age": 23,
            "non_big5_only": "true",
        },
    )
    assert scout_resp.status_code == 200
    assert scout_resp.json()["diagnostics"]["score_column"] == "future_scout_blend_score"
    shortlist_item = scout_resp.json()["items"][0]
    assert "current_level_score" in shortlist_item
    assert "future_potential_score" in shortlist_item
    assert "future_potential_confidence" in shortlist_item
    assert "score_families" in shortlist_item
    assert "score_explanations" in shortlist_item


def test_watchlist_add_list_delete(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    watchlist_path = tmp_path / "watchlist.jsonl"
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_WATCHLIST_PATH", str(watchlist_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    add_resp = client.post(
        "/market-value/watchlist/items",
        json={
            "player_id": "ered_fw_1",
            "split": "test",
            "tag": "u23",
            "notes": "priority scout target",
        },
    )
    assert add_resp.status_code == 200
    item = add_resp.json()["item"]
    assert item["player_id"] == "ered_fw_1"
    assert item["watch_id"]
    assert item["tag"] == "u23"

    list_resp = client.get("/market-value/watchlist", params={"tag": "u23"})
    assert list_resp.status_code == 200
    payload = list_resp.json()
    assert payload["count"] == 1
    assert payload["items"][0]["player_id"] == "ered_fw_1"

    del_resp = client.delete(f"/market-value/watchlist/items/{item['watch_id']}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    list_after = client.get("/market-value/watchlist")
    assert list_after.status_code == 200
    assert list_after.json()["count"] == 0


def test_watchlist_empty_list_and_missing_delete_are_safe(tmp_path: Path, monkeypatch) -> None:
    watchlist_path = tmp_path / "watchlist.jsonl"
    monkeypatch.setenv("SCOUTING_WATCHLIST_PATH", str(watchlist_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    list_resp = client.get("/market-value/watchlist")
    assert list_resp.status_code == 200
    payload = list_resp.json()
    assert payload["path"] == str(watchlist_path)
    assert payload["total"] == 0
    assert payload["count"] == 0
    assert payload["items"] == []

    del_resp = client.delete("/market-value/watchlist/items/missing-watch-id")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is False


def test_watchlist_duplicate_add_updates_existing_entry(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    watchlist_path = tmp_path / "watchlist.jsonl"
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_WATCHLIST_PATH", str(watchlist_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    first_resp = client.post(
        "/market-value/watchlist/items",
        json={
            "player_id": "ered_fw_1",
            "split": "test",
            "tag": "u23",
            "notes": "first pass",
            "source": "manual",
        },
    )
    assert first_resp.status_code == 200
    first_item = first_resp.json()["item"]

    second_resp = client.post(
        "/market-value/watchlist/items",
        json={
            "player_id": "ered_fw_1",
            "split": "test",
            "tag": "u23",
            "notes": "updated notes",
            "source": "frontend_workbench",
        },
    )
    assert second_resp.status_code == 200
    second_item = second_resp.json()["item"]

    assert second_item["watch_id"] == first_item["watch_id"]
    assert second_item["notes"] == "updated notes"
    assert second_item["source"] == "frontend_workbench"

    list_resp = client.get("/market-value/watchlist", params={"tag": "u23"})
    assert list_resp.status_code == 200
    payload = list_resp.json()
    assert payload["count"] == 1
    assert payload["items"][0]["watch_id"] == first_item["watch_id"]
    assert payload["items"][0]["notes"] == "updated notes"

    lines = [line for line in watchlist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1


def test_watchlist_recovers_from_corrupt_file_and_backs_up_before_rewrite(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    watchlist_path = tmp_path / "watchlist.jsonl"
    watchlist_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "watch_id": "existing-watch",
                        "created_at_utc": "2026-01-01T00:00:00+00:00",
                        "player_id": "ered_fw_1",
                        "split": "test",
                        "season": "2024/25",
                        "tag": "u23",
                        "notes": "existing",
                    }
                ),
                "{bad json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_WATCHLIST_PATH", str(watchlist_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    list_resp = client.get("/market-value/watchlist")
    assert list_resp.status_code == 200
    assert list_resp.json()["count"] == 1
    assert list_resp.json()["items"][0]["watch_id"] == "existing-watch"

    add_resp = client.post(
        "/market-value/watchlist/items",
        json={
            "player_id": "liga_mf_1",
            "split": "test",
            "tag": "u23",
            "notes": "new item",
        },
    )
    assert add_resp.status_code == 200

    lines = [line for line in watchlist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    backups = list(tmp_path.glob("watchlist.jsonl.bak.*"))
    assert backups


def test_predictions_default_sort_uses_capped_gap(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get("/market-value/predictions", params={"split": "test", "limit": 2})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["sort_by"] == "value_gap_capped_eur"
    assert payload["count"] == 2
    assert "value_gap_capped_eur" in payload["items"][0]


def test_predictions_endpoint_exposes_board_row_fields(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_recruitment_filter_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get("/market-value/predictions", params={"split": "test", "limit": 1})
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert {
        "player_id",
        "name",
        "club",
        "league",
        "season",
        "model_position",
        "market_value_eur",
        "value_gap_conservative_eur",
        "value_gap_capped_eur",
        "undervaluation_confidence",
    } <= set(item)


def test_predictions_endpoint_supports_limited_columns_for_coverage_fetch(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/predictions",
        params={
            "split": "test",
            "limit": 5,
            "columns": "season,league,undervalued_flag,undervaluation_confidence,value_gap_conservative_eur",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] >= 1
    for item in payload["items"]:
        assert set(item) <= {
            "season",
            "league",
            "undervalued_flag",
            "undervaluation_confidence",
            "value_gap_conservative_eur",
        }
        assert {"season", "league", "undervaluation_confidence", "value_gap_conservative_eur"} <= set(item)


def test_predictions_endpoint_supports_recruitment_filters(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_recruitment_filter_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/predictions",
        params={
            "split": "test",
            "limit": 20,
            "role_keys": "W",
            "min_age": 20,
            "max_age": 22,
            "max_market_value_eur": 7_000_000,
            "max_contract_years_left": 2.0,
            "non_big5_only": "true",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert [item["player_id"] for item in payload["items"]] == ["wing_target"]


def test_shortlist_and_scout_targets_support_role_budget_and_contract_filters(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_recruitment_filter_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)

    shortlist = client.get(
        "/market-value/shortlist",
        params={
            "split": "test",
            "top_n": 10,
            "min_age": 19,
            "max_age": 21,
            "role_keys": "CB",
            "max_market_value_eur": 5_000_000,
            "max_contract_years_left": 1.0,
            "non_big5_only": "true",
        },
    )
    assert shortlist.status_code == 200
    shortlist_payload = shortlist.json()
    assert [item["player_id"] for item in shortlist_payload["items"]] == ["cb_target"]

    scout_targets = client.get(
        "/market-value/scout-targets",
        params={
            "split": "test",
            "top_n": 10,
            "min_age": 20,
            "max_age": 22,
            "role_keys": "W",
            "max_market_value_eur": 7_000_000,
            "max_contract_years_left": 2.0,
            "non_big5_only": "true",
        },
    )
    assert scout_targets.status_code == 200
    scout_payload = scout_targets.json()
    assert [item["player_id"] for item in scout_payload["items"]] == ["wing_target"]


def test_shortlist_endpoint_exposes_diagnostics_and_active_score_field(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_recruitment_filter_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/shortlist",
        params={"split": "test", "top_n": 5, "min_minutes": 900, "max_age": 23},
    )
    assert resp.status_code == 200
    payload = resp.json()
    diagnostics = payload["diagnostics"]
    assert {"score_column", "ranking_basis", "precision_at_k"} <= set(diagnostics)
    assert isinstance(diagnostics["precision_at_k"].get("rows"), list)
    assert payload["count"] >= 1
    active_score = diagnostics["score_column"]
    assert active_score
    assert active_score in payload["items"][0]


def test_scout_targets_empty_state_retains_diagnostics_shape(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_recruitment_filter_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/scout-targets",
        params={
            "split": "test",
            "top_n": 10,
            "min_minutes": 900,
            "min_confidence": 0.5,
            "min_value_gap_eur": 25_000_000,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 0
    assert payload["items"] == []
    diagnostics = payload["diagnostics"]
    assert {"score_column", "ranking_basis", "precision_at_k"} <= set(diagnostics)
    assert isinstance(diagnostics["precision_at_k"].get("rows"), list)


def test_system_fit_templates_endpoint_returns_two_named_systems() -> None:
    client = _ASGITestClient(app)
    resp = client.get("/market-value/system-fit/templates")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["default_template_key"] == "high_press_433"
    assert [template["template_key"] for template in payload["templates"]] == [
        "high_press_433",
        "transition_4231",
    ]
    assert payload["templates"][0]["slots"][0]["slot_key"] == "GK"


def test_system_fit_query_returns_slot_lists_and_respects_default_trust_scope(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_system_fit_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(
        mvs,
        "load_ingestion_health_payload",
        lambda clean_dataset_path=None: {
            "rows": [
                {"league_name": "Eredivisie", "season": "2024/25", "status": "healthy"},
                {"league_name": "Primeira Liga", "season": "2024/25", "status": "watch"},
                {"league_name": "Belgian Pro League", "season": "2024/25", "status": "blocked"},
            ]
        },
    )
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.post(
        "/market-value/system-fit/query",
        json={
            "template_key": "high_press_433",
            "split": "test",
            "active_lane": "future_shortlist",
            "top_n_per_slot": 5,
            "trust_scope": "trusted_and_watch",
            "slot_role_overrides": {"RW": "transition_winger"},
            "filters": {
                "budget_eur": 5_000_000,
                "min_minutes": 1200,
                "max_age": 24,
                "non_big5_only": True,
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["active_lane"] == "future_shortlist"
    assert payload["lane_posture"]["lane_state"] == "live"
    rw_slot = next(slot for slot in payload["slots"] if slot["slot_key"] == "RW")
    assert rw_slot["role_template_key"] == "transition_winger"
    assert rw_slot["items"]
    assert all(item["league_trust_tier"] in {"trusted", "watch"} for item in rw_slot["items"])
    assert "sf_rw_blocked" not in {item["player_id"] for item in rw_slot["items"]}
    top_item = rw_slot["items"][0]
    assert top_item["budget_status"] == "within_budget"
    assert top_item["active_lane_score"] == pytest.approx(round(top_item["future_potential_score"], 2))
    assert isinstance(top_item["fit_reasons"], list) and top_item["fit_reasons"]


def test_system_fit_query_supports_trust_scope_all_and_valuation_lane(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_system_fit_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(
        mvs,
        "load_ingestion_health_payload",
        lambda clean_dataset_path=None: {
            "rows": [
                {"league_name": "Eredivisie", "season": "2024/25", "status": "healthy"},
                {"league_name": "Primeira Liga", "season": "2024/25", "status": "watch"},
                {"league_name": "Belgian Pro League", "season": "2024/25", "status": "blocked"},
            ]
        },
    )
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.post(
        "/market-value/system-fit/query",
        json={
            "template_key": "transition_4231",
            "split": "test",
            "active_lane": "valuation",
            "top_n_per_slot": 5,
            "trust_scope": "all",
            "filters": {
                "include_leagues": ["Eredivisie", "Belgian Pro League"],
                "budget_eur": 6_000_000,
                "min_confidence": 40,
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    rw_slot = next(slot for slot in payload["slots"] if slot["slot_key"] == "RW")
    ids = {item["player_id"] for item in rw_slot["items"]}
    assert "sf_rw_good" in ids
    assert "sf_rw_blocked" in ids
    blocked_item = next(item for item in rw_slot["items"] if item["player_id"] == "sf_rw_blocked")
    assert blocked_item["league_trust_tier"] == "blocked"
    assert blocked_item["budget_status"] == "stretch"
    assert blocked_item["active_lane_score"] == pytest.approx(round(blocked_item["current_level_score"], 2))


def test_predictions_endpoint_applies_league_adjusted_pricing_by_default(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_league_adjustment_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(
        mvs,
        "load_ingestion_health_payload",
        lambda clean_dataset_path=None: {
            "rows": [
                {"league_name": "Estonian Meistriliiga", "season": "2024/25", "status": "watch"},
                {"league_name": "Czech Fortuna Liga", "season": "2024/25", "status": "healthy"},
                {"league_name": "Eredivisie", "season": "2024/25", "status": "healthy"},
            ]
        },
    )
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/predictions",
        params={"split": "test", "limit": 20, "sort_by": "player_id", "sort_order": "asc"},
    )
    assert resp.status_code == 200
    by_id = {item["player_id"]: item for item in resp.json()["items"]}

    est = by_id["est_winger"]
    assert est["league_adjustment_bucket"] == "severe_failed"
    assert est["league_adjustment_alpha"] <= 0.25
    assert est["fair_value_eur"] == pytest.approx(est["league_adjusted_fair_value_eur"])
    assert est["raw_fair_value_eur"] > est["fair_value_eur"]
    assert est["raw_value_gap_conservative_eur"] > est["value_gap_conservative_eur"]

    czech = by_id["cze_cm"]
    assert czech["league_adjustment_bucket"] == "failed"
    assert czech["league_adjustment_alpha"] <= 0.45

    dutch = by_id["ned_winger"]
    assert dutch["league_adjustment_bucket"] == "standard"
    assert dutch["league_adjustment_alpha"] > czech["league_adjustment_alpha"]

    finland = by_id["fin_striker"]
    assert finland["league_adjustment_bucket"] == "unknown"
    assert finland["league_adjustment_alpha"] <= 0.60


def test_predictions_downrank_failed_leagues_without_hiding_them(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_league_adjustment_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(
        mvs,
        "load_ingestion_health_payload",
        lambda clean_dataset_path=None: {
            "rows": [
                {"league_name": "Estonian Meistriliiga", "season": "2024/25", "status": "watch"},
                {"league_name": "Czech Fortuna Liga", "season": "2024/25", "status": "healthy"},
                {"league_name": "Eredivisie", "season": "2024/25", "status": "healthy"},
            ]
        },
    )
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/predictions",
        params={"split": "test", "limit": 20, "sort_by": "value_gap_conservative_eur", "sort_order": "desc"},
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    ordered_ids = [item["player_id"] for item in items]
    assert "est_winger" in ordered_ids
    assert ordered_ids.index("ned_winger") < ordered_ids.index("est_winger")
    est = next(item for item in items if item["player_id"] == "est_winger")
    ned = next(item for item in items if item["player_id"] == "ned_winger")
    assert float(est["discovery_reliability_weight"]) < float(ned["discovery_reliability_weight"])


def test_player_report_exposes_raw_vs_adjusted_valuation_guardrails(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_league_adjustment_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(
        mvs,
        "load_ingestion_health_payload",
        lambda clean_dataset_path=None: {
            "rows": [
                {"league_name": "Estonian Meistriliiga", "season": "2024/25", "status": "watch"},
                {"league_name": "Eredivisie", "season": "2024/25", "status": "healthy"},
            ]
        },
    )
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/player/est_winger/report",
        params={"split": "test", "top_metrics": 4},
    )
    assert resp.status_code == 200
    report = resp.json()["report"]
    guardrails = report["valuation_guardrails"]

    assert guardrails["league_adjustment_bucket"] == "severe_failed"
    assert guardrails["raw_fair_value_eur"] > guardrails["fair_value_eur"]
    assert guardrails["league_adjusted_fair_value_eur"] == pytest.approx(guardrails["fair_value_eur"])
    assert guardrails["raw_value_gap_conservative_eur"] > guardrails["value_gap_conservative_eur"]
    assert any(flag["code"] == "league_pricing_adjusted" for flag in report["risk_flags"])
    assert "league-adjusted fair value" in report["summary_text"].lower()


def test_system_fit_query_carries_league_adjustment_warning_reasons(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_league_adjustment_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(
        mvs,
        "load_ingestion_health_payload",
        lambda clean_dataset_path=None: {
            "rows": [
                {"league_name": "Estonian Meistriliiga", "season": "2024/25", "status": "watch"},
                {"league_name": "Eredivisie", "season": "2024/25", "status": "healthy"},
            ]
        },
    )
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.post(
        "/market-value/system-fit/query",
        json={
            "template_key": "high_press_433",
            "split": "test",
            "active_lane": "valuation",
            "top_n_per_slot": 5,
            "trust_scope": "all",
            "filters": {
                "include_leagues": ["Estonian Meistriliiga", "Eredivisie"],
                "budget_eur": 3_000_000,
                "min_minutes": 1500,
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    rw_slot = next(slot for slot in payload["slots"] if slot["slot_key"] == "RW")
    est_item = next(item for item in rw_slot["items"] if item["player_id"] == "est_winger")
    assert est_item["fair_value_eur"] == pytest.approx(est_item["league_adjusted_fair_value_eur"])
    assert est_item["league_adjustment_bucket"] == "severe_failed"
    assert any("Pricing" in reason for reason in est_item["fit_reasons"])


def test_system_fit_soft_penalizes_failed_leagues_in_slot_order(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_league_adjustment_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(
        mvs,
        "load_ingestion_health_payload",
        lambda clean_dataset_path=None: {
            "rows": [
                {"league_name": "Estonian Meistriliiga", "season": "2024/25", "status": "watch"},
                {"league_name": "Eredivisie", "season": "2024/25", "status": "healthy"},
            ]
        },
    )
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.post(
        "/market-value/system-fit/query",
        json={
            "template_key": "high_press_433",
            "split": "test",
            "active_lane": "valuation",
            "top_n_per_slot": 5,
            "trust_scope": "all",
            "filters": {"include_leagues": ["Estonian Meistriliiga", "Eredivisie"], "min_minutes": 1500},
        },
    )
    assert resp.status_code == 200
    rw_slot = next(slot for slot in resp.json()["slots"] if slot["slot_key"] == "RW")
    ordered_ids = [item["player_id"] for item in rw_slot["items"]]
    assert ordered_ids.index("ned_winger") < ordered_ids.index("est_winger")


def test_validate_strict_artifact_env_requires_env_vars(monkeypatch) -> None:
    monkeypatch.delenv("SCOUTING_TEST_PREDICTIONS_PATH", raising=False)
    monkeypatch.delenv("SCOUTING_VAL_PREDICTIONS_PATH", raising=False)
    monkeypatch.delenv("SCOUTING_METRICS_PATH", raising=False)
    with pytest.raises(RuntimeError, match="required env vars are missing"):
        mvs.validate_strict_artifact_env()


def test_market_value_health_returns_503_payload_when_strict_artifacts_missing(tmp_path: Path, monkeypatch) -> None:
    missing_test = tmp_path / "missing_test.csv"
    missing_val = tmp_path / "missing_val.csv"
    missing_metrics = tmp_path / "missing_metrics.json"

    monkeypatch.setenv("SCOUTING_STRICT_ARTIFACTS", "1")
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(missing_test))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(missing_val))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(missing_metrics))
    _reset_service_caches()

    client = _ASGITestClient(app)
    health_resp = client.get("/market-value/health")
    assert health_resp.status_code == 503
    payload = health_resp.json()
    assert payload["status"] == "error"
    assert payload["strict_artifacts"] is True
    assert "artifact files do not exist" in payload["strict_artifacts_error"]
    assert payload["artifacts"]["test_predictions_exists"] is False
    assert payload["artifacts"]["val_predictions_exists"] is False
    assert payload["artifacts"]["metrics_exists"] is False

    metrics_resp = client.get("/market-value/metrics")
    assert metrics_resp.status_code == 503
    metrics_payload = metrics_resp.json()
    assert metrics_payload["status"] == "error"
    assert "Market-value artifacts are not ready" in metrics_payload["detail"]


def test_frontend_bootstrap_endpoints_expose_expected_payload_keys(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps({"summary": {"label": "tmp benchmark"}, "segments": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_BENCHMARK_REPORT_PATH", str(benchmark_path))
    _reset_service_caches()

    client = _ASGITestClient(app)

    health_resp = client.get("/market-value/health")
    assert health_resp.status_code == 200
    health_payload = health_resp.json()
    assert health_payload["status"] == "ok"
    assert {"artifacts", "metrics_loaded", "test_rows", "val_rows"} <= set(health_payload)

    metrics_resp = client.get("/market-value/metrics")
    assert metrics_resp.status_code == 200
    metrics_payload = metrics_resp.json()["payload"]
    assert {"dataset", "test_season", "val_season"} <= set(metrics_payload)

    manifest_resp = client.get("/market-value/model-manifest")
    assert manifest_resp.status_code == 200
    manifest_payload = manifest_resp.json()["payload"]
    assert manifest_payload["artifacts"]["test_predictions"]["path"] == str(test_path)
    assert manifest_payload["artifacts"]["metrics"]["path"] == str(metrics_path)

    active_resp = client.get("/market-value/active-artifacts")
    assert active_resp.status_code == 200
    active_payload = active_resp.json()["payload"]
    assert active_payload["test_predictions_path"] == str(test_path)
    assert "valuation" in active_payload
    assert "future_shortlist" in active_payload

    benchmark_resp = client.get("/market-value/benchmarks")
    assert benchmark_resp.status_code == 200
    assert benchmark_resp.json()["payload"]["summary"]["label"] == "tmp benchmark"


def test_player_report_endpoint_returns_profile(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/player/profile_fw_target/report",
        params={"split": "test", "top_metrics": 4},
    )
    assert resp.status_code == 200
    payload = resp.json()
    report = payload["report"]

    assert payload["split"] == "test"
    assert report["player"]["player_id"] == "profile_fw_target"
    assert report["cohort"]["size"] >= 35
    assert isinstance(report["strengths"], list)
    assert isinstance(report["weaknesses"], list)
    assert isinstance(report["development_levers"], list)
    assert len(report["strengths"]) >= 1
    assert len(report["development_levers"]) >= 1
    assert report["confidence"]["label"] in {"low", "medium", "high"}
    assert isinstance(report["summary_text"], str)
    assert report["summary_text"]
    assert "valuation_guardrails" in report
    assert report["valuation_guardrails"]["cap_applied"] is True
    assert report["valuation_guardrails"]["value_gap_capped_eur"] is not None
    assert report["valuation_guardrails"]["market_value_eur"] is not None
    assert report["valuation_guardrails"]["fair_value_eur"] is not None
    assert "player_type" in report
    assert "formation_fit" in report
    assert "radar_profile" in report
    assert isinstance(report["strengths"][0].get("label"), str)
    assert report["strengths"][0]["label"]
    assert isinstance(report["weaknesses"][0].get("label"), str)
    assert isinstance(report["development_levers"][0].get("label"), str)
    assert isinstance(report["risk_flags"], list)
    assert report["risk_flags"]
    assert {"severity", "code", "message"} <= set(report["risk_flags"][0])
    assert isinstance(report["player_type"].get("archetype"), str)
    assert report["player_type"].get("archetype")
    assert report["player_type"].get("position_key") == "W"
    assert isinstance(report["player_type"].get("summary_text"), str)
    assert report["formation_fit"].get("position_key") == "W"
    assert isinstance(report["formation_fit"].get("summary_text"), str)
    assert report["radar_profile"].get("position_key") == "W"

    recommended = report["formation_fit"].get("recommended")
    assert isinstance(recommended, list)
    assert len(recommended) >= 1
    top_fit = recommended[0]
    assert isinstance(top_fit.get("formation"), str)
    assert top_fit.get("formation")
    assert isinstance(top_fit.get("role"), str)
    assert top_fit.get("role")
    assert 0.0 <= float(top_fit.get("fit_score_0_to_1", 0.0)) <= 1.0

    radar_axes = report["radar_profile"].get("axes")
    assert isinstance(radar_axes, list)
    assert len(radar_axes) >= 4
    available_axes = [a for a in radar_axes if bool(a.get("available"))]
    assert len(available_axes) >= 3
    for axis in available_axes:
        norm = axis.get("normalized_0_to_100")
        assert norm is not None
        assert 0.0 <= float(norm) <= 100.0


def test_player_advanced_profile_endpoint_returns_payload(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/player/profile_fw_target/advanced-profile",
        params={"split": "test", "top_metrics": 6},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["split"] == "test"
    profile = payload["profile"]
    assert profile["player"]["player_id"] == "profile_fw_target"
    assert isinstance(profile["player_type"].get("archetype"), str)
    assert profile["player_type"].get("archetype")
    assert profile["player_type"].get("position_key") == "W"

    recommended = profile["formation_fit"].get("recommended")
    assert isinstance(recommended, list)
    assert len(recommended) >= 1

    radar_axes = profile["radar_profile"].get("axes")
    assert isinstance(radar_axes, list)
    assert len(radar_axes) >= 4
    assert any(bool(a.get("available")) for a in radar_axes)
    assert isinstance(profile.get("summary_text"), str)


def test_player_profile_endpoint_returns_grouped_stats_and_similar_players(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(
        mvs,
        "_load_similar_player_matches",
        lambda player_id, top_k, **kwargs: [
            {
                "player_id": "profile_fw_2",
                "score": 0.91,
                "justification": "play as FW; similar age band",
            }
        ],
    )
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/player/profile_fw_target/profile",
        params={"split": "test", "top_metrics": 6, "similar_top_k": 3},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["split"] == "test"

    profile = payload["profile"]
    assert profile["player"]["player_id"] == "profile_fw_target"
    assert isinstance(profile.get("summary_text"), str)
    assert profile["player_type"].get("position_key") == "W"
    assert isinstance(profile.get("history_strength"), dict)
    assert isinstance(profile.get("stat_groups"), list)
    assert len(profile["stat_groups"]) >= 2
    first_group = profile["stat_groups"][0]
    assert "group" in first_group
    assert isinstance(first_group.get("items"), list)
    assert first_group["group"]
    assert "label" in first_group["items"][0]
    assert "display_value" in first_group["items"][0]
    assert "kind" in first_group["items"][0]

    similar = profile.get("similar_players", {})
    assert similar["available"] is True
    assert len(similar["items"]) == 1
    assert similar["items"][0]["player_id"] == "profile_fw_2"
    assert similar["items"][0]["name"] == "Profile FW 2"
    assert profile["proxy_estimates"]["available"] is False
    assert profile["external_tactical_context"]["available"] is True
    assert isinstance(profile["external_tactical_context"]["summary_text"], str)
    assert isinstance(profile["external_tactical_context"]["signals"], list)
    assert profile["external_tactical_context"]["preferred_formations"][0]["formation"] == "433"
    assert profile["availability_context"]["available"] is True
    assert isinstance(profile["availability_context"]["summary_text"], str)
    assert isinstance(profile["availability_context"]["signals"], list)
    assert profile["market_context"]["available"] is True
    assert isinstance(profile["market_context"]["summary_text"], str)
    assert isinstance(profile["market_context"]["signals"], list)
    assert profile["provider_coverage"]["statsbomb"] is True
    assert profile["provider_coverage"]["availability_provider"] is True
    assert profile["provider_coverage"]["market_provider"] is True
    assert profile["provider_coverage"]["providers"]["statsbomb"]["snapshot_date"] == "2026-03-12"
    assert profile["provider_coverage"]["latest_snapshot_date"] == "2026-03-12"
    assert profile["data_freshness"]["status"] == "stable"
    assert profile["data_freshness"]["partial_season"] is False
    assert profile["data_freshness"]["latest_snapshot_date"] == "2026-03-12"
    assert isinstance(profile["talent_view"], dict)
    assert profile["talent_view"]["current_level_score"] is not None
    assert profile["talent_view"]["future_potential_score"] is not None
    assert profile["talent_view"]["talent_position_family"] == "W"
    assert isinstance(profile["talent_view"]["score_families"], dict)
    assert isinstance(profile["talent_view"]["score_explanations"], dict)
    assert isinstance(profile["talent_view"]["current_level_confidence_reasons"], list)
    assert isinstance(profile["talent_view"]["future_potential_confidence_reasons"], list)
    codes = {flag["code"] for flag in profile.get("risk_flags", [])}
    assert "provider_injury_load" in codes
    assert "provider_schedule_congestion" in codes


def test_player_profile_endpoint_handles_missing_similarity_resources(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))

    def _raise_similarity(*args, **kwargs):
        raise RuntimeError("faiss unavailable")

    monkeypatch.setattr(mvs, "_load_similar_player_matches", _raise_similarity)
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/player/profile_fw_target/profile",
        params={"split": "test", "top_metrics": 6, "similar_top_k": 3},
    )
    assert resp.status_code == 200
    similar = resp.json()["profile"]["similar_players"]
    assert similar["available"] is False
    assert similar["items"] == []
    assert "faiss unavailable" in similar["reason"]


def test_player_reports_endpoint_returns_bulk_profiles(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/player-reports",
        params={
            "split": "test",
            "limit": 3,
            "top_metrics": 3,
            "include_history": "true",
            "sort_by": "undervaluation_confidence",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["split"] == "test"
    assert payload["count"] == 3
    assert payload["total"] >= 40
    assert payload["sort_by"] == "undervaluation_confidence"
    assert isinstance(payload["items"], list)

    first = payload["items"][0]
    assert first["player_id"]
    assert "report" in first
    assert "history_strength" in first
    report = first["report"]
    assert isinstance(report.get("summary_text"), str)
    assert isinstance(report.get("strengths"), list)
    assert isinstance(report.get("development_levers"), list)
    assert isinstance(report.get("player_type"), dict)
    assert isinstance(report.get("formation_fit"), dict)
    assert isinstance(report.get("radar_profile"), dict)
    assert report.get("player_type", {}).get("archetype")
    assert report.get("player_type", {}).get("position_key")
    assert len(report.get("formation_fit", {}).get("recommended", [])) >= 1
    assert len(report.get("radar_profile", {}).get("axes", [])) >= 4
    assert isinstance(first["history_strength"].get("summary_text"), str)


def test_infer_position_role_key_supports_subroles() -> None:
    winger = pd.Series({"model_position": "FW", "position_main": "Right Winger", "position_alt": "Attack"})
    striker = pd.Series({"model_position": "FW", "position_main": "Forward", "position_alt": "Attack, Centre"})
    support = pd.Series({"model_position": "FW", "position_main": "Second Striker", "position_alt": "Attack"})
    dm = pd.Series({"model_position": "MF", "position_main": "Defensive Midfield", "position_alt": "Midfield"})
    am = pd.Series({"model_position": "MF", "position_main": "Attacking Midfield", "position_alt": "Midfield"})
    cm = pd.Series({"model_position": "MF", "position_main": "Central Midfield", "position_alt": "Midfield"})
    cb = pd.Series({"model_position": "DF", "position_main": "Back", "position_alt": "Defender, Centre"})
    fb = pd.Series({"model_position": "DF", "position_main": "Right Back", "position_alt": "Defender, Right"})
    gk = pd.Series({"model_position": "GK", "position_main": "Goalkeeper"})

    assert mvs._infer_position_role_key(winger) == "W"
    assert mvs._infer_position_role_key(striker) == "ST"
    assert mvs._infer_position_role_key(support) == "SS"
    assert mvs._infer_position_role_key(dm) == "DM"
    assert mvs._infer_position_role_key(am) == "AM"
    assert mvs._infer_position_role_key(cm) == "CM"
    assert mvs._infer_position_role_key(cb) == "CB"
    assert mvs._infer_position_role_key(fb) == "FB"
    assert mvs._infer_position_role_key(gk) == "GK"


def test_player_history_strength_endpoint_returns_breakdown(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/player/profile_fw_target/history-strength",
        params={"split": "test"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["split"] == "test"

    breakdown = payload["breakdown"]
    assert breakdown["player"]["player_id"] == "profile_fw_target"
    history = breakdown["history_strength"]
    assert 0.0 <= float(history["score_0_to_100"]) <= 100.0
    assert 0.0 <= float(history["coverage_0_to_1"]) <= 1.0
    assert isinstance(history["components"], list)
    assert len(history["components"]) >= 4
    keys = {item["key"] for item in history["components"]}
    assert "history_minutes_component" in keys
    assert "history_injury_component" in keys
    assert isinstance(history["summary_text"], str)
    assert history["summary_text"]


def test_player_history_strength_endpoint_handles_sparse_context(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get("/market-value/player/ered_fw_1/history-strength", params={"split": "test"})
    assert resp.status_code == 200
    history = resp.json()["breakdown"]["history_strength"]
    assert isinstance(history["summary_text"], str)
    assert 0.0 <= float(history["coverage_0_to_1"]) <= 1.0
    assert history["tier"] in {"uncertain", "developing", "strong", "elite"}
    assert isinstance(history["components"], list)


def test_player_similar_endpoint_returns_comparisons(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    clean_path = _write_clean_profile_dataset(tmp_path, test_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_CLEAN_DATASET_PATH", str(clean_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get(
        "/market-value/player/profile_fw_target/similar",
        params={"split": "test", "n": 3, "same_position": "true"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["player_id"] == "profile_fw_target"
    assert payload["position_group"] == "FW"
    assert int(payload["feature_count_used"]) >= 10
    assert isinstance(payload["feature_columns_used"], list)
    comparisons = payload["comparisons"]
    assert len(comparisons) == 3
    first = comparisons[0]
    assert first["player_id"] != "profile_fw_target"
    assert 0.0 <= float(first["similarity_score"]) <= 1.0
    assert first["market_value_eur"] is not None
    assert first["predicted_value"] is not None
    assert isinstance(first["league"], str)


def test_similarity_service_uses_position_specific_feature_sets(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for idx in range(6):
        rows.append(
            {
                "player_id": f"gk_{idx}",
                "season": "2024/25",
                "name": f"GK {idx}",
                "club": "Keepers FC",
                "league": "Eredivisie",
                "model_position": "GK",
                "market_value_eur": 1_000_000 + idx,
                "fair_value_eur": 1_200_000 + idx,
                "sofa_saves_per90": 3.0 + idx * 0.1,
                "sofa_cleanSheets_per90": 0.2 + idx * 0.01,
                "sofa_crossesStoppedPercentage": 10 + idx,
                "sofa_claimedCrosses_per90": 0.4 + idx * 0.02,
                "sb_launches_per90": 8 + idx,
                "sb_throw_distance_avg": 28 + idx,
                "sb_keeper_actions_per90": 1.5 + idx * 0.1,
                "sb_sweeper_actions_per90": 0.8 + idx * 0.1,
                "sofa_accuratePassesPercentage": 72 + idx,
                "sofa_shotsSavedInsideBox_per90": 1.4 + idx * 0.1,
                "current_level_confidence": 60 + idx,
            }
        )
        rows.append(
            {
                "player_id": f"fw_{idx}",
                "season": "2024/25",
                "name": f"FW {idx}",
                "club": "Forwards FC",
                "league": "Eredivisie",
                "model_position": "FW",
                "market_value_eur": 2_000_000 + idx,
                "fair_value_eur": 2_400_000 + idx,
                "sofa_goals_per90": 0.3 + idx * 0.05,
                "sofa_totalShots_per90": 2.0 + idx * 0.2,
                "sofa_assists_per90": 0.1 + idx * 0.03,
                "sofa_expectedGoals_per90": 0.25 + idx * 0.04,
                "sofa_keyPasses_per90": 1.1 + idx * 0.1,
                "sofa_successfulDribbles_per90": 1.4 + idx * 0.2,
                "sb_progressive_carries_per90": 2.2 + idx * 0.15,
                "sb_passes_into_box_per90": 1.0 + idx * 0.08,
                "sb_touch_in_box_per90": 5.0 + idx * 0.3,
                "sb_pressures_per90": 8.0 + idx * 0.4,
                "current_level_confidence": 62 + idx,
            }
        )
    clean_path = tmp_path / "similarity_groups.parquet"
    pd.DataFrame(rows).to_parquet(clean_path, index=False)
    service = similarity_service_module.SimilarityService(dataset_path=clean_path)
    gk_features = set(service._index.position_feature_columns["GK"])
    fw_features = set(service._index.position_feature_columns["FW"])
    assert gk_features
    assert fw_features
    assert gk_features != fw_features
    assert "market_value_eur" not in gk_features
    assert "fair_value_eur" not in fw_features


def test_player_report_exposes_proxy_estimates_for_sparse_profile(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    clean_path = _write_clean_profile_dataset(tmp_path, test_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_CLEAN_DATASET_PATH", str(clean_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get("/market-value/player/profile_fw_sparse/report", params={"split": "test"})
    assert resp.status_code == 200
    proxy_payload = resp.json()["report"]["proxy_estimates"]
    assert proxy_payload["available"] is True
    assert len(proxy_payload["metrics"]) >= 3
    first = proxy_payload["metrics"][0]
    assert first["metric_key"]
    assert first["neighbor_count"] >= 5
    assert float(first["mean_similarity"]) >= 0.60
    assert first["support_label"] in {"strong", "moderate", "weak"}


def test_player_trajectory_endpoint_returns_multi_season_payload(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    clean_path = _write_clean_profile_dataset(tmp_path, test_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_CLEAN_DATASET_PATH", str(clean_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get("/market-value/player/profile_fw_target/trajectory", params={"split": "test"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["player_id"] == "profile_fw_target"
    assert payload["trajectory_label"] == "ascending"
    assert float(payload["projected_next_value"]) > 0.0
    assert isinstance(payload["peak_season"], str)
    assert len(payload["seasons"]) >= 3
    assert payload["seasons"][0]["delta_predicted_value"] is None
    assert payload["seasons"][1]["delta_predicted_value"] is not None


def test_player_memo_pdf_endpoint_streams_pdf(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    clean_path = _write_clean_profile_dataset(tmp_path, test_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_CLEAN_DATASET_PATH", str(clean_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.get("/market-value/player/profile_fw_target/memo.pdf", params={"split": "test"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/pdf")
    assert "attachment;" in resp.headers.get("content-disposition", "")
    assert resp.content.startswith(b"%PDF")


def test_scout_decision_post_and_history_include_latest_decision(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    clean_path = _write_clean_profile_dataset(tmp_path, test_path)
    watchlist_path = tmp_path / "watchlist.jsonl"
    decisions_path = tmp_path / "scout_decisions.jsonl"
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_CLEAN_DATASET_PATH", str(clean_path))
    monkeypatch.setenv("SCOUTING_WATCHLIST_PATH", str(watchlist_path))
    monkeypatch.setenv("SCOUTING_DECISIONS_PATH", str(decisions_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    save_resp = client.post(
        "/market-value/decisions",
        json={
            "player_id": "profile_fw_target",
            "split": "test",
            "season": "2024/25",
            "action": "shortlist",
            "reason_tags": ["price_gap", "trajectory"],
            "note": "Fits the current recruitment brief.",
            "source_surface": "workbench",
            "ranking_context": {"mode": "shortlist", "rank": 4, "discovery_reliability_weight": 0.92},
        },
    )
    assert save_resp.status_code == 200
    saved = save_resp.json()
    assert saved["latest_decision"]["action"] == "shortlist"
    assert saved["watchlist_item"]["decision_action"] == "shortlist"

    history_resp = client.get(
        "/market-value/player/profile_fw_target/decisions",
        params={"split": "test", "season": "2024/25"},
    )
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert history["player_id"] == "profile_fw_target"
    assert history["latest_decision"]["action"] == "shortlist"
    assert len(history["events"]) == 1
    assert history["events"][0]["reason_tags"] == ["price_gap", "trajectory"]

    report_resp = client.get("/market-value/player/profile_fw_target/report", params={"split": "test"})
    profile_resp = client.get("/market-value/player/profile_fw_target/profile", params={"split": "test"})
    assert report_resp.status_code == 200
    assert profile_resp.status_code == 200
    assert report_resp.json()["report"]["latest_decision"]["action"] == "shortlist"
    assert profile_resp.json()["profile"]["latest_decision"]["action"] == "shortlist"


def test_scout_decision_validation_requires_reasons_for_shortlist_and_pass(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_DECISIONS_PATH", str(tmp_path / "scout_decisions.jsonl"))
    _reset_service_caches()

    client = _ASGITestClient(app)
    shortlist_resp = client.post(
        "/market-value/decisions",
        json={"player_id": "profile_fw_target", "split": "test", "action": "shortlist", "reason_tags": []},
    )
    pass_resp = client.post(
        "/market-value/decisions",
        json={"player_id": "profile_fw_target", "split": "test", "action": "pass", "reason_tags": []},
    )
    assert shortlist_resp.status_code == 400
    assert "requires at least one reason tag" in shortlist_resp.json()["detail"]
    assert pass_resp.status_code == 400
    assert "requires at least one reason tag" in pass_resp.json()["detail"]


def test_positive_decisions_auto_sync_watchlist_without_duplication(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    clean_path = _write_clean_profile_dataset(tmp_path, test_path)
    watchlist_path = tmp_path / "watchlist.jsonl"
    decisions_path = tmp_path / "scout_decisions.jsonl"
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_CLEAN_DATASET_PATH", str(clean_path))
    monkeypatch.setenv("SCOUTING_WATCHLIST_PATH", str(watchlist_path))
    monkeypatch.setenv("SCOUTING_DECISIONS_PATH", str(decisions_path))
    _reset_service_caches()

    client = _ASGITestClient(app)
    manual_resp = client.post(
        "/market-value/watchlist/items",
        json={
            "player_id": "profile_fw_target",
            "split": "test",
            "season": "2024/25",
            "tag": "summer_window",
            "notes": "manual row",
            "source": "manual",
        },
    )
    assert manual_resp.status_code == 200

    shortlist_resp = client.post(
        "/market-value/decisions",
        json={
            "player_id": "profile_fw_target",
            "split": "test",
            "season": "2024/25",
            "action": "shortlist",
            "reason_tags": ["system_fit"],
        },
    )
    watch_resp = client.post(
        "/market-value/decisions",
        json={
            "player_id": "profile_fw_target",
            "split": "test",
            "season": "2024/25",
            "action": "watch_live",
            "reason_tags": ["availability"],
        },
    )
    assert shortlist_resp.status_code == 200
    assert watch_resp.status_code == 200

    watchlist_resp = client.get("/market-value/watchlist", params={"split": "test"})
    payload = watchlist_resp.json()
    assert watchlist_resp.status_code == 200
    assert payload["total"] == 1
    item = payload["items"][0]
    assert item["tag"] == "summer_window"
    assert item["decision_action"] == "watch_live"
    assert item["decision_reason_tags"] == ["availability"]


def test_pass_decision_does_not_create_watchlist_row(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_WATCHLIST_PATH", str(tmp_path / "watchlist.jsonl"))
    monkeypatch.setenv("SCOUTING_DECISIONS_PATH", str(tmp_path / "scout_decisions.jsonl"))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.post(
        "/market-value/decisions",
        json={
            "player_id": "profile_fw_target",
            "split": "test",
            "season": "2024/25",
            "action": "pass",
            "reason_tags": ["league_risk"],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["watchlist_item"] is None

    watchlist_resp = client.get("/market-value/watchlist", params={"split": "test"})
    assert watchlist_resp.status_code == 200
    assert watchlist_resp.json()["total"] == 0


def test_scout_decision_store_recovers_from_corrupt_log(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    decisions_path = tmp_path / "scout_decisions.jsonl"
    decisions_path.write_text("{bad json}\n", encoding="utf-8")
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_DECISIONS_PATH", str(decisions_path))
    monkeypatch.setenv("SCOUTING_WATCHLIST_PATH", str(tmp_path / "watchlist.jsonl"))
    _reset_service_caches()

    client = _ASGITestClient(app)
    resp = client.post(
        "/market-value/decisions",
        json={
            "player_id": "profile_fw_target",
            "split": "test",
            "season": "2024/25",
            "action": "request_report",
            "reason_tags": ["availability"],
        },
    )
    assert resp.status_code == 200
    text = decisions_path.read_text(encoding="utf-8")
    assert "request_report" in text
    backups = list(tmp_path.glob("scout_decisions.jsonl.bak.*"))
    assert backups


def test_players_routes_return_503_when_experimental_nlp_is_disabled(monkeypatch) -> None:
    monkeypatch.delenv("SCOUTING_ENABLE_EXPERIMENTAL_NLP_ROUTES", raising=False)
    _reset_service_caches()

    client = _ASGITestClient(app)
    report_resp = client.get("/players/example-player/scouting-report")
    assert report_resp.status_code == 503
    assert "Experimental NLP routes are disabled" in report_resp.json()["detail"]

    similar_resp = client.get("/players/example-player/similar")
    assert similar_resp.status_code == 503
    assert "Experimental NLP routes are disabled" in similar_resp.json()["detail"]


def test_players_report_and_role_routes_map_missing_optional_deps_to_503(monkeypatch) -> None:
    monkeypatch.setenv("SCOUTING_ENABLE_EXPERIMENTAL_NLP_ROUTES", "1")
    _reset_service_caches()

    import scouting_ml.services.scouting_report_service as report_service

    monkeypatch.setattr(
        report_service,
        "get_scouting_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ModuleNotFoundError("transformers missing")),
    )
    fake_role_module = types.ModuleType("scouting_ml.nlp.role_classifier")

    def _raise_role(*_args, **_kwargs):
        raise ModuleNotFoundError("transformers missing")

    fake_role_module.classify_player_role = _raise_role
    monkeypatch.setitem(sys.modules, "scouting_ml.nlp.role_classifier", fake_role_module)

    client = _ASGITestClient(app)
    report_resp = client.get("/players/example-player/scouting-report")
    assert report_resp.status_code == 503
    assert "Experimental NLP dependencies are unavailable" in report_resp.json()["detail"]

    role_resp = client.get("/players/example-player/role")
    assert role_resp.status_code == 503
    assert "Experimental NLP dependencies are unavailable" in role_resp.json()["detail"]
