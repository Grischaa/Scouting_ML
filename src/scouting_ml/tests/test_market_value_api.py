from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from scouting_ml.api import main as api_main
from scouting_ml.api.main import app
from scouting_ml.scripts.lock_market_value_artifacts import build_lock_bundle
from scouting_ml.services import market_value_service as mvs


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
                "sb_minutes_in_433": float(rng.integers(0, 1400)),
                "sb_minutes_in_4231": float(rng.integers(0, 900)),
                "avail_reports": float(rng.integers(0, 12)),
                "avail_start_share": float(rng.uniform(0.2, 1.0)),
                "avail_injury_count": float(rng.integers(0, 4)),
                "fixture_matches": float(rng.integers(6, 30)),
                "fixture_mean_rest_days": float(rng.uniform(3.0, 8.0)),
                "fixture_congestion_share": float(rng.uniform(0.0, 0.5)),
                "odds_implied_team_strength": float(rng.uniform(0.2, 0.7)),
                "odds_upset_probability": float(rng.uniform(0.2, 0.8)),
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
        }
    )

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


def _reset_service_caches() -> None:
    mvs._PRED_CACHE.clear()  # type: ignore[attr-defined]
    mvs._METRICS_CACHE = {}  # type: ignore[attr-defined]
    mvs._RESIDUAL_CALIBRATION_CACHE = None  # type: ignore[attr-defined]
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

    merged = mvs.get_predictions("test")
    row = merged[merged["player_id"] == "dual_fw_1"].iloc[0]
    assert float(row["fair_value_eur"]) == 13_500_000.0
    assert float(row["future_scout_blend_score"]) == 0.93

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
        lambda player_id, top_k: [
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
