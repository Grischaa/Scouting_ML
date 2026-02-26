from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scouting_ml.api.main import app
from scouting_ml.services import market_value_service as mvs


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
        rows.append(
            {
                "player_id": f"profile_fw_{i}",
                "name": f"Profile FW {i}",
                "league": "Eredivisie",
                "club": "Profile Club",
                "season": "2024/25",
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
                "injury_days_per_1000_min": float(rng.uniform(0.0, 12.0)),
                "contract_years_left": float(rng.uniform(0.3, 4.2)),
            }
        )

    rows[1].update(
        {
            "player_id": "profile_fw_target",
            "name": "Profile Target",
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


def _reset_service_caches() -> None:
    mvs._PRED_CACHE.clear()  # type: ignore[attr-defined]
    mvs._METRICS_CACHE = None  # type: ignore[attr-defined]
    mvs._RESIDUAL_CALIBRATION_CACHE = None  # type: ignore[attr-defined]


def test_api_root_returns_index() -> None:
    client = TestClient(app)
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

    client = TestClient(app)

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


def test_watchlist_add_list_delete(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    watchlist_path = tmp_path / "watchlist.jsonl"
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_WATCHLIST_PATH", str(watchlist_path))
    _reset_service_caches()

    client = TestClient(app)
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


def test_predictions_default_sort_uses_capped_gap(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_test_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = TestClient(app)
    resp = client.get("/market-value/predictions", params={"split": "test", "limit": 2})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["sort_by"] == "value_gap_capped_eur"
    assert payload["count"] == 2
    assert "value_gap_capped_eur" in payload["items"][0]


def test_validate_strict_artifact_env_requires_env_vars(monkeypatch) -> None:
    monkeypatch.delenv("SCOUTING_TEST_PREDICTIONS_PATH", raising=False)
    monkeypatch.delenv("SCOUTING_VAL_PREDICTIONS_PATH", raising=False)
    monkeypatch.delenv("SCOUTING_METRICS_PATH", raising=False)
    with pytest.raises(RuntimeError, match="required env vars are missing"):
        mvs.validate_strict_artifact_env()


def test_player_report_endpoint_returns_profile(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = TestClient(app)
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
    assert "player_type" in report
    assert "formation_fit" in report
    assert "radar_profile" in report
    assert isinstance(report["player_type"].get("archetype"), str)
    assert isinstance(report["formation_fit"].get("recommended"), list)
    assert isinstance(report["radar_profile"].get("axes"), list)


def test_player_advanced_profile_endpoint_returns_payload(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = TestClient(app)
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
    assert isinstance(profile["formation_fit"].get("recommended"), list)
    assert isinstance(profile["radar_profile"].get("axes"), list)
    assert isinstance(profile.get("summary_text"), str)


def test_player_reports_endpoint_returns_bulk_profiles(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = TestClient(app)
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
    assert isinstance(first["history_strength"].get("summary_text"), str)


def test_player_history_strength_endpoint_returns_breakdown(tmp_path: Path, monkeypatch) -> None:
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    _reset_service_caches()

    client = TestClient(app)
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
