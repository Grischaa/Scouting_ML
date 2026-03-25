from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from scouting_ml.api import main as api_main
from scouting_ml.api.main import app
from scouting_ml.services import market_value_service as mvs
from scouting_ml.services import proxy_estimate_service as proxy_estimate_service_module
from scouting_ml.services import similarity_service as similarity_service_module
from scouting_ml.services import trajectory_service as trajectory_service_module
from scouting_ml.team import db as team_db_module


class _StatefulASGITestClient:
    def __init__(self, app) -> None:
        self._app = app
        self._cookies = httpx.Cookies()

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def _send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
                cookies=self._cookies,
            ) as client:
                response = await client.request(method, url, **kwargs)
                self._cookies.update(response.cookies)
                return response

        return asyncio.run(_send())

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def patch(self, url: str, **kwargs) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def put(self, url: str, **kwargs) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


def _reset_team_and_market_value_caches() -> None:
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
    team_db_module.reset_team_db_caches()
    api_main._STARTUP_CHECKS_DONE = False  # type: ignore[attr-defined]
    api_main._STARTUP_CHECKS_ERROR = None  # type: ignore[attr-defined]


def _write_team_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    current_rows = [
        {
            "player_id": "team_fw_1",
            "name": "Team Winger One",
            "league": "Eredivisie",
            "club": "Shared FC",
            "season": "2024/25",
            "age": 21,
            "model_position": "FW",
            "position_main": "Right Winger",
            "position_alt": "Attack",
            "minutes": 2200,
            "market_value_eur": 7_000_000.0,
            "fair_value_eur": 11_500_000.0,
            "expected_value_eur": 11_500_000.0,
            "expected_value_low_eur": 9_500_000.0,
            "expected_value_high_eur": 13_000_000.0,
            "value_gap_eur": 4_500_000.0,
            "value_gap_conservative_eur": 3_900_000.0,
            "undervaluation_confidence": 1.04,
            "current_level_score": 77.0,
            "future_potential_score": 86.0,
            "prior_mae_eur": 900_000.0,
            "contract_years_left": 1.2,
            "talent_position_family": "W",
            "leaguectx_league_strength_index": 0.52,
            "sofa_goals_per90": 0.38,
            "sofa_assists_per90": 0.26,
            "sofa_expectedGoals_per90": 0.41,
            "sofa_totalShots_per90": 3.7,
            "sofa_keyPasses_per90": 1.6,
            "sofa_successfulDribbles_per90": 2.8,
            "sofa_accuratePassesPercentage": 78.0,
            "sofa_totalDuelsWonPercentage": 50.0,
            "sb_progressive_passes_per90": 5.7,
            "sb_progressive_carries_per90": 4.1,
            "sb_passes_into_box_per90": 1.6,
            "sb_pressures_per90": 7.4,
        },
        {
            "player_id": "team_fw_2",
            "name": "Team Winger Two",
            "league": "Primeira Liga",
            "club": "Shared FC",
            "season": "2024/25",
            "age": 22,
            "model_position": "FW",
            "position_main": "Left Winger",
            "position_alt": "Attack",
            "minutes": 2050,
            "market_value_eur": 8_500_000.0,
            "fair_value_eur": 10_300_000.0,
            "expected_value_eur": 10_300_000.0,
            "expected_value_low_eur": 8_800_000.0,
            "expected_value_high_eur": 11_900_000.0,
            "value_gap_eur": 1_800_000.0,
            "value_gap_conservative_eur": 1_200_000.0,
            "undervaluation_confidence": 0.88,
            "current_level_score": 74.0,
            "future_potential_score": 80.0,
            "prior_mae_eur": 1_100_000.0,
            "contract_years_left": 2.1,
            "talent_position_family": "W",
            "leaguectx_league_strength_index": 0.47,
            "sofa_goals_per90": 0.31,
            "sofa_assists_per90": 0.18,
            "sofa_expectedGoals_per90": 0.34,
            "sofa_totalShots_per90": 3.0,
            "sofa_keyPasses_per90": 1.3,
            "sofa_successfulDribbles_per90": 2.4,
            "sofa_accuratePassesPercentage": 79.0,
            "sofa_totalDuelsWonPercentage": 49.0,
            "sb_progressive_passes_per90": 5.1,
            "sb_progressive_carries_per90": 3.8,
            "sb_passes_into_box_per90": 1.3,
            "sb_pressures_per90": 7.0,
        },
        {
            "player_id": "team_mf_1",
            "name": "Team Mid One",
            "league": "Belgian Pro League",
            "club": "Shared FC",
            "season": "2024/25",
            "age": 23,
            "model_position": "MF",
            "position_main": "Central Midfielder",
            "position_alt": "Midfield",
            "minutes": 2300,
            "market_value_eur": 6_000_000.0,
            "fair_value_eur": 8_900_000.0,
            "expected_value_eur": 8_900_000.0,
            "expected_value_low_eur": 7_100_000.0,
            "expected_value_high_eur": 10_200_000.0,
            "value_gap_eur": 2_900_000.0,
            "value_gap_conservative_eur": 2_100_000.0,
            "undervaluation_confidence": 0.93,
            "current_level_score": 73.0,
            "future_potential_score": 78.0,
            "prior_mae_eur": 1_000_000.0,
            "contract_years_left": 1.5,
            "talent_position_family": "CM",
            "leaguectx_league_strength_index": 0.43,
            "sofa_goals_per90": 0.12,
            "sofa_assists_per90": 0.17,
            "sofa_expectedGoals_per90": 0.16,
            "sofa_totalShots_per90": 1.5,
            "sofa_keyPasses_per90": 1.9,
            "sofa_successfulDribbles_per90": 1.4,
            "sofa_accuratePassesPercentage": 85.0,
            "sofa_totalDuelsWonPercentage": 56.0,
            "sb_progressive_passes_per90": 7.2,
            "sb_progressive_carries_per90": 2.7,
            "sb_passes_into_box_per90": 1.4,
            "sb_pressures_per90": 8.2,
        },
    ]
    current_df = pd.DataFrame(current_rows)
    history_rows: list[dict[str, object]] = []
    for player_id, season, market, fair, minutes, goals, assists, xg, prog_pass, prog_carry in (
        ("team_fw_1", "2022/23", 3_200_000.0, 5_000_000.0, 1600, 9.0, 5.0, 7.4, 82.0, 60.0),
        ("team_fw_1", "2023/24", 5_200_000.0, 8_100_000.0, 1900, 11.0, 7.0, 9.3, 95.0, 70.0),
        ("team_fw_2", "2023/24", 6_800_000.0, 8_400_000.0, 1850, 8.0, 5.0, 7.0, 88.0, 61.0),
    ):
        base = dict(next(row for row in current_rows if row["player_id"] == player_id))
        base["season"] = season
        base["market_value_eur"] = market
        base["fair_value_eur"] = fair
        base["expected_value_eur"] = fair
        base["minutes"] = minutes
        base["goals"] = goals
        base["assists"] = assists
        base["xg"] = xg
        base["progressive_passes"] = prog_pass
        base["progressive_carries"] = prog_carry
        history_rows.append(base)

    clean_df = pd.concat([current_df, pd.DataFrame(history_rows)], ignore_index=True)
    clean_df["goals"] = clean_df.get("goals", clean_df["sofa_goals_per90"] * 30)
    clean_df["assists"] = clean_df.get("assists", clean_df["sofa_assists_per90"] * 25)
    clean_df["xg"] = clean_df.get("xg", clean_df["sofa_expectedGoals_per90"] * 25)
    clean_df["progressive_passes"] = clean_df.get("progressive_passes", clean_df["sb_progressive_passes_per90"] * 20)
    clean_df["progressive_carries"] = clean_df.get("progressive_carries", clean_df["sb_progressive_carries_per90"] * 20)

    test_path = tmp_path / "team_predictions_test.csv"
    val_path = tmp_path / "team_predictions_val.csv"
    metrics_path = tmp_path / "team_metrics.json"
    clean_path = tmp_path / "champion_players_clean.parquet"
    current_df.to_csv(test_path, index=False)
    current_df.to_csv(val_path, index=False)
    clean_df.to_parquet(clean_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": "tmp_team_dataset",
                "val_season": "2023/24",
                "test_season": "2024/25",
                "overall": {"test": {"r2": 0.71, "mae_eur": 3_200_000}},
            }
        ),
        encoding="utf-8",
    )
    return test_path, val_path, metrics_path, clean_path


@pytest.fixture()
def configured_team_env(tmp_path: Path, monkeypatch):
    test_path, val_path, metrics_path, clean_path = _write_team_artifacts(tmp_path)
    db_path = tmp_path / "team.sqlite3"
    monkeypatch.setenv("SCOUTING_TEST_PREDICTIONS_PATH", str(test_path))
    monkeypatch.setenv("SCOUTING_VAL_PREDICTIONS_PATH", str(val_path))
    monkeypatch.setenv("SCOUTING_METRICS_PATH", str(metrics_path))
    monkeypatch.setenv("SCOUTING_CLEAN_DATASET_PATH", str(clean_path))
    monkeypatch.setenv("SCOUTING_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SCOUTING_TEAM_MODE", "1")
    monkeypatch.setenv("SCOUTING_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("SCOUTING_SESSION_COOKIE_NAME", "scoutml_team_test")
    monkeypatch.delenv("SCOUTING_MODEL_MANIFEST_PATH", raising=False)
    _reset_team_and_market_value_caches()
    yield
    _reset_team_and_market_value_caches()


def _workspace_headers(workspace_id: str) -> dict[str, str]:
    return {"X-ScoutML-Workspace": workspace_id}


def _bootstrap_admin(client: _StatefulASGITestClient) -> tuple[str, str]:
    response = client.post(
        "/auth/bootstrap-admin",
        json={
            "email": "admin@example.com",
            "password": "admin-pass-123",
            "full_name": "Admin Scout",
            "workspace_name": "Shared Recruitment",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    workspace = payload["active_workspace"]
    return workspace["workspace_id"], payload["user"]["user_id"]


def test_auth_me_reports_local_mode_when_database_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("SCOUTING_DATABASE_URL", raising=False)
    monkeypatch.delenv("SCOUTING_TEAM_MODE", raising=False)
    _reset_team_and_market_value_caches()

    client = _StatefulASGITestClient(app)
    response = client.get("/auth/me")
    assert response.status_code == 200
    payload = response.json()
    assert payload["team_mode"] is False
    assert payload["authenticated"] is False


def test_team_auth_bootstrap_logout_and_workspace_isolation(configured_team_env) -> None:
    admin_client = _StatefulASGITestClient(app)
    workspace_id, _ = _bootstrap_admin(admin_client)

    auth_me = admin_client.get("/auth/me", headers=_workspace_headers(workspace_id))
    assert auth_me.status_code == 200
    assert auth_me.json()["authenticated"] is True

    invite = admin_client.post(
        f"/workspaces/{workspace_id}/invites",
        headers=_workspace_headers(workspace_id),
        json={"email": "scout@example.com", "role": "scout"},
    )
    assert invite.status_code == 200
    token = invite.json()["token"]

    extra_workspace = admin_client.post("/workspaces", headers=_workspace_headers(workspace_id), json={"name": "Secret Workspace"})
    assert extra_workspace.status_code == 200
    second_workspace_id = extra_workspace.json()["active_workspace"]["workspace_id"]

    scout_client = _StatefulASGITestClient(app)
    accepted = scout_client.post(
        f"/invites/{token}/accept",
        json={
            "email": "scout@example.com",
            "password": "scout-pass-123",
            "full_name": "Scout Two",
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["authenticated"] is True

    forbidden = scout_client.get("/team/watchlist", headers=_workspace_headers(second_workspace_id))
    assert forbidden.status_code == 403

    logout = admin_client.post("/auth/logout")
    assert logout.status_code == 200
    logged_out_state = admin_client.get("/auth/me")
    assert logged_out_state.status_code == 200
    assert logged_out_state.json()["authenticated"] is False


def test_team_decisions_sync_watchlist_and_override_profile_latest_decision(configured_team_env) -> None:
    client = _StatefulASGITestClient(app)
    workspace_id, _ = _bootstrap_admin(client)
    headers = _workspace_headers(workspace_id)

    saved = client.post(
        "/team/decisions",
        headers=headers,
        json={
            "player_id": "team_fw_1",
            "split": "test",
            "season": "2024/25",
            "action": "shortlist",
            "reason_tags": ["price_gap"],
            "note": "Fits the current market window.",
            "source_surface": "detail",
            "ranking_context": {"mode": "shortlist", "rank": 1},
        },
    )
    assert saved.status_code == 200
    saved_payload = saved.json()
    assert saved_payload["latest_decision"]["action"] == "shortlist"
    assert saved_payload["watchlist_item"]["decision_action"] == "shortlist"

    updated = client.post(
        "/team/decisions",
        headers=headers,
        json={
            "player_id": "team_fw_1",
            "split": "test",
            "season": "2024/25",
            "action": "request_report",
            "reason_tags": [],
            "note": "Need live confirmation from the network.",
            "source_surface": "detail",
            "ranking_context": {"mode": "shortlist", "rank": 1},
        },
    )
    assert updated.status_code == 200
    assert updated.json()["latest_decision"]["action"] == "request_report"

    watchlist = client.get("/team/watchlist", headers=headers)
    assert watchlist.status_code == 200
    assert watchlist.json()["total"] == 1
    assert watchlist.json()["items"][0]["decision_action"] == "request_report"

    decision_history = client.get("/team/player/team_fw_1/decisions", headers=headers, params={"split": "test", "season": "2024/25"})
    assert decision_history.status_code == 200
    history_payload = decision_history.json()
    assert history_payload["latest_decision"]["action"] == "request_report"
    assert len(history_payload["events"]) == 2

    profile = client.get(
        "/market-value/player/team_fw_1/profile",
        headers=headers,
        params={"split": "test", "season": "2024/25"},
    )
    assert profile.status_code == 200
    assert profile.json()["profile"]["latest_decision"]["action"] == "request_report"

    report = client.get(
        "/market-value/player/team_fw_1/report",
        headers=headers,
        params={"split": "test", "season": "2024/25"},
    )
    assert report.status_code == 200
    assert report.json()["report"]["latest_decision"]["action"] == "request_report"


def test_team_assignments_comments_activity_compare_lists_and_preferences(configured_team_env) -> None:
    client = _StatefulASGITestClient(app)
    workspace_id, user_id = _bootstrap_admin(client)
    headers = _workspace_headers(workspace_id)

    assignment = client.post(
        "/team/assignments",
        headers=headers,
        json={
            "player_id": "team_fw_1",
            "split": "test",
            "season": "2024/25",
            "assignee_user_id": user_id,
            "status": "to_watch",
            "note": "Watch this player live next week.",
        },
    )
    assert assignment.status_code == 200
    assert assignment.json()["assignment"]["assignee_user_id"] == user_id

    comment = client.post(
        "/team/player/team_fw_1/comments",
        headers=headers,
        json={"split": "test", "season": "2024/25", "body": "Shared note from the admin scout."},
    )
    assert comment.status_code == 200

    compare = client.post("/team/compare-lists", headers=headers, json={"name": "Left Wing Targets", "notes": "Initial compare set"})
    assert compare.status_code == 200
    compare_id = compare.json()["compare_list"]["compare_id"]

    for player_id in ("team_fw_1", "team_fw_2"):
        added = client.post(
            f"/team/compare-lists/{compare_id}/players",
            headers=headers,
            json={"player_id": player_id, "split": "test", "season": "2024/25", "pinned": player_id == "team_fw_1"},
        )
        assert added.status_code == 200

    compare_lists = client.get("/team/compare-lists", headers=headers)
    assert compare_lists.status_code == 200
    compare_payload = compare_lists.json()["items"]
    assert len(compare_payload) == 1
    assert len(compare_payload[0]["players"]) == 2

    preferences = client.put(
        "/team/preferences/me",
        headers=headers,
        json={
            "name": "Aggressive Wing Search",
            "target_age_min": 18,
            "target_age_max": 22,
            "budget_posture": "disciplined",
            "trusted_league_posture": "trusted_first",
            "role_priorities": {"FW": 1.3},
            "system_template_default": "high_press_433",
            "must_have_tags": ["trajectory"],
            "avoid_tags": ["injury_risk"],
            "risk_tolerance": "conservative",
            "active_lane_preference": "valuation",
            "apply_by_default": True,
        },
    )
    assert preferences.status_code == 200
    assert preferences.json()["name"] == "Aggressive Wing Search"

    preferences_get = client.get("/team/preferences/me", headers=headers)
    assert preferences_get.status_code == 200
    assert preferences_get.json()["role_priorities"]["FW"] == pytest.approx(1.3)

    shortlist = client.get(
        "/market-value/shortlist",
        headers=headers,
        params={"split": "test", "top_n": 5, "apply_preferences": "true"},
    )
    assert shortlist.status_code == 200
    shortlist_items = shortlist.json()["items"]
    assert shortlist_items
    assert "preference_overlay_score" in shortlist_items[0]

    assignments = client.get("/team/assignments", headers=headers, params={"player_id": "team_fw_1"})
    assert assignments.status_code == 200
    assert assignments.json()["items"][0]["status"] == "to_watch"

    comments = client.get("/team/player/team_fw_1/comments", headers=headers, params={"split": "test", "season": "2024/25"})
    assert comments.status_code == 200
    assert comments.json()["items"][0]["body"] == "Shared note from the admin scout."

    activity = client.get("/team/activity", headers=headers, params={"limit": 20})
    assert activity.status_code == 200
    summaries = [item["summary"] for item in activity.json()["items"]]
    assert any("compare list" in summary.lower() for summary in summaries)
    assert any("updated their scout preferences" in summary.lower() for summary in summaries)


def test_viewer_can_read_but_cannot_edit_team_state(configured_team_env) -> None:
    admin_client = _StatefulASGITestClient(app)
    workspace_id, _ = _bootstrap_admin(admin_client)
    headers = _workspace_headers(workspace_id)

    invite = admin_client.post(
        f"/workspaces/{workspace_id}/invites",
        headers=headers,
        json={"email": "viewer@example.com", "role": "viewer"},
    )
    assert invite.status_code == 200
    token = invite.json()["token"]

    viewer_client = _StatefulASGITestClient(app)
    accepted = viewer_client.post(
        f"/invites/{token}/accept",
        json={
            "email": "viewer@example.com",
            "password": "viewer-pass-123",
            "full_name": "Viewer Scout",
        },
    )
    assert accepted.status_code == 200

    readable = viewer_client.get("/team/watchlist", headers=headers)
    assert readable.status_code == 200

    forbidden = viewer_client.post(
        "/team/player/team_fw_1/comments",
        headers=headers,
        json={"split": "test", "season": "2024/25", "body": "Viewers should not write comments."},
    )
    assert forbidden.status_code == 403
