from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scouting_ml.providers.identity import load_link_table, merge_club_links, merge_player_links, normalize_season_label


def _payload_items(payload: Any, provider: str) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    if provider == "sportmonks":
        data = payload.get("data")
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    if provider == "api-football":
        data = payload.get("response")
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    data = payload.get("data")
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _parse_dt(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", utc=True)


def normalize_fixtures(payload: Any, *, provider: str) -> pd.DataFrame:
    rows: list[dict] = []
    for item in _payload_items(payload, provider):
        if provider == "sportmonks":
            participants = item.get("participants") or []
            home = next((p for p in participants if str((p.get("meta") or {}).get("location", "")).lower() == "home"), {})
            away = next((p for p in participants if str((p.get("meta") or {}).get("location", "")).lower() == "away"), {})
            scores = item.get("scores") or []
            score_map: dict[tuple[str, str], object] = {}
            for score in scores:
                if not isinstance(score, dict):
                    continue
                desc = str(score.get("description") or "").lower()
                participant = str(score.get("participant_id") or "")
                value = score.get("score")
                if isinstance(value, dict):
                    value = value.get("goals")
                score_map[(participant, desc)] = value
            rows.append(
                {
                    "provider_fixture_id": str(item.get("id") or ""),
                    "season": normalize_season_label(item.get("season_id") or ((item.get("season") or {}).get("name") if isinstance(item.get("season"), dict) else item.get("season"))),
                    "league": ((item.get("league") or {}).get("name") if isinstance(item.get("league"), dict) else item.get("league_name")),
                    "match_date": item.get("starting_at") or ((item.get("starting_at") or {}).get("date_time") if isinstance(item.get("starting_at"), dict) else None),
                    "home_team_id": str(home.get("id") or ""),
                    "home_team_name": home.get("name"),
                    "away_team_id": str(away.get("id") or ""),
                    "away_team_name": away.get("name"),
                    "home_goals": score_map.get((str(home.get("id") or ""), "current")),
                    "away_goals": score_map.get((str(away.get("id") or ""), "current")),
                }
            )
            continue
        fixture = item.get("fixture") or {}
        league = item.get("league") or {}
        teams = item.get("teams") or {}
        goals = item.get("goals") or {}
        rows.append(
            {
                "provider_fixture_id": str(fixture.get("id") or ""),
                "season": normalize_season_label(league.get("season")),
                "league": league.get("name"),
                "match_date": fixture.get("date"),
                "home_team_id": str(((teams.get("home") or {}).get("id")) or ""),
                "home_team_name": (teams.get("home") or {}).get("name"),
                "away_team_id": str(((teams.get("away") or {}).get("id")) or ""),
                "away_team_name": (teams.get("away") or {}).get("name"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["match_date"] = out["match_date"].apply(_parse_dt)
    return out


def build_fixture_context(
    fixtures: pd.DataFrame,
    *,
    provider: str,
    club_links_path: str | None = None,
    snapshot_date: str | None = None,
    retrieved_at: str | None = None,
) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame()
    home = fixtures.rename(
        columns={
            "home_team_id": "provider_team_id",
            "home_team_name": "team_name",
            "away_team_name": "opponent_name",
            "home_goals": "goals_for",
            "away_goals": "goals_against",
        }
    ).copy()
    home["is_home"] = 1
    away = fixtures.rename(
        columns={
            "away_team_id": "provider_team_id",
            "away_team_name": "team_name",
            "home_team_name": "opponent_name",
            "away_goals": "goals_for",
            "home_goals": "goals_against",
        }
    ).copy()
    away["is_home"] = 0
    team_rows = pd.concat([home, away], ignore_index=True, sort=False)
    team_rows = team_rows.sort_values(["league", "season", "team_name", "match_date"]).reset_index(drop=True)
    team_rows["rest_days"] = (
        team_rows.groupby(["league", "season", "team_name"], dropna=False)["match_date"]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
    )
    gf = pd.to_numeric(team_rows["goals_for"], errors="coerce")
    ga = pd.to_numeric(team_rows["goals_against"], errors="coerce")
    team_rows["points"] = np.select([gf > ga, gf == ga], [3.0, 1.0], default=0.0)

    strength = (
        team_rows.groupby(["league", "season", "team_name"], dropna=False)
        .agg(team_matches=("provider_fixture_id", "nunique"), team_points=("points", "sum"), team_gf=("goals_for", "sum"), team_ga=("goals_against", "sum"))
        .reset_index()
    )
    strength["team_points_per_match"] = strength["team_points"] / strength["team_matches"].replace({0: np.nan})
    strength["team_goal_diff_per_match"] = (pd.to_numeric(strength["team_gf"], errors="coerce") - pd.to_numeric(strength["team_ga"], errors="coerce")) / strength["team_matches"].replace({0: np.nan})
    team_rows = team_rows.merge(
        strength[["league", "season", "team_name", "team_points_per_match", "team_goal_diff_per_match"]]
        .rename(columns={"team_name": "opponent_name", "team_points_per_match": "opponent_points_per_match", "team_goal_diff_per_match": "opponent_goal_diff_per_match"}),
        on=["league", "season", "opponent_name"],
        how="left",
    )
    out = (
        team_rows.groupby(["league", "season", "provider_team_id", "team_name"], dropna=False)
        .agg(
            fixture_matches=("provider_fixture_id", "nunique"),
            fixture_home_share=("is_home", "mean"),
            fixture_mean_rest_days=("rest_days", "mean"),
            fixture_min_rest_days=("rest_days", "min"),
            fixture_congestion_share=("rest_days", lambda s: float((pd.to_numeric(s, errors="coerce") <= 3.0).mean())),
            fixture_opponent_strength=("opponent_points_per_match", "mean"),
            fixture_opponent_goal_diff_strength=("opponent_goal_diff_per_match", "mean"),
        )
        .reset_index()
    )
    links = load_link_table(club_links_path)
    out = merge_club_links(out, links, provider=provider, provider_team_id_col="provider_team_id", team_name_col="team_name")
    out["source_provider"] = provider
    out["source_version"] = "snapshot"
    out["retrieved_at"] = retrieved_at
    out["snapshot_date"] = snapshot_date
    out["coverage_note"] = "Fixture context is season-aggregated from provider fixture snapshots."
    return out


def normalize_player_availability(payload: Any, *, provider: str) -> pd.DataFrame:
    rows: list[dict] = []
    for item in _payload_items(payload, provider):
        if provider == "sportmonks":
            player = item.get("player") or item
            team = item.get("team") or {}
            fixture = item.get("fixture") or {}
            rows.append(
                {
                    "provider_player_id": str(player.get("id") or item.get("player_id") or ""),
                    "player_name": player.get("name") or item.get("player_name"),
                    "provider_team_id": str(team.get("id") or item.get("team_id") or ""),
                    "team_name": team.get("name") or item.get("team_name"),
                    "league": ((item.get("league") or {}).get("name") if isinstance(item.get("league"), dict) else item.get("league_name")),
                    "season": normalize_season_label(item.get("season_id") or ((item.get("season") or {}).get("name") if isinstance(item.get("season"), dict) else item.get("season"))),
                    "provider_fixture_id": str(fixture.get("id") or item.get("fixture_id") or ""),
                    "match_date": fixture.get("starting_at") or item.get("match_date"),
                    "minutes": item.get("minutes") or ((item.get("stats") or {}).get("minutes") if isinstance(item.get("stats"), dict) else None),
                    "start_flag": int(bool(item.get("starting") or item.get("is_starting"))),
                    "bench_flag": int(bool(item.get("bench") or item.get("is_bench"))),
                    "injury_flag": int(bool(item.get("injured") or item.get("is_injured"))),
                    "suspension_flag": int(bool(item.get("suspended") or item.get("is_suspended"))),
                    "expected_start_flag": int(bool(item.get("expected_starting") or item.get("predicted_starting"))),
                }
            )
            continue
        player = item.get("player") or {}
        team = item.get("team") or {}
        fixture = item.get("fixture") or {}
        rows.append(
            {
                "provider_player_id": str(player.get("id") or item.get("player_id") or ""),
                "player_name": player.get("name") or item.get("player_name"),
                "provider_team_id": str(team.get("id") or item.get("team_id") or ""),
                "team_name": team.get("name") or item.get("team_name"),
                "league": ((item.get("league") or {}).get("name") if isinstance(item.get("league"), dict) else item.get("league_name")),
                "season": normalize_season_label(((item.get("league") or {}).get("season")) if isinstance(item.get("league"), dict) else item.get("season")),
                "provider_fixture_id": str(fixture.get("id") or item.get("fixture_id") or ""),
                "match_date": fixture.get("date") or item.get("match_date"),
                "minutes": ((item.get("statistics") or {}).get("minutes") if isinstance(item.get("statistics"), dict) else item.get("minutes")),
                "start_flag": int(bool(item.get("starting") or item.get("is_starting"))),
                "bench_flag": int(bool(item.get("bench") or item.get("is_bench"))),
                "injury_flag": int(bool(item.get("injured") or item.get("is_injured"))),
                "suspension_flag": int(bool(item.get("suspended") or item.get("is_suspended"))),
                "expected_start_flag": int(bool(item.get("expected_starting") or item.get("predicted_starting"))),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["match_date"] = out["match_date"].apply(_parse_dt)
    return out


def build_player_availability(
    availability: pd.DataFrame,
    *,
    provider: str,
    player_links_path: str | None = None,
    club_links_path: str | None = None,
    snapshot_date: str | None = None,
    retrieved_at: str | None = None,
) -> pd.DataFrame:
    if availability.empty:
        return pd.DataFrame()
    work = availability.copy()
    work["minutes"] = pd.to_numeric(work["minutes"], errors="coerce")
    out = (
        work.groupby(["league", "season", "provider_player_id", "player_name", "provider_team_id", "team_name"], dropna=False)
        .agg(
            avail_reports=("provider_fixture_id", "nunique"),
            avail_minutes=("minutes", "sum"),
            avail_start_count=("start_flag", "sum"),
            avail_bench_count=("bench_flag", "sum"),
            avail_injury_count=("injury_flag", "sum"),
            avail_suspension_count=("suspension_flag", "sum"),
            avail_expected_start_rate=("expected_start_flag", "mean"),
        )
        .reset_index()
    )
    out["avail_start_share"] = out["avail_start_count"] / out["avail_reports"].replace({0: np.nan})
    out["avail_bench_share"] = out["avail_bench_count"] / out["avail_reports"].replace({0: np.nan})
    player_links = load_link_table(player_links_path)
    club_links = load_link_table(club_links_path)
    out = merge_player_links(out, player_links, provider=provider, provider_id_col="provider_player_id", player_name_col="player_name", club_col="team_name")
    out = merge_club_links(out, club_links, provider=provider, provider_team_id_col="provider_team_id", team_name_col="team_name")
    out["source_provider"] = provider
    out["source_version"] = "snapshot"
    out["retrieved_at"] = retrieved_at
    out["snapshot_date"] = snapshot_date
    out["coverage_note"] = "Availability is season-aggregated from vendor player status snapshots."
    return out
