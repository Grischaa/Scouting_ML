from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scouting_ml.providers.identity import (
    normalize_club_name,
    normalize_person_name,
    normalize_season_label,
)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_players_source(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.rglob("*_with_sofa.csv"))
        if not files:
            raise ValueError(f"No *_with_sofa.csv files found in {path}")
        return pd.concat([pd.read_csv(file) for file in files], ignore_index=True, sort=False)
    return _read_table(path)


def _safe_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col in frame.columns:
        return frame[col]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _coerce_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
        .fillna("")
    )


def _parse_dt(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    if isinstance(value, (int, float)) and not pd.isna(value):
        ts = float(value)
        if ts > 10_000_000_000:
            ts = ts / 1000.0
        return pd.to_datetime(ts, unit="s", errors="coerce", utc=True)
    return pd.to_datetime(value, errors="coerce", utc=True)


def _payload_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    events = payload.get("events")
    if isinstance(events, list):
        return [item for item in events if isinstance(item, dict)]
    matches = payload.get("matches")
    if isinstance(matches, list):
        out: list[dict[str, Any]] = []
        for item in matches:
            if not isinstance(item, dict):
                continue
            event = item.get("event")
            if isinstance(event, dict):
                out.append(event)
        return out
    return []


def _payload_matches(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    matches = payload.get("matches")
    if isinstance(matches, list):
        return [item for item in matches if isinstance(item, dict)]
    if "event" in payload and "lineups" in payload:
        return [payload]
    return []


def _event_league_name(event: dict[str, Any], payload: dict[str, Any] | None = None) -> str | None:
    tournament = event.get("tournament") or {}
    unique = tournament.get("uniqueTournament") if isinstance(tournament, dict) else {}
    for value in (
        unique.get("name") if isinstance(unique, dict) else None,
        tournament.get("name") if isinstance(tournament, dict) else None,
        payload.get("competition", {}).get("league") if isinstance(payload, dict) else None,
    ):
        if value:
            return str(value)
    return None


def _event_season_label(event: dict[str, Any], payload: dict[str, Any] | None = None) -> str | None:
    tournament = event.get("tournament") or {}
    season = tournament.get("season") if isinstance(tournament, dict) else {}
    for value in (
        season.get("name") if isinstance(season, dict) else None,
        event.get("season", {}).get("name") if isinstance(event.get("season"), dict) else None,
        payload.get("competition", {}).get("season") if isinstance(payload, dict) else None,
    ):
        season_label = normalize_season_label(value)
        if season_label:
            return season_label
    return None


def _score_value(score: Any) -> Any:
    if isinstance(score, dict):
        return score.get("current", score.get("display"))
    return score


def normalize_fixtures(payload: Any) -> pd.DataFrame:
    payload_dict = payload if isinstance(payload, dict) else None
    rows: list[dict[str, Any]] = []
    for event in _payload_events(payload):
        home = event.get("homeTeam") or {}
        away = event.get("awayTeam") or {}
        rows.append(
            {
                "provider_fixture_id": str(event.get("id") or ""),
                "season": _event_season_label(event, payload_dict),
                "league": _event_league_name(event, payload_dict),
                "match_date": _parse_dt(event.get("startTimestamp") or event.get("startDate") or event.get("start_time")),
                "home_team_id": str(home.get("id") or ""),
                "home_team_name": home.get("name"),
                "away_team_id": str(away.get("id") or ""),
                "away_team_name": away.get("name"),
                "home_goals": _score_value(event.get("homeScore")),
                "away_goals": _score_value(event.get("awayScore")),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["season"] = out["season"].apply(normalize_season_label)
    return out


def _lineup_players(side_payload: dict[str, Any]) -> list[dict[str, Any]]:
    players = side_payload.get("players")
    if isinstance(players, list):
        return [item for item in players if isinstance(item, dict)]
    starters = side_payload.get("starters")
    bench = side_payload.get("substitutes")
    out: list[dict[str, Any]] = []
    if isinstance(starters, list):
        out.extend([dict(item, substitute=False) for item in starters if isinstance(item, dict)])
    if isinstance(bench, list):
        out.extend([dict(item, substitute=True) for item in bench if isinstance(item, dict)])
    return out


def _player_minutes(player_row: dict[str, Any]) -> float | None:
    return _player_metric(player_row, "minutesPlayed", "minutes")


def _metric_to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        for key in ("value", "current", "display", "original", "rating"):
            if key in value:
                parsed = _metric_to_float(value.get(key))
                if parsed is not None:
                    return parsed
        for nested in value.values():
            parsed = _metric_to_float(nested)
            if parsed is not None:
                return parsed
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _player_metric(player_row: dict[str, Any], *keys: str) -> float | None:
    stats = player_row.get("statistics") or {}
    sources = [stats if isinstance(stats, dict) else {}, player_row]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key not in source:
                continue
            parsed = _metric_to_float(source.get(key))
            if parsed is not None:
                return parsed
    return None


def _player_flag(player_row: dict[str, Any], *keys: str) -> int:
    stats = player_row.get("statistics") or {}
    sources = [player_row, stats if isinstance(stats, dict) else {}]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key not in source:
                continue
            value = source.get(key)
            if isinstance(value, bool):
                return int(value)
            parsed = _metric_to_float(value)
            if parsed is not None:
                return int(parsed > 0.0)
            text = str(value).strip().casefold()
            if text in {"true", "yes", "captain", "starter"}:
                return 1
            if text in {"false", "no"}:
                return 0
    return 0


def _player_started(player_row: dict[str, Any]) -> int:
    if bool(player_row.get("substitute")):
        return 0
    if player_row.get("formationPosition") is not None:
        return 1
    return 1


def _status_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("reason", "type", "status", "description", "label", "text"):
            text = value.get(key)
            if text:
                return str(text).strip().lower()
        return ""
    if value is None:
        return ""
    return str(value).strip().lower()


def _injury_flag(value: Any) -> int:
    text = _status_text(value)
    return int(any(token in text for token in ("injur", "knock", "ill", "fitness", "doubt")))


def _suspension_flag(value: Any) -> int:
    text = _status_text(value)
    return int(any(token in text for token in ("suspend", "ban", "red card")))


def normalize_player_availability(payload: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _payload_matches(payload):
        event = item.get("event") if isinstance(item.get("event"), dict) else {}
        lineups = item.get("lineups") if isinstance(item.get("lineups"), dict) else item
        league = _event_league_name(event, payload if isinstance(payload, dict) else None)
        season = _event_season_label(event, payload if isinstance(payload, dict) else None)
        match_date = _parse_dt(event.get("startTimestamp") or event.get("startDate") or event.get("start_time"))
        provider_fixture_id = str(event.get("id") or "")
        for side_name, team_key in (("home", "homeTeam"), ("away", "awayTeam")):
            side_payload = lineups.get(side_name) or {}
            team = event.get(team_key) or {}
            provider_team_id = str(team.get("id") or side_payload.get("team", {}).get("id") or "")
            team_name = team.get("name") or (side_payload.get("team") or {}).get("name")

            for player_row in _lineup_players(side_payload):
                player = player_row.get("player") or player_row
                start_flag = _player_started(player_row)
                bench_flag = int(bool(player_row.get("substitute")))
                minutes = _player_minutes(player_row)
                appearance_flag = int((minutes or 0.0) > 0.0 or start_flag > 0)
                sub_on_flag = int(bench_flag > 0 and (minutes or 0.0) > 0.0)
                unused_bench_flag = int(bench_flag > 0 and (minutes or 0.0) <= 0.0)
                full_match_flag = int((minutes or 0.0) >= 85.0)
                captain_flag = _player_flag(player_row, "captain", "isCaptain")
                rows.append(
                    {
                        "provider_player_id": str(player.get("id") or ""),
                        "player_name": player.get("name"),
                        "provider_team_id": provider_team_id,
                        "team_name": team_name,
                        "league": league,
                        "season": season,
                        "provider_fixture_id": provider_fixture_id,
                        "match_date": match_date,
                        "minutes": minutes,
                        "appearance_flag": appearance_flag,
                        "start_flag": start_flag,
                        "bench_flag": bench_flag,
                        "sub_on_flag": sub_on_flag,
                        "unused_bench_flag": unused_bench_flag,
                        "full_match_flag": full_match_flag,
                        "captain_flag": captain_flag,
                        "injury_flag": 0,
                        "suspension_flag": 0,
                        "expected_start_flag": start_flag,
                        "rating": _player_metric(player_row, "rating", "averageRating", "totalRating", "ratingVersions"),
                        "goals": _player_metric(player_row, "goals"),
                        "assists": _player_metric(player_row, "assists"),
                        "shots": _player_metric(player_row, "totalShots", "shots"),
                        "shots_on_target": _player_metric(player_row, "shotsOnTarget"),
                        "key_passes": _player_metric(player_row, "keyPasses"),
                        "touches": _player_metric(player_row, "touches"),
                        "total_duels_won": _player_metric(player_row, "totalDuelsWon", "duelsWon"),
                        "ground_duels_won": _player_metric(player_row, "groundDuelsWon"),
                        "aerial_duels_won": _player_metric(player_row, "aerialDuelsWon"),
                        "dribbles": _player_metric(player_row, "successfulDribbles", "dribbles"),
                        "tackles": _player_metric(player_row, "tackles"),
                        "interceptions": _player_metric(player_row, "interceptions"),
                        "clearances": _player_metric(player_row, "clearances"),
                        "saves": _player_metric(player_row, "saves"),
                    }
                )

            missing_players = side_payload.get("missingPlayers")
            if isinstance(missing_players, list):
                for player_row in missing_players:
                    if not isinstance(player_row, dict):
                        continue
                    player = player_row.get("player") or player_row
                    reason = (
                        player_row.get("reason")
                        or player_row.get("type")
                        or player_row.get("status")
                        or player_row.get("absence")
                    )
                    rows.append(
                        {
                            "provider_player_id": str(player.get("id") or ""),
                            "player_name": player.get("name"),
                            "provider_team_id": provider_team_id,
                            "team_name": team_name,
                            "league": league,
                            "season": season,
                            "provider_fixture_id": provider_fixture_id,
                            "match_date": match_date,
                            "minutes": 0.0,
                            "appearance_flag": 0,
                            "start_flag": 0,
                            "bench_flag": 0,
                            "sub_on_flag": 0,
                            "unused_bench_flag": 0,
                            "full_match_flag": 0,
                            "captain_flag": 0,
                            "injury_flag": _injury_flag(reason),
                            "suspension_flag": _suspension_flag(reason),
                            "expected_start_flag": 0,
                            "rating": np.nan,
                            "goals": 0.0,
                            "assists": 0.0,
                            "shots": 0.0,
                            "shots_on_target": 0.0,
                            "key_passes": 0.0,
                            "touches": 0.0,
                            "total_duels_won": 0.0,
                            "ground_duels_won": 0.0,
                            "aerial_duels_won": 0.0,
                            "dribbles": 0.0,
                            "tackles": 0.0,
                            "interceptions": 0.0,
                            "clearances": 0.0,
                            "saves": 0.0,
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["season"] = out["season"].apply(normalize_season_label)
    out["match_date"] = out["match_date"].apply(_parse_dt)
    out["provider_player_id"] = _coerce_id(out["provider_player_id"])
    out["provider_team_id"] = _coerce_id(out["provider_team_id"])
    return out.drop_duplicates(
        subset=["season", "provider_fixture_id", "provider_team_id", "provider_player_id", "player_name"],
        keep="last",
    ).reset_index(drop=True)


def _load_identity_lookups(players_source: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(players_source)
    base = _load_players_source(source)
    if base.empty:
        raise ValueError(f"No player rows found in players source: {source}")
    base = base.copy()
    base["season"] = _safe_series(base, "season").apply(normalize_season_label)
    base["player_id"] = _coerce_id(_safe_series(base, "player_id"))
    base["transfermarkt_id"] = _coerce_id(_safe_series(base, "transfermarkt_id"))
    base["sofa_player_id"] = _coerce_id(_safe_series(base, "sofa_player_id"))
    base["sofa_team_id"] = _coerce_id(_safe_series(base, "sofa_team_id"))
    base["name"] = _safe_series(base, "name").astype(str)
    base["dob"] = _safe_series(base, "dob").astype(str)
    base["club"] = _safe_series(base, "club").astype(str)
    base["league"] = _safe_series(base, "league").astype(str)
    base["sofa_team_name"] = _safe_series(base, "sofa_team_name").fillna(base["club"]).astype(str)
    base["_norm_name"] = base["name"].apply(normalize_person_name)
    base["_norm_club"] = base["club"].apply(normalize_club_name)
    base["_norm_sofa_team_name"] = base["sofa_team_name"].apply(normalize_club_name)
    base["_norm_league"] = base["league"].astype(str).str.strip().str.lower()

    player_lookup = (
        base[
            [
                "season",
                "sofa_player_id",
                "player_id",
                "transfermarkt_id",
                "dob",
                "name",
                "_norm_name",
                "_norm_club",
                "_norm_league",
            ]
        ]
        .replace({"": pd.NA})
        .dropna(subset=["season"])
        .drop_duplicates(["season", "sofa_player_id"], keep="last")
    )
    club_lookup = (
        base[
            [
                "season",
                "sofa_team_id",
                "sofa_team_name",
                "club",
                "league",
                "_norm_sofa_team_name",
                "_norm_club",
                "_norm_league",
            ]
        ]
        .replace({"": pd.NA})
        .dropna(subset=["season"])
        .drop_duplicates(["season", "sofa_team_id"], keep="last")
    )
    return player_lookup, club_lookup


def _merge_club_lookup(frame: pd.DataFrame, club_lookup: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_provider_team_key"] = _coerce_id(out.get("provider_team_id", pd.Series("", index=out.index, dtype="object")))
    out["_norm_team_name"] = out.get("team_name", pd.Series("", index=out.index, dtype="object")).apply(normalize_club_name)
    out["_norm_league"] = out.get("league", pd.Series("", index=out.index, dtype="object")).astype(str).str.strip().str.lower()

    by_id = club_lookup.copy()
    by_id["_provider_team_key"] = _coerce_id(by_id["sofa_team_id"])
    if by_id["_provider_team_key"].ne("").any() and out["_provider_team_key"].ne("").any():
        out = out.merge(
            by_id[["season", "_provider_team_key", "club", "league"]].drop_duplicates(["season", "_provider_team_key"], keep="last"),
            on=["season", "_provider_team_key"],
            how="left",
            suffixes=("", "_lookup"),
        )
        if "club_lookup" in out.columns:
            current = out.get("club")
            out["club"] = current.where(current.notna(), out["club_lookup"]) if current is not None else out["club_lookup"]
            out = out.drop(columns=["club_lookup"], errors="ignore")
        if "league_lookup" in out.columns:
            current = out.get("league")
            out["league"] = (
                current.where(current.notna(), out["league_lookup"]) if current is not None else out["league_lookup"]
            )
            out = out.drop(columns=["league_lookup"], errors="ignore")

    if "club" not in out.columns or out["club"].isna().any():
        fallback = club_lookup.rename(columns={"_norm_sofa_team_name": "_norm_team_name"})
        out = out.merge(
            fallback[["season", "_norm_team_name", "_norm_league", "club", "league"]].drop_duplicates(
                ["season", "_norm_team_name", "_norm_league"],
                keep="last",
            ),
            on=["season", "_norm_team_name", "_norm_league"],
            how="left",
            suffixes=("", "_fallback"),
        )
        if "club_fallback" in out.columns:
            if "club" in out.columns:
                out["club"] = out["club"].where(out["club"].notna(), out["club_fallback"])
            else:
                out["club"] = out["club_fallback"]
        if "league_fallback" in out.columns:
            current = out.get("league")
            out["league"] = (
                current.where(current.notna(), out["league_fallback"]) if current is not None else out["league_fallback"]
            )

    return out.drop(
        columns=[
            "_provider_team_key",
            "_norm_team_name",
            "_norm_league",
            "club_fallback",
            "league_fallback",
        ],
        errors="ignore",
    )


def _merge_player_lookup(frame: pd.DataFrame, player_lookup: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_provider_player_key"] = _coerce_id(out.get("provider_player_id", pd.Series("", index=out.index, dtype="object")))
    out["_norm_player_name"] = out.get("player_name", pd.Series("", index=out.index, dtype="object")).apply(normalize_person_name)
    out["_norm_league"] = out.get("league", pd.Series("", index=out.index, dtype="object")).astype(str).str.strip().str.lower()

    by_id = player_lookup.copy()
    by_id["_provider_player_key"] = _coerce_id(by_id["sofa_player_id"])
    if by_id["_provider_player_key"].ne("").any() and out["_provider_player_key"].ne("").any():
        out = out.merge(
            by_id[["season", "_provider_player_key", "player_id", "transfermarkt_id", "dob"]].drop_duplicates(
                ["season", "_provider_player_key"],
                keep="last",
            ),
            on=["season", "_provider_player_key"],
            how="left",
            suffixes=("", "_lookup"),
        )
        for col in ("player_id", "transfermarkt_id", "dob"):
            lookup_col = f"{col}_lookup"
            if lookup_col in out.columns:
                if col in out.columns:
                    out[col] = out[col].where(out[col].notna(), out[lookup_col])
                else:
                    out[col] = out[lookup_col]
                out = out.drop(columns=[lookup_col], errors="ignore")

    if "player_id" not in out.columns or out["player_id"].isna().any():
        out = out.merge(
            player_lookup[["season", "_norm_name", "_norm_league", "player_id", "transfermarkt_id", "dob"]]
            .drop_duplicates(["season", "_norm_name", "_norm_league"], keep="last"),
            left_on=["season", "_norm_player_name", "_norm_league"],
            right_on=["season", "_norm_name", "_norm_league"],
            how="left",
            suffixes=("", "_fallback"),
        )
        for col in ("player_id", "transfermarkt_id", "dob"):
            fallback_col = f"{col}_fallback"
            if fallback_col in out.columns:
                if col in out.columns:
                    out[col] = out[col].where(out[col].notna(), out[fallback_col])
                else:
                    out[col] = out[fallback_col]

    return out.drop(
        columns=[
            "_provider_player_key",
            "_norm_player_name",
            "_norm_name",
            "_norm_league",
            "player_id_fallback",
            "transfermarkt_id_fallback",
            "dob_fallback",
        ],
        errors="ignore",
    )


def build_fixture_context(
    fixtures: pd.DataFrame,
    *,
    players_source: str | Path,
    snapshot_date: str | None = None,
    retrieved_at: str | None = None,
) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame()
    _, club_lookup = _load_identity_lookups(players_source)
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
    team_rows["goal_diff"] = gf - ga
    team_rows["win_flag"] = (gf > ga).astype(float)
    team_rows["draw_flag"] = (gf == ga).astype(float)
    team_rows["loss_flag"] = (gf < ga).astype(float)
    team_rows["clean_sheet_flag"] = (ga == 0).astype(float)
    team_rows["failed_to_score_flag"] = (gf == 0).astype(float)
    team_rows["total_goals_environment"] = gf + ga
    strength = (
        team_rows.groupby(["league", "season", "team_name"], dropna=False)
        .agg(
            team_matches=("provider_fixture_id", "nunique"),
            team_points=("points", "sum"),
            team_gf=("goals_for", "sum"),
            team_ga=("goals_against", "sum"),
        )
        .reset_index()
    )
    strength["team_points_per_match"] = strength["team_points"] / strength["team_matches"].replace({0: np.nan})
    strength["team_goal_diff_per_match"] = (
        pd.to_numeric(strength["team_gf"], errors="coerce")
        - pd.to_numeric(strength["team_ga"], errors="coerce")
    ) / strength["team_matches"].replace({0: np.nan})
    team_rows = team_rows.merge(
        strength[["league", "season", "team_name", "team_points_per_match", "team_goal_diff_per_match"]]
        .rename(
            columns={
                "team_name": "opponent_name",
                "team_points_per_match": "opponent_points_per_match",
                "team_goal_diff_per_match": "opponent_goal_diff_per_match",
            }
        ),
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
            fixture_points_per_match=("points", "mean"),
            fixture_goal_diff_per_match=("goal_diff", "mean"),
            fixture_goals_for_per_match=("goals_for", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            fixture_goals_against_per_match=("goals_against", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            fixture_win_share=("win_flag", "mean"),
            fixture_draw_share=("draw_flag", "mean"),
            fixture_loss_share=("loss_flag", "mean"),
            fixture_clean_sheet_share=("clean_sheet_flag", "mean"),
            fixture_failed_to_score_share=("failed_to_score_flag", "mean"),
            fixture_scoring_environment=("total_goals_environment", "mean"),
            fixture_opponent_strength=("opponent_points_per_match", "mean"),
            fixture_opponent_goal_diff_strength=("opponent_goal_diff_per_match", "mean"),
        )
        .reset_index()
    )
    out = _merge_club_lookup(out, club_lookup)
    out["source_provider"] = "sofascore"
    out["source_version"] = "website_snapshot"
    out["retrieved_at"] = retrieved_at
    out["snapshot_date"] = snapshot_date
    out["coverage_note"] = "Fixture context is season-aggregated from SofaScore website match snapshots."
    return out


def build_player_availability(
    availability: pd.DataFrame,
    *,
    players_source: str | Path,
    snapshot_date: str | None = None,
    retrieved_at: str | None = None,
) -> pd.DataFrame:
    if availability.empty:
        return pd.DataFrame()
    player_lookup, club_lookup = _load_identity_lookups(players_source)
    work = availability.copy()
    work["minutes"] = pd.to_numeric(work["minutes"], errors="coerce")
    raw_numeric_cols = [
        "appearance_flag",
        "sub_on_flag",
        "unused_bench_flag",
        "full_match_flag",
        "captain_flag",
        "goals",
        "assists",
        "shots",
        "shots_on_target",
        "key_passes",
        "touches",
        "total_duels_won",
        "ground_duels_won",
        "aerial_duels_won",
        "dribbles",
        "tackles",
        "interceptions",
        "clearances",
        "saves",
        "rating",
    ]
    for col in raw_numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    out = (
        work.groupby(
            ["league", "season", "provider_player_id", "player_name", "provider_team_id", "team_name"],
            dropna=False,
        )
        .agg(
            avail_reports=("provider_fixture_id", "nunique"),
            avail_appearance_count=("appearance_flag", "sum"),
            avail_minutes=("minutes", "sum"),
            avail_start_count=("start_flag", "sum"),
            avail_bench_count=("bench_flag", "sum"),
            avail_sub_on_count=("sub_on_flag", "sum"),
            avail_unused_bench_count=("unused_bench_flag", "sum"),
            avail_full_match_count=("full_match_flag", "sum"),
            avail_captain_count=("captain_flag", "sum"),
            avail_injury_count=("injury_flag", "sum"),
            avail_suspension_count=("suspension_flag", "sum"),
            avail_expected_start_rate=("expected_start_flag", "mean"),
            avail_rating_mean=("rating", "mean"),
            avail_goals=("goals", "sum"),
            avail_assists=("assists", "sum"),
            avail_shots=("shots", "sum"),
            avail_shots_on_target=("shots_on_target", "sum"),
            avail_key_passes=("key_passes", "sum"),
            avail_touches=("touches", "sum"),
            avail_total_duels_won=("total_duels_won", "sum"),
            avail_ground_duels_won=("ground_duels_won", "sum"),
            avail_aerial_duels_won=("aerial_duels_won", "sum"),
            avail_dribbles=("dribbles", "sum"),
            avail_tackles=("tackles", "sum"),
            avail_interceptions=("interceptions", "sum"),
            avail_clearances=("clearances", "sum"),
            avail_saves=("saves", "sum"),
        )
        .reset_index()
    )
    reports = out["avail_reports"].replace({0: np.nan})
    appearances = out["avail_appearance_count"].replace({0: np.nan})
    minutes_capacity = reports * 90.0
    out["avail_start_share"] = out["avail_start_count"] / reports
    out["avail_bench_share"] = out["avail_bench_count"] / reports
    out["avail_appearance_share"] = out["avail_appearance_count"] / reports
    out["avail_sub_on_share"] = out["avail_sub_on_count"] / reports
    out["avail_unused_bench_share"] = out["avail_unused_bench_count"] / reports
    out["avail_full_match_share"] = out["avail_full_match_count"] / reports
    out["avail_captain_share"] = out["avail_captain_count"] / reports
    out["avail_minutes_per_report"] = out["avail_minutes"] / reports
    out["avail_minutes_per_appearance"] = out["avail_minutes"] / appearances
    out["avail_minutes_share"] = (out["avail_minutes"] / minutes_capacity).clip(lower=0.0, upper=1.0)
    out["avail_goal_contrib"] = out["avail_goals"].fillna(0.0) + out["avail_assists"].fillna(0.0)
    out["avail_goal_contrib_per_report"] = out["avail_goal_contrib"] / reports
    out["avail_goal_contrib_per90"] = out["avail_goal_contrib"] / (out["avail_minutes"].replace({0: np.nan}) / 90.0)
    out["avail_shots_per_report"] = out["avail_shots"] / reports
    out["avail_shots_on_target_per_report"] = out["avail_shots_on_target"] / reports
    out["avail_key_passes_per_report"] = out["avail_key_passes"] / reports
    out["avail_touches_per_report"] = out["avail_touches"] / reports
    out["avail_duels_won_per_report"] = out["avail_total_duels_won"] / reports
    out["avail_def_actions_per_report"] = (
        out["avail_tackles"].fillna(0.0)
        + out["avail_interceptions"].fillna(0.0)
        + out["avail_clearances"].fillna(0.0)
    ) / reports
    out["avail_saves_per_report"] = out["avail_saves"] / reports
    out = _merge_player_lookup(out, player_lookup)
    out = _merge_club_lookup(out, club_lookup)
    out["source_provider"] = "sofascore"
    out["source_version"] = "website_snapshot"
    out["retrieved_at"] = retrieved_at
    out["snapshot_date"] = snapshot_date
    out["coverage_note"] = "Availability is season-aggregated from SofaScore website lineups and missing-player snapshots."
    return out
