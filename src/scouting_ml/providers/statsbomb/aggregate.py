from __future__ import annotations

from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scouting_ml.providers.identity import load_link_table, merge_player_links, normalize_season_label
from scouting_ml.providers.statsbomb.open_data import load_event_payload, load_match_records

BOX_X_MIN = 102.0
BOX_Y_MIN = 18.0
BOX_Y_MAX = 62.0
GOAL_X = 120.0
GOAL_Y = 40.0
KNOWN_FORMATIONS = ("433", "4231", "442", "343", "352", "3421", "4141", "4411", "532", "541")


def _event_minute(event: dict) -> float:
    minute = float(event.get("minute") or 0.0)
    second = float(event.get("second") or 0.0)
    return minute + (second / 60.0)


def _type_name(event: dict) -> str:
    return str(((event.get("type") or {}).get("name")) or "").strip()


def _location(event: dict) -> tuple[float, float] | None:
    loc = event.get("location")
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        try:
            return float(loc[0]), float(loc[1])
        except (TypeError, ValueError):
            return None
    return None


def _end_location(event: dict) -> tuple[float, float] | None:
    for key in ("pass", "carry", "shot", "dribble"):
        payload = event.get(key)
        if not isinstance(payload, dict):
            continue
        loc = payload.get("end_location")
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            try:
                return float(loc[0]), float(loc[1])
            except (TypeError, ValueError):
                return None
    return None


def _distance_to_goal(loc: tuple[float, float] | None) -> float | None:
    if loc is None:
        return None
    return sqrt(((GOAL_X - loc[0]) ** 2) + ((GOAL_Y - loc[1]) ** 2))


def _is_progressive(event: dict) -> bool:
    start = _location(event)
    end = _end_location(event)
    start_d = _distance_to_goal(start)
    end_d = _distance_to_goal(end)
    if start_d is None or end_d is None:
        return False
    return (start_d - end_d) >= 10.0


def _is_box_entry(event: dict) -> bool:
    end = _end_location(event)
    if end is None:
        return False
    return end[0] >= BOX_X_MIN and BOX_Y_MIN <= end[1] <= BOX_Y_MAX


def _player_key(event: dict) -> tuple[str, str]:
    player = event.get("player") or {}
    return str(player.get("id") or ""), str(player.get("name") or "")


def _team_key(event: dict) -> tuple[str, str]:
    team = event.get("team") or {}
    return str(team.get("id") or ""), str(team.get("name") or "")


def _is_completed_pass(event: dict) -> bool:
    outcome = ((event.get("pass") or {}).get("outcome")) or {}
    return not bool(outcome)


def _duel_outcome_name(event: dict) -> str:
    outcome = ((event.get("duel") or {}).get("outcome")) or {}
    return str(outcome.get("name") or "").lower()


def _aerial_outcome_name(event: dict) -> str:
    outcome = ((event.get("50_50") or {}).get("outcome")) or {}
    return str(outcome.get("name") or "").lower()


def _receipt_between_lines(event: dict) -> bool:
    if _type_name(event) != "Ball Receipt*":
        return False
    loc = _location(event)
    return bool(loc and 60.0 <= loc[0] <= 100.0 and BOX_Y_MIN <= loc[1] <= BOX_Y_MAX)


def _central_final_third_touch(event: dict) -> bool:
    loc = _location(event)
    return bool(loc and loc[0] >= 80.0 and BOX_Y_MIN <= loc[1] <= BOX_Y_MAX)


def _normalize_formation(value: object) -> str:
    return "".join(ch for ch in str(value or "").strip() if ch.isdigit())


def _formation_minutes_for_match(events: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    team_state: dict[str, dict] = {}
    minutes: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    match_end = max(90.0, max((_event_minute(ev) for ev in events), default=90.0))

    def flush_segment(team_id: str, end_minute: float) -> None:
        state = team_state.get(team_id)
        if not state or not state.get("formation"):
            return
        start_minute = float(state.get("last_minute", 0.0))
        delta = max(0.0, end_minute - start_minute)
        if delta <= 0:
            state["last_minute"] = end_minute
            return
        for player_id, player_name in state.get("on_field", {}).items():
            minutes[(str(player_id), str(player_name))][state["formation"]] += delta
        state["last_minute"] = end_minute

    for event in sorted(events, key=_event_minute):
        ev_type = _type_name(event)
        team_id, _ = _team_key(event)
        if not team_id:
            continue
        state = team_state.setdefault(team_id, {"formation": "", "last_minute": 0.0, "on_field": {}})
        current_minute = _event_minute(event)
        if ev_type == "Starting XI":
            lineup = (((event.get("tactics") or {}).get("lineup")) or [])
            state["formation"] = _normalize_formation(((event.get("tactics") or {}).get("formation")))
            state["on_field"] = {
                str(((item.get("player") or {}).get("id")) or ""): str(((item.get("player") or {}).get("name")) or "")
                for item in lineup
                if isinstance(item, dict) and ((item.get("player") or {}).get("id")) is not None
            }
            state["last_minute"] = current_minute
            continue
        if ev_type == "Tactical Shift":
            flush_segment(team_id, current_minute)
            state["formation"] = _normalize_formation(((event.get("tactics") or {}).get("formation")))
            continue
        if ev_type == "Substitution":
            flush_segment(team_id, current_minute)
            sub = event.get("substitution") or {}
            repl = sub.get("replacement") or {}
            outgoing_id, _outgoing_name = _player_key(event)
            if outgoing_id:
                state["on_field"].pop(outgoing_id, None)
            incoming_id = str(repl.get("id") or "")
            incoming_name = str(repl.get("name") or "")
            if incoming_id:
                state["on_field"][incoming_id] = incoming_name

    for team_id in list(team_state):
        flush_segment(team_id, match_end)
    return minutes


def aggregate_player_season_features(
    open_data_root: str | Path,
    *,
    competition_ids: Iterable[int] | None = None,
    season_ids: Iterable[int] | None = None,
    player_links_path: str | Path | None = None,
    snapshot_date: str | None = None,
    retrieved_at: str | None = None,
) -> pd.DataFrame:
    matches = load_match_records(open_data_root, competition_ids=competition_ids, season_ids=season_ids)
    if matches.empty:
        return pd.DataFrame()

    counter: dict[tuple, dict[str, float | str | None]] = {}
    for match in matches.to_dict(orient="records"):
        match_id = match.get("match_id")
        if match_id is None:
            continue
        events = load_event_payload(open_data_root, match_id)
        if not events:
            continue
        season = normalize_season_label(match.get("season_name") or match.get("season_id"))
        formation_minutes = _formation_minutes_for_match(events)

        for event in events:
            player_id, player_name = _player_key(event)
            if not player_id:
                continue
            team_id, team_name = _team_key(event)
            key = (
                str(player_id),
                str(player_name),
                str(team_id),
                str(team_name),
                str(season or ""),
                str(match.get("competition_name") or ""),
            )
            row = counter.setdefault(
                key,
                {
                    "provider_player_id": str(player_id),
                    "player_name": str(player_name),
                    "provider_team_id": str(team_id),
                    "team_name": str(team_name),
                    "season": season,
                    "league": match.get("competition_name"),
                    "matches": 0.0,
                    "minutes": 0.0,
                    "completed_passes": 0.0,
                    "progressive_passes": 0.0,
                    "progressive_carries": 0.0,
                    "passes_into_box": 0.0,
                    "shot_assists": 0.0,
                    "pressures": 0.0,
                    "counterpressures": 0.0,
                    "high_regains": 0.0,
                    "central_final_third_touches": 0.0,
                    "between_lines_receipts": 0.0,
                    "duels": 0.0,
                    "duel_wins": 0.0,
                    "aerial_duels": 0.0,
                    "aerial_duel_wins": 0.0,
                    "in_possession_actions": 0.0,
                    "out_of_possession_actions": 0.0,
                },
            )
            ev_type = _type_name(event)
            if ev_type in {"Pass", "Carry", "Dribble", "Ball Receipt*", "Shot"}:
                row["in_possession_actions"] += 1.0
            if ev_type in {"Pressure", "Duel", "Interception", "Ball Recovery", "Clearance", "Foul Committed", "50/50"}:
                row["out_of_possession_actions"] += 1.0
            if ev_type == "Pass" and _is_completed_pass(event):
                row["completed_passes"] += 1.0
            if ev_type == "Pass" and _is_progressive(event):
                row["progressive_passes"] += 1.0
            if ev_type == "Carry" and _is_progressive(event):
                row["progressive_carries"] += 1.0
            if ev_type == "Pass" and _is_box_entry(event):
                row["passes_into_box"] += 1.0
            if ev_type == "Pass" and bool((event.get("pass") or {}).get("shot_assist")):
                row["shot_assists"] += 1.0
            if ev_type == "Pressure":
                row["pressures"] += 1.0
                if bool(event.get("counterpress")):
                    row["counterpressures"] += 1.0
            if ev_type in {"Ball Recovery", "Interception"}:
                loc = _location(event)
                if loc and loc[0] >= 60.0:
                    row["high_regains"] += 1.0
            if _central_final_third_touch(event):
                row["central_final_third_touches"] += 1.0
            if _receipt_between_lines(event):
                row["between_lines_receipts"] += 1.0
            if ev_type == "Duel":
                row["duels"] += 1.0
                if any(token in _duel_outcome_name(event) for token in ("won", "success")):
                    row["duel_wins"] += 1.0
            if ev_type == "50/50":
                row["aerial_duels"] += 1.0
                if any(token in _aerial_outcome_name(event) for token in ("won", "success")):
                    row["aerial_duel_wins"] += 1.0

        for (player_id, player_name), formations in formation_minutes.items():
            team_id = ""
            team_name = ""
            for event in events:
                ev_player_id, ev_player_name = _player_key(event)
                if ev_player_id == player_id and ev_player_name == player_name:
                    team_id, team_name = _team_key(event)
                    break
            key = (
                str(player_id),
                str(player_name),
                str(team_id),
                str(team_name),
                str(season or ""),
                str(match.get("competition_name") or ""),
            )
            row = counter.setdefault(
                key,
                {
                    "provider_player_id": str(player_id),
                    "player_name": str(player_name),
                    "provider_team_id": str(team_id),
                    "team_name": str(team_name),
                    "season": season,
                    "league": match.get("competition_name"),
                },
            )
            row["matches"] = float(row.get("matches", 0.0)) + 1.0
            row["minutes"] = float(row.get("minutes", 0.0)) + float(sum(formations.values()))
            for formation, mins in formations.items():
                if formation:
                    row[f"minutes_in_{formation}"] = float(row.get(f"minutes_in_{formation}", 0.0)) + float(mins)

    out = pd.DataFrame(counter.values())
    if out.empty:
        return out

    for col in [
        "matches",
        "minutes",
        "completed_passes",
        "progressive_passes",
        "progressive_carries",
        "passes_into_box",
        "shot_assists",
        "pressures",
        "counterpressures",
        "high_regains",
        "central_final_third_touches",
        "between_lines_receipts",
        "duels",
        "duel_wins",
        "aerial_duels",
        "aerial_duel_wins",
        "in_possession_actions",
        "out_of_possession_actions",
    ]:
        if col not in out.columns:
            out[col] = 0.0

    out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce")
    per90 = 90.0 / out["minutes"].replace({0: np.nan})
    for col in [
        "completed_passes",
        "progressive_passes",
        "progressive_carries",
        "passes_into_box",
        "shot_assists",
        "pressures",
        "counterpressures",
        "high_regains",
        "central_final_third_touches",
        "between_lines_receipts",
    ]:
        out[f"{col}_per90"] = pd.to_numeric(out[col], errors="coerce") * per90
    out["duel_win_rate"] = pd.to_numeric(out["duel_wins"], errors="coerce") / pd.to_numeric(out["duels"], errors="coerce").replace({0: np.nan})
    out["aerial_win_rate"] = pd.to_numeric(out["aerial_duel_wins"], errors="coerce") / pd.to_numeric(out["aerial_duels"], errors="coerce").replace({0: np.nan})
    total_actions = (
        pd.to_numeric(out["in_possession_actions"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["out_of_possession_actions"], errors="coerce").fillna(0.0)
    ).replace({0: np.nan})
    out["in_possession_action_share"] = pd.to_numeric(out["in_possession_actions"], errors="coerce") / total_actions
    out["out_of_possession_action_share"] = pd.to_numeric(out["out_of_possession_actions"], errors="coerce") / total_actions
    for formation in KNOWN_FORMATIONS:
        col = f"minutes_in_{formation}"
        if col not in out.columns:
            out[col] = 0.0

    links = load_link_table(player_links_path)
    out = merge_player_links(out, links, provider="statsbomb", provider_id_col="provider_player_id", player_name_col="player_name", club_col="team_name")
    out["source_provider"] = "statsbomb"
    out["source_version"] = "open-data"
    out["retrieved_at"] = retrieved_at
    out["snapshot_date"] = snapshot_date
    out["coverage_note"] = "StatsBomb open-data coverage is partial and competition-dependent."
    return out
