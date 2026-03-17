from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_match_records(
    root: str | Path,
    *,
    competition_ids: Iterable[int] | None = None,
    season_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    base = Path(root)
    match_dir = base / "matches"
    comp_filter = {int(v) for v in competition_ids} if competition_ids else None
    season_filter = {int(v) for v in season_ids} if season_ids else None
    rows: list[dict] = []
    for path in sorted(match_dir.glob("*/*.json")):
        try:
            competition_id = int(path.parent.name)
            season_id = int(path.stem)
        except ValueError:
            continue
        if comp_filter and competition_id not in comp_filter:
            continue
        if season_filter and season_id not in season_filter:
            continue
        payload = _read_json(path)
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "match_id": item.get("match_id"),
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "competition_name": ((item.get("competition") or {}).get("competition_name") if isinstance(item.get("competition"), dict) else None),
                    "season_name": ((item.get("season") or {}).get("season_name") if isinstance(item.get("season"), dict) else None),
                    "match_date": item.get("match_date") or item.get("match_date_utc"),
                    "home_team_id": ((item.get("home_team") or {}).get("home_team_id") if isinstance(item.get("home_team"), dict) else None),
                    "home_team_name": ((item.get("home_team") or {}).get("home_team_name") if isinstance(item.get("home_team"), dict) else None),
                    "away_team_id": ((item.get("away_team") or {}).get("away_team_id") if isinstance(item.get("away_team"), dict) else None),
                    "away_team_name": ((item.get("away_team") or {}).get("away_team_name") if isinstance(item.get("away_team"), dict) else None),
                }
            )
    return pd.DataFrame(rows)


def load_event_payload(root: str | Path, match_id: int | str) -> list[dict]:
    path = Path(root) / "events" / f"{match_id}.json"
    payload = _read_json(path)
    return payload if isinstance(payload, list) else []
