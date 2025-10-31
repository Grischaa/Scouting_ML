# src/scouting_ml/sofa_parser.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from scouting_ml.paths import ensure_dirs, RAW_DIR

SOFA_ROOT = RAW_DIR / "sofascore"
PLAYERS_DIR = SOFA_ROOT / "players"


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _parse_one_season_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse ONE file like:
      data/raw/sofascore/players/123456/season_54186.json
    and return a flat dict with the most useful fields.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Sofascore usually has something like {"player": {...}, "statistics": {...}} or just {"statistics": {...}}
    player_id = None
    if "player" in data and isinstance(data["player"], dict):
        player_id = data["player"].get("id")
        player_name = data["player"].get("name")
    else:
        # fall back to folder name
        try:
            player_id = int(path.parent.name)
        except Exception:
            player_id = None
        player_name = None

    # detect season id from filename
    # season_54186.json -> 54186
    season_id = None
    if path.stem.startswith("season_"):
        try:
            season_id = int(path.stem.split("_", 1)[1])
        except Exception:
            pass

    stats = data.get("statistics") or data.get("player") or {}
    # Different endpoints put stats in different places. Try a few.
    appearances = _safe_get(stats, "appearances", default=None)
    minutes = _safe_get(stats, "minutes", default=None)
    goals = _safe_get(stats, "goals", default=None)
    assists = _safe_get(stats, "assists", default=None)
    yellow = _safe_get(stats, "yellowCards", default=None)
    red = _safe_get(stats, "redCards", default=None)
    rating = _safe_get(stats, "rating", default=None)

    # sometimes it's nested under "total"
    if appearances is None:
        appearances = _safe_get(stats, "total", "appearances")
    if minutes is None:
        minutes = _safe_get(stats, "total", "minutes")
    if goals is None:
        goals = _safe_get(stats, "total", "goals")
    if assists is None:
        assists = _safe_get(stats, "total", "assists")
    if rating is None:
        rating = _safe_get(stats, "total", "rating")

    return {
        "sofa_player_id": player_id,
        "player_name": player_name,
        "season_id": season_id,
        "appearances": appearances,
        "minutes": minutes,
        "goals": goals,
        "assists": assists,
        "yellow_cards": yellow,
        "red_cards": red,
        "rating": rating,
        "source_file": str(path.relative_to(SOFA_ROOT)),
    }


def parse_all_players(root: Path = PLAYERS_DIR) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for player_dir in root.glob("*"):
        if not player_dir.is_dir():
            continue
        for season_file in player_dir.glob("season_*.json"):
            rec = _parse_one_season_json(season_file)
            if rec:
                rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # basic cleaning
    num_cols = ["appearances", "minutes", "goals", "assists", "yellow_cards", "red_cards", "rating"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ensure_dirs()
    df = parse_all_players()
    if df.empty:
        print("[sofa_parser] No Sofascore player season files found.")
        return
    out_path = Path("data/processed/sofa_players_seasons.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[sofa_parser] Wrote {len(df)} rows -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
