from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _get_json(url: str) -> object:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def download_statsbomb_open_data_slice(
    *,
    output_root: str,
    competition_id: int,
    season_id: int,
    max_matches: int | None = None,
    include_lineups: bool = False,
) -> dict[str, object]:
    out_root = Path(output_root)
    events_dir = out_root / "events"
    matches_dir = out_root / "matches" / str(competition_id)
    lineups_dir = out_root / "lineups"
    events_dir.mkdir(parents=True, exist_ok=True)
    matches_dir.mkdir(parents=True, exist_ok=True)
    if include_lineups:
        lineups_dir.mkdir(parents=True, exist_ok=True)

    competitions_url = f"{RAW_BASE}/competitions.json"
    competitions = _get_json(competitions_url)
    (out_root / "competitions.json").write_text(json.dumps(competitions), encoding="utf-8")

    matches_url = f"{RAW_BASE}/matches/{competition_id}/{season_id}.json"
    matches = _get_json(matches_url)
    if not isinstance(matches, list):
        raise ValueError(f"Unexpected StatsBomb matches payload for competition={competition_id} season={season_id}")
    selected = matches[: max_matches or len(matches)]
    (matches_dir / f"{season_id}.json").write_text(json.dumps(selected), encoding="utf-8")

    downloaded = 0
    for item in selected:
        if not isinstance(item, dict):
            continue
        match_id = item.get("match_id")
        if match_id is None:
            continue
        event_url = f"{RAW_BASE}/events/{match_id}.json"
        event_payload = _get_json(event_url)
        (events_dir / f"{match_id}.json").write_text(json.dumps(event_payload), encoding="utf-8")
        if include_lineups:
            lineup_url = f"{RAW_BASE}/lineups/{match_id}.json"
            lineup_payload = _get_json(lineup_url)
            (lineups_dir / f"{match_id}.json").write_text(json.dumps(lineup_payload), encoding="utf-8")
        downloaded += 1

    return {
        "output_root": str(out_root),
        "competition_id": int(competition_id),
        "season_id": int(season_id),
        "match_count": int(len(selected)),
        "events_downloaded": int(downloaded),
        "include_lineups": bool(include_lineups),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small StatsBomb open-data slice for one competition-season.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--competition-id", type=int, required=True)
    parser.add_argument("--season-id", type=int, required=True)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--include-lineups", action="store_true")
    args = parser.parse_args()

    payload = download_statsbomb_open_data_slice(
        output_root=args.output_root,
        competition_id=args.competition_id,
        season_id=args.season_id,
        max_matches=args.max_matches,
        include_lineups=args.include_lineups,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
