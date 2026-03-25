from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from scouting_ml.league_registry import (
    LeagueConfig,
    get_league,
    list_leagues,
    season_slug,
    slugify,
)
from scouting_ml.paths import PROCESSED_DIR, ensure_dirs
from scouting_ml.reporting.operator_health import write_json_sidecar
from scouting_ml.utils.import_guard import *  # noqa: F403


EMPTY_SOFA_COLUMNS = ["player", "team", "player id", "team id"]


def _load_sofascore_runtime():
    from ScraperFC import sofascore as sofa_module
    from ScraperFC.sofascore import Sofascore

    return sofa_module, Sofascore


def ensure_league_registered(league_key: str, tournament_id: int | None) -> None:
    if not league_key:
        raise ValueError("Sofascore league key is required.")
    if tournament_id is None:
        raise ValueError("Unique tournament id is required to register the league.")
    sofa_module, _ = _load_sofascore_runtime()
    existing = sofa_module.comps.get(league_key)
    if isinstance(existing, dict):
        existing["SOFASCORE"] = int(tournament_id)
        sofa_module.comps[league_key] = existing
        return
    if existing is None:
        sofa_module.comps[league_key] = {"SOFASCORE": int(tournament_id)}
        return
    sofa_module.comps[league_key] = {"SOFASCORE": int(tournament_id)}


def resolve_league(league: str) -> LeagueConfig | None:
    try:
        return get_league(league)
    except KeyError:
        return None


def list_registered() -> None:
    for league in list_leagues():
        print(f"{league.slug:>24}  {league.name}")


def _coerce_output_frame(df: object) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        if len(df.columns) == 0:
            return pd.DataFrame(columns=EMPTY_SOFA_COLUMNS)
        return df
    return pd.DataFrame(columns=EMPTY_SOFA_COLUMNS)


def _season_label_candidates(preferred: str | None, tm_season: str | None) -> list[str]:
    def add(target: list[str], seen: set[str], value: str | None) -> None:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            target.append(text)

    candidates: list[str] = []
    seen: set[str] = set()
    for raw in (preferred, tm_season):
        text = str(raw or "").strip()
        add(candidates, seen, text)
        if not text:
            continue
        match_four = re.match(r"^(\d{4})[/-](\d{2,4})$", text)
        if match_four:
            start = match_four.group(1)
            end = match_four.group(2)
            add(candidates, seen, f"{start}/{str(end)[-2:]}")
            add(candidates, seen, f"{start}-{str(end)[-2:]}")
            add(candidates, seen, start)
            continue
        match_two = re.match(r"^(\d{2})[/-](\d{2})$", text)
        if match_two:
            start = f"20{match_two.group(1)}"
            end = match_two.group(2)
            add(candidates, seen, f"{start}/{end}")
            add(candidates, seen, f"{start}-{end}")
            add(candidates, seen, start)
    return candidates


def _resolve_sofa_season_label(
    seasons: dict[str, object],
    *,
    preferred: str | None,
    tm_season: str | None,
) -> str | None:
    for candidate in _season_label_candidates(preferred, tm_season):
        if candidate in seasons:
            return candidate
    return None


def pull(
    league: str = "austrian_bundesliga",
    season: str | None = None,
    sofa_season: str | None = None,
    sofa_league_key: str | None = None,
    tournament_id: int | None = None,
    accumulation: str = "total",
    positions: Optional[List[str]] = None,
    outfile: Optional[Path] = None,
) -> None:
    ensure_dirs()

    config = resolve_league(league)
    slug = config.slug if config else slugify(league)

    league_name = config.name if config else league
    tm_season = season or (config.tm_season_label if config else None)
    sofa_label = sofa_season or (config.sofa_season_label if config else tm_season)
    league_key = sofa_league_key or (config.sofa_league_key if config else league_name)
    tournament = tournament_id if tournament_id is not None else (
        config.sofa_tournament_id if config else None
    )
    selected_positions = positions or ["Goalkeepers", "Defenders", "Midfielders", "Forwards"]

    ensure_league_registered(league_key, tournament)

    _, Sofascore = _load_sofascore_runtime()
    sofa = Sofascore()
    seasons = sofa.get_valid_seasons(league_key)
    resolved_sofa_label = _resolve_sofa_season_label(
        seasons,
        preferred=sofa_label,
        tm_season=tm_season,
    )
    if resolved_sofa_label is None:
        available = ", ".join(sorted(seasons.keys(), reverse=True))
        raise ValueError(
            f"Season '{sofa_label}' not available for {league_key}. "
            f"Available: {available}"
        )

    print(
        f"[sofa] Fetching {league_name} ({league_key}) season {resolved_sofa_label} → accumulation={accumulation}"
    )

    df = sofa.scrape_player_league_stats(
        year=resolved_sofa_label,
        league=league_key,
        accumulation=accumulation,
        selected_positions=selected_positions,
    )
    df = _coerce_output_frame(df)

    if outfile is None:
        season_part = season_slug(tm_season or resolved_sofa_label or "")
        outfile = PROCESSED_DIR / f"sofa_{slug}_{season_part}.csv"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)
    write_json_sidecar(
        outfile,
        {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "league_slug": slug,
            "league_name": league_name,
            "league_key": league_key,
            "tournament_id": int(tournament) if tournament is not None else None,
            "season": tm_season,
            "requested_sofa_season": sofa_label,
            "resolved_sofa_season": resolved_sofa_label,
            "rows": int(len(df)),
            "zero_rows": bool(df.empty),
            "header_only": bool(df.empty and len(df.columns) > 0),
            "columns": [str(col) for col in df.columns],
            "accumulation": accumulation,
            "selected_positions": list(selected_positions),
        },
    )

    if df.empty:
        print("[sofa] Warning: zero rows returned; wrote a header-only CSV so downstream merge can continue.")
    print(f"[sofa] Saved {len(df)} rows → {outfile.resolve()}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape Sofascore aggregated player stats for a given league/season."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available league configs bundled with the project.")

    pull_parser = subparsers.add_parser("pull", help="Download aggregated league stats.")
    pull_parser.add_argument(
        "league",
        nargs="?",
        default="austrian_bundesliga",
        help="League slug (from registry) or free-form label.",
    )
    pull_parser.add_argument(
        "--season",
        "-s",
        default=None,
        help="Display season label (e.g. '2025/26'). Defaults to registry value.",
    )
    pull_parser.add_argument(
        "--sofa-season",
        default=None,
        help="Sofascore season label (e.g. '24/25' or '2024'). Defaults to registry value.",
    )
    pull_parser.add_argument(
        "--league-key",
        default=None,
        help="Label used by Sofascore (defaults to registry or league name).",
    )
    pull_parser.add_argument(
        "--tournament-id",
        type=int,
        default=None,
        help="Sofascore unique tournament id. Required if league is not in registry.",
    )
    pull_parser.add_argument(
        "--accumulation",
        default="total",
        help="Aggregation mode passed to Sofascore (total or per90).",
    )
    pull_parser.add_argument(
        "--position",
        "-p",
        action="append",
        default=[],
        help="Position to include. Repeat the flag to restrict positions.",
    )
    pull_parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Where to write the CSV (defaults based on league/season).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "list":
            list_registered()
            return
        if args.command == "pull":
            pull(
                league=args.league,
                season=args.season,
                sofa_season=args.sofa_season,
                sofa_league_key=args.league_key,
                tournament_id=args.tournament_id,
                accumulation=args.accumulation,
                positions=args.position or None,
                outfile=Path(args.out) if args.out else None,
            )
            return
        parser.error(f"Unknown command: {args.command}")
    except ValueError as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

