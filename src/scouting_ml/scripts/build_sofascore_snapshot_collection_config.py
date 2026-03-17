from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

from scouting_ml.league_registry import LEAGUES, LeagueConfig, get_league


DEFAULT_NON_BIG5_SLUGS = [
    "portuguese_primeira_liga",
    "dutch_eredivisie",
    "belgian_pro_league",
    "turkish_super_lig",
    "scottish_premiership",
    "greek_super_league",
    "austrian_bundesliga",
]


def _csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _league_slugs_for_mode(mode: str) -> list[str]:
    if mode == "non_big5":
        return [slug for slug in DEFAULT_NON_BIG5_SLUGS if slug in LEAGUES]
    if mode == "all":
        return sorted(LEAGUES.keys())
    raise ValueError(f"Unsupported mode: {mode}")


def _has_processed_dataset(config: LeagueConfig, season: str) -> bool:
    return config.guess_processed_dataset(season).exists()


def _competition_payload(
    *,
    config: LeagueConfig,
    season: str,
    use_team_schedule: bool,
    max_pages: int,
    team_schedule_max_pages: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": f"{config.slug}_{season.replace('/', '_')}",
        "league": config.name,
        "season": season,
        "tournament_id": int(config.sofa_tournament_id),
        "segments": ["last"],
        "max_pages": int(max_pages),
    }
    if use_team_schedule:
        payload["events_mode"] = "team_schedule"
        payload["team_schedule_max_pages"] = int(team_schedule_max_pages)
    return payload


def build_sofascore_snapshot_collection_config(
    *,
    season: str,
    league_slugs: Iterable[str],
    output_path: str | Path,
    players_source: str | Path = "data/processed",
    raw_output_dir: str | Path = "data/raw/providers",
    provider_config_out: str | Path | None = None,
    summary_out: str | Path | None = None,
    base_url: str = "",
    rps: float = 1.0,
    retries: int = 6,
    backoff: float = 1.0,
    use_team_schedule: bool = True,
    max_pages: int = 30,
    team_schedule_max_pages: int = 8,
    include_statsbomb: bool = False,
    statsbomb_open_data_root: str = "data/raw/statsbomb/bundesliga_2023_24",
    statsbomb_competition_ids: list[int] | None = None,
    statsbomb_season_ids: list[int] | None = None,
    require_processed_dataset: bool = True,
) -> dict[str, object]:
    season = str(season).strip()
    if not season:
        raise ValueError("season is required")

    selected_configs: list[LeagueConfig] = []
    missing: list[str] = []
    unavailable: list[str] = []
    for slug in league_slugs:
        config = get_league(str(slug).strip())
        if season not in config.seasons:
            missing.append(config.slug)
            continue
        if require_processed_dataset and not _has_processed_dataset(config, season):
            unavailable.append(config.slug)
            continue
        selected_configs.append(config)

    if not selected_configs:
        detail = []
        if missing:
            detail.append(f"season_missing={','.join(sorted(missing))}")
        if unavailable:
            detail.append(f"processed_missing={','.join(sorted(unavailable))}")
        raise ValueError("No eligible leagues selected" + (f" ({'; '.join(detail)})" if detail else ""))

    output_path = Path(output_path)
    raw_output_dir = Path(raw_output_dir)
    provider_config_out = Path(provider_config_out) if provider_config_out else raw_output_dir / f"sofascore_provider_pipeline_{season.replace('/', '-')}.generated.json"
    summary_out = Path(summary_out) if summary_out else raw_output_dir / f"sofascore_snapshot_summary_{season.replace('/', '-')}.json"

    competitions = [
        _competition_payload(
            config=config,
            season=season,
            use_team_schedule=use_team_schedule,
            max_pages=max_pages,
            team_schedule_max_pages=team_schedule_max_pages,
        )
        for config in selected_configs
    ]

    payload: dict[str, object] = {
        "snapshot_date": datetime.now(timezone.utc).date().isoformat(),
    }
    payload["raw_output_dir"] = str(raw_output_dir)
    payload["provider_config_out"] = str(provider_config_out)
    payload["summary_out"] = str(summary_out)
    payload["players_source"] = str(players_source)
    if base_url:
        payload["base_url"] = base_url
        payload["rps"] = float(rps)
        payload["retries"] = int(retries)
        payload["backoff"] = float(backoff)
    if include_statsbomb:
        payload["statsbomb"] = {
            "open_data_root": statsbomb_open_data_root,
            "competition_ids": statsbomb_competition_ids or [9],
            "season_ids": statsbomb_season_ids or [281],
        }
    payload["fixture_context"] = {"provider": "sofascore", "competitions": competitions}
    payload["player_availability"] = {"provider": "sofascore", "competitions": competitions}
    if missing:
        payload["season_missing_slugs"] = sorted(missing)
    if unavailable:
        payload["processed_missing_slugs"] = sorted(unavailable)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a multi-league SofaScore snapshot-collection config from the league registry."
    )
    parser.add_argument("--season", required=True, help="Season label, e.g. 2024/25 or 2025.")
    parser.add_argument(
        "--leagues",
        default="",
        help="Comma-separated league slugs. If omitted, use --mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["non_big5", "all"],
        default="non_big5",
        help="Default league set when --leagues is omitted.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/providers/sofascore_snapshot_collection.non_big5_2024-25.generated.json",
        help="Output JSON path.",
    )
    parser.add_argument("--players-source", default="data/processed")
    parser.add_argument("--raw-output-dir", default="data/raw/providers")
    parser.add_argument("--provider-config-out", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional Sofa base URL. Set to https://sofascore.p.rapidapi.com for RapidAPI mode.",
    )
    parser.add_argument("--rps", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--backoff", type=float, default=1.0)
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--team-schedule-max-pages", type=int, default=8)
    parser.add_argument("--no-team-schedule", action="store_true")
    parser.add_argument("--include-statsbomb", action="store_true")
    parser.add_argument("--statsbomb-open-data-root", default="data/raw/statsbomb/bundesliga_2023_24")
    parser.add_argument("--statsbomb-competition-ids", default="")
    parser.add_argument("--statsbomb-season-ids", default="")
    parser.add_argument("--allow-missing-processed", action="store_true")
    args = parser.parse_args()

    league_slugs = _csv_list(args.leagues) or _league_slugs_for_mode(args.mode)
    statsbomb_competition_ids = [int(item) for item in _csv_list(args.statsbomb_competition_ids)] or None
    statsbomb_season_ids = [int(item) for item in _csv_list(args.statsbomb_season_ids)] or None

    payload = build_sofascore_snapshot_collection_config(
        season=args.season,
        league_slugs=league_slugs,
        output_path=args.output,
        players_source=args.players_source,
        raw_output_dir=args.raw_output_dir,
        provider_config_out=args.provider_config_out or None,
        summary_out=args.summary_out or None,
        base_url=args.base_url,
        rps=args.rps,
        retries=args.retries,
        backoff=args.backoff,
        use_team_schedule=not args.no_team_schedule,
        max_pages=args.max_pages,
        team_schedule_max_pages=args.team_schedule_max_pages,
        include_statsbomb=args.include_statsbomb,
        statsbomb_open_data_root=args.statsbomb_open_data_root,
        statsbomb_competition_ids=statsbomb_competition_ids,
        statsbomb_season_ids=statsbomb_season_ids,
        require_processed_dataset=not args.allow_missing_processed,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
