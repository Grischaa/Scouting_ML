from __future__ import annotations

import argparse
from typing import Iterable, List, Optional

from scouting_ml.league_registry import get_league, list_leagues
from scouting_ml.pipeline import merge_tm_sofa, run_sofascore, run_transfermarkt
from scouting_ml.reporting.operator_health import regenerate_ingestion_health_report


def refresh_league(
    slug: str,
    *,
    seasons: Optional[Iterable[str]] = None,
    force: bool = False,
    python_executable: str | None = None,
) -> None:
    config = get_league(slug)
    seasons_to_run: List[str] = list(seasons) if seasons else list(config.seasons)
    missing = [season for season in seasons_to_run if season not in config.seasons]
    if missing:
        raise ValueError(
            f"{config.name}: no metadata for season(s) {', '.join(missing)}. "
            "Update scouting_ml.league_registry to include them."
        )

    for season in seasons_to_run:
        print(f"[refresh] {config.name} :: {season}")
        tm_result = run_transfermarkt(config, season, force=force, python_executable=python_executable)
        sofa_path = run_sofascore(
            config,
            season,
            force=force,
            python_executable=python_executable,
        )
        merge_tm_sofa(
            config,
            season,
            tm_clean_path=tm_result.clean_path,
            sofa_path=sofa_path,
            force=force,
        )
    report = regenerate_ingestion_health_report()
    print(f"[refresh] updated ingestion health report -> {report['_meta']['json_path']}")

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh Transfermarkt + Sofascore datasets for configured leagues."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    league_parser = subparsers.add_parser(
        "league",
        help="Refresh one configured league.",
    )
    league_parser.add_argument(
        "slug",
        help="League slug, e.g. english_premier_league.",
    )
    league_parser.add_argument(
        "--season",
        "-s",
        action="append",
        default=[],
        help="Season label to refresh. Repeat the flag to run multiple seasons.",
    )
    league_parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if files already exist.",
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Refresh every configured league.",
    )
    all_parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if files already exist.",
    )
    all_parser.add_argument(
        "--max-seasons",
        "-m",
        type=int,
        default=None,
        help="Limit to the first N seasons per league.",
    )
    return parser


def _run_league_command(slug: str, seasons: list[str], force: bool) -> int:
    try:
        refresh_league(slug, seasons=seasons or None, force=force)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1
    return 0


def _run_all_command(force: bool, max_seasons: Optional[int]) -> int:
    for config in list_leagues():
        seasons = config.seasons[:max_seasons] if max_seasons else None
        try:
            refresh_league(config.slug, seasons=seasons, force=force)
        except ValueError as exc:
            print(f"[error] {exc}")
            return 1
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "league":
        raise SystemExit(_run_league_command(args.slug, args.season, args.force))
    if args.command == "all":
        raise SystemExit(_run_all_command(args.force, args.max_seasons))
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
