from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from ScraperFC import sofascore as sofa_module
from ScraperFC.sofascore import Sofascore

from scouting_ml.league_registry import (
    LeagueConfig,
    get_league,
    list_leagues,
    season_slug,
    slugify,
)
from scouting_ml.paths import PROCESSED_DIR, ensure_dirs
from scouting_ml.utils.import_guard import *  # noqa: F403


app = typer.Typer(
    add_completion=False,
    help="Scrape Sofascore aggregated player stats for a given league/season.",
)


def ensure_league_registered(league_key: str, tournament_id: int | None) -> None:
    if not league_key:
        raise typer.BadParameter("Sofascore league key is required.")
    if tournament_id is None:
        raise typer.BadParameter("Unique tournament id is required to register the league.")
    if league_key not in sofa_module.comps:
        sofa_module.comps[league_key] = tournament_id


def resolve_league(league: str) -> LeagueConfig | None:
    try:
        return get_league(league)
    except KeyError:
        return None


@app.command("list")
def list_registered() -> None:
    """List available league configs bundled with the project."""
    for league in list_leagues():
        typer.echo(f"{league.slug:>24}  {league.name}")


@app.command("pull")
def pull(
    league: str = typer.Argument(
        "austrian_bundesliga",
        help="League slug (from registry) or free-form label.",
    ),
    season: str = typer.Option(
        None,
        "--season",
        "-s",
        help="Display season label (e.g. '2025/26'). Defaults to registry value.",
    ),
    sofa_season: str = typer.Option(
        None,
        "--sofa-season",
        help="Sofascore season label (e.g. '24/25' or '2024'). Defaults to registry value.",
    ),
    sofa_league_key: str = typer.Option(
        None,
        "--league-key",
        help="Label used by Sofascore (defaults to registry or league name).",
    ),
    tournament_id: int = typer.Option(
        None,
        "--tournament-id",
        help="Sofascore unique tournament id. Required if league is not in registry.",
    ),
    accumulation: str = typer.Option(
        "total",
        help="Aggregation mode passed to Sofascore (total or per90).",
    ),
    positions: List[str] = typer.Option(
        ["Goalkeepers", "Defenders", "Midfielders", "Forwards"],
        "--position",
        "-p",
        help="Positions to include",
    ),
    outfile: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Where to write the CSV (defaults based on league/season).",
    ),
) -> None:
    """Download aggregated Sofascore league statistics to the processed directory."""
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

    ensure_league_registered(league_key, tournament)

    sofa = Sofascore()
    seasons = sofa.get_valid_seasons(league_key)
    if sofa_label not in seasons:
        available = ", ".join(sorted(seasons.keys(), reverse=True))
        raise typer.BadParameter(
            f"Season '{sofa_label}' not available for {league_key}. "
            f"Available: {available}"
        )

    typer.echo(
        f"[sofa] Fetching {league_name} ({league_key}) season {sofa_label} → accumulation={accumulation}"
    )

    df = sofa.scrape_player_league_stats(
        year=sofa_label,
        league=league_key,
        accumulation=accumulation,
        selected_positions=positions,
    )

    if outfile is None:
        season_part = season_slug(sofa_label or tm_season or "")
        outfile = PROCESSED_DIR / f"sofa_{slug}_{season_part}.csv"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile, index=False)

    typer.echo(f"[sofa] Saved {len(df)} rows → {outfile.resolve()}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

