from __future__ import annotations

from typing import Iterable, List, Optional

import typer

from scouting_ml.league_registry import LEAGUES, get_league, list_leagues
from scouting_ml.pipeline import merge_tm_sofa, run_sofascore, run_transfermarkt


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


app = typer.Typer(help="Refresh Transfermarkt + Sofascore datasets for configured leagues.")


@app.command()
def league(
    slug: str = typer.Argument(..., help="League slug, e.g. english_premier_league."),
    season: List[str] = typer.Option(
        None,
        "--season",
        "-s",
        help="Season label(s) to refresh (defaults to all seasons configured for the league).",
    ),
    force: bool = typer.Option(False, help="Redownload even if files already exist."),
) -> None:
    try:
        refresh_league(slug, seasons=season or None, force=force)
    except ValueError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(1)


@app.command()
def all(
    force: bool = typer.Option(False, help="Redownload even if files already exist."),
    max_seasons: Optional[int] = typer.Option(
        None,
        "--max-seasons",
        "-m",
        help="Limit to the first N seasons per league.",
    ),
) -> None:
    for config in list_leagues():
        seasons = config.seasons[:max_seasons] if max_seasons else None
        try:
            refresh_league(config.slug, seasons=seasons, force=force)
        except ValueError as exc:
            typer.echo(f"[error] {exc}", err=True)
            raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
