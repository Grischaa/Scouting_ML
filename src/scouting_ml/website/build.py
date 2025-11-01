from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import typer

from scouting_ml.league_registry import LeagueConfig, get_league, list_leagues
from scouting_ml.paths import PROJECT_ROOT


DEFAULT_DEST = Path(__file__).resolve().parent / "static" / "data" / "players.js"

AGE_GROUP_MAP = {
    "<18": "u18",
    "18-22": "18-22",
    "23-26": "23-26",
    "27-30": "27-30",
    "30+": "30+",
    "30-34": "30+",  # fallback for legacy groupings
    "35+": "30+",
}

NUMERIC_COLUMNS = [
    "age",
    "market_value_eur",
    "sofa_minutesPlayed",
    "sofa_goals",
    "sofa_assists",
    "sofa_expectedGoals",
    "sofa_totalDuelsWonPercentage",
    "sofa_accuratePassesPercentage",
]


@dataclass
class DatasetSpec:
    config: LeagueConfig
    source: Path
    season_display: str | None


def select_columns(frame: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "name",
        "club",
        "position_group",
        "position_main",
        "age",
        "age_group",
        "market_value_eur",
        "link",
        "sofa_minutesPlayed",
        "sofa_goals",
        "sofa_assists",
        "sofa_expectedGoals",
        "sofa_totalDuelsWonPercentage",
        "sofa_accuratePassesPercentage",
        "player_id",
        "league",
        "season",
    ]

    available = [column for column in expected_columns if column in frame.columns]
    missing = set(expected_columns) - set(available)
    if missing:
        typer.echo(
            f"[warn] Missing columns in source data: {', '.join(sorted(missing))}",
            err=True,
        )

    subset = frame[available].copy()

    for column in NUMERIC_COLUMNS:
        if column in subset.columns:
            subset[column] = pd.to_numeric(subset[column], errors="coerce")

    return subset


def enrich_frame(frame: pd.DataFrame, league_label: str, season_label: str | None) -> pd.DataFrame:
    frame = frame.copy()

    if "age_group" in frame.columns:
        frame["age_band"] = (
            frame["age_group"]
            .map(AGE_GROUP_MAP)
            .fillna(frame["age_group"])
            .fillna("")
        )
    else:
        frame["age_band"] = ""

    if "age" in frame.columns:
        frame["age"] = frame["age"].round(1)

    decimal_cols = {
        "sofa_expectedGoals": 2,
        "sofa_totalDuelsWonPercentage": 2,
        "sofa_accuratePassesPercentage": 2,
    }
    for column, digits in decimal_cols.items():
        if column in frame.columns:
            frame[column] = frame[column].round(digits)

    # ensure metadata fallback so the front-end can rely on it
    if "league" not in frame.columns or frame["league"].isna().all():
        frame["league"] = league_label
    else:
        frame["league"] = frame["league"].fillna(league_label)

    if season_label:
        if "season" not in frame.columns or frame["season"].isna().all():
            frame["season"] = season_label
        else:
            frame["season"] = frame["season"].fillna(season_label)

    return frame


def pretty_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_dataset(spec: DatasetSpec, *, limit: int | None) -> dict:
    typer.echo(f"[build] Reading dataset from {pretty_path(spec.source)}")
    frame = pd.read_csv(spec.source)
    frame = select_columns(frame)

    if limit is not None:
        frame = frame.head(limit)

    frame = enrich_frame(frame, league_label=spec.config.name, season_label=spec.season_display)

    records = frame.to_dict(orient="records")
    typer.echo(f"[build] Prepared {len(records)} players for {spec.config.name}")
    return {
        "slug": spec.config.slug,
        "meta": {
            "slug": spec.config.slug,
            "name": spec.config.name,
            "season": spec.season_display,
            "source": pretty_path(spec.source),
        },
        "players": records,
    }


def to_payload(datasets: List[dict]) -> str:
    data_map = {item["slug"]: {"meta": item["meta"], "players": item["players"]} for item in datasets}
    league_list = [
        {"slug": item["slug"], "name": item["meta"]["name"], "season": item["meta"]["season"]}
        for item in datasets
    ]
    default_slug = league_list[0]["slug"] if league_list else None
    return (
        "window.SCOUTING_DATA = "
        + json.dumps(data_map, ensure_ascii=False, indent=2)
        + ";\n"
        + "window.SCOUTING_LEAGUES = "
        + json.dumps(league_list, ensure_ascii=False, indent=2)
        + ";\n"
        + "window.SCOUTING_DEFAULT_LEAGUE = "
        + json.dumps(default_slug, ensure_ascii=False)
        + ";\n"
    )


def collect_specs(
    leagues: Optional[List[str]] = None,
    *,
    season_override: str | None = None,
) -> List[DatasetSpec]:
    if leagues:
        configs = []
        for slug in leagues:
            try:
                configs.append(get_league(slug))
            except KeyError as exc:
                raise typer.BadParameter(str(exc))
    else:
        configs = list_leagues()

    specs: List[DatasetSpec] = []
    for config in configs:
        season_label = season_override or config.tm_season_label or config.sofa_season_label
        source = config.guess_processed_dataset(season_label)

        if source is None or not source.exists():
            expected = pretty_path(source) if source else "unknown"
            typer.echo(
                f"[warn] Skipping {config.name}: dataset not found "
                f"(expected {expected})",
                err=True,
            )
            continue

        specs.append(
            DatasetSpec(
                config=config,
                source=source,
                season_display=season_label,
            )
        )

    if not specs:
        raise typer.BadParameter("No datasets available. Ensure processed CSVs exist or update the registry.")

    return specs


def main(
    leagues: List[str] = typer.Option(
        None,
        "--league",
        "-l",
        help="League slugs to include (defaults to all registry entries).",
    ),
    season: str = typer.Option(
        None,
        "--season",
        help="Override season label for all selected leagues.",
    ),
    destination: Path = typer.Option(
        DEFAULT_DEST,
        "--dest",
        "-d",
        help="Destination JavaScript bundle.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-n",
        min=1,
        help="Optionally restrict the number of players exported per league.",
    ),
) -> None:
    specs = collect_specs(leagues, season_override=season)
    datasets = [load_dataset(spec, limit=limit) for spec in specs]

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = to_payload(datasets)
    destination.write_text(payload, encoding="utf-8")
    typer.echo(f"[build] Wrote {len(datasets)} league payload(s) → {pretty_path(destination)}")


if __name__ == "__main__":
    typer.run(main)
