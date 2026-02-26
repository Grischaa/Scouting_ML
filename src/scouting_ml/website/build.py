from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
    "30-34": "30+",
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
    season_display: str


# --------------------------------------------------------------------------
# COLUMN SELECTION + CLEANING
# --------------------------------------------------------------------------

def select_columns(frame: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "name", "club", "position_group", "position_main", "age",
        "age_group", "market_value_eur", "link",
        "sofa_minutesPlayed", "sofa_goals", "sofa_assists",
        "sofa_expectedGoals", "sofa_totalDuelsWonPercentage",
        "sofa_accuratePassesPercentage",
        "player_id", "league", "season",
    ]

    available = [c for c in expected if c in frame.columns]
    missing = set(expected) - set(available)
    if missing:
        typer.echo(f"[warn] Missing columns: {', '.join(sorted(missing))}", err=True)

    frame = frame[available].copy()

    # Convert numeric columns safely
    for col in NUMERIC_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Fill empty cells so frontend doesn't break
    frame = frame.fillna("")

    return frame


# --------------------------------------------------------------------------
# ENRICHMENT FOR CONSISTENT FRONT-END FORMATTING
# --------------------------------------------------------------------------

def enrich_frame(frame: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    frame = frame.copy()

    # Age band normalization
    if "age_group" in frame.columns:
        frame["age_band"] = (
            frame["age_group"].map(AGE_GROUP_MAP).fillna(frame["age_group"]).fillna("")
        )
    else:
        frame["age_band"] = ""

    # Round age
    if "age" in frame.columns:
        frame["age"] = pd.to_numeric(frame["age"], errors="coerce").round(1)

    # Round specific columns
    rounding_map = {
        "sofa_expectedGoals": 2,
        "sofa_totalDuelsWonPercentage": 2,
        "sofa_accuratePassesPercentage": 2,
    }
    for col, digits in rounding_map.items():
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").round(digits)

    # Metadata fallback
    frame["league"] = league
    frame["season"] = season

    return frame


# --------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------

def pretty_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


# --------------------------------------------------------------------------
# MULTI-SEASON SUPPORT
# --------------------------------------------------------------------------

def list_all_datasets_for_league(config: LeagueConfig) -> List[DatasetSpec]:
    """
    Automatically finds all processed CSV files for a league.
    Expected naming pattern: <league_slug>_<season>.csv
    """

    # If league has a dedicated processed directory
    processed_dir = config.processed_dir
    if not processed_dir.exists():
        return []

    specs: List[DatasetSpec] = []

    for csv in processed_dir.glob("*.csv"):
        # Extract season from filename
        stem = csv.stem  # e.g. bundesliga_2023
        parts = stem.split("_")
        season = parts[-1]

        # Skip if season cannot be inferred
        if not season:
            continue

        specs.append(DatasetSpec(config=config, source=csv, season_display=season))

    return specs


# --------------------------------------------------------------------------
# LOADING + PACKAGING
# --------------------------------------------------------------------------

def load_dataset(spec: DatasetSpec) -> dict:
    typer.echo(f"[build] Reading {spec.config.name} ({spec.season_display}) from {pretty_path(spec.source)}")

    df = pd.read_csv(spec.source)
    df = select_columns(df)
    df = enrich_frame(df, league=spec.config.name, season=spec.season_display)

    records = df.to_dict(orient="records")
    typer.echo(f"[build] → {spec.config.name} {spec.season_display}: {len(records)} players")

    return {
        "league_slug": spec.config.slug,
        "season": spec.season_display,
        "players": records,
        "meta": {
            "league": spec.config.name,
            "season": spec.season_display,
            "source": pretty_path(spec.source),
        },
    }


def to_payload(datasets: List[dict]) -> str:
    """
    Nest structure:
    SCOUTING_DATA = {
        "bundesliga": {
            "meta": {...},
            "seasons": {
                "2021": {...},
                "2022": {...}
            }
        }
    }
    """

    data_map: dict[str, dict] = {}

    for ds in datasets:
        slug = ds["league_slug"]
        season = ds["season"]

        if slug not in data_map:
            data_map[slug] = {
                "meta": {"slug": slug, "name": ds["meta"]["league"]},
                "seasons": {}
            }

        data_map[slug]["seasons"][season] = {
            "players": ds["players"],
            "meta": ds["meta"],
        }

    leagues_list = [
        {
            "slug": slug,
            "name": info["meta"]["name"],
            "seasons": sorted(info["seasons"].keys())
        }
        for slug, info in data_map.items()
    ]

    default_slug = leagues_list[0]["slug"] if leagues_list else None

    return (
        "window.SCOUTING_DATA = "
        + json.dumps(data_map, ensure_ascii=False, indent=2)
        + ";\n"
        + "window.SCOUTING_LEAGUES = "
        + json.dumps(leagues_list, ensure_ascii=False, indent=2)
        + ";\n"
        + "window.SCOUTING_DEFAULT_LEAGUE = "
        + json.dumps(default_slug, ensure_ascii=False)
        + ";\n"
    )


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main(
    leagues: List[str] = typer.Option(
        None,
        "--league",
        "-l",
        help="League slugs to include. Defaults to all leagues in registry."
    ),
    destination: Path = typer.Option(
        DEFAULT_DEST,
        "--dest",
        "-d",
        help="Destination JS bundle."
    ),
) -> None:

    # Determine which leagues to process
    if leagues:
        configs = [get_league(slug) for slug in leagues]
    else:
        configs = list_leagues()

    # Build full spec list for all seasons
    all_specs: List[DatasetSpec] = []
    for config in configs:
        specs = list_all_datasets_for_league(config)
        if not specs:
            typer.echo(f"[warn] No datasets found for {config.name}", err=True)
            continue
        all_specs.extend(specs)

    if not all_specs:
        raise typer.BadParameter("No datasets available. Please ensure processed CSVs exist.")

    # Load datasets
    datasets = [load_dataset(spec) for spec in all_specs]

    # Write JS bundle
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = to_payload(datasets)
    destination.write_text(payload, encoding="utf-8")

    typer.echo(f"\n[build] Wrote {len(datasets)} season datasets → {pretty_path(destination)}")


if __name__ == "__main__":
    typer.run(main)
