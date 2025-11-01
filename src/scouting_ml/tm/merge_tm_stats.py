from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from scouting_ml.league_registry import get_league, season_slug, slugify
from scouting_ml.utils.import_guard import *  # noqa: F403


def read_many(files: Iterable[Path]) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:  # pragma: no cover - resilience logging
            print(f"[merge] Skipped {path} ({exc})")
    return frames


def merge_stats(
    directory: Path,
    *,
    pattern: str = "*_team_stats.csv",
    fallback_pattern: str = "*_team_players.csv",
) -> pd.DataFrame | None:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        candidates = sorted(directory.glob(fallback_pattern))

    if not candidates:
        print(
            f"[merge] No files found in {directory} matching {pattern} "
            f"or fallback {fallback_pattern}"
        )
        return None

    frames = read_many(candidates)
    if not frames:
        print("[merge] No readable CSVs to merge.")
        return None

    merged = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[merge] Merged {len(frames)} files ({len(merged)} rows)")
    return merged


def derive_output_path(
    *,
    directory: Path,
    league_label: str | None,
    season_label: str | None,
) -> Path:
    slug = slugify(league_label or "league")
    season_part = season_slug(season_label or "")
    if season_part:
        name = f"{slug}_{season_part}_full_stats.csv"
    else:
        name = f"{slug}_full_stats.csv"
    return directory / name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Transfermarkt derived stats across clubs for a league."
    )
    parser.add_argument(
        "--directory",
        default="data/processed",
        help="Directory housing *_team_stats.csv files.",
    )
    parser.add_argument(
        "--league",
        help="League slug from league_registry or free-form label for output naming.",
    )
    parser.add_argument(
        "--season",
        help="Season label used for output naming (defaults to registry season if available).",
    )
    parser.add_argument(
        "--outfile",
        help="Optional explicit output path for the merged CSV.",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    merged = merge_stats(directory)
    if merged is None:
        return

    league_label = args.league
    season_label = args.season

    if args.league:
        try:
            config = get_league(args.league)
        except KeyError:
            config = None
        if config:
            league_label = config.name
            season_label = season_label or config.tm_season_label or config.sofa_season_label

    if args.outfile:
        out_path = Path(args.outfile)
    else:
        out_path = derive_output_path(
            directory=directory,
            league_label=league_label,
            season_label=season_label,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[merge] Wrote league stats -> {out_path.resolve()}")


if __name__ == "__main__":
    main()

