from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scouting_ml.paths import PROCESSED_DIR
from scouting_ml.league_registry import LeagueConfig, season_slug_label


def _has_csv_header(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            return bool(handle.readline().strip())
    except OSError:
        return False


def run_sofascore(
    config: LeagueConfig,
    season: str,
    *,
    force: bool = False,
    python_executable: str | None = None,
) -> Path:
    """
    Pull Sofascore aggregated stats for a league season.
    """

    python_cmd = python_executable or sys.executable
    season_slug = season_slug_label(season)
    out_path = PROCESSED_DIR / f"sofa_{config.slug}_{season_slug}.csv"

    if out_path.exists() and not force:
        if _has_csv_header(out_path):
            print(f"[sofa] Using cached payload {out_path}")
            return out_path
        print(f"[sofa] Cached payload {out_path} is empty; rebuilding.")

    sofa_label = config.sofa_season_map[season]
    cmd = [
        python_cmd,
        "-m",
        "scouting_ml.sofa.league_pull",
        "pull",
        config.sofa_league_key,
        "--league-key",
        config.sofa_league_key,
        "--tournament-id",
        str(config.sofa_tournament_id),
        "--season",
        season,
        "--sofa-season",
        sofa_label,
        "--out",
        str(out_path),
    ]
    print(f"[sofa] pull :: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"[sofa] Sofascore pull failed with exit code {result.returncode}")

    return out_path
