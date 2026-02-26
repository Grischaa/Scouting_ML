from __future__ import annotations

from pathlib import Path

from scouting_ml.league_registry import LeagueConfig


def merge_tm_sofa(
    config: LeagueConfig,
    season: str,
    tm_clean_path: Path,
    sofa_path: Path,
    *,
    force: bool = False,
) -> Path:
    """
    Merge the cleaned Transfermarkt dataset with Sofascore metrics.
    """

    output_path = config.guess_processed_dataset(season)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"[merge] Using cached merged dataset {output_path}")
        return output_path

    print(f"[merge] Combining TM + Sofascore → {output_path}")
    from scouting_ml.tm.merge_tm_sofa import merge_tm_sofa as _merge  # lazy import

    _merge(
        tm_path=str(tm_clean_path),
        sofa_path=str(sofa_path),
        out_path=str(output_path),
    )
    return output_path
