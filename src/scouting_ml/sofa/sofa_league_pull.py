# src/scouting_ml/sofa_league_pull.py
from pathlib import Path

import pandas as pd
from ScraperFC import sofascore as sofa_module
from ScraperFC.sofascore import Sofascore

from scouting_ml.utils.import_guard import *  # noqa: F403
from scouting_ml.paths import ensure_dirs

AUSTRIAN_UNIQUE_TOURNAMENT_ID = 45
AUSTRIAN_LEAGUE_KEY = "Austrian Bundesliga"
DEFAULT_SEASON_LABEL = "24/25"  # Sofascore season label (see get_valid_seasons output)


def ensure_league_registered() -> None:
    """Register the Austrian Bundesliga with ScraperFC's Sofascore league map if missing."""
    if AUSTRIAN_LEAGUE_KEY not in sofa_module.comps:
        sofa_module.comps[AUSTRIAN_LEAGUE_KEY] = AUSTRIAN_UNIQUE_TOURNAMENT_ID


def main(season_label: str = DEFAULT_SEASON_LABEL) -> None:
    ensure_dirs()
    ensure_league_registered()

    sofa = Sofascore()
    seasons = sofa.get_valid_seasons(AUSTRIAN_LEAGUE_KEY)
    if season_label not in seasons:
        raise ValueError(
            f"Season '{season_label}' not available. Pick one of: {', '.join(sorted(seasons.keys(), reverse=True))}"
        )

    df = sofa.scrape_player_league_stats(
        year=season_label,
        league=AUSTRIAN_LEAGUE_KEY,
        accumulation="total",
        selected_positions=["Goalkeepers", "Defenders", "Midfielders", "Forwards"],
    )

    season_slug = season_label.replace("/", "-").replace(" ", "_")
    out = Path(f"data/processed/sofa_{AUSTRIAN_LEAGUE_KEY.lower().replace(' ', '_')}_{season_slug}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[sofa_league_pull] {AUSTRIAN_LEAGUE_KEY} {season_label} -> {out.resolve()}")


if __name__ == "__main__":
    import sys

    season_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SEASON_LABEL
    main(season_arg)
