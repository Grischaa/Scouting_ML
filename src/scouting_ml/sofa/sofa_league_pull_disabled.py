# src/scouting_ml/sofa_league_pull.py
from pathlib import Path
import pandas as pd
from ScraperFC.sofascore import Sofascore
from scouting_ml.utils.import_guard import *  # noqa: F403
from scouting_ml.paths import ensure_dirs

def main():
    ensure_dirs()
    sofa = Sofascore()

    # TODO: adjust to exact code in scraperfc/comps.yaml
    league = "austrian-bundesliga"   # check name
    year = "2024-2025"               # or "2025-2026" if available

    df = sofa.scrape_player_league_stats(
        year=year,
        league=league,
        accumulation="total",
        selected_positions=["Goalkeepers", "Defenders", "Midfielders", "Forwards"],
    )

    out = Path("data/processed/sofa_player_league_stats.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[sofa_league_pull] wrote {len(df)} rows -> {out.resolve()}")

if __name__ == "__main__":
    main()
