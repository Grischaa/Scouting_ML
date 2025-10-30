from pathlib import Path
import pandas as pd
import re

def _slug(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_.-]", "-", s)
    return s.lower()

stats_dir = Path("data/processed")
files = list(stats_dir.glob("*_team_players.csv"))


if not files:
    print("[merge] No stats files found in data/interim/tm/")
else:
    dfs = [pd.read_csv(p) for p in files]
    merged = pd.concat(dfs, ignore_index=True)
    season = "2025/26"
    out = stats_dir / f"{_slug('Austrian Bundesliga')}_{_slug(season)}_full_stats.csv"
    merged.to_csv(out, index=False)
    print(f"[merge] Wrote league stats -> {out.resolve()}")
