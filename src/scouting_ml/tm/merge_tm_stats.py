from pathlib import Path
import pandas as pd
import re

LEAGUE = "Austrian Bundesliga"
SEASON = "2025/26"

def _slug(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_.-]", "-", s)
    return s.lower()

stats_dir = Path("data/processed")

# try to merge stats first, then fall back to players
files = list(stats_dir.glob("*_team_stats.csv"))
if not files:
    files = list(stats_dir.glob("*_team_players.csv"))

if not files:
    print(f"[merge] No files found in {stats_dir}/ matching *_team_stats.csv or *_team_players.csv")
else:
    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            print(f"[merge] Skipped {p} ({e})")

    if not dfs:
        print("[merge] No readable CSVs to merge.")
    else:
        merged = pd.concat(dfs, ignore_index=True, sort=False)
        out = stats_dir / f"{_slug(LEAGUE)}_{_slug(SEASON)}_full_stats.csv"
        merged.to_csv(out, index=False)
        print(f"[merge] Wrote league stats -> {out.resolve()}")
