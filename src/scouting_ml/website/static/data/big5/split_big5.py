import pandas as pd
import os

# Path to your raw CSV
INPUT = r"C:\Users\chris\Scout_Pred\data\raw\big5_players.csv"

# Output folder for website-only datasets
OUTPUT = r"C:\Users\chris\Scout_Pred\src\scouting_ml\website\static\data\big5"
os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv(INPUT)

slug_map = {
    "English Premier League": "premier_league",
    "Ligue 1": "ligue_1",
    "Bundesliga": "bundesliga",
    "Serie A": "serie_a",
    "LaLiga": "laliga"
}

for league_name, slug in slug_map.items():
    subset = df[df["league"] == league_name]
    if len(subset) == 0:
        print(f"No rows for {league_name}")
        continue
    out_path = os.path.join(OUTPUT, f"{slug}_dataset.csv")
    subset.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
