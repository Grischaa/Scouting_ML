import pandas as pd
from scouting_ml.utils.import_guard import *  # noqa: F403

path = "data/processed/austrian_bundesliga_2025-26_features.csv"
df = pd.read_csv(path)

print("---- Columns ----")
print(df.columns.tolist())

print("\n---- First few rows ----")
print(df.head(3))
