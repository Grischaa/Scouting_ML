# src/scouting_ml/inspect_merged.py

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
from scouting_ml.utils.import_guard import *  # noqa: F403
from scouting_ml.utils.import_guard import *  # noqa: F403
from scouting_ml.core_features import add_basic_features


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect and enrich merged league CSV.")
    ap.add_argument(
        "--infile",
        default="data/processed/austrian_bundesliga_2025-26_full_stats.csv",
        help="Path to merged league CSV.",
    )
    ap.add_argument(
        "--outfile",
        default="data/processed/austrian_bundesliga_2025-26_features.csv",
        help="Where to write enriched CSV.",
    )
    args = ap.parse_args()

    in_path = Path(args.infile)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[inspect] Loading {in_path} ...")
    df = pd.read_csv(in_path)

    print(f"[inspect] Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print("[inspect] First 5 columns:", df.columns[:5].tolist())

    # quick missing overview
    na_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    print("[inspect] Missing values (top 15):")
    print(na_pct.head(15))

    # add features
    df = add_basic_features(df)
    print(f"[inspect] After feature engineering: {df.shape[1]} columns")

    # write back
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[inspect] ✅ wrote enriched file -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
