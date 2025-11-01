# src/scouting_ml/clean_tm.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from scouting_ml.analysis.core_features import ensure_position_group
from scouting_ml.league_registry import get_league
from scouting_ml.utils.import_guard import *  # noqa: F403


def clean_transfermarkt(
    df: pd.DataFrame,
    *,
    default_league: str = "Unknown League",
    default_season: str = "",
) -> pd.DataFrame:
    """Final consistency cleaning for Transfermarkt merged dataset."""
    df = df.copy()

    # --- 1️⃣ Column standardization ---
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # --- 2️⃣ Ensure essential fields exist ---
    required = ["player_id", "name", "club", "league", "season", "market_value_eur"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # --- 3️⃣ Ensure position_group ---
    df = ensure_position_group(df)
    df["position_group"] = df["position_group"].fillna("Unknown")
    if "position_alt" in df.columns:
        df["position_alt"] = df["position_alt"].astype("string").fillna("")


    # --- 4️⃣ Numeric sanity checks ---
    if "age" in df.columns:
        df.loc[(df["age"] < 14) | (df["age"] > 45), "age"] = np.nan

    if "height_cm" in df.columns:
        df.loc[(df["height_cm"] < 150) | (df["height_cm"] > 210), "height_cm"] = np.nan

    if "market_value_eur" in df.columns:
        # ✅ Do NOT reconvert — just verify range
        df.loc[(df["market_value_eur"] <= 0) | (df["market_value_eur"] > 1e8), "market_value_eur"] = np.nan

    # --- 5️⃣ Drop duplicates ---
    if "player_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["player_id"], keep="first")
        print(f"[clean] Removed {before - len(df)} duplicates")

    # --- 6️⃣ Fill missing metadata ---
    df["club"] = df["club"].fillna("Unknown Club")
    df["league"] = df["league"].fillna(default_league)
    df["season"] = df["season"].fillna(default_season)

    # --- 7️⃣ Sort for readability ---
    sort_cols = [c for c in ["club", "position_group", "age"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def main():
    import argparse
    p = argparse.ArgumentParser(description="Clean merged Transfermarkt dataset for final use.")
    p.add_argument("--infile", required=True, help="Path to the merged features file")
    p.add_argument("--outfile", required=True, help="Path to save cleaned CSV")
    p.add_argument(
        "--league",
        help="League slug from scouting_ml.league_registry to use for default labels.",
    )
    p.add_argument(
        "--default-league",
        help="Fallback league label when missing from the dataset.",
    )
    p.add_argument(
        "--default-season",
        help="Fallback season label when missing from the dataset.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.infile)
    print(f"[clean] Loaded {len(df)} rows from {args.infile}")

    default_league = args.default_league
    default_season = args.default_season

    if args.league:
        try:
            config = get_league(args.league)
        except KeyError as exc:
            raise SystemExit(str(exc))
        default_league = default_league or config.name
        default_season = default_season or (config.tm_season_label or config.sofa_season_label or "")

    df_clean = clean_transfermarkt(
        df,
        default_league=default_league or "Unknown League",
        default_season=default_season or "",
    )

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print(f"[clean] ✅ Saved cleaned data -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
