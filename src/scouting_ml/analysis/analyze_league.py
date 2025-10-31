# src/scouting_ml/analyse_league.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from scouting_ml.utils.import_guard import *  # noqa: F403



def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read CSV safely with friendly error if not found."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e


def show_summary(df: pd.DataFrame) -> None:
    print(f"\n[summary] {len(df):,} players | {df['club'].nunique()} clubs | {df['position_group'].nunique()} position groups")
    print(f"Columns: {', '.join(df.columns[:10])} ...")
    print("\nPlayers per position group:")
    print(df["position_group"].value_counts(dropna=False).to_string())


def top_young_undervalued(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the youngest players with low market value per age ratio."""
    cols = ["name", "club", "age", "position_group", "market_value_eur", "value_per_age"]
    mask = (df["is_young"] == 1) & df["market_value_eur"].notna()
    tmp = df.loc[mask, cols].copy()
    tmp = tmp.sort_values("value_per_age", ascending=True)
    return tmp.head(top_n)


def median_value_by_club(df: pd.DataFrame) -> pd.DataFrame:
    """Return median player market value per club."""
    tmp = df.groupby("club")["market_value_eur"].median().sort_values(ascending=False)
    return tmp.reset_index().rename(columns={"market_value_eur": "median_value_eur"})


def top_value_per_age(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top players with highest value_per_age."""
    cols = ["name", "club", "age", "position_group", "market_value_eur", "value_per_age"]
    tmp = df.dropna(subset=["value_per_age"]).sort_values("value_per_age", ascending=False)
    return tmp.head(top_n)[cols]


def export_top_lists(df: pd.DataFrame, out_dir: Path, top_n: int = 10) -> None:
    """Write top players lists to CSVs under data/reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    top_young = top_young_undervalued(df, top_n)
    top_young.to_csv(out_dir / "top_young_undervalued.csv", index=False)

    top_vpa = top_value_per_age(df, top_n)
    top_vpa.to_csv(out_dir / "top_value_per_age.csv", index=False)

    med = median_value_by_club(df)
    med.to_csv(out_dir / "median_value_by_club.csv", index=False)

    print(f"[export] wrote reports to {out_dir.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick scouting analysis for merged league dataset.")
    ap.add_argument(
        "--infile",
        default="data/processed/austrian_bundesliga_2025-26_features.csv",
        help="Path to enriched league CSV (from inspect_merged).",
    )
    ap.add_argument("--top-n", type=int, default=10, help="Number of top players to display/export.")
    ap.add_argument("--export", action="store_true", help="Write CSV reports to data/reports.")
    args = ap.parse_args()

    df = safe_read_csv(Path(args.infile))
    show_summary(df)

    print("\nTop young undervalued players (by value/age):")
    print(top_young_undervalued(df, args.top_n).to_string(index=False))

    print("\nTop value-per-age players overall:")
    print(top_value_per_age(df, args.top_n).to_string(index=False))

    print("\nMedian market value by club:")
    print(median_value_by_club(df).head(10).to_string(index=False))

    if args.export:
        export_top_lists(df, Path("data/reports"), args.top_n)


if __name__ == "__main__":
    main()
