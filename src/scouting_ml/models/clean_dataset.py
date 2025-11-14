from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clean_dataset(input_path: str, output_path: str, min_minutes: float = 450.0) -> None:
    df = pd.read_parquet(input_path)
    if "minutes" not in df.columns and "sofa_minutesPlayed" in df.columns:
        df["minutes"] = pd.to_numeric(df["sofa_minutesPlayed"], errors="coerce")

    df["market_value_eur"] = pd.to_numeric(df.get("market_value_eur"), errors="coerce")
    df = df[df["market_value_eur"].notna()]

    if "minutes" in df.columns:
        df = df[df["minutes"].fillna(0) >= min_minutes]

    columns_to_drop = [c for c in ["link", "type", "player_name", "source_file"] if c in df.columns]
    df = df.drop(columns=columns_to_drop)

    for col in ("season", "league", "club"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[clean] wrote {len(df):,} rows → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean combined Big Five dataset for modeling.")
    parser.add_argument("--input", default="data/model/big5_players.parquet")
    parser.add_argument("--output", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--min-minutes", type=float, default=450.0)
    args = parser.parse_args()
    clean_dataset(args.input, args.output, args.min_minutes)
