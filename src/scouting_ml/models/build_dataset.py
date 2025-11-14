from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from scouting_ml.paths import PROCESSED_DIR


def load_merged_files(files: List[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("No merged files found; run the league refresh first.")
    return pd.concat(frames, ignore_index=True, sort=False)


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    def to_numeric(col: str) -> pd.Series:
        return pd.to_numeric(result[col], errors="coerce") if col in result.columns else pd.Series(np.nan, index=result.index)

    if "market_value_eur" in result.columns:
        mv = to_numeric("market_value_eur")
        result["market_value_eur"] = mv
        result["log_market_value"] = np.log1p(mv).where(mv > 0)
    else:
        result["log_market_value"] = np.nan

    minutes = to_numeric("sofa_minutesPlayed")
    result["minutes"] = minutes
    per90 = (90.0 / minutes.replace({0: np.nan})).replace([np.inf, -np.inf], np.nan)

    volume_cols = [
        "sofa_goals",
        "sofa_assists",
        "sofa_expectedGoals",
        "sofa_totalShots",
        "sofa_shotsOnTarget",
        "sofa_keyPasses",
        "sofa_accurateFinalThirdPasses",
        "sofa_accuratePasses",
        "sofa_totalDuelsWon",
        "sofa_groundDuelsWon",
        "sofa_aerialDuelsWon",
        "sofa_successfulDribbles",
        "sofa_tackles",
        "sofa_interceptions",
        "sofa_clearances",
    ]

    for col in volume_cols:
        if col in result.columns:
            result[f"{col}_per90"] = to_numeric(col) * per90

    ratio_pairs = [
        ("sofa_goals", "sofa_totalShots", "shot_conversion"),
        ("sofa_assists", "sofa_keyPasses", "assist_to_keypass"),
        ("sofa_successfulDribbles", "sofa_totalDuelsWon", "dribble_success"),
        ("sofa_totalDuelsWon", "sofa_totalDuelsWon_per90", "duel_consistency"),
    ]

    for num_col, den_col, name in ratio_pairs:
        if num_col in result.columns and den_col in result.columns:
            num = to_numeric(num_col)
            den = to_numeric(den_col).replace({0: np.nan})
            result[name] = (num / den).replace([np.inf, -np.inf], np.nan)

    for pct_col in [
        "sofa_totalDuelsWonPercentage",
        "sofa_groundDuelsWonPercentage",
        "sofa_aerialDuelsWonPercentage",
        "sofa_successfulDribblesPercentage",
        "sofa_accuratePassesPercentage",
    ]:
        if pct_col in result.columns:
            pct = to_numeric(pct_col)
            result[pct_col] = pct

    if "age" in result.columns:
        age = to_numeric("age")
        result["age"] = age
        result["age_sq"] = age ** 2

    if "season" in result.columns:
        result["season_end_year"] = (
            result["season"]
            .astype(str)
            .str.extract(r"(\d{4})")
            .astype(float)
        )

    if "position_group" in result.columns:
        result["is_forward"] = (result["position_group"].str.upper() == "FW").astype(int)
        result["is_midfielder"] = (result["position_group"].str.upper() == "MF").astype(int)
        result["is_defender"] = (result["position_group"].str.upper() == "DF").astype(int)
        result["is_goalkeeper"] = (result["position_group"].str.upper() == "GK").astype(int)

    return result


def main(data_dir: str = "data/processed/Clubs combined", output: str = "data/model/big5_players.parquet") -> None:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*_with_sofa.csv"))
    df = load_merged_files(files)
    df = add_model_features(df)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[dataset] wrote {len(df):,} rows → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine *_with_sofa.csv files into a modeling dataset.")
    parser.add_argument("--data-dir", default="data/processed/Clubs combined", help="Directory containing *_with_sofa.csv files.")
    parser.add_argument("--output", default="data/model/big5_players.parquet", help="Output Parquet path.")
    args = parser.parse_args()
    main(args.data_dir, args.output)
