# src/scouting_ml/core_features.py

from __future__ import annotations
import pandas as pd
import numpy as np


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add generic, analysis-friendly columns to a Transfermarkt player dataframe.
    Safe to call even if some columns are missing.
    """
    df = df.copy()

    # 1) log market value
    if "market_value_eur" in df.columns:
        df["market_value_log"] = np.log1p(df["market_value_eur"])

    # 2) age grouping + flags
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 18, 22, 26, 30, 35, 50],
            labels=["<18", "18-22", "22-26", "26-30", "30-35", "35+"],
        )
        df["is_young"] = (df["age"] <= 22).astype(int)
        df["is_veteran"] = (df["age"] >= 30).astype(int)

    # 3) value per age
    if {"market_value_eur", "age"}.issubset(df.columns):
        df["value_per_age"] = df["market_value_eur"] / df["age"].clip(lower=1)

    # 4) height z-score
    if "height_cm" in df.columns and df["height_cm"].notna().any():
        mean_h = df["height_cm"].mean()
        std_h = df["height_cm"].std(ddof=0) or 1
        df["height_zscore"] = (df["height_cm"] - mean_h) / std_h

    return df
