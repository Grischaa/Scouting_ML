# src/scouting_ml/core_features.py

from __future__ import annotations
import pandas as pd
import numpy as np
import re


# ---------- helpers -----------------------------------------------------------

def _map_position_group(pos: str | None) -> str | None:
    """Best-effort mapping from raw TM position text to GK/DF/MF/FW."""
    if not isinstance(pos, str):
        return None
    p = pos.lower()

    # goalkeepers
    if "goalkeeper" in p or "torwart" in p or "keeper" in p:
        return "GK"

    # defenders
    if any(k in p for k in [
        "verteidiger", "defender", "centre-back", "center-back",
        "innenverteidiger", "left-back", "right-back",
        "full-back", "wing-back", "linker verteidiger", "rechter verteidiger"
    ]):
        return "DF"

    # midfielders
    if any(k in p for k in [
        "mittelfeld", "midfield", "defensive midfield", "attacking midfield",
        "winger", "flügel", "wide midfielder", "central midfield"
    ]):
        return "MF"

    # forwards
    if any(k in p for k in [
        "stürmer", "sturm", "forward", "striker", "centre-forward", "center-forward",
        "second striker"
    ]):
        return "FW"

    return None


def ensure_position_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure df has a 'position_group' column.
    If it's missing or has NaNs, try to derive from 'position'.
    """
    df = df.copy()
    if "position_group" not in df.columns:
        df["position_group"] = pd.NA

    # fill missing from 'position'
    if "position" in df.columns:
        mask = df["position_group"].isna() | (df["position_group"].astype(str).str.strip() == "")
        df.loc[mask, "position_group"] = df.loc[mask, "position"].map(_map_position_group)

    return df


# ---------- main feature function --------------------------------------------

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add generic, analysis-friendly columns to a Transfermarkt/Sofa-like player dataframe.
    This function is safe to call multiple times.
    """
    df = df.copy()

    # 0) make sure we have position_group
    df = ensure_position_group(df)

    # 1) log market value
    if "market_value_eur" in df.columns:
        df["market_value_log"] = np.log1p(df["market_value_eur"])

    # 2) age stuff
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

    # 5) position one-hots
    # normalize position_group once
    pg = df["position_group"].astype(str).str.upper()
    df["is_gk"] = (pg == "GK").astype(int)
    df["is_df"] = (pg == "DF").astype(int)
    df["is_mf"] = (pg == "MF").astype(int)
    df["is_fw"] = (pg == "FW").astype(int)

    # 6) minute-based features (will kick in once we add Sofascore)
    # we assume columns like 'minutes', 'goals', 'assists' if present
    if "minutes" in df.columns:
        mins = df["minutes"].replace(0, np.nan)

        if "goals" in df.columns:
            df["goals_per_90"] = (df["goals"] / mins) * 90

        if "assists" in df.columns:
            df["assists_per_90"] = (df["assists"] / mins) * 90

        if "market_value_eur" in df.columns:
            df["value_per_90"] = df["market_value_eur"] / (mins / 90)

    return df
