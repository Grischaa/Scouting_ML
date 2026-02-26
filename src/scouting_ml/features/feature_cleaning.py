from __future__ import annotations
from typing import Dict

import pandas as pd

# Explicit leakage columns (target or near-target, identity)
LEAK_COLS = {
    "market_value",
    "market_value_eur",
    "player_id",
    "name",
    "dob",
    "dob_age",
}

# Pure ID-like columns
ID_COLS = {
    "team_id",
    "club_id",
    "transfermarkt_id",
    "sofa_player_id",
    "sofa_team_id",
}

# Very high-cardinality categoricals that explode one-hot
HIGH_CARD_COLS = {
    "nationality",
    "club",
    "sofa_team_name",
}


def _drop_leakage_and_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=[c for c in LEAK_COLS if c in out.columns], errors="ignore")
    out = out.drop(columns=[c for c in ID_COLS if c in out.columns], errors="ignore")
    for col in ["season", "league", "position_group"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def fit_high_cardinality_maps(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Fit frequency maps for high-cardinality categoricals.
    IMPORTANT: call this on training data only.
    """
    maps: Dict[str, pd.Series] = {}
    for col in list(HIGH_CARD_COLS.intersection(df.columns)):
        freqs = df[col].value_counts(normalize=True)
        maps[col] = freqs
    return maps


def apply_high_cardinality_maps(df: pd.DataFrame, maps: Dict[str, pd.Series]) -> pd.DataFrame:
    out = df.copy()
    for col in list(HIGH_CARD_COLS.intersection(out.columns)):
        freq_map = maps.get(col, pd.Series(dtype=float))
        default = float(freq_map.mean()) if len(freq_map) else 0.0
        out[f"{col}_freq"] = out[col].map(freq_map).fillna(default)
        out = out.drop(columns=[col], errors="ignore")
    return out


def clean_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for modeling:
    - Drop leakage / ID-like columns.
    - Replace very high-cardinality categoricals with frequency encodings
      to preserve signal without exploding one-hot dimensions.
    - Ensure string dtypes for core categoricals.
    """
    base = _drop_leakage_and_ids(df)
    freq_maps = fit_high_cardinality_maps(base)
    cleaned = apply_high_cardinality_maps(base, freq_maps)
    return cleaned


def clean_train_val_test_for_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leakage-safe cleaning:
    - drop leakage/id columns in each split
    - fit high-card frequency maps on train only
    - apply the same maps to val/test
    """
    train_base = _drop_leakage_and_ids(train_df)
    val_base = _drop_leakage_and_ids(val_df)
    test_base = _drop_leakage_and_ids(test_df)

    maps = fit_high_cardinality_maps(train_base)
    train_clean = apply_high_cardinality_maps(train_base, maps)
    val_clean = apply_high_cardinality_maps(val_base, maps)
    test_clean = apply_high_cardinality_maps(test_base, maps)
    return train_clean, val_clean, test_clean
