from __future__ import annotations
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

# Very high-cardinality categoricals that explode one-hot and add little signal
HIGH_CARD_COLS = {
    "nationality",
    "club",
    "sofa_team_name",
}


def clean_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for modeling:
    - Drop leakage / ID-like / high-cardinality columns.
    - Ensure string dtypes for core categoricals.
    """
    df = df.copy()

    df = df.drop(columns=[c for c in LEAK_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in HIGH_CARD_COLS if c in df.columns], errors="ignore")

    for col in ["season", "league", "position_group"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df
