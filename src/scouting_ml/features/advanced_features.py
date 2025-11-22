from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. Utility: safe numeric getter
# ---------------------------------------------------------
def get(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else 0


# ---------------------------------------------------------
# 2. Progressive Passing Metrics & xThreat-lite
# ---------------------------------------------------------
def add_progression_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # progressive passes estimate (Sofa doesn't give it, we approximate)
    df["progressive_passing"] = (
        0.7 * get(df, "sofa_accurateFinalThirdPasses_per90") +
        0.3 * get(df, "sofa_keyPasses_per90")
    )

    # progressive carries proxy
    if "sofa_successfulDribbles_per90" in df.columns:
        df["progressive_carries"] = df["sofa_successfulDribbles_per90"]
    else:
        df["progressive_carries"] = 0

    # xThreat light approximation
    df["xThreat_proxy"] = (
        0.5 * df["progressive_passing"]
        + 0.5 * df["progressive_carries"]
    )

    return df


# ---------------------------------------------------------
# 3. Shot Quality, Goal Threat & Creativity
# ---------------------------------------------------------
def add_attacking_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["shot_quality"] = get(df, "sofa_expectedGoals_per90") / (
        get(df, "sofa_totalShots_per90") + 1e-6
    )

    df["chance_creation"] = (
        0.5 * get(df, "sofa_keyPasses_per90") +
        0.5 * get(df, "sofa_bigChancesCreated_per90")
    )

    df["attacking_index_adv"] = (
        get(df, "sofa_goals_per90")
        + get(df, "sofa_assists_per90")
        + get(df, "sofa_expectedGoals_per90")
        + 0.5 * df["chance_creation"]
        + 0.5 * df["progressive_passing"]
    )

    return df


# ---------------------------------------------------------
# 4. Defensive Metrics (possession-adjusted)
# ---------------------------------------------------------
def add_defensive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["defensive_actions"] = (
        get(df, "sofa_tackles_per90")
        + get(df, "sofa_interceptions_per90")
        + get(df, "sofa_clearances_per90")
    )

    # safe team possession
    team_poss = df["team_possession"] if "team_possession" in df.columns else 50.0

    df["padj_defensive_actions"] = df["defensive_actions"] / (1 + 0.01 * team_poss)

    df["duel_intensity"] = (
        get(df, "sofa_totalDuelsWon_per90")
        + get(df, "sofa_aerialDuelsWon_per90")
    )

    df["defensive_index_adv"] = (
        df["padj_defensive_actions"]
        + 0.5 * df["duel_intensity"]
    )

    return df



# ---------------------------------------------------------
# 5. Passing Metrics
# ---------------------------------------------------------
def add_passing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["passing_efficiency_adv"] = (
        get(df, "sofa_accuratePasses_per90") *
        (get(df, "sofa_accuratePassesPercentage") / 100)
    )

    df["passing_style_index"] = (
        0.6 * df["passing_efficiency_adv"]
        + 0.4 * df["progressive_passing"]
    )

    return df


# ---------------------------------------------------------
# 6. Age Curve Feature
# ---------------------------------------------------------
def add_age_curve(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" in df.columns:
        df["age_curve"] = -(df["age"] - 27).abs()  # peak age ~27
    else:
        df["age_curve"] = 0
    return df


# ---------------------------------------------------------
# 7. League Strength (UEFA coefficient)
# ---------------------------------------------------------
def add_league_strength(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    league_strength = {
        "English Premier League": 1.00,
        "LaLiga": 0.92,
        "Bundesliga": 0.90,
        "Serie A": 0.88,
        "Ligue 1": 0.86,
    }

    if "league" in df.columns:
        df["league_strength"] = df["league"].map(league_strength).fillna(0.80)
    else:
        df["league_strength"] = 0.80

    return df


# ---------------------------------------------------------
# MASTER: Apply all advanced features
# ---------------------------------------------------------
def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_progression_features(df)
    df = add_attacking_features(df)
    df = add_defensive_features(df)
    df = add_passing_features(df)
    df = add_age_curve(df)
    df = add_league_strength(df)
    return df
