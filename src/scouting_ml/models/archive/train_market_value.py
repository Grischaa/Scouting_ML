from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMRegressor


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------
def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "market_value",
        "market_value_eur",
        "player_id",
        "name",
        "dob",
        "dob_age",
        "transfermarkt_id",
    ]
    present = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=present)
    print("[clean] dropped leakage columns:", present)
    return df


def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in df.columns if c.endswith("_id") or c.lower() in {"id", "club_id", "team_id"}]
    if id_cols:
        df = df.drop(columns=id_cols)
        print("[clean] dropped id-like columns:", id_cols)
    return df


def drop_high_cardinality(df: pd.DataFrame, max_unique=30) -> pd.DataFrame:
    drop_cols = []
    for col in df.select_dtypes(include=["object", "category"]):
        if df[col].nunique() > max_unique:
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print("[clean] dropped high-cardinality categoricals:", drop_cols)
    return df


def add_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "age" in df.columns:
        df["age_bucket"] = pd.cut(
            df["age"],
            bins=[0, 21, 24, 28, 33, 100],
            labels=["u21", "21–24", "25–28", "29–33", "34+"],
            right=False,
        )
    return df


# ---------------------------------------------------------
# CUSTOM SCOUTING FEATURES
# ---------------------------------------------------------
def add_football_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Attacking index
    df["attacking_index"] = (
        df.get("sofa_goals_per90", 0)
        + df.get("sofa_expectedGoals_per90", 0)
        + df.get("sofa_totalShots_per90", 0)
        + df.get("sofa_keyPasses_per90", 0)
        + df.get("sofa_successfulDribbles_per90", 0)
    )

    # Defensive index
    df["defensive_index"] = (
        df.get("sofa_tackles_per90", 0)
        + df.get("sofa_interceptions_per90", 0)
        + df.get("sofa_clearances_per90", 0)
        + df.get("sofa_aerialDuelsWon_per90", 0)
    )

    # Passing efficiency index
    df["passing_efficiency"] = (
        df.get("sofa_accuratePasses_per90", 0)
        * df.get("sofa_accuratePassesPercentage", 0)
        / 100.0
    )

    # Goal involvement
    df["goal_involvement_per90"] = (
        df.get("sofa_goals_per90", 0)
        + df.get("sofa_assists_per90", 0)
    )

    return df


# ---------------------------------------------------------
# COLUMN DETECTION
# ---------------------------------------------------------
def infer_numeric(df: pd.DataFrame):
    blocked = {"log_market_value", "expected_value_eur", "value_diff"}
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() > 0
        and c not in blocked
    ]


def infer_categorical(df: pd.DataFrame):
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return cats


# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------
def build_pipeline(num_cols, cat_cols):
    numeric_proc = SimpleImputer(strategy="median")

    cat_proc = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_proc, num_cols),
        ("cat", cat_proc, cat_cols),
    ])

    model = LGBMRegressor(
        n_estimators=1800,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=64,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("prep", preprocessor),
        ("lgbm", model),
    ])


# ---------------------------------------------------------
# SPLIT + PREDICT
# ---------------------------------------------------------
def split(df, test_season):
    return df[df["season"] != test_season], df[df["season"] == test_season]


def add_predictions(df, preds_log):
    df = df.copy()
    df["expected_value_eur"] = np.expm1(preds_log)
    df["value_diff"] = df["market_value_eur"] - df["expected_value_eur"]
    return df


# ---------------------------------------------------------
# SHAP
# ---------------------------------------------------------
def generate_shap(pipe, X_sample):
    import shap
    import matplotlib.pyplot as plt

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["lgbm"]

    # Transform sample
    sample = X_sample.sample(min(500, len(X_sample)), random_state=42)
    transformed = prep.transform(sample)

    feature_names = prep.get_feature_names_out()

    expl = shap.TreeExplainer(model)
    shap_vals = expl.shap_values(transformed)

    shap_obj = shap.Explanation(shap_vals, transformed, feature_names=feature_names)

    Path("logs/shap").mkdir(parents=True, exist_ok=True)
    out = Path("logs/shap/shap_global_bar.png")

    shap.plots.bar(shap_obj, max_display=25, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    print("[shap] saved global plot →", out)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main(dataset_path, test_season, output_path, run_shap=False):

    df_raw = load_dataset(dataset_path)
    df_raw = df_raw[df_raw["log_market_value"].notna()].copy()

    df = df_raw.copy()

    df = drop_leakage(df)
    df = drop_id(df)
    df = drop_high_cardinality(df, max_unique=30)
    df = add_age_bucket(df)
    df = add_football_features(df)

    num_cols = infer_numeric(df)
    cat_cols = infer_categorical(df)

    train, test = split(df, test_season)
    train_raw, test_raw = split(df_raw, test_season)

    X_train = train[num_cols + cat_cols]
    y_train = train_raw["log_market_value"]

    X_test = test[num_cols + cat_cols]
    y_test = test_raw["log_market_value"]

    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    if run_shap:
        generate_shap(pipe, X_train)

    preds_log = pipe.predict(X_test)
    preds = add_predictions(test_raw, preds_log)

    mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds_log))
    mape = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(preds_log))
    r2 = r2_score(np.expm1(y_test), np.expm1(preds_log))

    print(f"[lgbm] R²  {r2*100:.2f}%")
    print(f"[lgbm] MAE €{mae:,.0f} | MAPE {mape*100:.2f}%")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False)
    print("[lgbm] wrote predictions →", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--test-season", dest="test_season", default="2024/25")
    parser.add_argument("--output", default="data/model/big5_predictions_2024-25.csv")
    parser.add_argument("--shap", action="store_true")
    args = parser.parse_args()

    main(
        args.dataset,
        args.test_season,
        args.output,
        run_shap=args.shap
    )
