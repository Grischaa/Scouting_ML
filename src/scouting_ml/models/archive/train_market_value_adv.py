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
    mean_absolute_percentage_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMRegressor
from scouting_ml.features.advanced_features import add_advanced_features


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------
def drop_leakage(df):
    cols = [
        "market_value",
        "market_value_eur",
        "player_id",
        "name",
        "dob",
        "dob_age",
        "transfermarkt_id",
    ]
    present = [c for c in cols if c in df.columns]
    df = df.drop(columns=present)
    print("[clean] dropped leakage:", present)
    return df


def drop_id_columns(df):
    to_drop = [c for c in df.columns if c.endswith("_id") or c in ["id", "club_id", "team_id"]]
    df = df.drop(columns=to_drop)
    if to_drop:
        print("[clean] dropped ID-like:", to_drop)
    return df


def drop_high_cardinality(df, max_unique=30):
    drop_cols = []
    for col in df.select_dtypes(include=["object", "category"]):
        if df[col].nunique() > max_unique:
            drop_cols.append(col)
    df = df.drop(columns=drop_cols)
    if drop_cols:
        print("[clean] dropped high-cardinality categoricals:", drop_cols)
    return df


def add_age_bucket(df):
    if "age" in df.columns:
        df["age_bucket"] = pd.cut(
            df["age"],
            bins=[0, 21, 24, 28, 33, 200],
            labels=["u21", "21–24", "25–28", "29–33", "34+"],
            right=False,
        )
    return df


# ---------------------------------------------------------
# COLUMN DETECTION
# ---------------------------------------------------------
def infer_numeric(df):
    blocked = {"log_market_value", "expected_value_eur", "value_diff"}
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in blocked
        and df[c].notna().sum() > 0
    ]


def infer_categorical(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


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
        n_estimators=2200,
        learning_rate=0.025,
        num_leaves=96,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        reg_alpha=1.0,
        reg_lambda=1.0,
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
    out = df.copy()
    out["expected_value_eur"] = np.expm1(preds_log)
    out["value_diff"] = df["market_value_eur"] - out["expected_value_eur"]
    return out


# ---------------------------------------------------------
# SHAP
# ---------------------------------------------------------
def generate_shap(pipe, X_sample):
    import shap
    import matplotlib.pyplot as plt

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["lgbm"]

    sample = X_sample.sample(min(500, len(X_sample)), random_state=42)
    transformed = prep.transform(sample)

    feature_names = prep.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(transformed)

    shap_obj = shap.Explanation(shap_vals, transformed, feature_names=feature_names)

    out_path = Path("logs/shap/shap_global_bar_adv.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shap.plots.bar(shap_obj, max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("[shap] saved advanced SHAP →", out_path)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main(dataset_path, test_season, output_path, save_shap):

    df_raw = load_dataset(dataset_path)
    df_raw = df_raw[df_raw["log_market_value"].notna()].copy()

    df = df_raw.copy()

    df = drop_leakage(df)
    df = drop_id_columns(df)
    df = drop_high_cardinality(df)
    df = add_age_bucket(df)

    # ⭐ Advanced features
    df = add_advanced_features(df)

    num_cols = infer_numeric(df)
    cat_cols = infer_categorical(df)

    train_df, test_df = split(df, test_season)
    train_raw, test_raw = split(df_raw, test_season)

    X_train = train_df[num_cols + cat_cols]
    y_train = train_raw["log_market_value"]

    X_test = test_df[num_cols + cat_cols]
    y_test = test_raw["log_market_value"]

    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    if save_shap:
        generate_shap(pipe, X_train)

    preds_log = pipe.predict(X_test)
    preds = add_predictions(test_raw, preds_log)

    mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds_log))
    mape = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(preds_log))
    r2 = r2_score(np.expm1(y_test), np.expm1(preds_log))

    print(f"[adv-lgbm] R²   {r2*100:.2f}%")
    print(f"[adv-lgbm] MAE  €{mae:,.0f}")
    print(f"[adv-lgbm] MAPE {mape*100:.2f}%")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False)
    print("[adv-lgbm] wrote predictions →", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Market Value Model with LightGBM.")
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--test-season", dest="test_season", default="2024/25")
    parser.add_argument("--output", default="data/model/big5_predictions_adv.csv")
    parser.add_argument("--shap", action="store_true")
    args = parser.parse_args()

    main(args.dataset, args.test_season, args.output, args.shap)
