from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor
import shap
import matplotlib.pyplot as plt


# ------------------------------
# Loading
# ------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


# ------------------------------
# Cleaning
# ------------------------------
LEAK_COLS = {
    "market_value",
    "market_value_eur",
    "player_id",
    "name",
    "dob",
    "dob_age",
}

ID_COLS = {
    "team_id",
    "club_id",
    "transfermarkt_id",
    "sofa_player_id",
    "sofa_team_id",
}

HIGH_CARD = {
    "nationality",
    "club",
    "sofa_team_name",
}


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in LEAK_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in HIGH_CARD if c in df.columns], errors="ignore")

    # Ensure correct types
    for col in ["season", "league", "position_group"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


# ------------------------------
# Column inference
# ------------------------------
def infer_numeric(df: pd.DataFrame):
    blocked = {"log_market_value", "expected_value_eur", "value_diff"}
    out = []
    for c in df.columns:
        if c in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 1:
            out.append(c)
    return out


def infer_categoricals(df: pd.DataFrame):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# ------------------------------
# Build model pipeline
# ------------------------------
def build_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    model = LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([("prep", pre), ("model", model)])


# ------------------------------
# SHAP
# ------------------------------
def save_shap(pipe, X, out_path):
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    sample = X.sample(min(500, len(X)), random_state=42)
    transformed = prep.transform(sample)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = prep.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(transformed)

    shap_expl = shap.Explanation(values=shap_vals, data=transformed, feature_names=feature_names)

    shap.plots.bar(shap_expl, max_display=25, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[shap] saved → {out_path}")


# ------------------------------
# Train single position model
# ------------------------------
def train_single_position(df_raw, pos, test_season, out_dir):
    dfpos = df_raw[df_raw["position_group"] == pos].copy()

    if len(dfpos) < 200:
        print(f"[warn] Skipping {pos}: too few samples ({len(dfpos)})")
        return None

    df = clean_df(dfpos)

    train = df[df["season"] != test_season].copy()
    test = df[df["season"] == test_season].copy()

    train_raw = dfpos[dfpos["season"] != test_season].copy()
    test_raw = dfpos[dfpos["season"] == test_season].copy()

    num_cols = infer_numeric(df)
    cat_cols = infer_categoricals(df)

    feat_cols = num_cols + cat_cols

    X_train = train[feat_cols]
    y_train = train_raw["log_market_value"]

    X_test = test[feat_cols]
    y_test = test_raw["log_market_value"]

    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)

    # SHAP
    shap_path = Path(out_dir) / f"shap_{pos}.png"
    save_shap(pipe, X_train, shap_path)

    # Predictions
    log_pred = pipe.predict(X_test)
    pred = np.expm1(log_pred)

    test_raw["expected_value_eur"] = pred
    test_raw["value_diff"] = test_raw["market_value_eur"] - pred

    mae = mean_absolute_error(np.expm1(y_test), pred)
    mape = mean_absolute_percentage_error(np.expm1(y_test), pred)
    r2 = r2_score(np.expm1(y_test), pred)

    print(f"[{pos}] R² {r2*100:,.2f}%  |  MAE €{mae:,.0f}  |  MAPE {mape*100:,.2f}%")

    return test_raw


# ------------------------------
# MAIN
# ------------------------------
def main(dataset, test_season, output, shap_out_dir="logs/shap"):
    df = load_dataset(dataset)
    df = df[df["log_market_value"].notna()].copy()

    Path(shap_out_dir).mkdir(parents=True, exist_ok=True)

    all_predictions = []

    for pos in ["GK", "DF", "MF", "FW"]:
        print(f"\n=========================\n TRAINING {pos}\n=========================")
        preds = train_single_position(df, pos, test_season, shap_out_dir)
        if preds is not None:
            all_predictions.append(preds)

    final = pd.concat(all_predictions, ignore_index=True)
    final.sort_values("value_diff").to_csv(output, index=False)
    print(f"\n[done] wrote merged predictions → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--test-season", default="2024/25")
    parser.add_argument("--output", default="data/model/big5_predictions_position_models.csv")
    args = parser.parse_args()

    main(args.dataset, args.test_season, args.output)
