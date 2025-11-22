from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

from scouting_ml.utils.data_utils import (
    load_dataset,
    split_by_season,
    infer_numeric_columns,
    infer_categorical_columns,
)
from scouting_ml.features.feature_cleaning import clean_for_model
from scouting_ml.features.optuna_tuner import tune_lgbm
from scouting_ml.features.shap_selector import select_top_features
from scouting_ml.utils.shap_utils import save_tree_shap_bar
from scouting_ml.models.base_models import make_lgbm, make_xgb, make_cat

@dataclass
class PositionMetrics:
    position: str
    r2: float
    mae: float
    mape: float


# -------------------------
# Preprocessor builder
# -------------------------
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_proc = SimpleImputer(strategy="median")
    cat_proc = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_proc, num_cols),
            ("cat", cat_proc, cat_cols),
        ]
    )


# -------------------------
# Metrics helper
# -------------------------
def compute_position_metrics(pos: str, y_true_lin, y_pred_lin) -> PositionMetrics:
    mae = mean_absolute_error(y_true_lin, y_pred_lin)
    mape = mean_absolute_percentage_error(y_true_lin, y_pred_lin)
    r2 = r2_score(y_true_lin, y_pred_lin)
    print(f"[{pos}] R² {r2*100:,.2f}% | MAE €{mae:,.0f} | MAPE {mape*100:,.2f}%")
    return PositionMetrics(position=pos, r2=r2, mae=mae, mape=mape)



# -------------------------
# Per-position training
# -------------------------
def train_for_position(
    df_raw: pd.DataFrame,
    pos: str,
    test_season: str,
    shap_dir: Path,
    n_optuna_trials: int = 60,
) -> tuple[pd.DataFrame, PositionMetrics] | None:

    df_pos_raw = df_raw[df_raw["position_group"] == pos].copy()

    if len(df_pos_raw) < 250:
        print(f"[warn] skipping {pos}: too few samples ({len(df_pos_raw)})")
        return None

    df_clean = clean_for_model(df_pos_raw)

    train_df, test_df = split_by_season(df_clean, test_season)
    train_raw, test_raw = split_by_season(df_pos_raw, test_season)

    blocked = {"log_market_value", "expected_value_eur", "value_diff"}
    num_cols = infer_numeric_columns(train_df, blocked=blocked)
    cat_cols = infer_categorical_columns(train_df)
    feat_cols = num_cols + cat_cols

    if not feat_cols:
        print(f"[warn] no features for {pos}, skipping.")
        return None

    # ---------------- Stage 1: Preprocess + Optuna (LGBM) on full feature set
    pre_full = build_preprocessor(num_cols, cat_cols)
    X_train_full = pre_full.fit_transform(train_df[feat_cols])
    y_train = train_raw["log_market_value"]

    # Tune LGBM in transformed space
    best_lgbm_params = tune_lgbm(X_train_full, y_train, n_trials=n_optuna_trials)

    # Fit tuned LGBM on full features for SHAP-based feature selection
    lgbm_full = make_lgbm(best_lgbm_params)
    lgbm_full.fit(X_train_full, y_train)

    # SHAP-based feature selection → top 25 base features
    num_sel, cat_sel = select_top_features(
        lgbm_full,
        pre_full,
        train_df[feat_cols],
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        top_n=25,
    )
    sel_cols: List[str] = num_sel + cat_sel
    if not sel_cols:
        # fallback: use original features if selection fails
        sel_cols = feat_cols
        num_sel = num_cols
        cat_sel = cat_cols

    # ---------------- Stage 2: Final preprocessing on selected features
    pre_sel = build_preprocessor(num_sel, cat_sel)
    X_train = pre_sel.fit_transform(train_df[sel_cols])
    X_test = pre_sel.transform(test_df[sel_cols])
    y_test = test_raw["log_market_value"]

    # ---------------- Stage 3: Models
    if pos == "GK":
        # GK → LGBM only, no ensemble
        lgbm = make_lgbm(best_lgbm_params)
        lgbm.fit(X_train, y_train)

        # SHAP plot for GK using LGBM
        save_tree_shap_bar(lgbm, pre_sel, train_df[sel_cols], shap_dir / f"shap_{pos}.png")

        log_pred = lgbm.predict(X_test)
        pred = np.expm1(log_pred)

    else:
        # Outfield positions → ensemble
        lgbm = make_lgbm(best_lgbm_params)
        xgb = make_xgb()
        cat = make_cat()

        lgbm.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        cat.fit(X_train, y_train)

        # SHAP from LGBM (stable, tree-based)
        save_tree_shap_bar(lgbm, pre_sel, train_df[sel_cols], shap_dir / f"shap_{pos}.png")

        P_train = np.vstack([
            lgbm.predict(X_train),
            xgb.predict(X_train),
            cat.predict(X_train),
        ]).T

        P_test = np.vstack([
            lgbm.predict(X_test),
            xgb.predict(X_test),
            cat.predict(X_test),
        ]).T

        meta = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=3,
            random_state=42,
        )
        meta.fit(P_train, y_train)

        log_pred = meta.predict(P_test)
        pred = np.expm1(log_pred)

    # ---------------- Metrics & output
    y_true_lin = np.expm1(y_test)
    metrics = compute_position_metrics(pos, y_true_lin, pred)

    test_out = test_raw.copy()
    test_out["expected_value_eur"] = pred
    test_out["value_diff"] = test_out["market_value_eur"] - test_out["expected_value_eur"]

    return test_out, metrics



# -------------------------
# MAIN
# -------------------------
def main(
    dataset_path: str,
    test_season: str,
    output_path: str,
    n_optuna_trials: int = 60,
) -> None:
    df_raw = load_dataset(dataset_path)
    df_raw = df_raw[df_raw["log_market_value"].notna()].copy()

    shap_dir = Path("logs/shap")
    shap_dir.mkdir(parents=True, exist_ok=True)

    all_preds: List[pd.DataFrame] = []
    all_metrics: List[PositionMetrics] = []

    for pos in ["GK", "DF", "MF", "FW"]:
        print("\n======================")
        print(f"  TRAINING {pos}")
        print("======================")
        res = train_for_position(df_raw, pos, test_season, shap_dir, n_optuna_trials)
        if res is not None:
            preds, metrics = res
            all_preds.append(preds)
            all_metrics.append(metrics)


    if not all_preds:
        print("[error] no position models produced predictions.")
        return

    final = pd.concat(all_preds, ignore_index=True)
    final.sort_values("value_diff").to_csv(output_path, index=False)

    # ---- metrics summary ----
    if all_metrics:
        print("\n========== POSITION SUMMARY ==========")
        # keep a stable order GK, DF, MF, FW if available
        order = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}
        all_metrics_sorted = sorted(all_metrics, key=lambda m: order.get(m.position, 99))
        for m in all_metrics_sorted:
            print(
                f"{m.position:>2} | "
                f"R² {m.r2*100:6.2f}% | "
                f"MAE €{m.mae:,.0f} | "
                f"MAPE {m.mape*100:6.2f}%"
            )

        # optional: overall metrics across all positions
        merged_true = np.concatenate(
            [np.expm1(load["log_market_value"].values) for load in all_preds]
        )
        merged_pred = np.concatenate([p["expected_value_eur"].values for p in all_preds])
        mae_all = mean_absolute_error(merged_true, merged_pred)
        mape_all = mean_absolute_percentage_error(merged_true, merged_pred)
        r2_all = r2_score(merged_true, merged_pred)
        print("--------------------------------------")
        print(
            f"ALL | R² {r2_all*100:6.2f}% | "
            f"MAE €{mae_all:,.0f} | "
            f"MAPE {mape_all*100:6.2f}%"
        )


    print(f"\n[done] wrote merged predictions → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full position-aware market value pipeline with Optuna + SHAP + ensemble.")
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--test-season", dest="test_season", default="2024/25")
    parser.add_argument("--output", default="data/model/big5_predictions_full_v2.csv")
    parser.add_argument("--trials", type=int, default=60, help="Optuna trials per position (full mode ~60).")
    args = parser.parse_args()

    main(
        dataset_path=args.dataset,
        test_season=args.test_season,
        output_path=args.output,
        n_optuna_trials=args.trials,
    )
