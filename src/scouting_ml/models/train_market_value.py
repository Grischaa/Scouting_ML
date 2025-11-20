from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.copy()


def remove_id_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns to avoid leakage (e.g., player_id, team_id)."""
    id_like = {
        "id",
        "player_id",
        "team_id",
        "club_id",
        "sofa_player_id",
        "sofa_team_id",
        "transfermarkt_id",
    }
    to_drop: list[str] = []
    for col in df.columns:
        low = col.lower()
        if low in id_like or low.endswith("_id") or low.startswith("id_"):
            to_drop.append(col)
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"[clean] dropped ID-like columns: {to_drop}")
    return df


def drop_per90_base_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer per90 stats over raw totals when both exist to reduce redundancy."""
    to_drop: list[str] = []
    for col in df.columns:
        if col.endswith("_per90"):
            base = col[: -len("_per90")]
            if base in df.columns:
                to_drop.append(base)
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"[clean] dropped raw totals in favor of per90 versions: {to_drop}")
    return df


def add_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if "age" not in df.columns:
        return df
    bins = [0, 21, 24, 28, 33, 100]
    labels = ["u21", "21-24", "25-28", "29-33", "34+"]
    df = df.copy()
    df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    return df


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    model = HistGradientBoostingRegressor(learning_rate=0.1, max_iter=1000, random_state=42)
    return Pipeline(steps=[("prep", preprocessor), ("hist", model)])


def run_grid_search(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    param_grid = {
        "hist__learning_rate": [0.05, 0.1, 0.2],
        "hist__max_iter": [400, 800, 1200],
        "hist__max_leaf_nodes": [15, 31, 63],
        "hist__min_samples_leaf": [10, 30, 60],
        "hist__l2_regularization": [0.0, 0.5, 1.0],
    }
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_mae = -search.best_score_
    print(f"[grid] best CV MAE €{best_mae:,.0f} with params {search.best_params_}")
    return search.best_estimator_


def filter_train_test(df: pd.DataFrame, test_season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"] != test_season].copy()
    test = df[df["season"] == test_season].copy()
    return train, test


def add_predictions(df: pd.DataFrame, log_preds: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["expected_value_eur"] = np.expm1(log_preds)
    out["value_diff"] = out["market_value_eur"] - out["expected_value_eur"]
    return out


def infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    blocked = {
        "market_value_eur",
        "log_market_value",
        "expected_value_eur",
        "value_diff",
    }
    numeric_cols: List[str] = []
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col]
            if series.notna().sum() == 0:
                continue
            if series.nunique(dropna=True) <= 1:
                continue
            numeric_cols.append(col)
    return numeric_cols


def infer_categorical_columns(df: pd.DataFrame) -> List[str]:
    preferred = ["league", "club", "position_group"]
    fallback = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cats = [col for col in preferred if col in df.columns]
    if not cats:
        cats = fallback
    return cats


def generate_shap_plot(
    pipe: Pipeline, X_sample: pd.DataFrame, output_dir: str = "logs/shap", max_display: int = 20
) -> Path:
    try:
        import shap  # type: ignore
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("SHAP plotting requires `pip install shap matplotlib`.") from exc

    if len(X_sample) == 0:
        raise ValueError("Cannot compute SHAP values with an empty sample.")

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["hist"]

    sample = X_sample.sample(min(500, len(X_sample)), random_state=42)
    transformed = prep.transform(sample)
    if hasattr(transformed, "todense"):
        transformed = np.asarray(transformed.todense())
    else:
        transformed = np.asarray(transformed)
    feature_names = prep.get_feature_names_out()

    explainer = shap.Explainer(model, transformed, feature_names=feature_names)
    shap_values = explainer(transformed, check_additivity=False)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_global_bar.png"

    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[shap] saved global bar plot → {out_path}")
    return out_path


def main(
    dataset_path: str = "data/model/big5_players_clean.parquet",
    test_season: str = "2024/25",
    output_path: str = "data/model/big5_predictions_2024-25.csv",
    run_grid: bool = False,
    save_shap: bool = False,
) -> None:
    df = load_dataset(dataset_path)

    df = df[df["log_market_value"].notna()].copy()
    df = remove_id_like_columns(df)
    df = drop_per90_base_duplicates(df)
    df = add_age_bucket(df)

    numeric_cols = infer_numeric_columns(df)
    categorical_cols = infer_categorical_columns(df)

    if not numeric_cols:
        raise ValueError("No numeric columns available for modeling.")

    feature_cols = list(dict.fromkeys(numeric_cols + categorical_cols))
    keep_cols = list(
        dict.fromkeys(feature_cols + ["season", "market_value_eur", "name", "log_market_value"])
    )
    available_keep = [col for col in keep_cols if col in df.columns]
    df = df[available_keep]

    train, test = filter_train_test(df, test_season=test_season)
    X_train = train[feature_cols]
    y_train = train["log_market_value"]
    X_test = test[feature_cols]
    y_test = test["log_market_value"]

    pipe = build_pipeline(numeric_cols, categorical_cols)
    if run_grid:
        pipe = run_grid_search(pipe, X_train, y_train)
    else:
        pipe.fit(X_train, y_train)

    if save_shap:
        generate_shap_plot(pipe, X_train, output_dir="logs/shap")

    log_preds = pipe.predict(X_test)
    preds = add_predictions(test, log_preds)

    mae = mean_absolute_error(np.expm1(y_test), np.expm1(log_preds))
    mape = mean_absolute_percentage_error(np.expm1(y_test), np.expm1(log_preds))
    r_squared = r2_score(np.expm1(y_test), np.expm1(log_preds))
    print(f"[ridge] R² {r_squared*100:,.2f}%")
    print(f"[ridge] MAE €{mae:,.0f} | MAPE {mape*100:,.2f}%")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    preds.sort_values("value_diff").to_csv(output_path, index=False)
    print(f"[ridge] wrote predictions → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ridge model to estimate market value.")
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--test-season", default="2024/25")
    parser.add_argument("--output", default="data/model/big5_predictions_2024-25.csv")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run a coarse grid search over HistGradientBoosting hyperparameters.",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Generate a SHAP global importance plot on a sample of the training data.",
    )
    args = parser.parse_args()
    main(
        dataset_path=args.dataset,
        test_season=args.test_season,
        output_path=args.output,
        run_grid=args.grid,
        save_shap=args.shap,
    )
