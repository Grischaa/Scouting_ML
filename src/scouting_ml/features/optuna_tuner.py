from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _select_rows(X, idx):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


def _make_splitter(
    groups: Optional[np.ndarray],
    random_state: int,
):
    if groups is None:
        return KFold(n_splits=3, shuffle=True, random_state=random_state), None

    unique_groups = np.unique(groups)
    n_splits = min(3, len(unique_groups))
    if n_splits < 2:
        return KFold(n_splits=3, shuffle=True, random_state=random_state), None
    return GroupKFold(n_splits=n_splits), groups


def _safe_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if not np.isfinite(denom) or denom <= 1e-9:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _band_wmape_from_log(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> float:
    y_true = np.clip(np.expm1(y_log_true), a_min=0.0, a_max=None)
    y_pred = np.clip(np.expm1(y_log_pred), a_min=0.0, a_max=None)
    bands = [
        (0.0, 5_000_000.0),
        (5_000_000.0, 20_000_000.0),
        (20_000_000.0, float("inf")),
    ]
    scores = []
    for lo, hi in bands:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        scores.append(_safe_wmape(y_true[mask], y_pred[mask]))
    if not scores:
        return _safe_wmape(y_true, y_pred)
    return float(np.mean(scores))


def _lowmid_wmape_from_log(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> float:
    """
    WMAPE objective focused on scouting-heavy ranges:
    - under_5m
    - 5m_to_20m
    Falls back to full-band WMAPE if the split has no low/mid rows.
    """
    y_true = np.clip(np.expm1(y_log_true), a_min=0.0, a_max=None)
    y_pred = np.clip(np.expm1(y_log_pred), a_min=0.0, a_max=None)
    bands = [
        (0.0, 5_000_000.0),
        (5_000_000.0, 20_000_000.0),
    ]
    scores = []
    for lo, hi in bands:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        scores.append(_safe_wmape(y_true[mask], y_pred[mask]))
    if not scores:
        return _band_wmape_from_log(y_log_true, y_log_pred)
    return float(np.mean(scores))


def tune_lgbm(
    X,
    y,
    numeric_cols,
    categorical_cols,
    n_trials: int = 60,
    random_state: int = 42,
    groups: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    optimize_metric: str = "mae",
) -> Dict[str, Any]:
    """
    Optuna tuning for LGBMRegressor on log-target.
    Preprocessing is fit within each CV fold to avoid fold leakage.
    """
    y_arr = np.asarray(y, dtype=float)
    weight_arr = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    splitter, cv_groups = _make_splitter(groups, random_state=random_state)
    metric = optimize_metric.lower().strip()

    def objective(trial: optuna.trial.Trial) -> float:
        objective_name = trial.suggest_categorical(
            "objective", ["regression", "regression_l1", "huber"]
        )
        params = {
            "objective": objective_name,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 600, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "n_jobs": -1,
            "random_state": random_state,
        }
        if objective_name == "huber":
            params["alpha"] = trial.suggest_float("alpha", 0.80, 0.98)

        pre = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), list(numeric_cols)),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("oh", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    list(categorical_cols),
                ),
            ]
        )
        if hasattr(pre, "set_output"):
            pre.set_output(transform="default")
        pipe = Pipeline([("prep", pre), ("model", LGBMRegressor(**params))])

        fold_scores = []
        split_iter = (
            splitter.split(X, y_arr, cv_groups)
            if cv_groups is not None
            else splitter.split(X, y_arr)
        )
        for train_idx, val_idx in split_iter:
            X_train = _select_rows(X, train_idx)
            X_val = _select_rows(X, val_idx)
            y_train = y_arr[train_idx]
            y_val = y_arr[val_idx]

            fit_kwargs: Dict[str, np.ndarray] = {}
            if weight_arr is not None:
                fit_kwargs["model__sample_weight"] = weight_arr[train_idx]

            fold_pipe = clone(pipe)
            fold_pipe.fit(X_train, y_train, **fit_kwargs)
            pred = fold_pipe.predict(X_val)

            if metric == "rmse":
                score = np.sqrt(mean_squared_error(y_val, pred))
            elif metric == "band_wmape":
                score = _band_wmape_from_log(y_val, pred)
            elif metric == "lowmid_wmape":
                score = _lowmid_wmape_from_log(y_val, pred)
            else:
                score = mean_absolute_error(y_val, pred)
            fold_scores.append(float(score))

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_trial.params)
    best_params.update({"n_jobs": -1, "random_state": random_state})
    print(f"[optuna] best LGBM params: {best_params}")
    return best_params
