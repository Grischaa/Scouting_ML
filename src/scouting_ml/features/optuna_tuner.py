from __future__ import annotations
from typing import Dict, Any

import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score


def tune_lgbm(
    X,
    y,
    n_trials: int = 60,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Optuna tuning for LGBMRegressor on log-target.
    Minimizes RMSE on log(market_value).
    X: transformed numeric feature matrix (after preprocessing).
    """

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "objective": "regression",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
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

        model = LGBMRegressor(**params)
        cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            model,
            X,
            y,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        rmse = np.sqrt(-scores.mean())
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_params.update(
        {
            "objective": "regression",
            "n_jobs": -1,
            "random_state": random_state,
        }
    )
    print(f"[optuna] best LGBM params: {best_params}")
    return best_params
