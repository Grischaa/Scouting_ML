from __future__ import annotations

from typing import Any, Dict, Optional


def make_lgbm(params: Optional[Dict[str, Any]] = None):
    from lightgbm import LGBMRegressor

    base = dict(
        objective="regression_l1",
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
    )
    if params:
        base.update(params)
    return LGBMRegressor(**base)


def make_xgb():
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def make_cat():
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
