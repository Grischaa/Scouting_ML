from __future__ import annotations
from typing import Dict, Any, Optional

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def make_lgbm(params: Optional[Dict[str, Any]] = None) -> LGBMRegressor:
    base = dict(
        objective="regression",
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


def make_xgb() -> XGBRegressor:
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


def make_cat() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
