from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def wmape(y_true, y_pred) -> float:
    """
    Weighted MAPE (a.k.a. MAD/Mean), robust when tiny targets dominate MAPE.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true_arr).sum()
    if denom <= 0:
        return float("nan")
    return float(np.abs(y_true_arr - y_pred_arr).sum() / denom)


def mape_with_floor(y_true, y_pred, min_denom: float = 1_000_000.0) -> float:
    """
    Mean absolute percentage error with denominator floor to avoid exploding
    percentages for very small market values.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.size == 0:
        return float("nan")
    abs_err = np.abs(y_true_arr - y_pred_arr)
    floor = max(float(min_denom), 0.0)
    denom = np.maximum(np.abs(y_true_arr), floor)
    if np.all(denom <= 0):
        return float("nan")
    return float(np.mean(abs_err / denom))


def regression_metrics(y_true, y_pred, mape_min_denom: float = 1_000_000.0) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true_arr - y_pred_arr)
    denom_raw = np.abs(y_true_arr)
    valid_raw = denom_raw > 0
    mape_raw = float(np.mean(abs_err[valid_raw] / denom_raw[valid_raw])) if np.any(valid_raw) else float("nan")
    return {
        "mae_eur": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "mape": float(mape_with_floor(y_true_arr, y_pred_arr, min_denom=mape_min_denom)),
        "mape_raw": mape_raw,
        "mape_min_denom_eur": float(max(float(mape_min_denom), 0.0)),
        "wmape": float(wmape(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }
