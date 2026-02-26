from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from scouting_ml.features.feature_cleaning import clean_train_val_test_for_model
from scouting_ml.features.optuna_tuner import tune_lgbm
from scouting_ml.features.shap_selector import select_top_features
from scouting_ml.models.base_models import make_cat, make_lgbm, make_xgb
from scouting_ml.utils.data_utils import (
    infer_categorical_columns,
    infer_numeric_columns,
    load_dataset,
    split_train_val_test_by_season,
)
from scouting_ml.utils.metrics import regression_metrics
from scouting_ml.utils.shap_utils import save_tree_shap_bar

logger = logging.getLogger(__name__)

POSITIONS = ["GK", "DF", "MF", "FW"]
VALUE_SEGMENTS = [
    ("under_5m", 0.0, 5_000_000.0),
    ("5m_to_20m", 5_000_000.0, 20_000_000.0),
    ("over_20m", 20_000_000.0, float("inf")),
]

LEAKAGE_EXACT_FEATURES = {
    "market_value_eur",
    "market_value",
    "log_market_value",
    "expected_value_eur",
    "expected_value_low_eur",
    "expected_value_high_eur",
    "fair_value_eur",
    "value_diff",
    "value_abs_error",
    "value_gap_eur",
    "value_gap_conservative_eur",
    "undervalued_flag",
    "undervaluation_confidence",
    "undervaluation_score",
}

LEAKAGE_TOKEN_PATTERNS = (
    "market_value",
    "log_market_value",
    "expected_value",
    "fair_value",
    "value_gap",
    "value_diff",
    "abs_error",
    "undervalu",
    "label",
    "future_",
    "next_season",
    "post_season",
)


@dataclass
class PositionMetrics:
    position: str
    split: str
    n_samples: int
    r2: float
    mae: float
    mape: float
    wmape: float


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_proc = SimpleImputer(strategy="median")
    cat_proc = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_proc, num_cols),
            ("cat", cat_proc, cat_cols),
        ]
    )
    # Force numpy/sparse output to keep model input types consistent.
    if hasattr(pre, "set_output"):
        pre.set_output(transform="default")
    return pre


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _filter_features(
    features: list[str],
    exclude_prefixes: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
) -> list[str]:
    prefixes = tuple(exclude_prefixes or [])
    excluded = set(exclude_columns or [])
    out = []
    for col in features:
        if col in excluded:
            continue
        if prefixes and any(col.startswith(p) for p in prefixes):
            continue
        out.append(col)
    return out


def _slugify(value: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return out.strip("_") or "unknown"


def _normalize_league_tokens(raw: Sequence[str] | None) -> set[str]:
    if raw is None:
        return set()
    return {str(x).strip().casefold() for x in raw if str(x).strip()}


def _filter_by_league(
    frame: pd.DataFrame,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> pd.DataFrame:
    if "league" not in frame.columns:
        return frame
    out = frame.copy()
    series = out["league"].astype(str).str.casefold()
    include_set = _normalize_league_tokens(include)
    exclude_set = _normalize_league_tokens(exclude)
    if include_set:
        out = out[series.isin(include_set)].copy()
        series = out["league"].astype(str).str.casefold()
    if exclude_set:
        out = out[~series.isin(exclude_set)].copy()
    return out


def _find_forbidden_feature_columns(columns: Sequence[str]) -> list[str]:
    forbidden: list[str] = []
    for col in columns:
        low = str(col).casefold()
        if col in LEAKAGE_EXACT_FEATURES or low in LEAKAGE_EXACT_FEATURES:
            forbidden.append(col)
            continue
        if any(tok in low for tok in LEAKAGE_TOKEN_PATTERNS):
            forbidden.append(col)
    return sorted(set(forbidden))


def _validate_no_leakage_features(
    features: Sequence[str],
    *,
    strict: bool = True,
) -> list[str]:
    forbidden = _find_forbidden_feature_columns(features)
    if forbidden and strict:
        raise ValueError(
            "Leakage guard failed. Forbidden feature columns detected: "
            + ", ".join(forbidden)
        )
    return forbidden


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    print(f"[warn] dropping duplicate columns: {dup_cols}")
    return df.loc[:, ~df.columns.duplicated()].copy()


def _deduplicate_player_season_rows(df: pd.DataFrame) -> pd.DataFrame:
    dedupe_keys = [
        c
        for c in ["season", "league", "club", "position_group", "player_id", "name"]
        if c in df.columns
    ]
    if len(dedupe_keys) < 4:
        return df

    sort_col = None
    if "minutes" in df.columns:
        sort_col = "minutes"
    elif "sofa_minutesPlayed" in df.columns:
        sort_col = "sofa_minutesPlayed"

    work = df.copy()
    if sort_col is not None:
        work[sort_col] = pd.to_numeric(work[sort_col], errors="coerce")
        work = work.sort_values(sort_col, ascending=False, na_position="last")

    before = len(work)
    work = work.drop_duplicates(subset=dedupe_keys, keep="first")
    removed = before - len(work)
    if removed > 0:
        print(f"[clean] removed {removed:,} duplicate player-season rows")
    return work


def _safe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def _build_data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_rates": {},
        "duplicate_checks": {},
        "league_season_rows": [],
        "scrape_success_rates": {},
        "quality_flags": [],
    }
    if df.empty:
        report["quality_flags"].append("empty_dataset")
        return report

    key_missing_cols = [
        c
        for c in [
            "market_value_eur",
            "log_market_value",
            "minutes",
            "sofa_minutesPlayed",
            "age",
            "season",
            "league",
            "position_group",
        ]
        if c in df.columns
    ]
    for col in key_missing_cols:
        miss = float(pd.to_numeric(df[col], errors="coerce").isna().mean()) if col not in {"season", "league", "position_group"} else float(df[col].isna().mean())
        report["missing_rates"][col] = miss
        if miss > 0.35:
            report["quality_flags"].append(f"high_missing:{col}:{miss:.3f}")

    dedupe_keys = [c for c in ["season", "league", "club", "position_group", "player_id", "name"] if c in df.columns]
    if len(dedupe_keys) >= 4:
        dup_rate = float(df.duplicated(subset=dedupe_keys, keep=False).mean())
        report["duplicate_checks"]["keys"] = dedupe_keys
        report["duplicate_checks"]["duplicate_rate"] = dup_rate
        if dup_rate > 0.03:
            report["quality_flags"].append(f"high_duplicate_rate:{dup_rate:.3f}")

    if {"league", "season"}.issubset(df.columns):
        league_rows = (
            df.groupby(["league", "season"], dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values(["league", "season"])
        )
        report["league_season_rows"] = league_rows.to_dict(orient="records")
        too_small = league_rows[league_rows["n_rows"] < 60]
        for _, row in too_small.iterrows():
            report["quality_flags"].append(
                f"small_league_season:{row['league']}:{row['season']}:{int(row['n_rows'])}"
            )

    scrape_cols = [c for c in df.columns if c.endswith("_scrape_success")]
    for col in scrape_cols:
        rate = _safe_float(pd.to_numeric(df[col], errors="coerce").mean())
        if rate is None:
            continue
        report["scrape_success_rates"][col] = rate
        if rate < 0.85:
            report["quality_flags"].append(f"low_scrape_success:{col}:{rate:.3f}")

    return report


def _apply_league_season_completeness_filter(
    df: pd.DataFrame,
    *,
    min_rows: int,
    min_completeness_ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": bool(min_rows > 0 or min_completeness_ratio > 0),
        "min_rows": int(max(min_rows, 0)),
        "min_completeness_ratio": float(max(min(min_completeness_ratio, 1.0), 0.0)),
        "dropped_groups": [],
        "n_rows_before": int(len(df)),
        "n_rows_after": int(len(df)),
        "n_rows_dropped": 0,
    }
    if df.empty or not {"league", "season"}.issubset(df.columns):
        return df, summary

    min_rows_eff = int(max(min_rows, 0))
    min_ratio_eff = float(max(min(min_completeness_ratio, 1.0), 0.0))
    if min_rows_eff <= 0 and min_ratio_eff <= 0.0:
        return df, summary

    grouped = (
        df.groupby(["league", "season"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    if grouped.empty:
        return df, summary

    grouped["league_max_rows"] = grouped.groupby("league", dropna=False)["n_rows"].transform("max")
    grouped["completeness_ratio"] = grouped["n_rows"] / grouped["league_max_rows"].replace(0, np.nan)
    grouped["completeness_ratio"] = grouped["completeness_ratio"].fillna(0.0)

    keep_mask = pd.Series(True, index=grouped.index)
    if min_rows_eff > 0:
        keep_mask &= grouped["n_rows"] >= min_rows_eff
    if min_ratio_eff > 0:
        keep_mask &= grouped["completeness_ratio"] >= min_ratio_eff

    dropped = grouped.loc[~keep_mask].copy()
    if dropped.empty:
        return df, summary

    drop_key = (
        dropped["league"].astype(str)
        + "||"
        + dropped["season"].astype(str)
    )
    df_key = df["league"].astype(str) + "||" + df["season"].astype(str)
    filtered = df.loc[~df_key.isin(set(drop_key.tolist()))].copy()

    summary["dropped_groups"] = dropped.sort_values(["league", "season"]).to_dict(orient="records")
    summary["n_rows_after"] = int(len(filtered))
    summary["n_rows_dropped"] = int(len(df) - len(filtered))
    return filtered, summary


def _fit_residual_calibration_table(
    val_frame: pd.DataFrame,
    *,
    min_samples: int = 30,
) -> dict[str, Any]:
    if val_frame.empty:
        return {
            "enabled": False,
            "min_samples": int(max(min_samples, 1)),
            "global_adjustment_eur": 0.0,
            "level1": {},
            "level2": {},
            "level3": {},
            "level4": {},
        }

    work = val_frame.copy()
    if "expected_value_eur" not in work.columns or "market_value_eur" not in work.columns:
        return {
            "enabled": False,
            "min_samples": int(max(min_samples, 1)),
            "global_adjustment_eur": 0.0,
            "level1": {},
            "level2": {},
            "level3": {},
            "level4": {},
        }

    work = _assign_value_segments(work)
    work["league_norm"] = work.get("league", pd.Series("", index=work.index)).astype(str).str.strip().str.casefold()
    work["position_norm"] = work.get("model_position", work.get("position_group", pd.Series("", index=work.index))).astype(str).str.upper()
    work["pred_eur"] = pd.to_numeric(work["expected_value_eur"], errors="coerce")
    work["market_eur"] = pd.to_numeric(work["market_value_eur"], errors="coerce")
    work["residual_eur"] = work["market_eur"] - work["pred_eur"]
    work = work[np.isfinite(work["residual_eur"])].copy()
    if work.empty:
        return {
            "enabled": False,
            "min_samples": int(max(min_samples, 1)),
            "global_adjustment_eur": 0.0,
            "level1": {},
            "level2": {},
            "level3": {},
            "level4": {},
        }

    min_samples_eff = int(max(min_samples, 1))

    def _group_map(cols: list[str], threshold: int) -> dict[str, float]:
        grouped = (
            work.groupby(cols, dropna=False)["residual_eur"]
            .agg(median_residual="median", n="count")
            .reset_index()
        )
        grouped = grouped[grouped["n"] >= int(max(threshold, 1))].copy()
        out: dict[str, float] = {}
        for _, row in grouped.iterrows():
            key = "||".join(str(row[c]) for c in cols)
            # Conservative one-sided calibration: only reduce optimistic values.
            out[key] = float(min(float(row["median_residual"]), 0.0))
        return out

    level1 = _group_map(["league_norm", "position_norm", "value_segment"], threshold=min_samples_eff)
    level2 = _group_map(["league_norm", "position_norm"], threshold=max(min_samples_eff // 2, 12))
    level3 = _group_map(["position_norm", "value_segment"], threshold=max(min_samples_eff // 2, 12))
    level4 = _group_map(["value_segment"], threshold=max(min_samples_eff // 3, 8))
    global_adjustment = 0.0
    if len(work) >= min_samples_eff:
        global_adjustment = float(min(float(work["residual_eur"].median()), 0.0))

    return {
        "enabled": True,
        "min_samples": min_samples_eff,
        "global_adjustment_eur": global_adjustment,
        "level1": level1,
        "level2": level2,
        "level3": level3,
        "level4": level4,
    }


def _apply_residual_calibration_table(
    frame: pd.DataFrame,
    table: dict[str, Any],
) -> pd.DataFrame:
    out = _assign_value_segments(frame)
    out["league_norm"] = out.get("league", pd.Series("", index=out.index)).astype(str).str.strip().str.casefold()
    out["position_norm"] = out.get("model_position", out.get("position_group", pd.Series("", index=out.index))).astype(str).str.upper()

    if not table.get("enabled", False):
        out["expected_value_raw_eur"] = pd.to_numeric(out.get("expected_value_eur"), errors="coerce")
        out["expected_value_calibration_eur"] = 0.0
        out["residual_calibration_applied"] = 0
        return out

    lvl1 = table.get("level1", {}) or {}
    lvl2 = table.get("level2", {}) or {}
    lvl3 = table.get("level3", {}) or {}
    lvl4 = table.get("level4", {}) or {}
    global_adj = float(table.get("global_adjustment_eur", 0.0) or 0.0)

    def _lookup(row: pd.Series) -> float:
        league = str(row.get("league_norm", ""))
        pos = str(row.get("position_norm", ""))
        seg = str(row.get("value_segment", "unknown"))
        k1 = f"{league}||{pos}||{seg}"
        k2 = f"{league}||{pos}"
        k3 = f"{pos}||{seg}"
        k4 = f"{seg}"
        if k1 in lvl1:
            return float(lvl1[k1])
        if k2 in lvl2:
            return float(lvl2[k2])
        if k3 in lvl3:
            return float(lvl3[k3])
        if k4 in lvl4:
            return float(lvl4[k4])
        return global_adj

    out["expected_value_raw_eur"] = pd.to_numeric(out.get("expected_value_eur"), errors="coerce")
    out["expected_value_calibration_eur"] = out.apply(_lookup, axis=1)
    out["expected_value_eur"] = (
        out["expected_value_raw_eur"]
        + pd.to_numeric(out["expected_value_calibration_eur"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)

    if "expected_value_low_eur" in out.columns:
        low_raw = pd.to_numeric(out["expected_value_low_eur"], errors="coerce")
        out["expected_value_low_raw_eur"] = low_raw
        out["expected_value_low_eur"] = (low_raw + out["expected_value_calibration_eur"]).clip(lower=0.0)
    if "expected_value_high_eur" in out.columns:
        high_raw = pd.to_numeric(out["expected_value_high_eur"], errors="coerce")
        out["expected_value_high_raw_eur"] = high_raw
        out["expected_value_high_eur"] = high_raw + out["expected_value_calibration_eur"]

    out["residual_calibration_applied"] = (
        pd.to_numeric(out["expected_value_calibration_eur"], errors="coerce").fillna(0.0).abs() > 0.0
    ).astype(int)

    market = pd.to_numeric(out.get("market_value_eur"), errors="coerce")
    out["value_diff"] = market - out["expected_value_eur"]
    out["value_abs_error"] = np.abs(out["value_diff"])
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_quality_report(report: dict[str, Any]) -> None:
    print("[quality] rows:", report.get("n_rows", 0))
    print("[quality] cols:", report.get("n_columns", 0))
    flags = report.get("quality_flags", [])
    if flags:
        print("[quality] flags:")
        for flag in flags:
            print("  -", flag)
    else:
        print("[quality] flags: none")


def _compute_recency_weights(train_df: pd.DataFrame, half_life: float) -> np.ndarray:
    if half_life <= 0:
        return np.ones(len(train_df), dtype=float)
    if "season_end_year" not in train_df.columns:
        return np.ones(len(train_df), dtype=float)

    years = pd.to_numeric(train_df["season_end_year"], errors="coerce")
    if years.notna().sum() == 0:
        return np.ones(len(train_df), dtype=float)

    max_year = years.max()
    age = (max_year - years).clip(lower=0)
    weights = np.power(0.5, age / half_life)
    return weights.fillna(1.0).to_numpy(dtype=float)


def _compute_value_segment_weights(
    train_df: pd.DataFrame,
    under_5m_weight: float,
    mid_5m_to_20m_weight: float,
    over_20m_weight: float,
) -> np.ndarray:
    if "market_value_eur" not in train_df.columns:
        return np.ones(len(train_df), dtype=float)

    values = pd.to_numeric(train_df["market_value_eur"], errors="coerce")
    weights = np.ones(len(train_df), dtype=float)

    under_mask = (values >= 0) & (values < 5_000_000.0)
    mid_mask = (values >= 5_000_000.0) & (values < 20_000_000.0)
    over_mask = values >= 20_000_000.0

    weights[under_mask.fillna(False).to_numpy()] = float(under_5m_weight)
    weights[mid_mask.fillna(False).to_numpy()] = float(mid_5m_to_20m_weight)
    weights[over_mask.fillna(False).to_numpy()] = float(over_20m_weight)
    return weights


def _as_model_input(X):
    """Convert pandas outputs to plain arrays for consistent estimator behavior."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return X


def _safe_model_predict(model, X):
    """Predict while silencing known LightGBM feature-name warning noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
        )
        return model.predict(X)


def _filter_target_notna(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = pd.to_numeric(raw_df["log_market_value"], errors="coerce")
    mask = target.notna()
    return raw_df.loc[mask].copy(), clean_df.loc[mask].copy()


def compute_position_metrics(
    pos: str,
    split: str,
    y_true_lin: np.ndarray,
    y_pred_lin: np.ndarray,
    mape_min_denom_eur: float = 1_000_000.0,
) -> PositionMetrics:
    metrics = regression_metrics(y_true_lin, y_pred_lin, mape_min_denom=mape_min_denom_eur)
    n_samples = int(len(y_true_lin))
    print(
        f"[{split}:{pos}] n={n_samples} | "
        f"R² {metrics['r2']*100:,.2f}% | "
        f"MAE €{metrics['mae_eur']:,.0f} | "
        f"MAPE {metrics['mape']*100:,.2f}% | "
        f"WMAPE {metrics['wmape']*100:,.2f}%"
    )
    return PositionMetrics(
        position=pos,
        split=split,
        n_samples=n_samples,
        r2=metrics["r2"],
        mae=metrics["mae_eur"],
        mape=metrics["mape"],
        wmape=metrics["wmape"],
    )


def evaluate_value_segments(
    frame: pd.DataFrame,
    split_name: str,
    mape_min_denom_eur: float = 1_000_000.0,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    y_true_all = pd.to_numeric(frame["market_value_eur"], errors="coerce")
    y_pred_all = pd.to_numeric(frame["expected_value_eur"], errors="coerce")

    for label, lo, hi in VALUE_SEGMENTS:
        mask = (y_true_all >= lo) & (y_true_all < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "split": split_name,
                    "segment": label,
                    "lower_eur": lo,
                    "upper_eur": hi,
                    "n_samples": 0,
                    "r2": float("nan"),
                    "mae_eur": float("nan"),
                    "mape": float("nan"),
                    "mape_raw": float("nan"),
                    "mape_min_denom_eur": float(max(float(mape_min_denom_eur), 0.0)),
                    "wmape": float("nan"),
                }
            )
            continue

        seg_metrics = regression_metrics(
            y_true_all[mask].to_numpy(),
            y_pred_all[mask].to_numpy(),
            mape_min_denom=mape_min_denom_eur,
        )
        rows.append(
            {
                "split": split_name,
                "segment": label,
                "lower_eur": lo,
                "upper_eur": hi,
                "n_samples": n,
                **seg_metrics,
            }
        )
    return rows


def _segment_labels() -> list[str]:
    return [label for label, _, _ in VALUE_SEGMENTS]


def _value_band_labels_from_eur(values: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    labels = np.full(arr.shape, "unknown", dtype=object)
    for label, lo, hi in VALUE_SEGMENTS:
        mask = (arr >= lo) & (arr < hi)
        labels[mask] = label
    return labels.astype(str)


def _assign_value_segments(
    frame: pd.DataFrame,
    value_col: str = "market_value_eur",
) -> pd.DataFrame:
    out = frame.copy()
    values = pd.to_numeric(out[value_col], errors="coerce")
    out["value_segment"] = "unknown"
    for label, lo, hi in VALUE_SEGMENTS:
        mask = (values >= lo) & (values < hi)
        out.loc[mask, "value_segment"] = label
    return out


def fit_error_priors(
    val_frame: pd.DataFrame,
    interval_q: float = 0.8,
) -> pd.DataFrame:
    """
    Build position x value-segment uncertainty priors from validation residuals.
    These priors are then applied to both val/test predictions as confidence bands.
    """
    if val_frame.empty:
        return pd.DataFrame(
            columns=[
                "model_position",
                "value_segment",
                "prior_mae_eur",
                "prior_medae_eur",
                "prior_p75ae_eur",
                "n_samples",
            ]
        )

    tmp = _assign_value_segments(val_frame)
    tmp["abs_error"] = np.abs(
        pd.to_numeric(tmp["market_value_eur"], errors="coerce")
        - pd.to_numeric(tmp["expected_value_eur"], errors="coerce")
    )
    q = float(interval_q)
    q = min(max(q, 0.5), 0.99)
    priors = (
        tmp.groupby(["model_position", "value_segment"], dropna=False)["abs_error"]
        .agg(
            prior_mae_eur="mean",
            prior_medae_eur="median",
            prior_p75ae_eur=lambda s: float(s.quantile(0.75)),
            prior_qae_eur=lambda s: float(s.quantile(q)),
            n_samples="count",
        )
        .reset_index()
    )
    priors["prior_interval_q"] = q
    return priors


def apply_confidence_scoring(
    frame: pd.DataFrame,
    priors: pd.DataFrame,
    interval_q: float = 0.8,
) -> pd.DataFrame:
    out = _assign_value_segments(frame)
    if out.empty:
        return out

    q = float(interval_q)
    q = min(max(q, 0.5), 0.99)
    if priors.empty:
        fallback_uncertainty = float(
            np.abs(
                pd.to_numeric(out["market_value_eur"], errors="coerce")
                - pd.to_numeric(out["expected_value_eur"], errors="coerce")
            ).median()
        )
        if not np.isfinite(fallback_uncertainty) or fallback_uncertainty <= 0:
            fallback_uncertainty = 1.0
        out["prior_mae_eur"] = fallback_uncertainty
        out["prior_medae_eur"] = fallback_uncertainty
        out["prior_p75ae_eur"] = fallback_uncertainty
        out["prior_qae_eur"] = fallback_uncertainty
        out["prior_interval_q"] = q
    else:
        out = out.merge(priors, on=["model_position", "value_segment"], how="left")
        fallback_mae = float(pd.to_numeric(priors["prior_mae_eur"], errors="coerce").median())
        fallback_medae = float(pd.to_numeric(priors["prior_medae_eur"], errors="coerce").median())
        fallback_p75 = float(pd.to_numeric(priors["prior_p75ae_eur"], errors="coerce").median())
        if "prior_qae_eur" in priors.columns:
            fallback_q = float(pd.to_numeric(priors["prior_qae_eur"], errors="coerce").median())
        else:
            fallback_q = float("nan")
        if not np.isfinite(fallback_mae) or fallback_mae <= 0:
            fallback_mae = 1.0
        if not np.isfinite(fallback_medae) or fallback_medae <= 0:
            fallback_medae = fallback_mae
        if not np.isfinite(fallback_p75) or fallback_p75 <= 0:
            fallback_p75 = fallback_mae
        if not np.isfinite(fallback_q) or fallback_q <= 0:
            fallback_q = fallback_p75
        out["prior_mae_eur"] = pd.to_numeric(out["prior_mae_eur"], errors="coerce").fillna(fallback_mae)
        out["prior_medae_eur"] = pd.to_numeric(out["prior_medae_eur"], errors="coerce").fillna(fallback_medae)
        out["prior_p75ae_eur"] = pd.to_numeric(out["prior_p75ae_eur"], errors="coerce").fillna(fallback_p75)
        out["prior_qae_eur"] = pd.to_numeric(out["prior_qae_eur"], errors="coerce").fillna(fallback_q)
        out["prior_interval_q"] = pd.to_numeric(out["prior_interval_q"], errors="coerce").fillna(q)

    out["market_value_eur"] = pd.to_numeric(out["market_value_eur"], errors="coerce")
    out["expected_value_eur"] = pd.to_numeric(out["expected_value_eur"], errors="coerce")
    out["fair_value_eur"] = out["expected_value_eur"]

    uncertainty = out["prior_qae_eur"].clip(lower=1.0)
    out["expected_value_low_eur"] = np.clip(out["expected_value_eur"] - uncertainty, a_min=0.0, a_max=None)
    out["expected_value_high_eur"] = out["expected_value_eur"] + uncertainty

    # Positive gap means model sees player as undervalued.
    out["value_gap_eur"] = out["expected_value_eur"] - out["market_value_eur"]
    out["value_gap_conservative_eur"] = out["expected_value_low_eur"] - out["market_value_eur"]
    out["undervaluation_confidence"] = out["value_gap_eur"] / uncertainty.replace(0, np.nan)
    out["undervalued_flag"] = (out["value_gap_conservative_eur"] > 0).astype(int)
    gap_cons = out["value_gap_conservative_eur"].clip(lower=0.0).fillna(0.0)
    conf = out["undervaluation_confidence"].clip(lower=0.0).fillna(0.0)
    if "minutes" in out.columns:
        mins = pd.to_numeric(out["minutes"], errors="coerce")
    elif "sofa_minutesPlayed" in out.columns:
        mins = pd.to_numeric(out["sofa_minutesPlayed"], errors="coerce")
    else:
        mins = pd.Series(np.nan, index=out.index, dtype=float)
    reliability = (mins.fillna(0.0) / 1800.0).clip(lower=0.25, upper=1.25)
    if "age" in out.columns:
        age_num = pd.to_numeric(out["age"], errors="coerce").fillna(26.0)
    else:
        age_num = pd.Series(26.0, index=out.index, dtype=float)
    age_factor = np.where(age_num <= 23, 1.1, np.where(age_num <= 26, 1.0, 0.9))
    out["undervaluation_score"] = (
        (gap_cons / 1_000_000.0) * np.log1p(conf) * reliability * age_factor
    )
    out["interval_contains_truth"] = (
        (out["market_value_eur"] >= out["expected_value_low_eur"])
        & (out["market_value_eur"] <= out["expected_value_high_eur"])
    ).astype(int)
    return out


def _interval_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "interval_coverage": float("nan"),
            "interval_avg_width_eur": float("nan"),
            "interval_median_width_eur": float("nan"),
        }
    required = {"expected_value_low_eur", "expected_value_high_eur", "market_value_eur"}
    if not required.issubset(frame.columns):
        return {
            "interval_coverage": float("nan"),
            "interval_avg_width_eur": float("nan"),
            "interval_median_width_eur": float("nan"),
        }
    low = pd.to_numeric(frame["expected_value_low_eur"], errors="coerce")
    high = pd.to_numeric(frame["expected_value_high_eur"], errors="coerce")
    truth = pd.to_numeric(frame["market_value_eur"], errors="coerce")
    width = (high - low).clip(lower=0.0)
    coverage = ((truth >= low) & (truth <= high)).mean()
    return {
        "interval_coverage": float(coverage),
        "interval_avg_width_eur": float(width.mean()),
        "interval_median_width_eur": float(width.median()),
    }


def _compute_league_shift_report(
    reference_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> dict[str, Any]:
    features = [
        c
        for c in [
            "age",
            "minutes",
            "sofa_goals_per90",
            "sofa_assists_per90",
            "sofa_expectedGoals_per90",
            "sofa_tackles_per90",
            "sofa_interceptions_per90",
            "clubctx_club_strength_proxy",
            "leaguectx_league_strength_index",
            "uefa_coeff_points",
        ]
        if c in reference_df.columns and c in holdout_df.columns
    ]
    if not features:
        return {"n_features": 0, "mean_abs_shift_z": float("nan"), "features": []}

    rows = []
    for col in features:
        ref = pd.to_numeric(reference_df[col], errors="coerce")
        cur = pd.to_numeric(holdout_df[col], errors="coerce")
        ref_mean = float(ref.mean())
        cur_mean = float(cur.mean())
        ref_std = float(ref.std())
        if not np.isfinite(ref_std) or ref_std <= 1e-9:
            shift = float("nan")
        else:
            shift = abs(cur_mean - ref_mean) / ref_std
        rows.append(
            {
                "feature": col,
                "ref_mean": ref_mean,
                "holdout_mean": cur_mean,
                "ref_std": ref_std,
                "abs_shift_z": shift,
            }
        )
    mean_shift = float(np.nanmean([r["abs_shift_z"] for r in rows]))
    return {
        "n_features": len(rows),
        "mean_abs_shift_z": mean_shift,
        "features": rows,
    }


def train_for_position(
    df_raw: pd.DataFrame,
    pos: str,
    val_season: str,
    test_season: str,
    shap_dir: Path,
    n_optuna_trials: int = 60,
    recency_half_life: float = 2.0,
    under_5m_weight: float = 1.0,
    mid_5m_to_20m_weight: float = 1.0,
    over_20m_weight: float = 1.0,
    exclude_prefixes: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
    optimize_metric: str = "lowmid_wmape",
    strict_leakage_guard: bool = True,
    two_stage_band_model: bool = True,
    band_min_samples: int = 120,
    band_blend_alpha: float = 0.70,
    mape_min_denom_eur: float = 1_000_000.0,
    train_exclude_leagues: Sequence[str] | None = None,
    val_exclude_leagues: Sequence[str] | None = None,
    test_include_leagues: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, List[PositionMetrics]] | None:
    df_pos_raw = df_raw[df_raw["position_group"] == pos].copy()
    if len(df_pos_raw) < 250:
        print(f"[warn] skipping {pos}: too few samples ({len(df_pos_raw)})")
        return None

    train_raw, val_raw, test_raw = split_train_val_test_by_season(
        df_pos_raw,
        val_season,
        test_season,
    )

    train_raw = _filter_by_league(train_raw, exclude=train_exclude_leagues)
    val_raw = _filter_by_league(val_raw, exclude=val_exclude_leagues)
    test_raw = _filter_by_league(test_raw, include=test_include_leagues)

    if train_raw.empty or val_raw.empty or test_raw.empty:
        print(
            f"[warn] skipping {pos}: empty split "
            f"(train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)})"
        )
        return None

    train_df, val_df, test_df = clean_train_val_test_for_model(train_raw, val_raw, test_raw)
    train_raw, train_df = _filter_target_notna(train_raw, train_df)
    val_raw, val_df = _filter_target_notna(val_raw, val_df)
    test_raw, test_df = _filter_target_notna(test_raw, test_df)

    if train_raw.empty or val_raw.empty or test_raw.empty:
        print(
            f"[warn] skipping {pos}: no usable rows after target filtering "
            f"(train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)})"
        )
        return None

    band_blend_alpha = float(np.clip(band_blend_alpha, 0.0, 1.0))

    blocked = set(LEAKAGE_EXACT_FEATURES)
    blocked.update({"log_market_value", "expected_value_eur", "value_diff", "value_abs_error"})
    num_cols = infer_numeric_columns(train_df, blocked=blocked)
    cat_cols = [c for c in infer_categorical_columns(train_df) if c not in blocked]
    feat_cols_raw = [c for c in (num_cols + cat_cols) if c in train_df.columns]
    feat_cols = _filter_features(
        feat_cols_raw,
        exclude_prefixes=exclude_prefixes,
        exclude_columns=exclude_columns,
    )
    forbidden_cols = _find_forbidden_feature_columns(feat_cols)
    if forbidden_cols:
        feat_cols = [c for c in feat_cols if c not in set(forbidden_cols)]
        print(
            f"[{pos}] leakage-guard dropped {len(forbidden_cols)} feature(s): "
            + ", ".join(forbidden_cols[:12])
            + (" ..." if len(forbidden_cols) > 12 else "")
        )
    _validate_no_leakage_features(feat_cols, strict=strict_leakage_guard)
    removed = len(feat_cols_raw) - len(feat_cols)
    if removed > 0:
        print(f"[{pos}] excluded {removed} features via --exclude-* filters")
    num_cols = [c for c in num_cols if c in feat_cols]
    cat_cols = [c for c in cat_cols if c in feat_cols]

    if not feat_cols:
        print(f"[warn] no features for {pos}, skipping.")
        return None

    X_train_frame = train_df[feat_cols]
    y_train = pd.to_numeric(train_raw["log_market_value"], errors="coerce").to_numpy(dtype=float)
    y_val = pd.to_numeric(val_raw["log_market_value"], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test_raw["log_market_value"], errors="coerce").to_numpy(dtype=float)
    y_train_eur = np.clip(np.expm1(y_train), a_min=0.0, a_max=None)
    y_val_eur = np.clip(np.expm1(y_val), a_min=0.0, a_max=None)
    y_test_eur = np.clip(np.expm1(y_test), a_min=0.0, a_max=None)

    groups = None
    if "season_end_year" in train_raw.columns:
        group_vals = pd.to_numeric(train_raw["season_end_year"], errors="coerce")
        if group_vals.notna().sum() >= 2:
            groups = group_vals.fillna(-1).to_numpy()
    elif "season" in train_raw.columns:
        groups = train_raw["season"].astype(str).to_numpy()

    recency_weight = _compute_recency_weights(train_raw, half_life=recency_half_life)
    segment_weight = _compute_value_segment_weights(
        train_raw,
        under_5m_weight=under_5m_weight,
        mid_5m_to_20m_weight=mid_5m_to_20m_weight,
        over_20m_weight=over_20m_weight,
    )
    sample_weight = recency_weight * segment_weight
    sample_weight = np.where(np.isfinite(sample_weight), sample_weight, 1.0)
    sample_weight = np.clip(sample_weight, a_min=1e-6, a_max=None)

    best_lgbm_params = tune_lgbm(
        X_train_frame,
        y_train,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        n_trials=n_optuna_trials,
        groups=groups,
        sample_weight=sample_weight,
        optimize_metric=optimize_metric,
    )

    pre_full = build_preprocessor(num_cols, cat_cols)
    X_train_full = _as_model_input(pre_full.fit_transform(X_train_frame))
    lgbm_full = make_lgbm(best_lgbm_params)
    lgbm_full.fit(X_train_full, y_train, sample_weight=sample_weight)

    num_sel, cat_sel = select_top_features(
        lgbm_full,
        pre_full,
        X_train_frame,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        top_n=25,
    )
    sel_cols: List[str] = num_sel + cat_sel
    if not sel_cols:
        sel_cols = feat_cols
        num_sel = num_cols
        cat_sel = cat_cols

    pre_sel = build_preprocessor(num_sel, cat_sel)
    X_train = _as_model_input(pre_sel.fit_transform(train_df[sel_cols]))
    X_val = _as_model_input(pre_sel.transform(val_df[sel_cols]))
    X_test = _as_model_input(pre_sel.transform(test_df[sel_cols]))

    if pos == "GK":
        lgbm = make_lgbm(best_lgbm_params)
        lgbm.fit(X_train, y_train, sample_weight=sample_weight)
        save_tree_shap_bar(lgbm, pre_sel, train_df[sel_cols], shap_dir / f"shap_{pos}.png")
        val_log_pred = _safe_model_predict(lgbm, X_val)
        test_log_pred = _safe_model_predict(lgbm, X_test)
    else:
        lgbm = make_lgbm(best_lgbm_params)
        xgb = make_xgb()
        cat = make_cat()

        lgbm.fit(X_train, y_train, sample_weight=sample_weight)
        xgb.fit(X_train, y_train, sample_weight=sample_weight)
        cat.fit(X_train, y_train, sample_weight=sample_weight)

        save_tree_shap_bar(lgbm, pre_sel, train_df[sel_cols], shap_dir / f"shap_{pos}.png")

        P_train = np.vstack(
            [
                _safe_model_predict(lgbm, X_train),
                xgb.predict(X_train),
                cat.predict(X_train),
            ]
        ).T
        P_val = np.vstack(
            [
                _safe_model_predict(lgbm, X_val),
                xgb.predict(X_val),
                cat.predict(X_val),
            ]
        ).T
        P_test = np.vstack(
            [
                _safe_model_predict(lgbm, X_test),
                xgb.predict(X_test),
                cat.predict(X_test),
            ]
        ).T

        meta = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=3,
            random_state=42,
        )
        try:
            meta.fit(P_train, y_train, sample_weight=sample_weight)
        except TypeError:
            meta.fit(P_train, y_train)

        val_log_pred = meta.predict(P_val)
        test_log_pred = meta.predict(P_test)

    base_val_log_pred = np.asarray(val_log_pred, dtype=float)
    base_test_log_pred = np.asarray(test_log_pred, dtype=float)
    routed_val_log_pred = base_val_log_pred.copy()
    routed_test_log_pred = base_test_log_pred.copy()

    band_train_true = _value_band_labels_from_eur(y_train_eur)
    band_val_true = _value_band_labels_from_eur(y_val_eur)
    band_test_true = _value_band_labels_from_eur(y_test_eur)
    band_val_pred = np.full(len(y_val), "global_only", dtype=object)
    band_test_pred = np.full(len(y_test), "global_only", dtype=object)
    val_band_model_used = np.zeros(len(y_val), dtype=int)
    test_band_model_used = np.zeros(len(y_test), dtype=int)
    val_low_value_model_used = np.zeros(len(y_val), dtype=int)
    test_low_value_model_used = np.zeros(len(y_test), dtype=int)
    use_two_stage = bool(two_stage_band_model and pos != "GK")

    if two_stage_band_model and not use_two_stage:
        print(f"[{pos}] two-stage routing disabled for GK (base model only)")

    if use_two_stage:
        # Stage 1: classify value band to route the downstream regressor.
        if len(np.unique(band_train_true)) >= 2:
            band_clf = LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=42,
            )
            band_clf.fit(X_train, band_train_true)
            band_val_pred = np.asarray(band_clf.predict(X_val), dtype=object)
            band_test_pred = np.asarray(band_clf.predict(X_test), dtype=object)
            val_band_acc = float(np.mean(band_val_pred == band_val_true))
            test_band_acc = float(np.mean(band_test_pred == band_test_true))
            print(
                f"[{pos}] band classifier accuracy | "
                f"val={val_band_acc*100:.2f}% test={test_band_acc*100:.2f}%"
            )
        else:
            default_band = str(pd.Series(band_train_true).mode().iloc[0])
            band_val_pred = np.full(len(y_val), default_band, dtype=object)
            band_test_pred = np.full(len(y_test), default_band, dtype=object)
            print(f"[{pos}] band classifier fallback: single train band '{default_band}'")

        # Stage 2: train one regressor per value band and blend with base model.
        band_models: dict[str, Any] = {}
        for band_label in _segment_labels():
            band_mask = band_train_true == band_label
            n_band = int(np.sum(band_mask))
            if n_band < int(band_min_samples):
                continue
            band_model = make_lgbm(best_lgbm_params)
            band_model.fit(
                X_train[band_mask],
                y_train[band_mask],
                sample_weight=sample_weight[band_mask],
            )
            band_models[band_label] = band_model
        if band_models:
            print(
                f"[{pos}] band experts trained: "
                + ", ".join(f"{k}" for k in sorted(band_models.keys()))
            )
            alpha = float(band_blend_alpha)
            for band_label, band_model in band_models.items():
                mask_val = band_val_pred == band_label
                if np.any(mask_val):
                    pred_val_band = _safe_model_predict(band_model, X_val[mask_val])
                    routed_val_log_pred[mask_val] = (
                        (1.0 - alpha) * routed_val_log_pred[mask_val]
                        + alpha * pred_val_band
                    )
                    val_band_model_used[mask_val] = 1

                mask_test = band_test_pred == band_label
                if np.any(mask_test):
                    pred_test_band = _safe_model_predict(band_model, X_test[mask_test])
                    routed_test_log_pred[mask_test] = (
                        (1.0 - alpha) * routed_test_log_pred[mask_test]
                        + alpha * pred_test_band
                    )
                    test_band_model_used[mask_test] = 1

            print(
                f"[{pos}] routed with band experts | "
                f"val_used={val_band_model_used.mean()*100:.1f}% "
                f"test_used={test_band_model_used.mean()*100:.1f}% "
                f"(alpha={alpha:.2f})"
            )
        else:
            print(
                f"[{pos}] no band experts met min samples ({band_min_samples}), "
                "using base model only"
            )

        # Stage 3: dedicated low-value expert (< €20m) to improve scouting-heavy segment.
        low_value_threshold_eur = 20_000_000.0
        low_value_labels = ("under_5m", "5m_to_20m")
        low_value_min_samples = max(int(band_min_samples), 220)
        low_value_alpha = float(np.clip(band_blend_alpha * 0.75, 0.15, 0.60))
        low_train_mask = y_train_eur < low_value_threshold_eur
        n_low_train = int(np.sum(low_train_mask))
        if n_low_train >= low_value_min_samples:
            low_value_model = make_lgbm(best_lgbm_params)
            low_value_model.fit(
                X_train[low_train_mask],
                y_train[low_train_mask],
                sample_weight=sample_weight[low_train_mask],
            )
            low_mask_val = np.isin(band_val_pred, low_value_labels)
            low_mask_test = np.isin(band_test_pred, low_value_labels)

            if np.any(low_mask_val):
                pred_val_low = _safe_model_predict(low_value_model, X_val[low_mask_val])
                routed_val_log_pred[low_mask_val] = (
                    (1.0 - low_value_alpha) * routed_val_log_pred[low_mask_val]
                    + low_value_alpha * pred_val_low
                )
                val_low_value_model_used[low_mask_val] = 1
            if np.any(low_mask_test):
                pred_test_low = _safe_model_predict(low_value_model, X_test[low_mask_test])
                routed_test_log_pred[low_mask_test] = (
                    (1.0 - low_value_alpha) * routed_test_log_pred[low_mask_test]
                    + low_value_alpha * pred_test_low
                )
                test_low_value_model_used[low_mask_test] = 1

            print(
                f"[{pos}] low-value expert routed (<€20m) | "
                f"val_used={val_low_value_model_used.mean()*100:.1f}% "
                f"test_used={test_low_value_model_used.mean()*100:.1f}% "
                f"(alpha={low_value_alpha:.2f})"
            )
        else:
            print(
                f"[{pos}] low-value expert skipped: only {n_low_train} train rows "
                f"(min={low_value_min_samples})"
            )

    val_log_pred = routed_val_log_pred
    test_log_pred = routed_test_log_pred
    val_pred = np.clip(np.expm1(val_log_pred), a_min=0.0, a_max=None)
    test_pred = np.clip(np.expm1(test_log_pred), a_min=0.0, a_max=None)
    val_true = y_val_eur
    test_true = y_test_eur

    val_metrics = compute_position_metrics(
        pos,
        "val",
        val_true,
        val_pred,
        mape_min_denom_eur=mape_min_denom_eur,
    )
    test_metrics = compute_position_metrics(
        pos,
        "test",
        test_true,
        test_pred,
        mape_min_denom_eur=mape_min_denom_eur,
    )

    val_out = val_raw.copy()
    val_out["expected_value_eur"] = val_pred
    val_out["value_diff"] = val_out["market_value_eur"] - val_out["expected_value_eur"]
    val_out["value_abs_error"] = np.abs(val_out["value_diff"])
    val_out["model_position"] = pos
    val_out["value_band_true"] = band_val_true
    val_out["value_band_pred"] = band_val_pred
    val_out["band_model_used"] = val_band_model_used
    val_out["low_value_model_used"] = val_low_value_model_used

    test_out = test_raw.copy()
    test_out["expected_value_eur"] = test_pred
    test_out["value_diff"] = test_out["market_value_eur"] - test_out["expected_value_eur"]
    test_out["value_abs_error"] = np.abs(test_out["value_diff"])
    test_out["model_position"] = pos
    test_out["value_band_true"] = band_test_true
    test_out["value_band_pred"] = band_test_pred
    test_out["band_model_used"] = test_band_model_used
    test_out["low_value_model_used"] = test_low_value_model_used

    return val_out, test_out, [val_metrics, test_metrics]


def _run_league_holdout_suite(
    df_raw: pd.DataFrame,
    holdout_leagues: Sequence[str],
    *,
    val_season: str,
    test_season: str,
    output_path: Path,
    shap_dir: Path,
    n_optuna_trials: int,
    recency_half_life: float,
    under_5m_weight: float,
    mid_5m_to_20m_weight: float,
    over_20m_weight: float,
    exclude_prefixes: Sequence[str] | None,
    exclude_columns: Sequence[str] | None,
    optimize_metric: str,
    strict_leakage_guard: bool,
    interval_q: float,
    two_stage_band_model: bool,
    band_min_samples: int,
    band_blend_alpha: float,
    mape_min_denom_eur: float,
) -> list[dict[str, Any]]:
    holdout_results: list[dict[str, Any]] = []
    if not holdout_leagues:
        return holdout_results

    normalized = []
    seen = set()
    for league in holdout_leagues:
        key = str(league).strip()
        if not key:
            continue
        low = key.casefold()
        if low in seen:
            continue
        seen.add(low)
        normalized.append(key)

    for holdout_league in normalized:
        print("\n======================")
        print(f"  LEAGUE HOLDOUT: {holdout_league}")
        print("======================")
        val_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []

        for pos in POSITIONS:
            result = train_for_position(
                df_raw,
                pos,
                val_season,
                test_season,
                shap_dir,
                n_optuna_trials=n_optuna_trials,
                recency_half_life=recency_half_life,
                under_5m_weight=under_5m_weight,
                mid_5m_to_20m_weight=mid_5m_to_20m_weight,
                over_20m_weight=over_20m_weight,
                exclude_prefixes=exclude_prefixes,
                exclude_columns=exclude_columns,
                optimize_metric=optimize_metric,
                strict_leakage_guard=strict_leakage_guard,
                two_stage_band_model=two_stage_band_model,
                band_min_samples=band_min_samples,
                band_blend_alpha=band_blend_alpha,
                mape_min_denom_eur=mape_min_denom_eur,
                train_exclude_leagues=[holdout_league],
                val_exclude_leagues=[holdout_league],
                test_include_leagues=[holdout_league],
            )
            if result is None:
                continue
            val_pos, test_pos, _ = result
            val_parts.append(val_pos)
            test_parts.append(test_pos)

        if not test_parts:
            print(f"[holdout] skipped {holdout_league}: no predictions generated.")
            holdout_results.append(
                {
                    "league": holdout_league,
                    "status": "skipped",
                    "reason": "no_predictions",
                }
            )
            continue

        val_frame = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
        test_frame = pd.concat(test_parts, ignore_index=True)
        priors = fit_error_priors(val_frame, interval_q=interval_q) if not val_frame.empty else pd.DataFrame()
        if not val_frame.empty:
            val_frame = apply_confidence_scoring(val_frame, priors, interval_q=interval_q)
        test_frame = apply_confidence_scoring(test_frame, priors, interval_q=interval_q)

        stem = output_path.stem
        suffix = output_path.suffix or ".csv"
        league_slug = _slugify(holdout_league)
        holdout_csv = output_path.with_name(f"{stem}.holdout_{league_slug}{suffix}")
        holdout_metrics_path = output_path.with_name(f"{stem}.holdout_{league_slug}.metrics.json")

        test_frame.sort_values("value_diff").to_csv(holdout_csv, index=False)
        holdout_overall = regression_metrics(
            test_frame["market_value_eur"].to_numpy(),
            test_frame["expected_value_eur"].to_numpy(),
            mape_min_denom=mape_min_denom_eur,
        )
        holdout_interval = _interval_summary(test_frame)
        holdout_segments = evaluate_value_segments(
            test_frame,
            "holdout",
            mape_min_denom_eur=mape_min_denom_eur,
        )
        ref_test = df_raw.copy()
        if "season" in ref_test.columns:
            ref_test = ref_test[ref_test["season"].astype(str) == str(test_season)].copy()
        ref_test = _filter_by_league(ref_test, exclude=[holdout_league])
        holdout_shift = _compute_league_shift_report(ref_test, test_frame)
        holdout_payload = {
            "league": holdout_league,
            "status": "ok",
            "n_samples": int(len(test_frame)),
            "overall": {
                **holdout_overall,
                **holdout_interval,
            },
            "segments": holdout_segments,
            "domain_shift": holdout_shift,
            "predictions_csv": str(holdout_csv),
        }
        _write_json(holdout_metrics_path, holdout_payload)
        holdout_payload["metrics_json"] = str(holdout_metrics_path)
        holdout_results.append(holdout_payload)
        print(
            f"[holdout:{holdout_league}] n={len(test_frame)} | "
            f"R² {holdout_overall['r2']*100:,.2f}% | "
            f"MAE €{holdout_overall['mae_eur']:,.0f} | "
            f"MAPE {holdout_overall['mape']*100:,.2f}%"
        )

    return holdout_results


def main(
    dataset_path: str,
    val_season: str,
    test_season: str,
    output_path: str,
    val_output_path: str | None = None,
    metrics_output_path: str | None = None,
    n_optuna_trials: int = 60,
    recency_half_life: float = 2.0,
    under_5m_weight: float = 1.0,
    mid_5m_to_20m_weight: float = 1.0,
    over_20m_weight: float = 1.0,
    exclude_prefixes: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
    optimize_metric: str = "lowmid_wmape",
    strict_leakage_guard: bool = True,
    interval_q: float = 0.8,
    two_stage_band_model: bool = True,
    band_min_samples: int = 160,
    band_blend_alpha: float = 0.35,
    quality_output_path: str | None = None,
    strict_quality_gate: bool = False,
    league_holdouts: Sequence[str] | None = None,
    drop_incomplete_league_seasons: bool = True,
    min_league_season_rows: int = 40,
    min_league_season_completeness: float = 0.55,
    residual_calibration_min_samples: int = 30,
    mape_min_denom_eur: float = 1_000_000.0,
) -> None:
    if val_season == test_season:
        raise ValueError("--val-season must be different from --test-season.")
    if min(under_5m_weight, mid_5m_to_20m_weight, over_20m_weight) <= 0:
        raise ValueError("All value-segment weights must be > 0.")
    if not (0.5 <= float(interval_q) <= 0.99):
        raise ValueError("--interval-q must be in [0.5, 0.99].")
    if int(band_min_samples) < 1:
        raise ValueError("--band-min-samples must be >= 1.")
    if not (0.0 <= float(band_blend_alpha) <= 1.0):
        raise ValueError("--band-blend-alpha must be in [0.0, 1.0].")
    if int(min_league_season_rows) < 0:
        raise ValueError("--min-league-season-rows must be >= 0.")
    if not (0.0 <= float(min_league_season_completeness) <= 1.0):
        raise ValueError("--min-league-season-completeness must be in [0.0, 1.0].")
    if int(residual_calibration_min_samples) < 1:
        raise ValueError("--residual-calibration-min-samples must be >= 1.")
    if float(mape_min_denom_eur) < 0:
        raise ValueError("--mape-min-denom-eur must be >= 0.")

    print(
        "[train] value-segment weights: "
        f"under_5m={under_5m_weight:.3f}, "
        f"5m_to_20m={mid_5m_to_20m_weight:.3f}, "
        f"over_20m={over_20m_weight:.3f}"
    )
    print(f"[train] optimize metric: {optimize_metric}")
    print(f"[train] MAPE denominator floor: €{float(mape_min_denom_eur):,.0f}")
    print(f"[train] conformal interval q: {interval_q:.2f}")
    print(
        "[train] two-stage band model: "
        f"{two_stage_band_model} "
        f"(min_samples={int(band_min_samples)}, alpha={float(band_blend_alpha):.2f})"
    )

    df_raw = load_dataset(dataset_path)
    df_raw = _drop_duplicate_columns(df_raw)
    df_raw = df_raw[df_raw["log_market_value"].notna()].copy()
    df_raw = _deduplicate_player_season_rows(df_raw)

    if "season" in df_raw.columns:
        df_raw["season"] = df_raw["season"].astype(str)
    if "position_group" in df_raw.columns:
        df_raw["position_group"] = df_raw["position_group"].astype(str).str.upper()

    completeness_filter_summary = {
        "enabled": False,
        "n_rows_before": int(len(df_raw)),
        "n_rows_after": int(len(df_raw)),
        "n_rows_dropped": 0,
        "dropped_groups": [],
    }
    if drop_incomplete_league_seasons:
        df_raw, completeness_filter_summary = _apply_league_season_completeness_filter(
            df_raw,
            min_rows=int(min_league_season_rows),
            min_completeness_ratio=float(min_league_season_completeness),
        )
        dropped = int(completeness_filter_summary.get("n_rows_dropped", 0))
        if dropped > 0:
            print(
                "[clean] dropped incomplete league-season groups: "
                f"{dropped:,} rows removed"
            )
            for row in completeness_filter_summary.get("dropped_groups", [])[:12]:
                print(
                    "  - "
                    f"{row.get('league')} | {row.get('season')} | "
                    f"rows={int(row.get('n_rows', 0))} | "
                    f"completeness={float(row.get('completeness_ratio', 0.0)):.2f}"
                )
            if len(completeness_filter_summary.get("dropped_groups", [])) > 12:
                print("  - ...")

    quality_report = _build_data_quality_report(df_raw)
    quality_report["completeness_filter"] = completeness_filter_summary
    _print_quality_report(quality_report)
    if strict_quality_gate and quality_report.get("quality_flags"):
        raise ValueError(
            "Quality gate failed. Resolve quality flags or rerun without --strict-quality-gate."
        )

    shap_dir = Path("logs/shap")
    shap_dir.mkdir(parents=True, exist_ok=True)

    all_val_preds: List[pd.DataFrame] = []
    all_test_preds: List[pd.DataFrame] = []
    all_metrics: List[PositionMetrics] = []

    for pos in POSITIONS:
        print("\n======================")
        print(f"  TRAINING {pos}")
        print("======================")
        result = train_for_position(
            df_raw,
            pos,
            val_season,
            test_season,
            shap_dir,
            n_optuna_trials=n_optuna_trials,
            recency_half_life=recency_half_life,
            under_5m_weight=under_5m_weight,
            mid_5m_to_20m_weight=mid_5m_to_20m_weight,
            over_20m_weight=over_20m_weight,
            exclude_prefixes=exclude_prefixes,
            exclude_columns=exclude_columns,
            optimize_metric=optimize_metric,
            strict_leakage_guard=strict_leakage_guard,
            two_stage_band_model=two_stage_band_model,
            band_min_samples=band_min_samples,
            band_blend_alpha=band_blend_alpha,
            mape_min_denom_eur=mape_min_denom_eur,
        )
        if result is None:
            continue
        val_preds, test_preds, pos_metrics = result
        all_val_preds.append(val_preds)
        all_test_preds.append(test_preds)
        all_metrics.extend(pos_metrics)

    if not all_test_preds:
        print("[error] no position models produced predictions.")
        return

    val_final = pd.concat(all_val_preds, ignore_index=True)
    test_final = pd.concat(all_test_preds, ignore_index=True)

    calibration_table = _fit_residual_calibration_table(
        val_final,
        min_samples=int(residual_calibration_min_samples),
    )
    val_final = _apply_residual_calibration_table(val_final, calibration_table)
    test_final = _apply_residual_calibration_table(test_final, calibration_table)

    error_priors = fit_error_priors(val_final, interval_q=interval_q)
    val_final = apply_confidence_scoring(val_final, error_priors, interval_q=interval_q)
    test_final = apply_confidence_scoring(test_final, error_priors, interval_q=interval_q)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    test_final.sort_values("value_diff").to_csv(output, index=False)

    if val_output_path is None:
        val_output_path = str(output.with_name(f"{output.stem}_val{output.suffix or '.csv'}"))
    val_output = Path(val_output_path)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    val_final.sort_values("value_diff").to_csv(val_output, index=False)

    error_prior_path = output.with_name(f"{output.stem}.error_priors.csv")
    error_priors.to_csv(error_prior_path, index=False)

    metrics_payload: Dict[str, object] = {
        "dataset": dataset_path,
        "val_season": val_season,
        "test_season": test_season,
        "trials_per_position": n_optuna_trials,
        "recency_half_life": recency_half_life,
        "under_5m_weight": under_5m_weight,
        "mid_5m_to_20m_weight": mid_5m_to_20m_weight,
        "over_20m_weight": over_20m_weight,
        "optimize_metric": optimize_metric,
        "interval_q": float(interval_q),
        "two_stage_band_model": bool(two_stage_band_model),
        "band_min_samples": int(band_min_samples),
        "band_blend_alpha": float(band_blend_alpha),
        "exclude_prefixes": list(exclude_prefixes or []),
        "exclude_columns": list(exclude_columns or []),
        "strict_leakage_guard": bool(strict_leakage_guard),
        "drop_incomplete_league_seasons": bool(drop_incomplete_league_seasons),
        "min_league_season_rows": int(min_league_season_rows),
        "min_league_season_completeness": float(min_league_season_completeness),
        "residual_calibration_min_samples": int(residual_calibration_min_samples),
        "mape_min_denom_eur": float(mape_min_denom_eur),
        "positions": [],
        "overall": {},
        "segments": {},
        "quality_report": quality_report,
        "residual_calibration": {
            "enabled": bool(calibration_table.get("enabled", False)),
            "min_samples": int(calibration_table.get("min_samples", int(residual_calibration_min_samples))),
            "global_adjustment_eur": float(calibration_table.get("global_adjustment_eur", 0.0) or 0.0),
            "n_level1": int(len(calibration_table.get("level1", {}) or {})),
            "n_level2": int(len(calibration_table.get("level2", {}) or {})),
            "n_level3": int(len(calibration_table.get("level3", {}) or {})),
            "n_level4": int(len(calibration_table.get("level4", {}) or {})),
        },
        "artifacts": {
            "val_predictions_csv": str(val_output),
            "test_predictions_csv": str(output),
            "error_priors_csv": str(error_prior_path),
        },
    }

    print("\n========== POSITION SUMMARY ==========")
    order = {pos: i for i, pos in enumerate(POSITIONS)}
    split_order = {"val": 0, "test": 1}
    for metric in sorted(
        all_metrics,
        key=lambda item: (split_order.get(item.split, 99), order.get(item.position, 99)),
    ):
        print(
            f"{metric.split:>4} | {metric.position:>2} | n={metric.n_samples:>4} | "
            f"R² {metric.r2*100:6.2f}% | "
            f"MAE €{metric.mae:,.0f} | "
            f"MAPE {metric.mape*100:6.2f}% | "
            f"WMAPE {metric.wmape*100:6.2f}%"
        )
        metrics_payload["positions"].append(
            {
                "split": metric.split,
                "position": metric.position,
                "n_samples": metric.n_samples,
                "r2": metric.r2,
                "mae_eur": metric.mae,
                "mape": metric.mape,
                "wmape": metric.wmape,
            }
        )

    for split_name, frame in [("val", val_final), ("test", test_final)]:
        split_metrics = regression_metrics(
            frame["market_value_eur"].to_numpy(),
            frame["expected_value_eur"].to_numpy(),
            mape_min_denom=mape_min_denom_eur,
        )
        interval_stats = _interval_summary(frame)
        print("--------------------------------------")
        print(
            f"{split_name:>4} ALL | "
            f"R² {split_metrics['r2']*100:6.2f}% | "
            f"MAE €{split_metrics['mae_eur']:,.0f} | "
            f"MAPE {split_metrics['mape']*100:6.2f}% | "
            f"WMAPE {split_metrics['wmape']*100:6.2f}%"
        )
        if np.isfinite(interval_stats["interval_coverage"]):
            print(
                f"       interval q={interval_q:.2f} | "
                f"coverage {interval_stats['interval_coverage']*100:6.2f}% | "
                f"avg width €{interval_stats['interval_avg_width_eur']:,.0f}"
            )
        metrics_payload["overall"][split_name] = {
            "n_samples": int(len(frame)),
            **split_metrics,
            **interval_stats,
        }

        segment_rows = evaluate_value_segments(
            frame,
            split_name,
            mape_min_denom_eur=mape_min_denom_eur,
        )
        metrics_payload["segments"][split_name] = segment_rows
        print(f"{split_name:>4} VALUE SEGMENTS:")
        for row in segment_rows:
            n = int(row["n_samples"])
            if n == 0:
                print(f"  - {row['segment']}: n=0")
                continue
            print(
                f"  - {row['segment']}: n={n} | "
                f"R² {row['r2']*100:6.2f}% | "
                f"MAE €{row['mae_eur']:,.0f} | "
                f"MAPE {row['mape']*100:6.2f}% | "
                f"WMAPE {row['wmape']*100:6.2f}%"
            )

    holdout_results = _run_league_holdout_suite(
        df_raw=df_raw,
        holdout_leagues=list(league_holdouts or []),
        val_season=val_season,
        test_season=test_season,
        output_path=output,
        shap_dir=shap_dir,
        n_optuna_trials=n_optuna_trials,
        recency_half_life=recency_half_life,
        under_5m_weight=under_5m_weight,
        mid_5m_to_20m_weight=mid_5m_to_20m_weight,
        over_20m_weight=over_20m_weight,
        exclude_prefixes=exclude_prefixes,
        exclude_columns=exclude_columns,
        optimize_metric=optimize_metric,
        strict_leakage_guard=strict_leakage_guard,
        interval_q=interval_q,
        two_stage_band_model=two_stage_band_model,
        band_min_samples=band_min_samples,
        band_blend_alpha=band_blend_alpha,
        mape_min_denom_eur=mape_min_denom_eur,
    )
    if holdout_results:
        metrics_payload["league_holdout"] = holdout_results
        metrics_payload["artifacts"]["holdout_count"] = len(holdout_results)

    if metrics_output_path is None:
        metrics_output_path = str(output.with_suffix(".metrics.json"))
    metrics_path = Path(metrics_output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if quality_output_path is None:
        quality_output_path = str(output.with_name(f"{output.stem}.quality.json"))
    quality_path = Path(quality_output_path)
    _write_json(quality_path, quality_report)

    print(f"\n[done] wrote validation predictions → {val_output}")
    print(f"[done] wrote test predictions → {output}")
    print(f"[done] wrote error priors → {error_prior_path}")
    print(f"[done] wrote metrics → {metrics_path}")
    print(f"[done] wrote quality report → {quality_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Position-aware market value pipeline with leakage-safe splits and Optuna tuning."
    )
    parser.add_argument("--dataset", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--val-season", dest="val_season", default="2023/24")
    parser.add_argument("--test-season", dest="test_season", default="2024/25")
    parser.add_argument("--output", default="data/model/big5_predictions_full_v2.csv")
    parser.add_argument(
        "--val-output",
        dest="val_output",
        default=None,
        help="Optional path for validation-season predictions CSV.",
    )
    parser.add_argument(
        "--metrics-output",
        dest="metrics_output",
        default=None,
        help="Optional path for JSON metrics artifact.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=60,
        help="Optuna trials per position.",
    )
    parser.add_argument(
        "--recency-half-life",
        type=float,
        default=2.0,
        help="Season half-life for sample weighting. <=0 disables recency weighting.",
    )
    parser.add_argument(
        "--under-5m-weight",
        type=float,
        default=1.0,
        help="Training weight multiplier for rows with market value < €5m.",
    )
    parser.add_argument(
        "--mid-5m-20m-weight",
        type=float,
        default=1.0,
        help="Training weight multiplier for rows with market value in [€5m, €20m).",
    )
    parser.add_argument(
        "--over-20m-weight",
        type=float,
        default=1.0,
        help="Training weight multiplier for rows with market value >= €20m.",
    )
    parser.add_argument(
        "--exclude-prefixes",
        default="",
        help="Comma-separated feature prefixes to exclude (e.g. contract_,injury_).",
    )
    parser.add_argument(
        "--exclude-columns",
        default="",
        help="Comma-separated exact feature names to exclude.",
    )
    parser.add_argument(
        "--optimize-metric",
        default="lowmid_wmape",
        choices=["mae", "rmse", "band_wmape", "lowmid_wmape"],
        help="Optuna objective metric.",
    )
    parser.add_argument(
        "--interval-q",
        type=float,
        default=0.80,
        help="Conformal interval quantile (absolute residual quantile in validation split).",
    )
    parser.add_argument(
        "--two-stage-band-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route predictions through value-band experts (2-stage band-first model).",
    )
    parser.add_argument(
        "--band-min-samples",
        type=int,
        default=160,
        help="Minimum training rows required to fit a value-band expert model.",
    )
    parser.add_argument(
        "--band-blend-alpha",
        type=float,
        default=0.35,
        help="Blend factor for band-expert predictions vs base model (0..1).",
    )
    parser.add_argument(
        "--strict-leakage-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise if leakage-like features are detected.",
    )
    parser.add_argument(
        "--quality-output",
        default=None,
        help="Optional JSON path for data quality report.",
    )
    parser.add_argument(
        "--strict-quality-gate",
        action="store_true",
        help="Fail training when quality flags are present.",
    )
    parser.add_argument(
        "--league-holdouts",
        default="",
        help="Optional comma-separated holdout leagues for unseen-league evaluation.",
    )
    parser.add_argument(
        "--drop-incomplete-league-seasons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop league-season groups below completeness thresholds before model training.",
    )
    parser.add_argument(
        "--min-league-season-rows",
        type=int,
        default=40,
        help="Minimum rows required per league-season when completeness filtering is enabled.",
    )
    parser.add_argument(
        "--min-league-season-completeness",
        type=float,
        default=0.55,
        help="Minimum ratio vs each league's max-season row count to keep a league-season.",
    )
    parser.add_argument(
        "--residual-calibration-min-samples",
        type=int,
        default=30,
        help="Minimum validation samples required to fit each residual-calibration bucket.",
    )
    parser.add_argument(
        "--mape-min-denom-eur",
        type=float,
        default=1_000_000.0,
        help="MAPE denominator floor in EUR to reduce tiny-value distortion.",
    )
    args = parser.parse_args()

    main(
        dataset_path=args.dataset,
        val_season=args.val_season,
        test_season=args.test_season,
        output_path=args.output,
        val_output_path=args.val_output,
        metrics_output_path=args.metrics_output,
        n_optuna_trials=args.trials,
        recency_half_life=args.recency_half_life,
        under_5m_weight=args.under_5m_weight,
        mid_5m_to_20m_weight=args.mid_5m_20m_weight,
        over_20m_weight=args.over_20m_weight,
        exclude_prefixes=_parse_csv_tokens(args.exclude_prefixes),
        exclude_columns=_parse_csv_tokens(args.exclude_columns),
        optimize_metric=args.optimize_metric,
        strict_leakage_guard=args.strict_leakage_guard,
        interval_q=args.interval_q,
        two_stage_band_model=args.two_stage_band_model,
        band_min_samples=args.band_min_samples,
        band_blend_alpha=args.band_blend_alpha,
        quality_output_path=args.quality_output,
        strict_quality_gate=args.strict_quality_gate,
        league_holdouts=_parse_csv_tokens(args.league_holdouts),
        drop_incomplete_league_seasons=args.drop_incomplete_league_seasons,
        min_league_season_rows=args.min_league_season_rows,
        min_league_season_completeness=args.min_league_season_completeness,
        residual_calibration_min_samples=args.residual_calibration_min_samples,
        mape_min_denom_eur=args.mape_min_denom_eur,
    )
