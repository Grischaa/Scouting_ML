"""Service functions for market-value prediction artifacts."""

from __future__ import annotations

import json
import os
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import numpy as np
import pandas as pd

from scouting_ml.features.history_strength import (
    HISTORY_COMPONENT_COLUMNS,
    HISTORY_COMPONENT_LABELS,
    HISTORY_COMPONENT_WEIGHTS,
    add_history_strength_features,
)

Split = Literal["test", "val"]

TEST_PRED_ENV = "SCOUTING_TEST_PREDICTIONS_PATH"
VAL_PRED_ENV = "SCOUTING_VAL_PREDICTIONS_PATH"
METRICS_ENV = "SCOUTING_METRICS_PATH"
MODEL_MANIFEST_ENV = "SCOUTING_MODEL_MANIFEST_PATH"
ENABLE_RESIDUAL_CALIBRATION_ENV = "SCOUTING_ENABLE_RESIDUAL_CALIBRATION"
CALIBRATION_MIN_SAMPLES_ENV = "SCOUTING_CALIBRATION_MIN_SAMPLES"
WATCHLIST_PATH_ENV = "SCOUTING_WATCHLIST_PATH"

DEFAULT_TEST_PRED = Path("data/model/big5_predictions_full_v2.csv")
DEFAULT_VAL_PRED = Path("data/model/big5_predictions_full_v2_val.csv")
DEFAULT_METRICS = Path("data/model/big5_predictions_full_v2.metrics.json")
DEFAULT_MODEL_MANIFEST = Path("data/model/model_manifest.json")
DEFAULT_WATCHLIST_PATH = Path("data/model/scout_watchlist.jsonl")

SPLIT_TO_PATH = {
    "test": (TEST_PRED_ENV, DEFAULT_TEST_PRED),
    "val": (VAL_PRED_ENV, DEFAULT_VAL_PRED),
}

BIG5_LEAGUES = {
    "premier league",
    "la liga",
    "laliga",
    "serie a",
    "bundesliga",
    "ligue 1",
}

PROFILE_METRIC_SPECS: tuple[tuple[str, str, int], ...] = (
    ("sofa_goals_per90", "Goals/90", 1),
    ("sofa_assists_per90", "Assists/90", 1),
    ("sofa_expectedGoals_per90", "xG/90", 1),
    ("sofa_totalShots_per90", "Shots/90", 1),
    ("sofa_keyPasses_per90", "Key passes/90", 1),
    ("sofa_successfulDribbles_per90", "Successful dribbles/90", 1),
    ("sofa_accuratePassesPercentage", "Pass accuracy %", 1),
    ("sofa_totalDuelsWonPercentage", "Duels won %", 1),
    ("sofa_tackles_per90", "Tackles/90", 1),
    ("sofa_interceptions_per90", "Interceptions/90", 1),
    ("sofa_clearances_per90", "Clearances/90", 1),
    ("injury_days_per_1000_min", "Injury days/1000 min", -1),
    ("history_strength_score", "History strength", 1),
)


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when required model artifacts are missing."""


@dataclass
class _FrameCache:
    path: Path
    mtime_ns: int
    frame: pd.DataFrame


@dataclass
class _ResidualCalibrationCache:
    path: Path
    mtime_ns: int
    min_samples: int
    payload: dict[str, Any]


_PRED_CACHE: Dict[str, _FrameCache] = {}
_METRICS_CACHE: tuple[Path, int, dict[str, Any]] | None = None
_RESIDUAL_CALIBRATION_CACHE: _ResidualCalibrationCache | None = None
_REQUIRED_ARTIFACT_ENVS = (TEST_PRED_ENV, VAL_PRED_ENV, METRICS_ENV)


def _resolve_path(env_var: str, default_path: Path) -> Path:
    value = os.getenv(env_var, "").strip()
    return Path(value) if value else default_path


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _watchlist_path() -> Path:
    return _resolve_path(WATCHLIST_PATH_ENV, DEFAULT_WATCHLIST_PATH)


def _file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "size_bytes": None,
            "mtime_utc": None,
            "mtime_epoch": None,
        }
    stat = path.stat()
    dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return {
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_utc": dt,
        "mtime_epoch": float(stat.st_mtime),
    }


def _safe_numeric(frame: pd.DataFrame, col: str) -> None:
    if col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _replace_inf_with_nan(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return out
    numeric = out.loc[:, numeric_cols]
    out.loc[:, numeric_cols] = numeric.where(np.isfinite(numeric), np.nan)
    return out


def get_resolved_artifact_paths() -> dict[str, str]:
    test_path = _resolve_path(*SPLIT_TO_PATH["test"])
    val_path = _resolve_path(*SPLIT_TO_PATH["val"])
    metrics_path = _resolve_path(METRICS_ENV, DEFAULT_METRICS)
    return {
        "test_predictions_path": str(test_path),
        "val_predictions_path": str(val_path),
        "metrics_path": str(metrics_path),
    }


def get_active_artifacts() -> dict[str, Any]:
    paths = get_resolved_artifact_paths()
    test_path = Path(paths["test_predictions_path"])
    val_path = Path(paths["val_predictions_path"])
    metrics_path = Path(paths["metrics_path"])
    return {
        "test_predictions_path": str(test_path),
        "val_predictions_path": str(val_path),
        "metrics_path": str(metrics_path),
        "test_predictions_sha256": _sha256_file(test_path),
        "val_predictions_sha256": _sha256_file(val_path),
        "metrics_sha256": _sha256_file(metrics_path),
    }


def validate_strict_artifact_env() -> None:
    missing_env = [env_name for env_name in _REQUIRED_ARTIFACT_ENVS if not os.getenv(env_name, "").strip()]
    if missing_env:
        raise RuntimeError(
            "Strict artifacts mode is enabled, but required env vars are missing: "
            + ", ".join(missing_env)
        )

    missing_files: list[str] = []
    for env_name in _REQUIRED_ARTIFACT_ENVS:
        raw = os.getenv(env_name, "").strip()
        path = Path(raw)
        if not path.exists():
            missing_files.append(f"{env_name}={path}")
    if missing_files:
        raise RuntimeError(
            "Strict artifacts mode is enabled, but artifact files do not exist: "
            + ", ".join(missing_files)
        )


def _clean_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    # Some exports may accidentally contain a duplicated header row.
    if "player_id" in out.columns:
        out = out[out["player_id"].astype(str).str.lower() != "player_id"].copy()

    numeric_cols = [
        "market_value_eur",
        "expected_value_eur",
        "fair_value_eur",
        "expected_value_low_eur",
        "expected_value_high_eur",
        "value_diff",
        "value_abs_error",
        "value_gap_eur",
        "value_gap_conservative_eur",
        "undervaluation_confidence",
        "undervaluation_score",
        "prior_mae_eur",
        "prior_medae_eur",
        "prior_p75ae_eur",
        "prior_qae_eur",
        "prior_interval_q",
        "age",
        "minutes",
        "sofa_minutesPlayed",
        "season_end_year",
    ]
    for col in numeric_cols:
        _safe_numeric(out, col)

    if "undervalued_flag" in out.columns:
        out["undervalued_flag"] = pd.to_numeric(out["undervalued_flag"], errors="coerce").fillna(0).astype(int)

    if "position_group" in out.columns:
        out["position_group"] = out["position_group"].astype(str).str.upper()
    if "model_position" in out.columns:
        out["model_position"] = out["model_position"].astype(str).str.upper()

    return out


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ArtifactNotFoundError(f"Predictions artifact not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    return _clean_prediction_frame(frame)


def get_predictions(split: Split = "test") -> pd.DataFrame:
    env_var, default_path = SPLIT_TO_PATH[split]
    path = _resolve_path(env_var, default_path)
    if not path.exists():
        raise ArtifactNotFoundError(f"{split} predictions artifact not found: {path}")

    mtime = path.stat().st_mtime_ns
    cached = _PRED_CACHE.get(split)
    if cached and cached.path == path and cached.mtime_ns == mtime:
        return cached.frame.copy()

    frame = _load_predictions(path)
    _PRED_CACHE[split] = _FrameCache(path=path, mtime_ns=mtime, frame=frame)
    return frame.copy()


def get_metrics() -> dict[str, Any]:
    global _METRICS_CACHE
    path = _resolve_path(METRICS_ENV, DEFAULT_METRICS)
    if not path.exists():
        raise ArtifactNotFoundError(f"Metrics artifact not found: {path}")

    mtime = path.stat().st_mtime_ns
    if _METRICS_CACHE and _METRICS_CACHE[0] == path and _METRICS_CACHE[1] == mtime:
        return dict(_METRICS_CACHE[2])

    payload = json.loads(path.read_text(encoding="utf-8"))
    _METRICS_CACHE = (path, mtime, payload)
    return dict(payload)


def _to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = _replace_inf_with_nan(frame)
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


def _minutes_series(frame: pd.DataFrame) -> pd.Series:
    if "minutes" in frame.columns:
        return pd.to_numeric(frame["minutes"], errors="coerce")
    if "sofa_minutesPlayed" in frame.columns:
        return pd.to_numeric(frame["sofa_minutesPlayed"], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _position_series(frame: pd.DataFrame) -> pd.Series:
    if "model_position" in frame.columns:
        return frame["model_position"].astype(str).str.upper()
    if "position_group" in frame.columns:
        return frame["position_group"].astype(str).str.upper()
    return pd.Series("UNK", index=frame.index)


def _prediction_value_column(frame: pd.DataFrame) -> str:
    if "fair_value_eur" in frame.columns:
        return "fair_value_eur"
    if "expected_value_eur" in frame.columns:
        return "expected_value_eur"
    return "fair_value_eur"


def _ensure_value_segment(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "value_segment" in out.columns:
        out["value_segment"] = out["value_segment"].astype(str)
        return out
    market = pd.to_numeric(out.get("market_value_eur"), errors="coerce")
    segment = pd.Series("unknown", index=out.index, dtype=object)
    segment.loc[(market >= 0.0) & (market < 5_000_000.0)] = "under_5m"
    segment.loc[(market >= 5_000_000.0) & (market < 20_000_000.0)] = "5m_to_20m"
    segment.loc[market >= 20_000_000.0] = "over_20m"
    out["value_segment"] = segment.astype(str)
    return out


def _to_numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _build_capped_gap_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cons = _to_numeric_series(out, "value_gap_conservative_eur")
    raw = _to_numeric_series(out, "value_gap_eur")
    cons = cons.fillna(raw)

    prior_q = _to_numeric_series(out, "prior_qae_eur")
    prior_p75 = _to_numeric_series(out, "prior_p75ae_eur")
    prior_mae = _to_numeric_series(out, "prior_mae_eur")

    cap_df = pd.DataFrame(
        {
            "q": np.where(prior_q > 0, 2.5 * prior_q, np.nan),
            "p75": np.where(prior_p75 > 0, 3.0 * prior_p75, np.nan),
            "mae": np.where(prior_mae > 0, 4.0 * prior_mae, np.nan),
        },
        index=out.index,
    )
    cap_threshold = cap_df.min(axis=1, skipna=True)
    capped = cons.copy()
    mask = cons.notna() & (cons > 0) & cap_threshold.notna()
    capped.loc[mask] = np.minimum(cons.loc[mask], cap_threshold.loc[mask])
    cap_applied = mask & (capped + 1.0 < cons)
    cap_ratio = cons / np.maximum(cap_threshold, 1.0)

    out["value_gap_cap_threshold_eur"] = cap_threshold
    out["value_gap_capped_eur"] = capped
    out["value_gap_cap_applied"] = cap_applied.astype(int)
    out["value_gap_cap_ratio"] = cap_ratio.where(np.isfinite(cap_ratio), np.nan)
    return out


def _fit_residual_calibrator(val_frame: pd.DataFrame, min_samples: int) -> dict[str, Any]:
    if val_frame.empty:
        return {"min_samples": int(min_samples), "global_adjustment_eur": 0.0}

    pred_col = _prediction_value_column(val_frame)
    if pred_col not in val_frame.columns or "market_value_eur" not in val_frame.columns:
        return {"min_samples": int(min_samples), "global_adjustment_eur": 0.0}

    work = _ensure_value_segment(val_frame)
    work["league_norm"] = work.get("league", pd.Series("", index=work.index)).astype(str).str.strip().str.casefold()
    work["position_norm"] = _position_series(work).astype(str).str.upper()
    work["pred_eur"] = pd.to_numeric(work[pred_col], errors="coerce")
    work["market_eur"] = pd.to_numeric(work["market_value_eur"], errors="coerce")
    work["residual_eur"] = work["market_eur"] - work["pred_eur"]
    work = work[np.isfinite(work["residual_eur"])].copy()

    if work.empty:
        return {"min_samples": int(min_samples), "global_adjustment_eur": 0.0}

    def _group_map(cols: list[str], threshold: int) -> dict[tuple[str, ...], float]:
        grouped = (
            work.groupby(cols, dropna=False)["residual_eur"]
            .agg(median_residual="median", n="count")
            .reset_index()
        )
        grouped = grouped[grouped["n"] >= int(max(threshold, 1))].copy()
        out: dict[tuple[str, ...], float] = {}
        for _, row in grouped.iterrows():
            key = tuple(str(row[c]) for c in cols)
            # One-sided conservative correction: only reduce optimistic predictions.
            out[key] = float(min(float(row["median_residual"]), 0.0))
        return out

    lvl1 = _group_map(["league_norm", "position_norm", "value_segment"], threshold=min_samples)
    lvl2 = _group_map(["league_norm", "position_norm"], threshold=max(min_samples // 2, 12))
    lvl3 = _group_map(["position_norm", "value_segment"], threshold=max(min_samples // 2, 12))
    lvl4 = _group_map(["value_segment"], threshold=max(min_samples // 3, 8))
    global_adjustment = 0.0
    if len(work) >= int(max(min_samples, 1)):
        global_adjustment = float(min(float(work["residual_eur"].median()), 0.0))

    return {
        "min_samples": int(min_samples),
        "level1": lvl1,
        "level2": lvl2,
        "level3": lvl3,
        "level4": lvl4,
        "global_adjustment_eur": global_adjustment,
    }


def _get_residual_calibrator() -> dict[str, Any]:
    global _RESIDUAL_CALIBRATION_CACHE
    if not _env_flag(ENABLE_RESIDUAL_CALIBRATION_ENV, default=True):
        return {"enabled": False, "global_adjustment_eur": 0.0}

    min_samples_raw = os.getenv(CALIBRATION_MIN_SAMPLES_ENV, "30").strip()
    try:
        min_samples = max(int(min_samples_raw), 1)
    except ValueError:
        min_samples = 30

    val_path = _resolve_path(*SPLIT_TO_PATH["val"])
    if not val_path.exists():
        return {"enabled": False, "global_adjustment_eur": 0.0}

    mtime = val_path.stat().st_mtime_ns
    if (
        _RESIDUAL_CALIBRATION_CACHE is not None
        and _RESIDUAL_CALIBRATION_CACHE.path == val_path
        and _RESIDUAL_CALIBRATION_CACHE.mtime_ns == mtime
        and _RESIDUAL_CALIBRATION_CACHE.min_samples == min_samples
    ):
        return dict(_RESIDUAL_CALIBRATION_CACHE.payload)

    val_frame = get_predictions(split="val")
    payload = _fit_residual_calibrator(val_frame=val_frame, min_samples=min_samples)
    payload["enabled"] = True
    _RESIDUAL_CALIBRATION_CACHE = _ResidualCalibrationCache(
        path=val_path,
        mtime_ns=mtime,
        min_samples=min_samples,
        payload=dict(payload),
    )
    return payload


def _lookup_residual_adjustment(row: pd.Series, calibrator: dict[str, Any]) -> float:
    if not calibrator.get("enabled", False):
        return 0.0
    league = str(row.get("league_norm", "")).strip().casefold()
    pos = str(row.get("position_norm", "")).strip().upper()
    seg = str(row.get("value_segment", "unknown"))

    lvl1 = calibrator.get("level1", {})
    lvl2 = calibrator.get("level2", {})
    lvl3 = calibrator.get("level3", {})
    lvl4 = calibrator.get("level4", {})

    if (league, pos, seg) in lvl1:
        return float(lvl1[(league, pos, seg)])
    if (league, pos) in lvl2:
        return float(lvl2[(league, pos)])
    if (pos, seg) in lvl3:
        return float(lvl3[(pos, seg)])
    if (seg,) in lvl4:
        return float(lvl4[(seg,)])
    return float(calibrator.get("global_adjustment_eur", 0.0) or 0.0)


def _apply_residual_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_value_segment(frame)
    if out.empty:
        return out

    # If artifacts already include a training-time residual calibration, avoid double-calibration.
    if "residual_calibration_applied" in out.columns:
        applied = pd.to_numeric(out["residual_calibration_applied"], errors="coerce").fillna(0)
        if (applied > 0).any() and "expected_value_calibration_eur" in out.columns:
            return _build_capped_gap_columns(out)

    pred_col = _prediction_value_column(out)
    if pred_col not in out.columns:
        out = _build_capped_gap_columns(out)
        return out

    out["league_norm"] = out.get("league", pd.Series("", index=out.index)).astype(str).str.strip().str.casefold()
    out["position_norm"] = _position_series(out).astype(str).str.upper()

    calibrator = _get_residual_calibrator()
    if calibrator.get("enabled", False):
        out["expected_value_calibration_eur"] = out.apply(
            lambda row: _lookup_residual_adjustment(row, calibrator), axis=1
        )
    else:
        out["expected_value_calibration_eur"] = 0.0

    pred_raw = _to_numeric_series(out, pred_col)
    out["expected_value_raw_eur"] = pred_raw
    out["expected_value_eur"] = pred_raw + pd.to_numeric(out["expected_value_calibration_eur"], errors="coerce").fillna(0.0)
    out["fair_value_eur"] = out["expected_value_eur"]

    if "expected_value_low_eur" in out.columns:
        low_raw = _to_numeric_series(out, "expected_value_low_eur")
        out["expected_value_low_raw_eur"] = low_raw
        out["expected_value_low_eur"] = (low_raw + out["expected_value_calibration_eur"]).clip(lower=0.0)
    if "expected_value_high_eur" in out.columns:
        high_raw = _to_numeric_series(out, "expected_value_high_eur")
        out["expected_value_high_raw_eur"] = high_raw
        out["expected_value_high_eur"] = high_raw + out["expected_value_calibration_eur"]

    market = _to_numeric_series(out, "market_value_eur")
    out["value_gap_raw_eur"] = out["expected_value_eur"] - market
    out["value_gap_eur"] = out["value_gap_raw_eur"]
    if "expected_value_low_eur" in out.columns:
        out["value_gap_conservative_eur"] = _to_numeric_series(out, "expected_value_low_eur") - market
    else:
        out["value_gap_conservative_eur"] = out["value_gap_raw_eur"]

    out = _build_capped_gap_columns(out)
    return out


def _prepare_predictions_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = _apply_residual_calibration(frame)
    out = add_history_strength_features(out)
    out = _replace_inf_with_nan(out)
    return out


def _ranking_gap_series(frame: pd.DataFrame) -> pd.Series:
    if "value_gap_capped_eur" in frame.columns:
        return pd.to_numeric(frame["value_gap_capped_eur"], errors="coerce")
    if "value_gap_conservative_eur" in frame.columns:
        return pd.to_numeric(frame["value_gap_conservative_eur"], errors="coerce")
    return pd.to_numeric(frame.get("value_gap_eur"), errors="coerce")


def _history_factor_series(frame: pd.DataFrame) -> pd.Series:
    if "history_strength_score" not in frame.columns:
        return pd.Series(1.0, index=frame.index, dtype=float)

    strength = pd.to_numeric(frame["history_strength_score"], errors="coerce") / 100.0
    strength = strength.clip(lower=0.0, upper=1.0)

    if "history_strength_coverage" in frame.columns:
        coverage = pd.to_numeric(frame["history_strength_coverage"], errors="coerce").clip(lower=0.0, upper=1.0)
    else:
        coverage = pd.Series(1.0, index=frame.index, dtype=float)

    factor = (0.85 + 0.35 * strength) * (0.90 + 0.10 * coverage)
    sparse = coverage < 0.35
    factor = factor.where(~sparse, 1.0)
    return factor.fillna(1.0)


def _infer_future_outcome_label(frame: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    candidates = [
        "future_outcome_label",
        "is_future_undervalued_success",
        "future_success",
        "value_growth_next_season_eur",
        "future_value_growth_eur",
    ]
    for col in candidates:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if col.endswith("_eur"):
            label = (series > 0).astype(float)
        else:
            label = (series > 0).astype(float)
        if label.notna().sum() >= 20:
            return label, col
    return None, None


def _precision_at_k(
    frame: pd.DataFrame,
    *,
    score_col: str,
    k_values: Sequence[int] = (10, 25, 50, 100),
) -> dict[str, Any]:
    if score_col not in frame.columns:
        return {"available": False, "reason": f"missing_score_col:{score_col}"}

    labels, label_col = _infer_future_outcome_label(frame)
    if labels is None or label_col is None:
        return {"available": False, "reason": "missing_future_outcome_label"}

    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_label"] = pd.to_numeric(labels, errors="coerce")
    work = work[np.isfinite(work["_score"]) & np.isfinite(work["_label"])].copy()
    if work.empty:
        return {"available": False, "reason": "no_rows_with_labels"}

    work = work.sort_values("_score", ascending=False).reset_index(drop=True)
    out_rows: list[dict[str, Any]] = []
    for k in sorted({int(max(k, 1)) for k in k_values}):
        top = work.head(k)
        n = int(len(top))
        if n == 0:
            continue
        precision = float((top["_label"] > 0).mean())
        out_rows.append({"k": int(k), "n": n, "precision": precision})

    return {
        "available": bool(out_rows),
        "label_column": label_col,
        "n_labeled_rows": int(len(work)),
        "rows": out_rows,
    }


def query_predictions(
    split: Split = "test",
    season: str | None = None,
    league: str | None = None,
    club: str | None = None,
    position: str | None = None,
    min_minutes: float | None = None,
    max_age: float | None = None,
    undervalued_only: bool = False,
    min_confidence: float | None = None,
    min_value_gap_eur: float | None = None,
    sort_by: str = "value_gap_capped_eur",
    sort_order: Literal["asc", "desc"] = "desc",
    limit: int = 100,
    offset: int = 0,
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))

    if season and "season" in frame.columns:
        frame = frame[frame["season"].astype(str) == str(season)].copy()
    if league and "league" in frame.columns:
        frame = frame[frame["league"].astype(str).str.casefold() == str(league).casefold()].copy()
    if club and "club" in frame.columns:
        frame = frame[frame["club"].astype(str).str.casefold() == str(club).casefold()].copy()
    if position:
        pos_series = _position_series(frame)
        frame = frame[pos_series == str(position).upper()].copy()

    if min_minutes is not None:
        frame = frame[_minutes_series(frame).fillna(0) >= float(min_minutes)].copy()
    if max_age is not None and "age" in frame.columns:
        age = pd.to_numeric(frame["age"], errors="coerce")
        frame = frame[age <= float(max_age)].copy()

    if undervalued_only:
        if "undervalued_flag" in frame.columns:
            frame = frame[pd.to_numeric(frame["undervalued_flag"], errors="coerce").fillna(0) == 1].copy()
        elif "value_gap_conservative_eur" in frame.columns:
            frame = frame[pd.to_numeric(frame["value_gap_conservative_eur"], errors="coerce") > 0].copy()

    if min_confidence is not None and "undervaluation_confidence" in frame.columns:
        conf = pd.to_numeric(frame["undervaluation_confidence"], errors="coerce")
        frame = frame[conf >= float(min_confidence)].copy()

    if min_value_gap_eur is not None:
        gap = _ranking_gap_series(frame)
        frame = frame[gap >= float(min_value_gap_eur)].copy()

    total = int(len(frame))
    if sort_by not in frame.columns:
        fallback = (
            "value_gap_capped_eur"
            if "value_gap_capped_eur" in frame.columns
            else (
                "value_gap_conservative_eur"
                if "value_gap_conservative_eur" in frame.columns
                else ("value_gap_eur" if "value_gap_eur" in frame.columns else frame.columns[0])
            )
        )
        sort_by = fallback
    ascending = sort_order == "asc"
    frame = frame.sort_values(sort_by, ascending=ascending, na_position="last")

    if columns:
        keep = [c for c in columns if c in frame.columns]
        if keep:
            frame = frame[keep].copy()

    start = max(int(offset), 0)
    end = start + max(int(limit), 0)
    page = frame.iloc[start:end].copy()

    return {
        "split": split,
        "total": total,
        "count": int(len(page)),
        "limit": int(limit),
        "offset": int(offset),
        "sort_by": sort_by,
        "sort_order": sort_order,
        "items": _to_records(page),
    }


def query_shortlist(
    split: Split = "test",
    top_n: int = 100,
    min_minutes: float = 900,
    max_age: float | None = 25,
    positions: Sequence[str] | None = None,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))

    pred_col = "fair_value_eur" if "fair_value_eur" in frame.columns else "expected_value_eur"
    if pred_col not in frame.columns or "market_value_eur" not in frame.columns:
        raise ValueError("Prediction artifact does not include required columns.")

    work = frame.copy()
    work["market_value_eur"] = pd.to_numeric(work["market_value_eur"], errors="coerce")
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")

    if "value_gap_conservative_eur" not in work.columns:
        work["value_gap_conservative_eur"] = work[pred_col] - work["market_value_eur"]
    else:
        work["value_gap_conservative_eur"] = pd.to_numeric(work["value_gap_conservative_eur"], errors="coerce")

    work["ranking_gap_eur"] = _ranking_gap_series(work).fillna(work["value_gap_conservative_eur"])

    if "undervaluation_confidence" not in work.columns:
        denom = max(float(np.nanmedian(np.abs(work[pred_col] - work["market_value_eur"]))), 1.0)
        work["undervaluation_confidence"] = work["value_gap_conservative_eur"] / denom
    else:
        work["undervaluation_confidence"] = pd.to_numeric(work["undervaluation_confidence"], errors="coerce")

    work["minutes_used"] = _minutes_series(work).fillna(0.0)
    work["position_used"] = _position_series(work)
    work["age_num"] = pd.to_numeric(work["age"], errors="coerce") if "age" in work.columns else np.nan

    work = work[work["ranking_gap_eur"] > 0].copy()
    work = work[work["minutes_used"] >= float(min_minutes)].copy()

    if max_age is not None:
        work = work[work["age_num"].fillna(999) <= float(max_age)].copy()

    if positions:
        pos_set = {p.upper() for p in positions}
        work = work[work["position_used"].isin(pos_set)].copy()

    reliability = np.clip(work["minutes_used"] / 1800.0, 0.3, 1.2)
    confidence = work["undervaluation_confidence"].clip(lower=0.0).fillna(0.0)
    age = work["age_num"].fillna(26.0)
    age_factor = np.where(age <= 23, 1.15, np.where(age <= 26, 1.0, 0.85))
    history_factor = _history_factor_series(work)

    work["shortlist_score"] = (
        (work["ranking_gap_eur"] / 1_000_000.0)
        * np.log1p(confidence)
        * reliability
        * age_factor
        * history_factor
    )
    work = work.sort_values(
        ["shortlist_score", "ranking_gap_eur", "value_gap_conservative_eur", "undervaluation_confidence"],
        ascending=False,
    )
    shortlist = work.head(max(int(top_n), 0)).copy()
    precision = _precision_at_k(
        work,
        score_col="shortlist_score",
        k_values=(10, 25, 50, int(top_n)),
    )
    return {
        "split": split,
        "total_candidates": int(len(work)),
        "count": int(len(shortlist)),
        "diagnostics": {
            "ranking_basis": "guardrailed_gap_confidence_history",
            "score_column": "shortlist_score",
            "precision_at_k": precision,
        },
        "items": _to_records(shortlist),
    }


def query_scout_targets(
    split: Split = "test",
    top_n: int = 100,
    min_minutes: float = 900,
    max_age: float | None = 23,
    min_confidence: float = 0.50,
    min_value_gap_eur: float = 1_000_000.0,
    positions: Sequence[str] | None = None,
    non_big5_only: bool = True,
    include_leagues: Sequence[str] | None = None,
    exclude_leagues: Sequence[str] | None = None,
    min_expected_value_eur: float | None = None,
    max_expected_value_eur: float | None = None,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    pred_col = "fair_value_eur" if "fair_value_eur" in frame.columns else "expected_value_eur"
    if pred_col not in frame.columns or "market_value_eur" not in frame.columns:
        raise ValueError("Prediction artifact does not include required valuation columns.")

    work = frame.copy()
    work["market_value_eur"] = pd.to_numeric(work["market_value_eur"], errors="coerce")
    work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")

    if "value_gap_conservative_eur" in work.columns:
        work["value_gap_conservative_eur"] = pd.to_numeric(
            work["value_gap_conservative_eur"], errors="coerce"
        )
    else:
        work["value_gap_conservative_eur"] = work[pred_col] - work["market_value_eur"]

    work["ranking_gap_eur"] = _ranking_gap_series(work).fillna(work["value_gap_conservative_eur"])

    if "undervaluation_confidence" in work.columns:
        work["undervaluation_confidence"] = pd.to_numeric(
            work["undervaluation_confidence"], errors="coerce"
        )
    else:
        denom = max(float(np.nanmedian(np.abs(work[pred_col] - work["market_value_eur"]))), 1.0)
        work["undervaluation_confidence"] = work["value_gap_conservative_eur"] / denom

    work["minutes_used"] = _minutes_series(work).fillna(0.0)
    work["position_used"] = _position_series(work)
    work["age_num"] = pd.to_numeric(work["age"], errors="coerce") if "age" in work.columns else np.nan
    work["league_norm"] = (
        work["league"].astype(str).str.strip().str.casefold() if "league" in work.columns else "unknown"
    )

    work = work[work["ranking_gap_eur"] > 0].copy()
    work = work[work["minutes_used"] >= float(min_minutes)].copy()
    work = work[work["undervaluation_confidence"].fillna(0.0) >= float(min_confidence)].copy()
    work = work[work["ranking_gap_eur"].fillna(0.0) >= float(min_value_gap_eur)].copy()

    if max_age is not None:
        work = work[work["age_num"].fillna(999.0) <= float(max_age)].copy()

    if positions:
        wanted = {p.strip().upper() for p in positions if str(p).strip()}
        work = work[work["position_used"].isin(wanted)].copy()

    if non_big5_only:
        work = work[~work["league_norm"].isin(BIG5_LEAGUES)].copy()

    if include_leagues:
        include_norm = {str(league).strip().casefold() for league in include_leagues if str(league).strip()}
        work = work[work["league_norm"].isin(include_norm)].copy()

    if exclude_leagues:
        exclude_norm = {str(league).strip().casefold() for league in exclude_leagues if str(league).strip()}
        work = work[~work["league_norm"].isin(exclude_norm)].copy()

    if min_expected_value_eur is not None:
        work = work[work[pred_col].fillna(0.0) >= float(min_expected_value_eur)].copy()
    if max_expected_value_eur is not None:
        work = work[work[pred_col].fillna(np.inf) <= float(max_expected_value_eur)].copy()

    confidence = work["undervaluation_confidence"].clip(lower=0.0).fillna(0.0)
    minutes_factor = np.clip(work["minutes_used"] / 1800.0, 0.35, 1.25)
    age = work["age_num"].fillna(26.0)
    age_factor = np.where(age <= 20, 1.25, np.where(age <= 23, 1.12, np.where(age <= 26, 1.0, 0.82)))
    market = work["market_value_eur"].fillna(1_000_000.0).clip(lower=1_000_000.0)
    value_efficiency = (work["ranking_gap_eur"] / market).clip(lower=0.0)
    history_factor = _history_factor_series(work)
    work["scout_target_score"] = (
        (work["ranking_gap_eur"] / 1_000_000.0)
        * (1.0 + np.log1p(confidence))
        * minutes_factor
        * age_factor
        * (1.0 + 0.30 * value_efficiency)
        * history_factor
    )

    work = work.sort_values(
        ["scout_target_score", "ranking_gap_eur", "value_gap_conservative_eur", "undervaluation_confidence"],
        ascending=False,
    )
    out = work.head(max(int(top_n), 0)).copy()
    precision = _precision_at_k(
        work,
        score_col="scout_target_score",
        k_values=(10, 25, 50, int(top_n)),
    )
    return {
        "split": split,
        "total_candidates": int(len(work)),
        "count": int(len(out)),
        "diagnostics": {
            "ranking_basis": "guardrailed_gap_confidence_history_efficiency",
            "score_column": "scout_target_score",
            "precision_at_k": precision,
        },
        "items": _to_records(out),
    }


def get_player_prediction(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    if "player_id" not in frame.columns:
        raise ValueError("Prediction artifact does not include 'player_id'.")

    subset = frame[frame["player_id"].astype(str) == str(player_id)].copy()
    if season is not None and "season" in subset.columns:
        subset = subset[subset["season"].astype(str) == str(season)].copy()

    if subset.empty:
        raise ValueError(f"No prediction found for player_id={player_id!r} in split={split}.")

    if season is None:
        sort_cols = []
        if "season_end_year" in subset.columns:
            sort_cols.append("season_end_year")
        if "minutes" in subset.columns:
            sort_cols.append("minutes")
        elif "sofa_minutesPlayed" in subset.columns:
            sort_cols.append("sofa_minutesPlayed")
        if sort_cols:
            subset = subset.sort_values(sort_cols, ascending=False, na_position="last")

    row = subset.iloc[0].to_frame().T
    records = _to_records(row)
    return records[0]


def _safe_float(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _select_player_row(frame: pd.DataFrame, player_id: str, season: str | None = None) -> pd.Series:
    if "player_id" not in frame.columns:
        raise ValueError("Prediction artifact does not include 'player_id'.")

    subset = frame[frame["player_id"].astype(str) == str(player_id)].copy()
    if season is not None and "season" in subset.columns:
        subset = subset[subset["season"].astype(str) == str(season)].copy()

    if subset.empty:
        raise ValueError(f"No prediction found for player_id={player_id!r} in split.")

    if season is None:
        sort_cols = []
        if "season_end_year" in subset.columns:
            sort_cols.append("season_end_year")
        if "minutes" in subset.columns:
            sort_cols.append("minutes")
        elif "sofa_minutesPlayed" in subset.columns:
            sort_cols.append("sofa_minutesPlayed")
        if sort_cols:
            subset = subset.sort_values(sort_cols, ascending=False, na_position="last")

    return subset.iloc[0].copy()


def _cohort_for_player(frame: pd.DataFrame, row: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort = frame.copy()
    filters: dict[str, Any] = {}

    player_position = None
    if "model_position" in row.index and pd.notna(row["model_position"]):
        player_position = str(row["model_position"]).upper().strip()
    elif "position_group" in row.index and pd.notna(row["position_group"]):
        player_position = str(row["position_group"]).upper().strip()
    if player_position:
        pos = _position_series(cohort)
        filtered = cohort[pos == player_position].copy()
        if len(filtered) >= 40:
            cohort = filtered
            filters["position"] = player_position

    player_age = _safe_float(row.get("age"))
    if player_age is not None and "age" in cohort.columns:
        ages = pd.to_numeric(cohort["age"], errors="coerce")
        for band, min_rows in ((2.0, 140), (3.0, 90), (4.0, 50)):
            filtered = cohort[(ages >= player_age - band) & (ages <= player_age + band)].copy()
            if len(filtered) >= min_rows:
                cohort = filtered
                filters["age_band"] = f"{max(int(player_age - band), 0)}-{int(player_age + band)}"
                break

    player_season = row.get("season")
    if player_season is not None and not pd.isna(player_season) and "season" in cohort.columns:
        filtered = cohort[cohort["season"].astype(str) == str(player_season)].copy()
        if len(filtered) >= 35:
            cohort = filtered
            filters["season"] = str(player_season)

    return cohort, filters


def _metric_value_corr_abs(cohort: pd.DataFrame, col: str) -> float:
    target_col = None
    if "fair_value_eur" in cohort.columns:
        target_col = "fair_value_eur"
    elif "expected_value_eur" in cohort.columns:
        target_col = "expected_value_eur"
    if target_col is None:
        return 0.0

    x = pd.to_numeric(cohort[col], errors="coerce")
    y = pd.to_numeric(cohort[target_col], errors="coerce")
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 25:
        return 0.0
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    if pd.isna(corr):
        return 0.0
    return min(abs(float(corr)), 1.0)


def _build_metric_profile(
    row: pd.Series,
    cohort: pd.DataFrame,
    top_metrics: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    for col, label, direction in PROFILE_METRIC_SPECS:
        if col not in cohort.columns or col not in row.index:
            continue

        player_value = _safe_float(row.get(col))
        if player_value is None:
            continue

        series = pd.to_numeric(cohort[col], errors="coerce").dropna()
        if len(series) < 30:
            continue

        percentile_raw = float((series <= player_value).mean())
        quality_percentile = percentile_raw if direction > 0 else (1.0 - percentile_raw)

        q25, q50, q75 = np.nanquantile(series.to_numpy(), [0.25, 0.5, 0.75])
        iqr = max(float(q75 - q25), 1e-9)
        if direction > 0:
            improvement_gap = max(float(q75 - player_value), 0.0)
        else:
            improvement_gap = max(float(player_value - q25), 0.0)
        improvement_gap_iqr = improvement_gap / iqr
        impact_score = improvement_gap_iqr * _metric_value_corr_abs(cohort, col)

        status = "neutral"
        if quality_percentile >= 0.67:
            status = "strength"
        elif quality_percentile <= 0.33:
            status = "weakness"

        metrics.append(
            {
                "metric": col,
                "label": label,
                "direction": "higher_is_better" if direction > 0 else "lower_is_better",
                "player_value": player_value,
                "cohort_median": float(q50),
                "cohort_p25": float(q25),
                "cohort_p75": float(q75),
                "percentile_raw": percentile_raw,
                "quality_percentile": quality_percentile,
                "status": status,
                "improvement_gap": improvement_gap,
                "improvement_gap_iqr": improvement_gap_iqr,
                "impact_score": float(impact_score),
            }
        )

    if not metrics:
        return {"strengths": [], "weaknesses": [], "development_levers": []}

    strengths = sorted(
        [m for m in metrics if m["status"] == "strength"],
        key=lambda m: m["quality_percentile"],
        reverse=True,
    )[:top_metrics]

    weaknesses = sorted(
        [m for m in metrics if m["status"] == "weakness"],
        key=lambda m: m["quality_percentile"],
    )[:top_metrics]

    development = sorted(
        [m for m in weaknesses if m["improvement_gap"] > 0.0],
        key=lambda m: (m["impact_score"], m["improvement_gap_iqr"]),
        reverse=True,
    )[: min(3, top_metrics)]

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "development_levers": development,
    }


def _build_confidence_summary(row: pd.Series, cohort: pd.DataFrame) -> dict[str, Any]:
    pred = _safe_float(row.get("fair_value_eur"))
    if pred is None:
        pred = _safe_float(row.get("expected_value_eur"))

    low = _safe_float(row.get("expected_value_low_eur"))
    high = _safe_float(row.get("expected_value_high_eur"))
    width = None
    width_ratio = None
    if low is not None and high is not None:
        width = max(high - low, 0.0)
        if pred is not None and pred > 0:
            width_ratio = width / pred

    confidence_signal = _safe_float(row.get("undervaluation_confidence"))
    prior_mae = _safe_float(row.get("prior_mae_eur"))
    prior_ratio = None
    if prior_mae is not None and pred is not None and pred > 0:
        prior_ratio = prior_mae / pred

    score = 0.5
    if confidence_signal is not None:
        score += min(max(confidence_signal, 0.0), 2.0) / 4.0
    if width_ratio is not None:
        score -= min(max(width_ratio, 0.0), 2.0) / 2.6
    if prior_ratio is not None:
        score -= min(max(prior_ratio, 0.0), 1.0) / 3.0
    score = float(np.clip(score, 0.0, 1.0))

    label = "low"
    if score >= 0.67:
        label = "high"
    elif score >= 0.40:
        label = "medium"

    cohort_median_prior_mae = None
    if "prior_mae_eur" in cohort.columns:
        cohort_median_prior_mae = _safe_float(
            pd.to_numeric(cohort["prior_mae_eur"], errors="coerce").median()
        )

    return {
        "label": label,
        "score": score,
        "undervaluation_confidence": confidence_signal,
        "interval_low_eur": low,
        "interval_high_eur": high,
        "interval_width_eur": width,
        "interval_width_ratio": width_ratio,
        "prior_mae_eur": prior_mae,
        "prior_mae_ratio_to_prediction": prior_ratio,
        "cohort_median_prior_mae_eur": cohort_median_prior_mae,
    }


def _build_valuation_guardrails(row: pd.Series) -> dict[str, Any]:
    pred = _safe_float(row.get("fair_value_eur"))
    if pred is None:
        pred = _safe_float(row.get("expected_value_eur"))
    market = _safe_float(row.get("market_value_eur"))

    raw_gap = _safe_float(row.get("value_gap_raw_eur"))
    if raw_gap is None:
        raw_gap = _safe_float(row.get("value_gap_eur"))
    cons_gap = _safe_float(row.get("value_gap_conservative_eur"))
    if raw_gap is None and pred is not None and market is not None:
        raw_gap = pred - market
    if cons_gap is None:
        cons_gap = raw_gap

    capped_gap = _safe_float(row.get("value_gap_capped_eur"))
    cap_threshold = _safe_float(row.get("value_gap_cap_threshold_eur"))
    cap_ratio = _safe_float(row.get("value_gap_cap_ratio"))
    cap_applied_raw = row.get("value_gap_cap_applied")
    cap_applied = bool(cap_applied_raw) if cap_applied_raw is not None and not pd.isna(cap_applied_raw) else False

    prior_mae = _safe_float(row.get("prior_mae_eur"))
    prior_p75ae = _safe_float(row.get("prior_p75ae_eur"))
    prior_qae = _safe_float(row.get("prior_qae_eur"))

    if cap_threshold is None:
        cap_candidates: list[float] = []
        if prior_qae is not None and prior_qae > 0:
            cap_candidates.append(2.5 * prior_qae)
        if prior_p75ae is not None and prior_p75ae > 0:
            cap_candidates.append(3.0 * prior_p75ae)
        if prior_mae is not None and prior_mae > 0:
            cap_candidates.append(4.0 * prior_mae)
        cap_threshold = min(cap_candidates) if cap_candidates else None

    if capped_gap is None:
        capped_gap = cons_gap
        if cons_gap is not None and cap_threshold is not None and cons_gap > 0:
            capped_gap = min(cons_gap, cap_threshold)
            cap_applied = bool(capped_gap < cons_gap - 1.0)

    if cap_ratio is None and cons_gap is not None and cap_threshold is not None and cap_threshold > 0:
        cap_ratio = cons_gap / cap_threshold

    return {
        "market_value_eur": market,
        "fair_value_eur": pred,
        "value_gap_raw_eur": raw_gap,
        "value_gap_conservative_eur": cons_gap,
        "value_gap_capped_eur": capped_gap,
        "cap_threshold_eur": cap_threshold,
        "cap_applied": cap_applied,
        "cap_ratio": cap_ratio,
        "prior_mae_eur": prior_mae,
        "prior_p75ae_eur": prior_p75ae,
        "prior_qae_eur": prior_qae,
    }


def _fmt_eur(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if v >= 1_000_000_000:
        return f"{sign}EUR {v / 1_000_000_000:.2f}bn"
    if v >= 1_000_000:
        return f"{sign}EUR {v / 1_000_000:.1f}m"
    if v >= 1_000:
        return f"{sign}EUR {v / 1_000:.0f}k"
    return f"{sign}EUR {v:.0f}"


def _build_summary_text(
    row: pd.Series,
    strengths: list[dict[str, Any]],
    development_levers: list[dict[str, Any]],
    risk_flags: list[dict[str, str]],
    confidence: dict[str, Any],
    valuation_guardrails: dict[str, Any],
) -> str:
    name = str(row.get("name") or row.get("player_id") or "Player")
    conf_label = str(confidence.get("label", "medium"))
    market = _fmt_eur(valuation_guardrails.get("market_value_eur"))
    fair = _fmt_eur(valuation_guardrails.get("fair_value_eur"))
    gap = _fmt_eur(valuation_guardrails.get("value_gap_conservative_eur"))
    capped = _fmt_eur(valuation_guardrails.get("value_gap_capped_eur"))

    top_strengths = ", ".join(m["label"] for m in strengths[:2]) if strengths else "no standout metric edge"
    top_levers = ", ".join(m["label"] for m in development_levers[:2]) if development_levers else "no clear lever"
    history_tier = str(row.get("history_strength_tier") or "").strip()
    history_score = _safe_float(row.get("history_strength_score"))
    history_note = None
    if history_tier or history_score is not None:
        if history_score is None:
            history_note = f"History profile: {history_tier}."
        elif history_tier:
            history_note = f"History profile: {history_tier} ({history_score:.0f}/100)."
        else:
            history_note = f"History profile score: {history_score:.0f}/100."

    sentences = [
        f"{name}: {conf_label}-confidence undervaluation signal (market {market}, fair value {fair}).",
        f"Conservative value gap is {gap}; guardrailed gap is {capped}.",
        f"Top strengths: {top_strengths}. Development focus: {top_levers}.",
    ]
    if history_note:
        sentences.append(history_note)
    if risk_flags:
        top_risks = ", ".join(flag["code"] for flag in risk_flags[:2])
        sentences.append(f"Key risk flags: {top_risks}.")
    else:
        sentences.append("No major risk flag triggered by current thresholds.")
    return " ".join(sentences)


def _build_risk_flags(
    row: pd.Series,
    cohort: pd.DataFrame,
    confidence: dict[str, Any],
    valuation_guardrails: dict[str, Any],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []

    minutes = _safe_float(row.get("minutes"))
    if minutes is None:
        minutes = _safe_float(row.get("sofa_minutesPlayed"))
    if minutes is not None and minutes < 900:
        flags.append(
            {
                "severity": "medium",
                "code": "low_minutes",
                "message": "Low minute sample this season; performance signal is less stable.",
            }
        )
    elif minutes is not None and minutes < 1200:
        flags.append(
            {
                "severity": "low",
                "code": "low_minutes_watch",
                "message": "Minutes are below a full-season sample; monitor stability.",
            }
        )

    width_ratio = confidence.get("interval_width_ratio")
    if width_ratio is not None and width_ratio >= 1.0:
        flags.append(
            {
                "severity": "high",
                "code": "high_uncertainty",
                "message": "Prediction interval is wide relative to predicted value.",
            }
        )
    elif width_ratio is not None and width_ratio >= 0.70:
        flags.append(
            {
                "severity": "medium",
                "code": "medium_uncertainty",
                "message": "Prediction interval is moderately wide; prefer conservative valuation.",
            }
        )

    injury_burden = _safe_float(row.get("injury_days_per_1000_min"))
    if injury_burden is not None and "injury_days_per_1000_min" in cohort.columns:
        cohort_injury = pd.to_numeric(cohort["injury_days_per_1000_min"], errors="coerce").dropna()
        if len(cohort_injury) >= 20:
            p65 = float(np.nanquantile(cohort_injury.to_numpy(), 0.65))
            p85 = float(np.nanquantile(cohort_injury.to_numpy(), 0.85))
            if injury_burden >= max(p85, 60.0):
                flags.append(
                    {
                        "severity": "high",
                        "code": "injury_burden_high",
                        "message": "Injury burden is high versus cohort and may reduce reliability.",
                    }
                )
            elif injury_burden >= max(p65, 35.0):
                flags.append(
                    {
                        "severity": "medium",
                        "code": "injury_burden",
                        "message": "Injury burden is above cohort baseline.",
                    }
                )

    contract_years = _safe_float(row.get("contract_years_left"))
    if contract_years is not None and contract_years <= 0.5:
        flags.append(
            {
                "severity": "medium",
                "code": "contract_very_short",
                "message": "Very short contract horizon can strongly distort transfer valuation.",
            }
        )
    elif contract_years is not None and contract_years <= 1.0:
        flags.append(
            {
                "severity": "low",
                "code": "contract_horizon",
                "message": "Short contract horizon can distort valuation and transfer dynamics.",
            }
        )

    history_cov = _safe_float(row.get("history_strength_coverage"))
    history_score = _safe_float(row.get("history_strength_score"))
    if history_cov is not None and history_cov < 0.35:
        flags.append(
            {
                "severity": "medium",
                "code": "history_data_sparse",
                "message": "Historical signal coverage is sparse; treat development trend cautiously.",
            }
        )
    elif history_score is not None and history_score < 40.0:
        flags.append(
            {
                "severity": "low",
                "code": "history_strength_low",
                "message": "Historical stability/momentum profile is below cohort baseline.",
            }
        )

    if valuation_guardrails.get("cap_applied"):
        cap_ratio = valuation_guardrails.get("cap_ratio")
        severity = "medium"
        if isinstance(cap_ratio, (float, int)) and cap_ratio >= 1.75:
            severity = "high"
        flags.append(
            {
                "severity": severity,
                "code": "valuation_optimism_guardrail",
                "message": "Raw undervaluation is above historical error priors; capped for conservative decisions.",
            }
        )

    return flags


def get_player_report(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    top_metrics: int = 5,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    row = _select_player_row(frame=frame, player_id=player_id, season=season)
    row_dict = _to_records(row.to_frame().T)[0]

    cohort, cohort_filters = _cohort_for_player(frame=frame, row=row)
    metric_profile = _build_metric_profile(row=row, cohort=cohort, top_metrics=max(int(top_metrics), 1))
    confidence = _build_confidence_summary(row=row, cohort=cohort)
    valuation_guardrails = _build_valuation_guardrails(row=row)
    risk_flags = _build_risk_flags(
        row=row,
        cohort=cohort,
        confidence=confidence,
        valuation_guardrails=valuation_guardrails,
    )
    summary_text = _build_summary_text(
        row=row,
        strengths=metric_profile["strengths"],
        development_levers=metric_profile["development_levers"],
        risk_flags=risk_flags,
        confidence=confidence,
        valuation_guardrails=valuation_guardrails,
    )

    return {
        "player": row_dict,
        "cohort": {
            "size": int(len(cohort)),
            "filters": cohort_filters,
        },
        "strengths": metric_profile["strengths"],
        "weaknesses": metric_profile["weaknesses"],
        "development_levers": metric_profile["development_levers"],
        "risk_flags": risk_flags,
        "confidence": confidence,
        "valuation_guardrails": valuation_guardrails,
        "summary_text": summary_text,
    }


def _build_history_strength_payload(row: pd.Series) -> dict[str, Any]:
    score = _safe_float(row.get("history_strength_score"))
    coverage = _safe_float(row.get("history_strength_coverage"))
    tier = row.get("history_strength_tier")
    tier_text = str(tier).strip() if tier is not None and not pd.isna(tier) else "uncertain"

    components: list[dict[str, Any]] = []
    for key in HISTORY_COMPONENT_COLUMNS:
        value = _safe_float(row.get(key))
        weight = float(HISTORY_COMPONENT_WEIGHTS.get(key, 0.0))
        label = HISTORY_COMPONENT_LABELS.get(key, key)
        weighted_points = None if value is None else float(value * weight * 100.0)
        components.append(
            {
                "key": key,
                "label": label,
                "value_0_to_1": value,
                "value_0_to_100": None if value is None else float(value * 100.0),
                "weight": weight,
                "weighted_points_0_to_100": weighted_points,
                "missing": value is None,
            }
        )

    components_sorted = sorted(
        components,
        key=lambda x: x["weighted_points_0_to_100"] if isinstance(x["weighted_points_0_to_100"], (float, int)) else -1.0,
        reverse=True,
    )
    strongest = [c for c in components_sorted if not c["missing"]][:3]

    weakest = sorted(
        [c for c in components if not c["missing"]],
        key=lambda x: x["value_0_to_1"] if isinstance(x["value_0_to_1"], (float, int)) else 1.0,
    )[:3]

    if score is None:
        narrative = "History strength score is unavailable because required history components are missing."
    else:
        narrative = f"History strength is {score:.1f}/100 ({tier_text})."
        if strongest:
            narrative += " Strongest components: " + ", ".join(c["label"] for c in strongest[:2]) + "."
        if weakest:
            narrative += " Development focus: " + ", ".join(c["label"] for c in weakest[:2]) + "."

    return {
        "score_0_to_100": score,
        "coverage_0_to_1": coverage,
        "tier": tier_text,
        "components": components,
        "strongest_components": strongest,
        "improvement_components": weakest,
        "summary_text": narrative,
    }


def get_player_history_strength(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    row = _select_player_row(frame=frame, player_id=player_id, season=season)
    row_dict = _to_records(row.to_frame().T)[0]
    return {
        "player": row_dict,
        "history_strength": _build_history_strength_payload(row=row),
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_watchlist_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _write_watchlist_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def list_watchlist(
    *,
    split: Split | None = None,
    tag: str | None = None,
    player_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    path = _watchlist_path()
    records = _read_watchlist_records(path)

    out: list[dict[str, Any]] = []
    for item in records:
        if split and str(item.get("split", "")).lower() != str(split):
            continue
        if tag and str(item.get("tag", "")).strip().casefold() != str(tag).strip().casefold():
            continue
        if player_id and str(item.get("player_id", "")) != str(player_id):
            continue
        out.append(item)

    out.sort(key=lambda x: str(x.get("created_at_utc", "")), reverse=True)
    total = len(out)
    start = max(int(offset), 0)
    end = start + max(int(limit), 0)
    page = out[start:end]

    return {
        "path": str(path),
        "total": int(total),
        "count": int(len(page)),
        "limit": int(limit),
        "offset": int(offset),
        "items": page,
    }


def add_watchlist_item(
    *,
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    tag: str | None = None,
    notes: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    row = get_player_prediction(player_id=player_id, split=split, season=season)
    report = get_player_report(player_id=player_id, split=split, season=season, top_metrics=5)
    guardrails = report.get("valuation_guardrails", {})
    confidence = report.get("confidence", {})
    risk_flags = report.get("risk_flags", [])

    record = {
        "watch_id": uuid.uuid4().hex,
        "created_at_utc": _now_utc_iso(),
        "split": split,
        "season": str(row.get("season") or season or ""),
        "player_id": str(row.get("player_id") or player_id),
        "name": row.get("name"),
        "league": row.get("league"),
        "club": row.get("club"),
        "position": row.get("model_position") or row.get("position_group"),
        "tag": tag or "",
        "notes": notes or "",
        "source": source or "manual",
        "market_value_eur": _safe_float(row.get("market_value_eur")),
        "fair_value_eur": _safe_float(row.get("fair_value_eur") or row.get("expected_value_eur")),
        "value_gap_capped_eur": _safe_float(guardrails.get("value_gap_capped_eur")),
        "value_gap_conservative_eur": _safe_float(guardrails.get("value_gap_conservative_eur")),
        "undervaluation_confidence": _safe_float(row.get("undervaluation_confidence")),
        "confidence_label": confidence.get("label"),
        "risk_codes": [str(flag.get("code")) for flag in risk_flags if isinstance(flag, dict) and flag.get("code")],
        "summary_text": report.get("summary_text"),
    }

    path = _watchlist_path()
    records = _read_watchlist_records(path)
    records.append(record)
    _write_watchlist_records(path, records)
    return record


def delete_watchlist_item(watch_id: str) -> dict[str, Any]:
    path = _watchlist_path()
    records = _read_watchlist_records(path)
    keep = [row for row in records if str(row.get("watch_id")) != str(watch_id)]
    deleted = len(keep) != len(records)
    if deleted:
        _write_watchlist_records(path, keep)
    return {
        "path": str(path),
        "watch_id": str(watch_id),
        "deleted": bool(deleted),
    }


def get_model_manifest() -> dict[str, Any]:
    test_path = _resolve_path(*SPLIT_TO_PATH["test"])
    val_path = _resolve_path(*SPLIT_TO_PATH["val"])
    metrics_path = _resolve_path(METRICS_ENV, DEFAULT_METRICS)
    manifest_path = _resolve_path(MODEL_MANIFEST_ENV, DEFAULT_MODEL_MANIFEST)

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["_meta"] = {
            "source": "file",
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "mtime_utc": _file_meta(manifest_path).get("mtime_utc"),
        }
        return payload

    out: dict[str, Any] = {
        "registry_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "derived",
        "artifacts": {
            "test_predictions": {
                "path": str(test_path),
                **_file_meta(test_path),
                "sha256": _sha256_file(test_path),
            },
            "val_predictions": {
                "path": str(val_path),
                **_file_meta(val_path),
                "sha256": _sha256_file(val_path),
            },
            "metrics": {
                "path": str(metrics_path),
                **_file_meta(metrics_path),
                "sha256": _sha256_file(metrics_path),
            },
        },
        "config": {},
        "summary": {},
    }

    try:
        metrics = get_metrics()
        out["config"] = {
            "dataset": metrics.get("dataset"),
            "val_season": metrics.get("val_season"),
            "test_season": metrics.get("test_season"),
            "trials_per_position": metrics.get("trials_per_position"),
            "recency_half_life": metrics.get("recency_half_life"),
            "optimize_metric": metrics.get("optimize_metric"),
            "interval_q": metrics.get("interval_q"),
            "two_stage_band_model": metrics.get("two_stage_band_model"),
            "band_min_samples": metrics.get("band_min_samples"),
            "band_blend_alpha": metrics.get("band_blend_alpha"),
            "strict_leakage_guard": metrics.get("strict_leakage_guard"),
        }
        summary = {
            "overall": metrics.get("overall"),
            "segments": metrics.get("segments"),
            "holdout": metrics.get("holdout"),
            "artifacts": metrics.get("artifacts"),
        }
        out["summary"] = summary
    except Exception as exc:
        out["metrics_error"] = str(exc)

    return out


def health_payload() -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "ok",
        "artifacts": {},
    }
    test_path = _resolve_path(*SPLIT_TO_PATH["test"])
    val_path = _resolve_path(*SPLIT_TO_PATH["val"])
    metrics_path = _resolve_path(METRICS_ENV, DEFAULT_METRICS)
    test_meta = _file_meta(test_path)
    val_meta = _file_meta(val_path)
    metrics_meta = _file_meta(metrics_path)
    out["artifacts"] = {
        "test_predictions_path": str(test_path),
        "val_predictions_path": str(val_path),
        "metrics_path": str(metrics_path),
        "test_predictions_exists": test_meta["exists"],
        "val_predictions_exists": val_meta["exists"],
        "metrics_exists": metrics_meta["exists"],
        "test_predictions_size_bytes": test_meta["size_bytes"],
        "val_predictions_size_bytes": val_meta["size_bytes"],
        "metrics_size_bytes": metrics_meta["size_bytes"],
        "test_predictions_mtime_utc": test_meta["mtime_utc"],
        "val_predictions_mtime_utc": val_meta["mtime_utc"],
        "metrics_mtime_utc": metrics_meta["mtime_utc"],
    }
    try:
        out["test_rows"] = int(len(get_predictions("test")))
    except Exception as exc:
        out["test_rows"] = None
        out["test_error"] = str(exc)
    try:
        out["val_rows"] = int(len(get_predictions("val")))
    except Exception as exc:
        out["val_rows"] = None
        out["val_error"] = str(exc)
    try:
        metrics = get_metrics()
        out["metrics_loaded"] = True
        out["metrics_dataset"] = metrics.get("dataset")
        out["metrics_test_season"] = metrics.get("test_season")
        out["metrics_val_season"] = metrics.get("val_season")
    except Exception as exc:
        out["metrics_loaded"] = False
        out["metrics_error"] = str(exc)
    return out


__all__ = [
    "ArtifactNotFoundError",
    "Split",
    "add_watchlist_item",
    "delete_watchlist_item",
    "get_active_artifacts",
    "get_model_manifest",
    "get_metrics",
    "get_player_history_strength",
    "get_player_report",
    "get_player_prediction",
    "get_predictions",
    "get_resolved_artifact_paths",
    "health_payload",
    "list_watchlist",
    "query_predictions",
    "query_scout_targets",
    "query_shortlist",
    "validate_strict_artifact_env",
]
