"""Service functions for market-value prediction artifacts."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import numpy as np
import pandas as pd

from scouting_ml.core.runtime_config import PRODUCTION_PIPELINE_DEFAULTS
from scouting_ml.features.history_strength import add_history_strength_features
from scouting_ml.reporting.operator_health import (
    build_valuation_promotion_gate,
    load_ingestion_health_payload,
    latest_timestamp,
)
from scouting_ml.reporting.market_value_benchmarks import build_market_value_benchmark_payload
from scouting_ml.scouting.system_fit import (
    TRUST_SCOPE_ALLOWED,
    list_system_fit_templates,
    rank_system_fit_slots,
)
from scouting_ml.services.market_value_artifacts import (
    BENCHMARK_REPORT_ENV,
    CALIBRATION_MIN_SAMPLES_ENV,
    ChampionRole,
    DEFAULT_BENCHMARK_REPORT,
    DEFAULT_METRICS,
    ENABLE_RESIDUAL_CALIBRATION_ENV,
    METRICS_ENV,
    MODEL_MANIFEST_ENV,
    ROLE_TO_MANIFEST_KEY,
    SPLIT_TO_PATH,
    Split,
    decisions_path as _decisions_path,
    env_flag as _env_flag,
    file_meta as _file_meta,
    get_active_artifacts,
    get_resolved_artifact_paths,
    lane_state_for_role,
    manifest_path as _manifest_path,
    manifest_role_section as _manifest_role_section,
    manifest_targets_active_artifacts as _manifest_targets_active_artifacts,
    normalized_path_str as _normalized_path_str,
    resolve_path as _resolve_path,
    resolve_role_artifact_paths as _resolve_role_artifact_paths,
    sha256_file as _sha256_file,
    validate_strict_artifact_env,
    watchlist_path as _watchlist_path,
)
from scouting_ml.services.watchlist_store import (
    delete_watchlist_record,
    read_watchlist_records,
    upsert_watchlist_record,
)
from scouting_ml.services.scout_decision_store import (
    append_scout_decision_record,
    read_scout_decision_records,
)
from scouting_ml.services.market_value_profile_context import (
    build_availability_context,
    build_external_tactical_context,
    build_history_strength_payload,
    build_market_context_payload,
    build_profile_stat_groups,
    build_provider_coverage,
    build_provider_risk_flags,
    format_eur,
)
from scouting_ml.services.market_value_profile_taxonomy import (
    ADVANCED_METRIC_DIRECTION,
    ADVANCED_METRIC_LABEL,
    ADVANCED_METRIC_SPECS,
    ARCHETYPE_TEMPLATES,
    FORMATION_FIT_TEMPLATES,
    PROFILE_METRIC_SPECS,
    RADAR_AXES_BY_POSITION,
    ROLE_KEY_LABELS,
)

FUTURE_OVERLAY_COLUMNS: tuple[str, ...] = (
    "_player_key",
    "_season_key",
    "future_growth_probability",
    "future_scout_blend_score",
    "future_scout_score",
    "future_potential_score",
    "future_family_weighted_score",
    "future_potential_confidence",
    "future_sample_adequacy_score",
    "future_data_quality_score",
    "future_context_coverage_score",
    "future_model_support_score",
    "talent_position_family",
    "talent_impact_score",
    "talent_impact_coverage",
    "talent_technical_score",
    "talent_technical_coverage",
    "talent_tactical_score",
    "talent_tactical_coverage",
    "talent_physical_score",
    "talent_physical_coverage",
    "talent_context_score",
    "talent_context_coverage",
    "talent_trajectory_score",
    "talent_trajectory_coverage",
    "has_next_season_target",
    "next_market_value_eur",
    "next_minutes",
    "next_season",
    "value_growth_gt25pct_flag",
    "value_growth_next_season_eur",
    "value_growth_next_season_log_delta",
    "value_growth_next_season_pct",
    "value_growth_positive_flag",
)

BIG5_LEAGUES = {
    "english premier league",
    "premier league",
    "spanish la liga",
    "la liga",
    "laliga",
    "italian serie a",
    "serie a",
    "german bundesliga",
    "bundesliga",
    "french ligue 1",
    "ligue 1",
}

TALENT_FAMILIES: tuple[str, ...] = (
    "impact",
    "technical",
    "tactical",
    "physical",
    "context",
    "trajectory",
)

TALENT_FAMILY_LABELS: dict[str, str] = {
    "impact": "Impact",
    "technical": "Technical",
    "tactical": "Tactical",
    "physical": "Physical",
    "context": "Context",
    "trajectory": "Trajectory",
}

CURRENT_LEVEL_FAMILY_WEIGHTS: dict[str, dict[str, float]] = {
    "GK": {"impact": 0.20, "technical": 0.14, "tactical": 0.26, "physical": 0.18, "context": 0.12, "trajectory": 0.10},
    "CB": {"impact": 0.22, "technical": 0.12, "tactical": 0.28, "physical": 0.18, "context": 0.12, "trajectory": 0.08},
    "FB": {"impact": 0.24, "technical": 0.15, "tactical": 0.20, "physical": 0.20, "context": 0.11, "trajectory": 0.10},
    "CM": {"impact": 0.28, "technical": 0.20, "tactical": 0.24, "physical": 0.08, "context": 0.10, "trajectory": 0.10},
    "AM": {"impact": 0.31, "technical": 0.22, "tactical": 0.18, "physical": 0.07, "context": 0.10, "trajectory": 0.12},
    "W": {"impact": 0.32, "technical": 0.21, "tactical": 0.14, "physical": 0.11, "context": 0.08, "trajectory": 0.14},
    "ST": {"impact": 0.36, "technical": 0.20, "tactical": 0.10, "physical": 0.09, "context": 0.10, "trajectory": 0.15},
}

FUTURE_POTENTIAL_FAMILY_WEIGHTS: dict[str, dict[str, float]] = {
    "GK": {"impact": 0.12, "technical": 0.12, "tactical": 0.26, "physical": 0.20, "context": 0.18, "trajectory": 0.12},
    "CB": {"impact": 0.16, "technical": 0.12, "tactical": 0.28, "physical": 0.22, "context": 0.14, "trajectory": 0.08},
    "FB": {"impact": 0.18, "technical": 0.14, "tactical": 0.22, "physical": 0.20, "context": 0.12, "trajectory": 0.14},
    "CM": {"impact": 0.16, "technical": 0.20, "tactical": 0.24, "physical": 0.10, "context": 0.12, "trajectory": 0.18},
    "AM": {"impact": 0.23, "technical": 0.20, "tactical": 0.18, "physical": 0.08, "context": 0.12, "trajectory": 0.19},
    "W": {"impact": 0.24, "technical": 0.19, "tactical": 0.14, "physical": 0.12, "context": 0.10, "trajectory": 0.21},
    "ST": {"impact": 0.28, "technical": 0.20, "tactical": 0.12, "physical": 0.10, "context": 0.16, "trajectory": 0.14},
}

class ArtifactNotFoundError(FileNotFoundError):
    """Raised when required model artifacts are missing."""


@dataclass
class _FrameCache:
    key: str
    version: tuple[tuple[str, int], ...]
    frame: pd.DataFrame


@dataclass
class _ResidualCalibrationCache:
    path: Path
    mtime_ns: int
    min_samples: int
    payload: dict[str, Any]


@dataclass
class _LeagueHoldoutCache:
    version: tuple[tuple[str, int], ...]
    payload: dict[str, dict[str, Any]]


@dataclass
class _SummaryPayloadCache:
    version: tuple[tuple[str, int], ...]
    payload: dict[str, Any]


_PRED_CACHE: Dict[str, _FrameCache] = {}
_METRICS_CACHE: Dict[str, tuple[Path, int, dict[str, Any]]] = {}
_RESIDUAL_CALIBRATION_CACHE: _ResidualCalibrationCache | None = None
_LEAGUE_HOLDOUT_CACHE: _LeagueHoldoutCache | None = None
_INGESTION_HEALTH_CACHE: _SummaryPayloadCache | None = None
_OPERATOR_HEALTH_CACHE: _SummaryPayloadCache | None = None
_UI_BOOTSTRAP_CACHE: Dict[str, _SummaryPayloadCache] = {}

LEAGUE_TRUST_ALPHA_ADJUSTMENTS: dict[str, float] = {
    "trusted": 0.0,
    "watch": -0.05,
    "unknown": -0.10,
    "blocked": -0.20,
}

LEAGUE_ADJUSTMENT_BUCKET_CAPS: dict[str, float] = {
    "severe_failed": 0.25,
    "failed": 0.45,
    "weak": 0.70,
    "standard": 0.90,
    "unknown": 0.60,
}
DISCOVERY_TRUST_WEIGHTS: dict[str, float] = {
    "trusted": 1.00,
    "watch": 0.94,
    "unknown": 0.88,
    "blocked": 0.72,
}
DISCOVERY_BUCKET_WEIGHTS: dict[str, float] = {
    "standard": 1.00,
    "weak": 0.90,
    "failed": 0.78,
    "severe_failed": 0.60,
    "unknown": 0.85,
}
DISCOVERY_WEIGHTED_SORT_COLUMNS: set[str] = {
    "value_gap_capped_eur",
    "value_gap_conservative_eur",
    "value_gap_eur",
    "undervaluation_score",
    "fair_value_eur",
    "expected_value_eur",
}
SCOUT_DECISION_ACTIONS: tuple[str, ...] = ("shortlist", "watch_live", "request_report", "pass")
POSITIVE_SCOUT_DECISION_ACTIONS: tuple[str, ...] = ("shortlist", "watch_live", "request_report")
SCOUT_DECISION_REASON_TAGS: dict[str, tuple[str, ...]] = {
    "positive": (
        "system_fit",
        "price_gap",
        "trajectory",
        "role_need",
        "high_confidence",
        "availability",
        "market_opportunity",
    ),
    "pass": (
        "too_expensive",
        "data_too_thin",
        "league_risk",
        "not_system_fit",
        "athletic_concern",
        "technical_ceiling",
        "injury_risk",
        "contract_blocked",
    ),
}


def _prediction_cache_version(*paths: Path) -> tuple[tuple[str, int], ...]:
    return tuple(
        (_normalized_path_str(path), int(path.stat().st_mtime_ns) if path.exists() else -1)
        for path in paths
    )


def _get_ingestion_health_payload() -> dict[str, Any]:
    global _INGESTION_HEALTH_CACHE
    clean_dataset_path = Path(PRODUCTION_PIPELINE_DEFAULTS.clean_output)
    version = _prediction_cache_version(clean_dataset_path)
    if _INGESTION_HEALTH_CACHE is not None and _INGESTION_HEALTH_CACHE.version == version:
        return _INGESTION_HEALTH_CACHE.payload
    payload = load_ingestion_health_payload(clean_dataset_path=clean_dataset_path)
    _INGESTION_HEALTH_CACHE = _SummaryPayloadCache(version=version, payload=payload)
    return payload


def _load_json_snapshot(path: str | Path | None) -> dict[str, Any] | list[Any] | None:
    if path is None:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    try:
        return json.loads(path_obj.read_text(encoding="utf-8"))
    except Exception:
        return None


def _slugify(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")


def _safe_numeric(frame: pd.DataFrame, col: str) -> None:
    if col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")


def _replace_inf_with_nan(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return out
    numeric = out.loc[:, numeric_cols]
    out.loc[:, numeric_cols] = numeric.where(np.isfinite(numeric), np.nan)
    return out


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
        "future_growth_probability",
        "future_scout_blend_score",
        "future_scout_score",
        "future_potential_score",
        "future_family_weighted_score",
        "future_potential_confidence",
        "future_sample_adequacy_score",
        "future_data_quality_score",
        "future_context_coverage_score",
        "future_model_support_score",
        "talent_impact_score",
        "talent_impact_coverage",
        "talent_technical_score",
        "talent_technical_coverage",
        "talent_tactical_score",
        "talent_tactical_coverage",
        "talent_physical_score",
        "talent_physical_coverage",
        "talent_context_score",
        "talent_context_coverage",
        "talent_trajectory_score",
        "talent_trajectory_coverage",
        "current_level_score",
        "current_level_confidence",
        "future_potential_confidence",
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


def _get_role_predictions(role: ChampionRole, split: Split) -> pd.DataFrame:
    paths = _resolve_role_artifact_paths(role)
    path = paths["test_predictions"] if split == "test" else paths["val_predictions"]
    if not path.exists():
        raise ArtifactNotFoundError(f"{role} {split} predictions artifact not found: {path}")

    cache_key = f"{role}:{split}"
    version = _prediction_cache_version(path)
    cached = _PRED_CACHE.get(cache_key)
    if cached and cached.version == version:
        return cached.frame.copy()

    frame = _load_predictions(path)
    _PRED_CACHE[cache_key] = _FrameCache(key=cache_key, version=version, frame=frame)
    return frame.copy()


def _merge_future_shortlist_overlay(base_frame: pd.DataFrame, future_frame: pd.DataFrame) -> pd.DataFrame:
    overlay_cols = [col for col in FUTURE_OVERLAY_COLUMNS if col in future_frame.columns and col not in base_frame.columns]
    if not overlay_cols:
        return base_frame.copy()

    merge_keys: list[str] = []
    for candidate in (["player_id", "season"], ["player_id"], ["name", "club", "season"]):
        if all(col in base_frame.columns and col in future_frame.columns for col in candidate):
            merge_keys = candidate
            break
    if not merge_keys:
        return base_frame.copy()

    overlay = future_frame.loc[:, merge_keys + overlay_cols].copy()
    overlay = overlay.drop_duplicates(subset=merge_keys, keep="first")
    return base_frame.merge(overlay, on=merge_keys, how="left")


def get_predictions(split: Split = "test") -> pd.DataFrame:
    valuation_paths = _resolve_role_artifact_paths("valuation")
    future_paths = _resolve_role_artifact_paths("future_shortlist")
    valuation_path = valuation_paths["test_predictions"] if split == "test" else valuation_paths["val_predictions"]
    future_path = future_paths["test_predictions"] if split == "test" else future_paths["val_predictions"]

    merged_cache_key = f"merged:{split}"
    merged_version = _prediction_cache_version(valuation_path, future_path)
    cached = _PRED_CACHE.get(merged_cache_key)
    if cached and cached.version == merged_version:
        return cached.frame.copy()

    base_frame = _get_role_predictions("valuation", split)
    if _normalized_path_str(valuation_path) == _normalized_path_str(future_path):
        _PRED_CACHE[merged_cache_key] = _FrameCache(key=merged_cache_key, version=merged_version, frame=base_frame)
        return base_frame.copy()

    future_frame = _get_role_predictions("future_shortlist", split)
    merged = _merge_future_shortlist_overlay(base_frame, future_frame)
    _PRED_CACHE[merged_cache_key] = _FrameCache(key=merged_cache_key, version=merged_version, frame=merged)
    return merged.copy()


def get_metrics(role: ChampionRole = "valuation") -> dict[str, Any]:
    path = _resolve_role_artifact_paths(role)["metrics"]
    if not path.exists():
        raise ArtifactNotFoundError(f"{role} metrics artifact not found: {path}")

    cache_key = f"{role}:{_normalized_path_str(path)}"
    mtime = path.stat().st_mtime_ns
    cached = _METRICS_CACHE.get(cache_key)
    if cached and cached[0] == path and cached[1] == mtime:
        return dict(cached[2])

    payload = json.loads(path.read_text(encoding="utf-8"))
    _METRICS_CACHE[cache_key] = (path, mtime, payload)
    return dict(payload)


def get_benchmark_report() -> dict[str, Any]:
    report_path = _resolve_path(BENCHMARK_REPORT_ENV, DEFAULT_BENCHMARK_REPORT)
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["_meta"] = {
            "source": "file",
            "path": str(report_path),
            "mtime_utc": _file_meta(report_path).get("mtime_utc"),
            "sha256": _sha256_file(report_path),
        }
        return payload

    valuation_paths = _resolve_role_artifact_paths("valuation")
    metrics_path = valuation_paths["metrics"]
    test_path = valuation_paths["test_predictions"]
    payload = build_market_value_benchmark_payload(
        metrics_path=str(metrics_path),
        predictions_path=str(test_path),
    )
    payload["_meta"] = {
        "source": "derived",
        "path": str(report_path),
        "mtime_utc": None,
        "sha256": None,
    }
    return payload


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
    computed_raw_gap = out["expected_value_eur"] - market
    existing_raw_gap = _to_numeric_series(out, "value_gap_raw_eur")
    existing_gap = _to_numeric_series(out, "value_gap_eur")
    out["value_gap_raw_eur"] = existing_raw_gap.where(existing_raw_gap.notna(), existing_gap)
    out["value_gap_raw_eur"] = out["value_gap_raw_eur"].where(out["value_gap_raw_eur"].notna(), computed_raw_gap)
    out["value_gap_eur"] = out["value_gap_raw_eur"]

    existing_cons_gap = _to_numeric_series(out, "value_gap_conservative_eur")
    if "expected_value_low_eur" in out.columns:
        conservative_gap = _to_numeric_series(out, "expected_value_low_eur") - market
    else:
        conservative_gap = out["value_gap_raw_eur"]
    out["value_gap_conservative_eur"] = existing_cons_gap.where(existing_cons_gap.notna(), conservative_gap)

    out = _build_capped_gap_columns(out)
    return out


def _holdout_metrics_candidate_paths(metrics_path: Path, metrics_payload: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for row in metrics_payload.get("league_holdout") or []:
        if not isinstance(row, dict):
            continue
        raw_path = _safe_text(row.get("metrics_json"))
        if raw_path:
            candidates.append(Path(raw_path))

    stem = metrics_path.name
    suffix = ".metrics.json"
    if stem.endswith(suffix):
        pattern = f"{stem[: -len(suffix)]}.holdout_*.metrics.json"
        candidates.extend(metrics_path.parent.glob(pattern))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = _normalized_path_str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _register_holdout_metrics(
    out: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> None:
    league = _safe_text(payload.get("league"))
    if not league:
        return
    slug = _slugify(league)
    if not slug:
        return
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    domain_shift = payload.get("domain_shift") if isinstance(payload.get("domain_shift"), dict) else {}
    out[slug] = {
        "league": league,
        "league_slug": slug,
        "r2": _safe_float(overall.get("r2")),
        "wmape": _safe_float(overall.get("wmape")),
        "interval_coverage": _safe_float(
            overall.get("interval_coverage")
            if "interval_coverage" in overall
            else overall.get("coverage")
        ),
        "mean_abs_shift_z": _safe_float(domain_shift.get("mean_abs_shift_z")),
        "metrics_json": str(source_path) if source_path is not None else _safe_text(payload.get("metrics_json")),
    }


def _get_league_holdout_index() -> dict[str, dict[str, Any]]:
    global _LEAGUE_HOLDOUT_CACHE

    try:
        metrics_path = _resolve_role_artifact_paths("valuation")["metrics"]
        metrics_payload = get_metrics(role="valuation")
    except Exception:
        return {}

    sibling_paths = _holdout_metrics_candidate_paths(metrics_path, metrics_payload)
    version = _prediction_cache_version(metrics_path, *sibling_paths)
    if _LEAGUE_HOLDOUT_CACHE is not None and _LEAGUE_HOLDOUT_CACHE.version == version:
        return {key: dict(value) for key, value in _LEAGUE_HOLDOUT_CACHE.payload.items()}

    out: dict[str, dict[str, Any]] = {}
    for row in metrics_payload.get("league_holdout") or []:
        if isinstance(row, dict):
            _register_holdout_metrics(out, row)

    for path in sibling_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            _register_holdout_metrics(out, payload, source_path=path)

    _LEAGUE_HOLDOUT_CACHE = _LeagueHoldoutCache(version=version, payload={key: dict(value) for key, value in out.items()})
    return {key: dict(value) for key, value in out.items()}


def _league_adjustment_bucket(
    *,
    r2: float | None,
    wmape: float | None,
    interval_coverage: float | None,
    mean_abs_shift_z: float | None,
) -> str:
    if all(value is None for value in (r2, wmape, interval_coverage, mean_abs_shift_z)):
        return "unknown"
    if (
        (r2 is not None and r2 < -5.0)
        or (wmape is not None and wmape > 3.0)
        or (
            interval_coverage is not None
            and interval_coverage < 0.40
            and mean_abs_shift_z is not None
            and mean_abs_shift_z > 1.5
        )
    ):
        return "severe_failed"
    if (
        (r2 is not None and r2 < 0.0)
        or (wmape is not None and wmape > 1.25)
        or (interval_coverage is not None and interval_coverage < 0.50)
    ):
        return "failed"
    if (
        (r2 is not None and r2 < 0.35)
        or (wmape is not None and wmape > 0.75)
        or (interval_coverage is not None and interval_coverage < 0.60)
    ):
        return "weak"
    return "standard"


def _league_adjustment_reason(bucket: str) -> str:
    normalized = str(bucket or "unknown").strip().lower()
    if normalized == "severe_failed":
        return "Pricing heavily adjusted for failed holdout league."
    if normalized == "failed":
        return "Pricing adjusted for failed holdout reliability."
    if normalized == "weak":
        return "Pricing adjusted for weak league holdout reliability."
    if normalized == "unknown":
        return "Pricing adjusted conservatively because league holdout reliability is unknown."
    return "Standard league pricing adjustment applied."


def _apply_league_value_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out

    raw_fair = _to_numeric_series(out, "fair_value_eur").fillna(_to_numeric_series(out, "expected_value_eur"))
    raw_expected = _to_numeric_series(out, "expected_value_eur").fillna(raw_fair)
    market = _to_numeric_series(out, "market_value_eur")
    raw_gap = _to_numeric_series(out, "value_gap_raw_eur").fillna(_to_numeric_series(out, "value_gap_eur"))
    inferred_gap = raw_fair - market
    raw_gap = raw_gap.where(raw_gap.notna(), inferred_gap)

    raw_cons_gap = _to_numeric_series(out, "value_gap_conservative_eur").where(
        _to_numeric_series(out, "value_gap_conservative_eur").notna(),
        raw_gap,
    )

    raw_low = _to_numeric_series(out, "expected_value_low_raw_eur").where(
        _to_numeric_series(out, "expected_value_low_raw_eur").notna(),
        _to_numeric_series(out, "expected_value_low_eur"),
    )
    raw_high = _to_numeric_series(out, "expected_value_high_raw_eur").where(
        _to_numeric_series(out, "expected_value_high_raw_eur").notna(),
        _to_numeric_series(out, "expected_value_high_eur"),
    )

    holdout_index = _get_league_holdout_index()
    if not holdout_index:
        out["raw_fair_value_eur"] = raw_fair
        out["raw_value_gap_conservative_eur"] = raw_cons_gap
        out["league_adjusted_gap_eur"] = raw_gap
        out["league_adjusted_fair_value_eur"] = raw_fair
        out["league_adjustment_alpha"] = 1.0
        out["league_adjustment_bucket"] = "unknown"
        out["league_adjustment_reason"] = "No holdout metrics available; raw pricing retained."
        out["league_holdout_r2"] = np.nan
        out["league_holdout_wmape"] = np.nan
        out["league_holdout_interval_coverage"] = np.nan
        return out

    league_slugs = out.get("league", pd.Series("", index=out.index, dtype=object)).astype(str).map(_slugify)
    holdout_rows = [holdout_index.get(slug, {}) for slug in league_slugs]
    holdout_frame = pd.DataFrame(holdout_rows, index=out.index)

    def holdout_series(column: str) -> pd.Series:
        if column not in holdout_frame.columns:
            return pd.Series(np.nan, index=out.index, dtype=float)
        return pd.to_numeric(holdout_frame[column], errors="coerce")

    r2 = holdout_series("r2")
    wmape = holdout_series("wmape")
    interval_coverage = holdout_series("interval_coverage")
    mean_abs_shift_z = holdout_series("mean_abs_shift_z")

    buckets = pd.Series(
        [
            _league_adjustment_bucket(
                r2=_safe_float(r2.loc[idx]),
                wmape=_safe_float(wmape.loc[idx]),
                interval_coverage=_safe_float(interval_coverage.loc[idx]),
                mean_abs_shift_z=_safe_float(mean_abs_shift_z.loc[idx]),
            )
            for idx in out.index
        ],
        index=out.index,
        dtype=object,
    )

    league_strength = _to_numeric_series(out, "leaguectx_league_strength_index").fillna(0.0).clip(lower=0.0)
    league_strength_norm = (league_strength / 0.45).clip(lower=0.0, upper=1.0)
    uefa_total = _to_numeric_series(out, "uefa_coeff_5yr_total").fillna(0.0).clip(lower=0.0)
    uefa_norm = (np.log1p(uefa_total) / np.log1p(25.0)).clip(lower=0.0, upper=1.0)
    strength_blend = (0.65 * league_strength_norm) + (0.35 * uefa_norm)

    trust_tier = out.get("league_trust_tier", pd.Series("unknown", index=out.index, dtype=object)).astype(str).str.strip().str.lower()
    trust_adjustment = trust_tier.map(LEAGUE_TRUST_ALPHA_ADJUSTMENTS).fillna(LEAGUE_TRUST_ALPHA_ADJUSTMENTS["unknown"])
    provisional_alpha = 0.35 + (0.55 * strength_blend) + trust_adjustment
    bucket_caps = buckets.map(LEAGUE_ADJUSTMENT_BUCKET_CAPS).fillna(LEAGUE_ADJUSTMENT_BUCKET_CAPS["unknown"])
    alpha = provisional_alpha.clip(lower=0.20)
    alpha = pd.Series(np.minimum(alpha.to_numpy(dtype=float), bucket_caps.to_numpy(dtype=float)), index=out.index, dtype=float)

    adjusted_gap = raw_gap * alpha
    adjusted_cons_gap = raw_cons_gap * alpha

    adjusted_fair = market + adjusted_gap
    adjusted_fair = adjusted_fair.where(market.notna(), raw_fair)
    adjusted_expected = market + ((raw_expected - market) * alpha)
    adjusted_expected = adjusted_expected.where(market.notna(), raw_expected)

    if raw_low.notna().any():
        adjusted_low = market + ((raw_low - market) * alpha)
        adjusted_low = adjusted_low.where(market.notna(), raw_low)
        out["expected_value_low_eur"] = adjusted_low
    if raw_high.notna().any():
        adjusted_high = market + ((raw_high - market) * alpha)
        adjusted_high = adjusted_high.where(market.notna(), raw_high)
        out["expected_value_high_eur"] = adjusted_high

    out["raw_fair_value_eur"] = raw_fair
    out["raw_value_gap_conservative_eur"] = raw_cons_gap
    out["league_adjusted_gap_eur"] = adjusted_gap
    out["league_adjusted_fair_value_eur"] = adjusted_fair
    out["league_adjustment_alpha"] = alpha
    out["league_adjustment_bucket"] = buckets
    out["league_adjustment_reason"] = buckets.map(_league_adjustment_reason)
    out["league_holdout_r2"] = r2
    out["league_holdout_wmape"] = wmape
    out["league_holdout_interval_coverage"] = interval_coverage

    out["expected_value_eur"] = adjusted_expected
    out["fair_value_eur"] = adjusted_fair
    out["value_gap_eur"] = adjusted_gap
    out["value_gap_conservative_eur"] = adjusted_cons_gap

    return out


def _attach_talent_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out

    if "talent_position_family" not in out.columns:
        out["talent_position_family"] = out.apply(_infer_talent_position_family, axis=1)
    else:
        out["talent_position_family"] = out["talent_position_family"].astype(str).str.upper()

    for family in TALENT_FAMILIES:
        score_col = f"talent_{family}_score"
        if score_col not in out.columns:
            out[score_col] = 50.0
        else:
            out[score_col] = _to_numeric_series(out, score_col).fillna(50.0).clip(lower=0.0, upper=100.0)
        coverage_col = f"talent_{family}_coverage"
        if coverage_col not in out.columns:
            out[coverage_col] = 0.0
        else:
            out[coverage_col] = _to_numeric_series(out, coverage_col).fillna(0.0).clip(lower=0.0, upper=100.0)

    pred_col = _prediction_value_column(out)
    pred_pct = _group_rank_percentile(
        _to_numeric_series(out, pred_col),
        out["talent_position_family"],
    )
    history_score = (_to_numeric_series(out, "history_strength_score").fillna(50.0) / 100.0).clip(lower=0.0, upper=1.0)
    current_family_mix = (_weighted_family_series(out, weights_by_family=CURRENT_LEVEL_FAMILY_WEIGHTS) / 100.0).clip(lower=0.0, upper=1.0)
    out["current_level_score"] = (
        (0.60 * pred_pct) + (0.30 * current_family_mix) + (0.10 * history_score)
    ).clip(lower=0.0, upper=1.0) * 100.0

    if "future_potential_score" in out.columns:
        out["future_potential_score"] = _to_numeric_series(out, "future_potential_score").fillna(
            _to_numeric_series(out, "future_scout_blend_score").fillna(0.5) * 100.0
        )
    else:
        future_mix = (_weighted_family_series(out, weights_by_family=FUTURE_POTENTIAL_FAMILY_WEIGHTS) / 100.0).clip(lower=0.0, upper=1.0)
        future_prob = _to_numeric_series(out, "future_growth_probability").fillna(0.5).clip(lower=0.0, upper=1.0)
        future_blend = _to_numeric_series(out, "future_scout_blend_score").fillna(future_prob)
        out["future_potential_score"] = (
            0.55 * future_blend.clip(lower=0.0, upper=1.0)
            + 0.30 * future_mix
            + 0.15 * history_score
        ).clip(lower=0.0, upper=1.0) * 100.0
    out["future_potential_score"] = _to_numeric_series(out, "future_potential_score").clip(lower=0.0, upper=100.0)

    minutes_sample = (_minutes_series(out).fillna(0.0) / 1800.0).clip(lower=0.15, upper=1.0)
    provider_quality = _provider_presence_score(out)
    out["_talent_provider_presence_score"] = provider_quality
    history_cov = (_to_numeric_series(out, "history_strength_coverage").fillna(0.35)).clip(lower=0.0, upper=1.0)
    trajectory_cov = (_to_numeric_series(out, "talent_trajectory_coverage").fillna(history_cov * 100.0) / 100.0).clip(lower=0.0, upper=1.0)

    pred = _to_numeric_series(out, "fair_value_eur").fillna(_to_numeric_series(out, "expected_value_eur"))
    low = _to_numeric_series(out, "expected_value_low_eur")
    high = _to_numeric_series(out, "expected_value_high_eur")
    width_ratio = ((high - low).clip(lower=0.0) / pred.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    confidence_signal = _to_numeric_series(out, "undervaluation_confidence")
    prior_ratio = (_to_numeric_series(out, "prior_mae_eur") / pred.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    base_model_conf = pd.Series(0.5, index=out.index, dtype=float)
    base_model_conf = base_model_conf + confidence_signal.clip(lower=0.0, upper=2.0).fillna(0.0) / 4.0
    base_model_conf = base_model_conf - width_ratio.clip(lower=0.0, upper=2.0).fillna(0.0) / 2.6
    base_model_conf = base_model_conf - prior_ratio.clip(lower=0.0, upper=1.0).fillna(0.0) / 3.0
    base_model_conf = base_model_conf.clip(lower=0.0, upper=1.0)

    out["current_level_confidence"] = (
        0.50 * base_model_conf
        + 0.20 * minutes_sample
        + 0.15 * provider_quality
        + 0.15 * history_cov
    ).clip(lower=0.0, upper=1.0) * 100.0

    if "future_potential_confidence" in out.columns:
        out["future_potential_confidence"] = _to_numeric_series(out, "future_potential_confidence").fillna(0.0)
    else:
        future_prob = _to_numeric_series(out, "future_growth_probability").fillna(0.5).clip(lower=0.0, upper=1.0)
        label_support = _to_numeric_series(out, "has_next_season_target").fillna(0.0).clip(lower=0.0, upper=1.0)
        model_support = (0.55 * future_prob) + (0.45 * np.where(label_support > 0, 1.0, 0.55))
        out["future_model_support_score"] = model_support * 100.0
        out["future_potential_confidence"] = (
            0.35 * minutes_sample
            + 0.25 * provider_quality
            + 0.20 * ((0.6 * history_cov) + (0.4 * trajectory_cov))
            + 0.20 * model_support
        ).clip(lower=0.0, upper=1.0) * 100.0

    out["current_level_confidence"] = _to_numeric_series(out, "current_level_confidence").clip(lower=0.0, upper=100.0)
    out["future_potential_confidence"] = _to_numeric_series(out, "future_potential_confidence").clip(lower=0.0, upper=100.0)

    score_family_payloads: list[dict[str, float]] = []
    score_explanations: list[dict[str, list[dict[str, Any]]]] = []
    current_conf_reasons: list[list[str]] = []
    future_conf_reasons: list[list[str]] = []
    for _, row in out.iterrows():
        score_family_payloads.append(_talent_family_payload(row))
        score_explanations.append(
            {
                "current_level": _family_driver_entries(row, lane="current_level"),
                "future_potential": _family_driver_entries(row, lane="future_potential"),
            }
        )
        current_conf_reasons.append(_current_confidence_reasons(row))
        future_conf_reasons.append(_future_confidence_reasons(row))
    out["score_families"] = score_family_payloads
    out["score_explanations"] = score_explanations
    out["current_level_confidence_reasons"] = current_conf_reasons
    out["future_potential_confidence_reasons"] = future_conf_reasons
    return out


def _prepare_predictions_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = _apply_residual_calibration(frame)
    out = _attach_league_trust_tier(out)
    out = _apply_league_value_adjustment(out)
    out = _attach_discovery_reliability(out)
    out = _build_capped_gap_columns(out)
    out = add_history_strength_features(out)
    out = _attach_talent_semantics(out)
    out = _replace_inf_with_nan(out)
    return out


def _discovery_penalty_reason(trust_tier: str, bucket: str) -> str:
    trust_token = str(trust_tier or "unknown").strip().lower()
    bucket_token = str(bucket or "unknown").strip().lower()
    parts: list[str] = []
    if bucket_token in {"weak", "failed", "severe_failed"}:
        label = {
            "weak": "weak league holdout reliability",
            "failed": "failed league holdout reliability",
            "severe_failed": "severely failed league holdout reliability",
        }.get(bucket_token, "league reliability risk")
        parts.append(label)
    elif bucket_token == "unknown":
        parts.append("unknown league holdout reliability")
    if trust_token in {"watch", "unknown", "blocked"}:
        parts.append(f"{trust_token} trust tier")
    return " | ".join(parts) if parts else "standard discovery reliability"


def _attach_discovery_reliability(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["discovery_reliability_weight"] = pd.Series(dtype=float)
        out["discovery_trust_weight"] = pd.Series(dtype=float)
        out["discovery_bucket_weight"] = pd.Series(dtype=float)
        out["discovery_penalty_reason"] = pd.Series(dtype=object)
        return out
    trust_tier = out.get("league_trust_tier", pd.Series("unknown", index=out.index, dtype=object)).astype(str).str.strip().str.lower()
    bucket = out.get("league_adjustment_bucket", pd.Series("unknown", index=out.index, dtype=object)).astype(str).str.strip().str.lower()
    trust_weight = trust_tier.map(DISCOVERY_TRUST_WEIGHTS).fillna(DISCOVERY_TRUST_WEIGHTS["unknown"])
    bucket_weight = bucket.map(DISCOVERY_BUCKET_WEIGHTS).fillna(DISCOVERY_BUCKET_WEIGHTS["unknown"])
    out["discovery_trust_weight"] = trust_weight.astype(float)
    out["discovery_bucket_weight"] = bucket_weight.astype(float)
    out["discovery_reliability_weight"] = (trust_weight * bucket_weight).clip(lower=0.0, upper=1.0)
    out["discovery_penalty_reason"] = [
        _discovery_penalty_reason(tier, bucket_token)
        for tier, bucket_token in zip(trust_tier.tolist(), bucket.tolist())
    ]
    return out


def _weighted_discovery_series(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame.get(column), errors="coerce")
    if column not in DISCOVERY_WEIGHTED_SORT_COLUMNS:
        return numeric
    weight = pd.to_numeric(frame.get("discovery_reliability_weight"), errors="coerce").fillna(1.0)
    return numeric * weight


def _adjusted_score_column(frame: pd.DataFrame, score_col: str, *, target_col: str) -> str:
    if score_col not in frame.columns:
        return score_col
    numeric = pd.to_numeric(frame[score_col], errors="coerce")
    weight = pd.to_numeric(frame.get("discovery_reliability_weight"), errors="coerce").fillna(1.0)
    frame[target_col] = numeric * weight
    return target_col


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
    empty_payload = {
        "available": False,
        "label_column": None,
        "n_labeled_rows": 0,
        "rows": [],
    }
    if score_col not in frame.columns:
        return {**empty_payload, "reason": f"missing_score_col:{score_col}"}

    labels, label_col = _infer_future_outcome_label(frame)
    if labels is None or label_col is None:
        return {**empty_payload, "reason": "missing_future_outcome_label"}

    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_label"] = pd.to_numeric(labels, errors="coerce")
    work = work[np.isfinite(work["_score"]) & np.isfinite(work["_label"])].copy()
    if work.empty:
        return {**empty_payload, "reason": "no_rows_with_labels", "label_column": label_col}

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


def _preferred_scout_score_col(frame: pd.DataFrame, *, default: str) -> str:
    for col in ("future_scout_blend_score", "future_growth_probability", default):
        if col in frame.columns:
            return col
    return default


def _ranking_basis_for_score_col(score_col: str, *, default_basis: str) -> str:
    if score_col == "future_scout_blend_score":
        return "future_target_tuned_blend"
    if score_col == "future_growth_probability":
        return "future_target_probability"
    return default_basis


def query_predictions(
    split: Split = "test",
    season: str | None = None,
    league: str | None = None,
    club: str | None = None,
    position: str | None = None,
    role_keys: Sequence[str] | None = None,
    min_minutes: float | None = None,
    min_age: float | None = None,
    max_age: float | None = None,
    max_market_value_eur: float | None = None,
    max_contract_years_left: float | None = None,
    non_big5_only: bool = False,
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

    if role_keys:
        wanted_roles = {str(role).strip().upper() for role in role_keys if str(role).strip()}
        if wanted_roles:
            inferred_roles = frame.apply(_infer_position_role_key, axis=1)
            frame = frame[inferred_roles.isin(wanted_roles)].copy()

    if min_minutes is not None:
        frame = frame[_minutes_series(frame).fillna(0) >= float(min_minutes)].copy()
    if (min_age is not None or max_age is not None) and "age" in frame.columns:
        age = pd.to_numeric(frame["age"], errors="coerce")
        if min_age is not None:
            frame = frame[age >= float(min_age)].copy()
            age = pd.to_numeric(frame["age"], errors="coerce")
        if max_age is not None:
            frame = frame[age <= float(max_age)].copy()

    if max_market_value_eur is not None and "market_value_eur" in frame.columns:
        market = pd.to_numeric(frame["market_value_eur"], errors="coerce")
        frame = frame[market <= float(max_market_value_eur)].copy()

    if max_contract_years_left is not None and "contract_years_left" in frame.columns:
        contract_years = pd.to_numeric(frame["contract_years_left"], errors="coerce")
        frame = frame[contract_years.notna() & (contract_years <= float(max_contract_years_left))].copy()

    if non_big5_only and "league" in frame.columns:
        league_norm = frame["league"].astype(str).str.strip().str.casefold()
        frame = frame[~league_norm.isin(BIG5_LEAGUES)].copy()

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
    sort_proxy_col = sort_by
    if sort_by in DISCOVERY_WEIGHTED_SORT_COLUMNS:
        sort_proxy_col = "_discovery_sort_value"
        frame[sort_proxy_col] = _weighted_discovery_series(frame, sort_by)
    frame = frame.sort_values(sort_proxy_col, ascending=ascending, na_position="last")

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
    min_age: float | None = None,
    max_age: float | None = 25,
    positions: Sequence[str] | None = None,
    role_keys: Sequence[str] | None = None,
    non_big5_only: bool = False,
    max_market_value_eur: float | None = None,
    max_contract_years_left: float | None = None,
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

    if min_age is not None:
        work = work[work["age_num"].fillna(-1) >= float(min_age)].copy()
    if max_age is not None:
        work = work[work["age_num"].fillna(999) <= float(max_age)].copy()

    if positions:
        pos_set = {p.upper() for p in positions}
        work = work[work["position_used"].isin(pos_set)].copy()

    if role_keys:
        wanted_roles = {str(role).strip().upper() for role in role_keys if str(role).strip()}
        if wanted_roles:
            inferred_roles = work.apply(_infer_position_role_key, axis=1)
            work = work[inferred_roles.isin(wanted_roles)].copy()

    if non_big5_only:
        league_norm = (
            work["league"].astype(str).str.strip().str.casefold()
            if "league" in work.columns
            else pd.Series("", index=work.index, dtype=str)
        )
        work = work[~league_norm.isin(BIG5_LEAGUES)].copy()

    if max_market_value_eur is not None:
        work = work[work["market_value_eur"].fillna(np.inf) <= float(max_market_value_eur)].copy()

    if max_contract_years_left is not None and "contract_years_left" in work.columns:
        contract_years = pd.to_numeric(work["contract_years_left"], errors="coerce")
        work = work[contract_years.notna() & (contract_years <= float(max_contract_years_left))].copy()

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
    base_score_col = _preferred_scout_score_col(work, default="shortlist_score")
    score_col = _adjusted_score_column(work, base_score_col, target_col="shortlist_score_adjusted")
    work = work.sort_values(
        [score_col, "ranking_gap_eur", "value_gap_conservative_eur", "undervaluation_confidence"],
        ascending=False,
    )
    shortlist = work.head(max(int(top_n), 0)).copy()
    precision = _precision_at_k(
        work,
        score_col=score_col,
        k_values=(10, 25, 50, int(top_n)),
    )
    return {
        "split": split,
        "total_candidates": int(len(work)),
        "count": int(len(shortlist)),
        "diagnostics": {
            "ranking_basis": _ranking_basis_for_score_col(
                base_score_col,
                default_basis="guardrailed_gap_confidence_history",
            ),
            "score_column": base_score_col,
            "base_score_column": base_score_col,
            "ranking_score_column": score_col,
            "precision_at_k": precision,
        },
        "items": _to_records(shortlist),
    }


def query_scout_targets(
    split: Split = "test",
    top_n: int = 100,
    min_minutes: float = 900,
    min_age: float | None = None,
    max_age: float | None = 23,
    min_confidence: float = 0.50,
    min_value_gap_eur: float = 1_000_000.0,
    positions: Sequence[str] | None = None,
    role_keys: Sequence[str] | None = None,
    non_big5_only: bool = True,
    include_leagues: Sequence[str] | None = None,
    exclude_leagues: Sequence[str] | None = None,
    min_expected_value_eur: float | None = None,
    max_expected_value_eur: float | None = None,
    max_market_value_eur: float | None = None,
    max_contract_years_left: float | None = None,
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

    if min_age is not None:
        work = work[work["age_num"].fillna(-1.0) >= float(min_age)].copy()

    if max_age is not None:
        work = work[work["age_num"].fillna(999.0) <= float(max_age)].copy()

    if positions:
        wanted = {p.strip().upper() for p in positions if str(p).strip()}
        work = work[work["position_used"].isin(wanted)].copy()

    if role_keys:
        wanted_roles = {str(role).strip().upper() for role in role_keys if str(role).strip()}
        if wanted_roles:
            inferred_roles = work.apply(_infer_position_role_key, axis=1)
            work = work[inferred_roles.isin(wanted_roles)].copy()

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

    if max_market_value_eur is not None:
        work = work[work["market_value_eur"].fillna(np.inf) <= float(max_market_value_eur)].copy()

    if max_contract_years_left is not None and "contract_years_left" in work.columns:
        contract_years = pd.to_numeric(work["contract_years_left"], errors="coerce")
        work = work[contract_years.notna() & (contract_years <= float(max_contract_years_left))].copy()

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
    base_score_col = _preferred_scout_score_col(work, default="scout_target_score")
    score_col = _adjusted_score_column(work, base_score_col, target_col="scout_target_score_adjusted")

    work = work.sort_values(
        [score_col, "ranking_gap_eur", "value_gap_conservative_eur", "undervaluation_confidence"],
        ascending=False,
    )
    out = work.head(max(int(top_n), 0)).copy()
    precision = _precision_at_k(
        work,
        score_col=score_col,
        k_values=(10, 25, 50, int(top_n)),
    )
    return {
        "split": split,
        "total_candidates": int(len(work)),
        "count": int(len(out)),
        "diagnostics": {
            "ranking_basis": _ranking_basis_for_score_col(
                base_score_col,
                default_basis="guardrailed_gap_confidence_history_efficiency",
            ),
            "score_column": base_score_col,
            "base_score_column": base_score_col,
            "ranking_score_column": score_col,
            "precision_at_k": precision,
        },
        "items": _to_records(out),
    }


def get_system_fit_templates() -> dict[str, Any]:
    return list_system_fit_templates()


def _season_sort_key(value: Any) -> tuple[int, str]:
    token = str(value or "").strip()
    if not token:
        return (-1, "")
    head = token.split("/")[0].strip()
    try:
        return (int(head), token)
    except ValueError:
        return (-1, token)


def _sequence_text_values(values: Sequence[Any] | None) -> list[str]:
    return [str(value).strip() for value in values or [] if str(value).strip()]


def _league_trust_maps() -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    try:
        payload = _get_ingestion_health_payload()
    except Exception:
        return {}, {}
    rows = list(payload.get("rows") or [])
    exact: dict[tuple[str, str], str] = {}
    latest: dict[str, tuple[tuple[int, str], str]] = {}
    status_to_tier = {
        "healthy": "trusted",
        "watch": "watch",
        "blocked": "blocked",
    }
    for row in rows:
        league_name = _safe_text(row.get("league_name"))
        season = _safe_text(row.get("season"))
        if not league_name or not season:
            continue
        tier = status_to_tier.get(str(row.get("status") or "").strip().lower(), "unknown")
        league_key = league_name.casefold()
        season_key = season.casefold()
        exact[(league_key, season_key)] = tier
        current = latest.get(league_key)
        season_sort = _season_sort_key(season)
        if current is None or season_sort > current[0]:
            latest[league_key] = (season_sort, tier)
    return exact, {league_key: payload[1] for league_key, payload in latest.items()}


def _attach_league_trust_tier(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["league_trust_tier"] = pd.Series(dtype=object)
        return out

    exact, latest = _league_trust_maps()
    league_key = out.get("league", pd.Series("", index=out.index, dtype=object)).astype(str).str.strip().str.casefold()
    season_key = out.get("season", pd.Series("", index=out.index, dtype=object)).astype(str).str.strip().str.casefold()
    exact_key = pd.Series(list(zip(league_key.tolist(), season_key.tolist())), index=out.index, dtype=object)
    exact_tier = exact_key.map(exact)
    latest_tier = league_key.map(latest)
    out["league_trust_tier"] = exact_tier.fillna(latest_tier).fillna("unknown")
    return out


def _apply_system_fit_filters(
    frame: pd.DataFrame,
    *,
    season: str | None = None,
    include_leagues: Sequence[str] | None = None,
    exclude_leagues: Sequence[str] | None = None,
    min_age: float | None = None,
    max_age: float | None = None,
    min_minutes: float | None = None,
    max_market_value_eur: float | None = None,
    max_contract_years_left: float | None = None,
    non_big5_only: bool = False,
) -> pd.DataFrame:
    work = frame.copy()
    if season and "season" in work.columns:
        work = work[work["season"].astype(str) == str(season)].copy()

    if include_leagues and "league" in work.columns:
        include_norm = {token.casefold() for token in _sequence_text_values(include_leagues)}
        if include_norm:
            league_norm = work["league"].astype(str).str.strip().str.casefold()
            work = work[league_norm.isin(include_norm)].copy()

    if exclude_leagues and "league" in work.columns:
        exclude_norm = {token.casefold() for token in _sequence_text_values(exclude_leagues)}
        if exclude_norm:
            league_norm = work["league"].astype(str).str.strip().str.casefold()
            work = work[~league_norm.isin(exclude_norm)].copy()

    if (min_age is not None or max_age is not None) and "age" in work.columns:
        age = pd.to_numeric(work["age"], errors="coerce")
        if min_age is not None:
            work = work[age >= float(min_age)].copy()
            age = pd.to_numeric(work["age"], errors="coerce")
        if max_age is not None:
            work = work[age <= float(max_age)].copy()

    if min_minutes is not None:
        work = work[_minutes_series(work).fillna(0.0) >= float(min_minutes)].copy()

    if max_market_value_eur is not None and "market_value_eur" in work.columns:
        market = pd.to_numeric(work["market_value_eur"], errors="coerce")
        work = work[market <= float(max_market_value_eur)].copy()

    if max_contract_years_left is not None and "contract_years_left" in work.columns:
        contract_years = pd.to_numeric(work["contract_years_left"], errors="coerce")
        work = work[contract_years.notna() & (contract_years <= float(max_contract_years_left))].copy()

    if non_big5_only and "league" in work.columns:
        league_norm = work["league"].astype(str).str.strip().str.casefold()
        work = work[~league_norm.isin(BIG5_LEAGUES)].copy()
    return work


def _apply_trust_scope(frame: pd.DataFrame, trust_scope: str) -> pd.DataFrame:
    allowed = TRUST_SCOPE_ALLOWED.get(str(trust_scope), TRUST_SCOPE_ALLOWED["trusted_and_watch"])
    if "league_trust_tier" not in frame.columns:
        return frame.copy()
    tiers = frame["league_trust_tier"].astype(str).str.strip().str.lower()
    return frame[tiers.isin(allowed)].copy()


def query_system_fit(
    *,
    split: Split = "test",
    template_key: str,
    active_lane: Literal["valuation", "future_shortlist"] = "valuation",
    top_n_per_slot: int = 10,
    slot_role_overrides: dict[str, str] | None = None,
    filters: dict[str, Any] | None = None,
    trust_scope: Literal["trusted_only", "trusted_and_watch", "all"] = "trusted_and_watch",
) -> dict[str, Any]:
    filter_values = dict(filters or {})
    frame = _prepare_predictions_frame(get_predictions(split=split))
    frame = _apply_system_fit_filters(
        frame,
        season=_safe_text(filter_values.get("season")),
        include_leagues=_sequence_text_values(filter_values.get("include_leagues")),
        exclude_leagues=_sequence_text_values(filter_values.get("exclude_leagues")),
        min_age=_safe_float(filter_values.get("min_age")),
        max_age=_safe_float(filter_values.get("max_age")),
        min_minutes=_safe_float(filter_values.get("min_minutes")),
        max_market_value_eur=_safe_float(filter_values.get("max_market_value_eur")),
        max_contract_years_left=_safe_float(filter_values.get("max_contract_years_left")),
        non_big5_only=bool(filter_values.get("non_big5_only")),
    )
    frame = _apply_trust_scope(frame, trust_scope)

    budget_eur = _safe_float(filter_values.get("budget_eur"))
    min_confidence = _safe_float(filter_values.get("min_confidence"))
    payload = rank_system_fit_slots(
        frame,
        template_key=template_key,
        active_lane=active_lane,
        top_n_per_slot=top_n_per_slot,
        slot_role_overrides=slot_role_overrides,
        budget_eur=budget_eur,
        min_confidence=min_confidence,
    )
    payload.update(
        {
            "split": split,
            "trust_scope": trust_scope,
            "filters_applied": {
                "season": _safe_text(filter_values.get("season")),
                "include_leagues": _sequence_text_values(filter_values.get("include_leagues")),
                "exclude_leagues": _sequence_text_values(filter_values.get("exclude_leagues")),
                "min_age": _safe_float(filter_values.get("min_age")),
                "max_age": _safe_float(filter_values.get("max_age")),
                "min_minutes": _safe_float(filter_values.get("min_minutes")),
                "max_market_value_eur": _safe_float(filter_values.get("max_market_value_eur")),
                "max_contract_years_left": _safe_float(filter_values.get("max_contract_years_left")),
                "min_confidence": min_confidence,
                "non_big5_only": bool(filter_values.get("non_big5_only")),
                "budget_eur": budget_eur,
            },
        }
    )
    return payload


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


def _safe_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _row_uses_future_overlay(row: pd.Series) -> bool:
    for key in ("future_scout_blend_score", "future_growth_probability", "future_scout_score"):
        if _safe_float(row.get(key)) is not None:
            return True
    return False


def _artifact_lane_payload(role: ChampionRole, section: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = section if isinstance(section, dict) else {}
    return {
        "artifact_role": role,
        "lane_state": str(entry.get("lane_state") or lane_state_for_role(role)),
        "promotion_state": str(entry.get("promotion_state") or "advisory_only"),
        "promotion_reasons": list(entry.get("promotion_reasons") or []),
        "artifact_label": _safe_text(entry.get("label")) or role,
        "artifact_generated_at_utc": _safe_text(entry.get("generated_at_utc")),
        "artifact_test_season": _safe_text((entry.get("config") or {}).get("test_season"))
        if isinstance(entry.get("config"), dict)
        else None,
    }


def _provider_signal_available(provider_coverage: dict[str, Any]) -> bool:
    return any(
        bool(provider_coverage.get(key))
        for key in ("statsbomb", "availability_provider", "market_provider", "fixture_provider", "odds_provider")
    )


def _build_data_freshness(
    row: pd.Series,
    *,
    provider_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = provider_coverage or build_provider_coverage(row)
    manifest = get_model_manifest()
    valuation_section = _manifest_role_section(manifest, "valuation") or {}
    future_section = _manifest_role_section(manifest, "future_shortlist") or {}

    row_season = _safe_text(row.get("season"))
    future_test_season = _safe_text((future_section.get("config") or {}).get("test_season"))
    uses_future_overlay = _row_uses_future_overlay(row)
    active_role: ChampionRole = (
        "future_shortlist"
        if uses_future_overlay and row_season is not None and row_season == future_test_season
        else "valuation"
    )
    active_section = future_section if active_role == "future_shortlist" else valuation_section
    lane_payload = _artifact_lane_payload(active_role, active_section if isinstance(active_section, dict) else None)
    latest_snapshot_date = _safe_text(coverage.get("latest_snapshot_date"))
    latest_retrieved_at = _safe_text(coverage.get("latest_retrieved_at"))
    signal_available = _provider_signal_available(coverage)
    freshness_meta_available = bool(latest_snapshot_date or latest_retrieved_at)

    if not signal_available or not freshness_meta_available:
        status = "limited"
    elif active_role == "future_shortlist" and row_season is not None and row_season == future_test_season:
        status = "live"
    else:
        status = "stable"

    if status == "live":
        message = "Live current-season overlay. Fresh performance context is available, but season outcomes are still in progress."
    elif status == "stable":
        message = "Stable valuation artifact. Use for benchmarked pricing and ranking."
    else:
        message = "Freshness is partially known because provider snapshot metadata is incomplete."

    return {
        "status": status,
        "lane_state": "limited" if status == "limited" else str(lane_payload.get("lane_state")),
        "partial_season": status == "live",
        "message": message,
        "artifact_role": active_role,
        "artifact_label": lane_payload.get("artifact_label"),
        "artifact_test_season": lane_payload.get("artifact_test_season"),
        "artifact_generated_at_utc": lane_payload.get("artifact_generated_at_utc"),
        "promotion_state": lane_payload.get("promotion_state"),
        "promotion_reasons": lane_payload.get("promotion_reasons"),
        "row_season": row_season,
        "using_live_lane": bool(active_role == "future_shortlist"),
        "latest_snapshot_date": latest_snapshot_date,
        "latest_retrieved_at": latest_retrieved_at,
    }


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


def _normalize_position_key(raw: Any) -> str:
    if raw is None or pd.isna(raw):
        return "UNK"
    token = str(raw).strip().upper()
    if token in {"GK", "DF", "MF", "FW"}:
        return token
    if token in {"BACK", "DEFENDER"} or "DEF" in token:
        return "DF"
    if token in {"MIDFIELD", "MIDFIELDER"} or "MID" in token:
        return "MF"
    if token in {"ATTACK", "ATTACKER", "FORWARD", "STRIKER"} or "WING" in token:
        return "FW"
    if "GOAL" in token:
        return "GK"
    return token


def _safe_text_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _position_role_label(role_key: str) -> str:
    return ROLE_KEY_LABELS.get(role_key, role_key)


def _infer_position_role_key(row: pd.Series) -> str:
    family_key = _normalize_position_key(row.get("model_position") or row.get("position_group"))
    if family_key == "GK":
        return "GK"

    primary = " ".join(
        part
        for part in (
            _safe_text_token(row.get("position_main")),
            _safe_text_token(row.get("position")),
            _safe_text_token(row.get("position_alt")),
        )
        if part
    ).lower()

    if family_key == "DF":
        if any(token in primary for token in ("wing-back", "wing back", "fullback", "full back", "left back", "right back")):
            return "FB"
        if "defender, left" in primary or "defender, right" in primary:
            return "FB"
        if any(token in primary for token in ("centre", "center", "back")):
            return "CB"
        return "DF"

    if family_key == "MF":
        if "defensive" in primary:
            return "DM"
        if "attacking" in primary:
            return "AM"
        if "left midfield" in primary or "right midfield" in primary or "wing" in primary:
            return "W"
        if "central" in primary:
            return "CM"
        return "MF"

    if family_key == "FW":
        if "winger" in primary:
            return "W"
        if "second striker" in primary or "support forward" in primary:
            return "SS"
        if "attack, centre" in primary or "striker" in primary or "forward" in primary:
            return "ST"
        return "FW"

    return family_key


def _talent_position_family_from_role(role_key: str) -> str:
    token = str(role_key or "").strip().upper()
    if token == "GK":
        return "GK"
    if token in {"FB"}:
        return "FB"
    if token in {"CB", "DF"}:
        return "CB"
    if token in {"AM", "SS"}:
        return "AM"
    if token in {"W"}:
        return "W"
    if token in {"ST", "FW"}:
        return "ST"
    if token in {"DM", "CM", "MF"}:
        return "CM"
    return "CM"


def _infer_talent_position_family(row: pd.Series) -> str:
    existing = _safe_text(row.get("talent_position_family"))
    if existing:
        return existing.upper()
    return _talent_position_family_from_role(_infer_position_role_key(row))


def _group_rank_percentile(values: pd.Series, groups: pd.Series, *, min_group_rows: int = 15) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    group_labels = groups.astype(str).fillna("unknown")
    fallback = numeric.rank(method="average", pct=True)
    grouped = numeric.groupby(group_labels).rank(method="average", pct=True)
    group_sizes = group_labels.map(group_labels.value_counts())
    out = grouped.where(group_sizes >= int(max(min_group_rows, 1)), fallback)
    median = float(fallback.dropna().median()) if fallback.notna().any() else 0.5
    return out.fillna(median).clip(lower=0.0, upper=1.0)


def _provider_presence_score(frame: pd.DataFrame) -> pd.Series:
    groups: list[pd.Series] = []
    for prefixes in (("sb_",), ("avail_",), ("fixture_",), ("odds_",), ("sofa_",)):
        cols = [col for col in frame.columns if any(str(col).startswith(prefix) for prefix in prefixes)]
        if cols:
            groups.append(frame[cols].notna().any(axis=1).astype(float))
    if not groups:
        return pd.Series(0.35, index=frame.index, dtype=float)
    return pd.concat(groups, axis=1).mean(axis=1).clip(lower=0.0, upper=1.0)


def _weighted_family_series(frame: pd.DataFrame, *, weights_by_family: dict[str, dict[str, float]]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for family, weights in weights_by_family.items():
        mask = frame["talent_position_family"].astype(str) == family
        if not mask.any():
            continue
        score = pd.Series(0.0, index=frame.index[mask], dtype=float)
        for metric_family, weight in weights.items():
            col = f"talent_{metric_family}_score"
            score = score + _to_numeric_series(frame.loc[mask], col).fillna(50.0) * float(weight)
        out.loc[mask] = score
    return out.fillna(50.0).clip(lower=0.0, upper=100.0)


def _talent_family_payload(row: pd.Series) -> dict[str, float]:
    payload: dict[str, float] = {}
    for family in TALENT_FAMILIES:
        value = _safe_float(row.get(f"talent_{family}_score"))
        payload[family] = value if value is not None else 50.0
    return payload


def _family_driver_entries(
    row: pd.Series,
    *,
    lane: Literal["current_level", "future_potential"],
) -> list[dict[str, Any]]:
    position_family = _infer_talent_position_family(row)
    weights = (
        CURRENT_LEVEL_FAMILY_WEIGHTS.get(position_family, CURRENT_LEVEL_FAMILY_WEIGHTS["CM"])
        if lane == "current_level"
        else FUTURE_POTENTIAL_FAMILY_WEIGHTS.get(position_family, FUTURE_POTENTIAL_FAMILY_WEIGHTS["CM"])
    )
    entries: list[dict[str, Any]] = []
    for family in TALENT_FAMILIES:
        score = _safe_float(row.get(f"talent_{family}_score")) or 50.0
        contribution = score * float(weights.get(family, 0.0))
        if score >= 67:
            tone = "strong"
        elif score >= 45:
            tone = "medium"
        else:
            tone = "watch"
        message = (
            f"{TALENT_FAMILY_LABELS[family]} is {score:.0f}/100 for the {position_family} profile "
            f"and carries {float(weights.get(family, 0.0)) * 100:.0f}% of the {lane.replace('_', ' ')} mix."
        )
        entries.append(
            {
                "family": family,
                "label": TALENT_FAMILY_LABELS[family],
                "score": round(score, 2),
                "weight": round(float(weights.get(family, 0.0)), 4),
                "contribution": round(contribution, 2),
                "tone": tone,
                "message": message,
            }
        )
    entries.sort(key=lambda item: item["contribution"], reverse=True)
    return entries[:3]


def _talent_confidence_label(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 70.0:
        return "high"
    if score >= 45.0:
        return "medium"
    return "low"


def _current_confidence_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    minutes = _safe_float(row.get("minutes")) or _safe_float(row.get("sofa_minutesPlayed")) or 0.0
    if minutes < 900:
        reasons.append("Minutes sample is still thin for stable current-level pricing.")
    history_cov = _safe_float(row.get("history_strength_coverage")) or 0.0
    if history_cov < 0.35:
        reasons.append("Trajectory/history coverage is sparse, so the pricing lane leans more on the current season.")
    provider = _safe_float(row.get("_talent_provider_presence_score")) or 0.0
    if provider < 0.5:
        reasons.append("Provider enrichment is partial, which reduces current-level confidence.")
    if not reasons:
        reasons.append("Minutes, provider coverage, and history support are adequate for pricing use.")
    return reasons[:3]


def _future_confidence_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    minutes = _safe_float(row.get("minutes")) or _safe_float(row.get("sofa_minutesPlayed")) or 0.0
    if minutes < 1200:
        reasons.append("Future-potential confidence is lighter because the current sample is still modest.")
    if (_safe_float(row.get("history_strength_coverage")) or 0.0) < 0.35:
        reasons.append("Trajectory support is limited, so the future lane is leaning more on one-season evidence.")
    if (_safe_float(row.get("future_model_support_score")) or 0.0) < 60.0:
        reasons.append("Model support is limited because next-season label support is incomplete for this row.")
    if (_safe_float(row.get("_talent_provider_presence_score")) or 0.0) < 0.5:
        reasons.append("Provider coverage is partial, so future potential should stay advisory.")
    if not reasons:
        reasons.append("Minutes, label support, and provider context are strong enough for an advisory future view.")
    return reasons[:3]


def _metric_snapshot(
    *,
    row: pd.Series,
    cohort: pd.DataFrame,
    metric: str,
    direction: int | None = None,
) -> dict[str, Any] | None:
    if metric not in row.index or metric not in cohort.columns:
        return None
    player_value = _safe_float(row.get(metric))
    if player_value is None:
        return None

    series = pd.to_numeric(cohort[metric], errors="coerce").dropna()
    if len(series) < 20:
        return None

    metric_direction = int(direction or ADVANCED_METRIC_DIRECTION.get(metric, 1))
    percentile_raw = float((series <= player_value).mean())
    quality_percentile = percentile_raw if metric_direction > 0 else (1.0 - percentile_raw)
    quality_percentile = float(np.clip(quality_percentile, 0.0, 1.0))

    q25, q50, q75 = np.nanquantile(series.to_numpy(), [0.25, 0.5, 0.75])
    return {
        "metric": metric,
        "label": ADVANCED_METRIC_LABEL.get(metric, metric),
        "direction": "higher_is_better" if metric_direction > 0 else "lower_is_better",
        "player_value": player_value,
        "cohort_p25": float(q25),
        "cohort_median": float(q50),
        "cohort_p75": float(q75),
        "percentile_raw": float(percentile_raw),
        "quality_percentile": quality_percentile,
    }


def _build_metric_snapshot_map(row: pd.Series, cohort: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for metric, _, direction in ADVANCED_METRIC_SPECS:
        snap = _metric_snapshot(row=row, cohort=cohort, metric=metric, direction=direction)
        if snap is not None:
            out[metric] = snap
    return out


def _score_template_fit(
    *,
    metric_map: dict[str, dict[str, Any]],
    targets: dict[str, float],
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for metric, target_pct in targets.items():
        snap = metric_map.get(metric)
        if snap is None:
            continue
        observed = float(snap["quality_percentile"])
        target = float(np.clip(target_pct, 0.0, 1.0))
        # Soft match around target percentile. 1.0 is perfect, 0.0 is very far.
        fit_score = max(0.0, 1.0 - abs(observed - target) / 0.75)
        parts.append(
            {
                "metric": metric,
                "label": snap["label"],
                "target_percentile": target,
                "observed_percentile": observed,
                "fit_score": float(fit_score),
            }
        )

    n_targets = max(len(targets), 1)
    coverage = len(parts) / n_targets
    if not parts:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "parts": [],
        }
    mean_fit = float(np.mean([p["fit_score"] for p in parts]))
    score = mean_fit * (0.65 + 0.35 * coverage)
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "coverage": float(np.clip(coverage, 0.0, 1.0)),
        "parts": parts,
    }


def _build_player_type_profile(
    *,
    row: pd.Series,
    metric_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    role_key = _infer_position_role_key(row)
    family_key = _normalize_position_key(row.get("model_position") or row.get("position_group"))
    templates = ARCHETYPE_TEMPLATES.get(role_key) or ARCHETYPE_TEMPLATES.get(family_key, ())
    if not templates:
        return {
            "position_key": role_key,
            "position_family_key": family_key,
            "position_label": _position_role_label(role_key),
            "archetype": "Unknown",
            "confidence_0_to_1": 0.0,
            "tier": "low",
            "runner_up": None,
            "candidates": [],
            "summary_text": "Not enough archetype templates for this position.",
        }

    scored: list[dict[str, Any]] = []
    for template in templates:
        fit = _score_template_fit(metric_map=metric_map, targets=template.get("targets", {}))
        scored.append(
            {
                "name": str(template.get("name") or "Unknown"),
                "description": str(template.get("description") or ""),
                "score_0_to_1": fit["score"],
                "coverage_0_to_1": fit["coverage"],
                "matched_metrics": fit["parts"],
            }
        )

    scored.sort(key=lambda x: (x["score_0_to_1"], x["coverage_0_to_1"]), reverse=True)
    best = scored[0]
    runner = scored[1] if len(scored) > 1 else None
    conf = float(best["score_0_to_1"])
    if runner is not None:
        conf = float(np.clip(conf - 0.15 * max(0.0, runner["score_0_to_1"]), 0.0, 1.0))

    tier = "low"
    if conf >= 0.72:
        tier = "high"
    elif conf >= 0.52:
        tier = "medium"

    runner_name = runner["name"] if isinstance(runner, dict) else None
    summary = f"Role lens: {_position_role_label(role_key)}. Archetype: {best['name']} ({tier} confidence)."
    if runner_name:
        summary += f" Runner-up: {runner_name}."

    return {
        "position_key": role_key,
        "position_family_key": family_key,
        "position_label": _position_role_label(role_key),
        "archetype": best["name"],
        "description": best["description"],
        "confidence_0_to_1": conf,
        "tier": tier,
        "runner_up": runner_name,
        "candidates": scored[:3],
        "summary_text": summary,
    }


def _build_formation_fit_profile(
    *,
    row: pd.Series,
    metric_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    role_key = _infer_position_role_key(row)
    family_key = _normalize_position_key(row.get("model_position") or row.get("position_group"))
    templates = FORMATION_FIT_TEMPLATES.get(role_key) or FORMATION_FIT_TEMPLATES.get(family_key, ())
    if not templates:
        return {
            "position_key": role_key,
            "position_family_key": family_key,
            "position_label": _position_role_label(role_key),
            "recommended": [],
            "summary_text": "No formation templates available for this position.",
        }

    fits: list[dict[str, Any]] = []
    for template in templates:
        fit = _score_template_fit(metric_map=metric_map, targets=template.get("targets", {}))
        score = float(fit["score"])
        tier = "low"
        if score >= 0.75:
            tier = "high"
        elif score >= 0.55:
            tier = "medium"
        fits.append(
            {
                "formation": str(template.get("formation") or ""),
                "role": str(template.get("role") or ""),
                "fit_score_0_to_1": score,
                "fit_tier": tier,
                "coverage_0_to_1": float(fit["coverage"]),
                "matched_metrics": fit["parts"],
            }
        )

    fits.sort(key=lambda x: (x["fit_score_0_to_1"], x["coverage_0_to_1"]), reverse=True)
    recommended = fits[:3]
    if recommended:
        top = recommended[0]
        summary = (
            f"Best tactical fit for {_position_role_label(role_key)}: {top['formation']} as {top['role']} "
            f"({top['fit_tier']} fit)."
        )
    else:
        summary = "No formation fit could be estimated from available metrics."
    return {
        "position_key": role_key,
        "position_family_key": family_key,
        "position_label": _position_role_label(role_key),
        "recommended": recommended,
        "summary_text": summary,
    }


def _build_radar_profile(
    *,
    row: pd.Series,
    metric_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    role_key = _infer_position_role_key(row)
    family_key = _normalize_position_key(row.get("model_position") or row.get("position_group"))
    axes_metrics = RADAR_AXES_BY_POSITION.get(role_key) or RADAR_AXES_BY_POSITION.get(family_key)
    if axes_metrics is None:
        axes_metrics = tuple(metric for metric, _, _ in PROFILE_METRIC_SPECS[:6])

    axes: list[dict[str, Any]] = []
    for metric in axes_metrics:
        snap = metric_map.get(metric)
        if snap is None:
            axes.append(
                {
                    "metric": metric,
                    "label": ADVANCED_METRIC_LABEL.get(metric, metric),
                    "available": False,
                    "normalized_0_to_100": None,
                    "quality_percentile": None,
                    "player_value": None,
                    "cohort_median": None,
                }
            )
            continue
        quality = float(snap["quality_percentile"])
        axes.append(
            {
                "metric": metric,
                "label": snap["label"],
                "available": True,
                "normalized_0_to_100": float(np.clip(quality * 100.0, 0.0, 100.0)),
                "quality_percentile": quality,
                "player_value": snap["player_value"],
                "cohort_median": snap["cohort_median"],
            }
        )

    available = [a for a in axes if a["available"]]
    coverage = float(len(available) / max(len(axes), 1))
    return {
        "position_key": role_key,
        "position_family_key": family_key,
        "position_label": _position_role_label(role_key),
        "ready_for_plot": coverage >= 0.50,
        "coverage_0_to_1": coverage,
        "axes": axes,
    }


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
    raw_fair = _safe_float(row.get("raw_fair_value_eur"))
    market = _safe_float(row.get("market_value_eur"))

    raw_gap = _safe_float(row.get("value_gap_raw_eur"))
    if raw_gap is None:
        raw_gap = _safe_float(row.get("value_gap_eur"))
    cons_gap = _safe_float(row.get("value_gap_conservative_eur"))
    raw_cons_gap = _safe_float(row.get("raw_value_gap_conservative_eur"))
    if raw_gap is None and pred is not None and market is not None:
        raw_gap = pred - market
    if cons_gap is None:
        cons_gap = raw_gap
    if raw_cons_gap is None:
        raw_cons_gap = raw_gap

    league_adjusted_gap = _safe_float(row.get("league_adjusted_gap_eur"))
    league_adjusted_fair = _safe_float(row.get("league_adjusted_fair_value_eur"))
    league_adjustment_alpha = _safe_float(row.get("league_adjustment_alpha"))
    league_adjustment_bucket = _safe_text(row.get("league_adjustment_bucket"))
    league_adjustment_reason = _safe_text(row.get("league_adjustment_reason"))
    league_holdout_r2 = _safe_float(row.get("league_holdout_r2"))
    league_holdout_wmape = _safe_float(row.get("league_holdout_wmape"))
    league_holdout_interval_coverage = _safe_float(row.get("league_holdout_interval_coverage"))

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
        "raw_fair_value_eur": raw_fair,
        "league_adjusted_fair_value_eur": league_adjusted_fair,
        "value_gap_raw_eur": raw_gap,
        "raw_value_gap_conservative_eur": raw_cons_gap,
        "value_gap_conservative_eur": cons_gap,
        "league_adjusted_gap_eur": league_adjusted_gap,
        "value_gap_capped_eur": capped_gap,
        "cap_threshold_eur": cap_threshold,
        "cap_applied": cap_applied,
        "cap_ratio": cap_ratio,
        "prior_mae_eur": prior_mae,
        "prior_p75ae_eur": prior_p75ae,
        "prior_qae_eur": prior_qae,
        "league_adjustment_alpha": league_adjustment_alpha,
        "league_adjustment_bucket": league_adjustment_bucket,
        "league_adjustment_reason": league_adjustment_reason,
        "league_holdout_r2": league_holdout_r2,
        "league_holdout_wmape": league_holdout_wmape,
        "league_holdout_interval_coverage": league_holdout_interval_coverage,
    }


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
    market = format_eur(valuation_guardrails.get("market_value_eur"))
    fair = format_eur(valuation_guardrails.get("fair_value_eur"))
    raw_fair = format_eur(valuation_guardrails.get("raw_fair_value_eur"))
    gap = format_eur(valuation_guardrails.get("value_gap_conservative_eur"))
    capped = format_eur(valuation_guardrails.get("value_gap_capped_eur"))
    bucket = str(valuation_guardrails.get("league_adjustment_bucket") or "").strip().lower()
    adjustment_reason = _safe_text(valuation_guardrails.get("league_adjustment_reason"))

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
        (
            f"{name}: {conf_label}-confidence undervaluation signal "
            f"(market {market}, league-adjusted fair value {fair}, raw model fair value {raw_fair})."
            if bucket in {"weak", "failed", "severe_failed"} and raw_fair != fair
            else f"{name}: {conf_label}-confidence undervaluation signal (market {market}, fair value {fair})."
        ),
        f"Conservative value gap is {gap}; guardrailed gap is {capped}.",
        f"Top strengths: {top_strengths}. Development focus: {top_levers}.",
    ]
    if adjustment_reason and bucket in {"weak", "failed", "severe_failed"}:
        sentences.append(adjustment_reason)
    if history_note:
        sentences.append(history_note)
    if risk_flags:
        top_risks = ", ".join(flag["code"] for flag in risk_flags[:2])
        sentences.append(f"Key risk flags: {top_risks}.")
    else:
        sentences.append("No major risk flag triggered by current thresholds.")
    return " ".join(sentences)


def _build_talent_view(row: pd.Series) -> dict[str, Any]:
    position_family = _infer_talent_position_family(row)
    current_level_score = _safe_float(row.get("current_level_score"))
    future_potential_score = _safe_float(row.get("future_potential_score"))
    current_level_confidence = _safe_float(row.get("current_level_confidence"))
    future_potential_confidence = _safe_float(row.get("future_potential_confidence"))
    score_families = row.get("score_families")
    if not isinstance(score_families, dict):
        score_families = _talent_family_payload(row)
    score_explanations = row.get("score_explanations")
    if not isinstance(score_explanations, dict):
        score_explanations = {
            "current_level": _family_driver_entries(row, lane="current_level"),
            "future_potential": _family_driver_entries(row, lane="future_potential"),
        }

    return {
        "talent_position_family": position_family,
        "current_level_score": current_level_score,
        "future_potential_score": future_potential_score,
        "current_level_confidence": current_level_confidence,
        "future_potential_confidence": future_potential_confidence,
        "current_level_confidence_label": _talent_confidence_label(current_level_confidence),
        "future_potential_confidence_label": _talent_confidence_label(future_potential_confidence),
        "current_level_confidence_reasons": list(row.get("current_level_confidence_reasons") or _current_confidence_reasons(row)),
        "future_potential_confidence_reasons": list(row.get("future_potential_confidence_reasons") or _future_confidence_reasons(row)),
        "score_families": score_families,
        "score_explanations": score_explanations,
    }


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

    league_bucket = str(valuation_guardrails.get("league_adjustment_bucket") or "").strip().lower()
    if league_bucket in {"weak", "failed", "severe_failed"}:
        severity = "medium" if league_bucket == "weak" else "high"
        flags.append(
            {
                "severity": severity,
                "code": "league_pricing_adjusted",
                "message": _safe_text(valuation_guardrails.get("league_adjustment_reason"))
                or "Pricing has been adjusted because league holdout reliability is weak.",
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
    report = _build_player_report_from_row(frame=frame, row=row, top_metrics=top_metrics)
    provider_coverage = build_provider_coverage(row)
    freshness = _build_data_freshness(row, provider_coverage=provider_coverage)
    report["provider_coverage"] = provider_coverage
    report["data_freshness"] = freshness
    report["artifact_role"] = freshness.get("artifact_role")
    report["lane_state"] = freshness.get("lane_state")
    report["promotion_state"] = freshness.get("promotion_state")
    report["promotion_reasons"] = freshness.get("promotion_reasons")
    report["similar_players"] = _build_similar_players_payload(frame=frame, row=row, top_k=5)
    report["proxy_estimates"] = _build_proxy_estimates_payload(row=row)
    latest = _list_scout_decisions_for_player(
        player_id=str(row.get("player_id") or player_id),
        split=split,
        season=str(row.get("season") or season or "") or None,
        limit=1,
    )
    report["latest_decision"] = dict(latest[0]) if latest else None
    return report


def _build_player_report_from_row(
    *,
    frame: pd.DataFrame,
    row: pd.Series,
    top_metrics: int = 5,
) -> dict[str, Any]:
    row_dict = _to_records(row.to_frame().T)[0]

    cohort, cohort_filters = _cohort_for_player(frame=frame, row=row)
    metric_map = _build_metric_snapshot_map(row=row, cohort=cohort)
    metric_profile = _build_metric_profile(row=row, cohort=cohort, top_metrics=max(int(top_metrics), 1))
    player_type = _build_player_type_profile(row=row, metric_map=metric_map)
    formation_fit = _build_formation_fit_profile(row=row, metric_map=metric_map)
    radar_profile = _build_radar_profile(row=row, metric_map=metric_map)
    confidence = _build_confidence_summary(row=row, cohort=cohort)
    talent_view = _build_talent_view(row)
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
        "player_type": player_type,
        "formation_fit": formation_fit,
        "radar_profile": radar_profile,
        "risk_flags": risk_flags,
        "confidence": confidence,
        "talent_view": talent_view,
        "valuation_guardrails": valuation_guardrails,
        "summary_text": summary_text,
        "proxy_estimates": _build_proxy_estimates_payload(row=row),
        "latest_decision": _latest_scout_decision_for_row(row),
    }


def query_player_reports(
    split: Split = "test",
    season: str | None = None,
    league: str | None = None,
    club: str | None = None,
    position: str | None = None,
    min_minutes: float | None = None,
    max_age: float | None = None,
    player_ids: Sequence[str] | None = None,
    top_metrics: int = 5,
    include_history: bool = True,
    sort_by: str = "undervaluation_score",
    sort_order: Literal["asc", "desc"] = "desc",
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    work = frame.copy()

    if season and "season" in work.columns:
        work = work[work["season"].astype(str) == str(season)].copy()
    if league and "league" in work.columns:
        work = work[work["league"].astype(str).str.casefold() == str(league).casefold()].copy()
    if club and "club" in work.columns:
        work = work[work["club"].astype(str).str.casefold() == str(club).casefold()].copy()
    if position:
        pos_series = _position_series(work)
        work = work[pos_series == str(position).upper()].copy()

    if min_minutes is not None:
        work = work[_minutes_series(work).fillna(0.0) >= float(min_minutes)].copy()
    if max_age is not None and "age" in work.columns:
        ages = pd.to_numeric(work["age"], errors="coerce")
        work = work[ages <= float(max_age)].copy()

    if player_ids:
        if "player_id" not in work.columns:
            raise ValueError("Prediction artifact does not include 'player_id'.")
        wanted_ids = {str(pid).strip() for pid in player_ids if str(pid).strip()}
        if wanted_ids:
            work = work[work["player_id"].astype(str).isin(wanted_ids)].copy()

    total = int(len(work))

    if sort_by not in work.columns:
        fallback = (
            "undervaluation_score"
            if "undervaluation_score" in work.columns
            else (
                "value_gap_capped_eur"
                if "value_gap_capped_eur" in work.columns
                else (
                    "value_gap_conservative_eur"
                    if "value_gap_conservative_eur" in work.columns
                    else ("value_gap_eur" if "value_gap_eur" in work.columns else work.columns[0])
                )
            )
        )
        sort_by = fallback

    ascending = sort_order == "asc"
    work = work.sort_values(sort_by, ascending=ascending, na_position="last")

    start = max(int(offset), 0)
    end = start + max(int(limit), 0)
    page = work.iloc[start:end].copy()

    items: list[dict[str, Any]] = []
    for _, row in page.iterrows():
        report = _build_player_report_from_row(frame=frame, row=row, top_metrics=top_metrics)
        item: dict[str, Any] = {
            "player_id": str(row.get("player_id") or ""),
            "season": row.get("season"),
            "report": report,
        }
        if include_history:
            item["history_strength"] = build_history_strength_payload(row=row)
        items.append(item)

    return {
        "split": split,
        "total": total,
        "count": int(len(items)),
        "limit": int(limit),
        "offset": int(offset),
        "sort_by": sort_by,
        "sort_order": sort_order,
        "items": items,
    }


def get_player_advanced_profile(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    top_metrics: int = 6,
) -> dict[str, Any]:
    report = get_player_report(
        player_id=player_id,
        split=split,
        season=season,
        top_metrics=top_metrics,
    )
    return {
        "player": report.get("player", {}),
        "cohort": report.get("cohort", {}),
        "player_type": report.get("player_type", {}),
        "formation_fit": report.get("formation_fit", {}),
        "radar_profile": report.get("radar_profile", {}),
        "strengths": report.get("strengths", []),
        "weaknesses": report.get("weaknesses", []),
        "development_levers": report.get("development_levers", []),
        "risk_flags": report.get("risk_flags", []),
        "confidence": report.get("confidence", {}),
        "talent_view": report.get("talent_view", {}),
        "valuation_guardrails": report.get("valuation_guardrails", {}),
        "summary_text": report.get("summary_text"),
        "artifact_role": report.get("artifact_role"),
        "lane_state": report.get("lane_state"),
        "promotion_state": report.get("promotion_state"),
        "promotion_reasons": report.get("promotion_reasons"),
        "latest_decision": report.get("latest_decision"),
    }


def _load_similar_player_matches(
    player_id: str,
    top_k: int,
    *,
    season: str | None = None,
    same_position: bool = True,
    exclude_big5: bool = False,
) -> dict[str, Any]:
    """Load raw similar-player matches plus metadata from the similarity service."""
    from scouting_ml.services.similarity_service import get_similarity_payload

    return dict(
        get_similarity_payload(
            player_id,
            n=top_k,
            same_position=same_position,
            exclude_big5=exclude_big5,
            season=season,
        )
    )


def _build_similar_players_payload(
    *,
    frame: pd.DataFrame,
    row: pd.Series,
    top_k: int = 5,
    same_position: bool = True,
    exclude_big5: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Build a UI-ready similar-player payload from raw similarity matches."""
    player_id = str(row.get("player_id") or "").strip()
    if not player_id:
        return {"available": False, "reason": "missing_player_id", "items": []}

    try:
        raw_payload = _load_similar_player_matches(
            player_id,
            top_k=max(int(top_k), 1),
            season=str(row.get("season") or "").strip() or None,
            same_position=same_position,
            exclude_big5=exclude_big5,
        )
    except Exception as exc:  # pragma: no cover - defensive
        if strict:
            raise
        return {"available": False, "reason": str(exc), "items": []}

    if isinstance(raw_payload, dict):
        raw_matches = list(raw_payload.get("comparisons") or [])
        meta = {
            "position_group": raw_payload.get("position_group"),
            "feature_count_used": raw_payload.get("feature_count_used"),
            "feature_columns_used": list(raw_payload.get("feature_columns_used") or []),
        }
    else:  # pragma: no cover - backward-compatibility for older monkeypatches
        raw_matches = list(raw_payload or [])
        meta = {
            "position_group": _safe_text(row.get("model_position") or row.get("position_group")),
            "feature_count_used": None,
            "feature_columns_used": [],
        }

    enriched: list[dict[str, Any]] = []
    lookup = frame.copy()
    if "player_id" in lookup.columns:
        lookup = lookup.set_index("player_id", drop=False)
    for match in raw_matches:
        candidate_id = str(match.get("player_id") or "").strip()
        item = {
            "player_id": candidate_id,
            "score": _safe_float(match.get("similarity_score")),
            "similarity_score": _safe_float(match.get("similarity_score")),
            "justification": str(match.get("justification") or ""),
            "predicted_value": _safe_float(match.get("predicted_value")),
            "market_value_eur": _safe_float(match.get("market_value_eur")),
        }
        if candidate_id and candidate_id in lookup.index:
            candidate = lookup.loc[candidate_id]
            if isinstance(candidate, pd.DataFrame):
                candidate = candidate.iloc[0]
            item.update(
                {
                    "name": candidate.get("name"),
                    "club": candidate.get("club"),
                    "league": candidate.get("league"),
                    "season": candidate.get("season"),
                    "position": candidate.get("model_position") or candidate.get("position_group"),
                    "market_value_eur": item.get("market_value_eur") or _safe_float(candidate.get("market_value_eur")),
                    "expected_value_eur": _safe_float(candidate.get("expected_value_eur")),
                    "predicted_value": item.get("predicted_value")
                    or _safe_float(candidate.get("fair_value_eur"))
                    or _safe_float(candidate.get("expected_value_eur")),
                }
            )
        enriched.append(item)
    return {
        "available": True,
        "reason": None,
        "position_group": meta["position_group"],
        "feature_count_used": meta["feature_count_used"],
        "feature_columns_used": meta["feature_columns_used"],
        "items": enriched,
    }


def _build_proxy_estimates_payload(*, row: pd.Series) -> dict[str, Any]:
    """Build advisory kNN proxy estimates for sparse player-detail views."""
    player_id = str(row.get("player_id") or "").strip()
    if not player_id:
        return {"available": False, "summary": "Missing player id for proxy estimates.", "metrics": []}
    from scouting_ml.services.proxy_estimate_service import get_player_proxy_estimates

    try:
        return dict(
            get_player_proxy_estimates(
                player_id=player_id,
                season=str(row.get("season") or "").strip() or None,
            )
        )
    except Exception as exc:  # pragma: no cover - advisory sidecar should never break primary report flows
        return {
            "available": False,
            "summary": f"Proxy estimates unavailable: {exc}",
            "metrics": [],
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
        "history_strength": build_history_strength_payload(row=row),
    }


def get_player_profile(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    top_metrics: int = 6,
    similar_top_k: int = 5,
) -> dict[str, Any]:
    frame = _prepare_predictions_frame(get_predictions(split=split))
    row = _select_player_row(frame=frame, player_id=player_id, season=season)
    report = _build_player_report_from_row(frame=frame, row=row, top_metrics=top_metrics)
    history_strength = build_history_strength_payload(row=row)
    player_payload = dict(report.get("player", {}))
    tactical_context = build_external_tactical_context(row)
    availability_context = build_availability_context(row)
    market_context = build_market_context_payload(row)
    provider_coverage = build_provider_coverage(row)
    freshness = _build_data_freshness(row, provider_coverage=provider_coverage)
    combined_risk_flags = list(report.get("risk_flags", [])) + build_provider_risk_flags(row)
    latest = _list_scout_decisions_for_player(
        player_id=str(row.get("player_id") or player_id),
        split=split,
        season=str(row.get("season") or season or "") or None,
        limit=1,
    )
    return {
        "player": player_payload,
        "cohort": report.get("cohort", {}),
        "strengths": report.get("strengths", []),
        "weaknesses": report.get("weaknesses", []),
        "development_levers": report.get("development_levers", []),
        "player_type": report.get("player_type", {}),
        "formation_fit": report.get("formation_fit", {}),
        "radar_profile": report.get("radar_profile", {}),
        "risk_flags": combined_risk_flags,
        "confidence": report.get("confidence", {}),
        "talent_view": report.get("talent_view", {}),
        "valuation_guardrails": report.get("valuation_guardrails", {}),
        "history_strength": history_strength,
        "summary_text": report.get("summary_text"),
        "external_tactical_context": tactical_context,
        "availability_context": availability_context,
        "market_context": market_context,
        "provider_coverage": provider_coverage,
        "data_freshness": freshness,
        "artifact_role": freshness.get("artifact_role"),
        "lane_state": freshness.get("lane_state"),
        "promotion_state": freshness.get("promotion_state"),
        "promotion_reasons": freshness.get("promotion_reasons"),
        "proxy_estimates": _build_proxy_estimates_payload(row=row),
        "latest_decision": dict(latest[0]) if latest else None,
        "stat_groups": build_profile_stat_groups(player_payload),
        "similar_players": _build_similar_players_payload(
            frame=frame,
            row=row,
            top_k=similar_top_k,
        ),
    }


def get_player_similar(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    n: int = 5,
    same_position: bool = True,
    exclude_big5: bool = False,
) -> dict[str, Any]:
    """Return enriched similar-player comparisons for one player."""
    frame = _prepare_predictions_frame(get_predictions(split=split))
    row = _select_player_row(frame=frame, player_id=player_id, season=season)
    payload = _build_similar_players_payload(
        frame=frame,
        row=row,
        top_k=n,
        same_position=same_position,
        exclude_big5=exclude_big5,
        strict=True,
    )
    return {
        "player_id": str(row.get("player_id") or player_id),
        "position_group": payload.get("position_group"),
        "feature_count_used": payload.get("feature_count_used"),
        "feature_columns_used": payload.get("feature_columns_used"),
        "comparisons": payload.get("items", []),
    }


def get_player_trajectory_view(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
) -> dict[str, Any]:
    """Return a compact multi-season trajectory payload for one player."""
    frame = _prepare_predictions_frame(get_predictions(split=split))
    row = _select_player_row(frame=frame, player_id=player_id, season=season)
    from scouting_ml.services.trajectory_service import get_player_trajectory

    return get_player_trajectory(
        player_id=str(row.get("player_id") or player_id),
        season=str(row.get("season") or "").strip() or season,
    )


def build_player_memo_pdf(
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    *,
    include_trajectory: bool = True,
    include_similar: bool = True,
) -> dict[str, Any]:
    """Generate a PDF memo byte payload and filename for one player."""
    report = get_player_report(player_id=player_id, split=split, season=season, top_metrics=5)
    player_payload = report.get("player") if isinstance(report.get("player"), dict) else {}
    resolved_player_id = str(player_payload.get("player_id") or player_id)
    resolved_season = str(player_payload.get("season") or season or "").strip() or None
    trajectory = (
        get_player_trajectory_view(
            player_id=resolved_player_id,
            split=split,
            season=resolved_season,
        )
        if include_trajectory
        else None
    )
    similar_payload = (
        get_player_similar(
            player_id=resolved_player_id,
            split=split,
            season=resolved_season,
            n=3,
            same_position=True,
            exclude_big5=False,
        )
        if include_similar
        else {"comparisons": []}
    )
    from scouting_ml.services.memo_service import get_memo_service

    memo_bytes = get_memo_service().render_pdf(
        report=report,
        trajectory=trajectory,
        similar_players=list(similar_payload.get("comparisons") or []),
        model_manifest=get_model_manifest(),
    )
    player_name = str(player_payload.get("name") or resolved_player_id or "player").strip() or "player"
    return {
        "filename": f"{re.sub(r'[^A-Za-z0-9._-]+', '_', player_name)}_scout_memo.pdf",
        "content": memo_bytes,
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decision_sort_key(record: dict[str, Any]) -> str:
    return str(record.get("created_at_utc") or "")


def _normalize_reason_tags(values: Sequence[Any] | None) -> list[str]:
    allowed = {tag for tags in SCOUT_DECISION_REASON_TAGS.values() for tag in tags}
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        token = str(value or "").strip().lower()
        if not token or token not in allowed or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _validate_scout_decision(action: str, reason_tags: Sequence[str]) -> None:
    if action not in SCOUT_DECISION_ACTIONS:
        raise ValueError(f"Unsupported scout decision action: {action!r}.")
    if action == "pass":
        invalid = [tag for tag in reason_tags if tag not in SCOUT_DECISION_REASON_TAGS["pass"]]
        if invalid:
            raise ValueError(f"Unsupported pass reason tag(s): {', '.join(invalid)}")
        if not reason_tags:
            raise ValueError("Scout decision 'pass' requires at least one reason tag.")
        return
    invalid = [tag for tag in reason_tags if tag not in SCOUT_DECISION_REASON_TAGS["positive"]]
    if invalid:
        raise ValueError(f"Unsupported positive reason tag(s): {', '.join(invalid)}")
    if action == "shortlist" and not reason_tags:
        raise ValueError("Scout decision 'shortlist' requires at least one reason tag.")


def _sanitize_ranking_context(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out = {
        "mode": _safe_text(payload.get("mode")),
        "sort_by": _safe_text(payload.get("sort_by")),
        "rank": int(float(payload.get("rank"))) if _safe_float(payload.get("rank")) is not None else None,
        "active_lane": _safe_text(payload.get("active_lane")),
        "system_template": _safe_text(payload.get("system_template")),
        "system_slot": _safe_text(payload.get("system_slot")),
        "discovery_reliability_weight": _safe_float(payload.get("discovery_reliability_weight")),
    }
    return {key: value for key, value in out.items() if value not in (None, "", [])}


def _build_decision_player_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_id": str(row.get("player_id") or ""),
        "name": row.get("name"),
        "club": row.get("club"),
        "league": row.get("league"),
        "position": row.get("model_position") or row.get("position_group"),
        "season": row.get("season"),
        "market_value_eur": _safe_float(row.get("market_value_eur")),
        "fair_value_eur": _safe_float(row.get("fair_value_eur") or row.get("expected_value_eur")),
        "league_trust_tier": _safe_text(row.get("league_trust_tier")) or "unknown",
        "league_adjustment_bucket": _safe_text(row.get("league_adjustment_bucket")) or "unknown",
    }


def _list_scout_decisions_for_player(
    *,
    player_id: str,
    split: Split | None = None,
    season: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    path = _decisions_path()
    records = read_scout_decision_records(path).records
    out: list[dict[str, Any]] = []
    for item in records:
        if str(item.get("player_id") or "") != str(player_id):
            continue
        if split and str(item.get("split") or "").lower() != str(split).lower():
            continue
        if season is not None and str(item.get("season") or "") != str(season):
            continue
        out.append(dict(item))
    out.sort(key=_decision_sort_key, reverse=True)
    return out[: max(int(limit), 0)]


def _latest_scout_decision_for_row(row: pd.Series | dict[str, Any]) -> dict[str, Any] | None:
    player_id = str(row.get("player_id") or "").strip()
    if not player_id:
        return None
    split = _safe_text(row.get("split")) or None
    season = _safe_text(row.get("season")) or None
    events = _list_scout_decisions_for_player(player_id=player_id, split=split, season=season, limit=1)
    return dict(events[0]) if events else None


def list_player_decisions(
    *,
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    events = _list_scout_decisions_for_player(
        player_id=player_id,
        split=split,
        season=season,
        limit=limit,
    )
    return {
        "player_id": str(player_id),
        "latest_decision": dict(events[0]) if events else None,
        "events": events,
    }


def _existing_watchlist_entry_for_player(
    *,
    path: Path,
    player_id: str,
    split: Split,
    season: str,
) -> dict[str, Any] | None:
    records = read_watchlist_records(path).records
    matches = [
        dict(item)
        for item in records
        if str(item.get("player_id") or "") == str(player_id)
        and str(item.get("split") or "").lower() == str(split).lower()
        and str(item.get("season") or "") == str(season)
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: str(item.get("last_decision_at_utc") or item.get("created_at_utc") or ""), reverse=True)
    return matches[0]


def list_watchlist(
    *,
    split: Split | None = None,
    tag: str | None = None,
    player_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    path = _watchlist_path()
    records = read_watchlist_records(path).records

    out: list[dict[str, Any]] = []
    for item in records:
        if split and str(item.get("split", "")).lower() != str(split):
            continue
        if tag and str(item.get("tag", "")).strip().casefold() != str(tag).strip().casefold():
            continue
        if player_id and str(item.get("player_id", "")) != str(player_id):
            continue
        out.append(item)

    out.sort(key=lambda x: str(x.get("last_decision_at_utc") or x.get("created_at_utc") or ""), reverse=True)
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
    preserve_existing_tag: bool = False,
    decision_action: str | None = None,
    decision_reason_tags: Sequence[str] | None = None,
    decision_note: str | None = None,
    last_decision_at_utc: str | None = None,
) -> dict[str, Any]:
    row = get_player_prediction(player_id=player_id, split=split, season=season)
    report = get_player_report(player_id=player_id, split=split, season=season, top_metrics=5)
    guardrails = report.get("valuation_guardrails", {})
    confidence = report.get("confidence", {})
    risk_flags = report.get("risk_flags", [])

    resolved_season = str(row.get("season") or season or "")
    path = _watchlist_path()
    resolved_tag = tag or ""
    existing_record = None
    if preserve_existing_tag:
        existing_record = _existing_watchlist_entry_for_player(
            path=path,
            player_id=str(row.get("player_id") or player_id),
            split=split,
            season=resolved_season,
        )
        if existing_record and not resolved_tag:
            resolved_tag = str(existing_record.get("tag") or "")

    record = {
        "watch_id": uuid.uuid4().hex,
        "created_at_utc": _now_utc_iso(),
        "split": split,
        "season": resolved_season,
        "player_id": str(row.get("player_id") or player_id),
        "name": row.get("name"),
        "league": row.get("league"),
        "club": row.get("club"),
        "position": row.get("model_position") or row.get("position_group"),
        "tag": resolved_tag,
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
        "decision_action": _safe_text(decision_action),
        "decision_reason_tags": list(decision_reason_tags or []),
        "last_decision_at_utc": _safe_text(last_decision_at_utc),
        "decision_note": decision_note or "",
    }
    return upsert_watchlist_record(path, record)


def save_scout_decision(
    *,
    player_id: str,
    split: Split = "test",
    season: str | None = None,
    action: str,
    reason_tags: Sequence[str] | None = None,
    note: str | None = None,
    actor: str | None = None,
    source_surface: str | None = None,
    ranking_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_action = str(action or "").strip().lower()
    normalized_reason_tags = _normalize_reason_tags(reason_tags)
    _validate_scout_decision(normalized_action, normalized_reason_tags)

    row = get_player_prediction(player_id=player_id, split=split, season=season)
    created_at_utc = _now_utc_iso()
    decision = {
        "decision_id": uuid.uuid4().hex,
        "created_at_utc": created_at_utc,
        "player_id": str(row.get("player_id") or player_id),
        "split": split,
        "season": str(row.get("season") or season or ""),
        "action": normalized_action,
        "reason_tags": normalized_reason_tags,
        "note": str(note or "").strip(),
        "actor": str(actor or "local").strip() or "local",
        "source_surface": str(source_surface or "detail").strip() or "detail",
        "player_snapshot": _build_decision_player_snapshot(row),
        "ranking_context": _sanitize_ranking_context(ranking_context),
    }
    path = _decisions_path()
    saved = append_scout_decision_record(path, decision)

    watchlist_item = None
    if normalized_action in POSITIVE_SCOUT_DECISION_ACTIONS:
        watchlist_item = add_watchlist_item(
            player_id=str(row.get("player_id") or player_id),
            split=split,
            season=str(row.get("season") or season or "") or None,
            tag="",
            notes="",
            source=str(source_surface or "decision_sync").strip() or "decision_sync",
            preserve_existing_tag=True,
            decision_action=normalized_action,
            decision_reason_tags=normalized_reason_tags,
            decision_note=str(note or "").strip(),
            last_decision_at_utc=created_at_utc,
        )

    return {
        "decision": dict(saved),
        "latest_decision": dict(saved),
        "watchlist_item": watchlist_item,
    }


def delete_watchlist_item(watch_id: str) -> dict[str, Any]:
    path = _watchlist_path()
    deleted = delete_watchlist_record(path, watch_id)
    return {
        "path": str(path),
        "watch_id": str(watch_id),
        "deleted": bool(deleted),
    }


def get_model_manifest() -> dict[str, Any]:
    test_path = _resolve_path(*SPLIT_TO_PATH["test"])
    val_path = _resolve_path(*SPLIT_TO_PATH["val"])
    metrics_path = _resolve_path(METRICS_ENV, DEFAULT_METRICS)
    manifest_env_raw = os.getenv(MODEL_MANIFEST_ENV, "").strip()
    manifest_path = _manifest_path()

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["source"] = str(payload.get("source") or "file")
        payload["_meta"] = {
            "source": payload["source"],
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "mtime_utc": _file_meta(manifest_path).get("mtime_utc"),
        }
        for role in ("valuation", "future_shortlist"):
            key = ROLE_TO_MANIFEST_KEY[role]
            if key not in payload:
                section = _manifest_role_section(payload, role)
                if isinstance(section, dict):
                    payload[key] = section
            elif isinstance(payload.get(key), dict):
                normalized_section = dict(payload.get(key) or {})
                normalized_section.setdefault("lane_state", lane_state_for_role(role))
                normalized_section.setdefault("promotion_state", "advisory_only")
                normalized_section.setdefault("promotion_reasons", [])
                payload[key] = normalized_section
        primary_section = _manifest_role_section(payload, "valuation") or _manifest_role_section(
            payload, "future_shortlist"
        )
        if isinstance(primary_section, dict):
            payload["lane_state"] = primary_section.get("lane_state")
            payload["promotion_state"] = primary_section.get("promotion_state")
            payload["promotion_reasons"] = primary_section.get("promotion_reasons")
        if manifest_env_raw or _manifest_targets_active_artifacts(
            payload,
            test_path=test_path,
            val_path=val_path,
            metrics_path=metrics_path,
        ):
            return payload

    valuation_paths = _resolve_role_artifact_paths("valuation")
    future_paths = _resolve_role_artifact_paths("future_shortlist")
    out: dict[str, Any] = {
        "registry_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "derived",
        "legacy_default_role": "valuation",
        "lane_state": lane_state_for_role("valuation"),
        "promotion_state": "advisory_only",
        "promotion_reasons": [],
        "artifacts": {
            "test_predictions": {
                "path": str(valuation_paths["test_predictions"]),
                **_file_meta(valuation_paths["test_predictions"]),
                "sha256": _sha256_file(valuation_paths["test_predictions"]),
            },
            "val_predictions": {
                "path": str(valuation_paths["val_predictions"]),
                **_file_meta(valuation_paths["val_predictions"]),
                "sha256": _sha256_file(valuation_paths["val_predictions"]),
            },
            "metrics": {
                "path": str(valuation_paths["metrics"]),
                **_file_meta(valuation_paths["metrics"]),
                "sha256": _sha256_file(valuation_paths["metrics"]),
            },
        },
        "config": {},
        "summary": {},
    }

    try:
        metrics = get_metrics(role="valuation")
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

    for role, role_paths in (("valuation", valuation_paths), ("future_shortlist", future_paths)):
        try:
            role_metrics = get_metrics(role=role)
            role_config = {
                "dataset": role_metrics.get("dataset"),
                "val_season": role_metrics.get("val_season"),
                "test_season": role_metrics.get("test_season"),
                "trials_per_position": role_metrics.get("trials_per_position"),
                "recency_half_life": role_metrics.get("recency_half_life"),
                "optimize_metric": role_metrics.get("optimize_metric"),
                "interval_q": role_metrics.get("interval_q"),
                "two_stage_band_model": role_metrics.get("two_stage_band_model"),
                "band_min_samples": role_metrics.get("band_min_samples"),
                "band_blend_alpha": role_metrics.get("band_blend_alpha"),
                "strict_leakage_guard": role_metrics.get("strict_leakage_guard"),
            }
            role_summary = {
                "overall": role_metrics.get("overall"),
                "segments": role_metrics.get("segments"),
                "holdout": role_metrics.get("holdout"),
                "artifacts": role_metrics.get("artifacts"),
            }
        except Exception as exc:
            role_config = {}
            role_summary = {"metrics_error": str(exc)}

        out[ROLE_TO_MANIFEST_KEY[role]] = {
            "role": role,
            "label": role,
            "lane_state": lane_state_for_role(role),
            "promotion_state": "advisory_only",
            "promotion_reasons": []
            if role == "valuation"
            else ["future_shortlist is a live scouting lane and should be treated as advisory by default."],
            "generated_at_utc": out["generated_at_utc"],
            "artifacts": {
                "test_predictions": {
                    "path": str(role_paths["test_predictions"]),
                    **_file_meta(role_paths["test_predictions"]),
                    "sha256": _sha256_file(role_paths["test_predictions"]),
                },
                "val_predictions": {
                    "path": str(role_paths["val_predictions"]),
                    **_file_meta(role_paths["val_predictions"]),
                    "sha256": _sha256_file(role_paths["val_predictions"]),
                },
                "metrics": {
                    "path": str(role_paths["metrics"]),
                    **_file_meta(role_paths["metrics"]),
                    "sha256": _sha256_file(role_paths["metrics"]),
                },
            },
            "config": role_config,
            "summary": role_summary,
        }

    return out


def health_payload() -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "ok",
        "artifacts": {},
        "strict_artifacts": _env_flag("SCOUTING_STRICT_ARTIFACTS", default=False),
        "strict_artifacts_error": None,
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
    if out["strict_artifacts"]:
        try:
            validate_strict_artifact_env()
        except Exception as exc:
            out["strict_artifacts_error"] = str(exc)
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
    if (
        out["strict_artifacts_error"]
        or not test_meta["exists"]
        or not val_meta["exists"]
        or not metrics_meta["exists"]
        or out.get("test_error")
        or out.get("val_error")
        or out.get("metrics_error")
    ):
        out["status"] = "error"
    return out


def _stale_provider_snapshot_summary(
    ingestion_rows: Sequence[dict[str, Any]],
    *,
    threshold_days: int = 14,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    stale_rows: list[dict[str, Any]] = []
    latest_seen: list[str] = []
    for row in ingestion_rows:
        snapshot = _safe_text(row.get("latest_provider_snapshot_date"))
        if snapshot:
            latest_seen.append(snapshot)
        try:
            parsed_text = f"{snapshot}T00:00:00+00:00" if snapshot and len(snapshot) == 10 else snapshot
            parsed = (
                datetime.fromisoformat(parsed_text.replace("Z", "+00:00")) if parsed_text else None
            )
        except Exception:
            parsed = None
        if snapshot is None or parsed is None:
            stale_rows.append(row)
            continue
        if (now - parsed).days > int(threshold_days):
            stale_rows.append(row)
    return {
        "threshold_days": int(threshold_days),
        "stale_count": int(len(stale_rows)),
        "latest_snapshot_date": latest_timestamp(latest_seen),
        "items": [
            {
                "league_name": row.get("league_name"),
                "season": row.get("season"),
                "status": row.get("status"),
                "latest_provider_snapshot_date": row.get("latest_provider_snapshot_date"),
            }
            for row in stale_rows[:8]
        ],
    }


def _live_partial_footprint() -> dict[str, Any]:
    try:
        frame = _prepare_predictions_frame(get_predictions(split="test"))
    except Exception as exc:
        return {"total_rows": 0, "live_rows": 0, "partial_season_rows": 0, "error": str(exc)}

    manifest = get_model_manifest()
    future_section = _manifest_role_section(manifest, "future_shortlist") or {}
    live_test_season = _safe_text((future_section.get("config") or {}).get("test_season")) if isinstance(
        future_section.get("config"), dict
    ) else None
    if frame.empty or "season" not in frame.columns or not live_test_season:
        return {
            "total_rows": int(len(frame)),
            "live_rows": 0,
            "partial_season_rows": 0,
            "live_test_season": live_test_season,
            "live_share": 0.0,
        }

    season_series = frame["season"].astype(str)
    overlay_mask = frame.apply(_row_uses_future_overlay, axis=1)
    live_mask = overlay_mask & season_series.eq(str(live_test_season))
    live_rows = int(live_mask.sum())
    total_rows = int(len(frame))
    return {
        "total_rows": total_rows,
        "live_rows": live_rows,
        "partial_season_rows": live_rows,
        "live_test_season": live_test_season,
        "live_share": float(live_rows / total_rows) if total_rows else 0.0,
    }


def _ui_bootstrap_cache_version(split: Split) -> tuple[tuple[str, int], ...]:
    valuation_paths = _resolve_role_artifact_paths("valuation")
    future_paths = _resolve_role_artifact_paths("future_shortlist")
    split_key = "test_predictions" if split == "test" else "val_predictions"
    return _prediction_cache_version(valuation_paths[split_key], future_paths[split_key])


def get_ui_bootstrap(split: Split = "test") -> dict[str, Any]:
    cache_key = str(split)
    version = _ui_bootstrap_cache_version(split)
    cached = _UI_BOOTSTRAP_CACHE.get(cache_key)
    if cached is not None and cached.version == version:
        return cached.payload

    frame = get_predictions(split=split)
    seasons = sorted(
        {
            str(value).strip()
            for value in frame.get("season", pd.Series(dtype=object)).tolist()
            if str(value).strip()
        },
        key=_season_sort_key,
        reverse=True,
    )
    leagues = sorted(
        {
            str(value).strip()
            for value in frame.get("league", pd.Series(dtype=object)).tolist()
            if str(value).strip()
        }
    )

    work = frame.copy()
    work["league"] = work.get("league", pd.Series("Unknown", index=work.index, dtype=object)).astype(str).str.strip()
    work["league"] = work["league"].where(work["league"].ne(""), "Unknown")
    gap = _to_numeric_series(work, "value_gap_conservative_eur")
    if gap.isna().all():
        gap = _to_numeric_series(work, "value_gap_eur")
    confidence = _to_numeric_series(work, "undervaluation_confidence")
    undervalued_flag = _to_numeric_series(work, "undervalued_flag")
    undervalued = pd.Series(False, index=work.index, dtype=bool)
    undervalued.loc[undervalued_flag.notna()] = undervalued_flag.loc[undervalued_flag.notna()] > 0
    if gap.notna().any():
        gap_mask = gap > 0
        undervalued = undervalued | gap_mask.fillna(False)
    work["_undervalued"] = undervalued.astype(float)
    work["_confidence"] = confidence

    grouped = (
        work.groupby("league", dropna=False)
        .agg(
            rows=("league", "size"),
            undervalued_share=("_undervalued", "mean"),
            avg_confidence=("_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["rows", "league"], ascending=[False, True], kind="mergesort")
    )

    payload = {
        "split": str(split),
        "seasons": seasons,
        "leagues": leagues,
        "coverage_rows": _to_records(grouped),
        "generated_at_utc": _now_utc_iso(),
    }
    _UI_BOOTSTRAP_CACHE[cache_key] = _SummaryPayloadCache(version=version, payload=payload)
    return payload


def _operator_health_cache_version() -> tuple[tuple[str, int], ...]:
    valuation_paths = _resolve_role_artifact_paths("valuation")
    future_paths = _resolve_role_artifact_paths("future_shortlist")
    backtest_json = Path("data/model/backtests/rolling_backtest_summary.json")
    backtest_csv = Path("data/model/backtests/rolling_backtest_summary.csv")
    clean_dataset_path = Path(PRODUCTION_PIPELINE_DEFAULTS.clean_output)
    manifest = _manifest_path()
    return _prediction_cache_version(
        valuation_paths["metrics"],
        valuation_paths["test_predictions"],
        future_paths["test_predictions"],
        backtest_json,
        backtest_csv,
        clean_dataset_path,
        manifest,
    )


def get_operator_health() -> dict[str, Any]:
    global _OPERATOR_HEALTH_CACHE
    version = _operator_health_cache_version()
    if _OPERATOR_HEALTH_CACHE is not None and _OPERATOR_HEALTH_CACHE.version == version:
        return _OPERATOR_HEALTH_CACHE.payload

    manifest = get_model_manifest()
    active_artifacts = get_active_artifacts()
    valuation_section = _manifest_role_section(manifest, "valuation") or {}
    future_section = _manifest_role_section(manifest, "future_shortlist") or {}

    try:
        valuation_metrics = get_metrics(role="valuation")
    except Exception as exc:
        valuation_metrics = {"metrics_error": str(exc)}

    backtest_json = Path("data/model/backtests/rolling_backtest_summary.json")
    backtest_csv = Path("data/model/backtests/rolling_backtest_summary.csv")
    backtest_payload = _load_json_snapshot(backtest_json)
    promotion_gate = build_valuation_promotion_gate(
        metrics_payload=valuation_metrics,
        backtest_payload=backtest_payload,
        backtest_rows_path=backtest_csv if backtest_csv.exists() else None,
        requested_backtest_test_seasons=[
            token.strip()
            for token in str(PRODUCTION_PIPELINE_DEFAULTS.backtest_test_seasons).split(",")
            if token.strip()
        ],
        min_test_r2=PRODUCTION_PIPELINE_DEFAULTS.backtest_min_test_r2,
        max_test_wmape=PRODUCTION_PIPELINE_DEFAULTS.backtest_max_test_wmape,
        max_under5m_wmape=PRODUCTION_PIPELINE_DEFAULTS.backtest_max_under5m_wmape,
        max_lowmid_weighted_wmape=PRODUCTION_PIPELINE_DEFAULTS.backtest_max_lowmid_weighted_wmape,
        max_segment_weighted_wmape=PRODUCTION_PIPELINE_DEFAULTS.backtest_max_segment_weighted_wmape,
    )
    ingestion = _get_ingestion_health_payload()
    ingestion_rows = list(ingestion.get("rows") or [])
    blocked_items = [row for row in ingestion_rows if str(row.get("status")) == "blocked"][:8]
    watch_items = [row for row in ingestion_rows if str(row.get("status")) == "watch"][:8]

    payload = {
        "generated_at_utc": _now_utc_iso(),
        "active_lanes": {
            "valuation": _artifact_lane_payload("valuation", valuation_section),
            "future_shortlist": _artifact_lane_payload("future_shortlist", future_section),
        },
        "active_artifacts": active_artifacts,
        "promotion_gate": promotion_gate,
        "holdout_coverage": promotion_gate.get("holdout_coverage"),
        "ingestion_health": {
            "summary": ingestion.get("summary", {}),
            "blocked_items": blocked_items,
            "watch_items": watch_items,
            "_meta": ingestion.get("_meta"),
        },
        "stale_provider_snapshots": _stale_provider_snapshot_summary(ingestion_rows),
        "live_partial_footprint": _live_partial_footprint(),
    }
    _OPERATOR_HEALTH_CACHE = _SummaryPayloadCache(version=version, payload=payload)
    return payload


__all__ = [
    "ArtifactNotFoundError",
    "Split",
    "add_watchlist_item",
    "delete_watchlist_item",
    "list_player_decisions",
    "save_scout_decision",
    "get_active_artifacts",
    "get_model_manifest",
    "get_operator_health",
    "get_metrics",
    "get_player_advanced_profile",
    "build_player_memo_pdf",
    "get_player_profile",
    "get_player_history_strength",
    "get_player_report",
    "get_player_similar",
    "get_player_prediction",
    "get_player_trajectory_view",
    "get_predictions",
    "get_ui_bootstrap",
    "get_system_fit_templates",
    "get_resolved_artifact_paths",
    "health_payload",
    "list_watchlist",
    "query_player_reports",
    "query_predictions",
    "query_scout_targets",
    "query_shortlist",
    "query_system_fit",
    "validate_strict_artifact_env",
]
