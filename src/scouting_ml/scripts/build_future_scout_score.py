from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scouting_ml.reporting.future_value_benchmarks import attach_future_targets, load_future_targets_frame


NUMERIC_CANDIDATES = [
    "undervaluation_score",
    "undervaluation_confidence",
    "value_gap_capped_eur",
    "value_gap_conservative_eur",
    "value_gap_eur",
    "value_efficiency_ratio",
    "market_value_eur",
    "expected_value_eur",
    "fair_value_eur",
    "age",
    "minutes",
    "sofa_minutesPlayed",
    "history_strength_score",
    "history_strength_coverage",
    "league_is_non_big5",
    "trajectory_value_change_pct_prev",
    "trajectory_minutes_change_pct_prev",
    "trajectory_prior_sample_count",
    "trajectory_history_coverage",
    "talent_impact_score",
    "talent_technical_score",
    "talent_tactical_score",
    "talent_physical_score",
    "talent_context_score",
    "talent_trajectory_score",
]

CATEGORICAL_CANDIDATES = [
    "model_position",
    "position_group",
    "league",
    "talent_position_family",
]

LABEL_MAP = {
    "positive_growth": "value_growth_positive_flag",
    "growth_gt25pct": "value_growth_gt25pct_flag",
}

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

FAMILY_ORDER = (
    "impact",
    "technical",
    "tactical",
    "physical",
    "context",
    "trajectory",
)

FAMILY_LABELS = {
    "impact": "Impact",
    "technical": "Technical",
    "tactical": "Tactical",
    "physical": "Physical",
    "context": "Context",
    "trajectory": "Trajectory",
}

POSITION_FAMILY_WEIGHTS: dict[str, dict[str, float]] = {
    "GK": {"impact": 0.12, "technical": 0.12, "tactical": 0.26, "physical": 0.20, "context": 0.18, "trajectory": 0.12},
    "CB": {"impact": 0.16, "technical": 0.12, "tactical": 0.28, "physical": 0.22, "context": 0.14, "trajectory": 0.08},
    "FB": {"impact": 0.18, "technical": 0.14, "tactical": 0.22, "physical": 0.20, "context": 0.12, "trajectory": 0.14},
    "CM": {"impact": 0.16, "technical": 0.20, "tactical": 0.24, "physical": 0.10, "context": 0.12, "trajectory": 0.18},
    "AM": {"impact": 0.23, "technical": 0.20, "tactical": 0.18, "physical": 0.08, "context": 0.12, "trajectory": 0.19},
    "W": {"impact": 0.24, "technical": 0.19, "tactical": 0.14, "physical": 0.12, "context": 0.10, "trajectory": 0.21},
    "ST": {"impact": 0.28, "technical": 0.20, "tactical": 0.12, "physical": 0.10, "context": 0.16, "trajectory": 0.14},
}

FAMILY_SPECS: dict[str, list[tuple[str, float, int]]] = {
    "impact": [
        ("undervaluation_score", 1.2, 1),
        ("value_gap_capped_eur", 1.4, 1),
        ("value_gap_conservative_eur", 1.2, 1),
        ("value_efficiency_ratio", 1.0, 1),
        ("fair_value_eur", 0.8, 1),
        ("expected_value_eur", 0.8, 1),
        ("history_strength_score", 0.6, 1),
    ],
    "technical": [
        ("sofa_expectedGoals_per90", 1.0, 1),
        ("sofa_goals_per90", 0.8, 1),
        ("sofa_assists_per90", 0.8, 1),
        ("sofa_totalShots_per90", 0.7, 1),
        ("sofa_keyPasses_per90", 0.9, 1),
        ("sofa_successfulDribbles_per90", 1.0, 1),
        ("sofa_accuratePassesPercentage", 0.7, 1),
        ("sofa_totalDuelsWonPercentage", 0.5, 1),
    ],
    "tactical": [
        ("sb_progressive_passes_per90", 1.0, 1),
        ("sb_progressive_carries_per90", 1.0, 1),
        ("sb_passes_into_box_per90", 0.9, 1),
        ("sb_pressures_per90", 0.6, 1),
        ("sofa_keyPasses_per90", 0.7, 1),
        ("sofa_successfulDribbles_per90", 0.6, 1),
        ("history_strength_coverage", 0.4, 1),
    ],
    "physical": [
        ("minutes", 1.0, 1),
        ("sofa_minutesPlayed", 1.0, 1),
        ("injury_days_per_1000_min", 0.9, -1),
    ],
    "context": [
        ("age", 1.1, -1),
        ("contract_years_left", 0.7, 1),
        ("league_is_non_big5", 0.5, 1),
        ("history_strength_coverage", 0.7, 1),
        ("undervaluation_confidence", 0.6, 1),
    ],
    "trajectory": [
        ("trajectory_value_change_pct_prev", 1.1, 1),
        ("trajectory_minutes_change_pct_prev", 0.9, 1),
        ("trajectory_prior_sample_count", 0.7, 1),
        ("trajectory_history_coverage", 1.0, 1),
        ("history_strength_score", 0.9, 1),
        ("history_strength_coverage", 0.7, 1),
    ],
}

PHYSICAL_KEYWORDS = (
    "sprint",
    "accel",
    "decel",
    "speed",
    "hsr",
    "distance",
    "top_speed",
    "max_speed",
    "workload",
)


def _resolve_base_rank_col(frame: pd.DataFrame) -> str:
    for col in ("scout_target_score", "undervaluation_score", "value_gap_capped_eur", "value_gap_conservative_eur", "value_gap_eur"):
        if col in frame.columns:
            return col
    raise ValueError("No suitable ranking column found for future scout score blending.")


def _minutes_series(frame: pd.DataFrame) -> pd.Series:
    if "minutes" in frame.columns:
        return pd.to_numeric(frame["minutes"], errors="coerce")
    if "sofa_minutesPlayed" in frame.columns:
        return pd.to_numeric(frame["sofa_minutesPlayed"], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _league_norm_series(frame: pd.DataFrame) -> pd.Series:
    if "league" not in frame.columns:
        return pd.Series("unknown", index=frame.index, dtype=object)
    return frame["league"].astype(str).str.strip().str.casefold()


def _parse_positions(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    out = {token.strip().upper() for token in str(raw).split(",") if token.strip()}
    return out or None


def _rank_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranks = values.rank(method="average", pct=True)
    return ranks.fillna(ranks.median() if ranks.notna().any() else 0.5).clip(lower=0.0, upper=1.0)


def _precision_at_k(frame: pd.DataFrame, *, score_col: str, label_col: str, k: int) -> float | None:
    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_label"] = pd.to_numeric(work[label_col], errors="coerce")
    work = work[work["_score"].notna() & work["_label"].notna()].copy()
    if work.empty:
        return None
    work = work.sort_values("_score", ascending=False).head(max(int(k), 1))
    if work.empty:
        return None
    return float((work["_label"] > 0).mean())


def _prepare_rows(
    frame: pd.DataFrame,
    *,
    min_minutes: float,
    max_age: float | None,
    positions: set[str] | None,
    include_leagues: set[str] | None,
    exclude_leagues: set[str] | None,
) -> pd.DataFrame:
    work = frame.copy()
    work["_minutes_used"] = _minutes_series(work).fillna(0.0)
    work["_age_num"] = _numeric_series(work, "age")
    work["_league_norm"] = _league_norm_series(work)

    work = work[work["_minutes_used"] >= float(min_minutes)].copy()
    if max_age is not None:
        work = work[work["_age_num"].fillna(999.0) <= float(max_age)].copy()
    if positions:
        pos_series = (
            work["model_position"].astype(str).str.upper()
            if "model_position" in work.columns
            else work.get("position_group", pd.Series("", index=work.index)).astype(str).str.upper()
        )
        work = work[pos_series.isin(positions)].copy()
    if include_leagues:
        work = work[work["_league_norm"].isin(include_leagues)].copy()
    if exclude_leagues:
        work = work[~work["_league_norm"].isin(exclude_leagues)].copy()
    return work


def _feature_lists(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [col for col in NUMERIC_CANDIDATES if col in frame.columns]
    categorical = [col for col in CATEGORICAL_CANDIDATES if col in frame.columns]
    if not numeric and not categorical:
        raise ValueError("No supported feature columns available for future scout score.")
    return numeric, categorical


def _top_coefficients(model: Pipeline, *, limit: int = 20) -> list[dict[str, Any]]:
    try:
        pre = model.named_steps["preprocess"]
        clf = model.named_steps["model"]
        feature_names = pre.get_feature_names_out()
        coefs = clf.coef_[0]
    except Exception:
        return []
    rows = [{"feature": str(name), "coefficient": float(weight)} for name, weight in zip(feature_names, coefs)]
    rows.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
    return rows[: max(int(limit), 1)]


def _season_sort_key(raw: Any) -> tuple[int, int]:
    text = str(raw or "").strip()
    if not text:
        return (0, 0)
    split_match = re.match(r"^(?P<start>\d{4})[/-](?P<end>\d{2,4})$", text)
    if split_match:
        start = int(split_match.group("start"))
        end_raw = split_match.group("end")
        end = int(end_raw)
        if len(end_raw) == 2:
            end = (start // 100) * 100 + end
        return (start, end)
    year_match = re.match(r"^\d{4}$", text)
    if year_match:
        year = int(text)
        return (year, year)
    return (0, 0)


def _build_trajectory_frame(dataset_path: str | None) -> pd.DataFrame:
    if not dataset_path:
        return pd.DataFrame(columns=[
            "player_id",
            "season",
            "trajectory_value_change_pct_prev",
            "trajectory_minutes_change_pct_prev",
            "trajectory_prior_sample_count",
            "trajectory_history_coverage",
        ])

    path = Path(dataset_path)
    if not path.exists():
        return pd.DataFrame(columns=[
            "player_id",
            "season",
            "trajectory_value_change_pct_prev",
            "trajectory_minutes_change_pct_prev",
            "trajectory_prior_sample_count",
            "trajectory_history_coverage",
        ])

    frame = pd.read_parquet(path)
    needed = {"player_id", "season"}
    if not needed.issubset(frame.columns):
        return pd.DataFrame(columns=[
            "player_id",
            "season",
            "trajectory_value_change_pct_prev",
            "trajectory_minutes_change_pct_prev",
            "trajectory_prior_sample_count",
            "trajectory_history_coverage",
        ])

    work = frame.copy()
    work["player_id"] = work["player_id"].astype(str)
    work["season"] = work["season"].astype(str)
    work["market_value_eur"] = _numeric_series(work, "market_value_eur")
    work["minutes"] = _numeric_series(work, "minutes")
    work["_season_key"] = work["season"].map(_season_sort_key)
    work = work.sort_values(["player_id", "_season_key"], na_position="last").copy()

    grouped = work.groupby("player_id", dropna=False)
    prev_value = grouped["market_value_eur"].shift(1)
    prev_minutes = grouped["minutes"].shift(1)
    prior_count = grouped.cumcount()

    value_delta = (work["market_value_eur"] - prev_value) / prev_value.replace(0.0, np.nan)
    minutes_delta = (work["minutes"] - prev_minutes) / prev_minutes.replace(0.0, np.nan)
    history_cov = np.clip(prior_count / 2.0, 0.0, 1.0)

    out = pd.DataFrame(
        {
            "player_id": work["player_id"].astype(str),
            "season": work["season"].astype(str),
            "trajectory_value_change_pct_prev": value_delta.replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=3.0),
            "trajectory_minutes_change_pct_prev": minutes_delta.replace([np.inf, -np.inf], np.nan).clip(lower=-1.0, upper=3.0),
            "trajectory_prior_sample_count": prior_count.astype(float),
            "trajectory_history_coverage": history_cov.astype(float),
        }
    )
    return out.drop_duplicates(subset=["player_id", "season"], keep="last").reset_index(drop=True)


def _safe_text_token(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _infer_position_family(row: pd.Series) -> str:
    family_key = _safe_text_token(row.get("model_position") or row.get("position_group")).upper()
    primary = " ".join(
        part
        for part in (
            _safe_text_token(row.get("position_main")),
            _safe_text_token(row.get("position")),
            _safe_text_token(row.get("position_alt")),
        )
        if part
    ).lower()

    if family_key == "GK" or "goal" in primary:
        return "GK"
    if family_key == "DF":
        if any(token in primary for token in ("wing-back", "wing back", "fullback", "full back", "left back", "right back")):
            return "FB"
        if "defender, left" in primary or "defender, right" in primary:
            return "FB"
        return "CB"
    if family_key == "MF":
        if "attacking" in primary:
            return "AM"
        if "left midfield" in primary or "right midfield" in primary or "wing" in primary:
            return "W"
        return "CM"
    if family_key == "FW":
        if "winger" in primary:
            return "W"
        return "ST"
    return "CM"


def _augment_base_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["market_value_eur"] = _numeric_series(out, "market_value_eur")
    out["fair_value_eur"] = _numeric_series(out, "fair_value_eur")
    out["expected_value_eur"] = _numeric_series(out, "expected_value_eur")
    out["value_gap_conservative_eur"] = _numeric_series(out, "value_gap_conservative_eur")
    out["value_gap_capped_eur"] = _numeric_series(out, "value_gap_capped_eur")
    out["value_gap_eur"] = _numeric_series(out, "value_gap_eur")
    out["history_strength_score"] = _numeric_series(out, "history_strength_score")
    out["history_strength_coverage"] = _numeric_series(out, "history_strength_coverage")
    out["contract_years_left"] = _numeric_series(out, "contract_years_left")
    out["age"] = _numeric_series(out, "age")
    out["minutes"] = _numeric_series(out, "minutes")
    out["sofa_minutesPlayed"] = _numeric_series(out, "sofa_minutesPlayed")
    market = out["market_value_eur"].replace(0.0, np.nan)
    gap = out["value_gap_capped_eur"].fillna(out["value_gap_conservative_eur"]).fillna(out["value_gap_eur"])
    out["value_efficiency_ratio"] = (gap / market).replace([np.inf, -np.inf], np.nan)
    league_norm = _league_norm_series(out)
    out["league_is_non_big5"] = (~league_norm.isin(BIG5_LEAGUES)).astype(float)
    out["talent_position_family"] = out.apply(_infer_position_family, axis=1)
    return out


def _extended_family_specs(frame: pd.DataFrame) -> dict[str, list[tuple[str, float, int]]]:
    specs = {family: list(rows) for family, rows in FAMILY_SPECS.items()}
    for col in frame.columns:
        key = str(col).casefold()
        if any(token in key for token in PHYSICAL_KEYWORDS) and col not in {spec[0] for spec in specs["physical"]}:
            specs["physical"].append((col, 0.5, 1))
    return specs


def _score_family(frame: pd.DataFrame, specs: list[tuple[str, float, int]]) -> tuple[pd.Series, pd.Series]:
    numerator = pd.Series(0.0, index=frame.index, dtype=float)
    denominator = pd.Series(0.0, index=frame.index, dtype=float)
    total_weight = 0.0

    for col, weight, direction in specs:
        if col not in frame.columns:
            continue
        total_weight += float(weight)
        series = pd.to_numeric(frame[col], errors="coerce")
        percentile = _rank_percentile(series)
        quality = percentile if direction > 0 else (1.0 - percentile)
        mask = series.notna()
        numerator = numerator + quality.where(mask, 0.0) * float(weight)
        denominator = denominator + mask.astype(float) * float(weight)

    if total_weight <= 0:
        return (
            pd.Series(50.0, index=frame.index, dtype=float),
            pd.Series(0.0, index=frame.index, dtype=float),
        )

    score = np.where(denominator > 0, numerator / denominator, 0.5)
    coverage = denominator / total_weight
    return (
        pd.Series(score, index=frame.index, dtype=float).clip(lower=0.0, upper=1.0) * 100.0,
        pd.Series(coverage, index=frame.index, dtype=float).clip(lower=0.0, upper=1.0),
    )


def _weighted_family_score(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for family, weights in POSITION_FAMILY_WEIGHTS.items():
        mask = frame["talent_position_family"].astype(str) == family
        if not mask.any():
            continue
        score = pd.Series(0.0, index=frame.index[mask], dtype=float)
        for family_name, weight in weights.items():
            score = score + pd.to_numeric(frame.loc[mask, f"talent_{family_name}_score"], errors="coerce").fillna(50.0) * float(weight)
        out.loc[mask] = score
    return out.fillna(50.0).clip(lower=0.0, upper=100.0)


def _future_confidence(frame: pd.DataFrame, *, model_probability: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    minutes_sample = _minutes_series(out).fillna(0.0)
    sample_adequacy = (minutes_sample / 1800.0).clip(lower=0.15, upper=1.0)

    provider_groups: list[pd.Series] = []
    for prefixes in (("sb_",), ("avail_",), ("fixture_",), ("odds_",), ("sofa_",)):
        cols = [col for col in out.columns if any(str(col).startswith(prefix) for prefix in prefixes)]
        if cols:
            provider_groups.append(out[cols].notna().any(axis=1).astype(float))
    if provider_groups:
        provider_quality = pd.concat(provider_groups, axis=1).mean(axis=1)
    else:
        provider_quality = pd.Series(0.35, index=out.index, dtype=float)

    history_cov = _numeric_series(out, "history_strength_coverage").fillna(0.35).clip(lower=0.0, upper=1.0)
    trajectory_cov = _numeric_series(out, "trajectory_history_coverage").fillna(0.25).clip(lower=0.0, upper=1.0)
    context_coverage = ((history_cov * 0.6) + (trajectory_cov * 0.4)).clip(lower=0.0, upper=1.0)

    has_label = _numeric_series(out, "has_next_season_target").fillna(0.0).clip(lower=0.0, upper=1.0)
    model_support = (
        0.55 * model_probability.fillna(0.5).clip(lower=0.0, upper=1.0)
        + 0.45 * np.where(has_label > 0, 1.0, 0.55)
    )
    confidence = (
        0.35 * sample_adequacy
        + 0.25 * provider_quality
        + 0.20 * context_coverage
        + 0.20 * model_support
    ).clip(lower=0.0, upper=1.0)

    out["future_sample_adequacy_score"] = sample_adequacy * 100.0
    out["future_data_quality_score"] = provider_quality * 100.0
    out["future_context_coverage_score"] = context_coverage * 100.0
    out["future_model_support_score"] = model_support * 100.0
    out["future_potential_confidence"] = confidence * 100.0
    return out


def _enrich_talent_scores(frame: pd.DataFrame, *, base_rank_col: str) -> pd.DataFrame:
    out = _augment_base_columns(frame)
    specs = _extended_family_specs(out)
    for family in FAMILY_ORDER:
        score, coverage = _score_family(out, specs[family])
        out[f"talent_{family}_score"] = score
        out[f"talent_{family}_coverage"] = coverage * 100.0

    out["future_family_weighted_score"] = _weighted_family_score(out)
    base_rank_pct = _rank_percentile(_numeric_series(out, base_rank_col)).clip(lower=0.0, upper=1.0)
    prob = _numeric_series(out, "future_growth_probability").fillna(0.5).clip(lower=0.0, upper=1.0)
    family_component = (_numeric_series(out, "future_family_weighted_score").fillna(50.0) / 100.0).clip(lower=0.0, upper=1.0)

    out["future_scout_score"] = (0.65 * family_component + 0.35 * base_rank_pct).clip(lower=0.0, upper=1.0)
    out["future_scout_blend_score"] = (
        0.55 * prob
        + 0.30 * family_component
        + 0.15 * base_rank_pct
    ).clip(lower=0.0, upper=1.0)
    out["future_potential_score"] = (out["future_scout_blend_score"] * 100.0).clip(lower=0.0, upper=100.0)
    out = _future_confidence(out, model_probability=prob)
    return out


def build_future_scout_score(
    *,
    val_predictions_path: str,
    test_predictions_path: str | None,
    out_val_path: str,
    out_test_path: str | None,
    diagnostics_out: str,
    future_targets_path: str | None = None,
    dataset_path: str | None = None,
    min_next_minutes: float = 450.0,
    min_minutes: float = 900.0,
    max_age: float | None = None,
    positions: set[str] | None = None,
    include_leagues: set[str] | None = None,
    exclude_leagues: set[str] | None = None,
    label_mode: str = "positive_growth",
    k_eval: int = 25,
) -> dict[str, Any]:
    if label_mode not in LABEL_MAP:
        raise ValueError(f"Unknown label_mode '{label_mode}'. Expected one of: {sorted(LABEL_MAP)}")
    label_col = LABEL_MAP[label_mode]

    targets, target_meta = load_future_targets_frame(
        future_targets_path=future_targets_path,
        dataset_path=dataset_path,
        min_next_minutes=min_next_minutes,
    )
    trajectory_frame = _build_trajectory_frame(dataset_path)

    def _attach_enrichment(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        joined, join_meta = attach_future_targets(frame, targets)
        if not trajectory_frame.empty and {"player_id", "season"}.issubset(joined.columns):
            joined = joined.merge(
                trajectory_frame,
                on=["player_id", "season"],
                how="left",
            )
        return joined, join_meta

    val_frame = pd.read_csv(val_predictions_path, low_memory=False)
    val_joined, val_join_meta = _attach_enrichment(val_frame)
    train_rows = _prepare_rows(
        val_joined,
        min_minutes=min_minutes,
        max_age=max_age,
        positions=positions,
        include_leagues=include_leagues,
        exclude_leagues=exclude_leagues,
    )
    train_rows = train_rows[_numeric_series(train_rows, "has_next_season_target") == 1].copy()
    train_rows["_label"] = _numeric_series(train_rows, label_col)
    train_rows = train_rows[train_rows["_label"].notna()].copy()
    if train_rows["_label"].nunique() < 2:
        raise ValueError("Future scout score needs both positive and negative labeled rows in the training split.")

    base_rank_col = _resolve_base_rank_col(val_joined)
    train_rows = _enrich_talent_scores(train_rows, base_rank_col=base_rank_col)

    numeric_features, categorical_features = _feature_lists(train_rows)
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    model = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    feature_cols = [*numeric_features, *categorical_features]
    model.fit(train_rows[feature_cols], train_rows["_label"].astype(int))

    def _apply(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        joined, join_meta = _attach_enrichment(frame)
        joined = _enrich_talent_scores(joined, base_rank_col=base_rank_col)
        prob = model.predict_proba(joined[feature_cols])[:, 1]
        joined["future_growth_probability"] = prob
        joined = _enrich_talent_scores(joined, base_rank_col=base_rank_col)
        return joined, join_meta

    val_scored, val_scored_join = _apply(val_frame)
    val_labeled = _prepare_rows(
        val_scored,
        min_minutes=min_minutes,
        max_age=max_age,
        positions=positions,
        include_leagues=include_leagues,
        exclude_leagues=exclude_leagues,
    )
    val_labeled = val_labeled[_numeric_series(val_labeled, "has_next_season_target") == 1].copy()
    val_labeled["_label"] = _numeric_series(val_labeled, label_col)
    val_labeled = val_labeled[val_labeled["_label"].notna()].copy()

    out_val = Path(out_val_path)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    val_scored.to_csv(out_val, index=False)

    test_payload: dict[str, Any] | None = None
    if test_predictions_path and out_test_path:
        test_frame = pd.read_csv(test_predictions_path, low_memory=False)
        test_scored, test_scored_join = _apply(test_frame)
        out_test = Path(out_test_path)
        out_test.parent.mkdir(parents=True, exist_ok=True)
        test_scored.to_csv(out_test, index=False)
        test_payload = {
            "predictions_path": test_predictions_path,
            "output_path": str(out_test),
            "join": test_scored_join,
            "position_family_counts": {
                key: int(value)
                for key, value in test_scored["talent_position_family"].astype(str).value_counts().sort_index().items()
            }
            if "talent_position_family" in test_scored.columns
            else {},
        }

    y_true = val_labeled["_label"].astype(int)
    y_pred = pd.to_numeric(val_labeled["future_growth_probability"], errors="coerce")
    family_distribution = {
        key: int(value)
        for key, value in train_rows["talent_position_family"].astype(str).value_counts().sort_index().items()
    }
    confidence_series = _numeric_series(val_scored, "future_potential_confidence")
    diagnostics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_source": target_meta,
        "label_mode": label_mode,
        "label_column": label_col,
        "training_filters": {
            "min_minutes": float(min_minutes),
            "max_age": None if max_age is None else float(max_age),
            "positions": sorted(positions) if positions else [],
            "include_leagues": sorted(include_leagues) if include_leagues else [],
            "exclude_leagues": sorted(exclude_leagues) if exclude_leagues else [],
        },
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "base_rank_column": base_rank_col,
            "family_columns": [f"talent_{family}_score" for family in FAMILY_ORDER],
        },
        "position_family_weights": POSITION_FAMILY_WEIGHTS,
        "training_rows": int(len(train_rows)),
        "training_positive_rate": float(y_true.mean()),
        "training_position_family_counts": family_distribution,
        "val_predictions_path": val_predictions_path,
        "val_output_path": str(out_val),
        "val_join": val_join_meta,
        "val_scored_join": val_scored_join,
        "val_metrics": {
            "roc_auc": float(roc_auc_score(y_true, y_pred)) if y_true.nunique() >= 2 else None,
            "average_precision": float(average_precision_score(y_true, y_pred)),
            "precision_at_k_probability": _precision_at_k(val_labeled, score_col="future_growth_probability", label_col="_label", k=k_eval),
            "precision_at_k_blend": _precision_at_k(val_labeled, score_col="future_scout_blend_score", label_col="_label", k=k_eval),
            "precision_at_k_base_rank": _precision_at_k(val_labeled, score_col=base_rank_col, label_col="_label", k=k_eval),
            "precision_at_k_future_potential": _precision_at_k(val_labeled, score_col="future_potential_score", label_col="_label", k=k_eval),
            "k_eval": int(k_eval),
        },
        "future_talent_summary": {
            "future_label_coverage": {
                "labeled_rows": int(_numeric_series(val_scored, "has_next_season_target").fillna(0.0).sum()),
                "total_rows": int(len(val_scored)),
            },
            "position_family_counts": {
                key: int(value)
                for key, value in val_scored["talent_position_family"].astype(str).value_counts().sort_index().items()
            }
            if "talent_position_family" in val_scored.columns
            else {},
            "future_potential_confidence_distribution": {
                "high": int((confidence_series >= 70.0).sum()),
                "medium": int(((confidence_series >= 45.0) & (confidence_series < 70.0)).sum()),
                "low": int((confidence_series < 45.0).sum()),
            },
        },
        "top_coefficients": _top_coefficients(model),
        "test": test_payload,
    }

    diagnostics_path = Path(diagnostics_out)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"[future-score] wrote val scored predictions -> {out_val}")
    if test_payload:
        print(f"[future-score] wrote test scored predictions -> {test_payload['output_path']}")
    print(f"[future-score] wrote diagnostics -> {diagnostics_path}")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a next-season-growth scouting score on labeled validation rows and write enriched prediction files "
            "with future growth probability, family scores, and future potential score."
        )
    )
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-predictions", default=None)
    parser.add_argument("--out-val", required=True)
    parser.add_argument("--out-test", default=None)
    parser.add_argument("--diagnostics-out", required=True)
    parser.add_argument("--future-targets", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--min-next-minutes", type=float, default=450.0)
    parser.add_argument("--min-minutes", type=float, default=900.0)
    parser.add_argument("--max-age", type=float, default=-1.0, help="Set negative to disable.")
    parser.add_argument("--positions", default="")
    parser.add_argument("--include-leagues", default="")
    parser.add_argument("--exclude-leagues", default="")
    parser.add_argument("--label-mode", default="positive_growth", choices=sorted(LABEL_MAP))
    parser.add_argument("--k-eval", type=int, default=25)
    args = parser.parse_args()

    build_future_scout_score(
        val_predictions_path=args.val_predictions,
        test_predictions_path=args.test_predictions,
        out_val_path=args.out_val,
        out_test_path=args.out_test,
        diagnostics_out=args.diagnostics_out,
        future_targets_path=args.future_targets,
        dataset_path=args.dataset,
        min_next_minutes=args.min_next_minutes,
        min_minutes=args.min_minutes,
        max_age=None if args.max_age < 0 else args.max_age,
        positions=_parse_positions(args.positions),
        include_leagues={token.strip().casefold() for token in str(args.include_leagues).split(",") if token.strip()} or None,
        exclude_leagues={token.strip().casefold() for token in str(args.exclude_leagues).split(",") if token.strip()} or None,
        label_mode=args.label_mode,
        k_eval=args.k_eval,
    )


if __name__ == "__main__":
    main()
