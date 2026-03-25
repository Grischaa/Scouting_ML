from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from scouting_ml.scouting.role_templates import get_role_template
from scouting_ml.scouting.system_profiles import (
    SUPPORTED_ACTIVE_LANES,
    SUPPORTED_TRUST_SCOPES,
    get_system_profile,
    list_system_profiles,
)

ActiveLane = Literal["valuation", "future_shortlist"]
TrustScope = Literal["trusted_only", "trusted_and_watch", "all"]

TRUST_TIER_PENALTIES: dict[str, float] = {
    "trusted": 0.0,
    "watch": 10.0,
    "unknown": 15.0,
    "blocked": 25.0,
}

TRUST_SCOPE_ALLOWED: dict[str, set[str]] = {
    "trusted_only": {"trusted"},
    "trusted_and_watch": {"trusted", "watch"},
    "all": {"trusted", "watch", "blocked", "unknown"},
}


def build_lane_posture(active_lane: str) -> dict[str, Any]:
    if str(active_lane or "").strip() == "future_shortlist":
        return {
            "role": "future_shortlist",
            "lane_state": "live",
            "label": "Future Potential / Advisory",
            "summary": "Advisory current-season system fit leaning on the live future_shortlist lane.",
        }
    return {
        "role": "valuation",
        "lane_state": "stable",
        "label": "Current Level / Pricing",
        "summary": "Stable pricing-oriented system fit leaning on the benchmarked valuation lane.",
    }


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _coerce_confidence_threshold(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed <= 1.0:
        return max(parsed, 0.0) * 100.0
    return max(parsed, 0.0)


def _normalize_trust_scope(value: str | None) -> TrustScope:
    token = str(value or "trusted_and_watch").strip().lower()
    if token not in SUPPORTED_TRUST_SCOPES:
        raise ValueError(f"Unsupported trust_scope: {value}")
    return token  # type: ignore[return-value]


def _normalize_active_lane(value: str | None) -> ActiveLane:
    token = str(value or "valuation").strip().lower()
    if token not in SUPPORTED_ACTIVE_LANES:
        raise ValueError(f"Unsupported active_lane: {value}")
    return token  # type: ignore[return-value]


def _family_score(row: pd.Series, family: str) -> float:
    score_families = row.get("score_families")
    if isinstance(score_families, dict):
        value = _safe_float(score_families.get(family))
        if value is not None:
            return float(np.clip(value, 0.0, 100.0))
    value = _safe_float(row.get(f"talent_{family}_score"))
    if value is None:
        return 50.0
    return float(np.clip(value, 0.0, 100.0))


def _family_coverage(row: pd.Series, family: str) -> float:
    value = _safe_float(row.get(f"talent_{family}_coverage"))
    if value is None:
        return 0.0
    if value <= 1.0:
        value *= 100.0
    return float(np.clip(value, 0.0, 100.0))


def _metric_series_percentile(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    series = pd.to_numeric(frame[metric], errors="coerce")
    if series.notna().sum() < 3:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    ranked = series.rank(method="average", pct=True)
    if "injury" in metric.lower():
        ranked = 1.0 - ranked
    return ranked.clip(lower=0.0, upper=1.0)


def _style_percentile_map(frame: pd.DataFrame, metric_weights: dict[str, float]) -> dict[str, pd.Series]:
    return {metric: _metric_series_percentile(frame, metric) for metric in metric_weights}


def _resolve_budget_status(
    *,
    market_value_eur: float | None,
    fair_value_eur: float | None,
    budget_eur: float | None,
) -> str:
    if budget_eur is None or budget_eur <= 0:
        return "unbounded"
    if market_value_eur is not None and market_value_eur <= budget_eur:
        return "within_budget"
    stretch_ceiling = 1.35 * budget_eur
    if market_value_eur is not None and market_value_eur <= stretch_ceiling:
        return "stretch"
    if fair_value_eur is not None and fair_value_eur <= budget_eur and market_value_eur is not None and market_value_eur <= (1.10 * stretch_ceiling):
        return "stretch"
    return "unrealistic"


def _affordability_score(
    *,
    budget_status: str,
    budget_eur: float | None,
    fair_value_eur: float | None,
    value_gap_conservative_eur: float | None,
) -> float:
    base = {
        "within_budget": 100.0,
        "stretch": 65.0,
        "unrealistic": 20.0,
        "unbounded": 55.0,
    }.get(str(budget_status), 55.0)
    if value_gap_conservative_eur is not None and value_gap_conservative_eur > 0:
        base += min(12.0, value_gap_conservative_eur / 1_500_000.0)
    if budget_eur is not None and budget_eur > 0 and fair_value_eur is not None and fair_value_eur <= budget_eur:
        base += 8.0
    return float(np.clip(base, 0.0, 100.0))


def _role_fit_score(row: pd.Series, family_weights: dict[str, float]) -> tuple[float, list[dict[str, Any]], float]:
    contributions: list[dict[str, Any]] = []
    coverage_weighted = 0.0
    total_weight = 0.0
    for family, weight in family_weights.items():
        family_score = _family_score(row, family)
        family_coverage = _family_coverage(row, family)
        contribution = family_score * float(weight)
        coverage_weighted += family_coverage * float(weight)
        total_weight += float(weight)
        contributions.append(
            {
                "family": family,
                "score": round(family_score, 2),
                "coverage": round(family_coverage, 2),
                "weight": round(float(weight), 4),
                "contribution": round(contribution, 2),
            }
        )
    contributions.sort(key=lambda item: item["contribution"], reverse=True)
    score = sum(item["contribution"] for item in contributions)
    coverage = coverage_weighted / total_weight if total_weight > 0 else 0.0
    return float(np.clip(score, 0.0, 100.0)), contributions[:2], float(np.clip(coverage, 0.0, 100.0))


def _style_fit_score(
    row: pd.Series,
    *,
    metric_weights: dict[str, float],
    percentile_map: dict[str, pd.Series],
) -> tuple[float, list[dict[str, Any]], float]:
    parts: list[dict[str, Any]] = []
    weighted_score = 0.0
    available_weight = 0.0
    total_weight = float(sum(metric_weights.values()) or 1.0)
    for metric, weight in metric_weights.items():
        series = percentile_map.get(metric)
        if series is None or row.name not in series.index:
            continue
        pct = _safe_float(series.loc[row.name])
        value = _safe_float(row.get(metric))
        if pct is None or value is None:
            continue
        weighted_score += pct * 100.0 * float(weight)
        available_weight += float(weight)
        parts.append(
            {
                "metric": metric,
                "weight": round(float(weight), 4),
                "score": round(pct * 100.0, 2),
                "value": round(value, 4),
            }
        )
    parts.sort(key=lambda item: item["score"] * item["weight"], reverse=True)
    score = weighted_score / available_weight if available_weight > 0 else 50.0
    coverage = (available_weight / total_weight) * 100.0 if total_weight > 0 else 0.0
    return float(np.clip(score, 0.0, 100.0)), parts[:2], float(np.clip(coverage, 0.0, 100.0))


def _system_fit_confidence(
    *,
    row: pd.Series,
    active_lane: ActiveLane,
    family_coverage: float,
    style_coverage: float,
) -> float:
    lane_conf_col = "future_potential_confidence" if active_lane == "future_shortlist" else "current_level_confidence"
    lane_conf = _safe_float(row.get(lane_conf_col)) or 0.0
    trust_tier = str(row.get("league_trust_tier") or "unknown").strip().lower()
    penalty = TRUST_TIER_PENALTIES.get(trust_tier, TRUST_TIER_PENALTIES["unknown"])
    confidence = (0.60 * lane_conf) + (0.25 * family_coverage) + (0.15 * style_coverage) - penalty
    return float(np.clip(confidence, 0.0, 100.0))


def _active_lane_score(row: pd.Series, active_lane: ActiveLane) -> float:
    col = "future_potential_score" if active_lane == "future_shortlist" else "current_level_score"
    return float(np.clip(_safe_float(row.get(col)) or 0.0, 0.0, 100.0))


def _fit_reasons(
    *,
    family_parts: list[dict[str, Any]],
    style_parts: list[dict[str, Any]],
    budget_status: str,
    league_trust_tier: str,
    league_adjustment_bucket: str | None = None,
    league_adjustment_reason: str | None = None,
) -> list[str]:
    reasons: list[str] = []
    if family_parts:
        family_text = ", ".join(
            f"{str(entry['family']).title()} {entry['score']:.0f}" for entry in family_parts[:2]
        )
        reasons.append(f"Talent-family match: {family_text}.")
    if style_parts:
        style_text = ", ".join(
            f"{entry['metric']} {entry['score']:.0f}" for entry in style_parts[:2]
        )
        reasons.append(f"System style match: {style_text}.")
    reasons.append(f"Budget posture: {budget_status.replace('_', ' ')}.")
    if league_trust_tier in {"watch", "blocked", "unknown"}:
        reasons.append(f"League trust: {league_trust_tier}.")
    if str(league_adjustment_bucket or "").strip().lower() in {"weak", "failed", "severe_failed"}:
        reasons.append(str(league_adjustment_reason or "Pricing adjusted for league reliability."))
    return reasons[:4]


def _recruitment_value_flag(
    *,
    system_fit_score: float,
    system_fit_confidence: float,
    budget_status: str,
) -> bool:
    return bool(system_fit_score >= 65.0 and system_fit_confidence >= 45.0 and budget_status in {"within_budget", "stretch", "unbounded"})


def _serialize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    out = dict(profile)
    out["slots"] = [dict(slot) for slot in profile.get("slots", [])]
    return out


def list_system_fit_templates() -> dict[str, Any]:
    profiles = [_serialize_profile(profile) for profile in list_system_profiles()]
    return {
        "default_template_key": profiles[0]["template_key"] if profiles else None,
        "supported_active_lanes": list(SUPPORTED_ACTIVE_LANES),
        "supported_trust_scopes": list(SUPPORTED_TRUST_SCOPES),
        "templates": profiles,
    }


def rank_system_fit_slots(
    frame: pd.DataFrame,
    *,
    template_key: str,
    active_lane: str = "valuation",
    top_n_per_slot: int = 10,
    slot_role_overrides: dict[str, str] | None = None,
    budget_eur: float | None = None,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    lane = _normalize_active_lane(active_lane)
    profile = get_system_profile(template_key)
    overrides = {str(key): str(value) for key, value in (slot_role_overrides or {}).items()}
    frame = frame.copy()
    top_n = max(int(top_n_per_slot), 1)
    confidence_threshold = _coerce_confidence_threshold(min_confidence)
    percentile_map = _style_percentile_map(frame, profile.get("style_metric_weights", {}))

    slots: list[dict[str, Any]] = []
    for slot in profile.get("slots", []):
        slot_key = str(slot.get("slot_key") or "")
        role_key = overrides.get(slot_key) or str(slot.get("default_role_key") or "")
        role_template = get_role_template(role_key)
        allowed_families = {str(item).upper() for item in role_template.get("allowed_position_families", [])}
        candidates = frame[frame["talent_position_family"].astype(str).str.upper().isin(allowed_families)].copy()
        ranked_items: list[dict[str, Any]] = []
        for _, row in candidates.iterrows():
            role_fit, family_parts, family_coverage = _role_fit_score(row, role_template.get("family_weights", {}))
            style_fit, style_parts, style_coverage = _style_fit_score(
                row,
                metric_weights=profile.get("style_metric_weights", {}),
                percentile_map=percentile_map,
            )
            active_score = _active_lane_score(row, lane)
            market_value = _safe_float(row.get("market_value_eur"))
            fair_value = _safe_float(row.get("fair_value_eur")) or _safe_float(row.get("expected_value_eur"))
            value_gap = _safe_float(row.get("value_gap_conservative_eur")) or _safe_float(row.get("value_gap_eur"))
            budget_status = _resolve_budget_status(
                market_value_eur=market_value,
                fair_value_eur=fair_value,
                budget_eur=budget_eur,
            )
            affordability = _affordability_score(
                budget_status=budget_status,
                budget_eur=budget_eur,
                fair_value_eur=fair_value,
                value_gap_conservative_eur=value_gap,
            )
            confidence = _system_fit_confidence(
                row=row,
                active_lane=lane,
                family_coverage=family_coverage,
                style_coverage=style_coverage,
            )
            if confidence_threshold is not None and confidence < confidence_threshold:
                continue
            base_score = (0.45 * role_fit) + (0.25 * active_score) + (0.15 * style_fit) + (0.15 * affordability)
            raw_final_score = base_score * (0.70 + (0.30 * confidence / 100.0))
            discovery_weight = _safe_float(row.get("discovery_reliability_weight")) or 1.0
            ranking_multiplier = 0.85 + (0.15 * float(np.clip(discovery_weight, 0.0, 1.0)))
            final_score = raw_final_score * ranking_multiplier
            trust_tier = str(row.get("league_trust_tier") or "unknown").strip().lower()
            league_adjustment_bucket = str(row.get("league_adjustment_bucket") or "").strip().lower()
            league_adjustment_reason = str(row.get("league_adjustment_reason") or "").strip()
            item = row.to_dict()
            item.update(
                {
                    "slot_key": slot_key,
                    "slot_label": str(slot.get("slot_label") or slot_key),
                    "role_template_key": role_key,
                    "role_template_label": str(role_template.get("label") or role_key),
                    "system_fit_score": round(float(np.clip(final_score, 0.0, 100.0)), 2),
                    "system_fit_score_raw": round(float(np.clip(raw_final_score, 0.0, 100.0)), 2),
                    "system_fit_confidence": round(float(np.clip(confidence, 0.0, 100.0)), 2),
                    "role_fit_score": round(float(np.clip(role_fit, 0.0, 100.0)), 2),
                    "style_fit_score": round(float(np.clip(style_fit, 0.0, 100.0)), 2),
                    "affordability_fit_score": round(float(np.clip(affordability, 0.0, 100.0)), 2),
                    "discovery_reliability_weight": round(float(np.clip(discovery_weight, 0.0, 1.0)), 4),
                    "budget_status": budget_status,
                    "recruitment_value_flag": _recruitment_value_flag(
                        system_fit_score=final_score,
                        system_fit_confidence=confidence,
                        budget_status=budget_status,
                    ),
                    "active_lane_score": round(float(np.clip(active_score, 0.0, 100.0)), 2),
                    "league_trust_tier": trust_tier,
                    "fit_reasons": _fit_reasons(
                        family_parts=family_parts,
                        style_parts=style_parts,
                        budget_status=budget_status,
                        league_trust_tier=trust_tier,
                        league_adjustment_bucket=league_adjustment_bucket,
                        league_adjustment_reason=league_adjustment_reason,
                    ),
                }
            )
            ranked_items.append(item)

        ranked_items.sort(
            key=lambda item: (
                float(item.get("system_fit_score") or 0.0),
                float(item.get("system_fit_confidence") or 0.0),
                float(item.get("active_lane_score") or 0.0),
                float(item.get("affordability_fit_score") or 0.0),
            ),
            reverse=True,
        )
        slots.append(
            {
                "slot_key": slot_key,
                "slot_label": str(slot.get("slot_label") or slot_key),
                "role_template_key": role_key,
                "role_template_label": str(role_template.get("label") or role_key),
                "result_count": int(len(ranked_items)),
                "items": ranked_items[:top_n],
            }
        )

    return {
        "system_profile": _serialize_profile(profile),
        "active_lane": lane,
        "lane_posture": build_lane_posture(lane),
        "slots": slots,
    }


__all__ = [
    "TRUST_SCOPE_ALLOWED",
    "TRUST_TIER_PENALTIES",
    "build_lane_posture",
    "list_system_fit_templates",
    "rank_system_fit_slots",
]
