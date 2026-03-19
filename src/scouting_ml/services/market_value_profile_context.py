"""UI-facing profile/context payload helpers for market-value detail views."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scouting_ml.features.history_strength import (
    HISTORY_COMPONENT_COLUMNS,
    HISTORY_COMPONENT_LABELS,
    HISTORY_COMPONENT_WEIGHTS,
)
from scouting_ml.services.market_value_profile_taxonomy import (
    PROFILE_CONTEXT_FIELDS,
    PROFILE_STAT_GROUP_ORDER,
    PROFILE_STAT_SKIP_FIELDS,
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_eur(value: float | None) -> str:
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


def build_history_strength_payload(row: pd.Series) -> dict[str, Any]:
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
        key=lambda item: item["weighted_points_0_to_100"]
        if isinstance(item["weighted_points_0_to_100"], (float, int))
        else -1.0,
        reverse=True,
    )
    strongest = [item for item in components_sorted if not item["missing"]][:3]
    weakest = sorted(
        [item for item in components if not item["missing"]],
        key=lambda item: item["value_0_to_1"] if isinstance(item["value_0_to_1"], (float, int)) else 1.0,
    )[:3]

    if score is None:
        narrative = "History strength score is unavailable because required history components are missing."
    else:
        narrative = f"History strength is {score:.1f}/100 ({tier_text})."
        if strongest:
            narrative += " Strongest components: " + ", ".join(item["label"] for item in strongest[:2]) + "."
        if weakest:
            narrative += " Development focus: " + ", ".join(item["label"] for item in weakest[:2]) + "."

    return {
        "score_0_to_100": score,
        "coverage_0_to_1": coverage,
        "tier": tier_text,
        "components": components,
        "strongest_components": strongest,
        "improvement_components": weakest,
        "summary_text": narrative,
    }


def _humanize_profile_field(key: str) -> str:
    raw = (
        str(key)
        .replace("sofa_", "")
        .replace("clubctx_", "club ")
        .replace("history_", "history ")
        .replace("prior_", "prior ")
        .replace("_", " ")
        .strip()
    )
    raw = " ".join(raw.split())
    if not raw:
        return str(key)
    words = raw.split(" ")
    return " ".join(word.upper() if word.lower() in {"eur", "xg", "xa"} else word.capitalize() for word in words)


def _is_profile_stat_value(key: str, value: Any) -> bool:
    key_str = str(key)
    if key_str in PROFILE_STAT_SKIP_FIELDS or key_str.startswith("_"):
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return key_str in PROFILE_CONTEXT_FIELDS and bool(value.strip())
    return _safe_float(value) is not None


def _profile_stat_kind(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "text"
    key_str = str(key).lower()
    if key_str.endswith("_eur"):
        return "currency"
    if key_str.endswith("_0_to_1") or "percentile" in key_str:
        return "fraction"
    if "percentage" in key_str:
        return "percentage"
    if "minutes" in key_str or key_str.endswith("_n") or "count" in key_str or "caps" in key_str:
        return "integer"
    return "number"


def _format_profile_stat_value(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, str):
        return value
    num = _safe_float(value)
    if num is None:
        return "n/a"
    kind = _profile_stat_kind(key, value)
    if kind == "currency":
        return format_eur(num)
    if kind == "fraction":
        return f"{num * 100.0:.1f}%"
    if kind == "percentage":
        return f"{num:.1f}%"
    if kind == "integer":
        return f"{int(round(num)):,}"
    return f"{num:.2f}" if abs(num) < 1000 else f"{num:,.0f}"


def _classify_profile_stat_group(key: str) -> str:
    key_str = str(key)
    key_lower = key_str.lower()
    if key_str in PROFILE_CONTEXT_FIELDS:
        return "Profile & Context"
    if key_lower.startswith("sb_"):
        return "External Tactical"
    if key_lower.startswith("fixture_") or key_lower.startswith("odds_"):
        return "Schedule & Market"
    if key_lower.startswith("avail_"):
        return "Availability & Physical"
    if any(
        token in key_lower
        for token in ("market_value", "expected_value", "fair_value", "gap", "confidence", "interval", "calibration", "prior_")
    ):
        return "Value & Model"
    if any(token in key_lower for token in ("goal", "assist", "shot", "xg", "xa", "dribble", "bigchance", "penalty")):
        return "Attacking"
    if any(
        token in key_lower
        for token in ("pass", "cross", "throughball", "longball", "keypass", "progressive", "chancecreated")
    ):
        return "Passing & Progression"
    if any(
        token in key_lower
        for token in ("tackle", "interception", "clearance", "blocked", "duel", "aerial", "recovery", "possessionwon")
    ):
        return "Defending & Duels"
    if any(token in key_lower for token in ("save", "highclaim", "runout", "goalsprevented", "cleansheet")):
        return "Goalkeeping"
    if any(token in key_lower for token in ("age", "minutes", "injury", "height", "weight", "contract", "foot")):
        return "Availability & Physical"
    if key_lower.startswith("clubctx_") or key_lower.startswith("history_") or "coeff" in key_lower:
        return "History & Context"
    return "Other Metrics"


def build_profile_stat_groups(row_payload: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in PROFILE_STAT_GROUP_ORDER}
    for key, value in row_payload.items():
        if not _is_profile_stat_value(key, value):
            continue
        group_name = _classify_profile_stat_group(key)
        grouped[group_name].append(
            {
                "key": str(key),
                "label": _humanize_profile_field(str(key)),
                "value": value,
                "display_value": _format_profile_stat_value(str(key), value),
                "kind": _profile_stat_kind(str(key), value),
            }
        )

    out: list[dict[str, Any]] = []
    for group_name in PROFILE_STAT_GROUP_ORDER:
        items = sorted(grouped[group_name], key=lambda item: str(item["label"]))
        if not items:
            continue
        out.append({"group": group_name, "count": int(len(items)), "items": items})
    return out


def build_external_tactical_context(row: pd.Series) -> dict[str, Any]:
    formations: list[dict[str, Any]] = []
    for key in row.index:
        key_str = str(key)
        if not key_str.startswith("sb_minutes_in_"):
            continue
        minutes = _safe_float(row.get(key_str))
        if minutes is None or minutes <= 0:
            continue
        formations.append({"formation": key_str.replace("sb_minutes_in_", ""), "minutes": minutes})
    formations.sort(key=lambda item: float(item["minutes"]), reverse=True)

    metrics = [
        ("Progressive passes/90", _safe_float(row.get("sb_progressive_passes_per90"))),
        ("Progressive carries/90", _safe_float(row.get("sb_progressive_carries_per90"))),
        ("Passes into box/90", _safe_float(row.get("sb_passes_into_box_per90"))),
        ("Shot assists/90", _safe_float(row.get("sb_shot_assists_per90"))),
        ("Pressures/90", _safe_float(row.get("sb_pressures_per90"))),
        ("Counterpressures/90", _safe_float(row.get("sb_counterpressures_per90"))),
        ("High regains/90", _safe_float(row.get("sb_high_regains_per90"))),
        ("Duel win rate", _safe_float(row.get("sb_duel_win_rate"))),
        ("Aerial win rate", _safe_float(row.get("sb_aerial_win_rate"))),
    ]
    signals = [
        {
            "label": label,
            "value": value,
            "display_value": f"{value * 100.0:.1f}%" if "rate" in label.lower() else f"{value:.2f}",
        }
        for label, value in metrics
        if value is not None
    ]
    if not formations and not signals:
        return {
            "available": False,
            "summary_text": "No external tactical provider signals available.",
            "preferred_formations": [],
            "signals": [],
        }

    if formations:
        top = formations[0]
        summary = f"StatsBomb profile leans toward {top['formation']} ({top['minutes']:.0f} tracked minutes)."
    else:
        summary = "StatsBomb tactical metrics available without formation exposure."

    return {
        "available": True,
        "summary_text": summary,
        "preferred_formations": formations[:4],
        "signals": signals[:6],
    }


def build_availability_context(row: pd.Series) -> dict[str, Any]:
    reports = _safe_float(row.get("avail_reports"))
    start_share = _safe_float(row.get("avail_start_share"))
    bench_share = _safe_float(row.get("avail_bench_share"))
    injury_count = _safe_float(row.get("avail_injury_count"))
    suspension_count = _safe_float(row.get("avail_suspension_count"))
    expected_start_rate = _safe_float(row.get("avail_expected_start_rate"))

    signals = []
    for label, value, fmt in [
        ("Availability reports", reports, "count"),
        ("Start share", start_share, "pct"),
        ("Bench share", bench_share, "pct"),
        ("Injury reports", injury_count, "count"),
        ("Suspension reports", suspension_count, "count"),
        ("Expected start rate", expected_start_rate, "pct"),
    ]:
        if value is None:
            continue
        display = f"{value * 100.0:.1f}%" if fmt == "pct" else f"{value:.0f}"
        signals.append({"label": label, "value": value, "display_value": display})

    if not signals:
        return {"available": False, "summary_text": "No provider availability signals available.", "signals": []}

    summary_parts: list[str] = []
    if start_share is not None:
        summary_parts.append(f"start share {start_share * 100.0:.0f}%")
    if injury_count is not None:
        summary_parts.append(f"{injury_count:.0f} injury reports")
    if expected_start_rate is not None:
        summary_parts.append(f"expected-start rate {expected_start_rate * 100.0:.0f}%")
    summary = (
        "Availability context: " + ", ".join(summary_parts) + "."
        if summary_parts
        else "Availability provider signals available."
    )
    return {"available": True, "summary_text": summary, "signals": signals}


def build_market_context_payload(row: pd.Series) -> dict[str, Any]:
    signals = []
    for label, key, fmt in [
        ("Fixture matches", "fixture_matches", "count"),
        ("Mean rest days", "fixture_mean_rest_days", "num"),
        ("Congestion share", "fixture_congestion_share", "pct"),
        ("Opponent strength", "fixture_opponent_strength", "num"),
        ("Team market strength", "odds_implied_team_strength", "pct"),
        ("Opponent market strength", "odds_implied_opponent_strength", "pct"),
        ("Upset probability", "odds_upset_probability", "pct"),
        ("Expected total goals", "odds_expected_total_goals", "num"),
    ]:
        value = _safe_float(row.get(key))
        if value is None:
            continue
        if fmt == "pct":
            display = f"{value * 100.0:.1f}%"
        elif fmt == "count":
            display = f"{value:.0f}"
        else:
            display = f"{value:.2f}"
        signals.append({"label": label, "value": value, "display_value": display})

    if not signals:
        return {"available": False, "summary_text": "No fixture or market-context signals available.", "signals": []}

    congestion = _safe_float(row.get("fixture_congestion_share"))
    team_strength = _safe_float(row.get("odds_implied_team_strength"))
    summary_bits: list[str] = []
    if congestion is not None:
        summary_bits.append(f"schedule congestion {congestion * 100.0:.0f}%")
    if team_strength is not None:
        summary_bits.append(f"market win strength {team_strength * 100.0:.0f}%")
    summary = (
        "Schedule + market context: " + ", ".join(summary_bits) + "."
        if summary_bits
        else "Fixture and market signals available."
    )
    return {"available": True, "summary_text": summary, "signals": signals}


def build_provider_coverage(row: pd.Series) -> dict[str, Any]:
    tactical = build_external_tactical_context(row)
    availability = build_availability_context(row)
    market = build_market_context_payload(row)
    return {
        "statsbomb": bool(tactical.get("available")),
        "availability_provider": bool(availability.get("available")),
        "market_provider": bool(market.get("available")),
    }


def build_provider_risk_flags(row: pd.Series) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    injury_count = _safe_float(row.get("avail_injury_count"))
    if injury_count is not None and injury_count >= 3:
        flags.append(
            {
                "severity": "medium",
                "code": "provider_injury_load",
                "message": f"External availability feed shows {injury_count:.0f} injury reports this season.",
            }
        )
    start_share = _safe_float(row.get("avail_start_share"))
    reports = _safe_float(row.get("avail_reports"))
    if reports is not None and reports >= 3 and start_share is not None and start_share < 0.5:
        flags.append(
            {
                "severity": "medium",
                "code": "provider_rotation_risk",
                "message": f"External lineup feed shows only {start_share * 100.0:.0f}% starts across tracked reports.",
            }
        )
    congestion = _safe_float(row.get("fixture_congestion_share"))
    if congestion is not None and congestion >= 0.4:
        flags.append(
            {
                "severity": "low",
                "code": "provider_schedule_congestion",
                "message": f"Fixture context indicates high congestion ({congestion * 100.0:.0f}% of matches on short rest).",
            }
        )
    return flags


__all__ = [
    "build_availability_context",
    "build_external_tactical_context",
    "build_history_strength_payload",
    "build_market_context_payload",
    "build_profile_stat_groups",
    "build_provider_coverage",
    "build_provider_risk_flags",
    "format_eur",
]
