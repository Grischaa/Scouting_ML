from __future__ import annotations

import numpy as np
import pandas as pd

# The feature names are intentionally stable so downstream scripts/API can rely on them.
HISTORY_COMPONENT_COLUMNS: tuple[str, ...] = (
    "history_stability_component",
    "history_transfer_signal_component",
    "history_contract_component",
    "history_injury_component",
    "history_national_team_component",
    "history_minutes_component",
    "history_minutes_momentum_component",
    "history_performance_momentum_component",
)

HISTORY_COMPONENT_LABELS: dict[str, str] = {
    "history_stability_component": "Stability (low transfer churn)",
    "history_transfer_signal_component": "Transfer signal quality",
    "history_contract_component": "Contract runway",
    "history_injury_component": "Injury resilience",
    "history_national_team_component": "National-team track record",
    "history_minutes_component": "Minutes reliability",
    "history_minutes_momentum_component": "Minutes momentum",
    "history_performance_momentum_component": "Performance momentum",
}

HISTORY_COMPONENT_WEIGHTS: dict[str, float] = {
    "history_stability_component": 0.18,
    "history_transfer_signal_component": 0.10,
    "history_contract_component": 0.12,
    "history_injury_component": 0.16,
    "history_national_team_component": 0.10,
    "history_minutes_component": 0.16,
    "history_minutes_momentum_component": 0.08,
    "history_performance_momentum_component": 0.10,
}


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)


def _history_tier(score: pd.Series, coverage: pd.Series) -> pd.Series:
    out = pd.Series("uncertain", index=score.index, dtype="object")
    enough = coverage >= 0.35
    out = out.where(~(enough & (score >= 75.0)), "elite")
    out = out.where(~(enough & (score >= 60.0) & (score < 75.0)), "strong")
    out = out.where(~(enough & (score >= 45.0) & (score < 60.0)), "developing")
    return out


def add_history_strength_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-season historical reliability/momentum features.

    The score is designed as a stable meta-feature:
    - range: 0..100
    - neutral handling of missing data via component coverage
    - no direct market-value fields (safe for leakage guard)
    """
    out = frame.copy()

    minutes = _num(out, "minutes")
    if minutes.isna().all():
        minutes = _num(out, "sofa_minutesPlayed")

    transfer_count_3y = _num(out, "transfer_count_3y").clip(lower=0.0)
    transfer_loans_3y = _num(out, "transfer_loans_3y").clip(lower=0.0)
    transfer_paid_moves_3y = _num(out, "transfer_paid_moves_3y").clip(lower=0.0)
    transfer_total_fees_3y = _num(out, "transfer_total_fees_3y_eur").clip(lower=0.0)
    transfer_max_fee = _num(out, "transfer_max_fee_career_eur").clip(lower=0.0)

    contract_years_left = _num(out, "contract_years_left")
    injury_days_per_1000_min = _num(out, "injury_days_per_1000_min")
    nt_total_caps = _num(out, "nt_total_caps")
    if nt_total_caps.isna().all():
        nt_total_caps = _num(out, "nt_senior_caps")

    delta_minutes = _num(out, "delta_minutes")
    if delta_minutes.isna().all() and "prev_minutes" in out.columns:
        delta_minutes = minutes - _num(out, "prev_minutes")

    delta_xg90 = _num(out, "delta_sofa_expectedGoals_per90")
    delta_ast90 = _num(out, "delta_sofa_assists_per90")
    delta_g90 = _num(out, "delta_sofa_goals_per90")

    # Fewer recent moves and fewer loans generally means more stable development context.
    move_load = transfer_count_3y + (0.5 * transfer_loans_3y)
    stability_component = 1.0 - _clip01(move_load / 4.0)

    paid_share = _safe_ratio(transfer_paid_moves_3y, transfer_count_3y).fillna(0.0)
    fee_recent_signal = _clip01(np.log1p(transfer_total_fees_3y) / np.log1p(120_000_000.0))
    fee_career_signal = _clip01(np.log1p(transfer_max_fee) / np.log1p(150_000_000.0))
    transfer_signal_component = (0.45 * paid_share) + (0.30 * fee_recent_signal) + (0.25 * fee_career_signal)

    contract_component = _clip01(contract_years_left / 4.0)
    injury_component = 1.0 - _clip01(injury_days_per_1000_min / 180.0)
    nt_component = _clip01(np.log1p(nt_total_caps) / np.log1p(80.0))
    minutes_component = _clip01(minutes / 2400.0)
    minutes_momentum_component = _clip01((delta_minutes + 900.0) / 1800.0)

    perf_momentum_signal = (0.45 * delta_xg90) + (0.30 * delta_ast90) + (0.25 * delta_g90)
    perf_momentum_component = _clip01((perf_momentum_signal + 0.12) / 0.24)

    components = pd.DataFrame(
        {
            "history_stability_component": stability_component,
            "history_transfer_signal_component": transfer_signal_component,
            "history_contract_component": contract_component,
            "history_injury_component": injury_component,
            "history_national_team_component": nt_component,
            "history_minutes_component": minutes_component,
            "history_minutes_momentum_component": minutes_momentum_component,
            "history_performance_momentum_component": perf_momentum_component,
        },
        index=out.index,
    )

    coverage = components.notna().mean(axis=1)
    weighted = pd.Series(0.0, index=out.index, dtype=float)
    for col, weight in HISTORY_COMPONENT_WEIGHTS.items():
        weighted += components[col].fillna(0.5) * float(weight)

    score = 100.0 * weighted * (0.65 + 0.35 * coverage)
    score = score.clip(lower=0.0, upper=100.0)

    out[list(HISTORY_COMPONENT_COLUMNS)] = components
    out["history_strength_coverage"] = coverage.astype(float)
    out["history_strength_score"] = score.astype(float)
    out["history_strength_tier"] = _history_tier(score=out["history_strength_score"], coverage=out["history_strength_coverage"])
    return out
