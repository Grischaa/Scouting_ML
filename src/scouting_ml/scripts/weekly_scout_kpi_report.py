from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


BIG5_LEAGUES = {
    "premier league",
    "la liga",
    "laliga",
    "serie a",
    "bundesliga",
    "ligue 1",
}


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _parse_int_tokens(raw: str | None, *, default: Sequence[int]) -> list[int]:
    tokens = _parse_csv_tokens(raw)
    if not tokens:
        return list(default)
    out: list[int] = []
    for token in tokens:
        try:
            value = int(float(token))
        except (TypeError, ValueError):
            continue
        if value > 0:
            out.append(value)
    return sorted({*out}) if out else list(default)


def _resolve_position_col(frame: pd.DataFrame) -> str | None:
    for col in ("model_position", "position_group", "position_main"):
        if col in frame.columns:
            return col
    return None


def _value_segment_series(frame: pd.DataFrame) -> pd.Series:
    if "value_segment" in frame.columns:
        return frame["value_segment"].astype(str)
    market = pd.to_numeric(frame.get("market_value_eur"), errors="coerce")
    out = pd.Series("unknown", index=frame.index, dtype=object)
    out.loc[(market >= 0.0) & (market < 5_000_000.0)] = "under_5m"
    out.loc[(market >= 5_000_000.0) & (market < 20_000_000.0)] = "5m_to_20m"
    out.loc[market >= 20_000_000.0] = "over_20m"
    return out.astype(str)


def _minutes_series(frame: pd.DataFrame) -> pd.Series:
    if "minutes" in frame.columns:
        return pd.to_numeric(frame["minutes"], errors="coerce")
    if "sofa_minutesPlayed" in frame.columns:
        return pd.to_numeric(frame["sofa_minutesPlayed"], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _infer_label(
    frame: pd.DataFrame, explicit: str | None = None
) -> tuple[pd.Series | None, str | None, bool]:
    candidates = [explicit] if explicit else []
    candidates.extend(
        [
            "future_outcome_label",
            "is_future_undervalued_success",
            "future_success",
            "value_growth_next_season_eur",
            "future_value_growth_eur",
            "interval_contains_truth",
        ]
    )
    proxy_labels = {"interval_contains_truth"}
    for col in candidates:
        if not col or col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if col.endswith("_eur"):
            label = (series > 0).astype(float)
        else:
            label = (series > 0).astype(float)
        if int(label.notna().sum()) >= 20:
            return label, col, col in proxy_labels
    return None, None, False


def _resolve_score_col(frame: pd.DataFrame, explicit: str | None = None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            "scout_target_score",
            "shortlist_score",
            "undervaluation_score",
            "value_gap_capped_eur",
            "value_gap_conservative_eur",
            "value_gap_eur",
        ]
    )
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError("No suitable score column found for KPI ranking.")


@dataclass(frozen=True)
class CohortSpec:
    name: str
    col: str | None


def _precision_rows_for_group(
    frame: pd.DataFrame,
    *,
    group_type: str,
    group_name: str,
    score_col: str,
    label_col: str,
    k_values: Sequence[int],
) -> list[dict[str, object]]:
    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_label"] = pd.to_numeric(work[label_col], errors="coerce")
    work = work[np.isfinite(work["_score"]) & np.isfinite(work["_label"])].copy()
    work = work.sort_values("_score", ascending=False)
    n_labeled = int(len(work))
    if n_labeled <= 0:
        return []

    label_rate = float((work["_label"] > 0).mean())
    out: list[dict[str, object]] = []
    for k in sorted({max(int(k), 1) for k in k_values}):
        n_eval = min(k, n_labeled)
        top = work.head(n_eval)
        precision = float((top["_label"] > 0).mean())
        out.append(
            {
                "cohort_type": group_type,
                "cohort": group_name,
                "k": int(k),
                "n_pool": int(len(frame)),
                "n_labeled": n_labeled,
                "n_eval": int(n_eval),
                "positive_rate": label_rate,
                "precision_at_k": precision,
                "lift_vs_base": float(precision - label_rate),
            }
        )
    return out


def build_weekly_kpi_report(
    *,
    predictions_path: str,
    out_dir: str,
    split: str = "test",
    k_values: Sequence[int] = (10, 25, 50),
    score_col: str | None = None,
    label_col: str | None = None,
    min_minutes: float = 900.0,
    max_age: float | None = 23.0,
    non_big5_only: bool = False,
    cohort_min_labeled: int = 40,
) -> dict[str, object]:
    pred_path = Path(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    frame = pd.read_csv(pred_path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Predictions file is empty: {pred_path}")

    if split and "split" in frame.columns:
        frame = frame[frame["split"].astype(str).str.lower() == str(split).lower()].copy()

    minutes = _minutes_series(frame)
    frame = frame[minutes.fillna(0.0) >= float(min_minutes)].copy()

    if max_age is not None and "age" in frame.columns:
        age = pd.to_numeric(frame["age"], errors="coerce")
        frame = frame[age.fillna(999.0) <= float(max_age)].copy()

    if non_big5_only and "league" in frame.columns:
        league_norm = frame["league"].astype(str).str.strip().str.casefold()
        frame = frame[~league_norm.isin(BIG5_LEAGUES)].copy()

    if frame.empty:
        raise ValueError("No rows remain after KPI filters.")

    label_series, label_source, label_is_proxy = _infer_label(frame, explicit=label_col)
    if label_series is None or label_source is None:
        raise ValueError(
            "No usable outcome label found. Expected one of: "
            "future_outcome_label, is_future_undervalued_success, future_success, "
            "value_growth_next_season_eur, future_value_growth_eur, interval_contains_truth."
        )
    frame["_label"] = label_series

    resolved_score_col = _resolve_score_col(frame, explicit=score_col)
    frame["value_segment"] = _value_segment_series(frame)
    position_col = _resolve_position_col(frame)
    if position_col is not None:
        frame[position_col] = frame[position_col].astype(str).str.upper()

    cohorts = [
        CohortSpec(name="overall", col=None),
        CohortSpec(name="league", col="league" if "league" in frame.columns else None),
        CohortSpec(name="position", col=position_col),
        CohortSpec(name="value_segment", col="value_segment"),
    ]

    rows: list[dict[str, object]] = []
    for spec in cohorts:
        if spec.col is None:
            rows.extend(
                _precision_rows_for_group(
                    frame,
                    group_type=spec.name,
                    group_name="ALL",
                    score_col=resolved_score_col,
                    label_col="_label",
                    k_values=k_values,
                )
            )
            continue

        grouped = frame.groupby(spec.col, dropna=False)
        for key, g in grouped:
            g_work = g.copy()
            n_labeled = int(pd.to_numeric(g_work["_label"], errors="coerce").notna().sum())
            if n_labeled < int(cohort_min_labeled):
                continue
            rows.extend(
                _precision_rows_for_group(
                    g_work,
                    group_type=spec.name,
                    group_name=str(key),
                    score_col=resolved_score_col,
                    label_col="_label",
                    k_values=k_values,
                )
            )

    if not rows:
        raise ValueError("No KPI rows produced (all cohorts below minimum labeled threshold).")

    out_frame = pd.DataFrame(rows).sort_values(["cohort_type", "cohort", "k"])
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    stem = f"weekly_scout_kpi_{stamp}"

    csv_out = out_root / f"{stem}.csv"
    json_out = out_root / f"{stem}.json"
    out_frame.to_csv(csv_out, index=False)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_path": str(pred_path),
        "split": split,
        "filters": {
            "min_minutes": float(min_minutes),
            "max_age": None if max_age is None else float(max_age),
            "non_big5_only": bool(non_big5_only),
            "cohort_min_labeled": int(cohort_min_labeled),
        },
        "k_values": [int(k) for k in k_values],
        "score_column": resolved_score_col,
        "label_column": label_source,
        "label_is_proxy": bool(label_is_proxy),
        "row_count": int(len(out_frame)),
        "csv_path": str(csv_out),
        "items": out_frame.to_dict(orient="records"),
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[weekly-kpi] wrote csv -> {csv_out}")
    print(f"[weekly-kpi] wrote json -> {json_out}")
    print(
        "[weekly-kpi] "
        f"score={resolved_score_col} | label={label_source} | "
        f"rows={len(out_frame)} | cohorts(min_labeled={cohort_min_labeled})"
    )
    if label_is_proxy:
        print("[weekly-kpi] note: using proxy label (interval_contains_truth), not future outcome.")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build weekly scout KPI report from predictions: precision@k by league, "
            "position, and value segment."
        )
    )
    parser.add_argument("--predictions", required=True, help="Predictions CSV path.")
    parser.add_argument("--out-dir", default="data/model/reports", help="Output directory for KPI report files.")
    parser.add_argument("--split", default="test", help="Split label to report (if split column exists).")
    parser.add_argument("--k-values", default="10,25,50", help="Comma-separated K values for precision@K.")
    parser.add_argument("--score-col", default=None, help="Optional explicit ranking score column.")
    parser.add_argument("--label-col", default=None, help="Optional explicit future-outcome label column.")
    parser.add_argument("--min-minutes", type=float, default=900.0)
    parser.add_argument("--max-age", type=float, default=23.0, help="Set negative to disable age filtering.")
    parser.add_argument("--non-big5-only", action="store_true")
    parser.add_argument("--cohort-min-labeled", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_weekly_kpi_report(
        predictions_path=args.predictions,
        out_dir=args.out_dir,
        split=args.split,
        k_values=_parse_int_tokens(args.k_values, default=(10, 25, 50)),
        score_col=args.score_col,
        label_col=args.label_col,
        min_minutes=float(args.min_minutes),
        max_age=None if args.max_age < 0 else float(args.max_age),
        non_big5_only=bool(args.non_big5_only),
        cohort_min_labeled=int(args.cohort_min_labeled),
    )


if __name__ == "__main__":
    main()
