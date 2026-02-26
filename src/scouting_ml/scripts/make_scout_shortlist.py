from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scouting_ml.features.history_strength import add_history_strength_features


def _parse_positions(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    parts = [x.strip().upper() for x in raw.split(",") if x.strip()]
    return set(parts) if parts else None


def _infer_future_label(frame: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
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


def _precision_at_k(frame: pd.DataFrame, score_col: str, top_n: int) -> dict[str, object]:
    labels, col = _infer_future_label(frame)
    if labels is None or col is None:
        return {"available": False, "reason": "missing_future_outcome_label"}

    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_label"] = pd.to_numeric(labels, errors="coerce")
    work = work[np.isfinite(work["_score"]) & np.isfinite(work["_label"])].copy()
    if work.empty:
        return {"available": False, "reason": "no_rows_with_labels"}
    work = work.sort_values("_score", ascending=False).reset_index(drop=True)
    ks = sorted({10, 25, 50, int(max(top_n, 1))})
    rows: list[dict[str, float | int]] = []
    for k in ks:
        top = work.head(k)
        if top.empty:
            continue
        rows.append(
            {
                "k": int(k),
                "n": int(len(top)),
                "precision": float((top["_label"] > 0).mean()),
            }
        )
    return {
        "available": bool(rows),
        "label_column": col,
        "n_labeled_rows": int(len(work)),
        "rows": rows,
    }


def build_shortlist(
    predictions_path: str,
    output_path: str,
    top_n: int = 100,
    min_minutes: int = 900,
    max_age: int | None = 25,
    positions: set[str] | None = None,
    metrics_output: str | None = None,
) -> None:
    df = pd.read_csv(predictions_path)

    pred_value_col = "fair_value_eur" if "fair_value_eur" in df.columns else "expected_value_eur"
    required = {"market_value_eur", pred_value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    for col in [
        "market_value_eur",
        "expected_value_eur",
        "fair_value_eur",
        "value_gap_conservative_eur",
        "undervaluation_confidence",
        "undervaluation_score",
        "age",
        "minutes",
        "sofa_minutesPlayed",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "value_gap_conservative_eur" not in work.columns:
        work["value_gap_conservative_eur"] = (
            work[pred_value_col] - work["market_value_eur"]
        )
    work = add_history_strength_features(work)
    if "value_gap_capped_eur" in work.columns:
        work["ranking_gap_eur"] = pd.to_numeric(work["value_gap_capped_eur"], errors="coerce")
    else:
        work["ranking_gap_eur"] = pd.to_numeric(work["value_gap_conservative_eur"], errors="coerce")
    if "undervaluation_confidence" not in work.columns:
        denom = np.maximum(np.abs(work[pred_value_col] - work["market_value_eur"]).median(), 1.0)
        work["undervaluation_confidence"] = work["value_gap_conservative_eur"] / denom

    if "minutes" in work.columns:
        mins = work["minutes"]
    elif "sofa_minutesPlayed" in work.columns:
        mins = work["sofa_minutesPlayed"]
    else:
        mins = pd.Series(0, index=work.index, dtype=float)
    work["minutes_used"] = mins.fillna(0.0)

    if "model_position" in work.columns:
        work["position_used"] = work["model_position"].astype(str).str.upper()
    elif "position_group" in work.columns:
        work["position_used"] = work["position_group"].astype(str).str.upper()
    else:
        work["position_used"] = "UNK"

    work = work[work["ranking_gap_eur"] > 0].copy()
    work = work[work["minutes_used"] >= float(min_minutes)].copy()

    if max_age is not None and "age" in work.columns:
        work = work[work["age"].fillna(999) <= max_age].copy()

    if positions:
        work = work[work["position_used"].isin(positions)].copy()

    if work.empty:
        raise ValueError("No candidates after filtering. Relax min-minutes/max-age/position filters.")

    reliability = np.clip(work["minutes_used"] / 1800.0, 0.3, 1.2)
    confidence = work["undervaluation_confidence"].clip(lower=0.0)
    age_factor = np.ones(len(work))
    if "age" in work.columns:
        age = work["age"].fillna(26.0)
        age_factor = np.where(age <= 23, 1.15, np.where(age <= 26, 1.0, 0.85))

    if "history_strength_score" in work.columns:
        history_strength = pd.to_numeric(work["history_strength_score"], errors="coerce") / 100.0
        history_strength = history_strength.clip(lower=0.0, upper=1.0)
        if "history_strength_coverage" in work.columns:
            history_cov = pd.to_numeric(work["history_strength_coverage"], errors="coerce").clip(lower=0.0, upper=1.0)
        else:
            history_cov = pd.Series(1.0, index=work.index, dtype=float)
        history_factor = (0.85 + 0.35 * history_strength) * (0.90 + 0.10 * history_cov)
        history_factor = history_factor.where(history_cov >= 0.35, 1.0).fillna(1.0)
    else:
        history_factor = pd.Series(1.0, index=work.index, dtype=float)

    work["scout_score"] = (
        (work["ranking_gap_eur"] / 1_000_000.0)
        * np.log1p(confidence)
        * reliability
        * age_factor
        * history_factor
    )

    sort_cols = ["scout_score", "ranking_gap_eur", "value_gap_conservative_eur", "undervaluation_confidence"]
    shortlist = work.sort_values(sort_cols, ascending=False).head(top_n).copy()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(output, index=False)
    print(f"[shortlist] wrote {len(shortlist):,} rows → {output}")

    diagnostics = {
        "total_candidates": int(len(work)),
        "count": int(len(shortlist)),
        "ranking_basis": "guardrailed_gap_confidence_history",
        "score_column": "scout_score",
        "precision_at_k": _precision_at_k(work, score_col="scout_score", top_n=int(top_n)),
    }
    if diagnostics["precision_at_k"].get("available"):
        rows = diagnostics["precision_at_k"].get("rows", [])
        for row in rows:
            print(
                f"[shortlist] precision@{int(row['k'])}: "
                f"{float(row['precision']) * 100:.2f}% (n={int(row['n'])})"
            )
    else:
        reason = diagnostics["precision_at_k"].get("reason", "unavailable")
        print(f"[shortlist] precision@k unavailable ({reason})")

    if metrics_output:
        metrics_path = Path(metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(f"[shortlist] wrote diagnostics → {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a confidence-aware scouting shortlist from prediction output CSV."
    )
    parser.add_argument(
        "--predictions",
        default="data/model/big5_predictions_full_v2.csv",
        help="Path to predictions CSV from train_market_value_full.",
    )
    parser.add_argument(
        "--output",
        default="data/model/scout_shortlist.csv",
        help="Output CSV path for ranked shortlist.",
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-minutes", type=int, default=900)
    parser.add_argument(
        "--max-age",
        type=int,
        default=25,
        help="Max age filter. Use -1 to disable.",
    )
    parser.add_argument(
        "--positions",
        default=None,
        help="Optional comma-separated positions, e.g. FW,MF",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional JSON output path for shortlist diagnostics (includes precision@k if future label exists).",
    )
    args = parser.parse_args()

    max_age = None if args.max_age is not None and args.max_age < 0 else args.max_age
    build_shortlist(
        predictions_path=args.predictions,
        output_path=args.output,
        top_n=args.top_n,
        min_minutes=args.min_minutes,
        max_age=max_age,
        positions=_parse_positions(args.positions),
        metrics_output=args.metrics_output,
    )


if __name__ == "__main__":
    main()
