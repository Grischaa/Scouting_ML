from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scouting_ml.scripts.onboard_non_big5_leagues import build_onboarding_report
from scouting_ml.scripts.run_scout_workflow import run_workflow
from scouting_ml.scripts.weekly_scout_kpi_report import build_weekly_kpi_report


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _parse_int_tokens(raw: str | None, *, default: list[int]) -> list[int]:
    values: list[int] = []
    for token in _parse_csv_tokens(raw):
        try:
            n = int(float(token))
        except (TypeError, ValueError):
            continue
        if n > 0:
            values.append(n)
    return sorted({*values}) if values else list(default)


def run_weekly_scout_ops(
    *,
    predictions: str,
    split: str = "test",
    reports_out_dir: str = "data/model/reports",
    k_values: list[int] | None = None,
    score_col: str | None = None,
    label_col: str | None = "interval_contains_truth",
    min_minutes: float = 900.0,
    max_age: float | None = 23.0,
    non_big5_only: bool = True,
    cohort_min_labeled: int = 40,
    manifest: str = "data/processed/organization_manifest.csv",
    holdout_metrics_glob: str = "data/model/**/*.holdout_*.metrics.json",
    onboarding_out_json: str = "data/model/onboarding/non_big5_onboarding_report.json",
    onboarding_out_csv: str = "data/model/onboarding/non_big5_onboarding_report.csv",
    onboarding_min_seasons: int = 2,
    onboarding_min_files: int = 2,
    onboarding_max_domain_shift_z: float = 1.25,
    onboarding_min_holdout_r2: float = 0.35,
    workflow_out_dir: str = "data/model/scout_workflow",
    workflow_top_n: int = 150,
    workflow_min_confidence: float = 0.5,
    workflow_min_value_gap_eur: float = 1_000_000.0,
    workflow_positions: list[str] | None = None,
    workflow_include_leagues: list[str] | None = None,
    workflow_exclude_leagues: list[str] | None = None,
    workflow_report_top_metrics: int = 5,
    workflow_memo_count: int = 25,
    workflow_write_watchlist: bool = True,
    workflow_watchlist_path: str | None = "data/model/scout_watchlist.jsonl",
    workflow_watchlist_tag: str | None = "u23_non_big5",
) -> dict[str, Any]:
    k_values = k_values or [10, 25, 50]
    generated_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "inputs": {
            "predictions": predictions,
            "split": split,
            "min_minutes": float(min_minutes),
            "max_age": None if max_age is None else float(max_age),
            "non_big5_only": bool(non_big5_only),
        },
        "steps": {},
        "status": "ok",
    }

    kpi_payload = build_weekly_kpi_report(
        predictions_path=predictions,
        out_dir=reports_out_dir,
        split=split,
        k_values=k_values,
        score_col=score_col,
        label_col=label_col,
        min_minutes=min_minutes,
        max_age=max_age,
        non_big5_only=non_big5_only,
        cohort_min_labeled=cohort_min_labeled,
    )
    summary["steps"]["weekly_kpi"] = {
        "status": "ok",
        "out_csv": kpi_payload.get("out_csv"),
        "out_json": kpi_payload.get("out_json"),
        "row_count": kpi_payload.get("row_count"),
    }

    onboarding_payload = build_onboarding_report(
        manifest_path=manifest,
        holdout_metrics_glob=holdout_metrics_glob,
        out_json=onboarding_out_json,
        out_csv=onboarding_out_csv,
        min_seasons=onboarding_min_seasons,
        min_files=onboarding_min_files,
        max_domain_shift_z=onboarding_max_domain_shift_z,
        min_holdout_r2=onboarding_min_holdout_r2,
    )
    summary["steps"]["onboarding"] = {
        "status": "ok",
        "out_csv": onboarding_out_csv,
        "out_json": onboarding_out_json,
        "status_counts": onboarding_payload.get("status_counts"),
    }

    workflow_payload = run_workflow(
        predictions_path=predictions,
        split=split,
        out_dir=workflow_out_dir,
        top_n=workflow_top_n,
        min_minutes=min_minutes,
        max_age=max_age,
        min_confidence=workflow_min_confidence,
        min_value_gap_eur=workflow_min_value_gap_eur,
        non_big5_only=non_big5_only,
        positions=workflow_positions or [],
        include_leagues=workflow_include_leagues or [],
        exclude_leagues=workflow_exclude_leagues or [],
        report_top_metrics=workflow_report_top_metrics,
        memo_count=workflow_memo_count,
        write_watchlist=workflow_write_watchlist,
        watchlist_path=workflow_watchlist_path,
        watchlist_tag=workflow_watchlist_tag,
    )
    summary["steps"]["workflow"] = {
        "status": "ok",
        "shortlist_count": workflow_payload.get("shortlist_count"),
        "memo_count": workflow_payload.get("memo_count"),
        "shortlist_csv": workflow_payload.get("shortlist_csv"),
        "summary_json": workflow_payload.get("summary_json"),
        "watchlist_added": workflow_payload.get("watchlist_added"),
    }

    out_dir = Path(workflow_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = out_dir / f"weekly_ops_summary_{split}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)

    print(f"[weekly-ops] wrote summary -> {summary_path}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run weekly scout operations end-to-end: KPI report, non-Big5 onboarding check, "
            "and shortlist + memo workflow."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--reports-out-dir", default="data/model/reports")
    parser.add_argument("--k-values", default="10,25,50")
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--label-col", default="interval_contains_truth")
    parser.add_argument("--min-minutes", type=float, default=900.0)
    parser.add_argument("--max-age", type=float, default=23.0, help="Set negative to disable.")
    parser.add_argument("--non-big5-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cohort-min-labeled", type=int, default=40)

    parser.add_argument("--manifest", default="data/processed/organization_manifest.csv")
    parser.add_argument("--holdout-metrics-glob", default="data/model/**/*.holdout_*.metrics.json")
    parser.add_argument("--onboarding-out-json", default="data/model/onboarding/non_big5_onboarding_report.json")
    parser.add_argument("--onboarding-out-csv", default="data/model/onboarding/non_big5_onboarding_report.csv")
    parser.add_argument("--onboarding-min-seasons", type=int, default=2)
    parser.add_argument("--onboarding-min-files", type=int, default=2)
    parser.add_argument("--onboarding-max-domain-shift-z", type=float, default=1.25)
    parser.add_argument("--onboarding-min-holdout-r2", type=float, default=0.35)

    parser.add_argument("--workflow-out-dir", default="data/model/scout_workflow")
    parser.add_argument("--workflow-top-n", type=int, default=150)
    parser.add_argument("--workflow-min-confidence", type=float, default=0.5)
    parser.add_argument("--workflow-min-value-gap-eur", type=float, default=1_000_000.0)
    parser.add_argument("--workflow-positions", default="")
    parser.add_argument("--workflow-include-leagues", default="")
    parser.add_argument("--workflow-exclude-leagues", default="")
    parser.add_argument("--workflow-report-top-metrics", type=int, default=5)
    parser.add_argument("--workflow-memo-count", type=int, default=25)
    parser.add_argument("--workflow-write-watchlist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workflow-watchlist-path", default="data/model/scout_watchlist.jsonl")
    parser.add_argument("--workflow-watchlist-tag", default="u23_non_big5")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_weekly_scout_ops(
        predictions=args.predictions,
        split=args.split,
        reports_out_dir=args.reports_out_dir,
        k_values=_parse_int_tokens(args.k_values, default=[10, 25, 50]),
        score_col=args.score_col,
        label_col=args.label_col,
        min_minutes=args.min_minutes,
        max_age=None if args.max_age < 0 else args.max_age,
        non_big5_only=args.non_big5_only,
        cohort_min_labeled=args.cohort_min_labeled,
        manifest=args.manifest,
        holdout_metrics_glob=args.holdout_metrics_glob,
        onboarding_out_json=args.onboarding_out_json,
        onboarding_out_csv=args.onboarding_out_csv,
        onboarding_min_seasons=args.onboarding_min_seasons,
        onboarding_min_files=args.onboarding_min_files,
        onboarding_max_domain_shift_z=args.onboarding_max_domain_shift_z,
        onboarding_min_holdout_r2=args.onboarding_min_holdout_r2,
        workflow_out_dir=args.workflow_out_dir,
        workflow_top_n=args.workflow_top_n,
        workflow_min_confidence=args.workflow_min_confidence,
        workflow_min_value_gap_eur=args.workflow_min_value_gap_eur,
        workflow_positions=_parse_csv_tokens(args.workflow_positions),
        workflow_include_leagues=_parse_csv_tokens(args.workflow_include_leagues),
        workflow_exclude_leagues=_parse_csv_tokens(args.workflow_exclude_leagues),
        workflow_report_top_metrics=args.workflow_report_top_metrics,
        workflow_memo_count=args.workflow_memo_count,
        workflow_write_watchlist=args.workflow_write_watchlist,
        workflow_watchlist_path=args.workflow_watchlist_path,
        workflow_watchlist_tag=args.workflow_watchlist_tag,
    )


if __name__ == "__main__":
    main()
