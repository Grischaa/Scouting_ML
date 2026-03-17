from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scouting_ml.scripts.run_scout_workflow import run_workflow


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows, start=1):
        player_id = str(row.get("player_id") or "").strip()
        if not player_id:
            continue
        payload = dict(row)
        payload["_rank"] = idx
        out[player_id] = payload
    return out


def _load_shortlist_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    shortlist_json = Path(str(summary.get("shortlist_json") or ""))
    if shortlist_json.exists():
        payload = json.loads(shortlist_json.read_text(encoding="utf-8"))
        items = payload.get("items")
        if isinstance(items, list):
            return [row for row in items if isinstance(row, dict)]
    return []


def compare_scout_workflow_variants(
    *,
    baseline_predictions: str,
    future_predictions: str,
    split: str,
    out_dir: str,
    top_n: int = 100,
    min_minutes: float = 900.0,
    max_age: float | None = 23.0,
    min_confidence: float = 0.5,
    min_value_gap_eur: float = 1_000_000.0,
    non_big5_only: bool = True,
    positions: list[str] | None = None,
    include_leagues: list[str] | None = None,
    exclude_leagues: list[str] | None = None,
    report_top_metrics: int = 5,
    memo_count: int = 15,
) -> dict[str, Any]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    baseline = run_workflow(
        predictions_path=baseline_predictions,
        split=split,
        out_dir=str(out_root / "baseline"),
        top_n=top_n,
        min_minutes=min_minutes,
        max_age=max_age,
        min_confidence=min_confidence,
        min_value_gap_eur=min_value_gap_eur,
        non_big5_only=non_big5_only,
        positions=positions or [],
        include_leagues=include_leagues or [],
        exclude_leagues=exclude_leagues or [],
        report_top_metrics=report_top_metrics,
        memo_count=memo_count,
        write_watchlist=False,
    )
    future = run_workflow(
        predictions_path=future_predictions,
        split=split,
        out_dir=str(out_root / "future_scored"),
        top_n=top_n,
        min_minutes=min_minutes,
        max_age=max_age,
        min_confidence=min_confidence,
        min_value_gap_eur=min_value_gap_eur,
        non_big5_only=non_big5_only,
        positions=positions or [],
        include_leagues=include_leagues or [],
        exclude_leagues=exclude_leagues or [],
        report_top_metrics=report_top_metrics,
        memo_count=memo_count,
        write_watchlist=False,
    )

    baseline_rows = _load_shortlist_rows(baseline)
    future_rows = _load_shortlist_rows(future)
    baseline_index = _index_rows(baseline_rows)
    future_index = _index_rows(future_rows)

    baseline_ids = set(baseline_index)
    future_ids = set(future_index)
    overlap_ids = baseline_ids & future_ids
    future_only_ids = future_ids - baseline_ids
    baseline_only_ids = baseline_ids - future_ids

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "split": split,
            "top_n": int(top_n),
            "min_minutes": float(min_minutes),
            "max_age": None if max_age is None else float(max_age),
            "min_confidence": float(min_confidence),
            "min_value_gap_eur": float(min_value_gap_eur),
            "non_big5_only": bool(non_big5_only),
            "baseline_predictions": baseline_predictions,
            "future_predictions": future_predictions,
        },
        "baseline": {
            "summary_json": baseline.get("summary_json"),
            "shortlist_count": baseline.get("shortlist_count"),
            "diagnostics": baseline.get("diagnostics"),
        },
        "future_scored": {
            "summary_json": future.get("summary_json"),
            "shortlist_count": future.get("shortlist_count"),
            "diagnostics": future.get("diagnostics"),
        },
        "comparison": {
            "overlap_count": int(len(overlap_ids)),
            "overlap_share_of_future": float(len(overlap_ids) / max(len(future_ids), 1)),
            "future_only_count": int(len(future_only_ids)),
            "baseline_only_count": int(len(baseline_only_ids)),
            "future_only_top": [
                future_index[player_id] for player_id in sorted(future_only_ids, key=lambda pid: future_index[pid]["_rank"])[:20]
            ],
            "baseline_only_top": [
                baseline_index[player_id]
                for player_id in sorted(baseline_only_ids, key=lambda pid: baseline_index[pid]["_rank"])[:20]
            ],
            "largest_rank_gains": sorted(
                [
                    {
                        "player_id": player_id,
                        "baseline_rank": int(baseline_index[player_id]["_rank"]),
                        "future_rank": int(future_index[player_id]["_rank"]),
                        "rank_gain": int(baseline_index[player_id]["_rank"]) - int(future_index[player_id]["_rank"]),
                        "name": future_index[player_id].get("name"),
                        "league": future_index[player_id].get("league"),
                        "season": future_index[player_id].get("season"),
                    }
                    for player_id in overlap_ids
                ],
                key=lambda row: (-(row["rank_gain"]), row["future_rank"]),
            )[:20],
        },
    }

    json_out = out_root / f"scout_workflow_variant_compare_{split}.json"
    md_out = out_root / f"scout_workflow_variant_compare_{split}.md"
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Scout Workflow Variant Comparison",
        "",
        f"- Split: `{split}`",
        f"- Baseline predictions: `{baseline_predictions}`",
        f"- Future-scored predictions: `{future_predictions}`",
        "",
        "## Summary",
        f"- Baseline shortlist count: `{baseline.get('shortlist_count')}`",
        f"- Future-scored shortlist count: `{future.get('shortlist_count')}`",
        f"- Overlap: `{payload['comparison']['overlap_count']}`",
        f"- Future-only: `{payload['comparison']['future_only_count']}`",
        f"- Baseline-only: `{payload['comparison']['baseline_only_count']}`",
        "",
        "## Diagnostics",
        f"- Baseline score column: `{(baseline.get('diagnostics') or {}).get('score_column')}`",
        f"- Future score column: `{(future.get('diagnostics') or {}).get('score_column')}`",
    ]
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[workflow-compare] wrote json -> {json_out}")
    print(f"[workflow-compare] wrote markdown -> {md_out}")
    return payload


def _parse_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline and future-scored scout workflows side by side.")
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--future-predictions", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--out-dir", default="data/model/reports/workflow_compare")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-minutes", type=float, default=900.0)
    parser.add_argument("--max-age", type=float, default=23.0, help="Set negative to disable.")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-value-gap-eur", type=float, default=1_000_000.0)
    parser.add_argument("--non-big5-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--positions", default="")
    parser.add_argument("--include-leagues", default="")
    parser.add_argument("--exclude-leagues", default="")
    parser.add_argument("--report-top-metrics", type=int, default=5)
    parser.add_argument("--memo-count", type=int, default=15)
    args = parser.parse_args()

    compare_scout_workflow_variants(
        baseline_predictions=args.baseline_predictions,
        future_predictions=args.future_predictions,
        split=args.split,
        out_dir=args.out_dir,
        top_n=args.top_n,
        min_minutes=args.min_minutes,
        max_age=None if args.max_age < 0 else args.max_age,
        min_confidence=args.min_confidence,
        min_value_gap_eur=args.min_value_gap_eur,
        non_big5_only=args.non_big5_only,
        positions=_parse_csv(args.positions),
        include_leagues=_parse_csv(args.include_leagues),
        exclude_leagues=_parse_csv(args.exclude_leagues),
        report_top_metrics=args.report_top_metrics,
        memo_count=args.memo_count,
    )


if __name__ == "__main__":
    main()
