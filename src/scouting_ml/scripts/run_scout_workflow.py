from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scouting_ml.services.market_value_service import (
    add_watchlist_item,
    get_player_history_strength,
    get_player_report,
    query_scout_targets,
)


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _slugify(value: str) -> str:
    keep = [c if c.isalnum() or c in {"_", "-", "."} else "_" for c in str(value)]
    return "".join(keep).strip("_") or "player"


def run_workflow(
    *,
    predictions_path: str,
    split: str,
    out_dir: str,
    top_n: int = 150,
    min_minutes: float = 900.0,
    max_age: float | None = 23.0,
    min_confidence: float = 0.5,
    min_value_gap_eur: float = 1_000_000.0,
    non_big5_only: bool = True,
    positions: list[str] | None = None,
    include_leagues: list[str] | None = None,
    exclude_leagues: list[str] | None = None,
    min_expected_value_eur: float | None = None,
    max_expected_value_eur: float | None = None,
    report_top_metrics: int = 5,
    memo_count: int = 25,
    write_watchlist: bool = False,
    watchlist_path: str | None = None,
    watchlist_tag: str | None = None,
) -> dict[str, Any]:
    split_key = str(split).strip().lower()
    if split_key not in {"test", "val"}:
        raise ValueError("split must be 'test' or 'val'.")

    pred_path = Path(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    if split_key == "test":
        os.environ["SCOUTING_TEST_PREDICTIONS_PATH"] = str(pred_path)
    else:
        os.environ["SCOUTING_VAL_PREDICTIONS_PATH"] = str(pred_path)

    if watchlist_path:
        os.environ["SCOUTING_WATCHLIST_PATH"] = str(Path(watchlist_path))

    shortlist = query_scout_targets(
        split=split_key,  # type: ignore[arg-type]
        top_n=int(top_n),
        min_minutes=float(min_minutes),
        max_age=max_age,
        min_confidence=float(min_confidence),
        min_value_gap_eur=float(min_value_gap_eur),
        positions=positions,
        non_big5_only=bool(non_big5_only),
        include_leagues=include_leagues,
        exclude_leagues=exclude_leagues,
        min_expected_value_eur=min_expected_value_eur,
        max_expected_value_eur=max_expected_value_eur,
    )

    rows = shortlist.get("items", [])
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shortlist_csv = out_root / f"scout_workflow_shortlist_{split_key}_{stamp}.csv"
    shortlist_json = out_root / f"scout_workflow_shortlist_{split_key}_{stamp}.json"
    pd.DataFrame(rows).to_csv(shortlist_csv, index=False)
    shortlist_json.write_text(json.dumps(shortlist, indent=2), encoding="utf-8")

    memo_dir = out_root / f"memos_{split_key}_{stamp}"
    memo_dir.mkdir(parents=True, exist_ok=True)
    memo_rows = rows[: max(int(memo_count), 0)]
    memo_index: list[dict[str, Any]] = []
    for idx, row in enumerate(memo_rows, start=1):
        player_id = str(row.get("player_id", "")).strip()
        if not player_id:
            continue
        season = str(row.get("season") or "").strip() or None
        report = get_player_report(
            player_id=player_id,
            split=split_key,  # type: ignore[arg-type]
            season=season,
            top_metrics=int(report_top_metrics),
        )
        history = get_player_history_strength(
            player_id=player_id,
            split=split_key,  # type: ignore[arg-type]
            season=season,
        )
        memo_payload = {
            "rank": idx,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": split_key,
            "scout_target": row,
            "report": report,
            "history_strength": history.get("history_strength"),
        }
        memo_name = f"{idx:03d}_{_slugify(player_id)}.json"
        memo_path = memo_dir / memo_name
        memo_path.write_text(json.dumps(memo_payload, indent=2), encoding="utf-8")
        memo_index.append({"rank": idx, "player_id": player_id, "memo_path": str(memo_path)})

    watchlist_added = 0
    if write_watchlist:
        for row in rows:
            player_id = str(row.get("player_id", "")).strip()
            if not player_id:
                continue
            season = str(row.get("season") or "").strip() or None
            add_watchlist_item(
                player_id=player_id,
                split=split_key,  # type: ignore[arg-type]
                season=season,
                tag=watchlist_tag,
                notes="auto-added by run_scout_workflow",
                source="workflow_script",
            )
            watchlist_added += 1

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split_key,
        "predictions_path": str(pred_path),
        "shortlist_count": int(len(rows)),
        "shortlist_csv": str(shortlist_csv),
        "shortlist_json": str(shortlist_json),
        "memo_count": int(len(memo_index)),
        "memo_dir": str(memo_dir),
        "memos": memo_index,
        "watchlist_added": int(watchlist_added),
        "diagnostics": shortlist.get("diagnostics"),
    }
    summary_json = out_root / f"scout_workflow_summary_{split_key}_{stamp}.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["summary_json"] = str(summary_json)

    print(f"[workflow] wrote shortlist csv -> {shortlist_csv}")
    print(f"[workflow] wrote shortlist json -> {shortlist_json}")
    print(f"[workflow] wrote memo dir -> {memo_dir}")
    if write_watchlist:
        print(f"[workflow] watchlist entries added -> {watchlist_added}")
    print(f"[workflow] wrote summary -> {summary_json}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scouting workflow end-to-end: shortlist -> report memos -> optional watchlist."
    )
    parser.add_argument("--predictions", required=True, help="Predictions CSV to use for the selected split.")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--out-dir", default="data/model/scout_workflow")
    parser.add_argument("--top-n", type=int, default=150)
    parser.add_argument("--min-minutes", type=float, default=900.0)
    parser.add_argument("--max-age", type=float, default=23.0, help="Set negative to disable.")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-value-gap-eur", type=float, default=1_000_000.0)
    parser.add_argument("--non-big5-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--positions", default="")
    parser.add_argument("--include-leagues", default="")
    parser.add_argument("--exclude-leagues", default="")
    parser.add_argument("--min-expected-value-eur", type=float, default=None)
    parser.add_argument("--max-expected-value-eur", type=float, default=None)
    parser.add_argument("--report-top-metrics", type=int, default=5)
    parser.add_argument("--memo-count", type=int, default=25)
    parser.add_argument("--write-watchlist", action="store_true")
    parser.add_argument("--watchlist-path", default=None)
    parser.add_argument("--watchlist-tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_workflow(
        predictions_path=args.predictions,
        split=args.split,
        out_dir=args.out_dir,
        top_n=args.top_n,
        min_minutes=args.min_minutes,
        max_age=None if args.max_age < 0 else args.max_age,
        min_confidence=args.min_confidence,
        min_value_gap_eur=args.min_value_gap_eur,
        non_big5_only=args.non_big5_only,
        positions=_parse_csv_tokens(args.positions),
        include_leagues=_parse_csv_tokens(args.include_leagues),
        exclude_leagues=_parse_csv_tokens(args.exclude_leagues),
        min_expected_value_eur=args.min_expected_value_eur,
        max_expected_value_eur=args.max_expected_value_eur,
        report_top_metrics=args.report_top_metrics,
        memo_count=args.memo_count,
        write_watchlist=args.write_watchlist,
        watchlist_path=args.watchlist_path,
        watchlist_tag=args.watchlist_tag,
    )


if __name__ == "__main__":
    main()
