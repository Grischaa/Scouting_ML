from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


POSITIVE_ACTIONS = {"shortlist", "watch_live", "request_report"}


def _read_decision_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL scout decisions, skipping malformed lines."""
    if not path.exists():
        raise FileNotFoundError(f"Scout decision log not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-like timestamp into UTC."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _rank_bucket(value: Any) -> str:
    """Map a numeric rank into the report's coarse approval buckets."""
    try:
        rank = int(float(value))
    except (TypeError, ValueError):
        return "unranked"
    if rank <= 10:
        return "1-10"
    if rank <= 25:
        return "11-25"
    return "26+"


def _stringify(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    return text or fallback


def _normalize_rows(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Flatten decision events into one analytics frame."""
    rows: list[dict[str, Any]] = []
    for record in records:
        snapshot = record.get("player_snapshot") if isinstance(record.get("player_snapshot"), dict) else {}
        ranking = record.get("ranking_context") if isinstance(record.get("ranking_context"), dict) else {}
        created_at = _parse_timestamp(record.get("created_at_utc"))
        action = _stringify(record.get("action"))
        reason_tags = record.get("reason_tags") if isinstance(record.get("reason_tags"), list) else []
        rows.append(
            {
                "decision_id": _stringify(record.get("decision_id"), fallback=""),
                "created_at_utc": created_at.isoformat() if created_at else "",
                "action": action,
                "is_positive": action in POSITIVE_ACTIONS,
                "league": _stringify(snapshot.get("league")),
                "position": _stringify(snapshot.get("position")),
                "trust_tier": _stringify(snapshot.get("league_trust_tier")),
                "league_adjustment_bucket": _stringify(snapshot.get("league_adjustment_bucket")),
                "source_surface": _stringify(record.get("source_surface")),
                "system_template": _stringify(ranking.get("system_template")),
                "system_slot": _stringify(ranking.get("system_slot")),
                "rank_bucket": _rank_bucket(ranking.get("rank")),
                "discovery_reliability_weight": float(ranking.get("discovery_reliability_weight"))
                if ranking.get("discovery_reliability_weight") is not None
                else float("nan"),
                "reason_tags": list(reason_tags),
            }
        )
    return pd.DataFrame(rows)


def _cohort_breakdown(frame: pd.DataFrame, column: str, cohort_type: str) -> pd.DataFrame:
    """Build one cohort breakdown table with positive/pass rates and reliability averages."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "cohort_type",
                "cohort",
                "total",
                "positive_count",
                "pass_count",
                "positive_rate",
                "avg_discovery_reliability_weight",
            ]
        )
    grouped = (
        frame.groupby(column, dropna=False)
        .agg(
            total=("decision_id", "count"),
            positive_count=("is_positive", "sum"),
            pass_count=("action", lambda s: int((s == "pass").sum())),
            avg_discovery_reliability_weight=("discovery_reliability_weight", "mean"),
        )
        .reset_index()
        .rename(columns={column: "cohort"})
    )
    grouped["cohort_type"] = cohort_type
    grouped["positive_rate"] = grouped["positive_count"] / grouped["total"].clip(lower=1)
    ordered = grouped[
        [
            "cohort_type",
            "cohort",
            "total",
            "positive_count",
            "pass_count",
            "positive_rate",
            "avg_discovery_reliability_weight",
        ]
    ].sort_values(["cohort_type", "positive_rate", "total", "cohort"], ascending=[True, False, False, True])
    return ordered


def build_weekly_scout_decision_feedback_report(
    *,
    decisions_path: str,
    out_dir: str,
    lookback_days: int = 7,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build JSON/CSV/Markdown decision feedback outputs for the recent lookback window."""
    path = Path(decisions_path)
    out_root = Path(out_dir)
    now = now_utc.astimezone(timezone.utc) if now_utc else datetime.now(timezone.utc)
    since = now - timedelta(days=max(int(lookback_days), 1))
    frame = _normalize_rows(_read_decision_records(path))
    if frame.empty:
        raise ValueError("Scout decision log is empty.")

    timestamps = pd.to_datetime(frame["created_at_utc"], errors="coerce", utc=True)
    frame = frame[timestamps >= since].copy()
    if frame.empty:
        raise ValueError("No scout decisions fall inside the requested lookback window.")

    frame["pass_reason"] = frame["reason_tags"].apply(
        lambda tags: [str(tag) for tag in tags if str(tag)] if isinstance(tags, list) else []
    )
    pass_reason_counts = (
        frame.loc[frame["action"] == "pass", ["pass_reason"]]
        .explode("pass_reason")
        .dropna()
        .groupby("pass_reason")
        .size()
        .sort_values(ascending=False)
        .rename("count")
        .reset_index()
    )

    cohort_frames = [
        _cohort_breakdown(frame, "league", "league"),
        _cohort_breakdown(frame, "position", "position"),
        _cohort_breakdown(frame, "trust_tier", "trust_tier"),
        _cohort_breakdown(frame, "league_adjustment_bucket", "league_adjustment_bucket"),
        _cohort_breakdown(frame, "source_surface", "source_surface"),
        _cohort_breakdown(frame, "system_template", "system_template"),
        _cohort_breakdown(frame, "rank_bucket", "rank_bucket"),
    ]
    breakdown = pd.concat(cohort_frames, ignore_index=True)
    action_counts = frame["action"].value_counts(dropna=False).rename_axis("action").reset_index(name="count")

    positive_mask = frame["is_positive"] == True
    pass_mask = frame["action"] == "pass"
    summary = {
        "generated_at_utc": now.isoformat(),
        "decisions_path": str(path),
        "lookback_days": int(lookback_days),
        "date_from_utc": since.isoformat(),
        "date_to_utc": now.isoformat(),
        "total_decisions": int(len(frame)),
        "action_counts": {str(row.action): int(row.count) for row in action_counts.itertuples(index=False)},
        "positive_decision_rate": float(frame["is_positive"].mean()),
        "pass_reason_counts": {str(row.pass_reason): int(row.count) for row in pass_reason_counts.itertuples(index=False)},
        "mean_reliability_weight_positive": float(frame.loc[positive_mask, "discovery_reliability_weight"].mean()),
        "mean_reliability_weight_pass": float(frame.loc[pass_mask, "discovery_reliability_weight"].mean()),
        "positive_rate_by_bucket": {
            str(row.cohort): float(row.positive_rate)
            for row in breakdown[breakdown["cohort_type"] == "league_adjustment_bucket"].itertuples(index=False)
        },
    }

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"weekly_scout_decisions_{stamp}.json"
    markdown_path = out_root / f"weekly_scout_decisions_{stamp}.md"
    breakdown_path = out_root / f"weekly_scout_decision_breakdowns_{stamp}.csv"
    action_counts_path = out_root / f"weekly_scout_decision_actions_{stamp}.csv"
    pass_reasons_path = out_root / f"weekly_scout_decision_pass_reasons_{stamp}.csv"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    breakdown.to_csv(breakdown_path, index=False)
    action_counts.to_csv(action_counts_path, index=False)
    pass_reason_counts.to_csv(pass_reasons_path, index=False)

    markdown_lines = [
        "# Weekly Scout Decision Feedback",
        "",
        f"- Generated: `{summary['generated_at_utc']}`",
        f"- Decisions: `{summary['total_decisions']}`",
        f"- Positive decision rate: `{summary['positive_decision_rate']:.2%}`",
        f"- Mean reliability weight, positive: `{summary['mean_reliability_weight_positive']:.3f}`",
        f"- Mean reliability weight, pass: `{summary['mean_reliability_weight_pass']:.3f}`",
        "",
        "## Action Counts",
        "",
    ]
    markdown_lines.extend(
        [f"- `{row.action}`: `{int(row.count)}`" for row in action_counts.itertuples(index=False)] or ["- No actions recorded."]
    )
    markdown_lines.extend(["", "## Pass Reasons", ""])
    markdown_lines.extend(
        [f"- `{row.pass_reason}`: `{int(row.count)}`" for row in pass_reason_counts.itertuples(index=False)]
        or ["- No pass reasons recorded."]
    )
    markdown_lines.extend(["", "## Top Cohorts", ""])
    top_rows = breakdown.sort_values(["positive_rate", "total"], ascending=[False, False]).head(8)
    markdown_lines.extend(
        [
            f"- `{row.cohort_type}` / `{row.cohort}`: `{float(row.positive_rate):.2%}` positive over `{int(row.total)}` decisions"
            for row in top_rows.itertuples(index=False)
        ]
        or ["- No cohort breakdowns available."]
    )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "paths": {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "breakdowns_csv": str(breakdown_path),
            "action_counts_csv": str(action_counts_path),
            "pass_reasons_csv": str(pass_reasons_path),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a weekly scout decision feedback report.")
    parser.add_argument("--decisions-path", required=True, help="Path to scout_decisions.jsonl")
    parser.add_argument("--out-dir", required=True, help="Directory to write JSON/CSV/Markdown outputs")
    parser.add_argument("--lookback-days", type=int, default=7, help="Lookback window in days")
    return parser


def main() -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args()
    payload = build_weekly_scout_decision_feedback_report(
        decisions_path=args.decisions_path,
        out_dir=args.out_dir,
        lookback_days=args.lookback_days,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
