from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _rows_for(
    payload: dict[str, Any],
    *,
    split: str,
    label_key: str,
    cohort_type: str,
    k: int,
) -> list[dict[str, Any]]:
    split_payload = (payload.get("splits") or {}).get(split) or {}
    rows = ((split_payload.get("precision_at_k") or {}).get(label_key) or [])
    filtered = [
        row
        for row in rows
        if str(row.get("cohort_type")) == str(cohort_type) and int(row.get("k") or 0) == int(k)
    ]
    return [dict(row) for row in filtered]


def _sort_best(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -(_safe_float(row.get("precision_at_k")) or -1.0),
            -(_safe_float(row.get("lift_vs_base")) or -999.0),
            -(int(row.get("n_labeled") or 0)),
            str(row.get("cohort") or ""),
        ),
    )


def _sort_worst(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            (_safe_float(row.get("precision_at_k")) or 999.0),
            (_safe_float(row.get("lift_vs_base")) or 999.0),
            -(int(row.get("n_labeled") or 0)),
            str(row.get("cohort") or ""),
        ),
    )


def _cohort_summary(
    payload: dict[str, Any],
    *,
    split: str,
    label_key: str,
    k: int,
    top_n: int,
) -> dict[str, Any]:
    league_rows = _rows_for(payload, split=split, label_key=label_key, cohort_type="league", k=k)
    position_rows = _rows_for(payload, split=split, label_key=label_key, cohort_type="position", k=k)
    value_rows = _rows_for(payload, split=split, label_key=label_key, cohort_type="value_segment", k=k)
    return {
        "league": {
            "count": len(league_rows),
            "best": _sort_best(league_rows)[:top_n],
            "worst": _sort_worst(league_rows)[:top_n],
        },
        "position": {
            "count": len(position_rows),
            "best": _sort_best(position_rows)[:top_n],
            "worst": _sort_worst(position_rows)[:top_n],
        },
        "value_segment": {
            "count": len(value_rows),
            "best": _sort_best(value_rows)[:top_n],
            "worst": _sort_worst(value_rows)[:top_n],
        },
    }


def build_future_value_diagnostics_payload(
    benchmark_payload: dict[str, Any],
    *,
    source_benchmark_json: str | None = None,
    k: int = 25,
    top_n: int = 5,
) -> dict[str, Any]:
    splits_out: dict[str, Any] = {}
    for split_name, split_payload in (benchmark_payload.get("splits") or {}).items():
        join = split_payload.get("join") or {}
        growth = split_payload.get("growth_summary") or {}
        warnings = list(split_payload.get("warnings") or [])
        splits_out[str(split_name)] = {
            "join": join,
            "growth_summary": growth,
            "warnings": warnings,
            "positive_growth": _cohort_summary(
                benchmark_payload,
                split=str(split_name),
                label_key="positive_growth",
                k=k,
                top_n=top_n,
            ),
            "growth_gt25pct": _cohort_summary(
                benchmark_payload,
                split=str(split_name),
                label_key="growth_gt25pct",
                k=k,
                top_n=top_n,
            ),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_benchmark_json": source_benchmark_json,
        "config": {
            "k": int(k),
            "top_n": int(top_n),
        },
        "target_source": benchmark_payload.get("target_source") or {},
        "splits": splits_out,
    }


def _format_pct(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed * 100:.2f}%"


def _render_table(lines: list[str], title: str, rows: list[dict[str, Any]], *, cohort_label: str) -> None:
    lines.append(f"#### {title}")
    if not rows:
        lines.append("")
        lines.append(f"No {cohort_label} rows available.")
        lines.append("")
        return
    lines.append("")
    lines.append(f"| {cohort_label} | precision@k | base rate | lift | labeled n |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row.get('cohort')} | "
            f"{_format_pct(row.get('precision_at_k'))} | "
            f"{_format_pct(row.get('positive_rate'))} | "
            f"{_format_pct(row.get('lift_vs_base'))} | "
            f"{int(row.get('n_labeled') or 0)} |"
        )
    lines.append("")


def write_future_value_diagnostics_report(
    payload: dict[str, Any],
    *,
    out_json: str,
    out_md: str,
) -> dict[str, str]:
    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Future Value Diagnostics")
    lines.append("")
    lines.append(f"- Generated: `{payload.get('generated_at_utc')}`")
    lines.append(f"- Source benchmark: `{payload.get('source_benchmark_json')}`")
    lines.append(f"- K: `{(payload.get('config') or {}).get('k')}`")
    lines.append("")

    for split_name, split_payload in (payload.get("splits") or {}).items():
        join = split_payload.get("join") or {}
        growth = split_payload.get("growth_summary") or {}
        lines.append(f"## {str(split_name).upper()} Split")
        lines.append("")
        lines.append(
            f"- Labeled coverage: `{join.get('labeled_rows', 0)}/{join.get('prediction_rows', 0)}` "
            f"(`{_format_pct(join.get('labeled_share'))}`)"
        )
        lines.append(
            f"- Positive growth rate: `{_format_pct(growth.get('positive_growth_rate'))}` | "
            f"`>=25%` growth rate: `{_format_pct(growth.get('growth_gt25pct_rate'))}`"
        )
        warnings = split_payload.get("warnings") or []
        if warnings:
            lines.append(f"- Warnings: `{', '.join(str(item) for item in warnings)}`")
        lines.append("")

        for label_key, title in (
            ("positive_growth", "Positive Growth"),
            ("growth_gt25pct", "Growth >= 25%"),
        ):
            block = split_payload.get(label_key) or {}
            lines.append(f"### {title}")
            lines.append("")
            _render_table(lines, "Best Leagues", ((block.get("league") or {}).get("best") or []), cohort_label="league")
            _render_table(lines, "Weakest Leagues", ((block.get("league") or {}).get("worst") or []), cohort_label="league")
            _render_table(lines, "Best Positions", ((block.get("position") or {}).get("best") or []), cohort_label="position")
            _render_table(lines, "Weakest Positions", ((block.get("position") or {}).get("worst") or []), cohort_label="position")
            _render_table(
                lines,
                "Best Value Segments",
                ((block.get("value_segment") or {}).get("best") or []),
                cohort_label="value segment",
            )
            _render_table(
                lines,
                "Weakest Value Segments",
                ((block.get("value_segment") or {}).get("worst") or []),
                cohort_label="value segment",
            )

    md_path = Path(out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
