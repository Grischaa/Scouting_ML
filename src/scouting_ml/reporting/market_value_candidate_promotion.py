from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _find_files(pattern: str) -> list[Path]:
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        matches = glob.glob(str(Path.cwd() / pattern), recursive=True)
    return sorted({Path(match).resolve() for match in matches if Path(match).is_file()})


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _metric_segments(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = metrics.get("segments", {}).get("test", [])
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("segment") or "").strip()
        if label:
            out[label] = row
    return out


def _weighted_under_20m_wmape(metrics: dict[str, Any]) -> float | None:
    segments = _metric_segments(metrics)
    under = segments.get("under_5m")
    mid = segments.get("5m_to_20m")
    if not under or not mid:
        return None
    under_n = int(under.get("n_samples") or 0)
    mid_n = int(mid.get("n_samples") or 0)
    total = under_n + mid_n
    if total <= 0:
        return None
    under_wmape = _safe_float(under.get("wmape"))
    mid_wmape = _safe_float(mid.get("wmape"))
    if under_wmape is None or mid_wmape is None:
        return None
    return ((under_wmape * under_n) + (mid_wmape * mid_n)) / total


def _holdout_rows(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _find_files(pattern):
        payload = _load_json(path)
        overall = payload.get("overall", {}) if isinstance(payload.get("overall"), dict) else {}
        rows.append(
            {
                "league": str(payload.get("league") or ""),
                "status": str(payload.get("status") or "ok"),
                "n_samples": int(payload.get("n_samples") or overall.get("n_samples") or 0),
                "r2": _safe_float(overall.get("r2")),
                "wmape": _safe_float(overall.get("wmape")),
                "mape": _safe_float(overall.get("mape")),
                "mae_eur": _safe_float(overall.get("mae_eur")),
                "metrics_json": str(path),
            }
        )
    rows.sort(key=lambda row: row["league"])
    return rows


def _summarize_holdouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return {
            "total": len(rows),
            "ok_count": 0,
            "mean_r2": None,
            "weighted_wmape": None,
            "weighted_mae_eur": None,
            "rows": rows,
        }

    total_n = sum(int(row.get("n_samples") or 0) for row in ok_rows)
    if total_n <= 0:
        total_n = len(ok_rows)
        weight = lambda row: 1.0
    else:
        weight = lambda row: float(row.get("n_samples") or 0)

    r2_vals = [float(row["r2"]) for row in ok_rows if row.get("r2") is not None]
    wmape_num = sum(float(row["wmape"]) * weight(row) for row in ok_rows if row.get("wmape") is not None)
    mae_num = sum(float(row["mae_eur"]) * weight(row) for row in ok_rows if row.get("mae_eur") is not None)

    return {
        "total": len(rows),
        "ok_count": len(ok_rows),
        "mean_r2": (sum(r2_vals) / len(r2_vals)) if r2_vals else None,
        "weighted_wmape": (wmape_num / total_n) if total_n > 0 else None,
        "weighted_mae_eur": (mae_num / total_n) if total_n > 0 else None,
        "rows": rows,
    }


def _path_label(path: str | Path) -> str:
    return Path(path).stem or "unknown"


def _future_benchmark_summary(
    path: str | Path | None,
    *,
    split: str = "val",
    label_key: str = "positive_growth",
    k: int = 25,
) -> dict[str, Any] | None:
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    payload = _load_json(path_obj)
    split_payload = payload.get("splits", {}).get(split, {})
    join = split_payload.get("join", {}) if isinstance(split_payload.get("join"), dict) else {}
    precision_rows = split_payload.get("precision_at_k", {}).get(label_key, [])
    overall_row = None
    for row in precision_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("cohort_type")) == "overall" and int(row.get("k") or 0) == int(k):
            overall_row = row
            break
    return {
        "path": str(path_obj),
        "split": split,
        "label_key": label_key,
        "k": int(k),
        "labeled_rows": int(join.get("labeled_rows") or 0),
        "labeled_share": _safe_float(join.get("labeled_share")),
        "warnings": split_payload.get("warnings") or [],
        "precision_at_k": _safe_float((overall_row or {}).get("precision_at_k")),
        "positive_rate": _safe_float((overall_row or {}).get("positive_rate")),
        "lift_vs_base": _safe_float((overall_row or {}).get("lift_vs_base")),
    }


def build_candidate_promotion_payload(
    *,
    champion_metrics_path: str,
    candidate_metrics_path: str,
    candidate_holdout_glob: str,
    reference_holdout_glob: str,
    candidate_future_benchmark_path: str | None = None,
    champion_future_benchmark_path: str | None = None,
    champion_label: str | None = None,
    candidate_label: str | None = None,
    reference_label: str = "full_reference",
    require_test_wmape_improvement: bool = True,
    require_under_20m_wmape_improvement: bool = True,
    require_holdout_wmape_improvement: bool = True,
    require_future_benchmark: bool = False,
    require_future_precision_vs_champion: bool = False,
    future_split: str = "val",
    future_label_key: str = "positive_growth",
    future_k: int = 25,
    future_min_label_coverage: float = 0.25,
) -> dict[str, Any]:
    champion = _load_json(champion_metrics_path)
    candidate = _load_json(candidate_metrics_path)
    champion_test = champion.get("overall", {}).get("test", {})
    candidate_test = candidate.get("overall", {}).get("test", {})
    champion_under_20m = _weighted_under_20m_wmape(champion)
    candidate_under_20m = _weighted_under_20m_wmape(candidate)
    candidate_holdout = _summarize_holdouts(_holdout_rows(candidate_holdout_glob))
    reference_holdout = _summarize_holdouts(_holdout_rows(reference_holdout_glob))
    candidate_future = _future_benchmark_summary(
        candidate_future_benchmark_path,
        split=future_split,
        label_key=future_label_key,
        k=future_k,
    )
    champion_future = _future_benchmark_summary(
        champion_future_benchmark_path,
        split=future_split,
        label_key=future_label_key,
        k=future_k,
    )

    gates = {
        "test_wmape": (
            _safe_float(candidate_test.get("wmape")) is not None
            and _safe_float(champion_test.get("wmape")) is not None
            and float(candidate_test["wmape"]) <= float(champion_test["wmape"])
        ),
        "under_20m_wmape": (
            candidate_under_20m is not None
            and champion_under_20m is not None
            and float(candidate_under_20m) <= float(champion_under_20m)
        ),
        "holdout_weighted_wmape": (
            candidate_holdout.get("weighted_wmape") is not None
            and reference_holdout.get("weighted_wmape") is not None
            and float(candidate_holdout["weighted_wmape"]) <= float(reference_holdout["weighted_wmape"])
        ),
        "future_label_coverage": (
            candidate_future is not None
            and candidate_future.get("labeled_share") is not None
            and float(candidate_future["labeled_share"]) >= float(future_min_label_coverage)
        ),
        "future_precision_vs_base": (
            candidate_future is not None
            and candidate_future.get("precision_at_k") is not None
            and candidate_future.get("positive_rate") is not None
            and float(candidate_future["precision_at_k"]) > float(candidate_future["positive_rate"])
        ),
        "future_precision_vs_champion": (
            candidate_future is not None
            and champion_future is not None
            and candidate_future.get("precision_at_k") is not None
            and champion_future.get("precision_at_k") is not None
            and float(candidate_future["precision_at_k"]) >= float(champion_future["precision_at_k"])
        ),
    }

    required = []
    if require_test_wmape_improvement:
        required.append("test_wmape")
    if require_under_20m_wmape_improvement:
        required.append("under_20m_wmape")
    if require_holdout_wmape_improvement:
        required.append("holdout_weighted_wmape")
    if require_future_benchmark:
        required.extend(["future_label_coverage", "future_precision_vs_base"])
    if require_future_precision_vs_champion:
        required.append("future_precision_vs_champion")

    passed = all(bool(gates[name]) for name in required)
    if passed:
        reason = "Candidate passed all configured promotion gates."
    else:
        failed = [name for name in required if not gates.get(name)]
        reason = "Candidate failed promotion gates: " + ", ".join(failed)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "champion_metrics_path": champion_metrics_path,
            "candidate_metrics_path": candidate_metrics_path,
            "candidate_holdout_glob": candidate_holdout_glob,
            "reference_holdout_glob": reference_holdout_glob,
            "candidate_future_benchmark_path": candidate_future_benchmark_path,
            "champion_future_benchmark_path": champion_future_benchmark_path,
        },
        "champion": {
            "label": champion_label or _path_label(champion_metrics_path),
            "test": champion_test,
            "segments_test": champion.get("segments", {}).get("test", []),
            "under_20m_wmape": champion_under_20m,
        },
        "candidate": {
            "label": candidate_label or _path_label(candidate_metrics_path),
            "test": candidate_test,
            "segments_test": candidate.get("segments", {}).get("test", []),
            "under_20m_wmape": candidate_under_20m,
            "holdout": candidate_holdout,
        },
        "reference_holdout": {
            "label": reference_label,
            **reference_holdout,
        },
        "future_benchmark": {
            "config": {
                "split": future_split,
                "label_key": future_label_key,
                "k": int(future_k),
                "min_label_coverage": float(future_min_label_coverage),
            },
            "candidate": candidate_future,
            "champion": champion_future,
        },
        "decision": {
            "required_gates": required,
            "gates": gates,
            "passed": passed,
            "reason": reason,
        },
    }


def candidate_passes_promotion_gates(payload: dict[str, Any]) -> bool:
    return bool(payload.get("decision", {}).get("passed"))


def write_candidate_promotion_report(
    payload: dict[str, Any],
    *,
    out_json: str,
    out_md: str,
) -> dict[str, str]:
    out_json_path = Path(out_json)
    out_md_path = Path(out_md)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    champion = payload.get("champion", {})
    candidate = payload.get("candidate", {})
    reference = payload.get("reference_holdout", {})
    future = payload.get("future_benchmark", {})
    decision = payload.get("decision", {})

    champ_test = champion.get("test", {})
    cand_test = candidate.get("test", {})
    cand_holdout = candidate.get("holdout", {})
    cand_future = future.get("candidate") or {}
    champ_future = future.get("champion") or {}

    def _pct(value: Any) -> str:
        val = _safe_float(value)
        return "n/a" if val is None else f"{val * 100:.2f}%"

    def _eur(value: Any) -> str:
        val = _safe_float(value)
        return "n/a" if val is None else f"EUR {val:,.0f}"

    lines = [
        "# Market Value Candidate Promotion",
        "",
        "## Overall Test",
        f"- {champion.get('label')}: R2 {_pct(champ_test.get('r2'))} | WMAPE {_pct(champ_test.get('wmape'))} | MAE {_eur(champ_test.get('mae_eur'))}",
        f"- {candidate.get('label')}: R2 {_pct(cand_test.get('r2'))} | WMAPE {_pct(cand_test.get('wmape'))} | MAE {_eur(cand_test.get('mae_eur'))}",
        "",
        "## Cheap-Player Focus",
        f"- {champion.get('label')} under_20m WMAPE: {_pct(champion.get('under_20m_wmape'))}",
        f"- {candidate.get('label')} under_20m WMAPE: {_pct(candidate.get('under_20m_wmape'))}",
        "",
        "## Holdout Comparison",
        f"- {candidate.get('label')}: mean R2 {_pct(cand_holdout.get('mean_r2'))} | weighted WMAPE {_pct(cand_holdout.get('weighted_wmape'))} | weighted MAE {_eur(cand_holdout.get('weighted_mae_eur'))}",
        f"- {reference.get('label')}: mean R2 {_pct(reference.get('mean_r2'))} | weighted WMAPE {_pct(reference.get('weighted_wmape'))} | weighted MAE {_eur(reference.get('weighted_mae_eur'))}",
    ]
    if cand_future:
        lines.extend(
            [
                "",
                "## Future Benchmark",
                f"- {candidate.get('label')}: labeled share {_pct(cand_future.get('labeled_share'))} | precision@{cand_future.get('k')} {_pct(cand_future.get('precision_at_k'))} | base {_pct(cand_future.get('positive_rate'))}",
            ]
        )
        if champ_future:
            lines.append(
                f"- {champion.get('label')}: labeled share {_pct(champ_future.get('labeled_share'))} | precision@{champ_future.get('k')} {_pct(champ_future.get('precision_at_k'))} | base {_pct(champ_future.get('positive_rate'))}"
            )
    lines.extend(["", "## Gates"])
    for gate_name in decision.get("required_gates", []):
        status = "pass" if decision.get("gates", {}).get(gate_name) else "fail"
        lines.append(f"- {gate_name}: {status}")
    lines.extend(
        [
            "",
            "## Decision",
            f"- passed: {bool(decision.get('passed'))}",
            f"- reason: {decision.get('reason')}",
        ]
    )

    out_md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(out_json_path), "markdown": str(out_md_path)}
