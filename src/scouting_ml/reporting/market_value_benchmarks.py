from __future__ import annotations

import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BIG5_LEAGUES = {
    "english premier league",
    "premier league",
    "spanish la liga",
    "la liga",
    "laliga",
    "italian serie a",
    "serie a",
    "german bundesliga",
    "bundesliga",
    "french ligue 1",
    "ligue 1",
}


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _season_slug(season: str | None) -> str | None:
    if not season:
        return None
    return str(season).strip().replace("/", "-").replace("\\", "-")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _find_files(pattern: str) -> list[Path]:
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        matches = glob.glob(str(Path.cwd() / pattern), recursive=True)
    return sorted({Path(match).resolve() for match in matches if Path(match).is_file()})


def _pick_latest_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def _resolve_latest_ablation_bundle(
    *,
    ablation_bundle_path: str | None = None,
    ablation_glob: str = "data/model/ablation/**/ablation_bundle_*.json",
    test_season: str | None = None,
) -> Path | None:
    if ablation_bundle_path:
        path = Path(ablation_bundle_path)
        return path if path.exists() else None

    candidates = _find_files(ablation_glob)
    if not candidates:
        return None

    season_slug = _season_slug(test_season)
    if season_slug:
        season_matches = [path for path in candidates if season_slug in path.name]
        if season_matches:
            return _pick_latest_file(season_matches)
    return _pick_latest_file(candidates)


def _coverage_rows_from_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    if not predictions_path.exists():
        return []

    frame = pd.read_csv(predictions_path, low_memory=False)
    if frame.empty or "league" not in frame.columns:
        return []

    work = frame.copy()
    work["league"] = work["league"].astype(str).str.strip()
    work = work[work["league"] != ""].copy()
    if work.empty:
        return []

    if "undervalued_flag" in work.columns:
        undervalued = pd.to_numeric(work["undervalued_flag"], errors="coerce").fillna(0.0) > 0.0
    else:
        conservative_gap = pd.to_numeric(work.get("value_gap_conservative_eur"), errors="coerce")
        undervalued = conservative_gap.fillna(0.0) > 0.0
    confidence = pd.to_numeric(work.get("undervaluation_confidence"), errors="coerce")
    age = pd.to_numeric(work.get("age"), errors="coerce")

    rows: list[dict[str, Any]] = []
    for league, group in work.groupby("league", dropna=False):
        idx = group.index
        league_norm = str(league).strip().casefold()
        conf = confidence.loc[idx]
        age_vals = age.loc[idx]
        rows.append(
            {
                "league": str(league),
                "rows": int(len(group)),
                "undervalued_share": float(undervalued.loc[idx].mean()),
                "avg_confidence": float(conf.mean()) if conf.notna().any() else None,
                "avg_age": float(age_vals.mean()) if age_vals.notna().any() else None,
                "is_big5": bool(league_norm in BIG5_LEAGUES),
            }
        )
    rows.sort(key=lambda row: (-int(row["rows"]), str(row["league"])))
    return rows


def _prediction_value_col(frame: pd.DataFrame) -> str | None:
    for col in ("fair_value_eur", "expected_value_eur"):
        if col in frame.columns:
            return col
    return None


def _r2_score(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    mask = y_true.notna() & y_pred.notna()
    if int(mask.sum()) < 2:
        return None
    yt = y_true.loc[mask].astype(float)
    yp = y_pred.loc[mask].astype(float)
    denom = float(((yt - yt.mean()) ** 2).sum())
    if denom <= 0.0:
        return None
    return float(1.0 - (((yt - yp) ** 2).sum() / denom))


def _prediction_league_rows(predictions_path: Path, *, mape_min_denom_eur: float = 1_000_000.0) -> list[dict[str, Any]]:
    if not predictions_path.exists():
        return []

    frame = pd.read_csv(predictions_path, low_memory=False)
    pred_col = _prediction_value_col(frame)
    if frame.empty or "league" not in frame.columns or "market_value_eur" not in frame.columns or pred_col is None:
        return []

    work = frame.copy()
    work["league"] = work["league"].astype(str).str.strip()
    work = work[work["league"] != ""].copy()
    if work.empty:
        return []

    work["_true"] = pd.to_numeric(work["market_value_eur"], errors="coerce")
    work["_pred"] = pd.to_numeric(work[pred_col], errors="coerce")
    work["_abs_error"] = (work["_true"] - work["_pred"]).abs()
    work["_denom"] = work["_true"].abs().clip(lower=float(mape_min_denom_eur))
    confidence = pd.to_numeric(work.get("undervaluation_confidence"), errors="coerce")

    rows: list[dict[str, Any]] = []
    for league, group in work.groupby("league", dropna=False):
        valid = group[group["_true"].notna() & group["_pred"].notna()].copy()
        if valid.empty:
            continue
        conf = confidence.loc[valid.index]
        denom_sum = float(valid["_denom"].sum())
        rows.append(
            {
                "league": str(league),
                "rows": int(len(valid)),
                "r2": _r2_score(valid["_true"], valid["_pred"]),
                "mae_eur": float(valid["_abs_error"].mean()),
                "mape": float((valid["_abs_error"] / valid["_denom"]).mean()),
                "wmape": float(valid["_abs_error"].sum() / denom_sum) if denom_sum > 0.0 else None,
                "avg_confidence": float(conf.mean()) if conf.notna().any() else None,
            }
        )
    rows.sort(key=lambda row: (-int(row["rows"]), str(row["league"])))
    return rows


def _summarize_prediction_leagues(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r2_values = [float(row["r2"]) for row in rows if row.get("r2") is not None]
    wmape_values = [float(row["wmape"]) for row in rows if row.get("wmape") is not None]
    return {
        "total": int(len(rows)),
        "mean_r2": float(np.mean(r2_values)) if r2_values else None,
        "median_r2": float(np.median(r2_values)) if r2_values else None,
        "median_wmape": float(np.median(wmape_values)) if wmape_values else None,
        "best_r2_rows": sorted(
            [row for row in rows if row.get("r2") is not None],
            key=lambda row: float(row["r2"]),
            reverse=True,
        )[:5],
        "worst_wmape_rows": sorted(
            [row for row in rows if row.get("wmape") is not None],
            key=lambda row: float(row["wmape"]),
            reverse=True,
        )[:5],
    }


def _holdout_rows_from_glob(pattern: str, onboarding_index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _find_files(pattern):
        payload = _load_json(path)
        if not payload:
            continue
        league = str(payload.get("league") or "").strip()
        if not league:
            continue
        overall = payload.get("overall", {}) if isinstance(payload.get("overall"), dict) else {}
        domain_shift = payload.get("domain_shift", {}) if isinstance(payload.get("domain_shift"), dict) else {}
        slug = _slugify(league)
        onboarding = onboarding_index.get(slug, {})
        rows.append(
            {
                "league": league,
                "status": str(payload.get("status") or "unknown"),
                "n_samples": payload.get("n_samples") or overall.get("n_samples"),
                "r2": _safe_float(overall.get("r2")),
                "wmape": _safe_float(overall.get("wmape")),
                "mape": _safe_float(overall.get("mape")),
                "mae_eur": _safe_float(overall.get("mae_eur")),
                "domain_shift_mean_abs_z": _safe_float(domain_shift.get("mean_abs_shift_z")),
                "metrics_json": str(path),
                "predictions_csv": payload.get("predictions_csv"),
                "onboarding_status": onboarding.get("status"),
                "onboarding_reasons": onboarding.get("reasons"),
            }
        )

    rows.sort(
        key=lambda row: (
            0 if str(row.get("status")) == "ok" else 1,
            1 if row.get("r2") is None else 0,
            -(row.get("r2") or -999.0),
            row.get("league") or "",
        )
    )
    return rows


def _summarize_holdouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if str(row.get("status")) == "ok"]
    r2_values = [float(row["r2"]) for row in ok_rows if row.get("r2") is not None]
    wmape_values = [float(row["wmape"]) for row in ok_rows if row.get("wmape") is not None]
    shift_values = [float(row["domain_shift_mean_abs_z"]) for row in ok_rows if row.get("domain_shift_mean_abs_z") is not None]

    def _top_rows(metric: str, *, reverse: bool, limit: int = 5) -> list[dict[str, Any]]:
        metric_rows = [row for row in ok_rows if row.get(metric) is not None]
        metric_rows.sort(key=lambda row: float(row[metric]), reverse=reverse)
        return metric_rows[:limit]

    return {
        "total": int(len(rows)),
        "ok_count": int(sum(1 for row in rows if str(row.get("status")) == "ok")),
        "skipped_count": int(sum(1 for row in rows if str(row.get("status")) == "skipped")),
        "mean_r2": float(np.mean(r2_values)) if r2_values else None,
        "median_wmape": float(np.median(wmape_values)) if wmape_values else None,
        "mean_domain_shift_abs_z": float(np.mean(shift_values)) if shift_values else None,
        "best_r2_rows": _top_rows("r2", reverse=True),
        "worst_wmape_rows": _top_rows("wmape", reverse=True),
        "highest_shift_rows": _top_rows("domain_shift_mean_abs_z", reverse=True),
    }


def _load_onboarding_summary(onboarding_json_path: str | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not onboarding_json_path:
        return {"status_counts": {}, "items": []}, {}
    payload = _load_json(Path(onboarding_json_path))
    if not payload:
        return {"status_counts": {}, "items": []}, {}

    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    index: dict[str, dict[str, Any]] = {}
    ready: list[str] = []
    blocked: list[str] = []
    watch: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        slug = _slugify(str(item.get("league_slug") or item.get("matched_holdout_slug") or ""))
        if slug:
            index[slug] = item
        status = str(item.get("status") or "")
        league_slug = str(item.get("league_slug") or "")
        if status == "ready":
            ready.append(league_slug)
        elif status == "blocked":
            blocked.append(league_slug)
        elif status == "watch":
            watch.append(league_slug)

    return (
        {
            "path": onboarding_json_path,
            "status_counts": payload.get("status_counts") or {},
            "ready_leagues": ready,
            "blocked_leagues": blocked,
            "watch_leagues": watch,
            "items": items,
        },
        index,
    )


def _load_ablation_summary(bundle_path: Path | None) -> dict[str, Any]:
    if bundle_path is None:
        return {"available": False}
    payload = _load_json(bundle_path)
    if not payload:
        return {"available": False, "path": str(bundle_path)}
    return {
        "available": True,
        "path": str(bundle_path),
        "best_overall_test": payload.get("best_overall_test"),
        "best_under_20m_test": payload.get("best_under_20m_test"),
        "best_under_5m_test": payload.get("best_under_5m_test"),
        "position_winners_test": payload.get("position_winners_test") or [],
        "value_segment_winners_test": payload.get("value_segment_winners_test") or [],
        "weakest_full_slices_test": payload.get("weakest_full_slices_test") or [],
        "artifacts": payload.get("artifacts") or {},
    }


def build_market_value_benchmark_payload(
    *,
    metrics_path: str,
    predictions_path: str,
    holdout_metrics_glob: str = "data/model/**/*.holdout_*.metrics.json",
    onboarding_json_path: str | None = "data/model/onboarding/non_big5_onboarding_report.json",
    ablation_bundle_path: str | None = None,
    ablation_glob: str = "data/model/ablation/**/ablation_bundle_*.json",
) -> dict[str, Any]:
    metrics_payload = _load_json(Path(metrics_path))
    if not metrics_payload:
        raise FileNotFoundError(f"Metrics payload not found or unreadable: {metrics_path}")

    test_season = str(metrics_payload.get("test_season") or "")
    ablation_path = _resolve_latest_ablation_bundle(
        ablation_bundle_path=ablation_bundle_path,
        ablation_glob=ablation_glob,
        test_season=test_season or None,
    )
    onboarding_summary, onboarding_index = _load_onboarding_summary(onboarding_json_path)
    holdout_rows = _holdout_rows_from_glob(holdout_metrics_glob, onboarding_index)
    coverage_rows = _coverage_rows_from_predictions(Path(predictions_path))
    prediction_league_rows = _prediction_league_rows(Path(predictions_path))

    overall_test = metrics_payload.get("overall", {}).get("test", {}) if isinstance(metrics_payload.get("overall"), dict) else {}
    overall_val = metrics_payload.get("overall", {}).get("val", {}) if isinstance(metrics_payload.get("overall"), dict) else {}

    summary = {
        "dataset": metrics_payload.get("dataset"),
        "val_season": metrics_payload.get("val_season"),
        "test_season": metrics_payload.get("test_season"),
        "test_r2": _safe_float(overall_test.get("r2")),
        "test_mae_eur": _safe_float(overall_test.get("mae_eur")),
        "test_mape": _safe_float(overall_test.get("mape")),
        "test_wmape": _safe_float(overall_test.get("wmape")),
        "val_r2": _safe_float(overall_val.get("r2")),
        "val_mae_eur": _safe_float(overall_val.get("mae_eur")),
        "val_mape": _safe_float(overall_val.get("mape")),
        "val_wmape": _safe_float(overall_val.get("wmape")),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "metrics_path": str(metrics_path),
            "predictions_path": str(predictions_path),
            "holdout_metrics_glob": holdout_metrics_glob,
            "onboarding_json_path": onboarding_json_path,
            "ablation_bundle_path": str(ablation_path) if ablation_path else None,
        },
        "model": summary,
        "league_holdout": {
            "summary": _summarize_holdouts(holdout_rows),
            "rows": holdout_rows,
        },
        "prediction_league": {
            "summary": _summarize_prediction_leagues(prediction_league_rows),
            "rows": prediction_league_rows,
        },
        "onboarding": onboarding_summary,
        "ablation": _load_ablation_summary(ablation_path),
        "coverage": {
            "split": "test",
            "rows": coverage_rows,
        },
    }


def write_market_value_benchmark_report(
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

    model = payload.get("model", {})
    holdout = payload.get("league_holdout", {})
    holdout_summary = holdout.get("summary", {}) if isinstance(holdout, dict) else {}
    holdout_rows = holdout.get("rows", []) if isinstance(holdout, dict) else []
    prediction_league = payload.get("prediction_league", {})
    prediction_league_summary = prediction_league.get("summary", {}) if isinstance(prediction_league, dict) else {}
    prediction_league_rows = prediction_league.get("rows", []) if isinstance(prediction_league, dict) else []
    onboarding = payload.get("onboarding", {})
    ablation = payload.get("ablation", {})
    coverage = payload.get("coverage", {})
    coverage_rows = coverage.get("rows", []) if isinstance(coverage, dict) else []

    lines: list[str] = []
    lines.append("# Market Value Benchmark Report")
    lines.append("")
    lines.append(f"- Generated: `{payload.get('generated_at_utc')}`")
    lines.append(f"- Dataset: `{model.get('dataset')}`")
    lines.append(f"- Validation season: `{model.get('val_season')}`")
    lines.append(f"- Test season: `{model.get('test_season')}`")
    lines.append("")
    lines.append("## Core Model Metrics")
    lines.append(
        f"- Test: R2={_safe_float(model.get('test_r2')) if model.get('test_r2') is not None else 'n/a'} | "
        f"WMAPE={_safe_float(model.get('test_wmape')) if model.get('test_wmape') is not None else 'n/a'}"
    )
    lines.append(
        f"- Validation: R2={_safe_float(model.get('val_r2')) if model.get('val_r2') is not None else 'n/a'} | "
        f"WMAPE={_safe_float(model.get('val_wmape')) if model.get('val_wmape') is not None else 'n/a'}"
    )
    lines.append("")

    lines.append("## League Holdout Summary")
    lines.append(
        f"- Holdouts: {holdout_summary.get('total', 0)} total | "
        f"{holdout_summary.get('ok_count', 0)} ok | "
        f"{holdout_summary.get('skipped_count', 0)} skipped"
    )
    lines.append(
        f"- Mean R2: {holdout_summary.get('mean_r2') if holdout_summary.get('mean_r2') is not None else 'n/a'} | "
        f"Median WMAPE: {holdout_summary.get('median_wmape') if holdout_summary.get('median_wmape') is not None else 'n/a'}"
    )
    lines.append("")
    if holdout_rows:
        lines.append("| League | Status | R2 | WMAPE | Domain Shift | Onboarding |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for row in holdout_rows[:10]:
            lines.append(
                f"| {row.get('league')} | {row.get('status')} | "
                f"{row.get('r2') if row.get('r2') is not None else '-'} | "
                f"{row.get('wmape') if row.get('wmape') is not None else '-'} | "
                f"{row.get('domain_shift_mean_abs_z') if row.get('domain_shift_mean_abs_z') is not None else '-'} | "
                f"{row.get('onboarding_status') or '-'} |"
            )
        lines.append("")

    lines.append("## Prediction-League Benchmarks")
    lines.append(
        f"- Leagues: {prediction_league_summary.get('total', 0)} | "
        f"Median R2: {prediction_league_summary.get('median_r2') if prediction_league_summary.get('median_r2') is not None else 'n/a'} | "
        f"Median WMAPE: {prediction_league_summary.get('median_wmape') if prediction_league_summary.get('median_wmape') is not None else 'n/a'}"
    )
    lines.append("")
    if prediction_league_rows:
        lines.append("| League | Rows | R2 | WMAPE | Avg Confidence |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in prediction_league_rows[:10]:
            lines.append(
                f"| {row.get('league')} | {row.get('rows')} | "
                f"{row.get('r2') if row.get('r2') is not None else '-'} | "
                f"{row.get('wmape') if row.get('wmape') is not None else '-'} | "
                f"{row.get('avg_confidence') if row.get('avg_confidence') is not None else '-'} |"
            )
        lines.append("")

    lines.append("## Onboarding Snapshot")
    status_counts = onboarding.get("status_counts") if isinstance(onboarding, dict) else {}
    if isinstance(status_counts, dict) and status_counts:
        lines.append(
            "- Status counts: "
            + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items()))
        )
    else:
        lines.append("- No onboarding report loaded.")
    lines.append("")

    lines.append("## Ablation Snapshot")
    if ablation.get("available"):
        best = ablation.get("best_overall_test") or {}
        cheap = ablation.get("best_under_20m_test") or {}
        lines.append(
            f"- Best overall: `{best.get('config', 'n/a')}` | {best.get('metric', 'metric')}={best.get('value', 'n/a')}"
        )
        lines.append(
            f"- Best under-20m: `{cheap.get('config', 'n/a')}` | {cheap.get('metric', 'metric')}={cheap.get('value', 'n/a')}"
        )
    else:
        lines.append("- No ablation bundle available.")
    lines.append("")

    lines.append("## Prediction Coverage Snapshot")
    if coverage_rows:
        lines.append("| League | Rows | Undervalued % | Avg Confidence | Avg Age |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in coverage_rows[:10]:
            undervalued_share = row.get("undervalued_share")
            avg_confidence = row.get("avg_confidence")
            avg_age = row.get("avg_age")
            lines.append(
                f"| {row.get('league')} | {row.get('rows')} | "
                f"{round(100.0 * float(undervalued_share), 2) if undervalued_share is not None else '-'} | "
                f"{round(float(avg_confidence), 3) if avg_confidence is not None else '-'} | "
                f"{round(float(avg_age), 2) if avg_age is not None else '-'} |"
            )
    else:
        lines.append("- No prediction coverage rows available.")
    lines.append("")

    out_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": str(out_json_path),
        "markdown": str(out_md_path),
    }
