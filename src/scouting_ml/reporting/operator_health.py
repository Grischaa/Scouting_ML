from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from scouting_ml.core.runtime_config import PRODUCTION_PIPELINE_DEFAULTS
from scouting_ml.league_registry import LEAGUES, LeagueConfig, season_slug_label
from scouting_ml.paths import PROCESSED_DIR

INGESTION_HEALTH_CSV = PROCESSED_DIR / "ingestion_health_report.csv"
INGESTION_HEALTH_JSON = PROCESSED_DIR / "ingestion_health_report.json"
INGESTION_META_SUFFIX = ".meta.json"
MIN_SUCCESSFUL_HOLDOUTS = 6

PROVIDER_DATE_COLUMNS: dict[str, str] = {
    "sb": "sb_snapshot_date",
    "avail": "avail_snapshot_date",
    "fixture": "fixture_snapshot_date",
    "odds": "odds_snapshot_date",
}
PROVIDER_RETRIEVED_COLUMNS: dict[str, str] = {
    "sb": "sb_retrieved_at",
    "avail": "avail_retrieved_at",
    "fixture": "fixture_retrieved_at",
    "odds": "odds_retrieved_at",
}


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + INGESTION_META_SUFFIX)


def write_json_sidecar(path: Path, payload: dict[str, Any]) -> Path:
    target = _json_sidecar_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def read_json_sidecar(path: Path) -> dict[str, Any] | None:
    target = _json_sidecar_path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_timestamp(values: Iterable[Any]) -> str | None:
    best_value: str | None = None
    best_dt: datetime | None = None
    for raw in values:
        text = _safe_text(raw)
        if not text:
            continue
        try:
            normalized = f"{text}T00:00:00+00:00" if len(text) == 10 and text[4] == "-" and text[7] == "-" else text
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except Exception:
            continue
        if best_dt is None or parsed > best_dt:
            best_dt = parsed
            best_value = text
    return best_value


def _csv_header_and_rows(path: Path) -> tuple[list[str], int, bool]:
    if not path.exists():
        return [], 0, False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            first_line = handle.readline()
            has_header = bool(first_line.strip())
            header = next(csv.reader([first_line])) if has_header else []
            rows = sum(1 for _ in handle)
    except OSError:
        return [], 0, False
    return header, int(rows), has_header


def _read_selected_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists() or not columns:
        return pd.DataFrame()
    header, _, has_header = _csv_header_and_rows(path)
    if not has_header:
        return pd.DataFrame()
    usecols = [col for col in columns if col in header]
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(path, usecols=usecols)


def _coerce_numeric(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def _clean_dataset_provider_dates(clean_dataset_path: Path | None) -> dict[tuple[str, str], dict[str, str | None]]:
    if clean_dataset_path is None or not clean_dataset_path.exists():
        return {}
    try:
        frame = pd.read_parquet(clean_dataset_path)
    except Exception:
        return {}
    if frame.empty or "league" not in frame.columns or "season" not in frame.columns:
        return {}

    wanted = {"league", "season", *PROVIDER_DATE_COLUMNS.values(), *PROVIDER_RETRIEVED_COLUMNS.values()}
    frame = frame[[col for col in frame.columns if col in wanted]].copy()
    out: dict[tuple[str, str], dict[str, str | None]] = {}
    for (league, season), group in frame.groupby(["league", "season"], dropna=False):
        league_name = _safe_text(league)
        season_label = _safe_text(season)
        if not league_name or not season_label:
            continue
        payload: dict[str, str | None] = {}
        for provider, col in PROVIDER_DATE_COLUMNS.items():
            payload[col] = latest_timestamp(group[col].tolist()) if col in group.columns else None
        for provider, col in PROVIDER_RETRIEVED_COLUMNS.items():
            payload[col] = latest_timestamp(group[col].tolist()) if col in group.columns else None
        out[(league_name, season_label)] = payload
    return out


def _league_season_row(
    *,
    config: LeagueConfig,
    season: str,
    processed_dir: Path,
    provider_dates_by_key: dict[tuple[str, str], dict[str, str | None]],
) -> dict[str, Any]:
    season_label = season_slug_label(season)
    tm_path = processed_dir / f"{config.slug}_{season_label}_clean.csv"
    sofa_path = processed_dir / f"sofa_{config.slug}_{season_label}.csv"
    merged_path = processed_dir / Path(config.guess_processed_dataset(season)).name

    tm_header, tm_rows, tm_has_header = _csv_header_and_rows(tm_path)
    sofa_header, sofa_rows_fallback, sofa_has_header = _csv_header_and_rows(sofa_path)
    merged_header, merged_rows_fallback, merged_has_header = _csv_header_and_rows(merged_path)

    sofa_meta = read_json_sidecar(sofa_path) or {}
    merged_meta = read_json_sidecar(merged_path) or {}

    tm_rows_value = int(tm_rows)
    sofa_rows = int(sofa_meta.get("rows", sofa_rows_fallback) or 0)
    sofa_zero_rows = bool(sofa_meta.get("zero_rows", sofa_rows == 0 and sofa_has_header))
    sofa_header_only = bool(sofa_meta.get("header_only", sofa_zero_rows and sofa_has_header))

    matched_rows = _coerce_numeric(merged_meta.get("matched_rows"))
    match_rate = _coerce_numeric(merged_meta.get("match_rate"))
    if matched_rows is None and merged_path.exists() and "sofa_matched" in merged_header:
        merged_sample = _read_selected_columns(merged_path, ["sofa_matched"])
        if not merged_sample.empty:
            matched_rows = float(pd.to_numeric(merged_sample["sofa_matched"], errors="coerce").fillna(0).sum())
    if match_rate is None and matched_rows is not None and tm_rows_value > 0:
        match_rate = float(matched_rows) / float(tm_rows_value)

    provider_dates = dict(
        provider_dates_by_key.get((config.name, season), {})
    )
    latest_provider_snapshot_date = latest_timestamp(
        [provider_dates.get(col) for col in PROVIDER_DATE_COLUMNS.values()]
    )
    missing_provider_flags = [
        provider
        for provider, col in PROVIDER_DATE_COLUMNS.items()
        if not _safe_text(provider_dates.get(col))
    ]

    reasons: list[str] = []
    if not tm_path.exists():
        reasons.append("tm_clean_missing")
    elif tm_rows_value <= 0:
        reasons.append("tm_clean_zero_rows")

    if not sofa_path.exists():
        reasons.append("sofa_csv_missing")
    elif sofa_zero_rows:
        reasons.append("sofa_zero_rows")

    if not merged_path.exists():
        reasons.append("merged_dataset_missing")
    elif not merged_has_header:
        reasons.append("merged_dataset_empty")

    if match_rate is not None:
        if match_rate < 0.20:
            reasons.append("match_rate_low")
        elif match_rate < 0.55:
            reasons.append("match_rate_watch")
    elif merged_path.exists() and tm_rows_value > 0:
        reasons.append("match_rate_unknown")

    reasons.extend([f"missing_provider_snapshot:{flag}" for flag in missing_provider_flags])

    if any(reason in reasons for reason in ("tm_clean_missing", "tm_clean_zero_rows", "sofa_csv_missing", "sofa_zero_rows", "merged_dataset_missing", "merged_dataset_empty", "match_rate_low")):
        status = "blocked"
    elif reasons:
        status = "watch"
    else:
        status = "healthy"

    return {
        "league_slug": config.slug,
        "league_name": config.name,
        "season": season,
        "tm_rows": tm_rows_value,
        "sofa_rows": int(sofa_rows),
        "matched_rows": int(matched_rows) if matched_rows is not None else None,
        "match_rate": float(match_rate) if match_rate is not None else None,
        "sofa_zero_rows": bool(sofa_zero_rows),
        "sofa_header_only": bool(sofa_header_only),
        "latest_provider_snapshot_date": latest_provider_snapshot_date,
        "sb_snapshot_date": _safe_text(provider_dates.get("sb_snapshot_date")),
        "avail_snapshot_date": _safe_text(provider_dates.get("avail_snapshot_date")),
        "fixture_snapshot_date": _safe_text(provider_dates.get("fixture_snapshot_date")),
        "odds_snapshot_date": _safe_text(provider_dates.get("odds_snapshot_date")),
        "missing_provider_flags": missing_provider_flags,
        "status": status,
        "status_reasons": reasons,
        "tm_clean_path": str(tm_path.resolve()),
        "sofa_path": str(sofa_path.resolve()),
        "merged_path": str(merged_path.resolve()),
    }


def build_ingestion_health_payload(
    *,
    processed_dir: Path = PROCESSED_DIR,
    clean_dataset_path: Path | None = None,
) -> dict[str, Any]:
    effective_clean_path = clean_dataset_path or Path(PRODUCTION_PIPELINE_DEFAULTS.clean_output)
    provider_dates_by_key = _clean_dataset_provider_dates(effective_clean_path if effective_clean_path.exists() else None)
    rows = [
        _league_season_row(
            config=config,
            season=season,
            processed_dir=processed_dir,
            provider_dates_by_key=provider_dates_by_key,
        )
        for config in LEAGUES.values()
        for season in config.seasons
    ]
    status_counts = {"healthy": 0, "watch": 0, "blocked": 0}
    for row in rows:
        status_counts[str(row["status"])] += 1
    payload = {
        "generated_at_utc": _now_utc_iso(),
        "rows": rows,
        "summary": {
            "total": len(rows),
            "status_counts": status_counts,
            "configured_leagues": len(LEAGUES),
            "healthy_rows": status_counts["healthy"],
            "watch_rows": status_counts["watch"],
            "blocked_rows": status_counts["blocked"],
            "processed_dir": str(processed_dir.resolve()),
            "clean_dataset_path": str(effective_clean_path.resolve()) if effective_clean_path.exists() else None,
        },
    }
    return payload


def write_ingestion_health_report(
    payload: dict[str, Any],
    *,
    out_csv: Path = INGESTION_HEALTH_CSV,
    out_json: Path = INGESTION_HEALTH_JSON,
) -> dict[str, str]:
    rows = list(payload.get("rows") or [])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_rows.append(
            {
                **row,
                "missing_provider_flags": ",".join(row.get("missing_provider_flags") or []),
                "status_reasons": ",".join(row.get("status_reasons") or []),
            }
        )
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    return {
        "csv": str(out_csv.resolve()),
        "json": str(out_json.resolve()),
    }


def regenerate_ingestion_health_report(
    *,
    processed_dir: Path = PROCESSED_DIR,
    clean_dataset_path: Path | None = None,
    out_csv: Path = INGESTION_HEALTH_CSV,
    out_json: Path = INGESTION_HEALTH_JSON,
) -> dict[str, Any]:
    payload = build_ingestion_health_payload(
        processed_dir=processed_dir,
        clean_dataset_path=clean_dataset_path,
    )
    report_paths = write_ingestion_health_report(payload, out_csv=out_csv, out_json=out_json)
    payload["_meta"] = {
        "source": "derived",
        "csv_path": report_paths["csv"],
        "json_path": report_paths["json"],
    }
    return payload


def load_ingestion_health_payload(
    *,
    report_json: Path = INGESTION_HEALTH_JSON,
    clean_dataset_path: Path | None = None,
) -> dict[str, Any]:
    if report_json.exists():
        payload = json.loads(report_json.read_text(encoding="utf-8"))
        payload["_meta"] = {
            "source": "file",
            "json_path": str(report_json.resolve()),
        }
        return payload
    return regenerate_ingestion_health_report(clean_dataset_path=clean_dataset_path)


def _backtest_rows_frame(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_valuation_promotion_gate(
    *,
    metrics_payload: dict[str, Any] | None,
    backtest_payload: dict[str, Any] | None,
    backtest_rows_path: Path | None,
    requested_backtest_test_seasons: Sequence[str] | None,
    min_successful_holdouts: int = MIN_SUCCESSFUL_HOLDOUTS,
    min_test_r2: float = PRODUCTION_PIPELINE_DEFAULTS.backtest_min_test_r2,
    max_test_wmape: float = PRODUCTION_PIPELINE_DEFAULTS.backtest_max_test_wmape,
    max_under5m_wmape: float = PRODUCTION_PIPELINE_DEFAULTS.backtest_max_under5m_wmape,
    max_lowmid_weighted_wmape: float = PRODUCTION_PIPELINE_DEFAULTS.backtest_max_lowmid_weighted_wmape,
    max_segment_weighted_wmape: float = PRODUCTION_PIPELINE_DEFAULTS.backtest_max_segment_weighted_wmape,
) -> dict[str, Any]:
    metrics = metrics_payload if isinstance(metrics_payload, dict) else {}
    backtest = backtest_payload if isinstance(backtest_payload, dict) else {}
    requested_holdouts = list(metrics.get("requested_league_holdouts") or [])
    holdout_results = list(metrics.get("league_holdout") or [])
    requested_tokens = [str(item.get("requested_token") or item.get("league") or "").strip() for item in requested_holdouts]
    requested_tokens = [token for token in requested_tokens if token]

    successful_holdouts = []
    skipped_holdouts = []
    completed_holdout_keys: set[str] = set()
    for result in holdout_results:
        key = str(result.get("requested_token") or result.get("league") or "").strip()
        if key:
            completed_holdout_keys.add(key)
        if str(result.get("status") or "ok").strip().lower() == "skipped":
            skipped_holdouts.append(key or str(result.get("league") or "unknown"))
        else:
            successful_holdouts.append(str(result.get("league") or key or "unknown"))
    missing_requested_holdouts = [token for token in requested_tokens if token not in completed_holdout_keys]

    failed_checks: list[str] = []
    warnings: list[str] = []

    if skipped_holdouts:
        failed_checks.append(
            "Skipped holdouts remain in the requested valuation holdout suite: " + ", ".join(sorted(set(skipped_holdouts)))
        )
    if missing_requested_holdouts:
        failed_checks.append(
            "Requested holdouts did not emit metrics: " + ", ".join(sorted(set(missing_requested_holdouts)))
        )
    if len(successful_holdouts) < int(min_successful_holdouts):
        failed_checks.append(
            f"Successful league holdouts {len(successful_holdouts)} < minimum required {int(min_successful_holdouts)}."
        )

    requested_backtests = [str(item).strip() for item in (requested_backtest_test_seasons or []) if str(item).strip()]
    skipped_runs = list(backtest.get("skipped_runs") or [])
    blocking_skips = []
    for item in skipped_runs:
        reasons = [str(reason) for reason in (item.get("reasons") or [])]
        if reasons == ["latest_dataset_season_excluded"]:
            continue
        blocking_skips.append(item)
    if blocking_skips:
        failed_checks.append(
            "Rolling backtest skipped requested seasons: "
            + ", ".join(str(item.get("test_season") or "unknown") for item in blocking_skips)
        )

    backtest_rows = _backtest_rows_frame(backtest_rows_path)
    completed_backtests = []
    if not backtest_rows.empty and "test_season" in backtest_rows.columns:
        completed_backtests = [str(value).strip() for value in backtest_rows["test_season"].tolist() if str(value).strip()]

    excluded_latest = _safe_text(backtest.get("excluded_latest_dataset_season"))
    expected_backtests = [season for season in requested_backtests if season and season != excluded_latest]
    missing_backtests = [season for season in expected_backtests if season not in completed_backtests]
    if requested_backtests and missing_backtests:
        failed_checks.append(
            "Rolling backtest did not complete the requested finished-season window: "
            + ", ".join(missing_backtests)
        )

    if backtest_rows.empty:
        if requested_backtests:
            failed_checks.append("Rolling backtest summary is missing for the valuation promotion review.")
    else:
        for _, row in backtest_rows.iterrows():
            season = str(row.get("test_season") or "unknown")
            r2 = _coerce_numeric(row.get("test_r2"))
            wmape = _coerce_numeric(row.get("test_wmape"))
            under5 = _coerce_numeric(row.get("test_under_5m_wmape"))
            lowmid = _coerce_numeric(row.get("test_lowmid_weighted_wmape"))
            segment = _coerce_numeric(row.get("test_segment_weighted_wmape"))
            if r2 is not None and r2 < float(min_test_r2):
                failed_checks.append(f"{season}: test_r2 {r2:.4f} < min_test_r2 {float(min_test_r2):.4f}")
            if wmape is not None and wmape > float(max_test_wmape):
                failed_checks.append(f"{season}: test_wmape {wmape:.4f} > max_test_wmape {float(max_test_wmape):.4f}")
            if under5 is not None and under5 > float(max_under5m_wmape):
                failed_checks.append(
                    f"{season}: test_under_5m_wmape {under5:.4f} > max_under5m_wmape {float(max_under5m_wmape):.4f}"
                )
            if lowmid is not None and lowmid > float(max_lowmid_weighted_wmape):
                failed_checks.append(
                    f"{season}: test_lowmid_weighted_wmape {lowmid:.4f} > max_lowmid_weighted_wmape {float(max_lowmid_weighted_wmape):.4f}"
                )
            if segment is not None and segment > float(max_segment_weighted_wmape):
                failed_checks.append(
                    f"{season}: test_segment_weighted_wmape {segment:.4f} > max_segment_weighted_wmape {float(max_segment_weighted_wmape):.4f}"
                )

    if excluded_latest:
        warnings.append(
            f"Latest dataset season {excluded_latest} was excluded from the valuation backtest window by design."
        )

    promotable = len(failed_checks) == 0
    return {
        "promotable": promotable,
        "mode": "soft_warn",
        "criteria": {
            "min_successful_holdouts": int(min_successful_holdouts),
            "min_test_r2": float(min_test_r2),
            "max_test_wmape": float(max_test_wmape),
            "max_under5m_wmape": float(max_under5m_wmape),
            "max_lowmid_weighted_wmape": float(max_lowmid_weighted_wmape),
            "max_segment_weighted_wmape": float(max_segment_weighted_wmape),
            "requested_backtest_test_seasons": requested_backtests,
            "requested_holdouts": requested_tokens,
        },
        "failed_checks": failed_checks,
        "warnings": warnings,
        "holdout_coverage": {
            "requested_count": len(requested_tokens),
            "successful_count": len(successful_holdouts),
            "successful_leagues": successful_holdouts,
            "skipped_holdouts": skipped_holdouts,
            "missing_requested_holdouts": missing_requested_holdouts,
        },
        "backtest_window": {
            "requested_test_seasons": requested_backtests,
            "completed_test_seasons": completed_backtests,
            "latest_dataset_season": _safe_text(backtest.get("latest_dataset_season")),
            "excluded_latest_dataset_season": excluded_latest,
            "skipped_runs": skipped_runs,
        },
    }


__all__ = [
    "INGESTION_HEALTH_CSV",
    "INGESTION_HEALTH_JSON",
    "MIN_SUCCESSFUL_HOLDOUTS",
    "build_ingestion_health_payload",
    "build_valuation_promotion_gate",
    "latest_timestamp",
    "load_ingestion_health_payload",
    "read_json_sidecar",
    "regenerate_ingestion_health_report",
    "write_ingestion_health_report",
    "write_json_sidecar",
]
