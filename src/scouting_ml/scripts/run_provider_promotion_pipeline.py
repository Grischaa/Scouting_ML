from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scouting_ml.core.runtime_config import PRODUCTION_PIPELINE_DEFAULTS
from scouting_ml.scripts.run_full_pipeline import run_full_pipeline

PROVIDER_PREFIXES = ("sb_", "avail_", "fixture_", "odds_")
PROVIDER_COVERAGE_META_SUFFIXES = (
    "provider_player_id",
    "player_name",
    "provider_team_id",
    "team_name",
    "league",
    "transfermarkt_id",
    "dob",
    "source_provider",
    "source_version",
    "retrieved_at",
    "snapshot_date",
    "coverage_note",
    "non_null_share",
    "has_data",
)
SEEDED_EXTERNAL_FILES = (
    "player_contracts.csv",
    "player_injuries.csv",
    "player_transfers.csv",
    "national_team_caps.csv",
    "club_context.csv",
    "league_context.csv",
    "uefa_country_coefficients.csv",
)


@dataclass(frozen=True)
class CandidatePaths:
    root_dir: Path
    external_dir: Path
    dataset_path: Path
    clean_dataset_path: Path
    predictions_path: Path
    val_predictions_path: Path
    metrics_path: Path
    quality_path: Path
    error_priors_path: Path
    backtest_dir: Path
    candidate_manifest_path: Path
    candidate_env_path: Path
    summary_path: Path
    full_pipeline_summary_path: Path
    initial_provider_build_path: Path
    linked_provider_build_path: Path
    effective_config_path: Path
    provider_audit_json_path: Path
    provider_audit_csv_path: Path
    bootstrap_summary_path: Path


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _season_slug(season: str) -> str:
    return str(season).replace("/", "-")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _artifact_meta(path: str | Path | None) -> dict[str, Any] | None:
    if path in (None, ""):
        return None
    resolved = Path(path)
    exists = resolved.exists()
    return {
        "path": str(resolved.resolve()),
        "exists": exists,
        "size_bytes": int(resolved.stat().st_size) if exists else None,
    }


def _require_artifact(path: str | Path | None, label: str) -> dict[str, Any] | None:
    meta = _artifact_meta(path)
    if meta is None:
        return None
    if not meta["exists"]:
        raise FileNotFoundError(f"Expected {label} artifact was not created: {path}")
    return meta


def bootstrap_provider_links(**kwargs) -> dict[str, Any]:
    from scouting_ml.scripts.bootstrap_provider_links import bootstrap_provider_links as _bootstrap_provider_links

    return _bootstrap_provider_links(**kwargs)


def build_provider_external_data(**kwargs) -> dict[str, Any]:
    from scouting_ml.scripts.build_provider_external_data import build_provider_external_data as _build_provider_external_data

    return _build_provider_external_data(**kwargs)


def build_provider_link_audit(**kwargs) -> dict[str, Any]:
    from scouting_ml.scripts.build_provider_link_audit import build_provider_link_audit as _build_provider_link_audit

    return _build_provider_link_audit(**kwargs)


def build_lock_bundle(**kwargs) -> None:
    from scouting_ml.scripts.lock_market_value_artifacts import build_lock_bundle as _build_lock_bundle

    _build_lock_bundle(**kwargs)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported clean dataset format for provider coverage: {path}")


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _series_has_value(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").notna()
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})
        .ne("")
    )


def _row_non_null_share(frame: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if not cols:
        return pd.Series(0.0, index=frame.index, dtype="float64")
    presence = pd.DataFrame({_col: _series_has_value(frame[_col]) for _col in cols}, index=frame.index)
    return presence.mean(axis=1).astype(float)


def _series_truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "", "none": "", "<na>": ""})
    )
    return text.isin({"1", "true", "yes", "y"})


def _provider_signal_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    cols = [col for col in frame.columns if col.startswith(prefix)]
    exclude = {f"{prefix}{suffix}" for suffix in PROVIDER_COVERAGE_META_SUFFIXES}
    return [col for col in cols if col not in exclude]


def _provider_presence_mask(frame: pd.DataFrame, prefix: str) -> tuple[pd.Series, list[str]]:
    flag_col = f"{prefix}has_data"
    signal_cols = _provider_signal_columns(frame, prefix)
    if flag_col in frame.columns:
        return _series_truthy(frame[flag_col]).astype(bool), signal_cols
    if signal_cols:
        return (_row_non_null_share(frame, signal_cols) > 0.0), signal_cols
    return pd.Series(False, index=frame.index, dtype=bool), signal_cols


def _weighted_by_n(rows: Sequence[tuple[int, float]]) -> float:
    total_n = 0
    weighted_sum = 0.0
    for n_samples, value in rows:
        n_int = int(n_samples)
        value_float = float(value)
        if n_int <= 0 or not np.isfinite(value_float):
            continue
        total_n += n_int
        weighted_sum += n_int * value_float
    if total_n <= 0:
        return float("nan")
    return float(weighted_sum / total_n)


def _extract_metrics_snapshot(payload: dict[str, Any], split: str) -> dict[str, float]:
    overall = payload.get("overall", {}).get(split, {})
    segment_rows = payload.get("segments", {}).get(split, [])
    segment_map = {
        str(row.get("segment")): row
        for row in segment_rows
        if isinstance(row, dict)
    }
    under = segment_map.get("under_5m", {})
    mid = segment_map.get("5m_to_20m", {})
    over = segment_map.get("over_20m", {})
    lowmid_wmape = _weighted_by_n(
        [
            (int(under.get("n_samples", 0) or 0), float(under.get("wmape", float("nan")))),
            (int(mid.get("n_samples", 0) or 0), float(mid.get("wmape", float("nan")))),
        ]
    )
    segment_wmape = _weighted_by_n(
        [
            (int(under.get("n_samples", 0) or 0), float(under.get("wmape", float("nan")))),
            (int(mid.get("n_samples", 0) or 0), float(mid.get("wmape", float("nan")))),
            (int(over.get("n_samples", 0) or 0), float(over.get("wmape", float("nan")))),
        ]
    )
    return {
        "n_samples": int(overall.get("n_samples", 0) or 0),
        "r2": float(overall.get("r2", float("nan"))),
        "mae_eur": float(overall.get("mae_eur", float("nan"))),
        "mape": float(overall.get("mape", float("nan"))),
        "wmape": float(overall.get("wmape", float("nan"))),
        "lowmid_weighted_wmape": lowmid_wmape,
        "segment_weighted_wmape": segment_wmape,
    }


def _extract_backtest_snapshot(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    return {
        "runs": int(payload.get("runs", 0) or 0),
        "mean_test_r2": float(payload.get("mean_test_r2", float("nan"))),
        "mean_test_wmape": float(payload.get("mean_test_wmape", float("nan"))),
        "mean_test_lowmid_weighted_wmape": float(payload.get("mean_test_lowmid_weighted_wmape", float("nan"))),
        "mean_test_segment_weighted_wmape": float(payload.get("mean_test_segment_weighted_wmape", float("nan"))),
    }


def _metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, cand_value in candidate.items():
        base_value = baseline.get(key, float("nan"))
        out[key] = float(cand_value) - float(base_value)
    return out


def _seed_external_dir(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for filename in SEEDED_EXTERNAL_FILES:
        src = source_dir / filename
        dst = target_dir / filename
        if not src.exists():
            missing.append(filename)
            continue
        shutil.copy2(src, dst)
        copied.append(filename)
    return {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "copied_files": copied,
        "missing_files": missing,
    }


def _build_candidate_paths(*, out_dir: str, candidate_tag: str, test_season: str) -> CandidatePaths:
    root_dir = Path(out_dir) / candidate_tag
    season_slug = _season_slug(test_season)
    predictions_path = root_dir / f"{candidate_tag}_predictions_{season_slug}.csv"
    return CandidatePaths(
        root_dir=root_dir,
        external_dir=root_dir / "external",
        dataset_path=root_dir / f"{candidate_tag}_players.parquet",
        clean_dataset_path=root_dir / f"{candidate_tag}_players_clean.parquet",
        predictions_path=predictions_path,
        val_predictions_path=predictions_path.with_name(f"{predictions_path.stem}_val{predictions_path.suffix}"),
        metrics_path=predictions_path.with_suffix(".metrics.json"),
        quality_path=predictions_path.with_suffix(".quality.json"),
        error_priors_path=predictions_path.with_name(f"{predictions_path.stem}.error_priors.csv"),
        backtest_dir=root_dir / "backtests",
        candidate_manifest_path=root_dir / "candidate_model_manifest.json",
        candidate_env_path=root_dir / "candidate_model_artifacts.env",
        summary_path=root_dir / "provider_promotion_summary.json",
        full_pipeline_summary_path=root_dir / "full_pipeline_summary.json",
        initial_provider_build_path=root_dir / "provider_build_initial.json",
        linked_provider_build_path=root_dir / "provider_build_linked.json",
        effective_config_path=root_dir / "provider_config.effective.json",
        provider_audit_json_path=root_dir / "provider_link_audit.json",
        provider_audit_csv_path=root_dir / "provider_link_audit.csv",
        bootstrap_summary_path=root_dir / "provider_link_bootstrap.json",
    )


def _prepare_effective_provider_config(
    *,
    config_path: Path,
    stage_external_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    payload = _read_json(config_path)
    warnings: list[str] = []
    active_sections: list[str] = []

    payload["player_links"] = str(stage_external_dir / "player_provider_links.csv")
    payload["club_links"] = str(stage_external_dir / "club_provider_links.csv")

    statsbomb_cfg = payload.get("statsbomb") or {}
    if statsbomb_cfg.get("open_data_root"):
        open_data_root = Path(str(statsbomb_cfg.get("open_data_root")))
        if open_data_root.exists():
            statsbomb_cfg["output"] = str(stage_external_dir / "statsbomb_player_season_features.csv")
            payload["statsbomb"] = statsbomb_cfg
            active_sections.append("statsbomb")
        else:
            warnings.append(f"Skipped statsbomb: open-data root missing -> {open_data_root}")
            payload.pop("statsbomb", None)
    elif "statsbomb" in payload:
        warnings.append("Skipped statsbomb: config missing open_data_root")
        payload.pop("statsbomb", None)

    for section, default_name in (
        ("fixture_context", "fixture_context.csv"),
        ("player_availability", "player_availability.csv"),
        ("market_context", "market_context.csv"),
    ):
        cfg = payload.get(section) or {}
        if not cfg:
            payload.pop(section, None)
            continue
        input_json = _as_str_list(cfg.get("input_json"))
        existing_json = [item for item in input_json if Path(item).exists()]
        missing_json = [item for item in input_json if not Path(item).exists()]
        api_urls = _as_str_list(cfg.get("api_url"))
        if missing_json:
            warnings.append(
                f"Section {section}: missing snapshot files -> {', '.join(missing_json)}"
            )
        cfg["input_json"] = existing_json
        cfg["output"] = str(stage_external_dir / default_name)
        if existing_json or api_urls:
            if api_urls:
                cfg["api_url"] = api_urls
            payload[section] = cfg
            active_sections.append(section)
        else:
            payload.pop(section, None)
            warnings.append(f"Skipped {section}: no available snapshots or api_url values")

    if not active_sections:
        raise ValueError("No provider sections are runnable after resolving config inputs.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "effective_config_path": str(output_path),
        "active_sections": active_sections,
        "warnings": warnings,
    }


def _provider_coverage_from_clean_dataset(
    *,
    clean_dataset_path: Path,
) -> dict[str, Any]:
    if not clean_dataset_path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {clean_dataset_path}")
    frame = _read_table(clean_dataset_path)
    if frame.empty or "season" not in frame.columns:
        return {
            "dataset_path": str(clean_dataset_path),
            "rows": int(len(frame)),
            "by_season": {},
        }

    out: dict[str, Any] = {
        "dataset_path": str(clean_dataset_path),
        "rows": int(len(frame)),
        "by_season": {},
    }
    for season, season_frame in frame.groupby("season", dropna=False):
        season_key = str(season)
        season_payload: dict[str, Any] = {
            "rows": int(len(season_frame)),
        }
        provider_presence: list[pd.Series] = []
        for prefix in PROVIDER_PREFIXES:
            presence_mask, signal_cols = _provider_presence_mask(season_frame, prefix)
            share = float(presence_mask.mean()) if len(presence_mask) else 0.0
            season_payload[prefix] = {
                "cols": int(len(signal_cols)),
                "row_coverage_share": share,
            }
            provider_presence.append(presence_mask.rename(prefix))
        season_payload["any_provider_coverage_share"] = float(
            pd.concat(provider_presence, axis=1).any(axis=1).mean()
        ) if provider_presence else 0.0
        out["by_season"][season_key] = season_payload
    return out


def _evaluate_promotion_gate(
    *,
    baseline_test: dict[str, float],
    candidate_test: dict[str, float],
    baseline_backtest: dict[str, float] | None,
    candidate_backtest: dict[str, float] | None,
    provider_coverage: dict[str, Any],
    test_season: str,
    min_test_provider_coverage: float,
    max_test_wmape_delta: float,
    min_test_r2_delta: float,
    max_test_lowmid_wmape_delta: float,
    max_backtest_test_wmape_delta: float,
    min_backtest_test_r2_delta: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    coverage_row = provider_coverage.get("by_season", {}).get(str(test_season), {})
    observed_test_coverage = float(coverage_row.get("any_provider_coverage_share", 0.0) or 0.0)
    test_wmape_delta = float(candidate_test.get("wmape", float("nan"))) - float(baseline_test.get("wmape", float("nan")))
    test_r2_delta = float(candidate_test.get("r2", float("nan"))) - float(baseline_test.get("r2", float("nan")))
    test_lowmid_delta = float(candidate_test.get("lowmid_weighted_wmape", float("nan"))) - float(
        baseline_test.get("lowmid_weighted_wmape", float("nan"))
    )

    if observed_test_coverage < float(min_test_provider_coverage):
        reasons.append(
            f"test provider coverage {observed_test_coverage:.4f} < min_test_provider_coverage {float(min_test_provider_coverage):.4f}"
        )
    if np.isfinite(test_wmape_delta) and test_wmape_delta > float(max_test_wmape_delta):
        reasons.append(
            f"test WMAPE delta {test_wmape_delta:+.4f} > max_test_wmape_delta {float(max_test_wmape_delta):+.4f}"
        )
    if np.isfinite(test_r2_delta) and test_r2_delta < float(min_test_r2_delta):
        reasons.append(
            f"test R2 delta {test_r2_delta:+.4f} < min_test_r2_delta {float(min_test_r2_delta):+.4f}"
        )
    if np.isfinite(test_lowmid_delta) and test_lowmid_delta > float(max_test_lowmid_wmape_delta):
        reasons.append(
            "test lowmid weighted WMAPE delta "
            f"{test_lowmid_delta:+.4f} > max_test_lowmid_wmape_delta {float(max_test_lowmid_wmape_delta):+.4f}"
        )

    backtest_delta: dict[str, float] | None = None
    if baseline_backtest and candidate_backtest:
        backtest_delta = _metric_delta(candidate_backtest, baseline_backtest)
        mean_backtest_wmape_delta = float(backtest_delta.get("mean_test_wmape", float("nan")))
        mean_backtest_r2_delta = float(backtest_delta.get("mean_test_r2", float("nan")))
        if np.isfinite(mean_backtest_wmape_delta) and mean_backtest_wmape_delta > float(max_backtest_test_wmape_delta):
            reasons.append(
                "backtest mean test WMAPE delta "
                f"{mean_backtest_wmape_delta:+.4f} > max_backtest_test_wmape_delta {float(max_backtest_test_wmape_delta):+.4f}"
            )
        if np.isfinite(mean_backtest_r2_delta) and mean_backtest_r2_delta < float(min_backtest_test_r2_delta):
            reasons.append(
                f"backtest mean test R2 delta {mean_backtest_r2_delta:+.4f} < min_backtest_test_r2_delta {float(min_backtest_test_r2_delta):+.4f}"
            )

    return {
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": {
            "min_test_provider_coverage": float(min_test_provider_coverage),
            "max_test_wmape_delta": float(max_test_wmape_delta),
            "min_test_r2_delta": float(min_test_r2_delta),
            "max_test_lowmid_wmape_delta": float(max_test_lowmid_wmape_delta),
            "max_backtest_test_wmape_delta": float(max_backtest_test_wmape_delta),
            "min_backtest_test_r2_delta": float(min_backtest_test_r2_delta),
        },
        "observed": {
            "test_provider_coverage": observed_test_coverage,
            "test_wmape_delta": test_wmape_delta,
            "test_r2_delta": test_r2_delta,
            "test_lowmid_weighted_wmape_delta": test_lowmid_delta,
            "backtest_delta": backtest_delta,
        },
    }


def run_provider_promotion_pipeline(
    *,
    provider_config_json: str,
    out_dir: str,
    candidate_tag: str,
    players_source: str,
    data_dir: str,
    external_dir: str,
    baseline_metrics_path: str,
    baseline_backtest_path: str,
    val_season: str,
    test_season: str,
    start_season: str,
    end_season: str,
    min_minutes: float,
    trials: int,
    optimize_metric: str,
    band_min_samples: int,
    band_blend_alpha: float,
    with_backtest: bool,
    backtest_test_seasons: Sequence[str],
    review_confidence_threshold: float,
    skip_injuries: bool,
    skip_contracts: bool,
    skip_transfers: bool,
    skip_national: bool,
    skip_context: bool,
    min_test_provider_coverage: float,
    max_test_wmape_delta: float,
    min_test_r2_delta: float,
    max_test_lowmid_wmape_delta: float,
    max_backtest_test_wmape_delta: float,
    min_backtest_test_r2_delta: float,
    promote_on_pass: bool,
    promotion_manifest_out: str,
    promotion_env_out: str,
    promotion_label: str,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    paths = _build_candidate_paths(out_dir=out_dir, candidate_tag=candidate_tag, test_season=test_season)
    paths.root_dir.mkdir(parents=True, exist_ok=True)

    seed_summary = _seed_external_dir(Path(external_dir), paths.external_dir)
    effective_cfg_summary = _prepare_effective_provider_config(
        config_path=Path(provider_config_json),
        stage_external_dir=paths.external_dir,
        output_path=paths.effective_config_path,
    )

    initial_provider_build = build_provider_external_data(
        config_path=paths.effective_config_path,
        external_dir=paths.external_dir,
    )
    _write_json(paths.initial_provider_build_path, initial_provider_build)

    bootstrap_summary = bootstrap_provider_links(
        players_source=players_source,
        external_dir=str(paths.external_dir),
        player_links_out=str(paths.external_dir / "player_provider_links.csv"),
        club_links_out=str(paths.external_dir / "club_provider_links.csv"),
        review_confidence_threshold=review_confidence_threshold,
    )
    _write_json(paths.bootstrap_summary_path, bootstrap_summary)

    linked_provider_build = build_provider_external_data(
        config_path=paths.effective_config_path,
        external_dir=paths.external_dir,
    )
    _write_json(paths.linked_provider_build_path, linked_provider_build)

    provider_audit = build_provider_link_audit(
        players_source=players_source,
        external_dir=str(paths.external_dir),
        player_links=str(paths.external_dir / "player_provider_links.csv"),
        club_links=str(paths.external_dir / "club_provider_links.csv"),
        out_json=str(paths.provider_audit_json_path),
        out_csv=str(paths.provider_audit_csv_path),
    )

    full_pipeline_summary = run_full_pipeline(
        players_source=players_source,
        data_dir=data_dir,
        external_dir=str(paths.external_dir),
        dataset_output=str(paths.dataset_path),
        clean_output=str(paths.clean_dataset_path),
        predictions_output=str(paths.predictions_path),
        val_season=val_season,
        test_season=test_season,
        start_season=start_season,
        end_season=end_season,
        min_minutes=min_minutes,
        trials=trials,
        recency_half_life=2.0,
        under_5m_weight=1.0,
        mid_5m_to_20m_weight=1.0,
        over_20m_weight=1.0,
        exclude_prefixes=[],
        exclude_columns=[],
        optimize_metric=optimize_metric,
        interval_q=0.8,
        two_stage_band_model=True,
        band_min_samples=band_min_samples,
        band_blend_alpha=band_blend_alpha,
        strict_leakage_guard=True,
        strict_quality_gate=False,
        league_holdouts=[],
        drop_incomplete_league_seasons=True,
        min_league_season_rows=40,
        min_league_season_completeness=0.55,
        residual_calibration_min_samples=30,
        mape_min_denom_eur=1_000_000.0,
        max_players=None,
        sleep_seconds=2.5,
        transfer_dynamic_fallback=False,
        transfer_dynamic_fallback_attempts=2,
        contracts_all_seasons=False,
        national_all_seasons=False,
        fetch_missing_profiles=False,
        fetch_national_page=False,
        with_ablation=False,
        with_backtest=with_backtest,
        ablation_configs=[],
        ablation_out_dir=str(paths.root_dir / "ablation"),
        backtest_out_dir=str(paths.backtest_dir),
        backtest_min_train_seasons=2,
        backtest_test_seasons=list(backtest_test_seasons),
        backtest_enforce_quality_gate=False,
        backtest_min_test_r2=PRODUCTION_PIPELINE_DEFAULTS.backtest_min_test_r2,
        backtest_max_test_mape=None,
        backtest_max_test_wmape=PRODUCTION_PIPELINE_DEFAULTS.backtest_max_test_wmape,
        backtest_max_under5m_wmape=PRODUCTION_PIPELINE_DEFAULTS.backtest_max_under5m_wmape,
        backtest_max_lowmid_weighted_wmape=PRODUCTION_PIPELINE_DEFAULTS.backtest_max_lowmid_weighted_wmape,
        backtest_max_segment_weighted_wmape=PRODUCTION_PIPELINE_DEFAULTS.backtest_max_segment_weighted_wmape,
        backtest_min_test_samples=PRODUCTION_PIPELINE_DEFAULTS.backtest_min_test_samples,
        backtest_min_test_under5m_samples=PRODUCTION_PIPELINE_DEFAULTS.backtest_min_test_under5m_samples,
        backtest_min_test_over20m_samples=PRODUCTION_PIPELINE_DEFAULTS.backtest_min_test_over20m_samples,
        backtest_skip_incomplete_test_seasons=PRODUCTION_PIPELINE_DEFAULTS.backtest_skip_incomplete_test_seasons,
        backtest_drop_incomplete_league_seasons=PRODUCTION_PIPELINE_DEFAULTS.drop_incomplete_league_seasons,
        backtest_min_league_season_rows=PRODUCTION_PIPELINE_DEFAULTS.min_league_season_rows,
        backtest_min_league_season_completeness=PRODUCTION_PIPELINE_DEFAULTS.min_league_season_completeness,
        backtest_residual_calibration_min_samples=PRODUCTION_PIPELINE_DEFAULTS.residual_calibration_min_samples,
        backtest_mape_min_denom_eur=PRODUCTION_PIPELINE_DEFAULTS.mape_min_denom_eur,
        with_future_targets=False,
        future_targets_output=str(paths.root_dir / "future_targets.parquet"),
        future_target_min_next_minutes=450.0,
        future_target_drop_na=False,
        skip_injuries=skip_injuries,
        skip_contracts=skip_contracts,
        skip_transfers=skip_transfers,
        skip_national=skip_national,
        skip_context=skip_context,
        provider_config_json=None,
        provider_audit_json=None,
        provider_audit_csv=None,
        skip_dataset_build=False,
        skip_clean=False,
        skip_train=False,
        lock_artifacts=False,
        lock_manifest_out=str(paths.candidate_manifest_path),
        lock_env_out=str(paths.candidate_env_path),
        lock_label=promotion_label,
        lock_strict_artifacts=True,
        optuna_study_namespace=f"provider_promotion_{candidate_tag}",
        optuna_load_if_exists=False,
        summary_json=str(paths.full_pipeline_summary_path),
    )

    build_lock_bundle(
        test_predictions=paths.predictions_path,
        val_predictions=paths.val_predictions_path,
        metrics_path=paths.metrics_path,
        manifest_out=paths.candidate_manifest_path,
        env_out=paths.candidate_env_path,
        strict_artifacts=True,
        label=f"{promotion_label}_candidate",
    )

    baseline_metrics = _read_json(Path(baseline_metrics_path))
    candidate_metrics = _read_json(paths.metrics_path)
    baseline_snapshot = {
        "val": _extract_metrics_snapshot(baseline_metrics, "val"),
        "test": _extract_metrics_snapshot(baseline_metrics, "test"),
    }
    candidate_snapshot = {
        "val": _extract_metrics_snapshot(candidate_metrics, "val"),
        "test": _extract_metrics_snapshot(candidate_metrics, "test"),
    }
    comparison = {
        split: {
            "baseline": baseline_snapshot[split],
            "candidate": candidate_snapshot[split],
            "delta": _metric_delta(candidate_snapshot[split], baseline_snapshot[split]),
        }
        for split in ("val", "test")
    }

    baseline_backtest = _extract_backtest_snapshot(Path(baseline_backtest_path))
    candidate_backtest = _extract_backtest_snapshot(paths.backtest_dir / "rolling_backtest_summary.json")
    provider_coverage = _provider_coverage_from_clean_dataset(clean_dataset_path=paths.clean_dataset_path)

    promotion = _evaluate_promotion_gate(
        baseline_test=baseline_snapshot["test"],
        candidate_test=candidate_snapshot["test"],
        baseline_backtest=baseline_backtest,
        candidate_backtest=candidate_backtest,
        provider_coverage=provider_coverage,
        test_season=test_season,
        min_test_provider_coverage=min_test_provider_coverage,
        max_test_wmape_delta=max_test_wmape_delta,
        min_test_r2_delta=min_test_r2_delta,
        max_test_lowmid_wmape_delta=max_test_lowmid_wmape_delta,
        max_backtest_test_wmape_delta=max_backtest_test_wmape_delta,
        min_backtest_test_r2_delta=min_backtest_test_r2_delta,
    )

    promoted = False
    if promote_on_pass and promotion["passed"]:
        build_lock_bundle(
            test_predictions=paths.predictions_path,
            val_predictions=paths.val_predictions_path,
            metrics_path=paths.metrics_path,
            manifest_out=Path(promotion_manifest_out),
            env_out=Path(promotion_env_out),
            strict_artifacts=True,
            label=promotion_label,
        )
        promoted = True

    finished_at = datetime.now(timezone.utc)
    summary = {
        "generated_at_utc": finished_at.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "inputs": {
            "provider_config_json": str(Path(provider_config_json).resolve()),
            "out_dir": str(Path(out_dir).resolve()),
            "candidate_tag": candidate_tag,
            "players_source": str(Path(players_source).resolve()),
            "data_dir": str(Path(data_dir).resolve()),
            "external_dir": str(Path(external_dir).resolve()),
            "val_season": val_season,
            "test_season": test_season,
        },
        "flags": {
            "with_backtest": bool(with_backtest),
            "skip_injuries": bool(skip_injuries),
            "skip_contracts": bool(skip_contracts),
            "skip_transfers": bool(skip_transfers),
            "skip_national": bool(skip_national),
            "skip_context": bool(skip_context),
            "promote_on_pass": bool(promote_on_pass),
        },
        "candidate_tag": candidate_tag,
        "effective_provider_config": effective_cfg_summary,
        "seed_external_dir": seed_summary,
        "provider_build_initial": initial_provider_build,
        "provider_link_bootstrap": bootstrap_summary,
        "provider_build_linked": linked_provider_build,
        "provider_audit": {
            "json_path": str(paths.provider_audit_json_path),
            "csv_path": str(paths.provider_audit_csv_path),
            "summary": provider_audit,
        },
        "artifacts": {
            "dataset": _require_artifact(paths.dataset_path, "dataset"),
            "clean_dataset": _require_artifact(paths.clean_dataset_path, "clean dataset"),
            "test_predictions": _require_artifact(paths.predictions_path, "test predictions"),
            "val_predictions": _require_artifact(paths.val_predictions_path, "validation predictions"),
            "metrics": _require_artifact(paths.metrics_path, "metrics"),
            "quality": _require_artifact(paths.quality_path, "quality"),
            "error_priors": _require_artifact(paths.error_priors_path, "error priors"),
            "candidate_manifest": _require_artifact(paths.candidate_manifest_path, "candidate manifest"),
            "candidate_env": _require_artifact(paths.candidate_env_path, "candidate env"),
            "full_pipeline_summary": _require_artifact(paths.full_pipeline_summary_path, "full pipeline summary"),
        },
        "snapshots": {
            "baseline_metrics": baseline_snapshot,
            "candidate_metrics": candidate_snapshot,
            "baseline_backtest": baseline_backtest,
            "candidate_backtest": candidate_backtest,
            "provider_audit": provider_audit,
            "full_pipeline": full_pipeline_summary,
        },
        "full_pipeline": full_pipeline_summary,
        "comparison": comparison,
        "baseline_backtest": baseline_backtest,
        "candidate_backtest": candidate_backtest,
        "provider_coverage": provider_coverage,
        "promotion": {
            **promotion,
            "promote_on_pass": bool(promote_on_pass),
            "promoted": promoted,
            "promotion_manifest_out": str(promotion_manifest_out),
            "promotion_env_out": str(promotion_env_out),
        },
        "status": "ok",
    }
    _write_json(paths.summary_path, summary)
    return summary


def main() -> None:
    defaults = PRODUCTION_PIPELINE_DEFAULTS
    parser = argparse.ArgumentParser(
        description=(
            "Build a provider-enriched candidate model end to end: provider tables, "
            "link bootstrap + audit, retrain/backtest, compare against champion, and "
            "optionally promote artifacts when gates pass."
        )
    )
    parser.add_argument("--provider-config-json", required=True)
    parser.add_argument("--out-dir", default="data/model/provider_promotion")
    parser.add_argument("--candidate-tag", default="provider_candidate")
    parser.add_argument("--players-source", default=defaults.players_source)
    parser.add_argument("--data-dir", default=defaults.data_dir)
    parser.add_argument("--external-dir", default=defaults.external_dir)
    parser.add_argument("--baseline-metrics-path", default=defaults.predictions_output.replace(".csv", ".metrics.json"))
    parser.add_argument("--baseline-backtest-path", default="data/model/backtests/rolling_backtest_summary.json")
    parser.add_argument("--val-season", default=defaults.val_season)
    parser.add_argument("--test-season", default=defaults.test_season)
    parser.add_argument("--start-season", default=defaults.start_season)
    parser.add_argument("--end-season", default=defaults.end_season)
    parser.add_argument("--min-minutes", type=float, default=defaults.min_minutes)
    parser.add_argument("--trials", type=int, default=defaults.trials)
    parser.add_argument("--optimize-metric", default=defaults.optimize_metric)
    parser.add_argument("--band-min-samples", type=int, default=defaults.band_min_samples)
    parser.add_argument("--band-blend-alpha", type=float, default=defaults.band_blend_alpha)
    parser.add_argument("--with-backtest", action=argparse.BooleanOptionalAction, default=defaults.with_backtest)
    parser.add_argument("--backtest-test-seasons", default=defaults.backtest_test_seasons)
    parser.add_argument("--review-confidence-threshold", type=float, default=0.80)
    parser.add_argument("--skip-injuries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-contracts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-transfers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-national", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-test-provider-coverage", type=float, default=0.05)
    parser.add_argument("--max-test-wmape-delta", type=float, default=0.0)
    parser.add_argument("--min-test-r2-delta", type=float, default=0.0)
    parser.add_argument("--max-test-lowmid-wmape-delta", type=float, default=0.0)
    parser.add_argument("--max-backtest-test-wmape-delta", type=float, default=0.0)
    parser.add_argument("--min-backtest-test-r2-delta", type=float, default=0.0)
    parser.add_argument("--promote-on-pass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--promotion-manifest-out", default=defaults.lock_manifest_out)
    parser.add_argument("--promotion-env-out", default=defaults.lock_env_out)
    parser.add_argument("--promotion-label", default="provider_candidate_bundle")
    args = parser.parse_args()

    payload = run_provider_promotion_pipeline(
        provider_config_json=args.provider_config_json,
        out_dir=args.out_dir,
        candidate_tag=args.candidate_tag,
        players_source=args.players_source,
        data_dir=args.data_dir,
        external_dir=args.external_dir,
        baseline_metrics_path=args.baseline_metrics_path,
        baseline_backtest_path=args.baseline_backtest_path,
        val_season=args.val_season,
        test_season=args.test_season,
        start_season=args.start_season,
        end_season=args.end_season,
        min_minutes=args.min_minutes,
        trials=args.trials,
        optimize_metric=args.optimize_metric,
        band_min_samples=args.band_min_samples,
        band_blend_alpha=args.band_blend_alpha,
        with_backtest=args.with_backtest,
        backtest_test_seasons=_parse_csv_tokens(args.backtest_test_seasons),
        review_confidence_threshold=args.review_confidence_threshold,
        skip_injuries=args.skip_injuries,
        skip_contracts=args.skip_contracts,
        skip_transfers=args.skip_transfers,
        skip_national=args.skip_national,
        skip_context=args.skip_context,
        min_test_provider_coverage=args.min_test_provider_coverage,
        max_test_wmape_delta=args.max_test_wmape_delta,
        min_test_r2_delta=args.min_test_r2_delta,
        max_test_lowmid_wmape_delta=args.max_test_lowmid_wmape_delta,
        max_backtest_test_wmape_delta=args.max_backtest_test_wmape_delta,
        min_backtest_test_r2_delta=args.min_backtest_test_r2_delta,
        promote_on_pass=args.promote_on_pass,
        promotion_manifest_out=args.promotion_manifest_out,
        promotion_env_out=args.promotion_env_out,
        promotion_label=args.promotion_label,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
