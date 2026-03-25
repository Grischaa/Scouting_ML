from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scouting_ml.core.runtime_config import PRODUCTION_PIPELINE_DEFAULTS
from scouting_ml.reporting.operator_health import regenerate_ingestion_health_report
from scouting_ml.scripts.run_full_pipeline import run_full_pipeline
from scouting_ml.scripts.run_weekly_scout_ops import run_weekly_scout_ops


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _artifact_bundle_from_predictions(predictions_output: str) -> dict[str, str]:
    test_csv = Path(predictions_output)
    val_csv = test_csv.with_name(f"{test_csv.stem}_val{test_csv.suffix or '.csv'}")
    metrics_json = test_csv.with_suffix(".metrics.json")
    quality_json = test_csv.with_suffix(".quality.json")
    return {
        "test_predictions": str(test_csv),
        "val_predictions": str(val_csv),
        "metrics_json": str(metrics_json),
        "quality_json": str(quality_json),
    }


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
        raise FileNotFoundError(f"[prod] required artifact missing after training: {label} -> {path}")
    return meta


def run_production_pipeline(
    *,
    players_source: str,
    data_dir: str,
    external_dir: str,
    dataset_output: str,
    clean_output: str,
    predictions_output: str,
    val_season: str,
    test_season: str,
    start_season: str,
    end_season: str,
    min_minutes: float,
    trials: int,
    optimize_metric: str,
    league_holdouts: list[str],
    band_min_samples: int,
    band_blend_alpha: float,
    mape_min_denom_eur: float,
    with_backtest: bool,
    backtest_test_seasons: list[str],
    backtest_enforce_quality_gate: bool,
    backtest_min_test_r2: float,
    backtest_max_test_wmape: float,
    backtest_max_under5m_wmape: float,
    backtest_max_lowmid_weighted_wmape: float,
    backtest_max_segment_weighted_wmape: float,
    backtest_min_test_samples: int,
    backtest_min_test_under5m_samples: int,
    backtest_min_test_over20m_samples: int,
    backtest_exclude_latest_season: bool,
    backtest_skip_incomplete_test_seasons: bool,
    drop_incomplete_league_seasons: bool,
    min_league_season_rows: int,
    min_league_season_completeness: float,
    residual_calibration_min_samples: int,
    provider_config_json: str | None,
    provider_audit_json: str | None,
    provider_audit_csv: str | None,
    skip_injuries: bool,
    skip_contracts: bool,
    skip_transfers: bool,
    skip_national: bool,
    skip_context: bool,
    skip_dataset_build: bool,
    skip_clean: bool,
    lock_manifest_out: str,
    lock_env_out: str,
    lock_label: str,
    run_weekly_ops: bool,
    weekly_split: str,
    weekly_reports_out_dir: str,
    weekly_non_big5_only: bool,
    weekly_max_age: float | None,
    weekly_min_minutes: float,
    weekly_watchlist_tag: str,
    production_summary_out: str,
    full_pipeline_summary_out: str | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    print("\n======================================================================")
    print("[prod] Start production pipeline")
    print("======================================================================")

    pipeline_summary_path = (
        str(Path(full_pipeline_summary_out))
        if full_pipeline_summary_out
        else str(Path(production_summary_out).with_name("full_pipeline_summary.json"))
    )
    full_pipeline_summary = run_full_pipeline(
        players_source=players_source,
        data_dir=data_dir,
        external_dir=external_dir,
        dataset_output=dataset_output,
        clean_output=clean_output,
        predictions_output=predictions_output,
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
        league_holdouts=league_holdouts,
        drop_incomplete_league_seasons=drop_incomplete_league_seasons,
        min_league_season_rows=min_league_season_rows,
        min_league_season_completeness=min_league_season_completeness,
        residual_calibration_min_samples=residual_calibration_min_samples,
        mape_min_denom_eur=mape_min_denom_eur,
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
        ablation_out_dir="data/model/ablation",
        backtest_out_dir="data/model/backtests",
        backtest_min_train_seasons=2,
        backtest_test_seasons=backtest_test_seasons,
        backtest_enforce_quality_gate=backtest_enforce_quality_gate,
        backtest_min_test_r2=backtest_min_test_r2,
        backtest_max_test_mape=None,
        backtest_max_test_wmape=backtest_max_test_wmape,
        backtest_max_under5m_wmape=backtest_max_under5m_wmape,
        backtest_max_lowmid_weighted_wmape=backtest_max_lowmid_weighted_wmape,
        backtest_max_segment_weighted_wmape=backtest_max_segment_weighted_wmape,
        backtest_min_test_samples=backtest_min_test_samples,
        backtest_min_test_under5m_samples=backtest_min_test_under5m_samples,
        backtest_min_test_over20m_samples=backtest_min_test_over20m_samples,
        backtest_exclude_latest_season=backtest_exclude_latest_season,
        backtest_skip_incomplete_test_seasons=backtest_skip_incomplete_test_seasons,
        backtest_drop_incomplete_league_seasons=drop_incomplete_league_seasons,
        backtest_min_league_season_rows=min_league_season_rows,
        backtest_min_league_season_completeness=min_league_season_completeness,
        backtest_residual_calibration_min_samples=residual_calibration_min_samples,
        backtest_mape_min_denom_eur=mape_min_denom_eur,
        with_future_targets=False,
        future_targets_output="data/model/big5_players_future_targets.parquet",
        future_target_min_next_minutes=450.0,
        future_target_drop_na=False,
        skip_injuries=skip_injuries,
        skip_contracts=skip_contracts,
        skip_transfers=skip_transfers,
        skip_national=skip_national,
        skip_context=skip_context,
        provider_config_json=provider_config_json,
        provider_audit_json=provider_audit_json,
        provider_audit_csv=provider_audit_csv,
        skip_dataset_build=skip_dataset_build,
        skip_clean=skip_clean,
        skip_train=False,
        lock_artifacts=True,
        lock_manifest_out=lock_manifest_out,
        lock_env_out=lock_env_out,
        lock_label=lock_label,
        lock_strict_artifacts=True,
        summary_json=pipeline_summary_path,
    )

    artifact_paths = _artifact_bundle_from_predictions(predictions_output)
    artifacts = {name: _require_artifact(path_str, name) for name, path_str in artifact_paths.items()}
    artifacts["model_manifest"] = _require_artifact(lock_manifest_out, "model_manifest")
    artifacts["artifact_env"] = _require_artifact(lock_env_out, "artifact_env")
    artifacts["provider_audit_json"] = _artifact_meta(provider_audit_json)
    artifacts["provider_audit_csv"] = _artifact_meta(provider_audit_csv)
    artifacts["full_pipeline_summary"] = _require_artifact(pipeline_summary_path, "full_pipeline_summary")
    ingestion_health_payload = regenerate_ingestion_health_report(clean_dataset_path=Path(clean_output))
    artifacts["ingestion_health_csv"] = _require_artifact(
        ingestion_health_payload["_meta"]["csv_path"], "ingestion health csv"
    )
    artifacts["ingestion_health_json"] = _require_artifact(
        ingestion_health_payload["_meta"]["json_path"], "ingestion health json"
    )

    weekly_payload: dict[str, Any] | None = None
    if run_weekly_ops:
        print("\n======================================================================")
        print("[prod] Run weekly scout ops")
        print("======================================================================")
        weekly_payload = run_weekly_scout_ops(
            predictions=predictions_output,
            split=weekly_split,
            reports_out_dir=weekly_reports_out_dir,
            k_values=[10, 25, 50],
            score_col=None,
            label_col="interval_contains_truth",
            min_minutes=weekly_min_minutes,
            max_age=weekly_max_age,
            non_big5_only=weekly_non_big5_only,
            cohort_min_labeled=40,
            workflow_write_watchlist=True,
            workflow_watchlist_tag=weekly_watchlist_tag,
        )
    else:
        print("[prod] skip weekly ops")

    finished_at = datetime.now(timezone.utc)
    summary = {
        "generated_at_utc": finished_at.isoformat(),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "inputs": {
            "players_source": str(Path(players_source).resolve()),
            "data_dir": str(Path(data_dir).resolve()),
            "external_dir": str(Path(external_dir).resolve()),
            "val_season": val_season,
            "test_season": test_season,
        },
        "flags": {
            "with_backtest": bool(with_backtest),
            "run_weekly_ops": bool(run_weekly_ops),
            "skip_injuries": bool(skip_injuries),
            "skip_contracts": bool(skip_contracts),
            "skip_transfers": bool(skip_transfers),
            "skip_national": bool(skip_national),
            "skip_context": bool(skip_context),
            "skip_dataset_build": bool(skip_dataset_build),
            "skip_clean": bool(skip_clean),
            "backtest_exclude_latest_season": bool(backtest_exclude_latest_season),
            "league_holdouts": list(league_holdouts or []),
        },
        "config": {
            "val_season": val_season,
            "test_season": test_season,
            "trials": int(trials),
            "optimize_metric": optimize_metric,
            "band_min_samples": int(band_min_samples),
            "band_blend_alpha": float(band_blend_alpha),
            "with_backtest": bool(with_backtest),
            "backtest_test_seasons": list(backtest_test_seasons),
            "backtest_exclude_latest_season": bool(backtest_exclude_latest_season),
            "provider_config_json": provider_config_json or "",
            "run_weekly_ops": bool(run_weekly_ops),
        },
        "artifacts": artifacts,
        "snapshots": {
            "full_pipeline": full_pipeline_summary,
            "weekly_ops": weekly_payload,
            "ingestion_health": ingestion_health_payload,
        },
        "full_pipeline": full_pipeline_summary,
        "weekly_ops": weekly_payload,
        "promotion_gate": full_pipeline_summary.get("promotion_gate"),
        "artifact_lanes": full_pipeline_summary.get("artifact_lanes"),
        "status": "ok",
    }

    out_path = Path(production_summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[prod] wrote summary -> {out_path}")
    return summary


def _parse_args() -> argparse.Namespace:
    defaults = PRODUCTION_PIPELINE_DEFAULTS
    parser = argparse.ArgumentParser(
        description=(
            "Production-grade orchestrator: full pipeline + artifact lock + optional weekly scout ops "
            "with one command."
        )
    )
    parser.add_argument("--players-source", default=defaults.players_source)
    parser.add_argument("--data-dir", default=defaults.data_dir)
    parser.add_argument("--external-dir", default=defaults.external_dir)
    parser.add_argument("--dataset-output", default=defaults.dataset_output)
    parser.add_argument("--clean-output", default=defaults.clean_output)
    parser.add_argument("--predictions-output", default=defaults.predictions_output)
    parser.add_argument("--val-season", default=defaults.val_season)
    parser.add_argument("--test-season", default=defaults.test_season)
    parser.add_argument("--start-season", default=defaults.start_season)
    parser.add_argument("--end-season", default=defaults.end_season)

    parser.add_argument("--min-minutes", type=float, default=defaults.min_minutes)
    parser.add_argument("--trials", type=int, default=defaults.trials)
    parser.add_argument("--optimize-metric", default=defaults.optimize_metric)
    parser.add_argument(
        "--league-holdouts",
        default="",
        help="Optional comma-separated holdout leagues for valuation promotion review.",
    )
    parser.add_argument("--band-min-samples", type=int, default=defaults.band_min_samples)
    parser.add_argument("--band-blend-alpha", type=float, default=defaults.band_blend_alpha)
    parser.add_argument("--mape-min-denom-eur", type=float, default=defaults.mape_min_denom_eur)

    parser.add_argument("--with-backtest", action=argparse.BooleanOptionalAction, default=defaults.with_backtest)
    parser.add_argument("--backtest-test-seasons", default=defaults.backtest_test_seasons)
    parser.add_argument("--backtest-enforce-quality-gate", action="store_true")
    parser.add_argument("--backtest-min-test-r2", type=float, default=defaults.backtest_min_test_r2)
    parser.add_argument("--backtest-max-test-wmape", type=float, default=defaults.backtest_max_test_wmape)
    parser.add_argument("--backtest-max-under5m-wmape", type=float, default=defaults.backtest_max_under5m_wmape)
    parser.add_argument("--backtest-max-lowmid-weighted-wmape", type=float, default=defaults.backtest_max_lowmid_weighted_wmape)
    parser.add_argument("--backtest-max-segment-weighted-wmape", type=float, default=defaults.backtest_max_segment_weighted_wmape)
    parser.add_argument("--backtest-min-test-samples", type=int, default=defaults.backtest_min_test_samples)
    parser.add_argument("--backtest-min-test-under5m-samples", type=int, default=defaults.backtest_min_test_under5m_samples)
    parser.add_argument("--backtest-min-test-over20m-samples", type=int, default=defaults.backtest_min_test_over20m_samples)
    parser.add_argument(
        "--backtest-exclude-latest-season",
        action=argparse.BooleanOptionalAction,
        default=defaults.backtest_exclude_latest_season,
    )
    parser.add_argument("--backtest-skip-incomplete-test-seasons", action=argparse.BooleanOptionalAction, default=defaults.backtest_skip_incomplete_test_seasons)

    parser.add_argument("--drop-incomplete-league-seasons", action=argparse.BooleanOptionalAction, default=defaults.drop_incomplete_league_seasons)
    parser.add_argument("--min-league-season-rows", type=int, default=defaults.min_league_season_rows)
    parser.add_argument("--min-league-season-completeness", type=float, default=defaults.min_league_season_completeness)
    parser.add_argument("--residual-calibration-min-samples", type=int, default=defaults.residual_calibration_min_samples)
    parser.add_argument("--provider-config-json", default=defaults.provider_config_json)
    parser.add_argument("--provider-audit-json", default=defaults.provider_audit_json)
    parser.add_argument("--provider-audit-csv", default=defaults.provider_audit_csv)

    parser.add_argument("--skip-injuries", action="store_true")
    parser.add_argument("--skip-contracts", action="store_true")
    parser.add_argument("--skip-transfers", action="store_true")
    parser.add_argument("--skip-national", action="store_true")
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--skip-dataset-build", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")

    parser.add_argument("--lock-manifest-out", default=defaults.lock_manifest_out)
    parser.add_argument("--lock-env-out", default=defaults.lock_env_out)
    parser.add_argument("--lock-label", default=defaults.lock_label)

    parser.add_argument("--run-weekly-ops", action=argparse.BooleanOptionalAction, default=defaults.run_weekly_ops)
    parser.add_argument("--weekly-split", default=defaults.weekly_split, choices=["test", "val"])
    parser.add_argument("--weekly-reports-out-dir", default=defaults.weekly_reports_out_dir)
    parser.add_argument("--weekly-non-big5-only", action=argparse.BooleanOptionalAction, default=defaults.weekly_non_big5_only)
    parser.add_argument("--weekly-min-minutes", type=float, default=defaults.weekly_min_minutes)
    parser.add_argument("--weekly-max-age", type=float, default=defaults.weekly_max_age, help="Set negative to disable.")
    parser.add_argument("--weekly-watchlist-tag", default=defaults.weekly_watchlist_tag)

    parser.add_argument("--production-summary-out", default=defaults.production_summary_out)
    parser.add_argument(
        "--full-pipeline-summary-out",
        default="",
        help="Optional structured summary for the underlying full-pipeline run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_production_pipeline(
        players_source=args.players_source,
        data_dir=args.data_dir,
        external_dir=args.external_dir,
        dataset_output=args.dataset_output,
        clean_output=args.clean_output,
        predictions_output=args.predictions_output,
        val_season=args.val_season,
        test_season=args.test_season,
        start_season=args.start_season,
        end_season=args.end_season,
        min_minutes=args.min_minutes,
        trials=args.trials,
        optimize_metric=args.optimize_metric,
        league_holdouts=_parse_csv_tokens(args.league_holdouts),
        band_min_samples=args.band_min_samples,
        band_blend_alpha=args.band_blend_alpha,
        mape_min_denom_eur=args.mape_min_denom_eur,
        with_backtest=args.with_backtest,
        backtest_test_seasons=_parse_csv_tokens(args.backtest_test_seasons),
        backtest_enforce_quality_gate=args.backtest_enforce_quality_gate,
        backtest_min_test_r2=args.backtest_min_test_r2,
        backtest_max_test_wmape=args.backtest_max_test_wmape,
        backtest_max_under5m_wmape=args.backtest_max_under5m_wmape,
        backtest_max_lowmid_weighted_wmape=args.backtest_max_lowmid_weighted_wmape,
        backtest_max_segment_weighted_wmape=args.backtest_max_segment_weighted_wmape,
        backtest_min_test_samples=args.backtest_min_test_samples,
        backtest_min_test_under5m_samples=args.backtest_min_test_under5m_samples,
        backtest_min_test_over20m_samples=args.backtest_min_test_over20m_samples,
        backtest_exclude_latest_season=args.backtest_exclude_latest_season,
        backtest_skip_incomplete_test_seasons=args.backtest_skip_incomplete_test_seasons,
        drop_incomplete_league_seasons=args.drop_incomplete_league_seasons,
        min_league_season_rows=args.min_league_season_rows,
        min_league_season_completeness=args.min_league_season_completeness,
        residual_calibration_min_samples=args.residual_calibration_min_samples,
        provider_config_json=args.provider_config_json or None,
        provider_audit_json=args.provider_audit_json or None,
        provider_audit_csv=args.provider_audit_csv or None,
        skip_injuries=args.skip_injuries,
        skip_contracts=args.skip_contracts,
        skip_transfers=args.skip_transfers,
        skip_national=args.skip_national,
        skip_context=args.skip_context,
        skip_dataset_build=args.skip_dataset_build,
        skip_clean=args.skip_clean,
        lock_manifest_out=args.lock_manifest_out,
        lock_env_out=args.lock_env_out,
        lock_label=args.lock_label,
        run_weekly_ops=args.run_weekly_ops,
        weekly_split=args.weekly_split,
        weekly_reports_out_dir=args.weekly_reports_out_dir,
        weekly_non_big5_only=args.weekly_non_big5_only,
        weekly_max_age=None if args.weekly_max_age < 0 else args.weekly_max_age,
        weekly_min_minutes=args.weekly_min_minutes,
        weekly_watchlist_tag=args.weekly_watchlist_tag,
        production_summary_out=args.production_summary_out,
        full_pipeline_summary_out=args.full_pipeline_summary_out or None,
    )


if __name__ == "__main__":
    main()
