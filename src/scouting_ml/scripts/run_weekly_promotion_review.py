from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scouting_ml.core.runtime_config import PRODUCTION_PIPELINE_DEFAULTS
from scouting_ml.scripts.run_future_data_refresh import _summarize_talent_outputs
from scouting_ml.scripts.run_production_pipeline import _parse_csv_tokens, run_production_pipeline


def _lane_line(role: str, section: dict[str, object] | None) -> str:
    payload = section if isinstance(section, dict) else {}
    label = str(payload.get("label") or role)
    lane_state = str(payload.get("lane_state") or "unknown")
    promotion_state = str(payload.get("promotion_state") or "advisory_only")
    return f"[weekly-review] {role}: {label} | {lane_state} | {promotion_state}"


def _talent_line(summary: dict[str, object] | None) -> str:
    payload = summary if isinstance(summary, dict) else {}
    label_cov = payload.get("future_label_coverage") if isinstance(payload.get("future_label_coverage"), dict) else {}
    confidence = payload.get("confidence_distribution") if isinstance(payload.get("confidence_distribution"), dict) else {}
    families = payload.get("position_family_counts") if isinstance(payload.get("position_family_counts"), dict) else {}
    family_text = ", ".join(f"{key}:{value}" for key, value in sorted(families.items())) or "none"
    return (
        "[weekly-review] future talent -> "
        f"labels {label_cov.get('labeled_rows', 0)}/{label_cov.get('total_rows', 0)} | "
        f"confidence high/med/low {confidence.get('high', 0)}/{confidence.get('medium', 0)}/{confidence.get('low', 0)} | "
        f"positions {family_text}"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = PRODUCTION_PIPELINE_DEFAULTS
    parser = argparse.ArgumentParser(
        description=(
            "Weekly/manual promotion-review wrapper: run the valuation production pipeline, "
            "evaluate promotability, and print the operator summary."
        )
    )
    parser.add_argument("--league-holdouts", default="")
    parser.add_argument("--summary-json", default=defaults.production_summary_out)
    parser.add_argument(
        "--full-pipeline-summary-out",
        default="data/model/production/full_pipeline_summary.json",
    )
    parser.add_argument("--lock-manifest-out", default=defaults.lock_manifest_out)
    parser.add_argument("--lock-env-out", default=defaults.lock_env_out)
    parser.add_argument("--with-backtest", action=argparse.BooleanOptionalAction, default=defaults.with_backtest)
    parser.add_argument("--run-weekly-ops", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--provider-config-json", default=defaults.provider_config_json)
    parser.add_argument("--trials", type=int, default=defaults.trials)
    parser.add_argument("--val-season", default=defaults.val_season)
    parser.add_argument("--test-season", default=defaults.test_season)
    parser.add_argument("--lock-label", default=defaults.lock_label)
    parser.add_argument(
        "--skip-injuries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing injury enrichments by default; disable to rebuild them.",
    )
    parser.add_argument(
        "--skip-contracts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing contract enrichments by default; disable to rebuild them.",
    )
    parser.add_argument(
        "--skip-transfers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing transfer enrichments by default; disable to rebuild them.",
    )
    parser.add_argument(
        "--skip-national",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing national-team enrichments by default; disable to rebuild them.",
    )
    parser.add_argument(
        "--skip-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing club/league context files by default; disable to rebuild them.",
    )
    parser.add_argument(
        "--skip-dataset-build",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse the existing modeling dataset parquet instead of rebuilding it.",
    )
    parser.add_argument(
        "--skip-clean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse the existing cleaned modeling dataset parquet instead of rebuilding it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    defaults = PRODUCTION_PIPELINE_DEFAULTS
    summary = run_production_pipeline(
        players_source=defaults.players_source,
        data_dir=defaults.data_dir,
        external_dir=defaults.external_dir,
        dataset_output=defaults.dataset_output,
        clean_output=defaults.clean_output,
        predictions_output=defaults.predictions_output,
        val_season=args.val_season,
        test_season=args.test_season,
        start_season=defaults.start_season,
        end_season=defaults.end_season,
        min_minutes=defaults.min_minutes,
        trials=args.trials,
        optimize_metric=defaults.optimize_metric,
        league_holdouts=_parse_csv_tokens(args.league_holdouts),
        band_min_samples=defaults.band_min_samples,
        band_blend_alpha=defaults.band_blend_alpha,
        mape_min_denom_eur=defaults.mape_min_denom_eur,
        with_backtest=args.with_backtest,
        backtest_test_seasons=_parse_csv_tokens(defaults.backtest_test_seasons),
        backtest_enforce_quality_gate=defaults.backtest_enforce_quality_gate,
        backtest_min_test_r2=defaults.backtest_min_test_r2,
        backtest_max_test_wmape=defaults.backtest_max_test_wmape,
        backtest_max_under5m_wmape=defaults.backtest_max_under5m_wmape,
        backtest_max_lowmid_weighted_wmape=defaults.backtest_max_lowmid_weighted_wmape,
        backtest_max_segment_weighted_wmape=defaults.backtest_max_segment_weighted_wmape,
        backtest_min_test_samples=defaults.backtest_min_test_samples,
        backtest_min_test_under5m_samples=defaults.backtest_min_test_under5m_samples,
        backtest_min_test_over20m_samples=defaults.backtest_min_test_over20m_samples,
        backtest_exclude_latest_season=defaults.backtest_exclude_latest_season,
        backtest_skip_incomplete_test_seasons=defaults.backtest_skip_incomplete_test_seasons,
        drop_incomplete_league_seasons=defaults.drop_incomplete_league_seasons,
        min_league_season_rows=defaults.min_league_season_rows,
        min_league_season_completeness=defaults.min_league_season_completeness,
        residual_calibration_min_samples=defaults.residual_calibration_min_samples,
        provider_config_json=args.provider_config_json or None,
        provider_audit_json=defaults.provider_audit_json,
        provider_audit_csv=defaults.provider_audit_csv,
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
        weekly_split=defaults.weekly_split,
        weekly_reports_out_dir=defaults.weekly_reports_out_dir,
        weekly_non_big5_only=defaults.weekly_non_big5_only,
        weekly_max_age=defaults.weekly_max_age,
        weekly_min_minutes=defaults.weekly_min_minutes,
        weekly_watchlist_tag=defaults.weekly_watchlist_tag,
        production_summary_out=args.summary_json,
        full_pipeline_summary_out=args.full_pipeline_summary_out,
    )

    promotion = summary.get("promotion_gate") or {}
    print(f"[weekly-review] summary -> {Path(args.summary_json).resolve()}")
    print(f"[weekly-review] manifest -> {Path(args.lock_manifest_out).resolve()}")
    artifacts = summary.get("artifacts") or {}
    ingestion_meta = artifacts.get("ingestion_health_json") if isinstance(artifacts, dict) else None
    if isinstance(ingestion_meta, dict) and ingestion_meta.get("path"):
        print(f"[weekly-review] operator health -> {ingestion_meta['path']}")
    print(
        "[weekly-review] promotion gate -> "
        f"{'promotable' if promotion.get('promotable') else 'advisory_only'}"
    )
    lanes = summary.get("artifact_lanes") or {}
    if isinstance(lanes, dict):
        for role in ("valuation", "future_shortlist"):
            if role in lanes:
                print(_lane_line(role, lanes.get(role)))
        future_lane = lanes.get("future_shortlist") if isinstance(lanes.get("future_shortlist"), dict) else {}
        artifact_paths = future_lane.get("artifact_paths") if isinstance(future_lane.get("artifact_paths"), dict) else {}
        test_predictions = artifact_paths.get("test_predictions")
        if test_predictions:
            print(_talent_line(_summarize_talent_outputs(test_predictions)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
