from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scouting_ml.models.build_dataset import main as build_dataset_main
from scouting_ml.models.clean_dataset import clean_dataset
from scouting_ml.models.train_market_value_full import main as train_market_value_main
from scouting_ml.scripts.build_club_league_context import build_club_and_league_context
from scouting_ml.scripts.build_national_team_caps import build_player_national_caps
from scouting_ml.scripts.build_player_contracts import build_player_contracts
from scouting_ml.scripts.build_player_injuries import build_player_injuries
from scouting_ml.scripts.build_player_transfers import build_player_transfers
from scouting_ml.scripts.build_future_value_targets import build_future_value_targets
from scouting_ml.scripts.lock_market_value_artifacts import build_lock_bundle
from scouting_ml.scripts.run_market_value_ablation import run_ablation
from scouting_ml.scripts.run_rolling_backtest import run_rolling_backtest


def _parse_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _step(label: str) -> None:
    print("\n" + "=" * 70)
    print(f"[pipeline] {label}")
    print("=" * 70)


def run_full_pipeline(
    players_source: str,
    data_dir: str,
    external_dir: str,
    dataset_output: str,
    clean_output: str,
    predictions_output: str,
    val_season: str,
    test_season: str,
    start_season: str | None,
    end_season: str | None,
    min_minutes: float,
    trials: int,
    recency_half_life: float,
    under_5m_weight: float,
    mid_5m_to_20m_weight: float,
    over_20m_weight: float,
    exclude_prefixes: Sequence[str],
    exclude_columns: Sequence[str],
    optimize_metric: str,
    interval_q: float,
    two_stage_band_model: bool,
    band_min_samples: int,
    band_blend_alpha: float,
    strict_leakage_guard: bool,
    strict_quality_gate: bool,
    league_holdouts: Sequence[str],
    drop_incomplete_league_seasons: bool,
    min_league_season_rows: int,
    min_league_season_completeness: float,
    residual_calibration_min_samples: int,
    mape_min_denom_eur: float,
    max_players: int | None,
    sleep_seconds: float,
    transfer_dynamic_fallback: bool,
    transfer_dynamic_fallback_attempts: int,
    contracts_all_seasons: bool,
    national_all_seasons: bool,
    fetch_missing_profiles: bool,
    fetch_national_page: bool,
    with_ablation: bool,
    with_backtest: bool,
    ablation_configs: Sequence[str],
    ablation_out_dir: str,
    backtest_out_dir: str,
    backtest_min_train_seasons: int,
    backtest_test_seasons: Sequence[str],
    backtest_enforce_quality_gate: bool,
    backtest_min_test_r2: float,
    backtest_max_test_mape: float | None,
    backtest_max_test_wmape: float,
    backtest_max_under5m_wmape: float,
    backtest_max_lowmid_weighted_wmape: float,
    backtest_max_segment_weighted_wmape: float,
    backtest_min_test_samples: int,
    backtest_min_test_under5m_samples: int,
    backtest_min_test_over20m_samples: int,
    backtest_skip_incomplete_test_seasons: bool,
    backtest_drop_incomplete_league_seasons: bool,
    backtest_min_league_season_rows: int,
    backtest_min_league_season_completeness: float,
    backtest_residual_calibration_min_samples: int,
    backtest_mape_min_denom_eur: float,
    with_future_targets: bool,
    future_targets_output: str,
    future_target_min_next_minutes: float,
    future_target_drop_na: bool,
    skip_injuries: bool,
    skip_contracts: bool,
    skip_transfers: bool,
    skip_national: bool,
    skip_context: bool,
    skip_dataset_build: bool,
    skip_clean: bool,
    skip_train: bool,
    lock_artifacts: bool,
    lock_manifest_out: str,
    lock_env_out: str,
    lock_label: str,
    lock_strict_artifacts: bool,
) -> None:
    ext_dir = Path(external_dir)
    ext_dir.mkdir(parents=True, exist_ok=True)

    if not skip_injuries:
        _step("Build player injuries")
        build_player_injuries(
            players_source=players_source,
            output=str(ext_dir / "player_injuries.csv"),
            sleep_seconds=sleep_seconds,
            max_players=max_players,
            start_season=start_season,
            end_season=end_season,
            include_failed=True,
            overwrite_cache=False,
        )
    else:
        print("[pipeline] skip injuries")

    if not skip_contracts:
        _step("Build player contracts")
        build_player_contracts(
            players_source=players_source,
            output=str(ext_dir / "player_contracts.csv"),
            start_season=start_season,
            end_season=end_season,
            latest_only=not contracts_all_seasons,
            max_players=max_players,
            fetch_missing=fetch_missing_profiles,
            sleep_seconds=sleep_seconds,
        )
    else:
        print("[pipeline] skip contracts")

    if not skip_transfers:
        _step("Build player transfers")
        build_player_transfers(
            players_source=players_source,
            output=str(ext_dir / "player_transfers.csv"),
            start_season=start_season,
            end_season=end_season,
            max_players=max_players,
            fetch_missing=True,
            sleep_seconds=sleep_seconds,
            overwrite_cache=False,
            include_failed=True,
            dynamic_fallback=transfer_dynamic_fallback,
            max_dynamic_fallback_attempts=transfer_dynamic_fallback_attempts,
        )
    else:
        print("[pipeline] skip transfers")

    if not skip_national:
        _step("Build national-team caps")
        build_player_national_caps(
            players_source=players_source,
            output=str(ext_dir / "national_team_caps.csv"),
            start_season=start_season,
            end_season=end_season,
            latest_only=not national_all_seasons,
            max_players=max_players,
            fetch_missing_profiles=fetch_missing_profiles,
            fetch_national_page=fetch_national_page,
            sleep_seconds=sleep_seconds,
            overwrite_cache=False,
            include_failed=True,
        )
    else:
        print("[pipeline] skip national-team caps")

    if not skip_context:
        _step("Build club + league context")
        build_club_and_league_context(
            players_source=data_dir,
            club_output=str(ext_dir / "club_context.csv"),
            league_output=str(ext_dir / "league_context.csv"),
            start_season=start_season,
            end_season=end_season,
            min_player_minutes=0,
        )
    else:
        print("[pipeline] skip context")

    if not skip_dataset_build:
        _step("Build modeling dataset")
        build_dataset_main(
            data_dir=data_dir,
            output=dataset_output,
            external_dir=external_dir,
        )
    else:
        print("[pipeline] skip dataset build")

    if not skip_clean:
        _step("Clean modeling dataset")
        clean_dataset(
            input_path=dataset_output,
            output_path=clean_output,
            min_minutes=min_minutes,
        )
    else:
        print("[pipeline] skip dataset clean")

    if with_future_targets:
        _step("Build future outcome targets (next-season value growth)")
        build_future_value_targets(
            input_path=clean_output,
            output_path=future_targets_output,
            min_next_minutes=future_target_min_next_minutes,
            drop_na_target=future_target_drop_na,
        )
    else:
        print("[pipeline] skip future outcome target build")

    if not skip_train:
        _step("Train and evaluate main market-value model")
        train_market_value_main(
            dataset_path=clean_output,
            val_season=val_season,
            test_season=test_season,
            output_path=predictions_output,
            val_output_path=None,
            metrics_output_path=None,
            n_optuna_trials=trials,
            recency_half_life=recency_half_life,
            under_5m_weight=under_5m_weight,
            mid_5m_to_20m_weight=mid_5m_to_20m_weight,
            over_20m_weight=over_20m_weight,
            exclude_prefixes=exclude_prefixes,
            exclude_columns=exclude_columns,
            optimize_metric=optimize_metric,
            interval_q=interval_q,
            two_stage_band_model=two_stage_band_model,
            band_min_samples=band_min_samples,
            band_blend_alpha=band_blend_alpha,
            strict_leakage_guard=strict_leakage_guard,
            strict_quality_gate=strict_quality_gate,
            league_holdouts=league_holdouts,
            drop_incomplete_league_seasons=drop_incomplete_league_seasons,
            min_league_season_rows=min_league_season_rows,
            min_league_season_completeness=min_league_season_completeness,
            residual_calibration_min_samples=residual_calibration_min_samples,
            mape_min_denom_eur=mape_min_denom_eur,
        )
    else:
        print("[pipeline] skip train")

    if with_ablation:
        _step("Run ablation study")
        run_ablation(
            dataset_path=clean_output,
            val_season=val_season,
            test_season=test_season,
            out_dir=ablation_out_dir,
            config_names=ablation_configs,
            trials=trials,
            recency_half_life=recency_half_life,
            under_5m_weight=under_5m_weight,
            mid_5m_to_20m_weight=mid_5m_to_20m_weight,
            over_20m_weight=over_20m_weight,
        )

    if with_backtest:
        _step("Run rolling backtest")
        run_rolling_backtest(
            dataset_path=clean_output,
            out_dir=backtest_out_dir,
            trials=trials,
            recency_half_life=recency_half_life,
            optimize_metric=optimize_metric,
            interval_q=interval_q,
            strict_leakage_guard=strict_leakage_guard,
            strict_quality_gate=strict_quality_gate,
            two_stage_band_model=two_stage_band_model,
            band_min_samples=band_min_samples,
            band_blend_alpha=band_blend_alpha,
            min_train_seasons=backtest_min_train_seasons,
            test_seasons=backtest_test_seasons,
            enforce_quality_gate=backtest_enforce_quality_gate,
            min_test_r2=backtest_min_test_r2,
            max_test_mape=backtest_max_test_mape,
            max_test_wmape=backtest_max_test_wmape,
            max_under5m_wmape=backtest_max_under5m_wmape,
            max_lowmid_weighted_wmape=backtest_max_lowmid_weighted_wmape,
            max_segment_weighted_wmape=backtest_max_segment_weighted_wmape,
            min_test_samples=backtest_min_test_samples,
            min_test_under5m_samples=backtest_min_test_under5m_samples,
            min_test_over20m_samples=backtest_min_test_over20m_samples,
            skip_incomplete_test_seasons=backtest_skip_incomplete_test_seasons,
            drop_incomplete_league_seasons=backtest_drop_incomplete_league_seasons,
            min_league_season_rows=backtest_min_league_season_rows,
            min_league_season_completeness=backtest_min_league_season_completeness,
            residual_calibration_min_samples=backtest_residual_calibration_min_samples,
            mape_min_denom_eur=backtest_mape_min_denom_eur,
            under_5m_weight=under_5m_weight,
            mid_5m_to_20m_weight=mid_5m_to_20m_weight,
            over_20m_weight=over_20m_weight,
            exclude_prefixes=exclude_prefixes,
            exclude_columns=exclude_columns,
        )

    if lock_artifacts:
        _step("Freeze active artifacts (manifest + env lock)")
        output = Path(predictions_output)
        val_output = output.with_name(f"{output.stem}_val{output.suffix or '.csv'}")
        metrics_output = output.with_suffix(".metrics.json")
        build_lock_bundle(
            test_predictions=output,
            val_predictions=val_output,
            metrics_path=metrics_output,
            manifest_out=Path(lock_manifest_out),
            env_out=Path(lock_env_out),
            strict_artifacts=bool(lock_strict_artifacts),
            label=str(lock_label),
        )

    _step("Pipeline complete")
    print(f"[pipeline] dataset: {dataset_output}")
    print(f"[pipeline] clean dataset: {clean_output}")
    if with_future_targets:
        print(f"[pipeline] future-target dataset: {future_targets_output}")
    print(f"[pipeline] predictions: {predictions_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full scouting ML pipeline end-to-end with one command."
    )

    parser.add_argument("--players-source", default="data/processed/Clubs combined")
    parser.add_argument("--data-dir", default="data/processed/Clubs combined")
    parser.add_argument("--external-dir", default="data/external")
    parser.add_argument("--dataset-output", default="data/model/big5_players.parquet")
    parser.add_argument("--clean-output", default="data/model/big5_players_clean.parquet")
    parser.add_argument("--predictions-output", default="data/model/big5_predictions_full_v2.csv")
    parser.add_argument("--val-season", default="2023/24")
    parser.add_argument("--test-season", default="2024/25")
    parser.add_argument("--start-season", default="2019/20")
    parser.add_argument("--end-season", default="2024/25")

    parser.add_argument("--min-minutes", type=float, default=450.0)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--recency-half-life", type=float, default=2.0)
    parser.add_argument("--under-5m-weight", type=float, default=1.0)
    parser.add_argument("--mid-5m-20m-weight", type=float, default=1.0)
    parser.add_argument("--over-20m-weight", type=float, default=1.0)
    parser.add_argument(
        "--optimize-metric",
        default="lowmid_wmape",
        choices=["mae", "rmse", "band_wmape", "lowmid_wmape"],
        help="Optuna objective for market-value training.",
    )
    parser.add_argument("--interval-q", type=float, default=0.8)
    parser.add_argument(
        "--two-stage-band-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable value-band classifier + expert regressors before final output.",
    )
    parser.add_argument(
        "--band-min-samples",
        type=int,
        default=160,
        help="Minimum samples needed to train each value-band expert.",
    )
    parser.add_argument(
        "--band-blend-alpha",
        type=float,
        default=0.35,
        help="Blend factor between global model and band expert predictions.",
    )
    parser.add_argument(
        "--strict-leakage-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if leakage-like columns appear in feature set.",
    )
    parser.add_argument(
        "--strict-quality-gate",
        action="store_true",
        help="Fail training if quality flags are detected.",
    )
    parser.add_argument(
        "--league-holdouts",
        default="",
        help="Comma-separated holdout leagues for unseen-league evaluation.",
    )
    parser.add_argument(
        "--drop-incomplete-league-seasons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop incomplete league-season groups before main training.",
    )
    parser.add_argument("--min-league-season-rows", type=int, default=40)
    parser.add_argument("--min-league-season-completeness", type=float, default=0.55)
    parser.add_argument("--residual-calibration-min-samples", type=int, default=30)
    parser.add_argument(
        "--mape-min-denom-eur",
        type=float,
        default=1_000_000.0,
        help="MAPE denominator floor in EUR for training/evaluation metrics.",
    )
    parser.add_argument("--exclude-prefixes", default="")
    parser.add_argument("--exclude-columns", default="")

    parser.add_argument("--max-players", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=2.5)
    parser.add_argument(
        "--transfer-dynamic-fallback",
        action="store_true",
        help="Try dynamic Transfermarkt fallback URLs for JS-only transfer pages.",
    )
    parser.add_argument(
        "--transfer-dynamic-fallback-attempts",
        type=int,
        default=2,
        help="Max fallback URL attempts per player when dynamic transfer fallback is enabled.",
    )
    parser.add_argument("--contracts-all-seasons", action="store_true")
    parser.add_argument("--national-all-seasons", action="store_true")
    parser.add_argument("--fetch-missing-profiles", action="store_true")
    parser.add_argument("--fetch-national-page", action="store_true")

    parser.add_argument("--with-ablation", action="store_true")
    parser.add_argument(
        "--with-backtest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run rolling time backtest (recommended default).",
    )
    parser.add_argument(
        "--ablation-configs",
        default="full,no_contract,no_injury,no_transfer,no_national,no_context,baseline_stats_only",
    )
    parser.add_argument("--ablation-out-dir", default="data/model/ablation")
    parser.add_argument("--backtest-out-dir", default="data/model/backtests")
    parser.add_argument("--backtest-min-train-seasons", type=int, default=2)
    parser.add_argument("--backtest-test-seasons", default="")
    parser.add_argument(
        "--backtest-enforce-quality-gate",
        action="store_true",
        help="Fail pipeline if rolling backtest quality thresholds are missed.",
    )
    parser.add_argument("--backtest-min-test-r2", type=float, default=0.60)
    parser.add_argument(
        "--backtest-max-test-mape",
        type=float,
        default=None,
        help="Optional MAPE gate threshold (set to disable MAPE gate).",
    )
    parser.add_argument("--backtest-max-test-wmape", type=float, default=0.42)
    parser.add_argument("--backtest-max-under5m-wmape", type=float, default=0.50)
    parser.add_argument("--backtest-max-lowmid-weighted-wmape", type=float, default=0.48)
    parser.add_argument("--backtest-max-segment-weighted-wmape", type=float, default=0.45)
    parser.add_argument("--backtest-min-test-samples", type=int, default=300)
    parser.add_argument("--backtest-min-test-under5m-samples", type=int, default=50)
    parser.add_argument("--backtest-min-test-over20m-samples", type=int, default=25)
    parser.add_argument(
        "--backtest-skip-incomplete-test-seasons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rolling backtest seasons with incomplete/small test splits.",
    )
    parser.add_argument(
        "--backtest-drop-incomplete-league-seasons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop incomplete league-season groups within each rolling backtest training run.",
    )
    parser.add_argument("--backtest-min-league-season-rows", type=int, default=40)
    parser.add_argument("--backtest-min-league-season-completeness", type=float, default=0.55)
    parser.add_argument("--backtest-residual-calibration-min-samples", type=int, default=30)
    parser.add_argument("--backtest-mape-min-denom-eur", type=float, default=1_000_000.0)
    parser.add_argument(
        "--with-future-targets",
        action="store_true",
        help="Build next-season value-growth target dataset for future-outcome modeling.",
    )
    parser.add_argument(
        "--future-targets-output",
        default="data/model/big5_players_future_targets.parquet",
    )
    parser.add_argument("--future-target-min-next-minutes", type=float, default=450.0)
    parser.add_argument(
        "--future-target-drop-na",
        action="store_true",
        help="Only keep rows that have a valid next-season target.",
    )

    parser.add_argument("--skip-injuries", action="store_true")
    parser.add_argument("--skip-contracts", action="store_true")
    parser.add_argument("--skip-transfers", action="store_true")
    parser.add_argument("--skip-national", action="store_true")
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--skip-dataset-build", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--lock-artifacts",
        action="store_true",
        help="Write model manifest + env lock file after pipeline run.",
    )
    parser.add_argument("--lock-manifest-out", default="data/model/model_manifest.json")
    parser.add_argument("--lock-env-out", default="data/model/model_artifacts.env")
    parser.add_argument("--lock-label", default="market_value_champion")
    parser.add_argument(
        "--lock-strict-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write strict artifact serving flag into lock env file.",
    )

    args = parser.parse_args()

    run_full_pipeline(
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
        recency_half_life=args.recency_half_life,
        under_5m_weight=args.under_5m_weight,
        mid_5m_to_20m_weight=args.mid_5m_20m_weight,
        over_20m_weight=args.over_20m_weight,
        optimize_metric=args.optimize_metric,
        interval_q=args.interval_q,
        two_stage_band_model=args.two_stage_band_model,
        band_min_samples=args.band_min_samples,
        band_blend_alpha=args.band_blend_alpha,
        strict_leakage_guard=args.strict_leakage_guard,
        strict_quality_gate=args.strict_quality_gate,
        league_holdouts=_parse_csv_tokens(args.league_holdouts),
        drop_incomplete_league_seasons=args.drop_incomplete_league_seasons,
        min_league_season_rows=args.min_league_season_rows,
        min_league_season_completeness=args.min_league_season_completeness,
        residual_calibration_min_samples=args.residual_calibration_min_samples,
        mape_min_denom_eur=args.mape_min_denom_eur,
        exclude_prefixes=_parse_csv_tokens(args.exclude_prefixes),
        exclude_columns=_parse_csv_tokens(args.exclude_columns),
        max_players=args.max_players,
        sleep_seconds=args.sleep,
        transfer_dynamic_fallback=args.transfer_dynamic_fallback,
        transfer_dynamic_fallback_attempts=args.transfer_dynamic_fallback_attempts,
        contracts_all_seasons=args.contracts_all_seasons,
        national_all_seasons=args.national_all_seasons,
        fetch_missing_profiles=args.fetch_missing_profiles,
        fetch_national_page=args.fetch_national_page,
        with_ablation=args.with_ablation,
        with_backtest=args.with_backtest,
        ablation_configs=_parse_csv_tokens(args.ablation_configs),
        ablation_out_dir=args.ablation_out_dir,
        backtest_out_dir=args.backtest_out_dir,
        backtest_min_train_seasons=args.backtest_min_train_seasons,
        backtest_test_seasons=_parse_csv_tokens(args.backtest_test_seasons),
        backtest_enforce_quality_gate=args.backtest_enforce_quality_gate,
        backtest_min_test_r2=args.backtest_min_test_r2,
        backtest_max_test_mape=args.backtest_max_test_mape,
        backtest_max_test_wmape=args.backtest_max_test_wmape,
        backtest_max_under5m_wmape=args.backtest_max_under5m_wmape,
        backtest_max_lowmid_weighted_wmape=args.backtest_max_lowmid_weighted_wmape,
        backtest_max_segment_weighted_wmape=args.backtest_max_segment_weighted_wmape,
        backtest_min_test_samples=args.backtest_min_test_samples,
        backtest_min_test_under5m_samples=args.backtest_min_test_under5m_samples,
        backtest_min_test_over20m_samples=args.backtest_min_test_over20m_samples,
        backtest_skip_incomplete_test_seasons=args.backtest_skip_incomplete_test_seasons,
        backtest_drop_incomplete_league_seasons=args.backtest_drop_incomplete_league_seasons,
        backtest_min_league_season_rows=args.backtest_min_league_season_rows,
        backtest_min_league_season_completeness=args.backtest_min_league_season_completeness,
        backtest_residual_calibration_min_samples=args.backtest_residual_calibration_min_samples,
        backtest_mape_min_denom_eur=args.backtest_mape_min_denom_eur,
        with_future_targets=args.with_future_targets,
        future_targets_output=args.future_targets_output,
        future_target_min_next_minutes=args.future_target_min_next_minutes,
        future_target_drop_na=args.future_target_drop_na,
        skip_injuries=args.skip_injuries,
        skip_contracts=args.skip_contracts,
        skip_transfers=args.skip_transfers,
        skip_national=args.skip_national,
        skip_context=args.skip_context,
        skip_dataset_build=args.skip_dataset_build,
        skip_clean=args.skip_clean,
        skip_train=args.skip_train,
        lock_artifacts=args.lock_artifacts,
        lock_manifest_out=args.lock_manifest_out,
        lock_env_out=args.lock_env_out,
        lock_label=args.lock_label,
        lock_strict_artifacts=args.lock_strict_artifacts,
    )


if __name__ == "__main__":
    main()
