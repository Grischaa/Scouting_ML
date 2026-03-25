from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scouting_ml.reporting.operator_health import (
    build_ingestion_health_payload,
    build_valuation_promotion_gate,
    write_json_sidecar,
)
from scouting_ml.scripts import run_nightly_live_refresh as nightly
from scouting_ml.scripts import run_weekly_promotion_review as weekly


def test_build_ingestion_health_payload_reports_healthy_and_blocked_rows(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    austrian_tm = processed_dir / "austrian_bundesliga_2024-25_clean.csv"
    austrian_tm.write_text("player_id,name\n1,A\n2,B\n", encoding="utf-8")
    austrian_sofa = processed_dir / "sofa_austrian_bundesliga_2024-25.csv"
    austrian_sofa.write_text("player_id,name\n1,A\n2,B\n", encoding="utf-8")
    write_json_sidecar(austrian_sofa, {"rows": 2, "zero_rows": False, "header_only": False})
    austrian_merged = processed_dir / "austrian_bundesliga_2024-25_with_sofa.csv"
    austrian_merged.write_text("sofa_matched\n1\n1\n", encoding="utf-8")
    write_json_sidecar(austrian_merged, {"matched_rows": 2, "match_rate": 1.0, "sofa_rows": 2, "tm_rows": 2})

    polish_tm = processed_dir / "polish_ekstraklasa_2024-25_clean.csv"
    polish_tm.write_text("player_id,name\n10,P\n", encoding="utf-8")
    polish_sofa = processed_dir / "sofa_polish_ekstraklasa_2024-25.csv"
    polish_sofa.write_text("player_id,name\n", encoding="utf-8")
    write_json_sidecar(polish_sofa, {"rows": 0, "zero_rows": True, "header_only": True})
    polish_merged = processed_dir / "polish_ekstraklasa_2024-25_with_sofa.csv"
    polish_merged.write_text("sofa_matched\n0\n", encoding="utf-8")
    write_json_sidecar(polish_merged, {"matched_rows": 0, "match_rate": 0.0, "sofa_rows": 0, "tm_rows": 1})

    clean_dataset = tmp_path / "clean.parquet"
    pd.DataFrame(
        [
            {
                "league": "Austrian Bundesliga",
                "season": "2024/25",
                "sb_snapshot_date": "2026-03-12",
                "avail_snapshot_date": "2026-03-11",
                "fixture_snapshot_date": "2026-03-10",
                "odds_snapshot_date": "2026-03-09",
                "sb_retrieved_at": "2026-03-12T09:00:00Z",
                "avail_retrieved_at": "2026-03-11T09:00:00Z",
                "fixture_retrieved_at": "2026-03-10T09:00:00Z",
                "odds_retrieved_at": "2026-03-09T09:00:00Z",
            }
        ]
    ).to_parquet(clean_dataset, index=False)

    payload = build_ingestion_health_payload(processed_dir=processed_dir, clean_dataset_path=clean_dataset)
    rows = {(row["league_slug"], row["season"]): row for row in payload["rows"]}

    austrian = rows[("austrian_bundesliga", "2024/25")]
    assert austrian["status"] == "healthy"
    assert austrian["tm_rows"] == 2
    assert austrian["sofa_rows"] == 2
    assert austrian["matched_rows"] == 2
    assert austrian["match_rate"] == 1.0
    assert austrian["missing_provider_flags"] == []

    poland = rows[("polish_ekstraklasa", "2024/25")]
    assert poland["status"] == "blocked"
    assert poland["sofa_zero_rows"] is True
    assert "sofa_zero_rows" in poland["status_reasons"]
    assert "match_rate_low" in poland["status_reasons"]


def test_build_valuation_promotion_gate_accepts_finished_window_with_six_holdouts(tmp_path: Path) -> None:
    requested = [f"League {idx}" for idx in range(1, 7)]
    metrics_payload = {
        "requested_league_holdouts": [{"requested_token": item} for item in requested],
        "league_holdout": [
            {"requested_token": item, "league": item, "status": "ok"}
            for item in requested
        ],
    }
    backtest_rows = pd.DataFrame(
        [
            {
                "test_season": "2022/23",
                "test_r2": 0.70,
                "test_wmape": 0.40,
                "test_under_5m_wmape": 0.44,
                "test_lowmid_weighted_wmape": 0.41,
                "test_segment_weighted_wmape": 0.40,
            },
            {
                "test_season": "2023/24",
                "test_r2": 0.71,
                "test_wmape": 0.39,
                "test_under_5m_wmape": 0.43,
                "test_lowmid_weighted_wmape": 0.40,
                "test_segment_weighted_wmape": 0.39,
            },
            {
                "test_season": "2024/25",
                "test_r2": 0.69,
                "test_wmape": 0.41,
                "test_under_5m_wmape": 0.45,
                "test_lowmid_weighted_wmape": 0.42,
                "test_segment_weighted_wmape": 0.41,
            },
        ]
    )
    backtest_csv = tmp_path / "rolling_backtest_summary.csv"
    backtest_rows.to_csv(backtest_csv, index=False)

    gate = build_valuation_promotion_gate(
        metrics_payload=metrics_payload,
        backtest_payload={
            "latest_dataset_season": "2025/26",
            "excluded_latest_dataset_season": "2025/26",
            "skipped_runs": [{"test_season": "2025/26", "reasons": ["latest_dataset_season_excluded"]}],
        },
        backtest_rows_path=backtest_csv,
        requested_backtest_test_seasons=["2022/23", "2023/24", "2024/25", "2025/26"],
    )

    assert gate["promotable"] is True
    assert gate["holdout_coverage"]["successful_count"] == 6
    assert gate["backtest_window"]["excluded_latest_dataset_season"] == "2025/26"
    assert gate["failed_checks"] == []


def test_build_valuation_promotion_gate_blocks_skipped_holdouts(tmp_path: Path) -> None:
    metrics_payload = {
        "requested_league_holdouts": [{"requested_token": "Austrian Bundesliga"}],
        "league_holdout": [
            {"requested_token": "Austrian Bundesliga", "league": "Austrian Bundesliga", "status": "skipped"}
        ],
    }
    backtest_csv = tmp_path / "rolling_backtest_summary.csv"
    pd.DataFrame(
        [
            {
                "test_season": "2024/25",
                "test_r2": 0.70,
                "test_wmape": 0.40,
                "test_under_5m_wmape": 0.44,
                "test_lowmid_weighted_wmape": 0.41,
                "test_segment_weighted_wmape": 0.40,
            }
        ]
    ).to_csv(backtest_csv, index=False)

    gate = build_valuation_promotion_gate(
        metrics_payload=metrics_payload,
        backtest_payload={"skipped_runs": []},
        backtest_rows_path=backtest_csv,
        requested_backtest_test_seasons=["2024/25"],
    )

    assert gate["promotable"] is False
    assert any("Skipped holdouts" in message for message in gate["failed_checks"])


def test_run_nightly_live_refresh_prints_operator_paths(tmp_path: Path, monkeypatch, capsys) -> None:
    defaults = argparse.Namespace(
        import_dir=None,
        season_filter="2025-26",
        summary_json=str(tmp_path / "future_refresh_summary.json"),
        run_promotion=False,
        manifest_out=str(tmp_path / "model_manifest.json"),
        env_out=str(tmp_path / "model_artifacts.env"),
        bundle_label="future_bundle",
        promote_on_pass=True,
        scored_val_output=str(tmp_path / "future_val.csv"),
        scored_test_output=str(tmp_path / "future_test.csv"),
        staging_dir=str(tmp_path / "staging"),
        source_dir=str(tmp_path / "processed"),
        combined_dir=str(tmp_path / "combined"),
        country_root=str(tmp_path / "by_country"),
        season_root=str(tmp_path / "by_season"),
        organization_manifest=str(tmp_path / "organization_manifest.csv"),
        no_clean_targets=False,
        dataset_output=str(tmp_path / "dataset.parquet"),
        clean_output=str(tmp_path / "clean.parquet"),
        external_dir=str(tmp_path / "external"),
        clean_min_minutes=450.0,
        future_targets_output=str(tmp_path / "future_targets.parquet"),
        future_audit_json=str(tmp_path / "future_audit.json"),
        future_audit_csv=str(tmp_path / "future_audit.csv"),
        future_source_glob="data/processed/**/*2025-26*_with_sofa.csv",
        min_next_minutes=450.0,
        base_val_predictions=str(tmp_path / "base_val.csv"),
        base_test_predictions=str(tmp_path / "base_test.csv"),
        diagnostics_output=str(tmp_path / "diagnostics.json"),
        label_mode="positive_growth",
        k_eval=25,
        scoring_min_minutes=900.0,
        scoring_max_age=-1.0,
        scoring_positions="",
        scoring_include_leagues="",
        scoring_exclude_leagues="",
        future_benchmark_json=str(tmp_path / "future_benchmark.json"),
        future_benchmark_md=str(tmp_path / "future_benchmark.md"),
        future_diagnostics_json=str(tmp_path / "future_diagnostics.json"),
        future_diagnostics_md=str(tmp_path / "future_diagnostics.md"),
        future_u23_nonbig5_json=str(tmp_path / "u23.json"),
        future_u23_nonbig5_md=str(tmp_path / "u23.md"),
        future_score_col="future_scout_blend_score",
        future_cohort_min_labeled=25,
        future_k_values="10,25,50",
        champion_metrics=str(tmp_path / "champion.metrics.json"),
        champion_label="champion",
        candidate_metrics=str(tmp_path / "candidate.metrics.json"),
        candidate_holdout_glob="candidate/*.json",
        reference_holdout_glob="reference/*.json",
        champion_future_benchmark_json=str(tmp_path / "champion_future_benchmark.json"),
        candidate_label="future_candidate",
        reference_label="reference",
        comparison_out_json=str(tmp_path / "comparison.json"),
        comparison_out_md=str(tmp_path / "comparison.md"),
        benchmark_out_json=str(tmp_path / "benchmark.json"),
        benchmark_out_md=str(tmp_path / "benchmark.md"),
        onboarding_json=str(tmp_path / "onboarding.json"),
        ablation_bundle=str(tmp_path / "ablation.json"),
    )

    monkeypatch.setattr(nightly, "_parse_future_defaults", lambda argv=None: defaults)
    monkeypatch.setattr(
        nightly,
        "run_future_data_refresh",
        lambda **kwargs: {
            "artifacts": {"clean_output": {"path": str((tmp_path / "clean.parquet").resolve())}},
            "artifact_lanes": {
                "valuation": {"label": "champion", "lane_state": "stable", "promotion_state": "advisory_only"},
                "future_shortlist": {"label": "future_bundle", "lane_state": "live", "promotion_state": "advisory_only"},
            },
        },
    )
    monkeypatch.setattr(
        nightly,
        "regenerate_ingestion_health_report",
        lambda clean_dataset_path=None: {"_meta": {"json_path": str((tmp_path / "ingestion_health.json").resolve())}},
    )

    rc = nightly.main([])
    output = capsys.readouterr().out
    assert rc == 0
    assert "[nightly] operator health ->" in output
    assert "[nightly] valuation:" in output
    assert "[nightly] future_shortlist:" in output


def test_run_nightly_live_refresh_falls_back_to_champion_baselines(tmp_path: Path, monkeypatch) -> None:
    champion_val = tmp_path / "champion_predictions_2024-25_val.csv"
    champion_test = tmp_path / "champion_predictions_2024-25.csv"
    champion_metrics = tmp_path / "champion_predictions_2024-25.metrics.json"
    champion_val.write_text("player_id\n1\n", encoding="utf-8")
    champion_test.write_text("player_id\n1\n", encoding="utf-8")
    champion_metrics.write_text("{}", encoding="utf-8")

    defaults = argparse.Namespace(
        import_dir=None,
        season_filter="2025-26",
        summary_json=str(tmp_path / "future_refresh_summary.json"),
        run_promotion=True,
        manifest_out=str(tmp_path / "model_manifest.json"),
        env_out=str(tmp_path / "model_artifacts.env"),
        bundle_label="future_bundle",
        promote_on_pass=True,
        scored_val_output=str(tmp_path / "future_val.csv"),
        scored_test_output=str(tmp_path / "future_test.csv"),
        staging_dir=str(tmp_path / "staging"),
        source_dir=str(tmp_path / "processed"),
        combined_dir=str(tmp_path / "combined"),
        country_root=str(tmp_path / "by_country"),
        season_root=str(tmp_path / "by_season"),
        organization_manifest=str(tmp_path / "organization_manifest.csv"),
        no_clean_targets=False,
        dataset_output=str(tmp_path / "dataset.parquet"),
        clean_output=str(tmp_path / "clean.parquet"),
        external_dir=str(tmp_path / "external"),
        clean_min_minutes=450.0,
        future_targets_output=str(tmp_path / "future_targets.parquet"),
        future_audit_json=str(tmp_path / "future_audit.json"),
        future_audit_csv=str(tmp_path / "future_audit.csv"),
        future_source_glob="data/processed/**/*2025-26*_with_sofa.csv",
        min_next_minutes=450.0,
        base_val_predictions=str(tmp_path / "legacy" / "cheap_aggressive_2024-25_val.csv"),
        base_test_predictions=str(tmp_path / "legacy" / "cheap_aggressive_2024-25.csv"),
        diagnostics_output=str(tmp_path / "diagnostics.json"),
        label_mode="positive_growth",
        k_eval=25,
        scoring_min_minutes=900.0,
        scoring_max_age=-1.0,
        scoring_positions="",
        scoring_include_leagues="",
        scoring_exclude_leagues="",
        future_benchmark_json=str(tmp_path / "future_benchmark.json"),
        future_benchmark_md=str(tmp_path / "future_benchmark.md"),
        future_diagnostics_json=str(tmp_path / "future_diagnostics.json"),
        future_diagnostics_md=str(tmp_path / "future_diagnostics.md"),
        future_u23_nonbig5_json=str(tmp_path / "u23.json"),
        future_u23_nonbig5_md=str(tmp_path / "u23.md"),
        future_score_col="future_scout_blend_score",
        future_cohort_min_labeled=25,
        future_k_values="10,25,50",
        champion_metrics=str(champion_metrics),
        champion_label="champion",
        candidate_metrics=str(tmp_path / "legacy" / "cheap_aggressive_2024-25.metrics.json"),
        candidate_holdout_glob="candidate/*.json",
        reference_holdout_glob="reference/*.json",
        champion_future_benchmark_json=str(tmp_path / "champion_future_benchmark.json"),
        candidate_label="future_candidate",
        reference_label="reference",
        comparison_out_json=str(tmp_path / "comparison.json"),
        comparison_out_md=str(tmp_path / "comparison.md"),
        benchmark_out_json=str(tmp_path / "benchmark.json"),
        benchmark_out_md=str(tmp_path / "benchmark.md"),
        onboarding_json=str(tmp_path / "onboarding.json"),
        ablation_bundle=str(tmp_path / "ablation.json"),
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(nightly, "_parse_future_defaults", lambda argv=None: defaults)

    def _run_future_data_refresh(**kwargs):
        captured.update(kwargs)
        return {
            "artifacts": {"clean_output": {"path": str((tmp_path / "clean.parquet").resolve())}},
            "artifact_lanes": {},
        }

    monkeypatch.setattr(nightly, "run_future_data_refresh", _run_future_data_refresh)
    monkeypatch.setattr(
        nightly,
        "regenerate_ingestion_health_report",
        lambda clean_dataset_path=None: {"_meta": {"json_path": str((tmp_path / "ingestion_health.json").resolve())}},
    )

    rc = nightly.main([])

    assert rc == 0
    assert captured["base_val_predictions"] == "data/model/champion_predictions_2024-25_val.csv"
    assert captured["base_test_predictions"] == "data/model/champion_predictions_2024-25.csv"
    promotion_args = captured["promotion_args"]
    candidate_metrics_idx = promotion_args.index("--candidate-metrics") + 1
    assert promotion_args[candidate_metrics_idx] == str(champion_metrics)
    assert "--candidate-future-benchmark-json" in promotion_args
    assert "--require-future-benchmark" in promotion_args
    assert "--require-future-precision-vs-champion" not in promotion_args
    label_idx = promotion_args.index("--label") + 1
    assert promotion_args[label_idx] == "future_bundle"
    primary_role_idx = promotion_args.index("--primary-role") + 1
    assert promotion_args[primary_role_idx] == "future_shortlist"


def test_run_weekly_promotion_review_prints_promotion_gate(tmp_path: Path, monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        weekly,
        "run_production_pipeline",
        lambda **kwargs: captured.update(kwargs)
        or {
            "promotion_gate": {"promotable": False},
            "artifact_lanes": {
                "valuation": {"label": "prod60", "lane_state": "stable", "promotion_state": "advisory_only"},
                "future_shortlist": {"label": "future_bundle", "lane_state": "live", "promotion_state": "advisory_only"},
            },
            "artifacts": {
                "ingestion_health_json": {"path": str((tmp_path / "ingestion_health.json").resolve())}
            },
        },
    )

    rc = weekly.main(["--summary-json", str(tmp_path / "production_summary.json")])
    output = capsys.readouterr().out
    assert rc == 0
    assert captured["skip_injuries"] is True
    assert captured["skip_contracts"] is True
    assert captured["skip_transfers"] is True
    assert captured["skip_national"] is True
    assert captured["skip_context"] is True
    assert captured["skip_dataset_build"] is False
    assert captured["skip_clean"] is False
    assert "[weekly-review] promotion gate -> advisory_only" in output
    assert "[weekly-review] valuation:" in output
    assert "[weekly-review] future_shortlist:" in output
