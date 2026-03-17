from __future__ import annotations

import json
from pathlib import Path

from scouting_ml.scripts import run_production_pipeline as rprod
from scouting_ml.scripts import run_provider_promotion_pipeline as rprov
from scouting_ml.tests.summary_contract import assert_common_summary_contract


def _write(path: Path, payload: str = "ok") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _artifact_meta(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": int(path.stat().st_size),
    }


def _metrics_payload(*, r2: float, wmape: float) -> dict:
    return {
        "overall": {
            "val": {"n_samples": 10, "r2": r2, "mae_eur": 1_000_000.0, "mape": 0.2, "wmape": wmape},
            "test": {"n_samples": 10, "r2": r2, "mae_eur": 1_000_000.0, "mape": 0.2, "wmape": wmape},
        },
        "segments": {
            "val": [
                {"segment": "under_5m", "n_samples": 4, "wmape": wmape},
                {"segment": "5m_to_20m", "n_samples": 4, "wmape": wmape},
                {"segment": "over_20m", "n_samples": 2, "wmape": wmape},
            ],
            "test": [
                {"segment": "under_5m", "n_samples": 4, "wmape": wmape},
                {"segment": "5m_to_20m", "n_samples": 4, "wmape": wmape},
                {"segment": "over_20m", "n_samples": 2, "wmape": wmape},
            ],
        },
    }


def test_run_production_pipeline_emits_common_summary_contract(tmp_path: Path, monkeypatch) -> None:
    def _run_full_pipeline_stub(**kwargs):
        output = Path(kwargs["predictions_output"])
        val_output = output.with_name(f"{output.stem}_val{output.suffix}")
        metrics_output = output.with_suffix(".metrics.json")
        quality_output = output.with_suffix(".quality.json")
        manifest_out = Path(kwargs["lock_manifest_out"])
        env_out = Path(kwargs["lock_env_out"])
        summary_path = Path(kwargs["summary_json"])

        _write(Path(kwargs["dataset_output"]), "dataset")
        _write(Path(kwargs["clean_output"]), "clean")
        _write(output, "player_id\np1\n")
        _write(val_output, "player_id\np1\n")
        _write(metrics_output, json.dumps(_metrics_payload(r2=0.7, wmape=0.4)))
        _write(quality_output, json.dumps({"status": "ok"}))
        _write(manifest_out, json.dumps({"label": kwargs["lock_label"]}))
        _write(env_out, "TEST=1\n")

        summary = {
            "generated_at_utc": "2026-03-15T00:00:00+00:00",
            "status": "ok",
            "inputs": {},
            "flags": {},
            "artifacts": {
                "dataset": _artifact_meta(Path(kwargs["dataset_output"])),
                "clean_dataset": _artifact_meta(Path(kwargs["clean_output"])),
                "test_predictions": _artifact_meta(output),
                "val_predictions": _artifact_meta(val_output),
                "metrics": _artifact_meta(metrics_output),
            },
            "snapshots": {"metrics": _metrics_payload(r2=0.7, wmape=0.4)},
        }
        _write(summary_path, json.dumps(summary))
        return summary

    def _weekly_stub(**kwargs):
        summary_path = tmp_path / "weekly_ops_summary.json"
        summary = {
            "generated_at_utc": "2026-03-15T00:00:01+00:00",
            "status": "ok",
            "inputs": {},
            "flags": {},
            "artifacts": {
                "weekly_kpi_json": _artifact_meta(_write(tmp_path / "weekly_kpi.json", json.dumps({"items": []}))),
            },
            "snapshots": {"weekly_kpi": {"items": []}},
            "summary_json": str(summary_path),
        }
        _write(summary_path, json.dumps(summary))
        return summary

    monkeypatch.setattr(rprod, "run_full_pipeline", _run_full_pipeline_stub)
    monkeypatch.setattr(rprod, "run_weekly_scout_ops", _weekly_stub)

    summary = rprod.run_production_pipeline(
        players_source=str(tmp_path / "players"),
        data_dir=str(tmp_path / "data"),
        external_dir=str(tmp_path / "external"),
        dataset_output=str(tmp_path / "dataset.parquet"),
        clean_output=str(tmp_path / "clean.parquet"),
        predictions_output=str(tmp_path / "predictions.csv"),
        val_season="2023/24",
        test_season="2024/25",
        start_season="2021/22",
        end_season="2024/25",
        min_minutes=450.0,
        trials=2,
        optimize_metric="lowmid_wmape",
        band_min_samples=100,
        band_blend_alpha=0.3,
        mape_min_denom_eur=1_000_000.0,
        with_backtest=False,
        backtest_test_seasons=[],
        backtest_enforce_quality_gate=False,
        backtest_min_test_r2=0.5,
        backtest_max_test_wmape=0.5,
        backtest_max_under5m_wmape=0.6,
        backtest_max_lowmid_weighted_wmape=0.55,
        backtest_max_segment_weighted_wmape=0.5,
        backtest_min_test_samples=100,
        backtest_min_test_under5m_samples=10,
        backtest_min_test_over20m_samples=10,
        backtest_skip_incomplete_test_seasons=True,
        drop_incomplete_league_seasons=True,
        min_league_season_rows=20,
        min_league_season_completeness=0.5,
        residual_calibration_min_samples=10,
        provider_config_json=None,
        provider_audit_json=None,
        provider_audit_csv=None,
        skip_injuries=True,
        skip_contracts=True,
        skip_transfers=True,
        skip_national=True,
        skip_context=True,
        skip_dataset_build=False,
        skip_clean=False,
        lock_manifest_out=str(tmp_path / "model_manifest.json"),
        lock_env_out=str(tmp_path / "model_artifacts.env"),
        lock_label="test_prod",
        run_weekly_ops=True,
        weekly_split="test",
        weekly_reports_out_dir=str(tmp_path / "reports"),
        weekly_non_big5_only=True,
        weekly_max_age=23.0,
        weekly_min_minutes=900.0,
        weekly_watchlist_tag="u23_non_big5",
        production_summary_out=str(tmp_path / "production_summary.json"),
        full_pipeline_summary_out=str(tmp_path / "full_pipeline_summary.json"),
    )

    assert_common_summary_contract(summary)
    assert_common_summary_contract(summary["snapshots"]["full_pipeline"])
    assert_common_summary_contract(summary["snapshots"]["weekly_ops"])


def test_run_provider_promotion_pipeline_emits_common_summary_contract(tmp_path: Path, monkeypatch) -> None:
    provider_config = _write(tmp_path / "provider_config.json", json.dumps({"snapshot_date": "2026-03-15"}))
    baseline_metrics = _write(tmp_path / "baseline.metrics.json", json.dumps(_metrics_payload(r2=0.68, wmape=0.42)))
    baseline_backtest = _write(
        tmp_path / "baseline_backtest.json",
        json.dumps({"runs": 2, "mean_test_r2": 0.66, "mean_test_wmape": 0.41}),
    )

    monkeypatch.setattr(rprov, "_seed_external_dir", lambda src, dst: {"copied_files": [], "missing_files": []})

    def _prepare_effective_provider_config(**kwargs):
        output_path = Path(kwargs["output_path"])
        _write(output_path, json.dumps({"snapshot_date": "2026-03-15"}))
        return {"effective_config_path": str(output_path), "active_sections": [], "warnings": []}

    def _build_provider_external_data(**kwargs):
        return {"status": "ok"}

    def _bootstrap_provider_links(**kwargs):
        return {"review_rows": 0}

    def _build_provider_link_audit(**kwargs):
        _write(Path(kwargs["out_json"]), json.dumps({"matched_rows": 0}))
        _write(Path(kwargs["out_csv"]), "matched_rows\n0\n")
        return {"matched_rows": 0}

    def _run_full_pipeline_stub(**kwargs):
        output = Path(kwargs["predictions_output"])
        val_output = output.with_name(f"{output.stem}_val{output.suffix}")
        metrics_output = output.with_suffix(".metrics.json")
        quality_output = output.with_suffix(".quality.json")
        error_priors_output = output.with_name(f"{output.stem}.error_priors.csv")
        backtest_summary = Path(kwargs["backtest_out_dir"]) / "rolling_backtest_summary.json"
        summary_path = Path(kwargs["summary_json"])

        _write(Path(kwargs["dataset_output"]), "dataset")
        _write(Path(kwargs["clean_output"]), "clean")
        _write(output, "player_id\np1\n")
        _write(val_output, "player_id\np1\n")
        _write(metrics_output, json.dumps(_metrics_payload(r2=0.72, wmape=0.39)))
        _write(quality_output, json.dumps({"status": "ok"}))
        _write(error_priors_output, "player_id,error\np1,0.1\n")
        _write(backtest_summary, json.dumps({"runs": 2, "mean_test_r2": 0.7, "mean_test_wmape": 0.39}))

        summary = {
            "generated_at_utc": "2026-03-15T00:00:00+00:00",
            "status": "ok",
            "inputs": {},
            "flags": {},
            "artifacts": {
                "dataset": _artifact_meta(Path(kwargs["dataset_output"])),
                "clean_dataset": _artifact_meta(Path(kwargs["clean_output"])),
                "test_predictions": _artifact_meta(output),
                "val_predictions": _artifact_meta(val_output),
                "metrics": _artifact_meta(metrics_output),
                "quality": _artifact_meta(quality_output),
                "error_priors": _artifact_meta(error_priors_output),
            },
            "snapshots": {
                "metrics": _metrics_payload(r2=0.72, wmape=0.39),
                "backtest": {"runs": 2, "mean_test_r2": 0.7, "mean_test_wmape": 0.39},
            },
        }
        _write(summary_path, json.dumps(summary))
        return summary

    def _build_lock_bundle(**kwargs):
        _write(Path(kwargs["manifest_out"]), json.dumps({"label": kwargs["label"]}))
        _write(Path(kwargs["env_out"]), "TEST=1\n")

    monkeypatch.setattr(rprov, "_prepare_effective_provider_config", _prepare_effective_provider_config)
    monkeypatch.setattr(rprov, "build_provider_external_data", _build_provider_external_data)
    monkeypatch.setattr(rprov, "bootstrap_provider_links", _bootstrap_provider_links)
    monkeypatch.setattr(rprov, "build_provider_link_audit", _build_provider_link_audit)
    monkeypatch.setattr(rprov, "run_full_pipeline", _run_full_pipeline_stub)
    monkeypatch.setattr(rprov, "build_lock_bundle", _build_lock_bundle)
    monkeypatch.setattr(
        rprov,
        "_provider_coverage_from_clean_dataset",
        lambda **kwargs: {"by_season": {"2024/25": {"any_provider_coverage_share": 0.2}}},
    )

    summary = rprov.run_provider_promotion_pipeline(
        provider_config_json=str(provider_config),
        out_dir=str(tmp_path / "provider_out"),
        candidate_tag="candidate_a",
        players_source=str(tmp_path / "players"),
        data_dir=str(tmp_path / "data"),
        external_dir=str(tmp_path / "external_seed"),
        baseline_metrics_path=str(baseline_metrics),
        baseline_backtest_path=str(baseline_backtest),
        val_season="2023/24",
        test_season="2024/25",
        start_season="2021/22",
        end_season="2024/25",
        min_minutes=450.0,
        trials=2,
        optimize_metric="lowmid_wmape",
        band_min_samples=100,
        band_blend_alpha=0.3,
        with_backtest=True,
        backtest_test_seasons=["2024/25"],
        review_confidence_threshold=0.75,
        skip_injuries=True,
        skip_contracts=True,
        skip_transfers=True,
        skip_national=True,
        skip_context=True,
        min_test_provider_coverage=0.1,
        max_test_wmape_delta=0.0,
        min_test_r2_delta=0.0,
        max_test_lowmid_wmape_delta=0.0,
        max_backtest_test_wmape_delta=0.0,
        min_backtest_test_r2_delta=0.0,
        promote_on_pass=True,
        promotion_manifest_out=str(tmp_path / "promotion_manifest.json"),
        promotion_env_out=str(tmp_path / "promotion_env.env"),
        promotion_label="provider_candidate",
    )

    assert_common_summary_contract(summary)
    assert_common_summary_contract(summary["snapshots"]["full_pipeline"])
