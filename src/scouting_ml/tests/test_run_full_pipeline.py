from __future__ import annotations

import json
from pathlib import Path

import pytest

from scouting_ml.scripts import run_full_pipeline as rfp
from scouting_ml.tests.summary_contract import assert_common_summary_contract


def _base_kwargs(tmp_path: Path) -> dict[str, object]:
    root = tmp_path
    return {
        "players_source": str(root / "players"),
        "data_dir": str(root / "data"),
        "external_dir": str(root / "external"),
        "dataset_output": str(root / "model" / "dataset.parquet"),
        "clean_output": str(root / "model" / "clean.parquet"),
        "predictions_output": str(root / "model" / "predictions.csv"),
        "val_season": "2023/24",
        "test_season": "2024/25",
        "start_season": "2021/22",
        "end_season": "2024/25",
        "min_minutes": 450.0,
        "trials": 5,
        "recency_half_life": 2.0,
        "under_5m_weight": 1.0,
        "mid_5m_to_20m_weight": 1.0,
        "over_20m_weight": 1.0,
        "exclude_prefixes": [],
        "exclude_columns": [],
        "optimize_metric": "lowmid_wmape",
        "interval_q": 0.8,
        "two_stage_band_model": True,
        "band_min_samples": 160,
        "band_blend_alpha": 0.35,
        "strict_leakage_guard": True,
        "strict_quality_gate": False,
        "league_holdouts": [],
        "drop_incomplete_league_seasons": True,
        "min_league_season_rows": 40,
        "min_league_season_completeness": 0.55,
        "residual_calibration_min_samples": 30,
        "mape_min_denom_eur": 1_000_000.0,
        "max_players": None,
        "sleep_seconds": 0.0,
        "transfer_dynamic_fallback": False,
        "transfer_dynamic_fallback_attempts": 0,
        "contracts_all_seasons": False,
        "national_all_seasons": False,
        "fetch_missing_profiles": False,
        "fetch_national_page": False,
        "with_ablation": False,
        "with_backtest": False,
        "ablation_configs": [],
        "ablation_out_dir": str(root / "model" / "ablation"),
        "backtest_out_dir": str(root / "model" / "backtests"),
        "backtest_min_train_seasons": 2,
        "backtest_test_seasons": [],
        "backtest_enforce_quality_gate": False,
        "backtest_min_test_r2": 0.60,
        "backtest_max_test_mape": None,
        "backtest_max_test_wmape": 0.42,
        "backtest_max_under5m_wmape": 0.50,
        "backtest_max_lowmid_weighted_wmape": 0.48,
        "backtest_max_segment_weighted_wmape": 0.45,
        "backtest_min_test_samples": 300,
        "backtest_min_test_under5m_samples": 50,
        "backtest_min_test_over20m_samples": 25,
        "backtest_skip_incomplete_test_seasons": True,
        "backtest_drop_incomplete_league_seasons": True,
        "backtest_min_league_season_rows": 40,
        "backtest_min_league_season_completeness": 0.55,
        "backtest_residual_calibration_min_samples": 30,
        "backtest_mape_min_denom_eur": 1_000_000.0,
        "with_future_targets": True,
        "future_targets_output": str(root / "model" / "future_targets.parquet"),
        "future_target_min_next_minutes": 450.0,
        "future_target_drop_na": False,
        "skip_injuries": True,
        "skip_contracts": True,
        "skip_transfers": True,
        "skip_national": True,
        "skip_context": True,
        "provider_config_json": None,
        "provider_audit_json": None,
        "provider_audit_csv": None,
        "skip_dataset_build": False,
        "skip_clean": False,
        "skip_train": False,
        "lock_artifacts": True,
        "lock_manifest_out": str(root / "model" / "model_manifest.json"),
        "lock_env_out": str(root / "model" / "model_artifacts.env"),
        "lock_label": "test_bundle",
        "lock_strict_artifacts": True,
        "save_shap_artifacts": False,
        "holdout_trials": 1,
        "optuna_study_namespace": "test",
        "optuna_load_if_exists": False,
        "summary_json": str(root / "model" / "full_pipeline_summary.json"),
    }


def test_run_full_pipeline_writes_summary_and_validates_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _write_file(path_like: str, payload: str = "ok") -> None:
        path = Path(path_like)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    monkeypatch.setattr(rfp, "build_dataset_main", lambda **kwargs: _write_file(kwargs["output"], "dataset"))
    monkeypatch.setattr(
        rfp,
        "clean_dataset",
        lambda input_path, output_path, min_minutes: _write_file(output_path, f"clean from {input_path} @ {min_minutes}"),
    )
    monkeypatch.setattr(
        rfp,
        "build_future_value_targets",
        lambda input_path, output_path, min_next_minutes, drop_na_target: _write_file(
            output_path, f"future from {input_path} @ {min_next_minutes} / {drop_na_target}"
        ),
    )

    def _train_stub(**kwargs) -> None:
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("player_id,name\np1,Player One\n", encoding="utf-8")
        output.with_name(f"{output.stem}_val{output.suffix}").write_text("player_id,name\np1,Player One\n", encoding="utf-8")
        output.with_suffix(".metrics.json").write_text(
            json.dumps({"overall": {"test": {"r2": 0.7, "wmape": 0.4}}}),
            encoding="utf-8",
        )

    monkeypatch.setattr(rfp, "_train_market_value_main", _train_stub)
    monkeypatch.setattr(
        rfp,
        "build_lock_bundle",
        lambda test_predictions, val_predictions, metrics_path, manifest_out, env_out, strict_artifacts, label: (
            _write_file(str(manifest_out), json.dumps({"label": label, "strict": bool(strict_artifacts)})),
            _write_file(str(env_out), f"TEST={test_predictions}\nVAL={val_predictions}\nMETRICS={metrics_path}\n"),
        ),
    )

    summary = rfp.run_full_pipeline(**_base_kwargs(tmp_path))
    summary_path = Path(_base_kwargs(tmp_path)["summary_json"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert_common_summary_contract(summary)
    assert_common_summary_contract(payload)
    assert summary_path.exists()
    assert payload["artifacts"]["dataset"]["exists"] is True
    assert payload["artifacts"]["clean_dataset"]["exists"] is True
    assert payload["artifacts"]["future_targets"]["exists"] is True
    assert payload["artifacts"]["test_predictions"]["exists"] is True
    assert payload["artifacts"]["val_predictions"]["exists"] is True
    assert payload["artifacts"]["metrics"]["exists"] is True
    assert payload["artifacts"]["lock_manifest"]["exists"] is True
    assert payload["artifacts"]["lock_env"]["exists"] is True
    assert isinstance(summary["snapshots"]["metrics"], dict)
    assert summary["snapshots"]["metrics"]["overall"]["test"]["r2"] == 0.7


def test_run_full_pipeline_raises_when_expected_artifact_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _write_file(path_like: str, payload: str = "ok") -> None:
        path = Path(path_like)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    monkeypatch.setattr(rfp, "build_dataset_main", lambda **kwargs: _write_file(kwargs["output"], "dataset"))
    monkeypatch.setattr(rfp, "clean_dataset", lambda input_path, output_path, min_minutes: _write_file(output_path, "clean"))
    monkeypatch.setattr(rfp, "build_future_value_targets", lambda input_path, output_path, min_next_minutes, drop_na_target: _write_file(output_path, "future"))

    def _train_stub_missing_metrics(**kwargs) -> None:
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("player_id,name\np1,Player One\n", encoding="utf-8")
        output.with_name(f"{output.stem}_val{output.suffix}").write_text("player_id,name\np1,Player One\n", encoding="utf-8")

    monkeypatch.setattr(rfp, "_train_market_value_main", _train_stub_missing_metrics)
    monkeypatch.setattr(
        rfp,
        "build_lock_bundle",
        lambda test_predictions, val_predictions, metrics_path, manifest_out, env_out, strict_artifacts, label: None,
    )

    with pytest.raises(FileNotFoundError, match="metrics"):
        rfp.run_full_pipeline(**_base_kwargs(tmp_path))
