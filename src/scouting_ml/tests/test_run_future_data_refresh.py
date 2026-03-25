from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts import run_future_data_refresh as rfdr
from scouting_ml.tests.summary_contract import assert_common_summary_contract


def _install_future_refresh_stubs(monkeypatch) -> None:
    def _write(path_like: str | Path, payload: str) -> str:
        path = Path(path_like)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return str(path)

    monkeypatch.setattr(
        rfdr,
        "build_dataset_main",
        lambda data_dir, output, external_dir: _write(output, f"dataset from {data_dir} via {external_dir}"),
    )
    monkeypatch.setattr(
        rfdr,
        "clean_dataset",
        lambda input_path, output_path, min_minutes: _write(output_path, f"clean from {input_path} @ {min_minutes}"),
    )

    def _build_future_target_coverage_audit(**kwargs):
        _write(kwargs["future_targets_output"], "future targets")
        payload = {
            "total_rows": 4,
            "labeled_rows": 2,
            "future_source_files": ["dutch_eredivisie_2025-26_with_sofa.csv"],
            "season_rows": [{"season": "2024/25", "labeled_rows": 2}],
        }
        _write(kwargs["out_json"], json.dumps(payload))
        _write(kwargs["out_csv"], "season,labeled_rows\n2024/25,2\n")
        return payload

    def _build_future_scout_score(**kwargs):
        pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "talent_position_family": "ST",
                    "future_potential_score": 81.0,
                    "future_potential_confidence": 74.0,
                    "future_scout_blend_score": 0.9,
                    "future_growth_probability": 0.8,
                    "has_next_season_target": 1,
                }
            ]
        ).to_csv(kwargs["out_val_path"], index=False)
        if kwargs.get("out_test_path"):
            pd.DataFrame(
                [
                    {
                        "player_id": "p1",
                        "talent_position_family": "ST",
                        "future_potential_score": 79.0,
                        "future_potential_confidence": 68.0,
                        "future_scout_blend_score": 0.85,
                        "future_growth_probability": 0.75,
                        "has_next_season_target": 1,
                    }
                ]
            ).to_csv(kwargs["out_test_path"], index=False)
        diagnostics = {
            "training_rows": 2,
            "training_positive_rate": 0.5,
            "val_metrics": {"precision_at_25": 0.8},
            "future_talent_summary": {
                "future_label_coverage": {"labeled_rows": 1, "total_rows": 1},
                "position_family_counts": {"ST": 1},
                "future_potential_confidence_distribution": {"high": 1, "medium": 0, "low": 0},
            },
        }
        _write(kwargs["diagnostics_out"], json.dumps(diagnostics))
        return diagnostics

    def _build_future_value_benchmark_payload(**kwargs):
        return {
            "score_col": kwargs["score_col"],
            "splits": {"val": {"coverage": {"labeled_rows": 2}}},
        }

    def _write_future_value_benchmark_report(payload, *, out_json, out_md):
        _write(out_json, json.dumps(payload))
        _write(out_md, "# benchmark\n")
        return {"json": out_json, "markdown": out_md}

    def _build_future_value_diagnostics_payload(payload, *, source_benchmark_json, k, top_n):
        return {
            "source_benchmark_json": source_benchmark_json,
            "config": {"k": k, "top_n": top_n},
            "splits": {"val": {"positive_growth": {"league": {"best": []}}}},
        }

    def _write_future_value_diagnostics_report(payload, *, out_json, out_md):
        _write(out_json, json.dumps(payload))
        _write(out_md, "# diagnostics\n")
        return {"json": out_json, "markdown": out_md}

    monkeypatch.setattr(rfdr, "build_future_target_coverage_audit", _build_future_target_coverage_audit)
    monkeypatch.setattr(rfdr, "build_future_scout_score", _build_future_scout_score)
    monkeypatch.setattr(rfdr, "build_future_value_benchmark_payload", _build_future_value_benchmark_payload)
    monkeypatch.setattr(rfdr, "write_future_value_benchmark_report", _write_future_value_benchmark_report)
    monkeypatch.setattr(rfdr, "build_future_value_diagnostics_payload", _build_future_value_diagnostics_payload)
    monkeypatch.setattr(rfdr, "write_future_value_diagnostics_report", _write_future_value_diagnostics_report)


def test_run_future_data_refresh_imports_and_rebuilds_future_artifacts(tmp_path: Path, monkeypatch) -> None:
    _install_future_refresh_stubs(monkeypatch)
    processed_root = tmp_path / "processed"
    processed_root.mkdir(parents=True)
    combined_dir = processed_root / "Clubs combined"
    country_root = processed_root / "by_country"
    season_root = processed_root / "by_season"
    staging_dir = processed_root / "_incoming_future"
    import_dir = tmp_path / "incoming"
    import_dir.mkdir(parents=True)
    external_dir = tmp_path / "external"
    external_dir.mkdir(parents=True)

    def _league_csv(path: Path, season: str, values: tuple[int, int]) -> None:
        pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "name": "Player One",
                    "dob": "2003-01-01",
                    "season": season,
                    "league": "Eredivisie",
                    "club": "Ajax",
                    "country": "Netherlands",
                    "model_position": "FW",
                    "position_group": "FW",
                    "age": 21,
                    "market_value_eur": values[0],
                    "sofa_minutesPlayed": 1400,
                    "minutes": 1400,
                },
                {
                    "player_id": "p2",
                    "name": "Player Two",
                    "dob": "2002-02-02",
                    "season": season,
                    "league": "Eredivisie",
                    "club": "PSV",
                    "country": "Netherlands",
                    "model_position": "MF",
                    "position_group": "MF",
                    "age": 22,
                    "market_value_eur": values[1],
                    "sofa_minutesPlayed": 1500,
                    "minutes": 1500,
                },
            ]
        ).to_csv(path, index=False)

    _league_csv(processed_root / "dutch_eredivisie_2023-24_with_sofa.csv", "2023/24", (2_000_000, 5_000_000))
    _league_csv(processed_root / "dutch_eredivisie_2024-25_with_sofa.csv", "2024/25", (4_000_000, 4_000_000))
    _league_csv(import_dir / "dutch_eredivisie_2025-26_with_sofa.csv", "2025/26", (6_000_000, 3_000_000))

    val_predictions = tmp_path / "cheap_aggressive_val.csv"
    test_predictions = tmp_path / "cheap_aggressive_test.csv"
    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "name": "Player One",
                "dob": "2003-01-01",
                "season": "2023/24",
                "league": "Eredivisie",
                "model_position": "FW",
                "market_value_eur": 2_000_000,
                "minutes": 1400,
                "age": 21,
                "undervaluation_score": 0.90,
                "undervaluation_confidence": 0.80,
                "value_gap_conservative_eur": 2_000_000,
            },
            {
                "player_id": "p2",
                "name": "Player Two",
                "dob": "2002-02-02",
                "season": "2023/24",
                "league": "Eredivisie",
                "model_position": "MF",
                "market_value_eur": 5_000_000,
                "minutes": 1500,
                "age": 22,
                "undervaluation_score": 0.10,
                "undervaluation_confidence": 0.70,
                "value_gap_conservative_eur": 250_000,
            },
        ]
    ).to_csv(val_predictions, index=False)
    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "name": "Player One",
                "dob": "2003-01-01",
                "season": "2024/25",
                "league": "Eredivisie",
                "model_position": "FW",
                "market_value_eur": 4_000_000,
                "minutes": 1450,
                "age": 22,
                "undervaluation_score": 0.85,
                "undervaluation_confidence": 0.78,
                "value_gap_conservative_eur": 1_000_000,
            },
            {
                "player_id": "p2",
                "name": "Player Two",
                "dob": "2002-02-02",
                "season": "2024/25",
                "league": "Eredivisie",
                "model_position": "MF",
                "market_value_eur": 4_000_000,
                "minutes": 1550,
                "age": 23,
                "undervaluation_score": 0.15,
                "undervaluation_confidence": 0.68,
                "value_gap_conservative_eur": 100_000,
            },
        ]
    ).to_csv(test_predictions, index=False)

    summary_path = tmp_path / "future_refresh_summary.json"
    scored_val_output = tmp_path / "future_scored_val.csv"
    scored_test_output = tmp_path / "future_scored_test.csv"
    diagnostics_output = tmp_path / "future_score_diagnostics.json"
    future_targets_output = tmp_path / "future_targets.parquet"
    future_audit_json = tmp_path / "future_audit.json"
    future_audit_csv = tmp_path / "future_audit.csv"
    future_benchmark_json = tmp_path / "future_benchmark.json"
    future_benchmark_md = tmp_path / "future_benchmark.md"
    future_diagnostics_json = tmp_path / "future_diagnostics.json"
    future_diagnostics_md = tmp_path / "future_diagnostics.md"
    future_u23_json = tmp_path / "future_u23.json"
    future_u23_md = tmp_path / "future_u23.md"
    dataset_output = tmp_path / "dataset.parquet"
    clean_output = tmp_path / "clean.parquet"

    summary = rfdr.run_future_data_refresh(
        import_dir=str(import_dir),
        staging_dir=str(staging_dir),
        season_filter="2025-26",
        source_dir=str(processed_root),
        combined_dir=str(combined_dir),
        country_root=str(country_root),
        season_root=str(season_root),
        organization_manifest=str(tmp_path / "organization_manifest.csv"),
        clean_targets=True,
        dataset_output=str(dataset_output),
        clean_output=str(clean_output),
        external_dir=str(external_dir),
        clean_min_minutes=450.0,
        future_targets_output=str(future_targets_output),
        future_audit_json=str(future_audit_json),
        future_audit_csv=str(future_audit_csv),
        future_source_glob=str(processed_root / "**" / "*2025-26*_with_sofa.csv"),
        min_next_minutes=450.0,
        base_val_predictions=str(val_predictions),
        base_test_predictions=str(test_predictions),
        scored_val_output=str(scored_val_output),
        scored_test_output=str(scored_test_output),
        diagnostics_output=str(diagnostics_output),
        label_mode="positive_growth",
        k_eval=2,
        scoring_min_minutes=900.0,
        scoring_max_age=None,
        scoring_positions=None,
        scoring_include_leagues=None,
        scoring_exclude_leagues=None,
        future_benchmark_json=str(future_benchmark_json),
        future_benchmark_md=str(future_benchmark_md),
        future_diagnostics_json=str(future_diagnostics_json),
        future_diagnostics_md=str(future_diagnostics_md),
        future_u23_nonbig5_json=str(future_u23_json),
        future_u23_nonbig5_md=str(future_u23_md),
        future_score_col="future_scout_blend_score",
        future_cohort_min_labeled=1,
        future_k_values=[2],
        promotion_enabled=False,
        promotion_args=[],
        summary_json=str(summary_path),
    )

    assert_common_summary_contract(summary)
    assert summary["import"]["copied_count"] == 1
    assert summary_path.exists()
    assert future_targets_output.exists()
    assert future_audit_json.exists()
    assert future_benchmark_json.exists()
    assert future_benchmark_md.exists()
    assert future_diagnostics_json.exists()
    assert future_diagnostics_md.exists()
    assert future_u23_json.exists()
    assert scored_val_output.exists()
    assert scored_test_output.exists()
    assert diagnostics_output.exists()
    assert (combined_dir / "dutch_eredivisie_2025-26_with_sofa.csv").exists()

    scored_val = pd.read_csv(scored_val_output)
    assert "future_scout_blend_score" in scored_val.columns
    assert "future_growth_probability" in scored_val.columns
    assert "future_potential_score" in scored_val.columns

    audit_payload = json.loads(future_audit_json.read_text(encoding="utf-8"))
    season_rows = {row["season"]: row for row in audit_payload["season_rows"]}
    assert season_rows["2024/25"]["labeled_rows"] == 2

    disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert_common_summary_contract(disk_summary)
    assert disk_summary["future_score"]["training_rows"] == 2
    assert disk_summary["promotion"]["enabled"] is False
    assert disk_summary["talent_summary"]["test"]["position_family_counts"]["ST"] == 1
    assert disk_summary["talent_summary"]["test"]["confidence_distribution"]["medium"] == 1


def test_run_future_data_refresh_skips_missing_import_dir(tmp_path: Path, monkeypatch) -> None:
    _install_future_refresh_stubs(monkeypatch)
    processed_root = tmp_path / "processed"
    processed_root.mkdir(parents=True)
    combined_dir = processed_root / "Clubs combined"
    country_root = processed_root / "by_country"
    season_root = processed_root / "by_season"
    external_dir = tmp_path / "external"
    external_dir.mkdir(parents=True)

    def _league_csv(path: Path, season: str, values: tuple[int, int]) -> None:
        pd.DataFrame(
            [
                {
                    "player_id": "p1",
                    "name": "Player One",
                    "dob": "2003-01-01",
                    "season": season,
                    "league": "Eredivisie",
                    "club": "Ajax",
                    "country": "Netherlands",
                    "model_position": "FW",
                    "position_group": "FW",
                    "age": 21,
                    "market_value_eur": values[0],
                    "sofa_minutesPlayed": 1400,
                    "minutes": 1400,
                },
                {
                    "player_id": "p2",
                    "name": "Player Two",
                    "dob": "2002-02-02",
                    "season": season,
                    "league": "Eredivisie",
                    "club": "PSV",
                    "country": "Netherlands",
                    "model_position": "MF",
                    "position_group": "MF",
                    "age": 22,
                    "market_value_eur": values[1],
                    "sofa_minutesPlayed": 1500,
                    "minutes": 1500,
                },
            ]
        ).to_csv(path, index=False)

    _league_csv(processed_root / "dutch_eredivisie_2023-24_with_sofa.csv", "2023/24", (2_000_000, 3_000_000))
    _league_csv(processed_root / "dutch_eredivisie_2024-25_with_sofa.csv", "2024/25", (4_000_000, 2_000_000))
    _league_csv(processed_root / "dutch_eredivisie_2025-26_with_sofa.csv", "2025/26", (6_000_000, 1_000_000))

    val_predictions = tmp_path / "val.csv"
    test_predictions = tmp_path / "test.csv"
    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "name": "Player One",
                "dob": "2003-01-01",
                "season": "2023/24",
                "league": "Eredivisie",
                "model_position": "FW",
                "market_value_eur": 2_000_000,
                "minutes": 1400,
                "age": 21,
                "undervaluation_score": 0.8,
                "undervaluation_confidence": 0.7,
                "value_gap_conservative_eur": 1_000_000,
            },
            {
                "player_id": "p2",
                "name": "Player Two",
                "dob": "2002-02-02",
                "season": "2023/24",
                "league": "Eredivisie",
                "model_position": "MF",
                "market_value_eur": 3_000_000,
                "minutes": 1500,
                "age": 22,
                "undervaluation_score": 0.2,
                "undervaluation_confidence": 0.6,
                "value_gap_conservative_eur": 100_000,
            },
        ]
    ).to_csv(val_predictions, index=False)
    pd.DataFrame(
        [
            {
                "player_id": "p1",
                "name": "Player One",
                "dob": "2003-01-01",
                "season": "2024/25",
                "league": "Eredivisie",
                "model_position": "FW",
                "market_value_eur": 4_000_000,
                "minutes": 1450,
                "age": 22,
                "undervaluation_score": 0.75,
                "undervaluation_confidence": 0.68,
                "value_gap_conservative_eur": 1_000_000,
            },
            {
                "player_id": "p2",
                "name": "Player Two",
                "dob": "2002-02-02",
                "season": "2024/25",
                "league": "Eredivisie",
                "model_position": "MF",
                "market_value_eur": 2_000_000,
                "minutes": 1550,
                "age": 23,
                "undervaluation_score": 0.25,
                "undervaluation_confidence": 0.58,
                "value_gap_conservative_eur": 50_000,
            },
        ]
    ).to_csv(test_predictions, index=False)

    summary = rfdr.run_future_data_refresh(
        import_dir=str(tmp_path / "missing_incoming"),
        staging_dir=str(processed_root / "_incoming_future"),
        season_filter="2025-26",
        source_dir=str(processed_root),
        combined_dir=str(combined_dir),
        country_root=str(country_root),
        season_root=str(season_root),
        organization_manifest=str(tmp_path / "organization_manifest.csv"),
        clean_targets=True,
        dataset_output=str(tmp_path / "dataset.parquet"),
        clean_output=str(tmp_path / "clean.parquet"),
        external_dir=str(external_dir),
        clean_min_minutes=450.0,
        future_targets_output=str(tmp_path / "future_targets.parquet"),
        future_audit_json=str(tmp_path / "future_audit.json"),
        future_audit_csv=str(tmp_path / "future_audit.csv"),
        future_source_glob=str(processed_root / "**" / "*2025-26*_with_sofa.csv"),
        min_next_minutes=450.0,
        base_val_predictions=str(val_predictions),
        base_test_predictions=str(test_predictions),
        scored_val_output=str(tmp_path / "future_scored_val.csv"),
        scored_test_output=str(tmp_path / "future_scored_test.csv"),
        diagnostics_output=str(tmp_path / "future_score_diagnostics.json"),
        label_mode="positive_growth",
        k_eval=1,
        scoring_min_minutes=900.0,
        scoring_max_age=None,
        scoring_positions=None,
        scoring_include_leagues=None,
        scoring_exclude_leagues=None,
        future_benchmark_json=str(tmp_path / "future_benchmark.json"),
        future_benchmark_md=str(tmp_path / "future_benchmark.md"),
        future_diagnostics_json=str(tmp_path / "future_diagnostics.json"),
        future_diagnostics_md=str(tmp_path / "future_diagnostics.md"),
        future_u23_nonbig5_json=None,
        future_u23_nonbig5_md=None,
        future_score_col="future_scout_blend_score",
        future_cohort_min_labeled=1,
        future_k_values=[1],
        promotion_enabled=False,
        promotion_args=[],
        summary_json=str(tmp_path / "future_refresh_summary.json"),
    )

    assert summary["import"]["skipped"] is True
    assert summary["import"]["skip_reason"] == "import_dir_missing"
