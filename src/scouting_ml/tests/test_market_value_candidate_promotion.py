from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts.run_market_value_candidate_promotion import main


def _metrics_payload(*, wmape: float, r2: float, under_5m_wmape: float, mid_wmape: float) -> dict:
    return {
        "dataset": "tmp_dataset",
        "val_season": "2023/24",
        "test_season": "2024/25",
        "trials_per_position": 20,
        "overall": {
            "test": {"r2": r2, "wmape": wmape, "mae_eur": 3_000_000},
            "val": {"r2": r2 - 0.02, "wmape": wmape + 0.01, "mae_eur": 3_200_000},
        },
        "segments": {
            "test": [
                {
                    "segment": "under_5m",
                    "n_samples": 100,
                    "wmape": under_5m_wmape,
                    "r2": -0.2,
                    "mae_eur": 900_000,
                    "mape": under_5m_wmape,
                },
                {
                    "segment": "5m_to_20m",
                    "n_samples": 80,
                    "wmape": mid_wmape,
                    "r2": -0.4,
                    "mae_eur": 3_900_000,
                    "mape": mid_wmape,
                },
                {
                    "segment": "over_20m",
                    "n_samples": 20,
                    "wmape": 0.37,
                    "r2": 0.05,
                    "mae_eur": 13_000_000,
                    "mape": 0.37,
                },
            ]
        },
    }


def _holdout_payload(*, league: str, wmape: float, r2: float, mae_eur: float) -> dict:
    return {
        "league": league,
        "status": "ok",
        "n_samples": 50,
        "overall": {"r2": r2, "wmape": wmape, "mape": wmape + 0.05, "mae_eur": mae_eur},
    }


def test_run_market_value_candidate_promotion_promotes_on_pass(tmp_path: Path) -> None:
    champion_metrics = tmp_path / "champion.metrics.json"
    candidate_metrics = tmp_path / "candidate.metrics.json"
    champion_metrics.write_text(
        json.dumps(
            _metrics_payload(
                wmape=0.4147,
                r2=0.7360,
                under_5m_wmape=0.6230,
                mid_wmape=0.4257,
            )
        ),
        encoding="utf-8",
    )
    candidate_metrics.write_text(
        json.dumps(
            _metrics_payload(
                wmape=0.4106,
                r2=0.7310,
                under_5m_wmape=0.5940,
                mid_wmape=0.4125,
            )
        ),
        encoding="utf-8",
    )

    candidate_predictions = tmp_path / "candidate.csv"
    candidate_val_predictions = tmp_path / "candidate_val.csv"
    rows = [
        {
            "league": "Primeira Liga",
            "market_value_eur": 8_000_000,
            "expected_value_eur": 9_000_000,
            "undervaluation_confidence": 0.8,
            "undervalued_flag": 1,
            "age": 22,
        },
        {
            "league": "Eredivisie",
            "market_value_eur": 2_000_000,
            "expected_value_eur": 2_500_000,
            "undervaluation_confidence": 0.7,
            "undervalued_flag": 1,
            "age": 20,
        },
    ]
    pd.DataFrame(rows).to_csv(candidate_predictions, index=False)
    pd.DataFrame(rows).to_csv(candidate_val_predictions, index=False)

    candidate_holdouts = tmp_path / "candidate_holdouts"
    reference_holdouts = tmp_path / "reference_holdouts"
    candidate_holdouts.mkdir()
    reference_holdouts.mkdir()
    (candidate_holdouts / "cand.holdout_primeira_liga.metrics.json").write_text(
        json.dumps(_holdout_payload(league="Primeira Liga", wmape=0.50, r2=0.64, mae_eur=2_300_000)),
        encoding="utf-8",
    )
    (candidate_holdouts / "cand.holdout_eredivisie.metrics.json").write_text(
        json.dumps(_holdout_payload(league="Eredivisie", wmape=0.71, r2=0.60, mae_eur=2_600_000)),
        encoding="utf-8",
    )
    (reference_holdouts / "ref.holdout_primeira_liga.metrics.json").write_text(
        json.dumps(_holdout_payload(league="Primeira Liga", wmape=0.55, r2=0.63, mae_eur=2_500_000)),
        encoding="utf-8",
    )
    (reference_holdouts / "ref.holdout_eredivisie.metrics.json").write_text(
        json.dumps(_holdout_payload(league="Eredivisie", wmape=0.75, r2=0.59, mae_eur=2_800_000)),
        encoding="utf-8",
    )

    onboarding_json = tmp_path / "onboarding.json"
    onboarding_json.write_text(json.dumps({"status_counts": {}, "items": []}), encoding="utf-8")
    ablation_bundle = tmp_path / "ablation_bundle_2024-25.json"
    ablation_bundle.write_text(
        json.dumps({"best_overall_test": {"config": "cheap_aggressive", "value": 0.4106}}),
        encoding="utf-8",
    )

    comparison_json = tmp_path / "comparison.json"
    comparison_md = tmp_path / "comparison.md"
    benchmark_json = tmp_path / "benchmark.json"
    benchmark_md = tmp_path / "benchmark.md"
    manifest = tmp_path / "model_manifest.json"
    env_out = tmp_path / "model.env"

    rc = main(
        [
            "--champion-metrics",
            str(champion_metrics),
            "--candidate-predictions",
            str(candidate_predictions),
            "--candidate-val-predictions",
            str(candidate_val_predictions),
            "--candidate-metrics",
            str(candidate_metrics),
            "--candidate-holdout-glob",
            str(candidate_holdouts / "*.json"),
            "--reference-holdout-glob",
            str(reference_holdouts / "*.json"),
            "--comparison-out-json",
            str(comparison_json),
            "--comparison-out-md",
            str(comparison_md),
            "--benchmark-out-json",
            str(benchmark_json),
            "--benchmark-out-md",
            str(benchmark_md),
            "--manifest-out",
            str(manifest),
            "--env-out",
            str(env_out),
            "--onboarding-json",
            str(onboarding_json),
            "--ablation-bundle",
            str(ablation_bundle),
            "--candidate-label",
            "cheap_aggressive",
            "--champion-label",
            "champion",
            "--promote-on-pass",
        ]
    )

    assert rc == 0
    assert comparison_json.exists()
    assert comparison_md.exists()
    assert benchmark_json.exists()
    assert benchmark_md.exists()
    assert manifest.exists()
    assert env_out.exists()

    comparison_payload = json.loads(comparison_json.read_text(encoding="utf-8"))
    assert comparison_payload["decision"]["passed"] is True
    assert comparison_payload["decision"]["gates"]["test_wmape"] is True

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["artifacts"]["metrics"]["path"] == str(candidate_metrics)

    benchmark_payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
    assert benchmark_payload["model"]["test_wmape"] == 0.4106


def test_run_market_value_candidate_promotion_requires_future_benchmark_when_enabled(tmp_path: Path) -> None:
    champion_metrics = tmp_path / "champion.metrics.json"
    candidate_metrics = tmp_path / "candidate.metrics.json"
    champion_metrics.write_text(
        json.dumps(_metrics_payload(wmape=0.4147, r2=0.7360, under_5m_wmape=0.6230, mid_wmape=0.4257)),
        encoding="utf-8",
    )
    candidate_metrics.write_text(
        json.dumps(_metrics_payload(wmape=0.4106, r2=0.7310, under_5m_wmape=0.5940, mid_wmape=0.4125)),
        encoding="utf-8",
    )

    candidate_predictions = tmp_path / "candidate.csv"
    candidate_val_predictions = tmp_path / "candidate_val.csv"
    pd.DataFrame(
        [
            {
                "league": "Primeira Liga",
                "market_value_eur": 8_000_000,
                "expected_value_eur": 9_000_000,
                "undervaluation_confidence": 0.8,
                "undervalued_flag": 1,
                "age": 22,
            }
        ]
    ).to_csv(candidate_predictions, index=False)
    pd.DataFrame(
        [
            {
                "league": "Primeira Liga",
                "market_value_eur": 8_000_000,
                "expected_value_eur": 9_000_000,
                "undervaluation_confidence": 0.8,
                "undervalued_flag": 1,
                "age": 22,
            }
        ]
    ).to_csv(candidate_val_predictions, index=False)

    candidate_holdouts = tmp_path / "candidate_holdouts"
    reference_holdouts = tmp_path / "reference_holdouts"
    candidate_holdouts.mkdir()
    reference_holdouts.mkdir()
    (candidate_holdouts / "cand.holdout_primeira_liga.metrics.json").write_text(
        json.dumps(_holdout_payload(league="Primeira Liga", wmape=0.50, r2=0.64, mae_eur=2_300_000)),
        encoding="utf-8",
    )
    (reference_holdouts / "ref.holdout_primeira_liga.metrics.json").write_text(
        json.dumps(_holdout_payload(league="Primeira Liga", wmape=0.55, r2=0.63, mae_eur=2_500_000)),
        encoding="utf-8",
    )

    champion_future = tmp_path / "champion_future.json"
    candidate_future = tmp_path / "candidate_future.json"
    champion_future.write_text(
        json.dumps(
            {
                "splits": {
                    "val": {
                        "join": {"labeled_rows": 200, "labeled_share": 0.50},
                        "precision_at_k": {
                            "positive_growth": [
                                {
                                    "cohort_type": "overall",
                                    "cohort": "ALL",
                                    "k": 25,
                                    "precision_at_k": 0.52,
                                    "positive_rate": 0.40,
                                    "lift_vs_base": 0.12,
                                }
                            ]
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    candidate_future.write_text(
        json.dumps(
            {
                "splits": {
                    "val": {
                        "join": {"labeled_rows": 220, "labeled_share": 0.60},
                        "precision_at_k": {
                            "positive_growth": [
                                {
                                    "cohort_type": "overall",
                                    "cohort": "ALL",
                                    "k": 25,
                                    "precision_at_k": 0.58,
                                    "positive_rate": 0.42,
                                    "lift_vs_base": 0.16,
                                }
                            ]
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    comparison_json = tmp_path / "comparison.json"
    comparison_md = tmp_path / "comparison.md"
    rc = main(
        [
            "--champion-metrics",
            str(champion_metrics),
            "--candidate-predictions",
            str(candidate_predictions),
            "--candidate-val-predictions",
            str(candidate_val_predictions),
            "--candidate-metrics",
            str(candidate_metrics),
            "--candidate-holdout-glob",
            str(candidate_holdouts / "*.json"),
            "--reference-holdout-glob",
            str(reference_holdouts / "*.json"),
            "--candidate-future-benchmark-json",
            str(candidate_future),
            "--champion-future-benchmark-json",
            str(champion_future),
            "--require-future-benchmark",
            "--require-future-precision-vs-champion",
            "--comparison-out-json",
            str(comparison_json),
            "--comparison-out-md",
            str(comparison_md),
        ]
    )

    assert rc == 0
    payload = json.loads(comparison_json.read_text(encoding="utf-8"))
    assert payload["decision"]["gates"]["future_label_coverage"] is True
    assert payload["decision"]["gates"]["future_precision_vs_base"] is True
    assert payload["decision"]["gates"]["future_precision_vs_champion"] is True
