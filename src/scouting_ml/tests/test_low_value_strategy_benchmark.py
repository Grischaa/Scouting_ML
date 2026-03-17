from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts.run_low_value_strategy_benchmark import run_low_value_strategy_benchmark


def _prediction_rows(segment_errors: dict[str, float], split: str) -> list[dict]:
    rows: list[dict] = []
    positions = ["FW", "MF", "DF", "GK"]
    leagues = ["League A", "League B"]
    values = [
        ("under_5m", 2_000_000.0),
        ("5m_to_20m", 8_000_000.0),
        ("over_20m", 30_000_000.0),
    ]
    idx = 0
    for league in leagues:
        for pos in positions:
            for segment, market_value in values:
                idx += 1
                signed_error = segment_errors[segment] * (1 if idx % 2 == 0 else -1)
                expected_value = market_value * (1.0 + signed_error)
                rows.append(
                    {
                        "player_id": f"{split}_{league}_{pos}_{segment}_{idx}",
                        "league": league,
                        "model_position": pos,
                        "value_segment": segment,
                        "market_value_eur": market_value,
                        "expected_value_eur": expected_value,
                    }
                )
    return rows


def test_run_low_value_strategy_benchmark_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    def fake_train_market_value_main(**kwargs) -> None:
        output_path = Path(kwargs["output_path"])
        val_output_path = Path(kwargs["val_output_path"])
        metrics_output_path = Path(kwargs["metrics_output_path"])
        under = float(kwargs.get("under_5m_weight", 1.0))
        mid = float(kwargs.get("mid_5m_to_20m_weight", 1.0))
        over = float(kwargs.get("over_20m_weight", 1.0))
        objective = str(kwargs.get("optimize_metric", "hybrid_wmape"))

        if objective == "lowmid_wmape" and under >= 2.0:
            segment_errors = {"under_5m": 0.05, "5m_to_20m": 0.05, "over_20m": 0.08}
        elif objective == "lowmid_wmape":
            segment_errors = {"under_5m": 0.06, "5m_to_20m": 0.055, "over_20m": 0.07}
        else:
            segment_errors = {"under_5m": 0.10, "5m_to_20m": 0.07, "over_20m": 0.04}

        if mid <= 1.0 and over >= 0.9:
            segment_errors["5m_to_20m"] += 0.01

        pd.DataFrame(_prediction_rows(segment_errors, "test")).to_csv(output_path, index=False)
        pd.DataFrame(_prediction_rows(segment_errors, "val")).to_csv(val_output_path, index=False)
        metrics_output_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "under_5m_weight": under,
                    "mid_5m_to_20m_weight": mid,
                    "over_20m_weight": over,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scouting_ml.scripts.run_low_value_strategy_benchmark._train_market_value_main",
        fake_train_market_value_main,
    )

    out_dir = tmp_path / "low_value"
    bundle = run_low_value_strategy_benchmark(
        dataset_path="data/model/example_clean.parquet",
        val_season="2023/24",
        test_season="2024/25",
        out_dir=str(out_dir),
        strategy_names=["baseline_full", "baseline_lowmid_objective", "cheap_aggressive"],
        trials=1,
        slice_min_samples=1,
        league_min_samples=1,
        report_top_n=4,
    )

    summary_path = out_dir / "low_value_strategy_summary_2024-25.csv"
    slices_path = out_dir / "low_value_strategy_slices_2024-25.csv"
    bundle_path = out_dir / "low_value_strategy_bundle_2024-25.json"
    report_path = out_dir / "low_value_strategy_report_2024-25.md"

    assert summary_path.exists()
    assert slices_path.exists()
    assert bundle_path.exists()
    assert report_path.exists()
    assert bundle["best_under_20m_test"]["config"] == "cheap_aggressive"

    summary = pd.read_csv(summary_path)
    baseline = summary.loc[summary["config"] == "baseline_full"].iloc[0]
    cheap = summary.loc[summary["config"] == "cheap_aggressive"].iloc[0]
    assert cheap["test_under_20m_wmape"] < baseline["test_under_20m_wmape"]
    assert cheap["test_under_5m_wmape"] < baseline["test_under_5m_wmape"]

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["best_overall_test"]["config"] == "baseline_full"
    assert payload["best_under_20m_test"]["config"] == "cheap_aggressive"

    report_text = report_path.read_text(encoding="utf-8")
    assert "Low-Value Strategy Benchmark" in report_text
    assert "Cheap-Player Tradeoffs" in report_text
