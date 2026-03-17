from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts.run_market_value_ablation import run_ablation


def _prediction_rows(multiplier: float, split: str) -> list[dict]:
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
                signed_error = multiplier * (1 if idx % 2 == 0 else -1)
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


def test_run_market_value_ablation_writes_slice_and_report_artifacts(tmp_path: Path, monkeypatch) -> None:
    def fake_train_market_value_main(**kwargs) -> None:
        output_path = Path(kwargs["output_path"])
        val_output_path = Path(kwargs["val_output_path"])
        metrics_output_path = Path(kwargs["metrics_output_path"])
        exclude_prefixes = tuple(kwargs.get("exclude_prefixes") or [])
        multiplier = 0.05 if "sb_" not in exclude_prefixes else 0.15

        pd.DataFrame(_prediction_rows(multiplier, "test")).to_csv(output_path, index=False)
        pd.DataFrame(_prediction_rows(multiplier, "val")).to_csv(val_output_path, index=False)
        metrics_output_path.write_text(
            json.dumps({"status": "ok", "exclude_prefixes": list(exclude_prefixes)}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scouting_ml.scripts.run_market_value_ablation._train_market_value_main",
        fake_train_market_value_main,
    )

    out_dir = tmp_path / "ablation"
    bundle = run_ablation(
        dataset_path="data/model/example_clean.parquet",
        val_season="2023/24",
        test_season="2024/25",
        out_dir=str(out_dir),
        config_names=["full", "no_provider"],
        trials=1,
        slice_min_samples=1,
        league_min_samples=1,
        report_top_n=4,
    )

    summary_path = out_dir / "ablation_summary_2024-25.csv"
    slices_path = out_dir / "ablation_slices_2024-25.csv"
    bundle_path = out_dir / "ablation_bundle_2024-25.json"
    report_path = out_dir / "ablation_report_2024-25.md"

    assert summary_path.exists()
    assert slices_path.exists()
    assert bundle_path.exists()
    assert report_path.exists()
    assert bundle["best_overall_test"]["config"] == "full"
    assert bundle["best_under_20m_test"]["config"] == "full"

    summary = pd.read_csv(summary_path)
    assert {"test_under_20m_wmape", "test_under_5m_wmape", "delta_test_wmape_vs_full"}.issubset(summary.columns)
    full_row = summary.loc[summary["config"] == "full"].iloc[0]
    no_provider_row = summary.loc[summary["config"] == "no_provider"].iloc[0]
    assert full_row["test_wmape"] < no_provider_row["test_wmape"]
    assert full_row["test_under_20m_wmape"] < no_provider_row["test_under_20m_wmape"]

    slices = pd.read_csv(slices_path)
    assert {"position", "league", "value_segment", "delta_wmape_vs_full", "wmape_rank_within_slice"}.issubset(
        slices.columns
    )
    assert {"overall", "position", "league", "value_segment", "value_focus", "position_value_segment"}.issubset(
        set(slices["slice_type"].astype(str))
    )

    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle_payload["best_overall_test"]["config"] == "full"
    assert len(bundle_payload["position_winners_test"]) >= 1
    assert len(bundle_payload["value_segment_winners_test"]) >= 1

    report_text = report_path.read_text(encoding="utf-8")
    assert "Best overall test config" in report_text
    assert "Best By Position" in report_text
    assert "Weakest Full-Model Test Slices" in report_text
