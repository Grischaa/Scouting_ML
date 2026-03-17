from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts import run_injury_feature_block_compare as compare_module


def test_run_injury_feature_block_compare_writes_comparison(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "clean.parquet"
    pd.DataFrame(
        [
            {
                "season": "2023/24",
                "league": "Primeira Liga",
                "log_market_value": 1.0,
                "injury_days": 10,
                "injury_avg_days_per_case": 5.0,
                "availability_risk_score": 0.2,
                "durability_score": 0.8,
            },
            {
                "season": "2024/25",
                "league": "Eredivisie",
                "log_market_value": 1.0,
                "injury_days": 15,
                "injury_avg_days_per_case": 6.0,
                "availability_risk_score": 0.3,
                "durability_score": 0.7,
            },
        ]
    ).to_parquet(dataset, index=False)

    def fake_train_market_value_main(**kwargs):
        metrics = {
            "overall": {
                "val": {"r2": 0.70, "wmape": 0.39},
                "test": {
                    "r2": 0.74 if not kwargs["exclude_prefixes"] else 0.72,
                    "wmape": 0.40 if not kwargs["exclude_prefixes"] else 0.42,
                },
            },
            "segments": {
                "test": [
                    {"segment": "under_5m", "wmape": 0.56 if not kwargs["exclude_prefixes"] else 0.59},
                    {"segment": "5m_to_20m", "wmape": 0.41 if not kwargs["exclude_prefixes"] else 0.43},
                ]
            },
        }
        Path(kwargs["metrics_output_path"]).write_text(json.dumps(metrics), encoding="utf-8")
        Path(kwargs["quality_output_path"]).write_text(json.dumps({"ok": True}), encoding="utf-8")
        Path(kwargs["output_path"]).write_text("player_id\n1\n", encoding="utf-8")
        Path(kwargs["val_output_path"]).write_text("player_id\n1\n", encoding="utf-8")
        if kwargs["exclude_prefixes"]:
            assert kwargs["exclude_prefixes"] == ["injury_"]
            assert sorted(kwargs["exclude_columns"]) == sorted(compare_module.INJURY_DERIVED_COLUMNS)

    monkeypatch.setattr(compare_module, "_train_market_value_main", fake_train_market_value_main)

    payload = compare_module.run_injury_feature_block_compare(
        dataset=str(dataset),
        val_season="2023/24",
        test_season="2024/25",
        out_dir=str(tmp_path / "compare"),
        trials=1,
        holdout_trials=1,
    )

    assert payload["decision"]["status"] == "completed"
    assert payload["decision"]["winner_by_test_wmape"] == "full"
    assert Path(tmp_path / "compare" / "injury_feature_block_compare.json").exists()
    assert Path(tmp_path / "compare" / "injury_feature_block_compare.md").exists()
