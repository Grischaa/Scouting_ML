from __future__ import annotations

import json
from pathlib import Path

from scouting_ml.scripts import run_league_strength_interaction_compare as compare_module


def test_league_strength_compare_namespaces_studies_by_dataset_signature(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "clean.parquet"
    dataset.write_text("placeholder", encoding="utf-8")

    captured_namespaces: list[str] = []
    captured_load_if_exists: list[bool] = []

    def fake_train_market_value_main(**kwargs):
        captured_namespaces.append(str(kwargs["optuna_study_namespace"]))
        captured_load_if_exists.append(bool(kwargs["optuna_load_if_exists"]))
        metrics = {
            "overall": {
                "val": {"r2": 0.70, "wmape": 0.39},
                "test": {"r2": 0.74, "wmape": 0.40},
            },
            "league_holdout": [],
        }
        Path(kwargs["metrics_output_path"]).write_text(json.dumps(metrics), encoding="utf-8")
        Path(kwargs["quality_output_path"]).write_text(json.dumps({"ok": True}), encoding="utf-8")
        Path(kwargs["output_path"]).write_text("player_id\n1\n", encoding="utf-8")
        Path(kwargs["val_output_path"]).write_text("player_id\n1\n", encoding="utf-8")

    monkeypatch.setattr(compare_module, "_train_market_value_main", fake_train_market_value_main)

    payload = compare_module.run_league_strength_interaction_compare(
        dataset=str(dataset),
        val_season="2023/24",
        test_season="2024/25",
        out_dir=str(tmp_path / "compare"),
        trials=1,
        holdout_trials=1,
        league_holdouts=["Estonian Meistriliiga", "Czech Fortuna Liga"],
    )

    dataset_signature = compare_module._dataset_study_suffix(str(dataset))
    assert captured_load_if_exists == [True, True]
    assert captured_namespaces == [
        f"league_strength_compare_baseline_{dataset_signature}",
        f"league_strength_compare_league_strength_interactions_{dataset_signature}",
    ]
    assert payload["decision"]["status"] == "hold_runtime_shrinkage"
    assert Path(tmp_path / "compare" / "league_strength_interaction_compare.json").exists()
    assert Path(tmp_path / "compare" / "league_strength_interaction_compare.md").exists()
