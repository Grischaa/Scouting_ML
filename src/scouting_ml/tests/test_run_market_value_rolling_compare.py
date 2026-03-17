from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts import run_market_value_rolling_compare as compare_module


def _write_metrics(path: Path, *, trials: int, test_r2: float, test_wmape: float, exclude_prefixes=None, exclude_columns=None) -> None:
    payload = {
        "trials_per_position": trials,
        "under_5m_weight": 2.5,
        "mid_5m_to_20m_weight": 1.75,
        "over_20m_weight": 0.6,
        "optimize_metric": "lowmid_wmape",
        "interval_q": 0.8,
        "two_stage_band_model": True,
        "band_min_samples": 160,
        "band_blend_alpha": 0.35,
        "exclude_prefixes": list(exclude_prefixes or []),
        "exclude_columns": list(exclude_columns or []),
        "overall": {
            "test": {
                "r2": test_r2,
                "wmape": test_wmape,
            }
        },
        "segments": {
            "test": [
                {"segment": "under_5m", "wmape": 0.56},
            ]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_market_value_rolling_compare_runs_variants_and_writes_summary(monkeypatch, tmp_path: Path) -> None:
    active = tmp_path / "active.metrics.json"
    prod60 = tmp_path / "prod60.metrics.json"
    no_injury = tmp_path / "no_injury.metrics.json"
    _write_metrics(active, trials=1, test_r2=0.73, test_wmape=0.41)
    _write_metrics(prod60, trials=60, test_r2=0.74, test_wmape=0.40)
    _write_metrics(
        no_injury,
        trials=3,
        test_r2=0.72,
        test_wmape=0.41,
        exclude_prefixes=["injury_"],
        exclude_columns=["availability_risk_score"],
    )

    monkeypatch.setattr(
        compare_module,
        "DEFAULT_VARIANTS",
        (
            compare_module.VariantSpec("active_base", active, "active"),
            compare_module.VariantSpec("prod60", prod60, "prod60"),
            compare_module.VariantSpec("no_injury_block", no_injury, "trimmed"),
        ),
    )

    calls: list[dict] = []

    def fake_run_rolling_backtest(**kwargs):
        calls.append(kwargs)
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if kwargs["exclude_prefixes"]:
            mean_r2 = 0.69
            mean_wmape = 0.405
            lowmid = 0.47
        elif out_dir.name == "prod60":
            mean_r2 = 0.72
            mean_wmape = 0.398
            lowmid = 0.46
        else:
            mean_r2 = 0.70
            mean_wmape = 0.410
            lowmid = 0.48
        (out_dir / "rolling_backtest_summary.json").write_text(
            json.dumps(
                {
                    "runs": 2,
                    "mean_test_r2": mean_r2,
                    "std_test_r2": 0.01,
                    "mean_test_wmape": mean_wmape,
                    "std_test_wmape": 0.01,
                    "mean_test_lowmid_weighted_wmape": lowmid,
                    "quality_gate": {"passed": True},
                    "skipped_runs": [],
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"test_season": "2023/24", "test_r2": mean_r2, "test_wmape": mean_wmape},
                {"test_season": "2024/25", "test_r2": mean_r2 + 0.01, "test_wmape": mean_wmape - 0.01},
            ]
        ).to_csv(out_dir / "rolling_backtest_summary.csv", index=False)

    monkeypatch.setattr(compare_module, "_run_rolling_backtest", fake_run_rolling_backtest)

    payload = compare_module.run_market_value_rolling_compare(
        dataset="data/model/tm_context_candidate_clean.parquet",
        out_dir=str(tmp_path / "rolling_compare"),
        variants=["active_base", "prod60", "no_injury_block"],
        trials_cap=5,
        test_seasons=["2023/24", "2024/25"],
    )

    assert len(calls) == 3
    assert calls[0]["trials"] == 1
    assert calls[1]["trials"] == 5
    assert calls[2]["exclude_prefixes"] == ["injury_"]
    assert payload["decision"]["winner_by_rolling_mean_test_r2"] == "prod60"
    assert payload["decision"]["winner_by_rolling_mean_test_wmape"] == "prod60"
    assert (tmp_path / "rolling_compare" / "rolling_compare_summary.json").exists()
    assert (tmp_path / "rolling_compare" / "rolling_compare_summary.csv").exists()
    assert (tmp_path / "rolling_compare" / "rolling_compare_summary.md").exists()
