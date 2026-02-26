from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts import run_rolling_backtest as rb


def _write_metrics(
    path: Path,
    *,
    test_n: int,
    under_5m_n: int,
    mid_n: int,
    over_20m_n: int,
    test_r2: float = 0.7,
) -> None:
    payload = {
        "overall": {
            "val": {
                "n_samples": 1000,
                "r2": 0.75,
                "mae_eur": 2_500_000,
                "mape": 0.35,
                "wmape": 0.30,
            },
            "test": {
                "n_samples": test_n,
                "r2": test_r2,
                "mae_eur": 3_000_000,
                "mape": 0.40,
                "wmape": 0.35,
            },
        },
        "segments": {
            "test": [
                {
                    "segment": "under_5m",
                    "n_samples": under_5m_n,
                    "r2": -0.2,
                    "mape": 0.7,
                    "wmape": 0.45,
                },
                {
                    "segment": "5m_to_20m",
                    "n_samples": mid_n,
                    "r2": 0.2,
                    "mape": 0.35,
                    "wmape": 0.33,
                },
                {
                    "segment": "over_20m",
                    "n_samples": over_20m_n,
                    "r2": 0.5,
                    "mape": 0.22,
                    "wmape": 0.25,
                },
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rolling_backtest_skips_incomplete_season(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "clean.parquet"
    out_dir = tmp_path / "backtests"

    rows: list[dict[str, object]] = []
    for season in ("2021/22", "2022/23", "2023/24"):
        rows.extend({"season": season, "market_value_eur": 3_000_000.0} for _ in range(500))

    rows.extend({"season": "2024/25", "market_value_eur": 3_000_000.0} for _ in range(707))
    rows.extend({"season": "2024/25", "market_value_eur": 10_000_000.0} for _ in range(640))
    rows.extend({"season": "2024/25", "market_value_eur": 40_000_000.0} for _ in range(459))

    rows.extend({"season": "2025/26", "market_value_eur": 3_000_000.0} for _ in range(117))
    rows.extend({"season": "2025/26", "market_value_eur": 10_000_000.0} for _ in range(13))

    pd.DataFrame(rows).to_parquet(dataset_path, index=False)

    def _fake_train_market_value_main(**kwargs) -> None:
        metrics_path = Path(kwargs["metrics_output_path"])
        test_season = str(kwargs["test_season"])
        if test_season == "2025/26":
            _write_metrics(
                metrics_path,
                test_n=130,
                under_5m_n=117,
                mid_n=13,
                over_20m_n=0,
                test_r2=0.1,
            )
        else:
            _write_metrics(
                metrics_path,
                test_n=1806,
                under_5m_n=707,
                mid_n=640,
                over_20m_n=459,
                test_r2=0.73,
            )

    monkeypatch.setattr(rb, "train_market_value_main", _fake_train_market_value_main)

    rb.run_rolling_backtest(
        dataset_path=str(dataset_path),
        out_dir=str(out_dir),
        trials=1,
        min_train_seasons=2,
        test_seasons=["2024/25", "2025/26"],
        min_test_samples=300,
        min_test_under5m_samples=50,
        min_test_over20m_samples=25,
        skip_incomplete_test_seasons=True,
    )

    summary = pd.read_csv(out_dir / "rolling_backtest_summary.csv")
    assert list(summary["test_season"]) == ["2024/25"]

    summary_json = json.loads((out_dir / "rolling_backtest_summary.json").read_text(encoding="utf-8"))
    skipped_runs = summary_json.get("skipped_runs", [])
    assert len(skipped_runs) == 1
    assert skipped_runs[0]["test_season"] == "2025/26"
