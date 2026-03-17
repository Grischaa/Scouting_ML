from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd

from scouting_ml.api.main import app
from scouting_ml.reporting.market_value_benchmarks import (
    build_market_value_benchmark_payload,
    write_market_value_benchmark_report,
)


class _ASGITestClient:
    def __init__(self, app) -> None:
        self._app = app

    def get(self, url: str, **kwargs) -> httpx.Response:
        async def _send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.get(url, **kwargs)

        return asyncio.run(_send())


def test_build_market_value_benchmark_payload_and_write_outputs(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.csv"
    holdout_dir = tmp_path / "holdouts"
    holdout_dir.mkdir()
    onboarding_path = tmp_path / "onboarding.json"
    ablation_dir = tmp_path / "ablation"
    ablation_dir.mkdir()

    metrics_path.write_text(
        json.dumps(
            {
                "dataset": "tmp_dataset",
                "val_season": "2023/24",
                "test_season": "2024/25",
                "overall": {
                    "test": {"r2": 0.71, "wmape": 0.38, "mae_eur": 3_100_000},
                    "val": {"r2": 0.68, "wmape": 0.41, "mae_eur": 3_400_000},
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "league": "Eredivisie",
                "undervalued_flag": 1,
                "undervaluation_confidence": 0.9,
                "age": 21,
                "market_value_eur": 6000000,
                "fair_value_eur": 7200000,
            },
            {
                "league": "Eredivisie",
                "undervalued_flag": 0,
                "undervaluation_confidence": 0.6,
                "age": 24,
                "market_value_eur": 8000000,
                "fair_value_eur": 7600000,
            },
            {
                "league": "Premier League",
                "undervalued_flag": 1,
                "undervaluation_confidence": 0.7,
                "age": 22,
                "market_value_eur": 15000000,
                "fair_value_eur": 14000000,
            },
        ]
    ).to_csv(predictions_path, index=False)

    (holdout_dir / "tmp.holdout_eredivisie.metrics.json").write_text(
        json.dumps(
            {
                "league": "Eredivisie",
                "status": "ok",
                "n_samples": 42,
                "overall": {"r2": 0.55, "wmape": 0.29, "mape": 0.34, "mae_eur": 1_800_000},
                "domain_shift": {"mean_abs_shift_z": 0.73},
            }
        ),
        encoding="utf-8",
    )
    (holdout_dir / "tmp.holdout_primeira_liga.metrics.json").write_text(
        json.dumps(
            {
                "league": "Primeira Liga",
                "status": "ok",
                "n_samples": 39,
                "overall": {"r2": 0.49, "wmape": 0.33, "mape": 0.39, "mae_eur": 2_100_000},
                "domain_shift": {"mean_abs_shift_z": 0.84},
            }
        ),
        encoding="utf-8",
    )

    onboarding_path.write_text(
        json.dumps(
            {
                "status_counts": {"ready": 1, "watch": 1},
                "items": [
                    {"league_slug": "eredivisie", "status": "ready", "reasons": ""},
                    {"league_slug": "primeira_liga", "status": "watch", "reasons": "missing_holdout_metrics"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (ablation_dir / "ablation_bundle_2024-25.json").write_text(
        json.dumps(
            {
                "best_overall_test": {"config": "full", "metric": "test_wmape", "value": 0.38},
                "best_under_20m_test": {"config": "no_transfer", "metric": "test_under_20m_wmape", "value": 0.29},
                "weakest_full_slices_test": [{"slice_type": "league", "slice_label": "Primeira Liga", "wmape": 0.44}],
            }
        ),
        encoding="utf-8",
    )

    payload = build_market_value_benchmark_payload(
        metrics_path=str(metrics_path),
        predictions_path=str(predictions_path),
        holdout_metrics_glob=str(holdout_dir / "*.json"),
        onboarding_json_path=str(onboarding_path),
        ablation_glob=str(ablation_dir / "*.json"),
    )

    assert payload["model"]["test_season"] == "2024/25"
    assert payload["league_holdout"]["summary"]["ok_count"] == 2
    assert payload["prediction_league"]["summary"]["total"] == 2
    assert payload["ablation"]["best_overall_test"]["config"] == "full"
    assert payload["coverage"]["rows"][0]["league"] == "Eredivisie"

    out_paths = write_market_value_benchmark_report(
        payload,
        out_json=str(tmp_path / "benchmark.json"),
        out_md=str(tmp_path / "benchmark.md"),
    )
    assert Path(out_paths["json"]).exists()
    assert Path(out_paths["markdown"]).exists()


def test_market_value_benchmarks_endpoint_uses_report_file(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "benchmark.json"
    report_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-03-13T00:00:00+00:00",
                "model": {"test_season": "2024/25"},
                "league_holdout": {"summary": {"ok_count": 1, "total": 1}, "rows": []},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SCOUTING_BENCHMARK_REPORT_PATH", str(report_path))

    client = _ASGITestClient(app)
    resp = client.get("/market-value/benchmarks")
    assert resp.status_code == 200
    payload = resp.json()["payload"]
    assert payload["model"]["test_season"] == "2024/25"
    assert payload["_meta"]["source"] == "file"
