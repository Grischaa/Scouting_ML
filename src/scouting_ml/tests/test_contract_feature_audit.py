from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scouting_ml.scripts.run_contract_feature_audit import run_contract_feature_audit


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


def test_run_contract_feature_audit_writes_summary_and_bundle(tmp_path: Path, monkeypatch) -> None:
    def fake_train_market_value_main(**kwargs) -> None:
        output_path = Path(kwargs["output_path"])
        val_output_path = Path(kwargs["val_output_path"])
        metrics_output_path = Path(kwargs["metrics_output_path"])
        excluded = tuple(kwargs.get("exclude_columns") or [])
        dropped = excluded[0] if excluded else ""

        if dropped == "contract_security_score":
            segment_errors = {"under_5m": 0.04, "5m_to_20m": 0.05, "over_20m": 0.03}
        elif dropped == "contract_years_left":
            segment_errors = {"under_5m": 0.15, "5m_to_20m": 0.12, "over_20m": 0.06}
        else:
            segment_errors = {"under_5m": 0.08, "5m_to_20m": 0.07, "over_20m": 0.04}

        pd.DataFrame(_prediction_rows(segment_errors, "test")).to_csv(output_path, index=False)
        pd.DataFrame(_prediction_rows(segment_errors, "val")).to_csv(val_output_path, index=False)
        metrics_output_path.write_text(
            json.dumps({"status": "ok", "exclude_columns": list(excluded)}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scouting_ml.scripts.run_contract_feature_audit._train_market_value_main",
        fake_train_market_value_main,
    )
    monkeypatch.setattr(
        "scouting_ml.scripts.run_contract_feature_audit._resolve_contract_columns",
        lambda **kwargs: ["contract_years_left", "contract_security_score"],
    )

    out_dir = tmp_path / "contract_audit"
    bundle = run_contract_feature_audit(
        dataset_path="data/model/example_clean.parquet",
        val_season="2023/24",
        test_season="2024/25",
        out_dir=str(out_dir),
        audit_set="compact",
        trials=1,
        slice_min_samples=1,
        league_min_samples=1,
        report_top_n=4,
    )

    summary_path = out_dir / "contract_feature_audit_summary_2024-25.csv"
    slices_path = out_dir / "contract_feature_audit_slices_2024-25.csv"
    bundle_path = out_dir / "contract_feature_audit_bundle_2024-25.json"
    report_path = out_dir / "contract_feature_audit_report_2024-25.md"

    assert summary_path.exists()
    assert slices_path.exists()
    assert bundle_path.exists()
    assert report_path.exists()
    assert bundle["best_overall_drop_test"]["column"] == "contract_security_score"

    summary = pd.read_csv(summary_path)
    full_row = summary.loc[summary["config"] == "full"].iloc[0]
    helpful_row = summary.loc[summary["column"] == "contract_years_left"].iloc[0]
    suspect_row = summary.loc[summary["column"] == "contract_security_score"].iloc[0]
    assert suspect_row["test_wmape"] < full_row["test_wmape"]
    assert helpful_row["test_wmape"] > full_row["test_wmape"]

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["most_suspect_contract_features_test"][0]["column"] == "contract_security_score"
    assert payload["most_helpful_contract_features_test"][0]["column"] == "contract_years_left"

    report_text = report_path.read_text(encoding="utf-8")
    assert "Contract Feature Audit" in report_text
    assert "Most Suspect Contract Features" in report_text
    assert "Most Helpful Contract Features" in report_text
