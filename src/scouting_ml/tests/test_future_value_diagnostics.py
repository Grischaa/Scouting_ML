from __future__ import annotations

import json
from pathlib import Path

from scouting_ml.reporting.future_value_diagnostics import (
    build_future_value_diagnostics_payload,
    write_future_value_diagnostics_report,
)


def test_future_value_diagnostics_payload_and_report(tmp_path: Path) -> None:
    benchmark = {
        "target_source": {"source": "unit_test"},
        "splits": {
            "test": {
                "join": {"prediction_rows": 100, "labeled_rows": 40, "labeled_share": 0.4},
                "warnings": ["low_labeled_rows"],
                "growth_summary": {
                    "positive_growth_rate": 0.3,
                    "growth_gt25pct_rate": 0.2,
                },
                "precision_at_k": {
                    "positive_growth": [
                        {
                            "cohort_type": "league",
                            "cohort": "Eredivisie",
                            "k": 25,
                            "n_labeled": 30,
                            "positive_rate": 0.3,
                            "precision_at_k": 0.68,
                            "lift_vs_base": 0.38,
                        },
                        {
                            "cohort_type": "league",
                            "cohort": "Belgian Pro League",
                            "k": 25,
                            "n_labeled": 28,
                            "positive_rate": 0.35,
                            "precision_at_k": 0.42,
                            "lift_vs_base": 0.07,
                        },
                        {
                            "cohort_type": "position",
                            "cohort": "FW",
                            "k": 25,
                            "n_labeled": 18,
                            "positive_rate": 0.33,
                            "precision_at_k": 0.61,
                            "lift_vs_base": 0.28,
                        },
                        {
                            "cohort_type": "value_segment",
                            "cohort": "under_5m",
                            "k": 25,
                            "n_labeled": 25,
                            "positive_rate": 0.22,
                            "precision_at_k": 0.54,
                            "lift_vs_base": 0.32,
                        },
                    ],
                    "growth_gt25pct": [
                        {
                            "cohort_type": "league",
                            "cohort": "Eredivisie",
                            "k": 25,
                            "n_labeled": 30,
                            "positive_rate": 0.2,
                            "precision_at_k": 0.48,
                            "lift_vs_base": 0.28,
                        }
                    ],
                },
            }
        },
    }

    payload = build_future_value_diagnostics_payload(
        benchmark,
        source_benchmark_json="bench.json",
        k=25,
        top_n=3,
    )

    assert payload["config"]["k"] == 25
    assert payload["splits"]["test"]["positive_growth"]["league"]["best"][0]["cohort"] == "Eredivisie"
    assert payload["splits"]["test"]["positive_growth"]["league"]["worst"][0]["cohort"] == "Belgian Pro League"
    assert payload["splits"]["test"]["positive_growth"]["position"]["best"][0]["cohort"] == "FW"

    out_paths = write_future_value_diagnostics_report(
        payload,
        out_json=str(tmp_path / "future_diag.json"),
        out_md=str(tmp_path / "future_diag.md"),
    )
    assert Path(out_paths["json"]).exists()
    written = json.loads(Path(out_paths["json"]).read_text(encoding="utf-8"))
    assert written["source_benchmark_json"] == "bench.json"

    md_text = Path(out_paths["markdown"]).read_text(encoding="utf-8")
    assert "Future Value Diagnostics" in md_text
    assert "Best Leagues" in md_text
    assert "Weakest Value Segments" in md_text
