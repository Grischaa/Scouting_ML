from __future__ import annotations

from pathlib import Path

import pandas as pd

from scouting_ml.scripts.compare_scout_workflow_variants import compare_scout_workflow_variants


def test_compare_scout_workflow_variants_writes_outputs(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    future = tmp_path / "future.csv"

    base_rows = [
        {
            "player_id": "p1",
            "name": "One",
            "league": "Eredivisie",
            "season": "2024/25",
            "market_value_eur": 2_000_000,
            "fair_value_eur": 4_500_000,
            "value_gap_conservative_eur": 2_500_000,
            "undervaluation_confidence": 0.9,
            "minutes": 1400,
            "age": 21,
        },
        {
            "player_id": "p2",
            "name": "Two",
            "league": "Eredivisie",
            "season": "2024/25",
            "market_value_eur": 2_000_000,
            "fair_value_eur": 4_400_000,
            "value_gap_conservative_eur": 2_400_000,
            "undervaluation_confidence": 0.85,
            "minutes": 1400,
            "age": 21,
        },
    ]
    future_rows = [
        {**base_rows[0], "future_scout_blend_score": 0.10},
        {**base_rows[1], "future_scout_blend_score": 0.95},
    ]
    pd.DataFrame(base_rows).to_csv(baseline, index=False)
    pd.DataFrame(future_rows).to_csv(future, index=False)

    payload = compare_scout_workflow_variants(
        baseline_predictions=str(baseline),
        future_predictions=str(future),
        split="test",
        out_dir=str(tmp_path / "compare"),
        top_n=2,
        min_minutes=900,
        max_age=23,
        min_confidence=0.5,
        min_value_gap_eur=1_000_000.0,
        non_big5_only=False,
        memo_count=0,
    )

    assert payload["comparison"]["overlap_count"] == 2
    assert payload["future_scored"]["diagnostics"]["score_column"] == "future_scout_blend_score"
    assert Path(tmp_path / "compare" / "scout_workflow_variant_compare_test.json").exists()
