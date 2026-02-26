from __future__ import annotations

import pandas as pd
import pytest

from scouting_ml.models.train_market_value_full import (
    _validate_no_leakage_features,
    apply_confidence_scoring,
)


def test_leakage_guard_blocks_value_proxy_features() -> None:
    features = ["sofa_goals_per90", "clubctx_club_strength_proxy", "market_value_proxy"]
    with pytest.raises(ValueError, match="Leakage guard failed"):
        _validate_no_leakage_features(features, strict=True)


def test_confidence_scoring_adds_fair_value_and_score_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_position": "FW",
                "value_segment": "5m_to_20m",
                "market_value_eur": 10_000_000,
                "expected_value_eur": 12_000_000,
                "minutes": 1500,
                "age": 22,
            }
        ]
    )
    priors = pd.DataFrame(
        [
            {
                "model_position": "FW",
                "value_segment": "5m_to_20m",
                "prior_mae_eur": 1_000_000,
                "prior_medae_eur": 800_000,
                "prior_p75ae_eur": 1_200_000,
                "prior_qae_eur": 1_100_000,
                "prior_interval_q": 0.8,
                "n_samples": 100,
            }
        ]
    )

    out = apply_confidence_scoring(frame, priors, interval_q=0.8)
    assert "fair_value_eur" in out.columns
    assert "undervaluation_score" in out.columns
    assert "interval_contains_truth" in out.columns
    assert out.loc[0, "fair_value_eur"] == pytest.approx(out.loc[0, "expected_value_eur"])
    assert out.loc[0, "undervaluation_score"] >= 0.0
