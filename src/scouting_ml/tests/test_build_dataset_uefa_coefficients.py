from __future__ import annotations

import pandas as pd

from scouting_ml.models import build_dataset as build_dataset_module
from scouting_ml.models.build_dataset import add_model_features


def test_add_model_features_adds_uefa_coefficients_without_merge_columns(monkeypatch) -> None:
    monkeypatch.setattr(
        build_dataset_module,
        "_load_uefa_coefficients",
        lambda path=build_dataset_module.UEFA_COEFF_PATH: pd.DataFrame(
            [
                {
                    "country": "Netherlands",
                    "season": "2024/25",
                    "uefa_points": 9.4,
                    "rank": 6,
                    "points_total": 56.1,
                }
            ]
        ),
    )

    base = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "name": "Player One",
                "league": "Eredivisie",
                "club": "Example FC",
                "season": "2024/25",
                "position_group": "FW",
                "age": 21,
                "market_value_eur": 7_000_000,
                "sofa_minutesPlayed": 1800,
                "clubctx_club_strength_proxy": 0.62,
                "leaguectx_league_strength_index": 0.51,
                "nt_total_caps": 8,
                "nt_senior_caps": 3,
                "nt_is_full_international": 1,
                "transfer_max_fee_career_eur": 12_000_000,
            }
        ]
    )

    out = add_model_features(base, external_dir=None)

    assert out.loc[0, "uefa_coeff_points"] == 9.4
    assert out.loc[0, "uefa_coeff_rank"] == 6
    assert out.loc[0, "uefa_coeff_5yr_total"] == 56.1
    assert out.loc[0, "league_strength_blend"] > 0.0
    assert out.loc[0, "club_league_strength_interaction"] >= 0.0
    assert out.loc[0, "international_league_strength_interaction"] >= 0.0
    assert out.loc[0, "elite_context_league_interaction"] >= 0.0
    assert "season_x" not in out.columns
    assert "season_y" not in out.columns
